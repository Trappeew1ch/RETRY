import { redirect } from 'next/navigation';

export default function NewChatPage() {
  redirect('/chat');
  return null;
}