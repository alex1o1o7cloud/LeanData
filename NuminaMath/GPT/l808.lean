import Mathlib

namespace evaluate_sixth_iteration_of_g_at_2_l808_808584

def g (x : ℤ) : ℤ := x^2 - 4 * x + 1

theorem evaluate_sixth_iteration_of_g_at_2 :
  g (g (g (g (g (g 2))))) = 59162302643740737293922 := by
  sorry

end evaluate_sixth_iteration_of_g_at_2_l808_808584


namespace sum_of_non_palindrome_integers_taking_exactly_five_steps_l808_808445

-- Define what it means for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let str_n := n.repr in
  str_n = str_n.reverse

-- Define the reverse of a number by converting it to string and reversing it
def reverse_number (n : ℕ) : ℕ :=
  string.to_nat! (n.repr.reverse)

-- Define the process of making a number into a palindrome
def steps_to_palindrome (n : ℕ) : ℕ :=
  let rec aux (m k : ℕ) : ℕ :=
    if is_palindrome m then k else aux (m + reverse_number m) (k + 1)
  in aux n 0

-- Define the set of numbers between 50 and 150 that are not initially palindromes
def non_palindrome_numbers : list ℕ :=
  list.filter (λ n, ¬ is_palindrome n) (list.range' 50 101)

-- Define the set of numbers that take exactly five steps to become palindromes
def numbers_with_five_steps : list ℕ :=
  list.filter (λ n, steps_to_palindrome n = 5) non_palindrome_numbers

-- Define the main problem statement
theorem sum_of_non_palindrome_integers_taking_exactly_five_steps :
  list.sum numbers_with_five_steps = 154 := by
    sorry

end sum_of_non_palindrome_integers_taking_exactly_five_steps_l808_808445


namespace pencil_count_l808_808193

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l808_808193


namespace find_angle_l808_808083

variable (a b : ℝ → ℝ → ℝ) -- Let a and b be unit vectors in the space of real-valued functions
variable (unit_a : ∀ x y, |a x y| = 1)
variable (unit_b : ∀ x y, |b x y| = 1)
variable (condition : ∀ x y, |a x y - b x y| = sqrt(3) * |a x y + b x y|)

def angle_between_vectors : ℝ :=
  ∀ x y, (a x y) • (b x y) = cos (2 * pi / 3)

theorem find_angle : angle_between_vectors a b :=
  sorry

end find_angle_l808_808083


namespace product_PA_PB_eq_nine_l808_808761

theorem product_PA_PB_eq_nine 
  (P A B : ℝ × ℝ) 
  (hP : P = (3, 1)) 
  (h1 : A ≠ B)
  (h2 : ∃ L : ℝ × ℝ → Prop, L P ∧ L A ∧ L B) 
  (h3 : A.fst ^ 2 + A.snd ^ 2 = 1) 
  (h4 : B.fst ^ 2 + B.snd ^ 2 = 1) : 
  |((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)| * |((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2)| = 9 := 
sorry

end product_PA_PB_eq_nine_l808_808761


namespace arithmetic_sequence_and_sum_l808_808243

noncomputable def a_n (n : ℕ) : ℤ := 2 * n + 10

def S_n (n : ℕ) : ℤ := n * (12 + 2 * n + 10) / 2

theorem arithmetic_sequence_and_sum :
    (a_n 10 = 30) ∧ 
    (a_n 20 = 50) ∧ 
    (∀ n, S_n n = 11 * n + n^2) ∧ 
    (S_n 3 = 42) :=
by {
    -- a_n 10 = 2 * 10 + 10 = 30
    -- a_n 20 = 2 * 20 + 10 = 50
    -- S_n n = n * (2n + 22) / 2 = 11n + n^2
    -- S_n 3 = 3 * 14 = 42
    sorry
}

end arithmetic_sequence_and_sum_l808_808243


namespace solve_equations_l808_808033

theorem solve_equations (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : (-x)^3 = (-8)^2) : x = 3 ∨ x = -3 ∨ x = -4 :=
by 
  sorry

end solve_equations_l808_808033


namespace area_increase_is_correct_l808_808730

noncomputable def original_rect_length : ℝ := 60
noncomputable def original_rect_width : ℝ := 20
noncomputable def original_area : ℝ := original_rect_length * original_rect_width
noncomputable def perimeter : ℝ := 2 * (original_rect_length + original_rect_width)
noncomputable def radius : ℝ := perimeter / (2 * Real.pi)
noncomputable def new_circle_area : ℝ := Real.pi * radius^2
noncomputable def area_increase : ℝ := new_circle_area - original_area

theorem area_increase_is_correct : area_increase ≈ 837.94 := by
  sorry

end area_increase_is_correct_l808_808730


namespace max_a_for_three_solutions_l808_808030

-- Define the equation as a Lean function
def equation (x a : ℝ) : ℝ :=
  (|x-2| + 2 * a)^2 - 3 * (|x-2| + 2 * a) + 4 * a * (3 - 4 * a)

-- Statement of the proof problem
theorem max_a_for_three_solutions :
  (∃ (a : ℝ), (∀ x : ℝ, equation x a = 0) ∧
  (∀ (b : ℝ), (∀ x : ℝ, equation x b = 0) → b ≤ 0.5)) :=
sorry

end max_a_for_three_solutions_l808_808030


namespace tumblonian_words_count_l808_808183

def numTumblonianWords : ℕ :=
  let alphabet_size := 6
  let max_word_length := 4
  let num_words n := alphabet_size ^ n
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4)

theorem tumblonian_words_count : numTumblonianWords = 1554 := by
  sorry

end tumblonian_words_count_l808_808183


namespace total_dogs_in_academy_l808_808798

def fetch_dogs : ℕ := 70
def fetch_jump_dogs : ℕ := 25
def jump_dogs : ℕ := 40
def jump_bark_dogs : ℕ := 15
def bark_dogs : ℕ := 45
def fetch_bark_dogs : ℕ := 20
def all_three_dogs : ℕ := 12
def no_trick_dogs : ℕ := 15

theorem total_dogs_in_academy : 
  fetch_dogs 
  + jump_dogs 
  + bark_dogs 
  - fetch_jump_dogs 
  - jump_bark_dogs 
  - fetch_bark_dogs 
  + all_three_dogs 
  + no_trick_dogs 
  = 122 := 
by 
  calc 
    (fetch_dogs + jump_dogs + bark_dogs 
    - fetch_jump_dogs - jump_bark_dogs - fetch_bark_dogs 
    + all_three_dogs + no_trick_dogs) 
    = (70 + 40 + 45 
    - 25 - 15 - 20 
    + 12 + 15) 
    : by rfl 
  ... = 122 : by norm_num
    
#eval total_dogs_in_academy

end total_dogs_in_academy_l808_808798


namespace chord_length_between_circles_l808_808084

theorem chord_length_between_circles :
  ∃ l : ℝ,
  let C1 := λ x y : ℝ, x^2 + y^2 - 2 * x + 4 * y - 4 = 0,
      C2 := λ x y : ℝ, x^2 + y^2 + 2 * x + 2 * y - 2 = 0,
      C3 := λ x y : ℝ, x^2 + y^2 - 2 * x - 2 * y - 14 / 5 = 0,
      common_chord := λ x y : ℝ, 2 * x - y + 1 = 0,
      center_C3 := (1 : ℝ, 1 : ℝ),
      radius_C3 := 2 * Real.sqrt(30) / 5,
      distance := |(2 : ℝ) * center_C3.1 - center_C3.2 + 1| / Real.sqrt((2 : ℝ)^2 + (-1 : ℝ)^2)
  in
  common_chord center_C3.1 center_C3.2 = 0 →
  radius_C3^2 - distance^2 = 4 →
  l = 2 * Real.sqrt(radius_C3^2 - distance^2) →
  l = 4 := by sorry

end chord_length_between_circles_l808_808084


namespace problem_statement_l808_808335

open Classical

variable (p q : Prop)

theorem problem_statement (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬ p) : (p = (5 + 2 = 6) ∧ q = (6 > 2)) :=
by
  have hp : p = False := by sorry
  have hq : q = True := by sorry
  exact ⟨by sorry, by sorry⟩

end problem_statement_l808_808335


namespace problem1_problem2_l808_808812

-- Problem 1 Lean 4 Statement
theorem problem1 : (1 : ℝ) - (-1 : ℝ) ^ 2018 - | real.sqrt 3 - 2 | + real.sqrt 81 + real.cbrt (-27) = 3 + real.sqrt 3 :=
by sorry

-- Problem 2 Lean 4 Statement
theorem problem2 : real.sqrt 2 * (real.sqrt 2 + 2) - 3 * real.sqrt 2 + real.sqrt 3 * (1 / real.sqrt 3) = 3 - real.sqrt 2 :=
by sorry

end problem1_problem2_l808_808812


namespace choose_non_overlapping_sets_for_any_n_l808_808894

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808894


namespace part1_part2_part3_l808_808062

noncomputable def sequence_a : ℕ → ℝ
| 0 := 1
| 1 := 3
| n + 2 := 3 * sequence_a (n + 1) - 2 * sequence_a n

theorem part1 : ∃ r, ∀ n, (sequence_a (n + 1) - sequence_a n) = r * (sequence_a n - sequence_a (n - 1)) :=
by
  use 2
  sorry

theorem part2 : ∀ n, sequence_a n = 2^n - 1 :=
by
  intro n
  sorry

theorem part3 : ∀ n ∈ ℕ, n > 0 → (↑n / 2) - (1 / 3) < (finset.range n).sum (λ k, sequence_a k / sequence_a (k + 1)) ∧ (finset.range n).sum (λ k, sequence_a k / sequence_a (k + 1)) < (↑n / 2) :=
by
  intro n hn
  have h : 0 < n := hn
  sorry

end part1_part2_part3_l808_808062


namespace correct_options_l808_808474

open Real

def vec_a : ℝ × ℝ × ℝ := (-2, -1, 1)
def vec_b : ℝ × ℝ × ℝ := (3, 4, 5)

def optionA (a b : ℝ × ℝ × ℝ) : Prop :=
  let vec := (5 * a.fst + 6 * b.fst, 5 * a.snd + 6 * b.snd, 5 * a.thd + 6 * b.thd)
  in a.fst * vec.fst + a.snd * vec.snd + a.thd * vec.thd = 0

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.fst ^ 2 + v.snd ^ 2 + v.thd ^ 2)

def optionB (a b : ℝ × ℝ × ℝ) : Prop :=
  5 * magnitude a = Real.sqrt 3 * magnitude b

def optionC (a b : ℝ × ℝ × ℝ) : Prop :=
  ¬ ∃ λ : ℝ, (2 * a.fst + b.fst = λ * a.fst ∧ 2 * a.snd + b.snd = λ * a.snd ∧ 2 * a.thd + b.thd = λ * a.thd)

def optionD (a b : ℝ × ℝ × ℝ) : Prop :=
  let dot_product := a.fst * b.fst + a.snd * b.snd + a.thd * b.thd
      b_magnitude_squared := b.fst^2 + b.snd^2 + b.thd^2
      proj := (dot_product / b_magnitude_squared * b.fst, dot_product / b_magnitude_squared * b.snd, dot_product / b_magnitude_squared * b.thd)
  in proj = (-3/10, -2/5, -1/2)

theorem correct_options : optionA vec_a vec_b ∧ optionB vec_a vec_b ∧ ¬ optionC vec_a vec_b ∧ optionD vec_a vec_b :=
by
  sorry

end correct_options_l808_808474


namespace smallest_coefficient_term_in_expansion_l808_808081

theorem smallest_coefficient_term_in_expansion {n : ℕ} (h : ∑ k in finset.range (n+1), |((-1 : ℤ)^k * nat.choose n k)| = 32) :
  has_term_with_smallest_coefficient (1 - x)^5 (-10 * x^3) := by sorry

end smallest_coefficient_term_in_expansion_l808_808081


namespace variance_increases_with_addition_of_two_numbers_l808_808465

def set_range (s : List ℤ) : ℤ := (s.foldr max (Int.ofNat 0)) - (s.foldr min (Int.ofNat 0))

theorem variance_increases_with_addition_of_two_numbers (p q : ℤ) (data_set : List ℤ)
  (h_data : data_set = [3, 1, 5, 3, 2])
  (h_range : set_range (data_set ++ [p, q]) = p - q) :
  (variance (data_set ++ [p, q]) > variance data_set) :=
sorry

end variance_increases_with_addition_of_two_numbers_l808_808465


namespace farmer_total_acres_l808_808755

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l808_808755


namespace tangent_line_at_0_maximum_integer_value_of_a_l808_808946

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - a*x + 2

-- Part (1)
-- Prove that the equation of the tangent line to f(x) at x = 0 is x + y - 2 = 0 when a = 2
theorem tangent_line_at_0 {a : ℝ} (h : a = 2) : ∀ x y : ℝ, (y = f x a) → (x = 0) → (y = 2 - x) :=
by 
  sorry

-- Part (2)
-- Prove that if f(x) + 2x + x log(x+1) ≥ 0 holds for all x ≥ 0, then the maximum integer value of a is 4
theorem maximum_integer_value_of_a 
  (h : ∀ x : ℝ, x ≥ 0 → f x a + 2 * x + x * Real.log (x + 1) ≥ 0) : a ≤ 4 :=
by
  sorry

end tangent_line_at_0_maximum_integer_value_of_a_l808_808946


namespace nums_between_2000_and_3000_div_by_360_l808_808519

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l808_808519


namespace find_x_l808_808456

theorem find_x (x : ℝ) : 
  (∑ k in Finset.range 999, x / (k+1) / (k+2)) = 999 → x = 1000 :=
sorry

end find_x_l808_808456


namespace avg_age_women_is_52_l808_808640

-- Definitions
def avg_age_men (A : ℚ) := 9 * A
def total_increase := 36
def combined_age_replaced := 36 + 32
def combined_age_women := combined_age_replaced + total_increase
def avg_age_women (W : ℚ) := W / 2

-- Theorem statement
theorem avg_age_women_is_52 (A : ℚ) : avg_age_women combined_age_women = 52 :=
by
  sorry

end avg_age_women_is_52_l808_808640


namespace find_n_l808_808498

def a_n (n : ℕ) : ℝ := (2^n - 1) / 2^n
def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, a_n (i + 1)

theorem find_n (n : ℕ) (h : S_n n = 321 / 64) : n = 6 := sorry

end find_n_l808_808498


namespace count_divisibles_l808_808517

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l808_808517


namespace no_valid_two_digit_factors_l808_808526

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Main theorem to show: there are no valid two-digit factorizations of 1976
theorem no_valid_two_digit_factors : 
  ∃ (factors : ℕ → ℕ → Prop), (∀ (a b : ℕ), factors a b → (a * b = 1976) → (is_two_digit a) → (is_two_digit b)) → 
  ∃ (count : ℕ), count = 0 := 
sorry

end no_valid_two_digit_factors_l808_808526


namespace carla_soda_water_problem_l808_808004

theorem carla_soda_water_problem
  (water : ℕ) (total_liquid : ℕ) (x : ℕ) : water = 15 → total_liquid = 54 →
  (∀ soda, soda = 3 * water → soda - x + water = total_liquid) → x = 6 :=
begin
  intros h1 h2 h3,
  have soda := 3 * water,
  rw h1 at soda,
  specialize h3 soda rfl,
  linarith,
end

end carla_soda_water_problem_l808_808004


namespace chord_length_of_parabola_l808_808097

variables {x1 x2 : ℝ}

def parabola (y x : ℝ) : Prop := y^2 = 4 * x
def focus_intersects_parabola_at (x y : ℝ) : Prop := parabola y x

theorem chord_length_of_parabola {x1 x2 : ℝ} (hpx1 : focus_intersects_parabola_at x1 2)
  (hpx2 : focus_intersects_parabola_at x2 2) (h_sum : x1 + x2 = 10) :
  ∃ AB, AB = 12 :=
by
  use (x1 + x2 + 2)
  have h_len : x1 + x2 + 2 = 12,
  exact h_sum.add_right 2,
  rw h_len
  exact rfl

end chord_length_of_parabola_l808_808097


namespace max_cos_sum_triangle_l808_808143

theorem max_cos_sum_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  ∃ A_max, (A_max = Real.pi / 3) ∧ (∀ A, cos A + 2 * cos ((B + C) / 2) ≤ cos (Real.pi / 3) + 2 * sin (Real.pi / 6)) ∧ 
  (cos (Real.pi / 3) + 2 * sin (Real.pi / 6) = 3 / 2) :=
by
  sorry

end max_cos_sum_triangle_l808_808143


namespace segment_length_l808_808108

theorem segment_length : 
  let p1 := (2 : ℝ, 7 : ℝ) in
  let p2 := (8 : ℝ, 18 : ℝ) in
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = real.sqrt 157 := 
by {
  intro p1 p2,
  simp only [p1, p2],
  sorry
}

end segment_length_l808_808108


namespace cosine_of_angle_between_a_and_b_projection_of_c_in_direction_of_a_l808_808581

noncomputable def vector_a := (-1: ℝ, 1: ℝ)
noncomputable def vector_b := (4: ℝ, 3: ℝ)
noncomputable def vector_c := (5: ℝ, -2: ℝ)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def cosine_angle (v1 v2 : ℝ × ℝ) : ℝ :=
dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ :=
dot_product v1 v2 / magnitude v1

theorem cosine_of_angle_between_a_and_b :
  cosine_angle vector_a vector_b = -real.sqrt 2 / 10 :=
by sorry

theorem projection_of_c_in_direction_of_a :
  projection vector_a vector_c = -7 / (2 * real.sqrt 2) :=
by sorry

end cosine_of_angle_between_a_and_b_projection_of_c_in_direction_of_a_l808_808581


namespace range_of_a_l808_808952

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) → a > 1 :=
by
  sorry

end range_of_a_l808_808952


namespace mappings_with_two_ones_l808_808502

def A : set ℕ := {i | 1 ≤ i ∧ i ≤ 15}
def B : set ℕ := {0, 1}

theorem mappings_with_two_ones :
  let total_mappings := 2 ^ 15,
      zero_ones := 1,
      one_one_mappings := 15 in
  (total_mappings - (zero_ones + one_one_mappings)) = 32752 :=
by
  let total_mappings := 2 ^ 15
  let zero_ones := 1
  let one_one_mappings := 15
  have h1 : total_mappings = 32768 := by sorry
  have h2 : zero_ones + one_one_mappings = 16 := by sorry
  have h3 : total_mappings - (zero_ones + one_one_mappings) = 32752 := by sorry
  exact h3

end mappings_with_two_ones_l808_808502


namespace only_positive_integer_a_squared_plus_2a_is_perfect_square_l808_808838

/-- Prove that the only positive integer \( a \) for which \( a^2 + 2a \) is a perfect square is \( a = 0 \). -/
theorem only_positive_integer_a_squared_plus_2a_is_perfect_square :
  ∀ (a : ℕ), (∃ (k : ℕ), a^2 + 2*a = k^2) → a = 0 :=
by
  intro a h
  sorry

end only_positive_integer_a_squared_plus_2a_is_perfect_square_l808_808838


namespace sequence_property_l808_808913

-- Conditions as definitions
def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = -(2 / 3)) ∧ (∀ n ≥ 2, S n + (1 / S n) + 2 = a n)

-- The desired property of the sequence
def S_property (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = -((n + 1) / (n + 2))

-- The main theorem
theorem sequence_property (a S : ℕ → ℝ) (h_seq : seq a S) : S_property S := sorry

end sequence_property_l808_808913


namespace batsman_inning_problem_l808_808311

-- Define the problem in Lean 4
theorem batsman_inning_problem (n R : ℕ) (h1 : R = 55 * n) (h2 : R + 110 = 60 * (n + 1)) : n + 1 = 11 := 
  sorry

end batsman_inning_problem_l808_808311


namespace trigonometric_expression_value_l808_808939

-- Define the conditions
variables {α : Real}
-- α is in the second quadrant
axiom sin_alpha : Real := 1/3
axiom alpha_in_second_quadrant : (0 < α ∧ α < π)
axiom tan_alpha : Real := 2

-- Lean statement for the problem
theorem trigonometric_expression_value : 
  ∀ α, (0 < α ∧ α < π) → (sin α = 1 / 3) → (tan α = 2) →
  (4 * cos (2 * π - α) + sin (π - α)) / (3 * sin (π / 2 - α) + 2 * cos (π / 2 + α)) = -6 :=
begin
  intros α h1 h2 h3,
  sorry
end

end trigonometric_expression_value_l808_808939


namespace indeterminate_157th_digit_l808_808120

theorem indeterminate_157th_digit 
  (h : (525 / 2027 : ℝ) = 0.258973) : 
  ∃ d : ℕ, d = 157 → 
  ¬∃ digit, (digit is the 157th digit of the decimal expansion of (525/2027)) :=
begin
  sorry
end

end indeterminate_157th_digit_l808_808120


namespace total_number_of_students_l808_808663

theorem total_number_of_students 
  (b g : ℕ) 
  (ratio_condition : 5 * g = 8 * b) 
  (girls_count : g = 160) : 
  b + g = 260 := by
  sorry

end total_number_of_students_l808_808663


namespace least_add_to_palindrome_l808_808306

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let chars := n.toString in
  chars = chars.reverse

theorem least_add_to_palindrome :
  ∃ n : ℕ, is_palindrome (n + 52712) ∧ ∀ m : ℕ, (is_palindrome (m + 52712) → n ≤ m) → n = 113 :=
begin
  sorry,
end

end least_add_to_palindrome_l808_808306


namespace triangle_inequality_l808_808147

theorem triangle_inequality 
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
by
  sorry

end triangle_inequality_l808_808147


namespace quadrilateral_parallelogram_iff_dist_sum_eq_semiperimeter_l808_808624

variable {A B C D K L M N : Type}
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

def midpoint (P Q : A) : A := (P + Q) / 2

-- Conditions: K, L, M, N are the midpoints of the sides AB, BC, CD, DA respectively.
variable (hK : K = midpoint A B)
variable (hL : L = midpoint B C)
variable (hM : M = midpoint C D)
variable (hN : N = midpoint D A)

def semiperimeter (P Q R S : ℝ) : ℝ := (P + Q + R + S) / 2

noncomputable def distance (P Q : A) : ℝ := (∥P - Q∥ : ℝ)

theorem quadrilateral_parallelogram_iff_dist_sum_eq_semiperimeter 
  (AB BC CD DA : ℝ)
  (hSum : distance K M + distance N L = semiperimeter AB BC CD DA) :
  ∃ (parallelogram : Prop), parallelogram ↔ 
  (∑ (distance K M) (distance N L) = semiperimeter AB BC CD DA) := sorry

end quadrilateral_parallelogram_iff_dist_sum_eq_semiperimeter_l808_808624


namespace original_sets_exist_l808_808881

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808881


namespace dan_blue_marbles_l808_808362

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end dan_blue_marbles_l808_808362


namespace part_I_probability_part_II_probability_l808_808497

def quadratic_function (a b x : ℝ) : ℝ := a * x^2 - 4 * b * x + 1

def set_P : Set ℝ := {1, 2, 3}
def set_Q : Set ℝ := {-1, 1, 2, 3, 4}

def increasing_in_interval (a b : ℝ) : Prop :=
  a > 0 ∧ 2 * b ≤ a

theorem part_I_probability :
  let favorable_outcomes := {(a, b) | a ∈ set_P ∧ b ∈ set_Q ∧ increasing_in_interval a b}
  let total_outcomes := set_P.product set_Q
  (favorable_outcomes.to_finset.card : ℚ) / (total_outcomes.to_finset.card : ℚ) = 1 / 3 :=
by
  sorry

def region (a b : ℝ) : Prop :=
  a + b - 8 ≤ 0 ∧ a > 0 ∧ b > 0

theorem part_II_probability :
  let favorable_region := {(a, b) | region a b ∧ increasing_in_interval a b}
  let total_region := {(a, b) | region a b}
  (favorable_region.to_finset.card : ℚ) / (total_region.to_finset.card : ℚ) = 1 / 3 :=
by
  sorry

end part_I_probability_part_II_probability_l808_808497


namespace katie_baked_7_cupcakes_l808_808848

theorem katie_baked_7_cupcakes :
  ∃ C : ℕ, (let total_baked = C + 5 in total_baked = 8 + 4) ∧ C = 7 :=
by
  sorry

end katie_baked_7_cupcakes_l808_808848


namespace find_CE_l808_808162

open Real EuclideanGeometry

noncomputable def length_CE (AB' CD AE : ℝ) : ℝ :=
  let area1 := (1 / 2) * AB' * CE
  let area2 := (1 / 2) * AE * CD
  have h1 : area1 = area2 := by sorry  -- Placeholder for actual area computation equality.
  have h2 : 7 * CE = 36 := by sorry   -- Derived from equating both area computations.
  CE = 36 / 7

theorem find_CE :
  AB' = 7 → CD = 12 → AE = 3 → AB' ∥ CE → CD ∥ AE → CE = 36 / 7 :=
by sorry  -- This is a placeholder for the actual proof.

end find_CE_l808_808162


namespace units_digit_7_pow_6_pow_5_l808_808410

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808410


namespace store_total_income_l808_808549

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l808_808549


namespace minimum_possible_perimeter_l808_808263

theorem minimum_possible_perimeter (a b c : ℤ) (h1 : 2 * a + 8 * c = 2 * b + 10 * c) 
                                  (h2 : 4 * c * (sqrt (a^2 - (4 * c)^2)) = 5 * c * (sqrt (b^2 - (5 * c)^2))) 
                                  (h3 : a - b = c) : 
    2 * a + 8 * c = 740 :=
by
  sorry

end minimum_possible_perimeter_l808_808263


namespace second_car_speed_l808_808691

def speed_of_second_car (d : ℝ) (t : ℝ) (v1 : ℝ) := (d - v1 * t) / t

theorem second_car_speed (d : ℝ := 60) (t : ℝ := 2) (v1 : ℝ := 13) :
  speed_of_second_car d t v1 = 17 :=
by
  rw [speed_of_second_car, show d - v1 * t = 34, by norm_num1, show 34 / t = 17, by norm_num1]
  norm_num

end second_car_speed_l808_808691


namespace rearrangement_count_l808_808372

/--
  Eight chairs are evenly spaced around a circular table. One person is seated in each chair.
  Each person gets up and sits down in a chair that is not the same and is not adjacent
  to the chair they originally occupied, so that again one person is seated in each chair.
  Prove that the number of ways this can be done is 42.
--/
theorem rearrangement_count : 
  let chairs := {0, 1, 2, 3, 4, 5, 6, 7} in
  ∃ σ : chairs → chairs, 
    (∀ i, σ i ≠ i) ∧
    (∀ i, σ i ≠ (i + 1) % 8) ∧
    (∀ i, σ i ≠ (i - 1 + 8) % 8) ∧
    (∀ i, ∃! j, σ j = i) ∧
    (∃ perm_count : ℕ, perm_count = 42) :=
sorry

end rearrangement_count_l808_808372


namespace units_digit_7_pow_6_pow_5_l808_808426

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808426


namespace two_people_each_room_probability_l808_808269

noncomputable def probability_person_each_room : ℚ :=
  let total_arrangements := 4
  let favorable_arrangements := 2
  favorable_arrangements / total_arrangements

theorem two_people_each_room_probability :
  let A := "Person A"
  let B := "Person B"
  let rooms := set.univ
  probability_person_each_room = 1 / 2 :=
sorry

end two_people_each_room_probability_l808_808269


namespace different_color_marbles_l808_808543

theorem different_color_marbles 
  (red marbles = 20)
  (green_marbles : ℕ := red_marbles * 3)
  (yellow_marbles : ℕ := green_marbles / 5)
  (total_marbles : ℕ := green_marbles * 4) :
  total_marbles - (red_marbles + green_marbles + yellow_marbles) = 148 :=
by
-- Proof skipped
sorry

end different_color_marbles_l808_808543


namespace solution_set_l808_808482

noncomputable def f : ℝ → ℝ := sorry
axiom f_at_1 : f 1 = 1
axiom f_der_lt_half : ∀ x : ℝ, f' x < (1 / 2)

theorem solution_set : { x : ℝ | f x < x / 2 + 1 / 2 } = set.Ioi 1 :=
sorry

end solution_set_l808_808482


namespace cindy_correct_answer_l808_808002

theorem cindy_correct_answer (x : ℝ) (h : (x - 10) / 5 = 50) : (x - 5) / 10 = 25.5 :=
sorry

end cindy_correct_answer_l808_808002


namespace find_line_l_l808_808930

noncomputable def line_equation (p₁ p₂ : ℝ) (q₁ q₂ : ℝ) :=
  (Set l : ℝ → ℝ → Prop | (λ (x y : ℝ), x * (p₂ - q₂) + y * (q₁ - p₁) + (p₁ * q₂ - q₁ * p₂) = 0))

def line_l (x y : ℝ) : Prop := (y = 0 * x - 1)
def l_intercept_point : ℝ := 7/2

def line_l1 (x y : ℝ) : Prop := (2 * x + y - 6 = 0)
def line_l2 (x y : ℝ) : Prop := (4 * x + 2 * y - 5 = 0)

theorem find_line_l (x y : ℝ) :
  (x = 0 ∧ line_l x y) ∨ (3 * x + 4 * y + 4 = 0 ∧ line_l x y) :=
sorry

end find_line_l_l808_808930


namespace semicircle_perimeter_l808_808298

def radius : ℝ := 6.4
def diameter : ℝ := 2 * radius
def circumference : ℝ := 2 * Real.pi * radius
def half_circumference : ℝ := Real.pi * radius

noncomputable def perimeter : ℝ := half_circumference + diameter

theorem semicircle_perimeter :
  perimeter ≈ 32.896 :=
begin
  sorry
end

end semicircle_perimeter_l808_808298


namespace probability_reaching_shore_l808_808791

-- Define the probabilities
def p : ℝ := 0.5
def q : ℝ := 1 - p

-- Define the probability of reaching the shore
def total_probability : ℝ := q / (1 - p * q)

-- The main theorem statement
theorem probability_reaching_shore : total_probability = 2 / 3 :=
by
  sorry

end probability_reaching_shore_l808_808791


namespace series_sum_calc_l808_808809

theorem series_sum_calc : 
  let series := list.range' 5 (200 - 1) → (series.enum.map (λ x, if x.1 % 2 = 0 then x.2 * 10 + 5 else -(x.2 * 10 + 5))),
  series.sum = 1000 :=
by
  sorry

end series_sum_calc_l808_808809


namespace matrix_det_expression_l808_808171

open Matrix

variables {R : Type*} [CommRing R]

theorem matrix_det_expression (a b c d r s t : R)
  (h1: a + b + c + d = 0) -- Sum of the roots is zero for the quartic polynomial
  (h2: a*b + a*c + a*d + b*c + b*d + c*d = r) -- Sum of product of roots taken two at a time
  (h3: a*b*c + a*b*d + a*c*d + b*c*d = -s) -- Sum of product of roots taken three at a time
  (h4: a*b*c*d = t) -- Product of all roots
  :
    det ![![1 + a, 1, 1, 1],
           ![1, 1 + b, 1, 1],
           ![1, 1, 1 + c, 1],
           ![1, 1, 1, 1 + d]] = t - s + r :=
begin
  sorry
end

end matrix_det_expression_l808_808171


namespace triangle_is_right_triangle_l808_808536

-- Define the vectors in the context of a triangle
variables {A B C : Type} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variable (AC : A)
variable (AB : B)
variable (BC : C)

-- Condition given in the problem
axiom condition : inner_product_space.inner BC AC - inner_product_space.inner AB AC = inner_product_space.inner AC AC

-- Definition of a right triangle
def right_triangle (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] := 
  ∃ (γ : ℝ), γ = π / 2

-- Theorem to prove
theorem triangle_is_right_triangle : right_triangle A B C :=
  sorry

end triangle_is_right_triangle_l808_808536


namespace percentage_cut_away_in_second_week_l808_808776

theorem percentage_cut_away_in_second_week :
  ∃(x : ℝ), (x / 100) * 142.5 * 0.9 = 109.0125 ∧ x = 15 :=
by
  sorry

end percentage_cut_away_in_second_week_l808_808776


namespace number_of_points_l808_808078

structure Point (α : Type*) := (x y : α)

variables {α : Type*} [Field α] 

def vector_add (a b : Point α) : Point α := Point.mk (a.x + b.x) (a.y + b.y)
def vector_scale (c : α) (a : Point α) : Point α := Point.mk (c * a.x) (c * a.y)

theorem number_of_points (A1 A2 A3 : Point α) (h_non_collinear : (A2.x - A1.x) * (A3.y - A1.y) ≠ (A3.x - A1.x) * (A2.y - A1.y)) :
  ∃ (M : Point α) (λ : α), 
  (vector_scale λ (vector_add (vector_add A1 A2) A3) = M) ∧
  let MA1 := Point.mk (-M.x + A1.x) (-M.y + A1.y),
      MA2 := Point.mk (A2.x - M.x) (A2.y - M.y),
      MA3 := Point.mk (A3.x - M.x) (A3.y - M.y) in
  (vector_add (vector_add MA1 MA2) MA3).x ^ 2 + (vector_add (vector_add MA1 MA2) MA3).y ^ 2 = 1 → ∃! λ, True :=
sorry

end number_of_points_l808_808078


namespace prime_has_property_P_l808_808845

theorem prime_has_property_P (n : ℕ) (hn : Prime n) (a : ℤ) (h : n ∣ a ^ n - 1) : n ^ 2 ∣ a ^ n - 1 := 
sorry

end prime_has_property_P_l808_808845


namespace workbook_arrangement_l808_808599

-- Define the condition of having different Korean and English workbooks
variables (K1 K2 : Type) (E1 E2 : Type)

-- The main theorem statement
theorem workbook_arrangement :
  ∃ (koreanWorkbooks englishWorkbooks : List (Type)), 
  (koreanWorkbooks.length = 2) ∧
  (englishWorkbooks.length = 2) ∧
  (∀ wb ∈ (koreanWorkbooks ++ englishWorkbooks), wb ≠ wb) ∧
  (∃ arrangements : Nat,
    arrangements = 12) :=
  sorry

end workbook_arrangement_l808_808599


namespace path_length_B_l808_808339

-- Definitions
def radius : Real := 4 / Real.pi

-- Theorem: Prove the length of the path B travels
theorem path_length_B (r : Real) (hr : r = radius) : 
  ∃ d : Real, d = 4 := by
  sorry

end path_length_B_l808_808339


namespace original_sets_exist_l808_808876

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808876


namespace correct_calculation_l808_808290

noncomputable def sqrt (x : ℝ) : ℝ :=
  if x < 0 then 0 else classical.some (exists_sqrt x)

theorem correct_calculation :
  let A := sqrt 2 + sqrt 3
  let B := 3 * sqrt 5 - sqrt 5
  let C := 3 * sqrt (1 / 3)
  let D := sqrt 12 / sqrt 3
  D = 2 :=
by
  sorry

end correct_calculation_l808_808290


namespace evaluate_expression_l808_808811

theorem evaluate_expression : 
  (2^2 + 2^1 + 2^0) / (2^(-1) + 2^(-2) + 2^(-3)) = 8 := 
by
  sorry

end evaluate_expression_l808_808811


namespace range_of_sum_l808_808590

def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 - 6*x + 6 else 3*x + 4

theorem range_of_sum (x1 x2 x3 : ℝ) (h1 : x1 ≠ x2) (h2 : x1 ≠ x3) (h3 : x2 ≠ x3) (hx : f x1 = f x2 ∧ f x1 = f x3) :
  \frac {11}{3} < x1 + x2 + x3 ∧ x1 + x2 + x3 < 6 := 
  sorry

end range_of_sum_l808_808590


namespace diane_head_start_l808_808129

theorem diane_head_start (x : ℝ) :
  (100 - 11.91) / (88.09 + x) = 99.25 / 100 ->
  abs (x - 12.68) < 0.01 := 
by
  sorry

end diane_head_start_l808_808129


namespace num_triangles_with_perimeter_9_l808_808508

theorem num_triangles_with_perimeter_9 : 
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 6 ∧ 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 9 ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ a ≤ b ∧ b ≤ c) := 
sorry

end num_triangles_with_perimeter_9_l808_808508


namespace find_m_plus_n_l808_808092

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + x

theorem find_m_plus_n (m n : ℝ) (h1 : m < n ∧ n ≤ 1) (h2 : ∀ (x : ℝ), m ≤ x ∧ x ≤ n → 3 * m ≤ f x ∧ f x ≤ 3 * n) : m + n = -4 :=
by
  have H1 : - (1 / 2) * m^2 + m = 3 * m := sorry
  have H2 : - (1 / 2) * n^2 + n = 3 * n := sorry
  sorry

end find_m_plus_n_l808_808092


namespace num_ways_to_select_sets_l808_808358

open Finset Function

-- Define the set T
def T : Finset ℕ := {u, v, w, x, y, z}

-- Define the constraints on subsets A and B
def valid_sets (A B : Finset ℕ) : Prop :=
  A ∪ B = T ∧ A ∩ B.card = 3

-- Main statement to be proved
theorem num_ways_to_select_sets : 
  (Finset.univ.filter (λ A => ∃ (B : Finset ℕ), valid_sets A B)).card = 80 :=
by sorry

end num_ways_to_select_sets_l808_808358


namespace minor_axis_of_ellipse_l808_808366

-- Conditions: The equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop := 9 * x^2 + y^2 = 36

-- The "length of the minor axis" is twice the semi-minor axis length
def minor_axis_length {x y : ℝ} (h : ellipse_equation x y) : ℝ := 
  let a := 6 in  -- Semi-major axis length
  let b := sqrt 4 in -- Semi-minor axis length calc from 9x^2 + y^2 = 36 converted to standard form form
  2 * b -- Length of the minor axis

-- Lean Statement: Given the ellipse equation, prove that the length of the minor axis is 4
theorem minor_axis_of_ellipse : ∀ (x y : ℝ), ellipse_equation x y → minor_axis_length _ = 4 :=
by sorry

end minor_axis_of_ellipse_l808_808366


namespace intersection_of_A_and_B_l808_808920

def setA : Set ℝ := {x | abs (x - 1) < 2}

def setB : Set ℝ := {x | x^2 + x - 2 > 0}

theorem intersection_of_A_and_B :
  (setA ∩ setB) = {x | 1 < x ∧ x < 3} :=
sorry

end intersection_of_A_and_B_l808_808920


namespace units_digit_7_pow_6_pow_5_l808_808423

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808423


namespace product_xyz_l808_808122

noncomputable def xyz_value (x y z : ℝ) :=
  x * y * z

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 3) :
  xyz_value x y z = -1 :=
by
  sorry

end product_xyz_l808_808122


namespace correct_statements_l808_808051

variable (a b : ℝ)

theorem correct_statements (hab : a * b > 0) :
  (|a + b| > |a| ∧ |a + b| > |a - b|) ∧ (¬ (|a + b| < |b|)) ∧ (¬ (|a + b| < |a - b|)) :=
by
  -- The proof is omitted as per instructions
  sorry

end correct_statements_l808_808051


namespace sum_of_roots_correct_l808_808842

open Polynomial

noncomputable def sum_of_roots_of_polynomials : ℚ :=
let p1 : Polynomial ℚ := 3 * X^4 + 2 * X^3 - 9 * X^2 + 13 * X - 7,
    p2 : Polynomial ℚ := 7 * X^3 - 49 * X^2 + 35 in
(p1.roots.sum + p2.roots.sum)

theorem sum_of_roots_correct :
  sum_of_roots_of_polynomials = 19 / 3 :=
sorry

end sum_of_roots_correct_l808_808842


namespace pencil_packing_l808_808189

theorem pencil_packing (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (a % 10 = 7) →
  (a % 12 = 9) →
  (a = 237 ∨ a = 297) :=
by {
  assume h_range h_red_boxes h_blue_boxes,
  sorry
}

end pencil_packing_l808_808189


namespace perpendicular_condition_l808_808060

-- The conditions of the problem
variables {α β : Type}
variables (a : line α) (α : plane β) (β : plane β)

-- Definitions for perpendicular line and plane
def line_contained_in_plane (a : line α) (α : plane β) : Prop := sorry
def line_perpendicular_plane (a : line α) (β : plane β) : Prop := sorry
def plane_perpendicular_plane (α : plane β) (β : plane β) : Prop := sorry

-- Given condition 
axiom line_a_is_in_plane_alpha : line_contained_in_plane a α

-- The proof problem
theorem perpendicular_condition (a : line α) (α β : plane β) (h : line_contained_in_plane a α) :
  (plane_perpendicular_plane α β → line_perpendicular_plane a β) ∧
  ¬(line_perpendicular_plane a β → plane_perpendicular_plane α β) :=
  
sorry

end perpendicular_condition_l808_808060


namespace arithmetic_sum_is_7737_5_l808_808007

noncomputable def arithmetic_sequence_sum := 
  \sum_{k=0}^{124} \lfloor 0.5 + k * 0.8 \rfloor

theorem arithmetic_sum_is_7737_5 :
  arithmetic_sequence_sum = 7737.5 := 
by
  sorry

end arithmetic_sum_is_7737_5_l808_808007


namespace find_n_l808_808067

-- Definitions for conditions given in the problem
def a₂ (a : ℕ → ℕ) : Prop := a 2 = 3
def consecutive_sum (S : ℕ → ℕ) (n : ℕ) : Prop := ∀ n > 3, S n - S (n - 3) = 51
def total_sum (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 100

-- The main proof problem
theorem find_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h₁ : a₂ a) (h₂ : consecutive_sum S n) (h₃ : total_sum S n) : n = 10 :=
sorry

end find_n_l808_808067


namespace line_PQ_symmetry_probability_l808_808775

-- Define the set of points on the 13x13 grid
def grid_points : set (ℤ × ℤ) := {(i, j) | i ∈ finset.range 13 ∧ j ∈ finset.range 13}

-- Define the center point P
def P : ℤ × ℤ := (6, 6)

-- Define symmetric points function
def symmetric_points (p : ℤ × ℤ) : set (ℤ × ℤ) :=
  { (2 * P.1 - p.1, p.2), (p.1, 2 * P.2 - p.2), (2 * P.1 - p.1, 2 * P.2 - p.2) } ∩ grid_points

-- Define the probability function for line PQ being a line of symmetry
def probability_symmetry : ℚ :=
  (4 * 12) / (169 - 1)

-- State the theorem
theorem line_PQ_symmetry_probability :
  probability_symmetry = 2 / 7 :=
sorry

end line_PQ_symmetry_probability_l808_808775


namespace intersection_of_M_and_N_l808_808953

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 1}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def intersection : Set ℝ := {z | -1 ≤ z ∧ z ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {z | -1 ≤ z ∧ z ≤ 3} := 
sorry

end intersection_of_M_and_N_l808_808953


namespace tom_purchases_l808_808250

def total_cost_before_discount (price_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  price_per_box * num_boxes

def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

def total_cost_after_discount (total_cost : ℝ) (discount_amount : ℝ) : ℝ :=
  total_cost - discount_amount

def remaining_boxes (total_boxes : ℕ) (given_boxes : ℕ) : ℕ :=
  total_boxes - given_boxes

def total_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  num_boxes * pieces_per_box

theorem tom_purchases
  (price_per_box : ℝ) (num_boxes : ℕ) (discount_rate : ℝ) (given_boxes : ℕ) (pieces_per_box : ℕ) :
  (price_per_box = 4) →
  (num_boxes = 12) →
  (discount_rate = 0.15) →
  (given_boxes = 7) →
  (pieces_per_box = 6) →
  total_cost_after_discount (total_cost_before_discount price_per_box num_boxes) 
                             (discount (total_cost_before_discount price_per_box num_boxes) discount_rate)
  = 40.80 ∧
  total_pieces (remaining_boxes num_boxes given_boxes) pieces_per_box
  = 30 :=
by
  intros
  sorry

end tom_purchases_l808_808250


namespace min_perimeter_of_polygon_formed_by_zeros_of_P_l808_808576

noncomputable def P (z : ℂ) : ℂ := z^8 + (4 * real.sqrt 3 + 6) * z^4 - (4 * real.sqrt 3 + 7)

theorem min_perimeter_of_polygon_formed_by_zeros_of_P : 
  let zs := { z : ℂ | P z = 0 }
  let perimeter := ∑ z in zs, complex.abs (z - complex.conj z)
  perimeter = 8 * real.sqrt 2 :=
by
  sorry

end min_perimeter_of_polygon_formed_by_zeros_of_P_l808_808576


namespace triangle_side_difference_l808_808565

theorem triangle_side_difference (x : ℕ) : 3 < x ∧ x < 17 → (∃ a b : ℕ, 3 < a ∧ a < 17 ∧ 3 < b ∧ b < 17 ∧ a - b = 12) :=
by
  sorry

end triangle_side_difference_l808_808565


namespace cashier_total_bills_l808_808313

theorem cashier_total_bills
  (total_value : ℕ)
  (num_ten_bills : ℕ)
  (num_twenty_bills : ℕ)
  (h1 : total_value = 330)
  (h2 : num_ten_bills = 27)
  (h3 : num_twenty_bills = 3) :
  num_ten_bills + num_twenty_bills = 30 :=
by
  -- Proof goes here
  sorry

end cashier_total_bills_l808_808313


namespace sum_of_satisfying_numbers_l808_808242

-- Lean doesn't support non-numerical floor and fractional parts directly,
-- hence, we define the necessary constructs explicitly.

def floor (x : ℝ) : ℤ := ⌊x⌋
def fractional_part (x : ℝ) : ℝ := x - floor x

def satisfies_condition (x : ℝ) : Prop := 
  25 * fractional_part x + floor x = 25

theorem sum_of_satisfying_numbers : 
  (∑ n in Finset.range 25, (1 : ℝ) + 0.96 * (n + 1)) = 337 :=
by
  -- For now, we skip the proof.
  sorry

end sum_of_satisfying_numbers_l808_808242


namespace initial_speed_solution_l808_808786

def initial_speed_problem : Prop :=
  ∃ V : ℝ, 
    (∀ t t_new : ℝ, 
      t = 300 / V ∧ 
      t_new = t - 4 / 5 ∧ 
      (∀ d d_remaining : ℝ, 
        d = V * (5 / 4) ∧ 
        d_remaining = 300 - d ∧ 
        t_new = (5 / 4) + d_remaining / (V + 16)) 
    ) → 
    V = 60

theorem initial_speed_solution : initial_speed_problem :=
by
  unfold initial_speed_problem
  sorry

end initial_speed_solution_l808_808786


namespace irrational_2pi_l808_808292

theorem irrational_2pi : 
  (∀ a b : ℤ, 11 / 3 ≠ a / b) ∧ 
  (∀ a b : ℤ, sqrt 4 ≠ a / b) ∧ 
  (∀ a b : ℤ, real.cbrt 8 ≠ a / b) ∧ 
  (∀ a b : ℤ, 2 * real.pi ≠ a / b) :=
by
  sorry

end irrational_2pi_l808_808292


namespace base_8_to_base_10_4652_l808_808819

def convert_base_8_to_base_10 (n : ℕ) : ℕ :=
  (4 * 8^3) + (6 * 8^2) + (5 * 8^1) + (2 * 8^0)

theorem base_8_to_base_10_4652 :
  convert_base_8_to_base_10 4652 = 2474 :=
by
  -- Skipping the proof steps
  sorry

end base_8_to_base_10_4652_l808_808819


namespace quadratic_axis_of_symmetry_l808_808650

theorem quadratic_axis_of_symmetry (b c : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0, 3))
  (hB : B = (2, 3)) 
  (h_passA : 3 = 0^2 + b * 0 + c) 
  (h_passB : 3 = 2^2 + b * 2 + c) : 
  ∃ x, x = 1 :=
by {
  -- Given: The quadratic function y = x^2 + bx + c
  -- Conditions: Passes through A(0, 3) and B(2, 3)
  -- We need to prove: The axis of symmetry is x = 1
  sorry,
}

end quadratic_axis_of_symmetry_l808_808650


namespace correct_propositions_l808_808076

noncomputable theory

variables (a b : Type*) [line a] [line b] (α : Type*) [plane α]

axiom distinct_lines (a ≠ b : Prop)

def prop1 : Prop := ∀ {a b α}, (a ∥ b) → (a ⊥ α) → (b ⊥ α)
def prop2 : Prop := ∀ {a b α}, (a ⊥ α) → (b ⊥ α) → (a ∥ b)

theorem correct_propositions (a b : Type*) [line a] [line b] (α : Type*) [plane α] :
  (prop1 a b α) ∧ (prop2 a b α) :=
by
  split,
  sorry,
  sorry

end correct_propositions_l808_808076


namespace inequality_proof_l808_808860

variable {f : ℝ → ℝ}

-- Conditions
def f_positive (x : ℝ) : Prop := f(x) > 0

def inequality_condition (x : ℝ) : Prop := (2 * x + 3) / x > - (f''(x) / f(x))

-- Theorem statement
theorem inequality_proof (h1 : ∀ x > 0, f_positive x) (h2 : ∀ x > 0, inequality_condition x) :
  e ^ (2 * e + 3) * f e < e ^ (2 * π) * π ^ 3 * f π :=
sorry

end inequality_proof_l808_808860


namespace max_distance_proof_l808_808472

noncomputable def max_distance_from_point_to_line : ℝ :=
  let point_A := (-1 / 2, 1 / 2) in
  let parabola_C := (λ (y : ℝ), (y ^ 2) / 2) in
  let p := 1 in
  let axis := - (p / 2) in
  let line_MN := (λ (y : ℝ), 2 * y) in
  let points_on_parabola := {M N : ℝ × ℝ // (∃ y : ℝ, M = (2 * y, y) ∧ N = (2 * (-y), -y))} in
  let origin := (0, 0) in
  let dot_product_condition := (λ (M N : ℝ × ℝ), M.1 * N.1 + M.2 * N.2 = 3) in
  let D := (3, 0) in
  let max_distance := Real.sqrt ((3 + 1 / 2) ^ 2 + (1 / 2) ^ 2) in
  max_distance

theorem max_distance_proof :
  ∃ MN : set (ℝ × ℝ), (point_A = (-1 / 2, 1 / 2) ∧
                       parabola_C = (λ (y : ℝ), (y ^ 2) / 2) ∧
                       ∀ M N ∈ MN, dot_product_condition M N) →
                      max_distance_from_point_to_line = (5 * Real.sqrt 2) / 2 :=
by {
  sorry
}

end max_distance_proof_l808_808472


namespace smallest_N_value_l808_808167

theorem smallest_N_value :
  ∃ (a b c d e f : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧
    a + b + c + d + e + f = 1800 ∧
    let N := max (max (max (max (a + b) (b + c)) (c + d)) (d + e)) (e + f) in
    N = 361 :=
begin
  sorry
end

end smallest_N_value_l808_808167


namespace limit_expression_l808_808354

open Filter

theorem limit_expression :
  tendsto (λ n : ℕ, (n^2 * (n + 6) : ℝ) / (12 * n^3 + 7)) atTop (𝓝 (1 / 12)) :=
sorry

end limit_expression_l808_808354


namespace cos_neg_60_eq_half_l808_808352

theorem cos_neg_60_eq_half :
  ∀ (θ : ℝ), θ = 60 → cos(-θ) = (1:ℝ)/2 :=
by
  sorry

end cos_neg_60_eq_half_l808_808352


namespace conic_section_represents_ellipse_y_axis_l808_808207

-- Definitions based on the conditions
def internal_angle (θ : ℝ) : Prop := θ ∈ (0, Real.pi)

def sin_plus_cos_eq (θ : ℝ) : Prop := Real.sin θ + Real.cos θ = 7 / 13

-- The main theorem statement
theorem conic_section_represents_ellipse_y_axis 
  (θ : ℝ) (h_angle : internal_angle θ) (h_sin_cos_eq : sin_plus_cos_eq θ) :
  ∃ (a b : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ a < b :=
sorry

end conic_section_represents_ellipse_y_axis_l808_808207


namespace number_from_first_group_is_6_l808_808695

-- Defining conditions
def num_students : Nat := 160
def sample_size : Nat := 20
def groups := List.range' 0 num_students (num_students / sample_size)

def num_from_group_16 (x : Nat) : Nat := 8 * 15 + x
def drawn_number_from_16 : Nat := 126

-- Main theorem
theorem number_from_first_group_is_6 : ∃ x : Nat, num_from_group_16 x = drawn_number_from_16 ∧ x = 6 := 
by
  sorry

end number_from_first_group_is_6_l808_808695


namespace total_length_infinite_sum_l808_808910

-- Define the infinite sums
noncomputable def S1 : ℝ := ∑' n : ℕ, (1 / (3^n))
noncomputable def S2 : ℝ := (∑' n : ℕ, (1 / (5^n))) * Real.sqrt 3
noncomputable def S3 : ℝ := (∑' n : ℕ, (1 / (7^n))) * Real.sqrt 5

-- Define the total length
noncomputable def total_length : ℝ := S1 + S2 + S3

-- The statement of the theorem
theorem total_length_infinite_sum : total_length = (3 / 2) + (Real.sqrt 3 / 4) + (Real.sqrt 5 / 6) :=
by
  sorry

end total_length_infinite_sum_l808_808910


namespace negation_of_both_even_l808_808232

-- Definitions
def even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Main statement
theorem negation_of_both_even (a b : ℕ) : ¬ (even a ∧ even b) ↔ (¬even a ∨ ¬even b) :=
by sorry

end negation_of_both_even_l808_808232


namespace original_sets_exist_l808_808906

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808906


namespace loaned_out_books_is_50_l808_808767

-- Define the conditions
def initial_books : ℕ := 75
def end_books : ℕ := 60
def percent_returned : ℝ := 0.70

-- Define the variable to represent the number of books loaned out
noncomputable def loaned_out_books := (15:ℝ) / (1 - percent_returned)

-- The target theorem statement we need to prove
theorem loaned_out_books_is_50 : loaned_out_books = 50 :=
by
  sorry

end loaned_out_books_is_50_l808_808767


namespace train_speed_in_km_per_hr_l808_808332

-- Conditions
def time_in_seconds : ℕ := 9
def length_in_meters : ℕ := 175

-- Conversion factor from m/s to km/hr
def meters_per_sec_to_km_per_hr (speed_m_per_s : ℚ) : ℚ :=
  speed_m_per_s * 3.6

-- Question as statement
theorem train_speed_in_km_per_hr :
  meters_per_sec_to_km_per_hr ((length_in_meters : ℚ) / (time_in_seconds : ℚ)) = 70 := by
  sorry

end train_speed_in_km_per_hr_l808_808332


namespace find_n_values_l808_808447

theorem find_n_values : {n : ℕ | n ≥ 1 ∧ n ≤ 6 ∧ ∃ a b c : ℤ, a^n + b^n = c^n + n} = {1, 2, 3} :=
by sorry

end find_n_values_l808_808447


namespace hexahedron_faces_l808_808459

-- Definition of hexahedron
def is_hexahedron (P : Type) : Prop := ∃ (f : P → Type), ∃ (fs : Fin 6), ∀ x : P, f x = fs

-- Number of faces of a hexahedron
theorem hexahedron_faces (P : Type) (h : is_hexahedron P) : ∃ n : ℕ, n = 6 := by
  exists 6
  rfl

end hexahedron_faces_l808_808459


namespace Kelly_egg_price_l808_808574

/-- Kelly has 8 chickens, and each chicken lays 3 eggs per day.
Kelly makes $280 in 4 weeks by selling all the eggs.
We want to prove that Kelly sells a dozen eggs for $5. -/
theorem Kelly_egg_price (chickens : ℕ) (eggs_per_day_per_chicken : ℕ) (earnings_in_4_weeks : ℕ)
  (days_in_4_weeks : ℕ) (eggs_per_dozen : ℕ) (price_per_dozen : ℕ) :
  chickens = 8 →
  eggs_per_day_per_chicken = 3 →
  earnings_in_4_weeks = 280 →
  days_in_4_weeks = 28 →
  eggs_per_dozen = 12 →
  price_per_dozen = earnings_in_4_weeks / ((chickens * eggs_per_day_per_chicken * days_in_4_weeks) / eggs_per_dozen) →
  price_per_dozen = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Kelly_egg_price_l808_808574


namespace probability_n_lt_m_add_one_l808_808309

theorem probability_n_lt_m_add_one :
  let events := { (m, n) | m ∈ {1, 2, 3, 4} ∧ n ∈ {1, 2, 3, 4} }
  let favorable_events := { (m, n) ∈ events | n < m + 1 }
  by
  have total_events : ℕ := 16
  have num_favorable_events : ℕ := 10
  have probability := (num_favorable_events : ℚ) / total_events
  exact probability = 5 / 8
sorry

end probability_n_lt_m_add_one_l808_808309


namespace russian_players_exclusive_pairs_probability_l808_808212

theorem russian_players_exclusive_pairs_probability :
  let total_players := 10
  let russian_players := 4
  (russian_players * (russian_players - 1) / (total_players - 1)) * ((russian_players - 2) * (russian_players - 3) / (total_players - 3)) = 1 / 21 := 
begin
  sorry,
end

end russian_players_exclusive_pairs_probability_l808_808212


namespace geo_sequence_a_k_l808_808937

noncomputable def a_k (n k : ℕ) : ℕ :=
let S_n := λ n, k * 2^n - 3 in
if n = 1 then S_n 1
else if n = 2 then S_n 2 - S_n 1
else S_n n - S_n (n-1)

theorem geo_sequence_a_k (k : ℕ) (hk : k = 3) : a_k 3 k = 12 :=
by
  have S_n : ℕ → ℕ := λ n, k * 2^n - 3
  rw hk at S_n
  unfold a_k
  simp [S_n]
  sorry

end geo_sequence_a_k_l808_808937


namespace cos_neg_60_eq_half_l808_808353

theorem cos_neg_60_eq_half :
  ∀ (θ : ℝ), θ = 60 → cos(-θ) = (1:ℝ)/2 :=
by
  sorry

end cos_neg_60_eq_half_l808_808353


namespace pentagon_ratio_l808_808911

theorem pentagon_ratio (ABCDE : Set.Point) (K : Point) (L : Point) 
  (h1 : regular_pentagon ABCDE)
  (h2 : K ∈ segment AE)
  (h3 : L ∈ segment CD)
  (h4 : ∠LAE + ∠KCD = 108)
  (h5 : AK / KE = 3 / 7) : 
  CL / AB = 0.7 := 
sorry

end pentagon_ratio_l808_808911


namespace range_of_log_composite_function_l808_808239

theorem range_of_log_composite_function : 
  (set.range (λ x: ℝ, real.logb 2 (3^x + 1))) = set.Ioi 0 :=
sorry

end range_of_log_composite_function_l808_808239


namespace Hilton_marbles_l808_808104

theorem Hilton_marbles :
  ∃ (F : ℕ),
    (let initial := 26 in
     let lost := 10 in
     let given_by_Lori := 2 * lost in
     let final := 42 in
     initial + F - lost + given_by_Lori = final) ∧
    F = 6 :=
begin
  use 6,
  dsimp,
  -- The proof goes here.
  sorry
end

end Hilton_marbles_l808_808104


namespace fraction_darker_tiles_l808_808160

variable (Garden : Type) [Finite (Set (Garden))] (totalTiles : ℕ)
variable (isDarkerTile : Garden → Prop)
variable (repeatedPattern : (Garden → ℕ) → Prop)
variable (centralBlockSize : ℕ := 4)
variable (centerDarkerTiles : ℕ := 3)
variable (sideLength : ℕ := 4) -- size of 4x4 block
variable (totalBlockTiles : ℕ := sideLength * sideLength)
variable (totalDarkerTiles : ℕ := centerDarkerTiles * (sideLength / 2))

-- Fraction of darker tiles in the garden
theorem fraction_darker_tiles : 
  (totalDarkerTiles : ℝ) / (totalBlockTiles : ℝ) = 3 / 4 :=
by {
  have h1 : totalBlockTiles = 16 := by sorry,
  have h2 : totalDarkerTiles = 12 := by sorry,
  have h3 : (totalDarkerTiles : ℝ) / (totalBlockTiles : ℝ) = 12 / 16 := by sorry,
  have h4 : 12 / 16 = 3 / 4 := by sorry,
  exact h4,
}

end fraction_darker_tiles_l808_808160


namespace triangle_area_ABC_is_100_l808_808364

noncomputable def hypotenuse : ℝ := 20

noncomputable def leg_length : ℝ := hypotenuse / Real.sqrt 2

def triangle_area (AB BC : ℝ) : ℝ := 1/2 * AB * BC

theorem triangle_area_ABC_is_100
  (h1 : IsRightTriangle AB BC hypotenuse)
  (h2 : AngleA = 45)
  (h3 : LengthOfHypotenuse AC = hypotenuse) :
  triangle_area (leg_length) (leg_length) = 100 :=
by
  sorry

end triangle_area_ABC_is_100_l808_808364


namespace six_digit_numbers_count_proof_six_digit_numbers_with_four_odd_count_proof_l808_808969

def six_digit_numbers_count : Nat := 
  10 * 9 * 8 * 7 * 6 * 5 - (10 * 9 * 8 * 7 * 6 * 5) / 10

def six_digit_numbers_with_four_odd_count : Nat := 
  (5.choose 4) * (5.choose 2) * 6! - ((5.choose 4) * 4 * 6!)

theorem six_digit_numbers_count_proof :
  six_digit_numbers_count = 136080 := 
by
  unfold six_digit_numbers_count
  rfl

theorem six_digit_numbers_with_four_odd_count_proof :
  six_digit_numbers_with_four_odd_count = 33600 := 
by
  unfold six_digit_numbers_with_four_odd_count
  rfl

end six_digit_numbers_count_proof_six_digit_numbers_with_four_odd_count_proof_l808_808969


namespace problem_1_problem_2_problem_3_l808_808563

-- Condition 1: Sequence definition
def a : ℕ → ℝ
| 1     := 1
| (n+1) := sorry -- we assume the sequence satisfies the condition 3a_na_{n-1} + a_n - a_{n-1} = 0

-- Problem (Ⅰ): Prove that {1 / a_n} is an arithmetic sequence
theorem problem_1 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, 3 * a n * a (n-1) + a n - a (n-1) = 0) :
  ∃ d : ℝ, ∀ n ≥ 2, 1 / a n = 1 / a 1 + (n - 2) * d := sorry

-- Problem (Ⅱ): Find the general term of the sequence {a_n}, it is a_n = 1 / (3n - 2)
theorem problem_2 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, 3 * a n * a (n-1) + a n - a (n-1) = 0) :
  ∀ n, a (n + 1) = 1 / (3 * (n + 1) - 2) := sorry

-- Problem (Ⅲ): Range of λ such that λ a_n + 1 / a_{n+1} ≥ λ holds for any integer n ≥ 2
theorem problem_3 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, 3 * a n * a (n-1) + a n - a (n-1) = 0) 
  (h_gen : ∀ n, a (n + 1) = 1 / (3 * (n + 1) - 2)) :
  ∀ λ, (∀ n ≥ 2, λ * a n + 1 / a (n + 1) ≥ λ) ↔ λ ≤ 28 / 3 := sorry

end problem_1_problem_2_problem_3_l808_808563


namespace da_dc_dot_eq_zero_l808_808558

variables {A B C D O : Type} [InnerProductSpace ℝ (Point) {A B C D O}]
variables (AC AO OC BO OD BA BC : ℝ)

-- Conditions
def ac_length {A C : Point} : AC = 4 := by sorry
def ba_bc_dot {B A C : Point} : InnerProductSpace.isoDot BA BC 12 := by sorry
def ao_eq_oc {A O C : Point} : AO = OC := by sorry
def bo_eq_2od {B O D : Point} : BO = 2 * OD := by sorry

-- Statement to prove
theorem da_dc_dot_eq_zero {D A O C : Point} : InnerProductSpace.isoDot DA DC 0 := by sorry

end da_dc_dot_eq_zero_l808_808558


namespace original_sets_exist_l808_808875

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808875


namespace pencil_count_l808_808192

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l808_808192


namespace f_inequality_l808_808592

-- Definition of odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f is an odd function
variable {f : ℝ → ℝ}
variable (h1 : is_odd_function f)

-- f has a period of 4
variable (h2 : ∀ x, f (x + 4) = f x)

-- f is monotonically increasing on [0, 2)
variable (h3 : ∀ x y, 0 ≤ x → x < y → y < 2 → f x < f y)

theorem f_inequality : f 3 < 0 ∧ 0 < f 1 :=
by 
  -- Place proof here
  sorry

end f_inequality_l808_808592


namespace median_is_3_l808_808089

def moons_list : list ℕ := [0, 0, 1, 2, 3, 5, 27, 29, 31]

def median (l : list ℕ) : ℕ :=
if h : l.length % 2 = 1 then
  let sorted := l.sort (≤)
  in sorted.nth_le (l.length / 2) (by sorry)
else
  0  -- Just a placeholder, since we know l.length % 2 = 1

theorem median_is_3 : median moons_list = 3 :=
by {
  -- proving the median is 3
  sorry
}

end median_is_3_l808_808089


namespace single_bacteria_colony_days_to_limit_l808_808741

theorem single_bacteria_colony_days_to_limit (n : ℕ) (h : ∀ t : ℕ, t ≤ 21 → (2 ^ t = 2 * 2 ^ (t - 1))) : n = 22 :=
by
  sorry

end single_bacteria_colony_days_to_limit_l808_808741


namespace greatest_integer_gcd_24_eq_4_l808_808274

theorem greatest_integer_gcd_24_eq_4 : ∃ n < 200, n % 4 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0 ∧ n = 196 :=
begin
  sorry
end

end greatest_integer_gcd_24_eq_4_l808_808274


namespace length_RS_l808_808137

theorem length_RS (XZ ZW : ℝ) (RS_perpendicular_to_XW : RS ⊥ XW) (XW_diagonal: XW = 5*√5) 
                  (RZ_length : RZ = 2.5*√5) (ZS_length : ZS = 10*√5) :
    RS = 12.5*√5 :=
by sorry

end length_RS_l808_808137


namespace dogs_in_garden_l808_808673

theorem dogs_in_garden (D : ℕ) (ducks : ℕ) (total_feet : ℕ) (dogs_feet : ℕ) (ducks_feet : ℕ) 
  (h1 : ducks = 2) 
  (h2 : total_feet = 28)
  (h3 : dogs_feet = 4)
  (h4 : ducks_feet = 2) 
  (h_eq : dogs_feet * D + ducks_feet * ducks = total_feet) : 
  D = 6 := by
  sorry

end dogs_in_garden_l808_808673


namespace minimum_common_perimeter_exists_l808_808259

noncomputable def find_minimum_perimeter
  (a b x : ℕ) 
  (is_int_sided_triangle_1 : 2 * a + 20 * x = 2 * b + 25 * x)
  (is_int_sided_triangle_2 : 20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2))) 
  (base_ratio : 20 * 2 * (a - b) = 25 * 2 * (a - b)): ℕ :=
2 * a + 20 * (2 * (a - b))

-- The final goal should prove the minimum perimeter under the given conditions.
theorem minimum_common_perimeter_exists :
∃ (minimum_perimeter : ℕ), 
  (∀ (a b x : ℕ), 
    2 * a + 20 * x = 2 * b + 25 * x → 
    20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2)) → 
    20 * 2 * (a - b) = 25 * 2 * (a - b) → 
    minimum_perimeter = 2 * a + 20 * x) :=
sorry

end minimum_common_perimeter_exists_l808_808259


namespace store_income_l808_808554

def pencil_store_income (p_with_eraser_qty p_with_eraser_cost p_regular_qty p_regular_cost p_short_qty p_short_cost : ℕ → ℝ) : ℝ :=
  (p_with_eraser_qty * p_with_eraser_cost) + (p_regular_qty * p_regular_cost) + (p_short_qty * p_short_cost)

theorem store_income : 
  pencil_store_income 200 0.8 40 0.5 35 0.4 = 194 := 
by sorry

end store_income_l808_808554


namespace product_of_positive_integers_for_real_roots_l808_808386

theorem product_of_positive_integers_for_real_roots :
  let discriminant_pos (a b c : ℤ) := b^2 - 4 * a * c > 0 in
  ∏ i in (finset.range (15)).filter (λ c, discriminant_pos 10 24 c), c = 87178291200 := by
  sorry

end product_of_positive_integers_for_real_roots_l808_808386


namespace units_digit_of_power_l808_808403

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808403


namespace parabolaIntersection_l808_808316

-- Definitions related to the problem conditions
def basePlane : Plane := plane.mk [... some data ...]
def horizontalPlane : Plane := plane.mk [... some data ...]
def rightCircularCone : Cone := cone.mk basePlane [... some parameters ...]

-- Condition 1: Positioning of the cone's base
def coneBaseOnPlane {x1 x2 h : ℝ} (h > 0) : basePlane = (x_1, x_2, 0)

-- Condition 2: Horizontal plane height
def horizontalPlaneHeight {x1 x2 h : ℝ} (h > 0) : horizontalPlane = (x_1, x_2, 2 / 3 * h)

-- Condition 3: Parabola intersection with plane
def intersectionParabola (S : Plane) (cone : Cone) : IsParabola (S ∩ cone)

-- Condition 4: Directrix definition via intersection
def directrixDefinition (horizontalPlane : Plane) : Directrix :=
  horizontalPlane ∩ line.from_direction [... some data ...]

-- Theorem statement without proof
theorem parabolaIntersection 
    (cone : Cone)
    (basePlane = (x_1, x_2, 0))
    (horizontalPlaneHeight = (x_1, x_2, 2 / 3 * h))
    (S : Plane parallel to (x_1, x_2))
    (parabola := intersectionParabola S cone):
  ∃ focus directrix,
    IsParabolaIntersection cone horizontalPlane S parabola focus directrix := 
sorry

end parabolaIntersection_l808_808316


namespace original_sets_exist_l808_808902

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808902


namespace units_digit_7_pow_6_pow_5_l808_808425

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808425


namespace collinearity_iff_midpoint_l808_808916

-- Define an acute triangle ABC with AB < AC
variables {A B C D M J K O : Type} [Geometry ℝ A B C D M J K O]

-- Conditions for acute triangle, lengths, foot of altitude, and point M
variable (h_acute : AcuteTriangle A B C)
variable (h_AB_lt_AC : AB < AC)
variable (h_D_altitude : AltitudeFoot D A C B)
variable (h_M_on_BC : M ≠ D ∧ Collinear B C M)

-- Conditions for points J, K, M lying on a common perpendicular line to BC
variable (h_perpendicular : PerpendicularLine (lineThrough K J M) (lineThrough B C))

-- Definitions of circumcircles and intersection point O
variable (circumcircle_ABJ : Circle A B J)
variable (circumcircle_ACK : Circle A C K)
variable (h_intersection : O ∈ circumcircle_ABJ ∧ O ∈ circumcircle_ACK)

-- Prove that J, O, M are collinear if and only if M is the midpoint of BC
theorem collinearity_iff_midpoint :
  (Collinear J O M) ↔ (Midpoint B C M) :=
  by sorry

end collinearity_iff_midpoint_l808_808916


namespace james_marbles_left_l808_808154

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end james_marbles_left_l808_808154


namespace tavern_keeper_solution_is_32_l808_808777

noncomputable def tavern_problem : Prop :=
  let initial_wine := 121
  let initial_water := 41
  let barrel := 12
  let container2 := 2
  let container8 := 8
  ∃ (n : ℕ), n = 32 ∧ 
    (∀ methods,  methods = number_of_ways_to_mix initial_wine initial_water barrel container2 container8 → methods = n)

-- Auxiliary function describing the number of ways to mix the given amounts with given containers
noncomputable def number_of_ways_to_mix (wine water barrel c2 c8 : ℕ) : ℕ :=
  sorry

theorem tavern_keeper_solution_is_32 : tavern_problem :=
  sorry

end tavern_keeper_solution_is_32_l808_808777


namespace least_words_to_ensure_score_l808_808111

theorem least_words_to_ensure_score (total_words : ℕ) (misremember_rate : ℝ) (required_percentage : ℝ) :
  total_words = 600 → misremember_rate = 0.1 → required_percentage = 0.9 → 
  ∀ (x : ℕ), (0.9 * x / total_words ≥ required_percentage) → x ≥ 600 :=
by
  intros h1 h2 h3,
  sorry

end least_words_to_ensure_score_l808_808111


namespace function_increasing_on_interval_l808_808365

noncomputable def f (x : ℝ) : ℝ := log (-x^2 + 2*x)

theorem function_increasing_on_interval :
  (∀ x ∈ set.Ioo 0 1, 0 < -x^2 + 2*x) →
  (∀ x1 x2 ∈ set.Ioo 0 1, x1 < x2 → -x1^2 + 2*x1 < -x2^2 + 2*x2) →
  (∀ x1 x2 ∈ set.Ioo 0 1, x1 < x2 → f x1 < f x2) :=
by 
  sorry

end function_increasing_on_interval_l808_808365


namespace football_game_attendance_l808_808684

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l808_808684


namespace line_DE_midpoint_LM_l808_808689

variables {A B C K L M D E : Type*}

-- Define points and their properties
def is_not_isosceles (ABC : Triangle A B C) : Prop := 
  ¬ (A = B ∨ B = C ∨ A = C)

def incircle_touches (ABC : Triangle A B C) (K L M : Point) : Prop :=
  touches_incircle (ABC) (K, L, M)

def parallel_through_point (LM : Line) (P : Point) (KP : Point) : Point :=
  line_parallel_through_point (LM) (P) intersects KP

-- Main theorem
theorem line_DE_midpoint_LM {A B C K L M D E : Point}
  (h1: is_not_isosceles (Triangle.mk A B C))
  (h2: incircle_touches (Triangle.mk A B C) K L M)
  (h3: parallel_through_point (Line.mk L M) B D)
  (h4: parallel_through_point (Line.mk L M) C E):
  passes_through_midpoint (Line.mk D E) (Midpoint L M) :=
sorry

end line_DE_midpoint_LM_l808_808689


namespace chuck_bicycle_trip_l808_808001

theorem chuck_bicycle_trip (D : ℝ) (h1 : D / 16 + D / 24 = 3) : D = 28.80 :=
by
  sorry

end chuck_bicycle_trip_l808_808001


namespace range_of_m_l808_808857

-- Definition of the quadratic function
def quadratic_function (m x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + 1

-- Statement of the proof problem in Lean
theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, 0 ≤ x ∧ x ≤ 5 → quadratic_function m x ≥ quadratic_function m (x + 1)) ↔ m ≤ -8 :=
by
  sorry

end range_of_m_l808_808857


namespace count_divisibles_l808_808516

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l808_808516


namespace domain_of_function_l808_808222

theorem domain_of_function : 
  {x : ℝ | 0 < x ∧ 4 - x^2 > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end domain_of_function_l808_808222


namespace min_sum_prob_l808_808929

open Classical

variables {A B : Prop} {x y : ℕ}

theorem min_sum_prob (h_mx : Disjoint A B)
  (h_PA : 4 / x)
  (h_PB : 1 / y)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (h_sum : 4 / x + 1 / y = 1) :
  x + y ≥ 9 :=
by
  sorry

end min_sum_prob_l808_808929


namespace quadratic_axis_of_symmetry_is_one_l808_808653

noncomputable def quadratic_axis_of_symmetry (b c : ℝ) : ℝ :=
  (-b / (2 * 1))

theorem quadratic_axis_of_symmetry_is_one
  (b c : ℝ)
  (hA : (0:ℝ)^2 + b * 0 + c = 3)
  (hB : (2:ℝ)^2 + b * 2 + c = 3) :
  quadratic_axis_of_symmetry b c = 1 :=
by
  sorry

end quadratic_axis_of_symmetry_is_one_l808_808653


namespace student_passing_percentage_l808_808331

variable (marks_obtained failed_by max_marks : ℕ)

def passing_marks (marks_obtained failed_by : ℕ) : ℕ :=
  marks_obtained + failed_by

def percentage_needed (passing_marks max_marks : ℕ) : ℚ :=
  (passing_marks : ℚ) / (max_marks : ℚ) * 100

theorem student_passing_percentage
  (h1 : marks_obtained = 125)
  (h2 : failed_by = 40)
  (h3 : max_marks = 500) :
  percentage_needed (passing_marks marks_obtained failed_by) max_marks = 33 := by
  sorry

end student_passing_percentage_l808_808331


namespace choose_non_overlapping_sets_for_any_n_l808_808895

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808895


namespace inscribed_circle_radius_l808_808662

theorem inscribed_circle_radius (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 9) : 
  let r := 18 / 17 in
  (1 / r) = (1 / a) + (1 / b) + (1 / c) + 2 * sqrt((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) :=
by
  sorry

end inscribed_circle_radius_l808_808662


namespace non_overlapping_sets_l808_808865

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808865


namespace sqrt_sums_eq_7_8_l808_808637

noncomputable theory

open Real

theorem sqrt_sums_eq_7_8 (y : ℝ) (h : sqrt (64 - y^2) * sqrt (36 - y^2) = 12) :
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7.8 :=
  sorry

end sqrt_sums_eq_7_8_l808_808637


namespace kolya_wins_game_l808_808301

theorem kolya_wins_game (initial_pile : ℕ) (H1 : initial_pile = 100) 
  (turns : ℕ → ℕ × ℕ)
  (H2 : ∀ n. ∀ p1 p2. turns n = (p1, p2) -> p1 + p2 = initial_pile 
    ∨ turns n = -1 -> (p1 = 1 ∧ p2 = 1))
  : ∃ strategy, ∀ moves, (∃ k, moves k = -1) -> strategy = "Kolya wins" :=
sorry

end kolya_wins_game_l808_808301


namespace sixth_distance_l808_808470

theorem sixth_distance (A B C D : Point)
  (dist_AB dist_AC dist_BC dist_AD dist_BD dist_CD : ℝ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_lengths : (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BC = 1 ∧ dist_AD = 1) ∨
               (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AB = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_CD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1))
  (h_one_point_two : dist_AB = 1.2 ∨ dist_AC = 1.2 ∨ dist_BC = 1.2 ∨ dist_AD = 1.2 ∨ dist_BD = 1.2 ∨ dist_CD = 1.2) :
  dist_AB = 1.84 ∨ dist_AB = 0.24 ∨ dist_AB = 1.6 ∨
  dist_AC = 1.84 ∨ dist_AC = 0.24 ∨ dist_AC = 1.6 ∨
  dist_BC = 1.84 ∨ dist_BC = 0.24 ∨ dist_BC = 1.6 ∨
  dist_AD = 1.84 ∨ dist_AD = 0.24 ∨ dist_AD = 1.6 ∨
  dist_BD = 1.84 ∨ dist_BD = 0.24 ∨ dist_BD = 1.6 ∨
  dist_CD = 1.84 ∨ dist_CD = 0.24 ∨ dist_CD = 1.6 :=
sorry

end sixth_distance_l808_808470


namespace sufficient_but_not_necessary_l808_808458

noncomputable def is_sufficient_but_not_necessary (a : ℝ) : Prop :=
  let z := complex.mk (a + 1) (-a)
  in complex.abs z = 1 ∧ ∃ b : ℝ, (complex.abs (complex.mk (b + 1) (-b)) = 1 ∧ b ≠ -1)

theorem sufficient_but_not_necessary (a : ℝ) (h : a = -1) : is_sufficient_but_not_necessary a := 
sorry

end sufficient_but_not_necessary_l808_808458


namespace quadratic_axis_of_symmetry_is_one_l808_808652

noncomputable def quadratic_axis_of_symmetry (b c : ℝ) : ℝ :=
  (-b / (2 * 1))

theorem quadratic_axis_of_symmetry_is_one
  (b c : ℝ)
  (hA : (0:ℝ)^2 + b * 0 + c = 3)
  (hB : (2:ℝ)^2 + b * 2 + c = 3) :
  quadratic_axis_of_symmetry b c = 1 :=
by
  sorry

end quadratic_axis_of_symmetry_is_one_l808_808652


namespace desired_overall_average_l808_808330

theorem desired_overall_average (P1 P2 P3 : ℕ) (h1 : P1 = 72) (h2 : P2 = 84) (h3 : P3 = 69) : 
  (P1 + P2 + P3) / 3 = 75 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end desired_overall_average_l808_808330


namespace evaluate_f_of_f_of_3_l808_808980

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem evaluate_f_of_f_of_3 :
  f (f 3) = 2943 :=
by
  sorry

end evaluate_f_of_f_of_3_l808_808980


namespace PR_diagonal_length_l808_808136

noncomputable def PR_length
  (PQ QR RS SP : ℝ) 
  (angle_SPR : ℝ)
  (hPQ : PQ = 12) 
  (hQR : QR = 10) 
  (hRS : RS = 20) 
  (hSP : SP = 20) 
  (h_angle_SPR : angle_SPR = 45) : ℝ :=
  20 * Real.sqrt(2 + Real.sqrt(2))

theorem PR_diagonal_length
  (PQ QR RS SP : ℝ) 
  (angle_SPR : ℝ)
  (hPQ : PQ = 12) 
  (hQR : QR = 10) 
  (hRS : RS = 20) 
  (hSP : SP = 20) 
  (h_angle_SPR : angle_SPR = 45) : 
  PR_length PQ QR RS SP angle_SPR hPQ hQR hRS hSP h_angle_SPR = 20 * Real.sqrt(2 + Real.sqrt(2)) := 
by
  sorry

end PR_diagonal_length_l808_808136


namespace min_value_of_f_monotonicity_of_F_ge_monotonicity_of_F_lt_l808_808854
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
theorem min_value_of_f : ∃ x : ℝ, x > 0 ∧ f x = -1 / Real.exp 1 := sorry

def F (x : ℝ) (a : ℝ) : ℝ := a * x^2 + (1 + Real.log x)
noncomputable def F' (x : ℝ) (a : ℝ) : ℝ := (2 * a * x + 1) / x

theorem monotonicity_of_F_ge (a : ℝ) (h : a ≥ 0) : 
  ∀ x : ℝ, x > 0 → F' x a > 0 := sorry

theorem monotonicity_of_F_lt (a : ℝ) (h : a < 0) :
  ∀ x : ℝ, x > 0 → (F' x a > 0 ↔ x < Real.sqrt (-1 / (2 * a))) ∧
  (F' x a < 0 ↔ x > Real.sqrt (-1 / (2 * a))) := sorry

end min_value_of_f_monotonicity_of_F_ge_monotonicity_of_F_lt_l808_808854


namespace cos_neg_60_eq_half_l808_808350

noncomputable def cos_neg_60 : ℝ :=
  cos (real.pi * (-60) / 180)

theorem cos_neg_60_eq_half : cos_neg_60 = 1 / 2 :=
by
  sorry

end cos_neg_60_eq_half_l808_808350


namespace sqrt_sqrt4_of_decimal_l808_808345

theorem sqrt_sqrt4_of_decimal (h : 0.000625 = 625 / (10 ^ 6)) :
  Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 625) / 1000)) = 0.4 :=
by
  sorry

end sqrt_sqrt4_of_decimal_l808_808345


namespace center_of_mass_equiv_l808_808621

noncomputable def center_of_mass (points : List (V × ℝ)) : V :=
  let total_mass := points.foldl (λ acc p, acc + p.2) 0
  (points.foldl (λ acc p, acc + (p.2 / total_mass) • p.1) 0)

theorem center_of_mass_equiv 
  (X Y : V) 
  (a b : ℝ) 
  (X_masses : List (V × ℝ))
  (Y_masses : List (V × ℝ)) 
  (hX_mass : a = X_masses.foldl (λ acc p, acc + p.2) 0)
  (hY_mass : b = Y_masses.foldl (λ acc p, acc + p.2) 0)
  (hX_com : X = center_of_mass X_masses)
  (hY_com : Y = center_of_mass Y_masses) :
  let combined_masses := X_masses ++ Y_masses in
  center_of_mass combined_masses =
  center_of_mass [(X, a), (Y, b)] :=
by
  sorry

end center_of_mass_equiv_l808_808621


namespace train_time_calculation_l808_808109

-- Definitions based on conditions
def length_of_train : ℝ := 450
def speed_of_train_kmh : ℝ := 120
def speed_of_train_mps : ℝ := speed_of_train_kmh * (1000 / 3600) -- Convert km/hr to m/s

-- Correct Answer
def time_to_pass_pole : ℝ := 13.5

theorem train_time_calculation :
  length_of_train / speed_of_train_mps = time_to_pass_pole :=
by
  sorry

end train_time_calculation_l808_808109


namespace expression_equals_10_4_l808_808846

def given_expression : ℝ :=
  let x1 := 6.5
  let x2 := 2 / 3
  let x3 := 2
  let x4 := 8.4
  ⌊x1⌋ * ⌊x2⌋ + ⌊x3⌋ * 7.2 + ⌊x4⌋ - 6.0

theorem expression_equals_10_4 : given_expression = 10.4 :=
by 
  sorry

end expression_equals_10_4_l808_808846


namespace part1_part2_l808_808098

theorem part1 (k : ℝ) : (∃ x : ℝ, x^2 - 4*x + 2*k = 0) ↔ k ≤ 2 :=
begin
  sorry
end

theorem part2 (m k x : ℝ) (hk : k = 2) (hx : x = 2) 
(h_root : x^2 - 4*x + 2*k = 0)
(h_common_root : x^2 - 2*m*x + 3*m - 1 = 0) : 
(m = 3 ∧ ∃ y : ℝ, y ≠ x ∧ y^2 - 2*m*y + 3*m - 1 = 0 := 
begin
  sorry
end

end part1_part2_l808_808098


namespace problem_statement_l808_808291

open Real

-- Define the base values x and y
def x := 0.6
def y := 0.8

-- Define the exponents
def a := 0.8
def b := 0.6

theorem problem_statement :
  (x^a < y^a ∧ x^a < y^b ∧ log y x > log x y) ∧ ¬(log y x < y^b) :=
by
  sorry

end problem_statement_l808_808291


namespace unique_shell_arrangements_l808_808155

theorem unique_shell_arrangements : 
  let shells := 12
  let symmetry_ops := 12
  let total_arrangements := Nat.factorial shells
  let distinct_arrangements := total_arrangements / symmetry_ops
  distinct_arrangements = 39916800 := by
  sorry

end unique_shell_arrangements_l808_808155


namespace estimate_population_l808_808764

noncomputable def fish_population_estimate : ℝ := 1312

theorem estimate_population : 
  ∀ (P : ℝ)
    (initial_tagged : ℝ)
    (june_sample : ℝ)
    (tagged_june : ℝ)
    (percent_left_area : ℝ)
    (percent_juveniles : ℝ),
  initial_tagged = 100 ->
  june_sample = 150 ->
  tagged_june = 4 ->
  percent_left_area = 0.3 ->
  percent_juveniles = 0.5 ->
  let old_fish_june_sample := june_sample * (1 - percent_juveniles) in
  let remaining_tagged := initial_tagged * (1 - percent_left_area) in
  let proportion_tagged := tagged_june / old_fish_june_sample in
  let population := remaining_tagged / proportion_tagged in
  P = population ->
  P = fish_population_estimate := 
by {
  sorry
}

end estimate_population_l808_808764


namespace alpha_correct_l808_808850

open Real

noncomputable def alpha (a : Real) : Prop :=
  a ∈ Ioo 0 (2 * π) ∧ sin (π / 6) = 1 / 2 ∧ cos (5 * π / 6) = - (sqrt 3 / 2) → a = 5 * π / 3

theorem alpha_correct (a : Real) (h : alpha a) : a = 5 * π / 3 :=
by
  sorry

end alpha_correct_l808_808850


namespace number_of_correct_statements_l808_808228

theorem number_of_correct_statements : 
  (∀ x : ℝ, ¬is_rat x ↔ ∃ y : ℕ, (x = y * (1/y))) → 
  (∀ x : ℚ, ∃ y : ℝ, x = y) → 
  (∀ x : ℝ, (x^2 = x) → (x = 0 ∨ x = 1)) → 
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = 2*x + 1 - x^2) → 
  false :=
by
  intro h1 h2 h3 h4
  sorry

end number_of_correct_statements_l808_808228


namespace aria_spent_on_cookies_in_march_l808_808844

/-- Aria purchased 4 cookies each day for the entire month of March,
    each cookie costs 19 dollars, and March has 31 days.
    Prove that the total amount Aria spent on cookies in March is 2356 dollars. -/
theorem aria_spent_on_cookies_in_march :
  (4 * 31) * 19 = 2356 := 
by 
  sorry

end aria_spent_on_cookies_in_march_l808_808844


namespace sum_binom_ineq_l808_808174

noncomputable theory

open_locale big_operators

theorem sum_binom_ineq (n : ℕ) (hn : n ≥ 2) :
  ∑ k in finset.range(n), (3 / 8) ^ k * (n.choose k) / ((n - 1).choose k) < 1 :=
sorry

end sum_binom_ineq_l808_808174


namespace profit_percentage_is_20_625_l808_808771

-- Definitions based on the conditions:
def cost_price_75_pens : ℝ := 60
def selling_price_per_pen_before_discount : ℝ := 1
def discount_per_pen : ℝ := 0.035

-- The theorem to prove the profit percentage is 20.625%
theorem profit_percentage_is_20_625 :
  let cp := cost_price_75_pens,
      sp_before_discount := selling_price_per_pen_before_discount,
      discount := discount_per_pen,
      sp_after_discount := sp_before_discount - discount,
      total_sp := sp_after_discount * 75,
      profit := total_sp - cp,
      profit_percentage := (profit / cp) * 100
  in profit_percentage = 20.625 := 
sorry

end profit_percentage_is_20_625_l808_808771


namespace units_digit_pow_7_6_5_l808_808414

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808414


namespace trig_inequalities_2013_l808_808813

theorem trig_inequalities_2013 :
  let θ := 2013 * real.pi / 180
  let x := 33 * real.pi / 180
  sin θ < 0 ∧ cos θ < 0 ∧ tan θ > 0 ∧ sin θ = - sin x ∧ cos θ = - cos x →
  tan θ > sin θ ∧ sin θ > cos θ :=
by intros θ x h sorry

end trig_inequalities_2013_l808_808813


namespace units_digit_of_power_l808_808404

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808404


namespace devin_basketball_chances_l808_808642

theorem devin_basketball_chances 
  (initial_chances : ℝ := 0.1) 
  (base_height : ℕ := 66) 
  (chance_increase_per_inch : ℝ := 0.1)
  (initial_height : ℕ := 65) 
  (growth : ℕ := 3) :
  initial_chances + (growth + initial_height - base_height) * chance_increase_per_inch = 0.3 := 
by 
  sorry

end devin_basketball_chances_l808_808642


namespace largest_possible_perimeter_l808_808780

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 11) : 
    5 + 6 + x ≤ 21 := 
  sorry

end largest_possible_perimeter_l808_808780


namespace number_of_lists_proof_l808_808787

noncomputable def number_of_lists_possible : ℕ :=
  11

theorem number_of_lists_proof :
  ∃ n : ℕ, (∃ a b c d e : ℕ, a + b + c + d + e = 6 ∧ a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e) ∧ n = number_of_lists_possible :=
begin
  -- The proof would go here
  sorry
end

end number_of_lists_proof_l808_808787


namespace arithmetic_sequence_sum_geometric_arithmetic_sum_l808_808917

-- Declare the problem involving an arithmetic sequence {a_n} 
-- where the sum of the first n terms is S_n
theorem arithmetic_sequence_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 4 = 24)
  (h2 : S 7 = 63)
  (h3 : ∀ n, S n = n * (a_n 1 + a_n n) / 2) :
  (∀ n, a_n n = 2 * n + 1) := 
sorry

-- Declare the problem involving the sequence {b_n} and its sum T_n
theorem geometric_arithmetic_sum (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : S 4 = 24)
  (h2 : S 7 = 63)
  (h3 : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (h4 : ∀ n, a_n n = 2 * n + 1)
  (h5 : ∀ n, b_n n = 2^(a_n n) + a_n n):
  (∀ n, T n = (8 / 3) * (1 - 4^n) + n^2 + 2 * n) := 
sorry

end arithmetic_sequence_sum_geometric_arithmetic_sum_l808_808917


namespace limit_of_sequence_l808_808112

noncomputable theory
open_locale classical

variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Conditions
def recurrence_relation (a : ℕ → ℝ) : Prop := ∀ n ≥ 1, (2 - a n) * a (n + 1) = 1

-- Statement
theorem limit_of_sequence (h : recurrence_relation a) : filter.tendsto a filter.at_top (nhds 1) :=
sorry

end limit_of_sequence_l808_808112


namespace girls_eq_barefoot_l808_808671

-- Let's define the number of barefoot boys (B_b)
variable (B_b : ℕ)

-- Define the number of girls with shoes (G_s)
variable (G_s : ℕ)

-- Define the number of barefoot girls (G_b)
variable (G_b : ℕ)

-- Given condition: B_b = G_s
axiom B_b_eq_G_s : B_b = G_s

-- Calculate the total number of girls (G)
def total_girls (G_b G_s : ℕ) : ℕ := G_b + G_s

-- Calculate the total number of barefoot children (B)
def total_barefoot (B_b G_b : ℕ) : ℕ := B_b + G_b

-- The goal is to prove total_girls = total_barefoot
theorem girls_eq_barefoot :
  total_girls G_b G_s = total_barefoot B_b G_b :=
by 
  rw [total_girls, total_barefoot, B_b_eq_G_s];
  exact eq.refl (G_b + G_s)

end girls_eq_barefoot_l808_808671


namespace minimum_common_perimeter_l808_808264

theorem minimum_common_perimeter :
  ∃ (a b c : ℕ), 
  let p := 2 * a + 10 * c in
  (a > b) ∧ 
  (b + 4c = a + 5c) ∧
  (5 * (a^2 - (5 * c)^2).sqrt = 4 * (b^2 - (4 * c)^2).sqrt) ∧
  p = 1180 :=
sorry

end minimum_common_perimeter_l808_808264


namespace pencil_count_l808_808194

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l808_808194


namespace other_number_is_150_l808_808719

-- Definitions of LCM and GCD (HCF)
def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Assuming the given conditions
variables (A B : ℕ)
  (h_lcm : lcm A B = 2310)
  (h_gcd : gcd A B = 30)
  (h_A : A = 462)

-- The proof we seek
theorem other_number_is_150 :
  B = 150 :=
sorry

end other_number_is_150_l808_808719


namespace find_k_domain_g_range_a_l808_808494

def f (x : ℝ) (k : ℝ) : ℝ := log (4 ^ x + 1) / log 4 + k * x
def g (x : ℝ) (a : ℝ) : ℝ := log (a * 2 ^ x - (4 / 3) * a) / log 4

theorem find_k (k : ℝ) : (∀ x : ℝ, f x k = f (-x) k) ↔ k = -1 / 2 := sorry

theorem domain_g (a : ℝ) :
  (a > 0 → ∀ x : ℝ, g x a = g x a → x ∈ set.Ioi (log (4 / 3) / log 2)) ∧
  (a < 0 → ∀ x : ℝ, g x a = g x a → x ∈ set.Iio (log (4 / 3) / log 2)) :=
sorry

theorem range_a (a : ℝ) :
  (∃ x : ℝ, f x (-1 / 2) = g x a) → (a > 1 ∨ a = -3) :=
sorry

end find_k_domain_g_range_a_l808_808494


namespace find_velocity_l808_808236

variable (k V : ℝ)
variable (P A : ℕ)

theorem find_velocity (k_eq : k = 1 / 200) 
  (initial_cond : P = 4 ∧ A = 2 ∧ V = 20) 
  (new_cond : P = 16 ∧ A = 4) : 
  V = 20 * Real.sqrt 2 :=
by
  sorry

end find_velocity_l808_808236


namespace original_sets_exist_l808_808908

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808908


namespace order_of_exponents_l808_808054

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem order_of_exponents : a > b ∧ b > c :=
by
  sorry

end order_of_exponents_l808_808054


namespace tetrahedron_ineq_l808_808555

variable (P Q R S : ℝ)

-- Given conditions
axiom ortho_condition : S^2 = P^2 + Q^2 + R^2

theorem tetrahedron_ineq (P Q R S : ℝ) (ortho_condition : S^2 = P^2 + Q^2 + R^2) :
  (P + Q + R) / S ≤ Real.sqrt 3 := by
  sorry

end tetrahedron_ineq_l808_808555


namespace min_distance_ln_curve_l808_808210

/-- Given M on the curve y = ln(x) -/
def M (x : ℝ) : ℝ × ℝ := (x, Real.log x)

/-- Given N on the line y = 2x + 6 -/
def N (x : ℝ) : ℝ × ℝ := (x, 2*x + 6)

theorem min_distance_ln_curve
  (x0 : ℝ) (hx0 : x0 = 1/2) (hx1 : 2 = 1/x0) :
  let M0 := M x0,
      N0 := N x0,
      dist := |2 * (x0) + Real.log 2 + 6| / Real.sqrt (4 + 1)
  in dist = (7 + Real.log 2) * Real.sqrt 5 / 5 := 
by
  sorry

end min_distance_ln_curve_l808_808210


namespace f_neg_1_eq_0_f_eq_f_neg_x_minus_6_f_7_eq_0_l808_808932

variable {ℝ : Type*} [LinearOrderedField ℝ]

/-- The domain of the function f(x) is ℝ. --/
axiom dom_f : ∀ x : ℝ, f x ∈ ℝ

/-- f(x - 3) is an even function. --/
axiom even_f : ∀ x : ℝ, f (x - 3) = f (- (x - 3))

/-- f(2x - 1) is an odd function. --/
axiom odd_f : ∀ x : ℝ, f (2x - 1) = -f (-(2x - 1))

/-- Prove that f(-1) = 0. --/
theorem f_neg_1_eq_0 : f (-1) = 0 :=
sorry

/-- Prove that f(x) = f(-x - 6). --/
theorem f_eq_f_neg_x_minus_6 : ∀ x : ℝ, f x = f (-x - 6) :=
sorry

/-- Given f(-1) = 0 and f(x) = f(-x - 6), prove that f(7) = 0. --/
theorem f_7_eq_0 : f (-1) = 0 → (∀ x : ℝ, f x = f (-x - 6)) → f 7 = 0 :=
sorry

end f_neg_1_eq_0_f_eq_f_neg_x_minus_6_f_7_eq_0_l808_808932


namespace max_expr_under_condition_l808_808981

-- Define the conditions and variables
variable {x : ℝ}

-- State the theorem about the maximum value of the given expression under the given condition
theorem max_expr_under_condition (h : x < -3) : 
  ∃ M, M = -2 * Real.sqrt 2 - 3 ∧ ∀ y, y < -3 → y + 2 / (y + 3) ≤ M :=
sorry

end max_expr_under_condition_l808_808981


namespace repeated_two_digit_number_divisible_by_101_l808_808774

theorem repeated_two_digit_number_divisible_by_101 (a b : ℕ) (h1 : 0 ≤ a) (h2 : a < 10) (h3 : 0 ≤ b) (h4 : b < 10) :
  let n := 10101 * (10 * a + b) in n % 101 = 0 :=
by 
  have h : n = 10101 * (10 * a + b), by {reflexivity},
  sorry

end repeated_two_digit_number_divisible_by_101_l808_808774


namespace divisors_ending_in_3_l808_808035

theorem divisors_ending_in_3 {n : ℕ} :
  let S_n := {d ∈ List.factors n | d % 10 = 3} in
  S_n.card ≤ (List.factors n).card / 2 :=
by
  sorry

end divisors_ending_in_3_l808_808035


namespace limit_of_f_at_4_l808_808806

noncomputable def f (x : ℝ) : ℝ :=
  (real.cbrt (16 * x) - 4) / (real.sqrt (4 + x) - real.sqrt (2 * x))

theorem limit_of_f_at_4 : 
  filter.tendsto f (nhds 4) (nhds (- (4 * real.sqrt 2) / 3)) :=
sorry

end limit_of_f_at_4_l808_808806


namespace distance_between_foci_of_ellipse_l808_808008

-- Define the problem parameters
def semi_major_axis : ℝ := 10
def semi_minor_axis : ℝ := 4
def focal_distance_from_center (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Define the theorem statement
theorem distance_between_foci_of_ellipse :
  2 * focal_distance_from_center semi_major_axis semi_minor_axis = 4 * Real.sqrt 21 :=
by
  sorry

end distance_between_foci_of_ellipse_l808_808008


namespace trig_equation_solution_l808_808710

noncomputable def solveTrigEquation (t k n : ℤ) : Prop :=
  2 * (sin t)^4 * (sin (2 * t) - 3) - 2 * (sin t)^2 * (sin (2 * t) - 3) - 1 = 0 →
  (t = (π / 4) * (4 * k + 1) ∨ t = (-1)^n * (1 / 2 * arcsin (1 - sqrt 3)) + (π * n) / 2)

-- Proof is omitted here, indicated by sorry.
theorem trig_equation_solution (t k n : ℤ) : solveTrigEquation t k n := sorry

end trig_equation_solution_l808_808710


namespace difference_of_units_digits_l808_808481

def is_positive_even (n : ℕ) : Prop := n % 2 = 0 ∧ n > 0
def has_positive_units_digit (n : ℕ) : Prop := (n % 10) ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
def units_digit (n : ℕ) : ℕ := n % 10 

theorem difference_of_units_digits (p : ℕ) 
  (h1 : is_positive_even p) 
  (h2 : has_positive_units_digit p) 
  (h3 : units_digit (p + 4) = 0) 
  : units_digit (p ^ 3) - units_digit (p ^ 2) = 0 :=
by
  sorry

end difference_of_units_digits_l808_808481


namespace arith_seq_general_term_geom_seq_sum_l808_808726

-- Arithmetic Sequence Problem
theorem arith_seq_general_term (S : ℕ → ℝ) (a d : ℝ):
  S 10 = 50 → S 20 = 300 → ∀ n : ℕ, (n : ℝ) * (2 * 1) / 2 + d = (2n - 6) := 
by intros S10_eq_50 S20_eq_300 n
  sorry

-- Geometric Sequence Problem
theorem geom_seq_sum (S a : ℕ → ℝ) (q : ℝ):
  S 3 = a 2 + 10 * a 1 → a 5 = 81 → ∀ n : ℕ, S n = (1/2) * (3^n - 1) :=
by intros S3_eq a5_eq n
  sorry

end arith_seq_general_term_geom_seq_sum_l808_808726


namespace range_of_quadratic_function_l808_808368

theorem range_of_quadratic_function :
  range (λ x : ℝ, x^2 - 2*x - 3) = set.Icc (-4 : ℝ) 0 :=
sorry

end range_of_quadratic_function_l808_808368


namespace correct_option_is_A_l808_808646

def a (n : ℕ) : ℤ :=
  match n with
  | 1 => -3
  | 2 => 7
  | _ => 0  -- This is just a placeholder for other values

def optionA (n : ℕ) : ℤ := (-1)^n * (4*n - 1)
def optionB (n : ℕ) : ℤ := (-1)^n * (4*n + 1)
def optionC (n : ℕ) : ℤ := 4*n - 7
def optionD (n : ℕ) : ℤ := (-1)^(n + 1) * (4*n - 1)

theorem correct_option_is_A :
  (a 1 = -3) ∧ (a 2 = 7) ∧
  (optionA 1 = -3 ∧ optionA 2 = 7) ∧
  ¬(optionB 1 = -3 ∧ optionB 2 = 7) ∧
  ¬(optionC 1 = -3 ∧ optionC 2 = 7) ∧
  ¬(optionD 1 = -3 ∧ optionD 2 = 7) :=
by
  sorry

end correct_option_is_A_l808_808646


namespace range_of_a_for_p_range_of_a_for_p_and_q_l808_808071

variable (a : ℝ)

/-- For any x ∈ ℝ, ax^2 - x + 3 > 0 if and only if a > 1/12 -/
def condition_p : Prop := ∀ x : ℝ, a * x^2 - x + 3 > 0

/-- There exists x ∈ [1, 2] such that 2^x * a ≥ 1 -/
def condition_q : Prop := ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a * 2^x ≥ 1

/-- Theorem (1): The range of values for a such that condition_p holds true is (1/12, +∞) -/
theorem range_of_a_for_p (h : condition_p a) : a > 1/12 :=
sorry

/-- Theorem (2): The range of values for a such that condition_p and condition_q have different truth values is (1/12, 1/4) -/
theorem range_of_a_for_p_and_q (h₁ : condition_p a) (h₂ : ¬condition_q a) : 1/12 < a ∧ a < 1/4 :=
sorry

end range_of_a_for_p_range_of_a_for_p_and_q_l808_808071


namespace count_legal_numbers_l808_808531

def is_legal (n : ℕ) (T : ℕ → ℕ) : Prop :=
  ∃ (s : finset ℕ), (∀ k ∈ s, k < n) ∧ (s.sum T = n)

def a_seq : ℕ → ℕ
| 0       := 1
| (k + 1) := k + 2 + a_seq k

theorem count_legal_numbers :
  ∃ m : ℕ, m = 2001 ∧ (∑ i in finset.range (m + 1), ite (is_legal i a_seq) 1 0) = 1995 :=
sorry

end count_legal_numbers_l808_808531


namespace smallest_prime_p_l808_808271

theorem smallest_prime_p 
  (p q r : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime q) 
  (h3 : r > 0) 
  (h4 : p + q = r) 
  (h5 : q < p) 
  (h6 : q = 2) 
  (h7 : Nat.Prime r)  
  : p = 3 := 
sorry

end smallest_prime_p_l808_808271


namespace inequality_proof_l808_808978

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end inequality_proof_l808_808978


namespace alex_singles_hits_percentage_l808_808130

theorem alex_singles_hits_percentage
    (total_hits : ℕ)
    (home_runs : ℕ)
    (triples : ℕ)
    (doubles : ℕ)
    (singles : ℕ)
    (percentage_singles : ℚ)
    (h_total_hits : total_hits = 50)
    (h_home_runs : home_runs = 2)
    (h_triples : triples = 3)
    (h_doubles : doubles = 10)
    (h_singles : singles = total_hits - (home_runs + triples + doubles))
    (h_percentage_singles : percentage_singles = (singles / total_hits) * 100) :
    singles = 35 ∧ percentage_singles = 70 :=
by
  split
  sorry
  sorry

end alex_singles_hits_percentage_l808_808130


namespace product_of_g_equals_evaluation_of_f_l808_808585

/-- Let y1, y2, y3, y4 be the roots of the polynomial f(y) = y^4 - y^3 + 2y - 1,
    and let g(y) = y^2 + y - 3. Prove that g(y1) * g(y2) * g(y3) * g(y4) 
    equals to the product of the evaluation of f(y) at the roots of g(y). -/
theorem product_of_g_equals_evaluation_of_f (y1 y2 y3 y4 : ℂ)
  (hy : (y - y1) * (y - y2) * (y - y3) * (y - y4) = y^4 - y^3 + 2 * y - 1)
  (hg : g = y^2 + y - 3) :
  g(y1) * g(y2) * g(y3) * g(y4) = f(((-1 + Complex.sqrt 13)/2)) * f(((-1 - Complex.sqrt 13)/2)) :=
sorry

end product_of_g_equals_evaluation_of_f_l808_808585


namespace find_angle_and_side_sum_l808_808992

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  ∃ A B C : ℝ, A + B + C = real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 + c^2 = 2 * (a * b * c)

theorem find_angle_and_side_sum
  (A B C a b c : ℝ)
  (h_triangle : triangle A B C a b c)
  (h_cos_ratio : (real.cos B) / (real.cos C) = b / (2 * a - c))
  (h_b : b = real.sqrt 7)
  (h_area : 1/2 * a * c * (real.sin (real.pi / 3)) = (3 * real.sqrt 3) / 2) :
  B = real.pi / 3 ∧ a + c = 5 :=
by
  sorry

end find_angle_and_side_sum_l808_808992


namespace minimum_club_count_l808_808545

-- Definitions representing the conditions
structure Club (α : Type) :=
  (members : Set α)
  (card_members : members.card = 3)

def intersects_exactly_one (clubs : List (Club α)) : Prop :=
  ∀ (C1 C2 : Club α) (h1 : C1 ∈ clubs) (h2 : C2 ∈ clubs), C1 ≠ C2 → 
    (C1.members ∩ C2.members).card = 1

def all_students_in_all_clubs (clubs : List (Club α)) :=
  ∃ (s : α), ∀ (C : Club α), C ∈ clubs → s ∈ C.members

-- Problem statement: Prove minimum n such that all_students_in_all_clubs is true for any set satisfying conditions.
theorem minimum_club_count (n : ℕ) (h : n = 8) :
  ∀ (α : Type) (clubs : List (Club α)), 
    clubs.length = n → intersects_exactly_one clubs → all_students_in_all_clubs clubs :=
by sorry

end minimum_club_count_l808_808545


namespace probability_of_vowel_initials_l808_808602

def students_26_initials_distinct : Prop :=
  ∀ (students : Fin 26 → Char × Char), function.Injective students

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

theorem probability_of_vowel_initials : 
  (∀ (students : Fin 26 → Char × Char), students_26_initials_distinct students) →
  (let vowels := ['A', 'E', 'I', 'O', 'U', 'Y'] in
   let num_vowels := vowels.length in
   let total_initials := 26 in
   (num_vowels : ℚ) / total_initials = 3 / 13) :=
by
  sorry

end probability_of_vowel_initials_l808_808602


namespace nums_between_2000_and_3000_div_by_360_l808_808523

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l808_808523


namespace farmer_total_acres_l808_808754

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l808_808754


namespace selling_price_is_correct_l808_808321

variable (milk_price_per_liter : ℝ)
variable (water_percentage : ℝ)
variable (gain_percentage : ℝ)
variable (final_selling_price : ℝ)

def total_volume_of_mixture (milk_volume : ℝ) : ℝ :=
  milk_volume + milk_volume * (water_percentage / 100)

def selling_price_with_gain (cost_price : ℝ) : ℝ :=
  cost_price * (1 + gain_percentage / 100)

def selling_price_per_liter (total_selling_price : ℝ) (total_volume : ℝ) : ℝ :=
  total_selling_price / total_volume

noncomputable def final_selling_price_per_liter
    (milk_price_per_liter water_percentage gain_percentage : ℝ) : ℝ :=
  selling_price_per_liter (selling_price_with_gain milk_price_per_liter) (total_volume_of_mixture 1)

theorem selling_price_is_correct :
  final_selling_price_per_liter milk_price_per_liter water_percentage gain_percentage = final_selling_price :=
by
  sorry

def given_conditions : Prop :=
  milk_price_per_liter = 12 ∧ water_percentage = 20 ∧ gain_percentage = 50 ∧ final_selling_price = 15

example : given_conditions → selling_price_is_correct :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  rw [h1, h2, h3, h4]
  exact sorry

end selling_price_is_correct_l808_808321


namespace regular_pentagon_area_l808_808525

noncomputable def area_of_regular_pentagon (s : ℝ) : ℝ :=
  (5 * s^2) / (4 * Real.tan (Real.pi / 5))

theorem regular_pentagon_area (s : ℝ) (h : s = 18) : area_of_regular_pentagon s ≈ 558 :=
by
  rw [h]
  have h_tan: Real.tan (Real.pi / 5) ≈ 0.726543 := sorry
  rw [area_of_regular_pentagon]
  sorry

end regular_pentagon_area_l808_808525


namespace fibonacci_mod_100_l808_808214

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fibonacci_mod_100 :
  fib 100 % 5 = 0 :=
by sorry

end fibonacci_mod_100_l808_808214


namespace floor_neg_4_7_l808_808829

theorem floor_neg_4_7 : int.floor (-4.7) = -5 := sorry

end floor_neg_4_7_l808_808829


namespace intersection_of_cylinders_within_sphere_l808_808622

theorem intersection_of_cylinders_within_sphere (a b c d e f : ℝ) :
    ∀ (x y z : ℝ), 
      (x - a)^2 + (y - b)^2 < 1 ∧ 
      (y - c)^2 + (z - d)^2 < 1 ∧ 
      (z - e)^2 + (x - f)^2 < 1 → 
      (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 :=
by
  sorry

end intersection_of_cylinders_within_sphere_l808_808622


namespace log_graph_translation_l808_808647

variables (a : ℝ) (x y : ℝ)

def log_graph (a : ℝ) (x : ℝ) : ℝ := log x / log a

theorem log_graph_translation (h1 : a > 0) (h2 : a ≠ 1) : 
  log_graph a (2 - 1) + 2 = 2 :=
by
  sorry

end log_graph_translation_l808_808647


namespace conditions_implying_d_one_or_neg_one_l808_808026

variable (d x y n : ℕ)
variable (h₀ : d ≠ 0)
variable (sq_f_d : ¬ ∃ b, b > 1 ∧ b * b = d)
variable (pos_x : x > 0)
variable (pos_y : y > 0)
variable (pos_n : n > 0)
variable (hx : x ^ 2 + d * y ^ 2 = 2 ^ n)

theorem conditions_implying_d_one_or_neg_one 
  (h₀ : d ≠ 0)
  (sq_f_d : ¬ ∃ b, b > 1 ∧ b * b = d)
  (pos_x : x > 0)
  (pos_y : y > 0)
  (pos_n : n > 0)
  (hx : x ^ 2 + d * y ^ 2 = 2 ^ n) :
  d = 1 ∨ d = -1 := 
sorry

end conditions_implying_d_one_or_neg_one_l808_808026


namespace probability_three_packages_correct_l808_808034

def probability_three_correct_deliveries 
  (n : ℕ) (k : ℕ) : ℚ :=
  if n = 5 ∧ k = 3 then
    ((Nat.choose 5 3 : ℚ) * (1 / 60))
  else 0

theorem probability_three_packages_correct :
  probability_three_correct_deliveries 5 3 = 1 / 6 :=
by
  sorry

end probability_three_packages_correct_l808_808034


namespace convex_function_m_range_l808_808589

theorem convex_function_m_range :
  (∀ x ∈ Ioo (1 : ℝ) 3, (x^2 - m*x - 3) < 0) ↔ (m ≥ 2) := sorry

end convex_function_m_range_l808_808589


namespace range_of_a_l808_808492

theorem range_of_a (a : ℝ) (x x₀ : ℝ) (e : ℝ) (h_e_pos : 0 < e) (h_e_exp : e = Real.exp 1):
  (∀ x₀ ∈ Ioo 0 e, ∃ x1 x2 ∈ Ioo 0 e, x1 ≠ x2 ∧ ((2-a) * (x1 - 1) - 2 * Real.log x1) = (x₀ * Real.exp (1 - x₀)) ∧ ((2-a) * (x2 - 1) - 2 * Real.log x2) = (x₀ * Real.exp (1 - x₀))) ↔ a ∈ set.Iic ((2*e - 5) / (e - 1)) :=
sorry

end range_of_a_l808_808492


namespace sum_not_divisible_by_6_1_to_100_l808_808564

def sum_not_divisible_by_6 (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x, ¬(6 ∣ x)).sum id

theorem sum_not_divisible_by_6_1_to_100 : sum_not_divisible_by_6 100 = 4234 := sorry

end sum_not_divisible_by_6_1_to_100_l808_808564


namespace label_subsets_l808_808577

open Finset

theorem label_subsets (S : Finset α) :
  ∃ (A : List (Finset α)), 
    A.head = ∅ ∧ 
    (∀ (n : ℕ), n < A.length - 1 → 
      (A.get n ⊆ A.get (n + 1) ∧ (A.get (n + 1) \ A.get n).card = 1) ∨ 
      (A.get (n + 1) ⊆ A.get n ∧ (A.get n \ A.get (n + 1)).card = 1)) :=
sorry

end label_subsets_l808_808577


namespace radian_measure_of_central_angle_l808_808215

-- Given conditions
variables (l r : ℝ)
variables (h1 : (1 / 2) * l * r = 1)
variables (h2 : 2 * r + l = 4)

-- The theorem to prove
theorem radian_measure_of_central_angle (l r : ℝ) (h1 : (1 / 2) * l * r = 1) (h2 : 2 * r + l = 4) : 
  l / r = 2 :=
by 
  -- Proof steps are not provided as per the requirement
  sorry

end radian_measure_of_central_angle_l808_808215


namespace non_overlapping_sets_l808_808864

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808864


namespace number_of_multiples_in_range_l808_808509

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l808_808509


namespace range_of_x_l808_808591

noncomputable def f (x : ℝ) : ℝ := log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x : 
  ∃ x : ℝ, (1 / 4 < x ∧ x < 1 / 2) ∧ f(x) > f(3 * x - 1) :=
sorry

end range_of_x_l808_808591


namespace verify_trees_in_other_row_l808_808766

-- Definition of a normal lemon tree lemon production per year
def normalLemonTreeProduction : ℕ := 60

-- Definition of the percentage increase in lemon production for specially engineered lemon trees
def percentageIncrease : ℕ := 50

-- Definition of lemon production for specially engineered lemon trees
def specialLemonTreeProduction : ℕ := normalLemonTreeProduction * (1 + percentageIncrease / 100)

-- Number of trees in one row of the grove
def treesInOneRow : ℕ := 50

-- Total lemon production in 5 years
def totalLemonProduction : ℕ := 675000

-- Number of years
def years : ℕ := 5

-- Total number of trees in the grove
def totalNumberOfTrees : ℕ := totalLemonProduction / (specialLemonTreeProduction * years)

-- Number of trees in the other row
def treesInOtherRow : ℕ := totalNumberOfTrees - treesInOneRow

-- Theorem: Verification of the number of trees in the other row
theorem verify_trees_in_other_row : treesInOtherRow = 1450 :=
  by
  -- Proof logic is omitted, leaving as sorry
  sorry

end verify_trees_in_other_row_l808_808766


namespace number_of_multiples_in_range_l808_808511

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l808_808511


namespace length_of_BE_l808_808146

theorem length_of_BE (A B C D E: Type) 
  (triangle_ABC : △ A B C)
  (angle_A: ∠ A = 60°) 
  (angle_C: ∠ C = 90°)
  (length_AC: AC = 1)
  (D_on_BC: D ∈ segment B C)
  (E_on_AB: E ∈ segment A B)
  (triangle_ADE : △ A D E)
  (triangle_ADE_isosceles_right: is_isosceles_right △ A D E)
  (angle_ADE: ∠ A D E = 90°): 
  BE = 4 - 2 * sqrt 3 := 
  sorry

end length_of_BE_l808_808146


namespace positive_value_of_x_l808_808824

variables (a b : ℂ) (x : ℝ)

-- Given conditions
def magnitude_a : |a| = 3 := by sorry
def magnitude_b : |b| = 5 := by sorry
def ab_eq : a * b = x + 3 * complex.I := by sorry

-- Proof statement
theorem positive_value_of_x :
  x = 6 * real.sqrt 6 :=
by sorry

end positive_value_of_x_l808_808824


namespace PQRS_product_eq_one_l808_808922

noncomputable def P := Real.sqrt 2011 + Real.sqrt 2012
noncomputable def Q := -Real.sqrt 2011 - Real.sqrt 2012
noncomputable def R := Real.sqrt 2011 - Real.sqrt 2012
noncomputable def S := Real.sqrt 2012 - Real.sqrt 2011

theorem PQRS_product_eq_one : P * Q * R * S = 1 := by
  sorry

end PQRS_product_eq_one_l808_808922


namespace perpendicular_lines_solution_l808_808124

theorem perpendicular_lines_solution (a : ℝ) :
  (a : ℝ) → ∃ a : ℝ, 3 + a * (-(a - 2)) = 0 :=
begin
  sorry,
end

end perpendicular_lines_solution_l808_808124


namespace non_overlapping_sets_l808_808866

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808866


namespace find_distance_from_home_to_airport_l808_808012

variable (d t : ℝ)

-- Conditions
def condition1 := d = 40 * (t + 0.75)
def condition2 := d - 40 = 60 * (t - 1.25)

-- Proof statement
theorem find_distance_from_home_to_airport (hd : condition1 d t) (ht : condition2 d t) : d = 160 :=
by
  sorry

end find_distance_from_home_to_airport_l808_808012


namespace sum_of_digits_of_factorials_last_two_digits_l808_808808

theorem sum_of_digits_of_factorials_last_two_digits :
  let seq := [1!, 1!, 2!, 3!, 5!, 8!, 13!, 21!, 34!, 34!, 55!, 89!]
  let last_two_digits n := n % 100
  let last_two_dig_factorials := seq.map last_two_digits
  let total_last_two_digits := last_two_dig_factorials.sum
  let digits_sum := (total_last_two_digits % 10) + (total_last_two_digits / 10)
  digits_sum = 5 := by
  let seq := [1!, 1!, 2!, 3!, 5!, 8!, 13!, 21!, 34!, 34!, 55!, 89!]
  let last_two_digits := λ n, n % 100
  let last_two_dig_factorials := seq.map last_two_digits
  let total_last_two_digits := last_two_dig_factorials.sum
  let digits_sum := total_last_two_digits % 10 + total_last_two_digits / 10
  exact digits_sum = 5

end sum_of_digits_of_factorials_last_two_digits_l808_808808


namespace inequality_proof_l808_808979

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end inequality_proof_l808_808979


namespace intersection_of_sets_l808_808072

-- Define the set A
def set_A : Set ℝ := { y | ∃ x : ℝ, y = Real.log x / Real.log 2 ∧ 0 < x ∧ x < 4 }

-- Define the set B
def set_B : Set ℝ := { y | ∃ x : ℝ, y = 2^x ∧ 0 < x }

-- Statement we want to prove
theorem intersection_of_sets : set_A ∩ set_B = { y | 1 < y ∧ y < 2 } :=
by
  sorry

end intersection_of_sets_l808_808072


namespace sum_first_n_terms_b_n_arithmetic_seq_l808_808935

-- Define the arithmetic sequence properties
def arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_arithmetic_seq (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 0 + d * (n * (n - 1)) / 2

-- Given conditions
def a1 : ℝ := 1
def a2 : ℝ := 5
def d : ℝ := a2 - a1 -- common difference definition

axiom an_arithmetic_seq : arithmetic_seq (λ n, a1 + n * d) -- aliased the arithmetic sequence

-- Prove that S_n = 2n^2 - n
theorem sum_first_n_terms (n : ℕ) : sum_arithmetic_seq (λ n, a1 + n * d) n = 2 * n^2 - n :=
sorry

-- Define a new sequence b_n based on S_n
def b_n (n : ℕ) : ℝ := (sum_arithmetic_seq (λ n, a1 + n * d) n) / (n - 1/2)

-- Prove that the sequence {b_n} is arithmetic
theorem b_n_arithmetic_seq : arithmetic_seq b_n :=
sorry

end sum_first_n_terms_b_n_arithmetic_seq_l808_808935


namespace cos_neg_60_eq_half_l808_808351

noncomputable def cos_neg_60 : ℝ :=
  cos (real.pi * (-60) / 180)

theorem cos_neg_60_eq_half : cos_neg_60 = 1 / 2 :=
by
  sorry

end cos_neg_60_eq_half_l808_808351


namespace given_system_solution_l808_808009

noncomputable def solve_system : Prop :=
  ∃ x y z : ℝ, 
  x + y + z = 1 ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 
  x^3 + y^3 + z^3 = 89 / 125 ∧ 
  (x = 2 / 5 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = 2 / 5 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = 2 / 5 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = 2 / 5)

theorem given_system_solution : solve_system :=
sorry

end given_system_solution_l808_808009


namespace probability_below_line_5_l808_808988

noncomputable def probability_below_line {m n : ℕ} (hm : m ∈ {1, 2, 3, 4, 5, 6}) (hn : n ∈ {1, 2, 3, 4, 5, 6}) : ℚ :=
  if m + n < 5 then 1 else 0

theorem probability_below_line_5 :
  (finset.univ.product finset.univ).sum (λ p, probability_below_line (finset.mem_univ p.1) (finset.mem_univ p.2)) / 36 = (1/6 : ℚ) :=
begin
  sorry
end

end probability_below_line_5_l808_808988


namespace harmonic_interval_k_l808_808759

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 4

theorem harmonic_interval_k (a b k : ℝ) (h_inc : ∀ x y, a ≤ x → x ≤ y → y ≤ b → f(x) ≤ f(y))
  (h_range : ∀ y, ka ≤ y → y ≤ kb → ∃ x, a ≤ x ∧ x ≤ b ∧ f(x) = y) :
  2 < k ∧ k ≤ 3 :=
by
  sorry

end harmonic_interval_k_l808_808759


namespace non_overlapping_sets_l808_808867

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808867


namespace min_b_minus_a_l808_808495

-- Definitions of the given functions
def f (x : ℝ) : ℝ := Real.exp (x - 1)
def g (x : ℝ) : ℝ := (1 / 2) + Real.log (x / 2)

-- Problem statement to be proved
theorem min_b_minus_a (a b : ℝ) (h : f a = g b) : b - a = 1 + Real.log 2 :=
sorry

end min_b_minus_a_l808_808495


namespace exact_five_letters_correct_l808_808696

-- Define the number of letters
def num_letters : ℕ := 10

-- Define the binomial coefficient
def binom (n k : ℕ) := Nat.binomial n k

-- Define the derangement of 5
def derangement_5 : ℕ := 44

-- Calculate the number of favorable outcomes
def favorable_outcomes := binom num_letters 5 * derangement_5

-- Calculate the total number of permutations
def total_permutations := Nat.factorial num_letters

-- Calculate the probability
def probability := favorable_outcomes / total_permutations

-- The final theorem we want to prove
theorem exact_five_letters_correct :
  probability = (11 : ℚ) / 3600 :=
by
  -- Here we would provide the proof steps, but we use sorry to indicate the proof is omitted
  sorry

end exact_five_letters_correct_l808_808696


namespace pairing_sums_perfect_square_l808_808378

theorem pairing_sums_perfect_square (n : ℕ) (h : n > 1) :
  ∃ P : Fin (2 * n) → Fin (2 * n), (∀ i, i < n → (P (2 * i) + P (2 * i + 1))^2) :=
sorry

end pairing_sums_perfect_square_l808_808378


namespace store_income_l808_808553

def pencil_store_income (p_with_eraser_qty p_with_eraser_cost p_regular_qty p_regular_cost p_short_qty p_short_cost : ℕ → ℝ) : ℝ :=
  (p_with_eraser_qty * p_with_eraser_cost) + (p_regular_qty * p_regular_cost) + (p_short_qty * p_short_cost)

theorem store_income : 
  pencil_store_income 200 0.8 40 0.5 35 0.4 = 194 := 
by sorry

end store_income_l808_808553


namespace primes_solution_l808_808839

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_solution (p : ℕ) (hp : is_prime p) :
  is_prime (p^2 + 2007 * p - 1) ↔ p = 3 :=
by
  sorry

end primes_solution_l808_808839


namespace smallest_part_division_l808_808982

theorem smallest_part_division (y : ℝ) (h1 : y > 0) :
  ∃ (x : ℝ), x = y / 9 ∧ (∃ (a b c : ℝ), a = x ∧ b = 3 * x ∧ c = 5 * x ∧ a + b + c = y) :=
sorry

end smallest_part_division_l808_808982


namespace find_coordinates_of_B_l808_808919

-- Definitions based on the conditions
def Point := (ℝ × ℝ)
def Vector := (ℝ × ℝ)

def A : Point := (2, 4)
def a : Vector := (3, 4)

-- Given condition: AB = 2 * a
def vector_AB (B : Point) : Vector := (B.1 - A.1, B.2 - A.2)

theorem find_coordinates_of_B (B : Point) (h : vector_AB B = (2 * 3, 2 * 4)) : B = (8, 12) :=
sorry

end find_coordinates_of_B_l808_808919


namespace poly_eq_holds_l808_808027

noncomputable def P (x : ℝ) : ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ := sorry

theorem poly_eq_holds :
  ∃ (P Q : ℝ → ℝ), (∀ x, P(x^2) + Q(x) = P(x) + x^5 * Q(x)) ∧
  (P ≠ 0) ∧ (Q ≠ 0) ∧
  (∃ p q : ℕ, P.degree = p ∧ Q.degree = q ∧ 2 * p = 5 + q) :=
sorry

end poly_eq_holds_l808_808027


namespace general_term_formula_l808_808938

noncomputable def S (n : ℕ) : ℕ := 2^n - 1
noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

theorem general_term_formula (n : ℕ) (hn : n > 0) : 
    a n = S n - S (n - 1) := 
by 
  sorry

end general_term_formula_l808_808938


namespace ellipse_solution_l808_808934

-- Define the initial conditions
def ellipse_eq(a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ (λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1)

def focus_F (a b : ℝ) : Prop := a^2 - b^2 = 9

def midpoint_condition (A B : ℝ × ℝ) : Prop := (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1

-- The main theorem
theorem ellipse_solution :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ focus_F a b ∧
    (∀ x y, (x, y) ∈ ellipse_eq a b) ∧ (∀ A B, midpoint_condition A B → 
    ( ∀ x y : ℝ, (x, y) ∈ (λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1))) :=
by
  sorry

end ellipse_solution_l808_808934


namespace find_incorrect_statement_A_l808_808706

-- Definitions based on conditions
def separated_sister_chromosomes_same_size_not_homologous : Prop :=
  ∀ (c1 c2 : Chromosome), (separated_into_two c1 c2) → (same_size c1 c2 ∧ ¬homologous c1 c2)

def centromere_equals_chromosome : Prop :=
  ∀ n : ℕ, (number_of_centromeres n) = (number_of_chromosomes n)

def tetrad_indicates_homologous_pairs : Prop :=
  ∀ N : ℕ, (presence_of_N_tetrads N) → (N_pairs_of_homologous_chromosomes N)

def gene_order_sister_chromatids_not_necessarily_same : Prop :=
  ∃ (c1 c2 : Chromosome), (sister_chromatids c1 c2) → (mutation_or_crossing_over c1 c2 → ¬same_gene_order c1 c2)

-- Proposition based on the conditions
def incorrect_statement_A : Prop :=
  ∀ (c1 c2 : Chromosome), (same_size c1 c2) → (¬homologous c1 c2)

-- Main theorem statement
theorem find_incorrect_statement_A : 
  separated_sister_chromosomes_same_size_not_homologous 
  ∧ centromere_equals_chromosome 
  ∧ tetrad_indicates_homologous_pairs 
  ∧ gene_order_sister_chromatids_not_necessarily_same 
  → incorrect_statement_A :=
by
  sorry

end find_incorrect_statement_A_l808_808706


namespace units_digit_of_7_pow_6_pow_5_l808_808434

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808434


namespace sales_amount_equal_salary_choices_l808_808157

-- Define the conditions and variables
def salary_choice_1 : ℝ := 1800
def base_salary_choice_2 : ℝ := 1600
def commission_rate : ℝ := 0.04

-- Define the sales amount variable
variable (S : ℝ)

-- Define the salary equation for choice 2
def salary_choice_2 (S : ℝ) : ℝ := base_salary_choice_2 + commission_rate * S

-- The statement to be proven
theorem sales_amount_equal_salary_choices : salary_choice_1 = salary_choice_2 5000 :=
by
  unfold salary_choice_2
  rw [← @eq_comm _ (base_salary_choice_2 + commission_rate * 5000)]
  calc
    1800 = 1600 + 0.04 * 5000 : sorry

end sales_amount_equal_salary_choices_l808_808157


namespace value_of_a_exists_unique_k_range_of_m_l808_808947

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log x

def g (x : ℝ) : ℝ := (2 * x^2) / Real.exp x

def tangentLine (a x : ℝ) : ℝ := (1 + a) * (x - 1)

-- Problem 1: Prove that the value of 'a' is 2
theorem value_of_a (a : ℝ) :
  (f a 1 = 0 ∧ tangentLine a 2 = 3) → a = 2 :=
sorry

-- Problem 2: Prove that there exists a natural number 'k' such that f(x) - g(x) 
-- has a unique zero in the interval (k, k + 1)
def φ (a x : ℝ) : ℝ := (x + a) * Real.log x - (2 * x^2) / Real.exp x

theorem exists_unique_k {a : ℝ} (zero_in_interval : ∃ k : ℕ, ∃ x₀ : ℝ, (φ a x₀ = 0) ∧ k < x₀ ∧ x₀ < k + 1) :
  a = 2 → ∃ (k : ℕ), (∀ x₀₁ : ℝ, (φ 2 x₀₁ > 0) → φ 2 x₀₁ ∈ Ico k (k + 1)) :=
sorry

-- Problem 3: Prove the range of values for 'm' such that there exists x₀ such that h(x₀) ≥ m
def h (a x : ℝ) : ℝ := min ((x + a) * Real.log x) ((2 * x^2) / Real.exp x)

theorem range_of_m {a : ℝ} (m : ℝ) :
  a = 2 → (∃ x₀ : ℝ, 0 < x₀ ∧ h 2 x₀ ≥ m) ↔ m ≤ (8 / Real.exp 2) :=
sorry

end value_of_a_exists_unique_k_range_of_m_l808_808947


namespace largest_part_of_proportional_division_l808_808113

theorem largest_part_of_proportional_division (sum : ℚ) (a b c largest : ℚ) 
  (prop1 prop2 prop3 : ℚ) 
  (h1 : sum = 156)
  (h2 : prop1 = 2)
  (h3 : prop2 = 1 / 2)
  (h4 : prop3 = 1 / 4)
  (h5 : sum = a + b + c)
  (h6 : a / prop1 = b / prop2 ∧ b / prop2 = c / prop3)
  (h7 : largest = max a (max b c)) :
  largest = 112 + 8 / 11 :=
by
  sorry

end largest_part_of_proportional_division_l808_808113


namespace total_words_in_poem_l808_808629

theorem total_words_in_poem 
  (stanzas : ℕ) 
  (lines_per_stanza : ℕ) 
  (words_per_line : ℕ) 
  (h_stanzas : stanzas = 20) 
  (h_lines_per_stanza : lines_per_stanza = 10) 
  (h_words_per_line : words_per_line = 8) : 
  stanzas * lines_per_stanza * words_per_line = 1600 := 
sorry

end total_words_in_poem_l808_808629


namespace intersection_point_l808_808018

variables (g : ℤ → ℤ) (b a : ℤ)
def g_def := ∀ x : ℤ, g x = 4 * x + b
def inv_def := ∀ y : ℤ, g y = -4 → y = a
def point_intersection := ∀ y : ℤ, (g y = -4) → (y = a) → (a = -16 + b)
def solution : ℤ := -4

theorem intersection_point (b a : ℤ) (h₁ : g_def g b) (h₂ : inv_def g a) (h₃ : point_intersection g a b) :
  a = solution :=
  sorry

end intersection_point_l808_808018


namespace storybook_pages_l808_808800

theorem storybook_pages :
  (10 + 5) / (1 - (1 / 5) * 2) = 25 := by
  sorry

end storybook_pages_l808_808800


namespace common_difference_is_neg3_l808_808068

variable (a : ℕ → ℤ) (d : ℤ) (n : ℕ) 
variable (h1 : ∑ i in finset.range n, a (2 * i + 1) = 90)
variable (h2 : ∑ i in finset.range n, a (2 * i + 2) = 72)
variable (h3 : a 1 - a (2 * n) = 33)
variable (h_seq : ∀ k, a (k + 1) = a k + d)

theorem common_difference_is_neg3 :
  d = -3 := by
  sorry

end common_difference_is_neg3_l808_808068


namespace gamma_k_is_circle_l808_808500

-- Define real numbers, plane, and distinct points A and B
variable (k : ℝ) (A B : ℝ × ℝ)
hypothesis h1 : k > 0 ∧ k ≠ -1
hypothesis h2 : A ≠ B

-- Definition of point I as the barycenter of (A, 1) and (B, -k)
def I : ℝ × ℝ := ((1 : ℝ) / (1 - k)) • A + ((-k : ℝ) / (1 - k)) • B

-- Definition of point J as the barycenter of (A, 1) and (B, k)
def J : ℝ × ℝ := ((1 : ℝ) / (1 + k)) • A + ((k : ℝ) / (1 + k)) • B

-- Define the set Γₖ
def Γₖ : set (ℝ × ℝ) := { M | dist M A = k * dist M B }

-- The statement we need to prove
theorem gamma_k_is_circle :
  Γₖ k A B = { M | ∃ x y: ℝ, M = (x, y) ∧ dist M I = dist M J } :=
sorry

end gamma_k_is_circle_l808_808500


namespace trapezoid_parallel_line_division_l808_808272

variables {A B C D K : Type} [linear_ordered_field K] 
variables [metric_space K] [normed_group K] [normed_space ℝ K]

/-- Given a trapezoid ABCD with bases AD and BC and diagonals AC and BD, 
    there exists a line l parallel to AD (or BC) such that the segment 
    inside the trapezoid is divided into three equal parts by the diagonals. -/
theorem trapezoid_parallel_line_division
  (A B C D K : K) 
  (h_trap: is_trapezoid A B C D)
  (h_parallel_base: ∥A-D∥ = ∥B-C∥) 
  (M : midpoint A D)
  (K : intersection (B M) (A C))
  (l : line_through K parallel_to (A D)) :
  divides_three_equal_parts l (A C) (B D) := 
sorry

end trapezoid_parallel_line_division_l808_808272


namespace complement_of_N_is_135_l808_808955

-- Define the universal set M and subset N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

-- Prove that the complement of N in M is {1, 3, 5}
theorem complement_of_N_is_135 : M \ N = {1, 3, 5} := 
by
  sorry

end complement_of_N_is_135_l808_808955


namespace length_of_DF_l808_808598

theorem length_of_DF (DP EQ : ℝ) (h1 : DP = 15) (h2 : EQ = 20) (perpendicular : DP * EQ = 0) :
  DF = 20 * sqrt 13 / 3 :=
by
  -- Given: DP == 15, EQ == 20, and they are perpendicular
  assume h1 : DP = 15
  assume h2 : EQ = 20
  assume perpendicular : DP * EQ = 0

  -- Prove: DF == 20 * sqrt 13 / 3
  sorry

end length_of_DF_l808_808598


namespace night_crew_fraction_l808_808799

theorem night_crew_fraction (D N B : ℕ) 
  (h1 : ∀ n, n * (3/4) * B = 0.3 * (D * B + N * (3/4) * B))
  (h2 : ∀ d, d * B = 0.7 * (D * B + N * (3/4) * B))
  : (N : ℚ) / D = 4 / 7 :=
by
  sorry

end night_crew_fraction_l808_808799


namespace tangent_line_eq_range_of_a_no_k_exists_l808_808090

section partI

variable (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem tangent_line_eq (h₀ : f x = a * x^2 + log x) (h₁ : a = -1) (h₂ : f'(x) = (2 * a * x + 1 / x)) :
  tangent_line_eq (y = f x, x = 1, y + 1 = - (x - 1), x + y = 0) :=
sorry

end partI

section partII

variable (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem range_of_a (h₀ : f x = a * x^2 + log x) (h₁ : a < 0) (h₂ : (∀ x, f x < - 1 / 2)):
  ∃ a : ℝ, a < - 1 / 2 :=
sorry

end partII

section partIII

variable (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem no_k_exists (h₀ : f x = a * x^2 + log x) (h₁ : a = 1) (h₂ : f'(x) = (2 * a * x + 1 / x)):
  ¬ ∃ k < 100, ∃ x_1 x_2 ... x_k ∈ [1, 10], f'(x_1) + f'(x_2) + ... + f'(x_k) ≥ 2012 :=
sorry

end partIII

end tangent_line_eq_range_of_a_no_k_exists_l808_808090


namespace scientific_notation_of_coronavirus_diameter_l808_808569

theorem scientific_notation_of_coronavirus_diameter :
  0.00000015 = 1.5 * 10^(-7) := 
begin
  -- This proof is skipped and left as an exercise
  sorry
end

end scientific_notation_of_coronavirus_diameter_l808_808569


namespace gathering_people_l808_808687

theorem gathering_people (total_chairs : ℕ) (total_people : ℕ) (empty_chairs : ℕ) 
        (h1 : empty_chairs = 9)
        (h2 : total_chairs = 9 * 3)
        (h3 : 18 = (2 / 3) * total_chairs)
        (h4 : (3 / 5) * total_people = 18) :
  total_people = 30 :=
begin
  sorry
end

end gathering_people_l808_808687


namespace range_of_f_l808_808021

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_f (k : ℝ) (h : k < 0) :
  set.range (λ x : {x : ℝ // 0 < x ∧ x ≤ 1}, f x.val k) = set.Ioi 0 :=
sorry

end range_of_f_l808_808021


namespace pencil_packing_l808_808190

theorem pencil_packing (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (a % 10 = 7) →
  (a % 12 = 9) →
  (a = 237 ∨ a = 297) :=
by {
  assume h_range h_red_boxes h_blue_boxes,
  sorry
}

end pencil_packing_l808_808190


namespace sum_of_first_9_terms_l808_808164

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 0 + a (n - 1)) / 2

theorem sum_of_first_9_terms (h_arith : is_arithmetic_sequence a)
  (h_sum : a 0 + a 4 + a 8 = 18) : arithmetic_sequence_sum a 9 = 54 := by
  sorry

end sum_of_first_9_terms_l808_808164


namespace longest_sequence_start_l808_808200

def machine_output (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else x + 3

def machine_sequence (x : ℕ) : List ℕ :=
  let rec aux (seen : List ℕ) (current : ℕ) : List ℕ :=
    if current ∈ seen then seen else aux (current :: seen) (machine_output current)
  aux [] x

def sequence_length (x : ℕ) : ℕ := (machine_sequence x).length

theorem longest_sequence_start (x ≤ 100) : sequence_length 67 = max {y : ℕ | y <= 100} sequence_length y := 
sorry

end longest_sequence_start_l808_808200


namespace product_of_three_numbers_l808_808668

theorem product_of_three_numbers
  (x y z n : ℤ)
  (h1 : x + y + z = 165)
  (h2 : n = 7 * x)
  (h3 : n = y - 9)
  (h4 : n = z + 9) :
  x * y * z = 64328 := 
by
  sorry

end product_of_three_numbers_l808_808668


namespace prob_B_takes_second_shot_prob_A_takes_i_shot_correct_expected_shots_A_l808_808616

-- Definitions based on the conditions
def shooting_percentage_A : ℝ := 0.6
def shooting_percentage_B : ℝ := 0.8
def initial_prob : ℝ := 0.5

-- Proof problem for Part 1
theorem prob_B_takes_second_shot :
  (initial_prob * (1 - shooting_percentage_A)) + 
  (initial_prob * shooting_percentage_B) = 0.6 :=
sorry

-- Definition of the probability player A takes the ith shot
def prob_A_takes_i_shot (i : ℕ) : ℝ :=
  1 / 3 + (1 / 6) * (2 / 5)^(i - 1)

-- Proof problem for Part 2
theorem prob_A_takes_i_shot_correct (i : ℕ) :
  prob_A_takes_i_shot i = 
  1 / 3 + (1 / 6) * (2 / 5)^(i - 1) :=
sorry

-- Proof problem for Part 3
theorem expected_shots_A (n : ℕ) :
  (∑ i in finset.range n, prob_A_takes_i_shot (i + 1)) = 
  (5 / 18) * (1 - (2 / 5)^n) + (n / 3) :=
sorry

end prob_B_takes_second_shot_prob_A_takes_i_shot_correct_expected_shots_A_l808_808616


namespace non_overlapping_original_sets_exists_l808_808872

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808872


namespace combine_like_terms_problem1_combine_like_terms_problem2_l808_808349

-- Problem 1 Statement
theorem combine_like_terms_problem1 (x y : ℝ) : 
  2*x - (x - y) + (x + y) = 2*x + 2*y :=
by
  sorry

-- Problem 2 Statement
theorem combine_like_terms_problem2 (x : ℝ) : 
  3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 :=
by
  sorry

end combine_like_terms_problem1_combine_like_terms_problem2_l808_808349


namespace transformed_function_value_l808_808229

-- Define the original function y = sin(2x + π/6)
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

-- Define the transformed function g
def g (x : ℝ) : ℝ := Real.sin (4 * (x - Real.pi / 3) + Real.pi / 6)

-- State the theorem to prove that g(π/3) = 1/2
theorem transformed_function_value : g (Real.pi / 3) = 1 / 2 :=
by
  sorry

end transformed_function_value_l808_808229


namespace paving_stones_required_l808_808015

noncomputable def area_trapezoid (a b h : ℝ) : ℝ :=
  1/2 * (a + b) * h

noncomputable def area_rectangle (l w : ℝ) : ℝ :=
  l * w

noncomputable def area_paving_stone : ℝ :=
  let rect1 := area_rectangle 2.5 2
  let rect2 := area_rectangle 1.5 3
  rect1 + rect2

noncomputable def number_of_paving_stones (courtyard_area paving_stone_area : ℝ) : ℝ :=
  (courtyard_area / paving_stone_area).ceil

theorem paving_stones_required :
  let courtyard_area := area_trapezoid 16.5 25 12
  let paving_stone_area := area_paving_stone
  number_of_paving_stones courtyard_area paving_stone_area = 27 :=
by
  sorry

end paving_stones_required_l808_808015


namespace triangle_solution_l808_808537

theorem triangle_solution
  (a : ℝ) (A : ℝ) (B : ℝ)
  (ha : a = 42) (hA : A = 45) (hB : B = 60) :
  let C := 180 - B - A in
  let b := 21 * Real.sqrt 6 in
  let c := 21 * (Real.sqrt 3 + 1) in
  C = 75 ∧ b = 21 * Real.sqrt 6 ∧ c = 21 * (Real.sqrt 3 + 1) :=
by
  sorry

end triangle_solution_l808_808537


namespace disjoint_sets_condition_l808_808070

theorem disjoint_sets_condition (A B : Set ℕ) (h_disjoint: Disjoint A B) (h_union: A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a > n ∧ b > n ∧ a ≠ b ∧ 
             ((a ∈ A ∧ b ∈ A ∧ a + b ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ a + b ∈ B)) := 
by
  sorry

end disjoint_sets_condition_l808_808070


namespace max_difference_between_even_and_odd_digit_numbers_l808_808359

theorem max_difference_between_even_and_odd_digit_numbers : 
  ∀ (n m : ℕ), 
  (n < 1000 ∧ n > 99 ∧ m < 1000 ∧ m > 99 ∧ 
   (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
     n = 100 * a + 10 * b + c ∧ 
     a ∈ {0, 2, 4, 6, 8} ∧ b ∈ {0, 2, 4, 6, 8} ∧ c ∈ {0, 2, 4, 6, 8}) ∧
   (∀ d e f : ℕ, d ≠ e ∧ e ≠ f ∧ d ≠ f ∧ 
     m = 100 * d + 10 * e + f ∧ 
     d ∈ {1, 3, 5, 7, 9} ∧ e ∈ {1, 3, 5, 7, 9} ∧ f ∈ {1, 3, 5, 7, 9})) → 
  n - m = 729 :=
by 
  sorry

end max_difference_between_even_and_odd_digit_numbers_l808_808359


namespace units_digit_7_power_l808_808396

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808396


namespace inclination_angle_tan_60_perpendicular_l808_808231

/-
The inclination angle of the line given by x = tan(60 degrees) is 90 degrees.
-/
theorem inclination_angle_tan_60_perpendicular : 
  ∀ (x : ℝ), x = Real.tan (60 *Real.pi / 180) → 
  ∃ θ : ℝ, θ = 90 :=
sorry

end inclination_angle_tan_60_perpendicular_l808_808231


namespace product_of_roots_l808_808807

theorem product_of_roots :
  (Real.sqrt (4 : ℝ).sqrt 16) * (Real.sqrt (5 : ℝ).sqrt 32) = 4 := 
sorry

end product_of_roots_l808_808807


namespace find_C_l808_808295

theorem find_C (A B C : ℕ) (h1 : A + B + C = 900) (h2 : A + C = 400) (h3 : B + C = 750) : C = 250 :=
by
  sorry

end find_C_l808_808295


namespace negation_of_universal_l808_808656

theorem negation_of_universal : 
  (¬ (∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, 2 * x^2 - x + 1 < 0) :=
by
  sorry

end negation_of_universal_l808_808656


namespace divisor_of_70th_number_l808_808588

-- Define the conditions
def s (d : ℕ) (n : ℕ) : ℕ := n * d + 5

-- Theorem stating the given problem
theorem divisor_of_70th_number (d : ℕ) (h : s d 70 = 557) : d = 8 :=
by
  -- The proof is to be filled in later. 
  -- Now, just create the structure.
  sorry

end divisor_of_70th_number_l808_808588


namespace exists_non_divisor_pair_l808_808326

theorem exists_non_divisor_pair :
  ∀ (A : Finset ℕ), (Finset.card A = 2003) →
  ∃ (a b ∈ A), a ≠ b ∧ ¬ (a + b ∣ ∑ x in A, x) :=
by
  intro A hA
  sorry

end exists_non_divisor_pair_l808_808326


namespace ratio_is_correct_l808_808600

-- Define the constants
def total_students : ℕ := 47
def current_students : ℕ := 6 * 3
def girls_bathroom : ℕ := 3
def new_groups : ℕ := 2 * 4
def foreign_exchange_students : ℕ := 3 * 3

-- The total number of missing students
def missing_students : ℕ := girls_bathroom + new_groups + foreign_exchange_students

-- The number of students who went to the canteen
def students_canteen : ℕ := total_students - current_students - missing_students

-- The ratio of students who went to the canteen to girls who went to the bathroom
def canteen_to_bathroom_ratio : ℕ × ℕ := (students_canteen, girls_bathroom)

theorem ratio_is_correct : canteen_to_bathroom_ratio = (3, 1) :=
by
  -- Proof goes here
  sorry

end ratio_is_correct_l808_808600


namespace tangent_line_eq_range_of_b_sum_sq_gt_e_l808_808050

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * sin x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * sqrt x

-- Part 1
theorem tangent_line_eq (a : ℝ) : ∀ (x f0 : ℝ), f a x = exp x - a * sin x → f0 = f a 0 → 
                                   y = (1 - a) x + 1 := by
sorry

-- Part 2i
theorem range_of_b (b : ℝ) : 
  ∀ (x : ℝ), f 0 x = exp x → g b x = b * sqrt x → 
  (∃ x, f 0 x = g b x) → b ≥ sqrt (2 * exp 1) := by
sorry

-- Part 2ii
theorem sum_sq_gt_e (a b : ℝ) : ∃ x, exp x = a * sin x + b * sqrt x → a^2 + b^2 > real.exp 1 := by
sorry

end tangent_line_eq_range_of_b_sum_sq_gt_e_l808_808050


namespace probability_xi_leq_7_l808_808540

noncomputable def probability_ball_draw_score : ℚ :=
  let red_balls := 4
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let score := λ (red black : ℕ), red + 3 * black
  let comb := λ n k, (nat.choose n k : ℚ)
  (comb red_balls 4 / comb total_balls 4) +
  (comb red_balls 3 * comb black_balls 1 / comb total_balls 4)

theorem probability_xi_leq_7 : probability_ball_draw_score = (13 / 35) := by
  sorry

end probability_xi_leq_7_l808_808540


namespace number_of_terms_in_sequence_l808_808966

theorem number_of_terms_in_sequence :
  ∃ n : ℕ, (1 + 4 * (n - 1) = 2025) ∧ n = 507 := by
  sorry

end number_of_terms_in_sequence_l808_808966


namespace first_player_wins_l808_808457

-- Definitions of dominos and moves
structure Domino :=
  (a : Nat)
  (b : Nat)

structure Game :=
  (dominos : Finset Domino)
  (chain : List Domino)
  (current_player : Bool) -- true for player 1, false for player 2

-- Assumptions needed
def valid_move (g : Game) (d : Domino) : Prop :=
  d ∈ g.dominos ∧ 
  (g.chain = [] ∨
   (d.a = g.chain.head.b ∨ d.b = g.chain.head.b ∨
    d.a = g.chain.last.a ∨ d.b = g.chain.last.a))

-- Move function to update the game state
def make_move (g : Game) (d : Domino) : Game :=
  { g with
    dominos := g.dominos.erase d,
    chain := if g.chain = [] then [d] else if d.a = g.chain.head.b then d :: g.chain else if d.b = g.chain.last.a then g.chain ++ [d] else g.chain,
    current_player := bnot g.current_player }

-- Winning strategy theorem
theorem first_player_wins (game : Game) :
  (∃ d : Domino, valid_move game d ∧ game.current_player = true) ∧
  (∀ g' : Game, (valid_move game d ∧ g' = make_move game d) → (∃ d' : Domino, valid_move g' d' ∧ g'.current_player = false)) :=
sorry

end first_player_wins_l808_808457


namespace trig_inequality_l808_808478

theorem trig_inequality (α β : ℝ) (h : cos α ≠ cos β) (k : ℕ) (hk : k > 1) :
  abs ((cos (k * β) * cos α - cos (k * α) * cos β) / (cos β - cos α)) < k^2 - 1 :=
by
  sorry

end trig_inequality_l808_808478


namespace dalton_needs_more_money_l808_808010

-- Definitions based on the conditions
def jumpRopeCost : ℕ := 7
def boardGameCost : ℕ := 12
def ballCost : ℕ := 4
def savedAllowance : ℕ := 6
def moneyFromUncle : ℕ := 13

-- Computation of how much more money is needed
theorem dalton_needs_more_money : 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  totalCost - totalMoney = 4 := 
by 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  have h1 : totalCost = 23 := by rfl
  have h2 : totalMoney = 19 := by rfl
  calc
    totalCost - totalMoney = 23 - 19 := by rw [h1, h2]
    _ = 4 := by rfl

end dalton_needs_more_money_l808_808010


namespace f_zero_value_f_analytic_expression_find_intersection_set_l808_808933

variables {f : ℝ → ℝ} {g : ℝ → ℝ} {a : ℝ} {x y : ℝ}

-- Given conditions
def f_property (f : ℝ → ℝ) : Prop := ∀ x y, f(x + y) - f(y) = x * (x + 2 * y + 1)
def f_at_1 (f : ℝ → ℝ) : Prop := f 1 = 0

-- Proof problem (1)
theorem f_zero_value (h : f_property f) (h1 : f_at_1 f) : f 0 = -2 :=
sorry

-- Proof problem (2)
theorem f_analytic_expression (h : f_property f) (h1 : f_at_1 f) : ∀ x, f x = x ^ 2 + x - 2 :=
sorry

-- Definitions for sets A and B (condition (3))
def set_A (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, 0 < x ∧ x < 1/2 → f x + 3 < 2 * x + a
def set_B (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, -2 ≤ x ∧ x ≤ 2 → ∀ y, (g y = f y - a * y) → (∀ x₁ x₂, x₁ ≤ x₂ → g x₁ ≤ g x₂)

-- Proof problem (3)
theorem find_intersection_set (h : f_property f) (h1 : f_at_1 f) :
  { a : ℝ | set_A f a } ∩ { a : ℝ | -3 < a ∧ a < 5 } = { a : ℝ | 1 ≤ a ∧ a < 5 } :=
sorry

end f_zero_value_f_analytic_expression_find_intersection_set_l808_808933


namespace unique_function_satisfying_condition_l808_808367

theorem unique_function_satisfying_condition (f : ℝ → ℝ) (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, f(x + f(k * y)) = x + y) → ∃! f : ℝ → ℝ, ∀ x y : ℝ, f(x + f(k * y)) = x + y :=
by
  sorry

end unique_function_satisfying_condition_l808_808367


namespace line_DC_perpendicular_to_AB_l808_808772

-- Definitions of the points and their relationships
variables {A B C A₁ B₁ A₂ B₂ D : Type} -- Points in the Euclidean plane
variable [euclidean_geometry : EuclideanGeometry A B C A₁ B₁ A₂ B₂ D]

-- Conditions of the problem
axiom h1 : SecantThroughPoint C
axiom h2 : Perpendicular A (SecantThroughPoint C) ↔ (Perpendicular AA₁ (SecantThroughPoint C) ∧ meetsAt A₁)
axiom h3 : Perpendicular B (SecantThroughPoint C) ↔ (Perpendicular BB₁ (SecantThroughPoint C) ∧ meetsAt B₁)
axiom h4 : Perpendicular A₁ (Side BC) ↔ (Perpendicular A₁A₂ (Side BC) ∧ meetsAt A₂)
axiom h5 : Perpendicular B₁ (Side AC) ↔ (Perpendicular B₁B₂ (Side AC) ∧ meetsAt B₂)
axiom h6 : meetsAt A₁A₂ B₁B₂ D

-- Goal: Prove that line DC is perpendicular to side AB
theorem line_DC_perpendicular_to_AB : Perpendicular DC (Side AB) :=
by
  sorry

end line_DC_perpendicular_to_AB_l808_808772


namespace original_sets_exist_l808_808897

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808897


namespace trigonometric_identity_proof_l808_808225

theorem trigonometric_identity_proof
  {x : ℝ}
  (h : cos^2 x + cos^2 (2 * x) + sin^2 (3 * x) + sin^2 (4 * x) = 3) :
  sin (2 * x) * sin (3 * x) * cos (2 * x) = 0 ∧ (2 + 3 + 2 = 7) :=
by {
  sorry -- Proof goes here
}

end trigonometric_identity_proof_l808_808225


namespace problem_statement_l808_808181

-- Define the types of marbles and their associated points.
inductive Marble
| red : Marble
| blue : Marble

def points : Marble → Int
| Marble.red  => 3
| Marble.blue => -2

-- Define the probability of failing to end with exactly one point
noncomputable def failure_probability (marbles : List Marble) : ℚ :=
  if marbles.length = 2013 then
    let all_red := List.all marbles (λ m => m = Marble.red)
    let all_blue := List.all marbles (λ m => m = Marble.blue)
    if all_red ∨ all_blue then 1 / 2 ^ 2012 else 1 - 1 / 2 ^ 2012
  else 0

-- Statement of the problem
theorem problem_statement :
  ∀ (marbles : List Marble), marbles.length = 2013 → failure_probability marbles = 1 / 2 ^ 2012 :=
by
  intros
  sorry

end problem_statement_l808_808181


namespace bl_is_angle_bisector_l808_808065

-- Defining the triangle and points
variables {A B C X Y L : Type*}
variables [triangle A B C]
variables [segment AX : A X, segment BY : B Y]
variables [segment AC : A C, segment BC : B C]
variables [segment L_on_AC : L ∈ segment A C]

-- Defining the conditions
variables (h1 : AX = BY)
variables (h2 : cyclic_quad A X Y C)
variables (h3 : parallel XL BC)

-- Conclusion to be proved in Lean 4 - statement that BL is the angle bisector
theorem bl_is_angle_bisector (A B C X Y L : Type*) [triangle A B C] [segment AX : A X]
  [segment BY : B Y] [segment AC : A C] [segment BC : B C] [segment L_on_AC : L ∈ segment A C]
  (h1 : AX = BY) (h2 : cyclic_quad A X Y C) (h3 : parallel XL BC) : 
  is_angle_bisector ABC B L :=
sorry

end bl_is_angle_bisector_l808_808065


namespace loaves_needed_l808_808732

theorem loaves_needed 
  (slices_per_loaf : ℕ)
  (slices_regular_sandwich : ℕ)
  (count_regular_sandwiches : ℕ)
  (slices_double_meat_sandwich : ℕ)
  (count_double_meat_sandwiches : ℕ)
  (slices_triple_decker_sandwich : ℕ)
  (count_triple_decker_sandwiches : ℕ)
  (slices_club_sandwich : ℕ)
  (count_club_sandwiches : ℕ)
  (N : ℕ)
  (H_slices_per_loaf : slices_per_loaf = 20)
  (H_slices_regular_sandwich : slices_regular_sandwich = 2)
  (H_count_regular_sandwiches : count_regular_sandwiches = 25)
  (H_slices_double_meat_sandwich : slices_double_meat_sandwich = 3)
  (H_count_double_meat_sandwiches : count_double_meat_sandwiches = 18)
  (H_slices_triple_decker_sandwich : slices_triple_decker_sandwich = 4)
  (H_count_triple_decker_sandwiches : count_triple_decker_sandwiches = 12)
  (H_slices_club_sandwich : slices_club_sandwich = 5)
  (H_count_club_sandwiches : count_club_sandwiches = 8)
  (H_N : N = 10) :
  N = 
   let total_slices := (count_regular_sandwiches * slices_regular_sandwich) +
                       (count_double_meat_sandwiches * slices_double_meat_sandwich) +
                       (count_triple_decker_sandwiches * slices_triple_decker_sandwich) +
                       (count_club_sandwiches * slices_club_sandwich) in
   (total_slices / slices_per_loaf).ceil := by
  sorry

end loaves_needed_l808_808732


namespace original_sets_exist_l808_808909

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808909


namespace units_digit_of_7_pow_6_pow_5_l808_808430

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808430


namespace Thabo_books_ratio_l808_808638

variable (P_f P_nf H_nf : ℕ)

theorem Thabo_books_ratio :
  P_f + P_nf + H_nf = 220 →
  H_nf = 40 →
  P_nf = H_nf + 20 →
  P_f / P_nf = 2 :=
by sorry

end Thabo_books_ratio_l808_808638


namespace integer_points_on_line_l808_808320

/-- Given a line that passes through points C(3, 3) and D(150, 250),
prove that the number of other points with integer coordinates
that lie strictly between C and D is 48. -/
theorem integer_points_on_line {C D : ℝ × ℝ} (hC : C = (3, 3)) (hD : D = (150, 250)) :
  ∃ (n : ℕ), n = 48 ∧ 
  ∀ p : ℝ × ℝ, C.1 < p.1 ∧ p.1 < D.1 ∧ 
  C.2 < p.2 ∧ p.2 < D.2 → 
  (∃ (k : ℤ), p.1 = ↑k ∧ p.2 = (5/3) * p.1 - 2) :=
sorry

end integer_points_on_line_l808_808320


namespace monotonicity_of_f_range_of_a_l808_808093

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → a ≤ 0 → 0 < (1 - a * x) / x) ∧
  (∀ x : ℝ, 0 < x → a > 0 → (0 < x ∧ x < 1 / a) → 0 < (1 - a * x) / x) ∧
  (∀ x : ℝ, 0 < x → a > 0 → (1 / a < x) → (1 - a * x) / x < 0) :=
by 
  sorry

noncomputable def g (x : ℝ) : ℝ := (log x) / (x + 1) + 1 / (e * (x + 1))

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → ln x - a * x ≤ (log x) / (x + 1) - x / (e * (x + 1))) →
  a ∈ Set.Ici (1 / e) :=
by 
  sorry

end monotonicity_of_f_range_of_a_l808_808093


namespace determine_m_values_l808_808942

theorem determine_m_values (m : ℚ) :
  ((∃ x y : ℚ, x = -3 ∧ y = 0 ∧ (m^2 - 2 * m - 3) * x + (2 * m^2 + m - 1) * y = 2 * m - 6) ∨
  (∃ k : ℚ, k = -1 ∧ (m^2 - 2 * m - 3) + (2 * m^2 + m - 1) * k = 0)) →
  (m = -5/3 ∨ m = 4/3) :=
by
  sorry

end determine_m_values_l808_808942


namespace finite_set_toggle_possible_not_possible_only_2x2_l808_808960

variable (grid : ℤ × ℤ → bool) -- infinite grid of lamps, represented as a function from coordinates to boolean values

def toggle_square (grid : ℤ × ℤ → bool) (x y size : ℤ) : ℤ × ℤ → bool :=
  λ pos => if x ≤ pos.1 ∧ pos.1 < x + size ∧ y ≤ pos.2 ∧ pos.2 < y + size then not (grid pos) else grid pos

-- (a) Part
theorem finite_set_toggle_possible (finite_set : finite (set (ℤ × ℤ))) :
  ∃ (toggled_grid : (ℤ × ℤ → bool)), 
    (∃ (seq : list (ℤ × ℤ × ℤ)), ∀ pos, grid pos = (toggled_grid pos) ⧸ seq) ∧ -- toggling function shows toggled_grid
    (∀ pos, toggled_grid pos = true ↔ pos ∈ finite_set) := 
sorry

-- (b) Part
theorem not_possible_only_2x2 (restricted_ops : (ℤ × ℤ × ℤ) → bool) :
  (restricted_ops = λ (x y size), size = 3 ∨ size = 4 ∨ size = 5 → false)
  ∧ (forall pos, not (grid pos)) → 
  ∀ (seq : list (ℤ × ℤ × ℤ)),
    (∀ topp, (restricted_ops topp)) →
    (∀ pos, grid pos = true → (exists (a, b, a+b = 2 ∧ b = 2) → false)) := 
sorry

end finite_set_toggle_possible_not_possible_only_2x2_l808_808960


namespace power_mod_8_l808_808284

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end power_mod_8_l808_808284


namespace sum_of_prime_factors_of_2_pow_10_minus_1_l808_808244

theorem sum_of_prime_factors_of_2_pow_10_minus_1 :
  let n := 2^10 - 1 in 
  ∃ p1 p2 p3 : ℕ, 
    p1.prime ∧ p2.prime ∧ p3.prime ∧ 
    n % p1 = 0 ∧ n % p2 = 0 ∧ n % p3 = 0 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ 
    p1 + p2 + p3 = 45 :=
by
  sorry

end sum_of_prime_factors_of_2_pow_10_minus_1_l808_808244


namespace a_formula_T_less_than_one_l808_808324

noncomputable def a_sequence : ℕ+ → ℝ
| ⟨1, _⟩ => 4
| ⟨(n+1), h⟩ => let an := a_sequence ⟨n+1, h_pred n h⟩ in 2*an

lemma seq_relation (n : ℕ+) : (a_sequence (n+1))^2 - 2*(a_sequence n)^2 = (a_sequence n) * (a_sequence (n+1)) := sorry

theorem a_formula (n : ℕ) : a_sequence ⟨n+1, by simp⟩ = 2^(n+1) := sorry

noncomputable def b_sequence (n : ℕ+) : ℝ := 
  1 / (Real.log 2 * (n + 1)) / (Real.log 2 * n)

noncomputable def T_sequence (n : ℕ) : ℝ := 
  ∑ i in Finset.range n, b_sequence ⟨i + 1, Nat.succ_pos' i⟩

theorem T_less_than_one (n : ℕ) : T_sequence n < 1 := sorry

end a_formula_T_less_than_one_l808_808324


namespace tens_digit_of_6_pow_19_l808_808300

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tens_digit_of_6_pow_19 : tens_digit (6 ^ 19) = 9 := 
by 
  sorry

end tens_digit_of_6_pow_19_l808_808300


namespace minimum_perimeter_l808_808255

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l808_808255


namespace red_ball_higher_probability_l808_808209

noncomputable def bins := {1, 2, 3, 4}
noncomputable def probability_of_bin (k : bins) : ℝ := (16 / 15) * 2^(-k)

noncomputable def same_bin_probability : ℝ :=
  ∑ k in bins, (probability_of_bin k) ^ 2

noncomputable def different_bin_probability : ℝ :=
  1 - same_bin_probability

noncomputable def higher_numbered_bin_probability : ℝ :=
  different_bin_probability / 2

theorem red_ball_higher_probability : 
  higher_numbered_bin_probability = 0.3533 :=
  by
    sorry

end red_ball_higher_probability_l808_808209


namespace angle_at_4_10_l808_808016

def degrees_between_hands (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_hand_angle := (hours % 12) * 30 + (minutes / 60) * 30
  let minute_hand_angle := minutes * 6
  let angle := abs (hour_hand_angle - minute_hand_angle)
  if angle > 180 then 360 - angle else angle

theorem angle_at_4_10 : degrees_between_hands 4 10 = 65 := by
  sorry

end angle_at_4_10_l808_808016


namespace initial_cows_l808_808604

theorem initial_cows (initial_pigs initial_goats added_cows added_pigs added_goats total_animals : ℕ) 
  (h1 : initial_pigs = 3)
  (h2 : initial_goats = 6)
  (h3 : added_cows = 3)
  (h4 : added_pigs = 5)
  (h5 : added_goats = 2)
  (h6 : total_animals = 21) : 
  ∃ (C : ℕ), 
  C + initial_pigs + initial_goats + added_cows + added_pigs + added_goats = total_animals ∧ 
  C = 2 :=
begin
  sorry
end

end initial_cows_l808_808604


namespace sonnets_not_read_l808_808963

-- Define the conditions in the original problem
def sonnet_lines := 14
def unheard_lines := 70

-- Define a statement that needs to be proven
-- Prove that the number of sonnets not read is 5
theorem sonnets_not_read : unheard_lines / sonnet_lines = 5 := by
  sorry

end sonnets_not_read_l808_808963


namespace combined_bus_capacity_eq_40_l808_808610

theorem combined_bus_capacity_eq_40 (train_capacity : ℕ) (fraction : ℚ) (num_buses : ℕ) 
  (h_train_capacity : train_capacity = 120)
  (h_fraction : fraction = 1/6)
  (h_num_buses : num_buses = 2) :
  num_buses * (train_capacity * fraction).toNat = 40 := by
  sorry

end combined_bus_capacity_eq_40_l808_808610


namespace greatest_integer_l808_808275

-- Define the conditions for the problem
def isMultiple4 (n : ℕ) : Prop := n % 4 = 0
def notMultiple8 (n : ℕ) : Prop := n % 8 ≠ 0
def notMultiple12 (n : ℕ) : Prop := n % 12 ≠ 0
def gcf4 (n : ℕ) : Prop := Nat.gcd n 24 = 4
def lessThan200 (n : ℕ) : Prop := n < 200

-- State the main theorem
theorem greatest_integer : ∃ n : ℕ, lessThan200 n ∧ gcf4 n ∧ n = 196 :=
by
  sorry

end greatest_integer_l808_808275


namespace units_digit_of_7_pow_6_pow_5_l808_808428

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808428


namespace minimum_common_perimeter_exists_l808_808257

noncomputable def find_minimum_perimeter
  (a b x : ℕ) 
  (is_int_sided_triangle_1 : 2 * a + 20 * x = 2 * b + 25 * x)
  (is_int_sided_triangle_2 : 20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2))) 
  (base_ratio : 20 * 2 * (a - b) = 25 * 2 * (a - b)): ℕ :=
2 * a + 20 * (2 * (a - b))

-- The final goal should prove the minimum perimeter under the given conditions.
theorem minimum_common_perimeter_exists :
∃ (minimum_perimeter : ℕ), 
  (∀ (a b x : ℕ), 
    2 * a + 20 * x = 2 * b + 25 * x → 
    20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2)) → 
    20 * 2 * (a - b) = 25 * 2 * (a - b) → 
    minimum_perimeter = 2 * a + 20 * x) :=
sorry

end minimum_common_perimeter_exists_l808_808257


namespace incorrectStatementB_l808_808705

-- Definitions based on the problem conditions
def parallel (l1 l2 : ℕ) := ∀ x, altInteriorAnglesEqual(x) ∧ correspondingAnglesEqual(x)

def altInteriorAnglesEqual (angles : ℕ) := true
def correspondingAnglesEqual (angles : ℕ) := true
def verticalAnglesEqual (angles : ℕ) := true
def transitiveParallel (l1 l2 l3 : ℕ) := parallel l1 l3 → parallel l2 l3 → parallel l1 l2

-- The theorem stating that Statement B is incorrect
theorem incorrectStatementB : ¬(∀ x, parallel x x ↔ correspondingAnglesEqual x) := sorry

end incorrectStatementB_l808_808705


namespace exists_t_constant_polynomial_l808_808038

-- Definitions as per the problem statement
def S (k : ℕ) : ℕ := k.digits.sum  -- Sum of the decimal digits of a natural number

-- Polynomials with non-negative integer coefficients
def P (x : ℕ) : ℕ := sorry
def Q (x : ℕ) : ℕ := sorry

-- Given condition: For all non-negative integers n
axiom S_P_eq_S_Q : ∀ (n : ℕ), S (P n) = S (Q n)

-- Proof goal: There exists an integer t such that P(x) - 10^t * Q(x) is a constant polynomial
theorem exists_t_constant_polynomial : ∃ t : ℤ, ∀ x : ℕ, (P x - 10^t * Q x) = k := sorry


end exists_t_constant_polynomial_l808_808038


namespace num_letters_dot_not_straight_line_l808_808718

variable (Total : ℕ)
variable (DS : ℕ)
variable (S_only : ℕ)
variable (D_only : ℕ)

theorem num_letters_dot_not_straight_line 
  (h1 : Total = 40) 
  (h2 : DS = 11) 
  (h3 : S_only = 24) 
  (h4 : Total - S_only - DS = D_only) : 
  D_only = 5 := 
by 
  sorry

end num_letters_dot_not_straight_line_l808_808718


namespace triangle_medians_perpendicular_right_triangle_l808_808993

-- Defining the problem conditionally as a theorem
theorem triangle_medians_perpendicular_right_triangle
  (D E F P Q : Point) -- Points defining the triangle and medians
  (h_triangle : RightTriangle D E F ∧ ∠D = 90°) -- Triangle DEF is right at D
  (h_medians : Medians DP EQ) -- DP and EQ are medians
  (h_perpendicular_medians : Perpendicular DP EQ) -- Medians DP and EQ are perpendicular
  (DP_length : Distance D P = 27) -- Length of median DP
  (EQ_length : Distance E Q = 30) -- Length of median EQ
  : Distance D E = 2 * sqrt 181 := 
sorry

end triangle_medians_perpendicular_right_triangle_l808_808993


namespace exists_triangle_with_sides_equal_and_parallel_l808_808304

variables {A1 A2 A3 A4 A5 A6 M1 M2 M3 M4 M5 M6 : Point}

-- Assuming convex hexagon and midpoints conditions
def is_midpoint (M A B : Point) : Prop := 2 • M = A + B

-- Definitions of midpoints conditions
axiom M1_midpoint : is_midpoint M1 A1 A2
axiom M2_midpoint : is_midpoint M2 A2 A3
axiom M3_midpoint : is_midpoint M3 A3 A4
axiom M4_midpoint : is_midpoint M4 A4 A5
axiom M5_midpoint : is_midpoint M5 A5 A6
axiom M6_midpoint : is_midpoint M6 A6 A1

-- Main theorem to prove
theorem exists_triangle_with_sides_equal_and_parallel :
  \rightarrow{M1 M2} + \rightarrow{M3 M4} + \rightarrow{M5 M6} = \rightarrow{0} :=
sorry

end exists_triangle_with_sides_equal_and_parallel_l808_808304


namespace farmer_total_acres_l808_808750

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l808_808750


namespace triangle_inequality_sqrt_AMGM_l808_808479

theorem triangle_inequality_sqrt_AMGM (a b c : ℝ)
  (h1: a > 0) (h2: b > 0) (h3: c > 0)
  (h_triangle : a + b > c) (h_triangle2 : b + c > a) (h_triangle3 : c + a > b) :
  √(a/(b+c-a)) + √(b/(c+a-b)) + √(c/(a+b-c)) ≥ 3 := sorry

end triangle_inequality_sqrt_AMGM_l808_808479


namespace exp_ge_e_for_all_x_l808_808703

theorem exp_ge_e_for_all_x (x : ℝ) : exp x ≥ exp 1 :=
by sorry

end exp_ge_e_for_all_x_l808_808703


namespace Elle_practice_time_l808_808449

def minutes_per_weekday : ℕ := 30
def days_weekdays : ℕ := 5
def times_more_on_saturday : ℕ := 3

theorem Elle_practice_time :
  let total_weekdays := minutes_per_weekday * days_weekdays,
      saturday_time := times_more_on_saturday * minutes_per_weekday,
      total_minutes := total_weekdays + saturday_time in
  total_minutes / 60 = 4 := 
by
  sorry

end Elle_practice_time_l808_808449


namespace non_overlapping_sets_l808_808862

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808862


namespace nums_between_2000_and_3000_div_by_360_l808_808522

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l808_808522


namespace three_digit_number_divisibility_four_digit_number_divisibility_l808_808123

-- Definition of three-digit number
def is_three_digit_number (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

-- Definition of four-digit number
def is_four_digit_number (b : ℕ) : Prop := 1000 ≤ b ∧ b ≤ 9999

-- First proof problem
theorem three_digit_number_divisibility (a : ℕ) (h : is_three_digit_number a) : 
  (1001 * a) % 7 = 0 ∧ (1001 * a) % 11 = 0 ∧ (1001 * a) % 13 = 0 := 
sorry

-- Second proof problem
theorem four_digit_number_divisibility (b : ℕ) (h : is_four_digit_number b) : 
  (10001 * b) % 73 = 0 ∧ (10001 * b) % 137 = 0 := 
sorry

end three_digit_number_divisibility_four_digit_number_divisibility_l808_808123


namespace sum_even_integers_12_to_40_l808_808287

theorem sum_even_integers_12_to_40 : 
  ∑ k in finset.filter (λ k, even k) (finset.range 41), k = 390 := by
  sorry

end sum_even_integers_12_to_40_l808_808287


namespace expand_product_l808_808376

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x ^ 2 + 52 * x + 84 := by
  sorry

end expand_product_l808_808376


namespace determine_number_on_reverse_side_l808_808603

variable (n : ℕ) (k : ℕ) (shown_cards : ℕ → Prop)

theorem determine_number_on_reverse_side :
    -- Conditions
    (∀ i, 1 ≤ i ∧ i ≤ n → (shown_cards (i - 1) ↔ shown_cards i)) →
    -- Prove
    (k = 0 ∨ k = n ∨ (1 ≤ k ∧ k < n ∧ (shown_cards (k - 1) ∨ shown_cards (k + 1)))) →
    (∃ j, (j = 1 ∧ k = 0) ∨ (j = n - 1 ∧ k = n) ∨ 
          (j = k - 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k + 1)) ∨ 
          (j = k + 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k - 1))) :=
by
  sorry

end determine_number_on_reverse_side_l808_808603


namespace total_students_in_class_l808_808216

theorem total_students_in_class 
  (avg_age_all : ℝ)
  (num_students1 : ℕ) (avg_age1 : ℝ)
  (num_students2 : ℕ) (avg_age2 : ℝ)
  (age_student17 : ℕ)
  (total_students : ℕ) :
  avg_age_all = 17 →
  num_students1 = 5 →
  avg_age1 = 14 →
  num_students2 = 9 →
  avg_age2 = 16 →
  age_student17 = 75 →
  total_students = num_students1 + num_students2 + 1 →
  total_students = 17 :=
by
  intro h_avg_all h_num1 h_avg1 h_num2 h_avg2 h_age17 h_total
  -- Additional proof steps would go here
  sorry

end total_students_in_class_l808_808216


namespace product_of_nonreal_roots_l808_808389

theorem product_of_nonreal_roots :
  let f := (λ x : ℂ, x^4 - 6*x^3 + 15*x^2 - 20*x - 2005)
  in ( ∀ x : ℂ, f x = 0 → x = (2 + complex.I * complex.sqrt (complex.ofReal 2021) ^ (1 / 4)) ∨ x = (2 - complex.I * complex.sqrt (complex.ofReal 2021) ^ (1 / 4)) ∨ x = (2 + complex.sqrt (complex.ofReal 2021) ^ (1 / 4)) ∨ x = (2 - complex.sqrt (complex.ofReal 2021) ^ (1 / 4)) ) →
     ((2 + complex.I * complex.sqrt (complex.ofReal 2021) ^ (1 / 4)) * (2 - complex.I * complex.sqrt (complex.ofReal 2021) ^ (1 / 4))) = 4 + real.sqrt 2021 :=
by
  sorry

end product_of_nonreal_roots_l808_808389


namespace pentagon_area_l808_808561

-- Define the variables and conditions
variables (PQ QR ST RS area : ℝ)
noncomputable def perimeter := 82
noncomputable def PQ_val := 13
noncomputable def QR_val := 18
noncomputable def ST_val := 30

-- Define the angles
constant angle_QRS : ℝ := 90
constant angle_RST : ℝ := 90
constant angle_STP : ℝ := 90

-- Define the problem statement
theorem pentagon_area (hPQ : PQ = PQ_val) (hQR : QR = QR_val) (hST : ST = ST_val)
  (h_perimeter: PQ + QR + RS + ST + (RS + 5) = perimeter)
  (h_angles : angle_QRS = 90 ∧ angle_RST = 90 ∧ angle_STP = 90) :
  area = 270 :=
sorry

end pentagon_area_l808_808561


namespace jamal_books_l808_808153

variable (hist : ℕ) (fict : ℕ) (child : ℕ) (wrong_books : ℕ) (left_to_shelve : ℕ)

theorem jamal_books : hist = 12 → fict = 19 → child = 8 → wrong_books = 4 → left_to_shelve = 16 → 
  let total_shelved := hist + fict + child,
      total_with_wrong := total_shelved + wrong_books,
      start_books := total_with_wrong + left_to_shelve
  in start_books = 59 :=
by
  intros; sorry

end jamal_books_l808_808153


namespace impossibility_of_domino_tiling_l808_808152

-- Define the size and shape of the chessboard and the dominoes
def board_size : ℕ := 8
def domino_length : ℕ := 2
def domino_width : ℕ := 1

-- Define a function to denote a valid tiling 
def valid_domino_tiling (positions : list (ℕ × ℕ)) : Prop :=
  ∀ d1 d2 ∈ positions, (d1 ≠ d2) → 
    ¬(d1.1 = d2.1 ∧ d1.2 = d2.2) ∧ 
    ∀ {i j : ℕ}, i < board_size → j < board_size → 
    (• (positions.contains (i, j) ∨ positions.contains (i+1, j) ∨ positions.contains (i, j+1)) → 
       (positions.contains (i+1, j+1) ∨ positions.contains (i+1, j-1) ∨ positions.contains (i-1, j+1) ∨ positions.contains (i-1, j-1) → False))
    
-- State the contradiction
theorem impossibility_of_domino_tiling : 
  ∀ (positions : list (ℕ × ℕ)), positions.length = (board_size * board_size) / (domino_length * domino_width) →
  valid_domino_tiling positions →
  False :=
begin
  sorry
end

end impossibility_of_domino_tiling_l808_808152


namespace football_attendance_l808_808677

open Nat

theorem football_attendance:
  (Saturday Monday Wednesday Friday expected_total actual_total: ℕ)
  (h₀: Saturday = 80)
  (h₁: Monday = Saturday - 20)
  (h₂: Wednesday = Monday + 50)
  (h₃: Friday = Saturday + Monday)
  (h₄: expected_total = 350)
  (h₅: actual_total = Saturday + Monday + Wednesday + Friday) :
  actual_total = expected_total + 40 :=
  sorry

end football_attendance_l808_808677


namespace dan_blue_marbles_l808_808361

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end dan_blue_marbles_l808_808361


namespace fraction_of_roll_per_present_l808_808197

theorem fraction_of_roll_per_present (total_fraction : ℚ) (num_presents : ℕ) (same_amount : Prop) :
  total_fraction = 1 / 2 →
  num_presents = 5 →
  same_amount →
  (total_fraction / num_presents = 1 / 10) :=
by
  intros h1 h2 h3
  rw [h1, h2]
  norm_num
  sorry

end fraction_of_roll_per_present_l808_808197


namespace non_overlapping_original_sets_exists_l808_808871

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808871


namespace motzkin_7_l808_808815

theorem motzkin_7 : ∃ M : ℕ → ℕ, M(7) = 127 ∧ M(0) = 1 ∧ M(1) = 1 ∧ M(2) = 2 ∧
  ∀ n, n ≥ 3 → M(n) = (M(n-1) + ∑ k in finset.range(n-1), M(k) * M(n-2-k)) :=
  by {
    let M : ℕ → ℕ := sorry, -- placeholder for the actual recursive definition
    use M,
    split,
    { -- Proof that M(7) = 127; requires full recursive definition and steps
      sorry
    },
    split,
    { -- Proof that M(0) = 1
      sorry
    },
    split,
    { -- Proof that M(1) = 1
      sorry
    },
    split,
    { -- Proof that M(2) = 2
      sorry
    },
    { -- Proof that M(n) satisfies the recursive formula
      assume n hn,
      sorry
    }
  }

end motzkin_7_l808_808815


namespace quadrilateral_inequality_l808_808722

theorem quadrilateral_inequality 
  (m n a b c d : ℝ)
  (h₁ : 0 < m) (h₂ : 0 < n)
  (h₃ : a^2 + b^2 + c^2 + d^2 ≤ 2 * (m^2 + n^2))
  (h₄ : m /= 0 ∧ n /= 0) :
  1 ≤ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ∧
  (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ≤ 2 :=
sorry

end quadrilateral_inequality_l808_808722


namespace exists_n_consecutive_good_numbers_l808_808202

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_good (n : ℕ) : Prop :=
  n % sum_of_digits n ≠ 0

theorem exists_n_consecutive_good_numbers (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, ∀ i : ℕ, i < n → is_good (k + i) :=
begin
  sorry
end

end exists_n_consecutive_good_numbers_l808_808202


namespace necessary_not_sufficient_condition_l808_808075

-- Definitions from conditions
variables {a : ℝ}

-- Lean statement
theorem necessary_not_sufficient_condition (h_real : a ∈ ℝ) (h_condition : a < 1) : ¬(a < 1 → 1/a > 1) ∧ (1/a > 1 → a < 1) :=
by
  sorry

end necessary_not_sufficient_condition_l808_808075


namespace three_largest_divisors_sum_1457_l808_808836

/-- Definition to capture the three largest divisors sum condition -/
def sum_of_three_largest_divisors (n : ℕ) : ℕ :=
  let divisors := (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).val.sort (≥);
  divisors.take 3 |>.sum

/-- The target statement to be proved -/
theorem three_largest_divisors_sum_1457 :
  { n : ℕ | sum_of_three_largest_divisors n = 1457 } = {987, 1023, 1085, 1175} :=
by
  sorry

end three_largest_divisors_sum_1457_l808_808836


namespace impossible_domino_covering_l808_808149

def Chessboard (n : Nat) := Fin n × Fin n
def Domino := Fin 2

def covers (dom : Domino) (cell : Chessboard 8) : Prop := 
  -- Define the covering relationship of domino on the chessboard cell here
  sorry

def forms_2x2_square (dom1 dom2 : Domino) : Prop := 
  -- Define the condition where two dominoes form a 2x2 square
  sorry

theorem impossible_domino_covering : 
  ¬∃ (f : Chessboard 8 → option Domino), 
    (∀ cell, ∃ d, f cell = some d) ∧ 
    (∀ cell1 cell2, f cell1 = f cell2 → covers (f cell1) cell1 ∧ covers (f cell2) cell2) ∧ 
    (∀ dom1 dom2, forms_2x2_square dom1 dom2 → dom1 ≠ dom2) :=
sorry

end impossible_domino_covering_l808_808149


namespace daily_wage_of_c_l808_808711

-- Define the conditions
variables (a b c : ℝ)
variables (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5)
variables (h_days : 6 * a + 9 * b + 4 * c = 1702)

-- Define the proof problem; to prove c = 115
theorem daily_wage_of_c (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) (h_days : 6 * a + 9 * b + 4 * c = 1702) : 
  c = 115 :=
sorry

end daily_wage_of_c_l808_808711


namespace goldfish_problem_l808_808177

theorem goldfish_problem :
  ∀ (d : ℕ), (∀ (initial_goldfish weekly_purchase remaining_goldfish after_weeks : ℕ),
    initial_goldfish = 18 →
    weekly_purchase = 3 →
    remaining_goldfish = 4 →
    after_weeks = 7 →
    initial_goldfish + weekly_purchase * after_weeks - d * after_weeks = remaining_goldfish →
    d = 5) := 
by
  intros d initial_goldfish weekly_purchase remaining_goldfish after_weeks
  intros h_initial h_weekly h_remaining h_weeks h_eq
  rw [h_initial, h_weekly, h_remaining, h_weeks] at h_eq
  linarith

end goldfish_problem_l808_808177


namespace y_value_at_x_50_l808_808235

theorem y_value_at_x_50 
  (h1 : ∃ m b, ∀ x y, (x, y) = (10, 30) → y = m * x + b)
  (h2 : ∃ m b, ∀ x y, (x, y) = (15, 45) → y = m * x + b)
  (h3 : ∃ m b, ∀ x y, (x, y) = (20, 60) → y = m * x + b) :
  ∃ y, y = 3 * 50 :=
begin
  sorry
end

end y_value_at_x_50_l808_808235


namespace zachary_more_crunches_than_pushups_l808_808709

def zachary_pushups : ℕ := 46
def zachary_crunches : ℕ := 58
def zachary_crunches_more_than_pushups : ℕ := zachary_crunches - zachary_pushups

theorem zachary_more_crunches_than_pushups : zachary_crunches_more_than_pushups = 12 := by
  sorry

end zachary_more_crunches_than_pushups_l808_808709


namespace line_eq_l808_808762

theorem line_eq (x_1 y_1 x_2 y_2 : ℝ) (h1 : x_1 + x_2 = 8) (h2 : y_1 + y_2 = 2)
  (h3 : x_1^2 - 4 * y_1^2 = 4) (h4 : x_2^2 - 4 * y_2^2 = 4) :
  ∃ l : ℝ, ∀ x y : ℝ, x - y - 3 = l :=
by sorry

end line_eq_l808_808762


namespace find_b_minus_a_l808_808234

noncomputable def rotate_90_counterclockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc + (-(y - yc)), yc + (x - xc))

noncomputable def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem find_b_minus_a (a b : ℝ) :
  let xc := 2
  let yc := 3
  let P := (a, b)
  let P_rotated := rotate_90_counterclockwise a b xc yc
  let P_reflected := reflect_about_y_eq_x P_rotated.1 P_rotated.2
  P_reflected = (4, 1) →
  b - a = 1 :=
by
  intros
  sorry

end find_b_minus_a_l808_808234


namespace find_principal_l808_808297

noncomputable def principal : ℝ :=
  let P := 2600 in
  let SI_formula := P * 4 * 5 / 100 in
  let SI_condition := P - 2080 in
  if SI_formula = SI_condition then
    P
  else
    0

theorem find_principal :
  principal = 2600 :=
sorry

end find_principal_l808_808297


namespace det_A_l808_808025

open Matrix

def A (z : ℤ) : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![z + 2, z, z],
    ![z, z + 2, z + 1],
    ![z, z + 1, z + 2]]

theorem det_A (z : ℤ) : det (A z) = 3 * z^2 + 9 * z + 9 := by
  sorry

end det_A_l808_808025


namespace number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l808_808847

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime 
  (n : ℕ) (h : n ≥ 2) : (∃ (a b : ℕ), a ≠ b ∧ is_prime (a^3 + 2) ∧ is_prime (b^3 + 2)) :=
by
  sorry

end number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l808_808847


namespace distinct_values_sum_l808_808654

/-- The number of distinct possible values of x + y given the condition (x - 1) ^ 2 + (y - 1) ^ 2 ≤ 1 -/
theorem distinct_values_sum (x y : ℤ) (h : (x - 1) ^ 2 + (y - 1) ^ 2 ≤ 1) :
  ∃ S : set ℤ, ∀ x y : ℤ, (x - 1) ^ 2 + (y - 1) ^ 2 ≤ 1 → S = {x + y} ∧ S.card = 3 :=
sorry

end distinct_values_sum_l808_808654


namespace log_defined_for_x_gt_2002_pow_2004_l808_808370

-- Definition of the conditions and the theorem statement
theorem log_defined_for_x_gt_2002_pow_2004 :
  ∀ x : ℝ, x > 2002^2004 →
  ∃ y : ℝ, (log 2005 (log 2004 (log 2003 (log 2002 x)))) = y :=
by
  intros x hx
  sorry

end log_defined_for_x_gt_2002_pow_2004_l808_808370


namespace odd_function_sum_zero_l808_808125

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

theorem odd_function_sum_zero (g : ℝ → ℝ) (a : ℝ) (h_odd : odd_function g) : 
  g a + g (-a) = 0 :=
by 
  sorry

end odd_function_sum_zero_l808_808125


namespace units_digit_of_7_pow_6_pow_5_l808_808439

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808439


namespace bags_per_trip_l808_808570

theorem bags_per_trip
  (trips_per_day : ℕ)
  (total_bags : ℕ)
  (days : ℕ)
  (trips_per_day = 20)
  (total_bags = 1000)
  (days = 5)
  : total_bags / (days * trips_per_day) = 10 := 
sorry

end bags_per_trip_l808_808570


namespace sqrt_prod_simplified_l808_808805

open Real

variable (x : ℝ)

theorem sqrt_prod_simplified (hx : 0 ≤ x) : sqrt (50 * x) * sqrt (18 * x) * sqrt (8 * x) = 30 * x * sqrt (2 * x) :=
by
  sorry

end sqrt_prod_simplified_l808_808805


namespace sum_of_first_3m_terms_l808_808666

variable (S : ℕ → ℕ)

def is_arithmetic_sequence (m : ℕ) :=
  (S m = 30) ∧ (S (2 * m) = 100)

theorem sum_of_first_3m_terms (m : ℕ) (h : is_arithmetic_sequence S m) : S (3 * m) = 210 :=
by
  -- Assuming the conditions hold 
  cases h with h1 h2
  -- Here we would write the proof steps if necessary, but it's excluded as per the instructions
  -- Proof omitted with sorry
  sorry

end sum_of_first_3m_terms_l808_808666


namespace store_total_income_l808_808550

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l808_808550


namespace exists_perpendicular_line_l808_808061

theorem exists_perpendicular_line (α : set (point × vector)) (l : set point) : 
  ∃ m ∈ α, is_perpendicular m l :=
sorry

end exists_perpendicular_line_l808_808061


namespace last_digit_of_decimal_expansion_l808_808698

theorem last_digit_of_decimal_expansion (n : ℕ) (h1 : n = 2^15 * 3) : 
  (∃ d : ℕ, d < 10 ∧ (∃ (k : ℚ), k / 10^15 = (5^15 : ℚ) * (10^(-15 : ℚ)) * (1 / 3 : ℚ) ∧ (k.floor()).digit 0 = d) → d = 6) := by
  sorry

end last_digit_of_decimal_expansion_l808_808698


namespace triangle_stability_l808_808312

theorem triangle_stability
    (triangle_has_stability : ∀ (T : Type) (a b c : T), triangle a b c → stable a b c) :
    (∀ (B : Type) (x : B) (stand : B), triangle stand → supported_by x stand → stands_firmly x) :=
by
  intros B x stand h_triangle h_supported
  have h_stability := triangle_has_stability _ _ _ _ h_triangle
  sorry

end triangle_stability_l808_808312


namespace percent_girls_in_class_l808_808995

/-
Given that the ratio of boys to girls in Mr. Smith's history class is 3:4 and there are 42 students in total,
we aim to prove that 57.14% of the students are girls.
-/
theorem percent_girls_in_class 
  (total_students : ℕ) 
  (ratio_boys_girls : ℕ × ℕ) 
  (total_ratio : ratio_boys_girls.fst + ratio_boys_girls.snd = 7) 
  (total_students_ratio : 42 / 7 = 6)
  (group_girls : ratio_boys_girls.snd = 4) 
  : (4 * 6 / total_students) * 100 = 57.14 := 
by
  simp
  norm_num
  sorry

end percent_girls_in_class_l808_808995


namespace find_a_n_l808_808463

-- Definitions from the conditions
def seq (a : ℕ → ℤ) : Prop :=
  ∀ n, (3 - a (n + 1)) * (6 + a n) = 18

-- The Lean statement of the problem
theorem find_a_n (a : ℕ → ℤ) (h_a0 : a 0 ≠ 3) (h_seq : seq a) :
  ∀ n, a n = 2 ^ (n + 2) - n - 3 :=
by
  sorry

end find_a_n_l808_808463


namespace units_digit_pow_7_6_5_l808_808418

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808418


namespace incorrect_operation_l808_808289

noncomputable def a : ℤ := -2

def operation_A (a : ℤ) : ℤ := abs a
def operation_B (a : ℤ) : ℤ := abs (a - 2) + abs (a + 1)
def operation_C (a : ℤ) : ℤ := -a ^ 3 + a + (-a) ^ 2
def operation_D (a : ℤ) : ℤ := abs a ^ 2

theorem incorrect_operation :
  operation_D a ≠ abs 4 :=
by
  sorry

end incorrect_operation_l808_808289


namespace min_value_eq_floor_div_two_l808_808469

noncomputable def min_possible_value (n : ℕ) (h : 2 ≤ n) 
  (a : Fin n → ℕ) (x y : Fin n → ℕ) : ℕ := 
  if ∀ i, 1 ≤ i.val ∧ i.val ≤ n → (x i = max_length_increasing_subsequence (a i) ∧ y i = max_length_decreasing_subsequence (a i)) 
  then ∑ i in finset.range n, | x i - y i |
  else 0 -- if the condition is not met, return 0

theorem min_value_eq_floor_div_two (n : ℕ) (h2 : 2 ≤ n) (a : Fin n → ℕ) (x y : Fin n → ℕ)
(hxy: ∀ i, 1 ≤ i.val ∧ i.val ≤ n → (x i = max_length_increasing_subsequence (a i) ∧ y i = max_length_decreasing_subsequence (a i))):
  min_possible_value n h2 a x y = n / 2 :=
begin
  sorry
end

end min_value_eq_floor_div_two_l808_808469


namespace reduced_price_l808_808712

noncomputable def reduced_price_per_dozen (P : ℝ) : ℝ := 12 * (P / 2)

theorem reduced_price (X P : ℝ) (h1 : X * P = 50) (h2 : (X + 50) * (P / 2) = 50) : reduced_price_per_dozen P = 6 :=
sorry

end reduced_price_l808_808712


namespace cory_fruit_eating_order_l808_808360

theorem cory_fruit_eating_order :
  let total_fruits := 8
  let apples := 4
  let oranges := 2
  let bananas := 2
  let total_arrangements := Nat.factorial total_fruits / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas)
  let apple_banana_permutation := Nat.choose (apples + bananas) apples
  in total_arrangements * apple_banana_permutation = 18900 :=
by
  let total_fruits := 8
  let apples := 4
  let oranges := 2
  let bananas := 2
  let total_arrangements := Nat.factorial total_fruits / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas)
  let apple_banana_permutation := Nat.choose (apples + bananas) apples
  show total_arrangements * apple_banana_permutation = 18900
  sorry

end cory_fruit_eating_order_l808_808360


namespace part1_part2a_part2b_l808_808505

-- Definitions of vectors a and b
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1/2, real.sqrt 3 / 2)

-- Dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Euclidean norm of a 2D vector
def norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

-- Calculate cosine of angle between vectors a and b when m = -sqrt(3)
def cos_angle (m : ℝ) : ℝ := 
  dot_product (a m) b / (norm (a m) * norm b)

-- Main theorems
theorem part1 (m : ℝ) (h : m = -real.sqrt 3) : 
  real.acos (cos_angle m) = 5 * real.pi / 6 :=
sorry

theorem part2a (h : dot_product (a m) b = 0) : 
  m = real.sqrt 3 :=
sorry

theorem part2b (k t : ℝ) (h1 : t ≠ 0) (h2 : dot_product (a (real.sqrt 3) + ((t^2 - 3) • b)) ((-k • a (real.sqrt 3) + t • b) = 0)) : 
  ∃ k t, (4 * k = t * (t^2 - 3) ∧ (t ≠ 0) ∧ ((k + t^2) / t) = -7/4) :=
sorry

end part1_part2a_part2b_l808_808505


namespace problem_proof_l808_808924

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (q : ℝ) (a₁ : ℝ)

-- Conditions
def isGeometricSeq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def sumOfFirstNTerms (S : ℕ → ℝ) (a₁ : ℝ) (q : ℝ) := 
  ∀ n, S n = if q = 1 then n * a₁ else a₁ * (1 - q ^ n) / (1 - q)
def givenCondition : Prop :=
  2 * a 2019 = a 2020 + a 2021

-- Target
theorem problem_proof 
  (h_geo : isGeometricSeq a q) 
  (h_sum : sumOfFirstNTerms S a₁ q)
  (h_cond : givenCondition):
  S 2022 + S 2023 = 2 * S 2021 :=
sorry -- Proof goes here

end problem_proof_l808_808924


namespace simplify_fraction_l808_808529

theorem simplify_fraction (a b x : ℝ) (h₁ : x = a / b) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) :=
sorry

end simplify_fraction_l808_808529


namespace circle_line_intersection_l808_808562

/-- The Cartesian equation of circle C is (x - 2)² + y² = 4.
    The parametric equation of line l is:
    x = 5 + (3/5)*t
    y = 6 + (4/5)*t 
    We need to prove that |MA| + |MB| = 66/5 if the line intersects the circle at points A and B. -/
theorem circle_line_intersection 
  (x y : ℝ) (t : ℝ) (slope : ℝ := 4/3) 
  (circle_eq : (x - 2)^2 + y^2 = 4) 
  (line_eq : (x, y) = (5 + (3/5)*t, 6 + (4/5)*t)) :
  |MA| + |MB| = 66 / 5 := 
sorry

end circle_line_intersection_l808_808562


namespace range_of_a_l808_808533

theorem range_of_a (a : ℝ) : 
  (∃ n : ℕ, (∀ x : ℕ, 1 ≤ x → x ≤ 5 → x < a) ∧ (∀ y : ℕ, x ≥ 1 → y ≥ 6 → y ≥ a)) ↔ (5 < a ∧ a < 6) :=
by
  sorry

end range_of_a_l808_808533


namespace find_n_from_ratio_l808_808082

theorem find_n_from_ratio (a b n : ℕ) (h : (a + 3 * b) ^ n = 4 ^ n)
  (h_ratio : 4 ^ n / 2 ^ n = 64) : 
  n = 6 := 
by
  sorry

end find_n_from_ratio_l808_808082


namespace pairing_sums_perfect_square_l808_808379

theorem pairing_sums_perfect_square (n : ℕ) (h : n > 1) :
  ∃ P : Fin (2 * n) → Fin (2 * n), (∀ i, i < n → (P (2 * i) + P (2 * i + 1))^2) :=
sorry

end pairing_sums_perfect_square_l808_808379


namespace sequence_satisfies_recurrence_l808_808837

theorem sequence_satisfies_recurrence (n : ℕ) (a : ℕ → ℕ) (h : ∀ k, 2 ≤ k → k ≤ n - 1 → a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1) :
  n = 3 ∨ n = 4 := by
  sorry

end sequence_satisfies_recurrence_l808_808837


namespace discounted_price_is_correct_l808_808735

def original_price_of_cork (C : ℝ) : Prop :=
  C + (C + 2.00) = 2.10

def discounted_price_of_cork (C : ℝ) : ℝ :=
  C - (C * 0.12)

theorem discounted_price_is_correct :
  ∃ C : ℝ, original_price_of_cork C ∧ discounted_price_of_cork C = 0.044 :=
by
  sorry

end discounted_price_is_correct_l808_808735


namespace max_y_coordinate_cos2theta_l808_808840

noncomputable def polar_to_cartesian_y (theta : ℝ) : ℝ :=
  (Real.cos (2 * theta)) * (Real.sin theta)

theorem max_y_coordinate_cos2theta :
  ∃ (y_max : ℝ), y_max = (√2 / 2) ∧ ∀ (theta : ℝ), polar_to_cartesian_y theta ≤ y_max :=
sorry

end max_y_coordinate_cos2theta_l808_808840


namespace f_odd_and_increasing_l808_808057

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)

-- Add the conditions as hypotheses
axiom additivity : ∀ x y : ℝ, f (x + y) = f (x) + f (y)
axiom positivity : ∀ x : ℝ, x > 0 → f x > 0

-- State the theorem to be proved
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
sorry

end f_odd_and_increasing_l808_808057


namespace eccentricity_of_ellipse_l808_808223

theorem eccentricity_of_ellipse : 
  let a := 5
  let b := 4
  let c := 3
  let e := c / a
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 16 = 1) → e = 3 / 5 := 
by
  intros x y hyp
  have h1 : a = 5 := rfl
  have h2 : b = 4 := rfl
  have h3 : c = 3 := rfl
  have h4 : e = 3 / 5 := by simp [e, c, a]
  sorry

end eccentricity_of_ellipse_l808_808223


namespace bathroom_area_is_eight_l808_808310

def bathroomArea (length width : ℕ) : ℕ :=
  length * width

theorem bathroom_area_is_eight : bathroomArea 4 2 = 8 := 
by
  -- Proof omitted.
  sorry

end bathroom_area_is_eight_l808_808310


namespace non_overlapping_original_sets_l808_808885

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808885


namespace fraction_of_robs_doubles_is_one_third_l808_808195

theorem fraction_of_robs_doubles_is_one_third 
  (total_robs_cards : ℕ) (total_jess_doubles : ℕ) 
  (times_jess_doubles_robs : ℕ)
  (robs_doubles : ℕ) :
  total_robs_cards = 24 →
  total_jess_doubles = 40 →
  times_jess_doubles_robs = 5 →
  total_jess_doubles = times_jess_doubles_robs * robs_doubles →
  (robs_doubles : ℚ) / total_robs_cards = 1 / 3 := 
by 
  intros h1 h2 h3 h4
  sorry

end fraction_of_robs_doubles_is_one_third_l808_808195


namespace find_value_of_p_l808_808931

theorem find_value_of_p (a b p : ℝ) (h1 : a^2 = 2 * p * b)
  (h2 : 0 < p)
  (h3 : a = 5 * real.sqrt 3)
  (h4 : b = 15)
  (h5 : ∃ A B : ℝ × ℝ, A ≠ B ∧ is_circle (0, 10) (dist (0, 0) A) A ∧ is_circle (0, 10) (dist (0, 0) B) B ∧ 
    ∃ equilateral : ℝ × ℝ × ℝ → Prop, equilateral ⟨A, B, (0,0)⟩) :
  p = 5 / 2 := 
sorry

end find_value_of_p_l808_808931


namespace find_b_l808_808649

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (1/12)*x^2 + a*x + b

def intersects (f : ℝ → ℝ) (x_val y_val : ℝ) : Prop := f x_val = y_val

def equidistant (p1 p2 p3 : ℝ × ℝ) (p : ℝ × ℝ) :=
  dist p p1 = dist p p2 ∧ dist p p2 = dist p p3

theorem find_b (a : ℝ) (b : ℝ) (x1 x2 : ℝ)
  (HA : intersects (f x a b) x1 0)
  (HC : intersects (f x a b) x2 0)
  (HB : intersects (f x a b) 0 b)
  (T : (ℝ × ℝ) := (3, 3))
  (H : equidistant (x1, 0) (0, b) (x2, 0) T) :
  b = -6 :=
sorry

end find_b_l808_808649


namespace price_of_chocolate_covered_pretzel_l808_808740

noncomputable def price_per_pretzel (revenue : ℝ) (fudge_weight fudge_price truffle_count truffle_price pretzel_count total_revenue : ℝ) : ℝ := 
(revenue - (fudge_weight * fudge_price + truffle_count * truffle_price)) / pretzel_count

theorem price_of_chocolate_covered_pretzel :
  let fudge_weight := 20
  let fudge_price := 2.50
  let truffle_count := 60  -- 5 dozen * 12 truffles/dozen
  let truffle_price := 1.50
  let pretzel_count := 36  -- 3 dozen * 12 pretzels/dozen
  let total_revenue := 212
  in price_per_pretzel total_revenue fudge_weight fudge_price truffle_count truffle_price pretzel_count total_revenue = 2 :=
by simp [price_per_pretzel]; sorry

end price_of_chocolate_covered_pretzel_l808_808740


namespace cyclotomic_polynomial_l808_808927

noncomputable def phi (n : ℕ) (x : ℂ) : ℂ :=
  ∏ d in (finset.range n).filter (λ d, nat.gcd d n = 1), (x - complex.exp (2 * real.pi * complex.I * d / n))

theorem cyclotomic_polynomial (p m : ℕ) (x : ℂ) [nat.prime p] : 
  (Phi_pm : ℂ → ℂ) :=
if h : p ∣ m then
    (phi m (x ^ p))
  else
    (phi m (x ^ p)) / (phi m x)

end cyclotomic_polynomial_l808_808927


namespace stationery_store_sales_l808_808546

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l808_808546


namespace units_digit_of_7_pow_6_pow_5_l808_808436

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808436


namespace minimum_common_perimeter_l808_808266

theorem minimum_common_perimeter :
  ∃ (a b c : ℕ), 
  let p := 2 * a + 10 * c in
  (a > b) ∧ 
  (b + 4c = a + 5c) ∧
  (5 * (a^2 - (5 * c)^2).sqrt = 4 * (b^2 - (4 * c)^2).sqrt) ∧
  p = 1180 :=
sorry

end minimum_common_perimeter_l808_808266


namespace prime_q_exists_l808_808632

theorem prime_q_exists (p : ℕ) (pp : Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ (∀ n, n > 0 → ¬ q ∣ n ^ p - p) := 
sorry

end prime_q_exists_l808_808632


namespace solve_inequality_l808_808634

theorem solve_inequality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 - 9) / (x^2 - 1) > 0 ↔ (x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1)) :=
sorry

end solve_inequality_l808_808634


namespace correct_pythagorean_triple_l808_808795

def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem correct_pythagorean_triple :
  (is_pythagorean_triple 1 2 3 = false) ∧ 
  (is_pythagorean_triple 4 5 6 = false) ∧ 
  (is_pythagorean_triple 6 8 9 = false) ∧ 
  (is_pythagorean_triple 7 24 25 = true) :=
by
  sorry

end correct_pythagorean_triple_l808_808795


namespace units_digit_7_pow_6_pow_5_l808_808411

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808411


namespace distinct_values_of_product_of_numbers_formed_from_digits_l808_808046

theorem distinct_values_of_product_of_numbers_formed_from_digits :
  ∃ n : ℕ, 
    (∀ A B : ℕ, 
    A ∈ {x : ℕ | 9999 < x ∧ x < 100000} → 
    B ∈ {x : ℕ | 9999 < x ∧ x < 100000} → 
    (∀ k : ℕ, (k = 7) → A + B = 111111 * k) → 
    n = 240) :=
begin
  sorry
end

end distinct_values_of_product_of_numbers_formed_from_digits_l808_808046


namespace employees_count_l808_808246

theorem employees_count (E M : ℝ) (h1 : M = 0.99 * E) (h2 : M - 299.9999999999997 = 0.98 * E) :
  E = 30000 :=
by sorry

end employees_count_l808_808246


namespace minimum_room_dimensions_l808_808305

-- Definitions of the dimensions and table properties
def table_width : ℝ := 9
def table_length : ℝ := 12
def table_diagonal : ℝ := real.sqrt (table_width ^ 2 + table_length ^ 2)

-- Conditions for the room dimensions
def room_length (S : ℝ) : Prop := S = 15
def room_width (T : ℝ) : Prop := T = 12

-- Proof goal: The dimensions of the room allow moving the table by rotating it 90 degrees.
theorem minimum_room_dimensions (S T : ℝ) (hS : room_length S) (hT : room_width T) : 
  table_diagonal <= S ∧ table_length <= T :=
by
  sorry

end minimum_room_dimensions_l808_808305


namespace minimum_blue_eyes_and_backpack_l808_808024

theorem minimum_blue_eyes_and_backpack (
  total_students : ℕ,
  blue_eye_students : ℕ,
  backpack_students : ℕ,
  glasses_students : ℕ,
  glasses_blue_eyes_students : ℕ
) (h1 : total_students = 35)
  (h2 : blue_eye_students = 18)
  (h3 : backpack_students = 25)
  (h4 : glasses_students = 10)
  (h5 : glasses_blue_eyes_students ≥ 2) :
  ∃ (min_blue_backpack_students : ℕ), min_blue_backpack_students = 10 :=
by {
  use 10,
  sorry
}

end minimum_blue_eyes_and_backpack_l808_808024


namespace calculate_f3_minus_f4_l808_808483

-- Defining the function f and the given conditions
variables (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (periodic_f : ∀ x, f (x + 2) = -f x)
variable (f1 : f 1 = 1)

-- Proving the required equality
theorem calculate_f3_minus_f4 : f 3 - f 4 = -1 :=
by
  sorry

end calculate_f3_minus_f4_l808_808483


namespace units_digit_7_pow_6_pow_5_l808_808424

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808424


namespace limit_of_sequence_l808_808618

noncomputable theory
open Classical

def sequence (n : ℕ) : ℝ := (1 - 2 * (n : ℝ)^2) / (2 + 4 * (n : ℝ)^2)

theorem limit_of_sequence :
  ∃ a : ℝ, a = -1/2 ∧ tendsto sequence at_top (𝓝 a) :=
by
  use -1/2
  split
  { refl }
  {
    -- Proof goes here by showing that the sequence tends to -1/2
    sorry
  }

end limit_of_sequence_l808_808618


namespace fraction_females_l808_808184

variable (y : ℝ)
variable (this_year_males : ℝ)
variable (this_year_females : ℝ)
variable (total_participants_this_year : ℝ)

axiom males_last_year : 15
axiom males_increase : 1.10
axiom females_last_year := y
axiom females_double : 2 * y
axiom total_increase : 1.15 * (15 + y)

theorem fraction_females :
  (2 * (15 / 17)) / ((1.10 * 15) + (2 * (15 / 17))) / (1.15 * (15 + y)) = 5 / 51 :=
by
  sorry

end fraction_females_l808_808184


namespace units_digit_pow_7_6_5_l808_808413

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808413


namespace rectangular_box_surface_area_l808_808667

theorem rectangular_box_surface_area 
  (a b c : ℝ)
  (h1 : 4 * a + 4 * b + 4 * c = 180)
  (h2 : real.sqrt (a^2 + b^2 + c^2) = 25)
  (h3 : c = 2 * a) : 
  2 * (a * b + b * c + c * a) = 1051.540 :=
by
  sorry

end rectangular_box_surface_area_l808_808667


namespace parabola_vertex_in_other_l808_808268

theorem parabola_vertex_in_other (p q a : ℝ) (h₁ : a ≠ 0) 
  (h₂ : ∀ (x : ℝ),  x = a → pa^2 = p * x^2) 
  (h₃ : ∀ (x : ℝ),  x = 0 → 0 = q * (x - a)^2 + pa^2) : 
  p + q = 0 := 
sorry

end parabola_vertex_in_other_l808_808268


namespace math_problem_l808_808792

theorem math_problem (x y z w : ℝ) :
  (log (log 10) = 0 ∧ log (log (exp 1)) = 0) ∧
  (10 = log x → x ≠ 10) ∧
  (exp 1 = log y → y ≠ exp 2) :=
by
  sorry

end math_problem_l808_808792


namespace arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l808_808135

-- Question 1
theorem arithmetic_sequence_n (a1 a4 a10 : ℤ) (d : ℤ) (n : ℤ) (Sn : ℤ) 
  (h1 : a1 + 3 * d = a4) 
  (h2 : a1 + 9 * d = a10)
  (h3 : Sn = n * (2 * a1 + (n - 1) * d) / 2)
  (h4 : a4 = 10)
  (h5 : a10 = -2)
  (h6 : Sn = 60)
  : n = 5 ∨ n = 6 := 
sorry

-- Question 2
theorem sum_arithmetic_sequence_S17 (a1 : ℤ) (d : ℤ) (a_n1 : ℤ → ℤ) (S17 : ℤ)
  (h1 : a1 = -7)
  (h2 : ∀ n, a_n1 (n + 1) = a_n1 n + d)
  (h3 : S17 = 17 * (2 * a1 + 16 * d) / 2)
  : S17 = 153 := 
sorry

-- Question 3
theorem arithmetic_sequence_S13 (a_2 a_7 a_12 : ℤ) (S13 : ℤ)
  (h1 : a_2 + a_7 + a_12 = 24)
  (h2 : S13 = a_7 * 13)
  : S13 = 104 := 
sorry

end arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l808_808135


namespace some_number_value_l808_808039

theorem some_number_value : 
  let x := 6.5.floor * (2 / 3).floor + 2.floor * 7.2 + 8.4.floor - 6.6 
  in x = 15.8 := 
by
  sorry

end some_number_value_l808_808039


namespace sphere_shot_radius_l808_808524

theorem sphere_shot_radius (R : ℝ) (N : ℕ) (π : ℝ) (r : ℝ) 
  (h₀ : R = 4) (h₁ : N = 64) 
  (h₂ : (4 / 3) * π * (R ^ 3) / ((4 / 3) * π * (r ^ 3)) = N) : 
  r = 1 := 
by
  sorry

end sphere_shot_radius_l808_808524


namespace inverse_of_fraction_l808_808347

theorem inverse_of_fraction :
  (- (3 / 2))⁻¹ = - (2 / 3) :=
sorry

end inverse_of_fraction_l808_808347


namespace original_sets_exist_l808_808907

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808907


namespace true_propositions_l808_808086

noncomputable def z : ℂ := (2 : ℂ) / (-1 + complex.I)
def p1 := complex.abs z = 2
def p2 := z ^ 2 = 2 * complex.I
def p3 := complex.conj z = 1 + complex.I
def p4 := z.im = -1

theorem true_propositions :
  p2 ∧ p4 ∧ ¬p1 ∧ ¬p3 := by
  sorry

end true_propositions_l808_808086


namespace store_total_income_l808_808551

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l808_808551


namespace Nina_total_problems_l808_808180

def Ruby_math_problems := 12
def Ruby_reading_problems := 4
def Ruby_science_problems := 5

def Nina_math_problems := 5 * Ruby_math_problems
def Nina_reading_problems := 9 * Ruby_reading_problems
def Nina_science_problems := 3 * Ruby_science_problems

def total_problems := Nina_math_problems + Nina_reading_problems + Nina_science_problems

theorem Nina_total_problems : total_problems = 111 :=
by
  sorry

end Nina_total_problems_l808_808180


namespace sin_2x_eq_4_l808_808972

open Real

theorem sin_2x_eq_4 
  (x : ℝ) 
  (h : sin x + cos x + 2 * tan x + 2 * cot x + sec x + csc x = 9) : 
  sin (2 * x) = 4 :=
sorry

end sin_2x_eq_4_l808_808972


namespace necessary_condition_l808_808115

variable (A B C : Prop)

theorem necessary_condition (h₁ : ¬A ↔ ¬B) (h₂ : ¬B → ¬C) : C → A :=
by
  intro hC
  by_contradiction hA
  have hB := (h₁.mpr hA)
  exact h₂ hB hC

end necessary_condition_l808_808115


namespace units_digit_7_pow_6_pow_5_l808_808422

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808422


namespace find_p_l808_808074

open Real

variable (A : ℝ × ℝ)
variable (p : ℝ) (hp : p > 0)

-- Conditions
def on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop := A.snd^2 = 2 * p * A.fst
def dist_focus (A : ℝ × ℝ) (p : ℝ) : Prop := sqrt ((A.fst - p / 2)^2 + A.snd^2) = 12
def dist_y_axis (A : ℝ × ℝ) : Prop := abs (A.fst) = 9

-- Theorem to prove
theorem find_p (h1 : on_parabola A p) (h2 : dist_focus A p) (h3 : dist_y_axis A) : p = 6 :=
sorry

end find_p_l808_808074


namespace G_five_times_of_2_l808_808208

def G (x : ℝ) : ℝ := (x - 2) ^ 2 - 1

theorem G_five_times_of_2 : G (G (G (G (G 2)))) = 1179395 := 
by 
  rw [G, G, G, G, G]; 
  sorry

end G_five_times_of_2_l808_808208


namespace units_digit_of_7_pow_6_pow_5_l808_808433

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808433


namespace units_digit_7_power_l808_808393

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808393


namespace minimum_value_expr_l808_808166

theorem minimum_value_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ∃ a' b' : ℝ, 
  a' ≠ 0 ∧ b' ≠ 0 ∧ 
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a^2 + b^2 + 1/a^2 + b/a + b^2/a^2 ≥ 3 / (Real.cbrt 2)) :=
sorry

end minimum_value_expr_l808_808166


namespace units_digit_7_pow_6_pow_5_l808_808408

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808408


namespace stationery_store_sales_l808_808548

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l808_808548


namespace true_proposition_l808_808055

-- Define p: ∀ x ∈ ℝ, x^2 - x + 1 > 0
def p := ∀ x : ℝ, x^2 - x + 1 > 0

-- Define q: ∃ x ∈ (0, +∞), sin x > 1
def q := ∃ x : ℝ, (0 < x) ∧ (sin x > 1)

-- Define the proposition to prove
theorem true_proposition : p ∨ ¬ q :=
by
  sorry

end true_proposition_l808_808055


namespace ratio_areas_l808_808227

variables (s : ℝ) (A_small A_large : ℝ)
def area_equilateral_triangle (a : ℝ) : ℝ := (a^2 * real.sqrt 3) / 4

-- Assumptions based on conditions
axiom three_small_triangles : A_small = 3 * area_equilateral_triangle s
axiom one_large_triangle   : A_large = area_equilateral_triangle (3 * s)

-- Theorem stating the ratio of areas
theorem ratio_areas (h₁ : A_small = 3 * area_equilateral_triangle s)
                    (h₂ : A_large = area_equilateral_triangle (3 * s)) :
  A_small / A_large = 1 / 3 :=
by
  have A_small_def : A_small = 3 * area_equilateral_triangle s := h₁
  have A_large_def : A_large = area_equilateral_triangle (3 * s) := h₂
  -- calculation of the ratio
  rw [A_small_def, A_large_def]
  sorry

end ratio_areas_l808_808227


namespace farmer_total_acres_l808_808753

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l808_808753


namespace non_overlapping_original_sets_l808_808887

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808887


namespace john_quiz_goal_l808_808343

theorem john_quiz_goal
  (total_quizzes : ℕ)
  (goal_percentage : ℕ)
  (quizzes_completed : ℕ)
  (quizzes_remaining : ℕ)
  (quizzes_with_A_completed : ℕ)
  (total_quizzes_with_A_needed : ℕ)
  (additional_A_needed : ℕ)
  (quizzes_below_A_allowed : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 75)
  (h3 : quizzes_completed = 40)
  (h4 : quizzes_remaining = total_quizzes - quizzes_completed)
  (h5 : quizzes_with_A_completed = 27)
  (h6 : total_quizzes_with_A_needed = total_quizzes * goal_percentage / 100)
  (h7 : additional_A_needed = total_quizzes_with_A_needed - quizzes_with_A_completed)
  (h8 : quizzes_below_A_allowed = quizzes_remaining - additional_A_needed)
  (h_goal : quizzes_below_A_allowed ≤ 2) : quizzes_below_A_allowed = 2 :=
by
  sorry

end john_quiz_goal_l808_808343


namespace candy_total_l808_808003

theorem candy_total (chocolate_boxes caramel_boxes mint_boxes berry_boxes : ℕ)
  (chocolate_pieces caramel_pieces mint_pieces berry_pieces : ℕ)
  (h_chocolate : chocolate_boxes = 7)
  (h_caramel : caramel_boxes = 3)
  (h_mint : mint_boxes = 5)
  (h_berry : berry_boxes = 4)
  (p_chocolate : chocolate_pieces = 8)
  (p_caramel : caramel_pieces = 8)
  (p_mint : mint_pieces = 10)
  (p_berry : berry_pieces = 12) :
  (chocolate_boxes * chocolate_pieces + caramel_boxes * caramel_pieces + mint_boxes * mint_pieces + berry_boxes * berry_pieces) = 178 := by
  sorry

end candy_total_l808_808003


namespace hyperbola_eccentricity_proof_l808_808079

noncomputable theory

open Real

def hyperbola_eccentricity (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) 
(distance_condition : dist (4, 0) ((λ x, (b/a) * x), (λ x, -(b/a) * x)) = sqrt 2) : ℝ :=
  let c := sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_proof (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
  (distance_condition : dist (4, 0) ((λ x, (b/a) * x), (λ x, -(b/a) * x)) = sqrt 2) :
  hyperbola_eccentricity a b h₁ h₂ distance_condition = (2 * sqrt 14) / 7 :=
sorry

end hyperbola_eccentricity_proof_l808_808079


namespace max_area_of_right_triangle_l808_808989

noncomputable def max_area_right_triangle (a b : ℝ) (h1 : a + b + Real.sqrt (a^2 + b^2) = 2) : ℝ :=
  let area := (1 / 2) * a * b in
  area

theorem max_area_of_right_triangle (a b : ℝ) (h1 : a + b + Real.sqrt (a^2 + b^2) = 2) :
  max_area_right_triangle a b h1 = 3 - 2 * Real.sqrt 2 := by sorry

end max_area_of_right_triangle_l808_808989


namespace picking_time_l808_808720

theorem picking_time (x : ℝ) 
  (h_wang : x * 8 - 0.25 = x * 7) : 
  x = 0.25 := 
by
  sorry

end picking_time_l808_808720


namespace total_words_in_poem_l808_808627

theorem total_words_in_poem (stanzas lines words : ℕ) 
  (h1 : stanzas = 20) 
  (h2 : lines = 10) 
  (h3 : words = 8) :
  stanzas * lines * words = 1600 :=
by
  rw [h1, h2, h3]
  norm_num

end total_words_in_poem_l808_808627


namespace range_of_m_l808_808052

noncomputable def p (x : ℝ) : Prop := |x - 3| ≤ 2
noncomputable def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m {m : ℝ} (H : ∀ (x : ℝ), ¬p x → ¬q x m) :
  2 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l808_808052


namespace largest_root_of_polynomial_l808_808230

theorem largest_root_of_polynomial :
  ∃ x ∈ ({4, 2, -1} : set ℝ), is_root (λ x, x^3 - 5*x^2 + 2*x + 8) x ∧ 
  (∀ y ∈ ({4, 2, -1} : set ℝ), y ≤ x) :=
sorry

end largest_root_of_polynomial_l808_808230


namespace max_adj_pairs_diff_color_l808_808056

-- Definition of the problem setup
def grid (m n : ℕ) := fin m × fin n

-- Definition of the row and column constraints
def col_balanced (g : grid 100 100 → bool) := ∀ j : fin 100, ∑ i, if g (i, j) then 1 else (0 : ℝ) = 50
def row_unique (g : grid 100 100 → bool) := ∀ i1 i2 : fin 100, i1 ≠ i2 → 
  (∑ j, if g (i1, j) then 1 else (0 : ℝ)) ≠ (∑ j, if g (i2, j) then 1 else (0 : ℝ))

-- Definition of adjacent cell pairs
def adj_pairs_diff_color (g: grid 100 100 → bool) : ℕ :=
  ∑ i j, if g (i, j) ≠ g (i + 1, j) then 1 else 0 + 
  ∑ i j, if g (i, j) ≠ g (i, j + 1) then 1 else 0

-- Statement to prove the maximum number of pairs of adjacent cells with different colors
theorem max_adj_pairs_diff_color : 
  ∃ g : grid 100 100 → bool, col_balanced g ∧ row_unique g ∧ adj_pairs_diff_color g = 14751 :=
by
  sorry

end max_adj_pairs_diff_color_l808_808056


namespace initial_gum_count_l808_808248

variables (t_given_gum : ℕ) (t_final_gum : ℕ)

theorem initial_gum_count (h_given : t_given_gum = 16) (h_final : t_final_gum = 54) : 
  t_final_gum - t_given_gum = 38 :=
by
  rw [h_given, h_final]
  norm_num

end initial_gum_count_l808_808248


namespace units_digit_of_power_l808_808399

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808399


namespace pq_sum_correct_l808_808768

noncomputable def find_p_q_sum (p q : ℝ) (h_fold_1: (0, 4) = (5, 0))
    (h_fold_2: ∃ m: ℝ, (y = m * x + b) is the perpendicular bisector of the segment connecting (0,4) and (5,0))
    (h_points: (p, q) is matched with (9, 6)): ℝ :=
  p + q

theorem pq_sum_correct
  (p q: ℝ)
  (h_fold_1: (0, 4) = (5, 0))
  (h_fold_2: ∃ m: ℝ, (y = m * x + b) is the perpendicular bisector of the segment connecting (0,4) and (5,0))
  (h_points: (p, q) is matched with (9, 6)) :
  find_p_q_sum p q h_fold_1 h_fold_2 h_points ∈ {12.0, 12.5, 13.0, 13.5} :=
  sorry

end pq_sum_correct_l808_808768


namespace bisector_of_C_is_bisector_of_MCN_l808_808145

noncomputable def Triangle (A B C : Type) [EuclideanGeometry A B C] : Prop :=
  ∃ M N : Point, 
    isMidpoint M (A, C) ∧
    isBisectorOfAngle B M ∧ 
    isMidpoint N (BisectorOf B) ∧
    isBisectorOfAngle C (Angle M C N)

theorem bisector_of_C_is_bisector_of_MCN (A B C M N : Point) 
  [Triangle A B C] :
  isBisectorOfAngle C (Angle M C N) :=
sorry

end bisector_of_C_is_bisector_of_MCN_l808_808145


namespace part_I_part_II_l808_808096

-- Define the inequality and its conditions
def inequality_condition (x a : ℝ) : Prop :=
  (x - 1)^2 ≤ a^2 ∧ a > 0

-- Define the definition domain of the function
def function_domain (x : ℝ) : Prop :=
  ((x < -2) ∨ (x > 2))

-- Define the function itself
def f (x : ℝ) : ℝ := Real.log ((x - 2) / (x + 2))

-- Define the intersection condition
def intersection_condition (a : ℝ) : Prop :=
  ∀ x, inequality_condition x a → ¬ function_domain x

-- Statement for part (I)
theorem part_I (a : ℝ) : 
  (∃ a, 0 < a ∧ a ≤ 1) ↔ intersection_condition a :=
sorry

-- Statement for part (II)
theorem part_II : 
  ∀ x, ((x < -2) ∨ (x > 2)) → f(-x) = - f(x) :=
sorry

end part_I_part_II_l808_808096


namespace watermelon_melon_weight_l808_808596

variables {W M : ℝ}

theorem watermelon_melon_weight :
  (2 * W > 3 * M ∨ 3 * W > 4 * M) ∧ ¬ (2 * W > 3 * M ∧ 3 * W > 4 * M) → 12 * W ≤ 18 * M :=
by
  sorry

end watermelon_melon_weight_l808_808596


namespace non_overlapping_original_sets_exists_l808_808873

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808873


namespace units_digit_7_pow_6_pow_5_l808_808409

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808409


namespace smallest_perimeter_l808_808506

theorem smallest_perimeter (m n : ℕ) 
  (h1 : (m - 4) * (n - 4) = 8) 
  (h2 : ∀ k l : ℕ, (k - 4) * (l - 4) = 8 → 2 * k + 2 * l ≥ 2 * m + 2 * n) : 
  (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
sorry

end smallest_perimeter_l808_808506


namespace choose_starting_lineup_l808_808733

/-- 
There are 12 ways to choose the point guard.
There are 330 ways to choose the remaining 4 players from 11 team members. The total number of ways to choose the starting lineup is therefore 12 * 330. 
-/
theorem choose_starting_lineup (total_members : ℕ) (lineup_size : ℕ) (point_guard : ℕ) (remaining_members : ℕ) 
  (comb : ℕ → ℕ → ℕ) (fact : ℕ → ℕ) [fact 11 = 39916800] [fact 4 = 24] [fact 7 = 5040] [comb 11 4 = ((fact 11) / ((fact 4) * (fact 7)))] :
  total_members = 12 ∧ lineup_size = 5 ∧ point_guard = 1 ∧ remaining_members = 4 ∧ (total_members * comb 11 4) = 3960 := 
sorry

end choose_starting_lineup_l808_808733


namespace seeds_in_first_plot_l808_808843

theorem seeds_in_first_plot (x : ℕ) (h1 : 200 = 200) (h2 : 0.25 * x = 0.25 * x) (h3 : 0.4 * 200 = 80) (h4 : 31 = 31) :
  (∃ (x : ℕ), (25 * x + 8000 = 31 * (x + 200)) ∧ x = 300) :=
by
  sorry

end seeds_in_first_plot_l808_808843


namespace net_gain_loss_l808_808601

-- Definitions of the initial conditions
structure InitialState :=
  (cash_x : ℕ) (painting_value : ℕ) (cash_y : ℕ)

-- Definitions of transactions
structure Transaction :=
  (sell_price : ℕ) (commission_rate : ℕ)

def apply_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash + (tr.sell_price - (tr.sell_price * tr.commission_rate / 100))

def revert_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash - tr.sell_price + (tr.sell_price * tr.commission_rate / 100)

def compute_final_cash (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction) : ℕ :=
  let cash_x_after_first := apply_transaction initial_states.cash_x trans1
  let cash_y_after_first := initial_states.cash_y - trans1.sell_price
  let cash_x_after_second := revert_transaction cash_x_after_first trans2
  let cash_y_after_second := cash_y_after_first + (trans2.sell_price - (trans2.sell_price * trans2.commission_rate / 100))
  cash_x_after_second - initial_states.cash_x + (cash_y_after_second - initial_states.cash_y)

-- Statement of the theorem
theorem net_gain_loss (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction)
  (h1 : initial_states.cash_x = 15000)
  (h2 : initial_states.painting_value = 15000)
  (h3 : initial_states.cash_y = 18000)
  (h4 : trans1.sell_price = 20000)
  (h5 : trans1.commission_rate = 5)
  (h6 : trans2.sell_price = 14000)
  (h7 : trans2.commission_rate = 5) : 
  compute_final_cash initial_states trans1 trans2 = 5000 - 6700 :=
sorry

end net_gain_loss_l808_808601


namespace parallel_if_and_only_if_parallel_to_planes_l808_808451

-- Definitions for planes and lines
variables {Plane Line : Type*}

-- Conditions on the planes \alpha and \beta, and lines m and n
variables (α β : Plane) (m n : Line)

-- Intersection of planes α and β is line m
axiom planes_intersect (hαβ : α ≠ β) : α ∩ β = m

-- Line n is not contained in plane α nor in plane β
axiom line_not_in_plane (hnα : ¬ n ⊆ α) (hnβ : ¬ n ⊆ β)

-- Axioms for parallelism of lines and planes
axiom parallel_lines : Prop -- placeholder for the parallelism of lines
axiom parallel_plane_line : Plane → Line → Prop -- placeholder for a line being parallel to a plane

-- The equivalent proof problem
theorem parallel_if_and_only_if_parallel_to_planes
  (hαβm : α ∩ β = m)
  (hn_not_subset_α : ¬ n ⊆ α)
  (hn_not_subset_β : ¬ n ⊆ β) :
  (parallel_lines n m) ↔ (parallel_plane_line α n ∧ parallel_plane_line β n) :=
sorry

end parallel_if_and_only_if_parallel_to_planes_l808_808451


namespace dartboard_distribution_l808_808790

theorem dartboard_distribution :
  ∃ lists : Finset (Multiset ℕ), 
    (∀ l ∈ lists, l.card = 5 ∧ l.sum = 6 ∧ l.order_of_bounds) ∧ 
    lists.card = 11 :=
begin 
  sorry 
end

end dartboard_distribution_l808_808790


namespace number_of_children_l808_808213

theorem number_of_children (C A : ℕ) (h1 : C = 2 * A) (h2 : C + A = 120) : C = 80 :=
by
  sorry

end number_of_children_l808_808213


namespace ratio_of_a5_to_a4_l808_808163

variable (a₁ d : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_first_n_terms (n : ℕ) : ℝ := (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d)

-- Condition S7 = 3 (a₁ + a₉)
axiom S7_eq : S 7 = 3 * (a₁ + nth_term 9)

-- Define a₄ and a₅ using the nth_term function
def a₄ := nth_term 4
def a₅ := nth_term 5

-- The theorem to prove
theorem ratio_of_a5_to_a4 : 
  (S = sum_of_first_n_terms) → 
  (a = nth_term) → 
  S 7 = 3 * (a₁ + a 9) →
  a₁ = 3 * d → 
  a 5 / a 4 = (7 : ℝ) / 6 :=
by sorry

end ratio_of_a5_to_a4_l808_808163


namespace arithmetic_sequence_problem_l808_808140

-- Define the arithmetic sequence
def isArithmeticSequence (a : ℕ → ℤ) : Prop := 
  ∀ (n : ℕ), a(n + 1) - a(n) = a(1) - a(0)

-- The specific condition of the problem
def condition (a : ℕ → ℤ) [isArithmeticSequence a] : Prop := 
  a(2) + a(8) = 180

-- The target result to be proved
theorem arithmetic_sequence_problem (a : ℕ → ℤ) [isArithmeticSequence a] (h : condition a) :
  a(3) + a(4) + a(5) + a(6) + a(7) = 450 := 
by
  sorry

end arithmetic_sequence_problem_l808_808140


namespace greatest_k_value_l808_808664

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b^2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

theorem greatest_k_value (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 17 = 0) →
  (let roots := quadratic_formula 1 k 17 in (roots.fst - roots.snd) = Real.sqrt 85) →
  k = Real.sqrt 153 :=
by
  sorry

end greatest_k_value_l808_808664


namespace solve_for_f_l808_808853

noncomputable def f (x : ℝ) : ℝ := 2 * x - x^2

theorem solve_for_f (x : ℝ) : f (1 - cos x) = sin x ^ 2 :=
by
  -- proof steps go here
  sorry

end solve_for_f_l808_808853


namespace factorize_expression_l808_808377

variable (a b c : ℝ)

theorem factorize_expression : 
  (a - 2 * b) * (a - 2 * b - 4) + 4 - c ^ 2 = ((a - 2 * b) - 2 + c) * ((a - 2 * b) - 2 - c) := 
by
  sorry

end factorize_expression_l808_808377


namespace range_of_m_l808_808058

noncomputable def locally_odd (f : ℝ → ℝ) (domain : set ℝ) : Prop :=
∃ x ∈ domain, f (-x) = -f x

theorem range_of_m (m : ℝ) :
  (locally_odd (λ x, m + 2^x) (set.Icc (-1:ℝ) 1)) ∧
  (let disc := (5 * m + 1)^2 - 4 in disc > 0) ∧
  (¬ ((locally_odd (λ x, m + 2^x) (set.Icc (-1:ℝ) 1)) ∧
     (let disc := (5 * m + 1)^2 - 4 in disc > 0))) ∧
  ((locally_odd (λ x, m + 2^x) (set.Icc (-1:ℝ) 1)) ∨
   (let disc := (5 * m + 1)^2 - 4 in disc > 0)) →
  (m < -5 / 4) ∨
  (-1 < m ∧ m < -3 / 5) ∨
  (m > 1 / 5) :=
by sorry

end range_of_m_l808_808058


namespace quadratic_function_value_l808_808760

theorem quadratic_function_value
  (p q r : ℝ)
  (h1 : p + q + r = 3)
  (h2 : 4 * p + 2 * q + r = 12) :
  p + q + 3 * r = -5 :=
by
  sorry

end quadratic_function_value_l808_808760


namespace scientific_notation_of_220_billion_l808_808639

theorem scientific_notation_of_220_billion :
  220000000000 = 2.2 * 10^11 :=
by
  sorry

end scientific_notation_of_220_billion_l808_808639


namespace measure_obtuse_angle_ADB_l808_808557

-- Define points and angles
variables {A B C D : Type} [triangle ABC]
variables (a b : Real) (angle_A angle_C angle_ADB : Real)

-- Given conditions of the isosceles right triangle ABC
def triangle_ABC_is_isosceles_right : Prop :=
  is_isosceles_right_triangle A B C ∧
  (angle_A = 45) ∧ (angle_C = 45)

-- Define the angle bisectors and the point of intersection D
def intersect_bisectors_at_D : Prop :=
  bisects A AD (angle_A / 2) ∧
  bisects C DC (angle_C / 2) ∧
  intersects AD DC D

-- The statement to be proved
theorem measure_obtuse_angle_ADB :
  triangle_ABC_is_isosceles_right →
  intersect_bisectors_at_D →
  angle_ADB = 135 :=
by
  intros h1 h2,
  sorry

end measure_obtuse_angle_ADB_l808_808557


namespace infinite_solutions_l808_808031

theorem infinite_solutions :
  ∃ (f : ℝ → (ℝ × ℝ)), function.injective f ∧ 
  ∀ x y : ℝ, (x, y) ∈ set.range f → 9^(x^2 - y) + 9^(x - y^2) = 1 :=
sorry

end infinite_solutions_l808_808031


namespace range_of_a2_l808_808467

-- Arithmetic sequence definitions and properties
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def third_term (a : ℕ → ℝ) (d : ℝ) :=
a 2 + d

def sixth_term (a : ℕ → ℝ) (d : ℝ) :=
a 2 + 4 * d

def sum_of_first_n_terms (a : ℕ → ℝ) (d : ℝ) (n : ℕ) :=
(n / 2) * (2 * a 1 + (n - 1) * d)

-- Given conditions
theorem range_of_a2 (a : ℕ → ℝ) (d : ℝ) (h_seq : arithmetic_sequence a d) (h_cond1 : 3 * third_term a d = sixth_term a d + 4) (h_sum_cond : sum_of_first_n_terms a d 5 < 10) :
∀ a2 : ℝ, a2 < 2 :=
by {
    intro a2,
    -- Proof should go here
    sorry
}

end range_of_a2_l808_808467


namespace pairing_product_is_square_l808_808381

theorem pairing_product_is_square (n : ℕ) (h : n > 1) :
  ∃ (pairs : list (ℕ × ℕ)), (∀ (p ∈ pairs), p.1 + p.2 ∈ list.range (2 * n + 1) ∧
  (list.foldr (*) 1 (list.map (λ q : ℕ × ℕ, q.1 + q.2) pairs) ∈ {k | ∃ m : ℕ, k = m * m})) := sorry

end pairing_product_is_square_l808_808381


namespace problem_statement_l808_808302

noncomputable def product_le_bound (k : ℕ) (n : ℕ → ℕ) (a b : ℕ) : Prop :=
  k ≥ 2 ∧
  (∀ i, 1 ≤ i → i ≤ k → 1 < n i) ∧
  (∀ i j, 1 ≤ i → i < j → j ≤ k → n i < n j) ∧
  ((1 - (1 / (n 1 : ℝ))) * (1 - (1 / (n 2 : ℝ))) * ... * (1 - (1 / (n k : ℝ)))) ≤ (a / b : ℝ) ∧
  (a / b : ℝ) < (1 - (1 / (n 1 : ℝ))) * (1 - (1 / (n 2 : ℝ))) * ... * (1 - (1 / (n (k-1) : ℝ)))

theorem problem_statement (k : ℕ) (n : ℕ → ℕ) (a b : ℕ) (h : product_le_bound k n a b) : 
  n 1 * n 2 * ... * n k ≤ (4 * a)^2^(k-1) :=
sorry

end problem_statement_l808_808302


namespace smallest_sum_XYZb_l808_808114

noncomputable def smallest_sum (X Y Z b : ℕ) : ℕ :=
  if X < 6 ∧ Y < 6 ∧ Z < 6 ∧ b > 6 ∧ 36 * X + 6 * Y + Z = 3 * b + 3 then
    X + Y + Z + b
  else
    0

theorem smallest_sum_XYZb : ∃ (X Y Z b : ℕ), 
  X < 6 ∧ Y < 6 ∧ Z < 6 ∧ b > 6 ∧ 36 * X + 6 * Y + Z = 3 * b + 3 ∧ smallest_sum X Y Z b = 28 :=
begin
  use [2, 1, 0, 25],
  split,
  {exact nat.lt_succ_self 5}, -- X < 6
  split,
  {exact nat.lt_succ_self 5}, -- Y < 6
  split,
  {exact nat.lt_succ_self 5}, -- Z < 6
  split,
  {exact nat.succ_lt_succ_iff.mpr (nat.le_of_lt (nat.lt_trans nat.zero_lt_six (nat.succ_lt_succ zero_lt_five)))}, -- b > 6
  split,
  {exact calc 36 * 2 + 6 * 1 + 0 = 72 + 6 + 0 : by ring
                           ... = 78 : by ring
                           ... = 3 * 25 + 3 : by ring},  -- 36X + 6Y + Z = 3b + 3
  {exact if_pos (and.intro (and.intro (nat.lt_succ_self 5) (and.intro (nat.lt_succ_self 5) (and.intro (nat.lt_succ_self 5) (nat.succ_lt_succ_iff.mpr (nat.le_of_lt (nat.lt_trans nat.zero_lt_six (nat.succ_lt_succ zero_lt_five)))))))
                         (eq.refl (36 * 2 + 6 * 1 + 0)))
  } -- smallest_sum X Y Z b = 2 + 1 + 0 + 25
end

end smallest_sum_XYZb_l808_808114


namespace distance_between_cross_sections_same_side_distance_between_cross_sections_opposite_side_l808_808063

-- Define the radius of the sphere and the circumferences.
def radius : ℝ := 5
def circumference1 : ℝ := 6 * Real.pi
def circumference2 : ℝ := 8 * Real.pi

-- Calculate the radii of the cross-sections
def radius1 : ℝ := circumference1 / (2 * Real.pi)
def radius2 : ℝ := circumference2 / (2 * Real.pi)

-- Calculate distances from the center of the sphere to the cross-sections using Pythagorean theorem
def distanceFromCenter1 : ℝ := Real.sqrt (radius ^ 2 - radius1 ^ 2)
def distanceFromCenter2 : ℝ := Real.sqrt (radius ^ 2 - radius2 ^ 2)

-- Prove the distances: same side or opposite sides
theorem distance_between_cross_sections_same_side :
  abs (distanceFromCenter1 - distanceFromCenter2) = 1 :=
sorry

theorem distance_between_cross_sections_opposite_side :
  distanceFromCenter1 + distanceFromCenter2 = 7 :=
sorry

end distance_between_cross_sections_same_side_distance_between_cross_sections_opposite_side_l808_808063


namespace salary_restoration_l808_808333

theorem salary_restoration (S : ℝ) : 
  let reduced_salary := 0.7 * S
  let restore_factor := 1 / 0.7
  let percentage_increase := restore_factor - 1
  percentage_increase * 100 = 42.857 :=
by
  sorry

end salary_restoration_l808_808333


namespace rhombus_area_8_cm2_l808_808133

open Real

noncomputable def rhombus_area (side : ℝ) (angle : ℝ) : ℝ :=
  (side * side * sin angle) / 2 * 2

theorem rhombus_area_8_cm2 (side : ℝ) (angle : ℝ) (h1 : side = 4) (h2 : angle = π / 4) : rhombus_area side angle = 8 :=
by
  -- Definitions and calculations are omitted and replaced with 'sorry'
  sorry

end rhombus_area_8_cm2_l808_808133


namespace initial_concentration_of_hydrochloric_acid_l808_808743

theorem initial_concentration_of_hydrochloric_acid
  (initial_mass : ℕ)
  (drained_mass : ℕ)
  (added_concentration : ℕ)
  (final_concentration : ℕ)
  (total_mass : ℕ)
  (initial_concentration : ℕ) :
  initial_mass = 300 ∧ drained_mass = 25 ∧ added_concentration = 80 ∧ final_concentration = 25 ∧ total_mass = 300 →
  (275 * initial_concentration / 100 + 20 = 75) →
  initial_concentration = 20 :=
by
  intros h_eq h_new_solution
  -- Rewriting the data given in h_eq and solving h_new_solution
  rcases h_eq with ⟨h_initial_mass, h_drained_mass, h_added_concentration, h_final_concentration, h_total_mass⟩
  sorry

end initial_concentration_of_hydrochloric_acid_l808_808743


namespace non_overlapping_original_sets_l808_808886

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808886


namespace cos_A_value_max_length_AD_l808_808103

variables {A B C : ℝ} {a b c : ℝ}
variable (triangle : PropType → Prop)
variable (angle_bisector : PropType → Prop)
variable (area : ℝ → ℝ → ℝ → ℝ)

-- Conditions
axiom condition1 : triangle ∆ ABC
axiom condition2 : (3 * (a - b) / c) = (3 * sin C - 2 * sin B) / (sin A + sin B)
axiom condition3 : area a b c = 2 * sqrt 2
axiom condition4 : angle_bisector AD A

-- The two Parts
theorem cos_A_value : cos A = 1 / 3 := sorry

theorem max_length_AD : max_length AD = 2 := sorry

end cos_A_value_max_length_AD_l808_808103


namespace prob_B_takes_second_shot_prob_A_takes_i_shot_correct_expected_shots_A_l808_808615

-- Definitions based on the conditions
def shooting_percentage_A : ℝ := 0.6
def shooting_percentage_B : ℝ := 0.8
def initial_prob : ℝ := 0.5

-- Proof problem for Part 1
theorem prob_B_takes_second_shot :
  (initial_prob * (1 - shooting_percentage_A)) + 
  (initial_prob * shooting_percentage_B) = 0.6 :=
sorry

-- Definition of the probability player A takes the ith shot
def prob_A_takes_i_shot (i : ℕ) : ℝ :=
  1 / 3 + (1 / 6) * (2 / 5)^(i - 1)

-- Proof problem for Part 2
theorem prob_A_takes_i_shot_correct (i : ℕ) :
  prob_A_takes_i_shot i = 
  1 / 3 + (1 / 6) * (2 / 5)^(i - 1) :=
sorry

-- Proof problem for Part 3
theorem expected_shots_A (n : ℕ) :
  (∑ i in finset.range n, prob_A_takes_i_shot (i + 1)) = 
  (5 / 18) * (1 - (2 / 5)^n) + (n / 3) :=
sorry

end prob_B_takes_second_shot_prob_A_takes_i_shot_correct_expected_shots_A_l808_808615


namespace find_derivative_l808_808080

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * x * f' 2

theorem find_derivative (f' : ℝ → ℝ) (h : ∀ x, f' x = 6 * x + 2 * f' 2) : f' 5 = 36 := by
  have h2 : f' 2 = 3 := by
    rw [h 2]
    linarith

  rw [h 5]
  rw [h2]
  linarith

end find_derivative_l808_808080


namespace full_day_students_l808_808342

theorem full_day_students (total_students : ℕ) (half_day_percentage : ℕ) (half_day_fraction : total_students * half_day_percentage / 100) :
  total_students = 165 ∧ half_day_percentage = 36 → 
  (total_students - half_day_fraction.floor) = 106 :=
begin
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  -- Rest does the proof
  sorry
end

end full_day_students_l808_808342


namespace factorize_expression_l808_808834

-- Define that a and b are arbitrary real numbers
variables (a b : ℝ)

-- The theorem statement claiming that 3a²b - 12b equals the factored form 3b(a + 2)(a - 2)
theorem factorize_expression : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) :=
by
  sorry  -- proof omitted

end factorize_expression_l808_808834


namespace pentagon_largest_angle_l808_808745

theorem pentagon_largest_angle :
  ∃ x : ℝ, 
    (2 * x - 8 + 3 * x + 12 + 4 * x + 8 + 5 * x - 18 + x + 6 = 540) ∧
    (∀ (y : ℝ), y ∈ {2 * x - 8, 3 * x + 12, 4 * x + 8, 5 * x - 18, x + 6} → y ≤ 162) ∧
    5 * x - 18 = 162 :=
by
  sorry

end pentagon_largest_angle_l808_808745


namespace common_tangent_lines_l808_808293

theorem common_tangent_lines :
  let L1 : ℝ → ℝ → Prop := fun x y => √3 * x - y - 2 = 0
  let L2 : ℝ → ℝ → Prop := fun x y => √3 * x + y + 2 = 0
  let circle : ℝ → ℝ → Prop := fun x y => x^2 + y^2 = 1
  let parabola : ℝ → ℝ → Prop := fun x y => x^2 = (8/3) * y
in 
(tangent_at L1 circle) ∧ (tangent_at L1 parabola) ∧ (tangent_at L2 circle) ∧ (tangent_at L2 parabola)
sorry

end common_tangent_lines_l808_808293


namespace good_word_count_l808_808821

def is_good_word (w : List Char) : Prop :=
  w.length = 10 ∧
  (∀ i, i < 9 → w.get? i = some 'A' → w.get? (i + 1) ≠ some 'B') ∧
  (∀ i, i < 9 → w.get? i = some 'B' → w.get? (i + 1) ≠ some 'C') ∧
  (∀ i, i < 9 → w.get? i = some 'C' →
    (w.get? (i + 1) ≠ some 'A' ∧ w.get? (i + 1) ≠ some 'B'))

def count_good_words : Nat :=
  {w : List Char | is_good_word w}.card

theorem good_word_count : count_good_words = 1536 :=
  sorry

end good_word_count_l808_808821


namespace sum_of_polynomial_roots_l808_808582

theorem sum_of_polynomial_roots:
  ∀ (a b : ℝ),
  (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) →
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b^3 + b * a^3 = 683 := by
  intros a b h
  sorry

end sum_of_polynomial_roots_l808_808582


namespace four_at_three_equals_thirty_l808_808527

def custom_operation (a b : ℕ) : ℕ :=
  3 * a^2 - 2 * b^2

theorem four_at_three_equals_thirty : custom_operation 4 3 = 30 :=
by
  sorry

end four_at_three_equals_thirty_l808_808527


namespace counterexample_to_conjecture_l808_808816

theorem counterexample_to_conjecture (n : ℕ) (h : n > 5) : 
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) ∨
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) :=
sorry

end counterexample_to_conjecture_l808_808816


namespace hexagon_diagonals_intersect_at_M_l808_808999

theorem hexagon_diagonals_intersect_at_M {A B C M A₁ B₁ C₁ : Point} 
    (hABC_acute : is_acute A B C)
    (hM_bisector : perpendicular_bisector_contains A B C M)
    (hA₁_circum : AM ⊂ circumcircle A B C → A₁ ∈ circumcircle A B C)
    (hB₁_circum : BM ⊂ circumcircle A B C → B₁ ∈ circumcircle A B C)
    (hC₁_circum : CM ⊂ circumcircle A B C → C₁ ∈ circumcircle A B C)
    (h_hexagon : hexagon_formed A B C A₁ B₁ C₁)
    (intersect_criteria : ∃ D E F G H I : Point, 
        hexagon_intersections D E F G H I A B C A₁ B₁ C₁) :
  ∃ D E F G H I : Point, diagonal_intersects_at_pt D E F G H I M :=
sorry

end hexagon_diagonals_intersect_at_M_l808_808999


namespace curve_transformation_l808_808785

theorem curve_transformation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →
  (x' = 4 * x) →
  (y' = 2 * y) →
  (x'^2 / 16 + y'^2 / 4 = 1) :=
by
  sorry

end curve_transformation_l808_808785


namespace domain_of_inverse_function_l808_808491

def f (x : ℝ) : ℝ := log x / log 2 + 3

theorem domain_of_inverse_function : set.Ici (3 : ℝ) = set.range (λ y, f y) :=
by
  sorry

end domain_of_inverse_function_l808_808491


namespace magnitude_difference_sqrt10_l808_808503

variables (a b : ℝ^3)

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

theorem magnitude_difference_sqrt10
  (ha : magnitude a = 3)
  (hb : magnitude b = 2)
  (hab : a.dot_product b = 3 / 2) :
  magnitude (a - b) = real.sqrt 10 :=
by
  sorry

end magnitude_difference_sqrt10_l808_808503


namespace total_feed_per_week_l808_808626

-- Define the conditions
def daily_feed_per_pig : ℕ := 10
def number_of_pigs : ℕ := 2
def days_per_week : ℕ := 7

-- Theorem statement
theorem total_feed_per_week : daily_feed_per_pig * number_of_pigs * days_per_week = 140 := 
  sorry

end total_feed_per_week_l808_808626


namespace cone_diameter_is_2_l808_808670

def surface_area_cone (r l : ℝ) : ℝ := π * r * (r + l)

def lateral_surface_area_is_semicircle (r l : ℝ) : Prop :=
  π * l = 2 * π * r

theorem cone_diameter_is_2 
  (r l : ℝ) (h1 : surface_area_cone r l = 3 * π)
  (h2 : lateral_surface_area_is_semicircle r l) : 
  2 * r = 2 :=
by
  sorry

end cone_diameter_is_2_l808_808670


namespace max_independent_set_size_l808_808315

theorem max_independent_set_size (students : Finset ℕ) (sits_together : ℕ → ℕ → Prop)
  (h_students_card : students.card = 26)
  (h_sits_together : ∀ (a b : ℕ), sits_together a b → a ≠ b)
  (h_each_sits : ∀ (a : ℕ), ∃! (b : ℕ), sits_together a b)
  (h_apart_now : ∀ (a b : ℕ), sits_together a b → ¬ sits_together a b)
  : ∃ (S : Finset ℕ), S.card = 13 ∧ ∀ (x y ∈ S), ¬ sits_together x y := 
sorry

end max_independent_set_size_l808_808315


namespace combined_bus_capacity_eq_40_l808_808608

theorem combined_bus_capacity_eq_40 (train_capacity : ℕ) (fraction : ℚ) (num_buses : ℕ) 
  (h_train_capacity : train_capacity = 120)
  (h_fraction : fraction = 1/6)
  (h_num_buses : num_buses = 2) :
  num_buses * (train_capacity * fraction).toNat = 40 := by
  sorry

end combined_bus_capacity_eq_40_l808_808608


namespace gather_herrings_impossible_l808_808314

theorem gather_herrings_impossible :
  ∀ (sectors : Fin 6 → ℕ), 
  (∀ i : Fin 6, sectors i ∈ {0, 1}) →
  (∀ m : ℕ, ∃ i j : Fin 6, i ≠ j ∧ adjacent i j ∧ move m → sectors (move m i) ≠ sectors (move m j)) →
  ¬ ∃ i : Fin 6, ∀ j : Fin 6, sectors j = 0 ∨ sectors j = 1 ->
  sectors j = sectors i := 
sorry

def adjacent (i j : Fin 6) : Prop :=
  (i.val + 1) % 6 = j.val ∨ (j.val + 1) % 6 = i.val

def move (m : ℕ) (i : Fin 6) : Fin 6 :=
  ⟨(i.val + m) % 6, by apply Nat.mod_lt; exact Nat.succ_pos' 6⟩

end gather_herrings_impossible_l808_808314


namespace AB_days_calc_l808_808738

noncomputable def A_work := B_work + C_work
noncomputable def B_work := 1 / 49
noncomputable def C_work := 1 / 35

theorem AB_days_calc : (A_work + B_work) = 1 / (14:ℝ) :=
begin
  sorry
end

end AB_days_calc_l808_808738


namespace probability_product_odd_l808_808348

theorem probability_product_odd :
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let totalWays := Nat.choose 9 2
  let oddNumbers := {1, 3, 5, 7, 9}
  let oddWays := Nat.choose 5 2
  totalWays ≠ 0 →
  (oddWays / totalWays : ℚ) = 5 / 18 := by
sorry

end probability_product_odd_l808_808348


namespace projection_matrix_inverse_is_zero_matrix_l808_808580

theorem projection_matrix_inverse_is_zero_matrix :
  let v : Vector ℝ 2 := ![4, -5]
  let Q : Matrix (Fin 2) (Fin 2) ℝ := (1 / (v ⬝ᵥ v)) • (Matrix.vec_mul_vec v v)
  det Q = 0 → Q⁻¹ = 0 := by
sorry

end projection_matrix_inverse_is_zero_matrix_l808_808580


namespace sum_fractions_correct_l808_808702

def sum_of_fractions (f1 f2 f3 f4 f5 f6 f7 : ℚ) : ℚ :=
  f1 + f2 + f3 + f4 + f5 + f6 + f7

theorem sum_fractions_correct : sum_of_fractions (1/3) (1/2) (-5/6) (1/5) (1/4) (-9/20) (-5/6) = -5/6 :=
by
  sorry

end sum_fractions_correct_l808_808702


namespace count_integers_six_times_sum_of_digits_l808_808968

theorem count_integers_six_times_sum_of_digits (n : ℕ) (h : n < 1000) 
    (digit_sum : ℕ → ℕ)
    (digit_sum_correct : ∀ (n : ℕ), digit_sum n = (n % 10) + ((n / 10) % 10) + (n / 100)) :
    ∃! n, n < 1000 ∧ n = 6 * digit_sum n :=
sorry

end count_integers_six_times_sum_of_digits_l808_808968


namespace bottom_price_l808_808970

open Nat

theorem bottom_price (B T : ℕ) (h1 : T = B + 300) (h2 : 3 * B + 3 * T = 21000) : B = 3350 := by
  sorry

end bottom_price_l808_808970


namespace solve_system_of_equations_l808_808635

theorem solve_system_of_equations (x y : ℝ) (h1 : 2 * x + 3 * y = 7) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
    -- The proof is not required, so we put a sorry here.
    sorry

end solve_system_of_equations_l808_808635


namespace intersection_A_B_eq_B_l808_808048

variable (a : ℝ) (A : Set ℝ) (B : Set ℝ)

def satisfies_quadratic (a : ℝ) (x : ℝ) : Prop := x^2 - a*x + 1 = 0

def set_A : Set ℝ := {1, 2, 3}

def set_B (a : ℝ) : Set ℝ := {x | satisfies_quadratic a x}

theorem intersection_A_B_eq_B (a : ℝ) (h : a ∈ set_A) : 
  (∀ x, x ∈ set_B a → x ∈ set_A) → (∃ x, x ∈ set_A ∧ satisfies_quadratic a x) →
  a = 2 :=
sorry

end intersection_A_B_eq_B_l808_808048


namespace largest_gcd_value_l808_808697

open Nat

theorem largest_gcd_value (n : ℕ) : ∃ m ∈ {k | gcd (n^2 + 3) ((n + 1)^2 + 3) = k}, k = 13 :=
by
  sorry

end largest_gcd_value_l808_808697


namespace count_strictly_ordered_numbers_l808_808967

theorem count_strictly_ordered_numbers :
  let valid_numbers := 
    { n // n ∈ finset.range 500 1000 
          ∧ ∃ a b c : ℕ, 
           (100 * a + 10 * b + c = n 
            ∧ 5 ≤ a ∧ a ≤ 9 
            ∧ 0 ≤ b ∧ b ≤ 9 
            ∧ 0 ≤ c ∧ c ≤ 9 
            ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c 
            ∧ (a < b ∧ b < c ∨ a > b ∧ b > c) )}
  in finset.card valid_numbers = 20 :=
begin
  sorry
end

end count_strictly_ordered_numbers_l808_808967


namespace probability_divisible_by_20_l808_808128

/- Define the list of digits -/
def digits : List ℕ := [1, 1, 2, 3, 4, 5, 6]

/- Function to check if a number is divisible by 20 -/
def divisible_by_20 (n : ℕ) : Prop := n % 20 = 0

/- Function to form seven-digit integers from the list of digits -/
-- This is a noncomputable setup since permutations of digits are involved.
noncomputable def combinations_digits (lst: List ℕ) : List ℕ :=
  lst.permutations.map (λ l, l.foldl (λ sum digit, sum * 10 + digit) 0)

/- The theorem stating the probability that a random arrangement of the given digits forms a number divisible by 20 is 1/21 -/
theorem probability_divisible_by_20 :
  let numbers := combinations_digits digits in
  let count_div_20 := numbers.countp divisible_by_20 in
  let total := numbers.length in
  (count_div_20 : ℚ) / total = 1 / 21 := 
by
  sorry

end probability_divisible_by_20_l808_808128


namespace units_digit_of_power_l808_808402

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808402


namespace distinct_sums_possible_iff_even_l808_808822

theorem distinct_sums_possible_iff_even (n : ℕ) (hn : n ≥ 2) : 
  (∃ A : matrix (fin n) (fin n) ℤ, 
    (∀ i, ∃ row_sum : ℤ, (matrix.fin_sum vec_univ (λ j, A i j) = row_sum ∧ ∀ j ≠ j', row_sum j ≠ row_sum j')) ∧
    (∀ j, ∃ col_sum : ℤ, (matrix.fin_sum vec_univ (λ i, A i j) = col_sum ∧ ∀ i ≠ i', col_sum i ≠ col_sum i')) ∧
    (∀ i j, A i j ∈ {-1, 0, 1})) ↔ even n :=
begin
sorry
end

end distinct_sums_possible_iff_even_l808_808822


namespace z_real_iff_z_complex_iff_z_purely_imaginary_iff_l808_808446

noncomputable def z (m : ℝ) : ℂ := (↑(m - 4) : ℂ) + (↑(m^2 - 5*m - 6) * complex.I)

-- Proof Problem I
theorem z_real_iff (m : ℝ) : z m ∈ ℝ ↔ (m = 6 ∨ m = -1) :=
sorry

-- Proof Problem II
theorem z_complex_iff (m : ℝ) : z m ∉ ℝ ↔ (m ≠ 6 ∧ m ≠ -1) :=
sorry

-- Proof Problem III
theorem z_purely_imaginary_iff (m : ℝ) : (z m).re = 0 ↔ (m = 4) :=
sorry

end z_real_iff_z_complex_iff_z_purely_imaginary_iff_l808_808446


namespace smallest_positive_period_f_properties_of_f_if_min_value_is_1_l808_808851

noncomputable def a (x : ℝ) : ℝ × ℝ :=
( (sqrt 3 / 2) * sin x, 2 * cos x )

noncomputable def b (x : ℝ) : ℝ × ℝ :=
( 2 * cos x, (1 / 2) * cos x )

noncomputable def f (x m : ℝ) : ℝ :=
(a x).1 * (b x).1 + (a x).2 * (b x).2 + m

theorem smallest_positive_period_f :
  ∀ m, ∃ T > 0, T = π ∧ ∀ x, f(x + T, m) = f(x, m) :=
sorry

theorem properties_of_f_if_min_value_is_1 :
  ∃ m, (∀ x, ( f(x, m) = sin (2 * x + π / 6) + 2 ) ∧
   (∀ t, 1 ≤ f(t, m) ∧ f(t, m) ≤ 3) ∧ 
   (m = 3 / 2) ∧ 
   (∀ k : ℤ, ∃ x,  x = k * (π / 2) - π / 12 )) :=
sorry

end smallest_positive_period_f_properties_of_f_if_min_value_is_1_l808_808851


namespace system_solution_l808_808918

theorem system_solution (n : ℕ) (hn : 3 ≤ n) (x : ℕ → ℝ) :
  (∀ i, 1 ≤ i ∧ i < n → tan (x i) + 3 / tan (x i) = 2 * tan (x (i + 1))) ∧
  tan (x n) + 3 / tan (x n) = 2 * tan (x 1) →
  ∃ k : ℤ, ∃ s : Fin n → bool, (∀ i : Fin n, x i = ↑k * π + (if s i then π/3 else -π/3)) :=
sorry

end system_solution_l808_808918


namespace range_of_expr_l808_808636

noncomputable def expr (x y : ℝ) : ℝ := (x + 2 * y + 3) / (x + 1)

theorem range_of_expr : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ x → 4 * x + 3 * y ≤ 12 → 3 ≤ expr x y ∧ expr x y ≤ 11) :=
by
  sorry

end range_of_expr_l808_808636


namespace volume_prism_section_l808_808781

noncomputable def prism_volume_part (A B C A₁ B₁ C₁ E F : Point) (V : ℝ) : ℝ :=
  if E = midpoint A A₁ ∧ 
     divides_ratio F B B₁ 1 2 ∧ 
     intersection_plane E F C
  then (5 / 18) * V
  else 0

-- Definitions
def midpoint (A B : Point) : Point := sorry
def divides_ratio (F B B₁ : Point) (m n : ℕ) : Prop := sorry
def intersection_plane (E F C : Point) : Prop := sorry

-- Main theorem
theorem volume_prism_section (A B C A₁ B₁ C₁ E F : Point) (V : ℝ)
  (hE : E = midpoint A A₁)
  (hF : divides_ratio F B B₁ 1 2)
  (hP : intersection_plane E F C) :
  prism_volume_part A B C A₁ B₁ C₁ E F V = (5 / 18) * V :=
by
  sorry

end volume_prism_section_l808_808781


namespace first_pipe_fill_time_l808_808270

theorem first_pipe_fill_time (T : ℝ) :
  (∀ T : ℝ, T > 0 → 
   let fill_rate_pipe1 := 1 / T in
   let fill_rate_pipe2 := 1 / 12 in
   let empty_rate_pipe3 := 1 / 25 in
   let combined_rate := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_pipe3 in
   combined_rate * 6.976744186046512 = 1 → T = 10) :=
by
  intro T hT
  let fill_rate_pipe1 := 1 / T
  let fill_rate_pipe2 := 1 / 12
  let empty_rate_pipe3 := 1 / 25
  let combined_rate := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_pipe3
  have h : combined_rate * 6.976744186046512 = 1,
  { sorry } -- Placeholder for the proof that combined_rate * 6.976744186046512 = 1 implies T = 10
  sorry -- Final proof showing T = 10

end first_pipe_fill_time_l808_808270


namespace system_solution_l808_808442

theorem system_solution:
  let k := 115 / 12 
  ∃ x y z: ℝ, 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    (x + k * y + 5 * z = 0) ∧
    (4 * x + k * y - 3 * z = 0) ∧
    (3 * x + 5 * y - 4 * z = 0) ∧ 
    ((1 : ℝ) / 15 = (x * z) / (y * y)) := 
by sorry

end system_solution_l808_808442


namespace books_added_after_lunch_l808_808961

-- Definitions for the given conditions
def initial_books : Int := 100
def books_borrowed_lunch : Int := 50
def books_remaining_lunch : Int := initial_books - books_borrowed_lunch
def books_borrowed_evening : Int := 30
def books_remaining_evening : Int := 60

-- Let X be the number of books added after lunchtime
variable (X : Int)

-- The proof goal in Lean statement
theorem books_added_after_lunch (h : books_remaining_lunch + X - books_borrowed_evening = books_remaining_evening) :
  X = 40 := by
  sorry

end books_added_after_lunch_l808_808961


namespace typhoon_tree_survival_l808_808962

def planted_trees : Nat := 150
def died_trees : Nat := 92
def slightly_damaged_trees : Nat := 15

def total_trees_affected : Nat := died_trees + slightly_damaged_trees
def trees_survived_without_damages : Nat := planted_trees - total_trees_affected
def more_died_than_survived : Nat := died_trees - trees_survived_without_damages

theorem typhoon_tree_survival :
  more_died_than_survived = 49 :=
by
  -- Define the necessary computations and assertions
  let total_trees_affected := 92 + 15
  let trees_survived_without_damages := 150 - total_trees_affected
  let more_died_than_survived := 92 - trees_survived_without_damages
  -- Prove the statement
  have : total_trees_affected = 107 := rfl
  have : trees_survived_without_damages = 43 := rfl
  have : more_died_than_survived = 49 := rfl
  exact this

end typhoon_tree_survival_l808_808962


namespace range_of_lambda_l808_808496

theorem range_of_lambda (a b λ: ℝ) 
  (h : ∀ a b : ℝ, a^2 + 8 * b^2 ≥ λ * b * (a + b)) : 
  -8 ≤ λ ∧ λ ≤ 4 :=
sorry

end range_of_lambda_l808_808496


namespace mass_of_barium_sulfate_l808_808823

-- Definitions of the chemical equation and molar masses
def barium_molar_mass : ℝ := 137.327
def sulfur_molar_mass : ℝ := 32.065
def oxygen_molar_mass : ℝ := 15.999
def molar_mass_BaSO4 : ℝ := barium_molar_mass + sulfur_molar_mass + 4 * oxygen_molar_mass

-- Given conditions
def moles_BaBr2 : ℝ := 4
def moles_BaSO4_produced : ℝ := moles_BaBr2 -- from balanced equation

-- Calculate mass of BaSO4 produced
def mass_BaSO4 : ℝ := moles_BaSO4_produced * molar_mass_BaSO4

-- Mass of Barium sulfate produced
theorem mass_of_barium_sulfate : mass_BaSO4 = 933.552 :=
by 
  -- Skip the proof
  sorry

end mass_of_barium_sulfate_l808_808823


namespace lisa_daily_walking_time_l808_808594

theorem lisa_daily_walking_time :
  (walk_speed : ℕ) (daily_distance : ℕ) (total_distance : ℕ) (days : ℕ)
  (h1 : walk_speed = 10)
  (h2 : total_distance = 1200)
  (h3 : days = 2)
  (h4 : daily_distance = total_distance / days) :
  daily_distance / walk_speed / 60 = 1 :=
by sorry

end lisa_daily_walking_time_l808_808594


namespace max_value_carl_can_carry_l808_808544

structure Stone :=
  (weight : ℕ)
  (value : ℕ)

def stones : List Stone :=
  [⟨7, 16⟩, ⟨3, 9⟩, ⟨2, 4⟩]

def max_weight : ℕ := 20
def max_7_pound_stones : ℕ := 2

def total_value (stones_list : List Stone) (counts : List ℕ) : ℕ :=
  (List.zip stones_list counts).map (λ ⟨s, c⟩ => s.value * c).sum

def total_weight (stones_list : List Stone) (counts : List ℕ) : ℕ :=
  (List.zip stones_list counts).map (λ ⟨s, c⟩ => s.weight * c).sum

theorem max_value_carl_can_carry : ∃ counts : List ℕ,
  List.length counts = List.length stones ∧
  (total_weight stones counts ≤ max_weight ∧
   counts.head (1) ≤ max_7_pound_stones ∧
   total_value stones counts = 58) :=
sorry

end max_value_carl_can_carry_l808_808544


namespace trapezoid_area_l808_808340

theorem trapezoid_area
  (A B C D E : Type)
  (triangle_ABC_is_isosceles : AB = AC)
  (all_triangles_are_similar : ∀ (T1 T2 : Triangle), similar T1 T2)
  (area_smallest_triangle_is_1 : area smallest_triangle = 1)
  (area_triangle_ABC_is_40 : area triangle_ABC = 40) :
  area trapezoid_DBCE = 24 :=
sorry

end trapezoid_area_l808_808340


namespace combined_capacity_is_40_l808_808613

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l808_808613


namespace x_coordinate_of_A_l808_808532

def parabola_equation (A : ℝ × ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

def distance_to_focus (A : ℝ × ℝ) (focus : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - focus.1) ^ 2 + (A.2 - focus.2) ^ 2)

theorem x_coordinate_of_A (A : ℝ × ℝ) (h1 : parabola_equation A) (h2 : distance_to_focus A (1, 0) = 6) : A.1 = 7 :=
  sorry

end x_coordinate_of_A_l808_808532


namespace units_digit_7_power_l808_808398

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808398


namespace confidence_interval_95_l808_808384

noncomputable def confidence_interval_for_population_mean (sample_mean : ℝ) (n : ℕ) (sigma : ℝ) (confidence_level : ℝ) : set ℝ :=
  let t := 1.96 in -- Z-score for 0.975
  let delta := t * sigma / real.sqrt n in
  set.Icc (sample_mean - delta) (sample_mean + delta)

theorem confidence_interval_95
  (sample_mean : ℝ) (n : ℕ) (sigma : ℝ) (confidence_level : ℝ)
  (h₁ : sample_mean = 10.43)
  (h₂ : n = 100)
  (h₃ : sigma = 5)
  (h₄ : confidence_level = 0.95) :
  confidence_interval_for_population_mean sample_mean n sigma confidence_level = set.Icc 9.45 11.41 :=
by {
  sorry
}

end confidence_interval_95_l808_808384


namespace largest_invertible_interval_l808_808357

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 8

theorem largest_invertible_interval :
  ∃ I : set ℝ, (I = set.Iic (-1)) ∧ (∀ x ∈ I, f x) ∧ (-3 ∈ I) ∧ (∀ x1 x2 ∈ I, x1 ≠ x2 → f x1 ≠ f x2) :=
begin
  sorry
end

end largest_invertible_interval_l808_808357


namespace radical_conjugate_sum_l808_808006

theorem radical_conjugate_sum:
  let a := 15 - Real.sqrt 500
  let b := 15 + Real.sqrt 500
  3 * (a + b) = 90 :=
by
  sorry

end radical_conjugate_sum_l808_808006


namespace graphs_with_inverses_l808_808832

axiom Graph : Type
variable (F G H I J : Graph)

def horizontal_line_test (g : Graph) : Prop := 
  ∀ y : ℝ, ∃! x : ℝ, g x = y

theorem graphs_with_inverses :
  (horizontal_line_test G) ∧ 
  (horizontal_line_test H) ∧ 
  (horizontal_line_test J) ∧ 
  ¬(horizontal_line_test F) ∧ 
  ¬(horizontal_line_test I) := 
sorry

end graphs_with_inverses_l808_808832


namespace find_p_l808_808973

variables (p q : ℚ)
variables (h1 : 2 * p + 5 * q = 10) (h2 : 5 * p + 2 * q = 20)

theorem find_p : p = 80 / 21 :=
by sorry

end find_p_l808_808973


namespace quadratic_has_real_roots_iff_l808_808534

theorem quadratic_has_real_roots_iff (k : ℝ) (hk : k ≠ 0) :
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ k ≤ 1 / 4 :=
by
  sorry

end quadratic_has_real_roots_iff_l808_808534


namespace three_pow_2010_mod_eight_l808_808281

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end three_pow_2010_mod_eight_l808_808281


namespace lcm_of_numbers_l808_808127

theorem lcm_of_numbers (a b lcm hcf : ℕ) (h_prod : a * b = 45276) (h_hcf : hcf = 22) (h_relation : a * b = hcf * lcm) : lcm = 2058 :=
by sorry

end lcm_of_numbers_l808_808127


namespace pyramid_volume_correct_l808_808915

open EuclideanGeometry

noncomputable def pyramid_volume (S A B C: Point) (BC : ℝ) (B_eq_C : dist S B = dist S C) 
  (H_eq_orthocenter : orthocenter S A B C) (BC_length : BC = 2) (dihedral_angle : ∠(S, B, C) = π / 3) : ℝ :=
  volume_pyramid S A B C = sqrt 3 / 3

theorem pyramid_volume_correct (S A B C: Point) (BC : ℝ) (B_eq_C : dist S B = dist S C)
  (H_eq_orthocenter : orthocenter S A B C) (BC_length : BC = 2) (dihedral_angle : ∠(S, B, C) = π / 3) :
  pyramid_volume S A B C BC B_eq_C H_eq_orthocenter BC_length dihedral_angle = sqrt 3 / 3 :=
sorry

end pyramid_volume_correct_l808_808915


namespace exists_b_for_a_ge_condition_l808_808383

theorem exists_b_for_a_ge_condition (a : ℝ) (h : a ≥ -Real.sqrt 2 - 1 / 4) :
  ∃ b : ℝ, ∃ x y : ℝ, 
    y = x^2 - a ∧
    x^2 + y^2 + 8 * b^2 = 4 * b * (y - x) + 1 :=
sorry

end exists_b_for_a_ge_condition_l808_808383


namespace jimmys_speed_l808_808595

theorem jimmys_speed 
(Mary_speed : ℕ) (total_distance : ℕ) (t : ℕ)
(h1 : Mary_speed = 5)
(h2 : total_distance = 9)
(h3 : t = 1)
: ∃ (Jimmy_speed : ℕ), Jimmy_speed = 4 :=
by
  -- calculation steps skipped here
  sorry

end jimmys_speed_l808_808595


namespace solution_exists_l808_808028

open Real

theorem solution_exists (x : ℝ) (h1 : x > 9) (h2 : sqrt (x - 3 * sqrt (x - 9)) + 3 = sqrt (x + 3 * sqrt (x - 9)) - 3) : x ≥ 18 :=
sorry

end solution_exists_l808_808028


namespace farmer_total_acres_l808_808747

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l808_808747


namespace sector_max_area_l808_808484

variable {r l : ℝ}

theorem sector_max_area (h1 : 2 * r + l = 4) : 2 * 2 * (1/2 * l * r) ≤ 2 := by
  have h2 : (1 / 2) * l * r = 1 := sorry
  rw ← h2
  exact le_of_eq (eq.refl 1)

end sector_max_area_l808_808484


namespace smallest_possible_lcm_l808_808672

theorem smallest_possible_lcm (a b c d : ℕ) (n : ℕ) :
  gcd a b c d = 154 →
  (∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), S.card = 154000 ∧ 
     ∀ (x ∈ S), gcd x.1 x.2 x.3 x.4 = 154) →
  n = 25520328 :=
by sorry

end smallest_possible_lcm_l808_808672


namespace description_of_T_l808_808165

-- Define the conditions
def T := { p : ℝ × ℝ | (∃ (c : ℝ), ((c = 5 ∨ c = p.1 + 3 ∨ c = p.2 - 6) ∧ (5 ≥ p.1 + 3) ∧ (5 ≥ p.2 - 6))) }

-- The main theorem
theorem description_of_T : 
  ∃ p : ℝ × ℝ, 
    (p = (2, 11)) ∧ 
    ∀ q ∈ T, 
      (q.fst = 2 ∧ q.snd ≤ 11) ∨ 
      (q.snd = 11 ∧ q.fst ≤ 2) ∨ 
      (q.snd = q.fst + 9 ∧ q.fst ≤ 2) :=
sorry

end description_of_T_l808_808165


namespace smallest_x_for_defined_expr_l808_808825

noncomputable def is_defined_expr (x : ℝ) : Prop :=
  ∃ y1 y2 y3 y4 : ℝ, y1 = x^(1 / 2) ∧ y1 > 2008 ∧
                     y2 = log 2007 y1 ∧ y2 > 2008 ∧
                     y3 = log 2008 y2 ∧ y3 > 2008 ∧
                     y4 = log 2009 y3 ∧ y4 > 2008

theorem smallest_x_for_defined_expr :
  ∀ x : ℝ, is_defined_expr x ↔ x > 2008^2 :=
by sorry

end smallest_x_for_defined_expr_l808_808825


namespace units_digit_of_power_l808_808405

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808405


namespace all_statements_true_l808_808036

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem all_statements_true :
  (∀ (x : ℤ), floor (x + 0.5) = floor x + 1) ∧
  (∀ (x y : ℝ), x < 1 → y < 1 → 
                x ≠ floor x ∧ y ≠ floor y → floor (x + y) = floor x + floor y) ∧
  (floor (2.3 + 3.2) = floor 2.3 + floor 3.2) :=
by
  sorry

end all_statements_true_l808_808036


namespace concentric_circles_radius_difference_l808_808693

theorem concentric_circles_radius_difference (r R : ℝ)
  (h : R^2 = 4 * r^2) :
  R - r = r :=
by
  sorry

end concentric_circles_radius_difference_l808_808693


namespace max_value_of_sum_of_12th_powers_l808_808175

theorem max_value_of_sum_of_12th_powers (x : Fin 1997 → ℝ) 
  (h1 : ∀ i, -1 / Real.sqrt 3 ≤ x i ∧ x i ≤ Real.sqrt 3)
  (h2 : ∑ i, x i = -318 * Real.sqrt 3) : 
  ∑ i, (x i) ^ 12 ≤ 189548 :=
sorry

end max_value_of_sum_of_12th_powers_l808_808175


namespace problem1_problem2_l808_808205

theorem problem1 (x : ℝ) : (x + 4) ^ 2 - 5 * (x + 4) = 0 → x = -4 ∨ x = 1 :=
by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 2 * x - 15 = 0 → x = -3 ∨ x = 5 :=
by
  sorry

end problem1_problem2_l808_808205


namespace triangle_dimensions_l808_808328

theorem triangle_dimensions (a : ℕ) (hₐ : a = 10) : 
  ∃ b c d, (b = a ∧ c = a ∧ d = a) :=
by
  use 10, 10, 10
  rw hₐ
  exact ⟨rfl, rfl, rfl⟩

end triangle_dimensions_l808_808328


namespace selection_methods_with_both_genders_l808_808045

noncomputable def selection_methods_count : ℕ :=
  (finset.card (finset.Ico (0 : ℕ) 10 : finset ℕ)).choose 5 -
  (finset.card (finset.Ico (0 : ℕ) 7 : finset ℕ)).choose 5

theorem selection_methods_with_both_genders :
  selection_methods_count = 120 :=
  by sorry

end selection_methods_with_both_genders_l808_808045


namespace minimum_perimeter_l808_808254

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l808_808254


namespace min_students_in_both_clubs_l808_808994

theorem min_students_in_both_clubs 
  (total_students : ℕ := 33)
  (club_1_required_percent : ℝ := 0.7)
  (club_2_required_percent : ℝ := 0.7)
  (min_club_1_students min_club_2_students : ℕ)
  (student_in_both_clubs : ℕ) :
  min_club_1_students = ceil (33 * 0.7) → min_club_2_students = ceil (33 * 0.7) →
  min_club_1_students + min_club_2_students - total_students = student_in_both_clubs →
  student_in_both_clubs = 15 :=
by
  intro h1 h2 h3
  sorry

end min_students_in_both_clubs_l808_808994


namespace cuboid_faces_meeting_at_vertex_l808_808676

-- Define a cuboid.
structure Cuboid where

-- State the theorem about the number of faces meeting at one vertex of a cuboid.
theorem cuboid_faces_meeting_at_vertex (c : Cuboid) : ∃ n : ℕ, n = 3 :=
by
  use 3
  sorry

end cuboid_faces_meeting_at_vertex_l808_808676


namespace pounds_per_camper_approx_l808_808185

def weight_trout : ℝ := 8
def number_salmons : ℕ := 2
def weight_each_salmon : ℝ := 12
def number_of_campers : ℕ := 22

def total_weight_fish : ℝ := weight_trout + (number_salmons * weight_each_salmon)
def pounds_per_camper : ℝ := total_weight_fish / number_of_campers

theorem pounds_per_camper_approx : pounds_per_camper ≈ 1.45 := by
  sorry

end pounds_per_camper_approx_l808_808185


namespace pencil_count_l808_808187

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l808_808187


namespace solution_set_absolute_value_sum_eq_three_l808_808486

theorem solution_set_absolute_value_sum_eq_three (m n : ℝ) (h : ∀ x : ℝ, (|2 * x - 3| ≤ 1) ↔ (m ≤ x ∧ x ≤ n)) : m + n = 3 :=
sorry

end solution_set_absolute_value_sum_eq_three_l808_808486


namespace probability_score_l808_808542

/-- Given:
1. A bag with 4 red balls and 3 black balls.
2. 4 balls drawn from the bag.
3. Drawing 1 red ball scores 1 point.
4. Drawing 1 black ball scores 3 points.
5. Score is a random variable ξ.

Prove that the probability P(ξ ≤ 7) equals 13/35.
-/
theorem probability_score (R B : ℕ) (drawn : ℕ) (score_red score_black : ℕ) (ξ : ℕ → ℕ) :
  R = 4 → B = 3 → drawn = 4 → score_red = 1 → score_black = 3 →
  (∀ n, ξ n = if n = 0 then 4 else if n = 1 then 6 else if n = 2 then 8 else if n = 3 then 10 else 0) →
  ∑ i in finset.range (ξ 2 + 1), if ξ i ≤ 7 then 1 else 0 / (nat.choose (R + B) drawn) = 13 / 35 :=
by
  intros hR hB hDrawn hscore_red hscore_black hξ sorry

end probability_score_l808_808542


namespace h_oplus_h_op_h_equals_h_l808_808013

def op (x y : ℝ) : ℝ := x^3 - y

theorem h_oplus_h_op_h_equals_h (h : ℝ) : op h (op h h) = h := by
  sorry

end h_oplus_h_op_h_equals_h_l808_808013


namespace pairing_product_is_square_l808_808380

theorem pairing_product_is_square (n : ℕ) (h : n > 1) :
  ∃ (pairs : list (ℕ × ℕ)), (∀ (p ∈ pairs), p.1 + p.2 ∈ list.range (2 * n + 1) ∧
  (list.foldr (*) 1 (list.map (λ q : ℕ × ℕ, q.1 + q.2) pairs) ∈ {k | ∃ m : ℕ, k = m * m})) := sorry

end pairing_product_is_square_l808_808380


namespace part1_part2_l808_808959

variables (x : ℝ) (x1 x2 : ℝ)
noncomputable def a := (2 * Real.cos x, 1 : ℝ)
noncomputable def b := (-Real.cos (x + π / 3), 1 / 2 : ℝ)
noncomputable def f (x : ℝ) := 2 * Real.cos x * (-Real.cos (x + π / 3)) + 1 / 2

theorem part1 (hx : x ∈ Set.Icc 0 (π / 2)) (h_parallel : 
  (2 * Real.cos x / -Real.cos (x + π / 3)) = (1 / (1 / 2))) : x = π / 3 :=
sorry

theorem part2 (hx1 : x1 ∈ Set.Icc 0 (π / 2)) (hx2 : x2 ∈ Set.Icc 0 (π / 2))
  (h_abs : |f x1 - f x2| ≤ 1.5) : ∃ λ >= 0, ∀ x1 x2, 
  (x1 ∈ Set.Icc 0 (π / 2)) → (x2 ∈ Set.Icc 0 (π / 2)) → (|f x1 - f x2| ≤ λ) :=
sorry

end part1_part2_l808_808959


namespace constant_t_fixed_l808_808247

-- Definition of the parabola and conditions of the problem
def parabola (x : ℝ) : ℝ := 4 * x ^ 2

-- Define the key point C and the chord AB passing through it
variables (d : ℝ) (C : ℝ × ℝ)
def is_chord_passing_C (A B : ℝ × ℝ) : Prop :=
  C = (0, d) ∧ parabola A.1 = A.2 ∧ parabola B.1 = B.2 ∧ A.2 = B.2 ∧ A.2 = d

-- Define the fixed constant t
def fixed_constant (A B : ℝ × ℝ) : ℝ :=
  1 / (real.sqrt (A.1 ^ 2 + (A.2 - d) ^ 2)) + 1 / (real.sqrt (B.1 ^ 2 + (B.2 - d) ^ 2))

-- Main theorem
theorem constant_t_fixed (d : ℝ) (C : ℝ × ℝ)
  (h1 : C = (0, d))
  (h2 : ∃ (A B : ℝ × ℝ), is_chord_passing_C d C A B)
  : fixed_constant d = 1 :=
sorry

end constant_t_fixed_l808_808247


namespace clocks_chime_together_l808_808692

theorem clocks_chime_together :
  ∃ t : ℕ, Nat.lcm 15 25 = t ∧ t = 75 :=
by
  use 75
  split
  . exact Nat.lcm_eq_left (Nat.gcd 15 25).symm
  . rfl

end clocks_chime_together_l808_808692


namespace proof_part_1_proof_part_2_l808_808958

variables {R : Type*} [IsRealField R]

-- Declare vectors a, b, c in ℝ×ℝ plane
def vector_a : EuclideanSpace ℝ (Fin 2) := ![1, 2]
def vector_b : EuclideanSpace ℝ (Fin 2)
def vector_c : EuclideanSpace ℝ (Fin 2) := ![-2, -4]

-- Given conditions
axiom norm_c_eq : ‖vector_c‖ = 2 * Real.sqrt 5
axiom norm_b_eq : ‖vector_b‖ = Real.sqrt 5 / 2
axiom dotprod_eq : (vector_a + 2 • vector_b).dot_product (2 • vector_a - vector_b) = 15 / 4

theorem proof_part_1 : vector_c = ![-2, -4] := sorry

theorem proof_part_2 : (vector_a.dot_product vector_b) / ‖vector_b‖ = -Real.sqrt 5 / 2 := sorry

end proof_part_1_proof_part_2_l808_808958


namespace perfect_squares_of_diophantine_l808_808620

theorem perfect_squares_of_diophantine (a b : ℤ) (h : 2 * a^2 + a = 3 * b^2 + b) :
  ∃ k m : ℤ, (a - b) = k^2 ∧ (2 * a + 2 * b + 1) = m^2 := by
  sorry

end perfect_squares_of_diophantine_l808_808620


namespace compute_expression_l808_808814

theorem compute_expression : 7^2 - 5 * 6 + 6^2 = 55 := by
  sorry

end compute_expression_l808_808814


namespace volume_of_tetrahedron_P_ABCD_l808_808064

noncomputable def volume_of_tetrahedron
  (A B C D P : Type)
  (dist_AB : ℝ := real.sqrt 2)
  (dist_BC : ℝ := real.sqrt 2)
  (dist_PA : ℝ := 1)
  (dist_PB : ℝ := 1)
  (dist_PC : ℝ := 1)
  (dist_PD : ℝ := 1) : ℝ := sorry

theorem volume_of_tetrahedron_P_ABCD :
  ∀ (A B C D P : Type)
    (dist_AB : ℝ = real.sqrt 2)
    (dist_BC : ℝ = real.sqrt 2)
    (dist_PA : ℝ = 1)
    (dist_PB : ℝ = 1)
    (dist_PC : ℝ = 1)
    (dist_PD : ℝ = 1),
  volume_of_tetrahedron A B C D P dist_AB dist_BC dist_PA dist_PB dist_PC dist_PD = real.sqrt 2 / 6 :=
begin
  sorry
end

end volume_of_tetrahedron_P_ABCD_l808_808064


namespace units_digit_of_7_pow_6_pow_5_l808_808440

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808440


namespace find_midpoint_in_polar_l808_808132

noncomputable def midpoint_polar_coordinates (r₁ θ₁ r₂ θ₂ : ℝ) [h₁ : 0 ≤ θ₁] [h₂ : θ₁ < 2 * Real.pi] [h₃ : 0 ≤ θ₂] [h₄ : θ₂ < 2 * Real.pi] :
  ℝ × ℝ := let x₁ := r₁ * Real.cos θ₁
              y₁ := r₁ * Real.sin θ₁
              x₂ := r₂ * Real.cos θ₂
              y₂ := r₂ * Real.sin θ₂
              mx := (x₁ + x₂) / 2
              my := (y₁ + y₂) / 2
              mr := Real.sqrt (mx^2 + my^2)
              mθ := Real.atan2 my mx
  in (mr, mθ)

theorem find_midpoint_in_polar :
  midpoint_polar_coordinates 10 (Real.pi / 4) 10 (3 * Real.pi / 4) = (5 * Real.sqrt 2, Real.pi / 2) := 
sorry

end find_midpoint_in_polar_l808_808132


namespace train_times_comparison_l808_808674

-- Defining the given conditions
variables (V1 T1 T2 D : ℝ)
variables (h1 : T1 = 2) (h2 : T2 = 7/3)
variables (train1_speed : V1 = D / T1)
variables (train2_speed : V2 = (3/5) * V1)

-- The proof statement to show that T2 is 1/3 hour longer than T1
theorem train_times_comparison 
  (h1 : (6/7) * V1 = D / (T1 + 1/3))
  (h2 : (3/5) * V1 = D / (T2 + 1)) :
  T2 - T1 = 1/3 :=
sorry

end train_times_comparison_l808_808674


namespace unique_integer_solution_l808_808965

theorem unique_integer_solution :
  ∃! (z : ℤ), 5 * z ≤ 2 * z - 8 ∧ -3 * z ≥ 18 ∧ 7 * z ≤ -3 * z - 21 :=
by
  sorry

end unique_integer_solution_l808_808965


namespace a_5_is_9_l808_808462

def sequence_sum (n : ℕ) : ℕ := n * n

def a (n : ℕ) : ℕ := 
  if n = 1 then sequence_sum n else sequence_sum n - sequence_sum (n - 1)

theorem a_5_is_9 : a 5 = 9 :=
by 
  have S_5 := sequence_sum 5
  have S_4 := sequence_sum 4
  have a_5 := S_5 - S_4
  show a_5 = 9
  sorry

end a_5_is_9_l808_808462


namespace units_digit_of_7_pow_6_pow_5_l808_808437

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808437


namespace non_overlapping_original_sets_l808_808882

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808882


namespace units_digit_pow_7_6_5_l808_808419

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808419


namespace complex_solution_l808_808218

noncomputable def verify_complex (z : ℂ) : Prop := 
  (1 + complex.i) * z = complex.abs (-4 * complex.i)

theorem complex_solution : verify_complex (2 - 2 * complex.i) :=
by
  -- verification steps
  sorry

end complex_solution_l808_808218


namespace number_of_non_congruent_triangles_l808_808921

theorem number_of_non_congruent_triangles :
  ∃ q : ℕ, q = 3 ∧ 
    (∀ (a b : ℕ), (a ≤ 2 ∧ 2 ≤ b) → (a + 2 > b) ∧ (a + b > 2) ∧ (2 + b > a) →
    (q = 3)) :=
by
  sorry

end number_of_non_congruent_triangles_l808_808921


namespace problem1_sequence_formula_problem2_find_n_l808_808461

noncomputable def a (n : ℕ) := if n = 1 then (2 / 3 : ℝ) else 2 * (1/3) ^ n

def S (n : ℕ) := ∑ i in finset.range (n + 1), a i

def b (n : ℕ) := real.log (1 - S (n + 1)) / real.log 3

theorem problem1_sequence_formula (n : ℕ) (hn : 0 < n) :
  a n = 2 * (1/3) ^ n :=
by sorry

theorem problem2_find_n (n : ℕ) (hn : 0 < n) :
  (∑ i in finset.range n, 1 / (b i * b (i + 1)) = 25 / 51) →
  n = 100 :=
by sorry

end problem1_sequence_formula_problem2_find_n_l808_808461


namespace area_of_triangle_BEF_l808_808341

open Real

theorem area_of_triangle_BEF (a b x y : ℝ) (h1 : a * b = 30) (h2 : (1/2) * abs (x * (b - y) + a * b - a * y) = 2) (h3 : (1/2) * abs (x * (-y) + a * y - x * b) = 3) :
  (1/2) * abs (x * y) = 35 / 8 :=
by
  sorry

end area_of_triangle_BEF_l808_808341


namespace digit_Q_is_0_l808_808940

theorem digit_Q_is_0 (M N P Q : ℕ) (hM : M < 10) (hN : N < 10) (hP : P < 10) (hQ : Q < 10) 
  (add_eq : 10 * M + N + 10 * P + M = 10 * Q + N) 
  (sub_eq : 10 * M + N - (10 * P + M) = N) : Q = 0 := 
by
  sorry

end digit_Q_is_0_l808_808940


namespace june_round_trip_time_l808_808572

theorem june_round_trip_time (d_June_Julia : ℝ) (t_June_Julia : ℝ) (d_June_Bernard : ℝ) :
    d_June_Julia = 2 ∧ t_June_Julia = 6 ∧ d_June_Bernard = 5 →
    let speed := d_June_Julia / t_June_Julia in
    let time_to_Bernard := d_June_Bernard / speed in
    let return_trip_time := time_to_Bernard in
    time_to_Bernard + return_trip_time = 30 :=
by
  intro h
  let ⟨h1, h2, h3⟩ := h
  let speed := d_June_Julia / t_June_Julia
  let time_to_Bernard := d_June_Bernard / speed
  let return_trip_time := time_to_Bernard
  have h4 : speed = 1 / 3 := by
    rw [h1, h2]
    exact congrArg (/) (by norm_num, by norm_num)
  have h5 : time_to_Bernard + return_trip_time = 30 := by
    rw [←h3]
    rw [h4]
    norm_num
  exact h5
  sorry

end june_round_trip_time_l808_808572


namespace sequence_inequality_l808_808473

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₇ : a 7 = 0) :
  ∃ k : ℕ, k ≤ 5 ∧ a k + a (k + 2) ≤ a (k + 1) * Real.sqrt 3 := 
sorry

end sequence_inequality_l808_808473


namespace closed_path_has_even_length_l808_808182

   theorem closed_path_has_even_length 
     (u d r l : ℤ) 
     (hu : u = d) 
     (hr : r = l) : 
     ∃ k : ℤ, 2 * (u + r) = 2 * k :=
   by
     sorry
   
end closed_path_has_even_length_l808_808182


namespace farmer_total_acres_l808_808749

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l808_808749


namespace tom_wins_competition_probability_l808_808249

theorem tom_wins_competition_probability :
  let tom_hits := 4 / 5
      geri_hits := 2 / 3
      tom_wins_if_hits_geri_misses := tom_hits * (1 - geri_hits)
      both_hit_or_miss := (tom_hits * geri_hits) + ((1 - tom_hits) * (1 - geri_hits))
      p := tom_wins_if_hits_geri_misses + both_hit_or_miss * p  -- recursive probability
  in p = 2 / 3 := 
sorry

end tom_wins_competition_probability_l808_808249


namespace rachel_took_money_l808_808625

theorem rachel_took_money (x y : ℕ) (h₁ : x = 5) (h₂ : y = 3) : x - y = 2 :=
by {
  sorry
}

end rachel_took_money_l808_808625


namespace triangle_is_right_l808_808144

noncomputable def is_right_triangle {A B C : Type} [EuclideanGeometry.angles A B C] 
(α β γ : C) : Prop :=
γ = 90

theorem triangle_is_right (A B C : Type) [EuclideanGeometry.angles A B C]
  (α β γ : C) (h : sin γ = sin α * cos β) : is_right_triangle α β γ :=
by
  sorry

end triangle_is_right_l808_808144


namespace boat_distance_upstream_l808_808734

-- Definitions based on conditions
def distance_downstream := 100 -- km
def time_downstream := 4 -- hours
def time_upstream := 15 -- hours
def speed_stream := 10 -- km/h

-- Given:
-- distance_downstream = 100 km
-- time_downstream = 4 hours
-- speed_stream = 10 km/h
-- time_upstream = 15 hours
-- Prove: Distance upstream = 75 km
theorem boat_distance_upstream :
  let speed_boat := (distance_downstream / time_downstream) - speed_stream in
  let speed_upstream := speed_boat - speed_stream in
  speed_upstream * time_upstream = 75 := 
by
  unfold distance_downstream time_downstream speed_stream time_upstream
  unfold let speed_boat := (100 / 4) - 10 
  unfold let speed_upstream := 15 - 10 
  conv_lhs { simp only [if_t_t_simple_eq_t] }
  unfold let  5 * 15 
  rw [nat.one_mul, nat.two_mul, nat.add_mul, nat.zero_mul]
  trivial
-- sorry

end boat_distance_upstream_l808_808734


namespace units_digit_7_pow_6_pow_5_l808_808420

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808420


namespace farmer_total_acres_l808_808752

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l808_808752


namespace inequality_implies_sum_nonneg_l808_808976

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := real.log x / real.log b

theorem inequality_implies_sum_nonneg (x y : ℝ) :
  (log_base 2 3) ^ x - (log_base 5 3) ^ x ≥ (log_base 2 3) ^ (-y) - (log_base 5 3) ^ (-y) →
  x + y ≥ 0 :=
by
  sorry

end inequality_implies_sum_nonneg_l808_808976


namespace sufficient_but_not_necessary_conditions_l808_808337

-- Definitions for the conditions
def condition_A (k x : ℝ) : Prop := 
  k > 1 → ∀ b : ℝ, (λ x : ℝ, k * x + b) x > x

def condition_B (m : ℝ) : Prop := 
  m < 1 → ∀ x > 1, (λ x : ℝ, x^2 - 2 * m * x) x > x

def condition_D (a b : ℝ) : Prop := 
  a < 0 → b < 0 → a + b < 0

-- Proving the correct answer set
theorem sufficient_but_not_necessary_conditions : 
  ({condition_A, condition_B, condition_D} : set (ℝ → Prop)) = {A, B, D} :=
sorry

end sufficient_but_not_necessary_conditions_l808_808337


namespace probability_no_positive_roots_l808_808338

def ordered_pair_set : Set (ℤ × ℤ) :=
  {p | abs p.1 ≤ 6 ∧ abs p.2 ≤ 6 }

def has_positive_real_root (b c : ℤ) : Prop :=
  let Δ := b^2 - 4 * c in
  Δ ≥ 0 ∧ (sqrt Δ > b ∨ -sqrt Δ > b)

def count_valid_pairs (s : Set (ℤ × ℤ)) : ℕ :=
  (s.filter (λ p => ¬ has_positive_real_root p.1 p.2)).card

theorem probability_no_positive_roots :
  let total_pairs := 169 in
  let valid_pairs := count_valid_pairs ordered_pair_set in
  ∃ N : ℕ, valid_pairs = N ∧ (N / total_pairs = valid_pairs / total_pairs) :=
sorry

end probability_no_positive_roots_l808_808338


namespace smallest_n_for_f_eq_4_l808_808285

def f (n : ℕ) : ℕ :=
  (Finset.filter (λ a => ∃ b, a^2 + b^2 = n ∧ a ≠ b)
    ((Finset.range n).product (Finset.range n))).card

theorem smallest_n_for_f_eq_4 : ∃ n : ℕ, f n = 4 ∧ ∀ m < n, f m ≠ 4 :=
by
  existsi 25
  split
  . unfold f
    sorry
  . intro m hm
    sorry

end smallest_n_for_f_eq_4_l808_808285


namespace shortest_distance_P_R_through_cube_l808_808770

theorem shortest_distance_P_R_through_cube :
  ∀ (PQ QR : ℝ), PQ = 20 ∧ QR = 15 →
  (let QS := Real.sqrt (PQ^2 + QR^2),
       XP := QR * (PQ / QS),
       YR := QR * (PQ / QS),
       XY := QS - XP - YR,
       RX := Real.sqrt (YR^2 + XY^2),
       PR := Real.sqrt (XP^2 + RX^2)
  in PR = Real.sqrt 337) := 
by
  intros PQ QR h,
  rcases h with ⟨hPQ, hQR⟩,
  let QS := Real.sqrt (PQ^2 + QR^2),
  let XP := QR * (PQ / QS),
  let YR := QR * (PQ / QS),
  let XY := QS - XP - YR,
  let RX := Real.sqrt (YR^2 + XY^2),
  let PR := Real.sqrt (XP^2 + RX^2),
  sorry

end shortest_distance_P_R_through_cube_l808_808770


namespace probability_three_girls_l808_808744

theorem probability_three_girls (total : ℕ) (boys : ℕ) (girls : ℕ) (choose : ℕ → ℕ → ℕ)
  (h_total : total = 12) (h_boys : boys = 7) (h_girls : girls = 5) : choose girls 3 / choose total 3 = 1 / 22 :=
by
  -- necessary definitions and conditions
  have h1 : total = 12 := h_total,
  have h2 : boys = 7 := h_boys,
  have h3 : girls = 5 := h_girls,
  have h4 : choose = λ n k, n.choose k := sorry, -- Assuming a function choose available

  -- goal statement
  sorry

end probability_three_girls_l808_808744


namespace problem_statement_l808_808802

noncomputable def scores : List ℕ := [18, 22, 25, 19]

def highest_score : ℕ := scores.maximum.get_or_else 0

def lowest_score : ℕ := scores.minimum.get_or_else 0

def diff_square_roots : ℝ := real.sqrt highest_score - real.sqrt lowest_score

def average_score : ℝ := (scores.sum / scores.length)

def cube_average_score : ℝ := average_score^3

theorem problem_statement :
  diff_square_roots ≈ 0.7574 ∧ cube_average_score = 9261 :=
by 
  sorry

end problem_statement_l808_808802


namespace train_crosses_bridge_in_time_l808_808779

noncomputable def length_of_train : ℝ := 125
noncomputable def length_of_bridge : ℝ := 250.03
noncomputable def speed_of_train_kmh : ℝ := 45

noncomputable def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_time :
  time_to_cross_bridge = 30.0024 :=
  sorry

end train_crosses_bridge_in_time_l808_808779


namespace units_digit_7_pow_6_pow_5_l808_808421

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l808_808421


namespace distance_from_F₁_to_line_F₂M_l808_808941

-- Definitions
def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 2) = 1
def F₁ : ℝ × ℝ := (0, 1)
def F₂ : ℝ × ℝ := (2, 1)
def M : ℝ × ℝ := (sqrt 2, 1)
def a := 2
def b := sqrt 2
def c := sqrt 2
def MF₁ := 1
def MF₂ := 3
def F₁F₂ := 2 * sqrt 2

-- Proposition to prove
theorem distance_from_F₁_to_line_F₂M : 
  (F₁F₂ * MF₁ / MF₂) = 2 * sqrt 2 / 3 := by 
sorry

end distance_from_F₁_to_line_F₂M_l808_808941


namespace find_magnitude_z_l808_808984

-- conditions
def condition (z : ℂ) : Prop :=
  (1 - complex.i) * z = (1 + complex.i)^2

-- proof statement
theorem find_magnitude_z (z : ℂ) (h : condition z) : complex.abs z = real.sqrt 2 :=
sorry

end find_magnitude_z_l808_808984


namespace root_of_quadratic_expression_l808_808077

theorem root_of_quadratic_expression (n : ℝ) (h : n^2 - 5 * n + 4 = 0) : n^2 - 5 * n = -4 :=
by
  sorry

end root_of_quadratic_expression_l808_808077


namespace repainting_possible_iff_n_multiple_of_3_l808_808066

theorem repainting_possible_iff_n_multiple_of_3 (n : ℕ) :
  (∃ (grid : matrix (fin n) (fin n) bool), 
    (∀ i j, grid i j = (i + j) % 2 = 0) ∧
    (grid 0 0 = false ∨ grid 0 (n-1) = false ∨ grid (n-1) 0 = false ∨ grid (n-1) (n-1) = false) ∧
    (∀ i j, ((i < n - 1) ∧ (j < n - 1)) → 
       ∃ x, x = grid i j ⊕ grid (i+1) j ⊕ grid i (j+1) ⊕ grid (i+1) (j+1))) ↔ 
    (∃ k : ℕ, n = 3 * k) :=
by
  sorry

end repainting_possible_iff_n_multiple_of_3_l808_808066


namespace total_wood_needed_for_tree_house_is_344_l808_808156

theorem total_wood_needed_for_tree_house_is_344 :
  let pillar_length_1 := 4 -- length of the first set of pillars
  let pillar_count_1 := 4 -- count of the first set of pillars
  let pillar_length_2 := 5 * Real.sqrt pillar_length_1 -- length of the second set of pillars
  let pillar_count_2 := 4 -- count of the second set of pillars
  let wall_length_1 := 6 -- length of the first set of walls
  let wall_count_1 := 10 -- count of the first set of walls
  let wall_length_2 := (2 / 3) * (wall_length_1 ^ (3 / 2)) -- length of the second set of walls
  let wall_count_2 := 10 -- count of the second set of walls
  let floor_piece_count := 8 -- number of floor pieces
  let floor_avg_length := 5.5 -- average length of the floor pieces
  let roof_first_piece := 2 * floor_avg_length -- length of the first piece of the roof
  let roof_common_diff := (1 / 3) * pillar_length_1 -- common difference of the arithmetic sequence in the roof pieces
  let roof_piece_count := 6 -- number of roof pieces
  let pillar_total := (pillar_count_1 * pillar_length_1) + (pillar_count_2 * pillar_length_2)
  let wall_total := (wall_count_1 * wall_length_1) + (wall_count_2 * wall_length_2)
  let floor_total := floor_piece_count * floor_avg_length
  let roof_total := Sum.toList (List.range roof_piece_count).map (λ n => roof_first_piece + n * roof_common_diff)
  let total_wood := pillar_total + wall_total + floor_total + roof_total
  in
  total_wood = 344 := sorry

end total_wood_needed_for_tree_house_is_344_l808_808156


namespace perimeter_ACFHK_is_correct_l808_808997

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the points of the pentagon within the dodecagon
def ACFHK_points : ℕ := 5

-- Define the perimeter of the pentagon ACFHK in the dodecagon
noncomputable def perimeter_of_ACFHK : ℝ :=
  let triangle_side := radius
  let isosceles_right_triangle_side := radius * Real.sqrt 2
  3 * triangle_side + 2 * isosceles_right_triangle_side

-- Verify that the calculated perimeter matches the expected value
theorem perimeter_ACFHK_is_correct : perimeter_of_ACFHK = 18 + 12 * Real.sqrt 2 :=
  sorry

end perimeter_ACFHK_is_correct_l808_808997


namespace power_function_monotonically_decreasing_l808_808951

theorem power_function_monotonically_decreasing (m : ℝ) (h1 : ∀ x > 0, derivative (λ x, (m^2 - 3*m + 3) * x^(m^2 - m - 1)) < 0) : m = 1 :=
sorry

end power_function_monotonically_decreasing_l808_808951


namespace actual_car_body_mass_is_1000kg_l808_808729

-- Define a structure for a car model with its mass and scale
structure CarModel where
  mass : ℝ
  scale : ℝ

-- Define the conditions
def model : CarModel := { mass := 1, scale := 1 / 10 }

-- Define the actual car mass according to the given scale
def actual_car_mass (model : CarModel) : ℝ :=
  model.mass * (model.scale⁻¹) ^ 3

-- Theorem stating that the mass of the actual car body is 1000 kg
theorem actual_car_body_mass_is_1000kg : actual_car_mass model = 1000 := 
by
  sorry

end actual_car_body_mass_is_1000kg_l808_808729


namespace factorization_only_D_l808_808707

-- Define the relevant transformations
def transformA : Prop := (x - 4) * (x + 4) = x^2 - 16
def transformB : Prop := x^2 - y^2 + 2 = (x + y) * (x - y) + 2
def transformC : Prop := x^2 + 1 = x * (x + 1 / x)
def transformD : Prop := a^2 * b + a * b^2 = a * b * (a + b)

-- Define that only transformD involves factorization based on the definition
theorem factorization_only_D : 
  (¬ transformA ∧ ¬ transformB ∧ ¬ transformC ∧ transformD) := 
by
  sorry

end factorization_only_D_l808_808707


namespace choose_non_overlapping_sets_for_any_n_l808_808892

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808892


namespace units_digit_of_7_pow_6_pow_5_l808_808431

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808431


namespace prove_trigonometric_identities_l808_808475

variable {α : ℝ}

theorem prove_trigonometric_identities
  (h1 : 0 < α ∧ α < π)
  (h2 : Real.cos α = -3/5) :
  Real.tan α = -4/3 ∧
  (Real.cos (2 * α) - Real.cos (π / 2 + α) = 13/25) := 
by
  sorry

end prove_trigonometric_identities_l808_808475


namespace sports_club_total_members_l808_808134

theorem sports_club_total_members :
  ∀ (B T Both Neither Total : ℕ),
    B = 17 → T = 19 → Both = 10 → Neither = 2 → Total = B + T - Both + Neither → Total = 28 :=
by
  intros B T Both Neither Total hB hT hBoth hNeither hTotal
  rw [hB, hT, hBoth, hNeither] at hTotal
  exact hTotal

end sports_club_total_members_l808_808134


namespace problem_statement_l808_808448

theorem problem_statement
  (a b c d : ℕ)
  (h1 : (b + c + d) / 3 + 2 * a = 54)
  (h2 : (a + c + d) / 3 + 2 * b = 50)
  (h3 : (a + b + d) / 3 + 2 * c = 42)
  (h4 : (a + b + c) / 3 + 2 * d = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 :=
by
  sorry

end problem_statement_l808_808448


namespace isosceles_triangle_side_length_l808_808658

theorem isosceles_triangle_side_length (P : ℕ := 53) (base : ℕ := 11) (x : ℕ)
  (h1 : x + x + base = P) : x = 21 :=
by {
  -- The proof goes here.
  sorry
}

end isosceles_triangle_side_length_l808_808658


namespace basketball_team_free_throws_l808_808233

theorem basketball_team_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a - 1)
  (h3 : 2 * a + 3 * b + x = 89) : 
  x = 29 :=
by
  sorry

end basketball_team_free_throws_l808_808233


namespace maximum_daily_sales_l808_808660

def price (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then t + 20
else if (25 ≤ t ∧ t ≤ 30) then -t + 100
else 0

def sales_volume (t : ℕ) : ℝ :=
if (0 < t ∧ t ≤ 30) then -t + 40
else 0

def daily_sales (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then (t + 20) * (-t + 40)
else if (25 ≤ t ∧ t ≤ 30) then (-t + 100) * (-t + 40)
else 0

theorem maximum_daily_sales : ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_sales t = 1125 :=
sorry

end maximum_daily_sales_l808_808660


namespace geometry_problem_l808_808468

noncomputable def ellipse_through_point (a b : ℝ) (ha : a > b) (hb : b > 0) 
  (eccentricity : ℝ) (h_ecc : eccentricity = (Real.sqrt 2) / 2) 
  (h_eq : a^2 - b^2 = (eccentricity * a)^2)
  (x y : ℝ) (h_xy : (x = Real.sqrt 6 ∧ y = 1)) 
  (h_point : (x^2 / a^2) + (y^2 / b^2) = 1) : Prop :=
(a^2 = 8 ∧ b^2 = 4)

noncomputable def angle_AOB_right_triangle (A B O : Point) (h_AB : line_through A B) 
  (h_tangent : tangent_line A B (circle O (Real.sqrt (8 / 3)))) 
  (h_angle : ∃ l, l ∩ ellipse = {A, B})
  (O_origin : O = (0,0)) : Prop :=
∠AOB = Real.pi / 2

-- Statement without proof
theorem geometry_problem (a b : ℝ) (ha : a > b) (hb : b > 0) 
  (eccentricity : ℝ) (h_ecc : eccentricity = (Real.sqrt 2) / 2) 
  (h_eq : a^2 - b^2 = (eccentricity * a)^2)
  (x y : ℝ) (h_xy : (x = Real.sqrt 6 ∧ y = 1)) 
  (h_point : (x^2 / a^2) + (y^2 / b^2) = 1)
  (A B O : Point) (h_AB : line_through A B) 
  (h_tangent : tangent_line A B (circle O (Real.sqrt (8 / 3)))) 
  (h_angle : ∃ l, l ∩ (ellipse_through_point a b ha hb eccentricity h_ecc h_eq x y h_xy h_point) = {A, B})
  (O_origin : O = (0,0)) : 
  ∠AOB = Real.pi / 2 := 
sorry

end geometry_problem_l808_808468


namespace rectangle_ratio_congruent_squares_l808_808694
-- Import the necessary library

-- Define the conditions and the proof statement
theorem rectangle_ratio_congruent_squares
  (s w h : ℝ)
  -- Inner square area
  (inner_square_area : s^2)
  -- Outer rectangle area is 3 times inner square
  (outer_rectangle_area : 3 * s^2)
  -- Dimensions of the outer rectangle
  (width : s + 2 * w)
  (height : s + h)
  -- Area relation
  (area_relation : (s + 2 * w) * (s + h) = 3 * s^2) :
  (h / w) = 1 :=
sorry

end rectangle_ratio_congruent_squares_l808_808694


namespace minimum_possible_perimeter_l808_808261

theorem minimum_possible_perimeter (a b c : ℤ) (h1 : 2 * a + 8 * c = 2 * b + 10 * c) 
                                  (h2 : 4 * c * (sqrt (a^2 - (4 * c)^2)) = 5 * c * (sqrt (b^2 - (5 * c)^2))) 
                                  (h3 : a - b = c) : 
    2 * a + 8 * c = 740 :=
by
  sorry

end minimum_possible_perimeter_l808_808261


namespace minimum_perimeter_l808_808252

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l808_808252


namespace max_unbounded_xy_sum_l808_808278

theorem max_unbounded_xy_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ M : ℝ, ∀ z : ℝ, z > 0 → ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ (xy + 1)^2 + (x - y)^2 > z := 
  sorry

end max_unbounded_xy_sum_l808_808278


namespace S_eq_Z_l808_808172

noncomputable def set_satisfies_conditions (S : Set ℤ) (a : Fin n → ℤ) :=
  (∀ i : Fin n, a i ∈ S) ∧
  (∀ i j : Fin n, (a i - a j) ∈ S) ∧
  (∀ x y : ℤ, x ∈ S → y ∈ S → x + y ∈ S → x - y ∈ S) ∧
  (Nat.gcd (List.foldr Nat.gcd 0 (Fin.val <$> List.finRange n)) = 1)

theorem S_eq_Z (S : Set ℤ) (a : Fin n → ℤ) (h_cond : set_satisfies_conditions S a) : S = Set.univ :=
  sorry

end S_eq_Z_l808_808172


namespace swimming_pool_volume_is_270_l808_808713

def volume_of_swimming_pool (width length shallow_depth deep_depth : ℝ) : ℝ :=
  (1 / 2) * (shallow_depth + deep_depth) * width * length

theorem swimming_pool_volume_is_270 :
  volume_of_swimming_pool 9 12 1 4 = 270 :=
by
  sorry

end swimming_pool_volume_is_270_l808_808713


namespace modulus_of_z_l808_808085

def z : ℂ := (3 - complex.I) / (1 + complex.I)

theorem modulus_of_z : complex.abs z = real.sqrt 5 := by
  sorry

end modulus_of_z_l808_808085


namespace stationery_store_sales_l808_808547

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l808_808547


namespace unit_vectors_eq_sq_norm_l808_808452

variables {E : Type*} [inner_product_space ℝ E]

-- Define unit vectors e1 and e2
variables (e1 e2 : E)
hypothesis h1 : ∥e1∥ = 1
hypothesis h2 : ∥e2∥ = 1

-- Prove that the squared magnitudes of the two vectors are equal
theorem unit_vectors_eq_sq_norm : ∥e1∥^2 = ∥e2∥^2 :=
by
  sorry

end unit_vectors_eq_sq_norm_l808_808452


namespace find_a_l808_808820

-- Definition of * in terms of 2a - b^2
def custom_mul (a b : ℤ) := 2 * a - b^2

-- The proof statement
theorem find_a (a : ℤ) : custom_mul a 3 = 3 → a = 6 :=
by
  sorry

end find_a_l808_808820


namespace alex_ascending_staircase_l808_808714

   noncomputable def a : ℕ → ℕ
   | 1     := 1
   | 2     := 2
   | (n+3) := a (n+2) + a (n+1)

   theorem alex_ascending_staircase : a 10 = 89 :=
   sorry
   
end alex_ascending_staircase_l808_808714


namespace combined_bus_capacity_l808_808605

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l808_808605


namespace impossible_domino_covering_l808_808150

def Chessboard (n : Nat) := Fin n × Fin n
def Domino := Fin 2

def covers (dom : Domino) (cell : Chessboard 8) : Prop := 
  -- Define the covering relationship of domino on the chessboard cell here
  sorry

def forms_2x2_square (dom1 dom2 : Domino) : Prop := 
  -- Define the condition where two dominoes form a 2x2 square
  sorry

theorem impossible_domino_covering : 
  ¬∃ (f : Chessboard 8 → option Domino), 
    (∀ cell, ∃ d, f cell = some d) ∧ 
    (∀ cell1 cell2, f cell1 = f cell2 → covers (f cell1) cell1 ∧ covers (f cell2) cell2) ∧ 
    (∀ dom1 dom2, forms_2x2_square dom1 dom2 → dom1 ≠ dom2) :=
sorry

end impossible_domino_covering_l808_808150


namespace kelly_total_apples_l808_808158

variable (initial_apples : ℕ) (additional_apples : ℕ)

theorem kelly_total_apples (h1 : initial_apples = 56) (h2 : additional_apples = 49) :
  initial_apples + additional_apples = 105 :=
by
  sorry

end kelly_total_apples_l808_808158


namespace no_good_polygon_in_division_of_equilateral_l808_808593

def is_equilateral_polygon (P : List Point) : Prop :=
  -- Definition of equilateral polygon
  sorry

def is_good_polygon (P : List Point) : Prop :=
  -- Definition of good polygon (having a pair of parallel sides)
  sorry

def is_divided_by_non_intersecting_diagonals (P : List Point) (polygons : List (List Point)) : Prop :=
  -- Definition for dividing by non-intersecting diagonals into several polygons
  sorry

def have_same_odd_sides (polygons : List (List Point)) : Prop :=
  -- Definition for all polygons having the same odd number of sides
  sorry

theorem no_good_polygon_in_division_of_equilateral (P : List Point) (polygons : List (List Point)) :
  is_equilateral_polygon P →
  is_divided_by_non_intersecting_diagonals P polygons →
  have_same_odd_sides polygons →
  ¬ ∃ gp ∈ polygons, is_good_polygon gp :=
by
  intro h_eq h_div h_odd
  intro h_good
  -- Proof goes here
  sorry

end no_good_polygon_in_division_of_equilateral_l808_808593


namespace find_k_eq_13_l808_808224

theorem find_k_eq_13 (k: ℝ) :
  (∀ x : ℂ, (x + 2 - 3 * complex.i) * (x + 2 + 3 * complex.i) = 0 -> 
  k = (x + 2) ^ 2 + (3 * complex.i) ^ 2) -> 
  k = 13 :=
by
  sorry

end find_k_eq_13_l808_808224


namespace evaluate_expression_l808_808830

theorem evaluate_expression : 
  (16 = 2^4) → 
  (32 = 2^5) → 
  (16^24 / 32^12 = 8^12) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end evaluate_expression_l808_808830


namespace banana_arrangements_l808_808105

theorem banana_arrangements: 
  let total_letters := 6 in
  let b_count := 1 in
  let n_count := 2 in
  let a_count := 3 in
  (total_letters.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)) = 60 := 
by 
  sorry

end banana_arrangements_l808_808105


namespace function_values_l808_808493

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem function_values (a b : ℝ) (h1 : f 1 a b = 2) (h2 : a = 2) : f 2 a b = 4 := by
  sorry

end function_values_l808_808493


namespace range_of_m_l808_808100

noncomputable def setA : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 4 }
noncomputable def setB (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

theorem range_of_m (m : ℝ) : (setB m ⊆ setA) ↔ (m ∈ Iic (5 / 2)) := by
  sorry

end range_of_m_l808_808100


namespace minimum_perimeter_l808_808253

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l808_808253


namespace problem_l808_808450

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def compN : Set ℝ := {x | x < -1 ∨ 1 < x}
def intersection : Set ℝ := {x | x < -1 ∨ (1 < x ∧ x ≤ 2)}

theorem problem (x : ℝ) : x ∈ (M ∩ compN) ↔ x ∈ intersection := by
  sorry

end problem_l808_808450


namespace probability_score_l808_808541

/-- Given:
1. A bag with 4 red balls and 3 black balls.
2. 4 balls drawn from the bag.
3. Drawing 1 red ball scores 1 point.
4. Drawing 1 black ball scores 3 points.
5. Score is a random variable ξ.

Prove that the probability P(ξ ≤ 7) equals 13/35.
-/
theorem probability_score (R B : ℕ) (drawn : ℕ) (score_red score_black : ℕ) (ξ : ℕ → ℕ) :
  R = 4 → B = 3 → drawn = 4 → score_red = 1 → score_black = 3 →
  (∀ n, ξ n = if n = 0 then 4 else if n = 1 then 6 else if n = 2 then 8 else if n = 3 then 10 else 0) →
  ∑ i in finset.range (ξ 2 + 1), if ξ i ≤ 7 then 1 else 0 / (nat.choose (R + B) drawn) = 13 / 35 :=
by
  intros hR hB hDrawn hscore_red hscore_black hξ sorry

end probability_score_l808_808541


namespace unique_solution_condition_l808_808088

theorem unique_solution_condition (m : ℝ) : ((m^2 + 2 * m + 3) * x = 3 * (x + 2) + m - 4) → 
  (∀ x, m ≠ 0 ∧ m ≠ -2) := 
by 
  sorry

end unique_solution_condition_l808_808088


namespace chandler_saves_for_laptop_l808_808000

theorem chandler_saves_for_laptop :
  ∃ x : ℕ, 140 + 20 * x = 800 ↔ x = 33 :=
by
  use 33
  sorry

end chandler_saves_for_laptop_l808_808000


namespace impossible_measurement_l808_808307

noncomputable def measure_all_distances (marks : Set ℝ) (totalLength : ℝ) : Prop :=
  (∀ d ∈ Icc 1 totalLength, ∃ mark1 mark2 ∈ marks, d = abs (mark1 - mark2))

theorem impossible_measurement :
  ¬ ∃ (marks : Set ℝ) (segments : List ℝ),
    marks.card = 4 ∧
    totalLength = 15 ∧
    (∀ seg ∈ segments, seg ∈ {1, 2, 3, 4, 5}) ∧
    segments.sum = 15 ∧
    measure_all_distances marks 15 := sorry

end impossible_measurement_l808_808307


namespace units_digit_7_power_l808_808397

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808397


namespace non_overlapping_original_sets_exists_l808_808870

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808870


namespace units_digit_7_power_l808_808392

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808392


namespace units_digit_of_power_l808_808400

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808400


namespace smallest_four_digit_multiple_of_37_l808_808701

theorem smallest_four_digit_multiple_of_37 : ∃ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 ∧ 37 ∣ n ∧ (∀ m : ℕ, m ≥ 1000 ∧ m ≤ 9999 ∧ 37 ∣ m → n ≤ m) ∧ n = 1036 :=
by
  sorry

end smallest_four_digit_multiple_of_37_l808_808701


namespace modulus_conjugate_z_l808_808487

-- Define the complex number z
def z : ℂ := 1 + complex.I

-- Define the conjugate of z
def conjugate_z : ℂ := complex.conj z

-- Prove that the modulus of the conjugate of z is √2
theorem modulus_conjugate_z : complex.abs conjugate_z = real.sqrt 2 :=
by
  sorry

end modulus_conjugate_z_l808_808487


namespace minimum_common_perimeter_exists_l808_808256

noncomputable def find_minimum_perimeter
  (a b x : ℕ) 
  (is_int_sided_triangle_1 : 2 * a + 20 * x = 2 * b + 25 * x)
  (is_int_sided_triangle_2 : 20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2))) 
  (base_ratio : 20 * 2 * (a - b) = 25 * 2 * (a - b)): ℕ :=
2 * a + 20 * (2 * (a - b))

-- The final goal should prove the minimum perimeter under the given conditions.
theorem minimum_common_perimeter_exists :
∃ (minimum_perimeter : ℕ), 
  (∀ (a b x : ℕ), 
    2 * a + 20 * x = 2 * b + 25 * x → 
    20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2)) → 
    20 * 2 * (a - b) = 25 * 2 * (a - b) → 
    minimum_perimeter = 2 * a + 20 * x) :=
sorry

end minimum_common_perimeter_exists_l808_808256


namespace correct_pythagorean_triple_l808_808796

def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem correct_pythagorean_triple :
  (is_pythagorean_triple 1 2 3 = false) ∧ 
  (is_pythagorean_triple 4 5 6 = false) ∧ 
  (is_pythagorean_triple 6 8 9 = false) ∧ 
  (is_pythagorean_triple 7 24 25 = true) :=
by
  sorry

end correct_pythagorean_triple_l808_808796


namespace square_window_side_length_l808_808827

def pane_ratio := { height : ℕ, width : ℕ } → Prop
def pane_ratio := λ (pane : { height : ℕ, width : ℕ }), pane.height * 2 = pane.width * 5

def total_width := 8 * 2 + 10
def total_height := 10 * 2 + 6

theorem square_window_side_length :
  (∀ pane : { height : ℕ, width : ℕ }, pane_ratio pane) → 
  total_width = total_height → 
  total_width = 26 := 
by {
  intros _ _,
  sorry
}

end square_window_side_length_l808_808827


namespace sin_alpha_eq_sqrt_five_five_l808_808925

variable (α : Real)
axiom tan_eq_neg_half : tan α = -1/2
axiom alpha_in_second_quad : π / 2 < α ∧ α < π

theorem sin_alpha_eq_sqrt_five_five
  (tan_eq_neg_half : tan α = -1/2)
  (alpha_in_second_quad : π / 2 < α ∧ α < π) :
  sin α = sqrt 5 / 5 := 
  by sorry

end sin_alpha_eq_sqrt_five_five_l808_808925


namespace non_overlapping_original_sets_exists_l808_808868

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808868


namespace factory_workers_total_payroll_l808_808556

theorem factory_workers_total_payroll (total_office_payroll : ℝ) (number_factory_workers : ℝ) 
(number_office_workers : ℝ) (salary_difference : ℝ) 
(average_office_salary : ℝ) (average_factory_salary : ℝ) 
(h1 : total_office_payroll = 75000) (h2 : number_factory_workers = 15)
(h3 : number_office_workers = 30) (h4 : salary_difference = 500)
(h5 : average_office_salary = total_office_payroll / number_office_workers)
(h6 : average_office_salary = average_factory_salary + salary_difference) :
  number_factory_workers * average_factory_salary = 30000 :=
by
  sorry

end factory_workers_total_payroll_l808_808556


namespace non_overlapping_original_sets_l808_808883

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808883


namespace trig_expression_equiv_l808_808831

noncomputable def cofunction_identity (θ : ℝ) : ℝ :=
  cos θ = sin (real.pi / 2 - θ)

noncomputable def tan_def (θ : ℝ) : ℝ :=
  tan θ = sin θ / cos θ

noncomputable def double_angle_identity (θ : ℝ) : ℝ :=
  sin (2 * θ) = 2 * sin θ * cos θ

noncomputable def sum_of_angles_identity (α β : ℝ) : ℝ :=
  sin (α + β) = sin α * cos β + cos α * sin β

noncomputable def exact_value_sin_30 : ℝ :=
  sin (real.pi / 6) = 1 / 2

noncomputable def exact_value_cos_30 : ℝ :=
  cos (real.pi / 6) = sqrt 3 / 2

theorem trig_expression_equiv : 4 * cos (50 * real.pi / 180) - tan (40 * real.pi / 180) = sqrt 3 :=
by
  sorry

end trig_expression_equiv_l808_808831


namespace problem1_problem2_problem3_l808_808069

-- Given an equilateral triangle ABC with side length 2
variables {A B C P : Point}
variables {PC PA PB : Vector}

-- Problem statement
-- Prove that given conditions, the required values hold
theorem problem1 
  (h1 : PC = -(PA + PB)) : 
  area A B P = Real.sqrt 3 / 3 := 
sorry

theorem problem2 
  (h2 : PB ∙ PC = 0) : 
  max_val (norm PB + norm PC) = 2 * Real.sqrt 2 := 
sorry

theorem problem3 : 
  min_val (2 * (PA ∙ PB) + PA ∙ PC) = -7 / 3 := 
sorry

end problem1_problem2_problem3_l808_808069


namespace original_sets_exist_l808_808898

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808898


namespace volume_of_circumscribed_sphere_of_triangular_prism_l808_808566

def triangleAreas (AB AC AD : ℝ) : Prop :=
  (1 / 2 * AB * AC = sqrt 2 / 2) ∧
  (1 / 2 * AC * AD = sqrt 3 / 2) ∧
  (1 / 2 * AD * AB = sqrt 6 / 2)

def circumradius (a b c : ℝ) : ℝ :=
  (1 / 2) * sqrt (a^2 + b^2 + c^2)

def volumeCircumscribedSphere (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

theorem volume_of_circumscribed_sphere_of_triangular_prism
  (a b c : ℝ)
  (h : triangleAreas a b c)
  :
  volumeCircumscribedSphere (circumradius a b c) = Real.pi * sqrt 6
  :=
  sorry

end volume_of_circumscribed_sphere_of_triangular_prism_l808_808566


namespace original_sets_exist_l808_808904

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808904


namespace net_change_in_price_net_change_percentage_l808_808990

theorem net_change_in_price (P : ℝ) :
  0.80 * P * 1.55 - P = 0.24 * P :=
by sorry

theorem net_change_percentage (P : ℝ) :
  ((0.80 * P * 1.55 - P) / P) * 100 = 24 :=
by sorry


end net_change_in_price_net_change_percentage_l808_808990


namespace original_sets_exist_l808_808879

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808879


namespace largest_x_satisfying_sqrt_eq_l808_808277

theorem largest_x_satisfying_sqrt_eq (x : ℝ) (hx : x ≥ 0) : 
  (sqrt (3 * x) = 6 * x^2) → x = 1 / (Real.cbrt 12) := 
by 
  sorry

end largest_x_satisfying_sqrt_eq_l808_808277


namespace non_overlapping_original_sets_l808_808888

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808888


namespace Xiao_Ming_mother_age_l808_808294

-- Conditions
def id_number := 6101131197410232923
def birth_year := 1974  -- Extracted from the ID number
def reference_year := 2014
def end_of_october_age := reference_year - birth_year

-- Goal
theorem Xiao_Ming_mother_age : end_of_october_age = 40 := 
by {
  -- Add the step for explanation purposes
  unfold end_of_october_age,
  simp,
  sorry
}

end Xiao_Ming_mother_age_l808_808294


namespace count_valid_subsets_l808_808464

open Set

def isEven (n : ℕ) : Prop := n % 2 = 0

/-- A subset M of {4, 7, 8} contains at most one even number -/
def at_most_one_even (M : Set ℕ) : Prop :=
  (4 ∈ M ∧ 8 ∈ M) = false

/-- The set of all subsets of {4, 7, 8} containing at most one even number -/
def valid_subsets : Set (Set ℕ) :=
  {M | M ⊆ {4, 7, 8} ∧ at_most_one_even M}

theorem count_valid_subsets : ∃ n, n = 6 ∧ n = cardinal.mk valid_subsets :=
by
  sorry

end count_valid_subsets_l808_808464


namespace train_cross_bridge_time_l808_808717

theorem train_cross_bridge_time (length_train length_bridge : ℕ) (speed_train_kmh : ℕ) : 
  (length_train = 140 ∧ length_bridge = 132 ∧ speed_train_kmh = 72) →
  (let speed_train_ms = speed_train_kmh * 1000 / 3600 in
   let total_distance = length_train + length_bridge in
   (total_distance : ℚ) / speed_train_ms = 13.6) := 
by
  intros h
  let ⟨h1, h2, h3⟩ := h
  rw [h1, h2, h3]
  let speed_train_ms := 72 * 1000 / 3600
  let total_distance := 140 + 132
  have speed_ms_calc : speed_train_ms = 20 := by norm_num
  have distance_calc : total_distance = 272 := by norm_num
  rw [speed_ms_calc, distance_calc]
  norm_num
  sorry

end train_cross_bridge_time_l808_808717


namespace nums_between_2000_and_3000_div_by_360_l808_808520

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l808_808520


namespace smallest_distance_is_at_least_sqrt113_minus_4_l808_808586

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  Complex.abs z - w

theorem smallest_distance_is_at_least_sqrt113_minus_4
  (z w : ℂ)
  (hz : Complex.abs (z + 2 + 2 * Complex.I) = 2)
  (hw : Complex.abs (w - (5 + 6 * Complex.I)) = 2) :
  smallest_distance z w ≥ Real.sqrt 113 - 4 :=
sorry

end smallest_distance_is_at_least_sqrt113_minus_4_l808_808586


namespace find_a_l808_808987

theorem find_a (a : ℝ) (h1 : a > 1) (h2 : ∀ x ∈ set.Icc 1 a, log 2 x ≤ 2) : a = 4 :=
sorry

end find_a_l808_808987


namespace sin_double_angle_solution_l808_808849

theorem sin_double_angle_solution (α : ℝ) 
  (h1 : 2 * cos (2 * α) = sin (π / 4 - α)) 
  (h2 : α > π / 2 ∧ α < π) : 
  sin (2 * α) = -7 / 8 :=
sorry

end sin_double_angle_solution_l808_808849


namespace num_valid_paths_l808_808828

-- Define the properties of the tetrahedron path problem
def vertex : Type := ℕ
def edge := vertex × vertex
def tetrahedron := {v1 : vertex // v1 < 4} -- Four vertices labeled 0 to 3

noncomputable def edges (v : vertex) : set edge := {
  (0, 1), (0, 2), (0, 3), 
  (1, 2), (1, 3), 
  (2, 3)
}

-- Function to determine if a set of edges forms a valid path
def is_valid_path (start_vertex end_vertex: vertex) (edges_visited : list edge) : Prop :=
  edges_visited.length = 6 ∧
  ∀ (e : edge), e ∈ edges_visited → (e.fst = start_vertex ∨ e.snd = start_vertex ∨ e.fst = end_vertex ∨ e.snd = end_vertex) ∧
  edges_visited.head = (start_vertex, 1) ∧ edges_visited.last = (3, end_vertex)

-- Example of conditions for vertices
def start_vertex : vertex := 0
def end_vertex : vertex := 3

-- Main theorem: Number of paths from start to end with given conditions
theorem num_valid_paths : 
∃ paths_count: ℕ, (paths_count = 2) :=
begin
  sorry
end

end num_valid_paths_l808_808828


namespace strong_word_count_l808_808014

/-- A strong word is a sequence of letters that consists only of the letters A, B, and C,
    where A is never immediately followed by B or C,
    B is never immediately followed by A or C,
    and C is never immediately followed by A or B. -/
def is_strong_word (s : String) : Prop :=
  ∀ i : ℕ, i < s.length - 1 →
    (s.get ⟨i, sorry⟩ = 'A' → s.get ⟨i + 1, sorry⟩ = 'A') ∧
    (s.get ⟨i, sorry⟩ = 'B' → s.get ⟨i + 1, sorry⟩ = 'B') ∧
    (s.get ⟨i, sorry⟩ = 'C' → s.get ⟨i + 1, sorry⟩ = 'C')

/-- The number of eight-letter strong words. -/
def num_eight_letter_strong_words : ℕ :=
  3

theorem strong_word_count :
  ∃ n : ℕ, n = 3 ∧ (∀ s : String, s.length = 8 → is_strong_word s → n = num_eight_letter_strong_words) :=
by {
  existsi 3,
  split,
  { refl, },
  { intros s h_len h_strong,
    refl, }
}

end strong_word_count_l808_808014


namespace smallest_EF_minus_DE_l808_808690

theorem smallest_EF_minus_DE (x y z : ℕ) (h1 : x < y) (h2 : y ≤ z) (h3 : x + y + z = 2050)
  (h4 : x + y > z) (h5 : y + z > x) (h6 : z + x > y) : y - x = 1 :=
by
  sorry

end smallest_EF_minus_DE_l808_808690


namespace eval_expression_l808_808977

def star (A B : ℝ) : ℝ := (A + B) / (A - B)

theorem eval_expression : star (star (star 10 5) 2) 3 = 4 := by
  sorry

end eval_expression_l808_808977


namespace base7_divisible_by_19_l808_808022

theorem base7_divisible_by_19 (y : ℕ) (h : y ≤ 6) :
  (7 * y + 247) % 19 = 0 ↔ y = 0 :=
by sorry

end base7_divisible_by_19_l808_808022


namespace man_rate_still_water_l808_808763

def speed_with_stream : ℝ := 6
def speed_against_stream : ℝ := 2

theorem man_rate_still_water : (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

end man_rate_still_water_l808_808763


namespace pencil_count_l808_808186

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l808_808186


namespace find_angle_of_inclination_l808_808017

-- Define the coefficients of the line equation
def a := 2
def b := 3
def c := 1

-- Define the equation of the line
def line_eq (x y: ℝ) : Prop := a * x + b * y - c = 0

-- Define the angle of inclination given the slope of the line
noncomputable def angle_of_inclination := Real.pi - Real.arctan (a / b)

-- The theorem stating the angle of inclination is as expected
theorem find_angle_of_inclination (x y: ℝ) (h: line_eq x y) :
    ∃ α : ℝ, α = angle_of_inclination :=
begin
  use angle_of_inclination,
  refl,
end

end find_angle_of_inclination_l808_808017


namespace geometric_progression_and_minimum_u_l808_808567

theorem geometric_progression_and_minimum_u
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 2 * real.cos (A - C) + real.cos (2 * B) = 1 + 2 * real.cos A * real.cos C)
  (h2 : a = 2 * R * real.sin A)
  (h3 : b = 2 * R * real.sin B)
  (h4 : c = 2 * R * real.sin C) :
  (b ≠ 0 ∧ ac = b^2 ∧ b = 2 ∧ ∃ (u : ℝ), u = abs ((a^2 + c^2 - 5) / (a - c)) ∧ ∀ (u : ℝ), u ≥ 2 * real.sqrt 3 ∧ ∃ (cosB : ℝ), cosB = (7 / 8)) :=
by
  sorry

end geometric_progression_and_minimum_u_l808_808567


namespace sufficiency_condition_for_perpendicular_vectors_l808_808102

theorem sufficiency_condition_for_perpendicular_vectors (n : ℝ) :
  let M : ℝ × ℝ := (2, 3)
      N : ℝ × ℝ := (4, n)
      E : ℝ × ℝ := (2, -2)
      NE : ℝ × ℝ := (E.1 - N.1, E.2 - N.2)
      MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
  in NE.1 * MN.1 + NE.2 * MN.2 = 0 → n = 2 :=
by
  sorry

end sufficiency_condition_for_perpendicular_vectors_l808_808102


namespace trey_bracelets_per_day_l808_808251

theorem trey_bracelets_per_day:
  ∀ (amount_needed days : ℕ), 
  amount_needed = 112 →
  days = 2 * 7 →
  (amount_needed / days) = 8 := 
by
  intros amount_needed days h1 h2
  have h3: days = 14 := by {sorry} -- simplifying 2 * 7 to 14
  have h4: amount_needed = 112 := by {sorry}
  have h5: amount_needed / days = 112 / 14 := by {sorry}
  have h6: 112 / 14 = 8 := by {sorry}
  exact h6

end trey_bracelets_per_day_l808_808251


namespace minimum_possible_perimeter_l808_808260

theorem minimum_possible_perimeter (a b c : ℤ) (h1 : 2 * a + 8 * c = 2 * b + 10 * c) 
                                  (h2 : 4 * c * (sqrt (a^2 - (4 * c)^2)) = 5 * c * (sqrt (b^2 - (5 * c)^2))) 
                                  (h3 : a - b = c) : 
    2 * a + 8 * c = 740 :=
by
  sorry

end minimum_possible_perimeter_l808_808260


namespace min_value_of_recips_l808_808648

-- Definitions for conditions
def log_func (a : ℝ) (x : ℝ) : ℝ := log x / log a - 1

-- The main theorem, formulated as a Lean statement
theorem min_value_of_recips (a m n : ℝ)
  (h1 : a ≠ 1)
  (h2 : a > 0)
  (h3 : log_func a 1 = -1)
  (h4 : m * (-2) + n * (-1) + 1 = 0)
  (h5 : m > 0)
  (h6 : n > 0)
  : (1 / m + 2 / n) = 8 := 
sorry

end min_value_of_recips_l808_808648


namespace kibble_consumption_rate_l808_808575

-- Kira fills her cat's bowl with 3 pounds of kibble before going to work.
def initial_kibble : ℚ := 3

-- There is still 1 pound left when she returns.
def remaining_kibble : ℚ := 1

-- Kira was away from home for 8 hours.
def time_away : ℚ := 8

-- Calculate the amount of kibble eaten
def kibble_eaten : ℚ := initial_kibble - remaining_kibble

-- Calculate the rate of consumption (hours per pound)
def rate_of_consumption (time: ℚ) (kibble: ℚ) : ℚ := time / kibble

-- Theorem statement: It takes 4 hours for Kira's cat to eat a pound of kibble.
theorem kibble_consumption_rate : rate_of_consumption time_away kibble_eaten = 4 := by
  sorry

end kibble_consumption_rate_l808_808575


namespace choose_non_overlapping_sets_for_any_n_l808_808890

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808890


namespace quadratic_range_l808_808661

noncomputable def f : ℝ → ℝ := sorry -- Quadratic function with a positive coefficient for its quadratic term

axiom symmetry_condition : ∀ x : ℝ, f x = f (4 - x)

theorem quadratic_range (x : ℝ) (h1 : f (1 - 2 * x ^ 2) < f (1 + 2 * x - x ^ 2)) : -2 < x ∧ x < 0 :=
by sorry

end quadratic_range_l808_808661


namespace units_digit_7_pow_6_pow_5_l808_808406

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808406


namespace find_t_l808_808087

-- Define the curves C1 and C2 as given in the problem
def C1 (t : ℝ) (x y : ℝ) : Prop := y^2 = t * x ∧ y > 0 ∧ t > 0
def C2 (x y : ℝ) : Prop := y = Real.exp (x + 1) - 1

-- Define the point M on curve C1
def M (t : ℝ) : ℝ × ℝ := (4 / t, 2)

-- Define the condition stating that the tangent line at M for C1 is also tangent to C2
def tangent_condition (t : ℝ) : Prop :=
  let slope1 := sqrt t * (1 / (2 * sqrt (4/t))) in
  let tangent_line_y := slope1 * (4 / t) + 1 - slope1 * (4 / t) in
  let slope2 m := Real.exp (m + 1) in
  ∃ m, slope1 = slope2 m

-- Prove that t = 4e^2 under the given conditions
theorem find_t : ∃ t > 0, tangent_condition t ∧ C1 t (M t).1 (M t).2 ∧ t = 4 * Real.exp 2 :=
by
  -- Skipping the proof here
  sorry

end find_t_l808_808087


namespace partition_1989_l808_808623

open Finset

noncomputable def partition_sets : Finset (Finset ℕ) :=
  if H : (1989005 % 117 = 0) then sorry else ∅

theorem partition_1989 (A : Finset (Finset ℕ)) (hA : partition_sets = A):
  (∃ Ai : Finset (Finset ℕ), (partition_sets = Ai) ∧ 
  (\bigcup i in (range 117), (Ai i)).card = 1989 ∧
  ∀ i ∈ range 117, (Ai i).card = 17 ∧ 
  ∀ i ∈ range 117, (Ai i).sum id = 17007) :=
sorry

end partition_1989_l808_808623


namespace problem_statement_l808_808336

noncomputable def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

noncomputable def hasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f (x)

theorem problem_statement : 
  isOdd (λ x => sin x * cos x) ∧ hasPeriod (λ x => sin x * cos x) π := 
by
  sorry

end problem_statement_l808_808336


namespace weight_of_fresh_grapes_l808_808044

theorem weight_of_fresh_grapes 
  (F : ℝ) (D : ℝ)
  (hfresh_water : 0.9 * F) 
  (hdried_water : 0.2 * D)
  (hw_dry_grapes : D = 1.25)
  (H : 0.10 * F = 0.80 * D) : 
  F = 10 := 
by {
  sorry
}

end weight_of_fresh_grapes_l808_808044


namespace combined_bus_capacity_l808_808606

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l808_808606


namespace original_sets_exist_l808_808880

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808880


namespace quadratic_inequality_solution_l808_808043

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 36 * x + 318 ≤ 0 ↔ 18 - Real.sqrt 6 ≤ x ∧ x ≤ 18 + Real.sqrt 6 := by
  sorry

end quadratic_inequality_solution_l808_808043


namespace projection_eq_l808_808858

variables (a b : EuclideanSpace ℝ (fin 3))
variables (norm_a : ‖a‖ = 3)
variables (norm_b : ‖b‖ = 5)
variables (dot_ab : ⟪a, b⟫ = 12)

theorem projection_eq : (‖a‖ * (⟪a, b⟫ / ‖b‖) = (12 / 5)) :=
by {
  sorry
}

end projection_eq_l808_808858


namespace non_overlapping_original_sets_exists_l808_808874

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808874


namespace find_A_maximum_area_triangle_l808_808996

variables {a b c : ℝ}
variables {A B C : ℝ} (sin cos : ℝ → ℝ)
variables {m n : ℝ × ℝ}

-- Given conditions
def parallel_vectors (m n : ℝ × ℝ) : Prop := m.1 * n.2 = m.2 * n.1
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0
def law_of_cosines (a b c : ℝ) (A : ℝ) : Prop := a^2 = b^2 + c^2 - 2 * b * c * cos A

-- Problem 1: Find angle A
theorem find_A (h1 : m = (a, b + c))
                (h2 : n = (1, cos C + sqrt 3 * sin C))
                (h3 : parallel_vectors m n) :
                A = π / 3 :=
sorry

-- Problem 2: Find the maximum area
def area_triangle (b c A : ℝ) : ℝ := 1/2 * b * c * sin A

theorem maximum_area_triangle (h1 : 3 * b * c = 16 - a^2)
                              (h2 : b + c = 4)
                              (h3 : A = π / 3) :
                              area_triangle b c A ≤ sqrt 3 :=
sorry

end find_A_maximum_area_triangle_l808_808996


namespace simplify_and_evaluate_expression_l808_808203

-- Define the parameters for m and n.
def m : ℚ := -1 / 3
def n : ℚ := 1 / 2

-- Define the expression to simplify and evaluate.
def complex_expr (m n : ℚ) : ℚ :=
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2)

-- State the theorem that proves the expression equals -5/3.
theorem simplify_and_evaluate_expression :
  complex_expr m n = -5 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l808_808203


namespace min_value_of_y_l808_808855

noncomputable def m : ℝ := (Real.tan (π / 8)) / (1 - (Real.tan (π / 8))^2)

def y (x : ℝ) : ℝ := 2 * m * x + 3 / (x - 1) + 1

theorem min_value_of_y : ∀ x : ℝ, x > 1 → y x ≥ 2 + 2 * Real.sqrt 3 := sorry

end min_value_of_y_l808_808855


namespace find_length_DE_l808_808641

noncomputable def triangle_ABC_base : ℝ := 15
def area_ratio_below_base : ℝ := 0.25
def expected_DE_length : ℝ := 11.25

theorem find_length_DE
  (AB : ℝ)
  (h1 : AB = triangle_ABC_base)
  (h2 : ∃ (DE : ℝ), DE ∥ AB) 
  (area_ratio : ℝ)
  (h3 : area_ratio = area_ratio_below_base) :
  ∃ DE : ℝ, DE = expected_DE_length :=
begin
  use expected_DE_length,
  sorry
end

end find_length_DE_l808_808641


namespace choose_non_overlapping_sets_for_any_n_l808_808889

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808889


namespace division_result_l808_808390

noncomputable def p : Polynomial ℝ := 
  Polynomial.C 12 + Polynomial.X * (Polynomial.C (-18) + Polynomial.X * (Polynomial.C 15 + Polynomial.X * (Polynomial.C 5 + Polynomial.X * (Polynomial.C (-24) + Polynomial.X * (Polynomial.C 2 + Polynomial.X)))))

noncomputable def d : Polynomial ℝ := Polynomial.X - Polynomial.C 3

theorem division_result :
  Polynomial.divMod p d = (Polynomial.C (-171) + Polynomial.X * (Polynomial.C (-51) + Polynomial.X * (Polynomial.C (-22) + Polynomial.X * (Polynomial.C (-9) + Polynomial.X * (Polynomial.C 5 + Polynomial.X)))), Polynomial.C (-501)) :=
  by
    sorry

end division_result_l808_808390


namespace deductive_reasoning_example_l808_808826

theorem deductive_reasoning_example :
  (∀ (a b : ℝ), (a + b = π) ↔ (a and b are the adjacent interior angles of two parallel lines)) :=
by {
  sorry
}

end deductive_reasoning_example_l808_808826


namespace farmer_total_acres_l808_808751

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l808_808751


namespace days_needed_to_produce_cans_l808_808119

variable (y : ℕ)

-- We assume the given conditions in the form of variables and functions.
def daily_production_per_cow := (y + 2) / (y * (y + 3))

def total_daily_production_per_cows (n : ℕ) := (n * daily_production_per_cow y)

-- Theorem statement
theorem days_needed_to_produce_cans : 
  (y + 4) * (\frac{y(y+3)}{y(y+3)}) := y(y+3)(y+7)}{(y+2)(y+4)}
  : (y * (y + 3) * (y + 7)) / ((y + 2) * (y + 4)) := (y + 4) * (/defns : (y + 7)posed to produce .)
  FCocan , Melissa(. to write '>\def uns to interpet the to to prove that Lean 4 . .han       in 

Sorry

 )


end days_needed_to_produce_cans_l808_808119


namespace simplify_expression_l808_808204

theorem simplify_expression : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end simplify_expression_l808_808204


namespace true_discount_computation_l808_808217

theorem true_discount_computation (BD PV : ℝ) (hBD : BD = 80) (hPV : PV = 560) : 
  let TD := BD / (1 + BD / PV) in
  TD = 70 := 
by
  sorry

end true_discount_computation_l808_808217


namespace probability_exactly_one_male_one_female_same_topic_l808_808245

theorem probability_exactly_one_male_one_female_same_topic :
  let numOutcomes := 8
  let desiredOutcomes := 4
  let probability := (desiredOutcomes : ℝ) / numOutcomes
  probability = 1 / 2 := by
  sorry

end probability_exactly_one_male_one_female_same_topic_l808_808245


namespace geom_sequence_sn_zero_condition_l808_808724

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def S (a : ℕ → ℝ) : ℕ → ℝ
| 0     := a 0
| (n+1) := S a n + a (n + 1)

theorem geom_sequence_sn_zero_condition (a : ℕ → ℝ) (h : is_geometric a) :
  (∀ n : ℕ, S a n ≠ 0) ∨ (∃ infinite n : ℕ, S a n = 0) :=
sorry

end geom_sequence_sn_zero_condition_l808_808724


namespace distinct_positive_differences_l808_808964

theorem distinct_positive_differences : 
  ∃ (S : Set ℕ), S = {n | ∃ (a b : ℕ), a ∈ {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} ∧ b ∈ {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} ∧ n = |a - b|} ∧ S.card = 17 :=
by
  sorry

end distinct_positive_differences_l808_808964


namespace total_words_in_poem_l808_808628

theorem total_words_in_poem (stanzas lines words : ℕ) 
  (h1 : stanzas = 20) 
  (h2 : lines = 10) 
  (h3 : words = 8) :
  stanzas * lines * words = 1600 :=
by
  rw [h1, h2, h3]
  norm_num

end total_words_in_poem_l808_808628


namespace horner_method_v3_value_l808_808810

theorem horner_method_v3_value :
  let f (x : ℤ) := 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12
  let v : ℤ := 3
  let v1 (x : ℤ) : ℤ := v * x + 5
  let v2 (x : ℤ) (v1x : ℤ) : ℤ := v1x * x + 6
  let v3 (x : ℤ) (v2x : ℤ) : ℤ := v2x * x + 79
  x = -4 →
  v3 x (v2 x (v1 x)) = -57 :=
by
  sorry

end horner_method_v3_value_l808_808810


namespace lara_cookies_l808_808159

theorem lara_cookies (total_cookies trays rows_per_row : ℕ)
  (h_total : total_cookies = 120)
  (h_trays : trays = 4)
  (h_rows_per_row : rows_per_row = 6) :
  total_cookies / rows_per_row / trays = 5 :=
by
  sorry

end lara_cookies_l808_808159


namespace football_attendance_l808_808678

open Nat

theorem football_attendance:
  (Saturday Monday Wednesday Friday expected_total actual_total: ℕ)
  (h₀: Saturday = 80)
  (h₁: Monday = Saturday - 20)
  (h₂: Wednesday = Monday + 50)
  (h₃: Friday = Saturday + Monday)
  (h₄: expected_total = 350)
  (h₅: actual_total = Saturday + Monday + Wednesday + Friday) :
  actual_total = expected_total + 40 :=
  sorry

end football_attendance_l808_808678


namespace nums_between_2000_and_3000_div_by_360_l808_808521

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l808_808521


namespace solve_for_x_l808_808099

noncomputable def M (x : ℝ) := {-2, 3 * x ^ 2 + 3 * x - 4, x ^ 2 + x - 4}

theorem solve_for_x (x : ℝ) : 2 ∈ M x → x = 2 ∨ x = -3 :=
by
  intro h
  have h1 : 2 = 3 * x ^ 2 + 3 * x - 4 → x = 1 ∨ x = -2 := by
    intro h_eq
    have h_quad : 3 * x ^ 2 + 3 * x - 4 - 2 = 0 := by linarith
    exact (eq_division).mpr h_quad
  
  have h2 : 2 = x ^ 2 + x - 4 → x = 2 ∨ x = -3 := by
    intro h_eq
    have h_quad : x ^ 2 + x - 4 - 2 = 0 := by linarith
    exact (eq_division).mpr h_quad
  
  sorry

end solve_for_x_l808_808099


namespace unbounded_line_symmetry_half_line_no_symmetry_segment_symmetry_l808_808716

-- 1. Symmetry of an unbounded line
theorem unbounded_line_symmetry (L : Set Point) (infinite : ∀ P : Point, P ∈ L) :
  ∀ O : Point, O ∈ L → ∀ P : Point, P ∈ L → ∃ P' : Point, P' ∈ L ∧ midpoint O P P' :=
sorry

-- 2. Symmetry of a half line
theorem half_line_no_symmetry (L : Set Point) (start : Point) (infinite : ∀ P : Point, P ∈ L → P ≠ start) :
  ¬ ∃ O : Point, O ∈ L ∧ ∀ P : Point, P ∈ L → ∃ P' : Point, P' ∈ L ∧ midpoint O P P' :=
sorry

-- 3. Symmetry of a line segment
theorem segment_symmetry (A B : Point) (L : Set Point) (segment : ∀ P : Point, P ∈ L ↔ P ∈ lineSegment A B) :
  ∃ M : Point, M = midpoint A B ∧ ∀ P : Point, P ∈ L → ∃ P' : Point, P' ∈ L ∧ midpoint M P P' :=
sorry

end unbounded_line_symmetry_half_line_no_symmetry_segment_symmetry_l808_808716


namespace units_digit_of_7_pow_6_pow_5_l808_808438

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808438


namespace courtyard_width_l808_808317

theorem courtyard_width 
  (L : ℝ) (N : ℕ) (brick_length brick_width : ℝ) (courtyard_area : ℝ)
  (hL : L = 18)
  (hN : N = 30000)
  (hbrick_length : brick_length = 0.12)
  (hbrick_width : brick_width = 0.06)
  (hcourtyard_area : courtyard_area = (N : ℝ) * (brick_length * brick_width)) :
  (courtyard_area / L) = 12 :=
by
  sorry

end courtyard_width_l808_808317


namespace minimum_possible_perimeter_l808_808262

theorem minimum_possible_perimeter (a b c : ℤ) (h1 : 2 * a + 8 * c = 2 * b + 10 * c) 
                                  (h2 : 4 * c * (sqrt (a^2 - (4 * c)^2)) = 5 * c * (sqrt (b^2 - (5 * c)^2))) 
                                  (h3 : a - b = c) : 
    2 * a + 8 * c = 740 :=
by
  sorry

end minimum_possible_perimeter_l808_808262


namespace calculate_expression_l808_808344

theorem calculate_expression :
  let a := 2^4
  let b := 2^2
  let c := 2^3
  (a^2 / b^3) * c^3 = 2048 :=
by
  sorry -- Proof is omitted as per instructions

end calculate_expression_l808_808344


namespace acute_triangle_cosine_inequality_l808_808040

variable {α β γ : ℝ}

theorem acute_triangle_cosine_inequality 
  (h₁ : 0 < α ∧ α < π/2) 
  (h₂ : 0 < β ∧ β < π/2) 
  (h₃ : 0 < γ ∧ γ < π/2) 
  (h₄ : α + β + γ = π) :
  (cos α / cos (β - γ)) + 
  (cos β / cos (γ - α)) + 
  (cos γ / cos (α - β)) ≥ 3 / 2 := 
sorry

end acute_triangle_cosine_inequality_l808_808040


namespace percentage_of_50_is_40_of_125_l808_808727

theorem percentage_of_50_is_40_of_125 :
  let part := 50 in
  let whole := 125 in
  (part / whole : ℝ) * 100 = 40 := by
  sorry

end percentage_of_50_is_40_of_125_l808_808727


namespace domain_of_f_f_is_odd_f_gt_zero_l808_808059

-- Definition of the function f(x) and its properties
def f (x : ℝ) (a : ℝ) : ℝ := log a (1 - x) - log a (1 + x)

-- Conditions
variables (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)

-- Part 1: Domain of the function
theorem domain_of_f : ∀ x, -1 < x ∧ x < 1 ↔ (1 - x > 0 ∧ 1 + x > 0) :=
by
  intros x
  split
  { intros hx
    exact ⟨sub_pos_of_lt hx.2, sub_pos_of_lt hx.1⟩ },
  { intros h
    exact ⟨sub_lt_iff_lt_add.mp h.2, add_lt_iff_neg_left.mp h.1⟩ },
  sorry

-- Part 2: Parity of the function
theorem f_is_odd : ∀ x, f x a = -f (-x) a :=
by
  intros x
  unfold f
  rw [log_div h₁, log_div h₁]
  exact log_neg_eq_log_of_pow_ne_zero h₂ h₁
  sorry

-- Part 3: Finding values where f(x) > 0
theorem f_gt_zero : f (3/5) a = 2 → f x a > 0 ↔ (0 < x ∧ x < 1) :=
by
  intros h
  have ha : a = 1/2, from sorry,
  rw [ha] at *,
  split
  { intros hx
    unfold f
    rw [log_div_of_lt, log_div_of_lt]
    exact sorry },
  { intros hx
    unfold f
    rw [log_div_of_lt, log_div_of_lt]
    exact sorry },

end domain_of_f_f_is_odd_f_gt_zero_l808_808059


namespace number_of_multiples_in_range_l808_808513

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l808_808513


namespace quadratic_equation_roots_l808_808983

theorem quadratic_equation_roots (η ζ : ℝ) 
  (h1 : (η + ζ) / 2 = 6)
  (h2 : Real.sqrt (η * ζ) = 10) :
  (Polynomial.monic (Polynomial.X^2 - Polynomial.C (η + ζ) * Polynomial.X + Polynomial.C (η * ζ))) := 
by
  sorry

end quadratic_equation_roots_l808_808983


namespace sin_cos_quotient_l808_808116

noncomputable def tan (x : ℝ) : ℝ := Math.tan x

theorem sin_cos_quotient (α β p q : ℝ) 
  (h1 : tan α = -p - Math.sqrt(p^2 - 4 * q) / 2)
  (h2 : tan β = -p + Math.sqrt(p^2 - 4 * q) / 2) :
  (Real.sin (α + β)) / (Real.cos (α - β)) = -p / (q + 1) := 
sorry

end sin_cos_quotient_l808_808116


namespace point_on_modified_function_sum_coords_l808_808485

theorem point_on_modified_function_sum_coords :
  (∃ f : ℝ → ℝ, f 8 = 5) →
  ∃ x y : ℝ, 3 * y = (f (3 * x)) / 3 + 3 ∧ x = 8 / 3 ∧ y = 14 / 9 ∧ x + y = 38 / 9 :=
by
  intros h
  -- Proof will go here
  sorry

end point_on_modified_function_sum_coords_l808_808485


namespace range_of_a_l808_808986

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ+, (-1) ^ n * a < 2 + (-1) ^ (n + 1) / (n : ℝ)) →
  -2 ≤ a ∧ a < 3 / 2 :=
by
  sorry

end range_of_a_l808_808986


namespace subtract_f_eq_2x_minus_1_l808_808121

noncomputable def f (x : ℝ) : ℝ := x^2

theorem subtract_f_eq_2x_minus_1 (x : ℝ) : f(x) - f(x - 1) = 2 * x - 1 := 
by sorry

end subtract_f_eq_2x_minus_1_l808_808121


namespace range_of_f_l808_808841

theorem range_of_f :
  ∀ x : ℝ, (f(x) = 1 - 2 * (sin x * cos x)^2 + (sin x * cos x)^2) -> 
           ∃ y : ℝ, y ∈ [3/4, 1] where
           f(x) = sin x ^ 4 + (sin x * cos x) ^ 2 + cos x ^ 4 :=
by
  sorry

end range_of_f_l808_808841


namespace football_attendance_l808_808679

open Nat

theorem football_attendance:
  (Saturday Monday Wednesday Friday expected_total actual_total: ℕ)
  (h₀: Saturday = 80)
  (h₁: Monday = Saturday - 20)
  (h₂: Wednesday = Monday + 50)
  (h₃: Friday = Saturday + Monday)
  (h₄: expected_total = 350)
  (h₅: actual_total = Saturday + Monday + Wednesday + Friday) :
  actual_total = expected_total + 40 :=
  sorry

end football_attendance_l808_808679


namespace units_digit_of_7_pow_6_pow_5_l808_808427

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808427


namespace quadratic_axis_of_symmetry_l808_808651

theorem quadratic_axis_of_symmetry (b c : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0, 3))
  (hB : B = (2, 3)) 
  (h_passA : 3 = 0^2 + b * 0 + c) 
  (h_passB : 3 = 2^2 + b * 2 + c) : 
  ∃ x, x = 1 :=
by {
  -- Given: The quadratic function y = x^2 + bx + c
  -- Conditions: Passes through A(0, 3) and B(2, 3)
  -- We need to prove: The axis of symmetry is x = 1
  sorry,
}

end quadratic_axis_of_symmetry_l808_808651


namespace omega_value_l808_808455

-- Define the problem conditions
def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- Define the main theorem
theorem omega_value (ω : ℝ) (hω : ω > 0) (h1 : f ω (Real.pi / 6) = f ω (Real.pi / 3))
  (h2 : ∀ x ∈ Ioo (Real.pi / 6) (Real.pi / 3), is_min (f ω) x) :
  ω = 14 / 3 :=
sorry

end omega_value_l808_808455


namespace car_distance_in_15_minutes_l808_808106

-- Define the conditions
def train_speed : ℝ := 120  -- the speed of the train in miles per hour
def car_speed : ℝ := (2 / 3) * train_speed  -- the speed of the car

-- Define the time in hours
def time_in_hours : ℝ := 15 / 60

-- State the theorem to be proven
theorem car_distance_in_15_minutes : car_speed * time_in_hours = 20 := by
  sorry

end car_distance_in_15_minutes_l808_808106


namespace find_m_l808_808504

variables {m : ℝ}
def vec_a : ℝ × ℝ := (-2, 3)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) (h : perpendicular vec_a (vec_b m)) : m = 2 :=
by
  sorry

end find_m_l808_808504


namespace smallest_positive_period_sin2_cos2_l808_808665

theorem smallest_positive_period_sin2_cos2 : 
  ∃ T > 0, (∀ x, sin (x + T)^2 + cos (x + T)^2 = sin x^2 + cos x^2) ∧ 
  (∀ T' > 0, (∀ x, sin (x + T')^2 + cos (x + T')^2 = sin x^2 + cos x^2) → T ≤ T') → T = π := 
  sorry

end smallest_positive_period_sin2_cos2_l808_808665


namespace pencil_packing_l808_808191

theorem pencil_packing (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (a % 10 = 7) →
  (a % 12 = 9) →
  (a = 237 ∨ a = 297) :=
by {
  assume h_range h_red_boxes h_blue_boxes,
  sorry
}

end pencil_packing_l808_808191


namespace f_decreasing_increasing_find_b_range_l808_808943

-- Define the function f(x) and prove its properties for x > 0 and x < 0
noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem f_decreasing_increasing (a : ℝ) (h : a > 0):
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt a → ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a) → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, 0 < Real.sqrt a → Real.sqrt a ≤ x → ∀ x1 x2 : ℝ, (Real.sqrt a ≤ x1 ∧ x1 < x2) → f x1 a < f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → -Real.sqrt a ≤ x ∧ x < 0 → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → x < -Real.sqrt a → f x1 a < f x2 a)
:= sorry

-- Define the function h(x) and find the range of b
noncomputable def h (x : ℝ) : ℝ := x + 4 / x - 8
noncomputable def g (x b : ℝ) : ℝ := -x - 2 * b

theorem find_b_range:
  (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 3 ∧ g x2 b = h x1) ↔
  1/2 ≤ b ∧ b ≤ 1
:= sorry

end f_decreasing_increasing_find_b_range_l808_808943


namespace arithmetic_progression_and_area_equality_l808_808477

theorem arithmetic_progression_and_area_equality
  (A B C I D : Type)
  (O : circcircle_triangle A B C)
  (I_incenter : incenter I A B C)
  (AI_intersects_O_at_D : ray_intersects_circle AI O D) :
  (arithmetic_progression AB BC CA) ↔ (area IBC = area DBC) :=
sorry

end arithmetic_progression_and_area_equality_l808_808477


namespace fixed_point_coordinates_l808_808138

theorem fixed_point_coordinates
  (m : ℝ) 
  (h_m : m ≠ 0)
  (h_m_range : -2 * Real.sqrt 2 < m ∧ m < 0 ∨ 0 < m ∧ m < 2 * Real.sqrt 2)
  (h_line_ellipse_intersection : ∃ x y, 
    (sqrt 2 * x - y + m = 0 ∧ (y^2 / 4 + x^2 / 2 = 1))) :
  ∃ P : (ℝ × ℝ), 
    (P = (1, Real.sqrt 2) ∨ P = (-1, -Real.sqrt 2)) ∧
    ∀ (A B : ℝ × ℝ), 
      ((sqrt 2 * A.1 - A.2 + m = 0 ∧ (A.2^2 / 4 + A.1^2 / 2 = 1)) →
       (sqrt 2 * B.1 - B.2 + m = 0 ∧ (B.2^2 / 4 + B.1^2 / 2 = 1)) →
       (let k_PA := (A.2 - P.2) / (A.1 - P.1),
            k_PB := (B.2 - P.2) / (B.1 - P.1) 
        in k_PA + k_PB = 0)) := sorry

end fixed_point_coordinates_l808_808138


namespace original_sets_exist_l808_808905

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808905


namespace store_income_l808_808552

def pencil_store_income (p_with_eraser_qty p_with_eraser_cost p_regular_qty p_regular_cost p_short_qty p_short_cost : ℕ → ℝ) : ℝ :=
  (p_with_eraser_qty * p_with_eraser_cost) + (p_regular_qty * p_regular_cost) + (p_short_qty * p_short_cost)

theorem store_income : 
  pencil_store_income 200 0.8 40 0.5 35 0.4 = 194 := 
by sorry

end store_income_l808_808552


namespace units_digit_of_7_pow_6_pow_5_l808_808432

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808432


namespace calculate_expression_l808_808226

theorem calculate_expression :
  6 * 1000 + 5 * 100 + 6 * 1 = 6506 :=
by
  sorry

end calculate_expression_l808_808226


namespace ab_equals_one_l808_808118

theorem ab_equals_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (f : ℝ → ℝ) (h3 : f = abs ∘ log) (h4 : f a = f b) : a * b = 1 :=
by
  sorry

end ab_equals_one_l808_808118


namespace sum_even_integers_12_to_40_l808_808286

theorem sum_even_integers_12_to_40 : 
  ∑ k in finset.filter (λ k, even k) (finset.range 41), k = 390 := by
  sorry

end sum_even_integers_12_to_40_l808_808286


namespace product_of_valid_c_l808_808388

def valid_c (c : ℤ) : Prop :=
  24 * 24 - 40 * c > 0 ∧ c > 0

theorem product_of_valid_c : (∏ c in finset.filter valid_c (finset.range 15), c) = 87178291200 := by
  sorry

end product_of_valid_c_l808_808388


namespace original_sets_exist_l808_808899

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808899


namespace remove_one_from_each_column_l808_808721

theorem remove_one_from_each_column (n : ℕ) (a b : Fin n → ℝ) (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_sum : ∀ i, a i + b i = 1) :
  ∃ a' b' : Fin n → ℝ,
    (∀ i, a' i = a i ∨ b' i = b i) ∧
    (∀ i, (a' i > 0 → b' i = 0) ∧ (b' i > 0 → a' i = 0)) ∧
    (∑ i, a' i ≤ (n + 1) / 4) ∧
    (∑ i, b' i ≤ (n + 1) / 4) := 
sorry

end remove_one_from_each_column_l808_808721


namespace incorrect_statement_about_sqrt5_is_d_l808_808704

noncomputable def sqrt (x : ℝ) : ℝ := if x >= 0 then Real.sqrt x else 0

theorem incorrect_statement_about_sqrt5_is_d :
  let sqrt5 : ℝ := sqrt 5
  let sqrt20 : ℝ := sqrt 20
  (Real.sqrt 5).is_irrational ∧
  (∀ a b : ℕ, a ≠ 0 → b ≠ 0 → a ^ 2 ≠ b ^ 2 * (5 : ℝ)) ∧
  (sqrt5 ^ 2 = 5) ∧
  (sqrt20 = 2 * sqrt5) →
  (¬ (sqrt5 = 2 * sqrt5)) :=
by
  sorry

end incorrect_statement_about_sqrt5_is_d_l808_808704


namespace shortest_distance_l808_808731

-- Define the conditions
def track_length : ℕ := 400 -- Length of the circular track in meters
def meeting_time : ℕ := 8 * 60 -- Time to meet for the third time in seconds
def speed_diff : ℝ := 0.1 -- Speed difference between A and B in meters per second

-- Define the speeds of A and B
def speed_b (x : ℝ) : ℝ := x -- Speed of B in meters per second
def speed_a (x : ℝ) : ℝ := x + speed_diff -- Speed of A in meters per second

-- Define the meeting equation
def meeting_eq (x : ℝ) : Prop :=
  meeting_time * (speed_a x + speed_b x) = track_length * 3

-- The final goal is to prove the shortest distance is 176 meters given the conditions
theorem shortest_distance
  (x : ℝ)
  (h : meeting_eq x) :
  1.2 * 60 * 8 % track_length = 176 :=
sorry

end shortest_distance_l808_808731


namespace combined_bus_capacity_eq_40_l808_808609

theorem combined_bus_capacity_eq_40 (train_capacity : ℕ) (fraction : ℚ) (num_buses : ℕ) 
  (h_train_capacity : train_capacity = 120)
  (h_fraction : fraction = 1/6)
  (h_num_buses : num_buses = 2) :
  num_buses * (train_capacity * fraction).toNat = 40 := by
  sorry

end combined_bus_capacity_eq_40_l808_808609


namespace work_rate_calculate_l808_808739

theorem work_rate_calculate (A_time B_time C_time total_time: ℕ) 
  (hA : A_time = 4) 
  (hB : B_time = 8)
  (hTotal : total_time = 2) : 
  C_time = 8 :=
by
  sorry

end work_rate_calculate_l808_808739


namespace range_f_on_interval_l808_808926

noncomputable def f (x : ℝ) (k : ℝ) := 1 / (x^k)

theorem range_f_on_interval (k : ℝ) (h : k > 0) :
  Set.Range (λ x, f x k) (Set.Ici 2) = Set.Ioc 0 (1 / (2^k)) :=
by
  sorry

end range_f_on_interval_l808_808926


namespace inequality_proof_l808_808444

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem inequality_proof :
  (a / (a + b)) * ((a + 2 * b) / (a + 3 * b)) < Real.sqrt (a / (a + 4 * b)) :=
sorry

end inequality_proof_l808_808444


namespace conditional_probability_l808_808049

def P : Type := ℚ -- assuming probabilities are rational numbers

variable (PAB PA : P)

theorem conditional_probability (h1 : PAB = 2 / 15) (h2 : PA = 2 / 5) : PAB / PA = 1 / 3 :=
by
  sorry

end conditional_probability_l808_808049


namespace salon_fingers_l808_808323

theorem salon_fingers (clients non_clients total_fingers cost_per_client total_earnings : Nat)
  (h1 : cost_per_client = 20)
  (h2 : total_earnings = 200)
  (h3 : total_fingers = 210)
  (h4 : non_clients = 11)
  (h_clients : clients = total_earnings / cost_per_client)
  (h_people : total_fingers / 10 = clients + non_clients) :
  10 = total_fingers / (clients + non_clients) :=
by
  sorry

end salon_fingers_l808_808323


namespace original_sets_exist_l808_808877

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808877


namespace find_tangent_line_l808_808480

theorem find_tangent_line (k : ℝ) :
  (∃ k : ℝ, ∀ (x y : ℝ), y = k * (x - 1) + 3 ∧ k^2 + 1 = 1) →
  (∃ k : ℝ, k = 4 / 3 ∧ (k * x - y + 3 - k = 0) ∨ (x = 1)) :=
sorry

end find_tangent_line_l808_808480


namespace greatest_integer_l808_808276

-- Define the conditions for the problem
def isMultiple4 (n : ℕ) : Prop := n % 4 = 0
def notMultiple8 (n : ℕ) : Prop := n % 8 ≠ 0
def notMultiple12 (n : ℕ) : Prop := n % 12 ≠ 0
def gcf4 (n : ℕ) : Prop := Nat.gcd n 24 = 4
def lessThan200 (n : ℕ) : Prop := n < 200

-- State the main theorem
theorem greatest_integer : ∃ n : ℕ, lessThan200 n ∧ gcf4 n ∧ n = 196 :=
by
  sorry

end greatest_integer_l808_808276


namespace find_p_a_l808_808471

variables (p : ℕ → ℝ) (a b : ℕ)

-- Given conditions
axiom p_b : p b = 0.5
axiom p_b_given_a : p b / p a = 0.2 
axiom p_a_inter_b : p a * p b = 0.36

-- Problem statement
theorem find_p_a : p a = 1.8 :=
by
  sorry

end find_p_a_l808_808471


namespace functions_not_linearly_independent_l808_808170

noncomputable def linear_independent (s : set (ℝ → ℝ)) : Prop :=
∀ (l : ℝ → ℝ), (∀ x ∈ s, l x = 0) → (∀ x ∈ s, 0 = l x)

theorem functions_not_linearly_independent
  (n : ℕ)
  (x : fin n → ℝ → ℝ)
  (a : fin n → fin n → ℝ)
  (h_diff : ∀ i, differentiable ℝ (x i))
  (h_de : ∀ i t, derivative_at ℝ ℝ (x i) t = finset.sum finset.univ (λ j, a i j * x j t))
  (h_coeff_nonneg: ∀ i j, 0 ≤ a i j)
  (h_lim : ∀ i, filter.tendsto (x i) filter.at_top (filter.principal {0})) :
  ¬linear_independent {x i | i : fin n} := 
sorry

end functions_not_linearly_independent_l808_808170


namespace ratio_of_oranges_l808_808131

def num_good_oranges : ℕ := 24
def num_bad_oranges : ℕ := 8
def ratio_good_to_bad : ℕ := num_good_oranges / num_bad_oranges

theorem ratio_of_oranges : ratio_good_to_bad = 3 := by
  show 24 / 8 = 3
  sorry

end ratio_of_oranges_l808_808131


namespace max_net_income_meeting_point_l808_808199

theorem max_net_income_meeting_point :
  let A := (9 : ℝ)
  let B := (6 : ℝ)
  let cost_per_mile := 1
  let payment_per_mile := 2
  ∃ x : ℝ, 
  let AP := Real.sqrt ((x - 9)^2 + 12^2)
  let PB := Real.sqrt ((x - 6)^2 + 3^2)
  let net_income := payment_per_mile * PB - (AP + PB)
  x = -12.5 := 
sorry

end max_net_income_meeting_point_l808_808199


namespace tangents_to_discriminant_parabola_l808_808037

variable (a : ℝ) (p q : ℝ)

theorem tangents_to_discriminant_parabola :
  (a^2 + a * p + q = 0) ↔ (p^2 - 4 * q = 0) :=
sorry

end tangents_to_discriminant_parabola_l808_808037


namespace angle_equality_in_quadrilateral_l808_808148

theorem angle_equality_in_quadrilateral
    {A B C D M : Type*}
    [EuclideanGeometry A B C D M]
    (ABMD_parallelogram : parallelogram A B M D)
    (angle_CBM_eq_angle_CDM : ∠CBM = ∠CDM) :
    ∠ACD = ∠BCM :=
by
-- Proof will be provided here
sorry

end angle_equality_in_quadrilateral_l808_808148


namespace ratio_of_third_week_growth_l808_808803

-- Define the given conditions
def week1_growth : ℕ := 2  -- growth in week 1
def week2_growth : ℕ := 2 * week1_growth  -- growth in week 2
def total_height : ℕ := 22  -- total height after three weeks

/- 
  Statement: Prove that the growth in the third week divided by 
  the growth in the second week is 4, i.e., the ratio 4:1.
-/
theorem ratio_of_third_week_growth :
  ∃ x : ℕ, 4 * x = (total_height - week1_growth - week2_growth) ∧ x = 4 :=
by
  use 4
  sorry

end ratio_of_third_week_growth_l808_808803


namespace games_new_friends_l808_808573

-- Definitions based on the conditions
def total_games_all_friends : ℕ := 141
def games_old_friends : ℕ := 53

-- Statement of the problem
theorem games_new_friends {games_new_friends : ℕ} :
  games_new_friends = total_games_all_friends - games_old_friends :=
sorry

end games_new_friends_l808_808573


namespace range_of_a_monotonically_increasing_l808_808095

noncomputable def f (a x : ℝ) : ℝ := if x ≤ 0.5 then a^x else (2 * a - 1) * x

theorem range_of_a_monotonically_increasing (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → a ∈ Ici ((2 + Real.sqrt 3) / 2) :=
by
  sorry

end range_of_a_monotonically_increasing_l808_808095


namespace Donovan_Mitchell_goal_l808_808371

theorem Donovan_Mitchell_goal 
  (current_avg : ℕ) 
  (current_games : ℕ) 
  (target_avg : ℕ) 
  (total_games : ℕ) 
  (remaining_games : ℕ) 
  (points_scored_so_far : ℕ)
  (points_needed_total : ℕ)
  (points_needed_remaining : ℕ) :
  (current_avg = 26) ∧
  (current_games = 15) ∧
  (target_avg = 30) ∧
  (total_games = 20) ∧
  (remaining_games = 5) ∧
  (points_scored_so_far = current_avg * current_games) ∧
  (points_needed_total = target_avg * total_games) ∧
  (points_needed_remaining = points_needed_total - points_scored_so_far) →
  (points_needed_remaining / remaining_games = 42) :=
by
  sorry

end Donovan_Mitchell_goal_l808_808371


namespace quadruple_count_l808_808107

theorem quadruple_count (n : ℕ) : 
  ∃ count : ℕ, count = (Finset.range (n + 2 + 1)).card.choose 4 ∧ count = ∑ i in Finset.Ico 1 (n + 2 + 1), 
                                                    ∑ j in Finset.Ico (i + 1) (n + 2 + 1), 
                                                    ∑ k in Finset.Ico (j + 1) (n + 2 + 1), 
                                                    ∑ h in Finset.Ico (k + 1) (n + 2 + 1), 1 := 
by { sorry }

end quadruple_count_l808_808107


namespace odd_function_m_value_l808_808985

theorem odd_function_m_value (m : ℝ) (h : ∀ x : ℝ, sin x + m - 1 = -(sin (-x) + m - 1)) : m = 1 :=
by
  sorry

end odd_function_m_value_l808_808985


namespace new_volume_l808_808769

theorem new_volume (l w h : ℝ) 
  (h1 : l * w * h = 4320)
  (h2 : l * w + w * h + l * h = 852)
  (h3 : l + w + h = 52) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := sorry

end new_volume_l808_808769


namespace quadratic_symmetry_l808_808535

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * x + 1

theorem quadratic_symmetry 
  (a b x1 x2 : ℝ) 
  (h_quad : f x1 a b = f x2 a b) 
  (h_diff : x1 ≠ x2) 
  (h_nonzero : a ≠ 0) :
  f (x1 + x2) a b = 1 := 
by
  sorry

end quadratic_symmetry_l808_808535


namespace probability_xi_leq_7_l808_808539

noncomputable def probability_ball_draw_score : ℚ :=
  let red_balls := 4
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let score := λ (red black : ℕ), red + 3 * black
  let comb := λ n k, (nat.choose n k : ℚ)
  (comb red_balls 4 / comb total_balls 4) +
  (comb red_balls 3 * comb black_balls 1 / comb total_balls 4)

theorem probability_xi_leq_7 : probability_ball_draw_score = (13 / 35) := by
  sorry

end probability_xi_leq_7_l808_808539


namespace angela_deliveries_l808_808797

theorem angela_deliveries
  (n_meals : ℕ)
  (h_meals : n_meals = 3)
  (n_packages : ℕ)
  (h_packages : n_packages = 8 * n_meals) :
  n_meals + n_packages = 27 := by
  sorry

end angela_deliveries_l808_808797


namespace star_three_five_l808_808856

def star (x y : ℕ) := x^2 + 2 * x * y + y^2

theorem star_three_five : star 3 5 = 64 :=
by
  sorry

end star_three_five_l808_808856


namespace pencil_count_l808_808188

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l808_808188


namespace non_overlapping_original_sets_l808_808884

theorem non_overlapping_original_sets (n : ℕ) (h : n ≥ 1) :
  ∃ (S : fin (n + 1) → set (fin n × fin n)),
    (∀ i, S i.card = n - 1) ∧ 
    (∀ i j k, i ≠ j → k ∈ S i → k ∉ S j) ∧
    (∀ i, ∀ ⦃x y⦄, x ∈ S i → y ∈ S i → x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) :=
sorry

end non_overlapping_original_sets_l808_808884


namespace units_digit_pow_7_6_5_l808_808415

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808415


namespace count_divisibles_l808_808514

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l808_808514


namespace number_of_lists_proof_l808_808788

noncomputable def number_of_lists_possible : ℕ :=
  11

theorem number_of_lists_proof :
  ∃ n : ℕ, (∃ a b c d e : ℕ, a + b + c + d + e = 6 ∧ a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e) ∧ n = number_of_lists_possible :=
begin
  -- The proof would go here
  sorry
end

end number_of_lists_proof_l808_808788


namespace algebraic_expression_value_l808_808991

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x + 3 = 7) : 3 * x^2 + 3 * x + 7 = 19 :=
sorry

end algebraic_expression_value_l808_808991


namespace units_digit_7_pow_6_pow_5_l808_808412

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808412


namespace three_pow_2010_mod_eight_l808_808282

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end three_pow_2010_mod_eight_l808_808282


namespace find_complex_number_l808_808675

-- Define complex numbers and the given conditions
def complex_num (x y : ℤ) : ℂ := x + y * complex.I

theorem find_complex_number (x y d : ℤ) 
  (hx : 0 < x) (hy : 0 < y)
  (h : (complex_num x y) ^ 4 = 82 + d * complex.I) :
  complex_num x y = 1 + 3 * complex.I :=
sorry

end find_complex_number_l808_808675


namespace train_length_approx_500_l808_808778

noncomputable def length_of_train (speed_km_per_hr : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  speed_m_per_s * time_sec

theorem train_length_approx_500 :
  length_of_train 120 15 = 500 :=
by
  sorry

end train_length_approx_500_l808_808778


namespace big_box_doll_count_l808_808801

theorem big_box_doll_count:
  (∀ (small_box_doll_count big_box_count small_box_count total_doll_count : ℕ), 
  (small_box_doll_count = 4) →
  (big_box_count = 5) →
  (small_box_count = 9) →
  (total_doll_count = 71) →
  ∃ (big_box_doll_count : ℕ), (total_doll_count = big_box_count * big_box_doll_count + small_box_count * small_box_doll_count) →
  big_box_doll_count = 7) :=
begin
  intros,
  use 7,
  sorry
end

end big_box_doll_count_l808_808801


namespace total_worth_of_stock_l808_808327

theorem total_worth_of_stock (total_worth profit_fraction profit_rate loss_fraction loss_rate overall_loss : ℝ) :
  profit_fraction = 0.20 ->
  profit_rate = 0.20 -> 
  loss_fraction = 0.80 -> 
  loss_rate = 0.10 -> 
  overall_loss = 500 ->
  total_worth - (profit_fraction * total_worth * profit_rate) - (loss_fraction * total_worth * loss_rate) = overall_loss ->
  total_worth = 12500 :=
by
  sorry

end total_worth_of_stock_l808_808327


namespace find_special_three_digit_numbers_l808_808382

theorem find_special_three_digit_numbers :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 
  (100 * a + 10 * b + (c + 3)) % 10 + (100 * a + 10 * (b + 1) + c).div 10 % 10 + (100 * (a + 1) + 10 * b + c).div 100 % 10 + 3 = 
  (a + b + c) / 3)} → n = 117 ∨ n = 207 ∨ n = 108 :=
by
  sorry

end find_special_three_digit_numbers_l808_808382


namespace unique_n_exists_l808_808578

theorem unique_n_exists (a : ℕ → ℕ) (h : ∀ n m, n < m → a n < a m) :
  ∃! n : ℕ, 1 ≤ n ∧ a n < (∑ i in range (n + 1), a i) / n ∧ (∑ i in range (n + 1), a i) / n ≤ a (n + 1) :=
by
  sorry

end unique_n_exists_l808_808578


namespace initial_deadline_is_75_days_l808_808179

-- Define constants for the problem
def initial_men : ℕ := 100
def initial_hours_per_day : ℕ := 8
def days_worked_initial : ℕ := 25
def fraction_work_completed : ℚ := 1 / 3
def additional_men : ℕ := 60
def new_hours_per_day : ℕ := 10
def total_man_hours : ℕ := 60000

-- Prove that the initial deadline for the project is 75 days
theorem initial_deadline_is_75_days : 
  ∃ (D : ℕ), (D * initial_men * initial_hours_per_day = total_man_hours) ∧ D = 75 := 
by {
  sorry
}

end initial_deadline_is_75_days_l808_808179


namespace product_of_positive_integers_for_real_roots_l808_808385

theorem product_of_positive_integers_for_real_roots :
  let discriminant_pos (a b c : ℤ) := b^2 - 4 * a * c > 0 in
  ∏ i in (finset.range (15)).filter (λ c, discriminant_pos 10 24 c), c = 87178291200 := by
  sorry

end product_of_positive_integers_for_real_roots_l808_808385


namespace evaluate_expr_l808_808373

theorem evaluate_expr (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 5 * x^(y+1) + 6 * y^(x+1) = 2751 :=
by
  rw [h₁, h₂]
  rfl

end evaluate_expr_l808_808373


namespace multinomial_vandermonde_convolution_l808_808723

variable {N N1 N2 r : ℕ} -- defining required variables
variable {n k : Fin r → ℕ} -- n_i and k_i are functions from Fin r to ℕ

def multinomialCoefficient (N : ℕ) (n : Fin r → ℕ) : ℕ :=
  Nat.factorial N / ((Finset.univ.prod (λ i => Nat.factorial (n i))))

theorem multinomial_vandermonde_convolution
  (h_sum : (Finset.univ.sum n) = N1 + N2)
  (h_sum_k : ∀ {k : Fin r → ℕ}, (Finset.univ.sum k) = N1 → (∀ i, 0 ≤ k i ∧ k i ≤ n i)) :
  multinomialCoefficient (N1 + N2) n =
    Finset.univ.sum (λ k, multinomialCoefficient N1 k * multinomialCoefficient N2 (λ i => n i - k i)) :=
sorry

end multinomial_vandermonde_convolution_l808_808723


namespace farmer_total_acres_l808_808757

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l808_808757


namespace combined_capacity_is_40_l808_808611

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l808_808611


namespace number_of_digits_is_nine_l808_808019

noncomputable def expression : ℝ := (8^4 * 4^12) / 2^8

theorem number_of_digits_is_nine (x : ℝ) (h : x = expression) : ⌊real.log10 x⌋ + 1 = 9 :=
sorry

end number_of_digits_is_nine_l808_808019


namespace min_distance_from_curve_to_line_l808_808617

open Real

-- Definitions and conditions
def curve_eq (x y: ℝ) : Prop := (x^2 - y - 2 * log (sqrt x) = 0)
def line_eq (x y: ℝ) : Prop := (4 * x + 4 * y + 1 = 0)

-- The main statement
theorem min_distance_from_curve_to_line :
  ∃ (x y : ℝ), curve_eq x y ∧ y = x^2 - 2 * log (sqrt x) ∧ line_eq x y ∧ y = -x - 1/4 ∧ 
               |4 * (1/2) + 4 * ((1/4) + log 2) + 1| / sqrt 32 = sqrt 2 / 2 * (1 + log 2) :=
by
  -- We skip the proof as requested, using sorry:
  sorry

end min_distance_from_curve_to_line_l808_808617


namespace equal_boys_girls_probability_l808_808817

theorem equal_boys_girls_probability :
  (let distributions := {s : list char | s.length = 4 ∧ (∀ c ∈ s, c = 'B' ∨ c = 'G')} in
   let successful := {s | s ∈ distributions ∧ s.count 'B' = 2 ∧ s.count 'G' = 2} in
   (successful.to_finset.card.to_rat / distributions.to_finset.card.to_rat) = (3 / 8))
:= by
  sorry

end equal_boys_girls_probability_l808_808817


namespace original_sets_exist_l808_808903

theorem original_sets_exist (n : ℕ) (h : 0 < n) :
  ∃ sets : list (set (ℕ × ℕ)), 
    (∀ s ∈ sets, s.card = n - 1 ∧ (∀ p1 p2 ∈ s, p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    sets.card = n + 1 ∧ 
    (∀ s1 s2 ∈ sets, s1 ≠ s2 → s1 ∩ s2 = ∅) :=
  sorry

end original_sets_exist_l808_808903


namespace max_coconuts_needed_l808_808597

theorem max_coconuts_needed (goats : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) 
  (final_goats : ℕ) : 
  goats = 19 ∧ coconuts_per_crab = 3 ∧ crabs_per_goat = 6 →
  ∃ coconuts, coconuts = 342 :=
by
  sorry

end max_coconuts_needed_l808_808597


namespace question_true_l808_808168
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end question_true_l808_808168


namespace smallest_three_digit_number_satisfying_conditions_l808_808032

def is_odd_digit (n : ℕ) : Prop :=
  n % 2 = 1

def sum_contains_only_odd_digits (n m : ℕ) : Prop :=
  (n + m).digits.all is_odd_digit

theorem smallest_three_digit_number_satisfying_conditions :
  ∃ n, n = 209 ∧ (sum_contains_only_odd_digits n (reverse_digits n)) :=
sorry

end smallest_three_digit_number_satisfying_conditions_l808_808032


namespace original_sets_exist_l808_808896

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808896


namespace minimum_common_perimeter_exists_l808_808258

noncomputable def find_minimum_perimeter
  (a b x : ℕ) 
  (is_int_sided_triangle_1 : 2 * a + 20 * x = 2 * b + 25 * x)
  (is_int_sided_triangle_2 : 20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2))) 
  (base_ratio : 20 * 2 * (a - b) = 25 * 2 * (a - b)): ℕ :=
2 * a + 20 * (2 * (a - b))

-- The final goal should prove the minimum perimeter under the given conditions.
theorem minimum_common_perimeter_exists :
∃ (minimum_perimeter : ℕ), 
  (∀ (a b x : ℕ), 
    2 * a + 20 * x = 2 * b + 25 * x → 
    20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2)) → 
    20 * 2 * (a - b) = 25 * 2 * (a - b) → 
    minimum_perimeter = 2 * a + 20 * x) :=
sorry

end minimum_common_perimeter_exists_l808_808258


namespace total_coins_l808_808971

theorem total_coins (total_value : ℕ) (value_2_coins : ℕ) (num_2_coins : ℕ) (num_1_coins : ℕ) : 
  total_value = 402 ∧ value_2_coins = 2 * num_2_coins ∧ num_2_coins = 148 ∧ total_value = value_2_coins + num_1_coins →
  num_1_coins + num_2_coins = 254 :=
by
  intros h
  sorry

end total_coins_l808_808971


namespace slope_angle_of_tangent_line_l808_808241

noncomputable def curve (x : ℝ) : ℝ := x^3 - 2 * x + 4

theorem slope_angle_of_tangent_line :
  let x := 1
  let y := curve x
  let derivative (x : ℝ) := 3 * x^2 - 2
  let slope := derivative x
  slope = 1 → 
  ∀ θ, tan θ = slope → θ = Real.pi / 4 := 
by 
  intros x y derivative slope hθ θ htan
  sorry

end slope_angle_of_tangent_line_l808_808241


namespace sum_of_squares_l808_808568

open Nat

def f (n : ℕ) : ℕ := (1 + 2 + ... + n)^2

theorem sum_of_squares (n : ℕ) : 
  f n = n * (n + 1) * (2 * n + 1) / 6 := 
by 
  sorry

end sum_of_squares_l808_808568


namespace repeating_decimal_fractional_representation_l808_808219

theorem repeating_decimal_fractional_representation :
  (0.36 : ℝ) = (4 / 11 : ℝ) :=
sorry

end repeating_decimal_fractional_representation_l808_808219


namespace valid_for_expression_c_l808_808141

def expression_a_defined (x : ℝ) : Prop := x ≠ 2
def expression_b_defined (x : ℝ) : Prop := x ≠ 3
def expression_c_defined (x : ℝ) : Prop := x ≥ 2
def expression_d_defined (x : ℝ) : Prop := x ≥ 3

theorem valid_for_expression_c :
  (expression_a_defined 2 = false ∧ expression_a_defined 3 = true) ∧
  (expression_b_defined 2 = true ∧ expression_b_defined 3 = false) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) ∧
  (expression_d_defined 2 = false ∧ expression_d_defined 3 = true) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) := by
  sorry

end valid_for_expression_c_l808_808141


namespace parity_D2021_D2022_D2023_l808_808325

-- Defining the sequence according to the given conditions
def D : ℕ → ℕ
| 0       := 1
| 1       := 1
| 2       := 1
| (n + 3) := D (n + 2) + D (n + 1)

-- Proving the parities of D_{2021}, D_{2022}, and D_{2023}
theorem parity_D2021_D2022_D2023 :
  (D 2021 % 2 = 1) ∧ (D 2022 % 2 = 1) ∧ (D 2023 % 2 = 1) :=
  sorry

end parity_D2021_D2022_D2023_l808_808325


namespace value_of_F_l808_808818

theorem value_of_F (D E F : ℕ) (hD : D < 10) (hE : E < 10) (hF : F < 10)
    (h1 : (8 + 5 + D + 7 + 3 + E + 2) % 3 = 0)
    (h2 : (4 + 1 + 7 + D + E + 6 + F) % 3 = 0) : 
    F = 6 :=
by
  sorry

end value_of_F_l808_808818


namespace perpendicular_line_eq_l808_808329

theorem perpendicular_line_eq :
  ∃ (A B C : ℝ), (A * 0 + B * 4 + C = 0) ∧ (A = 3) ∧ (B = 1) ∧ (C = -4) ∧ (3 * 1 + 1 * -3 = 0) :=
sorry

end perpendicular_line_eq_l808_808329


namespace number_of_multiples_in_range_l808_808510

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l808_808510


namespace count_divisibles_l808_808515

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l808_808515


namespace intersection_eq_l808_808101

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 0 }
def N : Set ℝ := { -1, 0, 1 }

theorem intersection_eq : M ∩ N = { -1, 0 } := by
  sorry

end intersection_eq_l808_808101


namespace molecular_weight_of_7_moles_l808_808280

structure AceticAcid :=
  (carbon_atoms : Int := 2)
  (hydrogen_atoms : Int := 4)
  (oxygen_atoms : Int := 2)
  (carbon_weight : Float := 12.01)
  (hydrogen_weight : Float := 1.008)
  (oxygen_weight : Float := 16.00)

def molecular_weight (acid : AceticAcid) : Float :=
  (acid.carbon_atoms * acid.carbon_weight) +
  (acid.hydrogen_atoms * acid.hydrogen_weight) +
  (acid.oxygen_atoms * acid.oxygen_weight)

def weight_of_moles (moles : Float) (molecular_weight : Float) : Float := 
  moles * molecular_weight

theorem molecular_weight_of_7_moles :
  weight_of_moles 7 (molecular_weight 
  {carbon_atoms:=2, hydrogen_atoms:=4, oxygen_atoms:=2, carbon_weight:=12.01, hydrogen_weight:=1.008, oxygen_weight:=16.00}) = 420.364 :=
  by sorry

end molecular_weight_of_7_moles_l808_808280


namespace derivative_of_sin_squared_minus_cos_squared_l808_808220

noncomputable def func (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem derivative_of_sin_squared_minus_cos_squared (x : ℝ) :
  deriv func x = 2 * Real.sin (2 * x) :=
sorry

end derivative_of_sin_squared_minus_cos_squared_l808_808220


namespace dartboard_distribution_l808_808789

theorem dartboard_distribution :
  ∃ lists : Finset (Multiset ℕ), 
    (∀ l ∈ lists, l.card = 5 ∧ l.sum = 6 ∧ l.order_of_bounds) ∧ 
    lists.card = 11 :=
begin 
  sorry 
end

end dartboard_distribution_l808_808789


namespace largest_divisor_of_n_l808_808296

theorem largest_divisor_of_n 
  (n : ℕ) (h_pos : n > 0) (h_div : 72 ∣ n^2) : 
  ∃ v : ℕ, v = 12 ∧ v ∣ n :=
by
  use 12
  sorry

end largest_divisor_of_n_l808_808296


namespace complement_union_l808_808176

open Set

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3}
noncomputable def C_UA : Set ℕ := U \ A

-- Statement to prove
theorem complement_union (U A B C_UA : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 5})
  (hB : B = {2, 3}) 
  (hCUA : C_UA = U \ A) : 
  (C_UA ∪ B) = {2, 3, 4} := 
sorry

end complement_union_l808_808176


namespace max_sum_squares_l808_808173

noncomputable def max_sum_squares_sides_diagonals (n : ℕ) (a : ℕ → ℂ) : ℂ :=
  ∑ i in (finset.range n), ∑ j in (finset.range n), complex.norm_sq (a i - a j)

/-- For a convex n-gon inscribed in a unit circle, the maximum sum of the squares of all sides
and diagonals is n^2. This maximum is achieved when the sum of the vertices' coordinates equals zero. -/
theorem max_sum_squares (n : ℕ) (a : fin n → ℂ) (h : ∀ i, complex.norm (a i) = 1) :
  max_sum_squares_sides_diagonals n a = n^2 :=
sorry

end max_sum_squares_l808_808173


namespace find_elf_6_nuts_l808_808211

-- Define the problem
def elves_nuts_answers : ℕ → ℕ
| 0 := 110
| 1 := 120
| 2 := 130
| 3 := 140
| 4 := 150
| 5 := 160
| 6 := 170
| 7 := 180
| 8 := 190
| 9 := 200
| _ := 0  -- Default case, not used

noncomputable def sum_even_positioned_elves (a_2 a_4 a_6 a_8 a_10 : ℕ) : ℕ :=
  a_2 + a_4 + a_6 + a_8 + a_10

axiom even_positioned_elves_sum : sum_even_positioned_elves 130 190 40 60 55 = 375

-- Define the main proof statement
theorem find_elf_6_nuts (a_2 a_4 a_6 a_8 a_10 : ℕ) 
  (h_sum_even: sum_even_positioned_elves a_2 a_4 a_6 a_8 a_10 = 375)
  (h_pair_24 : a_2 + a_4 = 130)
  (h_pair_810 : a_8 + a_10 = 190) :
  a_6 = 55 :=
by
  -- Skipping the proof process
  sorry

end find_elf_6_nuts_l808_808211


namespace non_overlapping_sets_l808_808863

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808863


namespace solve_logarithmic_eq_l808_808728

theorem solve_logarithmic_eq {x : ℝ} 
  (h : 7.72 * log (sqrt 3) x * sqrt (log (sqrt 3) 3 - log x 9) + 4 = 0) : 
  x = 1 / 3 :=
sorry

end solve_logarithmic_eq_l808_808728


namespace part1_monotonic_intervals_part2_extreme_values_l808_808094

noncomputable def f (x a : ℝ) := x + a * Real.exp x

theorem part1_monotonic_intervals (a : ℝ) (h : a = -1) :
  StrictMonoOn (λ x, x + a * Real.exp x) set.Iio 0 ∧ StrictAntiOn (λ x, x + a * Real.exp x) set.Ioi 0 :=
begin
    split;
    sorry
end

theorem part2_extreme_values (a : ℝ) :
  (a ≥ 0 → ∀ x y : ℝ, (x < y → f x a < f y a)) ∧
  (a < 0 → ∃ x : ℝ, x = Real.log (-1 / a) ∧ f x a = Real.log (-1 / a) - 1) :=
begin
  split;
  sorry
end

end part1_monotonic_intervals_part2_extreme_values_l808_808094


namespace line_equation_through_point_and_angle_l808_808644

noncomputable def sqrt3 : ℝ := Real.sqrt 3

theorem line_equation_through_point_and_angle :
  ∀ (x y : ℝ), let k := Real.tan (Real.pi / 3) in
    let A := (sqrt3, 1) in A = (sqrt3, 1) → x = sqrt3 ∧ y = 1 → 
    let line_eq := k * x - y - 2 in 
      line_eq = 0 :=
by
  sorry

end line_equation_through_point_and_angle_l808_808644


namespace average_speed_uphill_l808_808196

theorem average_speed_uphill (d : ℝ) (v : ℝ) :
  (2 * d) / ((d / v) + (d / 100)) = 9.523809523809524 → v = 5 :=
by
  intro h1
  sorry

end average_speed_uphill_l808_808196


namespace ratio_is_integer_l808_808161

noncomputable def a (r s : ℕ) : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := r * a (r s) (n+1) + s * a (r s) n

noncomputable def f (r s : ℕ) (n : ℕ) : ℕ :=
  ∏ i in (Finset.range n).filter (λ j, j > 0), a r s i

theorem ratio_is_integer (r s n k : ℕ) (hr : 0 < r) (hs : 0 < s) (hkn : 0 < k) (hkn' : k < n) :
  (f r s n) / ((f r s k) * (f r s (n - k))) ∈ ℤ :=
by {
  -- proof omitted
  sorry
}

end ratio_is_integer_l808_808161


namespace range_of_a_l808_808945

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≥ -1 → x2 ≥ -1 → x1 ≤ x2 → 
  (log (1/2) (3 * x2^2 - a * x2 + 5) ≤ log (1/2) (3 * x1^2 - a * x1 + 5))) ↔ (a ∈ Ioc (-8:ℝ) (-6:ℝ)) :=
sorry

end range_of_a_l808_808945


namespace taylor_probability_l808_808833

open Nat Real

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem taylor_probability :
  (binomial_probability 5 2 (3/5) = 144 / 625) :=
by
  sorry

end taylor_probability_l808_808833


namespace arithmetic_sequence_common_difference_l808_808139

variable {α : Type*} [AddGroup α] [AddCommGroup α] [AffineSpace α α]

def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def common_difference {α : Type*} [AddGroup α] [AddCommGroup α] (a : ℕ → α) (n : ℕ) :=
  a (n + 1) - a n

theorem arithmetic_sequence_common_difference {a : ℕ → ℤ} (h1 : a 5 = 3) (h2 : a 6 = -2) :
  common_difference a 5 = -5 :=
by
  unfold common_difference
  rw [h2, h1]
  norm_num
  sorry

end arithmetic_sequence_common_difference_l808_808139


namespace orthic_triangle_perimeter_equal_l808_808619

theorem orthic_triangle_perimeter_equal (ABC : Triangle) (h_c : ℝ) (γ : ℝ) 
  (h_acute : ABC.acute) 
  (h_altitudes_eq : ABC.altitude = h_c) 
  (h_angles_eq : ABC.angle = γ) :
  ABC.orthic_triangle.perimeter = 2 * h_c * Real.cos γ :=
by
  sorry

end orthic_triangle_perimeter_equal_l808_808619


namespace more_people_attended_l808_808682

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l808_808682


namespace workshop_assistants_and_average_salary_l808_808998

theorem workshop_assistants_and_average_salary
  (workers_total : ℕ)
  (tech_number : ℕ) (tech_avg_salary : ℕ)
  (mgr_number : ℕ) (mgr_avg_salary : ℕ)
  (ast_avg_salary : ℕ) (total_avg_salary : ℕ)
  (total_number : ℕ) (tech_total_salary : ℕ)
  (mgr_total_salary : ℕ) (combined_mgr_tech_avg_salary : ℕ) :
  workers_total = 20 →
  tech_number = 7 →
  tech_avg_salary = 12000 →
  mgr_number = 5 →
  mgr_avg_salary = 15000 →
  ast_avg_salary = 6000 →
  total_avg_salary = 8000 →
  tech_total_salary = tech_number * tech_avg_salary →
  mgr_total_salary = mgr_number * mgr_avg_salary →
  combined_mgr_tech_avg_salary = (tech_total_salary + mgr_total_salary) / (tech_number + mgr_number) →
  ∃ (A : ℕ), A = workers_total - tech_number - mgr_number ∧
  combined_mgr_tech_avg_salary = 13250 :=
by
  intros h_total hw_t hn_t ha_t hm_n h_terms hm_t hc_tr hns_htb h_cb_mb_st hw_ent hw_avg_dt
  use workers_total - tech_number - mgr_number
  split
  { sorry }
  { sorry }

end workshop_assistants_and_average_salary_l808_808998


namespace dog_roaming_area_l808_808322

def leash_length : ℝ := 10
def radius_of_pillar : ℝ := 2
def effective_radius : ℝ := leash_length + radius_of_pillar

theorem dog_roaming_area : π * effective_radius^2 = 144 * π := by
  have h : effective_radius = 12 := by 
    unfold effective_radius leash_length radius_of_pillar
    norm_num
  rw [h]
  ring
  norm_num
  sorry

end dog_roaming_area_l808_808322


namespace minimum_common_perimeter_l808_808265

theorem minimum_common_perimeter :
  ∃ (a b c : ℕ), 
  let p := 2 * a + 10 * c in
  (a > b) ∧ 
  (b + 4c = a + 5c) ∧
  (5 * (a^2 - (5 * c)^2).sqrt = 4 * (b^2 - (4 * c)^2).sqrt) ∧
  p = 1180 :=
sorry

end minimum_common_perimeter_l808_808265


namespace choose_non_overlapping_sets_for_any_n_l808_808891

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808891


namespace heartbeats_per_minute_l808_808928

theorem heartbeats_per_minute (t : ℝ) :
  let f (t : ℝ) := 24 * Real.sin (160 * Real.pi * t) + 110 in
  160 * Real.pi > 0 → 
  (1 / (2 * Real.pi / (160 * Real.pi))) = 80 :=
by 
  intro h
  sorry

end heartbeats_per_minute_l808_808928


namespace more_people_attended_l808_808681

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l808_808681


namespace units_digit_of_7_pow_6_pow_5_l808_808429

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808429


namespace farmer_total_acres_l808_808758

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l808_808758


namespace distinct_real_roots_a1_l808_808041

theorem distinct_real_roots_a1 {x : ℝ} :
  ∀ a : ℝ, a = 1 →
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + (1 - a) * x1 - 1 = 0) ∧ (a * x2^2 + (1 - a) * x2 - 1 = 0) :=
by sorry

end distinct_real_roots_a1_l808_808041


namespace zachary_pushups_l808_808011

theorem zachary_pushups (d z : ℕ) (h1 : d = z + 30) (h2 : d = 37) : z = 7 := by
  sorry

end zachary_pushups_l808_808011


namespace monotonic_intervals_l808_808859

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem monotonic_intervals (a : ℝ) :
  (a = 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > 0 → f a x > f a 1))) ∧
  (a > 0 → (∀ x : ℝ, (x < -2 / a → f a x < f a (-2 / a - 1)) ∧ (x > 0 → f a x > f a 1) ∧ 
                  ((-2 / a) < x ∧ x < 0 → f a x < f a (-2 / a + 1)))) ∧
  (a < 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > -2 / a → f a x < f a (-2 / a - 1)) ∧
                  (0 < x ∧ x < -2 / a → f a x > f a (-2 / a + 1))))
:= sorry

end monotonic_intervals_l808_808859


namespace max_avg_score_l808_808538

-- Define the score function for given time n if 2n is an integer
def score (n : ℝ) : ℝ :=
  if (2 * n) % 1 = 0 then 100 * (1 - 4^(-n)) else 0

-- Define the maximum possible average score as a Lean theorem
theorem max_avg_score :
  ∀ (t1 t2 t3 t4 : ℝ),
    (t1 + t2 + t3 + t4 = 4) →
    (2 * t1) % 1 = 0 →
    (2 * t2) % 1 = 0 →
    (2 * t3) % 1 = 0 →
    (2 * t4) % 1 = 0 →
    (score t1 + score t2 + score t3 + score t4) / 4 = 75 :=
by
  -- Proof would be filled in here
  sorry

end max_avg_score_l808_808538


namespace eccentricity_range_hyperbola_l808_808949

variable {a b x y : ℝ}
variable {e : ℝ}
variable {P F1 F2 O : ℝ}

def is_hyperbola (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def point_on_hyperbola_condition (P F1 F2 : ℝ) : Prop :=
  4 * abs (P - F1 + (P - F2)) ≥ 3 * abs (F1 - F2)

def focal_distance (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2)

def eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem eccentricity_range_hyperbola (a b x y : ℝ) (h1 : is_hyperbola a b x y) (h2 : point_on_hyperbola_condition P F1 F2) : 
  1 < e ∧ e ≤ (4/3) :=
  sorry

end eccentricity_range_hyperbola_l808_808949


namespace original_sets_exist_l808_808901

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808901


namespace units_digit_7_pow_6_pow_5_l808_808407

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l808_808407


namespace percentage_increase_l808_808659

noncomputable def price_increase (d new_price : ℝ) : ℝ :=
  ((new_price - d) / d) * 100

theorem percentage_increase 
  (d new_price : ℝ)
  (h1 : 2 * d = 585)
  (h2 : new_price = 351) :
  price_increase d new_price = 20 :=
by
  sorry

end percentage_increase_l808_808659


namespace remainder_is_correct_l808_808369

noncomputable def remainder (P Q : Polynomial ℤ) : Polynomial ℤ :=
  (Polynomial.modByMonic P (Q.monic))

def P : Polynomial ℤ := (X^5 - 1) * (X^3 - 1)
def Q : Polynomial ℤ := X^3 + X^2 + 1

theorem remainder_is_correct : remainder P Q = -2 * X^2 + X + 1 := by
  sorry

end remainder_is_correct_l808_808369


namespace range_of_a_l808_808453

noncomputable def a_range (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧
  ¬(∀ (x : ℝ), x > 0 → (log a (x+1) < log a ((x+1)+1))) ∧
  ¬((2*a - 3)^2 - 4 > 0)

theorem range_of_a (a : ℝ) : a_range a → a ∈ set.Ioc 1 (5/2) :=
by sorry

end range_of_a_l808_808453


namespace parabola_condition_tangent_line_condition1_tangent_line_condition2_l808_808950

noncomputable def parabola (m c x : ℝ) := m * x ^ 2 - (1 - 4 * m) * x + c

theorem parabola_condition (a c : ℝ) (h1 : parabola m c 0 = -1) (h2 : parabola m c 1 = a) (h3 : parabola m c (-1) = a) :
  ∃ (m : ℝ), parabola m (-1) x = 1 / 4 * x ^ 2 - 1 :=
begin
  sorry
end

noncomputable def tangent_line (k b x : ℝ) := k * x + b

theorem tangent_line_condition1 (k : ℝ) (h1 : tangent_line k (-3) 0 = -3)
  (h2 : ∃ (x : ℝ), tangent_line k (-3) x = parabola (1/4) (-1) x) :
  (k = sqrt 2 ∨ k = -sqrt 2) ∧
  (tangent_line (sqrt 2) (-3) = λ x, sqrt 2 * x - 3) ∧
  (tangent_line (-sqrt 2) (-3) = λ x, -sqrt 2 * x - 3) :=
begin
  sorry
end

theorem tangent_line_condition2 (f : ℝ) (k : ℝ)
  (h1 : tangent_line k (-k^2 - 1) (k^2 + 3) / k = 2)
  (h2 : tangent_line k (-k^2 - 1) (k^2 - 3) / k = -4)
  (h3 : (f - 2 + ((k^2 + 3) / k))^2 - (f - (-4) + ((k^2 - 3) / k))^2 = -4) :
  f = -1/2 :=
begin
  sorry
end

end parabola_condition_tangent_line_condition1_tangent_line_condition2_l808_808950


namespace find_a_min_value_slope_intersection_l808_808944

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + a * x^2 - 3 * x

theorem find_a (h : ∀ x, Deriv g x = 0 → Tangent.Line (g, (1, g 1)) x x-axis.parallel) : ∃ a, a = 1 :=
  by
    sorry

theorem min_value : ∃ x, ∀ y, g x ≤ g y :=
  by
    sorry

theorem slope_intersection (k : ℝ) (x1 x2 : ℝ) (hx1 : x1 < x2) (hx2 : f x1 = y_1) (hx3 : f x2 = y_2) : ∀ k, 1 / x2 < k ∧ k < 1 / x1 :=
  by
    sorry

end find_a_min_value_slope_intersection_l808_808944


namespace pass_rate_eq_l808_808655

theorem pass_rate_eq (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : (1 - a) * (1 - b) = ab - a - b + 1 :=
by
  sorry

end pass_rate_eq_l808_808655


namespace hypotenuse_length_l808_808355

-- Define the triangle with given lengths and right angle at C
structure Triangle :=
(a b : ℝ) (hypotenuse : ℝ)
(right_angle_at_C : a^2 + b^2 = hypotenuse^2)

-- Define the trisection points over the hypotenuse
def D (a b : ℝ) : ℝ × ℝ := (2/3 * a, 1/3 * b)
def E (a b : ℝ) : ℝ × ℝ := (1/3 * a, 2/3 * b)

-- Define the distances CD and CE given as sin(alpha) and cos(alpha) respectively
def dist (x1 y1 x2 y2 : ℝ) := sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Problem statement
theorem hypotenuse_length (a b : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (sin_alpha : dist a 0 (2 / 3 * a) (1 / 3 * b) = sin α)
  (cos_alpha : dist a 0 (1 / 3 * a) (2 / 3 * b) = cos α)
  : sqrt (a^2 + b^2) = 3 / sqrt 5 :=
sorry

end hypotenuse_length_l808_808355


namespace product_of_valid_c_l808_808387

def valid_c (c : ℤ) : Prop :=
  24 * 24 - 40 * c > 0 ∧ c > 0

theorem product_of_valid_c : (∏ c in finset.filter valid_c (finset.range 15), c) = 87178291200 := by
  sorry

end product_of_valid_c_l808_808387


namespace abs_diff_eq_l808_808669

-- Define the conditions
variables (x y : ℝ)
axiom h1 : x + y = 30
axiom h2 : x * y = 162

-- Define the problem to prove
theorem abs_diff_eq : |x - y| = 6 * Real.sqrt 7 :=
by sorry

end abs_diff_eq_l808_808669


namespace evaluate_27_pow_x_plus_1_l808_808974

-- Define the variables and the condition
variable (x : ℝ)
variable (h : 3^(2 * x) = 17)

-- State the theorem
theorem evaluate_27_pow_x_plus_1 : 27^(x + 1) = 27 * 17^(3/2) :=
by
  sorry

end evaluate_27_pow_x_plus_1_l808_808974


namespace sufficient_but_not_necessary_l808_808528

theorem sufficient_but_not_necessary (a : ℝ) : ((a = 2) → ((a - 1) * (a - 2) = 0)) ∧ (¬(((a - 1) * (a - 2) = 0) → (a = 2))) := 
by 
sorry

end sufficient_but_not_necessary_l808_808528


namespace circle_equation_l808_808029

theorem circle_equation {x y : ℝ} :
  (∃ c : ℝ, x^2 + y^2 - x + y - 2 + c * (x^2 + y^2 - 5) = 0) ∧
  (∃ h k : ℝ, h = (1 - (-3/2)) / (1 + (-3/2)) ∧ k = (5 + (-3/2)) / (1 + (-3/2)) ∧
               3 * h + 4 * k - 1 = 0 ∧
               x^2 + y^2 + 2 * x - 2 * y - 11 = 0) :=
begin
  sorry
end

end circle_equation_l808_808029


namespace positive_difference_x_intersects_l808_808501

structure Line where
  slope : ℝ
  y_intercept : ℝ

def lineL : Line := {
  slope := -2,
  y_intercept := 8
}

def lineM : Line := {
  slope := -2 / 3,
  y_intercept := 6
}

def x_intersect (l : Line) (y : ℝ) : ℝ :=
  (y - l.y_intercept) / l.slope

theorem positive_difference_x_intersects : 
  |x_intersect lineL 20 - x_intersect lineM 20| = 15 :=
by
  sorry

end positive_difference_x_intersects_l808_808501


namespace solve_for_x_l808_808042

theorem solve_for_x (x : ℝ) (h : (4 + x) / (6 + x) = (1 + x) / (2 + x)) : x = 2 :=
sorry

end solve_for_x_l808_808042


namespace tan_theta_plus_pi_over_six_l808_808073

theorem tan_theta_plus_pi_over_six 
  (θ : ℝ) 
  (h1 : sqrt 2 * sin (θ - π / 4) * cos (π + θ) = cos (2 * θ))
  (h2 : sin θ ≠ 0) : tan (θ + π / 6) = 2 + sqrt 3 :=
by
  -- Proof goes here
  sorry

end tan_theta_plus_pi_over_six_l808_808073


namespace g_five_l808_808583

def g (x : ℝ) : ℝ := 4 * x + 2

theorem g_five : g 5 = 22 := by
  sorry

end g_five_l808_808583


namespace pedal_triangles_common_circumcircle_l808_808715

-- Define the isotomic conjugate relationship
structure Triangle (α : Type) :=
(A B C : α)

variables {α : Type} [MetricSpace α] [NormedAddCommGroup α] [NormedSpace ℝ α]

def IsIsotomicConjugate (T : Triangle α) (P1 P2 : α) : Prop :=
sorry -- Definition needs to be formalized.

-- Perpendicular or given angle construction
def PerpendicularToSides (T : Triangle α) (P : α) : Prop :=
sorry -- Definition of constructing perpendiculars from P to sides of T.

def GivenAngleLines (T : Triangle α) (P : α) (θ : ℝ) : Prop :=
sorry -- Definition of constructing lines from P at a given angle to sides of T.

-- Defining a pedal triangle and common circumcircle with required conditions
theorem pedal_triangles_common_circumcircle (T : Triangle α) (P1 P2 : α) (θ : ℝ)
  (h1 : IsIsotomicConjugate T P1 P2)
  (h2 : PerpendicularToSides T P1 ∨ GivenAngleLines T P1 θ)
  (h3 : PerpendicularToSides T P2 ∨ GivenAngleLines T P2 θ) :
  ∃ (O : α), IsMidPoint O P1 P2 ∧ HasCirumcircle (PedalTriangle T P1) (O) ∧ HasCirumcircle (PedalTriangle T P2) (O) :=
sorry

-- Additional necessary definitions
def IsMidPoint (O P1 P2 : α) : Prop :=
sorry -- Definition of midpoint O of P1, P2.

def PedalTriangle (T : Triangle α) (P : α) : Triangle α :=
sorry -- Definition of the pedal triangle of P with respect to T.

def HasCirumcircle (T : Triangle α) (O : α) : Prop :=
sorry -- Definition of T having a circumcircle centered at O.


end pedal_triangles_common_circumcircle_l808_808715


namespace power_mod_8_l808_808283

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end power_mod_8_l808_808283


namespace probability_rain_once_l808_808238

theorem probability_rain_once (p : ℚ) 
  (h₁ : p = 1 / 2) 
  (h₂ : 1 - p = 1 / 2) 
  (h₃ : (1 - p) ^ 4 = 1 / 16) 
  : 1 - (1 - p) ^ 4 = 15 / 16 :=
by
  sorry

end probability_rain_once_l808_808238


namespace units_digit_pow_7_6_5_l808_808417

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808417


namespace probability_no_shaded_rectangle_l808_808308

-- Definitions based on the problem conditions
def num_columns := 1001
def num_vertical_segments := num_columns + 1
def total_rectangles := (num_vertical_segments * (num_vertical_segments - 1)) / 2

def shaded_positions : set (ℕ × ℕ) := {(1, 1), (1, 501), (1, 1001)}

def count_shaded_rectangles (n : ℕ) : ℕ :=
  let m := num_columns in
    2 * m + ((m / 2) * (m / 2))

-- The main theorem with the provided answer
theorem probability_no_shaded_rectangle :
  (total_rectangles - count_shaded_rectangles num_columns) / total_rectangles = 249499 / 501501 :=
by
  sorry

end probability_no_shaded_rectangle_l808_808308


namespace length_of_GH_l808_808700

def EF := 180
def IJ := 120

theorem length_of_GH (EF_parallel_GH : true) (GH_parallel_IJ : true) : GH = 72 := 
sorry

end length_of_GH_l808_808700


namespace range_f_2_sqrt_2_l808_808091

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2*x else log a x - 1/2

theorem range_f_2_sqrt_2 (a : ℝ) (h : 0 < a ∧ a ≤ 1/4) : 
  ∃ y ∈ set.Ioo (-5/4 : ℝ) (-1/2), f (2 * real.sqrt 2) a = y :=
by
  sorry

end range_f_2_sqrt_2_l808_808091


namespace trapezoid_area_l808_808221

theorem trapezoid_area (d1 d2 b_mid : ℝ) (h1 : d1 = 3) (h2 : d2 = 5) (h3 : b_mid = 2) : 
  ∃ (area : ℝ), area = 6 :=
by
  use 6
  sorry

end trapezoid_area_l808_808221


namespace axes_are_not_vectors_l808_808559

def is_vector (v : Type) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ), magnitude > 0

def x_axis : Type := ℝ
def y_axis : Type := ℝ

-- The Cartesian x-axis and y-axis are not vectors
theorem axes_are_not_vectors : ¬ (is_vector x_axis) ∧ ¬ (is_vector y_axis) :=
by
  sorry

end axes_are_not_vectors_l808_808559


namespace unique_intersection_points_l808_808303

noncomputable def line_segments_intersection_points (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : ℕ :=
  Nat.choose m 2 * Nat.choose n 2

theorem unique_intersection_points (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  line_segments_intersection_points m n hm hn = Nat.choose m 2 * Nat.choose n 2 := by
  -- Start proof (actual proof is not required as per instructions)
  sorry

end unique_intersection_points_l808_808303


namespace factorize_1_factorize_2_l808_808835

-- Define the variables involved
variables (a x y : ℝ)

-- Problem (1): 18a^2 - 32 = 2 * (3a + 4) * (3a - 4)
theorem factorize_1 (a : ℝ) : 
  18 * a^2 - 32 = 2 * (3 * a + 4) * (3 * a - 4) :=
sorry

-- Problem (2): y - 6xy + 9x^2y = y * (1 - 3x) ^ 2
theorem factorize_2 (x y : ℝ) : 
  y - 6 * x * y + 9 * x^2 * y = y * (1 - 3 * x) ^ 2 :=
sorry

end factorize_1_factorize_2_l808_808835


namespace average_of_seven_consecutive_l808_808198

theorem average_of_seven_consecutive (
  a : ℤ 
  ) (c : ℤ) 
  (h1 : c = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 7 := 
by 
  sorry

end average_of_seven_consecutive_l808_808198


namespace units_digit_pow_7_6_5_l808_808416

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l808_808416


namespace seojun_pizza_l808_808631

/-- If Seojun misunderstood the teacher's words and gave away 7/3 of a pizza when he was supposed to bring it from the next class, and he ended up with 3/2 pizza left, then the total amount of pizza Seojun should have if he did the errand correctly is 37/6. -/
theorem seojun_pizza (x : ℚ) (h1 : x = 23/6) (hx : x - 7/3 = 3/2) : x + 7/3 = 37/6 :=
by
  rw [h1]
  norm_num
  sorry

end seojun_pizza_l808_808631


namespace find_m_value_l808_808126

def power_function_increasing (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2*m - 1 > 0)

theorem find_m_value (m : ℝ) (h : power_function_increasing m) : m = -1 :=
  sorry

end find_m_value_l808_808126


namespace non_overlapping_sets_l808_808861

theorem non_overlapping_sets (n : ℕ) : 
  ∃ sets : fin (n+1) → fin (n-1) → fin n × fin n, 
    (∀ i j, i ≠ j → sets i ≠ sets j) ∧ -- non-overlapping sets
    (∀ i, function.injective (λ k, (sets i k).fst) ∧ function.injective (λ k, (sets i k).snd)) := -- no two cells in the same row or column
  sorry

end non_overlapping_sets_l808_808861


namespace f_13_eq_223_l808_808117

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_eq_223 : f 13 = 223 :=
by
  sorry

end f_13_eq_223_l808_808117


namespace decrease_by_150_percent_l808_808288

theorem decrease_by_150_percent (x : ℝ) (h : x = 80) : x - 1.5 * x = -40 :=
by
  sorry

end decrease_by_150_percent_l808_808288


namespace proposition_one_proposition_two_proposition_three_proposition_four_true_propositions_l808_808645

-- Define the propositions as separate hypotheses
theorem proposition_one : ∀ (Q : Type) [Quadrilateral Q], 
  bisects_equal_diagonals Q → rect Q := sorry

theorem proposition_two : ∀ (Q : Type) [Quadrilateral Q],
  perpendicular_diagonals Q → rhombus Q := sorry

theorem proposition_three : ∀ (Q : Type) [Quadrilateral Q],
  all_sides_equal Q → square Q := sorry

theorem proposition_four : ∀ (Q : Type) [Quadrilateral Q],
  all_sides_equal Q → rhombus Q := sorry

-- The main theorem to prove which propositions are true
theorem true_propositions :
  (proposition_one ∧ proposition_four) ∧ 
  (¬proposition_two ∧ ¬proposition_three) := sorry

end proposition_one_proposition_two_proposition_three_proposition_four_true_propositions_l808_808645


namespace count_divisibles_l808_808518

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l808_808518


namespace setD_is_pythagorean_triple_l808_808793

def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the sets
def setA := (1, 2, 3)
def setB := (4, 5, 6)
def setC := (6, 8, 9)
def setD := (7, 24, 25)

-- Prove that Set D is a Pythagorean triple
theorem setD_is_pythagorean_triple : is_pythagorean_triple 7 24 25 :=
by
  show 7^2 + 24^2 = 25^2,
  calc
  7^2 + 24^2 = 49 + 576 := by norm_num
  ... = 625 := by norm_num
  ... = 25^2 := by norm_num

end setD_is_pythagorean_triple_l808_808793


namespace max_ab_value_l808_808948

theorem max_ab_value
  (f g : ℝ → ℝ)
  (P : ℝ × ℝ)
  (l1 l2 : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = x^2 - 2*x + 2)
  (h2 : ∀ x, g x = -x^2 + a*x + b - 1/2)
  (hP : P.1 ≠ 0)
  (hf_P : ∀ x, l1 P.1 = (2 * P.1 - 2) * (x - P.1) + f P.1)
  (hg_P : ∀ x, l2 P.1 = (-2 * P.1 + a) * (x - P.1) + g P.1)
  (h_perpendicular : (2 * P.1 - 2) * (-2 * P.1 + a) = -1)
  : ab = 9/4 :=
begin
  sorry
end

end max_ab_value_l808_808948


namespace stickers_per_sheet_l808_808334

/-
  Conditions:
    Xia started with 150 stickers.
    Xia shared 100 stickers with her friends.
    Xia had 5 sheets of stickers left.
  Question:
    How many stickers were on each sheet?
  
  Proof: Prove that (150 - 100) / 5 = 10.
-/

theorem stickers_per_sheet (initial_stickers shared_stickers sheets : ℕ) 
  (h1 : initial_stickers = 150) 
  (h2 : shared_stickers = 100) 
  (h3 : sheets = 5) : 
  (initial_stickers - shared_stickers) / sheets = 10 := 
by 
  rw [h1, h2, h3] 
  exact dec_trivial

end stickers_per_sheet_l808_808334


namespace four_lines_range_l808_808391

def point : Type := ℝ × ℝ

def distance (p q : point) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def circle (center : point) (radius : ℝ) : set (point) :=
  { p | distance p center = radius }

noncomputable def d_range (A B : point) : set ℝ :=
  { d | 0 < d ∧ d < 4 }

theorem four_lines_range (A B : point) (d : ℝ) (hA : A = (1,2)) (hB : B = (5,5)) :
  (0 < d ∧ d < 4) :=
by sorry

end four_lines_range_l808_808391


namespace football_game_attendance_l808_808683

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l808_808683


namespace more_expensive_candy_price_l808_808318

-- Definition of conditions
def price_per_pound_cheap := 2.00
def price_per_pound_mixture := 2.20
def total_pounds := 80
def pounds_cheap := 64

-- Calculation for the total value that the mixture should amount to
def total_value := total_pounds * price_per_pound_mixture

-- Calculation for the total value of the 2-dollar candy used
def value_cheap := pounds_cheap * price_per_pound_cheap

-- Remaining value should be made up by the more expensive candy
def value_expensive := total_value - value_cheap

-- Remaining pounds needed
def pounds_expensive := total_pounds - pounds_cheap

-- Calculate the price per pound of the more expensive candy
def price_per_pound_expensive := value_expensive / pounds_expensive

theorem more_expensive_candy_price :
  price_per_pound_expensive = 3.00 := by
  sorry

end more_expensive_candy_price_l808_808318


namespace compute_value_l808_808587

   noncomputable def a : ℚ := 4 / 7
   noncomputable def b : ℚ := 5 / 3

   theorem compute_value : 2 * a^(-3) * b^2 = 17150 / 576 :=
   by
     sorry
   
end compute_value_l808_808587


namespace count_words_with_consonant_l808_808507

def letters := {'A', 'B', 'C', 'D', 'E'}
def vowels := {'A', 'E'}
def consonants := {'B', 'C', 'D'}

theorem count_words_with_consonant :
  let total_words := 5^5
  let vowel_only_words := 2^5
  total_words - vowel_only_words = 3093 :=
by
  sorry

end count_words_with_consonant_l808_808507


namespace combined_capacity_is_40_l808_808612

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l808_808612


namespace units_digit_7_power_l808_808394

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808394


namespace binary_div_four_remainder_l808_808375

theorem binary_div_four_remainder (n : ℕ) (h : n = 0b111001001101) : n % 4 = 1 := 
sorry

end binary_div_four_remainder_l808_808375


namespace travis_payment_l808_808688

def total_payment (glass_bowls ceramic_bowls lost_glass broken_glass lost_ceramic broken_ceramic : ℕ) : ℤ :=
  let base_fee := 100
  let glass_bowls_safe := glass_bowls - lost_glass - broken_glass
  let ceramic_bowls_safe := ceramic_bowls - lost_ceramic - broken_ceramic
  let payment_safe := (glass_bowls_safe + ceramic_bowls_safe) * 3
  let charges := (lost_glass * 6 + broken_glass * 5 + lost_ceramic * 3 + broken_ceramic * 4)
  let additional_fee := glass_bowls * 0.50 + ceramic_bowls * 0.25
  base_fee + payment_safe - charges + additional_fee

theorem travis_payment :
  total_payment 375 263 9 10 3 5 = 2053.25 :=
by {
  -- conditions
  have glass_bowls := 375,
  have ceramic_bowls := 263,
  have lost_glass := 9,
  have broken_glass := 10,
  have lost_ceramic := 3,
  have broken_ceramic := 5,
  -- verifying total payment
  calc total_payment glass_bowls ceramic_bowls lost_glass broken_glass lost_ceramic broken_ceramic
      = 2053.25 : by simp
  }

end travis_payment_l808_808688


namespace greatest_integer_gcd_24_eq_4_l808_808273

theorem greatest_integer_gcd_24_eq_4 : ∃ n < 200, n % 4 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0 ∧ n = 196 :=
begin
  sorry
end

end greatest_integer_gcd_24_eq_4_l808_808273


namespace percentage_decrease_increase_l808_808614

theorem percentage_decrease_increase (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 ↔ x = 50 :=
by
  sorry

end percentage_decrease_increase_l808_808614


namespace system_of_equations_solution_l808_808206

theorem system_of_equations_solution (x y z : ℝ) :
  (x = 6 + Real.sqrt 29 ∧ y = (5 - 2 * (6 + Real.sqrt 29)) / 3 ∧ z = (4 - (6 + Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) ∨
  (x = 6 - Real.sqrt 29 ∧ y = (5 - 2 * (6 - Real.sqrt 29)) / 3 ∧ z = (4 - (6 - Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) :=
sorry

end system_of_equations_solution_l808_808206


namespace general_term_find_max_n_l808_808914

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 8
  else sorry -- We provide the form a_n in proofs

noncomputable def S (n : ℕ) : ℕ :=
  if n = 1 then sequence 1
  else if n = 2 then (sequence 1 + sequence 2)
  else sorry -- Sum of the first n terms

noncomputable def recurrence_condition (n : ℕ) : Prop :=
  S (n + 1) + 4 * S (n - 1) = 5 * S n

theorem general_term (n : ℕ) (h_cond : n >= 2) : 
  sequence n = 2^(2*n - 1) := 
sorry

noncomputable def log_sequence_sum (n : ℕ) : ℕ :=
  (List.range n).sum (λ i, Int.log2 (sequence i + 1))

noncomputable def product_expression (n : ℕ) : ℚ :=
(List.range (n - 1)).prod (λ k, 1 - (1 : ℚ) / ((k + 1 : ℕ) ^ 2))

theorem find_max_n : 
  ∃ n : ℕ, (1 - 1 / (log_sequence_sum 2)) * (product_expression 3) * ... * (product_expression n) > 51 / 101 
  ∧ n = 100 :=
sorry

end general_term_find_max_n_l808_808914


namespace find_derivative_at_one_l808_808454

def f (x : ℝ) : ℝ := x^(1 / 3)

theorem find_derivative_at_one :
  (derivative f 1) = 1 / 3 :=
sorry

end find_derivative_at_one_l808_808454


namespace Q1_Q2_Q3_l808_808460

def A1 : Set ℕ := {1, 3}
def S1 : Set ℕ := {x | ∃ a b ∈ A1, x = a + b}
def T1 : Set ℕ := {x | ∃ a b ∈ A1, x = abs (a - b)}

theorem Q1 : S1 = {2, 4, 6} ∧ T1 = {0, 2} := sorry

variable {x1 x2 x3 x4 : ℕ} (hx : x1 < x2 ∧ x2 < x3 ∧ x3 < x4)

def A2 : Set ℕ := {x1, x2, x3, x4}
def T2 : Set ℕ := {x | ∃ a b ∈ A2, x = abs (a - b)}

theorem Q2 (hT : T2 = A2) : x1 + x4 = x2 + x3 := sorry

def A3 : Set ℕ := {x | x ∈ Finset.range 2024}
def S3 (A : Set ℕ) : Set ℕ := {x | ∃ a b ∈ A, x = a + b}
def T3 (A : Set ℕ) : Set ℕ := {x | ∃ a b ∈ A, x = abs (a - b)}

theorem Q3 (hST : ∀ A ⊆ A3, S3 A ∩ T3 A = ∅) : ∃ (A : Set ℕ), (A ⊆ A3 ∧ A.card = 1349) := sorry

end Q1_Q2_Q3_l808_808460


namespace trip_duration_correct_l808_808571

open Time

def clock_hand_together_time_between_10_and_11 : Time :=
  -- Assuming the function 'Time.of_seconds' converts given seconds since midnight to Time
  Time.of_seconds (10 * 3600 + 54 * 60 + 33)

def clock_hands_180_degrees_apart_time_between_4_and_5 : Time :=
  -- Assuming the function 'Time.of_seconds' converts given seconds since midnight to Time
  Time.of_seconds (16 * 3600 + 50 * 60)

def duration (start_time end_time : Time) : Time := 
  (end_time - start_time)

theorem trip_duration_correct : 
  duration clock_hand_together_time_between_10_and_11 clock_hands_180_degrees_apart_time_between_4_and_5 = Time.mk_time 5 55 :=
by 
  sorry

end trip_duration_correct_l808_808571


namespace median_of_first_twelve_natural_numbers_starting_from_3_l808_808279

theorem median_of_first_twelve_natural_numbers_starting_from_3 : 
  let numbers := [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] in
  (numbers.drop 5).head + (numbers.drop 6).head / (2 : ℝ) = 8.5 :=
sorry

end median_of_first_twelve_natural_numbers_starting_from_3_l808_808279


namespace number_description_l808_808110

theorem number_description :
  4 * 10000 + 3 * 1000 + 7 * 100 + 5 * 10 + 2 + 8 / 10 + 4 / 100 = 43752.84 :=
by
  sorry

end number_description_l808_808110


namespace find_percentage_l808_808742

-- conditions
def N : ℕ := 160
def expected_percentage : ℕ := 35

-- statement to prove
theorem find_percentage (P : ℕ) (h : P / 100 * N = 50 / 100 * N - 24) : P = expected_percentage :=
sorry

end find_percentage_l808_808742


namespace problem_part1_problem_part2_l808_808053

theorem problem_part1 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x * y = 1 :=
sorry

theorem problem_part2 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x^2 * y - 2 * x * y^2 = 3 :=
sorry

end problem_part1_problem_part2_l808_808053


namespace equal_semi_circles_radius_l808_808643

-- Define the segments and semicircles given in the problem as conditions.
def segment1 : ℝ := 12
def segment2 : ℝ := 22
def segment3 : ℝ := 22
def segment4 : ℝ := 16
def segment5 : ℝ := 22

def total_horizontal_path1 (r : ℝ) : ℝ := 2*r + segment1 + 2*r + segment1 + 2*r
def total_horizontal_path2 (r : ℝ) : ℝ := segment2 + 2*r + segment4 + 2*r + segment5

-- The theorem that proves the radius is 18.
theorem equal_semi_circles_radius : ∃ r : ℝ, total_horizontal_path1 r = total_horizontal_path2 r ∧ r = 18 := by
  use 18
  simp [total_horizontal_path1, total_horizontal_path2, segment1, segment2, segment3, segment4, segment5]
  sorry

end equal_semi_circles_radius_l808_808643


namespace equation_solution_20_solutions_l808_808169

theorem equation_solution_20_solutions (n : ℕ) (h_pos : 0 < n) 
  (h_solutions : ∃ s : Finset (ℕ × ℕ × ℕ), (∀ p ∈ s, let ⟨x, y, z⟩ := p in 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 4 * y + z = n) ∧ s.card = 20) :
  n = 21 ∨ n = 22 :=
sorry

end equation_solution_20_solutions_l808_808169


namespace units_digit_of_power_l808_808401

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l808_808401


namespace total_amount_paid_l808_808784

theorem total_amount_paid (days_worked1 : ℚ) (days_worked2 : ℚ) (days_worked3 : ℚ) (days_worked4 : ℚ) (pay_per_day : ℚ) :
  days_worked1 = 11/3 → days_worked2 = 2/3 → days_worked3 = 1/8 → days_worked4 = 3/4 →
  pay_per_day = 20 →
  (days_worked1 + days_worked2 + days_worked3 + days_worked4) * pay_per_day ≈ 104 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  calc
    (11/3 + 2/3 + 1/8 + 3/4) * 20 = (125/24) * 20 : by norm_num
                               ... ≈ 104 : sorry

end total_amount_paid_l808_808784


namespace greatest_distance_from_C_l808_808912

noncomputable def right_triangle_max_distance (x y : ℝ) : Prop :=
  let u := Real.sqrt (x*x + y*y)
  let v := Real.sqrt ((x-1)*(x-1) + y*y)
  let w := Real.sqrt (x*x + (y-1)*(y-1))
  (x*x + y*y - ((x-1)*(x-1) + y*y) + 2 * (x*x + (y-1)*(y-1)) = 1) →
  w ≤ Real.sqrt 5

theorem greatest_distance_from_C : right_triangle_max_distance x y :=
sorry

end greatest_distance_from_C_l808_808912


namespace combined_bus_capacity_l808_808607

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l808_808607


namespace cube_edge_length_l808_808746

theorem cube_edge_length (a : ℝ) (base_length : ℝ) (base_width : ℝ) (rise_height : ℝ) 
  (h_conditions : base_length = 20 ∧ base_width = 15 ∧ rise_height = 11.25 ∧ 
                  (base_length * base_width * rise_height) = a^3) : 
  a = 15 := 
by
  sorry

end cube_edge_length_l808_808746


namespace hyperbola_equation_l808_808488

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ) (m n : ℝ), 
  m * (-3)^2 - n * (2 * real.sqrt 7)^2 = 1 ∧
  m * (6 * real.sqrt 2)^2 - n * (-7)^2 = 1 ∧
  (x^2 / 4 - y^2 / 3 = -3) ∧
  ((6 * real.sqrt 2) ^ 2 * m - (-7) ^ 2 * n = 1) ∧
  ((2 ^ 2) / 9 - ((2 * real.sqrt 3) ^ 2) / 12 = 1) ∧
  (m = -1 / 75 ∧ n = -1 / 25)

theorem hyperbola_equation :
  problem_statement :=
by
  sorry

end hyperbola_equation_l808_808488


namespace bug_total_distance_l808_808737

theorem bug_total_distance : 
  let start := 3
  let first_point := 9
  let second_point := -4
  let distance_1 := abs (first_point - start)
  let distance_2 := abs (second_point - first_point)
  distance_1 + distance_2 = 19 := 
by
  sorry

end bug_total_distance_l808_808737


namespace find_coordinates_of_D_find_value_of_k_l808_808579

/-- Let A, B, C be points such that A = (1, 3), B = (2, -2), C = (4, -1).
If vector AB is equal to vector CD, then the coordinates of point D are (5, -6). -/
theorem find_coordinates_of_D : 
  ∃ D : (ℝ × ℝ), (∃ A B C : (ℝ × ℝ), A = (1,3) ∧ B = (2,-2) ∧ C = (4,-1) ∧ 
  vector.eq (B.1 - A.1, B.2 - A.2) (D.1 - C.1, D.2 - C.2)) ∧ D = (5, -6) :=
sorry

/-- Let vectors a and b be defined as a = vector AB and b = vector BC.
If k * a - b is parallel to a + 3 * b, then the real number k is -1/3. -/
theorem find_value_of_k : 
  ∃ (k : ℝ), 
  (let A := (1,3), B := (2,-2), C := (4,-1) in
  let a := (B.1 - A.1, B.2 - A.2), b := (C.1 - B.1, C.2 - B.2) in
  ∀ u v : (ℝ × ℝ), u = (k * a.1 - b.1, k * a.2 - b.2) → 
                  v = (a.1 + 3 * b.1, a.2 + 3 * b.2) →
                  ∃ λ : ℝ, u.1 = λ * v.1 ∧ u.2 = λ * v.2) ∧ k = -1/3 :=
sorry

end find_coordinates_of_D_find_value_of_k_l808_808579


namespace arithm_prog_diff_max_l808_808657

noncomputable def find_most_common_difference (a b c : Int) : Prop :=
  let d := a - b
  (b = a - d) ∧ (c = a - 2 * d) ∧
  (2 * a * 2 * a - 4 * 2 * a * c ≥ 0) ∧
  (2 * a * 2 * b - 4 * 2 * a * c ≥ 0) ∧
  (2 * b * 2 * b - 4 * 2 * b * c ≥ 0) ∧
  (2 * b * c - 4 * 2 * b * a ≥ 0) ∧
  (c * c - 4 * c * 2 * b ≥ 0) ∧
  ((2 * a * c - 4 * 2 * c * b) ≥ 0)

theorem arithm_prog_diff_max (a b c Dmax: Int) : 
  find_most_common_difference 4 (-1) (-6) ∧ Dmax = -5 :=
by 
  sorry

end arithm_prog_diff_max_l808_808657


namespace part1_increasing_decreasing_intervals_part2_inequality_range_a_l808_808490

def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

theorem part1_increasing_decreasing_intervals :
  (∀ x, x < -3 → ∀ y, f'(x) > f'(y)) ∧ (∀ x, x > -3 → ∀ y, f'(x) < f'(y)) := by
  sorry

theorem part2_inequality_range_a :
  (∀ x, x ≥ 0 → ∀ a, 0 ≤ a ∧ a ≤ 2 → (f x - Real.exp x) / (a * x + 1) ≥ 1) := by
  sorry

end part1_increasing_decreasing_intervals_part2_inequality_range_a_l808_808490


namespace line_AC_parallel_plane_D₁EF_sine_dihedral_angle_distance_line_to_plane_l808_808142

variables (A B C D A₁ B₁ C₁ D₁ E F O M N : Point)
variables (unit_cube : UnitCube AB CD A₁ B₁ C₁ D₁)
variables (E_midpoint : Midpoint E A B)
variables (F_midpoint : Midpoint F B C)

-- Proving that line AC is parallel to plane D₁EF
theorem line_AC_parallel_plane_D₁EF (AC : Line A C) (D₁EF : Plane D₁ E F) :
  Parallel AC D₁EF := by
  sorry

-- Finding the sine of the dihedral angle between planes D-EF and D₁-EF
theorem sine_dihedral_angle (DEF : Plane D E F) (D₁EF : Plane D₁ E F) :
  sin (dihedral_angle DEF D₁EF) = (2 * sqrt 34) / 17 := by
  sorry

-- Finding the distance from the line AC to the plane D₁EF
theorem distance_line_to_plane (AC : Line A C) (D₁EF : Plane D₁ E F) :
  distance AC D₁EF = (sqrt 17) / 17 := by
  sorry

end line_AC_parallel_plane_D₁EF_sine_dihedral_angle_distance_line_to_plane_l808_808142


namespace positive_integers_count_l808_808363

theorem positive_integers_count :
  {n : ℕ | (n + 1500) / 90 = Nat.floor (Real.sqrt (n + 1))}.to_finset.card = 3 :=
by
  sorry

end positive_integers_count_l808_808363


namespace sum_c_n_l808_808923

noncomputable def a (n : ℕ) : ℕ := 3 * n

noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c (n : ℕ) : ℕ := n * b n

noncomputable def S (n : ℕ) : ℕ := n * (3 + 3 * (n - 1)) / 2

theorem sum_c_n (n : ℕ) : (∑ k in Finset.range n, c (k + 1)) = (n - 1) * 2^n + 1 :=
by
  sorry

end sum_c_n_l808_808923


namespace range_of_a_l808_808441

theorem range_of_a (a : ℝ) : (∀ x > 1, x^2 ≥ a) ↔ (a ≤ 1) :=
by {
  sorry
}

end range_of_a_l808_808441


namespace football_game_attendance_l808_808685

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l808_808685


namespace consecutive_odd_numbers_square_difference_l808_808954

theorem consecutive_odd_numbers_square_difference (a b : ℤ) :
  (a - b = 2 ∨ b - a = 2) → (a^2 - b^2 = 2000) → (a = 501 ∧ b = 499 ∨ a = -501 ∧ b = -499) :=
by 
  intros h1 h2
  sorry

end consecutive_odd_numbers_square_difference_l808_808954


namespace sequence_eq_l808_808773

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 1 / 2 else
  (1 / n.to_real) * (Finset.sum (Finset.range (n - 2 + 1)) (λ i, sequence (i + 1)))

theorem sequence_eq (n : ℕ) (hne2 : 2 ≤ n) : 
  sequence n = Finset.sum (Finset.range (n - 1)) (λ i, (-1) ^ (i + 2) / ((i + 2)!.to_real)) :=
begin
  sorry
end

end sequence_eq_l808_808773


namespace matrix_sequence_product_l808_808346

open Matrix

theorem matrix_sequence_product :
  ∏ k in (Finset.range 50).map (Function.Embedding.mk (λ n, 2 + 4 * n) sorry) (λ k, !![1, k; 0, 1])
  = !![1, 5000; 0, 1] :=
by
  sorry

end matrix_sequence_product_l808_808346


namespace units_digit_of_7_pow_6_pow_5_l808_808435

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l808_808435


namespace min_length_AB_given_conditions_l808_808476

-- Definition of circle C
def circle (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 2

-- Line l
def line (x y : ℝ) := x - y - 3 = 0

-- Definition of perpendicularity (for example purposes, not strictly required by the prompt, but added for clarity)
def perp (a b c d : ℝ) : Prop := (a - b) * (c - d) = - (b - a) * (d - c)

-- Proving the minimum length of AB based on given conditions
theorem min_length_AB_given_conditions :
  ∀ A B : ℝ × ℝ, (∃ E F : ℝ × ℝ, 
  (∀ x y, circle x y → perp (fst E) (snd E) (fst F) (snd F) ∧ (x + y) / 2 = fst ((fst E + fst F) / 2, (snd E + snd F) / 2)) ∧
   line (fst A) (snd A) ∧ line (fst B) (snd B) ∧ 
   ∀ P : ℝ × ℝ, circle (fst P) (snd P) → (∡ (fst A) (snd A) (fst P) (snd P) (fst B) (snd B)) ≥ π / 2) →
  dist A B ≥ 4 * sqrt 2 + 2 := 
sorry

end min_length_AB_given_conditions_l808_808476


namespace more_people_attended_l808_808680

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l808_808680


namespace f_f_f_f_i_l808_808443

noncomputable def f : ℂ → ℂ :=
λ z, if z.im ≠ 0 then 2 * z^2 + 1 else -z^2 - 1

theorem f_f_f_f_i : f (f (f (f complex.I))) = -26 := 
by
  sorry

end f_f_f_f_i_l808_808443


namespace diana_paint_fraction_remain_l808_808023

theorem diana_paint_fraction_remain (paint_per_statue : ℚ) (statues : ℕ) :
  (paint_per_statue = (1 / 6)) → (statues = 3) → (statues * paint_per_statue = 1 / 2) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end diana_paint_fraction_remain_l808_808023


namespace minimum_common_perimeter_l808_808267

theorem minimum_common_perimeter :
  ∃ (a b c : ℕ), 
  let p := 2 * a + 10 * c in
  (a > b) ∧ 
  (b + 4c = a + 5c) ∧
  (5 * (a^2 - (5 * c)^2).sqrt = 4 * (b^2 - (4 * c)^2).sqrt) ∧
  p = 1180 :=
sorry

end minimum_common_perimeter_l808_808267


namespace gari_fare_probability_l808_808047

theorem gari_fare_probability :
  let total_coins := 1 + 2 + 6,
      total_ways := Nat.choose total_coins 4,
      unfavorable_outcomes := Nat.choose 6 4 in
  total_coins = 9 ∧
  total_ways = 126 ∧
  unfavorable_outcomes = 15 →
  (total_ways - unfavorable_outcomes).toRat / total_ways.toRat = 37 / 42 :=
by
  intros total_coins total_ways unfavorable_outcomes h
  sorry

end gari_fare_probability_l808_808047


namespace farmer_total_acres_l808_808748

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l808_808748


namespace find_p_q_r_sum_l808_808782

/-- Definitions for the given problem -/
def radius_tower : ℕ := 10
def rope_length : ℕ := 30
def unicorn_height : ℕ := 6
def distance_rope_ground : ℕ := 6

def p : ℕ := 120
def q : ℕ := 1156
def r : ℕ := 14

/-- The proof statement -/
theorem find_p_q_r_sum :
  let length_touching := (p - (q.sqrt)) / r in
  p + q + r = 1290 :=
by
  sorry

end find_p_q_r_sum_l808_808782


namespace number_of_multiples_in_range_l808_808512

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l808_808512


namespace sequence_has_no_primes_l808_808356

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53

theorem sequence_has_no_primes :
  ∀ (m : ℕ), 2 ≤ m → m ≤ 55 → ¬ Prime (Q + m) :=
by {
    intro m,
    intro h1,
    intro h2,
    sorry
}

end sequence_has_no_primes_l808_808356


namespace units_digit_7_power_l808_808395

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l808_808395


namespace max_gcd_of_consecutive_terms_seq_b_l808_808699

-- Define the sequence b_n
def sequence_b (n : ℕ) : ℕ := n.factorial + 3 * n

-- Define the gcd function for two terms in the sequence
def gcd_two_terms (n : ℕ) : ℕ := Nat.gcd (sequence_b n) (sequence_b (n + 1))

-- Define the condition of n being greater than or equal to 0
def n_ge_zero (n : ℕ) : Prop := n ≥ 0

-- The theorem statement
theorem max_gcd_of_consecutive_terms_seq_b : ∃ n : ℕ, n_ge_zero n ∧ gcd_two_terms n = 14 := 
sorry

end max_gcd_of_consecutive_terms_seq_b_l808_808699


namespace equivalence_of_sum_cubed_expression_l808_808852

theorem equivalence_of_sum_cubed_expression (a b : ℝ) 
  (h₁ : a + b = 5) (h₂ : a * b = -14) : a^3 + a^2 * b + a * b^2 + b^3 = 265 :=
sorry

end equivalence_of_sum_cubed_expression_l808_808852


namespace cube_geometry_problem_l808_808466

theorem cube_geometry_problem (A B C D A1 B1 C1 D1 M : Point) :
  -- Given: A unit cube and M is the midpoint of BB1
  is_unit_cube A B C D A1 B1 C1 D1 ∧ 
  midpoint M B B1 → 
  -- Prove: 
  -- 1. The angle between AB1 and DM is 45 degrees
  angle (line A B1) (line D M) = 45 ∧
  -- 2. The distance between AB1 and DM is 1/3
  distance (line A B1) (line D M) = 1/3 ∧
  -- 3. The common perpendicular divides DM and AB1 in the ratio 8:1
  ratio_divide (common_perpendicular (line A B1) (line D M)) DM AB1 = (8:1) :=
sorry

end cube_geometry_problem_l808_808466


namespace integral_cos_plus_one_l808_808374

theorem integral_cos_plus_one :
  ∫ x in - (Real.pi / 2).. (Real.pi / 2), (1 + Real.cos x) = Real.pi + 2 :=
by
  sorry

end integral_cos_plus_one_l808_808374


namespace vehicle_distribution_l808_808319

theorem vehicle_distribution :
  ∃ B T U : ℕ, 2 * B + 3 * T + U = 18 ∧ ∀ n : ℕ, n ≤ 18 → ∃ t : ℕ, ∃ (u : ℕ), 2 * (n - t) + u = 18 ∧ 2 * Nat.gcd t u + 3 * t + u = 18 ∧
  10 + 8 + 7 + 5 + 4 + 2 + 1 = 37 := by
  sorry

end vehicle_distribution_l808_808319


namespace bobby_additional_candy_l808_808804

variable (initial_candy additional_candy chocolate total_candy : ℕ)
variable (bobby_initial_candy : initial_candy = 38)
variable (bobby_ate_chocolate : chocolate = 16)
variable (bobby_more_candy : initial_candy + additional_candy = 58 + chocolate)

theorem bobby_additional_candy :
  additional_candy = 36 :=
by {
  sorry
}

end bobby_additional_candy_l808_808804


namespace vector_magnitude_problem_l808_808957

open Real -- To use real number functions and operations

-- Definition of the problem's conditions and the target statement
theorem vector_magnitude_problem : 
  ∀ (a b : ℝ × ℝ), a = (3, 0) → ∥b∥ = 2 → 
  (∃ θ : ℝ, θ = 2 * π / 3 ∧ cos θ = -1 / 2) →
  ∥(a.1 + 2 * b.1, a.2 + 2 * b.2)∥ = √13 := 
by 
  intros a b ha hnormb htheta
  sorry

end vector_magnitude_problem_l808_808957


namespace population_of_men_percentage_of_women_eq_200_l808_808530

variable (M : ℝ)
variable (W : ℝ)

-- Define the condition: W is 50% of M
def women_population (M : ℝ) : ℝ := 0.5 * M

theorem population_of_men_percentage_of_women_eq_200 :
  W = women_population M → (M / W) * 100 = 200 := 
by
  intro h
  rw [h, women_population]
  calc
    (M / (0.5 * M)) * 100 = (1 / 0.5) * 100 := by rw div_mul_cancel M (by norm_num)
                      _   = 2 * 100 := by norm_num
                      _   = 200 := by norm_num

end population_of_men_percentage_of_women_eq_200_l808_808530


namespace num_distinct_points_l808_808020

noncomputable def distinctPoints : ℕ :=
  let system1 := (2 * x - y - 3 = 0) ∧ (x + y + 2 = 0)
  let system2 := (2 * x - y - 3 = 0) ∧ (4 * x - 3 * y - 1 = 0)
  -- Check the solutions to both systems
  if (∃ x y, system1) ∧ (∃ x y, system2) then 2 else 0

theorem num_distinct_points : distinctPoints = 2 := by
  -- Proof omitted as per instructions
  sorry

end num_distinct_points_l808_808020


namespace carnations_percentage_l808_808736

variables {α : Type*} [field α]

def fraction_blue (total : α) : α := total * (1/2)
def fraction_red (total : α) : α := total * (1/2)
def fraction_blue_roses (total : α) : α := fraction_blue total * (2/5)
def fraction_blue_carnations (total : α) : α := fraction_blue total * (3/5)
def fraction_red_carnations (total : α) : α := fraction_red total * (2/3)

theorem carnations_percentage (total : α) : 
  (fraction_blue_carnations total + fraction_red_carnations total) / total * 100 = 63 := 
sorry

end carnations_percentage_l808_808736


namespace smallest_x_divisible_l808_808299

theorem smallest_x_divisible :
  ∃ x : ℤ, 2 * x + 2 = 73260 ∧ (∀ a ∈ {33, 44, 55, 666}, a ∣ (2 * x + 2)) :=
begin
  use 36629,
  split,
  { norm_num, },
  { intros a ha,
    fin_cases ha;
    norm_num }
end

end smallest_x_divisible_l808_808299


namespace fuel_used_l808_808178

theorem fuel_used (x : ℝ) (h1 : x + 0.8 * x = 27) : x = 15 :=
sorry

end fuel_used_l808_808178


namespace impossibility_of_domino_tiling_l808_808151

-- Define the size and shape of the chessboard and the dominoes
def board_size : ℕ := 8
def domino_length : ℕ := 2
def domino_width : ℕ := 1

-- Define a function to denote a valid tiling 
def valid_domino_tiling (positions : list (ℕ × ℕ)) : Prop :=
  ∀ d1 d2 ∈ positions, (d1 ≠ d2) → 
    ¬(d1.1 = d2.1 ∧ d1.2 = d2.2) ∧ 
    ∀ {i j : ℕ}, i < board_size → j < board_size → 
    (• (positions.contains (i, j) ∨ positions.contains (i+1, j) ∨ positions.contains (i, j+1)) → 
       (positions.contains (i+1, j+1) ∨ positions.contains (i+1, j-1) ∨ positions.contains (i-1, j+1) ∨ positions.contains (i-1, j-1) → False))
    
-- State the contradiction
theorem impossibility_of_domino_tiling : 
  ∀ (positions : list (ℕ × ℕ)), positions.length = (board_size * board_size) / (domino_length * domino_width) →
  valid_domino_tiling positions →
  False :=
begin
  sorry
end

end impossibility_of_domino_tiling_l808_808151


namespace jessica_attended_games_l808_808686

/-- 
Let total_games be the total number of soccer games.
Let initially_planned be the number of games Jessica initially planned to attend.
Let commitments_skipped be the number of games skipped due to other commitments.
Let rescheduled_games be the rescheduled games during the season.
Let additional_missed be the additional games missed due to rescheduling.
-/
theorem jessica_attended_games
    (total_games initially_planned commitments_skipped rescheduled_games additional_missed : ℕ)
    (h1 : total_games = 12)
    (h2 : initially_planned = 8)
    (h3 : commitments_skipped = 3)
    (h4 : rescheduled_games = 2)
    (h5 : additional_missed = 4) :
    (initially_planned - commitments_skipped) - additional_missed = 1 := by
  sorry

end jessica_attended_games_l808_808686


namespace non_overlapping_original_sets_exists_l808_808869

-- Define the grid and the notion of an original set
def grid (n : ℕ) := {cells : set (ℕ × ℕ) // cells ⊆ ({i | i < n} × {i | i < n})}

def is_original_set (n : ℕ) (cells : set (ℕ × ℕ)) : Prop :=
  cells.card = n - 1 ∧ ∀ (c1 c2 : ℕ × ℕ), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 →
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

-- Prove that there exist n+1 non-overlapping original sets for any n
theorem non_overlapping_original_sets_exists (n : ℕ) :
  ∃ (sets : fin (n+1) → set (ℕ × ℕ)), (∀ i, is_original_set n (sets i)) ∧
  (∀ i j, i ≠ j → sets i ∩ sets j = ∅) :=
sorry

end non_overlapping_original_sets_exists_l808_808869


namespace setD_is_pythagorean_triple_l808_808794

def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the sets
def setA := (1, 2, 3)
def setB := (4, 5, 6)
def setC := (6, 8, 9)
def setD := (7, 24, 25)

-- Prove that Set D is a Pythagorean triple
theorem setD_is_pythagorean_triple : is_pythagorean_triple 7 24 25 :=
by
  show 7^2 + 24^2 = 25^2,
  calc
  7^2 + 24^2 = 49 + 576 := by norm_num
  ... = 625 := by norm_num
  ... = 25^2 := by norm_num

end setD_is_pythagorean_triple_l808_808794


namespace y_axis_intersection_l808_808725

-- Define the structures and the given points
structure Point :=
  (x : ℝ)
  (y : ℝ)

def p1 : Point := ⟨3, 27⟩
def p2 : Point := ⟨-7, -5⟩

-- Define the function for computing the intersection with y-axis
def y_intercept (p1 p2 : Point) : ℝ :=
  let slope := (p2.y - p1.y) / (p2.x - p1.x)
  let intercept := p1.y - slope * p1.x
  intercept

-- The main theorem
theorem y_axis_intersection :
  y_intercept p1 p2 = 17.4 :=
by
  -- Placeholder for the actual proof
  sorry

end y_axis_intersection_l808_808725


namespace quadratic_inequality_solution_set_l808_808936

theorem quadratic_inequality_solution_set {a b x : ℝ}
  (h₀ : ∀ x, -1 < x ∧ x < 2 → ax^2 + bx + 2 > 0)
  (h₁ : ∀ x, ax^2 + bx + 2 = 0 → (x = -1 ∨ x = 2))
  (h₂ : a < 0) :
  (∀ x, 2x^2 + bx + a < 0 ↔ -1 < x ∧ x < 0.5) :=
by
  sorry

end quadratic_inequality_solution_set_l808_808936


namespace max_sales_l808_808237

def P (t : ℕ) : ℝ :=
  if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else if 1 ≤ t ∧ t ≤ 24 then t + 20
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def y (t : ℕ) : ℝ :=
  if 25 ≤ t ∧ t ≤ 30 then t^2 - 140 * t + 4000
  else if 1 ≤ t ∧ t ≤ 24 then -t^2 + 20 * t + 800
  else 0

theorem max_sales : ∃ tₘ : ℕ, 1 ≤ tₘ ∧ tₘ ≤ 30 ∧ y tₘ = 1125 := 
by
  have t₁ := 25
  have t₁_in_range : 25 ≤ t₁ ∧ t₁ ≤ 30 := ⟨by norm_num, by norm_num⟩
  have y_t₁ : y t₁ = 1125 := 
    by 
      have t_in_y_range : 1 ≤ t₁ ∧ t₁ ≤ 30 := ⟨by norm_num, by norm_num⟩
      simp [y, t_in_y_range, t₁_in_range]
      norm_num
      
  exact ⟨t₁, by norm_num, y_t₁⟩

end max_sales_l808_808237


namespace total_words_in_poem_l808_808630

theorem total_words_in_poem 
  (stanzas : ℕ) 
  (lines_per_stanza : ℕ) 
  (words_per_line : ℕ) 
  (h_stanzas : stanzas = 20) 
  (h_lines_per_stanza : lines_per_stanza = 10) 
  (h_words_per_line : words_per_line = 8) : 
  stanzas * lines_per_stanza * words_per_line = 1600 := 
sorry

end total_words_in_poem_l808_808630


namespace chord_length_l808_808560

-- Definitions and conditions for the problem
variables (A D B C G E F : Point)

-- Lengths and radii in the problem
noncomputable def radius : Real := 10
noncomputable def AB : Real := 20
noncomputable def BC : Real := 20
noncomputable def CD : Real := 20

-- Centers of circles
variables (O N P : Circle) (AN ND : Real)

-- Tangent properties and intersection points
variable (tangent_AG : Tangent AG P G)
variable (intersect_AG_N : Intersects AG N E F)

-- Given the geometry setup, prove the length of chord EF.
theorem chord_length (EF_length : Real) :
  EF_length = 2 * Real.sqrt 93.75 := sorry

end chord_length_l808_808560


namespace expr1_correct_expr2_correct_l808_808005

noncomputable def expr1 : Float :=
  0.027 ^ (-1 / 3) - ((-1 / 7) ^ (-2)) + (256 ^ (3 / 4)) - (3 ^ (-1)) + ((Real.sqrt 2 - 1) ^ 0)

noncomputable def expr2 : Float :=
  (Real.logBase 2.5 6.25) + (Real.log 10 0.01) + (Real.log (Real.sqrt Float e)) - (2 ^ (1 + Real.log 2 3))

theorem expr1_correct : expr1 = 19 := by
  sorry

theorem expr2_correct : expr2 = -11 / 2 := by
  sorry

end expr1_correct_expr2_correct_l808_808005


namespace number_of_true_propositions_l808_808489

theorem number_of_true_propositions (a b : ℝ) (h_rhombus_sides : ∀ (q : Type), q -> Prop)
                                     (h_isosceles_angles : ∀ (t : Type), t -> Prop) :
  ((∀ (a b : ℝ), (a / b < 1) → (a < b)) ↔ (∀ (a b : ℝ), (a < b) → (a / b < 1)))
  ∧ ((∀ (q : Type), h_rhombus_sides q) ↔ (∀ (q : Type), h_rhombus_sides q))
  ∧ ((∀ (t : Type), h_isosceles_angles t) ↔ (∀ (t : Type), h_isosceles_angles t)) →
  2 := sorry

end number_of_true_propositions_l808_808489


namespace second_investment_amount_l808_808975

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

theorem second_investment_amount :
  ∀ (P₁ P₂ I₁ I₂ r t : ℝ), 
    P₁ = 5000 →
    I₁ = 250 →
    I₂ = 1000 →
    I₁ = simple_interest P₁ r t →
    I₂ = simple_interest P₂ r t →
    P₂ = 20000 := 
by 
  intros P₁ P₂ I₁ I₂ r t hP₁ hI₁ hI₂ hI₁_eq hI₂_eq
  sorry

end second_investment_amount_l808_808975


namespace choose_non_overlapping_sets_for_any_n_l808_808893

def original_set (n : ℕ) (s : set (ℕ × ℕ)) : Prop :=
  s.card = n - 1 ∧ (∀ (i j : ℕ × ℕ), i ∈ s → j ∈ s → i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2)

theorem choose_non_overlapping_sets_for_any_n (n : ℕ) :
  ∃ sets : set (set (ℕ × ℕ)), sets.card = n + 1 ∧ (∀ s ∈ sets, original_set n s) ∧ 
    ∀ (s1 s2 ∈ sets), s1 ≠ s2 → s1 ∩ s2 = ∅ :=
sorry

end choose_non_overlapping_sets_for_any_n_l808_808893


namespace original_sets_exist_l808_808900

noncomputable def original_set (n : ℕ) (cell_set : finset (ℕ × ℕ)) : Prop :=
  cell_set.card = n - 1 ∧
  ∀ ⦃c1 c2 : ℕ × ℕ⦄, c1 ∈ cell_set → c2 ∈ cell_set → c1 ≠ c2 → c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2

theorem original_sets_exist (n : ℕ) (h : 1 ≤ n) :
  ∃ sets : finset (finset (ℕ × ℕ)), sets.card = n + 1 ∧ ∀ s ∈ sets, original_set n s :=
sorry

end original_sets_exist_l808_808900


namespace worker_total_hours_worked_l808_808783

-- Given conditions
variables (h : ℝ) (ot : ℝ := 8) (ordinary_rate : ℝ := 0.60) (overtime_rate : ℝ := 0.90) (total_earnings : ℝ := 32.40)

-- The total hours worked
def total_hours_worked : ℝ := h + ot

-- The earnings equations
def ordinary_earnings : ℝ := ordinary_rate * h
def overtime_earnings : ℝ := overtime_rate * ot

-- The main theorem
theorem worker_total_hours_worked : ordinary_earnings + overtime_earnings = total_earnings → total_hours_worked = 50 :=
by
  -- We skip the proof; the statement is what matters
  sorry

end worker_total_hours_worked_l808_808783


namespace natural_numbers_finite_movement_l808_808708

noncomputable def move_operation (m : ℕ) (nums : List ℕ) : List ℕ :=
  nums.take m ++ nums.drop m.rotate_left

def finite_moves (n : ℕ) : Prop :=
  ∀ m : ℕ, ∃ N : ℕ, ∀ k ≥ N, n ∈ move_operation k [1, 2, 3, ..., n]

theorem natural_numbers_finite_movement : ∀ n : ℕ, finite_moves n :=
by
  sorry

end natural_numbers_finite_movement_l808_808708


namespace farmer_total_acres_l808_808756

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l808_808756


namespace original_sets_exist_l808_808878

theorem original_sets_exist (n : ℕ) : ∃ (sets : fin (n+1) → set (fin n × fin n)), 
  (∀ i : fin (n+1), (∀ j k : (fin n × fin n), j ∈ sets i → k ∈ sets i → (j ≠ k) → j.1 ≠ k.1 ∧ j.2 ≠ k.2)) ∧
  (∀ i j : fin (n+1), i ≠ j → sets i ∩ sets j = ∅) ∧
  (∀ i : fin (n+1), ∃ (l : fin (n-1)), sets i = { x : (fin n × fin n) | x ∈ sets i }) :=
sorry

end original_sets_exist_l808_808878


namespace three_lines_intersect_at_one_point_l808_808956

theorem three_lines_intersect_at_one_point
  (M O1 O2 O3 A B C : Point)
  (r : ℝ)
  (circle1 : Circle O1 r)
  (circle2 : Circle O2 r)
  (circle3 : Circle O3 r)
  (h_intersect_O1O2 : A ∈ circle1 ∧ A ∈ circle2 ∧ M ∈ circle1 ∧ M ∈ circle2)
  (h_intersect_O2O3 : B ∈ circle2 ∧ B ∈ circle3 ∧ M ∈ circle2 ∧ M ∈ circle3)
  (h_intersect_O1O3 : C ∈ circle1 ∧ C ∈ circle3 ∧ M ∈ circle1 ∧ M ∈ circle3) :
  ∃ P : Point, Collinear P A O3 ∧ Collinear P B O1 ∧ Collinear P C O2 :=
by
  sorry

end three_lines_intersect_at_one_point_l808_808956


namespace number_of_full_time_employees_l808_808765

theorem number_of_full_time_employees (total_employees part_time_employees : ℕ) (h1 : total_employees = 65134) (h2 : part_time_employees = 2041) : 
  total_employees - part_time_employees = 63093 := 
by 
  rw [h1, h2]
  -- you would complete the proof here if required
  -- however, as instructed, we're including sorry in this task
  sorry

end number_of_full_time_employees_l808_808765


namespace simplification_correct_l808_808633

noncomputable def simplify_expr : ℝ :=
  √(528 / 32) - √(297 / 99)

theorem simplification_correct : simplify_expr = 2.318 := by
  sorry

end simplification_correct_l808_808633


namespace measurable_set_eq_l808_808201

variables {Ω : Type*} {F : Set (Set Ω)}
variables [MeasurableSpace Ω]

-- Assume ξ and η are measurable w.r.t. the σ-algebra F.
variable {ξ η : Ω → ℝ}
hypothesis Hξ : Measurable ξ 
hypothesis Hη : Measurable η 

-- Define the set {ω : ξ(ω) = η(ω)}
def set_eq : Set Ω := {ω | ξ ω = η ω}

-- Statement to prove
theorem measurable_set_eq : set_eq ∈ F :=
sorry

end measurable_set_eq_l808_808201


namespace sum_of_possible_m_plus_n_l808_808240

theorem sum_of_possible_m_plus_n : 
  ∃ (m n : ℝ), m ≠ n ∧ (m ≠ 2 ∧ m ≠ 5 ∧ m ≠ 8 ∧ m ≠ 11) ∧ (n ≠ 2 ∧ n ≠ 5 ∧ n ≠ 8 ∧ n ≠ 11) ∧ 
  let set := {2, 5, 8, 11, m, n} in
  (∀ sorted_set : list ℝ, sorted_set = list.sort (≤) (list.of_set set) → 
  (sorted_set.nth 2 + sorted_set.nth 3) / 2 = (2 + 5 + 8 + 11 + m + n) / 6) →
  m + n = 44 :=
by sorry

end sum_of_possible_m_plus_n_l808_808240


namespace complement_intersection_l808_808499

-- Define sets P and Q.
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the complement of P.
def complement_P : Set ℝ := {x | x < 2}

-- The theorem we need to prove.
theorem complement_intersection : complement_P ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l808_808499
