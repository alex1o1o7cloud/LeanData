import Mathlib

namespace smallest_consecutive_even_sum_560_l77_77214

theorem smallest_consecutive_even_sum_560 (n : ℕ) (h : 7 * n + 42 = 560) : n = 74 :=
  by
    sorry

end smallest_consecutive_even_sum_560_l77_77214


namespace pipe_A_fills_tank_in_16_hours_l77_77322

theorem pipe_A_fills_tank_in_16_hours
  (A : ℝ)
  (h1 : ∀ t : ℝ, t = 12.000000000000002 → (1/A + 1/24) * t = 5/4) :
  A = 16 :=
by sorry

end pipe_A_fills_tank_in_16_hours_l77_77322


namespace total_weight_four_pets_l77_77565

-- Define the weights
def Evan_dog := 63
def Ivan_dog := Evan_dog / 7
def combined_weight_dogs := Evan_dog + Ivan_dog
def Kara_cat := combined_weight_dogs * 5
def combined_weight_dogs_and_cat := Evan_dog + Ivan_dog + Kara_cat
def Lisa_parrot := combined_weight_dogs_and_cat * 3
def total_weight := Evan_dog + Ivan_dog + Kara_cat + Lisa_parrot

-- Total weight of the four pets
theorem total_weight_four_pets : total_weight = 1728 := by
  sorry

end total_weight_four_pets_l77_77565


namespace combined_original_price_of_books_l77_77544

theorem combined_original_price_of_books (p1 p2 : ℝ) (h1 : p1 / 8 = 8) (h2 : p2 / 9 = 9) :
  p1 + p2 = 145 :=
sorry

end combined_original_price_of_books_l77_77544


namespace largest_of_five_consecutive_integers_l77_77439

theorem largest_of_five_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120) : n + 4 = 9 :=
sorry

end largest_of_five_consecutive_integers_l77_77439


namespace part1_part2_l77_77927

-- Statements derived from Step c)
theorem part1 {m : ℝ} (h : ∃ x : ℝ, m - |5 - 2 * x| - |2 * x - 1| = 0) : 4 ≤ m := by
  sorry

theorem part2 {x : ℝ} (hx : |x - 3| + |x + 4| ≤ 8) : -9 / 2 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

end part1_part2_l77_77927


namespace steve_speed_during_race_l77_77535

theorem steve_speed_during_race 
  (distance_gap : ℝ) 
  (john_speed : ℝ) 
  (time : ℝ) 
  (john_ahead : ℝ)
  (steve_speed : ℝ) :
  distance_gap = 16 →
  john_speed = 4.2 →
  time = 36 →
  john_ahead = 2 →
  steve_speed = (151.2 - 18) / 36 :=
by
  sorry

end steve_speed_during_race_l77_77535


namespace slope_range_l77_77390

theorem slope_range (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ k : ℝ, k = (y + 2) / (x + 1) ∧ k ∈ Set.Ici (3 / 4) :=
sorry

end slope_range_l77_77390


namespace stickers_per_page_l77_77726

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end stickers_per_page_l77_77726


namespace tanya_erasers_l77_77602

theorem tanya_erasers (H R TR T : ℕ) 
  (h1 : H = 2 * R) 
  (h2 : R = TR / 2 - 3) 
  (h3 : H = 4) 
  (h4 : TR = T / 2) : 
  T = 20 := 
by 
  sorry

end tanya_erasers_l77_77602


namespace smallest_number_l77_77579

theorem smallest_number (N : ℤ) : (∃ (k : ℤ), N = 24 * k + 34) ∧ ∀ n, (∃ (k : ℤ), n = 24 * k + 10) -> n ≥ 34 := sorry

end smallest_number_l77_77579


namespace sum_of_f_values_l77_77102

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_of_f_values 
  (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_periodic : ∀ x, f (2 - x) = f x)
  (hf_neg_one : f (-1) = 1) :
  f 1 + f 2 + f 3 + f 4 + (502 * (f 1 + f 2 + f 3 + f 4)) = -1 := 
sorry

end sum_of_f_values_l77_77102


namespace sarah_books_check_out_l77_77657

theorem sarah_books_check_out
  (words_per_minute : ℕ)
  (words_per_page : ℕ)
  (pages_per_book : ℕ)
  (reading_hours : ℕ)
  (number_of_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  pages_per_book = 80 →
  reading_hours = 20 →
  number_of_books = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sarah_books_check_out_l77_77657


namespace six_degree_below_zero_is_minus_six_degrees_l77_77319

def temp_above_zero (temp: Int) : String := "+" ++ toString temp ++ "°C"

def temp_below_zero (temp: Int) : String := "-" ++ toString temp ++ "°C"

-- Statement of the theorem
theorem six_degree_below_zero_is_minus_six_degrees:
  temp_below_zero 6 = "-6°C" :=
by
  sorry

end six_degree_below_zero_is_minus_six_degrees_l77_77319


namespace woman_worked_days_l77_77908

-- Define variables and conditions
variables (W I : ℕ)

-- Conditions
def total_days : Prop := W + I = 25
def net_earnings : Prop := 20 * W - 5 * I = 450

-- Main theorem statement
theorem woman_worked_days (h1 : total_days W I) (h2 : net_earnings W I) : W = 23 :=
sorry

end woman_worked_days_l77_77908


namespace problem1_problem2_l77_77976

-- Problem 1: Prove (-a^3)^2 * (-a^2)^3 / a = -a^11 given a is a real number.
theorem problem1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 :=
  sorry

-- Problem 2: Prove (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 given m, n are real numbers.
theorem problem2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 :=
  sorry

end problem1_problem2_l77_77976


namespace total_value_of_goods_l77_77403

theorem total_value_of_goods (V : ℝ)
  (h1 : 0 < V)
  (h2 : ∃ t, V - 600 = t ∧ 0.12 * t = 134.4) :
  V = 1720 := 
sorry

end total_value_of_goods_l77_77403


namespace intersection_complement_l77_77431

universe u
variable {α : Type u}

-- Define the sets I, M, N, and their complement with respect to I
def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}
def complement_I (s : Set ℕ) : Set ℕ := { x ∈ I | x ∉ s }

-- Statement of the theorem
theorem intersection_complement :
  M ∩ (complement_I N) = {1} :=
by
  sorry

end intersection_complement_l77_77431


namespace average_fixed_points_of_permutation_l77_77209

open Finset

noncomputable def average_fixed_points (n : ℕ) : ℕ :=
  1

theorem average_fixed_points_of_permutation (n : ℕ) :
  ∀ (σ : (Fin n) → (Fin n)), 
  (1: ℚ) = (1: ℕ) :=
by
  sorry

end average_fixed_points_of_permutation_l77_77209


namespace fraction_sum_divided_by_2_equals_decimal_l77_77457

theorem fraction_sum_divided_by_2_equals_decimal :
  let f1 := (3 : ℚ) / 20
  let f2 := (5 : ℚ) / 200
  let f3 := (7 : ℚ) / 2000
  let sum := f1 + f2 + f3
  let result := sum / 2
  result = 0.08925 := 
by
  sorry

end fraction_sum_divided_by_2_equals_decimal_l77_77457


namespace total_fruit_salads_correct_l77_77055

-- Definitions for the conditions
def alayas_fruit_salads : ℕ := 200
def angels_fruit_salads : ℕ := 2 * alayas_fruit_salads
def total_fruit_salads : ℕ := alayas_fruit_salads + angels_fruit_salads

-- Theorem statement
theorem total_fruit_salads_correct : total_fruit_salads = 600 := by
  -- Proof goes here, but is not required for this task
  sorry

end total_fruit_salads_correct_l77_77055


namespace THIS_code_is_2345_l77_77437

def letterToDigit (c : Char) : Option Nat :=
  match c with
  | 'M' => some 0
  | 'A' => some 1
  | 'T' => some 2
  | 'H' => some 3
  | 'I' => some 4
  | 'S' => some 5
  | 'F' => some 6
  | 'U' => some 7
  | 'N' => some 8
  | _   => none

def codeToNumber (code : String) : Option String :=
  code.toList.mapM letterToDigit >>= fun digits => some (digits.foldl (fun acc d => acc ++ toString d) "")

theorem THIS_code_is_2345 :
  codeToNumber "THIS" = some "2345" :=
by
  sorry

end THIS_code_is_2345_l77_77437


namespace value_of_expression_l77_77356

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  (a + b + c + d).sqrt + (a^2 - 2*a + 3 - b).sqrt - (b - c^2 + 4*c - 8).sqrt = 3

theorem value_of_expression (a b c d : ℝ) (h : proof_problem a b c d) : a - b + c - d = -7 :=
sorry

end value_of_expression_l77_77356


namespace bob_grade_is_35_l77_77830

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l77_77830


namespace smallest_ratio_l77_77737

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end smallest_ratio_l77_77737


namespace blocks_combination_count_l77_77060

-- Definition statements reflecting all conditions in the problem
def select_4_blocks_combinations : ℕ :=
  let choose (n k : ℕ) := Nat.choose n k
  let factorial (n : ℕ) := Nat.factorial n
  choose 6 4 * choose 6 4 * factorial 4

-- Theorem stating the result we want to prove
theorem blocks_combination_count : select_4_blocks_combinations = 5400 :=
by
  -- We will provide the proof steps here
  sorry

end blocks_combination_count_l77_77060


namespace problem_statement_l77_77574

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions of the problem
def cond1 : Prop := (1 / x) + (1 / y) = 2
def cond2 : Prop := (x * y) + x - y = 6

-- The corresponding theorem to prove: x² - y² = 2
theorem problem_statement (h1 : cond1) (h2 : cond2) : x^2 - y^2 = 2 :=
  sorry

end problem_statement_l77_77574


namespace work_together_l77_77380

variable (W : ℝ) -- 'W' denotes the total work
variable (a_days b_days c_days : ℝ)

-- Conditions provided in the problem
axiom a_work : a_days = 18
axiom b_work : b_days = 6
axiom c_work : c_days = 12

-- The statement to be proved
theorem work_together :
  (W / a_days + W / b_days + W / c_days) * (36 / 11) = W := by
  sorry

end work_together_l77_77380


namespace total_animals_l77_77991

-- Define the number of pigs and giraffes
def num_pigs : ℕ := 7
def num_giraffes : ℕ := 6

-- Theorem stating the total number of giraffes and pigs
theorem total_animals : num_pigs + num_giraffes = 13 :=
by sorry

end total_animals_l77_77991


namespace proof_b_lt_a_lt_c_l77_77904

noncomputable def a : ℝ := 2^(4/5)
noncomputable def b : ℝ := 4^(2/7)
noncomputable def c : ℝ := 25^(1/5)

theorem proof_b_lt_a_lt_c : b < a ∧ a < c := by
  sorry

end proof_b_lt_a_lt_c_l77_77904


namespace possible_values_of_m_l77_77697

open Complex

theorem possible_values_of_m (p q r s m : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
  (h5 : p * m^3 + q * m^2 + r * m + s = 0)
  (h6 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end possible_values_of_m_l77_77697


namespace solution_set_of_abs_inequality_l77_77894

theorem solution_set_of_abs_inequality :
  {x : ℝ // |2 * x - 1| < 3} = {x : ℝ // -1 < x ∧ x < 2} :=
by sorry

end solution_set_of_abs_inequality_l77_77894


namespace max_f_l77_77146

noncomputable def f (x : ℝ) : ℝ :=
  1 / (|x + 3| + |x + 1| + |x - 2| + |x - 5|)

theorem max_f : ∃ x : ℝ, f x = 1 / 11 :=
by
  sorry

end max_f_l77_77146


namespace find_sum_l77_77736

theorem find_sum {x y : ℝ} (h1 : x = 13.0) (h2 : x + y = 24) : 7 * x + 5 * y = 146 := 
by
  sorry

end find_sum_l77_77736


namespace number_is_580_l77_77649

noncomputable def find_number (x : ℝ) : Prop :=
  0.20 * x = 116

theorem number_is_580 (x : ℝ) (h : find_number x) : x = 580 :=
  by sorry

end number_is_580_l77_77649


namespace truncated_pyramid_smaller_base_area_l77_77770

noncomputable def smaller_base_area (a : ℝ) (α β : ℝ) : ℝ :=
  (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2

theorem truncated_pyramid_smaller_base_area (a α β : ℝ) :
  smaller_base_area a α β = (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2 :=
by
  unfold smaller_base_area
  sorry

end truncated_pyramid_smaller_base_area_l77_77770


namespace units_digit_of_8_pow_47_l77_77496

theorem units_digit_of_8_pow_47 : (8 ^ 47) % 10 = 2 := by
  sorry

end units_digit_of_8_pow_47_l77_77496


namespace find_m_range_l77_77546

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2 * (m - 1) * x + 2 

theorem find_m_range (m : ℝ) : (∀ x ≤ 4, f x m ≤ f (x + 1) m) → m ≤ -3 :=
by
  sorry

end find_m_range_l77_77546


namespace fiona_pairs_l77_77053

theorem fiona_pairs : Nat.choose 12 2 = 66 := by
  sorry

end fiona_pairs_l77_77053


namespace find_a1_an_l77_77756

noncomputable def arith_geo_seq (a : ℕ → ℝ) : Prop :=
  (∃ d ≠ 0, (a 2 + a 4 = 10) ∧ (a 2 ^ 2 = a 1 * a 5))

theorem find_a1_an (a : ℕ → ℝ)
  (h_arith_geo_seq : arith_geo_seq a) :
  a 1 = 1 ∧ (∀ n, a n = 2 * n - 1) :=
sorry

end find_a1_an_l77_77756


namespace sin_value_of_arithmetic_sequence_l77_77818

open Real

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sin_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : arithmetic_sequence a) 
  (h_cond : a 1 + a 5 + a 9 = 5 * π) : 
  sin (a 2 + a 8) = - (sqrt 3 / 2) :=
by
  sorry

end sin_value_of_arithmetic_sequence_l77_77818


namespace solve_for_x_l77_77027

def star (a b : ℕ) := a * b + a + b

theorem solve_for_x : ∃ x : ℕ, star 3 x = 27 ∧ x = 6 :=
by {
  sorry
}

end solve_for_x_l77_77027


namespace jenny_chocolate_milk_probability_l77_77733

-- Define the binomial probability function.
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  ( Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Given conditions: probability each day and total number of days.
def probability_each_day : ℚ := 2 / 3
def num_days : ℕ := 7
def successful_days : ℕ := 3

-- The problem statement to prove.
theorem jenny_chocolate_milk_probability :
  binomial_probability num_days successful_days probability_each_day = 280 / 2187 :=
by
  sorry

end jenny_chocolate_milk_probability_l77_77733


namespace fraction_of_air_conditioned_rooms_rented_l77_77204

variable (R : ℚ)
variable (h1 : R > 0)
variable (rented_rooms : ℚ := (3/4) * R)
variable (air_conditioned_rooms : ℚ := (3/5) * R)
variable (not_rented_rooms : ℚ := (1/4) * R)
variable (air_conditioned_not_rented_rooms : ℚ := (4/5) * not_rented_rooms)
variable (air_conditioned_rented_rooms : ℚ := air_conditioned_rooms - air_conditioned_not_rented_rooms)
variable (fraction_air_conditioned_rented : ℚ := air_conditioned_rented_rooms / air_conditioned_rooms)

theorem fraction_of_air_conditioned_rooms_rented :
  fraction_air_conditioned_rented = (2/3) := by
  sorry

end fraction_of_air_conditioned_rooms_rented_l77_77204


namespace y_in_terms_of_x_l77_77536

theorem y_in_terms_of_x (p x y : ℝ) (h1 : x = 2 + 2^p) (h2 : y = 1 + 2^(-p)) : 
  y = (x-1)/(x-2) :=
by
  sorry

end y_in_terms_of_x_l77_77536


namespace quotient_of_37_div_8_l77_77305

theorem quotient_of_37_div_8 : (37 / 8) = 4 :=
by
  sorry

end quotient_of_37_div_8_l77_77305


namespace emma_finishes_first_l77_77443

noncomputable def david_lawn_area : ℝ := sorry
noncomputable def emma_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 3
noncomputable def fiona_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 4

noncomputable def david_mowing_rate : ℝ := sorry
noncomputable def fiona_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 6
noncomputable def emma_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 2

theorem emma_finishes_first (z w : ℝ) (hz : z > 0) (hw : w > 0) :
  (z / w) > (2 * z / (3 * w)) ∧ (3 * z / (2 * w)) > (2 * z / (3 * w)) :=
by
  sorry

end emma_finishes_first_l77_77443


namespace interval_contains_n_l77_77118

theorem interval_contains_n (n : ℕ) (h1 : n < 1000) (h2 : n ∣ 999) (h3 : n + 6 ∣ 99) : 1 ≤ n ∧ n ≤ 250 := 
sorry

end interval_contains_n_l77_77118


namespace find_added_value_l77_77759

theorem find_added_value (N : ℕ) (V : ℕ) (H : N = 1280) :
  ((N + V) / 125 = 7392 / 462) → V = 720 :=
by 
  sorry

end find_added_value_l77_77759


namespace determine_a_l77_77041

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 2 * x

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 0 :=
by
  intros h
  sorry

end determine_a_l77_77041


namespace M_inter_N_l77_77459

def M : Set ℝ := { x | -2 < x ∧ x < 1 }
def N : Set ℤ := { x | Int.natAbs x ≤ 2 }

theorem M_inter_N : { x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) < 1 } ∩ N = { -1, 0 } :=
by
  simp [M, N]
  sorry

end M_inter_N_l77_77459


namespace inequality_abc_squared_l77_77589

theorem inequality_abc_squared (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := 
sorry

end inequality_abc_squared_l77_77589


namespace find_number_l77_77474

theorem find_number (x : ℕ) (h : x * 99999 = 65818408915) : x = 658185 :=
sorry

end find_number_l77_77474


namespace symmetric_function_l77_77647

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def symmetric_about_axis (f : ℤ → ℤ) (axis : ℤ) : Prop :=
  ∀ x : ℤ, f (axis - x) = f (axis + x)

theorem symmetric_function (a : ℕ → ℤ) (d : ℤ) (f : ℤ → ℤ) (a1 a2 : ℤ) (axis : ℤ) :
  (∀ x, f x = |x - a1| + |x - a2|) →
  arithmetic_sequence a d →
  d ≠ 0 →
  axis = (a1 + a2) / 2 →
  symmetric_about_axis f axis :=
by
  -- Proof goes here
  sorry

end symmetric_function_l77_77647


namespace smallest_positive_omega_l77_77400

theorem smallest_positive_omega (f g : ℝ → ℝ) (ω : ℝ) 
  (hf : ∀ x, f x = Real.cos (ω * x)) 
  (hg : ∀ x, g x = Real.sin (ω * x - π / 4)) 
  (heq : ∀ x, f (x - π / 2) = g x) :
  ω = 3 / 2 :=
sorry

end smallest_positive_omega_l77_77400


namespace intercepts_equal_l77_77452

theorem intercepts_equal (a : ℝ) :
  (∃ x y : ℝ, ax + y - 2 - a = 0 ∧
              y = 0 ∧ x = (a + 2) / a ∧
              x = 0 ∧ y = 2 + a) →
  (a = 1 ∨ a = -2) :=
by
  sorry

end intercepts_equal_l77_77452


namespace problem_solution_l77_77553

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end problem_solution_l77_77553


namespace trader_profit_l77_77173

theorem trader_profit (P : ℝ) :
  let buy_price := 0.80 * P
  let sell_price := 1.20 * P
  sell_price - P = 0.20 * P := 
by
  sorry

end trader_profit_l77_77173


namespace alcohol_quantity_l77_77980

theorem alcohol_quantity (A W : ℕ) (h1 : 4 * W = 3 * A) (h2 : 4 * (W + 8) = 5 * A) : A = 16 := 
by
  sorry

end alcohol_quantity_l77_77980


namespace parallelogram_area_l77_77325

theorem parallelogram_area (θ : ℝ) (a b : ℝ) (hθ : θ = 100) (ha : a = 20) (hb : b = 10):
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  area = 200 * Real.cos 10 := 
by
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  sorry

end parallelogram_area_l77_77325


namespace claire_photos_l77_77115

theorem claire_photos (L R C : ℕ) (h1 : L = R) (h2 : L = 3 * C) (h3 : R = C + 28) : C = 14 := by
  sorry

end claire_photos_l77_77115


namespace max_f_max_ab_plus_bc_l77_77013

def f (x : ℝ) := |x - 3| - 2 * |x + 1|

theorem max_f : ∃ (m : ℝ), m = 4 ∧ (∀ x : ℝ, f x ≤ m) := 
  sorry

theorem max_ab_plus_bc (a b c : ℝ) : a > 0 ∧ b > 0 → a^2 + 2 * b^2 + c^2 = 4 → (ab + bc) ≤ 2 :=
  sorry

end max_f_max_ab_plus_bc_l77_77013


namespace sequences_count_equals_fibonacci_n_21_l77_77611

noncomputable def increasing_sequences_count (n: ℕ) : ℕ := 
  -- Function to count the number of valid increasing sequences
  sorry

def fibonacci : ℕ → ℕ 
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sequences_count_equals_fibonacci_n_21 :
  increasing_sequences_count 20 = fibonacci 21 :=
sorry

end sequences_count_equals_fibonacci_n_21_l77_77611


namespace John_gave_the_store_20_dollars_l77_77492

def slurpee_cost : ℕ := 2
def change_received : ℕ := 8
def slurpees_bought : ℕ := 6
def total_money_given : ℕ := slurpee_cost * slurpees_bought + change_received

theorem John_gave_the_store_20_dollars : total_money_given = 20 := 
by 
  sorry

end John_gave_the_store_20_dollars_l77_77492


namespace compute_div_mul_l77_77240

noncomputable def a : ℚ := 0.24
noncomputable def b : ℚ := 0.006

theorem compute_div_mul : ((a / b) * 2) = 80 := by
  sorry

end compute_div_mul_l77_77240


namespace cells_remain_illuminated_l77_77816

-- The rect grid screen of size m × n with more than (m - 1)(n - 1) cells illuminated 
-- with the condition that in any 2 × 2 square if three cells are not illuminated, 
-- then the fourth cell also turns off eventually.
theorem cells_remain_illuminated 
  {m n : ℕ} 
  (h1 : ∃ k : ℕ, k > (m - 1) * (n - 1) ∧ k ≤ m * n) 
  (h2 : ∀ (i j : ℕ) (hiv : i < m - 1) (hjv : j < n - 1), 
    (∃ c1 c2 c3 c4 : ℕ, 
      c1 + c2 + c3 + c4 = 4 ∧ 
      (c1 = 1 ∨ c2 = 1 ∨ c3 = 1 ∨ c4 = 1) → 
      (c1 = 0 ∧ c2 = 0 ∧ c3 = 0 ∧ c4 = 0))) :
  ∃ (i j : ℕ) (hil : i < m) (hjl : j < n), true := sorry

end cells_remain_illuminated_l77_77816


namespace initial_deposit_l77_77679

theorem initial_deposit (x : ℝ) 
  (h1 : x - (1 / 4) * x - (4 / 9) * ((3 / 4) * x) - 640 = (3 / 20) * x) 
  : x = 2400 := 
by 
  sorry

end initial_deposit_l77_77679


namespace average_of_three_numbers_l77_77934

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end average_of_three_numbers_l77_77934


namespace majority_vote_is_280_l77_77608

-- Definitions based on conditions from step (a)
def totalVotes : ℕ := 1400
def winningPercentage : ℝ := 0.60
def losingPercentage : ℝ := 0.40

-- Majority computation based on the winning and losing percentages
def majorityVotes : ℝ := totalVotes * winningPercentage - totalVotes * losingPercentage

-- Theorem statement
theorem majority_vote_is_280 : majorityVotes = 280 := by
  sorry

end majority_vote_is_280_l77_77608


namespace ineq_sqrt_two_l77_77423

theorem ineq_sqrt_two (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by 
  sorry

end ineq_sqrt_two_l77_77423


namespace solution_set_eq_l77_77850

noncomputable def f (x : ℝ) : ℝ := x^6 + x^2
noncomputable def g (x : ℝ) : ℝ := (2*x + 3)^3 + 2*x + 3

theorem solution_set_eq : {x : ℝ | f x = g x} = {-1, 3} :=
by
  sorry

end solution_set_eq_l77_77850


namespace time_to_finish_typing_l77_77847

-- Definitions
def words_per_minute : ℕ := 38
def total_words : ℕ := 4560

-- Theorem to prove
theorem time_to_finish_typing : (total_words / words_per_minute) / 60 = 2 := by
  sorry

end time_to_finish_typing_l77_77847


namespace arithmetic_mean_l77_77297

variable {x b c : ℝ}

theorem arithmetic_mean (hx : x ≠ 0) (hb : b ≠ c) : 
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) :=
by
  sorry

end arithmetic_mean_l77_77297


namespace option_A_option_B_option_C_option_D_l77_77584

theorem option_A (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) : a 20 = 211 :=
sorry

theorem option_B (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2^n * a n) : a 5 = 2^10 :=
sorry

theorem option_C (S : ℕ → ℝ) (h₀ : ∀ n, S n = 3^n + 1/2) : ¬(∃ r : ℝ, ∀ n, S n = S 1 * r ^ (n - 1)) :=
sorry

theorem option_D (S : ℕ → ℝ) (a : ℕ → ℝ) (h₀ : S 1 = 1) 
  (h₁ : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1))
  (h₂ : (S 8) / 8 - (S 4) / 4 = 8) : a 6 = 21 :=
sorry

end option_A_option_B_option_C_option_D_l77_77584


namespace john_made_money_l77_77875

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money_l77_77875


namespace rate_per_kg_for_grapes_l77_77395

theorem rate_per_kg_for_grapes (G : ℝ) (h : 9 * G + 9 * 55 = 1125) : G = 70 :=
by
  -- sorry to skip the proof
  sorry

end rate_per_kg_for_grapes_l77_77395


namespace millie_bracelets_left_l77_77150

def millie_bracelets_initial : ℕ := 9
def millie_bracelets_lost : ℕ := 2

theorem millie_bracelets_left : millie_bracelets_initial - millie_bracelets_lost = 7 := 
by
  sorry

end millie_bracelets_left_l77_77150


namespace man_l77_77892

/-- A man can row downstream at the rate of 45 kmph.
    A man can row upstream at the rate of 23 kmph.
    The rate of current is 11 kmph.
    The man's rate in still water is 34 kmph. -/
theorem man's_rate_in_still_water
  (v c : ℕ)
  (h1 : v + c = 45)
  (h2 : v - c = 23)
  (h3 : c = 11) : v = 34 := by
  sorry

end man_l77_77892


namespace verify_sum_of_new_rates_proof_l77_77191

-- Given conditions and initial setup
variable (k : ℕ)
variable (h_initial : ℕ := 5 * k) -- Hanhan's initial hourly rate
variable (x_initial : ℕ := 4 * k) -- Xixi's initial hourly rate
variable (increment : ℕ := 20)    -- Increment in hourly rates

-- New rates after increment
variable (h_new : ℕ := h_initial + increment) -- Hanhan's new hourly rate
variable (x_new : ℕ := x_initial + increment) -- Xixi's new hourly rate

-- Given ratios
variable (initial_ratio : h_initial / x_initial = 5 / 4) 
variable (new_ratio : h_new / x_new = 6 / 5)

-- Target sum of the new hourly rates
def sum_of_new_rates_proof : Prop :=
  h_new + x_new = 220

theorem verify_sum_of_new_rates_proof : sum_of_new_rates_proof k :=
by
  sorry

end verify_sum_of_new_rates_proof_l77_77191


namespace determine_parallel_planes_l77_77876

def Plane : Type := sorry
def Line : Type := sorry
def Parallel (x y : Line) : Prop := sorry
def Skew (x y : Line) : Prop := sorry
def PlaneParallel (α β : Plane) : Prop := sorry

variables (α β : Plane) (a b : Line)
variable (hSkew : Skew a b)
variable (hαa : Parallel a α) 
variable (hαb : Parallel b α)
variable (hβa : Parallel a β)
variable (hβb : Parallel b β)

theorem determine_parallel_planes : PlaneParallel α β := sorry

end determine_parallel_planes_l77_77876


namespace fisherman_bass_count_l77_77091

theorem fisherman_bass_count (B T G : ℕ) (h1 : T = B / 4) (h2 : G = 2 * B) (h3 : B + T + G = 104) : B = 32 :=
by
  sorry

end fisherman_bass_count_l77_77091


namespace smallest_perfect_cube_divisor_l77_77778

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  ∃ (a b c : ℕ), a = 6 ∧ b = 6 ∧ c = 6 ∧ (p^a * q^b * r^c) = (p^2 * q^2 * r^2)^3 ∧ 
  (p^a * q^b * r^c) % (p^2 * q^3 * r^4) = 0 := 
by
  sorry

end smallest_perfect_cube_divisor_l77_77778


namespace solve_system1_solve_system2_l77_77404

-- Define the conditions and the proof problem for System 1
theorem solve_system1 (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : 3 * x + 2 * y = 7) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- Define the conditions and the proof problem for System 2
theorem solve_system2 (x y : ℝ) (h1 : x - y = 3) (h2 : (x - y - 3) / 2 - y / 3 = -1) :
  x = 6 ∧ y = 3 := by
  sorry

end solve_system1_solve_system2_l77_77404


namespace minimum_a_plus_3b_l77_77011

-- Define the conditions
variables (a b : ℝ)
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_eq : a + 3 * b = 1 / a + 3 / b

-- State the theorem
theorem minimum_a_plus_3b (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 3 * b = 1 / a + 3 / b) : 
  a + 3 * b ≥ 4 :=
sorry

end minimum_a_plus_3b_l77_77011


namespace circle_center_radius_l77_77601

theorem circle_center_radius :
  ∀ x y : ℝ,
  x^2 + y^2 + 4 * x - 6 * y - 3 = 0 →
  (∃ h k r : ℝ, (x + h)^2 + (y + k)^2 = r^2 ∧ h = -2 ∧ k = 3 ∧ r = 4) :=
by
  intros x y hxy
  sorry

end circle_center_radius_l77_77601


namespace solve_system_of_equations_l77_77080

theorem solve_system_of_equations :
  ∃ (x y z : ℤ), (x + y + z = 6) ∧ (x + y * z = 7) ∧ 
  ((x = 7 ∧ y = 0 ∧ z = -1) ∨ 
   (x = 7 ∧ y = -1 ∧ z = 0) ∨ 
   (x = 1 ∧ y = 3 ∧ z = 2) ∨ 
   (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end solve_system_of_equations_l77_77080


namespace polynomial_divisibility_l77_77198

theorem polynomial_divisibility (P : Polynomial ℝ) (h_nonconstant : ∃ n : ℕ, P.degree = n ∧ n ≥ 1)
  (h_div : ∀ x : ℝ, P.eval (x^3 + 8) = 0 → P.eval (x^2 - 2*x + 4) = 0) :
  ∃ a : ℝ, ∃ n : ℕ, a ≠ 0 ∧ P = Polynomial.C a * Polynomial.X ^ n :=
sorry

end polynomial_divisibility_l77_77198


namespace total_ages_is_32_l77_77196

variable (a b c : ℕ)
variable (h_b : b = 12)
variable (h_a : a = b + 2)
variable (h_c : b = 2 * c)

theorem total_ages_is_32 (h_b : b = 12) (h_a : a = b + 2) (h_c : b = 2 * c) : a + b + c = 32 :=
by
  sorry

end total_ages_is_32_l77_77196


namespace gcf_60_75_l77_77104

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l77_77104


namespace return_speed_is_48_l77_77320

variable (d r : ℕ)
variable (t_1 t_2 : ℚ)

-- Given conditions
def distance_each_way : Prop := d = 120
def time_to_travel_A_to_B : Prop := t_1 = d / 80
def time_to_travel_B_to_A : Prop := t_2 = d / r
def average_speed_round_trip : Prop := 60 * (t_1 + t_2) = 2 * d

-- Statement to prove
theorem return_speed_is_48 :
  distance_each_way d ∧
  time_to_travel_A_to_B d t_1 ∧
  time_to_travel_B_to_A d r t_2 ∧
  average_speed_round_trip d t_1 t_2 →
  r = 48 :=
by
  intros
  sorry

end return_speed_is_48_l77_77320


namespace value_of_a_l77_77887

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - 3 * x - Real.log x + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem value_of_a (a x0 : ℝ) (h : f x0 a = 3) : a = 1 - Real.log 2 :=
by
  sorry

end value_of_a_l77_77887


namespace largest_package_markers_l77_77951

def Alex_markers : ℕ := 36
def Becca_markers : ℕ := 45
def Charlie_markers : ℕ := 60

theorem largest_package_markers (d : ℕ) :
  d ∣ Alex_markers ∧ d ∣ Becca_markers ∧ d ∣ Charlie_markers → d ≤ 3 :=
by
  sorry

end largest_package_markers_l77_77951


namespace cost_per_pouch_is_20_l77_77010

theorem cost_per_pouch_is_20 :
  let boxes := 10
  let pouches_per_box := 6
  let dollars := 12
  let cents_per_dollar := 100
  let total_pouches := boxes * pouches_per_box
  let total_cents := dollars * cents_per_dollar
  let cost_per_pouch := total_cents / total_pouches
  cost_per_pouch = 20 :=
by
  sorry

end cost_per_pouch_is_20_l77_77010


namespace simplify_div_expr_l77_77729

theorem simplify_div_expr (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
sorry

end simplify_div_expr_l77_77729


namespace number_of_possible_values_for_a_l77_77732

theorem number_of_possible_values_for_a :
  ∀ (a b c d : ℕ), 
  a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 3010 ∧ a^2 - b^2 + c^2 - d^2 = 3010 →
  ∃ n, n = 751 :=
by {
  sorry
}

end number_of_possible_values_for_a_l77_77732


namespace problem_equation_l77_77421

def interest_rate : ℝ := 0.0306
def principal : ℝ := 5000
def interest_tax : ℝ := 0.20

theorem problem_equation (x : ℝ) :
  x + principal * interest_rate * interest_tax = principal * (1 + interest_rate) :=
sorry

end problem_equation_l77_77421


namespace problem_solution_l77_77910

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a m : ℕ) (inv : ℕ) : Prop := 
  (a * inv) % m = 1

theorem problem_solution :
  is_right_triangle 60 144 156 ∧ multiplicative_inverse 300 3751 3618 :=
by
  sorry

end problem_solution_l77_77910


namespace ordered_pairs_l77_77324

theorem ordered_pairs (a b : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (x : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a * x (n + 1) - b * x n| < ε) :
  (a = 0 ∧ 0 < b) ∨ (0 < a ∧ |b / a| < 1) :=
sorry

end ordered_pairs_l77_77324


namespace sum_of_intervals_length_l77_77426

theorem sum_of_intervals_length (m : ℝ) (h : m ≠ 0) (h_pos : m > 0) :
  (∃ l : ℝ, ∀ x : ℝ, (1 < x ∧ x ≤ x₁) ∨ (2 < x ∧ x ≤ x₂) → 
  l = x₁ - 1 + x₂ - 2) → 
  l = 3 / m :=
sorry

end sum_of_intervals_length_l77_77426


namespace sean_whistles_l77_77891

def charles_whistles : ℕ := 128
def sean_more_whistles : ℕ := 95

theorem sean_whistles : charles_whistles + sean_more_whistles = 223 :=
by {
  sorry
}

end sean_whistles_l77_77891


namespace discount_amount_correct_l77_77858

noncomputable def cost_price : ℕ := 180
noncomputable def markup_percentage : ℝ := 0.45
noncomputable def profit_percentage : ℝ := 0.20

theorem discount_amount_correct : 
  let markup := cost_price * markup_percentage
  let mp := cost_price + markup
  let profit := cost_price * profit_percentage
  let sp := cost_price + profit
  let discount_amount := mp - sp
  discount_amount = 45 :=
by
  sorry

end discount_amount_correct_l77_77858


namespace train_crossing_time_l77_77764

/--
A train requires 8 seconds to pass a pole while it requires some seconds to cross a stationary train which is 400 meters long. 
The speed of the train is 144 km/h. Prove that it takes 18 seconds for the train to cross the stationary train.
-/
theorem train_crossing_time
  (train_speed_kmh : ℕ)
  (time_to_pass_pole : ℕ)
  (length_stationary_train : ℕ)
  (speed_mps : ℕ)
  (length_moving_train : ℕ)
  (total_length : ℕ)
  (crossing_time : ℕ) :
  train_speed_kmh = 144 →
  time_to_pass_pole = 8 →
  length_stationary_train = 400 →
  speed_mps = (train_speed_kmh * 1000) / 3600 →
  length_moving_train = speed_mps * time_to_pass_pole →
  total_length = length_moving_train + length_stationary_train →
  crossing_time = total_length / speed_mps →
  crossing_time = 18 :=
by
  intros;
  sorry

end train_crossing_time_l77_77764


namespace johns_new_total_lift_l77_77275

theorem johns_new_total_lift :
  let initial_squat := 700
  let initial_bench := 400
  let initial_deadlift := 800
  let squat_loss_percentage := 30 / 100.0
  let squat_loss := squat_loss_percentage * initial_squat
  let new_squat := initial_squat - squat_loss
  let new_bench := initial_bench
  let new_deadlift := initial_deadlift - 200
  new_squat + new_bench + new_deadlift = 1490 := 
by
  -- Proof will go here
  sorry

end johns_new_total_lift_l77_77275


namespace conjugate_axis_length_l77_77864

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
variable (e : ℝ) (h3 : e = Real.sqrt 7 / 2)
variable (c : ℝ) (h4 : c = a * e)
variable (P : ℝ × ℝ) (h5 : P = (c, b^2 / a))
variable (F1 F2 : ℝ × ℝ) (h6 : F1 = (-c, 0)) (h7 : F2 = (c, 0))
variable (h8 : dist P F2 = 9 / 2)
variable (h9 : P.1 = c) (h10 : P.2 = b^2 / a)
variable (h11 : PF_2 ⊥ F_1F_2)

theorem conjugate_axis_length : 2 * b = 6 * Real.sqrt 3 := by
  sorry

end conjugate_axis_length_l77_77864


namespace main_theorem_l77_77257

variable (x : ℤ)

def H : ℤ := 12 - (3 + 7) + x
def T : ℤ := 12 - 3 + 7 + x

theorem main_theorem : H - T + x = -14 + x :=
by
  sorry

end main_theorem_l77_77257


namespace gambler_win_percentage_l77_77032

theorem gambler_win_percentage :
  ∀ (T W play_extra : ℕ) (P_win_extra P_week P_current P_required : ℚ),
    T = 40 →
    P_win_extra = 0.80 →
    play_extra = 40 →
    P_week = 0.60 →
    P_required = 48 →
    (W + P_win_extra * play_extra = P_required) →
    (P_current = (W : ℚ) / T * 100) →
    P_current = 40 :=
by
  intros T W play_extra P_win_extra P_week P_current P_required h1 h2 h3 h4 h5 h6 h7
  sorry

end gambler_win_percentage_l77_77032


namespace triangle_angle_A_l77_77603

theorem triangle_angle_A (AC BC : ℝ) (angle_B : ℝ) (h_AC : AC = Real.sqrt 2) (h_BC : BC = 1) (h_angle_B : angle_B = 45) :
  ∃ (angle_A : ℝ), angle_A = 30 :=
by
  sorry

end triangle_angle_A_l77_77603


namespace number_of_parallelograms_l77_77707

theorem number_of_parallelograms : 
  (∀ b d k : ℕ, k > 1 → k * b * d = 500000 → (b * d > 0 ∧ y = x ∧ y = k * x)) → 
  (∃ N : ℕ, N = 720) :=
sorry

end number_of_parallelograms_l77_77707


namespace angle_B_is_180_l77_77947

variables {l k : Line} {A B C: Point}

def parallel (l k : Line) : Prop := sorry 
def angle (A B C : Point) : ℝ := sorry

theorem angle_B_is_180 (h1 : parallel l k) (h2 : angle A = 110) (h3 : angle C = 70) :
  angle B = 180 := 
by
  sorry

end angle_B_is_180_l77_77947


namespace abs_neg_four_minus_six_l77_77809

theorem abs_neg_four_minus_six : abs (-4 - 6) = 10 := 
by
  sorry

end abs_neg_four_minus_six_l77_77809


namespace sum_of_x_and_y_l77_77984

theorem sum_of_x_and_y (x y : ℤ) (h1 : 3 + x = 5) (h2 : -3 + y = 5) : x + y = 10 :=
by
  sorry

end sum_of_x_and_y_l77_77984


namespace positive_multiples_of_11_ending_with_7_l77_77554

-- Definitions for conditions
def is_multiple_of_11 (n : ℕ) : Prop := (n % 11 = 0)
def ends_with_7 (n : ℕ) : Prop := (n % 10 = 7)

-- Main theorem statement
theorem positive_multiples_of_11_ending_with_7 :
  ∃ n, (n = 13) ∧ ∀ k, is_multiple_of_11 k ∧ ends_with_7 k ∧ 0 < k ∧ k < 1500 → k = 77 + (k / 110) * 110 := 
sorry

end positive_multiples_of_11_ending_with_7_l77_77554


namespace hannah_age_is_48_l77_77936

-- Define the ages of the brothers
def num_brothers : ℕ := 3
def age_each_brother : ℕ := 8

-- Define the sum of brothers' ages
def sum_brothers_ages : ℕ := num_brothers * age_each_brother

-- Define the age of Hannah
def hannah_age : ℕ := 2 * sum_brothers_ages

-- The theorem to prove Hannah's age is 48 years
theorem hannah_age_is_48 : hannah_age = 48 := by
  sorry

end hannah_age_is_48_l77_77936


namespace steve_writes_24_pages_per_month_l77_77716

/-- Calculate the number of pages Steve writes in a month given the conditions. -/
theorem steve_writes_24_pages_per_month :
  (∃ (days_in_month : ℕ) (letter_interval : ℕ) (letter_minutes : ℕ) (page_minutes : ℕ) 
      (long_letter_factor : ℕ) (long_letter_minutes : ℕ) (total_pages : ℕ),
    days_in_month = 30 ∧ 
    letter_interval = 3 ∧ 
    letter_minutes = 20 ∧ 
    page_minutes = 10 ∧ 
    long_letter_factor = 2 ∧ 
    long_letter_minutes = 80 ∧ 
    total_pages = 24 ∧ 
    (days_in_month / letter_interval * (letter_minutes / page_minutes)
      + long_letter_minutes / (long_letter_factor * page_minutes) = total_pages)) :=
sorry

end steve_writes_24_pages_per_month_l77_77716


namespace triangle_to_initial_position_l77_77213

-- Definitions for triangle vertices
structure Point where
  x : Int
  y : Int

def p1 : Point := { x := 0, y := 0 }
def p2 : Point := { x := 6, y := 0 }
def p3 : Point := { x := 0, y := 4 }

-- Definitions for transformations
def rotate90 (p : Point) : Point := { x := -p.y, y := p.x }
def rotate180 (p : Point) : Point := { x := -p.x, y := -p.y }
def rotate270 (p : Point) : Point := { x := p.y, y := -p.x }
def reflect_y_eq_x (p : Point) : Point := { x := p.y, y := p.x }
def reflect_y_eq_neg_x (p : Point) : Point := { x := -p.y, y := -p.x }

-- Definitions for combination of transformations
-- This part defines how to combine transformations, e.g., as a sequence of three transformations.
def transform (fs : List (Point → Point)) (p : Point) : Point :=
  fs.foldl (fun acc f => f acc) p

-- The total number of valid sequences that return the triangle to its original position
def valid_sequences_count : Int := 6

-- Lean 4 statement
theorem triangle_to_initial_position : valid_sequences_count = 6 := by
  sorry

end triangle_to_initial_position_l77_77213


namespace evaluate_expression_l77_77255

theorem evaluate_expression : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end evaluate_expression_l77_77255


namespace water_required_to_prepare_saline_solution_l77_77061

theorem water_required_to_prepare_saline_solution (water_ratio : ℝ) (required_volume : ℝ) : 
  water_ratio = 3 / 8 ∧ required_volume = 0.64 → required_volume * water_ratio = 0.24 :=
by
  sorry

end water_required_to_prepare_saline_solution_l77_77061


namespace simplify_expression_l77_77151

def real_numbers (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^3 + b^3 = a^2 + b^2

theorem simplify_expression (a b : ℝ) (h : real_numbers a b) :
  (a^2 / b + b^2 / a - 1 / (a * a * b * b)) = (a^4 + 2 * a * b + b^4 - 1) / (a * b) :=
by
  sorry

end simplify_expression_l77_77151


namespace parallel_lines_sufficient_but_not_necessary_l77_77083

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 ↔ ((ax + y - 1 = 0) ∧ (x + ay + 1 = 0) → False)) := 
sorry

end parallel_lines_sufficient_but_not_necessary_l77_77083


namespace work_efficiency_ratio_l77_77622

theorem work_efficiency_ratio (A B : ℝ) (k : ℝ)
  (h1 : A = k * B)
  (h2 : B = 1 / 27)
  (h3 : A + B = 1 / 9) :
  k = 2 :=
by
  sorry

end work_efficiency_ratio_l77_77622


namespace smallest_n_for_coloring_l77_77177

theorem smallest_n_for_coloring (n : ℕ) : n = 4 :=
sorry

end smallest_n_for_coloring_l77_77177


namespace total_amount_is_4200_l77_77298

variables (p q r : ℕ)
variable (total_amount : ℕ)
variable (r_has_two_thirds : total_amount / 3 * 2 = 2800)
variable (r_value : r = 2800)

theorem total_amount_is_4200 (h1 : total_amount / 3 * 2 = 2800) (h2 : r = 2800) : total_amount = 4200 :=
by
  sorry

end total_amount_is_4200_l77_77298


namespace cases_in_1990_l77_77186

theorem cases_in_1990 (cases_1970 cases_2000 : ℕ) (linear_decrease : ℕ → ℝ) :
  cases_1970 = 300000 →
  cases_2000 = 600 →
  (∀ t, linear_decrease t = cases_1970 - (cases_1970 - cases_2000) * t / 30) →
  linear_decrease 20 = 100400 :=
by
  intros h1 h2 h3
  sorry

end cases_in_1990_l77_77186


namespace venue_cost_correct_l77_77676

noncomputable def cost_per_guest : ℤ := 500
noncomputable def johns_guests : ℤ := 50
noncomputable def wifes_guests : ℤ := johns_guests + (60 * johns_guests) / 100
noncomputable def total_wedding_cost : ℤ := 50000
noncomputable def guests_cost : ℤ := wifes_guests * cost_per_guest
noncomputable def venue_cost : ℤ := total_wedding_cost - guests_cost

theorem venue_cost_correct : venue_cost = 10000 := 
  by
  -- Proof can be filled in here.
  sorry

end venue_cost_correct_l77_77676


namespace average_salary_is_8000_l77_77097

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

def average_salary : ℕ := total_salary / num_people

theorem average_salary_is_8000 : average_salary = 8000 := by
  sorry

end average_salary_is_8000_l77_77097


namespace quadratic_no_real_roots_l77_77313

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 :=
by
  sorry

end quadratic_no_real_roots_l77_77313


namespace length_of_bridge_l77_77677

-- Define the conditions
def train_length : ℕ := 130 -- length of the train in meters
def train_speed : ℕ := 45  -- speed of the train in km/hr
def crossing_time : ℕ := 30  -- time to cross the bridge in seconds

-- Prove that the length of the bridge is 245 meters
theorem length_of_bridge : 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 245 := 
by
  sorry

end length_of_bridge_l77_77677


namespace tumblers_count_correct_l77_77537

section MrsPetersonsTumblers

-- Define the cost of one tumbler
def tumbler_cost : ℕ := 45

-- Define the amount paid in total by Mrs. Petersons
def total_paid : ℕ := 5 * 100

-- Define the change received by Mrs. Petersons
def change_received : ℕ := 50

-- Calculate the total amount spent
def total_spent : ℕ := total_paid - change_received

-- Calculate the number of tumblers bought
def tumblers_bought : ℕ := total_spent / tumbler_cost

-- Prove the number of tumblers bought is 10
theorem tumblers_count_correct : tumblers_bought = 10 :=
  by
    -- Proof steps will be filled here
    sorry

end MrsPetersonsTumblers

end tumblers_count_correct_l77_77537


namespace cookies_per_bag_l77_77985

theorem cookies_per_bag (b T : ℕ) (h1 : b = 37) (h2 : T = 703) : (T / b) = 19 :=
by
  -- Placeholder for proof
  sorry

end cookies_per_bag_l77_77985


namespace red_car_initial_distance_ahead_l77_77903

theorem red_car_initial_distance_ahead 
    (Speed_red Speed_black : ℕ) (Time : ℝ)
    (H1 : Speed_red = 10)
    (H2 : Speed_black = 50)
    (H3 : Time = 0.5) :
    let Distance_black := Speed_black * Time
    let Distance_red := Speed_red * Time
    Distance_black - Distance_red = 20 := 
by
  let Distance_black := Speed_black * Time
  let Distance_red := Speed_red * Time
  sorry

end red_car_initial_distance_ahead_l77_77903


namespace intersection_A_B_l77_77532

open Set

def isInSetA (x : ℕ) : Prop := ∃ n : ℕ, x = 3 * n + 2
def A : Set ℕ := { x | isInSetA x }
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_A_B :
  A ∩ B = {8, 14} :=
sorry

end intersection_A_B_l77_77532


namespace macey_needs_to_save_three_more_weeks_l77_77983

def cost_of_shirt : ℝ := 3.0
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

theorem macey_needs_to_save_three_more_weeks :
  ∃ W : ℝ, W * saving_per_week = cost_of_shirt - amount_saved ∧ W = 3 := by
  sorry

end macey_needs_to_save_three_more_weeks_l77_77983


namespace positive_real_number_solution_l77_77815

theorem positive_real_number_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 11) (h3 : (x - 6) / 11 = 6 / (x - 11)) : x = 17 :=
sorry

end positive_real_number_solution_l77_77815


namespace lucy_additional_kilometers_l77_77572

theorem lucy_additional_kilometers
  (mary_distance : ℚ := (3/8) * 24)
  (edna_distance : ℚ := (2/3) * mary_distance)
  (lucy_distance : ℚ := (5/6) * edna_distance) :
  (mary_distance - lucy_distance) = 4 :=
by
  sorry

end lucy_additional_kilometers_l77_77572


namespace percentage_difference_l77_77312

theorem percentage_difference (w x y z : ℝ) (h1 : w = 0.6 * x) (h2 : x = 0.6 * y) (h3 : z = 0.54 * y) : 
  ((z - w) / w) * 100 = 50 :=
by
  sorry

end percentage_difference_l77_77312


namespace a_minus_b_is_30_l77_77755

-- Definition of the sum of the arithmetic series
def sum_arithmetic_series (first last : ℕ) (n : ℕ) : ℕ :=
  (n * (first + last)) / 2

-- Definitions based on problem conditions
def a : ℕ := sum_arithmetic_series 2 60 30
def b : ℕ := sum_arithmetic_series 1 59 30

theorem a_minus_b_is_30 : a - b = 30 :=
  by sorry

end a_minus_b_is_30_l77_77755


namespace simplify_expression_l77_77344

theorem simplify_expression (x : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 9 = 45 * x + 27 :=
by
  sorry

end simplify_expression_l77_77344


namespace shaded_squares_percentage_l77_77164

theorem shaded_squares_percentage : 
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2
  (shaded_squares / total_squares) * 100 = 50 :=
by
  /- Definitions and conditions -/
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2

  /- Required proof statement -/
  have percentage_shaded : (shaded_squares / total_squares) * 100 = 50 := sorry

  /- Return the proof -/
  exact percentage_shaded

end shaded_squares_percentage_l77_77164


namespace sum_of_two_digit_integers_l77_77802

theorem sum_of_two_digit_integers :
  let a := 10
  let l := 99
  let d := 1
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = 4905 :=
by
  sorry

end sum_of_two_digit_integers_l77_77802


namespace geometric_sequence_solution_l77_77857

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 * q ^ (n - 1)

theorem geometric_sequence_solution {a : ℕ → ℝ} {q a1 : ℝ}
  (h1 : geometric_sequence a q a1)
  (h2 : a 3 + a 5 = 20)
  (h3 : a 4 = 8) :
  a 2 + a 6 = 34 := by
  sorry

end geometric_sequence_solution_l77_77857


namespace largest_number_l77_77485

theorem largest_number
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ)
  (ha : a = 0.883) (hb : b = 0.8839) (hc : c = 0.88) (hd : d = 0.839) (he : e = 0.889) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by {
  sorry
}

end largest_number_l77_77485


namespace factorization_of_x12_minus_4096_l77_77812

variable (x : ℝ)

theorem factorization_of_x12_minus_4096 : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end factorization_of_x12_minus_4096_l77_77812


namespace rectangle_area_l77_77845

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w^2 + (2 * w)^2 = x^2) : 
  2 * (w^2) = (2 / 5) * x^2 :=
by
  sorry

end rectangle_area_l77_77845


namespace people_in_room_l77_77839

theorem people_in_room (P C : ℚ) (H1 : (3 / 5) * P = (2 / 3) * C) (H2 : C / 3 = 5) : 
  P = 50 / 3 :=
by
  -- The proof would go here
  sorry

end people_in_room_l77_77839


namespace min_expression_value_l77_77045

open Real

-- Define the conditions given in the problem: x, y, z are positive reals and their product is 32
variables {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 32)

-- Define the expression that we want to find the minimum for: x^2 + 4xy + 4y^2 + 2z^2
def expression (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

-- State the theorem: proving that the minimum value of the expression given the conditions is 96
theorem min_expression_value : ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 32 ∧ expression x y z = 96 :=
sorry

end min_expression_value_l77_77045


namespace find_m_l77_77801

theorem find_m (x y m : ℝ) (h₁ : x - 2 * y = m) (h₂ : x = 2) (h₃ : y = 1) : m = 0 :=
by 
  -- Proof omitted
  sorry

end find_m_l77_77801


namespace goblins_return_l77_77529

theorem goblins_return (n : ℕ) (f : Fin n → Fin n) (h1 : ∀ a, ∃! b, f a = b) (h2 : ∀ b, ∃! a, f a = b) : 
  ∃ k : ℕ, ∀ x : Fin n, (f^[k]) x = x := 
sorry

end goblins_return_l77_77529


namespace least_common_multiple_increments_l77_77989

theorem least_common_multiple_increments :
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  Nat.lcm (Nat.lcm (Nat.lcm a' b') c') d' = 8645 :=
by
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  sorry

end least_common_multiple_increments_l77_77989


namespace set_C_cannot_form_right_triangle_l77_77906

theorem set_C_cannot_form_right_triangle :
  ¬(5^2 + 2^2 = 5^2) :=
by
  sorry

end set_C_cannot_form_right_triangle_l77_77906


namespace proof_problem_l77_77517

theorem proof_problem (p q : Prop) (hnpq : ¬ (p ∧ q)) (hnp : ¬ p) : ¬ p :=
by
  exact hnp

end proof_problem_l77_77517


namespace rounds_played_l77_77101

-- Define the given conditions as Lean constants
def totalPoints : ℝ := 378.5
def pointsPerRound : ℝ := 83.25

-- Define the goal as a Lean theorem
theorem rounds_played :
  Int.ceil (totalPoints / pointsPerRound) = 5 := 
by 
  sorry

end rounds_played_l77_77101


namespace geometric_sequence_r_value_l77_77775

theorem geometric_sequence_r_value (S : ℕ → ℚ) (r : ℚ) (n : ℕ) (h : n ≥ 2) (h1 : ∀ n, S n = 3^n + r) :
    r = -1 :=
sorry

end geometric_sequence_r_value_l77_77775


namespace distribution_ways_l77_77093

theorem distribution_ways :
  let friends := 12
  let problems := 6
  (friends ^ problems = 2985984) :=
by
  sorry

end distribution_ways_l77_77093


namespace line_passing_through_first_and_third_quadrants_l77_77678

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l77_77678


namespace sum_of_reciprocals_of_squares_l77_77986

open Real

theorem sum_of_reciprocals_of_squares {a b c : ℝ} (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = -7) (h3 : a * b * c = -2) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 73 / 4 :=
by
  sorry

end sum_of_reciprocals_of_squares_l77_77986


namespace f_at_2_is_neg_1_l77_77779

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

-- Given condition: f(-2) = 5
axiom h : ∀ (a b : ℝ), f a b (-2) = 5

-- Prove that f(2) = -1 given the above conditions
theorem f_at_2_is_neg_1 (a b : ℝ) (h_ab : f a b (-2) = 5) : f a b 2 = -1 := by
  sorry

end f_at_2_is_neg_1_l77_77779


namespace sugar_packs_l77_77100

variable (totalSugar : ℕ) (packWeight : ℕ) (sugarLeft : ℕ)

noncomputable def numberOfPacks (totalSugar packWeight sugarLeft : ℕ) : ℕ :=
  (totalSugar - sugarLeft) / packWeight

theorem sugar_packs : numberOfPacks 3020 250 20 = 12 := by
  sorry

end sugar_packs_l77_77100


namespace probability_of_c_between_l77_77824

noncomputable def probability_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : ℝ :=
  let c := a / (a + b)
  if (1 / 4 : ℝ) ≤ c ∧ c ≤ (3 / 4 : ℝ) then sorry else sorry
  
theorem probability_of_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : 
  probability_c_between a b hab = (2 / 3 : ℝ) :=
sorry

end probability_of_c_between_l77_77824


namespace probability_of_positive_l77_77106

-- Definitions based on the conditions
def balls : List ℚ := [-2, 0, 1/4, 3]
def total_balls : ℕ := 4
def positive_filter (x : ℚ) : Bool := x > 0
def positive_balls : List ℚ := balls.filter positive_filter
def positive_count : ℕ := positive_balls.length
def probability : ℚ := positive_count / total_balls

-- Statement to prove
theorem probability_of_positive : probability = 1 / 2 := by
  sorry

end probability_of_positive_l77_77106


namespace faster_pump_rate_ratio_l77_77988

theorem faster_pump_rate_ratio (S F : ℝ) 
  (h1 : S + F = 1/5) 
  (h2 : S = 1/12.5) : F / S = 1.5 :=
by
  sorry

end faster_pump_rate_ratio_l77_77988


namespace repeating_decimal_as_fraction_l77_77418

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l77_77418


namespace polynomial_square_b_value_l77_77453

theorem polynomial_square_b_value (a b : ℚ) (h : ∃ (p q : ℚ), x^4 + 3 * x^3 + x^2 + a * x + b = (x^2 + p * x + q)^2) : 
  b = 25/64 := 
by 
  -- Proof steps go here
  sorry

end polynomial_square_b_value_l77_77453


namespace total_revenue_l77_77012

theorem total_revenue (C A : ℕ) (P_C P_A total_tickets adult_tickets revenue : ℕ)
  (hCC : C = 6) -- Children's ticket price
  (hAC : A = 9) -- Adult's ticket price
  (hTT : total_tickets = 225) -- Total tickets sold
  (hAT : adult_tickets = 175) -- Adult tickets sold
  (hTR : revenue = 1875) -- Total revenue
  : revenue = adult_tickets * A + (total_tickets - adult_tickets) * C := sorry

end total_revenue_l77_77012


namespace not_prime_p_l77_77717

theorem not_prime_p (x k p : ℕ) (h : x^5 + 2 * x + 3 = p * k) : ¬ (Nat.Prime p) :=
by
  sorry -- Placeholder for the proof

end not_prime_p_l77_77717


namespace mail_sorting_time_l77_77838

theorem mail_sorting_time :
  (1 / (1 / 3 + 1 / 6) = 2) :=
by
  sorry

end mail_sorting_time_l77_77838


namespace solution_l77_77750

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l77_77750


namespace percentage_to_pass_is_correct_l77_77689

-- Define the conditions
def marks_obtained : ℕ := 130
def marks_failed_by : ℕ := 14
def max_marks : ℕ := 400

-- Define the function to calculate the passing percentage
def passing_percentage (obtained : ℕ) (failed_by : ℕ) (max : ℕ) : ℚ :=
  ((obtained + failed_by : ℕ) / (max : ℚ)) * 100

-- Statement of the problem
theorem percentage_to_pass_is_correct :
  passing_percentage marks_obtained marks_failed_by max_marks = 36 := 
sorry

end percentage_to_pass_is_correct_l77_77689


namespace total_cost_of_items_is_correct_l77_77047

theorem total_cost_of_items_is_correct :
  ∀ (M R F : ℝ),
  (10 * M = 24 * R) →
  (F = 2 * R) →
  (F = 24) →
  (4 * M + 3 * R + 5 * F = 271.2) :=
by
  intros M R F h1 h2 h3
  sorry

end total_cost_of_items_is_correct_l77_77047


namespace cars_in_group_l77_77670

open Nat

theorem cars_in_group (C : ℕ) : 
  (47 ≤ C) →                  -- At least 47 cars in the group
  (53 ≤ C) →                  -- At least 53 cars in the group
  C ≥ 100 :=                  -- Conclusion: total cars is at least 100
by
  -- Begin the proof
  sorry                       -- Skip proof for now

end cars_in_group_l77_77670


namespace arun_weight_l77_77930

theorem arun_weight (W B : ℝ) (h1 : 65 < W ∧ W < 72) (h2 : B < W ∧ W < 70) (h3 : W ≤ 68) (h4 : (B + 68) / 2 = 67) : B = 66 :=
sorry

end arun_weight_l77_77930


namespace part1_l77_77698

theorem part1 (m : ℕ) (n : ℕ) (h1 : m = 6 * 10 ^ n + m / 25) : ∃ i : ℕ, m = 625 * 10 ^ (3 * i) := sorry

end part1_l77_77698


namespace two_integers_divide_2_pow_96_minus_1_l77_77148

theorem two_integers_divide_2_pow_96_minus_1 : 
  ∃ a b : ℕ, (60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧ a ≠ b ∧ a ∣ (2^96 - 1) ∧ b ∣ (2^96 - 1) ∧ a = 63 ∧ b = 65) := 
sorry

end two_integers_divide_2_pow_96_minus_1_l77_77148


namespace cos_neg_13pi_over_4_l77_77383

theorem cos_neg_13pi_over_4 : Real.cos (-13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_13pi_over_4_l77_77383


namespace find_real_a_l77_77794

open Complex

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem find_real_a (a : ℝ) (i : ℂ) (h_i : i = Complex.I) :
  pure_imaginary ((2 + i) * (a - (2 * i))) ↔ a = -1 :=
by
  sorry

end find_real_a_l77_77794


namespace problem_statement_l77_77929

-- Definitions of parallel and perpendicular predicates (should be axioms or definitions in the context)
-- For simplification, assume we have a space with lines and planes, with corresponding relations.

axiom Line : Type
axiom Plane : Type
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Plane → Prop
axiom subset : Line → Plane → Prop

-- Assume the necessary conditions: m and n are lines, a and b are planes, with given relationships.
variables (m n : Line) (a b : Plane)

-- The conditions given.
variables (m_parallel_n : parallel m n)
variables (m_perpendicular_a : perpendicular m a)

-- The proposition to prove: If m parallel n and m perpendicular to a, then n is perpendicular to a.
theorem problem_statement : perpendicular n a :=
sorry

end problem_statement_l77_77929


namespace fraction_equality_solution_l77_77304

theorem fraction_equality_solution (x : ℝ) : (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 :=
by
  intro h
  sorry

end fraction_equality_solution_l77_77304


namespace cubic_poly_l77_77748

noncomputable def q (x : ℝ) : ℝ := - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3)

theorem cubic_poly:
  ( ∃ (a b c d : ℝ), 
    (∀ x : ℝ, q x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    ∧ q 1 = -6
    ∧ q 2 = -8
    ∧ q 3 = -14
    ∧ q 4 = -28
  ) → 
  q x = - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3) := 
sorry

end cubic_poly_l77_77748


namespace harry_pencils_remaining_l77_77513

def num_pencils_anna : ℕ := 50
def num_pencils_harry_initial := 2 * num_pencils_anna
def num_pencils_lost_harry := 19

def pencils_left_harry (pencils_anna : ℕ) (pencils_harry_initial : ℕ) (pencils_lost : ℕ) : ℕ :=
  pencils_harry_initial - pencils_lost

theorem harry_pencils_remaining : pencils_left_harry num_pencils_anna num_pencils_harry_initial num_pencils_lost_harry = 81 :=
by
  sorry

end harry_pencils_remaining_l77_77513


namespace age_difference_l77_77289

variable (A B C : ℕ)

-- Conditions
def ages_total_condition (a b c : ℕ) : Prop :=
  a + b = b + c + 11

-- Proof problem statement
theorem age_difference (a b c : ℕ) (h : ages_total_condition a b c) : a - c = 11 :=
by
  sorry

end age_difference_l77_77289


namespace margin_expression_l77_77271

variable (C S M : ℝ)
variable (n : ℕ)

theorem margin_expression (h : M = (C + S) / n) : M = (2 * S) / (n + 1) :=
sorry

end margin_expression_l77_77271


namespace cube_negative_iff_l77_77525

theorem cube_negative_iff (x : ℝ) : x < 0 ↔ x^3 < 0 :=
sorry

end cube_negative_iff_l77_77525


namespace unknown_diagonal_length_l77_77254

noncomputable def rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem unknown_diagonal_length
  (area : ℝ) (d2 : ℝ) (h_area : area = 150)
  (h_d2 : d2 = 30) :
  rhombus_diagonal_length area d2 = 10 :=
  by
  rw [h_area, h_d2]
  -- Here, the essential proof would go
  -- Since solving would require computation,
  -- which we are omitting, we use:
  sorry

end unknown_diagonal_length_l77_77254


namespace max_distinct_fans_l77_77422

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l77_77422


namespace find_multiplier_l77_77367

theorem find_multiplier (x y: ℤ) (h1: x = 127)
  (h2: x * y - 152 = 102): y = 2 :=
by
  sorry

end find_multiplier_l77_77367


namespace sticker_count_l77_77870

def stickers_per_page : ℕ := 25
def num_pages : ℕ := 35
def total_stickers : ℕ := 875

theorem sticker_count : num_pages * stickers_per_page = total_stickers :=
by {
  sorry
}

end sticker_count_l77_77870


namespace used_mystery_books_l77_77987

theorem used_mystery_books (total_books used_adventure_books new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13.0)
  (h3 : new_crime_books = 15.0) :
  total_books - (used_adventure_books + new_crime_books) = 17.0 := by
  sorry

end used_mystery_books_l77_77987


namespace correct_operation_l77_77785

theorem correct_operation (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by sorry

end correct_operation_l77_77785


namespace average_weight_all_children_l77_77433

theorem average_weight_all_children (avg_boys_weight avg_girls_weight : ℝ) (num_boys num_girls : ℕ)
    (hb : avg_boys_weight = 155) (nb : num_boys = 8)
    (hg : avg_girls_weight = 125) (ng : num_girls = 7) :
    (num_boys + num_girls = 15) → (avg_boys_weight * num_boys + avg_girls_weight * num_girls) / (num_boys + num_girls) = 141 := by
  intro h_sum
  sorry

end average_weight_all_children_l77_77433


namespace find_number_of_rabbits_l77_77059

variable (R P : ℕ)

theorem find_number_of_rabbits (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : R = 36 := 
by
  sorry

end find_number_of_rabbits_l77_77059


namespace betty_afternoon_catch_l77_77749

def flies_eaten_per_day := 2
def days_in_week := 7
def flies_needed_for_week := days_in_week * flies_eaten_per_day
def flies_caught_morning := 5
def additional_flies_needed := 4
def flies_currently_have := flies_needed_for_week - additional_flies_needed
def flies_caught_afternoon := flies_currently_have - flies_caught_morning
def flies_escaped := 1

theorem betty_afternoon_catch :
  flies_caught_afternoon + flies_escaped = 6 :=
by
  sorry

end betty_afternoon_catch_l77_77749


namespace find_certain_number_l77_77958

theorem find_certain_number (x : ℝ) (h : 25 * x = 675) : x = 27 :=
by {
  sorry
}

end find_certain_number_l77_77958


namespace find_d_l77_77348

-- Define the proportional condition
def in_proportion (a b c d : ℕ) : Prop := a * d = b * c

-- Given values as parameters
variables {a b c d : ℕ}

-- Theorem to be proven
theorem find_d (h : in_proportion a b c d) (ha : a = 1) (hb : b = 2) (hc : c = 3) : d = 6 :=
sorry

end find_d_l77_77348


namespace possible_values_of_ABCD_l77_77835

noncomputable def discriminant (a b c : ℕ) : ℕ :=
  b^2 - 4*a*c

theorem possible_values_of_ABCD 
  (A B C D : ℕ)
  (AB BC CD : ℕ)
  (hAB : AB = 10*A + B)
  (hBC : BC = 10*B + C)
  (hCD : CD = 10*C + D)
  (h_no_9 : A ≠ 9 ∧ B ≠ 9 ∧ C ≠ 9 ∧ D ≠ 9)
  (h_leading_nonzero : A ≠ 0)
  (h_quad1 : discriminant A B CD ≥ 0)
  (h_quad2 : discriminant A BC D ≥ 0)
  (h_quad3 : discriminant AB C D ≥ 0) :
  ABCD = 1710 ∨ ABCD = 1810 :=
sorry

end possible_values_of_ABCD_l77_77835


namespace annes_initial_bottle_caps_l77_77881

-- Define the conditions
def albert_bottle_caps : ℕ := 9
def annes_added_bottle_caps : ℕ := 5
def annes_total_bottle_caps : ℕ := 15

-- Question (to prove)
theorem annes_initial_bottle_caps :
  annes_total_bottle_caps - annes_added_bottle_caps = 10 :=
by sorry

end annes_initial_bottle_caps_l77_77881


namespace range_of_a_l77_77495

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l77_77495


namespace initial_average_age_l77_77948

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 9) (h2 : (n * A + 35) / (n + 1) = 17) :
  A = 15 :=
by
  sorry

end initial_average_age_l77_77948


namespace roots_poly_cond_l77_77609

theorem roots_poly_cond (α β p q γ δ : ℝ) 
  (h1 : α ^ 2 + p * α - 1 = 0) 
  (h2 : β ^ 2 + p * β - 1 = 0) 
  (h3 : γ ^ 2 + q * γ - 1 = 0) 
  (h4 : δ ^ 2 + q * δ - 1 = 0)
  (h5 : γ * δ = -1) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = -(p - q) ^ 2 := 
by 
  sorry

end roots_poly_cond_l77_77609


namespace average_grade_of_female_students_l77_77292

theorem average_grade_of_female_students
  (avg_all_students : ℝ)
  (avg_male_students : ℝ)
  (num_males : ℕ)
  (num_females : ℕ)
  (total_students := num_males + num_females)
  (total_score_all_students := avg_all_students * total_students)
  (total_score_male_students := avg_male_students * num_males) :
  avg_all_students = 90 →
  avg_male_students = 87 →
  num_males = 8 →
  num_females = 12 →
  ((total_score_all_students - total_score_male_students) / num_females) = 92 := by
  intros h_avg_all h_avg_male h_num_males h_num_females
  sorry

end average_grade_of_female_students_l77_77292


namespace graph_of_equation_l77_77250

theorem graph_of_equation {x y : ℝ} (h : (x - 2 * y)^2 = x^2 - 4 * y^2) :
  (y = 0) ∨ (x = 2 * y) :=
by
  sorry

end graph_of_equation_l77_77250


namespace solve_for_x_l77_77414

theorem solve_for_x :
  ∀ x : ℚ, 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) → x = 22 / 5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l77_77414


namespace problem_solution_l77_77885

noncomputable def solveSystem : Prop :=
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ),
    (x1 + x2 + x3 = 6) ∧
    (x2 + x3 + x4 = 9) ∧
    (x3 + x4 + x5 = 3) ∧
    (x4 + x5 + x6 = -3) ∧
    (x5 + x6 + x7 = -9) ∧
    (x6 + x7 + x8 = -6) ∧
    (x7 + x8 + x1 = -2) ∧
    (x8 + x1 + x2 = 2) ∧
    (x1 = 1) ∧
    (x2 = 2) ∧
    (x3 = 3) ∧
    (x4 = 4) ∧
    (x5 = -4) ∧
    (x6 = -3) ∧
    (x7 = -2) ∧
    (x8 = -1)

theorem problem_solution : solveSystem :=
by
  -- Skip the proof for now
  sorry

end problem_solution_l77_77885


namespace range_of_a_l77_77182

-- Definitions for the conditions
def p (x : ℝ) := x ≤ 2
def q (x : ℝ) (a : ℝ) := x < a + 2

-- Theorem statement
theorem range_of_a (a : ℝ) : (∀ x : ℝ, q x a → p x) → a ≤ 0 := by
  sorry

end range_of_a_l77_77182


namespace roots_polynomial_sum_pow_l77_77569

open Real

theorem roots_polynomial_sum_pow (a b : ℝ) (h : a^2 - 5 * a + 6 = 0) (h_b : b^2 - 5 * b + 6 = 0) :
  a^5 + a^4 * b + b^5 = -16674 := by
sorry

end roots_polynomial_sum_pow_l77_77569


namespace base12_remainder_l77_77708

theorem base12_remainder (x : ℕ) (h : x = 2 * 12^3 + 7 * 12^2 + 4 * 12 + 5) : x % 5 = 2 :=
by {
    -- Proof would go here
    sorry
}

end base12_remainder_l77_77708


namespace visits_exactly_two_friends_l77_77393

theorem visits_exactly_two_friends (a_visits b_visits c_visits vacation_period : ℕ) (full_period days : ℕ)
(h_a : a_visits = 4)
(h_b : b_visits = 5)
(h_c : c_visits = 6)
(h_vacation : vacation_period = 30)
(h_full_period : full_period = Nat.lcm (Nat.lcm a_visits b_visits) c_visits)
(h_days : days = 360)
(h_start_vacation : ∀ n, ∃ k, n = k * vacation_period + 30):
  ∃ n, n = 24 :=
by {
  sorry
}

end visits_exactly_two_friends_l77_77393


namespace factor_x10_minus_1024_l77_77162

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end factor_x10_minus_1024_l77_77162


namespace maxProfitAchievable_l77_77562

namespace BarrelProduction

structure ProductionPlan where
  barrelsA : ℕ
  barrelsB : ℕ

def profit (plan : ProductionPlan) : ℕ :=
  300 * plan.barrelsA + 400 * plan.barrelsB

def materialAUsage (plan : ProductionPlan) : ℕ :=
  plan.barrelsA + 2 * plan.barrelsB

def materialBUsage (plan : ProductionPlan) : ℕ :=
  2 * plan.barrelsA + plan.barrelsB

def isValidPlan (plan : ProductionPlan) : Prop :=
  materialAUsage plan ≤ 12 ∧ materialBUsage plan ≤ 12

def maximumProfit : ℕ :=
  2800

theorem maxProfitAchievable : 
  ∃ (plan : ProductionPlan), isValidPlan plan ∧ profit plan = maximumProfit :=
sorry

end BarrelProduction

end maxProfitAchievable_l77_77562


namespace cookies_taken_in_four_days_l77_77768

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end cookies_taken_in_four_days_l77_77768


namespace f_zero_f_odd_range_of_x_l77_77192

variable {f : ℝ → ℝ}

axiom func_property (x y : ℝ) : f (x + y) = f x + f y
axiom f_third : f (1 / 3) = 1
axiom f_positive (x : ℝ) : x > 0 → f x > 0

-- Part (1)
theorem f_zero : f 0 = 0 :=
sorry

-- Part (2)
theorem f_odd (x : ℝ) : f (-x) = -f x :=
sorry

-- Part (3)
theorem range_of_x (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 :=
sorry

end f_zero_f_odd_range_of_x_l77_77192


namespace inequality_proof_l77_77084

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := 
  sorry

end inequality_proof_l77_77084


namespace scientific_notation_of_61345_05_billion_l77_77288

theorem scientific_notation_of_61345_05_billion :
  ∃ x : ℝ, (61345.05 * 10^9) = x ∧ x = 6.134505 * 10^12 :=
by
  sorry

end scientific_notation_of_61345_05_billion_l77_77288


namespace geometric_sum_ratio_l77_77168

theorem geometric_sum_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (1 - q^4) / (1 - q^2) = 5) :
  (1 - q^8) / (1 - q^4) = 17 := 
by
  sorry

end geometric_sum_ratio_l77_77168


namespace total_items_sold_at_garage_sale_l77_77259

-- Define the conditions for the problem
def items_more_expensive_than_radio : Nat := 16
def items_less_expensive_than_radio : Nat := 23

-- Declare the total number of items using the given conditions
theorem total_items_sold_at_garage_sale 
  (h1 : items_more_expensive_than_radio = 16)
  (h2 : items_less_expensive_than_radio = 23) :
  items_more_expensive_than_radio + 1 + items_less_expensive_than_radio = 40 :=
by
  sorry

end total_items_sold_at_garage_sale_l77_77259


namespace range_of_a_odd_not_even_l77_77410

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def A : Set ℝ := Set.Ioo (-1 : ℝ) 1

def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

theorem range_of_a (a : ℝ) (h1 : B a ⊆ A) : -1 ≤ a ∧ a ≤ 0 := by
  sorry

theorem odd_not_even : (∀ x ∈ A, f (-x) = - f x) ∧ ¬ (∀ x ∈ A, f x = f (-x)) := by
  sorry

end range_of_a_odd_not_even_l77_77410


namespace best_years_to_scrap_l77_77564

-- Define the conditions from the problem
def purchase_cost : ℕ := 150000
def annual_cost : ℕ := 15000
def maintenance_initial : ℕ := 3000
def maintenance_difference : ℕ := 3000

-- Define the total_cost function
def total_cost (n : ℕ) : ℕ :=
  purchase_cost + annual_cost * n + (n * (2 * maintenance_initial + (n - 1) * maintenance_difference)) / 2

-- Define the average annual cost function
def average_annual_cost (n : ℕ) : ℕ :=
  total_cost n / n

-- Statement to be proven: the best number of years to minimize average annual cost is 10
theorem best_years_to_scrap : 
  (∀ n : ℕ, average_annual_cost 10 ≤ average_annual_cost n) :=
by
  sorry
  
end best_years_to_scrap_l77_77564


namespace fraction_paint_used_second_week_l77_77593

noncomputable def total_paint : ℕ := 360
noncomputable def paint_used_first_week : ℕ := total_paint / 4
noncomputable def remaining_paint_after_first_week : ℕ := total_paint - paint_used_first_week
noncomputable def total_paint_used : ℕ := 135
noncomputable def paint_used_second_week : ℕ := total_paint_used - paint_used_first_week
noncomputable def remaining_paint_after_first_week_fraction : ℚ := paint_used_second_week / remaining_paint_after_first_week

theorem fraction_paint_used_second_week : remaining_paint_after_first_week_fraction = 1 / 6 := by
  sorry

end fraction_paint_used_second_week_l77_77593


namespace combine_octahedrons_tetrahedrons_to_larger_octahedron_l77_77895

theorem combine_octahedrons_tetrahedrons_to_larger_octahedron (edge : ℝ) :
  ∃ (octahedrons : ℕ) (tetrahedrons : ℕ),
    octahedrons = 6 ∧ tetrahedrons = 8 ∧
    (∃ (new_octahedron_edge : ℝ), new_octahedron_edge = 2 * edge) :=
by {
  -- The proof will construct the larger octahedron
  sorry
}

end combine_octahedrons_tetrahedrons_to_larger_octahedron_l77_77895


namespace ahmed_total_distance_l77_77787

theorem ahmed_total_distance (d : ℝ) (h : (3 / 4) * d = 12) : d = 16 := 
by 
  sorry

end ahmed_total_distance_l77_77787


namespace additional_hours_to_travel_l77_77330

theorem additional_hours_to_travel (distance1 time1 distance2 : ℝ) (rate : ℝ) 
  (h1 : distance1 = 270) 
  (h2 : time1 = 3)
  (h3 : distance2 = 180)
  (h4 : rate = distance1 / time1) :
  distance2 / rate = 2 := by
  sorry

end additional_hours_to_travel_l77_77330


namespace number_of_yellow_balls_l77_77002

theorem number_of_yellow_balls (x : ℕ) :
  (4 : ℕ) / (4 + x) = 2 / 3 → x = 2 :=
by
  sorry

end number_of_yellow_balls_l77_77002


namespace minimum_bats_examined_l77_77038

theorem minimum_bats_examined 
  (bats : Type) 
  (R L : bats → Prop) 
  (total_bats : ℕ)
  (right_eye_bats : ∀ {b: bats}, R b → Fin 2)
  (left_eye_bats : ∀ {b: bats}, L b → Fin 3)
  (not_left_eye_bats: ∀ {b: bats}, ¬ L b → Fin 4)
  (not_right_eye_bats: ∀ {b: bats}, ¬ R b → Fin 5)
  : total_bats ≥ 7 := sorry

end minimum_bats_examined_l77_77038


namespace number_of_boys_l77_77419

variable {total_marbles : ℕ} (marbles_per_boy : ℕ := 10)
variable (H_total_marbles : total_marbles = 20)

theorem number_of_boys (total_marbles_marbs_eq_20 : total_marbles = 20) (marbles_per_boy_eq_10 : marbles_per_boy = 10) :
  total_marbles / marbles_per_boy = 2 :=
by {
  sorry
}

end number_of_boys_l77_77419


namespace ice_cream_stack_order_l77_77237

theorem ice_cream_stack_order (scoops : Finset ℕ) (h_scoops : scoops.card = 5) :
  (scoops.prod id) = 120 :=
by
  sorry

end ice_cream_stack_order_l77_77237


namespace xiaoli_estimate_larger_l77_77804

variable (x y z w : ℝ)
variable (hxy : x > y) (hy0 : y > 0) (hz1 : z > 1) (hw0 : w > 0)

theorem xiaoli_estimate_larger : (x + w) - (y - w) * z > x - y * z :=
by sorry

end xiaoli_estimate_larger_l77_77804


namespace correct_operation_l77_77245

variable (a : ℝ)

theorem correct_operation : 
  (3 * a^2 + 2 * a^4 ≠ 5 * a^6) ∧
  (a^2 * a^3 ≠ a^6) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) ∧
  ((-2 * a^3)^2 = 4 * a^6) := by
  sorry

end correct_operation_l77_77245


namespace find_speeds_l77_77339

theorem find_speeds 
  (x v u : ℝ)
  (hx : x = u / 4)
  (hv : 0 < v)
  (hu : 0 < u)
  (t_car : 30 / v + 1.25 = 30 / x)
  (meeting_cars : 0.05 * v + 0.05 * u = 5) :
  x = 15 ∧ v = 40 ∧ u = 60 :=
by 
  sorry

end find_speeds_l77_77339


namespace gcd_40304_30203_eq_1_l77_77454

theorem gcd_40304_30203_eq_1 : Nat.gcd 40304 30203 = 1 := 
by 
  sorry

end gcd_40304_30203_eq_1_l77_77454


namespace fourth_person_height_l77_77827

-- Definitions based on conditions
def h1 : ℕ := 73  -- height of first person
def h2 : ℕ := h1 + 2  -- height of second person
def h3 : ℕ := h2 + 2  -- height of third person
def h4 : ℕ := h3 + 6  -- height of fourth person

theorem fourth_person_height : h4 = 83 :=
by
  -- calculation to check the average height and arriving at h1
  -- (all detailed calculations are skipped using "sorry")
  sorry

end fourth_person_height_l77_77827


namespace find_X_l77_77258

theorem find_X (X : ℝ) (h : 0.80 * X - 0.35 * 300 = 31) : X = 170 :=
by
  sorry

end find_X_l77_77258


namespace initial_bees_l77_77357

theorem initial_bees (B : ℕ) (h : B + 8 = 24) : B = 16 := 
by {
  sorry
}

end initial_bees_l77_77357


namespace b_money_used_for_10_months_l77_77583

theorem b_money_used_for_10_months
  (a_capital_ratio : ℚ)
  (a_time_used : ℕ)
  (b_profit_share : ℚ)
  (h1 : a_capital_ratio = 1 / 4)
  (h2 : a_time_used = 15)
  (h3 : b_profit_share = 2 / 3) :
  ∃ (b_time_used : ℕ), b_time_used = 10 :=
by
  sorry

end b_money_used_for_10_months_l77_77583


namespace symmetric_line_equation_l77_77190

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end symmetric_line_equation_l77_77190


namespace inequality_proof_l77_77810

theorem inequality_proof (x : ℝ) (hx : x ≥ 1) : x^5 - 1 / x^4 ≥ 9 * (x - 1) := 
by sorry

end inequality_proof_l77_77810


namespace correct_statement_is_c_l77_77711

-- Definitions corresponding to conditions
def lateral_surface_of_cone_unfolds_into_isosceles_triangle : Prop :=
  false -- This is false because it unfolds into a sector.

def prism_with_two_congruent_bases_other_faces_rectangles : Prop :=
  false -- This is false because the bases are congruent and parallel, and all other faces are parallelograms.

def frustum_complemented_with_pyramid_forms_new_pyramid : Prop :=
  true -- This is true, as explained in the solution.

def point_on_lateral_surface_of_truncated_cone_has_countless_generatrices : Prop :=
  false -- This is false because there is exactly one generatrix through such a point.

-- The main proof statement
theorem correct_statement_is_c :
  ¬lateral_surface_of_cone_unfolds_into_isosceles_triangle ∧
  ¬prism_with_two_congruent_bases_other_faces_rectangles ∧
  frustum_complemented_with_pyramid_forms_new_pyramid ∧
  ¬point_on_lateral_surface_of_truncated_cone_has_countless_generatrices :=
by
  -- The proof involves evaluating all the conditions above.
  sorry

end correct_statement_is_c_l77_77711


namespace tangent_parallel_l77_77700

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the slope of the line 4x - y - 1 = 0, which is 4
def line_slope : ℝ := 4

-- The main theorem statement
theorem tangent_parallel (a b : ℝ) (h1 : f a = b) (h2 : f' a = line_slope) :
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -4) :=
sorry

end tangent_parallel_l77_77700


namespace percent_round_trip_tickets_is_100_l77_77226

noncomputable def percent_round_trip_tickets (P : ℕ) (x : ℚ) : ℚ :=
  let R := x / 0.20
  R

theorem percent_round_trip_tickets_is_100
  (P : ℕ)
  (x : ℚ)
  (h : 20 * x = P) :
  percent_round_trip_tickets P (x / P) = 100 :=
by
  sorry

end percent_round_trip_tickets_is_100_l77_77226


namespace geometric_sequence_S_n_l77_77364

-- Definitions related to the sequence
def a_n (n : ℕ) : ℕ := sorry  -- Placeholder for the actual sequence

-- Sum of the first n terms
def S_n (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms

-- Given conditions
axiom a1 : a_n 1 = 1
axiom Sn_eq_2an_plus1 : ∀ (n : ℕ), S_n n = 2 * a_n (n + 1)

-- Theorem to be proved
theorem geometric_sequence_S_n 
    (n : ℕ) (h : n > 1) 
    : S_n n = (3/2)^(n-1) := 
by 
  sorry

end geometric_sequence_S_n_l77_77364


namespace find_b_of_triangle_ABC_l77_77766

theorem find_b_of_triangle_ABC (a b c : ℝ) (cos_A : ℝ) 
  (h1 : a = 2) 
  (h2 : c = 2 * Real.sqrt 3) 
  (h3 : cos_A = Real.sqrt 3 / 2) 
  (h4 : b < c) : 
  b = 2 := 
by
  sorry

end find_b_of_triangle_ABC_l77_77766


namespace find_m_l77_77044

-- Definitions for the lines and the condition of parallelism
def line1 (m : ℝ) (x y : ℝ): Prop := x + m * y + 6 = 0
def line2 (m : ℝ) (x y : ℝ): Prop := 3 * x + (m - 2) * y + 2 * m = 0

-- Condition for lines being parallel
def parallel_lines (m : ℝ) : Prop := 1 * (m - 2) - 3 * m = 0

-- Main formal statement
theorem find_m (m : ℝ) (h1 : ∀ x y, line1 m x y)
                (h2 : ∀ x y, line2 m x y)
                (h_parallel : parallel_lines m) : m = -1 :=
sorry

end find_m_l77_77044


namespace range_of_a_l77_77974

noncomputable def A : Set ℝ := {x | x ≥ abs (x^2 - 2 * x)}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l77_77974


namespace number_corresponding_to_8_minutes_l77_77109

theorem number_corresponding_to_8_minutes (x : ℕ) : 
  (12 / 6 = x / 480) → x = 960 :=
by
  sorry

end number_corresponding_to_8_minutes_l77_77109


namespace pyramid_volume_l77_77780

noncomputable def volume_of_pyramid (S α β : ℝ) : ℝ :=
  (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β)))

theorem pyramid_volume 
  (S α β : ℝ)
  (base_area : S > 0)
  (equal_lateral_edges : true)
  (dihedral_angles : α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2) :
  volume_of_pyramid S α β = (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β))) :=
by
  sorry

end pyramid_volume_l77_77780


namespace sum_of_powers_pattern_l77_77753

theorem sum_of_powers_pattern :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 :=
  sorry

end sum_of_powers_pattern_l77_77753


namespace log_eq_15_given_log_base3_x_eq_5_l77_77068

variable (x : ℝ)
variable (log_base3_x : ℝ)
variable (h : log_base3_x = 5)

theorem log_eq_15_given_log_base3_x_eq_5 (h : log_base3_x = 5) : log_base3_x * 3 = 15 :=
by
  sorry

end log_eq_15_given_log_base3_x_eq_5_l77_77068


namespace minimum_ab_l77_77287

theorem minimum_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : ab + 2 = 2 * (a + b)) : ab ≥ 6 + 4 * Real.sqrt 2 :=
by
  sorry

end minimum_ab_l77_77287


namespace num_solutions_eq_4_l77_77391

theorem num_solutions_eq_4 (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  ∃ n : ℕ, n = 4 ∧ (2 + 4 * Real.cos θ - 6 * Real.sin (2 * θ) + 3 * Real.tan θ = 0) :=
sorry

end num_solutions_eq_4_l77_77391


namespace arithmetic_series_remainder_l77_77639

noncomputable def arithmetic_series_sum_mod (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d) / 2) % 10

theorem arithmetic_series_remainder :
  let a := 3
  let d := 5
  let n := 21
  arithmetic_series_sum_mod a d n = 3 :=
by
  sorry

end arithmetic_series_remainder_l77_77639


namespace percentage_increase_first_year_l77_77077

theorem percentage_increase_first_year (P : ℝ) (x : ℝ) :
  (1 + x / 100) * 0.7 = 1.0499999999999998 → x = 50 := 
by
  sorry

end percentage_increase_first_year_l77_77077


namespace square_of_neg_three_l77_77962

theorem square_of_neg_three : (-3 : ℤ)^2 = 9 := by
  sorry

end square_of_neg_three_l77_77962


namespace work_completion_l77_77966

theorem work_completion 
  (x_work_days : ℕ) 
  (y_work_days : ℕ) 
  (y_worked_days : ℕ) 
  (x_rate := 1 / (x_work_days : ℚ)) 
  (y_rate := 1 / (y_work_days : ℚ)) 
  (work_remaining := 1 - y_rate * y_worked_days) 
  (remaining_work_days := work_remaining / x_rate) : 
  x_work_days = 18 → 
  y_work_days = 15 → 
  y_worked_days = 5 → 
  remaining_work_days = 12 := 
by
  intros
  sorry

end work_completion_l77_77966


namespace quadratic_has_two_distinct_real_roots_l77_77244

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l77_77244


namespace oil_output_per_capita_l77_77159

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end oil_output_per_capita_l77_77159


namespace ferris_wheel_cost_l77_77278

theorem ferris_wheel_cost (roller_coaster_cost log_ride_cost zach_initial_tickets zach_additional_tickets total_tickets ferris_wheel_cost : ℕ) 
  (h1 : roller_coaster_cost = 7)
  (h2 : log_ride_cost = 1)
  (h3 : zach_initial_tickets = 1)
  (h4 : zach_additional_tickets = 9)
  (h5 : total_tickets = zach_initial_tickets + zach_additional_tickets)
  (h6 : total_tickets - (roller_coaster_cost + log_ride_cost) = ferris_wheel_cost) :
  ferris_wheel_cost = 2 := 
by
  sorry

end ferris_wheel_cost_l77_77278


namespace unique_positive_real_solution_l77_77005

theorem unique_positive_real_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) : x = 1 ∧ y = 1 ∧ z = 1 :=
sorry

end unique_positive_real_solution_l77_77005


namespace maximum_side_length_range_l77_77922

variable (P : ℝ)
variable (a b c : ℝ)
variable (h1 : a + b + c = P)
variable (h2 : a ≤ b)
variable (h3 : b ≤ c)
variable (h4 : a + b > c)

theorem maximum_side_length_range : 
  (P / 3) ≤ c ∧ c < (P / 2) :=
by
  sorry

end maximum_side_length_range_l77_77922


namespace no_valid_height_configuration_l77_77133

-- Define the heights and properties
variables {a : Fin 7 → ℝ}
variables {p : ℝ}

-- Define the condition as a theorem
theorem no_valid_height_configuration (h : ∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                                         p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) :
  ¬ (∃ (a : Fin 7 → ℝ), 
    (∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                  p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) ∧
    true) :=
sorry

end no_valid_height_configuration_l77_77133


namespace find_principal_amount_l77_77058

-- Define the parameters
def R : ℝ := 11.67
def T : ℝ := 5
def A : ℝ := 950

-- State the theorem
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + (R/100) * T) :=
by { 
  use 600, 
  -- Skip the proof 
  sorry 
}

end find_principal_amount_l77_77058


namespace fourth_root_sum_of_square_roots_eq_l77_77575

theorem fourth_root_sum_of_square_roots_eq :
  (1 + Real.sqrt 2 + Real.sqrt 3) = 
    Real.sqrt (Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608) ^ 4 :=
by
  sorry

end fourth_root_sum_of_square_roots_eq_l77_77575


namespace quotient_of_m_and_n_l77_77299

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem quotient_of_m_and_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) :
  n / m = Real.exp 2 :=
by
  sorry

end quotient_of_m_and_n_l77_77299


namespace dice_circle_probability_l77_77201

theorem dice_circle_probability :
  ∀ (d : ℕ), (2 ≤ d ∧ d ≤ 432) ∧
  ((∃ (x y : ℕ), (1 ≤ x ∧ x ≤ 6) ∧ (1 ≤ y ∧ y <= 6) ∧ d = x^3 + y^3)) →
  ((d * (d - 4) < 0) ↔ (d = 2)) →
  (∃ (P : ℚ), P = 1 / 36) :=
by
  sorry

end dice_circle_probability_l77_77201


namespace total_sticks_used_l77_77397

-- Definitions based on the conditions
def hexagons : Nat := 800
def sticks_for_first_hexagon : Nat := 6
def sticks_per_additional_hexagon : Nat := 5

-- The theorem to prove
theorem total_sticks_used :
  sticks_for_first_hexagon + (hexagons - 1) * sticks_per_additional_hexagon = 4001 := by
  sorry

end total_sticks_used_l77_77397


namespace find_third_number_l77_77144

theorem find_third_number 
  (h1 : (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3) : x = 53 := by
  sorry

end find_third_number_l77_77144


namespace gcd_105_90_l77_77428

theorem gcd_105_90 : Nat.gcd 105 90 = 15 :=
by
  sorry

end gcd_105_90_l77_77428


namespace problem_false_statements_l77_77267

noncomputable def statement_I : Prop :=
  ∀ x : ℝ, ⌊x + Real.pi⌋ = ⌊x⌋ + 3

noncomputable def statement_II : Prop :=
  ∀ x : ℝ, ⌊x + Real.sqrt 2⌋ = ⌊x⌋ + ⌊Real.sqrt 2⌋

noncomputable def statement_III : Prop :=
  ∀ x : ℝ, ⌊x * Real.pi⌋ = ⌊x⌋ * ⌊Real.pi⌋

theorem problem_false_statements : ¬(statement_I ∨ statement_II ∨ statement_III) := 
by
  sorry

end problem_false_statements_l77_77267


namespace a_in_s_l77_77943

-- Defining the sets and the condition
def S : Set ℕ := {1, 2}
def T (a : ℕ) : Set ℕ := {a}

-- The Lean theorem statement
theorem a_in_s (a : ℕ) (h : S ∪ T a = S) : a = 1 ∨ a = 2 := 
by 
  sorry

end a_in_s_l77_77943


namespace hannah_probability_12_flips_l77_77796

/-!
We need to prove that the probability of getting fewer than 4 heads when flipping 12 coins is 299/4096.
-/

def probability_fewer_than_4_heads (flips : ℕ) : ℚ :=
  let total_outcomes := 2^flips
  let favorable_outcomes := (Nat.choose flips 0) + (Nat.choose flips 1) + (Nat.choose flips 2) + (Nat.choose flips 3)
  favorable_outcomes / total_outcomes

theorem hannah_probability_12_flips : probability_fewer_than_4_heads 12 = 299 / 4096 := by
  sorry

end hannah_probability_12_flips_l77_77796


namespace find_ratio_l77_77686

theorem find_ratio (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
  P / (x + 6) + Q / (x * (x - 5)) = (x^2 - x + 15) / (x^3 + x^2 - 30 * x)) :
  Q / P = 5 / 6 := sorry

end find_ratio_l77_77686


namespace find_negative_integer_l77_77187

theorem find_negative_integer (M : ℤ) (h_neg : M < 0) (h_eq : M^2 + M = 12) : M = -4 :=
sorry

end find_negative_integer_l77_77187


namespace option_D_is_empty_l77_77774

theorem option_D_is_empty :
  {x : ℝ | x^2 + x + 1 = 0} = ∅ :=
by
  sorry

end option_D_is_empty_l77_77774


namespace total_cost_tom_pays_for_trip_l77_77341

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end total_cost_tom_pays_for_trip_l77_77341


namespace chord_line_equation_l77_77415

theorem chord_line_equation 
  (x y : ℝ)
  (ellipse_eq : x^2 / 4 + y^2 / 3 = 1)
  (midpoint_condition : ∃ x1 y1 x2 y2 : ℝ, (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1
   ∧ (x1^2 / 4 + y1^2 / 3 = 1) ∧ (x2^2 / 4 + y2^2 / 3 = 1))
  : 3 * x - 4 * y + 7 = 0 :=
sorry

end chord_line_equation_l77_77415


namespace cost_of_soccer_basketball_balls_max_basketballs_l77_77890

def cost_of_balls (x y : ℕ) : Prop :=
  (7 * x = 5 * y) ∧ (40 * x + 20 * y = 3400)

def cost_constraint (x y m : ℕ) : Prop :=
  (x = 50) ∧ (y = 70) ∧ (70 * m + 50 * (100 - m) ≤ 6300)

theorem cost_of_soccer_basketball_balls (x y : ℕ) (h : cost_of_balls x y) : x = 50 ∧ y = 70 :=
  by sorry

theorem max_basketballs (x y m : ℕ) (h : cost_constraint x y m) : m ≤ 65 :=
  by sorry

end cost_of_soccer_basketball_balls_max_basketballs_l77_77890


namespace find_k_l77_77693

-- Define the conditions
variables (a b : Real) (x y : Real)

-- The problem's conditions
def tan_x : Prop := Real.tan x = a / b
def tan_2x : Prop := Real.tan (x + x) = b / (a + b)
def y_eq_x : Prop := y = x

-- The goal to prove
theorem find_k (ha : tan_x a b x) (hb : tan_2x a b x) (hy : y_eq_x x y) :
  ∃ k, x = Real.arctan k ∧ k = 1 / (a + 2) :=
sorry

end find_k_l77_77693


namespace number_of_subsets_with_four_adj_chairs_l77_77542

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l77_77542


namespace term_of_arithmetic_sequence_l77_77577

variable (a₁ : ℕ) (d : ℕ) (n : ℕ)

theorem term_of_arithmetic_sequence (h₁: a₁ = 2) (h₂: d = 5) (h₃: n = 50) :
    a₁ + (n - 1) * d = 247 := by
  sorry

end term_of_arithmetic_sequence_l77_77577


namespace sqrt_inequality_l77_77075

theorem sqrt_inequality (n : ℕ) : 
  (n ≥ 0) → (Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n) := 
by
  intro h
  sorry

end sqrt_inequality_l77_77075


namespace probability_perfect_square_l77_77420

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def successful_outcomes : Finset ℕ := {1, 4}

def total_possible_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_perfect_square :
  (successful_outcomes.card : ℚ) / (total_possible_outcomes.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_perfect_square_l77_77420


namespace find_function_perfect_square_condition_l77_77674

theorem find_function_perfect_square_condition (g : ℕ → ℕ)
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end find_function_perfect_square_condition_l77_77674


namespace product_of_integers_l77_77982

theorem product_of_integers :
  ∃ (A B C : ℤ), A + B + C = 33 ∧ C = 3 * B ∧ A = C - 23 ∧ A * B * C = 192 :=
by
  sorry

end product_of_integers_l77_77982


namespace find_a_l77_77866

variable {a : ℝ}

def p (a : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁ * x₁ + 2 * a * x₁ + 1 = 0 ∧ x₂ * x₂ + 2 * a * x₂ + 1 = 0

def q (a : ℝ) := ∀ x : ℝ, a * x * x - a * x + 1 > 0 

theorem find_a (a : ℝ) : (p a ∨ q a) ∧ ¬ q a → a ≤ -1 :=
sorry

end find_a_l77_77866


namespace axis_of_symmetry_shifted_cos_l77_77543

noncomputable def shifted_cos_axis_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * (Real.pi / 2) - (Real.pi / 12)

theorem axis_of_symmetry_shifted_cos :
  shifted_cos_axis_symmetry x :=
sorry

end axis_of_symmetry_shifted_cos_l77_77543


namespace M_in_fourth_quadrant_l77_77385

-- Define the conditions
variables (a b : ℝ)

/-- Condition that point A(a, 3) and B(2, b) are symmetric with respect to the x-axis -/
def symmetric_points : Prop :=
  a = 2 ∧ 3 = -b

-- Define the point M and quadrant check
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- The theorem stating that if A(a, 3) and B(2, b) are symmetric wrt x-axis, M is in the fourth quadrant
theorem M_in_fourth_quadrant (a b : ℝ) (h : symmetric_points a b) : in_fourth_quadrant a b :=
by {
  sorry
}

end M_in_fourth_quadrant_l77_77385


namespace ratio_of_part_to_whole_l77_77407

theorem ratio_of_part_to_whole : 
  (1 / 4) * (2 / 5) * P = 15 → 
  (40 / 100) * N = 180 → 
  P / N = 1 / 6 := 
by
  intros h1 h2
  sorry

end ratio_of_part_to_whole_l77_77407


namespace convert_base10_to_base9_l77_77070

theorem convert_base10_to_base9 : 
  (2 * 9^3 + 6 * 9^2 + 7 * 9^1 + 7 * 9^0) = 2014 :=
by
  sorry

end convert_base10_to_base9_l77_77070


namespace quadratic_inequality_solution_l77_77120

theorem quadratic_inequality_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m ≤ 0) ↔ m ≤ 1 :=
sorry

end quadratic_inequality_solution_l77_77120


namespace jian_wins_cases_l77_77981

inductive Move
| rock : Move
| paper : Move
| scissors : Move

def wins (jian shin : Move) : Prop :=
  (jian = Move.rock ∧ shin = Move.scissors) ∨
  (jian = Move.paper ∧ shin = Move.rock) ∨
  (jian = Move.scissors ∧ shin = Move.paper)

theorem jian_wins_cases : ∃ n : Nat, n = 3 ∧ (∀ jian shin, wins jian shin → n = 3) :=
by
  sorry

end jian_wins_cases_l77_77981


namespace eval_expr_l77_77715

theorem eval_expr : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end eval_expr_l77_77715


namespace triangle_side_c_l77_77039

noncomputable def area_of_triangle (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

noncomputable def law_of_cosines (a b C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)

theorem triangle_side_c (a b C : ℝ) (h1 : a = 3) (h2 : C = Real.pi * 2 / 3) (h3 : area_of_triangle a b C = 15 * Real.sqrt 3 / 4) : law_of_cosines a b C = 2 :=
by
  sorry

end triangle_side_c_l77_77039


namespace concert_attendance_difference_l77_77007

/-- Define the number of people attending the first concert. -/
def first_concert_attendance : ℕ := 65899

/-- Define the number of people attending the second concert. -/
def second_concert_attendance : ℕ := 66018

/-- The proof statement that the difference in attendance between the second and first concert is 119. -/
theorem concert_attendance_difference :
  (second_concert_attendance - first_concert_attendance = 119) := by
  sorry

end concert_attendance_difference_l77_77007


namespace samuel_teacups_left_l77_77813

-- Define the initial conditions
def total_boxes := 60
def pans_boxes := 12
def decoration_fraction := 1 / 4
def decoration_trade := 3
def trade_gain := 1
def teacups_per_box := 6 * 4 * 2
def broken_per_pickup := 4

-- Calculate the number of boxes initially containing teacups
def remaining_boxes := total_boxes - pans_boxes
def decoration_boxes := decoration_fraction * remaining_boxes
def initial_teacup_boxes := remaining_boxes - decoration_boxes

-- Adjust the number of teacup boxes after the trade
def teacup_boxes := initial_teacup_boxes + trade_gain

-- Calculate total number of teacups and the number of teacups broken
def total_teacups := teacup_boxes * teacups_per_box
def total_broken := teacup_boxes * broken_per_pickup

-- Calculate the number of teacups left
def teacups_left := total_teacups - total_broken

-- State the theorem
theorem samuel_teacups_left : teacups_left = 1628 := by
  sorry

end samuel_teacups_left_l77_77813


namespace perfect_squares_factors_360_l77_77769

theorem perfect_squares_factors_360 : 
  let n := 360
  let prime_factors := (2, 3, 5)
  let exponents := (3, 2, 1)
  ∃ (count : ℕ), count = 4 :=
by
  let n := 360
  let prime_factors := (2, 3, 5)
  let exponents := (3, 2, 1)
  -- Calculation by hand has shown us that there are 4 perfect square factors
  exact ⟨4, rfl⟩

end perfect_squares_factors_360_l77_77769


namespace h_in_terms_of_f_l77_77512

-- Definitions based on conditions in a)
def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) := f (-x)
def shift_left (f : ℝ → ℝ) (x : ℝ) (c : ℝ) := f (x + c)

-- Express h(x) in terms of f(x) based on conditions
theorem h_in_terms_of_f (f : ℝ → ℝ) (x : ℝ) :
  reflect_y_axis (shift_left f 2) x = f (-x - 2) :=
by
  sorry

end h_in_terms_of_f_l77_77512


namespace scientific_notation_113700_l77_77328

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end scientific_notation_113700_l77_77328


namespace repeating_decimal_eq_l77_77432

-- Defining the repeating decimal as a hypothesis
def repeating_decimal : ℚ := 0.7 + 3/10^2 * (1/(1 - 1/10))
-- We will prove this later by simplifying the fraction
def expected_fraction : ℚ := 11/15

theorem repeating_decimal_eq : repeating_decimal = expected_fraction := 
by
  sorry

end repeating_decimal_eq_l77_77432


namespace percentage_increase_in_gross_revenue_l77_77740

theorem percentage_increase_in_gross_revenue 
  (P R : ℝ) 
  (hP : P > 0) 
  (hR : R > 0) 
  (new_price : ℝ := 0.80 * P) 
  (new_quantity : ℝ := 1.60 * R) : 
  (new_price * new_quantity - P * R) / (P * R) * 100 = 28 := 
by
  sorry

end percentage_increase_in_gross_revenue_l77_77740


namespace cos_2015_eq_neg_m_l77_77373

variable (m : ℝ)

-- Given condition
axiom sin_55_eq_m : Real.sin (55 * Real.pi / 180) = m

-- The proof problem
theorem cos_2015_eq_neg_m : Real.cos (2015 * Real.pi / 180) = -m :=
by
  sorry

end cos_2015_eq_neg_m_l77_77373


namespace sum_of_fully_paintable_numbers_l77_77521

def is_fully_paintable (h t u : ℕ) : Prop :=
  (∀ n : ℕ, (∀ k1 : ℕ, n ≠ 1 + k1 * h) ∧ (∀ k2 : ℕ, n ≠ 3 + k2 * t) ∧ (∀ k3 : ℕ, n ≠ 2 + k3 * u)) → False

theorem sum_of_fully_paintable_numbers :  ∃ L : List ℕ, (∀ x ∈ L, ∃ (h t u : ℕ), is_fully_paintable h t u ∧ 100 * h + 10 * t + u = x) ∧ L.sum = 944 :=
sorry

end sum_of_fully_paintable_numbers_l77_77521


namespace uncle_bradley_bills_l77_77074

theorem uncle_bradley_bills :
  ∃ (fifty_bills hundred_bills : ℕ),
    (fifty_bills = 300 / 50) ∧ (hundred_bills = 700 / 100) ∧ (300 + 700 = 1000) ∧ (50 * fifty_bills + 100 * hundred_bills = 1000) ∧ (fifty_bills + hundred_bills = 13) :=
by
  sorry

end uncle_bradley_bills_l77_77074


namespace determinant_problem_l77_77424

theorem determinant_problem 
  (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  ((x * (8 * z + 4 * w)) - (z * (8 * x + 4 * y))) = 28 :=
by 
  sorry

end determinant_problem_l77_77424


namespace trajectory_of_moving_circle_l77_77617

noncomputable def trajectory_equation_of_moving_circle_center 
  (x y : Real) : Prop :=
  (∃ r : Real, 
    ((x + 5)^2 + y^2 = 16) ∧ 
    ((x - 5)^2 + y^2 = 16)
  ) → (x > 0 → x^2 / 16 - y^2 / 9 = 1)

-- here's the statement of the proof problem
theorem trajectory_of_moving_circle
  (h₁ : ∀ x y : Real, (x + 5)^2 + y^2 = 16)
  (h₂ : ∀ x y : Real, (x - 5)^2 + y^2 = 16) :
  ∀ x y : Real, trajectory_equation_of_moving_circle_center x y :=
sorry

end trajectory_of_moving_circle_l77_77617


namespace inverse_value_l77_77520

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end inverse_value_l77_77520


namespace janet_more_siblings_than_carlos_l77_77384

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end janet_more_siblings_than_carlos_l77_77384


namespace sugar_and_granulated_sugar_delivered_l77_77763

theorem sugar_and_granulated_sugar_delivered (total_bags : ℕ) (percentage_more : ℚ) (mass_ratio : ℚ) (total_weight : ℚ)
    (h_total_bags : total_bags = 63)
    (h_percentage_more : percentage_more = 1.25)
    (h_mass_ratio : mass_ratio = 3 / 4)
    (h_total_weight : total_weight = 4.8) :
    ∃ (sugar_weight granulated_sugar_weight : ℚ),
        (granulated_sugar_weight = 1.8) ∧ (sugar_weight = 3) ∧
        ((sugar_weight + granulated_sugar_weight = total_weight) ∧
        (sugar_weight / 28 = (granulated_sugar_weight / 35) * mass_ratio)) :=
by
    sorry

end sugar_and_granulated_sugar_delivered_l77_77763


namespace farmer_field_l77_77862

theorem farmer_field (m : ℤ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 :=
by
  sorry

end farmer_field_l77_77862


namespace numbers_equal_l77_77343

theorem numbers_equal (a b c d : ℕ)
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  a = b ∨ b = c ∨ c = d ∨ a = c ∨ a = d ∨ b = d ∨ (a = b ∧ b = c) ∨ (b = c ∧ c = d) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) :=
sorry

end numbers_equal_l77_77343


namespace find_numbers_l77_77379

theorem find_numbers (x y z u n : ℤ)
  (h1 : x + y + z + u = 36)
  (h2 : x + n = y - n)
  (h3 : x + n = z * n)
  (h4 : x + n = u / n) :
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
sorry

end find_numbers_l77_77379


namespace intersection_M_N_l77_77599

open Set Real

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | abs (x - 1) ≤ 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l77_77599


namespace yellow_block_weight_proof_l77_77752

-- Define the weights and the relationship between them
def green_block_weight : ℝ := 0.4
def additional_weight : ℝ := 0.2
def yellow_block_weight : ℝ := green_block_weight + additional_weight

-- The theorem to prove
theorem yellow_block_weight_proof : yellow_block_weight = 0.6 :=
by
  -- Proof will be supplied here
  sorry

end yellow_block_weight_proof_l77_77752


namespace total_amount_paid_l77_77832

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end total_amount_paid_l77_77832


namespace system_real_solutions_l77_77971

theorem system_real_solutions (a b c : ℝ) :
  (∃ x : ℝ, 
    a * x^2 + b * x + c = 0 ∧ 
    b * x^2 + c * x + a = 0 ∧ 
    c * x^2 + a * x + b = 0) ↔ 
  a + b + c = 0 :=
sorry

end system_real_solutions_l77_77971


namespace bunch_of_bananas_cost_l77_77350

def cost_of_bananas (A : ℝ) : ℝ := 5 - A

theorem bunch_of_bananas_cost (A B T : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = T) : B = cost_of_bananas A :=
by
  sorry

end bunch_of_bananas_cost_l77_77350


namespace product_modulo_l77_77236

theorem product_modulo : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m := 
  sorry

end product_modulo_l77_77236


namespace reciprocals_sum_eq_neg_one_over_three_l77_77277

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end reciprocals_sum_eq_neg_one_over_three_l77_77277


namespace bricks_required_for_courtyard_l77_77772

/-- 
A courtyard is 45 meters long and 25 meters broad needs to be paved with bricks of 
dimensions 15 cm by 7 cm. What will be the total number of bricks required?
-/
theorem bricks_required_for_courtyard 
  (courtyard_length : ℕ) (courtyard_width : ℕ)
  (brick_length : ℕ) (brick_width : ℕ)
  (H1 : courtyard_length = 4500) (H2 : courtyard_width = 2500)
  (H3 : brick_length = 15) (H4 : brick_width = 7) :
  let courtyard_area_cm : ℕ := courtyard_length * courtyard_width
  let brick_area_cm : ℕ := brick_length * brick_width
  let total_bricks : ℕ := (courtyard_area_cm + brick_area_cm - 1) / brick_area_cm
  total_bricks = 107143 := by
  sorry

end bricks_required_for_courtyard_l77_77772


namespace triangle_inequality_l77_77112

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l77_77112


namespace area_of_rhombus_l77_77225

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 4) (h2 : d2 = 4) :
    (d1 * d2) / 2 = 8 := by
  sorry

end area_of_rhombus_l77_77225


namespace xy_plus_four_is_square_l77_77158

theorem xy_plus_four_is_square (x y : ℕ) (h : ((1 / (x : ℝ)) + (1 / (y : ℝ)) + 1 / (x * y : ℝ)) = (1 / (x + 4 : ℝ) + 1 / (y - 4 : ℝ) + 1 / ((x + 4) * (y - 4) : ℝ))) : 
  ∃ (k : ℕ), xy + 4 = k^2 :=
by
  sorry

end xy_plus_four_is_square_l77_77158


namespace age_of_b_l77_77636

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 := 
  sorry

end age_of_b_l77_77636


namespace odd_power_sum_divisible_l77_77792

theorem odd_power_sum_divisible (x y : ℤ) (n : ℕ) (h_odd : ∃ k : ℕ, n = 2 * k + 1) :
  (x ^ n + y ^ n) % (x + y) = 0 := 
sorry

end odd_power_sum_divisible_l77_77792


namespace giant_kite_area_72_l77_77878

-- Definition of the vertices of the medium kite
def vertices_medium_kite : List (ℕ × ℕ) := [(1,6), (4,9), (7,6), (4,1)]

-- Given condition function to check if the giant kite is created by doubling the height and width
def double_coordinates (c : (ℕ × ℕ)) : (ℕ × ℕ) := (2 * c.1, 2 * c.2)

def vertices_giant_kite : List (ℕ × ℕ) := vertices_medium_kite.map double_coordinates

-- Function to calculate the area of the kite based on its vertices
def kite_area (vertices : List (ℕ × ℕ)) : ℕ := sorry -- The way to calculate the kite area can be complex

-- Theorem to prove the area of the giant kite
theorem giant_kite_area_72 :
  kite_area vertices_giant_kite = 72 := 
sorry

end giant_kite_area_72_l77_77878


namespace geometric_sequence_ratio_l77_77949

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_q : q = -1 / 2) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 :=
sorry

end geometric_sequence_ratio_l77_77949


namespace triangle_area_l77_77654

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a + b > c) (h3 : a + c > b) (h4 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 3 ∧ c = 5 ∨ a = 5 ∧ b = 4 ∧ c = 3 ∨
  a = 5 ∧ b = 3 ∧ c = 4 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 3 ∧ b = 5 ∧ c = 4 → 
  (1 / 2 : ℝ) * ↑a * ↑b = 6 := by
  sorry

end triangle_area_l77_77654


namespace total_fish_correct_l77_77309

def Billy_fish : ℕ := 10
def Tony_fish : ℕ := 3 * Billy_fish
def Sarah_fish : ℕ := Tony_fish + 5
def Bobby_fish : ℕ := 2 * Sarah_fish
def Jenny_fish : ℕ := Bobby_fish - 4
def total_fish : ℕ := Billy_fish + Tony_fish + Sarah_fish + Bobby_fish + Jenny_fish

theorem total_fish_correct : total_fish = 211 := by
  sorry

end total_fish_correct_l77_77309


namespace ten_pow_n_plus_eight_div_nine_is_integer_l77_77902

theorem ten_pow_n_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, 10^n + 8 = 9 * k := 
sorry

end ten_pow_n_plus_eight_div_nine_is_integer_l77_77902


namespace euclid1976_partb_problem2_l77_77504

theorem euclid1976_partb_problem2
  (x y : ℝ)
  (geo_prog : y^2 = 2 * x)
  (arith_prog : 2 / y = 1 / x + 9 / x^2) :
  x * y = 27 / 2 := by 
  sorry

end euclid1976_partb_problem2_l77_77504


namespace min_cans_needed_l77_77937

theorem min_cans_needed (oz_per_can : ℕ) (total_oz_needed : ℕ) (H1 : oz_per_can = 15) (H2 : total_oz_needed = 150) :
  ∃ n : ℕ, 15 * n ≥ 150 ∧ ∀ m : ℕ, 15 * m ≥ 150 → n ≤ m :=
by
  sorry

end min_cans_needed_l77_77937


namespace inverse_function_f_l77_77181

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_f : ∀ x > 0, f_inv (f x) = x :=
by
  intro x hx
  dsimp [f, f_inv]
  sorry

end inverse_function_f_l77_77181


namespace cost_difference_of_buses_l77_77598

-- Definitions from the conditions
def bus_cost_equations (x y : ℝ) :=
  (x + 2 * y = 260) ∧ (2 * x + y = 280)

-- The statement to prove
theorem cost_difference_of_buses (x y : ℝ) (h : bus_cost_equations x y) :
  x - y = 20 :=
sorry

end cost_difference_of_buses_l77_77598


namespace correct_operation_l77_77805

theorem correct_operation : (3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3) ∧ 
                            ¬(a^2 * a^3 = a^6) ∧ 
                            ¬(a^6 / a^2 = a^3) ∧ 
                            ¬((a^2)^3 = a^5) :=
by
  sorry

end correct_operation_l77_77805


namespace polynomial_coefficient_sum_equality_l77_77318

theorem polynomial_coefficient_sum_equality :
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
    (∀ x : ℝ, (2 * x + 1)^4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
    (a₀ - a₁ + a₂ - a₃ + a₄ = 1) :=
by
  intros
  sorry

end polynomial_coefficient_sum_equality_l77_77318


namespace problem_i_l77_77722

theorem problem_i (n : ℕ) (h : n ≥ 1) : n ∣ 2^n - 1 ↔ n = 1 := by
  sorry

end problem_i_l77_77722


namespace correct_option_is_A_l77_77566

def second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (0, 4)
def point_D : ℝ × ℝ := (5, -6)

theorem correct_option_is_A :
  (second_quadrant point_A) ∧
  ¬(second_quadrant point_B) ∧
  ¬(second_quadrant point_C) ∧
  ¬(second_quadrant point_D) :=
by sorry

end correct_option_is_A_l77_77566


namespace sufficient_but_not_necessary_condition_for_prop_l77_77160

theorem sufficient_but_not_necessary_condition_for_prop :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_for_prop_l77_77160


namespace jackie_first_tree_height_l77_77125

theorem jackie_first_tree_height
  (h : ℝ)
  (avg_height : (h + 2 * (h / 2) + (h + 200)) / 4 = 800) :
  h = 1000 :=
by
  sorry

end jackie_first_tree_height_l77_77125


namespace count_positive_integers_in_range_l77_77552

theorem count_positive_integers_in_range :
  ∃ (count : ℕ), count = 11 ∧
    ∀ (n : ℕ), 300 < n^2 ∧ n^2 < 800 → (n ≥ 18 ∧ n ≤ 28) :=
by
  sorry

end count_positive_integers_in_range_l77_77552


namespace dubblefud_red_balls_l77_77879

theorem dubblefud_red_balls (R B G : ℕ) 
  (h1 : 3^R * 7^B * 11^G = 5764801)
  (h2 : B = G) :
  R = 7 :=
by
  sorry

end dubblefud_red_balls_l77_77879


namespace cos_7theta_l77_77427

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (7 * θ) = 49 / 2187 := 
  sorry

end cos_7theta_l77_77427


namespace expected_value_of_third_flip_l77_77855

-- Definitions for the conditions
def prob_heads : ℚ := 2/5
def prob_tails : ℚ := 3/5
def win_amount : ℚ := 4
def base_loss : ℚ := 3
def doubled_loss : ℚ := 2 * base_loss
def first_two_flips_were_tails : Prop := true 

-- The main statement: Proving the expected value of the third flip
theorem expected_value_of_third_flip (h : first_two_flips_were_tails) : 
  (prob_heads * win_amount + prob_tails * -doubled_loss) = -2 := by
  sorry

end expected_value_of_third_flip_l77_77855


namespace conservation_center_total_turtles_l77_77731

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l77_77731


namespace basketball_team_points_l77_77701

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end basketball_team_points_l77_77701


namespace oranges_per_group_l77_77156

theorem oranges_per_group (total_oranges groups : ℕ) (h1 : total_oranges = 384) (h2 : groups = 16) :
  total_oranges / groups = 24 := by
  sorry

end oranges_per_group_l77_77156


namespace balls_into_boxes_problem_l77_77615

theorem balls_into_boxes_problem :
  ∃ (n : ℕ), n = 144 ∧ ∃ (balls : Fin 4 → ℕ), 
  (∃ (boxes : Fin 4 → Fin 4), 
    (∀ (b : Fin 4), boxes b < 4 ∧ boxes b ≠ b) ∧ 
    (∃! (empty_box : Fin 4), ∀ (b : Fin 4), (boxes b = empty_box) → false)) := 
by
  sorry

end balls_into_boxes_problem_l77_77615


namespace inequality_ge_five_halves_l77_77747

open Real

noncomputable def xy_yz_zx_eq_one (x y z : ℝ) := x * y + y * z + z * x = 1
noncomputable def non_neg (x y z : ℝ) := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem inequality_ge_five_halves (x y z : ℝ) (h1 : xy_yz_zx_eq_one x y z) (h2 : non_neg x y z) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 := 
sorry

end inequality_ge_five_halves_l77_77747


namespace geometric_sequence_seventh_term_l77_77965

theorem geometric_sequence_seventh_term (a r : ℝ) 
    (h1 : a * r^3 = 8) 
    (h2 : a * r^9 = 2) : 
    a * r^6 = 1 := 
by 
    sorry

end geometric_sequence_seventh_term_l77_77965


namespace integer_count_between_cubes_l77_77540

-- Definitions and conditions
def a : ℝ := 10.7
def b : ℝ := 10.8

-- Precomputed values
def a_cubed : ℝ := 1225.043
def b_cubed : ℝ := 1259.712

-- The theorem to prove
theorem integer_count_between_cubes (ha : a ^ 3 = a_cubed) (hb : b ^ 3 = b_cubed) :
  let start := Int.ceil a_cubed
  let end_ := Int.floor b_cubed
  end_ - start + 1 = 34 :=
by
  sorry

end integer_count_between_cubes_l77_77540


namespace average_speed_of_train_l77_77108

-- Condition: Distance traveled is 42 meters
def distance : ℕ := 42

-- Condition: Time taken is 6 seconds
def time : ℕ := 6

-- Average speed computation
theorem average_speed_of_train : distance / time = 7 := by
  -- Left to the prover
  sorry

end average_speed_of_train_l77_77108


namespace ratio_small_to_large_is_one_to_one_l77_77822

theorem ratio_small_to_large_is_one_to_one
  (total_beads : ℕ)
  (large_beads_per_bracelet : ℕ)
  (bracelets_count : ℕ)
  (small_beads : ℕ)
  (large_beads : ℕ)
  (small_beads_per_bracelet : ℕ) :
  total_beads = 528 →
  large_beads_per_bracelet = 12 →
  bracelets_count = 11 →
  large_beads = total_beads / 2 →
  large_beads >= bracelets_count * large_beads_per_bracelet →
  small_beads = total_beads / 2 →
  small_beads_per_bracelet = small_beads / bracelets_count →
  small_beads_per_bracelet / large_beads_per_bracelet = 1 :=
by sorry

end ratio_small_to_large_is_one_to_one_l77_77822


namespace sum_of_squares_l77_77272

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hwx : w * x = a^2) 
  (hwy : w * y = b^2) 
  (hwz : w * z = c^2) 
  (hw : w ≠ 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := 
by
  sorry

end sum_of_squares_l77_77272


namespace is_condition_B_an_algorithm_l77_77953

-- Definitions of conditions A, B, C, D
def condition_A := "At home, it is generally the mother who cooks"
def condition_B := "The steps to cook rice include washing the pot, rinsing the rice, adding water, and heating"
def condition_C := "Cooking outdoors is called camping cooking"
def condition_D := "Rice is necessary for cooking"

-- Definition of being considered an algorithm
def is_algorithm (s : String) : Prop :=
  s = condition_B  -- Based on the analysis that condition_B meets the criteria of an algorithm

-- The proof statement to show that condition_B can be considered an algorithm
theorem is_condition_B_an_algorithm : is_algorithm condition_B :=
by
  sorry

end is_condition_B_an_algorithm_l77_77953


namespace greatest_AB_CBA_div_by_11_l77_77479

noncomputable def AB_CBA_max_value (A B C : ℕ) : ℕ := 10001 * A + 1010 * B + 100 * C + 10 * B + A

theorem greatest_AB_CBA_div_by_11 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  2 * A - 2 * B + C % 11 = 0 ∧ 
  ∀ (A' B' C' : ℕ),
    A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
    2 * A' - 2 * B' + C' % 11 = 0 → 
    AB_CBA_max_value A B C ≥ AB_CBA_max_value A' B' C' :=
  by sorry

end greatest_AB_CBA_div_by_11_l77_77479


namespace triangle_third_side_l77_77103

theorem triangle_third_side {x : ℕ} (h1 : 3 < x) (h2 : x < 7) (h3 : x % 2 = 1) : x = 5 := by
  sorry

end triangle_third_side_l77_77103


namespace geometric_sequence_at_t_l77_77808

theorem geometric_sequence_at_t (a : ℕ → ℕ) (S : ℕ → ℕ) (t : ℕ) :
  (∀ n, a n = a 1 * (3 ^ (n - 1))) →
  a 1 = 1 →
  S t = (a 1 * (1 - 3 ^ t)) / (1 - 3) →
  S t = 364 →
  a t = 243 :=
by {
  sorry
}

end geometric_sequence_at_t_l77_77808


namespace tablecloth_covers_table_l77_77212

theorem tablecloth_covers_table
(length_ellipse : ℝ) (width_ellipse : ℝ) (length_tablecloth : ℝ) (width_tablecloth : ℝ)
(h1 : length_ellipse = 160)
(h2 : width_ellipse = 100)
(h3 : length_tablecloth = 140)
(h4 : width_tablecloth = 130) :
length_tablecloth >= width_ellipse ∧ width_tablecloth >= width_ellipse ∧
(length_tablecloth ^ 2 + width_tablecloth ^ 2) >= (length_ellipse ^ 2 + width_ellipse ^ 2) :=
by
  sorry

end tablecloth_covers_table_l77_77212


namespace jason_games_planned_last_month_l77_77852

-- Define the conditions
variable (games_planned_this_month : Nat) (games_missed : Nat) (games_attended : Nat)

-- Define what we want to prove
theorem jason_games_planned_last_month (h1 : games_planned_this_month = 11)
                                        (h2 : games_missed = 16)
                                        (h3 : games_attended = 12) :
                                        (games_attended + games_missed - games_planned_this_month = 17) := 
by
  sorry

end jason_games_planned_last_month_l77_77852


namespace find_values_l77_77702

theorem find_values (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 :=
by 
  sorry

end find_values_l77_77702


namespace part1_part2_1_part2_2_l77_77591

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 2) * x + 4
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x + b - 3) / (a * x^2 + 2)

theorem part1 (a : ℝ) (b : ℝ) :
  (∀ x, f x a = f (-x) a) → b = 3 :=
by sorry

theorem part2_1 (a : ℝ) (b : ℝ) :
  a = 2 → b = 3 →
  ∀ x₁ x₂, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ x₁ < x₂ → g x₁ a b < g x₂ a b :=
by sorry

theorem part2_2 (a : ℝ) (b : ℝ) (t : ℝ) :
  a = 2 → b = 3 →
  g (t - 1) a b + g (2 * t) a b < 0 →
  0 < t ∧ t < 1 / 3 :=
by sorry

end part1_part2_1_part2_2_l77_77591


namespace no_psafe_numbers_l77_77877

def is_psafe (n p : ℕ) : Prop := 
  ¬ (n % p = 0 ∨ n % p = 1 ∨ n % p = 2 ∨ n % p = 3 ∨ n % p = p - 3 ∨ n % p = p - 2 ∨ n % p = p - 1)

theorem no_psafe_numbers (N : ℕ) (hN : N = 10000) :
  ∀ n, (n ≤ N ∧ is_psafe n 5 ∧ is_psafe n 7 ∧ is_psafe n 11) → false :=
by
  sorry

end no_psafe_numbers_l77_77877


namespace points_on_same_side_after_25_seconds_l77_77889

def movement_time (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) : ℕ :=
  25

theorem points_on_same_side_after_25_seconds (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) :
  side_length = 100 ∧ perimeter = 400 ∧ speed_A = 5 ∧ speed_B = 10 ∧ start_mid_B = 50 →
  movement_time side_length perimeter speed_A speed_B start_mid_B = 25 :=
by
  intros h
  sorry

end points_on_same_side_after_25_seconds_l77_77889


namespace sum_of_consecutive_powers_divisible_l77_77127

theorem sum_of_consecutive_powers_divisible (a : ℕ) (n : ℕ) (h : 0 ≤ n) : 
  a^n + a^(n + 1) ∣ a * (a + 1) :=
sorry

end sum_of_consecutive_powers_divisible_l77_77127


namespace sin_cos_product_l77_77232

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l77_77232


namespace expected_value_of_win_is_3_5_l77_77721

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_3_5_l77_77721


namespace isosceles_triangle_angle_l77_77152

theorem isosceles_triangle_angle
  (A B C : ℝ)
  (h1 : A = C)
  (h2 : B = 2 * A - 40)
  (h3 : A + B + C = 180) :
  B = 70 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_angle_l77_77152


namespace tetrahedron_pythagorean_theorem_l77_77340

noncomputable section

variables {a b c : ℝ} {S_ABC S_VAB S_VBC S_VAC : ℝ}

-- Conditions
def is_right_triangle (a b c : ℝ) := c^2 = a^2 + b^2
def is_right_tetrahedron (S_ABC S_VAB S_VBC S_VAC : ℝ) := 
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2

-- Theorem Statement
theorem tetrahedron_pythagorean_theorem (a b c S_ABC S_VAB S_VBC S_VAC : ℝ) 
  (h1 : is_right_triangle a b c)
  (h2 : S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2) :
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2 := 
by sorry

end tetrahedron_pythagorean_theorem_l77_77340


namespace total_spent_is_13_l77_77460

-- Let cost_cb represent the cost of the candy bar
def cost_cb : ℕ := 7

-- Let cost_ch represent the cost of the chocolate
def cost_ch : ℕ := 6

-- Define the total cost as the sum of cost_cb and cost_ch
def total_cost : ℕ := cost_cb + cost_ch

-- Theorem to prove the total cost equals $13
theorem total_spent_is_13 : total_cost = 13 := by
  sorry

end total_spent_is_13_l77_77460


namespace flower_beds_fraction_l77_77000

open Real

noncomputable def parkArea (a b h : ℝ) := (a + b) / 2 * h
noncomputable def triangleArea (a : ℝ) := (1 / 2) * a ^ 2

theorem flower_beds_fraction 
  (a b h : ℝ) 
  (h_a: a = 15) 
  (h_b: b = 30) 
  (h_h: h = (b - a) / 2) :
  (2 * triangleArea h) / parkArea a b h = 1 / 4 := by 
  sorry

end flower_beds_fraction_l77_77000


namespace range_of_a_l77_77964

theorem range_of_a {A : Set ℝ} (h1: ∀ x ∈ A, 2 * x + a > 0) (h2: 1 ∉ A) (h3: 2 ∈ A) : -4 < a ∧ a ≤ -2 := 
sorry

end range_of_a_l77_77964


namespace proof_problem_l77_77978

theorem proof_problem (a b : ℤ) (h1 : ∃ k, a = 5 * k) (h2 : ∃ m, b = 10 * m) :
  (∃ n, b = 5 * n) ∧ (∃ p, a - b = 5 * p) :=
by
  sorry

end proof_problem_l77_77978


namespace max_weight_of_chocolates_l77_77264

def max_total_weight (chocolates : List ℕ) (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) : ℕ :=
300

theorem max_weight_of_chocolates (chocolates : List ℕ)
  (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) :
  max_total_weight chocolates H_wt H_div = 300 :=
sorry

end max_weight_of_chocolates_l77_77264


namespace range_of_a_l77_77282

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ (¬ ∃ x : ℝ, x < 0 ∧ |x| = a * x - a) ↔ (a > 1 ∨ a ≤ -1) :=
sorry

end range_of_a_l77_77282


namespace prism_height_relation_l77_77629

theorem prism_height_relation (a b c h : ℝ) 
  (h_perp : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_height : 0 < h) 
  (h_right_angles : true) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 :=
by 
  sorry 

end prism_height_relation_l77_77629


namespace original_deck_size_l77_77362

noncomputable def initial_red_probability (r b : ℕ) : Prop := r / (r + b) = 1 / 4
noncomputable def added_black_probability (r b : ℕ) : Prop := r / (r + (b + 6)) = 1 / 6

theorem original_deck_size (r b : ℕ) 
  (h1 : initial_red_probability r b) 
  (h2 : added_black_probability r b) : 
  r + b = 12 := 
sorry

end original_deck_size_l77_77362


namespace quadratic_inequality_solution_l77_77281

theorem quadratic_inequality_solution (x : ℝ) : 16 ≤ x ∧ x ≤ 20 → x^2 - 36 * x + 323 ≤ 3 :=
by
  sorry

end quadratic_inequality_solution_l77_77281


namespace initial_number_of_persons_l77_77163

theorem initial_number_of_persons (n : ℕ) 
  (w_increase : ∀ (k : ℕ), k = 4) 
  (old_weight new_weight : ℕ) 
  (h_old : old_weight = 58) 
  (h_new : new_weight = 106) 
  (h_difference : new_weight - old_weight = 48) 
  : n = 12 := 
by
  sorry

end initial_number_of_persons_l77_77163


namespace find_number_l77_77037

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 :=
sorry

end find_number_l77_77037


namespace fraction_difference_l77_77117

def A : ℕ := 3 + 6 + 9
def B : ℕ := 2 + 5 + 8

theorem fraction_difference : (A / B) - (B / A) = 11 / 30 := by
  sorry

end fraction_difference_l77_77117


namespace find_a_l77_77402

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 18 * x^3 + ((86 : ℝ)) * x^2 + 200 * x - 1984

-- Define the condition and statement
theorem find_a (α β γ δ : ℝ) (hαβγδ : α * β * γ * δ = -1984)
  (hαβ : α * β = -32) (hγδ : γ * δ = 62) :
  (∀ a : ℝ, a = 86) :=
  sorry

end find_a_l77_77402


namespace solve_equation_1_solve_equation_2_l77_77220

theorem solve_equation_1 (x : ℝ) : (2 * x - 1) ^ 2 - 25 = 0 ↔ x = 3 ∨ x = -2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (1 / 3) * (x + 3) ^ 3 - 9 = 0 ↔ x = 0 := 
sorry

end solve_equation_1_solve_equation_2_l77_77220


namespace difference_in_earnings_in_currency_B_l77_77135

-- Definitions based on conditions
def num_red_stamps : Nat := 30
def num_white_stamps : Nat := 80
def price_per_red_stamp_currency_A : Nat := 5
def price_per_white_stamp_currency_B : Nat := 50
def exchange_rate_A_to_B : Nat := 2

-- Theorem based on the question and correct answer
theorem difference_in_earnings_in_currency_B : 
  num_white_stamps * price_per_white_stamp_currency_B - 
  (num_red_stamps * price_per_red_stamp_currency_A * exchange_rate_A_to_B) = 3700 := 
  by
  sorry

end difference_in_earnings_in_currency_B_l77_77135


namespace greatest_x_inequality_l77_77034

theorem greatest_x_inequality :
  ∃ x, -x^2 + 11 * x - 28 = 0 ∧ (∀ y, -y^2 + 11 * y - 28 ≥ 0 → y ≤ x) ∧ x = 7 :=
sorry

end greatest_x_inequality_l77_77034


namespace polygon_is_octahedron_l77_77291

theorem polygon_is_octahedron (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_is_octahedron_l77_77291


namespace max_tetrahedron_volume_l77_77269

theorem max_tetrahedron_volume 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (right_triangle : ∃ A B C : Type, 
    ∃ (angle_C : ℝ) (h_angle_C : angle_C = π / 2), 
    ∃ (BC CA : ℝ), BC = a ∧ CA = b) : 
  ∃ V : ℝ, V = (a^2 * b^2) / (6 * (a^(2/3) + b^(2/3))^(3/2)) := 
sorry

end max_tetrahedron_volume_l77_77269


namespace ezekiel_painted_faces_l77_77231

noncomputable def cuboid_faces_painted (num_cuboids : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
num_cuboids * faces_per_cuboid

theorem ezekiel_painted_faces :
  cuboid_faces_painted 8 6 = 48 := 
by
  sorry

end ezekiel_painted_faces_l77_77231


namespace sequence_solution_exists_l77_77612

noncomputable def math_problem (a : ℕ → ℝ) : Prop :=
  ∀ n < 1990, a n > 0 ∧ a 1990 < 0

theorem sequence_solution_exists {a0 c : ℝ} (h_a0 : a0 > 0) (h_c : c > 0) :
  ∃ (a : ℕ → ℝ),
    a 0 = a0 ∧
    (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
    math_problem a :=
by
  sorry

end sequence_solution_exists_l77_77612


namespace prime_number_property_l77_77036

open Nat

-- Definition that p is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Conjecture to prove: if p is a prime number and p^4 - 3p^2 + 9 is also a prime number, then p = 2.
theorem prime_number_property (p : ℕ) (h1 : is_prime p) (h2 : is_prime (p^4 - 3*p^2 + 9)) : p = 2 :=
sorry

end prime_number_property_l77_77036


namespace longest_side_enclosure_l77_77997

variable (l w : ℝ)

theorem longest_side_enclosure (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 :=
sorry

end longest_side_enclosure_l77_77997


namespace quadratic_root_one_is_minus_one_l77_77368

theorem quadratic_root_one_is_minus_one (m : ℝ) (h : ∃ x : ℝ, x = -1 ∧ m * x^2 + x - m^2 + 1 = 0) : m = 1 :=
by
  sorry

end quadratic_root_one_is_minus_one_l77_77368


namespace online_store_commission_l77_77530

theorem online_store_commission (cost : ℝ) (desired_profit_pct : ℝ) (online_price : ℝ) (commission_pct : ℝ) :
  cost = 19 →
  desired_profit_pct = 0.20 →
  online_price = 28.5 →
  commission_pct = 25 :=
by
  sorry

end online_store_commission_l77_77530


namespace greatest_three_digit_multiple_of_17_l77_77487

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77487


namespace problem_equivalent_l77_77446

theorem problem_equivalent (a b : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^2 * x^2)/2 + (a^3 * x^3)/6 + (a^4 * x^4)/24 + (a^5 * x^5)/120) : 
  a - b = -38 :=
sorry

end problem_equivalent_l77_77446


namespace base7_to_base10_l77_77018

theorem base7_to_base10 : 
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  digit0 * base^0 + digit1 * base^1 + digit2 * base^2 + digit3 * base^3 = 1934 :=
by
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  sorry

end base7_to_base10_l77_77018


namespace opposite_of_negative_2023_l77_77046

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l77_77046


namespace angle_F_measure_l77_77509

-- Define angle B
def angle_B := 120

-- Define angle C being supplementary to angle B on a straight line
def angle_C := 180 - angle_B

-- Define angle D
def angle_D := 45

-- Define angle E
def angle_E := 30

-- Define the vertically opposite angle F to angle C
def angle_F := angle_C

theorem angle_F_measure : angle_F = 60 :=
by
  -- Provide a proof by specifying sorry to indicate the proof is not complete
  sorry

end angle_F_measure_l77_77509


namespace find_number_l77_77656

theorem find_number (x : ℚ) (h : (3 * x / 2) + 6 = 11) : x = 10 / 3 :=
sorry

end find_number_l77_77656


namespace length_HD_is_3_l77_77840

noncomputable def square_side : ℝ := 8

noncomputable def midpoint_AD : ℝ := square_side / 2

noncomputable def length_FD : ℝ := midpoint_AD

theorem length_HD_is_3 :
  ∃ (x : ℝ), 0 < x ∧ x < square_side ∧ (8 - x) ^ 2 = x ^ 2 + length_FD ^ 2 ∧ x = 3 :=
by
  sorry

end length_HD_is_3_l77_77840


namespace contradiction_proof_l77_77448

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →
    false := 
by
  intros a b c d h1 h2 h3 h4
  sorry

end contradiction_proof_l77_77448


namespace payment_required_l77_77378

-- Definitions of the conditions
def price_suit : ℕ := 200
def price_tie : ℕ := 40
def num_suits : ℕ := 20
def discount_option_1 (x : ℕ) (hx : x > 20) : ℕ := price_suit * num_suits + (x - num_suits) * price_tie
def discount_option_2 (x : ℕ) (hx : x > 20) : ℕ := (price_suit * num_suits + x * price_tie) * 9 / 10

-- Theorem that needs to be proved
theorem payment_required (x : ℕ) (hx : x > 20) :
  discount_option_1 x hx = 40 * x + 3200 ∧ discount_option_2 x hx = 3600 + 36 * x :=
by sorry

end payment_required_l77_77378


namespace find_Pete_original_number_l77_77515

noncomputable def PeteOriginalNumber (x : ℝ) : Prop :=
  5 * (3 * x + 15) = 200

theorem find_Pete_original_number : ∃ x : ℝ, PeteOriginalNumber x ∧ x = 25 / 3 :=
by
  sorry

end find_Pete_original_number_l77_77515


namespace find_ordered_pairs_of_b_c_l77_77165

theorem find_ordered_pairs_of_b_c : 
  ∃! (pairs : ℕ × ℕ), 
    (pairs.1 > 0 ∧ pairs.2 > 0) ∧ 
    (pairs.1 * pairs.1 = 4 * pairs.2) ∧ 
    (pairs.2 * pairs.2 = 4 * pairs.1) :=
sorry

end find_ordered_pairs_of_b_c_l77_77165


namespace salary_increase_is_57point35_percent_l77_77219

variable (S : ℝ)

-- Assume Mr. Blue receives a 12% raise every year.
def annualRaise : ℝ := 1.12

-- After four years
theorem salary_increase_is_57point35_percent (h : annualRaise ^ 4 = 1.5735):
  ((annualRaise ^ 4 - 1) * S) / S = 0.5735 :=
by
  sorry

end salary_increase_is_57point35_percent_l77_77219


namespace bernie_savings_l77_77526

-- Defining conditions
def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def chocolates_total : ℕ := chocolates_per_week * weeks
def local_store_cost_per_chocolate : ℕ := 3
def different_store_cost_per_chocolate : ℕ := 2

-- Defining the costs in both stores
def local_store_total_cost : ℕ := chocolates_total * local_store_cost_per_chocolate
def different_store_total_cost : ℕ := chocolates_total * different_store_cost_per_chocolate

-- The statement we want to prove
theorem bernie_savings : local_store_total_cost - different_store_total_cost = 6 :=
by
  sorry

end bernie_savings_l77_77526


namespace mn_sum_eq_neg_one_l77_77837

theorem mn_sum_eq_neg_one (m n : ℤ) (h : (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m * x + n)) :
  m + n = -1 :=
sorry

end mn_sum_eq_neg_one_l77_77837


namespace total_number_of_cows_l77_77412

variable (D C : ℕ) -- D is the number of ducks and C is the number of cows

-- Define the condition given in the problem
def legs_eq : Prop := 2 * D + 4 * C = 2 * (D + C) + 28

theorem total_number_of_cows (h : legs_eq D C) : C = 14 := by
  sorry

end total_number_of_cows_l77_77412


namespace solve_for_s_l77_77666

noncomputable def compute_s : Set ℝ :=
  { s | ∀ (x : ℝ), (x ≠ -1) → ((s * x - 3) / (x + 1) = x ↔ x^2 + (1 - s) * x + 3 = 0) ∧
    ((1 - s) ^ 2 - 4 * 3 = 0) }

theorem solve_for_s (h : ∀ s ∈ compute_s, s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :
  compute_s = {1 + 2 * Real.sqrt 3, 1 - 2 * Real.sqrt 3} :=
by
  sorry

end solve_for_s_l77_77666


namespace calc_expression1_calc_expression2_l77_77353

-- Problem 1
theorem calc_expression1 (x y : ℝ) : (1/2 * x * y)^2 * 6 * x^2 * y = (3/2) * x^4 * y^3 := 
sorry

-- Problem 2
theorem calc_expression2 (a b : ℝ) : (2 * a + b)^2 = 4 * a^2 + 4 * a * b + b^2 := 
sorry

end calc_expression1_calc_expression2_l77_77353


namespace point_symmetric_to_line_l77_77279

-- Define the problem statement
theorem point_symmetric_to_line (M : ℝ × ℝ) (l : ℝ × ℝ) (N : ℝ × ℝ) :
  M = (1, 4) →
  l = (1, -1) →
  (∃ a b, N = (a, b) ∧ a + b = 5 ∧ a - b = 1) →
  N = (3, 2) :=
by
  sorry

end point_symmetric_to_line_l77_77279


namespace complete_square_transform_l77_77222

theorem complete_square_transform :
  ∀ x : ℝ, x^2 - 4 * x - 6 = 0 → (x - 2)^2 = 10 :=
by
  intros x h
  sorry

end complete_square_transform_l77_77222


namespace marc_journey_fraction_l77_77003

-- Defining the problem based on identified conditions
def total_cycling_time (k : ℝ) : ℝ := 20 * k
def total_walking_time (k : ℝ) : ℝ := 60 * (1 - k)
def total_travel_time (k : ℝ) : ℝ := total_cycling_time k + total_walking_time k

theorem marc_journey_fraction:
  ∀ (k : ℝ), total_travel_time k = 52 → k = 1 / 5 :=
by
  sorry

end marc_journey_fraction_l77_77003


namespace exists_unique_inverse_l77_77856

theorem exists_unique_inverse (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h_gcd : Nat.gcd p a = 1) : 
  ∃! (b : ℕ), b ∈ Finset.range p ∧ (a * b) % p = 1 := 
sorry

end exists_unique_inverse_l77_77856


namespace harold_grocery_expense_l77_77090

theorem harold_grocery_expense:
  ∀ (income rent car_payment savings utilities remaining groceries : ℝ),
    income = 2500 →
    rent = 700 →
    car_payment = 300 →
    utilities = 0.5 * car_payment →
    remaining = income - rent - car_payment - utilities →
    savings = 0.5 * remaining →
    (remaining - savings) = 650 →
    groceries = (remaining - 650) →
    groceries = 50 :=
by
  intros income rent car_payment savings utilities remaining groceries
  intro h_income
  intro h_rent
  intro h_car_payment
  intro h_utilities
  intro h_remaining
  intro h_savings
  intro h_final_remaining
  intro h_groceries
  sorry

end harold_grocery_expense_l77_77090


namespace courtyard_length_is_60_l77_77105

noncomputable def stone_length : ℝ := 2.5
noncomputable def stone_breadth : ℝ := 2.0
noncomputable def num_stones : ℕ := 198
noncomputable def courtyard_breadth : ℝ := 16.5

theorem courtyard_length_is_60 :
  ∃ (courtyard_length : ℝ), courtyard_length = 60 ∧
  num_stones * (stone_length * stone_breadth) = courtyard_length * courtyard_breadth :=
sorry

end courtyard_length_is_60_l77_77105


namespace line_equation_l77_77482

variable (x y : ℝ)

theorem line_equation (x1 y1 m : ℝ) (h : x1 = -2 ∧ y1 = 3 ∧ m = 2) :
    -2 * x + y = 1 := by
  sorry

end line_equation_l77_77482


namespace max_quarters_l77_77635

/-- Prove that given the conditions for the number of nickels, dimes, and quarters,
    the maximum number of quarters can be 20. --/
theorem max_quarters {a b c : ℕ} (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) :
  c ≤ 20 :=
sorry

end max_quarters_l77_77635


namespace reflect_y_axis_l77_77445

theorem reflect_y_axis (x y z : ℝ) : (x, y, z) = (1, -2, 3) → (-x, y, -z) = (-1, -2, -3) :=
by
  intros
  sorry

end reflect_y_axis_l77_77445


namespace vector_CD_l77_77300

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b c : V)
variable (h1 : B - A = a)
variable (h2 : B - C = b)
variable (h3 : D - A = c)

theorem vector_CD :
  D - C = -a + b + c :=
by
  -- Proof omitted
  sorry

end vector_CD_l77_77300


namespace total_fish_is_22_l77_77848

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7
def total_fish : ℕ := gold_fish + blue_fish

theorem total_fish_is_22 : total_fish = 22 :=
by
  -- the proof should be written here
  sorry

end total_fish_is_22_l77_77848


namespace moving_circle_passes_through_fixed_point_l77_77898
-- We will start by importing the necessary libraries and setting up the problem conditions.

-- Define the parabola y^2 = 8x.
def parabola (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = 8 * p.1

-- Define the line x + 2 = 0.
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 = -2

-- Define the fixed point.
def fixed_point : ℝ × ℝ :=
  (2, 0)

-- Define the moving circle passing through the fixed point.
def moving_circle (p : ℝ × ℝ) (c : ℝ × ℝ) :=
  p = fixed_point

-- Bring it all together in the theorem.
theorem moving_circle_passes_through_fixed_point (c : ℝ × ℝ) (p : ℝ × ℝ)
  (h_parabola : parabola c)
  (h_tangent : tangent_line p) :
  moving_circle p c :=
sorry

end moving_circle_passes_through_fixed_point_l77_77898


namespace apples_shared_equally_l77_77942

-- Definitions of the given conditions
def num_apples : ℕ := 9
def num_friends : ℕ := 3

-- Statement of the problem
theorem apples_shared_equally : num_apples / num_friends = 3 := by
  sorry

end apples_shared_equally_l77_77942


namespace spending_difference_l77_77246

-- Define the given conditions
def ice_cream_cartons := 19
def yoghurt_cartons := 4
def ice_cream_cost_per_carton := 7
def yoghurt_cost_per_carton := 1

-- Calculate the total cost based on the given conditions
def total_ice_cream_cost := ice_cream_cartons * ice_cream_cost_per_carton
def total_yoghurt_cost := yoghurt_cartons * yoghurt_cost_per_carton

-- The statement to prove
theorem spending_difference :
  total_ice_cream_cost - total_yoghurt_cost = 129 :=
by
  sorry

end spending_difference_l77_77246


namespace tan_neg_585_eq_neg_1_l77_77831

theorem tan_neg_585_eq_neg_1 : Real.tan (-585 * Real.pi / 180) = -1 := by
  sorry

end tan_neg_585_eq_neg_1_l77_77831


namespace negation_of_existential_l77_77590

theorem negation_of_existential :
  (¬ (∃ x : ℝ, x^2 - x - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end negation_of_existential_l77_77590


namespace least_hourly_number_l77_77548

def is_clock_equivalent (a b : ℕ) : Prop := (a - b) % 12 = 0

theorem least_hourly_number : ∃ n ≥ 6, is_clock_equivalent n (n * n) ∧ ∀ m ≥ 6, is_clock_equivalent m (m * m) → 9 ≤ m → n = 9 := 
by
  sorry

end least_hourly_number_l77_77548


namespace find_a_l77_77873

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (2^x) = x + 3) (h2 : f a = 5) : a = 4 := 
by
  sorry

end find_a_l77_77873


namespace floor_expression_correct_l77_77618

theorem floor_expression_correct :
  (∃ x : ℝ, x = 2007 ^ 3 / (2005 * 2006) - 2005 ^ 3 / (2006 * 2007) ∧ ⌊x⌋ = 8) := 
sorry

end floor_expression_correct_l77_77618


namespace totalBottleCaps_l77_77438

-- Variables for the conditions
def bottleCapsPerBox : ℝ := 35.0
def numberOfBoxes : ℝ := 7.0

-- Theorem stating the equivalent proof problem
theorem totalBottleCaps : bottleCapsPerBox * numberOfBoxes = 245.0 := by
  sorry

end totalBottleCaps_l77_77438


namespace count_four_digit_numbers_with_5_or_7_l77_77425

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l77_77425


namespace t_bounds_f_bounds_l77_77640

noncomputable def t (x : ℝ) : ℝ := 3^x

noncomputable def f (x : ℝ) : ℝ := 9^x - 2 * 3^x + 4

theorem t_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (1/3 ≤ t x ∧ t x ≤ 9) :=
sorry

theorem f_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (3 ≤ f x ∧ f x ≤ 67) :=
sorry

end t_bounds_f_bounds_l77_77640


namespace factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l77_77699

-- Problem 1: Prove the factorization of x^4 - 3x^2 + 1
theorem factorize_x4_minus_3x2_plus_1 (x : ℝ) : 
  x^4 - 3 * x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := 
by
  sorry

-- Problem 2: Prove the factorization of a^5 + a^4 - 2a + 1
theorem factorize_a5_plus_a4_minus_2a_plus_1 (a : ℝ) : 
  a^5 + a^4 - 2 * a + 1 = (a^2 + a - 1) * (a^3 + a - 1) := 
by
  sorry

-- Problem 3: Prove the factorization of m^5 - 2m^3 - m - 1
theorem factorize_m5_minus_2m3_minus_m_minus_1 (m : ℝ) : 
  m^5 - 2 * m^3 - m - 1 = (m^3 + m^2 + 1) * (m^2 - m - 1) := 
by
  sorry

end factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l77_77699


namespace complex_addition_l77_77086

def c : ℂ := 3 - 2 * Complex.I
def d : ℂ := 1 + 3 * Complex.I

theorem complex_addition : 3 * c + 4 * d = 13 + 6 * Complex.I := by
  -- proof goes here
  sorry

end complex_addition_l77_77086


namespace BoatsRUs_canoes_l77_77917

theorem BoatsRUs_canoes :
  let a := 6
  let r := 3
  let n := 5
  let S := a * (r^n - 1) / (r - 1)
  S = 726 := by
  -- Proof
  sorry

end BoatsRUs_canoes_l77_77917


namespace expand_expression_l77_77975

variable {x y z : ℝ}

theorem expand_expression :
  (2 * x + 5) * (3 * y + 15 + 4 * z) = 6 * x * y + 30 * x + 8 * x * z + 15 * y + 20 * z + 75 :=
by
  sorry

end expand_expression_l77_77975


namespace pie_remaining_portion_l77_77638

theorem pie_remaining_portion (Carlos_share Maria_share remaining: ℝ)
  (hCarlos : Carlos_share = 0.65)
  (hRemainingAfterCarlos : remaining = 1 - Carlos_share)
  (hMaria : Maria_share = remaining / 2) :
  remaining - Maria_share = 0.175 :=
by
  sorry

end pie_remaining_portion_l77_77638


namespace speed_of_first_train_l77_77351

noncomputable def length_of_first_train : ℝ := 280
noncomputable def speed_of_second_train_kmph : ℝ := 80
noncomputable def length_of_second_train : ℝ := 220.04
noncomputable def time_to_cross : ℝ := 9

noncomputable def relative_speed_mps := (length_of_first_train + length_of_second_train) / time_to_cross

noncomputable def relative_speed_kmph := relative_speed_mps * (3600 / 1000)

theorem speed_of_first_train :
  (relative_speed_kmph - speed_of_second_train_kmph) = 120.016 :=
by
  sorry

end speed_of_first_train_l77_77351


namespace sodium_acetate_formed_is_3_l77_77883

-- Definitions for chemicals involved in the reaction
def AceticAcid : Type := ℕ -- Number of moles of acetic acid
def SodiumHydroxide : Type := ℕ -- Number of moles of sodium hydroxide
def SodiumAcetate : Type := ℕ -- Number of moles of sodium acetate

-- Given conditions as definitions
def reaction (acetic_acid naoh : ℕ) : ℕ :=
  if acetic_acid = naoh then acetic_acid else min acetic_acid naoh

-- Lean theorem statement
theorem sodium_acetate_formed_is_3 
  (acetic_acid naoh : ℕ) 
  (h1 : acetic_acid = 3) 
  (h2 : naoh = 3) :
  reaction acetic_acid naoh = 3 :=
by
  -- Proof body (to be completed)
  sorry

end sodium_acetate_formed_is_3_l77_77883


namespace initial_workers_l77_77406

theorem initial_workers (M : ℝ) :
  let totalLength : ℝ := 15
  let totalDays : ℝ := 300
  let completedLength : ℝ := 2.5
  let completedDays : ℝ := 100
  let remainingLength : ℝ := totalLength - completedLength
  let remainingDays : ℝ := totalDays - completedDays
  let extraMen : ℝ := 60
  let rateWithM : ℝ := completedLength / completedDays
  let newRate : ℝ := remainingLength / remainingDays
  let newM : ℝ := M + extraMen
  (rateWithM * M = newRate * newM) → M = 100 :=
by
  intros h
  sorry

end initial_workers_l77_77406


namespace product_equals_one_l77_77586

theorem product_equals_one (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 / (1 + x + x^2)) + (1 / (1 + y + y^2)) + (1 / (1 + x + y)) = 1) : 
  x * y = 1 :=
by
  sorry

end product_equals_one_l77_77586


namespace maximum_value_f_zeros_l77_77161

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if 1 < x then k * x + 1
  else 0

theorem maximum_value_f_zeros (k : ℝ) (x1 x2 : ℝ) :
  0 < k ∧ ∀ x, f x k = 0 ↔ x = x1 ∨ x = x2 → x1 ≠ x2 →
  x1 > 0 → x2 > 0 → -1 < k ∧ k < 0 →
  (x1 = -1 / k) ∧ (x2 = 1 / (1 + Real.sqrt (1 + k))) →
  ∃ y, (1 / x1) + (1 / x2) = y ∧ y = 9 / 4 := sorry

end maximum_value_f_zeros_l77_77161


namespace intersection_M_N_l77_77919

-- Define set M
def M : Set Int := {-2, -1, 0, 1}

-- Define set N using the given condition
def N : Set Int := {n : Int | -1 <= n ∧ n <= 3}

-- State that the intersection of M and N is the set {-1, 0, 1}
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l77_77919


namespace polynomial_has_root_l77_77760

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l77_77760


namespace negation_equiv_l77_77483

theorem negation_equiv (p : Prop) : 
  (p = (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) → 
  (¬ p = (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by
  sorry

end negation_equiv_l77_77483


namespace lights_ratio_l77_77095

theorem lights_ratio (M S L : ℕ) (h1 : M = 12) (h2 : S = M + 10) (h3 : 118 = (S * 1) + (M * 2) + (L * 3)) :
  L = 24 ∧ L / M = 2 :=
by
  sorry

end lights_ratio_l77_77095


namespace find_k_l77_77665

noncomputable def series_sum (k : ℝ) : ℝ :=
  3 + ∑' (n : ℕ), (3 + (n + 1) * k) / 4^(n + 1)

theorem find_k : ∃ k : ℝ, series_sum k = 8 ∧ k = 9 :=
by
  use 9
  have h : series_sum 9 = 8 := sorry
  exact ⟨h, rfl⟩

end find_k_l77_77665


namespace cows_total_l77_77028

theorem cows_total {n : ℕ} :
  (n / 3) + (n / 6) + (n / 8) + (n / 24) + 15 = n ↔ n = 45 :=
by {
  sorry
}

end cows_total_l77_77028


namespace probability_even_sum_includes_ball_15_l77_77744

-- Definition of the conditions in Lean
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

def odd_balls : Set ℕ := {n ∈ balls | n % 2 = 1}
def even_balls : Set ℕ := {n ∈ balls | n % 2 = 0}
def ball_15 : ℕ := 15

-- The number of ways to choose k elements from a set of n elements
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Number of ways to draw 7 balls ensuring the sum is even and ball 15 is included
def favorable_outcomes : ℕ :=
  choose 6 5 * choose 8 1 +   -- 5 other odd and 1 even
  choose 6 3 * choose 8 3 +   -- 3 other odd and 3 even
  choose 6 1 * choose 8 5     -- 1 other odd and 5 even

-- Total number of ways to choose 7 balls including ball 15:
def total_outcomes : ℕ := choose 14 6

-- Probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- The proof we require
theorem probability_even_sum_includes_ball_15 :
  probability = 1504 / 3003 :=
by
  -- proof omitted for brevity
  sorry

end probability_even_sum_includes_ball_15_l77_77744


namespace find_k_l77_77828

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n-1)) / 2 * d

theorem find_k (a₁ d : ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h₁ : a₁ = 1) (h₂ : d = 2) (h₃ : ∀ n, S (n+2) = 28 + S n) :
  k = 6 := by
  sorry

end find_k_l77_77828


namespace cost_price_for_fabrics_l77_77242

noncomputable def total_cost_price (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  selling_price - (meters_sold * profit_per_meter)

noncomputable def cost_price_per_meter (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  total_cost_price meters_sold selling_price profit_per_meter / meters_sold

theorem cost_price_for_fabrics :
  cost_price_per_meter 45 6000 12 = 121.33 ∧
  cost_price_per_meter 60 10800 15 = 165 ∧
  cost_price_per_meter 30 3900 10 = 120 :=
by
  sorry

end cost_price_for_fabrics_l77_77242


namespace new_probability_of_blue_ball_l77_77066

theorem new_probability_of_blue_ball 
  (initial_total_balls : ℕ) (initial_blue_balls : ℕ) (removed_blue_balls : ℕ) :
  initial_total_balls = 18 →
  initial_blue_balls = 6 →
  removed_blue_balls = 3 →
  (initial_blue_balls - removed_blue_balls) / (initial_total_balls - removed_blue_balls) = 1 / 5 :=
by
  sorry

end new_probability_of_blue_ball_l77_77066


namespace intersection_eq_interval_l77_77228

def P : Set ℝ := {x | x * (x - 3) < 0}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_eq_interval : P ∩ Q = {x | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_eq_interval_l77_77228


namespace quadratic_completing_square_t_l77_77735

theorem quadratic_completing_square_t : 
  ∀ (x k t : ℝ), (4 * x^2 + 16 * x - 400 = 0) →
  ((x + k)^2 = t) →
  t = 104 :=
by
  intros x k t h1 h2
  sorry

end quadratic_completing_square_t_l77_77735


namespace sum_of_squares_l77_77247

theorem sum_of_squares (x y z a b c k : ℝ)
  (h₁ : x * y = k * a)
  (h₂ : x * z = b)
  (h₃ : y * z = c)
  (hk : k ≠ 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) :=
by
  sorry

end sum_of_squares_l77_77247


namespace count_multiples_l77_77925

theorem count_multiples (n : ℕ) : 
  n = 1 ↔ ∃ k : ℕ, k < 500 ∧ k > 0 ∧ k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0 ∧ k % 7 = 0 :=
by
  sorry

end count_multiples_l77_77925


namespace final_payment_order_450_l77_77270

noncomputable def finalPayment (orderAmount : ℝ) : ℝ :=
  let serviceCharge := if orderAmount < 500 then 0.04 * orderAmount
                      else if orderAmount < 1000 then 0.05 * orderAmount
                      else 0.06 * orderAmount
  let salesTax := if orderAmount < 500 then 0.05 * orderAmount
                  else if orderAmount < 1000 then 0.06 * orderAmount
                  else 0.07 * orderAmount
  let totalBeforeDiscount := orderAmount + serviceCharge + salesTax
  let discount := if totalBeforeDiscount < 600 then 0.05 * totalBeforeDiscount
                  else if totalBeforeDiscount < 800 then 0.10 * totalBeforeDiscount
                  else 0.15 * totalBeforeDiscount
  totalBeforeDiscount - discount

theorem final_payment_order_450 :
  finalPayment 450 = 465.98 := by
  sorry

end final_payment_order_450_l77_77270


namespace distance_from_M_to_x_axis_l77_77369

-- Define the point M and its coordinates.
def point_M : ℤ × ℤ := (-9, 12)

-- Define the distance to the x-axis is simply the absolute value of the y-coordinate.
def distance_to_x_axis (p : ℤ × ℤ) : ℤ := Int.natAbs p.snd

-- Theorem stating the distance from point M to the x-axis is 12.
theorem distance_from_M_to_x_axis : distance_to_x_axis point_M = 12 := by
  sorry

end distance_from_M_to_x_axis_l77_77369


namespace Oleg_age_proof_l77_77417

-- Defining the necessary conditions
variables (x y z : ℕ) -- defining the ages of Oleg, his father, and his grandfather

-- Stating the conditions
axiom h1 : y = x + 32
axiom h2 : z = y + 32
axiom h3 : (x - 3) + (y - 3) + (z - 3) < 100

-- Stating the proof problem
theorem Oleg_age_proof : 
  (x = 4) ∧ (y = 36) ∧ (z = 68) :=
by
  sorry

end Oleg_age_proof_l77_77417


namespace sufficient_not_necessary_condition_l77_77705

variable (a b : ℝ)

theorem sufficient_not_necessary_condition (h : a > |b|) : a^2 > b^2 :=
by 
  sorry

end sufficient_not_necessary_condition_l77_77705


namespace total_wheels_in_both_garages_l77_77960

/-- Each cycle type has a different number of wheels. --/
def wheels_per_cycle (cycle_type: String) : ℕ :=
  if cycle_type = "bicycle" then 2
  else if cycle_type = "tricycle" then 3
  else if cycle_type = "unicycle" then 1
  else if cycle_type = "quadracycle" then 4
  else 0

/-- Define the counts of each type of cycle in each garage. --/
def garage1_counts := [("bicycle", 5), ("tricycle", 6), ("unicycle", 9), ("quadracycle", 3)]
def garage2_counts := [("bicycle", 2), ("tricycle", 1), ("unicycle", 3), ("quadracycle", 4)]

/-- Total steps for the calculation --/
def wheels_in_garage (garage_counts: List (String × ℕ)) (missing_wheels_unicycles: ℕ) : ℕ :=
  List.foldl (λ acc (cycle_count: String × ℕ) => 
              acc + (if cycle_count.1 = "unicycle" then (cycle_count.2 * wheels_per_cycle cycle_count.1 - missing_wheels_unicycles) 
                     else (cycle_count.2 * wheels_per_cycle cycle_count.1))) 0 garage_counts

/-- The total number of wheels in both garages. --/
def total_wheels : ℕ := wheels_in_garage garage1_counts 0 + wheels_in_garage garage2_counts 3

/-- Prove that the total number of wheels in both garages is 72. --/
theorem total_wheels_in_both_garages : total_wheels = 72 :=
  by sorry

end total_wheels_in_both_garages_l77_77960


namespace find_n_equal_roots_l77_77355

theorem find_n_equal_roots (x n : ℝ) (hx : x ≠ 2) : n = -1 ↔
  let a := 1
  let b := -2
  let c := -(n^2 + 2 * n)
  b^2 - 4 * a * c = 0 :=
by
  sorry

end find_n_equal_roots_l77_77355


namespace algebra_expr_solution_l77_77758

theorem algebra_expr_solution (a b : ℝ) (h : 2 * a - b = 5) : 2 * b - 4 * a + 8 = -2 :=
by
  sorry

end algebra_expr_solution_l77_77758


namespace interest_rate_is_correct_l77_77447

variable (A P I : ℝ)
variable (T R : ℝ)

theorem interest_rate_is_correct
  (hA : A = 1232)
  (hP : P = 1100)
  (hT : T = 12 / 5)
  (hI : I = A - P) :
  R = I * 100 / (P * T) :=
by
  sorry

end interest_rate_is_correct_l77_77447


namespace eqidistant_point_on_x_axis_l77_77023

theorem eqidistant_point_on_x_axis (x : ℝ) : 
    (dist (x, 0) (-3, 0) = dist (x, 0) (2, 5)) → 
    x = 2 := by
  sorry

end eqidistant_point_on_x_axis_l77_77023


namespace unique_solution_l77_77653

theorem unique_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a * b - a - b = 1) : (a, b) = (3, 2) :=
by
  sorry

end unique_solution_l77_77653


namespace x_is_integer_if_conditions_hold_l77_77321

theorem x_is_integer_if_conditions_hold (x : ℝ)
  (h1 : ∃ (k : ℤ), x^2 - x = k)
  (h2 : ∃ (n : ℕ), n ≥ 3 ∧ ∃ (m : ℤ), x^n - x = m) :
  ∃ (z : ℤ), x = z :=
sorry

end x_is_integer_if_conditions_hold_l77_77321


namespace container_capacity_l77_77914

-- Define the given conditions
def initially_full (x : ℝ) : Prop := (1 / 4) * x + 300 = (3 / 4) * x

-- Define the proof problem to show that the total capacity is 600 liters
theorem container_capacity : ∃ x : ℝ, initially_full x → x = 600 := sorry

end container_capacity_l77_77914


namespace problem_statement_l77_77463

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - Real.pi * x

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  ((deriv f x < 0) ∧ (f x < 0)) :=
by
  sorry

end problem_statement_l77_77463


namespace find_n_l77_77761

theorem find_n (n : ℕ) (S : ℕ) (h1 : S = n * (n + 1) / 2)
  (h2 : ∃ a : ℕ, a > 0 ∧ a < 10 ∧ S = 111 * a) : n = 36 :=
sorry

end find_n_l77_77761


namespace total_volume_tetrahedra_l77_77924

theorem total_volume_tetrahedra (side_length : ℝ) (x : ℝ) (sqrt_2 : ℝ := Real.sqrt 2) 
  (cube_to_octa_length : x = 2 * (sqrt_2 - 1)) 
  (volume_of_one_tetra : ℝ := ((6 - 4 * sqrt_2) * (3 - sqrt_2)) / 6) :
  side_length = 2 → 
  8 * volume_of_one_tetra = (104 - 72 * sqrt_2) / 3 :=
by
  intros
  sorry

end total_volume_tetrahedra_l77_77924


namespace fraction_sum_eq_five_fourths_l77_77195

theorem fraction_sum_eq_five_fourths (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b) / c = 5 / 4 :=
by
  sorry

end fraction_sum_eq_five_fourths_l77_77195


namespace c_alone_finishes_in_6_days_l77_77746

theorem c_alone_finishes_in_6_days (a b c : ℝ) (W : ℝ) :
  (1 / 36) * W + (1 / 18) * W + (1 / c) * W = (1 / 4) * W → c = 6 :=
by
  intros h
  simp at h
  sorry

end c_alone_finishes_in_6_days_l77_77746


namespace john_subtraction_number_l77_77884

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l77_77884


namespace sue_received_votes_l77_77167

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end sue_received_votes_l77_77167


namespace simplify_expr_l77_77880

theorem simplify_expr : 
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = (5 : ℚ) / 4 := 
by
  sorry

end simplify_expr_l77_77880


namespace actual_selling_price_l77_77541

-- Define the original price m
variable (m : ℝ)

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the selling price
def selling_price := m * (1 - discount_rate)

-- The theorem states the relationship between the original price and the selling price after discount
theorem actual_selling_price : selling_price m = 0.8 * m :=
by
-- Proof step would go here
sorry

end actual_selling_price_l77_77541


namespace e_is_dq_sequence_l77_77484

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a₀, ∀ n, a n = a₀ + n * d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q b₀, q > 0 ∧ ∀ n, b n = b₀ * q^n

def is_dq_sequence (c : ℕ → ℕ) : Prop :=
  ∃ a b, is_arithmetic_sequence a ∧ is_geometric_sequence b ∧ ∀ n, c n = a n + b n

def e (n : ℕ) : ℕ :=
  n + 2^n

theorem e_is_dq_sequence : is_dq_sequence e :=
  sorry

end e_is_dq_sequence_l77_77484


namespace eagles_min_additional_wins_l77_77508

theorem eagles_min_additional_wins {N : ℕ} (eagles_initial_wins falcons_initial_wins : ℕ) (initial_games : ℕ)
  (total_games_won_fraction : ℚ) (required_fraction : ℚ) :
  eagles_initial_wins = 3 →
  falcons_initial_wins = 4 →
  initial_games = eagles_initial_wins + falcons_initial_wins →
  total_games_won_fraction = (3 + N) / (7 + N) →
  required_fraction = 9 / 10 →
  total_games_won_fraction = required_fraction →
  N = 33 :=
by
  sorry

end eagles_min_additional_wins_l77_77508


namespace only_one_real_solution_l77_77728

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem only_one_real_solution (a : ℝ) (h : ∀ x : ℝ, abs (f x) = g a x → x = 1) : a < 0 := 
by
  sorry

end only_one_real_solution_l77_77728


namespace arcsin_one_half_eq_pi_six_l77_77915

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry -- Proof omitted

end arcsin_one_half_eq_pi_six_l77_77915


namespace students_more_than_rabbits_l77_77979

/- Define constants for the problem. -/
def students_per_class : ℕ := 20
def rabbits_per_class : ℕ := 3
def num_classes : ℕ := 5

/- Define total counts based on given conditions. -/
def total_students : ℕ := students_per_class * num_classes
def total_rabbits : ℕ := rabbits_per_class * num_classes

/- The theorem we need to prove: The difference between total students and total rabbits is 85. -/
theorem students_more_than_rabbits : total_students - total_rabbits = 85 := by
  sorry

end students_more_than_rabbits_l77_77979


namespace total_students_l77_77022

-- Define the conditions
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 3 * (girls / 2)
def boys_girls_difference (boys girls : ℕ) : Prop := boys = girls + 20

-- Define the property to be proved
theorem total_students (boys girls : ℕ) 
  (h1 : ratio_boys_to_girls boys girls)
  (h2 : boys_girls_difference boys girls) :
  boys + girls = 100 :=
sorry

end total_students_l77_77022


namespace books_before_grant_l77_77803

-- Define the conditions 
def books_purchased_with_grant : ℕ := 2647
def total_books_now : ℕ := 8582

-- Prove the number of books before the grant
theorem books_before_grant : 
  (total_books_now - books_purchased_with_grant = 5935) := 
by
  sorry

end books_before_grant_l77_77803


namespace sandwiches_sold_out_l77_77331

-- Define the parameters as constant values
def original : ℕ := 9
def available : ℕ := 4

-- The theorem stating the problem and the expected result
theorem sandwiches_sold_out : (original - available) = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end sandwiches_sold_out_l77_77331


namespace no_such_rectangle_l77_77923

theorem no_such_rectangle (a b x y : ℝ) (ha : a < b)
  (hx : x < a / 2) (hy : y < a / 2)
  (h_perimeter : 2 * (x + y) = a + b)
  (h_area : x * y = (a * b) / 2) :
  false :=
sorry

end no_such_rectangle_l77_77923


namespace greatest_whole_number_satisfying_inequalities_l77_77969

theorem greatest_whole_number_satisfying_inequalities :
  ∃ x : ℕ, 3 * (x : ℤ) - 5 < 1 - x ∧ 2 * (x : ℤ) + 4 ≤ 8 ∧ ∀ y : ℕ, y > x → ¬ (3 * (y : ℤ) - 5 < 1 - y ∧ 2 * (y : ℤ) + 4 ≤ 8) :=
sorry

end greatest_whole_number_satisfying_inequalities_l77_77969


namespace kim_easy_round_correct_answers_l77_77782

variable (E : ℕ)

theorem kim_easy_round_correct_answers 
    (h1 : 2 * E + 3 * 2 + 5 * 4 = 38) : 
    E = 6 := 
sorry

end kim_easy_round_correct_answers_l77_77782


namespace original_fraction_is_two_thirds_l77_77334

theorem original_fraction_is_two_thirds
  (x y : ℕ)
  (h1 : x / (y + 1) = 1 / 2)
  (h2 : (x + 1) / y = 1) :
  x / y = 2 / 3 := by
  sorry

end original_fraction_is_two_thirds_l77_77334


namespace plane_equivalent_l77_77872

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2*s - 3*t, 1 + s, 4 - 3*s + t)

def plane_equation (x y z : ℝ) : Prop :=
  x - 7*y + 3*z - 8 = 0

theorem plane_equivalent :
  ∃ (s t : ℝ), parametric_plane s t = (x, y, z) ↔ plane_equation x y z :=
by
  sorry

end plane_equivalent_l77_77872


namespace pyramid_angles_sum_pi_over_four_l77_77398

theorem pyramid_angles_sum_pi_over_four :
  ∃ (α β : ℝ), 
    α + β = Real.pi / 4 ∧ 
    α = Real.arctan ((Real.sqrt 17 - 3) / 4) ∧ 
    β = Real.pi / 4 - Real.arctan ((Real.sqrt 17 - 3) / 4) :=
by
  sorry

end pyramid_angles_sum_pi_over_four_l77_77398


namespace trigonometric_relationship_l77_77333

noncomputable def a : ℝ := Real.sin (46 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (46 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (46 * Real.pi / 180)

theorem trigonometric_relationship : c > a ∧ a > b :=
by
  -- This is the statement part; the proof will be handled here
  sorry

end trigonometric_relationship_l77_77333


namespace reduced_price_per_dozen_l77_77926

variables {P R : ℝ}

theorem reduced_price_per_dozen
  (H1 : R = 0.6 * P)
  (H2 : 40 / P - 40 / R = 64) :
  R = 3 := 
sorry

end reduced_price_per_dozen_l77_77926


namespace alex_loan_difference_l77_77363

theorem alex_loan_difference :
  let P := (15000 : ℝ)
  let r1 := (0.08 : ℝ)
  let n := (2 : ℕ)
  let t := (12 : ℕ)
  let r2 := (0.09 : ℝ)
  
  -- Calculate the amount owed after 6 years with compound interest (first option)
  let A1_half := P * (1 + r1 / n)^(n * t / 2)
  let half_payment := A1_half / 2
  let remaining_balance := A1_half / 2
  let A1_final := remaining_balance * (1 + r1 / n)^(n * t / 2)
  
  -- Total payment for the first option
  let total1 := half_payment + A1_final
  
  -- Total payment for the second option (simple interest)
  let simple_interest := P * r2 * t
  let total2 := P + simple_interest
  
  -- Compute the positive difference
  let difference := abs (total1 - total2)
  
  difference = 24.59 :=
  by
  sorry

end alex_loan_difference_l77_77363


namespace correct_comprehensive_survey_l77_77081

-- Definitions for the types of surveys.
inductive Survey
| A : Survey
| B : Survey
| C : Survey
| D : Survey

-- Function that identifies the survey suitable for a comprehensive survey.
def is_comprehensive_survey (s : Survey) : Prop :=
  match s with
  | Survey.A => False            -- A is for sampling, not comprehensive
  | Survey.B => False            -- B is for sampling, not comprehensive
  | Survey.C => False            -- C is for sampling, not comprehensive
  | Survey.D => True             -- D is suitable for comprehensive survey

-- The theorem to prove that D is the correct answer.
theorem correct_comprehensive_survey : is_comprehensive_survey Survey.D = True := by
  sorry

end correct_comprehensive_survey_l77_77081


namespace higher_concentration_acid_solution_l77_77886

theorem higher_concentration_acid_solution (x : ℝ) (h1 : 2 * (8 / 100 : ℝ) = 1.2 * (x / 100) + 0.8 * (5 / 100)) : x = 10 :=
sorry

end higher_concentration_acid_solution_l77_77886


namespace find_positive_real_solution_l77_77253

theorem find_positive_real_solution (x : ℝ) (h₁ : 0 < x) (h₂ : 1/2 * (4 * x ^ 2 - 4) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 410 :=
by
  sorry

end find_positive_real_solution_l77_77253


namespace sad_employees_left_geq_cheerful_l77_77107

-- Define the initial number of sad employees
def initial_sad_employees : Nat := 36

-- Define the final number of remaining employees after the game
def final_remaining_employees : Nat := 1

-- Define the total number of employees hit and out of the game
def employees_out : Nat := initial_sad_employees - final_remaining_employees

-- Define the number of cheerful employees who have left
def cheerful_employees_left := employees_out

-- Define the number of sad employees who have left
def sad_employees_left := employees_out

-- The theorem stating the problem proof
theorem sad_employees_left_geq_cheerful:
    sad_employees_left ≥ cheerful_employees_left :=
by
  -- Proof is omitted
  sorry

end sad_employees_left_geq_cheerful_l77_77107


namespace find_circle_parameter_l77_77202

theorem find_circle_parameter (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y + c = 0 ∧ ((x + 4)^2 + (y - 1)^2 = 25)) → c = -8 :=
by
  sorry

end find_circle_parameter_l77_77202


namespace average_median_eq_l77_77366

theorem average_median_eq (a b c : ℤ) (h1 : (a + b + c) / 3 = 4 * b)
  (h2 : a < b) (h3 : b < c) (h4 : a = 0) : c / b = 11 := 
by
  sorry

end average_median_eq_l77_77366


namespace major_axis_length_l77_77743

noncomputable def length_of_major_axis (f1 f2 : ℝ × ℝ) (tangent_y_axis : Bool) (tangent_line_y : ℝ) : ℝ :=
  if f1 = (-Real.sqrt 5, 2) ∧ f2 = (Real.sqrt 5, 2) ∧ tangent_y_axis ∧ tangent_line_y = 1 then 2
  else 0

theorem major_axis_length :
  length_of_major_axis (-Real.sqrt 5, 2) (Real.sqrt 5, 2) true 1 = 2 :=
by
  sorry

end major_axis_length_l77_77743


namespace num_dimes_is_3_l77_77189

noncomputable def num_dimes (pennies nickels dimes quarters : ℕ) : ℕ :=
  dimes

theorem num_dimes_is_3 (h_total_coins : pennies + nickels + dimes + quarters = 11)
  (h_total_value : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 118)
  (h_at_least_one_each : 0 < pennies ∧ 0 < nickels ∧ 0 < dimes ∧ 0 < quarters) :
  num_dimes pennies nickels dimes quarters = 3 :=
sorry

end num_dimes_is_3_l77_77189


namespace necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l77_77134

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition_for_increasing_geometric_sequence
  (a : ℕ → ℝ)
  (h0 : a 0 > 0)
  (h_geom : is_geometric_sequence a) :
  (a 0^2 < a 1^2) ↔ (is_increasing_sequence a) ∧ ¬ (∀ n, a n > 0 → a (n + 1) > 0) :=
sorry

end necessary_but_not_sufficient_condition_for_increasing_geometric_sequence_l77_77134


namespace ellipse_focal_distance_l77_77550

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 16 + y^2 / m = 1) ∧ (2 * Real.sqrt (16 - m) = 2 * Real.sqrt 7)) → m = 9 :=
by
  intro h
  sorry

end ellipse_focal_distance_l77_77550


namespace total_money_in_dollars_l77_77470

/-- You have some amount in nickels and quarters.
    You have 40 nickels and the same number of quarters.
    Prove that the total amount of money in dollars is 12. -/
theorem total_money_in_dollars (n_nickels n_quarters : ℕ) (value_nickel value_quarter : ℕ) 
  (h1: n_nickels = 40) (h2: n_quarters = 40) (h3: value_nickel = 5) (h4: value_quarter = 25) : 
  (n_nickels * value_nickel + n_quarters * value_quarter) / 100 = 12 :=
  sorry

end total_money_in_dollars_l77_77470


namespace find_positive_integer_M_l77_77911

theorem find_positive_integer_M (M : ℕ) (h : 36^2 * 81^2 = 18^2 * M^2) : M = 162 := by
  sorry

end find_positive_integer_M_l77_77911


namespace trig_identity_condition_l77_77310

open Real

theorem trig_identity_condition (a : Real) (h : ∃ x ≥ 0, (tan a = -1 ∧ cos a ≠ 0)) :
  (sin a / sqrt (1 - sin a ^ 2) + sqrt (1 - cos a ^ 2) / cos a) = 0 :=
by
  sorry

end trig_identity_condition_l77_77310


namespace area_triangle_tangent_circles_l77_77092

theorem area_triangle_tangent_circles :
  ∃ (A B C : Type) (radius1 radius2 : ℝ) 
    (tangent1 tangent2 : ℝ → ℝ → Prop)
    (congruent_sides : ℝ → Prop),
    radius1 = 1 ∧ radius2 = 2 ∧
    (∀ x y, tangent1 x y) ∧ (∀ x y, tangent2 x y) ∧
    congruent_sides 1 ∧ congruent_sides 2 ∧
    ∃ (area : ℝ), area = 16 * Real.sqrt 2 :=
by
  -- This is where the proof would be written
  sorry

end area_triangle_tangent_circles_l77_77092


namespace least_number_to_add_l77_77178

theorem least_number_to_add (x : ℕ) (h : 1056 % 23 = 21) : (1056 + x) % 23 = 0 ↔ x = 2 :=
by {
    sorry
}

end least_number_to_add_l77_77178


namespace find_x_from_percentage_l77_77570

theorem find_x_from_percentage : 
  ∃ x : ℚ, 0.65 * x = 0.20 * 487.50 := 
sorry

end find_x_from_percentage_l77_77570


namespace clock_confusion_times_l77_77154

-- Conditions translated into Lean definitions
def h_move : ℝ := 0.5  -- hour hand moves at 0.5 degrees per minute
def m_move : ℝ := 6.0  -- minute hand moves at 6 degrees per minute

-- Overlap condition formulated
def overlap_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 10 ∧ 11 * (n : ℝ) = k * 360

-- The final theorem statement in Lean 4
theorem clock_confusion_times : 
  ∃ (count : ℕ), count = 132 ∧ 
    (∀ n < 144, (overlap_condition n → false)) :=
by
  -- Proof to be inserted here
  sorry

end clock_confusion_times_l77_77154


namespace number_of_students_l77_77956

theorem number_of_students (groups : ℕ) (students_per_group : ℕ) (minutes_per_student : ℕ) (minutes_per_group : ℕ) :
    groups = 3 →
    minutes_per_student = 4 →
    minutes_per_group = 24 →
    minutes_per_group = students_per_group * minutes_per_student →
    18 = groups * students_per_group :=
by
  intros h_groups h_minutes_per_student h_minutes_per_group h_relation
  sorry

end number_of_students_l77_77956


namespace part1_part2_l77_77867

open Set

-- Definitions from conditions in a)
def R : Set ℝ := univ
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Question part (1)
theorem part1 (a : ℝ) (h : a = 1) :
  (compl A) ∪ B a = {x | x ≤ -2 ∨ x > 1} :=
by 
  simp [h]
  sorry

-- Question part (2)
theorem part2 (a : ℝ) :
  A ⊆ B a → a ≤ -2 :=
by 
  sorry

end part1_part2_l77_77867


namespace running_speed_l77_77863

variables (w t_w t_r : ℝ)

-- Given conditions
def walking_speed : w = 8 := sorry
def walking_time_hours : t_w = 4.75 := sorry
def running_time_hours : t_r = 2 := sorry

-- Prove the man's running speed
theorem running_speed (w t_w t_r : ℝ) 
  (H1 : w = 8) 
  (H2 : t_w = 4.75) 
  (H3 : t_r = 2) : 
  (w * t_w) / t_r = 19 := 
sorry

end running_speed_l77_77863


namespace Nadine_pebbles_l77_77834

theorem Nadine_pebbles :
  ∀ (white red blue green x : ℕ),
    white = 20 →
    red = white / 2 →
    blue = red / 3 →
    green = blue + 5 →
    red = (1/5) * x →
    x = 50 :=
by
  intros white red blue green x h_white h_red h_blue h_green h_percentage
  sorry

end Nadine_pebbles_l77_77834


namespace eddie_games_l77_77215

-- Define the study block duration in minutes
def study_block_duration : ℕ := 60

-- Define the homework time in minutes
def homework_time : ℕ := 25

-- Define the time for one game in minutes
def game_time : ℕ := 5

-- Define the total time Eddie can spend playing games
noncomputable def time_for_games : ℕ := study_block_duration - homework_time

-- Define the number of games Eddie can play
noncomputable def number_of_games : ℕ := time_for_games / game_time

-- Theorem stating the number of games Eddie can play while completing his homework
theorem eddie_games : number_of_games = 7 := by
  sorry

end eddie_games_l77_77215


namespace quadratic_eqn_a_range_l77_77488

variable {a : ℝ}

theorem quadratic_eqn_a_range (a : ℝ) : (∃ x : ℝ, (a - 3) * x^2 - 4 * x + 1 = 0) ↔ a ≠ 3 :=
by sorry

end quadratic_eqn_a_range_l77_77488


namespace Connor_spends_36_dollars_l77_77854

def tickets_cost := 10.00
def combo_meal_cost := 11.00
def candy_cost := 2.50
def total_cost := tickets_cost * 2 + combo_meal_cost + candy_cost * 2

theorem Connor_spends_36_dollars : total_cost = 36.00 := 
by 
  sorry

end Connor_spends_36_dollars_l77_77854


namespace bill_sunday_miles_l77_77616

-- Define the variables
variables (B S J : ℕ) -- B for miles Bill ran on Saturday, S for miles Bill ran on Sunday, J for miles Julia ran on Sunday

-- State the conditions
def condition1 (B S : ℕ) : Prop := S = B + 4
def condition2 (B S J : ℕ) : Prop := J = 2 * S
def condition3 (B S J : ℕ) : Prop := B + S + J = 20

-- The final theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (B S J : ℕ) 
  (h1 : condition1 B S)
  (h2 : condition2 B S J)
  (h3 : condition3 B S J) : 
  S = 6 := 
sorry

end bill_sunday_miles_l77_77616


namespace sqrt_9_minus_2_pow_0_plus_abs_neg1_l77_77130

theorem sqrt_9_minus_2_pow_0_plus_abs_neg1 :
  (Real.sqrt 9 - 2^0 + abs (-1) = 3) :=
by
  -- Proof omitted for brevity
  sorry

end sqrt_9_minus_2_pow_0_plus_abs_neg1_l77_77130


namespace jogger_ahead_of_train_l77_77624

theorem jogger_ahead_of_train (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (time_to_pass : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 100) 
  (h4 : time_to_pass = 34) : 
  ∃ d : ℝ, d = 240 :=
by
  sorry

end jogger_ahead_of_train_l77_77624


namespace kekai_garage_sale_l77_77137

theorem kekai_garage_sale :
  let shirts := 5
  let shirt_price := 1
  let pants := 5
  let pant_price := 3
  let total_money := (shirts * shirt_price) + (pants * pant_price)
  let money_kept := total_money / 2
  money_kept = 10 :=
by
  sorry

end kekai_garage_sale_l77_77137


namespace at_least_3_defective_correct_l77_77977

/-- Number of products in batch -/
def total_products : ℕ := 50

/-- Number of defective products -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

/-- Number of ways to draw at least 3 defective products out of 5 -/
def num_ways_at_least_3_defective : ℕ :=
  (Nat.choose defective_products 4) * (Nat.choose (total_products - defective_products) 1) +
  (Nat.choose defective_products 3) * (Nat.choose (total_products - defective_products) 2)

theorem at_least_3_defective_correct : num_ways_at_least_3_defective = 4186 := by
  sorry

end at_least_3_defective_correct_l77_77977


namespace b_work_time_l77_77632

theorem b_work_time (W : ℝ) (days_A days_combined : ℝ)
  (hA : W / days_A = W / 16)
  (h_combined : W / days_combined = W / (16 / 3)) :
  ∃ days_B, days_B = 8 :=
by
  sorry

end b_work_time_l77_77632


namespace intersecting_lines_l77_77642

theorem intersecting_lines (p : ℝ) :
    (∃ x y : ℝ, y = 3 * x - 6 ∧ y = -4 * x + 8 ∧ y = 7 * x + p) ↔ p = -14 :=
by {
    sorry
}

end intersecting_lines_l77_77642


namespace sum_of_numbers_l77_77408

theorem sum_of_numbers : (4.75 + 0.303 + 0.432) = 5.485 := 
by  
  sorry

end sum_of_numbers_l77_77408


namespace smallest_possible_value_l77_77436

theorem smallest_possible_value (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) : 
  ∃ (m : ℝ), m = -1/12 ∧ (∀ x y : ℝ, (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → (x + y) / (x^2) ≥ m) :=
sorry

end smallest_possible_value_l77_77436


namespace banana_difference_l77_77025

theorem banana_difference (d : ℕ) :
  (8 + (8 + d) + (8 + 2 * d) + (8 + 3 * d) + (8 + 4 * d) = 100) →
  d = 6 :=
by
  sorry

end banana_difference_l77_77025


namespace solve_for_x_l77_77841

theorem solve_for_x (x : ℝ) (h : 3 - (1 / (2 - x)) = (1 / (2 - x))) : x = 4 / 3 := 
by {
  sorry
}

end solve_for_x_l77_77841


namespace max_area_rectangle_shorter_side_l77_77844

theorem max_area_rectangle_shorter_side (side_length : ℕ) (n : ℕ)
  (hsq : side_length = 40) (hn : n = 5) :
  ∃ (shorter_side : ℕ), shorter_side = 8 := by
  sorry

end max_area_rectangle_shorter_side_l77_77844


namespace circles_through_two_points_in_4x4_grid_l77_77738

noncomputable def number_of_circles (n : ℕ) : ℕ :=
  if n = 4 then
    52
  else
    sorry

theorem circles_through_two_points_in_4x4_grid :
  number_of_circles 4 = 52 :=
by
  exact rfl  -- Reflexivity of equality shows the predefined value of 52

end circles_through_two_points_in_4x4_grid_l77_77738


namespace campaign_meaning_l77_77085

-- Define a function that gives the meaning of "campaign" as a noun
def meaning_of_campaign_noun : String :=
  "campaign, activity"

-- The theorem asserts that the meaning of "campaign" as a noun is "campaign, activity"
theorem campaign_meaning : meaning_of_campaign_noun = "campaign, activity" :=
by
  -- We add sorry here because we are not required to provide the proof
  sorry

end campaign_meaning_l77_77085


namespace initial_average_correct_l77_77337

theorem initial_average_correct (A : ℕ) 
  (num_students : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (wrong_avg : ℕ) (correct_avg : ℕ) 
  (h1 : num_students = 30)
  (h2 : wrong_mark = 70)
  (h3 : correct_mark = 10)
  (h4 : correct_avg = 98)
  (h5 : num_students * correct_avg = (num_students * A) - (wrong_mark - correct_mark)) :
  A = 100 := 
sorry

end initial_average_correct_l77_77337


namespace intersection_of_P_and_Q_l77_77458

def P : Set ℤ := {-3, -2, 0, 2}
def Q : Set ℤ := {-1, -2, -3, 0, 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-3, -2, 0} := by
  sorry

end intersection_of_P_and_Q_l77_77458


namespace find_y_when_x4_l77_77456

theorem find_y_when_x4 : 
  (∀ x y : ℚ, 5 * y + 3 = 344 / (x ^ 3)) ∧ (5 * (8:ℚ) + 3 = 344 / (2 ^ 3)) → 
  (∃ y : ℚ, 5 * y + 3 = 344 / (4 ^ 3) ∧ y = 19 / 40) := 
by
  sorry

end find_y_when_x4_l77_77456


namespace radha_profit_percentage_l77_77142

theorem radha_profit_percentage (SP CP : ℝ) (hSP : SP = 144) (hCP : CP = 90) :
  ((SP - CP) / CP) * 100 = 60 := by
  sorry

end radha_profit_percentage_l77_77142


namespace find_a_l77_77781

theorem find_a (a : ℝ) (h : 0.005 * a = 65) : a = 13000 / 100 :=
by
  sorry

end find_a_l77_77781


namespace triangle_area_ratio_l77_77706

theorem triangle_area_ratio (a b c a' b' c' r : ℝ)
    (h1 : a^2 + b^2 = c^2)
    (h2 : a'^2 + b'^2 = c'^2)
    (h3 : r = c' / 2)
    (S : ℝ := (1/2) * a * b)
    (S' : ℝ := (1/2) * a' * b') :
    S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_ratio_l77_77706


namespace people_got_off_at_second_stop_l77_77592

theorem people_got_off_at_second_stop (x : ℕ) :
  (10 - x) + 20 - 18 + 2 = 12 → x = 2 :=
  by sorry

end people_got_off_at_second_stop_l77_77592


namespace telescope_visual_range_increased_l77_77335

/-- A certain telescope increases the visual range from 100 kilometers to 150 kilometers. 
    Proof that the visual range is increased by 50% using the telescope.
-/
theorem telescope_visual_range_increased :
  let original_range := 100
  let new_range := 150
  (new_range - original_range) / original_range * 100 = 50 := 
by
  sorry

end telescope_visual_range_increased_l77_77335


namespace area_of_circle_l77_77784

theorem area_of_circle 
  (r : ℝ → ℝ)
  (h : ∀ θ : ℝ, r θ = 3 * Real.cos θ - 4 * Real.sin θ) :
  ∃ A : ℝ, A = (25 / 4) * Real.pi :=
by
  sorry

end area_of_circle_l77_77784


namespace final_temperature_is_correct_l77_77051

def initial_temperature : ℝ := 40
def after_jerry_temperature (T : ℝ) : ℝ := 2 * T
def after_dad_temperature (T : ℝ) : ℝ := T - 30
def after_mother_temperature (T : ℝ) : ℝ := T - 0.30 * T
def after_sister_temperature (T : ℝ) : ℝ := T + 24

theorem final_temperature_is_correct :
  after_sister_temperature (after_mother_temperature (after_dad_temperature (after_jerry_temperature initial_temperature))) = 59 :=
sorry

end final_temperature_is_correct_l77_77051


namespace valid_division_l77_77531

theorem valid_division (A B C E F G H K : ℕ) (hA : A = 7) (hB : B = 1) (hC : C = 2)
    (hE : E = 6) (hF : F = 8) (hG : G = 5) (hH : H = 4) (hK : K = 9) :
    (A * 10 + B) / ((C * 100 + A * 10 + B) / 100 + E + B * F * D) = 71 / 271 :=
by {
  sorry
}

end valid_division_l77_77531


namespace work_completion_days_l77_77899

theorem work_completion_days (Dx : ℕ) (Dy : ℕ) (days_y_worked : ℕ) (days_x_finished_remaining : ℕ)
  (work_rate_y : ℝ) (work_rate_x : ℝ) 
  (h1 : Dy = 24)
  (h2 : days_y_worked = 12)
  (h3 : days_x_finished_remaining = 18)
  (h4 : work_rate_y = 1 / Dy)
  (h5 : 12 * work_rate_y = 1 / 2)
  (h6 : work_rate_x = 1 / (2 * days_x_finished_remaining))
  (h7 : Dx * work_rate_x = 1) : Dx = 36 := sorry

end work_completion_days_l77_77899


namespace camryn_flute_practice_interval_l77_77210

theorem camryn_flute_practice_interval (x : ℕ) 
  (h1 : ∃ n : ℕ, n * 11 = 33) 
  (h2 : x ∣ 33) 
  (h3 : x < 11) 
  (h4 : x > 1) 
  : x = 3 := 
sorry

end camryn_flute_practice_interval_l77_77210


namespace percent_difference_l77_77188

theorem percent_difference :
  (0.90 * 40) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_l77_77188


namespace tiffany_cans_l77_77260

variable {M : ℕ}

theorem tiffany_cans : (M + 12 = 2 * M) → (M = 12) :=
by
  intro h
  sorry

end tiffany_cans_l77_77260


namespace student_A_more_stable_l77_77016

-- Defining the variances of students A and B as constants
def S_A_sq : ℝ := 0.04
def S_B_sq : ℝ := 0.13

-- Statement of the theorem
theorem student_A_more_stable : S_A_sq < S_B_sq → true :=
by
  -- proof will go here
  sorry

end student_A_more_stable_l77_77016


namespace quadratic_solution_l77_77441

theorem quadratic_solution :
  (∀ x : ℝ, (x^2 - x - 1 = 0) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2)) :=
by
  intro x
  rw [sub_eq_neg_add, sub_eq_neg_add]
  sorry

end quadratic_solution_l77_77441


namespace sum_of_squares_eq_expansion_l77_77478

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end sum_of_squares_eq_expansion_l77_77478


namespace hash_hash_hash_100_l77_77307

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_100 : hash (hash (hash 100)) = 11.08 :=
by sorry

end hash_hash_hash_100_l77_77307


namespace polynomial_value_l77_77691

variables (x y p q : ℝ)

theorem polynomial_value (h1 : x + y = -p) (h2 : xy = q) :
  x * (1 + y) - y * (x * y - 1) - x^2 * y = pq + q - p :=
by
  sorry

end polynomial_value_l77_77691


namespace ping_pong_tournament_l77_77545

theorem ping_pong_tournament :
  ∃ n: ℕ, 
    (∃ m: ℕ, m ≥ 0 ∧ m ≤ 2 ∧ 2 * n + m = 29) ∧
    n = 14 ∧
    (n + 2 = 16) := 
by {
  sorry
}

end ping_pong_tournament_l77_77545


namespace value_expression_l77_77030

-- Definitions
variable (m n : ℝ)
def reciprocals (m n : ℝ) := m * n = 1

-- Theorem statement
theorem value_expression (m n : ℝ) (h : reciprocals m n) : m * n^2 - (n - 3) = 3 := by
  sorry

end value_expression_l77_77030


namespace binary_to_decimal_1100_l77_77467

theorem binary_to_decimal_1100 : 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 12 := 
by
  sorry

end binary_to_decimal_1100_l77_77467


namespace chi_squared_confidence_l77_77468

theorem chi_squared_confidence (K_squared : ℝ) :
  (99.5 / 100 : ℝ) = 0.995 → (K_squared ≥ 7.879) :=
sorry

end chi_squared_confidence_l77_77468


namespace total_doughnuts_made_l77_77200

def num_doughnuts_per_box : ℕ := 10
def num_boxes_sold : ℕ := 27
def doughnuts_given_away : ℕ := 30

theorem total_doughnuts_made :
  num_boxes_sold * num_doughnuts_per_box + doughnuts_given_away = 300 :=
by
  sorry

end total_doughnuts_made_l77_77200


namespace trapezoid_area_correct_l77_77631

noncomputable def trapezoid_area : ℝ := 
  let base1 : ℝ := 8
  let base2 : ℝ := 4
  let height : ℝ := 2
  (1 / 2) * (base1 + base2) * height

theorem trapezoid_area_correct :
  trapezoid_area = 12.0 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end trapezoid_area_correct_l77_77631


namespace max_product_of_sum_2000_l77_77008

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l77_77008


namespace arrangement_problem_l77_77262
   
   def numberOfArrangements (n : Nat) : Nat :=
     n.factorial

   def exclusiveArrangements (total people : Nat) (positions : Nat) : Nat :=
     (positions.choose 2) * (total - 2).factorial

   theorem arrangement_problem : 
     (numberOfArrangements 5) - (exclusiveArrangements 5 3) = 84 := 
   by
     sorry
   
end arrangement_problem_l77_77262


namespace hindi_books_count_l77_77049

theorem hindi_books_count (H : ℕ) (h1 : 22 = 22) (h2 : Nat.choose 23 H = 1771) : H = 3 :=
sorry

end hindi_books_count_l77_77049


namespace max_product_of_two_integers_sum_2000_l77_77797

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l77_77797


namespace order_of_abc_l77_77088

noncomputable def a := Real.log 1.2
noncomputable def b := (11 / 10) - (10 / 11)
noncomputable def c := 1 / (5 * Real.exp 0.1)

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l77_77088


namespace scale_model_height_l77_77931

theorem scale_model_height :
  let scale_ratio : ℚ := 1 / 25
  let actual_height : ℚ := 151
  let model_height : ℚ := actual_height * scale_ratio
  round model_height = 6 :=
by
  sorry

end scale_model_height_l77_77931


namespace min_buses_needed_l77_77680

theorem min_buses_needed (x y : ℕ) (h1 : 45 * x + 35 * y ≥ 530) (h2 : y ≥ 3) : x + y = 13 :=
by
  sorry

end min_buses_needed_l77_77680


namespace time_to_cross_stationary_train_l77_77633

theorem time_to_cross_stationary_train (t_pole : ℝ) (speed_train : ℝ) (length_stationary_train : ℝ) 
  (t_pole_eq : t_pole = 5) (speed_train_eq : speed_train = 64.8) (length_stationary_train_eq : length_stationary_train = 360) :
  (t_pole * speed_train + length_stationary_train) / speed_train = 10.56 := 
by
  rw [t_pole_eq, speed_train_eq, length_stationary_train_eq]
  norm_num
  sorry

end time_to_cross_stationary_train_l77_77633


namespace task2_X_alone_l77_77762

namespace TaskWork

variables (r_X r_Y r_Z : ℝ)

-- Task 1 conditions
axiom task1_XY : r_X + r_Y = 1 / 4
axiom task1_YZ : r_Y + r_Z = 1 / 6
axiom task1_XZ : r_X + r_Z = 1 / 3

-- Task 2 condition
axiom task2_XYZ : r_X + r_Y + r_Z = 1 / 2

-- Theorem to be proven
theorem task2_X_alone : 1 / r_X = 4.8 :=
sorry

end TaskWork

end task2_X_alone_l77_77762


namespace car_R_average_speed_l77_77651

theorem car_R_average_speed 
  (R P S: ℝ)
  (h1: S = 2 * P)
  (h2: P + 2 = R)
  (h3: P = R + 10)
  (h4: S = R + 20) :
  R = 25 :=
by 
  sorry

end car_R_average_speed_l77_77651


namespace intersection_M_N_l77_77268

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l77_77268


namespace train_pass_man_time_l77_77811

/--
Prove that the train, moving at 120 kmph, passes a man running at 10 kmph in the opposite direction in approximately 13.85 seconds, given the train is 500 meters long.
-/
theorem train_pass_man_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) : 
  length_of_train = 500 →
  speed_of_train = 120 →
  speed_of_man = 10 →
  abs ((500 / ((speed_of_train + speed_of_man) * 1000 / 3600)) - 13.85) < 0.01 :=
by
  intro h1 h2 h3
  -- This is where the proof would go
  sorry

end train_pass_man_time_l77_77811


namespace abs_inequality_solution_l77_77409

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end abs_inequality_solution_l77_77409


namespace sum_of_reciprocals_eq_one_l77_77218

theorem sum_of_reciprocals_eq_one {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y = (x * y) ^ 2) : (1/x) + (1/y) = 1 :=
sorry

end sum_of_reciprocals_eq_one_l77_77218


namespace compare_abc_l77_77865

/-- Define the constants a, b, and c as given in the problem -/
noncomputable def a : ℝ := -5 / 4 * Real.log (4 / 5)
noncomputable def b : ℝ := Real.exp (1 / 4) / 4
noncomputable def c : ℝ := 1 / 3

/-- The theorem to be proved: a < b < c -/
theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l77_77865


namespace david_ate_more_than_emma_l77_77555

-- Definitions and conditions
def contestants : Nat := 8
def pies_david_ate : Nat := 8
def pies_emma_ate : Nat := 2
def pies_by_david (contestants pies_david_ate: Nat) : Prop := pies_david_ate = 8
def pies_by_emma (contestants pies_emma_ate: Nat) : Prop := pies_emma_ate = 2

-- Theorem statement
theorem david_ate_more_than_emma (contestants pies_david_ate pies_emma_ate : Nat) (h_david : pies_by_david contestants pies_david_ate) (h_emma : pies_by_emma contestants pies_emma_ate) : pies_david_ate - pies_emma_ate = 6 :=
by
  sorry

end david_ate_more_than_emma_l77_77555


namespace sum_sequence_formula_l77_77069

theorem sum_sequence_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) ∧ a 1 = 1 →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end sum_sequence_formula_l77_77069


namespace expected_value_correct_l77_77893

-- Define the problem conditions
def num_balls : ℕ := 5

def prob_swapped_twice : ℚ := (2 / 25)
def prob_never_swapped : ℚ := (9 / 25)
def prob_original_position : ℚ := prob_swapped_twice + prob_never_swapped

-- Define the expected value calculation
def expected_num_in_original_position : ℚ :=
  num_balls * prob_original_position

-- Claim: The expected number of balls that occupy their original positions after two successive transpositions is 2.2.
theorem expected_value_correct :
  expected_num_in_original_position = 2.2 :=
sorry

end expected_value_correct_l77_77893


namespace integer_solutions_l77_77176

theorem integer_solutions (m n : ℤ) (h1 : m * (m + n) = n * 12) (h2 : n * (m + n) = m * 3) :
  (m = 4 ∧ n = 2) :=
by sorry

end integer_solutions_l77_77176


namespace geometric_progression_value_l77_77303

variable (a : ℕ → ℕ)
variable (r : ℕ)
variable (h_geo : ∀ n, a (n + 1) = a n * r)

theorem geometric_progression_value (h2 : a 2 = 2) (h6 : a 6 = 162) : a 10 = 13122 :=
by
  sorry

end geometric_progression_value_l77_77303


namespace coopers_daily_pie_count_l77_77907

-- Definitions of conditions
def total_pies_made_per_day (x : ℕ) : ℕ := x
def days := 12
def pies_eaten_by_ashley := 50
def remaining_pies := 34

-- Lean 4 statement of the problem to prove
theorem coopers_daily_pie_count (x : ℕ) : 
  12 * total_pies_made_per_day x - pies_eaten_by_ashley = remaining_pies → 
  x = 7 := 
by
  intro h
  -- Solution steps (not included in the theorem)
  -- Given proof follows from the Lean 4 statement
  sorry

end coopers_daily_pie_count_l77_77907


namespace combined_percentage_tennis_is_31_l77_77630

-- Define the number of students at North High School
def students_north : ℕ := 1800

-- Define the number of students at South Elementary School
def students_south : ℕ := 2200

-- Define the percentage of students who prefer tennis at North High School
def percentage_tennis_north : ℚ := 25/100

-- Define the percentage of students who prefer tennis at South Elementary School
def percentage_tennis_south : ℚ := 35/100

-- Calculate the number of students who prefer tennis at North High School
def tennis_students_north : ℚ := students_north * percentage_tennis_north

-- Calculate the number of students who prefer tennis at South Elementary School
def tennis_students_south : ℚ := students_south * percentage_tennis_south

-- Calculate the total number of students who prefer tennis in both schools
def total_tennis_students : ℚ := tennis_students_north + tennis_students_south

-- Calculate the total number of students in both schools
def total_students : ℚ := students_north + students_south

-- Calculate the combined percentage of students who prefer tennis
def combined_percentage_tennis : ℚ := (total_tennis_students / total_students) * 100

-- Main statement to prove
theorem combined_percentage_tennis_is_31 :
  round combined_percentage_tennis = 31 := by sorry

end combined_percentage_tennis_is_31_l77_77630


namespace vertical_angles_are_congruent_l77_77905

def supplementary_angles (a b : ℝ) : Prop := a + b = 180
def corresponding_angles (l1 l2 t : ℝ) : Prop := l1 = l2
def exterior_angle_greater (ext int1 int2 : ℝ) : Prop := ext = int1 + int2
def vertical_angles_congruent (a b : ℝ) : Prop := a = b

theorem vertical_angles_are_congruent (a b : ℝ) (h : vertical_angles_congruent a b) : a = b := by
  sorry

end vertical_angles_are_congruent_l77_77905


namespace triangle_shape_l77_77851

open Real

noncomputable def triangle (a b c A B C S : ℝ) :=
  ∃ (a b c A B C S : ℝ),
    a = 2 * sqrt 3 ∧
    A = π / 3 ∧
    S = 2 * sqrt 3 ∧
    (S = (1 / 2) * b * c * sin A) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) ∧
    (b = 2 ∧ c = 4 ∨ b = 4 ∧ c = 2)

theorem triangle_shape (A B C : ℝ) (h : sin (C - B) = sin (2 * B) - sin A):
    (B = π / 2 ∨ C = B) :=
sorry

end triangle_shape_l77_77851


namespace time_addition_sum_l77_77285

/-- Given the start time of 3:15:20 PM and adding a duration of 
    305 hours, 45 minutes, and 56 seconds, the resultant hour, 
    minute, and second values sum to 26. -/
theorem time_addition_sum : 
  let current_hour := 15
  let current_minute := 15
  let current_second := 20
  let added_hours := 305
  let added_minutes := 45
  let added_seconds := 56
  let final_hour := ((current_hour + (added_hours % 12) + ((current_minute + added_minutes) / 60) + ((current_second + added_seconds) / 3600)) % 12)
  let final_minute := ((current_minute + added_minutes + ((current_second + added_seconds) / 60)) % 60)
  let final_second := ((current_second + added_seconds) % 60)
  final_hour + final_minute + final_second = 26 := 
  sorry

end time_addition_sum_l77_77285


namespace original_intensity_45_percent_l77_77776

variable (I : ℝ) -- Intensity of the original red paint in percentage.

-- Conditions
variable (h1 : 25 * 0.25 + 0.75 * I = 40) -- Given conditions about the intensities and the new solution.
variable (h2 : ∀ I : ℝ, 0.75 * I + 25 * 0.25 = 40) -- Rewriting the given condition to look specifically for I.

theorem original_intensity_45_percent (I : ℝ) (h1 : 25 * 0.25 + 0.75 * I = 40) : I = 45 := by
  -- We only need the statement. Proof is not required.
  sorry

end original_intensity_45_percent_l77_77776


namespace correct_average_l77_77205

theorem correct_average (n : Nat) (incorrect_avg correct_mark incorrect_mark : ℝ) 
  (h1 : n = 30) (h2 : incorrect_avg = 60) (h3 : correct_mark = 15) (h4 : incorrect_mark = 90) :
  (incorrect_avg * n - incorrect_mark + correct_mark) / n = 57.5 :=
by
  sorry

end correct_average_l77_77205


namespace unique_integers_exist_l77_77123

theorem unique_integers_exist (p : ℕ) (hp : p > 1) : 
  ∃ (a b c : ℤ), b^2 - 4*a*c = 1 - 4*p ∧ 0 < a ∧ a ≤ c ∧ -a ≤ b ∧ b < a :=
sorry

end unique_integers_exist_l77_77123


namespace unique_n_degree_polynomial_exists_l77_77595

theorem unique_n_degree_polynomial_exists (n : ℕ) (h : n > 0) :
  ∃! (f : Polynomial ℝ), Polynomial.degree f = n ∧
    f.eval 0 = 1 ∧
    ∀ x : ℝ, (x + 1) * (f.eval x)^2 - 1 = -((x + 1) * (f.eval (-x))^2 - 1) := 
sorry

end unique_n_degree_polynomial_exists_l77_77595


namespace victor_earnings_l77_77444

variable (wage hours_mon hours_tue : ℕ)

def hourly_wage : ℕ := 6
def hours_worked_monday : ℕ := 5
def hours_worked_tuesday : ℕ := 5

theorem victor_earnings :
  (hours_worked_monday + hours_worked_tuesday) * hourly_wage = 60 :=
by
  sorry

end victor_earnings_l77_77444


namespace cost_price_computer_table_l77_77413

theorem cost_price_computer_table (C : ℝ) (S : ℝ) (H1 : S = C + 0.60 * C) (H2 : S = 2000) : C = 1250 :=
by
  -- Proof goes here
  sorry

end cost_price_computer_table_l77_77413


namespace factor_polynomial_l77_77567

theorem factor_polynomial :
  ∀ u : ℝ, (u^4 - 81 * u^2 + 144) = (u^2 - 72) * (u - 3) * (u + 3) :=
by
  intro u
  -- Establish the polynomial and its factorization in Lean
  have h : u^4 - 81 * u^2 + 144 = (u^2 - 72) * (u - 3) * (u + 3) := sorry
  exact h

end factor_polynomial_l77_77567


namespace range_of_a_l77_77179

theorem range_of_a (a : ℝ) (h1 : a > 0)
  (h2 : ∃ x : ℝ, abs (Real.sin x) > a)
  (h3 : ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ico (Real.sqrt 2 / 2) 1 :=
sorry

end range_of_a_l77_77179


namespace cody_games_remaining_l77_77239

-- Definitions based on the conditions
def initial_games : ℕ := 9
def games_given_away : ℕ := 4

-- Theorem statement
theorem cody_games_remaining : initial_games - games_given_away = 5 :=
by sorry

end cody_games_remaining_l77_77239


namespace greg_rolls_probability_l77_77469

noncomputable def probability_of_more_ones_than_twos_and_threes_combined : ℚ :=
  (3046.5 : ℚ) / 7776

theorem greg_rolls_probability :
  probability_of_more_ones_than_twos_and_threes_combined = (3046.5 : ℚ) / 7776 := 
by 
  sorry

end greg_rolls_probability_l77_77469


namespace arithmetic_sequence_general_term_sum_sequence_proof_l77_77725

theorem arithmetic_sequence_general_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ) (a1 : ℝ)
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d > 0)
  (h3 : a1 * (a1 + 3 * d) = 22)
  (h4 : 4 * a1 + 6 * d = 26) :
  ∀ n, a_n n = 3 * n - 1 := sorry

theorem sum_sequence_proof (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n = 3 * n - 1)
  (h2 : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1)))
  (h3 : ∀ n, T_n n = (Finset.range n).sum b_n)
  (n : ℕ) :
  T_n n < 1 / 6 := sorry

end arithmetic_sequence_general_term_sum_sequence_proof_l77_77725


namespace function_identity_l77_77296

theorem function_identity (f : ℕ → ℕ) (h₁ : ∀ n, 0 < f n)
  (h₂ : ∀ n, f (n + 1) > f (f n)) :
∀ n, f n = n :=
sorry

end function_identity_l77_77296


namespace arithmetic_sequence_common_difference_l77_77767

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 = 2)
  (h3 : ∃ r, a 2 = r * a 1 ∧ a 5 = r * a 2) :
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l77_77767


namespace Lizzy_total_after_loan_returns_l77_77673

theorem Lizzy_total_after_loan_returns : 
  let initial_amount := 50
  let alice_loan := 25 
  let alice_interest_rate := 0.15
  let bob_loan := 20
  let bob_interest_rate := 0.20
  let alice_interest := alice_loan * alice_interest_rate
  let bob_interest := bob_loan * bob_interest_rate
  let total_alice := alice_loan + alice_interest
  let total_bob := bob_loan + bob_interest
  let total_amount := total_alice + total_bob
  total_amount = 52.75 :=
by
  sorry

end Lizzy_total_after_loan_returns_l77_77673


namespace increase_in_average_weight_l77_77078

theorem increase_in_average_weight :
  let initial_group_size := 6
  let initial_weight := 65
  let new_weight := 74
  let initial_avg_weight := A
  (new_weight - initial_weight) / initial_group_size = 1.5 := by
    sorry

end increase_in_average_weight_l77_77078


namespace find_n_l77_77491

theorem find_n (n : ℕ) (h1 : ∃ k : ℕ, 12 - n = k * k) : n = 11 := 
by sorry

end find_n_l77_77491


namespace cartons_per_stack_l77_77203

-- Declare the variables and conditions
def total_cartons := 799
def stacks := 133

-- State the theorem
theorem cartons_per_stack : (total_cartons / stacks) = 6 := by
  sorry

end cartons_per_stack_l77_77203


namespace nonneg_real_sum_inequality_l77_77594

theorem nonneg_real_sum_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end nonneg_real_sum_inequality_l77_77594


namespace johns_drawings_l77_77193

theorem johns_drawings (total_pictures : ℕ) (back_pictures : ℕ) 
  (h1 : total_pictures = 15) (h2 : back_pictures = 9) : total_pictures - back_pictures = 6 := by
  -- proof goes here
  sorry

end johns_drawings_l77_77193


namespace tom_drives_distance_before_karen_wins_l77_77043

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end tom_drives_distance_before_karen_wins_l77_77043


namespace truncated_pyramid_ratio_l77_77461

noncomputable def volume_prism (L1 H : ℝ) : ℝ := L1^2 * H
noncomputable def volume_truncated_pyramid (L1 L2 H : ℝ) : ℝ := 
  (H / 3) * (L1^2 + L1 * L2 + L2^2)

theorem truncated_pyramid_ratio (L1 L2 H : ℝ) 
  (h_vol : volume_truncated_pyramid L1 L2 H = (2/3) * volume_prism L1 H) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end truncated_pyramid_ratio_l77_77461


namespace people_who_like_both_l77_77669

-- Conditions
variables (total : ℕ) (a : ℕ) (b : ℕ) (none : ℕ)
-- Express the problem
theorem people_who_like_both : total = 50 → a = 23 → b = 20 → none = 14 → (a + b - (total - none) = 7) :=
by
  intros
  sorry

end people_who_like_both_l77_77669


namespace part1_part2_l77_77646

-- Define the system of equations
def system_eq (x y k : ℝ) : Prop := 
  3 * x + y = k + 1 ∧ x + 3 * y = 3

-- Part (1): x and y are opposite in sign implies k = -4
theorem part1 (x y k : ℝ) (h_eq : system_eq x y k) (h_sign : x * y < 0) : k = -4 := by
  sorry

-- Part (2): range of values for k given extra inequalities
theorem part2 (x y k : ℝ) (h_eq : system_eq x y k) 
  (h_ineq1 : x + y < 3) (h_ineq2 : x - y > 1) : 4 < k ∧ k < 8 := by
  sorry

end part1_part2_l77_77646


namespace coin_change_count_ways_l77_77358

theorem coin_change_count_ways :
  ∃ n : ℕ, (∀ q h : ℕ, (25 * q + 50 * h = 1500) ∧ q > 0 ∧ h > 0 → (1 ≤ h ∧ h < 30)) ∧ n = 29 :=
  sorry

end coin_change_count_ways_l77_77358


namespace multiplication_verification_l77_77667

theorem multiplication_verification (x : ℕ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end multiplication_verification_l77_77667


namespace remainder_div_polynomial_l77_77859

theorem remainder_div_polynomial :
  ∀ (x : ℝ), 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    R x = (3^101 - 2^101) * x + (2^101 - 2 * 3^101) ∧
    x^101 = (x^2 - 5 * x + 6) * Q x + R x :=
by
  sorry

end remainder_div_polynomial_l77_77859


namespace lcm_of_132_and_315_l77_77650

def n1 : ℕ := 132
def n2 : ℕ := 315

theorem lcm_of_132_and_315 :
  (Nat.lcm n1 n2) = 13860 :=
by
  -- Proof goes here
  sorry

end lcm_of_132_and_315_l77_77650


namespace part1_part2_l77_77286

noncomputable def f : ℝ → ℝ 
| x => if 0 ≤ x then 2^x - 1 else -2^(-x) + 1

theorem part1 (x : ℝ) (h : x < 0) : f x = -2^(-x) + 1 := sorry

theorem part2 (a : ℝ) : f a ≤ 3 ↔ a ≤ 2 := sorry

end part1_part2_l77_77286


namespace ratio_of_still_lifes_to_portraits_l77_77208

noncomputable def total_paintings : ℕ := 80
noncomputable def portraits : ℕ := 16
noncomputable def still_lifes : ℕ := total_paintings - portraits
axiom still_lifes_is_multiple_of_portraits : ∃ k : ℕ, still_lifes = k * portraits

theorem ratio_of_still_lifes_to_portraits : still_lifes / portraits = 4 := by
  -- proof would go here
  sorry

end ratio_of_still_lifes_to_portraits_l77_77208


namespace investor_more_money_in_A_l77_77440

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end investor_more_money_in_A_l77_77440


namespace new_foreign_students_l77_77128

theorem new_foreign_students 
  (total_students : ℕ)
  (percent_foreign : ℕ)
  (foreign_students_next_sem : ℕ)
  (current_foreign_students : ℕ := total_students * percent_foreign / 100) : 
  total_students = 1800 → 
  percent_foreign = 30 → 
  foreign_students_next_sem = 740 → 
  foreign_students_next_sem - current_foreign_students = 200 :=
by
  intros
  sorry

end new_foreign_students_l77_77128


namespace rectangle_is_possible_l77_77480

def possibleToFormRectangle (stick_lengths : List ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (a + b) * 2 = List.sum stick_lengths

noncomputable def sticks : List ℕ := List.range' 1 99

theorem rectangle_is_possible : possibleToFormRectangle sticks :=
sorry

end rectangle_is_possible_l77_77480


namespace cyclic_quadrilateral_JMIT_l77_77861

theorem cyclic_quadrilateral_JMIT
  (a b c : ℂ)
  (I J M N T : ℂ)
  (hI : I = -(a*b + b*c + c*a))
  (hJ : J = a*b - b*c + c*a)
  (hM : M = (b^2 + c^2) / 2)
  (hN : N = b*c)
  (hT : T = 2*a^2 - b*c) :
  ∃ (k : ℝ), k = ((M - I) * (T - J)) / ((J - I) * (T - M)) :=
by
  sorry

end cyclic_quadrilateral_JMIT_l77_77861


namespace complement_union_l77_77346

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l77_77346


namespace positive_difference_of_squares_l77_77416

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l77_77416


namespace mowing_lawn_time_l77_77399

def maryRate := 1 / 3
def tomRate := 1 / 4
def combinedRate := 7 / 12
def timeMaryAlone := 1
def lawnLeft := 1 - (timeMaryAlone * maryRate)

theorem mowing_lawn_time:
  (7 / 12) * (8 / 7) = (2 / 3) :=
by
  sorry

end mowing_lawn_time_l77_77399


namespace distance_between_points_on_line_l77_77961

theorem distance_between_points_on_line (a b c d m k : ℝ) 
  (hab : b = m * a + k) (hcd : d = m * c + k) :
  dist (a, b) (c, d) = |a - c| * Real.sqrt (1 + m^2) :=
by
  sorry

end distance_between_points_on_line_l77_77961


namespace find_min_k_l77_77660

theorem find_min_k (k : ℕ) 
  (h1 : k > 0) 
  (h2 : ∀ (A : Finset ℕ), A ⊆ (Finset.range 26).erase 0 → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (2 / 3 : ℝ) ≤ x / y ∧ x / y ≤ (3 / 2 : ℝ)) : 
  k = 7 :=
by {
  sorry
}

end find_min_k_l77_77660


namespace james_twitch_income_l77_77661

theorem james_twitch_income :
  let tier1_base := 120
  let tier2_base := 50
  let tier3_base := 30
  let tier1_gifted := 10
  let tier2_gifted := 25
  let tier3_gifted := 15
  let tier1_new := tier1_base + tier1_gifted
  let tier2_new := tier2_base + tier2_gifted
  let tier3_new := tier3_base + tier3_gifted
  let tier1_income := tier1_new * 4.99
  let tier2_income := tier2_new * 9.99
  let tier3_income := tier3_new * 24.99
  let total_income := tier1_income + tier2_income + tier3_income
  total_income = 2522.50 :=
by
  sorry

end james_twitch_income_l77_77661


namespace find_a6_a7_l77_77442

variable {a : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_given : a 2 + a 3 + a 10 + a 11 = 48

theorem find_a6_a7 (arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d) (h : a 2 + a 3 + a 10 + a 11 = 48) :
  a 6 + a 7 = 24 :=
by
  sorry

end find_a6_a7_l77_77442


namespace division_multiplication_example_l77_77557

theorem division_multiplication_example : 120 / 4 / 2 * 3 = 45 := by
  sorry

end division_multiplication_example_l77_77557


namespace max_period_initial_phase_function_l77_77234

theorem max_period_initial_phase_function 
  (A ω ϕ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : A = 1/2) 
  (h2 : ω = 6) 
  (h3 : ϕ = π/4) 
  (h4 : ∀ x, f x = A * Real.sin (ω * x + ϕ)) : 
  ∀ x, f x = (1/2) * Real.sin (6 * x + (π/4)) :=
by
  sorry

end max_period_initial_phase_function_l77_77234


namespace grid_coloring_count_l77_77015

/-- Let n be a positive integer with n ≥ 2. Each of the 2n vertices in a 2 × n grid need to be 
colored red (R), yellow (Y), or blue (B). The three vertices at the endpoints are already colored 
as shown in the problem description. For the remaining 2n-3 vertices, each vertex must be colored 
exactly one color, and adjacent vertices must be colored differently. We aim to show that the 
number of distinct ways to color the vertices is 3^(n-1). -/
theorem grid_coloring_count (n : ℕ) (hn : n ≥ 2) : 
  ∃ a_n b_n c_n : ℕ, 
    (a_n + b_n + c_n = 3^(n-1)) ∧ 
    (a_n = b_n) ∧ 
    (a_n = 2 * b_n + c_n) := 
by 
  sorry

end grid_coloring_count_l77_77015


namespace difference_of_squares_l77_77518

theorem difference_of_squares (a b c : ℤ) (h₁ : a < b) (h₂ : b < c) (h₃ : a % 2 = 0) (h₄ : b % 2 = 0) (h₅ : c % 2 = 0) (h₆ : a + b + c = 1992) :
  c^2 - a^2 = 5312 :=
by
  sorry

end difference_of_squares_l77_77518


namespace hawks_first_half_score_l77_77690

variable (H1 H2 E : ℕ)

theorem hawks_first_half_score (H1 H2 E : ℕ) 
  (h1 : H1 + H2 + E = 120)
  (h2 : E = H1 + H2 + 16)
  (h3 : H2 = H1 + 8) :
  H1 = 22 :=
by
  sorry

end hawks_first_half_score_l77_77690


namespace initial_bananas_per_child_l77_77643

theorem initial_bananas_per_child (B x : ℕ) (total_children : ℕ := 780) (absent_children : ℕ := 390) :
  390 * (x + 2) = total_children * x → x = 2 :=
by
  intros h
  sorry

end initial_bananas_per_child_l77_77643


namespace bob_work_days_per_week_l77_77169

theorem bob_work_days_per_week (daily_hours : ℕ) (monthly_hours : ℕ) (average_days_per_month : ℕ) (days_per_week : ℕ)
  (h1 : daily_hours = 10)
  (h2 : monthly_hours = 200)
  (h3 : average_days_per_month = 30)
  (h4 : days_per_week = 7) :
  (monthly_hours / daily_hours) / (average_days_per_month / days_per_week) = 5 := by
  -- Now we will skip the proof itself. The focus here is on the structure.
  sorry

end bob_work_days_per_week_l77_77169


namespace factorization1_factorization2_l77_77959

-- Definitions for the first problem
def expr1 (x : ℝ) := 3 * x^2 - 12
def factorized_form1 (x : ℝ) := 3 * (x + 2) * (x - 2)

-- Theorem for the first problem
theorem factorization1 (x : ℝ) : expr1 x = factorized_form1 x :=
  sorry

-- Definitions for the second problem
def expr2 (a x y : ℝ) := a * x^2 - 4 * a * x * y + 4 * a * y^2
def factorized_form2 (a x y : ℝ) := a * (x - 2 * y) * (x - 2 * y)

-- Theorem for the second problem
theorem factorization2 (a x y : ℝ) : expr2 a x y = factorized_form2 a x y :=
  sorry

end factorization1_factorization2_l77_77959


namespace find_b_value_l77_77265

-- Define the conditions: line equation and given range for b
def line_eq (x : ℝ) (b : ℝ) : ℝ := b - x

-- Define the points P, Q, S
def P (b : ℝ) : ℝ × ℝ := ⟨0, b⟩
def Q (b : ℝ) : ℝ × ℝ := ⟨b, 0⟩
def S (b : ℝ) : ℝ × ℝ := ⟨6, b - 6⟩

-- Define the area ratio condition
def area_ratio_condition (b : ℝ) : Prop :=
  (0 < b ∧ b < 6) ∧ ((6 - b) / b) ^ 2 = 4 / 25

-- Define the main theorem to prove
theorem find_b_value (b : ℝ) : area_ratio_condition b → b = 4.3 := by
  sorry

end find_b_value_l77_77265


namespace speed_in_still_water_l77_77682

-- Define the conditions: upstream and downstream speeds.
def upstream_speed : ℝ := 10
def downstream_speed : ℝ := 20

-- Define the still water speed theorem.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 15 := by
  sorry

end speed_in_still_water_l77_77682


namespace g_inv_zero_solution_l77_77798

noncomputable def g (a b x : ℝ) : ℝ := 1 / (2 * a * x + b)

theorem g_inv_zero_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  g a b (g a b 0) = 0 ↔ g a b 0 = 1 / b :=
by
  sorry

end g_inv_zero_solution_l77_77798


namespace ratio_of_cakes_l77_77251

/-- Define the usual number of cheesecakes, muffins, and red velvet cakes baked in a week -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet_cakes : ℕ := 8

/-- Define the total number of cakes usually baked in a week -/
def usual_cakes : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet_cakes

/-- Assume Carter baked this week a multiple of usual cakes, denoted as x -/
def multiple (x : ℕ) : Prop := usual_cakes * x = usual_cakes + 38

/-- Assume he baked usual_cakes + 38 equals 57 cakes -/
def total_cakes_this_week : ℕ := 57

/-- The theorem stating the problem: proving the ratio is 3:1 -/
theorem ratio_of_cakes (x : ℕ) (hx : multiple x) : 
  (total_cakes_this_week : ℚ) / (usual_cakes : ℚ) = (3 : ℚ) :=
by
  sorry

end ratio_of_cakes_l77_77251


namespace smallest_number_of_brownies_l77_77020

noncomputable def total_brownies (m n : ℕ) : ℕ := m * n
def perimeter_brownies (m n : ℕ) : ℕ := 2 * m + 2 * n - 4
def interior_brownies (m n : ℕ) : ℕ := (m - 2) * (n - 2)

theorem smallest_number_of_brownies : 
  ∃ (m n : ℕ), 2 * interior_brownies m n = perimeter_brownies m n ∧ total_brownies m n = 36 :=
by
  sorry

end smallest_number_of_brownies_l77_77020


namespace unique_solution_triplet_l77_77157

theorem unique_solution_triplet :
  ∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x^y + y^x = z^y ∧ x^y + 2012 = y^(z+1)) ∧ (x = 6 ∧ y = 2 ∧ z = 10) := 
by {
  sorry
}

end unique_solution_triplet_l77_77157


namespace smallest_solution_l77_77957

theorem smallest_solution (x : ℝ) : (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) → x = 4 - (Real.sqrt 15) / 3 :=
by
  sorry

end smallest_solution_l77_77957


namespace ernie_circles_l77_77449

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l77_77449


namespace cow_calf_ratio_l77_77375

theorem cow_calf_ratio (cost_cow cost_calf : ℕ) (h_cow : cost_cow = 880) (h_calf : cost_calf = 110) :
  cost_cow / cost_calf = 8 :=
by {
  sorry
}

end cow_calf_ratio_l77_77375


namespace multiples_of_7_are_128_l77_77972

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l77_77972


namespace lg_45_eq_l77_77685

variable (m n : ℝ)
axiom lg_2 : Real.log 2 = m
axiom lg_3 : Real.log 3 = n

theorem lg_45_eq : Real.log 45 = 1 - m + 2 * n := by
  -- proof to be filled in
  sorry

end lg_45_eq_l77_77685


namespace april_total_earned_l77_77874

variable (r_price t_price d_price : ℕ)
variable (r_sold t_sold d_sold : ℕ)
variable (r_total t_total d_total : ℕ)

-- Define prices
def rose_price : ℕ := 4
def tulip_price : ℕ := 3
def daisy_price : ℕ := 2

-- Define quantities sold
def roses_sold : ℕ := 9
def tulips_sold : ℕ := 6
def daisies_sold : ℕ := 12

-- Define total money earned for each type of flower
def rose_total := roses_sold * rose_price
def tulip_total := tulips_sold * tulip_price
def daisy_total := daisies_sold * daisy_price

-- Define total money earned
def total_earned := rose_total + tulip_total + daisy_total

-- Statement to prove
theorem april_total_earned : total_earned = 78 :=
by sorry

end april_total_earned_l77_77874


namespace smallest_number_divisible_l77_77573

theorem smallest_number_divisible (x : ℕ) : 
  (∃ x, x + 7 % 8 = 0 ∧ x + 7 % 11 = 0 ∧ x + 7 % 24 = 0) ∧
  (∀ y, (y + 7 % 8 = 0 ∧ y + 7 % 11 = 0 ∧ y + 7 % 24 = 0) → 257 ≤ y) :=
by { sorry }

end smallest_number_divisible_l77_77573


namespace range_of_a_l77_77963

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 + (a+1)*x + a < 0) → a ∈ Set.Iio (-2 / 3) := 
sorry

end range_of_a_l77_77963


namespace ratio_of_areas_of_concentric_circles_l77_77114

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l77_77114


namespace gcd_36_54_l77_77652

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l77_77652


namespace solution_set_inequality_l77_77842

theorem solution_set_inequality (x : ℝ) : (0 < x ∧ x < 1) ↔ (1 / (x - 1) < -1) :=
by
  sorry

end solution_set_inequality_l77_77842


namespace hollow_cylinder_surface_area_l77_77019

theorem hollow_cylinder_surface_area (h : ℝ) (r_outer r_inner : ℝ) (h_eq : h = 12) (r_outer_eq : r_outer = 5) (r_inner_eq : r_inner = 2) :
  (2 * π * ((r_outer ^ 2 - r_inner ^ 2)) + 2 * π * r_outer * h + 2 * π * r_inner * h) = 210 * π :=
by
  rw [h_eq, r_outer_eq, r_inner_eq]
  sorry

end hollow_cylinder_surface_area_l77_77019


namespace total_students_l77_77499

-- Condition 1: 20% of students are below 8 years of age.
-- Condition 2: The number of students of 8 years of age is 72.
-- Condition 3: The number of students above 8 years of age is 2/3 of the number of students of 8 years of age.

variable {T : ℝ} -- Total number of students

axiom cond1 : 0.20 * T = (T - (72 + (2 / 3) * 72))
axiom cond2 : 72 = 72
axiom cond3 : (T - 72 - (2 / 3) * 72) = 0

theorem total_students : T = 150 := by
  -- Proof goes here
  sorry

end total_students_l77_77499


namespace sqrt_recursive_value_l77_77912

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l77_77912


namespace major_axis_length_l77_77788

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.8 * minor_axis) :
  major_axis = 7.2 :=
sorry

end major_axis_length_l77_77788


namespace sam_watermelons_second_batch_l77_77606

theorem sam_watermelons_second_batch
  (initial_watermelons : ℕ)
  (total_watermelons : ℕ)
  (second_batch_watermelons : ℕ) :
  initial_watermelons = 4 →
  total_watermelons = 7 →
  second_batch_watermelons = total_watermelons - initial_watermelons →
  second_batch_watermelons = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_watermelons_second_batch_l77_77606


namespace evaluate_expressions_l77_77197

theorem evaluate_expressions : (∀ (a b c d : ℤ), a = -(-3) → b = -(|-3|) → c = -(-(3^2)) → d = ((-3)^2) → b < 0) :=
by
  sorry

end evaluate_expressions_l77_77197


namespace maximum_value_is_l77_77745

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l77_77745


namespace sum_of_eight_digits_l77_77853

open Nat

theorem sum_of_eight_digits {a b c d e f g h : ℕ} 
  (h_distinct : ∀ i j, i ∈ [a, b, c, d, e, f, g, h] → j ∈ [a, b, c, d, e, f, g, h] → i ≠ j → i ≠ j)
  (h_vertical_sum : a + b + c + d + e = 25)
  (h_horizontal_sum : f + g + h + b = 15) 
  (h_digits_set : ∀ x ∈ [a, b, c, d, e, f, g, h], x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) : 
  a + b + c + d + e + f + g + h - b = 39 := 
sorry

end sum_of_eight_digits_l77_77853


namespace prove_mutually_exclusive_l77_77597

def bag : List String := ["red", "red", "red", "black", "black"]

def at_least_one_black (drawn : List String) : Prop :=
  "black" ∈ drawn

def all_red (drawn : List String) : Prop :=
  ∀ b ∈ drawn, b = "red"

def events_mutually_exclusive : Prop :=
  ∀ drawn, at_least_one_black drawn → ¬all_red drawn

theorem prove_mutually_exclusive :
  events_mutually_exclusive
:= by
  sorry

end prove_mutually_exclusive_l77_77597


namespace marked_price_l77_77549

theorem marked_price (x : ℝ) (payment : ℝ) (discount : ℝ) (hx : (payment = 90) ∧ ((x ≤ 100 ∧ discount = 0.1) ∨ (x > 100 ∧ discount = 0.2))) :
  (x = 100 ∨ x = 112.5) := by
  sorry

end marked_price_l77_77549


namespace correct_operation_l77_77703

theorem correct_operation :
  ¬ ( (-3 : ℤ) * x ^ 2 * y ) ^ 3 = -9 * (x ^ 6) * y ^ 3 ∧
  ¬ (a + b) * (a + b) = (a ^ 2 + b ^ 2) ∧
  (4 * x ^ 3 * y ^ 2) * (x ^ 2 * y ^ 3) = (4 * x ^ 5 * y ^ 5) ∧
  ¬ ((-a) + b) * (a - b) = (a ^ 2 - b ^ 2) :=
by
  sorry

end correct_operation_l77_77703


namespace tournament_teams_l77_77710

theorem tournament_teams (n : ℕ) (H : 240 = 2 * n * (n - 1)) : n = 12 := 
by sorry

end tournament_teams_l77_77710


namespace distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l77_77799

-- Definitions for each day's recorded distance deviation
def day_1_distance := -8
def day_2_distance := -11
def day_3_distance := -14
def day_4_distance := 0
def day_5_distance := 8
def day_6_distance := 41
def day_7_distance := -16

-- Parameters and conditions
def actual_distance (recorded: Int) : Int := 50 + recorded

noncomputable def distance_3rd_day : Int := actual_distance day_3_distance
noncomputable def longest_distance : Int :=
    max (max (max (day_1_distance) (day_2_distance)) (max (day_3_distance) (day_4_distance)))
        (max (max (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def shortest_distance : Int :=
    min (min (min (day_1_distance) (day_2_distance)) (min (day_3_distance) (day_4_distance)))
        (min (min (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def average_distance : Int :=
    50 + (day_1_distance + day_2_distance + day_3_distance + day_4_distance +
          day_5_distance + day_6_distance + day_7_distance) / 7

-- Theorems to prove each part of the problem
theorem distance_on_third_day_is_36 : distance_3rd_day = 36 := by
  sorry

theorem difference_between_longest_and_shortest_is_57 : 
  (actual_distance longest_distance - actual_distance shortest_distance) = 57 := by
  sorry

theorem average_daily_distance_is_50 : average_distance = 50 := by
  sorry

end distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l77_77799


namespace intersection_point_l77_77695

def L1 (x y : ℚ) : Prop := y = -3 * x
def L2 (x y : ℚ) : Prop := y + 4 = 9 * x

theorem intersection_point : ∃ x y : ℚ, L1 x y ∧ L2 x y ∧ x = 1/3 ∧ y = -1 := sorry

end intersection_point_l77_77695


namespace min_value_of_a_l77_77155

theorem min_value_of_a (x y : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) 
  (h : ∀ x y, 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  4 ≤ a :=
sorry

end min_value_of_a_l77_77155


namespace max_m_eq_half_l77_77381

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + m * Real.log x

theorem max_m_eq_half :
  ∃ m : ℝ, (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ 2) → (1 ≤ x2 ∧ x2 ≤ 2) → 
  x1 < x2 → |f x1 m - f x2 m| < x2^2 - x1^2)) ∧ m = 1/2 :=
sorry

end max_m_eq_half_l77_77381


namespace find_smallest_d_l77_77126

-- Given conditions: The known digits sum to 26
def sum_known_digits : ℕ := 5 + 2 + 4 + 7 + 8 

-- Define the smallest digit d such that 52,d47,8 is divisible by 9
def smallest_d (d : ℕ) (sum_digits_with_d : ℕ) : Prop :=
  sum_digits_with_d = sum_known_digits + d ∧ (sum_digits_with_d % 9 = 0)

theorem find_smallest_d : ∃ d : ℕ, smallest_d d 27 :=
sorry

end find_smallest_d_l77_77126


namespace max_value_expression_l77_77938

noncomputable def expression (x : ℝ) : ℝ := 5^x - 25^x

theorem max_value_expression : 
  (∀ x : ℝ, expression x ≤ 1/4) ∧ (∃ x : ℝ, expression x = 1/4) := 
by 
  sorry

end max_value_expression_l77_77938


namespace A_B_work_together_finish_l77_77096
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end A_B_work_together_finish_l77_77096


namespace gcd_of_cubic_sum_and_linear_is_one_l77_77694

theorem gcd_of_cubic_sum_and_linear_is_one (n : ℕ) (h : n > 27) : Nat.gcd (n^3 + 8) (n + 3) = 1 :=
sorry

end gcd_of_cubic_sum_and_linear_is_one_l77_77694


namespace average_people_per_row_l77_77921

theorem average_people_per_row (boys girls rows : ℕ) (h_boys : boys = 24) (h_girls : girls = 24) (h_rows : rows = 6) : 
  (boys + girls) / rows = 8 :=
by
  sorry

end average_people_per_row_l77_77921


namespace determine_S5_l77_77556

noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1 / x^m

theorem determine_S5 (x : ℝ) (h : x + 1 / x = 3) : S x 5 = 123 :=
by
  sorry

end determine_S5_l77_77556


namespace minimum_area_of_rectangle_l77_77184

theorem minimum_area_of_rectangle (x y : ℝ) (h1 : x = 3) (h2 : y = 4) : 
  (min_area : ℝ) = (2.3 * 3.3) :=
by
  have length_min := x - 0.7
  have width_min := y - 0.7
  have min_area := length_min * width_min
  sorry

end minimum_area_of_rectangle_l77_77184


namespace pump_fill_time_without_leak_l77_77696

def time_with_leak := 10
def leak_empty_time := 10

def combined_rate_with_leak := 1 / time_with_leak
def leak_rate := 1 / leak_empty_time

def T : ℝ := 5

theorem pump_fill_time_without_leak
  (time_with_leak : ℝ)
  (leak_empty_time : ℝ)
  (combined_rate_with_leak : ℝ)
  (leak_rate : ℝ)
  (T : ℝ)
  (h1 : combined_rate_with_leak = 1 / time_with_leak)
  (h2 : leak_rate = 1 / leak_empty_time)
  (h_combined : 1 / T - leak_rate = combined_rate_with_leak) :
  T = 5 :=
by {
  sorry
}

end pump_fill_time_without_leak_l77_77696


namespace symmetric_points_on_parabola_l77_77992

theorem symmetric_points_on_parabola (x1 x2 y1 y2 m : ℝ)
  (h1: y1 = 2 * x1 ^ 2)
  (h2: y2 = 2 * x2 ^ 2)
  (h3: x1 * x2 = -1 / 2)
  (h4: y2 - y1 = 2 * (x2 ^ 2 - x1 ^ 2))
  (h5: (x1 + x2) / 2 = -1 / 4)
  (h6: (y1 + y2) / 2 = (x1 + x2) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end symmetric_points_on_parabola_l77_77992


namespace cost_price_to_marked_price_l77_77217

theorem cost_price_to_marked_price (MP CP SP : ℝ)
  (h1 : SP = MP * 0.87)
  (h2 : SP = CP * 1.359375) :
  (CP / MP) * 100 = 64 := by
  sorry

end cost_price_to_marked_price_l77_77217


namespace skittles_left_l77_77932

theorem skittles_left (initial_skittles : ℕ) (skittles_given : ℕ) (final_skittles : ℕ) :
  initial_skittles = 50 → skittles_given = 7 → final_skittles = initial_skittles - skittles_given → final_skittles = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end skittles_left_l77_77932


namespace ball_bounces_height_l77_77073

theorem ball_bounces_height : ∃ k : ℕ, ∀ n ≥ k, 800 * (2 / 3: ℝ) ^ n < 10 :=
by
  sorry

end ball_bounces_height_l77_77073


namespace number_of_new_players_l77_77940

-- Definitions based on conditions
def total_groups : Nat := 2
def players_per_group : Nat := 5
def returning_players : Nat := 6

-- Convert conditions to definition
def total_players : Nat := total_groups * players_per_group

-- Define what we want to prove
def new_players : Nat := total_players - returning_players

-- The proof problem statement
theorem number_of_new_players :
  new_players = 4 :=
by
  sorry

end number_of_new_players_l77_77940


namespace train_length_l77_77704

variable (L : ℝ) -- The length of the train

def length_of_platform : ℝ := 250 -- The length of the platform

def time_to_cross_platform : ℝ := 33 -- Time to cross the platform in seconds

def time_to_cross_pole : ℝ := 18 -- Time to cross the signal pole in seconds

-- The speed of the train is constant whether it crosses the platform or the signal pole.
-- Therefore, we equate the expressions for speed and solve for L.
theorem train_length (h1 : time_to_cross_platform * L = time_to_cross_pole * (L + length_of_platform)) :
  L = 300 :=
by
  -- Proof will be here
  sorry

end train_length_l77_77704


namespace reciprocal_of_sum_l77_77933

theorem reciprocal_of_sum : (1 / (1 / 3 + 1 / 4)) = 12 / 7 := 
by sorry

end reciprocal_of_sum_l77_77933


namespace students_not_solving_any_problem_l77_77561

variable (A_0 A_1 A_2 A_3 A_4 A_5 A_6 : ℕ)

-- Given conditions
def number_of_students := 2006
def condition_1 := A_1 = 4 * A_2
def condition_2 := A_2 = 4 * A_3
def condition_3 := A_3 = 4 * A_4
def condition_4 := A_4 = 4 * A_5
def condition_5 := A_5 = 4 * A_6
def total_students := A_0 + A_1 = 2006

-- The final statement to be proven
theorem students_not_solving_any_problem : 
  (A_1 = 4 * A_2) →
  (A_2 = 4 * A_3) →
  (A_3 = 4 * A_4) →
  (A_4 = 4 * A_5) →
  (A_5 = 4 * A_6) →
  (A_0 + A_1 = 2006) →
  (A_0 = 982) :=
by
  intro h1 h2 h3 h4 h5 h6
  -- Proof should go here
  sorry

end students_not_solving_any_problem_l77_77561


namespace integer_part_sqrt_sum_l77_77233

theorem integer_part_sqrt_sum {a b c : ℤ} 
  (h_a : |a| = 4) 
  (h_b_sqrt : b^2 = 9) 
  (h_c_cubert : c^3 = -8) 
  (h_order : a > b ∧ b > c) 
  : (⌊ Real.sqrt (a + b + c) ⌋) = 2 := 
by 
  sorry

end integer_part_sqrt_sum_l77_77233


namespace probability_of_vowel_initials_l77_77216

/-- In a class with 26 students, each student has unique initials that are double letters
    (i.e., AA, BB, ..., ZZ). If the vowels are A, E, I, O, U, and W, then the probability of
    randomly picking a student whose initials are vowels is 3/13. -/
theorem probability_of_vowel_initials :
  let total_students := 26
  let vowels := ['A', 'E', 'I', 'O', 'U', 'W']
  let num_vowels := 6
  let probability := num_vowels / total_students
  probability = 3 / 13 :=
by
  sorry

end probability_of_vowel_initials_l77_77216


namespace value_of_x_minus_2y_l77_77560

theorem value_of_x_minus_2y (x y : ℝ) (h1 : 0.5 * x = y + 20) : x - 2 * y = 40 :=
by
  sorry

end value_of_x_minus_2y_l77_77560


namespace sequence_sum_l77_77916

theorem sequence_sum (P Q R S T U V : ℕ) (h1 : S = 7)
  (h2 : P + Q + R = 21) (h3 : Q + R + S = 21)
  (h4 : R + S + T = 21) (h5 : S + T + U = 21)
  (h6 : T + U + V = 21) : P + V = 14 :=
by
  sorry

end sequence_sum_l77_77916


namespace average_score_difference_l77_77637

theorem average_score_difference {A B : ℝ} (hA : (19 * A + 125) / 20 = A + 5) (hB : (17 * B + 145) / 18 = B + 6) :
  (B + 6) - (A + 5) = 13 :=
  sorry

end average_score_difference_l77_77637


namespace tooth_fairy_left_amount_l77_77790

-- Define the values of the different types of coins
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50
def dime_value : ℝ := 0.10

-- Define the number of each type of coins Joan received
def num_quarters : ℕ := 14
def num_half_dollars : ℕ := 14
def num_dimes : ℕ := 14

-- Calculate the total values for each type of coin
def total_quarters_value : ℝ := num_quarters * quarter_value
def total_half_dollars_value : ℝ := num_half_dollars * half_dollar_value
def total_dimes_value : ℝ := num_dimes * dime_value

-- The total amount of money left by the tooth fairy
def total_amount_left := total_quarters_value + total_half_dollars_value + total_dimes_value

-- The theorem stating that the total amount is $11.90
theorem tooth_fairy_left_amount : total_amount_left = 11.90 := by 
  sorry

end tooth_fairy_left_amount_l77_77790


namespace find_E_coordinates_l77_77466

structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := -2, y := 1}
def B : Point := {x := 1, y := 4}
def C : Point := {x := 4, y := -3}
def D : Point := {x := (-2 * 1 + 1 * (-2)) / (1 + 2), y := (1 * 4 + 2 * 1) / (1 + 2)}

def externalDivision (P1 P2 : Point) (m n : ℚ) : Point :=
  {x := (m * P2.x - n * P1.x) / (m - n), y := (m * P2.y - n * P1.y) / (m - n)}

theorem find_E_coordinates :
  let E := externalDivision D C 1 4
  E.x = -8 / 3 ∧ E.y = 11 / 3 := 
by 
  let E := externalDivision D C 1 4
  sorry

end find_E_coordinates_l77_77466


namespace girls_with_brown_eyes_and_light_brown_skin_l77_77901

theorem girls_with_brown_eyes_and_light_brown_skin 
  (total_girls : ℕ)
  (light_brown_skin_girls : ℕ)
  (blue_eyes_fair_skin_girls : ℕ)
  (brown_eyes_total : ℕ)
  (total_girls_50 : total_girls = 50)
  (light_brown_skin_31 : light_brown_skin_girls = 31)
  (blue_eyes_fair_skin_14 : blue_eyes_fair_skin_girls = 14)
  (brown_eyes_18 : brown_eyes_total = 18) :
  ∃ (brown_eyes_light_brown_skin_girls : ℕ), brown_eyes_light_brown_skin_girls = 13 :=
by sorry

end girls_with_brown_eyes_and_light_brown_skin_l77_77901


namespace hyperbola_foci_distance_l77_77345

theorem hyperbola_foci_distance (c : ℝ) (h : c = Real.sqrt 2) : 
  let f1 := (c * Real.sqrt 2, c * Real.sqrt 2)
  let f2 := (-c * Real.sqrt 2, -c * Real.sqrt 2)
  Real.sqrt ((f2.1 - f1.1) ^ 2 + (f2.2 - f1.2) ^ 2) = 4 * Real.sqrt 2 := 
by
  sorry

end hyperbola_foci_distance_l77_77345


namespace last_three_digits_of_5_pow_9000_l77_77472

theorem last_three_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 800]) : 5^9000 ≡ 1 [MOD 800] :=
by
  -- The proof is omitted here according to the instruction
  sorry

end last_three_digits_of_5_pow_9000_l77_77472


namespace angle_bisector_eqn_l77_77455

-- Define the vertices A, B, and C
def A : (ℝ × ℝ) := (4, 3)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (9, -7)

-- State the theorem with conditions and the given answer
theorem angle_bisector_eqn (A B C : (ℝ × ℝ)) (hA : A = (4, 3)) (hB : B = (-4, -1)) (hC : C = (9, -7)) :
  ∃ b c, (3:ℝ) * (3:ℝ) - b * (3:ℝ) + c = 0 ∧ b + c = -6 := 
by 
  use -1, -5
  simp
  sorry

end angle_bisector_eqn_l77_77455


namespace non_deg_ellipse_projection_l77_77026

theorem non_deg_ellipse_projection (m : ℝ) : 
  (3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m → (m > -21)) := 
by
  sorry

end non_deg_ellipse_projection_l77_77026


namespace combination_sum_eq_l77_77514

theorem combination_sum_eq :
  ∀ (n : ℕ), (2 * n ≥ 10 - 2 * n) ∧ (3 + n ≥ 2 * n) →
  Nat.choose (2 * n) (10 - 2 * n) + Nat.choose (3 + n) (2 * n) = 16 :=
by
  intro n h
  cases' h with h1 h2
  sorry

end combination_sum_eq_l77_77514


namespace solution_set_of_quadratic_inequality_l77_77223

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) := 
sorry

end solution_set_of_quadratic_inequality_l77_77223


namespace ball_radius_and_surface_area_l77_77571

theorem ball_radius_and_surface_area (d h r : ℝ) (radius_eq : d / 2 = 6) (depth_eq : h = 2) 
  (pythagorean : (r - h)^2 + (d / 2)^2 = r^2) :
  r = 10 ∧ (4 * Real.pi * r^2 = 400 * Real.pi) :=
by
  sorry

end ball_radius_and_surface_area_l77_77571


namespace find_rate_of_interest_l77_77994

-- Define the problem conditions
def principal_B : ℝ := 4000
def principal_C : ℝ := 2000
def time_B : ℝ := 2
def time_C : ℝ := 4
def total_interest : ℝ := 2200

-- Define the unknown rate of interest per annum
noncomputable def rate_of_interest (R : ℝ) : Prop :=
  let interest_B := (principal_B * R * time_B) / 100
  let interest_C := (principal_C * R * time_C) / 100
  interest_B + interest_C = total_interest

-- Statement to prove that the rate of interest is 13.75%
theorem find_rate_of_interest : rate_of_interest 13.75 := by
  sorry

end find_rate_of_interest_l77_77994


namespace log_b_243_values_l77_77720

theorem log_b_243_values : 
  ∃! (s : Finset ℕ), (∀ b ∈ s, ∃ n : ℕ, b^n = 243) ∧ s.card = 2 :=
by 
  sorry

end log_b_243_values_l77_77720


namespace area_increase_l77_77389

theorem area_increase (r₁ r₂: ℝ) (A₁ A₂: ℝ) (side1 side2: ℝ) 
  (h1: side1 = 8) (h2: side2 = 12) (h3: r₁ = side2 / 2) (h4: r₂ = side1 / 2)
  (h5: A₁ = 2 * (1/2 * Real.pi * r₁ ^ 2) + 2 * (1/2 * Real.pi * r₂ ^ 2))
  (h6: A₂ = 4 * (Real.pi * r₂ ^ 2))
  (h7: A₁ = 52 * Real.pi) (h8: A₂ = 64 * Real.pi) :
  ((A₁ + A₂) - A₁) / A₁ * 100 = 123 :=
by
  sorry

end area_increase_l77_77389


namespace negation_of_exists_l77_77709

theorem negation_of_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 2 > 0) = ∀ x : ℝ, x^2 - x + 2 ≤ 0 := by
  sorry

end negation_of_exists_l77_77709


namespace scientific_notation_of_384000_l77_77006

theorem scientific_notation_of_384000 : 384000 = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l77_77006


namespace number_of_comedies_rented_l77_77945

noncomputable def comedies_rented (r : ℕ) (a : ℕ) : ℕ := 3 * a

theorem number_of_comedies_rented (a : ℕ) (h : a = 5) : comedies_rented 3 a = 15 := by
  rw [h]
  exact rfl

end number_of_comedies_rented_l77_77945


namespace cos_reflected_value_l77_77684

theorem cos_reflected_value (x : ℝ) (h : Real.cos (π / 6 + x) = 1 / 3) :
  Real.cos (5 * π / 6 - x) = -1 / 3 := 
by {
  sorry
}

end cos_reflected_value_l77_77684


namespace manicure_cost_before_tip_l77_77681

theorem manicure_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_paid = 39 → tip_percentage = 0.30 → total_paid = cost_before_tip + tip_percentage * cost_before_tip → cost_before_tip = 30 :=
by
  intro h1 h2 h3
  sorry

end manicure_cost_before_tip_l77_77681


namespace original_price_l77_77347

variable (P : ℝ)

theorem original_price (h : 560 = 1.05 * (0.72 * P)) : P = 740.46 := 
by
  sorry

end original_price_l77_77347


namespace mutually_exclusive_not_complementary_l77_77692

-- Definitions of events
def EventA (n : ℕ) : Prop := n % 2 = 1
def EventB (n : ℕ) : Prop := n % 2 = 0
def EventC (n : ℕ) : Prop := n % 2 = 0
def EventD (n : ℕ) : Prop := n = 2 ∨ n = 4

-- Mutual exclusivity and complementarity
def mutually_exclusive {α : Type} (A B : α → Prop) : Prop :=
∀ x, ¬ (A x ∧ B x)

def complementary {α : Type} (A B : α → Prop) : Prop :=
∀ x, A x ∨ B x

-- The statement to be proved
theorem mutually_exclusive_not_complementary :
  mutually_exclusive EventA EventD ∧ ¬ complementary EventA EventD :=
by sorry

end mutually_exclusive_not_complementary_l77_77692


namespace sufficient_but_not_necessary_l77_77342

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 0 → x^2 + x > 0) ∧ (∃ y : ℝ, y < -1 ∧ y^2 + y > 0) :=
by
  sorry

end sufficient_but_not_necessary_l77_77342


namespace range_of_a_l77_77581

-- Define the propositions
def p (x : ℝ) := (x - 1) * (x - 2) > 0
def q (a x : ℝ) := x^2 + (a - 1) * x - a > 0

-- Define the solution sets
def A := {x : ℝ | p x}
def B (a : ℝ) := {x : ℝ | q a x}

-- State the proof problem
theorem range_of_a (a : ℝ) : 
  (∀ x, p x → q a x) ∧ (∃ x, ¬p x ∧ q a x) → -2 < a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l77_77581


namespace find_a_n_geo_b_find_S_2n_l77_77821
noncomputable def S : ℕ → ℚ
| n => (n^2 + n + 1) / 2

def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else n

theorem find_a_n (n : ℕ) : a n = if n = 1 then 3/2 else n :=
by
  sorry

def b (n : ℕ) : ℚ :=
  a (2 * n - 1) + a (2 * n)

theorem geo_b (n : ℕ) : b (n + 1) = 3 * b n :=
by
  sorry

theorem find_S_2n (n : ℕ) : S (2 * n) = 3/2 * (3^n - 1) :=
by
  sorry

end find_a_n_geo_b_find_S_2n_l77_77821


namespace simplify_expression_l77_77793

noncomputable def a : ℝ := Real.sqrt 3 - 1

theorem simplify_expression : 
  ( (a - 1) / (a^2 - 2 * a + 1) / ( (a^2 + a) / (a^2 - 1) + 1 / (a - 1) ) = Real.sqrt 3 / 3 ) :=
by
  sorry

end simplify_expression_l77_77793


namespace xy_solution_l77_77944

theorem xy_solution (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end xy_solution_l77_77944


namespace rotated_angle_l77_77610

theorem rotated_angle (angle_ACB_initial : ℝ) (rotation_angle : ℝ) (h1 : angle_ACB_initial = 60) (h2 : rotation_angle = 630) : 
  ∃ (angle_ACB_new : ℝ), angle_ACB_new = 30 :=
by
  -- Define the effective rotation
  let effective_rotation := rotation_angle % 360 -- Modulo operation
  
  -- Calculate the new angle
  let angle_new := angle_ACB_initial + effective_rotation
  
  -- Ensure the angle is acute by converting if needed
  let acute_angle_new := if angle_new > 180 then 360 - angle_new else angle_new
  
  -- The acute angle should be 30 degrees
  use acute_angle_new
  have : acute_angle_new = 30 := sorry
  exact this

end rotated_angle_l77_77610


namespace sum_arithmetic_sequence_l77_77712

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n + 1) / 2 * (2 * a 0 + n * (a 1 - a 0))

theorem sum_arithmetic_sequence (h_arith : arithmetic_sequence a) (h_condition : a 3 + a 4 + a 5 + a 6 = 18) :
  S a 9 = 45 :=
sorry

end sum_arithmetic_sequence_l77_77712


namespace largest_four_digit_sum_20_l77_77048

-- Defining the four-digit number and conditions.
def is_four_digit_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  ∃ a b c d : ℕ, a + b + c + d = s ∧ n = 1000 * a + 100 * b + 10 * c + d

-- Proof problem statement.
theorem largest_four_digit_sum_20 : ∃ n, is_four_digit_number n ∧ digits_sum_to n 20 ∧ ∀ m, is_four_digit_number m ∧ digits_sum_to m 20 → m ≤ n :=
  sorry

end largest_four_digit_sum_20_l77_77048


namespace ratio_of_areas_l77_77494

-- Definitions and conditions
variables (s r : ℝ)
variables (h1 : 4 * s = 4 * π * r)

-- Statement to prove
theorem ratio_of_areas (h1 : 4 * s = 4 * π * r) : s^2 / (π * r^2) = π := by
  sorry

end ratio_of_areas_l77_77494


namespace population_net_increase_period_l77_77370

def period_in_hours (birth_rate : ℕ) (death_rate : ℕ) (net_increase : ℕ) : ℕ :=
  let net_rate_per_second := (birth_rate / 2) - (death_rate / 2)
  let period_in_seconds := net_increase / net_rate_per_second
  period_in_seconds / 3600

theorem population_net_increase_period :
  period_in_hours 10 2 345600 = 24 :=
by
  unfold period_in_hours
  sorry

end population_net_increase_period_l77_77370


namespace find_N_l77_77374

theorem find_N (N : ℕ) (h₁ : ∃ (d₁ d₂ : ℕ), d₁ + d₂ = 3333 ∧ N = max d₁ d₂ ∧ (max d₁ d₂) / (min d₁ d₂) = 2) : 
  N = 2222 := sorry

end find_N_l77_77374


namespace correct_option_B_l77_77644

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end correct_option_B_l77_77644


namespace emily_lives_lost_l77_77329

variable (L : ℕ)
variable (initial_lives : ℕ) (extra_lives : ℕ) (final_lives : ℕ)

-- Conditions based on the problem statement
axiom initial_lives_def : initial_lives = 42
axiom extra_lives_def : extra_lives = 24
axiom final_lives_def : final_lives = 41

-- Mathematically equivalent proof statement
theorem emily_lives_lost : initial_lives - L + extra_lives = final_lives → L = 25 := by
  sorry

end emily_lives_lost_l77_77329


namespace parallel_lines_from_perpendicularity_l77_77145

variables (a b : Type) (α β : Type)

-- Define the necessary conditions
def is_line (l : Type) : Prop := sorry
def is_plane (p : Type) : Prop := sorry
def perpendicular (l : Type) (p : Type) : Prop := sorry
def parallel (l1 l2 : Type) : Prop := sorry

axiom line_a : is_line a
axiom line_b : is_line b
axiom plane_alpha : is_plane α
axiom plane_beta : is_plane β
axiom a_perp_alpha : perpendicular a α
axiom b_perp_alpha : perpendicular b α

-- State the theorem
theorem parallel_lines_from_perpendicularity : parallel a b :=
  sorry

end parallel_lines_from_perpendicularity_l77_77145


namespace probability_one_solves_l77_77563

theorem probability_one_solves :
  let pA := 0.8
  let pB := 0.7
  (pA * (1 - pB) + pB * (1 - pA)) = 0.38 :=
by
  sorry

end probability_one_solves_l77_77563


namespace sequence_count_zeros_ones_15_l77_77227

-- Definition of the problem
def count_sequences (n : Nat) : Nat := sorry -- Function calculating the number of valid sequences

-- The theorem stating that for sequence length 15, the number of such sequences is 266
theorem sequence_count_zeros_ones_15 : count_sequences 15 = 266 := 
by {
  sorry -- Proof goes here
}

end sequence_count_zeros_ones_15_l77_77227


namespace probability_at_least_one_visits_guangzhou_l77_77582

-- Define the probabilities of visiting for persons A, B, and C
def p_A : ℚ := 2 / 3
def p_B : ℚ := 1 / 4
def p_C : ℚ := 3 / 5

-- Calculate the probability that no one visits
def p_not_A : ℚ := 1 - p_A
def p_not_B : ℚ := 1 - p_B
def p_not_C : ℚ := 1 - p_C

-- Calculate the probability that at least one person visits
def p_none_visit : ℚ := p_not_A * p_not_B * p_not_C
def p_at_least_one_visit : ℚ := 1 - p_none_visit

-- The statement we need to prove
theorem probability_at_least_one_visits_guangzhou : p_at_least_one_visit = 9 / 10 :=
by 
  sorry

end probability_at_least_one_visits_guangzhou_l77_77582


namespace cos_pi_minus_2alpha_l77_77001

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 := 
by
  sorry

end cos_pi_minus_2alpha_l77_77001


namespace solve_real_numbers_l77_77361

theorem solve_real_numbers (x y : ℝ) :
  (x = 3 * x^2 * y - y^3) ∧ (y = x^3 - 3 * x * y^2) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 + Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = (Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 + Real.sqrt 2)) / 2)) :=
by
  sorry

end solve_real_numbers_l77_77361


namespace marcella_matching_pairs_l77_77021

theorem marcella_matching_pairs (P : ℕ) (L : ℕ) (H : P = 20) (H1 : L = 9) : (P - L) / 2 = 11 :=
by
  -- definition of P and L are given by 20 and 9 respectively
  -- proof is omitted for the statement focus
  sorry

end marcella_matching_pairs_l77_77021


namespace problem_I_problem_II_l77_77730

namespace ProofProblems

def f (x a : ℝ) : ℝ := |x - a| + |x + 5|

theorem problem_I (x : ℝ) : (f x 1) ≥ 2 * |x + 5| ↔ x ≤ -2 := 
by sorry

theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, (f x a) ≥ 8) ↔ (a ≥ 3 ∨ a ≤ -13) := 
by sorry

end ProofProblems

end problem_I_problem_II_l77_77730


namespace susie_pizza_sales_l77_77817

theorem susie_pizza_sales :
  ∃ x : ℕ, 
    (24 * 3 + 15 * x = 117) ∧ 
    x = 3 := 
by
  sorry

end susie_pizza_sales_l77_77817


namespace inequality_proof_l77_77359

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l77_77359


namespace percent_of_games_lost_l77_77800

theorem percent_of_games_lost (w l : ℕ) (h1 : w / l = 8 / 5) (h2 : w + l = 65) :
  (l * 100 / 65 : ℕ) = 38 :=
sorry

end percent_of_games_lost_l77_77800


namespace larger_number_is_391_l77_77315

-- Define the H.C.F and factors
def HCF := 23
def factor1 := 13
def factor2 := 17
def LCM := HCF * factor1 * factor2

-- Define the two numbers based on the factors
def number1 := HCF * factor1
def number2 := HCF * factor2

-- Theorem statement
theorem larger_number_is_391 : max number1 number2 = 391 := 
by
  sorry

end larger_number_is_391_l77_77315


namespace δ_can_be_arbitrarily_small_l77_77014

-- Define δ(r) as the distance from the circle to the nearest point with integer coordinates.
def δ (r : ℝ) : ℝ := sorry -- exact definition would depend on the implementation details

-- The main theorem to be proven.
theorem δ_can_be_arbitrarily_small (ε : ℝ) (hε : ε > 0) : ∃ r : ℝ, r > 0 ∧ δ r < ε :=
sorry

end δ_can_be_arbitrarily_small_l77_77014


namespace opposite_of_9_is_neg_9_l77_77360

-- Definition of opposite number according to the given condition
def opposite (n : Int) : Int := -n

-- Proof statement that the opposite of 9 is -9
theorem opposite_of_9_is_neg_9 : opposite 9 = -9 :=
by
  sorry

end opposite_of_9_is_neg_9_l77_77360


namespace sum_of_altitudes_of_triangle_l77_77659

open Real

noncomputable def sum_of_altitudes (a b c : ℝ) : ℝ :=
  let inter_x := -c / a
  let inter_y := -c / b
  let vertex1 := (inter_x, 0)
  let vertex2 := (0, inter_y)
  let vertex3 := (0, 0)
  let area_triangle := (1 / 2) * abs (inter_x * inter_y)
  let altitude_x := abs inter_x
  let altitude_y := abs inter_y
  let altitude_line := abs c / sqrt (a ^ 2 + b ^ 2)
  altitude_x + altitude_y + altitude_line

theorem sum_of_altitudes_of_triangle :
  sum_of_altitudes 15 6 90 = 21 + 10 * sqrt (1 / 29) :=
by
  sorry

end sum_of_altitudes_of_triangle_l77_77659


namespace inequality_proof_l77_77627

theorem inequality_proof
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 := 
by
  sorry

end inequality_proof_l77_77627


namespace tetrahedron_edges_midpoint_distances_sum_l77_77252

theorem tetrahedron_edges_midpoint_distances_sum (a b c d e f m1 m2 m3 m4 m5 m6 : ℝ) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (m1^2 + m2^2 + m3^2 + m4^2 + m5^2 + m6^2) :=
sorry

end tetrahedron_edges_midpoint_distances_sum_l77_77252


namespace domain_composite_l77_77539

-- Define the conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- The theorem statement
theorem domain_composite (h : ∀ x, domain_f x → 0 ≤ x ∧ x ≤ 4) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2) :=
by
  sorry

end domain_composite_l77_77539


namespace mean_goals_l77_77094

theorem mean_goals :
  let goals := 2 * 3 + 4 * 2 + 5 * 1 + 6 * 1
  let players := 3 + 2 + 1 + 1
  goals / players = 25 / 7 :=
by
  sorry

end mean_goals_l77_77094


namespace jerome_gave_to_meg_l77_77132

theorem jerome_gave_to_meg (init_money half_money given_away meg bianca : ℝ) 
    (h1 : half_money = 43) 
    (h2 : init_money = 2 * half_money) 
    (h3 : 54 = init_money - given_away)
    (h4 : given_away = meg + bianca)
    (h5 : bianca = 3 * meg) : 
    meg = 8 :=
by
  sorry

end jerome_gave_to_meg_l77_77132


namespace quadratic_transformation_l77_77843

noncomputable def transform_roots (p q r : ℚ) (u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) : Prop :=
  ∃ y : ℚ, y^2 - q^2 + 4 * p * r = 0

theorem quadratic_transformation (p q r u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) :
  ∃ y : ℚ, (y - (2 * p * u + q)) * (y - (2 * p * v + q)) = y^2 - q^2 + 4 * p * r :=
by {
  sorry
}

end quadratic_transformation_l77_77843


namespace total_hangers_is_65_l77_77072

noncomputable def calculate_hangers_total : ℕ :=
  let pink := 7
  let green := 4
  let blue := green - 1
  let yellow := blue - 1
  let orange := 2 * (pink + green)
  let purple := (blue - yellow) + 3
  let red := (pink + green + blue) / 3
  let brown := 3 * red + 1
  let gray := (3 * purple) / 5
  pink + green + blue + yellow + orange + purple + red + brown + gray

theorem total_hangers_is_65 : calculate_hangers_total = 65 := 
by 
  sorry

end total_hangers_is_65_l77_77072


namespace find_a_plus_b_l77_77900

theorem find_a_plus_b (x a b : ℝ) (ha : x = a + Real.sqrt b)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : x^2 + 5 * x + 4/x + 1/(x^2) = 34) : a + b = 5 :=
sorry

end find_a_plus_b_l77_77900


namespace deg_d_eq_6_l77_77172

theorem deg_d_eq_6
  (f d q : Polynomial ℝ)
  (r : Polynomial ℝ)
  (hf : f.degree = 15)
  (hdq : (d * q + r) = f)
  (hq : q.degree = 9)
  (hr : r.degree = 4) :
  d.degree = 6 :=
by sorry

end deg_d_eq_6_l77_77172


namespace nth_equation_pattern_l77_77719

theorem nth_equation_pattern (n: ℕ) :
  (∀ k : ℕ, 1 ≤ k → ∃ a b c d : ℕ, (a * c ≠ 0) ∧ (b * d ≠ 0) ∧ (a = k) ∧ (b = k + 1) → 
    (a + 3 * (2 * a)) / (b + 3 * (2 * b)) = a / b) :=
by
  sorry

end nth_equation_pattern_l77_77719


namespace solve_for_m_l77_77283

def z1 := Complex.mk 3 2
def z2 (m : ℝ) := Complex.mk 1 m

theorem solve_for_m (m : ℝ) (h : (z1 * z2 m).re = 0) : m = 3 / 2 :=
by
  sorry

end solve_for_m_l77_77283


namespace cut_piece_ratio_l77_77773

noncomputable def original_log_length : ℕ := 20
noncomputable def weight_per_foot : ℕ := 150
noncomputable def cut_piece_weight : ℕ := 1500

theorem cut_piece_ratio :
  (cut_piece_weight / weight_per_foot / original_log_length) = (1 / 2) := by
  sorry

end cut_piece_ratio_l77_77773


namespace find_integer_pairs_l77_77751

theorem find_integer_pairs :
  ∃ (x y : ℤ),
    (x, y) = (-7, -99) ∨ (x, y) = (-1, -9) ∨ (x, y) = (1, 5) ∨ (x, y) = (7, -97) ∧
    2 * x^3 + x * y - 7 = 0 :=
by
  sorry

end find_integer_pairs_l77_77751


namespace fraction_comparison_l77_77306

theorem fraction_comparison : (9 / 16) > (5 / 9) :=
by {
  sorry -- the detailed proof is not required for this task
}

end fraction_comparison_l77_77306


namespace distance_interval_l77_77266

theorem distance_interval (d : ℝ) (h1 : ¬(d ≥ 8)) (h2 : ¬(d ≤ 7)) (h3 : ¬(d ≤ 6 → north)):
  7 < d ∧ d < 8 :=
by
  have h_d8 : d < 8 := by linarith
  have h_d7 : d > 7 := by linarith
  exact ⟨h_d7, h_d8⟩

end distance_interval_l77_77266


namespace smallest_number_of_coins_to_pay_up_to_2_dollars_l77_77241

def smallest_number_of_coins_to_pay_up_to (max_amount : Nat) : Nat :=
  sorry  -- This function logic needs to be defined separately

theorem smallest_number_of_coins_to_pay_up_to_2_dollars :
  smallest_number_of_coins_to_pay_up_to 199 = 11 :=
sorry

end smallest_number_of_coins_to_pay_up_to_2_dollars_l77_77241


namespace polynomial_evaluation_l77_77999

theorem polynomial_evaluation 
  (x : ℝ) 
  (h1 : x^2 - 3 * x - 10 = 0) 
  (h2 : x > 0) : 
  (x^4 - 3 * x^3 + 2 * x^2 + 5 * x - 7) = 318 :=
by
  sorry

end polynomial_evaluation_l77_77999


namespace tom_total_seashells_l77_77860

-- Define the number of seashells Tom gave to Jessica.
def seashells_given_to_jessica : ℕ := 2

-- Define the number of seashells Tom still has.
def seashells_tom_has_now : ℕ := 3

-- Theorem stating that the total number of seashells Tom found is the sum of seashells_given_to_jessica and seashells_tom_has_now.
theorem tom_total_seashells : seashells_given_to_jessica + seashells_tom_has_now = 5 := 
by
  sorry

end tom_total_seashells_l77_77860


namespace arithmetic_expression_evaluation_l77_77171

theorem arithmetic_expression_evaluation : 
  ∃ (a b c d e f : Float),
  a - b * c / d + e = 0 ∧
  a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 ∧ e = 1 := sorry

end arithmetic_expression_evaluation_l77_77171


namespace find_x_l77_77522

theorem find_x (x : ℝ) (h : 128/x + 75/x + 57/x = 6.5) : x = 40 :=
by
  sorry

end find_x_l77_77522


namespace total_weight_of_towels_is_40_lbs_l77_77967

def number_of_towels_Mary := 24
def factor_Mary_Frances := 4
def weight_Frances_towels_oz := 128
def pounds_per_ounce := 1 / 16

def number_of_towels_Frances := number_of_towels_Mary / factor_Mary_Frances

def total_number_of_towels := number_of_towels_Mary + number_of_towels_Frances
def weight_per_towel_oz := weight_Frances_towels_oz / number_of_towels_Frances

def total_weight_oz := total_number_of_towels * weight_per_towel_oz
def total_weight_lbs := total_weight_oz * pounds_per_ounce

theorem total_weight_of_towels_is_40_lbs :
  total_weight_lbs = 40 :=
sorry

end total_weight_of_towels_is_40_lbs_l77_77967


namespace calculate_total_money_l77_77626

noncomputable def cost_per_gumdrop : ℕ := 4
noncomputable def number_of_gumdrops : ℕ := 20
noncomputable def total_money : ℕ := 80

theorem calculate_total_money : 
  cost_per_gumdrop * number_of_gumdrops = total_money := 
by
  sorry

end calculate_total_money_l77_77626


namespace func_increasing_l77_77995

noncomputable def func (x : ℝ) : ℝ :=
  x^3 + x + 1

theorem func_increasing : ∀ x : ℝ, deriv func x > 0 := by
  sorry

end func_increasing_l77_77995


namespace total_marks_secured_l77_77067

-- Define the conditions
def correct_points_per_question := 4
def wrong_points_per_question := 1
def total_questions := 60
def correct_questions := 40

-- Calculate the remaining incorrect questions
def wrong_questions := total_questions - correct_questions

-- Calculate total marks secured by the student
def total_marks := (correct_questions * correct_points_per_question) - (wrong_questions * wrong_points_per_question)

-- The statement to be proven
theorem total_marks_secured : total_marks = 140 := by
  -- This will be proven in Lean's proof assistant
  sorry

end total_marks_secured_l77_77067


namespace find_g_neg_6_l77_77946

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l77_77946


namespace work_completion_days_l77_77727

theorem work_completion_days (A B C : ℕ) (work_rate_A : A = 4) (work_rate_B : B = 10) (work_rate_C : C = 20 / 3) :
  (1 / A) + (1 / B) + (3 / C) = 1 / 2 :=
by
  sorry

end work_completion_days_l77_77727


namespace halfway_fraction_l77_77819

-- Assume a definition for the two fractions
def fracA : ℚ := 1 / 4
def fracB : ℚ := 1 / 7

-- Define the target property we want to prove
theorem halfway_fraction : (fracA + fracB) / 2 = 11 / 56 := 
by 
  -- Proof will happen here, adding sorry to indicate it's skipped for now
  sorry

end halfway_fraction_l77_77819


namespace min_value_frac_2_over_a_plus_3_over_b_l77_77256

theorem min_value_frac_2_over_a_plus_3_over_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 :=
sorry

end min_value_frac_2_over_a_plus_3_over_b_l77_77256


namespace power_of_two_ends_with_identical_digits_l77_77568

theorem power_of_two_ends_with_identical_digits : ∃ (k : ℕ), k ≥ 10 ∧ (∀ (x y : ℕ), 2^k = 1000 * x + 111 * y → y = 8 → (2^k % 1000 = 888)) :=
by sorry

end power_of_two_ends_with_identical_digits_l77_77568


namespace problem_statement_l77_77897

def reading_method (n : ℕ) : String := sorry
-- Assume reading_method correctly implements the reading method for integers

def is_read_with_only_one_zero (n : ℕ) : Prop :=
  (reading_method n).count '0' = 1

theorem problem_statement : is_read_with_only_one_zero 83721000 = false := sorry

end problem_statement_l77_77897


namespace max_product_of_xy_l77_77996

open Real

theorem max_product_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) :
  x * y ≤ 1 / 16 := 
sorry

end max_product_of_xy_l77_77996


namespace inverse_proportion_inequality_l77_77274

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l77_77274


namespace longest_tape_length_l77_77968

theorem longest_tape_length (a b c : ℕ) (h1 : a = 600) (h2 : b = 500) (h3 : c = 1200) : Nat.gcd (Nat.gcd a b) c = 100 :=
by
  sorry

end longest_tape_length_l77_77968


namespace generic_packages_needed_eq_2_l77_77476

-- Define parameters
def tees_per_generic_package : ℕ := 12
def tees_per_aero_package : ℕ := 2
def members_foursome : ℕ := 4
def tees_needed_per_member : ℕ := 20
def aero_packages_purchased : ℕ := 28

-- Calculate total tees needed and total tees obtained from aero packages
def total_tees_needed : ℕ := members_foursome * tees_needed_per_member
def aero_tees_obtained : ℕ := aero_packages_purchased * tees_per_aero_package
def generic_tees_needed : ℕ := total_tees_needed - aero_tees_obtained

-- Prove the number of generic packages needed is 2
theorem generic_packages_needed_eq_2 : 
  generic_tees_needed / tees_per_generic_package = 2 :=
  sorry

end generic_packages_needed_eq_2_l77_77476


namespace stratified_sampling_red_balls_l77_77377

-- Define the conditions
def total_balls : ℕ := 1000
def red_balls : ℕ := 50
def sampled_balls : ℕ := 100

-- Prove that the number of red balls sampled using stratified sampling is 5
theorem stratified_sampling_red_balls :
  (red_balls : ℝ) / (total_balls : ℝ) * (sampled_balls : ℝ) = 5 := 
by
  sorry

end stratified_sampling_red_balls_l77_77377


namespace find_fake_coin_l77_77506

def coin_value (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def coin_weight (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def is_fake (weight : Nat) : Prop :=
  weight ≠ coin_weight 1 ∧ weight ≠ coin_weight 2 ∧ weight ≠ coin_weight 3 ∧ weight ≠ coin_weight 4

theorem find_fake_coin :
  ∃ (n : Nat) (w : Nat), (is_fake w) → ∃! (m : Nat), m ≠ w ∧ (m = coin_weight 1 ∨ m = coin_weight 2 ∨ m = coin_weight 3 ∨ m = coin_weight 4) := 
sorry

end find_fake_coin_l77_77506


namespace choir_members_count_l77_77534

theorem choir_members_count : 
  ∃ n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 1 ∧
    n % 8 = 5 ∧
    n % 9 = 2 ∧
    n = 241 :=
by
  -- Proof will follow
  sorry

end choir_members_count_l77_77534


namespace scientific_notation_correct_l77_77671

theorem scientific_notation_correct :
  ∃! (n : ℝ) (a : ℝ), 0.000000012 = a * 10 ^ n ∧ a = 1.2 ∧ n = -8 :=
by
  sorry

end scientific_notation_correct_l77_77671


namespace f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l77_77119

noncomputable def f (x : ℝ) : ℝ := (4 * Real.exp x) / (Real.exp x + 1)

theorem f_sin_periodic : ∀ x, f (Real.sin (x + 2 * Real.pi)) = f (Real.sin x) := sorry

theorem f_monotonically_increasing : ∀ x y, x < y → f x < f y := sorry

theorem f_minus_2_not_even : ¬(∀ x, f x - 2 = f (-x) - 2) := sorry

theorem f_symmetric_about_point : ∀ x, f x + f (-x) = 4 := sorry

end f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l77_77119


namespace Oliver_Battle_Gremlins_Card_Count_l77_77004

theorem Oliver_Battle_Gremlins_Card_Count 
  (MonsterClubCards AlienBaseballCards BattleGremlinsCards : ℕ)
  (h1 : MonsterClubCards = 2 * AlienBaseballCards)
  (h2 : BattleGremlinsCards = 3 * AlienBaseballCards)
  (h3 : MonsterClubCards = 32) : 
  BattleGremlinsCards = 48 := by
  sorry

end Oliver_Battle_Gremlins_Card_Count_l77_77004


namespace negation_of_p_l77_77294

noncomputable def p : Prop := ∀ x : ℝ, x > 0 → 2 * x^2 + 1 > 0

theorem negation_of_p : (∃ x : ℝ, x > 0 ∧ 2 * x^2 + 1 ≤ 0) ↔ ¬p :=
by
  sorry

end negation_of_p_l77_77294


namespace angles_in_interval_l77_77648

open Real

theorem angles_in_interval
    (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ ≤ 2 * π)
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 * sin θ - x * (2 - x) + (2 - x)^2 * cos θ > 0) :
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by
  sorry

end angles_in_interval_l77_77648


namespace range_of_m_l77_77604

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)) :
  0 < m ∧ m ≤ 1 / Real.exp 1 :=
sorry

end range_of_m_l77_77604


namespace solution_unique_l77_77121

def is_solution (x : ℝ) : Prop :=
  ⌊x * ⌊x⌋⌋ = 48

theorem solution_unique (x : ℝ) : is_solution x → x = -48 / 7 :=
by
  intro h
  -- Proof goes here
  sorry

end solution_unique_l77_77121


namespace solve_equation_l77_77290

theorem solve_equation (x : ℝ) (hx : x ≠ 0) : 
  x^2 + 36 / x^2 = 13 ↔ (x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3) := by
  sorry

end solve_equation_l77_77290


namespace find_a_100_l77_77888

noncomputable def a : Nat → Nat
| 0 => 0
| 1 => 2
| (n+1) => a n + 2 * n

theorem find_a_100 : a 100 = 9902 := 
  sorry

end find_a_100_l77_77888


namespace ratio_of_radii_l77_77734

theorem ratio_of_radii (a b c : ℝ) (h1 : π * c^2 - π * a^2 = 4 * π * a^2) (h2 : π * b^2 = (π * a^2 + π * c^2) / 2) :
  a / c = 1 / Real.sqrt 5 := by
  sorry

end ratio_of_radii_l77_77734


namespace books_per_shelf_l77_77587

def initial_coloring_books : ℕ := 86
def sold_coloring_books : ℕ := 37
def shelves : ℕ := 7

theorem books_per_shelf : (initial_coloring_books - sold_coloring_books) / shelves = 7 := by
  sorry

end books_per_shelf_l77_77587


namespace smallest_possible_value_l77_77786

theorem smallest_possible_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^2 + b^2) / (a * b) + (a * b) / (a^2 + b^2) ≥ 2 :=
sorry

end smallest_possible_value_l77_77786


namespace least_positive_divisible_by_five_primes_l77_77052

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l77_77052


namespace solve_for_x_l77_77295

theorem solve_for_x (x : ℝ) (hx₁ : x ≠ 3) (hx₂ : x ≠ -2) 
  (h : (x + 5) / (x - 3) = (x - 2) / (x + 2)) : x = -1 / 3 :=
by
  sorry

end solve_for_x_l77_77295


namespace driving_time_to_beach_l77_77138

theorem driving_time_to_beach (total_trip_time : ℝ) (k : ℝ) (x : ℝ)
  (h1 : total_trip_time = 14)
  (h2 : k = 2.5)
  (h3 : total_trip_time = (2 * x) + (k * (2 * x))) :
  x = 2 := by 
  sorry

end driving_time_to_beach_l77_77138


namespace intersects_negative_half_axis_range_l77_77394

noncomputable def f (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * m * x + 2 * m - 6

theorem intersects_negative_half_axis_range (m : ℝ) :
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) ↔ (∃ x : ℝ, f m x < 0) :=
sorry

end intersects_negative_half_axis_range_l77_77394


namespace minimize_quadratic_l77_77625

theorem minimize_quadratic (x : ℝ) :
  (∀ y : ℝ, x^2 + 14*x + 6 ≤ y^2 + 14*y + 6) ↔ x = -7 :=
by
  sorry

end minimize_quadratic_l77_77625


namespace product_of_distances_is_one_l77_77477

theorem product_of_distances_is_one (k : ℝ) (x1 x2 : ℝ)
  (h1 : x1^2 - k*x1 - 1 = 0)
  (h2 : x2^2 - k*x2 - 1 = 0)
  (h3 : x1 ≠ x2) :
  (|x1| * |x2| = 1) :=
by
  -- Proof goes here
  sorry

end product_of_distances_is_one_l77_77477


namespace sequence_sum_l77_77580

theorem sequence_sum (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sequence_sum_l77_77580


namespace smallest_odd_integer_of_set_l77_77662

theorem smallest_odd_integer_of_set (S : Set Int) 
  (h1 : ∃ m : Int, m ∈ S ∧ m = 149)
  (h2 : ∃ n : Int, n ∈ S ∧ n = 159)
  (h3 : ∀ a b : Int, a ∈ S → b ∈ S → a ≠ b → (a - b) % 2 = 0) : 
  ∃ s : Int, s ∈ S ∧ s = 137 :=
by sorry

end smallest_odd_integer_of_set_l77_77662


namespace isosceles_triangle_perimeter_l77_77273

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₁ : a = 12) (h₂ : b = 12) (h₃ : c = 17) : a + b + c = 41 :=
by
  rw [h₁, h₂, h₃]
  norm_num

end isosceles_triangle_perimeter_l77_77273


namespace percentage_error_edge_percentage_error_edge_l77_77388

open Real

-- Define the main context, E as the actual edge and E' as the calculated edge
variables (E E' : ℝ)

-- Condition: Error in calculating the area is 4.04%
axiom area_error : E' * E' = E * E * 1.0404

-- Statement: To prove that the percentage error in edge calculation is 2%
theorem percentage_error_edge : (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

-- Alternatively, include variable and condition definitions in the actual theorem statement
theorem percentage_error_edge' (E E' : ℝ) (h : E' * E' = E * E * 1.0404) : 
    (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

end percentage_error_edge_percentage_error_edge_l77_77388


namespace speed_of_second_train_l77_77131

/-- 
Given:
1. A train leaves Mumbai at 9 am at a speed of 40 kmph.
2. After one hour, another train leaves Mumbai in the same direction at an unknown speed.
3. The two trains meet at a distance of 80 km from Mumbai.

Prove that the speed of the second train is 80 kmph.
-/
theorem speed_of_second_train (v : ℝ) :
  (∃ (distance_first : ℝ) (distance_meet : ℝ) (initial_speed_first : ℝ) (hours_later : ℤ),
    distance_first = 40 ∧ distance_meet = 80 ∧ initial_speed_first = 40 ∧ hours_later = 1 ∧
    v = distance_meet / (distance_meet / initial_speed_first - hours_later)) → v = 80 := by
  sorry

end speed_of_second_train_l77_77131


namespace find_a2019_l77_77502

-- Arithmetic sequence
def a (n : ℕ) : ℤ := sorry -- to be defined later

-- Given conditions
def sum_first_five_terms (a: ℕ → ℤ) : Prop := a 1 + a 2 + a 3 + a 4 + a 5 = 15
def term_six (a: ℕ → ℤ) : Prop := a 6 = 6

-- Question (statement to be proved)
def term_2019 (a: ℕ → ℤ) : Prop := a 2019 = 2019

-- Main theorem to be proved
theorem find_a2019 (a: ℕ → ℤ) 
  (h1 : sum_first_five_terms a)
  (h2 : term_six a) : 
  term_2019 a := 
by
  sorry

end find_a2019_l77_77502


namespace parabola_focus_l77_77071

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l77_77071


namespace x_14_and_inverse_x_14_l77_77099

theorem x_14_and_inverse_x_14 (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + x⁻¹^14 = -1 :=
by
  sorry

end x_14_and_inverse_x_14_l77_77099


namespace point_distance_l77_77645

theorem point_distance (x : ℤ) : abs x = 2021 → (x = 2021 ∨ x = -2021) := 
sorry

end point_distance_l77_77645


namespace hotel_rolls_l77_77528

theorem hotel_rolls (m n : ℕ) (rel_prime : Nat.gcd m n = 1) : 
  let num_nut_rolls := 3
  let num_cheese_rolls := 3
  let num_fruit_rolls := 3
  let total_rolls := 9
  let num_guests := 3
  let rolls_per_guest := 3
  let probability_first_guest := (3 / 9) * (3 / 8) * (3 / 7)
  let probability_second_guest := (2 / 6) * (2 / 5) * (2 / 4)
  let probability_third_guest := 1
  let overall_probability := probability_first_guest * probability_second_guest * probability_third_guest
  overall_probability = (9 / 70) → m = 9 ∧ n = 70 → m + n = 79 :=
by
  intros
  sorry

end hotel_rolls_l77_77528


namespace stanley_walk_distance_l77_77742

variable (run_distance walk_distance : ℝ)

theorem stanley_walk_distance : 
  run_distance = 0.4 ∧ run_distance = walk_distance + 0.2 → walk_distance = 0.2 :=
by
  sorry

end stanley_walk_distance_l77_77742


namespace total_bill_is_correct_l77_77607

def number_of_adults : ℕ := 2
def number_of_children : ℕ := 5
def meal_cost : ℕ := 8

-- Define total number of people
def total_people : ℕ := number_of_adults + number_of_children

-- Define the total bill
def total_bill : ℕ := total_people * meal_cost

-- Theorem stating the total bill amount
theorem total_bill_is_correct : total_bill = 56 := by
  sorry

end total_bill_is_correct_l77_77607


namespace initial_birds_was_one_l77_77183

def initial_birds (b : Nat) : Prop :=
  b + 4 = 5

theorem initial_birds_was_one : ∃ b, initial_birds b ∧ b = 1 :=
by
  use 1
  unfold initial_birds
  sorry

end initial_birds_was_one_l77_77183


namespace amount_of_money_around_circumference_l77_77757

-- Define the given conditions
def horizontal_coins : ℕ := 6
def vertical_coins : ℕ := 4
def coin_value_won : ℕ := 100

-- The goal is to prove the total amount of money around the circumference
theorem amount_of_money_around_circumference : 
  (2 * (horizontal_coins - 2) + 2 * (vertical_coins - 2) + 4) * coin_value_won = 1600 :=
by
  sorry

end amount_of_money_around_circumference_l77_77757


namespace number_in_eighth_group_l77_77655

theorem number_in_eighth_group (employees groups n l group_size numbering_drawn starting_number: ℕ) 
(h1: employees = 200) 
(h2: groups = 40) 
(h3: n = 5) 
(h4: number_in_fifth_group = 23) 
(h5: starting_number + 4 * n = number_in_fifth_group) : 
  starting_number + 7 * n = 38 :=
by
  sorry

end number_in_eighth_group_l77_77655


namespace fuel_remaining_l77_77836

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end fuel_remaining_l77_77836


namespace ribeye_steak_cost_l77_77207

/-- Define the conditions in Lean -/
def appetizer_cost : ℕ := 8
def wine_cost : ℕ := 3
def wine_glasses : ℕ := 2
def dessert_cost : ℕ := 6
def total_spent : ℕ := 38
def tip_percentage : ℚ := 0.20

/-- Proving the cost of the ribeye steak before the discount -/
theorem ribeye_steak_cost (S : ℚ) (h : 20 + (S / 2) + (tip_percentage * (20 + S)) = total_spent) : S = 20 :=
by
  sorry

end ribeye_steak_cost_l77_77207


namespace batsman_new_average_l77_77206

-- Let A be the average score before the 16th inning
def avg_before (A : ℝ) : Prop :=
  ∃ total_runs: ℝ, total_runs = 15 * A

-- Condition 1: The batsman makes 64 runs in the 16th inning
def score_in_16th_inning := 64

-- Condition 2: This increases his average by 3 runs
def avg_increase (A : ℝ) : Prop :=
  A + 3 = (15 * A + score_in_16th_inning) / 16

theorem batsman_new_average (A : ℝ) (h1 : avg_before A) (h2 : avg_increase A) :
  (A + 3) = 19 :=
sorry

end batsman_new_average_l77_77206


namespace number_leaves_remainder_five_l77_77547

theorem number_leaves_remainder_five (k : ℕ) (n : ℕ) (least_num : ℕ) 
  (h₁ : least_num = 540)
  (h₂ : ∀ m, m % 12 = 5 → m ≥ least_num)
  (h₃ : n = 107) 
  : 540 % 107 = 5 :=
by sorry

end number_leaves_remainder_five_l77_77547


namespace finish_work_in_time_l77_77185

noncomputable def work_in_days_A (DA : ℕ) := DA
noncomputable def work_in_days_B (DA : ℕ) := DA / 2
noncomputable def combined_work_rate (DA : ℕ) : ℚ := 1 / work_in_days_A DA + 2 / work_in_days_A DA

theorem finish_work_in_time (DA : ℕ) (h_combined_rate : combined_work_rate DA = 0.25) : DA = 12 :=
sorry

end finish_work_in_time_l77_77185


namespace division_remainder_l77_77033

theorem division_remainder (q d D R : ℕ) (h_q : q = 40) (h_d : d = 72) (h_D : D = 2944) (h_div : D = d * q + R) : R = 64 :=
by sorry

end division_remainder_l77_77033


namespace Eiffel_Tower_model_scale_l77_77510

theorem Eiffel_Tower_model_scale
  (h_tower : ℝ := 324)
  (h_model_cm : ℝ := 18) :
  (h_tower / (h_model_cm / 100)) / 100 = 18 :=
by
  sorry

end Eiffel_Tower_model_scale_l77_77510


namespace positive_difference_between_two_numbers_l77_77791

theorem positive_difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : y^2 - 4 * x^2 = 80) : 
  |y - x| = 179.33 := 
by sorry

end positive_difference_between_two_numbers_l77_77791


namespace equal_lead_concentration_l77_77501

theorem equal_lead_concentration (x : ℝ) (h1 : 0 < x) (h2 : x < 6) (h3 : x < 12) 
: (x / 6 = (12 - x) / 12) → x = 4 := by
  sorry

end equal_lead_concentration_l77_77501


namespace range_of_a_l77_77789

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end range_of_a_l77_77789


namespace triangular_pyramid_nonexistence_l77_77238

theorem triangular_pyramid_nonexistence
    (h : ℕ)
    (hb : ℕ)
    (P : ℕ)
    (h_eq : h = 60)
    (hb_eq : hb = 61)
    (P_eq : P = 62) :
    ¬ ∃ (a b c : ℝ), a + b + c = P ∧ 60^2 = 61^2 - (a^2 / 3) :=
by 
  sorry

end triangular_pyramid_nonexistence_l77_77238


namespace square_carpet_side_length_l77_77147

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ s : ℝ, s * s = area ∧ 3 < s ∧ s < 4 :=
by
  sorry

end square_carpet_side_length_l77_77147


namespace total_savings_during_sale_l77_77465

theorem total_savings_during_sale :
  let regular_price_fox := 15
  let regular_price_pony := 20
  let pairs_fox := 3
  let pairs_pony := 2
  let total_discount := 22
  let discount_pony := 18.000000000000014
  let regular_total := (pairs_fox * regular_price_fox) + (pairs_pony * regular_price_pony)
  let discount_fox := total_discount - discount_pony
  (discount_fox / 100 * (pairs_fox * regular_price_fox)) + (discount_pony / 100 * (pairs_pony * regular_price_pony)) = 9 := by
  sorry

end total_savings_during_sale_l77_77465


namespace prove_value_of_custom_ops_l77_77503

-- Define custom operations to match problem statement
def custom_op1 (x : ℤ) : ℤ := 7 - x
def custom_op2 (x : ℤ) : ℤ := x - 10

-- The main proof statement
theorem prove_value_of_custom_ops : custom_op2 (custom_op1 12) = -15 :=
by sorry

end prove_value_of_custom_ops_l77_77503


namespace range_of_x_l77_77180

theorem range_of_x (a b x : ℝ) (h1 : a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) → (-7 ≤ x ∧ x ≤ 11) :=
by
  -- we provide the exact statement we aim to prove.
  sorry

end range_of_x_l77_77180


namespace initial_children_on_bus_l77_77613

-- Definitions based on conditions
variable (x : ℕ) -- number of children who got off the bus
variable (y : ℕ) -- initial number of children on the bus
variable (after_exchange : ℕ := 30) -- number of children on the bus after exchange
variable (got_on : ℕ := 82) -- number of children who got on the bus
variable (extra_on : ℕ := 2) -- extra children who got on compared to got off

-- Problem translated to Lean 4 statement
theorem initial_children_on_bus (h : got_on = x + extra_on) (hx : y + got_on - x = after_exchange) : y = 28 :=
by
  sorry

end initial_children_on_bus_l77_77613


namespace female_adults_present_l77_77918

variable (children : ℕ) (male_adults : ℕ) (total_people : ℕ)
variable (children_count : children = 80) (male_adults_count : male_adults = 60) (total_people_count : total_people = 200)

theorem female_adults_present : ∃ (female_adults : ℕ), 
  female_adults = total_people - (children + male_adults) ∧ 
  female_adults = 60 :=
by
  sorry

end female_adults_present_l77_77918


namespace sqrt_exp_sum_eq_eight_sqrt_two_l77_77136

theorem sqrt_exp_sum_eq_eight_sqrt_two : 
  (Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2) :=
by
  sorry

end sqrt_exp_sum_eq_eight_sqrt_two_l77_77136


namespace max_xy_l77_77221

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1) : xy <= 1 / 12 :=
by
  sorry

end max_xy_l77_77221


namespace find_y_l77_77124

variable {x y : ℤ}
variables (h1 : y = 2 * x - 3) (h2 : x + y = 57)

theorem find_y : y = 37 :=
by {
    sorry
}

end find_y_l77_77124


namespace cos_8_identity_l77_77352

theorem cos_8_identity (m : ℝ) (h : Real.sin 74 = m) : 
  Real.cos 8 = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_identity_l77_77352


namespace greatest_possible_difference_l77_77009

def is_reverse (q r : ℕ) : Prop :=
  let q_tens := q / 10
  let q_units := q % 10
  let r_tens := r / 10
  let r_units := r % 10
  (q_tens = r_units) ∧ (q_units = r_tens)

theorem greatest_possible_difference (q r : ℕ) (hq1 : q ≥ 10) (hq2 : q < 100)
  (hr1 : r ≥ 10) (hr2 : r < 100) (hrev : is_reverse q r) (hpos_diff : q - r < 30) :
  q - r ≤ 27 :=
by
  sorry

end greatest_possible_difference_l77_77009


namespace kimberly_peanuts_per_visit_l77_77170

theorem kimberly_peanuts_per_visit 
  (trips : ℕ) (total_peanuts : ℕ) 
  (h1 : trips = 3) 
  (h2 : total_peanuts = 21) : 
  total_peanuts / trips = 7 :=
by
  sorry

end kimberly_peanuts_per_visit_l77_77170


namespace custom_op_seven_three_l77_77739

def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b + 1

theorem custom_op_seven_three : custom_op 7 3 = 23 := by
  -- proof steps would go here
  sorry

end custom_op_seven_three_l77_77739


namespace maurice_late_467th_trip_l77_77605

-- Define the recurrence relation
def p (n : ℕ) : ℚ := 
  if n = 0 then 0
  else 1 / 4 * (p (n - 1) + 1)

-- Define the steady-state probability
def steady_state_p : ℚ := 1 / 3

-- Define L_n as the probability Maurice is late on the nth day
def L (n : ℕ) : ℚ := 1 - p n

-- The main goal (probability Maurice is late on his 467th trip)
theorem maurice_late_467th_trip :
  L 467 = 2 / 3 :=
sorry

end maurice_late_467th_trip_l77_77605


namespace radius_of_circle_l77_77079

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem radius_of_circle : 
  ∃ r : ℝ, circle_area r = circle_circumference r → r = 2 := 
by 
  sorry

end radius_of_circle_l77_77079


namespace age_of_first_man_replaced_l77_77723

theorem age_of_first_man_replaced (x : ℕ) (avg_before : ℝ) : avg_before * 15 + 30 = avg_before * 15 + 74 - (x + 23) → (37 * 2 - (x + 23) = 30) → x = 21 :=
sorry

end age_of_first_man_replaced_l77_77723


namespace problem_l77_77261

def x : ℕ := 660
def percentage_25_of_x : ℝ := 0.25 * x
def percentage_12_of_1500 : ℝ := 0.12 * 1500
def difference_of_percentages : ℝ := percentage_12_of_1500 - percentage_25_of_x

theorem problem : difference_of_percentages = 15 := by
  -- begin proof (content replaced by sorry)
  sorry

end problem_l77_77261


namespace projections_proportional_to_squares_l77_77464

theorem projections_proportional_to_squares
  (a b c a1 b1 : ℝ)
  (h₀ : c ≠ 0)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a1 = (a^2) / c)
  (h₃ : b1 = (b^2) / c) :
  (a1 / b1) = (a^2 / b^2) :=
by sorry

end projections_proportional_to_squares_l77_77464


namespace remainder_division_l77_77998
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end remainder_division_l77_77998


namespace students_not_made_the_cut_l77_77634

-- Define the constants for the number of girls, boys, and students called back
def girls := 17
def boys := 32
def called_back := 10

-- Total number of students trying out for the team
def total_try_out := girls + boys

-- Number of students who didn't make the cut
def not_made_the_cut := total_try_out - called_back

-- The theorem to be proved
theorem students_not_made_the_cut : not_made_the_cut = 39 := by
  -- Adding the proof is not required, so we use sorry
  sorry

end students_not_made_the_cut_l77_77634


namespace Anton_thought_number_is_729_l77_77523

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l77_77523


namespace option_A_cannot_be_true_l77_77056

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (r : ℝ) -- common ratio for the geometric sequence
variable (n : ℕ) -- number of terms

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem option_A_cannot_be_true
  (h_geom : is_geometric_sequence a r)
  (h_sum : sum_of_geometric_sequence a S) :
  a 2016 * (S 2016 - S 2015) ≠ 0 :=
sorry

end option_A_cannot_be_true_l77_77056


namespace peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l77_77623

-- Define the conditions
variable (a b c : ℕ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)

-- Part 1
theorem peter_can_transfer_all_money_into_two_accounts :
  ∃ x y, (x + y = a + b + c ∧ y = 0) ∨
          (∃ z, (a + b + c = x + y + z ∧ y = 0 ∧ z = 0)) :=
  sorry

-- Part 2
theorem peter_cannot_always_transfer_all_money_into_one_account :
  ((a + b + c) % 2 = 1 → ¬ ∃ x, x = a + b + c) :=
  sorry

end peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l77_77623


namespace length_of_common_internal_tangent_l77_77434

-- Define the conditions
def circles_centers_distance : ℝ := 50
def radius_smaller_circle : ℝ := 7
def radius_larger_circle : ℝ := 10

-- Define the statement to be proven
theorem length_of_common_internal_tangent :
  let d := circles_centers_distance
  let r₁ := radius_smaller_circle
  let r₂ := radius_larger_circle
  ∃ (length_tangent : ℝ), length_tangent = Real.sqrt (d^2 - (r₁ + r₂)^2) := by
  -- Provide the correct answer based on the conditions
  sorry

end length_of_common_internal_tangent_l77_77434


namespace souvenir_cost_l77_77149

def total_souvenirs : ℕ := 1000
def total_cost : ℝ := 220
def unknown_souvenirs : ℕ := 400
def known_cost : ℝ := 0.20

theorem souvenir_cost :
  ∃ x : ℝ, x = 0.25 ∧ total_cost = unknown_souvenirs * x + (total_souvenirs - unknown_souvenirs) * known_cost :=
by
  sorry

end souvenir_cost_l77_77149


namespace cylinder_surface_area_l77_77990

theorem cylinder_surface_area (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 2 * r) : 
  2 * Real.pi * r^2 + 2 * Real.pi * r * l = 24 * Real.pi :=
by
  subst h1
  subst h2
  sorry

end cylinder_surface_area_l77_77990


namespace range_of_m_l77_77663

-- Definitions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x^2 - 4*x + 4 - m^2) ≤ 0

-- Theorem Statement
theorem range_of_m (m : ℝ) (h_m : m > 0) : 
  (¬(∃ x, ¬p x) → ¬(∃ x, ¬q x m)) → m ≥ 8 := 
sorry -- Proof not required

end range_of_m_l77_77663


namespace worker_late_time_l77_77928

noncomputable def usual_time : ℕ := 60
noncomputable def speed_factor : ℚ := 4 / 5

theorem worker_late_time (T T_new : ℕ) (S : ℚ) :
  T = usual_time →
  T = 60 →
  T_new = (5 / 4) * T →
  T_new - T = 15 :=
by
  intros
  subst T
  sorry

end worker_late_time_l77_77928


namespace mass_percentage_of_C_in_benzene_l77_77327

theorem mass_percentage_of_C_in_benzene :
  let C_molar_mass := 12.01 -- g/mol
  let H_molar_mass := 1.008 -- g/mol
  let benzene_C_atoms := 6
  let benzene_H_atoms := 6
  let C_total_mass := benzene_C_atoms * C_molar_mass
  let H_total_mass := benzene_H_atoms * H_molar_mass
  let benzene_total_mass := C_total_mass + H_total_mass
  let mass_percentage_C := (C_total_mass / benzene_total_mass) * 100
  (mass_percentage_C = 92.26) :=
by
  sorry

end mass_percentage_of_C_in_benzene_l77_77327


namespace apple_distribution_ways_l77_77396

-- Definitions based on conditions
def distribute_apples (a b c : ℕ) : Prop := a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

-- Non-negative integer solutions to a' + b' + c' = 21
def num_solutions := Nat.choose 23 2

-- Theorem to prove
theorem apple_distribution_ways : distribute_apples 10 10 10 → num_solutions = 253 :=
by
  intros
  sorry

end apple_distribution_ways_l77_77396


namespace prime_divisors_difference_l77_77248

def prime_factors (n : ℕ) : ℕ := sorry -- definition placeholder

theorem prime_divisors_difference (n : ℕ) (hn : 0 < n) : 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ k - m = n ∧ prime_factors k - prime_factors m = 1 := 
sorry

end prime_divisors_difference_l77_77248


namespace find_triples_l77_77641

theorem find_triples (x y p : ℤ) (prime_p : Prime p) :
  x^2 - 3 * x * y + p^2 * y^2 = 12 * p ↔ 
  (p = 3 ∧ ( (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) ∨ (x = 4 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -4 ∧ y = -2) ) ) := 
by
  sorry

end find_triples_l77_77641


namespace intersection_P_Q_range_a_l77_77143

def set_P : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
def set_Q (a : ℝ) : Set ℝ := { x | (x - a) * (x - a - 1) ≤ 0 }

theorem intersection_P_Q (a : ℝ) (h_a : a = 1) :
  set_P ∩ set_Q 1 = {1} :=
sorry

theorem range_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set_P → x ∈ set_Q a) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end intersection_P_Q_range_a_l77_77143


namespace point_to_polar_coordinates_l77_77600

noncomputable def convert_to_polar_coordinates (x y : ℝ) (r θ : ℝ) : Prop :=
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x)

theorem point_to_polar_coordinates :
  convert_to_polar_coordinates 8 (2 * Real.sqrt 6) 
    (2 * Real.sqrt 22) (Real.arctan (Real.sqrt 6 / 4)) :=
sorry

end point_to_polar_coordinates_l77_77600


namespace value_of_x_l77_77042

theorem value_of_x (x : ℝ) (h : 2 ≤ |x - 3| ∧ |x - 3| ≤ 6) : x ∈ Set.Icc (-3 : ℝ) 1 ∪ Set.Icc 5 9 :=
by
  sorry

end value_of_x_l77_77042


namespace shanille_probability_l77_77833

-- Defining the probability function according to the problem's conditions.
def hit_probability (n k : ℕ) : ℚ :=
  if n = 100 ∧ k = 50 then 1 / 99 else 0

-- Prove that the probability Shanille hits exactly 50 of her first 100 shots is 1/99.
theorem shanille_probability :
  hit_probability 100 50 = 1 / 99 :=
by
  -- proof omitted
  sorry

end shanille_probability_l77_77833


namespace pythagorean_triple_345_l77_77533

theorem pythagorean_triple_345 : (3^2 + 4^2 = 5^2) := 
by 
  -- Here, the proof will be filled in, but we use 'sorry' for now.
  sorry

end pythagorean_triple_345_l77_77533


namespace tunnel_digging_duration_l77_77954

theorem tunnel_digging_duration (daily_progress : ℕ) (total_length_km : ℕ) 
    (meters_per_km : ℕ) (days_per_year : ℕ) : 
    daily_progress = 5 → total_length_km = 2 → meters_per_km = 1000 → days_per_year = 365 → 
    total_length_km * meters_per_km / daily_progress > 365 :=
by
  intros hprog htunnel hmeters hdays
  /- ... proof steps will go here -/
  sorry

end tunnel_digging_duration_l77_77954


namespace largest_multiple_of_7_negation_gt_neg150_l77_77024

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l77_77024


namespace gcd_1020_multiple_38962_l77_77516

-- Define that x is a multiple of 38962
def multiple_of (x n : ℤ) : Prop := ∃ k : ℤ, x = k * n

-- The main theorem statement
theorem gcd_1020_multiple_38962 (x : ℤ) (h : multiple_of x 38962) : Int.gcd 1020 x = 6 := 
sorry

end gcd_1020_multiple_38962_l77_77516


namespace maximum_withdraw_l77_77559

theorem maximum_withdraw (initial_amount withdraw deposit : ℕ) (h_initial : initial_amount = 500)
    (h_withdraw : withdraw = 300) (h_deposit : deposit = 198) :
    ∃ x y : ℕ, initial_amount - x * withdraw + y * deposit ≥ 0 ∧ initial_amount - x * withdraw + y * deposit = 194 ∧ initial_amount - x * withdraw = 300 := sorry

end maximum_withdraw_l77_77559


namespace rectangle_perimeter_l77_77371

theorem rectangle_perimeter 
  (w : ℝ) (l : ℝ) (hw : w = Real.sqrt 3) (hl : l = Real.sqrt 6) : 
  2 * (w + l) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := 
by 
  sorry

end rectangle_perimeter_l77_77371


namespace factorize_expression_l77_77823

variable (x y : ℝ)

theorem factorize_expression : 
  (y - 2 * x * y + x^2 * y) = y * (1 - x)^2 := 
by
  sorry

end factorize_expression_l77_77823


namespace polygon_sides_l77_77031

-- Definitions of the conditions
def is_regular_polygon (n : ℕ) (int_angle ext_angle : ℝ) : Prop :=
  int_angle = 5 * ext_angle ∧ (int_angle + ext_angle = 180)

-- Main theorem statement
theorem polygon_sides (n : ℕ) (int_angle ext_angle : ℝ) :
  is_regular_polygon n int_angle ext_angle →
  (ext_angle = 360 / n) →
  n = 12 :=
sorry

end polygon_sides_l77_77031


namespace initial_percentage_of_chemical_x_l77_77920

theorem initial_percentage_of_chemical_x (P : ℝ) (h1 : 20 + 80 * P = 44) : P = 0.3 :=
by sorry

end initial_percentage_of_chemical_x_l77_77920


namespace prime_factors_1260_l77_77619

theorem prime_factors_1260 (w x y z : ℕ) (h : 2 ^ w * 3 ^ x * 5 ^ y * 7 ^ z = 1260) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
by sorry

end prime_factors_1260_l77_77619


namespace polygon_sides_eq_eight_l77_77401

theorem polygon_sides_eq_eight (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 :=
by
  intro h
  sorry

end polygon_sides_eq_eight_l77_77401


namespace divisibility_theorem_l77_77941

theorem divisibility_theorem (n : ℕ) (h1 : n > 0) (h2 : ¬(2 ∣ n)) (h3 : ¬(3 ∣ n)) (k : ℤ) :
  (k + 1) ^ n - k ^ n - 1 ∣ k ^ 2 + k + 1 :=
sorry

end divisibility_theorem_l77_77941


namespace f_g_eq_g_f_iff_n_zero_l77_77017

def f (x n : ℝ) : ℝ := x + n
def g (x q : ℝ) : ℝ := x^2 + q

theorem f_g_eq_g_f_iff_n_zero (x n q : ℝ) : (f (g x q) n = g (f x n) q) ↔ n = 0 := by 
  sorry

end f_g_eq_g_f_iff_n_zero_l77_77017


namespace traveler_distance_l77_77336

theorem traveler_distance (a b c d : ℕ) (h1 : a = 24) (h2 : b = 15) (h3 : c = 10) (h4 : d = 9) :
  let net_ns := a - c
  let net_ew := b - d
  let distance := Real.sqrt ((net_ns ^ 2) + (net_ew ^ 2))
  distance = 2 * Real.sqrt 58 := 
by
  sorry

end traveler_distance_l77_77336


namespace woman_wait_time_for_man_to_catch_up_l77_77451

theorem woman_wait_time_for_man_to_catch_up :
  ∀ (mans_speed womans_speed : ℕ) (time_after_passing : ℕ) (distance_up_slope : ℕ) (incline_percentage : ℕ),
  mans_speed = 5 →
  womans_speed = 25 →
  time_after_passing = 5 →
  distance_up_slope = 1 →
  incline_percentage = 5 →
  max 0 (mans_speed - incline_percentage * 1) = 0 →
  time_after_passing = 0 :=
by
  intros
  -- Insert proof here when needed
  sorry

end woman_wait_time_for_man_to_catch_up_l77_77451


namespace has_minimum_value_iff_l77_77826

noncomputable def f (a x : ℝ) : ℝ :=
if x < a then -a * x + 4 else (x - 2) ^ 2

theorem has_minimum_value_iff (a : ℝ) : (∃ m, ∀ x, f a x ≥ m) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end has_minimum_value_iff_l77_77826


namespace min_volume_for_cone_l77_77500

noncomputable def min_cone_volume (V1 : ℝ) : Prop :=
  ∀ V2 : ℝ, (V1 = 1) → 
    V2 ≥ (4 / 3)

-- The statement without proof
theorem min_volume_for_cone : 
  min_cone_volume 1 :=
sorry

end min_volume_for_cone_l77_77500


namespace weight_of_B_l77_77849

theorem weight_of_B (A B C : ℝ) (h1 : (A + B + C) / 3 = 45) (h2 : (A + B) / 2 = 40) (h3 : (B + C) / 2 = 46) : B = 37 :=
by
  sorry

end weight_of_B_l77_77849


namespace factorial_ratio_l77_77230

theorem factorial_ratio : Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 5120 := by
  sorry

end factorial_ratio_l77_77230


namespace maximum_correct_answers_l77_77064

theorem maximum_correct_answers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : 5 * a - 2 * c = 150) : a ≤ 38 :=
by
  sorry

end maximum_correct_answers_l77_77064


namespace circle_radius_zero_l77_77754

theorem circle_radius_zero : ∀ (x y : ℝ), x^2 + 10 * x + y^2 - 4 * y + 29 = 0 → 0 = 0 :=
by intro x y h
   sorry

end circle_radius_zero_l77_77754


namespace divide_condition_l77_77089

theorem divide_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end divide_condition_l77_77089


namespace no_solution_fractional_eq_l77_77243

theorem no_solution_fractional_eq (y : ℝ) (h : y ≠ 3) : 
  ¬ ( (y-2)/(y-3) = 2 - 1/(3-y) ) :=
by
  sorry

end no_solution_fractional_eq_l77_77243


namespace peaches_per_basket_l77_77471

-- Given conditions as definitions in Lean 4
def red_peaches : Nat := 7
def green_peaches : Nat := 3

-- The proof statement showing each basket contains 10 peaches in total.
theorem peaches_per_basket : red_peaches + green_peaches = 10 := by
  sorry

end peaches_per_basket_l77_77471


namespace greatest_integer_x_l77_77869

theorem greatest_integer_x (x : ℤ) (h : 7 - 3 * x + 2 > 23) : x ≤ -5 :=
by {
  sorry
}

end greatest_integer_x_l77_77869


namespace find_a_in_third_quadrant_l77_77955

theorem find_a_in_third_quadrant :
  ∃ a : ℝ, a < 0 ∧ 3 * a^2 + 4 * a^2 = 28 ∧ a = -2 :=
by
  sorry

end find_a_in_third_quadrant_l77_77955


namespace perpendicular_value_of_k_parallel_value_of_k_l77_77314

variables (a b : ℝ × ℝ) (k : ℝ)

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-3, 1)
def ka_plus_b (k : ℝ) : ℝ × ℝ := (2*k - 3, 3*k + 1)
def a_minus_3b : ℝ × ℝ := (11, 0)

theorem perpendicular_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  a - ka_plus_b k = a_minus_3b → k = (3 / 2) :=
sorry

theorem parallel_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  ∃ k, (ka_plus_b (-1/3)) = (-1/3 * 11, -1/3 * 0) ∧ k = -1 / 3 :=
sorry

end perpendicular_value_of_k_parallel_value_of_k_l77_77314


namespace largest_root_divisible_by_17_l77_77110

theorem largest_root_divisible_by_17 (a : ℝ) (h : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0) (root_large : ∀ x ∈ {b | Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0}, x ≤ a) :
  a^1788 % 17 = 0 ∧ a^1988 % 17 = 0 :=
by
  sorry

end largest_root_divisible_by_17_l77_77110


namespace total_population_l77_77316

-- Define the predicates for g, b, and s based on t
variables (g b t s : ℕ)

-- The conditions given in the problem
def condition1 : Prop := g = 4 * t
def condition2 : Prop := b = 6 * g
def condition3 : Prop := s = t / 2

-- The theorem stating the total population is equal to (59 * t) / 2
theorem total_population (g b t s : ℕ) (h1 : condition1 g t) (h2 : condition2 b g) (h3 : condition3 s t) :
  b + g + t + s = 59 * t / 2 :=
by sorry

end total_population_l77_77316


namespace factorization_correct_l77_77166

theorem factorization_correct (x : ℤ) :
  (3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2) =
  ((3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6)) :=
by sorry

end factorization_correct_l77_77166


namespace solve_inequality_system_l77_77829

theorem solve_inequality_system
  (x : ℝ)
  (h1 : 3 * (x - 1) < 5 * x + 11)
  (h2 : 2 * x > (9 - x) / 4) :
  x > 1 :=
sorry

end solve_inequality_system_l77_77829


namespace characterize_functional_equation_l77_77435

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

theorem characterize_functional_equation (f : ℝ → ℝ) (h : satisfies_condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end characterize_functional_equation_l77_77435


namespace nat_gt_10_is_diff_of_hypotenuse_numbers_l77_77551

def is_hypotenuse_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem nat_gt_10_is_diff_of_hypotenuse_numbers (n : ℕ) (h : n > 10) : 
  ∃ (n₁ n₂ : ℕ), is_hypotenuse_number n₁ ∧ is_hypotenuse_number n₂ ∧ n = n₁ - n₂ :=
by
  sorry

end nat_gt_10_is_diff_of_hypotenuse_numbers_l77_77551


namespace form_x2_sub_2y2_l77_77511

theorem form_x2_sub_2y2 (x y : ℤ) (hx : x % 2 = 1) : (x^2 - 2*y^2) % 8 = 1 ∨ (x^2 - 2*y^2) % 8 = -1 := 
sorry

end form_x2_sub_2y2_l77_77511


namespace rectangle_invalid_perimeter_l77_77087

-- Define conditions
def positive_integer (n : ℕ) : Prop := n > 0

-- Define the rectangle with given area
def area_24 (length width : ℕ) : Prop := length * width = 24

-- Define the function to calculate perimeter for given length and width
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem to prove
theorem rectangle_invalid_perimeter (length width : ℕ) (h₁ : positive_integer length) (h₂ : positive_integer width) (h₃ : area_24 length width) : 
  (perimeter length width) ≠ 36 :=
sorry

end rectangle_invalid_perimeter_l77_77087


namespace seating_arrangements_exactly_two_adjacent_empty_seats_l77_77376

theorem seating_arrangements_exactly_two_adjacent_empty_seats : 
  (∃ (arrangements : ℕ), arrangements = 72) :=
by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_seats_l77_77376


namespace cube_root_solutions_l77_77505

theorem cube_root_solutions (p : ℕ) (hp : p > 3) :
    (∃ (k : ℤ) (h1 : k^2 ≡ -3 [ZMOD p]), ∀ x, x^3 ≡ 1 [ZMOD p] → 
        (x = 1 ∨ (x^2 + x + 1 ≡ 0 [ZMOD p])) )
    ∨ 
    (∀ x, x^3 ≡ 1 [ZMOD p] → x = 1) := 
sorry

end cube_root_solutions_l77_77505


namespace maximum_b_n_T_l77_77405

/-- Given a sequence {a_n} defined recursively and b_n = a_n / n.
   We need to prove that for all n in positive natural numbers,
   b_n is greater than or equal to T, and the maximum such T is 3. -/
theorem maximum_b_n_T (T : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  (∀ n, n ≥ 1 → b n = a n / n) →
  (∀ n, n ≥ 1 → b n ≥ T) →
  T ≤ 3 :=
by
  sorry

end maximum_b_n_T_l77_77405


namespace train_cross_bridge_time_l77_77519

def train_length : ℕ := 170
def train_speed_kmph : ℕ := 45
def bridge_length : ℕ := 205

def total_distance : ℕ := train_length + bridge_length
def train_speed_mps : ℕ := (train_speed_kmph * 1000) / 3600

theorem train_cross_bridge_time : (total_distance / train_speed_mps) = 30 := 
sorry

end train_cross_bridge_time_l77_77519


namespace proof_problem_l77_77062

variable {ι : Type} [LinearOrderedField ι]

-- Let A be a family of sets indexed by natural numbers
variables {A : ℕ → Set ι}

-- Hypotheses
def condition1 (A : ℕ → Set ι) : Prop :=
  (⋃ i, A i) = Set.univ

def condition2 (A : ℕ → Set ι) (a : ι) : Prop :=
  ∀ i b c, b > c → b - c ≥ a ^ i → b ∈ A i → c ∈ A i

theorem proof_problem (A : ℕ → Set ι) (a : ι) :
  condition1 A → condition2 A a → 0 < a → a < 2 :=
sorry

end proof_problem_l77_77062


namespace roger_shelves_l77_77354

theorem roger_shelves (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : 
  total_books = 24 → 
  books_taken = 3 → 
  books_per_shelf = 4 → 
  Nat.ceil ((total_books - books_taken) / books_per_shelf) = 6 :=
by
  intros h_total h_taken h_per_shelf
  rw [h_total, h_taken, h_per_shelf]
  sorry

end roger_shelves_l77_77354


namespace no_solution_ineq_l77_77973

theorem no_solution_ineq (m : ℝ) :
  (¬ ∃ (x : ℝ), x - 1 > 1 ∧ x < m) → m ≤ 2 :=
by
  sorry

end no_solution_ineq_l77_77973


namespace c_share_of_profit_l77_77596

-- Definitions for the investments and total profit
def investments_a := 800
def investments_b := 1000
def investments_c := 1200
def total_profit := 1000

-- Definition for the share of profits based on the ratio of investments
def share_of_c : ℕ :=
  let ratio_a := 4
  let ratio_b := 5
  let ratio_c := 6
  let total_ratio := ratio_a + ratio_b + ratio_c
  (ratio_c * total_profit) / total_ratio

-- The theorem to be proved
theorem c_share_of_profit : share_of_c = 400 := by
  sorry

end c_share_of_profit_l77_77596


namespace correct_exponentiation_operation_l77_77129

theorem correct_exponentiation_operation (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_exponentiation_operation_l77_77129


namespace number_of_black_cats_l77_77970

-- Definitions of the conditions.
def white_cats : Nat := 2
def gray_cats : Nat := 3
def total_cats : Nat := 15

-- The theorem we want to prove.
theorem number_of_black_cats : ∃ B : Nat, B = total_cats - (white_cats + gray_cats) ∧ B = 10 := by
  -- Proof will go here.
  sorry

end number_of_black_cats_l77_77970


namespace possible_to_form_square_l77_77909

noncomputable def shape : Type := sorry
noncomputable def is_square (s : shape) : Prop := sorry
noncomputable def divide_into_parts (s : shape) (n : ℕ) : Prop := sorry
noncomputable def all_triangles (s : shape) : Prop := sorry

theorem possible_to_form_square (s : shape) :
  (∃ (parts : ℕ), parts ≤ 4 ∧ divide_into_parts s parts ∧ is_square s) ∧
  (∃ (parts : ℕ), parts ≤ 5 ∧ divide_into_parts s parts ∧ all_triangles s ∧ is_square s) :=
sorry

end possible_to_form_square_l77_77909


namespace exponent_division_l77_77687

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l77_77687


namespace length_of_room_l77_77111

theorem length_of_room (width : ℝ) (cost_per_sq_meter : ℝ) (total_cost : ℝ) (L : ℝ) 
  (h_width : width = 2.75)
  (h_cost_per_sq_meter : cost_per_sq_meter = 600)
  (h_total_cost : total_cost = 10725)
  (h_area_cost_eq : total_cost = L * width * cost_per_sq_meter) : 
  L = 6.5 :=
by 
  simp [h_width, h_cost_per_sq_meter, h_total_cost, h_area_cost_eq] at *
  sorry

end length_of_room_l77_77111


namespace ratio_proof_l77_77481

theorem ratio_proof (a b c : ℝ) (ha : b / a = 3) (hb : c / b = 4) :
    (a + 2 * b) / (b + 2 * c) = 7 / 27 := by
  sorry

end ratio_proof_l77_77481


namespace triangle_perimeter_l77_77714

theorem triangle_perimeter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) (angle_A : ℝ)
  (h1 : AB = 4) (h2 : AC = 4) (h3 : angle_A = 60) : 
  AB + AC + AB = 12 :=
by {
  sorry
}

end triangle_perimeter_l77_77714


namespace jack_cleaning_time_is_one_hour_l77_77882

def jackGrove : ℕ × ℕ := (4, 5)
def timeToCleanEachTree : ℕ := 6
def timeReductionFactor : ℕ := 2
def totalCleaningTimeWithHelpMin : ℕ :=
  (jackGrove.fst * jackGrove.snd) * (timeToCleanEachTree / timeReductionFactor)
def totalCleaningTimeWithHelpHours : ℕ :=
  totalCleaningTimeWithHelpMin / 60

theorem jack_cleaning_time_is_one_hour :
  totalCleaningTimeWithHelpHours = 1 := by
  sorry

end jack_cleaning_time_is_one_hour_l77_77882


namespace solve_for_b_l77_77820

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l77_77820


namespace distinct_digits_sum_base7_l77_77076

theorem distinct_digits_sum_base7
    (A B C : ℕ)
    (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
    (h_nonzero : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
    (h_base7 : A < 7 ∧ B < 7 ∧ C < 7)
    (h_sum_eq : ((7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B)) = (7^3 * A + 7^2 * A + 7 * A)) :
    B + C = 6 :=
by {
    sorry
}

end distinct_digits_sum_base7_l77_77076


namespace find_f_2n_l77_77741

variable (f : ℤ → ℤ)
variable (n : ℕ)

axiom axiom1 {x y : ℤ} : f (x + y) = f x + f y + 2 * x * y + 1
axiom axiom2 : f (-2) = 1

theorem find_f_2n (n : ℕ) (h : n > 0) : f (2 * n) = 4 * n^2 + 2 * n - 1 := sorry

end find_f_2n_l77_77741


namespace garden_contains_53_33_percent_tulips_l77_77229

theorem garden_contains_53_33_percent_tulips :
  (∃ (flowers : ℕ) (yellow tulips flowers_in_garden : ℕ) (yellow_flowers blue_flowers yellow_tulips blue_tulips : ℕ),
    flowers_in_garden = yellow_flowers + blue_flowers ∧
    yellow_flowers = 4 * flowers / 5 ∧
    blue_flowers = 1 * flowers / 5 ∧
    yellow_tulips = yellow_flowers / 2 ∧
    blue_tulips = 2 * blue_flowers / 3 ∧
    (yellow_tulips + blue_tulips) = 8 * flowers / 15) →
    0.5333 ∈ ([46.67, 53.33, 60, 75, 80] : List ℝ) := sorry

end garden_contains_53_33_percent_tulips_l77_77229


namespace longest_side_similar_triangle_l77_77718

theorem longest_side_similar_triangle 
  (a b c : ℕ) (p : ℕ) (longest_side : ℕ)
  (h1 : a = 6) (h2 : b = 7) (h3 : c = 9) (h4 : p = 110) 
  (h5 : longest_side = 45) :
  ∃ x : ℕ, (6 * x + 7 * x + 9 * x = 110) ∧ (9 * x = longest_side) :=
by
  sorry

end longest_side_similar_triangle_l77_77718


namespace calculate_order_cost_l77_77054

-- Defining the variables and given conditions
variables (C E S D W : ℝ)

-- Given conditions as assumptions
axiom h1 : (2 / 5) * C = E * S
axiom h2 : (1 / 4) * (3 / 5) * C = D * W

-- Theorem statement for the amount paid for the orders
theorem calculate_order_cost (C E S D W : ℝ) (h1 : (2 / 5) * C = E * S) (h2 : (1 / 4) * (3 / 5) * C = D * W) : 
  (9 / 20) * C = C - ((2 / 5) * C + (3 / 20) * C) :=
sorry

end calculate_order_cost_l77_77054


namespace intervals_increasing_max_min_value_range_of_m_l77_77082

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem intervals_increasing : ∀ (x : ℝ), ∃ k : ℤ, -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π := sorry

theorem max_min_value (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  (f (π/3) = 0) ∧ (f (π/2) = -1/2) :=
  sorry

theorem range_of_m (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  ∀ m : ℝ, (∀ y : ℝ, (π/4 ≤ y ∧ y ≤ π/2) → |f y - m| < 1) ↔ (-1 < m ∧ m < 1/2) :=
  sorry

end intervals_increasing_max_min_value_range_of_m_l77_77082


namespace unique_positive_integer_k_for_rational_solutions_l77_77386

theorem unique_positive_integer_k_for_rational_solutions :
  ∃ (k : ℕ), (k > 0) ∧ (∀ (x : ℤ), x * x = 256 - 4 * k * k → x = 8) ∧ (k = 7) :=
by
  sorry

end unique_positive_integer_k_for_rational_solutions_l77_77386


namespace map_representation_l77_77175

-- Defining the conditions
noncomputable def map_scale : ℝ := 28 -- 1 inch represents 28 miles

-- Defining the specific instance provided in the problem
def inches_represented : ℝ := 13.7
def miles_represented : ℝ := 383.6

-- Statement of the problem
theorem map_representation (D : ℝ) : (D / map_scale) = (D : ℝ) / 28 := 
by
  -- Prove the statement
  sorry

end map_representation_l77_77175


namespace tourist_groupings_l77_77302

-- Assume a function to count valid groupings exists
noncomputable def num_groupings (guides tourists : ℕ) :=
  if tourists < guides * 2 then 0 
  else sorry -- placeholder for the actual combinatorial function

theorem tourist_groupings : num_groupings 4 8 = 105 := 
by
  -- The proof is omitted intentionally 
  sorry

end tourist_groupings_l77_77302


namespace singleBase12Digit_l77_77846

theorem singleBase12Digit (n : ℕ) : 
  (7 ^ 6 ^ 5 ^ 3 ^ 2 ^ 1) % 11 = 4 :=
sorry

end singleBase12Digit_l77_77846


namespace div_by_eleven_l77_77280

theorem div_by_eleven (a b : ℤ) (h : (a^2 + 9 * a * b + b^2) % 11 = 0) : 
  (a^2 - b^2) % 11 = 0 :=
sorry

end div_by_eleven_l77_77280


namespace ceil_minus_floor_eq_one_imp_ceil_minus_x_l77_77493

variable {x : ℝ}

theorem ceil_minus_floor_eq_one_imp_ceil_minus_x (H : ⌈x⌉ - ⌊x⌋ = 1) : ∃ (n : ℤ) (f : ℝ), (x = n + f) ∧ (0 < f) ∧ (f < 1) ∧ (⌈x⌉ - x = 1 - f) := sorry

end ceil_minus_floor_eq_one_imp_ceil_minus_x_l77_77493


namespace necessary_condition_for_ellipse_l77_77140

theorem necessary_condition_for_ellipse (m : ℝ) : 
  (5 - m > 0) → (m + 3 > 0) → (5 - m ≠ m + 3) → (-3 < m ∧ m < 5 ∧ m ≠ 1) :=
by sorry

end necessary_condition_for_ellipse_l77_77140


namespace fish_catch_l77_77323

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end fish_catch_l77_77323


namespace smallest_n_exists_unique_k_l77_77116

/- The smallest positive integer n for which there exists
   a unique integer k such that 9/16 < n / (n + k) < 7/12 is n = 1. -/

theorem smallest_n_exists_unique_k :
  ∃! (n : ℕ), n > 0 ∧ (∃! (k : ℤ), (9 : ℚ)/16 < (n : ℤ)/(n + k) ∧ (n : ℤ)/(n + k) < (7 : ℚ)/12) :=
sorry

end smallest_n_exists_unique_k_l77_77116


namespace find_unknown_value_l77_77807

theorem find_unknown_value (x : ℝ) (h : (3 + 5 + 6 + 8 + x) / 5 = 7) : x = 13 :=
by
  sorry

end find_unknown_value_l77_77807


namespace domain_of_function_l77_77199

open Real

theorem domain_of_function : 
  ∀ x, 
    (x + 1 ≠ 0) ∧ 
    (-x^2 - 3 * x + 4 > 0) ↔ 
    (-4 < x ∧ x < -1) ∨ ( -1 < x ∧ x < 1) := 
by 
  sorry

end domain_of_function_l77_77199


namespace fraction_always_defined_l77_77498

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 := 
by
  -- proof is not required
  sorry

end fraction_always_defined_l77_77498


namespace non_congruent_triangles_count_l77_77122

-- Let there be 15 equally spaced points on a circle,
-- and considering triangles formed by connecting 3 of these points.
def num_non_congruent_triangles (n : Nat) : Nat :=
  (if n = 15 then 19 else 0)

theorem non_congruent_triangles_count :
  num_non_congruent_triangles 15 = 19 :=
by
  sorry

end non_congruent_triangles_count_l77_77122


namespace new_person_weight_is_75_l77_77783

noncomputable def new_person_weight (previous_person_weight: ℝ) (average_increase: ℝ) (total_people: ℕ): ℝ :=
  previous_person_weight + total_people * average_increase

theorem new_person_weight_is_75 :
  new_person_weight 55 2.5 8 = 75 := 
by
  sorry

end new_person_weight_is_75_l77_77783


namespace probability_selecting_A_l77_77486

theorem probability_selecting_A :
  let total_people := 4
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_people
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_A_l77_77486


namespace p_and_q_necessary_not_sufficient_l77_77332

variable (a m x : ℝ) (P Q : Prop)

def p (a m : ℝ) : Prop := a < 0 ∧ m^2 - 4 * a * m + 3 * a^2 < 0

def q (m : ℝ) : Prop := ∀ x > 0, x + 4 / x ≥ 1 - m

theorem p_and_q_necessary_not_sufficient :
  (∀ (a m : ℝ), p a m → q m) ∧ (∀ a : ℝ, -1 ≤ a ∧ a < 0) :=
sorry

end p_and_q_necessary_not_sufficient_l77_77332


namespace cost_of_pencils_and_notebooks_l77_77806

variable (P N : ℝ)

theorem cost_of_pencils_and_notebooks
  (h1 : 4 * P + 3 * N = 9600)
  (h2 : 2 * P + 2 * N = 5400) :
  8 * P + 7 * N = 20400 := by
  sorry

end cost_of_pencils_and_notebooks_l77_77806


namespace difference_of_numbers_l77_77372

theorem difference_of_numbers (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 :=
by
  sorry

end difference_of_numbers_l77_77372


namespace spaces_per_row_l77_77585

theorem spaces_per_row 
  (kind_of_tomatoes : ℕ)
  (tomatoes_per_kind : ℕ)
  (kind_of_cucumbers : ℕ)
  (cucumbers_per_kind : ℕ)
  (potatoes : ℕ)
  (rows : ℕ)
  (additional_spaces : ℕ)
  (h1 : kind_of_tomatoes = 3)
  (h2 : tomatoes_per_kind = 5)
  (h3 : kind_of_cucumbers = 5)
  (h4 : cucumbers_per_kind = 4)
  (h5 : potatoes = 30)
  (h6 : rows = 10)
  (h7 : additional_spaces = 85) :
  (kind_of_tomatoes * tomatoes_per_kind + kind_of_cucumbers * cucumbers_per_kind + potatoes + additional_spaces) / rows = 15 :=
by
  sorry

end spaces_per_row_l77_77585


namespace sufficient_but_not_necessary_l77_77338

def l1 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => m * x + (m + 1) * y + 2

def l2 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => (m + 1) * x + (m + 4) * y - 3

def perpendicular_slopes (m : ℝ) : Prop :=
  let slope_l1 := -m / (m + 1)
  let slope_l2 := -(m + 1) / (m + 4)
  slope_l1 * slope_l2 = -1

theorem sufficient_but_not_necessary (m : ℝ) : m = -2 → (∃ k, m = -k ∧ perpendicular_slopes k) :=
by
  sorry

end sufficient_but_not_necessary_l77_77338


namespace complement_of_M_l77_77392

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M :
  ∀ x, x ∈ U \ M ↔ x < -2 ∨ x > 2 :=
by
  sorry

end complement_of_M_l77_77392


namespace greatest_k_value_l77_77029

-- Define a type for triangle and medians intersecting at centroid
structure Triangle :=
(medianA : ℝ)
(medianB : ℝ)
(medianC : ℝ)
(angleA : ℝ)
(angleB : ℝ)
(angleC : ℝ)
(centroid : ℝ)

-- Define a function to determine if the internal angles formed by medians 
-- are greater than 30 degrees
def angle_greater_than_30 (θ : ℝ) : Prop :=
  θ > 30

-- A proof statement that given a triangle and its medians dividing an angle
-- into six angles, the greatest possible number of these angles greater than 30° is 3.
theorem greatest_k_value (T : Triangle) : ∃ k : ℕ, k = 3 ∧ 
  (∀ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ, 
    (angle_greater_than_30 θ₁ ∨ angle_greater_than_30 θ₂ ∨ angle_greater_than_30 θ₃ ∨ 
     angle_greater_than_30 θ₄ ∨ angle_greater_than_30 θ₅ ∨ angle_greater_than_30 θ₆) → 
    k = 3) := 
sorry

end greatest_k_value_l77_77029


namespace unique_solution_l77_77558

theorem unique_solution (a b c : ℝ) (hb : b ≠ 2) (hc : c ≠ 0) : 
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by
  sorry

end unique_solution_l77_77558


namespace part1_part2_l77_77507

-- Define the conditions for p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) <= 0
def q (x m : ℝ) : Prop := (2 - m <= x) ∧ (x <= 2 + m)

-- Proof statement for part (1)
theorem part1 (m: ℝ) : 
  (∀ x : ℝ, p x → q x m) → 4 <= m :=
sorry

-- Proof statement for part (2)
theorem part2 (x : ℝ) (m : ℝ) : 
  (m = 5) → (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → x ∈ Set.Ico (-3) (-2) ∪ Set.Ioc 6 7 :=
sorry

end part1_part2_l77_77507


namespace concatenated_number_not_power_of_two_l77_77950

theorem concatenated_number_not_power_of_two :
  ∀ (N : ℕ), (∀ i, 11111 ≤ i ∧ i ≤ 99999) →
  (N ≡ 0 [MOD 11111]) → ¬ ∃ k, N = 2^k :=
by
  sorry

end concatenated_number_not_power_of_two_l77_77950


namespace population_growth_l77_77382

theorem population_growth (scale_factor1 scale_factor2 : ℝ)
    (h1 : scale_factor1 = 1.2)
    (h2 : scale_factor2 = 1.26) :
    (scale_factor1 * scale_factor2) - 1 = 0.512 :=
by
  sorry

end population_growth_l77_77382


namespace Shane_current_age_44_l77_77065

-- Declaring the known conditions and definitions
variable (Garret_present_age : ℕ) (Shane_past_age : ℕ) (Shane_present_age : ℕ)
variable (h1 : Garret_present_age = 12)
variable (h2 : Shane_past_age = 2 * Garret_present_age)
variable (h3 : Shane_present_age = Shane_past_age + 20)

theorem Shane_current_age_44 : Shane_present_age = 44 :=
by
  -- Proof to be filled here
  sorry

end Shane_current_age_44_l77_77065


namespace discriminant_zero_l77_77913

theorem discriminant_zero (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -2) (h₃ : c = 1) :
  (b^2 - 4 * a * c) = 0 :=
by
  sorry

end discriminant_zero_l77_77913


namespace winning_lottery_ticket_is_random_l77_77293

-- Definitions of the events
inductive Event
| certain : Event
| impossible : Event
| random : Event

open Event

-- Conditions
def boiling_water_event : Event := certain
def lottery_ticket_event : Event := random
def athlete_running_30mps_event : Event := impossible
def draw_red_ball_event : Event := impossible

-- Problem Statement
theorem winning_lottery_ticket_is_random : 
    lottery_ticket_event = random :=
sorry

end winning_lottery_ticket_is_random_l77_77293


namespace sum_of_coefficients_3x_minus_1_pow_7_l77_77462

theorem sum_of_coefficients_3x_minus_1_pow_7 :
  let f (x : ℕ) := (3 * x - 1) ^ 7
  (f 1) = 128 :=
by
  sorry

end sum_of_coefficients_3x_minus_1_pow_7_l77_77462


namespace molecular_weight_one_mole_l77_77765

theorem molecular_weight_one_mole (mw_three_moles : ℕ) (h : mw_three_moles = 882) : mw_three_moles / 3 = 294 :=
by
  -- proof is omitted
  sorry

end molecular_weight_one_mole_l77_77765


namespace remainder_divisible_by_4_l77_77194

theorem remainder_divisible_by_4 (z : ℕ) (h : z % 4 = 0) : ((z * (2 + 4 + z) + 3) % 2) = 1 :=
by
  sorry

end remainder_divisible_by_4_l77_77194


namespace division_result_l77_77588

def numerator : ℕ := 3 * 4 * 5
def denominator : ℕ := 2 * 3
def quotient : ℕ := numerator / denominator

theorem division_result : quotient = 10 := by
  sorry

end division_result_l77_77588


namespace system_solutions_l77_77365

theorem system_solutions (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = -1) : 
  b = -22 :=
by 
  sorry

end system_solutions_l77_77365


namespace solve_fractional_equation_l77_77235

theorem solve_fractional_equation : 
  ∃ x : ℝ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := 
sorry

end solve_fractional_equation_l77_77235


namespace hours_per_day_initial_l77_77475

-- Definition of the problem and conditions
def initial_men : ℕ := 75
def depth1 : ℕ := 50
def additional_men : ℕ := 65
def total_men : ℕ := initial_men + additional_men
def depth2 : ℕ := 70
def hours_per_day2 : ℕ := 6
def work1 (H : ℝ) := initial_men * H * depth1
def work2 := total_men * hours_per_day2 * depth2

-- Statement to prove
theorem hours_per_day_initial (H : ℝ) (h1 : work1 H = work2) : H = 15.68 :=
by
  sorry

end hours_per_day_initial_l77_77475


namespace halfway_fraction_between_is_one_fourth_l77_77578

theorem halfway_fraction_between_is_one_fourth : 
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  ((f1 + f2 + f3) / 3) = (1 / 4) := 
by
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  sorry

end halfway_fraction_between_is_one_fourth_l77_77578


namespace kyoko_payment_l77_77939

noncomputable def total_cost (balls skipropes frisbees : ℕ) (ball_cost rope_cost frisbee_cost : ℝ) : ℝ :=
  (balls * ball_cost) + (skipropes * rope_cost) + (frisbees * frisbee_cost)

noncomputable def final_amount (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (discount_rate * total_cost)

theorem kyoko_payment :
  let balls := 3
  let skipropes := 2
  let frisbees := 4
  let ball_cost := 1.54
  let rope_cost := 3.78
  let frisbee_cost := 2.63
  let discount_rate := 0.07
  final_amount (total_cost balls skipropes frisbees ball_cost rope_cost frisbee_cost) discount_rate = 21.11 :=
by
  sorry

end kyoko_payment_l77_77939


namespace triangle_angle_sum_l77_77276

theorem triangle_angle_sum (A : ℕ) (h1 : A = 55) (h2 : ∀ (B : ℕ), B = 2 * A) : (A + 2 * A = 165) :=
by
  sorry

end triangle_angle_sum_l77_77276


namespace vacation_days_l77_77249

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def towels_per_day_per_person : ℕ := 1
def washer_capacity : ℕ := 14
def num_loads : ℕ := 6

def total_people : ℕ := num_families * people_per_family
def towels_per_day : ℕ := total_people * towels_per_day_per_person
def total_towels : ℕ := num_loads * washer_capacity

def days_at_vacation_rental := total_towels / towels_per_day

theorem vacation_days : days_at_vacation_rental = 7 := by
  sorry

end vacation_days_l77_77249


namespace initial_students_began_contest_l77_77538

theorem initial_students_began_contest
  (n : ℕ)
  (first_round_fraction : ℚ)
  (second_round_fraction : ℚ)
  (remaining_students : ℕ) :
  first_round_fraction * second_round_fraction * n = remaining_students →
  remaining_students = 18 →
  first_round_fraction = 0.3 →
  second_round_fraction = 0.5 →
  n = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_students_began_contest_l77_77538


namespace coordinates_of_Q_l77_77301

theorem coordinates_of_Q (m : ℤ) (P Q : ℤ × ℤ) (hP : P = (m + 2, 2 * m + 4))
  (hQ_move : Q = (P.1, P.2 + 2)) (hQ_x_axis : Q.2 = 0) : Q = (-1, 0) :=
sorry

end coordinates_of_Q_l77_77301


namespace solve_quadratic_eq_l77_77724

theorem solve_quadratic_eq (a : ℝ) (x : ℝ) 
  (h : a ∈ ({-1, 1, a^2} : Set ℝ)) : 
  (x^2 - (1 - a) * x - 2 = 0) → (x = 2 ∨ x = -1) := by
  sorry

end solve_quadratic_eq_l77_77724


namespace length_of_platform_l77_77672

noncomputable def train_length : ℝ := 300
noncomputable def time_to_cross_platform : ℝ := 39
noncomputable def time_to_cross_pole : ℝ := 9

theorem length_of_platform : ∃ P : ℝ, P = 1000 :=
by
  let train_speed := train_length / time_to_cross_pole
  let total_distance_cross_platform := train_length + 1000
  let platform_length := total_distance_cross_platform - train_length
  existsi platform_length
  sorry

end length_of_platform_l77_77672


namespace find_x_for_divisibility_18_l77_77664

theorem find_x_for_divisibility_18 (x : ℕ) (h_digits : x < 10) :
  (1001 * x + 150) % 18 = 0 ↔ x = 6 :=
by
  sorry

end find_x_for_divisibility_18_l77_77664


namespace company_workers_count_l77_77489

-- Definitions
def num_supervisors := 13
def team_leads_per_supervisor := 3
def workers_per_team_lead := 10

-- Hypothesis
def team_leads := num_supervisors * team_leads_per_supervisor
def workers := team_leads * workers_per_team_lead

-- Theorem to prove
theorem company_workers_count : workers = 390 :=
by
  sorry

end company_workers_count_l77_77489


namespace sum_a1_a5_l77_77871

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1)
  (ha : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 5 = 11 :=
sorry

end sum_a1_a5_l77_77871


namespace determinant_modified_l77_77614

variable (a b c d : ℝ)

theorem determinant_modified (h : a * d - b * c = 10) :
  (a + 2 * c) * d - (b + 3 * d) * c = 10 - c * d := by
  sorry

end determinant_modified_l77_77614


namespace range_of_a_l77_77524

theorem range_of_a (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_inc : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_ineq : f (a - 3) < f 4) : -1 < a ∧ a < 7 :=
by
  sorry

end range_of_a_l77_77524


namespace initial_pencils_count_l77_77211

-- Define the conditions
def students : ℕ := 25
def pencils_per_student : ℕ := 5

-- Statement of the proof problem
theorem initial_pencils_count : students * pencils_per_student = 125 :=
by
  sorry

end initial_pencils_count_l77_77211


namespace original_bales_l77_77326

/-
There were some bales of hay in the barn. Jason stacked 23 bales in the barn today.
There are now 96 bales of hay in the barn. Prove that the original number of bales of hay 
in the barn was 73.
-/

theorem original_bales (stacked : ℕ) (total : ℕ) (original : ℕ) 
  (h1 : stacked = 23) (h2 : total = 96) : original = 73 :=
by
  sorry

end original_bales_l77_77326


namespace smallest_number_with_sum_32_and_distinct_digits_l77_77868

def is_distinct (l : List Nat) : Prop := l.Nodup

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

theorem smallest_number_with_sum_32_and_distinct_digits :
  ∀ n : Nat, is_distinct (Nat.digits 10 n) ∧ sum_of_digits n = 32 → 26789 ≤ n :=
by
  intro n
  intro h
  sorry

end smallest_number_with_sum_32_and_distinct_digits_l77_77868


namespace compute_ratio_l77_77683

variable {p q r u v w : ℝ}

theorem compute_ratio
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (h1 : p^2 + q^2 + r^2 = 49) 
  (h2 : u^2 + v^2 + w^2 = 64) 
  (h3 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 := 
sorry

end compute_ratio_l77_77683


namespace total_minutes_of_game_and_ceremony_l77_77675

-- Define the components of the problem
def game_hours : ℕ := 2
def game_additional_minutes : ℕ := 35
def ceremony_minutes : ℕ := 25

-- Prove the total minutes is 180
theorem total_minutes_of_game_and_ceremony (h: game_hours = 2) (ga: game_additional_minutes = 35) (c: ceremony_minutes = 25) :
  (game_hours * 60 + game_additional_minutes + ceremony_minutes) = 180 :=
  sorry

end total_minutes_of_game_and_ceremony_l77_77675


namespace solution_set_of_fraction_inequality_l77_77497

theorem solution_set_of_fraction_inequality (a b x : ℝ) (h1: ∀ x, ax - b > 0 ↔ x ∈ Set.Iio 1) (h2: a < 0) (h3: a - b = 0) :
  ∀ x, (a * x + b) / (x - 2) > 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 2 := 
sorry

end solution_set_of_fraction_inequality_l77_77497


namespace undefined_expression_l77_77620

theorem undefined_expression (y : ℝ) : (y^2 - 16 * y + 64 = 0) ↔ (y = 8) := by
  sorry

end undefined_expression_l77_77620


namespace tan_A_in_right_triangle_l77_77141

theorem tan_A_in_right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (angle_A angle_B angle_C : ℝ) 
  (sin_B : ℚ) (tan_A : ℚ) :
  angle_C = 90 ∧ sin_B = 3 / 5 → tan_A = 4 / 3 := by
  sorry

end tan_A_in_right_triangle_l77_77141


namespace certain_number_is_36_75_l77_77063

theorem certain_number_is_36_75 (A B C X : ℝ) (h_ratio_A : A = 5 * (C / 8)) (h_ratio_B : B = 6 * (C / 8)) (h_C : C = 42) (h_relation : A + C = B + X) :
  X = 36.75 :=
by
  sorry

end certain_number_is_36_75_l77_77063


namespace two_a_minus_b_equals_four_l77_77349

theorem two_a_minus_b_equals_four (a b : ℕ) 
    (consec_integers : b = a + 1)
    (min_a : min (Real.sqrt 30) a = a)
    (min_b : min (Real.sqrt 30) b = Real.sqrt 30) : 
    2 * a - b = 4 := 
sorry

end two_a_minus_b_equals_four_l77_77349


namespace count_dna_sequences_Rthea_l77_77035

-- Definition of bases
inductive Base | H | M | N | T

-- Function to check whether two bases can be adjacent on the same strand
def can_be_adjacent (x y : Base) : Prop :=
  match x, y with
  | Base.H, Base.M => False
  | Base.M, Base.H => False
  | Base.N, Base.T => False
  | Base.T, Base.N => False
  | _, _ => True

-- Function to count the number of valid sequences
noncomputable def count_valid_sequences : Nat := 12 * 7^4

-- Theorem stating the expected count of valid sequences
theorem count_dna_sequences_Rthea : count_valid_sequences = 28812 := by
  sorry

end count_dna_sequences_Rthea_l77_77035


namespace students_at_end_l77_77628

def initial_students : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0

theorem students_at_end : initial_students - students_left - students_transferred = 28.0 :=
by
  -- Proof omitted
  sorry

end students_at_end_l77_77628


namespace fruits_in_good_condition_percentage_l77_77993

theorem fruits_in_good_condition_percentage (total_oranges total_bananas rotten_oranges_percentage rotten_bananas_percentage : ℝ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percentage = 0.15) 
  (h4 : rotten_bananas_percentage = 0.08) : 
  (1 - ((rotten_oranges_percentage * total_oranges + rotten_bananas_percentage * total_bananas) / (total_oranges + total_bananas))) * 100 = 87.8 :=
by 
  sorry

end fruits_in_good_condition_percentage_l77_77993


namespace green_ball_probability_l77_77814

/-
  There are four containers:
  - Container A holds 5 red balls and 7 green balls.
  - Container B holds 7 red balls and 3 green balls.
  - Container C holds 8 red balls and 2 green balls.
  - Container D holds 4 red balls and 6 green balls.
  The probability of choosing containers A, B, C, and D is 1/4 each.
-/

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 1 / 4
def prob_D : ℚ := 1 / 4

def prob_Given_A : ℚ := 7 / 12
def prob_Given_B : ℚ := 3 / 10
def prob_Given_C : ℚ := 1 / 5
def prob_Given_D : ℚ := 3 / 5

def total_prob_green : ℚ :=
  prob_A * prob_Given_A + prob_B * prob_Given_B +
  prob_C * prob_Given_C + prob_D * prob_Given_D

theorem green_ball_probability : total_prob_green = 101 / 240 := 
by
  -- here would normally be the proof steps, but we use sorry to skip it.
  sorry

end green_ball_probability_l77_77814


namespace roots_negative_reciprocal_l77_77113

theorem roots_negative_reciprocal (a b c : ℝ) (α β : ℝ) (h_eq : a * α ^ 2 + b * α + c = 0)
  (h_roots : α * β = -1) : c = -a :=
sorry

end roots_negative_reciprocal_l77_77113


namespace LCM_activities_l77_77317

theorem LCM_activities :
  ∃ (d : ℕ), d = Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) ∧ d = 48 :=
by
  sorry

end LCM_activities_l77_77317


namespace min_value_of_expression_l77_77263

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) :
  2 * a + b ≥ 4 * Real.sqrt 2 - 3 := 
sorry

end min_value_of_expression_l77_77263


namespace flowmaster_pump_output_l77_77098

theorem flowmaster_pump_output (hourly_rate : ℕ) (time_minutes : ℕ) (output_gallons : ℕ) 
  (h1 : hourly_rate = 600) 
  (h2 : time_minutes = 30) 
  (h3 : output_gallons = (hourly_rate * time_minutes) / 60) : 
  output_gallons = 300 :=
by sorry

end flowmaster_pump_output_l77_77098


namespace joann_third_day_lollipops_l77_77490

theorem joann_third_day_lollipops
  (a b c d e : ℕ)
  (h1 : b = a + 6)
  (h2 : c = b + 6)
  (h3 : d = c + 6)
  (h4 : e = d + 6)
  (h5 : a + b + c + d + e = 100) :
  c = 20 :=
by
  sorry

end joann_third_day_lollipops_l77_77490


namespace at_least_one_negative_l77_77040

-- Defining the circle partition and the properties given in the problem.
def circle_partition (a : Fin 7 → ℤ) : Prop :=
  ∃ (l1 l2 l3 : Finset (Fin 7)),
    l1.card = 4 ∧ l2.card = 4 ∧ l3.card = 4 ∧
    (∀ i ∈ l1, ∀ j ∉ l1, a i + a j = 0) ∧
    (∀ i ∈ l2, ∀ j ∉ l2, a i + a j = 0) ∧
    (∀ i ∈ l3, ∀ j ∉ l3, a i + a j = 0) ∧
    ∃ i, a i = 0

-- The main theorem to prove.
theorem at_least_one_negative : 
  ∀ (a : Fin 7 → ℤ), 
  circle_partition a → 
  ∃ i, a i < 0 :=
by
  sorry

end at_least_one_negative_l77_77040


namespace final_customer_boxes_l77_77527

theorem final_customer_boxes (f1 f2 f3 f4 goal left boxes_first : ℕ) 
  (h1 : boxes_first = 5) 
  (h2 : f2 = 4 * boxes_first) 
  (h3 : f3 = f2 / 2) 
  (h4 : f4 = 3 * f3)
  (h5 : goal = 150) 
  (h6 : left = 75) 
  (h7 : goal - left = f1 + f2 + f3 + f4) : 
  (goal - left - (f1 + f2 + f3 + f4) = 10) := 
sorry

end final_customer_boxes_l77_77527


namespace range_of_a_l77_77825

variable (a : ℝ)
def f (x : ℝ) := x^2 + 2 * (a - 1) * x + 2
def f_deriv (x : ℝ) := 2 * x + 2 * (a - 1)

theorem range_of_a (h : ∀ x ≥ -4, f_deriv a x ≥ 0) : a ≥ 5 :=
sorry

end range_of_a_l77_77825


namespace complex_div_l77_77473

theorem complex_div (i : ℂ) (hi : i^2 = -1) : (1 + i) / i = 1 - i := by
  sorry

end complex_div_l77_77473


namespace minimum_value_expression_l77_77411

theorem minimum_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  (x^2 + 6 * x * y + 9 * y^2 + 3/2 * z^2) ≥ 102 :=
sorry

end minimum_value_expression_l77_77411


namespace total_candies_count_l77_77311

variable (purple_candies orange_candies yellow_candies : ℕ)

theorem total_candies_count
  (ratio_condition : purple_candies / orange_candies = 2 / 4 ∧ purple_candies / yellow_candies = 2 / 5)
  (yellow_candies_count : yellow_candies = 40) :
  purple_candies + orange_candies + yellow_candies = 88 :=
by
  sorry

end total_candies_count_l77_77311


namespace problem_l77_77896

theorem problem (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y + x * y = 3) :
  (0 < x * y ∧ x * y ≤ 1) ∧ (∀ z : ℝ, z = x + 2 * y → z = 4 * Real.sqrt 2 - 3) :=
by
  sorry

end problem_l77_77896


namespace angle_B_side_b_l77_77429

variable (A B C a b c : ℝ)
variable (S : ℝ := 5 * Real.sqrt 3)

-- Conditions
variable (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
variable (h2 : 1/2 * a * c * Real.sin B = S)
variable (h3 : a = 5)

-- The two parts to prove
theorem angle_B (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B) : 
  B = π / 3 := 
  sorry

theorem side_b (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
  (h2 : 1/2 * a * c * Real.sin B = S) (h3 : a = 5) : 
  b = Real.sqrt 21 := 
  sorry

end angle_B_side_b_l77_77429


namespace inequality_proof_l77_77057

variables (x y : ℝ) (n : ℕ)

theorem inequality_proof (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3) + y^n / (x^3 + y)) ≥ (2^(4-n) / 5) := by
  sorry

end inequality_proof_l77_77057


namespace max_expr_value_l77_77713

theorem max_expr_value (a b c d : ℝ) (h_a : -8.5 ≤ a ∧ a ≤ 8.5)
                       (h_b : -8.5 ≤ b ∧ b ≤ 8.5)
                       (h_c : -8.5 ≤ c ∧ c ≤ 8.5)
                       (h_d : -8.5 ≤ d ∧ d ≤ 8.5) :
                       a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 306 :=
sorry

end max_expr_value_l77_77713


namespace min_value_x2y2z2_l77_77576

open Real

noncomputable def condition (x y z : ℝ) : Prop := (1 / x + 1 / y + 1 / z = 3)

theorem min_value_x2y2z2 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : condition x y z) :
  x^2 * y^2 * z^2 ≥ 1 / 64 :=
by
  sorry

end min_value_x2y2z2_l77_77576


namespace fraction_of_phones_l77_77387

-- The total number of valid 8-digit phone numbers (b)
def valid_phone_numbers_total : ℕ := 5 * 10^7

-- The number of valid phone numbers that begin with 5 and end with 2 (a)
def valid_phone_numbers_special : ℕ := 10^6

-- The fraction of phone numbers that begin with 5 and end with 2
def fraction_phone_numbers_special : ℚ := valid_phone_numbers_special / valid_phone_numbers_total

-- Prove that the fraction of such phone numbers is 1/50
theorem fraction_of_phones : fraction_phone_numbers_special = 1 / 50 := by
  sorry

end fraction_of_phones_l77_77387


namespace range_of_a_l77_77621

-- Definitions of the propositions in Lean terms
def proposition_p (a : ℝ) := 
  ∃ x : ℝ, x ∈ [-1, 1] ∧ x^2 - (2 + a) * x + 2 * a = 0

def proposition_q (a : ℝ) := 
  ∃ x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The main theorem to prove that the range of values for a is [-1, 0]
theorem range_of_a {a : ℝ} (h : proposition_p a ∧ proposition_q a) : 
  -1 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l77_77621


namespace evaluate_expression_l77_77224

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem evaluate_expression : (nabla (nabla 2 3) 4) = 16777219 :=
by sorry

end evaluate_expression_l77_77224


namespace inequality_f_l77_77050

-- Definitions of the given conditions
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

-- Theorem statement
theorem inequality_f (a b : ℝ) : 
  abs (f 1 a b) + 2 * abs (f 2 a b) + abs (f 3 a b) ≥ 2 :=
by sorry

end inequality_f_l77_77050


namespace part1_part2_l77_77795

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l77_77795


namespace height_of_room_is_twelve_l77_77777

-- Defining the dimensions of the room
def length : ℝ := 25
def width : ℝ := 15

-- Defining the dimensions of the door and windows
def door_area : ℝ := 6 * 3
def window_area : ℝ := 3 * (4 * 3)

-- Total cost of whitewashing
def total_cost : ℝ := 5436

-- Cost per square foot for whitewashing
def cost_per_sqft : ℝ := 6

-- The equation to solve for height
def height_equation (h : ℝ) : Prop :=
  cost_per_sqft * (2 * (length + width) * h - (door_area + window_area)) = total_cost

theorem height_of_room_is_twelve : ∃ h : ℝ, height_equation h ∧ h = 12 := by
  -- Proof would go here
  sorry

end height_of_room_is_twelve_l77_77777


namespace mean_temperature_correct_l77_77658

-- Define the list of temperatures
def temperatures : List ℤ := [-8, -5, -5, -6, 0, 4]

-- Define the mean temperature calculation
def mean_temperature (temps: List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

-- The theorem we want to prove
theorem mean_temperature_correct :
  mean_temperature temperatures = -10 / 3 :=
by
  sorry

end mean_temperature_correct_l77_77658


namespace travel_time_l77_77308

theorem travel_time (speed distance : ℕ) (h_speed : speed = 100) (h_distance : distance = 500) :
  distance / speed = 5 := by
  sorry

end travel_time_l77_77308


namespace thomas_payment_weeks_l77_77139

theorem thomas_payment_weeks 
    (weekly_rate : ℕ) 
    (total_amount_paid : ℕ) 
    (h1 : weekly_rate = 4550) 
    (h2 : total_amount_paid = 19500) :
    (19500 / 4550 : ℕ) = 4 :=
by {
  sorry
}

end thomas_payment_weeks_l77_77139


namespace range_of_m_l77_77284

theorem range_of_m (m : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ x - (m^2 - 2 * m + 4) * y + 6 > 0) →
  -1 < m ∧ m < 3 :=
by
  intros h
  rcases h with ⟨x, y, hx, hy, hineq⟩
  rw [hx, hy] at hineq
  sorry

end range_of_m_l77_77284


namespace quadratic_roots_condition_l77_77153

theorem quadratic_roots_condition (k : ℝ) : 
  ((∃ x : ℝ, (k - 1) * x^2 + 4 * x + 1 = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2) ↔ (k < 5 ∧ k ≠ 1) :=
by {
  sorry  
}

end quadratic_roots_condition_l77_77153


namespace condition_iff_inequality_l77_77771

theorem condition_iff_inequality (a b : ℝ) (h : a * b ≠ 0) : (0 < a ∧ 0 < b) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
by
  -- Proof goes here
  sorry 

end condition_iff_inequality_l77_77771


namespace quadratic_roots_range_l77_77952

theorem quadratic_roots_range (a : ℝ) :
  (a-1) * x^2 - 2*x + 1 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a-1) * x1^2 - 2*x1 + 1 = 0 ∧ (a-1) * x2^2 - 2*x2 + 1 = 0) → (a < 2 ∧ a ≠ 1) :=
sorry

end quadratic_roots_range_l77_77952


namespace max_gcd_bn_bnp1_l77_77668

def b_n (n : ℕ) : ℤ := (7 ^ n - 4) / 3
def b_n_plus_1 (n : ℕ) : ℤ := (7 ^ (n + 1) - 4) / 3

theorem max_gcd_bn_bnp1 (n : ℕ) : ∃ d_max : ℕ, (∀ d : ℕ, (gcd (b_n n) (b_n_plus_1 n) ≤ d) → d ≤ d_max) ∧ d_max = 3 :=
sorry

end max_gcd_bn_bnp1_l77_77668


namespace total_length_of_segments_in_new_figure_l77_77174

-- Defining the given conditions.
def left_side := 10
def top_side := 3
def right_side := 8
def segments_removed_from_bottom := [2, 1, 2] -- List of removed segments from the bottom.

-- This is the theorem statement that confirms the total length of the new figure's sides.
theorem total_length_of_segments_in_new_figure :
  (left_side + top_side + right_side) = 21 :=
by
  -- This is where the proof would be written.
  sorry

end total_length_of_segments_in_new_figure_l77_77174


namespace infinite_geometric_series_sum_l77_77688

theorem infinite_geometric_series_sum : 
  ∑' n : ℕ, (1 / 3) ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l77_77688


namespace sqrt_2x_plus_y_eq_4_l77_77430

theorem sqrt_2x_plus_y_eq_4 (x y : ℝ) 
  (h1 : (3 * x + 1) = 4) 
  (h2 : (2 * y - 1) = 27) : 
  Real.sqrt (2 * x + y) = 4 := 
by 
  sorry

end sqrt_2x_plus_y_eq_4_l77_77430


namespace sum_of_coefficients_l77_77935

theorem sum_of_coefficients (A B C : ℤ) 
  (h_factorization : ∀ x, x^3 + A * x^2 + B * x + C = (x + 2) * (x - 2) * (x - 1)) :
  A + B + C = -1 :=
by sorry

end sum_of_coefficients_l77_77935


namespace marla_errand_total_time_l77_77450

theorem marla_errand_total_time :
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  total_time = 110 :=
by
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  show total_time = 110
  sorry

end marla_errand_total_time_l77_77450
