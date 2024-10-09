import Mathlib

namespace sum_of_digits_x_squared_l1784_178448

theorem sum_of_digits_x_squared {r x p q : ℕ} (h_r : r ≤ 400) 
  (h_x_form : x = p * r^3 + p * r^2 + q * r + q) 
  (h_pq_condition : 7 * q = 17 * p) 
  (h_x2_form : ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + d * r^3 + c * r^2 + b * r + a ∧ d = 0) :
  p + p + q + q = 400 := 
sorry

end sum_of_digits_x_squared_l1784_178448


namespace find_larger_page_l1784_178451

theorem find_larger_page {x y : ℕ} (h1 : y = x + 1) (h2 : x + y = 125) : y = 63 :=
by
  sorry

end find_larger_page_l1784_178451


namespace calculate_train_length_l1784_178412

noncomputable def train_length (speed_kmph : ℕ) (time_secs : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * time_secs
  total_distance - bridge_length_m

theorem calculate_train_length :
  train_length 60 14.998800095992321 140 = 110 :=
by
  sorry

end calculate_train_length_l1784_178412


namespace find_functions_l1784_178493

variable (f : ℝ → ℝ)

theorem find_functions (h : ∀ x y : ℝ, f (x + f y) = f x + f y ^ 2 + 2 * x * f y) :
  ∃ c : ℝ, (∀ x, f x = x ^ 2 + c) ∨ (∀ x, f x = 0) :=
by
  sorry

end find_functions_l1784_178493


namespace length_of_goods_train_l1784_178431

theorem length_of_goods_train 
  (speed_kmph : ℝ) (platform_length : ℝ) (time_sec : ℝ) (train_length : ℝ) 
  (h1 : speed_kmph = 72)
  (h2 : platform_length = 270) 
  (h3 : time_sec = 26) 
  (h4 : train_length = (speed_kmph * 1000 / 3600 * time_sec) - platform_length)
  : train_length = 250 := 
  by
    sorry

end length_of_goods_train_l1784_178431


namespace smallest_n_for_congruence_l1784_178460

theorem smallest_n_for_congruence :
  ∃ n : ℕ, 827 * n % 36 = 1369 * n % 36 ∧ n > 0 ∧ (∀ m : ℕ, 827 * m % 36 = 1369 * m % 36 ∧ m > 0 → m ≥ 18) :=
by sorry

end smallest_n_for_congruence_l1784_178460


namespace smallest_number_remainder_l1784_178474

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l1784_178474


namespace number_of_whole_numbers_between_sqrt2_and_3e_is_7_l1784_178432

noncomputable def number_of_whole_numbers_between_sqrt2_and_3e : ℕ :=
  let sqrt2 : ℝ := Real.sqrt 2
  let e : ℝ := Real.exp 1
  let small_int := Nat.ceil sqrt2 -- This is 2
  let large_int := Nat.floor (3 * e) -- This is 8
  large_int - small_int + 1 -- The number of integers between small_int and large_int (inclusive)

theorem number_of_whole_numbers_between_sqrt2_and_3e_is_7 :
  number_of_whole_numbers_between_sqrt2_and_3e = 7 := by
  sorry

end number_of_whole_numbers_between_sqrt2_and_3e_is_7_l1784_178432


namespace kanul_initial_amount_l1784_178424

-- Definition based on the problem conditions
def spent_on_raw_materials : ℝ := 3000
def spent_on_machinery : ℝ := 2000
def spent_on_labor : ℝ := 1000
def percent_spent : ℝ := 0.15

-- Definition of the total amount initially had by Kanul
def total_amount_initial (X : ℝ) : Prop :=
  spent_on_raw_materials + spent_on_machinery + percent_spent * X + spent_on_labor = X

-- Theorem stating the conclusion based on the given conditions
theorem kanul_initial_amount : ∃ X : ℝ, total_amount_initial X ∧ X = 7058.82 :=
by {
  sorry
}

end kanul_initial_amount_l1784_178424


namespace a4_b4_c4_double_square_l1784_178429

theorem a4_b4_c4_double_square (a b c : ℤ) (h : a = b + c) : 
  a^4 + b^4 + c^4 = 2 * ((a^2 - b * c)^2) :=
by {
  sorry -- proof is not provided as per instructions
}

end a4_b4_c4_double_square_l1784_178429


namespace crossnumber_unique_solution_l1784_178430

-- Definition of two-digit numbers
def two_digit_numbers (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Definition of prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of square
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The given conditions reformulated
def crossnumber_problem : Prop :=
  ∃ (one_across one_down two_down three_across : ℕ),
    two_digit_numbers one_across ∧ is_prime one_across ∧
    two_digit_numbers one_down ∧ is_square one_down ∧
    two_digit_numbers two_down ∧ is_square two_down ∧
    two_digit_numbers three_across ∧ is_square three_across ∧
    one_across = 83 ∧ one_down = 81 ∧ two_down = 16 ∧ three_across = 16

theorem crossnumber_unique_solution : crossnumber_problem :=
by
  sorry

end crossnumber_unique_solution_l1784_178430


namespace evaluate_product_l1784_178490

theorem evaluate_product (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5 * n^3 + 4 * n^2 + 4 * n := 
by
  -- Omitted proof steps
  sorry

end evaluate_product_l1784_178490


namespace y_intercept_line_l1784_178476

theorem y_intercept_line : 
  ∃ m b : ℝ, 
  (2 * m + b = -3) ∧ 
  (6 * m + b = 5) ∧ 
  b = -7 :=
by 
  sorry

end y_intercept_line_l1784_178476


namespace jack_estimate_larger_l1784_178419

variable {x y a b : ℝ}

theorem jack_estimate_larger (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (ha : 0 < a) (hb : 0 < b) : 
  (x + a) - (y - b) > x - y :=
by
  sorry

end jack_estimate_larger_l1784_178419


namespace even_sine_function_phi_eq_pi_div_2_l1784_178484
open Real

theorem even_sine_function_phi_eq_pi_div_2 (φ : ℝ) (h : 0 ≤ φ ∧ φ ≤ π)
    (even_f : ∀ x : ℝ, sin (x + φ) = sin (-x + φ)) : φ = π / 2 :=
sorry

end even_sine_function_phi_eq_pi_div_2_l1784_178484


namespace sock_pairs_l1784_178403

def total_ways (n_white n_brown n_blue n_red : ℕ) : ℕ :=
  n_blue * n_white + n_blue * n_brown + n_blue * n_red

theorem sock_pairs (n_white n_brown n_blue n_red : ℕ) (h_white : n_white = 5) (h_brown : n_brown = 4) (h_blue : n_blue = 2) (h_red : n_red = 1) :
  total_ways n_white n_brown n_blue n_red = 20 := by
  -- insert the proof steps here
  sorry

end sock_pairs_l1784_178403


namespace length_in_scientific_notation_l1784_178455

theorem length_in_scientific_notation : (161000 : ℝ) = 1.61 * 10^5 := 
by 
  -- Placeholder proof
  sorry

end length_in_scientific_notation_l1784_178455


namespace no_tiling_triminos_l1784_178465

theorem no_tiling_triminos (board_size : ℕ) (trimino_size : ℕ) (remaining_squares : ℕ) 
  (H_board : board_size = 8) (H_trimino : trimino_size = 3) (H_remaining : remaining_squares = 63) : 
  ¬ ∃ (triminos : ℕ), triminos * trimino_size = remaining_squares :=
by {
  sorry
}

end no_tiling_triminos_l1784_178465


namespace find_angle_C_find_area_l1784_178428

open Real

-- Definition of the problem conditions and questions

-- Condition: Given a triangle and the trigonometric relationship
variables {A B C : ℝ} {a b c : ℝ}

-- Condition 1: Trigonometric identity provided in the problem
axiom trig_identity : (sqrt 3) * c / (cos C) = a / (cos (3 * π / 2 + A))

-- First part of the problem
theorem find_angle_C (h1 : sqrt 3 * c / cos C = a / cos (3 * π / 2 + A)) : C = π / 6 :=
sorry

-- Second part of the problem
noncomputable def area_of_triangle (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

variables {c' b' : ℝ}
-- Given conditions for the second question 
axiom condition_c_a : c' / a = 2
axiom condition_b : b' = 4 * sqrt 3

-- Definitions to align with the given problem
def c_from_a (a : ℝ) : ℝ := 2 * a

-- The final theorem for the second part
theorem find_area (hC : C = π / 6) (hc : c_from_a a = c') (hb : b' = 4 * sqrt 3) :
  area_of_triangle a b' C = 2 * sqrt 15 - 2 * sqrt 3 :=
sorry

end find_angle_C_find_area_l1784_178428


namespace binomial_expansion_l1784_178446

theorem binomial_expansion (a b : ℕ) (h_a : a = 34) (h_b : b = 5) :
  a^2 + 2*a*b + b^2 = 1521 :=
by
  rw [h_a, h_b]
  sorry

end binomial_expansion_l1784_178446


namespace integral_cosine_l1784_178496

noncomputable def a : ℝ := 2 * Real.pi / 3

theorem integral_cosine (ha : a = 2 * Real.pi / 3) :
  ∫ x in -a..a, Real.cos x = Real.sqrt 3 := 
sorry

end integral_cosine_l1784_178496


namespace second_polygon_sides_l1784_178499

theorem second_polygon_sides (a b n m : ℕ) (s : ℝ) 
  (h1 : a = 45) 
  (h2 : b = 3 * s)
  (h3 : n * b = m * s)
  (h4 : n = 45) : m = 135 := 
by
  sorry

end second_polygon_sides_l1784_178499


namespace max_value_of_x2_plus_y2_l1784_178492

theorem max_value_of_x2_plus_y2 {x y : ℝ} 
  (h1 : x ≥ 1)
  (h2 : y ≥ x)
  (h3 : x - 2 * y + 3 ≥ 0) : 
  x^2 + y^2 ≤ 18 :=
sorry

end max_value_of_x2_plus_y2_l1784_178492


namespace simplify_sqrt_l1784_178481

theorem simplify_sqrt (x : ℝ) (h : x < 2) : Real.sqrt (x^2 - 4*x + 4) = 2 - x :=
by
  sorry

end simplify_sqrt_l1784_178481


namespace chris_birthday_after_45_days_l1784_178435

theorem chris_birthday_after_45_days (k : ℕ) (h : k = 45) (tuesday : ℕ) (h_tuesday : tuesday = 2) : 
  (tuesday + k) % 7 = 5 := 
sorry

end chris_birthday_after_45_days_l1784_178435


namespace sequence_formula_l1784_178408

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = -2 ^ (n - 1) := 
by 
  sorry

end sequence_formula_l1784_178408


namespace molly_swam_28_meters_on_sunday_l1784_178449

def meters_swam_on_saturday : ℕ := 45
def total_meters_swum : ℕ := 73
def meters_swam_on_sunday := total_meters_swum - meters_swam_on_saturday

theorem molly_swam_28_meters_on_sunday : meters_swam_on_sunday = 28 :=
by
  -- sorry to skip the proof
  sorry

end molly_swam_28_meters_on_sunday_l1784_178449


namespace weight_of_A_l1784_178445

variable (A B C D E : ℕ)

axiom cond1 : A + B + C = 180
axiom cond2 : A + B + C + D = 260
axiom cond3 : E = D + 3
axiom cond4 : B + C + D + E = 256

theorem weight_of_A : A = 87 :=
by
  sorry

end weight_of_A_l1784_178445


namespace original_cost_of_article_l1784_178468

theorem original_cost_of_article : ∃ C : ℝ, 
  (∀ S : ℝ, S = 1.35 * C) ∧
  (∀ C_new : ℝ, C_new = 0.75 * C) ∧
  (∀ S_new : ℝ, (S_new = 1.35 * C - 25) ∧ (S_new = 1.0875 * C)) ∧
  (C = 95.24) :=
sorry

end original_cost_of_article_l1784_178468


namespace problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l1784_178406

-- Problem 1: Prove the remainder of the Euclidean division of \(9^{100}\) by 8 is 1.
theorem problem1_remainder_of_9_power_100_mod_8 :
  (9 ^ 100) % 8 = 1 :=
by
sorry

-- Problem 2: Prove the last digit of \(2012^{2012}\) is 6.
theorem problem2_last_digit_of_2012_power_2012 :
  (2012 ^ 2012) % 10 = 6 :=
by
sorry

end problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l1784_178406


namespace find_other_solution_l1784_178407

theorem find_other_solution (x₁ : ℚ) (x₂ : ℚ) 
  (h₁ : x₁ = 3 / 4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) 
  (eq : 72 * x₂^2 + 39 * x₂ - 18 = 0 ∧ x₂ ≠ x₁) : 
  x₂ = -31 / 6 := 
sorry

end find_other_solution_l1784_178407


namespace difference_between_two_numbers_l1784_178453

theorem difference_between_two_numbers :
  ∃ a b : ℕ, 
    a + 5 * b = 23405 ∧ 
    (∃ b' : ℕ, b = 10 * b' + 5 ∧ b' = 5 * a) ∧ 
    5 * b - a = 21600 :=
by {
  sorry
}

end difference_between_two_numbers_l1784_178453


namespace number_of_unique_combinations_l1784_178452

-- Define the inputs and the expected output.
def n := 8
def r := 3
def expected_combinations := 56

-- We state our theorem indicating that the combination of 8 toppings chosen 3 at a time
-- equals 56.
theorem number_of_unique_combinations :
  (Nat.choose n r = expected_combinations) :=
by
  sorry

end number_of_unique_combinations_l1784_178452


namespace find_ab_pairs_l1784_178418

open Set

-- Definitions
def f (a b x : ℝ) : ℝ := a * x + b

-- Main theorem
theorem find_ab_pairs (a b : ℝ) :
  (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → 
    f a b x * f a b y + f a b (x + y - x * y) ≤ 0) ↔ 
  (-1 ≤ b ∧ b ≤ 0 ∧ -(b + 1) ≤ a ∧ a ≤ -b) :=
by sorry

end find_ab_pairs_l1784_178418


namespace units_digit_base7_of_multiplied_numbers_l1784_178420

-- Define the numbers in base 10
def num1 : ℕ := 325
def num2 : ℕ := 67

-- Define the modulus used for base 7
def base : ℕ := 7

-- Function to determine the units digit of the base-7 representation
def units_digit_base7 (n : ℕ) : ℕ := n % base

-- Prove that units_digit_base7 (num1 * num2) = 5
theorem units_digit_base7_of_multiplied_numbers :
  units_digit_base7 (num1 * num2) = 5 :=
by
  sorry

end units_digit_base7_of_multiplied_numbers_l1784_178420


namespace game_spinner_probability_l1784_178425

theorem game_spinner_probability (P_A P_B P_D P_C : ℚ) (h₁ : P_A = 1/4) (h₂ : P_B = 1/3) (h₃ : P_D = 1/6) (h₄ : P_A + P_B + P_C + P_D = 1) :
  P_C = 1/4 :=
by
  sorry

end game_spinner_probability_l1784_178425


namespace common_difference_l1784_178400

noncomputable def a : ℕ := 3
noncomputable def an : ℕ := 28
noncomputable def Sn : ℕ := 186

theorem common_difference (d : ℚ) (n : ℕ) (h1 : an = a + (n-1) * d) (h2 : Sn = n * (a + an) / 2) : d = 25 / 11 :=
sorry

end common_difference_l1784_178400


namespace abes_present_age_l1784_178417

theorem abes_present_age :
  ∃ A : ℕ, A + (A - 7) = 27 ∧ A = 17 :=
by
  sorry

end abes_present_age_l1784_178417


namespace find_pairs_of_positive_integers_l1784_178469

theorem find_pairs_of_positive_integers (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  3 * 2^m + 1 = n^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) :=
sorry

end find_pairs_of_positive_integers_l1784_178469


namespace apples_after_operations_l1784_178454

-- Define the initial conditions
def initial_apples : ℕ := 38
def used_apples : ℕ := 20
def bought_apples : ℕ := 28

-- State the theorem we want to prove
theorem apples_after_operations : initial_apples - used_apples + bought_apples = 46 :=
by
  sorry

end apples_after_operations_l1784_178454


namespace intersection_of_A_and_B_l1784_178437

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l1784_178437


namespace bamboo_capacity_l1784_178433

theorem bamboo_capacity :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 d : ℚ),
    a_1 + a_2 + a_3 = 4 ∧
    a_6 + a_7 + a_8 + a_9 = 3 ∧
    a_2 = a_1 + d ∧
    a_3 = a_1 + 2*d ∧
    a_4 = a_1 + 3*d ∧
    a_5 = a_1 + 4*d ∧
    a_7 = a_1 + 5*d ∧
    a_8 = a_1 + 6*d ∧
    a_9 = a_1 + 7*d ∧
    a_4 = 1 + 8/66 ∧
    a_5 = 1 + 1/66 :=
sorry

end bamboo_capacity_l1784_178433


namespace square_area_inscribed_triangle_l1784_178411

-- Definitions from the conditions of the problem
variable (EG : ℝ) (hF : ℝ)

-- Since EG = 12 inches and the altitude from F to EG is 7 inches
theorem square_area_inscribed_triangle 
(EG_eq : EG = 12) 
(hF_eq : hF = 7) :
  ∃ (AB : ℝ), AB ^ 2 = 36 :=
by 
  sorry

end square_area_inscribed_triangle_l1784_178411


namespace extra_mangoes_l1784_178479

-- Definitions of the conditions
def original_price_per_mango := 433.33 / 130
def new_price_per_mango := original_price_per_mango - 0.10 * original_price_per_mango
def mangoes_at_original_price := 360 / original_price_per_mango
def mangoes_at_new_price := 360 / new_price_per_mango

-- Statement to be proved
theorem extra_mangoes : mangoes_at_new_price - mangoes_at_original_price = 12 := 
by {
  sorry
}

end extra_mangoes_l1784_178479


namespace ratio_of_fusilli_to_penne_l1784_178401

def number_of_students := 800
def preferred_pasta_types := ["penne", "tortellini", "fusilli", "spaghetti"]
def students_prefer_fusilli := 320
def students_prefer_penne := 160

theorem ratio_of_fusilli_to_penne : (students_prefer_fusilli / students_prefer_penne) = 2 := by
  -- Here we would provide the proof, but since it's a statement, we use sorry
  sorry

end ratio_of_fusilli_to_penne_l1784_178401


namespace gcd_problem_l1784_178495

variable (A B : ℕ)
variable (hA : A = 2 * 3 * 5)
variable (hB : B = 2 * 2 * 5 * 7)

theorem gcd_problem : Nat.gcd A B = 10 :=
by
  -- Proof is omitted.
  sorry

end gcd_problem_l1784_178495


namespace problem_statement_l1784_178486

/-- Definition of the function f that relates the input n with floor functions -/
def f (n : ℕ) : ℤ :=
  n + ⌊(n : ℤ) / 6⌋ - ⌊(n : ℤ) / 2⌋ - ⌊2 * (n : ℤ) / 3⌋

/-- Prove the main statement -/
theorem problem_statement (n : ℕ) (hpos : 0 < n) :
  f n = 0 ↔ ∃ k : ℕ, n = 6 * k + 1 :=
sorry -- Proof goes here.

end problem_statement_l1784_178486


namespace no_solution_for_p_eq_7_l1784_178441

theorem no_solution_for_p_eq_7 : ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ( (x-3)/(x-4) = (x-7)/(x-8) ) → false := by
  intro x h1 h2 h
  sorry

end no_solution_for_p_eq_7_l1784_178441


namespace find_t_l1784_178416

theorem find_t (t : ℝ) : 
  (∃ a b : ℝ, a^2 = t^2 ∧ b^2 = 5 * t ∧ (a - b = 2 * Real.sqrt 6 ∨ b - a = 2 * Real.sqrt 6)) → 
  (t = 2 ∨ t = 3 ∨ t = 6) := 
by
  sorry

end find_t_l1784_178416


namespace sum_of_areas_of_squares_l1784_178475

def is_right_angle (a b c : ℝ) : Prop := (a^2 + b^2 = c^2)

def isSquare (side : ℝ) : Prop := (side > 0)

def area_of_square (side : ℝ) : ℝ := side^2

theorem sum_of_areas_of_squares 
  (P Q R S X Y : ℝ) 
  (h1 : is_right_angle P Q R)
  (h2 : PR = 15)
  (h3 : isSquare PR)
  (h4 : isSquare PQ) :
  area_of_square PR + area_of_square PQ = 450 := 
sorry


end sum_of_areas_of_squares_l1784_178475


namespace expression_evaluation_l1784_178436

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := 
by sorry

end expression_evaluation_l1784_178436


namespace physical_fitness_test_l1784_178405

theorem physical_fitness_test (x : ℝ) (hx : x > 0) :
  (1000 / x - 1000 / (1.25 * x) = 30) :=
sorry

end physical_fitness_test_l1784_178405


namespace point_comparison_on_inverse_proportion_l1784_178494

theorem point_comparison_on_inverse_proportion :
  (∃ y1 y2, (y1 = 2 / 1) ∧ (y2 = 2 / 2) ∧ y1 > y2) :=
by
  use 2
  use 1
  sorry

end point_comparison_on_inverse_proportion_l1784_178494


namespace domain_h_l1784_178470

def domain_f : Set ℝ := Set.Icc (-12) 6
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3*x)

theorem domain_h {f : ℝ → ℝ} (hf : ∀ x, x ∈ domain_f → f x ∈ Set.univ) {x : ℝ} :
  h f x ∈ Set.univ ↔ x ∈ Set.Icc (-2) 4 :=
by
  sorry

end domain_h_l1784_178470


namespace min_value_inequality_l1784_178443

open Real

theorem min_value_inequality (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 47 :=
sorry

end min_value_inequality_l1784_178443


namespace barbi_monthly_loss_l1784_178497

variable (x : Real)

theorem barbi_monthly_loss : 
  (∃ x : Real, 12 * x = 99 - 81) → x = 1.5 :=
by
  sorry

end barbi_monthly_loss_l1784_178497


namespace area_of_triangle_BP_Q_is_24_l1784_178459

open Real

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem area_of_triangle_BP_Q_is_24
  (A B C P H Q : ℝ × ℝ)
  (h_triangle_ABC_right : C.1 = 0 ∧ C.2 = 0 ∧ B.2 = 0 ∧ A.2 ≠ 0)
  (h_BC_diameter : distance B C = 26)
  (h_tangent_AP : distance P B = distance P C ∧ P ≠ C)
  (h_PH_perpendicular_BC : P.1 = H.1 ∧ H.2 = 0)
  (h_PH_intersects_AB_at_Q : H.1 = Q.1 ∧ Q.2 ≠ 0)
  (h_BH_CH_ratio : 4 * distance B H = 9 * distance C H)
  : triangle_area B P Q = 24 :=
sorry

end area_of_triangle_BP_Q_is_24_l1784_178459


namespace tyler_saltwater_aquariums_l1784_178413

def num_animals_per_aquarium : ℕ := 39
def total_saltwater_animals : ℕ := 2184

theorem tyler_saltwater_aquariums : 
  total_saltwater_animals / num_animals_per_aquarium = 56 := 
by
  sorry

end tyler_saltwater_aquariums_l1784_178413


namespace bread_weight_eq_anton_weight_l1784_178456

-- Definitions of variables
variables (A B F X : ℝ)

-- Given conditions
axiom cond1 : X + F = A + B
axiom cond2 : B + X = A + F

-- Theorem to prove
theorem bread_weight_eq_anton_weight : X = A :=
by
  sorry

end bread_weight_eq_anton_weight_l1784_178456


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l1784_178461

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_13 : 
  ∃ p : ℕ, (p ∣ 16385 ∧ Nat.Prime p ∧ (∀ q : ℕ, q ∣ 16385 → Nat.Prime q → q ≤ p)) ∧ (Nat.digits 10 p).sum = 13 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l1784_178461


namespace negation_proof_l1784_178421

theorem negation_proof :
  (∃ x₀ : ℝ, x₀ < 2) → ¬ (∀ x : ℝ, x < 2) :=
by
  sorry

end negation_proof_l1784_178421


namespace solve_arithmetic_sequence_problem_l1784_178409

noncomputable def arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) : Prop :=
  (∀ n, a n = a 0 + n * (a 1 - a 0)) ∧  -- Condition: sequence is arithmetic
  (∀ n, S n = (n * (a 0 + a (n - 1))) / 2) ∧  -- Condition: sum of first n terms
  (m > 1) ∧  -- Condition: m > 1
  (a (m - 1) + a (m + 1) - a m ^ 2 = 0) ∧  -- Given condition
  (S (2 * m - 1) = 38)  -- Given that sum of first 2m-1 terms equals 38

-- The statement we need to prove
theorem solve_arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) :
  arithmetic_sequence_problem a S m → m = 10 :=
by
  sorry  -- Proof to be completed

end solve_arithmetic_sequence_problem_l1784_178409


namespace rice_in_first_5_days_l1784_178489

-- Define the arithmetic sequence for number of workers dispatched each day
def num_workers (n : ℕ) : ℕ := 64 + (n - 1) * 7

-- Function to compute the sum of the first n terms of the arithmetic sequence
def sum_workers (n : ℕ) : ℕ := n * 64 + (n * (n - 1)) / 2 * 7

-- Given the rice distribution conditions
def rice_per_worker : ℕ := 3

-- Given the problem specific conditions
def total_rice_distributed_first_5_days : ℕ := 
  rice_per_worker * (sum_workers 1 + sum_workers 2 + sum_workers 3 + sum_workers 4 + sum_workers 5)
  
-- Proof goal
theorem rice_in_first_5_days : total_rice_distributed_first_5_days = 3300 :=
  by
  sorry

end rice_in_first_5_days_l1784_178489


namespace multiple_of_9_digit_l1784_178415

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end multiple_of_9_digit_l1784_178415


namespace find_x_value_l1784_178426

theorem find_x_value (x : ℚ) (h : 5 * (x - 10) = 3 * (3 - 3 * x) + 9) : x = 34 / 7 := by
  sorry

end find_x_value_l1784_178426


namespace incorrect_statement_d_l1784_178478

theorem incorrect_statement_d :
  (¬(abs 2 = -2)) :=
by sorry

end incorrect_statement_d_l1784_178478


namespace muscovy_more_than_cayuga_l1784_178404

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end muscovy_more_than_cayuga_l1784_178404


namespace expression_evaluation_l1784_178498

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end expression_evaluation_l1784_178498


namespace correct_exponentiation_calculation_l1784_178482

theorem correct_exponentiation_calculation (a : ℝ) : a^2 * a^6 = a^8 :=
by sorry

end correct_exponentiation_calculation_l1784_178482


namespace clock_angle_at_330_l1784_178439

/--
At 3:00, the hour hand is at 90 degrees from the 12 o'clock position.
The minute hand at 3:30 is at 180 degrees from the 12 o'clock position.
The hour hand at 3:30 has moved an additional 15 degrees (0.5 degrees per minute).
Prove that the smaller angle formed by the hour and minute hands of a clock at 3:30 is 75.0 degrees.
-/
theorem clock_angle_at_330 : 
  let hour_pos_at_3 := 90
  let min_pos_at_330 := 180
  let hour_additional := 15
  (min_pos_at_330 - (hour_pos_at_3 + hour_additional) = 75)
  :=
  by
  sorry

end clock_angle_at_330_l1784_178439


namespace problem1_l1784_178427

theorem problem1 (a b : ℝ) : 
  ((-2 * a) ^ 3 * (- (a * b^2)) ^ 3 - 4 * a * b^2 * (2 * a^5 * b^4 + (1 / 2) * a * b^3 - 5)) / (-2 * a * b) = a * b^4 - 10 * b :=
sorry

end problem1_l1784_178427


namespace work_duration_l1784_178488

theorem work_duration (X_full_days : ℕ) (Y_full_days : ℕ) (Y_worked_days : ℕ) (R : ℚ) :
  X_full_days = 18 ∧ Y_full_days = 15 ∧ Y_worked_days = 5 ∧ R = (2 / 3) →
  (R / (1 / X_full_days)) = 12 :=
by
  intros h
  sorry

end work_duration_l1784_178488


namespace water_volume_in_B_when_A_is_0_point_4_l1784_178480

noncomputable def pool_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

noncomputable def valve_rate (volume time : ℝ) : ℝ :=
  volume / time

theorem water_volume_in_B_when_A_is_0_point_4 :
  ∀ (length width depth : ℝ)
    (time_A_fill time_A_to_B : ℝ)
    (depth_A_target : ℝ),
    length = 3 → width = 2 → depth = 1.2 →
    time_A_fill = 18 → time_A_to_B = 24 →
    depth_A_target = 0.4 →
    pool_volume length width depth = 7.2 →
    valve_rate 7.2 time_A_fill = 0.4 →
    valve_rate 7.2 time_A_to_B = 0.3 →
    ∃ (time_required : ℝ),
    time_required = 24 →
    (valve_rate 7.2 time_A_to_B * time_required = 7.2) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end water_volume_in_B_when_A_is_0_point_4_l1784_178480


namespace triangle_inequality_sqrt_sides_l1784_178477

theorem triangle_inequality_sqrt_sides {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b):
  (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) 
  ∧ (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_sqrt_sides_l1784_178477


namespace quadratic_real_roots_range_l1784_178438

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (m-1)*x^2 + x + 1 = 0) → (m ≤ 5/4 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_range_l1784_178438


namespace reflect_center_of_circle_l1784_178467

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

theorem reflect_center_of_circle :
  reflect_point (3, -7) = (7, -3) :=
by
  sorry

end reflect_center_of_circle_l1784_178467


namespace ellen_dinner_calories_proof_l1784_178473

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l1784_178473


namespace vector_parallel_x_value_l1784_178440

theorem vector_parallel_x_value :
  ∀ (x : ℝ), let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  (∃ k : ℝ, b = (k * 3, k * 1)) → x = -9 :=
by
  intro x
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  intro h
  sorry

end vector_parallel_x_value_l1784_178440


namespace candies_per_friend_l1784_178463

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l1784_178463


namespace white_surface_area_fraction_l1784_178434

theorem white_surface_area_fraction
    (total_cubes : ℕ)
    (white_cubes : ℕ)
    (red_cubes : ℕ)
    (edge_length : ℕ)
    (white_exposed_area : ℕ)
    (total_surface_area : ℕ)
    (fraction : ℚ)
    (h1 : total_cubes = 64)
    (h2 : white_cubes = 14)
    (h3 : red_cubes = 50)
    (h4 : edge_length = 4)
    (h5 : white_exposed_area = 6)
    (h6 : total_surface_area = 96)
    (h7 : fraction = 1 / 16)
    (h8 : white_cubes + red_cubes = total_cubes)
    (h9 : 6 * (edge_length * edge_length) = total_surface_area)
    (h10 : white_exposed_area / total_surface_area = fraction) :
    fraction = 1 / 16 := by
    sorry

end white_surface_area_fraction_l1784_178434


namespace exists_distinct_nonzero_ints_for_poly_factorization_l1784_178423

theorem exists_distinct_nonzero_ints_for_poly_factorization :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ P Q : Polynomial ℤ, (P * Q = Polynomial.X * (Polynomial.X - Polynomial.C a) * 
   (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) + 1) ∧ 
   P.leadingCoeff = 1 ∧ Q.leadingCoeff = 1) :=
by
  sorry

end exists_distinct_nonzero_ints_for_poly_factorization_l1784_178423


namespace noah_garden_larger_by_75_l1784_178471

-- Define the dimensions of Liam's garden
def length_liam : ℕ := 30
def width_liam : ℕ := 50

-- Define the dimensions of Noah's garden
def length_noah : ℕ := 35
def width_noah : ℕ := 45

-- Define the areas of the gardens
def area_liam : ℕ := length_liam * width_liam
def area_noah : ℕ := length_noah * width_noah

theorem noah_garden_larger_by_75 :
  area_noah - area_liam = 75 :=
by
  -- The proof goes here
  sorry

end noah_garden_larger_by_75_l1784_178471


namespace watermelon_vendor_profit_l1784_178491

theorem watermelon_vendor_profit 
  (purchase_price : ℝ) (selling_price_initial : ℝ) (initial_quantity_sold : ℝ) 
  (decrease_factor : ℝ) (additional_quantity_per_decrease : ℝ) (fixed_cost : ℝ) 
  (desired_profit : ℝ) 
  (x : ℝ)
  (h_purchase : purchase_price = 2)
  (h_selling_initial : selling_price_initial = 3)
  (h_initial_quantity : initial_quantity_sold = 200)
  (h_decrease_factor : decrease_factor = 0.1)
  (h_additional_quantity : additional_quantity_per_decrease = 40)
  (h_fixed_cost : fixed_cost = 24)
  (h_desired_profit : desired_profit = 200) :
  (x = 2.8 ∨ x = 2.7) ↔ 
  ((x - purchase_price) * (initial_quantity_sold + additional_quantity_per_decrease / decrease_factor * (selling_price_initial - x)) - fixed_cost = desired_profit) :=
by sorry

end watermelon_vendor_profit_l1784_178491


namespace perpendicular_vectors_solution_l1784_178472

theorem perpendicular_vectors_solution (m : ℝ) (a : ℝ × ℝ := (m-1, 2)) (b : ℝ × ℝ := (m, -3)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : m = 3 ∨ m = -2 :=
by sorry

end perpendicular_vectors_solution_l1784_178472


namespace bus_driver_compensation_l1784_178458

theorem bus_driver_compensation : 
  let regular_rate := 16
  let regular_hours := 40
  let total_hours_worked := 57
  let overtime_rate := regular_rate + (0.75 * regular_rate)
  let regular_pay := regular_hours * regular_rate
  let overtime_hours_worked := total_hours_worked - regular_hours
  let overtime_pay := overtime_hours_worked * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1116 :=
by
  sorry

end bus_driver_compensation_l1784_178458


namespace prob_A_and_B_succeed_prob_vaccine_A_successful_l1784_178464

-- Define the probabilities of success for Company A, Company B, and Company C
def P_A := (2 : ℚ) / 3
def P_B := (1 : ℚ) / 2
def P_C := (3 : ℚ) / 5

-- Define the theorem statements

-- Theorem for the probability that both Company A and Company B succeed
theorem prob_A_and_B_succeed : P_A * P_B = 1 / 3 := by
  sorry

-- Theorem for the probability that vaccine A is successfully developed
theorem prob_vaccine_A_successful : 1 - ((1 - P_A) * (1 - P_B)) = 5 / 6 := by
  sorry

end prob_A_and_B_succeed_prob_vaccine_A_successful_l1784_178464


namespace floor_e_eq_two_l1784_178444

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l1784_178444


namespace add_neg_two_and_three_l1784_178414

theorem add_neg_two_and_three : -2 + 3 = 1 :=
by
  sorry

end add_neg_two_and_three_l1784_178414


namespace lily_ducks_l1784_178422

variable (D G : ℕ)
variable (Rayden_ducks : ℕ := 3 * D)
variable (Rayden_geese : ℕ := 4 * G)
variable (Lily_geese : ℕ := 10) -- Given G = 10
variable (Rayden_extra : ℕ := 70) -- Given Rayden has 70 more ducks and geese

theorem lily_ducks (h : 3 * D + 4 * Lily_geese = D + Lily_geese + Rayden_extra) : D = 20 :=
by sorry

end lily_ducks_l1784_178422


namespace point_on_line_y_coordinate_l1784_178487

variables (m b x : ℝ)

def line_equation := m * x + b

theorem point_on_line_y_coordinate : m = 4 → b = 4 → x = 199 → line_equation m b x = 800 :=
by 
  intros h_m h_b h_x
  unfold line_equation
  rw [h_m, h_b, h_x]
  norm_num
  done

end point_on_line_y_coordinate_l1784_178487


namespace correct_multiplier_l1784_178485

theorem correct_multiplier
  (x : ℕ)
  (incorrect_multiplier : ℕ := 34)
  (difference : ℕ := 1215)
  (number_to_be_multiplied : ℕ := 135) :
  number_to_be_multiplied * x - number_to_be_multiplied * incorrect_multiplier = difference →
  x = 43 :=
  sorry

end correct_multiplier_l1784_178485


namespace reservoir_original_content_l1784_178457

noncomputable def original_content (T O : ℝ) : Prop :=
  (80 / 100) * T = O + 120 ∧
  O = (50 / 100) * T

theorem reservoir_original_content (T : ℝ) (h1 : (80 / 100) * T = (50 / 100) * T + 120) : 
  (50 / 100) * T = 200 :=
by
  sorry

end reservoir_original_content_l1784_178457


namespace nth_term_correct_l1784_178466

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end nth_term_correct_l1784_178466


namespace loan_balance_formula_l1784_178410

variable (c V : ℝ) (t n : ℝ)

theorem loan_balance_formula :
  V = c / (1 + t)^(3 * n) →
  n = (Real.log (c / V)) / (3 * Real.log (1 + t)) :=
by sorry

end loan_balance_formula_l1784_178410


namespace factors_are_divisors_l1784_178462

theorem factors_are_divisors (a b c d : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : d = 5) : 
  a ∣ 30 ∧ b ∣ 30 ∧ c ∣ 30 ∧ d ∣ 30 :=
by
  sorry

end factors_are_divisors_l1784_178462


namespace nonagon_perimeter_is_28_l1784_178442

-- Definitions based on problem conditions
def numSides : Nat := 9
def lengthSides1 : Nat := 3
def lengthSides2 : Nat := 4
def numSidesOfLength1 : Nat := 8
def numSidesOfLength2 : Nat := 1

-- Theorem statement proving that the perimeter is 28 units
theorem nonagon_perimeter_is_28 : 
  numSides = numSidesOfLength1 + numSidesOfLength2 →
  8 * lengthSides1 + 1 * lengthSides2 = 28 :=
by
  intros
  sorry

end nonagon_perimeter_is_28_l1784_178442


namespace opposite_of_a_is_2022_l1784_178450

theorem opposite_of_a_is_2022 (a : Int) (h : -a = -2022) : a = 2022 := by
  sorry

end opposite_of_a_is_2022_l1784_178450


namespace ratio_of_numbers_l1784_178447

theorem ratio_of_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hsum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_numbers_l1784_178447


namespace range_of_a1_l1784_178483

noncomputable def sequence_a (n : ℕ) : ℤ := sorry
noncomputable def sum_S (n : ℕ) : ℤ := sorry

theorem range_of_a1 :
  (∀ n : ℕ, n > 0 → sum_S n + sum_S (n+1) = 2 * n^2 + n) ∧
  (∀ n : ℕ, n > 0 → sequence_a n < sequence_a (n+1)) →
  -1/4 < sequence_a 1 ∧ sequence_a 1 < 3/4 := sorry

end range_of_a1_l1784_178483


namespace square_root_condition_l1784_178402

-- Define the condition
def meaningful_square_root (x : ℝ) : Prop :=
  x - 5 ≥ 0

-- Define the theorem that x must be greater than or equal to 5 for the square root to be meaningful
theorem square_root_condition (x : ℝ) : meaningful_square_root x ↔ x ≥ 5 := by
  sorry

end square_root_condition_l1784_178402
