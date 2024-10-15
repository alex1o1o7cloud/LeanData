import Mathlib

namespace NUMINAMATH_GPT_polynomial_simplification_l2354_235413

noncomputable def given_polynomial (x : ℝ) : ℝ :=
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 + 2 * x^3

theorem polynomial_simplification (x : ℝ) :
  given_polynomial x = 2 * x^3 - x^2 - 11 * x + 27 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l2354_235413


namespace NUMINAMATH_GPT_flooring_area_already_installed_l2354_235401

variable (living_room_length : ℕ) (living_room_width : ℕ) 
variable (flooring_sqft_per_box : ℕ)
variable (remaining_boxes_needed : ℕ)
variable (already_installed : ℕ)

theorem flooring_area_already_installed 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_sqft_per_box = 10)
  (h4 : remaining_boxes_needed = 7)
  (h5 : living_room_length * living_room_width = 320)
  (h6 : already_installed = 320 - remaining_boxes_needed * flooring_sqft_per_box) : 
  already_installed = 250 :=
by
  sorry

end NUMINAMATH_GPT_flooring_area_already_installed_l2354_235401


namespace NUMINAMATH_GPT_parabola_translation_left_by_two_units_l2354_235480

/-- 
The parabola y = x^2 + 4x + 5 is obtained by translating the parabola y = x^2 + 1. 
Prove that this translation is 2 units to the left.
-/
theorem parabola_translation_left_by_two_units :
  ∀ x : ℝ, (x^2 + 4*x + 5) = ((x+2)^2 + 1) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_parabola_translation_left_by_two_units_l2354_235480


namespace NUMINAMATH_GPT_part1_part2_l2354_235467

variables (a b c d m : Real) 

-- Condition: a and b are opposite numbers
def opposite_numbers (a b : Real) : Prop := a = -b

-- Condition: c and d are reciprocals
def reciprocals (c d : Real) : Prop := c = 1 / d

-- Condition: |m| = 3
def absolute_value_three (m : Real) : Prop := abs m = 3

-- Statement for part 1
theorem part1 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) :
  a + b = 0 ∧ c * d = 1 ∧ (m = 3 ∨ m = -3) :=
by
  sorry

-- Statement for part 2
theorem part2 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) (h4 : m < 0) :
  m^3 + c * d + (a + b) / m = -26 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2354_235467


namespace NUMINAMATH_GPT_new_average_after_multiplication_l2354_235433

theorem new_average_after_multiplication
  (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 7)
  (h2 : a = 25)
  (h3 : m = 5):
  (n * a * m / n) = 125 :=
by
  sorry


end NUMINAMATH_GPT_new_average_after_multiplication_l2354_235433


namespace NUMINAMATH_GPT_part_one_part_two_l2354_235485

variable {x : ℝ}

def setA (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def setB : Set ℝ := {x | -1 / 2 < x ∧ x ≤ 2}

theorem part_one (a : ℝ) (h : a = 1) : setB ⊆ setA a :=
by
  sorry

theorem part_two (a : ℝ) : (setA a ⊆ setB) ↔ (a < -8 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l2354_235485


namespace NUMINAMATH_GPT_exponential_simplification_l2354_235487

theorem exponential_simplification : 
  (10^0.25) * (10^0.25) * (10^0.5) * (10^0.5) * (10^0.75) * (10^0.75) = 1000 := 
by 
  sorry

end NUMINAMATH_GPT_exponential_simplification_l2354_235487


namespace NUMINAMATH_GPT_f_has_two_zeros_l2354_235477

def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
by
  sorry

end NUMINAMATH_GPT_f_has_two_zeros_l2354_235477


namespace NUMINAMATH_GPT_find_stream_speed_l2354_235459

-- Define the problem based on the provided conditions
theorem find_stream_speed (b s : ℝ) (h1 : b + s = 250 / 7) (h2 : b - s = 150 / 21) : s = 14.28 :=
by
  sorry

end NUMINAMATH_GPT_find_stream_speed_l2354_235459


namespace NUMINAMATH_GPT_wax_current_eq_l2354_235400

-- Define the constants for the wax required and additional wax needed
def w_required : ℕ := 166
def w_more : ℕ := 146

-- Define the term to represent the current wax he has
def w_current : ℕ := w_required - w_more

-- Theorem statement to prove the current wax quantity
theorem wax_current_eq : w_current = 20 := by
  -- Proof outline would go here, but per instructions, we skip with sorry
  sorry

end NUMINAMATH_GPT_wax_current_eq_l2354_235400


namespace NUMINAMATH_GPT_find_b_l2354_235407

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_find_b_l2354_235407


namespace NUMINAMATH_GPT_ab_eq_six_l2354_235452

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end NUMINAMATH_GPT_ab_eq_six_l2354_235452


namespace NUMINAMATH_GPT_intersection_in_quadrant_II_l2354_235438

theorem intersection_in_quadrant_II (x y : ℝ) 
  (h1: y ≥ -2 * x + 3) 
  (h2: y ≤ 3 * x + 6) 
  (h_intersection: x = -3 / 5 ∧ y = 21 / 5) :
  x < 0 ∧ y > 0 := 
sorry

end NUMINAMATH_GPT_intersection_in_quadrant_II_l2354_235438


namespace NUMINAMATH_GPT_factorial_division_identity_l2354_235436

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end NUMINAMATH_GPT_factorial_division_identity_l2354_235436


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2354_235486

variables {x p q : ℝ}

theorem quadratic_inequality_solution
  (h1 : ∀ x, x^2 + p * x + q < 0 ↔ -1/2 < x ∧ x < 1/3) : 
  ∀ x, q * x^2 + p * x + 1 > 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2354_235486


namespace NUMINAMATH_GPT_A_union_B_eq_B_l2354_235481

-- Define set A
def A : Set ℝ := {-1, 0, 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- The proof problem
theorem A_union_B_eq_B : A ∪ B = B := 
  sorry

end NUMINAMATH_GPT_A_union_B_eq_B_l2354_235481


namespace NUMINAMATH_GPT_units_digit_of_eight_consecutive_odd_numbers_is_zero_l2354_235410

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem units_digit_of_eight_consecutive_odd_numbers_is_zero (n : ℤ)
  (h₀ : is_odd n) :
  ((n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) % 10 = 0) :=
sorry

end NUMINAMATH_GPT_units_digit_of_eight_consecutive_odd_numbers_is_zero_l2354_235410


namespace NUMINAMATH_GPT_calculate_expression_l2354_235461

theorem calculate_expression :
  150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2354_235461


namespace NUMINAMATH_GPT_length_of_each_cut_section_xiao_hong_age_l2354_235491

theorem length_of_each_cut_section (x : ℝ) (h : 60 - 2 * x = 10) : x = 25 := sorry

theorem xiao_hong_age (y : ℝ) (h : 2 * y + 10 = 30) : y = 10 := sorry

end NUMINAMATH_GPT_length_of_each_cut_section_xiao_hong_age_l2354_235491


namespace NUMINAMATH_GPT_solve_inequalities_l2354_235458

theorem solve_inequalities (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 3 → x - a < 1 ∧ x - 2 * b > 3) ↔ (a = 2 ∧ b = -2) := 
  by 
    sorry

end NUMINAMATH_GPT_solve_inequalities_l2354_235458


namespace NUMINAMATH_GPT_lcm_gcd_product_l2354_235468

def a : ℕ := 20 -- Defining the first number as 20
def b : ℕ := 90 -- Defining the second number as 90

theorem lcm_gcd_product : Nat.lcm a b * Nat.gcd a b = 1800 := 
by 
  -- Computation and proof steps would go here
  sorry -- Replace with actual proof

end NUMINAMATH_GPT_lcm_gcd_product_l2354_235468


namespace NUMINAMATH_GPT_urn_marbles_100_white_l2354_235482

theorem urn_marbles_100_white 
(initial_white initial_black final_white final_black : ℕ) 
(h_initial : initial_white = 150 ∧ initial_black = 50)
(h_operations : 
  (∀ n, (initial_white - 3 * n + 2 * n = final_white ∧ initial_black + n = final_black) ∨
  (initial_white - 2 * n - 1 = initial_white ∧ initial_black = final_black) ∨
  (initial_white - 1 * n - 2 = final_white ∧ initial_black - 1 * n = final_black) ∨
  (initial_white - 3 * n + 2 = final_white ∧ initial_black + 1 * n = final_black)) →
  ((initial_white = 150 ∧ initial_black = 50) →
   ∃ m: ℕ, final_white = 100)) :
∃ n: ℕ, initial_white - 3 * n + 2 * n = 100 ∧ initial_black + n = final_black :=
sorry

end NUMINAMATH_GPT_urn_marbles_100_white_l2354_235482


namespace NUMINAMATH_GPT_ab_necessary_but_not_sufficient_l2354_235404

theorem ab_necessary_but_not_sufficient (a b : ℝ) (i : ℂ) (hi : i^2 = -1) : 
  ab < 0 → ¬ (ab >= 0) ∧ (¬ (ab <= 0)) → (z = i * (a + b * i)) ∧ a > 0 ∧ -b > 0 := 
  sorry

end NUMINAMATH_GPT_ab_necessary_but_not_sufficient_l2354_235404


namespace NUMINAMATH_GPT_two_digit_plus_one_multiple_of_3_4_5_6_7_l2354_235475

theorem two_digit_plus_one_multiple_of_3_4_5_6_7 (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) :
  (∃ m : ℕ, (m = n - 1 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 ∧ m % 7 = 0)) → False :=
sorry

end NUMINAMATH_GPT_two_digit_plus_one_multiple_of_3_4_5_6_7_l2354_235475


namespace NUMINAMATH_GPT_find_a_of_square_roots_l2354_235471

theorem find_a_of_square_roots (a : ℤ) (n : ℤ) (h₁ : 2 * a + 1 = n) (h₂ : a + 5 = n) : a = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_a_of_square_roots_l2354_235471


namespace NUMINAMATH_GPT_handshakes_count_l2354_235446

-- Define the parameters
def teams : ℕ := 3
def players_per_team : ℕ := 7
def referees : ℕ := 3

-- Calculate handshakes among team members
def handshakes_among_teams :=
  let unique_handshakes_per_team := players_per_team * 2 * players_per_team / 2
  unique_handshakes_per_team * teams

-- Calculate handshakes between players and referees
def players_shake_hands_with_referees :=
  teams * players_per_team * referees

-- Calculate total handshakes
def total_handshakes :=
  handshakes_among_teams + players_shake_hands_with_referees

-- Proof statement
theorem handshakes_count : total_handshakes = 210 := by
  sorry

end NUMINAMATH_GPT_handshakes_count_l2354_235446


namespace NUMINAMATH_GPT_odd_function_value_at_2_l2354_235473

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

theorem odd_function_value_at_2 : f (-2) + f (2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_value_at_2_l2354_235473


namespace NUMINAMATH_GPT_students_in_same_month_l2354_235419

theorem students_in_same_month (students : ℕ) (months : ℕ) 
  (h : students = 50) (h_months : months = 12) : 
  ∃ k ≥ 5, ∃ i, i < months ∧ ∃ f : ℕ → ℕ, (∀ j < students, f j < months) 
  ∧ ∃ n ≥ 5, ∃ j < students, f j = i :=
by 
  sorry

end NUMINAMATH_GPT_students_in_same_month_l2354_235419


namespace NUMINAMATH_GPT_greg_age_is_16_l2354_235418

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end NUMINAMATH_GPT_greg_age_is_16_l2354_235418


namespace NUMINAMATH_GPT_no_solution_condition_l2354_235470

theorem no_solution_condition (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_condition_l2354_235470


namespace NUMINAMATH_GPT_average_of_three_l2354_235422

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end NUMINAMATH_GPT_average_of_three_l2354_235422


namespace NUMINAMATH_GPT_min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l2354_235434

theorem min_questions_to_find_phone_number : 
  ∃ n : ℕ, ∀ (N : ℕ), (N = 100000 → 2 ^ n ≥ N) ∧ (2 ^ (n - 1) < N) := sorry

-- In simpler form, since log_2(100000) ≈ 16.60965, we have:
theorem min_questions_to_find_phone_number_is_17 : 
  ∀ (N : ℕ), (N = 100000 → 17 = Nat.ceil (Real.logb 2 100000)) := sorry

end NUMINAMATH_GPT_min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l2354_235434


namespace NUMINAMATH_GPT_area_of_wrapping_paper_l2354_235494

theorem area_of_wrapping_paper (l w h: ℝ) (l_pos: 0 < l) (w_pos: 0 < w) (h_pos: 0 < h) :
  ∃ s: ℝ, s = l + w ∧ s^2 = (l + w)^2 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_wrapping_paper_l2354_235494


namespace NUMINAMATH_GPT_value_of_f_2017_l2354_235427

def f (x : ℕ) : ℕ := x^2 - x * (0 : ℕ) - 1

theorem value_of_f_2017 : f 2017 = 2016 * 2018 := by
  sorry

end NUMINAMATH_GPT_value_of_f_2017_l2354_235427


namespace NUMINAMATH_GPT_simplify_expression_l2354_235474

theorem simplify_expression :
  ((3 + 4 + 5 + 6) ^ 2 / 4) + ((3 * 6 + 9) ^ 2 / 3) = 324 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2354_235474


namespace NUMINAMATH_GPT_books_left_in_library_l2354_235426

theorem books_left_in_library (initial_books : ℕ) (borrowed_books : ℕ) (left_books : ℕ) 
  (h1 : initial_books = 75) (h2 : borrowed_books = 18) : left_books = 57 :=
by
  sorry

end NUMINAMATH_GPT_books_left_in_library_l2354_235426


namespace NUMINAMATH_GPT_equilibrium_mass_l2354_235460

variable (l m2 S g : ℝ) (m1 : ℝ)

-- Given conditions
def length_of_rod : ℝ := 0.5 -- length l in meters
def mass_of_rod : ℝ := 2 -- mass m2 in kg
def distance_S : ℝ := 0.1 -- distance S in meters
def gravity : ℝ := 9.8 -- gravitational acceleration in m/s^2

-- Equivalence statement
theorem equilibrium_mass (h1 : l = length_of_rod)
                         (h2 : m2 = mass_of_rod)
                         (h3 : S = distance_S)
                         (h4 : g = gravity) :
  m1 = 10 := sorry

end NUMINAMATH_GPT_equilibrium_mass_l2354_235460


namespace NUMINAMATH_GPT_first_prize_ticket_numbers_l2354_235456

theorem first_prize_ticket_numbers :
  {n : ℕ | n < 10000 ∧ (n % 1000 = 418)} = {418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by
  sorry

end NUMINAMATH_GPT_first_prize_ticket_numbers_l2354_235456


namespace NUMINAMATH_GPT_no_perf_square_of_prime_three_digit_l2354_235495

theorem no_perf_square_of_prime_three_digit {A B C : ℕ} (h_prime: Prime (100 * A + 10 * B + C)) : ¬ ∃ n : ℕ, B^2 - 4 * A * C = n^2 :=
by
  sorry

end NUMINAMATH_GPT_no_perf_square_of_prime_three_digit_l2354_235495


namespace NUMINAMATH_GPT_pencils_bought_l2354_235476

theorem pencils_bought (payment change pencil_cost glue_cost : ℕ)
  (h_payment : payment = 1000)
  (h_change : change = 100)
  (h_pencil_cost : pencil_cost = 210)
  (h_glue_cost : glue_cost = 270) :
  (payment - change - glue_cost) / pencil_cost = 3 :=
by sorry

end NUMINAMATH_GPT_pencils_bought_l2354_235476


namespace NUMINAMATH_GPT_number_of_cows_l2354_235420

theorem number_of_cows (H : ℕ) (C : ℕ) (h1 : H = 6) (h2 : C / H = 7 / 2) : C = 21 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l2354_235420


namespace NUMINAMATH_GPT_negation_of_existential_statement_l2354_235414

theorem negation_of_existential_statement {f : ℝ → ℝ} :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_statement_l2354_235414


namespace NUMINAMATH_GPT_exists_coeff_less_than_neg_one_l2354_235450

theorem exists_coeff_less_than_neg_one 
  (P : Polynomial ℤ)
  (h1 : P.eval 1 = 0)
  (h2 : P.eval 2 = 0) :
  ∃ i, P.coeff i < -1 := sorry

end NUMINAMATH_GPT_exists_coeff_less_than_neg_one_l2354_235450


namespace NUMINAMATH_GPT_sequence_AMS_ends_in_14_l2354_235472

def start := 3
def add_two (x : ℕ) := x + 2
def multiply_three (x : ℕ) := x * 3
def subtract_one (x : ℕ) := x - 1

theorem sequence_AMS_ends_in_14 : 
  subtract_one (multiply_three (add_two start)) = 14 :=
by
  -- The proof would go here if required.
  sorry

end NUMINAMATH_GPT_sequence_AMS_ends_in_14_l2354_235472


namespace NUMINAMATH_GPT_y_intercept_of_line_l2354_235403

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l2354_235403


namespace NUMINAMATH_GPT_shaina_chocolate_amount_l2354_235441

variable (total_chocolate : ℚ) (num_piles : ℕ) (fraction_kept : ℚ)
variable (eq_total_chocolate : total_chocolate = 72 / 7)
variable (eq_num_piles : num_piles = 6)
variable (eq_fraction_kept : fraction_kept = 1 / 3)

theorem shaina_chocolate_amount :
  (total_chocolate / num_piles) * (1 - fraction_kept) = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_shaina_chocolate_amount_l2354_235441


namespace NUMINAMATH_GPT_sum_of_731_and_one_fifth_l2354_235492

theorem sum_of_731_and_one_fifth :
  (7.31 + (1 / 5) = 7.51) :=
sorry

end NUMINAMATH_GPT_sum_of_731_and_one_fifth_l2354_235492


namespace NUMINAMATH_GPT_solution_exists_l2354_235405

theorem solution_exists (x y z u v : ℕ) (hx : x > 2000) (hy : y > 2000) (hz : z > 2000) (hu : u > 2000) (hv : v > 2000) : 
  x^2 + y^2 + z^2 + u^2 + v^2 = x * y * z * u * v - 65 :=
sorry

end NUMINAMATH_GPT_solution_exists_l2354_235405


namespace NUMINAMATH_GPT_number_of_vegetarians_l2354_235424

-- Define the conditions
def only_veg : ℕ := 11
def only_nonveg : ℕ := 6
def both_veg_and_nonveg : ℕ := 9

-- Define the total number of vegetarians
def total_veg : ℕ := only_veg + both_veg_and_nonveg

-- The statement to be proved
theorem number_of_vegetarians : total_veg = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_vegetarians_l2354_235424


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2354_235499

variable {a : ℕ → ℕ}

noncomputable def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2354_235499


namespace NUMINAMATH_GPT_correct_removal_of_parentheses_l2354_235455

theorem correct_removal_of_parentheses (x : ℝ) : (1/3) * (6 * x - 3) = 2 * x - 1 :=
by sorry

end NUMINAMATH_GPT_correct_removal_of_parentheses_l2354_235455


namespace NUMINAMATH_GPT_average_monthly_balance_l2354_235431

theorem average_monthly_balance :
  let balances := [100, 200, 250, 50, 300, 300]
  (balances.sum / balances.length : ℕ) = 200 :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_balance_l2354_235431


namespace NUMINAMATH_GPT_infinite_series_converges_l2354_235484

open BigOperators

noncomputable def problem : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0

theorem infinite_series_converges : problem = 61 / 24 :=
sorry

end NUMINAMATH_GPT_infinite_series_converges_l2354_235484


namespace NUMINAMATH_GPT_balls_in_boxes_l2354_235425

-- Define the conditions
def num_balls : ℕ := 3
def num_boxes : ℕ := 4

-- Define the problem
theorem balls_in_boxes : (num_boxes ^ num_balls) = 64 :=
by
  -- We acknowledge that we are skipping the proof details here
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l2354_235425


namespace NUMINAMATH_GPT_compute_fraction_square_l2354_235489

theorem compute_fraction_square : 6 * (3 / 7) ^ 2 = 54 / 49 :=
by 
  sorry

end NUMINAMATH_GPT_compute_fraction_square_l2354_235489


namespace NUMINAMATH_GPT_original_expenditure_beginning_month_l2354_235443

theorem original_expenditure_beginning_month (A E : ℝ)
  (h1 : E = 35 * A)
  (h2 : E + 84 = 42 * (A - 1))
  (h3 : E + 124 = 37 * (A + 1))
  (h4 : E + 154 = 40 * (A + 1)) :
  E = 630 := 
sorry

end NUMINAMATH_GPT_original_expenditure_beginning_month_l2354_235443


namespace NUMINAMATH_GPT_evaluate_f_at_3_l2354_235498

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^7 + a * x^5 + b * x - 5

theorem evaluate_f_at_3 (a b : ℝ)
  (h : f (-3) a b = 5) : f 3 a b = -15 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_3_l2354_235498


namespace NUMINAMATH_GPT_fill_time_eight_faucets_l2354_235429

theorem fill_time_eight_faucets (r : ℝ) (h1 : 4 * r * 8 = 150) :
  8 * r * (50 / (8 * r)) * 60 = 80 := by
  sorry

end NUMINAMATH_GPT_fill_time_eight_faucets_l2354_235429


namespace NUMINAMATH_GPT_triangle_side_length_BC_49_l2354_235447

theorem triangle_side_length_BC_49
  (angle_A : ℝ)
  (AC : ℝ)
  (area_ABC : ℝ)
  (h1 : angle_A = 60)
  (h2 : AC = 16)
  (h3 : area_ABC = 220 * Real.sqrt 3) : 
  ∃ (BC : ℝ), BC = 49 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_BC_49_l2354_235447


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_plus_z_squared_l2354_235493

theorem find_x_squared_plus_y_squared_plus_z_squared
  (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 :=
by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_plus_z_squared_l2354_235493


namespace NUMINAMATH_GPT_solve_for_x_l2354_235457

theorem solve_for_x (x : ℕ) : 8 * 4^x = 2048 → x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2354_235457


namespace NUMINAMATH_GPT_percent_of_b_l2354_235469

theorem percent_of_b (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) : c = 0.1 * b := 
by
  sorry

end NUMINAMATH_GPT_percent_of_b_l2354_235469


namespace NUMINAMATH_GPT_smallest_positive_period_one_increasing_interval_l2354_235478

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

def is_periodic_with_period (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem smallest_positive_period :
  is_periodic_with_period f Real.pi :=
sorry

theorem one_increasing_interval :
  is_increasing_on f (-(Real.pi / 8)) (3 * Real.pi / 8) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_one_increasing_interval_l2354_235478


namespace NUMINAMATH_GPT_solution_comparison_l2354_235479

theorem solution_comparison (a a' b b' k : ℝ) (h1 : a ≠ 0) (h2 : a' ≠ 0) (h3 : 0 < k) :
  (k * b * a') > (a * b') :=
sorry

end NUMINAMATH_GPT_solution_comparison_l2354_235479


namespace NUMINAMATH_GPT_school_cases_of_water_l2354_235442

theorem school_cases_of_water (bottles_per_case bottles_used_first_game bottles_left_after_second_game bottles_used_second_game : ℕ)
  (h1 : bottles_per_case = 20)
  (h2 : bottles_used_first_game = 70)
  (h3 : bottles_left_after_second_game = 20)
  (h4 : bottles_used_second_game = 110) :
  let total_bottles_used := bottles_used_first_game + bottles_used_second_game
  let total_bottles_initial := total_bottles_used + bottles_left_after_second_game
  let number_of_cases := total_bottles_initial / bottles_per_case
  number_of_cases = 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_school_cases_of_water_l2354_235442


namespace NUMINAMATH_GPT_charts_per_associate_professor_l2354_235453

-- Definitions
def A : ℕ := 3
def B : ℕ := 4
def C : ℕ := 1

-- Conditions based on the given problem
axiom h1 : 2 * A + B = 10
axiom h2 : A * C + 2 * B = 11
axiom h3 : A + B = 7

-- The theorem to be proven
theorem charts_per_associate_professor : C = 1 := by
  sorry

end NUMINAMATH_GPT_charts_per_associate_professor_l2354_235453


namespace NUMINAMATH_GPT_kopeechka_items_l2354_235488

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end NUMINAMATH_GPT_kopeechka_items_l2354_235488


namespace NUMINAMATH_GPT_coordinates_of_focus_with_greater_x_coordinate_l2354_235496

noncomputable def focus_of_ellipse_with_greater_x_coordinate : (ℝ × ℝ) :=
  let center : ℝ × ℝ := (3, -2)
  let a : ℝ := 3 -- semi-major axis length
  let b : ℝ := 2 -- semi-minor axis length
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let focus_x : ℝ := 3 + c
  (focus_x, -2)

theorem coordinates_of_focus_with_greater_x_coordinate :
  focus_of_ellipse_with_greater_x_coordinate = (3 + Real.sqrt 5, -2) := 
sorry

end NUMINAMATH_GPT_coordinates_of_focus_with_greater_x_coordinate_l2354_235496


namespace NUMINAMATH_GPT_count_valid_n_le_30_l2354_235449

theorem count_valid_n_le_30 :
  ∀ n : ℕ, (0 < n ∧ n ≤ 30) → (n! * 2) % (n * (n + 1)) = 0 := by
  sorry

end NUMINAMATH_GPT_count_valid_n_le_30_l2354_235449


namespace NUMINAMATH_GPT_glen_animals_total_impossible_l2354_235430

theorem glen_animals_total_impossible (t : ℕ) :
  ¬ (∃ t : ℕ, 41 * t = 108) := sorry

end NUMINAMATH_GPT_glen_animals_total_impossible_l2354_235430


namespace NUMINAMATH_GPT_child_l2354_235451

-- Definitions of the given conditions
def total_money : ℕ := 35
def adult_ticket_cost : ℕ := 8
def number_of_children : ℕ := 9

-- Statement of the math proof problem
theorem child's_ticket_cost : ∃ C : ℕ, total_money - adult_ticket_cost = C * number_of_children ∧ C = 3 :=
by
  sorry

end NUMINAMATH_GPT_child_l2354_235451


namespace NUMINAMATH_GPT_range_of_a_l2354_235464

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, 0 < x ∧ 3*x + a ≤ 2 → x = 1 ∨ x = 2) ↔ (-7 < a ∧ a ≤ -4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2354_235464


namespace NUMINAMATH_GPT_range_of_x_squared_plus_y_squared_l2354_235428

def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y
def symmetric_about_origin (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem range_of_x_squared_plus_y_squared 
  (f : ℝ → ℝ) 
  (h_incr : increasing f) 
  (h_symm : symmetric_about_origin f) 
  (h_ineq : ∀ x y, f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) : 
  ∀ x y, 16 < x^2 + y^2 ∧ x^2 + y^2 < 36 := 
sorry

end NUMINAMATH_GPT_range_of_x_squared_plus_y_squared_l2354_235428


namespace NUMINAMATH_GPT_find_b_value_l2354_235412

-- Let's define the given conditions as hypotheses in Lean

theorem find_b_value 
  (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 2)) 
  (h2 : (x2, y2) = (8, 14)) 
  (midpoint : ∃ (m1 m2 : ℤ), m1 = (x1 + x2) / 2 ∧ m2 = (y1 + y2) / 2 ∧ (m1, m2) = (5, 8))
  (perpendicular_bisector : ∀ (x y : ℤ), x + y = b → (x, y) = (5, 8)) :
  b = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_b_value_l2354_235412


namespace NUMINAMATH_GPT_droneSystemEquations_l2354_235408

-- Definitions based on conditions
def typeADrones (x y : ℕ) : Prop := x = (1/2 : ℝ) * (x + y) + 11
def typeBDrones (x y : ℕ) : Prop := y = (1/3 : ℝ) * (x + y) - 2

-- Theorem statement
theorem droneSystemEquations (x y : ℕ) :
  typeADrones x y ∧ typeBDrones x y ↔
  (x = (1/2 : ℝ) * (x + y) + 11 ∧ y = (1/3 : ℝ) * (x + y) - 2) :=
by sorry

end NUMINAMATH_GPT_droneSystemEquations_l2354_235408


namespace NUMINAMATH_GPT_range_of_d_l2354_235497

theorem range_of_d (a_1 d : ℝ) (h : (a_1 + 2 * d) * (a_1 + 3 * d) + 1 = 0) :
  d ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_GPT_range_of_d_l2354_235497


namespace NUMINAMATH_GPT_journey_time_l2354_235416

-- Conditions
def initial_speed : ℝ := 80  -- miles per hour
def initial_time : ℝ := 5    -- hours
def new_speed : ℝ := 50      -- miles per hour
def distance : ℝ := initial_speed * initial_time

-- Statement
theorem journey_time :
  distance / new_speed = 8.00 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_l2354_235416


namespace NUMINAMATH_GPT_range_of_a_l2354_235490

theorem range_of_a (a : ℝ) (h : ∀ x, a ≤ x ∧ x ≤ a + 2 → |x + a| ≥ 2 * |x|) : a ≤ -3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2354_235490


namespace NUMINAMATH_GPT_wendys_brother_pieces_l2354_235402

-- Definitions based on conditions
def number_of_boxes : ℕ := 2
def pieces_per_box : ℕ := 3
def total_pieces : ℕ := 12

-- Summarization of Wendy's pieces of candy
def wendys_pieces : ℕ := number_of_boxes * pieces_per_box

-- Lean statement: Prove the number of pieces Wendy's brother had
theorem wendys_brother_pieces : total_pieces - wendys_pieces = 6 :=
by
  sorry

end NUMINAMATH_GPT_wendys_brother_pieces_l2354_235402


namespace NUMINAMATH_GPT_line_circle_no_intersection_l2354_235448

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l2354_235448


namespace NUMINAMATH_GPT_train_speed_is_300_kmph_l2354_235483

noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_is_300_kmph :
  train_speed 1250 15 = 300 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_300_kmph_l2354_235483


namespace NUMINAMATH_GPT_trig_expression_value_l2354_235440

theorem trig_expression_value {θ : Real} (h : Real.tan θ = 2) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l2354_235440


namespace NUMINAMATH_GPT_find_roots_of_star_eq_l2354_235454

def star (a b : ℝ) : ℝ := a^2 - b^2

theorem find_roots_of_star_eq :
  (star (star 2 3) x = 9) ↔ (x = 4 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_find_roots_of_star_eq_l2354_235454


namespace NUMINAMATH_GPT_chess_team_boys_l2354_235415

variable (B G : ℕ)

theorem chess_team_boys (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 20) : B = 15 := by
  sorry

end NUMINAMATH_GPT_chess_team_boys_l2354_235415


namespace NUMINAMATH_GPT_find_b_minus_a_l2354_235465

theorem find_b_minus_a (a b : ℝ) (h : ∀ x : ℝ, 0 ≤ x → 
  0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) : 
  b - a = 2 :=
sorry

end NUMINAMATH_GPT_find_b_minus_a_l2354_235465


namespace NUMINAMATH_GPT_initialNumberMembers_l2354_235432

-- Define the initial number of members in the group
def initialMembers (n : ℕ) : Prop :=
  let W := n * 48 -- Initial total weight
  let newWeight := W + 78 + 93 -- New total weight after two members join
  let newAverageWeight := (n + 2) * 51 -- New total weight based on the new average weight
  newWeight = newAverageWeight -- The condition that the new total weights are equal

-- Theorem stating that the initial number of members is 23
theorem initialNumberMembers : initialMembers 23 :=
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_initialNumberMembers_l2354_235432


namespace NUMINAMATH_GPT_merry_go_round_cost_per_child_l2354_235439

-- Definitions
def num_children := 5
def ferris_wheel_cost_per_child := 5
def num_children_on_ferris_wheel := 3
def ice_cream_cost_per_cone := 8
def ice_cream_cones_per_child := 2
def total_spent := 110

-- Totals
def ferris_wheel_total_cost := num_children_on_ferris_wheel * ferris_wheel_cost_per_child
def ice_cream_total_cost := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone
def merry_go_round_total_cost := total_spent - ferris_wheel_total_cost - ice_cream_total_cost

-- Final proof statement
theorem merry_go_round_cost_per_child : 
  merry_go_round_total_cost / num_children = 3 :=
by
  -- We skip the actual proof here
  sorry

end NUMINAMATH_GPT_merry_go_round_cost_per_child_l2354_235439


namespace NUMINAMATH_GPT_cory_fruits_arrangement_l2354_235423

-- Conditions
def apples : ℕ := 4
def oranges : ℕ := 2
def lemon : ℕ := 1
def total_fruits : ℕ := apples + oranges + lemon

-- Formula to calculate the number of distinct ways
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count : ℕ :=
  factorial total_fruits / (factorial apples * factorial oranges * factorial lemon)

theorem cory_fruits_arrangement : arrangement_count = 105 := by
  -- Sorry is placed here to skip the actual proof
  sorry

end NUMINAMATH_GPT_cory_fruits_arrangement_l2354_235423


namespace NUMINAMATH_GPT_rectangle_area_l2354_235421

theorem rectangle_area (r : ℝ) (w l : ℝ) (h_radius : r = 7) 
  (h_ratio : l = 3 * w) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2354_235421


namespace NUMINAMATH_GPT_three_mathematicians_same_language_l2354_235435

theorem three_mathematicians_same_language
  (M : Fin 9 → Finset string)
  (h1 : ∀ i j k : Fin 9, ∃ lang, i ≠ j → i ≠ k → j ≠ k → lang ∈ M i ∧ lang ∈ M j)
  (h2 : ∀ i : Fin 9, (M i).card ≤ 3)
  : ∃ lang ∈ ⋃ i, M i, ∃ (A B C : Fin 9), A ≠ B → A ≠ C → B ≠ C → lang ∈ M A ∧ lang ∈ M B ∧ lang ∈ M C :=
sorry

end NUMINAMATH_GPT_three_mathematicians_same_language_l2354_235435


namespace NUMINAMATH_GPT_largest_integer_divides_product_l2354_235409

theorem largest_integer_divides_product (n : ℕ) : 
  ∃ m, ∀ k : ℕ, k = (2*n-1)*(2*n)*(2*n+2) → m ≥ 1 ∧ m = 8 ∧ m ∣ k :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_divides_product_l2354_235409


namespace NUMINAMATH_GPT_rebecca_gemstones_needed_l2354_235411

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end NUMINAMATH_GPT_rebecca_gemstones_needed_l2354_235411


namespace NUMINAMATH_GPT_sin_four_arcsin_eq_l2354_235417

theorem sin_four_arcsin_eq (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) :=
by
  sorry

end NUMINAMATH_GPT_sin_four_arcsin_eq_l2354_235417


namespace NUMINAMATH_GPT_alicia_satisfaction_l2354_235445

theorem alicia_satisfaction (t : ℚ) (h_sat : t * (12 - t) = (4 - t) * (2 * t + 2)) : t = 2 :=
by
  sorry

end NUMINAMATH_GPT_alicia_satisfaction_l2354_235445


namespace NUMINAMATH_GPT_exists_square_with_digit_sum_2002_l2354_235463

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_square_with_digit_sum_2002 :
  ∃ (n : ℕ), sum_of_digits (n^2) = 2002 :=
sorry

end NUMINAMATH_GPT_exists_square_with_digit_sum_2002_l2354_235463


namespace NUMINAMATH_GPT_rectangle_in_triangle_area_l2354_235406

theorem rectangle_in_triangle_area
  (PR : ℝ) (h_PR : PR = 15)
  (Q_altitude : ℝ) (h_Q_altitude : Q_altitude = 9)
  (x : ℝ)
  (AD : ℝ) (h_AD : AD = x)
  (AB : ℝ) (h_AB : AB = x / 3) :
  (AB * AD = 675 / 64) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_in_triangle_area_l2354_235406


namespace NUMINAMATH_GPT_original_number_of_candies_l2354_235466

theorem original_number_of_candies (x : ℝ) (h₀ : x * (0.7 ^ 3) = 40) : x = 117 :=
by 
  sorry

end NUMINAMATH_GPT_original_number_of_candies_l2354_235466


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2354_235437

theorem necessary_but_not_sufficient (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 → m < 2 :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2354_235437


namespace NUMINAMATH_GPT_possible_values_of_x_l2354_235444

theorem possible_values_of_x (x : ℕ) (h1 : ∃ k : ℕ, k * k = 8 - x) (h2 : 1 ≤ x ∧ x ≤ 8) :
  x = 4 ∨ x = 7 ∨ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_x_l2354_235444


namespace NUMINAMATH_GPT_calculate_star_value_l2354_235462

def custom_operation (a b : ℕ) : ℕ :=
  (a + b)^3

theorem calculate_star_value : custom_operation 3 5 = 512 :=
by
  sorry

end NUMINAMATH_GPT_calculate_star_value_l2354_235462
