import Mathlib

namespace NUMINAMATH_GPT_oldest_sibling_age_difference_l712_71212

theorem oldest_sibling_age_difference 
  (D : ℝ) 
  (avg_age : ℝ) 
  (hD : D = 25.75) 
  (h_avg : avg_age = 30) :
  ∃ A : ℝ, (A - D ≥ 17) :=
by
  sorry

end NUMINAMATH_GPT_oldest_sibling_age_difference_l712_71212


namespace NUMINAMATH_GPT_trigonometric_operation_l712_71286

theorem trigonometric_operation :
  let m := Real.cos (Real.pi / 6)
  let n := Real.sin (Real.pi / 6)
  let op (m n : ℝ) := m^2 - m * n - n^2
  op m n = (1 / 2 : ℝ) - (Real.sqrt 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_operation_l712_71286


namespace NUMINAMATH_GPT_solve_complex_problem_l712_71231

-- Define the problem
def complex_sum_eq_two (a b : ℝ) (i : ℂ) : Prop :=
  a + b = 2

-- Define the conditions
def conditions (a b : ℝ) (i : ℂ) : Prop :=
  a + b * i = (1 - i) * (2 + i)

-- State the theorem
theorem solve_complex_problem (a b : ℝ) (i : ℂ) (h : conditions a b i) : complex_sum_eq_two a b i :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_solve_complex_problem_l712_71231


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l712_71281

theorem solve_eq1 (x : ℝ) : (12 * (x - 1) ^ 2 = 3) ↔ (x = 3/2 ∨ x = 1/2) := 
by sorry

theorem solve_eq2 (x : ℝ) : ((x + 1) ^ 3 = 0.125) ↔ (x = -0.5) := 
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l712_71281


namespace NUMINAMATH_GPT_circle_symmetric_line_l712_71213

theorem circle_symmetric_line (m : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) → (3*x + y + m = 0)) →
  m = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_circle_symmetric_line_l712_71213


namespace NUMINAMATH_GPT_quarters_number_l712_71257

theorem quarters_number (total_value : ℝ)
    (bills1 : ℝ := 2)
    (bill5 : ℝ := 5)
    (dimes : ℝ := 20 * 0.1)
    (nickels : ℝ := 8 * 0.05)
    (pennies : ℝ := 35 * 0.01) :
    total_value = 13 → (total_value - (bills1 + bill5 + dimes + nickels + pennies)) / 0.25 = 13 :=
by
  intro h
  have h_total := h
  sorry

end NUMINAMATH_GPT_quarters_number_l712_71257


namespace NUMINAMATH_GPT_percentage_of_students_chose_spring_is_10_l712_71290

-- Define the constants given in the problem
def total_students : ℕ := 10
def students_spring : ℕ := 1

-- Define the percentage calculation formula
def percentage (part total : ℕ) : ℕ := (part * 100) / total

-- State the theorem
theorem percentage_of_students_chose_spring_is_10 :
  percentage students_spring total_students = 10 :=
by
  -- We don't need to provide a proof here, just state it.
  sorry

end NUMINAMATH_GPT_percentage_of_students_chose_spring_is_10_l712_71290


namespace NUMINAMATH_GPT_phi_range_l712_71294

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem phi_range (φ : ℝ) : 
  (|φ| ≤ Real.pi / 2) ∧ 
  (∀ x ∈ Set.Ioo (Real.pi / 24) (Real.pi / 3), f x φ > 2) →
  (Real.pi / 12 ≤ φ ∧ φ ≤ Real.pi / 6) :=
by
  sorry

end NUMINAMATH_GPT_phi_range_l712_71294


namespace NUMINAMATH_GPT_max_value_abcd_l712_71236

-- Define the digits and constraints on them
def distinct_digits (a b c d e : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Encode the given problem as a Lean theorem
theorem max_value_abcd (a b c d e : ℕ) 
  (h₀ : distinct_digits a b c d e)
  (h₁ : 0 ≤ a ∧ a ≤ 9) 
  (h₂ : 0 ≤ b ∧ b ≤ 9) 
  (h₃ : 0 ≤ c ∧ c ≤ 9) 
  (h₄ : 0 ≤ d ∧ d ≤ 9)
  (h₅ : 0 ≤ e ∧ e ≤ 9)
  (h₆ : e ≠ 0)
  (h₇ : a * 1000 + b * 100 + c * 10 + d = (a * 100 + a * 10 + d) * e) :
  a * 1000 + b * 100 + c * 10 + d = 3015 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_abcd_l712_71236


namespace NUMINAMATH_GPT_circle_inequality_l712_71265

-- Given a circle of 100 pairwise distinct numbers a : ℕ → ℝ for 1 ≤ i ≤ 100
variables {a : ℕ → ℝ}
-- Hypothesis 1: distinct numbers
def distinct_numbers (a : ℕ → ℝ) := ∀ i j : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (1 ≤ j ∧ j ≤ 100) ∧ (i ≠ j) → a i ≠ a j

-- Theorem: Prove that there exist four consecutive numbers such that the sum of the first and the last number is strictly greater than the sum of the two middle numbers
theorem circle_inequality (h_distinct : distinct_numbers a) : 
  ∃ i : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100)) :=
sorry

end NUMINAMATH_GPT_circle_inequality_l712_71265


namespace NUMINAMATH_GPT_find_period_l712_71207

theorem find_period (A P R : ℕ) (I : ℕ) (T : ℚ) 
  (hA : A = 1120) 
  (hP : P = 896) 
  (hR : R = 5) 
  (hSI : I = A - P) 
  (hT : I = (P * R * T) / 100) :
  T = 5 := by 
  sorry

end NUMINAMATH_GPT_find_period_l712_71207


namespace NUMINAMATH_GPT_trigonometric_identity_l712_71211

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (2 * Real.sin θ - 4 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l712_71211


namespace NUMINAMATH_GPT_determine_a_perpendicular_l712_71218

theorem determine_a_perpendicular 
  (a : ℝ)
  (h1 : 2 * x + 3 * y + 5 = 0)
  (h2 : a * x + 3 * y - 4 = 0) 
  (h_perpendicular : ∀ x y, (2 * x + 3 * y + 5 = 0) → ∀ x y, (a * x + 3 * y - 4 = 0) → (-(2 : ℝ) / (3 : ℝ)) * (-(a : ℝ) / (3 : ℝ)) = -1) :
  a = -9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_perpendicular_l712_71218


namespace NUMINAMATH_GPT_problem_statement_l712_71202

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a + b + c + 2 = a * b * c) :
  (a+1) * (b+1) * (c+1) ≥ 27 ∧ ((a+1) * (b+1) * (c+1) = 27 → a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l712_71202


namespace NUMINAMATH_GPT_intersection_M_N_l712_71219

def is_M (x : ℝ) : Prop := x^2 + x - 6 < 0
def is_N (x : ℝ) : Prop := abs (x - 1) <= 2

theorem intersection_M_N : {x : ℝ | is_M x} ∩ {x : ℝ | is_N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l712_71219


namespace NUMINAMATH_GPT_reservoir_solution_l712_71204

theorem reservoir_solution (x y z : ℝ) :
  8 * (1 / x - 1 / y) = 1 →
  24 * (1 / x - 1 / y - 1 / z) = 1 →
  8 * (1 / y + 1 / z) = 1 →
  x = 8 ∧ y = 24 ∧ z = 12 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_reservoir_solution_l712_71204


namespace NUMINAMATH_GPT_possible_values_of_a_l712_71208

theorem possible_values_of_a (a : ℝ) : (2 < a ∧ a < 3 ∨ 3 < a ∧ a < 5) → (a = 5/2 ∨ a = 4) := 
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l712_71208


namespace NUMINAMATH_GPT_ratio_sum_eq_seven_eight_l712_71243

theorem ratio_sum_eq_seven_eight 
  (a b c x y z : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7/8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sum_eq_seven_eight_l712_71243


namespace NUMINAMATH_GPT_num_primes_with_squares_in_range_l712_71258

/-- There are exactly 6 prime numbers whose squares are between 2500 and 5500. -/
theorem num_primes_with_squares_in_range : 
  ∃ primes : Finset ℕ, 
    (∀ p ∈ primes, Prime p) ∧
    (∀ p ∈ primes, 2500 < p^2 ∧ p^2 < 5500) ∧
    primes.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_primes_with_squares_in_range_l712_71258


namespace NUMINAMATH_GPT_problem1_problem2_l712_71220

def prop_p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem1 (a : ℝ) (h_a : a = 1) (h_pq : ∃ x, prop_p x a ∧ prop_q x) :
  ∃ x, 2 < x ∧ x < 3 :=
by sorry

theorem problem2 (h_qp : ∀ x (a : ℝ), prop_q x → prop_p x a) :
  ∃ a, 1 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l712_71220


namespace NUMINAMATH_GPT_second_hand_degree_per_minute_l712_71216

theorem second_hand_degree_per_minute :
  (∀ (t : ℝ), t = 60 → 360 / t = 6) :=
by
  intro t
  intro ht
  rw [ht]
  norm_num

end NUMINAMATH_GPT_second_hand_degree_per_minute_l712_71216


namespace NUMINAMATH_GPT_problem_statement_l712_71205

def T (m : ℕ) : ℕ := sorry
def H (m : ℕ) : ℕ := sorry

def p (m k : ℕ) : ℝ := 
  if k % 2 = 1 then 0 else sorry

theorem problem_statement (m : ℕ) : p m 0 ≥ p (m + 1) 0 := sorry

end NUMINAMATH_GPT_problem_statement_l712_71205


namespace NUMINAMATH_GPT_tyrone_gave_15_marbles_l712_71249

variables (x : ℕ)

-- Define initial conditions for Tyrone and Eric
def initial_tyrone := 120
def initial_eric := 20

-- Define the condition after giving marbles
def condition_after_giving (x : ℕ) := 120 - x = 3 * (20 + x)

theorem tyrone_gave_15_marbles (x : ℕ) : condition_after_giving x → x = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tyrone_gave_15_marbles_l712_71249


namespace NUMINAMATH_GPT_obtuse_angles_in_second_quadrant_l712_71225

theorem obtuse_angles_in_second_quadrant
  (θ : ℝ) 
  (is_obtuse : θ > 90 ∧ θ < 180) :
  90 < θ ∧ θ < 180 :=
by sorry

end NUMINAMATH_GPT_obtuse_angles_in_second_quadrant_l712_71225


namespace NUMINAMATH_GPT_adrian_water_amount_l712_71215

theorem adrian_water_amount
  (O S W : ℕ) 
  (h1 : S = 3 * O)
  (h2 : W = 5 * S)
  (h3 : O = 4) : W = 60 :=
by
  sorry

end NUMINAMATH_GPT_adrian_water_amount_l712_71215


namespace NUMINAMATH_GPT_maize_storage_l712_71289

theorem maize_storage (x : ℝ)
  (h1 : 24 * x - 5 + 8 = 27) : x = 1 :=
  sorry

end NUMINAMATH_GPT_maize_storage_l712_71289


namespace NUMINAMATH_GPT_quadratic_equation_with_given_roots_l712_71282

theorem quadratic_equation_with_given_roots :
  (∃ (x : ℝ), (x - 3) * (x + 4) = 0 ↔ x = 3 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_with_given_roots_l712_71282


namespace NUMINAMATH_GPT_total_rectangles_l712_71267

-- Definitions
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4
def exclude_line_pair: ℕ := 1
def total_combinations (n m : ℕ) : ℕ := Nat.choose n m

-- Statement
theorem total_rectangles (h_lines : ℕ) (v_lines : ℕ) 
  (exclude_pair : ℕ) (valid_h_comb : ℕ) (valid_v_comb : ℕ) :
  h_lines = horizontal_lines →
  v_lines = vertical_lines →
  exclude_pair = exclude_line_pair →
  valid_h_comb = total_combinations 5 2 - exclude_pair →
  valid_v_comb = total_combinations 4 2 →
  valid_h_comb * valid_v_comb = 54 :=
by intros; sorry

end NUMINAMATH_GPT_total_rectangles_l712_71267


namespace NUMINAMATH_GPT_cut_into_four_and_reassemble_l712_71264

-- Definitions as per conditions in the problem
def figureArea : ℕ := 36
def nParts : ℕ := 4
def squareArea (s : ℕ) : ℕ := s * s

-- Property to be proved
theorem cut_into_four_and_reassemble :
  ∃ (s : ℕ), squareArea s = figureArea / nParts ∧ s * s = figureArea :=
by
  sorry

end NUMINAMATH_GPT_cut_into_four_and_reassemble_l712_71264


namespace NUMINAMATH_GPT_barnyard_owl_hoots_per_minute_l712_71230

theorem barnyard_owl_hoots_per_minute :
  (20 - 5) / 3 = 5 := 
by
  sorry

end NUMINAMATH_GPT_barnyard_owl_hoots_per_minute_l712_71230


namespace NUMINAMATH_GPT_sum_of_squares_eq_l712_71273

theorem sum_of_squares_eq :
  (1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 + 1005^2 + 1006^2) = 7042091 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_squares_eq_l712_71273


namespace NUMINAMATH_GPT_parallel_lines_slope_l712_71298

theorem parallel_lines_slope (b : ℝ) 
  (h₁ : ∀ x y : ℝ, 3 * y - 3 * b = 9 * x → (b = 3 - 9)) 
  (h₂ : ∀ x y : ℝ, y + 2 = (b + 9) * x → (b = 3 - 9)) : b = -6 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l712_71298


namespace NUMINAMATH_GPT_right_triangle_longer_leg_l712_71292

theorem right_triangle_longer_leg (a b c : ℕ) (h₀ : a^2 + b^2 = c^2) (h₁ : c = 65) (h₂ : a < b) : b = 60 :=
sorry

end NUMINAMATH_GPT_right_triangle_longer_leg_l712_71292


namespace NUMINAMATH_GPT_smallest_factor_to_end_with_four_zeros_l712_71228

theorem smallest_factor_to_end_with_four_zeros :
  ∃ x : ℕ, (975 * 935 * 972 * x) % 10000 = 0 ∧
           (∀ y : ℕ, (975 * 935 * 972 * y) % 10000 = 0 → x ≤ y) ∧
           x = 20 := by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_smallest_factor_to_end_with_four_zeros_l712_71228


namespace NUMINAMATH_GPT_range_of_a_l712_71271

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) ↔ 1 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l712_71271


namespace NUMINAMATH_GPT_strawberry_harvest_l712_71259

theorem strawberry_harvest
  (length : ℕ) (width : ℕ)
  (plants_per_sqft : ℕ) (yield_per_plant : ℕ)
  (garden_area : ℕ := length * width) 
  (total_plants : ℕ := plants_per_sqft * garden_area) 
  (expected_strawberries : ℕ := yield_per_plant * total_plants) :
  length = 10 ∧ width = 12 ∧ plants_per_sqft = 5 ∧ yield_per_plant = 8 → 
  expected_strawberries = 4800 := by
  sorry

end NUMINAMATH_GPT_strawberry_harvest_l712_71259


namespace NUMINAMATH_GPT_sandy_books_second_shop_l712_71261

theorem sandy_books_second_shop (x : ℕ) (h1 : 65 = 1080 / 16) 
                                (h2 : x * 16 = 840) 
                                (h3 : (1080 + 840) / 16 = 120) : 
                                x = 55 :=
by
  sorry

end NUMINAMATH_GPT_sandy_books_second_shop_l712_71261


namespace NUMINAMATH_GPT_six_letter_words_no_substring_amc_l712_71280

theorem six_letter_words_no_substring_amc : 
  let alphabet := ['A', 'M', 'C']
  let totalNumberOfWords := 3^6
  let numberOfWordsContainingAMC := 4 * 3^3 - 1
  let numberOfWordsNotContainingAMC := totalNumberOfWords - numberOfWordsContainingAMC
  numberOfWordsNotContainingAMC = 622 :=
by
  sorry

end NUMINAMATH_GPT_six_letter_words_no_substring_amc_l712_71280


namespace NUMINAMATH_GPT_johns_share_is_1100_l712_71235

def total_amount : ℕ := 6600
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6
def total_parts : ℕ := ratio_john + ratio_jose + ratio_binoy
def value_per_part : ℚ := total_amount / total_parts
def amount_received_by_john : ℚ := value_per_part * ratio_john

theorem johns_share_is_1100 : amount_received_by_john = 1100 := by
  sorry

end NUMINAMATH_GPT_johns_share_is_1100_l712_71235


namespace NUMINAMATH_GPT_power_fraction_example_l712_71234

theorem power_fraction_example : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := 
by
  sorry

end NUMINAMATH_GPT_power_fraction_example_l712_71234


namespace NUMINAMATH_GPT_number_that_multiplies_b_l712_71293

variable (a b x : ℝ)

theorem number_that_multiplies_b (h1 : 7 * a = x * b) (h2 : a * b ≠ 0) (h3 : (a / 8) / (b / 7) = 1) : x = 8 := 
sorry

end NUMINAMATH_GPT_number_that_multiplies_b_l712_71293


namespace NUMINAMATH_GPT_cone_volume_l712_71287

noncomputable def radius_of_sector : ℝ := 6
noncomputable def arc_length_of_sector : ℝ := (1 / 2) * (2 * Real.pi * radius_of_sector)
noncomputable def radius_of_base : ℝ := arc_length_of_sector / (2 * Real.pi)
noncomputable def slant_height : ℝ := radius_of_sector
noncomputable def height_of_cone : ℝ := Real.sqrt (slant_height^2 - radius_of_base^2)
noncomputable def volume_of_cone : ℝ := (1 / 3) * Real.pi * (radius_of_base^2) * height_of_cone

theorem cone_volume : volume_of_cone = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_cone_volume_l712_71287


namespace NUMINAMATH_GPT_numbers_to_be_left_out_l712_71221

axiom problem_conditions :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  let grid_numbers := [1, 9, 14, 5]
  numbers.sum + grid_numbers.sum = 106 ∧
  ∃ (left_out : ℕ) (remaining_numbers : List ℕ),
    numbers.erase left_out = remaining_numbers ∧
    (numbers.sum + grid_numbers.sum - left_out) = 96 ∧
    remaining_numbers.length = 8

theorem numbers_to_be_left_out :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  10 ∈ numbers ∧
  let grid_numbers := [1, 9, 14, 5]
  let total_sum := numbers.sum + grid_numbers.sum
  let grid_sum := total_sum - 10
  grid_sum % 12 = 0 ∧
  grid_sum = 96 :=
sorry

end NUMINAMATH_GPT_numbers_to_be_left_out_l712_71221


namespace NUMINAMATH_GPT_remainder_of_k_l712_71274

theorem remainder_of_k {k : ℕ} (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 8 = 7) (h4 : k % 11 = 3) (h5 : k < 168) :
  k % 13 = 8 := 
sorry

end NUMINAMATH_GPT_remainder_of_k_l712_71274


namespace NUMINAMATH_GPT_mars_mission_cost_per_person_l712_71232

theorem mars_mission_cost_per_person
  (total_cost : ℕ) (number_of_people : ℕ)
  (h1 : total_cost = 50000000000) (h2 : number_of_people = 500000000) :
  (total_cost / number_of_people) = 100 := 
by
  sorry

end NUMINAMATH_GPT_mars_mission_cost_per_person_l712_71232


namespace NUMINAMATH_GPT_part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l712_71266

section Problem

-- Universal set is ℝ
def universal_set : Set ℝ := Set.univ

-- Set A
def set_A : Set ℝ := { x | x^2 - x - 6 ≤ 0 }

-- Set A complement in ℝ
def complement_A : Set ℝ := universal_set \ set_A

-- Set B
def set_B : Set ℝ := { x | (x - 4)/(x + 1) < 0 }

-- Set C
def set_C (a : ℝ) : Set ℝ := { x | 2 - a < x ∧ x < 2 + a }

-- Prove (complement_A ∩ set_B = (3, 4))
theorem part1 : (complement_A ∩ set_B) = { x | 3 < x ∧ x < 4 } :=
  sorry

-- Assume a definition for real number a (non-negative)
variable (a : ℝ)

-- Prove range of a given the conditions
-- Condition 1: A ∩ C = C implies a ≤ 1
theorem condition1_implies_a_le_1 (h : set_A ∩ set_C a = set_C a) : a ≤ 1 :=
  sorry

-- Condition 2: B ∪ C = B implies a ≤ 2
theorem condition2_implies_a_le_2 (h : set_B ∪ set_C a = set_B) : a ≤ 2 :=
  sorry

-- Condition 3: C ⊆ (A ∩ B) implies a ≤ 1
theorem condition3_implies_a_le_1 (h : set_C a ⊆ set_A ∩ set_B) : a ≤ 1 :=
  sorry

end Problem

end NUMINAMATH_GPT_part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l712_71266


namespace NUMINAMATH_GPT_range_of_a_l712_71246

-- Definitions of sets A and B
def A (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| < 2
def B (x a : ℝ) : Prop := x^2 - (a + 1) * x + a < 0

-- The condition A ∩ B ≠ ∅
def nonempty_intersection (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a

-- Proving the required range of a
theorem range_of_a : {a : ℝ | nonempty_intersection a} = {a : ℝ | a < 1 ∨ a > 3} := by
  sorry

end NUMINAMATH_GPT_range_of_a_l712_71246


namespace NUMINAMATH_GPT_balance_blue_balls_l712_71206

noncomputable def weight_balance (G B Y W : ℝ) : ℝ :=
  3 * G + 3 * Y + 5 * W

theorem balance_blue_balls (G B Y W : ℝ)
  (hG : G = 2 * B)
  (hY : Y = 2 * B)
  (hW : W = (5 / 3) * B) :
  weight_balance G B Y W = (61 / 3) * B :=
by
  sorry

end NUMINAMATH_GPT_balance_blue_balls_l712_71206


namespace NUMINAMATH_GPT_initial_percentage_water_l712_71288

theorem initial_percentage_water (P : ℝ) (H1 : 150 * P / 100 + 10 = 40) : P = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_water_l712_71288


namespace NUMINAMATH_GPT_simplify_radicals_l712_71201

theorem simplify_radicals (q : ℝ) (hq : 0 < q) :
  (Real.sqrt (42 * q)) * (Real.sqrt (7 * q)) * (Real.sqrt (14 * q)) = 98 * q * Real.sqrt (3 * q) :=
by
  sorry

end NUMINAMATH_GPT_simplify_radicals_l712_71201


namespace NUMINAMATH_GPT_equal_values_on_plane_l712_71260

theorem equal_values_on_plane (f : ℤ × ℤ → ℕ)
    (h_avg : ∀ (i j : ℤ), f (i, j) = (f (i+1, j) + f (i-1, j) + f (i, j+1) + f (i, j-1)) / 4) :
  ∃ c : ℕ, ∀ (i j : ℤ), f (i, j) = c :=
by
  sorry

end NUMINAMATH_GPT_equal_values_on_plane_l712_71260


namespace NUMINAMATH_GPT_travel_time_l712_71299

theorem travel_time (distance speed : ℕ) (h_distance : distance = 810) (h_speed : speed = 162) :
  distance / speed = 5 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_l712_71299


namespace NUMINAMATH_GPT_parabola_units_shift_l712_71214

noncomputable def parabola_expression (A B : ℝ × ℝ) (x : ℝ) : ℝ :=
  let b := -5
  let c := 6
  x^2 + b * x + c

theorem parabola_units_shift (A B : ℝ × ℝ) (x : ℝ) (y : ℝ) :
  A = (2, 0) → B = (0, 6) → parabola_expression A B 4 = 2 →
  (y - 2 = 0) → true :=
by
  intro hA hB h4 hy
  sorry

end NUMINAMATH_GPT_parabola_units_shift_l712_71214


namespace NUMINAMATH_GPT_coffee_ratio_correct_l712_71226

noncomputable def ratio_of_guests (cups_weak : ℕ) (cups_strong : ℕ) (tablespoons_weak : ℕ) (tablespoons_strong : ℕ) (total_tablespoons : ℕ) : ℤ :=
  if (cups_weak * tablespoons_weak + cups_strong * tablespoons_strong = total_tablespoons) then
    (cups_weak * tablespoons_weak / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong)) /
    (cups_strong * tablespoons_strong / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong))
  else 0

theorem coffee_ratio_correct :
  ratio_of_guests 12 12 1 2 36 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_coffee_ratio_correct_l712_71226


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l712_71276

def A := {x : ℝ | x^2 - 5 * x + 6 > 0}
def B := {x : ℝ | x / (x - 1) < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l712_71276


namespace NUMINAMATH_GPT_katie_candy_l712_71275

theorem katie_candy (K : ℕ) (H1 : K + 6 - 9 = 7) : K = 10 :=
by
  sorry

end NUMINAMATH_GPT_katie_candy_l712_71275


namespace NUMINAMATH_GPT_parabola_chord_length_l712_71229

theorem parabola_chord_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) (h2 : y2^2 = 4 * x2) 
  (hx : x1 + x2 = 9) 
  (focus_line : ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b → y^2 = 4 * x) :
  |(x1 - 1, y1) - (x2 - 1, y2)| = 11 := 
sorry

end NUMINAMATH_GPT_parabola_chord_length_l712_71229


namespace NUMINAMATH_GPT_books_per_author_l712_71251

theorem books_per_author (total_books : ℕ) (authors : ℕ) (h1 : total_books = 198) (h2 : authors = 6) : total_books / authors = 33 :=
by sorry

end NUMINAMATH_GPT_books_per_author_l712_71251


namespace NUMINAMATH_GPT_complex_number_equality_l712_71242

theorem complex_number_equality (i : ℂ) (h : i^2 = -1) : 1 + i + i^2 = i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_equality_l712_71242


namespace NUMINAMATH_GPT_math_problem_l712_71210

variable {f : ℝ → ℝ}

theorem math_problem (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                     (h2 : ∀ x : ℝ, x > 0 → f x > 0)
                     (h3 : f 1 = 2) :
                     f 0 = 0 ∧
                     (∀ x : ℝ, f (-x) = -f x) ∧
                     (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧
                     (∃ a : ℝ, f (2 - a) = 6 ∧ a = -1) := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l712_71210


namespace NUMINAMATH_GPT_saved_percent_correct_l712_71284

noncomputable def price_kit : ℝ := 144.20
noncomputable def price1 : ℝ := 21.75
noncomputable def price2 : ℝ := 18.60
noncomputable def price3 : ℝ := 23.80
noncomputable def price4 : ℝ := 29.35

noncomputable def total_price_individual : ℝ := 2 * price1 + 2 * price2 + price3 + 2 * price4
noncomputable def amount_saved : ℝ := total_price_individual - price_kit
noncomputable def percent_saved : ℝ := 100 * (amount_saved / total_price_individual)

theorem saved_percent_correct : percent_saved = 11.64 := by
  sorry

end NUMINAMATH_GPT_saved_percent_correct_l712_71284


namespace NUMINAMATH_GPT_range_of_cars_l712_71256

def fuel_vehicle_cost_per_km (x : ℕ) : ℚ := (40 * 9) / x
def new_energy_vehicle_cost_per_km (x : ℕ) : ℚ := (60 * 0.6) / x

theorem range_of_cars : ∃ x : ℕ, fuel_vehicle_cost_per_km x = new_energy_vehicle_cost_per_km x + 0.54 ∧ x = 600 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_cars_l712_71256


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l712_71279

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a^2 - 9 * a + 18 = 0) (h2 : b^2 - 9 * b + 18 = 0) (h3 : a ≠ b) :
  a + 2 * b = 15 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l712_71279


namespace NUMINAMATH_GPT_Yoongi_has_fewest_apples_l712_71278

def Jungkook_apples : Nat := 6 * 3
def Yoongi_apples : Nat := 4
def Yuna_apples : Nat := 5

theorem Yoongi_has_fewest_apples :
  Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end NUMINAMATH_GPT_Yoongi_has_fewest_apples_l712_71278


namespace NUMINAMATH_GPT_reduced_price_after_discount_l712_71203

theorem reduced_price_after_discount (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 1500 / R - 1500 / P = 10) :
  R = 30 := 
by
  sorry

end NUMINAMATH_GPT_reduced_price_after_discount_l712_71203


namespace NUMINAMATH_GPT_minimum_value_of_expression_l712_71262

theorem minimum_value_of_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l712_71262


namespace NUMINAMATH_GPT_sibling_age_difference_l712_71245

theorem sibling_age_difference (Y : ℝ) (Y_eq : Y = 25.75) (avg_age_eq : (Y + (Y + 3) + (Y + 6) + (Y + x)) / 4 = 30) : (Y + 6) - Y = 6 :=
by
  sorry

end NUMINAMATH_GPT_sibling_age_difference_l712_71245


namespace NUMINAMATH_GPT_unique_positive_real_solution_of_polynomial_l712_71244

theorem unique_positive_real_solution_of_polynomial :
  ∃! x : ℝ, x > 0 ∧ (x^11 + 8 * x^10 + 15 * x^9 + 1000 * x^8 - 1200 * x^7 = 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_real_solution_of_polynomial_l712_71244


namespace NUMINAMATH_GPT_michael_made_small_balls_l712_71254

def num_small_balls (total_bands : ℕ) (bands_per_small : ℕ) (bands_per_large : ℕ) (num_large : ℕ) : ℕ :=
  (total_bands - num_large * bands_per_large) / bands_per_small

theorem michael_made_small_balls :
  num_small_balls 5000 50 300 13 = 22 :=
by
  sorry

end NUMINAMATH_GPT_michael_made_small_balls_l712_71254


namespace NUMINAMATH_GPT_business_ownership_l712_71296

variable (x : ℝ) (total_value : ℝ)
variable (fraction_sold : ℝ)
variable (sale_amount : ℝ)

-- Conditions
axiom total_value_condition : total_value = 10000
axiom fraction_sold_condition : fraction_sold = 3 / 5
axiom sale_amount_condition : sale_amount = 2000
axiom equation_condition : (fraction_sold * x * total_value = sale_amount)

theorem business_ownership : x = 1 / 3 := by 
  have hv := total_value_condition
  have hf := fraction_sold_condition
  have hs := sale_amount_condition
  have he := equation_condition
  sorry

end NUMINAMATH_GPT_business_ownership_l712_71296


namespace NUMINAMATH_GPT_tan_neg_1140_eq_neg_sqrt3_l712_71253

theorem tan_neg_1140_eq_neg_sqrt3 
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_periodicity : ∀ θ : ℝ, ∀ n : ℤ, Real.tan (θ + n * 180) = Real.tan θ)
  (tan_60 : Real.tan 60 = Real.sqrt 3) :
  Real.tan (-1140) = -Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_tan_neg_1140_eq_neg_sqrt3_l712_71253


namespace NUMINAMATH_GPT_empty_seats_after_second_stop_l712_71233

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end NUMINAMATH_GPT_empty_seats_after_second_stop_l712_71233


namespace NUMINAMATH_GPT_water_inflow_rate_in_tank_A_l712_71291

-- Definitions from the conditions
def capacity := 20
def inflow_rate_B := 4
def extra_time_A := 5

-- Target variable
noncomputable def inflow_rate_A : ℕ :=
  let time_B := capacity / inflow_rate_B
  let time_A := time_B + extra_time_A
  capacity / time_A

-- Hypotheses
def tank_capacity : capacity = 20 := rfl
def tank_B_inflow : inflow_rate_B = 4 := rfl
def tank_A_extra_time : extra_time_A = 5 := rfl

-- Theorem statement
theorem water_inflow_rate_in_tank_A : inflow_rate_A = 2 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_water_inflow_rate_in_tank_A_l712_71291


namespace NUMINAMATH_GPT_perpendicular_vectors_x_value_l712_71297

theorem perpendicular_vectors_x_value:
  ∀ (x : ℝ), let a : ℝ × ℝ := (1, 2)
             let b : ℝ × ℝ := (x, 1)
             (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 :=
by
  intro x
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_value_l712_71297


namespace NUMINAMATH_GPT_rectangle_diagonals_equal_l712_71250

-- Define the properties of a rectangle
def is_rectangle (AB CD AD BC : ℝ) (diagonal1 diagonal2 : ℝ) : Prop :=
  AB = CD ∧ AD = BC ∧ diagonal1 = diagonal2

-- State the theorem to prove that the diagonals of a rectangle are equal
theorem rectangle_diagonals_equal (AB CD AD BC diagonal1 diagonal2 : ℝ) (h : is_rectangle AB CD AD BC diagonal1 diagonal2) :
  diagonal1 = diagonal2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonals_equal_l712_71250


namespace NUMINAMATH_GPT_area_of_triangle_KDC_l712_71217

open Real

noncomputable def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem area_of_triangle_KDC
  (radius : ℝ) (chord_length : ℝ) (seg_KA : ℝ)
  (OX distance_DY : ℝ)
  (parallel : ∀ (PA PB : ℝ), PA = PB)
  (collinear : ∀ (PK PA PQ PB : ℝ), PK + PA + PQ + PB = PK + PQ + PA + PB)
  (hyp_radius : radius = 10)
  (hyp_chord_length : chord_length = 12)
  (hyp_seg_KA : seg_KA = 24)
  (hyp_OX : OX = 8)
  (hyp_distance_DY : distance_DY = 8) :
  triangle_area chord_length distance_DY = 48 :=
  by
  sorry

end NUMINAMATH_GPT_area_of_triangle_KDC_l712_71217


namespace NUMINAMATH_GPT_hearty_beads_count_l712_71209

theorem hearty_beads_count :
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  total_beads = 320 :=
by
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  show total_beads = 320
  sorry

end NUMINAMATH_GPT_hearty_beads_count_l712_71209


namespace NUMINAMATH_GPT_remainder_when_xy_div_by_22_l712_71248

theorem remainder_when_xy_div_by_22
  (x y : ℤ)
  (h1 : x % 126 = 37)
  (h2 : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
  sorry

end NUMINAMATH_GPT_remainder_when_xy_div_by_22_l712_71248


namespace NUMINAMATH_GPT_product_lcm_gcd_eq_2160_l712_71237

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end NUMINAMATH_GPT_product_lcm_gcd_eq_2160_l712_71237


namespace NUMINAMATH_GPT_repair_time_l712_71295

theorem repair_time {x : ℝ} :
  (∀ (a b : ℝ), a = 3 ∧ b = 6 → (((1 / a) + (1 / b)) * x = 1) → x = 2) :=
by
  intros a b hab h
  rcases hab with ⟨ha, hb⟩
  sorry

end NUMINAMATH_GPT_repair_time_l712_71295


namespace NUMINAMATH_GPT_lambda_sum_ellipse_l712_71270

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4)

noncomputable def intersects_y_axis (k : ℝ) : ℝ × ℝ :=
  (0, -4 * k)

noncomputable def lambda1 (x1 : ℝ) : ℝ :=
  x1 / (4 - x1)

noncomputable def lambda2 (x2 : ℝ) : ℝ :=
  x2 / (4 - x2)

theorem lambda_sum_ellipse {k x1 x2 : ℝ}
  (h1 : ellipse x1 (k * (x1 - 4)))
  (h2 : ellipse x2 (k * (x2 - 4)))
  (h3 : line_through_focus k x1 (k * (x1 - 4)))
  (h4 : line_through_focus k x2 (k * (x2 - 4))) :
  lambda1 x1 + lambda2 x2 = -50 / 9 := 
sorry

end NUMINAMATH_GPT_lambda_sum_ellipse_l712_71270


namespace NUMINAMATH_GPT_hours_worked_each_day_l712_71269

-- Definitions based on problem conditions
def total_hours_worked : ℝ := 8.0
def number_of_days_worked : ℝ := 4.0

-- Theorem statement to prove the number of hours worked each day
theorem hours_worked_each_day :
  total_hours_worked / number_of_days_worked = 2.0 :=
sorry

end NUMINAMATH_GPT_hours_worked_each_day_l712_71269


namespace NUMINAMATH_GPT_kids_meals_sold_l712_71247

theorem kids_meals_sold (x y : ℕ) (h1 : x / y = 2) (h2 : x + y = 12) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_kids_meals_sold_l712_71247


namespace NUMINAMATH_GPT_faster_by_airplane_l712_71200

theorem faster_by_airplane : 
  let driving_time := 3 * 60 + 15 
  let airport_drive := 10
  let wait_to_board := 20
  let flight_duration := driving_time / 3
  let exit_plane := 10
  driving_time - (airport_drive + wait_to_board + flight_duration + exit_plane) = 90 := 
by
  let driving_time : ℕ := 3 * 60 + 15
  let airport_drive : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_duration : ℕ := driving_time / 3
  let exit_plane : ℕ := 10
  have h1 : driving_time = 195 := rfl
  have h2 : flight_duration = 65 := by norm_num [h1]
  have h3 : 195 - (10 + 20 + 65 + 10) = 195 - 105 := by norm_num
  have h4 : 195 - 105 = 90 := by norm_num
  exact h4

end NUMINAMATH_GPT_faster_by_airplane_l712_71200


namespace NUMINAMATH_GPT_xy_sufficient_not_necessary_l712_71238

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy_lt_zero : x * y < 0) → abs (x - y) = abs x + abs y ∧ (abs (x - y) = abs x + abs y → x * y ≥ 0) := 
by
  sorry

end NUMINAMATH_GPT_xy_sufficient_not_necessary_l712_71238


namespace NUMINAMATH_GPT_probability_of_same_suit_l712_71223

-- Definitions for the conditions
def total_cards : ℕ := 52
def suits : ℕ := 4
def cards_per_suit : ℕ := 13
def total_draws : ℕ := 2

-- Definition of factorial for binomial coefficient calculation
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Calculation of the probability
def prob_same_suit : ℚ :=
  let ways_to_choose_2_cards_from_52 := binomial_coeff total_cards total_draws
  let ways_to_choose_2_cards_per_suit := binomial_coeff cards_per_suit total_draws
  let total_ways_to_choose_2_same_suit := suits * ways_to_choose_2_cards_per_suit
  total_ways_to_choose_2_same_suit / ways_to_choose_2_cards_from_52

theorem probability_of_same_suit :
  prob_same_suit = 4 / 17 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_same_suit_l712_71223


namespace NUMINAMATH_GPT_find_f_of_2_l712_71240

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x - b

theorem find_f_of_2 (a b : ℝ) (h_pos : 0 < a)
  (h1 : ∀ x : ℝ, a * f x a b - b = 4 * x - 3)
  : f 2 a b = 3 := 
sorry

end NUMINAMATH_GPT_find_f_of_2_l712_71240


namespace NUMINAMATH_GPT_dividend_from_tonys_stock_l712_71277

theorem dividend_from_tonys_stock (investment price_per_share total_income : ℝ) 
  (h1 : investment = 3200) (h2 : price_per_share = 85) (h3 : total_income = 250) : 
  (total_income / (investment / price_per_share)) = 6.76 :=
by 
  sorry

end NUMINAMATH_GPT_dividend_from_tonys_stock_l712_71277


namespace NUMINAMATH_GPT_travel_cost_is_correct_l712_71227

-- Definitions of the conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 60
def road_width : ℝ := 15
def cost_per_sq_m : ℝ := 3

-- Areas of individual roads
def area_road_length := road_width * lawn_breadth
def area_road_breadth := road_width * lawn_length
def intersection_area := road_width * road_width

-- Adjusted area for roads discounting intersection area
def total_area_roads := area_road_length + area_road_breadth - intersection_area

-- Total cost of traveling the roads
def total_cost := total_area_roads * cost_per_sq_m

theorem travel_cost_is_correct : total_cost = 5625 := by
  sorry

end NUMINAMATH_GPT_travel_cost_is_correct_l712_71227


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_n_value_l712_71224

theorem ellipse_hyperbola_tangent_n_value :
  (∃ n : ℝ, (∀ x y : ℝ, 4 * x^2 + y^2 = 4 ∧ x^2 - n * (y - 1)^2 = 1) ↔ n = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_n_value_l712_71224


namespace NUMINAMATH_GPT_midpoint_on_hyperbola_l712_71285

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_on_hyperbola_l712_71285


namespace NUMINAMATH_GPT_find_speed_of_stream_l712_71283

def boat_speeds (V_b V_s : ℝ) : Prop :=
  V_b + V_s = 10 ∧ V_b - V_s = 8

theorem find_speed_of_stream (V_b V_s : ℝ) (h : boat_speeds V_b V_s) : V_s = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_stream_l712_71283


namespace NUMINAMATH_GPT_min_value_problem_l712_71241

theorem min_value_problem (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 57 * a + 88 * b + 125 * c ≥ 1148) :
  240 ≤ a^3 + b^3 + c^3 + 5 * a^2 + 5 * b^2 + 5 * c^2 :=
sorry

end NUMINAMATH_GPT_min_value_problem_l712_71241


namespace NUMINAMATH_GPT_withdrawal_amount_in_2008_l712_71222

noncomputable def total_withdrawal (a : ℕ) (p : ℝ) : ℝ :=
  (a / p) * ((1 + p) - (1 + p)^8)

theorem withdrawal_amount_in_2008 (a : ℕ) (p : ℝ) (h_pos : 0 < p) (h_neg_one_lt : -1 < p) :
  total_withdrawal a p = (a / p) * ((1 + p) - (1 + p)^8) :=
by
  -- Conditions
  -- Starting from May 10th, 2001, multiple annual deposits.
  -- Annual interest rate p > 0 and p > -1.
  sorry

end NUMINAMATH_GPT_withdrawal_amount_in_2008_l712_71222


namespace NUMINAMATH_GPT_total_paintable_area_is_2006_l712_71263

-- Define the dimensions of the bedrooms and the hallway
def bedroom_length := 14
def bedroom_width := 11
def bedroom_height := 9

def hallway_length := 20
def hallway_width := 7
def hallway_height := 9

def num_bedrooms := 4
def doorway_window_area := 70

-- Compute the areas of the bedroom walls and the hallway walls
def bedroom_wall_area : ℕ :=
  2 * (bedroom_length * bedroom_height) +
  2 * (bedroom_width * bedroom_height)

def paintable_bedroom_wall_area : ℕ :=
  bedroom_wall_area - doorway_window_area

def total_paintable_bedroom_area : ℕ :=
  num_bedrooms * paintable_bedroom_wall_area

def hallway_wall_area : ℕ :=
  2 * (hallway_length * hallway_height) +
  2 * (hallway_width * hallway_height)

-- Compute the total paintable area
def total_paintable_area : ℕ :=
  total_paintable_bedroom_area + hallway_wall_area

-- Theorem stating the total paintable area is 2006 sq ft
theorem total_paintable_area_is_2006 : total_paintable_area = 2006 := 
  by
    unfold total_paintable_area
    rw [total_paintable_bedroom_area, paintable_bedroom_wall_area, bedroom_wall_area]
    rw [hallway_wall_area]
    norm_num
    sorry -- Proof omitted

end NUMINAMATH_GPT_total_paintable_area_is_2006_l712_71263


namespace NUMINAMATH_GPT_find_px_value_l712_71239

noncomputable def p (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_px_value {a b c : ℤ} 
  (h1 : p a b c 2 = 2) 
  (h2 : p a b c (-2) = -2) 
  (h3 : p a b c 9 = 3) 
  (h : a = -2 / 11) 
  (h4 : b = 1)
  (h5 : c = 8 / 11) :
  p a b c 14 = -230 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_px_value_l712_71239


namespace NUMINAMATH_GPT_ratio_is_five_ninths_l712_71255

-- Define the conditions
def total_profit : ℕ := 48000
def total_income : ℕ := 108000

-- Define the total spending based on conditions
def total_spending : ℕ := total_income - total_profit

-- Define the ratio of spending to income
def ratio_spending_to_income : ℚ := total_spending / total_income

-- The theorem we need to prove
theorem ratio_is_five_ninths : ratio_spending_to_income = 5 / 9 := 
  sorry

end NUMINAMATH_GPT_ratio_is_five_ninths_l712_71255


namespace NUMINAMATH_GPT_total_stamps_is_38_l712_71252

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end NUMINAMATH_GPT_total_stamps_is_38_l712_71252


namespace NUMINAMATH_GPT_solve_for_x_l712_71272

theorem solve_for_x (x : ℤ) : 3 * (5 - x) = 9 → x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l712_71272


namespace NUMINAMATH_GPT_object_speed_conversion_l712_71268

theorem object_speed_conversion 
  (distance : ℝ)
  (velocity : ℝ) 
  (conversion_factor : ℝ) 
  (distance_in_km : ℝ)
  (time_in_seconds : ℝ) 
  (time_in_minutes : ℝ) 
  (speed_in_kmh : ℝ) :
  distance = 200 ∧ 
  velocity = 1/3 ∧ 
  time_in_seconds = distance / velocity ∧ 
  time_in_minutes = time_in_seconds / 60 ∧ 
  conversion_factor = 3600 * 0.001 ∧ 
  speed_in_kmh = velocity * conversion_factor ↔ 
  speed_in_kmh = 0.4 :=
by sorry

end NUMINAMATH_GPT_object_speed_conversion_l712_71268
