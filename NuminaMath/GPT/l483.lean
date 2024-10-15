import Mathlib

namespace NUMINAMATH_GPT_quadratic_function_condition_l483_48368

theorem quadratic_function_condition (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
  sorry

end NUMINAMATH_GPT_quadratic_function_condition_l483_48368


namespace NUMINAMATH_GPT_pairs_characterization_l483_48363

noncomputable def valid_pairs (A : ℝ) : Set (ℕ × ℕ) :=
  { p | ∃ x : ℝ, x > 0 ∧ (1 + x) ^ p.1 = (1 + A * x) ^ p.2 }

theorem pairs_characterization (A : ℝ) (hA : A > 1) :
  valid_pairs A = { p | p.2 < p.1 ∧ p.1 < A * p.2 } :=
by
  sorry

end NUMINAMATH_GPT_pairs_characterization_l483_48363


namespace NUMINAMATH_GPT_line_eqn_with_given_conditions_l483_48343

theorem line_eqn_with_given_conditions : 
  ∃(m c : ℝ), (∀ x y : ℝ, y = m*x + c → x + y - 3 = 0) ↔ 
  ∀ x y, x + y = 3 :=
sorry

end NUMINAMATH_GPT_line_eqn_with_given_conditions_l483_48343


namespace NUMINAMATH_GPT_maximum_value_expression_l483_48395

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_expression_l483_48395


namespace NUMINAMATH_GPT_seating_arrangement_correct_l483_48318

noncomputable def seating_arrangements_around_table : Nat :=
  7

def B_G_next_to_C (A B C D E F G : Prop) (d : Nat) : Prop :=
  d = 48

theorem seating_arrangement_correct : ∃ d, d = 48 := sorry

end NUMINAMATH_GPT_seating_arrangement_correct_l483_48318


namespace NUMINAMATH_GPT_minimum_b_l483_48321

theorem minimum_b (k a b : ℝ) (h1 : 1 < k) (h2 : k < a) (h3 : a < b)
  (h4 : ¬(k + a > b)) (h5 : ¬(1/a + 1/b > 1/k)) :
  2 * k ≤ b :=
by
  sorry

end NUMINAMATH_GPT_minimum_b_l483_48321


namespace NUMINAMATH_GPT_price_per_liter_after_discount_l483_48306

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end NUMINAMATH_GPT_price_per_liter_after_discount_l483_48306


namespace NUMINAMATH_GPT_max_triangle_perimeter_l483_48319

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_max_triangle_perimeter_l483_48319


namespace NUMINAMATH_GPT_green_pens_l483_48341

theorem green_pens (blue_pens green_pens : ℕ) (ratio_blue_to_green : blue_pens / green_pens = 4 / 3) (total_blue : blue_pens = 16) : green_pens = 12 :=
by sorry

end NUMINAMATH_GPT_green_pens_l483_48341


namespace NUMINAMATH_GPT_total_boys_l483_48358

theorem total_boys (T F : ℕ) 
  (avg_all : 37 * T = 39 * 110 + 15 * F) 
  (total_eq : T = 110 + F) : 
  T = 120 := 
sorry

end NUMINAMATH_GPT_total_boys_l483_48358


namespace NUMINAMATH_GPT_exists_common_element_l483_48337

variable (S : Fin 2011 → Set ℤ)
variable (h1 : ∀ i, (S i).Nonempty)
variable (h2 : ∀ i j, (S i ∩ S j).Nonempty)

theorem exists_common_element :
  ∃ a : ℤ, ∀ i, a ∈ S i :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_common_element_l483_48337


namespace NUMINAMATH_GPT_lashawn_three_times_kymbrea_l483_48307

-- Definitions based on the conditions
def kymbrea_collection (months : ℕ) : ℕ := 50 + 3 * months
def lashawn_collection (months : ℕ) : ℕ := 20 + 5 * months

-- Theorem stating the core of the problem
theorem lashawn_three_times_kymbrea (x : ℕ) 
  (h : lashawn_collection x = 3 * kymbrea_collection x) : x = 33 := 
sorry

end NUMINAMATH_GPT_lashawn_three_times_kymbrea_l483_48307


namespace NUMINAMATH_GPT_average_age_is_25_l483_48320

theorem average_age_is_25 (A B C : ℝ) (h_avg_ac : (A + C) / 2 = 29) (h_b : B = 17) :
  (A + B + C) / 3 = 25 := 
  by
    sorry

end NUMINAMATH_GPT_average_age_is_25_l483_48320


namespace NUMINAMATH_GPT_smallest_n_l483_48365

theorem smallest_n (n : ℕ) : 
  (n % 6 = 2) ∧ (n % 7 = 3) ∧ (n % 8 = 4) → n = 8 :=
  by sorry

end NUMINAMATH_GPT_smallest_n_l483_48365


namespace NUMINAMATH_GPT_diagonal_BD_size_cos_A_value_l483_48389

noncomputable def AB := 250
noncomputable def CD := 250
noncomputable def angle_A := 120
noncomputable def angle_C := 120
noncomputable def AD := 150
noncomputable def BC := 150
noncomputable def perimeter := 800

/-- The size of the diagonal BD in isosceles trapezoid ABCD is 350, given the conditions -/
theorem diagonal_BD_size (AB CD AD BC : ℕ) (angle_A angle_C : ℝ) :
  AB = 250 → CD = 250 → AD = 150 → BC = 150 →
  angle_A = 120 → angle_C = 120 →
  ∃ BD : ℝ, BD = 350 :=
by
  sorry

/-- The cosine of angle A is -0.5, given the angle is 120 degrees -/
theorem cos_A_value (angle_A : ℝ) :
  angle_A = 120 → ∃ cos_A : ℝ, cos_A = -0.5 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_BD_size_cos_A_value_l483_48389


namespace NUMINAMATH_GPT_area_of_triangle_AEC_l483_48338

theorem area_of_triangle_AEC (BE EC : ℝ) (h_ratio : BE / EC = 3 / 2) (area_abe : ℝ) (h_area_abe : area_abe = 27) : 
  ∃ area_aec, area_aec = 18 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_AEC_l483_48338


namespace NUMINAMATH_GPT_evaluate_expression_l483_48305

-- Definitions for conditions
def x := (1 / 4 : ℚ)
def y := (1 / 2 : ℚ)
def z := (3 : ℚ)

-- Statement of the problem
theorem evaluate_expression : 
  4 * (x^3 * y^2 * z^2) = 9 / 64 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l483_48305


namespace NUMINAMATH_GPT_vincent_total_cost_l483_48391

theorem vincent_total_cost :
  let day1_packs := 15
  let day1_pack_cost := 2.50
  let discount_percent := 0.10
  let day2_packs := 25
  let day2_pack_cost := 3.00
  let tax_percent := 0.05
  let day1_total_cost_before_discount := day1_packs * day1_pack_cost
  let day1_discount_amount := discount_percent * day1_total_cost_before_discount
  let day1_total_cost_after_discount := day1_total_cost_before_discount - day1_discount_amount
  let day2_total_cost_before_tax := day2_packs * day2_pack_cost
  let day2_tax_amount := tax_percent * day2_total_cost_before_tax
  let day2_total_cost_after_tax := day2_total_cost_before_tax + day2_tax_amount
  let total_cost := day1_total_cost_after_discount + day2_total_cost_after_tax
  total_cost = 112.50 :=
by 
  -- Mathlib can be used for floating point calculations, if needed
  -- For the purposes of this example, we assume calculations are correct.
  sorry

end NUMINAMATH_GPT_vincent_total_cost_l483_48391


namespace NUMINAMATH_GPT_rational_sum_is_negative_then_at_most_one_positive_l483_48304

theorem rational_sum_is_negative_then_at_most_one_positive (a b : ℚ) (h : a + b < 0) :
  (a > 0 ∧ b ≤ 0) ∨ (a ≤ 0 ∧ b > 0) ∨ (a ≤ 0 ∧ b ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_rational_sum_is_negative_then_at_most_one_positive_l483_48304


namespace NUMINAMATH_GPT_lines_intersect_not_perpendicular_l483_48399

noncomputable def slopes_are_roots (m k1 k2 : ℝ) : Prop :=
  k1^2 + m*k1 - 2 = 0 ∧ k2^2 + m*k2 - 2 = 0

theorem lines_intersect_not_perpendicular (m k1 k2 : ℝ) (h : slopes_are_roots m k1 k2) : (k1 * k2 = -2 ∧ k1 ≠ k2) → ∃ l1 l2 : ℝ, l1 ≠ l2 ∧ l1 = k1 ∧ l2 = k2 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_not_perpendicular_l483_48399


namespace NUMINAMATH_GPT_problem1_problem2_l483_48336

section ProofProblems

-- Definitions for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Prove that n! = binom(n, k) * k! * (n-k)!
theorem problem1 (n k : ℕ) : n.factorial = binom n k * k.factorial * (n - k).factorial :=
by sorry

-- Problem 2: Prove that binom(n, k) = binom(n-1, k) + binom(n-1, k-1)
theorem problem2 (n k : ℕ) : binom n k = binom (n-1) k + binom (n-1) (k-1) :=
by sorry

end ProofProblems

end NUMINAMATH_GPT_problem1_problem2_l483_48336


namespace NUMINAMATH_GPT_smallest_number_of_eggs_l483_48350

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_eggs_l483_48350


namespace NUMINAMATH_GPT_find_constant_l483_48382

theorem find_constant (c : ℝ) (f : ℝ → ℝ)
  (h : f x = c * x^3 + 19 * x^2 - 4 * c * x + 20)
  (hx : f (-7) = 0) :
  c = 3 :=
sorry

end NUMINAMATH_GPT_find_constant_l483_48382


namespace NUMINAMATH_GPT_calculate_y_l483_48316

theorem calculate_y (x : ℤ) (y : ℤ) (h1 : x = 121) (h2 : 2 * x - y = 102) : y = 140 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_calculate_y_l483_48316


namespace NUMINAMATH_GPT_new_tax_rate_l483_48325

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end NUMINAMATH_GPT_new_tax_rate_l483_48325


namespace NUMINAMATH_GPT_complex_coordinates_l483_48364

theorem complex_coordinates (i : ℂ) (z : ℂ) (h : i^2 = -1) (h_z : z = (1 + 2 * i^3) / (2 + i)) :
  z = -i := 
by {
  sorry
}

end NUMINAMATH_GPT_complex_coordinates_l483_48364


namespace NUMINAMATH_GPT_sqrt_product_eq_l483_48323

theorem sqrt_product_eq :
  (Int.sqrt (2 ^ 2 * 3 ^ 4) : ℤ) = 18 :=
sorry

end NUMINAMATH_GPT_sqrt_product_eq_l483_48323


namespace NUMINAMATH_GPT_sqrt_diff_eq_neg_sixteen_l483_48300

theorem sqrt_diff_eq_neg_sixteen : 
  (Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2)) = -16 := 
  sorry

end NUMINAMATH_GPT_sqrt_diff_eq_neg_sixteen_l483_48300


namespace NUMINAMATH_GPT_smallest_h_divisible_by_primes_l483_48354

theorem smallest_h_divisible_by_primes :
  ∃ h k : ℕ, (∀ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p > 8 ∧ q > 11 ∧ r > 24 → (h + k) % (p * q * r) = 0 ∧ h = 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_h_divisible_by_primes_l483_48354


namespace NUMINAMATH_GPT_scientific_notation_of_0_00000012_l483_48333

theorem scientific_notation_of_0_00000012 :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_00000012_l483_48333


namespace NUMINAMATH_GPT_solve_equation_l483_48379

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l483_48379


namespace NUMINAMATH_GPT_fraction_product_simplified_l483_48356

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end NUMINAMATH_GPT_fraction_product_simplified_l483_48356


namespace NUMINAMATH_GPT_cubic_sum_divisible_by_9_l483_48384

theorem cubic_sum_divisible_by_9 (n : ℕ) (hn : n > 0) : 
  ∃ k, n^3 + (n+1)^3 + (n+2)^3 = 9*k := by
  sorry

end NUMINAMATH_GPT_cubic_sum_divisible_by_9_l483_48384


namespace NUMINAMATH_GPT_trip_distance_l483_48339

theorem trip_distance (D : ℝ) (t1 t2 : ℝ) :
  (30 / 60 = t1) →
  (70 / 35 = t2) →
  (t1 + t2 = 2.5) →
  (40 = D / (t1 + t2)) →
  D = 100 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_trip_distance_l483_48339


namespace NUMINAMATH_GPT_two_n_plus_m_value_l483_48361

theorem two_n_plus_m_value (n m : ℤ) :
  3 * n - m < 5 ∧ n + m > 26 ∧ 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
sorry

end NUMINAMATH_GPT_two_n_plus_m_value_l483_48361


namespace NUMINAMATH_GPT_charlie_metal_storage_l483_48377

theorem charlie_metal_storage (total_needed : ℕ) (amount_to_buy : ℕ) (storage : ℕ) 
    (h1 : total_needed = 635) 
    (h2 : amount_to_buy = 359) 
    (h3 : total_needed = storage + amount_to_buy) : 
    storage = 276 := 
sorry

end NUMINAMATH_GPT_charlie_metal_storage_l483_48377


namespace NUMINAMATH_GPT_inheritance_amount_l483_48347

-- Definitions based on conditions given
def inheritance (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_federal := x - federal_tax
  let state_tax := 0.15 * remaining_after_federal
  let total_tax := federal_tax + state_tax
  total_tax = 15000

-- The statement to be proven
theorem inheritance_amount (x : ℝ) (hx : inheritance x) : x = 41379 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inheritance_amount_l483_48347


namespace NUMINAMATH_GPT_abs_diff_31st_term_l483_48335

-- Define the sequences C and D
def C (n : ℕ) : ℤ := 40 + 20 * (n - 1)
def D (n : ℕ) : ℤ := 40 - 20 * (n - 1)

-- Question: What is the absolute value of the difference between the 31st term of C and D?
theorem abs_diff_31st_term : |C 31 - D 31| = 1200 := by
  sorry

end NUMINAMATH_GPT_abs_diff_31st_term_l483_48335


namespace NUMINAMATH_GPT_area_excluding_hole_correct_l483_48326

def large_rectangle_area (x: ℝ) : ℝ :=
  4 * (x + 7) * (x + 5)

def hole_area (x: ℝ) : ℝ :=
  9 * (2 * x - 3) * (x - 2)

def area_excluding_hole (x: ℝ) : ℝ :=
  large_rectangle_area x - hole_area x

theorem area_excluding_hole_correct (x: ℝ) :
  area_excluding_hole x = -14 * x^2 + 111 * x + 86 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_area_excluding_hole_correct_l483_48326


namespace NUMINAMATH_GPT_inequality_proof_l483_48387

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3)^(1 / 8) - 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l483_48387


namespace NUMINAMATH_GPT_competitive_exam_candidates_l483_48311

theorem competitive_exam_candidates (x : ℝ)
  (A_selected : ℝ := 0.06 * x) 
  (B_selected : ℝ := 0.07 * x) 
  (h : B_selected = A_selected + 81) :
  x = 8100 := by
  sorry

end NUMINAMATH_GPT_competitive_exam_candidates_l483_48311


namespace NUMINAMATH_GPT_base_nine_to_mod_five_l483_48397

-- Define the base-nine number N
def N : ℕ := 2 * 9^10 + 7 * 9^9 + 0 * 9^8 + 0 * 9^7 + 6 * 9^6 + 0 * 9^5 + 0 * 9^4 + 0 * 9^3 + 0 * 9^2 + 5 * 9^1 + 2 * 9^0

-- Theorem statement
theorem base_nine_to_mod_five : N % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_base_nine_to_mod_five_l483_48397


namespace NUMINAMATH_GPT_constant_expression_l483_48301

theorem constant_expression 
  (x y : ℝ) 
  (h₁ : x + y = 1) 
  (h₂ : x ≠ 1) 
  (h₃ : y ≠ 1) : 
  (x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3)) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_constant_expression_l483_48301


namespace NUMINAMATH_GPT_problem_statement_l483_48369

def f (x : ℝ) : ℝ := x^2 - 3 * x + 6

def g (x : ℝ) : ℝ := x + 4

theorem problem_statement : f (g 3) - g (f 3) = 24 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l483_48369


namespace NUMINAMATH_GPT_shaded_figure_perimeter_l483_48314

theorem shaded_figure_perimeter (a b : ℝ) (area_overlap : ℝ) (side_length : ℝ) (side_length_overlap : ℝ):
    a = 5 → b = 5 → area_overlap = 4 → side_length_overlap * side_length_overlap = area_overlap →
    side_length_overlap = 2 →
    ((4 * a) + (4 * b) - (4 * side_length_overlap)) = 32 :=
by
  intros
  sorry

end NUMINAMATH_GPT_shaded_figure_perimeter_l483_48314


namespace NUMINAMATH_GPT_n_squared_divisible_by_12_l483_48394

theorem n_squared_divisible_by_12 (n : ℕ) : 12 ∣ n^2 * (n^2 - 1) :=
  sorry

end NUMINAMATH_GPT_n_squared_divisible_by_12_l483_48394


namespace NUMINAMATH_GPT_greatest_multiple_of_5_and_7_less_than_800_l483_48303

theorem greatest_multiple_of_5_and_7_less_than_800 : 
    ∀ n : ℕ, (n < 800 ∧ 35 ∣ n) → n ≤ 770 := 
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_5_and_7_less_than_800_l483_48303


namespace NUMINAMATH_GPT_evaluate_expression_l483_48328

theorem evaluate_expression :
  let a := (1 : ℚ) / 5
  let b := (1 : ℚ) / 3
  let c := (3 : ℚ) / 7
  let d := (1 : ℚ) / 4
  (a + b) / (c - d) = 224 / 75 := by
sorry

end NUMINAMATH_GPT_evaluate_expression_l483_48328


namespace NUMINAMATH_GPT_poly_has_one_positive_and_one_negative_root_l483_48386

theorem poly_has_one_positive_and_one_negative_root :
  ∃! r1, r1 > 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) ∧ 
  ∃! r2, r2 < 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) := by
sorry

end NUMINAMATH_GPT_poly_has_one_positive_and_one_negative_root_l483_48386


namespace NUMINAMATH_GPT_number_of_ways_to_select_starting_lineup_l483_48324

noncomputable def choose (n k : ℕ) : ℕ := 
if h : k ≤ n then Nat.choose n k else 0

theorem number_of_ways_to_select_starting_lineup (n k : ℕ) (h : n = 12) (h1 : k = 5) : 
  12 * choose 11 4 = 3960 := 
by sorry

end NUMINAMATH_GPT_number_of_ways_to_select_starting_lineup_l483_48324


namespace NUMINAMATH_GPT_inequality_holds_for_all_reals_l483_48367

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_reals_l483_48367


namespace NUMINAMATH_GPT_absolute_value_expression_l483_48376

theorem absolute_value_expression {x : ℤ} (h : x = 2024) :
  abs (abs (abs x - x) - abs x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_expression_l483_48376


namespace NUMINAMATH_GPT_single_intersection_not_necessarily_tangent_l483_48398

structure Hyperbola where
  -- Placeholder for hyperbola properties
  axis1 : Real
  axis2 : Real

def is_tangent (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for tangency
  ∃ p : Real × Real, l = { p }

def is_parallel_to_asymptote (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for parallelism to asymptote 
  ∃ A : Real, l = { (x, A * x) | x : Real }

theorem single_intersection_not_necessarily_tangent
  (l : Set (Real × Real)) (H : Hyperbola) (h : ∃ p : Real × Real, l = { p }) :
  ¬ is_tangent l H ∨ is_parallel_to_asymptote l H :=
sorry

end NUMINAMATH_GPT_single_intersection_not_necessarily_tangent_l483_48398


namespace NUMINAMATH_GPT_nonnegative_interval_l483_48345

theorem nonnegative_interval (x : ℝ) : 
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3) ≥ 0 ↔ (x ≥ 0 ∧ x < 3) :=
by sorry

end NUMINAMATH_GPT_nonnegative_interval_l483_48345


namespace NUMINAMATH_GPT_probability_of_circle_in_square_l483_48309

open Real Set

theorem probability_of_circle_in_square :
  ∃ (p : ℝ), (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → (x^2 + y^2 < 1/4) → True)
  → p = π / 16 :=
by
  use π / 16
  sorry

end NUMINAMATH_GPT_probability_of_circle_in_square_l483_48309


namespace NUMINAMATH_GPT_tangent_line_ratio_l483_48392

variables {x1 x2 : ℝ}

theorem tangent_line_ratio (h1 : 2 * x1 = 3 * x2^2) (h2 : x1^2 = 2 * x2^3) : (x1 / x2) = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_tangent_line_ratio_l483_48392


namespace NUMINAMATH_GPT_angelina_speed_l483_48344

theorem angelina_speed (v : ℝ) (h₁ : ∀ t : ℝ, t = 100 / v) (h₂ : ∀ t : ℝ, t = 180 / (2 * v)) 
  (h₃ : ∀ d t : ℝ, 100 / v - 40 = 180 / (2 * v)) : 
  2 * v = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_angelina_speed_l483_48344


namespace NUMINAMATH_GPT_files_to_organize_in_afternoon_l483_48329

-- Defining the given conditions.
def initial_files : ℕ := 60
def files_organized_in_the_morning : ℕ := initial_files / 2
def missing_files_in_the_afternoon : ℕ := 15

-- The theorem to prove:
theorem files_to_organize_in_afternoon : 
  files_organized_in_the_morning + missing_files_in_the_afternoon = initial_files / 2 →
  ∃ afternoon_files : ℕ, 
    afternoon_files = (initial_files - files_organized_in_the_morning) - missing_files_in_the_afternoon :=
by
  -- Proof will go here, skipping with sorry for now.
  sorry

end NUMINAMATH_GPT_files_to_organize_in_afternoon_l483_48329


namespace NUMINAMATH_GPT_elaine_rent_percentage_l483_48360

theorem elaine_rent_percentage (E : ℝ) (hE : E > 0) :
  let rent_last_year := 0.20 * E
  let earnings_this_year := 1.25 * E
  let rent_this_year := 0.30 * earnings_this_year
  (rent_this_year / rent_last_year) * 100 = 187.5 :=
by
  sorry

end NUMINAMATH_GPT_elaine_rent_percentage_l483_48360


namespace NUMINAMATH_GPT_watermelon_and_banana_weight_l483_48371

variables (w b : ℕ)
variables (h1 : 2 * w + b = 8100)
variables (h2 : 2 * w + 3 * b = 8300)

theorem watermelon_and_banana_weight (Hw : w = 4000) (Hb : b = 100) :
  2 * w + b = 8100 ∧ 2 * w + 3 * b = 8300 :=
by
  sorry

end NUMINAMATH_GPT_watermelon_and_banana_weight_l483_48371


namespace NUMINAMATH_GPT_calculate_F_5_f_6_l483_48372

def f (a : ℤ) : ℤ := a + 3

def F (a b : ℤ) : ℤ := b^3 - 2 * a

theorem calculate_F_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end NUMINAMATH_GPT_calculate_F_5_f_6_l483_48372


namespace NUMINAMATH_GPT_shaded_region_area_l483_48396

-- Define the problem conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def area_of_shaded_region : ℝ := 50

-- State the theorem to prove the area of the shaded region
theorem shaded_region_area (n : ℕ) (d : ℝ) (area : ℝ) (h1 : n = num_squares) (h2 : d = diagonal_length) : 
  area = area_of_shaded_region :=
sorry

end NUMINAMATH_GPT_shaded_region_area_l483_48396


namespace NUMINAMATH_GPT_flower_bed_width_l483_48373

theorem flower_bed_width (length area : ℝ) (h_length : length = 4) (h_area : area = 143.2) :
  area / length = 35.8 :=
by
  sorry

end NUMINAMATH_GPT_flower_bed_width_l483_48373


namespace NUMINAMATH_GPT_yura_picture_dimensions_l483_48302

-- Definitions based on the problem conditions
variable {a b : ℕ} -- dimensions of the picture
variable (hasFrame : ℕ × ℕ → Prop) -- definition sketch

-- The main statement to prove
theorem yura_picture_dimensions (h : (a + 2) * (b + 2) - a * b = 2 * a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
  sorry

end NUMINAMATH_GPT_yura_picture_dimensions_l483_48302


namespace NUMINAMATH_GPT_distance_between_points_l483_48375

theorem distance_between_points (a b c d m k : ℝ) 
  (h1 : b = 2 * m * a + k) (h2 : d = -m * c + k) : 
  (Real.sqrt ((c - a)^2 + (d - b)^2)) = Real.sqrt ((1 + m^2) * (c - a)^2) := 
by {
  sorry
}

end NUMINAMATH_GPT_distance_between_points_l483_48375


namespace NUMINAMATH_GPT_sandy_total_money_l483_48370

-- Definitions based on conditions
def X_initial (X : ℝ) : Prop := 
  X - 0.30 * X = 210

def watch_cost : ℝ := 50

-- Question translated into a proof goal
theorem sandy_total_money (X : ℝ) (h : X_initial X) : 
  X + watch_cost = 350 := by
  sorry

end NUMINAMATH_GPT_sandy_total_money_l483_48370


namespace NUMINAMATH_GPT_parallel_lines_l483_48312

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end NUMINAMATH_GPT_parallel_lines_l483_48312


namespace NUMINAMATH_GPT_least_possible_value_of_c_l483_48342

theorem least_possible_value_of_c (a b c : ℕ) 
  (h1 : a + b + c = 60) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : b = a + 13) : c = 45 :=
sorry

end NUMINAMATH_GPT_least_possible_value_of_c_l483_48342


namespace NUMINAMATH_GPT_group_d_forms_triangle_l483_48346

-- Definitions for the stick lengths in each group
def group_a := (1, 2, 6)
def group_b := (2, 2, 4)
def group_c := (1, 2, 3)
def group_d := (2, 3, 4)

-- Statement to prove that Group D can form a triangle
theorem group_d_forms_triangle (a b c : ℕ) : a = 2 → b = 3 → c = 4 → a + b > c ∧ a + c > b ∧ b + c > a := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  apply And.intro
  sorry
  apply And.intro
  sorry
  sorry

end NUMINAMATH_GPT_group_d_forms_triangle_l483_48346


namespace NUMINAMATH_GPT_incorrect_correlation_coefficient_range_l483_48385

noncomputable def regression_analysis_conditions 
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) : Prop :=
  non_deterministic_relationship ∧
  correlation_coefficient_range ∧
  perfect_correlation ∧
  correlation_coefficient_sign

theorem incorrect_correlation_coefficient_range
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) :
  regression_analysis_conditions 
    non_deterministic_relationship 
    correlation_coefficient_range 
    perfect_correlation 
    correlation_coefficient_sign →
  ¬ correlation_coefficient_range :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end NUMINAMATH_GPT_incorrect_correlation_coefficient_range_l483_48385


namespace NUMINAMATH_GPT_sally_earnings_l483_48331

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_sally_earnings_l483_48331


namespace NUMINAMATH_GPT_functional_eq_l483_48388

theorem functional_eq (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end NUMINAMATH_GPT_functional_eq_l483_48388


namespace NUMINAMATH_GPT_increasing_sequence_nec_but_not_suf_l483_48349

theorem increasing_sequence_nec_but_not_suf (a : ℕ → ℝ) :
  (∀ n, abs (a (n + 1)) > a n) → (∀ n, a (n + 1) > a n) ↔ 
  ∃ (n : ℕ), ¬ (abs (a (n + 1)) > a n) ∧ (a (n + 1) > a n) :=
sorry

end NUMINAMATH_GPT_increasing_sequence_nec_but_not_suf_l483_48349


namespace NUMINAMATH_GPT_cheese_cost_l483_48378

theorem cheese_cost (bread_cost cheese_cost total_paid total_change coin_change nickels_value : ℝ) 
                    (quarter dime nickels_count : ℕ)
                    (h1 : bread_cost = 4.20)
                    (h2 : total_paid = 7.00)
                    (h3 : quarter = 1)
                    (h4 : dime = 1)
                    (h5 : nickels_count = 8)
                    (h6 : coin_change = (quarter * 0.25) + (dime * 0.10) + (nickels_count * 0.05))
                    (h7 : total_change = total_paid - bread_cost)
                    (h8 : cheese_cost = total_change - coin_change) :
                    cheese_cost = 2.05 :=
by {
    sorry
}

end NUMINAMATH_GPT_cheese_cost_l483_48378


namespace NUMINAMATH_GPT_choir_girls_count_l483_48351

noncomputable def number_of_girls_in_choir (o b t c b_boys : ℕ) : ℕ :=
  c - b_boys

theorem choir_girls_count (o b t b_boys : ℕ) (h1 : o = 20) (h2 : b = 2 * o) (h3 : t = 88)
  (h4 : b_boys = 12) : number_of_girls_in_choir o b t (t - (o + b)) b_boys = 16 :=
by
  sorry

end NUMINAMATH_GPT_choir_girls_count_l483_48351


namespace NUMINAMATH_GPT_degree_of_d_l483_48348

noncomputable def f : Polynomial ℝ := sorry
noncomputable def d : Polynomial ℝ := sorry
noncomputable def q : Polynomial ℝ := sorry
noncomputable def r : Polynomial ℝ := 5 * Polynomial.X^2 + 3 * Polynomial.X - 8

axiom deg_f : f.degree = 15
axiom deg_q : q.degree = 7
axiom deg_r : r.degree = 2
axiom poly_div : f = d * q + r

theorem degree_of_d : d.degree = 8 :=
by
  sorry

end NUMINAMATH_GPT_degree_of_d_l483_48348


namespace NUMINAMATH_GPT_leftover_balls_when_placing_60_in_tetrahedral_stack_l483_48310

def tetrahedral_number (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) / 6

/--
  When placing 60 balls in a tetrahedral stack, the number of leftover balls is 4.
-/
theorem leftover_balls_when_placing_60_in_tetrahedral_stack :
  ∃ n, tetrahedral_number n ≤ 60 ∧ 60 - tetrahedral_number n = 4 := by
  sorry

end NUMINAMATH_GPT_leftover_balls_when_placing_60_in_tetrahedral_stack_l483_48310


namespace NUMINAMATH_GPT_sum_of_non_domain_elements_l483_48315

theorem sum_of_non_domain_elements :
    let f (x : ℝ) : ℝ := 1 / (1 + 1 / (1 + 1 / (1 + 1 / x)))
    let is_not_in_domain (x : ℝ) := x = 0 ∨ x = -1 ∨ x = -1/2 ∨ x = -2/3
    (0 : ℝ) + (-1) + (-1/2) + (-2/3) = -19/6 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_non_domain_elements_l483_48315


namespace NUMINAMATH_GPT_hunter_time_comparison_l483_48355

-- Definitions for time spent in swamp, forest, and highway
variables {a b c : ℝ}

-- Given conditions
-- 1. Total time equation
#check a + b + c = 4

-- 2. Total distance equation
#check 2 * a + 4 * b + 6 * c = 17

-- Prove that the hunter spent more time on the highway than in the swamp
theorem hunter_time_comparison (h1 : a + b + c = 4) (h2 : 2 * a + 4 * b + 6 * c = 17) : c > a :=
by sorry

end NUMINAMATH_GPT_hunter_time_comparison_l483_48355


namespace NUMINAMATH_GPT_fraction_sum_equals_decimal_l483_48366

theorem fraction_sum_equals_decimal : 
  (3 / 30 + 9 / 300 + 27 / 3000 = 0.139) :=
by sorry

end NUMINAMATH_GPT_fraction_sum_equals_decimal_l483_48366


namespace NUMINAMATH_GPT_sum_zero_implies_product_terms_nonpositive_l483_48332

theorem sum_zero_implies_product_terms_nonpositive (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_zero_implies_product_terms_nonpositive_l483_48332


namespace NUMINAMATH_GPT_window_dimensions_l483_48308

-- Given conditions
def panes := 12
def rows := 3
def columns := 4
def height_to_width_ratio := 3
def border_width := 2

-- Definitions based on given conditions
def width_per_pane (x : ℝ) := x
def height_per_pane (x : ℝ) := 3 * x

def total_width (x : ℝ) := columns * width_per_pane x + (columns + 1) * border_width
def total_height (x : ℝ) := rows * height_per_pane x + (rows + 1) * border_width

-- Theorem statement: width and height of the window
theorem window_dimensions (x : ℝ) : 
  total_width x = 4 * x + 10 ∧ 
  total_height x = 9 * x + 8 := by
  sorry

end NUMINAMATH_GPT_window_dimensions_l483_48308


namespace NUMINAMATH_GPT_min_cubes_are_three_l483_48340

/-- 
  A toy construction set consists of cubes, each with one button on one side and socket holes on the other five sides.
  Prove that the minimum number of such cubes required to build a structure where all buttons are hidden, and only the sockets are visible is 3.
--/

def min_cubes_to_hide_buttons (num_cubes : ℕ) : Prop :=
  num_cubes = 3

theorem min_cubes_are_three : ∃ (n : ℕ), (∀ (num_buttons : ℕ), min_cubes_to_hide_buttons num_buttons) :=
by
  use 3
  sorry

end NUMINAMATH_GPT_min_cubes_are_three_l483_48340


namespace NUMINAMATH_GPT_cos_half_angle_quadrant_l483_48359

theorem cos_half_angle_quadrant 
  (α : ℝ) 
  (h1 : 25 * Real.sin α ^ 2 + Real.sin α - 24 = 0) 
  (h2 : π / 2 < α ∧ α < π) 
  : Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_half_angle_quadrant_l483_48359


namespace NUMINAMATH_GPT_probability_third_attempt_success_l483_48327

noncomputable def P_xi_eq_3 : ℚ :=
  (4 / 5) * (3 / 4) * (1 / 3)

theorem probability_third_attempt_success :
  P_xi_eq_3 = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_third_attempt_success_l483_48327


namespace NUMINAMATH_GPT_f_is_odd_l483_48383

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2 * x

-- State the problem
theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

end NUMINAMATH_GPT_f_is_odd_l483_48383


namespace NUMINAMATH_GPT_sum_of_positive_numbers_is_360_l483_48317

variable (x y : ℝ)
variable (h1 : x * y = 50 * (x + y))
variable (h2 : x * y = 75 * (x - y))

theorem sum_of_positive_numbers_is_360 (hx : 0 < x) (hy : 0 < y) : x + y = 360 :=
by sorry

end NUMINAMATH_GPT_sum_of_positive_numbers_is_360_l483_48317


namespace NUMINAMATH_GPT_yangmei_1_yangmei_2i_yangmei_2ii_l483_48390

-- Problem 1: Prove that a = 20
theorem yangmei_1 (a : ℕ) (h : 160 * a + 270 * a = 8600) : a = 20 := by
  sorry

-- Problem 2 (i): Prove x = 44 and y = 36
theorem yangmei_2i (x y : ℕ) (h1 : 160 * x + 270 * y = 16760) (h2 : 8 * x + 18 * y = 1000) : x = 44 ∧ y = 36 := by
  sorry

-- Problem 2 (ii): Prove b = 9 or 18
theorem yangmei_2ii (m n b : ℕ) (h1 : 8 * (m + b) + 18 * n = 1000) (h2 : 160 * m + 270 * n = 16760) (h3 : 0 < b)
: b = 9 ∨ b = 18 := by
  sorry

end NUMINAMATH_GPT_yangmei_1_yangmei_2i_yangmei_2ii_l483_48390


namespace NUMINAMATH_GPT_smallest_integer_ratio_l483_48322

theorem smallest_integer_ratio (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (h_sum : x + y = 120) (h_even : x % 2 = 0) : ∃ (k : ℕ), k = x / y ∧ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_ratio_l483_48322


namespace NUMINAMATH_GPT_round_robin_points_change_l483_48362

theorem round_robin_points_change (n : ℕ) (athletes : Finset ℕ) (tournament1_scores tournament2_scores : ℕ → ℚ) :
  Finset.card athletes = 2 * n →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) ≥ n) →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) = n) :=
by
  sorry

end NUMINAMATH_GPT_round_robin_points_change_l483_48362


namespace NUMINAMATH_GPT_river_width_l483_48357

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : depth = 5 → flow_rate_kmph = 2 → volume_per_minute = 5833.333333333333 → 
  (volume_per_minute / ((flow_rate_kmph * 1000 / 60) * depth) = 35) :=
by 
  intros h_depth h_flow_rate h_volume
  sorry

end NUMINAMATH_GPT_river_width_l483_48357


namespace NUMINAMATH_GPT_yarn_cut_parts_l483_48352

-- Define the given conditions
def total_length : ℕ := 10
def crocheted_parts : ℕ := 3
def crocheted_length : ℕ := 6

-- The main problem statement
theorem yarn_cut_parts (total_length crocheted_parts crocheted_length : ℕ) (h1 : total_length = 10) (h2 : crocheted_parts = 3) (h3 : crocheted_length = 6) :
  (total_length / (crocheted_length / crocheted_parts)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_yarn_cut_parts_l483_48352


namespace NUMINAMATH_GPT_volume_of_prism_l483_48381

theorem volume_of_prism 
  (a b c : ℝ) 
  (h₁ : a * b = 51) 
  (h₂ : b * c = 52) 
  (h₃ : a * c = 53) 
  : (a * b * c) = 374 :=
by sorry

end NUMINAMATH_GPT_volume_of_prism_l483_48381


namespace NUMINAMATH_GPT_net_change_in_price_net_change_percentage_l483_48393

theorem net_change_in_price (P : ℝ) :
  0.80 * P * 1.55 - P = 0.24 * P :=
by sorry

theorem net_change_percentage (P : ℝ) :
  ((0.80 * P * 1.55 - P) / P) * 100 = 24 :=
by sorry


end NUMINAMATH_GPT_net_change_in_price_net_change_percentage_l483_48393


namespace NUMINAMATH_GPT_trig_identity_l483_48380

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l483_48380


namespace NUMINAMATH_GPT_car_distance_l483_48334

theorem car_distance (time_am_18 : ℕ) (time_car_48 : ℕ) (h : time_am_18 = time_car_48) : 
  let distance_am_18 := 18
  let distance_car_48 := 48
  let total_distance_am := 675
  let distance_ratio := (distance_am_18 : ℝ) / (distance_car_48 : ℝ)
  let distance_car := (total_distance_am : ℝ) * (distance_car_48 : ℝ) / (distance_am_18 : ℝ)
  distance_car = 1800 :=
by
  sorry

end NUMINAMATH_GPT_car_distance_l483_48334


namespace NUMINAMATH_GPT_billy_total_tickets_l483_48353

theorem billy_total_tickets :
  let ferris_wheel_rides := 7
  let bumper_car_rides := 3
  let roller_coaster_rides := 4
  let teacups_rides := 5
  let ferris_wheel_cost := 5
  let bumper_car_cost := 6
  let roller_coaster_cost := 8
  let teacups_cost := 4
  let total_ferris_wheel := ferris_wheel_rides * ferris_wheel_cost
  let total_bumper_cars := bumper_car_rides * bumper_car_cost
  let total_roller_coaster := roller_coaster_rides * roller_coaster_cost
  let total_teacups := teacups_rides * teacups_cost
  let total_tickets := total_ferris_wheel + total_bumper_cars + total_roller_coaster + total_teacups
  total_tickets = 105 := 
sorry

end NUMINAMATH_GPT_billy_total_tickets_l483_48353


namespace NUMINAMATH_GPT_stopped_clock_more_accurate_l483_48374

theorem stopped_clock_more_accurate (slow_correct_time_frequency : ℕ)
  (stopped_correct_time_frequency : ℕ)
  (h1 : slow_correct_time_frequency = 720)
  (h2 : stopped_correct_time_frequency = 2) :
  stopped_correct_time_frequency > slow_correct_time_frequency / 720 :=
by
  sorry

end NUMINAMATH_GPT_stopped_clock_more_accurate_l483_48374


namespace NUMINAMATH_GPT_midpoint_on_hyperbola_l483_48330

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end NUMINAMATH_GPT_midpoint_on_hyperbola_l483_48330


namespace NUMINAMATH_GPT_fraction_halfway_between_l483_48313

theorem fraction_halfway_between (a b : ℚ) (h₁ : a = 1 / 6) (h₂ : b = 2 / 5) : (a + b) / 2 = 17 / 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_halfway_between_l483_48313
