import Mathlib

namespace NUMINAMATH_GPT_bottles_more_than_apples_l202_20212

-- Definitions given in the conditions
def apples : ℕ := 36
def regular_soda_bottles : ℕ := 80
def diet_soda_bottles : ℕ := 54

-- Theorem statement representing the question
theorem bottles_more_than_apples : (regular_soda_bottles + diet_soda_bottles) - apples = 98 :=
by
  sorry

end NUMINAMATH_GPT_bottles_more_than_apples_l202_20212


namespace NUMINAMATH_GPT_slices_served_during_dinner_l202_20286

theorem slices_served_during_dinner (slices_lunch slices_total slices_dinner : ℕ)
  (h1 : slices_lunch = 7)
  (h2 : slices_total = 12)
  (h3 : slices_dinner = slices_total - slices_lunch) :
  slices_dinner = 5 := 
by 
  sorry

end NUMINAMATH_GPT_slices_served_during_dinner_l202_20286


namespace NUMINAMATH_GPT_largest_n_consecutive_product_l202_20271

theorem largest_n_consecutive_product (n : ℕ) : n = 0 ↔ (n! = (n+1) * (n+2) * (n+3) * (n+4) * (n+5)) := by
  sorry

end NUMINAMATH_GPT_largest_n_consecutive_product_l202_20271


namespace NUMINAMATH_GPT_evaluate_expression_l202_20291

noncomputable def a := Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def b := -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def d := -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 3 / 50 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l202_20291


namespace NUMINAMATH_GPT_adi_baller_prob_l202_20210

theorem adi_baller_prob (a b : ℕ) (p : ℝ) (h_prime: Nat.Prime a) (h_pos_b: 0 < b)
  (h_p: p = (1 / 2) ^ (1 / 35)) : a + b = 37 :=
sorry

end NUMINAMATH_GPT_adi_baller_prob_l202_20210


namespace NUMINAMATH_GPT_bronson_cost_per_bushel_is_12_l202_20231

noncomputable def cost_per_bushel 
  (sale_price_per_apple : ℝ := 0.40)
  (apples_per_bushel : ℕ := 48)
  (profit_from_100_apples : ℝ := 15)
  (number_of_apples_sold : ℕ := 100) 
  : ℝ :=
  let revenue := number_of_apples_sold * sale_price_per_apple
  let cost := revenue - profit_from_100_apples
  let number_of_bushels := (number_of_apples_sold : ℝ) / apples_per_bushel
  cost / number_of_bushels

theorem bronson_cost_per_bushel_is_12 :
  cost_per_bushel = 12 :=
by
  sorry

end NUMINAMATH_GPT_bronson_cost_per_bushel_is_12_l202_20231


namespace NUMINAMATH_GPT_continuity_f_at_3_l202_20256

noncomputable def f (x : ℝ) := if x ≤ 3 then 3 * x^2 - 5 else 18 * x - 32

theorem continuity_f_at_3 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x - f 3) < ε := by
  intro ε ε_pos
  use 1
  simp
  sorry

end NUMINAMATH_GPT_continuity_f_at_3_l202_20256


namespace NUMINAMATH_GPT_christina_walking_speed_l202_20254

-- Definitions based on the conditions
def initial_distance : ℝ := 150  -- Jack and Christina are 150 feet apart
def jack_speed : ℝ := 7  -- Jack's speed in feet per second
def lindy_speed : ℝ := 10  -- Lindy's speed in feet per second
def lindy_total_distance : ℝ := 100  -- Total distance Lindy travels

-- Proof problem: Prove that Christina's walking speed is 8 feet per second
theorem christina_walking_speed : 
  ∃ c : ℝ, (lindy_total_distance / lindy_speed) * jack_speed + (lindy_total_distance / lindy_speed) * c = initial_distance ∧ 
  c = 8 :=
by {
  use 8,
  sorry
}

end NUMINAMATH_GPT_christina_walking_speed_l202_20254


namespace NUMINAMATH_GPT_min_value_of_frac_expr_l202_20241

theorem min_value_of_frac_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 / a) + (2 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_min_value_of_frac_expr_l202_20241


namespace NUMINAMATH_GPT_perfect_square_trinomial_l202_20233

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x + a)^2) ∨ (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x - a)^2)) ↔ m = 5 ∨ m = -3 :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l202_20233


namespace NUMINAMATH_GPT_expression_simplifies_to_32_l202_20279

noncomputable def simplified_expression (a : ℝ) : ℝ :=
  8 / (1 + a^8) + 4 / (1 + a^4) + 2 / (1 + a^2) + 1 / (1 + a) + 1 / (1 - a)

theorem expression_simplifies_to_32 :
  simplified_expression (2^(-1/16 : ℝ)) = 32 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplifies_to_32_l202_20279


namespace NUMINAMATH_GPT_geometric_to_arithmetic_l202_20250

theorem geometric_to_arithmetic {a1 a2 a3 a4 q : ℝ}
  (hq : q ≠ 1)
  (geom_seq : a2 = a1 * q ∧ a3 = a1 * q^2 ∧ a4 = a1 * q^3)
  (arith_seq : (2 * a3 = a1 + a4 ∨ 2 * a2 = a1 + a4)) :
  q = (1 + Real.sqrt 5) / 2 ∨ q = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_to_arithmetic_l202_20250


namespace NUMINAMATH_GPT_solve_equation_l202_20223

theorem solve_equation : ∀ x : ℝ, 2 * x - 6 = 3 * x * (x - 3) ↔ (x = 3 ∨ x = 2 / 3) := by sorry

end NUMINAMATH_GPT_solve_equation_l202_20223


namespace NUMINAMATH_GPT_geom_series_first_term_l202_20229

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_first_term_l202_20229


namespace NUMINAMATH_GPT_distance_between_centers_same_side_distance_between_centers_opposite_side_l202_20234

open Real

noncomputable def distance_centers_same_side (r : ℝ) : ℝ := (r * (sqrt 6 + sqrt 2)) / 2

noncomputable def distance_centers_opposite_side (r : ℝ) : ℝ := (r * (sqrt 6 - sqrt 2)) / 2

theorem distance_between_centers_same_side (r : ℝ):
  ∃ dist, dist = distance_centers_same_side r :=
sorry

theorem distance_between_centers_opposite_side (r : ℝ):
  ∃ dist, dist = distance_centers_opposite_side r :=
sorry

end NUMINAMATH_GPT_distance_between_centers_same_side_distance_between_centers_opposite_side_l202_20234


namespace NUMINAMATH_GPT_max_ladder_height_reached_l202_20221

def distance_from_truck_to_building : ℕ := 5
def ladder_extension : ℕ := 13

theorem max_ladder_height_reached :
  (ladder_extension ^ 2 - distance_from_truck_to_building ^ 2) = 144 :=
by
  -- This is where the proof should go
  sorry

end NUMINAMATH_GPT_max_ladder_height_reached_l202_20221


namespace NUMINAMATH_GPT_inequality_solution_l202_20258

theorem inequality_solution (x : ℝ) : x > 0 ∧ (x^(1/3) < 3 - x) ↔ x < 3 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l202_20258


namespace NUMINAMATH_GPT_min_lit_bulbs_l202_20289

theorem min_lit_bulbs (n : ℕ) (h : n ≥ 1) : 
  ∃ rows cols, (rows ⊆ Finset.range n) ∧ (cols ⊆ Finset.range n) ∧ 
  (∀ i j, (i ∈ rows ∧ j ∈ cols) ↔ (i + j) % 2 = 1) ∧ 
  rows.card * (n - cols.card) + cols.card * (n - rows.card) = 2 * n - 2 :=
by sorry

end NUMINAMATH_GPT_min_lit_bulbs_l202_20289


namespace NUMINAMATH_GPT_sqrt_expression_value_l202_20283

theorem sqrt_expression_value :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_value_l202_20283


namespace NUMINAMATH_GPT_polynomial_expansion_l202_20209

-- Define the polynomial expressions
def poly1 (s : ℝ) : ℝ := 3 * s^3 - 4 * s^2 + 5 * s - 2
def poly2 (s : ℝ) : ℝ := 2 * s^2 - 3 * s + 4

-- Define the expanded form of the product of the two polynomials
def expanded_poly (s : ℝ) : ℝ :=
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8

-- The theorem to prove the equivalence
theorem polynomial_expansion (s : ℝ) :
  (poly1 s) * (poly2 s) = expanded_poly s :=
sorry -- proof goes here

end NUMINAMATH_GPT_polynomial_expansion_l202_20209


namespace NUMINAMATH_GPT_smallest_n_to_make_183_divisible_by_11_l202_20232

theorem smallest_n_to_make_183_divisible_by_11 : ∃ n : ℕ, 183 + n % 11 = 0 ∧ n = 4 :=
by
  have h1 : 183 % 11 = 7 := 
    sorry
  let n := 11 - (183 % 11)
  have h2 : 183 + n % 11 = 0 :=
    sorry
  exact ⟨n, h2, sorry⟩

end NUMINAMATH_GPT_smallest_n_to_make_183_divisible_by_11_l202_20232


namespace NUMINAMATH_GPT_zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l202_20298

theorem zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three :
  (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) :=
by
  sorry

end NUMINAMATH_GPT_zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l202_20298


namespace NUMINAMATH_GPT_minimum_value_l202_20228

noncomputable def polynomial_expr (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5

theorem minimum_value : ∃ x y : ℝ, (polynomial_expr x y = 8) := 
sorry

end NUMINAMATH_GPT_minimum_value_l202_20228


namespace NUMINAMATH_GPT_number_of_books_is_8_l202_20213

def books_and_albums (x y p_a p_b : ℕ) : Prop :=
  (x * p_b = 1056) ∧ (p_b = p_a + 100) ∧ (x = y + 6)

theorem number_of_books_is_8 (y p_a p_b : ℕ) (h : books_and_albums 8 y p_a p_b) : 8 = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_books_is_8_l202_20213


namespace NUMINAMATH_GPT_geom_seq_sum_seven_terms_l202_20267

-- Defining the conditions
def a0 : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 7

-- Definition for the sum of the first n terms in a geometric series
def geom_series_sum (a r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Statement to prove the sum of the first seven terms equals 1093/2187
theorem geom_seq_sum_seven_terms : geom_series_sum a0 r n = 1093 / 2187 := 
by 
  sorry

end NUMINAMATH_GPT_geom_seq_sum_seven_terms_l202_20267


namespace NUMINAMATH_GPT_series_sum_eq_five_l202_20239

open Nat Real

noncomputable def sum_series : ℝ := ∑' (n : ℕ), (2 * n ^ 2 - n) / (n * (n + 1) * (n + 2))

theorem series_sum_eq_five : sum_series = 5 :=
sorry

end NUMINAMATH_GPT_series_sum_eq_five_l202_20239


namespace NUMINAMATH_GPT_cos_alpha_minus_270_l202_20248

open Real

theorem cos_alpha_minus_270 (α : ℝ) : 
  sin (540 * (π / 180) + α) = -4 / 5 → cos (α - 270 * (π / 180)) = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_270_l202_20248


namespace NUMINAMATH_GPT_find_PQ_length_l202_20220

-- Defining the problem parameters
variables {X Y Z P Q R : Type}
variables (dXY dXZ dPQ dPR : ℝ)
variable (angle_common : ℝ)

-- Conditions:
def angle_XYZ_PQR_common : Prop :=
  angle_common = 150 ∧ 
  dXY = 10 ∧
  dXZ = 20 ∧
  dPQ = 5 ∧
  dPR = 12

-- Question: Prove PQ = 2.5 given the conditions
theorem find_PQ_length
  (h : angle_XYZ_PQR_common dXY dXZ dPQ dPR angle_common) :
  dPQ = 2.5 :=
sorry

end NUMINAMATH_GPT_find_PQ_length_l202_20220


namespace NUMINAMATH_GPT_john_needs_29_planks_for_house_wall_l202_20237

def total_number_of_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

theorem john_needs_29_planks_for_house_wall :
  total_number_of_planks 12 17 = 29 :=
by
  sorry

end NUMINAMATH_GPT_john_needs_29_planks_for_house_wall_l202_20237


namespace NUMINAMATH_GPT_total_pencil_length_l202_20219

-- Definitions from the conditions
def purple_length : ℕ := 3
def black_length : ℕ := 2
def blue_length : ℕ := 1

-- Proof statement
theorem total_pencil_length :
  purple_length + black_length + blue_length = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_pencil_length_l202_20219


namespace NUMINAMATH_GPT_false_inverse_proposition_l202_20252

theorem false_inverse_proposition (a b : ℝ) : (a^2 = b^2) → (a = b ∨ a = -b) := sorry

end NUMINAMATH_GPT_false_inverse_proposition_l202_20252


namespace NUMINAMATH_GPT_find_m_n_sum_l202_20277

theorem find_m_n_sum (x y m n : ℤ) 
  (h1 : x = 2)
  (h2 : y = 1)
  (h3 : m * x + y = -3)
  (h4 : x - 2 * y = 2 * n) : 
  m + n = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_n_sum_l202_20277


namespace NUMINAMATH_GPT_sum_of_first_15_squares_l202_20263

noncomputable def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_first_15_squares :
  sum_of_squares 15 = 1240 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_squares_l202_20263


namespace NUMINAMATH_GPT_numbers_written_in_red_l202_20261

theorem numbers_written_in_red :
  ∃ (x : ℕ), x > 0 ∧ x <= 101 ∧ 
  ∀ (largest_blue_num : ℕ) (smallest_red_num : ℕ), 
  (largest_blue_num = x) ∧ 
  (smallest_red_num = x + 1) ∧ 
  (smallest_red_num = (101 - x) / 2) → 
  (101 - x = 68) := by
  sorry

end NUMINAMATH_GPT_numbers_written_in_red_l202_20261


namespace NUMINAMATH_GPT_clerical_percentage_l202_20299

theorem clerical_percentage (total_employees clerical_fraction reduce_fraction: ℕ) 
  (h1 : total_employees = 3600) 
  (h2 : clerical_fraction = 1 / 3)
  (h3 : reduce_fraction = 1 / 2) : 
  ( (reduce_fraction * (clerical_fraction * total_employees)) / 
    (total_employees - reduce_fraction * (clerical_fraction * total_employees))) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_clerical_percentage_l202_20299


namespace NUMINAMATH_GPT_solve_for_x_l202_20266

theorem solve_for_x (x : ℝ) : (5 * x + 9 * x = 350 - 10 * (x - 5)) -> x = 50 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l202_20266


namespace NUMINAMATH_GPT_max_a_l202_20253

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a (a m n : ℝ) (h₀ : 1 ≤ m ∧ m ≤ 5)
                      (h₁ : 1 ≤ n ∧ n ≤ 5)
                      (h₂ : n - m ≥ 2)
                      (h_eq : f a m = f a n) :
  a ≤ Real.log 3 / 4 :=
sorry

end NUMINAMATH_GPT_max_a_l202_20253


namespace NUMINAMATH_GPT_shift_graph_to_right_l202_20287

theorem shift_graph_to_right (x : ℝ) : 
  4 * Real.cos (2 * x + π / 4) = 4 * Real.cos (2 * (x - π / 8) + π / 4) :=
by 
  -- sketch of the intended proof without actual steps for clarity
  sorry

end NUMINAMATH_GPT_shift_graph_to_right_l202_20287


namespace NUMINAMATH_GPT_maximum_s_squared_l202_20216

-- Definitions based on our conditions
def semicircle_radius : ℝ := 5
def diameter_length : ℝ := 10

-- Statement of the problem (no proof, statement only)
theorem maximum_s_squared (A B C : ℝ×ℝ) (AC BC : ℝ) (h : AC + BC = s) :
    (A.2 = 0) ∧ (B.2 = 0) ∧ (dist A B = diameter_length) ∧
    (dist C (5,0) = semicircle_radius) ∧ (s = AC + BC) →
    s^2 ≤ 200 :=
sorry

end NUMINAMATH_GPT_maximum_s_squared_l202_20216


namespace NUMINAMATH_GPT_total_canoes_by_end_of_march_l202_20272

theorem total_canoes_by_end_of_march
  (canoes_jan : ℕ := 3)
  (canoes_feb : ℕ := canoes_jan * 2)
  (canoes_mar : ℕ := canoes_feb * 2) :
  canoes_jan + canoes_feb + canoes_mar = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_canoes_by_end_of_march_l202_20272


namespace NUMINAMATH_GPT_decrease_in_average_age_l202_20273

theorem decrease_in_average_age (original_avg_age : ℕ) (new_students_avg_age : ℕ) 
    (original_strength : ℕ) (new_students_strength : ℕ) 
    (h1 : original_avg_age = 40) (h2 : new_students_avg_age = 32) 
    (h3 : original_strength = 8) (h4 : new_students_strength = 8) : 
    (original_avg_age - ((original_strength * original_avg_age + new_students_strength * new_students_avg_age) / (original_strength + new_students_strength))) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_decrease_in_average_age_l202_20273


namespace NUMINAMATH_GPT_total_people_at_evening_l202_20249

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end NUMINAMATH_GPT_total_people_at_evening_l202_20249


namespace NUMINAMATH_GPT_machine_present_value_l202_20225

/-- A machine depreciates at a certain rate annually.
    Given the future value after a certain number of years and the depreciation rate,
    prove the present value of the machine. -/
theorem machine_present_value
  (depreciation_rate : ℝ := 0.25)
  (future_value : ℝ := 54000)
  (years : ℕ := 3)
  (pv : ℝ := 128000) :
  (future_value = pv * (1 - depreciation_rate) ^ years) :=
sorry

end NUMINAMATH_GPT_machine_present_value_l202_20225


namespace NUMINAMATH_GPT_condition_sufficiency_but_not_necessity_l202_20280

variable (p q : Prop)

theorem condition_sufficiency_but_not_necessity:
  (¬ (p ∨ q) → ¬ p) ∧ (¬ p → ¬ (p ∨ q) → False) := 
by
  sorry

end NUMINAMATH_GPT_condition_sufficiency_but_not_necessity_l202_20280


namespace NUMINAMATH_GPT_percentage_of_books_returned_l202_20235

theorem percentage_of_books_returned
  (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) (returned_books_percentage : ℚ) 
  (h1 : initial_books = 75) 
  (h2 : end_books = 68) 
  (h3 : loaned_books = 20)
  (h4 : returned_books_percentage = (end_books - (initial_books - loaned_books)) * 100 / loaned_books):
  returned_books_percentage = 65 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_books_returned_l202_20235


namespace NUMINAMATH_GPT_sequence_formula_l202_20243

theorem sequence_formula (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h : ∀ n : ℕ, S n = 3 * a n + (-1)^n) :
  ∀ n : ℕ, a n = (1/10) * (3/2)^(n-1) - (2/5) * (-1)^n :=
by sorry

end NUMINAMATH_GPT_sequence_formula_l202_20243


namespace NUMINAMATH_GPT_dixie_cup_ounces_l202_20294

def gallons_to_ounces (gallons : ℕ) : ℕ := gallons * 128

def initial_water_gallons (gallons : ℕ) : ℕ := gallons_to_ounces gallons

def total_chairs (rows chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

theorem dixie_cup_ounces (initial_gallons rows chairs_per_row water_left : ℕ) 
  (h1 : initial_gallons = 3) 
  (h2 : rows = 5) 
  (h3 : chairs_per_row = 10) 
  (h4 : water_left = 84) 
  (h5 : 128 = 128) : 
  (initial_water_gallons initial_gallons - water_left) / total_chairs rows chairs_per_row = 6 :=
by 
  sorry

end NUMINAMATH_GPT_dixie_cup_ounces_l202_20294


namespace NUMINAMATH_GPT_find_b_l202_20224

theorem find_b (a b : ℝ) (h1 : 2 * a + b = 6) (h2 : -2 * a + b = 2) : b = 4 :=
sorry

end NUMINAMATH_GPT_find_b_l202_20224


namespace NUMINAMATH_GPT_trigonometric_inequality_l202_20251

noncomputable def a : Real := (1/2) * Real.cos (8 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (8 * Real.pi / 180)
noncomputable def b : Real := (2 * Real.tan (14 * Real.pi / 180)) / (1 - (Real.tan (14 * Real.pi / 180))^2)
noncomputable def c : Real := Real.sqrt ((1 - Real.cos (48 * Real.pi / 180)) / 2)

theorem trigonometric_inequality :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l202_20251


namespace NUMINAMATH_GPT_n_divisible_by_6_l202_20265

open Int -- Open integer namespace for convenience

theorem n_divisible_by_6 (m n : ℤ)
    (h1 : ∃ (a b : ℤ), a + b = -m ∧ a * b = -n)
    (h2 : ∃ (c d : ℤ), c + d = m ∧ c * d = n) :
    6 ∣ n := 
sorry

end NUMINAMATH_GPT_n_divisible_by_6_l202_20265


namespace NUMINAMATH_GPT_inv_three_mod_thirty_seven_l202_20293

theorem inv_three_mod_thirty_seven : (3 * 25) % 37 = 1 :=
by
  -- Explicit mention to skip the proof with sorry
  sorry

end NUMINAMATH_GPT_inv_three_mod_thirty_seven_l202_20293


namespace NUMINAMATH_GPT_lowest_score_of_14_scores_l202_20203

theorem lowest_score_of_14_scores (mean_14 : ℝ) (new_mean_12 : ℝ) (highest_score : ℝ) (lowest_score : ℝ) :
  mean_14 = 85 ∧ new_mean_12 = 88 ∧ highest_score = 105 → lowest_score = 29 :=
by
  sorry

end NUMINAMATH_GPT_lowest_score_of_14_scores_l202_20203


namespace NUMINAMATH_GPT_remainder_when_divided_l202_20226

theorem remainder_when_divided (L S R : ℕ) (h1: L - S = 1365) (h2: S = 270) (h3: L = 6 * S + R) : 
  R = 15 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l202_20226


namespace NUMINAMATH_GPT_no_solution_for_n_eq_neg2_l202_20281

theorem no_solution_for_n_eq_neg2 : ∀ (x y : ℝ), ¬ (2 * x = 1 + -2 * y ∧ -2 * x = 1 + 2 * y) :=
by sorry

end NUMINAMATH_GPT_no_solution_for_n_eq_neg2_l202_20281


namespace NUMINAMATH_GPT_find_m_l202_20201

variable (a : ℝ) (m : ℝ)

theorem find_m (h : a^(m + 1) * a^(2 * m - 1) = a^9) : m = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l202_20201


namespace NUMINAMATH_GPT_shells_total_l202_20244

theorem shells_total (a s v : ℕ) 
  (h1 : s = v + 16) 
  (h2 : v = a - 5) 
  (h3 : a = 20) : 
  s + v + a = 66 := 
by
  sorry

end NUMINAMATH_GPT_shells_total_l202_20244


namespace NUMINAMATH_GPT_arithmetic_sequence_sufficient_not_necessary_l202_20284

variables {a b c d : ℤ}

-- Proving sufficiency: If a, b, c, d form an arithmetic sequence, then a + d = b + c.
def arithmetic_sequence (a b c d : ℤ) : Prop := 
  a + d = 2*b ∧ b + c = 2*a

theorem arithmetic_sequence_sufficient_not_necessary (h : arithmetic_sequence a b c d) : a + d = b + c ∧ ∃ (x y z w : ℤ), x + w = y + z ∧ ¬ arithmetic_sequence x y z w :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_sufficient_not_necessary_l202_20284


namespace NUMINAMATH_GPT_trigonometric_identity_l202_20211

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l202_20211


namespace NUMINAMATH_GPT_dilation_origin_distance_l202_20260

open Real

-- Definition of points and radii
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Given conditions as definitions
def original_circle := Circle.mk (3, 3) 3
def dilated_circle := Circle.mk (8, 10) 5
def dilation_factor := 5 / 3

-- Problem statement to prove
theorem dilation_origin_distance :
  let d₀ := dist (0, 0) (-6, -6)
  let d₁ := dilation_factor * d₀
  d₁ - d₀ = 4 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_dilation_origin_distance_l202_20260


namespace NUMINAMATH_GPT_Q_polynomial_l202_20217

def cos3x_using_cos2x (cos_α : ℝ) := (2 * cos_α^2 - 1) * cos_α - 2 * (1 - cos_α^2) * cos_α

def Q (x : ℝ) := 4 * x^3 - 3 * x

theorem Q_polynomial (α : ℝ) : Q (Real.cos α) = Real.cos (3 * α) := by
  rw [Real.cos_three_mul]
  sorry

end NUMINAMATH_GPT_Q_polynomial_l202_20217


namespace NUMINAMATH_GPT_range_of_x_l202_20274

theorem range_of_x (x : ℝ) :
  (∀ y : ℝ, 0 < y → y^2 + (2*x - 5)*y - x^2 * (Real.log x - Real.log y) ≤ 0) ↔ x = 5 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l202_20274


namespace NUMINAMATH_GPT_total_expenditure_of_Louis_l202_20276

def fabric_cost (yards price_per_yard : ℕ) : ℕ :=
  yards * price_per_yard

def thread_cost (spools price_per_spool : ℕ) : ℕ :=
  spools * price_per_spool

def total_cost (yards price_per_yard pattern_cost spools price_per_spool : ℕ) : ℕ :=
  fabric_cost yards price_per_yard + pattern_cost + thread_cost spools price_per_spool

theorem total_expenditure_of_Louis :
  total_cost 5 24 15 2 3 = 141 :=
by
  sorry

end NUMINAMATH_GPT_total_expenditure_of_Louis_l202_20276


namespace NUMINAMATH_GPT_distance_from_Q_to_BC_l202_20247

-- Definitions for the problem
structure Square :=
(A B C D : ℝ × ℝ)
(side_length : ℝ)

def P : (ℝ × ℝ) := (3, 6)
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 6)^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25
def side_BC (x y : ℝ) : Prop := x = 6

-- Lean proof statement
theorem distance_from_Q_to_BC (Q : ℝ × ℝ) (hQ1 : circle1 Q.1 Q.2) (hQ2 : circle2 Q.1 Q.2) :
  Exists (fun d : ℝ => Q.1 = 6 ∧ Q.2 = d) := sorry

end NUMINAMATH_GPT_distance_from_Q_to_BC_l202_20247


namespace NUMINAMATH_GPT_real_solution_unique_l202_20262

variable (x : ℝ)

theorem real_solution_unique :
  (x ≠ 2 ∧ (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = 3) ↔ x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_real_solution_unique_l202_20262


namespace NUMINAMATH_GPT_students_more_than_pets_l202_20257

theorem students_more_than_pets
    (num_classrooms : ℕ)
    (students_per_classroom : ℕ)
    (rabbits_per_classroom : ℕ)
    (hamsters_per_classroom : ℕ)
    (total_students : ℕ)
    (total_pets : ℕ)
    (difference : ℕ)
    (classrooms_eq : num_classrooms = 5)
    (students_eq : students_per_classroom = 20)
    (rabbits_eq : rabbits_per_classroom = 2)
    (hamsters_eq : hamsters_per_classroom = 1)
    (total_students_eq : total_students = num_classrooms * students_per_classroom)
    (total_pets_eq : total_pets = num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom)
    (difference_eq : difference = total_students - total_pets) :
  difference = 85 := by
  sorry

end NUMINAMATH_GPT_students_more_than_pets_l202_20257


namespace NUMINAMATH_GPT_triangle_sides_angles_l202_20278

open Real

variables {a b c : ℝ} {α β γ : ℝ}

theorem triangle_sides_angles
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (angles_sum : α + β + γ = π)
  (condition : 3 * α + 2 * β = π) :
  a^2 + b * c - c^2 = 0 :=
sorry

end NUMINAMATH_GPT_triangle_sides_angles_l202_20278


namespace NUMINAMATH_GPT_seq_le_n_squared_l202_20230

theorem seq_le_n_squared (a : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (h_property : ∀ t, ∃ i j, t = a i ∨ t = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_seq_le_n_squared_l202_20230


namespace NUMINAMATH_GPT_remainder_of_b2_minus_3a_div_6_l202_20268

theorem remainder_of_b2_minus_3a_div_6 (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 5) : 
  (b^2 - 3 * a) % 6 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_of_b2_minus_3a_div_6_l202_20268


namespace NUMINAMATH_GPT_polar_equation_C1_intersection_C2_C1_distance_l202_20214

noncomputable def parametric_to_cartesian (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + 2 * Real.cos α ∧ y = 4 + 2 * Real.sin α

noncomputable def cartesian_to_polar (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 4

noncomputable def polar_equation_of_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 16 = 0

noncomputable def C2_line_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem polar_equation_C1 (α : ℝ) (ρ θ : ℝ) :
  parametric_to_cartesian α →
  cartesian_to_polar (2 + 2 * Real.cos α) (4 + 2 * Real.sin α) →
  polar_equation_of_C1 ρ θ :=
by
  sorry

theorem intersection_C2_C1_distance (ρ θ : ℝ) (t1 t2 : ℝ) :
  C2_line_polar θ →
  polar_equation_of_C1 ρ θ →
  (t1 + t2 = 6 * Real.sqrt 2) ∧ (t1 * t2 = 16) →
  |t1 - t2| = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_polar_equation_C1_intersection_C2_C1_distance_l202_20214


namespace NUMINAMATH_GPT_average_children_in_families_with_children_l202_20222

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_average_children_in_families_with_children_l202_20222


namespace NUMINAMATH_GPT_fabric_per_pair_of_pants_l202_20269

theorem fabric_per_pair_of_pants 
  (jenson_shirts_per_day : ℕ)
  (kingsley_pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric_needed : ℕ)
  (days : ℕ)
  (fabric_per_pant : ℕ) :
  jenson_shirts_per_day = 3 →
  kingsley_pants_per_day = 5 →
  fabric_per_shirt = 2 →
  total_fabric_needed = 93 →
  days = 3 →
  fabric_per_pant = 5 :=
by sorry

end NUMINAMATH_GPT_fabric_per_pair_of_pants_l202_20269


namespace NUMINAMATH_GPT_science_votes_percentage_l202_20285

theorem science_votes_percentage 
  (math_votes : ℕ) (english_votes : ℕ) (science_votes : ℕ) (history_votes : ℕ) (art_votes : ℕ) 
  (total_votes : ℕ := math_votes + english_votes + science_votes + history_votes + art_votes) 
  (percentage : ℕ := ((science_votes * 100) / total_votes)) :
  math_votes = 80 →
  english_votes = 70 →
  science_votes = 90 →
  history_votes = 60 →
  art_votes = 50 →
  percentage = 26 :=
by
  intros
  sorry

end NUMINAMATH_GPT_science_votes_percentage_l202_20285


namespace NUMINAMATH_GPT_inequality_proof_problem_l202_20215

theorem inequality_proof_problem (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) ≤ 1 / a) :=
sorry

end NUMINAMATH_GPT_inequality_proof_problem_l202_20215


namespace NUMINAMATH_GPT_find_number_l202_20238

theorem find_number (N M : ℕ) 
  (h1 : N + M = 3333) (h2 : N - M = 693) :
  N = 2013 :=
sorry

end NUMINAMATH_GPT_find_number_l202_20238


namespace NUMINAMATH_GPT_birds_in_trees_l202_20245

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end NUMINAMATH_GPT_birds_in_trees_l202_20245


namespace NUMINAMATH_GPT_collinear_vectors_parallel_right_angle_triangle_abc_l202_20236

def vec_ab (k : ℝ) : ℝ × ℝ := (2 - k, -1)
def vec_ac (k : ℝ) : ℝ × ℝ := (1, k)

-- Prove that if vectors AB and AC are collinear, then k = 1 ± √2
theorem collinear_vectors_parallel (k : ℝ) :
  (2 - k) * k - 1 = 0 ↔ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2 :=
by
  sorry

def vec_bc (k : ℝ) : ℝ × ℝ := (k - 1, k + 1)

-- Prove that if triangle ABC is right-angled, then k = 1 or k = -1 ± √2
theorem right_angle_triangle_abc (k : ℝ) :
  ( (2 - k) * 1 + (-1) * k = 0 ∨ (k - 1) * 1 + (k + 1) * k = 0 ) ↔ 
  k = 1 ∨ k = -1 + Real.sqrt 2 ∨ k = -1 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_collinear_vectors_parallel_right_angle_triangle_abc_l202_20236


namespace NUMINAMATH_GPT_length_AC_l202_20200
open Real

-- Define the conditions and required proof
theorem length_AC (AB DC AD : ℝ) (h1 : AB = 17) (h2 : DC = 25) (h3 : AD = 8) : 
  abs (sqrt ((AD + DC - AD)^2 + (DC - sqrt (AB^2 - AD^2))^2) - 33.6) < 0.1 := 
  by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_GPT_length_AC_l202_20200


namespace NUMINAMATH_GPT_intervals_of_monotonicity_range_of_a_for_zeros_l202_20297

open Real

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * log x

theorem intervals_of_monotonicity (a : ℝ) (ha : a ≠ 0) :
  (0 < a → ∀ x, (0 < x ∧ x < a → f x a < f (x + 1) a)
            ∧ (a < x ∧ x < 2 * a → f x a > f (x + 1) a)
            ∧ (2 * a < x → f x a < f (x + 1) a))
  ∧ (a < 0 → ∀ x, (0 < x → f x a < f (x + 1) a)) :=
sorry

theorem range_of_a_for_zeros (a x : ℝ) (ha : 0 < a) 
  (h1 : f a a > 0) (h2 : f (2 * a) a < 0) :
  e ^ (5 / 4) < a ∧ a < e ^ 2 / 2 :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_range_of_a_for_zeros_l202_20297


namespace NUMINAMATH_GPT_find_M_value_l202_20227

-- Statements of the problem conditions and the proof goal
theorem find_M_value (a b c M : ℤ) (h1 : a + b + c = 75) (h2 : a + 4 = M) (h3 : b - 5 = M) (h4 : 3 * c = M) : M = 31 := 
by
  sorry

end NUMINAMATH_GPT_find_M_value_l202_20227


namespace NUMINAMATH_GPT_girls_in_school_l202_20282

theorem girls_in_school (boys girls : ℕ) (ratio : ℕ → ℕ → Prop) (h1 : ratio 5 4) (h2 : boys = 1500) :
    girls = 1200 :=
by
  sorry

end NUMINAMATH_GPT_girls_in_school_l202_20282


namespace NUMINAMATH_GPT_mean_of_six_numbers_l202_20255

theorem mean_of_six_numbers (sum_of_six : ℚ) (h : sum_of_six = 3 / 4) : (sum_of_six / 6) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_six_numbers_l202_20255


namespace NUMINAMATH_GPT_range_of_a_l202_20202

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, ¬ (x^2 + (a-1)*x + 1 ≤ 0)
def proposition_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a-1)^x < (a-1)^y

theorem range_of_a (a : ℝ) :
  ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) →
  (-1 < a ∧ a ≤ 2) ∨ (3 ≤ a) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l202_20202


namespace NUMINAMATH_GPT_charles_paints_l202_20295

-- Define the ratio and total work conditions
def ratio_a_to_c (a c : ℕ) := a * 6 = c * 2

def total_work (total : ℕ) := total = 320

-- Define the question, i.e., the amount of work Charles does
theorem charles_paints (a c total : ℕ) (h_ratio : ratio_a_to_c a c) (h_total : total_work total) : 
  (total / (a + c)) * c = 240 :=
by 
  -- We include sorry to indicate the need for proof here
  sorry

end NUMINAMATH_GPT_charles_paints_l202_20295


namespace NUMINAMATH_GPT_number_of_questions_in_test_l202_20240

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_questions_in_test_l202_20240


namespace NUMINAMATH_GPT_solution_l202_20208

noncomputable def problem (x : ℝ) (h : x ≠ 3) : ℝ :=
  (3 * x / (x - 3)) + ((x + 6) / (3 - x))

theorem solution (x : ℝ) (h : x ≠ 3) : problem x h = 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_l202_20208


namespace NUMINAMATH_GPT_find_x_l202_20205

variable {a b x : ℝ}
variable (h₁ : b ≠ 0)
variable (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b)

theorem find_x (h₁ : b ≠ 0) (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a :=
by
  sorry

end NUMINAMATH_GPT_find_x_l202_20205


namespace NUMINAMATH_GPT_quadratic_equality_l202_20259

theorem quadratic_equality (a_2 : ℝ) (a_1 : ℝ) (a_0 : ℝ) (r : ℝ) (s : ℝ) (x : ℝ)
  (h₁ : a_2 ≠ 0)
  (h₂ : a_0 ≠ 0)
  (h₃ : a_2 * r^2 + a_1 * r + a_0 = 0)
  (h₄ : a_2 * s^2 + a_1 * s + a_0 = 0) :
  a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equality_l202_20259


namespace NUMINAMATH_GPT_sand_needed_for_sandbox_l202_20288

def length1 : ℕ := 50
def width1 : ℕ := 30
def length2 : ℕ := 20
def width2 : ℕ := 15
def area_per_bag : ℕ := 80
def weight_per_bag : ℕ := 30

theorem sand_needed_for_sandbox :
  (length1 * width1 + length2 * width2 + area_per_bag - 1) / area_per_bag * weight_per_bag = 690 :=
by sorry

end NUMINAMATH_GPT_sand_needed_for_sandbox_l202_20288


namespace NUMINAMATH_GPT_square_area_eq_36_l202_20292

theorem square_area_eq_36 (A_triangle : ℝ) (P_triangle : ℝ) 
  (h1 : A_triangle = 16 * Real.sqrt 3)
  (h2 : P_triangle = 3 * (Real.sqrt (16 * 4 * Real.sqrt 3)))
  (h3 : ∀ a, 4 * a = P_triangle) : 
  a^2 = 36 :=
by sorry

end NUMINAMATH_GPT_square_area_eq_36_l202_20292


namespace NUMINAMATH_GPT_cube_polygon_area_l202_20207

theorem cube_polygon_area (cube_side : ℝ) 
  (A B C D : ℝ × ℝ × ℝ)
  (P Q R : ℝ × ℝ × ℝ)
  (hP : P = (10, 0, 0))
  (hQ : Q = (30, 0, 20))
  (hR : R = (30, 5, 30))
  (hA : A = (0, 0, 0))
  (hB : B = (30, 0, 0))
  (hC : C = (30, 0, 30))
  (hD : D = (30, 30, 30))
  (cube_length : cube_side = 30) :
  ∃ area, area = 450 := 
sorry

end NUMINAMATH_GPT_cube_polygon_area_l202_20207


namespace NUMINAMATH_GPT_negation_of_exists_prop_l202_20264

theorem negation_of_exists_prop (x : ℝ) :
  (¬ ∃ (x : ℝ), (x > 0) ∧ (|x| + x >= 0)) ↔ (∀ (x : ℝ), x > 0 → |x| + x < 0) := 
sorry

end NUMINAMATH_GPT_negation_of_exists_prop_l202_20264


namespace NUMINAMATH_GPT_calculate_molar_mass_l202_20206

-- Definitions from the conditions
def number_of_moles : ℝ := 8
def weight_in_grams : ℝ := 1600

-- Goal: Prove that the molar mass is 200 grams/mole
theorem calculate_molar_mass : (weight_in_grams / number_of_moles) = 200 :=
by
  sorry

end NUMINAMATH_GPT_calculate_molar_mass_l202_20206


namespace NUMINAMATH_GPT_michael_wants_to_buy_more_packs_l202_20242

theorem michael_wants_to_buy_more_packs
  (initial_packs : ℕ)
  (cost_per_pack : ℝ)
  (total_value_after_purchase : ℝ)
  (value_of_current_packs : ℝ := initial_packs * cost_per_pack)
  (additional_value_needed : ℝ := total_value_after_purchase - value_of_current_packs)
  (packs_to_buy : ℝ := additional_value_needed / cost_per_pack)
  (answer : ℕ := 2) :
  initial_packs = 4 → cost_per_pack = 2.5 → total_value_after_purchase = 15 → packs_to_buy = answer :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end NUMINAMATH_GPT_michael_wants_to_buy_more_packs_l202_20242


namespace NUMINAMATH_GPT_book_arrangements_l202_20290

theorem book_arrangements :
  let math_books := 4
  let english_books := 4
  let groups := 2
  (groups.factorial) * (math_books.factorial) * (english_books.factorial) = 1152 :=
by
  sorry

end NUMINAMATH_GPT_book_arrangements_l202_20290


namespace NUMINAMATH_GPT_ratio_of_radii_of_circles_l202_20204

theorem ratio_of_radii_of_circles 
  (a b : ℝ) 
  (h1 : a = 6) 
  (h2 : b = 8) 
  (h3 : ∃ (c : ℝ), c = Real.sqrt (a^2 + b^2)) 
  (h4 : ∃ (r R : ℝ), R = c / 2 ∧ r = 24 / (a + b + c)) : R / r = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_of_circles_l202_20204


namespace NUMINAMATH_GPT_coefficient_x3_in_product_l202_20246

-- Definitions for the polynomials
def P(x : ℕ → ℕ) : ℕ → ℤ
| 4 => 3
| 3 => 4
| 2 => -2
| 1 => 8
| 0 => -5
| _ => 0

def Q(x : ℕ → ℕ) : ℕ → ℤ
| 3 => 2
| 2 => -7
| 1 => 5
| 0 => -3
| _ => 0

-- Statement of the problem
theorem coefficient_x3_in_product :
  (P 3 * Q 0 + P 2 * Q 1 + P 1 * Q 2) = -78 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x3_in_product_l202_20246


namespace NUMINAMATH_GPT_min_value_expr_l202_20218

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_expr_l202_20218


namespace NUMINAMATH_GPT_solve_equation_l202_20270

theorem solve_equation (x : ℂ) (h : (x^2 + 3*x + 4) / (x + 3) = x + 6) : x = -7 / 3 := sorry

end NUMINAMATH_GPT_solve_equation_l202_20270


namespace NUMINAMATH_GPT_jazmin_dolls_correct_l202_20275

-- Define the number of dolls Geraldine has.
def geraldine_dolls : ℕ := 2186

-- Define the number of extra dolls Geraldine has compared to Jazmin.
def extra_dolls : ℕ := 977

-- Define the calculation of the number of dolls Jazmin has.
def jazmin_dolls : ℕ := geraldine_dolls - extra_dolls

-- Prove that the number of dolls Jazmin has is 1209.
theorem jazmin_dolls_correct : jazmin_dolls = 1209 := by
  -- Include the required steps in the future proof here.
  sorry

end NUMINAMATH_GPT_jazmin_dolls_correct_l202_20275


namespace NUMINAMATH_GPT_televisions_bought_l202_20296

theorem televisions_bought (T : ℕ)
  (television_cost : ℕ := 50)
  (figurine_cost : ℕ := 1)
  (num_figurines : ℕ := 10)
  (total_spent : ℕ := 260) :
  television_cost * T + figurine_cost * num_figurines = total_spent → T = 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_televisions_bought_l202_20296
