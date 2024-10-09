import Mathlib

namespace sum_of_roots_eq_five_thirds_l1242_124220

-- Define the quadratic equation
def quadratic_eq (n : ℝ) : Prop := 3 * n^2 - 5 * n - 4 = 0

-- Prove that the sum of the solutions to the quadratic equation is 5/3
theorem sum_of_roots_eq_five_thirds :
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 5 / 3) :=
sorry

end sum_of_roots_eq_five_thirds_l1242_124220


namespace problem1_correct_problem2_correct_l1242_124266

noncomputable def problem1 : ℚ :=
  (1/2 - 5/9 + 7/12) * (-36)

theorem problem1_correct : problem1 = -19 := 
by 
  sorry

noncomputable def mixed_number (a : ℤ) (b : ℚ) : ℚ := a + b

noncomputable def problem2 : ℚ :=
  (mixed_number (-199) (24/25)) * 5

theorem problem2_correct : problem2 = -999 - 4/5 :=
by
  sorry

end problem1_correct_problem2_correct_l1242_124266


namespace polygon_largest_area_l1242_124273

-- Definition for the area calculation of each polygon based on given conditions
def area_A : ℝ := 3 * 1 + 2 * 0.5
def area_B : ℝ := 6 * 1
def area_C : ℝ := 4 * 1 + 3 * 0.5
def area_D : ℝ := 5 * 1 + 1 * 0.5
def area_E : ℝ := 7 * 1

-- Theorem stating the problem
theorem polygon_largest_area :
  area_E = max (max (max (max area_A area_B) area_C) area_D) area_E :=
by
  -- The proof steps would go here.
  sorry

end polygon_largest_area_l1242_124273


namespace domain_log_base_2_l1242_124278

theorem domain_log_base_2 (x : ℝ) : (1 - x > 0) ↔ (x < 1) := by
  sorry

end domain_log_base_2_l1242_124278


namespace yards_green_correct_l1242_124299

-- Define the conditions
def total_yards_silk := 111421
def yards_pink := 49500

-- Define the question as a theorem statement
theorem yards_green_correct :
  (total_yards_silk - yards_pink = 61921) :=
by
  sorry

end yards_green_correct_l1242_124299


namespace john_replace_bedroom_doors_l1242_124209

variable (B O : ℕ)
variable (cost_outside cost_bedroom total_cost : ℕ)

def john_has_to_replace_bedroom_doors : Prop :=
  let outside_doors_replaced := 2
  let cost_of_outside_door := 20
  let cost_of_bedroom_door := 10
  let total_replacement_cost := 70
  O = outside_doors_replaced ∧
  cost_outside = cost_of_outside_door ∧
  cost_bedroom = cost_of_bedroom_door ∧
  total_cost = total_replacement_cost ∧
  20 * O + 10 * B = total_cost →
  B = 3

theorem john_replace_bedroom_doors : john_has_to_replace_bedroom_doors B O cost_outside cost_bedroom total_cost :=
sorry

end john_replace_bedroom_doors_l1242_124209


namespace find_x_squared_perfect_square_l1242_124258

theorem find_x_squared_perfect_square (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n ≠ m)
  (h4 : n > m) (h5 : n % 2 ≠ m % 2) : 
  ∃ x : ℤ, x = 0 ∧ ∀ x, (x = 0) → ∃ k : ℕ, (x ^ (2 ^ n) - 1) / (x ^ (2 ^ m) - 1) = k^2 :=
sorry

end find_x_squared_perfect_square_l1242_124258


namespace find_x_l1242_124215

theorem find_x (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (3, x)) (h : (a.fst * b.fst + a.snd * b.snd) = 3) : x = 3 :=
by
  sorry

end find_x_l1242_124215


namespace slope_l1_parallel_lines_math_proof_problem_l1242_124207

-- Define the two lines
def l1 := ∀ x y : ℝ, x + 2 * y + 2 = 0
def l2 (a : ℝ) := ∀ x y : ℝ, a * x + y - 4 = 0

-- Define the assertions
theorem slope_l1 : ∀ x y : ℝ, x + 2 * y + 2 = 0 ↔ y = -1 / 2 * x - 1 := sorry

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) ↔ a = 1 / 2 := sorry

-- Using the assertions to summarize what we need to prove
theorem math_proof_problem (a : ℝ) :
  ((∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) → a = 1 / 2) ∧
  (∀ x y : ℝ, x + 2 * y + 2 = 0 → y = -1 / 2 * x - 1) := sorry

end slope_l1_parallel_lines_math_proof_problem_l1242_124207


namespace smallest_integer_solution_system_of_inequalities_solution_l1242_124233

-- Define the conditions and problem
variable (x : ℝ)

-- Part 1: Prove smallest integer solution for 5x + 15 > x - 1
theorem smallest_integer_solution :
  5 * x + 15 > x - 1 → x = -3 := sorry

-- Part 2: Prove solution set for system of inequalities
theorem system_of_inequalities_solution :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) → (-4 < x ∧ x ≤ 1) := sorry

end smallest_integer_solution_system_of_inequalities_solution_l1242_124233


namespace functional_equation_zero_l1242_124240

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_equation_zero_l1242_124240


namespace consecutive_squares_not_arithmetic_sequence_l1242_124230

theorem consecutive_squares_not_arithmetic_sequence (x y z w : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h_order: x < y ∧ y < z ∧ z < w) :
  ¬ (∃ d : ℕ, y^2 = x^2 + d ∧ z^2 = y^2 + d ∧ w^2 = z^2 + d) :=
sorry

end consecutive_squares_not_arithmetic_sequence_l1242_124230


namespace stewart_farm_sheep_count_l1242_124252

theorem stewart_farm_sheep_count
  (ratio : ℕ → ℕ → Prop)
  (S H : ℕ)
  (ratio_S_H : ratio S H)
  (one_sheep_seven_horses : ratio 1 7)
  (food_per_horse : ℕ)
  (total_food : ℕ)
  (food_per_horse_val : food_per_horse = 230)
  (total_food_val : total_food = 12880)
  (calc_horses : H = total_food / food_per_horse)
  (calc_sheep : S = H / 7) :
  S = 8 :=
by {
  /- Given the conditions, we need to show that S = 8 -/
  sorry
}

end stewart_farm_sheep_count_l1242_124252


namespace beads_pulled_out_l1242_124204

theorem beads_pulled_out (white_beads black_beads : ℕ) (frac_black frac_white : ℚ) (h_black : black_beads = 90) (h_white : white_beads = 51) (h_frac_black : frac_black = (1/6)) (h_frac_white : frac_white = (1/3)) : 
  white_beads * frac_white + black_beads * frac_black = 32 := 
by
  sorry

end beads_pulled_out_l1242_124204


namespace average_visitors_other_days_l1242_124216

theorem average_visitors_other_days 
  (avg_sunday : ℕ) (avg_day : ℕ)
  (num_days : ℕ) (sunday_offset : ℕ)
  (other_days_count : ℕ) (total_days : ℕ) 
  (total_avg_visitors : ℕ)
  (sunday_avg_visitors : ℕ) :
  avg_sunday = 150 →
  avg_day = 125 →
  num_days = 30 →
  sunday_offset = 5 →
  total_days = 30 →
  total_avg_visitors * total_days =
    (sunday_offset * sunday_avg_visitors) + (other_days_count * avg_sunday) →
  125 = total_avg_visitors →
  150 = sunday_avg_visitors →
  other_days_count = num_days - sunday_offset →
  (125 * 30 = (5 * 150) + (other_days_count * avg_sunday)) →
  avg_sunday = 120 :=
by
  sorry

end average_visitors_other_days_l1242_124216


namespace additional_land_cost_l1242_124249

noncomputable def initial_land := 300
noncomputable def final_land := 900
noncomputable def cost_per_square_meter := 20

theorem additional_land_cost : (final_land - initial_land) * cost_per_square_meter = 12000 :=
by
  -- Define the amount of additional land purchased
  let additional_land := final_land - initial_land
  -- Calculate the cost of the additional land            
  show additional_land * cost_per_square_meter = 12000
  sorry

end additional_land_cost_l1242_124249


namespace sum_of_squares_positive_l1242_124242

theorem sum_of_squares_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2 > 0) ∧ (b^2 + c^2 > 0) ∧ (c^2 + a^2 > 0) :=
by
  sorry

end sum_of_squares_positive_l1242_124242


namespace company_KW_price_l1242_124208

theorem company_KW_price (A B : ℝ) (x : ℝ) (h1 : P = x * A) (h2 : P = 2 * B) (h3 : P = (6 / 7) * (A + B)) : x = 1.666666666666667 := 
sorry

end company_KW_price_l1242_124208


namespace problem_solution_l1242_124205

theorem problem_solution :
  (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := 
  by 
  sorry

end problem_solution_l1242_124205


namespace smallest_lcm_for_80k_quadruples_l1242_124277

-- Declare the gcd and lcm functions for quadruples
def gcd_quad (a b c d : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)
def lcm_quad (a b c d : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

-- Main statement we need to prove
theorem smallest_lcm_for_80k_quadruples :
  ∃ m : ℕ, (∃ (a b c d : ℕ), gcd_quad a b c d = 100 ∧ lcm_quad a b c d = m) ∧
    (∀ m', m' < m → ¬ (∃ (a' b' c' d' : ℕ), gcd_quad a' b' c' d' = 100 ∧ lcm_quad a' b' c' d' = m')) ∧
    m = 2250000 :=
sorry

end smallest_lcm_for_80k_quadruples_l1242_124277


namespace man_overtime_hours_correctness_l1242_124228

def man_worked_overtime_hours (r h_r t : ℕ): ℕ :=
  let regular_pay := r * h_r
  let overtime_pay := t - regular_pay
  let overtime_rate := 2 * r
  overtime_pay / overtime_rate

theorem man_overtime_hours_correctness : man_worked_overtime_hours 3 40 186 = 11 := by
  sorry

end man_overtime_hours_correctness_l1242_124228


namespace min_value_f_a_neg3_max_value_g_ge_7_l1242_124287

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) * (x^2 + a * x + 1)

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^3 + 3 * (b + 1) * x^2 + 6 * b * x + 6

theorem min_value_f_a_neg3 (h : -3 ≤ -1) : 
  (∀ x : ℝ, f x (-3) ≥ -Real.exp 2) := 
sorry

theorem max_value_g_ge_7 (a : ℝ) (h : a ≤ -1) (b : ℝ) (h_b : b = a + 1) :
  ∃ m : ℝ, (∀ x : ℝ, g x b ≤ m) ∧ (m ≥ 7) := 
sorry

end min_value_f_a_neg3_max_value_g_ge_7_l1242_124287


namespace range_of_a_l1242_124227

theorem range_of_a (a : ℝ) : 
  ((-1 + a) ^ 2 + (-1 - a) ^ 2 < 4) ↔ (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l1242_124227


namespace cryptarithm_solutions_unique_l1242_124261

/- Definitions corresponding to the conditions -/
def is_valid_digit (d : Nat) : Prop := d < 10

def is_six_digit_number (n : Nat) : Prop := n >= 100000 ∧ n < 1000000

def matches_cryptarithm (abcdef bcdefa : Nat) : Prop := abcdef * 3 = bcdefa

/- Prove that the two identified solutions are valid and no other solutions exist -/
theorem cryptarithm_solutions_unique :
  ∀ (A B C D E F : Nat),
  is_valid_digit A → is_valid_digit B → is_valid_digit C →
  is_valid_digit D → is_valid_digit E → is_valid_digit F →
  let abcdef := 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F
  let bcdefa := 100000 * B + 10000 * C + 1000 * D + 100 * E + 10 * F + A
  is_six_digit_number abcdef →
  is_six_digit_number bcdefa →
  matches_cryptarithm abcdef bcdefa →
  (abcdef = 142857 ∨ abcdef = 285714) :=
by
  intros A B C D E F A_valid B_valid C_valid D_valid E_valid F_valid abcdef bcdefa abcdef_six_digit bcdefa_six_digit cryptarithm_match
  sorry

end cryptarithm_solutions_unique_l1242_124261


namespace cylinder_base_radii_l1242_124214

theorem cylinder_base_radii {l w : ℝ} (hl : l = 3 * Real.pi) (hw : w = Real.pi) :
  (∃ r : ℝ, l = 2 * Real.pi * r ∧ r = 3 / 2) ∨ (∃ r : ℝ, w = 2 * Real.pi * r ∧ r = 1 / 2) :=
sorry

end cylinder_base_radii_l1242_124214


namespace sequence_terms_distinct_l1242_124239

theorem sequence_terms_distinct (n m : ℕ) (hnm : n ≠ m) : 
  (n / (n + 1) : ℚ) ≠ (m / (m + 1) : ℚ) :=
sorry

end sequence_terms_distinct_l1242_124239


namespace find_percentage_l1242_124255

/-- 
Given some percentage P of 6,000, when subtracted from 1/10th of 6,000 (which is 600), 
the difference is 693. Prove that P equals 1.55.
-/
theorem find_percentage (P : ℝ) (h₁ : 6000 / 10 = 600) (h₂ : 600 - (P / 100) * 6000 = 693) : 
  P = 1.55 :=
  sorry

end find_percentage_l1242_124255


namespace arithmetic_progression_x_value_l1242_124298

theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), (3 * x + 2) - (2 * x - 4) = (5 * x - 1) - (3 * x + 2) → x = 9 :=
by
  intros x h
  sorry

end arithmetic_progression_x_value_l1242_124298


namespace quadratic_axis_of_symmetry_l1242_124210

theorem quadratic_axis_of_symmetry (b c : ℝ) (h : -b / 2 = 3) : b = 6 :=
by
  sorry

end quadratic_axis_of_symmetry_l1242_124210


namespace units_digit_odd_product_l1242_124295

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end units_digit_odd_product_l1242_124295


namespace veromont_clicked_ads_l1242_124219

def ads_on_first_page := 12
def ads_on_second_page := 2 * ads_on_first_page
def ads_on_third_page := ads_on_second_page + 24
def ads_on_fourth_page := (3 / 4) * ads_on_second_page
def total_ads := ads_on_first_page + ads_on_second_page + ads_on_third_page + ads_on_fourth_page
def ads_clicked := (2 / 3) * total_ads

theorem veromont_clicked_ads : ads_clicked = 68 := 
by
  sorry

end veromont_clicked_ads_l1242_124219


namespace find_v1_l1242_124282

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := y^2 + 4 * y - 5

theorem find_v1 : v 1 = 11.25 := by
  sorry

end find_v1_l1242_124282


namespace line_equation_through_point_slope_l1242_124286

theorem line_equation_through_point_slope :
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ (a * 1 + b * 3 + c = 0) ∧ (y = -4 * x → k = -4 / 9) ∧ (∀ (x y : ℝ), y - 3 = k * (x - 1) → 4 * x + 3 * y - 13 = 0) :=
sorry

end line_equation_through_point_slope_l1242_124286


namespace jackson_chairs_l1242_124236

theorem jackson_chairs (a b c d : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : c = 12) (h4 : d = 6) : a * b + c * d = 96 := 
by sorry

end jackson_chairs_l1242_124236


namespace breadth_of_hall_l1242_124269

/-- Given a hall of length 20 meters and a uniform verandah width of 2.5 meters,
    with a cost of Rs. 700 for flooring the verandah at Rs. 3.50 per square meter,
    prove that the breadth of the hall is 15 meters. -/
theorem breadth_of_hall (h_length : ℝ) (v_width : ℝ) (cost : ℝ) (rate : ℝ) (b : ℝ) :
  h_length = 20 ∧ v_width = 2.5 ∧ cost = 700 ∧ rate = 3.50 →
  25 * (b + 5) - 20 * b = 200 →
  b = 15 :=
by
  intros hc ha
  sorry

end breadth_of_hall_l1242_124269


namespace circle_condition_l1242_124200

noncomputable def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, circle_eq x y m) → m < 1 / 4 :=
by
  sorry

end circle_condition_l1242_124200


namespace decreasing_function_l1242_124291

theorem decreasing_function (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (m + 3) * x1 - 2 > (m + 3) * x2 - 2) ↔ m < -3 :=
by
  sorry

end decreasing_function_l1242_124291


namespace g_at_4_l1242_124251

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

theorem g_at_4 : g 4 = -2 :=
by
  -- Proof would go here
  sorry

end g_at_4_l1242_124251


namespace neg_p_l1242_124267

-- Proposition p : For any x in ℝ, cos x ≤ 1
def p : Prop := ∀ (x : ℝ), Real.cos x ≤ 1

-- Negation of p: There exists an x₀ in ℝ such that cos x₀ > 1
theorem neg_p : ¬p ↔ (∃ (x₀ : ℝ), Real.cos x₀ > 1) := sorry

end neg_p_l1242_124267


namespace matrix_determinant_eq_9_l1242_124285

theorem matrix_determinant_eq_9 (x : ℝ) :
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  (a * d - b * c = 9) → x = -2 :=
by 
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  sorry

end matrix_determinant_eq_9_l1242_124285


namespace log_eighteen_fifteen_l1242_124238

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_eighteen_fifteen (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  log_base 18 15 = (b - a + 1) / (a + 2 * b) :=
by sorry

end log_eighteen_fifteen_l1242_124238


namespace sara_sent_letters_l1242_124254

theorem sara_sent_letters (J : ℕ)
  (h1 : 9 + 3 * J + J = 33) : J = 6 :=
by
  sorry

end sara_sent_letters_l1242_124254


namespace quadratic_distinct_roots_l1242_124296

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end quadratic_distinct_roots_l1242_124296


namespace vector_expression_l1242_124247

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- The target relationship
theorem vector_expression :
  c = (- (3 / 2) • a + (1 / 2) • b) :=
sorry

end vector_expression_l1242_124247


namespace value_of_a_l1242_124237

theorem value_of_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
by
  sorry

end value_of_a_l1242_124237


namespace parallel_vectors_determine_t_l1242_124265

theorem parallel_vectors_determine_t (t : ℝ) (h : (t, -6) = (k * -3, k * 2)) : t = 9 :=
by
  sorry

end parallel_vectors_determine_t_l1242_124265


namespace no_common_points_l1242_124275

theorem no_common_points 
  (x x_o y y_o : ℝ) 
  (h_parabola : y^2 = 4 * x) 
  (h_inside : y_o^2 < 4 * x_o) : 
  ¬ ∃ (x y : ℝ), y * y_o = 2 * (x + x_o) ∧ y^2 = 4 * x :=
by
  sorry

end no_common_points_l1242_124275


namespace quadratic_solutions_l1242_124270

theorem quadratic_solutions (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end quadratic_solutions_l1242_124270


namespace triangle_equilateral_l1242_124246

noncomputable def is_equilateral (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) :
  is_equilateral a b c A B C :=
by
  sorry

end triangle_equilateral_l1242_124246


namespace simplify_expression_l1242_124283

open Real

-- Assuming lg refers to the common logarithm log base 10
noncomputable def problem_expression : ℝ :=
  log 4 + 2 * log 5 + 4^(-1/2:ℝ)

theorem simplify_expression : problem_expression = 5 / 2 :=
by
  -- Placeholder proof, actual steps not required
  sorry

end simplify_expression_l1242_124283


namespace binom_60_3_eq_34220_l1242_124292

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l1242_124292


namespace no_four_consecutive_product_square_l1242_124225

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end no_four_consecutive_product_square_l1242_124225


namespace quadratic_roots_sum_product_l1242_124274

theorem quadratic_roots_sum_product : 
  ∃ x1 x2 : ℝ, (x1^2 - 2*x1 - 4 = 0) ∧ (x2^2 - 2*x2 - 4 = 0) ∧ 
  (x1 ≠ x2) ∧ (x1 + x2 + x1 * x2 = -2) :=
sorry

end quadratic_roots_sum_product_l1242_124274


namespace pyramid_new_volume_l1242_124244

-- Define constants
def V : ℝ := 100
def l : ℝ := 3
def w : ℝ := 2
def h : ℝ := 1.20

-- Define the theorem
theorem pyramid_new_volume : (l * w * h) * V = 720 := by
  sorry -- Proof is skipped

end pyramid_new_volume_l1242_124244


namespace second_number_value_l1242_124218

-- Definition of the problem conditions
variables (x y z : ℝ)
axiom h1 : z = 4.5 * y
axiom h2 : y = 2.5 * x
axiom h3 : (x + y + z) / 3 = 165

-- The goal is to prove y = 82.5 given the conditions h1, h2, and h3
theorem second_number_value : y = 82.5 :=
by
  sorry

end second_number_value_l1242_124218


namespace triangular_prism_skew_pair_count_l1242_124226

-- Definition of a triangular prism with 6 vertices and 15 lines through any two vertices
structure TriangularPrism :=
  (vertices : Fin 6)   -- 6 vertices
  (lines : Fin 15)     -- 15 lines through any two vertices

-- A function to check if two lines are skew lines 
-- (not intersecting and not parallel in three-dimensional space)
def is_skew (line1 line2 : Fin 15) : Prop := sorry

-- Function to count pairs of lines that are skew in a triangular prism
def count_skew_pairs (prism : TriangularPrism) : Nat := sorry

-- Theorem stating the number of skew pairs in a triangular prism is 36
theorem triangular_prism_skew_pair_count (prism : TriangularPrism) :
  count_skew_pairs prism = 36 := 
sorry

end triangular_prism_skew_pair_count_l1242_124226


namespace total_cost_is_26_30_l1242_124213

open Real

-- Define the costs
def cost_snake_toy : ℝ := 11.76
def cost_cage : ℝ := 14.54

-- Define the total cost of purchases
def total_cost : ℝ := cost_snake_toy + cost_cage

-- Prove the total cost equals $26.30
theorem total_cost_is_26_30 : total_cost = 26.30 :=
by
  sorry

end total_cost_is_26_30_l1242_124213


namespace sum_eq_2184_l1242_124221

variable (p q r s : ℝ)

-- Conditions
axiom h1 : r + s = 12 * p
axiom h2 : r * s = 14 * q
axiom h3 : p + q = 12 * r
axiom h4 : p * q = 14 * s
axiom distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

-- Problem: Prove that p + q + r + s = 2184
theorem sum_eq_2184 : p + q + r + s = 2184 := 
by {
  sorry
}

end sum_eq_2184_l1242_124221


namespace train_speed_l1242_124234

-- Definition of the problem
def train_length : ℝ := 350
def time_to_cross_man : ℝ := 4.5
def expected_speed : ℝ := 77.78

-- Theorem statement
theorem train_speed :
  train_length / time_to_cross_man = expected_speed :=
sorry

end train_speed_l1242_124234


namespace simplify_polynomial_l1242_124264

theorem simplify_polynomial (x : ℝ) : 
  (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 = 32 * x ^ 5 := 
by 
  sorry

end simplify_polynomial_l1242_124264


namespace inequality_log_equality_log_l1242_124288

theorem inequality_log (x : ℝ) (hx : x < 0 ∨ x > 0) :
  max 0 (Real.log (|x|)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) := 
sorry

theorem equality_log (x : ℝ) :
  (max 0 (Real.log (|x|)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) := 
sorry

end inequality_log_equality_log_l1242_124288


namespace trader_gain_pens_l1242_124229

theorem trader_gain_pens (C S : ℝ) (h1 : S = 1.25 * C) 
                         (h2 : 80 * S = 100 * C) : S - C = 0.25 * C :=
by
  have h3 : S = 1.25 * C := h1
  have h4 : 80 * S = 100 * C := h2
  sorry

end trader_gain_pens_l1242_124229


namespace vertex_on_x_axis_l1242_124235

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end vertex_on_x_axis_l1242_124235


namespace product_of_numbers_eq_zero_l1242_124259

theorem product_of_numbers_eq_zero (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := 
by
  sorry

end product_of_numbers_eq_zero_l1242_124259


namespace number_of_true_propositions_l1242_124272

def inverse_proposition (x y : ℝ) : Prop :=
  ¬(x + y = 0 → (x ≠ -y))

def contrapositive_proposition (a b : ℝ) : Prop :=
  (a^2 ≤ b^2) → (a ≤ b)

def negation_proposition (x : ℝ) : Prop :=
  (x ≤ -3) → ¬(x^2 + x - 6 > 0)

theorem number_of_true_propositions : 
  (∃ (x y : ℝ), inverse_proposition x y) ∧
  (∃ (a b : ℝ), contrapositive_proposition a b) ∧
  ¬(∃ (x : ℝ), negation_proposition x) → 
  2 = 2 :=
by
  sorry

end number_of_true_propositions_l1242_124272


namespace probability_abs_diff_gt_half_is_7_over_16_l1242_124211

noncomputable def probability_abs_diff_gt_half : ℚ :=
  let p_tail := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping tails
  let p_head := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping heads
  let p_x_tail_y_tail := p_tail * p_tail   -- Both first flips tails
  let p_x1_y_tail := p_head * p_tail / 2     -- x = 1, y flip tails
  let p_x_tail_y0 := p_tail * p_head / 2     -- x flip tails, y = 0
  let p_x1_y0 := p_head * p_head / 4         -- x = 1, y = 0
  -- Individual probabilities for x − y > 1/2
  let p_x_tail_y_tail_diff := (1 : ℚ) / (8 : ℚ) * p_x_tail_y_tail
  let p_x1_y_tail_diff := (1 : ℚ) / (2 : ℚ) * p_x1_y_tail
  let p_x_tail_y0_diff := (1 : ℚ) / (2 : ℚ) * p_x_tail_y0
  let p_x1_y0_diff := (1 : ℚ) * p_x1_y0
  -- Combined probability for x − y > 1/2
  let p_x_y_diff_gt_half := p_x_tail_y_tail_diff +
                            p_x1_y_tail_diff +
                            p_x_tail_y0_diff +
                            p_x1_y0_diff
  -- Final probability for |x − y| > 1/2 is twice of x − y > 1/2
  2 * p_x_y_diff_gt_half

theorem probability_abs_diff_gt_half_is_7_over_16 :
  probability_abs_diff_gt_half = (7 : ℚ) / 16 := 
  sorry

end probability_abs_diff_gt_half_is_7_over_16_l1242_124211


namespace div_sqrt3_mul_inv_sqrt3_eq_one_l1242_124245

theorem div_sqrt3_mul_inv_sqrt3_eq_one :
  (3 / Real.sqrt 3) * (1 / Real.sqrt 3) = 1 :=
by
  sorry

end div_sqrt3_mul_inv_sqrt3_eq_one_l1242_124245


namespace exists_marked_sum_of_three_l1242_124253

theorem exists_marked_sum_of_three (s : Finset ℕ) (h₀ : s.card = 22) (h₁ : ∀ x ∈ s, x ≤ 30) :
  ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, ∃ d ∈ s, a = b + c + d :=
by
  sorry

end exists_marked_sum_of_three_l1242_124253


namespace ones_digit_of_4567_times_3_is_1_l1242_124284

theorem ones_digit_of_4567_times_3_is_1 :
  let n := 4567
  let m := 3
  (n * m) % 10 = 1 :=
by
  let n := 4567
  let m := 3
  have h : (n * m) % 10 = ((4567 * 3) % 10) := by rfl -- simplifying the product
  sorry -- this is where the proof would go, if required

end ones_digit_of_4567_times_3_is_1_l1242_124284


namespace truck_speed_on_dirt_road_l1242_124294

theorem truck_speed_on_dirt_road 
  (total_distance: ℝ) (time_on_dirt: ℝ) (time_on_paved: ℝ) (speed_difference: ℝ)
  (h1: total_distance = 200) (h2: time_on_dirt = 3) (h3: time_on_paved = 2) (h4: speed_difference = 20) : 
  ∃ v: ℝ, (time_on_dirt * v + time_on_paved * (v + speed_difference) = total_distance) ∧ v = 32 := 
sorry

end truck_speed_on_dirt_road_l1242_124294


namespace thief_speed_is_43_75_l1242_124231

-- Given Information
def speed_owner : ℝ := 50
def time_head_start : ℝ := 0.5
def total_time_to_overtake : ℝ := 4

-- Question: What is the speed of the thief's car v?
theorem thief_speed_is_43_75 (v : ℝ) (hv : 4 * v = speed_owner * (total_time_to_overtake - time_head_start)) : v = 43.75 := 
by {
  -- The proof of this theorem is omitted as it is not required.
  sorry
}

end thief_speed_is_43_75_l1242_124231


namespace two_is_four_percent_of_fifty_l1242_124293

theorem two_is_four_percent_of_fifty : (2 / 50) * 100 = 4 := 
by
  sorry

end two_is_four_percent_of_fifty_l1242_124293


namespace radius_for_visibility_l1242_124223

def is_concentric (hex_center : ℝ × ℝ) (circle_center : ℝ × ℝ) : Prop :=
  hex_center = circle_center

def regular_hexagon (side_length : ℝ) : Prop :=
  side_length = 3

theorem radius_for_visibility
  (r : ℝ)
  (hex_center : ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (P_visible: ℝ)
  (prob_Four_sides_visible: ℝ ) :
  is_concentric hex_center circle_center →
  regular_hexagon 3 →
  prob_Four_sides_visible = 1 / 3 →
  P_visible = 4 →
  r = 2.6 :=
by sorry

end radius_for_visibility_l1242_124223


namespace initially_calculated_average_height_l1242_124297

theorem initially_calculated_average_height
  (A : ℝ)
  (h1 : ∀ heights : List ℝ, heights.length = 35 → (heights.sum + (106 - 166) = heights.sum) → (heights.sum / 35) = 180) :
  A = 181.71 :=
sorry

end initially_calculated_average_height_l1242_124297


namespace find_rectangle_area_l1242_124241

noncomputable def rectangle_area (a b : ℕ) : ℕ :=
  a * b

theorem find_rectangle_area (a b : ℕ) :
  (5 : ℚ) / 8 = (a : ℚ) / b ∧ (a + 6) * (b + 6) - a * b = 114 ∧ a + b = 13 →
  rectangle_area a b = 40 :=
by
  sorry

end find_rectangle_area_l1242_124241


namespace max_side_length_is_11_l1242_124224

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l1242_124224


namespace rooster_count_l1242_124280

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end rooster_count_l1242_124280


namespace evaluate_cyclotomic_sum_l1242_124276

theorem evaluate_cyclotomic_sum : 
  (Complex.I ^ 1520 + Complex.I ^ 1521 + Complex.I ^ 1522 + Complex.I ^ 1523 + Complex.I ^ 1524 = 2) :=
by sorry

end evaluate_cyclotomic_sum_l1242_124276


namespace find_C_coordinates_l1242_124281

open Real

noncomputable def coordC (A B : ℝ × ℝ) : ℝ × ℝ :=
  let n := A.1
  let m := B.1
  let coord_n_y : ℝ := n
  let coord_m_y : ℝ := m
  let y_value (x : ℝ) : ℝ := sqrt 3 / x
  (sqrt 3 / 2, 2)

theorem find_C_coordinates :
  ∃ C : ℝ × ℝ, 
  (∃ A B : ℝ × ℝ, 
   A.2 = sqrt 3 / A.1 ∧
   B.2 = sqrt 3 / B.1 + 6 ∧
   A.2 + 6 = B.2 ∧
   B.2 > A.2 ∧ 
   (sqrt 3 / 2, 2) = coordC A B) ∧
   (sqrt 3 / 2, 2) = (C.1, C.2) :=
by
  sorry

end find_C_coordinates_l1242_124281


namespace trajectory_eq_l1242_124268

theorem trajectory_eq (M : Type) [MetricSpace M] : 
  (∀ (r x y : ℝ), (x + 2)^2 + y^2 = (r + 1)^2 ∧ |x - 1| = 1 → y^2 = -8 * x) :=
by sorry

end trajectory_eq_l1242_124268


namespace total_selection_methods_l1242_124243

-- Define the students and days
inductive Student
| S1 | S2 | S3 | S4 | S5

inductive Day
| Wednesday | Thursday | Friday | Saturday | Sunday

-- The condition where S1 cannot be on Saturday and S2 cannot be on Sunday
def valid_arrangement (arrangement : Day → Student) : Prop :=
  arrangement Day.Saturday ≠ Student.S1 ∧
  arrangement Day.Sunday ≠ Student.S2

-- The main statement
theorem total_selection_methods : ∃ (arrangement_count : ℕ), 
  arrangement_count = 78 ∧
  ∀ (arrangement : Day → Student), valid_arrangement arrangement → 
  arrangement_count = 78 :=
sorry

end total_selection_methods_l1242_124243


namespace time_for_q_to_complete_work_alone_l1242_124256

theorem time_for_q_to_complete_work_alone (P Q : ℝ) (h1 : (1 / P) + (1 / Q) = 1 / 40) (h2 : (20 / P) + (12 / Q) = 1) : Q = 64 / 3 :=
by
  sorry

end time_for_q_to_complete_work_alone_l1242_124256


namespace salary_increase_percentage_l1242_124248

theorem salary_increase_percentage (old_salary new_salary : ℕ) (h1 : old_salary = 10000) (h2 : new_salary = 10200) : 
    ((new_salary - old_salary) / old_salary : ℚ) * 100 = 2 := 
by 
  sorry

end salary_increase_percentage_l1242_124248


namespace last_digit_two_power_2015_l1242_124279

/-- The last digit of powers of 2 cycles through 2, 4, 8, 6. Therefore, the last digit of 2^2015 is the same as 2^3, which is 8. -/
theorem last_digit_two_power_2015 : (2^2015) % 10 = 8 :=
by sorry

end last_digit_two_power_2015_l1242_124279


namespace balls_picking_l1242_124262

theorem balls_picking (red_bag blue_bag : ℕ) (h_red : red_bag = 3) (h_blue : blue_bag = 5) : (red_bag * blue_bag = 15) :=
by
  sorry

end balls_picking_l1242_124262


namespace comparison_inequalities_l1242_124222

open Real

theorem comparison_inequalities
  (m : ℝ) (h1 : 3 ^ m = Real.exp 1) 
  (a : ℝ) (h2 : a = cos m) 
  (b : ℝ) (h3 : b = 1 - 1/2 * m^2)
  (c : ℝ) (h4 : c = sin m / m) :
  c > a ∧ a > b := by
  sorry

end comparison_inequalities_l1242_124222


namespace simplify_expression_l1242_124290

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (5 * x) * (x^4) = 27 * x^5 :=
by
  sorry

end simplify_expression_l1242_124290


namespace tank_capacity_l1242_124203

-- Define the conditions given in the problem.
def tank_full_capacity (x : ℝ) : Prop :=
  (0.25 * x = 60) ∧ (0.15 * x = 36)

-- State the theorem that needs to be proved.
theorem tank_capacity : ∃ x : ℝ, tank_full_capacity x ∧ x = 240 := 
by 
  sorry

end tank_capacity_l1242_124203


namespace apples_per_basket_l1242_124202

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) (h : total_apples = 629) (k : num_baskets = 37) :
  total_apples / num_baskets = 17 :=
by
  -- proof omitted
  sorry

end apples_per_basket_l1242_124202


namespace arithmetic_square_root_16_l1242_124232

theorem arithmetic_square_root_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_16_l1242_124232


namespace inscribed_triangle_perimeter_geq_half_l1242_124271

theorem inscribed_triangle_perimeter_geq_half (a : ℝ) (s' : ℝ) (h_a_pos : a > 0) 
  (h_equilateral : ∀ (A B C : Type) (a b c : A), a = b ∧ b = c ∧ c = a) :
  2 * s' >= (3 * a) / 2 :=
by
  sorry

end inscribed_triangle_perimeter_geq_half_l1242_124271


namespace Jenny_ate_65_l1242_124201

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end Jenny_ate_65_l1242_124201


namespace difference_in_spectators_l1242_124263

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end difference_in_spectators_l1242_124263


namespace range_G_l1242_124250

noncomputable def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

theorem range_G : Set.range G = Set.Icc (-8 : ℝ) 8 := sorry

end range_G_l1242_124250


namespace dominoes_per_player_l1242_124260

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end dominoes_per_player_l1242_124260


namespace negation_proposition_l1242_124212

theorem negation_proposition : ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ ∃ x : ℝ, x > 0 ∧ x < 1 := 
by
  sorry

end negation_proposition_l1242_124212


namespace percentage_off_at_sale_l1242_124217

theorem percentage_off_at_sale
  (sale_price original_price : ℝ)
  (h1 : sale_price = 140)
  (h2 : original_price = 350) :
  (original_price - sale_price) / original_price * 100 = 60 :=
by
  sorry

end percentage_off_at_sale_l1242_124217


namespace total_hovering_time_is_24_hours_l1242_124257

-- Define the initial conditions
def mountain_time_day1 : ℕ := 3
def central_time_day1 : ℕ := 4
def eastern_time_day1 : ℕ := 2

-- Define the additional time hovered in each zone on the second day
def additional_time_per_zone_day2 : ℕ := 2

-- Calculate the total time spent on each day
def total_time_day1 : ℕ := mountain_time_day1 + central_time_day1 + eastern_time_day1
def total_additional_time_day2 : ℕ := 3 * additional_time_per_zone_day2 -- there are three zones
def total_time_day2 : ℕ := total_time_day1 + total_additional_time_day2

-- Calculate the total time over the two days
def total_time_two_days : ℕ := total_time_day1 + total_time_day2

-- Prove that the total time over the two days is 24 hours
theorem total_hovering_time_is_24_hours : total_time_two_days = 24 := by
  sorry

end total_hovering_time_is_24_hours_l1242_124257


namespace arithmetic_sum_s6_l1242_124206

theorem arithmetic_sum_s6 (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) 
  (h1 : ∀ n, a (n+1) - a n = d)
  (h2 : a 1 = 2)
  (h3 : S 4 = 20)
  (hS : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) :
  S 6 = 42 :=
by sorry

end arithmetic_sum_s6_l1242_124206


namespace projection_magnitude_of_a_onto_b_equals_neg_three_l1242_124289

variables {a b : ℝ}

def vector_magnitude (v : ℝ) : ℝ := abs v

def dot_product (a b : ℝ) : ℝ := a * b

noncomputable def projection (a b : ℝ) : ℝ := (dot_product a b) / (vector_magnitude b)

theorem projection_magnitude_of_a_onto_b_equals_neg_three
  (ha : vector_magnitude a = 5)
  (hb : vector_magnitude b = 3)
  (hab : dot_product a b = -9) :
  projection a b = -3 :=
by sorry

end projection_magnitude_of_a_onto_b_equals_neg_three_l1242_124289
