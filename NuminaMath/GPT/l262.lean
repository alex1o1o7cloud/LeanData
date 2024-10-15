import Mathlib

namespace NUMINAMATH_GPT_minimum_score_to_win_l262_26248

namespace CompetitionPoints

-- Define points awarded for each position
def points_first : ℕ := 5
def points_second : ℕ := 3
def points_third : ℕ := 1

-- Define the number of competitions
def competitions : ℕ := 3

-- Total points in one competition
def total_points_one_competition : ℕ := points_first + points_second + points_third

-- Total points in all competitions
def total_points_all_competitions : ℕ := total_points_one_competition * competitions

theorem minimum_score_to_win : ∃ m : ℕ, m = 13 ∧ (∀ s : ℕ, s < 13 → ¬ ∃ c1 c2 c3 : ℕ, 
  c1 ≤ competitions ∧ c2 ≤ competitions ∧ c3 ≤ competitions ∧ 
  ((c1 * points_first) + (c2 * points_second) + (c3 * points_third)) = s) :=
by {
  sorry
}

end CompetitionPoints

end NUMINAMATH_GPT_minimum_score_to_win_l262_26248


namespace NUMINAMATH_GPT_cubic_polynomial_unique_l262_26216

-- Define the polynomial q(x)
def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

-- State the conditions
theorem cubic_polynomial_unique :
  q 1 = -8 ∧
  q 2 = -10 ∧
  q 3 = -16 ∧
  q 4 = -32 :=
by
  -- Expand the function definition for the given inputs.
  -- Add these expansions in the proof part.
  sorry

end NUMINAMATH_GPT_cubic_polynomial_unique_l262_26216


namespace NUMINAMATH_GPT_smallest_c_for_3_in_range_l262_26297

theorem smallest_c_for_3_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, (x^2 - 6 * x + c) = 3) ↔ (c ≥ 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_c_for_3_in_range_l262_26297


namespace NUMINAMATH_GPT_solve_quadratic_equation_l262_26236

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 6 * x - 3 = 0 ↔ x = 3 + 2 * Real.sqrt 3 ∨ x = 3 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l262_26236


namespace NUMINAMATH_GPT_parallel_lines_suff_cond_not_necess_l262_26276

theorem parallel_lines_suff_cond_not_necess (a : ℝ) :
  a = -2 → 
  (∀ x y : ℝ, (2 * x + y - 3 = 0) ∧ (2 * x + y + 4 = 0) → 
    (∃ a : ℝ, a = -2 ∨ a = 1)) ∧
    (a = -2 → ∃ a : ℝ, a = -2 ∨ a = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_lines_suff_cond_not_necess_l262_26276


namespace NUMINAMATH_GPT_minimum_value_of_xy_l262_26221

theorem minimum_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy = 64 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_xy_l262_26221


namespace NUMINAMATH_GPT_range_of_a_l262_26209

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < -1 ↔ x ≤ a) ↔ a < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l262_26209


namespace NUMINAMATH_GPT_fraction_to_decimal_l262_26233

theorem fraction_to_decimal (numer: ℚ) (denom: ℕ) (h_denom: denom = 2^5 * 5^1) :
  numer.den = 160 → numer.num = 59 → numer == 0.36875 :=
by
  intros
  sorry  

end NUMINAMATH_GPT_fraction_to_decimal_l262_26233


namespace NUMINAMATH_GPT_tourist_tax_l262_26238

theorem tourist_tax (total_value : ℕ) (non_taxable_amount : ℕ) (tax_rate : ℚ) (tax : ℚ) : 
  total_value = 1720 → 
  non_taxable_amount = 600 → 
  tax_rate = 0.12 → 
  tax = (total_value - non_taxable_amount : ℕ) * tax_rate → 
  tax = 134.40 := 
by 
  intros total_value_eq non_taxable_amount_eq tax_rate_eq tax_eq
  sorry

end NUMINAMATH_GPT_tourist_tax_l262_26238


namespace NUMINAMATH_GPT_intersection_A_B_l262_26288

-- Definition of set A
def A : Set ℝ := { x | x ≤ 3 }

-- Definition of set B
def B : Set ℝ := {2, 3, 4, 5}

-- Proof statement
theorem intersection_A_B :
  A ∩ B = {2, 3} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l262_26288


namespace NUMINAMATH_GPT_percentage_of_divisible_l262_26237

def count_divisible (n m : ℕ) : ℕ :=
(n / m)

def calculate_percentage (part total : ℕ) : ℚ :=
(part * 100 : ℚ) / (total : ℚ)

theorem percentage_of_divisible (n : ℕ) (k : ℕ) (h₁ : n = 150) (h₂ : k = 6) :
  calculate_percentage (count_divisible n k) n = 16.67 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_divisible_l262_26237


namespace NUMINAMATH_GPT_meaningful_fraction_x_range_l262_26280

theorem meaningful_fraction_x_range (x : ℝ) : (x-2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end NUMINAMATH_GPT_meaningful_fraction_x_range_l262_26280


namespace NUMINAMATH_GPT_vivian_mail_june_l262_26289

theorem vivian_mail_june :
  ∀ (m_apr m_may m_jul m_aug : ℕ),
  m_apr = 5 →
  m_may = 10 →
  m_jul = 40 →
  ∃ m_jun : ℕ,
  ∃ pattern : ℕ → ℕ,
  (pattern m_apr = m_may) →
  (pattern m_may = m_jun) →
  (pattern m_jun = m_jul) →
  (pattern m_jul = m_aug) →
  (m_aug = 80) →
  pattern m_may = m_may * 2 →
  pattern m_jun = m_jun * 2 →
  pattern m_jun = 20 :=
by
  sorry

end NUMINAMATH_GPT_vivian_mail_june_l262_26289


namespace NUMINAMATH_GPT_larger_number_l262_26223

/-- The difference of two numbers is 1375 and the larger divided by the smaller gives a quotient of 6 and a remainder of 15. 
Prove that the larger number is 1647. -/
theorem larger_number (L S : ℕ) 
  (h1 : L - S = 1375) 
  (h2 : L = 6 * S + 15) : 
  L = 1647 := 
sorry

end NUMINAMATH_GPT_larger_number_l262_26223


namespace NUMINAMATH_GPT_S_calculation_T_calculation_l262_26253

def S (a b : ℕ) : ℕ := 4 * a + 6 * b
def T (a b : ℕ) : ℕ := 5 * a + 3 * b

theorem S_calculation : S 6 3 = 42 :=
by sorry

theorem T_calculation : T 6 3 = 39 :=
by sorry

end NUMINAMATH_GPT_S_calculation_T_calculation_l262_26253


namespace NUMINAMATH_GPT_sequence_x_values_3001_l262_26240

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end NUMINAMATH_GPT_sequence_x_values_3001_l262_26240


namespace NUMINAMATH_GPT_telephone_charge_l262_26239

theorem telephone_charge (x : ℝ) (h1 : ∀ t : ℝ, t = 18.70 → x + 39 * 0.40 = t) : x = 3.10 :=
by
  sorry

end NUMINAMATH_GPT_telephone_charge_l262_26239


namespace NUMINAMATH_GPT_Tara_loss_point_l262_26263

theorem Tara_loss_point :
  ∀ (clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal) 
  (H1 : initial_savings = 10)
  (H2 : clarinet_cost = 90)
  (H3 : book_price = 5)
  (H4 : total_books_sold = 25)
  (H5 : books_sold_to_goal = (clarinet_cost - initial_savings) / book_price)
  (H6 : additional_books = total_books_sold - books_sold_to_goal),
  additional_books * book_price = 45 :=
by
  intros clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal
  intros H1 H2 H3 H4 H5 H6
  sorry

end NUMINAMATH_GPT_Tara_loss_point_l262_26263


namespace NUMINAMATH_GPT_general_formula_compare_Tn_l262_26269

open scoped BigOperators

-- Define the sequence {a_n} and its sum S_n
noncomputable def aSeq (n : ℕ) : ℕ := n + 1
noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range n, aSeq (k + 1)

-- Given condition
axiom given_condition (n : ℕ) : 2 * S n = (aSeq n - 1) * (aSeq n + 2)

-- Prove the general formula of the sequence
theorem general_formula (n : ℕ) : aSeq n = n + 1 :=
by
  sorry  -- proof

-- Define T_n sequence
noncomputable def T (n : ℕ) : ℕ := ∑ k in Finset.range n, (k - 1) * 2^k / (k * aSeq k)

-- Compare T_n with the given expression
theorem compare_Tn (n : ℕ) : 
  if n < 17 then T n < (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else if n = 17 then T n = (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else T n > (2^(n+1)*(18-n)-2*n-2)/(n+1) :=
by
  sorry  -- proof

end NUMINAMATH_GPT_general_formula_compare_Tn_l262_26269


namespace NUMINAMATH_GPT_melanie_initial_plums_l262_26225

-- define the conditions as constants
def plums_given_to_sam : ℕ := 3
def plums_left_with_melanie : ℕ := 4

-- define the statement to be proven
theorem melanie_initial_plums : (plums_given_to_sam + plums_left_with_melanie = 7) :=
by
  sorry

end NUMINAMATH_GPT_melanie_initial_plums_l262_26225


namespace NUMINAMATH_GPT_modulus_product_l262_26202

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end NUMINAMATH_GPT_modulus_product_l262_26202


namespace NUMINAMATH_GPT_factorize_expression_l262_26254

theorem factorize_expression (x : ℝ) : 2 * x ^ 3 - 4 * x ^ 2 - 6 * x = 2 * x * (x - 3) * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l262_26254


namespace NUMINAMATH_GPT_common_ratio_l262_26231

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n, a (n+1) = r * a n)
variable (h1 : a 5 * a 11 = 3)
variable (h2 : a 3 + a 13 = 4)

theorem common_ratio (h_geom : ∀ n, a (n+1) = r * a n) (h1 : a 5 * a 11 = 3) (h2 : a 3 + a 13 = 4) :
  (r = 3 ∨ r = -3) :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_l262_26231


namespace NUMINAMATH_GPT_regular_polygon_sides_l262_26261

theorem regular_polygon_sides (ratio : ℕ) (interior exterior : ℕ) (sum_angles : ℕ) 
  (h1 : ratio = 5)
  (h2 : interior = 5 * exterior)
  (h3 : interior + exterior = sum_angles)
  (h4 : sum_angles = 180) : 

∃ (n : ℕ), n = 12 := 
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l262_26261


namespace NUMINAMATH_GPT_fraction_of_positive_number_l262_26262

theorem fraction_of_positive_number (x : ℝ) (f : ℝ) (h : x = 0.4166666666666667 ∧ f * x = (25/216) * (1/x)) : f = 2/3 :=
sorry

end NUMINAMATH_GPT_fraction_of_positive_number_l262_26262


namespace NUMINAMATH_GPT_part1_part2_l262_26234

/-- Given a triangle ABC with sides opposite to angles A, B, C being a, b, c respectively,
and a sin A sin B + b cos^2 A = 5/3 a,
prove that (1) b / a = 5/3. -/
theorem part1 (a b : ℝ) (A B : ℝ) (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a) :
  b / a = 5 / 3 :=
sorry

/-- Given the previous result b / a = 5/3 and the condition c^2 = a^2 + 8/5 b^2,
prove that (2) angle C = 2π / 3. -/
theorem part2 (a b c : ℝ) (A B C : ℝ)
  (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a)
  (h₂ : c^2 = a^2 + (8 / 5) * b^2)
  (h₃ : b / a = 5 / 3) :
  C = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l262_26234


namespace NUMINAMATH_GPT_tangent_line_circle_l262_26275

theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*y = 0 → y = a) → (a = 0 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l262_26275


namespace NUMINAMATH_GPT_tan_of_acute_angle_l262_26244

open Real

theorem tan_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 2 * sin (α - 15 * π / 180) - 1 = 0) : tan α = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_acute_angle_l262_26244


namespace NUMINAMATH_GPT_only_positive_odd_integer_dividing_3n_plus_1_l262_26271

theorem only_positive_odd_integer_dividing_3n_plus_1 : 
  ∀ (n : ℕ), (0 < n) → (n % 2 = 1) → (n ∣ (3 ^ n + 1)) → n = 1 := by
  sorry

end NUMINAMATH_GPT_only_positive_odd_integer_dividing_3n_plus_1_l262_26271


namespace NUMINAMATH_GPT_eliminate_y_substitution_l262_26217

theorem eliminate_y_substitution (x y : ℝ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) : 3 * x - x + 5 = 8 := 
by
  sorry

end NUMINAMATH_GPT_eliminate_y_substitution_l262_26217


namespace NUMINAMATH_GPT_maxwell_distance_l262_26200

-- Define the given conditions
def distance_between_homes : ℝ := 65
def maxwell_speed : ℝ := 2
def brad_speed : ℝ := 3

-- The statement we need to prove
theorem maxwell_distance :
  ∃ (x t : ℝ), 
    x = maxwell_speed * t ∧
    distance_between_homes - x = brad_speed * t ∧
    x = 26 := by sorry

end NUMINAMATH_GPT_maxwell_distance_l262_26200


namespace NUMINAMATH_GPT_invest_in_yourself_examples_l262_26212

theorem invest_in_yourself_examples (example1 example2 example3 : String)
  (benefit1 benefit2 benefit3 : String)
  (h1 : example1 = "Investment in Education")
  (h2 : benefit1 = "Spending money on education improves knowledge and skills, leading to better job opportunities and higher salaries. Education appreciates over time, providing financial stability.")
  (h3 : example2 = "Investment in Physical Health")
  (h4 : benefit2 = "Spending on sports activities, fitness programs, or healthcare prevents chronic diseases, saves future medical expenses, and enhances overall well-being.")
  (h5 : example3 = "Time Spent on Reading Books")
  (h6 : benefit3 = "Reading books expands knowledge, improves vocabulary and cognitive abilities, develops critical thinking and analytical skills, and fosters creativity and empathy."):
  "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." = "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." :=
by
  sorry

end NUMINAMATH_GPT_invest_in_yourself_examples_l262_26212


namespace NUMINAMATH_GPT_convert_quadratic_l262_26270

theorem convert_quadratic :
  ∀ x : ℝ, (x^2 + 2*x + 4) = ((x + 1)^2 + 3) :=
by
  sorry

end NUMINAMATH_GPT_convert_quadratic_l262_26270


namespace NUMINAMATH_GPT_simplify_problem_1_simplify_problem_2_l262_26250

-- Problem 1: Statement of Simplification Proof
theorem simplify_problem_1 :
  (- (99 + (71 / 72)) * 36 = - (3599 + 1 / 2)) :=
by sorry

-- Problem 2: Statement of Simplification Proof
theorem simplify_problem_2 :
  (-3 * (1 / 4) - 2.5 * (-2.45) + (7 / 2) * (1 / 4) = 6 + 1 / 4) :=
by sorry

end NUMINAMATH_GPT_simplify_problem_1_simplify_problem_2_l262_26250


namespace NUMINAMATH_GPT_line_equation_through_point_and_area_l262_26204

theorem line_equation_through_point_and_area (b S x y : ℝ) 
  (h1 : ∀ y, (x, y) = (-2*b, 0) → True) 
  (h2 : ∀ p1 p2 p3 : ℝ × ℝ, p1 = (-2*b, 0) → p2 = (0, 0) → 
        ∃ k, p3 = (0, k) ∧ S = 1/2 * (2*b) * k) : 2*S*x - b^2*y + 4*b*S = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_through_point_and_area_l262_26204


namespace NUMINAMATH_GPT_lisa_socks_total_l262_26205

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end NUMINAMATH_GPT_lisa_socks_total_l262_26205


namespace NUMINAMATH_GPT_problem1_problem2_l262_26273

open Real

theorem problem1 : sin (420 * π / 180) * cos (330 * π / 180) + sin (-690 * π / 180) * cos (-660 * π / 180) = 1 := by
  sorry

theorem problem2 (α : ℝ) : 
  (sin (π / 2 + α) * cos (π / 2 - α) / cos (π + α)) + 
  (sin (π - α) * cos (π / 2 + α) / sin (π + α)) = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l262_26273


namespace NUMINAMATH_GPT_value_of_f_csc_squared_l262_26281

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 ∧ x ≠ 1 then 1 / x else 0

lemma csc_sq_identity (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  (f (x / (x - 1)) = 1 / x) := 
  by sorry

theorem value_of_f_csc_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π / 2) :
  f ((1 / (Real.sin t) ^ 2)) = - (Real.cos t) ^ 2 :=
  by sorry

end NUMINAMATH_GPT_value_of_f_csc_squared_l262_26281


namespace NUMINAMATH_GPT_right_triangle_area_l262_26266

theorem right_triangle_area (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : a^2 + b^2 = c^2) :
  (1/2) * (a : ℝ) * b = 30 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l262_26266


namespace NUMINAMATH_GPT_transformed_curve_l262_26229

variables (x y x' y' : ℝ)

def original_curve := (x^2) / 4 - y^2 = 1
def transformation_x := x' = (1/2) * x
def transformation_y := y' = 2 * y

theorem transformed_curve : original_curve x y → transformation_x x x' → transformation_y y y' → x^2 - (y^2) / 4 = 1 := 
sorry

end NUMINAMATH_GPT_transformed_curve_l262_26229


namespace NUMINAMATH_GPT_aleena_vs_bob_distance_l262_26292

theorem aleena_vs_bob_distance :
  let AleenaDistance := 75
  let BobDistance := 60
  AleenaDistance - BobDistance = 15 :=
by
  let AleenaDistance := 75
  let BobDistance := 60
  show AleenaDistance - BobDistance = 15
  sorry

end NUMINAMATH_GPT_aleena_vs_bob_distance_l262_26292


namespace NUMINAMATH_GPT_cherries_cost_l262_26278

def cost_per_kg (total_cost kilograms : ℕ) : ℕ :=
  total_cost / kilograms

theorem cherries_cost 
  (genevieve_amount : ℕ) 
  (short_amount : ℕ)
  (total_kilograms : ℕ) 
  (total_cost : ℕ := genevieve_amount + short_amount) 
  (cost : ℕ := cost_per_kg total_cost total_kilograms) : 
  cost = 8 :=
by
  have h1 : genevieve_amount = 1600 := by sorry
  have h2 : short_amount = 400 := by sorry
  have h3 : total_kilograms = 250 := by sorry
  sorry

end NUMINAMATH_GPT_cherries_cost_l262_26278


namespace NUMINAMATH_GPT_markup_percentage_l262_26213

theorem markup_percentage 
  (CP : ℝ) (x : ℝ) (MP : ℝ) (SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (x / 100) * CP)
  (h3 : SP = MP - (10 / 100) * MP)
  (h4 : SP = CP + (35 / 100) * CP) :
  x = 50 :=
by sorry

end NUMINAMATH_GPT_markup_percentage_l262_26213


namespace NUMINAMATH_GPT_math_problem_l262_26224

theorem math_problem (a b c m n : ℝ)
  (h1 : a = -b)
  (h2 : c = -1)
  (h3 : m * n = 1) : 
  (a + b) / 3 + c^2 - 4 * m * n = -3 := 
by 
  -- Proof steps would be here
  sorry

end NUMINAMATH_GPT_math_problem_l262_26224


namespace NUMINAMATH_GPT_projection_of_difference_eq_l262_26251

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vec_projection (v w : ℝ × ℝ) : ℝ :=
vec_dot (v - w) v / vec_magnitude v

variables (a b : ℝ × ℝ)
  (congruence_cond : vec_magnitude a / vec_magnitude b = Real.cos θ)

theorem projection_of_difference_eq (h : vec_magnitude a / vec_magnitude b = Real.cos θ) :
  vec_projection (a - b) a = (vec_dot a a - vec_dot b b) / vec_magnitude a :=
sorry

end NUMINAMATH_GPT_projection_of_difference_eq_l262_26251


namespace NUMINAMATH_GPT_two_hours_charge_l262_26211

def charge_condition_1 (F A : ℕ) : Prop :=
  F = A + 35

def charge_condition_2 (F A : ℕ) : Prop :=
  F + 4 * A = 350

theorem two_hours_charge (F A : ℕ) (h1 : charge_condition_1 F A) (h2 : charge_condition_2 F A) : 
  F + A = 161 := 
sorry

end NUMINAMATH_GPT_two_hours_charge_l262_26211


namespace NUMINAMATH_GPT_math_problem_l262_26242

theorem math_problem (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 1| = 0) : (a + b) ^ 2023 = -1 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l262_26242


namespace NUMINAMATH_GPT_contrapositive_example_l262_26230

theorem contrapositive_example (x : ℝ) : (x > 2 → x > 0) ↔ (x ≤ 2 → x ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l262_26230


namespace NUMINAMATH_GPT_find_sum_l262_26298

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l262_26298


namespace NUMINAMATH_GPT_area_of_new_triangle_geq_twice_sum_of_areas_l262_26295

noncomputable def area_of_triangle (a b c : ℝ) (alpha : ℝ) : ℝ :=
  0.5 * a * b * (Real.sin alpha)

theorem area_of_new_triangle_geq_twice_sum_of_areas
  (a1 b1 c a2 b2 alpha : ℝ)
  (h1 : a1 <= b1) (h2 : b1 <= c) (h3 : a2 <= b2) (h4 : b2 <= c) :
  let α_1 := Real.arcsin ((a1 + a2) / (2 * c))
  let area1 := area_of_triangle a1 b1 c alpha
  let area2 := area_of_triangle a2 b2 c alpha
  let area_new := area_of_triangle (a1 + a2) (b1 + b2) (2 * c) α_1
  area_new >= 2 * (area1 + area2) :=
sorry

end NUMINAMATH_GPT_area_of_new_triangle_geq_twice_sum_of_areas_l262_26295


namespace NUMINAMATH_GPT_sasha_tree_planting_cost_l262_26286

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end NUMINAMATH_GPT_sasha_tree_planting_cost_l262_26286


namespace NUMINAMATH_GPT_positive_diff_probability_fair_coin_l262_26265

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_positive_diff_probability_fair_coin_l262_26265


namespace NUMINAMATH_GPT_largest_visits_is_four_l262_26287

noncomputable def largest_num_visits (stores people visits : ℕ) (eight_people_two_stores : ℕ) 
  (one_person_min : ℕ) : ℕ := 4 -- This represents the largest number of stores anyone could have visited.

theorem largest_visits_is_four 
  (stores : ℕ) (total_visits : ℕ) (people_shopping : ℕ) 
  (eight_people_two_stores : ℕ) (each_one_store : ℕ) 
  (H1 : stores = 8) 
  (H2 : total_visits = 23) 
  (H3 : people_shopping = 12) 
  (H4 : eight_people_two_stores = 8)
  (H5 : each_one_store = 1) :
  largest_num_visits stores people_shopping total_visits eight_people_two_stores each_one_store = 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_visits_is_four_l262_26287


namespace NUMINAMATH_GPT_pairs_solution_l262_26293

theorem pairs_solution (x y : ℝ) :
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := 
by
  sorry

end NUMINAMATH_GPT_pairs_solution_l262_26293


namespace NUMINAMATH_GPT_carla_sheep_l262_26277

theorem carla_sheep (T : ℝ) (pen_sheep wilderness_sheep : ℝ) 
(h1: 0.90 * T = 81) (h2: pen_sheep = 81) 
(h3: wilderness_sheep = 0.10 * T) : wilderness_sheep = 9 :=
sorry

end NUMINAMATH_GPT_carla_sheep_l262_26277


namespace NUMINAMATH_GPT_age_of_first_person_added_l262_26252

theorem age_of_first_person_added :
  ∀ (T A x : ℕ),
    (T = 7 * A) →
    (T + x = 8 * (A + 2)) →
    (T + 15 = 8 * (A - 1)) →
    x = 39 :=
by
  intros T A x h1 h2 h3
  sorry

end NUMINAMATH_GPT_age_of_first_person_added_l262_26252


namespace NUMINAMATH_GPT_lines_parallel_l262_26255

/--
Given two lines represented by the equations \(2x + my - 2m + 4 = 0\) and \(mx + 2y - m + 2 = 0\), 
prove that the value of \(m\) that makes these two lines parallel is \(m = -2\).
-/
theorem lines_parallel (m : ℝ) : 
    (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0) ∧ (∀ x y : ℝ, m * x + 2 * y - m + 2 = 0) 
    → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_l262_26255


namespace NUMINAMATH_GPT_trees_to_plant_total_l262_26215

def trees_chopped_first_half := 200
def trees_chopped_second_half := 300
def trees_to_plant_per_tree_chopped := 3

theorem trees_to_plant_total : 
  (trees_chopped_first_half + trees_chopped_second_half) * trees_to_plant_per_tree_chopped = 1500 :=
by
  sorry

end NUMINAMATH_GPT_trees_to_plant_total_l262_26215


namespace NUMINAMATH_GPT_polygon_sides_sum_l262_26299

theorem polygon_sides_sum :
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  -- The sides of shapes that are adjacent on one side each (triangle and nonagon)
  let adjacent_triangle_nonagon := triangle_sides + nonagon_sides - 2
  -- The sides of the intermediate shapes that are each adjacent on two sides
  let adjacent_other_shapes := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - 5 * 2
  -- Summing up all the sides exposed to the outside
  adjacent_triangle_nonagon + adjacent_other_shapes = 30 := by
sorry

end NUMINAMATH_GPT_polygon_sides_sum_l262_26299


namespace NUMINAMATH_GPT_jaylen_dog_food_consumption_l262_26283

theorem jaylen_dog_food_consumption :
  ∀ (morning evening daily_consumption total_food : ℕ)
  (days : ℕ),
  (morning = evening) →
  (total_food = 32) →
  (days = 16) →
  (daily_consumption = total_food / days) →
  (morning + evening = daily_consumption) →
  morning = 1 := by
  intros morning evening daily_consumption total_food days h_eq h_total h_days h_daily h_sum
  sorry

end NUMINAMATH_GPT_jaylen_dog_food_consumption_l262_26283


namespace NUMINAMATH_GPT_determinant_scaled_l262_26282

-- Define the initial determinant condition
def init_det (x y z w : ℝ) : Prop :=
  x * w - y * z = -3

-- Define the scaled determinant
def scaled_det (x y z w : ℝ) : ℝ :=
  3 * x * (3 * w) - 3 * y * (3 * z)

-- State the theorem we want to prove
theorem determinant_scaled (x y z w : ℝ) (h : init_det x y z w) :
  scaled_det x y z w = -27 :=
by
  sorry

end NUMINAMATH_GPT_determinant_scaled_l262_26282


namespace NUMINAMATH_GPT_abs_neg_six_l262_26272

theorem abs_neg_six : |(-6)| = 6 := by
  sorry

end NUMINAMATH_GPT_abs_neg_six_l262_26272


namespace NUMINAMATH_GPT_emma_total_investment_l262_26222

theorem emma_total_investment (X : ℝ) (h : 0.09 * 6000 + 0.11 * (X - 6000) = 980) : X = 10000 :=
sorry

end NUMINAMATH_GPT_emma_total_investment_l262_26222


namespace NUMINAMATH_GPT_percentage_of_non_honda_red_cars_l262_26274

/-- 
Total car population in Chennai is 9000.
Honda cars in Chennai is 5000.
Out of every 100 Honda cars, 90 are red.
60% of the total car population is red.
Prove that the percentage of non-Honda cars that are red is 22.5%.
--/
theorem percentage_of_non_honda_red_cars 
  (total_cars : ℕ) (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) (total_red_ratio : ℚ) 
  (h : total_cars = 9000) 
  (h1 : honda_cars = 5000) 
  (h2 : red_honda_ratio = 90 / 100) 
  (h3 : total_red_ratio = 60 / 100) : 
  (900 / (9000 - 5000) * 100 = 22.5) := 
sorry

end NUMINAMATH_GPT_percentage_of_non_honda_red_cars_l262_26274


namespace NUMINAMATH_GPT_abs_eq_zero_iff_l262_26232

theorem abs_eq_zero_iff {a : ℝ} (h : |a + 3| = 0) : a = -3 :=
sorry

end NUMINAMATH_GPT_abs_eq_zero_iff_l262_26232


namespace NUMINAMATH_GPT_op_plus_18_plus_l262_26285

def op_plus (y: ℝ) : ℝ := 9 - y
def plus_op (y: ℝ) : ℝ := y - 9

theorem op_plus_18_plus :
  plus_op (op_plus 18) = -18 := by
  sorry

end NUMINAMATH_GPT_op_plus_18_plus_l262_26285


namespace NUMINAMATH_GPT_irreducible_fraction_denominator_l262_26206

theorem irreducible_fraction_denominator :
  let num := 201920192019
  let denom := 191719171917
  let gcd_num_denom := Int.gcd num denom
  let irreducible_denom := denom / gcd_num_denom
  irreducible_denom = 639 :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fraction_denominator_l262_26206


namespace NUMINAMATH_GPT_value_of_x_plus_2y_l262_26214

theorem value_of_x_plus_2y :
  let x := 3
  let y := 1
  x + 2 * y = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_2y_l262_26214


namespace NUMINAMATH_GPT_problem_statement_l262_26246

theorem problem_statement (x y a : ℝ) (h1 : x + a < y + a) (h2 : a * x > a * y) : x < y ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l262_26246


namespace NUMINAMATH_GPT_unique_solution_of_functional_eqn_l262_26247

theorem unique_solution_of_functional_eqn (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1) → (∀ x : ℝ, f x = x) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_unique_solution_of_functional_eqn_l262_26247


namespace NUMINAMATH_GPT_arithmetic_seq_sum_a3_a15_l262_26210

theorem arithmetic_seq_sum_a3_a15 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_eq : a 1 - a 5 + a 9 - a 13 + a 17 = 117) :
  a 3 + a 15 = 234 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_a3_a15_l262_26210


namespace NUMINAMATH_GPT_find_number_of_small_branches_each_branch_grows_l262_26258

theorem find_number_of_small_branches_each_branch_grows :
  ∃ x : ℕ, 1 + x + x^2 = 43 ∧ x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_of_small_branches_each_branch_grows_l262_26258


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l262_26227

theorem relationship_among_a_b_c (a b c : ℝ) (h₁ : a = 0.09) (h₂ : -2 < b ∧ b < -1) (h₃ : 1 < c ∧ c < 2) : b < a ∧ a < c := 
by 
  -- proof will involve but we only need to state this
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l262_26227


namespace NUMINAMATH_GPT_largest_real_number_condition_l262_26243

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_real_number_condition_l262_26243


namespace NUMINAMATH_GPT_find_t_l262_26245

theorem find_t (t : ℝ) (h : (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 3)) : t = -8 := 
by 
  sorry

end NUMINAMATH_GPT_find_t_l262_26245


namespace NUMINAMATH_GPT_laser_total_distance_l262_26294

noncomputable def laser_path_distance : ℝ :=
  let A := (2, 4)
  let B := (2, -4)
  let C := (-2, -4)
  let D := (8, 4)
  let distance (p q : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B + distance B C + distance C D

theorem laser_total_distance :
  laser_path_distance = 12 + 2 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_GPT_laser_total_distance_l262_26294


namespace NUMINAMATH_GPT_term_in_AP_is_zero_l262_26208

theorem term_in_AP_is_zero (a d : ℤ) 
  (h : (a + 4 * d) + (a + 20 * d) = (a + 7 * d) + (a + 14 * d) + (a + 12 * d)) :
  a + (-9) * d = 0 :=
by
  sorry

end NUMINAMATH_GPT_term_in_AP_is_zero_l262_26208


namespace NUMINAMATH_GPT_roots_condition_implies_m_range_l262_26256

theorem roots_condition_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ (x₁^2 + (m-1)*x₁ + m^2 - 2 = 0) ∧ (x₂^2 + (m-1)*x₂ + m^2 - 2 = 0))
  → -2 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_condition_implies_m_range_l262_26256


namespace NUMINAMATH_GPT_train_crosses_in_26_seconds_l262_26284

def speed_km_per_hr := 72
def length_of_train := 250
def length_of_platform := 270

def total_distance := length_of_train + length_of_platform

noncomputable def speed_m_per_s := (speed_km_per_hr * 1000 / 3600)  -- Convert km/hr to m/s

noncomputable def time_to_cross := total_distance / speed_m_per_s

theorem train_crosses_in_26_seconds :
  time_to_cross = 26 := 
sorry

end NUMINAMATH_GPT_train_crosses_in_26_seconds_l262_26284


namespace NUMINAMATH_GPT_willie_stickers_l262_26267

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 124 → given_stickers = 23 → remaining_stickers = initial_stickers - given_stickers → remaining_stickers = 101 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining.trans rfl

end NUMINAMATH_GPT_willie_stickers_l262_26267


namespace NUMINAMATH_GPT_class_5_matches_l262_26260

theorem class_5_matches (matches_c1 matches_c2 matches_c3 matches_c4 matches_c5 : ℕ)
  (C1 : matches_c1 = 2)
  (C2 : matches_c2 = 4)
  (C3 : matches_c3 = 4)
  (C4 : matches_c4 = 3) :
  matches_c5 = 3 :=
sorry

end NUMINAMATH_GPT_class_5_matches_l262_26260


namespace NUMINAMATH_GPT_tapanga_corey_candies_l262_26219

theorem tapanga_corey_candies (corey_candies : ℕ) (tapanga_candies : ℕ) 
                              (h1 : corey_candies = 29) 
                              (h2 : tapanga_candies = corey_candies + 8) : 
                              corey_candies + tapanga_candies = 66 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_tapanga_corey_candies_l262_26219


namespace NUMINAMATH_GPT_asymptote_hyperbola_condition_l262_26207

theorem asymptote_hyperbola_condition : 
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = 4/3 * x ∨ y = -4/3 * x)) ∧
  ¬(∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x → x^2 / 9 - y^2 / 16 = 1)) :=
by sorry

end NUMINAMATH_GPT_asymptote_hyperbola_condition_l262_26207


namespace NUMINAMATH_GPT_smallest_value_of_x_l262_26268

theorem smallest_value_of_x (x : ℝ) (h : 4 * x^2 - 20 * x + 24 = 0) : x = 2 :=
    sorry

end NUMINAMATH_GPT_smallest_value_of_x_l262_26268


namespace NUMINAMATH_GPT_num_common_points_of_three_lines_l262_26201

def three_planes {P : Type} [AddCommGroup P] (l1 l2 l3 : Set P) : Prop :=
  let p12 := Set.univ \ (l1 ∪ l2)
  let p13 := Set.univ \ (l1 ∪ l3)
  let p23 := Set.univ \ (l2 ∪ l3)
  ∃ (pl12 pl13 pl23 : Set P), 
    p12 = pl12 ∧ p13 = pl13 ∧ p23 = pl23

theorem num_common_points_of_three_lines (l1 l2 l3 : Set ℝ) 
  (h : three_planes l1 l2 l3) : ∃ n : ℕ, n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_GPT_num_common_points_of_three_lines_l262_26201


namespace NUMINAMATH_GPT_sin_double_angle_plus_pi_over_2_l262_26257

theorem sin_double_angle_plus_pi_over_2 (θ : ℝ) (h : Real.cos θ = -1/3) :
  Real.sin (2 * θ + Real.pi / 2) = -7/9 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_plus_pi_over_2_l262_26257


namespace NUMINAMATH_GPT_find_quotient_l262_26203

-- Definitions based on given conditions
def remainder : ℕ := 8
def dividend : ℕ := 997
def divisor : ℕ := 23

-- Hypothesis based on the division formula
def quotient_formula (q : ℕ) : Prop :=
  dividend = (divisor * q) + remainder

-- Statement of the problem
theorem find_quotient (q : ℕ) (h : quotient_formula q) : q = 43 :=
sorry

end NUMINAMATH_GPT_find_quotient_l262_26203


namespace NUMINAMATH_GPT_min_value_of_a1_plus_a7_l262_26226

variable {a : ℕ → ℝ}
variable {a3 a5 : ℝ}

-- Conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) := 
  ∀ n, a n > 0 ∧ (∃ r, ∀ i, a (i + 1) = a i * r)

def condition (a : ℕ → ℝ) (a3 a5 : ℝ) :=
  a 3 = a3 ∧ a 5 = a5 ∧ a3 * a5 = 64

-- Prove that the minimum value of a1 + a7 is 16
theorem min_value_of_a1_plus_a7
  (h1 : is_positive_geometric_sequence a)
  (h2 : condition a a3 a5) :
  ∃ a1 a7, a 1 = a1 ∧ a 7 = a7 ∧ (∃ (min_sum : ℝ), min_sum = 16 ∧ ∀ sum, sum = a1 + a7 → sum ≥ min_sum) :=
sorry

end NUMINAMATH_GPT_min_value_of_a1_plus_a7_l262_26226


namespace NUMINAMATH_GPT_quadratic_no_real_solutions_l262_26279

theorem quadratic_no_real_solutions (k : ℝ) :
  k < -9 / 4 ↔ ∀ x : ℝ, ¬ (x^2 - 3 * x - k = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_solutions_l262_26279


namespace NUMINAMATH_GPT_geometric_sum_4500_l262_26290

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end NUMINAMATH_GPT_geometric_sum_4500_l262_26290


namespace NUMINAMATH_GPT_fisherman_total_fish_l262_26241

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end NUMINAMATH_GPT_fisherman_total_fish_l262_26241


namespace NUMINAMATH_GPT_convex_ngon_sides_l262_26228

theorem convex_ngon_sides (n : ℕ) (h : (n * (n - 3)) / 2 = 27) : n = 9 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_convex_ngon_sides_l262_26228


namespace NUMINAMATH_GPT_shaded_area_l262_26235

theorem shaded_area (r : ℝ) (π : ℝ) (shaded_area : ℝ) (h_r : r = 4) (h_π : π = 3) : shaded_area = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l262_26235


namespace NUMINAMATH_GPT_find_f_2010_l262_26259

def f (x : ℝ) : ℝ := sorry

theorem find_f_2010 (h₁ : ∀ x, f (x + 1) = - f x) (h₂ : f 1 = 4) : f 2010 = -4 :=
by 
  sorry

end NUMINAMATH_GPT_find_f_2010_l262_26259


namespace NUMINAMATH_GPT_percent_blue_marbles_l262_26218

theorem percent_blue_marbles (total_items buttons red_marbles : ℝ) 
  (H1 : buttons = 0.30 * total_items)
  (H2 : red_marbles = 0.50 * (total_items - buttons)) :
  (total_items - buttons - red_marbles) / total_items = 0.35 :=
by 
  sorry

end NUMINAMATH_GPT_percent_blue_marbles_l262_26218


namespace NUMINAMATH_GPT_div_polynomial_l262_26264

noncomputable def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 2
noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*s*x + t

theorem div_polynomial 
  (p q s t : ℝ) 
  (h : ∀ x : ℝ, f x = 0 → g x p q s t = 0) : 
  (p + q + s) * t = -6 :=
by
  sorry

end NUMINAMATH_GPT_div_polynomial_l262_26264


namespace NUMINAMATH_GPT_missed_questions_proof_l262_26249

def num_missed_questions : ℕ := 180

theorem missed_questions_proof (F : ℕ) (h1 : 5 * F + F = 216) : F = 36 ∧ 5 * F = num_missed_questions :=
by {
  sorry
}

end NUMINAMATH_GPT_missed_questions_proof_l262_26249


namespace NUMINAMATH_GPT_chlorine_weight_is_35_l262_26220

def weight_Na : Nat := 23
def weight_O : Nat := 16
def molecular_weight : Nat := 74

theorem chlorine_weight_is_35 (Cl : Nat) 
  (h : molecular_weight = weight_Na + Cl + weight_O) : 
  Cl = 35 := by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_chlorine_weight_is_35_l262_26220


namespace NUMINAMATH_GPT_evaluate_expression_l262_26291

variable (x : ℝ)
variable (hx : x^3 - 3 * x = 6)

theorem evaluate_expression : x^7 - 27 * x^2 = 9 * (x + 1) * (x + 6) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l262_26291


namespace NUMINAMATH_GPT_maximum_sum_is_42_l262_26296

-- Definitions according to the conditions in the problem

def initial_faces : ℕ := 7 -- 2 pentagonal + 5 rectangular
def initial_vertices : ℕ := 10 -- 5 at the top and 5 at the bottom
def initial_edges : ℕ := 15 -- 5 for each pentagon and 5 linking them

def added_faces : ℕ := 5 -- 5 new triangular faces
def added_vertices : ℕ := 1 -- 1 new vertex at the apex of the pyramid
def added_edges : ℕ := 5 -- 5 new edges connecting the new vertex to the pentagon's vertices

-- New quantities after adding the pyramid
def new_faces : ℕ := initial_faces - 1 + added_faces
def new_vertices : ℕ := initial_vertices + added_vertices
def new_edges : ℕ := initial_edges + added_edges

-- Sum of the new shape's characteristics
def sum_faces_vertices_edges : ℕ := new_faces + new_vertices + new_edges

-- Statement to be proved
theorem maximum_sum_is_42 : sum_faces_vertices_edges = 42 := by
  sorry

end NUMINAMATH_GPT_maximum_sum_is_42_l262_26296
