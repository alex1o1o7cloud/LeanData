import Mathlib

namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l2122_212288

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 5 * (a - b)) :
  a / b = 3 / 2 := by
sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l2122_212288


namespace NUMINAMATH_GPT_starting_number_l2122_212208

theorem starting_number (x : ℝ) (h : (x + 26) / 2 = 19) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_starting_number_l2122_212208


namespace NUMINAMATH_GPT_find_lesser_number_l2122_212278

theorem find_lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by sorry

end NUMINAMATH_GPT_find_lesser_number_l2122_212278


namespace NUMINAMATH_GPT_alicia_local_tax_in_cents_l2122_212269

theorem alicia_local_tax_in_cents (hourly_wage : ℝ) (tax_rate : ℝ)
  (h_hourly_wage : hourly_wage = 30) (h_tax_rate : tax_rate = 0.021) :
  (hourly_wage * tax_rate * 100) = 63 := by
  sorry

end NUMINAMATH_GPT_alicia_local_tax_in_cents_l2122_212269


namespace NUMINAMATH_GPT_total_age_in_3_years_l2122_212206

theorem total_age_in_3_years (Sam Sue Kendra : ℕ)
  (h1 : Kendra = 18)
  (h2 : Kendra = 3 * Sam)
  (h3 : Sam = 2 * Sue) :
  Sam + Sue + Kendra + 3 * 3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_age_in_3_years_l2122_212206


namespace NUMINAMATH_GPT_min_value_y_l2122_212258

theorem min_value_y (x : ℝ) (h : x > 1) : 
  ∃ y_min : ℝ, (∀ y, y = (1 / (x - 1) + x) → y ≥ y_min) ∧ y_min = 3 :=
sorry

end NUMINAMATH_GPT_min_value_y_l2122_212258


namespace NUMINAMATH_GPT_stream_speed_l2122_212215

theorem stream_speed (c v : ℝ) (h1 : c - v = 9) (h2 : c + v = 12) : v = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l2122_212215


namespace NUMINAMATH_GPT_number_of_factors_n_l2122_212203

-- Defining the value of n with its prime factorization
def n : ℕ := 2^5 * 3^9 * 5^5

-- Theorem stating the number of natural-number factors of n
theorem number_of_factors_n : 
  (Nat.divisors n).card = 360 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_factors_n_l2122_212203


namespace NUMINAMATH_GPT_plus_signs_count_l2122_212216

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end NUMINAMATH_GPT_plus_signs_count_l2122_212216


namespace NUMINAMATH_GPT_tangent_line_equation_l2122_212293

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + a * x^2 + (a - 3) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_equation (a : ℝ) (h : ∀ x : ℝ, f a (-x) = f a x) :
    9 * (2 : ℝ) - f a 2 - 16 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l2122_212293


namespace NUMINAMATH_GPT_length_after_y_months_isabella_hair_length_l2122_212242

-- Define the initial length of the hair
def initial_length : ℝ := 18

-- Define the growth rate of the hair per month
def growth_rate (x : ℝ) : ℝ := x

-- Define the number of months passed
def months_passed (y : ℕ) : ℕ := y

-- Prove the length of the hair after 'y' months
theorem length_after_y_months (x : ℝ) (y : ℕ) : ℝ :=
  initial_length + growth_rate x * y

-- Theorem statement to prove that the length of Isabella's hair after y months is 18 + xy
theorem isabella_hair_length (x : ℝ) (y : ℕ) : length_after_y_months x y = 18 + x * y :=
by sorry

end NUMINAMATH_GPT_length_after_y_months_isabella_hair_length_l2122_212242


namespace NUMINAMATH_GPT_total_number_of_squares_l2122_212247

theorem total_number_of_squares (n : ℕ) (h : n = 12) : 
  ∃ t, t = 17 :=
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_total_number_of_squares_l2122_212247


namespace NUMINAMATH_GPT_maximum_F_value_l2122_212287

open Real

noncomputable def F (a b c x : ℝ) := abs ((a * x^2 + b * x + c) * (c * x^2 + b * x + a))

theorem maximum_F_value (a b c : ℝ) (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1)
    (hfx : abs (a * x^2 + b * x + c) ≤ 1) :
    ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ F a b c x = 2 := 
  sorry

end NUMINAMATH_GPT_maximum_F_value_l2122_212287


namespace NUMINAMATH_GPT_value_of_a3_a6_a9_l2122_212261

variable (a : ℕ → ℤ) -- Define the sequence a as a function from natural numbers to integers
variable (d : ℤ) -- Define the common difference d as an integer

-- Conditions
axiom h1 : a 1 + a 4 + a 7 = 39
axiom h2 : a 2 + a 5 + a 8 = 33
axiom h3 : ∀ n : ℕ, a (n+1) = a n + d -- This condition ensures the sequence is arithmetic

-- Theorem: We need to prove the value of a_3 + a_6 + a_9 is 27
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 27 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a3_a6_a9_l2122_212261


namespace NUMINAMATH_GPT_women_to_total_population_ratio_l2122_212234

/-- original population of Salem -/
def original_population (pop_leesburg : ℕ) : ℕ := 15 * pop_leesburg

/-- new population after people moved out -/
def new_population (orig_pop : ℕ) (moved_out : ℕ) : ℕ := orig_pop - moved_out

/-- ratio of two numbers -/
def ratio (num : ℕ) (denom : ℕ) : ℚ := num / denom

/-- population data -/
structure PopulationData :=
  (pop_leesburg : ℕ)
  (moved_out : ℕ)
  (women : ℕ)

/-- prove ratio of women to the total population in Salem -/
theorem women_to_total_population_ratio (data : PopulationData)
  (pop_leesburg_eq : data.pop_leesburg = 58940)
  (moved_out_eq : data.moved_out = 130000)
  (women_eq : data.women = 377050) : 
  ratio data.women (new_population (original_population data.pop_leesburg) data.moved_out) = 377050 / 754100 :=
by
  sorry

end NUMINAMATH_GPT_women_to_total_population_ratio_l2122_212234


namespace NUMINAMATH_GPT_system_of_equations_solution_l2122_212238

theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y + 2 * x * y = 11 ∧ 2 * x^2 * y + x * y^2 = 15) ↔
  ((x = 1/2 ∧ y = 5) ∨ (x = 1 ∧ y = 3) ∨ (x = 3/2 ∧ y = 2) ∨ (x = 5/2 ∧ y = 1)) :=
by 
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2122_212238


namespace NUMINAMATH_GPT_cost_of_scooter_l2122_212205

-- Given conditions
variables (M T : ℕ)
axiom h1 : T = M + 4
axiom h2 : T = 15

-- Proof goal: The cost of the scooter is $26
theorem cost_of_scooter : M + T = 26 :=
by sorry

end NUMINAMATH_GPT_cost_of_scooter_l2122_212205


namespace NUMINAMATH_GPT_intersection_with_single_element_union_equals_A_l2122_212211

-- Definitions of the sets A and B
def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

-- Statement for question (1)
theorem intersection_with_single_element (a : ℝ) (H : A = {1, 2} ∧ A ∩ B a = {2}) : a = -1 ∨ a = -3 :=
by
  sorry

-- Statement for question (2)
theorem union_equals_A (a : ℝ) (H1 : A = {1, 2}) (H2 : A ∪ B a = A) : (a ≥ -3 ∧ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_single_element_union_equals_A_l2122_212211


namespace NUMINAMATH_GPT_S_5_is_121_l2122_212267

-- Definitions of the sequence and its terms
def S : ℕ → ℕ := sorry  -- Define S_n
def a : ℕ → ℕ := sorry  -- Define a_n

-- Conditions
axiom S_2 : S 2 = 4
axiom recurrence_relation : ∀ n : ℕ, S (n + 1) = 1 + 2 * S n

-- Proof that S_5 = 121 given the conditions
theorem S_5_is_121 : S 5 = 121 := by
  sorry

end NUMINAMATH_GPT_S_5_is_121_l2122_212267


namespace NUMINAMATH_GPT_determine_a_value_l2122_212232

theorem determine_a_value (a : ℝ) :
  (∀ y₁ y₂ : ℝ, ∃ m₁ m₂ : ℝ, (m₁, y₁) = (a, -2) ∧ (m₂, y₂) = (3, -4) ∧ (m₁ = m₂)) → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_value_l2122_212232


namespace NUMINAMATH_GPT_volume_of_rect_box_l2122_212264

open Real

/-- Proof of the volume of a rectangular box given its face areas -/
theorem volume_of_rect_box (l w h : ℝ) 
  (A1 : l * w = 40) 
  (A2 : w * h = 10) 
  (A3 : l * h = 8) : 
  l * w * h = 40 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_rect_box_l2122_212264


namespace NUMINAMATH_GPT_evaluate_f_difference_l2122_212218

def f (x : ℤ) : ℤ := x^6 + 3 * x^4 - 4 * x^3 + x^2 + 2 * x

theorem evaluate_f_difference : f 3 - f (-3) = -204 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_difference_l2122_212218


namespace NUMINAMATH_GPT_range_of_k_l2122_212296

noncomputable def f (k x : ℝ) := k * x - Real.exp x
noncomputable def g (x : ℝ) := Real.exp x / x

theorem range_of_k (k : ℝ) (h : ∃ x : ℝ, x ≠ 0 ∧ f k x = 0) :
  k < 0 ∨ k ≥ Real.exp 1 := sorry

end NUMINAMATH_GPT_range_of_k_l2122_212296


namespace NUMINAMATH_GPT_line_l_prime_eq_2x_minus_3y_plus_5_l2122_212250

theorem line_l_prime_eq_2x_minus_3y_plus_5 (m : ℝ) (x y : ℝ) : 
  (2 * m + 1) * x + (m + 1) * y + m = 0 →
  (2 * -1 + 1) * (-1) + (1 + 1) * 1 + m = 0 →
  ∀ a b : ℝ, (3 * b, 2 * b) = (3 * 1, 2 * 1) → (a, b) = (-1, 1) → 
  2 * x - 3 * y + 5 = 0 :=
by
  intro h1 h2 a b h3 h4
  sorry

end NUMINAMATH_GPT_line_l_prime_eq_2x_minus_3y_plus_5_l2122_212250


namespace NUMINAMATH_GPT_sin_div_one_minus_tan_eq_neg_three_fourths_l2122_212279

variable (α : ℝ)

theorem sin_div_one_minus_tan_eq_neg_three_fourths
  (h : Real.sin (α - Real.pi / 4) = Real.sqrt 2 / 4) :
  (Real.sin α) / (1 - Real.tan α) = -3 / 4 := sorry

end NUMINAMATH_GPT_sin_div_one_minus_tan_eq_neg_three_fourths_l2122_212279


namespace NUMINAMATH_GPT_salesman_bonus_l2122_212299

theorem salesman_bonus (S B : ℝ) 
  (h1 : S > 10000) 
  (h2 : 0.09 * S + 0.03 * (S - 10000) = 1380) 
  : B = 0.03 * (S - 10000) :=
sorry

end NUMINAMATH_GPT_salesman_bonus_l2122_212299


namespace NUMINAMATH_GPT_subcommittees_with_at_least_one_teacher_l2122_212275

-- Define the total number of members and the count of teachers
def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 5

-- Define binomial coefficient calculation
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Problem statement: number of five-person subcommittees with at least one teacher
theorem subcommittees_with_at_least_one_teacher :
  binom total_members subcommittee_size - binom (total_members - teacher_count) subcommittee_size = 771 := by
  sorry

end NUMINAMATH_GPT_subcommittees_with_at_least_one_teacher_l2122_212275


namespace NUMINAMATH_GPT_pints_in_5_liters_l2122_212202

-- Define the condition based on the given conversion factor from liters to pints
def conversion_factor : ℝ := 2.1

-- The statement we need to prove
theorem pints_in_5_liters : 5 * conversion_factor = 10.5 :=
by sorry

end NUMINAMATH_GPT_pints_in_5_liters_l2122_212202


namespace NUMINAMATH_GPT_smallest_sum_zero_l2122_212248

theorem smallest_sum_zero : ∃ x ∈ ({-1, -2, 1, 2} : Set ℤ), ∀ y ∈ ({-1, -2, 1, 2} : Set ℤ), x + 0 ≤ y + 0 :=
sorry

end NUMINAMATH_GPT_smallest_sum_zero_l2122_212248


namespace NUMINAMATH_GPT_find_square_l2122_212286

theorem find_square (q x : ℝ) 
  (h1 : x + q = 74) 
  (h2 : x + 2 * q^2 = 180) : 
  x = 66 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_square_l2122_212286


namespace NUMINAMATH_GPT_work_rate_solution_l2122_212251

theorem work_rate_solution (x : ℝ) (hA : 60 > 0) (hB : x > 0) (hTogether : 15 > 0) :
  (1 / 60 + 1 / x = 1 / 15) → (x = 20) :=
by 
  sorry -- Proof Placeholder

end NUMINAMATH_GPT_work_rate_solution_l2122_212251


namespace NUMINAMATH_GPT_log_six_two_l2122_212246

noncomputable def log_six (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_six_two (a : ℝ) (h : log_six 3 = a) : log_six 2 = 1 - a :=
by
  sorry

end NUMINAMATH_GPT_log_six_two_l2122_212246


namespace NUMINAMATH_GPT_neg_product_B_l2122_212281

def expr_A := (-1 / 3) * (1 / 4) * (-6)
def expr_B := (-9) * (1 / 8) * (-4 / 7) * 7 * (-1 / 3)
def expr_C := (-3) * (-1 / 2) * 7 * 0
def expr_D := (-1 / 5) * 6 * (-2 / 3) * (-5) * (-1 / 2)

theorem neg_product_B :
  expr_B < 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_product_B_l2122_212281


namespace NUMINAMATH_GPT_evaluate_expression_l2122_212230

open Complex

theorem evaluate_expression (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + a * b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 18 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2122_212230


namespace NUMINAMATH_GPT_red_balls_approximation_l2122_212212

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end NUMINAMATH_GPT_red_balls_approximation_l2122_212212


namespace NUMINAMATH_GPT_greatest_area_difference_l2122_212265

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) 
  (h₁ : 2 * l₁ + 2 * w₁ = 160) 
  (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  1521 = (l₁ * w₁ - l₂ * w₂) → 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 1600 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) ∧ 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 79 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) :=
sorry

end NUMINAMATH_GPT_greatest_area_difference_l2122_212265


namespace NUMINAMATH_GPT_total_weight_30_l2122_212239

-- Definitions of initial weights and ratio conditions
variables (a b : ℕ)
def initial_weights (h1 : a = 4 * b) : Prop := True

-- Definitions of transferred weights
def transferred_weights (a' b' : ℕ) (h2 : a' = a - 10) (h3 : b' = b + 10) : Prop := True

-- Definition of the new ratio condition
def new_ratio (a' b' : ℕ) (h4 : 8 * a' = 7 * b') : Prop := True

-- The final proof statement
theorem total_weight_30 (a b a' b' : ℕ)
    (h1 : a = 4 * b) 
    (h2 : a' = a - 10) 
    (h3 : b' = b + 10)
    (h4 : 8 * a' = 7 * b') : a + b = 30 := 
    sorry

end NUMINAMATH_GPT_total_weight_30_l2122_212239


namespace NUMINAMATH_GPT_ratio_of_radii_l2122_212227

open Real

theorem ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 5 * π * a^2) : a / b = 1 / sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_l2122_212227


namespace NUMINAMATH_GPT_find_y_l2122_212298

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y : ∃ y : ℕ, F 3 y 5 15 = 490 ∧ y = 6 := by
  sorry

end NUMINAMATH_GPT_find_y_l2122_212298


namespace NUMINAMATH_GPT_hyperbola_range_of_k_l2122_212207

theorem hyperbola_range_of_k (x y k : ℝ) :
  (∃ x y : ℝ, (x^2 / (1 - 2 * k) - y^2 / (k - 2) = 1) ∧ (1 - 2 * k < 0) ∧ (k - 2 < 0)) →
  (1 / 2 < k ∧ k < 2) :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_range_of_k_l2122_212207


namespace NUMINAMATH_GPT_shopkeeper_loss_amount_l2122_212219

theorem shopkeeper_loss_amount (total_stock_worth : ℝ)
                               (portion_sold_at_profit : ℝ)
                               (portion_sold_at_loss : ℝ)
                               (profit_percentage : ℝ)
                               (loss_percentage : ℝ) :
  total_stock_wworth = 14999.999999999996 →
  portion_sold_at_profit = 0.2 →
  portion_sold_at_loss = 0.8 →
  profit_percentage = 0.10 →
  loss_percentage = 0.05 →
  (total_stock_worth - ((portion_sold_at_profit * total_stock_worth * (1 + profit_percentage)) + 
                        (portion_sold_at_loss * total_stock_worth * (1 - loss_percentage)))) = 300 := 
by 
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_amount_l2122_212219


namespace NUMINAMATH_GPT_ratio_of_products_l2122_212277

variable (a b c d : ℚ) -- assuming a, b, c, d are rational numbers

theorem ratio_of_products (h1 : a = 3 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end NUMINAMATH_GPT_ratio_of_products_l2122_212277


namespace NUMINAMATH_GPT_cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l2122_212235

theorem cannot_be_expressed_as_difference_of_squares (a b : ℤ) (h : 2006 = a^2 - b^2) : False := sorry

theorem can_be_expressed_as_difference_of_squares_2004 : ∃ (a b : ℤ), 2004 = a^2 - b^2 := by
  use 502, 500
  norm_num

theorem can_be_expressed_as_difference_of_squares_2005 : ∃ (a b : ℤ), 2005 = a^2 - b^2 := by
  use 1003, 1002
  norm_num

theorem can_be_expressed_as_difference_of_squares_2007 : ∃ (a b : ℤ), 2007 = a^2 - b^2 := by
  use 1004, 1003
  norm_num

end NUMINAMATH_GPT_cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l2122_212235


namespace NUMINAMATH_GPT_circle_equation_correct_l2122_212233

def line_through_fixed_point (a : ℝ) :=
  ∀ x y : ℝ, (x + y - 1) - a * (x + 1) = 0 → x = -1 ∧ y = 2

def equation_of_circle (x y: ℝ) :=
  (x + 1)^2 + (y - 2)^2 = 5

theorem circle_equation_correct (a : ℝ) (h : line_through_fixed_point a) :
  ∀ x y : ℝ, equation_of_circle x y ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_correct_l2122_212233


namespace NUMINAMATH_GPT_solve_inequality_l2122_212241

theorem solve_inequality (x : ℝ) (h : x ≠ -2 / 3) :
  3 - (1 / (3 * x + 2)) < 5 ↔ (x < -7 / 6 ∨ x > -2 / 3) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2122_212241


namespace NUMINAMATH_GPT_farmer_price_per_dozen_l2122_212243

noncomputable def price_per_dozen 
(farmer_chickens : ℕ) 
(eggs_per_chicken : ℕ) 
(total_money_made : ℕ) 
(total_weeks : ℕ) 
(eggs_per_dozen : ℕ) 
: ℕ :=
total_money_made / (total_weeks * (farmer_chickens * eggs_per_chicken) / eggs_per_dozen)

theorem farmer_price_per_dozen 
  (farmer_chickens : ℕ) 
  (eggs_per_chicken : ℕ) 
  (total_money_made : ℕ) 
  (total_weeks : ℕ) 
  (eggs_per_dozen : ℕ) 
  (h_chickens : farmer_chickens = 46) 
  (h_eggs_per_chicken : eggs_per_chicken = 6) 
  (h_money : total_money_made = 552) 
  (h_weeks : total_weeks = 8) 
  (h_dozen : eggs_per_dozen = 12) 
: price_per_dozen farmer_chickens eggs_per_chicken total_money_made total_weeks eggs_per_dozen = 3 := 
by 
  rw [h_chickens, h_eggs_per_chicken, h_money, h_weeks, h_dozen]
  have : (552 : ℕ) / (8 * (46 * 6) / 12) = 3 := by norm_num
  exact this

end NUMINAMATH_GPT_farmer_price_per_dozen_l2122_212243


namespace NUMINAMATH_GPT_distance_between_x_intercepts_l2122_212270

theorem distance_between_x_intercepts (x1 y1 : ℝ) 
  (m1 m2 : ℝ)
  (hx1 : x1 = 10) (hy1 : y1 = 15)
  (hm1 : m1 = 3) (hm2 : m2 = 5) :
  let x_intercept1 := (y1 - m1 * x1) / -m1
  let x_intercept2 := (y1 - m2 * x1) / -m2
  dist (x_intercept1, 0) (x_intercept2, 0) = 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_x_intercepts_l2122_212270


namespace NUMINAMATH_GPT_range_of_function_l2122_212255

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = 4^x + 2^x - 3 ↔ y > -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l2122_212255


namespace NUMINAMATH_GPT_wall_area_l2122_212260

theorem wall_area (width : ℝ) (height : ℝ) (h1 : width = 2) (h2 : height = 4) : width * height = 8 := by
  sorry

end NUMINAMATH_GPT_wall_area_l2122_212260


namespace NUMINAMATH_GPT_pirate_treasure_chest_coins_l2122_212297

theorem pirate_treasure_chest_coins:
  ∀ (gold_coins silver_coins bronze_coins: ℕ) (chests: ℕ),
    gold_coins = 3500 →
    silver_coins = 500 →
    bronze_coins = 2 * silver_coins →
    chests = 5 →
    (gold_coins / chests + silver_coins / chests + bronze_coins / chests = 1000) :=
by
  intros gold_coins silver_coins bronze_coins chests gold_eq silv_eq bron_eq chest_eq
  sorry

end NUMINAMATH_GPT_pirate_treasure_chest_coins_l2122_212297


namespace NUMINAMATH_GPT_jorge_goals_this_season_l2122_212240

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end NUMINAMATH_GPT_jorge_goals_this_season_l2122_212240


namespace NUMINAMATH_GPT_shoveling_driveway_time_l2122_212295

theorem shoveling_driveway_time (S : ℝ) (Wayne_rate : ℝ) (combined_rate : ℝ) :
  (S = 1 / 7) → (Wayne_rate = 6 * S) → (combined_rate = Wayne_rate + S) → (combined_rate = 1) :=
by { sorry }

end NUMINAMATH_GPT_shoveling_driveway_time_l2122_212295


namespace NUMINAMATH_GPT_largest_square_factor_of_1800_l2122_212257

theorem largest_square_factor_of_1800 : 
  ∃ n, n^2 ∣ 1800 ∧ ∀ m, m^2 ∣ 1800 → m^2 ≤ n^2 :=
sorry

end NUMINAMATH_GPT_largest_square_factor_of_1800_l2122_212257


namespace NUMINAMATH_GPT_area_of_sector_l2122_212280

/-- The area of a sector of a circle with radius 10 meters and central angle 42 degrees is 35/3 * pi square meters. -/
theorem area_of_sector (r θ : ℕ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360 : ℝ) * (Real.pi : ℝ) * (r : ℝ)^2 = (35 / 3 : ℝ) * (Real.pi : ℝ) :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_sector_l2122_212280


namespace NUMINAMATH_GPT_certain_amount_is_19_l2122_212209

theorem certain_amount_is_19 (x y certain_amount : ℤ) 
  (h1 : x + y = 15)
  (h2 : 3 * x = 5 * y - certain_amount)
  (h3 : x = 7) : 
  certain_amount = 19 :=
by
  sorry

end NUMINAMATH_GPT_certain_amount_is_19_l2122_212209


namespace NUMINAMATH_GPT_shaded_area_correct_l2122_212236

def first_rectangle_area (w l : ℕ) : ℕ := w * l
def second_rectangle_area (w l : ℕ) : ℕ := w * l
def overlap_triangle_area (b h : ℕ) : ℕ := (b * h) / 2
def total_shaded_area (area1 area2 overlap : ℕ) : ℕ := area1 + area2 - overlap

theorem shaded_area_correct :
  let w1 := 4
  let l1 := 12
  let w2 := 5
  let l2 := 10
  let b := 4
  let h := 5
  let area1 := first_rectangle_area w1 l1
  let area2 := second_rectangle_area w2 l2
  let overlap := overlap_triangle_area b h
  total_shaded_area area1 area2 overlap = 88 := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l2122_212236


namespace NUMINAMATH_GPT_ratio_revenue_l2122_212237

variable (N D J : ℝ)

theorem ratio_revenue (h1 : J = N / 3) (h2 : D = 2.5 * (N + J) / 2) : N / D = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_revenue_l2122_212237


namespace NUMINAMATH_GPT_unit_digit_product_7858_1086_4582_9783_l2122_212291

theorem unit_digit_product_7858_1086_4582_9783 : 
  (7858 * 1086 * 4582 * 9783) % 10 = 8 :=
by
  -- Given that the unit digits of the numbers are 8, 6, 2, and 3.
  let d1 := 7858 % 10 -- This unit digit is 8
  let d2 := 1086 % 10 -- This unit digit is 6
  let d3 := 4582 % 10 -- This unit digit is 2
  let d4 := 9783 % 10 -- This unit digit is 3
  -- We need to prove that the unit digit of the product is 8
  sorry -- The actual proof steps are skipped

end NUMINAMATH_GPT_unit_digit_product_7858_1086_4582_9783_l2122_212291


namespace NUMINAMATH_GPT_sin_identity_l2122_212266

theorem sin_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + π / 4) ^ 2 = 5 / 6 := 
sorry

end NUMINAMATH_GPT_sin_identity_l2122_212266


namespace NUMINAMATH_GPT_find_m_eq_2_l2122_212229

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_eq_2_l2122_212229


namespace NUMINAMATH_GPT_old_man_gold_coins_l2122_212285

theorem old_man_gold_coins (x y : ℕ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := 
sorry

end NUMINAMATH_GPT_old_man_gold_coins_l2122_212285


namespace NUMINAMATH_GPT_gambler_largest_amount_proof_l2122_212217

noncomputable def largest_amount_received_back (initial_amount : ℝ) (value_25 : ℝ) (value_75 : ℝ) (value_250 : ℝ) 
                                               (total_lost_chips : ℝ) (coef_25_75_lost : ℝ) (coef_75_250_lost : ℝ) : ℝ :=
    initial_amount - (
    coef_25_75_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_25 +
    (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_75 +
    coef_75_250_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_250)

theorem gambler_largest_amount_proof :
    let initial_amount := 15000
    let value_25 := 25
    let value_75 := 75
    let value_250 := 250
    let total_lost_chips := 40
    let coef_25_75_lost := 2 -- number of lost $25 chips is twice the number of lost $75 chips
    let coef_75_250_lost := 2 -- number of lost $250 chips is twice the number of lost $75 chips
    largest_amount_received_back initial_amount value_25 value_75 value_250 total_lost_chips coef_25_75_lost coef_75_250_lost = 10000 :=
by {
    sorry
}

end NUMINAMATH_GPT_gambler_largest_amount_proof_l2122_212217


namespace NUMINAMATH_GPT_center_in_triangle_probability_l2122_212283

theorem center_in_triangle_probability (n : ℕ) :
  let vertices := 2 * n + 1
  let total_ways := vertices.choose 3
  let no_center_ways := vertices * (n.choose 2) / 2
  let p_no_center := no_center_ways / total_ways
  let p_center := 1 - p_no_center
  p_center = (n + 1) / (4 * n - 2) := sorry

end NUMINAMATH_GPT_center_in_triangle_probability_l2122_212283


namespace NUMINAMATH_GPT_Evan_earnings_Markese_less_than_Evan_l2122_212276

-- Definitions from conditions
def MarkeseEarnings : ℕ := 16
def TotalEarnings : ℕ := 37

-- Theorem statements
theorem Evan_earnings (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E = 21 :=
by {
  sorry
}

theorem Markese_less_than_Evan (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E - MarkeseEarnings = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_Evan_earnings_Markese_less_than_Evan_l2122_212276


namespace NUMINAMATH_GPT_elena_subtracts_99_to_compute_49_squared_l2122_212222

noncomputable def difference_between_squares_50_49 : ℕ := 99

theorem elena_subtracts_99_to_compute_49_squared :
  ∀ (n : ℕ), n = 50 → (n - 1)^2 = n^2 - difference_between_squares_50_49 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_elena_subtracts_99_to_compute_49_squared_l2122_212222


namespace NUMINAMATH_GPT_compute_expression_l2122_212244

theorem compute_expression (x : ℝ) (hx : x + 1 / x = 7) : 
  (x - 3)^2 + 36 / (x - 3)^2 = 12.375 := 
  sorry

end NUMINAMATH_GPT_compute_expression_l2122_212244


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2122_212223

theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11) 
  (h2 : b - s = 3) : b = 7 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2122_212223


namespace NUMINAMATH_GPT_general_term_of_sequence_l2122_212284

def S (n : ℕ) : ℕ := n^2 + 3 * n + 1

def a (n : ℕ) : ℕ := 
  if n = 1 then 5 
  else 2 * n + 2

theorem general_term_of_sequence (n : ℕ) : 
  a n = if n = 1 then 5 else (S n - S (n - 1)) := 
by 
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l2122_212284


namespace NUMINAMATH_GPT_greatest_divisor_4665_6905_l2122_212290

def digits_sum (n : ℕ) : ℕ :=
(n.digits 10).sum

theorem greatest_divisor_4665_6905 :
  ∃ n : ℕ, (n ∣ 4665) ∧ (n ∣ 6905) ∧ (digits_sum n = 4) ∧
  (∀ m : ℕ, ((m ∣ 4665) ∧ (m ∣ 6905) ∧ (digits_sum m = 4)) → (m ≤ n)) :=
sorry

end NUMINAMATH_GPT_greatest_divisor_4665_6905_l2122_212290


namespace NUMINAMATH_GPT_counterexample_to_proposition_l2122_212214

theorem counterexample_to_proposition (a b : ℝ) (ha : a = 1) (hb : b = -1) :
  a > b ∧ ¬ (1 / a < 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_counterexample_to_proposition_l2122_212214


namespace NUMINAMATH_GPT_sum_of_number_and_its_radical_conjugate_l2122_212289

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_its_radical_conjugate_l2122_212289


namespace NUMINAMATH_GPT_greatest_number_of_police_officers_needed_l2122_212245

-- Define the conditions within Math City
def number_of_streets : ℕ := 10
def number_of_tunnels : ℕ := 2
def intersections_without_tunnels : ℕ := (number_of_streets * (number_of_streets - 1)) / 2
def intersections_bypassed_by_tunnels : ℕ := number_of_tunnels

-- Define the number of police officers required (which is the same as the number of intersections not bypassed)
def police_officers_needed : ℕ := intersections_without_tunnels - intersections_bypassed_by_tunnels

-- The main theorem: Given the conditions, the greatest number of police officers needed is 43.
theorem greatest_number_of_police_officers_needed : police_officers_needed = 43 := 
by {
  -- Proof would go here, but we'll use sorry to indicate it's not provided.
  sorry
}

end NUMINAMATH_GPT_greatest_number_of_police_officers_needed_l2122_212245


namespace NUMINAMATH_GPT_strawberries_eaten_l2122_212272

-- Definitions based on the conditions
def strawberries_picked : ℕ := 35
def strawberries_remaining : ℕ := 33

-- Statement of the proof problem
theorem strawberries_eaten :
  strawberries_picked - strawberries_remaining = 2 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_eaten_l2122_212272


namespace NUMINAMATH_GPT_find_s_l2122_212228

theorem find_s (a b r1 r2 : ℝ) (h1 : r1 + r2 = -a) (h2 : r1 * r2 = b) :
    let new_root1 := (r1 + r2) * (r1 + r2)
    let new_root2 := (r1 * r2) * (r1 + r2)
    let s := b * a - a * a
    s = ab - a^2 :=
  by
    -- the proof goes here
    sorry

end NUMINAMATH_GPT_find_s_l2122_212228


namespace NUMINAMATH_GPT_percentage_big_bottles_sold_l2122_212231

-- Definitions of conditions
def total_small_bottles : ℕ := 6000
def total_big_bottles : ℕ := 14000
def small_bottles_sold_percentage : ℕ := 20
def total_bottles_remaining : ℕ := 15580

-- Theorem statement
theorem percentage_big_bottles_sold : 
  let small_bottles_sold := (small_bottles_sold_percentage * total_small_bottles) / 100
  let small_bottles_remaining := total_small_bottles - small_bottles_sold
  let big_bottles_remaining := total_bottles_remaining - small_bottles_remaining
  let big_bottles_sold := total_big_bottles - big_bottles_remaining
  (100 * big_bottles_sold) / total_big_bottles = 23 := 
by
  sorry

end NUMINAMATH_GPT_percentage_big_bottles_sold_l2122_212231


namespace NUMINAMATH_GPT_hyperbola_intersection_l2122_212226

theorem hyperbola_intersection (b : ℝ) (h₁ : b > 0) :
  (b > 1) → (∀ x y : ℝ, ((x + 3 * y - 1 = 0) → ( ∃ x y : ℝ, (x^2 / 4 - y^2 / b^2 = 1) ∧ (x + 3 * y - 1 = 0))))
  :=
  sorry

end NUMINAMATH_GPT_hyperbola_intersection_l2122_212226


namespace NUMINAMATH_GPT_set_intersection_complement_l2122_212224

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem set_intersection_complement :
  (compl A ∩ B) = {x | 0 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l2122_212224


namespace NUMINAMATH_GPT_certain_number_is_1862_l2122_212201

theorem certain_number_is_1862 (G N : ℕ) (hG: G = 4) (hN: ∃ k : ℕ, N = G * k + 6) (h1856: ∃ m : ℕ, 1856 = G * m + 4) : N = 1862 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_1862_l2122_212201


namespace NUMINAMATH_GPT_chapter_page_difference_l2122_212256

/-- The first chapter of a book has 37 pages -/
def first_chapter_pages : Nat := 37

/-- The second chapter of a book has 80 pages -/
def second_chapter_pages : Nat := 80

/-- Prove the difference in the number of pages between the second and the first chapter is 43 -/
theorem chapter_page_difference : (second_chapter_pages - first_chapter_pages) = 43 := by
  sorry

end NUMINAMATH_GPT_chapter_page_difference_l2122_212256


namespace NUMINAMATH_GPT_odd_number_diff_of_squares_l2122_212200

theorem odd_number_diff_of_squares (k : ℕ) : ∃ n : ℕ, k = (n+1)^2 - n^2 ↔ ∃ m : ℕ, k = 2 * m + 1 := 
by 
  sorry

end NUMINAMATH_GPT_odd_number_diff_of_squares_l2122_212200


namespace NUMINAMATH_GPT_mady_balls_2010th_step_l2122_212213

theorem mady_balls_2010th_step :
  let base_5_digits (n : Nat) : List Nat := (Nat.digits 5 n)
  (base_5_digits 2010).sum = 6 := by
  sorry

end NUMINAMATH_GPT_mady_balls_2010th_step_l2122_212213


namespace NUMINAMATH_GPT_petya_oranges_l2122_212274

theorem petya_oranges (m o : ℕ) (h1 : m + 6 * m + o = 20) (h2 : 6 * m > o) : o = 6 :=
by 
  sorry

end NUMINAMATH_GPT_petya_oranges_l2122_212274


namespace NUMINAMATH_GPT_ratio_initial_to_doubled_l2122_212225

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 5) = 105) : x / (2 * x) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_initial_to_doubled_l2122_212225


namespace NUMINAMATH_GPT_domain_of_f_l2122_212273

noncomputable def f (x : ℝ) : ℝ := (x + 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : 
  {x : ℝ | Real.sqrt (x^2 - 5 * x + 6) ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2122_212273


namespace NUMINAMATH_GPT_increasing_sequence_range_of_a_l2122_212263

theorem increasing_sequence_range_of_a (a : ℝ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, a_n n = a * n ^ 2 + n) (increasing : ∀ n : ℕ, a_n (n + 1) > a_n n) : 0 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_increasing_sequence_range_of_a_l2122_212263


namespace NUMINAMATH_GPT_brandon_investment_percentage_l2122_212271

noncomputable def jackson_initial_investment : ℕ := 500
noncomputable def brandon_initial_investment : ℕ := 500
noncomputable def jackson_final_investment : ℕ := 2000
noncomputable def difference_in_investments : ℕ := 1900
noncomputable def brandon_final_investment : ℕ := jackson_final_investment - difference_in_investments

theorem brandon_investment_percentage :
  (brandon_final_investment : ℝ) / (brandon_initial_investment : ℝ) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_brandon_investment_percentage_l2122_212271


namespace NUMINAMATH_GPT_time_for_tom_to_finish_wall_l2122_212253

theorem time_for_tom_to_finish_wall (avery_rate tom_rate : ℝ) (combined_duration : ℝ) (remaining_wall : ℝ) :
  avery_rate = 1 / 2 ∧ tom_rate = 1 / 4 ∧ combined_duration = 1 ∧ remaining_wall = 1 / 4 →
  (remaining_wall / tom_rate) = 1 :=
by
  intros h
  -- Definitions from conditions
  let avery_rate := 1 / 2
  let tom_rate := 1 / 4
  let combined_duration := 1
  let remaining_wall := 1 / 4
  -- Question to be proven
  sorry

end NUMINAMATH_GPT_time_for_tom_to_finish_wall_l2122_212253


namespace NUMINAMATH_GPT_range_x_minus_y_l2122_212204

-- Definition of the curve in polar coordinates
def curve_polar (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta + 2 * Real.sin theta

-- Conversion to rectangular coordinates
noncomputable def curve_rectangular (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 2 * y

-- The final Lean 4 statement
theorem range_x_minus_y (x y : ℝ) (h : curve_rectangular x y) :
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_range_x_minus_y_l2122_212204


namespace NUMINAMATH_GPT_enclosed_area_l2122_212282

theorem enclosed_area {x y : ℝ} (h : x^2 + y^2 = 2 * |x| + 2 * |y|) : ∃ (A : ℝ), A = 8 :=
sorry

end NUMINAMATH_GPT_enclosed_area_l2122_212282


namespace NUMINAMATH_GPT_symmetric_point_of_P_l2122_212294

-- Let P be a point with coordinates (5, -3)
def P : ℝ × ℝ := (5, -3)

-- Definition of the symmetric point with respect to the x-axis
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Theorem stating that the symmetric point to P with respect to the x-axis is (5, 3)
theorem symmetric_point_of_P : symmetric_point P = (5, 3) := 
  sorry

end NUMINAMATH_GPT_symmetric_point_of_P_l2122_212294


namespace NUMINAMATH_GPT_vans_for_field_trip_l2122_212292

-- Definitions based on conditions
def students := 25
def adults := 5
def van_capacity := 5

-- Calculate total number of people
def total_people := students + adults

-- Calculate number of vans needed
def vans_needed := total_people / van_capacity

-- Theorem statement
theorem vans_for_field_trip : vans_needed = 6 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_vans_for_field_trip_l2122_212292


namespace NUMINAMATH_GPT_min_sides_of_polygon_that_overlaps_after_rotation_l2122_212262

theorem min_sides_of_polygon_that_overlaps_after_rotation (θ : ℝ) (n : ℕ) 
  (hθ: θ = 36) (hdiv: 360 % θ = 0) :
    n = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_sides_of_polygon_that_overlaps_after_rotation_l2122_212262


namespace NUMINAMATH_GPT_largest_of_three_consecutive_integers_sum_18_l2122_212210

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_three_consecutive_integers_sum_18_l2122_212210


namespace NUMINAMATH_GPT_inverse_proportional_l2122_212259

/-- Given that α is inversely proportional to β and α = -3 when β = -6,
    prove that α = 9/4 when β = 8. --/
theorem inverse_proportional (α β : ℚ) 
  (h1 : α * β = 18)
  (h2 : β = 8) : 
  α = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportional_l2122_212259


namespace NUMINAMATH_GPT_pears_left_l2122_212221

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) : 
  jason_pears + keith_pears - mike_ate = 81 := 
by 
  sorry

end NUMINAMATH_GPT_pears_left_l2122_212221


namespace NUMINAMATH_GPT_original_triangle_area_l2122_212220

theorem original_triangle_area (area_of_new_triangle : ℝ) (side_length_ratio : ℝ) (quadrupled : side_length_ratio = 4) (new_area : area_of_new_triangle = 128) : 
  (area_of_new_triangle / side_length_ratio ^ 2) = 8 := by
  sorry

end NUMINAMATH_GPT_original_triangle_area_l2122_212220


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l2122_212268

theorem sum_of_fourth_powers (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 37 / 6 := 
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l2122_212268


namespace NUMINAMATH_GPT_art_class_students_not_in_science_l2122_212252

theorem art_class_students_not_in_science (n S A S_inter_A_only_A : ℕ) 
  (h_n : n = 120) 
  (h_S : S = 85) 
  (h_A : A = 65) 
  (h_union: n = S + A - S_inter_A_only_A) : 
  S_inter_A_only_A = 30 → 
  A - S_inter_A_only_A = 35 :=
by
  intros h
  rw [h]
  sorry

end NUMINAMATH_GPT_art_class_students_not_in_science_l2122_212252


namespace NUMINAMATH_GPT_ratio_of_weights_l2122_212249

variable (x : ℝ)

-- Conditions as definitions in Lean 4
def seth_loss : ℝ := 17.5
def jerome_loss : ℝ := 17.5 * x
def veronica_loss : ℝ := 17.5 + 1.5 -- 19 pounds
def total_loss : ℝ := seth_loss + jerome_loss x + veronica_loss

-- Statement to prove
theorem ratio_of_weights (h : total_loss x = 89) : jerome_loss x / seth_loss = 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_weights_l2122_212249


namespace NUMINAMATH_GPT_widget_cost_reduction_l2122_212254

theorem widget_cost_reduction:
  ∀ (C C_reduced : ℝ), 
  6 * C = 27.60 → 
  8 * C_reduced = 27.60 → 
  C - C_reduced = 1.15 := 
by
  intros C C_reduced h1 h2
  sorry

end NUMINAMATH_GPT_widget_cost_reduction_l2122_212254
