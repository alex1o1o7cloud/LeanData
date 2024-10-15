import Mathlib

namespace NUMINAMATH_GPT_prism_is_five_sided_l2164_216484

-- Definitions based on problem conditions
def prism_faces (total_faces base_faces : Nat) := total_faces = 7 ∧ base_faces = 2

-- Theorem to prove based on the conditions
theorem prism_is_five_sided (total_faces base_faces : Nat) (h : prism_faces total_faces base_faces) : total_faces - base_faces = 5 :=
sorry

end NUMINAMATH_GPT_prism_is_five_sided_l2164_216484


namespace NUMINAMATH_GPT_cost_price_as_percentage_l2164_216498

theorem cost_price_as_percentage (SP CP : ℝ) 
  (profit_percentage : ℝ := 4.166666666666666) 
  (P : ℝ := SP - CP)
  (profit_eq : P = (profit_percentage / 100) * SP) :
  CP = (95.83333333333334 / 100) * SP := 
by
  sorry

end NUMINAMATH_GPT_cost_price_as_percentage_l2164_216498


namespace NUMINAMATH_GPT_xiaoyu_reading_days_l2164_216492

theorem xiaoyu_reading_days
  (h1 : ∀ (p d : ℕ), p = 15 → d = 24 → p * d = 360)
  (h2 : ∀ (p t : ℕ), t = 360 → p = 18 → t / p = 20) :
  ∀ d : ℕ, d = 20 :=
by
  sorry

end NUMINAMATH_GPT_xiaoyu_reading_days_l2164_216492


namespace NUMINAMATH_GPT_wrenches_in_comparison_group_l2164_216440

theorem wrenches_in_comparison_group (H W : ℝ) (x : ℕ) 
  (h1 : W = 2 * H)
  (h2 : 2 * H + 2 * W = (1 / 3) * (8 * H + x * W)) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_wrenches_in_comparison_group_l2164_216440


namespace NUMINAMATH_GPT_pens_given_to_sharon_l2164_216478

def initial_pens : Nat := 20
def mikes_pens : Nat := 22
def final_pens : Nat := 65

def total_pens_after_mike : Nat := initial_pens + mikes_pens
def total_pens_after_cindy : Nat := total_pens_after_mike * 2

theorem pens_given_to_sharon :
  total_pens_after_cindy - final_pens = 19 :=
by
  sorry

end NUMINAMATH_GPT_pens_given_to_sharon_l2164_216478


namespace NUMINAMATH_GPT_solve_for_y_l2164_216460

theorem solve_for_y :
  ∃ (y : ℝ), 
    (∑' n : ℕ, (4 * (n + 1) - 2) * y^n) = 100 ∧ |y| < 1 ∧ y = 0.6036 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l2164_216460


namespace NUMINAMATH_GPT_find_larger_number_l2164_216446

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 10) : L = 1636 := 
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l2164_216446


namespace NUMINAMATH_GPT_g_at_50_l2164_216455

variable (g : ℝ → ℝ)

axiom g_functional_eq (x y : ℝ) : g (x * y) = x * g y
axiom g_at_1 : g 1 = 40

theorem g_at_50 : g 50 = 2000 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_g_at_50_l2164_216455


namespace NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l2164_216497

variable (a : ℝ)

theorem condition_necessary_but_not_sufficient (h : a^2 < 1) : (a < 1) ∧ (¬(a < 1 → a^2 < 1)) := sorry

end NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l2164_216497


namespace NUMINAMATH_GPT_initial_people_lifting_weights_l2164_216473

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_initial_people_lifting_weights_l2164_216473


namespace NUMINAMATH_GPT_diff_g_eq_l2164_216402

def g (n : ℤ) : ℚ := (1/6) * n * (n+1) * (n+3)

theorem diff_g_eq :
  ∀ (r : ℤ), g r - g (r - 1) = (3/2) * r^2 + (5/2) * r :=
by
  intro r
  sorry

end NUMINAMATH_GPT_diff_g_eq_l2164_216402


namespace NUMINAMATH_GPT_xiaoming_pens_l2164_216450

theorem xiaoming_pens (P M : ℝ) (hP : P > 0) (hM : M > 0) :
  (M / (7 / 8 * P) - M / P = 13) → (M / P = 91) := 
by
  sorry

end NUMINAMATH_GPT_xiaoming_pens_l2164_216450


namespace NUMINAMATH_GPT_sum_of_a_and_b_l2164_216438

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l2164_216438


namespace NUMINAMATH_GPT_calculate_expression_l2164_216466

theorem calculate_expression : -2 - 2 * Real.sin (Real.pi / 4) + (Real.pi - 3.14) * 0 + (-1) ^ 3 = -3 - Real.sqrt 2 := by 
sorry

end NUMINAMATH_GPT_calculate_expression_l2164_216466


namespace NUMINAMATH_GPT_least_number_to_subtract_l2164_216477

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 42398) (h2 : d = 15) (h3 : r = 8) : 
  ∃ k, n - r = k * d :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l2164_216477


namespace NUMINAMATH_GPT_maximum_n_value_l2164_216490

theorem maximum_n_value (a b c d : ℝ) (n : ℕ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > d) 
(h₃ : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end NUMINAMATH_GPT_maximum_n_value_l2164_216490


namespace NUMINAMATH_GPT_daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l2164_216425

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end NUMINAMATH_GPT_daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l2164_216425


namespace NUMINAMATH_GPT_product_of_integers_P_Q_R_S_l2164_216441

theorem product_of_integers_P_Q_R_S (P Q R S : ℤ)
  (h1 : 0 < P) (h2 : 0 < Q) (h3 : 0 < R) (h4 : 0 < S)
  (h_sum : P + Q + R + S = 50)
  (h_rel : P + 4 = Q - 4 ∧ P + 4 = R * 3 ∧ P + 4 = S / 3) :
  P * Q * R * S = 43 * 107 * 75 * 225 / 1536 := 
by { sorry }

end NUMINAMATH_GPT_product_of_integers_P_Q_R_S_l2164_216441


namespace NUMINAMATH_GPT_max_value_of_f_l2164_216412

-- Define the function f(x) = 5x - x^2
def f (x : ℝ) : ℝ := 5 * x - x^2

-- The theorem we want to prove, stating the maximum value of f(x) is 6.25
theorem max_value_of_f : ∃ x, f x = 6.25 :=
by
  -- Placeholder proof, to be completed
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2164_216412


namespace NUMINAMATH_GPT_problem_1_problem_2_l2164_216491

def f (x : ℝ) : ℝ := abs (2 * x + 3) + abs (2 * x - 1)

theorem problem_1 (x : ℝ) : (f x ≤ 5) ↔ (-7/4 ≤ x ∧ x ≤ 3/4) :=
by sorry

theorem problem_2 (m : ℝ) : (∃ x, f x < abs (m - 1)) ↔ (m > 5 ∨ m < -3) :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2164_216491


namespace NUMINAMATH_GPT_calculate_amount_after_two_years_l2164_216495

noncomputable def amount_after_years (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + rate) ^ years

theorem calculate_amount_after_two_years :
  amount_after_years 51200 0.125 2 = 64800 :=
by
  sorry

end NUMINAMATH_GPT_calculate_amount_after_two_years_l2164_216495


namespace NUMINAMATH_GPT_calculate_p_p_l2164_216427

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 2*y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else 3*x + y

theorem calculate_p_p : p (p 2 (-3)) (p (-4) 1) = 290 :=
by {
  -- required statement of proof problem
  sorry
}

end NUMINAMATH_GPT_calculate_p_p_l2164_216427


namespace NUMINAMATH_GPT_general_term_formula_l2164_216459

-- Conditions: sequence \(\frac{1}{2}\), \(\frac{1}{3}\), \(\frac{1}{4}\), \(\frac{1}{5}, \ldots\)
-- Let seq be the sequence in question.

def seq (n : ℕ) : ℚ := 1 / (n + 1)

-- Question: prove the general term formula is \(\frac{1}{n+1}\)
theorem general_term_formula (n : ℕ) : seq n = 1 / (n + 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_general_term_formula_l2164_216459


namespace NUMINAMATH_GPT_ellipse_focal_distance_m_value_l2164_216442

-- Define the given conditions 
def focal_distance := 2
def ellipse_equation (x y : ℝ) (m : ℝ) := (x^2 / m) + (y^2 / 4) = 1

-- The proof statement
theorem ellipse_focal_distance_m_value :
  ∀ (m : ℝ), 
    (∃ c : ℝ, (2 * c = focal_distance) ∧ (m = 4 + c^2)) →
      m = 5 := by
  sorry

end NUMINAMATH_GPT_ellipse_focal_distance_m_value_l2164_216442


namespace NUMINAMATH_GPT_meat_sales_beyond_plan_l2164_216429

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_meat_sales_beyond_plan_l2164_216429


namespace NUMINAMATH_GPT_systematic_sampling_sequence_l2164_216433

theorem systematic_sampling_sequence :
  ∃ k : ℕ, ∃ b : ℕ, (∀ n : ℕ, n < 6 → (3 + n * k = b + n * 10)) ∧ (b = 3 ∨ b = 13 ∨ b = 23 ∨ b = 33 ∨ b = 43 ∨ b = 53) :=
sorry

end NUMINAMATH_GPT_systematic_sampling_sequence_l2164_216433


namespace NUMINAMATH_GPT_sin_alpha_beta_l2164_216499

theorem sin_alpha_beta (a b c α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
    (h3 : a * Real.cos α + b * Real.sin α + c = 0) (h4 : a * Real.cos β + b * Real.sin β + c = 0) 
    (h5 : α ≠ β) : Real.sin (α + β) = (2 * a * b) / (a ^ 2 + b ^ 2) := 
sorry

end NUMINAMATH_GPT_sin_alpha_beta_l2164_216499


namespace NUMINAMATH_GPT_area_triangle_l2164_216462

noncomputable def area_of_triangle_ABC (AB BC : ℝ) : ℝ := 
    (1 / 2) * AB * BC 

theorem area_triangle (AC : ℝ) (h1 : AC = 40)
    (h2 : ∃ B C : ℝ, B = (1/2) * AC ∧ C = B * Real.sqrt 3) :
    area_of_triangle_ABC ((1 / 2) * AC) (((1 / 2) * AC) * Real.sqrt 3) = 200 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_area_triangle_l2164_216462


namespace NUMINAMATH_GPT_increase_in_avg_commission_l2164_216424

def new_avg_commission := 250
def num_sales := 6
def big_sale_commission := 1000

theorem increase_in_avg_commission :
  (new_avg_commission - (500 / (num_sales - 1))) = 150 := by
  sorry

end NUMINAMATH_GPT_increase_in_avg_commission_l2164_216424


namespace NUMINAMATH_GPT_large_hexagon_toothpicks_l2164_216485

theorem large_hexagon_toothpicks (n : Nat) (h : n = 1001) : 
  let T_half := (n * (n + 1)) / 2
  let T_total := 2 * T_half + n
  let boundary_toothpicks := 6 * T_half
  let total_toothpicks := 3 * T_total - boundary_toothpicks
  total_toothpicks = 3006003 :=
by
  sorry

end NUMINAMATH_GPT_large_hexagon_toothpicks_l2164_216485


namespace NUMINAMATH_GPT_number_of_restaurants_l2164_216449

def first_restaurant_meals_per_day := 20
def second_restaurant_meals_per_day := 40
def third_restaurant_meals_per_day := 50
def total_meals_per_week := 770

theorem number_of_restaurants :
  (first_restaurant_meals_per_day * 7) + 
  (second_restaurant_meals_per_day * 7) + 
  (third_restaurant_meals_per_day * 7) = total_meals_per_week → 
  3 = 3 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_number_of_restaurants_l2164_216449


namespace NUMINAMATH_GPT_simplify_and_rationalize_l2164_216410

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l2164_216410


namespace NUMINAMATH_GPT_find_x_l2164_216419

theorem find_x (x : ℝ) (h : 2 * x = 26 - x + 19) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2164_216419


namespace NUMINAMATH_GPT_pizzeria_large_pizzas_l2164_216404

theorem pizzeria_large_pizzas (price_small : ℕ) (price_large : ℕ) (total_revenue : ℕ) (small_pizzas_sold : ℕ) (L : ℕ) 
    (h1 : price_small = 2) 
    (h2 : price_large = 8) 
    (h3 : total_revenue = 40) 
    (h4 : small_pizzas_sold = 8) 
    (h5 : price_small * small_pizzas_sold + price_large * L = total_revenue) :
    L = 3 := 
by 
  -- Lean will expect a proof here; add sorry for now
  sorry

end NUMINAMATH_GPT_pizzeria_large_pizzas_l2164_216404


namespace NUMINAMATH_GPT_library_fiction_percentage_l2164_216428

theorem library_fiction_percentage:
  let original_volumes := 18360
  let fiction_percentage := 0.30
  let fraction_transferred := 1/3
  let fraction_fiction_transferred := 1/5
  let initial_fiction := fiction_percentage * original_volumes
  let transferred_volumes := fraction_transferred * original_volumes
  let transferred_fiction := fraction_fiction_transferred * transferred_volumes
  let remaining_fiction := initial_fiction - transferred_fiction
  let remaining_volumes := original_volumes - transferred_volumes
  let remaining_fiction_percentage := (remaining_fiction / remaining_volumes) * 100
  remaining_fiction_percentage = 35 := 
by
  sorry

end NUMINAMATH_GPT_library_fiction_percentage_l2164_216428


namespace NUMINAMATH_GPT_triangle_area_proof_l2164_216493

noncomputable def area_of_triangle_ABC : ℝ :=
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  let area := 3 / 11
  area

theorem triangle_area_proof :
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  area_of_triangle_ABC = 3 / 11 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l2164_216493


namespace NUMINAMATH_GPT_sections_in_orchard_l2164_216487

-- Conditions: Farmers harvest 45 sacks from each section daily, 360 sacks are harvested daily
def harvest_sacks_per_section : ℕ := 45
def total_sacks_harvested_daily : ℕ := 360

-- Statement: Prove that the number of sections is 8 given the conditions
theorem sections_in_orchard (h1 : harvest_sacks_per_section = 45) (h2 : total_sacks_harvested_daily = 360) :
  total_sacks_harvested_daily / harvest_sacks_per_section = 8 :=
sorry

end NUMINAMATH_GPT_sections_in_orchard_l2164_216487


namespace NUMINAMATH_GPT_vendor_apples_sold_l2164_216435

theorem vendor_apples_sold (x : ℝ) (h : 0.15 * (1 - x / 100) + 0.50 * (1 - x / 100) * 0.85 = 0.23) : x = 60 :=
sorry

end NUMINAMATH_GPT_vendor_apples_sold_l2164_216435


namespace NUMINAMATH_GPT_sum_of_differences_l2164_216463

theorem sum_of_differences (x : ℝ) (h : (45 + x) / 2 = 38) : abs (x - 45) + abs (x - 30) = 15 := by
  sorry

end NUMINAMATH_GPT_sum_of_differences_l2164_216463


namespace NUMINAMATH_GPT_fountains_for_m_4_fountains_for_m_3_l2164_216471

noncomputable def ceil_div (a b : ℕ) : ℕ :=
  (a + b - 1) / b

-- Problem for m = 4
theorem fountains_for_m_4 (n : ℕ) : ∃ f : ℕ, f = 2 * ceil_div n 3 := 
sorry

-- Problem for m = 3
theorem fountains_for_m_3 (n : ℕ) : ∃ f : ℕ, f = 3 * ceil_div n 3 :=
sorry

end NUMINAMATH_GPT_fountains_for_m_4_fountains_for_m_3_l2164_216471


namespace NUMINAMATH_GPT_avg_age_l2164_216405

-- Given conditions
variables (A B C : ℕ)
variable (h1 : (A + C) / 2 = 29)
variable (h2 : B = 20)

-- to prove
theorem avg_age (A B C : ℕ) (h1 : (A + C) / 2 = 29) (h2 : B = 20) : (A + B + C) / 3 = 26 :=
sorry

end NUMINAMATH_GPT_avg_age_l2164_216405


namespace NUMINAMATH_GPT_intersection_point_on_y_eq_neg_x_l2164_216489

theorem intersection_point_on_y_eq_neg_x 
  (α β : ℝ)
  (h1 : ∃ x y : ℝ, (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧ 
                   (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧ 
                   (y = -x)) :
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 :=
sorry

end NUMINAMATH_GPT_intersection_point_on_y_eq_neg_x_l2164_216489


namespace NUMINAMATH_GPT_identify_true_statements_l2164_216469

-- Definitions of the given statements
def statement1 (a x y : ℝ) : Prop := a * (x + y) = a * x + a * y
def statement2 (a x y : ℝ) : Prop := a ^ (x + y) = a ^ x + a ^ y
def statement3 (x y : ℝ) : Prop := (x + y) ^ 2 = x ^ 2 + y ^ 2
def statement4 (a b : ℝ) : Prop := Real.sqrt (a ^ 2 + b ^ 2) = a + b
def statement5 (a b c : ℝ) : Prop := a * (b / c) = (a * b) / c

-- The statement to prove
theorem identify_true_statements (a x y b c : ℝ) :
  statement1 a x y ∧ statement5 a b c ∧
  ¬ statement2 a x y ∧ ¬ statement3 x y ∧ ¬ statement4 a b :=
sorry

end NUMINAMATH_GPT_identify_true_statements_l2164_216469


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2164_216432

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n = (a 0) * q^n)
  (h2 : ∀ n, a n > a (n + 1))
  (h3 : a 2 + a 3 + a 4 = 28)
  (h4 : a 3 + 2 = (a 2 + a 4) / 2) :
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 63 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_sum_l2164_216432


namespace NUMINAMATH_GPT_max_abs_x_y_l2164_216480

theorem max_abs_x_y (x y : ℝ) (h : 4 * x^2 + y^2 = 4) : |x| + |y| ≤ 2 :=
by sorry

end NUMINAMATH_GPT_max_abs_x_y_l2164_216480


namespace NUMINAMATH_GPT_b_is_arithmetic_sequence_l2164_216445

theorem b_is_arithmetic_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a 1 = 1 →
  a 2 = 2 →
  (∀ n, a (n + 2) = 2 * a (n + 1) - a n + 2) →
  (∀ n, b n = a (n + 1) - a n) →
  ∃ d, ∀ n, b (n + 1) = b n + d :=
by
  intros h1 h2 h3 h4
  use 2
  sorry

end NUMINAMATH_GPT_b_is_arithmetic_sequence_l2164_216445


namespace NUMINAMATH_GPT_standard_deviations_below_mean_l2164_216457

theorem standard_deviations_below_mean (μ σ x : ℝ) (hμ : μ = 14.5) (hσ : σ = 1.7) (hx : x = 11.1) :
    (μ - x) / σ = 2 := by
  sorry

end NUMINAMATH_GPT_standard_deviations_below_mean_l2164_216457


namespace NUMINAMATH_GPT_toys_cost_price_gain_l2164_216423

theorem toys_cost_price_gain (selling_price : ℕ) (cost_price_per_toy : ℕ) (num_toys : ℕ)
    (total_cost_price : ℕ) (gain : ℕ) (x : ℕ) :
    selling_price = 21000 →
    cost_price_per_toy = 1000 →
    num_toys = 18 →
    total_cost_price = num_toys * cost_price_per_toy →
    gain = selling_price - total_cost_price →
    x = gain / cost_price_per_toy →
    x = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at *
  sorry

end NUMINAMATH_GPT_toys_cost_price_gain_l2164_216423


namespace NUMINAMATH_GPT_inequality_always_true_l2164_216482

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end NUMINAMATH_GPT_inequality_always_true_l2164_216482


namespace NUMINAMATH_GPT_min_M_value_l2164_216464

noncomputable def max_pq (p q : ℝ) : ℝ := if p ≥ q then p else q

noncomputable def M (x y : ℝ) : ℝ := max_pq (|x^2 + y + 1|) (|y^2 - x + 1|)

theorem min_M_value : (∀ x y : ℝ, M x y ≥ (3 : ℚ) / 4) ∧ (∃ x y : ℝ, M x y = (3 : ℚ) / 4) :=
sorry

end NUMINAMATH_GPT_min_M_value_l2164_216464


namespace NUMINAMATH_GPT_smallest_B_to_divisible_3_l2164_216407

-- Define the problem
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the digits in the integer
def digit_sum (B : ℕ) : ℕ := 8 + B + 4 + 6 + 3 + 5

-- Prove that the smallest digit B that makes 8B4,635 divisible by 3 is 1
theorem smallest_B_to_divisible_3 : ∃ B : ℕ, B ≥ 0 ∧ B ≤ 9 ∧ is_divisible_by_3 (digit_sum B) ∧ ∀ B' : ℕ, B' < B → ¬ is_divisible_by_3 (digit_sum B') ∧ B = 1 :=
sorry

end NUMINAMATH_GPT_smallest_B_to_divisible_3_l2164_216407


namespace NUMINAMATH_GPT_contrapositive_equiv_l2164_216411

variable (a b : ℝ)

def original_proposition : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

def contrapositive_proposition : Prop := a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0

theorem contrapositive_equiv : original_proposition a b ↔ contrapositive_proposition a b :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_equiv_l2164_216411


namespace NUMINAMATH_GPT_total_distance_traveled_l2164_216444

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l2164_216444


namespace NUMINAMATH_GPT_rectangle_area_stage4_l2164_216414

-- Define the condition: area of one square
def square_area : ℕ := 25

-- Define the condition: number of squares at Stage 4
def num_squares_stage4 : ℕ := 4

-- Define the total area of rectangle at Stage 4
def total_area_stage4 : ℕ := num_squares_stage4 * square_area

-- Prove that total_area_stage4 equals 100 square inches
theorem rectangle_area_stage4 : total_area_stage4 = 100 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_stage4_l2164_216414


namespace NUMINAMATH_GPT_point_on_coordinate_axes_l2164_216483

theorem point_on_coordinate_axes (x y : ℝ) (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by sorry

end NUMINAMATH_GPT_point_on_coordinate_axes_l2164_216483


namespace NUMINAMATH_GPT_min_path_length_l2164_216454

noncomputable def problem_statement : Prop :=
  let XY := 12
  let XZ := 8
  let angle_XYZ := 30
  let YP_PQ_QZ := by {
    -- Reflect Z across XY to get Z' and Y across XZ to get Y'.
    -- Use the Law of cosines in triangle XY'Z'.
    let cos_150 := -Real.sqrt 3 / 2
    let Y_prime_Z_prime := Real.sqrt (8^2 + 12^2 + 2 * 8 * 12 * cos_150)
    exact Y_prime_Z_prime
  }
  ∃ (P Q : Type), (YP_PQ_QZ = Real.sqrt (208 + 96 * Real.sqrt 3))

-- Goal is to prove the problem statement
theorem min_path_length : problem_statement := sorry

end NUMINAMATH_GPT_min_path_length_l2164_216454


namespace NUMINAMATH_GPT_Serezha_puts_more_berries_l2164_216420

theorem Serezha_puts_more_berries (berries : ℕ) 
    (Serezha_puts : ℕ) (Serezha_eats : ℕ)
    (Dima_puts : ℕ) (Dima_eats : ℕ)
    (Serezha_rate : ℕ) (Dima_rate : ℕ)
    (total_berries : berries = 450)
    (Serezha_pattern : Serezha_puts = 1 ∧ Serezha_eats = 1)
    (Dima_pattern : Dima_puts = 2 ∧ Dima_eats = 1)
    (Serezha_faster : Serezha_rate = 2 * Dima_rate) : 
    ∃ (Serezha_in_basket : ℕ) (Dima_in_basket : ℕ),
      Serezha_in_basket > Dima_in_basket ∧ Serezha_in_basket - Dima_in_basket = 50 :=
by
  sorry -- Skip the proof

end NUMINAMATH_GPT_Serezha_puts_more_berries_l2164_216420


namespace NUMINAMATH_GPT_combination_square_octagon_tiles_l2164_216468

-- Define the internal angles of the polygons
def internal_angle (shape : String) : Float :=
  match shape with
  | "Square"   => 90.0
  | "Pentagon" => 108.0
  | "Hexagon"  => 120.0
  | "Octagon"  => 135.0
  | _          => 0.0

-- Define the condition for the combination of two regular polygons to tile seamlessly
def can_tile (shape1 shape2 : String) : Bool :=
  let angle1 := internal_angle shape1
  let angle2 := internal_angle shape2
  angle1 + 2 * angle2 == 360.0

-- Define the tiling problem
theorem combination_square_octagon_tiles : can_tile "Square" "Octagon" = true :=
by {
  -- The proof of this theorem should show that Square and Octagon can indeed tile seamlessly
  sorry
}

end NUMINAMATH_GPT_combination_square_octagon_tiles_l2164_216468


namespace NUMINAMATH_GPT_function_is_linear_l2164_216443

noncomputable def f : ℕ → ℕ :=
  λ n => n + 1

axiom f_at_0 : f 0 = 1
axiom f_at_2016 : f 2016 = 2017
axiom f_equation : ∀ n : ℕ, f (f n) + f n = 2 * n + 3

theorem function_is_linear : ∀ n : ℕ, f n = n + 1 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_function_is_linear_l2164_216443


namespace NUMINAMATH_GPT_sample_size_is_150_l2164_216401

-- Define the conditions
def total_parents : ℕ := 823
def sampled_parents : ℕ := 150
def negative_attitude_parents : ℕ := 136

-- State the theorem
theorem sample_size_is_150 : sampled_parents = 150 := 
by
  sorry

end NUMINAMATH_GPT_sample_size_is_150_l2164_216401


namespace NUMINAMATH_GPT_greatest_possible_value_of_y_l2164_216430

theorem greatest_possible_value_of_y 
  (x y : ℤ) 
  (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ (exists x, x * y + 7 * x + 6 * y = -8) := 
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_y_l2164_216430


namespace NUMINAMATH_GPT_first_group_people_count_l2164_216470

theorem first_group_people_count (P : ℕ) (W : ℕ) 
  (h1 : P * 3 * W = 3 * W) 
  (h2 : 8 * 3 * W = 8 * W) : 
  P = 3 :=
by
  sorry

end NUMINAMATH_GPT_first_group_people_count_l2164_216470


namespace NUMINAMATH_GPT_increase_fraction_l2164_216479

theorem increase_fraction (A F : ℝ) 
  (h₁ : A = 83200) 
  (h₂ : A * (1 + F) ^ 2 = 105300) : 
  F = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_increase_fraction_l2164_216479


namespace NUMINAMATH_GPT_cornbread_pieces_l2164_216422

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h₁ : pan_length = 24) (h₂ : pan_width = 20) 
  (h₃ : piece_length = 3) (h₄ : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end NUMINAMATH_GPT_cornbread_pieces_l2164_216422


namespace NUMINAMATH_GPT_peter_fraction_is_1_8_l2164_216400

-- Define the total number of slices, slices Peter ate alone, and slices Peter shared with Paul
def total_slices := 16
def peter_alone_slices := 1
def shared_slices := 2

-- Define the fraction of the pizza Peter ate alone
def peter_fraction_alone := peter_alone_slices / total_slices

-- Define the fraction of the pizza Peter ate from the shared slices
def shared_fraction := shared_slices * (1 / 2) / total_slices

-- Define the total fraction of the pizza Peter ate
def total_fraction_peter_ate := peter_fraction_alone + shared_fraction

-- Prove that the total fraction of the pizza Peter ate is 1/8
theorem peter_fraction_is_1_8 : total_fraction_peter_ate = 1/8 := by
  sorry

end NUMINAMATH_GPT_peter_fraction_is_1_8_l2164_216400


namespace NUMINAMATH_GPT_min_f_on_interval_l2164_216426

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_f_on_interval : 
  ∀ x, 0 < x ∧ x < π / 2 → f x ≥ 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_f_on_interval_l2164_216426


namespace NUMINAMATH_GPT_number_of_valid_grids_l2164_216453

-- Define the concept of a grid and the necessary properties
structure Grid (n : ℕ) :=
  (cells: Fin (n * n) → ℕ)
  (unique: Function.Injective cells)
  (ordered_rows: ∀ i j : Fin n, i < j → cells ⟨i * n + j, sorry⟩ > cells ⟨i * n + j - 1, sorry⟩)
  (ordered_columns: ∀ i j : Fin n, i < j → cells ⟨j * n + i, sorry⟩ > cells ⟨(j - 1) * n + i, sorry⟩)

-- Define the 4x4 grid
def grid_4x4 := Grid 4

-- Statement of the problem: prove there are 2 valid grid_4x4 configurations
theorem number_of_valid_grids : ∃ g : grid_4x4, (∃ g1 g2 : grid_4x4, (g1 ≠ g2) ∧ (∀ g3 : grid_4x4, g3 = g1 ∨ g3 = g2)) :=
  sorry

end NUMINAMATH_GPT_number_of_valid_grids_l2164_216453


namespace NUMINAMATH_GPT_max_daily_sales_revenue_l2164_216418

noncomputable def f (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t < 15 
  then (1 / 3) * t + 8
  else if 15 ≤ t ∧ t < 30 
  then -(1 / 3) * t + 18
  else 0

noncomputable def g (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 30
  then -t + 30
  else 0

noncomputable def W (t : ℕ) : ℝ :=
  f t * g t

theorem max_daily_sales_revenue : ∃ t : ℕ, W t = 243 :=
by
  existsi 3
  sorry

end NUMINAMATH_GPT_max_daily_sales_revenue_l2164_216418


namespace NUMINAMATH_GPT_eval_f_pi_over_8_l2164_216403

noncomputable def f (θ : ℝ) : ℝ :=
(2 * (Real.sin (θ / 2)) ^ 2 - 1) / (Real.sin (θ / 2) * Real.cos (θ / 2)) + 2 * Real.tan θ

theorem eval_f_pi_over_8 : f (π / 8) = -4 :=
sorry

end NUMINAMATH_GPT_eval_f_pi_over_8_l2164_216403


namespace NUMINAMATH_GPT_domain_of_y_l2164_216448

noncomputable def domain_of_function (x : ℝ) : Bool :=
  x < 0 ∧ x ≠ -1

theorem domain_of_y :
  {x : ℝ | (∃ y, y = (x + 1) ^ 0 / Real.sqrt (|x| - x)) } =
  {x : ℝ | domain_of_function x} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_y_l2164_216448


namespace NUMINAMATH_GPT_first_term_geometric_sequence_l2164_216451

theorem first_term_geometric_sequence (a r : ℚ) 
  (h3 : a * r^(3-1) = 24)
  (h4 : a * r^(4-1) = 36) :
  a = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_l2164_216451


namespace NUMINAMATH_GPT_percentage_decrease_l2164_216486

theorem percentage_decrease 
  (P0 : ℕ) (P2 : ℕ) (H0 : P0 = 10000) (H2 : P2 = 9600) 
  (P1 : ℕ) (H1 : P1 = P0 + (20 * P0) / 100) :
  ∃ (D : ℕ), P2 = P1 - (D * P1) / 100 ∧ D = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l2164_216486


namespace NUMINAMATH_GPT_exists_nat_numbers_except_two_three_l2164_216439

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end NUMINAMATH_GPT_exists_nat_numbers_except_two_three_l2164_216439


namespace NUMINAMATH_GPT_set_1234_excellent_no_proper_subset_excellent_l2164_216413

open Set

namespace StepLength

def excellent_set (D : Set ℤ) : Prop :=
∀ A : Set ℤ, ∃ a d : ℤ, d ∈ D → ({a - d, a, a + d} ⊆ A ∨ {a - d, a, a + d} ⊆ (univ \ A))

noncomputable def S : Set (Set ℤ) := {{1}, {2}, {3}, {4}}

theorem set_1234_excellent : excellent_set {1, 2, 3, 4} := sorry

theorem no_proper_subset_excellent :
  ¬ (excellent_set {1, 3, 4} ∨ excellent_set {1, 2, 3} ∨ excellent_set {1, 2, 4} ∨ excellent_set {2, 3, 4}) := sorry

end StepLength

end NUMINAMATH_GPT_set_1234_excellent_no_proper_subset_excellent_l2164_216413


namespace NUMINAMATH_GPT_mary_puts_back_correct_number_of_oranges_l2164_216488

namespace FruitProblem

def price_apple := 40
def price_orange := 60
def total_fruits := 10
def average_price_all := 56
def average_price_kept := 50

theorem mary_puts_back_correct_number_of_oranges :
  ∀ (A O O' T: ℕ),
  A + O = total_fruits →
  A * price_apple + O * price_orange = total_fruits * average_price_all →
  A = 2 →
  T = A + O' →
  A * price_apple + O' * price_orange = T * average_price_kept →
  O - O' = 6 :=
by
  sorry

end FruitProblem

end NUMINAMATH_GPT_mary_puts_back_correct_number_of_oranges_l2164_216488


namespace NUMINAMATH_GPT_halfway_between_one_fourth_and_one_seventh_l2164_216409

theorem halfway_between_one_fourth_and_one_seventh : (1 / 4 + 1 / 7) / 2 = 11 / 56 := by
  sorry

end NUMINAMATH_GPT_halfway_between_one_fourth_and_one_seventh_l2164_216409


namespace NUMINAMATH_GPT_value_of_expression_l2164_216481

theorem value_of_expression (m n : ℝ) (h : m + n = -2) : 5 * m^2 + 5 * n^2 + 10 * m * n = 20 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2164_216481


namespace NUMINAMATH_GPT_coach_A_spent_less_l2164_216421

-- Definitions of costs and discounts for coaches purchases
def total_cost_before_discount_A : ℝ := 10 * 29 + 5 * 15
def total_cost_before_discount_B : ℝ := 14 * 2.50 + 1 * 18 + 4 * 25 + 1 * 72
def total_cost_before_discount_C : ℝ := 8 * 32 + 12 * 12

def discount_A : ℝ := 0.05 * total_cost_before_discount_A
def discount_B : ℝ := 0.10 * total_cost_before_discount_B
def discount_C : ℝ := 0.07 * total_cost_before_discount_C

def total_cost_after_discount_A : ℝ := total_cost_before_discount_A - discount_A
def total_cost_after_discount_B : ℝ := total_cost_before_discount_B - discount_B
def total_cost_after_discount_C : ℝ := total_cost_before_discount_C - discount_C

def combined_cost_B_C : ℝ := total_cost_after_discount_B + total_cost_after_discount_C
def difference_A_BC : ℝ := total_cost_after_discount_A - combined_cost_B_C

theorem coach_A_spent_less : difference_A_BC = -227.75 := by
  sorry

end NUMINAMATH_GPT_coach_A_spent_less_l2164_216421


namespace NUMINAMATH_GPT_tan_105_eq_minus_2_minus_sqrt_3_l2164_216474

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_105_eq_minus_2_minus_sqrt_3_l2164_216474


namespace NUMINAMATH_GPT_misha_total_students_l2164_216458

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end NUMINAMATH_GPT_misha_total_students_l2164_216458


namespace NUMINAMATH_GPT_range_of_z_l2164_216472

theorem range_of_z (x y : ℝ) (hx1 : x - 2 * y + 1 ≥ 0) (hx2 : y ≥ x) (hx3 : x ≥ 0) :
  ∃ z, z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_z_l2164_216472


namespace NUMINAMATH_GPT_total_monthly_feed_l2164_216415

def daily_feed (pounds_per_pig_per_day : ℕ) (number_of_pigs : ℕ) : ℕ :=
  pounds_per_pig_per_day * number_of_pigs

def monthly_feed (daily_feed : ℕ) (days_per_month : ℕ) : ℕ :=
  daily_feed * days_per_month

theorem total_monthly_feed :
  let pounds_per_pig_per_day := 15
  let number_of_pigs := 4
  let days_per_month := 30
  monthly_feed (daily_feed pounds_per_pig_per_day number_of_pigs) days_per_month = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_monthly_feed_l2164_216415


namespace NUMINAMATH_GPT_reciprocal_of_neg_seven_l2164_216494

theorem reciprocal_of_neg_seven : (1 : ℚ) / (-7 : ℚ) = -1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_seven_l2164_216494


namespace NUMINAMATH_GPT_rectangle_area_divisible_by_12_l2164_216496

theorem rectangle_area_divisible_by_12
  (x y z : ℤ)
  (h : x^2 + y^2 = z^2) :
  12 ∣ (x * y) :=
sorry

end NUMINAMATH_GPT_rectangle_area_divisible_by_12_l2164_216496


namespace NUMINAMATH_GPT_cut_wood_into_5_pieces_l2164_216461

-- Definitions
def pieces_to_cuts (pieces : ℕ) : ℕ := pieces - 1
def time_per_cut (total_time : ℕ) (cuts : ℕ) : ℕ := total_time / cuts
def total_time_for_pieces (pieces : ℕ) (time_per_cut : ℕ) : ℕ := (pieces_to_cuts pieces) * time_per_cut

-- Given conditions
def conditions : Prop :=
  pieces_to_cuts 4 = 3 ∧
  time_per_cut 24 (pieces_to_cuts 4) = 8

-- Problem statement
theorem cut_wood_into_5_pieces (h : conditions) : total_time_for_pieces 5 8 = 32 :=
by sorry

end NUMINAMATH_GPT_cut_wood_into_5_pieces_l2164_216461


namespace NUMINAMATH_GPT_abel_overtake_kelly_chris_overtake_both_l2164_216437

-- Given conditions and variables
variable (d : ℝ)  -- distance at which Abel overtakes Kelly
variable (d_c : ℝ)  -- distance at which Chris overtakes both Kelly and Abel
variable (t_k : ℝ)  -- time taken by Kelly to run d meters
variable (t_a : ℝ)  -- time taken by Abel to run (d + 3) meters
variable (t_c : ℝ)  -- time taken by Chris to run the required distance
variable (k_speed : ℝ := 9)  -- Kelly's speed
variable (a_speed : ℝ := 9.5)  -- Abel's speed
variable (c_speed : ℝ := 10)  -- Chris's speed
variable (head_start_k : ℝ := 3)  -- Kelly's head start over Abel
variable (head_start_c : ℝ := 2)  -- Chris's head start behind Abel
variable (lost_by : ℝ := 0.75)  -- Abel lost by distance

-- Proof problem for Abel overtaking Kelly
theorem abel_overtake_kelly 
  (hk : t_k = d / k_speed) 
  (ha : t_a = (d + head_start_k) / a_speed) 
  (h_lost : lost_by = 0.75):
  d + lost_by = 54.75 := 
sorry

-- Proof problem for Chris overtaking both Kelly and Abel
theorem chris_overtake_both 
  (hc : t_c = (d_c + 5) / c_speed)
  (h_56 : d_c = 56):
  d_c = c_speed * (56 / c_speed) :=
sorry

end NUMINAMATH_GPT_abel_overtake_kelly_chris_overtake_both_l2164_216437


namespace NUMINAMATH_GPT_circles_intersect_in_two_points_l2164_216476

def circle1 (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = (3/2)^2
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

theorem circles_intersect_in_two_points :
  ∃! (p : ℝ × ℝ), (circle1 p.1 p.2) ∧ (circle2 p.1 p.2) := 
sorry

end NUMINAMATH_GPT_circles_intersect_in_two_points_l2164_216476


namespace NUMINAMATH_GPT_debate_schedule_ways_l2164_216456

-- Definitions based on the problem conditions
def east_debaters : Fin 4 := 4
def west_debaters : Fin 4 := 4
def total_debates := east_debaters.val * west_debaters.val
def debates_per_session := 3
def sessions := 5
def rest_debates := total_debates - sessions * debates_per_session

-- Claim that the number of scheduling ways is the given number
theorem debate_schedule_ways : (Nat.factorial total_debates) / ((Nat.factorial debates_per_session) ^ sessions * Nat.factorial rest_debates) = 20922789888000 :=
by
  -- Proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_debate_schedule_ways_l2164_216456


namespace NUMINAMATH_GPT_no_integer_solutions_l2164_216465

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 := 
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_solutions_l2164_216465


namespace NUMINAMATH_GPT_larger_of_two_numbers_with_hcf_25_l2164_216467

theorem larger_of_two_numbers_with_hcf_25 (a b : ℕ) (h_hcf: Nat.gcd a b = 25)
  (h_lcm_factors: 13 * 14 = (25 * 13 * 14) / (Nat.gcd a b)) :
  max a b = 350 :=
sorry

end NUMINAMATH_GPT_larger_of_two_numbers_with_hcf_25_l2164_216467


namespace NUMINAMATH_GPT_solution_set_inequality_l2164_216416

theorem solution_set_inequality (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l2164_216416


namespace NUMINAMATH_GPT_rona_age_l2164_216408

theorem rona_age (R : ℕ) (hR1 : ∀ Rachel Collete : ℕ, Rachel = 2 * R ∧ Collete = R / 2 ∧ Rachel - Collete = 12) : R = 12 :=
sorry

end NUMINAMATH_GPT_rona_age_l2164_216408


namespace NUMINAMATH_GPT_value_of_a5_l2164_216475

theorem value_of_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = 2 * n * (n + 1)) (ha : ∀ n, a n = S n - S (n - 1)) :
  a 5 = 20 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a5_l2164_216475


namespace NUMINAMATH_GPT_value_of_a_in_terms_of_b_l2164_216452

noncomputable def value_of_a (b : ℝ) : ℝ :=
  b * (38.1966 / 61.8034)

theorem value_of_a_in_terms_of_b (b a : ℝ) :
  (∀ x : ℝ, (b / x = 61.80339887498949 / 100) ∧ (x = (a + b) * (61.80339887498949 / 100)))
  → a = value_of_a b :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_in_terms_of_b_l2164_216452


namespace NUMINAMATH_GPT_initial_buckets_correct_l2164_216417

-- Define the conditions as variables
def total_buckets : ℝ := 9.8
def added_buckets : ℝ := 8.8
def initial_buckets : ℝ := total_buckets - added_buckets

-- The theorem to prove the initial amount of water is 1 bucket
theorem initial_buckets_correct : initial_buckets = 1 := 
by
  sorry

end NUMINAMATH_GPT_initial_buckets_correct_l2164_216417


namespace NUMINAMATH_GPT_range_of_m_l2164_216406

theorem range_of_m (m : ℝ) : (∀ x > 1, 2*x + m + 8/(x-1) > 0) → m > -10 := 
by
  -- The formal proof will be completed here.
  sorry

end NUMINAMATH_GPT_range_of_m_l2164_216406


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l2164_216431

theorem geometric_sequence_fifth_term : 
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  a₅ = 1 / 2048 :=
by
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l2164_216431


namespace NUMINAMATH_GPT_instantaneous_speed_at_3_l2164_216447

noncomputable def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

theorem instantaneous_speed_at_3 : deriv s 3 = 11 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_speed_at_3_l2164_216447


namespace NUMINAMATH_GPT_sum_of_roots_of_cubic_eq_l2164_216436

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 72 * x + 6

-- Define the statement to prove
theorem sum_of_roots_of_cubic_eq : 
  ∀ (r p q : ℝ), (cubic_eq r = 0) ∧ (cubic_eq p = 0) ∧ (cubic_eq q = 0) → 
  (r + p + q) = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_cubic_eq_l2164_216436


namespace NUMINAMATH_GPT_fraction_product_l2164_216434

theorem fraction_product :
  ((5/4) * (8/16) * (20/12) * (32/64) * (50/20) * (40/80) * (70/28) * (48/96) : ℚ) = 625/768 := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_l2164_216434
