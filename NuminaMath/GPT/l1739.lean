import Mathlib

namespace NUMINAMATH_GPT_negation_universal_proposition_l1739_173914

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x := 
by 
  sorry

end NUMINAMATH_GPT_negation_universal_proposition_l1739_173914


namespace NUMINAMATH_GPT_spend_on_laundry_detergent_l1739_173988

def budget : ℕ := 60
def price_shower_gel : ℕ := 4
def num_shower_gels : ℕ := 4
def price_toothpaste : ℕ := 3
def remaining_budget : ℕ := 30

theorem spend_on_laundry_detergent : 
  (budget - remaining_budget) = (num_shower_gels * price_shower_gel + price_toothpaste) + 11 := 
by
  sorry

end NUMINAMATH_GPT_spend_on_laundry_detergent_l1739_173988


namespace NUMINAMATH_GPT_minimum_value_range_l1739_173926

noncomputable def f (a x : ℝ) : ℝ := abs (3 * x - 1) + a * x + 2

theorem minimum_value_range (a : ℝ) :
  (-3 ≤ a ∧ a ≤ 3) ↔ ∃ m, ∀ x, f a x ≥ m := sorry

end NUMINAMATH_GPT_minimum_value_range_l1739_173926


namespace NUMINAMATH_GPT_eggs_distributed_equally_l1739_173953

-- Define the total number of eggs
def total_eggs : ℕ := 8484

-- Define the number of baskets
def baskets : ℕ := 303

-- Define the expected number of eggs per basket
def eggs_per_basket : ℕ := 28

-- State the theorem
theorem eggs_distributed_equally :
  total_eggs / baskets = eggs_per_basket := sorry

end NUMINAMATH_GPT_eggs_distributed_equally_l1739_173953


namespace NUMINAMATH_GPT_Jessica_paid_1000_for_rent_each_month_last_year_l1739_173980

/--
Jessica paid $200 for food each month last year.
Jessica paid $100 for car insurance each month last year.
This year her rent goes up by 30%.
This year food costs increase by 50%.
This year the cost of her car insurance triples.
Jessica pays $7200 more for her expenses over the whole year compared to last year.
-/
theorem Jessica_paid_1000_for_rent_each_month_last_year
  (R : ℝ) -- monthly rent last year
  (h1 : 12 * (0.30 * R + 100 + 200) = 7200) :
  R = 1000 :=
sorry

end NUMINAMATH_GPT_Jessica_paid_1000_for_rent_each_month_last_year_l1739_173980


namespace NUMINAMATH_GPT_negation_of_exists_l1739_173961

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + x + 1 < 0) : ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l1739_173961


namespace NUMINAMATH_GPT_minimum_value_of_expr_l1739_173978

noncomputable def expr (x y : ℝ) : ℝ := 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4

theorem minimum_value_of_expr : ∃ x y : ℝ, expr x y = -1 ∧ ∀ (a b : ℝ), expr a b ≥ -1 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expr_l1739_173978


namespace NUMINAMATH_GPT_mixed_oil_rate_is_correct_l1739_173998

def rate_of_mixed_oil (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℕ :=
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2)

theorem mixed_oil_rate_is_correct :
  rate_of_mixed_oil 10 50 5 68 = 56 := by
  sorry

end NUMINAMATH_GPT_mixed_oil_rate_is_correct_l1739_173998


namespace NUMINAMATH_GPT_find_geometric_ratio_l1739_173987

-- Definitions for the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def geometric_sequence (a1 a3 a4 : ℝ) (q : ℝ) : Prop :=
  a3 * a3 = a1 * a4 ∧ a3 = a1 * q ∧ a4 = a3 * q

-- Definition for the proof statement
theorem find_geometric_ratio (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hnz : ∀ n, a n ≠ 0)
  (hq : ∃ (q : ℝ), geometric_sequence (a 0) (a 2) (a 3) q) :
  ∃ q, q = 1 ∨ q = 1 / 2 := sorry

end NUMINAMATH_GPT_find_geometric_ratio_l1739_173987


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1739_173915

theorem problem1 : 0.175 / 0.25 / 4 = 0.175 := by
  sorry

theorem problem2 : 1.4 * 99 + 1.4 = 140 := by 
  sorry

theorem problem3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by
  sorry

theorem problem4 : (3.2 + 0.16) / 0.8 = 4.2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1739_173915


namespace NUMINAMATH_GPT_parabola_equation_standard_form_l1739_173973

theorem parabola_equation_standard_form (p : ℝ) (x y : ℝ)
    (h₁ : y^2 = 2 * p * x)
    (h₂ : y = -4)
    (h₃ : x = -2) : y^2 = -8 * x := by
  sorry

end NUMINAMATH_GPT_parabola_equation_standard_form_l1739_173973


namespace NUMINAMATH_GPT_number_of_groups_of_three_books_l1739_173970

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end NUMINAMATH_GPT_number_of_groups_of_three_books_l1739_173970


namespace NUMINAMATH_GPT_water_to_add_l1739_173994

theorem water_to_add (x : ℚ) (alcohol water : ℚ) (ratio : ℚ) :
  alcohol = 4 → water = 4 →
  (3 : ℚ) / (3 + 5) = (3 : ℚ) / 8 →
  (5 : ℚ) / (3 + 5) = (5 : ℚ) / 8 →
  ratio = 5 / 8 →
  (4 + x) / (8 + x) = ratio →
  x = 8 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_water_to_add_l1739_173994


namespace NUMINAMATH_GPT_no_unique_y_exists_l1739_173999

theorem no_unique_y_exists (x y : ℕ) (k m : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + 7) % y = 12) :
  ¬ ∃! y, (∃ k m : ℤ, x = 82 * k + 5 ∧ (x + 7) = y * m + 12) :=
by
  sorry

end NUMINAMATH_GPT_no_unique_y_exists_l1739_173999


namespace NUMINAMATH_GPT_reggie_books_l1739_173948

/-- 
Reggie's father gave him $48. Reggie bought some books, each of which cost $2, 
and now he has $38 left. How many books did Reggie buy?
-/
theorem reggie_books (initial_amount spent_amount remaining_amount book_cost books_bought : ℤ)
  (h_initial : initial_amount = 48)
  (h_remaining : remaining_amount = 38)
  (h_book_cost : book_cost = 2)
  (h_spent : spent_amount = initial_amount - remaining_amount)
  (h_books_bought : books_bought = spent_amount / book_cost) :
  books_bought = 5 :=
by
  sorry

end NUMINAMATH_GPT_reggie_books_l1739_173948


namespace NUMINAMATH_GPT_integer_roots_sum_abs_eq_94_l1739_173938

theorem integer_roots_sum_abs_eq_94 {a b c m : ℤ} :
  (∃ m, (x : ℤ) * (x : ℤ) * (x : ℤ) - 2013 * (x : ℤ) + m = 0 ∧ a + b + c = 0 ∧ ab + bc + ac = -2013) →
  |a| + |b| + |c| = 94 :=
sorry

end NUMINAMATH_GPT_integer_roots_sum_abs_eq_94_l1739_173938


namespace NUMINAMATH_GPT_daily_profit_at_45_selling_price_for_1200_profit_l1739_173906

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end NUMINAMATH_GPT_daily_profit_at_45_selling_price_for_1200_profit_l1739_173906


namespace NUMINAMATH_GPT_intersection_M_N_l1739_173993

def M : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1739_173993


namespace NUMINAMATH_GPT_graph_function_quadrant_l1739_173944

theorem graph_function_quadrant (x y : ℝ): 
  (∀ x : ℝ, y = -x + 2 → (x < 0 → y ≠ -3 + - x)) := 
sorry

end NUMINAMATH_GPT_graph_function_quadrant_l1739_173944


namespace NUMINAMATH_GPT_jayson_age_l1739_173930

/-- When Jayson is a certain age J, his dad is four times his age,
    and his mom is 2 years younger than his dad. Jayson's mom was
    28 years old when he was born. Prove that Jayson is 10 years old
    when his dad is four times his age. -/
theorem jayson_age {J : ℕ} (h1 : ∀ J, J > 0 → J * 4 < J + 4) 
                   (h2 : ∀ J, (4 * J - 2) = J + 28) 
                   (h3 : J - (4 * J - 28) = 0): 
                   J = 10 :=
by 
  sorry

end NUMINAMATH_GPT_jayson_age_l1739_173930


namespace NUMINAMATH_GPT_coins_with_specific_probabilities_impossible_l1739_173974

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end NUMINAMATH_GPT_coins_with_specific_probabilities_impossible_l1739_173974


namespace NUMINAMATH_GPT_average_salary_without_manager_l1739_173985

theorem average_salary_without_manager (A : ℝ) (H : 15 * A + 4200 = 16 * (A + 150)) : A = 1800 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_salary_without_manager_l1739_173985


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_iff_l1739_173901

theorem ellipse_foci_y_axis_iff (m n : ℝ) (h : m > n ∧ n > 0) :
  (m > n ∧ n > 0) ↔ (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → ∃ a b : ℝ, a^2 - b^2 = 1 ∧ x^2/b^2 + y^2/a^2 = 1 ∧ a > b) :=
sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_iff_l1739_173901


namespace NUMINAMATH_GPT_one_fourth_way_from_x1_to_x2_l1739_173975

-- Definitions of the points
def x1 : ℚ := 1 / 5
def x2 : ℚ := 4 / 5

-- Problem statement: Prove that one fourth of the way from x1 to x2 is 7/20
theorem one_fourth_way_from_x1_to_x2 : (3 * x1 + 1 * x2) / 4 = 7 / 20 := by
  sorry

end NUMINAMATH_GPT_one_fourth_way_from_x1_to_x2_l1739_173975


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1739_173965

def condition_p (x : ℝ) : Prop := abs (x - 1) < 2
def condition_q (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

theorem p_sufficient_but_not_necessary_for_q : 
  (∀ x, condition_p x → condition_q x) ∧ 
  ¬ (∀ x, condition_q x → condition_p x) := 
by
  sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1739_173965


namespace NUMINAMATH_GPT_radish_patch_size_l1739_173921

theorem radish_patch_size (R P : ℕ) (h1 : P = 2 * R) (h2 : P / 6 = 5) : R = 15 := by
  sorry

end NUMINAMATH_GPT_radish_patch_size_l1739_173921


namespace NUMINAMATH_GPT_initial_apples_l1739_173986

-- Define the number of initial fruits
def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def fruits_given : ℕ := 40
def fruits_left : ℕ := 15

-- Define the equation for the initial number of fruits
def initial_total_fruits (A : ℕ) : Prop :=
  initial_plums + initial_guavas + A = fruits_left + fruits_given

-- Define the proof problem to find the number of apples
theorem initial_apples : ∃ A : ℕ, initial_total_fruits A ∧ A = 21 :=
  by
    sorry

end NUMINAMATH_GPT_initial_apples_l1739_173986


namespace NUMINAMATH_GPT_airplane_speeds_l1739_173950

theorem airplane_speeds (v : ℝ) 
  (h1 : 2.5 * v + 2.5 * 250 = 1625) : 
  v = 400 := 
sorry

end NUMINAMATH_GPT_airplane_speeds_l1739_173950


namespace NUMINAMATH_GPT_sin_double_angle_l1739_173949

theorem sin_double_angle (θ : ℝ) 
    (h : Real.sin (Real.pi / 4 + θ) = 1 / 3) : 
    Real.sin (2 * θ) = -7 * Real.sqrt 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1739_173949


namespace NUMINAMATH_GPT_problem_l1739_173963

theorem problem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 := 
sorry

end NUMINAMATH_GPT_problem_l1739_173963


namespace NUMINAMATH_GPT_trapezoid_perimeter_l1739_173917

noncomputable def semiCircularTrapezoidPerimeter (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2) : ℝ :=
-((x^2) / 8) + 2 * x + 32

theorem trapezoid_perimeter 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2)
  (r : ℝ) 
  (h_r : r = 8) 
  (AB : ℝ) 
  (h_AB : AB = 2 * r)
  (CD_on_circumference : true) :
  semiCircularTrapezoidPerimeter x hx = -((x^2) / 8) + 2 * x + 32 :=   
sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l1739_173917


namespace NUMINAMATH_GPT_remaining_water_at_end_of_hike_l1739_173991

-- Define conditions
def initial_water : ℝ := 9
def hike_length : ℝ := 7
def hike_duration : ℝ := 2
def leak_rate : ℝ := 1
def drink_rate_6_miles : ℝ := 0.6666666666666666
def drink_last_mile : ℝ := 2

-- Define the question and answer
def remaining_water (initial: ℝ) (duration: ℝ) (leak: ℝ) (drink6: ℝ) (drink_last: ℝ) : ℝ :=
  initial - ((drink6 * 6) + drink_last + (leak * duration))

-- Theorem stating the proof problem 
theorem remaining_water_at_end_of_hike :
  remaining_water initial_water hike_duration leak_rate drink_rate_6_miles drink_last_mile = 1 :=
by
  sorry

end NUMINAMATH_GPT_remaining_water_at_end_of_hike_l1739_173991


namespace NUMINAMATH_GPT_last_three_digits_product_l1739_173941

theorem last_three_digits_product (a b c : ℕ) 
  (h1 : (a + b) % 10 = c % 10) 
  (h2 : (b + c) % 10 = a % 10) 
  (h3 : (c + a) % 10 = b % 10) :
  (a * b * c) % 1000 = 250 ∨ (a * b * c) % 1000 = 500 ∨ (a * b * c) % 1000 = 750 ∨ (a * b * c) % 1000 = 0 := 
by
  sorry

end NUMINAMATH_GPT_last_three_digits_product_l1739_173941


namespace NUMINAMATH_GPT_sum_of_squares_l1739_173967

theorem sum_of_squares (a b c : ℝ)
  (h1 : a + b + c = 19)
  (h2 : a * b + b * c + c * a = 131) :
  a^2 + b^2 + c^2 = 99 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1739_173967


namespace NUMINAMATH_GPT_complement_correct_l1739_173947

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the set of real numbers such that -1 ≤ x < 2
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define the complement of A in U
def complement_U_A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

-- The proof statement: the complement of A in U is the expected set
theorem complement_correct : (U \ A) = complement_U_A := 
by
  sorry

end NUMINAMATH_GPT_complement_correct_l1739_173947


namespace NUMINAMATH_GPT_pants_cost_is_250_l1739_173966

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end NUMINAMATH_GPT_pants_cost_is_250_l1739_173966


namespace NUMINAMATH_GPT_money_left_after_shopping_l1739_173902

def initial_money : ℕ := 158
def shoe_cost : ℕ := 45
def bag_cost := shoe_cost - 17
def lunch_cost := bag_cost / 4
def total_expenses := shoe_cost + bag_cost + lunch_cost
def remaining_money := initial_money - total_expenses

theorem money_left_after_shopping : remaining_money = 78 := by
  sorry

end NUMINAMATH_GPT_money_left_after_shopping_l1739_173902


namespace NUMINAMATH_GPT_correct_answer_l1739_173984

noncomputable def original_number (y : ℝ) :=
  (y - 14) / 2 = 50

theorem correct_answer (y : ℝ) (h : original_number y) :
  (y - 5) / 7 = 15 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l1739_173984


namespace NUMINAMATH_GPT_revenue_percentage_change_l1739_173903

theorem revenue_percentage_change (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  let P_new := 1.30 * P
  let S_new := 0.80 * S
  let R := P * S
  let R_new := P_new * S_new
  (R_new - R) / R * 100 = 4 := by
  sorry

end NUMINAMATH_GPT_revenue_percentage_change_l1739_173903


namespace NUMINAMATH_GPT_financial_loss_example_l1739_173931

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := 
  P * (1 + r * t)

theorem financial_loss_example :
  let P := 10000
  let r1 := 0.06
  let r2 := 0.05
  let t := 3
  let n := 4
  let A1 := compound_interest P r1 n t
  let A2 := simple_interest P r2 t
  abs (A1 - A2 - 456.18) < 0.01 := by
    sorry

end NUMINAMATH_GPT_financial_loss_example_l1739_173931


namespace NUMINAMATH_GPT_find_y_l1739_173979

theorem find_y (x k m y : ℤ) 
  (h1 : x = 82 * k + 5) 
  (h2 : x + y = 41 * m + 12) : 
  y = 7 := 
sorry

end NUMINAMATH_GPT_find_y_l1739_173979


namespace NUMINAMATH_GPT_problem1_problem2_l1739_173928

theorem problem1 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) :
  (2 < x1 + x2 ∧ x1 + x2 < 6) ∧ |x1 - x2| < 2 :=
by
  sorry

theorem problem2 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2 - x + 1) :
  |x1 - x2| < |f x1 - f x2| ∧ |f x1 - f x2| < 5 * |x1 - x2| :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1739_173928


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1739_173992

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) : 
  (1 / x + 1 / y = 1 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1739_173992


namespace NUMINAMATH_GPT_arithmetic_series_first_term_l1739_173936

theorem arithmetic_series_first_term (a d : ℚ) 
  (h1 : 15 * (2 * a + 29 * d) = 450) 
  (h2 : 15 * (2 * a + 89 * d) = 1950) : 
  a = -55 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_series_first_term_l1739_173936


namespace NUMINAMATH_GPT_C_must_be_2_l1739_173924

-- Define the given digits and their sum conditions
variables (A B C D : ℤ)

-- The sum of known digits for the first number
def sum1_known_digits := 7 + 4 + 5 + 2

-- The sum of known digits for the second number
def sum2_known_digits := 3 + 2 + 6 + 5

-- The first number must be divisible by 3
def divisible_by_3 (n : ℤ) : Prop := n % 3 = 0

-- Conditions for the divisibility by 3 of both numbers
def conditions := divisible_by_3 (sum1_known_digits + A + B + D) ∧ 
                  divisible_by_3 (sum2_known_digits + A + B + C)

-- The statement of the theorem
theorem C_must_be_2 (A B D : ℤ) (h : conditions A B 2 D) : C = 2 :=
  sorry

end NUMINAMATH_GPT_C_must_be_2_l1739_173924


namespace NUMINAMATH_GPT_birds_find_more_than_half_millet_on_thursday_l1739_173922

def millet_on_day (n : ℕ) : ℝ :=
  2 - 2 * (0.7 ^ n)

def more_than_half_millet (day : ℕ) : Prop :=
  millet_on_day day > 1

theorem birds_find_more_than_half_millet_on_thursday : more_than_half_millet 4 :=
by
  sorry

end NUMINAMATH_GPT_birds_find_more_than_half_millet_on_thursday_l1739_173922


namespace NUMINAMATH_GPT_sum_a1_a5_l1739_173918

def sequence_sum (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 1

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_sum : sequence_sum S)
  (h_a1 : a 1 = S 1)
  (h_a5 : a 5 = S 5 - S 4) :
  a 1 + a 5 = 11 := by
  sorry

end NUMINAMATH_GPT_sum_a1_a5_l1739_173918


namespace NUMINAMATH_GPT_solve_for_x_l1739_173955

theorem solve_for_x (x : ℝ) (h : (3 * x - 15) / 4 = (x + 9) / 5) : x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1739_173955


namespace NUMINAMATH_GPT_evaluate_expression_at_x_l1739_173945

theorem evaluate_expression_at_x (x : ℝ) (h : x = Real.sqrt 2 - 3) : 
  (3 * x / (x^2 - 9)) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_l1739_173945


namespace NUMINAMATH_GPT_rice_weight_per_container_l1739_173956

-- Given total weight of rice in pounds
def totalWeightPounds : ℚ := 25 / 2

-- Conversion factor from pounds to ounces
def poundsToOunces : ℚ := 16

-- Number of containers
def numberOfContainers : ℕ := 4

-- Total weight in ounces
def totalWeightOunces : ℚ := totalWeightPounds * poundsToOunces

-- Weight per container in ounces
def weightPerContainer : ℚ := totalWeightOunces / numberOfContainers

theorem rice_weight_per_container :
  weightPerContainer = 50 := 
sorry

end NUMINAMATH_GPT_rice_weight_per_container_l1739_173956


namespace NUMINAMATH_GPT_paco_min_cookies_l1739_173923

theorem paco_min_cookies (x : ℕ) (h_initial : 25 - x ≥ 0) : 
  x + (3 + 2) ≥ 5 := by
  sorry

end NUMINAMATH_GPT_paco_min_cookies_l1739_173923


namespace NUMINAMATH_GPT_smallest_period_cos_l1739_173957

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem smallest_period_cos (x : ℝ) : 
  smallest_positive_period (λ x => 2 * (Real.cos x)^2 + 1) Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_smallest_period_cos_l1739_173957


namespace NUMINAMATH_GPT_five_peso_coins_count_l1739_173909

theorem five_peso_coins_count (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 5 * y = 440) (h3 : x = 24 ∨ y = 24) : y = 24 :=
by sorry

end NUMINAMATH_GPT_five_peso_coins_count_l1739_173909


namespace NUMINAMATH_GPT_garrison_reinforcement_l1739_173940

/-- A garrison has initial provisions for 2000 men for 65 days. 
    After 15 days, reinforcement arrives and the remaining provisions last for 20 more days. 
    The size of the reinforcement is 3000 men.  -/
theorem garrison_reinforcement (P : ℕ) (M1 M2 D1 D2 D3 R : ℕ) 
  (h1 : M1 = 2000) (h2 : D1 = 65) (h3 : D2 = 15) (h4 : D3 = 20) 
  (h5 : P = M1 * D1) (h6 : P - M1 * D2 = (M1 + R) * D3) : 
  R = 3000 := 
sorry

end NUMINAMATH_GPT_garrison_reinforcement_l1739_173940


namespace NUMINAMATH_GPT_total_carpet_area_correct_l1739_173935

-- Define dimensions of the rooms
def room1_width : ℝ := 12
def room1_length : ℝ := 15
def room2_width : ℝ := 7
def room2_length : ℝ := 9
def room3_width : ℝ := 10
def room3_length : ℝ := 11

-- Define the areas of the rooms
def room1_area : ℝ := room1_width * room1_length
def room2_area : ℝ := room2_width * room2_length
def room3_area : ℝ := room3_width * room3_length

-- Total carpet area
def total_carpet_area : ℝ := room1_area + room2_area + room3_area

-- The theorem to prove
theorem total_carpet_area_correct :
  total_carpet_area = 353 :=
sorry

end NUMINAMATH_GPT_total_carpet_area_correct_l1739_173935


namespace NUMINAMATH_GPT_product_equals_16896_l1739_173972

theorem product_equals_16896 (A B C D : ℕ) (h1 : A + B + C + D = 70)
  (h2 : A = 3 * C + 1) (h3 : B = 3 * C + 5) (h4 : C = C) (h5 : D = 3 * C^2) :
  A * B * C * D = 16896 :=
by
  sorry

end NUMINAMATH_GPT_product_equals_16896_l1739_173972


namespace NUMINAMATH_GPT_arithmetic_sequence_a12_l1739_173962

def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) (a1 d : ℤ) (h : arithmetic_sequence a a1 d) :
  a 11 = 23 :=
by
  -- condtions
  let a1_val := 1
  let d_val := 2
  have ha1 : a1 = a1_val := sorry
  have hd : d = d_val := sorry
  
  -- proof
  rw [ha1, hd] at h
  
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a12_l1739_173962


namespace NUMINAMATH_GPT_parameterization_theorem_l1739_173958

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end NUMINAMATH_GPT_parameterization_theorem_l1739_173958


namespace NUMINAMATH_GPT_simplify_T_l1739_173927

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6 * (x + 2)^5 + 15 * (x + 2)^4 + 20 * (x + 2)^3 + 15 * (x + 2)^2 + 6 * (x + 2) + 1 = (x + 3)^6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_T_l1739_173927


namespace NUMINAMATH_GPT_queen_middle_school_teachers_l1739_173904

theorem queen_middle_school_teachers
  (students : ℕ) 
  (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (h_students : students = 1500)
  (h_classes_per_student : classes_per_student = 6)
  (h_classes_per_teacher : classes_per_teacher = 5)
  (h_students_per_class : students_per_class = 25) : 
  (students * classes_per_student / students_per_class) / classes_per_teacher = 72 :=
by
  sorry

end NUMINAMATH_GPT_queen_middle_school_teachers_l1739_173904


namespace NUMINAMATH_GPT_region_to_the_upper_left_of_line_l1739_173968

variable (x y : ℝ)

def line_eqn := 3 * x - 2 * y - 6 = 0

def region := 3 * x - 2 * y - 6 < 0

theorem region_to_the_upper_left_of_line :
  ∃ rect_upper_left, (rect_upper_left = region) := 
sorry

end NUMINAMATH_GPT_region_to_the_upper_left_of_line_l1739_173968


namespace NUMINAMATH_GPT_calculate_expression_l1739_173911

def f (x : ℝ) := x^2 + 3
def g (x : ℝ) := 2 * x + 4

theorem calculate_expression : f (g 2) - g (f 2) = 49 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1739_173911


namespace NUMINAMATH_GPT_quadratic_equation_with_means_l1739_173951

theorem quadratic_equation_with_means (α β : ℝ) 
  (h_am : (α + β) / 2 = 8) 
  (h_gm : Real.sqrt (α * β) = 15) : 
  (Polynomial.X^2 - Polynomial.C (α + β) * Polynomial.X + Polynomial.C (α * β) = 0) := 
by
  have h1 : α + β = 16 := by linarith
  have h2 : α * β = 225 := by sorry
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_quadratic_equation_with_means_l1739_173951


namespace NUMINAMATH_GPT_compute_value_l1739_173989

theorem compute_value : 302^2 - 298^2 = 2400 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_l1739_173989


namespace NUMINAMATH_GPT_efficiency_ratio_l1739_173933

-- Define the work efficiencies
def EA : ℚ := 1 / 12
def EB : ℚ := 1 / 24
def EAB : ℚ := 1 / 8

-- State the theorem
theorem efficiency_ratio (EAB_eq : EAB = EA + EB) : (EA / EB) = 2 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_efficiency_ratio_l1739_173933


namespace NUMINAMATH_GPT_smallest_d_for_inverse_l1739_173919

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse :
  ∃ d, (∀ x₁ x₂, d ≤ x₁ ∧ d ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) ∧ (∀ e, (∀ x₁ x₂, e ≤ x₁ ∧ e ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) → d ≤ e) ∧ d = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_for_inverse_l1739_173919


namespace NUMINAMATH_GPT_directrix_parabola_l1739_173939

theorem directrix_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 8 * x^2 + 5) : 
  ∃ c : ℝ, ∀ x, y x = 8 * x^2 + 5 ∧ c = 159 / 32 :=
by
  use 159 / 32
  repeat { sorry }

end NUMINAMATH_GPT_directrix_parabola_l1739_173939


namespace NUMINAMATH_GPT_g_inv_g_inv_14_l1739_173925

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end NUMINAMATH_GPT_g_inv_g_inv_14_l1739_173925


namespace NUMINAMATH_GPT_polar_circle_equation_l1739_173920

theorem polar_circle_equation {r : ℝ} {phi : ℝ} {rho theta : ℝ} :
  (r = 2) → (phi = π / 3) → (rho = 4 * Real.cos (theta - π / 3)) :=
by
  intros hr hphi
  sorry

end NUMINAMATH_GPT_polar_circle_equation_l1739_173920


namespace NUMINAMATH_GPT_right_triangle_sides_l1739_173983

theorem right_triangle_sides (a b c : ℝ)
    (h1 : a + b + c = 30)
    (h2 : a^2 + b^2 = c^2)
    (h3 : ∃ r, a = (5 * r) / 2 ∧ a + b = 5 * r ∧ ∀ x y, x / y = 2 / 3) :
  a = 5 ∧ b = 12 ∧ c = 13 :=
sorry

end NUMINAMATH_GPT_right_triangle_sides_l1739_173983


namespace NUMINAMATH_GPT_mandy_book_length_l1739_173946

theorem mandy_book_length :
  let initial_length := 8
  let initial_age := 6
  let doubled_age := 2 * initial_age
  let length_at_doubled_age := 5 * initial_length
  let later_age := doubled_age + 8
  let length_at_later_age := 3 * length_at_doubled_age
  let final_length := 4 * length_at_later_age
  final_length = 480 :=
by
  sorry

end NUMINAMATH_GPT_mandy_book_length_l1739_173946


namespace NUMINAMATH_GPT_tory_sold_each_toy_gun_for_l1739_173905

theorem tory_sold_each_toy_gun_for :
  ∃ (x : ℤ), 8 * 18 = 7 * x + 4 ∧ x = 20 := 
by
  use 20
  constructor
  · sorry
  · sorry

end NUMINAMATH_GPT_tory_sold_each_toy_gun_for_l1739_173905


namespace NUMINAMATH_GPT_largest_angle_of_pentagon_l1739_173977

-- Define the angles of the pentagon and the conditions on them.
def is_angle_of_pentagon (A B C D E : ℝ) :=
  A = 108 ∧ B = 72 ∧ C = D ∧ E = 3 * C ∧
  A + B + C + D + E = 540

-- Prove the largest angle in the pentagon is 216
theorem largest_angle_of_pentagon (A B C D E : ℝ) (h : is_angle_of_pentagon A B C D E) :
  max (max (max (max A B) C) D) E = 216 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_of_pentagon_l1739_173977


namespace NUMINAMATH_GPT_no_prime_quadruple_l1739_173995

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_quadruple 
    (a b c d : ℕ)
    (ha_prime : is_prime a) 
    (hb_prime : is_prime b)
    (hc_prime : is_prime c)
    (hd_prime : is_prime d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    (1 / a + 1 / d ≠ 1 / b + 1 / c) := 
by 
  sorry

end NUMINAMATH_GPT_no_prime_quadruple_l1739_173995


namespace NUMINAMATH_GPT_surface_area_parallelepiped_l1739_173964

theorem surface_area_parallelepiped (a b : ℝ) :
  ∃ S : ℝ, (S = 3 * a * b) :=
sorry

end NUMINAMATH_GPT_surface_area_parallelepiped_l1739_173964


namespace NUMINAMATH_GPT_probability_odd_product_l1739_173960

-- Given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the proof problem
theorem probability_odd_product (h: choices = 15 ∧ odd_choices = 3) :
  (odd_choices : ℚ) / choices = 1 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_odd_product_l1739_173960


namespace NUMINAMATH_GPT_system_of_equations_solution_l1739_173997

theorem system_of_equations_solution (x y : ℝ) (h1 : |y - x| - (|x| / x) + 1 = 0) (h2 : |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0) (hx : x ≠ 0) :
  (0 < x ∧ x ≤ 0.5 ∧ y = x) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1739_173997


namespace NUMINAMATH_GPT_divide_5440_K_l1739_173929

theorem divide_5440_K (a b c d : ℕ) 
  (h1 : 5440 = a + b + c + d)
  (h2 : 2 * b = 3 * a)
  (h3 : 3 * c = 5 * b)
  (h4 : 5 * d = 6 * c) : 
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 :=
by 
  sorry

end NUMINAMATH_GPT_divide_5440_K_l1739_173929


namespace NUMINAMATH_GPT_max_S_value_l1739_173971

noncomputable def maximize_S (a b c : ℝ) : ℝ :=
  (a^2 - a * b + b^2) * (b^2 - b * c + c^2) * (c^2 - c * a + a^2)

theorem max_S_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 3 →
  maximize_S a b c ≤ 12 :=
by sorry

end NUMINAMATH_GPT_max_S_value_l1739_173971


namespace NUMINAMATH_GPT_max_gcd_dn_l1739_173959

def a (n : ℕ) := 101 + n^2

def d (n : ℕ) := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_dn : ∃ n : ℕ, ∀ m : ℕ, d m ≤ 3 := sorry

end NUMINAMATH_GPT_max_gcd_dn_l1739_173959


namespace NUMINAMATH_GPT_P_union_Q_eq_Q_l1739_173982

noncomputable def P : Set ℝ := {x : ℝ | x > 1}
noncomputable def Q : Set ℝ := {x : ℝ | x^2 - x > 0}

theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end NUMINAMATH_GPT_P_union_Q_eq_Q_l1739_173982


namespace NUMINAMATH_GPT_bathroom_new_area_l1739_173942

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end NUMINAMATH_GPT_bathroom_new_area_l1739_173942


namespace NUMINAMATH_GPT_Marcia_wardrobe_cost_l1739_173952

-- Definitions from the problem
def skirt_price : ℝ := 20
def blouse_price : ℝ := 15
def pant_price : ℝ := 30

def num_skirts : ℕ := 3
def num_blouses : ℕ := 5
def num_pants : ℕ := 2

-- The main theorem statement
theorem Marcia_wardrobe_cost :
  (num_skirts * skirt_price) + (num_blouses * blouse_price) + (pant_price + (pant_price / 2)) = 180 :=
by
  sorry

end NUMINAMATH_GPT_Marcia_wardrobe_cost_l1739_173952


namespace NUMINAMATH_GPT_max_two_terms_eq_one_l1739_173937

theorem max_two_terms_eq_one (a b c x y z : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z) :
  ∀ (P : ℕ → ℝ), -- Define P(i) as given expressions
  ((P 1 = a * x + b * y + c * z) ∧
   (P 2 = a * x + b * z + c * y) ∧
   (P 3 = a * y + b * x + c * z) ∧
   (P 4 = a * y + b * z + c * x) ∧
   (P 5 = a * z + b * x + c * y) ∧
   (P 6 = a * z + b * y + c * x)) →
  (P 1 = 1 ∨ P 2 = 1 ∨ P 3 = 1 ∨ P 4 = 1 ∨ P 5 = 1 ∨ P 6 = 1) →
  (∃ i j, i ≠ j ∧ P i = 1 ∧ P j = 1) →
  ¬(∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ P i = 1 ∧ P j = 1 ∧ P k = 1) :=
sorry

end NUMINAMATH_GPT_max_two_terms_eq_one_l1739_173937


namespace NUMINAMATH_GPT_jason_two_weeks_eggs_l1739_173981

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end NUMINAMATH_GPT_jason_two_weeks_eggs_l1739_173981


namespace NUMINAMATH_GPT_compare_a_b_c_l1739_173908

noncomputable def a := Real.sin (Real.pi / 5)
noncomputable def b := Real.logb (Real.sqrt 2) (Real.sqrt 3)
noncomputable def c := (1 / 4)^(2 / 3)

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_compare_a_b_c_l1739_173908


namespace NUMINAMATH_GPT_remaining_budget_after_purchases_l1739_173969

theorem remaining_budget_after_purchases :
  let budget := 80
  let fried_chicken_cost := 12
  let beef_cost_per_pound := 3
  let beef_quantity := 4.5
  let soup_cost_per_can := 2
  let soup_quantity := 3
  let milk_original_price := 4
  let milk_discount := 0.10
  let beef_cost := beef_quantity * beef_cost_per_pound
  let paid_soup_quantity := soup_quantity / 2
  let milk_discounted_price := milk_original_price * (1 - milk_discount)
  let total_cost := fried_chicken_cost + beef_cost + (paid_soup_quantity * soup_cost_per_can) + milk_discounted_price
  let remaining_budget := budget - total_cost
  remaining_budget = 47.90 :=
by
  sorry

end NUMINAMATH_GPT_remaining_budget_after_purchases_l1739_173969


namespace NUMINAMATH_GPT_tennis_balls_in_each_container_l1739_173943

theorem tennis_balls_in_each_container (initial_balls : ℕ) (half_gone : ℕ) (remaining_balls : ℕ) (containers : ℕ) 
  (h1 : initial_balls = 100) 
  (h2 : half_gone = initial_balls / 2)
  (h3 : remaining_balls = initial_balls - half_gone)
  (h4 : containers = 5) :
  remaining_balls / containers = 10 := 
by
  sorry

end NUMINAMATH_GPT_tennis_balls_in_each_container_l1739_173943


namespace NUMINAMATH_GPT_bill_amount_each_person_shared_l1739_173913

noncomputable def total_bill : ℝ := 139.00
noncomputable def tip_percentage : ℝ := 0.10
noncomputable def num_people : ℝ := 7.00

noncomputable def tip : ℝ := tip_percentage * total_bill
noncomputable def total_bill_with_tip : ℝ := total_bill + tip
noncomputable def amount_each_person_pays : ℝ := total_bill_with_tip / num_people

theorem bill_amount_each_person_shared :
  amount_each_person_pays = 21.84 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_bill_amount_each_person_shared_l1739_173913


namespace NUMINAMATH_GPT_john_has_500_dollars_l1739_173954

-- Define the initial amount and the condition
def initial_amount : ℝ := 1600
def condition (spent : ℝ) : Prop := (1600 - spent) = (spent - 600)

-- The final amount of money John still has
def final_amount (spent : ℝ) : ℝ := initial_amount - spent

-- The main theorem statement
theorem john_has_500_dollars : ∃ (spent : ℝ), condition spent ∧ final_amount spent = 500 :=
by
  sorry

end NUMINAMATH_GPT_john_has_500_dollars_l1739_173954


namespace NUMINAMATH_GPT_min_value_inverse_sum_l1739_173990

theorem min_value_inverse_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) :
  (1 / x + 1 / y + 1 / z) ≥ 9 :=
  sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l1739_173990


namespace NUMINAMATH_GPT_equidistant_points_quadrants_l1739_173934

theorem equidistant_points_quadrants (x y : ℝ)
  (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : abs x = abs y) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_equidistant_points_quadrants_l1739_173934


namespace NUMINAMATH_GPT_no_even_is_prime_equiv_l1739_173996

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end NUMINAMATH_GPT_no_even_is_prime_equiv_l1739_173996


namespace NUMINAMATH_GPT_remainder_of_product_divided_by_7_l1739_173932

theorem remainder_of_product_divided_by_7 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_divided_by_7_l1739_173932


namespace NUMINAMATH_GPT_Alfred_repair_cost_l1739_173907

noncomputable def scooter_price : ℕ := 4700
noncomputable def sale_price : ℕ := 5800
noncomputable def gain_percent : ℚ := 9.433962264150944
noncomputable def gain_value (repair_cost : ℚ) : ℚ := sale_price - (scooter_price + repair_cost)

theorem Alfred_repair_cost : ∃ R : ℚ, gain_percent = (gain_value R / (scooter_price + R)) * 100 ∧ R = 600 :=
by
  sorry

end NUMINAMATH_GPT_Alfred_repair_cost_l1739_173907


namespace NUMINAMATH_GPT_student_weight_l1739_173910

-- Define the weights of the student and sister
variables (S R : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := S - 5 = 1.25 * R
def condition2 : Prop := S + R = 104

-- The theorem we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 60 := 
by
  sorry

end NUMINAMATH_GPT_student_weight_l1739_173910


namespace NUMINAMATH_GPT_simplify_fraction_l1739_173912

theorem simplify_fraction :
  (6 * x ^ 3 + 13 * x ^ 2 + 15 * x - 25) / (2 * x ^ 3 + 4 * x ^ 2 + 4 * x - 10) =
  (6 * x - 5) / (2 * x - 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1739_173912


namespace NUMINAMATH_GPT_birth_year_1957_l1739_173976

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem birth_year_1957 (y x : ℕ) (h : y = 2023) (h1 : sum_of_digits x = y - x) : x = 1957 :=
by
  sorry

end NUMINAMATH_GPT_birth_year_1957_l1739_173976


namespace NUMINAMATH_GPT_part1_max_value_l1739_173916

variable (f : ℝ → ℝ)
def is_maximum (y : ℝ) := ∀ x : ℝ, f x ≤ y

theorem part1_max_value (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + m*x + 1) :
  m = 0 → (exists y, is_maximum f y ∧ y = 1) := 
sorry

end NUMINAMATH_GPT_part1_max_value_l1739_173916


namespace NUMINAMATH_GPT_houses_distance_l1739_173900

theorem houses_distance (num_houses : ℕ) (total_length : ℝ) (at_both_ends : Bool) 
  (h1: num_houses = 6) (h2: total_length = 11.5) (h3: at_both_ends = true) : 
  total_length / (num_houses - 1) = 2.3 := 
by
  sorry

end NUMINAMATH_GPT_houses_distance_l1739_173900
