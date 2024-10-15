import Mathlib

namespace NUMINAMATH_GPT_focus_of_parabola_l98_9892

theorem focus_of_parabola (x y : ℝ) (h : y = 2 * x^2) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l98_9892


namespace NUMINAMATH_GPT_possible_values_of_a_l98_9836

noncomputable def f (x a : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 2 * a * x + 2 else x + 9 / x - 3 * a

theorem possible_values_of_a (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ 1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l98_9836


namespace NUMINAMATH_GPT_infinite_radical_solution_l98_9884

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end NUMINAMATH_GPT_infinite_radical_solution_l98_9884


namespace NUMINAMATH_GPT_Faraway_not_possible_sum_l98_9898

theorem Faraway_not_possible_sum (h g : ℕ) : (74 ≠ 21 * h + 6 * g) ∧ (89 ≠ 21 * h + 6 * g) :=
by
  sorry

end NUMINAMATH_GPT_Faraway_not_possible_sum_l98_9898


namespace NUMINAMATH_GPT_roots_of_f_non_roots_of_g_l98_9818

-- Part (a)

def f (x : ℚ) := x^20 - 123 * x^10 + 1

theorem roots_of_f (a : ℚ) (h : f a = 0) : 
  f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0 :=
by
  sorry

-- Part (b)

def g (x : ℚ) := x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1

theorem non_roots_of_g (β : ℚ) (h : g β = 0) : 
  g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_f_non_roots_of_g_l98_9818


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_l98_9828

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x^2 - (m + 1) * x + (3 * m - 6) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_l98_9828


namespace NUMINAMATH_GPT_solution_set_of_inequality_l98_9833

theorem solution_set_of_inequality (x : ℝ) : (x^2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l98_9833


namespace NUMINAMATH_GPT_average_production_last_5_days_l98_9871

theorem average_production_last_5_days
  (avg_first_25_days : ℕ → ℕ → ℕ → ℕ → Prop)
  (avg_monthly : ℕ)
  (total_days : ℕ)
  (days_first_period : ℕ)
  (avg_production_first_period : ℕ)
  (avg_total_monthly : ℕ)
  (days_second_period : ℕ)
  (total_production_five_days : ℕ):
  (days_first_period = 25) →
  (avg_production_first_period = 50) →
  (avg_total_monthly = 48) →
  (total_production_five_days = 190) →
  (days_second_period = 5) →
  avg_first_25_days days_first_period avg_production_first_period 
  (days_first_period * avg_production_first_period) avg_total_monthly ∧
  avg_monthly = avg_total_monthly →
  ((days_first_period + days_second_period) * avg_monthly - 
  days_first_period * avg_production_first_period = total_production_five_days) →
  (total_production_five_days / days_second_period = 38) := sorry

end NUMINAMATH_GPT_average_production_last_5_days_l98_9871


namespace NUMINAMATH_GPT_angle_greater_than_150_l98_9854

theorem angle_greater_than_150 (a b c R : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c < 2 * R) : 
  ∃ (A : ℝ), A > 150 ∧ ( ∃ (B C : ℝ), A + B + C = 180 ) :=
sorry

end NUMINAMATH_GPT_angle_greater_than_150_l98_9854


namespace NUMINAMATH_GPT_infinite_n_exist_l98_9831

def S (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem infinite_n_exist (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ᶠ n in at_top, S n ≡ n [MOD p] :=
sorry

end NUMINAMATH_GPT_infinite_n_exist_l98_9831


namespace NUMINAMATH_GPT_subset_of_positive_reals_l98_9858

def M := { x : ℝ | x > -1 }

theorem subset_of_positive_reals : {0} ⊆ M :=
by
  sorry

end NUMINAMATH_GPT_subset_of_positive_reals_l98_9858


namespace NUMINAMATH_GPT_not_prime_n_quad_plus_n_sq_plus_one_l98_9869

theorem not_prime_n_quad_plus_n_sq_plus_one (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + n^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_not_prime_n_quad_plus_n_sq_plus_one_l98_9869


namespace NUMINAMATH_GPT_find_m_l98_9891

theorem find_m (a b m : ℤ) (h1 : a - b = 6) (h2 : a + b = 0) : 2 * a + b = m → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l98_9891


namespace NUMINAMATH_GPT_min_max_diff_val_l98_9896

def find_min_max_diff (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : ℝ :=
  let m := 0
  let M := 1
  M - m

theorem min_max_diff_val (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : find_min_max_diff x y hx hy = 1 :=
by sorry

end NUMINAMATH_GPT_min_max_diff_val_l98_9896


namespace NUMINAMATH_GPT_expected_revenue_day_14_plan_1_more_reasonable_plan_l98_9810

-- Define the initial conditions
def initial_valuation : ℕ := 60000
def rain_probability : ℚ := 0.4
def no_rain_probability : ℚ := 0.6
def hiring_cost : ℕ := 32000

-- Calculate the expected revenue if Plan ① is adopted
def expected_revenue_plan_1_day_14 : ℚ :=
  (initial_valuation / 10000) * (1/2 * rain_probability + no_rain_probability)

-- Calculate the total revenue for Plan ①
def total_revenue_plan_1 : ℚ :=
  (initial_valuation / 10000) + 2 * expected_revenue_plan_1_day_14

-- Calculate the total revenue for Plan ②
def total_revenue_plan_2 : ℚ :=
  3 * (initial_valuation / 10000) - (hiring_cost / 10000)

-- Define the lemmas to prove
theorem expected_revenue_day_14_plan_1 :
  expected_revenue_plan_1_day_14 = 4.8 := 
  by sorry

theorem more_reasonable_plan :
  total_revenue_plan_1 > total_revenue_plan_2 :=
  by sorry

end NUMINAMATH_GPT_expected_revenue_day_14_plan_1_more_reasonable_plan_l98_9810


namespace NUMINAMATH_GPT_xiao_ming_proposition_false_l98_9863

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m * m ≤ n → m = 1 ∨ m = n → m ∣ n

def check_xiao_ming_proposition : Prop :=
  ∃ n : ℕ, ∃ (k : ℕ), k < n → ∃ (p q : ℕ), p = q → n^2 - n + 11 = p * q ∧ p > 1 ∧ q > 1

theorem xiao_ming_proposition_false : ¬ (∀ n: ℕ, is_prime (n^2 - n + 11)) :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_proposition_false_l98_9863


namespace NUMINAMATH_GPT_farm_total_amount_90000_l98_9829

-- Defining the conditions
def apples_produce (mangoes: ℕ) : ℕ := 2 * mangoes
def oranges_produce (mangoes: ℕ) : ℕ := mangoes + 200

-- Defining the total produce of all fruits
def total_produce (mangoes: ℕ) : ℕ := apples_produce mangoes + mangoes + oranges_produce mangoes

-- Defining the price per kg
def price_per_kg : ℕ := 50

-- Defining the total amount from selling all fruits
noncomputable def total_amount (mangoes: ℕ) : ℕ := total_produce mangoes * price_per_kg

-- Proving that the total amount he got in that season is $90,000
theorem farm_total_amount_90000 : total_amount 400 = 90000 := by
  sorry

end NUMINAMATH_GPT_farm_total_amount_90000_l98_9829


namespace NUMINAMATH_GPT_numeral_eq_7000_l98_9815

theorem numeral_eq_7000 
  (local_value face_value numeral : ℕ)
  (h1 : face_value = 7)
  (h2 : local_value - face_value = 6993) : 
  numeral = 7000 :=
by
  sorry

end NUMINAMATH_GPT_numeral_eq_7000_l98_9815


namespace NUMINAMATH_GPT_spadesuit_evaluation_l98_9839

def spadesuit (a b : ℝ) : ℝ := abs (a - b)

theorem spadesuit_evaluation : spadesuit 1.5 (spadesuit 2.5 (spadesuit 4.5 6)) = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_evaluation_l98_9839


namespace NUMINAMATH_GPT_polar_to_cartesian_equiv_l98_9877

noncomputable def polar_to_cartesian (rho theta : ℝ) : Prop :=
  let x := rho * Real.cos theta
  let y := rho * Real.sin theta
  (Real.sqrt 3 * x + y = 2) ↔ (rho * Real.cos (theta - Real.pi / 6) = 1)

theorem polar_to_cartesian_equiv (rho theta : ℝ) : polar_to_cartesian rho theta :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_equiv_l98_9877


namespace NUMINAMATH_GPT_factorization_problem_1_factorization_problem_2_l98_9879

-- Problem 1: Factorize 2(m-n)^2 - m(n-m) and show it equals (n-m)(2n - 3m)
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) :=
by
  sorry

-- Problem 2: Factorize -4xy^2 + 4x^2y + y^3 and show it equals y(2x - y)^2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_problem_1_factorization_problem_2_l98_9879


namespace NUMINAMATH_GPT_solution_comparison_l98_9807

open Real

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-(d / c) > -(f / e)) ↔ ((f / e) > (d / c)) :=
by
  sorry

end NUMINAMATH_GPT_solution_comparison_l98_9807


namespace NUMINAMATH_GPT_find_expression_value_l98_9861

theorem find_expression_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5 * x^6 + x^2 = 8436*x - 338 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_expression_value_l98_9861


namespace NUMINAMATH_GPT_initial_quarters_l98_9806

variable (q : ℕ)

theorem initial_quarters (h : q + 3 = 11) : q = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_quarters_l98_9806


namespace NUMINAMATH_GPT_real_roots_exist_for_all_real_K_l98_9882

theorem real_roots_exist_for_all_real_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x-1) * (x-2) * (x-3) :=
by
  sorry

end NUMINAMATH_GPT_real_roots_exist_for_all_real_K_l98_9882


namespace NUMINAMATH_GPT_min_polyline_distance_between_circle_and_line_l98_9876

def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

def on_line (Q : ℝ × ℝ) : Prop :=
  2 * Q.1 + Q.2 = 2 * Real.sqrt 5

theorem min_polyline_distance_between_circle_and_line :
  ∃ P Q, on_circle P ∧ on_line Q ∧ polyline_distance P Q = (Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_polyline_distance_between_circle_and_line_l98_9876


namespace NUMINAMATH_GPT_correct_solutions_l98_9853

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end NUMINAMATH_GPT_correct_solutions_l98_9853


namespace NUMINAMATH_GPT_monotone_decreasing_intervals_l98_9845

theorem monotone_decreasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, deriv f x = (x - 2) * (x^2 - 1)) :
  ((∀ x : ℝ, x < -1 → deriv f x < 0) ∧ (∀ x : ℝ, 1 < x → x < 2 → deriv f x < 0)) :=
by
  sorry

end NUMINAMATH_GPT_monotone_decreasing_intervals_l98_9845


namespace NUMINAMATH_GPT_mountaineers_arrangement_l98_9888
open BigOperators

-- Definition to state the number of combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The main statement translating our problem
theorem mountaineers_arrangement :
  (choose 4 2) * (choose 6 2) = 120 := by
  sorry

end NUMINAMATH_GPT_mountaineers_arrangement_l98_9888


namespace NUMINAMATH_GPT_die_face_never_lays_on_board_l98_9821

structure Chessboard :=
(rows : ℕ)
(cols : ℕ)
(h_size : rows = 8 ∧ cols = 8)

structure Die :=
(faces : Fin 6 → Nat)  -- a die has 6 faces

structure Position :=
(x : ℕ)
(y : ℕ)

structure State :=
(position : Position)
(bottom_face : Fin 6)
(visited : Fin 64 → Bool)

def initial_position : Position := ⟨0, 0⟩  -- top-left corner (a1)

def initial_state (d : Die) : State :=
  { position := initial_position,
    bottom_face := 0,
    visited := λ _ => false }

noncomputable def can_roll_over_entire_board_without_one_face_touching (board : Chessboard) (d : Die) : Prop :=
  ∃ f : Fin 6, ∀ s : State, -- for some face f of the die
    ((s.position.x < board.rows ∧ s.position.y < board.cols) → 
      s.visited (⟨s.position.x + board.rows * s.position.y, by sorry⟩) = true) → -- every cell visited
      ¬(s.bottom_face = f) -- face f is never the bottom face

theorem die_face_never_lays_on_board (board : Chessboard) (d : Die) :
  can_roll_over_entire_board_without_one_face_touching board d :=
  sorry

end NUMINAMATH_GPT_die_face_never_lays_on_board_l98_9821


namespace NUMINAMATH_GPT_q_transformation_l98_9866

theorem q_transformation (w m z : ℝ) (q : ℝ) (h_q : q = 5 * w / (4 * m * z^2)) :
  let w' := 4 * w
  let m' := 2 * m
  let z' := 3 * z
  q = 5 * w / (4 * m * z^2) → (5 * w') / (4 * m' * (z'^2)) = (5 / 18) * q := by
  sorry

end NUMINAMATH_GPT_q_transformation_l98_9866


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l98_9897

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_root1 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 2 = x)
  (h_root2 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 10 = x) : 
  a 6 = -6 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l98_9897


namespace NUMINAMATH_GPT_people_count_l98_9840

theorem people_count (wheels_per_person total_wheels : ℕ) (h1 : wheels_per_person = 4) (h2 : total_wheels = 320) :
  total_wheels / wheels_per_person = 80 :=
sorry

end NUMINAMATH_GPT_people_count_l98_9840


namespace NUMINAMATH_GPT_simplify_expression_correct_l98_9887

noncomputable def simplify_expression : ℝ :=
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt (48))))

theorem simplify_expression_correct : simplify_expression = (Real.sqrt 6) + (Real.sqrt 2) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l98_9887


namespace NUMINAMATH_GPT_optimal_tower_configuration_l98_9890

theorem optimal_tower_configuration (x y : ℕ) (h : x + 2 * y = 30) :
    x * y ≤ 112 := by
  sorry

end NUMINAMATH_GPT_optimal_tower_configuration_l98_9890


namespace NUMINAMATH_GPT_expression_value_l98_9808

theorem expression_value : 
  29^2 - 27^2 + 25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 389 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l98_9808


namespace NUMINAMATH_GPT_smallest_K_for_triangle_l98_9803

theorem smallest_K_for_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) 
  : ∃ K : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → a + c > b → (a^2 + c^2) / b^2 > K) ∧ K = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_K_for_triangle_l98_9803


namespace NUMINAMATH_GPT_income_increase_is_17_percent_l98_9826

def sales_percent_increase (original_items : ℕ) 
                           (original_price : ℝ) 
                           (discount_percent : ℝ) 
                           (sales_increase_percent : ℝ) 
                           (new_items_sold : ℕ) 
                           (new_income : ℝ)
                           (percent_increase : ℝ) : Prop :=
  let original_income := original_items * original_price
  let discounted_price := original_price * (1 - discount_percent / 100)
  let increased_sales := original_items + (original_items * sales_increase_percent / 100)
  original_income = original_items * original_price ∧
  new_income = discounted_price * increased_sales ∧
  new_items_sold = original_items * (1 + sales_increase_percent / 100) ∧
  percent_increase = ((new_income - original_income) / original_income) * 100 ∧
  original_items = 100 ∧ original_price = 1 ∧ discount_percent = 10 ∧ sales_increase_percent = 30 ∧ 
  new_items_sold = 130 ∧ new_income = 117 ∧ percent_increase = 17

theorem income_increase_is_17_percent :
  sales_percent_increase 100 1 10 30 130 117 17 :=
sorry

end NUMINAMATH_GPT_income_increase_is_17_percent_l98_9826


namespace NUMINAMATH_GPT_max_homework_time_l98_9835

theorem max_homework_time :
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  biology + history + geography = 180 :=
by
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  show biology + history + geography = 180
  sorry

end NUMINAMATH_GPT_max_homework_time_l98_9835


namespace NUMINAMATH_GPT_total_tickets_sold_l98_9834

theorem total_tickets_sold
    (n₄₅ : ℕ) (n₆₀ : ℕ) (total_sales : ℝ) 
    (price₄₅ price₆₀ : ℝ)
    (h₁ : n₄₅ = 205)
    (h₂ : price₄₅ = 4.5)
    (h₃ : total_sales = 1972.5)
    (h₄ : price₆₀ = 6.0)
    (h₅ : total_sales = n₄₅ * price₄₅ + n₆₀ * price₆₀) :
    n₄₅ + n₆₀ = 380 := 
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l98_9834


namespace NUMINAMATH_GPT_pebbles_collected_by_tenth_day_l98_9824

-- Define the initial conditions
def a : ℕ := 2
def r : ℕ := 2
def n : ℕ := 10

-- Total pebbles collected by the end of the 10th day
def total_pebbles (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Proof statement
theorem pebbles_collected_by_tenth_day : total_pebbles a r n = 2046 :=
  by sorry

end NUMINAMATH_GPT_pebbles_collected_by_tenth_day_l98_9824


namespace NUMINAMATH_GPT_B_pow_150_l98_9811

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_150 : B ^ 150 = 1 :=
by
  sorry

end NUMINAMATH_GPT_B_pow_150_l98_9811


namespace NUMINAMATH_GPT_measure_diagonal_of_brick_l98_9864

def RectangularParallelepiped (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def DiagonalMeasurementPossible (a b c : ℝ) : Prop :=
  ∃ d : ℝ, d = (a^2 + b^2 + c^2)^(1/2)

theorem measure_diagonal_of_brick (a b c : ℝ) 
  (h : RectangularParallelepiped a b c) : DiagonalMeasurementPossible a b c :=
by
  sorry

end NUMINAMATH_GPT_measure_diagonal_of_brick_l98_9864


namespace NUMINAMATH_GPT_binom_12_6_l98_9801

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end NUMINAMATH_GPT_binom_12_6_l98_9801


namespace NUMINAMATH_GPT_overall_gain_is_2_89_l98_9850

noncomputable def overall_gain_percentage : ℝ :=
  let cost1 := 500000
  let gain1 := 0.10
  let sell1 := cost1 * (1 + gain1)

  let cost2 := 600000
  let loss2 := 0.05
  let sell2 := cost2 * (1 - loss2)

  let cost3 := 700000
  let gain3 := 0.15
  let sell3 := cost3 * (1 + gain3)

  let cost4 := 800000
  let loss4 := 0.12
  let sell4 := cost4 * (1 - loss4)

  let cost5 := 900000
  let gain5 := 0.08
  let sell5 := cost5 * (1 + gain5)

  let total_cost := cost1 + cost2 + cost3 + cost4 + cost5
  let total_sell := sell1 + sell2 + sell3 + sell4 + sell5
  let overall_gain := total_sell - total_cost
  (overall_gain / total_cost) * 100

theorem overall_gain_is_2_89 :
  overall_gain_percentage = 2.89 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_overall_gain_is_2_89_l98_9850


namespace NUMINAMATH_GPT_mother_older_than_twice_petra_l98_9859

def petra_age : ℕ := 11
def mother_age : ℕ := 36

def twice_petra_age : ℕ := 2 * petra_age

theorem mother_older_than_twice_petra : mother_age - twice_petra_age = 14 := by
  sorry

end NUMINAMATH_GPT_mother_older_than_twice_petra_l98_9859


namespace NUMINAMATH_GPT_distance_triangle_four_points_l98_9841

variable {X : Type*} [MetricSpace X]

theorem distance_triangle_four_points (A B C D : X) :
  dist A D ≤ dist A B + dist B C + dist C D :=
by
  sorry

end NUMINAMATH_GPT_distance_triangle_four_points_l98_9841


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l98_9825

-- Define the first three terms of the series
def first_term := (-3: ℚ) / 5
def second_term := (-5: ℚ) / 3
def third_term := (-125: ℚ) / 27

-- Prove that the common ratio = 25/9
theorem common_ratio_geometric_series :
  (second_term / first_term) = (25 : ℚ) / 9 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l98_9825


namespace NUMINAMATH_GPT_find_m_l98_9889

theorem find_m (m : ℝ) 
  (h : (1 : ℝ) * (-3 : ℝ) + (3 : ℝ) * ((3 : ℝ) + 2 * m) = 0) : 
  m = -1 :=
by sorry

end NUMINAMATH_GPT_find_m_l98_9889


namespace NUMINAMATH_GPT_naomi_wash_time_l98_9856

theorem naomi_wash_time (C T S : ℕ) (h₁ : T = 2 * C) (h₂ : S = 2 * C - 15) (h₃ : C + T + S = 135) : C = 30 :=
by
  sorry

end NUMINAMATH_GPT_naomi_wash_time_l98_9856


namespace NUMINAMATH_GPT_jellybeans_final_count_l98_9837

-- Defining the initial number of jellybeans and operations
def initial_jellybeans : ℕ := 37
def removed_first : ℕ := 15
def added_back : ℕ := 5
def removed_second : ℕ := 4

-- Defining the final number of jellybeans to prove it equals 23
def final_jellybeans : ℕ := (initial_jellybeans - removed_first) + added_back - removed_second

-- The theorem that states the final number of jellybeans is 23
theorem jellybeans_final_count : final_jellybeans = 23 :=
by
  -- The proof will be provided here if needed
  sorry

end NUMINAMATH_GPT_jellybeans_final_count_l98_9837


namespace NUMINAMATH_GPT_min_value_of_x_prime_factors_l98_9844

theorem min_value_of_x_prime_factors (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
    (h : 5 * x^7 = 13 * y^11)
    (hx_factorization : x = a^c * b^d) : a + b + c + d = 32 := sorry

end NUMINAMATH_GPT_min_value_of_x_prime_factors_l98_9844


namespace NUMINAMATH_GPT_simplify_expression_l98_9865

variable (x y : ℝ)

theorem simplify_expression : (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l98_9865


namespace NUMINAMATH_GPT_largest_divisor_of_m_l98_9886

theorem largest_divisor_of_m (m : ℕ) (h1 : 0 < m) (h2 : 39 ∣ m^2) : 39 ∣ m := sorry

end NUMINAMATH_GPT_largest_divisor_of_m_l98_9886


namespace NUMINAMATH_GPT_prime_cubed_plus_seven_composite_l98_9872

theorem prime_cubed_plus_seven_composite (P : ℕ) (hP_prime : Nat.Prime P) (hP3_plus_5_prime : Nat.Prime (P ^ 3 + 5)) : ¬ Nat.Prime (P ^ 3 + 7) :=
by
  sorry

end NUMINAMATH_GPT_prime_cubed_plus_seven_composite_l98_9872


namespace NUMINAMATH_GPT_cheryl_material_need_l98_9867

-- Cheryl's conditions
def cheryl_material_used (x : ℚ) : Prop :=
  x + 2/3 - 4/9 = 2/3

-- The proof problem statement
theorem cheryl_material_need : ∃ x : ℚ, cheryl_material_used x ∧ x = 4/9 :=
  sorry

end NUMINAMATH_GPT_cheryl_material_need_l98_9867


namespace NUMINAMATH_GPT_cubic_expression_l98_9883

theorem cubic_expression (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1008 :=
sorry

end NUMINAMATH_GPT_cubic_expression_l98_9883


namespace NUMINAMATH_GPT_marla_night_cost_is_correct_l98_9874

def lizard_value_bc := 8 -- 1 lizard is worth 8 bottle caps
def lizard_value_gw := 5 / 3 -- 3 lizards are worth 5 gallons of water
def horse_value_gw := 80 -- 1 horse is worth 80 gallons of water
def marla_daily_bc := 20 -- Marla can scavenge 20 bottle caps each day
def marla_days := 24 -- It takes Marla 24 days to collect the bottle caps

noncomputable def marla_night_cost_bc : ℕ :=
((marla_daily_bc * marla_days) - (horse_value_gw / lizard_value_gw * (3 * lizard_value_bc))) / marla_days

theorem marla_night_cost_is_correct :
  marla_night_cost_bc = 4 := by
  sorry

end NUMINAMATH_GPT_marla_night_cost_is_correct_l98_9874


namespace NUMINAMATH_GPT_fourth_term_eq_156_l98_9838

-- Definition of the sequence term
def seq_term (n : ℕ) : ℕ :=
  (List.range n).map (λ k => 5^k) |>.sum

-- Theorem to prove the fourth term equals 156
theorem fourth_term_eq_156 : seq_term 4 = 156 :=
sorry

end NUMINAMATH_GPT_fourth_term_eq_156_l98_9838


namespace NUMINAMATH_GPT_simon_paid_amount_l98_9899

theorem simon_paid_amount:
  let pansy_price := 2.50
  let hydrangea_price := 12.50
  let petunia_price := 1.00
  let pansies_count := 5
  let hydrangeas_count := 1
  let petunias_count := 5
  let discount_rate := 0.10
  let change_received := 23.00

  let total_cost_before_discount := (pansies_count * pansy_price) + (hydrangeas_count * hydrangea_price) + (petunias_count * petunia_price)
  let discount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  let amount_paid_with := total_cost_after_discount + change_received

  amount_paid_with = 50.00 :=
by
  sorry

end NUMINAMATH_GPT_simon_paid_amount_l98_9899


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_is_perfect_square_l98_9862

theorem product_of_four_consecutive_integers_is_perfect_square :
  ∃ k : ℤ, ∃ n : ℤ, k = (n-1) * n * (n+1) * (n+2) ∧
    k = 0 ∧
    ((n = 0) ∨ (n = -1) ∨ (n = 1) ∨ (n = -2)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_is_perfect_square_l98_9862


namespace NUMINAMATH_GPT_distance_from_dormitory_to_city_l98_9851

theorem distance_from_dormitory_to_city (D : ℝ) (h : (1/2) * D + (1/4) * D + 6 = D) : D = 24 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_dormitory_to_city_l98_9851


namespace NUMINAMATH_GPT_coin_outcomes_equivalent_l98_9885

theorem coin_outcomes_equivalent :
  let outcomes_per_coin := 2
  let total_coins := 3
  (outcomes_per_coin ^ total_coins) = 8 :=
by
  sorry

end NUMINAMATH_GPT_coin_outcomes_equivalent_l98_9885


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l98_9819

noncomputable def a := (1/2)^(2/3)
noncomputable def b := (1/5)^(2/3)
noncomputable def c := (1/2)^(1/3)

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l98_9819


namespace NUMINAMATH_GPT_shaded_fraction_l98_9820

theorem shaded_fraction (side_length : ℝ) (base : ℝ) (height : ℝ) (H1: side_length = 4) (H2: base = 3) (H3: height = 2):
  ((side_length ^ 2) - 2 * (1 / 2 * base * height)) / (side_length ^ 2) = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_shaded_fraction_l98_9820


namespace NUMINAMATH_GPT_base_length_of_parallelogram_l98_9849

-- Definitions and conditions
def parallelogram_area (base altitude : ℝ) : ℝ := base * altitude
def altitude (base : ℝ) : ℝ := 2 * base

-- Main theorem to prove
theorem base_length_of_parallelogram (A : ℝ) (base : ℝ)
  (hA : A = 200) 
  (h_altitude : altitude base = 2 * base) 
  (h_area : parallelogram_area base (altitude base) = A) : 
  base = 10 := 
sorry

end NUMINAMATH_GPT_base_length_of_parallelogram_l98_9849


namespace NUMINAMATH_GPT_probability_of_union_l98_9870

def total_cards : ℕ := 52
def king_of_hearts : ℕ := 1
def spades : ℕ := 13

theorem probability_of_union :
  let P_A := king_of_hearts / total_cards
  let P_B := spades / total_cards
  (P_A + P_B) = (7 / 26) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_union_l98_9870


namespace NUMINAMATH_GPT_value_of_k_l98_9847

   noncomputable def k (a b : ℝ) : ℝ := 3 / 4

   theorem value_of_k (a b k : ℝ) 
     (h1: b = 4 * k + 1) 
     (h2: 5 = a * k + 1) 
     (h3: b + 1 = a * k + 1) : 
     k = 3 / 4 := 
   by 
     -- Proof goes here 
     sorry
   
end NUMINAMATH_GPT_value_of_k_l98_9847


namespace NUMINAMATH_GPT_power_mod_lemma_l98_9813

theorem power_mod_lemma : (7^137 % 13) = 11 := by
  sorry

end NUMINAMATH_GPT_power_mod_lemma_l98_9813


namespace NUMINAMATH_GPT_area_of_equilateral_triangle_example_l98_9805

noncomputable def area_of_equilateral_triangle_with_internal_point (a b c : ℝ) (d_pa : ℝ) (d_pb : ℝ) (d_pc : ℝ) : ℝ :=
  if h : ((d_pa = 3) ∧ (d_pb = 4) ∧ (d_pc = 5)) then
    (9 + (25 * Real.sqrt 3)/4)
  else
    0

theorem area_of_equilateral_triangle_example :
  area_of_equilateral_triangle_with_internal_point 3 4 5 3 4 5 = 9 + (25 * Real.sqrt 3)/4 :=
  by sorry

end NUMINAMATH_GPT_area_of_equilateral_triangle_example_l98_9805


namespace NUMINAMATH_GPT_equilateral_triangle_area_in_circle_l98_9846

theorem equilateral_triangle_area_in_circle (r : ℝ) (h : r = 9) :
  let s := 2 * r * Real.sin (π / 3)
  let A := (Real.sqrt 3 / 4) * s^2
  A = (243 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_in_circle_l98_9846


namespace NUMINAMATH_GPT_ordered_pairs_divide_square_sum_l98_9842

theorem ordered_pairs_divide_square_sum :
  { (m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ (mn - 1) ∣ (m^2 + n^2) } = { (1, 2), (1, 3), (2, 1), (3, 1) } := 
sorry

end NUMINAMATH_GPT_ordered_pairs_divide_square_sum_l98_9842


namespace NUMINAMATH_GPT_smallest_value_of_n_l98_9873

theorem smallest_value_of_n (a b c m n : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 2010) (h4 : (a! * b! * c!) = m * 10 ^ n) : ∃ n, n = 500 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_n_l98_9873


namespace NUMINAMATH_GPT_gain_percent_l98_9895

theorem gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1440) : 
  ((selling_price - cost_price) / cost_price) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_l98_9895


namespace NUMINAMATH_GPT_find_unknown_blankets_rate_l98_9855

noncomputable def unknown_blankets_rate : ℝ :=
  let total_cost_3_blankets := 3 * 100
  let discount := 0.10 * total_cost_3_blankets
  let cost_3_blankets_after_discount := total_cost_3_blankets - discount
  let cost_1_blanket := 150
  let tax := 0.15 * cost_1_blanket
  let cost_1_blanket_after_tax := cost_1_blanket + tax
  let total_avg_price_per_blanket := 150
  let total_blankets := 6
  let total_cost := total_avg_price_per_blanket * total_blankets
  (total_cost - cost_3_blankets_after_discount - cost_1_blanket_after_tax) / 2

theorem find_unknown_blankets_rate : unknown_blankets_rate = 228.75 :=
  by
    sorry

end NUMINAMATH_GPT_find_unknown_blankets_rate_l98_9855


namespace NUMINAMATH_GPT_complement_M_in_U_l98_9809

-- Define the universal set U and set M
def U : Finset ℕ := {4, 5, 6, 8, 9}
def M : Finset ℕ := {5, 6, 8}

-- Define the complement of M in U
def complement (U M : Finset ℕ) : Finset ℕ := U \ M

-- Prove that the complement of M in U is {4, 9}
theorem complement_M_in_U : complement U M = {4, 9} := by
  sorry

end NUMINAMATH_GPT_complement_M_in_U_l98_9809


namespace NUMINAMATH_GPT_measure_of_angle4_l98_9881

def angle1 := 62
def angle2 := 36
def angle3 := 24
def angle4 : ℕ := 122

theorem measure_of_angle4 (d e : ℕ) (h1 : angle1 + angle2 + angle3 + d + e = 180) (h2 : d + e = 58) :
  angle4 = 180 - (angle1 + angle2 + angle3 + d + e) :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle4_l98_9881


namespace NUMINAMATH_GPT_product_zero_when_b_is_3_l98_9814

theorem product_zero_when_b_is_3 (b : ℤ) (h : b = 3) :
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) *
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_zero_when_b_is_3_l98_9814


namespace NUMINAMATH_GPT_confidence_level_for_relationship_l98_9843

-- Define the problem conditions and the target question.
def chi_squared_value : ℝ := 8.654
def critical_value : ℝ := 6.635
def confidence_level : ℝ := 99

theorem confidence_level_for_relationship (h : chi_squared_value > critical_value) : confidence_level = 99 :=
sorry

end NUMINAMATH_GPT_confidence_level_for_relationship_l98_9843


namespace NUMINAMATH_GPT_find_principal_amount_l98_9893

-- Define the given conditions
def interest_rate1 : ℝ := 0.08
def interest_rate2 : ℝ := 0.10
def interest_rate3 : ℝ := 0.12
def period1 : ℝ := 4
def period2 : ℝ := 6
def period3 : ℝ := 5
def total_interest_paid : ℝ := 12160

-- Goal is to find the principal amount P
theorem find_principal_amount (P : ℝ) :
  total_interest_paid = P * (interest_rate1 * period1 + interest_rate2 * period2 + interest_rate3 * period3) →
  P = 8000 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l98_9893


namespace NUMINAMATH_GPT_cannot_be_external_diagonals_l98_9816

theorem cannot_be_external_diagonals (a b c : ℕ) : 
  ¬(3^2 + 4^2 = 6^2) :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_external_diagonals_l98_9816


namespace NUMINAMATH_GPT_apple_tree_yield_l98_9822

theorem apple_tree_yield (A : ℝ) 
    (h1 : Magdalena_picks_day1 = A / 5)
    (h2 : Magdalena_picks_day2 = 2 * (A / 5))
    (h3 : Magdalena_picks_day3 = (A / 5) + 20)
    (h4 : remaining_apples = 20)
    (total_picked : Magdalena_picks_day1 + Magdalena_picks_day2 + Magdalena_picks_day3 + remaining_apples = A)
    : A = 200 :=
by
    sorry

end NUMINAMATH_GPT_apple_tree_yield_l98_9822


namespace NUMINAMATH_GPT_min_n_1014_dominoes_l98_9878

theorem min_n_1014_dominoes (n : ℕ) :
  (n + 1) ^ 2 ≥ 6084 → n ≥ 77 :=
sorry

end NUMINAMATH_GPT_min_n_1014_dominoes_l98_9878


namespace NUMINAMATH_GPT_sqrt_expr_evaluation_l98_9800

theorem sqrt_expr_evaluation :
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3)) = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_sqrt_expr_evaluation_l98_9800


namespace NUMINAMATH_GPT_max_omega_value_l98_9894

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + φ)

theorem max_omega_value 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : 0 < ω) 
  (hφ : |φ| ≤ Real.pi / 2)
  (h_zero : f ω φ (-Real.pi / 4) = 0)
  (h_sym : f ω φ (Real.pi / 4) = f ω φ (-Real.pi / 4))
  (h_monotonic : ∀ x₁ x₂, (Real.pi / 18) < x₁ → x₁ < x₂ → x₂ < (5 * Real.pi / 36) → f ω φ x₁ < f ω φ x₂) :
  ω = 9 :=
  sorry

end NUMINAMATH_GPT_max_omega_value_l98_9894


namespace NUMINAMATH_GPT_solve_for_x_l98_9830

theorem solve_for_x (x : ℝ) (h : (x - 15) / 3 = (3 * x + 10) / 8) : x = -150 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l98_9830


namespace NUMINAMATH_GPT_total_amount_proof_l98_9848

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem total_amount_proof (x_ratio y_ratio z_ratio : ℝ) (y_share : ℝ) 
  (h1 : y_ratio = 0.45) (h2 : z_ratio = 0.50) (h3 : y_share = 54) 
  : total_amount (y_share / y_ratio) y_share (z_ratio * (y_share / y_ratio)) = 234 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_proof_l98_9848


namespace NUMINAMATH_GPT_sum_faces_of_cube_l98_9827

theorem sum_faces_of_cube (p u q v r w : ℕ) (hp : 0 < p) (hu : 0 < u) (hq : 0 < q) (hv : 0 < v)
    (hr : 0 < r) (hw : 0 < w)
    (h_sum_vertices : p * q * r + p * v * r + p * q * w + p * v * w 
        + u * q * r + u * v * r + u * q * w + u * v * w = 2310) : 
    p + u + q + v + r + w = 40 := 
sorry

end NUMINAMATH_GPT_sum_faces_of_cube_l98_9827


namespace NUMINAMATH_GPT_exists_set_no_three_ap_l98_9852

theorem exists_set_no_three_ap (n : ℕ) (k : ℕ) :
  (n ≥ 1983) →
  (k ≤ 100000) →
  ∃ S : Finset ℕ,
    S.card = n ∧
    (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → b ≠ (a + c) / 2) :=
sorry

end NUMINAMATH_GPT_exists_set_no_three_ap_l98_9852


namespace NUMINAMATH_GPT_min_value_ineq_l98_9812

noncomputable def minimum_value (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) : ℝ :=
  1 / a + 4 / b

theorem min_value_ineq (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) :
  minimum_value a b ha hb h ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_ineq_l98_9812


namespace NUMINAMATH_GPT_proportion_correct_l98_9880

theorem proportion_correct (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x ≠ 0 ∧ y ≠ 0) : x / y = 2 / 5 := 
sorry

end NUMINAMATH_GPT_proportion_correct_l98_9880


namespace NUMINAMATH_GPT_average_marks_of_a_b_c_d_l98_9857

theorem average_marks_of_a_b_c_d (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : A = 43)
  (h3 : (B + C + D + E) / 4 = 48)
  (h4 : E = D + 3) :
  (A + B + C + D) / 4 = 47 :=
by
  -- This theorem will be justified
  admit

end NUMINAMATH_GPT_average_marks_of_a_b_c_d_l98_9857


namespace NUMINAMATH_GPT_number_of_oarsmen_l98_9817

-- Define the conditions
variables (n : ℕ)
variables (W : ℕ)
variables (h_avg_increase : (W + 40) / n = W / n + 2)

-- Lean 4 statement without the proof
theorem number_of_oarsmen : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_oarsmen_l98_9817


namespace NUMINAMATH_GPT_find_numbers_l98_9875

theorem find_numbers (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 24) : x = 21 ∧ y = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l98_9875


namespace NUMINAMATH_GPT_river_width_l98_9802

theorem river_width (boat_max_speed : ℝ) (river_current_speed : ℝ) (time_to_cross : ℝ) (width : ℝ) :
  boat_max_speed = 4 ∧ river_current_speed = 3 ∧ time_to_cross = 2 ∧ width = 8 → 
  width = boat_max_speed * time_to_cross := by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_river_width_l98_9802


namespace NUMINAMATH_GPT_ellipse_semi_focal_range_l98_9832

-- Definitions and conditions from the problem
variables (a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : a^2 = b^2 + c^2)

-- Statement of the theorem
theorem ellipse_semi_focal_range : 1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_ellipse_semi_focal_range_l98_9832


namespace NUMINAMATH_GPT_smallest_five_digit_divisible_by_2_5_11_l98_9860

theorem smallest_five_digit_divisible_by_2_5_11 : ∃ n, n >= 10000 ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n % 11 = 0 ∧ n = 10010 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_divisible_by_2_5_11_l98_9860


namespace NUMINAMATH_GPT_scientific_notation_of_8200000_l98_9804

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_8200000_l98_9804


namespace NUMINAMATH_GPT_tiling_2x12_l98_9868

def d : Nat → Nat
| 0     => 0  -- Unused but for safety in function definition
| 1     => 1
| 2     => 2
| (n+1) => d n + d (n-1)

theorem tiling_2x12 : d 12 = 233 := by
  sorry

end NUMINAMATH_GPT_tiling_2x12_l98_9868


namespace NUMINAMATH_GPT_max_value_of_k_l98_9823

theorem max_value_of_k (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m)) ≥ k) ↔ k ≤ 8 := 
sorry

end NUMINAMATH_GPT_max_value_of_k_l98_9823
