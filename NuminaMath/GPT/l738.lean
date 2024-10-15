import Mathlib

namespace NUMINAMATH_GPT_initial_students_count_l738_73815

variable (n T : ℕ)
variables (initial_average remaining_average dropped_score : ℚ)
variables (initial_students remaining_students : ℕ)

theorem initial_students_count :
  initial_average = 62.5 →
  remaining_average = 63 →
  dropped_score = 55 →
  T = initial_average * n →
  T - dropped_score = remaining_average * (n - 1) →
  n = 16 :=
by
  intros h_avg_initial h_avg_remaining h_dropped_score h_total h_total_remaining
  sorry

end NUMINAMATH_GPT_initial_students_count_l738_73815


namespace NUMINAMATH_GPT_evaluate_expression_l738_73811

theorem evaluate_expression :
  (18 : ℝ) / (14 * 5.3) = (1.8 : ℝ) / 7.42 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l738_73811


namespace NUMINAMATH_GPT_find_b_skew_lines_l738_73883

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3*t, 3 + 4*t, b + 5*t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6*u, 6 + 3*u, 1 + 2*u)

noncomputable def lines_are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem find_b_skew_lines (b : ℝ) : b ≠ -12 / 5 → lines_are_skew b :=
by
  sorry

end NUMINAMATH_GPT_find_b_skew_lines_l738_73883


namespace NUMINAMATH_GPT_linen_tablecloth_cost_l738_73841

def num_tables : ℕ := 20
def cost_per_place_setting : ℕ := 10
def num_place_settings_per_table : ℕ := 4
def cost_per_rose : ℕ := 5
def num_roses_per_centerpiece : ℕ := 10
def cost_per_lily : ℕ := 4
def num_lilies_per_centerpiece : ℕ := 15
def total_decoration_cost : ℕ := 3500

theorem linen_tablecloth_cost :
  (total_decoration_cost - (num_tables * num_place_settings_per_table * cost_per_place_setting + num_tables * (num_roses_per_centerpiece * cost_per_rose + num_lilies_per_centerpiece * cost_per_lily))) / num_tables = 25 :=
  sorry

end NUMINAMATH_GPT_linen_tablecloth_cost_l738_73841


namespace NUMINAMATH_GPT_coplanar_points_scalar_eq_l738_73816

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D O : V) (k : ℝ)

theorem coplanar_points_scalar_eq:
  (3 • (A - O) - 2 • (B - O) + 5 • (C - O) + k • (D - O) = (0 : V)) →
  k = -6 :=
by sorry

end NUMINAMATH_GPT_coplanar_points_scalar_eq_l738_73816


namespace NUMINAMATH_GPT_powerThreeExpression_l738_73889

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end NUMINAMATH_GPT_powerThreeExpression_l738_73889


namespace NUMINAMATH_GPT_cos_value_third_quadrant_l738_73844

theorem cos_value_third_quadrant (x : Real) (h1 : Real.sin x = -1 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_value_third_quadrant_l738_73844


namespace NUMINAMATH_GPT_geometric_sequence_sum_l738_73824

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h₀ : q > 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₂ : ∀ x : ℝ, 4 * x^2 - 8 * x + 3 = 0 → (x = a 2005 ∨ x = a 2006)) : 
  a 2007 + a 2008 = 18 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l738_73824


namespace NUMINAMATH_GPT_circle_radius_five_eq_neg_eight_l738_73852

theorem circle_radius_five_eq_neg_eight (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ∧ (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_five_eq_neg_eight_l738_73852


namespace NUMINAMATH_GPT_find_product_x_plus_1_x_minus_1_l738_73870

theorem find_product_x_plus_1_x_minus_1 (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x = 128) : (x + 1) * (x - 1) = 24 := sorry

end NUMINAMATH_GPT_find_product_x_plus_1_x_minus_1_l738_73870


namespace NUMINAMATH_GPT_double_meat_sandwich_bread_count_l738_73868

theorem double_meat_sandwich_bread_count (x : ℕ) :
  14 * 2 + 12 * x = 64 → x = 3 := by
  intro h
  sorry

end NUMINAMATH_GPT_double_meat_sandwich_bread_count_l738_73868


namespace NUMINAMATH_GPT_rex_cards_left_l738_73808

-- Definitions
def nicole_cards : ℕ := 400
def cindy_cards : ℕ := 2 * nicole_cards
def combined_total : ℕ := nicole_cards + cindy_cards
def rex_cards : ℕ := combined_total / 2
def people_count : ℕ := 4
def cards_per_person : ℕ := rex_cards / people_count

-- Proof statement
theorem rex_cards_left : cards_per_person = 150 := by
  sorry

end NUMINAMATH_GPT_rex_cards_left_l738_73808


namespace NUMINAMATH_GPT_nested_sqrt_eq_five_l738_73820

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end NUMINAMATH_GPT_nested_sqrt_eq_five_l738_73820


namespace NUMINAMATH_GPT_combined_resistance_l738_73893

theorem combined_resistance (x y r : ℝ) (hx : x = 5) (hy : y = 7) (h_parallel : 1 / r = 1 / x + 1 / y) : 
  r = 35 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_combined_resistance_l738_73893


namespace NUMINAMATH_GPT_not_taking_ship_probability_l738_73882

-- Real non-negative numbers as probabilities
variables (P_train P_ship P_car P_airplane : ℝ)

-- Conditions
axiom h_train : 0 ≤ P_train ∧ P_train ≤ 1 ∧ P_train = 0.3
axiom h_ship : 0 ≤ P_ship ∧ P_ship ≤ 1 ∧ P_ship = 0.1
axiom h_car : 0 ≤ P_car ∧ P_car ≤ 1 ∧ P_car = 0.4
axiom h_airplane : 0 ≤ P_airplane ∧ P_airplane ≤ 1 ∧ P_airplane = 0.2

-- Prove that the probability of not taking a ship is 0.9
theorem not_taking_ship_probability : 1 - P_ship = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_not_taking_ship_probability_l738_73882


namespace NUMINAMATH_GPT_initial_average_mark_l738_73897

-- Define the conditions
def total_students := 13
def average_mark := 72
def excluded_students := 5
def excluded_students_average := 40
def remaining_students := total_students - excluded_students
def remaining_students_average := 92

-- Define the total marks calculations
def initial_total_marks (A : ℕ) : ℕ := total_students * A
def excluded_total_marks : ℕ := excluded_students * excluded_students_average
def remaining_total_marks : ℕ := remaining_students * remaining_students_average

-- Prove the initial average mark
theorem initial_average_mark : 
  initial_total_marks average_mark = excluded_total_marks + remaining_total_marks →
  average_mark = 72 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_mark_l738_73897


namespace NUMINAMATH_GPT_factor_polynomial_l738_73873

theorem factor_polynomial (a b c : ℝ) : 
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) = (a - b) * (b - c) * (c - a) * (a * b + a * c + b * c) :=
by 
  sorry

end NUMINAMATH_GPT_factor_polynomial_l738_73873


namespace NUMINAMATH_GPT_max_distance_from_circle_to_line_l738_73890

theorem max_distance_from_circle_to_line :
  ∀ (P : ℝ × ℝ), (P.1 - 1)^2 + P.2^2 = 9 →
  ∀ (x y : ℝ), 5 * x + 12 * y + 8 = 0 →
  ∃ (d : ℝ), d = 4 :=
by
  -- Proof is omitted as instructed.
  sorry

end NUMINAMATH_GPT_max_distance_from_circle_to_line_l738_73890


namespace NUMINAMATH_GPT_distance_to_river_l738_73847

theorem distance_to_river (d : ℝ) (h1 : ¬ (d ≥ 8)) (h2 : ¬ (d ≤ 7)) (h3 : ¬ (d ≤ 6)) : 7 < d ∧ d < 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_river_l738_73847


namespace NUMINAMATH_GPT_b_alone_days_l738_73876

theorem b_alone_days {a b : ℝ} (h1 : a + b = 1/6) (h2 : a = 1/11) : b = 1/(66/5) :=
by sorry

end NUMINAMATH_GPT_b_alone_days_l738_73876


namespace NUMINAMATH_GPT_ice_cream_tubs_eaten_l738_73810

-- Conditions
def number_of_pans := 2
def pieces_per_pan := 16
def percentage_eaten_second_pan := 0.75
def scoops_per_tub := 8
def scoops_per_guest := 2
def guests_not_eating_ala_mode := 4

-- Questions
def tubs_of_ice_cream_eaten : Nat :=
  sorry

theorem ice_cream_tubs_eaten :
  tubs_of_ice_cream_eaten = 6 := by
  sorry

end NUMINAMATH_GPT_ice_cream_tubs_eaten_l738_73810


namespace NUMINAMATH_GPT_consecutive_integers_sum_l738_73853

theorem consecutive_integers_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < Real.sqrt 17) (h4 : Real.sqrt 17 < b) : a + b = 9 :=
sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l738_73853


namespace NUMINAMATH_GPT_option_D_correct_l738_73861

noncomputable def y1 (x : ℝ) : ℝ := 1 / x
noncomputable def y2 (x : ℝ) : ℝ := x^2
noncomputable def y3 (x : ℝ) : ℝ := (1 / 2)^x
noncomputable def y4 (x : ℝ) : ℝ := 1 / x^2

theorem option_D_correct :
  (∀ x : ℝ, y4 x = y4 (-x)) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → y4 x₁ > y4 x₂) :=
by
  sorry

end NUMINAMATH_GPT_option_D_correct_l738_73861


namespace NUMINAMATH_GPT_profit_per_box_type_A_and_B_maximize_profit_l738_73877

-- Condition definitions
def total_boxes : ℕ := 600
def profit_type_A : ℕ := 40000
def profit_type_B : ℕ := 160000
def profit_difference : ℕ := 200

-- Question 1: Proving the profit per box for type A and B
theorem profit_per_box_type_A_and_B (x : ℝ) :
  (profit_type_A / x + profit_type_B / (x + profit_difference) = total_boxes)
  → (x = 200) ∧ (x + profit_difference = 400) :=
sorry

-- Condition definitions for question 2
def price_reduction_per_box_A (a : ℕ) : ℕ := 5 * a
def price_increase_per_box_B (a : ℕ) : ℕ := 5 * a

-- Initial number of boxes sold for type A and B
def initial_boxes_sold_A : ℕ := 200
def initial_boxes_sold_B : ℕ := 400

-- General profit function
def profit (a : ℕ) : ℝ :=
  (initial_boxes_sold_A + 2 * a) * (200 - price_reduction_per_box_A a) +
  (initial_boxes_sold_B - 2 * a) * (400 + price_increase_per_box_B a)

-- Question 2: Proving the price reduction and maximum profit
theorem maximize_profit (a : ℕ) :
  ((price_reduction_per_box_A a = 75) ∧ (profit a = 204500)) :=
sorry

end NUMINAMATH_GPT_profit_per_box_type_A_and_B_maximize_profit_l738_73877


namespace NUMINAMATH_GPT_exists_large_absolute_value_solutions_l738_73869

theorem exists_large_absolute_value_solutions : 
  ∃ (x1 x2 y1 y2 y3 y4 : ℤ), 
    x1 + x2 = y1 + y2 + y3 + y4 ∧ 
    x1^2 + x2^2 = y1^2 + y2^2 + y3^2 + y4^2 ∧ 
    x1^3 + x2^3 = y1^3 + y2^3 + y3^3 + y4^3 ∧ 
    abs x1 > 2020 ∧ abs x2 > 2020 ∧ abs y1 > 2020 ∧ abs y2 > 2020 ∧ abs y3 > 2020 ∧ abs y4 > 2020 :=
  by
  sorry

end NUMINAMATH_GPT_exists_large_absolute_value_solutions_l738_73869


namespace NUMINAMATH_GPT_sum_of_squares_l738_73804

open Int

theorem sum_of_squares (p q r s t u : ℤ) (h : ∀ x : ℤ, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l738_73804


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l738_73835

theorem area_of_triangle_ABC
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_C_eq : Real.sin C = Real.sqrt 3 / 3)
  (sin_CBA_eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A))
  (a_minus_b_eq : a - b = 3 - Real.sqrt 6)
  (c_eq : c = Real.sqrt 3) :
  1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 2 / 2 := sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l738_73835


namespace NUMINAMATH_GPT_restaurant_total_cost_l738_73888

def total_cost
  (adults kids : ℕ)
  (adult_meal_cost adult_drink_cost adult_dessert_cost kid_drink_cost kid_dessert_cost : ℝ) : ℝ :=
  let num_adults := adults
  let num_kids := kids
  let adult_total := num_adults * (adult_meal_cost + adult_drink_cost + adult_dessert_cost)
  let kid_total := num_kids * (kid_drink_cost + kid_dessert_cost)
  adult_total + kid_total

theorem restaurant_total_cost :
  total_cost 4 9 7 4 3 2 1.5 = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_total_cost_l738_73888


namespace NUMINAMATH_GPT_candy_cost_proof_l738_73857

theorem candy_cost_proof (x : ℝ) (h1 : 10 ≤ 30) (h2 : 0 ≤ 5) (h3 : 0 ≤ 6) 
(h4 : 10 * x + 20 * 5 = 6 * 30) : x = 8 := by
  sorry

end NUMINAMATH_GPT_candy_cost_proof_l738_73857


namespace NUMINAMATH_GPT_complex_z_pow_l738_73879

open Complex

theorem complex_z_pow {z : ℂ} (h : (1 + z) / (1 - z) = (⟨0, 1⟩ : ℂ)) : z ^ 2019 = -⟨0, 1⟩ := by
  sorry

end NUMINAMATH_GPT_complex_z_pow_l738_73879


namespace NUMINAMATH_GPT_polar_to_rect_l738_73848

open Real 

theorem polar_to_rect (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 3 * π / 4) : 
  (r * cos θ, r * sin θ) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) :=
by
  -- Optional step: you can introduce the variables as they have already been proved using the given conditions
  have hr : r = 3 := h_r
  have hθ : θ = 3 * π / 4 := h_θ
  -- Goal changes according to the values of r and θ derived from the conditions
  sorry

end NUMINAMATH_GPT_polar_to_rect_l738_73848


namespace NUMINAMATH_GPT_pizza_cost_l738_73846

theorem pizza_cost
  (P T : ℕ)
  (hT : T = 1)
  (h_total : 3 * P + 4 * T + 5 = 39) :
  P = 10 :=
by
  sorry

end NUMINAMATH_GPT_pizza_cost_l738_73846


namespace NUMINAMATH_GPT_ratio_of_ages_l738_73802

theorem ratio_of_ages (D R : ℕ) (h1 : D = 3) (h2 : R + 22 = 26) : R / D = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l738_73802


namespace NUMINAMATH_GPT_combined_weight_difference_l738_73828

-- Define the weights of the textbooks
def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := 5.25
def biology_weight : ℝ := 3.75

-- Define the problem statement that needs to be proven
theorem combined_weight_difference :
  ((calculus_weight + biology_weight) - (chemistry_weight - geometry_weight)) = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_difference_l738_73828


namespace NUMINAMATH_GPT_max_value_S_n_l738_73839

open Nat

noncomputable def a_n (n : ℕ) : ℤ := 20 + (n - 1) * (-2)

noncomputable def S_n (n : ℕ) : ℤ := n * 20 + (n * (n - 1)) * (-2) / 2

theorem max_value_S_n : ∃ n : ℕ, S_n n = 110 :=
by
  sorry

end NUMINAMATH_GPT_max_value_S_n_l738_73839


namespace NUMINAMATH_GPT_system_has_three_real_k_with_unique_solution_l738_73851

theorem system_has_three_real_k_with_unique_solution :
  (∃ (k : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) → (x, y) = (0, 0)) → 
  ∃ (k : ℝ), ∃ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) :=
by
  sorry

end NUMINAMATH_GPT_system_has_three_real_k_with_unique_solution_l738_73851


namespace NUMINAMATH_GPT_umbrellas_problem_l738_73840

theorem umbrellas_problem :
  ∃ (b r : ℕ), b = 36 ∧ r = 27 ∧ 
  b = (45 + r) / 2 ∧ 
  r = (45 + b) / 3 :=
by sorry

end NUMINAMATH_GPT_umbrellas_problem_l738_73840


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l738_73837

variable {a_n : ℕ → ℤ} {S_n : ℕ → ℤ}
variable (d : ℤ)
variable (a1 a3 a4 : ℤ)
variable (h_geom : a3^2 = a1 * a4)
variable (h_seq : ∀ n, a_n (n+1) = a_n n + d)
variable (h_sum : ∀ n, S_n n = (n * (2 * a1 + (n - 1) * d)) / 2)

theorem arithmetic_sequence_ratio :
  (S_n 3 - S_n 2) / (S_n 5 - S_n 3) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l738_73837


namespace NUMINAMATH_GPT_fence_length_l738_73891

theorem fence_length {w l : ℕ} (h1 : l = 2 * w) (h2 : 30 = 2 * l + 2 * w) : l = 10 := by
  sorry

end NUMINAMATH_GPT_fence_length_l738_73891


namespace NUMINAMATH_GPT_calculation_l738_73842

noncomputable def seq (n : ℕ) : ℕ → ℚ := sorry

axiom cond1 : ∀ (n : ℕ), seq (n + 1) - 2 * seq n = 0
axiom cond2 : ∀ (n : ℕ), seq n ≠ 0

theorem calculation :
  (2 * seq 1 + seq 2) / (seq 3 + seq 5) = 1 / 5 :=
  sorry

end NUMINAMATH_GPT_calculation_l738_73842


namespace NUMINAMATH_GPT_evaluate_fractions_l738_73864

theorem evaluate_fractions (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 35 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fractions_l738_73864


namespace NUMINAMATH_GPT_initial_time_between_maintenance_checks_l738_73817

theorem initial_time_between_maintenance_checks (x : ℝ) (h1 : 1.20 * x = 30) : x = 25 := by
  sorry

end NUMINAMATH_GPT_initial_time_between_maintenance_checks_l738_73817


namespace NUMINAMATH_GPT_degree_of_p_x2_q_x4_l738_73874

-- Definitions to capture the given problem conditions
def is_degree_3 (p : Polynomial ℝ) : Prop := p.degree = 3
def is_degree_6 (q : Polynomial ℝ) : Prop := q.degree = 6

-- Statement of the proof problem
theorem degree_of_p_x2_q_x4 (p q : Polynomial ℝ) (hp : is_degree_3 p) (hq : is_degree_6 q) :
  (p.comp (Polynomial.X ^ 2) * q.comp (Polynomial.X ^ 4)).degree = 30 :=
sorry

end NUMINAMATH_GPT_degree_of_p_x2_q_x4_l738_73874


namespace NUMINAMATH_GPT_all_tutors_work_together_in_90_days_l738_73823

theorem all_tutors_work_together_in_90_days :
  lcm 5 (lcm 6 (lcm 9 10)) = 90 := by
  sorry

end NUMINAMATH_GPT_all_tutors_work_together_in_90_days_l738_73823


namespace NUMINAMATH_GPT_analogical_reasoning_correctness_l738_73886

theorem analogical_reasoning_correctness 
  (a b c : ℝ)
  (va vb vc : ℝ) :
  (a + b) * c = (a * c + b * c) ↔ 
  (va + vb) * vc = (va * vc + vb * vc) := 
sorry

end NUMINAMATH_GPT_analogical_reasoning_correctness_l738_73886


namespace NUMINAMATH_GPT_frustum_volume_correct_l738_73887

-- Define the base edge of the original pyramid
def base_edge_pyramid := 16

-- Define the height (altitude) of the original pyramid
def height_pyramid := 10

-- Define the base edge of the smaller pyramid after the cut
def base_edge_smaller_pyramid := 8

-- Define the function to calculate the volume of a square pyramid
def volume_square_pyramid (base_edge : ℕ) (height : ℕ) : ℚ :=
  (1 / 3) * (base_edge ^ 2) * height

-- Calculate the volume of the original pyramid
def V := volume_square_pyramid base_edge_pyramid height_pyramid

-- Calculate the volume of the smaller pyramid
def V_small := volume_square_pyramid base_edge_smaller_pyramid (height_pyramid / 2)

-- Calculate the volume of the frustum
def V_frustum := V - V_small

-- Prove that the volume of the frustum is 213.33 cubic centimeters
theorem frustum_volume_correct : V_frustum = 213.33 := by
  sorry

end NUMINAMATH_GPT_frustum_volume_correct_l738_73887


namespace NUMINAMATH_GPT_find_a_l738_73866

theorem find_a (a x : ℝ) (h1 : 2 * (x - 1) - 6 = 0) (h2 : 1 - (3 * a - x) / 3 = 0) (h3 : x = 4) : a = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l738_73866


namespace NUMINAMATH_GPT_solve_for_product_l738_73832

theorem solve_for_product (a b c d : ℚ) (h1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
                          (h2 : 4 * (d + c) = b) 
                          (h3 : 4 * b + 2 * c = a) 
                          (h4 : c - 2 = d) : 
                          a * b * c * d = -1032192 / 1874161 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_product_l738_73832


namespace NUMINAMATH_GPT_vector_subtraction_l738_73859

open Real

def vector_a : (ℝ × ℝ) := (3, 2)
def vector_b : (ℝ × ℝ) := (0, -1)

theorem vector_subtraction : 
  3 • vector_b - vector_a = (-3, -5) :=
by 
  -- Proof needs to be written here.
  sorry

end NUMINAMATH_GPT_vector_subtraction_l738_73859


namespace NUMINAMATH_GPT_twenty_four_point_solution_l738_73898

theorem twenty_four_point_solution : (5 - (1 / 5)) * 5 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_twenty_four_point_solution_l738_73898


namespace NUMINAMATH_GPT_find_y_square_divisible_by_three_between_50_and_120_l738_73826

theorem find_y_square_divisible_by_three_between_50_and_120 :
  ∃ (y : ℕ), y = 81 ∧ (∃ (n : ℕ), y = n^2) ∧ (3 ∣ y) ∧ (50 < y) ∧ (y < 120) :=
by
  sorry

end NUMINAMATH_GPT_find_y_square_divisible_by_three_between_50_and_120_l738_73826


namespace NUMINAMATH_GPT_total_games_attended_l738_73862

def games_in_months (this_month previous_month next_month following_month fifth_month : ℕ) : ℕ :=
  this_month + previous_month + next_month + following_month + fifth_month

theorem total_games_attended :
  games_in_months 24 32 29 19 34 = 138 :=
by
  -- Proof will be provided, but ignored for this problem
  sorry

end NUMINAMATH_GPT_total_games_attended_l738_73862


namespace NUMINAMATH_GPT_min_max_values_f_l738_73825

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end NUMINAMATH_GPT_min_max_values_f_l738_73825


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l738_73894

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 2 * x - 4 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 6 = x * (3 - x)

-- State the first proof problem
theorem solve_equation1 (x : ℝ) :
  equation1 x ↔ (x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) := by
  sorry

-- State the second proof problem
theorem solve_equation2 (x : ℝ) :
  equation2 x ↔ (x = 3 ∨ x = -2) := by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l738_73894


namespace NUMINAMATH_GPT_diagonal_of_rectangular_prism_l738_73860

theorem diagonal_of_rectangular_prism
  (width height depth : ℕ)
  (h1 : width = 15)
  (h2 : height = 20)
  (h3 : depth = 25) : 
  (width ^ 2 + height ^ 2 + depth ^ 2).sqrt = 25 * (2 : ℕ).sqrt :=
by {
  sorry
}

end NUMINAMATH_GPT_diagonal_of_rectangular_prism_l738_73860


namespace NUMINAMATH_GPT_trains_meet_80_km_from_A_l738_73865

-- Define the speeds of the trains
def speed_train_A : ℝ := 60 
def speed_train_B : ℝ := 90 

-- Define the distance between locations A and B
def distance_AB : ℝ := 200 

-- Define the time when the trains meet
noncomputable def meeting_time : ℝ := distance_AB / (speed_train_A + speed_train_B)

-- Define the distance from location A to where the trains meet
noncomputable def distance_from_A (speed_A : ℝ) (meeting_time : ℝ) : ℝ :=
  speed_A * meeting_time

-- Prove the statement
theorem trains_meet_80_km_from_A :
  distance_from_A speed_train_A meeting_time = 80 :=
by
  -- leaving the proof out, it's just an assumption due to 'sorry'
  sorry

end NUMINAMATH_GPT_trains_meet_80_km_from_A_l738_73865


namespace NUMINAMATH_GPT_unique_solution_inequality_l738_73892

theorem unique_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, -3 ≤ x^2 - 2 * a * x + a ∧ x^2 - 2 * a * x + a ≤ -2 → ∃! x : ℝ, x^2 - 2 * a * x + a = -2) ↔ (a = 2 ∨ a = -1) :=
sorry

end NUMINAMATH_GPT_unique_solution_inequality_l738_73892


namespace NUMINAMATH_GPT_number_of_toys_sold_l738_73801

theorem number_of_toys_sold (n : ℕ) 
  (sell_price : ℕ) (gain_price : ℕ) (cost_price_per_toy : ℕ) :
  sell_price = 27300 → 
  gain_price = 3 * cost_price_per_toy → 
  cost_price_per_toy = 1300 →
  n * cost_price_per_toy + gain_price = sell_price → 
  n = 18 :=
by sorry

end NUMINAMATH_GPT_number_of_toys_sold_l738_73801


namespace NUMINAMATH_GPT_sqrt_operation_l738_73836

def operation (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt_operation (sqrt5 : ℝ) (h : sqrt5 = Real.sqrt 5) : 
  operation sqrt5 sqrt5 = 20 := by
  sorry

end NUMINAMATH_GPT_sqrt_operation_l738_73836


namespace NUMINAMATH_GPT_difference_of_scores_l738_73895

variable {x y : ℝ}

theorem difference_of_scores (h : x / y = 4) : x - y = 3 * y := by
  sorry

end NUMINAMATH_GPT_difference_of_scores_l738_73895


namespace NUMINAMATH_GPT_books_sold_l738_73818

theorem books_sold {total_books sold_fraction left_fraction : ℕ} (h_total : total_books = 9900)
    (h_fraction : left_fraction = 4/6) (h_sold : sold_fraction = 1 - left_fraction) : 
  (sold_fraction * total_books) = 3300 := 
  by 
  sorry

end NUMINAMATH_GPT_books_sold_l738_73818


namespace NUMINAMATH_GPT_decrease_percent_revenue_l738_73867

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.05 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 16 := 
by
  sorry

end NUMINAMATH_GPT_decrease_percent_revenue_l738_73867


namespace NUMINAMATH_GPT_speedster_convertibles_l738_73875

noncomputable def total_inventory (not_speedsters : Nat) (fraction_not_speedsters : ℝ) : ℝ :=
  (not_speedsters : ℝ) / fraction_not_speedsters

noncomputable def number_speedsters (total_inventory : ℝ) (fraction_speedsters : ℝ) : ℝ :=
  total_inventory * fraction_speedsters

noncomputable def number_convertibles (number_speedsters : ℝ) (fraction_convertibles : ℝ) : ℝ :=
  number_speedsters * fraction_convertibles

theorem speedster_convertibles : (not_speedsters = 30) ∧ (fraction_not_speedsters = 2 / 3) ∧ (fraction_speedsters = 1 / 3) ∧ (fraction_convertibles = 4 / 5) →
  number_convertibles (number_speedsters (total_inventory not_speedsters fraction_not_speedsters) fraction_speedsters) fraction_convertibles = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_speedster_convertibles_l738_73875


namespace NUMINAMATH_GPT_solve_abs_equation_l738_73807

theorem solve_abs_equation (x : ℝ) (h : abs (x - 20) + abs (x - 18) = abs (2 * x - 36)) : x = 19 :=
sorry

end NUMINAMATH_GPT_solve_abs_equation_l738_73807


namespace NUMINAMATH_GPT_euler_characteristic_convex_polyhedron_l738_73833

-- Define the context of convex polyhedron with vertices (V), edges (E), and faces (F)
structure ConvexPolyhedron :=
  (V : ℕ) -- number of vertices
  (E : ℕ) -- number of edges
  (F : ℕ) -- number of faces
  (convex : Prop) -- property stating the polyhedron is convex

-- Euler characteristic theorem for convex polyhedra
theorem euler_characteristic_convex_polyhedron (P : ConvexPolyhedron) (h : P.convex) : P.V - P.E + P.F = 2 :=
sorry

end NUMINAMATH_GPT_euler_characteristic_convex_polyhedron_l738_73833


namespace NUMINAMATH_GPT_find_triplets_l738_73822

theorem find_triplets (x y z : ℕ) (h1 : x ≤ y) (h2 : x^2 + y^2 = 3 * 2016^z + 77) :
  (x, y, z) = (4, 8, 0) ∨ (x, y, z) = (14, 77, 1) ∨ (x, y, z) = (35, 70, 1) :=
  sorry

end NUMINAMATH_GPT_find_triplets_l738_73822


namespace NUMINAMATH_GPT_find_polynomial_l738_73827

theorem find_polynomial (P : ℝ → ℝ) (h_poly : ∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) :
  ∃ r s : ℝ, ∀ x : ℝ, P x = r * x^4 + s * x^2 :=
sorry

end NUMINAMATH_GPT_find_polynomial_l738_73827


namespace NUMINAMATH_GPT_galya_number_l738_73830

theorem galya_number (N k : ℤ) (h : (k - N + 1 = k - 7729)) : N = 7730 := 
by
  sorry

end NUMINAMATH_GPT_galya_number_l738_73830


namespace NUMINAMATH_GPT_max_value_of_f_l738_73872

noncomputable def f (x : ℝ) : ℝ :=
  2022 * x ^ 2 * Real.log (x + 2022) / ((Real.log (x + 2022)) ^ 3 + 2 * x ^ 3)

theorem max_value_of_f : ∃ x : ℝ, 0 < x ∧ f x ≤ 674 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l738_73872


namespace NUMINAMATH_GPT_ivans_profit_l738_73838

def price_meat_per_kg : ℕ := 500
def kg_meat_sold : ℕ := 100
def price_eggs_per_dozen : ℕ := 50
def eggs_sold : ℕ := 20000
def annual_expenses : ℕ := 100000

def revenue_meat : ℕ := kg_meat_sold * price_meat_per_kg
def revenue_eggs : ℕ := eggs_sold * (price_eggs_per_dozen / 10)
def total_revenue : ℕ := revenue_meat + revenue_eggs

def profit : ℕ := total_revenue - annual_expenses

theorem ivans_profit : profit = 50000 := by
  sorry

end NUMINAMATH_GPT_ivans_profit_l738_73838


namespace NUMINAMATH_GPT_value_of_f_neg1_l738_73834

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_f_neg1_l738_73834


namespace NUMINAMATH_GPT_tom_tickets_l738_73821

theorem tom_tickets :
  let tickets_whack_a_mole := 32
  let tickets_skee_ball := 25
  let tickets_spent_on_hat := 7
  let total_tickets := tickets_whack_a_mole + tickets_skee_ball
  let tickets_left := total_tickets - tickets_spent_on_hat
  tickets_left = 50 :=
by
  sorry

end NUMINAMATH_GPT_tom_tickets_l738_73821


namespace NUMINAMATH_GPT_sandy_total_puppies_l738_73899

-- Definitions based on conditions:
def original_puppies : ℝ := 8.0
def additional_puppies : ℝ := 4.0

-- Theorem statement: total_puppies should be 12.0
theorem sandy_total_puppies : original_puppies + additional_puppies = 12.0 := 
by
  sorry

end NUMINAMATH_GPT_sandy_total_puppies_l738_73899


namespace NUMINAMATH_GPT_inverse_89_mod_90_l738_73831

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end NUMINAMATH_GPT_inverse_89_mod_90_l738_73831


namespace NUMINAMATH_GPT_warehouse_bins_total_l738_73806

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end NUMINAMATH_GPT_warehouse_bins_total_l738_73806


namespace NUMINAMATH_GPT_find_obtuse_angle_l738_73845

-- Define the conditions
def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180

-- Lean statement assuming the needed conditions
theorem find_obtuse_angle (α : ℝ) (h1 : is_obtuse α) (h2 : 4 * α = 360 + α) : α = 120 :=
by sorry

end NUMINAMATH_GPT_find_obtuse_angle_l738_73845


namespace NUMINAMATH_GPT_smallest_a_l738_73858

theorem smallest_a (x a : ℝ) (hx : x > 0) (ha : a > 0) (hineq : x + a / x ≥ 4) : a ≥ 4 :=
sorry

end NUMINAMATH_GPT_smallest_a_l738_73858


namespace NUMINAMATH_GPT_rashmi_late_time_is_10_l738_73863

open Real

noncomputable def rashmi_late_time : ℝ :=
  let d : ℝ := 9.999999999999993
  let v1 : ℝ := 5 / 60 -- km per minute
  let v2 : ℝ := 6 / 60 -- km per minute
  let time1 := d / v1 -- time taken at 5 kmph
  let time2 := d / v2 -- time taken at 6 kmph
  let difference := time1 - time2
  let T := difference / 2 -- The time she was late or early
  T

theorem rashmi_late_time_is_10 : rashmi_late_time = 10 := by
  simp [rashmi_late_time]
  sorry

end NUMINAMATH_GPT_rashmi_late_time_is_10_l738_73863


namespace NUMINAMATH_GPT_jake_fewer_peaches_undetermined_l738_73885

theorem jake_fewer_peaches_undetermined 
    (steven_peaches : ℕ) 
    (steven_apples : ℕ) 
    (jake_fewer_peaches : steven_peaches > jake_peaches) 
    (jake_more_apples : jake_apples = steven_apples + 3) 
    (steven_peaches_val : steven_peaches = 9) 
    (steven_apples_val : steven_apples = 8) : 
    ∃ n : ℕ, jake_peaches = n ∧ ¬(∃ m : ℕ, steven_peaches - jake_peaches = m) := 
sorry

end NUMINAMATH_GPT_jake_fewer_peaches_undetermined_l738_73885


namespace NUMINAMATH_GPT_parallel_lines_a_unique_l738_73813

theorem parallel_lines_a_unique (a : ℝ) :
  (∀ x y : ℝ, x + (a + 1) * y + (a^2 - 1) = 0 → x + 2 * y = 0 → -a / 2 = -1 / (a + 1)) →
  a = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_unique_l738_73813


namespace NUMINAMATH_GPT_percentage_of_local_arts_students_is_50_l738_73805

-- Definitions
def total_students_arts := 400
def total_students_science := 100
def total_students_commerce := 120
def percent_local_science := 25 / 100
def percent_local_commerce := 85 / 100
def total_locals := 327

-- Problem statement in Lean
theorem percentage_of_local_arts_students_is_50
  (x : ℕ) -- Percentage of local arts students as a natural number
  (h1 : percent_local_science * total_students_science = 25)
  (h2 : percent_local_commerce * total_students_commerce = 102)
  (h3 : (x / 100 : ℝ) * total_students_arts + 25 + 102 = total_locals) :
  x = 50 :=
sorry

end NUMINAMATH_GPT_percentage_of_local_arts_students_is_50_l738_73805


namespace NUMINAMATH_GPT_rational_number_25_units_away_l738_73800

theorem rational_number_25_units_away (x : ℚ) (h : |x| = 2.5) : x = 2.5 ∨ x = -2.5 := 
by
  sorry

end NUMINAMATH_GPT_rational_number_25_units_away_l738_73800


namespace NUMINAMATH_GPT_largest_y_coordinate_l738_73849

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  intro h
  -- This is where the proofs steps would go if required.
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_l738_73849


namespace NUMINAMATH_GPT_price_reduction_l738_73829

variable (x : ℝ)

theorem price_reduction :
  28 * (1 - x) * (1 - x) = 16 :=
sorry

end NUMINAMATH_GPT_price_reduction_l738_73829


namespace NUMINAMATH_GPT_distance_between_trees_l738_73812

theorem distance_between_trees (length_yard : ℕ) (num_trees : ℕ) (dist : ℕ) :
  length_yard = 275 → num_trees = 26 → dist = length_yard / (num_trees - 1) → dist = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  assumption

end NUMINAMATH_GPT_distance_between_trees_l738_73812


namespace NUMINAMATH_GPT_length_of_larger_sheet_l738_73880

theorem length_of_larger_sheet : 
  ∃ L : ℝ, 2 * (L * 11) = 2 * (5.5 * 11) + 100 ∧ L = 10 :=
by
  sorry

end NUMINAMATH_GPT_length_of_larger_sheet_l738_73880


namespace NUMINAMATH_GPT_days_worked_per_week_l738_73871

theorem days_worked_per_week
  (hourly_wage : ℕ) (hours_per_day : ℕ) (total_earnings : ℕ) (weeks : ℕ)
  (H_wage : hourly_wage = 12) (H_hours : hours_per_day = 9) (H_earnings : total_earnings = 3780) (H_weeks : weeks = 7) :
  (total_earnings / weeks) / (hourly_wage * hours_per_day) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_days_worked_per_week_l738_73871


namespace NUMINAMATH_GPT_corn_purchase_l738_73855

theorem corn_purchase : ∃ c b : ℝ, c + b = 30 ∧ 89 * c + 55 * b = 2170 ∧ c = 15.3 := 
by
  sorry

end NUMINAMATH_GPT_corn_purchase_l738_73855


namespace NUMINAMATH_GPT_jack_afternoon_emails_l738_73843

theorem jack_afternoon_emails : 
  ∀ (morning_emails afternoon_emails : ℕ), 
  morning_emails = 6 → 
  afternoon_emails = morning_emails + 2 → 
  afternoon_emails = 8 := 
by
  intros morning_emails afternoon_emails hm ha
  rw [hm] at ha
  exact ha

end NUMINAMATH_GPT_jack_afternoon_emails_l738_73843


namespace NUMINAMATH_GPT_sum_of_squares_of_medians_l738_73819

-- Define the components of the triangle
variables (a b c : ℝ)

-- Define the medians of the triangle
variables (s_a s_b s_c : ℝ)

-- State the theorem
theorem sum_of_squares_of_medians (h1 : s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2)) : 
  s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2) :=
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_sum_of_squares_of_medians_l738_73819


namespace NUMINAMATH_GPT_students_taking_neither_l738_73856

theorem students_taking_neither (total students_cs students_electronics students_both : ℕ)
  (h1 : total = 60) (h2 : students_cs = 42) (h3 : students_electronics = 35) (h4 : students_both = 25) :
  total - (students_cs - students_both + students_electronics - students_both + students_both) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_taking_neither_l738_73856


namespace NUMINAMATH_GPT_five_fourths_of_fifteen_fourths_l738_73854

theorem five_fourths_of_fifteen_fourths :
  (5 / 4) * (15 / 4) = 75 / 16 := by
  sorry

end NUMINAMATH_GPT_five_fourths_of_fifteen_fourths_l738_73854


namespace NUMINAMATH_GPT_simplify_T_l738_73881

variable (x : ℝ)

theorem simplify_T :
  9 * (x + 2)^2 - 12 * (x + 2) + 4 = 4 * (1.5 * x + 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_T_l738_73881


namespace NUMINAMATH_GPT_train_speed_l738_73803

theorem train_speed (length : ℤ) (time : ℤ) 
  (h_length : length = 280) (h_time : time = 14) : 
  (length * 3600) / (time * 1000) = 72 := 
by {
  -- The proof would go here, this part is omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_train_speed_l738_73803


namespace NUMINAMATH_GPT_range_of_f_is_pi_div_four_l738_73814

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end NUMINAMATH_GPT_range_of_f_is_pi_div_four_l738_73814


namespace NUMINAMATH_GPT_problem_l738_73896

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem (a : ℝ) (x : ℝ) (hx : x ∈ Set.Ici (-5)) (ha : a = 1) : 
  f x a + x + 5 ≥ -6 / Real.exp 5 := 
sorry

end NUMINAMATH_GPT_problem_l738_73896


namespace NUMINAMATH_GPT_kiran_has_105_l738_73850

theorem kiran_has_105 
  (R G K L : ℕ) 
  (ratio_rg : 6 * G = 7 * R)
  (ratio_gk : 6 * K = 15 * G)
  (R_value : R = 36) : 
  K = 105 :=
by
  sorry

end NUMINAMATH_GPT_kiran_has_105_l738_73850


namespace NUMINAMATH_GPT_fraction_multiplication_l738_73884

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end NUMINAMATH_GPT_fraction_multiplication_l738_73884


namespace NUMINAMATH_GPT_rationalize_denominator_l738_73809

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l738_73809


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l738_73878

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l738_73878
