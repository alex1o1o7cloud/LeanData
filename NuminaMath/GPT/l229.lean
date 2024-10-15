import Mathlib

namespace NUMINAMATH_GPT_germination_percentage_l229_22988

theorem germination_percentage (seeds_plot1 seeds_plot2 : ℕ) (percent_germ_plot1 : ℕ) (total_percent_germ : ℕ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  percent_germ_plot1 = 20 →
  total_percent_germ = 26 →
  ∃ (percent_germ_plot2 : ℕ), percent_germ_plot2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_germination_percentage_l229_22988


namespace NUMINAMATH_GPT_gcd_m_n_l229_22904

noncomputable def m : ℕ := 5 * 11111111
noncomputable def n : ℕ := 111111111

theorem gcd_m_n : gcd m n = 11111111 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l229_22904


namespace NUMINAMATH_GPT_find_n_l229_22934

theorem find_n (x y : ℤ) (n : ℕ) (h1 : (x:ℝ)^n + (y:ℝ)^n = 91) (h2 : (x:ℝ) * y = 11.999999999999998) :
  n = 3 := 
sorry

end NUMINAMATH_GPT_find_n_l229_22934


namespace NUMINAMATH_GPT_project_completion_days_l229_22933

theorem project_completion_days 
  (total_mandays : ℕ)
  (initial_workers : ℕ)
  (leaving_workers : ℕ)
  (remaining_workers : ℕ)
  (days_total : ℕ) :
  total_mandays = 200 →
  initial_workers = 10 →
  leaving_workers = 4 →
  remaining_workers = 6 →
  days_total = 40 :=
by
  intros h0 h1 h2 h3
  sorry

end NUMINAMATH_GPT_project_completion_days_l229_22933


namespace NUMINAMATH_GPT_functional_equation_divisibility_l229_22959

theorem functional_equation_divisibility (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, (f x)^2 + y ∣ f y + x^2) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_divisibility_l229_22959


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l229_22918

theorem no_positive_integer_solutions (A : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) :
  ¬(∃ x : ℕ, x^2 - 2 * A * x + A0 = 0) :=
by sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l229_22918


namespace NUMINAMATH_GPT_sum_of_fractions_and_decimal_l229_22914

theorem sum_of_fractions_and_decimal : 
    (3 / 25 : ℝ) + (1 / 5) + 55.21 = 55.53 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_fractions_and_decimal_l229_22914


namespace NUMINAMATH_GPT_fisherman_caught_total_fish_l229_22974

noncomputable def number_of_boxes : ℕ := 15
noncomputable def fish_per_box : ℕ := 20
noncomputable def fish_outside_boxes : ℕ := 6

theorem fisherman_caught_total_fish :
  number_of_boxes * fish_per_box + fish_outside_boxes = 306 :=
by
  sorry

end NUMINAMATH_GPT_fisherman_caught_total_fish_l229_22974


namespace NUMINAMATH_GPT_MrC_loses_240_after_transactions_l229_22946

theorem MrC_loses_240_after_transactions :
  let house_initial_value := 12000
  let first_transaction_loss_percent := 0.15
  let second_transaction_gain_percent := 0.20
  let house_value_after_first_transaction :=
    house_initial_value * (1 - first_transaction_loss_percent)
  let house_value_after_second_transaction :=
    house_value_after_first_transaction * (1 + second_transaction_gain_percent)
  house_value_after_second_transaction - house_initial_value = 240 :=
by
  sorry

end NUMINAMATH_GPT_MrC_loses_240_after_transactions_l229_22946


namespace NUMINAMATH_GPT_sum_even_numbers_from_2_to_60_l229_22994

noncomputable def sum_even_numbers_seq : ℕ :=
  let a₁ := 2
  let d := 2
  let aₙ := 60
  let n := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

theorem sum_even_numbers_from_2_to_60:
  sum_even_numbers_seq = 930 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_numbers_from_2_to_60_l229_22994


namespace NUMINAMATH_GPT_athlete_D_is_selected_l229_22949

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end NUMINAMATH_GPT_athlete_D_is_selected_l229_22949


namespace NUMINAMATH_GPT_algebraic_expression_defined_iff_l229_22943

theorem algebraic_expression_defined_iff (x : ℝ) : (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_defined_iff_l229_22943


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l229_22957

theorem remainder_when_divided_by_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) : k % 7 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l229_22957


namespace NUMINAMATH_GPT_gabby_fruit_total_l229_22919

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end NUMINAMATH_GPT_gabby_fruit_total_l229_22919


namespace NUMINAMATH_GPT_handshaking_remainder_l229_22964

noncomputable def num_handshaking_arrangements_modulo (n : ℕ) : ℕ := sorry

theorem handshaking_remainder (N : ℕ) (h : num_handshaking_arrangements_modulo 9 = N) :
  N % 1000 = 16 :=
sorry

end NUMINAMATH_GPT_handshaking_remainder_l229_22964


namespace NUMINAMATH_GPT_calculate_expression_l229_22920

theorem calculate_expression : 12 * (1 / (2 / 3 - 1 / 4 + 1 / 6)) = 144 / 7 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l229_22920


namespace NUMINAMATH_GPT_ordered_triple_unique_l229_22982

variable (a b c : ℝ)

theorem ordered_triple_unique
  (h_pos_a : a > 4)
  (h_pos_b : b > 4)
  (h_pos_c : c > 4)
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) := 
sorry

end NUMINAMATH_GPT_ordered_triple_unique_l229_22982


namespace NUMINAMATH_GPT_solve_for_y_l229_22954

theorem solve_for_y (y : ℝ)
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1 / 9 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l229_22954


namespace NUMINAMATH_GPT_sol_earnings_in_a_week_l229_22980

-- Define the number of candy bars sold each day using recurrence relation
def candies_sold (n : ℕ) : ℕ :=
  match n with
  | 0     => 10  -- Day 1
  | (n+1) => candies_sold n + 4  -- Each subsequent day

-- Define the total candies sold in a week and total earnings in dollars
def total_candies_sold_in_a_week : ℕ :=
  List.sum (List.map candies_sold [0, 1, 2, 3, 4, 5])

def total_earnings_in_dollars : ℕ :=
  (total_candies_sold_in_a_week * 10) / 100

-- Proving that Sol will earn 12 dollars in a week
theorem sol_earnings_in_a_week : total_earnings_in_dollars = 12 := by
  sorry

end NUMINAMATH_GPT_sol_earnings_in_a_week_l229_22980


namespace NUMINAMATH_GPT_graph_intersect_x_axis_exactly_once_l229_22951

theorem graph_intersect_x_axis_exactly_once (a : ℝ) :
    (∀ x : ℝ, (a-1) * x^2 - 4 * x + 2 * a = 0 → x = -(1/2)) ∨ -- Quadratic condition with one real root giving unique intersection
    ((a-1) = 0 ∧ ∃ x : ℝ, -4 * x + 2 * a = 0) -- Linear condition giving unique intersection
    ↔ a = -1 ∨ a = 2 ∨ a = 1 :=
by
    sorry

end NUMINAMATH_GPT_graph_intersect_x_axis_exactly_once_l229_22951


namespace NUMINAMATH_GPT_imag_part_of_complex_l229_22925

open Complex

theorem imag_part_of_complex : (im ((5 + I) / (1 + I))) = -2 :=
by
  sorry

end NUMINAMATH_GPT_imag_part_of_complex_l229_22925


namespace NUMINAMATH_GPT_problem_l229_22985

theorem problem (h : ℤ) : (∃ x : ℤ, x = -2 ∧ x^3 + h * x - 12 = 0) → h = -10 := by
  sorry

end NUMINAMATH_GPT_problem_l229_22985


namespace NUMINAMATH_GPT_resistor_value_l229_22984

-- Definitions based on given conditions
def U : ℝ := 9 -- Volt reading by the voltmeter
def I : ℝ := 2 -- Current reading by the ammeter
def U_total : ℝ := 2 * U -- Total voltage in the series circuit

-- Stating the theorem
theorem resistor_value (R₀ : ℝ) :
  (U_total = I * (2 * R₀)) → R₀ = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_resistor_value_l229_22984


namespace NUMINAMATH_GPT_odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l229_22979

def is_in_A (a : ℤ) : Prop := ∃ (x y : ℤ), a = x^2 - y^2

theorem odd_numbers_in_A :
  ∀ (n : ℤ), n % 2 = 1 → is_in_A n :=
sorry

theorem even_4k_minus_2_not_in_A :
  ∀ (k : ℤ), ¬ is_in_A (4 * k - 2) :=
sorry

theorem product_in_A :
  ∀ (a b : ℤ), is_in_A a → is_in_A b → is_in_A (a * b) :=
sorry

end NUMINAMATH_GPT_odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l229_22979


namespace NUMINAMATH_GPT_smallest_positive_integer_form_l229_22924

theorem smallest_positive_integer_form (m n : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d = 1205 * m + 27090 * n ∧ (∀ e, e > 0 → (∃ x y : ℤ, d = 1205 * x + 27090 * y) → d ≤ e) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_form_l229_22924


namespace NUMINAMATH_GPT_angle_BAC_eq_69_l229_22944

-- Definitions and conditions
def AM_Squared_EQ_CM_MN (AM CM MN : ℝ) : Prop := AM^2 = CM * MN
def AM_EQ_MK (AM MK : ℝ) : Prop := AM = MK
def angle_AMN_EQ_CMK (angle_AMN angle_CMK : ℝ) : Prop := angle_AMN = angle_CMK
def angle_B : ℝ := 47
def angle_C : ℝ := 64

-- Final proof statement
theorem angle_BAC_eq_69 (AM CM MN MK : ℝ)
  (h1: AM_Squared_EQ_CM_MN AM CM MN)
  (h2: AM_EQ_MK AM MK)
  (h3: angle_AMN_EQ_CMK 70 70) -- Placeholder angle values since angles must be given/defined
  : ∃ angle_BAC : ℝ, angle_BAC = 69 :=
sorry

end NUMINAMATH_GPT_angle_BAC_eq_69_l229_22944


namespace NUMINAMATH_GPT_solution_for_system_of_inequalities_l229_22965

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end NUMINAMATH_GPT_solution_for_system_of_inequalities_l229_22965


namespace NUMINAMATH_GPT_common_difference_l229_22995

-- Definitions
variable (a₁ d : ℝ) -- First term and common difference of the arithmetic sequence

-- Conditions
def mean_nine_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 8 * d)) = 10

def mean_ten_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 9 * d)) = 13

-- Theorem to prove the common difference is 6
theorem common_difference (a₁ d : ℝ) :
  mean_nine_terms a₁ d → 
  mean_ten_terms a₁ d → 
  d = 6 := by
  intros
  sorry

end NUMINAMATH_GPT_common_difference_l229_22995


namespace NUMINAMATH_GPT_arithmetic_seq_term_ratio_l229_22902

-- Assume two arithmetic sequences a and b
def arithmetic_seq_a (n : ℕ) : ℕ := sorry
def arithmetic_seq_b (n : ℕ) : ℕ := sorry

-- Sum of first n terms of the sequences
def sum_a (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_a |>.sum
def sum_b (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_b |>.sum

-- The given condition: Sn / Tn = (7n + 2) / (n + 3)
axiom sum_condition (n : ℕ) : (sum_a n) / (sum_b n) = (7 * n + 2) / (n + 3)

-- The goal: a4 / b4 = 51 / 10
theorem arithmetic_seq_term_ratio : (arithmetic_seq_a 4 : ℚ) / (arithmetic_seq_b 4 : ℚ) = 51 / 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_term_ratio_l229_22902


namespace NUMINAMATH_GPT_length_of_place_mat_l229_22908

noncomputable def radius : ℝ := 6
noncomputable def width : ℝ := 1.5
def inner_corner_touch (n : ℕ) : Prop := n = 6

theorem length_of_place_mat (y : ℝ) (h1 : radius = 6) (h2 : width = 1.5) (h3 : inner_corner_touch 6) :
  y = (Real.sqrt 141.75 + 1.5) / 2 :=
sorry

end NUMINAMATH_GPT_length_of_place_mat_l229_22908


namespace NUMINAMATH_GPT_parallelogram_probability_l229_22907

theorem parallelogram_probability (P Q R S : ℝ × ℝ) 
  (hP : P = (4, 2)) 
  (hQ : Q = (-2, -2)) 
  (hR : R = (-6, -6)) 
  (hS : S = (0, -2)) :
  let parallelogram_area := 24 -- given the computed area based on provided geometry
  let divided_area := parallelogram_area / 2
  let not_above_x_axis_area := divided_area
  (not_above_x_axis_area / parallelogram_area) = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_probability_l229_22907


namespace NUMINAMATH_GPT_perpendicular_condition_l229_22960

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + 2 * y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := 3 * x - a * y + 1

def perpendicular_lines (a : ℝ) : Prop := 
  ∀ (x y : ℝ), line1 a x y = 0 → line2 a x y = 0 → 3 * a - 2 * a = 0 

theorem perpendicular_condition (a : ℝ) (h : perpendicular_lines a) : a = 0 := sorry

end NUMINAMATH_GPT_perpendicular_condition_l229_22960


namespace NUMINAMATH_GPT_stability_comparison_l229_22990

-- Definitions of conditions
def variance_A : ℝ := 3
def variance_B : ℝ := 1.2

-- Definition of the stability metric
def more_stable (performance_A performance_B : ℝ) : Prop :=
  performance_B < performance_A

-- Target Proposition
theorem stability_comparison (h_variance_A : variance_A = 3)
                            (h_variance_B : variance_B = 1.2) :
  more_stable variance_A variance_B = true :=
by
  sorry

end NUMINAMATH_GPT_stability_comparison_l229_22990


namespace NUMINAMATH_GPT_find_cost_of_book_sold_at_loss_l229_22970

-- Definitions from the conditions
def total_cost (C1 C2 : ℝ) : Prop := C1 + C2 = 540
def selling_price_loss (C1 : ℝ) : ℝ := 0.85 * C1
def selling_price_gain (C2 : ℝ) : ℝ := 1.19 * C2
def same_selling_price (SP1 SP2 : ℝ) : Prop := SP1 = SP2

theorem find_cost_of_book_sold_at_loss (C1 C2 : ℝ) 
  (h1 : total_cost C1 C2) 
  (h2 : same_selling_price (selling_price_loss C1) (selling_price_gain C2)) :
  C1 = 315 :=
by {
   sorry
}

end NUMINAMATH_GPT_find_cost_of_book_sold_at_loss_l229_22970


namespace NUMINAMATH_GPT_quadratic_touches_x_axis_l229_22958

theorem quadratic_touches_x_axis (a : ℝ) : 
  (∃ x : ℝ, 2 * x ^ 2 - 8 * x + a = 0) ∧ (∀ y : ℝ, y^2 - 4 * a = 0 → y = 0) → a = 8 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_touches_x_axis_l229_22958


namespace NUMINAMATH_GPT_sum_and_product_of_reciprocals_l229_22939

theorem sum_and_product_of_reciprocals (x y : ℝ) (h_sum : x + y = 12) (h_prod : x * y = 32) :
  (1/x + 1/y = 3/8) ∧ (1/x * 1/y = 1/32) :=
by
  sorry

end NUMINAMATH_GPT_sum_and_product_of_reciprocals_l229_22939


namespace NUMINAMATH_GPT_specific_certain_event_l229_22931

theorem specific_certain_event :
  ∀ (A B C D : Prop), 
    (¬ A) →
    (¬ B) →
    (¬ C) →
    D →
    D :=
by
  intros A B C D hA hB hC hD
  exact hD

end NUMINAMATH_GPT_specific_certain_event_l229_22931


namespace NUMINAMATH_GPT_exists_x_y_l229_22952

theorem exists_x_y (n : ℕ) (hn : 0 < n) :
  ∃ x y : ℕ, n < x ∧ ¬ x ∣ y ∧ x^x ∣ y^y :=
by sorry

end NUMINAMATH_GPT_exists_x_y_l229_22952


namespace NUMINAMATH_GPT_increasing_function_range_l229_22930

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x - 1 else x + 1

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 / 2 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_increasing_function_range_l229_22930


namespace NUMINAMATH_GPT_smallest_n_for_candy_distribution_l229_22972

theorem smallest_n_for_candy_distribution : ∃ (n : ℕ), (∀ (a : ℕ), ∃ (x : ℕ), (x * (x + 1)) / 2 % n = a % n) ∧ n = 2 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_candy_distribution_l229_22972


namespace NUMINAMATH_GPT_tan_beta_formula_l229_22977

theorem tan_beta_formula (α β : ℝ) 
  (h1 : Real.tan α = -2/3)
  (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 7/4 :=
sorry

end NUMINAMATH_GPT_tan_beta_formula_l229_22977


namespace NUMINAMATH_GPT_patty_coins_value_l229_22956

theorem patty_coins_value (n d q : ℕ) (h₁ : n + d + q = 30) (h₂ : 5 * n + 15 * d - 20 * q = 120) : 
  5 * n + 10 * d + 25 * q = 315 := by
sorry

end NUMINAMATH_GPT_patty_coins_value_l229_22956


namespace NUMINAMATH_GPT_halfway_fraction_l229_22923

theorem halfway_fraction (a b : ℚ) (h1 : a = 1/5) (h2 : b = 1/3) : (a + b) / 2 = 4 / 15 :=
by 
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_halfway_fraction_l229_22923


namespace NUMINAMATH_GPT_range_of_a_l229_22986

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
if x < 1 then -x + 2 else a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 1 ∧ (0 < -x + 2)) ∧ (∀ x : ℝ, x ≥ 1 → (0 < a / x)) → a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l229_22986


namespace NUMINAMATH_GPT_compute_expression_l229_22912

theorem compute_expression (x y z : ℝ) (h₀ : x ≠ y) (h₁ : y ≠ z) (h₂ : z ≠ x) (h₃ : x + y + z = 3) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 9 / (2 * (x^2 + y^2 + z^2)) - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l229_22912


namespace NUMINAMATH_GPT_factorize_mn_minus_mn_cubed_l229_22921

theorem factorize_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n ^ 3 = m * n * (1 + n) * (1 - n) :=
by {
  sorry
}

end NUMINAMATH_GPT_factorize_mn_minus_mn_cubed_l229_22921


namespace NUMINAMATH_GPT_evaluate_expression_l229_22998

theorem evaluate_expression :
  2 * 7^(-1/3 : ℝ) + (1/2 : ℝ) * Real.log (1/64) / Real.log 2 = -3 := 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l229_22998


namespace NUMINAMATH_GPT_triangle_XYZ_PQZ_lengths_l229_22981

theorem triangle_XYZ_PQZ_lengths :
  ∀ (X Y Z P Q : Type) (d_XZ d_YZ d_PQ : ℝ),
  d_XZ = 9 → d_YZ = 12 → d_PQ = 3 →
  ∀ (XY YP : ℝ),
  XY = Real.sqrt (d_XZ^2 + d_YZ^2) →
  YP = (d_PQ / d_XZ) * d_YZ →
  YP = 4 :=
by
  intros X Y Z P Q d_XZ d_YZ d_PQ hXZ hYZ hPQ XY YP hXY hYP
  -- Skipping detailed proof
  sorry

end NUMINAMATH_GPT_triangle_XYZ_PQZ_lengths_l229_22981


namespace NUMINAMATH_GPT_range_of_a_l229_22900

open Real

noncomputable def p (a : ℝ) := ∀ (x : ℝ), x ≥ 1 → (2 * x - 3 * a) ≥ 0
noncomputable def q (a : ℝ) := (0 < 2 * a - 1) ∧ (2 * a - 1 < 1)

theorem range_of_a (a : ℝ) : p a ∧ q a ↔ (1/2 < a ∧ a ≤ 2/3) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l229_22900


namespace NUMINAMATH_GPT_ratio_of_votes_l229_22991

theorem ratio_of_votes (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h1 : randy_votes = 16)
  (h2 : shaun_votes = 5 * randy_votes)
  (h3 : eliot_votes = 160) :
  eliot_votes / shaun_votes = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_votes_l229_22991


namespace NUMINAMATH_GPT_number_equals_14_l229_22937

theorem number_equals_14 (n : ℕ) (h1 : 2^n - 2^(n-2) = 3 * 2^12) (h2 : n = 14) : n = 14 := 
by 
  sorry

end NUMINAMATH_GPT_number_equals_14_l229_22937


namespace NUMINAMATH_GPT_solve_xy_eq_x_plus_y_l229_22906

theorem solve_xy_eq_x_plus_y (x y : ℤ) (h : x * y = x + y) : (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_xy_eq_x_plus_y_l229_22906


namespace NUMINAMATH_GPT_john_total_spent_l229_22968

noncomputable def total_spent (computer_cost : ℝ) (peripheral_ratio : ℝ) (base_video_cost : ℝ) : ℝ :=
  let peripheral_cost := computer_cost * peripheral_ratio
  let upgraded_video_cost := base_video_cost * 2
  computer_cost + peripheral_cost + (upgraded_video_cost - base_video_cost)

theorem john_total_spent :
  total_spent 1500 0.2 300 = 2100 :=
by
  sorry

end NUMINAMATH_GPT_john_total_spent_l229_22968


namespace NUMINAMATH_GPT_two_roots_iff_a_gt_neg1_l229_22936

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end NUMINAMATH_GPT_two_roots_iff_a_gt_neg1_l229_22936


namespace NUMINAMATH_GPT_gcd_lcm_divisible_l229_22975

theorem gcd_lcm_divisible (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b + Nat.lcm a b = a + b) : a % b = 0 ∨ b % a = 0 := 
sorry

end NUMINAMATH_GPT_gcd_lcm_divisible_l229_22975


namespace NUMINAMATH_GPT_triangle_area_l229_22969

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : is_right_triangle a b c) :
  (1 / 2 : ℝ) * a * b = 180 :=
by sorry

end NUMINAMATH_GPT_triangle_area_l229_22969


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l229_22927

theorem parabola_focus_coordinates (h : ∀ y, y^2 = 4 * x) : ∃ x, x = 1 ∧ y = 0 := 
sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l229_22927


namespace NUMINAMATH_GPT_circle_diameter_from_area_l229_22993

theorem circle_diameter_from_area (A : ℝ) (hA : A = 400 * Real.pi) :
    ∃ D : ℝ, D = 40 := 
by
  -- Consider the formula for the area of a circle with radius r.
  -- The area is given as A = π * r^2.
  let r := Real.sqrt 400 -- Solve for radius r.
  have hr : r = 20 := by sorry
  -- The diameter D is twice the radius.
  let D := 2 * r 
  existsi D
  have hD : D = 40 := by sorry
  exact hD

end NUMINAMATH_GPT_circle_diameter_from_area_l229_22993


namespace NUMINAMATH_GPT_number_of_bottles_l229_22948

-- Define the weights and total weight based on given conditions
def weight_of_two_bags_chips : ℕ := 800
def total_weight_five_bags_and_juices : ℕ := 2200
def weight_difference_chip_Juice : ℕ := 350

-- Considering 1 bag of chips weighs 400 g (derived from the condition)
def weight_of_one_bag_chips : ℕ := 400
def weight_of_one_bottle_juice : ℕ := weight_of_one_bag_chips - weight_difference_chip_Juice

-- Define the proof of the question
theorem number_of_bottles :
  (total_weight_five_bags_and_juices - (5 * weight_of_one_bag_chips)) / weight_of_one_bottle_juice = 4 := by sorry

end NUMINAMATH_GPT_number_of_bottles_l229_22948


namespace NUMINAMATH_GPT_pentagon_area_proof_l229_22967

noncomputable def area_of_pentagon : ℕ :=
  let side1 := 18
  let side2 := 25
  let side3 := 30
  let side4 := 28
  let side5 := 25
  -- Assuming the total area calculated from problem's conditions
  950

theorem pentagon_area_proof : area_of_pentagon = 950 := by
  sorry

end NUMINAMATH_GPT_pentagon_area_proof_l229_22967


namespace NUMINAMATH_GPT_units_digit_of_A_is_1_l229_22997

-- Definition of A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Main theorem stating that the units digit of A is 1
theorem units_digit_of_A_is_1 : (A % 10) = 1 :=
by 
  -- Given conditions about powers of 3 and their properties in modulo 10
  sorry

end NUMINAMATH_GPT_units_digit_of_A_is_1_l229_22997


namespace NUMINAMATH_GPT_integral_percentage_l229_22916

variable (a b : ℝ)

theorem integral_percentage (h : ∀ x, x^2 > 0) :
  (∫ x in a..b, (1 / 20 * x^2 + 3 / 10 * x^2)) = 0.35 * (∫ x in a..b, x^2) :=
by
  sorry

end NUMINAMATH_GPT_integral_percentage_l229_22916


namespace NUMINAMATH_GPT_find_y_l229_22976

variable (a b c x : ℝ) (p q r y : ℝ)
variable (log : ℝ → ℝ) -- represents the logarithm function

-- Conditions as hypotheses
axiom log_eq : (log a) / p = (log b) / q
axiom log_eq' : (log b) / q = (log c) / r
axiom log_eq'' : (log c) / r = log x
axiom x_ne_one : x ≠ 1
axiom eq_exp : (b^3) / (a^2 * c) = x^y

-- Statement to be proven
theorem find_y : y = 3 * q - 2 * p - r := by
  sorry

end NUMINAMATH_GPT_find_y_l229_22976


namespace NUMINAMATH_GPT_max_min_page_difference_l229_22903

-- Define the number of pages in each book
variables (Poetry Documents Rites Changes SpringAndAutumn : ℤ)

-- Define the conditions as given in the problem
axiom h1 : abs (Poetry - Documents) = 24
axiom h2 : abs (Documents - Rites) = 17
axiom h3 : abs (Rites - Changes) = 27
axiom h4 : abs (Changes - SpringAndAutumn) = 19
axiom h5 : abs (SpringAndAutumn - Poetry) = 15

-- Assertion to prove
theorem max_min_page_difference : 
  ∃ a b c d e : ℤ, a = Poetry ∧ b = Documents ∧ c = Rites ∧ d = Changes ∧ e = SpringAndAutumn ∧ 
  abs (a - b) = 24 ∧ abs (b - c) = 17 ∧ abs (c - d) = 27 ∧ abs (d - e) = 19 ∧ abs (e - a) = 15 ∧ 
  (max a (max b (max c (max d e))) - min a (min b (min c (min d e)))) = 34 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_min_page_difference_l229_22903


namespace NUMINAMATH_GPT_deepak_age_l229_22942

theorem deepak_age : ∀ (R D : ℕ), (R / D = 4 / 3) ∧ (R + 6 = 18) → D = 9 :=
by
  sorry

end NUMINAMATH_GPT_deepak_age_l229_22942


namespace NUMINAMATH_GPT_alice_bob_meet_l229_22909

theorem alice_bob_meet :
  ∃ k : ℕ, (4 * k - 4 * (k / 5) ≡ 8 * k [MOD 15]) ∧ (k = 5) :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meet_l229_22909


namespace NUMINAMATH_GPT_ratio_problem_l229_22941

theorem ratio_problem 
  (x y z w : ℚ) 
  (h1 : x / y = 12) 
  (h2 : z / y = 4) 
  (h3 : z / w = 3 / 4) : 
  w / x = 4 / 9 := 
  sorry

end NUMINAMATH_GPT_ratio_problem_l229_22941


namespace NUMINAMATH_GPT_find_triples_of_positive_integers_l229_22910

theorem find_triples_of_positive_integers :
  ∀ (x y z : ℕ), 2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023 ↔ 
  (x = 3 ∧ y = 3 ∧ z = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 3 ∧ y = 3 ∧ z = 2) := 
by 
  sorry

end NUMINAMATH_GPT_find_triples_of_positive_integers_l229_22910


namespace NUMINAMATH_GPT_constant_function_l229_22926

theorem constant_function {f : ℕ → ℕ} (h : ∀ x y : ℕ, x * f y + y * f x = (x + y) * f (x^2 + y^2)) : ∃ c : ℕ, ∀ x, f x = c := 
sorry

end NUMINAMATH_GPT_constant_function_l229_22926


namespace NUMINAMATH_GPT_red_car_count_l229_22945

-- Define the ratio and the given number of black cars
def ratio_red_to_black (R B : ℕ) : Prop := R * 8 = B * 3

-- Define the given number of black cars
def black_cars : ℕ := 75

-- State the theorem we want to prove
theorem red_car_count : ∃ R : ℕ, ratio_red_to_black R black_cars ∧ R = 28 :=
by
  sorry

end NUMINAMATH_GPT_red_car_count_l229_22945


namespace NUMINAMATH_GPT_framed_painting_ratio_correct_l229_22922

/-- Define the conditions -/
def painting_height : ℕ := 30
def painting_width : ℕ := 20
def width_ratio : ℕ := 3

/-- Calculate the framed dimensions and check the area conditions -/
def framed_smaller_dimension (x : ℕ) : ℕ := painting_width + 2 * x
def framed_larger_dimension (x : ℕ) : ℕ := painting_height + 6 * x

theorem framed_painting_ratio_correct (x : ℕ) (h : (painting_width + 2 * x) * (painting_height + 6 * x) = 2 * (painting_width * painting_height)) :
  framed_smaller_dimension x / framed_larger_dimension x = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_framed_painting_ratio_correct_l229_22922


namespace NUMINAMATH_GPT_gain_amount_l229_22950

theorem gain_amount (gain_percent : ℝ) (gain : ℝ) (amount : ℝ) 
  (h_gain_percent : gain_percent = 1) 
  (h_gain : gain = 0.70) 
  : amount = 70 :=
by
  sorry

end NUMINAMATH_GPT_gain_amount_l229_22950


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l229_22915

variable (x : ℚ)

def is_integer (n : ℚ) : Prop := ∃ (k : ℤ), n = k

theorem sufficient_but_not_necessary :
  (is_integer x → is_integer (2 * x + 1)) ∧
  (¬ (is_integer (2 * x + 1) → is_integer x)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l229_22915


namespace NUMINAMATH_GPT_tammy_driving_rate_l229_22989

-- Define the conditions given in the problem
def total_miles : ℕ := 1980
def total_hours : ℕ := 36

-- Define the desired rate to prove
def expected_rate : ℕ := 55

-- The theorem stating that given the conditions, Tammy's driving rate is correct
theorem tammy_driving_rate :
  total_miles / total_hours = expected_rate :=
by
  -- Detailed proof would go here
  sorry

end NUMINAMATH_GPT_tammy_driving_rate_l229_22989


namespace NUMINAMATH_GPT_spending_difference_is_65_l229_22962

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_spending_difference_is_65_l229_22962


namespace NUMINAMATH_GPT_solution_set_l229_22999

-- Definitions representing the given conditions
def cond1 (x : ℝ) := x - 3 < 0
def cond2 (x : ℝ) := x + 1 ≥ 0

-- The problem: Prove the solution set is as given
theorem solution_set (x : ℝ) :
  (cond1 x) ∧ (cond2 x) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l229_22999


namespace NUMINAMATH_GPT_current_in_circuit_l229_22978

open Complex

theorem current_in_circuit
  (V : ℂ := 2 + 3 * I)
  (Z : ℂ := 4 - 2 * I) :
  (V / Z) = (1 / 10 + 4 / 5 * I) :=
  sorry

end NUMINAMATH_GPT_current_in_circuit_l229_22978


namespace NUMINAMATH_GPT_solve_for_x_l229_22983

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -1 then x + 2 
  else if x < 2 then x^2 
  else 2 * x

theorem solve_for_x (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l229_22983


namespace NUMINAMATH_GPT_squares_sum_l229_22966

theorem squares_sum (a b c : ℝ) 
  (h1 : 36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c) ^ 2) : 
  a^2 + b^2 + c^2 = 14 := 
by
  sorry

end NUMINAMATH_GPT_squares_sum_l229_22966


namespace NUMINAMATH_GPT_functional_eq_solution_l229_22901

theorem functional_eq_solution (f : ℝ → ℝ) 
  (H : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
by 
  sorry

end NUMINAMATH_GPT_functional_eq_solution_l229_22901


namespace NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l229_22961

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l229_22961


namespace NUMINAMATH_GPT_golden_apples_first_six_months_l229_22996

-- Use appropriate namespaces
namespace ApolloProblem

-- Define the given conditions
def total_cost : ℕ := 54
def months_in_half_year : ℕ := 6

-- Prove that the number of golden apples charged for the first six months is 18
theorem golden_apples_first_six_months (X : ℕ) 
  (h1 : 6 * X + 6 * (2 * X) = total_cost) : 
  6 * X = 18 := 
sorry

end ApolloProblem

end NUMINAMATH_GPT_golden_apples_first_six_months_l229_22996


namespace NUMINAMATH_GPT_min_value_abc2_l229_22913

variables (a b c d : ℝ)

def condition_1 : Prop := a + b = 9 / (c - d)
def condition_2 : Prop := c + d = 25 / (a - b)

theorem min_value_abc2 :
  condition_1 a b c d → condition_2 a b c d → (a^2 + b^2 + c^2 + d^2) = 34 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_min_value_abc2_l229_22913


namespace NUMINAMATH_GPT_polynomial_value_l229_22935

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l229_22935


namespace NUMINAMATH_GPT_find_remainder_l229_22932

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end NUMINAMATH_GPT_find_remainder_l229_22932


namespace NUMINAMATH_GPT_at_least_one_negative_l229_22929

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) : a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_GPT_at_least_one_negative_l229_22929


namespace NUMINAMATH_GPT_remainder_of_large_number_l229_22953

theorem remainder_of_large_number :
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  last_four_digits % 16 = 9 := 
by
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  show last_four_digits % 16 = 9
  sorry

end NUMINAMATH_GPT_remainder_of_large_number_l229_22953


namespace NUMINAMATH_GPT_cans_of_type_B_purchased_l229_22928

variable (T P R : ℕ)

-- Conditions
def cost_per_can_A : ℕ := P / T
def cost_per_can_B : ℕ := 2 * cost_per_can_A T P
def quarters_in_dollar : ℕ := 4

-- Question and proof target
theorem cans_of_type_B_purchased (T P R : ℕ) (hT : T > 0) (hP : P > 0) (hR : R > 0) :
  (4 * R) / (2 * P / T) = 2 * R * T / P :=
by
  sorry

end NUMINAMATH_GPT_cans_of_type_B_purchased_l229_22928


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_l229_22911

-- Define the conditions
def isEccentricity (e : ℝ) := e = (Real.sqrt 3) / 3
def segmentLength (L : ℝ) := L = (4 * Real.sqrt 3) / 3

-- Define properties
def is_ellipse (a b c : ℝ) := a > b ∧ b > 0 ∧ (a^2 = b^2 + c^2) ∧ (c = (Real.sqrt 3) / 3 * a)

-- The problem statement
theorem standard_equation_of_ellipse
(a b c : ℝ) (E L : ℝ)
(hE : isEccentricity E)
(hL : segmentLength L)
(h : is_ellipse a b c)
: (a = Real.sqrt 3) ∧ (c = 1) ∧ (b = Real.sqrt 2) ∧ (segmentLength L)
  → ( ∀ x y : ℝ, ((x^2 / 3) + (y^2 / 2) = 1) ) := by
  sorry

end NUMINAMATH_GPT_standard_equation_of_ellipse_l229_22911


namespace NUMINAMATH_GPT_factorization_of_polynomial_l229_22905

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^2 + 6 * x + 9 - 64 * x^4 = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  intro x
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l229_22905


namespace NUMINAMATH_GPT_max_c_value_for_f_x_range_l229_22971

theorem max_c_value_for_f_x_range:
  (∀ c : ℝ, (∃ x : ℝ, x^2 + 4 * x + c = -2) → c ≤ 2) ∧ (∃ (x : ℝ), x^2 + 4 * x + 2 = -2) :=
sorry

end NUMINAMATH_GPT_max_c_value_for_f_x_range_l229_22971


namespace NUMINAMATH_GPT_arithmetic_sequence_120th_term_l229_22947

theorem arithmetic_sequence_120th_term :
  let a1 := 6
  let d := 6
  let n := 120
  let a_n := a1 + (n - 1) * d
  a_n = 720 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_120th_term_l229_22947


namespace NUMINAMATH_GPT_digit_difference_l229_22992

theorem digit_difference (x y : ℕ) (h : 10 * x + y - (10 * y + x) = 45) : x - y = 5 :=
sorry

end NUMINAMATH_GPT_digit_difference_l229_22992


namespace NUMINAMATH_GPT_f_g_of_3_l229_22917

def f (x : ℤ) : ℤ := 2 * x + 3
def g (x : ℤ) : ℤ := x^3 - 6

theorem f_g_of_3 : f (g 3) = 45 := by
  sorry

end NUMINAMATH_GPT_f_g_of_3_l229_22917


namespace NUMINAMATH_GPT_find_constants_and_calculate_result_l229_22963

theorem find_constants_and_calculate_result :
  ∃ (a b : ℤ), 
    (∀ (x : ℤ), (x + a) * (x + 6) = x^2 + 8 * x + 12) ∧ 
    (∀ (x : ℤ), (x - a) * (x + b) = x^2 + x - 6) ∧ 
    (∀ (x : ℤ), (x + a) * (x + b) = x^2 + 5 * x + 6) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_and_calculate_result_l229_22963


namespace NUMINAMATH_GPT_Trumpington_marching_band_max_l229_22987

theorem Trumpington_marching_band_max (n : ℕ) (k : ℕ) 
  (h1 : 20 * n % 26 = 4)
  (h2 : n = 8 + 13 * k)
  (h3 : 20 * n < 1000) 
  : 20 * (8 + 13 * 3) = 940 := 
by
  sorry

end NUMINAMATH_GPT_Trumpington_marching_band_max_l229_22987


namespace NUMINAMATH_GPT_triangle_inequality_circumradius_l229_22973

theorem triangle_inequality_circumradius (a b c R : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
  (circumradius_def : R = (a * b * c) / (4 * (Real.sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))))) :
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R ^ 2)) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_circumradius_l229_22973


namespace NUMINAMATH_GPT_factorization_correct_l229_22938

-- Define the input expression
def expr (x y : ℝ) : ℝ := 2 * x^3 - 18 * x * y^2

-- Define the factorized form
def factorized_expr (x y : ℝ) : ℝ := 2 * x * (x + 3*y) * (x - 3*y)

-- Prove that the original expression is equal to the factorized form
theorem factorization_correct (x y : ℝ) : expr x y = factorized_expr x y := 
by sorry

end NUMINAMATH_GPT_factorization_correct_l229_22938


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l229_22955

open BigOperators

theorem cyclic_sum_inequality {n : ℕ} (h : 0 < n) (a : ℕ → ℝ)
  (hpos : ∀ i, 0 < a i) :
  (∑ k in Finset.range n, a k / (a (k+1) + a (k+2))) > n / 4 := by
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l229_22955


namespace NUMINAMATH_GPT_smallest_integer_solution_m_l229_22940

theorem smallest_integer_solution_m :
  (∃ x y m : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) →
  ∃ m : ℤ, (∀ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) ↔ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_m_l229_22940
