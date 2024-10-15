import Mathlib

namespace NUMINAMATH_GPT_sin_cos_sixth_power_l1579_157919

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1/2) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sixth_power_l1579_157919


namespace NUMINAMATH_GPT_greatest_x_plus_z_l1579_157933

theorem greatest_x_plus_z (x y z c d : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 1 ≤ z ∧ z ≤ 9)
  (h4 : 700 - c = 700)
  (h5 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 693)
  (h6 : x > z) :
  d = 11 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_plus_z_l1579_157933


namespace NUMINAMATH_GPT_linear_eq_represents_plane_l1579_157970

theorem linear_eq_represents_plane (A B C : ℝ) (h : ¬ (A = 0 ∧ B = 0 ∧ C = 0)) :
  ∃ (P : ℝ × ℝ × ℝ → Prop), (∀ (x y z : ℝ), P (x, y, z) ↔ A * x + B * y + C * z = 0) ∧ 
  (P (0, 0, 0)) :=
by
  -- To be filled in with the proof steps
  sorry

end NUMINAMATH_GPT_linear_eq_represents_plane_l1579_157970


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1579_157908

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_15th_term :
  arithmetic_sequence (-3) 4 15 = 53 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1579_157908


namespace NUMINAMATH_GPT_distinct_products_count_is_26_l1579_157947

open Finset

def set_numbers : Finset ℕ := {2, 3, 5, 7, 11}

def distinct_products_count (s : Finset ℕ) : ℕ :=
  let pairs := s.powerset.filter (λ t => 2 ≤ t.card)
  pairs.card

theorem distinct_products_count_is_26 : distinct_products_count set_numbers = 26 := by
  sorry

end NUMINAMATH_GPT_distinct_products_count_is_26_l1579_157947


namespace NUMINAMATH_GPT_profit_correct_l1579_157939

-- Conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def sets : ℕ := 500

-- Definitions used in the problem
def manufacturing_cost : ℕ := initial_outlay + (sets * cost_per_set)
def revenue : ℕ := sets * selling_price_per_set
def profit : ℕ := revenue - manufacturing_cost

-- The theorem statement
theorem profit_correct : profit = 5000 := by
  sorry

end NUMINAMATH_GPT_profit_correct_l1579_157939


namespace NUMINAMATH_GPT_point_D_in_fourth_quadrant_l1579_157985

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (-1, -2)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (1, -2)

theorem point_D_in_fourth_quadrant : is_in_fourth_quadrant (point_D.1) (point_D.2) :=
by
  sorry

end NUMINAMATH_GPT_point_D_in_fourth_quadrant_l1579_157985


namespace NUMINAMATH_GPT_harvest_weeks_l1579_157925

/-- Lewis earns $403 every week during a certain number of weeks of harvest. 
If he has to pay $49 rent every week, and he earns $93,899 during the harvest season, 
we need to prove that the number of weeks in the harvest season is 265. --/
theorem harvest_weeks 
  (E : ℕ) (R : ℕ) (T : ℕ) (W : ℕ) 
  (hE : E = 403) (hR : R = 49) (hT : T = 93899) 
  (hW : W = 265) : 
  W = (T / (E - R)) := 
by sorry

end NUMINAMATH_GPT_harvest_weeks_l1579_157925


namespace NUMINAMATH_GPT_max_product_of_xy_on_circle_l1579_157927

theorem max_product_of_xy_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  ∃ (x y : ℤ), (x^2 + y^2 = 100) ∧ (∀ x y : ℤ, x^2 + y^2 = 100 → x * y ≤ 48) ∧ x * y = 48 := by
  sorry

end NUMINAMATH_GPT_max_product_of_xy_on_circle_l1579_157927


namespace NUMINAMATH_GPT_new_dressing_contains_12_percent_vinegar_l1579_157905

-- Definitions
def new_dressing_vinegar_percentage (p_vinegar q_vinegar p_fraction q_fraction : ℝ) : ℝ :=
  p_vinegar * p_fraction + q_vinegar * q_fraction

-- Conditions
def p_vinegar : ℝ := 0.30
def q_vinegar : ℝ := 0.10
def p_fraction : ℝ := 0.10
def q_fraction : ℝ := 0.90

-- The theorem to be proven
theorem new_dressing_contains_12_percent_vinegar :
  new_dressing_vinegar_percentage p_vinegar q_vinegar p_fraction q_fraction = 0.12 := 
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_new_dressing_contains_12_percent_vinegar_l1579_157905


namespace NUMINAMATH_GPT_irrational_sum_root_l1579_157984

theorem irrational_sum_root
  (α : ℝ) (hα : Irrational α)
  (n : ℕ) (hn : 0 < n) :
  Irrational ((α + (α^2 - 1).sqrt)^(1/n : ℝ) + (α - (α^2 - 1).sqrt)^(1/n : ℝ)) := sorry

end NUMINAMATH_GPT_irrational_sum_root_l1579_157984


namespace NUMINAMATH_GPT_representable_as_product_l1579_157911

theorem representable_as_product (n : ℤ) (p q : ℚ) (h1 : n > 1995) (h2 : 0 < p) (h3 : p < 1) :
  ∃ (terms : List ℚ), p = terms.prod ∧ ∀ t ∈ terms, ∃ n, t = (n^2 - 1995^2) / (n^2 - 1994^2) ∧ n > 1995 :=
sorry

end NUMINAMATH_GPT_representable_as_product_l1579_157911


namespace NUMINAMATH_GPT_probability_of_snow_during_holiday_l1579_157963

theorem probability_of_snow_during_holiday
  (P_snow_Friday : ℝ)
  (P_snow_Monday : ℝ)
  (P_snow_independent : true) -- Placeholder since we assume independence
  (h_Friday: P_snow_Friday = 0.30)
  (h_Monday: P_snow_Monday = 0.45) :
  ∃ P_snow_holiday, P_snow_holiday = 0.615 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_snow_during_holiday_l1579_157963


namespace NUMINAMATH_GPT_Albert_cabbage_count_l1579_157904

-- Define the conditions
def rows := 12
def heads_per_row := 15

-- State the theorem
theorem Albert_cabbage_count : rows * heads_per_row = 180 := 
by sorry

end NUMINAMATH_GPT_Albert_cabbage_count_l1579_157904


namespace NUMINAMATH_GPT_evaluate_expression_l1579_157932

theorem evaluate_expression (a b : ℝ) (h1 : a = 4) (h2 : b = -1) : -2 * a ^ 2 - 3 * b ^ 2 + 2 * a * b = -43 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1579_157932


namespace NUMINAMATH_GPT_solve_problem_l1579_157992

open Real

noncomputable def problem_statement : ℝ :=
  2 * log (sqrt 2) + (log 5 / log 2) * log 2

theorem solve_problem : problem_statement = 1 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l1579_157992


namespace NUMINAMATH_GPT_parking_cost_savings_l1579_157934

theorem parking_cost_savings
  (weekly_rate : ℕ := 10)
  (monthly_rate : ℕ := 24)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12) :
  (weekly_rate * weeks_in_year) - (monthly_rate * months_in_year) = 232 :=
by
  sorry

end NUMINAMATH_GPT_parking_cost_savings_l1579_157934


namespace NUMINAMATH_GPT_median_number_of_children_is_three_l1579_157950

/-- Define the context of the problem with total number of families. -/
def total_families : Nat := 15

/-- Prove that given the conditions, the median number of children is 3. -/
theorem median_number_of_children_is_three 
  (h : total_families = 15) : 
  ∃ median : Nat, median = 3 :=
by
  sorry

end NUMINAMATH_GPT_median_number_of_children_is_three_l1579_157950


namespace NUMINAMATH_GPT_find_initial_amount_l1579_157974

-- Let x be the initial amount Mark paid for the Magic card
variable {x : ℝ}

-- Condition 1: The card triples in value, resulting in 3x
-- Condition 2: Mark makes a profit of 200
def initial_amount (x : ℝ) : Prop := (3 * x - x = 200)

-- Theorem: Prove that the initial amount x equals 100 given the conditions
theorem find_initial_amount (h : initial_amount x) : x = 100 := by
  sorry

end NUMINAMATH_GPT_find_initial_amount_l1579_157974


namespace NUMINAMATH_GPT_calculate_expression_l1579_157988

theorem calculate_expression : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1579_157988


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l1579_157957

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - 4 * x - 9 * y^2 - 18 * y = 56

-- Define the distance between the foci of the hyperbola
def distance_between_foci (d : ℝ) : Prop :=
  d = 2 * Real.sqrt (170 / 3)

-- The theorem stating that the distance between the foci of the given hyperbola
theorem hyperbola_foci_distance :
  ∃ d, hyperbola_eq x y → distance_between_foci d :=
by { sorry }

end NUMINAMATH_GPT_hyperbola_foci_distance_l1579_157957


namespace NUMINAMATH_GPT_Cameron_list_count_l1579_157903

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end NUMINAMATH_GPT_Cameron_list_count_l1579_157903


namespace NUMINAMATH_GPT_sqrt_x_minus_2_meaningful_l1579_157986

theorem sqrt_x_minus_2_meaningful (x : ℝ) (hx : x = 0 ∨ x = -1 ∨ x = -2 ∨ x = 2) : (x = 2) ↔ (x - 2 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_meaningful_l1579_157986


namespace NUMINAMATH_GPT_range_of_a_l1579_157965

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x = 1 → a * x^2 + 2 * x + 1 < 0) ↔ a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1579_157965


namespace NUMINAMATH_GPT_oranges_in_shop_l1579_157951

-- Define the problem conditions
def ratio (M O A : ℕ) : Prop := (10 * O = 2 * M) ∧ (10 * A = 3 * M)

noncomputable def numMangoes : ℕ := 120
noncomputable def numApples : ℕ := 36

-- Statement of the problem
theorem oranges_in_shop (ratio_factor : ℕ) (h_ratio : ratio numMangoes (2 * ratio_factor) numApples) :
  (2 * ratio_factor) = 24 := by
  sorry

end NUMINAMATH_GPT_oranges_in_shop_l1579_157951


namespace NUMINAMATH_GPT_infinite_series_sum_l1579_157940

theorem infinite_series_sum :
  (∑' n : ℕ, (2 * (n + 1) * (n + 1) + (n + 1) + 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))) = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1579_157940


namespace NUMINAMATH_GPT_point_of_tangency_of_circles_l1579_157914

/--
Given two circles defined by the following equations:
1. \( x^2 - 2x + y^2 - 10y + 17 = 0 \)
2. \( x^2 - 8x + y^2 - 10y + 49 = 0 \)
Prove that the coordinates of the point of tangency of these circles are \( (2.5, 5) \).
-/
theorem point_of_tangency_of_circles :
  (∃ x y : ℝ, (x^2 - 2*x + y^2 - 10*y + 17 = 0) ∧ (x = 2.5) ∧ (y = 5)) ∧ 
  (∃ x' y' : ℝ, (x'^2 - 8*x' + y'^2 - 10*y' + 49 = 0) ∧ (x' = 2.5) ∧ (y' = 5)) :=
sorry

end NUMINAMATH_GPT_point_of_tangency_of_circles_l1579_157914


namespace NUMINAMATH_GPT_smaller_angle_in_parallelogram_l1579_157996

theorem smaller_angle_in_parallelogram (a b : ℝ) (h1 : a + b = 180)
  (h2 : b = a + 70) : a = 55 :=
by sorry

end NUMINAMATH_GPT_smaller_angle_in_parallelogram_l1579_157996


namespace NUMINAMATH_GPT_sequence_an_general_formula_sequence_bn_sum_l1579_157943

theorem sequence_an_general_formula
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3) :
  ∀ n, a n = 3 ^ n := sorry

theorem sequence_bn_sum
  (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3)
  (h3 : ∀ n, b n = a (n + 1) / (S n * S (n + 1))) :
  ∀ n, T n = (2 / 3) * (1 / 2 - 1 / (3 ^ (n + 1) - 1)) := sorry

end NUMINAMATH_GPT_sequence_an_general_formula_sequence_bn_sum_l1579_157943


namespace NUMINAMATH_GPT_sum_of_distances_l1579_157980

theorem sum_of_distances (A B C D M P : ℝ × ℝ) 
    (hA : A = (0, 0))
    (hB : B = (4, 0))
    (hC : C = (4, 4))
    (hD : D = (0, 4))
    (hM : M = (2, 0))
    (hP : P = (0, 2)) :
    dist A M + dist A P = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distances_l1579_157980


namespace NUMINAMATH_GPT_circumference_of_base_l1579_157928

-- Definitions used for the problem
def radius : ℝ := 6
def sector_angle : ℝ := 300
def full_circle_angle : ℝ := 360

-- Ask for the circumference of the base of the cone formed by the sector
theorem circumference_of_base (r : ℝ) (theta_sector : ℝ) (theta_full : ℝ) :
  (theta_sector / theta_full) * (2 * π * r) = 10 * π :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_base_l1579_157928


namespace NUMINAMATH_GPT_fraction_problem_l1579_157973

theorem fraction_problem : 
  (  (1/4 - 1/5) / (1/3 - 1/4)  ) = 3/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l1579_157973


namespace NUMINAMATH_GPT_find_acute_angle_of_parallel_vectors_l1579_157982

open Real

theorem find_acute_angle_of_parallel_vectors (x : ℝ) (hx1 : (sin x) * (1 / 2 * cos x) = 1 / 4) (hx2 : 0 < x ∧ x < π / 2) : x = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_acute_angle_of_parallel_vectors_l1579_157982


namespace NUMINAMATH_GPT_notebooks_to_sell_to_earn_profit_l1579_157971

-- Define the given conditions
def notebooks_purchased : ℕ := 2000
def cost_per_notebook : ℚ := 0.15
def selling_price_per_notebook : ℚ := 0.30
def desired_profit : ℚ := 120

-- Define the total cost
def total_cost := notebooks_purchased * cost_per_notebook

-- Define the total revenue needed
def total_revenue_needed := total_cost + desired_profit

-- Define the number of notebooks to be sold to achieve the total revenue
def notebooks_to_sell := total_revenue_needed / selling_price_per_notebook

-- Prove that the number of notebooks to be sold is 1400 to make a profit of $120
theorem notebooks_to_sell_to_earn_profit : notebooks_to_sell = 1400 := 
by {
  sorry
}

end NUMINAMATH_GPT_notebooks_to_sell_to_earn_profit_l1579_157971


namespace NUMINAMATH_GPT_andrea_rhinestones_needed_l1579_157964

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end NUMINAMATH_GPT_andrea_rhinestones_needed_l1579_157964


namespace NUMINAMATH_GPT_circle_symmetry_l1579_157926

theorem circle_symmetry (a b : ℝ) 
  (h1 : ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ↔ (x - 1)^2 + (y - 3)^2 = 1) 
  (symm_line : ∀ x y : ℝ, y = x + 1) : a + b = 2 :=
sorry

end NUMINAMATH_GPT_circle_symmetry_l1579_157926


namespace NUMINAMATH_GPT_trailing_zeroes_500_fact_l1579_157998

theorem trailing_zeroes_500_fact : 
  let count_multiples (n m : ℕ) := n / m 
  let count_5 := count_multiples 500 5
  let count_25 := count_multiples 500 25
  let count_125 := count_multiples 500 125
-- We don't count multiples of 625 because 625 > 500, thus its count is 0. 
-- Therefore: total trailing zeroes = count_5 + count_25 + count_125
  count_5 + count_25 + count_125 = 124 := sorry

end NUMINAMATH_GPT_trailing_zeroes_500_fact_l1579_157998


namespace NUMINAMATH_GPT_orange_pyramid_total_l1579_157954

theorem orange_pyramid_total :
  let base_length := 7
  let base_width := 9
  -- layer 1 -> dimensions (7, 9)
  -- layer 2 -> dimensions (6, 8)
  -- layer 3 -> dimensions (5, 6)
  -- layer 4 -> dimensions (4, 5)
  -- layer 5 -> dimensions (3, 3)
  -- layer 6 -> dimensions (2, 2)
  -- layer 7 -> dimensions (1, 1)
  (base_length * base_width) + ((base_length - 1) * (base_width - 1))
  + ((base_length - 2) * (base_width - 3)) + ((base_length - 3) * (base_width - 4))
  + ((base_length - 4) * (base_width - 6)) + ((base_length - 5) * (base_width - 7))
  + ((base_length - 6) * (base_width - 8)) = 175 := sorry

end NUMINAMATH_GPT_orange_pyramid_total_l1579_157954


namespace NUMINAMATH_GPT_perry_more_games_than_phil_l1579_157969

theorem perry_more_games_than_phil (dana_wins charlie_wins perry_wins : ℕ) :
  perry_wins = dana_wins + 5 →
  charlie_wins = dana_wins - 2 →
  charlie_wins + 3 = 12 →
  perry_wins - 12 = 4 :=
by
  sorry

end NUMINAMATH_GPT_perry_more_games_than_phil_l1579_157969


namespace NUMINAMATH_GPT_light_year_scientific_notation_l1579_157922

def sci_not_eq : Prop := 
  let x := 9500000000000
  let y := 9.5 * 10^12
  x = y

theorem light_year_scientific_notation : sci_not_eq :=
  by sorry

end NUMINAMATH_GPT_light_year_scientific_notation_l1579_157922


namespace NUMINAMATH_GPT_bakery_ratio_l1579_157960

theorem bakery_ratio (F B : ℕ) 
    (h1 : F = 10 * B)
    (h2 : F = 8 * (B + 60))
    (sugar : ℕ)
    (h3 : sugar = 3000) :
    sugar / F = 5 / 4 :=
by sorry

end NUMINAMATH_GPT_bakery_ratio_l1579_157960


namespace NUMINAMATH_GPT_vectors_parallel_eq_l1579_157987

-- Defining the problem
variables {m : ℝ}

-- Main statement
theorem vectors_parallel_eq (h : ∃ k : ℝ, (k ≠ 0) ∧ (k * 1 = m) ∧ (k * m = 2)) :
  m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_vectors_parallel_eq_l1579_157987


namespace NUMINAMATH_GPT_remainder_sum_abc_mod5_l1579_157991

theorem remainder_sum_abc_mod5 (a b c : ℕ) (h1 : a < 5) (h2 : b < 5) (h3 : c < 5)
  (h4 : a * b * c ≡ 1 [MOD 5])
  (h5 : 4 * c ≡ 3 [MOD 5])
  (h6 : 3 * b ≡ 2 + b [MOD 5]) :
  (a + b + c) % 5 = 1 :=
  sorry

end NUMINAMATH_GPT_remainder_sum_abc_mod5_l1579_157991


namespace NUMINAMATH_GPT_correct_statements_l1579_157962

-- Given the values of x and y on the parabola
def parabola (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the points on the parabola
def points_on_parabola (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 3 ∧
  parabola a b c 0 = 0 ∧
  parabola a b c 1 = -1 ∧
  parabola a b c 2 = 0 ∧
  parabola a b c 3 = 3

-- Prove the correct statements
theorem correct_statements (a b c : ℝ) (h : points_on_parabola a b c) : 
  ¬(∃ x, parabola a b c x < 0 ∧ x < 0) ∧
  parabola a b c 2 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_correct_statements_l1579_157962


namespace NUMINAMATH_GPT_find_speed_l1579_157900

theorem find_speed (v d : ℝ) (h1 : d > 0) (h2 : 1.10 * v > 0) (h3 : 84 = 2 * d / (d / v + d / (1.10 * v))) : v = 80.18 := 
sorry

end NUMINAMATH_GPT_find_speed_l1579_157900


namespace NUMINAMATH_GPT_sin_105_value_cos_75_value_trigonometric_identity_l1579_157946

noncomputable def sin_105_eq : Real := Real.sin (105 * Real.pi / 180)
noncomputable def cos_75_eq : Real := Real.cos (75 * Real.pi / 180)
noncomputable def cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq : Real := 
  Real.cos (Real.pi / 5) * Real.cos (3 * Real.pi / 10) - Real.sin (Real.pi / 5) * Real.sin (3 * Real.pi / 10)

theorem sin_105_value : sin_105_eq = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
  by sorry

theorem cos_75_value : cos_75_eq = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
  by sorry

theorem trigonometric_identity : cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq = 0 := 
  by sorry

end NUMINAMATH_GPT_sin_105_value_cos_75_value_trigonometric_identity_l1579_157946


namespace NUMINAMATH_GPT_union_of_A_and_B_l1579_157948

open Set

theorem union_of_A_and_B : 
  let A := {x : ℝ | x + 2 > 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = Real.cos x}
  A ∪ B = {z : ℝ | z > -2} := 
by
  intros
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1579_157948


namespace NUMINAMATH_GPT_women_in_third_group_l1579_157994

variables (m w : ℝ)

theorem women_in_third_group (h1 : 3 * m + 8 * w = 6 * m + 2 * w) (x : ℝ) (h2 : 2 * m + x * w = 0.5 * (3 * m + 8 * w)) :
  x = 4 :=
sorry

end NUMINAMATH_GPT_women_in_third_group_l1579_157994


namespace NUMINAMATH_GPT_neg_proposition_P_l1579_157901

theorem neg_proposition_P : 
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_proposition_P_l1579_157901


namespace NUMINAMATH_GPT_evaluate_expression_l1579_157902

def acbd (a b c d : ℝ) : ℝ := a * d - b * c

theorem evaluate_expression (x : ℝ) (h : x^2 - 3 * x + 1 = 0) :
  acbd (x + 1) (x - 2) (3 * x) (x - 1) = 1 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1579_157902


namespace NUMINAMATH_GPT_megan_initial_strawberry_jelly_beans_l1579_157979

variables (s g : ℕ)

theorem megan_initial_strawberry_jelly_beans :
  (s = 3 * g) ∧ (s - 15 = 4 * (g - 15)) → s = 135 :=
by
  sorry

end NUMINAMATH_GPT_megan_initial_strawberry_jelly_beans_l1579_157979


namespace NUMINAMATH_GPT_jelly_beans_total_l1579_157923

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end NUMINAMATH_GPT_jelly_beans_total_l1579_157923


namespace NUMINAMATH_GPT_jasmine_laps_l1579_157999

theorem jasmine_laps (x : ℕ) :
  (∀ (x : ℕ), ∃ (y : ℕ), y = 60 * x) :=
by
  sorry

end NUMINAMATH_GPT_jasmine_laps_l1579_157999


namespace NUMINAMATH_GPT_missing_number_approximately_1400_l1579_157959

theorem missing_number_approximately_1400 :
  ∃ x : ℤ, x * 54 = 75625 ∧ abs (x - Int.ofNat (75625 / 54)) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_missing_number_approximately_1400_l1579_157959


namespace NUMINAMATH_GPT_feeding_times_per_day_l1579_157983

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end NUMINAMATH_GPT_feeding_times_per_day_l1579_157983


namespace NUMINAMATH_GPT_greatest_power_of_two_factor_l1579_157918

theorem greatest_power_of_two_factor (n m : ℕ) (h1 : n = 12) (h2 : m = 8) :
  ∃ k, k = 1209 ∧ 2^k ∣ n^603 - m^402 :=
by
  sorry

end NUMINAMATH_GPT_greatest_power_of_two_factor_l1579_157918


namespace NUMINAMATH_GPT_correct_transformation_l1579_157931

-- Given conditions
variables {a b : ℝ}
variable (h : 3 * a = 4 * b)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)

-- Statement of the problem
theorem correct_transformation : (a / 4) = (b / 3) :=
sorry

end NUMINAMATH_GPT_correct_transformation_l1579_157931


namespace NUMINAMATH_GPT_A_on_curve_slope_at_A_l1579_157929

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x ^ 2

-- Define the point A on the curve f
def A : ℝ × ℝ := (2, 8)

-- Define the condition that A is on the curve f
theorem A_on_curve : A.2 = f A.1 := by
  -- * left as a proof placeholder
  sorry

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

-- State and prove the main theorem
theorem slope_at_A : (deriv f) 2 = 8 := by
  -- * left as a proof placeholder
  sorry

end NUMINAMATH_GPT_A_on_curve_slope_at_A_l1579_157929


namespace NUMINAMATH_GPT_five_to_one_ratio_to_eleven_is_fifty_five_l1579_157944

theorem five_to_one_ratio_to_eleven_is_fifty_five (y : ℚ) (h : 5 / 1 = y / 11) : y = 55 :=
by
  sorry

end NUMINAMATH_GPT_five_to_one_ratio_to_eleven_is_fifty_five_l1579_157944


namespace NUMINAMATH_GPT_terminating_decimal_expansion_l1579_157921

theorem terminating_decimal_expansion (a b : ℕ) (h : 1600 = 2^6 * 5^2) :
  (13 : ℚ) / 1600 = 65 / 1000 :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_l1579_157921


namespace NUMINAMATH_GPT_inequality_general_l1579_157976

theorem inequality_general {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_general_l1579_157976


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1579_157978

theorem geometric_sequence_first_term (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 * a 2 * a 3 = 27) (h3 : a 6 = 27) : a 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1579_157978


namespace NUMINAMATH_GPT_marc_average_speed_l1579_157945

theorem marc_average_speed 
  (d : ℝ) -- Define d as a real number representing distance
  (chantal_speed1 : ℝ := 3) -- Chantal's speed for the first half
  (chantal_speed2 : ℝ := 1.5) -- Chantal's speed for the second half
  (chantal_speed3 : ℝ := 2) -- Chantal's speed while descending
  (marc_meeting_point : ℝ := (2 / 3) * d) -- One-third point from the trailhead
  (chantal_time1 : ℝ := d / chantal_speed1) 
  (chantal_time2 : ℝ := (d / chantal_speed2))
  (chantal_time3 : ℝ := (d / 6)) -- Chantal's time for the descent from peak to one-third point
  (total_time : ℝ := chantal_time1 + chantal_time2 + chantal_time3) : 
  marc_meeting_point / total_time = 12 / 13 := 
  by 
  -- Leaving the proof as sorry to indicate where the proof would be
  sorry

end NUMINAMATH_GPT_marc_average_speed_l1579_157945


namespace NUMINAMATH_GPT_minimum_and_maximum_attendees_more_than_one_reunion_l1579_157924

noncomputable def minimum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  let total_unique_attendees := oates_attendees + hall_attendees + brown_attendees
  total_unique_attendees - total_guests

noncomputable def maximum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  oates_attendees

theorem minimum_and_maximum_attendees_more_than_one_reunion
  (total_guests oates_attendees hall_attendees brown_attendees : ℕ)
  (H1 : total_guests = 200)
  (H2 : oates_attendees = 60)
  (H3 : hall_attendees = 90)
  (H4 : brown_attendees = 80) :
  minimum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 30 ∧
  maximum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 60 :=
by
  sorry

end NUMINAMATH_GPT_minimum_and_maximum_attendees_more_than_one_reunion_l1579_157924


namespace NUMINAMATH_GPT_am_gm_example_l1579_157968

theorem am_gm_example {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 :=
sorry

end NUMINAMATH_GPT_am_gm_example_l1579_157968


namespace NUMINAMATH_GPT_candy_cases_total_l1579_157972

theorem candy_cases_total
  (choco_cases lolli_cases : ℕ)
  (h1 : choco_cases = 25)
  (h2 : lolli_cases = 55) : 
  (choco_cases + lolli_cases) = 80 := by
-- The proof is omitted as requested.
sorry

end NUMINAMATH_GPT_candy_cases_total_l1579_157972


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1579_157915

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (A B : ℕ → ℝ)
  (hA9 : A 9 = (a 5) ^ 9)
  (hB9 : B 9 = (b 5) ^ 9)
  (h_ratio : a 5 / b 5 = 2) :
  (A 9 / B 9) = 512 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1579_157915


namespace NUMINAMATH_GPT_part1_part2_l1579_157941

noncomputable def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 0  -- since there is no a_0 (we use ℕ*), we set it to 0
  | 1     => 1/3
  | n + 1 => seq n + (seq n) ^ 2 / (n : ℚ) ^ 2

theorem part1 (n : ℕ) (h : 0 < n) :
  seq n < seq (n + 1) ∧ seq (n + 1) < 1 :=
sorry

theorem part2 (n : ℕ) (h : 0 < n) :
  seq n > 1/2 - 1/(4 * n) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1579_157941


namespace NUMINAMATH_GPT_sin_3theta_over_sin_theta_l1579_157967

theorem sin_3theta_over_sin_theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 2) : 
  Real.sin (3 * θ) / Real.sin θ = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_3theta_over_sin_theta_l1579_157967


namespace NUMINAMATH_GPT_lines_intersect_at_l1579_157995

theorem lines_intersect_at :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 7 * y = -3 * x - 4 ∧ x = 54 / 5 ∧ y = -26 / 5 := 
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_l1579_157995


namespace NUMINAMATH_GPT_trapezium_height_l1579_157966

theorem trapezium_height (a b A h : ℝ) (ha : a = 12) (hb : b = 16) (ha_area : A = 196) :
  (A = 0.5 * (a + b) * h) → h = 14 :=
by
  intros h_eq
  rw [ha, hb, ha_area] at h_eq
  sorry

end NUMINAMATH_GPT_trapezium_height_l1579_157966


namespace NUMINAMATH_GPT_incorrect_statement_among_given_options_l1579_157913

theorem incorrect_statement_among_given_options :
  (∀ (b h : ℝ), 3 * (b * h) = (3 * b) * h) ∧
  (∀ (b h : ℝ), 3 * (1 / 2 * b * h) = 1 / 2 * b * (3 * h)) ∧
  (∀ (π r : ℝ), 9 * (π * r * r) ≠ (π * (3 * r) * (3 * r))) ∧
  (∀ (a b : ℝ), (3 * a) / (2 * b) ≠ a / b) ∧
  (∀ (x : ℝ), x < 0 → 3 * x < x) →
  false :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_among_given_options_l1579_157913


namespace NUMINAMATH_GPT_hypotenuse_length_l1579_157993

-- Define the conditions
def right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- State the theorem using the conditions and correct answer
theorem hypotenuse_length : right_triangle 20 21 29 :=
by
  -- To be filled in by proof steps
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1579_157993


namespace NUMINAMATH_GPT_purchase_price_of_furniture_l1579_157906

theorem purchase_price_of_furniture (marked_price discount_rate profit_rate : ℝ) 
(h_marked_price : marked_price = 132) 
(h_discount_rate : discount_rate = 0.1)
(h_profit_rate : profit_rate = 0.1)
: ∃ a : ℝ, (marked_price * (1 - discount_rate) - a = profit_rate * a) ∧ a = 108 := by
  sorry

end NUMINAMATH_GPT_purchase_price_of_furniture_l1579_157906


namespace NUMINAMATH_GPT_parabola_y_coordinate_l1579_157937

theorem parabola_y_coordinate (x y : ℝ) :
  x^2 = 4 * y ∧ (x - 0)^2 + (y - 1)^2 = 16 → y = 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_y_coordinate_l1579_157937


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l1579_157990

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : a 2 + a 4 = 8) (h_seq : a 2 + a 4 = 2 * a 3) :
  a 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l1579_157990


namespace NUMINAMATH_GPT_f_at_one_f_decreasing_f_min_on_interval_l1579_157955

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_defined : ∀ x, 0 < x → ∃ y, f y = y
axiom f_eq : ∀ x1 x2, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2
axiom f_neg : ∀ x, 1 < x → f x < 0

-- Proof statements
theorem f_at_one : f 1 = 0 := sorry

theorem f_decreasing : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2 := sorry

axiom f_at_three : f 3 = -1

theorem f_min_on_interval : ∀ x, 2 ≤ x ∧ x ≤ 9 → f x ≥ -2 := sorry

end NUMINAMATH_GPT_f_at_one_f_decreasing_f_min_on_interval_l1579_157955


namespace NUMINAMATH_GPT_tel_aviv_rain_days_l1579_157912

-- Define the conditions
def chance_of_rain : ℝ := 0.5
def days_considered : ℕ := 6
def given_probability : ℝ := 0.234375

-- Helper function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function P(X = k)
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- The main theorem to prove
theorem tel_aviv_rain_days :
  ∃ k, binomial_probability days_considered k chance_of_rain = given_probability ∧ k = 2 := by
  sorry

end NUMINAMATH_GPT_tel_aviv_rain_days_l1579_157912


namespace NUMINAMATH_GPT_radius_inscribed_sphere_quadrilateral_pyramid_l1579_157935

noncomputable def radius_of_inscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt 5 - 1) / 4

theorem radius_inscribed_sphere_quadrilateral_pyramid (a : ℝ) :
  r = radius_of_inscribed_sphere a :=
by
  -- problem conditions:
  -- side of the base a
  -- height a
  -- result: r = a * (Real.sqrt 5 - 1) / 4
  sorry

end NUMINAMATH_GPT_radius_inscribed_sphere_quadrilateral_pyramid_l1579_157935


namespace NUMINAMATH_GPT_maximize_revenue_l1579_157975

def revenue_function (p : ℝ) : ℝ :=
  p * (200 - 6 * p)

theorem maximize_revenue :
  ∃ (p : ℝ), (p ≤ 30) ∧ (∀ q : ℝ, (q ≤ 30) → revenue_function p ≥ revenue_function q) ∧ p = 50 / 3 :=
by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l1579_157975


namespace NUMINAMATH_GPT_quadratic_function_coefficient_not_zero_l1579_157958

theorem quadratic_function_coefficient_not_zero (m : ℝ) : (∀ x : ℝ, (m-2)*x^2 + 2*x - 3 ≠ 0) → m ≠ 2 :=
by
  intro h
  by_contra h1
  exact sorry

end NUMINAMATH_GPT_quadratic_function_coefficient_not_zero_l1579_157958


namespace NUMINAMATH_GPT_possible_vertex_angles_of_isosceles_triangle_l1579_157907

def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (β = γ) ∨ (γ = α)

def altitude_half_side (α β γ a b c : ℝ) : Prop :=
  (a = α / 2) ∨ (b = β / 2) ∨ (c = γ / 2)

theorem possible_vertex_angles_of_isosceles_triangle (α β γ a b c : ℝ) :
  isosceles_triangle α β γ →
  altitude_half_side α β γ a b c →
  α = 30 ∨ α = 120 ∨ α = 150 :=
by
  sorry

end NUMINAMATH_GPT_possible_vertex_angles_of_isosceles_triangle_l1579_157907


namespace NUMINAMATH_GPT_maximum_value_of_d_l1579_157910

theorem maximum_value_of_d 
  (d e : ℕ) 
  (h1 : 0 ≤ d ∧ d < 10) 
  (h2: 0 ≤ e ∧ e < 10) 
  (h3 : (18 + d + e) % 3 = 0) 
  (h4 : (15 - (d + e)) % 11 = 0) 
  : d ≤ 0 := 
sorry

end NUMINAMATH_GPT_maximum_value_of_d_l1579_157910


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l1579_157952

-- Define conditions
def avg_height_students := 180
def avg_height_female := 170
def avg_height_male := 185

-- This is the math proof problem statement
theorem ratio_of_men_to_women (M W : ℕ) (h1 : (M * avg_height_male + W * avg_height_female) = (M + W) * avg_height_students) : 
  M / W = 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l1579_157952


namespace NUMINAMATH_GPT_cubic_root_relationship_l1579_157949

theorem cubic_root_relationship 
  (r : ℝ) (h : r^3 - r + 3 = 0) : 
  (r^2)^3 - 2 * (r^2)^2 + (r^2) - 9 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_root_relationship_l1579_157949


namespace NUMINAMATH_GPT_find_divisor_l1579_157961

-- Definitions from the conditions
def remainder : ℤ := 8
def quotient : ℤ := 43
def dividend : ℤ := 997
def is_prime (n : ℤ) : Prop := n ≠ 1 ∧ (∀ d : ℤ, d ∣ n → d = 1 ∨ d = n)

-- The proof problem statement
theorem find_divisor (d : ℤ) 
  (hd : is_prime d) 
  (hdiv : dividend = (d * quotient) + remainder) : 
  d = 23 := 
sorry

end NUMINAMATH_GPT_find_divisor_l1579_157961


namespace NUMINAMATH_GPT_compare_y1_y2_l1579_157916

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Define the points
def y1 := f 1
def y2 := f 3

-- The theorem to be proved
theorem compare_y1_y2 : y1 > y2 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_compare_y1_y2_l1579_157916


namespace NUMINAMATH_GPT_dave_apps_added_l1579_157938

theorem dave_apps_added (initial_apps : ℕ) (total_apps_after_adding : ℕ) (apps_added : ℕ) 
  (h1 : initial_apps = 17) (h2 : total_apps_after_adding = 18) 
  (h3 : total_apps_after_adding = initial_apps + apps_added) : 
  apps_added = 1 := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_dave_apps_added_l1579_157938


namespace NUMINAMATH_GPT_total_points_scored_l1579_157920

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_points_scored_l1579_157920


namespace NUMINAMATH_GPT_sum_complex_l1579_157909

-- Define the given complex numbers
def z1 : ℂ := ⟨2, 5⟩
def z2 : ℂ := ⟨3, -7⟩

-- State the theorem to prove the sum
theorem sum_complex : z1 + z2 = ⟨5, -2⟩ :=
by
  sorry

end NUMINAMATH_GPT_sum_complex_l1579_157909


namespace NUMINAMATH_GPT_outer_circle_increase_l1579_157917

theorem outer_circle_increase : 
  let R_o := 6
  let R_i := 4
  let R_i_new := (3 : ℝ)  -- 4 * (3/4)
  let A_original := 20 * Real.pi  -- π * (6^2 - 4^2)
  let A_new := 72 * Real.pi  -- 3.6 * A_original
  ∃ (x : ℝ), 
    let R_o_new := R_o * (1 + x / 100)
    π * R_o_new^2 - π * R_i_new^2 = A_new →
    x = 50 := 
sorry

end NUMINAMATH_GPT_outer_circle_increase_l1579_157917


namespace NUMINAMATH_GPT_minimum_toothpicks_to_remove_l1579_157981

-- Conditions
def number_of_toothpicks : ℕ := 60
def largest_triangle_side : ℕ := 3
def smallest_triangle_side : ℕ := 1

-- Problem Statement
theorem minimum_toothpicks_to_remove (toothpicks_total : ℕ) (largest_side : ℕ) (smallest_side : ℕ) 
  (h1 : toothpicks_total = 60) 
  (h2 : largest_side = 3) 
  (h3 : smallest_side = 1) : 
  ∃ n : ℕ, n = 20 := by
  sorry

end NUMINAMATH_GPT_minimum_toothpicks_to_remove_l1579_157981


namespace NUMINAMATH_GPT_solve_for_x_l1579_157936

variable {x : ℝ}

def is_positive (x : ℝ) : Prop := x > 0

def area_of_triangle_is_150 (x : ℝ) : Prop :=
  let base := 2 * x
  let height := 3 * x
  (1/2) * base * height = 150

theorem solve_for_x (hx : is_positive x) (ha : area_of_triangle_is_150 x) : x = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1579_157936


namespace NUMINAMATH_GPT_total_cost_of_vitamins_l1579_157977

-- Definitions based on the conditions
def original_price : ℝ := 15.00
def discount_percentage : ℝ := 0.20
def coupon_value : ℝ := 2.00
def num_coupons : ℕ := 3
def num_bottles : ℕ := 3

-- Lean statement to prove the final cost
theorem total_cost_of_vitamins
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (coupon_value : ℝ)
  (num_coupons : ℕ)
  (num_bottles : ℕ)
  (discounted_price_per_bottle : ℝ := original_price * (1 - discount_percentage))
  (total_coupon_value : ℝ := coupon_value * num_coupons)
  (total_cost_before_coupons : ℝ := discounted_price_per_bottle * num_bottles) :
  (total_cost_before_coupons - total_coupon_value) = 30.00 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_vitamins_l1579_157977


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1579_157930

variable {α : Type*} [LinearOrder α] [Field α]

def is_geometric_sequence (a : ℕ → α) :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

theorem geometric_sequence_problem (a : ℕ → α) (r : α) (h1 : a 1 = 1) (h2 : is_geometric_sequence a) (h3 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1579_157930


namespace NUMINAMATH_GPT_min_value_of_Box_l1579_157989

theorem min_value_of_Box (c d : ℤ) (hcd : c * d = 42) (distinct_values : c ≠ d ∧ c ≠ 85 ∧ d ≠ 85) :
  ∃ (Box : ℤ), (c^2 + d^2 = Box) ∧ (Box = 85) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_Box_l1579_157989


namespace NUMINAMATH_GPT_sum_of_interior_edges_l1579_157942

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_edges_l1579_157942


namespace NUMINAMATH_GPT_option_b_is_same_type_l1579_157956

def polynomial_same_type (p1 p2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y, (p1 x y = 1 → p2 x y = 1) ∧ (p2 x y = 1 → p1 x y = 1)

def ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0

def a_squared_b (a b : ℕ) := if a = 2 ∧ b = 1 then 1 else 0
def negative_two_ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0
def ab (a b : ℕ) := if a = 1 ∧ b = 1 then 1 else 0
def ab_squared_c (a b c : ℕ) := if a = 1 ∧ b = 2 ∧ c = 1 then 1 else 0

theorem option_b_is_same_type : polynomial_same_type ab_squared negative_two_ab_squared :=
by
  sorry

end NUMINAMATH_GPT_option_b_is_same_type_l1579_157956


namespace NUMINAMATH_GPT_basketball_cards_per_box_l1579_157997

-- Given conditions
def num_basketball_boxes : ℕ := 9
def num_football_boxes := num_basketball_boxes - 3
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255
def total_football_cards := num_football_boxes * cards_per_football_box

-- We want to prove that the number of cards in each basketball card box is 15
theorem basketball_cards_per_box :
  (total_cards - total_football_cards) / num_basketball_boxes = 15 := by
  sorry

end NUMINAMATH_GPT_basketball_cards_per_box_l1579_157997


namespace NUMINAMATH_GPT_box_office_scientific_notation_l1579_157953

def billion : ℝ := 10^9
def box_office_revenue : ℝ := 57.44 * billion
def scientific_notation (n : ℝ) : ℝ × ℝ := (5.744, 10^10)

theorem box_office_scientific_notation :
  scientific_notation box_office_revenue = (5.744, 10^10) :=
by
  sorry

end NUMINAMATH_GPT_box_office_scientific_notation_l1579_157953
