import Mathlib

namespace NUMINAMATH_GPT_relationship_in_size_l2078_207827

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 2.1
noncomputable def c : ℝ := Real.log (1.5) / Real.log (2)

theorem relationship_in_size : b > a ∧ a > c := by
  sorry

end NUMINAMATH_GPT_relationship_in_size_l2078_207827


namespace NUMINAMATH_GPT_calculate_total_travel_time_l2078_207874

/-- The total travel time, including stops, from the first station to the last station. -/
def total_travel_time (d1 d2 d3 : ℕ) (s1 s2 s3 : ℕ) (t1 t2 : ℕ) : ℚ :=
  let leg1_time := d1 / s1
  let stop1_time := t1 / 60
  let leg2_time := d2 / s2
  let stop2_time := t2 / 60
  let leg3_time := d3 / s3
  leg1_time + stop1_time + leg2_time + stop2_time + leg3_time

/-- Proof that total travel time is 2 hours and 22.5 minutes. -/
theorem calculate_total_travel_time :
  total_travel_time 30 40 50 60 40 80 10 5 = 2.375 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_travel_time_l2078_207874


namespace NUMINAMATH_GPT_arithmetic_mean_of_roots_l2078_207831

-- Definitions corresponding to the conditions
def quadratic_eqn (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The term statement for the quadratic equation mean
theorem arithmetic_mean_of_roots : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 1 → (∃ (x1 x2 : ℝ), quadratic_eqn a b c x1 ∧ quadratic_eqn a b c x2 ∧ -4 / 2 = -2) :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_roots_l2078_207831


namespace NUMINAMATH_GPT_total_tickets_sold_l2078_207816

def price_adult_ticket : ℕ := 7
def price_child_ticket : ℕ := 4
def total_revenue : ℕ := 5100
def adult_tickets_sold : ℕ := 500

theorem total_tickets_sold : 
  ∃ (child_tickets_sold : ℕ), 
    price_adult_ticket * adult_tickets_sold + price_child_ticket * child_tickets_sold = total_revenue ∧
    adult_tickets_sold + child_tickets_sold = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2078_207816


namespace NUMINAMATH_GPT_no_three_partition_exists_l2078_207835

/-- Define the partitioning property for three subsets -/
def partitions (A B C : Set ℤ) : Prop :=
  ∀ n : ℤ, (n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ (n ∈ A ↔ n-50 ∈ B ∧ n+1987 ∈ C) ∧ (n-50 ∈ A ∨ n-50 ∈ B ∨ n-50 ∈ C) ∧ (n-50 ∈ B ↔ n-50-50 ∈ A ∧ n-50+1987 ∈ C) ∧ (n+1987 ∈ A ∨ n+1987 ∈ B ∨ n+1987 ∈ C) ∧ (n+1987 ∈ C ↔ n+1987-50 ∈ A ∧ n+1987+1987 ∈ B)

/-- The main theorem stating that no such partition is possible -/
theorem no_three_partition_exists :
  ¬∃ A B C : Set ℤ, partitions A B C :=
sorry

end NUMINAMATH_GPT_no_three_partition_exists_l2078_207835


namespace NUMINAMATH_GPT_largest_value_l2078_207805

theorem largest_value :
  let A := 1/2
  let B := 1/3 + 1/4
  let C := 1/4 + 1/5 + 1/6
  let D := 1/5 + 1/6 + 1/7 + 1/8
  let E := 1/6 + 1/7 + 1/8 + 1/9 + 1/10
  E > A ∧ E > B ∧ E > C ∧ E > D := by
sorry

end NUMINAMATH_GPT_largest_value_l2078_207805


namespace NUMINAMATH_GPT_weighted_average_yield_l2078_207825

-- Define the conditions
def face_value_A : ℝ := 1000
def market_price_A : ℝ := 1200
def yield_A : ℝ := 0.18

def face_value_B : ℝ := 1000
def market_price_B : ℝ := 800
def yield_B : ℝ := 0.22

def face_value_C : ℝ := 1000
def market_price_C : ℝ := 1000
def yield_C : ℝ := 0.15

def investment_A : ℝ := 5000
def investment_B : ℝ := 3000
def investment_C : ℝ := 2000

-- Prove the weighted average yield
theorem weighted_average_yield :
  (investment_A + investment_B + investment_C) = 10000 →
  ((investment_A / (investment_A + investment_B + investment_C)) * yield_A +
   (investment_B / (investment_A + investment_B + investment_C)) * yield_B +
   (investment_C / (investment_A + investment_B + investment_C)) * yield_C) = 0.186 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_yield_l2078_207825


namespace NUMINAMATH_GPT_rectangular_area_l2078_207883

theorem rectangular_area (length width : ℝ) (h₁ : length = 0.4) (h₂ : width = 0.22) :
  (length * width = 0.088) :=
by sorry

end NUMINAMATH_GPT_rectangular_area_l2078_207883


namespace NUMINAMATH_GPT_compare_expression_l2078_207830

variable (m x : ℝ)

theorem compare_expression : x^2 - x + 1 > -2 * m^2 - 2 * m * x := 
sorry

end NUMINAMATH_GPT_compare_expression_l2078_207830


namespace NUMINAMATH_GPT_older_brother_catches_up_l2078_207840

theorem older_brother_catches_up :
  ∃ (x : ℝ), 0 ≤ x ∧ 6 * x = 2 + 2 * x ∧ x + 1 < 1.75 :=
by
  sorry

end NUMINAMATH_GPT_older_brother_catches_up_l2078_207840


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2078_207833

variable {α : Type*} [LinearOrder α]

def is_decreasing (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (domain_cond : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → x ∈ Set.Ioo (-2 : ℝ) 2)
  : { x | x > 0 ∧ x < 1 } = { x | f x > f (2 - x) } :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_of_inequality_l2078_207833


namespace NUMINAMATH_GPT_quadratic_difference_square_l2078_207889

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_GPT_quadratic_difference_square_l2078_207889


namespace NUMINAMATH_GPT_james_needs_more_marbles_l2078_207837

def number_of_additional_marbles (friends marbles : Nat) : Nat :=
  let required_marbles := (friends * (friends + 1)) / 2
  (if marbles < required_marbles then required_marbles - marbles else 0)

theorem james_needs_more_marbles :
  number_of_additional_marbles 15 80 = 40 := by
  sorry

end NUMINAMATH_GPT_james_needs_more_marbles_l2078_207837


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_is_4_l2078_207859

theorem common_ratio_of_geometric_sequence_is_4 
  (a_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ) 
  (d : ℝ) 
  (h₁ : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h₂ : d ≠ 0)
  (h₃ : (a_n 3)^2 = (a_n 2) * (a_n 7)) :
  b_n 2 / b_n 1 = 4 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_is_4_l2078_207859


namespace NUMINAMATH_GPT_largest_odd_not_sum_of_three_distinct_composites_l2078_207834

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem largest_odd_not_sum_of_three_distinct_composites :
  ∀ n : ℕ, is_odd n → (¬ ∃ (a b c : ℕ), is_composite a ∧ is_composite b ∧ is_composite c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a + b + c) → n ≤ 17 :=
by
  sorry

end NUMINAMATH_GPT_largest_odd_not_sum_of_three_distinct_composites_l2078_207834


namespace NUMINAMATH_GPT_perpendicular_line_through_A_l2078_207822

variable (m : ℝ)

-- Conditions
def line1 (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def line2 (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0
def pointA : ℝ × ℝ := (3, 2)

-- Question and proof
theorem perpendicular_line_through_A (h_parallel : ∃ x y, line1 m x y ∧ line2 m x y) :
  ∃ (t : ℝ), ∀ (x y : ℝ), (y = 2 * x + t) ↔ (2 * x - y - 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_A_l2078_207822


namespace NUMINAMATH_GPT_sum_of_roots_eq_14_l2078_207898

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_14_l2078_207898


namespace NUMINAMATH_GPT_town_population_l2078_207820

variable (P₀ P₁ P₂ : ℝ)

def population_two_years_ago (P₀ : ℝ) : Prop := P₀ = 800

def first_year_increase (P₀ P₁ : ℝ) : Prop := P₁ = P₀ * 1.25

def second_year_increase (P₁ P₂ : ℝ) : Prop := P₂ = P₁ * 1.15

theorem town_population 
  (h₀ : population_two_years_ago P₀)
  (h₁ : first_year_increase P₀ P₁)
  (h₂ : second_year_increase P₁ P₂) : 
  P₂ = 1150 := 
sorry

end NUMINAMATH_GPT_town_population_l2078_207820


namespace NUMINAMATH_GPT_jogging_path_diameter_l2078_207821

theorem jogging_path_diameter 
  (d_pond : ℝ)
  (w_flowerbed : ℝ)
  (w_jogging_path : ℝ)
  (h_pond : d_pond = 20)
  (h_flowerbed : w_flowerbed = 10)
  (h_jogging_path : w_jogging_path = 12) :
  2 * (d_pond / 2 + w_flowerbed + w_jogging_path) = 64 :=
by
  sorry

end NUMINAMATH_GPT_jogging_path_diameter_l2078_207821


namespace NUMINAMATH_GPT_secret_code_count_l2078_207887

-- Conditions
def num_colors : ℕ := 8
def num_slots : ℕ := 5

-- The proof statement
theorem secret_code_count : (num_colors ^ num_slots) = 32768 := by
  sorry

end NUMINAMATH_GPT_secret_code_count_l2078_207887


namespace NUMINAMATH_GPT_min_value_of_expression_l2078_207832

noncomputable def minExpression (x : ℝ) : ℝ := (15 - x) * (14 - x) * (15 + x) * (14 + x)

theorem min_value_of_expression : ∀ x : ℝ, ∃ m : ℝ, (m ≤ minExpression x) ∧ (m = -142.25) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2078_207832


namespace NUMINAMATH_GPT_parabola_equation_l2078_207852

theorem parabola_equation (a b c d e f : ℤ)
  (h1 : a = 0 )    -- The equation should have no x^2 term
  (h2 : b = 0 )    -- The equation should have no xy term
  (h3 : c > 0)     -- The coefficient of y^2 should be positive
  (h4 : d = -2)    -- The coefficient of x in the final form should be -2
  (h5 : e = -8)    -- The coefficient of y in the final form should be -8
  (h6 : f = 16)    -- The constant term in the final form should be 16
  (pass_through : (2 : ℤ) = k * (6 - 4) ^ 2)
  (vertex : (0 : ℤ) = k * (sym_axis - 4) ^ 2)
  (symmetry_axis_parallel_x : True)
  (vertex_on_y_axis : True):
  ax^2 + bxy + cy^2 + dx + ey + f = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l2078_207852


namespace NUMINAMATH_GPT_excess_percentage_l2078_207847

theorem excess_percentage (x : ℝ) 
  (L W : ℝ) (hL : L > 0) (hW : W > 0) 
  (h1 : L * (1 + x / 100) * W * 0.96 = L * W * 1.008) : 
  x = 5 :=
by sorry

end NUMINAMATH_GPT_excess_percentage_l2078_207847


namespace NUMINAMATH_GPT_complement_union_l2078_207868

namespace SetComplement

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union :
  U \ (A ∪ B) = {1, 2, 6} := by
  sorry

end SetComplement

end NUMINAMATH_GPT_complement_union_l2078_207868


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l2078_207882

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l2078_207882


namespace NUMINAMATH_GPT_sum_of_abc_l2078_207838

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (eq1 : a^2 + b * c = 115) (eq2 : b^2 + a * c = 127) (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 := by
  sorry

end NUMINAMATH_GPT_sum_of_abc_l2078_207838


namespace NUMINAMATH_GPT_convert_mixed_decimals_to_fractions_l2078_207870

theorem convert_mixed_decimals_to_fractions :
  (4.26 = 4 + 13/50) ∧
  (1.15 = 1 + 3/20) ∧
  (3.08 = 3 + 2/25) ∧
  (2.37 = 2 + 37/100) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_convert_mixed_decimals_to_fractions_l2078_207870


namespace NUMINAMATH_GPT_scientific_notation_of_one_point_six_million_l2078_207848

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_one_point_six_million_l2078_207848


namespace NUMINAMATH_GPT_continuity_of_f_at_3_l2078_207880

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3*x^2 + 2*x - 4 else b*x + 7

theorem continuity_of_f_at_3 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x b - f 3 b) < ε) ↔ b = 22 / 3 :=
by
  sorry

end NUMINAMATH_GPT_continuity_of_f_at_3_l2078_207880


namespace NUMINAMATH_GPT_katrina_cookies_left_l2078_207814

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end NUMINAMATH_GPT_katrina_cookies_left_l2078_207814


namespace NUMINAMATH_GPT_solution_to_quadratic_inequality_l2078_207867

theorem solution_to_quadratic_inequality 
  (a : ℝ)
  (h : ∀ x : ℝ, x^2 - a * x + 1 < 0 ↔ (1 / 2 : ℝ) < x ∧ x < 2) :
  a = 5 / 2 :=
sorry

end NUMINAMATH_GPT_solution_to_quadratic_inequality_l2078_207867


namespace NUMINAMATH_GPT_rate_of_dividend_is_12_l2078_207886

-- Defining the conditions
def total_investment : ℝ := 4455
def price_per_share : ℝ := 8.25
def annual_income : ℝ := 648
def face_value_per_share : ℝ := 10

-- Expected rate of dividend
def expected_rate_of_dividend : ℝ := 12

-- The proof problem statement: Prove that the rate of dividend is 12% given the conditions.
theorem rate_of_dividend_is_12 :
  ∃ (r : ℝ), r = 12 ∧ annual_income = 
    (total_investment / price_per_share) * (r / 100) * face_value_per_share :=
by 
  use 12
  sorry

end NUMINAMATH_GPT_rate_of_dividend_is_12_l2078_207886


namespace NUMINAMATH_GPT_exterior_angle_BAC_l2078_207808

theorem exterior_angle_BAC (square_octagon_coplanar : Prop) (common_side_AD : Prop) : 
    angle_BAC = 135 :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_BAC_l2078_207808


namespace NUMINAMATH_GPT_area_of_rectangle_l2078_207836

-- Define the given conditions
def side_length_of_square (s : ℝ) (ABCD : ℝ) : Prop :=
  ABCD = 4 * s^2

def perimeter_of_rectangle (s : ℝ) (perimeter : ℝ): Prop :=
  perimeter = 8 * s

-- Statement of the proof problem
theorem area_of_rectangle (s perimeter_area : ℝ) (h_perimeter : perimeter_of_rectangle s 160) :
  side_length_of_square s 1600 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2078_207836


namespace NUMINAMATH_GPT_convert_1623_to_base7_l2078_207849

theorem convert_1623_to_base7 :
  ∃ a b c d : ℕ, 1623 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
  a = 4 ∧ b = 5 ∧ c = 0 ∧ d = 6 :=
by
  sorry

end NUMINAMATH_GPT_convert_1623_to_base7_l2078_207849


namespace NUMINAMATH_GPT_exists_1998_distinct_natural_numbers_l2078_207806

noncomputable def exists_1998_distinct_numbers : Prop :=
  ∃ (s : Finset ℕ), s.card = 1998 ∧
    (∀ {x y : ℕ}, x ∈ s → y ∈ s → x ≠ y → (x * y) % ((x - y) ^ 2) = 0)

theorem exists_1998_distinct_natural_numbers : exists_1998_distinct_numbers :=
by
  sorry

end NUMINAMATH_GPT_exists_1998_distinct_natural_numbers_l2078_207806


namespace NUMINAMATH_GPT_min_adjacent_seat_occupation_l2078_207881

def minOccupiedSeats (n : ℕ) : ℕ :=
  n / 3

theorem min_adjacent_seat_occupation (n : ℕ) (h : n = 150) :
  minOccupiedSeats n = 50 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_min_adjacent_seat_occupation_l2078_207881


namespace NUMINAMATH_GPT_range_of_k_l2078_207853

theorem range_of_k (k : ℝ) (H : ∀ x : ℤ, |(x : ℝ) - 1| < k * x ↔ x ∈ ({1, 2, 3} : Set ℤ)) : 
  (2 / 3 : ℝ) < k ∧ k ≤ (3 / 4 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2078_207853


namespace NUMINAMATH_GPT_g_sqrt_45_l2078_207877

noncomputable def g (x : ℝ) : ℝ :=
if x % 1 = 0 then 7 * x + 6 else ⌊x⌋ + 7

theorem g_sqrt_45 : g (Real.sqrt 45) = 13 := by
  sorry

end NUMINAMATH_GPT_g_sqrt_45_l2078_207877


namespace NUMINAMATH_GPT_cost_price_per_meter_l2078_207884

theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (total_cost_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5)
  (h4 : total_cost_price = selling_price + total_meters * loss_per_meter)
  (h5 : cost_price_per_meter = total_cost_price / total_meters) :
  cost_price_per_meter = 50 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l2078_207884


namespace NUMINAMATH_GPT_compare_polynomials_l2078_207826

theorem compare_polynomials (x : ℝ) : 2 * x^2 - 2 * x + 1 > x^2 - 2 * x := 
by
  sorry

end NUMINAMATH_GPT_compare_polynomials_l2078_207826


namespace NUMINAMATH_GPT_joan_bought_72_eggs_l2078_207876

def dozen := 12
def joan_eggs (dozens: Nat) := dozens * dozen

theorem joan_bought_72_eggs : joan_eggs 6 = 72 :=
by
  sorry

end NUMINAMATH_GPT_joan_bought_72_eggs_l2078_207876


namespace NUMINAMATH_GPT_correct_calculation_l2078_207803

theorem correct_calculation (x : ℤ) (h : 7 * (x + 24) / 5 = 70) :
  (5 * x + 24) / 7 = 22 :=
sorry

end NUMINAMATH_GPT_correct_calculation_l2078_207803


namespace NUMINAMATH_GPT_height_of_taller_tree_l2078_207809

theorem height_of_taller_tree 
  (h : ℝ) 
  (ratio_condition : (h - 20) / h = 2 / 3) : 
  h = 60 := 
by 
  sorry

end NUMINAMATH_GPT_height_of_taller_tree_l2078_207809


namespace NUMINAMATH_GPT_chord_slope_range_l2078_207811

theorem chord_slope_range (x1 y1 x2 y2 x0 y0 : ℝ) (h1 : x1^2 + (y1^2)/4 = 1) (h2 : x2^2 + (y2^2)/4 = 1)
  (h3 : x0 = (x1 + x2) / 2) (h4 : y0 = (y1 + y2) / 2)
  (h5 : x0 = 1/2) (h6 : 1/2 ≤ y0 ∧ y0 ≤ 1) :
  -4 ≤ (-2 / y0) ∧ -2 ≤ (-2 / y0) :=
by
  sorry

end NUMINAMATH_GPT_chord_slope_range_l2078_207811


namespace NUMINAMATH_GPT_intersection_is_N_l2078_207890

-- Define the sets M and N as given in the problem
def M := {x : ℝ | x > 0}
def N := {x : ℝ | Real.log x > 0}

-- State the theorem for the intersection of M and N
theorem intersection_is_N : (M ∩ N) = N := 
  by 
    sorry

end NUMINAMATH_GPT_intersection_is_N_l2078_207890


namespace NUMINAMATH_GPT_polynomial_multiplication_correct_l2078_207807

noncomputable def polynomial_expansion : Polynomial ℤ :=
  (Polynomial.C (3 : ℤ) * Polynomial.X ^ 3 + Polynomial.C (4 : ℤ) * Polynomial.X ^ 2 - Polynomial.C (8 : ℤ) * Polynomial.X - Polynomial.C (5 : ℤ)) *
  (Polynomial.C (2 : ℤ) * Polynomial.X ^ 4 - Polynomial.C (3 : ℤ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℤ))

theorem polynomial_multiplication_correct :
  polynomial_expansion = Polynomial.C (6 : ℤ) * Polynomial.X ^ 7 +
                         Polynomial.C (12 : ℤ) * Polynomial.X ^ 6 -
                         Polynomial.C (25 : ℤ) * Polynomial.X ^ 5 -
                         Polynomial.C (20 : ℤ) * Polynomial.X ^ 4 +
                         Polynomial.C (34 : ℤ) * Polynomial.X ^ 2 -
                         Polynomial.C (8 : ℤ) * Polynomial.X -
                         Polynomial.C (5 : ℤ) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_correct_l2078_207807


namespace NUMINAMATH_GPT_impossible_to_place_50_pieces_on_torus_grid_l2078_207812

theorem impossible_to_place_50_pieces_on_torus_grid :
  ¬ (∃ (a b c x y z : ℕ),
    a + b + c = 50 ∧
    2 * a ≤ x ∧ x ≤ 2 * b ∧
    2 * b ≤ y ∧ y ≤ 2 * c ∧
    2 * c ≤ z ∧ z ≤ 2 * a) :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_place_50_pieces_on_torus_grid_l2078_207812


namespace NUMINAMATH_GPT_probability_blue_given_not_red_l2078_207858

theorem probability_blue_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let blue_balls := 10
  let non_red_balls := yellow_balls + blue_balls
  let blue_given_not_red := (blue_balls : ℚ) / non_red_balls
  blue_given_not_red = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_probability_blue_given_not_red_l2078_207858


namespace NUMINAMATH_GPT_percentage_problem_l2078_207878

noncomputable def percentage_of_value (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (y / x) * 100

theorem percentage_problem :
  percentage_of_value 2348 (528.0642570281125 * 4.98) = 112 := 
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l2078_207878


namespace NUMINAMATH_GPT_total_number_of_coins_l2078_207800

theorem total_number_of_coins (num_5c : Nat) (num_10c : Nat) (h1 : num_5c = 16) (h2 : num_10c = 16) : num_5c + num_10c = 32 := by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l2078_207800


namespace NUMINAMATH_GPT_length_of_platform_l2078_207897

-- Definitions for conditions
def train_length : ℕ := 300
def time_cross_platform : ℕ := 39
def time_cross_signal : ℕ := 12

-- Speed calculation
def train_speed := train_length / time_cross_signal

-- Total distance calculation while crossing the platform
def total_distance := train_speed * time_cross_platform

-- Length of the platform
def platform_length : ℕ := total_distance - train_length

-- Theorem stating the length of the platform
theorem length_of_platform :
  platform_length = 675 := by
  sorry

end NUMINAMATH_GPT_length_of_platform_l2078_207897


namespace NUMINAMATH_GPT_jason_books_l2078_207817

theorem jason_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 45 → num_shelves = 7 → total_books = books_per_shelf * num_shelves → total_books = 315 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_jason_books_l2078_207817


namespace NUMINAMATH_GPT_obrien_hats_theorem_l2078_207854

-- Define the number of hats Fire Chief Simpson has.
def simpson_hats : ℕ := 15

-- Define the number of hats Policeman O'Brien had before any hats were stolen.
def obrien_hats_before (simpson_hats : ℕ) : ℕ := 2 * simpson_hats + 5

-- Define the number of hats Policeman O'Brien has now, after x hats were stolen.
def obrien_hats_now (x : ℕ) : ℕ := obrien_hats_before simpson_hats - x

-- Define the theorem stating the problem
theorem obrien_hats_theorem (x : ℕ) : obrien_hats_now x = 35 - x :=
by
  sorry

end NUMINAMATH_GPT_obrien_hats_theorem_l2078_207854


namespace NUMINAMATH_GPT_find_k_l2078_207864

theorem find_k (m : ℝ) (h : ∃ A B : ℝ, (m^3 - 24*m + 16) = (m^2 - 8*m) * (A*m + B) ∧ A - 8 = -k ∧ -8*B = -24) : k = 5 :=
sorry

end NUMINAMATH_GPT_find_k_l2078_207864


namespace NUMINAMATH_GPT_find_sum_mod_7_l2078_207885

open ZMod

-- Let a, b, and c be elements of the cyclic group modulo 7
def a : ZMod 7 := sorry
def b : ZMod 7 := sorry
def c : ZMod 7 := sorry

-- Conditions
axiom h1 : a * b * c = 1
axiom h2 : 4 * c = 5
axiom h3 : 5 * b = 4 + b

-- Goal
theorem find_sum_mod_7 : a + b + c = 2 := by
  sorry

end NUMINAMATH_GPT_find_sum_mod_7_l2078_207885


namespace NUMINAMATH_GPT_infinite_series_sum_l2078_207842

theorem infinite_series_sum : 
  ∑' k : ℕ, (5^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 := 
sorry

end NUMINAMATH_GPT_infinite_series_sum_l2078_207842


namespace NUMINAMATH_GPT_arrangement_problem_l2078_207823
noncomputable def num_arrangements : ℕ := 144

theorem arrangement_problem (A B C D E F : ℕ) 
  (adjacent_easy : A = B) 
  (not_adjacent_difficult : E ≠ F) : num_arrangements = 144 :=
by sorry

end NUMINAMATH_GPT_arrangement_problem_l2078_207823


namespace NUMINAMATH_GPT_suit_price_after_discount_l2078_207872

-- Define the original price of the suit.
def original_price : ℝ := 150

-- Define the increase rate and the discount rate.
def increase_rate : ℝ := 0.20
def discount_rate : ℝ := 0.20

-- Define the increased price after the 20% increase.
def increased_price : ℝ := original_price * (1 + increase_rate)

-- Define the final price after applying the 20% discount.
def final_price : ℝ := increased_price * (1 - discount_rate)

-- Prove that the final price is $144.
theorem suit_price_after_discount : final_price = 144 := by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_suit_price_after_discount_l2078_207872


namespace NUMINAMATH_GPT_part_one_part_two_l2078_207839

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * x + 6

theorem part_one (x : ℝ) : ∀ a, a = 5 → f x a < 0 ↔ -3 < x ∧ x < -2 :=
by
  sorry

theorem part_two : ∀ a, (∀ x, f x a > 0) ↔ - 2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l2078_207839


namespace NUMINAMATH_GPT_expression_equals_5_l2078_207828

theorem expression_equals_5 : (3^2 - 2^2) = 5 := by
  calc
    (3^2 - 2^2) = 5 := by sorry

end NUMINAMATH_GPT_expression_equals_5_l2078_207828


namespace NUMINAMATH_GPT_ellipse_k_range_ellipse_k_eccentricity_l2078_207844

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) ↔ (1 < k ∧ k < 5 ∨ 5 < k ∧ k < 9) := 
sorry

theorem ellipse_k_eccentricity (k : ℝ) (h : ∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) : 
  eccentricity = Real.sqrt (6/7) → (k = 2 ∨ k = 8) := 
sorry

end NUMINAMATH_GPT_ellipse_k_range_ellipse_k_eccentricity_l2078_207844


namespace NUMINAMATH_GPT_g_at_five_l2078_207818

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_five :
  (∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) →
  g (5) = 22 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_g_at_five_l2078_207818


namespace NUMINAMATH_GPT_arccos_neg_one_eq_pi_l2078_207802

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arccos_neg_one_eq_pi_l2078_207802


namespace NUMINAMATH_GPT_line_equation_l2078_207851

theorem line_equation (x y : ℝ) : 
  (3 * x + y = 0) ∧ (x + y - 2 = 0) ∧ 
  ∃ m : ℝ, -2 = -(1 / m) ∧ 
  (∃ b : ℝ, (y = m * x + b) ∧ (3 = m * (-1) + b)) ∧ 
  x - 2 * y + 7 = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_l2078_207851


namespace NUMINAMATH_GPT_xiaoming_additional_games_l2078_207865

variable (total_games games_won target_percentage : ℕ)

theorem xiaoming_additional_games :
  total_games = 20 →
  games_won = 95 * total_games / 100 →
  target_percentage = 96 →
  ∃ additional_games, additional_games = 5 ∧
    (games_won + additional_games) / (total_games + additional_games) = target_percentage / 100 :=
by
  sorry

end NUMINAMATH_GPT_xiaoming_additional_games_l2078_207865


namespace NUMINAMATH_GPT_polynomial_value_at_2018_l2078_207869

theorem polynomial_value_at_2018 (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f (-x^2 - x - 1) = x^4 + 2*x^3 + 2022*x^2 + 2021*x + 2019) : 
  f 2018 = -2019 :=
sorry

end NUMINAMATH_GPT_polynomial_value_at_2018_l2078_207869


namespace NUMINAMATH_GPT_model_car_cost_l2078_207841

theorem model_car_cost (x : ℕ) :
  (5 * x) + (5 * 10) + (5 * 2) = 160 → x = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_model_car_cost_l2078_207841


namespace NUMINAMATH_GPT_swimming_speed_l2078_207896

theorem swimming_speed (v_m v_s : ℝ) 
  (h1 : v_m + v_s = 6)
  (h2 : v_m - v_s = 8) : 
  v_m = 7 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_l2078_207896


namespace NUMINAMATH_GPT_andrew_donuts_l2078_207804

/--
Andrew originally asked for 3 donuts for each of his 2 friends, Brian and Samuel. 
Then invited 2 more friends and asked for the same amount of donuts for them. 
Andrew’s mother wants to buy one more donut for each of Andrew’s friends. 
Andrew's mother is also going to buy the same amount of donuts for Andrew as everybody else.
Given these conditions, the total number of donuts Andrew’s mother needs to buy is 20.
-/
theorem andrew_donuts : (3 * 2) + (3 * 2) + 4 + 4 = 20 :=
by
  -- Given:
  -- 1. Andrew asked for 3 donuts for each of his two friends, Brian and Samuel.
  -- 2. He later invited 2 more friends and asked for the same amount of donuts for them.
  -- 3. Andrew’s mother wants to buy one more donut for each of Andrew’s friends.
  -- 4. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else.
  -- Prove: The total number of donuts Andrew’s mother needs to buy is 20.
  sorry

end NUMINAMATH_GPT_andrew_donuts_l2078_207804


namespace NUMINAMATH_GPT_vasya_birthday_day_l2078_207891

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end NUMINAMATH_GPT_vasya_birthday_day_l2078_207891


namespace NUMINAMATH_GPT_shares_owned_l2078_207843

theorem shares_owned (expected_earnings dividend_ratio additional_per_10c actual_earnings total_dividend : ℝ)
  ( h1 : expected_earnings = 0.80 )
  ( h2 : dividend_ratio = 0.50 )
  ( h3 : additional_per_10c = 0.04 )
  ( h4 : actual_earnings = 1.10 )
  ( h5 : total_dividend = 156.0 ) :
  ∃ shares : ℝ, shares = total_dividend / (expected_earnings * dividend_ratio + (max ((actual_earnings - expected_earnings) / 0.10) 0) * additional_per_10c) ∧ shares = 300 := 
sorry

end NUMINAMATH_GPT_shares_owned_l2078_207843


namespace NUMINAMATH_GPT_rationalize_denominator_eqn_l2078_207871

theorem rationalize_denominator_eqn : 
  let expr := (3 + Real.sqrt 2) / (2 - Real.sqrt 5)
  let rationalized := -6 - 3 * Real.sqrt 5 - 2 * Real.sqrt 2 - Real.sqrt 10
  let A := -6
  let B := -2
  let C := 2
  expr = rationalized ∧ A * B * C = -24 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_eqn_l2078_207871


namespace NUMINAMATH_GPT_range_of_m_l2078_207866
-- Import the entire math library

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0 
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0 

-- Main theorem statement
theorem range_of_m (m : ℝ) (h1 : 0 < m) 
(hsuff : ∀ x : ℝ, p x → q x m) 
(hnsuff : ¬ (∀ x : ℝ, q x m → p x)) : m ≥ 9 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2078_207866


namespace NUMINAMATH_GPT_triangle_side_lengths_l2078_207879

theorem triangle_side_lengths (a : ℝ) :
  (∃ (b c : ℝ), b = 1 - 2 * a ∧ c = 8 ∧ (3 + b > c ∧ 3 + c > b ∧ b + c > 3)) ↔ (-5 < a ∧ a < -2) :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_l2078_207879


namespace NUMINAMATH_GPT_find_x_l2078_207829

theorem find_x (x : ℕ) (h₁ : 3 * (Nat.factorial 8) / (Nat.factorial (8 - x)) = 4 * (Nat.factorial 9) / (Nat.factorial (9 - (x - 1)))) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_x_l2078_207829


namespace NUMINAMATH_GPT_cost_per_adult_meal_l2078_207845

-- Definitions and given conditions
def total_people : ℕ := 13
def num_kids : ℕ := 9
def total_cost : ℕ := 28

-- Question translated into a proof statement
theorem cost_per_adult_meal : (total_cost / (total_people - num_kids)) = 7 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_adult_meal_l2078_207845


namespace NUMINAMATH_GPT_correct_M_min_t_for_inequality_l2078_207875

-- Define the set M
def M : Set ℝ := {a | 0 ≤ a ∧ a < 4}

-- Prove that M is correct given ax^2 + ax + 2 > 0 for all x ∈ ℝ implies 0 ≤ a < 4
theorem correct_M (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

-- Prove the minimum value of t given t > 0 and the inequality holds for all a ∈ M
theorem min_t_for_inequality (t : ℝ) (h : 0 < t) : 
  (∀ a ∈ M, (a^2 - 2 * a) * t ≤ t^2 + 3 * t - 46) ↔ 46 ≤ t :=
sorry

end NUMINAMATH_GPT_correct_M_min_t_for_inequality_l2078_207875


namespace NUMINAMATH_GPT_range_of_a_l2078_207862

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, -1 ≤ x → f a x ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2078_207862


namespace NUMINAMATH_GPT_blueprint_conversion_proof_l2078_207856

-- Let inch_to_feet be the conversion factor from blueprint inches to actual feet.
def inch_to_feet : ℝ := 500

-- Let line_segment_inch be the length of the line segment on the blueprint in inches.
def line_segment_inch : ℝ := 6.5

-- Then, line_segment_feet is the actual length of the line segment in feet.
def line_segment_feet : ℝ := line_segment_inch * inch_to_feet

-- Theorem statement to prove
theorem blueprint_conversion_proof : line_segment_feet = 3250 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_blueprint_conversion_proof_l2078_207856


namespace NUMINAMATH_GPT_full_price_ticket_revenue_l2078_207815

theorem full_price_ticket_revenue (f t : ℕ) (p : ℝ) 
  (h1 : f + t = 160) 
  (h2 : f * p + t * (p / 3) = 2500) 
  (h3 : p = 30) :
  f * p = 1350 := 
by sorry

end NUMINAMATH_GPT_full_price_ticket_revenue_l2078_207815


namespace NUMINAMATH_GPT_daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l2078_207824

-- Part (1)
theorem daily_sales_volume_and_profit (x : ℝ) :
  let increase_in_sales := 2 * x
  let profit_per_piece := 40 - x
  increase_in_sales = 2 * x ∧ profit_per_piece = 40 - x :=
by
  sorry

-- Part (2)
theorem profit_for_1200_yuan (x : ℝ) (h1 : (40 - x) * (20 + 2 * x) = 1200) :
  x = 10 ∨ x = 20 :=
by
  sorry

-- Part (3)
theorem profit_impossible_for_1800_yuan :
  ¬ ∃ y : ℝ, (40 - y) * (20 + 2 * y) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l2078_207824


namespace NUMINAMATH_GPT_total_books_l2078_207850

variable (Sandy_books Benny_books Tim_books : ℕ)
variable (h_Sandy : Sandy_books = 10)
variable (h_Benny : Benny_books = 24)
variable (h_Tim : Tim_books = 33)

theorem total_books :
  Sandy_books + Benny_books + Tim_books = 67 :=
by sorry

end NUMINAMATH_GPT_total_books_l2078_207850


namespace NUMINAMATH_GPT_find_a9_l2078_207857

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions of the problem
variables {a : ℕ → ℝ}
axiom h_geom_seq : is_geometric_sequence a
axiom h_root1 : a 3 * a 15 = 1
axiom h_root2 : a 3 + a 15 = -4

-- The proof statement
theorem find_a9 : a 9 = 1 := 
by sorry

end NUMINAMATH_GPT_find_a9_l2078_207857


namespace NUMINAMATH_GPT_max_M_inequality_l2078_207810

theorem max_M_inequality :
  ∃ M : ℝ, (∀ x y : ℝ, x + y ≥ 0 → (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x * y - x - y)) ∧ M = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_M_inequality_l2078_207810


namespace NUMINAMATH_GPT_polynomial_simplification_l2078_207861

theorem polynomial_simplification (p : ℤ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) = 
  2 * p^4 + 6 * p^3 + p^2 + p + 4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l2078_207861


namespace NUMINAMATH_GPT_herd_total_cows_l2078_207846

theorem herd_total_cows (n : ℕ) : 
  let first_son := 1 / 3 * n
  let second_son := 1 / 6 * n
  let third_son := 1 / 8 * n
  let remaining := n - (first_son + second_son + third_son)
  remaining = 9 ↔ n = 24 := 
by
  -- Skipping proof, placeholder
  sorry

end NUMINAMATH_GPT_herd_total_cows_l2078_207846


namespace NUMINAMATH_GPT_odd_blue_faces_in_cubes_l2078_207894

noncomputable def count_odd_blue_faces (length width height : ℕ) : ℕ :=
if length = 6 ∧ width = 4 ∧ height = 2 then 16 else 0

theorem odd_blue_faces_in_cubes : count_odd_blue_faces 6 4 2 = 16 := 
by
  -- The proof would involve calculating the corners, edges, etc.
  sorry

end NUMINAMATH_GPT_odd_blue_faces_in_cubes_l2078_207894


namespace NUMINAMATH_GPT_find_solution_l2078_207888

-- Definitions for the problem
def is_solution (x y z t : ℕ) : Prop := (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ (2^y + 2^z * 5^t - 5^x = 1))

-- Statement of the theorem
theorem find_solution : ∀ x y z t : ℕ, is_solution x y z t → (x, y, z, t) = (2, 4, 1, 1) := by
  sorry

end NUMINAMATH_GPT_find_solution_l2078_207888


namespace NUMINAMATH_GPT_cost_of_each_shirt_l2078_207813

theorem cost_of_each_shirt
  (x : ℝ) 
  (h : 3 * x + 2 * 20 = 85) : x = 15 :=
sorry

end NUMINAMATH_GPT_cost_of_each_shirt_l2078_207813


namespace NUMINAMATH_GPT_cakes_served_during_lunch_today_l2078_207863

theorem cakes_served_during_lunch_today (L : ℕ) 
  (h_total : L + 6 + 3 = 14) : 
  L = 5 :=
sorry

end NUMINAMATH_GPT_cakes_served_during_lunch_today_l2078_207863


namespace NUMINAMATH_GPT_minimum_value_of_function_l2078_207899

theorem minimum_value_of_function :
  ∀ x : ℝ, (x > -2) → (x + (16 / (x + 2)) ≥ 6) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l2078_207899


namespace NUMINAMATH_GPT_gain_percentage_for_40_clocks_is_10_l2078_207860

-- Condition: Cost price per clock
def cost_price := 79.99999999999773

-- Condition: Selling price of 50 clocks at a gain of 20%
def selling_price_50 := 50 * cost_price * 1.20

-- Uniform profit condition
def uniform_profit_total := 90 * cost_price * 1.15

-- Given total revenue difference Rs. 40
def total_revenue := uniform_profit_total + 40

-- Question: Prove that selling price of 40 clocks leads to 10% gain
theorem gain_percentage_for_40_clocks_is_10 :
    40 * cost_price * 1.10 = total_revenue - selling_price_50 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_for_40_clocks_is_10_l2078_207860


namespace NUMINAMATH_GPT_cricket_target_l2078_207893

theorem cricket_target (run_rate_first_10overs run_rate_next_40overs : ℝ) (overs_first_10 next_40_overs : ℕ)
    (h_first : run_rate_first_10overs = 3.2) 
    (h_next : run_rate_next_40overs = 6.25) 
    (h_overs_first : overs_first_10 = 10) 
    (h_overs_next : next_40_overs = 40) 
    : (overs_first_10 * run_rate_first_10overs + next_40_overs * run_rate_next_40overs) = 282 :=
by
  sorry

end NUMINAMATH_GPT_cricket_target_l2078_207893


namespace NUMINAMATH_GPT_frank_bought_2_bags_of_chips_l2078_207873

theorem frank_bought_2_bags_of_chips
  (cost_choco_bar : ℕ)
  (num_choco_bar : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (cost_bag_chip : ℕ)
  (num_bags_chip : ℕ)
  (h1 : cost_choco_bar = 2)
  (h2 : num_choco_bar = 5)
  (h3 : total_money = 20)
  (h4 : change = 4)
  (h5 : cost_bag_chip = 3)
  (h6 : total_money - change = (cost_choco_bar * num_choco_bar) + (cost_bag_chip * num_bags_chip)) :
  num_bags_chip = 2 := by
  sorry

end NUMINAMATH_GPT_frank_bought_2_bags_of_chips_l2078_207873


namespace NUMINAMATH_GPT_find_expression_l2078_207801

-- Definitions based on the conditions provided
def prop_rel (y x : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 2)

def prop_value_k (k : ℝ) : Prop :=
  k = -4

def prop_value_y (y x : ℝ) : Prop :=
  y = -4 * x + 8

theorem find_expression (y x k : ℝ) : 
  (prop_rel y x k) → 
  (x = 3) → 
  (y = -4) → 
  (prop_value_k k) → 
  (prop_value_y y x) :=
by
  intros h1 h2 h3 h4
  subst h4
  subst h3
  subst h2
  sorry

end NUMINAMATH_GPT_find_expression_l2078_207801


namespace NUMINAMATH_GPT_relative_frequency_defective_books_l2078_207819

theorem relative_frequency_defective_books 
  (N_defective : ℤ) (N_total : ℤ)
  (h_defective : N_defective = 5)
  (h_total : N_total = 100) :
  (N_defective : ℚ) / N_total = 0.05 := by
  sorry

end NUMINAMATH_GPT_relative_frequency_defective_books_l2078_207819


namespace NUMINAMATH_GPT_poster_width_l2078_207892
   
   theorem poster_width (h : ℕ) (A : ℕ) (w : ℕ) (h_eq : h = 7) (A_eq : A = 28) (area_eq : w * h = A) : w = 4 :=
   by
   sorry
   
end NUMINAMATH_GPT_poster_width_l2078_207892


namespace NUMINAMATH_GPT_Q_after_move_up_4_units_l2078_207895

-- Define the initial coordinates.
def Q_initial : (ℤ × ℤ) := (-4, -6)

-- Define the transformation - moving up 4 units.
def move_up (P : ℤ × ℤ) (units : ℤ) : (ℤ × ℤ) := (P.1, P.2 + units)

-- State the theorem to be proved.
theorem Q_after_move_up_4_units : move_up Q_initial 4 = (-4, -2) :=
by 
  sorry

end NUMINAMATH_GPT_Q_after_move_up_4_units_l2078_207895


namespace NUMINAMATH_GPT_chicago_bulls_wins_l2078_207855

theorem chicago_bulls_wins (B H : ℕ) (h1 : B + H = 145) (h2 : H = B + 5) : B = 70 :=
by
  sorry

end NUMINAMATH_GPT_chicago_bulls_wins_l2078_207855
