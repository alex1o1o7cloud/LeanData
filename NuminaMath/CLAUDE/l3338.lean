import Mathlib

namespace NUMINAMATH_CALUDE_total_rainfall_three_years_l3338_333846

def average_rainfall_2003 : ℝ := 50
def rainfall_increase_2004 : ℝ := 3
def rainfall_increase_2005 : ℝ := 5
def months_per_year : ℕ := 12

theorem total_rainfall_three_years : 
  (average_rainfall_2003 * months_per_year) +
  ((average_rainfall_2003 + rainfall_increase_2004) * months_per_year) +
  ((average_rainfall_2003 + rainfall_increase_2004 + rainfall_increase_2005) * months_per_year) = 1932 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_three_years_l3338_333846


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l3338_333894

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l3338_333894


namespace NUMINAMATH_CALUDE_derivative_f_at_neg_one_l3338_333874

def f (x : ℝ) : ℝ := x^2 * (x + 1)

theorem derivative_f_at_neg_one :
  (deriv f) (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_neg_one_l3338_333874


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_range_l3338_333850

/-- Given a quadratic function f(x) = x^2 + 4mx + n that is decreasing on the interval [2, 6],
    prove that m ≤ -3. -/
theorem quadratic_decreasing_implies_m_range
  (f : ℝ → ℝ)
  (m n : ℝ)
  (h_f : ∀ x, f x = x^2 + 4*m*x + n)
  (h_decreasing : ∀ x y, x ∈ Set.Icc 2 6 → y ∈ Set.Icc 2 6 → x < y → f x > f y) :
  m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_range_l3338_333850


namespace NUMINAMATH_CALUDE_novels_per_month_l3338_333831

/-- Given that each novel has 200 pages and 9600 pages of novels are read in a year,
    prove that 4 novels are read in a month. -/
theorem novels_per_month :
  ∀ (pages_per_novel : ℕ) (pages_per_year : ℕ) (months_per_year : ℕ),
    pages_per_novel = 200 →
    pages_per_year = 9600 →
    months_per_year = 12 →
    (pages_per_year / pages_per_novel) / months_per_year = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_novels_per_month_l3338_333831


namespace NUMINAMATH_CALUDE_train_speed_fraction_l3338_333820

/-- Given a train journey where:
  1. The train reached its destination in 8 hours at a certain fraction of its own speed.
  2. If the train had run at its full speed, it would have taken 4 hours less.
  This theorem proves that the fraction of the train's own speed at which it was running is 1/2. -/
theorem train_speed_fraction (full_speed : ℝ) (fraction : ℝ) 
  (h1 : fraction * full_speed * 8 = full_speed * 4) : fraction = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l3338_333820


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3338_333870

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = -4) →
  a = 8 ∨ a = -8 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3338_333870


namespace NUMINAMATH_CALUDE_R_share_is_1295_l3338_333807

/-- Represents the capital invested by a partner -/
structure Capital where
  amount : ℚ
  is_positive : amount > 0

/-- Represents the investment scenario of the four partners -/
structure InvestmentScenario where
  P : Capital
  Q : Capital
  R : Capital
  S : Capital
  ratio_PQ : 4 * P.amount = 6 * Q.amount
  ratio_QR : 6 * Q.amount = 10 * R.amount
  S_investment : S.amount = P.amount + Q.amount
  total_profit : ℚ
  profit_is_positive : total_profit > 0

/-- Calculates the share of profit for partner R -/
def calculate_R_share (scenario : InvestmentScenario) : ℚ :=
  let total_capital := scenario.P.amount + scenario.Q.amount + scenario.R.amount + scenario.S.amount
  (scenario.total_profit * scenario.R.amount) / total_capital

/-- Theorem stating that R's share of profit is 1295 given the investment scenario -/
theorem R_share_is_1295 (scenario : InvestmentScenario) (h : scenario.total_profit = 12090) :
  calculate_R_share scenario = 1295 := by
  sorry

end NUMINAMATH_CALUDE_R_share_is_1295_l3338_333807


namespace NUMINAMATH_CALUDE_midnight_temperature_l3338_333879

/-- Given the temperature changes throughout the day in a certain city, 
    prove that the temperature at midnight is 24°C. -/
theorem midnight_temperature 
  (morning_temp : ℝ)
  (afternoon_increase : ℝ)
  (midnight_decrease : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_increase = 1)
  (h3 : midnight_decrease = 7) :
  morning_temp + afternoon_increase - midnight_decrease = 24 :=
by sorry

end NUMINAMATH_CALUDE_midnight_temperature_l3338_333879


namespace NUMINAMATH_CALUDE_smallest_positive_root_floor_l3338_333804

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x - 3 * Real.cos x + Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_positive_root_floor :
  ∃ s, is_smallest_positive_root s ∧ ⌊s⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_root_floor_l3338_333804


namespace NUMINAMATH_CALUDE_sequence_property_l3338_333893

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n ^ 2 + n * sequence_a n

def S : Set ℕ := {p : ℕ | Nat.Prime p ∧ ∃ i, p ∣ sequence_a i}

theorem sequence_property :
  (Set.Infinite S) ∧ (S ≠ {p : ℕ | Nat.Prime p}) := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3338_333893


namespace NUMINAMATH_CALUDE_altitude_segment_length_l3338_333810

/-- An acute triangle with two altitudes dividing two sides -/
structure AcuteTriangleWithAltitudes where
  -- Sides
  AC : ℝ
  BC : ℝ
  -- Segments created by altitudes
  AD : ℝ
  DC : ℝ
  CE : ℝ
  EB : ℝ
  -- Conditions
  acute : AC > 0 ∧ BC > 0  -- Simplification for acute triangle
  altitude_division : AD + DC = AC ∧ CE + EB = BC
  given_lengths : AD = 6 ∧ DC = 4 ∧ CE = 3

/-- The theorem stating that y (EB) equals 11/3 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.EB = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l3338_333810


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_5_l3338_333873

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity_sqrt_5 
  {a b c : ℝ} (h : Hyperbola a b) 
  (focus_c : left_focus h = (-c, 0))
  (point_P : c^2/a^2 - 4 = 1) : 
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_5_l3338_333873


namespace NUMINAMATH_CALUDE_set_intersection_example_l3338_333849

theorem set_intersection_example : 
  let A : Set Int := {-1, 1}
  let B : Set Int := {-3, 0, 1}
  A ∩ B = {1} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3338_333849


namespace NUMINAMATH_CALUDE_inequality_solution_l3338_333898

noncomputable section

variables (a x : ℝ)

def inequality := (a * (x - 1)) / (x - 2) > 1

def solution : Prop :=
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a - 2) / (a - 1)) ∧
  (a = 1 → x > 2) ∧
  (a > 1 → x > 2 ∨ x < (a - 2) / (a - 1))

theorem inequality_solution (h : a > 0) : inequality a x ↔ solution a x := by sorry

end

end NUMINAMATH_CALUDE_inequality_solution_l3338_333898


namespace NUMINAMATH_CALUDE_optimal_distribution_l3338_333855

/-- Represents the production process for assembling products -/
structure ProductionProcess where
  totalWorkers : ℕ
  productsToAssemble : ℕ
  paintTime : ℕ
  dryTime : ℕ
  assemblyTime : ℕ

/-- Represents the distribution of workers -/
structure WorkerDistribution where
  painters : ℕ
  assemblers : ℕ

/-- Calculates the production time for a given worker distribution -/
def productionTime (process : ProductionProcess) (dist : WorkerDistribution) : ℕ :=
  sorry

/-- Checks if a worker distribution is valid for the given process -/
def isValidDistribution (process : ProductionProcess) (dist : WorkerDistribution) : Prop :=
  dist.painters + dist.assemblers ≤ process.totalWorkers

/-- Theorem stating the optimal worker distribution for the given process -/
theorem optimal_distribution (process : ProductionProcess) 
  (h1 : process.totalWorkers = 10)
  (h2 : process.productsToAssemble = 50)
  (h3 : process.paintTime = 10)
  (h4 : process.dryTime = 5)
  (h5 : process.assemblyTime = 20) :
  ∃ (optDist : WorkerDistribution), 
    optDist.painters = 3 ∧ 
    optDist.assemblers = 6 ∧
    isValidDistribution process optDist ∧
    ∀ (dist : WorkerDistribution), 
      isValidDistribution process dist → 
      productionTime process optDist ≤ productionTime process dist :=
  sorry

end NUMINAMATH_CALUDE_optimal_distribution_l3338_333855


namespace NUMINAMATH_CALUDE_max_value_of_sum_l3338_333819

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) :
  ∃ (M : ℝ), M = Real.sqrt 70 ∧ x + 2*y + 3*z ≤ M ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + 2*y₀ + 3*z₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l3338_333819


namespace NUMINAMATH_CALUDE_sticker_distribution_l3338_333854

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 9 identical stickers among 3 distinct sheets of paper -/
theorem sticker_distribution : distribute 9 3 = 55 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3338_333854


namespace NUMINAMATH_CALUDE_machine_value_theorem_l3338_333892

/-- Calculates the machine's value after 2 years given the initial conditions -/
def machine_value_after_two_years (initial_value : ℝ) (depreciation_rate_year1 : ℝ) 
  (depreciation_rate_subsequent : ℝ) (inflation_rate_year1 : ℝ) (inflation_rate_year2 : ℝ) 
  (maintenance_cost_year1 : ℝ) (maintenance_cost_increase_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the machine's value after 2 years is $754.58 -/
theorem machine_value_theorem : 
  machine_value_after_two_years 1000 0.12 0.08 0.02 0.035 50 0.05 = 754.58 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_theorem_l3338_333892


namespace NUMINAMATH_CALUDE_distance_between_stations_l3338_333875

def passenger_train_speed : ℚ := 1/2
def express_train_speed : ℚ := 1
def catch_up_distance : ℚ := 244/9

theorem distance_between_stations (x : ℚ) : 
  (x/3 + 4*(x/3 - catch_up_distance)/3 = x - catch_up_distance) → x = 528 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_stations_l3338_333875


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3338_333872

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 82 ways to put 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : ballsInBoxes 6 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3338_333872


namespace NUMINAMATH_CALUDE_second_term_of_sequence_l3338_333897

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ aₙ = a₁ + (n - 1) * d

theorem second_term_of_sequence (a₁ a₂ aₙ d : ℕ) :
  a₁ = 34 → d = 11 → aₙ = 89 → arithmetic_sequence a₁ d aₙ → a₂ = 45 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_sequence_l3338_333897


namespace NUMINAMATH_CALUDE_total_sibling_age_l3338_333877

/-- Represents the ages of the siblings -/
structure SiblingAges where
  susan : ℝ
  arthur : ℝ
  tom : ℝ
  bob : ℝ
  emily : ℝ
  david : ℝ
  youngest : ℝ

/-- Theorem stating the total age of the siblings -/
theorem total_sibling_age (ages : SiblingAges) : 
  ages.arthur = ages.susan + 2 →
  ages.tom = ages.bob - 3 →
  ages.emily = ages.susan / 2 →
  ages.david = (ages.arthur + ages.tom + ages.emily) / 3 →
  ages.susan - ages.tom = 2 * (ages.emily - ages.david) →
  ages.bob = 11 →
  ages.susan = 15 →
  ages.emily = ages.youngest + 2.5 →
  ages.susan + ages.arthur + ages.tom + ages.bob + ages.emily + ages.david + ages.youngest = 74.5 := by
  sorry


end NUMINAMATH_CALUDE_total_sibling_age_l3338_333877


namespace NUMINAMATH_CALUDE_tan_beta_plus_pi_fourth_l3338_333847

theorem tan_beta_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan α = 1/3) : 
  Real.tan (β + π/4) = 11/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_plus_pi_fourth_l3338_333847


namespace NUMINAMATH_CALUDE_point_same_side_l3338_333814

def line (x y : ℝ) : ℝ := x + y - 1

def same_side (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (line x₁ y₁ > 0 ∧ line x₂ y₂ > 0) ∨ (line x₁ y₁ < 0 ∧ line x₂ y₂ < 0)

theorem point_same_side : same_side 1 2 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_same_side_l3338_333814


namespace NUMINAMATH_CALUDE_integral_f_equals_three_l3338_333800

-- Define the function to be integrated
def f (x : ℝ) : ℝ := 2 - |1 - x|

-- State the theorem
theorem integral_f_equals_three :
  ∫ x in (0 : ℝ)..2, f x = 3 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_three_l3338_333800


namespace NUMINAMATH_CALUDE_circle_equations_l3338_333821

-- Define the circle N
def circle_N (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 10

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 10

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 5/2)^2 + (y - 2)^2 = 5/2

-- Define points A, B, and C
def point_A : ℝ × ℝ := (3, 1)
def point_B : ℝ × ℝ := (-1, 3)
def point_C : ℝ × ℝ := (3, 0)

-- Define the line that contains the center of circle N
def center_line (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Define a point D on circle N
def point_D (x y : ℝ) : Prop := circle_N x y

-- Define the midpoint M of segment CD
def midpoint_M (x y x_D y_D : ℝ) : Prop := x = (x_D + 3)/2 ∧ y = y_D/2

theorem circle_equations :
  (∀ x y, circle_N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) ∧
  (∀ x y, symmetric_circle x y ↔ (x - 1)^2 + (y - 5)^2 = 10) ∧
  (∀ x y, (∃ x_D y_D, point_D x_D y_D ∧ midpoint_M x y x_D y_D) → trajectory_M x y) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_l3338_333821


namespace NUMINAMATH_CALUDE_square_of_binomial_l3338_333862

theorem square_of_binomial (a b : ℝ) : (2*a - 3*b)^2 = 4*a^2 - 12*a*b + 9*b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3338_333862


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l3338_333834

theorem factorization_a_squared_minus_ab (a b : ℝ) : a^2 - a*b = a*(a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l3338_333834


namespace NUMINAMATH_CALUDE_june_math_book_price_l3338_333887

/-- The price of a math book that satisfies June's shopping constraints -/
def math_book_price : ℝ → Prop := λ x =>
  let total_budget : ℝ := 500
  let num_math_books : ℕ := 4
  let num_science_books : ℕ := num_math_books + 6
  let science_book_price : ℝ := 10
  let num_art_books : ℕ := 2 * num_math_books
  let art_book_price : ℝ := 20
  let music_books_cost : ℝ := 160
  (num_math_books : ℝ) * x + 
  (num_science_books : ℝ) * science_book_price + 
  (num_art_books : ℝ) * art_book_price + 
  music_books_cost = total_budget

theorem june_math_book_price : ∃ x : ℝ, math_book_price x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_june_math_book_price_l3338_333887


namespace NUMINAMATH_CALUDE_bag_balls_count_l3338_333857

theorem bag_balls_count (blue_balls : ℕ) (prob_blue : ℚ) : 
  blue_balls = 8 →
  prob_blue = 1/3 →
  ∃ (green_balls : ℕ),
    (blue_balls : ℚ) / ((blue_balls : ℚ) + (green_balls : ℚ)) = prob_blue ∧
    blue_balls + green_balls = 24 :=
by sorry

end NUMINAMATH_CALUDE_bag_balls_count_l3338_333857


namespace NUMINAMATH_CALUDE_pattern_circle_area_ratio_l3338_333809

-- Define the circle
def circle_radius : ℝ := 3

-- Define the rectangle
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 6

-- Define the number of arcs
def num_arcs : ℕ := 6

-- Theorem statement
theorem pattern_circle_area_ratio :
  let circle_area := π * circle_radius^2
  let pattern_area := circle_area  -- Assumption: rearranged arcs preserve total area
  pattern_area / circle_area = 1 := by sorry

end NUMINAMATH_CALUDE_pattern_circle_area_ratio_l3338_333809


namespace NUMINAMATH_CALUDE_tree_planting_probability_l3338_333871

def num_cedar : ℕ := 4
def num_pine : ℕ := 3
def num_alder : ℕ := 6

def total_trees : ℕ := num_cedar + num_pine + num_alder

def probability_no_adjacent_alders : ℚ := 2 / 4290

theorem tree_planting_probability :
  let total_arrangements : ℕ := (Nat.factorial total_trees) / 
    (Nat.factorial num_cedar * Nat.factorial num_pine * Nat.factorial num_alder)
  let valid_arrangements : ℕ := Nat.choose (num_cedar + num_pine + 1) num_alder * 
    (Nat.factorial (num_cedar + num_pine) / (Nat.factorial num_cedar * Nat.factorial num_pine))
  (valid_arrangements : ℚ) / total_arrangements = probability_no_adjacent_alders :=
sorry

end NUMINAMATH_CALUDE_tree_planting_probability_l3338_333871


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l3338_333801

theorem set_equality_implies_sum_of_powers (a b : ℝ) : 
  ({b, b/a, 0} : Set ℝ) = {a, a+b, 1} → a^2018 + b^2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l3338_333801


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3338_333860

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (h : ℝ) 
  (lateral_surface_area : ℝ → ℝ → ℝ) :
  r = 2 →
  h = 4 * Real.sqrt 2 →
  lateral_surface_area r h = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3338_333860


namespace NUMINAMATH_CALUDE_intersection_condition_l3338_333886

/-- 
Given two equations:
1) y = √(2x^2 + 2x - m)
2) y = x - 2
This theorem states that for these equations to have a real intersection, 
m must be greater than or equal to 12.
-/
theorem intersection_condition (x y m : ℝ) : 
  (y = Real.sqrt (2 * x^2 + 2 * x - m) ∧ y = x - 2) → m ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3338_333886


namespace NUMINAMATH_CALUDE_remaining_money_l3338_333822

def base_8_to_10 (n : ℕ) : ℕ := 
  4 * 8^3 + 4 * 8^2 + 4 * 8^1 + 4 * 8^0

def savings : ℕ := base_8_to_10 4444

def ticket_cost : ℕ := 1000

theorem remaining_money : 
  savings - ticket_cost = 1340 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l3338_333822


namespace NUMINAMATH_CALUDE_linear_function_passes_through_points_linear_function_unique_l3338_333838

/-- A linear function passing through two points (2, 3) and (3, 2) -/
def linearFunction (x : ℝ) : ℝ := -x + 5

/-- The theorem stating that the linear function passes through the given points -/
theorem linear_function_passes_through_points :
  linearFunction 2 = 3 ∧ linearFunction 3 = 2 := by
  sorry

/-- The theorem stating that the linear function is unique -/
theorem linear_function_unique (f : ℝ → ℝ) :
  f 2 = 3 → f 3 = 2 → ∀ x, f x = linearFunction x := by
  sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_points_linear_function_unique_l3338_333838


namespace NUMINAMATH_CALUDE_min_team_a_size_l3338_333852

theorem min_team_a_size (a b : ℕ) : 
  (∃ c : ℕ, 2 * (a - 90) = b + 90 ∧ a + c = 6 * (b - c)) →
  a ≥ 153 :=
sorry

end NUMINAMATH_CALUDE_min_team_a_size_l3338_333852


namespace NUMINAMATH_CALUDE_inverse_proposition_geometric_sequence_l3338_333842

theorem inverse_proposition_geometric_sequence (a b c : ℝ) :
  (∀ {a b c : ℝ}, (∃ r : ℝ, b = a * r ∧ c = b * r) → b^2 = a * c) →
  (b^2 = a * c → ∃ r : ℝ, b = a * r ∧ c = b * r) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_geometric_sequence_l3338_333842


namespace NUMINAMATH_CALUDE_opposite_of_2021_l3338_333853

theorem opposite_of_2021 : ∃ x : ℝ, x + 2021 = 0 ∧ x = -2021 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2021_l3338_333853


namespace NUMINAMATH_CALUDE_initial_amount_problem_l3338_333895

theorem initial_amount_problem (initial_amount : ℝ) : 
  (initial_amount * (1 + 1/8) * (1 + 1/8) = 97200) → initial_amount = 76800 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_problem_l3338_333895


namespace NUMINAMATH_CALUDE_total_books_read_is_72sc_l3338_333844

/-- Calculates the total number of books read by a school's student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 6
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read by the entire student body in one year is 72sc -/
theorem total_books_read_is_72sc (c s : ℕ) : total_books_read c s = 72 * c * s := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_72sc_l3338_333844


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3338_333815

-- Problem 1
theorem problem_1 : 
  |(-2 : ℝ)| + Real.sqrt 2 * Real.tan (45 * π / 180) - Real.sqrt 8 - (2023 - Real.pi) ^ (0 : ℝ) = 1 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, x ≠ 2 → ((2 * x - 3) / (x - 2) - 1 / (2 - x) = 1 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3338_333815


namespace NUMINAMATH_CALUDE_plot_length_is_60_l3338_333826

/-- Proves that the length of a rectangular plot is 60 meters given the specified conditions -/
theorem plot_length_is_60 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 60 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_60_l3338_333826


namespace NUMINAMATH_CALUDE_max_intersections_after_300_turns_l3338_333878

/-- The number of intersections formed by n lines on a plane -/
def num_intersections (n : ℕ) : ℕ := n.choose 2

/-- The number of lines after 300 turns -/
def total_lines : ℕ := 3 + 300

theorem max_intersections_after_300_turns :
  num_intersections total_lines = 45853 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_after_300_turns_l3338_333878


namespace NUMINAMATH_CALUDE_tangent_product_30_60_l3338_333811

theorem tangent_product_30_60 (A B : Real) (hA : A = 30 * π / 180) (hB : B = 60 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = (3 + 4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_30_60_l3338_333811


namespace NUMINAMATH_CALUDE_emily_score_proof_l3338_333888

def emily_scores : List ℝ := [88, 92, 85, 90, 97]
def target_mean : ℝ := 91
def sixth_score : ℝ := 94

theorem emily_score_proof :
  let all_scores := emily_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_score_proof_l3338_333888


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3338_333859

-- Problem 1
theorem problem_1 : (-6) + (-13) = -19 := by sorry

-- Problem 2
theorem problem_2 : (3 : ℚ) / 5 + (-3 / 4) = -3 / 20 := by sorry

-- Problem 3
theorem problem_3 : (4.7 : ℝ) + (-0.8) + 5.3 + (-8.2) = 1 := by sorry

-- Problem 4
theorem problem_4 : (-1 : ℚ) / 6 + 1 / 3 + (-1 / 12) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3338_333859


namespace NUMINAMATH_CALUDE_jake_final_bitcoin_count_l3338_333867

def bitcoin_transactions (initial : ℕ) (investment : ℕ) (donation1 : ℕ) (debt : ℕ) (donation2 : ℕ) : ℕ :=
  let after_investment := initial - investment + 2 * investment
  let after_donation1 := after_investment - donation1
  let after_sharing := after_donation1 - (after_donation1 / 2)
  let after_debt_collection := after_sharing + debt
  let after_quadrupling := 4 * after_debt_collection
  after_quadrupling - donation2

theorem jake_final_bitcoin_count :
  bitcoin_transactions 120 40 25 5 15 = 277 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_bitcoin_count_l3338_333867


namespace NUMINAMATH_CALUDE_counterexample_ten_l3338_333858

theorem counterexample_ten : 
  ¬(¬(Nat.Prime 10) → Nat.Prime (10 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_ten_l3338_333858


namespace NUMINAMATH_CALUDE_point_not_on_line_l3338_333882

/-- Given a line y = mx + b where m and b are real numbers satisfying mb > 0,
    prove that the point (2023, 0) cannot lie on this line. -/
theorem point_not_on_line (m b : ℝ) (h : m * b > 0) :
  ¬(0 = 2023 * m + b) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l3338_333882


namespace NUMINAMATH_CALUDE_anna_spending_l3338_333805

/-- Anna's spending problem -/
theorem anna_spending (original : ℚ) (left : ℚ) (h1 : original = 32) (h2 : left = 24) :
  (original - left) / original = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_anna_spending_l3338_333805


namespace NUMINAMATH_CALUDE_inverse_g_87_l3338_333884

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 6

-- State the theorem
theorem inverse_g_87 : g⁻¹ 87 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_87_l3338_333884


namespace NUMINAMATH_CALUDE_divisor_problem_l3338_333840

theorem divisor_problem (d : ℕ+) (n : ℕ) (h1 : n % d = 3) (h2 : (2 * n) % d = 2) : d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3338_333840


namespace NUMINAMATH_CALUDE_regular_polygon_108_degrees_has_5_sides_l3338_333848

/-- A regular polygon with interior angles measuring 108 degrees has 5 sides. -/
theorem regular_polygon_108_degrees_has_5_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (180 * (n - 2) : ℝ) = (108 * n : ℝ) →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_108_degrees_has_5_sides_l3338_333848


namespace NUMINAMATH_CALUDE_library_repacking_l3338_333833

theorem library_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1500 → 
  books_per_initial_box = 45 → 
  books_per_new_box = 47 → 
  (initial_boxes * books_per_initial_box) % books_per_new_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l3338_333833


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l3338_333839

theorem onion_harvest_weight (initial_bags : ℕ) (trips : ℕ) (bag_weight : ℕ) : 
  initial_bags = 10 → trips = 20 → bag_weight = 50 →
  (initial_bags * ((2 ^ trips) - 1)) * bag_weight = 524287500 := by
  sorry

end NUMINAMATH_CALUDE_onion_harvest_weight_l3338_333839


namespace NUMINAMATH_CALUDE_total_students_l3338_333832

theorem total_students (N : ℕ) 
  (provincial_total : ℕ) (provincial_sample : ℕ)
  (experimental_sample : ℕ) (regular_sample : ℕ) (sino_canadian_sample : ℕ)
  (h1 : provincial_total = 96)
  (h2 : provincial_sample = 12)
  (h3 : experimental_sample = 21)
  (h4 : regular_sample = 25)
  (h5 : sino_canadian_sample = 43)
  (h6 : N * provincial_sample = provincial_total * (provincial_sample + experimental_sample + regular_sample + sino_canadian_sample)) :
  N = 808 := by
sorry

end NUMINAMATH_CALUDE_total_students_l3338_333832


namespace NUMINAMATH_CALUDE_power_sum_difference_l3338_333883

theorem power_sum_difference : 3^(2+3+4) - (3^2 + 3^3 + 3^4 + 3^5) = 19323 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l3338_333883


namespace NUMINAMATH_CALUDE_parabola_vertex_l3338_333823

/-- The parabola defined by y = x^2 - 2 -/
def parabola (x : ℝ) : ℝ := x^2 - 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (0, -2)

/-- Theorem: The vertex of the parabola y = x^2 - 2 is at the point (0, -2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3338_333823


namespace NUMINAMATH_CALUDE_vacuum_time_solution_l3338_333836

def chores_problem (vacuum_time : ℝ) : Prop :=
  let other_chores_time := 3 * vacuum_time
  vacuum_time + other_chores_time = 12

theorem vacuum_time_solution :
  ∃ (t : ℝ), chores_problem t ∧ t = 3 :=
sorry

end NUMINAMATH_CALUDE_vacuum_time_solution_l3338_333836


namespace NUMINAMATH_CALUDE_max_display_sum_l3338_333863

def DigitalWatch := ℕ × ℕ

def valid_time (t : DigitalWatch) : Prop :=
  1 ≤ t.1 ∧ t.1 ≤ 12 ∧ t.2 < 60

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def display_sum (t : DigitalWatch) : ℕ :=
  digit_sum t.1 + digit_sum t.2

theorem max_display_sum :
  ∃ (t : DigitalWatch), valid_time t ∧
    ∀ (t' : DigitalWatch), valid_time t' →
      display_sum t' ≤ display_sum t ∧
      display_sum t = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_display_sum_l3338_333863


namespace NUMINAMATH_CALUDE_two_roots_of_f_l3338_333813

/-- The function f(x) = 2^x - 3x has exactly two real roots -/
theorem two_roots_of_f : ∃! (n : ℕ), n = 2 ∧ (∃ (S : Set ℝ), S = {x : ℝ | 2^x - 3*x = 0} ∧ Finite S ∧ Nat.card S = n) := by
  sorry

end NUMINAMATH_CALUDE_two_roots_of_f_l3338_333813


namespace NUMINAMATH_CALUDE_institutions_made_happy_l3338_333841

theorem institutions_made_happy (people_per_institution : ℕ) (total_people_happy : ℕ) : 
  people_per_institution = 80 → total_people_happy = 480 → 
  total_people_happy / people_per_institution = 6 := by
sorry

end NUMINAMATH_CALUDE_institutions_made_happy_l3338_333841


namespace NUMINAMATH_CALUDE_radius_is_five_l3338_333899

/-- A configuration of tangent lines to a circle -/
structure TangentConfiguration where
  -- The radius of the circle
  r : ℝ
  -- The length of tangent line AB
  ab : ℝ
  -- The length of tangent line CD
  cd : ℝ
  -- The length of tangent line EF
  ef : ℝ
  -- AB and CD are parallel
  parallel_ab_cd : True
  -- A, C, and D are points of tangency
  tangency_points : True
  -- EF intersects AB and CD
  ef_intersects : True
  -- Tangency point for EF falls between AB and CD
  ef_tangency_between : True
  -- Given lengths
  ab_length : ab = 7
  cd_length : cd = 12
  ef_length : ef = 25

/-- The theorem stating that the radius is 5 given the configuration -/
theorem radius_is_five (config : TangentConfiguration) : config.r = 5 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_five_l3338_333899


namespace NUMINAMATH_CALUDE_gum_given_by_steve_l3338_333891

theorem gum_given_by_steve (initial_gum : ℕ) (final_gum : ℕ) 
  (h1 : initial_gum = 38) (h2 : final_gum = 54) :
  final_gum - initial_gum = 16 := by
  sorry

end NUMINAMATH_CALUDE_gum_given_by_steve_l3338_333891


namespace NUMINAMATH_CALUDE_no_solution_for_four_l3338_333864

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def three_digit_number (x y : ℕ) : ℕ :=
  100 * x + 30 + y

theorem no_solution_for_four :
  ∀ y : ℕ, y < 10 →
    ¬(is_divisible_by_11 (three_digit_number 4 y)) ∧
    (∀ x : ℕ, x < 10 → x ≠ 4 →
      ∃ y : ℕ, y < 10 ∧ is_divisible_by_11 (three_digit_number x y)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_four_l3338_333864


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3338_333843

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  -3 + 23*x - x^2 + 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3338_333843


namespace NUMINAMATH_CALUDE_largest_whole_number_times_eleven_less_than_150_l3338_333802

theorem largest_whole_number_times_eleven_less_than_150 :
  (∃ x : ℕ, x = 13 ∧ 11 * x < 150 ∧ ∀ y : ℕ, y > x → 11 * y ≥ 150) :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_times_eleven_less_than_150_l3338_333802


namespace NUMINAMATH_CALUDE_range_of_a_l3338_333885

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → -1 ≤ a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3338_333885


namespace NUMINAMATH_CALUDE_problem_solution_l3338_333808

theorem problem_solution (a b : ℝ) 
  (h1 : 2 * a - 1 = 9) 
  (h2 : 3 * a + b - 1 = 16) : 
  a + 2 * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3338_333808


namespace NUMINAMATH_CALUDE_largest_k_for_g_range_l3338_333803

/-- The function g(x) defined as x^2 + 5x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + k

/-- The property that -5 is in the range of g(x) -/
def inRange (k : ℝ) : Prop := ∃ x, g k x = -5

/-- The theorem stating that 5/4 is the largest value of k such that -5 is in the range of g(x) -/
theorem largest_k_for_g_range :
  (∀ k > 5/4, ¬ inRange k) ∧ inRange (5/4) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_g_range_l3338_333803


namespace NUMINAMATH_CALUDE_complete_square_existence_l3338_333869

theorem complete_square_existence :
  ∃ (k : ℤ) (a : ℝ), ∀ z : ℝ, z^2 - 6*z + 17 = (z + a)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_existence_l3338_333869


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l3338_333896

/-- Simple interest calculation theorem -/
theorem simple_interest_time_calculation
  (P : ℝ) (R : ℝ) (SI : ℝ)
  (h_P : P = 800)
  (h_R : R = 6.25)
  (h_SI : SI = 200)
  (h_formula : SI = P * R * (SI * 100 / (P * R)) / 100) :
  SI * 100 / (P * R) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l3338_333896


namespace NUMINAMATH_CALUDE_solve_for_e_l3338_333890

-- Define the functions p and q
def p (x : ℝ) : ℝ := 5 * x - 17
def q (e : ℝ) (x : ℝ) : ℝ := 4 * x - e

-- State the theorem
theorem solve_for_e : 
  ∀ e : ℝ, p (q e 3) = 23 → e = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_e_l3338_333890


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_396_l3338_333889

theorem six_digit_divisible_by_396 (n : ℕ) :
  (∃ x y z : ℕ, 
    0 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    n = 100000 * x + 10000 * y + 3420 + z) →
  n % 396 = 0 →
  n = 453420 ∨ n = 413424 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_396_l3338_333889


namespace NUMINAMATH_CALUDE_real_and_equal_roots_l3338_333806

/-- The quadratic equation in the problem -/
def quadratic_equation (k x : ℝ) : ℝ := 3 * x^2 - k * x + 2 * x + 15

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (k - 2)^2 - 4 * 3 * 15

theorem real_and_equal_roots (k : ℝ) :
  (∃ x : ℝ, quadratic_equation k x = 0 ∧
    ∀ y : ℝ, quadratic_equation k y = 0 → y = x) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
sorry

end NUMINAMATH_CALUDE_real_and_equal_roots_l3338_333806


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3338_333861

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ vertices
  finite : Finite vertices

/-- The perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- A triangle formed by three vertices of a polygon -/
def triangle_of_polygon (p : ConvexPolygon) : Set (ConvexPolygon) := sorry

theorem triangle_perimeter_bound (G : ConvexPolygon) :
  ∃ T ∈ triangle_of_polygon G, perimeter T ≥ 0.7 * perimeter G := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3338_333861


namespace NUMINAMATH_CALUDE_five_digit_cube_root_l3338_333837

theorem five_digit_cube_root (n : ℕ) : 
  (10000 ≤ n ∧ n < 100000) →  -- n is a five-digit number
  (n % 10 = 3) →              -- n ends in 3
  (∃ k : ℕ, k^3 = n) →        -- n has an integer cube root
  (n = 19683 ∨ n = 50653) :=  -- n is either 19683 or 50653
by sorry

end NUMINAMATH_CALUDE_five_digit_cube_root_l3338_333837


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l3338_333866

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 3x + 4)(5x^2 + 7x + 6) is 47 -/
theorem x_cubed_coefficient : 
  let p₁ : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 3 * X + 4
  let p₂ : Polynomial ℤ := 5 * X^2 + 7 * X + 6
  (p₁ * p₂).coeff 3 = 47 := by
sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l3338_333866


namespace NUMINAMATH_CALUDE_representative_distribution_l3338_333818

/-- The number of ways to distribute n items into k groups with at least one item in each group -/
def distribute_with_minimum (n k : ℕ) : ℕ := sorry

/-- The number of classes from which representatives are selected -/
def num_classes : ℕ := 4

/-- The total number of student representatives to be selected -/
def total_representatives : ℕ := 6

/-- Theorem stating that the number of ways to distribute 6 representatives among 4 classes,
    with at least one representative in each class, is equal to 10 -/
theorem representative_distribution :
  distribute_with_minimum total_representatives num_classes = 10 := by sorry

end NUMINAMATH_CALUDE_representative_distribution_l3338_333818


namespace NUMINAMATH_CALUDE_divisible_by_eight_count_l3338_333865

theorem divisible_by_eight_count : 
  (Finset.filter (fun n => n % 8 = 0) (Finset.Icc 200 400)).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_count_l3338_333865


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l3338_333829

theorem compare_sqrt_expressions : 7 * Real.sqrt 2 < 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l3338_333829


namespace NUMINAMATH_CALUDE_min_value_a_l3338_333824

theorem min_value_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - a) ≥ 7) ↔ a ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l3338_333824


namespace NUMINAMATH_CALUDE_no_obtuse_angle_at_center_l3338_333812

/-- Represents a point on a circle -/
structure CirclePoint where
  arc : Fin 3
  position : ℝ
  h_position : 0 ≤ position ∧ position < 2 * Real.pi / 3

/-- Represents a configuration of 6 points on a circle -/
def CircleConfiguration := Fin 6 → CirclePoint

/-- Checks if three points form an obtuse angle at the center -/
def has_obtuse_angle_at_center (config : CircleConfiguration) (p1 p2 p3 : Fin 6) : Prop :=
  ∃ θ, θ > Real.pi / 2 ∧
    θ = min (2 * Real.pi / 3) (abs ((config p2).position - (config p1).position) +
      abs ((config p3).position - (config p2).position) +
      abs ((config p1).position - (config p3).position))

/-- The main theorem statement -/
theorem no_obtuse_angle_at_center (config : CircleConfiguration) :
  ∀ p1 p2 p3 : Fin 6, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  ¬(has_obtuse_angle_at_center config p1 p2 p3) :=
sorry

end NUMINAMATH_CALUDE_no_obtuse_angle_at_center_l3338_333812


namespace NUMINAMATH_CALUDE_roots_on_circle_l3338_333830

open Complex

theorem roots_on_circle : ∃ (r : ℝ), r = 2 * Real.sqrt 3 / 3 ∧
  ∀ (z : ℂ), (z - 2) ^ 6 = 64 * z ^ 6 →
    ∃ (c : ℂ), abs (z - c) = r :=
by sorry

end NUMINAMATH_CALUDE_roots_on_circle_l3338_333830


namespace NUMINAMATH_CALUDE_line_equation_through_points_l3338_333876

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = x₁ + t * (x₂ - x₁) ∧ p.2 = y₁ + t * (y₂ - y₁)}

-- Define the general form of a line equation
def general_line_equation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem line_equation_through_points :
  line_through_points 2 5 0 3 = general_line_equation 1 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l3338_333876


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l3338_333817

theorem difference_of_squares_factorization (m : ℤ) : m^2 - 1 = (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l3338_333817


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l3338_333851

theorem largest_n_binomial_sum : ∃ (n : ℕ), (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m → m ≤ n) ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l3338_333851


namespace NUMINAMATH_CALUDE_frog_border_probability_l3338_333828

/-- Represents a position on the 4x4 grid -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines the grid and movement rules -/
def Grid :=
  { pos : Position // true }

/-- Determines if a position is on the border of the grid -/
def is_border (pos : Position) : Bool :=
  pos.x = 0 ∨ pos.x = 3 ∨ pos.y = 0 ∨ pos.y = 3

/-- Calculates the next position after a hop in a given direction -/
def next_position (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Up => ⟨(pos.x + 1) % 4, pos.y⟩
  | Direction.Down => ⟨(pos.x + 3) % 4, pos.y⟩
  | Direction.Left => ⟨pos.x, (pos.y + 3) % 4⟩
  | Direction.Right => ⟨pos.x, (pos.y + 1) % 4⟩

/-- Calculates the probability of reaching the border within n hops -/
def border_probability (start : Position) (n : Nat) : Rat :=
  sorry

/-- The main theorem to be proved -/
theorem frog_border_probability :
  border_probability ⟨1, 1⟩ 3 = 39 / 64 :=
sorry

end NUMINAMATH_CALUDE_frog_border_probability_l3338_333828


namespace NUMINAMATH_CALUDE_projectile_max_height_l3338_333880

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

-- Theorem statement
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l3338_333880


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l3338_333845

theorem cubic_equation_one_real_root (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 2*a*x + a^2 - 1 = 0) → a < 3/4 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l3338_333845


namespace NUMINAMATH_CALUDE_power_division_sum_product_difference_l3338_333881

theorem power_division_sum_product_difference (a b c d e f g : ℤ) :
  a = -4 ∧ b = 4 ∧ c = 2 ∧ d = 3 ∧ e = 7 →
  a^6 / b^4 + c^5 * d - e^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_product_difference_l3338_333881


namespace NUMINAMATH_CALUDE_f_6_equals_16_l3338_333856

def f : ℕ → ℕ 
  | x => if x < 5 then 2^x else f (x-1)

theorem f_6_equals_16 : f 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_16_l3338_333856


namespace NUMINAMATH_CALUDE_richards_average_touchdowns_l3338_333827

def archie_record : ℕ := 89
def total_games : ℕ := 16
def richards_games : ℕ := 14
def remaining_games : ℕ := 2
def remaining_avg : ℕ := 3

theorem richards_average_touchdowns :
  let total_touchdowns := archie_record + 1
  let remaining_touchdowns := remaining_games * remaining_avg
  let richards_touchdowns := total_touchdowns - remaining_touchdowns
  (richards_touchdowns : ℚ) / richards_games = 6 := by sorry

end NUMINAMATH_CALUDE_richards_average_touchdowns_l3338_333827


namespace NUMINAMATH_CALUDE_vector_proof_l3338_333835

/-- Given two planar vectors a and b, with a parallel to b, and a linear combination
    of these vectors with a third vector c equal to the zero vector,
    prove that c is equal to (-7, 14). -/
theorem vector_proof (a b c : ℝ × ℝ) (m : ℝ) : 
  a = (1, -2) →
  b = (2, m) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  3 • a + 2 • b + c = (0, 0) →
  c = (-7, 14) := by
  sorry

end NUMINAMATH_CALUDE_vector_proof_l3338_333835


namespace NUMINAMATH_CALUDE_imon_disentanglement_l3338_333816

/-- Represents the set of imons and their entanglements -/
structure ImonConfiguration where
  imons : Set Nat
  entangled : Nat → Nat → Bool

/-- Operation (i): Remove an imon entangled with an odd number of other imons -/
def removeOddEntangled (config : ImonConfiguration) : ImonConfiguration :=
  sorry

/-- Operation (ii): Double the set of imons -/
def doubleImons (config : ImonConfiguration) : ImonConfiguration :=
  sorry

/-- Checks if there are any entangled imons in the configuration -/
def hasEntangledImons (config : ImonConfiguration) : Bool :=
  sorry

/-- Represents a sequence of operations -/
inductive Operation
  | Remove
  | Double

theorem imon_disentanglement 
  (initial : ImonConfiguration) : 
  ∃ (ops : List Operation), 
    let final := ops.foldl (λ config op => 
      match op with
      | Operation.Remove => removeOddEntangled config
      | Operation.Double => doubleImons config
    ) initial
    ¬ hasEntangledImons final :=
  sorry

end NUMINAMATH_CALUDE_imon_disentanglement_l3338_333816


namespace NUMINAMATH_CALUDE_exist_non_adjacent_non_sharing_l3338_333825

/-- A simple graph with 17 vertices where each vertex has degree 4. -/
structure Graph17Deg4 where
  vertices : Finset (Fin 17)
  edges : Finset (Fin 17 × Fin 17)
  vertex_count : vertices.card = 17
  degree_4 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- Two vertices are adjacent if there's an edge between them. -/
def adjacent (G : Graph17Deg4) (u v : Fin 17) : Prop :=
  (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- Two vertices share a common neighbor if there exists a vertex adjacent to both. -/
def share_neighbor (G : Graph17Deg4) (u v : Fin 17) : Prop :=
  ∃ w : Fin 17, w ∈ G.vertices ∧ adjacent G u w ∧ adjacent G v w

/-- There exist two vertices that are neither adjacent nor share a common neighbor. -/
theorem exist_non_adjacent_non_sharing (G : Graph17Deg4) :
  ∃ u v : Fin 17, u ∈ G.vertices ∧ v ∈ G.vertices ∧ u ≠ v ∧
    ¬(adjacent G u v) ∧ ¬(share_neighbor G u v) := by
  sorry

end NUMINAMATH_CALUDE_exist_non_adjacent_non_sharing_l3338_333825


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3338_333868

theorem min_sum_absolute_values :
  ∀ x : ℝ, |x + 3| + |x + 6| + |x + 7| ≥ 4 ∧
  ∃ x : ℝ, |x + 3| + |x + 6| + |x + 7| = 4 :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3338_333868
