import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l890_89062

theorem quadratic_equation_properties (k m : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                   ∧ (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0) : 
  m^2 < 1 + 2*k^2 
  ∧ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                 → (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0 
                 → x₁*x₂ < 2)
  ∧ (∃ S : ℝ → ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                                       → (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0 
                                       → S m = |m| * Real.sqrt ((x₁ + x₂)^2 - 4*x₁*x₂))
     ∧ (∀ m : ℝ, S m ≤ Real.sqrt 2)
     ∧ (∃ m : ℝ, S m = Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l890_89062


namespace NUMINAMATH_CALUDE_g_monotonically_decreasing_l890_89067

/-- The function g(x) defined in terms of parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the conditions for g(x) to be monotonically decreasing -/
theorem g_monotonically_decreasing (a : ℝ) :
  (∀ x < a / 3, g_derivative a x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_g_monotonically_decreasing_l890_89067


namespace NUMINAMATH_CALUDE_number_120_more_than_third_l890_89089

theorem number_120_more_than_third : ∃ x : ℚ, x = (1/3) * x + 120 ∧ x = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_120_more_than_third_l890_89089


namespace NUMINAMATH_CALUDE_solution_difference_l890_89086

theorem solution_difference (r₁ r₂ : ℝ) : r₁^2 - 7*r₁ + 10 = 0 → r₂^2 - 7*r₂ + 10 = 0 → |r₁ - r₂| = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l890_89086


namespace NUMINAMATH_CALUDE_inequality_proof_l890_89057

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13)
  (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l890_89057


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l890_89093

theorem sum_of_squares_bound {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^4 + y^4 + z^4 = 1) : x^2 + y^2 + z^2 < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l890_89093


namespace NUMINAMATH_CALUDE_exists_same_color_four_directions_l890_89049

/-- A color in the grid -/
inductive Color
| Red
| Yellow
| Green
| Blue

/-- A position in the grid -/
structure Position where
  x : Fin 50
  y : Fin 50

/-- A coloring of the grid -/
def Coloring := Position → Color

/-- A position has a same-colored square above it -/
def has_same_color_above (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.x = p.x ∧ q.y > p.y ∧ c q = c p

/-- A position has a same-colored square below it -/
def has_same_color_below (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.x = p.x ∧ q.y < p.y ∧ c q = c p

/-- A position has a same-colored square to its left -/
def has_same_color_left (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.y = p.y ∧ q.x < p.x ∧ c q = c p

/-- A position has a same-colored square to its right -/
def has_same_color_right (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.y = p.y ∧ q.x > p.x ∧ c q = c p

/-- Main theorem: There exists a position with same-colored squares in all four directions -/
theorem exists_same_color_four_directions (c : Coloring) : 
  ∃ p : Position, 
    has_same_color_above c p ∧ 
    has_same_color_below c p ∧ 
    has_same_color_left c p ∧ 
    has_same_color_right c p := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_four_directions_l890_89049


namespace NUMINAMATH_CALUDE_shop_length_calculation_l890_89022

/-- Given a shop with specified dimensions and rent, calculate its length -/
theorem shop_length_calculation (width : ℝ) (monthly_rent : ℝ) (annual_rent_per_sqft : ℝ) :
  width = 20 →
  monthly_rent = 3600 →
  annual_rent_per_sqft = 120 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 18 :=
by sorry

end NUMINAMATH_CALUDE_shop_length_calculation_l890_89022


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l890_89099

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The number of divisors of n -/
def numDivisors (n : ℕ) : ℕ := (Finset.filter (λ d => n % d = 0) (Finset.range (n + 1))).card

/-- The number of odd divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := (Finset.filter (λ d => n % d = 0 ∧ d % 2 ≠ 0) (Finset.range (n + 1))).card

/-- The probability of choosing an odd divisor of n -/
def probOddDivisor (n : ℕ) : ℚ := numOddDivisors n / numDivisors n

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l890_89099


namespace NUMINAMATH_CALUDE_go_stones_perimeter_l890_89053

/-- The number of stones on one side of the square arrangement. -/
def side_length : ℕ := 6

/-- The number of sides in a square. -/
def num_sides : ℕ := 4

/-- The number of corners in a square. -/
def num_corners : ℕ := 4

/-- Calculates the number of stones on the perimeter of a square arrangement. -/
def perimeter_stones (n : ℕ) : ℕ := n * num_sides - num_corners

theorem go_stones_perimeter :
  perimeter_stones side_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_perimeter_l890_89053


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l890_89059

-- Define the equation
def equation (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 5

-- Define the roots of the equation
def roots (m : ℝ) : Set ℝ := {x | equation m x = 0}

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  base_positive : base > 0
  side_positive : side > 0
  isosceles : side ≥ base

-- State the theorem
theorem isosceles_triangle_perimeter : 
  ∃ (m : ℝ) (t : IsoscelesTriangle), 
    1 ∈ roots m ∧ 
    (∃ (x : ℝ), x ∈ roots m ∧ x ≠ 1) ∧
    {t.base, t.side} = roots m ∧
    t.base + 2 * t.side = 11 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l890_89059


namespace NUMINAMATH_CALUDE_percent_relation_l890_89051

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : c = 0.5 * b) :
  b = 0.5 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l890_89051


namespace NUMINAMATH_CALUDE_problem_solution_l890_89061

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 3 * t + 6) 
  (h3 : x = -6) : 
  y = 19.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l890_89061


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l890_89069

theorem baseball_card_value_decrease (x : ℝ) : 
  (1 - x / 100) * 0.9 = 0.36 → x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l890_89069


namespace NUMINAMATH_CALUDE_mrs_hilt_hot_dog_cost_l890_89002

/-- The total cost of hot dogs in cents -/
def total_cost (num_hot_dogs : ℕ) (cost_per_hot_dog : ℕ) : ℕ :=
  num_hot_dogs * cost_per_hot_dog

/-- Theorem: Mrs. Hilt's total cost for hot dogs is 300 cents -/
theorem mrs_hilt_hot_dog_cost :
  total_cost 6 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_hot_dog_cost_l890_89002


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l890_89007

/-- Represents the sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the profit as a function of unit price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

/-- Theorem stating that the profit-maximizing price is 35 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), ∀ (y : ℝ), profit y ≤ profit x ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l890_89007


namespace NUMINAMATH_CALUDE_mystery_number_proof_l890_89090

theorem mystery_number_proof (mystery : ℕ) : mystery * 24 = 173 * 240 → mystery = 1730 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_proof_l890_89090


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l890_89080

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_current : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 7) 
  (h2 : river_current = 1) 
  (h3 : round_trip_time = 1) : 
  ∃ (distance : ℝ), distance = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l890_89080


namespace NUMINAMATH_CALUDE_arithmetic_geometric_subset_l890_89066

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a3_eq_3 : a 3 = 3
  a5_eq_6 : a 5 = 6
  geometric_subset : ∃ m, (a 3) * (a m) = (a 5)^2

/-- The theorem stating that m = 9 for the given conditions -/
theorem arithmetic_geometric_subset (seq : ArithmeticSequence) :
  ∃ m, seq.a m = 12 ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_subset_l890_89066


namespace NUMINAMATH_CALUDE_blueberry_trade_l890_89024

/-- The number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 7

/-- The number of containers of blueberries that can be exchanged for zucchinis -/
def containers_per_exchange : ℕ := 7

/-- The number of zucchinis received in one exchange -/
def zucchinis_per_exchange : ℕ := 3

/-- The total number of zucchinis Natalie wants to trade for -/
def target_zucchinis : ℕ := 63

/-- The number of bushes needed to trade for the target number of zucchinis -/
def bushes_needed : ℕ := 21

theorem blueberry_trade :
  bushes_needed * containers_per_bush * zucchinis_per_exchange =
  target_zucchinis * containers_per_exchange :=
by sorry

end NUMINAMATH_CALUDE_blueberry_trade_l890_89024


namespace NUMINAMATH_CALUDE_willys_work_problem_l890_89075

/-- Willy's work problem -/
theorem willys_work_problem (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) 
  (h_total_days : total_days = 30)
  (h_daily_wage : daily_wage = 8)
  (h_daily_fine : daily_fine = 10)
  (h_no_money_owed : ∃ (days_worked : ℚ), 
    0 ≤ days_worked ∧ 
    days_worked ≤ total_days ∧ 
    days_worked * daily_wage = (total_days - days_worked) * daily_fine) :
  ∃ (days_worked : ℚ) (days_missed : ℚ),
    days_worked = 50 / 3 ∧
    days_missed = 40 / 3 ∧
    days_worked + days_missed = total_days ∧
    days_worked * daily_wage = days_missed * daily_fine :=
sorry

end NUMINAMATH_CALUDE_willys_work_problem_l890_89075


namespace NUMINAMATH_CALUDE_eight_person_round_robin_matches_l890_89083

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: An 8-person round-robin tennis tournament has 28 matches -/
theorem eight_person_round_robin_matches :
  roundRobinMatches 8 = 28 := by
  sorry

#eval roundRobinMatches 8  -- This should output 28

end NUMINAMATH_CALUDE_eight_person_round_robin_matches_l890_89083


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l890_89076

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 > 0 ∧ x - 3 < 2}
  S = Set.Ioo (-1) 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l890_89076


namespace NUMINAMATH_CALUDE_arrangements_equal_42_l890_89012

/-- The number of departments in the unit -/
def num_departments : ℕ := 3

/-- The number of people returning after training -/
def num_returning : ℕ := 2

/-- The maximum number of people that can be accommodated in each department -/
def max_per_department : ℕ := 1

/-- A function that calculates the number of different arrangements -/
def num_arrangements (n d r m : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the number of arrangements is 42 -/
theorem arrangements_equal_42 : 
  num_arrangements num_departments num_departments num_returning max_per_department = 42 :=
sorry

end NUMINAMATH_CALUDE_arrangements_equal_42_l890_89012


namespace NUMINAMATH_CALUDE_opposite_sides_range_l890_89071

/-- Two points are on opposite sides of a line if and only if the product of their distances from the line is negative. -/
def opposite_sides (x₁ y₁ x₂ y₂ a b c : ℝ) : Prop :=
  (a * x₁ + b * y₁ + c) * (a * x₂ + b * y₂ + c) < 0

/-- The theorem stating the range of m for which (1, 2) and (1, 1) are on opposite sides of 3x - y + m = 0. -/
theorem opposite_sides_range :
  ∀ m : ℝ, opposite_sides 1 2 1 1 3 (-1) m ↔ -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l890_89071


namespace NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l890_89040

theorem reciprocal_of_sqrt_two :
  (1 : ℝ) / Real.sqrt 2 = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l890_89040


namespace NUMINAMATH_CALUDE_digit_150_of_1_13_l890_89031

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => match n % 6 with
    | 0 => 0
    | 1 => 7
    | 2 => 6
    | 3 => 9
    | 4 => 2
    | 5 => 3
    | _ => 0  -- This case should never occur

theorem digit_150_of_1_13 : decimal_rep_1_13 150 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_1_13_l890_89031


namespace NUMINAMATH_CALUDE_fish_to_rice_value_l890_89003

-- Define the trade rates
def fish_to_bread_rate : ℚ := 2 / 3
def bread_to_rice_rate : ℚ := 4

-- Theorem statement
theorem fish_to_rice_value :
  fish_to_bread_rate * bread_to_rice_rate = 8 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fish_to_rice_value_l890_89003


namespace NUMINAMATH_CALUDE_fifteenth_term_binomial_expansion_l890_89019

theorem fifteenth_term_binomial_expansion : 
  let n : ℕ := 20
  let k : ℕ := 14
  let z : ℂ := -1 + Complex.I
  Nat.choose n k * (-1)^(n - k) * Complex.I^k = -38760 := by sorry

end NUMINAMATH_CALUDE_fifteenth_term_binomial_expansion_l890_89019


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l890_89070

/-- Given an arithmetic sequence starting with 3, 7, 11, ..., 
    prove that its 5th term is 19. -/
theorem fifth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℝ), 
    (a 0 = 3) →  -- First term is 3
    (a 1 = 7) →  -- Second term is 7
    (a 2 = 11) → -- Third term is 11
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) → -- Arithmetic sequence property
    a 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l890_89070


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l890_89041

theorem algebraic_expression_value (m n : ℝ) (h : m^2 + 3*n - 1 = 2) :
  2*m^2 + 6*n + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l890_89041


namespace NUMINAMATH_CALUDE_arithmetic_operations_l890_89014

theorem arithmetic_operations : 
  (-3 + 5 - (-2) = 4) ∧ 
  (-6 / (1/4) * (-4) = 96) ∧ 
  ((5/6 - 3/4 + 1/3) * (-24) = -10) ∧ 
  ((-1)^2023 - (4 - (-3)^2) / (2/7 - 1) = -8) := by
  sorry

#check arithmetic_operations

end NUMINAMATH_CALUDE_arithmetic_operations_l890_89014


namespace NUMINAMATH_CALUDE_octagon_diagonals_l890_89092

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals :
  num_diagonals 8 = 20 := by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l890_89092


namespace NUMINAMATH_CALUDE_complex_arithmetic_l890_89000

theorem complex_arithmetic : ((2 : ℂ) + 5*I + (3 : ℂ) - 3*I) - ((1 : ℂ) + 2*I) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l890_89000


namespace NUMINAMATH_CALUDE_min_value_x_l890_89042

theorem min_value_x (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
  (h4 : ∀ a b, a > 0 → b > 0 → (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))
  (h5 : ∀ a b, a > 0 → b > 0 → (4*a + b*(1 - a) = 0)) :
  x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_l890_89042


namespace NUMINAMATH_CALUDE_distance_center_to_point_l890_89074

/-- Given a circle with polar equation ρ = 4cosθ and a point P with polar coordinates (4, π/3),
    prove that the distance between the center of the circle and point P is 2√3. -/
theorem distance_center_to_point (θ : Real) (ρ : Real → Real) (P : Real × Real) :
  (ρ = fun θ => 4 * Real.cos θ) →
  P = (4, Real.pi / 3) →
  ∃ C : Real × Real, 
    (C.1 - P.1)^2 + (C.2 - P.2)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l890_89074


namespace NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l890_89077

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 6) + Complex.abs (z - Complex.I * 5) = 7) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 6) + Complex.abs (w - Complex.I * 5) = 7 ∧ Complex.abs w = 30 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l890_89077


namespace NUMINAMATH_CALUDE_optimal_strategy_with_bicycle_l890_89001

/-- The optimal strategy for two people to reach a destination with one bicycle. -/
theorem optimal_strategy_with_bicycle 
  (total_distance : ℝ) 
  (walking_speed : ℝ) 
  (cycling_speed : ℝ) 
  (ha : total_distance > 0) 
  (hw : walking_speed > 0) 
  (hc : cycling_speed > walking_speed) :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < total_distance ∧ 
    (x / walking_speed + (total_distance - x) / cycling_speed = 
     x / walking_speed + (total_distance - x) / walking_speed) ∧
    ∀ (y : ℝ), 
      0 < y → 
      y < total_distance → 
      (y / walking_speed + (total_distance - y) / cycling_speed ≥
       x / walking_speed + (total_distance - x) / cycling_speed) :=
by sorry

end NUMINAMATH_CALUDE_optimal_strategy_with_bicycle_l890_89001


namespace NUMINAMATH_CALUDE_smallest_base_for_90_in_three_digits_l890_89010

theorem smallest_base_for_90_in_three_digits : 
  ∀ b : ℕ, b > 0 → (b^2 ≤ 90 ∧ 90 < b^3) → b ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_90_in_three_digits_l890_89010


namespace NUMINAMATH_CALUDE_train_passing_time_specific_train_passing_time_l890_89039

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length > 0 → train_speed > man_speed → train_speed > 0 → man_speed ≥ 0 →
  ∃ (t : ℝ), t > 0 ∧ t < 19 ∧ t * (train_speed - man_speed) * (5 / 18) = train_length :=
by sorry

/-- Specific instance of the train passing time problem -/
theorem specific_train_passing_time :
  ∃ (t : ℝ), t > 0 ∧ t < 19 ∧ t * (68 - 8) * (5 / 18) = 300 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_specific_train_passing_time_l890_89039


namespace NUMINAMATH_CALUDE_complement_P_subset_Q_l890_89094

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- State the theorem
theorem complement_P_subset_Q : (Set.univ \ P) ⊆ Q := by sorry

end NUMINAMATH_CALUDE_complement_P_subset_Q_l890_89094


namespace NUMINAMATH_CALUDE_density_of_powers_of_two_and_three_l890_89033

open Real

theorem density_of_powers_of_two_and_three :
  ∀ (x : ℝ) (ε : ℝ), x > 0 → ε > 0 →
  ∃ (m n : ℤ), |2^m * 3^n - x| < ε :=
sorry

end NUMINAMATH_CALUDE_density_of_powers_of_two_and_three_l890_89033


namespace NUMINAMATH_CALUDE_scenario_1_scenario_2_l890_89052

-- Define the lines l₁ and l₂
def l₁ (a b x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a x y : ℝ) : Prop := (a - 1) * x + y + 2 = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (1 - a) = -b

-- Define parallelism of lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Theorem for Scenario 1
theorem scenario_1 (a b : ℝ) : 
  l₁ a b (-3) (-1) ∧ perpendicular a b → a = 2 ∧ b = 2 :=
sorry

-- Theorem for Scenario 2
theorem scenario_2 (a b : ℝ) :
  parallel a b ∧ (4 / b = -3) → a = 4 ∧ b = -4/3 :=
sorry

end NUMINAMATH_CALUDE_scenario_1_scenario_2_l890_89052


namespace NUMINAMATH_CALUDE_quadrilateral_ae_length_l890_89081

/-- Represents a convex quadrilateral ABCD with point E at the intersection of diagonals -/
structure ConvexQuadrilateral :=
  (A B C D E : ℝ × ℝ)

/-- Properties of the specific quadrilateral in the problem -/
def QuadrilateralProperties (quad : ConvexQuadrilateral) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist quad.A quad.B = 10 ∧
  dist quad.C quad.D = 15 ∧
  dist quad.A quad.C = 17 ∧
  (quad.E.1 - quad.A.1) * (quad.D.2 - quad.A.2) = (quad.E.2 - quad.A.2) * (quad.D.1 - quad.A.1) ∧
  (quad.E.1 - quad.B.1) * (quad.C.2 - quad.B.2) = (quad.E.2 - quad.B.2) * (quad.C.1 - quad.B.1)

theorem quadrilateral_ae_length 
  (quad : ConvexQuadrilateral) 
  (h : QuadrilateralProperties quad) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist quad.A quad.E = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_ae_length_l890_89081


namespace NUMINAMATH_CALUDE_min_value_f_max_value_y_l890_89008

-- Problem 1
theorem min_value_f (x : ℝ) (hx : x > 0) :
  2 / x + 2 * x ≥ 4 ∧ (2 / x + 2 * x = 4 ↔ x = 1) :=
by sorry

-- Problem 2
theorem max_value_y (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) :
  x * (1 - 3 * x) ≤ 1/12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_max_value_y_l890_89008


namespace NUMINAMATH_CALUDE_expression_simplification_l890_89016

theorem expression_simplification (x : ℚ) (h : x = -3) :
  (1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x^2 - 1)) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l890_89016


namespace NUMINAMATH_CALUDE_line_through_points_m_plus_b_l890_89036

/-- Given a line passing through points (1, 3) and (3, 7) that follows the equation y = mx + b,
    prove that m + b = 3 -/
theorem line_through_points_m_plus_b (m b : ℝ) : 
  (3 : ℝ) = m * (1 : ℝ) + b ∧ 
  (7 : ℝ) = m * (3 : ℝ) + b → 
  m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_m_plus_b_l890_89036


namespace NUMINAMATH_CALUDE_fruit_cost_prices_l890_89021

/-- Represents the cost and selling prices of fruits -/
structure FruitPrices where
  appleCost : ℚ
  appleSell : ℚ
  orangeCost : ℚ
  orangeSell : ℚ
  bananaCost : ℚ
  bananaSell : ℚ

/-- Calculates the cost prices of fruits based on selling prices and profit/loss percentages -/
def calculateCostPrices (p : FruitPrices) : Prop :=
  p.appleSell = p.appleCost - (1/6 * p.appleCost) ∧
  p.orangeSell = p.orangeCost + (1/5 * p.orangeCost) ∧
  p.bananaSell = p.bananaCost

/-- Theorem stating the correct cost prices of fruits -/
theorem fruit_cost_prices :
  ∃ (p : FruitPrices),
    p.appleSell = 15 ∧
    p.orangeSell = 20 ∧
    p.bananaSell = 10 ∧
    calculateCostPrices p ∧
    p.appleCost = 18 ∧
    p.orangeCost = 100/6 ∧
    p.bananaCost = 10 :=
  sorry

end NUMINAMATH_CALUDE_fruit_cost_prices_l890_89021


namespace NUMINAMATH_CALUDE_burger_expenditure_l890_89037

theorem burger_expenditure (total : ℝ) (movie_frac ice_cream_frac music_frac : ℚ) :
  total = 50 ∧
  movie_frac = 1/4 ∧
  ice_cream_frac = 1/6 ∧
  music_frac = 1/3 →
  total - (movie_frac + ice_cream_frac + music_frac) * total = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_burger_expenditure_l890_89037


namespace NUMINAMATH_CALUDE_max_xy_value_l890_89046

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 9*y = 12) :
  ∀ z : ℝ, z = x * y → z ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l890_89046


namespace NUMINAMATH_CALUDE_matrix_equality_proof_l890_89018

open Matrix

-- Define the condition for matrix congruence modulo 3
def congruent_mod_3 (X Y : Matrix (Fin 6) (Fin 6) ℤ) : Prop :=
  ∀ i j, (X i j - Y i j) % 3 = 0

-- Main theorem statement
theorem matrix_equality_proof (A B : Matrix (Fin 6) (Fin 6) ℤ)
  (h1 : congruent_mod_3 A (1 : Matrix (Fin 6) (Fin 6) ℤ))
  (h2 : congruent_mod_3 B (1 : Matrix (Fin 6) (Fin 6) ℤ))
  (h3 : A ^ 3 * B ^ 3 * A ^ 3 = B ^ 3) :
  A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_proof_l890_89018


namespace NUMINAMATH_CALUDE_total_cost_calculation_l890_89097

/-- The total cost of buying pens and exercise books -/
def total_cost (pen_price : ℝ) (book_price : ℝ) : ℝ :=
  2 * pen_price + 3 * book_price

/-- Theorem: The total cost of 2 pens at m yuan each and 3 exercise books at n yuan each is 2m + 3n yuan -/
theorem total_cost_calculation (m n : ℝ) : total_cost m n = 2 * m + 3 * n := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l890_89097


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_is_16_l890_89045

/-- The polynomial f(x) = x^4 - 6x^3 + 11x^2 + 12x - 20 -/
def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 12*x - 20

/-- The remainder when f(x) is divided by (x - 2) is equal to f(2) -/
theorem remainder_theorem (x : ℝ) : 
  ∃ (q : ℝ → ℝ), f x = (x - 2) * q x + f 2 := by sorry

/-- The remainder when x^4 - 6x^3 + 11x^2 + 12x - 20 is divided by x - 2 is 16 -/
theorem remainder_is_16 : f 2 = 16 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_is_16_l890_89045


namespace NUMINAMATH_CALUDE_modular_inverse_of_two_mod_127_l890_89048

theorem modular_inverse_of_two_mod_127 : ∃ x : ℕ, x < 127 ∧ (2 * x) % 127 = 1 :=
  by
    use 64
    sorry

end NUMINAMATH_CALUDE_modular_inverse_of_two_mod_127_l890_89048


namespace NUMINAMATH_CALUDE_fox_jeans_price_l890_89096

/-- The regular price of Fox jeans -/
def F : ℝ := 15

/-- The regular price of Pony jeans -/
def P : ℝ := 18

/-- The discount rate for Fox jeans -/
def discount_rate_fox : ℝ := 0.08

/-- The discount rate for Pony jeans -/
def discount_rate_pony : ℝ := 0.14

/-- The total savings on 5 pairs of jeans (3 Fox, 2 Pony) -/
def total_savings : ℝ := 8.64

theorem fox_jeans_price :
  F = 15 ∧
  P = 18 ∧
  discount_rate_fox + discount_rate_pony = 0.22 ∧
  3 * (F * discount_rate_fox) + 2 * (P * discount_rate_pony) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_fox_jeans_price_l890_89096


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l890_89004

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 15)
  (h3 : circle_diameter = 8)
  (h4 : circle_diameter ≤ rectangle_width)
  (h5 : circle_diameter ≤ rectangle_height) :
  let max_horizontal_distance := rectangle_width - circle_diameter
  let max_vertical_distance := rectangle_height - circle_diameter
  Real.sqrt (max_horizontal_distance ^ 2 + max_vertical_distance ^ 2) = Real.sqrt 193 :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l890_89004


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_P_is_640_l890_89017

-- Define the points
def Q : ℝ × ℝ := (0, 0)
def R : ℝ × ℝ := (307, 0)
def S : ℝ × ℝ := (450, 280)
def T : ℝ × ℝ := (460, 290)

-- Define the areas of the triangles
def area_PQR : ℝ := 1739
def area_PST : ℝ := 6956

-- Define the function to calculate the sum of possible x-coordinates of P
noncomputable def sum_of_x_coordinates_P : ℝ := sorry

-- Theorem statement
theorem sum_of_x_coordinates_P_is_640 :
  sum_of_x_coordinates_P = 640 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_P_is_640_l890_89017


namespace NUMINAMATH_CALUDE_complex_number_location_l890_89028

theorem complex_number_location (z : ℂ) (h : (z - 1) * Complex.I = Complex.I + 1) : 
  0 < z.re ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_location_l890_89028


namespace NUMINAMATH_CALUDE_expected_knowers_value_l890_89006

/-- The number of scientists at the conference -/
def total_scientists : ℕ := 18

/-- The number of scientists who initially know the news -/
def initial_knowers : ℕ := 10

/-- The probability that an initially unknowing scientist learns the news during the coffee break -/
def prob_learn : ℚ := 10 / 17

/-- The expected number of scientists who know the news after the coffee break -/
def expected_knowers : ℚ := initial_knowers + (total_scientists - initial_knowers) * prob_learn

theorem expected_knowers_value : expected_knowers = 248 / 17 := by sorry

end NUMINAMATH_CALUDE_expected_knowers_value_l890_89006


namespace NUMINAMATH_CALUDE_locus_of_parabola_vertices_l890_89047

/-- The locus of vertices of parabolas -/
theorem locus_of_parabola_vertices
  (a c : ℝ) (hz : a > 0) (hc : c > 0) :
  ∀ (z : ℝ), ∃ (x_z y_z : ℝ),
    (x_z = -z / (2 * a)) ∧
    (y_z = a * x_z^2 + z * x_z + c) ∧
    (y_z = -a * x_z^2 + c) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_parabola_vertices_l890_89047


namespace NUMINAMATH_CALUDE_cookie_price_is_two_l890_89027

/-- The price of each cookie in dollars, given the baking and sales conditions -/
def cookie_price (clementine_cookies jake_cookies tory_cookies total_revenue : ℕ) : ℚ :=
  total_revenue / (clementine_cookies + jake_cookies + tory_cookies)

theorem cookie_price_is_two :
  let clementine_cookies : ℕ := 72
  let jake_cookies : ℕ := 2 * clementine_cookies
  let tory_cookies : ℕ := (clementine_cookies + jake_cookies) / 2
  let total_revenue : ℕ := 648
  cookie_price clementine_cookies jake_cookies tory_cookies total_revenue = 2 := by
sorry

#eval cookie_price 72 144 108 648

end NUMINAMATH_CALUDE_cookie_price_is_two_l890_89027


namespace NUMINAMATH_CALUDE_large_positive_integer_product_l890_89038

theorem large_positive_integer_product : ∃ n : ℕ, n > 10^100 ∧ 
  (2+3)*(2^2+3^2)*(2^4-3^4)*(2^8+3^8)*(2^16-3^16)*(2^32+3^32)*(2^64-3^64) = n := by
  sorry

end NUMINAMATH_CALUDE_large_positive_integer_product_l890_89038


namespace NUMINAMATH_CALUDE_exists_product_in_A_l890_89060

/-- The set A(m, n) containing all integers of the form x^2 + mx + n for x ∈ ℤ -/
def A (m n : ℤ) : Set ℤ :=
  {y | ∃ x : ℤ, y = x^2 + m*x + n}

/-- For any integers m and n, there exist three distinct integers a, b, c in A(m, n) such that a = b * c -/
theorem exists_product_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a = b * c :=
by sorry

end NUMINAMATH_CALUDE_exists_product_in_A_l890_89060


namespace NUMINAMATH_CALUDE_plants_cost_theorem_l890_89030

/-- Calculates the final cost of plants given the original price, discount rate, tax rate, and delivery surcharge. -/
def finalCost (originalPrice discountRate taxRate deliverySurcharge : ℚ) : ℚ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let withTax := discountedPrice * (1 + taxRate)
  withTax + deliverySurcharge

/-- Theorem stating that the final cost of the plants is $440.71 given the specified conditions. -/
theorem plants_cost_theorem :
  finalCost 467 0.15 0.08 12 = 440.71 := by
  sorry

#eval finalCost 467 0.15 0.08 12

end NUMINAMATH_CALUDE_plants_cost_theorem_l890_89030


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l890_89085

/-- An even function that is monotonically increasing on the positive reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y)

/-- Theorem: For an even function that is monotonically increasing on the positive reals,
    f(-3) > f(2) > f(-1) -/
theorem even_increasing_function_inequality (f : ℝ → ℝ) 
  (hf : EvenIncreasingFunction f) : f (-3) > f 2 ∧ f 2 > f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l890_89085


namespace NUMINAMATH_CALUDE_sqrt_equals_self_implies_zero_or_one_l890_89088

theorem sqrt_equals_self_implies_zero_or_one (x : ℝ) : Real.sqrt x = x → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equals_self_implies_zero_or_one_l890_89088


namespace NUMINAMATH_CALUDE_tau_phi_sum_equation_l890_89029

/-- τ(n) represents the number of positive divisors of n -/
def tau (n : ℕ) : ℕ := sorry

/-- φ(n) represents the number of positive integers less than n and relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

/-- A predicate to check if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

theorem tau_phi_sum_equation (n : ℕ) (h : n > 1) :
  tau n + phi n = n + 1 ↔ n = 4 ∨ isPrime n := by sorry

end NUMINAMATH_CALUDE_tau_phi_sum_equation_l890_89029


namespace NUMINAMATH_CALUDE_x_value_proof_l890_89025

theorem x_value_proof (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l890_89025


namespace NUMINAMATH_CALUDE_inequality_proof_l890_89073

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) :
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l890_89073


namespace NUMINAMATH_CALUDE_sum_reciprocals_l890_89054

theorem sum_reciprocals (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a/b + b/c + c/a = 100) : 
  b/a + c/b + a/c = -101 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l890_89054


namespace NUMINAMATH_CALUDE_power_function_m_value_l890_89015

theorem power_function_m_value (m : ℕ+) (f : ℝ → ℝ) : 
  (∀ x, f x = x ^ (m.val ^ 2 + m.val)) → 
  f (Real.sqrt 2) = 2 → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_m_value_l890_89015


namespace NUMINAMATH_CALUDE_ratio_equality_l890_89078

theorem ratio_equality (x y : ℝ) (h1 : 3 * x = 5 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) :
  x / y = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l890_89078


namespace NUMINAMATH_CALUDE_tarantula_legs_tarantula_leg_count_l890_89065

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := 1000

/-- The number of baby tarantula legs in one less than 5 egg sacs -/
def total_legs : ℕ := 32000

/-- The number of egg sacs containing the total legs -/
def num_sacs : ℕ := 5 - 1

/-- Proves that a tarantula has 8 legs -/
theorem tarantula_legs : ℕ :=
  8

/-- Proves that the number of legs a tarantula has is 8 -/
theorem tarantula_leg_count : tarantula_legs = 8 := by
  sorry

end NUMINAMATH_CALUDE_tarantula_legs_tarantula_leg_count_l890_89065


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l890_89063

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l890_89063


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l890_89009

theorem camping_trip_percentage (total_students : ℕ) 
  (march_trip_percentage : ℝ) (march_over_100_percentage : ℝ)
  (june_trip_percentage : ℝ) (june_over_100_percentage : ℝ)
  (over_100_march_percentage : ℝ) :
  march_trip_percentage = 0.2 →
  march_over_100_percentage = 0.35 →
  june_trip_percentage = 0.15 →
  june_over_100_percentage = 0.4 →
  over_100_march_percentage = 0.7 →
  (march_trip_percentage + june_trip_percentage) * total_students = 
    0.35 * total_students :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l890_89009


namespace NUMINAMATH_CALUDE_melanie_dimes_l890_89098

/-- The number of dimes Melanie's mother gave her -/
def mother_gave : ℕ := sorry

theorem melanie_dimes :
  let initial : ℤ := 7
  let gave_dad : ℤ := 8
  let final : ℤ := 3
  (initial - gave_dad + mother_gave : ℤ) = final → mother_gave = 4 :=
by sorry

end NUMINAMATH_CALUDE_melanie_dimes_l890_89098


namespace NUMINAMATH_CALUDE_marlon_lollipops_l890_89011

theorem marlon_lollipops (initial_lollipops : ℕ) (kept_lollipops : ℕ) (lou_lollipops : ℕ) :
  initial_lollipops = 42 →
  kept_lollipops = 4 →
  lou_lollipops = 10 →
  (initial_lollipops - kept_lollipops - lou_lollipops : ℚ) / initial_lollipops = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_marlon_lollipops_l890_89011


namespace NUMINAMATH_CALUDE_union_equals_B_implies_B_is_real_l890_89056

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the theorem
theorem union_equals_B_implies_B_is_real (B : Set ℝ) (h : A ∪ B = B) : B = Set.univ :=
sorry

end NUMINAMATH_CALUDE_union_equals_B_implies_B_is_real_l890_89056


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l890_89013

theorem inequalities_satisfied (a b c x y z : ℝ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x*y + y*z + z*x ≤ a*b + b*c + c*a + 3) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 + 3) ∧ 
  (x*y*z ≤ a*b*c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l890_89013


namespace NUMINAMATH_CALUDE_circle_area_comparison_l890_89072

theorem circle_area_comparison (r s : ℝ) (h : 2 * r = (3 + Real.sqrt 2) * s) :
  π * r^2 = ((11 + 6 * Real.sqrt 2) / 4) * (π * s^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_comparison_l890_89072


namespace NUMINAMATH_CALUDE_min_pencils_in_box_l890_89050

theorem min_pencils_in_box (total_pencils : ℕ) (num_boxes : ℕ) (max_capacity : ℕ)
  (h1 : total_pencils = 74)
  (h2 : num_boxes = 13)
  (h3 : max_capacity = 6) :
  ∃ (min_pencils : ℕ), 
    (∀ (box : ℕ), box ≤ num_boxes → min_pencils ≤ (total_pencils / num_boxes)) ∧
    (∃ (box : ℕ), box ≤ num_boxes ∧ (total_pencils / num_boxes) - min_pencils < 1) ∧
    min_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_pencils_in_box_l890_89050


namespace NUMINAMATH_CALUDE_distribution_problem_l890_89032

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition n indistinguishable objects into k nonempty subsets. -/
def partition (n k : ℕ) : ℕ := sorry

theorem distribution_problem :
  distribute 6 4 = 1560 :=
sorry

end NUMINAMATH_CALUDE_distribution_problem_l890_89032


namespace NUMINAMATH_CALUDE_fruit_weights_correct_l890_89044

structure Fruit where
  name : String
  weight : Nat

def banana : Fruit := ⟨"banana", 170⟩
def orange : Fruit := ⟨"orange", 180⟩
def watermelon : Fruit := ⟨"watermelon", 1400⟩
def kiwi : Fruit := ⟨"kiwi", 200⟩
def apple : Fruit := ⟨"apple", 210⟩

def fruits : List Fruit := [banana, orange, watermelon, kiwi, apple]

theorem fruit_weights_correct : 
  (∀ f ∈ fruits, f.weight ∈ [170, 180, 200, 210, 1400]) ∧ 
  (watermelon.weight > banana.weight + orange.weight + kiwi.weight + apple.weight) ∧
  (orange.weight + kiwi.weight = banana.weight + apple.weight) ∧
  (orange.weight > banana.weight) ∧
  (orange.weight < kiwi.weight) := by
  sorry

end NUMINAMATH_CALUDE_fruit_weights_correct_l890_89044


namespace NUMINAMATH_CALUDE_imaginary_part_of_2i_plus_1_l890_89005

theorem imaginary_part_of_2i_plus_1 :
  let z : ℂ := 2 * Complex.I * (1 + Complex.I)
  (z.im : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2i_plus_1_l890_89005


namespace NUMINAMATH_CALUDE_xsin2x_necessary_not_sufficient_l890_89068

theorem xsin2x_necessary_not_sufficient (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (∀ x, (0 < x ∧ x < π/2) → (x * Real.sin x < 1 → x * Real.sin x * Real.sin x < 1)) ∧
  (∃ x, (0 < x ∧ x < π/2) ∧ x * Real.sin x * Real.sin x < 1 ∧ x * Real.sin x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_xsin2x_necessary_not_sufficient_l890_89068


namespace NUMINAMATH_CALUDE_mothers_biscuits_l890_89087

/-- Represents the number of biscuits in Randy's scenario -/
structure BiscuitCount where
  initial : Nat
  fromFather : Nat
  fromMother : Nat
  eatenByBrother : Nat
  final : Nat

/-- Calculates the total number of biscuits Randy had before his brother ate some -/
def totalBeforeEating (b : BiscuitCount) : Nat :=
  b.initial + b.fromFather + b.fromMother

/-- Theorem: Randy's mother gave him 15 biscuits -/
theorem mothers_biscuits (b : BiscuitCount) 
  (h1 : b.initial = 32)
  (h2 : b.fromFather = 13)
  (h3 : b.eatenByBrother = 20)
  (h4 : b.final = 40)
  (h5 : totalBeforeEating b = b.final + b.eatenByBrother) : 
  b.fromMother = 15 := by
  sorry

#check mothers_biscuits

end NUMINAMATH_CALUDE_mothers_biscuits_l890_89087


namespace NUMINAMATH_CALUDE_triangle_area_l890_89064

/-- Given a triangle with perimeter 24 and inradius 2.5, prove its area is 30 -/
theorem triangle_area (p r A : ℝ) (h1 : p = 24) (h2 : r = 2.5) (h3 : A = r * p / 2) : A = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l890_89064


namespace NUMINAMATH_CALUDE_tennis_racket_price_l890_89035

theorem tennis_racket_price
  (sneakers_cost sports_outfit_cost total_spent : ℝ)
  (racket_discount sales_tax : ℝ)
  (h1 : sneakers_cost = 200)
  (h2 : sports_outfit_cost = 250)
  (h3 : racket_discount = 0.2)
  (h4 : sales_tax = 0.1)
  (h5 : total_spent = 750)
  : ∃ (original_price : ℝ),
    (1 + sales_tax) * ((1 - racket_discount) * original_price + sneakers_cost + sports_outfit_cost) = total_spent ∧
    original_price = 255 / 0.88 :=
by sorry

end NUMINAMATH_CALUDE_tennis_racket_price_l890_89035


namespace NUMINAMATH_CALUDE_gmat_test_percentage_l890_89023

theorem gmat_test_percentage (S B N : ℝ) : 
  S = 70 → B = 60 → N = 5 → 100 - S + B - N = 85 :=
sorry

end NUMINAMATH_CALUDE_gmat_test_percentage_l890_89023


namespace NUMINAMATH_CALUDE_parallelogram_vertex_d_l890_89026

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Theorem: Given a parallelogram ABCD with vertices A(-1,-2), B(3,-1), and C(5,6), 
    the coordinates of vertex D are (1,5) -/
theorem parallelogram_vertex_d (ABCD : Parallelogram) 
    (h1 : ABCD.A = ⟨-1, -2⟩) 
    (h2 : ABCD.B = ⟨3, -1⟩) 
    (h3 : ABCD.C = ⟨5, 6⟩) : 
    ABCD.D = ⟨1, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_vertex_d_l890_89026


namespace NUMINAMATH_CALUDE_ball_probabilities_l890_89034

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 5

/-- Represents the number of yellow balls initially in the bag -/
def initial_yellow_balls : ℕ := 10

/-- Represents the total number of balls added to the bag -/
def added_balls : ℕ := 9

/-- Calculates the probability of drawing a red ball -/
def prob_red_ball : ℚ := initial_red_balls / (initial_red_balls + initial_yellow_balls)

/-- Represents the number of red balls added to the bag -/
def red_balls_added : ℕ := 7

/-- Represents the number of yellow balls added to the bag -/
def yellow_balls_added : ℕ := 2

theorem ball_probabilities :
  (prob_red_ball = 1/3) ∧
  ((initial_red_balls + red_balls_added) / (initial_red_balls + initial_yellow_balls + added_balls) =
   (initial_yellow_balls + yellow_balls_added) / (initial_red_balls + initial_yellow_balls + added_balls)) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l890_89034


namespace NUMINAMATH_CALUDE_mod_equiv_problem_l890_89020

theorem mod_equiv_problem (m : ℕ) : 
  197 * 879 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_equiv_problem_l890_89020


namespace NUMINAMATH_CALUDE_certain_number_problem_l890_89091

theorem certain_number_problem : 
  ∃ x : ℝ, 0.60 * x = 0.30 * 30 + 21 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l890_89091


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l890_89058

theorem restaurant_bill_proof (n : ℕ) (extra : ℚ) (total_bill : ℚ) : 
  n = 10 →
  extra = 3 →
  (n - 1) * ((total_bill / n) + extra) = total_bill →
  total_bill = 270 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l890_89058


namespace NUMINAMATH_CALUDE_otimes_inequality_solutions_l890_89043

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

-- Define the set of non-negative integers satisfying the inequality
def solution_set : Set ℕ := {x | otimes 2 ↑x ≥ 3}

-- Theorem statement
theorem otimes_inequality_solutions :
  solution_set = {0, 1} := by sorry

end NUMINAMATH_CALUDE_otimes_inequality_solutions_l890_89043


namespace NUMINAMATH_CALUDE_unique_cube_root_property_l890_89084

theorem unique_cube_root_property : ∃! (n : ℕ), n > 0 ∧ (∃ (a b : ℕ), 
  n = 1000 * a + b ∧ 
  b < 1000 ∧ 
  a^3 = n ∧ 
  a = n / 1000) :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_root_property_l890_89084


namespace NUMINAMATH_CALUDE_fraction_value_l890_89079

theorem fraction_value (m n : ℝ) (h : (m - 8)^2 + |n + 6| = 0) : n / m = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l890_89079


namespace NUMINAMATH_CALUDE_sehnenviereck_ungleichung_infinitely_many_equality_cases_l890_89095

theorem sehnenviereck_ungleichung (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (ha1 : a < 1) (hb1 : b < 1) (hc1 : c < 1) (hd1 : d < 1)
  (sum : a + b + c + d = 2) :
  Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ (a * c + b * d) / 2 :=
sorry

theorem infinitely_many_equality_cases :
  ∃ S : Set (ℝ × ℝ × ℝ × ℝ), Cardinal.mk S = Cardinal.mk ℝ ∧
  ∀ (a b c d : ℝ), (a, b, c, d) ∈ S →
    0 < a ∧ a < 1 ∧
    0 < b ∧ b < 1 ∧
    0 < c ∧ c < 1 ∧
    0 < d ∧ d < 1 ∧
    a + b + c + d = 2 ∧
    Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = (a * c + b * d) / 2 :=
sorry

end NUMINAMATH_CALUDE_sehnenviereck_ungleichung_infinitely_many_equality_cases_l890_89095


namespace NUMINAMATH_CALUDE_system_equation_solution_l890_89055

theorem system_equation_solution (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l890_89055


namespace NUMINAMATH_CALUDE_fraction_multiplication_l890_89082

theorem fraction_multiplication : (2 : ℚ) / 15 * 5 / 8 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l890_89082
