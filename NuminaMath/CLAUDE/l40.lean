import Mathlib

namespace NUMINAMATH_CALUDE_little_john_theorem_l40_4016

def little_john_problem (initial_amount : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) (amount_left : ℚ) : ℚ :=
  initial_amount - (given_to_each_friend * num_friends) - amount_left

theorem little_john_theorem (initial_amount : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) (amount_left : ℚ) :
  little_john_problem initial_amount given_to_each_friend num_friends amount_left =
  initial_amount - (given_to_each_friend * num_friends) - amount_left :=
by
  sorry

#eval little_john_problem 10.50 2.20 2 3.85

end NUMINAMATH_CALUDE_little_john_theorem_l40_4016


namespace NUMINAMATH_CALUDE_locus_equation_l40_4060

/-- Circle C₁ with equation x² + y² = 4 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₂ with equation (x - 3)² + y² = 81 -/
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 81}

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center.1^2 + center.2^2 = (radius + 2)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 3)^2 + center.2^2 = (9 - radius)^2

/-- The locus of centers of circles externally tangent to C₁ and internally tangent to C₂ -/
def locus : Set (ℝ × ℝ) :=
  {p | ∃ r : ℝ, externally_tangent_C₁ p r ∧ internally_tangent_C₂ p r}

theorem locus_equation : locus = {p : ℝ × ℝ | 12 * p.1^2 + 169 * p.2^2 - 36 * p.1 - 1584 = 0} := by
  sorry

end NUMINAMATH_CALUDE_locus_equation_l40_4060


namespace NUMINAMATH_CALUDE_problem_solution_l40_4079

theorem problem_solution : (((2304 + 88) - 2400)^2 : ℚ) / 121 = 64 / 121 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l40_4079


namespace NUMINAMATH_CALUDE_soccer_team_theorem_l40_4028

def soccer_team_problem (total_players starting_players first_half_subs : ℕ) : ℕ :=
  let second_half_subs := first_half_subs + (first_half_subs + 1) / 2
  let total_played := starting_players + first_half_subs + second_half_subs
  total_players - total_played

theorem soccer_team_theorem :
  soccer_team_problem 36 11 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_theorem_l40_4028


namespace NUMINAMATH_CALUDE_horseshoe_production_theorem_l40_4095

/-- Represents the manufacturing and sales scenario for horseshoes --/
structure HorseshoeScenario where
  initialOutlay : ℕ
  costPerSet : ℕ
  sellingPricePerSet : ℕ
  profit : ℕ

/-- Calculates the number of sets of horseshoes produced and sold --/
def setsProducedAndSold (scenario : HorseshoeScenario) : ℕ :=
  (scenario.profit + scenario.initialOutlay) / (scenario.sellingPricePerSet - scenario.costPerSet)

/-- Theorem stating that the number of sets produced and sold is 500 --/
theorem horseshoe_production_theorem (scenario : HorseshoeScenario) 
  (h1 : scenario.initialOutlay = 10000)
  (h2 : scenario.costPerSet = 20)
  (h3 : scenario.sellingPricePerSet = 50)
  (h4 : scenario.profit = 5000) :
  setsProducedAndSold scenario = 500 := by
  sorry

#eval setsProducedAndSold { initialOutlay := 10000, costPerSet := 20, sellingPricePerSet := 50, profit := 5000 }

end NUMINAMATH_CALUDE_horseshoe_production_theorem_l40_4095


namespace NUMINAMATH_CALUDE_falls_difference_l40_4076

/-- The number of falls for each person --/
structure Falls where
  steven : ℕ
  stephanie : ℕ
  sonya : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (f : Falls) : Prop :=
  f.steven = 3 ∧
  f.stephanie > f.steven ∧
  f.sonya = 6 ∧
  f.sonya = f.stephanie / 2 - 2

/-- The theorem to prove --/
theorem falls_difference (f : Falls) (h : satisfies_conditions f) :
  f.stephanie - f.steven = 13 := by
  sorry

end NUMINAMATH_CALUDE_falls_difference_l40_4076


namespace NUMINAMATH_CALUDE_slope_is_plus_minus_two_l40_4064

/-- The slope of a line passing through (-1,0) that intersects the parabola y^2 = 4x
    such that the midpoint of the intersection points lies on x = 3 -/
def slope_of_intersecting_line : ℝ → Prop :=
  λ k : ℝ => ∃ (x₁ x₂ y₁ y₂ : ℝ),
    -- Line equation
    y₁ = k * (x₁ + 1) ∧
    y₂ = k * (x₂ + 1) ∧
    -- Parabola equation
    y₁^2 = 4 * x₁ ∧
    y₂^2 = 4 * x₂ ∧
    -- Midpoint condition
    (x₁ + x₂) / 2 = 3

theorem slope_is_plus_minus_two :
  ∀ k : ℝ, slope_of_intersecting_line k ↔ k = 2 ∨ k = -2 :=
sorry

end NUMINAMATH_CALUDE_slope_is_plus_minus_two_l40_4064


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l40_4042

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  2 * r₁^2 + 5 * r₁ = 12 ∧
  2 * r₂^2 + 5 * r₂ = 12 ∧
  abs (r₁ - r₂) = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l40_4042


namespace NUMINAMATH_CALUDE_committee_arrangement_count_l40_4021

def committee_size : ℕ := 10
def num_men : ℕ := 3
def num_women : ℕ := 7

theorem committee_arrangement_count :
  (committee_size.choose num_men) = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_count_l40_4021


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_2_plus_sqrt2_min_value_achieved_l40_4046

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(2*b) = 2 → x + 4*y ≤ a + 4*b :=
by sorry

theorem min_value_is_2_plus_sqrt2 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) :
  x + 4*y ≥ 2 + Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 2 ∧ a + 4*b = 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_2_plus_sqrt2_min_value_achieved_l40_4046


namespace NUMINAMATH_CALUDE_factory_output_doubling_time_l40_4055

theorem factory_output_doubling_time (growth_rate : ℝ) (doubling_time : ℝ) :
  growth_rate = 0.1 →
  (1 + growth_rate) ^ doubling_time = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_factory_output_doubling_time_l40_4055


namespace NUMINAMATH_CALUDE_baseball_team_points_l40_4057

/- Define the structure of the team -/
structure BaseballTeam where
  totalPlayers : Nat
  totalPoints : Nat
  startingPlayers : Nat
  reservePlayers : Nat
  rookiePlayers : Nat
  totalGames : Nat

/- Define the theorem -/
theorem baseball_team_points 
  (team : BaseballTeam)
  (h1 : team.totalPlayers = 15)
  (h2 : team.totalPoints = 900)
  (h3 : team.startingPlayers = 7)
  (h4 : team.reservePlayers = 3)
  (h5 : team.rookiePlayers = 5)
  (h6 : team.totalGames = 20) :
  ∃ (startingAvg reserveAvg rookieAvg : ℕ),
    (startingAvg * team.startingPlayers * team.totalGames +
     reserveAvg * team.reservePlayers * 15 +
     rookieAvg * team.rookiePlayers * ((20 + 10 + 10 + 5 + 5) / 5) + 
     (10 + 15 + 15)) = team.totalPoints := by
  sorry


end NUMINAMATH_CALUDE_baseball_team_points_l40_4057


namespace NUMINAMATH_CALUDE_income_percentage_difference_l40_4091

/-- Given the monthly incomes of A, B, and C, prove that B's income is 12% more than C's -/
theorem income_percentage_difference :
  ∀ (a b c : ℝ),
  -- A's and B's monthly incomes are in the ratio 5:2
  a / b = 5 / 2 →
  -- C's monthly income is 12000
  c = 12000 →
  -- A's annual income is 403200.0000000001
  12 * a = 403200.0000000001 →
  -- B's monthly income is 12% more than C's
  b = 1.12 * c := by
sorry


end NUMINAMATH_CALUDE_income_percentage_difference_l40_4091


namespace NUMINAMATH_CALUDE_perfect_square_implies_congruence_l40_4001

theorem perfect_square_implies_congruence (p a : ℕ) (h_prime : Nat.Prime p) :
  (∃ t : ℤ, ∃ k : ℤ, p * t + a = k^2) →
  a^((p - 1) / 2) ≡ 1 [ZMOD p] :=
sorry

end NUMINAMATH_CALUDE_perfect_square_implies_congruence_l40_4001


namespace NUMINAMATH_CALUDE_cubic_one_real_root_l40_4072

theorem cubic_one_real_root (a b : ℝ) : 
  (∃! x : ℝ, x^3 - a*x + b = 0) ↔ 
  ((a = 0 ∧ b = 2) ∨ (a = -3 ∧ b = 2) ∨ (a = 3 ∧ b = -3)) :=
sorry

end NUMINAMATH_CALUDE_cubic_one_real_root_l40_4072


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l40_4018

/-- Represents a cyclist with a constant speed --/
structure Cyclist where
  speed : ℝ

/-- Calculates the distance traveled by a cyclist in a given time --/
def distance_traveled (cyclist : Cyclist) (time : ℝ) : ℝ :=
  cyclist.speed * time

/-- Theorem: The difference in distance traveled between two cyclists after 5 hours --/
theorem cyclist_distance_difference 
  (carlos dana : Cyclist)
  (h1 : carlos.speed = 0.9)
  (h2 : dana.speed = 0.72)
  : distance_traveled carlos 5 - distance_traveled dana 5 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l40_4018


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l40_4096

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 5 = 9 →
  a 2 + a 4 + a 6 = 15 →
  a 3 + a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l40_4096


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l40_4034

/-- M is a positive integer such that M^2 = 36^50 * 50^36 -/
def M : ℕ+ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of M is 21 -/
theorem sum_of_digits_M : sum_of_digits M.val = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l40_4034


namespace NUMINAMATH_CALUDE_perpendicular_slope_l40_4070

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let m₁ := a / b
  let m₂ := -1 / m₁
  (a * x - b * y = c) → (m₂ = -b / a) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l40_4070


namespace NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_negation_q_l40_4002

theorem negation_p_sufficient_not_necessary_for_negation_q :
  ∃ (x : ℝ),
    (∀ x, (|x + 1| > 0 → (5*x - 6 > x^2)) →
      (x = -1 → (x ≤ 2 ∨ x ≥ 3)) ∧
      ¬(x ≤ 2 ∨ x ≥ 3 → x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_negation_q_l40_4002


namespace NUMINAMATH_CALUDE_divisor_property_l40_4054

theorem divisor_property (x y : ℕ) (h1 : x % 63 = 11) (h2 : x % y = 2) :
  ∃ (k : ℕ), y ∣ (63 * k + 9) := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l40_4054


namespace NUMINAMATH_CALUDE_A_can_win_with_5_A_cannot_win_with_6_or_more_min_k_for_A_cannot_win_l40_4008

/-- Represents a hexagonal grid game. -/
structure HexGame where
  k : ℕ
  -- Add other necessary components of the game state

/-- Defines a valid move for Player A. -/
def valid_move_A (game : HexGame) : Prop :=
  -- Define conditions for a valid move by Player A
  sorry

/-- Defines a valid move for Player B. -/
def valid_move_B (game : HexGame) : Prop :=
  -- Define conditions for a valid move by Player B
  sorry

/-- Defines the winning condition for Player A. -/
def A_wins (game : HexGame) : Prop :=
  -- Define the condition when Player A wins
  sorry

/-- States that Player A can win in a finite number of moves for k = 5. -/
theorem A_can_win_with_5 : 
  ∃ (game : HexGame), game.k = 5 ∧ A_wins game :=
sorry

/-- States that Player A cannot win in a finite number of moves for k ≥ 6. -/
theorem A_cannot_win_with_6_or_more :
  ∀ (game : HexGame), game.k ≥ 6 → ¬(A_wins game) :=
sorry

/-- The main theorem stating that 6 is the minimum value of k for which
    Player A cannot win in a finite number of moves. -/
theorem min_k_for_A_cannot_win : 
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (game : HexGame), game.k ≥ k → ¬(A_wins game)) ∧
  (∀ (k' : ℕ), k' < k → ∃ (game : HexGame), game.k = k' ∧ A_wins game) :=
sorry

end NUMINAMATH_CALUDE_A_can_win_with_5_A_cannot_win_with_6_or_more_min_k_for_A_cannot_win_l40_4008


namespace NUMINAMATH_CALUDE_smallest_value_sum_of_fractions_lower_bound_achievable_l40_4075

theorem smallest_value_sum_of_fractions (a b : ℤ) (h : a > b) :
  (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) ≥ 2 :=
sorry

theorem lower_bound_achievable :
  ∃ (a b : ℤ), a > b ∧ (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_sum_of_fractions_lower_bound_achievable_l40_4075


namespace NUMINAMATH_CALUDE_triangle_dot_product_l40_4023

/-- Given a triangle ABC with side lengths a, b, c, prove that if a = 2, b - c = 1,
    and the area of the triangle is √3, then the dot product of vectors AB and AC is 13/4 -/
theorem triangle_dot_product (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b - c = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  b * c * Real.cos A = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l40_4023


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l40_4044

/-- Given three points on the graph of y = -4/x, prove their y-coordinates' relationship -/
theorem inverse_proportion_y_relationship 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -4 / x₁)
  (h2 : y₂ = -4 / x₂)
  (h3 : y₃ = -4 / x₃)
  (hx : x₁ < 0 ∧ 0 < x₂ ∧ x₂ < x₃) :
  y₁ > y₃ ∧ y₃ > y₂ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l40_4044


namespace NUMINAMATH_CALUDE_min_omega_value_l40_4059

theorem min_omega_value (y : ℝ → ℝ) (ω : ℝ) :
  (∀ x, y x = 2 * Real.sin (ω * x + π / 3)) →
  ω > 0 →
  (∀ x, y x = y (x - π / 3)) →
  (∃ k : ℕ, ω = 6 * k) →
  6 ≤ ω :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l40_4059


namespace NUMINAMATH_CALUDE_circle_intersection_range_l40_4024

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 25 - r^2 = 0

-- Define the condition for common points
def have_common_points (r : ℝ) : Prop :=
  ∃ x y, circle1 x y ∧ circle2 x y r

-- State the theorem
theorem circle_intersection_range :
  ∃ m n : ℝ, (∀ r, have_common_points r ↔ m ≤ r ∧ r ≤ n) ∧ n - m = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l40_4024


namespace NUMINAMATH_CALUDE_latest_time_60_degrees_l40_4081

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 40

-- State the theorem
theorem latest_time_60_degrees :
  ∃ t : ℝ, t ≤ 12 ∧ t ≥ 0 ∧ temperature t = 60 ∧
  ∀ s : ℝ, s > t ∧ s ≥ 0 → temperature s ≠ 60 :=
by sorry

end NUMINAMATH_CALUDE_latest_time_60_degrees_l40_4081


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l40_4077

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5 / 11) 
  (h2 : x - y = 1 / 101) : 
  x^2 - y^2 = 5 / 1111 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l40_4077


namespace NUMINAMATH_CALUDE_day_of_week_problem_l40_4069

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year :=
  (number : ℤ)

/-- Represents a day in a year -/
structure DayInYear :=
  (year : Year)
  (dayNumber : ℕ)
  (dayOfWeek : DayOfWeek)

def isLeapYear (y : Year) : Prop := sorry

theorem day_of_week_problem 
  (M : Year)
  (day250_M : DayInYear)
  (day150_M_plus_2 : DayInYear)
  (day50_M_minus_1 : DayInYear)
  (h1 : day250_M.year = M)
  (h2 : day250_M.dayNumber = 250)
  (h3 : day250_M.dayOfWeek = DayOfWeek.Friday)
  (h4 : ¬ isLeapYear M)
  (h5 : day150_M_plus_2.year.number = M.number + 2)
  (h6 : day150_M_plus_2.dayNumber = 150)
  (h7 : day150_M_plus_2.dayOfWeek = DayOfWeek.Friday)
  (h8 : day50_M_minus_1.year.number = M.number - 1)
  (h9 : day50_M_minus_1.dayNumber = 50) :
  day50_M_minus_1.dayOfWeek = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_problem_l40_4069


namespace NUMINAMATH_CALUDE_chicken_entree_cost_l40_4086

/-- Calculates the cost of each chicken entree given the wedding catering constraints. -/
theorem chicken_entree_cost
  (total_guests : ℕ)
  (steak_to_chicken_ratio : ℕ)
  (steak_cost : ℕ)
  (total_budget : ℕ)
  (h_total_guests : total_guests = 80)
  (h_ratio : steak_to_chicken_ratio = 3)
  (h_steak_cost : steak_cost = 25)
  (h_total_budget : total_budget = 1860) :
  (total_budget - steak_cost * (steak_to_chicken_ratio * total_guests / (steak_to_chicken_ratio + 1))) /
  (total_guests / (steak_to_chicken_ratio + 1)) = 18 := by
  sorry

#check chicken_entree_cost

end NUMINAMATH_CALUDE_chicken_entree_cost_l40_4086


namespace NUMINAMATH_CALUDE_stock_price_change_l40_4007

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_decrease := initial_price * (1 - 0.05)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 4.5 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l40_4007


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l40_4043

theorem cubic_root_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 504 * r + 1008 = 0) →
  (6 * s^3 + 504 * s + 1008 = 0) →
  (6 * t^3 + 504 * t + 1008 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l40_4043


namespace NUMINAMATH_CALUDE_fraction_evaluation_l40_4031

theorem fraction_evaluation : (18 : ℝ) / (4.9 * 106) = 18 / 519.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l40_4031


namespace NUMINAMATH_CALUDE_tessellation_theorem_l40_4066

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  interiorAngle : ℝ

/-- Checks if two regular polygons can tessellate -/
def canTessellate (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n1 n2 : ℕ), n1 * p1.interiorAngle + n2 * p2.interiorAngle = 360

theorem tessellation_theorem :
  let triangle : RegularPolygon := ⟨3, 60⟩
  let square : RegularPolygon := ⟨4, 90⟩
  let hexagon : RegularPolygon := ⟨6, 120⟩
  let octagon : RegularPolygon := ⟨8, 135⟩

  (canTessellate triangle square) ∧
  (canTessellate triangle hexagon) ∧
  (canTessellate octagon square) ∧
  ¬(canTessellate hexagon square) :=
by sorry

end NUMINAMATH_CALUDE_tessellation_theorem_l40_4066


namespace NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l40_4033

/-- The equation of a potential ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (3 - m) + y^2 / (m - 1) = 1

/-- The condition on m -/
def m_condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- The equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  3 - m > 0 ∧ m - 1 > 0 ∧ m ≠ 2

/-- The m_condition is necessary but not sufficient for the equation to represent an ellipse -/
theorem m_condition_necessary_not_sufficient :
  (∀ m, is_ellipse m → m_condition m) ∧
  ¬(∀ m, m_condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l40_4033


namespace NUMINAMATH_CALUDE_work_completed_by_two_workers_l40_4000

/-- The fraction of work completed by two workers in one day -/
def work_completed_together (days_a : ℕ) (days_b : ℕ) : ℚ :=
  1 / days_a + 1 / days_b

/-- Theorem: Two workers A and B, where A takes 12 days and B takes half the time of A,
    can complete 1/4 of the work in one day when working together -/
theorem work_completed_by_two_workers :
  let days_a : ℕ := 12
  let days_b : ℕ := days_a / 2
  work_completed_together days_a days_b = 1 / 4 := by
sorry


end NUMINAMATH_CALUDE_work_completed_by_two_workers_l40_4000


namespace NUMINAMATH_CALUDE_beach_pets_l40_4088

theorem beach_pets (total : ℕ) (cat_only : ℕ) (both : ℕ) (dog_only : ℕ) (neither : ℕ) : 
  total = 522 →
  total = cat_only + both + dog_only + neither →
  (both : ℚ) / (cat_only + both) = 1/5 →
  (dog_only : ℚ) / (both + dog_only) = 7/10 →
  (neither : ℚ) / (neither + dog_only) = 1/2 →
  neither = 126 :=
by sorry

end NUMINAMATH_CALUDE_beach_pets_l40_4088


namespace NUMINAMATH_CALUDE_factor_expression_l40_4056

theorem factor_expression (x : ℝ) : 60 * x^2 + 45 * x = 15 * x * (4 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l40_4056


namespace NUMINAMATH_CALUDE_fraction_order_l40_4006

theorem fraction_order : 
  (21 : ℚ) / 17 < 18 / 13 ∧ 18 / 13 < 16 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l40_4006


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_7_32_l40_4017

theorem ten_thousandths_digit_of_7_32 : ∃ (d : ℕ), d < 10 ∧ 
  (∃ (n : ℕ), (7 : ℚ) / 32 = (n * 10 + d : ℚ) / 100000 ∧ d = 5) := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_7_32_l40_4017


namespace NUMINAMATH_CALUDE_billy_candy_per_house_l40_4058

/-- 
Given:
- Anna gets 14 pieces of candy per house
- Anna visits 60 houses
- Billy visits 75 houses
- Anna gets 15 more pieces of candy than Billy

Prove that Billy gets 11 pieces of candy per house
-/
theorem billy_candy_per_house :
  let anna_candy_per_house : ℕ := 14
  let anna_houses : ℕ := 60
  let billy_houses : ℕ := 75
  let candy_difference : ℕ := 15
  ∃ billy_candy_per_house : ℕ,
    anna_candy_per_house * anna_houses = billy_candy_per_house * billy_houses + candy_difference ∧
    billy_candy_per_house = 11 :=
by sorry

end NUMINAMATH_CALUDE_billy_candy_per_house_l40_4058


namespace NUMINAMATH_CALUDE_remainder_2345678_div_5_l40_4005

theorem remainder_2345678_div_5 : 2345678 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345678_div_5_l40_4005


namespace NUMINAMATH_CALUDE_max_points_is_four_l40_4050

/-- A configuration of points and associated real numbers satisfying the distance property -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  radii : Fin n → ℝ
  distance_property : ∀ (i j : Fin n), i ≠ j →
    Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 = radii i + radii j

/-- The maximal number of points in a valid configuration is 4 -/
theorem max_points_is_four :
  (∃ (config : PointConfiguration), config.n = 4) ∧
  (∀ (config : PointConfiguration), config.n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_points_is_four_l40_4050


namespace NUMINAMATH_CALUDE_fraction_squared_l40_4094

theorem fraction_squared (x : ℚ) : x^2 = 0.0625 → x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_squared_l40_4094


namespace NUMINAMATH_CALUDE_isosceles_triangle_unique_point_l40_4098

-- Define the triangle and point
def Triangle (A B C P : ℝ × ℝ) : Prop :=
  ∃ (s t : ℝ),
    -- Triangle ABC is isosceles with AB = AC = s
    dist A B = s ∧ dist A C = s ∧
    -- BC = t
    dist B C = t ∧
    -- Point P is inside the triangle (simplified assumption)
    true ∧
    -- AP = 2
    dist A P = 2 ∧
    -- BP = √5
    dist B P = Real.sqrt 5 ∧
    -- CP = 3
    dist C P = 3

-- The theorem to prove
theorem isosceles_triangle_unique_point 
  (A B C P : ℝ × ℝ) 
  (h : Triangle A B C P) : 
  ∃ (s t : ℝ), s = 2 * Real.sqrt 3 ∧ t = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_unique_point_l40_4098


namespace NUMINAMATH_CALUDE_distance_formula_l40_4073

/-- The distance between two points on a real number line -/
def distance (a b : ℝ) : ℝ := |b - a|

/-- Theorem: The distance between two points A and B with coordinates a and b is |b - a| -/
theorem distance_formula (a b : ℝ) : distance a b = |b - a| := by sorry

end NUMINAMATH_CALUDE_distance_formula_l40_4073


namespace NUMINAMATH_CALUDE_lcm_of_45_and_200_l40_4013

theorem lcm_of_45_and_200 : Nat.lcm 45 200 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_45_and_200_l40_4013


namespace NUMINAMATH_CALUDE_at_least_two_equal_l40_4083

theorem at_least_two_equal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : a^2 + b + c = 1/a)
  (eq2 : b^2 + c + a = 1/b)
  (eq3 : c^2 + a + b = 1/c) :
  ¬(a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l40_4083


namespace NUMINAMATH_CALUDE_perimeter_of_similar_triangle_l40_4082

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Determines if two triangles are similar -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem perimeter_of_similar_triangle (abc pqr : Triangle) :
  abc.a = abc.b ∧ 
  abc.a = 12 ∧ 
  abc.c = 14 ∧ 
  similar abc pqr ∧
  max pqr.a (max pqr.b pqr.c) = 35 →
  perimeter pqr = 95 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_similar_triangle_l40_4082


namespace NUMINAMATH_CALUDE_unique_function_existence_l40_4052

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_function_existence :
  ∃! f : RealFunction, ∀ x y : ℝ, f (x + f y) = x + y + 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l40_4052


namespace NUMINAMATH_CALUDE_original_number_proof_l40_4040

theorem original_number_proof (x : ℕ) : 
  (x + 4) % 23 = 0 → x > 0 → x = 19 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l40_4040


namespace NUMINAMATH_CALUDE_petya_wins_2021_petya_wins_l40_4047

/-- Represents the game state -/
structure GameState :=
  (piles : ℕ)

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- Defines a valid move in the game -/
def valid_move (state : GameState) : Prop :=
  state.piles ≥ 3

/-- Applies a move to the game state -/
def apply_move (state : GameState) : GameState :=
  { piles := state.piles - 2 }

/-- Determines the winner of the game -/
def winner (initial_piles : ℕ) : Player :=
  if initial_piles % 2 = 0 then Player.Vasya else Player.Petya

/-- Theorem stating that Petya wins the game with 2021 initial piles -/
theorem petya_wins_2021 : winner 2021 = Player.Petya := by
  sorry

/-- Main theorem proving Petya's victory -/
theorem petya_wins :
  ∀ (initial_state : GameState),
    initial_state.piles = 2021 →
    winner initial_state.piles = Player.Petya := by
  sorry

end NUMINAMATH_CALUDE_petya_wins_2021_petya_wins_l40_4047


namespace NUMINAMATH_CALUDE_total_price_calculation_l40_4004

/-- Calculates the total price of an order of ice-cream bars and sundaes -/
theorem total_price_calculation (ice_cream_bars sundaes : ℕ) (ice_cream_price sundae_price : ℚ) :
  ice_cream_bars = 125 →
  sundaes = 125 →
  ice_cream_price = 0.60 →
  sundae_price = 1.40 →
  ice_cream_bars * ice_cream_price + sundaes * sundae_price = 250 :=
by
  sorry

#check total_price_calculation

end NUMINAMATH_CALUDE_total_price_calculation_l40_4004


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l40_4039

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  lastTwoDigits (sumFactorials 15) = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l40_4039


namespace NUMINAMATH_CALUDE_students_per_class_l40_4015

/-- Given that John buys index cards for his students, this theorem proves
    the number of students in each class. -/
theorem students_per_class
  (total_packs : ℕ)
  (num_classes : ℕ)
  (packs_per_student : ℕ)
  (h1 : total_packs = 360)
  (h2 : num_classes = 6)
  (h3 : packs_per_student = 2) :
  total_packs / (num_classes * packs_per_student) = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_per_class_l40_4015


namespace NUMINAMATH_CALUDE_min_angle_in_prime_triangle_l40_4078

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem min_angle_in_prime_triangle (a b c : ℕ) : 
  (a + b + c = 180) →
  (is_prime a) →
  (is_prime b) →
  (is_prime c) →
  (a > b) →
  (b > c) →
  c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_angle_in_prime_triangle_l40_4078


namespace NUMINAMATH_CALUDE_child_ticket_cost_l40_4027

theorem child_ticket_cost (adult_price : ℕ) (total_sales : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) :
  adult_price = 5 →
  total_sales = 178 →
  total_tickets = 42 →
  child_tickets = 16 →
  ∃ (child_price : ℕ), child_price = 3 ∧
    total_sales = adult_price * (total_tickets - child_tickets) + child_price * child_tickets :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l40_4027


namespace NUMINAMATH_CALUDE_second_player_prevents_complete_2x2_l40_4037

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the state of a square (colored by first player, second player, or uncolored) -/
inductive SquareState
  | FirstPlayer
  | SecondPlayer
  | Uncolored

/-- Represents the game state -/
def GameState := Square → SquareState

/-- Represents a 2x2 square on the board -/
structure Square2x2 where
  topLeft : Square

/-- The strategy function for the second player -/
def secondPlayerStrategy (gs : GameState) (lastMove : Square) : Square := sorry

/-- Checks if a 2x2 square is completely colored by the first player -/
def isComplete2x2FirstPlayer (gs : GameState) (s : Square2x2) : Bool := sorry

/-- The main theorem stating that the second player can always prevent
    the first player from coloring any 2x2 square completely -/
theorem second_player_prevents_complete_2x2 :
  ∀ (numMoves : Nat) (gs : GameState),
    (∀ (s : Square), gs s = SquareState.Uncolored) →
    ∀ (moves : Fin numMoves → Square),
      let finalState := sorry  -- Final game state after all moves
      ∀ (s : Square2x2), ¬(isComplete2x2FirstPlayer finalState s) :=
sorry

end NUMINAMATH_CALUDE_second_player_prevents_complete_2x2_l40_4037


namespace NUMINAMATH_CALUDE_bruce_payment_l40_4084

/-- The amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1000 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 8 55 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l40_4084


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l40_4065

def n : ℕ := 2^5 * 3^5 * 5^4 * 7^6

theorem smallest_n_for_integer_roots :
  (∃ (a b c : ℕ), (5 * n = a^5) ∧ (6 * n = b^6) ∧ (7 * n = c^7)) ∧
  (∀ m : ℕ, m < n → ¬(∃ (x y z : ℕ), (5 * m = x^5) ∧ (6 * m = y^6) ∧ (7 * m = z^7))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l40_4065


namespace NUMINAMATH_CALUDE_salary_comparison_l40_4053

theorem salary_comparison (raja_salary : ℝ) (ram_salary : ℝ) 
  (h : ram_salary = raja_salary * 1.25) : 
  (raja_salary / ram_salary) = 0.8 := by
sorry

end NUMINAMATH_CALUDE_salary_comparison_l40_4053


namespace NUMINAMATH_CALUDE_crosswalks_per_intersection_l40_4026

/-- Given a road with intersections and crosswalks, prove the number of crosswalks per intersection. -/
theorem crosswalks_per_intersection
  (num_intersections : ℕ)
  (lines_per_crosswalk : ℕ)
  (total_lines : ℕ)
  (h1 : num_intersections = 5)
  (h2 : lines_per_crosswalk = 20)
  (h3 : total_lines = 400) :
  total_lines / lines_per_crosswalk / num_intersections = 4 :=
by sorry

end NUMINAMATH_CALUDE_crosswalks_per_intersection_l40_4026


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l40_4092

theorem binomial_coefficient_ratio (n : ℕ) (k : ℕ) : 
  n = 14 ∧ k = 4 →
  (Nat.choose n k = 1001 ∧ 
   Nat.choose n (k+1) = 2002 ∧ 
   Nat.choose n (k+2) = 3003) ∧
  ∀ m : ℕ, m > 3 → 
    ¬(∃ j : ℕ, ∀ i : ℕ, i < m → 
      Nat.choose n (j+i+1) = (i+1) * Nat.choose n j) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l40_4092


namespace NUMINAMATH_CALUDE_smallest_number_of_points_l40_4090

/-- The length of the circle -/
def circleLength : ℕ := 1956

/-- The distance between adjacent points in the sequence -/
def distanceStep : ℕ := 3

/-- The number of points required -/
def numPoints : ℕ := 2 * (circleLength / distanceStep)

/-- Theorem stating the smallest number of points satisfying the conditions -/
theorem smallest_number_of_points :
  numPoints = 1304 ∧
  ∀ n : ℕ, n < numPoints →
    ¬(∀ i : Fin n,
      ∃! j : Fin n, i ≠ j ∧ (circleLength * (i.val - j.val : ℤ) / n).natAbs % circleLength = 1 ∧
      ∃! k : Fin n, i ≠ k ∧ (circleLength * (i.val - k.val : ℤ) / n).natAbs % circleLength = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_points_l40_4090


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_C_l40_4068

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (4 - x^2)}
def B : Set ℝ := {x : ℝ | 2*x < 1}

-- State the theorem
theorem A_intersect_B_equals_C : A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_C_l40_4068


namespace NUMINAMATH_CALUDE_combine_squared_binomial_simplify_given_equation_solve_system_of_equations_l40_4049

-- Problem 1
theorem combine_squared_binomial (m n : ℝ) :
  3 * (m - n)^2 - 4 * (m - n)^2 + 3 * (m - n)^2 = 2 * (m - n)^2 :=
sorry

-- Problem 2
theorem simplify_given_equation (x y : ℝ) (h : x^2 + 2*y = 4) :
  3*x^2 + 6*y - 2 = 10 :=
sorry

-- Problem 3
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x^2 + x*y = 2) (h2 : 2*y^2 + 3*x*y = 5) :
  2*x^2 + 11*x*y + 6*y^2 = 19 :=
sorry

end NUMINAMATH_CALUDE_combine_squared_binomial_simplify_given_equation_solve_system_of_equations_l40_4049


namespace NUMINAMATH_CALUDE_smallest_valid_n_l40_4048

def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

def is_valid_n (n : ℕ) : Prop :=
  ∀ i ∈ Finset.range 6, ∃ a : ℕ, a > 0 ∧ doubling_sum a (i + 1) = n

theorem smallest_valid_n :
  is_valid_n 9765 ∧ ∀ m < 9765, ¬is_valid_n m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l40_4048


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_12_80_l40_4035

theorem gcd_lcm_sum_12_80 : Nat.gcd 12 80 + Nat.lcm 12 80 = 244 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_12_80_l40_4035


namespace NUMINAMATH_CALUDE_range_of_a_l40_4020

def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a + 7 ≥ m + 2

def proposition_q (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ = 2 ∧ x₂^2 + a*x₂ = 2

theorem range_of_a :
  ∃ S : Set ℝ, (∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔ a ∈ S) ∧
  S = {a : ℝ | -2*Real.sqrt 2 ≤ a ∧ a ≤ 1 ∨ 2*Real.sqrt 2 < a ∧ a < 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l40_4020


namespace NUMINAMATH_CALUDE_cookie_dough_thickness_l40_4080

/-- The thickness of a cylindrical layer formed by doubling the volume of a sphere
    and spreading it over a circular area. -/
theorem cookie_dough_thickness 
  (initial_radius : ℝ) 
  (final_radius : ℝ) 
  (initial_radius_value : initial_radius = 3)
  (final_radius_value : final_radius = 9) :
  let initial_volume := (4/3) * Real.pi * initial_radius^3
  let doubled_volume := 2 * initial_volume
  let final_area := Real.pi * final_radius^2
  let thickness := doubled_volume / final_area
  thickness = 8/9 := by sorry

end NUMINAMATH_CALUDE_cookie_dough_thickness_l40_4080


namespace NUMINAMATH_CALUDE_max_arrangement_length_l40_4097

def student_height : ℝ → Prop := λ h => h = 1.60 ∨ h = 1.22

def valid_arrangement (arrangement : List ℝ) : Prop :=
  (∀ i, i + 3 < arrangement.length → 
    (arrangement.take (i + 4)).sum / 4 > 1.50) ∧
  (∀ i, i + 6 < arrangement.length → 
    (arrangement.take (i + 7)).sum / 7 < 1.50)

theorem max_arrangement_length :
  ∃ (arrangement : List ℝ),
    arrangement.length = 9 ∧
    (∀ h ∈ arrangement, student_height h) ∧
    valid_arrangement arrangement ∧
    ∀ (longer_arrangement : List ℝ),
      longer_arrangement.length > 9 →
      (∀ h ∈ longer_arrangement, student_height h) →
      ¬(valid_arrangement longer_arrangement) :=
sorry

end NUMINAMATH_CALUDE_max_arrangement_length_l40_4097


namespace NUMINAMATH_CALUDE_wilsons_theorem_l40_4093

theorem wilsons_theorem (p : ℕ) (h : p > 1) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1) : ℤ) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l40_4093


namespace NUMINAMATH_CALUDE_bike_ride_distance_l40_4063

/-- Calculates the total distance traveled given the conditions of the bike ride --/
theorem bike_ride_distance (total_time : ℝ) (speed_out speed_back : ℝ) : 
  total_time = 7 ∧ speed_out = 24 ∧ speed_back = 18 →
  2 * (total_time / (1 / speed_out + 1 / speed_back)) = 144 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_distance_l40_4063


namespace NUMINAMATH_CALUDE_sum_calculation_l40_4012

theorem sum_calculation : 3 * 198 + 2 * 198 + 198 + 197 = 1385 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l40_4012


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l40_4089

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal and not undefined -/
def parallel (l1 l2 : Line) : Prop :=
  l1.b ≠ 0 ∧ l2.b ≠ 0 ∧ l1.a / l1.b = l2.a / l2.b

theorem parallel_lines_a_value :
  ∃ (a : ℝ), parallel (Line.mk 2 a (-2)) (Line.mk a (a + 4) (-4)) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l40_4089


namespace NUMINAMATH_CALUDE_sum_of_fractions_l40_4014

theorem sum_of_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 11 = (37 : ℚ) / 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l40_4014


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l40_4045

theorem salary_increase_percentage (S : ℝ) : 
  S * 1.1 = 770.0000000000001 → 
  S * (1 + 16 / 100) = 812 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l40_4045


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l40_4036

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def swap_hundreds_units (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * 1000 + b * 100 + d * 10 + c

def swap_thousands_tens (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  c * 1000 + b * 100 + a * 10 + d

theorem unique_number_satisfying_conditions (n : ℕ) :
  is_valid_number n ∧
  n + swap_hundreds_units n = 3332 ∧
  n + swap_thousands_tens n = 7886 ↔
  n = 1468 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l40_4036


namespace NUMINAMATH_CALUDE_N_divisible_by_7_and_9_l40_4074

def N : ℕ := 1234567765432  -- This is the octal representation as a decimal number

theorem N_divisible_by_7_and_9 : 
  7 ∣ N ∧ 9 ∣ N :=
sorry

end NUMINAMATH_CALUDE_N_divisible_by_7_and_9_l40_4074


namespace NUMINAMATH_CALUDE_triangle_area_l40_4071

/-- Given a triangle ABC where BC = 12 cm, AC = 5 cm, and the angle between BC and AC is 30°,
    prove that the area of the triangle is 15 square centimeters. -/
theorem triangle_area (BC AC : ℝ) (angle : Real) (h : BC = 12 ∧ AC = 5 ∧ angle = 30 * Real.pi / 180) :
  (1 / 2 : ℝ) * BC * (AC * Real.sin angle) = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l40_4071


namespace NUMINAMATH_CALUDE_three_solutions_l40_4032

/-- Represents a solution to the equation A^B = BA --/
structure Solution :=
  (A B : Nat)
  (h1 : A ≠ B)
  (h2 : A ≥ 1 ∧ A ≤ 9)
  (h3 : B ≥ 1 ∧ B ≤ 9)
  (h4 : A^B = 10*B + A)
  (h5 : 10*B + A ≠ A*B)

/-- The set of all valid solutions --/
def validSolutions : Set Solution := {s | s.A^s.B = 10*s.B + s.A ∧ s.A ≠ s.B ∧ s.A ≥ 1 ∧ s.A ≤ 9 ∧ s.B ≥ 1 ∧ s.B ≤ 9 ∧ 10*s.B + s.A ≠ s.A*s.B}

/-- Theorem stating that there are exactly three solutions --/
theorem three_solutions :
  ∃ (s1 s2 s3 : Solution),
    validSolutions = {s1, s2, s3} ∧
    ((s1.A = 2 ∧ s1.B = 5) ∨ (s1.A = 6 ∧ s1.B = 2) ∨ (s1.A = 4 ∧ s1.B = 3)) ∧
    ((s2.A = 2 ∧ s2.B = 5) ∨ (s2.A = 6 ∧ s2.B = 2) ∨ (s2.A = 4 ∧ s2.B = 3)) ∧
    ((s3.A = 2 ∧ s3.B = 5) ∨ (s3.A = 6 ∧ s3.B = 2) ∨ (s3.A = 4 ∧ s3.B = 3)) ∧
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 :=
sorry

end NUMINAMATH_CALUDE_three_solutions_l40_4032


namespace NUMINAMATH_CALUDE_standard_notation_expression_l40_4062

/-- A predicate to check if an expression conforms to standard algebraic notation -/
def is_standard_notation : String → Prop := sorry

/-- The set of given expressions -/
def expressions : Set String :=
  {"18 * b", "1 1/4 x", "-b / a^2", "m ÷ 2n"}

/-- Theorem stating that "-b / a^2" conforms to standard algebraic notation -/
theorem standard_notation_expression :
  ∃ e ∈ expressions, is_standard_notation e ∧ e = "-b / a^2" := by sorry

end NUMINAMATH_CALUDE_standard_notation_expression_l40_4062


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l40_4085

theorem decimal_arithmetic : 25.3 - 0.432 + 1.25 = 26.118 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l40_4085


namespace NUMINAMATH_CALUDE_max_area_rectangle_l40_4030

/-- Given a rectangle with integer side lengths and a perimeter of 150 feet,
    the maximum possible area is 1406 square feet. -/
theorem max_area_rectangle (x y : ℕ) : 
  x + y = 75 → x * y ≤ 1406 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l40_4030


namespace NUMINAMATH_CALUDE_circle_area_difference_l40_4067

theorem circle_area_difference (r₁ r₂ d : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 15) (h₃ : d = 8) 
  (h₄ : d = r₁ + r₂) : π * r₂^2 - π * r₁^2 = 200 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l40_4067


namespace NUMINAMATH_CALUDE_x_value_l40_4022

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l40_4022


namespace NUMINAMATH_CALUDE_max_value_at_neg_two_l40_4051

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_neg_two (c : ℝ) :
  (∀ x : ℝ, f c (-2) ≥ f c x) → c = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_neg_two_l40_4051


namespace NUMINAMATH_CALUDE_circle_center_correct_l40_4029

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y - 12 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- Theorem: The center of the circle given by the equation x^2 + 4x + y^2 - 6y - 12 = 0 is (-2, 3) -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x + 2)^2 + (y - 3)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l40_4029


namespace NUMINAMATH_CALUDE_four_Z_one_l40_4003

/-- Define the Z operation -/
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

/-- Theorem: The value of 4 Z 1 is 27 -/
theorem four_Z_one : Z 4 1 = 27 := by sorry

end NUMINAMATH_CALUDE_four_Z_one_l40_4003


namespace NUMINAMATH_CALUDE_customer_payment_proof_l40_4010

-- Define the cost price of the computer table
def cost_price : ℕ := 6500

-- Define the markup percentage
def markup_percentage : ℚ := 30 / 100

-- Define the function to calculate the selling price
def selling_price (cost : ℕ) (markup : ℚ) : ℚ :=
  cost * (1 + markup)

-- Theorem statement
theorem customer_payment_proof :
  selling_price cost_price markup_percentage = 8450 := by
  sorry

end NUMINAMATH_CALUDE_customer_payment_proof_l40_4010


namespace NUMINAMATH_CALUDE_alpha_beta_values_l40_4038

theorem alpha_beta_values (n k : ℤ) :
  let α : ℝ := π / 4 + 2 * π * (n : ℝ)
  let β : ℝ := π / 3 + 2 * π * (k : ℝ)
  (∃ m : ℤ, α = π / 4 + 2 * π * (m : ℝ)) ∧
  (∃ l : ℤ, β = π / 3 + 2 * π * (l : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_alpha_beta_values_l40_4038


namespace NUMINAMATH_CALUDE_conic_section_union_l40_4009

/-- The equation y^4 - 6x^4 = 3y^2 - 2 represents the union of a hyperbola and an ellipse -/
theorem conic_section_union (x y : ℝ) : 
  (y^4 - 6*x^4 = 3*y^2 - 2) ↔ 
  ((y^2 - 3*x^2 = 2 ∨ y^2 - 2*x^2 = 1) ∨ (y^2 + 3*x^2 = 2 ∨ y^2 + 2*x^2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_conic_section_union_l40_4009


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l40_4041

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the "in plane" relation
variable (inPlane : Line → Plane → Prop)

-- Define the intersection relation
variable (intersect : Line → Line → Prop)

-- Define our specific planes and lines
variable (α β : Plane)
variable (m n l₁ l₂ : Line)

-- State the theorem
theorem parallel_planes_condition
  (h1 : m ≠ n)
  (h2 : inPlane m α)
  (h3 : inPlane n α)
  (h4 : inPlane l₁ β)
  (h5 : inPlane l₂ β)
  (h6 : intersect l₁ l₂) :
  (lineParallel m l₁ ∧ lineParallel n l₂ → parallel α β) ∧
  ¬(parallel α β → lineParallel m l₁ ∧ lineParallel n l₂) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l40_4041


namespace NUMINAMATH_CALUDE_average_weight_problem_l40_4019

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average weight of b and c is 43. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 ∧ 
  (a + b) / 2 = 40 ∧ 
  b = 31 → 
  (b + c) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l40_4019


namespace NUMINAMATH_CALUDE_third_divisor_is_seventeen_l40_4099

theorem third_divisor_is_seventeen : ∃ (d : ℕ), d = 17 ∧ d > 11 ∧ 
  (3374 % 9 = 8) ∧ (3374 % 11 = 8) ∧ (3374 % d = 8) ∧
  (∀ (x : ℕ), x > 11 ∧ x < d → (3374 % x ≠ 8)) :=
by sorry

end NUMINAMATH_CALUDE_third_divisor_is_seventeen_l40_4099


namespace NUMINAMATH_CALUDE_total_pencils_l40_4025

/-- Given the number of pencils in different locations, prove the total number of pencils -/
theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (dan_added : ℕ)
  (h1 : drawer = 43)
  (h2 : desk_initial = 19)
  (h3 : dan_added = 16) :
  drawer + desk_initial + dan_added = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l40_4025


namespace NUMINAMATH_CALUDE_problem_solution_l40_4087

theorem problem_solution (c d : ℚ) 
  (eq1 : 4 + c = 5 - d) 
  (eq2 : 5 + d = 9 + c) : 
  4 - c = 11/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l40_4087


namespace NUMINAMATH_CALUDE_miriam_flower_care_l40_4061

/-- Calculates the number of flowers Miriam can take care of in a given number of days -/
def flowers_cared_for (hours_per_day : ℕ) (flowers_per_day : ℕ) (num_days : ℕ) : ℕ :=
  (hours_per_day * num_days) * (flowers_per_day / hours_per_day)

/-- Proves that Miriam can take care of 360 flowers in 6 days -/
theorem miriam_flower_care :
  flowers_cared_for 5 60 6 = 360 := by
  sorry

#eval flowers_cared_for 5 60 6

end NUMINAMATH_CALUDE_miriam_flower_care_l40_4061


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l40_4011

-- Define the equation
def equation (x : ℝ) : Prop := 2 * Real.log x = 7 - 2 * x

-- Define the inequality
def inequality (n : ℤ) : Prop := (n : ℝ) - 2 < (n : ℝ)

theorem greatest_integer_solution :
  (∃ x : ℝ, equation x) →
  (∃ n : ℤ, inequality n ∧ ∀ m : ℤ, inequality m → m ≤ n) ∧
  (∀ n : ℤ, inequality n → n ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l40_4011
