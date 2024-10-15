import Mathlib

namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1955_195534

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 3) * (|x| + 1) < 0
def q (x : ℝ) : Prop := |1 - x| < 2

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1955_195534


namespace NUMINAMATH_CALUDE_square_roll_around_octagon_l1955_195514

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- Represents a square -/
def Square := RegularPolygon 4

/-- Represents an octagon -/
def Octagon := RegularPolygon 8

/-- Represents the position of a point on the edge of a square -/
inductive EdgePosition
  | Bottom
  | Left
  | Top
  | Right

/-- Calculates the final position of a point after rolling around an octagon -/
def finalPosition (start : EdgePosition) : EdgePosition :=
  start  -- This is a placeholder, the actual implementation would depend on the proof

theorem square_roll_around_octagon 
  (octagon : Octagon) 
  (square : Square) 
  (start_pos : EdgePosition) :
  start_pos = EdgePosition.Bottom → 
  finalPosition start_pos = EdgePosition.Bottom :=
by sorry

end NUMINAMATH_CALUDE_square_roll_around_octagon_l1955_195514


namespace NUMINAMATH_CALUDE_parallel_lines_coefficient_l1955_195593

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (3*a - 1)*x - a*y - 1 = 0) → 
  (a = 0 ∨ a = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_coefficient_l1955_195593


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1955_195541

/-- Given a hyperbola with center at (-3, 1), one focus at (-3 + √41, 1), and one vertex at (-7, 1),
    prove that h + k + a + b = 7, where (h, k) is the center, a is the distance from the center to
    the vertex, and b² = c² - a² (c being the distance from the center to the focus). -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 ∧ 
  k = 1 ∧ 
  (h + Real.sqrt 41 - h)^2 = c^2 ∧ 
  (h - 4 - h)^2 = a^2 ∧ 
  b^2 = c^2 - a^2 → 
  h + k + a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1955_195541


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1955_195518

def total_amount : ℕ := 600
def amount_a : ℕ := 200

theorem ratio_a_to_b : 
  (amount_a : ℚ) / ((total_amount - amount_a) : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1955_195518


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l1955_195573

/-- Two circles are tangent internally if the distance between their centers
    is equal to the absolute difference of their radii. -/
def InternallyTangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

/-- The problem statement -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 3  -- radius of circle O₁
  let r₂ : ℝ := 5  -- radius of circle O₂
  let d : ℝ := 2   -- distance between centers
  InternallyTangent r₁ r₂ d :=
by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l1955_195573


namespace NUMINAMATH_CALUDE_complete_square_equation_l1955_195586

theorem complete_square_equation : ∃ (a b c : ℤ), a > 0 ∧ 
  (∀ x : ℝ, 100 * x^2 + 60 * x - 49 = 0 ↔ (a * x + b)^2 = c) ∧
  a = 10 ∧ b = 3 ∧ c = 58 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equation_l1955_195586


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_1987_l1955_195506

theorem last_three_digits_of_7_to_1987 : 7^1987 % 1000 = 543 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_1987_l1955_195506


namespace NUMINAMATH_CALUDE_smallest_integer_square_equation_l1955_195587

theorem smallest_integer_square_equation : 
  ∃ (x : ℤ), x^2 = 3*x + 78 ∧ ∀ (y : ℤ), y^2 = 3*y + 78 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_equation_l1955_195587


namespace NUMINAMATH_CALUDE_shaded_region_probability_is_half_l1955_195529

/-- Represents a game board as an equilateral triangle with six equal regions -/
structure GameBoard where
  regions : Nat
  shaded_regions : Nat

/-- Probability of an event occurring -/
def probability (favorable_outcomes : Nat) (total_outcomes : Nat) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

/-- The probability of landing on a shaded region in the game board -/
def shaded_region_probability (board : GameBoard) : ℚ :=
  probability board.shaded_regions board.regions

/-- Theorem stating that the probability of landing on a shaded region
    in the described game board is 1/2 -/
theorem shaded_region_probability_is_half :
  ∀ (board : GameBoard),
    board.regions = 6 →
    board.shaded_regions = 3 →
    shaded_region_probability board = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_probability_is_half_l1955_195529


namespace NUMINAMATH_CALUDE_f_inequality_l1955_195594

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem f_inequality (a : ℝ) (h : a ≤ -2) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → |f a x₁ - f a x₂| ≥ 4 * |x₁ - x₂| := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1955_195594


namespace NUMINAMATH_CALUDE_dice_rotation_probability_l1955_195598

/-- The number of faces on a die -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 3

/-- The total number of ways to paint a single die -/
def ways_to_paint_one_die : ℕ := num_colors ^ num_faces

/-- The total number of ways to paint two dice -/
def total_paint_combinations : ℕ := ways_to_paint_one_die ^ 2

/-- The number of ways two dice can appear identical after rotation -/
def identical_after_rotation : ℕ := 1119

/-- The probability that two independently painted dice appear identical after rotation -/
theorem dice_rotation_probability :
  (identical_after_rotation : ℚ) / total_paint_combinations = 1119 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_dice_rotation_probability_l1955_195598


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1955_195512

/-- A linear function f(x) = -x + 1 -/
def f (x : ℝ) : ℝ := -x + 1

theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : f (-2) = y1) 
  (h2 : f 2 = y2) : 
  y1 > y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1955_195512


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l1955_195546

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 62 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 62 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l1955_195546


namespace NUMINAMATH_CALUDE_soccer_league_games_l1955_195562

theorem soccer_league_games (n : ℕ) (regular_games_per_matchup : ℕ) (promotional_games_per_team : ℕ) : 
  n = 20 → 
  regular_games_per_matchup = 3 → 
  promotional_games_per_team = 3 → 
  (n * (n - 1) * regular_games_per_matchup) / 2 + n * promotional_games_per_team = 1200 := by
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1955_195562


namespace NUMINAMATH_CALUDE_jakes_balloons_l1955_195520

theorem jakes_balloons (allan_initial : ℕ) (allan_bought : ℕ) (jake_difference : ℕ) :
  allan_initial = 2 →
  allan_bought = 3 →
  jake_difference = 1 →
  allan_initial + allan_bought + jake_difference = 6 :=
by sorry

end NUMINAMATH_CALUDE_jakes_balloons_l1955_195520


namespace NUMINAMATH_CALUDE_max_x_value_l1955_195568

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 6) 
  (sum_prod_eq : x*y + x*z + y*z = 10) : 
  x ≤ 2 ∧ ∃ (y z : ℝ), x = 2 ∧ x + y + z = 6 ∧ x*y + x*z + y*z = 10 :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l1955_195568


namespace NUMINAMATH_CALUDE_parabola_min_y_l1955_195554

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y + x = (y - x)^2 + 3*(y - x) + 3

/-- The minimum y-value of the parabola -/
theorem parabola_min_y : ∃ (y_min : ℝ), y_min = -1/2 ∧
  (∀ (x y : ℝ), parabola_equation x y → y ≥ y_min) :=
sorry

end NUMINAMATH_CALUDE_parabola_min_y_l1955_195554


namespace NUMINAMATH_CALUDE_min_pumps_needed_l1955_195584

/-- Represents the rate at which water flows into the well (units per minute) -/
def M : ℝ := sorry

/-- Represents the rate at which one pump removes water (units per minute) -/
def A : ℝ := sorry

/-- Represents the initial amount of water in the well (units) -/
def W : ℝ := sorry

/-- The time it takes to empty the well with 4 pumps (minutes) -/
def time_4_pumps : ℝ := 40

/-- The time it takes to empty the well with 5 pumps (minutes) -/
def time_5_pumps : ℝ := 30

/-- The target time to empty the well (minutes) -/
def target_time : ℝ := 24

/-- Condition: 4 pumps take 40 minutes to empty the well -/
axiom condition_4_pumps : 4 * A * time_4_pumps = W + M * time_4_pumps

/-- Condition: 5 pumps take 30 minutes to empty the well -/
axiom condition_5_pumps : 5 * A * time_5_pumps = W + M * time_5_pumps

/-- Theorem: The minimum number of pumps needed to empty the well in 24 minutes is 6 -/
theorem min_pumps_needed : ∃ (n : ℕ), n * A * target_time = W + M * target_time ∧ n = 6 :=
  sorry

end NUMINAMATH_CALUDE_min_pumps_needed_l1955_195584


namespace NUMINAMATH_CALUDE_fraction_sum_negative_l1955_195547

theorem fraction_sum_negative (a b : ℝ) (h1 : a * b < 0) (h2 : a + b > 0) :
  1 / a + 1 / b < 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_negative_l1955_195547


namespace NUMINAMATH_CALUDE_randy_blocks_difference_l1955_195574

/-- Given that Randy has 95 blocks in total, uses 20 blocks for a house and 50 blocks for a tower,
    prove that he used 30 more blocks for the tower than for the house. -/
theorem randy_blocks_difference (total : ℕ) (house : ℕ) (tower : ℕ) 
    (h1 : total = 95)
    (h2 : house = 20)
    (h3 : tower = 50) :
  tower - house = 30 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_difference_l1955_195574


namespace NUMINAMATH_CALUDE_inequalities_with_negative_numbers_l1955_195501

theorem inequalities_with_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  (a^2 > b^2) ∧ (a*b > b^2) ∧ (1/a > 1/b) ∧ (1/(a+b) > 1/a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_with_negative_numbers_l1955_195501


namespace NUMINAMATH_CALUDE_red_toys_removed_l1955_195528

theorem red_toys_removed (total : ℕ) (red_after : ℕ) : 
  total = 134 →
  red_after = 88 →
  red_after = 2 * (total - red_after) →
  total - red_after - (red_after - 2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_red_toys_removed_l1955_195528


namespace NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l1955_195589

theorem fraction_of_one_third_is_one_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l1955_195589


namespace NUMINAMATH_CALUDE_two_lines_iff_m_eq_one_l1955_195567

/-- The equation x^2 - my^2 + 2x + 2y = 0 represents two lines if and only if m = 1 -/
theorem two_lines_iff_m_eq_one (m : ℝ) :
  (∃ (a b c d : ℝ), ∀ (x y : ℝ),
    (x^2 - m*y^2 + 2*x + 2*y = 0) ↔ ((a*x + b*y = 1) ∨ (c*x + d*y = 1))) ↔
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_two_lines_iff_m_eq_one_l1955_195567


namespace NUMINAMATH_CALUDE_first_company_base_rate_l1955_195552

/-- The base rate of the first telephone company -/
def base_rate_1 : ℝ := 7

/-- The per-minute rate of the first telephone company -/
def rate_1 : ℝ := 0.25

/-- The base rate of the second telephone company -/
def base_rate_2 : ℝ := 12

/-- The per-minute rate of the second telephone company -/
def rate_2 : ℝ := 0.20

/-- The number of minutes for which the bills are equal -/
def minutes : ℝ := 100

theorem first_company_base_rate :
  base_rate_1 + rate_1 * minutes = base_rate_2 + rate_2 * minutes →
  base_rate_1 = 7 := by
sorry

end NUMINAMATH_CALUDE_first_company_base_rate_l1955_195552


namespace NUMINAMATH_CALUDE_round_trip_percentage_l1955_195503

/-- Represents the percentage of ship passengers -/
@[ext] structure ShipPassengers where
  total : ℝ
  roundTrip : ℝ
  roundTripWithCar : ℝ

/-- Conditions for the ship passengers -/
def validShipPassengers (p : ShipPassengers) : Prop :=
  0 ≤ p.roundTrip ∧ p.roundTrip ≤ 100 ∧
  0 ≤ p.roundTripWithCar ∧ p.roundTripWithCar ≤ p.roundTrip ∧
  p.roundTripWithCar = 0.4 * p.roundTrip

theorem round_trip_percentage (p : ShipPassengers) 
  (h : validShipPassengers p) : 
  p.roundTrip = p.roundTripWithCar / 0.4 :=
sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l1955_195503


namespace NUMINAMATH_CALUDE_total_balloons_is_370_l1955_195525

/-- The number of remaining balloons after some burst -/
def remaining_balloons (bags : ℕ) (per_bag : ℕ) (burst : ℕ) : ℕ :=
  bags * per_bag - burst

/-- The total number of remaining balloons -/
def total_remaining_balloons : ℕ :=
  let round := remaining_balloons 5 25 5
  let long := remaining_balloons 4 35 7
  let heart := remaining_balloons 3 40 3
  round + long + heart

theorem total_balloons_is_370 : total_remaining_balloons = 370 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_is_370_l1955_195525


namespace NUMINAMATH_CALUDE_division_by_power_equals_negative_exponent_l1955_195570

theorem division_by_power_equals_negative_exponent 
  (a : ℝ) (n : ℤ) (h : a > 0) : 
  1 / (a ^ n) = a ^ (0 - n) := by sorry

end NUMINAMATH_CALUDE_division_by_power_equals_negative_exponent_l1955_195570


namespace NUMINAMATH_CALUDE_intersection_count_is_four_l1955_195550

/-- The number of distinct intersection points between two curves -/
def intersection_count (f g : ℝ × ℝ → ℝ) : ℕ :=
  sorry

/-- First equation: (x + 2y - 6)(3x - y + 4) = 0 -/
def f (p : ℝ × ℝ) : ℝ :=
  (p.1 + 2*p.2 - 6) * (3*p.1 - p.2 + 4)

/-- Second equation: (2x - 3y + 1)(x + y - 2) = 0 -/
def g (p : ℝ × ℝ) : ℝ :=
  (2*p.1 - 3*p.2 + 1) * (p.1 + p.2 - 2)

/-- Theorem stating that the two curves intersect at exactly 4 points -/
theorem intersection_count_is_four :
  intersection_count f g = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_is_four_l1955_195550


namespace NUMINAMATH_CALUDE_cube_cutting_l1955_195588

theorem cube_cutting (n s : ℕ) : n > s → n^3 - s^3 = 152 → n = 6 := by sorry

end NUMINAMATH_CALUDE_cube_cutting_l1955_195588


namespace NUMINAMATH_CALUDE_train_length_calculation_l1955_195578

/-- Calculates the length of a train given the speeds of two trains, time to clear, and length of the other train --/
theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (l2 : ℝ) (h1 : v1 = 80) (h2 : v2 = 65) (h3 : t = 7.199424046076314) (h4 : l2 = 180) : 
  ∃ l1 : ℝ, l1 = 110 ∧ (v1 + v2) * t * 1000 / 3600 = l1 + l2 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1955_195578


namespace NUMINAMATH_CALUDE_total_boys_and_girls_l1955_195558

theorem total_boys_and_girls (total_amount : ℕ) (boys : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) : 
  total_amount = 460 → 
  boys = 33 → 
  amount_per_boy = 12 → 
  amount_per_girl = 8 → 
  ∃ (girls : ℕ), boys + girls = 41 ∧ total_amount = boys * amount_per_boy + girls * amount_per_girl :=
by
  sorry


end NUMINAMATH_CALUDE_total_boys_and_girls_l1955_195558


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l1955_195561

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 245 / 35 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) ^ 2 = area_ratio ∧
    a = 1 ∧ b = 7 ∧ c = 1 ∧
    a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l1955_195561


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1955_195544

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1955_195544


namespace NUMINAMATH_CALUDE_carla_marbles_count_l1955_195536

/-- The number of marbles Carla had before -/
def initial_marbles : ℝ := 187.0

/-- The number of marbles Carla bought -/
def bought_marbles : ℝ := 134.0

/-- The total number of marbles Carla has now -/
def total_marbles : ℝ := initial_marbles + bought_marbles

/-- Theorem: The total number of marbles Carla has now is 321.0 -/
theorem carla_marbles_count : total_marbles = 321.0 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_count_l1955_195536


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1955_195596

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (2 * ↑x - 1 > 3 ∧ ↑x ≤ 2 * a - 1) ∧
    (2 * ↑y - 1 > 3 ∧ ↑y ≤ 2 * a - 1) ∧
    (2 * ↑z - 1 > 3 ∧ ↑z ≤ 2 * a - 1) ∧
    (∀ (w : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z → ¬(2 * ↑w - 1 > 3 ∧ ↑w ≤ 2 * a - 1))) →
  (3 ≤ a ∧ a < 3.5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1955_195596


namespace NUMINAMATH_CALUDE_only_statement_4_implies_negation_l1955_195591

theorem only_statement_4_implies_negation (p q : Prop) :
  -- Define the four statements
  let s1 := p ∨ q
  let s2 := p ∧ ¬q
  let s3 := ¬p ∨ q
  let s4 := ¬p ∧ ¬q
  -- Define the negation of "p or q is true"
  let neg_p_or_q := ¬(p ∨ q)
  -- The theorem: only s4 implies neg_p_or_q
  (s1 → neg_p_or_q) = False ∧
  (s2 → neg_p_or_q) = False ∧
  (s3 → neg_p_or_q) = False ∧
  (s4 → neg_p_or_q) = True :=
by
  sorry

#check only_statement_4_implies_negation

end NUMINAMATH_CALUDE_only_statement_4_implies_negation_l1955_195591


namespace NUMINAMATH_CALUDE_abc_inequality_l1955_195542

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a + b + c = 1) :
  min (a - a*b) (min (b - b*c) (c - c*a)) ≤ 1/4 ∧ 
  max (a - a*b) (max (b - b*c) (c - c*a)) ≥ 2/9 := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1955_195542


namespace NUMINAMATH_CALUDE_sin_7pi_6_minus_2alpha_l1955_195577

theorem sin_7pi_6_minus_2alpha (α : ℝ) (h : Real.sin α - Real.sqrt 3 * Real.cos α = 1) :
  Real.sin ((7 * Real.pi / 6) - 2 * α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_7pi_6_minus_2alpha_l1955_195577


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l1955_195504

theorem sqrt_sum_greater_than_sqrt_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l1955_195504


namespace NUMINAMATH_CALUDE_M_remainder_l1955_195553

/-- A function that checks if a natural number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 12 with all distinct digits -/
def M : ℕ := sorry

/-- M is a multiple of 12 -/
axiom M_multiple_of_12 : 12 ∣ M

/-- M has all distinct digits -/
axiom M_distinct_digits : has_distinct_digits M

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, 12 ∣ n → has_distinct_digits n → n ≤ M

/-- The remainder when M is divided by 2000 is 960 -/
theorem M_remainder : M % 2000 = 960 := by sorry

end NUMINAMATH_CALUDE_M_remainder_l1955_195553


namespace NUMINAMATH_CALUDE_sum_of_ages_is_48_l1955_195543

/-- The sum of ages of 4 children with a 4-year age difference -/
def sum_of_ages (eldest_age : ℕ) : ℕ :=
  eldest_age + (eldest_age - 4) + (eldest_age - 8) + (eldest_age - 12)

/-- Theorem: The sum of ages of 4 children, where each child is born 4 years apart
    and the eldest is 18 years old, is equal to 48 years. -/
theorem sum_of_ages_is_48 : sum_of_ages 18 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_48_l1955_195543


namespace NUMINAMATH_CALUDE_mixing_solutions_l1955_195564

/-- Proves that mixing 28 ounces of a 30% solution with 12 ounces of an 80% solution 
    results in a 45% solution with a total volume of 40 ounces. -/
theorem mixing_solutions (volume_30 volume_80 total_volume : ℝ) 
  (h1 : volume_30 = 28)
  (h2 : volume_80 = 12)
  (h3 : total_volume = volume_30 + volume_80) :
  (0.30 * volume_30 + 0.80 * volume_80) / total_volume = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_mixing_solutions_l1955_195564


namespace NUMINAMATH_CALUDE_factorization_equality_l1955_195569

theorem factorization_equality (x y : ℝ) : -x^2*y + 6*y*x^2 - 9*y^3 = -y*(x - 3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1955_195569


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_seven_l1955_195510

theorem smallest_m_divisible_by_seven :
  ∃ m : ℕ, m = 6 ∧
  (∀ k : ℕ, k < m → (k^3 + 3^k) % 7 ≠ 0 ∨ (k^2 + 3^k) % 7 ≠ 0) ∧
  (m^3 + 3^m) % 7 = 0 ∧ (m^2 + 3^m) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_seven_l1955_195510


namespace NUMINAMATH_CALUDE_nils_geese_count_l1955_195579

/-- Represents the number of days the feed lasts -/
def FeedDuration : ℕ → ℕ → ℕ
  | n, k => k

/-- Represents the change in feed duration when selling geese -/
def SellGeese (n : ℕ) : ℕ := FeedDuration (n - 50) (FeedDuration n 0 + 20)

/-- Represents the change in feed duration when buying geese -/
def BuyGeese (n : ℕ) : ℕ := FeedDuration (n + 100) (FeedDuration n 0 - 10)

/-- The theorem stating that Nils has 300 geese -/
theorem nils_geese_count :
  ∃ (n : ℕ), n = 300 ∧ 
  SellGeese n = FeedDuration n 0 + 20 ∧
  BuyGeese n = FeedDuration n 0 - 10 :=
sorry

end NUMINAMATH_CALUDE_nils_geese_count_l1955_195579


namespace NUMINAMATH_CALUDE_division_remainder_l1955_195513

theorem division_remainder : ∃ q : ℕ, 1234567 = 123 * q + 41 ∧ 41 < 123 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1955_195513


namespace NUMINAMATH_CALUDE_product_sum_squares_problem_l1955_195535

theorem product_sum_squares_problem :
  ∃ x y : ℝ,
    x * y = 120 ∧
    x^2 + y^2 = 289 ∧
    x + y = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_squares_problem_l1955_195535


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1955_195545

theorem sum_of_a_and_b (a b : ℝ) : 
  ({a, a^2} : Set ℝ) = ({1, b} : Set ℝ) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1955_195545


namespace NUMINAMATH_CALUDE_favorite_subject_fraction_l1955_195517

theorem favorite_subject_fraction (total_students : ℕ) 
  (math_fraction : ℚ) (english_fraction : ℚ) (no_favorite : ℕ) : 
  total_students = 30 →
  math_fraction = 1 / 5 →
  english_fraction = 1 / 3 →
  no_favorite = 12 →
  let math_students := total_students * math_fraction
  let english_students := total_students * english_fraction
  let students_with_favorite := total_students - no_favorite
  let science_students := students_with_favorite - math_students - english_students
  let remaining_students := students_with_favorite - math_students - english_students
  science_students / remaining_students = 1 := by
sorry

end NUMINAMATH_CALUDE_favorite_subject_fraction_l1955_195517


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1955_195559

/-- The smallest possible next divisor after 221 for an even 4-digit number -/
theorem smallest_next_divisor_after_221 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) 
  (h3 : Even n) (h4 : n % 221 = 0) :
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ (∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d) ∧ d = 442 := by
sorry


end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1955_195559


namespace NUMINAMATH_CALUDE_find_number_l1955_195505

theorem find_number : ∃ x : ℕ, 5 + x = 20 ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_find_number_l1955_195505


namespace NUMINAMATH_CALUDE_inverse_proportional_solution_l1955_195565

-- Define the inverse proportionality constant
def C : ℝ := 315

-- Define the relationship between x and y
def inverse_proportional (x y : ℝ) : Prop := x * y = C

-- State the theorem
theorem inverse_proportional_solution :
  ∀ x y : ℝ,
  inverse_proportional x y →
  x + y = 36 →
  x - y = 6 →
  x = 7 →
  y = 45 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportional_solution_l1955_195565


namespace NUMINAMATH_CALUDE_edwards_initial_money_l1955_195538

theorem edwards_initial_money (initial_amount : ℝ) : 
  initial_amount > 0 →
  (initial_amount * 0.6 * 0.75 * 1.2) = 28 →
  initial_amount = 77.78 := by
sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l1955_195538


namespace NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l1955_195560

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the third quadrant -/
def thirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Defines the fourth quadrant -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Given point A in the third quadrant, prove point B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant (m n : ℝ) 
  (hA : thirdQuadrant ⟨-m, n⟩) : 
  fourthQuadrant ⟨m+1, n-1⟩ := by
  sorry


end NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l1955_195560


namespace NUMINAMATH_CALUDE_bill_sunday_miles_l1955_195522

/-- Represents the number of miles run by a person on a specific day -/
structure DailyMiles where
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Calculates the total miles run over three days -/
def totalMiles (person : DailyMiles) : ℝ :=
  person.friday + person.saturday + person.sunday

theorem bill_sunday_miles (bill julia : DailyMiles) :
  bill.friday = 2 * bill.saturday →
  bill.sunday = bill.saturday + 4 →
  julia.saturday = 0 →
  julia.sunday = 2 * bill.sunday →
  julia.friday = 2 * bill.friday - 3 →
  totalMiles bill + totalMiles julia = 30 →
  bill.sunday = 6.1 := by
sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_l1955_195522


namespace NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l1955_195533

/-- The speed of a space shuttle orbiting the Earth -/
def shuttle_speed_km_per_sec : ℝ := 9

/-- The number of seconds in an hour -/
def seconds_per_hour : ℝ := 3600

/-- Theorem stating that the speed of the space shuttle in km/h is 32400 -/
theorem shuttle_speed_km_per_hour :
  shuttle_speed_km_per_sec * seconds_per_hour = 32400 := by
  sorry

end NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l1955_195533


namespace NUMINAMATH_CALUDE_longest_segment_squared_l1955_195571

-- Define the diameter of the pizza
def diameter : ℝ := 16

-- Define the number of slices
def num_slices : ℕ := 4

-- Define the longest line segment in a slice
def longest_segment (d : ℝ) (n : ℕ) : ℝ := d

-- Theorem statement
theorem longest_segment_squared (d : ℝ) (n : ℕ) :
  d = diameter → n = num_slices → (longest_segment d n)^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_squared_l1955_195571


namespace NUMINAMATH_CALUDE_other_factor_proof_l1955_195595

theorem other_factor_proof (w : ℕ) (h1 : w > 0) 
  (h2 : ∃ k : ℕ, 936 * w = 2^5 * 13^2 * k) 
  (h3 : w ≥ 156) 
  (h4 : ∀ v : ℕ, v > 0 → v < 156 → ¬(∃ k : ℕ, 936 * v = 2^5 * 13^2 * k)) : 
  ∃ m : ℕ, w = 3 * m ∧ ∃ k : ℕ, 936 * m = 2^5 * 13^2 * k := by
sorry

end NUMINAMATH_CALUDE_other_factor_proof_l1955_195595


namespace NUMINAMATH_CALUDE_phil_quarters_proof_l1955_195551

/-- Represents the number of quarters Phil collected every third month in the third year -/
def quarters_collected_third_year : ℕ := sorry

/-- The initial number of quarters Phil had -/
def initial_quarters : ℕ := 50

/-- The number of quarters Phil had after the first year -/
def quarters_after_first_year : ℕ := 2 * initial_quarters

/-- The number of quarters Phil collected in the second year -/
def quarters_collected_second_year : ℕ := 3 * 12

/-- The number of quarters Phil had after the second year -/
def quarters_after_second_year : ℕ := quarters_after_first_year + quarters_collected_second_year

/-- The number of quarters Phil had after the third year -/
def quarters_after_third_year : ℕ := quarters_after_second_year + 4 * quarters_collected_third_year

/-- The number of quarters Phil had after losing some in the fourth year -/
def final_quarters : ℕ := 105

theorem phil_quarters_proof :
  (3 * (quarters_after_third_year : ℚ)) / 4 = final_quarters :=
sorry

end NUMINAMATH_CALUDE_phil_quarters_proof_l1955_195551


namespace NUMINAMATH_CALUDE_box_volume_calculation_l1955_195539

/-- Calculates the total volume occupied by boxes given their dimensions, cost per box, and total monthly payment -/
theorem box_volume_calculation (length width height cost_per_box total_payment : ℝ) :
  length = 15 ∧ 
  width = 12 ∧ 
  height = 10 ∧ 
  cost_per_box = 0.8 ∧ 
  total_payment = 480 →
  (total_payment / cost_per_box) * (length * width * height) = 1080000 := by
  sorry

#check box_volume_calculation

end NUMINAMATH_CALUDE_box_volume_calculation_l1955_195539


namespace NUMINAMATH_CALUDE_class_size_calculation_l1955_195575

theorem class_size_calculation (total : ℕ) 
  (h1 : (40 : ℚ) / 100 * total = (↑total * (40 : ℚ) / 100).floor)
  (h2 : (70 : ℚ) / 100 * ((40 : ℚ) / 100 * total) = 21) : 
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_class_size_calculation_l1955_195575


namespace NUMINAMATH_CALUDE_expression_evaluation_l1955_195563

theorem expression_evaluation (x y : ℚ) (hx : x = 4 / 7) (hy : y = 6 / 8) :
  (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1955_195563


namespace NUMINAMATH_CALUDE_expression_equals_two_l1955_195581

theorem expression_equals_two : 2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l1955_195581


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1955_195566

theorem smallest_sum_of_a_and_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 - a*x + 3*b = 0) →
  (∃ x : ℝ, x^2 - 3*b*x + a = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x : ℝ, x^2 - a'*x + 3*b' = 0) →
    (∃ x : ℝ, x^2 - 3*b'*x + a' = 0) →
    a + b ≤ a' + b') →
  a + b = 32 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1955_195566


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1955_195585

/-- Proves that when a rectangle's length is increased by 15% and its breadth is decreased by 20%, 
    the resulting area is 92% of the original area. -/
theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (L * 1.15) * (B * 0.8) = 0.92 * (L * B) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1955_195585


namespace NUMINAMATH_CALUDE_staircase_steps_l1955_195515

/-- The number of steps Akvort skips at a time -/
def akvort_skip : ℕ := 3

/-- The number of steps Barnden skips at a time -/
def barnden_skip : ℕ := 4

/-- The number of steps Croft skips at a time -/
def croft_skip : ℕ := 5

/-- The minimum number of steps in the staircase -/
def min_steps : ℕ := 19

theorem staircase_steps :
  (min_steps + 1) % akvort_skip = 0 ∧
  (min_steps + 1) % barnden_skip = 0 ∧
  (min_steps + 1) % croft_skip = 0 ∧
  ∀ n : ℕ, n < min_steps →
    ((n + 1) % akvort_skip = 0 ∧
     (n + 1) % barnden_skip = 0 ∧
     (n + 1) % croft_skip = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_staircase_steps_l1955_195515


namespace NUMINAMATH_CALUDE_g_expression_l1955_195532

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g using the given condition
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l1955_195532


namespace NUMINAMATH_CALUDE_complex_z_magnitude_l1955_195508

theorem complex_z_magnitude (z : ℂ) (h : (1 + Complex.I)^2 * z = 1 - Complex.I^3) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_z_magnitude_l1955_195508


namespace NUMINAMATH_CALUDE_anands_age_l1955_195572

theorem anands_age (anand_age bala_age : ℕ) : 
  (anand_age - 10 = (bala_age - 10) / 3) →
  (bala_age = anand_age + 10) →
  anand_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_anands_age_l1955_195572


namespace NUMINAMATH_CALUDE_coloring_schemes_formula_l1955_195580

/-- The number of different coloring schemes for n connected regions using m colors -/
def coloringSchemes (m n : ℕ) : ℕ :=
  ((-1 : ℤ) ^ n * (m - 1 : ℤ) + (m - 1 : ℤ) ^ n).natAbs

/-- Theorem stating the number of different coloring schemes for n connected regions using m colors -/
theorem coloring_schemes_formula (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  coloringSchemes m n = ((-1 : ℤ) ^ n * (m - 1 : ℤ) + (m - 1 : ℤ) ^ n).natAbs := by
  sorry

end NUMINAMATH_CALUDE_coloring_schemes_formula_l1955_195580


namespace NUMINAMATH_CALUDE_wedding_gift_cost_l1955_195524

/-- The cost of a single steak knife set -/
def steak_knife_set_cost : ℝ := 80

/-- The number of steak knife sets Elizabeth wants to buy -/
def num_steak_knife_sets : ℕ := 2

/-- The cost of the dinnerware set -/
def dinnerware_set_cost : ℝ := 200

/-- The discount rate applied to the total purchase -/
def discount_rate : ℝ := 0.1

/-- The sales tax rate applied after the discount -/
def sales_tax_rate : ℝ := 0.05

/-- The total cost Elizabeth will spend on the wedding gift -/
def total_cost : ℝ :=
  let initial_cost := steak_knife_set_cost * num_steak_knife_sets + dinnerware_set_cost
  let discounted_cost := initial_cost * (1 - discount_rate)
  discounted_cost * (1 + sales_tax_rate)

theorem wedding_gift_cost : total_cost = 340.20 := by
  sorry

end NUMINAMATH_CALUDE_wedding_gift_cost_l1955_195524


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1955_195516

/-- Given two vectors a = (-1, 2) and b = (m, 3) where m ∈ ℝ, 
    if a ⊥ b, then m = 6 -/
theorem perpendicular_vectors_m_value :
  ∀ (m : ℝ), 
  let a : Fin 2 → ℝ := ![(-1), 2]
  let b : Fin 2 → ℝ := ![m, 3]
  (∀ (i j : Fin 2), i ≠ j → a i * b j = a j * b i) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1955_195516


namespace NUMINAMATH_CALUDE_cos_theta_minus_pi_third_l1955_195583

theorem cos_theta_minus_pi_third (θ : ℝ) 
  (h : Real.sin (3 * Real.pi - θ) = (Real.sqrt 5 / 2) * Real.sin (Real.pi / 2 + θ)) :
  Real.cos (θ - Real.pi / 3) = (1 / 3 + Real.sqrt 15 / 6) ∨ 
  Real.cos (θ - Real.pi / 3) = -(1 / 3 + Real.sqrt 15 / 6) :=
by sorry

end NUMINAMATH_CALUDE_cos_theta_minus_pi_third_l1955_195583


namespace NUMINAMATH_CALUDE_pythagorean_triple_sequence_l1955_195592

theorem pythagorean_triple_sequence (k : ℕ+) :
  ∃ (c : ℕ), (k * (2 * k - 2))^2 + (2 * k - 1)^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_sequence_l1955_195592


namespace NUMINAMATH_CALUDE_prob_non_red_twelve_sided_l1955_195548

/-- Represents a 12-sided die with colored faces -/
structure ColoredDie where
  total_faces : ℕ
  red_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ
  green_faces : ℕ
  face_sum : total_faces = red_faces + yellow_faces + blue_faces + green_faces

/-- The probability of rolling a non-red face on the given die -/
def prob_non_red (d : ColoredDie) : ℚ :=
  (d.total_faces - d.red_faces : ℚ) / d.total_faces

/-- The specific 12-sided die described in the problem -/
def twelve_sided_die : ColoredDie where
  total_faces := 12
  red_faces := 5
  yellow_faces := 4
  blue_faces := 2
  green_faces := 1
  face_sum := by rfl

theorem prob_non_red_twelve_sided : prob_non_red twelve_sided_die = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_red_twelve_sided_l1955_195548


namespace NUMINAMATH_CALUDE_initial_balls_count_l1955_195540

def process (x : ℕ) : ℕ := x / 2 + 1

def iterate_process (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | m + 1 => process (iterate_process n m)

theorem initial_balls_count (n : ℕ) : iterate_process n 2010 = 2 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_balls_count_l1955_195540


namespace NUMINAMATH_CALUDE_workbook_problems_l1955_195599

theorem workbook_problems (P : ℚ) 
  (h1 : (1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P) : 
  P = 240 :=
by sorry

end NUMINAMATH_CALUDE_workbook_problems_l1955_195599


namespace NUMINAMATH_CALUDE_centered_hexagonal_characterization_l1955_195556

/-- Definition of centered hexagonal number -/
def is_centered_hexagonal (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k^2 - 3 * k + 1

/-- Definition of arithmetic sequence -/
def is_arithmetic_seq (x y z : ℕ) : Prop :=
  y - x = z - y

/-- Definition of geometric sequence -/
def is_geometric_seq (x y z : ℕ) : Prop :=
  x * z = y^2

theorem centered_hexagonal_characterization
  (a b c d : ℕ) 
  (h_arith : is_arithmetic_seq 1 a b)
  (h_geom : is_geometric_seq 1 c d)
  (h_sum : a + b = c + d) :
  is_centered_hexagonal a ↔ ∃ k : ℕ, a = 3 * k^2 - 3 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_centered_hexagonal_characterization_l1955_195556


namespace NUMINAMATH_CALUDE_point_on_exponential_graph_l1955_195530

theorem point_on_exponential_graph (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∃ P : ℝ × ℝ, ∀ x : ℝ, a^(x + 2) = P.2 → x = P.1 → P = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_exponential_graph_l1955_195530


namespace NUMINAMATH_CALUDE_problem_solution_l1955_195576

theorem problem_solution (x : ℝ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1955_195576


namespace NUMINAMATH_CALUDE_binomial_coefficient_relation_l1955_195521

theorem binomial_coefficient_relation :
  ∀ n : ℤ, n > 3 →
  ∃ m : ℤ, m > 1 ∧
    Nat.choose (m.toNat) 2 = 3 * Nat.choose (n.toNat) 4 ∧
    m = (n^2 - 3*n + 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_relation_l1955_195521


namespace NUMINAMATH_CALUDE_two_color_theorem_l1955_195509

/-- A type representing a region in the plane --/
structure Region

/-- A type representing a color --/
inductive Color
| Red
| Blue

/-- A type representing a line or circle --/
inductive Divider
| Line
| Circle

/-- Predicate to check if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop := sorry

/-- Function to represent a coloring of regions --/
def coloring (R : Set Region) : Region → Color := sorry

/-- The main theorem --/
theorem two_color_theorem (S : Set Divider) :
  ∃ (R : Set Region) (c : Region → Color),
    (∀ r1 r2 : Region, r1 ∈ R → r2 ∈ R → adjacent r1 r2 → c r1 ≠ c r2) :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l1955_195509


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l1955_195523

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z / (1 + 2*I) = 1 - 2*I) : 
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l1955_195523


namespace NUMINAMATH_CALUDE_statistical_properties_l1955_195555

def data1 : List ℝ := [3, 5, 7, 9]
def data2 : List ℝ := [6, 10, 14, 18]
def data3 : List ℝ := [4, 6, 7, 7, 9, 4]

def standardDeviation (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

theorem statistical_properties :
  (standardDeviation data1 = (1/2) * standardDeviation data2) ∧
  (median data3 = 6.5) := by
  sorry

end NUMINAMATH_CALUDE_statistical_properties_l1955_195555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1955_195597

/-- An arithmetic sequence {a_n} with a_1 + a_9 = 10 and a_2 = -1 has a common difference of 2. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 + a 9 = 10 →                     -- given condition
  a 2 = -1 →                           -- given condition
  a 2 - a 1 = 2 :=                     -- conclusion: common difference is 2
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1955_195597


namespace NUMINAMATH_CALUDE_fifth_dog_weight_l1955_195502

theorem fifth_dog_weight (w1 w2 w3 w4 y : ℝ) : 
  w1 = 25 ∧ w2 = 31 ∧ w3 = 35 ∧ w4 = 33 →
  (w1 + w2 + w3 + w4) / 4 = (w1 + w2 + w3 + w4 + y) / 5 →
  y = 31 :=
by sorry

end NUMINAMATH_CALUDE_fifth_dog_weight_l1955_195502


namespace NUMINAMATH_CALUDE_cherries_left_l1955_195537

theorem cherries_left (initial_cherries eaten_cherries : ℕ) 
  (h1 : initial_cherries = 74)
  (h2 : eaten_cherries = 72) : 
  initial_cherries - eaten_cherries = 2 := by
  sorry

end NUMINAMATH_CALUDE_cherries_left_l1955_195537


namespace NUMINAMATH_CALUDE_sin_70_deg_l1955_195549

theorem sin_70_deg (k : ℝ) (h : Real.sin (10 * π / 180) = k) : 
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_deg_l1955_195549


namespace NUMINAMATH_CALUDE_sibling_age_sum_l1955_195500

/-- Given the ages of three siblings, prove that the sum of the younger and older siblings' ages is correct. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = 10 → 
  juliet = maggie + 3 → 
  ralph = juliet + 2 → 
  maggie + ralph = 19 := by
sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l1955_195500


namespace NUMINAMATH_CALUDE_max_x0_value_l1955_195531

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h_pos : ∀ i, x i > 0)
  (h_cycle : x 0 = x 1995)
  (h_relation : ∀ i : Fin 1995, x (i + 1) + 2 / x i = 2 * x i + 1 / x (i + 1)) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1996 → ℝ, 
    (∀ i, y i > 0) ∧ 
    (y 0 = y 1995) ∧ 
    (∀ i : Fin 1995, y (i + 1) + 2 / y i = 2 * y i + 1 / y (i + 1)) ∧
    y 0 = 2^997 :=
by sorry

end NUMINAMATH_CALUDE_max_x0_value_l1955_195531


namespace NUMINAMATH_CALUDE_lecture_schedule_ways_l1955_195519

/-- The number of lecturers --/
def n : ℕ := 8

/-- The number of constrained pairs --/
def k : ℕ := 2

/-- The number of ways to schedule n lecturers with k constrained pairs --/
def schedule_ways (n : ℕ) (k : ℕ) : ℕ := n.factorial / (2^k)

/-- Theorem stating the number of ways to schedule the lectures --/
theorem lecture_schedule_ways :
  schedule_ways n k = 10080 := by
  sorry

end NUMINAMATH_CALUDE_lecture_schedule_ways_l1955_195519


namespace NUMINAMATH_CALUDE_niklaus_distance_l1955_195590

theorem niklaus_distance (lionel_miles : ℕ) (esther_yards : ℕ) (total_feet : ℕ) :
  lionel_miles = 4 →
  esther_yards = 975 →
  total_feet = 25332 →
  ∃ niklaus_feet : ℕ,
    niklaus_feet = total_feet - (lionel_miles * 5280 + esther_yards * 3) ∧
    niklaus_feet = 1287 :=
by sorry

end NUMINAMATH_CALUDE_niklaus_distance_l1955_195590


namespace NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_two_l1955_195527

theorem sqrt_a_plus_b_equals_two (a b : ℝ) (h : |a - 1| + (b - 3)^2 = 0) : 
  Real.sqrt (a + b) = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_two_l1955_195527


namespace NUMINAMATH_CALUDE_fraction_left_handed_l1955_195582

/-- Represents the ratio of red to blue participants -/
def red_blue_ratio : ℚ := 10 / 5

/-- Fraction of left-handed red participants -/
def left_handed_red : ℚ := 1 / 3

/-- Fraction of left-handed blue participants -/
def left_handed_blue : ℚ := 2 / 3

/-- Theorem: The fraction of left-handed participants is 4/9 -/
theorem fraction_left_handed :
  let total_ratio := red_blue_ratio + 1
  let left_handed_ratio := red_blue_ratio * left_handed_red + left_handed_blue
  left_handed_ratio / total_ratio = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_left_handed_l1955_195582


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1955_195511

/-- Given a sum P at simple interest rate R for 10 years, if increasing the rate by 5%
    results in Rs. 300 more interest, then P must equal 600. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 300 → P = 600 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1955_195511


namespace NUMINAMATH_CALUDE_angle_from_elevation_and_depression_l1955_195507

/-- Given an elevation angle and a depression angle observed from a point,
    calculate the angle between the two observed points and the observing point. -/
theorem angle_from_elevation_and_depression
  (elevation_angle : ℝ) (depression_angle : ℝ) :
  elevation_angle = 60 →
  depression_angle = 70 →
  elevation_angle + depression_angle = 130 :=
by sorry

end NUMINAMATH_CALUDE_angle_from_elevation_and_depression_l1955_195507


namespace NUMINAMATH_CALUDE_periodic_function_phase_shift_l1955_195526

theorem periodic_function_phase_shift (f : ℝ → ℝ) (ω φ : ℝ) :
  (ω > 0) →
  (-π / 2 < φ) →
  (φ < π / 2) →
  (∀ x : ℝ, f x = 2 * Real.sin (ω * x + φ)) →
  (∀ x : ℝ, f (x + π / 6) = f (x - π / 6)) →
  (∀ x : ℝ, f (5 * π / 18 + x) = f (5 * π / 18 - x)) →
  φ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_phase_shift_l1955_195526


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1955_195557

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 8*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1955_195557
