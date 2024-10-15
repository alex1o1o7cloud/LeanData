import Mathlib

namespace NUMINAMATH_CALUDE_festival_sunny_days_l1970_197036

def probability_exactly_two_sunny (n : ℕ) (p : ℝ) : ℝ :=
  (n.choose 2 : ℝ) * (1 - p)^2 * p^(n - 2)

theorem festival_sunny_days :
  probability_exactly_two_sunny 5 0.6 = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_festival_sunny_days_l1970_197036


namespace NUMINAMATH_CALUDE_chess_tournament_orders_l1970_197028

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 4

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := num_players - 1

/-- Calculates the total number of possible finishing orders -/
def total_possible_orders : ℕ := outcomes_per_match ^ num_matches

/-- Theorem stating that there are exactly 8 different possible finishing orders -/
theorem chess_tournament_orders : total_possible_orders = 8 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_orders_l1970_197028


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1970_197070

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1970_197070


namespace NUMINAMATH_CALUDE_correct_result_l1970_197026

theorem correct_result (x : ℤ) (h : x - 27 + 19 = 84) : x - 19 + 27 = 100 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l1970_197026


namespace NUMINAMATH_CALUDE_stating_policeman_speed_is_10_l1970_197048

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- Initial distance in meters
  thief_speed : ℝ       -- Thief's speed in km/hr
  thief_distance : ℝ    -- Distance thief runs before being caught in meters
  policeman_speed : ℝ   -- Policeman's speed in km/hr

/-- 
Theorem stating that given the specific conditions of the chase,
the policeman's speed must be 10 km/hr
-/
theorem policeman_speed_is_10 (chase : ChaseScenario) 
  (h1 : chase.initial_distance = 100)
  (h2 : chase.thief_speed = 8)
  (h3 : chase.thief_distance = 400) :
  chase.policeman_speed = 10 := by
  sorry

#check policeman_speed_is_10

end NUMINAMATH_CALUDE_stating_policeman_speed_is_10_l1970_197048


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1970_197094

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1970_197094


namespace NUMINAMATH_CALUDE_triangle_area_three_lines_l1970_197069

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_three_lines : 
  let line1 : ℝ → ℝ := λ x => 3 * x - 4
  let line2 : ℝ → ℝ := λ x => -2 * x + 16
  let y_axis : ℝ → ℝ := λ x => 0
  let intersection_x : ℝ := (16 + 4) / (3 + 2)
  let intersection_y : ℝ := line1 intersection_x
  let y_intercept1 : ℝ := line1 0
  let y_intercept2 : ℝ := line2 0
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  let area : ℝ := (1/2) * base * height
  area = 40 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_three_lines_l1970_197069


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1970_197063

theorem complex_modulus_problem (i z : ℂ) (h1 : i^2 = -1) (h2 : i * z = (1 - 2*i)^2) : 
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1970_197063


namespace NUMINAMATH_CALUDE_valid_arrangements_l1970_197061

/-- Represents the number of plates of each color -/
structure PlateCount where
  yellow : Nat
  blue : Nat
  red : Nat
  purple : Nat

/-- Calculates the total number of plates -/
def totalPlates (count : PlateCount) : Nat :=
  count.yellow + count.blue + count.red + count.purple

/-- Calculates the number of circular arrangements -/
def circularArrangements (count : PlateCount) : Nat :=
  sorry

/-- Calculates the number of circular arrangements with red plates adjacent -/
def redAdjacentArrangements (count : PlateCount) : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem valid_arrangements (count : PlateCount) 
  (h1 : count.yellow = 4)
  (h2 : count.blue = 3)
  (h3 : count.red = 2)
  (h4 : count.purple = 1) :
  circularArrangements count - redAdjacentArrangements count = 980 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l1970_197061


namespace NUMINAMATH_CALUDE_even_odd_sum_relation_l1970_197087

theorem even_odd_sum_relation (n : ℕ) : 
  (n * (n + 1) = 4970) → (n^2 = 4900) := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_relation_l1970_197087


namespace NUMINAMATH_CALUDE_r_daily_earnings_l1970_197001

/-- Represents the daily earnings of individuals p, q, and r -/
structure DailyEarnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : DailyEarnings) : Prop :=
  9 * (e.p + e.q + e.r) = 1710 ∧
  5 * (e.p + e.r) = 600 ∧
  7 * (e.q + e.r) = 910

/-- Theorem stating that given the problem conditions, r's daily earnings are 60 -/
theorem r_daily_earnings (e : DailyEarnings) :
  problem_conditions e → e.r = 60 := by
  sorry


end NUMINAMATH_CALUDE_r_daily_earnings_l1970_197001


namespace NUMINAMATH_CALUDE_tan_function_property_l1970_197058

/-- Given a function f(x) = a * tan(b * x) where a and b are positive constants,
    if f has vertical asymptotes at x = ±π/4 and passes through (π/8, 3),
    then a * b = 6 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, x ≠ π/4 ∧ x ≠ -π/4 → ∃ y, y = a * Real.tan (b * x)) →
  a * Real.tan (b * π/8) = 3 →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l1970_197058


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l1970_197088

theorem largest_prime_divisor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l1970_197088


namespace NUMINAMATH_CALUDE_leo_weight_proof_l1970_197077

/-- Leo's current weight -/
def leo_weight : ℝ := 103.6

/-- Kendra's weight -/
def kendra_weight : ℝ := 68

/-- Jake's weight -/
def jake_weight : ℝ := kendra_weight + 30

theorem leo_weight_proof :
  -- Condition 1: If Leo gains 12 pounds, he will weigh 70% more than Kendra
  (leo_weight + 12 = 1.7 * kendra_weight) ∧
  -- Condition 2: The combined weight of Leo, Kendra, and Jake is 270 pounds
  (leo_weight + kendra_weight + jake_weight = 270) ∧
  -- Condition 3: Jake weighs 30 pounds more than Kendra
  (jake_weight = kendra_weight + 30) →
  -- Conclusion: Leo's current weight is 103.6 pounds
  leo_weight = 103.6 := by
sorry

end NUMINAMATH_CALUDE_leo_weight_proof_l1970_197077


namespace NUMINAMATH_CALUDE_sleeper_probability_l1970_197092

def total_delegates : ℕ := 9
def mexico_delegates : ℕ := 2
def canada_delegates : ℕ := 3
def us_delegates : ℕ := 4
def sleepers : ℕ := 3

theorem sleeper_probability :
  let total_outcomes := Nat.choose total_delegates sleepers
  let favorable_outcomes := 
    Nat.choose mexico_delegates 2 * Nat.choose canada_delegates 1 +
    Nat.choose mexico_delegates 2 * Nat.choose us_delegates 1 +
    Nat.choose canada_delegates 2 * Nat.choose mexico_delegates 1 +
    Nat.choose canada_delegates 2 * Nat.choose us_delegates 1 +
    Nat.choose us_delegates 2 * Nat.choose mexico_delegates 1 +
    Nat.choose us_delegates 2 * Nat.choose canada_delegates 1
  (favorable_outcomes : ℚ) / total_outcomes = 55 / 84 := by
  sorry

end NUMINAMATH_CALUDE_sleeper_probability_l1970_197092


namespace NUMINAMATH_CALUDE_prob_sum_7_twice_l1970_197035

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The set of possible outcomes for a single die roll -/
def outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling a sum of 7 with two dice -/
def prob_sum_7 : ℚ := 6 / 36

/-- The probability of rolling a sum of 7 twice in a row with two dice -/
theorem prob_sum_7_twice (h : sides = 6) : prob_sum_7 * prob_sum_7 = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_7_twice_l1970_197035


namespace NUMINAMATH_CALUDE_fish_pond_estimation_l1970_197080

theorem fish_pond_estimation (x : ℕ) 
  (h1 : x > 0)  -- Ensure the pond has fish
  (h2 : 30 ≤ x) -- Ensure we can catch 30 fish initially
  : (2 : ℚ) / 30 = 30 / x → x = 450 := by
  sorry

#check fish_pond_estimation

end NUMINAMATH_CALUDE_fish_pond_estimation_l1970_197080


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1970_197089

/-- Given an ellipse and a hyperbola with the same foci, prove that the parameter n is √6 -/
theorem ellipse_hyperbola_same_foci (n : ℝ) :
  n > 0 →
  (∀ x y : ℝ, x^2 / 16 + y^2 / n^2 = 1 ↔ x^2 / n^2 - y^2 / 4 = 1) →
  n = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1970_197089


namespace NUMINAMATH_CALUDE_domain_of_f_l1970_197079

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.tan x - 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | -3*π/4 < x ∧ x < -π/2} ∪ {x | π/4 < x ∧ x < π/2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l1970_197079


namespace NUMINAMATH_CALUDE_coin_inverted_after_two_rolls_l1970_197024

/-- Represents the orientation of a coin -/
inductive CoinOrientation
  | Upright
  | Inverted

/-- Represents a single roll of the coin -/
def single_roll_rotation : ℕ := 270

/-- The total rotation after two equal rolls -/
def total_rotation : ℕ := 2 * single_roll_rotation

/-- Function to determine the final orientation after a given rotation -/
def final_orientation (initial : CoinOrientation) (rotation : ℕ) : CoinOrientation :=
  if rotation % 360 = 180 then
    match initial with
    | CoinOrientation.Upright => CoinOrientation.Inverted
    | CoinOrientation.Inverted => CoinOrientation.Upright
  else initial

/-- Theorem stating that after two equal rolls, the coin will be inverted -/
theorem coin_inverted_after_two_rolls (initial : CoinOrientation) :
  final_orientation initial total_rotation = CoinOrientation.Inverted :=
sorry

end NUMINAMATH_CALUDE_coin_inverted_after_two_rolls_l1970_197024


namespace NUMINAMATH_CALUDE_line_chart_most_appropriate_for_temperature_over_time_l1970_197037

-- Define the types of charts
inductive ChartType
| PieChart
| LineChart
| BarChart

-- Define the properties of the data
structure DataProperties where
  isTemperature : Bool
  isOverTime : Bool
  needsChangeObservation : Bool

-- Define the function to determine the most appropriate chart type
def mostAppropriateChart (props : DataProperties) : ChartType :=
  if props.isTemperature ∧ props.isOverTime ∧ props.needsChangeObservation then
    ChartType.LineChart
  else
    ChartType.BarChart  -- Default to BarChart for other cases

-- Theorem statement
theorem line_chart_most_appropriate_for_temperature_over_time 
  (props : DataProperties) 
  (h1 : props.isTemperature = true) 
  (h2 : props.isOverTime = true) 
  (h3 : props.needsChangeObservation = true) : 
  mostAppropriateChart props = ChartType.LineChart := by
  sorry


end NUMINAMATH_CALUDE_line_chart_most_appropriate_for_temperature_over_time_l1970_197037


namespace NUMINAMATH_CALUDE_five_skill_players_wait_l1970_197008

/-- Represents the water cooler scenario for a football team -/
structure WaterCooler where
  totalWater : ℕ
  numLinemen : ℕ
  numSkillPlayers : ℕ
  linemenWater : ℕ
  skillPlayerWater : ℕ

/-- Calculates the number of skill position players who must wait for water -/
def skillPlayersWaiting (wc : WaterCooler) : ℕ :=
  let linemenTotalWater := wc.numLinemen * wc.linemenWater
  let remainingWater := wc.totalWater - linemenTotalWater
  let skillPlayersServed := remainingWater / wc.skillPlayerWater
  wc.numSkillPlayers - skillPlayersServed

/-- Theorem stating that 5 skill position players must wait for water in the given scenario -/
theorem five_skill_players_wait (wc : WaterCooler) 
  (h1 : wc.totalWater = 126)
  (h2 : wc.numLinemen = 12)
  (h3 : wc.numSkillPlayers = 10)
  (h4 : wc.linemenWater = 8)
  (h5 : wc.skillPlayerWater = 6) :
  skillPlayersWaiting wc = 5 := by
  sorry

#eval skillPlayersWaiting { totalWater := 126, numLinemen := 12, numSkillPlayers := 10, linemenWater := 8, skillPlayerWater := 6 }

end NUMINAMATH_CALUDE_five_skill_players_wait_l1970_197008


namespace NUMINAMATH_CALUDE_williams_children_probability_l1970_197039

theorem williams_children_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each child being a boy (or girl)
  let total_outcomes : ℕ := 2^n  -- total number of possible gender combinations
  let balanced_outcomes : ℕ := n.choose (n/2)  -- number of combinations with equal boys and girls
  
  (total_outcomes - balanced_outcomes : ℚ) / total_outcomes = 93/128 :=
by sorry

end NUMINAMATH_CALUDE_williams_children_probability_l1970_197039


namespace NUMINAMATH_CALUDE_factory_production_l1970_197099

/-- The number of computers produced per day by a factory -/
def computers_per_day : ℕ := 1500

/-- The selling price of each computer in dollars -/
def price_per_computer : ℕ := 150

/-- The revenue from one week's production in dollars -/
def weekly_revenue : ℕ := 1575000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem factory_production :
  computers_per_day * price_per_computer * days_in_week = weekly_revenue :=
by sorry

end NUMINAMATH_CALUDE_factory_production_l1970_197099


namespace NUMINAMATH_CALUDE_sentences_started_today_l1970_197076

/-- Calculates the number of sentences Janice started with today given her typing speed and work schedule. -/
theorem sentences_started_today (
  typing_speed : ℕ)  -- Sentences typed per minute
  (initial_typing_time : ℕ)  -- Minutes typed before break
  (extra_typing_time : ℕ)  -- Additional minutes typed after break
  (erased_sentences : ℕ)  -- Number of sentences erased due to errors
  (final_typing_time : ℕ)  -- Minutes typed after meeting
  (total_sentences : ℕ)  -- Total sentences in the paper by end of day
  (h1 : typing_speed = 6)
  (h2 : initial_typing_time = 20)
  (h3 : extra_typing_time = 15)
  (h4 : erased_sentences = 40)
  (h5 : final_typing_time = 18)
  (h6 : total_sentences = 536)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_sentences_started_today_l1970_197076


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l1970_197034

/-- The ratio of the volume of an octahedron formed by the centers of the faces of a cube
    to the volume of the cube itself, given that the cube has side length 2. -/
theorem octahedron_cube_volume_ratio :
  let cube_side_length : ℝ := 2
  let cube_volume : ℝ := cube_side_length ^ 3
  let octahedron_volume : ℝ := 4 / 3
  octahedron_volume / cube_volume = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l1970_197034


namespace NUMINAMATH_CALUDE_identity_holds_iff_k_equals_negative_one_l1970_197044

theorem identity_holds_iff_k_equals_negative_one :
  ∀ k : ℝ, (∀ a b c : ℝ, (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + k * a * b * c) ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_identity_holds_iff_k_equals_negative_one_l1970_197044


namespace NUMINAMATH_CALUDE_salary_restoration_l1970_197031

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := original_salary * (1 - 0.2)
  reduced_salary * (1 + 0.25) = original_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_restoration_l1970_197031


namespace NUMINAMATH_CALUDE_compute_fraction_power_l1970_197072

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l1970_197072


namespace NUMINAMATH_CALUDE_vector_c_solution_l1970_197091

theorem vector_c_solution (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (2, -3) → 
  (∃ k : ℝ, c + a = k • b) →
  c • (a + b) = 0 →
  c = (-7/9, -7/3) := by sorry

end NUMINAMATH_CALUDE_vector_c_solution_l1970_197091


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l1970_197067

theorem modulo_eleven_residue : (312 + 6 * 47 + 8 * 154 + 5 * 22) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l1970_197067


namespace NUMINAMATH_CALUDE_exists_1990_edge_polyhedron_no_triangles_l1970_197073

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Define the necessary properties of a convex polyhedron
  isConvex : Bool
  edges : Nat
  faces : List Nat

/-- Checks if a polyhedron has no triangular faces. -/
def hasNoTriangularFaces (p : ConvexPolyhedron) : Bool :=
  p.faces.all (· > 3)

/-- Theorem stating the existence of a convex polyhedron with 1990 edges and no triangular faces. -/
theorem exists_1990_edge_polyhedron_no_triangles : 
  ∃ p : ConvexPolyhedron, p.isConvex ∧ p.edges = 1990 ∧ hasNoTriangularFaces p :=
sorry

end NUMINAMATH_CALUDE_exists_1990_edge_polyhedron_no_triangles_l1970_197073


namespace NUMINAMATH_CALUDE_equation_solution_l1970_197059

theorem equation_solution : 
  ∃! x : ℚ, (x - 30) / 3 = (3 * x + 10) / 8 - 2 ∧ x = -222 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1970_197059


namespace NUMINAMATH_CALUDE_intersection_product_is_three_l1970_197097

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + 2*x + y^2 + 4*y + 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + 4*x + y^2 + 4*y + 7 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (-1.5, -2)

-- Theorem statement
theorem intersection_product_is_three :
  let (x, y) := intersection_point
  circle1 x y ∧ circle2 x y ∧ x * y = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_product_is_three_l1970_197097


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l1970_197010

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  (z.re = z.im) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l1970_197010


namespace NUMINAMATH_CALUDE_distance_between_points_l1970_197021

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3.5, -4.5)
  let p2 : ℝ × ℝ := (3.5, 2.5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1970_197021


namespace NUMINAMATH_CALUDE_percentage_problem_l1970_197009

theorem percentage_problem (N P : ℝ) : 
  N = 50 → 
  N = (P / 100) * N + 40 → 
  P = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1970_197009


namespace NUMINAMATH_CALUDE_problem_solution_l1970_197071

theorem problem_solution :
  -- Part 1
  let n : ℕ := Finset.sum (Finset.range 16) (λ i => 2 * i + 1)
  let m : ℕ := Finset.sum (Finset.range 16) (λ i => 2 * (i + 1))
  m - n = 16 ∧
  -- Part 2
  let trapezium_area (a b h : ℝ) := (a + b) * h / 2
  trapezium_area 4 16 16 = 160 ∧
  -- Part 3
  let isosceles_triangle (side angle : ℝ) := side > 0 ∧ 0 < angle ∧ angle < π
  ∀ side angle, isosceles_triangle side angle → angle = π / 3 → 3 = 3 ∧
  -- Part 4
  let f (x : ℝ) := 3 * x^(2/3) - 8 * x^(1/3) + 4
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ x = 8/27 ∧ ∀ y, y > 0 → f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1970_197071


namespace NUMINAMATH_CALUDE_average_sales_is_16_l1970_197093

def january_sales : ℕ := 15
def february_sales : ℕ := 16
def march_sales : ℕ := 17

def total_months : ℕ := 3

def average_sales : ℚ := (january_sales + february_sales + march_sales : ℚ) / total_months

theorem average_sales_is_16 : average_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_is_16_l1970_197093


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_l1970_197014

theorem negation_of_exists_greater (p : Prop) :
  (¬ ∃ (n : ℕ), 2^n > 1000) ↔ (∀ (n : ℕ), 2^n ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_l1970_197014


namespace NUMINAMATH_CALUDE_intersection_point_property_l1970_197015

theorem intersection_point_property (x₀ : ℝ) (h1 : x₀ ≠ 0) (h2 : Real.tan x₀ = -x₀) :
  (x₀^2 + 1) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l1970_197015


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1970_197020

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < -2}
def N : Set ℝ := {x | x^2 + 5*x + 6 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -3 < x ∧ x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1970_197020


namespace NUMINAMATH_CALUDE_truck_speed_l1970_197053

/-- Proves that a truck traveling 600 meters in 20 seconds has a speed of 108 kilometers per hour. -/
theorem truck_speed (distance : ℝ) (time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ) : 
  distance = 600 →
  time = 20 →
  speed_ms = distance / time →
  speed_kmh = speed_ms * 3.6 →
  speed_kmh = 108 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_l1970_197053


namespace NUMINAMATH_CALUDE_unique_solution_abc_l1970_197078

theorem unique_solution_abc : ∃! (a b c : ℝ),
  a > 2 ∧ b > 2 ∧ c > 2 ∧
  ((a + 1)^2) / (b + c - 1) + ((b + 3)^2) / (c + a - 3) + ((c + 5)^2) / (a + b - 5) = 27 ∧
  a = 9 ∧ b = 7 ∧ c = 2 := by
  sorry

#check unique_solution_abc

end NUMINAMATH_CALUDE_unique_solution_abc_l1970_197078


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1970_197082

/-- The coefficient of x^3 in the expansion of (1-2x^2)(1+x)^4 is -4 -/
theorem coefficient_x_cubed_in_expansion : ∃ (p : Polynomial ℤ), 
  p = (1 - 2 * X^2) * (1 + X)^4 ∧ p.coeff 3 = -4 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1970_197082


namespace NUMINAMATH_CALUDE_tournament_rankings_l1970_197062

/-- Represents a team in the volleyball tournament -/
inductive Team : Type
| E | F | G | H | I | J

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  saturday_matches : Vector Match 3
  no_ties : Bool

/-- Calculates the number of possible ranking sequences -/
def possible_rankings (t : Tournament) : Nat :=
  6 * 6

/-- Theorem: The number of possible six-team ranking sequences is 36 -/
theorem tournament_rankings (t : Tournament) :
  t.no_ties → possible_rankings t = 36 := by
  sorry

end NUMINAMATH_CALUDE_tournament_rankings_l1970_197062


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1970_197086

/-- Given 12 numbers with an initial average of 22, where three numbers were incorrectly read
    (52 as 32, 47 as 27, and 68 as 45), the correct average is 27.25. -/
theorem correct_average_after_error_correction (total_numbers : ℕ) (initial_average : ℚ)
  (incorrect_num1 incorrect_num2 incorrect_num3 : ℚ)
  (correct_num1 correct_num2 correct_num3 : ℚ) :
  total_numbers = 12 →
  initial_average = 22 →
  incorrect_num1 = 32 →
  incorrect_num2 = 27 →
  incorrect_num3 = 45 →
  correct_num1 = 52 →
  correct_num2 = 47 →
  correct_num3 = 68 →
  ((total_numbers : ℚ) * initial_average - incorrect_num1 - incorrect_num2 - incorrect_num3 +
    correct_num1 + correct_num2 + correct_num3) / total_numbers = 27.25 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1970_197086


namespace NUMINAMATH_CALUDE_money_split_ratio_l1970_197004

/-- Given two people splitting money in a ratio of 2:3, where the smaller share is $50,
    prove that the total amount shared is $125. -/
theorem money_split_ratio (parker_share richie_share total : ℕ) : 
  parker_share = 50 →
  parker_share + richie_share = total →
  2 * richie_share = 3 * parker_share →
  total = 125 := by
sorry

end NUMINAMATH_CALUDE_money_split_ratio_l1970_197004


namespace NUMINAMATH_CALUDE_basketball_only_count_l1970_197025

theorem basketball_only_count (total students_basketball students_table_tennis students_neither : ℕ) 
  (h1 : total = 30)
  (h2 : students_basketball = 15)
  (h3 : students_table_tennis = 10)
  (h4 : students_neither = 8)
  (h5 : total = students_basketball + students_table_tennis - students_both + students_neither)
  (students_both : ℕ) :
  students_basketball - students_both = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_only_count_l1970_197025


namespace NUMINAMATH_CALUDE_sum_integers_11_to_24_l1970_197052

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_integers_11_to_24 : sum_integers 11 24 = 245 := by sorry

end NUMINAMATH_CALUDE_sum_integers_11_to_24_l1970_197052


namespace NUMINAMATH_CALUDE_abs_sum_diff_inequality_l1970_197049

theorem abs_sum_diff_inequality (x y : ℝ) :
  (abs x < 1 ∧ abs y < 1) ↔ abs (x + y) + abs (x - y) < 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_diff_inequality_l1970_197049


namespace NUMINAMATH_CALUDE_committee_formation_count_l1970_197047

/-- The number of ways to choose a committee under given conditions -/
def committee_formations (total_boys : ℕ) (total_girls : ℕ) (committee_size : ℕ) 
  (boys_with_event_planning : ℕ) (girls_with_leadership : ℕ) : ℕ :=
  let boys_to_choose := committee_size / 2
  let girls_to_choose := committee_size / 2
  let remaining_boys := total_boys - boys_with_event_planning
  let remaining_girls := total_girls - girls_with_leadership
  (Nat.choose remaining_boys (boys_to_choose - 1)) * 
  (Nat.choose remaining_girls (girls_to_choose - 1))

/-- Theorem stating the number of ways to form the committee -/
theorem committee_formation_count :
  committee_formations 8 6 8 1 1 = 350 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1970_197047


namespace NUMINAMATH_CALUDE_kids_to_adult_meals_ratio_l1970_197064

theorem kids_to_adult_meals_ratio 
  (kids_meals : ℕ) 
  (total_meals : ℕ) 
  (h1 : kids_meals = 8) 
  (h2 : total_meals = 12) : 
  (kids_meals : ℚ) / ((total_meals - kids_meals) : ℚ) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_kids_to_adult_meals_ratio_l1970_197064


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1970_197046

/-- The roots of the quadratic equation x^2 - 7x + 12 = 0 -/
def roots : Set ℝ := {x : ℝ | x^2 - 7*x + 12 = 0}

/-- An isosceles triangle with two sides from the roots set -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 ∈ roots) ∨ (side1 = side3 ∧ side2 ∈ roots) ∨ (side2 = side3 ∧ side1 ∈ roots)
  sides_from_roots : {side1, side2, side3} ∩ roots = {side1, side2} ∨ {side1, side2, side3} ∩ roots = {side1, side3} ∨ {side1, side2, side3} ∩ roots = {side2, side3}

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem: The perimeter of the isosceles triangle is either 10 or 11 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1970_197046


namespace NUMINAMATH_CALUDE_relation_between_exponents_l1970_197041

-- Define variables
variable (a d c e : ℝ)
variable (u v w r : ℝ)

-- State the theorem
theorem relation_between_exponents 
  (h1 : a^u = d^r) 
  (h2 : d^r = c)
  (h3 : d^v = a^w)
  (h4 : a^w = e) :
  r * w = v * u := by
  sorry

end NUMINAMATH_CALUDE_relation_between_exponents_l1970_197041


namespace NUMINAMATH_CALUDE_b_completion_time_l1970_197050

-- Define the work rates and time worked by A
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def a_time_worked : ℕ := 10

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem statement
theorem b_completion_time :
  let work_done_by_a : ℚ := a_rate * a_time_worked
  let remaining_work : ℚ := total_work - work_done_by_a
  remaining_work / b_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l1970_197050


namespace NUMINAMATH_CALUDE_water_amount_l1970_197016

/-- The number of boxes -/
def num_boxes : ℕ := 10

/-- The number of bottles in each box -/
def bottles_per_box : ℕ := 50

/-- The capacity of each bottle in liters -/
def bottle_capacity : ℚ := 12

/-- The fraction of the bottle's capacity that is filled -/
def fill_fraction : ℚ := 3/4

/-- The total amount of water in liters contained in all boxes -/
def total_water : ℚ := num_boxes * bottles_per_box * bottle_capacity * fill_fraction

theorem water_amount : total_water = 4500 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_l1970_197016


namespace NUMINAMATH_CALUDE_real_number_classification_l1970_197003

theorem real_number_classification :
  Set.univ = {x : ℝ | x > 0} ∪ {x : ℝ | x < 0} ∪ {(0 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_real_number_classification_l1970_197003


namespace NUMINAMATH_CALUDE_actual_tax_raise_expectation_l1970_197065

-- Define the population
def Population := ℝ

-- Define the fraction of liars and economists
def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 0.9

-- Define the affirmative answer percentages
def taxes_raised : ℝ := 0.4
def money_supply_increased : ℝ := 0.3
def bonds_issued : ℝ := 0.5
def reserves_spent : ℝ := 0

-- Define the theorem
theorem actual_tax_raise_expectation :
  let total_affirmative := taxes_raised + money_supply_increased + bonds_issued + reserves_spent
  fraction_liars * 3 + fraction_economists = total_affirmative →
  taxes_raised - fraction_liars = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_actual_tax_raise_expectation_l1970_197065


namespace NUMINAMATH_CALUDE_orange_calculation_l1970_197098

/-- Calculates the total number and weight of oranges given the number of children,
    oranges per child, and average weight per orange. -/
theorem orange_calculation (num_children : ℕ) (oranges_per_child : ℕ) (avg_weight : ℚ) :
  num_children = 4 →
  oranges_per_child = 3 →
  avg_weight = 3/10 →
  (num_children * oranges_per_child = 12 ∧
   (num_children * oranges_per_child : ℚ) * avg_weight = 18/5) :=
by sorry

end NUMINAMATH_CALUDE_orange_calculation_l1970_197098


namespace NUMINAMATH_CALUDE_impossible_to_guarantee_same_state_l1970_197055

/-- Represents the state of a usamon (has an electron or not) -/
inductive UsamonState
| HasElectron
| NoElectron

/-- Represents a usamon with its current state -/
structure Usamon :=
  (state : UsamonState)

/-- Represents the action of connecting a diode between two usamons -/
def connectDiode (a b : Usamon) : Usamon × Usamon :=
  match a.state, b.state with
  | UsamonState.HasElectron, UsamonState.NoElectron => 
      ({ state := UsamonState.NoElectron }, { state := UsamonState.HasElectron })
  | _, _ => (a, b)

/-- The main theorem stating that it's impossible to guarantee two usamons are in the same state -/
theorem impossible_to_guarantee_same_state (usamons : Fin 2015 → Usamon) :
  ∀ (sequence : List (Fin 2015 × Fin 2015)),
  ¬∃ (i j : Fin 2015), i ≠ j ∧ (usamons i).state = (usamons j).state := by
  sorry


end NUMINAMATH_CALUDE_impossible_to_guarantee_same_state_l1970_197055


namespace NUMINAMATH_CALUDE_gcd_bound_from_lcm_l1970_197090

theorem gcd_bound_from_lcm (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 → 
  Nat.gcd a b < 100 := by
sorry

end NUMINAMATH_CALUDE_gcd_bound_from_lcm_l1970_197090


namespace NUMINAMATH_CALUDE_unique_prime_six_digit_number_l1970_197083

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B : ℕ) : ℕ :=
  303100 + B

theorem unique_prime_six_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (six_digit_number B) ∧ six_digit_number B = 303101 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_six_digit_number_l1970_197083


namespace NUMINAMATH_CALUDE_min_value_theorem_l1970_197051

theorem min_value_theorem (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  (x / (x^2 + 1)) + (y / (y^2 + 1)) + (z / (z^2 + 1)) ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1970_197051


namespace NUMINAMATH_CALUDE_betty_order_hair_color_cost_l1970_197002

/-- Given Betty's order details, prove the cost of each hair color. -/
theorem betty_order_hair_color_cost 
  (total_items : ℕ) 
  (slipper_count : ℕ) 
  (slipper_cost : ℚ) 
  (lipstick_count : ℕ) 
  (lipstick_cost : ℚ) 
  (hair_color_count : ℕ) 
  (total_paid : ℚ) 
  (h_total_items : total_items = 18) 
  (h_slipper_count : slipper_count = 6) 
  (h_slipper_cost : slipper_cost = 5/2) 
  (h_lipstick_count : lipstick_count = 4) 
  (h_lipstick_cost : lipstick_cost = 5/4) 
  (h_hair_color_count : hair_color_count = 8) 
  (h_total_paid : total_paid = 44) 
  (h_item_sum : total_items = slipper_count + lipstick_count + hair_color_count) : 
  (total_paid - (slipper_count * slipper_cost + lipstick_count * lipstick_cost)) / hair_color_count = 3 := by
  sorry


end NUMINAMATH_CALUDE_betty_order_hair_color_cost_l1970_197002


namespace NUMINAMATH_CALUDE_survey_respondents_count_l1970_197007

theorem survey_respondents_count :
  ∀ (x y : ℕ),
    x = 200 →
    4 * y = x →
    x + y = 250 :=
by sorry

end NUMINAMATH_CALUDE_survey_respondents_count_l1970_197007


namespace NUMINAMATH_CALUDE_min_triangles_for_100gon_l1970_197011

/-- A convex polygon with 100 sides -/
def ConvexPolygon100 : Type := Unit

/-- The number of triangles needed to represent a convex 100-gon as their intersection -/
def num_triangles (p : ConvexPolygon100) : ℕ := sorry

/-- The smallest number of triangles needed to represent any convex 100-gon as their intersection -/
def min_num_triangles : ℕ := sorry

theorem min_triangles_for_100gon :
  min_num_triangles = 50 := by sorry

end NUMINAMATH_CALUDE_min_triangles_for_100gon_l1970_197011


namespace NUMINAMATH_CALUDE_product_of_roots_l1970_197032

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 4 / x = y + 4 / y) : x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1970_197032


namespace NUMINAMATH_CALUDE_stream_speed_l1970_197019

/-- Proves that given a boat's travel times and distances upstream and downstream, the speed of the stream is 1 km/h -/
theorem stream_speed (b : ℝ) (s : ℝ) 
  (h1 : (b + s) * 10 = 100) 
  (h2 : (b - s) * 25 = 200) : 
  s = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1970_197019


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l1970_197084

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_minimum (a b : ℝ) :
  (∀ x : ℝ, f a b x ≥ f a b (-1)) ∧ (f a b (-1) = 0) →
  ∀ x : ℝ, f a b x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l1970_197084


namespace NUMINAMATH_CALUDE_english_marks_proof_l1970_197038

def average (numbers : List ℕ) : ℚ :=
  (numbers.sum : ℚ) / numbers.length

theorem english_marks_proof (marks : List ℕ) (h1 : marks.length = 5) 
  (h2 : average marks = 76) 
  (h3 : 69 ∈ marks) (h4 : 92 ∈ marks) (h5 : 64 ∈ marks) (h6 : 82 ∈ marks) : 
  73 ∈ marks := by
  sorry

#check english_marks_proof

end NUMINAMATH_CALUDE_english_marks_proof_l1970_197038


namespace NUMINAMATH_CALUDE_ant_problem_l1970_197013

/-- Represents the number of ants for each species on Day 0 -/
structure AntCounts where
  a : ℕ  -- Species A
  b : ℕ  -- Species B
  c : ℕ  -- Species C

/-- Calculates the total number of ants on a given day -/
def totalAnts (day : ℕ) (counts : AntCounts) : ℕ :=
  2^day * counts.a + 3^day * counts.b + 4^day * counts.c

theorem ant_problem (counts : AntCounts) :
  totalAnts 0 counts = 50 →
  totalAnts 4 counts = 6561 →
  4^4 * counts.c = 5632 := by
  sorry

end NUMINAMATH_CALUDE_ant_problem_l1970_197013


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1970_197075

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1970_197075


namespace NUMINAMATH_CALUDE_total_time_circling_island_l1970_197081

/-- The time in minutes to navigate around the island once. -/
def time_per_round : ℕ := 30

/-- The number of rounds completed on Saturday. -/
def saturday_rounds : ℕ := 11

/-- The number of rounds completed on Sunday. -/
def sunday_rounds : ℕ := 15

/-- The total time spent circling the island over the weekend. -/
theorem total_time_circling_island : 
  (saturday_rounds + sunday_rounds) * time_per_round = 780 := by sorry

end NUMINAMATH_CALUDE_total_time_circling_island_l1970_197081


namespace NUMINAMATH_CALUDE_manufacturer_cost_effectiveness_l1970_197056

/-- Represents the cost calculation for manufacturers A and B -/
def cost_calculation (x : ℝ) : Prop :=
  let desk_price : ℝ := 200
  let chair_price : ℝ := 50
  let desk_quantity : ℝ := 60
  let discount_rate : ℝ := 0.9
  let cost_A : ℝ := desk_price * desk_quantity + chair_price * (x - desk_quantity)
  let cost_B : ℝ := (desk_price * desk_quantity + chair_price * x) * discount_rate
  (x ≥ desk_quantity) ∧
  (x < 360 → cost_A < cost_B) ∧
  (x > 360 → cost_B < cost_A)

/-- Theorem stating the conditions for cost-effectiveness of manufacturers A and B -/
theorem manufacturer_cost_effectiveness :
  ∀ x : ℝ, cost_calculation x :=
sorry

end NUMINAMATH_CALUDE_manufacturer_cost_effectiveness_l1970_197056


namespace NUMINAMATH_CALUDE_least_common_multiple_of_denominators_l1970_197023

theorem least_common_multiple_of_denominators : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_denominators_l1970_197023


namespace NUMINAMATH_CALUDE_problem_solution_l1970_197040

theorem problem_solution (m : ℤ) (a : ℝ) : 
  ((-2 : ℝ)^(2*m) = a^(3-m)) → (m = 1) → (a = 2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1970_197040


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l1970_197029

/-- The area of the union of a rectangle and a circle with specified dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 12
  let rectangle_height : ℝ := 15
  let circle_radius : ℝ := 15
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  let union_area : ℝ := rectangle_area + circle_area - overlap_area
  union_area = 180 + 168.75 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l1970_197029


namespace NUMINAMATH_CALUDE_rudolph_trip_signs_per_mile_l1970_197096

/-- Rudolph's car trip across town -/
def rudolph_trip (miles_base : ℕ) (miles_extra : ℕ) (signs_base : ℕ) (signs_less : ℕ) : ℚ :=
  let total_miles : ℕ := miles_base + miles_extra
  let total_signs : ℕ := signs_base - signs_less
  (total_signs : ℚ) / (total_miles : ℚ)

/-- Theorem stating the number of stop signs per mile Rudolph encountered -/
theorem rudolph_trip_signs_per_mile :
  rudolph_trip 5 2 17 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rudolph_trip_signs_per_mile_l1970_197096


namespace NUMINAMATH_CALUDE_volunteers_selection_ways_l1970_197000

/-- The number of ways to select volunteers for community service --/
def select_volunteers (n : ℕ) : ℕ :=
  let both_days := n  -- Select 1 person for both days
  let saturday := n - 1  -- Select 1 person for Saturday from remaining n-1
  let sunday := n - 2  -- Select 1 person for Sunday from remaining n-2
  both_days * saturday * sunday

/-- Theorem: The number of ways to select exactly one person to serve for both days
    out of 5 volunteers, with 2 people selected each day, is equal to 60 --/
theorem volunteers_selection_ways :
  select_volunteers 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_selection_ways_l1970_197000


namespace NUMINAMATH_CALUDE_complex_sixth_power_real_count_l1970_197074

theorem complex_sixth_power_real_count : 
  ∃! (n : ℤ), (Complex.I + n : ℂ)^6 ∈ Set.range (Complex.ofReal : ℝ → ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_power_real_count_l1970_197074


namespace NUMINAMATH_CALUDE_total_amount_paid_l1970_197018

/-- Calculates the discounted price for a fruit given its weight, price per kg, and discount percentage. -/
def discountedPrice (weight : Float) (pricePerKg : Float) (discountPercent : Float) : Float :=
  weight * pricePerKg * (1 - discountPercent / 100)

/-- Represents the shopping trip and calculates the total amount paid. -/
def shoppingTrip : Float :=
  discountedPrice 8 70 10 +    -- Grapes
  discountedPrice 11 55 0 +    -- Mangoes
  discountedPrice 5 45 20 +    -- Oranges
  discountedPrice 3 90 5 +     -- Apples
  discountedPrice 4.5 120 0    -- Cherries

/-- Theorem stating that the total amount paid is $2085.50 -/
theorem total_amount_paid : shoppingTrip = 2085.50 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1970_197018


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1970_197054

/-- Proves that the length of a rectangular plot is 70 meters given the specified conditions. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 40 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1970_197054


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1970_197006

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  side1 : ℝ
  side2 : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.base1 + t.base2 + t.side1 + t.side2

/-- Theorem: The perimeter of the specific trapezoid is 42 -/
theorem trapezoid_perimeter : 
  let t : Trapezoid := { base1 := 10, base2 := 14, side1 := 9, side2 := 9 }
  perimeter t = 42 := by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1970_197006


namespace NUMINAMATH_CALUDE_unique_number_base_conversion_l1970_197068

def is_valid_base_8_digit (d : ℕ) : Prop := d < 8
def is_valid_base_6_digit (d : ℕ) : Prop := d < 6

def base_8_to_decimal (a b : ℕ) : ℕ := 8 * a + b
def base_6_to_decimal (b a : ℕ) : ℕ := 6 * b + a

theorem unique_number_base_conversion : ∃! n : ℕ, 
  ∃ (a b : ℕ), 
    is_valid_base_8_digit a ∧
    is_valid_base_6_digit b ∧
    n = base_8_to_decimal a b ∧
    n = base_6_to_decimal b a ∧
    n = 45 := by sorry

end NUMINAMATH_CALUDE_unique_number_base_conversion_l1970_197068


namespace NUMINAMATH_CALUDE_factors_of_product_l1970_197066

/-- A natural number with exactly three factors is the square of a prime. -/
def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^2

/-- The number of factors of n^k where n is a prime square. -/
def num_factors_prime_square_pow (n k : ℕ) : ℕ :=
  2 * k + 1

/-- The main theorem -/
theorem factors_of_product (a b c : ℕ) 
  (ha : is_prime_square a) 
  (hb : is_prime_square b)
  (hc : is_prime_square c)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (num_factors_prime_square_pow a 3) * 
  (num_factors_prime_square_pow b 4) * 
  (num_factors_prime_square_pow c 5) = 693 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_product_l1970_197066


namespace NUMINAMATH_CALUDE_therapy_pricing_theorem_l1970_197085

/-- Represents the pricing structure of a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  firstHourPremium : ℕ
  fiveHourTotal : ℕ

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating the conditions and the result to be proved. -/
theorem therapy_pricing_theorem (pricing : TherapyPricing) 
  (h1 : pricing.firstHourCharge = pricing.additionalHourCharge + pricing.firstHourPremium)
  (h2 : pricing.firstHourPremium = 35)
  (h3 : pricing.fiveHourTotal = 350)
  (h4 : totalCharge pricing 5 = pricing.fiveHourTotal) :
  totalCharge pricing 2 = 161 := by
  sorry


end NUMINAMATH_CALUDE_therapy_pricing_theorem_l1970_197085


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1970_197033

/-- Given a hyperbola C with equation x²/m - y² = 1 and one focus at (2, 0),
    prove that its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let focus : ℝ × ℝ := (2, 0)
  focus ∈ {f | ∃ (x y : ℝ), (x, y) ∈ C ∧ (x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2 + (y - f.2)^2} →
  let e := Real.sqrt ((2 : ℝ)^2 / m)
  e = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1970_197033


namespace NUMINAMATH_CALUDE_complex_magnitude_l1970_197022

theorem complex_magnitude (i : ℂ) (h : i * i = -1) :
  Complex.abs (i + 2 * i^2 + 3 * i^3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1970_197022


namespace NUMINAMATH_CALUDE_subtraction_equivalence_l1970_197012

theorem subtraction_equivalence : 596 - 130 - 270 = 596 - (130 + 270) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equivalence_l1970_197012


namespace NUMINAMATH_CALUDE_min_exponent_sum_l1970_197017

theorem min_exponent_sum (h : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h)
  (h_div_216 : 216 ∣ h)
  (h_factorization : h = 2^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ)) :
  a + b + c ≥ 8 ∧ ∃ (h' : ℕ+) (a' b' c' : ℕ+), 
    225 ∣ h' ∧ 216 ∣ h' ∧ h' = 2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ) ∧ a' + b' + c' = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_exponent_sum_l1970_197017


namespace NUMINAMATH_CALUDE_sum_of_divisors_77_and_not_perfect_l1970_197057

def sum_of_divisors (n : ℕ) : ℕ := sorry

def is_perfect_number (n : ℕ) : Prop :=
  sum_of_divisors n = 2 * n

theorem sum_of_divisors_77_and_not_perfect :
  sum_of_divisors 77 = 96 ∧ ¬(is_perfect_number 77) := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_77_and_not_perfect_l1970_197057


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1970_197043

theorem quadratic_factorization_sum (h b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (h * x + b) * (c * x + d)) → 
  |h| + |b| + |c| + |d| = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1970_197043


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l1970_197027

theorem simplify_nested_expression (x : ℝ) : 2 - (3 - (4 - (5 - (2*x - 3)))) = -5 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l1970_197027


namespace NUMINAMATH_CALUDE_scientific_notation_21600_l1970_197095

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_21600 :
  toScientificNotation 21600 = ScientificNotation.mk 2.16 4 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_21600_l1970_197095


namespace NUMINAMATH_CALUDE_expression_simplification_l1970_197045

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 2) :
  (1 / (a - 2) - 2 / (a^2 - 4)) / ((a^2 - 2*a) / (a^2 - 4)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1970_197045


namespace NUMINAMATH_CALUDE_transform_1220_to_2012_not_transform_1220_to_2021_l1970_197060

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  digits : Fin 4 → Fin 10

/-- Defines the allowed transformations on a 4-digit number -/
def transform (n : FourDigitNumber) (i : Fin 3) : Option FourDigitNumber :=
  if n.digits i ≠ 0 ∧ n.digits (i + 1) ≠ 0 then
    some ⟨fun j => if j = i ∨ j = i + 1 then n.digits j - 1 else n.digits j⟩
  else if n.digits i ≠ 9 ∧ n.digits (i + 1) ≠ 9 then
    some ⟨fun j => if j = i ∨ j = i + 1 then n.digits j + 1 else n.digits j⟩
  else
    none

/-- Defines the reachability of one number from another through transformations -/
def reachable (start finish : FourDigitNumber) : Prop :=
  ∃ (seq : List (Fin 3)), finish = seq.foldl (fun n i => (transform n i).getD n) start

/-- The initial number 1220 -/
def initial : FourDigitNumber := ⟨fun i => match i with | 0 => 1 | 1 => 2 | 2 => 2 | 3 => 0⟩

/-- The target number 2012 -/
def target1 : FourDigitNumber := ⟨fun i => match i with | 0 => 2 | 1 => 0 | 2 => 1 | 3 => 2⟩

/-- The target number 2021 -/
def target2 : FourDigitNumber := ⟨fun i => match i with | 0 => 2 | 1 => 0 | 2 => 2 | 3 => 1⟩

theorem transform_1220_to_2012 : reachable initial target1 := by sorry

theorem not_transform_1220_to_2021 : ¬reachable initial target2 := by sorry

end NUMINAMATH_CALUDE_transform_1220_to_2012_not_transform_1220_to_2021_l1970_197060


namespace NUMINAMATH_CALUDE_correct_proposition_l1970_197042

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 1 → x > 2

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1

-- Theorem to prove
theorem correct_proposition : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l1970_197042


namespace NUMINAMATH_CALUDE_journey_time_ratio_l1970_197005

theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 180)
  (h2 : original_time = 6)
  (h3 : new_speed = 20)
  : (distance / new_speed) / original_time = 3 / 2 := by
  sorry

#check journey_time_ratio

end NUMINAMATH_CALUDE_journey_time_ratio_l1970_197005


namespace NUMINAMATH_CALUDE_complement_of_A_l1970_197030

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0 ∨ x ≥ 1}

theorem complement_of_A : Set.compl A = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1970_197030
