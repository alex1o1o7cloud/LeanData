import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_ones_count_l3592_359231

/-- Recursive function to calculate the number of n-digit numbers with no two consecutive 1's -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => a (n + 1) + a n

/-- The number of 12-digit positive integers with digits 1 or 2 -/
def total_12_digit : ℕ := 2^12

/-- The main theorem stating the number of 12-digit integers with at least one pair of consecutive 1's -/
theorem consecutive_ones_count : 
  total_12_digit - a 12 = 3719 := by sorry

end NUMINAMATH_CALUDE_consecutive_ones_count_l3592_359231


namespace NUMINAMATH_CALUDE_xy_value_l3592_359276

theorem xy_value (x y : ℝ) 
  (h1 : x + y = 2) 
  (h2 : x^2 * y^3 + y^2 * x^3 = 32) : 
  x * y = -8 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3592_359276


namespace NUMINAMATH_CALUDE_integral_proof_l3592_359204

open Real

theorem integral_proof (x : ℝ) (h : x ≠ -2 ∧ x ≠ 8) : 
  deriv (fun x => (1/10) * log (abs ((x - 8) / (x + 2)))) x = 1 / (x^2 - 6*x - 16) :=
sorry

end NUMINAMATH_CALUDE_integral_proof_l3592_359204


namespace NUMINAMATH_CALUDE_spacing_change_at_20th_post_l3592_359210

/-- Represents the fence with its posts and spacings -/
structure Fence where
  initialSpacing : ℝ
  changedSpacing : ℝ
  changePost : ℕ

/-- The fence satisfies the given conditions -/
def satisfiesConditions (f : Fence) : Prop :=
  f.initialSpacing > f.changedSpacing ∧
  f.initialSpacing * 15 = 48 ∧
  f.changedSpacing * (28 - f.changePost) + f.initialSpacing * (f.changePost - 16) = 36 ∧
  f.changePost > 16 ∧ f.changePost ≤ 28

/-- The theorem stating that the 20th post is where the spacing changes -/
theorem spacing_change_at_20th_post (f : Fence) (h : satisfiesConditions f) : f.changePost = 20 := by
  sorry

end NUMINAMATH_CALUDE_spacing_change_at_20th_post_l3592_359210


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3592_359294

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (p q : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ p * q = k

theorem inverse_proportion_problem (p q : ℝ) :
  InverselyProportional p q →
  p + q = 40 →
  p - q = 10 →
  p = 7 →
  q = 375 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3592_359294


namespace NUMINAMATH_CALUDE_kellys_vacation_duration_l3592_359247

/-- Kelly's vacation duration calculation -/
theorem kellys_vacation_duration :
  let travel_days : ℕ := 1 + 1 + 2 + 2  -- Sum of all travel days
  let stay_days : ℕ := 5 + 5 + 5        -- Sum of all stay days
  let total_days : ℕ := travel_days + stay_days
  let days_per_week : ℕ := 7
  (total_days / days_per_week : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_kellys_vacation_duration_l3592_359247


namespace NUMINAMATH_CALUDE_high_school_ratio_problem_l3592_359237

theorem high_school_ratio_problem (initial_boys initial_girls : ℕ) 
  (boys_left girls_left : ℕ) (final_boys final_girls : ℕ) : 
  (initial_boys : ℚ) / initial_girls = 3 / 4 →
  girls_left = 2 * boys_left →
  boys_left = 10 →
  (final_boys : ℚ) / final_girls = 4 / 5 →
  final_boys = initial_boys - boys_left →
  final_girls = initial_girls - girls_left →
  initial_boys = 90 := by
sorry


end NUMINAMATH_CALUDE_high_school_ratio_problem_l3592_359237


namespace NUMINAMATH_CALUDE_wrap_vs_sleeve_difference_l3592_359297

def raw_squat : ℝ := 600
def sleeve_addition : ℝ := 30
def wrap_percentage : ℝ := 0.25

theorem wrap_vs_sleeve_difference :
  (raw_squat * wrap_percentage) - sleeve_addition = 120 := by
  sorry

end NUMINAMATH_CALUDE_wrap_vs_sleeve_difference_l3592_359297


namespace NUMINAMATH_CALUDE_spider_total_distance_l3592_359269

def spider_crawl (start end_1 end_2 end_3 : Int) : Int :=
  |end_1 - start| + |end_2 - end_1| + |end_3 - end_2|

theorem spider_total_distance :
  spider_crawl (-3) (-8) 0 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_spider_total_distance_l3592_359269


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3592_359271

def U : Set Int := Set.univ
def M : Set Int := {1, 2}
def P : Set Int := {-2, -1, 0, 1, 2}

theorem intersection_complement_equals_set : P ∩ (U \ M) = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3592_359271


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l3592_359270

-- Define the cost prices of the two articles
def cost_price_1 : ℝ := sorry
def cost_price_2 : ℝ := sorry

-- Define the conditions
def condition_1 : Prop :=
  (350 - cost_price_1) = 1.12 * (280 - cost_price_1)

def condition_2 : Prop :=
  (420 - cost_price_2) = 1.08 * (380 - cost_price_2)

-- Theorem to prove
theorem combined_cost_theorem :
  condition_1 ∧ condition_2 → cost_price_1 + cost_price_2 = 423.33 :=
by sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l3592_359270


namespace NUMINAMATH_CALUDE_shop_item_cost_prices_l3592_359279

theorem shop_item_cost_prices :
  ∀ (c1 c2 : ℝ),
    (0.30 * c1 - 0.15 * c1 = 120) →
    (0.25 * c2 - 0.10 * c2 = 150) →
    c1 = 800 ∧ c2 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_shop_item_cost_prices_l3592_359279


namespace NUMINAMATH_CALUDE_distinct_roots_sum_bound_l3592_359293

theorem distinct_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 9 = 0 → 
  r₂^2 + p*r₂ + 9 = 0 → 
  |r₁ + r₂| > 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_bound_l3592_359293


namespace NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l3592_359292

theorem probability_even_sum_two_wheels (wheel1_total : ℕ) (wheel1_even : ℕ) (wheel2_total : ℕ) (wheel2_even : ℕ) :
  wheel1_total = 2 * wheel1_even ∧ 
  wheel2_total = 5 ∧ 
  wheel2_even = 2 →
  (wheel1_even : ℚ) / wheel1_total * (wheel2_even : ℚ) / wheel2_total + 
  (wheel1_even : ℚ) / wheel1_total * ((wheel2_total - wheel2_even) : ℚ) / wheel2_total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l3592_359292


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3592_359221

-- Problem 1
theorem problem_1 : (12 : ℤ) - (-18) + (-7) - 15 = 8 := by sorry

-- Problem 2
theorem problem_2 : (-81 : ℚ) / (9/4) * (4/9) / (-16) = 1 := by sorry

-- Problem 3
theorem problem_3 : ((1/3 : ℚ) - 5/6 + 7/9) * (-18) = -5 := by sorry

-- Problem 4
theorem problem_4 : -(1 : ℚ)^4 - (1/5) * (2 - (-3))^2 = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3592_359221


namespace NUMINAMATH_CALUDE_exam_max_marks_l3592_359278

theorem exam_max_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) :
  percentage = 0.90 →
  scored_marks = 405 →
  percentage * max_marks = scored_marks →
  max_marks = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_max_marks_l3592_359278


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3592_359262

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 3) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 3 → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (1/x + 1/y = 1 + 2*Real.sqrt 2/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3592_359262


namespace NUMINAMATH_CALUDE_part_one_part_two_l3592_359272

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | f 1 x ≥ 4 - |x - 1|} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  ({x : ℝ | f ((1/m) + 1/(2*n)) x ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) →
  (∀ k l : ℝ, k > 0 → l > 0 → k * l ≥ m * n) →
  m * n = 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3592_359272


namespace NUMINAMATH_CALUDE_problem_statement_l3592_359209

theorem problem_statement (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = -2) : 
  y = 37 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3592_359209


namespace NUMINAMATH_CALUDE_max_p_value_l3592_359243

theorem max_p_value (p q r s : ℕ+) 
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90) :
  p ≤ 5324 ∧ ∃ (p' q' r' s' : ℕ+), 
    p' = 5324 ∧ 
    p' < 3 * q' ∧ 
    q' < 4 * r' ∧ 
    r' < 5 * s' ∧ 
    s' < 90 :=
by sorry

end NUMINAMATH_CALUDE_max_p_value_l3592_359243


namespace NUMINAMATH_CALUDE_max_value_constraint_l3592_359261

theorem max_value_constraint (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 19 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 4 → x'^2 + 8*y' + 3 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3592_359261


namespace NUMINAMATH_CALUDE_intersection_range_l3592_359287

/-- The function f(x) = x^3 - 3x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- Predicate to check if a line y = m intersects f at three distinct points -/
def has_three_distinct_intersections (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m

/-- Theorem stating the range of m for which y = m intersects f at three distinct points -/
theorem intersection_range :
  ∀ m : ℝ, has_three_distinct_intersections m ↔ m > -3 ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3592_359287


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l3592_359206

theorem quadratic_equation_from_roots (α β : ℝ) (h1 : α + β = 5) (h2 : α * β = 6) :
  ∃ a b c : ℝ, a ≠ 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 ∧ 
  a = 1 ∧ b = -5 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l3592_359206


namespace NUMINAMATH_CALUDE_circle_k_range_l3592_359274

/-- The equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 5*k = 0

/-- The condition for the equation to represent a circle -/
def is_circle (k : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y k ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The theorem stating the range of k for which the equation represents a circle -/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l3592_359274


namespace NUMINAMATH_CALUDE_adjacent_above_350_l3592_359242

/-- Represents a position in the triangular grid -/
structure GridPosition where
  row : ℕ
  column : ℕ

/-- Returns the number at a given position in the triangular grid -/
def numberAt (pos : GridPosition) : ℕ := sorry

/-- Returns the position of a given number in the triangular grid -/
def positionOf (n : ℕ) : GridPosition := sorry

/-- Returns the number in the horizontally adjacent triangle in the row above -/
def adjacentAbove (n : ℕ) : ℕ := sorry

theorem adjacent_above_350 : adjacentAbove 350 = 314 := by sorry

end NUMINAMATH_CALUDE_adjacent_above_350_l3592_359242


namespace NUMINAMATH_CALUDE_power_sum_inequality_l3592_359298

theorem power_sum_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_eq_three : a + b + c = 3) : 
  a^a + b^b + c^c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l3592_359298


namespace NUMINAMATH_CALUDE_distance_to_work_l3592_359230

-- Define the problem parameters
def speed_to_work : ℝ := 45
def speed_from_work : ℝ := 30
def total_commute_time : ℝ := 1

-- Define the theorem
theorem distance_to_work :
  ∃ (d : ℝ), d / speed_to_work + d / speed_from_work = total_commute_time ∧ d = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_distance_to_work_l3592_359230


namespace NUMINAMATH_CALUDE_revenue_decrease_l3592_359254

theorem revenue_decrease (projected_increase : ℝ) (actual_vs_projected : ℝ) 
  (h1 : projected_increase = 0.20)
  (h2 : actual_vs_projected = 0.625) : 
  (1 - actual_vs_projected * (1 + projected_increase)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l3592_359254


namespace NUMINAMATH_CALUDE_four_groups_four_spots_l3592_359233

/-- The number of ways to arrange tour groups among scenic spots with one spot unvisited -/
def tourArrangements (numGroups numSpots : ℕ) : ℕ :=
  (numGroups.choose 2) * (numSpots.factorial / (numSpots - 3).factorial)

/-- Theorem stating the number of arrangements for 4 groups and 4 spots -/
theorem four_groups_four_spots :
  tourArrangements 4 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_four_groups_four_spots_l3592_359233


namespace NUMINAMATH_CALUDE_janes_stick_length_l3592_359246

-- Define the lengths of the sticks and other quantities
def pat_stick_length : ℕ := 30
def covered_length : ℕ := 7
def feet_to_inches : ℕ := 12

-- Define the theorem
theorem janes_stick_length :
  let uncovered_length : ℕ := pat_stick_length - covered_length
  let sarahs_stick_length : ℕ := 2 * uncovered_length
  let janes_stick_length : ℕ := sarahs_stick_length - 2 * feet_to_inches
  janes_stick_length = 22 := by
sorry

end NUMINAMATH_CALUDE_janes_stick_length_l3592_359246


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l3592_359295

theorem min_value_cos_sin (θ : Real) (h : θ ∈ Set.Icc (-π/12) (π/12)) :
  ∃ (m : Real), m = (Real.sqrt 3 - 1) / 2 ∧ 
    ∀ x ∈ Set.Icc (-π/12) (π/12), 
      Real.cos (x + π/4) + Real.sin (2*x) ≥ m ∧
      ∃ y ∈ Set.Icc (-π/12) (π/12), Real.cos (y + π/4) + Real.sin (2*y) = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l3592_359295


namespace NUMINAMATH_CALUDE_complex_magnitude_l3592_359289

theorem complex_magnitude (z : ℂ) (h : z + (z - 1) * Complex.I = 3) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3592_359289


namespace NUMINAMATH_CALUDE_three_intersections_iff_zero_l3592_359238

/-- The number of distinct intersection points between the curves x^2 - y^2 = a^2 and (x-1)^2 + y^2 = 1 -/
def intersection_count (a : ℝ) : ℕ :=
  sorry

/-- The condition for exactly three distinct intersection points -/
def has_three_intersections (a : ℝ) : Prop :=
  intersection_count a = 3

theorem three_intersections_iff_zero (a : ℝ) :
  has_three_intersections a ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_three_intersections_iff_zero_l3592_359238


namespace NUMINAMATH_CALUDE_cone_sphere_volume_difference_l3592_359290

/-- Given an equilateral cone with an inscribed sphere, prove that the difference in volume
    between the cone and the sphere is (10/3) * √(2/π) dm³ when the surface area of the cone
    is 10 dm² more than the surface area of the sphere. -/
theorem cone_sphere_volume_difference (R : ℝ) (h : R > 0) :
  let r := R / Real.sqrt 3
  let cone_surface_area := 3 * Real.pi * R^2
  let sphere_surface_area := 4 * Real.pi * r^2
  let cone_volume := (Real.pi * Real.sqrt 3 / 3) * R^3
  let sphere_volume := (4 * Real.pi / 3) * r^3
  cone_surface_area = sphere_surface_area + 10 →
  cone_volume - sphere_volume = (10 / 3) * Real.sqrt (2 / Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_difference_l3592_359290


namespace NUMINAMATH_CALUDE_circle_inequality_l3592_359264

theorem circle_inequality (a b c d : ℝ) (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x1^2 + y1^2 = 1) (h2 : x2^2 + y2^2 = 1) 
  (h3 : x3^2 + y3^2 = 1) (h4 : x4^2 + y4^2 = 1) :
  (a*y1 + b*y2 + c*y3 + d*y4)^2 + (a*x4 + b*x3 + c*x2 + d*x1)^2 
  ≤ 2 * ((a^2 + b^2)/(a*b) + (c^2 + d^2)/(c*d)) := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l3592_359264


namespace NUMINAMATH_CALUDE_diamond_paths_count_l3592_359258

/-- Represents a position in the diamond grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a move in the diamond grid -/
inductive Move
  | Right
  | Down
  | DiagonalDown

/-- The diamond-shaped grid containing the word "DIAMOND" -/
def diamond_grid : List (List Char) := sorry

/-- Check if a move is valid in the diamond grid -/
def is_valid_move (grid : List (List Char)) (pos : Position) (move : Move) : Bool := sorry

/-- Get the next position after a move -/
def next_position (pos : Position) (move : Move) : Position := sorry

/-- Check if a path spells "DIAMOND" -/
def spells_diamond (grid : List (List Char)) (path : List Move) : Bool := sorry

/-- Count the number of valid paths spelling "DIAMOND" -/
def count_diamond_paths (grid : List (List Char)) : ℕ := sorry

theorem diamond_paths_count :
  count_diamond_paths diamond_grid = 64 := by sorry

end NUMINAMATH_CALUDE_diamond_paths_count_l3592_359258


namespace NUMINAMATH_CALUDE_centroid_trajectory_l3592_359227

/-- Given a triangle ABC with vertices A(-3, 0), B(3, 0), and C(m, n) on the parabola y² = 6x,
    the centroid (x, y) of the triangle satisfies the equation y² = 2x for x ≠ 0. -/
theorem centroid_trajectory (m n x y : ℝ) : 
  n^2 = 6*m →                   -- C is on the parabola y² = 6x
  3*x = m →                     -- x-coordinate of centroid
  3*y = n →                     -- y-coordinate of centroid
  x ≠ 0 →                       -- x is non-zero
  y^2 = 2*x                     -- equation of centroid's trajectory
  := by sorry

end NUMINAMATH_CALUDE_centroid_trajectory_l3592_359227


namespace NUMINAMATH_CALUDE_school_bus_time_theorem_l3592_359291

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Converts 24-hour format to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  if t.hours ≤ 12 then t else { hours := t.hours - 12, minutes := t.minutes }

/-- Calculates the time difference in minutes between two Time values -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

theorem school_bus_time_theorem :
  let schoolEndTime : Time := { hours := 16, minutes := 45 }
  let arrivalTime : Time := { hours := 17, minutes := 20 }
  (to12HourFormat schoolEndTime = { hours := 4, minutes := 45 }) ∧
  (timeDifference schoolEndTime arrivalTime = 35) :=
by sorry

end NUMINAMATH_CALUDE_school_bus_time_theorem_l3592_359291


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3592_359201

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The specific problem of a ball dropped from 120 feet with 1/3 rebound ratio -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 248 + 26/27 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3592_359201


namespace NUMINAMATH_CALUDE_problem_solution_l3592_359260

theorem problem_solution : ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ (x = 230) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3592_359260


namespace NUMINAMATH_CALUDE_whatsis_whosis_equals_so_plus_so_l3592_359299

/-- A structure representing the variables in the problem -/
structure Variables where
  whatsis : ℝ
  whosis : ℝ
  is : ℝ
  so : ℝ
  pos_whatsis : 0 < whatsis
  pos_whosis : 0 < whosis
  pos_is : 0 < is
  pos_so : 0 < so

/-- The main theorem representing the problem -/
theorem whatsis_whosis_equals_so_plus_so (v : Variables) 
  (h1 : v.whatsis = v.so)
  (h2 : v.whosis = v.is)
  (h3 : v.so + v.so = v.is * v.so)
  (h4 : v.whosis = v.so)
  (h5 : v.so + v.so = v.so * v.so)
  (h6 : v.is = 2) :
  v.whosis * v.whatsis = v.so + v.so := by
  sorry


end NUMINAMATH_CALUDE_whatsis_whosis_equals_so_plus_so_l3592_359299


namespace NUMINAMATH_CALUDE_expression_value_l3592_359228

theorem expression_value (x y : ℝ) (h : (x - y) / y = 2) :
  ((1 / (x - y) + 1 / (x + y)) / (x / (x - y)^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3592_359228


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3592_359232

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - (k+2)*x + 2*k - 1 = 0

theorem quadratic_equation_properties :
  (∀ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) ∧
  (∃ k : ℝ, quadratic_equation k 3 ∧ quadratic_equation k 1 ∧ k = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3592_359232


namespace NUMINAMATH_CALUDE_product_divisors_24_power_5_l3592_359235

/-- The product of divisors function -/
def prod_divisors (n : ℕ+) : ℕ+ :=
  sorry

theorem product_divisors_24_power_5 (n : ℕ+) :
  prod_divisors n = (24 : ℕ+) ^ 240 → n = (24 : ℕ+) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_product_divisors_24_power_5_l3592_359235


namespace NUMINAMATH_CALUDE_adams_stairs_l3592_359252

theorem adams_stairs (total_steps : ℕ) (steps_left : ℕ) (steps_climbed : ℕ) : 
  total_steps = 96 → steps_left = 22 → steps_climbed = total_steps - steps_left → steps_climbed = 74 := by
  sorry

end NUMINAMATH_CALUDE_adams_stairs_l3592_359252


namespace NUMINAMATH_CALUDE_gwen_book_count_l3592_359222

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_book_count : total_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_gwen_book_count_l3592_359222


namespace NUMINAMATH_CALUDE_digit_156_is_zero_l3592_359256

-- Define the fraction
def fraction : ℚ := 37 / 740

-- Define a function to get the nth digit after the decimal point
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_156_is_zero : nthDigitAfterDecimal fraction 156 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_156_is_zero_l3592_359256


namespace NUMINAMATH_CALUDE_abs_neg_four_minus_two_l3592_359225

theorem abs_neg_four_minus_two : |(-4 : ℤ) - 2| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_four_minus_two_l3592_359225


namespace NUMINAMATH_CALUDE_birch_not_adjacent_probability_l3592_359266

def maple_count : ℕ := 3
def oak_count : ℕ := 4
def birch_count : ℕ := 5

def total_trees : ℕ := maple_count + oak_count + birch_count

def total_arrangements : ℕ := Nat.factorial total_trees / (Nat.factorial maple_count * Nat.factorial oak_count * Nat.factorial birch_count)

def favorable_arrangements : ℕ := (Nat.choose (maple_count + oak_count + 1) birch_count) * (Nat.factorial (maple_count + oak_count))

theorem birch_not_adjacent_probability : 
  (favorable_arrangements : ℚ) / total_arrangements = 7 / 99 := by sorry

end NUMINAMATH_CALUDE_birch_not_adjacent_probability_l3592_359266


namespace NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_coefficients_l3592_359277

-- Definition of a double root equation
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

-- Theorem 1
theorem first_equation_is_double_root :
  is_double_root_equation 1 (-3) 2 :=
sorry

-- Theorem 2
theorem second_equation_coefficients (a b : ℝ) :
  is_double_root_equation a b (-6) ∧ (a * 2^2 + b * 2 - 6 = 0) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_coefficients_l3592_359277


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l3592_359251

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 21.5) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l3592_359251


namespace NUMINAMATH_CALUDE_min_value_expression_l3592_359248

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x * y) ≥ 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (((x₀^2 + y₀^2) * (4*x₀^2 + y₀^2)).sqrt) / (x₀ * y₀) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3592_359248


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3592_359250

theorem complex_equation_solution (x y : ℝ) :
  (x + y * Complex.I) / (3 - 2 * Complex.I) = 1 + Complex.I →
  Complex.im (x + y * Complex.I) = 1 ∧ Complex.abs (x + y * Complex.I) = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3592_359250


namespace NUMINAMATH_CALUDE_exam_candidates_count_l3592_359218

theorem exam_candidates_count :
  ∀ (N : ℕ) (total_avg marks_11th : ℝ) (avg_first_10 avg_last_11 : ℝ),
    total_avg = 48 →
    avg_first_10 = 55 →
    avg_last_11 = 40 →
    marks_11th = 66 →
    N * total_avg = 10 * avg_first_10 + 11 * avg_last_11 - marks_11th →
    N = 21 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l3592_359218


namespace NUMINAMATH_CALUDE_final_sum_after_transformations_l3592_359281

/-- Given two numbers x and y with sum T, prove that after transformations, 
    the sum of the resulting numbers is 4T + 22 -/
theorem final_sum_after_transformations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 4) + 2 * (y + 5) = 4 * T + 22 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformations_l3592_359281


namespace NUMINAMATH_CALUDE_mans_speed_against_stream_l3592_359265

theorem mans_speed_against_stream 
  (rate : ℝ) 
  (speed_with_stream : ℝ) 
  (h1 : rate = 4) 
  (h2 : speed_with_stream = 12) : 
  abs (rate - (speed_with_stream - rate)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_stream_l3592_359265


namespace NUMINAMATH_CALUDE_combinations_theorem_l3592_359205

/-- The number of choices in the art group -/
def art_choices : ℕ := 2

/-- The number of choices in the sports group -/
def sports_choices : ℕ := 3

/-- The number of choices in the music group -/
def music_choices : ℕ := 4

/-- The total number of possible combinations -/
def total_combinations : ℕ := art_choices * sports_choices * music_choices

theorem combinations_theorem : total_combinations = 24 := by
  sorry

end NUMINAMATH_CALUDE_combinations_theorem_l3592_359205


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l3592_359284

/-- A tetrahedron with an inscribed sphere --/
structure Tetrahedron :=
  (r : ℝ) -- radius of inscribed sphere
  (a b : ℝ) -- lengths of a pair of opposite edges
  (r_pos : r > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Theorem: The radius of the inscribed sphere is less than ab/(2(a+b)) --/
theorem inscribed_sphere_radius_bound (t : Tetrahedron) : t.r < (t.a * t.b) / (2 * (t.a + t.b)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l3592_359284


namespace NUMINAMATH_CALUDE_albino_deer_antlers_l3592_359213

theorem albino_deer_antlers (total_deer : ℕ) (albino_deer : ℕ) 
  (h1 : total_deer = 920)
  (h2 : albino_deer = 23)
  (h3 : albino_deer = (total_deer * 10 / 100) / 4) : 
  albino_deer = 23 := by
  sorry

end NUMINAMATH_CALUDE_albino_deer_antlers_l3592_359213


namespace NUMINAMATH_CALUDE_parabola_point_relation_l3592_359288

theorem parabola_point_relation (a y₁ y₂ y₃ : ℝ) :
  a < -1 →
  y₁ = (a - 1)^2 →
  y₂ = a^2 →
  y₃ = (a + 1)^2 →
  y₁ > y₂ ∧ y₂ > y₃ :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l3592_359288


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l3592_359275

/-- Given a sinusoidal function with specific properties, prove its exact form and the set of x-values where it equals 1. -/
theorem sinusoidal_function_properties (A ω φ : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h2 : A > 0)
  (h3 : ω > 0)
  (h4 : |φ| < π)
  (h5 : f (π/8) = 2)
  (h6 : f (5*π/8) = -2) :
  (∀ x, f x = 2 * Real.sin (2*x + π/4)) ∧ 
  (∀ x, f x = 1 ↔ ∃ k : ℤ, x = -π/24 + k*π ∨ x = 7*π/24 + k*π) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l3592_359275


namespace NUMINAMATH_CALUDE_correct_selling_price_B_l3592_359207

/-- Represents the pricing and sales data for laundry detergents --/
structure LaundryDetergentData where
  cost_diff : ℝ               -- Cost difference between brands
  total_cost_A : ℝ            -- Total cost for brand A
  total_cost_B : ℝ            -- Total cost for brand B
  sell_price_A : ℝ            -- Selling price of brand A
  daily_sales_A : ℝ           -- Daily sales of brand A
  base_price_B : ℝ            -- Base selling price of brand B
  base_sales_B : ℝ            -- Base daily sales of brand B
  price_sales_ratio : ℝ       -- Ratio of price increase to sales decrease for B

/-- Calculates the selling price of brand B for a given total daily profit --/
def calculate_selling_price_B (data : LaundryDetergentData) (total_profit : ℝ) : ℝ :=
  sorry

/-- Theorem stating the correct selling price for brand B --/
theorem correct_selling_price_B (data : LaundryDetergentData) :
  let d := {
    cost_diff := 10,
    total_cost_A := 3000,
    total_cost_B := 4000,
    sell_price_A := 45,
    daily_sales_A := 100,
    base_price_B := 50,
    base_sales_B := 140,
    price_sales_ratio := 2
  }
  calculate_selling_price_B d 4700 = 80 := by sorry

end NUMINAMATH_CALUDE_correct_selling_price_B_l3592_359207


namespace NUMINAMATH_CALUDE_basketball_team_selection_with_twins_l3592_359219

def number_of_players : ℕ := 16
def number_of_starters : ℕ := 7

theorem basketball_team_selection_with_twins :
  (Nat.choose (number_of_players - 2) (number_of_starters - 2)) +
  (Nat.choose (number_of_players - 2) number_of_starters) =
  (Nat.choose 14 5) + (Nat.choose 14 7) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_with_twins_l3592_359219


namespace NUMINAMATH_CALUDE_fifth_term_is_123_40_l3592_359229

-- Define the arithmetic sequence
def arithmeticSequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + y
  | 1 => x - y
  | 2 => x * y
  | 3 => x / y
  | n + 4 => arithmeticSequence x y 3 - (n + 1) * (2 * y)

-- Theorem statement
theorem fifth_term_is_123_40 (x y : ℚ) :
  x - y - (x + y) = -2 * y →
  x - 3 * y = x * y →
  x - 5 * y = x / y →
  y ≠ 0 →
  arithmeticSequence x y 4 = 123 / 40 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_is_123_40_l3592_359229


namespace NUMINAMATH_CALUDE_sum_of_differences_base7_l3592_359223

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else go (m / 7) ((m % 7) :: acc)
  go n []

/-- Calculates the difference between two base 7 numbers -/
def diffBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a - toDecimal b)

/-- Calculates the sum of two base 7 numbers -/
def sumBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a + toDecimal b)

theorem sum_of_differences_base7 :
  let a := [5, 2, 4, 3]
  let b := [3, 1, 0, 5]
  let c := [6, 6, 6, 5]
  let d := [4, 3, 1, 2]
  let result := [4, 4, 5, 2]
  sumBase7 (diffBase7 a b) (diffBase7 c d) = result :=
by sorry

end NUMINAMATH_CALUDE_sum_of_differences_base7_l3592_359223


namespace NUMINAMATH_CALUDE_inequalities_proof_l3592_359214

theorem inequalities_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : Real.sqrt a ^ 3 + Real.sqrt b ^ 3 + Real.sqrt c ^ 3 = 1) :
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3592_359214


namespace NUMINAMATH_CALUDE_base9_addition_theorem_l3592_359212

/-- Addition of numbers in base 9 --/
def base9_add (a b c : ℕ) : ℕ :=
  sorry

/-- Conversion from base 9 to base 10 --/
def base9_to_base10 (n : ℕ) : ℕ :=
  sorry

theorem base9_addition_theorem :
  base9_add 2175 1714 406 = 4406 :=
by sorry

end NUMINAMATH_CALUDE_base9_addition_theorem_l3592_359212


namespace NUMINAMATH_CALUDE_range_of_k_for_inequality_l3592_359285

theorem range_of_k_for_inequality (k : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → k ∈ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_inequality_l3592_359285


namespace NUMINAMATH_CALUDE_abc_product_l3592_359241

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 171) (h2 : b * (c + a) = 180) (h3 : c * (a + b) = 189) :
  a * b * c = 270 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3592_359241


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_two_i_squared_l3592_359220

theorem imaginary_part_of_one_minus_two_i_squared (i : ℂ) : 
  i * i = -1 → Complex.im ((1 - 2*i)^2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_two_i_squared_l3592_359220


namespace NUMINAMATH_CALUDE_M_intersect_P_eq_y_geq_1_l3592_359245

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def P : Set ℝ := {y | ∃ x : ℝ, y = Real.log x}

-- State the theorem
theorem M_intersect_P_eq_y_geq_1 : M ∩ P = {y : ℝ | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_P_eq_y_geq_1_l3592_359245


namespace NUMINAMATH_CALUDE_puzzle_solution_l3592_359226

theorem puzzle_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 17) : 
  w - y = 73 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3592_359226


namespace NUMINAMATH_CALUDE_fraction_count_l3592_359296

theorem fraction_count : ∃ (S : Finset ℕ), 
  (∀ a ∈ S, a > 1 ∧ a < 7) ∧ 
  (∀ a ∉ S, a ≤ 1 ∨ a ≥ 7 ∨ (a - 1) / a ≥ 6 / 7) ∧
  S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_count_l3592_359296


namespace NUMINAMATH_CALUDE_perimeter_marbles_12_l3592_359239

/-- A square made of marbles -/
structure MarbleSquare where
  side_length : ℕ
  
/-- The number of marbles on the perimeter of a square -/
def perimeter_marbles (square : MarbleSquare) : ℕ :=
  4 * square.side_length - 4

theorem perimeter_marbles_12 :
  ∀ (square : MarbleSquare),
    square.side_length = 12 →
    perimeter_marbles square = 44 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_marbles_12_l3592_359239


namespace NUMINAMATH_CALUDE_train_late_speed_l3592_359234

/-- Proves that the late average speed is 35 kmph given the conditions of the train problem -/
theorem train_late_speed (distance : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  distance = 70 →
  on_time_speed = 40 →
  late_time = (distance / on_time_speed) + (15 / 60) →
  ∃ (late_speed : ℝ), late_speed = distance / late_time ∧ late_speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_late_speed_l3592_359234


namespace NUMINAMATH_CALUDE_star_example_l3592_359208

-- Define the * operation
def star (a b : ℕ) : ℕ := a + 2 * b

-- State the theorem
theorem star_example : star (star 2 3) 4 = 16 := by sorry

end NUMINAMATH_CALUDE_star_example_l3592_359208


namespace NUMINAMATH_CALUDE_average_problem_l3592_359267

theorem average_problem (x : ℝ) : 
  (2 + 4 + 1 + 3 + x) / 5 = 3 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3592_359267


namespace NUMINAMATH_CALUDE_prob_red_or_white_is_eleven_thirteenths_l3592_359236

/-- Represents the color of a marble -/
inductive MarbleColor
  | Blue
  | Red
  | White

/-- Represents the properties of marbles in a bag -/
structure MarbleBag where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ
  blue_size : ℚ
  red_size : ℚ
  white_size : ℚ
  h_total : total = blue + red + white
  h_blue_size : blue_size = 2 * red_size
  h_white_size : white_size = red_size

/-- Calculates the probability of selecting a red or white marble from the bag -/
def prob_red_or_white (bag : MarbleBag) : ℚ :=
  let total_size := bag.blue * bag.blue_size + bag.red * bag.red_size + bag.white * bag.white_size
  (bag.red * bag.red_size + bag.white * bag.white_size) / total_size

theorem prob_red_or_white_is_eleven_thirteenths (bag : MarbleBag) 
  (h_total : bag.total = 60)
  (h_blue : bag.blue = 5)
  (h_red : bag.red = 9)
  (h_red_size : bag.red_size = 1) :
  prob_red_or_white bag = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_or_white_is_eleven_thirteenths_l3592_359236


namespace NUMINAMATH_CALUDE_range_of_a_l3592_359244

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|4*x - 3| ≤ 1 → x^2 - (2*a + 1)*x + a^2 + a ≤ 0) ∧
   ¬(∀ x : ℝ, x^2 - (2*a + 1)*x + a^2 + a ≤ 0 → |4*x - 3| ≤ 1)) →
  0 ≤ a ∧ a ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3592_359244


namespace NUMINAMATH_CALUDE_three_not_in_range_of_g_l3592_359273

/-- The quadratic function g(x) = x^2 + 3x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

/-- 3 is not in the range of g(x) if and only if c > 21/4 -/
theorem three_not_in_range_of_g (c : ℝ) :
  (∀ x, g c x ≠ 3) ↔ c > 21/4 := by sorry

end NUMINAMATH_CALUDE_three_not_in_range_of_g_l3592_359273


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l3592_359240

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l3592_359240


namespace NUMINAMATH_CALUDE_quotient_problem_l3592_359203

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 139)
  (h2 : divisor = 19)
  (h3 : remainder = 6)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 7 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3592_359203


namespace NUMINAMATH_CALUDE_freshmen_assignment_l3592_359216

/-- The number of ways to assign n freshmen to k classes with at least one freshman in each class -/
def assignFreshmen (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange m groups into k classes -/
def arrangeGroups (m k : ℕ) : ℕ :=
  sorry

theorem freshmen_assignment :
  assignFreshmen 5 3 * arrangeGroups 3 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_freshmen_assignment_l3592_359216


namespace NUMINAMATH_CALUDE_log_ride_cost_l3592_359283

/-- The cost of a single log ride, given the following conditions:
  * Dolly wants to ride the Ferris wheel twice
  * Dolly wants to ride the roller coaster three times
  * Dolly wants to ride the log ride seven times
  * The Ferris wheel costs 2 tickets per ride
  * The roller coaster costs 5 tickets per ride
  * Dolly has 20 tickets
  * Dolly needs to buy 6 more tickets
-/
theorem log_ride_cost : ℕ := by
  sorry

#check log_ride_cost

end NUMINAMATH_CALUDE_log_ride_cost_l3592_359283


namespace NUMINAMATH_CALUDE_largest_common_divisor_360_450_l3592_359255

theorem largest_common_divisor_360_450 : Nat.gcd 360 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_360_450_l3592_359255


namespace NUMINAMATH_CALUDE_rectangle_area_l3592_359268

/-- Given a rectangle with length four times its width and perimeter 200 cm, 
    its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h₁ : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 → l * w = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3592_359268


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3592_359215

/-- Atomic weights of elements in g/mol -/
def atomic_weight : String → ℝ
  | "H" => 1
  | "N" => 14
  | "O" => 16
  | "S" => 32
  | "Fe" => 56
  | _ => 0

/-- Molecular weight of a compound given its chemical formula -/
def molecular_weight (formula : String) : ℝ := sorry

/-- The compound (NH4)2SO4·Fe2(SO4)3·6H2O -/
def compound : String := "(NH4)2SO4·Fe2(SO4)3·6H2O"

/-- Theorem stating that the molecular weight of (NH4)2SO4·Fe2(SO4)3·6H2O is 772 g/mol -/
theorem compound_molecular_weight :
  molecular_weight compound = 772 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3592_359215


namespace NUMINAMATH_CALUDE_abs_neg_two_l3592_359253

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_l3592_359253


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3592_359257

theorem consecutive_odd_numbers_sum (n : ℕ) : 
  (n % 2 = 1) → (n + (n + 2) = 48) → n = 23 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3592_359257


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3592_359211

theorem sum_of_fifth_powers (a b c : ℝ) (h : a + b + c = 0) :
  2 * (a^5 + b^5 + c^5) = 5 * a * b * c * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3592_359211


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3592_359224

/-- A quadratic function satisfying certain conditions -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x : ℝ, ∀ y : ℝ, f a b c x ≤ f a b c y) →
  (∃ x : ℝ, f a b c x = 0) →
  (∃ m : ℝ, m > 1 ∧ 
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x)) ∧
  (∀ m : ℝ, (m > 1 ∧ 
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x)) → m = 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3592_359224


namespace NUMINAMATH_CALUDE_festival_attendance_ratio_l3592_359249

/-- Represents a 3-day music festival attendance --/
structure FestivalAttendance where
  total : ℕ
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- The conditions of the festival attendance --/
def festivalConditions (f : FestivalAttendance) : Prop :=
  f.total = 2700 ∧
  f.day2 = f.day1 / 2 ∧
  f.day2 = 300 ∧
  f.total = f.day1 + f.day2 + f.day3

/-- The theorem stating the ratio of third day to first day attendance --/
theorem festival_attendance_ratio (f : FestivalAttendance) 
  (h : festivalConditions f) : f.day3 = 3 * f.day1 := by
  sorry

#check festival_attendance_ratio

end NUMINAMATH_CALUDE_festival_attendance_ratio_l3592_359249


namespace NUMINAMATH_CALUDE_tromino_bounds_l3592_359259

/-- A tromino is a 1 x 3 rectangle that covers exactly three squares on a board. -/
structure Tromino

/-- The board is an n x n grid where trominoes can be placed. -/
structure Board (n : ℕ) where
  size : n > 0

/-- f(n) is the smallest number of trominoes required to stop any more being placed on an n x n board. -/
noncomputable def f (n : ℕ) : ℕ :=
  sorry

/-- For all positive n, there exist real numbers h and k such that
    (n^2 / 7) + hn ≤ f(n) ≤ (n^2 / 5) + kn -/
theorem tromino_bounds (n : ℕ) (b : Board n) :
  ∃ (h k : ℝ), (n^2 / 7 : ℝ) + h * n ≤ f n ∧ (f n : ℝ) ≤ n^2 / 5 + k * n :=
sorry

end NUMINAMATH_CALUDE_tromino_bounds_l3592_359259


namespace NUMINAMATH_CALUDE_total_lemons_picked_l3592_359200

theorem total_lemons_picked (sally_lemons mary_lemons : ℕ) 
  (h1 : sally_lemons = 7)
  (h2 : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 := by
sorry

end NUMINAMATH_CALUDE_total_lemons_picked_l3592_359200


namespace NUMINAMATH_CALUDE_floor_sum_equals_140_l3592_359263

theorem floor_sum_equals_140 (p q r s : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : p^2 + q^2 = 2500) (h2 : r^2 + s^2 = 2500) (h3 : p * r = 1200) (h4 : q * s = 1200) :
  ⌊p + q + r + s⌋ = 140 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equals_140_l3592_359263


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l3592_359217

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define axis of symmetry for a function
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  EvenFunction f → AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l3592_359217


namespace NUMINAMATH_CALUDE_extrema_of_quadratic_form_l3592_359282

theorem extrema_of_quadratic_form (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_extrema_of_quadratic_form_l3592_359282


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3592_359286

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 1)
  (h_sum : a 3 + a 4 = 8) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3592_359286


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3592_359280

theorem simplify_and_evaluate : 
  ∀ (a b : ℝ), 
    (a + 2*b)^2 - (a + b)*(a - b) = 4*a*b + 5*b^2 ∧
    (((-1/2) + 2*2)^2 - ((-1/2) + 2)*((-1/2) - 2) = 16) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3592_359280


namespace NUMINAMATH_CALUDE_abs_sum_inequalities_l3592_359202

theorem abs_sum_inequalities (a b : ℝ) (h : a * b > 0) : 
  (abs (a + b) > abs a) ∧ (abs (a + b) > abs (a - b)) := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequalities_l3592_359202
