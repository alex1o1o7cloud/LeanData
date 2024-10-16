import Mathlib

namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l3648_364829

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l3648_364829


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3648_364867

/-- If x^2 + 3x + a = 0 has -1 as one of its roots, then the other root is -2 -/
theorem other_root_of_quadratic (a : ℝ) : 
  ((-1 : ℝ)^2 + 3*(-1) + a = 0) → 
  (∃ x : ℝ, x ≠ -1 ∧ x^2 + 3*x + a = 0 ∧ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3648_364867


namespace NUMINAMATH_CALUDE_charity_event_revenue_l3648_364815

theorem charity_event_revenue (total_tickets : Nat) (total_revenue : Nat) 
  (full_price_tickets : Nat) (discount_tickets : Nat) (full_price : Nat) :
  total_tickets = 190 →
  total_revenue = 2871 →
  full_price_tickets + discount_tickets = total_tickets →
  full_price_tickets * full_price + discount_tickets * (full_price / 3) = total_revenue →
  full_price_tickets * full_price = 1900 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_revenue_l3648_364815


namespace NUMINAMATH_CALUDE_divide_by_sqrt_two_l3648_364816

theorem divide_by_sqrt_two : 2 / Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_sqrt_two_l3648_364816


namespace NUMINAMATH_CALUDE_line_through_two_points_l3648_364890

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem stating that the equation represents a line through two points
theorem line_through_two_points (M N : Point2D) (h : M ≠ N) :
  ∃! l : Line2D, pointOnLine M l ∧ pointOnLine N l ∧
  ∀ P : Point2D, pointOnLine P l ↔ (P.x - M.x) / (N.x - M.x) = (P.y - M.y) / (N.y - M.y) :=
sorry

end NUMINAMATH_CALUDE_line_through_two_points_l3648_364890


namespace NUMINAMATH_CALUDE_frac_repeating_block_length_l3648_364839

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7/13 is a rational number -/
def frac : ℚ := 7 / 13

theorem frac_repeating_block_length : 
  ∃ (n : ℕ) (k : ℕ+) (a b : ℕ), 
    frac * 10^n = (a : ℚ) + (b : ℚ) / (10^repeating_block_length - 1) ∧
    b < 10^repeating_block_length - 1 ∧
    ∀ m < repeating_block_length, 
      ¬∃ (c d : ℕ), frac * 10^n = (c : ℚ) + (d : ℚ) / (10^m - 1) ∧ d < 10^m - 1 :=
sorry

end NUMINAMATH_CALUDE_frac_repeating_block_length_l3648_364839


namespace NUMINAMATH_CALUDE_max_sundays_in_45_days_l3648_364888

/-- Represents the day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date within the first 45 days of a year -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Returns the number of Sundays in the first 45 days of a year -/
def countSundays (startDay : DayOfWeek) : Nat :=
  sorry

/-- The maximum number of Sundays in the first 45 days of a year -/
def maxSundays : Nat :=
  sorry

theorem max_sundays_in_45_days :
  maxSundays = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sundays_in_45_days_l3648_364888


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3648_364852

theorem marble_bag_problem (total_marbles : ℕ) (red_marbles : ℕ) 
  (probability_non_red : ℚ) : 
  red_marbles = 12 → 
  probability_non_red = 36 / 49 → 
  (((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = probability_non_red) → 
  total_marbles = 84 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3648_364852


namespace NUMINAMATH_CALUDE_valid_grid_probability_l3648_364828

/-- Represents a 3x3 grid filled with numbers 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if a number is odd --/
def isOdd (n : Fin 9) : Bool :=
  n.val % 2 ≠ 0

/-- Checks if the sum of numbers in a row is odd --/
def isRowSumOdd (g : Grid) (row : Fin 3) : Bool :=
  isOdd (g row 0 + g row 1 + g row 2)

/-- Checks if the sum of numbers in a column is odd --/
def isColumnSumOdd (g : Grid) (col : Fin 3) : Bool :=
  isOdd (g 0 col + g 1 col + g 2 col)

/-- Checks if all rows and columns have odd sums --/
def isValidGrid (g : Grid) : Bool :=
  (∀ row, isRowSumOdd g row) ∧ (∀ col, isColumnSumOdd g col)

/-- The total number of possible 3x3 grids filled with numbers 1 to 9 --/
def totalGrids : Nat :=
  Nat.factorial 9

/-- The number of valid grids where all rows and columns have odd sums --/
def validGrids : Nat :=
  9

/-- The main theorem stating the probability of a valid grid --/
theorem valid_grid_probability :
  (validGrids : ℚ) / totalGrids = 1 / 14 :=
sorry

end NUMINAMATH_CALUDE_valid_grid_probability_l3648_364828


namespace NUMINAMATH_CALUDE_range_of_a_l3648_364860

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (4*x - 3)^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, x ∉ B a → x ∉ A) ∧ ¬(∀ x, x ∉ A → x ∉ B a)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3648_364860


namespace NUMINAMATH_CALUDE_circle_center_perpendicular_line_l3648_364882

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := sorry

-- Define the center of the circle
def center : ℝ × ℝ := sorry

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the perpendicular line passing through the center
def perpendicular_line (x y : ℝ) : Prop := x + y - 3 = 0

theorem circle_center_perpendicular_line :
  (1, 0) ∈ circle_C ∧
  center.1 > 0 ∧
  center.2 = 0 ∧
  (∃ (a b : ℝ), (a, b) ∈ circle_C ∧ line_l a b ∧
    (a - center.1)^2 + (b - center.2)^2 = 8) →
  ∀ x y, perpendicular_line x y ↔ 
    (x - center.1) * 1 + (y - center.2) * 1 = 0 ∧
    center ∈ ({p : ℝ × ℝ | perpendicular_line p.1 p.2} : Set (ℝ × ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_perpendicular_line_l3648_364882


namespace NUMINAMATH_CALUDE_carpet_fit_l3648_364833

theorem carpet_fit (carpet_area : ℝ) (cut_length : ℝ) (room_area : ℝ) : 
  carpet_area = 169 →
  cut_length = 2 →
  room_area = (Real.sqrt carpet_area) * (Real.sqrt carpet_area - cut_length) →
  room_area = 143 := by
sorry

end NUMINAMATH_CALUDE_carpet_fit_l3648_364833


namespace NUMINAMATH_CALUDE_sam_puppies_l3648_364859

theorem sam_puppies (initial_puppies : Float) (given_away : Float) :
  initial_puppies = 6.0 → given_away = 2.0 → initial_puppies - given_away = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_puppies_l3648_364859


namespace NUMINAMATH_CALUDE_min_participants_in_race_l3648_364808

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race with its participants -/
structure Race where
  participants : List Participant

/-- Checks if the given race satisfies the conditions for Andrei -/
def satisfiesAndreiCondition (race : Race) : Prop :=
  ∃ (x : Nat), 3 * x + 1 = race.participants.length

/-- Checks if the given race satisfies the conditions for Dima -/
def satisfiesDimaCondition (race : Race) : Prop :=
  ∃ (y : Nat), 4 * y + 1 = race.participants.length

/-- Checks if the given race satisfies the conditions for Lenya -/
def satisfiesLenyaCondition (race : Race) : Prop :=
  ∃ (z : Nat), 5 * z + 1 = race.participants.length

/-- Checks if all participants have unique finishing positions -/
def uniqueFinishingPositions (race : Race) : Prop :=
  ∀ p1 p2 : Participant, p1 ∈ race.participants → p2 ∈ race.participants → 
    p1 ≠ p2 → p1.position ≠ p2.position

/-- The main theorem stating the minimum number of participants -/
theorem min_participants_in_race : 
  ∀ race : Race, 
    uniqueFinishingPositions race →
    satisfiesAndreiCondition race →
    satisfiesDimaCondition race →
    satisfiesLenyaCondition race →
    race.participants.length ≥ 61 :=
by
  sorry

end NUMINAMATH_CALUDE_min_participants_in_race_l3648_364808


namespace NUMINAMATH_CALUDE_quadrangular_pyramid_edge_length_l3648_364801

/-- A quadrangular pyramid with equal edge lengths -/
structure QuadrangularPyramid where
  edge_length : ℝ
  sum_of_edges : ℝ
  edge_sum_eq : sum_of_edges = 8 * edge_length

/-- Theorem: In a quadrangular pyramid with equal edge lengths, 
    if the sum of edge lengths is 14.8 meters, then each edge is 1.85 meters long -/
theorem quadrangular_pyramid_edge_length 
  (pyramid : QuadrangularPyramid) 
  (h : pyramid.sum_of_edges = 14.8) : 
  pyramid.edge_length = 1.85 := by
  sorry

#check quadrangular_pyramid_edge_length

end NUMINAMATH_CALUDE_quadrangular_pyramid_edge_length_l3648_364801


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l3648_364823

/-- The original number of houses in Lincoln County -/
def original_houses : ℕ := 20817

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 97741

/-- The current number of houses in Lincoln County -/
def current_houses : ℕ := 118558

/-- Theorem stating that the original number of houses plus the houses built
    during the boom equals the current number of houses -/
theorem lincoln_county_houses :
  original_houses + houses_built = current_houses := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l3648_364823


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3648_364861

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3648_364861


namespace NUMINAMATH_CALUDE_initial_tagged_fish_l3648_364878

/-- The number of fish initially caught and tagged -/
def T : ℕ := sorry

/-- The total number of fish in the pond -/
def N : ℕ := 800

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 40

/-- The number of tagged fish in the second catch -/
def tagged_in_second : ℕ := 2

theorem initial_tagged_fish :
  (T : ℚ) / N = tagged_in_second / second_catch ∧ T = 40 := by sorry

end NUMINAMATH_CALUDE_initial_tagged_fish_l3648_364878


namespace NUMINAMATH_CALUDE_triangle_properties_l3648_364802

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.c * Real.sin t.C - t.a * Real.sin t.A = (Real.sqrt 3 * t.c - t.b) * Real.sin t.B) :
  -- Part 1: Angle A is 30 degrees (π/6 radians)
  t.A = π / 6 ∧
  -- Part 2: If a = 1, the maximum area is (2 + √3) / 4
  (t.a = 1 → 
    ∃ (S : ℝ), S = (2 + Real.sqrt 3) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3648_364802


namespace NUMINAMATH_CALUDE_intersection_A_and_naturals_l3648_364872

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_A_and_naturals :
  A ∩ Set.univ.image (Nat.cast : ℕ → ℝ) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_and_naturals_l3648_364872


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l3648_364870

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x - (x + 1)^2

theorem extreme_value_and_range :
  (∃ x : ℝ, ∀ y : ℝ, f (-1) y ≤ f (-1) x ∧ f (-1) x = 1 / Real.exp 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≤ 0) ↔ a ∈ Set.Icc 0 (4 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l3648_364870


namespace NUMINAMATH_CALUDE_scientific_notation_6500_l3648_364827

theorem scientific_notation_6500 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 6500 = a * (10 : ℝ) ^ n ∧ a = 6.5 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_6500_l3648_364827


namespace NUMINAMATH_CALUDE_fixed_stable_points_range_l3648_364845

/-- The function f(x) = a x^2 - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

/-- The set of fixed points of f -/
def fixedPoints (a : ℝ) : Set ℝ := {x | f a x = x}

/-- The set of stable points of f -/
def stablePoints (a : ℝ) : Set ℝ := {x | f a (f a x) = x}

/-- Theorem stating the range of a for which the fixed points and stable points are equal and non-empty -/
theorem fixed_stable_points_range (a : ℝ) :
  (fixedPoints a = stablePoints a ∧ (fixedPoints a).Nonempty) ↔ -1/4 ≤ a ∧ a ≤ 3/4 :=
sorry

end NUMINAMATH_CALUDE_fixed_stable_points_range_l3648_364845


namespace NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3648_364873

theorem divide_eight_by_repeating_third (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3648_364873


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3648_364864

theorem polynomial_factorization (a b c : ℝ) :
  2*a*(b - c)^3 + 3*b*(c - a)^3 + 2*c*(a - b)^3 = (a - b)*(b - c)*(c - a)*(5*b - c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3648_364864


namespace NUMINAMATH_CALUDE_range_of_a_l3648_364849

theorem range_of_a (A B : Set ℝ) (a : ℝ) 
  (h1 : A = {x : ℝ | x ≤ 2})
  (h2 : B = {x : ℝ | x ≥ a})
  (h3 : A ⊆ B) : 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3648_364849


namespace NUMINAMATH_CALUDE_fraction_simplification_l3648_364844

theorem fraction_simplification :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3648_364844


namespace NUMINAMATH_CALUDE_hall_reunion_attendance_l3648_364803

theorem hall_reunion_attendance (total : ℕ) (oates : ℕ) (both : ℕ) (hall : ℕ) : 
  total = 150 → oates = 70 → both = 28 → total = oates + hall - both → hall = 108 := by
  sorry

end NUMINAMATH_CALUDE_hall_reunion_attendance_l3648_364803


namespace NUMINAMATH_CALUDE_nick_hid_ten_chocolates_l3648_364850

/-- The number of chocolates Nick hid -/
def nick_chocolates : ℕ := sorry

/-- The number of chocolates Alix hid initially -/
def alix_initial_chocolates : ℕ := 3 * nick_chocolates

/-- The number of chocolates Alix has after mom took 5 -/
def alix_current_chocolates : ℕ := alix_initial_chocolates - 5

theorem nick_hid_ten_chocolates : 
  alix_current_chocolates = nick_chocolates + 15 → nick_chocolates = 10 := by
  sorry

end NUMINAMATH_CALUDE_nick_hid_ten_chocolates_l3648_364850


namespace NUMINAMATH_CALUDE_original_number_problem_l3648_364825

theorem original_number_problem (x : ℝ) : 3 * (2 * x + 5) = 111 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l3648_364825


namespace NUMINAMATH_CALUDE_unit_circle_sector_angle_l3648_364876

/-- In a unit circle, a sector with area 1 has a central angle of 2 radians -/
theorem unit_circle_sector_angle (r : ℝ) (area : ℝ) (angle : ℝ) :
  r = 1 → area = 1 → angle = 2 * area / r → angle = 2 :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_sector_angle_l3648_364876


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3648_364897

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (r : ℕ), r < d ∧ (n - r) % d = 0 ∧ ∀ (s : ℕ), s < r → (n - s) % d ≠ 0 := by
  sorry

theorem problem_solution :
  ∃ (r : ℕ), r = 43 ∧ (62575 - r) % 99 = 0 ∧ ∀ (s : ℕ), s < r → (62575 - s) % 99 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3648_364897


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l3648_364800

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  additional_cubes : ℕ

/-- Calculates the surface area of the CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ :=
  2 * (solid.length * solid.height + solid.additional_cubes) + -- front and back
  (solid.length * solid.width + (solid.length * solid.width - solid.additional_cubes)) + -- top and bottom
  2 * (solid.width * solid.height) -- left and right

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid :=
  { length := 4
    width := 3
    height := 1
    additional_cubes := 2 }

theorem problem_solid_surface_area :
  surface_area problem_solid = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l3648_364800


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l3648_364810

/-- A cubic polynomial with rational coefficients -/
def cubic_polynomial (a b c : ℚ) (x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

theorem integer_root_of_cubic (a b c : ℚ) :
  (∃ (r : ℤ), cubic_polynomial a b c r = 0) →
  cubic_polynomial a b c (3 - Real.sqrt 5) = 0 →
  ∃ (r : ℤ), cubic_polynomial a b c r = 0 ∧ r = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l3648_364810


namespace NUMINAMATH_CALUDE_algebraic_expression_properties_l3648_364857

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^5 + b * x^3 + 3 * x + c

theorem algebraic_expression_properties :
  (f 0 = -1) →
  (f 1 = -1) →
  (f 3 = -10) →
  (c = -1 ∧ a + b + c = -4 ∧ f (-3) = 8) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_properties_l3648_364857


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_b_16_minus_1_l3648_364895

-- Define b as a natural number
def b : ℕ := 2

-- Define the function for the number of distinct prime factors
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

-- Define the function for the largest prime factor
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_b_16_minus_1 :
  num_distinct_prime_factors (b^16 - 1) = 4 →
  largest_prime_factor (b^16 - 1) = 257 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_b_16_minus_1_l3648_364895


namespace NUMINAMATH_CALUDE_head_start_value_l3648_364835

/-- A race between two runners A and B -/
structure Race where
  length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧ r.speed_ratio = 2 ∧ r.head_start > 0

/-- Both runners finish at the same time -/
def equal_finish_time (r : Race) : Prop :=
  r.length / r.speed_ratio = (r.length - r.head_start) / 1

theorem head_start_value (r : Race) 
  (h1 : race_conditions r) 
  (h2 : equal_finish_time r) : 
  r.head_start = 50 := by
  sorry

#check head_start_value

end NUMINAMATH_CALUDE_head_start_value_l3648_364835


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisector_square_area_l3648_364866

/-- Given a right triangle where the bisector of the right angle cuts the hypotenuse
    into segments of lengths a and b, the area of the square whose side is this bisector
    is equal to 2a²b² / (a² + b²). -/
theorem right_triangle_angle_bisector_square_area
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let bisector_length := Real.sqrt (2 * a^2 * b^2 / (a^2 + b^2))
  (bisector_length)^2 = 2 * a^2 * b^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_bisector_square_area_l3648_364866


namespace NUMINAMATH_CALUDE_triangle_area_from_square_areas_l3648_364865

theorem triangle_area_from_square_areas (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_square_areas_l3648_364865


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l3648_364862

theorem cab_driver_average_income 
  (incomes : List ℝ) 
  (h_incomes : incomes = [400, 250, 650, 400, 500]) 
  (h_days : incomes.length = 5) : 
  (incomes.sum / incomes.length : ℝ) = 440 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l3648_364862


namespace NUMINAMATH_CALUDE_least_three_digit_product_8_l3648_364899

-- Define a function to check if a number is three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define a function to get the product of digits
def digitProduct (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

-- State the theorem
theorem least_three_digit_product_8 :
  ∀ n : ℕ, isThreeDigit n → digitProduct n = 8 → n ≥ 118 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_product_8_l3648_364899


namespace NUMINAMATH_CALUDE_four_line_theorem_l3648_364892

/-- A line in a plane -/
structure Line where
  -- Add necessary fields here
  
/-- A point in a plane -/
structure Point where
  -- Add necessary fields here

/-- A circle in a plane -/
structure Circle where
  -- Add necessary fields here

/-- The set of four lines in the plane -/
def FourLines : Type := Fin 4 → Line

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- Predicate to check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Get the intersection point of two lines -/
def intersection (l1 l2 : Line) : Point := sorry

/-- Get the circumcircle of three points -/
def circumcircle (p1 p2 p3 : Point) : Circle := sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem four_line_theorem (lines : FourLines) :
  (∀ i j, i ≠ j → ¬are_parallel (lines i) (lines j)) →
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬are_concurrent (lines i) (lines j) (lines k)) →
  ∃ p : Point, ∀ i j k l,
    i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    point_on_circle p (circumcircle 
      (intersection (lines i) (lines j))
      (intersection (lines j) (lines k))
      (intersection (lines k) (lines i))) :=
sorry

end NUMINAMATH_CALUDE_four_line_theorem_l3648_364892


namespace NUMINAMATH_CALUDE_ellipse_locus_theorem_l3648_364834

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define point B
def point_B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define a point P on the ellipse
def point_P (a b x y : ℝ) : Prop :=
  ellipse a b x y ∧ (x, y) ≠ point_A a ∧ (x, y) ≠ point_B a

-- Define the locus of M
def locus_M (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (a^2 / b)^2 = 1

-- Theorem statement
theorem ellipse_locus_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∀ x y : ℝ, point_P a b x y → ∃ m_x m_y : ℝ, locus_M a b m_x m_y :=
sorry

end NUMINAMATH_CALUDE_ellipse_locus_theorem_l3648_364834


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3648_364871

theorem divisible_by_nine (k : ℕ+) : 
  (9 : ℤ) ∣ (3 * (2 + 7^(k : ℕ))) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3648_364871


namespace NUMINAMATH_CALUDE_percentage_decrease_l3648_364863

theorem percentage_decrease (x y z : ℝ) : 
  x = 1.3 * y ∧ x = 0.65 * z → y = 0.5 * z :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l3648_364863


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3648_364893

/-- Given two regular polygons with equal perimeters, where one has 50 sides
    and its side length is three times the other's, prove that the number of
    sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure positive side length
  50 * (3 * s) = n * s →  -- Equal perimeters
  n = 150 := by
  sorry


end NUMINAMATH_CALUDE_second_polygon_sides_l3648_364893


namespace NUMINAMATH_CALUDE_ellipse_and_dot_product_range_l3648_364896

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 = 1

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

-- Define the dot product of OA and OB
def dot_product (xa ya xb yb : ℝ) : ℝ := xa * xb + ya * yb

theorem ellipse_and_dot_product_range :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 →
  (∀ x y, ellipse a b x y → x^2 / a^2 + y^2 / b^2 = 1) →
  a^2 / b^2 - 1 = 1/4 →
  (∃ x, hyperbola 0 x) →
  (∀ m : ℝ, m ≠ 0 → ∃ xa ya xb yb,
    line m xa ya ∧ line m xb yb ∧
    ellipse a b xa ya ∧ ellipse a b xb yb) →
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ xa ya xb yb,
    ellipse a b xa ya ∧ ellipse a b xb yb →
    -4 ≤ dot_product xa ya xb yb ∧ dot_product xa ya xb yb < 13/4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_dot_product_range_l3648_364896


namespace NUMINAMATH_CALUDE_tournament_equation_l3648_364841

/-- Represents a single round-robin tournament --/
structure Tournament where
  teams : ℕ
  games : ℕ

/-- The number of games in a single round-robin tournament --/
def Tournament.gameCount (t : Tournament) : ℕ :=
  t.teams * (t.teams - 1) / 2

/-- Theorem: In a single round-robin tournament with x teams and 28 games, 
    the equation (1/2)x(x-1) = 28 holds --/
theorem tournament_equation (t : Tournament) (h : t.games = 28) : 
  t.gameCount = 28 := by
  sorry

end NUMINAMATH_CALUDE_tournament_equation_l3648_364841


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_optimal_plan_is_valid_l3648_364806

/-- Represents the rental plan for buses -/
structure RentalPlan where
  large_buses : ℕ
  small_buses : ℕ

/-- Calculates the total number of seats for a given rental plan -/
def total_seats (plan : RentalPlan) : ℕ :=
  plan.large_buses * 45 + plan.small_buses * 30

/-- Calculates the total cost for a given rental plan -/
def total_cost (plan : RentalPlan) : ℕ :=
  plan.large_buses * 400 + plan.small_buses * 300

/-- Checks if a rental plan is valid according to the given conditions -/
def is_valid_plan (plan : RentalPlan) : Prop :=
  total_seats plan ≥ 240 ∧ 
  plan.large_buses + plan.small_buses ≤ 6 ∧
  total_cost plan ≤ 2300

/-- The theorem stating that the most cost-effective valid plan is 4 large buses and 2 small buses -/
theorem most_cost_effective_plan :
  ∀ (plan : RentalPlan),
    is_valid_plan plan →
    total_cost plan ≥ total_cost { large_buses := 4, small_buses := 2 } :=
by sorry

/-- The theorem stating that the plan with 4 large buses and 2 small buses is valid -/
theorem optimal_plan_is_valid :
  is_valid_plan { large_buses := 4, small_buses := 2 } :=
by sorry

end NUMINAMATH_CALUDE_most_cost_effective_plan_optimal_plan_is_valid_l3648_364806


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3648_364894

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := { a := a, b := 2, c := 6 }
  let l2 : Line := { a := 1, b := a - 1, c := 3 }
  parallel l1 l2 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3648_364894


namespace NUMINAMATH_CALUDE_problem_statement_l3648_364818

/-- Given M = 6021 ÷ 4, N = 2M, and X = N - M + 500, prove that X = 3005.25 -/
theorem problem_statement (M N X : ℚ) 
  (hM : M = 6021 / 4)
  (hN : N = 2 * M)
  (hX : X = N - M + 500) :
  X = 3005.25 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3648_364818


namespace NUMINAMATH_CALUDE_edward_rides_l3648_364821

theorem edward_rides (total_tickets : ℕ) (spent_tickets : ℕ) (cost_per_ride : ℕ) : 
  total_tickets = 79 → spent_tickets = 23 → cost_per_ride = 7 →
  (total_tickets - spent_tickets) / cost_per_ride = 8 := by
sorry

end NUMINAMATH_CALUDE_edward_rides_l3648_364821


namespace NUMINAMATH_CALUDE_surface_dots_eq_105_l3648_364813

/-- Represents a standard die -/
structure Die where
  faces : Fin 6 → Nat
  sum_21 : (faces 0) + (faces 1) + (faces 2) + (faces 3) + (faces 4) + (faces 5) = 21

/-- Represents the solid made of glued dice -/
structure DiceSolid where
  dice : Fin 7 → Die
  glued_faces_same : ∀ (i j : Fin 7) (f1 f2 : Fin 6), 
    (dice i).faces f1 = (dice j).faces f2 → i ≠ j

def surface_dots (solid : DiceSolid) : Nat :=
  sorry

theorem surface_dots_eq_105 (solid : DiceSolid) : 
  surface_dots solid = 105 := by
  sorry

end NUMINAMATH_CALUDE_surface_dots_eq_105_l3648_364813


namespace NUMINAMATH_CALUDE_student_multiplication_error_l3648_364822

theorem student_multiplication_error (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  (78 : ℚ) * ((1 + (100 * a + 10 * b + c : ℚ) / 999) - (1 + (a / 10 + b / 100 + c / 1000))) = (3 / 5) →
  100 * a + 10 * b + c = 765 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_error_l3648_364822


namespace NUMINAMATH_CALUDE_g_of_three_equals_twentyone_l3648_364819

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_three_equals_twentyone :
  (∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) →
  g 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_equals_twentyone_l3648_364819


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3648_364879

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ m) ∧ 
  (∃ (c d : ℝ), (c + 5)^2 + (d - 12)^2 = 14^2 ∧ c^2 + d^2 = m) ∧
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3648_364879


namespace NUMINAMATH_CALUDE_five_black_cards_taken_out_l3648_364804

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (original_black_cards : ℕ)
  (remaining_black_cards : ℕ)

/-- Defines a standard deck with 52 total cards and 26 black cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    original_black_cards := 26,
    remaining_black_cards := 21 }

/-- Calculates the number of black cards taken out from a deck -/
def black_cards_taken_out (d : Deck) : ℕ :=
  d.original_black_cards - d.remaining_black_cards

/-- Theorem stating that 5 black cards were taken out from the standard deck -/
theorem five_black_cards_taken_out :
  black_cards_taken_out standard_deck = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_black_cards_taken_out_l3648_364804


namespace NUMINAMATH_CALUDE_cannot_transform_to_target_l3648_364874

/-- Represents a natural number with its digits. -/
structure DigitNumber where
  digits : List Nat
  first_nonzero : digits.head? ≠ some 0

/-- Represents the allowed operations on the number. -/
inductive Operation
  | multiply_by_five
  | rearrange_digits

/-- Defines the target 150-digit number 5222...2223. -/
def target_number : DigitNumber := {
  digits := 5 :: List.replicate 148 2 ++ [2, 3]
  first_nonzero := by simp
}

/-- Applies an operation to a DigitNumber. -/
def apply_operation (n : DigitNumber) (op : Operation) : DigitNumber :=
  sorry

/-- Checks if a DigitNumber can be transformed into the target number using the allowed operations. -/
def can_transform (n : DigitNumber) : Prop :=
  ∃ (ops : List Operation), (ops.foldl apply_operation n) = target_number

/-- The initial number 1. -/
def initial_number : DigitNumber := {
  digits := [1]
  first_nonzero := by simp
}

/-- The main theorem stating that it's impossible to transform 1 into the target number. -/
theorem cannot_transform_to_target : ¬(can_transform initial_number) :=
  sorry

end NUMINAMATH_CALUDE_cannot_transform_to_target_l3648_364874


namespace NUMINAMATH_CALUDE_john_and_alice_money_sum_l3648_364848

theorem john_and_alice_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by sorry

end NUMINAMATH_CALUDE_john_and_alice_money_sum_l3648_364848


namespace NUMINAMATH_CALUDE_advertisement_arrangements_l3648_364837

theorem advertisement_arrangements : ℕ := by
  -- Define the total number of advertisements
  let total_ads : ℕ := 6
  -- Define the number of commercial advertisements
  let commercial_ads : ℕ := 4
  -- Define the number of public service advertisements
  let public_service_ads : ℕ := 2
  -- Define the condition that public service ads must be at the beginning and end
  let public_service_at_ends : Prop := true

  -- The theorem to prove
  have : (public_service_at_ends ∧ 
          total_ads = commercial_ads + public_service_ads) → 
         (Nat.factorial public_service_ads * Nat.factorial commercial_ads = 48) := by
    sorry

  -- The final statement
  exact 48

end NUMINAMATH_CALUDE_advertisement_arrangements_l3648_364837


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l3648_364869

/-- The quadratic inequality ax^2 + bx + c > 0 has all real numbers as its solution set
    if and only if a > 0 and the discriminant is negative. -/
theorem quadratic_inequality_all_reals 
  (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (a > 0 ∧ Δ < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l3648_364869


namespace NUMINAMATH_CALUDE_both_correct_calculation_l3648_364826

/-- Represents a class test scenario -/
structure ClassTest where
  total : ℕ
  correct1 : ℕ
  correct2 : ℕ
  absent : ℕ

/-- Calculates the number of students who answered both questions correctly -/
def bothCorrect (test : ClassTest) : ℕ :=
  test.correct1 + test.correct2 - (test.total - test.absent)

/-- Theorem stating the number of students who answered both questions correctly -/
theorem both_correct_calculation (test : ClassTest) 
  (h1 : test.total = 25)
  (h2 : test.correct1 = 22)
  (h3 : test.correct2 = 20)
  (h4 : test.absent = 3) :
  bothCorrect test = 17 := by
  sorry

end NUMINAMATH_CALUDE_both_correct_calculation_l3648_364826


namespace NUMINAMATH_CALUDE_paula_twice_karl_age_l3648_364851

/-- Represents the ages of Paula and Karl -/
structure Ages where
  paula : ℕ
  karl : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.paula + ages.karl = 50 ∧
  ages.paula - 7 = 3 * (ages.karl - 7)

/-- The theorem to prove -/
theorem paula_twice_karl_age (ages : Ages) :
  problem_conditions ages →
  ∃ x : ℕ, x = 2 ∧ ages.paula + x = 2 * (ages.karl + x) :=
sorry

end NUMINAMATH_CALUDE_paula_twice_karl_age_l3648_364851


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l3648_364843

theorem midpoint_sum_equals_vertex_sum (a b c d : ℝ) :
  a + b + c + d = 15 →
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l3648_364843


namespace NUMINAMATH_CALUDE_shirts_made_over_two_days_l3648_364811

/-- Calculates the total number of shirts made by an industrial machine over two days -/
theorem shirts_made_over_two_days 
  (shirts_per_minute : ℕ) -- Number of shirts the machine can make per minute
  (minutes_worked_yesterday : ℕ) -- Number of minutes the machine worked yesterday
  (shirts_made_today : ℕ) -- Number of shirts made today
  (h1 : shirts_per_minute = 6)
  (h2 : minutes_worked_yesterday = 12)
  (h3 : shirts_made_today = 14) :
  shirts_per_minute * minutes_worked_yesterday + shirts_made_today = 86 :=
by
  sorry

#check shirts_made_over_two_days

end NUMINAMATH_CALUDE_shirts_made_over_two_days_l3648_364811


namespace NUMINAMATH_CALUDE_labourer_fine_problem_l3648_364830

/-- Calculates the fine per day of absence for a labourer --/
def calculate_fine_per_day (total_days : ℕ) (daily_wage : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_wage
  (total_earned - total_received) / days_absent

/-- Theorem stating the fine per day of absence for the given problem --/
theorem labourer_fine_problem :
  calculate_fine_per_day 25 2 (37 + 1/2) 5 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_labourer_fine_problem_l3648_364830


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3648_364809

def is_valid (n : ℕ) : Prop :=
  ∀ m : ℕ+, ∃ S : Finset ℕ, S ⊆ Finset.range n ∧ 
    (S.prod id : ℕ) ≡ m [ZMOD 100]

theorem smallest_valid_n :
  is_valid 17 ∧ ∀ k < 17, ¬ is_valid k :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3648_364809


namespace NUMINAMATH_CALUDE_ariels_age_ariels_current_age_l3648_364807

theorem ariels_age (birth_year : Nat) (fencing_start_year : Nat) (years_fencing : Nat) : Nat :=
  let current_year := fencing_start_year + years_fencing
  let age := current_year - birth_year
  age

theorem ariels_current_age : 
  ariels_age 1992 2006 16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ariels_age_ariels_current_age_l3648_364807


namespace NUMINAMATH_CALUDE_train_length_l3648_364858

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 210 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 310 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3648_364858


namespace NUMINAMATH_CALUDE_bobs_mile_time_l3648_364886

/-- Bob's mile run time problem -/
theorem bobs_mile_time (sister_time : ℝ) (improvement_percent : ℝ) (bob_time : ℝ) : 
  sister_time = 9 * 60 + 42 →
  improvement_percent = 9.062499999999996 →
  bob_time = sister_time * (1 + improvement_percent / 100) →
  bob_time = 634.5 := by
  sorry

end NUMINAMATH_CALUDE_bobs_mile_time_l3648_364886


namespace NUMINAMATH_CALUDE_polynomial_value_l3648_364847

theorem polynomial_value (a : ℝ) (h : a^2 + 3*a = 2) : 2*a^2 + 6*a - 10 = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3648_364847


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_divisible_by_others_l3648_364881

theorem not_divisible_by_seven (n : ℤ) : ¬(7 ∣ (n^2225 - n^2005)) :=
sorry

theorem divisible_by_others (n : ℤ) : 
  (3 ∣ (n^2225 - n^2005)) ∧ 
  (5 ∣ (n^2225 - n^2005)) ∧ 
  (11 ∣ (n^2225 - n^2005)) ∧ 
  (23 ∣ (n^2225 - n^2005)) :=
sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_divisible_by_others_l3648_364881


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3648_364875

theorem absolute_value_equation (x z : ℝ) : 
  |3*x - 2*Real.log z| = 3*x + 2*Real.log z → x = 0 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3648_364875


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l3648_364885

theorem polynomial_coefficient_sums (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 129) ∧
  (a₁ + a₃ + a₅ + a₇ = 8256) ∧
  (a₀ + a₂ + a₄ + a₆ = -8128) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 16384) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l3648_364885


namespace NUMINAMATH_CALUDE_cos_42_cos_78_minus_sin_42_sin_78_l3648_364898

theorem cos_42_cos_78_minus_sin_42_sin_78 :
  Real.cos (42 * π / 180) * Real.cos (78 * π / 180) -
  Real.sin (42 * π / 180) * Real.sin (78 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_42_cos_78_minus_sin_42_sin_78_l3648_364898


namespace NUMINAMATH_CALUDE_vincent_train_books_l3648_364868

theorem vincent_train_books (animal_books : ℕ) (space_books : ℕ) (book_cost : ℕ) (total_spent : ℕ) :
  animal_books = 10 →
  space_books = 1 →
  book_cost = 16 →
  total_spent = 224 →
  ∃ (train_books : ℕ), train_books = 3 ∧ total_spent = book_cost * (animal_books + space_books + train_books) :=
by sorry

end NUMINAMATH_CALUDE_vincent_train_books_l3648_364868


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3648_364817

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3648_364817


namespace NUMINAMATH_CALUDE_sum_15_terms_l3648_364824

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- Sum of the first 5 terms -/
  sum5 : ℝ
  /-- Sum of the first 10 terms -/
  sum10 : ℝ
  /-- The sequence is arithmetic -/
  is_arithmetic : True
  /-- The sum of the first 5 terms is 10 -/
  sum5_eq_10 : sum5 = 10
  /-- The sum of the first 10 terms is 50 -/
  sum10_eq_50 : sum10 = 50

/-- Theorem: The sum of the first 15 terms is 120 -/
theorem sum_15_terms (seq : ArithmeticSequence) : ∃ (sum15 : ℝ), sum15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_15_terms_l3648_364824


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3648_364891

theorem factorial_equation_solution : 
  ∃ (n : ℕ), n > 0 ∧ (Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 1320) ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3648_364891


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l3648_364855

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 15 →
  (n * original_mean - n * decrement) / n = 185 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l3648_364855


namespace NUMINAMATH_CALUDE_sum_of_squares_unique_l3648_364884

theorem sum_of_squares_unique (p q r : ℕ+) : 
  p + q + r = 33 → 
  Nat.gcd p.val q.val + Nat.gcd q.val r.val + Nat.gcd r.val p.val = 11 → 
  p^2 + q^2 + r^2 = 419 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_unique_l3648_364884


namespace NUMINAMATH_CALUDE_angle_equality_l3648_364889

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ + 2 * Real.sin θ) : 
  θ = 10 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3648_364889


namespace NUMINAMATH_CALUDE_no_prime_solution_l3648_364854

theorem no_prime_solution : ¬∃ p : ℕ, Nat.Prime p ∧ 2 * p^3 - p^2 - 16 * p + 26 = 0 := by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3648_364854


namespace NUMINAMATH_CALUDE_range_of_a_l3648_364842

theorem range_of_a (z : ℂ) (a : ℝ) : 
  z.im ≠ 0 →  -- z is imaginary
  (z + 3 / (2 * z)).im = 0 →  -- z + 3/(2z) is real
  (z + 3 / (2 * z))^2 - 2 * a * (z + 3 / (2 * z)) + 1 - 3 * a = 0 →  -- root condition
  (a ≥ (Real.sqrt 13 - 3) / 2 ∨ a ≤ -(Real.sqrt 13 + 3) / 2) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3648_364842


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3648_364805

theorem arithmetic_evaluation : 2 * (5 - 2) - 5^2 = -19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3648_364805


namespace NUMINAMATH_CALUDE_geometric_progression_constant_l3648_364836

theorem geometric_progression_constant (x : ℝ) : 
  (((30 + x) ^ 2 = (10 + x) * (90 + x)) ↔ x = 0) ∧
  (∀ y : ℝ, ((30 + y) ^ 2 = (10 + y) * (90 + y)) → y = 0) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_constant_l3648_364836


namespace NUMINAMATH_CALUDE_blue_tshirts_count_l3648_364877

/-- Calculates the number of blue t-shirts in each pack -/
def blue_tshirts_per_pack (white_packs : ℕ) (blue_packs : ℕ) (white_per_pack : ℕ) (total_tshirts : ℕ) : ℕ :=
  let white_total := white_packs * white_per_pack
  let blue_total := total_tshirts - white_total
  blue_total / blue_packs

/-- Proves that the number of blue t-shirts in each pack is 9 -/
theorem blue_tshirts_count : blue_tshirts_per_pack 5 3 6 57 = 9 := by
  sorry

end NUMINAMATH_CALUDE_blue_tshirts_count_l3648_364877


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l3648_364840

def g (x : ℝ) : ℝ := -3 * x^3 + 12

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
  sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l3648_364840


namespace NUMINAMATH_CALUDE_smallest_x_value_solution_exists_l3648_364832

theorem smallest_x_value (x : ℝ) : 
  (x^2 - x - 72) / (x - 9) = 3 / (x + 6) → x ≥ -9 :=
by
  sorry

theorem solution_exists : 
  ∃ x : ℝ, (x^2 - x - 72) / (x - 9) = 3 / (x + 6) ∧ x = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_solution_exists_l3648_364832


namespace NUMINAMATH_CALUDE_divisible_by_sixteen_l3648_364838

theorem divisible_by_sixteen (n : ℕ) : ∃ k : ℤ, (2*n - 1)^3 - (2*n)^2 + 2*n + 1 = 16 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_sixteen_l3648_364838


namespace NUMINAMATH_CALUDE_pyramid_volume_l3648_364887

/-- The volume of a pyramid with a rectangular base and a slant edge perpendicular to two adjacent sides of the base. -/
theorem pyramid_volume (base_length base_width slant_edge : ℝ) 
  (hl : base_length = 10) 
  (hw : base_width = 6) 
  (hs : slant_edge = 20) : 
  (1 / 3 : ℝ) * base_length * base_width * Real.sqrt (slant_edge^2 - base_length^2) = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3648_364887


namespace NUMINAMATH_CALUDE_car_demand_and_profit_l3648_364814

-- Define the total demand function
def R (x : ℕ) : ℚ := (1/2) * x * (x + 1) * (39 - 2*x)

-- Define the purchase price function
def W (x : ℕ) : ℚ := 150000 + 2000*x

-- Define the constraints
def valid_x (x : ℕ) : Prop := x > 0 ∧ x ≤ 6

-- Define the demand function
def g (x : ℕ) : ℚ := -3*x^2 + 40*x

-- Define the monthly profit function
def f (x : ℕ) : ℚ := (185000 - W x) * g x

theorem car_demand_and_profit 
  (h : ∀ x, valid_x x → R x - R (x-1) = g x) :
  (∀ x, valid_x x → g x = -3*x^2 + 40*x) ∧ 
  (∀ x, valid_x x → f x ≤ f 5) ∧
  (f 5 = 3125000) := by
  sorry


end NUMINAMATH_CALUDE_car_demand_and_profit_l3648_364814


namespace NUMINAMATH_CALUDE_range_of_a_l3648_364883

theorem range_of_a (p q : Prop) (h_p : ∀ x ∈ Set.Icc 1 2, 2 * x^2 - a ≥ 0)
  (h_q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) (h_pq : p ∧ q) :
  a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3648_364883


namespace NUMINAMATH_CALUDE_time_for_A_to_reach_B_l3648_364880

-- Define the total distance between points A and B
variable (S : ℝ) 

-- Define the speeds of A and B
variable (v_A v_B : ℝ)

-- Define the time when B catches up to A for the first time
variable (t : ℝ)

-- Theorem statement
theorem time_for_A_to_reach_B 
  (h1 : v_A * (t + 48/60) = v_B * t) 
  (h2 : v_A * (t + 48/60) = 2/3 * S) 
  (h3 : v_A * (t + 48/60 + 1/2 * t + 6/60) + 6/60 * v_B = S) 
  : (108 : ℝ) - (96 : ℝ) = 12 := by
  sorry


end NUMINAMATH_CALUDE_time_for_A_to_reach_B_l3648_364880


namespace NUMINAMATH_CALUDE_beggars_and_mothers_attitude_l3648_364820

structure Neighborhood where
  has_nearby_railway : Bool
  has_frequent_beggars : Bool

structure Mother where
  treats_beggars_equally : Bool
  provides_newspapers : Bool
  father_helped_in_depression : Bool

def reason_for_beggars_visits (n : Neighborhood) : Bool :=
  n.has_nearby_railway

def mother_treatment_of_beggars (m : Mother) : Bool :=
  m.treats_beggars_equally

def purpose_of_newspapers (m : Mother) : Bool :=
  m.provides_newspapers

def explanation_for_mothers_attitude (m : Mother) : Bool :=
  m.father_helped_in_depression

theorem beggars_and_mothers_attitude 
  (n : Neighborhood) 
  (m : Mother) 
  (h1 : n.has_nearby_railway = true)
  (h2 : n.has_frequent_beggars = true)
  (h3 : m.treats_beggars_equally = true)
  (h4 : m.provides_newspapers = true)
  (h5 : m.father_helped_in_depression = true) :
  reason_for_beggars_visits n = true ∧
  mother_treatment_of_beggars m = true ∧
  purpose_of_newspapers m = true ∧
  explanation_for_mothers_attitude m = true := by
  sorry

end NUMINAMATH_CALUDE_beggars_and_mothers_attitude_l3648_364820


namespace NUMINAMATH_CALUDE_rainfall_problem_l3648_364831

/-- Rainfall problem statement -/
theorem rainfall_problem (day1 day2 day3 normal_avg this_year_total : ℝ) 
  (h1 : day1 = 26)
  (h2 : day3 = day2 - 12)
  (h3 : normal_avg = 140)
  (h4 : this_year_total = normal_avg - 58)
  (h5 : this_year_total = day1 + day2 + day3) :
  day2 = 34 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l3648_364831


namespace NUMINAMATH_CALUDE_non_negative_for_all_non_negative_exists_l3648_364846

-- Define the function f
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Theorem for part (1)
theorem non_negative_for_all (m : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f x m ≥ 0) ↔ m ≥ 1 :=
sorry

-- Theorem for part (2)
theorem non_negative_exists (m : ℝ) :
  (∃ x ∈ Set.Icc 0 3, f x m ≥ 0) ↔ m ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_non_negative_for_all_non_negative_exists_l3648_364846


namespace NUMINAMATH_CALUDE_total_score_l3648_364812

theorem total_score (darius_score marius_score matt_score : ℕ) : 
  darius_score = 10 →
  marius_score = darius_score + 3 →
  matt_score = darius_score + 5 →
  darius_score + marius_score + matt_score = 38 := by
sorry

end NUMINAMATH_CALUDE_total_score_l3648_364812


namespace NUMINAMATH_CALUDE_pauls_lawn_mowing_earnings_l3648_364853

/-- 
Given that:
1. Paul's total money is the sum of money from mowing lawns and $28 from weed eating
2. Paul spends $9 per week
3. Paul's money lasts for 8 weeks

Prove that Paul made $44 mowing lawns.
-/
theorem pauls_lawn_mowing_earnings :
  ∀ (M : ℕ), -- M represents the amount Paul made mowing lawns
  (M + 28 = 9 * 8) → -- Total money equals weekly spending times number of weeks
  M = 44 := by
sorry

end NUMINAMATH_CALUDE_pauls_lawn_mowing_earnings_l3648_364853


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l3648_364856

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l3648_364856
