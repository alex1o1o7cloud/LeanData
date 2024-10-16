import Mathlib

namespace NUMINAMATH_CALUDE_water_heater_theorem_l2927_292701

/-- Calculates the total amount of water in two water heaters -/
def total_water (wallace_capacity : ℚ) : ℚ :=
  let catherine_capacity := wallace_capacity / 2
  let wallace_water := (3 / 4) * wallace_capacity
  let catherine_water := (3 / 4) * catherine_capacity
  wallace_water + catherine_water

/-- Theorem stating that given the conditions, the total water is 45 gallons -/
theorem water_heater_theorem (wallace_capacity : ℚ) 
  (h1 : wallace_capacity = 40) : total_water wallace_capacity = 45 := by
  sorry

#eval total_water 40

end NUMINAMATH_CALUDE_water_heater_theorem_l2927_292701


namespace NUMINAMATH_CALUDE_cards_given_to_miguel_l2927_292781

/-- Represents the card distribution problem --/
def card_distribution (total_cards : ℕ) (kept_cards : ℕ) (friends : ℕ) (cards_per_friend : ℕ) 
  (sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  let remaining_after_keeping := total_cards - kept_cards
  let given_to_friends := friends * cards_per_friend
  let remaining_after_friends := remaining_after_keeping - given_to_friends
  let given_to_sisters := sisters * cards_per_sister
  remaining_after_friends - given_to_sisters

/-- Theorem stating the number of cards Rick gave to Miguel --/
theorem cards_given_to_miguel : 
  card_distribution 250 25 12 15 4 7 = 17 := by
  sorry


end NUMINAMATH_CALUDE_cards_given_to_miguel_l2927_292781


namespace NUMINAMATH_CALUDE_min_box_height_l2927_292793

theorem min_box_height (x : ℝ) (h : x > 0) : 
  (10 * x^2 ≥ 150) → (∀ y : ℝ, y > 0 → 10 * y^2 ≥ 150 → y ≥ x) → 2 * x = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_min_box_height_l2927_292793


namespace NUMINAMATH_CALUDE_candidates_count_l2927_292791

theorem candidates_count (average_marks : ℝ) (total_marks : ℝ) (h1 : average_marks = 35) (h2 : total_marks = 4200) :
  total_marks / average_marks = 120 := by
  sorry

end NUMINAMATH_CALUDE_candidates_count_l2927_292791


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_4_l2927_292740

theorem binomial_coefficient_16_4 : Nat.choose 16 4 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_4_l2927_292740


namespace NUMINAMATH_CALUDE_difference_zero_point_eight_and_one_eighth_l2927_292702

theorem difference_zero_point_eight_and_one_eighth : (0.8 : ℝ) - (1 / 8 : ℝ) = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_difference_zero_point_eight_and_one_eighth_l2927_292702


namespace NUMINAMATH_CALUDE_points_on_line_relationship_l2927_292725

/-- Given a line y = -3x + b and three points on this line, prove that y₁ > y₂ > y₃ -/
theorem points_on_line_relationship (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -3 * (-2) + b)
  (h2 : y₂ = -3 * (-1) + b)
  (h3 : y₃ = -3 * 1 + b) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_relationship_l2927_292725


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2927_292779

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a + 1/b + 1/c ≥ 9) ∧ 
  (1/a + 1/b + 1/c = 9 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2927_292779


namespace NUMINAMATH_CALUDE_area_scaled_and_shifted_l2927_292708

-- Define a function g: ℝ → ℝ
variable (g : ℝ → ℝ)

-- Define the area between a function and the x-axis
def area_between_curve_and_axis (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_scaled_and_shifted (h : area_between_curve_and_axis g = 15) :
  area_between_curve_and_axis (fun x ↦ 4 * g (x + 3)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_scaled_and_shifted_l2927_292708


namespace NUMINAMATH_CALUDE_random_sampling_appropriate_for_air_quality_l2927_292717

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| RandomSampling

/-- Represents a scenario for which a survey method is chosen -/
inductive Scenario
| LightBulbLifespan
| FoodPreservatives
| SpaceEquipmentQuality
| AirQuality

/-- Determines if a survey method is appropriate for a given scenario -/
def isAppropriate (method : SurveyMethod) (scenario : Scenario) : Prop :=
  match scenario with
  | Scenario.LightBulbLifespan => method = SurveyMethod.RandomSampling
  | Scenario.FoodPreservatives => method = SurveyMethod.RandomSampling
  | Scenario.SpaceEquipmentQuality => method = SurveyMethod.Comprehensive
  | Scenario.AirQuality => method = SurveyMethod.RandomSampling

/-- Theorem stating that random sampling is appropriate for air quality measurement -/
theorem random_sampling_appropriate_for_air_quality :
  isAppropriate SurveyMethod.RandomSampling Scenario.AirQuality :=
by
  sorry

#check random_sampling_appropriate_for_air_quality

end NUMINAMATH_CALUDE_random_sampling_appropriate_for_air_quality_l2927_292717


namespace NUMINAMATH_CALUDE_discount_markup_percentage_l2927_292771

theorem discount_markup_percentage (original_price : ℝ) (discount_rate : ℝ) (h1 : discount_rate = 0.2) :
  let discounted_price := original_price * (1 - discount_rate)
  let markup_rate := (original_price - discounted_price) / discounted_price
  markup_rate = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_discount_markup_percentage_l2927_292771


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l2927_292703

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l2927_292703


namespace NUMINAMATH_CALUDE_total_balls_is_135_l2927_292768

/-- Represents a school with elementary and middle school classes -/
structure School where
  elementary : Nat
  middle : Nat

/-- Calculates the total number of soccer balls donated to all schools -/
def totalSoccerBalls (schools : List School) (ballsPerClass : Nat) : Nat :=
  schools.foldl (fun acc school => acc + (school.elementary + school.middle) * ballsPerClass) 0

/-- Theorem: The total number of soccer balls donated is 135 -/
theorem total_balls_is_135 (schoolA schoolB schoolC : School) (ballsPerClass : Nat) :
  schoolA.elementary = 4 →
  schoolA.middle = 5 →
  schoolB.elementary = 5 →
  schoolB.middle = 3 →
  schoolC.elementary = 6 →
  schoolC.middle = 4 →
  ballsPerClass = 5 →
  totalSoccerBalls [schoolA, schoolB, schoolC] ballsPerClass = 135 := by
  sorry


end NUMINAMATH_CALUDE_total_balls_is_135_l2927_292768


namespace NUMINAMATH_CALUDE_plate_distance_l2927_292730

/-- Given a square table with a circular plate, prove that the distance from the bottom edge
    of the table to the plate is 53 cm, given the distances from other edges. -/
theorem plate_distance (left_distance right_distance top_distance : ℝ) :
  left_distance = 10 →
  right_distance = 63 →
  top_distance = 20 →
  ∃ (plate_diameter bottom_distance : ℝ),
    left_distance + plate_diameter + right_distance = top_distance + plate_diameter + bottom_distance ∧
    bottom_distance = 53 :=
by sorry

end NUMINAMATH_CALUDE_plate_distance_l2927_292730


namespace NUMINAMATH_CALUDE_remaining_amount_l2927_292757

def initial_amount : ℕ := 20
def peach_quantity : ℕ := 3
def peach_price : ℕ := 2

theorem remaining_amount : 
  initial_amount - (peach_quantity * peach_price) = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_l2927_292757


namespace NUMINAMATH_CALUDE_distinct_triangles_in_3x3_grid_l2927_292772

/-- The number of points in a row or column of the grid -/
def gridSize : Nat := 3

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of ways to choose 3 points from the total points -/
def totalCombinations : Nat := Nat.choose totalPoints 3

/-- The number of sets of collinear points in the grid -/
def collinearSets : Nat := 2 * gridSize + 2

/-- The number of distinct triangles in a 3x3 grid -/
def distinctTriangles : Nat := totalCombinations - collinearSets

theorem distinct_triangles_in_3x3_grid :
  distinctTriangles = 76 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_3x3_grid_l2927_292772


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l2927_292754

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 1050) :
  210 = Nat.gcd q r ∧ ∀ x : ℕ, x < 210 → x ≠ Nat.gcd q r :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l2927_292754


namespace NUMINAMATH_CALUDE_february_2013_days_l2927_292737

/-- A function that determines if a given year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 = 0

/-- A function that returns the number of days in February for a given year -/
def daysInFebruary (year : ℕ) : ℕ :=
  if isLeapYear year then 29 else 28

/-- Theorem stating that February in 2013 has 28 days -/
theorem february_2013_days : daysInFebruary 2013 = 28 := by
  sorry

#eval daysInFebruary 2013

end NUMINAMATH_CALUDE_february_2013_days_l2927_292737


namespace NUMINAMATH_CALUDE_fixed_points_of_quadratic_l2927_292777

/-- The quadratic function f(x) always passes through two fixed points -/
theorem fixed_points_of_quadratic (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + (3*a - 1)*x - (10*a + 3)
  (f 2 = -5 ∧ f (-5) = 2) := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_quadratic_l2927_292777


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l2927_292752

theorem right_triangle_arithmetic_sequence (a b c : ℝ) (area : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a < b → b < c →
  c - b = b - a →
  a^2 + b^2 = c^2 →
  area = (1/2) * a * b →
  area = 1350 →
  (a, b, c) = (45, 60, 75) := by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l2927_292752


namespace NUMINAMATH_CALUDE_is_center_of_ellipse_l2927_292712

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * x * y + y^2 + 2 * x + 2 * y - 4 = 0

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (0, -1)

/-- Theorem stating that the given point is the center of the ellipse -/
theorem is_center_of_ellipse :
  ∀ (x y : ℝ), ellipse_equation x y →
  ellipse_center = (0, -1) := by sorry

end NUMINAMATH_CALUDE_is_center_of_ellipse_l2927_292712


namespace NUMINAMATH_CALUDE_football_team_yardage_l2927_292719

theorem football_team_yardage (initial_loss : ℤ) (gain : ℤ) (final_progress : ℤ) : 
  gain = 9 ∧ final_progress = 4 → initial_loss = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_yardage_l2927_292719


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2927_292705

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (3 + m) * (-3) + 4 * 3 - 3 + 3 * m = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2927_292705


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2927_292741

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 2)^2 - 34*(a 2) + 64 = 0 →
  (a 6)^2 - 34*(a 6) + 64 = 0 →
  a 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2927_292741


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_12_l2927_292739

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ+) : ℕ+ := sorry

theorem eleventh_number_with_digit_sum_12 :
  nth_number_with_digit_sum_12 11 = 147 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_12_l2927_292739


namespace NUMINAMATH_CALUDE_number_of_boys_l2927_292755

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 300 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 75 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l2927_292755


namespace NUMINAMATH_CALUDE_cost_per_friend_is_1650_l2927_292716

/-- The cost per friend when buying erasers and pencils -/
def cost_per_friend (eraser_count : ℕ) (eraser_cost : ℕ) (pencil_count : ℕ) (pencil_cost : ℕ) (friend_count : ℕ) : ℚ :=
  ((eraser_count * eraser_cost + pencil_count * pencil_cost) : ℚ) / friend_count

/-- Theorem: The cost per friend for the given scenario is 1650 won -/
theorem cost_per_friend_is_1650 :
  cost_per_friend 5 200 7 800 4 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_friend_is_1650_l2927_292716


namespace NUMINAMATH_CALUDE_sum_squares_s_r_l2927_292747

def r : Finset Int := {-2, -1, 0, 1, 3}
def r_range : Finset Int := {-1, 0, 3, 4, 6}

def s_domain : Finset Int := {0, 1, 2, 3, 4, 5}
def s (x : Int) : Int := x^2 + x + 1

theorem sum_squares_s_r : 
  (r_range ∩ s_domain).sum (fun x => (s x)^2) = 611 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_s_r_l2927_292747


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2927_292720

/-- Discriminant of a quadratic equation ax² + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Condition for a quadratic equation to have two equal real roots -/
def has_two_equal_real_roots (a b c : ℝ) : Prop := discriminant a b c = 0

theorem quadratic_equal_roots :
  has_two_equal_real_roots 1 (-2) 1 ∧
  ¬has_two_equal_real_roots 1 (-3) 2 ∧
  ¬has_two_equal_real_roots 1 (-2) 3 ∧
  ¬has_two_equal_real_roots 1 0 (-9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2927_292720


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l2927_292785

def a (n : ℕ) : ℚ := n + 1 / (2^n)

theorem sequence_formula_correct : 
  (a 1 = 3/2) ∧ (a 2 = 9/4) ∧ (a 3 = 25/8) ∧ (a 4 = 65/16) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l2927_292785


namespace NUMINAMATH_CALUDE_smallest_third_altitude_l2927_292790

/-- An isosceles triangle with integer altitudes -/
structure IsoscelesTriangle where
  -- The length of the equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The altitude to the equal sides
  altitude_to_equal_side : ℝ
  -- The altitude to the base
  altitude_to_base : ℝ
  -- Constraint: the triangle is isosceles
  isosceles : side > 0
  -- Constraint: altitudes are positive
  altitude_to_equal_side_pos : altitude_to_equal_side > 0
  altitude_to_base_pos : altitude_to_base > 0
  -- Constraint: altitudes are integers
  altitude_to_equal_side_int : ∃ n : ℤ, altitude_to_equal_side = n
  altitude_to_base_int : ∃ n : ℤ, altitude_to_base = n

/-- The theorem stating the smallest possible value for the third altitude -/
theorem smallest_third_altitude (t : IsoscelesTriangle) 
  (h1 : t.altitude_to_equal_side = 15)
  (h2 : t.altitude_to_base = 5) :
  ∃ h : ℝ, h ≥ 5 ∧ 
  (∀ h' : ℝ, (∃ n : ℤ, h' = n) → 
    (2 * t.side * t.base = t.altitude_to_equal_side * t.base + t.altitude_to_base * t.side) → 
    h' ≥ h) := by
  sorry

end NUMINAMATH_CALUDE_smallest_third_altitude_l2927_292790


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2927_292746

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a quadrant in the Cartesian plane -/
inductive Quadrant
  | I   -- x > 0, y > 0
  | II  -- x < 0, y > 0
  | III -- x < 0, y < 0
  | IV  -- x > 0, y < 0

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I   => ∃ x > 0, f.m * x + f.b > 0
  | Quadrant.II  => ∃ x < 0, f.m * x + f.b > 0
  | Quadrant.III => ∃ x < 0, f.m * x + f.b < 0
  | Quadrant.IV  => ∃ x > 0, f.m * x + f.b < 0

/-- The main theorem to prove -/
theorem linear_function_quadrants (f : LinearFunction) 
  (h1 : f.m = 4) 
  (h2 : f.b = 2) : 
  passesThrough f Quadrant.I ∧ 
  passesThrough f Quadrant.II ∧ 
  passesThrough f Quadrant.III :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l2927_292746


namespace NUMINAMATH_CALUDE_food_supply_duration_l2927_292764

/-- Proves that the initial food supply was planned to last for 22 days given the problem conditions -/
theorem food_supply_duration (initial_men : ℕ) (additional_men : ℕ) (remaining_days : ℕ) : 
  initial_men = 760 → 
  additional_men = 2280 → 
  remaining_days = 5 → 
  (initial_men * (22 - 2) : ℕ) = (initial_men + additional_men) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_food_supply_duration_l2927_292764


namespace NUMINAMATH_CALUDE_bricklayers_theorem_l2927_292710

/-- Represents the problem of two bricklayers building a wall --/
structure BricklayersProblem where
  total_bricks : ℕ
  time_first : ℕ
  time_second : ℕ
  joint_decrease : ℕ
  joint_time : ℕ

/-- The solution to the bricklayers problem --/
def solve_bricklayers_problem (p : BricklayersProblem) : Prop :=
  p.total_bricks = 288 ∧
  p.time_first = 8 ∧
  p.time_second = 12 ∧
  p.joint_decrease = 12 ∧
  p.joint_time = 6 ∧
  (p.total_bricks / p.time_first + p.total_bricks / p.time_second - p.joint_decrease) * p.joint_time = p.total_bricks

theorem bricklayers_theorem (p : BricklayersProblem) :
  solve_bricklayers_problem p :=
sorry

end NUMINAMATH_CALUDE_bricklayers_theorem_l2927_292710


namespace NUMINAMATH_CALUDE_box_volume_count_l2927_292774

theorem box_volume_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun x => x > 2 ∧ (x + 3) * (x - 2) * (x^2 + 10) < 500) 
    (Finset.range 100)).card :=
by
  sorry

end NUMINAMATH_CALUDE_box_volume_count_l2927_292774


namespace NUMINAMATH_CALUDE_new_average_after_joining_l2927_292727

theorem new_average_after_joining (initial_count : ℕ) (initial_average : ℚ) (new_member_amount : ℚ) :
  initial_count = 7 →
  initial_average = 14 →
  new_member_amount = 56 →
  (initial_count * initial_average + new_member_amount) / (initial_count + 1) = 19.25 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_joining_l2927_292727


namespace NUMINAMATH_CALUDE_total_climb_length_l2927_292773

def keaton_ladder_length : ℕ := 30
def keaton_climbs : ℕ := 20
def reece_ladder_difference : ℕ := 4
def reece_climbs : ℕ := 15
def inches_per_foot : ℕ := 12

theorem total_climb_length : 
  (keaton_ladder_length * keaton_climbs + 
   (keaton_ladder_length - reece_ladder_difference) * reece_climbs) * 
   inches_per_foot = 11880 := by
  sorry

end NUMINAMATH_CALUDE_total_climb_length_l2927_292773


namespace NUMINAMATH_CALUDE_speed_calculation_l2927_292786

/-- Given a distance of 900 meters covered in 180 seconds, prove that the speed is 18 km/h -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 900) (h2 : time = 180) :
  (distance / 1000) / (time / 3600) = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l2927_292786


namespace NUMINAMATH_CALUDE_perfect_square_from_fraction_pairs_l2927_292796

theorem perfect_square_from_fraction_pairs (N : ℕ+) 
  (h : ∃! (pairs : Finset (ℕ+ × ℕ+)), pairs.card = 2005 ∧ 
    ∀ (x y : ℕ+), (x, y) ∈ pairs ↔ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / N) : 
  ∃ (k : ℕ+), N = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_fraction_pairs_l2927_292796


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2927_292728

/-- 
Given a quadratic equation (m-2)x^2 + 2x - 1 = 0, prove that for it to have real roots, 
m must be greater than or equal to 1.
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 + 2 * x - 1 = 0) → m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2927_292728


namespace NUMINAMATH_CALUDE_xy_value_l2927_292782

theorem xy_value (x y : ℝ) (h : x * (x + 2*y) = x^2 + 10) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2927_292782


namespace NUMINAMATH_CALUDE_intersection_points_min_distance_l2927_292770

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x + y + 1 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Theorem for intersection points
theorem intersection_points :
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y ↔ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1)) :=
sorry

-- Theorem for minimum distance
theorem min_distance :
  (∃ d : ℝ, d = Real.sqrt 2 - 1 ∧
    ∀ x₁ y₁ x₂ y₂ : ℝ, C₂ x₁ y₁ → C₃ x₂ y₂ →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_min_distance_l2927_292770


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2927_292715

-- Define the square
def Square (perimeter : ℝ) : Type :=
  { side : ℝ // perimeter = 4 * side }

-- Theorem statement
theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 80) :
  ∃ (s : Square perimeter), (s.val)^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2927_292715


namespace NUMINAMATH_CALUDE_max_B_bins_l2927_292792

/-- The cost of an A brand garbage bin in yuan -/
def cost_A : ℕ := 120

/-- The cost of a B brand garbage bin in yuan -/
def cost_B : ℕ := 150

/-- The total number of garbage bins to be purchased -/
def total_bins : ℕ := 30

/-- The maximum budget in yuan -/
def max_budget : ℕ := 4000

/-- Theorem stating the maximum number of B brand bins that can be purchased -/
theorem max_B_bins : 
  ∀ m : ℕ, 
  m ≤ total_bins ∧ 
  cost_B * m + cost_A * (total_bins - m) ≤ max_budget →
  m ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_max_B_bins_l2927_292792


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l2927_292706

/-- Represents a hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The hyperbola passes through a given point -/
def passes_through (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- The asymptotes of the hyperbola -/
def has_asymptotes (h : Hyperbola) (f g : ℝ → ℝ) : Prop :=
  ∀ x, (h.equation x (f x) ∨ h.equation x (g x))

/-- The foci of the hyperbola -/
def foci (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance from a point to a line -/
def distance_to_line (x y : ℝ) (m b : ℝ) : ℝ := sorry

theorem hyperbola_theorem (h : Hyperbola) 
  (center_origin : h.equation 0 0)
  (asymptotes : has_asymptotes h (λ x => Real.sqrt 3 * x) (λ x => -Real.sqrt 3 * x))
  (point : passes_through h (Real.sqrt 2) (Real.sqrt 3)) :
  (∀ x y, h.equation x y ↔ x^2 - y^2/3 = 1) ∧ 
  (let (fx, fy) := foci h
   distance_to_line fx fy (Real.sqrt 3) 0 = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l2927_292706


namespace NUMINAMATH_CALUDE_total_passengers_l2927_292745

theorem total_passengers (on_time : ℕ) (late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) :
  on_time + late = 14720 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_l2927_292745


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2927_292763

theorem polynomial_factorization (y : ℝ) : 
  y^4 - 4*y^2 + 4 + 49*y^2 = (y^2 + 1) * (y^2 + 13) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2927_292763


namespace NUMINAMATH_CALUDE_common_point_exists_l2927_292788

-- Define the basic structures
structure Ray where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the given conditions
def intersect_point : ℝ × ℝ := sorry

def ray1 : Ray := sorry
def ray2 : Ray := sorry

def a : ℝ := sorry
axiom a_positive : 0 < a

-- Define the circle properties
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop := sorry

def circle_intersects_ray (c : Circle) (r : Ray) : ℝ × ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem common_point_exists :
  ∀ (c : Circle),
    circle_passes_through c intersect_point ∧
    ∃ (B C : ℝ × ℝ),
      B = circle_intersects_ray c ray1 ∧
      C = circle_intersects_ray c ray2 ∧
      distance intersect_point B + distance intersect_point C = a
    →
    ∃ (Z : ℝ × ℝ), Z ≠ intersect_point ∧
      ∀ (c' : Circle),
        circle_passes_through c' intersect_point ∧
        ∃ (B' C' : ℝ × ℝ),
          B' = circle_intersects_ray c' ray1 ∧
          C' = circle_intersects_ray c' ray2 ∧
          distance intersect_point B' + distance intersect_point C' = a
        →
        circle_passes_through c' Z :=
sorry

end NUMINAMATH_CALUDE_common_point_exists_l2927_292788


namespace NUMINAMATH_CALUDE_expression_simplification_l2927_292769

theorem expression_simplification (x : ℝ) : 7*x + 15 - 3*x + 2 = 4*x + 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2927_292769


namespace NUMINAMATH_CALUDE_min_max_sum_of_a_l2927_292732

theorem min_max_sum_of_a (a b c : ℝ) (sum_eq : a + b + c = 5) (sum_sq_eq : a^2 + b^2 + c^2 = 8) :
  ∃ (m M : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8) → m ≤ x ∧ x ≤ M) ∧ m + M = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_of_a_l2927_292732


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_x_minus_sqrt3_power10_l2927_292743

/-- The binomial coefficient of the third term in the expansion of (x - √3)^10 is 45 -/
theorem binomial_coefficient_third_term_x_minus_sqrt3_power10 :
  Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_x_minus_sqrt3_power10_l2927_292743


namespace NUMINAMATH_CALUDE_terese_tuesday_run_l2927_292778

-- Define the days Terese runs
inductive RunDay
| monday
| tuesday
| wednesday
| thursday

-- Define a function that returns the distance run on each day
def distance_run (day : RunDay) : Real :=
  match day with
  | RunDay.monday => 4.2
  | RunDay.wednesday => 3.6
  | RunDay.thursday => 4.4
  | RunDay.tuesday => 3.8  -- This is what we want to prove

-- Define the average distance
def average_distance : Real := 4

-- Define the number of days Terese runs
def num_run_days : Nat := 4

-- Theorem statement
theorem terese_tuesday_run :
  (distance_run RunDay.monday +
   distance_run RunDay.tuesday +
   distance_run RunDay.wednesday +
   distance_run RunDay.thursday) / num_run_days = average_distance :=
by
  sorry


end NUMINAMATH_CALUDE_terese_tuesday_run_l2927_292778


namespace NUMINAMATH_CALUDE_bucket_capacity_l2927_292780

/-- Calculates the capacity of a bucket used to fill a pool -/
theorem bucket_capacity
  (fill_time : ℕ)           -- Time to fill and empty one bucket (in seconds)
  (pool_capacity : ℕ)       -- Capacity of the pool (in gallons)
  (total_time : ℕ)          -- Total time to fill the pool (in minutes)
  (h1 : fill_time = 20)     -- Given: Time to fill and empty one bucket is 20 seconds
  (h2 : pool_capacity = 84) -- Given: Pool capacity is 84 gallons
  (h3 : total_time = 14)    -- Given: Total time to fill the pool is 14 minutes
  : ℕ := by
  sorry

#check bucket_capacity

end NUMINAMATH_CALUDE_bucket_capacity_l2927_292780


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2927_292753

theorem cyclic_sum_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a * b + b * c + c * a = a * b * c) : 
  (a^4 + b^4) / (a * b * (a^3 + b^3)) + 
  (b^4 + c^4) / (b * c * (b^3 + c^3)) + 
  (c^4 + a^4) / (c * a * (c^3 + a^3)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2927_292753


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2927_292787

theorem triangle_angle_measure (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) 
  (h7 : A + B + C = π) (h8 : Real.sqrt 3 / Real.sin A = 1 / Real.sin (π/6)) (h9 : B = π/6) : 
  A = π/3 ∨ A = 2*π/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2927_292787


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2927_292789

theorem unique_solution_for_equation :
  ∀ x y : ℕ+,
    x > y →
    (x - y : ℕ+) ^ (x * y : ℕ) = x ^ (y : ℕ) * y ^ (x : ℕ) →
    x = 4 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2927_292789


namespace NUMINAMATH_CALUDE_no_integer_coordinate_equilateral_triangle_l2927_292756

theorem no_integer_coordinate_equilateral_triangle :
  ¬ ∃ (A B C : ℤ × ℤ), 
    (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧ 
    (B.1 ≠ C.1 ∨ B.2 ≠ C.2) ∧ 
    (C.1 ≠ A.1 ∨ C.2 ≠ A.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 := by
  sorry


end NUMINAMATH_CALUDE_no_integer_coordinate_equilateral_triangle_l2927_292756


namespace NUMINAMATH_CALUDE_factoring_theorem_l2927_292742

theorem factoring_theorem (x : ℝ) : x^2 * (x + 3) + 2 * (x + 3) + (x + 3) = (x^2 + 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_theorem_l2927_292742


namespace NUMINAMATH_CALUDE_f_properties_l2927_292775

def f (x : ℝ) := x^3 - 12*x

theorem f_properties :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧ 
  (∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y) ∧ 
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ f (-2)) ∧
  (∀ x, f 2 ≤ f x) ∧
  (f (-2) = 16) ∧
  (f 2 = -16) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2927_292775


namespace NUMINAMATH_CALUDE_eve_hit_ten_l2927_292711

-- Define the set of possible scores
def ScoreSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a type for players
inductive Player : Type
| Alex | Becca | Carli | Dan | Eve | Fiona

-- Define a function that returns a player's score
def player_score : Player → ℕ
| Player.Alex => 20
| Player.Becca => 5
| Player.Carli => 13
| Player.Dan => 15
| Player.Eve => 21
| Player.Fiona => 6

-- Define a function that returns a pair of scores for a player
def player_throws (p : Player) : ℕ × ℕ := sorry

-- State the theorem
theorem eve_hit_ten :
  ∀ (p : Player),
    (∀ (q : Player), p ≠ q → player_throws p ≠ player_throws q) ∧
    (∀ (p : Player), (player_throws p).1 ∈ ScoreSet ∧ (player_throws p).2 ∈ ScoreSet) ∧
    (∀ (p : Player), (player_throws p).1 + (player_throws p).2 = player_score p) →
    (player_throws Player.Eve).1 = 10 ∨ (player_throws Player.Eve).2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_eve_hit_ten_l2927_292711


namespace NUMINAMATH_CALUDE_matching_allocation_theorem_l2927_292709

/-- Represents the allocation of workers to produce parts A and B -/
structure WorkerAllocation where
  partA : ℕ
  partB : ℕ

/-- Checks if the given allocation produces matching sets of parts A and B -/
def isMatchingAllocation (totalWorkers : ℕ) (prodRateA : ℕ) (prodRateB : ℕ) (allocation : WorkerAllocation) : Prop :=
  allocation.partA + allocation.partB = totalWorkers ∧
  prodRateB * allocation.partB = 2 * (prodRateA * allocation.partA)

/-- Theorem stating that the given allocation produces matching sets -/
theorem matching_allocation_theorem :
  let totalWorkers : ℕ := 50
  let prodRateA : ℕ := 40
  let prodRateB : ℕ := 120
  let allocation : WorkerAllocation := ⟨30, 20⟩
  isMatchingAllocation totalWorkers prodRateA prodRateB allocation := by
  sorry

#check matching_allocation_theorem

end NUMINAMATH_CALUDE_matching_allocation_theorem_l2927_292709


namespace NUMINAMATH_CALUDE_discount_difference_l2927_292744

def original_price : ℚ := 30
def flat_discount : ℚ := 5
def percent_discount : ℚ := 0.25

def price_flat_then_percent : ℚ := (original_price - flat_discount) * (1 - percent_discount)
def price_percent_then_flat : ℚ := (original_price * (1 - percent_discount)) - flat_discount

theorem discount_difference :
  (price_flat_then_percent - price_percent_then_flat) * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l2927_292744


namespace NUMINAMATH_CALUDE_abs_eq_solution_l2927_292723

theorem abs_eq_solution (x : ℝ) : |x + 1| = 2*x + 4 ↔ x = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_solution_l2927_292723


namespace NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_independent_l2927_292795

-- Define the sample space
def S : Set (ℕ × ℕ) := {p | p.1 ∈ Finset.range 6 ∧ p.2 ∈ Finset.range 6}

-- Define events A, B, and C
def A : Set (ℕ × ℕ) := {p ∈ S | p.1 + p.2 = 7}
def B : Set (ℕ × ℕ) := {p ∈ S | Odd (p.1 * p.2)}
def C : Set (ℕ × ℕ) := {p ∈ S | p.1 > 3}

-- Define probability measure
noncomputable def P : Set (ℕ × ℕ) → ℝ := sorry

-- Theorem statements
theorem A_B_mutually_exclusive : A ∩ B = ∅ := by sorry

theorem A_C_independent : P (A ∩ C) = P A * P C := by sorry

end NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_independent_l2927_292795


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2927_292759

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : ∃ R : ℝ,
  R > 0 ∧ R < 100 ∧ (P * R * 10) / 100 = P / 5 ∧ R = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2927_292759


namespace NUMINAMATH_CALUDE_product_of_roots_l2927_292783

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃)) → 
  ∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = 50 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2927_292783


namespace NUMINAMATH_CALUDE_twenty_seven_binary_l2927_292733

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number in base 2 -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem twenty_seven_binary :
  toBinary 27 = [true, true, false, true, true] :=
sorry

end NUMINAMATH_CALUDE_twenty_seven_binary_l2927_292733


namespace NUMINAMATH_CALUDE_center_square_side_length_l2927_292721

theorem center_square_side_length : 
  let large_square_side : ℝ := 120
  let total_area : ℝ := large_square_side ^ 2
  let l_shape_area : ℝ := (1 / 5) * total_area
  let center_square_area : ℝ := total_area - 4 * l_shape_area
  let center_square_side : ℝ := Real.sqrt center_square_area
  center_square_side = 54 := by
  sorry

end NUMINAMATH_CALUDE_center_square_side_length_l2927_292721


namespace NUMINAMATH_CALUDE_stock_percentage_problem_l2927_292799

/-- Calculates the percentage of a stock given income, investment, and stock price. -/
def stock_percentage (income : ℚ) (investment : ℚ) (stock_price : ℚ) : ℚ :=
  (income * stock_price) / investment

/-- Theorem stating that given the specific values in the problem, the stock percentage is 30%. -/
theorem stock_percentage_problem :
  let income : ℚ := 500
  let investment : ℚ := 1500
  let stock_price : ℚ := 90
  stock_percentage income investment stock_price = 30 := by sorry

end NUMINAMATH_CALUDE_stock_percentage_problem_l2927_292799


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2927_292735

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2927_292735


namespace NUMINAMATH_CALUDE_square_sum_geq_two_l2927_292738

theorem square_sum_geq_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_two_l2927_292738


namespace NUMINAMATH_CALUDE_age_difference_is_27_l2927_292765

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h : tens < 10 ∧ ones < 10)

def Age.value (a : Age) : Nat := 10 * a.tens + a.ones

def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.h.symm⟩

theorem age_difference_is_27 (alan_age bob_age : Age) : 
  (alan_age.reverse = bob_age) →
  (bob_age.value = alan_age.value / 2 + 6) →
  (alan_age.value + 2 = 5 * (bob_age.value - 4)) →
  (alan_age.value - bob_age.value = 27) :=
sorry

end NUMINAMATH_CALUDE_age_difference_is_27_l2927_292765


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l2927_292761

theorem sum_and_equal_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 150)
  (equal_numbers : a - 3 = b + 4 ∧ b + 4 = 4 * c) : 
  a = 631 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l2927_292761


namespace NUMINAMATH_CALUDE_greatest_four_digit_satisfying_conditions_l2927_292794

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_satisfying_conditions :
  is_four_digit 9999 ∧
  ¬(product_of_first_n 9999 % sum_of_first_n 9999 = 0) ∧
  is_perfect_square (9999 + 1) ∧
  ∀ n : ℕ, is_four_digit n →
    n > 9999 ∨
    (product_of_first_n n % sum_of_first_n n = 0) ∨
    ¬(is_perfect_square (n + 1)) :=
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_satisfying_conditions_l2927_292794


namespace NUMINAMATH_CALUDE_calculate_expression_l2927_292766

theorem calculate_expression : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2927_292766


namespace NUMINAMATH_CALUDE_vanaspati_percentage_l2927_292797

/-- Proves that the percentage of vanaspati in the original ghee mixture is 40% -/
theorem vanaspati_percentage
  (original_quantity : ℝ)
  (pure_ghee_percentage : ℝ)
  (added_pure_ghee : ℝ)
  (new_vanaspati_percentage : ℝ)
  (h1 : original_quantity = 10)
  (h2 : pure_ghee_percentage = 0.6)
  (h3 : added_pure_ghee = 10)
  (h4 : new_vanaspati_percentage = 0.2)
  (h5 : (1 - pure_ghee_percentage) * original_quantity = 
        new_vanaspati_percentage * (original_quantity + added_pure_ghee)) :
  (1 - pure_ghee_percentage) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_vanaspati_percentage_l2927_292797


namespace NUMINAMATH_CALUDE_price_and_distance_proportions_l2927_292748

-- Define the relationships
def inverse_proportion (x y : ℝ) (k : ℝ) : Prop := x * y = k
def direct_proportion (x y : ℝ) (k : ℝ) : Prop := x / y = k

-- State the theorem
theorem price_and_distance_proportions :
  -- For any positive real numbers representing unit price, quantity, and total price
  ∀ (unit_price quantity total_price : ℝ) (hp : unit_price > 0) (hq : quantity > 0) (ht : total_price > 0),
  -- When the total price is fixed
  (unit_price * quantity = total_price) →
  -- The unit price and quantity are in inverse proportion
  inverse_proportion unit_price quantity total_price ∧
  -- For any positive real numbers representing map distance, actual distance, and scale
  ∀ (map_distance actual_distance scale : ℝ) (hm : map_distance > 0) (ha : actual_distance > 0) (hs : scale > 0),
  -- When the scale is fixed
  (map_distance / actual_distance = scale) →
  -- The map distance and actual distance are in direct proportion
  direct_proportion map_distance actual_distance scale :=
by sorry

end NUMINAMATH_CALUDE_price_and_distance_proportions_l2927_292748


namespace NUMINAMATH_CALUDE_total_cups_on_table_l2927_292731

theorem total_cups_on_table (juice_cups milk_cups : ℕ) 
  (h1 : juice_cups = 3) 
  (h2 : milk_cups = 4) : 
  juice_cups + milk_cups = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_on_table_l2927_292731


namespace NUMINAMATH_CALUDE_village_foods_tomato_sales_l2927_292734

theorem village_foods_tomato_sales (customers : ℕ) (lettuce_per_customer : ℕ) 
  (lettuce_price : ℚ) (tomato_price : ℚ) (total_sales : ℚ) 
  (h1 : customers = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomato_price = 1/2)
  (h5 : total_sales = 2000) :
  (total_sales - (↑customers * ↑lettuce_per_customer * lettuce_price)) / (↑customers * tomato_price) = 4 := by
sorry

end NUMINAMATH_CALUDE_village_foods_tomato_sales_l2927_292734


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2927_292776

/-- The length of the diagonal of a rectangular prism with dimensions 12, 16, and 21 -/
def prism_diagonal : ℝ := 29

/-- Theorem: The diagonal of a rectangular prism with dimensions 12, 16, and 21 is 29 -/
theorem rectangular_prism_diagonal :
  let a : ℝ := 12
  let b : ℝ := 16
  let c : ℝ := 21
  Real.sqrt (a^2 + b^2 + c^2) = prism_diagonal :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2927_292776


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l2927_292749

def Rectangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def vertices : Rectangle := ((1, 1), (1, 5), (6, 5), (6, 1))

def length (r : Rectangle) : ℝ := 
  let ((x1, _), (_, _), (x2, _), _) := r
  |x2 - x1|

def width (r : Rectangle) : ℝ := 
  let ((_, y1), (_, y2), _, _) := r
  |y2 - y1|

def perimeter (r : Rectangle) : ℝ := 2 * (length r + width r)

def area (r : Rectangle) : ℝ := length r * width r

theorem rectangle_perimeter_area_sum :
  perimeter vertices + area vertices = 38 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l2927_292749


namespace NUMINAMATH_CALUDE_coexistent_pair_properties_l2927_292714

/-- Definition of coexistent rational number pairs -/
def is_coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

theorem coexistent_pair_properties :
  /- (1) -/
  is_coexistent_pair 3 (1/2) ∧
  /- (2) -/
  (∀ m n : ℚ, is_coexistent_pair m n → is_coexistent_pair (-n) (-m)) ∧
  /- (3) -/
  is_coexistent_pair 4 (3/5) ∧
  /- (4) -/
  (∀ a : ℚ, is_coexistent_pair a 3 → a = -2) :=
by sorry

end NUMINAMATH_CALUDE_coexistent_pair_properties_l2927_292714


namespace NUMINAMATH_CALUDE_some_number_solution_l2927_292767

theorem some_number_solution : 
  ∃ x : ℝ, 45 - (28 - (x - (15 - 15))) = 54 ∧ x = 37 := by sorry

end NUMINAMATH_CALUDE_some_number_solution_l2927_292767


namespace NUMINAMATH_CALUDE_kayak_production_sum_l2927_292762

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem kayak_production_sum :
  let a := 5  -- Initial production in February
  let r := 3  -- Growth ratio
  let n := 6  -- Number of months (February to July)
  geometric_sum a r n = 1820 := by
sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l2927_292762


namespace NUMINAMATH_CALUDE_largest_value_l2927_292707

theorem largest_value : 
  ∀ (a b c d : ℤ), a = 2^3 ∧ b = -3^2 ∧ c = (-3)^2 ∧ d = (-2)^3 →
  (c ≥ a ∧ c ≥ b ∧ c ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l2927_292707


namespace NUMINAMATH_CALUDE_video_game_players_l2927_292729

/-- The number of players who quit the game -/
def players_quit : ℕ := 7

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 8

/-- The total number of lives after players quit -/
def total_lives : ℕ := 72

/-- The initial number of players in the game -/
def initial_players : ℕ := 16

theorem video_game_players :
  initial_players = players_quit + total_lives / lives_per_player :=
by sorry

end NUMINAMATH_CALUDE_video_game_players_l2927_292729


namespace NUMINAMATH_CALUDE_percentage_increase_l2927_292751

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 60 → new = 150 → (new - original) / original * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2927_292751


namespace NUMINAMATH_CALUDE_checker_in_center_l2927_292722

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a checker placement on the board -/
structure Placement :=
  (board : Board)
  (positions : Finset (ℕ × ℕ))

/-- Defines symmetry with respect to both main diagonals -/
def is_symmetric (p : Placement) : Prop :=
  ∀ (i j : ℕ), (i, j) ∈ p.positions ↔
    (j, i) ∈ p.positions ∧
    (p.board.size - 1 - i, p.board.size - 1 - j) ∈ p.positions

/-- The central cell of the board -/
def central_cell (b : Board) : ℕ × ℕ :=
  (b.size / 2, b.size / 2)

/-- The main theorem -/
theorem checker_in_center (p : Placement)
  (h_size : p.board.size = 25)
  (h_count : p.positions.card = 25)
  (h_sym : is_symmetric p) :
  central_cell p.board ∈ p.positions :=
sorry

end NUMINAMATH_CALUDE_checker_in_center_l2927_292722


namespace NUMINAMATH_CALUDE_crate_height_determination_l2927_292784

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ

/-- Theorem stating the height of the crate given the conditions -/
theorem crate_height_determination 
  (crate : CrateDimensions) 
  (tank : GasTank) 
  (h1 : crate.length = 12)
  (h2 : crate.width = 16)
  (h3 : tank.radius = 8)
  (h4 : tank.radius * 2 ≤ min crate.length crate.width) :
  crate.height = 16 := by
  sorry

end NUMINAMATH_CALUDE_crate_height_determination_l2927_292784


namespace NUMINAMATH_CALUDE_kennel_dogs_l2927_292726

theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 3 / 4 →
  cats = dogs - 8 →
  dogs = 32 := by
sorry

end NUMINAMATH_CALUDE_kennel_dogs_l2927_292726


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l2927_292750

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l2927_292750


namespace NUMINAMATH_CALUDE_number_satisfies_condition_l2927_292736

theorem number_satisfies_condition : ∃ n : ℤ, n + 2 = 3 * n - 8 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_condition_l2927_292736


namespace NUMINAMATH_CALUDE_product_PQRS_l2927_292760

theorem product_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 := by
  sorry

end NUMINAMATH_CALUDE_product_PQRS_l2927_292760


namespace NUMINAMATH_CALUDE_puzzle_solution_l2927_292700

theorem puzzle_solution :
  ∃ (g n o u w : Nat),
    g ∈ Finset.range 10 ∧
    n ∈ Finset.range 10 ∧
    o ∈ Finset.range 10 ∧
    u ∈ Finset.range 10 ∧
    w ∈ Finset.range 10 ∧
    (100 * g + 10 * u + n) ^ 2 = 100000 * w + 10000 * o + 1000 * w + 100 * g + 10 * u + n ∧
    o - w = 3 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2927_292700


namespace NUMINAMATH_CALUDE_henry_walk_distance_l2927_292798

/-- Represents a 2D point --/
structure Point where
  x : Float
  y : Float

/-- Calculates the distance between two points --/
def distance (p1 p2 : Point) : Float :=
  Float.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Converts meters to feet --/
def metersToFeet (meters : Float) : Float :=
  meters * 3.281

theorem henry_walk_distance : 
  let start := Point.mk 0 0
  let end_point := Point.mk 40 (-(metersToFeet 15 + 48))
  Float.abs (distance start end_point - 105.1) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_henry_walk_distance_l2927_292798


namespace NUMINAMATH_CALUDE_emily_spending_l2927_292758

theorem emily_spending (x : ℝ) : 
  x + 2*x + 3*x = 120 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_spending_l2927_292758


namespace NUMINAMATH_CALUDE_three_dozen_quarters_value_l2927_292718

/-- Proves that 3 dozen quarters is equal to $9 --/
theorem three_dozen_quarters_value : 
  let dozen : ℕ := 12
  let quarter_value : ℕ := 25  -- in cents
  let cents_per_dollar : ℕ := 100
  (3 * dozen * quarter_value) / cents_per_dollar = 9 := by
  sorry

end NUMINAMATH_CALUDE_three_dozen_quarters_value_l2927_292718


namespace NUMINAMATH_CALUDE_orvin_balloon_purchase_l2927_292724

def regular_price : ℕ := 4
def initial_balloons : ℕ := 35
def discount_ratio : ℚ := 1/2

def max_balloons : ℕ := 42

theorem orvin_balloon_purchase :
  let total_money := initial_balloons * regular_price
  let discounted_set_cost := 2 * regular_price + discount_ratio * regular_price
  let num_sets := total_money / discounted_set_cost
  num_sets * 3 = max_balloons :=
by sorry

end NUMINAMATH_CALUDE_orvin_balloon_purchase_l2927_292724


namespace NUMINAMATH_CALUDE_garden_area_theorem_l2927_292713

/-- Represents a rectangular garden with given properties -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  perimeter_walk_count : ℕ
  length_walk_count : ℕ
  total_distance : ℝ

/-- The theorem stating the area of the garden given the conditions -/
theorem garden_area_theorem (g : RectangularGarden) 
  (h1 : g.perimeter_walk_count = 20)
  (h2 : g.length_walk_count = 50)
  (h3 : g.total_distance = 1500)
  (h4 : 2 * (g.length + g.width) = g.total_distance / g.perimeter_walk_count)
  (h5 : g.length = g.total_distance / g.length_walk_count) :
  g.length * g.width = 225 := by
  sorry

#check garden_area_theorem

end NUMINAMATH_CALUDE_garden_area_theorem_l2927_292713


namespace NUMINAMATH_CALUDE_area_triangle_OAB_l2927_292704

/-- Given a polar coordinate system with pole O, point A(1, π/6), and point B(2, π/2),
    the area of triangle OAB is √3/2. -/
theorem area_triangle_OAB :
  let r₁ : ℝ := 1
  let θ₁ : ℝ := π / 6
  let r₂ : ℝ := 2
  let θ₂ : ℝ := π / 2
  let area := (1 / 2) * r₁ * r₂ * Real.sin (θ₂ - θ₁)
  area = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_OAB_l2927_292704
