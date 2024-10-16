import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_expressions_l1495_149556

theorem arithmetic_mean_expressions (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x + a) / y + (y - b) / x) / 2 = (x^2 + a*x + y^2 - b*y) / (2*x*y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_expressions_l1495_149556


namespace NUMINAMATH_CALUDE_xiaogangMathScore_l1495_149595

theorem xiaogangMathScore (chineseScore englishScore averageScore : ℕ) (mathScore : ℕ) :
  chineseScore = 88 →
  englishScore = 91 →
  averageScore = 90 →
  (chineseScore + mathScore + englishScore) / 3 = averageScore →
  mathScore = 91 := by
  sorry

end NUMINAMATH_CALUDE_xiaogangMathScore_l1495_149595


namespace NUMINAMATH_CALUDE_jeffreys_steps_calculation_l1495_149549

-- Define the number of steps for Andrew and Jeffrey
def andrews_steps : ℕ := 150
def jeffreys_steps : ℕ := 200

-- Define the ratio of Andrew's steps to Jeffrey's steps
def step_ratio : ℚ := 3 / 4

-- Theorem statement
theorem jeffreys_steps_calculation :
  andrews_steps * 4 = jeffreys_steps * 3 :=
by sorry

end NUMINAMATH_CALUDE_jeffreys_steps_calculation_l1495_149549


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1495_149532

theorem smallest_positive_solution (x : ℕ) : 
  (∃ k : ℤ, 45 * x + 15 = 5 + 28 * k) ∧ 
  (∀ y : ℕ, y < x → ¬(∃ k : ℤ, 45 * y + 15 = 5 + 28 * k)) → 
  x = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1495_149532


namespace NUMINAMATH_CALUDE_common_roots_solution_l1495_149558

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (c d : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧
    p^3 + c*p^2 + 7*p + 4 = 0 ∧
    p^3 + d*p^2 + 10*p + 6 = 0 ∧
    q^3 + c*q^2 + 7*q + 4 = 0 ∧
    q^3 + d*q^2 + 10*q + 6 = 0

/-- The theorem stating the unique solution for c and d -/
theorem common_roots_solution :
  ∀ c d : ℝ, has_two_common_roots c d → c = -5 ∧ d = -6 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_solution_l1495_149558


namespace NUMINAMATH_CALUDE_additional_miles_with_bakery_stop_l1495_149518

/-- The additional miles driven with a bakery stop compared to without -/
theorem additional_miles_with_bakery_stop
  (apartment_to_bakery : ℕ)
  (bakery_to_grandma : ℕ)
  (grandma_to_apartment : ℕ)
  (h1 : apartment_to_bakery = 9)
  (h2 : bakery_to_grandma = 24)
  (h3 : grandma_to_apartment = 27) :
  (apartment_to_bakery + bakery_to_grandma + grandma_to_apartment) -
  (2 * grandma_to_apartment) = 6 :=
by sorry

end NUMINAMATH_CALUDE_additional_miles_with_bakery_stop_l1495_149518


namespace NUMINAMATH_CALUDE_leak_drains_in_26_hours_l1495_149511

/-- Represents the time it takes for a leak to drain a tank, given the fill times with and without the leak -/
def leak_drain_time (pump_fill_time leak_fill_time : ℚ) : ℚ :=
  let pump_rate := 1 / pump_fill_time
  let combined_rate := 1 / leak_fill_time
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that given the specific fill times, the leak drains the tank in 26 hours -/
theorem leak_drains_in_26_hours :
  leak_drain_time 2 (13/6) = 26 := by sorry

end NUMINAMATH_CALUDE_leak_drains_in_26_hours_l1495_149511


namespace NUMINAMATH_CALUDE_equation_solution_l1495_149566

theorem equation_solution : ∃! n : ℚ, (1 : ℚ) / (n + 1) + (2 : ℚ) / (n + 1) + n / (n + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1495_149566


namespace NUMINAMATH_CALUDE_bottling_probability_l1495_149587

def chocolate_prob : ℚ := 3/4
def vanilla_prob : ℚ := 1/2
def total_days : ℕ := 6
def chocolate_days : ℕ := 4
def vanilla_days : ℕ := 3

theorem bottling_probability : 
  (Nat.choose total_days chocolate_days * chocolate_prob ^ chocolate_days * (1 - chocolate_prob) ^ (total_days - chocolate_days)) *
  (1 - (Nat.choose total_days 0 * vanilla_prob ^ 0 * (1 - vanilla_prob) ^ total_days +
        Nat.choose total_days 1 * vanilla_prob ^ 1 * (1 - vanilla_prob) ^ (total_days - 1) +
        Nat.choose total_days 2 * vanilla_prob ^ 2 * (1 - vanilla_prob) ^ (total_days - 2))) =
  25515/131072 := by
sorry

end NUMINAMATH_CALUDE_bottling_probability_l1495_149587


namespace NUMINAMATH_CALUDE_history_book_cost_l1495_149548

theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 80 →
  math_book_cost = 4 →
  total_price = 373 →
  math_books = 27 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end NUMINAMATH_CALUDE_history_book_cost_l1495_149548


namespace NUMINAMATH_CALUDE_existence_of_infinite_set_l1495_149590

def PositiveInt := { n : ℕ // n > 0 }

def SatisfiesCondition (f : PositiveInt → PositiveInt) : Prop :=
  ∀ x : PositiveInt, (f x).val + (f ⟨x.val + 2, sorry⟩).val ≤ 2 * (f ⟨x.val + 1, sorry⟩).val

theorem existence_of_infinite_set (f : PositiveInt → PositiveInt) (h : SatisfiesCondition f) :
  ∃ M : Set PositiveInt, Set.Infinite M ∧
    ∀ i j k : PositiveInt, i ∈ M → j ∈ M → k ∈ M →
      (i.val - j.val) * (f k).val + (j.val - k.val) * (f i).val + (k.val - i.val) * (f j).val = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_infinite_set_l1495_149590


namespace NUMINAMATH_CALUDE_tyler_puppies_l1495_149531

/-- The number of puppies Tyler has, given the number of dogs and puppies per dog. -/
def total_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) : ℕ :=
  num_dogs * puppies_per_dog

/-- Theorem stating that Tyler has 75 puppies given 15 dogs with 5 puppies each. -/
theorem tyler_puppies : total_puppies 15 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_tyler_puppies_l1495_149531


namespace NUMINAMATH_CALUDE_calculation_proof_l1495_149512

theorem calculation_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |3034 - (1002 / 20.04) - 2983.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1495_149512


namespace NUMINAMATH_CALUDE_worker_loading_time_l1495_149567

/-- The time taken by two workers to load a truck together -/
def combined_time : ℝ := 3.428571428571429

/-- The time taken by the second worker to load the truck alone -/
def second_worker_time : ℝ := 8

/-- The time taken by the first worker to load the truck alone -/
def first_worker_time : ℝ := 1.142857142857143

/-- Theorem stating the relationship between the workers' loading times -/
theorem worker_loading_time :
  (1 / combined_time) = (1 / first_worker_time) + (1 / second_worker_time) :=
by sorry

end NUMINAMATH_CALUDE_worker_loading_time_l1495_149567


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1495_149530

/-- The range of m for which a line y = kx + 1 and an ellipse (x^2)/5 + (y^2)/m = 1 always have common points -/
theorem line_ellipse_intersection_range (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1 → (∃ x' y' : ℝ, y' = k * x' + 1 ∧ x'^2 / 5 + y'^2 / m = 1)) →
  m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1495_149530


namespace NUMINAMATH_CALUDE_peanut_butter_jars_l1495_149543

/-- Given 2032 ounces of peanut butter distributed equally among jars of 16, 28, 40, and 52 ounces,
    the total number of jars is 60. -/
theorem peanut_butter_jars :
  let total_ounces : ℕ := 2032
  let jar_sizes : List ℕ := [16, 28, 40, 52]
  let num_sizes : ℕ := jar_sizes.length
  ∃ (x : ℕ),
    (x * (jar_sizes.sum)) = total_ounces ∧
    (num_sizes * x) = 60
  := by sorry

end NUMINAMATH_CALUDE_peanut_butter_jars_l1495_149543


namespace NUMINAMATH_CALUDE_work_completion_time_l1495_149539

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (2 * (1/x + 1/10) + 10 * (1/x) = 1) →  -- Work completion equation
  (x = 15) :=  -- A's solo completion time is 15 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1495_149539


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_l1495_149505

theorem prime_with_integer_roots (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ x y : ℤ, x^2 + p*x - 204*p = 0 ∧ y^2 + p*y - 204*p = 0) → p = 17 := by
  sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_l1495_149505


namespace NUMINAMATH_CALUDE_chord_intercept_l1495_149568

/-- The value of 'a' in the equation of a line that intercepts a chord of length √3 on a circle -/
theorem chord_intercept (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 3 ∧ x + y + a = 0) →  -- Line intersects circle
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + y1^2 = 3 ∧ x2^2 + y2^2 = 3 ∧  -- Two points on circle
    x1 + y1 + a = 0 ∧ x2 + y2 + a = 0 ∧  -- Two points on line
    (x1 - x2)^2 + (y1 - y2)^2 = 3) →  -- Distance between points is √3
  a = 3 * Real.sqrt 2 / 2 ∨ a = -3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_chord_intercept_l1495_149568


namespace NUMINAMATH_CALUDE_distance_to_origin_l1495_149516

/-- The distance from point (5, -12) to the origin in the Cartesian coordinate system is 13. -/
theorem distance_to_origin : Real.sqrt (5^2 + (-12)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1495_149516


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1495_149536

theorem quadratic_discriminant :
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  let discriminant := b^2 - 4*a*c
  discriminant = 576/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1495_149536


namespace NUMINAMATH_CALUDE_square_difference_equality_l1495_149559

theorem square_difference_equality : (23 + 15)^2 - 3 * (23 - 15)^2 = 1252 := by sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1495_149559


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1495_149577

-- Define the number of students and rows
def num_students : ℕ := 12
def num_rows : ℕ := 2
def students_per_row : ℕ := num_students / num_rows

-- Define the number of test versions
def num_versions : ℕ := 2

-- Define the function to calculate the number of seating arrangements
def seating_arrangements : ℕ := 2 * (Nat.factorial students_per_row)^2

-- Theorem statement
theorem correct_seating_arrangements :
  seating_arrangements = 1036800 :=
sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1495_149577


namespace NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l1495_149552

/-- A pentagonal prism is a three-dimensional geometric shape with two congruent pentagonal bases 
    and five rectangular lateral faces. --/
structure PentagonalPrism where
  base : Pentagon
  height : ℝ
  height_pos : height > 0

/-- The angle φ is the angle between adjacent edges in a lateral face of the pentagonal prism. --/
def lateral_face_angle (prism : PentagonalPrism) : ℝ := sorry

/-- Theorem: In a pentagonal prism, the angle φ between adjacent edges in a lateral face must be 90°. --/
theorem pentagonal_prism_lateral_angle (prism : PentagonalPrism) : 
  lateral_face_angle prism = 90 := by sorry

end NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l1495_149552


namespace NUMINAMATH_CALUDE_max_xy_value_fraction_inequality_l1495_149535

-- Part I
theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 5 * y = 20) :
  ∃ (max_val : ℝ), max_val = 10 ∧ ∀ (z : ℝ), x * y ≤ z → z ≤ max_val :=
sorry

-- Part II
theorem fraction_inequality (a b c d k : ℝ) (hab : a > b) (hb : b > 0) (hcd : c < d) (hd : d < 0) (hk : k < 0) :
  k / (a - c) > k / (b - d) :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_fraction_inequality_l1495_149535


namespace NUMINAMATH_CALUDE_stamps_ratio_l1495_149563

def stamps_problem (bert ernie peggy : ℕ) : Prop :=
  bert = 4 * ernie ∧
  ∃ k : ℕ, ernie = k * peggy ∧
  peggy = 75 ∧
  bert = peggy + 825

theorem stamps_ratio (bert ernie peggy : ℕ) 
  (h : stamps_problem bert ernie peggy) : ernie / peggy = 3 := by
  sorry

end NUMINAMATH_CALUDE_stamps_ratio_l1495_149563


namespace NUMINAMATH_CALUDE_inequality_preservation_l1495_149550

theorem inequality_preservation (x y : ℝ) (h : x > y) : x / 2 > y / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1495_149550


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1495_149591

theorem intersection_of_lines (p q : ℝ) : 
  (∃ x y : ℝ, y = p * x + 4 ∧ p * y = q * x - 7 ∧ x = 3 ∧ y = 1) → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1495_149591


namespace NUMINAMATH_CALUDE_race_distance_proof_l1495_149583

/-- The distance in meters by which runner A beats runner B -/
def beat_distance : ℝ := 56

/-- The time in seconds by which runner A beats runner B -/
def beat_time : ℝ := 7

/-- Runner A's time to complete the race in seconds -/
def a_time : ℝ := 8

/-- The total distance of the race in meters -/
def race_distance : ℝ := 120

theorem race_distance_proof :
  (beat_distance / beat_time) * (a_time + beat_time) = race_distance :=
sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1495_149583


namespace NUMINAMATH_CALUDE_irene_income_l1495_149527

/-- Calculates the total income for a given number of hours worked -/
def total_income (regular_income : ℕ) (overtime_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  let regular_hours := 40
  let overtime_hours := max (hours_worked - regular_hours) 0
  regular_income + overtime_rate * overtime_hours

/-- Irene's income calculation theorem -/
theorem irene_income :
  total_income 500 20 50 = 700 := by
  sorry

#eval total_income 500 20 50

end NUMINAMATH_CALUDE_irene_income_l1495_149527


namespace NUMINAMATH_CALUDE_geologists_can_reach_station_l1495_149570

/-- Represents the problem of geologists traveling to a station. -/
structure GeologistsProblem where
  totalDistance : ℝ
  timeLimit : ℝ
  motorcycleSpeed : ℝ
  walkingSpeed : ℝ
  numberOfGeologists : ℕ

/-- Checks if the geologists can reach the station within the time limit. -/
def canReachStation (problem : GeologistsProblem) : Prop :=
  ∃ (strategy : Unit), 
    let twoGeologistsTime := problem.totalDistance / problem.motorcycleSpeed
    let walkingTime := problem.totalDistance / problem.walkingSpeed
    let meetingTime := (problem.totalDistance - problem.walkingSpeed) / (problem.motorcycleSpeed + problem.walkingSpeed)
    let returnTime := (problem.totalDistance - problem.walkingSpeed * meetingTime) / problem.motorcycleSpeed
    twoGeologistsTime ≤ problem.timeLimit ∧ 
    walkingTime ≤ problem.timeLimit ∧
    meetingTime + returnTime ≤ problem.timeLimit

/-- The specific problem instance. -/
def geologistsProblem : GeologistsProblem :=
  { totalDistance := 60
  , timeLimit := 3
  , motorcycleSpeed := 50
  , walkingSpeed := 5
  , numberOfGeologists := 3 }

/-- Theorem stating that the geologists can reach the station within the time limit. -/
theorem geologists_can_reach_station : canReachStation geologistsProblem := by
  sorry


end NUMINAMATH_CALUDE_geologists_can_reach_station_l1495_149570


namespace NUMINAMATH_CALUDE_percentage_qualified_school_B_l1495_149544

/-- Percentage of students qualified from school A -/
def percentage_qualified_A : ℝ := 70

/-- Ratio of students appeared in school B compared to school A -/
def ratio_appeared_B_to_A : ℝ := 1.2

/-- Ratio of students qualified from school B compared to school A -/
def ratio_qualified_B_to_A : ℝ := 1.5

/-- Theorem: The percentage of students qualified from school B is 87.5% -/
theorem percentage_qualified_school_B :
  (ratio_qualified_B_to_A * percentage_qualified_A) / (ratio_appeared_B_to_A * 100) * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_qualified_school_B_l1495_149544


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1495_149596

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ+, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36 →
  a 2 + a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1495_149596


namespace NUMINAMATH_CALUDE_angle_from_point_l1495_149526

theorem angle_from_point (a : Real) (h1 : 0 < a ∧ a < π/2) : 
  (∃ (x y : Real), x = 4 * Real.sin 3 ∧ y = -4 * Real.cos 3 ∧ 
   x = 4 * Real.sin a ∧ y = -4 * Real.cos a) → 
  a = 3 - π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_from_point_l1495_149526


namespace NUMINAMATH_CALUDE_plates_problem_l1495_149585

theorem plates_problem (initial_plates added_plates total_plates : ℕ) 
  (h1 : added_plates = 37)
  (h2 : total_plates = 83)
  (h3 : initial_plates + added_plates = total_plates) :
  initial_plates = 46 := by
  sorry

end NUMINAMATH_CALUDE_plates_problem_l1495_149585


namespace NUMINAMATH_CALUDE_equidistant_points_characterization_l1495_149597

/-- A ray in a plane --/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The set of points equidistant from two rays --/
def EquidistantPoints (ray1 ray2 : Ray) : Set (ℝ × ℝ) :=
  sorry

/-- Angle bisector of two lines --/
def AngleBisector (line1 line2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- Perpendicular bisector of a segment --/
def PerpendicularBisector (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Parabola with focus and directrix --/
def Parabola (focus : ℝ × ℝ) (directrix : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- The line containing a ray --/
def LineContainingRay (ray : Ray) : Set (ℝ × ℝ) :=
  sorry

theorem equidistant_points_characterization (ray1 ray2 : Ray) :
  EquidistantPoints ray1 ray2 =
    (AngleBisector (LineContainingRay ray1) (LineContainingRay ray2)) ∪
    (if ray1.start ≠ ray2.start then PerpendicularBisector ray1.start ray2.start else ∅) ∪
    (Parabola ray1.start (LineContainingRay ray2)) ∪
    (Parabola ray2.start (LineContainingRay ray1)) :=
  sorry

end NUMINAMATH_CALUDE_equidistant_points_characterization_l1495_149597


namespace NUMINAMATH_CALUDE_clothing_store_loss_l1495_149520

/-- Proves that selling two sets of clothes at 168 yuan each, with one set having a 20% profit
    and the other having a 20% loss, results in a total loss of 14 yuan. -/
theorem clothing_store_loss (selling_price : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ) :
  selling_price = 168 →
  profit_percentage = 0.2 →
  loss_percentage = 0.2 →
  let profit_cost := selling_price / (1 + profit_percentage)
  let loss_cost := selling_price / (1 - loss_percentage)
  (2 * selling_price) - (profit_cost + loss_cost) = -14 := by
sorry

end NUMINAMATH_CALUDE_clothing_store_loss_l1495_149520


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1495_149569

theorem grade_assignment_count : (4 : ℕ) ^ 15 = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1495_149569


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2013_l1495_149589

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The sequence of last four digits of powers of 5 -/
def lastFourDigitsSequence : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2013 :
  lastFourDigits (5^2013) = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2013_l1495_149589


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1495_149533

theorem fraction_multiplication (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (6 * x * y) / (5 * z^2) * (10 * z^3) / (9 * x * y) = 4 * z / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1495_149533


namespace NUMINAMATH_CALUDE_coloring_theorem_l1495_149502

/-- A point in the plane with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- A line parallel to one of the coordinate axes -/
inductive Line
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- The set of points on a given line -/
def pointsOnLine (S : Finset Point) (L : Line) : Finset Point :=
  match L with
  | Line.Horizontal y => S.filter (fun p => p.y = y)
  | Line.Vertical x => S.filter (fun p => p.x = x)

/-- The main theorem -/
theorem coloring_theorem (S : Finset Point) :
  ∃ (f : Point → Int),
    (∀ p ∈ S, f p = 1 ∨ f p = -1) ∧
    (∀ L : Line, (pointsOnLine S L).sum f ∈ ({-1, 0, 1} : Finset Int)) := by
  sorry


end NUMINAMATH_CALUDE_coloring_theorem_l1495_149502


namespace NUMINAMATH_CALUDE_covering_recurrence_l1495_149581

/-- Number of ways to cover a 2 × n rectangle with 1 × 2 pieces -/
def coveringWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => coveringWays (n + 1) + coveringWays n

/-- The recurrence relation for covering a 2 × n rectangle with 1 × 2 pieces -/
theorem covering_recurrence (n : ℕ) (h : n ≥ 2) :
  coveringWays n = coveringWays (n - 1) + coveringWays (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_covering_recurrence_l1495_149581


namespace NUMINAMATH_CALUDE_bag_probability_l1495_149523

theorem bag_probability (d x : ℕ) : 
  d = x + (x + 1) + (x + 2) →
  (x : ℚ) / d < 1 / 6 →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_bag_probability_l1495_149523


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1495_149524

theorem rectangular_solid_surface_area
  (a b c : ℝ)
  (sum_edges : a + b + c = 14)
  (diagonal : a^2 + b^2 + c^2 = 11^2) :
  2 * (a * b + b * c + a * c) = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1495_149524


namespace NUMINAMATH_CALUDE_cricket_average_proof_l1495_149551

def average_runs (total_runs : ℕ) (innings : ℕ) : ℚ :=
  (total_runs : ℚ) / (innings : ℚ)

theorem cricket_average_proof 
  (initial_innings : ℕ) 
  (next_innings_runs : ℕ) 
  (average_increase : ℚ) :
  initial_innings = 10 →
  next_innings_runs = 74 →
  average_increase = 4 →
  ∃ (initial_total_runs : ℕ),
    average_runs (initial_total_runs + next_innings_runs) (initial_innings + 1) =
    average_runs initial_total_runs initial_innings + average_increase →
    average_runs initial_total_runs initial_innings = 30 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_proof_l1495_149551


namespace NUMINAMATH_CALUDE_sebastians_orchestra_size_l1495_149528

/-- Represents the composition of an orchestra --/
structure Orchestra where
  percussion : Nat
  trombone : Nat
  trumpet : Nat
  french_horn : Nat
  violin : Nat
  cello : Nat
  contrabass : Nat
  clarinet : Nat
  flute : Nat
  maestro : Nat

/-- The total number of people in the orchestra --/
def Orchestra.total (o : Orchestra) : Nat :=
  o.percussion + o.trombone + o.trumpet + o.french_horn +
  o.violin + o.cello + o.contrabass +
  o.clarinet + o.flute + o.maestro

/-- The specific orchestra composition from the problem --/
def sebastians_orchestra : Orchestra :=
  { percussion := 1
  , trombone := 4
  , trumpet := 2
  , french_horn := 1
  , violin := 3
  , cello := 1
  , contrabass := 1
  , clarinet := 3
  , flute := 4
  , maestro := 1
  }

/-- Theorem stating that the total number of people in Sebastian's orchestra is 21 --/
theorem sebastians_orchestra_size :
  sebastians_orchestra.total = 21 := by
  sorry

end NUMINAMATH_CALUDE_sebastians_orchestra_size_l1495_149528


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1495_149573

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The property of geometric sequences: if m + n = p + q, then a_m * a_n = a_p * a_q -/
axiom geometric_sequence_property {a : ℕ → ℝ} (h : IsGeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_problem (a : ℕ → ℝ) (h : IsGeometricSequence a) 
  (h1 : a 5 * a 14 = 5) : a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1495_149573


namespace NUMINAMATH_CALUDE_quadratic_factor_problem_l1495_149501

theorem quadratic_factor_problem (d e : ℤ) :
  let q : ℝ → ℝ := fun x ↦ x^2 + d*x + e
  (∃ r : ℝ → ℝ, (fun x ↦ x^4 + x^3 + 8*x^2 + 7*x + 18) = q * r) ∧
  (∃ s : ℝ → ℝ, (fun x ↦ 2*x^4 + 3*x^3 + 9*x^2 + 8*x + 20) = q * s) →
  q 1 = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factor_problem_l1495_149501


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_l1495_149592

theorem smallest_integer_gcd_lcm (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  m = 60 ∨ n = 60 →
  Nat.gcd m n = x + 3 →
  Nat.lcm m n = x * (x + 3) →
  (m = 60 → n ≥ 45) ∧ (n = 60 → m ≥ 45) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_l1495_149592


namespace NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l1495_149537

def S : Finset Nat := Finset.range 1999

def f (X : Finset Nat) : Nat :=
  X.sum id

theorem sum_over_subsets_equals_power_of_two :
  (Finset.powerset S).sum (fun E => (f E : ℚ) / (f S : ℚ)) = (2 : ℚ) ^ 1998 :=
sorry

end NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l1495_149537


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1495_149534

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ (a d : ℝ), 
    (36 + k : ℝ) = (a - d)^2 ∧ 
    (300 + k : ℝ) = a^2 ∧ 
    (596 + k : ℝ) = (a + d)^2) ↔ 
  k = 925 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1495_149534


namespace NUMINAMATH_CALUDE_slope_less_than_two_l1495_149594

/-- Theorem: For two different points on a linear function, if the product of differences is negative, then the slope is less than 2 -/
theorem slope_less_than_two (k a b c d : ℝ) : 
  a ≠ c →  -- A and B are different points
  b = k * a - 2 * a - 1 →  -- A is on the line
  d = k * c - 2 * c - 1 →  -- B is on the line
  (c - a) * (d - b) < 0 →  -- Given condition
  k < 2 := by
  sorry


end NUMINAMATH_CALUDE_slope_less_than_two_l1495_149594


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l1495_149542

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l1495_149542


namespace NUMINAMATH_CALUDE_roots_transformation_l1495_149500

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 5*r₁^2 + 12 = 0) ∧ 
  (r₂^3 - 5*r₂^2 + 12 = 0) ∧ 
  (r₃^3 - 5*r₃^2 + 12 = 0) → 
  ((3*r₁)^3 - 15*(3*r₁)^2 + 324 = 0) ∧ 
  ((3*r₂)^3 - 15*(3*r₂)^2 + 324 = 0) ∧ 
  ((3*r₃)^3 - 15*(3*r₃)^2 + 324 = 0) := by
sorry

end NUMINAMATH_CALUDE_roots_transformation_l1495_149500


namespace NUMINAMATH_CALUDE_lcm_36_150_l1495_149562

theorem lcm_36_150 : Nat.lcm 36 150 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_150_l1495_149562


namespace NUMINAMATH_CALUDE_book_reading_theorem_l1495_149578

def book_reading_problem (total_pages : ℕ) (reading_rate : ℕ) (monday_hours : ℕ) (tuesday_hours : ℚ) : ℚ :=
  let pages_read := monday_hours * reading_rate + tuesday_hours * reading_rate
  let pages_left := total_pages - pages_read
  pages_left / reading_rate

theorem book_reading_theorem :
  book_reading_problem 248 16 3 (13/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_theorem_l1495_149578


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1495_149576

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -32*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-8, 0)

-- Define a point on the parabola
def point_on_parabola (x₀ : ℝ) : ℝ × ℝ := (x₀, 4)

-- State the theorem
theorem parabola_focus_distance (x₀ : ℝ) :
  parabola x₀ 4 →
  let P := point_on_parabola x₀
  let F := focus
  dist P F = 17/2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1495_149576


namespace NUMINAMATH_CALUDE_joe_egg_hunt_l1495_149503

theorem joe_egg_hunt (park_eggs : ℕ) (town_hall_eggs : ℕ) (total_eggs : ℕ) 
  (h1 : park_eggs = 5)
  (h2 : town_hall_eggs = 3)
  (h3 : total_eggs = 20) :
  total_eggs - park_eggs - town_hall_eggs = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_egg_hunt_l1495_149503


namespace NUMINAMATH_CALUDE_min_study_tools_l1495_149582

theorem min_study_tools (n : ℕ) : n^3 ≥ 366 ∧ (n-1)^3 < 366 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_study_tools_l1495_149582


namespace NUMINAMATH_CALUDE_f_minus_five_eq_zero_l1495_149514

open Function

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_minus_five_eq_zero
  (f : ℝ → ℝ)
  (h1 : is_even (fun x ↦ f (1 - 2*x)))
  (h2 : is_odd (fun x ↦ f (x - 1))) :
  f (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_five_eq_zero_l1495_149514


namespace NUMINAMATH_CALUDE_certain_number_proof_l1495_149521

theorem certain_number_proof : ∃ x : ℕ, (3 * 16) + (3 * 17) + (3 * 20) + x = 170 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1495_149521


namespace NUMINAMATH_CALUDE_prob_neither_red_nor_green_l1495_149541

/-- Given a bag with green, black, and red pens, this theorem proves the probability
    of picking a pen that is neither red nor green. -/
theorem prob_neither_red_nor_green (green black red : ℕ) 
  (h_green : green = 5) 
  (h_black : black = 6) 
  (h_red : red = 7) : 
  (black : ℚ) / (green + black + red) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_neither_red_nor_green_l1495_149541


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1495_149515

/-- Proves that the ratio of A's speed to B's speed is 2:1 in a race where A gives B a head start -/
theorem race_speed_ratio (race_length : ℝ) (head_start : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : race_length = 142)
  (h2 : head_start = 71)
  (h3 : race_length / speed_A = (race_length - head_start) / speed_B) :
  speed_A / speed_B = 2 := by
  sorry

#check race_speed_ratio

end NUMINAMATH_CALUDE_race_speed_ratio_l1495_149515


namespace NUMINAMATH_CALUDE_fraction_of_a_equal_to_quarter_of_b_l1495_149529

theorem fraction_of_a_equal_to_quarter_of_b : ∀ (a b x : ℚ), 
  a + b = 1210 →
  b = 484 →
  x * a = (1/4) * b →
  x = 1/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_a_equal_to_quarter_of_b_l1495_149529


namespace NUMINAMATH_CALUDE_john_restringing_problem_l1495_149522

/-- The number of basses John needs to restring -/
def num_basses : ℕ := 3

/-- The number of guitars John needs to restring -/
def num_guitars : ℕ := 2 * num_basses

/-- The number of 8-string guitars John needs to restring -/
def num_8string_guitars : ℕ := num_guitars - 3

/-- The total number of strings needed -/
def total_strings : ℕ := 72

theorem john_restringing_problem :
  4 * num_basses + 6 * num_guitars + 8 * num_8string_guitars = total_strings :=
by sorry

end NUMINAMATH_CALUDE_john_restringing_problem_l1495_149522


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l1495_149538

theorem no_linear_term_implies_m_equals_negative_three (m : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l1495_149538


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1495_149509

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y

/-- The main theorem stating that any function satisfying the functional equation must be f(x) = 3x -/
theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), SatisfiesFunctionalEq f → (∀ x, f x = 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1495_149509


namespace NUMINAMATH_CALUDE_opposite_face_of_A_is_F_l1495_149561

-- Define the set of labels
inductive Label
| A | B | C | D | E | F

-- Define the structure of a cube face
structure CubeFace where
  label : Label

-- Define the structure of a cube
structure Cube where
  faces : List CubeFace
  adjacent : Label → List Label

-- Define the property of being opposite faces
def isOpposite (cube : Cube) (l1 l2 : Label) : Prop :=
  l1 ∉ cube.adjacent l2 ∧ l2 ∉ cube.adjacent l1

-- Theorem statement
theorem opposite_face_of_A_is_F (cube : Cube) 
  (h1 : cube.faces.length = 6)
  (h2 : ∀ l : Label, l ∈ (cube.faces.map CubeFace.label))
  (h3 : cube.adjacent Label.A = [Label.B, Label.C, Label.D, Label.E]) :
  isOpposite cube Label.A Label.F :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_A_is_F_l1495_149561


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1495_149506

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

-- Theorem statement
theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1495_149506


namespace NUMINAMATH_CALUDE_fraction_of_task_completed_l1495_149580

theorem fraction_of_task_completed (total_time minutes : ℕ) (h : total_time = 60) (h2 : minutes = 15) :
  (minutes : ℚ) / total_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_task_completed_l1495_149580


namespace NUMINAMATH_CALUDE_butterfly_collection_l1495_149572

theorem butterfly_collection (total : ℕ) (blue : ℕ) : 
  total = 19 → 
  blue = 6 → 
  ∃ (yellow : ℕ), blue = 2 * yellow → 
  ∃ (black : ℕ), black = total - (blue + yellow) ∧ black = 10 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l1495_149572


namespace NUMINAMATH_CALUDE_line_intersecting_ellipse_slope_l1495_149593

/-- The slope of a line intersecting an ellipse -/
theorem line_intersecting_ellipse_slope (m : ℝ) : 
  (∃ x y : ℝ, 25 * x^2 + 16 * y^2 = 400 ∧ y = m * x + 8) ↔ m^2 ≥ 39/16 :=
sorry

end NUMINAMATH_CALUDE_line_intersecting_ellipse_slope_l1495_149593


namespace NUMINAMATH_CALUDE_not_perfect_square_l1495_149588

theorem not_perfect_square (x y : ℤ) : ∃ (z : ℤ), (x^2 + 3*x + 1)^2 + (y^2 + 3*y + 1)^2 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1495_149588


namespace NUMINAMATH_CALUDE_power_equation_equality_l1495_149540

theorem power_equation_equality : 4^3 - 8 = 5^2 + 31 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_equality_l1495_149540


namespace NUMINAMATH_CALUDE_matrix_power_four_l1495_149508

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![34, 21; 21, 13] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1495_149508


namespace NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_4600_l1495_149517

/-- Calculates the total profit given the investments, time periods, and Rajan's profit share -/
theorem calculate_total_profit (rajan_investment : ℕ) (rakesh_investment : ℕ) (mukesh_investment : ℕ)
  (rajan_months : ℕ) (rakesh_months : ℕ) (mukesh_months : ℕ) (rajan_profit : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_months
  let rakesh_share := rakesh_investment * rakesh_months
  let mukesh_share := mukesh_investment * mukesh_months
  let total_share := rajan_share + rakesh_share + mukesh_share
  let total_profit := (rajan_profit * total_share) / rajan_share
  total_profit

/-- Proves that the total profit is 4600 given the specific investments and Rajan's profit share -/
theorem total_profit_is_4600 :
  calculate_total_profit 20000 25000 15000 12 4 8 2400 = 4600 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_4600_l1495_149517


namespace NUMINAMATH_CALUDE_binary_111011_is_59_l1495_149586

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zip b (List.range b.length).reverse).foldl
    (fun acc (bit, power) => acc + if bit then 2^power else 0) 0

theorem binary_111011_is_59 :
  binary_to_decimal [true, true, true, false, true, true] = 59 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011_is_59_l1495_149586


namespace NUMINAMATH_CALUDE_slower_rider_speed_l1495_149574

/-- The speed of the slower rider in miles per hour -/
def slower_speed : ℚ := 5/3

/-- The speed of the faster rider in miles per hour -/
def faster_speed : ℚ := 2 * slower_speed

/-- The distance between the cyclists in miles -/
def distance : ℚ := 20

/-- The time it takes for the cyclists to meet when riding towards each other in hours -/
def time_towards : ℚ := 4

/-- The time it takes for the faster rider to catch up when riding in the same direction in hours -/
def time_same_direction : ℚ := 10

theorem slower_rider_speed :
  (distance = (faster_speed + slower_speed) * time_towards) ∧
  (distance = (faster_speed - slower_speed) * time_same_direction) ∧
  (faster_speed = 2 * slower_speed) →
  slower_speed = 5/3 := by sorry

end NUMINAMATH_CALUDE_slower_rider_speed_l1495_149574


namespace NUMINAMATH_CALUDE_exists_equal_face_products_l1495_149579

/-- A type representing the arrangement of numbers on a cube's edges -/
def CubeArrangement := Fin 12 → Fin 12

/-- Predicate to check if an arrangement is valid (each number used once) -/
def is_valid_arrangement (arr : CubeArrangement) : Prop :=
  Function.Injective arr

/-- The set of indices representing the top face of the cube -/
def top_face : Finset (Fin 12) :=
  {0, 1, 2, 3}

/-- The set of indices representing the bottom face of the cube -/
def bottom_face : Finset (Fin 12) :=
  {4, 5, 6, 7}

/-- The product of numbers on a face given an arrangement -/
def face_product (arr : CubeArrangement) (face : Finset (Fin 12)) : ℕ :=
  face.prod (fun i => (arr i).val + 1)

/-- Theorem stating that there exists a valid arrangement satisfying the condition -/
theorem exists_equal_face_products : ∃ (arr : CubeArrangement), 
  is_valid_arrangement arr ∧ 
  face_product arr top_face = face_product arr bottom_face :=
sorry

end NUMINAMATH_CALUDE_exists_equal_face_products_l1495_149579


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1495_149599

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 13 = 25) ∧ 
  (d^2 - 6*d + 13 = 25) ∧ 
  (c ≥ d) →
  c + 2*d = 9 - Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1495_149599


namespace NUMINAMATH_CALUDE_farmer_earnings_example_l1495_149598

/-- Calculates a farmer's earnings from egg sales over a given number of weeks -/
def farmer_earnings (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  let total_eggs := num_chickens * eggs_per_chicken * num_weeks
  let dozens := total_eggs / 12
  dozens * price_per_dozen

theorem farmer_earnings_example : farmer_earnings 46 6 3 8 = 552 := by
  sorry

end NUMINAMATH_CALUDE_farmer_earnings_example_l1495_149598


namespace NUMINAMATH_CALUDE_square_area_ratio_l1495_149513

/-- Given three squares A, B, and C with side lengths x, 3x, and 2x respectively,
    prove that the ratio of the area of Square A to the combined area of Square B and Square C is 1/13 -/
theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2 + (2*x)^2) = 1 / 13 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1495_149513


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1495_149555

/-- Proves that the cost of fencing per meter for a rectangular plot with given dimensions and total fencing cost is 26.50 Rs. -/
theorem fencing_cost_per_meter
  (length breadth : ℝ)
  (length_relation : length = breadth + 10)
  (length_value : length = 55)
  (total_cost : ℝ)
  (total_cost_value : total_cost = 5300)
  : total_cost / (2 * (length + breadth)) = 26.50 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1495_149555


namespace NUMINAMATH_CALUDE_circle_equation_through_point_l1495_149564

/-- The equation of a circle with center (1, 0) passing through (1, -1) -/
theorem circle_equation_through_point :
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let equation (x y : ℝ) := (x - center.1)^2 + (y - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2
  ∀ x y : ℝ, equation x y ↔ (x - 1)^2 + y^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_through_point_l1495_149564


namespace NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l1495_149565

def remaining_rolls_to_sell (total_required : ℕ) (sales_to_grandmother : ℕ) (sales_to_uncle : ℕ) (sales_to_neighbor : ℕ) : ℕ :=
  total_required - (sales_to_grandmother + sales_to_uncle + sales_to_neighbor)

theorem chandler_wrapping_paper_sales : 
  remaining_rolls_to_sell 12 3 4 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l1495_149565


namespace NUMINAMATH_CALUDE_ball_probability_problem_l1495_149547

theorem ball_probability_problem (R B : ℕ) : 
  (R * (R - 1)) / ((R + B) * (R + B - 1)) = 2/7 →
  (2 * R * B) / ((R + B) * (R + B - 1)) = 1/2 →
  R = 105 ∧ B = 91 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_problem_l1495_149547


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1495_149525

/-- The polar equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The Cartesian equation of a circle with center (h, k) and radius r -/
def cartesian_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the circle ρ = 2cosθ has center (1, 0) and radius 1 -/
theorem circle_center_and_radius :
  ∀ x y ρ θ : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  cartesian_equation x y 1 0 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1495_149525


namespace NUMINAMATH_CALUDE_vector_operation_l1495_149560

/-- Given two vectors in ℝ², prove that their specific linear combination equals a certain vector. -/
theorem vector_operation (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1495_149560


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l1495_149575

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- The problem statement -/
theorem similar_triangles_problem 
  (t1 t2 : Triangle)  -- Two triangles
  (h1 : t1.area > t2.area)  -- t1 is the larger triangle
  (h2 : t1.area - t2.area = 32)  -- Area difference is 32
  (h3 : ∃ k : ℕ, t1.area / t2.area = k^2)  -- Ratio of areas is square of an integer
  (h4 : ∃ n : ℕ, t2.area = n)  -- Smaller triangle area is an integer
  (h5 : t2.side = 4)  -- Side of smaller triangle is 4
  : t1.side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l1495_149575


namespace NUMINAMATH_CALUDE_set_equality_l1495_149571

open Set

-- Define the sets
def R : Set ℝ := univ
def A : Set ℝ := {x | x^2 ≥ 4}
def B : Set ℝ := {y | ∃ x, y = |Real.tan x|}

-- State the theorem
theorem set_equality : (R \ A) ∩ B = {x | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1495_149571


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1495_149545

/-- Given a parabola y = ax^2 + bx + c with vertex (h, k) and passing through (0, -k) where k ≠ 0,
    prove that b = 4k/h -/
theorem parabola_coefficient (a b c h k : ℝ) (hk : k ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k) →
  a * 0^2 + b * 0 + c = -k →
  b = 4 * k / h := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1495_149545


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l1495_149557

/-- Proves that for a quadratic function y = ax^2 + bx + c passing through points 
(-2,8), (4,8), and (7,15), the x-coordinate of the vertex is 1. -/
theorem parabola_vertex_x_coordinate (a b c : ℝ) : 
  (8 = a * (-2)^2 + b * (-2) + c) →
  (8 = a * 4^2 + b * 4 + c) →
  (15 = a * 7^2 + b * 7 + c) →
  (∃ (x : ℝ), x = 1 ∧ ∀ (t : ℝ), a * t^2 + b * t + c ≥ a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l1495_149557


namespace NUMINAMATH_CALUDE_sector_max_area_and_angle_l1495_149504

/-- Given a sector of a circle with perimeter 30 cm, prove that the maximum area is 225/4 cm² 
    and the corresponding central angle is 2 radians. -/
theorem sector_max_area_and_angle (r : ℝ) (l : ℝ) (α : ℝ) (area : ℝ) :
  l + 2 * r = 30 →                            -- Perimeter condition
  l = r * α →                                 -- Arc length formula
  area = (1 / 2) * r * l →                    -- Area formula for sector
  (∀ r' l' α' area', l' + 2 * r' = 30 → l' = r' * α' → area' = (1 / 2) * r' * l' → area' ≤ area) →
  area = 225 / 4 ∧ α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_and_angle_l1495_149504


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1495_149553

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  interval : ℕ
  first_number : ℕ

/-- Calculates the number drawn from a given group -/
def number_from_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_number + s.interval * (group - 1)

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_students = 160)
  (h2 : s.num_groups = 20)
  (h3 : s.interval = 8)
  (h4 : number_from_group s 16 = 123) :
  number_from_group s 2 = 11 := by
  sorry

#check systematic_sampling_theorem

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1495_149553


namespace NUMINAMATH_CALUDE_book_price_problem_l1495_149507

theorem book_price_problem (cost_price : ℝ) : 
  (110 / 100 * cost_price = 1100) → 
  (80 / 100 * cost_price = 800) := by
  sorry

end NUMINAMATH_CALUDE_book_price_problem_l1495_149507


namespace NUMINAMATH_CALUDE_power_quotient_rule_l1495_149554

theorem power_quotient_rule (a : ℝ) : a^5 / a^3 = a^2 := by sorry

end NUMINAMATH_CALUDE_power_quotient_rule_l1495_149554


namespace NUMINAMATH_CALUDE_path_area_and_cost_l1495_149519

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per square meter -/
def construction_cost (path_area cost_per_sqm : ℝ) : ℝ :=
  path_area * cost_per_sqm

theorem path_area_and_cost (field_length field_width path_width cost_per_sqm : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 3.5)
  (h4 : cost_per_sqm = 2) :
  path_area field_length field_width path_width = 959 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 1918 := by
  sorry

#eval path_area 75 55 3.5
#eval construction_cost (path_area 75 55 3.5) 2

end NUMINAMATH_CALUDE_path_area_and_cost_l1495_149519


namespace NUMINAMATH_CALUDE_lock_combinations_l1495_149510

def digits : ℕ := 10
def dials : ℕ := 4
def even_digits : ℕ := 5

theorem lock_combinations : 
  (even_digits) * (digits - 1) * (digits - 2) * (digits - 3) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_lock_combinations_l1495_149510


namespace NUMINAMATH_CALUDE_solution_set_and_range_l1495_149546

def f (a x : ℝ) : ℝ := -x^2 + a*x + 4

def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem solution_set_and_range :
  (∀ x ∈ Set.Icc (-1 : ℝ) ((Real.sqrt 17 - 1) / 2), f 1 x ≥ g x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x > g x) ∧
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ g x) ∧
  (∀ a < -1, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x) ∧
  (∀ a > 1, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l1495_149546


namespace NUMINAMATH_CALUDE_factorial_sum_of_squares_solutions_l1495_149584

theorem factorial_sum_of_squares_solutions :
  ∀ a b n : ℕ+,
    a ≤ b →
    n < 14 →
    a^2 + b^2 = n! →
    ((a = 1 ∧ b = 1 ∧ n = 2) ∨ (a = 12 ∧ b = 24 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_of_squares_solutions_l1495_149584
