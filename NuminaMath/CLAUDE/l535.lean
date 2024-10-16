import Mathlib

namespace NUMINAMATH_CALUDE_exists_number_with_removable_digit_l535_53552

-- Define a function to check if a number has a non-zero digit
def has_nonzero_digit (n : ℕ) : Prop :=
  ∃ (k : ℕ), (n / 10^k) % 10 ≠ 0

-- Define a function to check if a number can be obtained by removing a non-zero digit from another number
def can_remove_nonzero_digit (n n' : ℕ) : Prop :=
  ∃ (k : ℕ), 
    let d := (n / 10^k) % 10
    d ≠ 0 ∧ n' = (n / 10^(k+1)) * 10^k + n % 10^k

theorem exists_number_with_removable_digit (d : ℕ) (hd : d > 0) : 
  ∃ (n : ℕ), 
    n % d = 0 ∧ 
    has_nonzero_digit n ∧ 
    ∃ (n' : ℕ), can_remove_nonzero_digit n n' ∧ n' % d = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_removable_digit_l535_53552


namespace NUMINAMATH_CALUDE_beach_ball_surface_area_l535_53573

theorem beach_ball_surface_area (d : ℝ) (h : d = 15) :
  4 * Real.pi * (d / 2)^2 = 225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_beach_ball_surface_area_l535_53573


namespace NUMINAMATH_CALUDE_other_color_counts_l535_53542

def total_students : ℕ := 800

def blue_shirt_percent : ℚ := 45/100
def red_shirt_percent : ℚ := 23/100
def green_shirt_percent : ℚ := 15/100

def black_pants_percent : ℚ := 30/100
def khaki_pants_percent : ℚ := 25/100
def jeans_percent : ℚ := 10/100

def white_shoes_percent : ℚ := 40/100
def black_shoes_percent : ℚ := 20/100
def brown_shoes_percent : ℚ := 15/100

theorem other_color_counts :
  let other_shirt_count := total_students - (blue_shirt_percent + red_shirt_percent + green_shirt_percent) * total_students
  let other_pants_count := total_students - (black_pants_percent + khaki_pants_percent + jeans_percent) * total_students
  let other_shoes_count := total_students - (white_shoes_percent + black_shoes_percent + brown_shoes_percent) * total_students
  (other_shirt_count : ℚ) = 136 ∧ (other_pants_count : ℚ) = 280 ∧ (other_shoes_count : ℚ) = 200 :=
by sorry

end NUMINAMATH_CALUDE_other_color_counts_l535_53542


namespace NUMINAMATH_CALUDE_range_of_fraction_l535_53598

theorem range_of_fraction (x y : ℝ) 
  (h1 : x - 2*y + 4 ≥ 0) 
  (h2 : x ≤ 2) 
  (h3 : x + y - 2 ≥ 0) : 
  1/4 ≤ (y + 1) / (x + 2) ∧ (y + 1) / (x + 2) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l535_53598


namespace NUMINAMATH_CALUDE_albums_theorem_l535_53536

-- Define the number of albums for each person
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- Define the total number of albums
def total_albums : ℕ := adele_albums + bridget_albums + katrina_albums + miriam_albums

-- Theorem to prove
theorem albums_theorem : total_albums = 585 := by
  sorry

end NUMINAMATH_CALUDE_albums_theorem_l535_53536


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l535_53504

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (final_people : ℕ) (final_avg_weight : ℝ) :
  initial_people = 6 ∧ 
  initial_avg_weight = 160 ∧ 
  final_people = 7 ∧ 
  final_avg_weight = 151 →
  (final_people * final_avg_weight) - (initial_people * initial_avg_weight) = 97 := by
sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l535_53504


namespace NUMINAMATH_CALUDE_steven_name_day_l535_53546

def wordsOnDay (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day (n : ℕ) : wordsOnDay n = 44 ↔ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_steven_name_day_l535_53546


namespace NUMINAMATH_CALUDE_modular_inverse_three_mod_seventeen_l535_53561

theorem modular_inverse_three_mod_seventeen :
  ∃! x : ℕ, x ≤ 16 ∧ (3 * x) % 17 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_three_mod_seventeen_l535_53561


namespace NUMINAMATH_CALUDE_M_equals_P_l535_53537

/-- Set M defined as {x | x = a² + 1, a ∈ ℝ} -/
def M : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}

/-- Set P defined as {y | y = b² - 4b + 5, b ∈ ℝ} -/
def P : Set ℝ := {y | ∃ b : ℝ, y = b^2 - 4*b + 5}

/-- Theorem stating that M = P -/
theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_M_equals_P_l535_53537


namespace NUMINAMATH_CALUDE_final_distance_after_checkpoints_l535_53500

/-- Represents the state of a car on the highway -/
structure CarState where
  position : ℝ
  speed : ℝ

/-- Represents a checkpoint on the highway -/
structure Checkpoint where
  position : ℝ
  new_speed : ℝ

/-- Updates the car state after passing a checkpoint -/
def update_car_state (car : CarState) (checkpoint : Checkpoint) : CarState :=
  { position := checkpoint.position, speed := checkpoint.new_speed }

/-- Calculates the final distance between two cars after passing checkpoints -/
def final_distance (initial_distance : ℝ) (initial_speed : ℝ) (checkpoints : List Checkpoint) : ℝ :=
  sorry

/-- Theorem stating the final distance between the cars -/
theorem final_distance_after_checkpoints :
  let initial_distance := 100
  let initial_speed := 60
  let checkpoints := [
    { position := 1000, new_speed := 80 },
    { position := 2000, new_speed := 100 },
    { position := 3000, new_speed := 120 }
  ]
  final_distance initial_distance initial_speed checkpoints = 200 := by
  sorry

end NUMINAMATH_CALUDE_final_distance_after_checkpoints_l535_53500


namespace NUMINAMATH_CALUDE_lg_calculation_l535_53543

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_calculation : lg 25 - 2 * lg (1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_calculation_l535_53543


namespace NUMINAMATH_CALUDE_triangle_circle_areas_l535_53507

theorem triangle_circle_areas (r s t : ℝ) : 
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  r > 0 →
  s > 0 →
  t > 0 →
  π * r^2 + π * s^2 + π * t^2 = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_triangle_circle_areas_l535_53507


namespace NUMINAMATH_CALUDE_smallest_age_difference_l535_53538

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : 0 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ units ∧ units ≤ 9

/-- Calculates the value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Reverses the digits of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  is_valid := by
    simp [n.is_valid]

/-- The difference between two natural numbers -/
def diff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

theorem smallest_age_difference :
  ∀ (mrs_age : TwoDigitNumber),
    diff (TwoDigitNumber.value mrs_age) (TwoDigitNumber.value (TwoDigitNumber.reverse mrs_age)) ≥ 9 ∧
    ∃ (age : TwoDigitNumber),
      diff (TwoDigitNumber.value age) (TwoDigitNumber.value (TwoDigitNumber.reverse age)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_age_difference_l535_53538


namespace NUMINAMATH_CALUDE_abs_neg_five_plus_three_l535_53531

theorem abs_neg_five_plus_three : |(-5 + 3)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_plus_three_l535_53531


namespace NUMINAMATH_CALUDE_range_of_a_l535_53526

theorem range_of_a (a : ℝ) : 
  (∀ x, 2*x^2 - x - 1 ≤ 0 → x^2 - (2*a-1)*x + a*(a-1) ≤ 0) ∧ 
  (∃ x, 2*x^2 - x - 1 ≤ 0 ∧ x^2 - (2*a-1)*x + a*(a-1) > 0) →
  1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l535_53526


namespace NUMINAMATH_CALUDE_rational_inequality_l535_53586

theorem rational_inequality (a b : ℚ) (h1 : a + b > 0) (h2 : a * b < 0) :
  a > 0 ∧ b < 0 ∧ |a| > |b| := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_l535_53586


namespace NUMINAMATH_CALUDE_unique_consecutive_digit_square_swap_l535_53589

/-- A function that checks if a number is formed by four consecutive digits -/
def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ a : ℕ, n = 1000 * a + 100 * (a + 1) + 10 * (a + 2) + (a + 3)

/-- A function that swaps the first two digits of a four-digit number -/
def swap_first_two_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let last_two := n % 100
  1000 * d2 + 100 * d1 + last_two

/-- The main theorem stating that 3456 is the only number satisfying the conditions -/
theorem unique_consecutive_digit_square_swap :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    (is_consecutive_digits n ∧ ∃ m : ℕ, swap_first_two_digits n = m ^ 2) ↔ n = 3456 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_digit_square_swap_l535_53589


namespace NUMINAMATH_CALUDE_same_combination_probability_l535_53511

/-- Represents the number of candies of each color in the jar -/
structure JarContents where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the probability of two people picking the same color combination -/
def probability_same_combination (jar : JarContents) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given jar contents -/
theorem same_combination_probability :
  let jar : JarContents := { red := 12, blue := 12, green := 6 }
  probability_same_combination jar = 2783 / 847525 := by
  sorry

end NUMINAMATH_CALUDE_same_combination_probability_l535_53511


namespace NUMINAMATH_CALUDE_right_triangle_longest_altitudes_sum_l535_53585

theorem right_triangle_longest_altitudes_sum (a b c : ℝ) : 
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 → 
  (max a b + min a b) = 21 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_longest_altitudes_sum_l535_53585


namespace NUMINAMATH_CALUDE_william_riding_time_l535_53520

theorem william_riding_time :
  let max_daily_time : ℝ := 6
  let total_days : ℕ := 6
  let max_time_days : ℕ := 2
  let min_time_days : ℕ := 2
  let half_time_days : ℕ := 2
  let min_daily_time : ℝ := 1.5

  max_time_days * max_daily_time +
  min_time_days * min_daily_time +
  half_time_days * (max_daily_time / 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_william_riding_time_l535_53520


namespace NUMINAMATH_CALUDE_point_x_coordinate_l535_53528

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_x_coordinate 
  (l : Line) 
  (p : Point) 
  (h1 : l.slope = 3.8666666666666667)
  (h2 : l.yIntercept = 20)
  (h3 : p.y = 600)
  (h4 : pointOnLine p l) :
  p.x = 150 :=
sorry

end NUMINAMATH_CALUDE_point_x_coordinate_l535_53528


namespace NUMINAMATH_CALUDE_division_equation_l535_53544

theorem division_equation : (786 * 74) / 30 = 1938.8 := by
  sorry

end NUMINAMATH_CALUDE_division_equation_l535_53544


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l535_53524

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmingScenario where
  v_m : ℝ  -- Speed of the man in still water (km/h)
  v_s : ℝ  -- Speed of the stream (km/h)

/-- Theorem stating that given the downstream and upstream swimming distances and times,
    the speed of the swimmer in still water is 12 km/h. -/
theorem swimmer_speed_in_still_water 
  (scenario : SwimmingScenario)
  (h_downstream : (scenario.v_m + scenario.v_s) * 3 = 54)
  (h_upstream : (scenario.v_m - scenario.v_s) * 3 = 18) :
  scenario.v_m = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l535_53524


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l535_53565

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity : ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y = Real.sqrt x ∧
  (∃ (m : ℝ), m * (x + 1) = y ∧ m = 1 / (2 * Real.sqrt x))) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = 1) →
  (a^2 + b^2) / a = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l535_53565


namespace NUMINAMATH_CALUDE_tetrahedron_volume_EFGH_l535_53548

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (EF EG EH FG FH GH : ℝ) : ℝ :=
  sorry

/-- Theorem: The volume of tetrahedron EFGH with given edge lengths is √3/2 -/
theorem tetrahedron_volume_EFGH :
  tetrahedron_volume 5 (3 * Real.sqrt 2) (2 * Real.sqrt 3) 4 (Real.sqrt 37) 3 = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_EFGH_l535_53548


namespace NUMINAMATH_CALUDE_class_size_l535_53533

/-- The number of students who like art -/
def art_students : ℕ := 35

/-- The number of students who like music -/
def music_students : ℕ := 32

/-- The number of students who like both art and music -/
def both_students : ℕ := 19

/-- The total number of students in the class -/
def total_students : ℕ := art_students + music_students - both_students

theorem class_size :
  total_students = 48 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l535_53533


namespace NUMINAMATH_CALUDE_polynomial_value_l535_53541

theorem polynomial_value (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) :
  -2 * x^2 - 3 * y + 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l535_53541


namespace NUMINAMATH_CALUDE_problem_solution_l535_53595

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 4 * x

theorem problem_solution :
  -- Part 1
  (∀ x : ℝ, f 2 x ≥ 2 * x + 1 ↔ x ∈ Set.Ici (-1)) ∧
  -- Part 2
  (∀ a : ℝ, a > 0 →
    (∀ x : ℝ, x ∈ Set.Ioi (-2) → f a (2 * x) > 7 * x + a^2 - 3) →
    a ∈ Set.Ioo 0 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l535_53595


namespace NUMINAMATH_CALUDE_carnation_bouquet_problem_l535_53547

/-- Given 3 bouquets of carnations with known quantities in the first and third bouquets,
    and a known average, prove the quantity in the second bouquet. -/
theorem carnation_bouquet_problem (b1 b3 avg : ℕ) (h1 : b1 = 9) (h3 : b3 = 13) (havg : avg = 12) :
  ∃ b2 : ℕ, (b1 + b2 + b3) / 3 = avg ∧ b2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_carnation_bouquet_problem_l535_53547


namespace NUMINAMATH_CALUDE_cuboid_volume_l535_53502

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : a * c = 5) (h3 : b * c = 15) : 
  a * b * c = 15 := by
sorry

end NUMINAMATH_CALUDE_cuboid_volume_l535_53502


namespace NUMINAMATH_CALUDE_wine_drinkers_l535_53518

theorem wine_drinkers (soda : Nat) (both : Nat) (total : Nat) (h1 : soda = 22) (h2 : both = 17) (h3 : total = 31) :
  ∃ (wine : Nat), wine + soda - both = total ∧ wine = 26 := by
  sorry

end NUMINAMATH_CALUDE_wine_drinkers_l535_53518


namespace NUMINAMATH_CALUDE_trapezoid_ratio_theorem_l535_53567

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the area of a triangle given its vertices -/
def triangleArea (p1 p2 p3 : Point2D) : ℝ := sorry

/-- Checks if a point is inside a trapezoid -/
def isInside (p : Point2D) (t : Trapezoid) : Prop := sorry

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point2D) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def arePerpendicular (p1 p2 p3 p4 : Point2D) : Prop := sorry

theorem trapezoid_ratio_theorem (ABCD : Trapezoid) (P : Point2D) :
  isInside P ABCD →
  areParallel ABCD.A ABCD.B ABCD.C ABCD.D →
  arePerpendicular ABCD.A ABCD.D ABCD.C ABCD.D →
  triangleArea P ABCD.C ABCD.D = 2 →
  triangleArea P ABCD.A ABCD.D = 4 →
  triangleArea P ABCD.A ABCD.B = 8 →
  triangleArea P ABCD.B ABCD.C = 6 →
  arePerpendicular P ABCD.C P ABCD.D →
  (ABCD.A.x - ABCD.D.x) / (ABCD.C.x - ABCD.D.x) = 4 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_ratio_theorem_l535_53567


namespace NUMINAMATH_CALUDE_min_value_implies_a_value_l535_53509

/-- The function f(x) = x^2 + ax - 1 has a minimum value of -2 on the interval [0, 3] -/
def has_min_value_neg_two (a : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, x^2 + a*x - 1 ≥ x₀^2 + a*x₀ - 1 ∧ x₀^2 + a*x₀ - 1 = -2

/-- If f(x) = x^2 + ax - 1 has a minimum value of -2 on [0, 3], then a = -10/3 -/
theorem min_value_implies_a_value (a : ℝ) :
  has_min_value_neg_two a → a = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_value_l535_53509


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l535_53525

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (∃ k : ℤ, a * b + c * d = k * (a - c)) ↔ (∃ m : ℤ, a * d + b * c = m * (a - c)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l535_53525


namespace NUMINAMATH_CALUDE_min_value_theorem_l535_53593

theorem min_value_theorem (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (z w : ℝ), z^2 + w^2 = 2 → |z| ≠ |w| →
    1 / (z + w)^2 + 1 / (z - w)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l535_53593


namespace NUMINAMATH_CALUDE_intersection_sum_l535_53534

/-- Given two functions h and j that intersect at specific points, 
    this theorem proves that the sum of coordinates where h(3x) and 3j(x) 
    intersect is 22. -/
theorem intersection_sum (h j : ℝ → ℝ) : 
  (h 3 = j 3 ∧ h 3 = 3) →
  (h 6 = j 6 ∧ h 6 = 9) →
  (h 9 = j 9 ∧ h 9 = 18) →
  (h 12 = j 12 ∧ h 12 = 18) →
  ∃ x y : ℝ, h (3 * x) = 3 * j x ∧ h (3 * x) = y ∧ x + y = 22 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l535_53534


namespace NUMINAMATH_CALUDE_robotics_club_subjects_l535_53522

theorem robotics_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (cs : ℕ) 
  (math_physics : ℕ) (math_cs : ℕ) (physics_cs : ℕ) (all_three : ℕ) : 
  total = 60 ∧ 
  math = 42 ∧ 
  physics = 35 ∧ 
  cs = 15 ∧ 
  math_physics = 25 ∧ 
  math_cs = 10 ∧ 
  physics_cs = 5 ∧ 
  all_three = 4 → 
  total - (math + physics + cs - math_physics - math_cs - physics_cs + all_three) = 0 :=
by sorry

end NUMINAMATH_CALUDE_robotics_club_subjects_l535_53522


namespace NUMINAMATH_CALUDE_total_pencils_after_operations_l535_53529

/-- 
Given:
- There are initially 43 pencils in a drawer
- There are initially 19 pencils on a desk
- 16 pencils are added to the desk
- 7 pencils are removed from the desk

Prove that the total number of pencils after these operations is 71.
-/
theorem total_pencils_after_operations : 
  ∀ (drawer_initial desk_initial added removed : ℕ),
    drawer_initial = 43 →
    desk_initial = 19 →
    added = 16 →
    removed = 7 →
    drawer_initial + (desk_initial + added - removed) = 71 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pencils_after_operations_l535_53529


namespace NUMINAMATH_CALUDE_impossibility_of_distinct_differences_l535_53584

theorem impossibility_of_distinct_differences : ¬ ∃ (a : Fin 2010 → Fin 2010),
  Function.Injective a ∧ 
  (∀ (i j : Fin 2010), i ≠ j → |a i - i| ≠ |a j - j|) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_distinct_differences_l535_53584


namespace NUMINAMATH_CALUDE_not_all_even_numbers_representable_l535_53527

theorem not_all_even_numbers_representable :
  ∃ k : ℕ, k > 1000 ∧ k % 2 = 0 ∧
  ∀ m n : ℕ, k ≠ n * (n + 1) * (n + 2) - m * (m + 1) :=
by sorry

end NUMINAMATH_CALUDE_not_all_even_numbers_representable_l535_53527


namespace NUMINAMATH_CALUDE_functional_equation_identity_l535_53539

open Function Real

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) →
  (∀ y : ℝ, f y = y) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l535_53539


namespace NUMINAMATH_CALUDE_plane_equation_correct_l535_53560

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space represented by the equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- The point where the perpendicular meets the plane -/
def perpendicularPoint : Point3D := ⟨10, -2, 5⟩

/-- The plane in question -/
def targetPlane : Plane := ⟨10, -2, 5, -129⟩

/-- Vector from origin to perpendicularPoint -/
def normalVector : Point3D := perpendicularPoint

theorem plane_equation_correct :
  (∀ (p : Point3D), p.liesOn targetPlane ↔ 
    (p.x - perpendicularPoint.x) * normalVector.x + 
    (p.y - perpendicularPoint.y) * normalVector.y + 
    (p.z - perpendicularPoint.z) * normalVector.z = 0) ∧
  perpendicularPoint.liesOn targetPlane :=
sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l535_53560


namespace NUMINAMATH_CALUDE_earrings_sold_count_l535_53576

def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earrings_price : ℕ := 10
def ensemble_price : ℕ := 45

def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def ensembles_sold : ℕ := 2

def total_sales : ℕ := 565

theorem earrings_sold_count :
  ∃ (x : ℕ), 
    necklace_price * necklaces_sold + 
    bracelet_price * bracelets_sold + 
    earrings_price * x + 
    ensemble_price * ensembles_sold = total_sales ∧
    x = 20 := by sorry

end NUMINAMATH_CALUDE_earrings_sold_count_l535_53576


namespace NUMINAMATH_CALUDE_calculation_proof_l535_53515

theorem calculation_proof :
  (Real.sqrt 8 - Real.sqrt 2 - Real.sqrt (1/3) * Real.sqrt 6 = 0) ∧
  (Real.sqrt 15 / Real.sqrt 3 + (Real.sqrt 5 - 1)^2 = 6 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l535_53515


namespace NUMINAMATH_CALUDE_quadratic_decreasing_iff_a_in_range_l535_53566

/-- A quadratic function f(x) = ax^2 + 2(a-3)x + 1 is decreasing on [-2, +∞) if and only if a ∈ [-3, 0] -/
theorem quadratic_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, -2 ≤ x ∧ x < y → (a*x^2 + 2*(a-3)*x + 1) > (a*y^2 + 2*(a-3)*y + 1)) ↔ 
  -3 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_iff_a_in_range_l535_53566


namespace NUMINAMATH_CALUDE_charlie_ate_fifteen_cookies_l535_53521

/-- The number of cookies eaten by Charlie's family -/
def total_cookies : ℕ := 30

/-- The number of cookies eaten by Charlie's father -/
def father_cookies : ℕ := 10

/-- The number of cookies eaten by Charlie's mother -/
def mother_cookies : ℕ := 5

/-- Charlie's cookies -/
def charlie_cookies : ℕ := total_cookies - (father_cookies + mother_cookies)

theorem charlie_ate_fifteen_cookies : charlie_cookies = 15 := by
  sorry

end NUMINAMATH_CALUDE_charlie_ate_fifteen_cookies_l535_53521


namespace NUMINAMATH_CALUDE_klinker_age_problem_l535_53587

/-- The age difference between Mr. Klinker and his daughter remains constant -/
theorem klinker_age_problem (klinker_age : ℕ) (daughter_age : ℕ) (years : ℕ) :
  klinker_age = 47 →
  daughter_age = 13 →
  klinker_age + years = 3 * (daughter_age + years) →
  years = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_klinker_age_problem_l535_53587


namespace NUMINAMATH_CALUDE_cake_eaten_after_six_trips_l535_53503

/-- The fraction of cake eaten after n trips, given that 1/3 is eaten on the first trip
    and half of the remaining cake is eaten on each subsequent trip -/
def cakeEaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else 1/3 + (1 - 1/3) * (1 - (1/2)^(n-1))

/-- The theorem stating that after 6 trips, 47/48 of the cake is eaten -/
theorem cake_eaten_after_six_trips :
  cakeEaten 6 = 47/48 := by sorry

end NUMINAMATH_CALUDE_cake_eaten_after_six_trips_l535_53503


namespace NUMINAMATH_CALUDE_three_positions_from_eight_people_l535_53519

theorem three_positions_from_eight_people :
  (8 : ℕ).descFactorial 3 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_eight_people_l535_53519


namespace NUMINAMATH_CALUDE_marsha_pay_per_mile_l535_53551

/-- Calculates the pay per mile for a delivery driver given their daily pay and distances driven --/
def pay_per_mile (daily_pay : ℚ) (first_distance second_distance : ℚ) : ℚ :=
  let third_distance := second_distance / 2
  let total_distance := first_distance + second_distance + third_distance
  daily_pay / total_distance

/-- Proves that Marsha's pay per mile is $2 given the specified conditions --/
theorem marsha_pay_per_mile :
  pay_per_mile 104 10 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marsha_pay_per_mile_l535_53551


namespace NUMINAMATH_CALUDE_evaluate_expression_l535_53563

theorem evaluate_expression : ((3^1 - 2 + 6^2 - 0)⁻¹ * 3 : ℚ) = 3 / 37 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l535_53563


namespace NUMINAMATH_CALUDE_max_value_theorem_equality_condition_l535_53568

theorem max_value_theorem (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 2 - x - 4 / x = -2 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_equality_condition_l535_53568


namespace NUMINAMATH_CALUDE_polynomial_remainder_l535_53517

/-- Given a polynomial Q(x) such that Q(17) = 53 and Q(53) = 17,
    the remainder when Q(x) is divided by (x - 17)(x - 53) is -x + 70 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 17 = 53) (h2 : Q 53 = 17) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 17) * (x - 53) * R x + (-x + 70) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l535_53517


namespace NUMINAMATH_CALUDE_candy_distribution_l535_53514

theorem candy_distribution (A B : ℕ) 
  (h1 : 7 * A = B + 12) 
  (h2 : 3 * A = B - 20) : 
  A + B = 52 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l535_53514


namespace NUMINAMATH_CALUDE_p_q_ratio_l535_53590

/-- The number of balls -/
def num_balls : ℕ := 30

/-- The number of bins -/
def num_bins : ℕ := 10

/-- The probability that two bins each have 2 balls and the other eight bins have 3 balls each -/
def p : ℚ := (Nat.choose num_bins 2) * (Nat.choose num_balls 2) * (Nat.choose (num_balls - 2) 2) / (Nat.choose num_balls num_balls)

/-- The probability that all bins have 3 balls each -/
def q : ℚ := 1 / (Nat.choose num_balls num_balls)

/-- The theorem stating that the ratio of p to q is 7371 -/
theorem p_q_ratio : p / q = 7371 := by sorry

end NUMINAMATH_CALUDE_p_q_ratio_l535_53590


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l535_53516

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l535_53516


namespace NUMINAMATH_CALUDE_right_angle_in_clerts_l535_53506

/-- In a system where a full circle is measured as 500 units, a right angle is 125 units. -/
theorem right_angle_in_clerts (full_circle : ℕ) (right_angle : ℕ) 
  (h1 : full_circle = 500) 
  (h2 : right_angle = full_circle / 4) : 
  right_angle = 125 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_in_clerts_l535_53506


namespace NUMINAMATH_CALUDE_ethanol_percentage_fuel_B_l535_53523

-- Define the constants
def tank_capacity : ℝ := 212
def fuel_A_ethanol_percentage : ℝ := 0.12
def fuel_A_volume : ℝ := 98
def total_ethanol : ℝ := 30

-- Define the theorem
theorem ethanol_percentage_fuel_B :
  let ethanol_A := fuel_A_ethanol_percentage * fuel_A_volume
  let ethanol_B := total_ethanol - ethanol_A
  let fuel_B_volume := tank_capacity - fuel_A_volume
  (ethanol_B / fuel_B_volume) * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_percentage_fuel_B_l535_53523


namespace NUMINAMATH_CALUDE_log_ratio_squared_l535_53588

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h2 : x * y = 243) :
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l535_53588


namespace NUMINAMATH_CALUDE_twelve_people_three_handshakes_l535_53591

/-- A handshake graph represents a group of people and their handshakes. -/
structure HandshakeGraph where
  n : ℕ  -- number of people
  k : ℕ  -- number of handshakes per person
  is_valid : n > 0 ∧ k < n

/-- Count of distinct handshaking arrangements for a given HandshakeGraph. -/
def countArrangements (g : HandshakeGraph) : ℕ :=
  sorry

/-- The main theorem to be proved. -/
theorem twelve_people_three_handshakes :
  ∃ (g : HandshakeGraph), g.n = 12 ∧ g.k = 3 ∧ countArrangements g % 1000 = 250 :=
sorry

end NUMINAMATH_CALUDE_twelve_people_three_handshakes_l535_53591


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l535_53512

theorem expression_simplification_and_evaluation :
  let x : ℤ := -1
  let original_expression := (x * (x + 1)) - ((x + 2) * (2 - x)) - (2 * (x + 2)^2)
  let simplified_expression := -2 * x^2 - 9 * x - 12
  original_expression = simplified_expression ∧ simplified_expression = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l535_53512


namespace NUMINAMATH_CALUDE_sara_quarters_l535_53583

/-- The number of quarters Sara initially had -/
def initial_quarters : ℕ := 783

/-- The number of quarters Sara's dad borrowed -/
def borrowed_quarters : ℕ := 271

/-- The number of quarters Sara has now -/
def remaining_quarters : ℕ := initial_quarters - borrowed_quarters

theorem sara_quarters : remaining_quarters = 512 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l535_53583


namespace NUMINAMATH_CALUDE_nilpotent_matrix_powers_zero_l535_53596

theorem nilpotent_matrix_powers_zero 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 ∧ A ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_powers_zero_l535_53596


namespace NUMINAMATH_CALUDE_characterization_of_valid_a_l535_53592

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f x + a * ⌊y⌋

/-- The set of valid values for a -/
def ValidA : Set ℝ :=
  {a | ∃ n : ℤ, a = -(n^2 : ℝ)}

/-- The main theorem stating the characterization of valid a values -/
theorem characterization_of_valid_a :
  ∀ a : ℝ, (∃ f : ℝ → ℝ, SatisfiesEquation f a) ↔ a ∈ ValidA :=
sorry

end NUMINAMATH_CALUDE_characterization_of_valid_a_l535_53592


namespace NUMINAMATH_CALUDE_optimal_time_correct_l535_53532

/-- The optimal time for Vasya and Petya to cover the distance -/
def optimal_time : ℝ := 0.5

/-- The total distance to be covered -/
def total_distance : ℝ := 3

/-- Vasya's running speed -/
def vasya_run_speed : ℝ := 4

/-- Vasya's skating speed -/
def vasya_skate_speed : ℝ := 8

/-- Petya's running speed -/
def petya_run_speed : ℝ := 5

/-- Petya's skating speed -/
def petya_skate_speed : ℝ := 10

/-- Theorem stating that the optimal time is correct -/
theorem optimal_time_correct :
  ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ total_distance ∧
    (x / vasya_skate_speed + (total_distance - x) / vasya_run_speed = optimal_time) ∧
    ((total_distance - x) / petya_skate_speed + x / petya_run_speed = optimal_time) ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_distance →
      max (y / vasya_skate_speed + (total_distance - y) / vasya_run_speed)
          ((total_distance - y) / petya_skate_speed + y / petya_run_speed) ≥ optimal_time :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_time_correct_l535_53532


namespace NUMINAMATH_CALUDE_field_trip_bus_capacity_l535_53540

theorem field_trip_bus_capacity 
  (total_vehicles : Nat) 
  (people_per_van : Nat) 
  (total_people : Nat) 
  (num_vans : Nat) 
  (num_buses : Nat) 
  (h1 : total_vehicles = num_vans + num_buses)
  (h2 : num_vans = 2)
  (h3 : num_buses = 3)
  (h4 : people_per_van = 8)
  (h5 : total_people = 76) :
  (total_people - num_vans * people_per_van) / num_buses = 20 := by
sorry

end NUMINAMATH_CALUDE_field_trip_bus_capacity_l535_53540


namespace NUMINAMATH_CALUDE_smallest_result_l535_53572

def S : Set Int := {-10, -4, 0, 2, 7}

theorem smallest_result (x y : Int) (hx : x ∈ S) (hy : y ∈ S) :
  (x * y ≥ -70 ∧ x + y ≥ -70) ∧ ∃ a b : Int, a ∈ S ∧ b ∈ S ∧ (a * b = -70 ∨ a + b = -70) :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l535_53572


namespace NUMINAMATH_CALUDE_expression_value_at_four_l535_53581

theorem expression_value_at_four :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 4 = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_four_l535_53581


namespace NUMINAMATH_CALUDE_bruce_age_bruce_current_age_l535_53508

theorem bruce_age : ℕ → Prop :=
  fun b =>
    let son_age : ℕ := 8
    let future_years : ℕ := 6
    (b + future_years = 3 * (son_age + future_years)) →
    b = 36

-- Proof
theorem bruce_current_age : ∃ b : ℕ, bruce_age b :=
  sorry

end NUMINAMATH_CALUDE_bruce_age_bruce_current_age_l535_53508


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l535_53549

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l535_53549


namespace NUMINAMATH_CALUDE_min_value_expression_l535_53594

theorem min_value_expression (x : ℝ) (h : 0 ≤ x ∧ x < 4) :
  ∃ (min : ℝ), min = Real.sqrt 5 ∧
  ∀ y, 0 ≤ y ∧ y < 4 → (y^2 + 2*y + 6) / (2*y + 2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l535_53594


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l535_53562

theorem fraction_sum_inequality (a b c : ℝ) :
  a / (a + 2*b + c) + b / (a + b + 2*c) + c / (2*a + b + c) ≥ 3/4 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l535_53562


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l535_53556

/-- The latus rectum of a parabola x^2 = -2y is y = 1/2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  x^2 = -2*y → (∃ (x₀ : ℝ), x₀^2 = -2*(1/2) ∧ x₀ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l535_53556


namespace NUMINAMATH_CALUDE_divisibility_properties_l535_53597

theorem divisibility_properties (a m n : ℕ) (ha : a ≥ 2) (hm : m > 0) (hn : n > 0) (h_div : m ∣ n) :
  (∃ k, a^n - 1 = k * (a^m - 1)) ∧
  ((∃ k, a^n + 1 = k * (a^m + 1)) ↔ Odd (n / m)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l535_53597


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_is_correct_l535_53599

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | abs x ≤ 2}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Define the complement of A ∩ B in ℝ
def complement_A_intersect_B : Set ℝ := {x : ℝ | x < -2 ∨ x > 0}

-- State the theorem
theorem complement_A_intersect_B_is_correct :
  (Set.univ : Set ℝ) \ (A ∩ B) = complement_A_intersect_B := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_is_correct_l535_53599


namespace NUMINAMATH_CALUDE_square_is_rectangle_and_rhombus_l535_53559

-- Define a quadrilateral
structure Quadrilateral :=
  (sides : Fin 4 → ℝ)
  (angles : Fin 4 → ℝ)

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, q.angles i = 90 ∧ q.sides i = q.sides ((i + 2) % 4)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  is_rectangle q ∧ is_rhombus q

-- Theorem statement
theorem square_is_rectangle_and_rhombus (q : Quadrilateral) :
  is_square q → is_rectangle q ∧ is_rhombus q :=
sorry

end NUMINAMATH_CALUDE_square_is_rectangle_and_rhombus_l535_53559


namespace NUMINAMATH_CALUDE_second_valid_number_is_068_l535_53535

/-- Represents a random number table as a list of natural numbers. -/
def RandomNumberTable : List ℕ := [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76]

/-- Represents the total number of units. -/
def TotalUnits : ℕ := 200

/-- Represents the starting column in the random number table. -/
def StartColumn : ℕ := 5

/-- Checks if a number is valid (i.e., between 1 and TotalUnits). -/
def isValidNumber (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ TotalUnits

/-- Finds the nth valid number in the random number table. -/
def nthValidNumber (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the second valid number is 068. -/
theorem second_valid_number_is_068 : nthValidNumber 2 = 68 := by sorry

end NUMINAMATH_CALUDE_second_valid_number_is_068_l535_53535


namespace NUMINAMATH_CALUDE_intersection_properties_l535_53564

/-- Two lines intersecting at point P -/
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x - y - 3 * m + 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := x + m * y - 3 * m - 1 = 0

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4

/-- Point P satisfies both lines -/
def point_P (m : ℝ) (x y : ℝ) : Prop := line1 m x y ∧ line2 m x y

/-- AB is a chord of circle C with length 2√3 -/
def chord_AB (xa ya xb yb : ℝ) : Prop :=
  circle_C xa ya ∧ circle_C xb yb ∧ (xa - xb)^2 + (ya - yb)^2 = 12

/-- Q is the midpoint of AB -/
def midpoint_Q (xa ya xb yb xq yq : ℝ) : Prop :=
  xq = (xa + xb) / 2 ∧ yq = (ya + yb) / 2

theorem intersection_properties (m : ℝ) :
  ∃ x y xa ya xb yb xq yq,
    point_P m x y ∧
    chord_AB xa ya xb yb ∧
    midpoint_Q xa ya xb yb xq yq →
    (¬ circle_C x y) ∧  -- P lies outside circle C
    (∃ pq_max, pq_max = 6 + Real.sqrt 2 ∧
      ∀ x' y', point_P m x' y' →
        ∀ xa' ya' xb' yb' xq' yq',
          chord_AB xa' ya' xb' yb' ∧
          midpoint_Q xa' ya' xb' yb' xq' yq' →
            ((x' - xq')^2 + (y' - yq')^2)^(1/2) ≤ pq_max) ∧  -- Max length of PQ
    (∃ pa_pb_min, pa_pb_min = 15 - 8 * Real.sqrt 2 ∧
      ∀ x' y', point_P m x' y' →
        ∀ xa' ya' xb' yb',
          chord_AB xa' ya' xb' yb' →
            (x' - xa') * (x' - xb') + (y' - ya') * (y' - yb') ≥ pa_pb_min)  -- Min value of PA · PB
  := by sorry

end NUMINAMATH_CALUDE_intersection_properties_l535_53564


namespace NUMINAMATH_CALUDE_max_intersections_sine_line_l535_53545

theorem max_intersections_sine_line (φ : ℝ) : 
  ∃ (n : ℕ), n ≤ 4 ∧ 
  (∀ (m : ℕ), (∃ (S : Finset ℝ), S.card = m ∧ 
    (∀ x ∈ S, x ∈ Set.Icc 0 Real.pi ∧ 3 * Real.sin (3 * x + φ) = 2)) → m ≤ n) :=
sorry

end NUMINAMATH_CALUDE_max_intersections_sine_line_l535_53545


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l535_53575

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt (x^2 + (y + 2)^2) = 10) ↔
  (y^2 / 25 + x^2 / 21 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l535_53575


namespace NUMINAMATH_CALUDE_triangle_with_prime_angles_exists_l535_53569

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem triangle_with_prime_angles_exists : ∃ p q r : ℕ, 
  isPrime p ∧ isPrime q ∧ isPrime r ∧ p + q + r = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_prime_angles_exists_l535_53569


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l535_53571

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 22 cm has a base of 8 cm. -/
theorem isosceles_triangle_base_length : ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 22 →
  base = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l535_53571


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l535_53557

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_focal_distance 
  (P : ℝ × ℝ) 
  (h_on_hyperbola : is_on_hyperbola P.1 P.2) 
  (h_left_distance : distance P left_focus = 3) : 
  distance P right_focus = 9 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l535_53557


namespace NUMINAMATH_CALUDE_min_value_of_expression_l535_53579

theorem min_value_of_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x*y/z + z*x/y + y*z/x) * (x/(y*z) + y/(z*x) + z/(x*y)) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l535_53579


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l535_53570

theorem quadratic_root_difference (x : ℝ) : 
  let roots := {r : ℝ | r^2 - 7*r + 11 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l535_53570


namespace NUMINAMATH_CALUDE_max_player_salary_l535_53550

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 25 →
  min_salary = 15000 →
  total_cap = 800000 →
  (n - 1) * min_salary + (total_cap - (n - 1) * min_salary) = 440000 :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l535_53550


namespace NUMINAMATH_CALUDE_square_root_calculations_l535_53501

theorem square_root_calculations :
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    Real.sqrt (a / b) * Real.sqrt c / Real.sqrt b = Real.sqrt (a * c / (b * b))) ∧
  (Real.sqrt (1 / 6) * Real.sqrt 96 / Real.sqrt 6 = 2 * Real.sqrt 6 / 3) ∧
  (Real.sqrt 80 - Real.sqrt 8 - Real.sqrt 45 + 4 * Real.sqrt (1 / 2) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_square_root_calculations_l535_53501


namespace NUMINAMATH_CALUDE_b_plus_c_equals_nine_l535_53578

theorem b_plus_c_equals_nine (a b c d : ℤ) 
  (h1 : a + b = 11) 
  (h2 : c + d = 3) 
  (h3 : a + d = 5) : 
  b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_nine_l535_53578


namespace NUMINAMATH_CALUDE_b_value_l535_53505

theorem b_value (b : ℚ) (h : b + b/4 - 1 = 3/2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l535_53505


namespace NUMINAMATH_CALUDE_sum_difference_odd_even_l535_53554

theorem sum_difference_odd_even : 
  let range := Finset.Icc 372 506
  let odd_sum := (range.filter (λ n => n % 2 = 1)).sum id
  let even_sum := (range.filter (λ n => n % 2 = 0)).sum id
  odd_sum - even_sum = 439 := by sorry

end NUMINAMATH_CALUDE_sum_difference_odd_even_l535_53554


namespace NUMINAMATH_CALUDE_largest_number_l535_53555

/-- Represents a number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a real number -/
noncomputable def toReal (x : RepeatingDecimal) : ℝ := sorry

/-- The numbers given in the problem -/
def a : ℝ := 9.12344
def b : RepeatingDecimal := ⟨9, [1, 2, 3], [4]⟩
def c : RepeatingDecimal := ⟨9, [1, 2], [3, 4]⟩
def d : RepeatingDecimal := ⟨9, [1], [2, 3, 4]⟩
def e : RepeatingDecimal := ⟨9, [], [1, 2, 3, 4]⟩

/-- Theorem stating that 9.123̄4 is the largest among the given numbers -/
theorem largest_number : 
  toReal b > a ∧ 
  toReal b > toReal c ∧ 
  toReal b > toReal d ∧ 
  toReal b > toReal e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l535_53555


namespace NUMINAMATH_CALUDE_externally_tangent_circles_distance_l535_53582

/-- The distance between the centers of two externally tangent circles
    is equal to the sum of their radii -/
theorem externally_tangent_circles_distance
  (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 2)
  (h₂ : r₂ = 3)
  (h_tangent : d = r₁ + r₂) :
  d = 5 := by sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_distance_l535_53582


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_l535_53553

theorem numerator_increase_percentage (P : ℚ) : 
  (1 + P / 100) / ((3 / 4) * 12) = 2 / 15 → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_l535_53553


namespace NUMINAMATH_CALUDE_miyeon_gets_48_sheets_l535_53580

/-- The number of sheets Miyeon gets given the conditions of the paper sharing problem -/
def miyeon_sheets (total_sheets : ℕ) (pink_sheets : ℕ) : ℕ :=
  (total_sheets - pink_sheets) / 2 + pink_sheets

/-- Theorem stating that Miyeon gets 48 sheets under the given conditions -/
theorem miyeon_gets_48_sheets :
  miyeon_sheets 85 11 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_miyeon_gets_48_sheets_l535_53580


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l535_53558

theorem function_inequality_implies_a_bound 
  (f g : ℝ → ℝ) 
  (h_f : ∀ x, f x = |x - a| + a) 
  (h_g : ∀ x, g x = 4 - x^2) 
  (h_exists : ∃ x, g x ≥ f x) : 
  a ≤ 17/8 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l535_53558


namespace NUMINAMATH_CALUDE_lucas_class_size_l535_53574

theorem lucas_class_size (n : ℕ) (best_rank : ℕ) (worst_rank : ℕ)
  (h1 : best_rank = 30)
  (h2 : worst_rank = 45)
  (h3 : n = best_rank + worst_rank - 1) :
  n = 74 := by
sorry

end NUMINAMATH_CALUDE_lucas_class_size_l535_53574


namespace NUMINAMATH_CALUDE_system_solution_implies_a_equals_five_l535_53510

theorem system_solution_implies_a_equals_five 
  (x y a : ℝ) 
  (eq1 : 2 * x - y = 1) 
  (eq2 : 3 * x + y = 2 * a - 1) 
  (eq3 : 2 * y - x = 4) : 
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_a_equals_five_l535_53510


namespace NUMINAMATH_CALUDE_plane_equation_proof_l535_53530

/-- A plane equation is represented by a tuple of integers (A, B, C, D) corresponding to the equation Ax + By + Cz + D = 0 --/
def PlaneEquation := (ℤ × ℤ × ℤ × ℤ)

/-- The given plane equation 3x - 2y + 4z = 10 --/
def given_plane : PlaneEquation := (3, -2, 4, -10)

/-- The point through which the new plane must pass --/
def point : (ℤ × ℤ × ℤ) := (2, -3, 5)

/-- Check if a plane equation passes through a given point --/
def passes_through (plane : PlaneEquation) (p : ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := plane
  let (x, y, z) := p
  A * x + B * y + C * z + D = 0

/-- Check if two plane equations are parallel --/
def is_parallel (plane1 plane2 : PlaneEquation) : Prop :=
  let (A1, B1, C1, _) := plane1
  let (A2, B2, C2, _) := plane2
  ∃ (k : ℚ), k ≠ 0 ∧ A1 = k * A2 ∧ B1 = k * B2 ∧ C1 = k * C2

/-- Check if the first coefficient of a plane equation is positive --/
def first_coeff_positive (plane : PlaneEquation) : Prop :=
  let (A, _, _, _) := plane
  A > 0

/-- Calculate the greatest common divisor of the absolute values of all coefficients --/
def gcd_of_coeffs (plane : PlaneEquation) : ℕ :=
  let (A, B, C, D) := plane
  Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D)))

theorem plane_equation_proof (solution : PlaneEquation) : 
  passes_through solution point ∧ 
  is_parallel solution given_plane ∧ 
  first_coeff_positive solution ∧ 
  gcd_of_coeffs solution = 1 ∧ 
  solution = (3, -2, 4, -32) := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l535_53530


namespace NUMINAMATH_CALUDE_mustard_at_second_table_l535_53513

/-- The amount of mustard found at each table and the total amount --/
def MustardProblem (total first second third : ℚ) : Prop :=
  total = first + second + third

theorem mustard_at_second_table :
  ∃ (second : ℚ), MustardProblem 0.88 0.25 second 0.38 ∧ second = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_mustard_at_second_table_l535_53513


namespace NUMINAMATH_CALUDE_system_solutions_l535_53577

-- Define the system of equations
def system (t y a : ℝ) : Prop :=
  (|t| - y = 1 - a^4 - a^4 * t^4) ∧ (t^2 + y^2 = 1)

-- Define the property of having multiple solutions
def has_multiple_solutions (a : ℝ) : Prop :=
  ∃ (t₁ y₁ t₂ y₂ : ℝ), t₁ ≠ t₂ ∧ system t₁ y₁ a ∧ system t₂ y₂ a

-- Define the property of having a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∀ (t y : ℝ), system t y a → t = 0 ∧ y = 1

-- Theorem statement
theorem system_solutions :
  (has_multiple_solutions 0) ∧
  (has_unique_solution (Real.sqrt (Real.sqrt 2))) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l535_53577
