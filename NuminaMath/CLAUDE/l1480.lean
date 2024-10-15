import Mathlib

namespace NUMINAMATH_CALUDE_last_digit_of_even_ten_digit_with_sum_89_l1480_148055

/-- A ten-digit integer -/
def TenDigitInt : Type := { n : ℕ // 1000000000 ≤ n ∧ n < 10000000000 }

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem last_digit_of_even_ten_digit_with_sum_89 (n : TenDigitInt) 
  (h_even : Even n.val)
  (h_sum : sum_of_digits n.val = 89) :
  n.val % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_even_ten_digit_with_sum_89_l1480_148055


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_in_5_12_13_triangle_l1480_148037

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a right triangle with circles at its vertices -/
structure TriangleWithCircles where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  is_right_triangle : side1^2 + side2^2 = hypotenuse^2
  circles_tangent : 
    circle1.radius + circle2.radius = side1 ∧
    circle2.radius + circle3.radius = side2 ∧
    circle1.radius + circle3.radius = hypotenuse

/-- The sum of the areas of the circles in a 5-12-13 right triangle with mutually tangent circles at its vertices is 81π -/
theorem sum_of_circle_areas_in_5_12_13_triangle (t : TriangleWithCircles) 
  (h1 : t.side1 = 5) (h2 : t.side2 = 12) (h3 : t.hypotenuse = 13) : 
  π * t.circle1.radius^2 + π * t.circle2.radius^2 + π * t.circle3.radius^2 = 81 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_in_5_12_13_triangle_l1480_148037


namespace NUMINAMATH_CALUDE_caramel_chews_theorem_l1480_148010

/-- Represents the distribution of candy bags -/
structure CandyDistribution where
  totalCandies : ℕ
  totalBags : ℕ
  heartsCount : ℕ
  kissesCount : ℕ
  jelliesCount : ℕ
  heartsExtra : ℕ
  jelliesMultiplier : ℚ

/-- Calculates the number of candies in caramel chews bags -/
def caramelChewsCandies (d : CandyDistribution) : ℕ :=
  let remainingBags := d.totalBags - (d.heartsCount + d.kissesCount + d.jelliesCount)
  let baseCandy := (d.totalCandies - d.heartsCount * d.heartsExtra) / d.totalBags
  remainingBags * baseCandy

/-- Theorem stating that for the given distribution, caramel chews bags contain 44 candies -/
theorem caramel_chews_theorem (d : CandyDistribution) 
  (h1 : d.totalCandies = 500)
  (h2 : d.totalBags = 20)
  (h3 : d.heartsCount = 6)
  (h4 : d.kissesCount = 8)
  (h5 : d.jelliesCount = 4)
  (h6 : d.heartsExtra = 2)
  (h7 : d.jelliesMultiplier = 3/2) :
  caramelChewsCandies d = 44 := by
  sorry

end NUMINAMATH_CALUDE_caramel_chews_theorem_l1480_148010


namespace NUMINAMATH_CALUDE_sqrt_nine_plus_sixteen_l1480_148078

theorem sqrt_nine_plus_sixteen : Real.sqrt (9 + 16) = 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_plus_sixteen_l1480_148078


namespace NUMINAMATH_CALUDE_line_equation_proof_l1480_148086

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line y = 4x + 3 -/
def given_line : Line :=
  { slope := 4, intercept := 3 }

/-- The point (1, 1) -/
def point : (ℝ × ℝ) :=
  (1, 1)

theorem line_equation_proof :
  ∃ (l : Line),
    parallel l given_line ∧
    passes_through l point.1 point.2 ∧
    l.slope = 4 ∧
    l.intercept = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1480_148086


namespace NUMINAMATH_CALUDE_number_and_square_relationship_l1480_148003

theorem number_and_square_relationship (n : ℕ) (h : n = 8) : n^2 + n = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_relationship_l1480_148003


namespace NUMINAMATH_CALUDE_festival_attendance_ratio_l1480_148040

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

end NUMINAMATH_CALUDE_festival_attendance_ratio_l1480_148040


namespace NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l1480_148091

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) (on_sand : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  on_sand = total - (swept_fraction * total).num → 
  on_sand = 28 := by
sorry

end NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l1480_148091


namespace NUMINAMATH_CALUDE_painting_frame_ratio_l1480_148070

theorem painting_frame_ratio {x l : ℝ} (h_positive : x > 0 ∧ l > 0) 
  (h_area_equality : (x + 2*l) * ((3/2)*x + 2*l) = 2 * (x * (3/2)*x)) :
  (x + 2*l) / ((3/2)*x + 2*l) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_painting_frame_ratio_l1480_148070


namespace NUMINAMATH_CALUDE_altitude_and_equidistant_lines_l1480_148064

/-- Given three points in a plane -/
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 4)
def C : ℝ × ℝ := (5, 7)

/-- Line l₁ containing the altitude from B to BC -/
def l₁ (x y : ℝ) : Prop := 7 * x + 3 * y - 10 = 0

/-- Lines l₂ passing through B with equal distances from A and C -/
def l₂₁ (y : ℝ) : Prop := y = 4
def l₂₂ (x y : ℝ) : Prop := 3 * x - 2 * y + 14 = 0

/-- Main theorem -/
theorem altitude_and_equidistant_lines :
  (∀ x y, l₁ x y ↔ (x - B.1) * (C.2 - B.2) = (y - B.2) * (C.1 - B.1)) ∧
  (∀ x y, (l₂₁ y ∨ l₂₂ x y) ↔ 
    ((x = B.1 ∧ y = B.2) ∨ 
     (abs ((y - A.2) - ((y - B.2) / (x - B.1)) * (A.1 - B.1)) = 
      abs ((y - C.2) - ((y - B.2) / (x - B.1)) * (C.1 - B.1))))) :=
sorry

end NUMINAMATH_CALUDE_altitude_and_equidistant_lines_l1480_148064


namespace NUMINAMATH_CALUDE_distance_center_to_endpoint_l1480_148054

/-- Given two points representing the endpoints of a circle's diameter,
    calculate the distance from the center of the circle to one of the endpoints. -/
theorem distance_center_to_endpoint
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (12, -8))
  (h2 : p2 = (-6, 4))
  : Real.sqrt ((12 - ((p1.1 + p2.1) / 2))^2 + (-8 - ((p1.2 + p2.2) / 2))^2) = Real.sqrt 117 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_endpoint_l1480_148054


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_11_l1480_148082

theorem smallest_five_digit_mod_11 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 11 = 9 → n ≥ 10000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_11_l1480_148082


namespace NUMINAMATH_CALUDE_trig_identities_l1480_148049

/-- Theorem: Trigonometric identities for specific angles -/
theorem trig_identities :
  (∃ (x y : ℝ), x = 263 * π / 180 ∧ y = 203 * π / 180 ∧
    Real.cos x * Real.cos y + Real.sin (83 * π / 180) * Real.sin (23 * π / 180) = 1/2) ∧
  (∃ (z : ℝ), z = 8 * π / 180 ∧
    (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin z) / Real.cos z =
    (Real.sqrt 6 + Real.sqrt 2) / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1480_148049


namespace NUMINAMATH_CALUDE_fermat_prime_l1480_148050

theorem fermat_prime (m : ℕ) (h : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) → Nat.Prime (2^(m+1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_l1480_148050


namespace NUMINAMATH_CALUDE_nine_ants_nine_trips_l1480_148025

/-- Represents the number of grains of rice that can be moved by a given number of ants in a given number of trips -/
def rice_moved (ants : ℕ) (trips : ℕ) : ℚ :=
  (24 : ℚ) * ants * trips / (12 * 6)

/-- Theorem stating that 9 ants can move 27 grains of rice in 9 trips -/
theorem nine_ants_nine_trips :
  rice_moved 9 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_ants_nine_trips_l1480_148025


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1480_148041

theorem complex_equation_solution (x y : ℝ) :
  (x + y * Complex.I) / (3 - 2 * Complex.I) = 1 + Complex.I →
  Complex.im (x + y * Complex.I) = 1 ∧ Complex.abs (x + y * Complex.I) = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1480_148041


namespace NUMINAMATH_CALUDE_comprehensive_survey_appropriate_for_grade9_vision_l1480_148030

/-- Represents the appropriateness of a survey method for a given scenario -/
inductive SurveyAppropriateness
  | Appropriate
  | Inappropriate

/-- Represents different survey methods -/
inductive SurveyMethod
  | Comprehensive
  | Sample

/-- Represents characteristics that can be surveyed -/
inductive Characteristic
  | Vision
  | EquipmentQuality

/-- Represents the size of a group being surveyed -/
inductive GroupSize
  | Large
  | Small

/-- Function to determine if a survey method is appropriate for a given characteristic and group size -/
def is_appropriate (method : SurveyMethod) (char : Characteristic) (size : GroupSize) : SurveyAppropriateness :=
  match method, char, size with
  | SurveyMethod.Comprehensive, Characteristic.Vision, GroupSize.Large => SurveyAppropriateness.Appropriate
  | _, _, _ => SurveyAppropriateness.Inappropriate

theorem comprehensive_survey_appropriate_for_grade9_vision :
  is_appropriate SurveyMethod.Comprehensive Characteristic.Vision GroupSize.Large = SurveyAppropriateness.Appropriate :=
by sorry

end NUMINAMATH_CALUDE_comprehensive_survey_appropriate_for_grade9_vision_l1480_148030


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l1480_148046

theorem greatest_integer_solution (x : ℝ) : 
  (((|x^2 - 2| - 7) * (|x + 3| - 5)) / (|x - 3| - |x - 1|) > 0) → 
  (∃ (n : ℤ), n ≤ x ∧ n ≤ 1 ∧ ∀ (m : ℤ), m ≤ x → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l1480_148046


namespace NUMINAMATH_CALUDE_expected_original_positions_l1480_148015

/-- The number of balls arranged in a circle -/
def numBalls : ℕ := 7

/-- The number of independent random transpositions -/
def numTranspositions : ℕ := 3

/-- The probability of a ball being in its original position after the transpositions -/
def probOriginalPosition : ℚ := 127 / 343

/-- The expected number of balls in their original positions after the transpositions -/
def expectedOriginalPositions : ℚ := numBalls * probOriginalPosition

theorem expected_original_positions :
  expectedOriginalPositions = 889 / 343 := by sorry

end NUMINAMATH_CALUDE_expected_original_positions_l1480_148015


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l1480_148009

theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  (2 * b - c) * Real.cos A = a * Real.cos C →
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l1480_148009


namespace NUMINAMATH_CALUDE_large_cube_edge_approx_l1480_148097

/-- The edge length of a smaller cube in centimeters -/
def small_cube_edge : ℝ := 20

/-- The approximate number of smaller cubes that fit in the larger cubical box -/
def num_small_cubes : ℝ := 125

/-- The approximate edge length of the larger cubical box in centimeters -/
def large_cube_edge : ℝ := 100

/-- Theorem stating that the edge length of the larger cubical box is approximately 100 cm -/
theorem large_cube_edge_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |large_cube_edge ^ 3 - num_small_cubes * small_cube_edge ^ 3| < ε * (num_small_cubes * small_cube_edge ^ 3) :=
sorry

end NUMINAMATH_CALUDE_large_cube_edge_approx_l1480_148097


namespace NUMINAMATH_CALUDE_cubic_function_extremum_value_l1480_148012

/-- A cubic function with an extremum at x = 1 and f(1) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_extremum_value (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → f a b 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extremum_value_l1480_148012


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l1480_148076

/-- Given a quadrilateral with one diagonal of length d, two offsets h1 and h2,
    and area A, this theorem states that if d = 30, h2 = 6, and A = 240,
    then h1 = 10. -/
theorem quadrilateral_offset (d h1 h2 A : ℝ) :
  d = 30 → h2 = 6 → A = 240 → A = (1/2) * d * (h1 + h2) → h1 = 10 := by
  sorry

#check quadrilateral_offset

end NUMINAMATH_CALUDE_quadrilateral_offset_l1480_148076


namespace NUMINAMATH_CALUDE_sum_of_digits_2023_base7_l1480_148094

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_2023_base7 :
  sumDigits (toBase7 2023) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_2023_base7_l1480_148094


namespace NUMINAMATH_CALUDE_other_person_speed_l1480_148061

/-- Proves that given Roja's speed and the final distance after a certain time, 
    the other person's speed can be determined. -/
theorem other_person_speed 
  (roja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : roja_speed = 2) 
  (h2 : time = 4) 
  (h3 : final_distance = 20) : 
  ∃ other_speed : ℝ, 
    other_speed = 3 ∧ 
    final_distance = (roja_speed + other_speed) * time :=
by
  sorry

#check other_person_speed

end NUMINAMATH_CALUDE_other_person_speed_l1480_148061


namespace NUMINAMATH_CALUDE_train_length_l1480_148004

/-- The length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 265 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1480_148004


namespace NUMINAMATH_CALUDE_sqrt_29_between_5_and_6_l1480_148002

theorem sqrt_29_between_5_and_6 : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_29_between_5_and_6_l1480_148002


namespace NUMINAMATH_CALUDE_pablos_payment_per_page_l1480_148031

/-- The amount Pablo's mother pays him per page, in dollars. -/
def payment_per_page : ℚ := 1 / 100

/-- The number of pages in each book Pablo reads. -/
def pages_per_book : ℕ := 150

/-- The number of books Pablo read. -/
def books_read : ℕ := 12

/-- The amount Pablo spent on candy, in dollars. -/
def candy_cost : ℕ := 15

/-- The amount Pablo had leftover, in dollars. -/
def leftover : ℕ := 3

theorem pablos_payment_per_page :
  payment_per_page * (pages_per_book * books_read : ℚ) = (candy_cost + leftover : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pablos_payment_per_page_l1480_148031


namespace NUMINAMATH_CALUDE_quadratic_completion_l1480_148084

theorem quadratic_completion (x : ℝ) : x^2 + 16*x + 72 = (x + 8)^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l1480_148084


namespace NUMINAMATH_CALUDE_rabbit_race_l1480_148060

theorem rabbit_race (pink_speed white_speed : ℝ) (time_difference : ℝ) :
  pink_speed = 15 →
  white_speed = 10 →
  time_difference = 0.5 →
  ∃ (pink_time : ℝ),
    pink_time * pink_speed = (pink_time + time_difference) * white_speed ∧
    pink_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_race_l1480_148060


namespace NUMINAMATH_CALUDE_number_exceeding_twenty_percent_l1480_148028

theorem number_exceeding_twenty_percent : ∃ x : ℝ, x = 0.20 * x + 40 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_twenty_percent_l1480_148028


namespace NUMINAMATH_CALUDE_harvester_equations_l1480_148059

theorem harvester_equations (x y : ℝ) : True → ∃ (eq1 eq2 : ℝ → ℝ → Prop),
  (∀ a b, eq1 a b ↔ 2 * (2 * a + 5 * b) = 3.6) ∧
  (∀ a b, eq2 a b ↔ 5 * (3 * a + 2 * b) = 8) ∧
  (eq1 x y ∧ eq2 x y) :=
by
  sorry

end NUMINAMATH_CALUDE_harvester_equations_l1480_148059


namespace NUMINAMATH_CALUDE_special_number_pair_l1480_148058

/-- Given two distinct positive integers a and b, such that b is a multiple of a,
    both a and b consist of 2n digits in decimal form with no leading zeros,
    and the first n digits of a are the same as the last n digits of b (and vice versa),
    prove that a = (10^(2n) - 1) / 7 and b = 6 * (10^(2n) - 1) / 7 -/
theorem special_number_pair (n : ℕ) (a b : ℕ) :
  (a ≠ b) →
  (a > 0) →
  (b > 0) →
  (∃ (k : ℕ), b = k * a) →
  (10^n ≤ a) →
  (a < 10^(2*n)) →
  (10^n ≤ b) →
  (b < 10^(2*n)) →
  (∃ (x y : ℕ), a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < 10^n ∧ y < 10^n) →
  (a = (10^(2*n) - 1) / 7 ∧ b = 6 * (10^(2*n) - 1) / 7) := by
sorry

end NUMINAMATH_CALUDE_special_number_pair_l1480_148058


namespace NUMINAMATH_CALUDE_large_cartridge_pages_large_cartridge_pages_proof_l1480_148021

theorem large_cartridge_pages : ℕ → ℕ → ℕ → Prop :=
  fun small_pages medium_pages large_pages =>
    (small_pages = 600) →
    (3 * small_pages = 2 * medium_pages) →
    (3 * medium_pages = 2 * large_pages) →
    (large_pages = 1350)

-- The proof would go here
theorem large_cartridge_pages_proof :
  ∃ (small_pages medium_pages large_pages : ℕ),
    large_cartridge_pages small_pages medium_pages large_pages :=
sorry

end NUMINAMATH_CALUDE_large_cartridge_pages_large_cartridge_pages_proof_l1480_148021


namespace NUMINAMATH_CALUDE_no_point_satisfies_conditions_l1480_148034

-- Define a triangle as a structure with three points
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is inside a triangle
def isInside (T : Triangle) (D : ℝ × ℝ) : Prop :=
  sorry

-- Define a function to get the shortest side of a triangle
def shortestSide (T : Triangle) : ℝ :=
  sorry

-- Main theorem
theorem no_point_satisfies_conditions (ABC : Triangle) :
  ¬ ∃ D : ℝ × ℝ,
    isInside ABC D ∧
    shortestSide (Triangle.mk ABC.B ABC.C D) = 1 ∧
    shortestSide (Triangle.mk ABC.A ABC.C D) = 2 ∧
    shortestSide (Triangle.mk ABC.A ABC.B D) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_no_point_satisfies_conditions_l1480_148034


namespace NUMINAMATH_CALUDE_denver_temperature_peak_l1480_148001

/-- The temperature function modeling a day in Denver, CO -/
def temperature (t : ℝ) : ℝ := -2 * t^2 + 24 * t + 100

/-- Theorem stating that 6 is the smallest non-negative real solution to the temperature equation -/
theorem denver_temperature_peak :
  (∀ t : ℝ, t ≥ 0 → temperature t = 148 → t ≥ 6) ∧
  temperature 6 = 148 := by
  sorry

end NUMINAMATH_CALUDE_denver_temperature_peak_l1480_148001


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l1480_148068

theorem consecutive_odd_numbers_sum (n : ℕ) : 
  (n % 2 = 1) → (n + (n + 2) = 48) → n = 23 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l1480_148068


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l1480_148042

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 21.5) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l1480_148042


namespace NUMINAMATH_CALUDE_divisibility_of_7_power_minus_1_l1480_148038

theorem divisibility_of_7_power_minus_1 : ∃ k : ℤ, 7^51 - 1 = 103 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_7_power_minus_1_l1480_148038


namespace NUMINAMATH_CALUDE_bus_problem_l1480_148063

/-- The number of people who boarded at the second stop of a bus journey --/
def second_stop_boarders (
  rows : ℕ) (seats_per_row : ℕ) 
  (initial_passengers : ℕ) 
  (first_stop_on : ℕ) (first_stop_off : ℕ)
  (second_stop_off : ℕ) 
  (final_empty_seats : ℕ) : ℕ := by
  sorry

/-- The number of people who boarded at the second stop is 17 --/
theorem bus_problem : 
  second_stop_boarders 23 4 16 15 3 10 57 = 17 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l1480_148063


namespace NUMINAMATH_CALUDE_smallest_pyramid_height_approx_l1480_148092

/-- Represents a square-based pyramid with a cylinder inside. -/
structure PyramidWithCylinder where
  base_side_length : ℝ
  cylinder_diameter : ℝ
  cylinder_length : ℝ

/-- Calculates the smallest possible height of the pyramid given its configuration. -/
def smallest_pyramid_height (p : PyramidWithCylinder) : ℝ :=
  sorry

/-- The theorem stating the smallest possible height of the pyramid. -/
theorem smallest_pyramid_height_approx :
  let p := PyramidWithCylinder.mk 20 10 10
  ∃ ε > 0, abs (smallest_pyramid_height p - 22.1) < ε :=
sorry

end NUMINAMATH_CALUDE_smallest_pyramid_height_approx_l1480_148092


namespace NUMINAMATH_CALUDE_initial_sets_count_l1480_148052

/-- The number of letters available (A through J) -/
def n : ℕ := 10

/-- The length of each set of initials -/
def k : ℕ := 3

/-- The number of different three-letter sets of initials possible using letters A through J, with no repetition -/
def num_initial_sets : ℕ := n * (n - 1) * (n - 2)

theorem initial_sets_count : num_initial_sets = 720 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l1480_148052


namespace NUMINAMATH_CALUDE_increment_and_differential_at_point_l1480_148073

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point and increment
def x₀ : ℝ := 2
def Δx : ℝ := 0.1

-- Define the increment of the function
def Δy (x : ℝ) (Δx : ℝ) : ℝ := f (x + Δx) - f x

-- Define the differential of the function
def dy (x : ℝ) (Δx : ℝ) : ℝ := 2 * x * Δx

-- Theorem statement
theorem increment_and_differential_at_point :
  (Δy x₀ Δx = 0.41) ∧ (dy x₀ Δx = 0.4) := by
  sorry

end NUMINAMATH_CALUDE_increment_and_differential_at_point_l1480_148073


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1480_148019

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1480_148019


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l1480_148056

theorem tangent_triangle_area (a : ℝ) : 
  a > 0 → 
  (1/2 * a/2 * a^2 = 2) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l1480_148056


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1480_148024

theorem consecutive_integers_square_sum (a b : ℤ) (h : b = a + 1) :
  a^2 + b^2 + (a*b)^2 = (a*b + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1480_148024


namespace NUMINAMATH_CALUDE_evaluate_expression_l1480_148036

theorem evaluate_expression : ((3^1 - 2 + 6^2 - 0)⁻¹ * 3 : ℚ) = 3 / 37 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1480_148036


namespace NUMINAMATH_CALUDE_solve_fish_problem_l1480_148043

def fish_problem (total_spent : ℕ) (cost_per_fish : ℕ) (fish_for_dog : ℕ) : Prop :=
  let total_fish : ℕ := total_spent / cost_per_fish
  let fish_for_cat : ℕ := total_fish - fish_for_dog
  (fish_for_cat : ℚ) / fish_for_dog = 1 / 2

theorem solve_fish_problem :
  fish_problem 240 4 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_fish_problem_l1480_148043


namespace NUMINAMATH_CALUDE_average_books_read_rounded_l1480_148085

/-- Represents the number of books read by each category of members -/
def books_read : List Nat := [1, 2, 3, 4, 5]

/-- Represents the number of members in each category -/
def members : List Nat := [3, 4, 1, 6, 2]

/-- Calculates the total number of books read -/
def total_books : Nat := (List.zip books_read members).map (fun (b, m) => b * m) |>.sum

/-- Calculates the total number of members -/
def total_members : Nat := members.sum

/-- Calculates the average number of books read per member -/
def average : Rat := total_books / total_members

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : Rat) : Int :=
  ⌊x + 1/2⌋

/-- Main theorem: The average number of books read, rounded to the nearest whole number, is 3 -/
theorem average_books_read_rounded : round_to_nearest average = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_books_read_rounded_l1480_148085


namespace NUMINAMATH_CALUDE_some_fire_breathing_mystical_l1480_148013

-- Define the sets
variable (U : Type) -- Universe set
variable (Dragon MysticalCreature FireBreathingCreature : Set U)

-- Define the conditions
variable (h1 : Dragon ⊆ FireBreathingCreature)
variable (h2 : ∃ x, x ∈ MysticalCreature ∧ x ∈ Dragon)

-- Theorem to prove
theorem some_fire_breathing_mystical :
  ∃ x, x ∈ FireBreathingCreature ∧ x ∈ MysticalCreature :=
by
  sorry


end NUMINAMATH_CALUDE_some_fire_breathing_mystical_l1480_148013


namespace NUMINAMATH_CALUDE_trent_total_travel_l1480_148077

/-- Represents the number of blocks Trent traveled -/
def trent_travel (blocks_to_bus_stop : ℕ) (blocks_on_bus : ℕ) : ℕ :=
  2 * (blocks_to_bus_stop + blocks_on_bus)

/-- Proves that Trent's total travel is 22 blocks given the problem conditions -/
theorem trent_total_travel :
  trent_travel 4 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_trent_total_travel_l1480_148077


namespace NUMINAMATH_CALUDE_brown_dog_weight_l1480_148017

/-- The weight of the brown dog -/
def brown_weight : ℝ := sorry

/-- The weight of the black dog -/
def black_weight : ℝ := brown_weight + 1

/-- The weight of the white dog -/
def white_weight : ℝ := 2 * brown_weight

/-- The weight of the grey dog -/
def grey_weight : ℝ := black_weight - 2

/-- The average weight of all dogs -/
def average_weight : ℝ := 5

theorem brown_dog_weight :
  (brown_weight + black_weight + white_weight + grey_weight) / 4 = average_weight →
  brown_weight = 4 := by sorry

end NUMINAMATH_CALUDE_brown_dog_weight_l1480_148017


namespace NUMINAMATH_CALUDE_orlando_weight_gain_l1480_148007

theorem orlando_weight_gain (x : ℝ) : 
  x + (2 * x + 2) + ((1 / 2) * (2 * x + 2) - 3) = 20 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_orlando_weight_gain_l1480_148007


namespace NUMINAMATH_CALUDE_rectangle_area_l1480_148006

/-- Given a rectangle with length twice its width and width of 5 inches, prove its area is 50 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 5 → length = 2 * width → width * length = 50 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1480_148006


namespace NUMINAMATH_CALUDE_star_example_l1480_148065

-- Define the * operation
def star (a b : ℕ) : ℕ := a + 2 * b

-- State the theorem
theorem star_example : star (star 2 3) 4 = 16 := by sorry

end NUMINAMATH_CALUDE_star_example_l1480_148065


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1480_148067

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k : ℚ) / (2 ^ n : ℚ) = 220 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1480_148067


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l1480_148093

-- Define the curve and line
def curve (x b c : ℝ) : ℝ := x^3 + b*x^2 + c
def line (x k : ℝ) : ℝ := k*x + 1

-- Define the derivative of the curve
def curve_derivative (x b : ℝ) : ℝ := 3*x^2 + 2*b*x

theorem tangent_line_b_value :
  ∀ (k b c : ℝ),
  -- The line passes through the point (1, 2)
  (line 1 k = 2) →
  -- The curve passes through the point (1, 2)
  (curve 1 b c = 2) →
  -- The slope of the line equals the derivative of the curve at x = 1
  (k = curve_derivative 1 b) →
  -- The value of b is -1
  (b = -1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l1480_148093


namespace NUMINAMATH_CALUDE_problem_statement_l1480_148062

theorem problem_statement (a b : ℝ) : (a - 1)^2 + |b + 2| = 0 → (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1480_148062


namespace NUMINAMATH_CALUDE_leap_years_in_200_years_l1480_148048

/-- A calendrical system where leap years occur every four years without exception. -/
structure CalendarSystem where
  /-- The period in years -/
  period : ℕ
  /-- The frequency of leap years -/
  leap_year_frequency : ℕ
  /-- Assertion that leap years occur every four years -/
  leap_year_every_four : leap_year_frequency = 4

/-- The number of leap years in a given period for a calendar system -/
def num_leap_years (c : CalendarSystem) : ℕ :=
  c.period / c.leap_year_frequency

/-- Theorem stating that in a 200-year period with leap years every 4 years, there are 50 leap years -/
theorem leap_years_in_200_years (c : CalendarSystem) 
  (h_period : c.period = 200) : num_leap_years c = 50 := by
  sorry

end NUMINAMATH_CALUDE_leap_years_in_200_years_l1480_148048


namespace NUMINAMATH_CALUDE_largest_common_divisor_360_450_l1480_148026

theorem largest_common_divisor_360_450 : Nat.gcd 360 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_360_450_l1480_148026


namespace NUMINAMATH_CALUDE_arithmetic_sequences_common_terms_l1480_148080

/-- The first arithmetic sequence -/
def seq1 (n : ℕ) : ℕ := 2 + 3 * n

/-- The second arithmetic sequence -/
def seq2 (n : ℕ) : ℕ := 4 + 5 * n

/-- The last term of the first sequence -/
def last1 : ℕ := 2015

/-- The last term of the second sequence -/
def last2 : ℕ := 2014

/-- The number of common terms between the two sequences -/
def commonTerms : ℕ := 134

theorem arithmetic_sequences_common_terms :
  (∃ (s : Finset ℕ), s.card = commonTerms ∧
    (∀ x ∈ s, ∃ n m : ℕ, seq1 n = x ∧ seq2 m = x ∧
      seq1 n ≤ last1 ∧ seq2 m ≤ last2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_common_terms_l1480_148080


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1480_148075

theorem fraction_equivalence : 
  let original := 8 / 9
  let target := 4 / 5
  let subtracted := 4
  (8 - subtracted) / (9 - subtracted) = target := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1480_148075


namespace NUMINAMATH_CALUDE_board_numbers_divisibility_l1480_148016

theorem board_numbers_divisibility (X Y N A B : ℤ) 
  (sum_eq : X + Y = N) 
  (tanya_div : (A * X + B * Y) % N = 0) : 
  (B * X + A * Y) % N = 0 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_divisibility_l1480_148016


namespace NUMINAMATH_CALUDE_exists_n_not_perfect_square_l1480_148029

theorem exists_n_not_perfect_square : ∃ n : ℕ, n > 1 ∧ ¬ ∃ m : ℕ, 2^(2^n - 1) - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_not_perfect_square_l1480_148029


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1480_148087

theorem negation_of_existence_proposition :
  (¬ ∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x^2 - x + c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1480_148087


namespace NUMINAMATH_CALUDE_words_per_page_larger_type_l1480_148072

/-- Given an article with a total of 48,000 words printed on 21 pages,
    where 17 pages use smaller type with 2,400 words each,
    prove that the remaining pages in larger type contain 1,800 words each. -/
theorem words_per_page_larger_type :
  ∀ (total_words total_pages smaller_type_pages words_per_page_smaller : ℕ),
    total_words = 48000 →
    total_pages = 21 →
    smaller_type_pages = 17 →
    words_per_page_smaller = 2400 →
    (total_pages - smaller_type_pages) * 
      ((total_words - smaller_type_pages * words_per_page_smaller) / (total_pages - smaller_type_pages)) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_larger_type_l1480_148072


namespace NUMINAMATH_CALUDE_ending_number_is_67_l1480_148081

-- Define the sum of first n odd integers
def sum_odd_integers (n : ℕ) : ℕ := n^2

-- Define the sum of odd integers from a to b inclusive
def sum_odd_range (a b : ℕ) : ℕ :=
  sum_odd_integers ((b - a) / 2 + 1) - sum_odd_integers ((a - 1) / 2)

-- The main theorem
theorem ending_number_is_67 :
  ∃ x : ℕ, x ≥ 11 ∧ sum_odd_range 11 x = 416 ∧ x = 67 :=
sorry

end NUMINAMATH_CALUDE_ending_number_is_67_l1480_148081


namespace NUMINAMATH_CALUDE_initial_student_count_l1480_148039

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) :
  initial_avg = 62.5 →
  new_avg = 62.0 →
  dropped_score = 70 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * initial_avg = ((n - 1) : ℝ) * new_avg + dropped_score ∧
    n = 16 :=
by sorry

end NUMINAMATH_CALUDE_initial_student_count_l1480_148039


namespace NUMINAMATH_CALUDE_domino_trick_l1480_148074

theorem domino_trick (x y : ℕ) (hx : x ≤ 6) (hy : y ≤ 6) :
  10 * x + y + 30 = 62 → x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_domino_trick_l1480_148074


namespace NUMINAMATH_CALUDE_soap_promotion_theorem_l1480_148023

/-- The original price of soap in yuan -/
def original_price : ℝ := 2

/-- The cost of buying n pieces of soap under Promotion 1 -/
def promotion1_cost (n : ℕ) : ℝ :=
  original_price + 0.7 * original_price * (n - 1 : ℝ)

/-- The cost of buying n pieces of soap under Promotion 2 -/
def promotion2_cost (n : ℕ) : ℝ :=
  0.8 * original_price * n

/-- The minimum number of soap pieces for Promotion 1 to be cheaper than Promotion 2 -/
def min_pieces : ℕ := 4

theorem soap_promotion_theorem :
  ∀ n : ℕ, n ≥ min_pieces →
    promotion1_cost n < promotion2_cost n ∧
    ∀ m : ℕ, m < min_pieces → promotion1_cost m ≥ promotion2_cost m :=
by sorry

end NUMINAMATH_CALUDE_soap_promotion_theorem_l1480_148023


namespace NUMINAMATH_CALUDE_integer_solutions_count_l1480_148008

theorem integer_solutions_count : 
  ∃! (S : Finset ℤ), 
    (∀ a ∈ S, ∃ x : ℤ, x^2 + a*x - 6*a = 0) ∧ 
    (∀ a : ℤ, (∃ x : ℤ, x^2 + a*x - 6*a = 0) → a ∈ S) ∧
    S.card = 9 :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l1480_148008


namespace NUMINAMATH_CALUDE_max_s_value_l1480_148083

/-- Given two regular polygons P₁ (r-gon) and P₂ (s-gon), where the interior angle of P₁ is 68/67 times
    the interior angle of P₂, this theorem states that the maximum possible value of s is 135. -/
theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) 
  (h_angle : (r - 2) * s * 68 = (s - 2) * r * 67) : s ≤ 135 ∧ ∃ r : ℕ, r ≥ 135 ∧ (135 - 2) * r * 67 = (r - 2) * 135 * 68 := by
  sorry

#check max_s_value

end NUMINAMATH_CALUDE_max_s_value_l1480_148083


namespace NUMINAMATH_CALUDE_rectangle_area_l1480_148098

def length (x : ℝ) : ℝ := 5 * x + 3

def width (x : ℝ) : ℝ := x - 7

def area (x : ℝ) : ℝ := length x * width x

theorem rectangle_area (x : ℝ) : area x = 5 * x^2 - 32 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1480_148098


namespace NUMINAMATH_CALUDE_puzzle_solution_l1480_148044

theorem puzzle_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 17) : 
  w - y = 73 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1480_148044


namespace NUMINAMATH_CALUDE_final_digit_mod_seven_l1480_148096

/-- Represents the allowed operations on the number --/
inductive Operation
  | increaseDecrease : Operation
  | subtractAdd : Operation
  | decreaseBySeven : Operation

/-- The initial number as a list of digits --/
def initialNumber : List Nat := List.replicate 100 8

/-- A function that applies an operation to a list of digits --/
def applyOperation (digits : List Nat) (op : Operation) : List Nat :=
  sorry

/-- A function that removes leading zeros from a list of digits --/
def removeLeadingZeros (digits : List Nat) : List Nat :=
  sorry

/-- A function that applies operations until a single digit remains --/
def applyOperationsUntilSingleDigit (digits : List Nat) : Nat :=
  sorry

/-- Theorem stating that the final single digit is equivalent to 3 modulo 7 --/
theorem final_digit_mod_seven (ops : List Operation) :
  (applyOperationsUntilSingleDigit initialNumber) % 7 = 3 :=
sorry

end NUMINAMATH_CALUDE_final_digit_mod_seven_l1480_148096


namespace NUMINAMATH_CALUDE_abs_neg_four_minus_two_l1480_148033

theorem abs_neg_four_minus_two : |(-4 : ℤ) - 2| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_four_minus_two_l1480_148033


namespace NUMINAMATH_CALUDE_eggs_from_gertrude_l1480_148032

/-- Represents the number of eggs collected from each chicken -/
structure EggCollection where
  gertrude : ℕ
  blanche : ℕ
  nancy : ℕ
  martha : ℕ

/-- The theorem stating the number of eggs Trevor got from Gertrude -/
theorem eggs_from_gertrude (collection : EggCollection) : 
  collection.blanche = 3 →
  collection.nancy = 2 →
  collection.martha = 2 →
  collection.gertrude + collection.blanche + collection.nancy + collection.martha = 11 →
  collection.gertrude = 4 := by
  sorry

#check eggs_from_gertrude

end NUMINAMATH_CALUDE_eggs_from_gertrude_l1480_148032


namespace NUMINAMATH_CALUDE_double_inequality_abc_l1480_148069

theorem double_inequality_abc (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a*b - b*c - c*a ∧ 
  a + b + c - a*b - b*c - c*a ≤ (1 + a^2 + b^2 + c^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_abc_l1480_148069


namespace NUMINAMATH_CALUDE_double_factorial_properties_l1480_148020

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def units_digit (n : ℕ) : ℕ := n % 10

theorem double_factorial_properties :
  (double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011) ∧
  ¬(double_factorial 2010 = 2 * Nat.factorial 1005) ∧
  ¬(double_factorial 2010 * double_factorial 2010 = Nat.factorial 2011) ∧
  (units_digit (double_factorial 2011) = 5) := by
  sorry

end NUMINAMATH_CALUDE_double_factorial_properties_l1480_148020


namespace NUMINAMATH_CALUDE_determinant_positive_range_l1480_148095

def second_order_determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_positive_range (x : ℝ) :
  second_order_determinant 2 (3 - x) 1 x > 0 ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_determinant_positive_range_l1480_148095


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1480_148057

/-- Proves that the cost of an adult ticket is $16 given the conditions of the problem -/
theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_attendance : ℕ) 
  (total_revenue : ℕ) (child_attendance : ℕ) :
  child_ticket_cost = 9 →
  total_attendance = 24 →
  total_revenue = 258 →
  child_attendance = 18 →
  (total_attendance - child_attendance) * 16 + child_attendance * child_ticket_cost = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1480_148057


namespace NUMINAMATH_CALUDE_cubic_equation_one_positive_root_l1480_148066

theorem cubic_equation_one_positive_root (a b : ℝ) (hb : b > 0) :
  ∃! x : ℝ, x > 0 ∧ x^3 + a*x^2 - b = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_one_positive_root_l1480_148066


namespace NUMINAMATH_CALUDE_janes_garden_area_l1480_148053

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * (g.long_side_posts - 1) * g.post_spacing

/-- Theorem stating the area of Jane's garden -/
theorem janes_garden_area :
  ∀ g : Garden,
    g.total_posts = 24 →
    g.post_spacing = 3 →
    g.long_side_posts = 3 * g.short_side_posts →
    g.total_posts = 2 * (g.short_side_posts + g.long_side_posts) - 4 →
    garden_area g = 144 := by
  sorry


end NUMINAMATH_CALUDE_janes_garden_area_l1480_148053


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l1480_148099

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x^2 - 3*x

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ (x y : ℝ), x < y → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l1480_148099


namespace NUMINAMATH_CALUDE_temperature_difference_l1480_148051

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = -9) (h2 : lowest = -22) :
  highest - lowest = 13 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l1480_148051


namespace NUMINAMATH_CALUDE_valid_seating_count_l1480_148005

/-- Represents a seating arrangement around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 chairs -/
def isAdjacent (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are across from each other on a round table with 12 chairs -/
def isAcross (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Represents a valid seating arrangement for 6 married couples -/
def ValidSeating (s : SeatingArrangement) : Prop :=
  ∀ i j : Fin 12,
    -- Men and women alternate
    (i.val % 2 = 0 → s i < 6) ∧
    (i.val % 2 = 1 → s i ≥ 6) ∧
    -- No one sits next to or across from their spouse
    (s i < 6 ∧ s j ≥ 6 ∧ s i + 6 = s j →
      ¬(isAdjacent i j ∨ isAcross i j))

/-- The number of valid seating arrangements -/
def numValidSeatings : ℕ := sorry

theorem valid_seating_count : numValidSeatings = 5184 := by sorry

end NUMINAMATH_CALUDE_valid_seating_count_l1480_148005


namespace NUMINAMATH_CALUDE_fraction_problem_l1480_148022

theorem fraction_problem (x y : ℚ) (h1 : x + y = 14/15) (h2 : x * y = 1/10) :
  min x y = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1480_148022


namespace NUMINAMATH_CALUDE_simplify_fraction_l1480_148089

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1480_148089


namespace NUMINAMATH_CALUDE_acetone_weight_approx_l1480_148088

/-- Atomic weight of Carbon in amu -/
def carbon_weight : Float := 12.01

/-- Atomic weight of Hydrogen in amu -/
def hydrogen_weight : Float := 1.008

/-- Atomic weight of Oxygen in amu -/
def oxygen_weight : Float := 16.00

/-- Number of Carbon atoms in Acetone -/
def carbon_count : Nat := 3

/-- Number of Hydrogen atoms in Acetone -/
def hydrogen_count : Nat := 6

/-- Number of Oxygen atoms in Acetone -/
def oxygen_count : Nat := 1

/-- Calculates the molecular weight of Acetone -/
def acetone_molecular_weight : Float :=
  carbon_weight * carbon_count.toFloat +
  hydrogen_weight * hydrogen_count.toFloat +
  oxygen_weight * oxygen_count.toFloat

/-- Theorem stating that the molecular weight of Acetone is approximately 58.08 amu -/
theorem acetone_weight_approx :
  (acetone_molecular_weight - 58.08).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_acetone_weight_approx_l1480_148088


namespace NUMINAMATH_CALUDE_subtract_from_zero_is_additive_inverse_l1480_148090

theorem subtract_from_zero_is_additive_inverse (a : ℚ) : 0 - a = -a := by sorry

end NUMINAMATH_CALUDE_subtract_from_zero_is_additive_inverse_l1480_148090


namespace NUMINAMATH_CALUDE_edge_coloring_theorem_l1480_148014

/-- A complete graph on n vertices -/
def CompleteGraph (n : ℕ) := Unit

/-- A coloring of edges with k colors -/
def Coloring (G : CompleteGraph 10) (k : ℕ) := Unit

/-- Predicate: Any subset of m vertices contains edges of all k colors -/
def AllColorsInSubset (G : CompleteGraph 10) (c : Coloring G k) (m k : ℕ) : Prop := sorry

theorem edge_coloring_theorem (G : CompleteGraph 10) :
  (∃ c : Coloring G 5, AllColorsInSubset G c 5 5) ∧
  (¬ ∃ c : Coloring G 4, AllColorsInSubset G c 4 4) := by sorry

end NUMINAMATH_CALUDE_edge_coloring_theorem_l1480_148014


namespace NUMINAMATH_CALUDE_average_income_problem_l1480_148011

theorem average_income_problem (M N O : ℕ) : 
  (M + N) / 2 = 5050 →
  (N + O) / 2 = 6250 →
  M = 4000 →
  (M + O) / 2 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_average_income_problem_l1480_148011


namespace NUMINAMATH_CALUDE_digit_156_is_zero_l1480_148027

-- Define the fraction
def fraction : ℚ := 37 / 740

-- Define a function to get the nth digit after the decimal point
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_156_is_zero : nthDigitAfterDecimal fraction 156 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_156_is_zero_l1480_148027


namespace NUMINAMATH_CALUDE_ab_minus_c_equals_six_l1480_148079

theorem ab_minus_c_equals_six (a b c : ℝ) 
  (h1 : a + b = 5) 
  (h2 : c^2 = a*b + b - 9) : 
  a*b - c = 6 := by
sorry

end NUMINAMATH_CALUDE_ab_minus_c_equals_six_l1480_148079


namespace NUMINAMATH_CALUDE_thirty_six_times_sum_of_digits_l1480_148000

def sum_of_digits (x : ℕ) : ℕ := sorry

theorem thirty_six_times_sum_of_digits :
  ∀ x : ℕ, x = 36 * sum_of_digits x ↔ x = 324 ∨ x = 648 := by sorry

end NUMINAMATH_CALUDE_thirty_six_times_sum_of_digits_l1480_148000


namespace NUMINAMATH_CALUDE_lucinda_jelly_beans_l1480_148035

/-- The number of grape jelly beans Lucinda originally had -/
def original_grape : ℕ := 180

/-- The number of lemon jelly beans Lucinda originally had -/
def original_lemon : ℕ := original_grape / 3

/-- The number of grape jelly beans Lucinda has after gifting -/
def remaining_grape : ℕ := original_grape - 20

/-- The number of lemon jelly beans Lucinda has after gifting -/
def remaining_lemon : ℕ := original_lemon - 20

theorem lucinda_jelly_beans :
  (original_grape = 3 * original_lemon) ∧
  (remaining_grape = 4 * remaining_lemon) →
  original_grape = 180 :=
by sorry

end NUMINAMATH_CALUDE_lucinda_jelly_beans_l1480_148035


namespace NUMINAMATH_CALUDE_evaluate_P_l1480_148018

-- Define the polynomial P(a)
def P (a : ℝ) : ℝ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

-- Theorem stating the values of P(4/3) and P(2)
theorem evaluate_P : P (4/3) = 0 ∧ P 2 = 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_P_l1480_148018


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_conditions_l1480_148045

theorem sufficient_not_necessary_conditions (a b : ℝ) :
  (∀ (a b : ℝ), a + b > 2 → a + b > 0) ∧
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → a + b > 0) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a + b > 2)) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) ∧
  (∃ (a b : ℝ), ¬(ab > 0) ∧ a + b > 0) ∧
  (∃ (a b : ℝ), ¬(a > 0 ∨ b > 0) ∧ a + b > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_conditions_l1480_148045


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1480_148071

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (A : Point) (B : Point) (C : Point)

/-- The perpendicular distance from a point to a line -/
def perpendicularDistance (P : Point) (A : Point) (B : Point) : ℝ := sorry

theorem equilateral_triangle_side_length 
  (ABC : EquilateralTriangle) (P : Point) :
  perpendicularDistance P ABC.A ABC.B = 2 →
  perpendicularDistance P ABC.B ABC.C = 4 →
  perpendicularDistance P ABC.C ABC.A = 6 →
  ∃ (side : ℝ), side = 8 * Real.sqrt 3 ∧ 
    (perpendicularDistance ABC.A ABC.B ABC.C = side ∧
     perpendicularDistance ABC.B ABC.C ABC.A = side ∧
     perpendicularDistance ABC.C ABC.A ABC.B = side) :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1480_148071


namespace NUMINAMATH_CALUDE_factor_expression_l1480_148047

theorem factor_expression (c : ℝ) : 180 * c^2 + 36 * c = 36 * c * (5 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1480_148047
