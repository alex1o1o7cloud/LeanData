import Mathlib

namespace NUMINAMATH_CALUDE_point_not_on_line_l235_23590

theorem point_not_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  ¬ (∃ (x y : ℝ), y = m * x + b ∧ x = 0 ∧ y = -2023) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l235_23590


namespace NUMINAMATH_CALUDE_man_speed_man_speed_result_l235_23591

/-- Calculates the speed of a man given a train passing him in the opposite direction -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * (3600 / 1000)
  man_speed_kmh

/-- The speed of the man is approximately 6 km/h -/
theorem man_speed_result :
  ∃ ε > 0, |man_speed 200 60 10.909090909090908 - 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_result_l235_23591


namespace NUMINAMATH_CALUDE_triangle_similarity_l235_23535

/-- Given five complex numbers a, b, c, u, v representing points on a plane,
    if the ratios (v-a)/(u-a), (u-v)/(b-v), and (c-u)/(v-u) are equal,
    then the ratio (v-a)/(u-a) is equal to (c-a)/(b-a). -/
theorem triangle_similarity (a b c u v : ℂ) :
  (v - a) / (u - a) = (u - v) / (b - v) ∧
  (v - a) / (u - a) = (c - u) / (v - u) →
  (v - a) / (u - a) = (c - a) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l235_23535


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l235_23589

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    if a_1, a_3, and a_9 form a geometric sequence, then a_3 / a_1 = 3 -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)  -- arithmetic sequence condition
  (h3 : (a 3 / a 1) = (a 9 / a 3)) -- geometric sequence condition
  : a 3 / a 1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l235_23589


namespace NUMINAMATH_CALUDE_birthday_attendees_l235_23584

theorem birthday_attendees : ∃ (n : ℕ), 
  (12 * (n + 2) = 16 * n) ∧ 
  (n = 6) := by
sorry

end NUMINAMATH_CALUDE_birthday_attendees_l235_23584


namespace NUMINAMATH_CALUDE_circle_equation_coefficients_l235_23547

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the coefficients of the general circle equation -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a given equation represents a specific circle -/
def represents_circle (eq : CircleEquation) (circle : Circle) : Prop :=
  ∀ (x y : ℝ), 
    x^2 + y^2 + 2*eq.a*x - eq.b*y + eq.c = 0 ↔ 
    (x - circle.center.1)^2 + (y - circle.center.2)^2 = circle.radius^2

/-- The main theorem to prove -/
theorem circle_equation_coefficients 
  (circle : Circle) 
  (h_center : circle.center = (2, 3)) 
  (h_radius : circle.radius = 3) :
  ∃ (eq : CircleEquation),
    represents_circle eq circle ∧ 
    eq.a = -2 ∧ 
    eq.b = 6 ∧ 
    eq.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_coefficients_l235_23547


namespace NUMINAMATH_CALUDE_clown_mobiles_count_l235_23525

theorem clown_mobiles_count (clowns_per_mobile : ℕ) (total_clowns : ℕ) (mobiles_count : ℕ) : 
  clowns_per_mobile = 28 → 
  total_clowns = 140 → 
  mobiles_count * clowns_per_mobile = total_clowns →
  mobiles_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_clown_mobiles_count_l235_23525


namespace NUMINAMATH_CALUDE_largest_number_l235_23596

theorem largest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -1) (hc : c = -2) (hd : d = 1) :
  d = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l235_23596


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l235_23506

def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

def monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x ≥ 1, monotonically_increasing (f a) 1 x) →
  (a ≤ -2 ∧ ∃ b, b ≤ 0 ∧ b > -2 ∧ ∀ x ≥ 1, monotonically_increasing (f b) 1 x) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l235_23506


namespace NUMINAMATH_CALUDE_max_constant_inequality_l235_23548

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ a) ∧
  (∀ (b : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ b) → b ≥ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l235_23548


namespace NUMINAMATH_CALUDE_upper_limit_range_l235_23580

theorem upper_limit_range (n x : ℝ) : 
  3 < n ∧ n < x ∧ 6 < n ∧ n < 10 ∧ n = 7 → x > 7 := by
sorry

end NUMINAMATH_CALUDE_upper_limit_range_l235_23580


namespace NUMINAMATH_CALUDE_tray_height_l235_23553

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = 5 →
  cut_angle = 45 →
  ∃ (height : ℝ), height = 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tray_height_l235_23553


namespace NUMINAMATH_CALUDE_not_necessarily_parallel_l235_23558

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem not_necessarily_parallel
  (m : Line) (α β : Plane)
  (h1 : parallel_plane_plane α β)
  (h2 : parallel_line_plane m α) :
  ¬ (∀ m α β, parallel_plane_plane α β → parallel_line_plane m α → parallel_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_parallel_l235_23558


namespace NUMINAMATH_CALUDE_alan_market_spend_l235_23579

/-- The total amount spent by Alan at the market -/
def total_spent (num_eggs : ℕ) (price_per_egg : ℕ) (num_chickens : ℕ) (price_per_chicken : ℕ) : ℕ :=
  num_eggs * price_per_egg + num_chickens * price_per_chicken

/-- Theorem: Alan spent $88 at the market -/
theorem alan_market_spend :
  total_spent 20 2 6 8 = 88 := by
  sorry

end NUMINAMATH_CALUDE_alan_market_spend_l235_23579


namespace NUMINAMATH_CALUDE_min_a_for_increasing_f_l235_23554

/-- The function f(x) defined as x² + (a-2)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 2) * x - 1

/-- The property that f is increasing on the interval [2, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f a x < f a y

/-- The theorem stating the minimum value of a for which f is increasing on [2, +∞) -/
theorem min_a_for_increasing_f :
  (∃ a_min : ℝ, (∀ a : ℝ, is_increasing_on_interval a ↔ a_min ≤ a) ∧ a_min = -2) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_increasing_f_l235_23554


namespace NUMINAMATH_CALUDE_smallest_number_l235_23585

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -2 * Real.sqrt 2) (h4 : d = -3) :
  min a (min b (min c d)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l235_23585


namespace NUMINAMATH_CALUDE_definite_integral_reciprocal_cosine_squared_l235_23549

theorem definite_integral_reciprocal_cosine_squared (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∫ x in (0)..(2 * Real.pi), 1 / (a + b * Real.cos x)^2 = (2 * Real.pi * a) / (a^2 - b^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_reciprocal_cosine_squared_l235_23549


namespace NUMINAMATH_CALUDE_negation_of_proposition_l235_23545

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, x ≥ 0 → x - 2 > 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l235_23545


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l235_23560

def is_single_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    is_single_digit d ∧
    is_single_digit e ∧
    is_prime d ∧
    is_prime e ∧
    is_prime (10 * d + e) ∧
    n = d * e * (10 * d + e) ∧
    (∀ (m : ℕ), m = d' * e' * (10 * d' + e') →
      is_single_digit d' →
      is_single_digit e' →
      is_prime d' →
      is_prime e' →
      is_prime (10 * d' + e') →
      m ≤ n) ∧
    sum_of_digits n = 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l235_23560


namespace NUMINAMATH_CALUDE_pots_needed_for_path_l235_23568

/-- Calculate the number of pots needed for a path with given specifications. -/
def calculate_pots (path_length : ℕ) (pot_distance : ℕ) : ℕ :=
  let pots_per_side := path_length / pot_distance + 1
  2 * pots_per_side

/-- Theorem stating that 152 pots are needed for the given path specifications. -/
theorem pots_needed_for_path : calculate_pots 150 2 = 152 := by
  sorry

#eval calculate_pots 150 2

end NUMINAMATH_CALUDE_pots_needed_for_path_l235_23568


namespace NUMINAMATH_CALUDE_taxi_initial_fee_l235_23576

/-- Represents the taxi service charging model -/
structure TaxiCharge where
  initialFee : ℝ
  additionalChargePerSegment : ℝ
  segmentLength : ℝ
  totalDistance : ℝ
  totalCharge : ℝ

/-- Theorem: Given the taxi service charging model, prove that the initial fee is $2.25 -/
theorem taxi_initial_fee (t : TaxiCharge) : 
  t.additionalChargePerSegment = 0.3 ∧ 
  t.segmentLength = 2/5 ∧ 
  t.totalDistance = 3.6 ∧ 
  t.totalCharge = 4.95 → 
  t.initialFee = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_taxi_initial_fee_l235_23576


namespace NUMINAMATH_CALUDE_line_through_point_not_perpendicular_l235_23565

/-- A line in the form y = k(x-2) passes through (2,0) and is not perpendicular to the x-axis -/
theorem line_through_point_not_perpendicular (k : ℝ) : 
  ∃ (x y : ℝ), y = k * (x - 2) → 
  (x = 2 ∧ y = 0) ∧ 
  (k ≠ 0 → ∃ (m : ℝ), m ≠ 0 ∧ k = 1 / m) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_not_perpendicular_l235_23565


namespace NUMINAMATH_CALUDE_negative_three_squared_l235_23557

theorem negative_three_squared : (-3 : ℤ) ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_l235_23557


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l235_23564

theorem largest_angle_in_special_triangle : 
  ∀ (y : ℝ), 
    60 + 70 + y = 180 →  -- Sum of angles in a triangle
    y = 70 + 15 →        -- y is 15° more than the second smallest angle (70°)
    max 60 (max 70 y) = 85 :=  -- The largest angle is 85°
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l235_23564


namespace NUMINAMATH_CALUDE_book_arrangement_count_l235_23552

/-- The number of ways to arrange math and history books -/
def arrange_books (math_books : ℕ) (history_books : ℕ) : ℕ :=
  let math_groupings := (math_books + 2 - 1).choose 2
  let math_permutations := Nat.factorial math_books
  let history_placements := history_books.choose 3 * history_books.choose 3
  math_groupings * math_permutations * history_placements

/-- Theorem stating the number of arrangements for 4 math books and 6 history books -/
theorem book_arrangement_count :
  arrange_books 4 6 = 96000 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l235_23552


namespace NUMINAMATH_CALUDE_field_trip_attendance_l235_23540

theorem field_trip_attendance :
  let num_vans : ℕ := 6
  let num_buses : ℕ := 8
  let people_per_van : ℕ := 6
  let people_per_bus : ℕ := 18
  let total_people : ℕ := num_vans * people_per_van + num_buses * people_per_bus
  total_people = 180 := by
sorry

end NUMINAMATH_CALUDE_field_trip_attendance_l235_23540


namespace NUMINAMATH_CALUDE_casper_initial_candies_l235_23514

def candy_problem (initial : ℕ) : Prop :=
  let day1_after_eating : ℚ := (3/4) * initial
  let day1_remaining : ℚ := day1_after_eating - 3
  let day2_after_eating : ℚ := (1/2) * day1_remaining
  let day2_remaining : ℚ := day2_after_eating - 2
  day2_remaining = 10

theorem casper_initial_candies :
  candy_problem 36 := by sorry

end NUMINAMATH_CALUDE_casper_initial_candies_l235_23514


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l235_23569

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (StudyScience : U → Prop)
variable (ClubMember : U → Prop)
variable (Honest : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Student x ∧ ¬StudyScience x)
variable (h2 : ∀ x, ClubMember x → (StudyScience x ∧ Honest x))

-- State the theorem
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_club_members_l235_23569


namespace NUMINAMATH_CALUDE_discount_gain_percent_l235_23533

theorem discount_gain_percent (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.86 * MP
  let profit := SP - CP
  let gain_percent := (profit / CP) * 100
  gain_percent = 34.375 := by
sorry

end NUMINAMATH_CALUDE_discount_gain_percent_l235_23533


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l235_23561

theorem sum_of_three_consecutive_even_numbers (n : ℕ) (h : n = 52) :
  n + (n + 2) + (n + 4) = 162 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l235_23561


namespace NUMINAMATH_CALUDE_stable_polygon_sides_l235_23519

/-- A polygon is stable if connecting all vertices from a point on one of its edges
    (not a vertex) results in a certain number of triangles. -/
def is_stable_polygon (n : ℕ) (t : ℕ) : Prop :=
  n > 2 ∧ t = n - 1

theorem stable_polygon_sides :
  ∀ n : ℕ, is_stable_polygon n 2022 → n = 2023 :=
by sorry

end NUMINAMATH_CALUDE_stable_polygon_sides_l235_23519


namespace NUMINAMATH_CALUDE_division_sum_theorem_l235_23527

theorem division_sum_theorem (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 65)
  (h_divisor : divisor = 24)
  (h_remainder : remainder = 5) :
  quotient * divisor + remainder = 1565 :=
by sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l235_23527


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l235_23581

/-- Given two lines that pass through a common point, prove that the line
    passing through the points defined by their coefficients has a specific equation. -/
theorem line_through_coefficient_points
  (a₁ b₁ a₂ b₂ : ℝ)
  (h₁ : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h₂ : 2 * a₂ + 3 * b₂ + 1 = 0) :
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2 * x + 3 * y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_through_coefficient_points_l235_23581


namespace NUMINAMATH_CALUDE_sideEdgeLength_of_rightTriangularPyramid_l235_23531

/-- A right triangular pyramid with three mutually perpendicular side edges of equal length -/
structure RightTriangularPyramid where
  sideEdgeLength : ℝ
  mutuallyPerpendicular : Bool
  equalLength : Bool

/-- The circumscribed sphere of a RightTriangularPyramid -/
def circumscribedSphere (pyramid : RightTriangularPyramid) : ℝ → Prop :=
  fun surfaceArea => surfaceArea = 4 * Real.pi

/-- Theorem: The length of a side edge of a right triangular pyramid is 2√3/3 -/
theorem sideEdgeLength_of_rightTriangularPyramid (pyramid : RightTriangularPyramid) 
  (h1 : pyramid.mutuallyPerpendicular = true)
  (h2 : pyramid.equalLength = true)
  (h3 : circumscribedSphere pyramid (4 * Real.pi)) :
  pyramid.sideEdgeLength = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sideEdgeLength_of_rightTriangularPyramid_l235_23531


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l235_23574

theorem quadratic_equation_coefficient :
  ∀ a b c : ℝ,
  (∀ x : ℝ, 2 * x^2 = 9 * x + 8) →
  (a * x^2 + b * x + c = 0 ↔ 2 * x^2 - 9 * x - 8 = 0) →
  a = 2 →
  b = -9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l235_23574


namespace NUMINAMATH_CALUDE_percentage_of_long_term_employees_l235_23571

/-- Represents the number of employees in each year range at the Pythagoras company -/
structure EmployeeDistribution where
  less_than_1_year : ℕ
  one_to_2_years : ℕ
  two_to_3_years : ℕ
  three_to_4_years : ℕ
  four_to_5_years : ℕ
  five_to_6_years : ℕ
  six_to_7_years : ℕ
  seven_to_8_years : ℕ
  eight_to_9_years : ℕ
  nine_to_10_years : ℕ
  ten_to_11_years : ℕ
  eleven_to_12_years : ℕ
  twelve_to_13_years : ℕ
  thirteen_to_14_years : ℕ
  fourteen_to_15_years : ℕ

/-- Calculates the total number of employees -/
def totalEmployees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.one_to_2_years + d.two_to_3_years + d.three_to_4_years +
  d.four_to_5_years + d.five_to_6_years + d.six_to_7_years + d.seven_to_8_years +
  d.eight_to_9_years + d.nine_to_10_years + d.ten_to_11_years + d.eleven_to_12_years +
  d.twelve_to_13_years + d.thirteen_to_14_years + d.fourteen_to_15_years

/-- Calculates the number of employees who have worked for 10 years or more -/
def employeesWithTenYearsOrMore (d : EmployeeDistribution) : ℕ :=
  d.ten_to_11_years + d.eleven_to_12_years + d.twelve_to_13_years +
  d.thirteen_to_14_years + d.fourteen_to_15_years

/-- Theorem: The percentage of employees who have worked at the Pythagoras company for 10 years or more is 15% -/
theorem percentage_of_long_term_employees (d : EmployeeDistribution)
  (h : d = { less_than_1_year := 4, one_to_2_years := 6, two_to_3_years := 7,
             three_to_4_years := 4, four_to_5_years := 3, five_to_6_years := 3,
             six_to_7_years := 2, seven_to_8_years := 2, eight_to_9_years := 1,
             nine_to_10_years := 1, ten_to_11_years := 2, eleven_to_12_years := 1,
             twelve_to_13_years := 1, thirteen_to_14_years := 1, fourteen_to_15_years := 1 }) :
  (employeesWithTenYearsOrMore d : ℚ) / (totalEmployees d : ℚ) = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_long_term_employees_l235_23571


namespace NUMINAMATH_CALUDE_gcd_459_357_l235_23505

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l235_23505


namespace NUMINAMATH_CALUDE_distance_to_focus_is_three_l235_23592

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between a point and a vertical line -/
def distanceToVerticalLine (P : Point) (x₀ : ℝ) : ℝ :=
  |P.x - x₀|

/-- Check if a point is on the parabola -/
def isOnParabola (P : Point) (p : Parabola) : Prop :=
  P.y^2 = 4 * p.a * P.x

/-- Distance from a point to the focus of the parabola -/
noncomputable def distanceToFocus (P : Point) (p : Parabola) : ℝ :=
  sorry

/-- Main theorem -/
theorem distance_to_focus_is_three
  (p : Parabola)
  (P : Point)
  (h_on_parabola : isOnParabola P p)
  (h_distance : distanceToVerticalLine P (-3) = 5)
  : distanceToFocus P p = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_three_l235_23592


namespace NUMINAMATH_CALUDE_basin_capacity_l235_23575

/-- The capacity of a basin given waterfall flow rate, leak rate, and fill time -/
theorem basin_capacity
  (waterfall_flow : ℝ)  -- Flow rate of the waterfall in gallons per second
  (leak_rate : ℝ)       -- Leak rate of the basin in gallons per second
  (fill_time : ℝ)       -- Time to fill the basin in seconds
  (h1 : waterfall_flow = 24)
  (h2 : leak_rate = 4)
  (h3 : fill_time = 13)
  : (waterfall_flow - leak_rate) * fill_time = 260 :=
by
  sorry

#check basin_capacity

end NUMINAMATH_CALUDE_basin_capacity_l235_23575


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l235_23537

/-- Given that M(5,3) is the midpoint of segment CD and C(2,6), prove that the sum of D's coordinates is 8 -/
theorem sum_of_coordinates_D (C D M : ℝ × ℝ) : 
  C = (2, 6) →
  M = (5, 3) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l235_23537


namespace NUMINAMATH_CALUDE_random_sampling_cannot_prove_inequality_l235_23536

-- Define the type for inequality proof methods
inductive InequalityProofMethod
  | Comparison
  | Synthetic
  | Analytic
  | Contradiction
  | Scaling
  | RandomSampling

-- Define a predicate for methods that can prove inequalities
def can_prove_inequality (method : InequalityProofMethod) : Prop :=
  match method with
  | InequalityProofMethod.Comparison => True
  | InequalityProofMethod.Synthetic => True
  | InequalityProofMethod.Analytic => True
  | InequalityProofMethod.Contradiction => True
  | InequalityProofMethod.Scaling => True
  | InequalityProofMethod.RandomSampling => False

-- Define random sampling as a sampling method
def is_sampling_method (method : InequalityProofMethod) : Prop :=
  method = InequalityProofMethod.RandomSampling

-- Theorem stating that random sampling cannot be used to prove inequalities
theorem random_sampling_cannot_prove_inequality :
  ∀ (method : InequalityProofMethod),
    is_sampling_method method → ¬(can_prove_inequality method) :=
by sorry

end NUMINAMATH_CALUDE_random_sampling_cannot_prove_inequality_l235_23536


namespace NUMINAMATH_CALUDE_triangle_property_l235_23573

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition
  (a - 2*c) * (Real.cos B) + b * (Real.cos A) = 0 →
  -- Given value for sin A
  Real.sin A = 3 * (Real.sqrt 10) / 10 →
  -- Prove these
  Real.cos B = 1/3 ∧ b/c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l235_23573


namespace NUMINAMATH_CALUDE_line_equidistant_point_value_l235_23523

/-- A line passing through (4, 4) with slope 0.5, equidistant from (0, A) and (12, 8), implies A = 32 -/
theorem line_equidistant_point_value (A : ℝ) : 
  let line_point : ℝ × ℝ := (4, 4)
  let line_slope : ℝ := 0.5
  let point_P : ℝ × ℝ := (0, A)
  let point_Q : ℝ × ℝ := (12, 8)
  (∃ (line : ℝ → ℝ), 
    (line (line_point.1) = line_point.2) ∧ 
    ((line (line_point.1 + 1) - line (line_point.1)) / 1 = line_slope) ∧
    (∃ (midpoint : ℝ × ℝ), 
      (midpoint.1 = (point_P.1 + point_Q.1) / 2) ∧
      (midpoint.2 = (point_P.2 + point_Q.2) / 2) ∧
      (line midpoint.1 = midpoint.2))) →
  A = 32 := by
sorry

end NUMINAMATH_CALUDE_line_equidistant_point_value_l235_23523


namespace NUMINAMATH_CALUDE_perfect_square_mod_three_l235_23511

theorem perfect_square_mod_three (n : ℤ) : (n^2) % 3 = 0 ∨ (n^2) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_mod_three_l235_23511


namespace NUMINAMATH_CALUDE_toy_store_shelves_l235_23520

/-- Calculates the number of shelves needed to display stuffed bears in a toy store. -/
def shelves_needed (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Proves that given the initial conditions, the number of shelves needed is 4. -/
theorem toy_store_shelves :
  shelves_needed 6 18 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l235_23520


namespace NUMINAMATH_CALUDE_min_distance_to_line_l235_23534

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line equation -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x + b*y + 1 = 0

/-- The line bisects the circle's circumference -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y → line_equation a b x y

/-- The theorem to be proved -/
theorem min_distance_to_line (a b : ℝ) :
  line_bisects_circle a b →
  (∃ min : ℝ, min = 5 ∧ ∀ a' b' : ℝ, line_bisects_circle a' b' →
    (a' - 2)^2 + (b' - 2)^2 ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l235_23534


namespace NUMINAMATH_CALUDE_area_of_square_on_hypotenuse_l235_23541

/-- Represents a right-angled isosceles triangle with squares on its sides -/
structure IsoscelesRightTriangle where
  /-- Length of the equal sides -/
  side : ℝ
  /-- Sum of the areas of squares on all sides -/
  squaresSum : ℝ
  /-- The sum of squares is 450 -/
  sum_eq_450 : squaresSum = 450

/-- The area of the square on the hypotenuse of an isosceles right triangle -/
def squareOnHypotenuse (t : IsoscelesRightTriangle) : ℝ := 2 * t.side^2

theorem area_of_square_on_hypotenuse (t : IsoscelesRightTriangle) :
  squareOnHypotenuse t = 225 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_on_hypotenuse_l235_23541


namespace NUMINAMATH_CALUDE_triangle_side_b_l235_23510

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_side_b (t : Triangle) : 
  t.B = π / 6 → t.a = Real.sqrt 3 → t.c = 1 → t.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_l235_23510


namespace NUMINAMATH_CALUDE_circles_product_radii_equals_sum_squares_l235_23570

/-- Given two circles passing through a point M(x₁, y₁) and tangent to both the x-axis and y-axis
    with radii r₁ and r₂, the product of their radii equals the sum of squares of the coordinates of M. -/
theorem circles_product_radii_equals_sum_squares (x₁ y₁ r₁ r₂ : ℝ) 
    (h1 : ∃ (a₁ b₁ : ℝ), (x₁ - a₁)^2 + (y₁ - b₁)^2 = r₁^2 ∧ |a₁| = r₁ ∧ |b₁| = r₁)
    (h2 : ∃ (a₂ b₂ : ℝ), (x₁ - a₂)^2 + (y₁ - b₂)^2 = r₂^2 ∧ |a₂| = r₂ ∧ |b₂| = r₂) :
  r₁ * r₂ = x₁^2 + y₁^2 := by
  sorry

end NUMINAMATH_CALUDE_circles_product_radii_equals_sum_squares_l235_23570


namespace NUMINAMATH_CALUDE_complex_equation_sum_l235_23539

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b - Complex.I →
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l235_23539


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l235_23544

/-- Given a trapezoid PQRS with the following properties:
  - Area is 200 cm²
  - Altitude is 10 cm
  - PQ is 15 cm
  - RS is 20 cm
  Prove that the length of QR is 20 - 2.5√5 - 5√3 cm -/
theorem trapezoid_side_length (area : ℝ) (altitude : ℝ) (pq : ℝ) (rs : ℝ) (qr : ℝ) :
  area = 200 →
  altitude = 10 →
  pq = 15 →
  rs = 20 →
  qr = 20 - 2.5 * Real.sqrt 5 - 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l235_23544


namespace NUMINAMATH_CALUDE_new_person_weight_l235_23572

/-- Proves that the weight of a new person is 85 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 85 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l235_23572


namespace NUMINAMATH_CALUDE_white_marbles_count_l235_23530

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (white : ℕ) 
  (h1 : total = 20)
  (h2 : blue = 5)
  (h3 : red = 7)
  (h4 : total = blue + red + white)
  (h5 : (red + white : ℚ) / total = 3/4) : 
  white = 8 := by
sorry

end NUMINAMATH_CALUDE_white_marbles_count_l235_23530


namespace NUMINAMATH_CALUDE_intersection_volume_is_half_l235_23598

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- The intersection of two regular tetrahedra -/
def tetrahedra_intersection (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Reflection of a regular tetrahedron through its center -/
def reflect_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron := sorry

theorem intersection_volume_is_half (t : RegularTetrahedron) 
  (h : t.volume = 1) : 
  tetrahedra_intersection t (reflect_tetrahedron t) = 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_is_half_l235_23598


namespace NUMINAMATH_CALUDE_necessary_condition_for_greater_than_not_sufficient_condition_for_greater_than_l235_23593

theorem necessary_condition_for_greater_than (a b : ℝ) :
  (a > b) → (a + 1 > b) :=
sorry

theorem not_sufficient_condition_for_greater_than :
  ∃ (a b : ℝ), (a + 1 > b) ∧ ¬(a > b) :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_for_greater_than_not_sufficient_condition_for_greater_than_l235_23593


namespace NUMINAMATH_CALUDE_combination_equation_solution_l235_23542

theorem combination_equation_solution (n : ℕ+) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l235_23542


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l235_23597

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 6*x*y - 4*x + 9*y - 366 = 0 ↔ (x = 3 ∧ y = 14) ∨ (x = -24 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l235_23597


namespace NUMINAMATH_CALUDE_prob_same_length_is_11_35_l235_23507

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of shorter diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 6

/-- The number of longer diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 3

/-- The total number of segments in a regular hexagon -/
def total_segments : ℕ := num_sides + num_short_diagonals + num_long_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_short_diagonals * (num_short_diagonals - 1) + num_long_diagonals * (num_long_diagonals - 1)) /
  (total_segments * (total_segments - 1))

theorem prob_same_length_is_11_35 : prob_same_length = 11 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_is_11_35_l235_23507


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_14_l235_23594

theorem binomial_coefficient_21_14 : Nat.choose 21 14 = 116280 :=
by
  have h1 : Nat.choose 20 13 = 77520 := by sorry
  have h2 : Nat.choose 20 14 = 38760 := by sorry
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_14_l235_23594


namespace NUMINAMATH_CALUDE_computer_additions_l235_23521

/-- The number of additions a computer can perform in 12 hours with pauses -/
def computeAdditions (additionsPerSecond : ℕ) (totalHours : ℕ) (pauseMinutes : ℕ) : ℕ :=
  let workingMinutesPerHour := 60 - pauseMinutes
  let workingSecondsPerHour := workingMinutesPerHour * 60
  let additionsPerHour := additionsPerSecond * workingSecondsPerHour
  additionsPerHour * totalHours

/-- Theorem stating that a computer with given specifications performs 540,000,000 additions in 12 hours -/
theorem computer_additions :
  computeAdditions 15000 12 10 = 540000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_additions_l235_23521


namespace NUMINAMATH_CALUDE_debby_stuffed_animal_tickets_l235_23546

/-- The number of tickets Debby spent on various items at the arcade -/
structure ArcadeTickets where
  hat : ℕ
  yoyo : ℕ
  stuffed_animal : ℕ
  total : ℕ

/-- Theorem about Debby's ticket spending at the arcade -/
theorem debby_stuffed_animal_tickets (d : ArcadeTickets) 
  (hat_tickets : d.hat = 2)
  (yoyo_tickets : d.yoyo = 2)
  (total_tickets : d.total = 14)
  (sum_correct : d.hat + d.yoyo + d.stuffed_animal = d.total) :
  d.stuffed_animal = 10 := by
  sorry

end NUMINAMATH_CALUDE_debby_stuffed_animal_tickets_l235_23546


namespace NUMINAMATH_CALUDE_y_value_at_243_l235_23578

-- Define the function y in terms of k and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/5)

-- State the theorem
theorem y_value_at_243 (k : ℝ) :
  y k 32 = 4 → y k 243 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_243_l235_23578


namespace NUMINAMATH_CALUDE_sams_weight_l235_23577

/-- Given the weights of Tyler, Sam, and Peter, prove Sam's weight -/
theorem sams_weight (tyler sam peter : ℝ) : 
  tyler = sam + 25 →
  peter = tyler / 2 →
  peter = 65 →
  sam = 105 := by
  sorry

end NUMINAMATH_CALUDE_sams_weight_l235_23577


namespace NUMINAMATH_CALUDE_thirteenth_root_unity_product_l235_23529

theorem thirteenth_root_unity_product (w : ℂ) : w^13 = 1 → (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_root_unity_product_l235_23529


namespace NUMINAMATH_CALUDE_expression_evaluation_l235_23515

theorem expression_evaluation :
  let x : ℚ := -3
  let y : ℚ := 1/5
  (2*x + y)^2 - (x + 2*y)*(x - 2*y) - (3*x - y)*(x - 5*y) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l235_23515


namespace NUMINAMATH_CALUDE_friction_negative_work_on_slope_l235_23543

/-- A slope-block system where a block slides down a slope -/
structure SlopeBlockSystem where
  M : ℝ  -- Mass of the slope
  m : ℝ  -- Mass of the block
  μ : ℝ  -- Coefficient of friction between block and slope
  θ : ℝ  -- Angle of the slope
  g : ℝ  -- Acceleration due to gravity

/-- The horizontal surface is smooth -/
def is_smooth_surface (system : SlopeBlockSystem) : Prop :=
  sorry

/-- The block is released from rest at the top of the slope -/
def block_released_from_rest (system : SlopeBlockSystem) : Prop :=
  sorry

/-- The friction force does negative work on the slope -/
def friction_does_negative_work (system : SlopeBlockSystem) : Prop :=
  sorry

/-- Main theorem: The friction force of the block on the slope does negative work on the slope -/
theorem friction_negative_work_on_slope (system : SlopeBlockSystem) 
  (h1 : system.M > 0) 
  (h2 : system.m > 0) 
  (h3 : system.μ > 0) 
  (h4 : system.θ > 0) 
  (h5 : system.g > 0) 
  (h6 : is_smooth_surface system) 
  (h7 : block_released_from_rest system) : 
  friction_does_negative_work system :=
sorry

end NUMINAMATH_CALUDE_friction_negative_work_on_slope_l235_23543


namespace NUMINAMATH_CALUDE_base5_43102_equals_2902_l235_23582

def base5_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5^(digits.length - 1 - i))) 0

theorem base5_43102_equals_2902 :
  base5_to_decimal [4, 3, 1, 0, 2] = 2902 := by
  sorry

end NUMINAMATH_CALUDE_base5_43102_equals_2902_l235_23582


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l235_23587

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l235_23587


namespace NUMINAMATH_CALUDE_gift_card_spending_ratio_l235_23518

theorem gift_card_spending_ratio :
  ∀ (M : ℚ),
  (200 - M - (1/4) * (200 - M) = 75) →
  (M / 200 = 1/2) := by
sorry

end NUMINAMATH_CALUDE_gift_card_spending_ratio_l235_23518


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l235_23588

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    a circle Ω with the real axis of C as its diameter,
    and P the intersection point of Ω and the asymptote of C in the first quadrant,
    if the slope of FP (where F is the right focus of C) is -b/a,
    then the equation of the asymptote of C is y = ±x -/
theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 + b^2), 0)
  let Ω := {(x, y) : ℝ × ℝ | x^2 + y^2 = a^2}
  let P := (a / Real.sqrt 2, a / Real.sqrt 2)
  (P.2 - F.2) / (P.1 - F.1) = -b / a →
  ∀ (x y : ℝ), (x, y) ∈ {(x, y) : ℝ × ℝ | y = x ∨ y = -x} ↔
    ∃ (t : ℝ), t ≠ 0 ∧ x = a * t ∧ y = b * t :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l235_23588


namespace NUMINAMATH_CALUDE_junior_prom_attendance_l235_23550

theorem junior_prom_attendance :
  ∀ (total_kids : ℕ),
    (total_kids / 4 : ℕ) = 25 + 10 →
    total_kids = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_junior_prom_attendance_l235_23550


namespace NUMINAMATH_CALUDE_average_of_one_eighth_and_one_sixth_l235_23517

theorem average_of_one_eighth_and_one_sixth :
  (1 / 8 + 1 / 6) / 2 = 7 / 48 := by sorry

end NUMINAMATH_CALUDE_average_of_one_eighth_and_one_sixth_l235_23517


namespace NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_power_15_l235_23566

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_3_plus_4_power_15 :
  tens_digit ((3 + 4)^15) + ones_digit ((3 + 4)^15) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_power_15_l235_23566


namespace NUMINAMATH_CALUDE_todd_remaining_money_l235_23538

def initial_amount : ℕ := 20
def candy_bars : ℕ := 4
def cost_per_bar : ℕ := 2

theorem todd_remaining_money :
  initial_amount - (candy_bars * cost_per_bar) = 12 := by
  sorry

end NUMINAMATH_CALUDE_todd_remaining_money_l235_23538


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l235_23504

def A : Set ℝ := {-1, 0, 3, 5}
def B : Set ℝ := {x : ℝ | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l235_23504


namespace NUMINAMATH_CALUDE_goldbach_counterexample_characterization_l235_23512

-- Define Goldbach's conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Define what constitutes a counterexample to Goldbach's conjecture
def is_goldbach_counterexample (n : ℕ) : Prop :=
  n > 2 ∧ Even n ∧ ¬(∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q)

-- The theorem to prove
theorem goldbach_counterexample_characterization :
  ∀ n : ℕ, is_goldbach_counterexample n ↔ ¬goldbach_conjecture :=
by sorry

end NUMINAMATH_CALUDE_goldbach_counterexample_characterization_l235_23512


namespace NUMINAMATH_CALUDE_curve_C_equation_m_equilateral_triangle_m_range_dot_product_negative_l235_23513

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > 0 ∧ p.1^2 = 4 * p.2}

-- Define point F
def F : ℝ × ℝ := (0, 1)

-- Define the line y = -2
def line_y_neg_2 (x : ℝ) : ℝ := -2

-- Define the distance condition
def distance_condition (p : ℝ × ℝ) : Prop :=
  Real.sqrt ((p.1 - F.1)^2 + (p.2 - F.2)^2) + 1 = p.2 - line_y_neg_2 p.1

-- Define the theorem for the equation of curve C
theorem curve_C_equation :
  ∀ p : ℝ × ℝ, p ∈ C ↔ p.2 > 0 ∧ p.1^2 = 4 * p.2 :=
sorry

-- Define the theorem for the value of m when triangle AFB is equilateral
theorem m_equilateral_triangle :
  ∀ m : ℝ, (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    (A.2 = B.2) ∧ (A.1 = -B.1) ∧
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) ∧
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (m = 7 + 4 * Real.sqrt 3 ∨ m = 7 - 4 * Real.sqrt 3) :=
sorry

-- Define the theorem for the range of m when FA · FB < 0
theorem m_range_dot_product_negative :
  ∀ m : ℝ, (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    ((A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) < 0)) →
  (3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_curve_C_equation_m_equilateral_triangle_m_range_dot_product_negative_l235_23513


namespace NUMINAMATH_CALUDE_robert_kicks_before_break_l235_23583

/-- The number of kicks Robert took before the break -/
def kicks_before_break (total : ℕ) (after_break : ℕ) (remaining : ℕ) : ℕ :=
  total - (after_break + remaining)

/-- Theorem stating that Robert took 43 kicks before the break -/
theorem robert_kicks_before_break :
  kicks_before_break 98 36 19 = 43 := by
  sorry

end NUMINAMATH_CALUDE_robert_kicks_before_break_l235_23583


namespace NUMINAMATH_CALUDE_djibo_age_problem_l235_23509

/-- Represents the problem of finding when Djibo and his sister's ages summed to 35 --/
theorem djibo_age_problem (djibo_current_age sister_current_age past_sum : ℕ) 
  (h1 : djibo_current_age = 17)
  (h2 : sister_current_age = 28)
  (h3 : past_sum = 35) :
  ∃ (years_ago : ℕ), 
    (djibo_current_age - years_ago) + (sister_current_age - years_ago) = past_sum ∧ 
    years_ago = 5 := by
  sorry


end NUMINAMATH_CALUDE_djibo_age_problem_l235_23509


namespace NUMINAMATH_CALUDE_jacques_initial_gumballs_l235_23522

theorem jacques_initial_gumballs : ℕ :=
  let joanna_initial : ℕ := 40
  let purchase_multiplier : ℕ := 4
  let final_each : ℕ := 250
  let jacques_initial : ℕ := 60

  have h1 : joanna_initial + jacques_initial + purchase_multiplier * (joanna_initial + jacques_initial) = 2 * final_each :=
    by sorry

  jacques_initial

end NUMINAMATH_CALUDE_jacques_initial_gumballs_l235_23522


namespace NUMINAMATH_CALUDE_composite_product_division_l235_23502

def first_five_composites : List Nat := [12, 14, 15, 16, 18]
def next_five_composites : List Nat := [21, 22, 24, 25, 26]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem composite_product_division :
  (product_of_list first_five_composites) / (product_of_list next_five_composites) = 72 / 715 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_division_l235_23502


namespace NUMINAMATH_CALUDE_specific_plate_probability_l235_23526

/-- The set of vowels used in Mathlandia license plates -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- The set of non-vowels used in Mathlandia license plates -/
def nonVowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z', '1'}

/-- The set of digits used in Mathlandia license plates -/
def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

/-- A license plate in Mathlandia -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char
  fifth : Char

/-- The probability of a specific license plate occurring in Mathlandia -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (vowels.card * vowels.card * nonVowels.card * (nonVowels.card - 1) * digits.card)

/-- The specific license plate "AIE19" -/
def specificPlate : LicensePlate := ⟨'A', 'I', 'E', '1', '9'⟩

theorem specific_plate_probability :
  licensePlateProbability specificPlate = 1 / 105000 :=
sorry

end NUMINAMATH_CALUDE_specific_plate_probability_l235_23526


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l235_23562

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  (1/2) * b * a * Real.sin C = 6 * Real.sqrt 3 →
  Real.sqrt 3 * a * Real.cos C - c * Real.sin A = 0 →
  C = π / 3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l235_23562


namespace NUMINAMATH_CALUDE_janes_bagels_l235_23551

theorem janes_bagels (b m : ℕ) : 
  b + m = 6 →
  (55 * b + 80 * m) % 100 = 0 →
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_janes_bagels_l235_23551


namespace NUMINAMATH_CALUDE_hexagon_ABCDEF_perimeter_l235_23524

def hexagon_perimeter (AB BC CD DE EF AF : ℝ) : ℝ :=
  AB + BC + CD + DE + EF + AF

theorem hexagon_ABCDEF_perimeter :
  ∀ (AB BC CD DE EF AF : ℝ),
    AB = 1 → BC = 1 → CD = 1 → DE = 1 → EF = 1 → AF = Real.sqrt 5 →
    hexagon_perimeter AB BC CD DE EF AF = 5 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ABCDEF_perimeter_l235_23524


namespace NUMINAMATH_CALUDE_sum_of_possible_p_values_l235_23559

theorem sum_of_possible_p_values : ∃ (S : Finset Nat), 
  (∀ p ∈ S, ∃ q : Nat, 
    Nat.Prime p ∧ 
    p > 0 ∧ 
    q > 0 ∧ 
    p ∣ (q - 1) ∧ 
    (p + q) ∣ (p^2 + 2020*q^2)) ∧
  (∀ p : Nat, 
    (∃ q : Nat, 
      Nat.Prime p ∧ 
      p > 0 ∧ 
      q > 0 ∧ 
      p ∣ (q - 1) ∧ 
      (p + q) ∣ (p^2 + 2020*q^2)) → 
    p ∈ S) ∧
  S.sum id = 35 := by
sorry


end NUMINAMATH_CALUDE_sum_of_possible_p_values_l235_23559


namespace NUMINAMATH_CALUDE_sqrt3_plus_minus_2_power_2023_l235_23563

theorem sqrt3_plus_minus_2_power_2023 :
  (Real.sqrt 3 + 2) ^ 2023 * (Real.sqrt 3 - 2) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_plus_minus_2_power_2023_l235_23563


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l235_23595

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 ∧ x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l235_23595


namespace NUMINAMATH_CALUDE_number_value_l235_23555

theorem number_value (x number : ℝ) 
  (h1 : (x + 5) * (number - 5) = 0)
  (h2 : ∀ y z : ℝ, (y + 5) * (z - 5) = 0 → x^2 + number^2 ≤ y^2 + z^2) :
  number = 5 := by
sorry

end NUMINAMATH_CALUDE_number_value_l235_23555


namespace NUMINAMATH_CALUDE_number_of_positive_divisors_of_M_l235_23503

def M : ℕ := 49^6 + 6*49^5 + 15*49^4 + 20*49^3 + 15*49^2 + 6*49 + 1

theorem number_of_positive_divisors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 91 := by
  sorry

end NUMINAMATH_CALUDE_number_of_positive_divisors_of_M_l235_23503


namespace NUMINAMATH_CALUDE_gcf_lcm_40_120_100_l235_23528

theorem gcf_lcm_40_120_100 :
  (let a := 40
   let b := 120
   let c := 100
   (Nat.gcd a (Nat.gcd b c) = 20) ∧
   (Nat.lcm a (Nat.lcm b c) = 600)) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_40_120_100_l235_23528


namespace NUMINAMATH_CALUDE_points_are_concyclic_l235_23532

-- Define the points
variable (A B C D E F G H : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the given conditions
axiom condition1 : angle A E H = angle F E B
axiom condition2 : angle E F B = angle C F G
axiom condition3 : angle C G F = angle D G H
axiom condition4 : angle D H G = angle A H E

-- Define concyclicity
def concyclic (A B C D : Point) : Prop := sorry

-- State the theorem
theorem points_are_concyclic : concyclic A B C D := sorry

end NUMINAMATH_CALUDE_points_are_concyclic_l235_23532


namespace NUMINAMATH_CALUDE_second_year_sampled_is_thirteen_l235_23599

/-- Calculates the number of second-year students sampled in a stratified survey. -/
def second_year_sampled (total_population : ℕ) (second_year_population : ℕ) (total_sampled : ℕ) : ℕ :=
  (second_year_population * total_sampled) / total_population

/-- Proves that the number of second-year students sampled is 13 given the problem conditions. -/
theorem second_year_sampled_is_thirteen :
  second_year_sampled 2100 780 35 = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_year_sampled_is_thirteen_l235_23599


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_AFCH_l235_23556

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the cross-shaped figure formed by two intersecting rectangles -/
structure CrossFigure where
  rect1 : Rectangle
  rect2 : Rectangle

/-- Theorem: Area of quadrilateral AFCH in the cross-shaped figure -/
theorem area_of_quadrilateral_AFCH (cf : CrossFigure)
  (h1 : cf.rect1.width = 9)
  (h2 : cf.rect1.height = 5)
  (h3 : cf.rect2.width = 3)
  (h4 : cf.rect2.height = 10) :
  area (Rectangle.mk 9 10) - (area cf.rect1 + area cf.rect2 - area (Rectangle.mk 3 5)) / 2 = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_AFCH_l235_23556


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l235_23500

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ -2023 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l235_23500


namespace NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l235_23501

theorem key_chain_manufacturing_cost 
  (selling_price : ℝ)
  (old_profit_percentage : ℝ)
  (new_profit_percentage : ℝ)
  (new_manufacturing_cost : ℝ)
  (h1 : old_profit_percentage = 0.3)
  (h2 : new_profit_percentage = 0.5)
  (h3 : new_manufacturing_cost = 50)
  (h4 : selling_price = new_manufacturing_cost / (1 - new_profit_percentage)) :
  selling_price * (1 - old_profit_percentage) = 70 := by
sorry

end NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l235_23501


namespace NUMINAMATH_CALUDE_trig_expression_equals_half_l235_23586

theorem trig_expression_equals_half : 
  Real.sin (π / 3) - Real.sqrt 3 * Real.cos (π / 3) + (1 / 2) * Real.tan (π / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_half_l235_23586


namespace NUMINAMATH_CALUDE_debate_club_committee_selection_l235_23516

theorem debate_club_committee_selection (n : ℕ) : 
  (n.choose 3 = 21) → (n.choose 4 = 126) := by
  sorry

end NUMINAMATH_CALUDE_debate_club_committee_selection_l235_23516


namespace NUMINAMATH_CALUDE_triangle_properties_l235_23508

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  b = c →
  2 * Real.sin B = Real.sqrt 3 * Real.sin A →
  0 < B →
  B < π / 2 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  (Real.sin B = Real.sqrt 6 / 3) ∧
  (Real.cos (2 * B + π / 3) = -(1 + 2 * Real.sqrt 6) / 6) ∧
  (b = 2 → (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l235_23508


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l235_23567

theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l235_23567
