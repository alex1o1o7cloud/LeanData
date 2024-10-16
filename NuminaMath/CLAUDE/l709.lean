import Mathlib

namespace NUMINAMATH_CALUDE_problem_statements_l709_70991

theorem problem_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ab - a - 2*b = 0 → a + 2*b ≥ 8) ∧
  (a + b = 1 → Real.sqrt (2*a + 4) + Real.sqrt (b + 1) ≤ 2 * Real.sqrt 3) ∧
  (1 / (a + 1) + 1 / (b + 2) = 1 / 3 → a*b + a + b ≥ 14 + 6 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l709_70991


namespace NUMINAMATH_CALUDE_nancy_zoo_pictures_nancy_zoo_pictures_proof_l709_70911

theorem nancy_zoo_pictures : ℕ → Prop :=
  fun zoo_pictures =>
    let museum_pictures := 8
    let deleted_pictures := 38
    let remaining_pictures := 19
    zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures →
    zoo_pictures = 49

-- Proof
theorem nancy_zoo_pictures_proof : nancy_zoo_pictures 49 := by
  sorry

end NUMINAMATH_CALUDE_nancy_zoo_pictures_nancy_zoo_pictures_proof_l709_70911


namespace NUMINAMATH_CALUDE_increase_both_averages_l709_70903

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem increase_both_averages :
  ∃ x ∈ group1,
    average (group1.filter (· ≠ x)) > average group1 ∧
    average (x :: group2) > average group2 :=
by sorry

end NUMINAMATH_CALUDE_increase_both_averages_l709_70903


namespace NUMINAMATH_CALUDE_students_on_field_trip_l709_70932

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 10

/-- The number of buses needed for the trip -/
def number_of_buses : ℕ := 6

/-- Theorem stating the number of students going on the field trip -/
theorem students_on_field_trip : seats_per_bus * number_of_buses = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_on_field_trip_l709_70932


namespace NUMINAMATH_CALUDE_lisa_sock_collection_l709_70989

/-- The number of sock pairs Lisa ends up with after contributions from various sources. -/
def total_socks (lisa_initial : ℕ) (sandra : ℕ) (mom_extra : ℕ) : ℕ :=
  lisa_initial + sandra + (sandra / 5) + (3 * lisa_initial + mom_extra)

/-- Theorem stating the total number of sock pairs Lisa ends up with. -/
theorem lisa_sock_collection : total_socks 12 20 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lisa_sock_collection_l709_70989


namespace NUMINAMATH_CALUDE_specific_trip_mpg_l709_70920

/-- Represents a car trip with odometer readings and fuel consumption --/
structure CarTrip where
  initial_odometer : ℕ
  initial_fuel : ℕ
  first_refill : ℕ
  second_refill_odometer : ℕ
  second_refill_amount : ℕ
  final_odometer : ℕ
  final_refill : ℕ

/-- Calculates the average miles per gallon for a car trip --/
def averageMPG (trip : CarTrip) : ℚ :=
  let total_distance := trip.final_odometer - trip.initial_odometer
  let total_fuel := trip.initial_fuel + trip.first_refill + trip.second_refill_amount + trip.final_refill
  (total_distance : ℚ) / total_fuel

/-- The specific car trip from the problem --/
def specificTrip : CarTrip := {
  initial_odometer := 58000
  initial_fuel := 2
  first_refill := 8
  second_refill_odometer := 58400
  second_refill_amount := 15
  final_odometer := 59000
  final_refill := 25
}

/-- Theorem stating that the average MPG for the specific trip is 20.0 --/
theorem specific_trip_mpg : averageMPG specificTrip = 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_trip_mpg_l709_70920


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l709_70974

/-- Two lines intersecting on the y-axis -/
structure IntersectingLines where
  k : ℝ
  line1 : ℝ → ℝ → ℝ := fun x y => 2*x + 3*y - k
  line2 : ℝ → ℝ → ℝ := fun x y => x - k*y + 12
  intersect_on_y_axis : ∃ y, line1 0 y = 0 ∧ line2 0 y = 0

/-- The value of k for intersecting lines -/
theorem intersecting_lines_k_value (l : IntersectingLines) : l.k = 6 ∨ l.k = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l709_70974


namespace NUMINAMATH_CALUDE_statement_falsity_l709_70905

theorem statement_falsity (x : ℝ) : x = -4 ∨ x = -2 → x ∈ Set.Iio 2 ∧ ¬(x^2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_statement_falsity_l709_70905


namespace NUMINAMATH_CALUDE_stratified_sampling_third_major_l709_70979

/-- Given a college with three majors and stratified sampling, prove the number of students
    to be drawn from the third major. -/
theorem stratified_sampling_third_major
  (total_students : ℕ)
  (major_a_students : ℕ)
  (major_b_students : ℕ)
  (total_sample : ℕ)
  (h1 : total_students = 1200)
  (h2 : major_a_students = 380)
  (h3 : major_b_students = 420)
  (h4 : total_sample = 120) :
  (total_students - (major_a_students + major_b_students)) * total_sample / total_students = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_major_l709_70979


namespace NUMINAMATH_CALUDE_circumcircle_intersection_l709_70926

-- Define a point in a plane
structure Point : Type :=
  (x : ℝ) (y : ℝ)

-- Define a circle
structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define a function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define a function to create a circumcircle of a triangle
def circumcircle (a b c : Point) : Circle :=
  sorry

-- Define the main theorem
theorem circumcircle_intersection
  (A₁ A₂ B₁ B₂ C₁ C₂ : Point)
  (h : ∃ (P : Point),
    pointOnCircle P (circumcircle A₁ B₁ C₁) ∧
    pointOnCircle P (circumcircle A₁ B₂ C₂) ∧
    pointOnCircle P (circumcircle A₂ B₁ C₂) ∧
    pointOnCircle P (circumcircle A₂ B₂ C₁)) :
  ∃ (Q : Point),
    pointOnCircle Q (circumcircle A₂ B₂ C₂) ∧
    pointOnCircle Q (circumcircle A₂ B₁ C₁) ∧
    pointOnCircle Q (circumcircle A₁ B₂ C₁) ∧
    pointOnCircle Q (circumcircle A₁ B₁ C₂) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_intersection_l709_70926


namespace NUMINAMATH_CALUDE_problem_statement_l709_70944

theorem problem_statement (a b x y : ℝ) 
  (sum_ab : a + b = 2)
  (sum_xy : x + y = 2)
  (product_sum : a * x + b * y = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l709_70944


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l709_70947

/-- Given two arithmetic sequences, prove the ratio of their 5th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  (∀ n, S n / T n = (7 * n) / (n + 3)) →  -- Given condition
  (∀ n, S n = (n / 2) * (a 1 + a n)) →    -- Definition of S_n for arithmetic sequence
  (∀ n, T n = (n / 2) * (b 1 + b n)) →    -- Definition of T_n for arithmetic sequence
  a 5 / b 5 = 21 / 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l709_70947


namespace NUMINAMATH_CALUDE_random_function_iff_stochastic_process_l709_70923

open MeasureTheory ProbabilityTheory

/-- A random function X = (X_t)_{t ∈ T} taking values in (ℝ^T, ℬ(ℝ^T)) -/
def RandomFunction (T : Type) (Ω : Type) [MeasurableSpace Ω] : Type :=
  Ω → (T → ℝ)

/-- A stochastic process (collection of random variables X_t) -/
def StochasticProcess (T : Type) (Ω : Type) [MeasurableSpace Ω] : Type :=
  T → (Ω → ℝ)

/-- Theorem stating the equivalence between random functions and stochastic processes -/
theorem random_function_iff_stochastic_process (T : Type) (Ω : Type) [MeasurableSpace Ω] :
  (∃ X : RandomFunction T Ω, Measurable X) ↔ (∃ Y : StochasticProcess T Ω, ∀ t, Measurable (Y t)) :=
sorry


end NUMINAMATH_CALUDE_random_function_iff_stochastic_process_l709_70923


namespace NUMINAMATH_CALUDE_egg_sales_total_l709_70931

theorem egg_sales_total (x : ℚ) : 
  (x = (1/3 * x + 15) + (7/9 * (2/3 * x - 15) + 10)) → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_egg_sales_total_l709_70931


namespace NUMINAMATH_CALUDE_max_black_balls_proof_l709_70940

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of ways to select 2 red balls and 1 black ball -/
def selection_ways : ℕ := 30

/-- Calculates the number of ways to select 2 red balls and 1 black ball -/
def calc_selection_ways (red : ℕ) : ℕ :=
  Nat.choose red 2 * Nat.choose (total_balls - red) 1

/-- Checks if a given number of red balls satisfies the selection condition -/
def satisfies_condition (red : ℕ) : Prop :=
  calc_selection_ways red = selection_ways

/-- The maximum number of black balls -/
def max_black_balls : ℕ := 3

theorem max_black_balls_proof :
  ∃ (red : ℕ), satisfies_condition red ∧
  ∀ (x : ℕ), satisfies_condition x → (total_balls - x ≤ max_black_balls) :=
sorry

end NUMINAMATH_CALUDE_max_black_balls_proof_l709_70940


namespace NUMINAMATH_CALUDE_trackball_mice_count_l709_70986

theorem trackball_mice_count (total : ℕ) (wireless_ratio optical_ratio : ℚ) : 
  total = 80 →
  wireless_ratio = 1/2 →
  optical_ratio = 1/4 →
  (wireless_ratio + optical_ratio + (1 - wireless_ratio - optical_ratio)) = 1 →
  ↑total * (1 - wireless_ratio - optical_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_trackball_mice_count_l709_70986


namespace NUMINAMATH_CALUDE_complex_sum_problem_l709_70950

theorem complex_sum_problem (a b c d e f : ℝ) :
  b = 5 →
  e = -2 * (a + c) →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 4 * Complex.I →
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l709_70950


namespace NUMINAMATH_CALUDE_shelly_money_proof_l709_70918

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $390 in total -/
theorem shelly_money_proof :
  let ten_dollar_bills : ℕ := 30
  let five_dollar_bills : ℕ := ten_dollar_bills - 12
  total_money ten_dollar_bills five_dollar_bills = 390 := by
sorry

end NUMINAMATH_CALUDE_shelly_money_proof_l709_70918


namespace NUMINAMATH_CALUDE_fourth_grade_students_l709_70984

/-- Calculates the final number of students in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 43 -/
theorem fourth_grade_students : final_student_count 4 3 42 = 43 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l709_70984


namespace NUMINAMATH_CALUDE_at_least_100_triangles_l709_70967

/-- Represents a set of lines in a plane -/
structure LineSet where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersection : Bool

/-- Counts the number of triangular regions formed by a set of lines -/
def count_triangles (lines : LineSet) : ℕ := sorry

/-- Theorem: 300 lines with given conditions form at least 100 triangles -/
theorem at_least_100_triangles (lines : LineSet) 
  (h1 : lines.num_lines = 300)
  (h2 : lines.no_parallel = true)
  (h3 : lines.no_triple_intersection = true) :
  count_triangles lines ≥ 100 := by sorry

end NUMINAMATH_CALUDE_at_least_100_triangles_l709_70967


namespace NUMINAMATH_CALUDE_inequality_and_optimality_l709_70972

theorem inequality_and_optimality :
  (∀ (x y : ℝ), x > 0 → y > 0 → (x + y)^5 ≥ 12 * x * y * (x^3 + y^3)) ∧
  (∀ (K : ℝ), K > 12 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y)^5 < K * x * y * (x^3 + y^3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_optimality_l709_70972


namespace NUMINAMATH_CALUDE_average_of_numbers_l709_70908

def number1 : Nat := 8642097531
def number2 : Nat := 6420875319
def number3 : Nat := 4208653197
def number4 : Nat := 2086431975
def number5 : Nat := 864219753

def numbers : List Nat := [number1, number2, number3, number4, number5]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : Rat) = 4444455555 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l709_70908


namespace NUMINAMATH_CALUDE_cubic_yard_to_cubic_inches_l709_70943

-- Define the conversion factor
def inches_per_yard : ℕ := 36

-- Theorem statement
theorem cubic_yard_to_cubic_inches :
  (inches_per_yard ^ 3 : ℕ) = 46656 :=
sorry

end NUMINAMATH_CALUDE_cubic_yard_to_cubic_inches_l709_70943


namespace NUMINAMATH_CALUDE_f_sum_property_l709_70913

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_property (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_property_l709_70913


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l709_70981

/-- Proves that a rectangular field with breadth 60% of length and area 37500 m² has perimeter 800 m -/
theorem rectangular_field_perimeter (length width : ℝ) : 
  width = 0.6 * length →
  length * width = 37500 →
  2 * (length + width) = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l709_70981


namespace NUMINAMATH_CALUDE_iains_pennies_l709_70959

/-- The number of pennies Iain had initially -/
def initial_pennies : ℕ := 200

/-- The number of old pennies removed -/
def old_pennies : ℕ := 30

/-- The percentage of remaining pennies kept after throwing out -/
def kept_percentage : ℚ := 80 / 100

/-- The number of pennies left after removing old pennies and throwing out some -/
def remaining_pennies : ℕ := 136

theorem iains_pennies :
  (kept_percentage * (initial_pennies - old_pennies : ℚ)).floor = remaining_pennies :=
sorry

end NUMINAMATH_CALUDE_iains_pennies_l709_70959


namespace NUMINAMATH_CALUDE_xy_value_l709_70987

theorem xy_value (x y : ℕ+) (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l709_70987


namespace NUMINAMATH_CALUDE_max_value_tangent_double_angle_l709_70916

/-- Given a function f(x) = 3sin(x) + cos(x) that reaches its maximum value at x = α, 
    prove that tan(2α) = -3/4 -/
theorem max_value_tangent_double_angle (f : ℝ → ℝ) (α : ℝ) 
  (h₁ : ∀ x, f x = 3 * Real.sin x + Real.cos x)
  (h₂ : IsLocalMax f α) : 
  Real.tan (2 * α) = -3/4 := by sorry

end NUMINAMATH_CALUDE_max_value_tangent_double_angle_l709_70916


namespace NUMINAMATH_CALUDE_triangle_side_length_l709_70961

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 3 →
  A = π / 4 →
  B = π / 3 →
  (Real.sin A) / a = (Real.sin B) / b →
  a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l709_70961


namespace NUMINAMATH_CALUDE_math_problem_proof_l709_70933

theorem math_problem_proof (b m n : ℕ) (B C : ℝ) (D : ℝ) :
  b = 4 →
  m = 1 →
  n = 1 →
  (b^m)^n + b^(m+n) = 20 →
  2^20 = B^10 →
  B > 0 →
  Real.sqrt ((20 * B + 45) / C) = C →
  D = C * Real.sin (30 * π / 180) →
  A = 20 ∧ B = 4 ∧ C = 5 ∧ D = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_math_problem_proof_l709_70933


namespace NUMINAMATH_CALUDE_ratio_equality_l709_70939

theorem ratio_equality (x y z m n k a b c : ℝ) 
  (h : x / (m * (n * b + k * c - m * a)) = 
       y / (n * (k * c + m * a - n * b)) ∧
       y / (n * (k * c + m * a - n * b)) = 
       z / (k * (m * a + n * b - k * c))) :
  m / (x * (b * y + c * z - a * x)) = 
  n / (y * (c * z + a * x - b * y)) ∧
  n / (y * (c * z + a * x - b * y)) = 
  k / (z * (a * x + b * y - c * z)) :=
by sorry

end NUMINAMATH_CALUDE_ratio_equality_l709_70939


namespace NUMINAMATH_CALUDE_four_inch_cube_worth_l709_70958

/-- The worth of a cube of gold in dollars -/
def worth (side_length : ℝ) : ℝ :=
  300 * side_length^3

/-- Theorem: The worth of a 4-inch cube of gold is $19200 -/
theorem four_inch_cube_worth : worth 4 = 19200 := by
  sorry

end NUMINAMATH_CALUDE_four_inch_cube_worth_l709_70958


namespace NUMINAMATH_CALUDE_turtle_count_difference_l709_70941

theorem turtle_count_difference (owen_initial : ℕ) (owen_final : ℕ) : 
  owen_initial = 21 →
  owen_final = 50 →
  ∃ (johanna_initial : ℕ),
    johanna_initial < owen_initial ∧
    owen_final = 2 * owen_initial + johanna_initial / 2 ∧
    owen_initial - johanna_initial = 5 :=
by sorry

end NUMINAMATH_CALUDE_turtle_count_difference_l709_70941


namespace NUMINAMATH_CALUDE_sin_30_degrees_l709_70930

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l709_70930


namespace NUMINAMATH_CALUDE_chess_players_lost_to_ai_l709_70964

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) : 
  total_players = 40 → never_lost_fraction = 1/4 → 
  (total_players : ℚ) * (1 - never_lost_fraction) = 30 := by
  sorry

end NUMINAMATH_CALUDE_chess_players_lost_to_ai_l709_70964


namespace NUMINAMATH_CALUDE_triangle_properties_l709_70956

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Define triangle ABC
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  -- Cosine law
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Statement A
  (a / Real.cos A = b / Real.sin B → A = π/4) ∧
  -- Statement D
  (A < π/2 ∧ B < π/2 ∧ C < π/2 → Real.sin A + Real.sin B > Real.cos A + Real.cos B) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l709_70956


namespace NUMINAMATH_CALUDE_average_age_combined_l709_70968

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 35 →
  let total_age_students := num_students * avg_age_students
  let total_age_parents := num_parents * avg_age_parents
  let total_people := num_students + num_parents
  let total_age := total_age_students + total_age_parents
  (total_age / total_people : ℚ) = 25.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l709_70968


namespace NUMINAMATH_CALUDE_rectangle_construction_l709_70949

/-- Given a length b and a sum s, prove the existence of a rectangle with side lengths a and b,
    such that s equals the sum of the diagonal and side b. -/
theorem rectangle_construction (b : ℝ) (s : ℝ) (h_pos : b > 0 ∧ s > b) :
  ∃ (a : ℝ), a > 0 ∧ s = a + (a^2 + b^2).sqrt := by
  sorry

#check rectangle_construction

end NUMINAMATH_CALUDE_rectangle_construction_l709_70949


namespace NUMINAMATH_CALUDE_same_average_speed_exists_l709_70993

theorem same_average_speed_exists : ∃ y : ℝ, 
  (y^2 - 14*y + 45 = (y^2 - 2*y - 35) / (y - 5)) ∧ 
  (y^2 - 14*y + 45 = 6) := by
  sorry

end NUMINAMATH_CALUDE_same_average_speed_exists_l709_70993


namespace NUMINAMATH_CALUDE_points_collinear_iff_k_eq_one_l709_70904

-- Define the vectors
def OA : ℝ × ℝ := (1, -3)
def OB : ℝ × ℝ := (2, -1)
def OC (k : ℝ) : ℝ × ℝ := (k + 1, k - 2)

-- Define collinearity condition
def areCollinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Theorem statement
theorem points_collinear_iff_k_eq_one :
  ∀ k : ℝ, areCollinear OA OB (OC k) ↔ k = 1 := by sorry

end NUMINAMATH_CALUDE_points_collinear_iff_k_eq_one_l709_70904


namespace NUMINAMATH_CALUDE_specific_convention_handshakes_l709_70948

/-- The number of handshakes in a convention with multiple companies -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes in the specific convention scenario -/
theorem specific_convention_handshakes :
  convention_handshakes 5 5 = 250 := by
  sorry

#eval convention_handshakes 5 5

end NUMINAMATH_CALUDE_specific_convention_handshakes_l709_70948


namespace NUMINAMATH_CALUDE_quadratic_equality_l709_70990

theorem quadratic_equality (p q : ℝ) : 
  (∀ x : ℝ, (x + 4) * (x - 1) = x^2 + p*x + q) → 
  (p = 3 ∧ q = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equality_l709_70990


namespace NUMINAMATH_CALUDE_dog_food_cans_per_package_l709_70925

/-- Proves that the number of cans in each package of dog food is 5 -/
theorem dog_food_cans_per_package : 
  ∀ (cat_packages dog_packages cat_cans_per_package : ℕ),
    cat_packages = 9 →
    dog_packages = 7 →
    cat_cans_per_package = 10 →
    cat_packages * cat_cans_per_package = dog_packages * 5 + 55 →
    5 = (cat_packages * cat_cans_per_package - 55) / dog_packages := by
  sorry

end NUMINAMATH_CALUDE_dog_food_cans_per_package_l709_70925


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l709_70962

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 * a 2 * a 3 = 3) →
  (a 10 * a 11 * a 12 = 24) →
  a 13 * a 14 * a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l709_70962


namespace NUMINAMATH_CALUDE_same_number_on_four_dice_l709_70998

theorem same_number_on_four_dice : 
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 4  -- number of dice
  (1 : ℚ) / n^(k-1) = (1 : ℚ) / 216 :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_four_dice_l709_70998


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l709_70912

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l709_70912


namespace NUMINAMATH_CALUDE_min_value_fraction_l709_70982

theorem min_value_fraction (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l709_70982


namespace NUMINAMATH_CALUDE_reeya_fourth_score_l709_70975

/-- Represents a student's scores -/
structure StudentScores where
  scores : List Float
  average : Float

/-- Calculates the fourth score given three known scores and the average -/
def calculateFourthScore (s : StudentScores) : Float :=
  4 * s.average - s.scores.sum

/-- Theorem: Given Reeya's scores and average, her fourth score is 98.4 -/
theorem reeya_fourth_score :
  let s : StudentScores := { scores := [65, 67, 76], average := 76.6 }
  calculateFourthScore s = 98.4 := by
  sorry

#eval calculateFourthScore { scores := [65, 67, 76], average := 76.6 }

end NUMINAMATH_CALUDE_reeya_fourth_score_l709_70975


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l709_70955

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem: For any real number a, (a ⊗ a) ⊗ (a ⊗ a) = 0 -/
theorem otimes_self_otimes_self (a : ℝ) : otimes (otimes a a) (otimes a a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l709_70955


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_l709_70952

/-- The area of a rectangle with dimensions (2x+14) and (2x+10), minus the area of a rectangular hole
    with dimensions y and x, where (y+1) = (x-2) and x = (2y+3), is equal to 2x^2 + 57x + 131. -/
theorem rectangle_with_hole_area (x y : ℝ) : 
  (y + 1 = x - 2) → 
  (x = 2*y + 3) → 
  (2*x + 14) * (2*x + 10) - y * x = 2*x^2 + 57*x + 131 := by
sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_l709_70952


namespace NUMINAMATH_CALUDE_range_of_a_l709_70938

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) → 
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l709_70938


namespace NUMINAMATH_CALUDE_pencil_cost_l709_70971

/-- Given that 150 pencils cost $45, prove that 3200 pencils cost $960 -/
theorem pencil_cost (box_size : ℕ) (box_cost : ℚ) (target_quantity : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  target_quantity = 3200 →
  (target_quantity : ℚ) * (box_cost / box_size) = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l709_70971


namespace NUMINAMATH_CALUDE_total_cost_of_toys_l709_70929

-- Define the costs of the toys
def marbles_cost : ℚ := 9.05
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

-- Theorem stating the total cost
theorem total_cost_of_toys :
  marbles_cost + football_cost + baseball_cost = 20.52 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_toys_l709_70929


namespace NUMINAMATH_CALUDE_sally_remaining_cards_l709_70927

def initial_cards : ℕ := 39
def cards_sold : ℕ := 24

theorem sally_remaining_cards : initial_cards - cards_sold = 15 := by
  sorry

end NUMINAMATH_CALUDE_sally_remaining_cards_l709_70927


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l709_70985

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem intersection_implies_a_equals_one :
  ∀ a : ℝ, (A ∩ B a = {3}) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l709_70985


namespace NUMINAMATH_CALUDE_diameter_segments_length_l709_70928

theorem diameter_segments_length (r : ℝ) (chord_length : ℝ) :
  r = 6 ∧ chord_length = 10 →
  ∃ (a b : ℝ), a + b = 2 * r ∧ a * b = (chord_length / 2) ^ 2 ∧
  a = 6 - Real.sqrt 11 ∧ b = 6 + Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_diameter_segments_length_l709_70928


namespace NUMINAMATH_CALUDE_joggers_meeting_times_l709_70994

theorem joggers_meeting_times (road_length : ℝ) (speed_a : ℝ) (speed_b : ℝ) (duration : ℝ) :
  road_length = 400 ∧
  speed_a = 3 ∧
  speed_b = 2.5 ∧
  duration = 20 * 60 →
  ∃ n : ℕ, n = 8 ∧ 
    (road_length + (n - 1) * 2 * road_length) / (speed_a + speed_b) = duration :=
by sorry

end NUMINAMATH_CALUDE_joggers_meeting_times_l709_70994


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l709_70983

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  x.nonRepeating + x.repeating / (1 - (1 / 10 ^ x.repeatingDigits))

theorem repeating_decimal_to_fraction :
  let x : RepeatingDecimal := { nonRepeating := 7/10, repeating := 36/100, repeatingDigits := 2 }
  x.toRational = 27 / 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l709_70983


namespace NUMINAMATH_CALUDE_no_increase_employees_l709_70901

theorem no_increase_employees (total : ℕ) (salary_percent : ℚ) (travel_percent : ℚ) :
  total = 480 →
  salary_percent = 10 / 100 →
  travel_percent = 20 / 100 →
  total - (total * salary_percent).floor - (total * travel_percent).floor = 336 :=
by sorry

end NUMINAMATH_CALUDE_no_increase_employees_l709_70901


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l709_70951

theorem inscribed_circle_radius_rhombus (d₁ d₂ : ℝ) (h₁ : d₁ = 8) (h₂ : d₂ = 30) :
  let a := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  let r := (d₁ * d₂) / (8 * a)
  r = 30 / Real.sqrt 241 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l709_70951


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_necessary_not_sufficient_l709_70995

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the specific lines and planes
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem perpendicular_implies_parallel_necessary_not_sufficient 
  (h1 : perp_plane l α) 
  (h2 : subset m β) :
  (∀ α β, parallel α β → perp l m) ∧ 
  (∃ α β, perp l m ∧ ¬ parallel α β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_necessary_not_sufficient_l709_70995


namespace NUMINAMATH_CALUDE_calculate_death_rate_l709_70937

/-- Calculates the death rate given birth rate and population growth rate -/
theorem calculate_death_rate (birth_rate : ℝ) (growth_rate : ℝ) : 
  birth_rate = 32 → growth_rate = 0.021 → 
  ∃ (death_rate : ℝ), death_rate = 11 ∧ birth_rate - death_rate = 1000 * growth_rate :=
by
  sorry

#check calculate_death_rate

end NUMINAMATH_CALUDE_calculate_death_rate_l709_70937


namespace NUMINAMATH_CALUDE_equation_solution_l709_70963

theorem equation_solution : 
  ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l709_70963


namespace NUMINAMATH_CALUDE_smallest_coprime_to_180_l709_70976

theorem smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ x < 7 → Nat.gcd x 180 ≠ 1 ∧ Nat.gcd 7 180 = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_180_l709_70976


namespace NUMINAMATH_CALUDE_remainder_sum_l709_70980

theorem remainder_sum (x y : ℤ) (hx : x % 60 = 53) (hy : y % 45 = 17) :
  (x + y) % 15 = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l709_70980


namespace NUMINAMATH_CALUDE_magnitude_of_b_l709_70960

/-- Given two planar vectors a and b satisfying the specified conditions, 
    the magnitude of b is 2. -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  (‖a‖ = 1) →
  (‖a - 2 • b‖ = Real.sqrt 21) →
  (a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * (-1/2)) →
  ‖b‖ = 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l709_70960


namespace NUMINAMATH_CALUDE_max_distance_from_point_to_unit_circle_l709_70906

theorem max_distance_from_point_to_unit_circle :
  ∃ (M : ℝ), M = 6 ∧ ∀ z : ℂ, Complex.abs z = 1 →
    Complex.abs (z - (3 - 4*I)) ≤ M ∧
    ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - (3 - 4*I)) = M :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_point_to_unit_circle_l709_70906


namespace NUMINAMATH_CALUDE_bijection_probability_l709_70914

/-- The probability of establishing a bijection from a subset of a 4-element set to a 5-element set -/
theorem bijection_probability (A : Finset α) (B : Finset β) 
  (hA : Finset.card A = 4) (hB : Finset.card B = 5) : 
  (Finset.card (Finset.powersetCard 4 B) / (Finset.card B ^ Finset.card A) : ℚ) = 24/125 :=
sorry

end NUMINAMATH_CALUDE_bijection_probability_l709_70914


namespace NUMINAMATH_CALUDE_adjacent_smaller_perfect_square_l709_70966

theorem adjacent_smaller_perfect_square (m : ℕ) (h : ∃ k : ℕ, m = k^2) :
  ∃ n : ℕ, n^2 = m - 2*Int.sqrt m + 1 ∧
    n^2 < m ∧
    ∀ k : ℕ, k^2 < m → k^2 ≤ n^2 :=
sorry

end NUMINAMATH_CALUDE_adjacent_smaller_perfect_square_l709_70966


namespace NUMINAMATH_CALUDE_max_homes_first_neighborhood_l709_70965

def revenue_first (n : ℕ) : ℕ := 4 * n

def revenue_second : ℕ := 50

theorem max_homes_first_neighborhood :
  ∀ n : ℕ, revenue_first n ≤ revenue_second → n ≤ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_max_homes_first_neighborhood_l709_70965


namespace NUMINAMATH_CALUDE_train_crossing_time_l709_70909

/-- Calculates the time for a train to cross a platform -/
theorem train_crossing_time
  (train_speed : Real)
  (man_crossing_time : Real)
  (platform_length : Real)
  (h1 : train_speed = 72 * (1000 / 3600)) -- 72 kmph converted to m/s
  (h2 : man_crossing_time = 20)
  (h3 : platform_length = 200) :
  let train_length := train_speed * man_crossing_time
  let total_length := train_length + platform_length
  total_length / train_speed = 30 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l709_70909


namespace NUMINAMATH_CALUDE_pizza_slices_pizza_has_eight_slices_l709_70978

theorem pizza_slices : ℕ → Prop :=
  fun total_slices =>
    let remaining_after_friend := total_slices - 2
    let james_slices := remaining_after_friend / 2
    james_slices = 3 → total_slices = 8

/-- The pizza has 8 slices. -/
theorem pizza_has_eight_slices : ∃ (n : ℕ), pizza_slices n :=
  sorry

end NUMINAMATH_CALUDE_pizza_slices_pizza_has_eight_slices_l709_70978


namespace NUMINAMATH_CALUDE_unique_solution_l709_70996

theorem unique_solution : ∃! x : ℝ, x + x^2 + 15 = 96 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l709_70996


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l709_70969

open Set

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {3, 4}

-- Define set B
def B : Set ℕ := {1, 4, 5}

-- Theorem statement
theorem union_of_A_and_complement_of_B : A ∪ (U \ B) = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l709_70969


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l709_70924

theorem average_cost_before_gratuity 
  (total_people : ℕ) 
  (total_bill : ℚ) 
  (gratuity_rate : ℚ) 
  (h1 : total_people = 9)
  (h2 : total_bill = 756)
  (h3 : gratuity_rate = 1/5) : 
  (total_bill / (1 + gratuity_rate)) / total_people = 70 :=
by sorry

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l709_70924


namespace NUMINAMATH_CALUDE_evaluate_expression_l709_70977

theorem evaluate_expression : ((-2)^3)^(1/3) - (-1)^0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l709_70977


namespace NUMINAMATH_CALUDE_conference_lefthandedness_l709_70942

theorem conference_lefthandedness 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h1 : red + blue = total) 
  (h2 : red = 7 * blue / 3) 
  (red_left : ℕ) 
  (blue_left : ℕ) 
  (h3 : red_left = red / 3) 
  (h4 : blue_left = 2 * blue / 3) : 
  (red_left + blue_left : ℚ) / total = 13 / 30 := by
sorry

end NUMINAMATH_CALUDE_conference_lefthandedness_l709_70942


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l709_70921

/-- Given the conditions of a man's speed in various situations, prove that his speed against the current with wind, waves, and raft is 4 km/hr. -/
theorem mans_speed_against_current (speed_with_current speed_of_current wind_effect wave_effect raft_effect : ℝ)
  (h1 : speed_with_current = 20)
  (h2 : speed_of_current = 5)
  (h3 : wind_effect = 2)
  (h4 : wave_effect = 1)
  (h5 : raft_effect = 3) :
  speed_with_current - speed_of_current - wind_effect - speed_of_current - wave_effect - raft_effect = 4 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l709_70921


namespace NUMINAMATH_CALUDE_printer_ink_problem_l709_70954

/-- The problem of calculating the additional money needed for printer inks --/
theorem printer_ink_problem (initial_amount : ℕ) (black_cost red_cost yellow_cost : ℕ)
  (black_quantity red_quantity yellow_quantity : ℕ) : 
  initial_amount = 50 →
  black_cost = 11 →
  red_cost = 15 →
  yellow_cost = 13 →
  black_quantity = 2 →
  red_quantity = 3 →
  yellow_quantity = 2 →
  (black_cost * black_quantity + red_cost * red_quantity + yellow_cost * yellow_quantity) - initial_amount = 43 := by
  sorry

#check printer_ink_problem

end NUMINAMATH_CALUDE_printer_ink_problem_l709_70954


namespace NUMINAMATH_CALUDE_fruit_selection_problem_l709_70945

theorem fruit_selection_problem (apple_price orange_price : ℚ)
  (initial_avg_price new_avg_price : ℚ) (oranges_removed : ℕ) :
  apple_price = 40 / 100 →
  orange_price = 60 / 100 →
  initial_avg_price = 54 / 100 →
  new_avg_price = 48 / 100 →
  oranges_removed = 5 →
  ∃ (apples oranges : ℕ),
    (apple_price * apples + orange_price * oranges) / (apples + oranges) = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_removed)) / (apples + oranges - oranges_removed) = new_avg_price ∧
    apples + oranges = 10 :=
by sorry

end NUMINAMATH_CALUDE_fruit_selection_problem_l709_70945


namespace NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l709_70902

def standard_deck_size : ℕ := 52
def heart_or_king_count : ℕ := 16

theorem probability_at_least_one_heart_or_king :
  let p : ℚ := 1 - (1 - heart_or_king_count / standard_deck_size) ^ 2
  p = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l709_70902


namespace NUMINAMATH_CALUDE_binary_1101011_equals_107_l709_70907

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101011_equals_107 :
  binary_to_decimal [true, true, false, true, false, true, true] = 107 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101011_equals_107_l709_70907


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l709_70973

/-- Two lines in the plane, parameterized by a real number a -/
structure Lines (a : ℝ) where
  l₁ : ℝ → ℝ → Prop := λ x y => x + 2*a*y - 1 = 0
  l₂ : ℝ → ℝ → Prop := λ x y => (a + 1)*x - a*y = 0

/-- The condition for two lines to be parallel -/
def parallel (a : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) / (2*a) = k * ((a + 1) / (-a))

theorem parallel_lines_a_value (a : ℝ) :
  parallel a → a = -3/2 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l709_70973


namespace NUMINAMATH_CALUDE_tangent_circle_right_triangle_l709_70910

/-- Given a right triangle DEF with right angle at E, DF = √85, DE = 7, and a circle with center 
    on DE tangent to DF and EF, prove that FQ = 6 where Q is the point where the circle meets DF. -/
theorem tangent_circle_right_triangle (D E F Q : ℝ × ℝ) 
  (h_right_angle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0)
  (h_df : Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = Real.sqrt 85)
  (h_de : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 7)
  (h_circle : ∃ (C : ℝ × ℝ), C ∈ Set.Icc D E ∧ 
    dist C D = dist C Q ∧ dist C E = dist C F ∧ dist C Q = dist C F)
  (h_q_on_df : (Q.1 - D.1) * (F.2 - D.2) = (Q.2 - D.2) * (F.1 - D.1)) :
  dist F Q = 6 := by sorry


end NUMINAMATH_CALUDE_tangent_circle_right_triangle_l709_70910


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l709_70917

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if it forms an acute angle of 60° with the y-axis,
    then its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_angle : b / a = Real.sqrt 3 / 3) : 
  let e := Real.sqrt (1 + (b / a)^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l709_70917


namespace NUMINAMATH_CALUDE_vector_triangle_sum_zero_l709_70919

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_triangle_sum_zero (A B C : E) :
  (B - A) + (C - B) + (A - C) = (0 : E) := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_sum_zero_l709_70919


namespace NUMINAMATH_CALUDE_manicure_cost_proof_l709_70922

/-- The cost of a hair updo in dollars -/
def hair_updo_cost : ℝ := 50

/-- The total cost including tips for both services in dollars -/
def total_cost_with_tips : ℝ := 96

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.20

/-- The cost of a manicure in dollars -/
def manicure_cost : ℝ := 30

theorem manicure_cost_proof :
  (hair_updo_cost + manicure_cost) * (1 + tip_percentage) = total_cost_with_tips := by
  sorry

end NUMINAMATH_CALUDE_manicure_cost_proof_l709_70922


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l709_70957

theorem complex_subtraction_simplification :
  (7 : ℂ) - 5*I - ((3 : ℂ) - 7*I) = (4 : ℂ) + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l709_70957


namespace NUMINAMATH_CALUDE_cone_intersection_volume_ratio_l709_70946

/-- A cone with a circular base -/
structure Cone :=
  (radius : ℝ)
  (height : ℝ)

/-- A plane passing through the vertex of the cone -/
structure IntersectingPlane :=
  (chord_length : ℝ)

/-- The theorem stating the ratio of volumes when a plane intersects a cone -/
theorem cone_intersection_volume_ratio
  (c : Cone)
  (p : IntersectingPlane)
  (h1 : p.chord_length = c.radius) :
  ∃ (v1 v2 : ℝ),
    v1 > 0 ∧ v2 > 0 ∧
    (v1 / v2 = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_cone_intersection_volume_ratio_l709_70946


namespace NUMINAMATH_CALUDE_no_integer_solutions_l709_70970

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 8*m^2 + 17*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l709_70970


namespace NUMINAMATH_CALUDE_root_equation_solution_l709_70999

theorem root_equation_solution (m n a : ℝ) : 
  (m^2 - 2*m - 1 = 0) →
  (n^2 - 2*n - 1 = 0) →
  ((7*m^2 - 14*m + a) * (3*n^2 - 6*n - 7) = 8) →
  a = -9 := by sorry

end NUMINAMATH_CALUDE_root_equation_solution_l709_70999


namespace NUMINAMATH_CALUDE_extremal_points_imply_a_gt_one_and_sum_gt_two_l709_70900

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - (1/2) * x^2 - a * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := exp x - x - a

theorem extremal_points_imply_a_gt_one_and_sum_gt_two
  (a : ℝ)
  (x₁ x₂ : ℝ)
  (h₁ : f' a x₁ = 0)
  (h₂ : f' a x₂ = 0)
  (h₃ : x₁ ≠ x₂)
  : a > 1 ∧ f a x₁ + f a x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_extremal_points_imply_a_gt_one_and_sum_gt_two_l709_70900


namespace NUMINAMATH_CALUDE_square_sequence_properties_l709_70953

/-- A quadratic sequence of unit squares -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The theorem stating the properties of the sequence -/
theorem square_sequence_properties :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 150 = 67951 := by
  sorry

#check square_sequence_properties

end NUMINAMATH_CALUDE_square_sequence_properties_l709_70953


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l709_70915

theorem sum_of_distinct_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72 →
  p + q + r + s + t = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l709_70915


namespace NUMINAMATH_CALUDE_expression_simplification_l709_70988

theorem expression_simplification :
  (((3 + 5 + 6 - 2) * 2) / 4) + ((3 * 4 + 6 - 4) / 3) = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l709_70988


namespace NUMINAMATH_CALUDE_rectangle_area_l709_70935

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  area : ℝ := side * side

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ
  area : ℝ := width * height

/-- The problem statement -/
theorem rectangle_area (s1 s2 s3 s4 : Square) (r : Rectangle) :
  s1.area = 4 ∧ s2.area = 4 ∧ s3.area = 1 ∧ s4.side = 2 * s3.side ∧
  r.width = s1.side + s4.side ∧ r.height = s1.side + s3.side →
  r.area = 12 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l709_70935


namespace NUMINAMATH_CALUDE_cancellable_fractions_characterization_l709_70997

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def cancellable_fraction (n d : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit d ∧
  ∃ (a b c : ℕ), 0 < a ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 < c ∧
    n = 10 * a + b ∧ d = 10 * b + c ∧ n * c = a * d

def valid_fractions : Set (ℕ × ℕ) :=
  {(19, 95), (16, 64), (11, 11), (26, 65), (22, 22), (33, 33),
   (49, 98), (44, 44), (55, 55), (66, 66), (77, 77), (88, 88), (99, 99)}

theorem cancellable_fractions_characterization :
  {p : ℕ × ℕ | cancellable_fraction p.1 p.2} = valid_fractions := by sorry

end NUMINAMATH_CALUDE_cancellable_fractions_characterization_l709_70997


namespace NUMINAMATH_CALUDE_min_sum_distances_l709_70936

/-- The minimum value of PA + PB for a point P on the parabola y² = 4x -/
theorem min_sum_distances (y : ℝ) : 
  let x := y^2 / 4
  let PA := x
  let PB := |x - y + 4| / Real.sqrt 2
  (∀ y', (y'^2 / 4 - y' / Real.sqrt 2 + 2 * Real.sqrt 2) ≥ 
         (y^2 / 4 - y / Real.sqrt 2 + 2 * Real.sqrt 2)) →
  PA + PB = 5 * Real.sqrt 2 / 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l709_70936


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l709_70992

/-- Given two rectangles A and B, where A has a perimeter of 40 cm and its length is twice its width,
    and B has an area equal to one-half the area of A and its length is twice its width,
    prove that the perimeter of B is 20√2 cm. -/
theorem rectangle_perimeter (width_A : ℝ) (width_B : ℝ) : 
  (2 * (width_A + 2 * width_A) = 40) →  -- Perimeter of A is 40 cm
  (width_B * (2 * width_B) = (width_A * (2 * width_A)) / 2) →  -- Area of B is half of A
  (2 * (width_B + 2 * width_B) = 20 * Real.sqrt 2) :=  -- Perimeter of B is 20√2 cm
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l709_70992


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l709_70934

theorem triangle_ratio_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π / 3 ∧
  b = 1 ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l709_70934
