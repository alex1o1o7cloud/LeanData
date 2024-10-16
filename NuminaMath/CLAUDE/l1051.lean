import Mathlib

namespace NUMINAMATH_CALUDE_prob_defective_bulb_selection_l1051_105103

/-- Given a box of electric bulbs, this function calculates the probability of
    selecting at least one defective bulb when choosing two bulbs at random. -/
def prob_at_least_one_defective (total : ℕ) (defective : ℕ) : ℚ :=
  1 - (total - defective : ℚ) / total * ((total - defective - 1) : ℚ) / (total - 1)

/-- Theorem stating that for a box with 24 bulbs, 4 of which are defective,
    the probability of choosing at least one defective bulb when randomly
    selecting two bulbs is equal to 43/138. -/
theorem prob_defective_bulb_selection :
  prob_at_least_one_defective 24 4 = 43 / 138 := by
  sorry

end NUMINAMATH_CALUDE_prob_defective_bulb_selection_l1051_105103


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1051_105125

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, 0 < m → m < n → ¬(537 * m ≡ 1073 * m [ZMOD 30])) → 
  (537 * n ≡ 1073 * n [ZMOD 30]) → 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1051_105125


namespace NUMINAMATH_CALUDE_square_difference_identity_l1051_105141

theorem square_difference_identity : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l1051_105141


namespace NUMINAMATH_CALUDE_marks_speed_l1051_105192

/-- Given information about Mark and Chris's journey to school, prove Mark's speed -/
theorem marks_speed (chris_speed : ℝ) (school_distance : ℝ) (mark_initial_distance : ℝ) (time_difference : ℝ) 
  (h1 : chris_speed = 3)
  (h2 : school_distance = 9)
  (h3 : mark_initial_distance = 3)
  (h4 : time_difference = 2) :
  let chris_time := school_distance / chris_speed
  let mark_total_distance := mark_initial_distance * 2 + school_distance
  let mark_time := chris_time + time_difference
  mark_total_distance / mark_time = 3 := by sorry

end NUMINAMATH_CALUDE_marks_speed_l1051_105192


namespace NUMINAMATH_CALUDE_college_entrance_exam_scoring_l1051_105166

theorem college_entrance_exam_scoring (total_questions raw_score questions_answered correct_answers : ℕ)
  (h1 : total_questions = 85)
  (h2 : questions_answered = 82)
  (h3 : correct_answers = 70)
  (h4 : raw_score = 67)
  (h5 : questions_answered ≤ total_questions)
  (h6 : correct_answers ≤ questions_answered) :
  ∃ (points_subtracted : ℚ),
    points_subtracted = 1/4 ∧
    (correct_answers : ℚ) - (questions_answered - correct_answers) * points_subtracted = raw_score := by
sorry

end NUMINAMATH_CALUDE_college_entrance_exam_scoring_l1051_105166


namespace NUMINAMATH_CALUDE_sum_of_integers_l1051_105108

theorem sum_of_integers : (-1) + 2 + (-3) + 1 + (-2) + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1051_105108


namespace NUMINAMATH_CALUDE_seventh_data_entry_is_18_l1051_105101

-- Define the given conditions
def total_results : ℕ := 15
def total_average : ℚ := 60
def first_set_count : ℕ := 7
def first_set_average : ℚ := 56
def second_set_count : ℕ := 6
def second_set_average : ℚ := 63
def last_set_count : ℕ := 6
def last_set_average : ℚ := 66

-- Theorem to prove
theorem seventh_data_entry_is_18 :
  ∃ (x : ℚ),
    x = 18 ∧
    total_average * total_results =
      first_set_average * first_set_count +
      second_set_average * second_set_count +
      x +
      (last_set_average * last_set_count - second_set_average * second_set_count - x) :=
by sorry

end NUMINAMATH_CALUDE_seventh_data_entry_is_18_l1051_105101


namespace NUMINAMATH_CALUDE_negation_equivalence_l1051_105197

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 2 ∧ x₀^2 - 2*x₀ - 2 > 0) ↔ 
  (∀ x : ℝ, x ≥ 2 → x^2 - 2*x - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1051_105197


namespace NUMINAMATH_CALUDE_ellipse_equation_fixed_point_l1051_105130

/-- Ellipse C with center at origin, foci on x-axis, and eccentricity 1/2 -/
structure EllipseC where
  equation : ℝ → ℝ → Prop
  center_origin : equation 0 0
  foci_on_x_axis : ∀ x y, equation x y → y = 0 → x ≠ 0
  eccentricity : (∀ x y, equation x y → x^2 + y^2 = 1) → 
                 (∃ c, c > 0 ∧ ∀ x y, equation x y → x^2 + y^2 = (1 - c^2) * x^2 + y^2)

/-- Parabola with equation x = 1/4 * y^2 -/
def parabola (x y : ℝ) : Prop := x = 1/4 * y^2

/-- One vertex of ellipse C coincides with the focus of the parabola -/
axiom vertex_coincides_focus (C : EllipseC) : 
  ∃ x y, C.equation x y ∧ x^2 + y^2 = 1 ∧ x = 1 ∧ y = 0

/-- Theorem: Standard equation of ellipse C -/
theorem ellipse_equation (C : EllipseC) : 
  ∀ x y, C.equation x y ↔ x^2 + 4/3 * y^2 = 1 :=
sorry

/-- Chord AB of ellipse C passing through (1, 0) -/
def chord (C : EllipseC) (m : ℝ) (x y : ℝ) : Prop :=
  C.equation x y ∧ y = m * (x - 1)

/-- A' is the reflection of A over the x-axis -/
def reflect_over_x (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Theorem: Line A'B passes through (1, 0) -/
theorem fixed_point (C : EllipseC) (m : ℝ) (h : m ≠ 0) :
  ∃ x₁ y₁ x₂ y₂, 
    chord C m x₁ y₁ ∧ 
    chord C m x₂ y₂ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    let (x₁', y₁') := reflect_over_x x₁ y₁
    (y₁' - y₂) / (x₁' - x₂) = (0 - y₂) / (1 - x₂) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_fixed_point_l1051_105130


namespace NUMINAMATH_CALUDE_abs_purely_imaginary_complex_l1051_105126

/-- Given a complex number z = (a + i) / (1 + i) where a is real,
    if z is purely imaginary, then its absolute value is 1. -/
theorem abs_purely_imaginary_complex (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 + Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_purely_imaginary_complex_l1051_105126


namespace NUMINAMATH_CALUDE_same_color_probability_l1051_105136

def total_balls : ℕ := 13 + 7
def green_balls : ℕ := 13
def red_balls : ℕ := 7

theorem same_color_probability :
  (green_balls / total_balls) ^ 3 + (red_balls / total_balls) ^ 3 = 127 / 400 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1051_105136


namespace NUMINAMATH_CALUDE_parabola_vertex_l1051_105150

/-- The vertex of the parabola y = 24x^2 - 48 has coordinates (0, -48) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 24 * x^2 - 48 → (0, -48) = (x, y) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1051_105150


namespace NUMINAMATH_CALUDE_job_completion_time_l1051_105171

/-- Represents the time (in minutes) it takes to complete a job when working together,
    given the individual completion times of two workers. -/
def time_working_together (sylvia_time carla_time : ℚ) : ℚ :=
  1 / (1 / sylvia_time + 1 / carla_time)

/-- Theorem stating that if Sylvia takes 45 minutes and Carla takes 30 minutes to complete a job individually,
    then together they will complete the job in 18 minutes. -/
theorem job_completion_time :
  time_working_together 45 30 = 18 := by
  sorry

#eval time_working_together 45 30

end NUMINAMATH_CALUDE_job_completion_time_l1051_105171


namespace NUMINAMATH_CALUDE_johnny_works_four_hours_on_third_job_l1051_105122

/-- Represents Johnny's work schedule and earnings --/
structure WorkSchedule where
  hours_job1 : ℕ
  rate_job1 : ℕ
  hours_job2 : ℕ
  rate_job2 : ℕ
  rate_job3 : ℕ
  days : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours worked on the third job each day --/
def hours_job3_per_day (w : WorkSchedule) : ℕ :=
  let daily_earnings_job12 := w.hours_job1 * w.rate_job1 + w.hours_job2 * w.rate_job2
  let total_earnings_job12 := daily_earnings_job12 * w.days
  let total_earnings_job3 := w.total_earnings - total_earnings_job12
  total_earnings_job3 / (w.rate_job3 * w.days)

/-- Theorem stating that given Johnny's work schedule, he works 4 hours on the third job each day --/
theorem johnny_works_four_hours_on_third_job (w : WorkSchedule)
  (h1 : w.hours_job1 = 3)
  (h2 : w.rate_job1 = 7)
  (h3 : w.hours_job2 = 2)
  (h4 : w.rate_job2 = 10)
  (h5 : w.rate_job3 = 12)
  (h6 : w.days = 5)
  (h7 : w.total_earnings = 445) :
  hours_job3_per_day w = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnny_works_four_hours_on_third_job_l1051_105122


namespace NUMINAMATH_CALUDE_face_value_of_shares_l1051_105120

/-- Theorem: Face value of shares given investment and dividend information -/
theorem face_value_of_shares 
  (investment : ℝ) 
  (premium_rate : ℝ) 
  (dividend_rate : ℝ) 
  (dividend_amount : ℝ) 
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_amount = 720) :
  ∃ (face_value : ℝ), 
    face_value = 12000 ∧ 
    investment = face_value * (1 + premium_rate) ∧
    dividend_amount = face_value * dividend_rate :=
by sorry

end NUMINAMATH_CALUDE_face_value_of_shares_l1051_105120


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1051_105151

def num_students : ℕ := 12
def num_grades : ℕ := 4

theorem grade_assignment_count :
  num_grades ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1051_105151


namespace NUMINAMATH_CALUDE_Q_value_at_8_l1051_105193

-- Define the polynomial Q(x)
def Q (x : ℂ) (g h i j k l m : ℝ) : ℂ :=
  (3 * x^4 - 54 * x^3 + g * x^2 + h * x + i) *
  (4 * x^5 - 100 * x^4 + j * x^3 + k * x^2 + l * x + m)

-- Define the set of roots
def roots : Set ℂ := {2, 3, 4, 6, 7}

-- Theorem statement
theorem Q_value_at_8 (g h i j k l m : ℝ) :
  (∀ z : ℂ, Q z g h i j k l m = 0 → z ∈ roots) →
  Q 8 g h i j k l m = 14400 := by
  sorry


end NUMINAMATH_CALUDE_Q_value_at_8_l1051_105193


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_small_seat_capacity_l1051_105163

/-- Represents the Ferris wheel in paradise park -/
structure FerrisWheel where
  small_seats : Nat
  large_seats : Nat
  small_seat_capacity : Nat

/-- Calculates the total capacity of small seats on the Ferris wheel -/
def total_small_seat_capacity (fw : FerrisWheel) : Nat :=
  fw.small_seats * fw.small_seat_capacity

theorem paradise_park_ferris_wheel_small_seat_capacity :
  ∃ (fw : FerrisWheel), 
    fw.small_seats = 2 ∧ 
    fw.large_seats = 23 ∧ 
    fw.small_seat_capacity = 14 ∧ 
    total_small_seat_capacity fw = 28 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_small_seat_capacity_l1051_105163


namespace NUMINAMATH_CALUDE_polynomial_identity_l1051_105196

/-- Given a polynomial P(m) that satisfies P(m) - 3m = 5m^2 - 3m - 5,
    prove that P(m) = 5m^2 - 5 -/
theorem polynomial_identity (m : ℝ) (P : ℝ → ℝ) 
    (h : ∀ m, P m - 3*m = 5*m^2 - 3*m - 5) : 
    P m = 5*m^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1051_105196


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1051_105144

def A : Set ℝ := {x | |x - 3| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1051_105144


namespace NUMINAMATH_CALUDE_prism_properties_l1051_105161

/-- Represents a prism with n sides in its base. -/
structure Prism (n : ℕ) where
  base_sides : n ≥ 3

/-- Properties of a prism. -/
def Prism.properties (p : Prism n) : Prop :=
  let lateral_faces := n
  let lateral_edges := n
  let total_edges := 3 * n
  let total_faces := n + 2
  let total_vertices := 2 * n
  lateral_faces = lateral_edges ∧
  total_edges % 3 = 0 ∧
  (n ≥ 4 → Even total_faces) ∧
  Even total_vertices

/-- Theorem stating the properties of a prism. -/
theorem prism_properties (n : ℕ) (p : Prism n) : p.properties := by
  sorry

end NUMINAMATH_CALUDE_prism_properties_l1051_105161


namespace NUMINAMATH_CALUDE_polygon_sides_theorem_l1051_105142

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The measure of one interior angle in a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

/-- Predicate to check if a pair of numbers satisfies the polygon conditions -/
def satisfies_conditions (x y : ℕ) : Prop :=
  y = x + 10 ∧
  num_diagonals y - num_diagonals x = interior_angle x - 15

theorem polygon_sides_theorem :
  ∀ x y : ℕ, satisfies_conditions x y → (x = 5 ∧ y = 15) ∨ (x = 8 ∧ y = 18) :=
sorry

#check polygon_sides_theorem

end NUMINAMATH_CALUDE_polygon_sides_theorem_l1051_105142


namespace NUMINAMATH_CALUDE_largest_value_l1051_105199

theorem largest_value (a b : ℝ) 
  (ha : 0 < a) (ha1 : a < 1) 
  (hb : 0 < b) (hb1 : b < 1) 
  (hab : a ≠ b) : 
  a + b ≥ 2 * Real.sqrt (a * b) ∧ a + b ≥ (a^2 + b^2) / (2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l1051_105199


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l1051_105167

theorem cubic_three_distinct_roots_in_interval 
  (p q : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    (-2 < x₁ ∧ x₁ < 4) ∧ (-2 < x₂ ∧ x₂ < 4) ∧ (-2 < x₃ ∧ x₃ < 4) ∧
    x₁^3 + p*x₁ + q = 0 ∧ x₂^3 + p*x₂ + q = 0 ∧ x₃^3 + p*x₃ + q = 0) ↔ 
  (4*p^3 + 27*q^2 < 0 ∧ 2*p + 8 < q ∧ q < -4*p - 64) :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l1051_105167


namespace NUMINAMATH_CALUDE_max_consecutive_sum_30_l1051_105116

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive integers starting from 2 -/
def sum_from_2 (n : ℕ) : ℕ := sum_first_n (n + 1) - 1

/-- 30 is the maximum number of consecutive positive integers 
    starting from 2 that can be added together without exceeding 500 -/
theorem max_consecutive_sum_30 :
  (∀ k : ℕ, k ≤ 30 → sum_from_2 k ≤ 500) ∧
  (∀ k : ℕ, k > 30 → sum_from_2 k > 500) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_30_l1051_105116


namespace NUMINAMATH_CALUDE_shelby_rain_time_l1051_105154

/-- Represents the driving scenario for Shelby -/
structure DrivingScenario where
  speed_sun : ℝ  -- Speed when not raining (miles per hour)
  speed_rain : ℝ  -- Speed when raining (miles per hour)
  total_distance : ℝ  -- Total distance driven (miles)
  total_time : ℝ  -- Total time driven (minutes)

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, Shelby drove 16 minutes in the rain -/
theorem shelby_rain_time :
  let scenario : DrivingScenario := {
    speed_sun := 40,
    speed_rain := 25,
    total_distance := 20,
    total_time := 36
  }
  time_in_rain scenario = 16 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rain_time_l1051_105154


namespace NUMINAMATH_CALUDE_junior_fraction_l1051_105115

theorem junior_fraction (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h3 : J * 3 = S * 4) :
  J / (J + S) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_junior_fraction_l1051_105115


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1051_105121

/-- The average speed of a round trip journey given outbound and inbound speeds -/
theorem round_trip_average_speed 
  (outbound_speed inbound_speed : ℝ) 
  (outbound_speed_pos : outbound_speed > 0)
  (inbound_speed_pos : inbound_speed > 0)
  (h_outbound : outbound_speed = 44)
  (h_inbound : inbound_speed = 36) :
  2 * outbound_speed * inbound_speed / (outbound_speed + inbound_speed) = 39.6 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l1051_105121


namespace NUMINAMATH_CALUDE_range_of_a_l1051_105185

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, (1/a) - (1/x) ≤ 2*x) : a ≥ Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1051_105185


namespace NUMINAMATH_CALUDE_f_extrema_l1051_105134

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem f_extrema :
  let a : ℝ := -1
  let b : ℝ := (Real.exp 2 - 3) / 2
  (∀ x ∈ Set.Icc a b, f (-1/2) ≤ f x) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ f ((Real.exp 2 - 3) / 2)) ∧
  f (-1/2) = Real.log 2 + 1/4 ∧
  f ((Real.exp 2 - 3) / 2) = 2 + (Real.exp 2 - 3)^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l1051_105134


namespace NUMINAMATH_CALUDE_triangle_area_with_perimeter_12_l1051_105138

/-- A triangle with integral sides and perimeter 12 has an area of 6 -/
theorem triangle_area_with_perimeter_12 :
  ∀ a b c : ℕ,
  a + b + c = 12 →
  a + b > c →
  b + c > a →
  c + a > b →
  (a * b : ℝ) / 2 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_perimeter_12_l1051_105138


namespace NUMINAMATH_CALUDE_reflected_point_spherical_coordinates_l1051_105110

/-- Given a point P with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    this function returns the spherical coordinates of the point Q(-x, y, z) -/
def spherical_coordinates_of_reflected_point (x y z ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that if a point P has rectangular coordinates (x, y, z) and 
    spherical coordinates (3, 5π/6, π/4), then the point Q(-x, y, z) has 
    spherical coordinates (3, π/6, π/4) -/
theorem reflected_point_spherical_coordinates 
  (x y z : Real) 
  (h1 : x = 3 * Real.sin (π/4) * Real.cos (5*π/6))
  (h2 : y = 3 * Real.sin (π/4) * Real.sin (5*π/6))
  (h3 : z = 3 * Real.cos (π/4)) :
  spherical_coordinates_of_reflected_point x y z 3 (5*π/6) (π/4) = (3, π/6, π/4) := by
  sorry

end NUMINAMATH_CALUDE_reflected_point_spherical_coordinates_l1051_105110


namespace NUMINAMATH_CALUDE_circle_max_area_center_l1051_105194

/-- Given a circle with equation x^2 + y^2 + kx + 2y + k^2 = 0,
    prove that its center is (0, -1) when the area is maximum. -/
theorem circle_max_area_center (k : ℝ) :
  let circle_eq := λ (x y : ℝ) => x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (-(k/2), -1)
  let radius_squared := 1 - (3/4) * k^2
  (∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius_squared) →
  (radius_squared ≤ 1) →
  (radius_squared = 1 ↔ k = 0) →
  (k = 0 → center = (0, -1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_max_area_center_l1051_105194


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l1051_105169

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 3 * b →   -- ratio of angles is 3:1
  |a - b| = 45  -- positive difference is 45°
:= by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l1051_105169


namespace NUMINAMATH_CALUDE_bryan_annual_commute_hours_l1051_105179

/-- Represents the time in minutes for each segment of Bryan's commute -/
structure CommuteSegment where
  walk_to_bus : ℕ
  bus_ride : ℕ
  walk_to_work : ℕ

/-- Represents Bryan's daily commute -/
def daily_commute : CommuteSegment :=
  { walk_to_bus := 5
  , bus_ride := 20
  , walk_to_work := 5 }

/-- Calculates the total time for a one-way commute in minutes -/
def one_way_commute_time (c : CommuteSegment) : ℕ :=
  c.walk_to_bus + c.bus_ride + c.walk_to_work

/-- Calculates the total daily commute time in hours -/
def daily_commute_hours (c : CommuteSegment) : ℚ :=
  (2 * one_way_commute_time c : ℚ) / 60

/-- The number of days Bryan works per year -/
def work_days_per_year : ℕ := 365

/-- Theorem stating that Bryan spends 365 hours per year commuting -/
theorem bryan_annual_commute_hours :
  (daily_commute_hours daily_commute * work_days_per_year : ℚ) = 365 := by
  sorry


end NUMINAMATH_CALUDE_bryan_annual_commute_hours_l1051_105179


namespace NUMINAMATH_CALUDE_diamonds_15_diamonds_eq_diamonds_closed_diamonds_closed_15_l1051_105191

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else diamonds (n - 1) + 4 * n

/-- The theorem stating that the 15th figure contains 480 diamonds -/
theorem diamonds_15 : diamonds 15 = 480 := by
  sorry

/-- Alternative definition using the closed form formula -/
def diamonds_closed (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Theorem stating the equivalence of the recursive and closed form definitions -/
theorem diamonds_eq_diamonds_closed (n : ℕ) : diamonds n = diamonds_closed n := by
  sorry

/-- The theorem stating that the 15th figure contains 480 diamonds using the closed form -/
theorem diamonds_closed_15 : diamonds_closed 15 = 480 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_15_diamonds_eq_diamonds_closed_diamonds_closed_15_l1051_105191


namespace NUMINAMATH_CALUDE_square_area_from_rectangles_l1051_105175

/-- The area of a square composed of four identical rectangles and a smaller square, 
    where the perimeter of each rectangle is 28. -/
theorem square_area_from_rectangles (l w : ℝ) : 
  (l + w ≥ 0) →  -- Ensure non-negative side length
  (2 * (l + w) = 28) →  -- Perimeter of rectangle
  (l + w) * (l + w) = 196 := by
  sorry

#check square_area_from_rectangles

end NUMINAMATH_CALUDE_square_area_from_rectangles_l1051_105175


namespace NUMINAMATH_CALUDE_treaty_to_university_founding_l1051_105182

theorem treaty_to_university_founding (treaty_day : Nat) (founding_day : Nat) : 
  treaty_day % 7 = 2 → -- Tuesday is represented as 2 (0 = Sunday, 1 = Monday, etc.)
  founding_day = treaty_day + 1204 →
  founding_day % 7 = 5 -- Friday is represented as 5
  := by sorry

end NUMINAMATH_CALUDE_treaty_to_university_founding_l1051_105182


namespace NUMINAMATH_CALUDE_AAA_not_congruence_l1051_105186

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles in radians

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define AAA condition
def AAA (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem: AAA does not imply congruence
theorem AAA_not_congruence :
  ∃ t1 t2 : Triangle, AAA t1 t2 ∧ ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_AAA_not_congruence_l1051_105186


namespace NUMINAMATH_CALUDE_petya_has_higher_chance_of_winning_l1051_105174

structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

def vasya_wins (game : CandyGame) : ℝ :=
  1 - game.prob_two_caramels

def petya_wins (game : CandyGame) : ℝ :=
  game.prob_two_caramels

theorem petya_has_higher_chance_of_winning (game : CandyGame)
  (h1 : game.total_candies = 25)
  (h2 : game.prob_two_caramels = 0.54)
  : petya_wins game > vasya_wins game := by
  sorry

end NUMINAMATH_CALUDE_petya_has_higher_chance_of_winning_l1051_105174


namespace NUMINAMATH_CALUDE_min_value_theorem_l1051_105105

theorem min_value_theorem (x : ℝ) (h : x > 10) :
  (x^2 + 100) / (x - 10) ≥ 20 + 20 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1051_105105


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1051_105109

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1051_105109


namespace NUMINAMATH_CALUDE_max_k_value_l1051_105107

theorem max_k_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : Real.log x + Real.log y = 0) :
  (∃ (k : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → Real.log x + Real.log y = 0 → k * (x + 2 * y) ≤ x^2 + 4 * y^2) ∧
  (∀ (k : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → Real.log x + Real.log y = 0 → k * (x + 2 * y) ≤ x^2 + 4 * y^2) → k ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1051_105107


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1051_105100

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (-3 + I) / (2 + I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1051_105100


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l1051_105147

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that given the specific amounts in the problem, the remaining money is 7 -/
theorem amanda_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l1051_105147


namespace NUMINAMATH_CALUDE_largest_less_than_point_seven_l1051_105168

theorem largest_less_than_point_seven : 
  let numbers : List ℝ := [0.8, 1/2, 0.9]
  let target : ℝ := 0.7
  (∀ x ∈ numbers, x ≤ target → x ≤ (1/2 : ℝ)) ∧ 
  ((1/2 : ℝ) ∈ numbers) ∧ 
  ((1/2 : ℝ) < target) := by
  sorry

end NUMINAMATH_CALUDE_largest_less_than_point_seven_l1051_105168


namespace NUMINAMATH_CALUDE_jean_kept_fraction_l1051_105183

theorem jean_kept_fraction (total : ℕ) (janet_got : ℕ) (janet_fraction : ℚ) :
  total = 60 →
  janet_got = 10 →
  janet_fraction = 1/4 →
  (total - (janet_got / janet_fraction)) / total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jean_kept_fraction_l1051_105183


namespace NUMINAMATH_CALUDE_xyz_value_l1051_105113

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1051_105113


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1051_105106

theorem opposite_of_negative_two : -((-2) : ℤ) = (2 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1051_105106


namespace NUMINAMATH_CALUDE_pendant_prices_and_optimal_plan_l1051_105145

/-- The price of a "Bing Dwen Dwen" pendant in yuan -/
def bing_price : ℝ := 8

/-- The price of a "Shuey Rong Rong" pendant in yuan -/
def shuey_price : ℝ := 10

/-- The cost of 2 "Bing Dwen Dwen" and 1 "Shuey Rong Rong" pendants -/
def cost1 : ℝ := 26

/-- The cost of 4 "Bing Dwen Dwen" and 3 "Shuey Rong Rong" pendants -/
def cost2 : ℝ := 62

/-- The total number of pendants to purchase -/
def total_pendants : ℕ := 100

/-- The number of "Bing Dwen Dwen" pendants in the optimal plan -/
def optimal_bing : ℕ := 75

/-- The number of "Shuey Rong Rong" pendants in the optimal plan -/
def optimal_shuey : ℕ := 25

/-- The minimum cost for the optimal plan -/
def min_cost : ℝ := 850

theorem pendant_prices_and_optimal_plan :
  (2 * bing_price + shuey_price = cost1) ∧
  (4 * bing_price + 3 * shuey_price = cost2) ∧
  (optimal_bing + optimal_shuey = total_pendants) ∧
  (3 * optimal_shuey ≥ optimal_bing) ∧
  (optimal_bing * bing_price + optimal_shuey * shuey_price = min_cost) ∧
  (∀ x y : ℕ, x + y = total_pendants → 3 * y ≥ x → 
    x * bing_price + y * shuey_price ≥ min_cost) :=
by sorry

#check pendant_prices_and_optimal_plan

end NUMINAMATH_CALUDE_pendant_prices_and_optimal_plan_l1051_105145


namespace NUMINAMATH_CALUDE_cassidy_poster_collection_l1051_105146

theorem cassidy_poster_collection (current_posters : ℕ) : current_posters = 22 :=
  by
  have two_years_ago : ℕ := 14
  have after_summer : ℕ := current_posters + 6
  have double_two_years_ago : after_summer = 2 * two_years_ago := by sorry
  sorry

end NUMINAMATH_CALUDE_cassidy_poster_collection_l1051_105146


namespace NUMINAMATH_CALUDE_angle_bisector_property_l1051_105176

theorem angle_bisector_property (x : ℝ) : 
  x > 0 ∧ x < 180 →
  x / 2 = (180 - x) / 3 →
  x = 72 := by
sorry

end NUMINAMATH_CALUDE_angle_bisector_property_l1051_105176


namespace NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l1051_105131

theorem gcd_nine_factorial_six_factorial_squared : 
  Nat.gcd (Nat.factorial 9) ((Nat.factorial 6)^2) = 43200 := by
  sorry

end NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l1051_105131


namespace NUMINAMATH_CALUDE_tractors_count_l1051_105112

/-- Represents the number of tractors initially ploughing the field -/
def T : ℕ := sorry

/-- The area of the field in hectares -/
def field_area : ℕ := sorry

/-- Each tractor ploughs this many hectares per day -/
def hectares_per_tractor_per_day : ℕ := 120

/-- The number of days it takes all tractors to plough the field -/
def days_all_tractors : ℕ := 4

/-- The number of tractors remaining after two are removed -/
def remaining_tractors : ℕ := 4

/-- The number of days it takes the remaining tractors to plough the field -/
def days_remaining_tractors : ℕ := 5

theorem tractors_count :
  (T * hectares_per_tractor_per_day * days_all_tractors = field_area) ∧
  (remaining_tractors * hectares_per_tractor_per_day * days_remaining_tractors = field_area) ∧
  (T = remaining_tractors + 2) →
  T = 10 := by sorry

end NUMINAMATH_CALUDE_tractors_count_l1051_105112


namespace NUMINAMATH_CALUDE_fish_ratio_problem_l1051_105152

/-- The ratio of tagged fish to total fish in a second catch -/
def fish_ratio (tagged_initial : ℕ) (second_catch : ℕ) (tagged_in_catch : ℕ) (total_fish : ℕ) : ℚ :=
  tagged_in_catch / second_catch

/-- Theorem stating the ratio of tagged fish to total fish in the second catch -/
theorem fish_ratio_problem :
  let tagged_initial : ℕ := 30
  let second_catch : ℕ := 50
  let tagged_in_catch : ℕ := 2
  let total_fish : ℕ := 750
  fish_ratio tagged_initial second_catch tagged_in_catch total_fish = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_fish_ratio_problem_l1051_105152


namespace NUMINAMATH_CALUDE_unique_x_value_l1051_105155

theorem unique_x_value : ∃! (x : ℝ), x^2 ∈ ({1, 0, x} : Set ℝ) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l1051_105155


namespace NUMINAMATH_CALUDE_fruit_platter_grapes_l1051_105133

theorem fruit_platter_grapes :
  ∀ (b r g c : ℚ),
  b + r + g + c = 360 →
  r = 3 * b →
  g = 4 * c →
  c = 5 * r →
  g = 21600 / 79 := by
sorry

end NUMINAMATH_CALUDE_fruit_platter_grapes_l1051_105133


namespace NUMINAMATH_CALUDE_pam_current_age_l1051_105140

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the current state -/
structure CurrentState where
  pam_age : Age
  rena_age : Age

/-- Represents the future state after 10 years -/
structure FutureState where
  pam_age : Age
  rena_age : Age

/-- The conditions of the problem -/
def problem_conditions (current : CurrentState) (future : FutureState) : Prop :=
  (current.pam_age.years * 2 = current.rena_age.years) ∧
  (future.rena_age.years = future.pam_age.years + 5) ∧
  (future.pam_age.years = current.pam_age.years + 10) ∧
  (future.rena_age.years = current.rena_age.years + 10)

/-- The theorem to prove -/
theorem pam_current_age
  (current : CurrentState)
  (future : FutureState)
  (h : problem_conditions current future) :
  current.pam_age.years = 5 := by
  sorry

end NUMINAMATH_CALUDE_pam_current_age_l1051_105140


namespace NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l1051_105170

/-- A right triangle with specific properties -/
structure SpecialRightTriangle where
  /-- Length of the shorter leg -/
  short_leg : ℝ
  /-- Length of the longer leg -/
  long_leg : ℝ
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The longer leg is 2 feet longer than twice the shorter leg -/
  leg_relation : long_leg = 2 * short_leg + 2
  /-- The area of the triangle is 96 square feet -/
  area_constraint : (1 / 2) * short_leg * long_leg = 96
  /-- Pythagorean theorem holds -/
  pythagorean : short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2

/-- Theorem: The hypotenuse of the special right triangle is √388 feet -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) :
  t.hypotenuse = Real.sqrt 388 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l1051_105170


namespace NUMINAMATH_CALUDE_large_cube_probabilities_l1051_105187

/-- Represents a large cube composed of 27 smaller dice -/
structure LargeCube where
  dice : Fin 27 → Die

/-- Represents a single die -/
structure Die where
  faces : Fin 6 → Nat

/-- Represents the position of a die in the large cube -/
inductive Position
  | FaceCenter
  | Edge
  | Corner

/-- Returns the position of a die given its index in the large cube -/
def diePosition (i : Fin 27) : Position := sorry

/-- Returns the probability of a specific face showing based on the die's position -/
def faceProbability (p : Position) (face : Nat) : ℚ := sorry

/-- Calculates the probability of exactly 25 sixes showing on the surface -/
def probExactly25Sixes (c : LargeCube) : ℚ := sorry

/-- Calculates the probability of at least one 'one' showing on the surface -/
def probAtLeastOne1 (c : LargeCube) : ℚ := sorry

/-- Calculates the expected number of sixes showing on the surface -/
def expectedSixes (c : LargeCube) : ℚ := sorry

/-- Calculates the expected sum of the numbers showing on the surface -/
def expectedSum (c : LargeCube) : ℚ := sorry

/-- Calculates the expected number of distinct digits appearing on the surface -/
def expectedDistinctDigits (c : LargeCube) : ℚ := sorry

theorem large_cube_probabilities (c : LargeCube) :
  probExactly25Sixes c = 31 / (2^13 * 3^18) ∧
  probAtLeastOne1 c = 1 - (5^6 / (2^2 * 3^18)) ∧
  expectedSixes c = 9 ∧
  expectedSum c = 189 ∧
  expectedDistinctDigits c = 6 * (1 - (5^6 / (2^2 * 3^18))) := by
  sorry

end NUMINAMATH_CALUDE_large_cube_probabilities_l1051_105187


namespace NUMINAMATH_CALUDE_study_group_size_l1051_105129

theorem study_group_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n * (n - 1) = 90 ∧ 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_study_group_size_l1051_105129


namespace NUMINAMATH_CALUDE_x_value_approximation_l1051_105157

/-- The value of x in the given equation is approximately 179692.08 -/
theorem x_value_approximation : 
  let x := 3.5 * ((3.6 * 0.48 * 2.50)^2 / (0.12 * 0.09 * 0.5)) * Real.log (2.5 * 4.3)
  ∃ ε > 0, |x - 179692.08| < ε :=
by sorry

end NUMINAMATH_CALUDE_x_value_approximation_l1051_105157


namespace NUMINAMATH_CALUDE_function_property_l1051_105172

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4

theorem function_property (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 → |f a x₁ - f a x₂| < 4) →
  0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1051_105172


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_and_product_l1051_105164

theorem largest_gcd_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 1130)
  (prod_eq : x * y = 100000) :
  ∃ (a b : ℕ+), a + b = 1130 ∧ a * b = 100000 ∧ 
    ∀ (c d : ℕ+), c + d = 1130 → c * d = 100000 → Nat.gcd c d ≤ Nat.gcd a b ∧ Nat.gcd a b = 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_and_product_l1051_105164


namespace NUMINAMATH_CALUDE_stock_percentage_sold_l1051_105137

/-- Proves that the percentage of stock sold is 0.25% given the specified conditions --/
theorem stock_percentage_sold (cash_realized : ℝ) (brokerage_rate : ℝ) (net_amount : ℝ)
  (h1 : cash_realized = 108.25)
  (h2 : brokerage_rate = 1 / 4 / 100)
  (h3 : net_amount = 108) :
  let brokerage_fee := cash_realized * brokerage_rate
  let percentage_sold := brokerage_fee / cash_realized * 100
  percentage_sold = 0.25 := by sorry

end NUMINAMATH_CALUDE_stock_percentage_sold_l1051_105137


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1051_105162

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → 3*x^2 + 9*x - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1051_105162


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l1051_105139

theorem line_passes_through_quadrants 
  (a b c : ℝ) 
  (h1 : a * b < 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), 
    (a * x + b * y + c = 0) ∧ 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l1051_105139


namespace NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l1051_105153

theorem complex_roots_equilateral_triangle (p q z₁ z₂ : ℂ) :
  z₂^2 + p*z₂ + q = 0 →
  z₁^2 + p*z₁ + q = 0 →
  z₂ = Complex.exp (2*Real.pi*Complex.I/3) * z₁ →
  p^2 / q = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l1051_105153


namespace NUMINAMATH_CALUDE_modulus_of_z_l1051_105128

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1051_105128


namespace NUMINAMATH_CALUDE_divisibility_implies_inequality_l1051_105159

theorem divisibility_implies_inequality (a k : ℕ+) 
  (h : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : 
  k ≥ a := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_inequality_l1051_105159


namespace NUMINAMATH_CALUDE_prob_two_red_from_bag_l1051_105181

/-- The probability of picking two red balls from a bag -/
def probability_two_red_balls (red blue green : ℕ) : ℚ :=
  let total := red + blue + green
  (red : ℚ) / total * ((red - 1) : ℚ) / (total - 1)

/-- Theorem: The probability of picking two red balls from a bag with 3 red, 2 blue, and 4 green balls is 1/12 -/
theorem prob_two_red_from_bag : probability_two_red_balls 3 2 4 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_from_bag_l1051_105181


namespace NUMINAMATH_CALUDE_simplify_expression_l1051_105165

theorem simplify_expression : 
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1051_105165


namespace NUMINAMATH_CALUDE_unique_third_rectangle_dimensions_l1051_105143

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Given three rectangles that form a larger rectangle without gaps and overlapping,
    where two of the rectangles are 3 cm × 8 cm and 2 cm × 5 cm,
    prove that there is only one possible set of dimensions for the third rectangle -/
theorem unique_third_rectangle_dimensions (r1 r2 r3 : Rectangle)
  (h1 : r1.width = 3 ∧ r1.height = 8)
  (h2 : r2.width = 2 ∧ r2.height = 5)
  (h_total_area : r1.area + r2.area + r3.area = (r1.width + r2.width + r3.width) * (r1.height + r2.height + r3.height)) :
  r3.width = 4 ∧ r3.height = 1 ∨ r3.width = 1 ∧ r3.height = 4 := by
  sorry

#check unique_third_rectangle_dimensions

end NUMINAMATH_CALUDE_unique_third_rectangle_dimensions_l1051_105143


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1051_105149

theorem sum_of_squares_lower_bound (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_eq_6 : a + b + c = 6) : 
  a^2 + b^2 + c^2 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1051_105149


namespace NUMINAMATH_CALUDE_system_solution_l1051_105188

theorem system_solution (x y : ℚ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : 
  (x + y) / 3 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1051_105188


namespace NUMINAMATH_CALUDE_complex_fourth_power_l1051_105156

theorem complex_fourth_power (z : ℂ) : z = Complex.I * Real.sqrt 2 → z^4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l1051_105156


namespace NUMINAMATH_CALUDE_func_f_properties_l1051_105127

/-- A function satisfying the given functional equation -/
noncomputable def FuncF (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * f b) ∧ 
  (f 0 ≠ 0) ∧
  (∃ c : ℝ, c > 0 ∧ f (c / 2) = 0)

theorem func_f_properties (f : ℝ → ℝ) (h : FuncF f) :
  (f 0 = 1) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, f (x + 2 * c) = f x) :=
by sorry

end NUMINAMATH_CALUDE_func_f_properties_l1051_105127


namespace NUMINAMATH_CALUDE_exists_greater_term_l1051_105118

/-- Two sequences of positive reals satisfying given recurrence relations -/
def SequencePair (x y : ℕ → ℝ) : Prop :=
  (∀ n, x n > 0 ∧ y n > 0) ∧
  (∀ n, x (n + 2) = x n + (x (n + 1))^2) ∧
  (∀ n, y (n + 2) = (y n)^2 + y (n + 1)) ∧
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

/-- There exists a k such that x_k > y_k -/
theorem exists_greater_term (x y : ℕ → ℝ) (h : SequencePair x y) :
  ∃ k, x k > y k := by
  sorry

end NUMINAMATH_CALUDE_exists_greater_term_l1051_105118


namespace NUMINAMATH_CALUDE_monomial_division_equality_l1051_105123

theorem monomial_division_equality (x y : ℝ) (m n : ℤ) :
  (x^m * y^n) / ((1/4) * x^3 * y) = 4 * x^2 ↔ m = 5 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_monomial_division_equality_l1051_105123


namespace NUMINAMATH_CALUDE_ray_gave_peter_30_cents_l1051_105178

/-- Given that Ray has 175 cents in nickels, gives twice as many cents to Randi as to Peter,
    and Randi has 6 more nickels than Peter, prove that Ray gave 30 cents to Peter. -/
theorem ray_gave_peter_30_cents (total : ℕ) (peter_cents : ℕ) (randi_cents : ℕ) : 
  total = 175 →
  randi_cents = 2 * peter_cents →
  randi_cents = peter_cents + 6 * 5 →
  peter_cents = 30 := by
sorry

end NUMINAMATH_CALUDE_ray_gave_peter_30_cents_l1051_105178


namespace NUMINAMATH_CALUDE_two_digit_addition_proof_l1051_105132

theorem two_digit_addition_proof (A B C : ℕ) : 
  A ≠ B → A ≠ C → B ≠ C →
  A < 10 → B < 10 → C < 10 →
  A > 0 → B > 0 → C > 0 →
  (10 * A + B) + (10 * B + C) = 100 * B + 10 * C + B →
  A = 9 := by
sorry

end NUMINAMATH_CALUDE_two_digit_addition_proof_l1051_105132


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1051_105148

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1051_105148


namespace NUMINAMATH_CALUDE_f_monotonicity_l1051_105119

noncomputable def f (x : ℝ) : ℝ := -2 * x / (1 + x^2)

theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_l1051_105119


namespace NUMINAMATH_CALUDE_student_survey_l1051_105198

theorem student_survey (total : ℕ) (mac_preference : ℕ) (both_preference : ℕ) :
  total = 210 →
  mac_preference = 60 →
  both_preference = mac_preference / 3 →
  total - (mac_preference + both_preference) = 130 := by
  sorry

end NUMINAMATH_CALUDE_student_survey_l1051_105198


namespace NUMINAMATH_CALUDE_b_equals_one_l1051_105184

-- Define the variables
variable (a b y : ℝ)

-- Define the conditions
def condition1 : Prop := |b - y| = b + y - a
def condition2 : Prop := |b + y| = b + a

-- State the theorem
theorem b_equals_one (h1 : condition1 a b y) (h2 : condition2 a b y) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_equals_one_l1051_105184


namespace NUMINAMATH_CALUDE_angle_CBO_is_20_degrees_l1051_105117

-- Define the triangle ABC
variable (A B C O : Point) (ABC : Triangle A B C)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem angle_CBO_is_20_degrees 
  (h1 : angle B A O = angle C A O)
  (h2 : angle C B O = angle A B O)
  (h3 : angle A C O = angle B C O)
  (h4 : angle A O C = 110)
  (h5 : ∀ P Q R : Point, angle P Q R + angle Q R P + angle R P Q = 180) :
  angle C B O = 20 := by sorry

end NUMINAMATH_CALUDE_angle_CBO_is_20_degrees_l1051_105117


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_exponent_l1051_105180

theorem divisibility_of_power_plus_exponent (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, n ∣ (2^m + m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_exponent_l1051_105180


namespace NUMINAMATH_CALUDE_blind_box_probabilities_l1051_105160

def total_boxes : ℕ := 7
def rabbit_boxes : ℕ := 4
def dog_boxes : ℕ := 3

theorem blind_box_probabilities :
  (∀ (n m : ℕ), n + m = total_boxes → n = rabbit_boxes → m = dog_boxes →
    (Nat.choose rabbit_boxes 1 * Nat.choose (total_boxes - 1) 1 ≠ 0 →
      (Nat.choose rabbit_boxes 1 * Nat.choose (rabbit_boxes - 1) 1 : ℚ) /
      (Nat.choose rabbit_boxes 1 * Nat.choose (total_boxes - 1) 1 : ℚ) = 1 / 2)) ∧
  (∀ (n m : ℕ), n + m = total_boxes → n = rabbit_boxes → m = dog_boxes →
    (Nat.choose total_boxes 1 ≠ 0 →
      (Nat.choose dog_boxes 1 : ℚ) / (Nat.choose total_boxes 1 : ℚ) = 3 / 7)) :=
sorry

end NUMINAMATH_CALUDE_blind_box_probabilities_l1051_105160


namespace NUMINAMATH_CALUDE_hiking_team_selection_l1051_105124

theorem hiking_team_selection (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_selection_l1051_105124


namespace NUMINAMATH_CALUDE_loss_percent_calculation_l1051_105189

theorem loss_percent_calculation (cost_price selling_price : ℝ) : 
  cost_price = 600 → 
  selling_price = 550 → 
  (cost_price - selling_price) / cost_price * 100 = 8.33 := by
sorry

end NUMINAMATH_CALUDE_loss_percent_calculation_l1051_105189


namespace NUMINAMATH_CALUDE_decreasing_function_implies_b_geq_4_l1051_105158

-- Define the function y
def y (x b : ℝ) : ℝ := x^3 - 3*b*x + 1

-- State the theorem
theorem decreasing_function_implies_b_geq_4 :
  ∀ b : ℝ, (∀ x ∈ Set.Ioo 1 2, ∀ h > 0, x + h ∈ Set.Ioo 1 2 → y (x + h) b < y x b) →
  b ≥ 4 := by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_b_geq_4_l1051_105158


namespace NUMINAMATH_CALUDE_sand_weight_difference_l1051_105111

def box_weight : ℕ := 250
def box_filled_weight : ℕ := 1780
def bucket_weight : ℕ := 460
def bucket_filled_weight : ℕ := 2250

theorem sand_weight_difference :
  (bucket_filled_weight - bucket_weight) - (box_filled_weight - box_weight) = 260 :=
by sorry

end NUMINAMATH_CALUDE_sand_weight_difference_l1051_105111


namespace NUMINAMATH_CALUDE_sum_of_six_least_l1051_105104

/-- τ(n) denotes the number of positive integer divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The set of positive integers n that satisfy τ(n) + τ(n+1) = 8 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ tau n + tau (n + 1) = 8}

/-- The six least elements of S -/
def six_least : Finset ℕ := sorry

theorem sum_of_six_least : (six_least.sum id) = 800 := by sorry

end NUMINAMATH_CALUDE_sum_of_six_least_l1051_105104


namespace NUMINAMATH_CALUDE_tree_distance_l1051_105190

/-- Given 10 equally spaced trees along a road, with 100 feet between the 1st and 5th tree,
    the distance between the 1st and 10th tree is 225 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 10) (h2 : d = 100) :
  let space := d / 4
  (n - 1) * space = 225 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l1051_105190


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1051_105195

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1/4 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1051_105195


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1051_105102

theorem subtraction_of_fractions : (5 : ℚ) / 6 - (1 : ℚ) / 3 = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1051_105102


namespace NUMINAMATH_CALUDE_kimberly_peanut_shopping_l1051_105177

/-- Kimberly's peanut shopping theorem -/
theorem kimberly_peanut_shopping (store_visits : ℕ) (total_peanuts : ℕ) 
  (h1 : store_visits = 3) 
  (h2 : total_peanuts = 21) : 
  total_peanuts / store_visits = 7 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_peanut_shopping_l1051_105177


namespace NUMINAMATH_CALUDE_friday_ice_cream_amount_l1051_105114

/-- The amount of ice cream eaten on Friday night, given the total amount eaten over two nights and the amount eaten on Saturday night. -/
theorem friday_ice_cream_amount (total : ℝ) (saturday : ℝ) (h1 : total = 3.5) (h2 : saturday = 0.25) :
  total - saturday = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_friday_ice_cream_amount_l1051_105114


namespace NUMINAMATH_CALUDE_periodic_last_digit_triangular_perfect_square_between_sums_l1051_105135

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def last_digit (n : ℕ) : ℕ := n % 10

def sum_triangular_numbers (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem periodic_last_digit_triangular :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, last_digit (triangular_number n) = last_digit (triangular_number (n + k)) :=
sorry

theorem perfect_square_between_sums (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, sum_triangular_numbers (n - 1) < k * k ∧ k * k < sum_triangular_numbers n :=
sorry

end NUMINAMATH_CALUDE_periodic_last_digit_triangular_perfect_square_between_sums_l1051_105135


namespace NUMINAMATH_CALUDE_eighth_grade_girls_l1051_105173

/-- Given the number of boys and girls in eighth grade, proves the number of girls -/
theorem eighth_grade_girls (total : ℕ) (boys girls : ℕ) : 
  total = 68 → 
  boys = 2 * girls - 16 → 
  boys + girls = total → 
  girls = 28 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_girls_l1051_105173
