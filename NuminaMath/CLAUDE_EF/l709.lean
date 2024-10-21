import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_specific_triangle_l709_70904

/-- A triangle with given altitudes -/
structure Triangle where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  h₃_pos : h₃ > 0

/-- The largest angle in a triangle -/
noncomputable def largest_angle (t : Triangle) : ℝ := sorry

/-- Theorem: The largest angle in a triangle with altitudes 10, 24, and 15 is arccos(7/8) -/
theorem largest_angle_specific_triangle :
  ∃ (t : Triangle), t.h₁ = 10 ∧ t.h₂ = 24 ∧ t.h₃ = 15 ∧ largest_angle t = Real.arccos (7/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_specific_triangle_l709_70904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_eq_sum_radii_problem_solution_l709_70942

/-- Two circles that are externally tangent -/
structure ExternallyTangentCircles where
  center_A : ℝ × ℝ
  center_B : ℝ × ℝ
  radius_A : ℝ
  radius_B : ℝ
  tangent_point : ℝ × ℝ
  h_positive_radii : 0 < radius_A ∧ 0 < radius_B
  h_externally_tangent : dist center_A tangent_point = radius_A ∧
                         dist center_B tangent_point = radius_B ∧
                         dist center_A center_B = radius_A + radius_B

/-- The distance between centers of externally tangent circles equals the sum of their radii -/
theorem distance_centers_eq_sum_radii (c : ExternallyTangentCircles) :
  dist c.center_A c.center_B = c.radius_A + c.radius_B := by
  sorry

/-- For the given problem, the distance between centers is 13 -/
theorem problem_solution (c : ExternallyTangentCircles) 
  (h_radius_A : c.radius_A = 10)
  (h_radius_B : c.radius_B = 3) :
  dist c.center_A c.center_B = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_eq_sum_radii_problem_solution_l709_70942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_two_solutions_l709_70965

open Real

/-- The set of angles a for which the system has two solutions -/
def solution_set : Set ℝ :=
  {a | ∃ n : ℤ, 
    (π/2 + n*π < a ∧ a < 3*π/4 - arcsin (sqrt 2/6) + n*π) ∨
    (3*π/4 + arcsin (sqrt 2/6) + n*π < a ∧ a < π + n*π)}

/-- The system of equations -/
def system (x y a : ℝ) : Prop :=
  x * sin a - (y - 6) * cos a = 0 ∧
  ((x - 3)^2 + (y - 3)^2 - 1) * ((x - 3)^2 + (y - 3)^2 - 9) = 0

theorem system_has_two_solutions (a : ℝ) :
  a ∈ solution_set ↔ ∃! x₁ y₁ x₂ y₂ : ℝ, 
    system x₁ y₁ a ∧ system x₂ y₂ a ∧ (x₁, y₁) ≠ (x₂, y₂) :=
sorry

#check system_has_two_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_two_solutions_l709_70965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_45_degrees_l709_70983

/-- The slope angle of a line with equation ax + by + c = 0 -/
noncomputable def slopeAngle (a b : ℝ) : ℝ := Real.arctan (- a / b)

theorem line_slope_angle_45_degrees :
  slopeAngle 1 (-1) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_45_degrees_l709_70983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_range_l709_70962

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x else Real.exp x

theorem f_less_than_one_range :
  {x : ℝ | f x < 1} = Set.Iio 0 ∪ Set.Ioo 1 (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_range_l709_70962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_element_in_two_pairs_l709_70961

/-- A structure representing a system of elements and pairs -/
structure ElementPairSystem (n : ℕ) where
  /-- The set of elements -/
  elements : Finset (Fin n)
  /-- The set of pairs -/
  pairs : Finset (Finset (Fin n))
  /-- The number of elements equals n -/
  elements_count : elements.card = n
  /-- The number of pairs equals n -/
  pairs_count : pairs.card = n
  /-- Each pair contains exactly two elements -/
  pair_size : ∀ p, p ∈ pairs → p.card = 2
  /-- The relation between elements and pairs -/
  pair_relation : ∀ i j : Fin n, i ∈ elements → j ∈ elements →
    (∃ p ∈ pairs, i ∈ p ∧ j ∈ p) ↔ 
    (∃ p₁ p₂, p₁ ∈ pairs ∧ p₂ ∈ pairs ∧ i ∈ p₁ ∧ j ∈ p₂ ∧ (p₁ ∩ p₂).Nonempty)

/-- Main theorem: Each element belongs to exactly two pairs -/
theorem element_in_two_pairs {n : ℕ} (sys : ElementPairSystem n) :
  ∀ i, i ∈ sys.elements → (sys.pairs.filter (λ p => i ∈ p)).card = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_element_in_two_pairs_l709_70961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l709_70909

theorem problem_statement (x n : ℝ) (f : ℝ) :
  x = (3 + Real.sqrt 8)^500 →
  n = ⌊x⌋ →
  f = x - n →
  x^2 * (1 - f) = (3 + Real.sqrt 8)^500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l709_70909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PADB_l709_70939

/-- The function f(x) = 2√(2x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt (2 * x)

/-- The circle D: x² + y² - 4x + 3 = 0 -/
def circle_D (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- Point P is on the graph of f -/
def point_on_f (P : ℝ × ℝ) : Prop := P.2 = f P.1

/-- A and B are points on the circle D -/
def points_on_circle (A B : ℝ × ℝ) : Prop := circle_D A.1 A.2 ∧ circle_D B.1 B.2

/-- The line PA is tangent to the circle at A -/
def tangent_line (P A : ℝ × ℝ) : Prop := sorry

/-- The area of quadrilateral PADB -/
noncomputable def area_PADB (P A D B : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_area_PADB :
  ∃ (P A B : ℝ × ℝ) (D : ℝ × ℝ),
    D = (2, 0) ∧
    point_on_f P ∧
    points_on_circle A B ∧
    tangent_line P A ∧
    tangent_line P B ∧
    (∀ (P' A' B' : ℝ × ℝ),
      point_on_f P' →
      points_on_circle A' B' →
      tangent_line P' A' →
      tangent_line P' B' →
      area_PADB P A D B ≤ area_PADB P' A' D B') ∧
    area_PADB P A D B = Real.sqrt 3 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PADB_l709_70939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l709_70984

noncomputable section

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the eccentricity
def eccentricity (a c : ℝ) : ℝ := c / a

theorem ellipse_properties 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_M : ellipse_C (Real.sqrt 2 / 2) (Real.sqrt 3 / 2) a b)
  (h_e : eccentricity a (Real.sqrt (a^2 - b^2)) = Real.sqrt 2 / 2)
  (m : ℝ)
  (h_m : m > 1)
  (h_A : ellipse_C m 0 a b ∧ circle_O m 0)
  (l₁ l₂ : ℝ → ℝ)
  (h_slopes : (deriv l₁ 0) * (deriv l₂ 0) = 1)
  (h_tangent : ∃ P : ℝ × ℝ, circle_O (l₁ P.1) (l₁ P.2) ∧ (deriv l₁ P.1) = P.2 / P.1)
  (h_intersect : ∃ M N : ℝ × ℝ, M ≠ N ∧ ellipse_C (l₂ M.1) (l₂ M.2) a b ∧ ellipse_C (l₂ N.1) (l₂ N.2) a b) :
  (∀ x y, ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (1 < m ∧ m < (Real.sqrt 5 + 1) / 2) ∧
  (∃ S : ℝ → ℝ → ℝ → ℝ, 
    (∀ O M N, S O M N ≤ Real.sqrt 2 / 2) ∧
    (∃ O M N, S O M N = Real.sqrt 2 / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l709_70984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_is_50_l709_70913

noncomputable def count_integers_satisfying_inequality : ℕ :=
  (Finset.filter (fun n : ℕ => n > 9 ∧ n < 60 ∧ (60 * n : ℝ)^40 > (n : ℝ)^80 ∧ (n : ℝ)^80 > 3^160) (Finset.range 60)).card

theorem count_integers_satisfying_inequality_is_50 : 
  count_integers_satisfying_inequality = 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_is_50_l709_70913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_decimal_l709_70970

/-- The number of zeros in the decimal representation of (7^30 - 1)/(7^6 - 1) -/
def num_zeros : ℕ := 470588

/-- R_k is defined as (7^k - 1)/6 -/
def R (k : ℕ) : ℚ := (7^k - 1)/6

/-- Q is defined as R_30 / R_6 -/
def Q : ℚ := R 30 / R 6

/-- Auxiliary function to count zeros in a list of digits -/
def count_zeros (digits : List ℕ) : ℕ := digits.count 0

/-- Theorem stating that the number of zeros in the decimal representation of Q is num_zeros -/
theorem zeros_in_Q_decimal : count_zeros (Q.num.natAbs.digits 10) = num_zeros := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_decimal_l709_70970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_kth_powers_divisible_l709_70902

/-- The set of positive integers less than n that are relatively prime to n -/
def relativelyPrimeSet (n : ℕ) : Finset ℕ :=
  Finset.filter (fun a => Nat.Coprime a n) (Finset.range n)

/-- The sum of k-th powers of elements in the relatively prime set -/
def sumOfKthPowers (n : ℕ) (k : ℕ) : ℕ :=
  (relativelyPrimeSet n).sum (fun a => a^k)

/-- Main theorem -/
theorem sum_of_kth_powers_divisible (n : ℕ) (h1 : n ≥ 2) :
  ∀ k : ℕ, k > 0 →
  (∀ p : ℕ, Nat.Prime p → p ∣ (relativelyPrimeSet n).card → p ∣ n) →
  (relativelyPrimeSet n).card ∣ sumOfKthPowers n k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_kth_powers_divisible_l709_70902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_problem_l709_70910

theorem greatest_integer_problem : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k m : ℕ), n = 9 * k - 1 ∧ n = 5 * m - 2) ∧
  (∀ (x : ℕ), x < 150 → 
    (∃ (k' m' : ℕ), x = 9 * k' - 1 ∧ x = 5 * m' - 2) → 
    x ≤ n) ∧
  n = 143 := by
  -- Proof goes here
  sorry

#check greatest_integer_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_problem_l709_70910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pioneer_circle_attendance_l709_70985

theorem pioneer_circle_attendance 
  (pioneers : Finset ℕ) 
  (circles : Finset ℕ) 
  (attendance : ℕ → Finset ℕ) 
  (h1 : pioneers.card = 11) 
  (h2 : circles.card = 5) 
  (h3 : ∀ p, p ∈ pioneers → attendance p ⊆ circles) :
  ∃ a b, a ∈ pioneers ∧ b ∈ pioneers ∧ a ≠ b ∧ attendance a ⊆ attendance b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pioneer_circle_attendance_l709_70985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l709_70990

noncomputable def f (x : ℝ) : ℝ := (2 * Real.exp x) / (1 + Real.exp x) + 1/2

-- Theorem statement
theorem f_is_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l709_70990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l709_70918

/-- The distance from a point to a plane defined by three points -/
noncomputable def distanceToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_to_specific_plane :
  let M₀ : ℝ × ℝ × ℝ := (-2, 4, 2)
  let M₁ : ℝ × ℝ × ℝ := (1, -1, 1)
  let M₂ : ℝ × ℝ × ℝ := (-2, 0, 3)
  let M₃ : ℝ × ℝ × ℝ := (2, 1, -1)
  distanceToPlane M₀ M₁ M₂ M₃ = 9 / Real.sqrt 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l709_70918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_origin_l709_70929

/-- A line with slope 0.2 passing through the midpoint of a rectangle also passes through the origin -/
theorem line_passes_through_origin (a b c d : ℝ × ℝ) :
  a = (1, 0) →
  b = (9, 0) →
  c = (1, 2) →
  d = (9, 2) →
  let midpoint := ((a.1 + d.1) / 2, (a.2 + d.2) / 2)
  let slope := 0.2
  let line := λ (x y : ℝ) ↦ y = slope * x + (midpoint.2 - slope * midpoint.1)
  line midpoint.1 midpoint.2 →
  line 0 0 := by
  sorry

#check line_passes_through_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_origin_l709_70929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_shrink_l709_70927

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the transformation
noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (4 * (x + Real.pi / 2))

-- Theorem statement
theorem sin_shift_shrink :
  ∀ x : ℝ, transform f x = Real.sin (4 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_shrink_l709_70927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2013th_derivative_l709_70938

-- Define f as sin x
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the nth derivative of f
noncomputable def nthDerivative (n : ℕ) (f : ℝ → ℝ) : ℝ → ℝ :=
  match n with
  | 0 => f
  | n + 1 => deriv (nthDerivative n f)

-- Theorem statement
theorem sin_2013th_derivative (x : ℝ) : 
  nthDerivative 2013 f x = Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2013th_derivative_l709_70938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_gasoline_consumption_l709_70912

/-- Calculates the gasoline consumption of a car given its travel time, speed, and fuel efficiency. -/
noncomputable def gasoline_consumption (travel_time_hours : ℝ) (travel_time_minutes : ℝ) (speed_km_per_hour : ℝ) (fuel_efficiency_l_per_km : ℝ) : ℝ :=
  let total_time_hours : ℝ := travel_time_hours + travel_time_minutes / 60
  let distance_km : ℝ := speed_km_per_hour * total_time_hours
  distance_km * fuel_efficiency_l_per_km

/-- Theorem stating that a car traveling for 2 hours and 36 minutes at 80 km/h, 
    consuming 0.08 liters of gasoline per kilometer, will consume 16.64 liters of gasoline. -/
theorem car_gasoline_consumption :
  gasoline_consumption 2 36 80 0.08 = 16.64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_gasoline_consumption_l709_70912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_slope_range_l709_70999

noncomputable section

/-- The function f(x) given in the problem -/
def f (a b x : ℝ) : ℝ := (a * x) / (x^2 + b)

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := (a * (x^2 + b) - 2 * a * x^2) / (x^2 + b)^2

theorem tangent_line_and_slope_range :
  ∃ (a b : ℝ),
    (f a b 1 = 2) ∧
    (f_derivative a b 1 = 0) ∧
    (∀ x, f a b x = (4 * x) / (x^2 + 1)) ∧
    (∀ x₀, -1/2 ≤ f_derivative a b x₀ ∧ f_derivative a b x₀ ≤ 4) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_slope_range_l709_70999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_two_y_squared_l709_70919

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (x y : ℝ) : ℕ → ℝ
  | 0 => 2 * x^2 + 3 * y^2  -- Added case for 0
  | 1 => 2 * x^2 + 3 * y^2
  | 2 => x^2 + 2 * y^2
  | 3 => 2 * x^2 - y^2
  | 4 => x^2 - y^2
  | n + 5 => arithmetic_sequence x y 4 + (n + 1) * (arithmetic_sequence x y 2 - arithmetic_sequence x y 1)

/-- The theorem stating that the fifth term of the sequence is -2y^2 -/
theorem fifth_term_is_negative_two_y_squared (x y : ℝ) :
  arithmetic_sequence x y 5 = -2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_two_y_squared_l709_70919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_activity_involvement_l709_70925

/-- The total number of students involved in a school activity -/
noncomputable def total_involved (y z : ℝ) : ℝ :=
  98 + 98 * z / y

/-- Theorem: Given the conditions, the total number of students involved in the activity is 98 + 98z/y -/
theorem activity_involvement (B G T y z : ℝ) 
  (h1 : B = 0.5 * T)
  (h2 : G = 0.5 * T)
  (h3 : 98 = y / 100 * B)
  (h4 : y > 0)
  (h5 : z ≥ 0) 
  (h6 : z ≤ 100) :
  total_involved y z = 98 + (z / 100) * G := by
  sorry

#check activity_involvement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_activity_involvement_l709_70925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_with_chord_length_l709_70933

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 4

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Define the line equation
def line_equation (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the vertical line equation
def vertical_line_equation (a : ℝ) (x : ℝ) : Prop := x = a

-- Theorem statement
theorem line_through_P_with_chord_length :
  ∃ (m b : ℝ), (line_equation m b point_P.1 point_P.2 ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
      line_equation m b x₁ y₁ ∧ line_equation m b x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) ∨
  (vertical_line_equation 2 point_P.1 ∧
    ∃ (y₁ y₂ : ℝ), 
      my_circle 2 y₁ ∧ my_circle 2 y₂ ∧
      (y₂ - y₁)^2 = chord_length^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_with_chord_length_l709_70933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_sum_of_squares_l709_70931

/-- A line segment connecting two points in 2D space. -/
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Parameterization of a line segment. -/
structure Parameterization where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Theorem stating the sum of squares of parameterization coefficients for a specific line segment. -/
theorem parameterization_sum_of_squares 
  (segment : LineSegment) 
  (param : Parameterization) :
  segment.start = (-2, 7) →
  segment.endpoint = (3, 11) →
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (param.a * t + param.b, param.c * t + param.d) = 
    (segment.start.1 + t * (segment.endpoint.1 - segment.start.1),
     segment.start.2 + t * (segment.endpoint.2 - segment.start.2))) →
  param.a^2 + param.b^2 + param.c^2 + param.d^2 = 94 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_sum_of_squares_l709_70931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_force_for_18_inch_wrench_l709_70922

/-- The constant derived from the inverse relationship between force and length -/
noncomputable def k : ℝ := 3600

/-- The length of the wrench handle in inches -/
noncomputable def L : ℝ := 18

/-- The force required to loosen the bolt -/
noncomputable def F : ℝ := k / L

theorem force_for_18_inch_wrench : F = 200 := by
  -- Unfold the definitions
  unfold F k L
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_force_for_18_inch_wrench_l709_70922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cups_for_continuous_operation_l709_70920

/-- Predicate for continuous operation of filling and emptying machines -/
def continuous_operation (a b : ℕ+) (n : ℕ) : Prop :=
  ∃ (t : ℕ), ∀ (t' : ℕ), t' ≥ t →
    (∃ (empty full : ℕ), 
      empty + full = n ∧
      empty ≥ a.val ∧
      full ≥ b.val)

/-- The minimum number of cups for continuous operation of filling and emptying machines -/
theorem min_cups_for_continuous_operation (a b : ℕ+) :
  ∃ (n : ℕ),
    n = 2 * a.val + 2 * b.val - 2 * Nat.gcd a.val b.val ∧
    (∀ (m : ℕ), m < n → ¬ (continuous_operation a b m)) ∧
    continuous_operation a b n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cups_for_continuous_operation_l709_70920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_intersection_condition_l709_70911

/-- The cubic function f(x) = (1/3)x^3 - 3x + m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - 3*x + m

/-- The derivative of f with respect to x -/
def f' (x : ℝ) : ℝ := x^2 - 3

/-- Condition for f to have exactly two x-axis intersections -/
def has_two_intersections (m : ℝ) : Prop :=
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) ∧
  (∀ x₁ x₂ x₃ : ℝ, (f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m x₃ = 0) → (x₁ = x₂ ∨ x₁ = x₃ ∨ x₂ = x₃))

/-- Theorem stating the condition for the cubic function to have exactly two x-axis intersections -/
theorem cubic_intersection_condition :
  ∀ m : ℝ, has_two_intersections m ↔ (m = -2 * Real.sqrt 3 ∨ m = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_intersection_condition_l709_70911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l709_70966

/-- A hyperbola with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  asymptote_angle : b / a = Real.sqrt 3
  eccentricity_def : e = Real.sqrt (a^2 + b^2) / a

/-- The minimum value of (a^2 + e) / b for a hyperbola with one asymptote at π/3 -/
theorem hyperbola_min_value (h : Hyperbola) :
    ∃ (m : ℝ), m = 2 * Real.sqrt 6 / 3 ∧ ∀ x : ℝ, m ≤ (h.a^2 + h.e) / h.b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l709_70966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l709_70982

theorem triangle_abc_properties (A B C : Real) (BC : Real) :
  (Real.sqrt 3 * Real.sin (2 * B) = 1 - Real.cos (2 * B)) →
  (B = Real.pi / 6) ∧
  (BC = 2 ∧ A = Real.pi / 4 →
    let AC : Real := (2 * Real.sqrt 2) / Real.sqrt 3
    let S : Real := (1 / 2) * AC * BC * Real.sin ((5 * Real.pi) / 12)
    S = (3 + Real.sqrt 3) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l709_70982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_6_plus_alpha_implies_cos_pi_3_minus_alpha_l709_70963

theorem sin_pi_6_plus_alpha_implies_cos_pi_3_minus_alpha (α : ℝ) :
  Real.sin (π / 6 + α) = Real.sqrt 3 / 3 → Real.cos (π / 3 - α) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_6_plus_alpha_implies_cos_pi_3_minus_alpha_l709_70963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_problem_l709_70921

/-- Two distinct quadratic polynomials with specific properties -/
theorem quadratic_polynomials_problem :
  ∃ (a b c d : ℝ) (f g : ℝ → ℝ),
  (∀ x, f x = x^2 + a*x + b) ∧  -- Definition of f
  (∀ x, g x = x^2 + c*x + d) ∧  -- Definition of g
  (f ≠ g) ∧  -- f and g are distinct
  (g (-a/2) = 0) ∧  -- Vertex of f is a root of g
  (f (-c/2) = 0) ∧  -- Vertex of g is a root of f
  (∃ m, ∀ x, f x ≥ m ∧ g x ≥ m) ∧  -- Both polynomials have the same minimum value
  (f 50 = -200 ∧ g 50 = -200) ∧  -- Polynomials intersect at (50, -200)
  a + c = -200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_problem_l709_70921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_foci_distance_l709_70906

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- The ellipse is tangent to the x-axis at (5, 0) and to the line y = 2 at (0, 2) -/
def special_ellipse : ParallelAxisEllipse :=
  { center := (5, 2),
    semi_major_axis := 5,
    semi_minor_axis := 2 }

/-- The distance between the foci of an ellipse -/
noncomputable def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  Real.sqrt (e.semi_major_axis^2 - e.semi_minor_axis^2)

/-- Theorem: The distance between the foci of the special ellipse is √21 -/
theorem special_ellipse_foci_distance :
  foci_distance special_ellipse = Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_foci_distance_l709_70906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_segments_specific_rectangle_l709_70977

/-- Represents a rectangle with sides a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the sum of lengths of all segments in the construction -/
noncomputable def sumOfSegments (rect : Rectangle) (n : ℕ) : ℝ :=
  (n - 1) * Real.sqrt (rect.a^2 + rect.b^2)

/-- Theorem stating the sum of segment lengths for the specific rectangle and division -/
theorem sum_of_segments_specific_rectangle :
  let rect := Rectangle.mk 5 4
  let n := 200
  sumOfSegments rect n = 199 * Real.sqrt 41 := by
  sorry

#check sum_of_segments_specific_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_segments_specific_rectangle_l709_70977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_H_in_right_triangle_l709_70997

/-- Given a right triangle HFG with HG = 12 and FG = 13, prove that sin H = 5/13 -/
theorem sin_H_in_right_triangle (H F G : ℝ × ℝ) : 
  -- Right triangle condition
  (F.1 - H.1) * (G.1 - H.1) + (F.2 - H.2) * (G.2 - H.2) = 0 →
  -- HG = 12 condition
  Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = 12 →
  -- FG = 13 condition
  Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) = 13 →
  -- Conclusion: sin H = 5/13
  Real.sin (Real.arctan ((F.2 - H.2) / (F.1 - H.1))) = 5 / 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_H_in_right_triangle_l709_70997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l709_70981

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.sqrt 3

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : g a (g a (Real.sqrt 3)) = -Real.sqrt 3) :
  a = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l709_70981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l709_70955

-- Define the function f(x) = (x-2)/(x-4)
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 4)

-- Define y as 2
def y : ℝ := 2

-- Theorem statement
theorem solution_set (x : ℝ) : f x ≤ 3 ∧ x ≠ y ↔ 4 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l709_70955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l709_70975

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A line intersecting an ellipse at two points -/
structure IntersectingLine (e : Ellipse) where
  slope : ℝ
  x_intersect : ℝ
  h_on_ellipse : x_intersect^2 / e.a^2 + (slope * x_intersect)^2 / e.b^2 = 1

theorem ellipse_slope_product (e : Ellipse) (l : IntersectingLine e) :
  e.eccentricity = Real.sqrt 2 / 2 →
  ∃ (k₁ k₂ : ℝ), k₁ * k₂ = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l709_70975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_cost_effective_room_A_l709_70964

-- Define the price range
def PriceRange (x : ℝ) : Prop := 0 < x ∧ x < 200

-- Define the payment function for Room A
def PaymentA (x : ℝ) : ℝ := 0.6 * x

-- Define the payment function for Room B
noncomputable def PaymentB (x : ℝ) : ℝ :=
  if x < 100 then x else x - 50

-- Theorem statement
theorem more_cost_effective_room_A (x : ℝ) (h : PriceRange x) :
  PaymentA x < PaymentB x ↔ x < 125 := by
  sorry

#check more_cost_effective_room_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_cost_effective_room_A_l709_70964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_l709_70948

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the coordinate plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The problem setup -/
structure Problem where
  A : Point
  B : Point
  C : Point
  line1 : Line
  line2 : Line
  line3 : Line
  a : ℝ
  b : ℝ

/-- The conditions of the problem -/
def conditions (p : Problem) : Prop :=
  p.B.y = 0 ∧  -- B is on Ox axis
  p.C.x = 0 ∧  -- C is on Oy axis
  -- The three lines have the given equations
  ((p.line1.m = p.a ∧ p.line1.b = 4) ∨
   (p.line2.m = p.a ∧ p.line2.b = 4) ∨
   (p.line3.m = p.a ∧ p.line3.b = 4)) ∧
  ((p.line1.m = 2 ∧ p.line1.b = p.b) ∨
   (p.line2.m = 2 ∧ p.line2.b = p.b) ∨
   (p.line3.m = 2 ∧ p.line3.b = p.b)) ∧
  ((p.line1.m = p.a/2 ∧ p.line1.b = 8) ∨
   (p.line2.m = p.a/2 ∧ p.line2.b = 8) ∨
   (p.line3.m = p.a/2 ∧ p.line3.b = 8))

/-- The theorem to be proved -/
theorem sum_of_coordinates (p : Problem) (h : conditions p) :
  p.A.x + p.A.y = 13 ∨ p.A.x + p.A.y = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_l709_70948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l709_70989

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x

theorem f_properties :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ 2) ∧
  (∀ (x : ℝ), f x = 2 * Real.sin (x + Real.pi / 3)) ∧
  (∀ (m : ℝ), (∃! (x₁ x₂ x₃ : ℝ), 
    0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 2 * Real.pi ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
    (∀ (x₁ x₂ x₃ : ℝ), 
      0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 2 * Real.pi ∧ 
      f x₁ = m ∧ f x₂ = m ∧ f x₃ = m →
      x₁ + x₂ + x₃ = 7 * Real.pi / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l709_70989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seven_items_equals_41023_75_l709_70917

/-- Calculates the sum of a sequence of seven items where the first item is 4080,
    the second item is 6120, and subsequent items follow an exponential progression. -/
noncomputable def sumOfSevenItems : ℝ :=
  let firstItem := (4080 : ℝ)
  let secondItem := (6120 : ℝ)
  let ratio := (secondItem - firstItem) / firstItem
  let thirdItem := secondItem + (secondItem - firstItem) * ratio
  let fourthItem := thirdItem + (thirdItem - secondItem) * ratio
  let fifthItem := fourthItem + (fourthItem - thirdItem) * ratio
  let sixthItem := fifthItem + (fifthItem - fourthItem) * ratio
  let seventhItem := sixthItem + (sixthItem - fifthItem) * ratio
  firstItem + secondItem + thirdItem + fourthItem + fifthItem + sixthItem + seventhItem

theorem sum_of_seven_items_equals_41023_75 :
  sumOfSevenItems = 41023.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seven_items_equals_41023_75_l709_70917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_properties_l709_70900

-- Define the path of the ship
structure ShipPath where
  semicircle : Set (ℝ × ℝ)  -- Semicircular path from A to B
  straight : Set (ℝ × ℝ)    -- Straight path from B to C
  center : ℝ × ℝ            -- Location of Island X

-- Define the distance function
noncomputable def distance (path : ShipPath) (t : ℝ) : ℝ := sorry

-- State the theorem
theorem ship_distance_properties (path : ShipPath) :
  (∃ r : ℝ, ∀ p ∈ path.semicircle, distance path (p.1) = r) ∧
  (∃ t_min : ℝ, ∃ p_min ∈ path.straight, 
    (∀ p ∈ path.straight, p.1 ≤ t_min → distance path p.1 ≥ distance path t_min) ∧
    (∀ p ∈ path.straight, p.1 ≥ t_min → distance path p.1 ≥ distance path t_min)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_properties_l709_70900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_cosine_difference_l709_70936

open Real

-- Define hyperbolic sine function
noncomputable def sh (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

-- Define hyperbolic cosine function
noncomputable def ch (x : ℝ) : ℝ := (exp x + exp (-x)) / 2

-- Theorem statement
theorem hyperbolic_cosine_difference (x y : ℝ) : ch (x - y) = ch x * ch y - sh x * sh y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_cosine_difference_l709_70936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2015_eq_4030_l709_70946

open Real

/-- Recursive definition of the sequence of functions fₙ(x) -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => fun x => (x^2 + 2*x + 1) * exp x
  | n+1 => fun x => deriv (f n) x

/-- Coefficient bₙ in the general form of fₙ(x) = (aₙx² + bₙx + cₙ)eˣ -/
def b (n : ℕ) : ℝ := 2 * n

/-- Main theorem: The 2015th coefficient b₂₀₁₅ equals 4030 -/
theorem b_2015_eq_4030 : b 2015 = 4030 := by
  unfold b
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2015_eq_4030_l709_70946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_l709_70974

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- Theorem: The area of the triangle with vertices (2, 3), (7, 3), and (4, 9) is 15 square units -/
theorem triangle_area_is_15 :
  let a : Point := ⟨2, 3⟩
  let b : Point := ⟨7, 3⟩
  let c : Point := ⟨4, 9⟩
  triangleArea a b c = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_l709_70974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_identity_l709_70967

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1/Real.sqrt 2, -1/Real.sqrt 2; 1/Real.sqrt 2, 1/Real.sqrt 2]

theorem smallest_power_identity : 
  (∀ k : ℕ, k > 0 → k < 8 → rotation_matrix ^ k ≠ 1) ∧ 
  rotation_matrix ^ 8 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_identity_l709_70967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l709_70934

/-- A polynomial of degree 6 with integer coefficients -/
def P (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ :=
  a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

/-- The theorem stating that if P(x) is divisible by 7 for all integer x, 
    then all coefficients are divisible by 7 -/
theorem polynomial_divisibility 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (7 : ℤ) ∣ P a₀ a₁ a₂ a₃ a₄ a₅ a₆ x) → 
  ((7 : ℤ) ∣ a₀) ∧ ((7 : ℤ) ∣ a₁) ∧ ((7 : ℤ) ∣ a₂) ∧ 
  ((7 : ℤ) ∣ a₃) ∧ ((7 : ℤ) ∣ a₄) ∧ ((7 : ℤ) ∣ a₅) ∧ ((7 : ℤ) ∣ a₆) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l709_70934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l709_70979

/-- Curve C in polar coordinates -/
def curve_C (ρ α : ℝ) : Prop := ρ^2 - 4*ρ*(Real.cos α) + 1 = 0

/-- Line y = x, x ≥ 0 in polar coordinates -/
def line_y_eq_x (ρ α : ℝ) : Prop := α = Real.pi/4 ∧ ρ ≥ 0

/-- The sum of distances from origin to intersection points is 2√2 -/
theorem intersection_distance_sum :
  ∃ ρ₁ ρ₂ : ℝ, 
    curve_C ρ₁ (Real.pi/4) ∧ 
    curve_C ρ₂ (Real.pi/4) ∧ 
    line_y_eq_x ρ₁ (Real.pi/4) ∧ 
    line_y_eq_x ρ₂ (Real.pi/4) ∧ 
    ρ₁ + ρ₂ = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l709_70979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l709_70901

/-- The number of days it takes A to complete the work -/
def A : ℝ := sorry

/-- The number of days it takes B to complete the work -/
def B : ℝ := 6

/-- The number of days it takes C to complete the work -/
def C : ℝ := 12

/-- The number of days it takes A, B, and C together to complete the work -/
def ABC : ℝ := 2

theorem work_completion_time :
  (1 / A + 1 / B + 1 / C = 1 / ABC) → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l709_70901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_micah_envelopes_l709_70954

/-- Represents the number of stamps needed for an envelope based on its weight -/
def stamps_needed (weight : Int) : Nat :=
  if weight > 5 then 5 else 2

/-- Calculates the total number of envelopes Micah needed to buy -/
def total_envelopes (total_stamps : Nat) (light_envelopes : Nat) : Nat :=
  let heavy_stamps := total_stamps - light_envelopes * stamps_needed 0
  light_envelopes + heavy_stamps / stamps_needed 6

theorem micah_envelopes :
  total_envelopes 52 6 = 14 := by
  sorry

#eval total_envelopes 52 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_micah_envelopes_l709_70954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l709_70926

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l709_70926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_solution_uniqueness_l709_70956

open Real

/-- The differential equation y'' + 6y' + 25y = 0 -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y) x) + 6 * (deriv y x) + 25 * (y x) = 0

/-- The general solution of the differential equation -/
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  exp (-3 * x) * (C₁ * cos (4 * x) + C₂ * sin (4 * x))

/-- Theorem stating that the general_solution satisfies the differential equation -/
theorem general_solution_satisfies_equation (C₁ C₂ : ℝ) :
  differential_equation (general_solution C₁ C₂) := by
  sorry

/-- Theorem stating that any solution of the differential equation 
    can be expressed as the general_solution for some C₁ and C₂ -/
theorem solution_uniqueness (y : ℝ → ℝ) 
  (h : differential_equation y) :
  ∃ C₁ C₂ : ℝ, ∀ x, y x = general_solution C₁ C₂ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_solution_uniqueness_l709_70956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increases_l709_70905

/-- Given a triangle with side lengths a, b, and c, calculate its area using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_increases (a b c a' b' c' : ℝ) 
  (ha : a = 15) (hb : b = 9) (hc : c = 12)
  (ha' : a' = 30) (hb' : b' = 4.5) (hc' : c' = 24)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_triangle' : a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a') :
  triangleArea a' b' c' > triangleArea a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increases_l709_70905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_elevation_angle_l709_70949

/-- Given a lighthouse and two ships, proves that the angle of elevation from the second ship is 45° -/
theorem lighthouse_elevation_angle 
  (h : ℝ) 
  (d : ℝ) 
  (θ₁ : ℝ) 
  (h_pos : h > 0)
  (d_pos : d > 0)
  (h_val : h = 100)
  (d_val : d = 273.2050807568877)
  (θ₁_val : θ₁ = 30 * Real.pi / 180) :
  ∃ θ₂ : ℝ, θ₂ = 45 * Real.pi / 180 ∧ 
  Real.tan θ₂ = h / (d - h / Real.tan θ₁) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_elevation_angle_l709_70949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_21_l709_70951

-- Define the points
def A : ℚ × ℚ := (2, 2)
def B : ℚ × ℚ := (8, 2)
def C : ℚ × ℚ := (5, 9)

-- Define the function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_area_is_21 : triangleArea A B C = 21 := by
  -- Unfold the definitions
  unfold triangleArea A B C
  -- Simplify the expression
  simp
  -- The proof is completed by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_21_l709_70951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l709_70924

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt (a * x - 1) else -x^2 - 4*x

-- State the theorem
theorem solve_for_a (a : ℝ) : f a (f a (-2)) = 3 → a = 5/2 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l709_70924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l709_70935

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the vectors
def m (A B : ℝ) : ℝ × ℝ := (Real.cos A, Real.cos B)
def n (a b c : ℝ) : ℝ × ℝ := (a, 2*c - b)

-- State the theorem
theorem triangle_abc_properties 
  (h_triangle : A + B + C = π) -- Triangle angle sum
  (h_parallel : ∃ (k : ℝ), m A B = k • (n a b c)) -- Vectors are parallel
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) -- Positive angles
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0) -- Positive sides
  : A = π/3 ∧ (a = 2 * Real.sqrt 5 → ∀ S : ℝ, S = 1/2 * a * b * Real.sin A → S ≤ 5 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l709_70935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_expression_and_sum_l709_70976

noncomputable def permutation_set : Set ℝ := {10, 20, 30, 40}

noncomputable def expression (A B C D : ℝ) : ℝ := 1 / (A - 1 / (B + 1 / (C - 1 / D)))

theorem maximize_expression_and_sum :
  ∃ (A B C D : ℝ), 
    A ∈ permutation_set ∧ 
    B ∈ permutation_set ∧ 
    C ∈ permutation_set ∧ 
    D ∈ permutation_set ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (∀ (A' B' C' D' : ℝ), 
      A' ∈ permutation_set → 
      B' ∈ permutation_set → 
      C' ∈ permutation_set → 
      D' ∈ permutation_set →
      A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
      expression A B C D ≥ expression A' B' C' D') ∧
    A = 10 ∧ B = 20 ∧ C = 30 ∧ D = 40 ∧
    A + 2 * B + 3 * C + 4 * D = 290 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_expression_and_sum_l709_70976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l709_70998

-- Define the function f(x) = x^3 - ax^2 - bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

theorem cubic_function_properties 
  (a b c : ℝ) 
  (h1 : f' a b 1 = 4) 
  (h2 : f' a b (-1) = 0) 
  (h3 : f a b c (-1) = 2) :
  (∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 1 ∧ 
    f a b c x = 2 ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 1 → f a b c y ≤ 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 1 ∧ 
    f a b c x = -1 ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 1 → f a b c y ≥ -1) ∧
  (a = -1 ∧ b = 1 ∧ c = 1) := by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l709_70998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floating_cone_properties_l709_70928

/-- Represents a cone floating in water -/
structure FloatingCone where
  s : ℝ  -- specific weight of the cone
  α : ℝ  -- angle at the vertex of its axial section
  m : ℝ  -- height of the portion protruding from the water

/-- Calculates the height of the submerged portion of the cone -/
noncomputable def submergedHeight (cone : FloatingCone) : ℝ :=
  cone.m * (((1 / (1 - cone.s))^(1/3)) - 1)

/-- Calculates the weight of the entire cone -/
noncomputable def coneWeight (cone : FloatingCone) : ℝ :=
  (Real.pi * cone.m^3 * cone.s * (Real.tan (cone.α / 2))^2) / (3 * (1 - cone.s))

/-- Theorem stating the properties of a floating cone -/
theorem floating_cone_properties (cone : FloatingCone) 
  (h1 : 0 < cone.s ∧ cone.s < 1) 
  (h2 : 0 < cone.α ∧ cone.α < Real.pi) 
  (h3 : cone.m > 0) :
  ∃ (x Q : ℝ),
    x = submergedHeight cone ∧
    Q = coneWeight cone ∧
    x > 0 ∧ Q > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floating_cone_properties_l709_70928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x5_coeff_is_21_l709_70941

-- Define the polynomial (1-x+x^2)(1+x)^6
noncomputable def p (x : ℝ) : ℝ := (1 - x + x^2) * (1 + x)^6

-- Define a function to extract the coefficient of x^n in a polynomial
noncomputable def coeff (f : ℝ → ℝ) (n : ℕ) : ℝ := 
  (deriv^[n] f 0) / (Nat.factorial n)

-- Theorem statement
theorem x5_coeff_is_21 : coeff p 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x5_coeff_is_21_l709_70941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l709_70903

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- State the theorem
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l709_70903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_three_halves_l709_70972

theorem complex_expression_equals_three_halves :
  (81/16)^(-(1/4 : ℝ)) + 1/4 * (Real.log 3 / Real.log (Real.sqrt 2)) * (Real.log 4 / Real.log 3) * (((-1/3)^2)^(1/2 : ℝ)) + 7^(Real.log (1/2) / Real.log 7) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_three_halves_l709_70972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l709_70960

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Point on a hyperbola -/
def PointOnHyperbola (h : Hyperbola) := 
  {p : ℝ × ℝ | (p.1^2 / h.a^2) - (p.2^2 / h.b^2) = 1}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Left focus of a hyperbola -/
noncomputable def leftFocus (h : Hyperbola) : ℝ × ℝ := (-Real.sqrt (h.a^2 + h.b^2), 0)

/-- Right focus of a hyperbola -/
noncomputable def rightFocus (h : Hyperbola) : ℝ × ℝ := (Real.sqrt (h.a^2 + h.b^2), 0)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + (h.b / h.a)^2)

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (p : PointOnHyperbola h) 
  (h_condition : (distance p (leftFocus h) - distance p (rightFocus h))^2 = h.b^2 - 3*h.a*h.b) : 
  eccentricity h = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l709_70960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_ratio_theorem_l709_70959

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Checks if a point is on the ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Represents a line segment between two points -/
structure Segment where
  p1 : Point
  p2 : Point

/-- Calculates the length of a segment -/
noncomputable def Segment.length (s : Segment) : ℝ :=
  Real.sqrt ((s.p2.x - s.p1.x)^2 + (s.p2.y - s.p1.y)^2)

/-- Main theorem -/
theorem ellipse_tangent_ratio_theorem
  (C : Ellipse) (P A B M N Q : Point)
  (h_P_outside : ¬C.contains P)
  (h_A_on_C : C.contains A)
  (h_B_on_C : C.contains B)
  (h_M_on_C : C.contains M)
  (h_N_on_C : C.contains N)
  (h_PA_tangent : True)  -- PA is tangent to C
  (h_PB_tangent : True)  -- PB is tangent to C
  (h_Q_on_AB : True)     -- Q is on line AB
  (h_PMNQ_collinear : True)  -- P, M, N, Q are collinear
  : (Segment.length ⟨P, M⟩) / (Segment.length ⟨P, N⟩) =
    (Segment.length ⟨Q, M⟩) / (Segment.length ⟨Q, N⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_ratio_theorem_l709_70959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l709_70978

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 1)

theorem range_of_f :
  Set.range f = Set.Icc 0 1 \ {1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l709_70978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l709_70994

/-- Definition of a hyperbola C -/
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of the intersecting line -/
noncomputable def intersecting_line (b θ : ℝ) (x y : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ = 2 * b

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- Main theorem statement -/
theorem hyperbola_eccentricity_range (a b : ℝ) :
  a > b ∧ b > 0 ∧
  (∀ θ : ℝ, ∃ x y : ℝ, hyperbola a b x y ∧ intersecting_line b θ x y) →
  Real.sqrt 5 / 2 < eccentricity a b ∧ eccentricity a b < Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l709_70994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_IK_l709_70973

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure SimilarTriangles (Triangle1 Triangle2 : Type) :=
  (ratio : ℝ)
  (side_ratio : ∀ (s1 : ℝ) (s2 : ℝ), ratio = s1 / s2)

/-- Triangle FGH -/
structure TriangleFGH :=
  (FG : ℝ)
  (GH : ℝ)
  (FH : ℝ)

/-- Triangle IJK -/
structure TriangleIJK :=
  (IJ : ℝ)
  (JK : ℝ)
  (IK : ℝ)

/-- Theorem: Length of IK in similar triangles -/
theorem length_of_IK 
  (sim : SimilarTriangles TriangleFGH TriangleIJK)
  (t1 : TriangleFGH)
  (t2 : TriangleIJK)
  (h1 : t1.GH = 30)
  (h2 : t2.IJ = 15)
  (h3 : t2.JK = 18)
  (h4 : t1.FG = 27) :
  t2.IK = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_IK_l709_70973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleXYZ_perimeter_l709_70987

/-- A right prism with triangular base -/
structure RightPrism where
  height : ℝ
  baseAB : ℝ
  baseAC : ℝ

/-- The length of the hypotenuse in a right triangle -/
noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

/-- The perimeter of triangle XYZ in the right prism -/
noncomputable def triangleXYZPerimeter (p : RightPrism) : ℝ :=
  let bc := hypotenuse p.baseAB p.baseAC
  let xz := hypotenuse (p.baseAB / 2) (bc / 2)
  let yz := bc / 2
  let xy := bc / 2
  xz + yz + xy

/-- The theorem stating the perimeter of triangle XYZ -/
theorem triangleXYZ_perimeter (p : RightPrism) 
  (h1 : p.height = 20)
  (h2 : p.baseAB = 15)
  (h3 : p.baseAC = 15) :
  triangleXYZPerimeter p = 13 + 15 + 7.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleXYZ_perimeter_l709_70987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_f_l709_70947

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log x - (x - 1) / x

-- State the theorem
theorem max_min_difference_f :
  ∃ (M m : ℝ), 
    (∀ x ∈ Set.Icc 1 (exp 1), f x ≤ M) ∧ 
    (∃ x ∈ Set.Icc 1 (exp 1), f x = M) ∧
    (∀ x ∈ Set.Icc 1 (exp 1), m ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 1 (exp 1), f x = m) ∧
    (M - m = 1 / exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_f_l709_70947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangement_solution_l709_70908

/-- Represents the arrangement of numbers in the triangle --/
structure TriangleArrangement where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Checks if the given arrangement satisfies the conditions --/
def isValidArrangement (arr : TriangleArrangement) : Prop :=
  arr.A ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.B ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.C ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.D ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.A ≠ arr.B ∧ arr.A ≠ arr.C ∧ arr.A ≠ arr.D ∧
  arr.B ≠ arr.C ∧ arr.B ≠ arr.D ∧
  arr.C ≠ arr.D ∧
  arr.A + arr.C + 3 + 4 = 20 ∧
  5 + arr.D + 2 + 4 = 20 ∧
  arr.B + 5 + 2 + 3 = 20

theorem triangle_arrangement_solution :
  ∃ (arr : TriangleArrangement), isValidArrangement arr ∧
  arr.A = 6 ∧ arr.B = 8 ∧ arr.C = 7 ∧ arr.D = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangement_solution_l709_70908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_alpha_beta_l709_70944

theorem cos_sum_alpha_beta (α β : ℝ) :
  α ∈ Set.Ioo (π/2) π →
  β ∈ Set.Ioo 0 (π/2) →
  Real.cos (α - β/2) = -1/3 →
  Real.sin (α/2 - β) = Real.sqrt 6 / 3 →
  Real.cos (α + β) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_alpha_beta_l709_70944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boundary_length_of_divided_square_l709_70986

/-- The length of the boundary of a figure formed by dividing a square of area 100 into 16 equal parts and connecting adjacent points with half-circle arcs -/
theorem boundary_length_of_divided_square : 
  let square_area : ℝ := 100
  let num_divisions : ℕ := 4
  let side_length : ℝ := Real.sqrt square_area
  let segment_length : ℝ := side_length / (num_divisions : ℝ)
  let num_segments : ℕ := 4
  let num_arcs : ℕ := 4
  let arc_radius : ℝ := segment_length
  let straight_length : ℝ := segment_length * (num_segments : ℝ)
  let arc_length : ℝ := 2 * Real.pi * arc_radius * (num_arcs : ℝ) / 2
  let total_length : ℝ := straight_length + arc_length
  total_length = 10 * Real.pi + 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boundary_length_of_divided_square_l709_70986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_housewife_spending_l709_70932

noncomputable def initial_amount : ℝ := 150
noncomputable def spent_fraction : ℝ := 2/3

theorem housewife_spending (amount_left : ℝ) :
  amount_left = initial_amount - (spent_fraction * initial_amount) →
  amount_left = 50 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_housewife_spending_l709_70932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_is_50_l709_70945

/-- The selling price of a car that satisfies the profit conditions -/
noncomputable def car_price : ℝ :=
  let car_material_cost : ℝ := 100
  let car_production : ℕ := 4
  let motorcycle_material_cost : ℝ := 250
  let motorcycle_production : ℕ := 8
  let motorcycle_price : ℝ := 50
  let profit_difference : ℝ := 50

  -- Define the function to calculate car profit
  let car_profit (price : ℝ) := car_production * price - car_material_cost

  -- Define the motorcycle profit
  let motorcycle_profit := motorcycle_production * motorcycle_price - motorcycle_material_cost

  -- The price that satisfies the profit condition
  (motorcycle_profit - profit_difference + car_material_cost) / car_production

/-- Theorem stating that the calculated car price is 50 -/
theorem car_price_is_50 : car_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_is_50_l709_70945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parallel_line_through_point_l709_70991

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define when a point is not on a plane
def NotOnPlane (p : Point) (P : Plane) : Prop :=
  ¬ P p.1 p.2.1 p.2.2

-- Define when a line is parallel to a plane
def LineParallelToPlane (l : Point → Point → Prop) (P : Plane) : Prop :=
  ∀ p q : Point, l p q → (P p.1 p.2.1 p.2.2 ↔ P q.1 q.2.1 q.2.2)

-- Theorem statement
theorem exists_parallel_line_through_point
  (P : Plane) (A : Point) (h : NotOnPlane A P) :
  ∃ l : Point → Point → Prop, l A A ∧ LineParallelToPlane l P :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parallel_line_through_point_l709_70991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_earning_difference_l709_70971

noncomputable section

def manager_hourly_wage : ℝ := 8.50
def manager_hours : ℝ := 8
def dishwasher_hours : ℝ := 6
def chef_hours : ℝ := 10
def daily_bonus : ℝ := 5

def dishwasher_hourly_wage : ℝ := manager_hourly_wage / 2
def chef_hourly_wage : ℝ := dishwasher_hourly_wage * 1.20

def manager_daily_earning : ℝ := manager_hourly_wage * manager_hours + daily_bonus
def dishwasher_daily_earning : ℝ := dishwasher_hourly_wage * dishwasher_hours + daily_bonus
def chef_daily_earning : ℝ := chef_hourly_wage * chef_hours + daily_bonus

theorem daily_earning_difference :
  manager_daily_earning - chef_daily_earning = 17 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_earning_difference_l709_70971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_score_set_exists_hockey_game_no_valid_score_set_l709_70958

/-- Represents a player's claims about goal scores -/
structure PlayerClaims where
  selfScore : Nat
  otherScore : Nat
  otherPlayer : Nat

/-- Represents the game state with each player's claims -/
structure GameState where
  anton : PlayerClaims
  ilya : PlayerClaims
  seryozha : PlayerClaims

/-- Checks if a given set of scores satisfies the conditions -/
def isValidScoreSet (antonScore ilyaScore seryozhaScore : Nat) (state : GameState) : Prop :=
  let totalTruths := 
    (if antonScore = state.anton.selfScore then 1 else 0) + 
    (if ilyaScore = state.ilya.selfScore then 1 else 0) + 
    (if seryozhaScore = state.seryozha.selfScore then 1 else 0) +
    (if ilyaScore = state.anton.otherScore then 1 else 0) +
    (if seryozhaScore = state.ilya.otherScore then 1 else 0) +
    (if antonScore = state.seryozha.otherScore then 1 else 0)
  totalTruths = 3 ∧ antonScore + ilyaScore + seryozhaScore = 10

/-- The main theorem stating that no valid score set exists -/
theorem no_valid_score_set_exists (state : GameState) : 
  ¬∃ (a i s : Nat), isValidScoreSet a i s state := by
  sorry

/-- The game state based on the problem description -/
def hockeyGameState : GameState := {
  anton := { selfScore := 3, otherScore := 1, otherPlayer := 2 }
  ilya := { selfScore := 4, otherScore := 5, otherPlayer := 1 }
  seryozha := { selfScore := 6, otherScore := 2, otherPlayer := 1 }
}

/-- The final theorem proving that the specific game state has no valid score set -/
theorem hockey_game_no_valid_score_set : 
  ¬∃ (a i s : Nat), isValidScoreSet a i s hockeyGameState := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_score_set_exists_hockey_game_no_valid_score_set_l709_70958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_to_line_point_one_one_on_parabola_closest_point_is_one_one_l709_70907

/-- The parabola defined by x^2 = y -/
def parabola (x y : ℝ) : Prop := x^2 = y

/-- The line defined by 2x - y - 4 = 0 -/
def line (x y : ℝ) : Prop := 2*x - y - 4 = 0

/-- The distance from a point (x, y) to the line 2x - y - 4 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |2*x - y - 4| / Real.sqrt 5

/-- The theorem stating that (1, 1) is the closest point on the parabola to the line -/
theorem closest_point_on_parabola_to_line :
  ∀ x y : ℝ, parabola x y → 
  distance_to_line x y ≥ distance_to_line 1 1 := by
  sorry

/-- The point (1, 1) satisfies the parabola equation -/
theorem point_one_one_on_parabola :
  parabola 1 1 := by
  simp [parabola]

/-- The point (1, 1) is on the parabola and is closest to the line -/
theorem closest_point_is_one_one :
  parabola 1 1 ∧ 
  (∀ x y : ℝ, parabola x y → 
  distance_to_line x y ≥ distance_to_line 1 1) := by
  constructor
  · exact point_one_one_on_parabola
  · exact closest_point_on_parabola_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_to_line_point_one_one_on_parabola_closest_point_is_one_one_l709_70907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l709_70940

-- Define the functions h and k
noncomputable def h : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

-- Define the ranges of h and k
axiom h_range : ∀ x, -3 ≤ h x ∧ h x ≤ 4
axiom k_range : ∀ x, 0 ≤ k x ∧ k x ≤ 2

-- Theorem statement
theorem max_product_value :
  ∃ c, c = 8 ∧ ∀ x, h x * k x ≤ c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l709_70940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_sum_property_l709_70914

def T (n : ℕ) : Set ℕ := {x | 5 ≤ x ∧ x ≤ n}

def hasSum (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b = c

theorem smallest_n_with_sum_property :
  ∀ n ≥ 5, (∀ A B : Set ℕ, A ∪ B = T n → A ∩ B = ∅ → hasSum A ∨ hasSum B) ↔ n ≥ 625 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_sum_property_l709_70914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l709_70930

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = 180 ∧ -- Sum of angles in a triangle
  a / Real.sin (A * Real.pi / 180) = b / Real.sin (B * Real.pi / 180) ∧ -- Sine law
  a / Real.sin (A * Real.pi / 180) = c / Real.sin (C * Real.pi / 180) ∧ -- Sine law
  b / Real.sin (B * Real.pi / 180) = c / Real.sin (C * Real.pi / 180) -- Sine law

-- Define the area of the triangle
noncomputable def area (b c A : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin (A * Real.pi / 180)

-- Theorem statement
theorem triangle_theorem :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_ABC A B C a b c →
  c = Real.sqrt 3 →
  b = 1 →
  B = 30 →
  ((C = 60 ∧ area b c A = Real.sqrt 3 / 2) ∨
   (C = 120 ∧ area b c A = Real.sqrt 3 / 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l709_70930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_error_l709_70988

/-- 
If the measurement of each side of a cube has a 2% excess error, 
then the percentage error in the calculated volume of the cube 
is approximately 6.12%.
-/
theorem cube_volume_error (a : ℝ) (a_pos : a > 0) : 
  abs ((((a * (1 + 0.02))^3 - a^3) / a^3) * 100 - 6.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_error_l709_70988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l709_70996

/-- Represents the type of an inhabitant -/
inductive Inhabitant : Type
| Vegetarian : Inhabitant
| Cannibal : Inhabitant

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if an arrangement is valid according to the problem conditions -/
def isValidArrangement (arr : List Inhabitant) : Prop :=
  arr.length > 0 ∧
  (∃ i, arr.get? i = some Inhabitant.Vegetarian) ∧
  (∀ i j, i < arr.length → j < arr.length → i ≠ j →
    arr.get? i = some Inhabitant.Vegetarian → arr.get? j = some Inhabitant.Vegetarian →
    isPrime (Int.natAbs (i - j))) ∧
  (∀ i, i < arr.length →
    arr.get? i = some Inhabitant.Cannibal →
    ∃ j, j < arr.length ∧ arr.get? j = some Inhabitant.Vegetarian ∧ ¬isPrime (Int.natAbs (i - j)))

/-- The main theorem stating that a valid arrangement exists for any number of inhabitants -/
theorem valid_arrangement_exists (n : ℕ) :
  ∃ arr : List Inhabitant, arr.length = n ∧ isValidArrangement arr := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l709_70996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l709_70992

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h_even : ∀ x, f a x = f a (-x)) 
  (h_slope : ∃ x, (deriv (f a)) x = 3/2) :
  ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l709_70992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_bc_length_l709_70915

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The reciprocal of the golden ratio -/
noncomputable def φ_inv : ℝ := (Real.sqrt 5 - 1) / 2

/-- Given a line segment AB of length 4, C is the golden section point of AB where AC < BC -/
structure GoldenSectionSegment where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  h1 : AB = 4
  h2 : AC < BC
  h3 : AB / BC = BC / AC

theorem golden_section_bc_length (g : GoldenSectionSegment) : g.BC = 2 * Real.sqrt 5 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_bc_length_l709_70915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_special_divisor_property_l709_70957

def isDivisor (d n : ℕ) : Bool := n % d = 0

def divisors (n : ℕ) : List ℕ := 
  (List.range (n + 1)).filter (λ d => isDivisor d n)

theorem largest_number_with_special_divisor_property : 
  ∃ (N : ℕ), 
    (∀ n : ℕ, n > N → 
      ¬((divisors n).length ≥ 3 ∧ 
       (divisors n)[2]! * 21 = (divisors n)[(divisors n).length - 3]!)) ∧
    (divisors N).length ≥ 3 ∧ 
    (divisors N)[2]! * 21 = (divisors N)[(divisors N).length - 3]! ∧
    N = 441 := by
  sorry

#eval divisors 441

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_special_divisor_property_l709_70957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_l709_70969

/-- Milk production calculation -/
theorem milk_production
  (a b c d e f : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  d * b * e / (a * c) + d * b * f / (2 * a * c) =
  d * b * e / (a * c) + d * b * f / (2 * a * c) :=
by
  -- The proof is trivial as we're asserting equality to itself
  rfl

#check milk_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_l709_70969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_markup_percentage_l709_70953

theorem merchant_markup_percentage (x : ℝ) : 
  (∀ (cost_price : ℝ), cost_price > 0 →
    let marked_price := cost_price * (1 + x / 100);
    let selling_price := marked_price * (1 - 15 / 100);
    selling_price = cost_price * (1 + 19 / 100)) →
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_markup_percentage_l709_70953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_5_l709_70950

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 1)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

-- State the theorem
theorem t_of_f_5 : t (f 5) = Real.sqrt (29 - 4 * Real.sqrt 21) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_5_l709_70950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_floors_l709_70995

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem sum_of_floors : 
  floor (Real.sin 1) + floor (Real.cos 2) + floor (Real.tan 3) + 
  floor (Real.sin 4) + floor (Real.cos 5) + floor (Real.tan 6) = -4 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_floors_l709_70995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l709_70937

-- Define x as a real variable
variable (x : ℝ)

theorem quadratic_factorization (b : ℤ) : 
  (b = 34 ∧ ∃ (m n p q : ℤ), (15 : ℝ) * x^2 + (b : ℝ) * x + 15 = (m : ℝ) * x + n * ((p : ℝ) * x + q)) ∨
  (b ∈ ({30, 36, 40} : Set ℤ) → ¬∃ (m n p q : ℤ), (15 : ℝ) * x^2 + (b : ℝ) * x + 15 = (m : ℝ) * x + n * ((p : ℝ) * x + q)) := by
  sorry

#check quadratic_factorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l709_70937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_reciprocal_distances_l709_70993

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through F
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - F.1)

-- Define perpendicular lines
def Perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Define the distance between two points
noncomputable def Distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Main theorem
theorem constant_sum_reciprocal_distances 
  (k1 k2 : ℝ) 
  (xa ya xb yb xc yc xd yd : ℝ) :
  Perpendicular k1 k2 →
  Line k1 xa ya → Line k1 xb yb → 
  Line k2 xc yc → Line k2 xd yd → 
  Ellipse xa ya → Ellipse xb yb → 
  Ellipse xc yc → Ellipse xd yd → 
  1 / Distance xa ya xb yb + 1 / Distance xc yc xd yd = 17 / 48 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_reciprocal_distances_l709_70993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l709_70968

/-- Represents an arithmetic sequence -/
structure ArithSeq where
  a : ℕ → ℝ
  d : ℝ
  is_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithSeq) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

theorem max_positive_sum (seq : ArithSeq) 
  (h16 : seq.a 16 > 0)
  (h17 : seq.a 17 < 0)
  (h_abs : seq.a 16 > |seq.a 17|) :
  (∀ n > 32, S seq n ≤ 0) ∧ S seq 32 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l709_70968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_count_l709_70980

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then -x^2 + 4*x else -x^2 - 4*x

theorem f_solutions_count : 
  ∃ (S : Finset ℝ), (∀ a ∈ S, f (f a) = 3) ∧ (∀ a : ℝ, f (f a) = 3 → a ∈ S) ∧ S.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_count_l709_70980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l709_70923

def word_length : ℕ := 7
def repeated_letters : ℕ := 2
def single_letters : ℕ := 3

theorem balloon_arrangements :
  (Nat.factorial word_length / (Nat.factorial 2 * Nat.factorial 2)) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l709_70923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l709_70943

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x - (a - 1) / x

-- State the theorem
theorem function_properties (a : ℝ) 
  (h : ∀ x > 0, f a x ≤ -1) :
  (a ≥ 1) ∧ 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f a (1 - Real.sin θ) ≤ f a (1 + Real.sin θ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l709_70943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l709_70916

/-- A line in the 2D plane represented by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def Line.evaluate (l : Line) (x : ℝ) : ℝ := l.m * x + l.b

noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.b - l1.b) / (l1.m - l2.m),
    y := l1.evaluate ((l2.b - l1.b) / (l1.m - l2.m)) }

noncomputable def xAxisIntersection (l : Line) : Point :=
  { x := -l.b / l.m, y := 0 }

noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem triangle_abc_area :
  let line1 : Line := { m := 1, b := 1 }
  let line2 : Line := { m := -1, b := 3 }
  let a : Point := intersectionPoint line1 line2
  let b : Point := xAxisIntersection line1
  let c : Point := xAxisIntersection line2
  triangleArea a b c = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l709_70916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_equals_2_49_l709_70952

def b : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 1) => (64 * (b n)^4)^(1/4)

theorem b_50_equals_2_49 : b 50 = 2^49 := by
  -- Proof sketch:
  -- 1. Prove by induction that b n = 2^(n-1) for n ≥ 1
  -- 2. Apply this result to n = 50
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_equals_2_49_l709_70952
