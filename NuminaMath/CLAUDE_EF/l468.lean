import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l468_46817

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem min_sum_of_product_factorial (p q r s : ℕ+) : 
  p * q * r * s = factorial 12 → 
  (∀ a b c d : ℕ+, a * b * c * d = factorial 12 → p + q + r + s ≤ a + b + c + d) → 
  p + q + r + s = 8800 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l468_46817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_form_nonnegative_l468_46815

theorem quadratic_form_nonnegative (m n : ℝ) (h : m ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3)) :
  (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 + 2*x*y + 2*m*z*x + 2*n*y*z ≥ 0) ↔
  (m - Real.sqrt (3 - m^2) ≤ n ∧ n ≤ m + Real.sqrt (3 - m^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_form_nonnegative_l468_46815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_disjoint_circles_are_disjoint_impl_l468_46852

-- Define the circles
def circle1 (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 = 4
def circle2 (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*m*y = 8 - m^2

-- Define the centers and radii
def center1 (m : ℝ) : ℝ × ℝ := (m, 0)
def center2 (m : ℝ) : ℝ × ℝ := (-1, m)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers (m : ℝ) : ℝ := 
  Real.sqrt ((m + 1)^2 + m^2)

-- Theorem statement
theorem circles_are_disjoint (m : ℝ) (h : m > 3) : 
  distance_between_centers m > radius1 + radius2 := by
  sorry

-- Additional theorem to show that the circles are disjoint
theorem circles_are_disjoint_impl (m : ℝ) (h : m > 3) :
  ∀ x y : ℝ, ¬(circle1 m x y ∧ circle2 m x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_disjoint_circles_are_disjoint_impl_l468_46852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_product_bound_l468_46847

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines the property of a point being on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Defines the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The product of distances from any point on the ellipse to its foci is at most 25 -/
theorem ellipse_foci_distance_product_bound (e : Ellipse) (p f1 f2 : Point) :
  e.a = 5 → e.b = 4 →
  isOnEllipse p e →
  (∀ q, isOnEllipse q e → distance q f1 + distance q f2 = 2 * e.a) →
  distance p f1 * distance p f2 ≤ 25 ∧ ∃ p', isOnEllipse p' e ∧ distance p' f1 * distance p' f2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_product_bound_l468_46847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l468_46808

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 3)

-- State the theorem
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3} := by
  sorry

-- Additional lemma to show the domain explicitly
lemma f_defined (x : ℝ) : x ≠ -3 → ∃ y, f x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l468_46808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_on_circle_max_min_on_region_l468_46830

open Set

theorem max_min_on_circle (a : ℝ) (h : a > 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 - y^2 + 2*a^2
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ a^2}
  (∃ p ∈ S, ∀ q ∈ S, f q ≤ f p) ∧
  (∃ p ∈ S, ∀ q ∈ S, f p ≤ f q) ∧
  (upperBounds {f p | p ∈ S} = {3*a^2}) ∧
  (lowerBounds {f p | p ∈ S} = {a^2}) :=
by
  sorry

theorem max_min_on_region :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ 2*x^3 + 4*x^2 + y^2 - 2*x*y
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≤ 4 ∧ p.2 ≥ p.1^2}
  (∃ p ∈ S, ∀ q ∈ S, f q ≤ f p) ∧
  (∃ p ∈ S, ∀ q ∈ S, f p ≤ f q) ∧
  (upperBounds {f p | p ∈ S} = {32}) ∧
  (lowerBounds {f p | p ∈ S} = {0}) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_on_circle_max_min_on_region_l468_46830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_3_radians_in_second_quadrant_l468_46894

/-- Represents the four quadrants of the unit circle -/
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

/-- Determines the quadrant of an angle given its radian measure -/
noncomputable def determine_quadrant (angle : ℝ) : Quadrant :=
  if 0 ≤ angle ∧ angle < Real.pi / 2 then Quadrant.First
  else if Real.pi / 2 ≤ angle ∧ angle < Real.pi then Quadrant.Second
  else if Real.pi ≤ angle ∧ angle < 3 * Real.pi / 2 then Quadrant.Third
  else Quadrant.Fourth

theorem angle_3_radians_in_second_quadrant :
  determine_quadrant 3 = Quadrant.Second := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_3_radians_in_second_quadrant_l468_46894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_highest_intensity_l468_46834

/-- Represents a regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  area : ℝ
  current : ℝ

/-- Calculates the magnetic field intensity at the center of a regular polygon -/
noncomputable def magneticFieldIntensity (p : RegularPolygon) : ℝ :=
  2 * p.current / Real.sqrt p.area * Real.sqrt ((p.n ^ 3 * (Real.sin (Real.pi / p.n)) ^ 3) / (Real.pi ^ 3 * Real.cos (Real.pi / p.n)))

/-- Theorem: The magnetic field intensity is highest for an equilateral triangle -/
theorem triangle_highest_intensity
  (triangle square pentagon hexagon : RegularPolygon)
  (circle_intensity : ℝ)
  (h_triangle : triangle.n = 3)
  (h_square : square.n = 4)
  (h_pentagon : pentagon.n = 5)
  (h_hexagon : hexagon.n = 6)
  (h_equal_area : triangle.area = square.area ∧ triangle.area = pentagon.area ∧ 
                  triangle.area = hexagon.area)
  (h_equal_current : triangle.current = square.current ∧ triangle.current = pentagon.current ∧ 
                     triangle.current = hexagon.current)
  (h_circle : circle_intensity = Real.pi * triangle.current / Real.sqrt triangle.area) :
  magneticFieldIntensity triangle > magneticFieldIntensity square ∧
  magneticFieldIntensity triangle > magneticFieldIntensity pentagon ∧
  magneticFieldIntensity triangle > magneticFieldIntensity hexagon ∧
  magneticFieldIntensity triangle > circle_intensity :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_highest_intensity_l468_46834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_freshmen_than_sophomores_l468_46832

theorem more_freshmen_than_sophomores 
  (total : ℕ) 
  (juniors_percent : ℚ)
  (not_sophomores_percent : ℚ)
  (not_freshmen_percent : ℚ)
  (seniors : ℕ)
  (advanced : ℕ)
  (h_total : total = 1200)
  (h_juniors : juniors_percent = 22 / 100)
  (h_not_sophomores : not_sophomores_percent = 55 / 100)
  (h_not_freshmen : not_freshmen_percent = 25 / 100)
  (h_seniors : seniors = 240)
  (h_advanced : advanced = 20) :
  (total - (↑total * not_freshmen_percent).floor) - 
  (total - (↑total * not_sophomores_percent).floor) - 
  (seniors + advanced) = 360 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_freshmen_than_sophomores_l468_46832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l468_46849

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def mag (v : ℝ × ℝ) : ℝ := Real.sqrt (dot v v)

-- Define tangent function
noncomputable def tan (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

-- Define angle function (placeholder)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_vector_relation (t : Triangle) :
  3 * (dot (vec t.C t.A + vec t.C t.B) (vec t.A t.B)) = 4 * (mag (vec t.A t.B))^2 →
  (tan (angle t.C t.A t.B)) / (tan (angle t.C t.B t.A)) = -7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l468_46849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l468_46819

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- The theorem stating the range of m for which f has only one solution in [0, 2] -/
theorem unique_solution_range (m : ℝ) : 
  (∃! x, x ∈ Set.Icc 0 2 ∧ f m x = 0) ↔ m ∈ Set.union (Set.Ioo (-2) 0) (Set.singleton 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l468_46819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_friends_count_num_possible_pairs_l468_46829

/-- Represents the number of people in the circular seating arrangement -/
def total_people : ℕ := 8

/-- Represents the number of friends Cara has -/
def num_friends : ℕ := 7

/-- Theorem stating the number of remaining friends after Alex is chosen -/
theorem remaining_friends_count :
  num_friends - 1 = 6 := by
  rfl

/-- Main theorem proving the number of different possible pairs Cara could sit between -/
theorem num_possible_pairs :
  (num_friends - 1) = 6 := by
  exact remaining_friends_count


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_friends_count_num_possible_pairs_l468_46829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l468_46814

/-- In a triangle ABC, if sin C = (3√3)/14 and a = (7/3)c, then sin A = √3/2 and a = 7 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧ 
  (0 < B) ∧ (B < π) ∧ 
  (0 < C) ∧ (C < π) ∧ 
  (A + B + C = π) ∧ 
  (Real.sin C = (3 * Real.sqrt 3) / 14) ∧ 
  (a = (7/3) * c) →
  (Real.sin A = Real.sqrt 3 / 2) ∧ (a = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l468_46814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_problem_l468_46854

/-- The distance traveled by two snails to reach a flower -/
noncomputable def snail_distance (speed_A speed_B : ℝ) (time_difference : ℝ) : ℝ :=
  let time_A := (speed_B * time_difference) / (speed_B - speed_A)
  speed_A * time_A

theorem snail_problem :
  let speed_A : ℝ := 10  -- meters per hour
  let speed_B : ℝ := 15  -- meters per hour
  let time_difference : ℝ := 0.5  -- 30 minutes in hours
  snail_distance speed_A speed_B time_difference = 15 := by
  sorry

#check snail_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_problem_l468_46854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_from_digit_set_l468_46839

def digit_set : Finset Nat := {3, 4, 5, 7}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def from_digit_set (n : Nat) : Prop :=
  ∃ (a b : Nat), a ∈ digit_set ∧ b ∈ digit_set ∧ a ≠ b ∧ n = 10 * a + b

theorem two_digit_primes_from_digit_set :
  ∃! (s : Finset Nat), 
    (∀ n ∈ s, is_two_digit n ∧ from_digit_set n ∧ Nat.Prime n) ∧ 
    s.card = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_from_digit_set_l468_46839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l468_46856

noncomputable section

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (1 - i) / i

-- Theorem statement
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l468_46856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l468_46822

/-- Circle C: (x)^2 + (y)^2 - 6x - 8y + 21 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 21 = 0

/-- Line l: kx - y - 4k + 3 = 0 -/
def line_l (k x y : ℝ) : Prop := k*x - y - 4*k + 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, 4)

/-- The statement that the line intersects the circle at two distinct points -/
def line_intersects_circle (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

/-- The value of k that minimizes the chord length -/
def k_min : ℝ := 1

/-- The length of the shortest chord -/
noncomputable def shortest_chord_length : ℝ := 2 * Real.sqrt 2

theorem circle_and_line_properties :
  (circle_center = (3, 4)) ∧
  (∀ k : ℝ, line_intersects_circle k) ∧
  (k_min = 1) ∧
  (shortest_chord_length = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l468_46822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_definition_l468_46868

theorem h_definition (x : ℝ) : 
  (λ x : ℝ => -12 * x^4 - 8 * x^3 + 4 * x^2 - 7 * x + 2) x +
  (12 * x^4 + 5 * x^3) = -3 * x^3 + 4 * x^2 - 7 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_definition_l468_46868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_acceleration_l468_46812

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 10

/-- The radius of the larger roller in meters -/
def R : ℝ := 1.25

/-- The radius of the smaller roller in meters -/
def r : ℝ := 0.75

/-- The mass of the plate in kg -/
def m : ℝ := 100

/-- The angle of inclination of the plate -/
noncomputable def α : ℝ := Real.arccos 0.92

/-- The magnitude of the plate's acceleration -/
def plate_acceleration_magnitude : ℝ := 2

/-- The direction of the plate's acceleration -/
noncomputable def plate_acceleration_direction : ℝ := Real.arcsin 0.2

theorem plate_acceleration (no_slip : Bool) :
  (no_slip = true) →
  (plate_acceleration_magnitude = g * Real.sin (α / 2)) ∧
  (plate_acceleration_direction = Real.arcsin (Real.sin (α / 2))) := by
  sorry

#eval g
#eval R
#eval r
#eval m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_acceleration_l468_46812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_monday_leap_day_l468_46858

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving BEq, Repr

/-- Checks if a year is a leap year -/
def isLeapYear (y : Nat) : Bool :=
  y % 4 == 0 && (y % 100 ≠ 0 || y % 400 == 0)

/-- Calculates the day of the week for February 29 in a given year, 
    assuming 2012 was a Wednesday -/
def februaryLeapDay (y : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next leap year after 2012 when February 29 falls on a Monday is 2016 -/
theorem next_monday_leap_day : 
  (isLeapYear 2012 = true) → 
  (februaryLeapDay 2012 = DayOfWeek.Wednesday) →
  (∀ y : Nat, 2012 < y → y < 2016 → 
    (isLeapYear y = true → februaryLeapDay y ≠ DayOfWeek.Monday)) →
  (isLeapYear 2016 = true) ∧ (februaryLeapDay 2016 = DayOfWeek.Monday) :=
by sorry

#eval isLeapYear 2012
#eval isLeapYear 2016

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_monday_leap_day_l468_46858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_radius_l468_46810

/-- The volume of the cylindrical wrapper in cubic centimeters -/
noncomputable def volume : ℝ := 27 * Real.pi

/-- The surface area of the cylindrical wrapper as a function of its base radius -/
noncomputable def surface_area (r : ℝ) : ℝ := Real.pi * r^2 + 2 * Real.pi * r * (volume / (Real.pi * r^2))

/-- The theorem stating that the base radius minimizing the surface area is 3 cm -/
theorem min_surface_area_radius : 
  ∃ (r : ℝ), r > 0 ∧ r = 3 ∧ ∀ (r' : ℝ), r' > 0 → surface_area r ≤ surface_area r' := by
  sorry

#check min_surface_area_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_radius_l468_46810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_weight_l468_46898

/-- Given a class of boys with an incorrect average weight calculation due to a misread weight,
    this theorem proves the correct average weight. -/
theorem correct_average_weight
  (num_boys : ℕ)
  (initial_avg : ℚ)
  (misread_weight : ℚ)
  (correct_weight : ℚ)
  (h_num_boys : num_boys = 20)
  (h_initial_avg : initial_avg = 58.4)
  (h_misread : misread_weight = 56)
  (h_correct : correct_weight = 65) :
  let total_weight := initial_avg * num_boys
  let weight_diff := correct_weight - misread_weight
  let corrected_total := total_weight + weight_diff
  corrected_total / num_boys = 58.85 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_weight_l468_46898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_theorem_l468_46853

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points A', B', C' on the sides of the triangle
variable (A' B' C' : ℝ × ℝ)

-- Define the point O where AA', BB', CC' are concurrent
variable (O : ℝ × ℝ)

-- Define the ratios
noncomputable def ratio_A (A O A' : ℝ × ℝ) : ℝ := dist A O / dist O A'
noncomputable def ratio_B (B O B' : ℝ × ℝ) : ℝ := dist B O / dist O B'
noncomputable def ratio_C (C O C' : ℝ × ℝ) : ℝ := dist C O / dist O C'

-- State the theorem
theorem triangle_concurrent_lines_theorem 
  (h1 : ratio_A A O A' = ratio_C C O C')
  (h2 : ratio_A A O A' + ratio_B B O B' + ratio_C C O C' = 50) :
  ratio_A A O A' * ratio_B B O B' * ratio_C C O C' = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_theorem_l468_46853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_and_triangle_area_l468_46875

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi / 6)

theorem function_range_and_triangle_area :
  ∃ (A B C a b c : ℝ),
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-1/2) (1/4)) ∧
  f A = 1/4 ∧
  a = Real.sqrt 3 ∧
  Real.sin B = 2 * Real.sin C ∧
  a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) ∧
  (∀ y ∈ Set.range f, y ∈ Set.Icc (-1/2) (1/4)) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_and_triangle_area_l468_46875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_real_numbers_inequality_l468_46867

theorem seven_real_numbers_inequality (S : Finset ℝ) (h : S.card = 7) :
  ∃ x y, x ∈ S ∧ y ∈ S ∧ 0 ≤ (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 / Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_real_numbers_inequality_l468_46867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l468_46823

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 16

-- Define the line
def line_eq (x y : ℝ) : Prop := 3*x - 4*y - 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x - 4*y - 2| / Real.sqrt (3^2 + (-4)^2)

-- Theorem statement
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_eq x y ∧ 
  (∀ (x' y' : ℝ), circle_eq x' y' → distance_to_line x y ≥ distance_to_line x' y') ∧
  distance_to_line x y = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l468_46823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_MN_l468_46824

/-- The value of a that minimizes the length of line segment MN --/
noncomputable def min_a : ℝ := Real.sqrt 2 / 2

/-- The x-coordinate of point M --/
def M_x (a : ℝ) : ℝ := a^2

/-- The x-coordinate of point N --/
noncomputable def N_x (a : ℝ) : ℝ := Real.log a

/-- The length of line segment MN --/
noncomputable def MN_length (a : ℝ) : ℝ := |M_x a - N_x a|

theorem min_length_MN :
  ∀ a > 0, MN_length a ≥ MN_length min_a :=
by
  sorry

#check min_length_MN

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_MN_l468_46824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l468_46816

def A : Set ℝ := {-1, 1}

def B (a : ℝ) : Set ℝ := {x | x * a = 1}

theorem possible_values_of_a (a : ℝ) : B a ⊆ A ↔ a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l468_46816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_G_is_S_l468_46826

noncomputable def G : ℂ := -0.6 + 0.8 * Complex.I
noncomputable def S : ℂ := -0.8 - 0.6 * Complex.I

theorem reciprocal_of_G_is_S :
  Complex.abs G = 1 →
  1 / G = S := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_G_is_S_l468_46826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l468_46889

/-- The function f(x) = a^x * (1 - 2^x) is odd when a = √2/2 --/
theorem odd_function_condition (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  (∀ x : ℝ, (fun x ↦ a^x * (1 - 2^x)) (-x) = -(a^x * (1 - 2^x))) →
  a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l468_46889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l468_46863

theorem cosine_equation_solutions : 
  ∃ (S : Finset ℝ), S.card = 7 ∧ 
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ π ∧ Real.cos (7 * x) = Real.cos (5 * x)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π ∧ Real.cos (7 * x) = Real.cos (5 * x) → x ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l468_46863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_self_intersections_count_l468_46836

noncomputable def x (t : ℝ) : ℝ := Real.cos t + (3 * t) / 2

noncomputable def y (t : ℝ) : ℝ := Real.sin t

noncomputable def period : ℝ := 2 * Real.pi

def x_lower : ℝ := 10

def x_upper : ℝ := 80

def num_intersections : ℕ := 8

theorem self_intersections_count :
  ∃ t_max : ℝ,
    x t_max = x_upper ∧
    (∀ t : ℝ, x_lower ≤ x t ∧ x t ≤ x_upper → t ≤ t_max) ∧
    num_intersections = ⌊t_max / period⌋ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_self_intersections_count_l468_46836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_triple_angle_identity_l468_46803

theorem sin_triple_angle_identity (α : ℝ) : 
  Real.sin (π + α) * Real.sin ((4 / 3) * π + α) * Real.sin ((2 / 3) * π + α) = (1 / 4) * Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_triple_angle_identity_l468_46803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l468_46840

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_quadrants (k : ℝ) :
  (inverse_proportion k (-2) = 1) →
  (∀ x y, y = inverse_proportion k x → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l468_46840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l468_46874

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ) (a b : ℝ),
    (∀ x, f (x + T) = f x) ∧  -- f has period T
    (∀ y, ∃ x, y = f x → y = Real.sin (2 * (x + a)) - b) ∧  -- f can be obtained by translating sin 2x
    (0 < a ∧ a < Real.pi / 2) ∧  -- constraints on a
    (a * b = (Real.sqrt 3 / 12) * Real.pi) ∧  -- value of ab
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      -Real.sqrt 3 ≤ f x ∧ f x ≤ 1 - Real.sqrt 3 / 2) ∧  -- range on [0, π/2]
    (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧
      f x₁ = -Real.sqrt 3 ∧ f x₂ = 1 - Real.sqrt 3 / 2) ∧  -- range endpoints are attained
    T = Real.pi  -- smallest positive period
    := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l468_46874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_b_l468_46821

/-- The function f(x) = x^3 + 3ax^2 + bx + a^2 -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_implies_a_b (a b : ℝ) :
  (f a b (-1) = 0) ∧ 
  (f_deriv a b (-1) = 0) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε) \ {-1}, f a b x ≥ 0) →
  a = 2 ∧ b = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_b_l468_46821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_time_approximation_l468_46862

-- Define the constants
noncomputable def walk_distance : ℝ := 5
noncomputable def run_distance : ℝ := 15
noncomputable def walk_speed : ℝ := 2.5
noncomputable def run_speed : ℝ := 4.5
noncomputable def days_per_week : ℝ := 7

-- Define the exercise time calculation function
noncomputable def exercise_time_per_week : ℝ :=
  ((walk_distance / walk_speed + run_distance / run_speed) * days_per_week)

-- State the theorem
theorem exercise_time_approximation :
  ∃ ε > 0, |exercise_time_per_week - 37.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_time_approximation_l468_46862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_value_l468_46850

-- Define the function f as noncomputable due to its dependency on Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * (Real.log x / Real.log 3)

-- State the theorem
theorem f_half_value (a : ℝ) :
  f a 2 = 6 → f a (1/2) = 17/8 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_value_l468_46850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_inscribed_rectangle_l468_46891

/-- Represents a circular sector with radius r and central angle 2α -/
structure CircularSector where
  r : ℝ
  α : ℝ

/-- Represents a rectangle inscribed in a circular sector -/
structure InscribedRectangle (sector : CircularSector) where
  ω : ℝ  -- Angle between radius to vertex on arc and perpendicular bisector of chord

/-- The area of the inscribed rectangle as a function of ω -/
noncomputable def rectangleArea (sector : CircularSector) (rect : InscribedRectangle sector) : ℝ :=
  (sector.r^2 / Real.sin sector.α) * (Real.cos (2 * rect.ω - sector.α) - Real.cos sector.α)

theorem max_area_inscribed_rectangle (sector : CircularSector) :
  ∃ (rect : InscribedRectangle sector),
    (∀ (other : InscribedRectangle sector),
      rectangleArea sector rect ≥ rectangleArea sector other) ∧
    rect.ω = sector.α / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_inscribed_rectangle_l468_46891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_period_l468_46804

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define the function y = tan(2x)
noncomputable def f (x : ℝ) : ℝ := tan (2 * x)

-- State the theorem
theorem tan_2x_period :
  ∃ (P : ℝ), P > 0 ∧ (∀ (x : ℝ), f (x + P) = f x) ∧
  (∀ (Q : ℝ), Q > 0 ∧ (∀ (x : ℝ), f (x + Q) = f x) → P ≤ Q) ∧
  P = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_period_l468_46804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_maximized_l468_46851

/-- An arithmetic sequence with common difference d and first term a₁ -/
noncomputable def ArithmeticSequence (d : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def SumArithmeticSequence (d : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_maximized (d : ℝ) (a₁ : ℝ) (h₁ : d < 0) (h₂ : a₁^2 = (ArithmeticSequence d a₁ 11)^2) :
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧
    ∀ m : ℕ, SumArithmeticSequence d a₁ m ≤ SumArithmeticSequence d a₁ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_maximized_l468_46851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l468_46887

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Calculates the rise in liquid level when a sphere is submerged in a cone -/
noncomputable def liquidRise (c : Cone) (s : Sphere) : ℝ :=
  sphereVolume s / (Real.pi * c.radius^2)

theorem liquid_rise_ratio
  (cone1 cone2 : Cone)
  (sphere : Sphere)
  (h1 : cone1.radius = 5)
  (h2 : cone2.radius = 10)
  (h3 : sphere.radius = 2)
  (h4 : coneVolume cone1 = coneVolume cone2) :
  liquidRise cone1 sphere / liquidRise cone2 sphere = 4 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l468_46887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_equals_two_l468_46888

/-- Given a function f such that f(2x+1) = 3x+2 for all x, prove that f(1) = 2 -/
theorem f_of_one_equals_two (f : ℝ → ℝ) (h : ∀ x, f (2*x + 1) = 3*x + 2) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_equals_two_l468_46888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_remainder_l468_46883

def array_sum (p : ℕ) : ℚ :=
  (2 * p^2) / ((2 * p - 1) * (p - 1))

theorem array_sum_remainder (p : ℕ) (h : p = 2023) :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ 
    array_sum p = m / n ∧
    (m + n) % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_remainder_l468_46883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l468_46878

-- Define the open interval (-1, +∞)
def OpenIntervalMinusOneInfty : Set ℝ := {x : ℝ | x > -1}

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the property of the function
def SatisfiesInequality (f : FunctionType) : Prop :=
  ∀ x y, x ∈ OpenIntervalMinusOneInfty → y ∈ OpenIntervalMinusOneInfty →
    f (x + f y + x * f y) ≥ y + f x + y * f x

-- Main theorem
theorem function_characterization (f : FunctionType)
  (h_cont : Continuous f)
  (h_mono : StrictMono f)
  (h_zero : f 0 = 0)
  (h_ineq : SatisfiesInequality f) :
  (∀ x ∈ OpenIntervalMinusOneInfty, f x = -x / (1 + x)) ∨
  (∀ x ∈ OpenIntervalMinusOneInfty, f x = x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l468_46878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l468_46886

/-- The repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_to_fraction : ∃ (x : ℚ), x = 56 / 99 ∧ x = 0 + ∑' (n : ℕ), (56 : ℚ) / (100 ^ (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l468_46886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_A_and_AunionB_l468_46872

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem sets_A_and_AunionB :
  (A = {x : ℝ | x < 0 ∨ x > 2}) ∧
  (A ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_A_and_AunionB_l468_46872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_contain_odd_with_unit_digit_3_l468_46861

def divisors (n : ℕ) : Set ℕ :=
  {d | d > 0 ∧ n % d = 0}

def hasUnitDigit3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem divisors_contain_odd_with_unit_digit_3 (n : ℕ) (hn : n > 0) :
  ∃ d ∈ divisors n, Odd d ∧ hasUnitDigit3 d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_contain_odd_with_unit_digit_3_l468_46861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l468_46820

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (h1 : a = 9) 
  (h2 : b = 10) 
  (h3 : θ = 150 * Real.pi / 180) 
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ)) : 
  c = Real.sqrt (181 + 90 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l468_46820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_is_planar_curve_l468_46892

-- Define the conical surfaces
structure ConicalSurface where
  a : ℝ
  b : ℝ
  c : ℝ
  k : ℝ

-- Define the properties of the conical surfaces
def parallel_axes (c1 c2 : ConicalSurface) : Prop :=
  ∃ (v : ℝ × ℝ × ℝ), v.2.1 = 0 ∧ v.2.2 = 1

def equal_angles (c1 c2 : ConicalSurface) : Prop :=
  c1.k = c2.k

-- Define the intersection of two conical surfaces
def intersection (c1 c2 : ConicalSurface) : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | (p.1 - c1.a)^2 + (p.2.1 - c1.b)^2 = c1.k^2 * (p.2.2 - c1.c)^2 ∧
                    (p.1 - c2.a)^2 + (p.2.1 - c2.b)^2 = c2.k^2 * (p.2.2 - c2.c)^2}

-- Define a planar curve
def is_planar_curve (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ), ∀ p ∈ s, a * p.1 + b * p.2.1 + c * p.2.2 + d = 0

-- State the theorem
theorem intersection_is_planar_curve (c1 c2 : ConicalSurface)
  (h_parallel : parallel_axes c1 c2) (h_equal_angles : equal_angles c1 c2) :
  is_planar_curve (intersection c1 c2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_is_planar_curve_l468_46892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_area_ratio_l468_46884

theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 12
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  perimeter / area = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_area_ratio_l468_46884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dorothy_remaining_money_l468_46813

noncomputable def annual_income : ℝ := 60000
noncomputable def tax_bracket_1 : ℝ := 10000
noncomputable def tax_bracket_2 : ℝ := 50000
noncomputable def tax_rate_1 : ℝ := 0.10
noncomputable def tax_rate_2 : ℝ := 0.15
noncomputable def tax_rate_3 : ℝ := 0.25
noncomputable def monthly_bills : ℝ := 800
noncomputable def annual_savings : ℝ := 5000
noncomputable def retirement_rate : ℝ := 0.06
noncomputable def monthly_healthcare : ℝ := 300

noncomputable def calculate_tax (income : ℝ) : ℝ :=
  tax_rate_1 * tax_bracket_1 +
  tax_rate_2 * (min (tax_bracket_2 - tax_bracket_1) (max (income - tax_bracket_1) 0)) +
  tax_rate_3 * (max (income - tax_bracket_2) 0)

noncomputable def calculate_expenses (income : ℝ) : ℝ :=
  calculate_tax income +
  12 * monthly_bills +
  12 * monthly_healthcare +
  retirement_rate * income +
  annual_savings

theorem dorothy_remaining_money :
  annual_income - calculate_expenses annual_income = 28700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dorothy_remaining_money_l468_46813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jersey_profit_theorem_l468_46882

-- Define the basic parameters
def cost_A : ℚ := 200
def cost_B : ℚ := 180
def price_A : ℚ := 320
def price_B : ℚ := 280
def total_jerseys : ℕ := 210

-- Define the relationship between A and B jerseys
axiom jersey_relation : (30000 : ℚ) / cost_A = 3 * ((9000 : ℚ) / cost_B)

-- Define the cost difference between A and B jerseys
axiom cost_difference : cost_A = cost_B + 20

-- Define the profit function
def profit (m : ℚ) : ℚ := 20 * m + 21000

-- Define the range of m
def m_range (m : ℚ) : Prop := 100 ≤ m ∧ m ≤ 140

-- Define the maximum profit after charity
noncomputable def max_profit_after_charity (a : ℚ) : ℚ :=
  if 0 < a ∧ a < 20 then 23800 - 140 * a
  else if a = 20 then 21000
  else 23000 - 100 * a

-- State the theorem
theorem jersey_profit_theorem (m : ℚ) (a : ℚ) :
  m_range m →
  (∀ m, m_range m → profit m = 20 * m + 21000) ∧
  (max_profit_after_charity a = 
    if 0 < a ∧ a < 20 then 23800 - 140 * a
    else if a = 20 then 21000
    else 23000 - 100 * a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jersey_profit_theorem_l468_46882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equal_heads_five_coins_proof_l468_46809

/-- The probability of two people getting the same number of heads
    when each throws 5 fair coins independently -/
def prob_equal_heads_five_coins : ℚ :=
  63 / 256

/-- Function to calculate the probability of getting k heads in n throws of a fair coin -/
noncomputable def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- Function to calculate the probability that both people get exactly k heads in n throws -/
noncomputable def prob_both_k_heads (n k : ℕ) : ℚ :=
  (prob_k_heads n k) ^ 2

/-- The sum of probabilities for all possible values of k from 0 to n -/
noncomputable def sum_prob_equal_heads (n : ℕ) : ℚ :=
  Finset.sum (Finset.range (n + 1)) (λ k => prob_both_k_heads n k)

theorem prob_equal_heads_five_coins_proof :
  sum_prob_equal_heads 5 = prob_equal_heads_five_coins := by
  sorry

#eval prob_equal_heads_five_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equal_heads_five_coins_proof_l468_46809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l468_46880

/-- The equation of a line parameterized by (x,y) = (3t + 7, 5t - 8) is y = (5/3)x - 59/3 -/
theorem line_equation (t : ℝ) : 
  (5 * t - 8) = (5/3) * (3 * t + 7) - 59/3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l468_46880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_river_crossing_exists_l468_46855

-- Define the animals and sides of the river
inductive Animal : Type
| mouse : Animal
| cat : Animal
| dog : Animal

inductive Side : Type
| A : Side  -- Starting side
| B : Side  -- Destination side

-- Define the state of the river crossing
structure RiverState :=
  (man : Side)
  (mouse : Side)
  (cat : Side)
  (dog : Side)

-- Define a safe state
def is_safe (state : RiverState) : Prop :=
  (state.cat = state.dog → state.man = state.cat) ∧
  (state.cat = state.mouse → state.man = state.cat)

-- Define a valid move
def valid_move (s1 s2 : RiverState) : Prop :=
  (s1.man ≠ s2.man) ∧
  (s1.mouse = s2.mouse ∨ (s1.mouse ≠ s2.mouse ∧ s1.man = s1.mouse ∧ s2.man = s2.mouse)) ∧
  (s1.cat = s2.cat ∨ (s1.cat ≠ s2.cat ∧ s1.man = s1.cat ∧ s2.man = s2.cat)) ∧
  (s1.dog = s2.dog ∨ (s1.dog ≠ s2.dog ∧ s1.man = s1.dog ∧ s2.man = s2.dog))

-- Define the initial and final states
def initial_state : RiverState :=
  { man := Side.A, mouse := Side.A, cat := Side.A, dog := Side.A }

def final_state : RiverState :=
  { man := Side.B, mouse := Side.B, cat := Side.B, dog := Side.B }

-- Theorem: There exists a safe sequence of moves from initial to final state
theorem safe_river_crossing_exists : ∃ (moves : List RiverState), 
  moves.head? = some initial_state ∧
  moves.getLast? = some final_state ∧
  (∀ i, i < moves.length - 1 → 
    valid_move (moves.get ⟨i, by sorry⟩) (moves.get ⟨i + 1, by sorry⟩) ∧
    is_safe (moves.get ⟨i, by sorry⟩) ∧
    is_safe (moves.get ⟨i + 1, by sorry⟩)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_river_crossing_exists_l468_46855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_is_ten_percent_l468_46857

/-- Calculates simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculates compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_rate_is_ten_percent 
  (principal_si : ℝ) 
  (principal_ci : ℝ) 
  (rate_si : ℝ)
  (rate_ci : ℝ)
  (time_si : ℝ) 
  (time_ci : ℝ) 
  (h1 : principal_si = 2800)
  (h2 : principal_ci = 4000)
  (h3 : rate_si = 5)
  (h4 : time_si = 3)
  (h5 : time_ci = 2)
  (h6 : simple_interest principal_si rate_si time_si = 
        (1/2) * compound_interest principal_ci rate_ci time_ci) : 
  rate_ci = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_is_ten_percent_l468_46857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l468_46899

/-- The sum of repeating decimals 0.6̄ + 0.2̄ - 0.9̄ + 0.3̄ equals 2/9 -/
theorem repeating_decimal_sum : 
  (2/3 : ℚ) + (2/9 : ℚ) - (1 : ℚ) + (1/3 : ℚ) = 2/9 := by
  sorry

/-- 0.6̄ as a rational number -/
def repeating_decimal_06 : ℚ := 2/3

/-- 0.2̄ as a rational number -/
def repeating_decimal_02 : ℚ := 2/9

/-- 0.9̄ as a rational number -/
def repeating_decimal_09 : ℚ := 1

/-- 0.3̄ as a rational number -/
def repeating_decimal_03 : ℚ := 1/3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l468_46899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_with_given_vectors_l468_46811

/-- The area of a parallelogram formed by two vectors -/
noncomputable def parallelogram_area (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 * b.2 - a.2 * b.1) ^ 2)

theorem parallelogram_area_with_given_vectors :
  ∃ (p q : ℝ × ℝ),
    let a := (2 * p.1 + 3 * q.1, 2 * p.2 + 3 * q.2)
    let b := (p.1 - 2 * q.1, p.2 - 2 * q.2)
    Real.sqrt (p.1 ^ 2 + p.2 ^ 2) = 2 ∧
    Real.sqrt (q.1 ^ 2 + q.2 ^ 2) = 3 ∧
    Real.arccos ((p.1 * q.1 + p.2 * q.2) / (Real.sqrt (p.1 ^ 2 + p.2 ^ 2) * Real.sqrt (q.1 ^ 2 + q.2 ^ 2))) = π / 4 ∧
    parallelogram_area a b = 21 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_with_given_vectors_l468_46811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_theorem_round_robin_theorem_l468_46897

/-- Represents a table tennis match between two players -/
structure TableTennisMatch where
  prob_a_win : ℝ  -- Probability of player A winning a point when serving
  prob_b_win : ℝ  -- Probability of player B winning a point when serving
  first_server : Bool  -- True if A serves first, False if B serves first

/-- Represents a round-robin tournament with three players -/
structure RoundRobinTournament where
  prob_win : ℝ  -- Probability of winning each match

/-- The probability of player B leading 2-1 after 3 serves in a table tennis match -/
def prob_b_leading_2_1 (m : TableTennisMatch) : ℝ :=
  sorry

/-- The probability of needing a fifth match in a round-robin tournament -/
def prob_fifth_match (t : RoundRobinTournament) : ℝ :=
  sorry

theorem table_tennis_theorem (m : TableTennisMatch) 
  (h1 : m.prob_a_win = 0.6) 
  (h2 : m.prob_b_win = 0.4) 
  (h3 : m.first_server = true) : 
  prob_b_leading_2_1 m = 0.352 :=
sorry

theorem round_robin_theorem (t : RoundRobinTournament) 
  (h : t.prob_win = 0.5) : 
  prob_fifth_match t = 3/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_theorem_round_robin_theorem_l468_46897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_case_result_l468_46837

/-- Given a passing threshold, student's score, and failure margin, calculate the minimum possible maximum marks -/
def minimum_maximum_marks 
  (passing_threshold : ℚ) 
  (student_score : ℕ) 
  (failure_margin : ℕ) : ℕ :=
  let passing_score := student_score + failure_margin
  let max_marks := (passing_score : ℚ) / passing_threshold
  (Int.ceil max_marks).toNat

/-- The minimum possible maximum marks for the given conditions is 667 -/
theorem specific_case_result : 
  minimum_maximum_marks (45/100) 225 75 = 667 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_case_result_l468_46837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_theorem_l468_46869

-- Define the type for our polynomials
def SpecialPolynomial (n : ℕ) := Fin n → Int

-- Define what it means for a polynomial to have all real roots
def AllRealRoots (n : ℕ) (p : SpecialPolynomial n) : Prop := sorry

-- Define the set of valid polynomials
def ValidPolynomials : Set (Σ n : ℕ, SpecialPolynomial n) :=
  {⟨1, λ _ => -1⟩, ⟨1, λ _ => 1⟩,
   ⟨2, λ i => if i = 0 then 1 else -1⟩,
   ⟨2, λ i => if i = 0 then -1 else -1⟩,
   ⟨3, λ i => if i = 1 then -1 else 1⟩,
   ⟨3, λ i => if i = 0 then -1 else if i = 1 then -1 else 1⟩}

theorem special_polynomial_theorem :
  ∀ n : ℕ, ∀ p : SpecialPolynomial n,
    AllRealRoots n p → (n ≤ 3 ∧ ⟨n, p⟩ ∈ ValidPolynomials) :=
by sorry

#check special_polynomial_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_theorem_l468_46869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_area_ratio_l468_46866

/-- Represents a triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents the oblique projection of a triangle -/
noncomputable def obliqueProjection (t : Triangle) : Triangle :=
  { base := t.base,
    height := t.height * (Real.sqrt 2 / 4) }

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

/-- Theorem stating the ratio of areas between original and projected triangles -/
theorem oblique_projection_area_ratio (t : Triangle) :
  area t = 2 * Real.sqrt 2 * area (obliqueProjection t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_area_ratio_l468_46866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_east_south_angle_in_ten_ray_decoration_l468_46818

/-- Represents a circular floor decoration with equally spaced rays -/
structure CircularDecoration where
  num_rays : ℕ
  north_ray_index : ℕ

/-- Calculates the index of the ray pointing closest to East -/
def east_ray_index (d : CircularDecoration) : ℕ :=
  (d.north_ray_index + (d.num_rays / 4)) % d.num_rays

/-- Calculates the index of the ray pointing South -/
def south_ray_index (d : CircularDecoration) : ℕ :=
  (d.north_ray_index + (d.num_rays / 2)) % d.num_rays

/-- Calculates the angle between two rays given their indices -/
noncomputable def angle_between_rays (d : CircularDecoration) (i j : ℕ) : ℝ :=
  360 * (min ((j - i) % d.num_rays) ((i - j) % d.num_rays) : ℝ) / d.num_rays

/-- The main theorem stating that the smaller angle between the ray closest to East
    and the ray pointing South in a 10-ray circular decoration is 54 degrees -/
theorem east_south_angle_in_ten_ray_decoration :
  ∃ (d : CircularDecoration),
    d.num_rays = 10 ∧
    angle_between_rays d (east_ray_index d) (south_ray_index d) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_east_south_angle_in_ten_ray_decoration_l468_46818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l468_46865

theorem divisible_by_six_percentage (n : ℕ) : n = 150 →
  (100 * (Finset.filter (λ x : ℕ => x % 6 = 0) (Finset.range (n + 1))).card) / n = 100 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l468_46865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_properties_l468_46805

def data_set : List ℝ := [-1, 0, 4, 6, 7, 14]

def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

noncomputable def median (l : List ℝ) : ℝ :=
  if l.length % 2 = 0
  then (l[l.length / 2 - 1]! + l[l.length / 2]!) / 2
  else l[l.length / 2]!

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem data_set_properties :
  is_ascending data_set ∧
  median data_set = 5 ∧
  mean data_set = 5 ∧
  variance data_set = 74/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_properties_l468_46805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_earnings_l468_46843

/-- Represents the work time in minutes for each day of the week -/
structure WorkWeek :=
  (monday : ℚ)
  (tuesday : ℚ)
  (thursday : ℚ)
  (saturday : ℚ)

/-- Calculates the total earnings for a given work week and hourly rate -/
def calculateEarnings (week : WorkWeek) (hourlyRate : ℚ) : ℚ :=
  let totalMinutes := week.monday + week.tuesday + week.thursday + week.saturday
  let totalHours := totalMinutes / 60
  totalHours * hourlyRate

/-- Anna's work week -/
def annasWeek : WorkWeek :=
  { monday := 150,    -- 2.5 hours in minutes
    tuesday := 80,
    thursday := 165,  -- 2 hours and 45 minutes in minutes
    saturday := 45 }

/-- Anna's hourly rate -/
def annasHourlyRate : ℚ := 5

theorem annas_earnings :
  ⌈calculateEarnings annasWeek annasHourlyRate⌉ = 37 := by
  -- Unfold definitions
  unfold calculateEarnings
  unfold annasWeek
  unfold annasHourlyRate
  -- Perform calculations
  simp [WorkWeek.monday, WorkWeek.tuesday, WorkWeek.thursday, WorkWeek.saturday]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_earnings_l468_46843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_gcd_l468_46801

/-- Represents a positive three-digit integer -/
def ThreeDigitInt := {n : ℕ // 100 ≤ n ∧ n < 1000}

/-- Constructs the seven-digit number from a three-digit number -/
def sevenDigitNumber (n : ThreeDigitInt) : ℕ :=
  n.val * 1000000 + n.val * 1000 + n.val

/-- The theorem stating that 1001001 is the GCD of all such seven-digit numbers -/
theorem seven_digit_gcd :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ThreeDigitInt), d ∣ sevenDigitNumber n) ∧
  (∀ (m : ℕ), m > 0 → (∀ (n : ThreeDigitInt), m ∣ sevenDigitNumber n) → m ≤ d) ∧
  d = 1001001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_gcd_l468_46801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l468_46807

-- Define the basic geometric objects
def Line : Type := Unit
def Plane : Type := Unit
def Cone : Type := Unit

-- Define the relationships between geometric objects
def parallel (a b : Line) : Prop := sorry
def perpendicular (a b : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def parallel_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane (p1 p2 : Plane) : Prop := sorry
def axial_section (c : Cone) : Plane := sorry
def cut_cone (c : Cone) (p : Plane) : Prop := sorry

-- Define a placeholder for area calculation
def area (p : Plane) : ℝ := sorry

-- Define the propositions
def proposition1 (a b : Line) (α : Plane) : Prop :=
  parallel a b → contained_in b α → parallel_plane a α

def proposition2 (a b : Line) (α β : Plane) : Prop :=
  contained_in a α → contained_in b β → perpendicular_plane α β → perpendicular a b

def proposition3 (c : Cone) (p : Plane) : Prop :=
  cut_cone c p → ∃ (c' f : Cone), True  -- Simplified representation

def proposition4 (c : Cone) : Prop :=
  ∀ (p : Plane), axial_section c = p → 
    ∀ (q : Plane), area (axial_section c) ≥ area q

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  (∃ a b α, ¬proposition1 a b α) ∧
  (∃ a b α β, ¬proposition2 a b α β) ∧
  (∃ c p, ¬proposition3 c p) ∧
  (∃ c, ¬proposition4 c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l468_46807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l468_46841

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x - 1 else x^2

-- State the theorem
theorem range_of_f (x : ℝ) : f (x + 1) < 4 ↔ x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l468_46841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_30_between_square_and_cube_l468_46885

theorem multiples_of_30_between_square_and_cube : 
  let start := 900
  let end_val := 27000
  let count := (Finset.range ((end_val - start) / 30 + 1)).filter (fun n => (start + 30 * n) % 30 = 0) |>.card
  count = 871 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_30_between_square_and_cube_l468_46885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_trajectory_properties_l468_46831

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P on the circle
def P (x y : ℝ) : Prop := myCircle x y

-- Define the midpoint M
def M (x y : ℝ) : Prop := ∃ (px py : ℝ), P px py ∧ x = px ∧ y = py / 2

-- State the theorem
theorem midpoint_trajectory :
  ∀ (x y : ℝ), M x y → x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Define the foci
noncomputable def foci : Set (ℝ × ℝ) := {(-Real.sqrt 3, 0), (Real.sqrt 3, 0)}

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

-- State the theorem about foci and eccentricity
theorem trajectory_properties :
  ∀ (x y : ℝ), M x y →
    (∃ (f : ℝ × ℝ), f ∈ foci ∧ 
      (x - f.1)^2 + y^2 = (1 - eccentricity^2) * (x^2 / 4 + y^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_trajectory_properties_l468_46831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_catch_time_l468_46896

/-- The usual time to catch the bus, in minutes -/
noncomputable def usual_time : ℚ := 12

/-- The ratio of reduced speed to usual speed -/
def speed_ratio : ℚ := 4/5

/-- The additional time taken when walking at reduced speed, in minutes -/
def additional_time : ℚ := 3

theorem bus_catch_time : 
  (∀ (s t : ℚ), s > 0 → t > 0 → s * t = speed_ratio * s * (t + additional_time)) →
  usual_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_catch_time_l468_46896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l468_46828

/-- The range of the area of a rhombus ABCD with specific constraints -/
theorem rhombus_area_range :
  ∀ (r : ℝ), 1 < r → r < 2 →
  ∃ (S : Set ℝ),
    S = { s | ∃ (x y a b : ℝ),
      -- Line l: y = √3x + 4
      y = Real.sqrt 3 * x + 4 ∧
      -- Circle O: x² + y² = r²
      x^2 + y^2 = r^2 ∧
      -- Rhombus ABCD with internal angle of 60°
      -- (implicitly used in the area formula)
      -- Vertices A and B on line l
      -- Vertices C and D on circle O
      -- (these conditions are implicitly used in the constraints)
      -- Area of rhombus
      s = (Real.sqrt 3 / 2) * a^2 ∧
      -- Range of a (derived from the problem constraints)
      ((0 < a ∧ a < Real.sqrt 3) ∨ (Real.sqrt 3 < a ∧ a < 2 * Real.sqrt 3))
    } ∧
    S = Set.Ioo 0 ((3 * Real.sqrt 3) / 2) ∪ Set.Ioo ((3 * Real.sqrt 3) / 2) (6 * Real.sqrt 3) := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l468_46828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMCN_equals_31_25_l468_46802

/-- Represents a parallelogram ABCD with given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Represents the area of a region AMCN in a parallelogram ABCD -/
noncomputable def area_AMCN (p : Parallelogram) : ℝ :=
  p.base * p.height - (p.base * (p.height / 2)) / 2 - (p.base * (p.height / 2)) / 2

/-- Theorem stating that the area of AMCN in the given parallelogram is 31.25 -/
theorem area_AMCN_equals_31_25 :
  let p : Parallelogram := { base := 10, height := 5 }
  area_AMCN p = 31.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMCN_equals_31_25_l468_46802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_intersection_not_always_parallel_l468_46825

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define planes and lines
variable (α β : Subspace ℝ V)
variable (m n : AffineSubspace ℝ V)

-- Define the conditions
variable (h_distinct_planes : α ≠ β)
variable (h_distinct_lines : m ≠ n)

-- Define parallelism and intersection
def parallel_plane_line (p : Subspace ℝ V) (l : AffineSubspace ℝ V) : Prop :=
  l.direction ≤ p

def intersection_planes (p q : Subspace ℝ V) : Subspace ℝ V :=
  p ⊓ q

-- State the theorem
theorem parallel_line_intersection_not_always_parallel
  (h_m_parallel_α : parallel_plane_line α m)
  (h_intersection : intersection_planes α β = n.direction) :
  ¬ (parallel_plane_line α m → n.direction ≤ m.direction) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_intersection_not_always_parallel_l468_46825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l468_46859

/-- Piecewise function f --/
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Ioo (1/2) 1 then (2 * x^3) / (x + 1)
  else if x ∈ Set.Icc 0 (1/2) then -x/3 + 1/6
  else 0

/-- Function g with parameter a --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi/6 * x) - 2*a + 2

/-- Theorem statement --/
theorem range_of_a (a : ℝ) :
  a > 0 →
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f x₁ = g a x₂) →
  a ∈ Set.Icc (1/2) (4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l468_46859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_to_compress_spring_l468_46806

/-- Work required to compress an elastic spring -/
theorem work_to_compress_spring (k : ℝ) (Δh : ℝ) :
  (∫ x in (0)..(Δh), -k * x) = -k * (Δh^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_to_compress_spring_l468_46806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l468_46879

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x

-- State the theorem
theorem f_lower_bound (m : ℝ) (h : m < Real.exp 1 / 2 - 1) :
  ∀ x : ℝ, x ≥ 0 → f m x > Real.exp 1 / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l468_46879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l468_46846

theorem trig_problem (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan β = 4/3) (h4 : Real.sin (α + β) = 5/13) :
  (Real.cos β)^2 + Real.sin (2*β) = 33/25 ∧ Real.cos α = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l468_46846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_avoiding_circle_l468_46800

/-- Helper function to calculate path length -/
noncomputable def path_length (path : ℝ → ℝ × ℝ) : ℝ := sorry

/-- The shortest path length from (0,0) to (20,25) avoiding a circle -/
theorem shortest_path_avoiding_circle :
  let start : ℝ × ℝ := (0, 0)
  let end_point : ℝ × ℝ := (20, 25)
  let circle_center : ℝ × ℝ := (10, 12.5)
  let circle_radius : ℝ := 8
  let is_inside_circle (p : ℝ × ℝ) : Prop :=
    (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 < circle_radius^2
  ∃ (path : ℝ → ℝ × ℝ),
    (path 0 = start) ∧
    (path 1 = end_point) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ¬is_inside_circle (path t)) ∧
    (∀ other_path : ℝ → ℝ × ℝ,
      (other_path 0 = start) →
      (other_path 1 = end_point) →
      (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ¬is_inside_circle (other_path t)) →
      path_length path ≤ path_length other_path) ∧
    path_length path = 27.732 + (8 * Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_avoiding_circle_l468_46800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_greater_than_radius_l468_46845

-- Define a circle with center O and radius 3
def myCircle (O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 3}

-- Define a point P outside the circle
def outside_point (O P : ℝ × ℝ) : Prop :=
  P ∉ myCircle O

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem statement
theorem distance_greater_than_radius (O P : ℝ × ℝ) 
  (h : outside_point O P) : distance O P > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_greater_than_radius_l468_46845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l468_46893

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (27 : ℝ)^x - 6 * (3 : ℝ)^x + 10

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -4 * Real.sqrt 2 + 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l468_46893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_symmetry_l468_46871

theorem sin_shift_symmetry (x : ℝ) : 
  Real.sin (x + 3 * Real.pi / 2) = Real.sin ((-x) + 3 * Real.pi / 2) := by
  sorry

#check sin_shift_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_symmetry_l468_46871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l468_46844

/-- Representation of an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  k : ℝ
  eq : (x : ℝ) → (y : ℝ) → Prop := λ x y => x^2 / a^2 + y^2 / b^2 = 1
  b_eq : b^2 = 4 - k

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem stating the possible values of k for an ellipse with eccentricity 1/3 -/
theorem ellipse_k_values (e : Ellipse) :
  e.a = 3 ∧ eccentricity e = 1/3 → e.k = -4 ∨ e.k = -49/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l468_46844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_asymptotes_specific_hyperbola_l468_46870

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def angle_between_asymptotes (a b : ℝ) : ℝ :=
  2 * Real.arctan (b / a)

/-- Proves that the angle between the asymptotes of the hyperbola x^2 - y^2/3 = 1 is π/3 -/
theorem angle_asymptotes_specific_hyperbola :
  angle_between_asymptotes 1 (Real.sqrt 3) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_asymptotes_specific_hyperbola_l468_46870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l468_46860

-- Define the function f(x) = (x-3)/(x-2)
noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≠ 2}

-- Theorem stating that the domain of f is {x ∈ ℝ | x ≠ 2}
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ ∃ y : ℝ, f x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l468_46860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_opposite_changes_l468_46877

noncomputable def f (x : ℝ) : ℝ :=
  if x < (1/2 : ℝ) then x + (1/2 : ℝ) else x^2

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => f (sequence_a a n)

noncomputable def sequence_b (b : ℝ) : ℕ → ℝ
  | 0 => b
  | n + 1 => f (sequence_b b n)

theorem existence_of_opposite_changes (a b : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  ∃ n : ℕ, (sequence_a a (n + 1) - sequence_a a n) * 
            (sequence_b b (n + 1) - sequence_b b n) < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_opposite_changes_l468_46877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_c_less_than_b_l468_46842

noncomputable def a : ℝ := (1/2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
noncomputable def b : ℝ := (2 * Real.tan (13 * Real.pi / 180)) / (1 + Real.tan (13 * Real.pi / 180) ^ 2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

-- State the theorem to be proved
theorem a_less_than_c_less_than_b : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_c_less_than_b_l468_46842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_minus_two_alpha_l468_46848

theorem cos_two_thirds_pi_minus_two_alpha (α : ℝ) :
  Real.sin (π / 6 + α) = 1 / 3 → Real.cos (2 * π / 3 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_minus_two_alpha_l468_46848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l468_46895

def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := U \ {1, 2}

theorem intersection_A_B : A ∩ B = {3} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l468_46895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l468_46838

theorem count_integers_in_range : 
  let lower_bound := (10 : ℕ)^4
  let upper_bound := (10 : ℕ)^5
  let count := Finset.filter (λ n : ℕ ↦ lower_bound ≤ n ∧ n ≤ upper_bound) (Finset.range (upper_bound + 1))
  Finset.card count = 9 * 10^4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l468_46838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_plus_n_value_l468_46835

theorem m_plus_n_value (m n : ℕ) (h1 : n > 1) (h2 : m^n = 2^25 * 3^40) : m + n = 209957 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_plus_n_value_l468_46835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_forms_right_triangle_l468_46864

noncomputable def right_triangle_identification (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def set_A : Prop := right_triangle_identification 1 1 1
noncomputable def set_B : Prop := right_triangle_identification 2 3 4
noncomputable def set_C : Prop := right_triangle_identification (Real.sqrt 5) 3 4
noncomputable def set_D : Prop := right_triangle_identification 1 (Real.sqrt 3) 2

theorem only_set_D_forms_right_triangle : 
  ¬set_A ∧ ¬set_B ∧ ¬set_C ∧ set_D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_forms_right_triangle_l468_46864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l468_46890

noncomputable def v (x : ℝ) : ℝ := 1 / (x^(3/2))

theorem domain_of_v :
  {x : ℝ | ∃ y, v x = y} = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l468_46890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l468_46876

noncomputable def f (a x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

theorem max_value_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 14 ∧ ∀ y : ℝ, -1 ≤ y ∧ y ≤ 1 → f a y ≤ 14) ↔ 
  (a = 3 ∨ a = 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l468_46876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l468_46873

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def solution_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2/3} ∪ {1}

theorem inequality_solution :
  {x : ℝ | |3*x + 1| - (floor x) - 3 ≤ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l468_46873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l468_46881

noncomputable section

def Circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 - 6*p.2 - 16) = 0}

def A : ℝ × ℝ := (-3, 0)

def Line := ℝ × ℝ → Prop

def intersects (l : Line) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ M N, M ∈ c ∧ N ∈ c ∧ M ≠ N ∧ l M ∧ l N

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem line_equation (l : Line) :
  (l A) ∧ 
  (intersects l Circle) ∧ 
  (∃ M N, M ∈ Circle ∧ N ∈ Circle ∧ M ≠ N ∧ l M ∧ l N ∧ distance M N = 8) →
  ((∀ p, l p ↔ p.1 = -3) ∨ (∀ p, l p ↔ p.2 = 0)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l468_46881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l468_46833

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 1/2; -3, 5]
  Matrix.det A = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l468_46833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l468_46827

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.sin (π/2 + 2*α) = -4/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (α + π/4) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l468_46827
