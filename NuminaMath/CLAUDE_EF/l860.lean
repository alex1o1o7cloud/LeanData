import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_ten_power_minus_two_equals_point_one_l860_86016

theorem sqrt_ten_power_minus_two_equals_point_one :
  Real.sqrt (10 ^ (-2 : ℤ)) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_ten_power_minus_two_equals_point_one_l860_86016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_is_10_sqrt_2_l860_86028

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 14*y + 15 = 0

/-- The diameter of the circle -/
noncomputable def circle_diameter : ℝ := 10 * Real.sqrt 2

/-- Theorem stating that the diameter of the circle described by the given equation is 10√2 -/
theorem circle_diameter_is_10_sqrt_2 :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_diameter = 2 * radius := by
  -- We'll use center (-4, 7) and radius 5√2
  let center_x := -4
  let center_y := 7
  let radius := 5 * Real.sqrt 2
  
  use center_x, center_y, radius
  
  constructor
  
  · intro x y
    -- The proof of the equivalence goes here
    sorry
  
  · -- Proof that circle_diameter = 2 * radius
    unfold circle_diameter
    simp [radius]
    -- The rest of the proof goes here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_is_10_sqrt_2_l860_86028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_pairs_count_l860_86064

def S : Finset Nat := Finset.range 30

theorem gcd_power_pairs_count :
  (Finset.filter (fun p : Nat × Nat =>
    p.1 ∈ S ∧ p.2 ∈ S ∧
    (Nat.gcd (2^p.1 + 1) (2^p.2 - 1) > 1))
    (Finset.product S S)).card = 295 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_pairs_count_l860_86064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l860_86035

open BigOperators

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1)) →
  (∀ n : ℕ, (S n - 1)^2 = a n * S n) →
  ∀ n : ℕ, a n = 1 / (n * (n + 1)) := by
  intros hS hGM
  -- Proof goes here
  sorry

#check sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l860_86035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l860_86058

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / b^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0 ∨ Real.sqrt 3 * x - y = 0

-- Theorem statement
theorem hyperbola_asymptotes (b : ℝ) (h1 : b > 0) :
  ∃ (x y : ℝ), hyperbola b x y ∧ asymptote x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l860_86058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alarm_clock_noon_time_l860_86003

/-- Represents the time difference between a slow alarm clock and a correct clock -/
def timeDifference (slowRate : ℚ) (elapsedTime : ℚ) : ℚ :=
  slowRate * elapsedTime

/-- Calculates when the alarm clock will show noon given its slow rate and elapsed time -/
def alarmClockNoon (slowRate : ℚ) (elapsedTime : ℚ) : ℚ :=
  12 + (timeDifference slowRate elapsedTime) / 60

theorem alarm_clock_noon_time (slowRate elapsedTime : ℚ) :
  slowRate = 4 / 60 → elapsedTime = 7 / 2 →
  14 / 60 = alarmClockNoon slowRate elapsedTime - 12 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval alarm_clock_noon_time (4/60) (7/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alarm_clock_noon_time_l860_86003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_two_l860_86025

/-- A function f(x) that reaches its maximum at x = 1 with a value of -2 -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a b : ℝ) (x : ℝ) : ℝ := a / x + (-b) / (x^2)

theorem f_derivative_at_two (a b : ℝ) :
  f a b 1 = -2 →
  f_derivative a b 1 = 0 →
  f_derivative a b 2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_two_l860_86025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l860_86049

theorem imaginary_part_of_z (θ : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0 ∧ z.im ≠ 0) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l860_86049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l860_86022

theorem tan_alpha_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) →  -- α is in the fourth quadrant
  (Real.sin α = -Real.sqrt 3 / 2) →    -- sin α = -√3/2
  (Real.tan α = -Real.sqrt 3) :=       -- tan α = -√3
by
  intros h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l860_86022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l860_86036

noncomputable def f (A ω φ B x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + B

theorem function_and_range 
  (A ω φ B : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : |φ| < π/2)
  (h_1 : f A ω φ B (-π/6) = -1)
  (h_2 : f A ω φ B (π/3) = 1)
  (h_3 : f A ω φ B (5*π/6) = 3)
  (h_4 : f A ω φ B (4*π/3) = 1)
  (h_5 : f A ω φ B (11*π/6) = -1) :
  (∀ x, f A ω φ B x = 2 * Real.sin (x - π/3) + 1) ∧
  (Set.image (λ x ↦ 2 * Real.sin (3*x - π/3) + 1) (Set.Icc 0 (π/3)) = Set.Icc (-Real.sqrt 3 + 1) 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l860_86036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_close_points_in_triangle_l860_86033

-- Define an equilateral triangle
structure EquilateralTriangle where
  sideLength : ℝ
  is_positive : sideLength > 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Helper function (not provable, just for statement completeness)
def point_in_triangle (t : EquilateralTriangle) (p : Point) : Prop :=
  sorry

-- State the theorem
theorem min_close_points_in_triangle 
  (triangle : EquilateralTriangle) 
  (points : List Point) : 
  triangle.sideLength = 2 → 
  points.length = 5 → 
  (∀ p, p ∈ points → point_in_triangle triangle p) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_close_points_in_triangle_l860_86033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_ratio_l860_86000

theorem garden_ratio (side_length : ℚ) (perimeter : ℚ) : 
  side_length = 15 →
  perimeter = 4 * side_length →
  side_length / perimeter = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_ratio_l860_86000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_diff_sum_l860_86093

-- Define the quadratic equation
noncomputable def quadratic (x : ℝ) := 5 * x^2 - 9 * x - 14

-- Define the roots of the quadratic equation
noncomputable def root1 : ℝ := (9 + Real.sqrt 361) / 10
noncomputable def root2 : ℝ := (9 - Real.sqrt 361) / 10

-- Define the positive difference between the roots
noncomputable def root_diff : ℝ := |root1 - root2|

-- Define the property of m not being divisible by the square of any prime
def not_divisible_by_prime_square (m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ m)

-- The main theorem
theorem quadratic_root_diff_sum (m n : ℕ) 
  (h1 : root_diff = Real.sqrt (m : ℝ) / n)
  (h2 : not_divisible_by_prime_square m) : 
  m + n = 366 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_diff_sum_l860_86093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_polynomial_characterization_l860_86066

/-- A polynomial that satisfies the identity x P(x-1) ≡ (x-26) P(x) -/
def IdentitySatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * P (x - 1) = (x - 26) * P x

/-- The specific form of polynomial that satisfies the identity -/
def SpecificPolynomial (c : ℝ) (x : ℝ) : ℝ :=
  c * x * (List.range 26).foldl (fun acc i => acc * (x - i)) 1

theorem identity_polynomial_characterization :
  ∀ P : ℝ → ℝ, IdentitySatisfyingPolynomial P ↔
    ∃ c : ℝ, ∀ x : ℝ, P x = SpecificPolynomial c x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_polynomial_characterization_l860_86066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l860_86080

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = ∑' i : ℕ, (24 : ℚ) / (100 : ℚ)^(i + 1) ∧ 
  n + d = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l860_86080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_to_line_l_l860_86088

noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

def line_l (x y : ℝ) : Prop := x + y = 3

noncomputable def distance_point_to_line (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  abs (x + y - 3) / Real.sqrt 2

theorem min_distance_curve_C_to_line_l :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 / 2 ∧
  ∀ (α : ℝ), distance_point_to_line (curve_C α) ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_to_line_l_l860_86088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_sphere_area_l860_86047

/-- A regular tetrahedron with base edge length √2 and all lateral faces being right triangles -/
structure RegularTetrahedron where
  base_edge_length : ℝ
  base_edge_length_eq : base_edge_length = Real.sqrt 2
  lateral_faces_right : Bool

/-- The circumscribed sphere of a regular tetrahedron -/
def circumscribed_sphere (t : RegularTetrahedron) (r : ℝ) : Prop :=
  sorry -- Definition to be filled later

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 
  4 * Real.pi * radius ^ 2

/-- Theorem: The surface area of the circumscribed sphere of a regular tetrahedron
    with base edge length √2 and all lateral faces being right triangles is 3π -/
theorem regular_tetrahedron_sphere_area (t : RegularTetrahedron) 
  (h1 : t.lateral_faces_right = true) :
  ∃ (r : ℝ), circumscribed_sphere t r ∧ sphere_surface_area r = 3 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_sphere_area_l860_86047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_heights_equal_l860_86091

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
structure Trapezoid where
  -- Define the trapezoid structure
  -- (We don't need to fully define it for this statement)
  mk :: -- Add a constructor to avoid warnings

/-- The height of a trapezoid at a given point is the perpendicular distance
    from that point to the opposite side. -/
def Trapezoid.height (t : Trapezoid) (p : ℝ × ℝ) : ℝ :=
  sorry

/-- All heights in a trapezoid are equal. -/
theorem trapezoid_heights_equal (t : Trapezoid) :
  ∀ p q : ℝ × ℝ, t.height p = t.height q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_heights_equal_l860_86091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_closed_interval_l860_86011

open Set
open Function
open Interval

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 2*x + 3)

-- Define the domain
def domain : Set ℝ := Icc 0 3

-- Theorem statement
theorem f_range_is_closed_interval :
  range (f ∘ (coe : domain → ℝ)) = Icc (1/6) (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_closed_interval_l860_86011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_calculation_l860_86045

theorem profit_percent_calculation (purchase_price repair_cost selling_price : ℚ) : 
  purchase_price = 45000 →
  repair_cost = 12000 →
  selling_price = 80000 →
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  profit_percent = 40.35 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_calculation_l860_86045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_iff_perpendicular_to_two_intersecting_lines_l860_86084

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields
  (point : ℝ × ℝ × ℝ)
  (direction : ℝ × ℝ × ℝ)

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields
  (point : ℝ × ℝ × ℝ)
  (normal : ℝ × ℝ × ℝ)

/-- Predicate for a line being contained in a plane -/
def Line3D.containedIn (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def Line3D.parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being perpendicular -/
def Line3D.perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to a plane -/
def Line3D.perpendicularToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate for two lines intersecting at a point -/
def Line3D.intersectAt (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem: A line is perpendicular to a plane if and only if it is perpendicular to two distinct intersecting lines in the plane -/
theorem line_perpendicular_to_plane_iff_perpendicular_to_two_intersecting_lines
  (a : Line3D) (p : Plane3D) :
  Line3D.perpendicularToPlane a p ↔
  ∃ (b c : Line3D),
    b.containedIn p ∧
    c.containedIn p ∧
    a.perpendicular b ∧
    a.perpendicular c ∧
    b.intersectAt c :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_iff_perpendicular_to_two_intersecting_lines_l860_86084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_nine_l860_86076

/-- Represents a digit in base 6 -/
def Base6Digit : Type := { n : ℕ // n < 6 }

/-- Represents a base greater than 7 -/
def BaseGreaterThan7 : Type := { n : ℕ // n > 7 }

/-- Converts a number in base 6 to decimal -/
def toDecimal (b : Base6Digit) : ℕ := 43 * b.val

/-- Converts a number in base c to decimal -/
def fromBaseC (c : BaseGreaterThan7) : ℕ := 5 * c.val + 5

/-- States that BBB₆ = 55ₖ -/
def base_equality (b : Base6Digit) (c : BaseGreaterThan7) : Prop :=
  toDecimal b = fromBaseC c

/-- The theorem to be proved -/
theorem smallest_sum_is_nine :
  ∃ (b : Base6Digit) (c : BaseGreaterThan7),
    base_equality b c ∧
    ∀ (b' : Base6Digit) (c' : BaseGreaterThan7),
      base_equality b' c' → b.val + c.val ≤ b'.val + c'.val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_nine_l860_86076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_percentage_l860_86008

-- Define the original price and reduced price
def P : ℚ := 2727 / 100  -- Rational representation of 27.27
def R : ℚ := 24

-- Define the additional amount of oil obtained and the amount spent
def additional_oil : ℚ := 6
def amount_spent : ℚ := 1200

-- Theorem to prove
theorem oil_price_reduction_percentage :
  let percentage_reduction := (P - R) / P * 100
  (amount_spent / R - amount_spent / P = additional_oil) →
  (abs (percentage_reduction - 1199 / 100) < 1 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_percentage_l860_86008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l860_86062

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
noncomputable def line1 : Line2D :=
  { point := (1, 4),
    direction := (2, -3) }

/-- The second line -/
noncomputable def line2 : Line2D :=
  { point := (3, 9),
    direction := (4, -1) }

/-- A point lies on a line if it satisfies the parametric equation -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

/-- The intersection point of the two lines -/
noncomputable def intersectionPoint : ℝ × ℝ := (-17/5, 53/5)

/-- Theorem: The intersection point lies on both lines and is unique -/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 → pointOnLine p line2 → p = intersectionPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l860_86062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_division_bound_l860_86007

def euclidean_division_step (a b : ℕ) : ℕ × ℕ :=
  if a ≥ b then (b, a % b) else (a, b % a)

def euclidean_division_process (a b : ℕ) : ℕ :=
  let rec aux (a b : ℕ) (steps : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then steps
    else if b = 0 then steps
    else 
      let (a', b') := euclidean_division_step a b
      aux a' b' (steps + 1) (fuel - 1)
  aux a b 0 7  -- We use 7 as the maximum number of steps (0 to 6)

theorem euclidean_division_bound (a b : ℕ) (h : max a b ≤ 1988) : 
  euclidean_division_process a b ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_division_bound_l860_86007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_product_l860_86098

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2012 * 3 * 7 is 18 -/
theorem sum_of_digits_of_product : ∃ (n : ℕ), 
  n = 2^2010 * 5^2012 * 3 * 7 ∧ 
  (Finset.sum (Finset.range 10) (λ d ↦ d * (n.digits 10).count d)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_product_l860_86098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_range_l860_86056

/-- The function f(x) defined as x|x^2 - a| -/
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x^2 - a)

/-- Theorem stating the equivalence between the inequality condition and the range of a -/
theorem f_inequality_iff_a_range (a : ℝ) :
  (a > 0 ∧ ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → 
    abs (f a x₁ - f a x₂) < Real.sqrt 2 / 2) ↔ 
  a ∈ Set.Ioo (1 - Real.sqrt 2 / 2) (3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_range_l860_86056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_selected_zero_l860_86072

-- Define the probabilities of selection for each sibling
noncomputable def prob_X : ℝ := 1/3
noncomputable def prob_Y : ℝ := 2/5
noncomputable def prob_Z : ℝ := 1/4

-- Define the maximum number of siblings that can be selected
def max_siblings_selected : ℕ := 2

-- Define a placeholder for the probability_all_selected function
noncomputable def probability_all_selected : ℝ := 0

-- Theorem stating that the probability of all three siblings being selected is 0
theorem prob_all_selected_zero :
  max_siblings_selected < 3 → prob_X > 0 ∧ prob_Y > 0 ∧ prob_Z > 0 → 
  probability_all_selected = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_selected_zero_l860_86072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_range_l860_86065

/-- The circle centered at (3, 0) with radius 2 -/
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

/-- The line y = 2x -/
def myLine (x y : ℝ) : Prop := y = 2 * x

/-- Theorem: If P(x₀, 2x₀) is on the line y = 2x and there exist points A and B on the circle
    (x-3)^2 + y^2 = 4 such that PA ⊥ PB, then x₀ ∈ [1/5, 1] -/
theorem point_range (x₀ : ℝ) :
  (∃ (y₀ : ℝ), myLine x₀ y₀) →
  (∃ (xa ya xb yb : ℝ), myCircle xa ya ∧ myCircle xb yb ∧
    ((xa - x₀) * (xb - x₀) + (ya - 2*x₀) * (yb - 2*x₀) = 0)) →
  x₀ ∈ Set.Icc (1/5 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_range_l860_86065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_f_le_g_l860_86034

open Real

/-- The function f(x) = ln x - x² + x -/
noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x

/-- The function g(x) = (m-1)x² + 2mx - 1 -/
def g (m : ℤ) (x : ℝ) : ℝ := (m - 1 : ℝ) * x^2 + 2 * (m : ℝ) * x - 1

/-- The theorem stating that the minimum integer value of m for which f(x) ≤ g(x) holds for all x > 0 is 1 -/
theorem min_m_for_f_le_g : 
  (∃ m : ℤ, ∀ x > 0, f x ≤ g m x) ∧ 
  (∀ m : ℤ, m < 1 → ∃ x > 0, f x > g m x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_f_le_g_l860_86034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_triangle_trigonometry_l860_86010

-- Problem 1
theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, -1/2 * x^2 + 2*x > m*x ↔ 0 < x ∧ x < 2) → m = 1 := by sorry

-- Problem 2
theorem triangle_trigonometry (A B C : ℝ) :
  Real.sin A = 5/13 ∧ Real.cos B = 3/5 → Real.cos C = -16/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_triangle_trigonometry_l860_86010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_example_l860_86092

/-- Calculates the total charge for a taxi trip given the initial fee, per-increment charge, increment distance, and total trip distance. -/
noncomputable def taxi_charge (initial_fee : ℝ) (per_increment_charge : ℝ) (increment_distance : ℝ) (trip_distance : ℝ) : ℝ :=
  initial_fee + (trip_distance / increment_distance) * per_increment_charge

/-- Proves that the total charge for a 3.6-mile trip with an initial fee of $2.25 and $0.15 per 2/5 mile is $4.68. -/
theorem taxi_charge_example : 
  taxi_charge 2.25 0.15 (2/5) 3.6 = 4.68 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_example_l860_86092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l860_86015

/-- Calculates the kilometers per liter for a car given its speed, time, fuel consumption, and conversion factors. -/
noncomputable def kilometers_per_liter (speed : ℝ) (time : ℝ) (fuel_consumed : ℝ) 
  (miles_to_km : ℝ) (gallons_to_liters : ℝ) : ℝ :=
  let distance_miles := speed * time
  let distance_km := distance_miles * miles_to_km
  let fuel_liters := fuel_consumed * gallons_to_liters
  distance_km / fuel_liters

/-- Theorem stating that a car traveling at 117 mph for 5.7 hours and consuming 3.9 gallons 
    can travel approximately 72.01 km per liter. -/
theorem car_fuel_efficiency :
  let speed : ℝ := 117
  let time : ℝ := 5.7
  let fuel_consumed : ℝ := 3.9
  let miles_to_km : ℝ := 1.6
  let gallons_to_liters : ℝ := 3.8
  |kilometers_per_liter speed time fuel_consumed miles_to_km gallons_to_liters - 72.01| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l860_86015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l860_86079

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the condition for C₃
def C₃_condition (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 = 0

-- Theorem statement
theorem intersection_points :
  ∃ (t₁ t₂ s₁ s₂ θ₁ θ₂ θ₃ θ₄ : ℝ),
    C₁ t₁ = C₃ θ₁ ∧ C₃_condition (C₃ θ₁) ∧ C₁ t₁ = (1/2, 1) ∧
    C₁ t₂ = C₃ θ₂ ∧ C₃_condition (C₃ θ₂) ∧ C₁ t₂ = (1, 2) ∧
    C₂ s₁ = C₃ θ₃ ∧ C₃_condition (C₃ θ₃) ∧ C₂ s₁ = (-1/2, -1) ∧
    C₂ s₂ = C₃ θ₄ ∧ C₃_condition (C₃ θ₄) ∧ C₂ s₂ = (-1, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l860_86079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l860_86019

open Real

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def solution_set : Set ℝ :=
  {x | ∃ n : ℤ, 
    (2 * Real.pi * n < x ∧ x < Real.pi/6 + 2 * Real.pi * n) ∨
    (Real.pi/6 + 2 * Real.pi * n < x ∧ x < Real.pi/4 + 2 * Real.pi * n) ∨
    (Real.pi/4 + 2 * Real.pi * n < x ∧ x < Real.pi/3 + 2 * Real.pi * n)}

theorem equation_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ 
    (floor (sin (2 * x)) - 2 * floor (cos x) = 3 * floor (sin (3 * x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l860_86019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_k_theorem_l860_86004

/-- Represents the curve formed by arcs around an equilateral triangle -/
structure CurveK where
  a : ℝ  -- Side length of the equilateral triangle
  l : ℝ  -- Half the radius of the arcs forming the curve

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a rectangle circumscribes the curve -/
def circumscribes (r : Rectangle) (k : CurveK) : Prop :=
  sorry  -- Definition to be implemented

/-- Calculate the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

/-- Calculate the length of the curve -/
noncomputable def curve_length (k : CurveK) : ℝ :=
  3 * Real.pi * k.l

/-- Calculate the area enclosed by the curve -/
noncomputable def enclosed_area (k : CurveK) : ℝ :=
  k.l^2 * (Real.pi + 2 * Real.sqrt 3 - 6)

/-- Properties of the curve K -/
def curve_properties (k : CurveK) : Prop :=
  -- All rectangles circumscribed around K have the same perimeter
  (∀ r : Rectangle, circumscribes r k → perimeter r = 4 * k.l) ∧
  -- The length of curve K
  (curve_length k = 3 * Real.pi * k.l) ∧
  -- The area enclosed by curve K
  (enclosed_area k = k.l^2 * (Real.pi + 2 * Real.sqrt 3 - 6))

/-- Main theorem about the properties of curve K -/
theorem curve_k_theorem (k : CurveK) (h : k.a > 0) : curve_properties k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_k_theorem_l860_86004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_at_six_l860_86099

/-- A quartic polynomial satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  (∀ n : ℕ, n ∈ ({1, 2, 3, 4, 5} : Set ℕ) → p (n : ℝ) = 1 / ((n : ℝ)^3))

/-- Theorem stating that a special polynomial p satisfies p(6) = 0 -/
theorem special_polynomial_at_six (p : ℝ → ℝ) (h : special_polynomial p) : p 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_at_six_l860_86099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_removal_l860_86032

theorem parentheses_removal :
  (5 : ℤ) - 3 - (-1) + (-5) = 5 - 3 + 1 - 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_removal_l860_86032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_difference_l860_86023

noncomputable section

-- Define the points for lines p and q
def p1 : ℝ × ℝ := (0, 3)
def p2 : ℝ × ℝ := (4, 0)
def q1 : ℝ × ℝ := (0, 6)
def q2 : ℝ × ℝ := (6, 0)

-- Define the slopes of lines p and q
noncomputable def slope_p : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
noncomputable def slope_q : ℝ := (q2.2 - q1.2) / (q2.1 - q1.1)

-- Define the y-intercepts of lines p and q
noncomputable def intercept_p : ℝ := p1.2 - slope_p * p1.1
noncomputable def intercept_q : ℝ := q1.2 - slope_q * q1.1

-- Define functions for lines p and q
noncomputable def line_p (x : ℝ) : ℝ := slope_p * x + intercept_p
noncomputable def line_q (x : ℝ) : ℝ := slope_q * x + intercept_q

-- Define the x-coordinates when y = 10 for both lines
noncomputable def x_p_at_10 : ℝ := (10 - intercept_p) / slope_p
noncomputable def x_q_at_10 : ℝ := (10 - intercept_q) / slope_q

theorem line_intersection_difference :
  (abs (x_p_at_10 - x_q_at_10) = 40/3) ∧
  (abs (line_p 5 - line_q 5) = 7/4) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_difference_l860_86023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_relation_l860_86053

/-- The number of intersection points of k lines in a plane, where no two lines are parallel
    and no three lines intersect at the same point. -/
def f (k : ℕ) : ℕ := sorry

/-- Theorem: The relationship between f(k+1) and f(k) is f(k+1) = f(k) + k -/
theorem intersection_points_relation (k : ℕ) : f (k + 1) = f k + k := by
  sorry

#check intersection_points_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_relation_l860_86053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_off_pyramid_volume_l860_86012

/-- Right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ
  height : ℝ

/-- Smaller pyramid cut off from the top of a larger pyramid -/
noncomputable def cut_off_pyramid (p : RightSquarePyramid) (cut_height : ℝ) : RightSquarePyramid :=
  { base_edge := p.base_edge * (cut_height / p.height)
    slant_edge := p.slant_edge * (cut_height / p.height)
    height := cut_height }

/-- Volume of a right square pyramid -/
noncomputable def pyramid_volume (p : RightSquarePyramid) : ℝ :=
  (1 / 3) * p.base_edge^2 * p.height

/-- Main theorem: Volume of the cut-off pyramid is 32 cubic units -/
theorem cut_off_pyramid_volume :
  let large_pyramid : RightSquarePyramid :=
    { base_edge := 8 * Real.sqrt 2
      slant_edge := 10
      height := 6 }
  let cut_height := 3
  let small_pyramid := cut_off_pyramid large_pyramid cut_height
  pyramid_volume small_pyramid = 32 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_off_pyramid_volume_l860_86012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l860_86057

theorem power_equality (y : ℝ) (h : (128 : ℝ)^3 = (16 : ℝ)^y) : 
  (2 : ℝ)^(-y) = 1 / ((2 : ℝ)^(21/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l860_86057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l860_86020

noncomputable def original_equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 27) / x = 14

noncomputable def smallest_solution : ℝ := (3 - Real.sqrt 333) / 6

theorem smallest_solution_correct :
  original_equation smallest_solution ∧
  ∀ y, original_equation y → y ≥ smallest_solution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l860_86020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_bound_l860_86043

noncomputable def f (x : ℝ) := |2*x - 3| + 3

noncomputable def g (a : ℝ) (x : ℝ) := 2*x + a/x

def M := {y : ℝ | ∃ x, y = f x}

def N (a : ℝ) := {y : ℝ | ∃ x, y = g a x}

theorem subset_implies_a_bound (a : ℝ) (h : a > 0) :
  M ⊆ N a → 0 < a ∧ a ≤ 9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_bound_l860_86043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_half_angle_l860_86017

theorem cos_sum_half_angle (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α ^ 2 + Real.cos β ^ 2 = 1) : Real.cos ((α + β) / 2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_half_angle_l860_86017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_wire_length_l860_86040

/-- The length of a wire in meters, given its volume and diameter -/
noncomputable def wire_length (volume : ℝ) (diameter : ℝ) : ℝ :=
  let radius : ℝ := diameter / 2 / 10  -- Convert mm to cm and get radius
  let area : ℝ := Real.pi * radius^2
  let length_cm : ℝ := volume / area
  length_cm / 100  -- Convert cm to m

/-- Theorem stating the length of a silver wire given its volume and diameter -/
theorem silver_wire_length :
  let volume : ℝ := 44  -- 44 cubic centimeters
  let diameter : ℝ := 1  -- 1 mm
  ∃ (x : ℝ), abs (wire_length volume diameter - x) < 0.01 ∧ x = 56.05 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_wire_length_l860_86040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l860_86069

/-- Two circles touching internally with radii r₁ and r₂ -/
structure TouchingCircles (r₁ r₂ : ℝ) where
  (r₁_pos : 0 < r₁)
  (r₂_pos : 0 < r₂)
  (r₁_ge_r₂ : r₁ ≥ r₂)

/-- The length of the tangent from the center of the larger circle to the smaller circle -/
noncomputable def tangentLength (c : TouchingCircles r₁ r₂) : ℝ :=
  Real.sqrt ((r₁ - r₂)^2 - r₂^2)

theorem tangent_length_specific_case :
  ∀ (c : TouchingCircles 8 3), tangentLength c = 4 := by
  sorry

#check tangent_length_specific_case

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l860_86069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_words_left_l860_86027

/-- Represents the number of words Leo has left to write -/
def words_left_to_write (total_words : ℕ) (words_per_line : ℕ) (lines_per_page : ℕ) (pages_filled : ℚ) : ℕ :=
  total_words - (words_per_line * lines_per_page * (pages_filled.num / pages_filled.den)).toNat

/-- Theorem stating that Leo has 100 words left to write -/
theorem leo_words_left : words_left_to_write 400 10 20 (3/2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_words_left_l860_86027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_starting_points_exist_l860_86014

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Represents the route taken by the athlete -/
def athleteRoute (start : EarthPoint) : EarthPoint → Prop :=
  sorry -- Implementation details omitted for brevity

/-- The Earth's radius in kilometers -/
def earthRadius : Real := 6371

/-- The function to calculate distance between two points on Earth's surface -/
def distanceBetween (p1 p2 : EarthPoint) : Real :=
  sorry -- Implementation details omitted for brevity

/-- Theorem: There exist two distinct points on Earth where the athlete's route returns to the start -/
theorem two_starting_points_exist : 
  ∃ (p1 p2 : EarthPoint), p1 ≠ p2 ∧ 
    athleteRoute p1 p1 ∧ 
    athleteRoute p2 p2 ∧
    distanceBetween p1 (EarthPoint.mk p1.latitude (p1.longitude + 5 / earthRadius)) = 5 ∧
    distanceBetween p2 (EarthPoint.mk p2.latitude (p2.longitude + 5 / earthRadius)) = 5 :=
by
  sorry -- Proof omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_starting_points_exist_l860_86014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l860_86048

noncomputable def f (x : ℝ) : ℝ := x / (2^x - 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem h_is_even : ∀ x : ℝ, h x = h (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l860_86048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l860_86059

theorem work_completion_time 
  (x_total_days : ℝ) 
  (x_worked_days : ℝ) 
  (y_remaining_days : ℝ) 
  (h1 : x_total_days = 40) 
  (h2 : x_worked_days = 8) 
  (h3 : y_remaining_days = 24) : 
  let x_work_rate : ℝ := 1 / x_total_days
  let x_work_done : ℝ := x_work_rate * x_worked_days
  let remaining_work : ℝ := 1 - x_work_done
  let y_work_rate : ℝ := remaining_work / y_remaining_days
  let y_total_days : ℝ := 1 / y_work_rate
  y_total_days = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l860_86059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l860_86090

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/3) = -1/2) 
  (h2 : α > 2*π/3 ∧ α < π) : 
  Real.sin α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l860_86090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digit_differences_equals_495_l860_86078

/-- Function to get the first digit of a natural number -/
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

/-- Function to get the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ :=
  n % 10

/-- The difference between the first and last digit of a natural number -/
def digitDifference (n : ℕ) : ℤ :=
  (firstDigit n : ℤ) - (lastDigit n : ℤ)

/-- The sum of digit differences for all numbers from 1 to 999 -/
def sumOfDigitDifferences : ℤ :=
  (List.range 999).map digitDifference |>.sum

theorem sum_of_digit_differences_equals_495 :
  sumOfDigitDifferences = 495 := by
  sorry

#eval sumOfDigitDifferences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digit_differences_equals_495_l860_86078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_equal_areas_l860_86067

/-- The curve function -/
def f (x : ℝ) : ℝ := -x^4 + 2*x^2

/-- The line function -/
def g (k : ℝ) : ℝ → ℝ := λ x => k

/-- The intersection points of f and g -/
def intersection_points (k : ℝ) : Set ℝ :=
  {x : ℝ | f x = g k x}

/-- The area between f and g on an interval -/
noncomputable def area_between (k : ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (f x - g k x)

/-- The three areas are equal when k satisfies this condition -/
def areas_equal (k : ℝ) : Prop :=
  ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    {a, b, c, d} ⊆ intersection_points k ∧
    area_between k a b = area_between k b c ∧
    area_between k b c = area_between k c d

/-- The main theorem -/
theorem unique_k_for_equal_areas :
  ∃! (k : ℝ), k ≥ 0 ∧ areas_equal k ∧ k = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_equal_areas_l860_86067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_focus_l860_86061

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the focus
noncomputable def focus : ℝ × ℝ := (-2*Real.sqrt 3, 0)

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem chord_through_focus (A B : PointOnEllipse) :
  distance (A.x, A.y) focus = 2 →
  distance (B.x, B.y) focus = 14/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_focus_l860_86061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l860_86051

-- Define the propositions
def prop1 (a b c : ℝ) : Prop :=
  (a = b ↔ a * c = b * c)

def prop2 (a : ℝ) : Prop :=
  (Irrational (a + 5) ↔ Irrational a)

def prop3 (a b : ℝ) : Prop :=
  (a > b → a ^ 2 > b ^ 2)

def prop4 (a : ℝ) : Prop :=
  (a < 3 → a < 4)

-- Theorem statement
theorem exactly_two_props_true :
  ∃! (n : ℕ), n = 2 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ i, i ∈ S → i ∈ ({1, 2, 3, 4} : Finset ℕ)) ∧
    (1 ∈ S ↔ ∀ a b c, prop1 a b c) ∧
    (2 ∈ S ↔ ∀ a, prop2 a) ∧
    (3 ∈ S ↔ ∀ a b, prop3 a b) ∧
    (4 ∈ S ↔ ∀ a, prop4 a)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l860_86051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_sheep_count_l860_86081

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (horse_food_per_day total_food : ℝ),
    (sheep : ℝ) / (horses : ℝ) = 1 / 7 →
    horse_food_per_day = 230 →
    total_food = 12880 →
    (horses : ℝ) * horse_food_per_day = total_food →
    sheep = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_sheep_count_l860_86081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_transfer_height_calculation_l860_86086

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of oil transferred from one cylinder to another -/
noncomputable def oilTransferVolume (source : Cylinder) (levelDrop : ℝ) : ℝ :=
  Real.pi * source.radius^2 * levelDrop

/-- Calculates the height of oil in a cylinder given its volume -/
noncomputable def oilHeight (cylinder : Cylinder) (volume : ℝ) : ℝ :=
  volume / (Real.pi * cylinder.radius^2)

theorem oil_transfer_height_calculation 
  (stationaryTank : Cylinder) 
  (oilTruck : Cylinder) 
  (levelDrop : ℝ) : 
  stationaryTank.radius = 100 →
  oilTruck.radius = 5 →
  levelDrop = 0.025 →
  oilHeight oilTruck (oilTransferVolume stationaryTank levelDrop) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_transfer_height_calculation_l860_86086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_meeting_time_l860_86001

/-- Represents a car traveling between two points -/
structure Car where
  speed : ℝ

/-- Represents the system of two cars traveling between two points -/
structure TwoCarSystem where
  carA : Car
  carB : Car
  distanceAB : ℝ

/-- The time it takes for the cars to meet for the nth time -/
noncomputable def meetingTime (system : TwoCarSystem) (n : ℕ) : ℝ :=
  sorry

theorem fifteenth_meeting_time 
  (system : TwoCarSystem)
  (h1 : system.distanceAB / system.carA.speed - system.distanceAB / system.carB.speed = 3)
  (h2 : system.distanceAB / system.carA.speed = 6)
  (h3 : system.distanceAB / system.carB.speed = 3) :
  meetingTime system 15 = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_meeting_time_l860_86001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_additional_visitors_for_distinct_digits_l860_86094

def is_distinct_digits (n : Nat) : Bool :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.eraseDups.length = digits.length

theorem min_additional_visitors_for_distinct_digits :
  let initial_visitors : Nat := 1879564
  let next_distinct_visitors : Nat := initial_visitors + 38
  (is_distinct_digits initial_visitors = true) →
  (is_distinct_digits next_distinct_visitors = true) →
  ∀ k : Nat, k ∈ Finset.range 38 →
    ¬(is_distinct_digits (initial_visitors + k) = true) :=
by sorry

#eval is_distinct_digits 1879564  -- Should return true
#eval is_distinct_digits 1879602  -- Should return true
#eval is_distinct_digits 1879565  -- Should return false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_additional_visitors_for_distinct_digits_l860_86094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l860_86085

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^4 - 3*(X^2) + 2 = (X - 3)^2 * q + (56*X - 78) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l860_86085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l860_86089

noncomputable def f (x : ℝ) : ℝ := Real.sin (4 * x + Real.pi / 6) + 2

theorem function_satisfies_conditions :
  (∀ x, f x ≤ 3) ∧
  (∀ x, f x ≥ 1) ∧
  (∃ x, f x = 3) ∧
  (∃ x, f x = 1) ∧
  (∀ x, f (x + Real.pi / 2) = f x) ∧
  (∀ x, f (Real.pi / 3 + x) = f (Real.pi / 3 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l860_86089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l860_86075

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem point_on_parabola (x y : ℝ) :
  is_on_parabola x y →
  distance (x, y) focus = 9 →
  (x = 7 ∧ (y = 2 * Real.sqrt 14 ∨ y = -2 * Real.sqrt 14)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l860_86075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_is_half_x_plus_y_is_one_l860_86046

/-- A pyramid with a square base and specific properties -/
structure SquarePyramid where
  -- Side length of the square base
  s : ℝ
  -- The pyramid has a square base
  square_base : Prop
  -- Edges PR, PQ, PS, and PT are congruent
  congruent_edges : Prop
  -- Angle QRT is 45 degrees
  angle_qrt_45 : Prop

/-- The cosine of the dihedral angle in a square pyramid with the given properties -/
noncomputable def dihedral_angle_cosine (p : SquarePyramid) : ℝ := 1/2

/-- Theorem stating that the cosine of the dihedral angle is 1/2 -/
theorem dihedral_angle_cosine_is_half (p : SquarePyramid) :
  dihedral_angle_cosine p = 1/2 := by sorry

/-- The sum of x and y, where cos φ = x + √y -/
def x_plus_y : ℕ := 1

/-- Theorem stating that x + y = 1 -/
theorem x_plus_y_is_one :
  x_plus_y = 1 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_is_half_x_plus_y_is_one_l860_86046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l860_86060

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l860_86060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_area_l860_86050

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 3 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  let c := Real.sqrt (a^2 + 3)
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define the circle
def circleEq (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Define the asymptotes
def asymptotes (x y : ℝ) (a : ℝ) : Prop :=
  y = (Real.sqrt 3 / a) * x ∨ y = -(Real.sqrt 3 / a) * x

-- Define the tangency condition
def is_tangent (circle_center : ℝ × ℝ) (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  circleEq circle_center a A ∧ circleEq circle_center a B ∧
  asymptotes A.1 A.2 a ∧ asymptotes B.1 B.2 a

-- The main theorem
theorem hyperbola_circle_area (a : ℝ) (F₁ F₂ : ℝ × ℝ) (A B : ℝ × ℝ) :
  hyperbola A.1 A.2 a →
  foci F₁ F₂ a →
  is_tangent F₁ a A B →
  (F₁.1 - F₂.1) * (A.2 - F₁.2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_area_l860_86050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workday_meetings_percentage_l860_86068

/-- Represents the duration of a workday in hours -/
noncomputable def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in hours -/
noncomputable def first_meeting_hours : ℝ := 1

/-- Calculates the duration of the second meeting in hours -/
noncomputable def second_meeting_hours : ℝ := 3 * first_meeting_hours

/-- Calculates the total meeting time in hours -/
noncomputable def total_meeting_hours : ℝ := first_meeting_hours + second_meeting_hours

/-- Represents the percentage of workday spent in meetings -/
noncomputable def meeting_percentage : ℝ := (total_meeting_hours / workday_hours) * 100

theorem workday_meetings_percentage :
  meeting_percentage = 40 := by
  -- Expand the definitions
  unfold meeting_percentage total_meeting_hours second_meeting_hours first_meeting_hours workday_hours
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workday_meetings_percentage_l860_86068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strange_animal_farm_l860_86005

theorem strange_animal_farm (cats dogs : ℕ) : 
  dogs = cats + 180 →
  (0.2 * (dogs : ℝ) + 0.8 * (cats : ℝ)) / ((dogs : ℝ) + (cats : ℝ)) = 0.32 →
  dogs = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strange_animal_farm_l860_86005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_satisfies_equation_P_satisfies_conditions_l860_86029

/-- The point P that satisfies the equidistant condition -/
def P : ℝ × ℝ × ℝ := (9, 1, -13)

/-- The given point from which P is equidistant -/
def A : ℝ × ℝ × ℝ := (3, -2, 4)

/-- Function to calculate the square of the distance between two points -/
def distanceSquared (p q : ℝ × ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2

/-- The equation that all equidistant points satisfy -/
def equidistantEquation (x y z : ℝ) : Prop :=
  12 * x + 6 * y - 18 * z = 36

theorem equidistant_point_satisfies_equation : 
  ∀ x y z : ℝ, 
  distanceSquared (x, y, z) A = distanceSquared (x, y, z) P → 
  equidistantEquation x y z :=
by
  sorry

theorem P_satisfies_conditions : 
  ∀ x y z : ℝ, 
  distanceSquared (x, y, z) A = distanceSquared (x, y, z) P ↔ 
  equidistantEquation x y z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_satisfies_equation_P_satisfies_conditions_l860_86029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_count_l860_86077

-- Define the constants from the problem
noncomputable def brenda_time : ℝ := 8
noncomputable def brandon_time : ℝ := 12
noncomputable def output_reduction : ℝ := 12
noncomputable def combined_time : ℝ := 4.5

-- Define the function to calculate the number of bricks
noncomputable def calculate_bricks : ℝ :=
  (combined_time * ((1 / brenda_time + 1 / brandon_time) * combined_time - output_reduction))

-- Theorem statement
theorem wall_bricks_count : calculate_bricks = 864 := by
  -- Unfold the definition of calculate_bricks
  unfold calculate_bricks
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_count_l860_86077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_cut_l860_86042

theorem square_diagonal_cut (s : ℝ) (h : s = 10) :
  let triangle_sides := [s, s, s]
  (∀ side, side ∈ triangle_sides → side = 10) :=
by
  intro triangle_sides
  intro side side_in_triangle
  simp [triangle_sides, h] at *
  exact side_in_triangle

#check square_diagonal_cut

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_cut_l860_86042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colored_cells_100x100_l860_86096

/-- Represents a square table with side length n -/
structure SquareTable (n : ℕ) where
  cells : Fin n → Fin n → Bool

/-- A coloring is valid if each colored cell is the only colored cell in its row or column -/
def is_valid_coloring (n : ℕ) (table : SquareTable n) : Prop :=
  ∀ i j : Fin n, table.cells i j → 
    (∀ k : Fin n, k ≠ j → ¬table.cells i k) ∨ 
    (∀ k : Fin n, k ≠ i → ¬table.cells k j)

/-- The number of colored cells in a table -/
def colored_cell_count (n : ℕ) (table : SquareTable n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin n)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin n)) fun j =>
      if table.cells i j then 1 else 0

/-- The main theorem -/
theorem max_colored_cells_100x100 : 
  ∀ (table : SquareTable 100), is_valid_coloring 100 table → 
    colored_cell_count 100 table ≤ 198 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colored_cells_100x100_l860_86096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_half_cistern_time_l860_86041

/-- Represents the time in minutes to fill a portion of a cistern -/
def fill_time (portion : ℚ) : ℕ := sorry

/-- The time to fill half of the cistern -/
def half_fill_time : ℕ := fill_time (1/2)

/-- Given condition: A fill pipe can fill 1/2 of the cistern in 10 minutes -/
axiom fill_half_in_ten : half_fill_time = 10

/-- Theorem: The time required to fill 1/2 of the cistern is 10 minutes -/
theorem fill_half_cistern_time : half_fill_time = 10 := by
  exact fill_half_in_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_half_cistern_time_l860_86041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l860_86002

noncomputable section

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (0, 3)
noncomputable def b : ℝ × ℝ := (Real.sqrt 3, 1)

/-- Vector operations -/
noncomputable def c : ℝ × ℝ := (3 * a.1 + 5 * b.1, 3 * a.2 + 5 * b.2)
noncomputable def d (m : ℝ) : ℝ × ℝ := (m * a.1 - 5 * b.1, m * a.2 - 5 * b.2)

/-- Parallel vectors have proportional components -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Perpendicular vectors have zero dot product -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem vector_relations :
  (parallel c (d (-3))) ∧ (perpendicular c (d (145/42))) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l860_86002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kamils_renovation_cost_l860_86006

/-- The cost of hiring professionals for kitchen renovation --/
def kitchen_renovation_cost 
  (num_professionals : ℕ) 
  (hours_per_day : ℕ) 
  (num_days : ℕ) 
  (hourly_rate : ℕ) : ℕ :=
  num_professionals * hours_per_day * num_days * hourly_rate

/-- Proof of Kamil's kitchen renovation cost --/
theorem kamils_renovation_cost : 
  kitchen_renovation_cost 2 6 7 15 = 1260 := by
  -- Unfold the definition of kitchen_renovation_cost
  unfold kitchen_renovation_cost
  -- Perform the arithmetic
  norm_num

-- Evaluate the function to check the result
#eval kitchen_renovation_cost 2 6 7 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kamils_renovation_cost_l860_86006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_sets_distribution_X_expected_value_X_expected_value_X_is_three_fourths_l860_86026

-- Define the number of housing sets and applicants
def num_housing_sets : ℕ := 4
def num_applicants : ℕ := 3

-- Define the probability of choosing a specific set
def prob_choose_set : ℚ := 1 / num_housing_sets

-- Define the random variable X (we'll use ℕ → ℝ for simplicity)
def X : ℕ → ℝ := λ k => if k ≤ num_applicants then (num_applicants.choose k : ℝ) * (prob_choose_set ^ k : ℝ) * ((1 - prob_choose_set) ^ (num_applicants - k) : ℝ) else 0

-- Theorem 1: Probability that two specific applicants do not choose the same set
theorem prob_different_sets : ℚ :=
  3 / 4

-- Theorem 2: Distribution of X
theorem distribution_X (k : ℕ) : ℝ :=
  X k

-- Theorem 3: Expected value of X
theorem expected_value_X : ℝ :=
  (num_applicants : ℝ) * (prob_choose_set : ℝ)

-- Corollary: The expected value of X is 3/4
theorem expected_value_X_is_three_fourths : ℝ :=
  3 / 4

-- Proofs
example : prob_different_sets = 3 / 4 := by sorry
example : expected_value_X = 3 / 4 := by sorry
example : expected_value_X_is_three_fourths = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_sets_distribution_X_expected_value_X_expected_value_X_is_three_fourths_l860_86026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_polygon_correct_l860_86074

-- Define the discrete random variable X
def X : Finset ℚ := {1, 3, 6, 8}

-- Define the probability function
def p : ℚ → ℚ
| 1 => 1/5
| 3 => 1/10
| 6 => 2/5
| 8 => 3/10
| _ => 0

-- Define the distribution polygon
def distributionPolygon : Set (ℚ × ℚ) :=
  {(x, p x) | x ∈ X}

-- Theorem statement
theorem distribution_polygon_correct :
  distributionPolygon = {(1, 1/5), (3, 1/10), (6, 2/5), (8, 3/10)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_polygon_correct_l860_86074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_equals_one_l860_86087

/-- Point type representing points on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Arc length between two points on a circle -/
noncomputable def arcLength (center : Point) (r : ℝ) (p q : Point) : ℝ :=
  sorry  -- Definition would depend on circle's center and radius

/-- Given a circle with radius r and points A, B, C on the circle,
    prove that AB/BC = 1 under specific conditions. -/
theorem chord_ratio_equals_one (r : ℝ) (center A B C : Point) : 
  r > 0 →  -- circle has positive radius
  A ≠ B → B ≠ C → A ≠ C →  -- A, B, C are distinct points
  dist A B = dist B C →  -- AB = BC
  dist A B > r →  -- AB > r
  arcLength center r B C = π * r / 2 →  -- Length of minor arc BC is πr/2
  dist A B / dist B C = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_equals_one_l860_86087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l860_86095

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (2 * x))

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x) ∧
  (∃ x : ℝ, f x = 0) ∧
  (∃ x : ℝ, ∀ y : ℝ, |y - x| < 1 → f y ≤ f x ∨ ∀ y : ℝ, |y - x| < 1 → f y ≥ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l860_86095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l860_86013

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangle PQRS in 3D space -/
structure Rectangle3D where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D

/-- Represents a pyramid VPQRS in 3D space -/
structure Pyramid3D where
  V : Point3D
  base : Rectangle3D

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Check if a vector is perpendicular to a plane defined by three points -/
def isPerpendicular (v : Point3D) (p1 p2 p3 : Point3D) : Prop :=
  sorry

/-- Calculate the volume of a pyramid -/
noncomputable def pyramidVolume (p : Pyramid3D) : ℝ :=
  sorry

theorem pyramid_volume_theorem (p : Pyramid3D) :
  isPerpendicular p.V p.base.P p.base.Q p.base.R →
  (∃ n : ℕ, distance p.V p.base.P = n) →
  (∃ k : ℕ, distance p.V p.base.Q = 2*k ∧
            distance p.V p.base.R = 2*k + 2 ∧
            distance p.V p.base.S = 2*k + 4) →
  pyramidVolume p = 80/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l860_86013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l860_86071

theorem marble_distribution (total_marbles : ℕ) (initial_people : ℕ) (join_people : ℕ) : 
  total_marbles = 220 →
  initial_people = 20 →
  (total_marbles / initial_people : ℕ) - (total_marbles / (initial_people + join_people) : ℕ) = 1 →
  join_people = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l860_86071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_between_zero_and_negative_one_l860_86063

theorem number_between_zero_and_negative_one : ∃! x : ℚ, 
  (x = -3/2 ∨ x = -1/3 ∨ x = 1/4 ∨ x = 5/3) ∧ 
  (-1 < x) ∧ 
  (x < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_between_zero_and_negative_one_l860_86063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_wins_probability_l860_86037

/-- The number of matches played by the team -/
def num_matches : ℕ := 8

/-- The probability of winning a single match -/
def win_prob : ℚ := 1/3

/-- The probability of losing a single match -/
def lose_prob : ℚ := 1/3

/-- The probability of tying a single match -/
def tie_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

theorem more_wins_probability :
  ∀ (n : ℕ) (p_win p_lose p_tie : ℚ),
  n = num_matches →
  p_win = win_prob →
  p_lose = lose_prob →
  p_tie = tie_prob →
  p_win + p_lose + p_tie = 1 →
  more_wins_prob = (Finset.sum (Finset.range (n + 1)) (λ i ↦
    Finset.sum (Finset.range (i + 1)) (λ j ↦
      (n.choose i) * ((n - i).choose j) * p_win^i * p_lose^j * p_tie^(n - i - j))))
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_wins_probability_l860_86037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_55_l860_86097

def a : ℕ → ℤ
  | 0 => 0  -- Add a case for 0
  | n+1 => if n ≥ 20 then 50 * a n + 2 * (n+1) else 0

theorem least_multiple_of_55 :
  (∀ k : ℕ, k > 20 ∧ k < 45 → ¬(55 ∣ a k)) ∧ (55 ∣ a 45) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_55_l860_86097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_in_second_quadrant_l860_86073

open Real

-- Define the concept of quadrants
def in_third_quadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3*π/2
def in_second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

-- State the theorem
theorem half_angle_in_second_quadrant (θ : ℝ) : 
  in_third_quadrant θ → |cos (θ/2)| = -cos (θ/2) → in_second_quadrant (θ/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_in_second_quadrant_l860_86073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_scalar_multiplication_l860_86083

/-- Given two 2D vectors and a scalar, prove that the result of the vector subtraction
    and scalar multiplication is equal to the expected result. -/
theorem vector_subtraction_scalar_multiplication :
  let v1 : Fin 2 → ℤ := ![3, -8]
  let v2 : Fin 2 → ℤ := ![2, 6]
  let scalar : ℤ := 3
  let result : Fin 2 → ℤ := ![-3, -26]
  v1 - scalar • v2 = result := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_scalar_multiplication_l860_86083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l860_86055

noncomputable def f (x : ℝ) := x^2 - Real.cos x

theorem f_range_theorem (x₀ : ℝ) :
  x₀ ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →
  f x₀ > f (Real.pi/6) ↔
  x₀ ∈ Set.Ioo (-Real.pi/2) (-Real.pi/6) ∪ Set.Ioo (Real.pi/6) (Real.pi/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l860_86055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_sculpture_volume_l860_86009

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of a cylinder
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Theorem statement
theorem snow_sculpture_volume :
  let sphere1 := sphere_volume 4
  let sphere2 := sphere_volume 6
  let sphere3 := sphere_volume 8
  let cylinder := cylinder_volume 5 10
  sphere1 + sphere2 + sphere3 + cylinder = 1250 * Real.pi :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_sculpture_volume_l860_86009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l860_86030

open Real

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (x/3))^3 - (Real.sin (x/3))^2 * (Real.cos (x/3)) - 3 * (Real.sin (x/3)) * (Real.cos (x/3))^2 + 3 * (Real.cos (x/3))^3 = 0 ↔ 
  (∃ k : ℤ, x = (3*π/4)*(4*k + 1)) ∨ (∃ n : ℤ, x = π*(3*n + 1) ∨ x = π*(3*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l860_86030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_probability_l860_86024

/-- The number of revolutions per minute made by the searchlight -/
noncomputable def revolutions_per_minute : ℝ := 3

/-- The minimum time (in seconds) that the man needs to stay in the dark -/
noncomputable def min_dark_time : ℝ := 10

/-- Calculates the time (in seconds) for one full revolution of the searchlight -/
noncomputable def revolution_time : ℝ := 60 / revolutions_per_minute

/-- Theorem: The probability of a man staying in the dark for at least 10 seconds
    when a searchlight makes 3 revolutions per minute is 1/2 -/
theorem dark_probability : 
  min_dark_time / revolution_time = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_probability_l860_86024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distortion_convexity_l860_86018

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon represented by its vertices -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a hexagon is regular with side length x -/
def isRegularHexagon (h : Hexagon) (x : ℝ) : Prop :=
  ∀ i : Fin 6, distance (h.vertices i) (h.vertices ((i + 1) % 6)) = x

/-- Check if a hexagon is a distortion of another hexagon -/
def isDistortion (h h' : Hexagon) : Prop :=
  ∀ i : Fin 6, distance (h.vertices i) (h'.vertices i) < 1

/-- Check if a hexagon is convex -/
def isConvex (h : Hexagon) : Prop := sorry

/-- Main theorem -/
theorem distortion_convexity (h : Hexagon) (x : ℝ) :
  isRegularHexagon h x → x ≥ 4 →
  ∀ h', isDistortion h h' → isConvex h' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distortion_convexity_l860_86018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_B_rate_l860_86082

-- Define the rates of envelope processing for each machine
noncomputable def rate_A : ℝ := 6000 / 3
noncomputable def rate_B_C : ℝ := 6000 / 2.5
noncomputable def rate_A_C : ℝ := 3000 / 1

-- Theorem statement
theorem machine_B_rate : 
  ∃ (rate_B rate_C : ℝ),
    rate_B + rate_C = rate_B_C ∧
    rate_A + rate_C = rate_A_C ∧
    rate_B = 1400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_B_rate_l860_86082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_die_prob_F_l860_86044

/-- A biased 12-sided die with special properties -/
structure BiasedDie where
  faces : Fin 12 → ℕ
  opposite_sum : ∀ i : Fin 12, faces i + faces (11 - i) = 13
  prob_F : ℚ
  prob_opposite_F : ℚ
  prob_other : ℚ
  prob_F_gt : prob_F > 1/12
  prob_opposite_F_lt : prob_opposite_F < 1/12
  prob_other_eq : prob_other = 1/12
  prob_sum : prob_F + prob_opposite_F + 10 * prob_other = 1

/-- The probability of obtaining a sum of 13 when rolling two such dice -/
noncomputable def prob_sum_13 (d : BiasedDie) : ℚ := 29/384

/-- The theorem to prove -/
theorem biased_die_prob_F (d : BiasedDie) :
  prob_sum_13 d = 29/384 → d.prob_F = 7/48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_die_prob_F_l860_86044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_pure_imaginary_Z_in_first_quadrant_l860_86031

-- Define the complex number Z as a function of m
noncomputable def Z (m : ℝ) : ℂ := Complex.log (m^2 - 2*m - 2 : ℂ) + Complex.I * (m^2 + 3*m + 2)

-- Theorem 1: Z is real iff m = -1 or m = -2
theorem Z_is_real (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

-- Theorem 2: Z is pure imaginary iff m = 3 or m = -1
theorem Z_is_pure_imaginary (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 ∨ m = -1 := by sorry

-- Theorem 3: Z is in the first quadrant iff m < -2 or m > 3
theorem Z_in_first_quadrant (m : ℝ) : (Z m).re > 0 ∧ (Z m).im > 0 ↔ m < -2 ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_pure_imaginary_Z_in_first_quadrant_l860_86031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_times_l860_86039

/-- Calculates the time (in seconds) for a train to cross a bridge. -/
noncomputable def timeToCrossBridge (trainLength : ℝ) (bridgeLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  (trainLength + bridgeLength) * 18 / (trainSpeed * 5)

/-- Proves the time calculations for a train crossing three different bridges. -/
theorem train_bridge_crossing_times 
  (trainLength : ℝ) (trainSpeed : ℝ) 
  (bridgeLengthA : ℝ) (bridgeLengthB : ℝ) (bridgeLengthC : ℝ) 
  (h1 : trainLength = 100) 
  (h2 : bridgeLengthA = 142) 
  (h3 : bridgeLengthB = 180) 
  (h4 : bridgeLengthC = 210) 
  (h5 : trainSpeed > 0) :
  (timeToCrossBridge trainLength bridgeLengthA trainSpeed = 4356 / (5 * trainSpeed)) ∧
  (timeToCrossBridge trainLength bridgeLengthB trainSpeed = 5040 / (5 * trainSpeed)) ∧
  (timeToCrossBridge trainLength bridgeLengthC trainSpeed = 5580 / (5 * trainSpeed)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_times_l860_86039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_theorem_l860_86038

-- Define the fixed circle B
noncomputable def circle_B (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*(Real.sqrt 5)*x - 31 = 0

-- Define the fixed point A
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the range of x + 2y
def range_sum (x y : ℝ) : Prop := -5 ≤ x + 2*y ∧ x + 2*y ≤ 5

-- Theorem statement
theorem circle_trajectory_theorem :
  ∀ (x y : ℝ),
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
      (circle_B x' y' ∨ (x' = point_A.1 ∧ y' = point_A.2)))) →
  trajectory_E x y ∧ range_sum x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_theorem_l860_86038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l860_86052

/-- The point on the line y = 4x + 3 that is closest to (2, -1) -/
noncomputable def closest_point : ℝ × ℝ := (-14/17, 5/17)

/-- The line equation y = 4x + 3 -/
def line_equation (x : ℝ) : ℝ := 4*x + 3

/-- The point we're finding the closest point to -/
def given_point : ℝ × ℝ := (2, -1)

/-- Theorem stating that closest_point is on the line and is the closest to given_point -/
theorem closest_point_is_correct : 
  (line_equation closest_point.fst = closest_point.snd) ∧ 
  (∀ p : ℝ × ℝ, line_equation p.fst = p.snd → 
    (closest_point.fst - given_point.fst)^2 + (closest_point.snd - given_point.snd)^2 ≤ 
    (p.fst - given_point.fst)^2 + (p.snd - given_point.snd)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l860_86052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_angle_theorem_l860_86070

/-- The angle between a lateral edge and the base plane of a regular n-sided pyramid -/
noncomputable def lateral_edge_angle (n : ℕ) (t : ℝ) : ℝ :=
  Real.arctan (Real.cos (Real.pi / n) * Real.sqrt (t^2 - 2*t))

/-- The total surface area of a regular n-sided pyramid -/
noncomputable def total_surface_area (n : ℕ) (β : ℝ) : ℝ :=
  sorry

/-- The base area of a regular n-sided pyramid -/
noncomputable def base_area (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The angle between a lateral edge and the base plane of a regular n-sided pyramid -/
theorem lateral_edge_angle_theorem (n : ℕ) (t : ℝ) (h1 : n ≥ 3) (h2 : t > 2) :
  let β := lateral_edge_angle n t
  let S := total_surface_area n β
  let S_base := base_area n
  S / S_base = t →
  β = Real.arctan (Real.cos (Real.pi / n) * Real.sqrt (t^2 - 2*t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_angle_theorem_l860_86070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l860_86021

/-- The perimeter of an equilateral triangle, given conditions about an isosceles triangle -/
theorem equilateral_triangle_perimeter 
  (equilateral_side : ℝ)
  (isosceles_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h1 : isosceles_perimeter = 70)
  (h2 : isosceles_base = 30)
  (h3 : isosceles_perimeter = 2 * equilateral_side + isosceles_base) :
  3 * equilateral_side = 60 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l860_86021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_six_arccos_one_fourth_l860_86054

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_six_arccos_one_fourth_l860_86054
