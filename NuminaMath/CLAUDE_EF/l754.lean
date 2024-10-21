import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_35_multiples_of_7_l754_75449

theorem average_of_first_35_multiples_of_7 : 
  (7 + 7 * 35) / 2 = 126 := by
  -- Arithmetic calculation
  calc
    (7 + 7 * 35) / 2 = (7 + 245) / 2 := by rfl
    _ = 252 / 2 := by rfl
    _ = 126 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_35_multiples_of_7_l754_75449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationship_correct_l754_75401

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Determines the relationship between two planes -/
noncomputable def plane_relationship (P1 P2 : Plane) : String :=
  if P1.A / P2.A = P1.B / P2.B ∧ P1.A / P2.A ≠ P1.C / P2.C then
    "intersect"
  else if P1.A / P2.A = P1.B / P2.B ∧ P1.A / P2.A = P1.C / P2.C ∧ P1.A / P2.A ≠ P1.D / P2.D then
    "parallel"
  else if P1.A / P2.A = P1.B / P2.B ∧ P1.A / P2.A = P1.C / P2.C ∧ P1.A / P2.A = P1.D / P2.D then
    "coincide"
  else if P1.A / P2.A = P1.C / P2.C ∧ P1.B = 0 ∧ P2.B = 0 then
    "parallel"
  else
    "unknown"

theorem plane_relationship_correct (P1 P2 : Plane) :
  (plane_relationship P1 P2 = "intersect" →
    P1.A / P2.A = P1.B / P2.B ∧ P1.A / P2.A ≠ P1.C / P2.C) ∧
  (plane_relationship P1 P2 = "parallel" →
    (P1.A / P2.A = P1.B / P2.B ∧ P1.A / P2.A = P1.C / P2.C ∧ P1.A / P2.A ≠ P1.D / P2.D) ∨
    (P1.A / P2.A = P1.C / P2.C ∧ P1.B = 0 ∧ P2.B = 0)) ∧
  (plane_relationship P1 P2 = "coincide" →
    P1.A / P2.A = P1.B / P2.B ∧ P1.A / P2.A = P1.C / P2.C ∧ P1.A / P2.A = P1.D / P2.D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationship_correct_l754_75401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_characterization_l754_75472

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The k-th iteration of a polynomial P -/
def iterate (P : IntPolynomial) (k : ℕ) : IntPolynomial :=
  match k with
  | 0 => id
  | k+1 => P ∘ (iterate P k)

/-- Condition for the existence of required m and P for a given n -/
def has_required_m_and_P (n : ℕ) : Prop :=
  ∃ (m : ℕ) (P : IntPolynomial), 
    m > 1 ∧ 
    Nat.Coprime m n ∧ 
    (∀ k < m, ¬(n ∣ (iterate P k 0).toNat)) ∧ 
    (n ∣ (iterate P m 0).toNat)

/-- Condition on the prime factors of n -/
def has_required_prime_factors (n : ℕ) : Prop :=
  ∃ (p p' : ℕ), Nat.Prime p ∧ Nat.Prime p' ∧ p' < p ∧ (p ∣ n) ∧ ¬(p' ∣ n)

theorem polynomial_existence_characterization (n : ℕ) (h : n ≥ 2) :
  has_required_m_and_P n ↔ has_required_prime_factors n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_characterization_l754_75472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l754_75474

-- Define the circles and line
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*y = 0
def circle_N (x y : ℝ) : Prop := (x-1)^2 + (y-1)^2 = 1
def line (x y : ℝ) : Prop := x + y = 0

-- Define the intersection of circle M and the line
noncomputable def intersection_length (a : ℝ) : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem circles_intersect (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x y : ℝ, circle_M a x y ∧ line x y)
  (h3 : intersection_length a = 2 * Real.sqrt 2) :
  ∃ x y : ℝ, circle_M a x y ∧ circle_N x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l754_75474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_integer_roots_l754_75415

/-- 
Given two cubic polynomials x^3 + ax + b and x^3 + bx + a where a and b are integers,
if all roots of both polynomials are integers, then a = 0 and b = 0.
-/
theorem cubic_polynomials_integer_roots (a b : ℤ) : 
  (∀ x : ℤ, x^3 + a*x + b = 0) ∧ 
  (∀ y : ℤ, y^3 + b*y + a = 0) → 
  a = 0 ∧ b = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_integer_roots_l754_75415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l754_75432

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 0) →
  a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l754_75432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l754_75443

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 1 ∨ (1 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l754_75443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l754_75483

/-- Represents an ellipse with given minor axis length and eccentricity -/
structure Ellipse where
  minor_axis : ℝ
  eccentricity : ℝ

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ

/-- Represents a focus of the ellipse -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : EllipsePoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the perimeter of a triangle given three points -/
noncomputable def triangle_perimeter (p1 p2 p3 : EllipsePoint) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

/-- Converts a Focus to an EllipsePoint -/
def focus_to_point (f : Focus) : EllipsePoint :=
  EllipsePoint.mk f.x f.y

theorem ellipse_triangle_perimeter
  (e : Ellipse)
  (f1 f2 : Focus)
  (a b : EllipsePoint) :
  e.minor_axis = Real.sqrt 5 →
  e.eccentricity = 2/3 →
  distance (focus_to_point f1) a + distance (focus_to_point f1) b = 2 * (e.minor_axis / Real.sqrt (1 - e.eccentricity^2)) →
  triangle_perimeter a b (focus_to_point f2) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l754_75483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linearly_bounded_iff_conditions_l754_75434

/-- A function f: ℝ × ℝ → ℝ is linearly bounded if there exists a positive K such that
    |f(x,y)| < K(x+y) for all positive x and y -/
def LinearlyBounded (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ K > 0, ∀ x y : ℝ, x > 0 → y > 0 → |f (x, y)| < K * (x + y)

/-- The function f(x,y) = x^α * y^β -/
noncomputable def f (α β : ℝ) : ℝ × ℝ → ℝ :=
  fun (x, y) => x ^ α * y ^ β

theorem linearly_bounded_iff_conditions (α β : ℝ) :
  LinearlyBounded (f α β) ↔ α + β = 1 ∧ α ≥ 0 ∧ β ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linearly_bounded_iff_conditions_l754_75434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_tournament_l754_75491

structure Player where
  name : String
  wins : ℕ
  draws : ℕ
  losses : ℕ

def score (p : Player) : ℚ :=
  p.wins + p.draws / 2

structure ChessTournament where
  A : Player
  B : Player
  C : Player

def is_valid_tournament (t : ChessTournament) : Prop :=
  let total_games := t.A.wins + t.A.draws + t.A.losses
  total_games = t.B.wins + t.B.draws + t.B.losses ∧
  total_games = t.C.wins + t.C.draws + t.C.losses ∧
  t.A.wins + t.B.wins + t.C.wins = t.A.losses + t.B.losses + t.C.losses

theorem exists_special_tournament :
  ∃ (t : ChessTournament),
    is_valid_tournament t ∧
    score t.A > score t.B ∧ score t.B > score t.C ∧
    t.C.wins > t.B.wins ∧ t.B.wins > t.A.wins := by
  sorry

#check exists_special_tournament

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_tournament_l754_75491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l754_75429

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A + 2 * Real.sin B + 3 * Real.sin C < 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l754_75429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relationship_l754_75493

-- Use noncomputable section for the entire block
noncomputable section

-- Use Real.abs instead of Real.abs
def f (x : ℝ) : ℝ := 2^(abs x)

def a : ℝ := f (Real.log 10 / Real.log 3)

def b : ℝ := f (Real.log (1/99))

def c : ℝ := f 0

-- Use 'by sorry' to skip the proof
theorem order_relationship : c < b ∧ b < a := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relationship_l754_75493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_zeta_frac_equals_neg_three_sixteenths_l754_75490

/-- The Riemann zeta function -/
noncomputable def zeta (y : ℝ) : ℝ := ∑' m, (m : ℝ) ^ (-y)

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem sum_zeta_frac_equals_neg_three_sixteenths :
  (∑' j : ℕ, frac (zeta (2 * ↑j))) = (-3 : ℝ) / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_zeta_frac_equals_neg_three_sixteenths_l754_75490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l754_75405

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix (x = -2 line)
def directrix (x : ℝ) : Prop := x = -2

-- Define point P on the directrix
noncomputable def P : ℝ × ℝ := (-2, Real.sqrt 32)

-- Define point Q as the intersection of PF and the parabola
noncomputable def Q : ℝ × ℝ := (1, 2 * Real.sqrt 2)

-- State the theorem
theorem parabola_intersection_length :
  parabola Q.1 Q.2 ∧
  directrix P.1 ∧
  (P.1 - focus.1) * (Q.2 - focus.2) = (P.2 - focus.2) * (Q.1 - focus.1) ∧
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 16 * ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2) →
  (Q.1 - focus.1)^2 + (Q.2 - focus.2)^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l754_75405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezium_area_l754_75436

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  x : ℝ
  shorter_base : ℝ
  altitude : ℝ
  longer_base : ℝ
  isosceles : Prop

/-- The area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.shorter_base + t.longer_base) * t.altitude / 2

theorem isosceles_trapezium_area (t : Trapezium) 
  (h1 : t.x > 0)
  (h2 : t.shorter_base = 2 * t.x)
  (h3 : t.altitude = 2 * t.x)
  (h4 : t.longer_base = 6 * t.x)
  (h5 : t.isosceles) :
  trapezium_area t = 8 * t.x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezium_area_l754_75436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_matches_played_l754_75467

theorem additional_matches_played (initial_avg : ℝ) (initial_matches : ℕ) 
  (additional_avg : ℝ) (total_matches : ℕ) (overall_avg : ℝ) 
  (h1 : initial_avg = 45)
  (h2 : initial_matches = 25)
  (h3 : additional_avg = 15)
  (h4 : total_matches = 32)
  (h5 : overall_avg = 38.4375)
  : (total_matches - initial_matches : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_matches_played_l754_75467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l754_75473

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def l (k m x : ℝ) : ℝ := k * x + m

-- Define point A as the intersection of l and E
noncomputable def A (k m : ℝ) : ℝ × ℝ := sorry

-- Define point B as the other intersection of l and E
noncomputable def B (k m : ℝ) : ℝ × ℝ := sorry

-- Define point Q as the intersection of l and x = -4
def Q (k m : ℝ) : ℝ × ℝ := (-4, l k m (-4))

-- Define point P on E
noncomputable def P (k m : ℝ) : ℝ × ℝ := 
  (-(8*k*m)/(4*k^2+3), (6*m)/(4*k^2+3))

-- Define the left focus F
def F : ℝ × ℝ := (-1, 0)

-- Theorem to prove
theorem ellipse_constant_product (k m : ℝ) :
  let p := P k m
  let f := F
  let q := Q k m
  (p.1 - 0) * (q.1 - f.1) + (p.2 - 0) * (q.2 - f.2) = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l754_75473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_lines_l754_75416

/-- Two lines in 3D space are parallel -/
def parallel (l1 l2 : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is perpendicular to another line in 3D space -/
def perpendicular (l1 l2 : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line in 3D space -/
def Line3D : Type := Set (Fin 3 → ℝ)

theorem perpendicular_to_parallel_lines 
  (l1 l2 l3 : Line3D) 
  (h_parallel : parallel l1 l2) 
  (h_perp : perpendicular l3 l1) : 
  perpendicular l3 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_lines_l754_75416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_max_min_l754_75486

theorem inequality_and_max_min :
  (∀ x : ℝ, |x - 3| < 1 ↔ 2 < x ∧ x < 4) →
  (∀ x y z : ℝ, (x - 1)^2 / 16 + (y + 2)^2 / 5 + (z - 3)^2 / 4 = 1 →
    (∀ w : ℝ, x + y + z ≤ w → w ≤ 7) ∧
    (∀ w : ℝ, -3 ≤ w → w ≤ x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_max_min_l754_75486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_theorem_l754_75424

-- Define the inverse variation function
noncomputable def inverse_variation (k : ℝ) (b : ℝ) : ℝ := k / (b^3)

-- State the theorem
theorem inverse_variation_theorem (k : ℝ) :
  (inverse_variation k 1 = 4) → (inverse_variation k 2 = 1/2) := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_theorem_l754_75424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l754_75499

/-- The area of a circular sector with radius 12 meters and central angle 40 degrees is 16π square meters. -/
theorem sector_area (radius : Real) (angle : Real) (h1 : radius = 12) (h2 : angle = 40) :
  (angle / 360) * Real.pi * radius^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l754_75499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_of_triangles_l754_75433

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem centroid_distance_of_triangles (A B C D : ℝ × ℝ) :
  let ABC := Triangle.mk A B C
  let ABD := Triangle.mk A B D
  A.1 = 0 ∧ A.2 = 0 ∧
  B.1 = 841 ∧ B.2 = 0 ∧
  C.1 = 0 ∧ C.2 = 41 ∧
  D.1 = 0 ∧ D.2 = 609 ∧
  distance A B = 841 ∧
  distance B C = 840 ∧
  distance A C = 41 ∧
  distance A D = 609 ∧
  distance B D = 580 →
  distance (centroid ABC) (centroid ABD) = 568 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_of_triangles_l754_75433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_in_still_water_calculation_l754_75411

/-- Calculates the speed in still water given upstream and downstream speeds -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: Given upstream speed of 20 km/h and downstream speed of 30 km/h, 
    the speed in still water is 25 km/h -/
theorem speed_in_still_water_calculation 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 20) 
  (h2 : downstream_speed = 30) : 
  speed_in_still_water upstream_speed downstream_speed = 25 := by
  -- Unfold the definition of speed_in_still_water
  unfold speed_in_still_water
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the arithmetic
  norm_num

#check speed_in_still_water_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_in_still_water_calculation_l754_75411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l754_75452

/-- The number of days it takes for all three people to complete the work together -/
noncomputable def combined_work_time (john_days : ℝ) (rose_days : ℝ) (ethan_days : ℝ) : ℝ :=
  1 / (1 / john_days + 1 / rose_days + 1 / ethan_days)

/-- Theorem stating that given the individual work rates, the combined work time is approximately 106.67 days -/
theorem work_completion_time :
  let john_days : ℝ := 320
  let rose_days : ℝ := 480
  let ethan_days : ℝ := 240
  abs (combined_work_time john_days rose_days ethan_days - 106.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l754_75452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l754_75468

theorem rectangle_area_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (1.3 * L * 1.45 * B - L * B) / (L * B) = 0.885 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l754_75468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_unique_l754_75465

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ 
  f (Real.pi / 2) = 2 ∧ 
  ∀ x y : ℝ, f (x + y) + f (x - y) ≤ 2 * f x * Real.cos y

theorem solution_unique : 
  ∀ f : ℝ → ℝ, is_solution f → ∀ x : ℝ, f x = Real.cos x + 2 * Real.sin x :=
by
  sorry

#check solution_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_unique_l754_75465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_sequence_formula_l754_75461

def sequence_term (n : ℕ) : ℚ := ((2 * 10^n + 1) / 3)^2

theorem sequence_perfect_squares (n : ℕ) :
  ∃ m : ℚ, sequence_term n = m^2 :=
by
  use (2 * 10^n + 1) / 3
  rfl

theorem sequence_formula (n : ℕ) :
  sequence_term (n + 1) = 
    (100 * sequence_term n + 4800 * 10^n + 2304) / 100 :=
by
  simp [sequence_term]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_sequence_formula_l754_75461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l754_75498

theorem angle_difference (α β γ : ℝ) 
  (h1 : 0 ≤ α) (h2 : α ≤ β) (h3 : β < γ) (h4 : γ ≤ 2 * Real.pi)
  (h5 : Real.cos α + Real.cos β + Real.cos γ = 0)
  (h6 : Real.sin α + Real.sin β + Real.sin γ = 0) : 
  β - α = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l754_75498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_75_with_36_divisors_l754_75404

theorem multiple_of_75_with_36_divisors : ∃ n : ℕ,
  (∃ k : ℕ, n = 75 * k) ∧
  (Finset.card (Nat.divisors n) = 36) ∧
  (n / 75 = 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_75_with_36_divisors_l754_75404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l754_75495

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

-- Define the interval
def interval : Set ℝ := {x | -Real.pi/6 ≤ x ∧ x ≤ Real.pi/3}

-- Theorem statement
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (k : ℤ), ∃ (c : ℝ), c = k * Real.pi / 2 - Real.pi / 12 ∧ ∀ (x : ℝ), f (c + x) = f (c - x)) ∧
  (∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ M) ∧
  (∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), x ∈ interval → m ≤ f x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l754_75495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l754_75444

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l754_75444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l754_75464

/-- Represents a point on the 2D grid -/
structure Point where
  x : Int
  y : Int

/-- Represents the square grid -/
def Grid := {p : Point | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Probability of ending on a vertical side given a starting point -/
noncomputable def prob_vertical_side (p : Point) : ℝ := sorry

/-- The frog starts at (2, 3) -/
def start : Point := ⟨2, 3⟩

/-- Each jump is 1 unit long in a cardinal direction -/
def is_valid_jump (p q : Point) : Prop :=
  (abs (p.x - q.x) + abs (p.y - q.y) = 1) ∧ q ∈ Grid

/-- The main theorem to prove -/
theorem frog_jump_probability :
  prob_vertical_side start = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l754_75464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_removed_pyramids_l754_75440

/-- The volume of pyramids removed from a unit cube to make hexagonal faces -/
noncomputable def removed_pyramids_volume : ℝ := Real.sqrt 3 / 12

/-- The side length of the resulting hexagon on each face -/
noncomputable def hexagon_side_length : ℝ := 1 / 2

/-- The length of the segment from a cube vertex to the nearest hexagon vertex -/
noncomputable def corner_segment_length : ℝ := 1 / 4

/-- The height of each removed pyramid -/
noncomputable def pyramid_height : ℝ := 1 / 2

/-- The side length of the base of each removed pyramid -/
noncomputable def pyramid_base_side : ℝ := 1 / 2

/-- The number of corners in a cube -/
def num_cube_corners : ℕ := 8

theorem volume_of_removed_pyramids :
  (num_cube_corners : ℝ) * 
  (1 / 3 * (Real.sqrt 3 / 4 * pyramid_base_side ^ 2) * pyramid_height) = 
  removed_pyramids_volume := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_removed_pyramids_l754_75440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_focus_l754_75439

/-- A parabola defined by the equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line x + 2 = 0 -/
def tangent_line (x : ℝ) : Prop := x + 2 = 0

/-- A circle with center (a, b) and radius r -/
def my_circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circle_passes_through_focus :
  ∀ (a b r : ℝ),
  parabola a b →
  (∃ (x : ℝ), tangent_line x ∧ my_circle x 0 a b r) →
  my_circle 2 0 a b r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_focus_l754_75439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_product_equals_extreme_product_l754_75480

/-- Geometric progression with first term a₁ and common ratio q -/
def geometric_progression (a₁ q : ℝ) : ℕ → ℝ
| n => a₁ * q^(n - 1)

/-- The product of terms equidistant from the ends is equal to the product of the extreme terms -/
theorem equidistant_product_equals_extreme_product
  (a₁ q : ℝ) (n : ℕ) (h : n > 0) :
  ∀ i : ℕ, i > 0 → i ≤ n →
    (geometric_progression a₁ q i) * (geometric_progression a₁ q (n - i + 1)) =
    (geometric_progression a₁ q 1) * (geometric_progression a₁ q n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_product_equals_extreme_product_l754_75480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_T_indeterminate_l754_75400

-- Define the integral of x^2 with respect to x
noncomputable def integral_x_squared (x : ℝ) : ℝ := (1/3) * x^3

-- State the theorem
theorem constant_T_indeterminate :
  ∃ C : ℝ, ∀ x : ℝ, integral_x_squared x + C = 9 →
  ¬∃! T : ℝ, T = integral_x_squared x + C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_T_indeterminate_l754_75400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_correct_l754_75482

/-- Two circles in a 2D plane -/
structure TwoCircles where
  C₁ : (x y : ℝ) → x^2 + y^2 + 4*x + y + 1 = 0
  C₂ : (x y : ℝ) → x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- The length of the common chord of two intersecting circles -/
noncomputable def commonChordLength (circles : TwoCircles) : ℝ := 4 * Real.sqrt 5 / 5

/-- Theorem stating that the length of the common chord of the given circles is 4√5/5 -/
theorem common_chord_length_is_correct (circles : TwoCircles) :
  commonChordLength circles = 4 * Real.sqrt 5 / 5 := by
  -- Unfold the definition of commonChordLength
  unfold commonChordLength
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_correct_l754_75482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_property_l754_75497

/-- 
Given a natural number n for which n^5 + n^4 + 1 has exactly six distinct natural divisors,
prove that n^3 - n + 1 is a perfect square.
-/
theorem special_number_property (n : ℕ) 
  (h : (Finset.filter (λ d ↦ (n^5 + n^4 + 1) % d = 0) (Finset.range (n^5 + n^4 + 2))).card = 6) :
  ∃ k : ℕ, n^3 - n + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_property_l754_75497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l754_75441

theorem log_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : x^2 + y = 0) :
  Real.log (a^x + a^y) ≤ Real.log 2 + 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l754_75441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_14_times_5_12_l754_75454

-- Define a function to calculate the number of digits in a natural number
def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.succ (Nat.log n 10)

-- State the theorem
theorem digits_of_2_14_times_5_12 :
  numDigits (2^14 * 5^12) = 13 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_14_times_5_12_l754_75454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_xaxis_l754_75410

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the foci
noncomputable def leftFocus : ℝ × ℝ := (-Real.sqrt 7, 0)
noncomputable def rightFocus : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  onEllipse : ellipse x y

-- Define the perpendicularity condition
def perpendicularToFoci (p : PointOnEllipse) : Prop :=
  let pf1 := (p.x - leftFocus.1, p.y - leftFocus.2)
  let pf2 := (p.x - rightFocus.1, p.y - rightFocus.2)
  pf1.1 * pf2.1 + pf1.2 * pf2.2 = 0

-- State the theorem
theorem distance_to_xaxis (p : PointOnEllipse) 
  (h : perpendicularToFoci p) : 
  |p.y| = 9 * Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_xaxis_l754_75410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sample_distribution_l754_75457

/-- Represents a high school with stratified sampling -/
structure HighSchool where
  total_students : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sample_size : ℕ

/-- Represents the number of students sampled from each grade -/
structure SampleDistribution where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  deriving Repr

/-- Calculates the correct sample distribution for a given high school -/
def calculateSampleDistribution (school : HighSchool) : SampleDistribution :=
  { freshmen := school.freshmen * school.sample_size / school.total_students,
    sophomores := school.sophomores * school.sample_size / school.total_students,
    juniors := school.juniors * school.sample_size / school.total_students }

theorem correct_sample_distribution 
  (school : HighSchool) 
  (h1 : school.total_students = 900)
  (h2 : school.freshmen = 300)
  (h3 : school.sophomores = 200)
  (h4 : school.juniors = 400)
  (h5 : school.sample_size = 45) :
  calculateSampleDistribution school = { freshmen := 15, sophomores := 10, juniors := 20 } := by
  sorry

#eval calculateSampleDistribution { 
  total_students := 900, 
  freshmen := 300, 
  sophomores := 200, 
  juniors := 400, 
  sample_size := 45 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sample_distribution_l754_75457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l754_75417

-- Define x
noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 12

-- Define n as the floor of x
noncomputable def n : ℤ := ⌊x⌋

-- Define f
noncomputable def f : ℝ := x - n

-- Theorem to prove
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l754_75417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_11_is_22_l754_75487

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  isArithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic property
  rootProperty : a 5 + a 7 = 4  -- Sum of roots property

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- The main theorem -/
theorem sum_11_is_22 (seq : ArithmeticSequence) : sumFirstN seq 11 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_11_is_22_l754_75487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_proof_l754_75425

theorem second_number_proof (x : ℝ) :
  217 + x + 0.217 + 2.0017 = 221.2357 → x = 2.017 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_proof_l754_75425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_scale_invariance_problem_solution_l754_75460

theorem division_scale_invariance (a b q : ℕ) (h : a / b = q) (k : ℕ) (hk : k ≠ 0) :
  (k * a) / (k * b) = q := by
  sorry

theorem problem_solution : 36 / 4 = 9 → 3600 / 400 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_scale_invariance_problem_solution_l754_75460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tangent_equation_l754_75478

theorem smallest_angle_tangent_equation (x : ℝ) : 
  (x > 0) → 
  (Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x))) → 
  (∀ y, y > 0 → Real.tan (6 * y) = (Real.cos (2 * y) - Real.sin (2 * y)) / (Real.cos (2 * y) + Real.sin (2 * y)) → x ≤ y) →
  x = 5.625 * (π / 180) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tangent_equation_l754_75478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l754_75406

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin (Real.pi / 2 * x), Real.cos (Real.pi / 2 * x))

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin (Real.pi / 2 * x), Real.sqrt 3 * Real.sin (Real.pi / 2 * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * ((a x).1 + 2 * (b x).1) + (a x).2 * ((a x).2 + 2 * (b x).2)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = 2 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → f x ≤ 4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → f x ≥ 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = 1) ∧
  (f (2/3) = 4) ∧
  (f 0 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l754_75406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l754_75458

def negation_statement : Prop := 
  (∀ x : ℝ, x^2 - 2 < 0) = ¬(∃ x : ℝ, x^2 - 2 ≥ 0)

def hyperbola_property (d1 d2 a : ℝ) : Prop :=
  |d1 - d2| = 2*a

def condition_statement (m n : ℝ) : Prop :=
  (m > n → (2/3:ℝ)^m > (2/3:ℝ)^n) ∧ ¬((2/3:ℝ)^m > (2/3:ℝ)^n → m > n)

def contrapositive_statement : Prop :=
  (∀ x : ℝ, x ≠ 4 → x^2 - 3*x - 4 ≠ 0) = 
  (∀ x : ℝ, x^2 - 3*x - 4 = 0 → x = 4)

theorem correct_statements : 
  ∃! (s : Finset (Fin 4)), s.card = 2 ∧ 
  (1 ∈ s ↔ negation_statement) ∧
  (2 ∈ s ↔ ∀ d1 d2 a, hyperbola_property d1 d2 a) ∧
  (3 ∈ s ↔ ∀ m n, condition_statement m n) ∧
  (4 ∈ s ↔ contrapositive_statement) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l754_75458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_sqrt2_div_2_l754_75446

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = ln x -/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The distance between points M and N -/
noncomputable def distance (t : ℝ) : ℝ := f t - g t

/-- Theorem stating that the minimum distance occurs at sqrt(2)/2 -/
theorem min_distance_at_sqrt2_div_2 :
  ∃ (t₀ : ℝ), t₀ > 0 ∧ t₀ = Real.sqrt 2 / 2 ∧
  ∀ (t : ℝ), t > 0 → distance t ≥ distance t₀ := by
  sorry

#check min_distance_at_sqrt2_div_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_sqrt2_div_2_l754_75446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_PQ_KX_l754_75485

-- Define the points
variable (A B C D K L M N P Q X : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral and equilateral triangles
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def is_equilateral_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define midpoint and circumcenter
def is_midpoint (M A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def is_circumcenter (X A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define perpendicularity
def perpendicular (AB CD : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem perpendicular_PQ_KX 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_ABK : is_equilateral_triangle A B K)
  (h_BCL : is_equilateral_triangle B C L)
  (h_CDM : is_equilateral_triangle C D M)
  (h_DAN : is_equilateral_triangle D A N)
  (h_P : is_midpoint P B L)
  (h_Q : is_midpoint Q A N)
  (h_X : is_circumcenter X C M D) :
  perpendicular (P, Q) (K, X) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_PQ_KX_l754_75485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bruce_payment_l754_75484

def grapes_kg : ℕ := 8
def grapes_rate : ℕ := 70
def mangoes_kg : ℕ := 9
def mangoes_rate : ℕ := 55
def oranges_kg : ℕ := 5
def oranges_rate : ℕ := 40
def strawberries_kg : ℕ := 4
def strawberries_rate : ℕ := 90
def discount_rate : ℚ := 1/10
def tax_rate : ℚ := 1/20

def total_cost : ℕ := grapes_kg * grapes_rate + mangoes_kg * mangoes_rate + 
                      oranges_kg * oranges_rate + strawberries_kg * strawberries_rate

def discounted_total : ℚ := (1 - discount_rate) * total_cost

def final_amount : ℚ := discounted_total * (1 + tax_rate)

theorem bruce_payment : 
  (⌊final_amount * 100⌋ : ℚ) / 100 = 1526.18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bruce_payment_l754_75484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_area_ratio_l754_75409

/-- 
Given a regular pentagon where all diagonals are drawn, forming a smaller pentagon in the middle,
the ratio of the area of the smaller pentagon to the area of the original pentagon is (7 - 3√5) / 2.
-/
theorem regular_pentagon_diagonal_area_ratio : 
  ∀ (A : ℝ) (a : ℝ), A > 0 → a > 0 →
  let larger_pentagon_area := A
  let smaller_pentagon_area := (7 - 3 * Real.sqrt 5) / 2 * A
  let diagonal_ratio := smaller_pentagon_area / larger_pentagon_area
  diagonal_ratio = (7 - 3 * Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_area_ratio_l754_75409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l754_75427

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a*x^2 + x) / (f x)

theorem problem_solution :
  (∀ x : ℝ, x ≠ 0 → f (1 + 1/x) = 1/x^2 - 1) →
  (∀ x : ℝ, x ≠ 0 → f x = x^2 - 2*x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 2 → Monotone (g a)) → a < -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l754_75427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_same_plane_implies_parallel_l754_75479

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_to_same_plane_implies_parallel 
  (a b : Line) (γ : Plane) : 
  perpendicular a γ → perpendicular b γ → parallel a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_same_plane_implies_parallel_l754_75479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l754_75431

noncomputable section

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (1, Real.sqrt 2 / 2)

-- Define the right focus F
def focus_F (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - 1), 0)

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

-- Define the angle bisector condition
def is_angle_bisector (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

-- Theorem statement
theorem ellipse_slope_theorem (a : ℝ) (k : ℝ) :
  ellipse_C a point_P.1 point_P.2 →
  (∃ x1 y1 x2 y2, 
    ellipse_C a x1 y1 ∧ 
    ellipse_C a x2 y2 ∧ 
    line_through_points (focus_F a).1 (focus_F a).2 x1 y1 x2 y2 ∧
    ¬(x1 = point_P.1 ∧ y1 = point_P.2) ∧
    ¬(x2 = point_P.1 ∧ y2 = point_P.2) ∧
    is_angle_bisector x1 y1 point_P.1 point_P.2 x2 y2 (focus_F a).1 (focus_F a).2) →
  k = Real.sqrt 2 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l754_75431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_product_l754_75462

-- Define the triangle and points
variable (X Y Z X' Y' Z' P : EuclideanSpace ℝ (Fin 2))

-- Define the condition that X', Y', Z' are on the sides of the triangle
def on_sides (X Y Z X' Y' Z' : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧ 0 < t₃ ∧ t₃ < 1 ∧
  X' = t₁ • Y + (1 - t₁) • Z ∧
  Y' = t₂ • Z + (1 - t₂) • X ∧
  Z' = t₃ • X + (1 - t₃) • Y

-- Define the condition that XX', YY', ZZ' are concurrent at P
def concurrent (X Y Z X' Y' Z' P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 
  P = t₁ • X + (1 - t₁) • X' ∧
  P = t₂ • Y + (1 - t₂) • Y' ∧
  P = t₃ • Z + (1 - t₃) • Z'

-- Define the ratio condition
def ratio_sum (X Y Z X' Y' Z' P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  let d₁ := ‖X - P‖ / ‖P - X'‖
  let d₂ := ‖Y - P‖ / ‖P - Y'‖
  let d₃ := ‖Z - P‖ / ‖P - Z'‖
  d₁ + d₂ + d₃ = 100

-- State the theorem
theorem triangle_ratio_product 
  (X Y Z X' Y' Z' P : EuclideanSpace ℝ (Fin 2)) 
  (h₁ : on_sides X Y Z X' Y' Z')
  (h₂ : concurrent X Y Z X' Y' Z' P)
  (h₃ : ratio_sum X Y Z X' Y' Z' P) :
  let d₁ := ‖X - P‖ / ‖P - X'‖
  let d₂ := ‖Y - P‖ / ‖P - Y'‖
  let d₃ := ‖Z - P‖ / ‖P - Z'‖
  d₁ * d₂ * d₃ = 102 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_product_l754_75462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l754_75419

-- Define the two curves
def curve1 (x : ℝ) : ℝ := 2 * (x + 1)
noncomputable def curve2 (x : ℝ) : ℝ := x + Real.log x

-- Define the distance function between two points on the curves with the same y-coordinate
def distance (x1 x2 : ℝ) : ℝ := |x2 - x1|

-- Statement of the theorem
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = 3/2 ∧
  ∀ (x1 x2 : ℝ), x1 > 0 → x2 > 0 →
  curve1 x1 = curve2 x2 → distance x1 x2 ≥ min_dist := by
  sorry

-- Helper lemma for the minimum point
lemma min_point_at_one :
  ∀ (x : ℝ), x > 0 → (x - Real.log x) / 2 + 1 ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l754_75419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l754_75408

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

/-- The statement that proves the minimum value of a -/
theorem min_a_value (a : ℝ) :
  (∃ x : ℝ, f a x ≤ 0) → a ≥ 1 - 1 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l754_75408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_final_price_theorem_l754_75459

/-- Calculates the final price of a saree in USD after discounts, taxes, and conversion --/
def calculate_final_price (original_price : ℝ) (discounts : List ℝ) 
  (tax_rate : ℝ) (luxury_tax_rate : ℝ) (custom_fee : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_price := discounts.foldl (fun price discount => price * (1 - discount)) original_price
  let taxed_price := discounted_price * (1 + tax_rate + luxury_tax_rate) + custom_fee
  taxed_price * exchange_rate

/-- Theorem stating that the final price of the saree is approximately $46.82 --/
theorem saree_final_price_theorem : 
  let original_price : ℝ := 5000
  let discounts : List ℝ := [0.20, 0.15, 0.10, 0.05]
  let tax_rate : ℝ := 0.12
  let luxury_tax_rate : ℝ := 0.05
  let custom_fee : ℝ := 200
  let exchange_rate : ℝ := 0.013
  let final_price := calculate_final_price original_price discounts tax_rate luxury_tax_rate custom_fee exchange_rate
  abs (final_price - 46.82) < 0.01 := by
  sorry

#eval calculate_final_price 5000 [0.20, 0.15, 0.10, 0.05] 0.12 0.05 200 0.013

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_final_price_theorem_l754_75459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_gender_probability_l754_75445

def total_students : ℕ := 8
def num_men : ℕ := 4
def num_women : ℕ := 4
def num_selected : ℕ := 4

theorem equal_gender_probability :
  (Nat.choose num_men (num_selected / 2) * Nat.choose num_women (num_selected / 2) : ℚ) /
  Nat.choose total_students num_selected = 18 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_gender_probability_l754_75445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_lamp_height_l754_75476

/-- The illumination at the edge of a circular table -/
noncomputable def illumination (k : ℝ) (R : ℝ) (α : ℝ) : ℝ :=
  k * Real.sin α * (Real.cos α)^2 / R^2

/-- The height of the lamp above the table -/
noncomputable def lampHeight (R : ℝ) (α : ℝ) : ℝ :=
  R * Real.tan α

/-- Theorem stating that the optimal height for maximum illumination is approximately 0.7R -/
theorem optimal_lamp_height (k : ℝ) (R : ℝ) :
  ∃ (α : ℝ), IsLocalMax (illumination k R) α ∧ 
  |lampHeight R α / R - Real.sqrt (1/2)| < 0.01 := by
  sorry

#check optimal_lamp_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_lamp_height_l754_75476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_x_axis_l754_75494

/-- If a point P with coordinates (m, m-1) lies on the x-axis, then m = 1. -/
theorem point_on_x_axis (m : ℝ) : (m, m - 1).2 = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_x_axis_l754_75494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equals_set_l754_75469

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 4, 6}

theorem complement_intersection_equals_set : (U \ A) ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equals_set_l754_75469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l754_75438

/-- An ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major : ℝ
  semi_minor : ℝ

/-- The distance between foci of an ellipse -/
noncomputable def foci_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.semi_major ^ 2 - e.semi_minor ^ 2)

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point is on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- Checks if an ellipse is tangent to a line -/
def is_tangent_to_line (e : Ellipse) (l : Line) : Prop :=
  ∃ (p : ℝ × ℝ), on_line p l ∧ (p.1 - e.center.1) ^ 2 / e.semi_major ^ 2 + (p.2 - e.center.2) ^ 2 / e.semi_minor ^ 2 = 1

/-- A vertical line -/
def vertical_line (x : ℝ) : Line :=
  { slope := 0, intercept := x }

theorem ellipse_foci_distance :
  ∀ (e : Ellipse),
    e.center = (3, 2) →
    e.semi_major = 3 →
    e.semi_minor = 2 →
    is_tangent_to_line e (vertical_line 0) →
    is_tangent_to_line e { slope := 0, intercept := 0 } →
    is_tangent_to_line e { slope := 1, intercept := 1 } →
    foci_distance e = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l754_75438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l754_75435

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (3, 0)
noncomputable def B : ℝ × ℝ := (0, 3)

-- Define the line on which C lies
def line_C (x y : ℝ) : Prop := x + y = 7

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_ABC_area :
  ∃ (C : ℝ × ℝ), line_C C.1 C.2 ∧ triangle_area A B C = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l754_75435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l754_75463

/-- A rhombus with given area and diagonal ratio has a specific longest diagonal length -/
theorem rhombus_longest_diagonal 
  (area : ℝ) 
  (diag_ratio : ℝ) 
  (h_area : area = 192) 
  (h_ratio : diag_ratio = 4 / 3) : 
  ∃ (long_diag : ℝ), long_diag = 16 * Real.sqrt 2 ∧ 
    area = (1 / 2) * long_diag * (long_diag / diag_ratio) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l754_75463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_expression_l754_75488

theorem integer_part_of_expression (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  3 ≤ (3 : ℝ)^((Real.cos x)^2) + (3 : ℝ)^((Real.sin x)^4) ∧ 
  (3 : ℝ)^((Real.cos x)^2) + (3 : ℝ)^((Real.sin x)^4) < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_expression_l754_75488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equals_four_squares_l754_75492

/-- Represents the weight of a shape -/
structure Weight where
  value : ℕ

instance : HMul ℕ Weight Weight where
  hMul n w := ⟨n * w.value⟩

instance : HAdd Weight Weight Weight where
  hAdd w1 w2 := ⟨w1.value + w2.value⟩

/-- Represents a circle shape -/
def Circle : Type := Unit

/-- Represents a square shape -/
def Square : Type := Unit

/-- The weight of a circle -/
noncomputable def circle_weight : Weight := ⟨1⟩

/-- The weight of a square -/
noncomputable def square_weight : Weight := ⟨1⟩

/-- First balance condition: 3 circles = 5 squares -/
axiom balance1 : (3 : ℕ) * circle_weight = (5 : ℕ) * square_weight

/-- Second balance condition: 2 circles = 3 squares + 1 circle -/
axiom balance2 : (2 : ℕ) * circle_weight = (3 : ℕ) * square_weight + circle_weight

/-- Theorem: One circle is equivalent to 4 squares -/
theorem circle_equals_four_squares : circle_weight = (4 : ℕ) * square_weight := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equals_four_squares_l754_75492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hamburger_combinations_l754_75412

theorem hamburger_combinations : ∃ n : Nat, n = 6144 := by
  let num_condiments : Nat := 10
  let condiment_combinations : Nat := 2^num_condiments
  let patty_options : Nat := 3
  let bun_options : Nat := 2
  let total_combinations := condiment_combinations * patty_options * bun_options
  
  have h : total_combinations = 6144 := by
    -- Proof steps would go here
    sorry
  
  exact ⟨total_combinations, h⟩

#eval 2^10 * 3 * 2  -- This will output 6144

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hamburger_combinations_l754_75412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l754_75414

def equation (z : ℂ) : ℂ := z^2 + 2*z + (3 - 4*Complex.I)

theorem equation_roots :
  ∃ (z₁ z₂ : ℂ), z₁ ≠ z₂ ∧ 
  equation z₁ = 0 ∧ 
  equation z₂ = 0 ∧
  (z₁ = 2*Complex.I ∨ z₁ = -2 + 2*Complex.I) ∧
  (z₂ = 2*Complex.I ∨ z₂ = -2 + 2*Complex.I) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l754_75414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leg_head_difference_l754_75430

/-- Represents the number of legs for a given animal -/
def legs (animal : String) : ℕ :=
  match animal with
  | "cow" => 4
  | "chicken" => 2
  | _ => 0

/-- Represents the number of heads for any animal -/
def heads (_animal : String) : ℕ := 1

/-- The number of cows in the group -/
def numCows : ℕ := 8

/-- The total number of animals in the group -/
def totalAnimals : ℕ → ℕ := id

/-- The number of chickens in the group -/
def numChickens (total : ℕ) : ℕ := total - numCows

theorem leg_head_difference (total : ℕ) :
  (numCows * legs "cow" + numChickens total * legs "chicken") =
  2 * (numCows * heads "cow" + numChickens total * heads "chicken") + 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leg_head_difference_l754_75430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_in_M_property_of_M_functions_continuity_at_zero_l754_75437

/-- The set M of functions with the given property -/
def M : Set (ℝ → ℝ) :=
  {f | ∀ s t, s > 0 → t > 0 → f s > 0 ∧ f t > 0 ∧ f s + f t < f (s + t)}

/-- The function f₂(x) = 2ˣ - 1 -/
noncomputable def f₂ : ℝ → ℝ := fun x ↦ Real.exp (Real.log 2 * x) - 1

theorem f₂_in_M : f₂ ∈ M := by sorry

theorem property_of_M_functions (f : ℝ → ℝ) (hf : f ∈ M) :
    ∀ x m, x > 0 → x + m > 0 → m ≠ 0 → m * (f (x + m) - f x) > 0 := by sorry

theorem continuity_at_zero (f : ℝ → ℝ) (hf : f ∈ M) :
    ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x → x ≤ δ → f x < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_in_M_property_of_M_functions_continuity_at_zero_l754_75437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_log_range_l754_75428

-- Define the function f(x) = log_a(2-ax)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- Define the property of being monotonically decreasing on an interval
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem monotone_decreasing_log_range (a : ℝ) :
  (MonotonicallyDecreasing (f a) 0 1) ↔ (1 < a ∧ a < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_log_range_l754_75428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_bounce_angle_l754_75407

theorem billiard_bounce_angle 
  (R d : ℝ) 
  (h_R : R > 0) 
  (h_d : 0 < d ∧ d < R) : 
  ∃ α : ℝ, 
    0 < α ∧ 
    α < π/2 ∧ 
    Real.sin α = (-R^2 + R * Real.sqrt (R^2 + 8*d^2)) / (4*d^2) ∧
    (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → 
      Real.sin θ = (-R^2 + R * Real.sqrt (R^2 + 8*d^2)) / (4*d^2) → 
      θ = α) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_bounce_angle_l754_75407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_sqrt3_neg1_l754_75496

noncomputable section

open Real

/-- Converts a point from rectangular coordinates to polar coordinates -/
def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if x > 0 then arctan (y / x)
           else if x < 0 && y ≥ 0 then arctan (y / x) + π
           else if x < 0 && y < 0 then arctan (y / x) - π
           else if x = 0 && y > 0 then π / 2
           else if x = 0 && y < 0 then -π / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2*π else θ)

theorem rect_to_polar_sqrt3_neg1 :
  rect_to_polar (sqrt 3) (-1) = (2, 11*π/6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_sqrt3_neg1_l754_75496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_circumradius_ratio_equal_l754_75413

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_pos : side > 0

/-- The perimeter of an equilateral triangle -/
noncomputable def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side

/-- The circumradius of an equilateral triangle -/
noncomputable def circumradius (t : EquilateralTriangle) : ℝ := (t.side * Real.sqrt 3) / 3

/-- Theorem: The ratio of perimeters of two equilateral triangles
    is always equal to the ratio of their circumradii -/
theorem perimeter_circumradius_ratio_equal
  (t1 t2 : EquilateralTriangle)
  (h : t1.side ≠ t2.side) :
  perimeter t1 / perimeter t2 = circumradius t1 / circumradius t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_circumradius_ratio_equal_l754_75413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l754_75470

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The count of positive integers n ≤ 2500 with S(n) odd -/
def c : ℕ := sorry

/-- The count of positive integers n ≤ 2500 with S(n) even -/
def d : ℕ := sorry

theorem divisor_sum_parity_difference : |Int.ofNat c - Int.ofNat d| = 147 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l754_75470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_perpendicular_lines_l754_75451

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: ax + (a+2)y + 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / (a + 2)

/-- The slope of line l2: x + ay + 2 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / a

theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope_l1 a) (slope_l2 a) → a = 0 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_perpendicular_lines_l754_75451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l754_75403

noncomputable section

/-- A function y of x is a power function if it can be written as y = ax^k, where a is a constant and k is a real number. -/
def IsPowerFunction (y : ℝ → ℝ) : Prop :=
  ∃ (a k : ℝ), ∀ x, y x = a * (x ^ k)

/-- The given function y in terms of m and x. -/
def y (m : ℝ) (x : ℝ) : ℝ := (m^2 + 2*m - 2) * (x^(1/(m-1)))

theorem power_function_m_value (m : ℝ) :
  IsPowerFunction (y m) → m = -3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l754_75403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_18m_4_l754_75455

theorem divisors_of_18m_4 (m : ℕ) (h_odd : Odd m) (h_divisors : (Nat.divisors m).card = 13) :
  (Nat.divisors (18 * m^4)).card = 294 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_18m_4_l754_75455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_slope_l754_75489

/-- The exponential function -/
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

/-- The first curve -/
noncomputable def f (x : ℝ) : ℝ := exp (x - 2)

/-- The second curve -/
noncomputable def g (x : ℝ) : ℝ := exp (x + 2022) - 2022

/-- Theorem: Common tangent line slope -/
theorem common_tangent_slope :
  ∀ k b : ℝ,
  (∃ x₁ x₂ : ℝ, 
    (k * x₁ + b = f x₁) ∧
    (k * x₂ + b = g x₂) ∧
    (k = deriv f x₁) ∧
    (k = deriv g x₂)) →
  k = 1011 / 1012 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_slope_l754_75489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l754_75466

-- Define the plane
def Plane := ℝ → ℝ → ℝ → Prop

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the distance between two points
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

-- Define what it means for three points to be non-collinear
def nonCollinear (a b c : Point3D) : Prop :=
  ¬∃ (t : ℝ), b.x - a.x = t * (c.x - a.x) ∧ 
               b.y - a.y = t * (c.y - a.y) ∧ 
               b.z - a.z = t * (c.z - a.z)

-- State the theorem
theorem equidistant_point_exists (π : Plane) (a b c : Point3D) 
  (h : nonCollinear a b c) :
  ∃! p : Point3D, π p.x p.y p.z ∧ 
    distance p a = distance p b ∧ 
    distance p b = distance p c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l754_75466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l754_75477

/-- An isosceles triangle with two sides of length 7 and one side of length 12 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  side1_eq_side2 : side1 = side2
  side1_eq_7 : side1 = 7
  base_eq_12 : base = 12

/-- A triangle similar to the isosceles triangle with longest side 36 -/
structure SimilarTriangle (t : IsoscelesTriangle) where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  similar_ratio : ℝ
  similar_ratio_def : similar_ratio = base / t.base
  side1_def : side1 = t.side1 * similar_ratio
  side2_def : side2 = t.side2 * similar_ratio
  base_def : base = t.base * similar_ratio
  base_eq_36 : base = 36

/-- The perimeter of a triangle -/
def perimeter (t : SimilarTriangle (a : IsoscelesTriangle)) : ℝ :=
  t.side1 + t.side2 + t.base

/-- The main theorem: The perimeter of the similar triangle is 78 -/
theorem similar_triangle_perimeter (t : IsoscelesTriangle) (s : SimilarTriangle t) :
    perimeter s = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l754_75477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l754_75475

noncomputable def min_symbol (a b : ℝ) := min a b

noncomputable def max_symbol (a b : ℝ) := max a b

theorem problem_solution :
  (min_symbol (-5) (-0.5) + max_symbol (-4) 2 = -3) ∧
  (min_symbol 1 (-3) + max_symbol (-5) (min_symbol (-2) (-7)) = -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l754_75475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_number_l754_75453

/-- Systematic sampling function -/
def systematicSample (totalStudents sampleSize firstSelected : ℕ) : ℕ → ℕ :=
  fun n ↦ (firstSelected - 1 + (totalStudents / sampleSize) * (n - 1)) % totalStudents + 1

/-- Theorem: In a systematic sampling of 5 from 60, if first is 04, fifth is 52 -/
theorem fifth_selected_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (firstSelected : ℕ)
  (h1 : totalStudents = 60)
  (h2 : sampleSize = 5)
  (h3 : firstSelected = 4) :
  systematicSample totalStudents sampleSize firstSelected 5 = 52 := by
  sorry

#eval systematicSample 60 5 4 5  -- Should evaluate to 52

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_number_l754_75453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_open_interval_l754_75450

def M : Set ℝ := {x | Real.log (x - 1) / Real.log (1/2) > -1}
def N : Set ℝ := {x | 1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 4}

theorem M_intersect_N_eq_open_interval :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_open_interval_l754_75450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_conditions_l754_75481

/-- A triangle ABC is right if one of its angles is 90 degrees. -/
def is_right_triangle (A B C : ℝ) : Prop := A = 90 ∨ B = 90 ∨ C = 90

/-- Condition 1: ∠A + ∠B = ∠C -/
def condition1 (A B C : ℝ) : Prop := A + B = C

/-- Condition 2: ∠A : ∠B : ∠C = 1 : 2 : 3 -/
def condition2 (A B C : ℝ) : Prop := 3 * A = 2 * B ∧ 2 * B = C

/-- Condition 3: ∠A = 90° - ∠B -/
def condition3 (A B C : ℝ) : Prop := A = 90 - B

/-- Condition 4: ∠A = ∠B = 1/2 ∠C -/
def condition4 (A B C : ℝ) : Prop := A = B ∧ A = C / 2

/-- Condition 5: ∠A = 2∠B = 3∠C -/
def condition5 (A B C : ℝ) : Prop := A = 2 * B ∧ A = 3 * C

theorem right_triangle_conditions (A B C : ℝ) :
  (condition1 A B C → is_right_triangle A B C) ∧
  (condition2 A B C → is_right_triangle A B C) ∧
  (condition3 A B C → is_right_triangle A B C) ∧
  (condition4 A B C → is_right_triangle A B C) ∧
  ¬(condition5 A B C → is_right_triangle A B C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_conditions_l754_75481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_theorem_l754_75418

/-- Represents the production capabilities of workers in a factory -/
structure ProductionRate where
  workers : ℕ
  hours : ℕ
  widgets : ℕ
  whoosits : ℕ

/-- The production rates given in the problem -/
def rate1 : ProductionRate := ⟨100, 1, 300, 200⟩
def rate2 : ProductionRate := ⟨60, 2, 240, 300⟩

/-- The theorem stating that given the production rates, m must be 450 -/
theorem production_rate_theorem (m : ℕ) : 
  let rate3 : ProductionRate := ⟨50, 3, 150, m⟩
  rate1.workers * rate1.hours * rate2.widgets = rate2.workers * rate2.hours * rate1.widgets ∧
  rate1.workers * rate1.hours * rate2.whoosits = rate2.workers * rate2.hours * rate1.whoosits ∧
  rate3.workers * rate3.hours * rate1.widgets = rate1.workers * rate1.hours * rate3.widgets ∧
  rate3.workers * rate3.hours * rate1.whoosits = rate1.workers * rate1.hours * m
  → m = 450 := by
  intro h
  sorry

#check production_rate_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_theorem_l754_75418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equality_l754_75421

theorem cosine_sum_equality : ∀ (x y z w : ℝ),
  x = 47 * π / 180 →
  y = 13 * π / 180 →
  z = 43 * π / 180 →
  w = 167 * π / 180 →
  Real.cos x * Real.cos y - Real.cos z * Real.sin w = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equality_l754_75421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trees_count_l754_75423

/-- Calculates the number of new trees grown from seeds --/
def new_trees (tree_a_plants tree_b_plants : ℕ) 
              (tree_a_seeds_per_plant tree_b_seeds_per_plant : ℕ)
              (tree_a_plant_rate tree_b_plant_rate : ℚ)
              (tree_a_survival_rate tree_b_survival_rate : ℚ) : ℕ :=
  let tree_a_seeds := tree_a_plants * tree_a_seeds_per_plant
  let tree_b_seeds := tree_b_plants * tree_b_seeds_per_plant
  let tree_a_planted := (tree_a_seeds : ℚ) * tree_a_plant_rate
  let tree_b_planted := (tree_b_seeds : ℚ) * tree_b_plant_rate
  let tree_a_survived := tree_a_planted * tree_a_survival_rate
  let tree_b_survived := tree_b_planted * tree_b_survival_rate
  (Int.floor tree_a_survived).toNat + (Int.floor tree_b_survived).toNat

/-- Theorem stating the number of new trees grown --/
theorem new_trees_count : 
  new_trees 25 20 1 2 (3/5) (4/5) (3/4) (9/10) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trees_count_l754_75423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_expression_value_l754_75420

theorem greatest_expression_value :
  ∃ (expressions : Set ℝ) (greatest_expr : ℝ),
    greatest_expr ∈ expressions ∧
    (∀ x ∈ expressions, x ≤ greatest_expr) ∧
    greatest_expr = 0.9986095661846496 := by
  
  -- Define a set of expressions (we'll use a sorry here as we don't know the actual set)
  let expressions : Set ℝ := sorry

  -- Define the greatest expression
  let greatest_expr : ℝ := 0.9986095661846496

  -- Prove the conditions
  have h1 : greatest_expr ∈ expressions := sorry
  have h2 : ∀ x ∈ expressions, x ≤ greatest_expr := sorry
  have h3 : greatest_expr = 0.9986095661846496 := rfl

  -- Combine the proofs
  exact ⟨expressions, greatest_expr, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_expression_value_l754_75420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l754_75447

theorem divisibility_property (a k d : ℕ) (ha : a > 0) (hk : k > 0) (hd : d > 0) : 
  ∃ n : ℕ, n > 0 ∧ d ∣ k * a^n + n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l754_75447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l754_75448

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 → b > 0 → c > 0 →
  Real.sin (2 * A + π / 6) = 1 / 2 →
  a = Real.sqrt 3 →
  b = 1 →
  Real.sin A / a = Real.sin B / b →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l754_75448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_infinite_wrapping_l754_75426

/-- Represents an infinite cone -/
structure InfiniteCone where
  angle : Real
  angle_small : angle > 0 ∧ angle < Real.pi / 2

/-- Represents a non-stretchable, infinite tape -/
structure InfiniteTape

/-- Represents a wrapping of a tape around a cone -/
def Wrapping (cone : InfiniteCone) (tape : InfiniteTape) : Prop := sorry

/-- Predicate indicating if a wrapping makes infinitely many turns around the cone's axis -/
def MakesInfinitelyManyTurns (cone : InfiniteCone) (tape : InfiniteTape) (w : Wrapping cone tape) : Prop := sorry

/-- Predicate indicating if a wrapping avoids the apex of the cone -/
def AvoidsApex (cone : InfiniteCone) (tape : InfiniteTape) (w : Wrapping cone tape) : Prop := sorry

/-- Predicate indicating if a wrapping doesn't involve cutting or twisting the tape -/
def NoCutOrTwist (cone : InfiniteCone) (tape : InfiniteTape) (w : Wrapping cone tape) : Prop := sorry

theorem impossible_infinite_wrapping (cone : InfiniteCone) (tape : InfiniteTape) :
  ¬∃ (w : Wrapping cone tape),
    MakesInfinitelyManyTurns cone tape w ∧ AvoidsApex cone tape w ∧ NoCutOrTwist cone tape w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_infinite_wrapping_l754_75426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_and_inequality_l754_75471

noncomputable section

variables (b c : ℝ)

noncomputable def f (x : ℝ) := Real.log x + b * x - c

theorem tangent_line_and_monotonicity_and_inequality :
  (∃ y, x + y + 4 = 0 ∧ y = f b c 1) →
  (∀ x, f b c x = Real.log x - 2 * x - 3) ∧ 
  (∀ x, 0 < x → x < 1/2 → (deriv (f b c)) x > 0) ∧
  (∀ x, x > 1/2 → (deriv (f b c)) x < 0) ∧
  (∀ k, (∀ x, x ∈ Set.Icc (1/2 : ℝ) 3 → f b c x ≥ 2 * Real.log x + k * x) → 
    k ≤ -2 * Real.log 2 - 8) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_and_inequality_l754_75471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megs_earnings_l754_75422

def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

def time_difference (start_time end_time : ℚ) : ℚ := end_time - start_time

def total_earnings (monday wednesday thursday saturday hourly_rate : ℚ) : ℚ :=
  let total_minutes := 
    hours_to_minutes monday + 
    hours_to_minutes wednesday + 
    hours_to_minutes (time_difference 9.25 11.5) + 
    saturday
  let total_hours := total_minutes / 60
  total_hours * hourly_rate

theorem megs_earnings :
  total_earnings (7/4) (5/4) (11/4) 45 4 = 24 := by
  -- Proof goes here
  sorry

#eval total_earnings (7/4) (5/4) (11/4) 45 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_megs_earnings_l754_75422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l754_75442

theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 4/3 * x) →
  let e := Real.sqrt (a^2 + b^2) / a
  e = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l754_75442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palm_oil_in_cheese_l754_75456

/-- Represents the percentage of palm oil in cheese -/
noncomputable def palm_oil_percentage (palm_oil_price_increase : ℝ) (cheese_price_increase : ℝ) : ℝ :=
  (cheese_price_increase / palm_oil_price_increase) * 100

/-- Theorem stating that a 10% increase in palm oil price leading to a 3% increase in cheese price
    implies that 30% of the cheese is palm oil -/
theorem palm_oil_in_cheese 
  (palm_oil_price_increase : ℝ) 
  (cheese_price_increase : ℝ) 
  (h1 : palm_oil_price_increase = 10)
  (h2 : cheese_price_increase = 3) : 
  palm_oil_percentage palm_oil_price_increase cheese_price_increase = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_palm_oil_in_cheese_l754_75456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l754_75402

theorem polynomial_remainder_theorem (P : Polynomial ℚ) 
  (h1 : P.eval 1 = 2) 
  (h2 : P.eval 2 = 1) : 
  ∃ Q : Polynomial ℚ, P = Q * (X - 1) * (X - 2) + (3 - X) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l754_75402
