import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_log_inequality_l1185_118569

/-- The function f(x) defined on the positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

/-- Theorem stating the conditions for f to be monotonically increasing and the inequality for logarithms. -/
theorem f_monotone_and_log_inequality :
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → Monotone (f a)) ↔ a ≤ 2) ∧
  (∀ m n : ℝ, m > n → n > 0 → Real.log m - Real.log n > 2 * (m - n) / (m + n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_log_inequality_l1185_118569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disorder_probability_given_positive_test_l1185_118532

/-- Probability of having the disorder -/
noncomputable def p_disorder : ℝ := 1 / 1000

/-- Probability of not having the disorder -/
noncomputable def p_no_disorder : ℝ := 1 - p_disorder

/-- Probability of testing positive given the person has the disorder -/
noncomputable def p_positive_given_disorder : ℝ := 1

/-- Probability of testing positive given the person does not have the disorder (false positive rate) -/
noncomputable def p_false_positive : ℝ := 0.05

/-- Probability of testing positive -/
noncomputable def p_positive : ℝ := p_positive_given_disorder * p_disorder + p_false_positive * p_no_disorder

/-- Probability of having the disorder given a positive test result -/
noncomputable def p_disorder_given_positive : ℝ := (p_positive_given_disorder * p_disorder) / p_positive

theorem disorder_probability_given_positive_test :
  |p_disorder_given_positive - 0.019627| < 0.000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disorder_probability_given_positive_test_l1185_118532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_in_ammonia_l1185_118543

/-- Molar mass of nitrogen in g/mol -/
noncomputable def molar_mass_N : ℝ := 14.01

/-- Molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of ammonia (NH3) in g/mol -/
noncomputable def molar_mass_NH3 : ℝ := molar_mass_N + 3 * molar_mass_H

/-- Mass percentage of nitrogen in ammonia -/
noncomputable def mass_percentage_N : ℝ := (molar_mass_N / molar_mass_NH3) * 100

theorem mass_percentage_N_in_ammonia :
  abs (mass_percentage_N - 82.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_in_ammonia_l1185_118543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_subset_with_double_exclusion_l1185_118567

theorem no_subset_with_double_exclusion :
  ¬ ∃ (A : Finset ℕ), A ⊆ Finset.range 3000 ∧ A.card = 2000 ∧
    ∀ x ∈ A, 2 * x ∉ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_subset_with_double_exclusion_l1185_118567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l1185_118528

-- Define the function f(x) = ∛(x²)
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (2/3)

-- State the theorem
theorem f_increasing_on_positive_reals :
  StrictMonoOn f (Set.Ioi 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l1185_118528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l1185_118553

/-- Represents the number of pieces of data Programmer B can input per minute -/
def x : ℝ := sorry

/-- Total pieces of data to be entered -/
def total_data : ℝ := 2640

/-- Time difference between Programmer A and B in hours -/
def time_difference : ℝ := 2

/-- Theorem stating that the equation correctly represents the situation -/
theorem correct_equation :
  (total_data / (2 * x) = total_data / x - time_difference * 60) ↔
  (∃ (speed_a speed_b : ℝ),
    speed_a = 2 * speed_b ∧
    total_data / speed_a + time_difference * 60 = total_data / speed_b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l1185_118553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_point_difference_l1185_118574

def rotate90CounterClockwise (center : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := p
  (cx - (py - cy), cy + (px - cx))

def reflectAboutNegativeX (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

theorem transform_point_difference (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90CounterClockwise (2, 4) p
  let final := reflectAboutNegativeX rotated
  final = (-4, 2) → b - a = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_point_difference_l1185_118574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_theorem_l1185_118521

/-- Represents the motion of a rider on a Ferris wheel -/
noncomputable def ferris_wheel_motion (t : ℝ) : ℝ :=
  30 * Real.cos (Real.pi / 60 * t) + 30

/-- The time it takes for the rider to reach the specified position -/
def time_to_position : ℝ := 70

theorem ferris_wheel_theorem :
  let R : ℝ := 30  -- radius in feet
  let ω : ℝ := Real.pi / 60  -- angular velocity in radians per second
  ferris_wheel_motion time_to_position = 20 ∧
  ferris_wheel_motion (time_to_position / 2) = 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_theorem_l1185_118521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1185_118503

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ :=
  (-2 + 2 * Real.cos θ, 2 * Real.sin θ)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * Real.sqrt 2 / Real.sin (θ + Real.pi / 4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_C₁_C₂ :
  ∃ (min_dist : ℝ),
    (∀ (θ₁ θ₂ : ℝ), distance (C₁ θ₁) (C₂ θ₂) ≥ min_dist) ∧
    (∃ (θ₁' θ₂' : ℝ), distance (C₁ θ₁') (C₂ θ₂') = min_dist) ∧
    min_dist = 3 * Real.sqrt 2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1185_118503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1185_118530

/-- The distance between foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: The distance between foci of an ellipse with semi-major axis 8 and semi-minor axis 3 is 2√55 -/
theorem ellipse_foci_distance :
  distance_between_foci 8 3 = 2 * Real.sqrt 55 := by
  unfold distance_between_foci
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1185_118530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_power_function_l1185_118568

-- Define the power function
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem tangent_line_of_power_function (α : ℝ) :
  f α (1/4) = 1/2 →
  let df := fun x ↦ α * x^(α-1)
  let m := df (1/4)
  let b := 1/2 - m * (1/4)
  4 * (1/4) - 4 * (1/2) + 1 = 0 ∧ ∀ x, 4*x - 4*(m*x + b) + 1 = 0 :=
by
  sorry

#check tangent_line_of_power_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_power_function_l1185_118568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1185_118522

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Side length a -/
  a : ℝ
  /-- Side length b -/
  b : ℝ
  /-- Side length c -/
  c : ℝ
  /-- Angle A in radians -/
  A : ℝ
  /-- Angle B in radians -/
  B : ℝ
  /-- Angle C in radians -/
  C : ℝ
  /-- Constraint: a = 3 -/
  ha : a = 3
  /-- Constraint: b = 2√6 -/
  hb : b = 2 * Real.sqrt 6
  /-- Constraint: B = 2A -/
  hB : B = 2 * A
  /-- Sum of angles in a triangle -/
  hSum : A + B + C = π
  /-- Law of sines -/
  hSine : a / Real.sin A = b / Real.sin B
  /-- Law of cosines -/
  hCosine : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- Main theorem about TriangleABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  Real.cos t.A = Real.sqrt 6 / 3 ∧ t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1185_118522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l1185_118571

-- Define a 3D space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define a line in the space
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Set V

-- Define parallel lines
def Parallel (l1 l2 : Line V) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (l1 l2 l3 : Line V) :
  Parallel l1 l3 → Parallel l2 l3 → Parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l1185_118571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1185_118533

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the foci
def leftFocus : ℝ × ℝ := (-3, 0)
def rightFocus : ℝ × ℝ := (3, 0)

-- Define the angle between PF₁ and PF₂
noncomputable def anglePF₁F₂ (P : ℝ × ℝ) : ℝ := Real.pi/3

-- Main theorem
theorem ellipse_properties (P : ℝ × ℝ) 
  (h_ellipse : ellipse P.1 P.2) 
  (h_angle : anglePF₁F₂ P = Real.pi/3) :
  (abs P.2 = 16 * Real.sqrt 3 / 9) ∧ 
  ((P.1 - leftFocus.1) * (P.1 - rightFocus.1) + 
   (P.2 - leftFocus.2) * (P.2 - rightFocus.2) = 32/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1185_118533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shekar_completion_time_l1185_118556

-- Define the work unit
noncomputable def W : ℝ := 1

-- Define the work rates for each person
noncomputable def malar_rate : ℝ := W / 60
noncomputable def malar_roja_rate : ℝ := W / 35

-- Define Roja's work rate in terms of Malar's
noncomputable def roja_rate : ℝ := malar_rate / 2

-- Define Shekar's work rate in terms of Malar's
noncomputable def shekar_rate : ℝ := malar_rate * 2

-- Theorem to prove
theorem shekar_completion_time :
  (W / shekar_rate) = 30 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shekar_completion_time_l1185_118556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1185_118502

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 1 * a 3 = 16)
  (h3 : a 2 + a 4 = 12) :
  q = Real.sqrt 2 ∨ q = -Real.sqrt 2 := by
  sorry

#check geometric_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1185_118502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_distances_l1185_118581

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Function to calculate distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Function to calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let a := distance t.A t.B
  let b := distance t.B t.C
  let c := distance t.C t.A
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Define membership for a point in a triangle -/
def Point.inTriangle (p : Point) (t : Triangle) : Prop :=
  let a := triangleArea (Triangle.mk p t.B t.C)
  let b := triangleArea (Triangle.mk t.A p t.C)
  let c := triangleArea (Triangle.mk t.A t.B p)
  let total := triangleArea t
  a + b + c ≤ total

theorem min_product_distances (ABC : Triangle) (D E F : Point) (RQS : Triangle) :
  (ABC.A.x = 0 ∧ ABC.A.y = 0) →
  (ABC.B.x = 4 ∧ ABC.B.y = 0) →
  (ABC.C.x = 2 ∧ ABC.C.y = 2 * Real.sqrt 3) →
  (D.x = 3 ∧ D.y = Real.sqrt 3) →
  (E.x = 1 ∧ E.y = Real.sqrt 3) →
  (F.x = 3 ∧ F.y = 0) →
  (RQS.A = D ∧ RQS.B = E ∧ RQS.C = F) →
  (∀ P : Point, P.inTriangle RQS →
    let x := triangleArea (Triangle.mk P ABC.B ABC.C) / distance ABC.B ABC.C
    let y := triangleArea (Triangle.mk P ABC.C ABC.A) / distance ABC.C ABC.A
    let z := triangleArea (Triangle.mk P ABC.A ABC.B) / distance ABC.A ABC.B
    x * y * z ≥ (648 / 2197) * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_distances_l1185_118581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapped_area_is_half_width_squared_l1185_118531

/-- Represents a rectangular piece of paper -/
structure Paper where
  width : ℝ
  length : ℝ

/-- The area of the overlapped region when the paper is folded -/
noncomputable def overlappedArea (p : Paper) : ℝ :=
  (p.width * p.width) / 2

/-- Theorem stating the area of the overlapped region -/
theorem overlapped_area_is_half_width_squared (w : ℝ) (h : w > 0) :
  let p : Paper := { width := w, length := 2 * w }
  overlappedArea p = w^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapped_area_is_half_width_squared_l1185_118531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_max_perimeter_theorem_l1185_118513

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 2 ∧ 2 * t.a * Real.cos t.B + Real.sqrt 3 * t.b = 2 * t.c

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.b * t.c * Real.sin t.A

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem for the area
theorem area_theorem (t : Triangle) (h : satisfiesConditions t) :
  area t = 2 * Real.sqrt 3 ∨ area t = Real.sqrt 3 := by sorry

-- Theorem for the maximum perimeter
theorem max_perimeter_theorem (t : Triangle) (h : satisfiesConditions t) :
  perimeter t ≤ 2 + 2 * (Real.sqrt 6 + Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_max_perimeter_theorem_l1185_118513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debby_candy_eaten_l1185_118591

/-- The number of candy pieces Debby ate -/
def candy_eaten (initial : ℕ) (received : ℕ) : ℕ :=
  (((2 : ℚ) / 3) * (initial + received)).floor.toNat

/-- Theorem stating that Debby ate 11 pieces of candy -/
theorem debby_candy_eaten :
  candy_eaten 12 5 = 11 := by
  -- Unfold the definition of candy_eaten
  unfold candy_eaten
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_debby_candy_eaten_l1185_118591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_to_l_l1185_118584

-- Define the necessary structures
structure Line
structure Plane

-- Define the perpendicular relation between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two lines
def perp_line_line (l1 l2 : Line) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line) : Prop := sorry

-- Define when a line is not contained in a plane
def not_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define intersecting planes
def intersect_planes (p1 p2 : Plane) : Prop := sorry

-- Define the line of intersection between two planes
def intersection_line (p1 p2 : Plane) : Line := sorry

-- State the theorem
theorem intersection_parallel_to_l 
  (m n l : Line) (α β : Plane) 
  (h1 : skew m n)
  (h2 : perp_line_plane m α)
  (h3 : perp_line_plane n β)
  (h4 : perp_line_line l m)
  (h5 : perp_line_line l n)
  (h6 : not_in_plane l α)
  (h7 : not_in_plane l β) :
  intersect_planes α β ∧ parallel_lines (intersection_line α β) l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_to_l_l1185_118584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l1185_118555

-- Define the propositions
def proposition1 : Prop := ∀ (p q : Prop), p ∧ ¬q → (p ∧ q)

noncomputable def proposition2 : Prop :=
  (∀ α : Real, α = Real.pi/6 → Real.sin α = 1/2) ∧
  ¬(∀ α : Real, Real.sin α = 1/2 → α = Real.pi/6)

def proposition3 : Prop :=
  (¬(∀ x : Real, (2 : Real)^x > 0)) ↔ (∃ x₀ : Real, (2 : Real)^x₀ ≤ 0)

-- The theorem to prove
theorem exactly_two_correct :
  (¬proposition1 ∧ proposition2 ∧ proposition3) ∨
  (proposition1 ∧ ¬proposition2 ∧ proposition3) ∨
  (proposition1 ∧ proposition2 ∧ ¬proposition3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l1185_118555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_three_meters_l1185_118558

/-- Represents the properties of a river --/
structure River where
  width : ℝ
  flowRate : ℝ
  volumePerMinute : ℝ

/-- Calculates the depth of a river based on its properties --/
noncomputable def calculateRiverDepth (r : River) : ℝ :=
  r.volumePerMinute / (r.width * (r.flowRate * 1000 / 60))

/-- Theorem stating that for a river with the given properties, its depth is 3 meters --/
theorem river_depth_is_three_meters :
  let r : River := { width := 36, flowRate := 2, volumePerMinute := 3600 }
  calculateRiverDepth r = 3 := by
  sorry

#check river_depth_is_three_meters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_three_meters_l1185_118558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_difference_l1185_118550

/-- The sequence of complex numbers as defined in the problem -/
noncomputable def a : ℕ → ℂ
  | 0 => 1 + Complex.I
  | n + 1 => a n * (1 + Complex.I / Real.sqrt (n + 2 : ℝ))

/-- The magnitude of a_n is equal to the square root of n+1 -/
axiom a_magnitude (n : ℕ) : Complex.abs (a n) = Real.sqrt (n + 1 : ℝ)

/-- The main theorem: the absolute difference between consecutive terms is always 1 -/
theorem a_difference (n : ℕ) : Complex.abs (a (n + 1) - a n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_difference_l1185_118550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_chord_length_circle_C₂_equation_l₁_always_intersects_C₁_shortest_chord_l₁_equation_l1185_118510

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the line parallel to the common chord
def common_chord_parallel (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Define the line l₁
def l₁ (lambda x y : ℝ) : Prop := 2*lambda*x - 2*y + 3 - lambda = 0

theorem circle_intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 :=
sorry

theorem circle_C₂_equation (E F : ℝ × ℝ) (hE : E = (1, -3)) (hF : F = (0, 4)) :
  ∃ (C₂ : ℝ → ℝ → Prop),
    C₂ E.1 E.2 ∧ C₂ F.1 F.2 ∧
    (∀ x y, C₂ x y ↔ x^2 + y^2 + 6*x - 16 = 0) ∧
    (∃ a b c, ∀ x y, (C₁ x y ∧ C₂ x y) → (a*x + b*y + c = 0 ∧ common_chord_parallel x y)) :=
sorry

theorem l₁_always_intersects_C₁ :
  ∀ lambda : ℝ, ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧ l₁ lambda P.1 P.2 ∧ l₁ lambda Q.1 Q.2 :=
sorry

theorem shortest_chord_l₁_equation :
  ∃ lambda : ℝ, ∀ x y, l₁ lambda x y ↔ x + y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_chord_length_circle_C₂_equation_l₁_always_intersects_C₁_shortest_chord_l₁_equation_l1185_118510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_variance_l1185_118570

/-- A set of n real numbers containing 0 and 1 -/
structure NumberSet (n : ℕ) where
  s : Finset ℝ
  card_eq : s.card = n
  zero_mem : 0 ∈ s
  one_mem : 1 ∈ s

/-- The variance of a set of numbers -/
noncomputable def variance (s : Finset ℝ) : ℝ :=
  let mean := s.sum id / s.card
  (s.sum (fun x => (x - mean)^2)) / s.card

/-- The optimal set achieving minimum variance -/
noncomputable def optimalSet (n : ℕ) : NumberSet n :=
  { s := sorry
  , card_eq := sorry
  , zero_mem := sorry
  , one_mem := sorry }

theorem minimum_variance (n : ℕ) (s : NumberSet n) :
  variance s.s ≥ 1 / (2 * n) ∧
  variance (optimalSet n).s = 1 / (2 * n) ∧
  ∀ x ∈ (optimalSet n).s, x = 0 ∨ x = 1 ∨ x = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_variance_l1185_118570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_equal_arcs_l1185_118505

/-- Two circles are equal if they have the same radius and center -/
structure EqualCircles (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  center₁ : α
  center₂ : α
  radius : ℝ
  radius_pos : radius > 0

/-- A central angle in a circle -/
structure CentralAngle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  circle : EqualCircles α
  angle : ℝ

/-- An arc in a circle -/
structure CircleArc (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  circle : EqualCircles α
  length : ℝ

/-- The theorem stating that in equal circles, equal central angles correspond to equal arcs -/
theorem equal_angles_equal_arcs 
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
  (c : EqualCircles α) 
  (θ₁ θ₂ : CentralAngle α) 
  (a₁ a₂ : CircleArc α)
  (h₁ : θ₁.circle = c)
  (h₂ : θ₂.circle = c)
  (h₃ : a₁.circle = c)
  (h₄ : a₂.circle = c)
  (h₅ : θ₁.angle = θ₂.angle)
  : a₁.length = a₂.length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_equal_arcs_l1185_118505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_thief_l1185_118545

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the property of being the thief
axiom is_thief : Person → Prop

-- Define the property of telling the truth
axiom tells_truth : Person → Prop

-- State the theorem
theorem jewelry_thief (h1 : ∃! p, tells_truth p) 
                      (h2 : ∃! p, is_thief p)
                      (h3 : tells_truth Person.A = ¬is_thief Person.A)
                      (h4 : tells_truth Person.B = is_thief Person.C)
                      (h5 : tells_truth Person.C = is_thief Person.D)
                      (h6 : tells_truth Person.D = ¬is_thief Person.D) :
  is_thief Person.A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_thief_l1185_118545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_l1185_118557

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define vector addition
def add_vectors (v w : MyVector) : MyVector :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalar_mult (c : ℝ) (v : MyVector) : MyVector :=
  (c * v.1, c * v.2)

-- Define parallel vectors
def parallel (v w : MyVector) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem vector_sum (m : ℝ) :
  let a : MyVector := (1, -2)
  let b : MyVector := (2, m)
  parallel a b →
  add_vectors (scalar_mult 3 a) (scalar_mult 2 b) = (7, -14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_l1185_118557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1185_118554

theorem trigonometric_identities (θ : ℝ) 
  (h1 : Real.sin θ = 3/5) 
  (h2 : π/2 < θ ∧ θ < π) : 
  Real.tan θ = -3/4 ∧ 
  Real.cos (2*θ - π/3) = (7 - 24*Real.sqrt 3) / 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1185_118554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_approx_l1185_118536

noncomputable section

-- Define the rectangle dimensions
def EF : ℝ := 4
def FG : ℝ := 6

-- Define the circle radii
def radius_E : ℝ := 2
def radius_F : ℝ := 3
def radius_G : ℝ := 4

-- Define π as a constant (approximation)
def π : ℝ := Real.pi

-- Define the area of the rectangle
def area_rectangle : ℝ := EF * FG

-- Define the area of the quarter circles
noncomputable def area_quarter_circles : ℝ := (π * radius_E^2 + π * radius_F^2 + π * radius_G^2) / 4

-- Define the area outside the circles
noncomputable def area_outside_circles : ℝ := area_rectangle - area_quarter_circles

-- Theorem statement
theorem area_outside_circles_approx :
  abs (area_outside_circles - 1.2) < 0.05 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_approx_l1185_118536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l1185_118519

/-- Arithmetic mean of two numbers -/
noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

/-- Geometric mean of two positive numbers -/
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

/-- Harmonic mean of two positive numbers -/
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 / (1/a + 1/b)

/-- Recursive definition of sequences A_n, G_n, and H_n -/
noncomputable def sequences (x z : ℝ) : ℕ → ℝ × ℝ × ℝ
  | 0 => (arithmetic_mean x z, geometric_mean x z, harmonic_mean x z)
  | n + 1 =>
    let (a, _, h) := sequences x z n
    (arithmetic_mean a h, geometric_mean a h, harmonic_mean a h)

theorem sequences_properties (x z : ℝ) (hx : x > 0) (hz : z > 0) (hxz : x ≠ z) :
  let a := fun n => (sequences x z n).1
  let g := fun n => (sequences x z n).2.1
  let h := fun n => (sequences x z n).2.2
  (∀ n, a (n + 1) < a n) ∧
  (∀ n, g (n + 1) = g n) ∧
  (∀ n, h n < h (n + 1)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l1185_118519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_is_unfair_l1185_118592

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow

/-- Represents the outcome of a single game -/
inductive GameOutcome
| XiaoYingWins
| XiaoMingWins

/-- The total number of balls in the bag -/
def totalBalls : ℚ := 3

/-- The number of red balls in the bag -/
def redBalls : ℚ := 2

/-- The number of yellow balls in the bag -/
def yellowBalls : ℚ := 1

/-- The probability of drawing a red ball -/
noncomputable def probRed : ℚ := redBalls / totalBalls

/-- The probability of drawing a yellow ball -/
noncomputable def probYellow : ℚ := yellowBalls / totalBalls

/-- The probability of Xiao Ying winning -/
noncomputable def probXiaoYingWins : ℚ := probRed * probRed + probYellow * probYellow

/-- The probability of Xiao Ming winning -/
noncomputable def probXiaoMingWins : ℚ := 1 - probXiaoYingWins

/-- Theorem stating that the game is unfair -/
theorem game_is_unfair : probXiaoYingWins ≠ probXiaoMingWins := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_is_unfair_l1185_118592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1185_118588

theorem remainder_theorem (a b c : ℕ) (ha : a > b) (hb : b > c)
  (hma : a % 12 = 7) (hmb : b % 12 = 4) (hmc : c % 12 = 2) (hc : c % 5 = 0) :
  (3 * a + 4 * b - 2 * c) % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1185_118588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_negative_x_l1185_118577

theorem eight_power_negative_x (x : ℝ) (h : (8 : ℝ)^(2*x) = 64) : (8 : ℝ)^(-x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_negative_x_l1185_118577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_n_value_l1185_118514

-- Define the equations of circles C and M
def circle_C (x y n : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + n = 0
def circle_M (x y : ℝ) : Prop := (x-3)^2 + y^2 = 1

-- Define the center of circle C
def center_C : ℝ × ℝ := (-1, 3)

-- Define the condition for circles being tangent
def are_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

-- Theorem statement
theorem tangent_circles_n_value :
  ∀ (n : ℝ),
  (∃ (x y : ℝ), circle_C x y n ∧ circle_M x y) →
  are_tangent center_C (3, 0) (Real.sqrt (10 - n)) 1 →
  n = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_n_value_l1185_118514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_combinations_exist_l1185_118589

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A quadrilateral defined by four grid points -/
structure Quadrilateral where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  d : GridPoint

/-- A function to check if a set of points forms a triangle -/
def isTriangle (points : Finset GridPoint) : Prop :=
  points.card = 3

/-- A function to check if a set of points forms a quadrilateral -/
def isQuadrilateral (points : Finset GridPoint) : Prop :=
  points.card = 4

/-- A function to check if a set of points forms a pentagon -/
def isPentagon (points : Finset GridPoint) : Prop :=
  points.card = 5

/-- The main theorem stating the existence of two quadrilaterals satisfying the conditions -/
theorem quadrilateral_combinations_exist :
  ∃ (q1 q2 : Quadrilateral),
    (∃ (triangle pentagon : Finset GridPoint),
      isTriangle triangle ∧
      isPentagon pentagon ∧
      (triangle ∪ pentagon : Set GridPoint) ⊆ {q1.a, q1.b, q1.c, q1.d, q2.a, q2.b, q2.c, q2.d}) ∧
    (∃ (triangle quad pentagon : Finset GridPoint),
      isTriangle triangle ∧
      isQuadrilateral quad ∧
      isPentagon pentagon ∧
      (triangle ∪ quad ∪ pentagon : Set GridPoint) ⊆ {q1.a, q1.b, q1.c, q1.d, q2.a, q2.b, q2.c, q2.d}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_combinations_exist_l1185_118589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1185_118578

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - a|
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x
noncomputable def M (a : ℝ) : ℝ := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-1) 1), f a x

theorem problem_solution :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x = 1) ∧
  (∀ a : ℝ, M a ≥ 1/2) ∧
  (M (1/2) = 1/2) ∧
  (∀ a : ℝ, (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 ∧ f a x + g a x = 0 ∧ f a y + g a y = 0) ↔ 1 ≤ a ∧ a ≤ 8/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1185_118578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_properties_l1185_118541

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Intersection points of a line and an ellipse -/
noncomputable def intersectionPoints (e : Ellipse) (l : Line) : Set (ℝ × ℝ) := sorry

/-- Definition of perpendicular vectors -/
def perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := sorry

/-- The perimeter of a triangle given its vertices -/
noncomputable def trianglePerimeter (A B C : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_properties (e : Ellipse) (l : Line) : 
  l.point = e.F₁ → 
  let intPoints := intersectionPoints e l
  ∀ A B, A ∈ intPoints → B ∈ intPoints → A ≠ B →
    (trianglePerimeter A B e.F₂ = 4 * Real.sqrt 2) ∧
    (perpendicular (A - e.F₂) (B - e.F₂) → triangleArea A B e.F₂ = 8/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_properties_l1185_118541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_to_directrix_l1185_118587

/-- An ellipse with semi-major axis 5 and semi-minor axis 3 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2/25 + y^2/9 = 1)

/-- The left focus of the ellipse -/
def left_focus : ℝ × ℝ := (-4, 0)

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (4, 0)

/-- The right directrix of the ellipse -/
noncomputable def right_directrix : ℝ := 25/4

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Distance from a point to the right directrix -/
noncomputable def distance_to_right_directrix (p : ℝ × ℝ) : ℝ :=
  |p.1 - right_directrix|

theorem ellipse_distance_to_directrix (M : Ellipse) :
  distance (M.x, M.y) left_focus = 8 →
  distance_to_right_directrix (M.x, M.y) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_to_directrix_l1185_118587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cylinder_transfer_l1185_118508

/-- Given a cylinder of height h that is 5/6 full of water, when poured into a new cylinder
    with radius 25% larger and height 72% of h, the fraction of the new cylinder filled
    with water is 20/27. -/
theorem water_cylinder_transfer (h : ℝ) (hpos : h > 0) : 
  let original_water_volume := (5/6 : ℝ) * Real.pi * h
  let new_cylinder_volume := Real.pi * ((5/4)^2 * h * (72/100))
  original_water_volume / new_cylinder_volume = 20/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cylinder_transfer_l1185_118508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_floor_eq_two_cos_squared_l1185_118565

noncomputable def floor_tan (x : ℝ) : ℤ := ⌊Real.tan x⌋

theorem tan_floor_eq_two_cos_squared (x : ℝ) :
  floor_tan x = ⌊(2 : ℝ) * (Real.cos x)^2⌋ ↔ ∃ k : ℤ, x = π / 4 + k * π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_floor_eq_two_cos_squared_l1185_118565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1185_118582

-- Define the domain of function f
def f : Set ℝ := Set.Icc (-12) 6

-- Define the function g in terms of f
def g (x : ℝ) : Prop := ∃ y ∈ f, y = 3 * x

-- Theorem statement
theorem domain_of_g : Set.Icc (-4) 2 = {x : ℝ | g x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1185_118582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_cartesian_to_polar_circle_cartesian_to_polar_l1185_118537

/-- Prove the equivalence of a line in Cartesian and polar coordinates -/
theorem line_cartesian_to_polar :
  ∀ (x y ρ : ℝ) (θ : ℝ),
  x + y = 0 ↔ (θ = 3 * π / 4 ∧ ρ ∈ Set.univ) :=
sorry

/-- Prove the equivalence of a circle in Cartesian and polar coordinates -/
theorem circle_cartesian_to_polar :
  ∀ (x y ρ : ℝ) (θ : ℝ) (a : ℝ),
  a ≠ 0 →
  x^2 + y^2 + 2*a*x = 0 ↔ ρ = -2*a*Real.cos θ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_cartesian_to_polar_circle_cartesian_to_polar_l1185_118537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_k_graph_l1185_118538

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1
noncomputable def g (x : ℝ) : ℝ := -x + 4
noncomputable def h : ℝ → ℝ := λ _ => 3
noncomputable def k (x : ℝ) : ℝ := min (min (f x) (g x)) (h x)

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- Define the length of the graph
noncomputable def graph_length : ℝ := Real.sqrt (86 + 2 * Real.sqrt 949)

-- Theorem statement
theorem length_of_k_graph :
  (∫ x in interval, Real.sqrt (1 + (deriv k x) ^ 2)) = graph_length := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_k_graph_l1185_118538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperjack_count_l1185_118548

/-- The number of pepperjack cheese sticks in Janet's multi-flavor pack. -/
def num_pepperjack : ℕ := 45

/-- The number of cheddar cheese sticks in the pack. -/
def num_cheddar : ℕ := 15

/-- The number of mozzarella cheese sticks in the pack. -/
def num_mozzarella : ℕ := 30

/-- The probability of picking a pepperjack cheese stick at random. -/
def prob_pepperjack : ℚ := 1/2

theorem pepperjack_count : 
  num_pepperjack = 45 ∧ 
  (num_pepperjack : ℚ) = (num_pepperjack + num_cheddar + num_mozzarella : ℚ) * prob_pepperjack := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperjack_count_l1185_118548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l1185_118585

/-- Represents a topping -/
inductive Topping
| pepperoni
| mushroom
| olive

/-- Represents whether a slice has a specific topping -/
def has (slice : ℕ) (topping : Topping) : Prop :=
  sorry -- Definition omitted for brevity

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) (olive_slices : ℕ)
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : olive_slices = 14)
  (h5 : ∀ slice, slice ∈ Finset.range total_slices → 
    ∃ topping, topping ∈ [Topping.pepperoni, Topping.mushroom, Topping.olive] ∧ has slice topping) :
  ∃ all_toppings_slices : ℕ, all_toppings_slices = 4 ∧
    ∀ slice, slice ∈ Finset.range total_slices →
      (has slice Topping.pepperoni ∧ has slice Topping.mushroom ∧ has slice Topping.olive) ↔ 
      slice ∈ Finset.range all_toppings_slices :=
by
  sorry -- Proof omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l1185_118585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l1185_118583

/-- The number of days A and B worked together -/
noncomputable def days_worked_together : ℝ := 2

/-- A's work rate per day -/
noncomputable def rate_A : ℝ := 1 / 5

/-- B's work rate per day -/
noncomputable def rate_B : ℝ := 1 / 10

/-- The number of days B worked alone to finish the remaining work -/
noncomputable def days_B_alone : ℝ := 4

/-- The total amount of work to be done -/
noncomputable def total_work : ℝ := 1

theorem work_completion :
  days_worked_together * (rate_A + rate_B) + days_B_alone * rate_B = total_work := by
  sorry

#check work_completion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l1185_118583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1185_118559

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the statements to be proved
theorem triangle_properties (t : Triangle) :
  -- Statement A
  (t.a / Real.sin t.A = (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C)) ∧
  -- Statement B (for oblique triangles)
  (t.A ≠ π/2 ∧ t.B ≠ π/2 ∧ t.C ≠ π/2 →
    Real.tan t.A + Real.tan t.B + Real.tan t.C = Real.tan t.A * Real.tan t.B * Real.tan t.C) ∧
  -- Statement D
  (t.a / Real.cos t.A = t.b / Real.cos t.B ∧ t.b / Real.cos t.B = t.c / Real.cos t.C →
    t.A = t.B ∧ t.B = t.C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1185_118559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1185_118527

theorem trigonometric_identities (α : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 ∧ Real.cos (π/2 - α) + Real.cos (3*π + α) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1185_118527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l1185_118525

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 9*m - 36) (m^2 - 2*m - 15)

theorem z_classification (m : ℝ) :
  ((z m).im = 0 ↔ m = -3 ∨ m = 5) ∧
  ((z m).im ≠ 0 ↔ m ≠ -3 ∧ m ≠ 5) ∧
  ((z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l1185_118525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l1185_118575

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_specific : 
  triangle_area 26 24 12 = Real.sqrt ((31 : ℝ) * (5 : ℝ) * (7 : ℝ) * (19 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l1185_118575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_after_13_throws_l1185_118523

/-- The number of elements in the circle -/
def n : ℕ := 13

/-- The number of elements skipped in each pass -/
def skip : ℕ := 4

/-- The function that determines the next position after a pass -/
def next (i : ℕ) : ℕ := (i + skip + 1) % n

/-- The sequence of positions starting from 0 -/
def ball_sequence : ℕ → ℕ
  | 0 => 0
  | i + 1 => next (ball_sequence i)

theorem ball_returns_after_13_throws :
  ball_sequence n = 0 ∧ ∀ i, i ∈ Finset.range n → ball_sequence i ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_after_13_throws_l1185_118523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1185_118517

/-- Sum of first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with common ratio q, if S_8, S_7, S_9 form an arithmetic sequence, then q = -2 -/
theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h : q ≠ 1) :
  let S := geometricSum a q
  2 * S 7 = S 8 + S 9 → q = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1185_118517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_k_eq_39_l1185_118544

/-- The sum of all positive integer values of k, where k is the sum of two integers whose product is 18 -/
def sum_of_k : ℕ :=
  (List.filter (λ k => k > 0) 
    (List.map (λ (a : Int × Int) => (a.1 + a.2).natAbs) 
      [(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1), 
       (-1, -18), (-2, -9), (-3, -6), (-6, -3), (-9, -2), (-18, -1)])).sum

theorem sum_of_k_eq_39 : sum_of_k = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_k_eq_39_l1185_118544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_three_l1185_118518

/-- Rectangle ABCD with points E and F on its sides -/
structure RectangleWithPoints where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- The distance of point E from A
  ae : ℝ
  -- The distance of point F from B
  bf : ℝ
  -- Ensure the rectangle has area 20
  area_eq : ab * bc = 20
  -- Ensure AE = 4
  ae_eq : ae = 4
  -- Ensure BF = 3
  bf_eq : bf = 3
  -- Ensure E and F are within the rectangle
  e_in_range : ae ≤ ab
  f_in_range : bf ≤ bc

/-- The area of trapezoid EFBA in the given rectangle -/
noncomputable def trapezoid_area (r : RectangleWithPoints) : ℝ :=
  ((r.ab + (r.bc - r.bf)) * (r.ab - r.ae)) / 2

/-- Theorem stating that the area of trapezoid EFBA is 3 square units -/
theorem trapezoid_area_is_three (r : RectangleWithPoints) : 
  trapezoid_area r = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_three_l1185_118518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1185_118512

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x) / (2^x + a*x)

-- State the theorem
theorem solve_for_a (a p q : ℝ) : 
  a > 0 → 
  f a p = 6/5 → 
  f a q = -1/5 → 
  2^(p+q) = 36*p*q → 
  a = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1185_118512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1185_118547

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 11 + y^2 / 7 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 4 = 1

-- Define the distance from foci to asymptotes
noncomputable def foci_to_asymptotes_distance : ℝ := Real.sqrt 2

-- Theorem statement
theorem hyperbola_equation 
  (h1 : ∀ x y, ellipse x y → (x = 2 ∨ x = -2) ∧ y = 0) 
  (h2 : ∀ x y, hyperbola x y → 
    let a := 2 -- semi-major axis
    let b := 2 -- semi-minor axis
    (a * b) / Real.sqrt (a^2 + b^2) = foci_to_asymptotes_distance) :
  ∀ x y, hyperbola x y ↔ x^2 / 4 - y^2 / 4 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1185_118547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_and_g_l1185_118594

noncomputable section

-- Function 1
def f (x : ℝ) : ℝ := 2 * x * Real.sin (2 * x + 5)

-- Function 2
noncomputable def g (x : ℝ) : ℝ := (x^3 - 1) / Real.sin x

theorem derivative_f_and_g :
  (∀ x, HasDerivAt f (2 * Real.sin (2 * x + 5) + 4 * x * Real.cos (2 * x + 5)) x) ∧
  (∀ x, x ≠ 0 → HasDerivAt g ((3 * x^2 * Real.sin x - (x^3 - 1) * Real.cos x) / (Real.sin x)^2) x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_and_g_l1185_118594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_angles_l1185_118540

theorem tan_sum_of_angles (α β : Real) 
  (h1 : Real.tan (α - π/6) = 3/7)
  (h2 : Real.tan (π/6 + β) = 2/5) :
  Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_angles_l1185_118540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zinc_copper_mixture_weight_l1185_118593

/-- Represents the weight of a metal mixture -/
structure MetalMixture where
  zinc : ℝ
  copper : ℝ

/-- Calculates the total weight of a metal mixture -/
def total_weight (m : MetalMixture) : ℝ := m.zinc + m.copper

/-- Represents the ratio of zinc to copper in a mixture -/
def zinc_copper_ratio : ℚ := 9 / 11

/-- The given weight of zinc in the mixture -/
def given_zinc_weight : ℝ := 27

/-- Theorem stating that for a zinc-copper mixture with a 9:11 ratio and 27 kg of zinc,
    the total weight is 60 kg -/
theorem zinc_copper_mixture_weight :
  ∃ (m : MetalMixture),
    m.zinc = given_zinc_weight ∧
    (m.zinc / m.copper : ℝ) = (zinc_copper_ratio : ℝ) ∧
    total_weight m = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zinc_copper_mixture_weight_l1185_118593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_sales_tax_percentage_l1185_118595

theorem aquarium_sales_tax_percentage 
  (original_price : ℝ) 
  (markdown_percentage : ℝ) 
  (total_cost_after_tax : ℝ) 
  (h1 : original_price = 120)
  (h2 : markdown_percentage = 50)
  (h3 : total_cost_after_tax = 63)
  : (total_cost_after_tax - (original_price * (1 - markdown_percentage / 100))) / 
    (original_price * (1 - markdown_percentage / 100)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_sales_tax_percentage_l1185_118595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_distance_l1185_118579

-- Define the curves and line
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.sin θ = -1
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, -2 + 2 * Real.sin θ)
def l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the intersection points in polar coordinates
noncomputable def intersection_points : Set (ℝ × ℝ) := {(2, -Real.pi/6), (2, 7*Real.pi/6)}

-- Define the maximum distance function
noncomputable def max_distance (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ := 
  2 * Real.sqrt 2 + 2

-- Theorem statement
theorem curves_intersection_and_max_distance :
  (∀ p, p ∈ intersection_points ↔ 
    (C₁ p.1 p.2 ∧ ∃ θ, C₂ θ = (p.1 * Real.cos p.2, p.1 * Real.sin p.2))) ∧
  (∀ θ, max_distance (C₂ θ) l = 2 * Real.sqrt 2 + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_distance_l1185_118579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_diff_sum_100_l1185_118572

theorem max_prime_diff_sum_100 :
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ≠ q ∧ 
    p + q = 100 ∧ 
    ∀ (r s : Nat), Nat.Prime r → Nat.Prime s → r ≠ s → r + s = 100 → 
      (q - p : Int) ≥ (s - r : Int) ∧
    q - p = 94 :=
by
  -- The proof goes here
  sorry

#check max_prime_diff_sum_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_diff_sum_100_l1185_118572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1185_118562

-- Define the line C3
noncomputable def C3 (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define the circle C2
def C2 (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + (y + 2)^2 = 1

-- Theorem statement
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ,
    C2 A.1 A.2 ∧ C2 B.1 B.2 ∧
    A.2 = C3 A.1 ∧ B.2 = C3 B.1 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1185_118562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_l1185_118511

theorem simplify_trigonometric_expression :
  (Real.sin (30 * π / 180))^3 + (Real.cos (30 * π / 180))^3 / 
  (Real.sin (30 * π / 180) + Real.cos (30 * π / 180)) = 1 - Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_l1185_118511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pattern_side_length_l1185_118515

inductive Color
| Black
| White

structure Tile :=
(color : Color)

theorem square_pattern_side_length 
  (total_tiles : ℕ) 
  (max_difference : ℕ) 
  (h_total : total_tiles = 95)
  (h_max_diff : max_difference = 85)
  (h_black_in_row : ∀ (row : List Tile), ∃ tile ∈ row, tile.color = Color.Black)
  (h_white_in_column : ∀ (column : List Tile), ∃ tile ∈ column, tile.color = Color.White) :
  ∃ (side_length : ℕ), side_length = 5 ∧ side_length * side_length = total_tiles :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pattern_side_length_l1185_118515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1185_118563

def angle_on_unit_circle (α : Real) : Prop :=
  ∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 4/5 ∧ y = 3/5 ∧
  Real.sin α = y ∧ Real.cos α = x ∧ Real.tan α = y / x

theorem trig_identities (α : Real) (h : angle_on_unit_circle α) :
  Real.sin α = 3/5 ∧ 
  Real.cos α = 4/5 ∧ 
  Real.tan α = 3/4 ∧ 
  (Real.sin (π + α) + 2 * Real.sin (π/2 - α)) / (2 * Real.cos (π - α)) = -25/64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1185_118563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_for_given_point_l1185_118546

/-- Linear regression model for weight based on height -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Calculate the predicted value given a LinearRegression model and an input -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.slope * x + model.intercept

/-- Calculate the residual given a LinearRegression model and a data point -/
def calculateResidual (model : LinearRegression) (x y : ℝ) : ℝ :=
  y - predict model x

/-- Theorem stating that the residual for the given data point is -0.79 -/
theorem residual_for_given_point :
  let model : LinearRegression := { slope := 0.85, intercept := -85.71 }
  let x : ℝ := 170
  let y : ℝ := 58
  calculateResidual model x y = -0.79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_for_given_point_l1185_118546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l1185_118580

/-- Given an ellipse with equation x²/m + y²/2 = 1, focus on the y-axis, and eccentricity 2/3, 
    the value of m is 10/9. -/
theorem ellipse_m_value (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, x^2/m + c^2/2 = 1) →  -- Focus on y-axis
  (let a := Real.sqrt 2
   let b := Real.sqrt m
   let c := Real.sqrt (a^2 - b^2)
   c / a = 2/3) →  -- Eccentricity is 2/3
  m = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l1185_118580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_quadratic_l1185_118520

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function h(t) = t^2 / 2 -/
noncomputable def h (t : ℝ) : ℝ := t^2 / 2

/-- Theorem: h is a quadratic function -/
theorem h_is_quadratic : is_quadratic h := by
  use (1/2 : ℝ), (0 : ℝ), (0 : ℝ)
  constructor
  · norm_num
  · intro x
    simp [h]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_quadratic_l1185_118520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l1185_118561

noncomputable def f (x : ℝ) := 3 - Real.sin x - 2 * (Real.cos x)^2

theorem f_max_min_difference :
  let a := π / 6
  let b := 7 * π / 6
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc a b, f x ≤ max ∧ min ≤ f x) ∧ max - min = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l1185_118561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1185_118598

/-- Given a triangle with circumradius R, inradius r, exradius r_a, semiperimeter p,
    side lengths a, b, c, and area S, prove the following inequalities -/
theorem triangle_inequalities (R r r_a p a b c S : ℝ) 
  (h_positive : R > 0 ∧ r > 0 ∧ r_a > 0 ∧ p > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0) 
  (h_semiperimeter : p = (a + b + c) / 2) 
  (h_area : S = Real.sqrt (p * (p - a) * (p - b) * (p - c))) : 
  (5 * R - r ≥ Real.sqrt 3 * p) ∧ 
  (4 * R - r_a ≥ (p - a) * (Real.sqrt 3 + (a^2 + (b - c)^2) / (2 * S))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1185_118598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_as_negative_457_l1185_118552

-- Define the set of angles with the same terminal side as -457°
def sameTerminalSide : Set ℝ :=
  {α : ℝ | ∃ k : ℤ, α = 263 + k * 360}

-- Theorem statement
theorem same_terminal_side_as_negative_457 :
  sameTerminalSide = {α : ℝ | ∃ k : ℤ, α = -457 + k * 360} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_as_negative_457_l1185_118552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1185_118586

theorem solve_exponential_equation (x : ℝ) :
  3 * (2:ℝ)^x + 2 * (2:ℝ)^(x+1) = 2048 → x = 11 - Real.log 7 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1185_118586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_f_to_g_transformation_l1185_118573

open Real

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * cos (x - π / 3) - 1

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 2 * cos (2 * x - 2 * π / 3) - 1

/-- Theorem stating that (π/12, -1) is a symmetry center of g(x) -/
theorem symmetry_center_of_g :
  ∀ x : ℝ, g (π / 6 - x) = g (π / 6 + x) := by
  sorry

/-- Theorem relating f and g -/
theorem f_to_g_transformation :
  ∀ x : ℝ, g x = f (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_f_to_g_transformation_l1185_118573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1185_118524

/-- Represents the properties of a train -/
structure Train where
  length : ℝ  -- length in meters
  speed : ℝ   -- speed in km/hr

/-- Calculates the time (in seconds) for two trains to cross each other -/
noncomputable def timeToCross (train1 train2 : Train) : ℝ :=
  let relativeSpeed := (train1.speed + train2.speed) * 1000 / 3600  -- Convert to m/s
  let combinedLength := train1.length + train2.length
  combinedLength / relativeSpeed

/-- Theorem stating the time for the given trains to cross each other -/
theorem trains_crossing_time : 
  let train1 := Train.mk 700 120
  let train2 := Train.mk 1000 80
  let crossingTime := timeToCross train1 train2
  |crossingTime - 30.58| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1185_118524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marys_income_percentage_l1185_118509

/-- Proves that Mary's income is 70% more than Tim's income given the conditions -/
theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.6)  -- Tim's income is 40% less than Juan's
  (h2 : mary = juan * 1.02)  -- Mary's income is 102% of Juan's
  : mary = tim * 1.7 := by  -- Mary's income is 70% more than Tim's
  sorry

#check marys_income_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marys_income_percentage_l1185_118509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_trajectory_l1185_118560

/-- The trajectory of the center of a circle that is tangent to x = 2,
    given a point A at (-2, 0) -/
theorem circle_center_trajectory :
  ∀ (x y : ℝ),
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 2)^2 + y^2 = r^2) ∧  -- Circle is tangent to x = 2
    ((x + 2)^2 + y^2 ≥ r^2))   -- Circle does not intersect point A (-2, 0)
  → y^2 = -8 * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_trajectory_l1185_118560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_is_31_l1185_118566

/-- The number of pillars to be painted -/
def num_pillars : ℕ := 20

/-- The height of each pillar in feet -/
def pillar_height : ℝ := 15

/-- The diameter of each pillar in feet -/
def pillar_diameter : ℝ := 8

/-- The area in square feet that one gallon of paint covers -/
def paint_coverage : ℝ := 250

/-- Calculate the minimum number of full gallons of paint needed -/
noncomputable def paint_gallons_needed : ℕ :=
  let radius := pillar_diameter / 2
  let single_pillar_area := 2 * Real.pi * radius * pillar_height
  let total_area := num_pillars * single_pillar_area
  Int.natAbs (Int.ceil (total_area / paint_coverage))

/-- Theorem stating the minimum number of full gallons of paint Alex needs to buy -/
theorem paint_needed_is_31 : paint_gallons_needed = 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_is_31_l1185_118566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_candy_percentage_l1185_118529

theorem halloween_candy_percentage (maggie_candy : ℕ) (neil_candy : ℕ) 
  (harper_percentage : ℚ) :
  maggie_candy = 50 →
  neil_candy = 91 →
  harper_percentage = 30 / 100 →
  let harper_candy := maggie_candy + (harper_percentage * maggie_candy).floor
  (((neil_candy : ℚ) - harper_candy) / harper_candy) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_candy_percentage_l1185_118529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspector_meters_examined_l1185_118506

theorem inspector_meters_examined (reject_rate : ℝ) (rejected_meters : ℕ) 
  (h1 : reject_rate = 0.10)
  (h2 : rejected_meters = 12) : ∃ total_meters : ℝ, 
  total_meters * reject_rate = rejected_meters ∧ total_meters = 120 := by
  let total_meters := rejected_meters / reject_rate
  have h3 : total_meters * reject_rate = rejected_meters := by
    sorry -- Proof steps would go here
  have h4 : total_meters = 120 := by
    sorry -- Proof steps would go here
  exact ⟨total_meters, h3, h4⟩

#check inspector_meters_examined

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspector_meters_examined_l1185_118506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_in_third_quadrant_l1185_118590

def angle_in_third_quadrant (A : ℝ) : Prop :=
  Real.sin A < 0 ∧ Real.cos A < 0

theorem cos_value_in_third_quadrant (A : ℝ) 
  (h1 : angle_in_third_quadrant A) 
  (h2 : Real.sin A = -5/8) : 
  Real.cos A = -Real.sqrt 39 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_in_third_quadrant_l1185_118590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mat8_paths_l1185_118597

/-- Represents a position on the grid -/
structure Position where
  x : ℕ
  y : ℕ
deriving Repr

/-- Represents a letter in the path -/
inductive Letter where
  | M
  | A
  | T
  | Eight
deriving Repr

/-- Represents the grid layout -/
def grid : List (List Letter) := sorry

/-- The starting position (central M) -/
def start : Position := ⟨2, 2⟩

/-- Checks if a move is valid (adjacent and within grid) -/
def is_valid_move (p q : Position) : Bool := sorry

/-- Counts the number of valid paths to spell MAT8 -/
def count_paths (grid : List (List Letter)) (start : Position) : ℕ := sorry

/-- The main theorem: there are 32 paths to spell MAT8 -/
theorem mat8_paths : count_paths grid start = 32 := by sorry

#eval start

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mat8_paths_l1185_118597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_pure_powers_l1185_118564

/-- A positive integer is a pure k-th power if it can be represented as m^k for some integer m. -/
def PureKthPower (k : ℕ) (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m ^ k

/-- For every positive integer n, there exist n distinct positive integers whose sum is a pure 2009-th power and whose product is a pure 2010-th power. -/
theorem exist_pure_powers (n : ℕ) : 
  ∃ (S : Finset ℕ), 
    Finset.card S = n ∧ 
    S.card > 0 ∧
    PureKthPower 2009 (Finset.sum S id) ∧ 
    PureKthPower 2010 (Finset.prod S id) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_pure_powers_l1185_118564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_ratio_l1185_118549

/-- Represents a cylinder formed by wrapping a rectangle --/
structure Cylinder where
  height : ℝ
  perimeter : ℝ

/-- The volume of the cylinder --/
noncomputable def volume (c : Cylinder) : ℝ :=
  let baseCircumference := c.perimeter - 2 * c.height
  let radius := baseCircumference / (2 * Real.pi)
  Real.pi * radius^2 * c.height

/-- The ratio of base circumference to height --/
noncomputable def baseCircumferenceToHeightRatio (c : Cylinder) : ℝ :=
  (c.perimeter - 2 * c.height) / c.height

theorem cylinder_max_volume_ratio (c : Cylinder) 
  (h_perimeter : c.perimeter = 12) 
  (h_max_volume : ∀ (c' : Cylinder), c'.perimeter = 12 → volume c' ≤ volume c) :
  baseCircumferenceToHeightRatio c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_ratio_l1185_118549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_l1185_118534

-- Define the vector type as a pair of real numbers
def MyVector := ℝ × ℝ

-- Define the dot product for our custom vector type
def dot_product (v w : MyVector) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector multiplication by a scalar
def scale_vector (r : ℝ) (v : MyVector) : MyVector := (r * v.1, r * v.2)

-- Define vector subtraction
def subtract_vectors (v w : MyVector) : MyVector := (v.1 - w.1, v.2 - w.2)

theorem orthogonal_vectors_k (k : ℝ) : 
  let a : MyVector := (1, -1)
  let b : MyVector := ((2 - k) / 2, (-2 * k - 3) / 2)
  (subtract_vectors a (scale_vector 2 b) = (k - 1, 2 * k + 2)) → 
  (dot_product a b = 0) →
  k = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_l1185_118534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_surface_area_l1185_118551

/-- Represents a right circular cone with vertices on a sphere -/
structure ConeSphere where
  R : ℝ  -- Radius of the sphere
  AB : ℝ  -- Side length of the equilateral triangle base

/-- The configuration satisfies the given conditions -/
def ValidConfiguration (cs : ConeSphere) : Prop :=
  cs.AB = Real.sqrt 3 * cs.R ∧
  (3 * Real.sqrt 3 / 4) * cs.R ^ 3 = 16 * Real.sqrt 3

/-- The surface area of the sphere -/
noncomputable def SphereArea (cs : ConeSphere) : ℝ :=
  4 * Real.pi * cs.R ^ 2

theorem cone_sphere_surface_area 
  (cs : ConeSphere) 
  (h : ValidConfiguration cs) : 
  SphereArea cs = 64 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_surface_area_l1185_118551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_median_and_mode_is_51_l1185_118599

/-- Represents a basketball player's scores -/
structure PlayerScores where
  scores : List Nat

/-- Calculate the median of a list of scores -/
def median (scores : List Nat) : Nat :=
  sorry -- Actual implementation would go here

/-- Calculate the mode of a list of scores -/
def mode (scores : List Nat) : Nat :=
  sorry -- Actual implementation would go here

/-- Theorem: The sum of player A's median and player B's mode is 51 -/
theorem sum_of_median_and_mode_is_51 (playerA playerB : PlayerScores) : 
  median playerA.scores + mode playerB.scores = 51 := by
  sorry -- Proof would go here

/-- The main result -/
def main : IO Unit := do
  IO.println s!"The sum of the median score of player A and the mode of player B's scores is 51"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_median_and_mode_is_51_l1185_118599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_3d_plus_15_l1185_118507

theorem divisors_of_3d_plus_15 (c d : ℤ) (h : 4 * d = 10 - 3 * c) :
  ∃ (S : Finset ℤ), S.card = 4 ∧ 
  (∀ x ∈ S, x > 0 ∧ x ≤ 5 ∧ (3 * d + 15) % x = 0) ∧
  (∀ y : ℤ, y > 0 ∧ y ≤ 5 ∧ y ∉ S → (3 * d + 15) % y ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_3d_plus_15_l1185_118507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1185_118500

/-- The speed of a train given its length and time to cross a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

theorem train_speed_calculation :
  let length : ℝ := 100
  let time : ℝ := 3.9996800255979523
  |train_speed length time - 90.003| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1185_118500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l1185_118596

theorem theta_value (θ : ℝ) (h1 : Real.cos (Real.pi + θ) = -2/3) (h2 : θ ∈ Set.Ioo (-Real.pi/2) 0) : 
  θ = -Real.arccos (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l1185_118596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_three_l1185_118535

theorem complex_expression_equals_three :
  Real.sqrt 18 / Real.sqrt 8 - (Real.sqrt 5 - 4) ^ (0 : ℤ) * (2 / 3) ^ (-1 : ℤ) + Real.sqrt ((-3) ^ 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_three_l1185_118535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_condition_l1185_118542

/-- A function f: ℝ → ℝ is decreasing on an interval I if for all x, y in I with x < y, f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x > f y

/-- The function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + 2*(a-1)*x + 2

/-- The interval (-∞, 4] -/
def I : Set ℝ := {x : ℝ | x ≤ 4}

theorem decreasing_function_condition (a : ℝ) : 
  DecreasingOn (f a) I → a < -3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_condition_l1185_118542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1185_118504

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem f_minimum_value :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1185_118504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l1185_118516

theorem periodic_decimal_to_fraction (x : ℝ) :
  (∃ n : ℕ, x = 19 + (87 / 99) * (1 / 10^n)) → ∃ a : ℕ, x = a / 99 ∧ a = 1968 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l1185_118516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_without_deriv_sign_change_l1185_118501

open Real Set

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then -x^2 * (sin (1/x) + 2) else 0

-- State the theorem
theorem exists_max_without_deriv_sign_change :
  ∃ (f : ℝ → ℝ), 
    (∀ x ∈ Ioo (-1) 1, DifferentiableAt ℝ f x) ∧ 
    (∀ x ∈ Ioo (-1) 1, f x ≤ f 0) ∧
    (∀ δ > 0, ∃ x₁ x₂, x₁ ∈ Ioo (-δ) δ ∧ x₂ ∈ Ioo (-δ) δ ∧ 
      deriv f x₁ > 0 ∧ deriv f x₂ < 0) :=
by
  -- Use the function f defined above
  use f
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_without_deriv_sign_change_l1185_118501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_speed_theorem_l1185_118539

/-- The speed at which John reaches his office 8 minutes earlier -/
noncomputable def early_speed : ℝ := 40

/-- The distance from John's house to his office in kilometers -/
noncomputable def distance : ℝ := 24

/-- The speed at which John arrives 4 minutes late in km/h -/
noncomputable def late_speed : ℝ := 30

/-- The time in hours that John is early when traveling at early_speed -/
noncomputable def early_time : ℝ := 8 / 60

/-- The time in hours that John is late when traveling at late_speed -/
noncomputable def late_time : ℝ := 4 / 60

/-- Theorem stating the relationship between travel times at different speeds -/
theorem john_speed_theorem :
  (distance / early_speed) + early_time = (distance / late_speed) - late_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_speed_theorem_l1185_118539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_impossible_roots_l1185_118526

/-- A quadratic polynomial with two distinct roots -/
structure QuadraticPolynomial where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  has_two_distinct_roots : ∃ α β : ℝ, α ≠ β ∧ f α = 0 ∧ f β = 0

/-- The number of distinct roots of an equation -/
noncomputable def num_distinct_roots (f : ℝ → ℝ) : ℕ :=
  Nat.card { x : ℝ | f x = 0 }

/-- Main theorem: It's impossible for f(f(x)) = 0 to have exactly 3 distinct roots
    and f(f(f(x))) = 0 to have exactly 7 distinct roots for a quadratic polynomial
    with two distinct roots -/
theorem quadratic_impossible_roots (q : QuadraticPolynomial) :
  ¬(num_distinct_roots (λ x ↦ q.f (q.f x)) = 3 ∧
    num_distinct_roots (λ x ↦ q.f (q.f (q.f x))) = 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_impossible_roots_l1185_118526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1185_118576

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 6)

theorem f_min_value (x : ℝ) (h : x ∈ Set.Icc (2 * Real.pi / 3) (5 * Real.pi / 4)) :
  f x ≥ -Real.sqrt 3 ∧ ∃ y ∈ Set.Icc (2 * Real.pi / 3) (5 * Real.pi / 4), f y = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1185_118576
