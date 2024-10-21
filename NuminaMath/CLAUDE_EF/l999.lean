import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_range_circumradius_range_open_l999_99981

-- Define a point on a parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

-- Define a triangle on a parabola
structure TriangleOnParabola where
  A : PointOnParabola
  B : PointOnParabola
  C : PointOnParabola
  distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C

-- Define the circumradius of a triangle
noncomputable def circumradius (t : TriangleOnParabola) : ℝ := sorry

-- Theorem statement
theorem circumradius_range (t : TriangleOnParabola) :
  circumradius t ∈ Set.Ioi (1/2 : ℝ) := by
  sorry

-- Additional theorem to show the range is open
theorem circumradius_range_open (t : TriangleOnParabola) :
  ∃ (r : ℝ), r > 1/2 ∧ circumradius t > r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_range_circumradius_range_open_l999_99981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l999_99941

def given_numbers : Set ℝ := {6, -3, 2.4, -3/4, 0, -3.14, 2, -7/2, 2/3}

def positive_numbers : Set ℝ := {x : ℝ | x ∈ given_numbers ∧ x > 0}
def non_negative_integers : Set ℤ := {x : ℤ | (x : ℝ) ∈ given_numbers ∧ x ≥ 0}
def negative_fractions : Set ℚ := {x : ℚ | (x : ℝ) ∈ given_numbers ∧ x < 0}

theorem number_categorization :
  positive_numbers = {6, 2.4, 2, 2/3} ∧
  non_negative_integers = {6, 0, 2} ∧
  negative_fractions = {-3/4, -3.14, -7/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l999_99941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_l999_99935

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 10) / (2*x + 2)

-- Define the domain
def domain (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 7

-- Theorem statement
theorem min_max_f :
  (∀ x, domain x → f x ≥ 11/3) ∧
  (∃ x, domain x ∧ f x = 11/3) ∧
  (∀ x, domain x → f x ≤ 87/16) ∧
  (∃ x, domain x ∧ f x = 87/16) := by
  sorry

#check min_max_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_l999_99935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_when_f_nonnegative_l999_99967

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (Real.exp x - a) - a^2 * x

-- Theorem for monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a = 0 → StrictMono (f a)) ∧
  (a > 0 → ∀ x y, x < y → y < Real.log a → f a y < f a x) ∧
  (a > 0 → ∀ x y, x < y → Real.log a < x → f a x < f a y) ∧
  (a < 0 → ∀ x y, x < y → y < Real.log (-a/2) → f a y < f a x) ∧
  (a < 0 → ∀ x y, x < y → Real.log (-a/2) < x → f a x < f a y) :=
sorry

-- Theorem for range of a when f(x) ≥ 0
theorem range_of_a_when_f_nonnegative (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) → a ∈ Set.Icc (-2 * Real.exp (3/4)) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_when_f_nonnegative_l999_99967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l999_99925

/-- Represents a hyperbola with equation x^2 - y^2/2 = 1 -/
structure Hyperbola where
  -- The equation is implicitly defined by the structure

/-- The asymptotes of the hyperbola -/
def asymptotes : Set (ℝ × ℝ) :=
  {(x, y) | y = 2*x ∨ y = -2*x}

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity : ℝ := Real.sqrt 3

/-- Theorem stating the properties of the given hyperbola -/
theorem hyperbola_properties :
  (asymptotes = {(x, y) | y = 2*x ∨ y = -2*x}) ∧
  (eccentricity = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l999_99925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l999_99960

-- Define the square ABCD
noncomputable def A : ℝ × ℝ := (0, 3)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (3, 0)
noncomputable def D : ℝ × ℝ := (3, 3)

-- Define midpoints E and F
noncomputable def E : ℝ × ℝ := (0, 1.5)
noncomputable def F : ℝ × ℝ := (1.5, 0)

-- Define intersection points I and H
noncomputable def I : ℝ × ℝ := (3/5, 9/5)
noncomputable def H : ℝ × ℝ := (9/4, 9/4)

-- Function to calculate area of a quadrilateral using Shoelace formula
noncomputable def quadrilateralArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  (1/2) * abs (x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_BEIH : quadrilateralArea B E I H = 27/200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l999_99960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l999_99948

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_gt_b : a > b

/-- The line passing through the left focus and a vertex of the ellipse -/
def line_through_focus_vertex (x y : ℝ) : Prop :=
  x - 2*y + 2 = 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem: If the line x - 2y + 2 = 0 passes through the left focus and one vertex
    of an ellipse with equation x²/a² + y²/b² = 1 (a > 0, b > 0, a > b),
    then its eccentricity is 2√5/5 -/
theorem ellipse_eccentricity_special_case (e : Ellipse) :
  (∃ (x y : ℝ), line_through_focus_vertex x y ∧ 
    x^2 / e.a^2 + y^2 / e.b^2 = 1) →
  eccentricity e = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l999_99948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l999_99966

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- First line equation: 3x - 4y + 2 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0

/-- Second line equation: 6x - my + 14 = 0 -/
def line2 (m x y : ℝ) : Prop := 6 * x - m * y + 14 = 0

/-- The lines are parallel -/
def parallel_lines (m : ℝ) : Prop := 3 / 4 = 6 / m

theorem distance_between_lines :
  ∀ m : ℝ, parallel_lines m →
  ∃ x y : ℝ, line1 x y ∧ line2 m x y ∧
  distance_parallel_lines 3 (-4) 2 7 = 1 :=
by
  sorry

#check distance_between_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l999_99966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l999_99918

noncomputable def original_expr : ℝ := (2 + Real.sqrt 5) / (3 - Real.sqrt 5)

noncomputable def rationalized_form (A B : ℚ) (C : ℕ) : ℝ := A + B * Real.sqrt (C : ℝ)

theorem rationalize_denominator :
  ∃ (A B : ℚ) (C : ℕ), 
    (rationalized_form A B C = original_expr) ∧ 
    (A * B * (C : ℚ) = 275 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l999_99918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_roof_angle_minimizes_slide_time_optimal_roof_angle_is_45_degrees_l999_99914

/-- The angle that minimizes the time for a raindrop to slide down a frictionless inclined plane -/
noncomputable def optimal_roof_angle : ℝ := Real.pi / 4

/-- The time taken for a raindrop to slide down the roof -/
noncomputable def slide_time (α : ℝ) (x : ℝ) (g : ℝ) : ℝ :=
  Real.sqrt ((4 * x) / (g * Real.sin (2 * α)))

theorem optimal_roof_angle_minimizes_slide_time (x : ℝ) (g : ℝ) (h_x : x > 0) (h_g : g > 0) :
  ∀ α : ℝ, α ∈ Set.Ioo 0 (Real.pi / 2) → 
    slide_time optimal_roof_angle x g ≤ slide_time α x g :=
by sorry

/-- The optimal roof angle is indeed π/4 (45 degrees) -/
theorem optimal_roof_angle_is_45_degrees :
  optimal_roof_angle = Real.pi / 4 :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_roof_angle_minimizes_slide_time_optimal_roof_angle_is_45_degrees_l999_99914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_ratio_l999_99985

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence is decreasing -/
def decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) : ℝ :=
  a 2 / a 1

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_decr : decreasing_sequence a)
  (h_prod : a 1 * a 5 = 9)
  (h_sum : a 2 + a 4 = 10) :
  common_ratio a = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_ratio_l999_99985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_result_l999_99991

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

-- State the theorem
theorem f_composition_equals_result : f (f (f (f (2 + I)))) = -12813107295652 - 1374662172928 * I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_result_l999_99991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_for_all_y_not_equal_l999_99945

theorem exists_x_for_all_y_not_equal :
  ∃ x : ℝ, ∀ y : ℝ, x * y^2 ≠ y^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_for_all_y_not_equal_l999_99945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l999_99976

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then (x - 1)^2 else 2/x

-- State the theorem
theorem monotonic_increase_interval :
  ∃ a b, a = 1 ∧ b = 2 ∧ StrictMonoOn f (Set.Icc a b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l999_99976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_distance_l999_99999

noncomputable def z : ℕ → ℂ
  | 0 => 0
  | n + 1 => (z n)^2 - Complex.I

theorem z_111_distance : Complex.abs (z 111) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_distance_l999_99999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l999_99920

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (4/3) * x^3 - 1/x

-- Define the derivative f'(x)
noncomputable def f' (x : ℝ) : ℝ := 4 * x^2 + 1/x^2

-- Theorem: The minimum value of f'(x) is 4
theorem min_value_f'_is_4 : ∀ x : ℝ, x ≠ 0 → f' x ≥ 4 :=
by
  intro x hx
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l999_99920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_trapezoid_l999_99937

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Trapezoid EFGH -/
structure TrapezoidEFGH where
  E : Point2D
  F : Point2D
  G : Point2D
  H : Point2D

/-- Definition of the specific trapezoid EFGH -/
def specificTrapezoid : TrapezoidEFGH := {
  E := { x := 0, y := 0 }
  F := { x := 0, y := 3 }
  G := { x := 3, y := 3 }
  H := { x := 6, y := 0 }
}

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : TrapezoidEFGH) : ℝ :=
  let base1 := t.G.x - t.F.x
  let base2 := t.H.x - t.E.x
  let height := t.F.y - t.E.y
  (base1 + base2) * height / 2

/-- Theorem: The area of the specific trapezoid EFGH is 13.5 square units -/
theorem area_of_specific_trapezoid :
  trapezoidArea specificTrapezoid = 13.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_trapezoid_l999_99937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_equals_22_l999_99927

-- Define the function g
noncomputable def g : ℝ → ℝ := fun y => 
  let x := (y + 7) / 3
  4 * x + 6

-- State the theorem
theorem g_of_5_equals_22 : g 5 = 22 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_equals_22_l999_99927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l999_99961

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l999_99961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_quadratic_equation_l999_99996

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic equation -/
noncomputable def roots (eq : QuadraticEquation) : Set ℝ :=
  {x : ℝ | eq.a * x^2 + eq.b * x + eq.c = 0}

/-- The sum of roots of a quadratic equation -/
noncomputable def sumOfRoots (eq : QuadraticEquation) : ℝ := -eq.b / eq.a

/-- The product of roots of a quadratic equation -/
noncomputable def productOfRoots (eq : QuadraticEquation) : ℝ := eq.c / eq.a

theorem correct_quadratic_equation :
  ∃ (eq : QuadraticEquation),
    eq.a = 1 ∧
    (∃ (eq1 : QuadraticEquation), 
      eq1.a = eq.a ∧ 
      eq1.b = eq.b ∧
      roots eq1 = {6, 3}) ∧
    (∃ (eq2 : QuadraticEquation),
      eq2.a = eq.a ∧
      eq2.c = eq.c ∧
      roots eq2 = {-12, -3}) →
    eq.b = -9 ∧ eq.c = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_quadratic_equation_l999_99996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l999_99929

-- Define the universal set I as ℝ
def I : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the set N
def N : Set ℝ := {x : ℝ | ∃ a : ℤ, |x - a| ≤ 1}

-- State the theorem
theorem intersection_M_N :
  (Set.compl M ∩ N = ∅) →
  (M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l999_99929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l999_99970

theorem square_side_length (perimeter : ℝ) (h : perimeter = 28) :
  perimeter / 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l999_99970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_3_l999_99917

-- Define the set of tickets
def tickets : Finset ℕ := Finset.range 20

-- Define the property of being a multiple of 3
def isMultipleOf3 (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Define a decidable version of isMultipleOf3
def isMultipleOf3Dec (n : ℕ) : Bool := n % 3 = 0

-- Define the set of tickets that are multiples of 3
def multiplesOf3 : Finset ℕ := tickets.filter (fun n => isMultipleOf3Dec n)

-- State the theorem
theorem probability_multiple_of_3 : 
  (multiplesOf3.card : ℚ) / tickets.card = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_3_l999_99917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_n_is_twice_prime_l999_99978

def S (n : ℕ+) : Set ℕ :=
  {a : ℕ | 1 < a ∧ a < n.val ∧ (n : ℕ) ∣ (a^(a-1) - 1)}

theorem prove_n_is_twice_prime (n : ℕ+) (h : S n = {(n : ℕ) - 1}) :
  ∃ p : ℕ, Nat.Prime p ∧ (n : ℕ) = 2 * p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_n_is_twice_prime_l999_99978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_T_l999_99924

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the perpendicular relationships
def perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define the distances
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- State the theorem
theorem distance_P_to_T (h1 : perpendicular P Q Q R)
                        (h2 : perpendicular Q R R S)
                        (h3 : perpendicular R S S T)
                        (h4 : distance P Q = 4)
                        (h5 : distance Q R = 8)
                        (h6 : distance R S = 8)
                        (h7 : distance S T = 3) :
  distance P T = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_T_l999_99924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_journey_properties_l999_99987

/-- Represents a segment of the cyclist's journey -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def segmentTime (s : Segment) : ℝ := s.distance / s.speed

/-- Represents the cyclist's journey -/
structure Journey where
  flat : Segment
  uphill : Segment
  downhill : Segment
  totalDistance : ℝ

/-- Calculates the total time of the journey -/
noncomputable def totalTime (j : Journey) : ℝ :=
  segmentTime j.flat + segmentTime j.uphill + segmentTime j.downhill

/-- Calculates the average speed of the journey -/
noncomputable def averageSpeed (j : Journey) : ℝ :=
  j.totalDistance / totalTime j

/-- Theorem stating the properties of the cyclist's journey -/
theorem cyclist_journey_properties (j : Journey) 
  (h1 : j.flat = { distance := 16, speed := 8 })
  (h2 : j.uphill = { distance := 12, speed := 6 })
  (h3 : j.downhill = { distance := 20, speed := 12 })
  (h4 : j.totalDistance = 48) : 
  totalTime j = 5.67 ∧ averageSpeed j = 48 / 5.67 := by
  sorry

#eval "Cyclist journey properties theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_journey_properties_l999_99987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l999_99934

/-- Given a parabola y² = 4x and a line y = 2x + b intersecting the parabola
    to form a chord of length 3√5, prove that b = -4 -/
theorem parabola_line_intersection (b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    y₁^2 = 4 * x₁ ∧ 
    y₁ = 2 * x₁ + b ∧
    y₂^2 = 4 * x₂ ∧ 
    y₂ = 2 * x₂ + b ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 45) →
  b = -4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l999_99934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_and_reciprocal_l999_99979

theorem sqrt_sum_and_reciprocal (x : ℝ) (h : x + x⁻¹ = 3) :
  Real.sqrt x + (Real.sqrt x)⁻¹ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_and_reciprocal_l999_99979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l999_99923

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem (x : ℝ) :
  x > 0 →
  compound_interest x 0.08 5 = 500 →
  ‖x - 340.28‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l999_99923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_club_mixed_groups_l999_99910

theorem chess_club_mixed_groups 
  (total_children : ℕ) 
  (total_groups : ℕ) 
  (children_per_group : ℕ) 
  (boy_vs_boy_games : ℕ) 
  (girl_vs_girl_games : ℕ) 
  (h1 : total_children = 90)
  (h2 : total_groups = 30)
  (h3 : children_per_group = 3)
  (h4 : total_children = total_groups * children_per_group)
  (h5 : boy_vs_boy_games = 30)
  (h6 : girl_vs_girl_games = 14)
  (h7 : ∀ n : ℕ, n = children_per_group → Nat.choose n 2 = 3)
  : ∃ mixed_groups : ℕ, mixed_groups = 23 ∧ 
    mixed_groups * 2 = total_groups * 3 - boy_vs_boy_games - girl_vs_girl_games :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_club_mixed_groups_l999_99910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_third_quadrant_l999_99908

open Real

theorem angle_third_quadrant (α : ℝ) 
  (h1 : 2 * (tan α)^2 - tan α - 1 = 0)
  (h2 : π < α ∧ α < 3*π/2) : 
  (2 * sin α - cos α) / (sin α + cos α) = -1/3 ∧ 
  cos α + sin α = -1/sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_third_quadrant_l999_99908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l999_99989

theorem sqrt_sum_equality : Real.sqrt 50 + Real.sqrt 32 + Real.sqrt 24 = 9 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l999_99989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l999_99915

noncomputable def π : ℝ := Real.pi

def equation (x : ℝ) : Prop :=
  Real.sin (π * (x^2 - x + 1)) = Real.sin (π * (x - 1))

def is_root (x : ℝ) : Prop :=
  equation x ∧ 0 ≤ x ∧ x ≤ 2

theorem sum_of_roots :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_root x) ∧
                    (∀ x, is_root x → x ∈ S) ∧
                    (S.sum id) = 3 + Real.sqrt 3 := by
  sorry

#check sum_of_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l999_99915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l999_99916

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (cos x) ^ 3 + (sin x) ^ 2 - cos x

-- State the theorem
theorem f_max_value : 
  ∃ (M : ℝ), M = 32/27 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l999_99916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2016_in_third_quadrant_l999_99953

def angle_in_third_quadrant (angle : ℝ) : Prop :=
  ∃ (k : ℤ) (θ : ℝ), angle = 360 * (k : ℝ) + θ ∧ 180 < θ ∧ θ < 270

theorem angle_2016_in_third_quadrant :
  angle_in_third_quadrant 2016 := by
  use 5
  use 216
  apply And.intro
  · norm_num
  · apply And.intro
    · norm_num
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2016_in_third_quadrant_l999_99953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_crossing_possible_l999_99992

-- Define the objects that need to be transported
inductive Object : Type
| Wolf : Object
| Goat : Object
| Cabbage : Object
deriving BEq, Repr

-- Define the banks of the river
inductive Bank : Type
| Left : Bank
| Right : Bank
deriving BEq, Repr

-- Define the state of the system
structure State where
  leftBank : List Object
  rightBank : List Object
  ferrymanBank : Bank
deriving Repr

-- Define a valid move
def validMove (s1 s2 : State) : Prop :=
  -- The ferryman changes banks
  s1.ferrymanBank ≠ s2.ferrymanBank ∧
  -- At most one object is moved
  (s1.leftBank.length + s1.rightBank.length) - (s2.leftBank.length + s2.rightBank.length) ≤ 1 ∧
  -- The move doesn't leave goat and cabbage alone
  ¬(s2.leftBank.contains Object.Goat ∧ s2.leftBank.contains Object.Cabbage ∧ s2.ferrymanBank = Bank.Right) ∧
  ¬(s2.rightBank.contains Object.Goat ∧ s2.rightBank.contains Object.Cabbage ∧ s2.ferrymanBank = Bank.Left) ∧
  -- The move doesn't leave goat and wolf alone
  ¬(s2.leftBank.contains Object.Goat ∧ s2.leftBank.contains Object.Wolf ∧ s2.ferrymanBank = Bank.Right) ∧
  ¬(s2.rightBank.contains Object.Goat ∧ s2.rightBank.contains Object.Wolf ∧ s2.ferrymanBank = Bank.Left)

-- Define the initial and final states
def initialState : State :=
  { leftBank := [Object.Wolf, Object.Goat, Object.Cabbage],
    rightBank := [],
    ferrymanBank := Bank.Left }

def finalState : State :=
  { leftBank := [],
    rightBank := [Object.Wolf, Object.Goat, Object.Cabbage],
    ferrymanBank := Bank.Right }

-- Theorem: There exists a sequence of valid moves from the initial state to the final state
theorem ferry_crossing_possible : ∃ (n : Nat) (seq : Fin (n + 1) → State),
  seq 0 = initialState ∧
  seq (Fin.last n) = finalState ∧
  ∀ i : Fin n, validMove (seq i) (seq (Fin.succ i)) :=
  sorry

#eval initialState
#eval finalState

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_crossing_possible_l999_99992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_has_12_sides_l999_99957

/-- The measure of an interior angle of a regular n-gon. -/
noncomputable def interior_angle_measure (n : ℕ) : ℝ :=
  180 * (n - 2) / n

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ, 
    n ≥ 3 →
    (∀ i : ℕ, i < n → interior_angle_measure n = 150) →
    n = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_has_12_sides_l999_99957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_l999_99940

noncomputable def line1 (x y : ℝ) : Prop := x + 3 * y = 0

noncomputable def line2 (x y : ℝ) : Prop := x + 3 * y + 2 = 0

noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

noncomputable def distance_to_line2 (x y : ℝ) : ℝ := |x + 3 * y + 2| / Real.sqrt 10

theorem equidistant_points : 
  ∃ (x y : ℝ), line1 x y ∧ 
    distance_to_origin x y = distance_to_line2 x y ∧
    ((x = -3/5 ∧ y = 1/5) ∨ (x = 3/5 ∧ y = -1/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_l999_99940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l999_99946

theorem min_value_sin_cos (x : ℝ) : 
  Real.sin x^4 + 2 * Real.cos x^4 ≥ 2/3 ∧ ∃ y : ℝ, Real.sin y^4 + 2 * Real.cos y^4 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l999_99946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l999_99964

/-- Calculates the total time for a train journey with three segments -/
theorem train_journey_time (x : ℝ) : x > 0 →
  (x / 50 + (2 * x) / 75 + (x / 2) / 30) = 19 * x / 300 := by
  intro hx
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l999_99964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_sum_of_roots_l999_99905

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial -/
def QuadraticPolynomial.eval (q : QuadraticPolynomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Sum of roots of a quadratic polynomial -/
noncomputable def QuadraticPolynomial.sumOfRoots (q : QuadraticPolynomial) : ℝ :=
  -q.b / q.a

theorem quadratic_polynomial_sum_of_roots
  (q : QuadraticPolynomial)
  (h : ∀ x : ℝ, q.eval (x^3 - x) ≥ q.eval (x^2 - 1)) :
  q.sumOfRoots = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_sum_of_roots_l999_99905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_prob_two_purses_l999_99902

/-- Represents the number of 5-pengo coins in each purse -/
def p : ℕ → ℕ := sorry

/-- Represents the number of 1-pengo coins in the first purse -/
def r₁ : ℕ → ℕ := sorry

/-- Represents the number of 1-pengo coins in the second purse -/
def r₂ : ℕ → ℕ := sorry

/-- Assumption that the number of 1-pengo coins is different in each purse -/
axiom h : ∀ n, r₁ n ≠ r₂ n

/-- The probability of drawing a 5-pengo coin when randomly selecting from two purses -/
noncomputable def prob_two_purses (n : ℕ) : ℚ :=
  (p n : ℚ) / 2 * ((2 * p n + r₁ n + r₂ n : ℚ) / ((p n + r₁ n) * (p n + r₂ n)))

/-- The probability of drawing a 5-pengo coin if all coins were in a single purse -/
noncomputable def prob_single_purse (n : ℕ) : ℚ :=
  (2 * p n : ℚ) / (2 * p n + r₁ n + r₂ n)

/-- Theorem stating that the probability of drawing a 5-pengo coin is higher when randomly selecting from two purses -/
theorem higher_prob_two_purses (n : ℕ) :
    prob_two_purses n > prob_single_purse n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_prob_two_purses_l999_99902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_absolute_value_l999_99933

theorem nested_absolute_value : ∀ x y z : ℝ, 
  (x = -1) → (y = 1) → (z = 1) → 
  abs (abs (abs (-(abs (x + y))) - z) + z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_absolute_value_l999_99933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l999_99998

theorem coefficient_x_cubed_expansion : ℤ := by
  let X : Polynomial ℚ := Polynomial.X
  let expansion := (1 + X)^5 - (1 + X)^4
  have h : expansion.coeff 3 = 6 := by sorry
  exact 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l999_99998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l999_99947

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the eccentricity
def eccentricity : ℝ := 1 / 2

-- Define the minor axis length
noncomputable def minor_axis_length : ℝ := 2 * Real.sqrt 3

-- Define the right focus F2
def F2 : ℝ × ℝ := (1, 0)

-- Define a line passing through F2
def line_through_F2 (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define the area of triangle F1AB
noncomputable def triangle_area (m : ℝ) : ℝ := 12 * Real.sqrt (m^2 + 1) / (3 * m^2 + 4)

-- The theorem to be proved
theorem max_triangle_area :
  ∃ (max_area : ℝ),
    (∀ m : ℝ, triangle_area m ≤ max_area) ∧
    (∃ m₀ : ℝ, triangle_area m₀ = max_area) ∧
    max_area = 3 ∧
    (∀ m : ℝ, triangle_area m = max_area → m = 0) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l999_99947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_l999_99980

/-- Calculates the compound interest amount after n years -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Calculates the simple interest amount after n years -/
def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate * (years : ℝ))

/-- The loan problem -/
theorem loan_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) 
  (h_principal : principal = 12000)
  (h_compound_rate : compound_rate = 0.08)
  (h_simple_rate : simple_rate = 0.09) :
  let compound_3 := compound_interest principal compound_rate 3
  let payment_3 := compound_3 / 3
  let remaining := compound_3 - payment_3
  let compound_12 := compound_interest remaining compound_rate 9 + payment_3
  let simple_12 := simple_interest principal simple_rate 12
  Int.floor (|compound_12 - simple_12| + 0.5) = 1731 := by
  sorry

#check loan_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_l999_99980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_imply_m_range_l999_99911

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - m * x + 8

theorem extreme_values_imply_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)) →
  m > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_imply_m_range_l999_99911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_implies_odd_l999_99963

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being invertible
def IsInvertible (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the property that the inverse of f(-x) is f^(-1)(-x)
def InverseProperty (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, IsInvertible f ∧ (∀ x, g (-x) = f⁻¹ (-x))

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem inverse_implies_odd (f : ℝ → ℝ) : 
  InverseProperty f → OddFunction f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_implies_odd_l999_99963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_max_area_line_equation_l999_99912

-- Define the ellipse parameters
variable (a b : ℝ) 

-- Define the point A and origin O
def A : ℝ × ℝ := (0, -2)
def O : ℝ × ℝ := (0, 0)

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the right focus F
variable (F : ℝ × ℝ)

-- Define the slope of line AF
noncomputable def slope_AF : ℝ := 2 * Real.sqrt 3 / 3

-- Define a line l passing through A
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x - 2

-- Define the area of triangle OPQ
noncomputable def area_OPQ (k : ℝ) : ℝ := 4 * Real.sqrt (4 * k^2 - 3) / (4 * k^2 + 1)

-- Theorem statements
theorem ellipse_equation : 
  a > 0 → b > 0 → a = 2 ∧ b = 1 := by sorry

theorem max_area_line_equation :
  let k₁ := Real.sqrt 7 / 2
  let k₂ := -Real.sqrt 7 / 2
  ∀ k, area_OPQ k ≤ area_OPQ k₁ ∧ area_OPQ k ≤ area_OPQ k₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_max_area_line_equation_l999_99912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l999_99975

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Definition of a triangle being equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Definition of sides being in geometric progression -/
def inGeometricProgression (a b c : Real) : Prop :=
  b^2 = a * c

theorem triangle_equilateral (t : Triangle) 
  (h1 : t.a * Real.cos t.C = t.c * Real.cos t.A)
  (h2 : inGeometricProgression t.a t.b t.c) : 
  isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l999_99975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_cosine_l999_99949

structure Triangle where
  baseAngle : Real
  vertexAngle : Real

def Triangle.isIsosceles (T : Triangle) : Prop := sorry

theorem isosceles_triangle_vertex_angle_cosine 
  (T : Triangle) 
  (h_isosceles : T.isIsosceles) 
  (h_base_angle : Real.cos T.baseAngle = 1/3) : 
  Real.cos T.vertexAngle = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_cosine_l999_99949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l999_99959

-- Define curve C1 in polar coordinates
noncomputable def C1 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define curve C2 in parametric form
def C2 (t : ℝ) : ℝ × ℝ :=
  (3 + 2 * t, t)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (θ t : ℝ),
    (∀ (θ' t' : ℝ),
      distance (C1 θ') (C2 t') ≥ distance (C1 θ) (C2 t)) ∧
    distance (C1 θ) (C2 t) = Real.sqrt 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l999_99959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_sqrt_4x_minus_2y_l999_99907

theorem square_root_of_sqrt_4x_minus_2y (x y : ℝ) :
  (2*x + 5*y + 4)^2 + |3*x - 4*y - 17| = 0 →
  ∃ z : ℝ, z^2 = Real.sqrt (4*x - 2*y) ∧ (z = 2 ∨ z = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_sqrt_4x_minus_2y_l999_99907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_tower_height_example_l999_99986

/-- Calculates the height of a scaled model tower given the original tower's dimensions and the model's volume -/
noncomputable def scaled_tower_height (original_height : ℝ) (original_volume : ℝ) (model_volume : ℝ) : ℝ :=
  original_height * (model_volume / original_volume) ^ (1/3 : ℝ)

/-- Theorem: The height of a scaled model tower with given parameters is 0.6 meters -/
theorem scaled_tower_height_example : 
  scaled_tower_height 60 150000 0.15 = 0.6 := by
  -- Unfold the definition of scaled_tower_height
  unfold scaled_tower_height
  -- Simplify the expression
  simp [Real.rpow_def]
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_tower_height_example_l999_99986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_vertex_exists_l999_99956

/-- A directed graph representing a transportation network between cities. -/
structure TransportNetwork where
  V : Type* -- Set of vertices (cities)
  E : V → V → Prop -- Edges (transportation routes)

/-- A path exists between two vertices in the graph. -/
def hasPath (G : TransportNetwork) (u v : G.V) : Prop :=
  ∃ (path : List G.V), path.head? = some u ∧ path.getLast? = some v ∧
    ∀ (i : Nat), i < path.length - 1 → G.E (path.get ⟨i, by sorry⟩) (path.get ⟨i + 1, by sorry⟩)

/-- All vertices are mutually reachable in the graph. -/
def allReachable (G : TransportNetwork) : Prop :=
  ∀ (u v : G.V), hasPath G u v

/-- There exists a vertex from which all other vertices are reachable. -/
def existsReachableVertex (G : TransportNetwork) : Prop :=
  ∃ (v : G.V), ∀ (u : G.V), hasPath G v u

/-- Main theorem: If all vertices are mutually reachable, then there exists a vertex
    from which all other vertices are reachable. -/
theorem reachable_vertex_exists (G : TransportNetwork) :
  allReachable G → existsReachableVertex G :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_vertex_exists_l999_99956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_percentage_after_addition_l999_99936

/-- Represents the composition of a kola solution -/
structure KolaSolution where
  total_volume : ℝ
  water_volume : ℝ
  kola_volume : ℝ
  sugar_volume : ℝ

/-- Calculates the percentage of a component in the solution -/
noncomputable def percentage (component_volume : ℝ) (total_volume : ℝ) : ℝ :=
  (component_volume / total_volume) * 100

/-- The theorem to be proved -/
theorem sugar_percentage_after_addition 
  (initial : KolaSolution)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (h1 : initial.total_volume = 340)
  (h2 : initial.water_volume = initial.total_volume * 0.64)
  (h3 : initial.kola_volume = initial.total_volume * 0.09)
  (h4 : initial.sugar_volume = initial.total_volume - initial.water_volume - initial.kola_volume)
  (h5 : added_sugar = 3.2)
  (h6 : added_water = 8)
  (h7 : added_kola = 6.8) :
  let new_solution := KolaSolution.mk 
    (initial.total_volume + added_sugar + added_water + added_kola)
    (initial.water_volume + added_water)
    (initial.kola_volume + added_kola)
    (initial.sugar_volume + added_sugar)
  ∃ ε > 0, |percentage new_solution.sugar_volume new_solution.total_volume - 26.54| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_percentage_after_addition_l999_99936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_bill_l999_99974

/-- The total bill for a group at Tom's Restaurant -/
def toms_restaurant_bill (num_adults num_children meal_cost : ℕ) : ℕ :=
  (num_adults + num_children) * meal_cost

/-- The specific bill for 2 adults and 5 children with $8 meals -/
theorem specific_bill : toms_restaurant_bill 2 5 8 = 56 :=
by
  -- Unfold the definition of toms_restaurant_bill
  unfold toms_restaurant_bill
  -- Evaluate the expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_bill_l999_99974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l999_99952

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 - Real.exp (-x)) / x

-- State the theorem
theorem f_properties :
  ∀ x > 0,
  (∀ y > 0, x < y → f x > f y) ∧
  f x > Real.exp (-x/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l999_99952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_90_proportionally_l999_99938

theorem divide_90_proportionally : 
  ∀ (x y z : ℚ), 
    x + y + z = 90 ∧
    x / 2 = y ∧ y / 2 = z →
    y = 16 + 4/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_90_proportionally_l999_99938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_common_multiple_remainder_l999_99962

theorem least_common_multiple_remainder (n : ℕ) : n = 25201 ↔ 
  (n > 1) ∧ 
  (∀ d : Fin 8, n % (d.val + 3) = 1) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ d : Fin 8, m % (d.val + 3) = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_common_multiple_remainder_l999_99962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_force_is_80_l999_99968

/-- Represents the force exerted by a pulley on its axis in a system with two masses. -/
noncomputable def pulley_force (m₁ m₂ g : ℝ) : ℝ :=
  let a := (m₂ - m₁) / (m₁ + m₂) * g
  let T := m₁ * (g + a)
  2 * T

/-- Theorem stating that the force exerted by the pulley on its axis is 80 N
    given the specified conditions. -/
theorem pulley_force_is_80 :
  pulley_force 3 6 10 = 80 := by
  -- Unfold the definition of pulley_force
  unfold pulley_force
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_force_is_80_l999_99968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_connection_theorem_specific_case_l999_99939

noncomputable def number_of_ways (n : ℕ) : ℚ := (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n)

theorem chord_connection_theorem (n : ℕ) :
  number_of_ways n = (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n) :=
by rfl

theorem specific_case : number_of_ways 10 = 16796 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_connection_theorem_specific_case_l999_99939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_in_U_l999_99931

def U : Set ℕ := {0, 1, 2, 3, 4}

def P : Set ℕ := {x : ℕ | x < 3}

theorem complement_of_P_in_U :
  (U \ P) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_in_U_l999_99931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_correct_l999_99932

/-- Represents the distance to a place given rowing speeds and time -/
noncomputable def distance_to_place (r c1 c2 t : ℝ) : ℝ :=
  (t * (r^2 - c1*c2)) / (2*r + c2 - c1)

/-- Theorem stating that the calculated distance is correct given the problem conditions -/
theorem distance_calculation_correct (r c1 c2 t : ℝ) 
  (hr : r > 0)
  (hc1 : c1 > 0)
  (hc2 : c2 > 0)
  (ht : t > 0)
  (hspeed : 2*r > c1 + c2) :
  let D := distance_to_place r c1 c2 t
  ∃ (T_against T_with : ℝ),
    T_against + T_with = t ∧
    D / (r - c1) = T_against ∧
    D / (r + c2) = T_with := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_correct_l999_99932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_122_l999_99993

/-- The repeating decimal 0.2323... -/
def repeating_decimal : ℚ := 23 / 99

/-- The sum of the numerator and denominator of the fraction representing 0.2323... in lowest terms -/
def sum_num_denom : ℕ := (repeating_decimal.num.natAbs) + (repeating_decimal.den)

theorem sum_is_122 : sum_num_denom = 122 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_122_l999_99993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_increase_l999_99909

theorem production_increase (n : ℕ) (prev_avg : ℝ) (new_avg : ℝ) (T : ℝ) : 
  n = 3 → 
  prev_avg = 70 → 
  new_avg = 75 → 
  T = (n + 1) * new_avg - n * prev_avg → 
  T = 90 := by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

#check production_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_increase_l999_99909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l999_99971

/-- Represents an ellipse with semi-major axis a, semi-minor axis b, and left focus at (-c, 0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_c_sq : c^2 = a^2 - b^2

/-- Point P(x, y) symmetric to F(-c, 0) about the line bx + cy = 0 -/
def symmetric_point (e : Ellipse) (x y : ℝ) : Prop :=
  y / (x + e.c) = e.c / e.b ∧ e.b * ((x - e.c) / 2) + (e.c * y / 2) = 0

/-- Point P(x, y) lies on the ellipse -/
def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  e.c / e.a

theorem ellipse_eccentricity (e : Ellipse) :
  ∃ x y : ℝ, symmetric_point e x y ∧ on_ellipse e x y →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l999_99971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l999_99994

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.b = 2 ∧ 
  Real.cos t.C = 3/4 ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 7/4

-- Theorem statement
theorem triangle_abc_properties (t : Triangle) 
  (h : triangle_properties t) : 
  t.a = 1 ∧ Real.sin (2 * t.A) = 5 * Real.sqrt 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l999_99994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l999_99919

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x : ℝ) (m : ℝ) : Prop := x = m

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the centroid condition
def is_centroid (F A B O : ℝ × ℝ) : Prop :=
  F.1 = (A.1 + B.1 + O.1) / 3 ∧ F.2 = (A.2 + B.2 + O.2) / 3

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The main theorem
theorem parabola_focus_distance (m : ℝ) (A B : ℝ × ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line A.1 m →
  line B.1 m →
  is_centroid focus A B origin →
  distance A focus = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l999_99919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_valid_paths_l999_99943

/-- Represents a point in the figure --/
inductive Point : Type
| A | B | C | D | E | F

/-- Represents a segment between two points --/
inductive Segment : Type
| mk : Point → Point → Segment

/-- The set of all valid segments in the figure --/
def validSegments : List Segment := [
  Segment.mk Point.A Point.C, Segment.mk Point.A Point.D,
  Segment.mk Point.C Point.B, Segment.mk Point.C Point.D, Segment.mk Point.C Point.F,
  Segment.mk Point.D Point.E, Segment.mk Point.D Point.F,
  Segment.mk Point.E Point.F,
  Segment.mk Point.F Point.B
]

/-- A path is a list of segments --/
def ValidPath := List Segment

/-- Checks if a path is valid according to the problem conditions --/
def isValidPath (p : ValidPath) : Bool :=
  sorry

/-- Counts the number of valid paths from A to B --/
def countValidPaths : Nat :=
  sorry

/-- The main theorem: there are exactly 10 valid paths from A to B --/
theorem ten_valid_paths : countValidPaths = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_valid_paths_l999_99943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thanksgiving_to_christmas_l999_99955

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

-- Define a function to add days to a given day of the week
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => addDays (nextDay start) n

theorem thanksgiving_to_christmas (thanksgiving : DayOfWeek) (days_between : Nat) :
  thanksgiving = DayOfWeek.Friday →
  days_between = 30 →
  addDays thanksgiving days_between = DayOfWeek.Sunday :=
by
  intro h1 h2
  rw [h1, h2]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

#eval addDays DayOfWeek.Friday 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thanksgiving_to_christmas_l999_99955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_c_is_four_l999_99977

/-- Represents a point along the river -/
structure Point where
  distance : ℝ

/-- Represents the river system -/
structure RiverSystem where
  A : Point
  B : Point
  C : Point
  v : ℝ  -- kayak speed in still water
  w : ℝ  -- river current speed

/-- The time taken to travel between two points -/
def travelTime (r : RiverSystem) (start finish : Point) : ℝ :=
  sorry

/-- The minimum distance of point C from the confluence -/
def minDistanceC (r : RiverSystem) : ℝ :=
  sorry

/-- Theorem stating the minimum distance of point C from the confluence -/
theorem min_distance_c_is_four (r : RiverSystem) 
  (hA : r.A.distance = 20)
  (hB : r.B.distance = 19)
  (hv : r.v > 0)
  (hw : r.w > 0)
  (hw_lt_v : r.w < r.v) :
  minDistanceC r = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_c_is_four_l999_99977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l999_99973

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 2^(x - b)

theorem range_of_f (b : ℝ) :
  (∀ x, 2 ≤ x → x ≤ 4 → f x b ≥ 1/2 ∧ f x b ≤ 2) ∧
  (∃ x, 2 ≤ x ∧ x ≤ 4 ∧ f x b = 1/2) ∧
  (∃ x, 2 ≤ x ∧ x ≤ 4 ∧ f x b = 2) ∧
  f 3 b = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l999_99973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_weight_is_4_l999_99944

/-- The weight of one carton of vegetables in kilograms -/
def carton_weight : ℚ := 3

/-- The number of crates in a load -/
def num_crates : ℕ := 12

/-- The number of cartons in a load -/
def num_cartons : ℕ := 16

/-- The total weight of a load in kilograms -/
def total_load_weight : ℚ := 96

/-- The weight of one crate of vegetables in kilograms -/
noncomputable def crate_weight : ℚ := (total_load_weight - num_cartons * carton_weight) / num_crates

theorem crate_weight_is_4 : crate_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_weight_is_4_l999_99944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equals_power_of_two_l999_99954

/-- Sequence y_k defined by recurrence relation -/
def y (m : ℕ) : ℕ → ℚ
  | 0 => 1
  | 1 => m
  | (k+2) => ((m + 1) * y m (k+1) + (m - k) * y m k) / (k + 2)

/-- Sum of all terms in the sequence -/
noncomputable def sequenceSum (m : ℕ) : ℝ := ∑' k, (y m k : ℝ)

theorem sequence_sum_equals_power_of_two (m : ℕ) : 
  sequenceSum m = (2 : ℝ)^(m + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_equals_power_of_two_l999_99954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_a_l999_99997

-- Define the complex number z and positive integer k
variable (z : ℂ) (k : ℕ)

-- Define the function f as the real part of z^n
def f (n : ℕ) : ℝ := (z ^ n).re

-- Define the parabola p
variable (a b c : ℝ)
def p (n : ℕ) : ℝ := a * n^2 + b * n + c

-- State the theorem
theorem largest_possible_a :
  (∃ (z : ℂ) (k : ℕ), k > 0 ∧
    (z^k).re > 0 ∧ (z^k).re ≠ 1 ∧
    (∀ n : Fin 4, f z n = p a b c n.val)) →
  (∀ a' : ℝ, a' ≤ Real.sqrt (1/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_a_l999_99997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_property_l999_99903

-- Define the triangle ABC
structure Triangle (α : Type*) [Field α] where
  A : α × α
  B : α × α
  C : α × α

-- Define the footpoint D
def footpoint (T : Triangle ℝ) : ℝ × ℝ := sorry

-- Define the centroid of a triangle
def centroid (T : Triangle ℝ) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (T : Triangle ℝ) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

theorem special_triangle_property (T : Triangle ℝ) 
  (h1 : length T.A T.B = 1)
  (h2 : let D := footpoint T
        centroid T = incenter {A := T.B, B := T.C, C := D}) :
  length T.A T.C = Real.sqrt (5/2) ∧ length T.B T.C = Real.sqrt (5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_property_l999_99903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imag_part_of_roots_l999_99951

theorem max_imag_part_of_roots (z : ℂ) : 
  z^6 - z^4 + z^2 - 1 = 0 → ∃ (root : ℂ), root^6 - root^4 + root^2 - 1 = 0 ∧ 
    ∀ (w : ℂ), w^6 - w^4 + w^2 - 1 = 0 → root.im ≥ w.im ∧
    root.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imag_part_of_roots_l999_99951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_problem_l999_99900

/-- Calculates the height of a cuboid given its width, length, and sum of all edges. -/
noncomputable def cuboid_height (width length sum_of_edges : ℝ) : ℝ :=
  (sum_of_edges - 4 * (width + length)) / 4

/-- Proves that a cuboid with width 30 cm, length 22 cm, and sum of all edges 224 cm has a height of 4 cm. -/
theorem cuboid_height_problem :
  cuboid_height 30 22 224 = 4 := by
  -- Unfold the definition of cuboid_height
  unfold cuboid_height
  -- Simplify the arithmetic expression
  simp [add_comm, mul_comm, mul_assoc]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_problem_l999_99900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l999_99982

def M : ℕ := 57^6 + 6*57^5 + 15*57^4 + 20*57^3 + 15*57^2 + 6*57 + 1

theorem number_of_factors_of_M : (Nat.divisors M).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l999_99982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l999_99965

-- Define the line
noncomputable def line (x y : ℝ) : Prop := y - x * Real.sqrt 2 + 4 = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x + 4

-- Define point Q
noncomputable def Q : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_theorem (C D : ℝ × ℝ) :
  line C.1 C.2 → line D.1 D.2 →
  parabola C.1 C.2 → parabola D.1 D.2 →
  C ≠ D →
  |distance C Q - distance D Q| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l999_99965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l999_99969

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l999_99969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_is_18_l999_99988

-- Define the type for digits (0 to 9)
def Digit := Fin 10

-- Define the problem conditions
def valid_arrangement (a b c d : Digit) : Prop :=
  -- Ensure all digits are distinct
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  -- Ensure the addition arrangement is satisfied
  (a.val + c.val : ℕ) % 10 = 1 ∧
  (b.val + c.val : ℕ) % 10 = 0 ∧
  (a.val + d.val : ℕ) % 10 = 1 ∧
  (a.val + b.val + c.val + d.val : ℕ) / 100 = 1

-- The theorem to prove
theorem digit_sum_is_18 (a b c d : Digit) :
  valid_arrangement a b c d → a.val + b.val + c.val + d.val = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_is_18_l999_99988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_correct_l999_99950

/-- Represents a tetrahedron with one equilateral face of side length 1 and three edges of length a -/
structure Tetrahedron where
  a : ℝ
  h : a > Real.sqrt 3 / 3

/-- The maximum area of the orthogonal projection of the tetrahedron onto a plane -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  if t.a ≤ Real.sqrt 3 / 2 then
    Real.sqrt 3 / 4
  else
    t.a / 2

theorem max_projection_area_correct (t : Tetrahedron) :
  max_projection_area t =
    if t.a ≤ Real.sqrt 3 / 2 then
      Real.sqrt 3 / 4
    else
      t.a / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_correct_l999_99950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_eq_neg_sqrt_3_l999_99942

/-- The sequence a_n is defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Add this case to handle Nat.zero
  | 1 => 0
  | n + 2 => (a (n + 1) - Real.sqrt 3) / (Real.sqrt 3 * a (n + 1) + 1)

/-- The 2018th term of the sequence equals -√3 -/
theorem a_2018_eq_neg_sqrt_3 : a 2018 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_eq_neg_sqrt_3_l999_99942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mastery_not_related_to_gender_expectation_X_is_nine_fifths_l999_99926

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  !![25, 33; 15, 27]

-- Define the K² formula
def k_squared (n : ℕ) (m : Matrix (Fin 2) (Fin 2) ℕ) : ℚ :=
  let a := m 0 0
  let b := m 0 1
  let c := m 1 0
  let d := m 1 1
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value
def critical_value : ℚ := 3841 / 1000

-- Define the hypergeometric distribution
def hypergeometric (N M n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose M k * Nat.choose (N - M) (n - k) : ℚ) / Nat.choose N n

-- Define the expectation of X
noncomputable def expectation_X : ℚ :=
  Finset.sum (Finset.range 4) (λ k => k * hypergeometric 10 6 3 k)

-- Theorem statements
theorem mastery_not_related_to_gender :
  k_squared 100 contingency_table < critical_value :=
by sorry

theorem expectation_X_is_nine_fifths :
  expectation_X = 9 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mastery_not_related_to_gender_expectation_X_is_nine_fifths_l999_99926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_a_l999_99901

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∀ x : ℝ, (x + 2) / 3 > x / 2 + 1 ∧ 4 * x + a < x - 1 ↔ x < -2) ∧
    (∃ y : ℝ, y > 0 ∧ y ≠ 1 ∧ (a + 2) / (y - 1) + (y + 2) / (1 - y) = 2)) ∧
  (∀ a : ℤ, 
    ((∀ x : ℝ, (x + 2) / 3 > x / 2 + 1 ∧ 4 * x + a < x - 1 ↔ x < -2) ∧
    (∃ y : ℝ, y > 0 ∧ y ≠ 1 ∧ (a + 2) / (y - 1) + (y + 2) / (1 - y) = 2))
    → a ∈ S) ∧
  S.sum id = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_a_l999_99901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l999_99972

-- Define the quadrilateral vertices
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 3)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (3, 0)

-- Define the quadrilateral
def quadrilateral : List (ℝ × ℝ) := [A, B, C, D]

-- Helper function to calculate the area
noncomputable def area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area : 
  area quadrilateral = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l999_99972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_plane_not_parallel_l999_99906

-- Define the basic types
variable (P : Type) -- Type for points
variable (L : Type) -- Type for lines
variable (M : Type) -- Type for planes (changed from Π to avoid Unicode issues)

-- Define the relationships
variable (parallel : L → L → Prop) -- Parallel relation between lines
variable (parallel_plane : L → M → Prop) -- Parallel relation between a line and a plane
variable (intersects : L → M → Prop) -- Intersection relation between a line and a plane
variable (in_plane : L → M → Prop) -- Relation for a line being in a plane

-- Theorem statement
theorem line_intersects_plane_not_parallel 
  (a : L) (m : M) : 
  intersects a m → ∀ b : L, in_plane b m → ¬ parallel a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_plane_not_parallel_l999_99906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l999_99904

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 9 = -1

-- Define the latus rectum of the parabola
noncomputable def latus_rectum : ℝ := -3

-- Define the asymptotes of the hyperbola
noncomputable def asymptote_slope : ℝ := Real.sqrt 3 / 3

-- Theorem statement
theorem triangle_area :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- The three points form a triangle
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧ (x₃ ≠ x₁ ∨ y₃ ≠ y₁) ∧
    -- One point is on the latus rectum
    x₁ = latus_rectum ∧
    -- One point is on the x-axis (origin)
    x₂ = 0 ∧ y₂ = 0 ∧
    -- One point is on the asymptote
    y₃ = asymptote_slope * x₃ ∧
    -- The area of the triangle is 3√3
    abs ((x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - y₁ * x₂ - y₂ * x₃ - y₃ * x₁) / 2) = 3 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l999_99904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_in_seat_two_l999_99984

-- Define the seats
inductive Seat
| one
| two
| three
| four

-- Define the people
inductive Person
| Alice
| Ben
| Cindy
| Dan

-- Define the seating arrangement
def seating : Person → Seat := sorry

-- Define Kate's statements
def statement1 : Prop := ∃ (s1 s2 : Seat), s1 ≠ s2 ∧ 
  ((seating Person.Ben = s1 ∧ seating Person.Cindy = s2) ∨ 
   (seating Person.Ben = s2 ∧ seating Person.Cindy = s1))

def statement2 : Prop := ∃ (s1 s2 s3 : Seat), s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
  seating Person.Alice = s2 ∧ 
  ((seating Person.Ben = s1 ∧ seating Person.Cindy = s3) ∨ 
   (seating Person.Ben = s3 ∧ seating Person.Cindy = s1))

def statement3 : Prop := seating Person.Cindy ≠ Seat.three

-- Theorem statement
theorem dan_in_seat_two :
  (seating Person.Alice = Seat.three) →
  (statement1 ∨ statement2 ∨ statement3) →
  ¬(statement1 ∧ statement2) →
  ¬(statement1 ∧ statement3) →
  ¬(statement2 ∧ statement3) →
  (seating Person.Dan = Seat.two) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_in_seat_two_l999_99984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_return_speed_l999_99922

/-- Proves that given a round trip of 80 km (40 km each way) and a total travel time of 6 hours,
    where the return speed is twice the outbound speed, the return speed is 20 km/h. -/
theorem steves_return_speed (distance_one_way : ℝ) (total_time : ℝ) (speed_ratio : ℝ) 
    (x : ℝ) : -- x represents Steve's speed on the way to work
  distance_one_way = 40 →
  total_time = 6 →
  speed_ratio = 2 →
  distance_one_way / x + distance_one_way / (speed_ratio * x) = total_time →
  speed_ratio * x = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_return_speed_l999_99922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l999_99983

-- Define the function f with domain (-1, 1)
def f : ℝ → Set ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-1) 1

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : Set ℝ := f (2 * x - 1)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ≠ ∅} = Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l999_99983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_79_l999_99921

/-- Represents the cost of tickets for a family concert outing -/
structure ConcertTickets where
  regular_price : ℚ
  senior_price : ℚ
  student_price : ℚ
  num_regular : ℕ
  num_senior : ℕ
  num_student : ℕ
  senior_discount : senior_price = regular_price * (70 / 100)
  student_discount : student_price = regular_price * (60 / 100)
  given_senior_price : senior_price = 7
  num_regular_is_4 : num_regular = 4
  num_senior_is_3 : num_senior = 3
  num_student_is_3 : num_student = 3

/-- The total cost of all tickets is 79 -/
theorem total_cost_is_79 (tickets : ConcertTickets) :
  tickets.regular_price * tickets.num_regular +
  tickets.senior_price * tickets.num_senior +
  tickets.student_price * tickets.num_student = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_79_l999_99921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_players_same_group_probability_l999_99913

/-- The number of players in the team -/
def total_players : ℕ := 5

/-- The number of players in the larger group -/
def larger_group : ℕ := 3

/-- The number of players in the smaller group -/
def smaller_group : ℕ := 2

/-- The probability of two specific players being in the same group -/
def probability_same_group : ℚ := 2/5

theorem two_players_same_group_probability :
  (total_players = larger_group + smaller_group) →
  (probability_same_group = (Nat.choose (total_players - 2) (larger_group - 2) + Nat.choose (total_players - 2) (smaller_group - 2)) / Nat.choose total_players larger_group) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_players_same_group_probability_l999_99913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_of_tans_equals_60_degrees_l999_99928

-- Define the necessary trigonometric values
noncomputable def tan30 : ℝ := 1 / Real.sqrt 3
noncomputable def tan15 : ℝ := 2 - Real.sqrt 3

-- Define the main theorem
theorem arctan_of_tans_equals_60_degrees :
  Real.arctan (1 / tan15 - 3 * tan30 - tan15) = 60 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_of_tans_equals_60_degrees_l999_99928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_not_parabola_l999_99990

-- Define the equation
def equation (x y θ : ℝ) : Prop := x^2 + y^2 * Real.sin θ = 4

-- Define what it means for the equation to represent a parabola
def represents_parabola (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, eq x y ↔ (a * x^2 + b * x * y + c * y^2 + d * x + e * y = 0)

-- Theorem statement
theorem equation_not_parabola : ¬∃ θ : ℝ, represents_parabola (λ x y => equation x y θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_not_parabola_l999_99990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_is_integer_l999_99958

/-- Definition of Q_n(x) -/
def Q (n : ℕ) (x : ℤ) : ℚ :=
  (Finset.range n).prod (fun i => (x - ↑i)) / n.factorial

/-- Theorem stating that Q_n(x) is an integer for all natural numbers n and integers x -/
theorem Q_is_integer (n : ℕ) (x : ℤ) : ∃ k : ℤ, Q n x = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_is_integer_l999_99958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sign_and_product_negative_l999_99930

theorem opposite_sign_and_product_negative (a b : ℝ) : 
  (Real.sqrt (a^3 + 8) + |b^2 - 9| = 0) → 
  (a * b < 0) → 
  (b - a)^a = 1/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sign_and_product_negative_l999_99930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_satisfying_union_condition_l999_99995

theorem sets_satisfying_union_condition : 
  ∃ (s : Finset (Set ℕ)), s = {A : Set ℕ | {1, 2} ∪ A = {1, 2, 3}} ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_satisfying_union_condition_l999_99995
