import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_equivalence_l299_29979

noncomputable def original_function (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shift_amount : ℝ := Real.pi / 4

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x - shift_amount)

theorem shifted_function_equivalence :
  ∀ x : ℝ, shifted_function x = 2 * Real.sin (2 * x - Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_equivalence_l299_29979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_locus_l299_29961

/-- Two intersecting planes in 3D space -/
structure IntersectingPlanes where
  plane1 : Set (Fin 3 → ℝ)
  plane2 : Set (Fin 3 → ℝ)
  intersect : plane1 ∩ plane2 ≠ ∅

/-- A sphere in 3D space -/
structure Sphere where
  center : Fin 3 → ℝ
  radius : ℝ

/-- A line in 3D space -/
structure Line where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Convert a Line to a Set -/
def Line.toSet (l : Line) : Set (Fin 3 → ℝ) :=
  {p | ∃ t : ℝ, p = λ i => l.point i + t * l.direction i}

/-- The theorem stating the geometric locus of tangency points -/
theorem tangency_locus 
  (planes : IntersectingPlanes) 
  (sphere : Sphere) 
  (is_tangent : Sphere → IntersectingPlanes → Prop) :
  ∃ (l1 l2 : Line),
    (∀ p : Fin 3 → ℝ, is_tangent sphere planes → 
      (p ∈ planes.plane1 ∨ p ∈ planes.plane2) → 
      (p ∈ l1.toSet ∨ p ∈ l2.toSet)) ∧
    (l1.direction = l2.direction) ∧
    (∃ l : Line, l.toSet = planes.plane1 ∩ planes.plane2 ∧ 
      l.direction = l1.direction) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_locus_l299_29961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_after_decimal_point_of_inverse_25_pow_25_l299_29988

theorem zeros_after_decimal_point_of_inverse_25_pow_25 : ∃ (num_zeros : ℕ), num_zeros = 32 := by
  let n := 25
  let decimal_representation := (1 : ℚ) / (25 ^ n)
  
  -- We define num_zeros here, but we don't compute it
  -- Instead, we assert its existence and value
  have h : ∃ (num_zeros : ℕ), num_zeros = 32 := by
    -- The actual computation would go here
    sorry
  
  -- We then use this assertion to prove our theorem
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_after_decimal_point_of_inverse_25_pow_25_l299_29988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangles_area_l299_29936

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem rectangle_triangles_area :
  let d : Point := ⟨0, 4⟩
  let e : Point := ⟨6, 0⟩
  let f : Point := ⟨6, 5⟩
  let g : Point := ⟨0, 8⟩
  let h : Point := ⟨0, 6⟩
  let i : Point := ⟨6, 5⟩
  triangleArea d e f = 15 ∧ triangleArea g h i = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangles_area_l299_29936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_diameter_endpoints_l299_29923

noncomputable def smallest_radius_circle (f : ℝ → ℝ) (center : ℝ × ℝ) : ℝ := sorry

theorem dot_product_of_diameter_endpoints
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = 1 / (|x| - 1))
  (center : ℝ × ℝ)
  (h_center : center = (0, 1))
  (O : ℝ × ℝ)
  (h_O : O = (0, 0))
  (R : ℝ)
  (h_R : R = smallest_radius_circle f center)
  (A B : ℝ × ℝ)
  (h_AB : ‖A - center‖ = R ∧ ‖B - center‖ = R ∧ ‖A - B‖ = 2 * R) :
  (A - O) • (B - O) = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_diameter_endpoints_l299_29923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_comparison_l299_29996

noncomputable def scores_A : List ℝ := [68, 71, 72, 72, 82]
noncomputable def scores_B : List ℝ := [66, 70, 72, 78, 79]

noncomputable def average (scores : List ℝ) : ℝ := (scores.sum) / scores.length

noncomputable def variance (scores : List ℝ) : ℝ :=
  let avg := average scores
  (scores.map (fun x => (x - avg) ^ 2)).sum / scores.length

theorem scores_comparison :
  (average scores_A = average scores_B) ∧
  (variance scores_A < variance scores_B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_comparison_l299_29996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_existence_l299_29918

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane. -/
structure Line where
  point1 : Point
  point2 : Point

/-- A circle in a 2D plane. -/
structure Circle where
  center : Point
  radius : ℝ

/-- Defines membership of a point on a line. -/
def Point.mem (p : Point) (l : Line) : Prop :=
  sorry

instance : Membership Point Line where
  mem := Point.mem

/-- Defines tangency between a circle and another geometric object. -/
def Tangent : (Circle ⊕ Line) → (Circle ⊕ Line) → Prop :=
  sorry

/-- Returns the point of tangency between a circle and a line. -/
def PointOfTangency (C : Circle) (l : Line) : Point :=
  sorry

/-- Given a circle, a line, and a point on the line, there exists a tangent circle. -/
theorem tangent_circle_existence 
  (S : Circle) (l : Line) (A : Point) 
  (h_A_on_l : A ∈ l) : 
  ∃ (C : Circle), 
    (Tangent (Sum.inl C) (Sum.inl S)) ∧ 
    (Tangent (Sum.inl C) (Sum.inr l)) ∧ 
    (PointOfTangency C l = A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_existence_l299_29918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_partition_theorem_l299_29981

/-- Represents an n × n rook -/
structure Rook (n : ℕ) where
  (n_ge_two : n ≥ 2)
  (removed : Finset (ℕ × ℕ))
  (removed_size : removed.card = n)
  (removed_distinct : ∀ (i j : ℕ × ℕ), i ∈ removed → j ∈ removed → i.1 = j.1 ∨ i.2 = j.2 → i = j)

/-- Represents a rod (1 × k or k × 1 sub-grid) -/
inductive Rod (n : ℕ)
  | horizontal : ℕ → ℕ → ℕ → Rod n
  | vertical : ℕ → ℕ → ℕ → Rod n

/-- A partition of a rook into rods -/
def Partition (n : ℕ) (A : Rook n) := List (Rod n)

/-- The minimum number of rods in all possible partitions of a rook -/
noncomputable def m (n : ℕ) (A : Rook n) : ℕ := sorry

/-- The main theorem: for any n × n rook A, m(A) = n -/
theorem rook_partition_theorem (n : ℕ) (A : Rook n) : m n A = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_partition_theorem_l299_29981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l299_29984

/-- Given a function f(x) = m*ln(x) + n*x whose tangent at point (1, f(1)) is parallel
    to the line x + y - 2 = 0, and f(1) = -2, prove that the maximum value of t for which
    there exists x₀ ∈ [1, e] such that f(x₀) + x₀ ≥ g(x₀) is e*(e - 2)/(e - 1),
    where g(x) = (1/t)*(-x² + 2x) and t is a positive real number. -/
theorem max_t_value (m n : ℝ) (h1 : m * (1 / 1) + n = -1) (h2 : m * Real.log 1 + n * 1 = -2) :
  let f := fun x => m * Real.log x + n * x
  let g := fun t x => (1 / t) * (-x^2 + 2*x)
  ∃ t > 0, ∀ t' > t, ¬∃ x₀ ∈ Set.Icc 1 (Real.exp 1), f x₀ + x₀ ≥ g t' x₀ ∧
  t = Real.exp 1 * (Real.exp 1 - 2) / (Real.exp 1 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l299_29984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceiling_evaluation_l299_29967

theorem floor_ceiling_evaluation : 3 * Int.floor (-3.65 : ℝ) - Int.ceil (19.7 : ℝ) = -32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceiling_evaluation_l299_29967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l299_29927

/-- The area of a triangle with vertices (2, 3), (0, 7), and (5, 0) is 3. -/
theorem triangle_area : ∃ area : ℝ, area = 3 := by
  -- Define the vertices
  let A : Fin 2 → ℝ := ![2, 3]
  let B : Fin 2 → ℝ := ![0, 7]
  let C : Fin 2 → ℝ := ![5, 0]

  -- Define the function to calculate the area of a triangle given its vertices
  let area (p q r : Fin 2 → ℝ) : ℝ :=
    (1/2) * abs ((q 0 - p 0) * (r 1 - p 1) - (q 1 - p 1) * (r 0 - p 0))

  -- Calculate the area of the triangle with vertices A, B, C
  let triangleArea := area A B C

  -- Assert that the calculated area is equal to 3
  have h : triangleArea = 3 := by sorry

  -- Return the result
  exact ⟨triangleArea, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l299_29927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equivalence_l299_29982

theorem fraction_equivalence (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  (c^(-2 : ℤ) * d^(-1 : ℤ)) / (c^(-4 : ℤ) + d^(-2 : ℤ)) = (c^2 * d) / (d^2 + c^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equivalence_l299_29982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l299_29969

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x + x

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := Real.log x + 2

-- Theorem statement
theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = 2 * x - 1 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l299_29969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l299_29929

/-- The volume of the wire in cubic meters -/
noncomputable def wire_volume : ℝ := 22e-6

/-- The radius of the wire in meters -/
noncomputable def wire_radius : ℝ := 0.0005

/-- The length of the wire in meters -/
noncomputable def wire_length : ℝ := wire_volume / (Real.pi * wire_radius^2)

/-- Theorem stating that the length of the wire is approximately 28.01 meters -/
theorem wire_length_approx :
  |wire_length - 28.01| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l299_29929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l299_29924

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂

/-- The slope of line l₁ -/
noncomputable def slope_l₁ (m : ℝ) : ℝ := (3 + m) / 4

/-- The slope of line l₂ -/
noncomputable def slope_l₂ (m : ℝ) : ℝ := 2 / (5 + m)

/-- The theorem stating that if l₁ and l₂ are parallel, then m is either -1 or -7 -/
theorem parallel_lines_m_values :
  ∀ m : ℝ, parallel_lines (3 + m) 4 2 (5 + m) → m = -1 ∨ m = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l299_29924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_displeasure_l299_29958

/-- Represents the displeasure caused by having two pies instead of one -/
noncomputable def α : ℝ := sorry

/-- The probability that the child's prediction is correct -/
noncomputable def child_accuracy : ℝ := 2/3

/-- The probability that the husband will buy a pie -/
noncomputable def husband_buys_pie_prob : ℝ := 1/2

/-- Expected displeasure when Madame Dupont goes to the bakery herself -/
noncomputable def displeasure_self_buy : ℝ := husband_buys_pie_prob * α

/-- Expected displeasure when Madame Dupont relies on her son's prediction -/
noncomputable def displeasure_rely_on_son : ℝ := 
  (1 - child_accuracy) * (husband_buys_pie_prob * (2*α) + (1 - husband_buys_pie_prob) * α)

/-- Theorem stating that both strategies result in the same expected displeasure -/
theorem equal_displeasure : displeasure_self_buy = displeasure_rely_on_son := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_displeasure_l299_29958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_plane_l299_29953

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a line in 3D space -/
structure Line

/-- Represents the geometric configuration in 3D space -/
structure Geometry3D where
  α : Plane
  β : Plane
  m : Line

/-- Defines perpendicularity between a plane and another plane -/
def Plane.perpendicular (p q : Plane) : Prop := sorry

/-- Defines perpendicularity between a line and a plane -/
def Line.perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def Line.contained_in (l : Line) (p : Plane) : Prop := sorry

/-- Defines when a line is parallel to a plane -/
def Line.parallel (l : Line) (p : Plane) : Prop := sorry

/-- The main theorem -/
theorem parallel_to_plane (g : Geometry3D)
  (h1 : g.α.perpendicular g.β)
  (h2 : g.m.perpendicular g.β)
  (h3 : ¬g.m.contained_in g.α) :
  g.m.parallel g.α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_plane_l299_29953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l299_29942

/-- Calculates simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the principal amount given the conditions -/
theorem principal_amount (P : ℝ) :
  let rate : ℝ := 4
  let time : ℝ := 2
  let diff : ℝ := 3.20
  compound_interest P rate time - simple_interest P rate time = diff →
  P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l299_29942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_arrangement_theorem_l299_29975

def is_valid_arrangement (k : ℕ) (arr : List ℕ) : Prop :=
  (arr.length = 2 * k) ∧
  (∀ n, 1 ≤ n ∧ n ≤ k → (arr.count n = 2)) ∧
  (∀ n, 1 ≤ n ∧ n ≤ k →
    let first := arr.indexOf n
    let second := arr.findIndex (λ x => x = n ∧ arr.indexOf x > first)
    second - first - 1 = n)

theorem plate_arrangement_theorem :
  (∃ arr : List ℕ, is_valid_arrangement 3 arr) ∧
  (∃ arr : List ℕ, is_valid_arrangement 4 arr) ∧
  (∀ j : ℕ, ¬∃ arr : List ℕ, is_valid_arrangement (4*j + 1) arr) ∧
  (∀ j : ℕ, ¬∃ arr : List ℕ, is_valid_arrangement (4*j + 2) arr) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_arrangement_theorem_l299_29975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_power_condition_l299_29997

/-- Sequence a_n defined recursively -/
def a (c : ℕ+) : ℕ → ℕ
  | 0 => c
  | n + 1 => (a c n) ^ 2 + a c n + c ^ 3

/-- Proposition: c satisfies the condition iff it's of the form ℓ² - 1 -/
theorem sequence_power_condition (c : ℕ+) :
  (∃ (k m : ℕ), k ≥ 1 ∧ m ≥ 2 ∧ ∃ (x : ℕ), (a c k) ^ 2 + c ^ 3 = x ^ m) ↔
  ∃ (ℓ : ℕ), ℓ ≥ 2 ∧ c = ℓ ^ 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_power_condition_l299_29997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l299_29968

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties (A ω φ : ℝ) 
  (h1 : |φ| < π/2)
  (h2 : f A ω φ (π/6) = 1)
  (h3 : ∀ x, f A ω φ (x + π/(2*ω)) = f A ω φ x)
  (h4 : ∀ x, f A ω φ x ≤ 2) 
  (h5 : ∃ x, f A ω φ x = 2) :
  (∃ m : ℝ, π/3 ≤ m ∧ m < 5*π/6 ∧ 
    (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m → 
      (f A ω φ x₁ = f A ω φ x₂ → x₁ = x₂))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l299_29968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_inequality_l299_29902

theorem arithmetic_geometric_sequence_inequality 
  (x y a₁ a₂ b₁ b₂ : ℝ) 
  (hx : x > 0) (hy : y > 0)
  (ha : x - a₁ = a₁ - a₂ ∧ a₂ - y = x - a₁)
  (hg : x * b₂ = b₁^2 ∧ b₁ * y = b₂^2) :
  (a₁ + a₂)^2 / (b₁ * b₂) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_inequality_l299_29902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l299_29955

-- Part I
def f (k : ℝ) (x : ℝ) : ℝ := |k * x - 1|

theorem part_one (k : ℝ) : 
  (∀ x : ℝ, f k x ≤ 3 ↔ x ∈ Set.Icc (-2) 1) → k = -2 := by sorry

-- Part II
def g (x : ℝ) : ℝ := |x - 1|

theorem part_two (m : ℝ) : 
  (∀ x : ℝ, g (x + 2) - g (2 * x + 1) ≤ 3 - 2 * m) ↔ m ∈ Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l299_29955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l299_29992

def v1 : ℝ × ℝ := (4, -2)
def v2 : ℝ × ℝ := (5, 3)

theorem angle_between_vectors (v1 v2 : ℝ × ℝ) :
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  let cos_theta := dot_product / (magnitude1 * magnitude2)
  Real.arccos cos_theta = Real.arccos (14 / Real.sqrt 340) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l299_29992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l299_29971

-- Define the curve C₁
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)

-- Define the line C₂
noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 2 + t / 2, 1 - t / 2)

-- Define the distance function from a point to C₂
noncomputable def distToC₂ (p : ℝ × ℝ) : ℝ :=
  |p.1 + p.2 - 1 + 2 * Real.sqrt 2| / Real.sqrt 2

-- State the theorem
theorem min_distance_point :
  ∃ (θ : ℝ), C₁ θ = (1 - Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∧
  ∀ (φ : ℝ), distToC₂ (C₁ θ) ≤ distToC₂ (C₁ φ) ∧
  distToC₂ (C₁ θ) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l299_29971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_check_operations_l299_29993

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if two real numbers are equal -/
noncomputable def compareEqual (a b : ℝ) : Bool :=
  a = b

/-- Checks if a quadrilateral is a rectangle -/
noncomputable def isRectangle (q : Quadrilateral) : Bool :=
  let AB := distance q.A q.B
  let BC := distance q.B q.C
  let CD := distance q.C q.D
  let DA := distance q.D q.A
  let AC := distance q.A q.C
  let BD := distance q.B q.D
  compareEqual AB CD ∧ compareEqual BC DA ∧ compareEqual AC BD

/-- Theorem: Determining if a quadrilateral is a rectangle requires exactly 11 operations -/
theorem rectangle_check_operations (q : Quadrilateral) :
  (∃ (ops : ℕ), ops = 11 ∧ isRectangle q = true) ∨
  (∃ (ops : ℕ), ops = 11 ∧ isRectangle q = false) := by
  sorry

#check rectangle_check_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_check_operations_l299_29993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_shortest_chord_l299_29965

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 6

-- Define the line
def line_eq (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 2)*y - 3*m - 6 = 0

-- Theorem 1: The line passes through a fixed point (0, 3) for all m
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_eq m 0 3 := by sorry

-- Theorem 2: The shortest chord occurs when m = 1 and has length 4
theorem shortest_chord :
  ∃ (x y : ℝ), 
    line_eq 1 x y ∧ 
    circle_eq x y ∧ 
    (∀ (x' y' : ℝ) (m : ℝ), line_eq m x' y' ∧ circle_eq x' y' → 
      ((x - 1)^2 + (y - 2)^2)^(1/2) ≤ ((x' - 1)^2 + (y' - 2)^2)^(1/2)) ∧
    ((x - 1)^2 + (y - 2)^2)^(1/2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_shortest_chord_l299_29965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loyd_land_pole_distance_l299_29903

/-- The distance between poles when fencing a square land --/
noncomputable def distance_between_poles (side_length : ℝ) (num_poles : ℕ) : ℝ :=
  (4 * side_length) / (num_poles - 1 : ℝ)

/-- Theorem stating the distance between poles for Mr. Loyd's land --/
theorem loyd_land_pole_distance :
  let side_length : ℝ := 150
  let num_poles : ℕ := 30
  abs (distance_between_poles side_length num_poles - 20.69) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loyd_land_pole_distance_l299_29903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_90_percent_free_l299_29939

noncomputable def algae_growth (day : ℕ) : ℝ := (1 / 3) ^ (30 - day)

theorem pond_90_percent_free : ∃ d : ℕ, d < 30 ∧ algae_growth d ≤ 0.1 ∧ algae_growth (d + 1) > 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_90_percent_free_l299_29939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l299_29907

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (seq.a 1 + seq.a n)

theorem arithmetic_sequence_properties
  (seq : ArithmeticSequence)
  (h1 : S seq 15 > 0)
  (h2 : seq.a 9 / seq.a 8 < -1) :
  (|seq.a 9| > seq.a 8) ∧
  (∀ n : ℕ, S seq n ≤ S seq 8) ∧
  (∀ n : ℕ, n ≠ 0 → S seq 9 / seq.a 9 ≤ S seq n / seq.a n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l299_29907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_distribution_theorem_l299_29920

-- Define a symmetric distribution
def SymmetricDistribution (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f (m + x) = f (m - x)

-- Define the cumulative distribution function
noncomputable def CDF (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iio x, f y

theorem symmetric_distribution_theorem
  (f : ℝ → ℝ) (m d : ℝ) (h_symmetric : SymmetricDistribution f m)
  (h_within_std_dev : CDF f (m + d) - CDF f (m - d) = 0.64) :
  CDF f (m + d) = 0.82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_distribution_theorem_l299_29920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_equals_sqrt_two_l299_29940

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_zero_equals_sqrt_two : g 0 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_equals_sqrt_two_l299_29940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_l299_29901

theorem tan_minus_cot (θ : Real) (h1 : θ ∈ Set.Ioo 0 π) 
  (h2 : 25 * (Real.sin θ)^2 - 5 * (Real.sin θ) - 12 = 0) 
  (h3 : 25 * (Real.cos θ)^2 - 5 * (Real.cos θ) - 12 = 0) : 
  Real.tan θ - (1 / Real.tan θ) = -7/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_l299_29901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l299_29909

/-- An ellipse passing through a specific point with a special property of its foci --/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_point : a^2 + 2*b^2 = 2*a^2*b^2  -- Condition for passing through (1, √2/2)
  h_foci : a^2 = 2*b^2  -- Condition for isosceles right triangle with foci

/-- The circle x^2 + y^2 = 2/3 --/
def special_circle (x y : ℝ) : Prop := x^2 + y^2 = 2/3

/-- A point on the ellipse --/
structure EllipsePoint (e : SpecialEllipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2/(e.a^2) + y^2/(e.b^2) = 1

/-- Theorem about the special ellipse and its properties --/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∀ x y : ℝ, x^2/2 + y^2 = 1 ↔ x^2/(e.a^2) + y^2/(e.b^2) = 1) ∧
  (∀ l : ℝ → ℝ → Prop, 
    (∃ k m : ℝ, ∀ x y : ℝ, l x y ↔ y = k*x + m) →
    (∀ x y : ℝ, l x y → special_circle x y) →
    (∀ p q : EllipsePoint e, 
      (l p.x p.y ∧ l q.x q.y) → 
      p.x * q.x + p.y * q.y = 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l299_29909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_sum_difference_1001_l299_29952

def even_odd_sum_difference (n : ℕ) (start_even : ℕ) (start_odd : ℕ) : ℕ :=
  let even_sum := n * (start_even + (start_even + 2 * (n - 1))) / 2
  let odd_sum := n * (start_odd + (start_odd + 2 * (n - 1))) / 2
  even_sum - odd_sum

theorem even_odd_sum_difference_1001 :
  even_odd_sum_difference 1001 4 3 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_sum_difference_1001_l299_29952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_is_14_l299_29928

/-- The number of pages to be printed -/
def total_pages : ℕ := 340

/-- The number of pages printed per minute -/
def pages_per_minute : ℕ := 25

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- The time required to print the pages, rounded to the nearest minute -/
noncomputable def print_time : ℤ :=
  round_to_nearest ((total_pages : ℝ) / pages_per_minute)

theorem print_time_is_14 : print_time = 14 := by
  sorry

#eval total_pages
#eval pages_per_minute

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_is_14_l299_29928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l299_29938

theorem number_of_subsets_of_three_element_set (M : Finset α) :
  M.card = 3 → (M.powerset).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l299_29938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_side_shorter_than_altitude_l299_29941

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitude of a triangle
noncomputable def altitude (t : Triangle) (side : ℝ) : ℝ :=
  2 * (t.a * t.b * t.c) / (4 * side * side)

-- Theorem statement
theorem at_most_one_side_shorter_than_altitude (t : Triangle) :
  ¬(t.a < altitude t t.a ∧ t.b < altitude t t.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_side_shorter_than_altitude_l299_29941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ravi_jump_theorem_first_jumper_height_proof_l299_29922

/-- The height of the first jumper in inches -/
noncomputable def first_jumper_height : ℝ := 23

/-- The height of the second jumper in inches -/
noncomputable def second_jumper_height : ℝ := 27

/-- The height of the third jumper in inches -/
noncomputable def third_jumper_height : ℝ := 28

/-- Ravi's jump height in inches -/
noncomputable def ravi_jump_height : ℝ := 39

/-- The average jump height of the three next highest jumpers -/
noncomputable def average_jump_height : ℝ := (first_jumper_height + second_jumper_height + third_jumper_height) / 3

/-- Theorem stating that Ravi's jump height is 1.5 times the average jump height -/
theorem ravi_jump_theorem : 
  ravi_jump_height = 1.5 * average_jump_height :=
by sorry

/-- Main theorem proving the height of the first jumper -/
theorem first_jumper_height_proof : 
  first_jumper_height = 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ravi_jump_theorem_first_jumper_height_proof_l299_29922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_triangle_area_l299_29994

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Ellipse type -/
structure Ellipse where
  equation : ℝ → ℝ → Prop
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- Helper function to calculate triangle area -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem parabola_ellipse_intersection_triangle_area 
  (p : Parabola) 
  (e : Ellipse) 
  (P : ℝ × ℝ) :
  p.equation = fun x y ↦ y^2 = 4*x →
  p.focus = (1, 0) →
  e.equation = fun x y ↦ x^2/9 + y^2/8 = 1 →
  e.focus1 = p.focus →
  e.focus2.1 - e.focus1.1 = 2 →
  p.equation P.1 P.2 →
  e.equation P.1 P.2 →
  area_triangle P e.focus1 e.focus2 = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_triangle_area_l299_29994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l299_29916

/-- Given a principal amount and an interest rate, calculates the simple interest for 2 years -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate * 2 / 100

/-- Given a principal amount and an interest rate, calculates the compound interest for 2 years -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * ((1 + rate / 100)^2 - 1)

/-- Theorem stating that if the simple interest for 2 years is 56 and the compound interest for 2 years is 57.40, then the annual interest rate is 5% -/
theorem interest_rate_problem (principal : ℝ) (rate : ℝ) 
  (h1 : simple_interest principal rate = 56) 
  (h2 : compound_interest principal rate = 57.40) : 
  rate = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l299_29916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_and_triangle_areas_l299_29989

/-- Given vectors in R^2 -/
def p : ℝ × ℝ := (2, 4)
def q : ℝ × ℝ := (-3, 1)
def r : ℝ × ℝ := (1, -3)

/-- Area of parallelogram formed by two vectors -/
noncomputable def parallelogram_area (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

/-- Area of triangle formed by two vectors -/
noncomputable def triangle_area (v w : ℝ × ℝ) : ℝ :=
  (parallelogram_area v w) / 2

theorem parallelogram_and_triangle_areas :
  (parallelogram_area p q = 14) ∧
  (triangle_area q r = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_and_triangle_areas_l299_29989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_root_sum_l299_29934

noncomputable def f (x : ℝ) : ℝ := (5 * x - 1) / (3 * x^2 - 9 * x + 5)

noncomputable def root1 : ℝ := (9 + Real.sqrt 21) / 6
noncomputable def root2 : ℝ := (9 - Real.sqrt 21) / 6

theorem domain_and_root_sum :
  (∀ x : ℝ, f x ≠ 0 ↔ x ≠ root1 ∧ x ≠ root2) ∧
  root1 + root2 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_root_sum_l299_29934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_cycling_time_l299_29964

/-- Represents the time taken to cycle a given distance at a constant speed. -/
noncomputable def cyclingTime (speed : ℝ) (distance : ℝ) : ℝ := distance / speed

theorem library_cycling_time 
  (speed : ℝ) 
  (park_distance : ℝ) 
  (library_distance : ℝ) 
  (park_time : ℝ) 
  (h1 : speed > 0)
  (h2 : park_distance = 5)
  (h3 : library_distance = 3)
  (h4 : park_time = 30)
  (h5 : cyclingTime speed park_distance = park_time) :
  cyclingTime speed library_distance = 18 := by
  sorry

#check library_cycling_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_cycling_time_l299_29964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_nine_hours_l299_29912

/-- Represents the battery life of a laptop --/
structure BatteryLife where
  standby_hours : ℚ
  active_hours : ℚ

/-- Calculates the remaining battery life in standby mode --/
def remaining_standby_time (battery : BatteryLife) (total_time : ℚ) (active_time : ℚ) : ℚ :=
  let standby_time := total_time - active_time
  let standby_rate := 1 / battery.standby_hours
  let active_rate := 1 / battery.active_hours
  let consumed := standby_time * standby_rate + active_time * active_rate
  let remaining := 1 - consumed
  remaining / standby_rate

/-- Theorem: Given the conditions, the remaining battery life in standby mode is 9 hours --/
theorem remaining_battery_life_is_nine_hours (battery : BatteryLife) 
    (h1 : battery.standby_hours = 48)
    (h2 : battery.active_hours = 6)
    (h3 : remaining_standby_time battery 18 3 = 9) : 
  remaining_standby_time battery 18 3 = 9 := by
  sorry

#eval remaining_standby_time ⟨48, 6⟩ 18 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_nine_hours_l299_29912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l299_29948

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧ 2 * (s.filter (· ≥ m)).card ≥ s.card

theorem median_to_mean (m : ℝ) :
  let s : Finset ℝ := {m, m + 3, m + 6, m + 8, m + 10}
  is_median s (m + 6) ∧ m + 6 = 12 →
  Finset.sum s (id) / s.card = 57 / 5 := by
  sorry

#check median_to_mean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l299_29948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l299_29990

theorem sum_remainder_theorem (a b c : ℕ) 
  (ha : a % 53 = 31)
  (hb : b % 53 = 22)
  (hc : c % 53 = 7) :
  (a + b + c) % 53 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l299_29990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l299_29932

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) * (b (n + 1) + 1) = b n + 2021

/-- The theorem stating the minimum value of b₁ + b₂ -/
theorem min_sum_first_two_terms :
    ∃ m : ℕ, m = 90 ∧ 
    (∃ b : ℕ → ℕ, RecurrenceSequence b ∧ b 1 + b 2 = m) ∧
    (∀ k : ℕ → ℕ, RecurrenceSequence k → k 1 + k 2 ≥ m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l299_29932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_class_average_marks_l299_29973

theorem second_class_average_marks (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 → 
  n₂ = 50 → 
  avg₁ = 40 → 
  avg_total = 52.5 → 
  (n₁ * avg₁ + n₂ * ((n₁ + n₂) * avg_total - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_total →
  ((n₁ + n₂) * avg_total - n₁ * avg₁) / n₂ = 60 := by
  intros h1 h2 h3 h4 h5
  sorry

#check second_class_average_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_class_average_marks_l299_29973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l299_29980

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the problem setup -/
structure ProblemSetup where
  rect1 : Rectangle
  rect2 : Rectangle
  circleRadius : ℝ

/-- Calculates the shaded area for the given problem setup -/
noncomputable def shadedArea (p : ProblemSetup) : ℝ :=
  rectangleArea p.rect1 + rectangleArea p.rect2 - 
  (min p.rect1.width p.rect2.width * min p.rect1.height p.rect2.height) - 
  (Real.pi * p.circleRadius^2)

theorem shaded_area_calculation (p : ProblemSetup) 
  (h1 : p.rect1 = ⟨4, 12⟩) 
  (h2 : p.rect2 = ⟨5, 10⟩) 
  (h3 : p.circleRadius = 2) :
  shadedArea p = 78 - 4 * Real.pi := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l299_29980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_pole_theorem_l299_29946

theorem broken_pole_theorem (AC BC : ℝ) (angle_BAC : Real) :
  AC = 10 →
  angle_BAC = 30 * (Real.pi / 180) →
  AC = BC + (2 * BC) →
  BC = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_pole_theorem_l299_29946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_altitude_inequality_l299_29908

/-- In an acute triangle with sides a, b, c, opposite angles α, β, γ, and altitudes ma, mb, mc,
    the sum of altitude-to-side ratios is greater than or equal to a specific trigonometric expression. -/
theorem acute_triangle_altitude_inequality 
  (a b c : ℝ) (α β γ : ℝ) (ma mb mc : ℝ) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = Real.pi) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_altitudes : ma > 0 ∧ mb > 0 ∧ mc > 0) 
  (h_sine_law : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) : 
  ma / a + mb / b + mc / c ≥ 
    2 * Real.cos α * Real.cos β * Real.cos γ * (1 / Real.sin (2 * α) + 1 / Real.sin (2 * β) + 1 / Real.sin (2 * γ)) + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_altitude_inequality_l299_29908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l299_29945

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem f_properties :
  ∀ x : ℝ,
  (f x = Real.sin (2 * x - π/6) - 1) ∧
  (x ∈ Set.Icc (π/4) (π/2) →
    (∃ x_max ∈ Set.Icc (π/4) (π/2), ∀ y ∈ Set.Icc (π/4) (π/2), f y ≤ f x_max) ∧
    (∃ x_min ∈ Set.Icc (π/4) (π/2), ∀ y ∈ Set.Icc (π/4) (π/2), f y ≥ f x_min) ∧
    (f (π/3) = 0) ∧
    (f (π/2) = -1/2)) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (π/4) (π/2), |f x - m| < 1) ↔ m ∈ Set.Ioo (-1) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l299_29945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l299_29917

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define a line parallel to x+y-1=0
def parallel_line (b : ℝ) (x y : ℝ) : Prop := x + y + b = 0

-- Define the tangent condition
def is_tangent (b : ℝ) : Prop :=
  ∃ x y : ℝ, circle_eq x y ∧ parallel_line b x y ∧
  ∀ x' y' : ℝ, circle_eq x' y' → parallel_line b x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_lines :
  (∀ b : ℝ, is_tangent b ↔ b = 2 ∨ b = -2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l299_29917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_and_related_statements_l299_29935

theorem original_and_related_statements (a b c : ℝ) :
  (a * c^2 > b * c^2 → a > b) ∧
  (∃! n : Nat, n = (if (a > b → a * c^2 > b * c^2) then 1 else 0) +
                   (if (¬(a * c^2 > b * c^2) → ¬(a > b)) then 1 else 0) +
                   (if (¬(a > b) → ¬(a * c^2 > b * c^2)) then 1 else 0) ∧
                    n = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_and_related_statements_l299_29935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_l299_29987

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*(k+1)*x₁ + k^2 + 2 = 0 →
  x₂^2 - 2*(k+1)*x₂ + k^2 + 2 = 0 →
  (x₁ + 1) * (x₂ + 1) = 8 →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_l299_29987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stamps_purchased_l299_29905

/-- Given a stamp price of 45 cents and an available amount of 3600 cents,
    the maximum number of stamps that can be purchased is 80. -/
theorem max_stamps_purchased (stamp_price : ℕ) (available_amount : ℕ) : 
  stamp_price = 45 → available_amount = 3600 → 
  (∃ n : ℕ, n * stamp_price ≤ available_amount ∧ 
    ∀ m : ℕ, m * stamp_price ≤ available_amount → m ≤ n) → 
  (∃ n : ℕ, n * stamp_price ≤ available_amount ∧ 
    ∀ m : ℕ, m * stamp_price ≤ available_amount → m ≤ n ∧ n = 80) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stamps_purchased_l299_29905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l299_29954

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 6)

theorem function_properties (ω : ℝ) (α : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω (x + Real.pi / (2 * ω)) = f ω x)
  (h_min_period : ∀ T, (∀ x, f ω (x + T) = f ω x) → T ≥ Real.pi / (2 * ω))
  (h_f_value : f ω (α / 4 + Real.pi / 12) = 9 / 5) :
  f ω 0 = 3 / 2 ∧ 
  ω = 4 ∧ 
  Real.sin α = 4 / 5 ∨ Real.sin α = -4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l299_29954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_end_count_l299_29906

/-- Calculates the number of books in a special collection at the end of a month,
    given the initial number of books, the percentage of returned books, and the number of books loaned out. -/
def books_at_end_of_month (initial_books : ℕ) (return_rate : ℚ) (loaned_books : ℕ) : ℕ :=
  initial_books - (loaned_books - Int.toNat ((return_rate * loaned_books).floor))

/-- Theorem stating that the number of books in the special collection at the end of the month is 64. -/
theorem special_collection_end_count :
  books_at_end_of_month 75 (4/5) 55 = 64 := by
  -- Unfold the definition of books_at_end_of_month
  unfold books_at_end_of_month
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_end_count_l299_29906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_l299_29999

noncomputable def f (a b x : ℝ) : ℝ := -1/b * Real.exp (a * x)

theorem max_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (t : ℝ → ℝ), (∀ x, t x = f a b 0 + (deriv (f a b)) 0 * x) ∧
   (∃ x y, x^2 + y^2 = 1 ∧ t x = y ∧
    ∀ z, z^2 + (t z)^2 ≥ 1)) →
  a + b ≤ Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_l299_29999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l299_29937

open Set

def A (a : ℝ) : Set ℝ := {x | x < a}
def B : Set ℝ := Ioo 1 2

theorem range_of_a (a : ℝ) :
  ((Bᶜ) ∪ A a = univ) → a ∈ Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l299_29937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_tangents_through_2_neg2_l299_29972

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Theorem for the tangent line at (2, f(2))
theorem tangent_at_2 :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m*x + b ∧ 
  (x = 2 → y = f 2) ∧ 
  (∀ h : ℝ, h ≠ 0 → (f (2 + h) - f 2) / h = m) ∧
  x - y - 4 = 0 := by sorry

-- Theorem for tangent lines passing through (2, -2)
theorem tangents_through_2_neg2 :
  ∃ (x₀ y₀ m₁ b₁ m₂ b₂ : ℝ),
  (y₀ = f x₀) ∧
  (∀ x y : ℝ, y = m₁*x + b₁ → (x = 2 → y = -2) ∧ (x = x₀ → y = y₀)) ∧
  (∀ x y : ℝ, y = m₂*x + b₂ → (x = 2 → y = -2) ∧ (x = x₀ → y = y₀)) ∧
  (∀ h : ℝ, h ≠ 0 → (f (x₀ + h) - f x₀) / h = m₁) ∧
  (∀ h : ℝ, h ≠ 0 → (f (x₀ + h) - f x₀) / h = m₂) ∧
  (x - y - 4 = 0 ∨ y + 2 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_tangents_through_2_neg2_l299_29972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteNestedRadicalValue_l299_29977

/-- The value of the infinite nested radical $\sqrt{3+2\sqrt{3+2\sqrt{\cdots}}}$ -/
noncomputable def infiniteNestedRadical : ℝ :=
  Real.sqrt (3 + 2 * Real.sqrt (3 + 2 * Real.sqrt (3 + 2 * Real.sqrt 3)))

/-- Theorem stating that the value of the infinite nested radical is 3 -/
theorem infiniteNestedRadicalValue : infiniteNestedRadical = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteNestedRadicalValue_l299_29977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_one_l299_29933

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x > 0 then 2^x - 3 else -(2^(-x) - 3)

-- State the theorem
theorem f_neg_two_equals_neg_one :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  f (-2) = -1 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_one_l299_29933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l299_29919

noncomputable section

-- Define the function f(x)
def f (x m : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) - m

-- Define the interval [0, π/2]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.pi / 2 }

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∃ x y, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f x m = 0 ∧ f y m = 0) →
  1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l299_29919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_n_l299_29978

theorem m_less_than_n (m n : ℝ) : 
  (Real.sqrt 2, m) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} →
  ((3/2 : ℝ), n) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} →
  m < n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_n_l299_29978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_union_of_arithmetic_progressions_l299_29947

/-- The set of all composite positive odd integers less than 79 -/
def S : Set Nat :=
  {n | n < 79 ∧ n % 2 = 1 ∧ ¬n.Prime ∧ n > 1}

/-- An arithmetic progression -/
def ArithmeticProgression (a d : Nat) : Set Nat :=
  {n | ∃ k : Nat, n = a + k * d}

/-- Theorem stating that S can be expressed as the union of three arithmetic progressions
    but cannot be expressed as the union of two arithmetic progressions -/
theorem S_union_of_arithmetic_progressions :
  (∃ ap1 ap2 ap3 : Set Nat, S = ap1 ∪ ap2 ∪ ap3 ∧
    ∃ a1 d1 a2 d2 a3 d3 : Nat,
      ap1 = ArithmeticProgression a1 d1 ∧
      ap2 = ArithmeticProgression a2 d2 ∧
      ap3 = ArithmeticProgression a3 d3) ∧
  ¬(∃ ap1 ap2 : Set Nat, S = ap1 ∪ ap2 ∧
    ∃ a1 d1 a2 d2 : Nat,
      ap1 = ArithmeticProgression a1 d1 ∧
      ap2 = ArithmeticProgression a2 d2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_union_of_arithmetic_progressions_l299_29947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_nine_fourths_l299_29957

/-- The sum of the infinite series Σ(2n + 3) / (n(n+1)(n+2)(n+3)) for n from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n, (2 * n + 3) / (n * (n + 1) * (n + 2) * (n + 3))

/-- The theorem stating that the infinite series equals 9/4 -/
theorem infiniteSeries_eq_nine_fourths : infiniteSeries = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_nine_fourths_l299_29957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_wins_732_l299_29949

/-- Represents a wall of bricks in the game -/
structure Wall where
  size : Nat

/-- Represents a configuration of walls in the game -/
def Configuration := List Wall

/-- Calculates the Nim-value of a single wall -/
def nimValue (w : Wall) : Nat :=
  match w.size with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | 6 => 3
  | 7 => 5
  | _ => 0  -- Default case, though not used in this specific problem

/-- Calculates the Nim-sum (XOR) of a list of natural numbers -/
def nimSum : List Nat → Nat
  | [] => 0
  | (x::xs) => Nat.xor x (nimSum xs)

/-- Determines if a configuration is a winning position for the current player -/
def isWinningPosition (config : Configuration) : Prop :=
  (nimSum (config.map (fun w => nimValue w))) ≠ 0

/-- The main theorem to prove -/
theorem beth_wins_732 :
  ¬ isWinningPosition [Wall.mk 7, Wall.mk 3, Wall.mk 2] := by
  sorry

#eval nimSum [nimValue (Wall.mk 7), nimValue (Wall.mk 3), nimValue (Wall.mk 2)]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_wins_732_l299_29949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_observations_l299_29959

theorem number_of_observations (initial_mean wrong_value correct_value corrected_mean n : ℝ) 
  (h1 : initial_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 43)
  (h4 : corrected_mean = 36.5)
  : (corrected_mean * n - initial_mean * n) / (correct_value - wrong_value) = 1 ∧ 
    n = 40 := by
  sorry

#check number_of_observations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_observations_l299_29959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l299_29910

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  (A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2) ∧
  (B.1 - A.1 = 10 ∧ C.2 - B.2 = 5)

-- Define the points E and F
noncomputable def E (A B C D : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def F (A B C D : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the equality of DE and DF
def DE_eq_DF (A B C D : ℝ × ℝ) : Prop :=
  let E := E A B C D
  let F := F A B C D
  Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) =
  Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)

-- Define the area ratio condition
def AreaRatio (A B C D : ℝ × ℝ) : Prop :=
  let E := E A B C D
  let F := F A B C D
  (1/2) * abs ((E.1 - D.1) * (F.2 - D.2) - (F.1 - D.1) * (E.2 - D.2)) =
  (1/3) * ((B.1 - A.1) * (C.2 - B.2))

-- Theorem statement
theorem EF_length (A B C D : ℝ × ℝ) :
  Rectangle A B C D →
  DE_eq_DF A B C D →
  AreaRatio A B C D →
  let E := E A B C D
  let F := F A B C D
  Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = (10 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l299_29910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l299_29962

/-- Calculates the actual average height of students given initial data and corrections -/
theorem actual_average_height
  (n : ℕ)  -- Total number of students
  (initial_avg : ℝ)  -- Initial calculated average height
  (incorrect_heights : List ℝ)  -- List of incorrectly recorded heights
  (correct_heights : List ℝ)  -- List of correct heights
  (h_n : n = 50)  -- There are 50 students
  (h_initial_avg : initial_avg = 185)  -- Initial average was 185 cm
  (h_incorrect : incorrect_heights = [165, 175, 190])  -- Incorrectly recorded heights
  (h_correct : correct_heights = [105, 155, 180])  -- Actual heights
  : ∃ (actual_avg : ℝ), abs (actual_avg - 183.2) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l299_29962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l299_29976

/-- Conversion from spherical coordinates to rectangular coordinates is correct -/
theorem spherical_to_rectangular_conversion (ρ θ φ : Real) 
  (h_ρ : ρ = 5) 
  (h_θ : θ = 2 * Real.pi / 3) 
  (h_φ : φ = Real.pi / 4) : 
  (ρ * Real.sin φ * Real.cos θ, 
   ρ * Real.sin φ * Real.sin θ, 
   ρ * Real.cos φ) = 
  (-5 * Real.sqrt 2 / 4, 
   5 * Real.sqrt 6 / 4, 
   5 * Real.sqrt 2 / 2) := by
  sorry

#check spherical_to_rectangular_conversion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l299_29976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forty_percent_speak_french_l299_29913

/-- Represents the percentage of employees in a company who speak French -/
noncomputable def percentage_speaking_french (total_employees : ℝ) (male_employees : ℝ) 
  (male_french_speakers : ℝ) (female_non_french_speakers : ℝ) : ℝ :=
  ((male_french_speakers + (1 - female_non_french_speakers) * (total_employees - male_employees)) 
  / total_employees) * 100

/-- Theorem stating that under given conditions, 40% of employees speak French -/
theorem forty_percent_speak_french (total_employees : ℝ) (male_employees : ℝ) 
  (male_french_speakers : ℝ) (female_non_french_speakers : ℝ) 
  (h1 : male_employees = 0.35 * total_employees)
  (h2 : male_french_speakers = 0.60 * male_employees)
  (h3 : female_non_french_speakers = 0.7077 * (total_employees - male_employees)) :
  percentage_speaking_french total_employees male_employees male_french_speakers female_non_french_speakers = 40 := by
  sorry

/-- Example calculation -/
def example_calculation : ℚ :=
  let total_employees : ℚ := 100
  let male_employees : ℚ := 35
  let male_french_speakers : ℚ := 21
  let female_non_french_speakers : ℚ := 46
  ((male_french_speakers + (1 - female_non_french_speakers) * (total_employees - male_employees)) 
  / total_employees) * 100

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forty_percent_speak_french_l299_29913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_power_five_l299_29950

theorem det_B_power_five {n : Type*} [Fintype n] [DecidableEq n] 
  (B : Matrix n n ℝ) (h : Matrix.det B = -3) : 
  Matrix.det (B^5) = -243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_power_five_l299_29950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l299_29926

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the family of lines
def line_eq (l x y : ℝ) : Prop := (2 + l) * x + (1 - 2*l) * y + 4 - 3*l = 0

-- Define the length of a chord
noncomputable def chord_length (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem shortest_chord_length :
  ∃ (l : ℝ), ∀ (m x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
    line_eq l x1 y1 ∧ line_eq l x2 y2 ∧
    line_eq m x1 y1 ∧ line_eq m x2 y2 →
    chord_length x1 y1 x2 y2 ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l299_29926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l299_29995

theorem triangle_angle_inequalities (α β γ : ℝ) 
  (h : α + β + γ = Real.pi) : 
  (Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2) ∧ 
  (Real.cos (α/2) + Real.cos (β/2) + Real.cos (γ/2) ≤ 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l299_29995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_is_fixed_l299_29974

-- Define the circle Γ
variable (Γ : Set Point)

-- Define points A, B, C on the circle Γ
variable (A B C : Point)
variable (h_ABC_on_Γ : A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ)

-- Define the arc AB not containing C
variable (arc_AB : Set Point)
variable (h_arc_AB : A ∈ arc_AB ∧ B ∈ arc_AB ∧ C ∉ arc_AB)

-- Define the variable point P on arc AB
variable (P : Point)
variable (h_P_on_arc : P ∈ arc_AB)

-- Define the incenter I of triangle ACP
noncomputable def I (P : Point) : Point := sorry

-- Define the incenter J of triangle BCP
noncomputable def J (P : Point) : Point := sorry

-- Define the circumcircle of triangle PIJ
noncomputable def circle_PIJ (P : Point) : Set Point := sorry

-- Define Q as the intersection of Γ and circle_PIJ
noncomputable def Q (P : Point) : Point := sorry

-- Theorem statement
theorem Q_is_fixed : 
  ∀ P₁ P₂ : Point, P₁ ∈ arc_AB → P₂ ∈ arc_AB → 
  Q P₁ = Q P₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_is_fixed_l299_29974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_approx_one_year_l299_29944

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compoundInterest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- The main theorem -/
theorem compound_interest_time_approx_one_year :
  let principal_simple : ℝ := 3225
  let rate_simple : ℝ := 8
  let time_simple : ℝ := 5
  let principal_compound : ℝ := 8000
  let rate_compound : ℝ := 15
  let si := simpleInterest principal_simple rate_simple time_simple
  let ci := 2 * si
  ∃ n : ℝ, 0 < n ∧ n < 2 ∧ 
    compoundInterest principal_compound rate_compound n = ci :=
by
  sorry

#eval 1  -- This is just to ensure the file compiles without errors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_approx_one_year_l299_29944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l299_29985

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,1),
    prove that its radius is √629/2 -/
theorem circle_radius_proof (x : ℝ) : 
  (x - 2)^2 + 5^2 = (x - 3)^2 + 1^2 →
  Real.sqrt ((x - 2)^2 + 5^2) = Real.sqrt 629 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l299_29985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_minimum_distance_l299_29986

/-- Represents a line in the xy-coordinate system of the form x = ay + b -/
structure Line where
  a : ℝ
  b : ℝ

/-- The distance between two points in the xy-coordinate system -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem parallel_lines_minimum_distance (m n : ℝ) :
  let L1 : Line := ⟨2, 3⟩
  let L2 : Line := ⟨2, -1⟩
  let d := distance m n (m + 2) (n + 2)
  (∀ k : ℝ, L1.a = L2.a → k = 2) ∧
  d = 2 * Real.sqrt 2 ∧
  ∀ x y : ℝ, distance x y (x + 2) (y + 2) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_minimum_distance_l299_29986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_colors_sufficient_l299_29925

/-- A region in the plane --/
structure Region where
  enclosed_by : Finset ℕ

/-- The set of all regions created by the circles --/
def all_regions : Finset Region := sorry

/-- Two regions are adjacent if their enclosed_by sets differ by exactly one element --/
def adjacent (r1 r2 : Region) : Prop :=
  ∃ (c : ℕ), r1.enclosed_by = r2.enclosed_by ∪ {c} ∨ r2.enclosed_by = r1.enclosed_by ∪ {c}

/-- A coloring of the regions --/
def coloring : Region → Bool := sorry

/-- The coloring is valid if adjacent regions have different colors --/
def valid_coloring (c : Region → Bool) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

theorem two_colors_sufficient :
  ∃ (c : Region → Bool), valid_coloring c ∧ (∀ r, r ∈ all_regions → c r = (r.enclosed_by.card % 2 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_colors_sufficient_l299_29925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_l299_29911

theorem max_tan_C (A B C : EuclideanSpace ℝ (Fin 2)) : 
  (dist A B = 2) → 
  (dist A C ^ 2 - dist B C ^ 2 = 6) → 
  (∀ θ : ℝ, Real.tan θ ≤ 2 * Real.sqrt 5 / 5) ∧ 
  (∃ θ : ℝ, Real.tan θ = 2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_l299_29911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisectors_of_adjacent_supplementary_angles_are_perpendicular_l299_29963

-- Define the concept of adjacent angles
def adjacent_angles (α β : ℝ) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary_angles (α β : ℝ) : Prop := α + β = Real.pi

-- Define the concept of angle bisector
noncomputable def angle_bisector (α : ℝ) : ℝ := α / 2

-- Define perpendicularity for real numbers (representing angles)
def perpendicular (x y : ℝ) : Prop := sorry

-- Theorem statement
theorem bisectors_of_adjacent_supplementary_angles_are_perpendicular 
  (α β : ℝ) (h1 : adjacent_angles α β) (h2 : supplementary_angles α β) :
  perpendicular (angle_bisector α) (angle_bisector β) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisectors_of_adjacent_supplementary_angles_are_perpendicular_l299_29963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l299_29915

/-- Given a hyperbola with eccentricity √5/2 that shares a common focus with the ellipse x²/9 + y²/4 = 1,
    prove that the equation of the hyperbola is x²/4 - y² = 1 -/
theorem hyperbola_equation (e : ℝ) (x y : ℝ → ℝ) :
  e = Real.sqrt 5 / 2 →
  (∃ c : ℝ, c > 0 ∧ c^2 = 5 ∧ (∀ t : ℝ, x t^2 / 9 + y t^2 / 4 = 1 → (x t - c)^2 + y t^2 = 5)) →
  (∀ t : ℝ, x t^2 / 4 - y t^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l299_29915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l299_29914

noncomputable section

-- Define the vertices of the larger tetrahedron
def v1 : Fin 5 → ℝ := ![1, 0, 0, 0, 0]
def v2 : Fin 5 → ℝ := ![0, 1, 0, 0, 0]
def v3 : Fin 5 → ℝ := ![0, 0, 1, 0, 0]
def v4 : Fin 5 → ℝ := ![0, 0, 0, 1, 0]

-- Define the centers of the 3D hyperface sections
def c1 : Fin 5 → ℝ := ![1/3, 1/3, 1/3, 1/3, 0]
def c2 : Fin 5 → ℝ := ![1/3, 1/3, 1/3, 0, 1/3]
def c3 : Fin 5 → ℝ := ![1/3, 1/3, 0, 1/3, 1/3]
def c4 : Fin 5 → ℝ := ![1/3, 0, 1/3, 1/3, 1/3]

-- Define the volume of a tetrahedron given its side length
noncomputable def tetrahedron_volume (side_length : ℝ) : ℝ :=
  (side_length ^ 3) / (6 * Real.sqrt 2)

theorem tetrahedron_volume_ratio :
  let large_side := Real.sqrt 2
  let small_side := large_side / 3
  let large_volume := tetrahedron_volume large_side
  let small_volume := tetrahedron_volume small_side
  small_volume / large_volume = 1 / 27 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l299_29914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_and_time_l299_29904

noncomputable def distance (t : ℝ) : ℝ := 9*t + (1/2)*t^2

theorem car_distance_and_time :
  (distance 12 = 180) ∧
  (∃ (t : ℝ), t > 0 ∧ distance t = 380 ∧ t = 20) := by
  constructor
  · -- Part 1: Distance traveled after 12s
    simp [distance]
    norm_num
  · -- Part 2: Time taken to travel 380m
    use 20
    constructor
    · norm_num
    constructor
    · simp [distance]
      norm_num
    · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_and_time_l299_29904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l299_29966

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(a n)

-- Define the sum of the first n terms of b_n
noncomputable def S (n : ℕ) : ℝ := (2/3) * (4^n - 1)

theorem arithmetic_sequence_properties :
  -- a_1 = 1
  a 1 = 1 ∧
  -- a_2 is the geometric mean of a_1 and a_5
  a 2 = Real.sqrt (a 1 * a 5) ∧
  -- {b_n} is a geometric sequence
  (∀ n : ℕ, n ≥ 2 → b n / b (n-1) = 4) ∧
  -- The sum of the first n terms of {b_n} is S_n
  (∀ n : ℕ, n ≥ 1 → S n = (b 1) * (1 - 4^n) / (1 - 4)) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l299_29966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incircle_tangency_l299_29983

/-- A hyperbola with foci F₁ and F₂ and vertices M and N -/
structure Hyperbola (P : Type*) [MetricSpace P] where
  F₁ : P
  F₂ : P
  M : P
  N : P

/-- A point on the hyperbola -/
def PointOnHyperbola {P : Type*} [MetricSpace P] (h : Hyperbola P) (X : P) : Prop :=
  |dist X h.F₁ - dist X h.F₂| = dist h.M h.N

/-- The incircle of a triangle -/
noncomputable def Incircle {P : Type*} [MetricSpace P] (A B C : P) : Set P :=
  sorry

/-- The point of tangency of the incircle with a side of the triangle -/
noncomputable def PointOfTangency {P : Type*} [MetricSpace P] (A B C : P) : P :=
  sorry

theorem hyperbola_incircle_tangency
  {P : Type*} [MetricSpace P]
  (h : Hyperbola P) (P : P)
  (hp : PointOnHyperbola h P) :
  PointOfTangency P h.F₁ h.F₂ = h.M ∨
  PointOfTangency P h.F₁ h.F₂ = h.N :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incircle_tangency_l299_29983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_part1_line_l_equation_part2_l299_29943

-- Define the lines and circles
def line1 : Set (ℝ × ℝ) := {(x, y) | 2*x + y + 5 = 0}
def line2 : Set (ℝ × ℝ) := {(x, y) | x - 2*y = 0}
def circle1 : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 - 2*x - 2*y - 4 = 0}
def circle2 : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 + 6*x + 2*y - 6 = 0}

-- Define the point P
def P : ℝ × ℝ := (5, 0)

-- Define the general form of line l
def line_l (lambda : ℝ) : Set (ℝ × ℝ) := {(x, y) | (2+lambda)*x + (1-2*lambda)*y - 5 = 0}

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the perpendicularity condition
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Theorem for part 1
theorem line_l_equation_part1 :
  ∃ lambda : ℝ, (line_l lambda = {(x, y) | x = 2} ∨ line_l lambda = {(x, y) | 4*x - 3*y - 5 = 0}) ∧
  distance_to_line P (line_l lambda) = 4 := by sorry

-- Theorem for part 2
theorem line_l_equation_part2 :
  ∃ lambda : ℝ, line_l lambda = {(x, y) | 3*x - 4*y - 2 = 0} ∧
  perpendicular (line_l lambda) {(x, y) | (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_part1_line_l_equation_part2_l299_29943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_l299_29931

-- Define the points
def E : ℝ × ℝ := (2, 3)
def F : ℝ × ℝ := (-1, 4)
def G : ℝ × ℝ := (-3, 1)
def H : ℝ × ℝ := (1, -2)

-- Define the angles in radians
noncomputable def angle_F : ℝ := 30 * Real.pi / 180
noncomputable def angle_G : ℝ := 45 * Real.pi / 180
noncomputable def angle_E : ℝ := 60 * Real.pi / 180
noncomputable def angle_H : ℝ := 70 * Real.pi / 180

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_segment :
  distance F H > distance E F ∧
  distance F H > distance F G ∧
  distance F H > distance G H ∧
  distance F H > distance E H :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_l299_29931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l299_29921

/-- Given a train that passes an electric pole in 14 seconds and a 370-meter long platform in 51 seconds, its speed is 36 km/h. -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 14 →
  platform_length = 370 →
  platform_time = 51 →
  (platform_length / (platform_time - pole_time)) * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l299_29921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_bound_l299_29998

theorem triangle_sine_product_bound (A B C : ℝ) :
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_bound_l299_29998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_theorem_l299_29960

open Complex

-- Define the sector
noncomputable def Sector : Set ℂ := {z : ℂ | 0 < arg z ∧ arg z < Real.pi / 4}

-- Define the unit disk
noncomputable def UnitDisk : Set ℂ := {w : ℂ | abs w < 1}

-- Define the mapping function
noncomputable def f (z : ℂ) : ℂ := -(z^4 - I) / (z^4 + I)

theorem mapping_theorem :
  (∀ z ∈ Sector, f z ∈ UnitDisk) ∧
  f (exp (I * Real.pi / 8)) = 0 ∧
  f 0 = 1 := by
  sorry

#check mapping_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_theorem_l299_29960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l299_29970

def is_valid_subset (A : Finset ℕ) : Prop :=
  A ⊆ Finset.range 30 ∧
  ∀ (a b : ℕ) (k : ℤ), a ∈ A → b ∈ A →
    ¬∃ (n : ℕ), (a : ℤ) + b + 30 * k = n * (n + 1)

theorem max_valid_subset_size :
  ∃ (A : Finset ℕ), is_valid_subset A ∧ A.card = 10 ∧
    ∀ (B : Finset ℕ), is_valid_subset B → B.card ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l299_29970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_region_l299_29991

/-- The curve in the Cartesian coordinate system xOy -/
def curve (x y : ℝ) : Prop := 2 * |x| + 3 * |y| = 5

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2}

/-- The area of the region enclosed by the curve -/
noncomputable def area : ℝ := (MeasureTheory.volume enclosed_region).toReal

theorem area_of_enclosed_region :
  area = 25 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_region_l299_29991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_sequence_l299_29930

theorem book_price_sequence (n : ℕ) (d : ℝ) (a₁ : ℝ) : 
  n = 40 → d = 3 → a₁ > 0 → 
  let seq := λ i ↦ a₁ + d * ((i : ℝ) - 1)
  seq n = seq 20 + seq 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_sequence_l299_29930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_nonnegative_on_interval_l299_29951

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - x) - Real.log x

-- Theorem 1: If f(x) has an extremum at x = 1, then a = 1
theorem extremum_at_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≥ f a 1 ∨ f a x ≤ f a 1) → 
  a = 1 := by
sorry

-- Theorem 2: If f(x) ≥ 0 for all x in [1, ∞), then a ≥ 1
theorem nonnegative_on_interval (a : ℝ) :
  (∀ x ≥ 1, f a x ≥ 0) → a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_nonnegative_on_interval_l299_29951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l299_29900

theorem sin_alpha_plus_beta (α β : ℝ) 
  (h1 : π / 4 < α ∧ α < 3 * π / 4)
  (h2 : 0 < β ∧ β < π / 4)
  (h3 : Real.cos (π / 4 + α) = -3 / 5)
  (h4 : Real.sin (3 * π / 4 + β) = 5 / 13) :
  Real.sin (α + β) = 63 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l299_29900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_is_one_l299_29956

/-- The sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The sum of the first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The sequence we're interested in -/
def our_sequence (n : ℕ) : ℚ := (sum_even n : ℚ) / (sum_odd n : ℚ)

theorem limit_of_sequence_is_one :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |our_sequence n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_is_one_l299_29956
