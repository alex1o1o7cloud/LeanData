import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_price_l275_27599

/-- The cost of a burger in cents -/
def burger_cost : ℕ → Prop := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ → Prop := sorry

/-- Alice's purchase: 3 burgers and 2 sodas cost 320 cents -/
axiom alice_purchase (b s : ℕ) : burger_cost b → soda_cost s → 3 * b + 2 * s = 320

/-- Bill's purchase: 2 burgers and 1 soda cost 200 cents -/
axiom bill_purchase (b s : ℕ) : burger_cost b → soda_cost s → 2 * b + s = 200

/-- The theorem stating that a burger costs 80 cents -/
theorem burger_price : ∃ b, burger_cost b ∧ b = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_price_l275_27599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l275_27517

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (the focus) and a fixed line (the directrix). -/
structure Parabola where
  points : Set (ℝ × ℝ)
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The standard form of a parabola with vertex at the origin is x^2 = 4py, where p is the distance from the vertex to the focus. -/
def standard_form (p : Parabola) : Prop :=
  p.vertex = (0, 0) → ∃ k : ℝ, ∀ x y : ℝ, (x, y) ∈ p.points ↔ x^2 = k * y

/-- Theorem: For a parabola with vertex at (0,0) and focus at (0,2), its standard equation is x^2 = 8y. -/
theorem parabola_equation (p : Parabola) :
  p.vertex = (0, 0) ∧ p.focus = (0, 2) → standard_form p ∧ ∀ x y : ℝ, (x, y) ∈ p.points ↔ x^2 = 8 * y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l275_27517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_14_times_count_l275_27520

/-- The transformation function applied to a set of points -/
def transform (S : Set (ℤ × ℤ)) : Set (ℤ × ℤ) := sorry

/-- The initial set of points -/
def initial_set : Set (ℤ × ℤ) := {(0, 0), (2, 0)}

/-- Apply the transformation n times -/
def apply_n_times (S : Set (ℤ × ℤ)) (n : ℕ) : Set (ℤ × ℤ) :=
  match n with
  | 0 => S
  | n + 1 => transform (apply_n_times S n)

/-- Count the number of elements in a set -/
def count_elements (S : Set (ℤ × ℤ)) : ℕ := sorry

theorem transform_14_times_count :
  count_elements (apply_n_times initial_set 14) = 477 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_14_times_count_l275_27520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_perfect_square_l275_27518

theorem at_least_one_not_perfect_square (x : ℕ+) :
  ∃ n ∈ ({2 * (x : ℤ) - 1, 5 * (x : ℤ) - 1, 13 * (x : ℤ) - 1} : Set ℤ), ¬∃ m : ℤ, n = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_perfect_square_l275_27518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_hexagon_intersections_l275_27505

/-- A regular dodecahedron in 3D space -/
structure RegularDodecahedron where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- Represents an intersection between a plane and a dodecahedron -/
structure Intersection where
  dodecahedron : RegularDodecahedron
  plane : Plane

/-- Predicate to check if an intersection forms a regular hexagon -/
def IsRegularHexagon (i : Intersection) : Prop :=
  sorry

/-- The number of large diagonals in a regular dodecahedron -/
def numLargeDiagonals : ℕ := 10

/-- The number of possible planes per large diagonal that form a regular hexagon -/
def planesPerDiagonal : ℕ := 3

/-- The main theorem: number of ways a plane can intersect a regular dodecahedron to form a regular hexagon -/
theorem dodecahedron_hexagon_intersections (d : RegularDodecahedron) :
  ∃ (s : Finset Intersection), s.card = numLargeDiagonals * planesPerDiagonal ∧
    ∀ i ∈ s, IsRegularHexagon i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_hexagon_intersections_l275_27505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l275_27578

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

/-- The distance between a point and the left focus -/
noncomputable def dist_left_focus (P : ℝ × ℝ) (F₁ : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)

/-- The distance between a point and the right focus -/
noncomputable def dist_right_focus (P : ℝ × ℝ) (F₂ : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)

theorem hyperbola_focus_distance (P F₁ F₂ : ℝ × ℝ) :
  hyperbola P.1 P.2 →
  F₁.1 < F₂.1 →
  dist_left_focus P F₁ = 9 →
  dist_right_focus P F₂ = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l275_27578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_halfway_distance_l275_27594

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℚ
  apogee : ℚ

/-- The distance from the sun to a point halfway along the orbit -/
def halfway_distance (orbit : EllipticalOrbit) : ℚ :=
  (orbit.apogee + orbit.perigee) / 2

/-- Theorem: For Zeta's orbit, the halfway distance is 9 AU -/
theorem zeta_halfway_distance :
  let zeta_orbit : EllipticalOrbit := { perigee := 3, apogee := 15 }
  halfway_distance zeta_orbit = 9 := by
  -- Unfold the definition of halfway_distance
  unfold halfway_distance
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

#check zeta_halfway_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_halfway_distance_l275_27594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_direction_vector_of_line_l275_27598

/-- The line equation y = 2x + 2 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 2

/-- The direction vector of the line -/
def direction_vector : ℝ × ℝ := (1, 2)

/-- The unit direction vector of the line -/
noncomputable def unit_direction_vector : ℝ × ℝ := (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5)

/-- Theorem stating the relationship between the direction vector and the unit direction vector -/
theorem unit_direction_vector_of_line :
  ∃ (ε : ℝ), ε = 1 ∨ ε = -1 ∧ 
  (ε * unit_direction_vector.1 = direction_vector.1 / Real.sqrt (direction_vector.1^2 + direction_vector.2^2) ∧
   ε * unit_direction_vector.2 = direction_vector.2 / Real.sqrt (direction_vector.1^2 + direction_vector.2^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_direction_vector_of_line_l275_27598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l275_27535

noncomputable def x (t : ℝ) (C₁ C₂ : ℝ) : ℝ := 
  C₁ * Real.exp (2 * t) + C₂ * Real.exp (4 * t) - 4 * Real.exp (3 * t) - Real.exp (-t)

noncomputable def y (t : ℝ) (C₁ C₂ : ℝ) : ℝ := 
  C₁ * Real.exp (2 * t) + (1/3) * C₂ * Real.exp (4 * t) - 2 * Real.exp (3 * t) - 2 * Real.exp (-t)

theorem solution_satisfies_system (t : ℝ) (C₁ C₂ : ℝ) :
  (deriv (x · C₁ C₂)) t = 5 * x t C₁ C₂ - 3 * y t C₁ C₂ + 2 * Real.exp (3 * t) ∧
  (deriv (y · C₁ C₂)) t = x t C₁ C₂ + y t C₁ C₂ + 5 * Real.exp (-t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l275_27535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_given_points_l275_27580

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem point_equidistant_from_given_points :
  let A : ℝ × ℝ := (2, 0)
  let P₁ : ℝ × ℝ := (-3, 2)
  let P₂ : ℝ × ℝ := (4, -5)
  distance A.1 A.2 P₁.1 P₁.2 = distance A.1 A.2 P₂.1 P₂.2 :=
by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expressions
  simp
  -- The proof is completed with sorry for now
  sorry

#check point_equidistant_from_given_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_given_points_l275_27580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_existence_l275_27548

/-- Definition of a line in 2D Cartesian coordinates --/
def is_line (L : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    ∀ (M : ℝ × ℝ), M ∈ L ↔ (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2

/-- A straight line in 2D Cartesian coordinates can be represented by an equation ax + by + c = 0 --/
theorem line_equation_existence :
  ∀ (L : Set (ℝ × ℝ)), is_line L →
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
    (∀ (x y : ℝ), (x, y) ∈ L ↔ a * x + b * y + c = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_existence_l275_27548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_surface_area_l275_27554

/-- Regular quadrilateral pyramid with inscribed sphere -/
structure RegularPyramid where
  /-- Distance from center of inscribed sphere to apex -/
  a : ℝ
  /-- Angle between lateral face and base plane -/
  α : ℝ
  /-- a is positive -/
  a_pos : 0 < a
  /-- α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < π / 2

/-- Total surface area of the regular quadrilateral pyramid -/
noncomputable def totalSurfaceArea (p : RegularPyramid) : ℝ :=
  8 * p.a^2 * Real.cos p.α * Real.cos (p.α / 2)^2 * (1 / Real.tan (p.α / 2))^2

/-- Theorem: The total surface area of a regular quadrilateral pyramid with an inscribed sphere -/
theorem regular_pyramid_surface_area (p : RegularPyramid) :
  totalSurfaceArea p = 8 * p.a^2 * Real.cos p.α * Real.cos (p.α / 2)^2 * (1 / Real.tan (p.α / 2))^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_surface_area_l275_27554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_theorem_l275_27534

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the center of circle C
def center_C : ℝ × ℝ := (1, 0)

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define a line with a given slope passing through a point
def line_with_slope (m : ℝ) (x0 y0 : ℝ) (x y : ℝ) : Prop :=
  y - y0 = m * (x - x0)

-- Define the length of a chord
noncomputable def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt (r^2 - d^2)

theorem circle_chord_theorem :
  -- Part 1
  (∀ x y : ℝ, line_through_points point_P.1 point_P.2 center_C.1 center_C.2 x y ↔ 2*x - y - 2 = 0) ∧
  -- Part 2
  (∀ x y : ℝ, line_with_slope 1 point_P.1 point_P.2 x y →
    chord_length 3 (1 / Real.sqrt 2) = Real.sqrt 34) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_theorem_l275_27534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_14_minutes_unique_equal_cost_duration_l275_27563

/-- Represents the cost of a call under Plan A -/
noncomputable def costPlanA (duration : ℝ) : ℝ :=
  if duration ≤ 8 then 0.60 else 0.60 + 0.06 * (duration - 8)

/-- Represents the cost of a call under Plan B -/
def costPlanB (duration : ℝ) : ℝ := 0.08 * duration

/-- Theorem stating that the costs of Plan A and Plan B are equal for a 14-minute call -/
theorem equal_cost_at_14_minutes : costPlanA 14 = costPlanB 14 := by
  sorry

/-- Theorem stating that 14 minutes is the only duration where the costs are equal -/
theorem unique_equal_cost_duration (d : ℝ) : 
  costPlanA d = costPlanB d ↔ d = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_14_minutes_unique_equal_cost_duration_l275_27563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photovoltaic_power_station_l275_27571

/-- Photovoltaic power station problem -/
theorem photovoltaic_power_station 
  (annual_expense : ℝ) 
  (years : ℕ) 
  (build_cost_coeff : ℝ) 
  (k : ℝ) 
  (x : ℝ) : 
  annual_expense = 24 ∧ 
  years = 16 ∧ 
  build_cost_coeff = 0.12 ∧ 
  x ≥ 0 → 
  (k = 1200 ∧ 
   let F := fun y => years * k / (y + 50) + build_cost_coeff * y;
   (∃ x_min : ℝ, x_min = 350 ∧ 
    ∀ y : ℝ, y ≥ 0 → F y ≥ F x_min) ∧
   F 350 = 90 ∧
   ∀ y : ℝ, 100 ≤ y ∧ y ≤ 3050/3 ↔ F y ≤ 140) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_photovoltaic_power_station_l275_27571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_center_relation_l275_27532

-- Define the origin
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the fixed point (a, b, c)
variable (a b c : ℝ)

-- Define the center of the sphere (p, q, r)
variable (p q r : ℝ)

-- Define points A, B, C on the axes
def A (p : ℝ) : ℝ × ℝ × ℝ := (2*p, 0, 0)
def B (q : ℝ) : ℝ × ℝ × ℝ := (0, 2*q, 0)
def C (r : ℝ) : ℝ × ℝ × ℝ := (0, 0, 2*r)

-- Define the radius of the sphere
variable (R : ℝ)

-- Axiom: A, B, C are distinct from O
axiom distinct_points : A p ≠ O ∧ B q ≠ O ∧ C r ≠ O

-- Axiom: (a, b, c) lies on the plane through A, B, C
axiom point_on_plane : a / (2*p) + b / (2*q) + c / (2*r) = 1

-- Axiom: (p, q, r) is equidistant from O, A, B, C
axiom equidistant : 
  p^2 + q^2 + r^2 = R^2 ∧
  (p - 2*p)^2 + q^2 + r^2 = R^2 ∧
  p^2 + (q - 2*q)^2 + r^2 = R^2 ∧
  p^2 + q^2 + (r - 2*r)^2 = R^2

-- Theorem to prove
theorem sphere_center_relation : a^2/p^2 + b^2/q^2 + c^2/r^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_center_relation_l275_27532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l275_27511

-- Define the circles and points
variable (O₁ O₂ Γ : Set (EuclideanSpace ℝ (Fin 2)))
variable (P Q A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
axiom circles_intersect : P ∈ O₁ ∧ P ∈ O₂ ∧ Q ∈ O₁ ∧ Q ∈ O₂
axiom common_tangent : A ∈ O₁ ∧ B ∈ O₂
axiom gamma_passes : A ∈ Γ ∧ B ∈ Γ
axiom gamma_intersects : C ∈ Γ ∧ C ∈ O₂ ∧ D ∈ Γ ∧ D ∈ O₁

-- Define the theorem
theorem ratio_equality : 
  (dist C P) / (dist C Q) = (dist D P) / (dist D Q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l275_27511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l275_27569

/-- Predicate to check if a triangle is acute -/
def IsAcute (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a line is an altitude of a triangle -/
def IsAltitude (A D B C : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the area of a triangle -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given an acute triangle ABC with altitude AD, prove its area is 289/250 under specific conditions -/
theorem triangle_area_proof (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  IsAcute A B C → 
  IsAltitude A D B C → 
  AB + AC * 4 * Real.sqrt (2/17) = Real.sqrt 17 → 
  4 * BC + 5 * AD = 17 →
  TriangleArea A B C = 289/250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l275_27569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l275_27586

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star (2^x) (2^(-x))

-- Theorem statement
theorem range_of_f :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 1) ∧
  (∀ ε > 0, ∃ x, f x < ε) ∧
  (∃ x, f x = 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l275_27586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l275_27560

/-- Represents the number of students who read only book A -/
def A : ℕ := sorry

/-- Represents the number of students who read only book B -/
def B : ℕ := sorry

/-- Represents the number of students who read both books A and B -/
def AB : ℕ := sorry

/-- The total number of students surveyed -/
def total : ℕ := A + B + AB

/-- 20% of those who read book A also read book B -/
axiom condition1 : AB = (A + AB) / 5

/-- 25% of those who read book B also read book A -/
axiom condition2 : AB = (B + AB) / 4

/-- The difference between students who read only A and only B is 75 -/
axiom condition3 : A - B = 75

/-- Each student read at least one of the books -/
axiom condition4 : total > 0

theorem survey_result : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l275_27560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_sequence_length_l275_27524

/-- A sequence of numbers satisfying specific average conditions -/
structure NumberSequence where
  numbers : List ℝ
  avg_all : (numbers.sum / numbers.length : ℝ) = 60
  avg_first_six : (numbers.take 6).sum / 6 = 78
  avg_last_six : (numbers.reverse.take 6).sum / 6 = 75
  sixth_number : numbers.length > 5 ∧ numbers[5]! = 258

/-- Theorem stating that a NumberSequence has 11 elements -/
theorem number_sequence_length (seq : NumberSequence) : seq.numbers.length = 11 := by
  sorry

#check number_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_sequence_length_l275_27524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l275_27596

theorem binomial_expansion_constant_term (a : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ (x - a / x)^6 = 20 + (x : ℝ) * (0 : ℝ) + (x^2 : ℝ) * (0 : ℝ) + 
   (x^3 : ℝ) * (0 : ℝ) + (x^4 : ℝ) * (0 : ℝ) + (x^5 : ℝ) * (0 : ℝ) + (x^6 : ℝ) * (0 : ℝ)) → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l275_27596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cost_constant_l275_27576

/-- The cost of a train ticket in rubles -/
noncomputable def ticket_cost : ℝ := 50

/-- The fine for traveling without a ticket in rubles -/
noncomputable def fine : ℝ := 450

/-- The probability of encountering a conductor on a trip -/
noncomputable def conductor_probability : ℝ := 1 / 10

/-- The expected cost of a trip given the probability of buying a ticket -/
noncomputable def expected_cost (p : ℝ) : ℝ := 
  p * ticket_cost + (1 - p) * (conductor_probability * (ticket_cost + fine))

theorem expected_cost_constant : 
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → expected_cost p = ticket_cost := by
  sorry

#check expected_cost_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cost_constant_l275_27576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l275_27589

noncomputable def is_valid_point (x y : ℝ) : Prop :=
  x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem longest_side_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_valid_point x₁ y₁ ∧
    is_valid_point x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ),
      is_valid_point x₃ y₃ →
      is_valid_point x₄ y₄ →
      distance x₃ y₃ x₄ y₄ ≤ distance x₁ y₁ x₂ y₂ ∧
      distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l275_27589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_sum_greater_than_two_l275_27550

open Real

/-- The function f(x) = (mx - 1) / x - ln(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * x - 1) / x - log x

theorem two_zeros_sum_greater_than_two (m : ℝ) (x₁ x₂ : ℝ) :
  0 < x₁ → x₁ < x₂ → f m x₁ = 0 → f m x₂ = 0 → x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_sum_greater_than_two_l275_27550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_lineups_count_l275_27566

def team_size : ℕ := 14
def lineup_size : ℕ := 5
def incompatible_players : ℕ := 3

theorem starting_lineups_count :
  (Nat.choose (team_size - incompatible_players) lineup_size) +
  (incompatible_players * Nat.choose (team_size - incompatible_players) (lineup_size - 1)) = 1452 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_lineups_count_l275_27566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_perpendicular_line_passes_through_focus_l275_27582

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Point M on the ellipse C -/
structure PointM where
  x : ℝ
  y : ℝ
  on_ellipse : C x y

/-- Point N, the foot of the perpendicular from M to the x-axis -/
def N (m : PointM) : ℝ × ℝ := (m.x, 0)

/-- Point P satisfying vector(NP) = sqrt(2) * vector(NM) -/
noncomputable def P (m : PointM) : ℝ × ℝ := (m.x, Real.sqrt 2 * m.y)

/-- Point Q on the line x = -3 -/
structure PointQ where
  y : ℝ

/-- The dot product of OP and PQ equals 1 -/
def dot_product_condition (m : PointM) (q : PointQ) : Prop :=
  let p := P m
  (p.1 * ((-3) - p.1) + p.2 * (q.y - p.2)) = 1

theorem trajectory_of_P :
  ∀ (x y : ℝ), (∃ (m : PointM), P m = (x, y)) ↔ x^2 + y^2 = 2 :=
sorry

theorem perpendicular_line_passes_through_focus (m : PointM) (q : PointQ) :
  dot_product_condition m q →
  ∃ (t : ℝ), P m + t • ((q.y - (P m).2), ((P m).1 - (-3))) = (-1, 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_perpendicular_line_passes_through_focus_l275_27582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_special_function_l275_27553

open Real MeasureTheory Interval

theorem integral_of_special_function :
  ∀ f : ℝ → ℝ, Differentiable ℝ f →
  (∀ x, f x = x^2 + 2 * ((deriv f) 2) * x + 3) →
  ∫ x in (Set.Icc 0 3), f x = -18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_special_function_l275_27553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_of_3_and_2048_l275_27573

noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem closest_integer_to_harmonic_mean_of_3_and_2048 :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |harmonic_mean 3 2048 - ↑n| ≤ |harmonic_mean 3 2048 - ↑m| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_of_3_and_2048_l275_27573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_pairs_2007_l275_27528

/-- The number of ordered pairs of positive integers (x,y) satisfying xy = 2007 -/
def number_of_pairs : ℕ := 6

/-- The prime factorization of 2007 -/
def prime_factorization_2007 : List (ℕ × ℕ) := [(3, 2), (223, 1)]

/-- Theorem stating that the number of ordered pairs (x,y) of positive integers
    satisfying xy = 2007 is equal to 6, given the prime factorization of 2007 -/
theorem number_of_pairs_2007 :
  (∀ (x y : ℕ), x * y = 2007 ↔ (x, y) ∈ {p : ℕ × ℕ | p.fst * p.snd = 2007 ∧ p.fst > 0 ∧ p.snd > 0}) →
  (List.prod (prime_factorization_2007.map (λ p => p.snd + 1))) = number_of_pairs :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_pairs_2007_l275_27528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_clovers_l275_27568

/-- The total number of clovers June picked -/
def total_clovers : ℚ := 554

/-- The proportion of clovers with 3 petals -/
def three_petal_prop : ℚ := 75 / 100

/-- The proportion of clovers with 2 petals -/
def two_petal_prop : ℚ := 24 / 100

/-- The proportion of clovers with 4 petals -/
def four_petal_prop : ℚ := 1 / 100

/-- The earnings in cents for each cloverleaf -/
def earnings_per_leaf : ℚ := 1

/-- The total earnings in cents -/
def total_earnings : ℚ := 554

/-- Theorem stating that the total number of clovers June picked is 554 -/
theorem june_clovers : 
  (three_petal_prop * total_clovers + two_petal_prop * total_clovers + four_petal_prop * total_clovers) * earnings_per_leaf = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_clovers_l275_27568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_20_marked_subgrid_l275_27539

/-- A configuration of marked cells on a grid plane. -/
structure MarkedGrid where
  cells : Finset (ℕ × ℕ)
  marked_count : Nat
  marked_count_eq : cells.card = marked_count

/-- A rectangular subgrid of a grid plane. -/
structure RectangularSubgrid where
  top_left : ℕ × ℕ
  bottom_right : ℕ × ℕ

/-- The number of marked cells in a rectangular subgrid. -/
def marked_in_subgrid (grid : MarkedGrid) (subgrid : RectangularSubgrid) : Nat :=
  (grid.cells ∩ (Finset.range (subgrid.bottom_right.1 - subgrid.top_left.1 + 1)).product
    (Finset.range (subgrid.bottom_right.2 - subgrid.top_left.2 + 1))).card

/-- Theorem stating that there exists a configuration of 40 marked cells
    where no rectangular subgrid contains exactly 20 marked cells. -/
theorem exists_no_20_marked_subgrid :
  ∃ (grid : MarkedGrid),
    grid.marked_count = 40 ∧
    ∀ (subgrid : RectangularSubgrid),
      marked_in_subgrid grid subgrid ≠ 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_20_marked_subgrid_l275_27539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_on_specific_wall_l275_27512

/-- The number of square tiles that can fit on a rectangular wall -/
def tiles_on_wall (wall_width : ℚ) (wall_length : ℚ) (tile_side : ℚ) : ℕ :=
  (wall_width * 100 * wall_length * 100 / (tile_side * tile_side)).floor.toNat

/-- Theorem stating the number of tiles that can fit on the given wall -/
theorem tiles_on_specific_wall :
  tiles_on_wall (3 + 1/4) (2 + 3/4) 25 = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_on_specific_wall_l275_27512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_for_money_order_l275_27541

/-- Represents the size of a juice pack -/
inductive JuiceSize
| Small
| Medium
| Large

/-- Represents the cost and quantity of a juice pack -/
structure JuicePack where
  size : JuiceSize
  cost : ℚ
  quantity : ℚ

/-- Calculates the value for money (quantity per unit cost) of a juice pack -/
noncomputable def valueForMoney (pack : JuicePack) : ℚ :=
  pack.quantity / pack.cost

/-- The conditions given in the problem -/
axiom small_pack : JuicePack
axiom medium_pack : JuicePack
axiom large_pack : JuicePack

axiom medium_cost : medium_pack.cost = 14/10 * small_pack.cost
axiom medium_quantity : medium_pack.quantity = 7/10 * large_pack.quantity
axiom large_quantity : large_pack.quantity = 15/10 * small_pack.quantity
axiom large_cost : large_pack.cost = 12/10 * medium_pack.cost

/-- Theorem stating that the value for money order is Small > Large > Medium -/
theorem value_for_money_order :
  valueForMoney small_pack > valueForMoney large_pack ∧
  valueForMoney large_pack > valueForMoney medium_pack := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_for_money_order_l275_27541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l275_27533

noncomputable section

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC is acute-angled
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- Sides a, b, c are opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  (Real.sin A - Real.sin B) * (Real.sin A + Real.sin B) = Real.sin (π/3 - B) * Real.sin (π/3 + B) →
  -- Given condition
  b * c * Real.cos A = 12 →
  -- Given condition
  a = 2 * Real.sqrt 7 →
  -- Given condition
  b < c →
  -- Prove
  A = π/3 ∧ b = 4 ∧ c = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l275_27533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l275_27510

/-- Represents the lawn mowing scenario -/
structure LawnMowing where
  length : ℚ  -- length of the lawn in feet
  width : ℚ   -- width of the lawn in feet
  swath_width : ℚ  -- mower swath width in inches
  overlap : ℚ  -- overlap in inches
  speed_long : ℚ  -- speed for strips longer than 50 feet (feet per hour)

/-- Calculates the time taken to mow the lawn -/
def mowing_time (scenario : LawnMowing) : ℚ :=
  let effective_swath := (scenario.swath_width - scenario.overlap) / 12  -- Convert to feet
  let num_strips := scenario.width / effective_swath
  let total_distance := num_strips * scenario.length
  total_distance / scenario.speed_long

/-- Theorem stating the time taken to mow the lawn is 0.9 hours -/
theorem lawn_mowing_time (scenario : LawnMowing) 
  (h1 : scenario.length = 60)
  (h2 : scenario.width = 120)
  (h3 : scenario.swath_width = 30)
  (h4 : scenario.overlap = 6)
  (h5 : scenario.speed_long = 4000)
  : mowing_time scenario = 9/10 := by
  sorry

#eval mowing_time { length := 60, width := 120, swath_width := 30, overlap := 6, speed_long := 4000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l275_27510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l275_27538

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 1)) / (x - 2) + Real.log (-x^2 + 2*x + 3) / Real.log 2

def DomainOf (f : ℝ → ℝ) : Set ℝ := {x : ℝ | ∃ y, f x = y}

theorem domain_of_f :
  DomainOf f = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x < 3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l275_27538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_eq_two_l275_27547

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- Length of the two equal sides -/
  side : ℝ
  /-- Length of the base -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : side > 0 ∧ base > 0
  /-- The area is numerically equal to the perimeter -/
  areaEqualsPerimeter : side * (side * side - (base / 2) ^ 2).sqrt = side + side + base

/-- The inradius of the specific isosceles triangle -/
noncomputable def inradius (t : IsoscelesTriangle) : ℝ :=
  (t.side + t.side + t.base) / 6

/-- Theorem: The inradius of the specific isosceles triangle is 2 -/
theorem inradius_eq_two (t : IsoscelesTriangle)
  (h1 : t.side = 12)
  (h2 : t.base = 18) :
  inradius t = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_eq_two_l275_27547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_conic_has_two_foci_and_directrices_l275_27502

-- Define a conic section
structure ConicSection where
  center : Real × Real
  focus1 : Real × Real
  directrix1 : Real → Real × Real

-- Define the property of a point being on the conic section
noncomputable def on_conic_section (c : ConicSection) (p : Real × Real) : Prop :=
  Real.sqrt ((p.1 - c.focus1.1)^2 + (p.2 - c.focus1.2)^2) =
  Real.sqrt ((p.1 - (c.directrix1 p.1).1)^2 + (p.2 - (c.directrix1 p.1).2)^2)

-- Theorem statement
theorem central_conic_has_two_foci_and_directrices (c : ConicSection) :
  ∃ (focus2 : Real × Real) (directrix2 : Real → Real × Real),
    focus2 ≠ c.focus1 ∧
    directrix2 ≠ c.directrix1 ∧
    ∀ (p : Real × Real), on_conic_section c p ↔
      Real.sqrt ((p.1 - focus2.1)^2 + (p.2 - focus2.2)^2) =
      Real.sqrt ((p.1 - (directrix2 p.1).1)^2 + (p.2 - (directrix2 p.1).2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_conic_has_two_foci_and_directrices_l275_27502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l275_27515

theorem hyperbola_equation (x y : ℝ) : 
  (∃ a b c d : ℝ, 
    -- Given hyperbola equation
    (∀ x y : ℝ, x^2 / 16 - y^2 / 4 = 1 → (x = a ∧ y = 0) ∨ (x = -a ∧ y = 0)) ∧
    -- Point on the new hyperbola
    (3 * Real.sqrt 2)^2 / c^2 - 2^2 / d^2 = 1 ∧
    -- Shared focus
    ((c = a ∧ d = b) ∨ (c = -a ∧ d = b)) ∧
    -- Positive denominator condition
    d^2 > 0) →
  -- Conclusion: equation of the new hyperbola
  x^2 / 12 - y^2 / 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l275_27515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reach_area_l275_27523

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex -/
noncomputable def doghouse_area (side_length : ℝ) (tether_length : ℝ) : ℝ :=
  let main_sector_area := Real.pi * tether_length^2 * (270 / 360)
  let small_sector_area := 2 * (Real.pi * (tether_length - side_length)^2 * (30 / 360))
  main_sector_area + small_sector_area

/-- Theorem stating the approximate area the dog can reach -/
theorem dog_reach_area :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |doghouse_area 1.5 2.5 - 4.73 * Real.pi| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reach_area_l275_27523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_logarithmic_relations_l275_27516

theorem sum_of_logarithmic_relations (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : Real.log b / Real.log a = 3/2)
  (h2 : Real.log d / Real.log c = 5/4)
  (h3 : a - c = 9) :
  a + b + c + d = 198 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_logarithmic_relations_l275_27516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equality_implies_power_relation_l275_27592

theorem gcd_equality_implies_power_relation (m n : ℕ) :
  (∀ k : ℕ, Nat.gcd (11 * k - 1) m = Nat.gcd (11 * k - 1) n) →
  ∃ l : ℕ, m = n * (11 ^ l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equality_implies_power_relation_l275_27592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l275_27514

/-- A line passing through a point and intersecting a circle -/
structure LineIntersectingCircle where
  /-- The x-coordinate of the point the line passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the line passes through -/
  point_y : ℝ
  /-- The x-coordinate of the circle's center -/
  circle_center_x : ℝ
  /-- The y-coordinate of the circle's center -/
  circle_center_y : ℝ
  /-- The radius of the circle -/
  circle_radius : ℝ
  /-- The length of the chord formed by the intersection -/
  chord_length : ℝ

/-- The equation of the line is either vertical or has a specific slope -/
def line_equation (l : LineIntersectingCircle) : Prop :=
  (∀ x y : ℝ, x = l.point_x) ∨
  (∃ a b c : ℝ, a * l.point_x + b * l.point_y + c = 0 ∧
                ∀ x y : ℝ, a * x + b * y + c = 0 ∧
                a = 4 ∧ b = 3 ∧ c = 25)

/-- The main theorem stating the equation of the line -/
theorem line_equation_theorem (l : LineIntersectingCircle)
  (h1 : l.point_x = -4)
  (h2 : l.point_y = -3)
  (h3 : l.circle_center_x = -1)
  (h4 : l.circle_center_y = -2)
  (h5 : l.circle_radius = 5)
  (h6 : l.chord_length = 8) :
  line_equation l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l275_27514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_chain_odd_winding_number_l275_27530

/-- A polygonal chain in 2D space -/
structure PolygonalChain (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- The property of being symmetric with respect to a point -/
def isSymmetricAbout (chain : PolygonalChain n) (center : ℝ × ℝ) : Prop :=
  ∀ (i : Fin n), ∃ (j : Fin n), chain.vertices j = 2 • center - chain.vertices i

/-- The winding number of a polygonal chain around a point -/
noncomputable def windingNumber (chain : PolygonalChain n) (center : ℝ × ℝ) : ℝ :=
  sorry

theorem symmetric_chain_odd_winding_number
  {n : ℕ} (chain : PolygonalChain (n + 1)) (center : ℝ × ℝ)
  (h_closed : chain.vertices 0 = chain.vertices n)
  (h_symmetric : isSymmetricAbout chain center)
  (h_not_on_chain : ∀ (i : Fin (n + 1)), chain.vertices i ≠ center) :
  ∃ (k : ℤ), windingNumber chain center = 2 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_chain_odd_winding_number_l275_27530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l275_27595

theorem parabola_tangent_to_line (b : ℝ) :
  (∀ x y : ℝ, y = b * x^2 + 4 ∧ y = 2 * x + 1 → 
    ∃! x', b * x'^2 + 4 = 2 * x' + 1) ↔ b = 1/3 := by
  sorry

#check parabola_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l275_27595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_pairs_card_l275_27513

open Set

noncomputable def reciprocal_sum_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun pair => pair.1 > 0 ∧ pair.2 > 0 ∧ (1 : ℚ) / pair.1 + (1 : ℚ) / pair.2 = 1/3) (Finset.range 13 ×ˢ Finset.range 13)

theorem reciprocal_sum_pairs_card : 
  reciprocal_sum_pairs.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_pairs_card_l275_27513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l275_27526

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line l₁
def l₁ (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the line l₂
def l₂ (x y : ℝ) : Prop := x + 2*y + 2 = 0

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the intersection points P and Q
noncomputable def P (k : ℝ) : ℝ × ℝ := sorry
noncomputable def Q (k : ℝ) : ℝ × ℝ := sorry

-- Define the midpoint M
noncomputable def M (k : ℝ) : ℝ × ℝ := ((k^2 + 4*k + 3) / (k^2 + 1), (4*k^2 + 2*k) / (k^2 + 1))

-- Define the intersection point N
noncomputable def N (k : ℝ) : ℝ × ℝ := ((2*k - 2) / (2*k + 1), -3*k / (2*k + 1))

-- State the theorem
theorem circle_line_intersection (k : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l₁ x₁ y₁ k ∧ l₁ x₂ y₂ k ∧ (x₁, y₁) ≠ (x₂, y₂)) →
  k > 3/4 ∧
  M k = ((k^2 + 4*k + 3) / (k^2 + 1), (4*k^2 + 2*k) / (k^2 + 1)) ∧
  ‖M k - A‖ * ‖N k - A‖ = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l275_27526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_five_hours_l275_27591

/-- Represents the train's journey between Scottsdale and Sherbourne -/
structure TrainJourney where
  track_length : ℝ
  forest_grove_fraction : ℝ
  forest_grove_to_sherbourne_time : ℝ

/-- Calculates the round trip time for the train journey -/
noncomputable def round_trip_time (journey : TrainJourney) : ℝ :=
  let forest_grove_distance := journey.track_length * journey.forest_grove_fraction
  let forest_grove_to_sherbourne_distance := journey.track_length - forest_grove_distance
  let train_speed := forest_grove_to_sherbourne_distance / journey.forest_grove_to_sherbourne_time
  2 * (journey.track_length / train_speed)

/-- Theorem stating that the round trip time for the given conditions is 5 hours -/
theorem round_trip_time_is_five_hours (journey : TrainJourney) 
    (h1 : journey.track_length = 200)
    (h2 : journey.forest_grove_fraction = 1/5)
    (h3 : journey.forest_grove_to_sherbourne_time = 2) :
    round_trip_time journey = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_five_hours_l275_27591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_is_45_degrees_l275_27581

def line_equation (x : ℝ) : ℝ := x - 1

def slope_of_line : ℝ := 1

noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

theorem angle_of_line_is_45_degrees :
  0 ≤ angle_of_inclination slope_of_line ∧
  angle_of_inclination slope_of_line < π →
  angle_of_inclination slope_of_line = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_is_45_degrees_l275_27581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rotation_matrix_exists_zero_matrix_is_answer_l275_27565

open Matrix

theorem no_rotation_matrix_exists : ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), 
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), 
    M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]] := by sorry

theorem zero_matrix_is_answer : 
  (¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), 
    ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), 
      M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) → 
  (0 : Matrix (Fin 2) (Fin 2) ℝ) = ![![0, 0], ![0, 0]] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rotation_matrix_exists_zero_matrix_is_answer_l275_27565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l275_27562

-- Define the condition p
noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m > 0

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 2*x^2 - m*x - 1

-- Define the condition q
noncomputable def q (m : ℝ) : Prop := ∀ x : ℝ, (deriv (f m)) x ≤ 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p m ∧ q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l275_27562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_6_5_l275_27540

/-- Represents the speed of a swimmer in still water given downstream and upstream information -/
noncomputable def swimmer_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) : ℝ :=
  (downstream_distance + upstream_distance) / (2 * time)

/-- Theorem stating that a swimmer's speed in still water is 6.5 km/h given specific conditions -/
theorem swimmer_speed_is_6_5 :
  swimmer_speed 16 10 2 = 6.5 := by
  unfold swimmer_speed
  norm_num

-- The following line is commented out because it's not computable
-- #eval swimmer_speed 16 10 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_6_5_l275_27540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l275_27567

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence -/
noncomputable def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2 + 1,
    b := (t.a + t.c - t.b) / 2 + 1,
    c := (t.a + t.b - t.c) / 2 + 1 }

/-- Checks if a triangle is valid (satisfies the triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The initial triangle T₁ -/
def T₁ : Triangle :=
  { a := 1008, b := 1009, c := 1007 }

/-- The sequence of triangles -/
noncomputable def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of the last valid triangle in the sequence is 1515/256 -/
theorem last_triangle_perimeter :
  ∃ n : ℕ, (∀ k < n, isValidTriangle (triangleSequence k)) ∧
           ¬isValidTriangle (triangleSequence n) ∧
           perimeter (triangleSequence (n - 1)) = 1515 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l275_27567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincidence_l275_27503

/-- The value of k for which the right focus of the hyperbola x²-(y²/k²)=1 (k>0) 
    coincides with the focus of the parabola y²=8x -/
theorem hyperbola_parabola_focus_coincidence (k : ℝ) : 
  k > 0 → 
  (∃ x y : ℝ, x^2 - y^2/k^2 = 1 ∧ y^2 = 8*x ∧ 
   x = Real.sqrt (1 + k^2) ∧ x = 2) → 
  k = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincidence_l275_27503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l275_27570

theorem min_value_theorem (y : ℝ) (h : y > 0) : 
  9 * y^7 + 7 * y^(-6 : ℤ) ≥ 16 ∧ 
  (9 * y^7 + 7 * y^(-6 : ℤ) = 16 ↔ y = 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l275_27570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l275_27555

open Real

-- Define the shaded areas for each figure
noncomputable def shaded_area_A (side : ℝ) : ℝ :=
  side^2 - Real.pi * (side/2)^2

noncomputable def shaded_area_B (leg1 leg2 : ℝ) : ℝ :=
  (1/2) * leg1 * leg2 - Real.pi * (min leg1 leg2)^2 / 4

noncomputable def shaded_area_C (side : ℝ) : ℝ :=
  side^2 - Real.pi * (side/2)^2

-- Theorem statement
theorem largest_shaded_area :
  shaded_area_B 3 4 > shaded_area_A 3 ∧ shaded_area_B 3 4 > shaded_area_C 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l275_27555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_max_area_and_intersecting_lines_l275_27544

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a line passing through point A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Helper function for triangle area
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Part I: Tangent line
theorem tangent_line_equation :
  ∃ k : ℝ, (∀ x y : ℝ, line_through_A k x y ↔ 3*x - 4*y - 3 = 0) ∧
  (∀ x y : ℝ, line_through_A k x y → circle_C x y → (x, y) = (1, 0) ∨ (x, y) = (5, 3)) :=
by sorry

-- Part II: Intersecting line and maximum area
theorem max_area_and_intersecting_lines :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧ circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
    (∃ k : ℝ, line_through_A k P.1 P.2 ∧ line_through_A k Q.1 Q.2) ∧
    (∀ R : ℝ × ℝ, circle_C R.1 R.2 ∧ (∃ k : ℝ, line_through_A k R.1 R.2) →
      area_triangle (3, 4) P Q ≥ area_triangle (3, 4) P R ∧
      area_triangle (3, 4) P Q ≥ area_triangle (3, 4) Q R) ∧
    area_triangle (3, 4) P Q = 2) ∧
  (∀ x y : ℝ, (∃ k : ℝ, line_through_A k x y ∧ circle_C x y) →
    (y = 7*x - 7 ∨ y = x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_max_area_and_intersecting_lines_l275_27544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_degrees_to_radians_l275_27593

/-- Conversion factor from degrees to radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

/-- Conversion from degrees to radians -/
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ := degrees * deg_to_rad

theorem negative_150_degrees_to_radians :
  degrees_to_radians (-150) = -5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_degrees_to_radians_l275_27593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l275_27501

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (t x y : ℝ) : Prop := x = t * y - 1

-- Define the intersection points
def intersection (t y₁ y₂ : ℝ) : Prop :=
  ellipse (t * y₁ - 1) y₁ ∧ ellipse (t * y₂ - 1) y₂ ∧ y₁ ≠ y₂

-- Define the area of the triangle
noncomputable def triangle_area (t y₁ y₂ : ℝ) : ℝ :=
  (1 / 2) * |y₁ - y₂|

theorem ellipse_properties :
  -- The ellipse passes through (1, √3/2)
  ellipse 1 (Real.sqrt 3 / 2) →
  -- The left vertex is at (-2, 0)
  ellipse (-2) 0 →
  -- Properties to prove
  (∃ e : ℝ, e = Real.sqrt 3 / 2 ∧ e^2 = 3 / 4) ∧
  (∀ t y₁ y₂ : ℝ, intersection t y₁ y₂ → y₁ * y₂ = -3 / (t^2 + 4)) ∧
  (∃ t : ℝ, triangle_area t (Real.sqrt (4 / (t^2 + 4) + 12 / (t^2 + 4)) / 2)
                           (-Real.sqrt (4 / (t^2 + 4) + 12 / (t^2 + 4)) / 2) = 4 / 5 ∧
                (t = 1 ∨ t = -1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l275_27501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l275_27549

/-- A piecewise function f defined as:
    f(x) = x + 10 for x < a
    f(x) = x^2 - 2x for x ≥ a
-/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then x + 10 else x^2 - 2*x

theorem range_of_a (a : ℝ) :
  (∀ b : ℝ, ∃ x₀ : ℝ, f a x₀ = b) →
  a ∈ Set.Icc (-11 : ℝ) 5 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l275_27549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_y_intercept_l275_27509

/-- Given a line parallel to y = -3x + 6 and passing through (1, -4), its y-intercept is -1 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x, HasDerivAt b (-3) x) →  -- b is parallel to y = -3x + 6
  b 1 = -4 →                    -- b passes through (1, -4)
  b 0 = -1 :=                   -- y-intercept of b is -1
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_y_intercept_l275_27509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_surface_area_ratio_l275_27558

/-- A sphere circumscribed by a cone with an equilateral axial section -/
structure SphereInCone where
  r : ℝ  -- radius of the sphere
  h : r > 0  -- radius is positive

namespace SphereInCone

variable (s : SphereInCone)

/-- Volume of the sphere -/
noncomputable def sphereVolume : ℝ := (4 / 3) * Real.pi * s.r^3

/-- Surface area of the sphere -/
noncomputable def sphereSurfaceArea : ℝ := 4 * Real.pi * s.r^2

/-- Volume of the cone -/
noncomputable def coneVolume : ℝ := (1 / 3) * Real.pi * (3 * s.r^2) * (3 * s.r)

/-- Surface area of the cone -/
noncomputable def coneSurfaceArea : ℝ := 9 * Real.pi * s.r^2

/-- Theorem: The volume ratio of the sphere to the cone is 4/9 -/
theorem volume_ratio : sphereVolume s / coneVolume s = 4 / 9 := by
  sorry

/-- Theorem: The surface area ratio of the sphere to the cone is 4/9 -/
theorem surface_area_ratio : sphereSurfaceArea s / coneSurfaceArea s = 4 / 9 := by
  sorry

end SphereInCone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_surface_area_ratio_l275_27558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_34_l275_27575

/-- Regular polygon with n sides and side length s -/
structure RegularPolygon where
  n : ℕ
  s : ℝ
  n_ge_3 : n ≥ 3

/-- The interior angle of a regular polygon -/
noncomputable def interior_angle (p : RegularPolygon) : ℝ :=
  180 * (p.n - 2) / p.n

/-- Three polygons meeting at a point -/
structure PolygonConfiguration where
  p1 : RegularPolygon
  p2 : RegularPolygon
  p3 : RegularPolygon
  square_exists : p1.n = 4 ∨ p2.n = 4 ∨ p3.n = 4
  congruent_pair : (p1 = p2) ∨ (p1 = p3) ∨ (p2 = p3)
  side_length_2 : p1.s = 2 ∧ p2.s = 2 ∧ p3.s = 2
  angle_sum_360 : interior_angle p1 + interior_angle p2 + interior_angle p3 = 360

/-- The perimeter of the new polygon formed by the configuration -/
def new_polygon_perimeter (config : PolygonConfiguration) : ℝ :=
  2 * (config.p1.n + config.p2.n + config.p3.n - 3)

/-- The main theorem -/
theorem max_perimeter_34 (config : PolygonConfiguration) :
  new_polygon_perimeter config ≤ 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_34_l275_27575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_income_l275_27519

def nose_price : ℚ := 20
def ear_price_increase : ℚ := 0.5
def noses_pierced : ℕ := 6
def ears_pierced : ℕ := 9

def total_income : ℚ := nose_price * noses_pierced + 
  (nose_price * (1 + ear_price_increase)) * ears_pierced

theorem jerry_income : total_income = 390 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_income_l275_27519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l275_27556

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ∃ (v : ℝ), v ∈ ({4, 0, -4} : Set ℝ) ∧ 
  x / abs x + y / abs y + z / abs z + (x * y * z) / abs (x * y * z) = v :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l275_27556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_quotient_l275_27564

theorem polynomial_division_quotient : ∃ (r : ℚ), 
  (fun x : ℚ => 8*x^3 + 26*x^2 + 69*x + 211) * (fun x : ℚ => x - 3) + (fun _ : ℚ => r) = 
  (fun x : ℚ => 8*x^4 + 2*x^3 - 9*x^2 + 4*x - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_quotient_l275_27564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_one_to_zero_l275_27583

noncomputable def f (x : ℝ) := 1 / (x^2 - 1)

theorem f_increasing_on_neg_one_to_zero :
  ∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → x₂ < 0 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_one_to_zero_l275_27583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_specific_investment_difference_l275_27500

/-- Calculates the final amount for an investment with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (compounds_per_year : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (time * compounds_per_year)

/-- The difference in final amounts between semi-annual and annual compounding -/
theorem investment_difference (principal : ℝ) (rate : ℝ) (time : ℝ) : 
  ∃ (diff : ℝ), 
    abs (compound_interest principal rate time 2 - compound_interest principal rate time 1 - diff) < 0.01 ∧ 
    abs (diff - 98.94) < 0.01 := by
  sorry

/-- Specific case with given values -/
theorem specific_investment_difference : 
  ∃ (diff : ℝ),
    abs (compound_interest 60000 0.05 3 2 - compound_interest 60000 0.05 3 1 - diff) < 0.01 ∧ 
    abs (diff - 98.94) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_specific_investment_difference_l275_27500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_six_terms_is_90_l275_27537

/-- An arithmetic sequence with a_3 + a_4 = 30 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a1 d : ℚ), ∀ n, a n = a1 + (n - 1) * d ∧ a 3 + a 4 = 30

/-- The sum of the first n terms of an arithmetic sequence -/
def SumFirstNTerms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (a 1 + a n)

theorem sum_first_six_terms_is_90 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  SumFirstNTerms a 6 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_six_terms_is_90_l275_27537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l275_27521

def A : Set ℤ := {1, 2, 3, 5, 7}

def B : Set ℤ := {x | 1 < x ∧ x ≤ 6}

def U : Set ℤ := A ∪ B

theorem intersection_complement_equality :
  A ∩ (U \ B) = {1, 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l275_27521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_isosceles_triangle_area_l275_27527

/-- An isosceles triangle with specific proportions -/
structure SpecialIsoscelesTriangle where
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The shorter leg of the triangle
  short_leg : ℝ
  -- Ensure the perimeter is positive
  perimeter_pos : perimeter > 0
  -- Ensure the short leg is positive
  short_leg_pos : short_leg > 0
  -- The perimeter is twice the sum of all sides
  perimeter_eq : perimeter = short_leg + 2 * short_leg + 3 * short_leg

/-- The area of the special isosceles triangle -/
noncomputable def triangle_area (t : SpecialIsoscelesTriangle) : ℝ :=
  (Real.sqrt 8 * t.perimeter ^ 2) / 6

/-- Theorem stating the area of the special isosceles triangle -/
theorem special_isosceles_triangle_area (t : SpecialIsoscelesTriangle) :
  triangle_area t = (Real.sqrt 8 * (t.perimeter / 2) ^ 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_isosceles_triangle_area_l275_27527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l275_27585

/-- The circle with center (1, 0) and radius 2 -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- A line passing through the point (2, 1) -/
def line_eq (m n : ℝ) (x y : ℝ) : Prop := m*x + n*y - 2*m - n = 0

/-- The shortest chord length intercepted by the circle and any line -/
noncomputable def shortest_chord_length : ℝ := 2 * Real.sqrt 2

theorem shortest_chord_theorem (m n : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq m n x₁ y₁ ∧ line_eq m n x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = shortest_chord_length ∧
    ∀ (x y : ℝ), circle_eq x y ∧ line_eq m n x y →
      Real.sqrt ((x - x₁)^2 + (y - y₁)^2) ≤ shortest_chord_length ∧
      Real.sqrt ((x - x₂)^2 + (y - y₂)^2) ≤ shortest_chord_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l275_27585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_strategy_l275_27507

/-- Represents the game state with red and green pieces -/
structure GameState where
  red : ℕ
  green : ℕ

/-- Represents a player's move -/
structure Move where
  color : Bool  -- true for red, false for green
  amount : ℕ

/-- Returns the even part of a natural number -/
def evenPart (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2^(n.log2 - n.log2 % 1)

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  if move.color
  then move.amount > 0 ∧ move.amount ≤ state.red ∧ state.green % move.amount = 0
  else move.amount > 0 ∧ move.amount ≤ state.green ∧ state.red % move.amount = 0

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  if move.color
  then { red := state.red - move.amount, green := state.green }
  else { red := state.red, green := state.green - move.amount }

/-- Represents the winning strategy for the game -/
inductive WinningStrategy : GameState → Prop
  | secondPlayer (state : GameState) :
      evenPart state.red = evenPart state.green →
      (∀ move, isValidMove state move → WinningStrategy (applyMove state move)) →
      WinningStrategy state
  | firstPlayer (state : GameState) :
      evenPart state.red ≠ evenPart state.green →
      (∃ move, isValidMove state move ∧
               evenPart (applyMove state move).red = evenPart (applyMove state move).green) →
      WinningStrategy state

/-- The main theorem stating the winning strategy for the game -/
theorem game_winning_strategy (initialState : GameState) :
  WinningStrategy initialState :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_strategy_l275_27507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_half_l275_27506

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a line passing through a vertex and a focus of an ellipse -/
structure VertexFocusLine (e : Ellipse) where
  slope : ℝ
  intercept : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center of the ellipse to a line -/
noncomputable def distanceToLine (e : Ellipse) (l : VertexFocusLine e) : ℝ :=
  abs l.intercept / Real.sqrt (1 + l.slope^2)

theorem eccentricity_is_half (e : Ellipse) (l : VertexFocusLine e) 
  (h : distanceToLine e l = e.b / 4) : eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_half_l275_27506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2015_is_one_fifth_l275_27531

def sequence_a : ℕ → ℚ
  | 0 => 4/5  -- Adding the base case for 0
  | 1 => 4/5
  | n + 1 => 
    let prev := sequence_a n
    if 0 ≤ prev ∧ prev ≤ 1/2 then 2 * prev
    else if 1/2 < prev ∧ prev ≤ 1 then 2 * prev - 1
    else prev  -- This case should never occur based on the problem definition

theorem sequence_a_2015_is_one_fifth : sequence_a 2015 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2015_is_one_fifth_l275_27531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l275_27557

/-- Ellipse parameters -/
structure EllipseParams where
  t : ℝ
  k : ℝ
  h : k > 0

/-- Points on the ellipse -/
structure EllipsePoints where
  A : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Main theorem -/
theorem ellipse_properties (params : EllipseParams) (points : EllipsePoints) :
  let E := fun x y ↦ x^2 / params.t + y^2 / 3 = 1
  let A := points.A
  let M := points.M
  let N := points.N
  (∀ x y, E x y → (x, y) ∈ Set.range (fun t ↦ (t, 0)) ∨ (x, y) ∈ Set.range (fun t ↦ (-t, 0))) →  -- foci on x-axis
  (E A.1 A.2 ∧ A.1 < 0 ∧ A.2 = 0) →  -- A is left vertex
  (E M.1 M.2 ∧ E N.1 N.2) →  -- M and N are on E
  (M.2 - A.2 = params.k * (M.1 - A.1)) →  -- line AM has slope k
  ((M.1 - A.1) * (N.1 - A.1) + (M.2 - A.2) * (N.2 - A.2) = 0) →  -- MA ⊥ NA
  (params.t = 4 → (M.1 - A.1)^2 + (M.2 - A.2)^2 = (N.1 - A.1)^2 + (N.2 - A.2)^2 →
    1/2 * abs ((M.1 - A.1) * (N.2 - A.2) - (M.2 - A.2) * (N.1 - A.1)) = 144/49) ∧
  (2 * ((M.1 - A.1)^2 + (M.2 - A.2)^2) = (N.1 - A.1)^2 + (N.2 - A.2)^2 →
    ∃ (a b : ℝ), a = Real.rpow 2 (1/3) ∧ b = 2 ∧ a < params.k ∧ params.k < b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l275_27557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_neg_2i_in_third_quadrant_l275_27536

-- Define the complex number e^(-2i)
noncomputable def z : ℂ := Complex.exp (-2 * Complex.I)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem e_neg_2i_in_third_quadrant : third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_neg_2i_in_third_quadrant_l275_27536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_f_eq_two_l275_27590

/-- Piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x + 3 else x^2 - 2

/-- Theorem stating that -2 is the unique real number m such that f(m) = 2 -/
theorem unique_m_for_f_eq_two :
  ∃! m : ℝ, f m = 2 ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_f_eq_two_l275_27590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l275_27579

/-- The function g(x) = (x+2)/√(x^2-5x+6) -/
noncomputable def g (x : ℝ) : ℝ := (x + 2) / Real.sqrt (x^2 - 5*x + 6)

/-- The domain of g -/
def domain_g : Set ℝ := {x | x ≤ 2 ∨ x ≥ 3}

/-- Theorem stating that the domain of g is correct -/
theorem domain_of_g :
  {x : ℝ | ∃ y, g x = y} = domain_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l275_27579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l275_27542

open Real

noncomputable def f (x : ℝ) : ℝ := log (x + Real.sqrt (x^2 + 1))

theorem m_range (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, f ((x + 1) / (x - 1)) + f (m / ((x - 1)^2 * (x - 6))) > 0) →
  m ∈ Set.Iic 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l275_27542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l275_27588

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def f (x : ℝ) : ℝ := -Real.log (-x) / Real.log 2

theorem symmetry_about_origin (x : ℝ) (hx : x < 0) :
  f x = -g (-x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l275_27588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_l275_27584

theorem find_y (y : ℝ) : (3 : ℝ)^(y - 2) = (9 : ℝ)^3 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_l275_27584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_message_exchange_l275_27543

structure Conference where
  N : Type -- Set of native scientists
  F : Type -- Set of foreign scientists
  sends_message : N → F → Prop -- native scientist sends message to foreign scientist
  receives_message : F → N → Prop -- foreign scientist sends message to native scientist

theorem conference_message_exchange 
  (conf : Conference)
  (h1 : ∀ n : conf.N, ∃ f : conf.F, conf.sends_message n f)
  (h2 : ∀ f : conf.F, ∃ n : conf.N, conf.receives_message f n)
  (h3 : ∃ n : conf.N, ¬ ∃ f : conf.F, conf.receives_message f n) :
  ∃ S : Set conf.N,
    (Set.diff (Set.univ : Set conf.N) S) = 
      {n : conf.N | ∃ f : conf.F, 
        conf.receives_message f n ∧ 
        ¬ ∃ s : conf.N, s ∈ S ∧ conf.sends_message s f} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_message_exchange_l275_27543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_count_l275_27525

def S : Finset Nat := {2, 4, 6, 8, 10, 12}

def differenceSet (S : Finset Nat) : Finset Nat :=
  Finset.biUnion S (fun a => Finset.image (fun b => a - b) (S.filter (fun b => b < a)))

theorem difference_count : Finset.card (differenceSet S) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_count_l275_27525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l275_27508

-- Define the three functions as noncomputable
noncomputable def f1 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt (Real.log 3 / Real.log x)
noncomputable def f3 (x : ℝ) : ℝ := 3 - 1 / Real.sqrt (Real.log 3 / Real.log x)

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 1)

-- Theorem statement
theorem unique_intersection :
  ∃! p : ℝ × ℝ, p.1 > 0 ∧
  ((f1 p.1 = f2 p.1 ∧ p.2 = f1 p.1) ∨
   (f1 p.1 = f3 p.1 ∧ p.2 = f1 p.1) ∨
   (f2 p.1 = f3 p.1 ∧ p.2 = f2 p.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l275_27508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_two_l275_27587

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (x * Real.sqrt (1 + Real.log x))

-- Define the integral
noncomputable def area_integral : ℝ := ∫ x in (1)..(Real.exp 3), f x

-- State the theorem
theorem area_is_two : area_integral = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_two_l275_27587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l275_27545

noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def isOnLine (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  let (x, y) := p
  a*x + b*y = c

theorem triangle_PQR_area :
  ∃ R : ℝ × ℝ,
    isOnLine R 1 1 4 ∧
    triangleArea (2, 1) (5, 4) R = 4.5 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l275_27545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l275_27522

/-- IsTriangle a b c means a, b, c form a valid triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = 8, b = 6, and c = 4, prove that sin A = √15/4 -/
theorem sin_A_in_triangle (A B C : ℝ) (a b c : ℝ) 
    (h_triangle : IsTriangle a b c)
    (h_a : a = 8)
    (h_b : b = 6)
    (h_c : c = 4) :
  Real.sin A = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l275_27522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l275_27577

theorem line_through_point_inequality (a b : ℝ) (α : ℝ) 
  (h : Real.cos α / a + Real.sin α / b = 1) : 
  1 / a^2 + 1 / b^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l275_27577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l275_27597

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 1)) / x

theorem domain_of_f : 
  {x : ℝ | x ∈ Set.Icc (-1) 0 ∪ Set.Ioi 0} = {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l275_27597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_property_l275_27559

/-- A function that checks if a four-digit number satisfies the property that 
    each of the last two digits is equal to the sum of the two preceding digits. -/
def satisfiesProperty (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits[2]! = digits[0]! + digits[1]! ∧
  digits[3]! = digits[1]! + digits[2]!

/-- Theorem stating that 9099 is the largest four-digit number satisfying the property. -/
theorem largest_number_with_property :
  satisfiesProperty 9099 ∧ ∀ n : ℕ, satisfiesProperty n → n ≤ 9099 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_property_l275_27559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_zero_l275_27551

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y = 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

-- Theorem statement
theorem intersection_distance_is_zero :
  ∀ p q, p ∈ intersection_points → q ∈ intersection_points → dist p q = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_zero_l275_27551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l275_27546

noncomputable def book_widths : List ℝ := [6, 1/2, 1, 2.5, 10]

theorem average_book_width :
  (List.sum book_widths) / (List.length book_widths) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l275_27546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l275_27552

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + 3*x + a) / Real.log a

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ a ≠ 1 ∧ (∀ y : ℝ, ∃ x : ℝ, f a x = y)) →
  (a ∈ Set.Ioo 0 1 ∪ Set.Ioc 1 (9/4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l275_27552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_divisible_by_digits_l275_27504

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≠ 0 → d < 10 → (n.digits 10).contains d → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ,
    100 ≤ n → n < 1000 →
    (n.digits 10).head? = some 8 →
    is_divisible_by_digits n →
    n ≤ 864 :=
by
  sorry

#check largest_three_digit_divisible_by_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_divisible_by_digits_l275_27504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derricks_yard_length_l275_27574

-- Define the lengths of the yards
variable (derricks_yard : ℝ)
variable (alexs_yard : ℝ)
variable (briannes_yard : ℝ)

-- State the conditions
axiom alexs_yard_half_derricks : alexs_yard = derricks_yard / 2
axiom briannes_yard_six_times_alexs : briannes_yard = 6 * alexs_yard
axiom briannes_yard_length : briannes_yard = 30

-- State the theorem to be proved
theorem derricks_yard_length : derricks_yard = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derricks_yard_length_l275_27574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_event_solution_l275_27572

/-- Represents the number of medal sets remaining at the beginning of day i -/
def remaining_medals (n : ℕ) (N : ℕ) : ℕ → ℕ
  | 0 => N  -- Add this case for Nat.zero
  | 1 => N
  | i + 1 => (6 * (remaining_medals n N i - i)) / 7

/-- The sports event problem -/
theorem sports_event_solution :
  ∀ n N : ℕ,
  (n > 0) →
  (N > 0) →
  (∀ i : ℕ, 1 ≤ i ∧ i < n → remaining_medals n N (i + 1) = remaining_medals n N i - i - (remaining_medals n N i - i) / 7) →
  (remaining_medals n N n = n) →
  (n = 6 ∧ N = 36) := by
  sorry

#check sports_event_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_event_solution_l275_27572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_l275_27561

/-- Regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Calculates the number of intersections between two regular polygons -/
def intersections (p1 p2 : RegularPolygon) : ℕ :=
  2 * min p1.sides p2.sides

/-- The set of regular polygons inscribed in the circle -/
def polygons : List RegularPolygon := [
  ⟨4⟩, ⟨5⟩, ⟨7⟩, ⟨9⟩
]

/-- Theorem stating the total number of intersections -/
theorem total_intersections :
  (List.sum (List.map
    (fun pair => intersections pair.1 pair.2)
    (List.filter
      (fun pair => pair.1.sides < pair.2.sides)
      (List.product polygons polygons)))) = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_l275_27561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_parallel_perpendicular_lines_l275_27529

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

def Line.contains_point (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

noncomputable def intersection_point (l1 l2 : Line) : Point :=
  { x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b),
    y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b) }

theorem intersection_and_parallel_perpendicular_lines :
  let l1 : Line := ⟨3, 4, -5⟩
  let l2 : Line := ⟨2, -3, 8⟩
  let l3 : Line := ⟨2, 1, 7⟩
  let M : Point := intersection_point l1 l2
  let l_parallel : Line := ⟨2, 1, 0⟩
  let l_perpendicular : Line := ⟨1, -2, 5⟩
  (M.x = -1 ∧ M.y = 2) ∧
  (l1.contains_point M ∧ l2.contains_point M) ∧
  (l_parallel.parallel l3 ∧ l_parallel.contains_point M) ∧
  (l_perpendicular.perpendicular l3 ∧ l_perpendicular.contains_point M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_parallel_perpendicular_lines_l275_27529
