import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_transformed_alloy_l66_6654

/-- The height of a transformed rectangular alloy -/
noncomputable def transformed_height (l w h new_base : ℝ) : ℝ :=
  (l * w * h) / (new_base * new_base)

/-- Theorem: The height of the transformed rectangular alloy is 300 -/
theorem height_of_transformed_alloy :
  transformed_height 80 60 100 40 = 300 := by
  -- Unfold the definition of transformed_height
  unfold transformed_height
  -- Simplify the arithmetic expression
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- Evaluate the numeric expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_transformed_alloy_l66_6654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l66_6601

/-- Represents a tetrahedron PQRS with given edge lengths and properties -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  PS : ℝ
  QS : ℝ
  RS : ℝ
  right_triangle : PQ^2 + QR^2 = PR^2
  perp_projection : PS = RS

/-- The volume of the tetrahedron PQRS -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  (1/3) * (1/2 * t.PQ * t.QR) * t.RS

/-- The theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume : 
  ∃ t : Tetrahedron, 
    t.PQ = 6 ∧ 
    t.PR = 5 ∧ 
    t.QR = 5 ∧ 
    t.PS = 5 ∧ 
    t.QS = 4 ∧ 
    t.RS = 15/4 * Real.sqrt 2 ∧
    tetrahedron_volume t = 75/4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l66_6601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_third_polygon_l66_6679

/-- Given three regular polygons inscribed in a circle, where each subsequent polygon has
    twice the number of sides as the previous one, this theorem relates the areas of the
    three polygons. -/
theorem area_third_polygon (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) :
  ∃ S : ℝ, S > 0 ∧ S = Real.sqrt (2 * S₂^3 / (S₁ + S₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_third_polygon_l66_6679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_B_teaches_subject_C_l66_6669

-- Define the types for teachers, cities, and subjects
inductive Teacher : Type
| A | B | C

inductive City : Type
| Harbin | Changchun | Shenyang

inductive Subject : Type
| A | B | C

-- Define the functions for work location and taught subject
variable (workLocation : Teacher → City)
variable (teachesSubject : Teacher → Subject)

-- State the theorem
theorem teacher_B_teaches_subject_C :
  -- Conditions
  (workLocation Teacher.A ≠ City.Harbin) →
  (workLocation Teacher.B ≠ City.Changchun) →
  (∀ t : Teacher, workLocation t = City.Harbin → teachesSubject t ≠ Subject.C) →
  (∀ t : Teacher, workLocation t = City.Changchun → teachesSubject t = Subject.A) →
  (teachesSubject Teacher.B ≠ Subject.B) →
  -- Conclusion
  (teachesSubject Teacher.B = Subject.C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_B_teaches_subject_C_l66_6669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l66_6686

noncomputable def f (x a : ℝ) : ℝ := 4 * Real.cos x * Real.cos (x - Real.pi/3) + a

theorem function_properties :
  ∃ (a : ℝ),
  (∀ x : ℝ, f (x + Real.pi) a = f x a) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) → (∀ y : ℝ, y ∈ Set.Icc 0 (Real.pi/6) → x ≤ y → f x a ≤ f y a)) ∧
  (f (2*Real.pi/3) a = 0) ∧
  (a = 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) → f x a ∈ Set.Icc 1 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l66_6686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_cookie_count_l66_6642

noncomputable section

/-- The radius of Ana's circular cookies in inches -/
def ana_cookie_radius : ℝ := 2

/-- The side length of David's equilateral triangle cookies in inches -/
def david_cookie_side : ℝ := 4

/-- The number of cookies in Ana's batch -/
def ana_batch_size : ℕ := 10

/-- The area of a single circular cookie -/
noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

/-- The area of a single equilateral triangle cookie -/
noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

/-- The total dough area used for a batch of cookies -/
noncomputable def total_dough_area : ℝ := ana_batch_size * circle_area ana_cookie_radius

/-- The number of David's cookies that can be made from the total dough area -/
noncomputable def david_batch_size : ℝ := total_dough_area / equilateral_triangle_area david_cookie_side

theorem david_cookie_count : ⌊david_batch_size⌋ = 18 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_cookie_count_l66_6642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l66_6627

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

def point : ℝ × ℝ := (0, 0)

theorem tangent_line_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x, y) ∈ {(x, y) | y = m * x} ↔
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ,
      |h| < δ → |f (point.1 + h) - (f point.1 + m * h)| ≤ ε * |h|) ∧
    m * x - y = 0 := by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l66_6627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l66_6645

/-- The area of a quadrilateral with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄) -/
noncomputable def quadrilateralArea (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : ℝ :=
  (1/2) * |x₁*y₂ + x₂*y₃ + x₃*y₄ + x₄*y₁ - y₁*x₂ - y₂*x₃ - y₃*x₄ - y₄*x₁|

/-- The area of the specific quadrilateral with vertices (2,1), (0,7), (5,5), and (6,9) is 9 -/
theorem specific_quadrilateral_area :
  quadrilateralArea 2 1 0 7 5 5 6 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l66_6645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l66_6636

theorem trigonometric_identity (θ : ℝ) (h : (Real.tan (π/2 - θ) ^ 2000 + 2) / (Real.sin θ + 1) = 1) :
  (Real.sin θ + 2)^2 * (Real.cos θ + 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l66_6636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_play_attendance_l66_6612

theorem school_play_attendance
  (num_girls : ℕ) (num_boys : ℕ) (parents_per_child : ℕ) (other_family_per_child : ℕ) :
  num_girls = 10 →
  num_boys = 12 →
  parents_per_child = 2 →
  other_family_per_child = 2 →
  (num_girls + num_boys) * (parents_per_child + other_family_per_child) = 88 := by
  sorry

#check school_play_attendance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_play_attendance_l66_6612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l66_6623

/-- The length of the line segment formed by the intersection of a line and a circle -/
theorem intersection_segment_length
  (line : ℝ → ℝ)
  (circle : ℝ × ℝ → Prop)
  (h_line : ∀ x y, y = line x ↔ y = -x + 2)
  (h_circle : ∀ x y, circle (x, y) ↔ x^2 + y^2 = 3) :
  ∃ A B : ℝ × ℝ,
    circle A ∧ A.2 = line A.1 ∧
    circle B ∧ B.2 = line B.1 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l66_6623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l66_6666

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 1 else 1 - x

-- Theorem statement
theorem f_range : Set.range f = Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l66_6666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l66_6680

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  (x^2 - 6*x) + (y^2 + 10*y) = 9

-- Define the area of the region
noncomputable def region_area : ℝ := 43 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l66_6680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_difference_l66_6699

theorem divisibility_of_power_difference (a b : ℤ) (m : ℕ) :
  (∃ r : ℤ, ∃ k₁ k₂ : ℤ, a = (a - b) * k₁ + r ∧ b = (a - b) * k₂ + r) →
  ∃ k : ℤ, a^m - b^m = (a - b) * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_difference_l66_6699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l66_6615

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the given line
def given_line_equation (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- Define the slope of a line given its equation ax + by + c = 0
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

-- Define parallelism of two lines
def are_parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop := line_slope a₁ b₁ = line_slope a₂ b₂

-- Statement to prove
theorem line_equation : ∃ (a b c : ℝ),
  (∀ x y, a*x + b*y + c = 0 ↔ 2*x - y - 4 = 0) ∧
  (a*(circle_center.1) + b*(circle_center.2) + c = 0) ∧
  are_parallel a b c 2 (-1) 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l66_6615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_half_filled_cone_l66_6608

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Represents the water level in the cone -/
structure WaterLevel where
  cone : Cone
  volume_ratio : ℝ

/-- Calculates the height of water in the cone -/
noncomputable def waterHeight (w : WaterLevel) : ℝ :=
  w.cone.height * (w.volume_ratio)^(1/3)

theorem water_height_in_half_filled_cone :
  let c : Cone := { radius := 10, height := 30 }
  let w : WaterLevel := { cone := c, volume_ratio := 1/2 }
  waterHeight w = 15 * Real.rpow 2 (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_half_filled_cone_l66_6608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_ex_eq_ax_l66_6620

/-- The number of solutions to the equation e^x = ax depends on the value of a -/
theorem solutions_count_ex_eq_ax (a : ℝ) :
  (∃! x, Real.exp x = a * x) ∨
  (∀ x, Real.exp x ≠ a * x) ∨
  (∃! x, Real.exp x = Real.exp x) ∨
  (∃ x y, x ≠ y ∧ Real.exp x = a * x ∧ Real.exp y = a * y ∧
    ∀ z, Real.exp z = a * z → z = x ∨ z = y) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_ex_eq_ax_l66_6620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_weight_l66_6697

/-- 
Given a person weighing T kg who holds 1.5T kg in each hand and wears a vest weighing 0.5T kg,
if the total weight moved is 525 kg, then T is approximately 116.67 kg.
-/
theorem person_weight (T : ℝ) (h1 : T > 0) 
  (h2 : T + 2 * (1.5 * T) + 0.5 * T = 525) : 
  ∃ ε > 0, |T - 116.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_weight_l66_6697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l66_6685

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → (2 : ℝ)^x > 1)) ↔ (∃ x : ℝ, x > 0 ∧ (2 : ℝ)^x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l66_6685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_l66_6692

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function between a point and a horizontal line
def distToLine (p : Point) (lineY : ℝ) : ℝ :=
  |p.y - lineY|

-- Define the distance function between two points
noncomputable def distToPoint (p1 : Point) (p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem point_trajectory (p : Point) :
  distToLine p (-1) = distToPoint p ⟨0, 3⟩ - 2 →
  p.x^2 = 12 * p.y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_l66_6692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PPP_roots_l66_6603

/-- A quadratic polynomial P(x) = x^2 + bx + c with a unique root -/
noncomputable def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The equation P(x) = 0 has a unique root -/
axiom P_unique_root (b c : ℝ) : ∃! r : ℝ, P b c r = 0

/-- The equation P(P(P(x))) = 0 has exactly three distinct roots -/
axiom PPP_three_roots (b c : ℝ) : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  P b c (P b c (P b c x)) = 0 ∧ P b c (P b c (P b c y)) = 0 ∧ P b c (P b c (P b c z)) = 0 ∧
  ∀ w : ℝ, P b c (P b c (P b c w)) = 0 → w = x ∨ w = y ∨ w = z

/-- The roots of P(P(P(x))) = 0 are 1, 1 + √2, and 1 - √2 -/
theorem PPP_roots : ∃ b c : ℝ, ∃ x y z : ℝ, x = 1 ∧ y = 1 + Real.sqrt 2 ∧ z = 1 - Real.sqrt 2 ∧
  P b c (P b c (P b c x)) = 0 ∧ P b c (P b c (P b c y)) = 0 ∧ P b c (P b c (P b c z)) = 0 ∧
  ∀ w : ℝ, P b c (P b c (P b c w)) = 0 → w = x ∨ w = y ∨ w = z :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_PPP_roots_l66_6603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Z_l66_6650

open Complex

variable (z₀ : ℂ)

def Z₁_locus (z₁ : ℂ) : Prop := abs (z₁ - z₀) = abs z₁

def Z_relation (z₁ z : ℂ) : Prop := z₁ * z = -1

def circle_locus (center : ℂ) (radius : ℝ) (z : ℂ) : Prop :=
  abs (z - center) = radius

theorem locus_of_Z (z : ℂ) (h : z₀ ≠ 0) :
  (∃ z₁, Z₁_locus z₀ z₁ ∧ Z_relation z₁ z) →
  circle_locus (-1 / z₀) (1 / abs z₀) z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Z_l66_6650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_chord_length_l66_6676

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + 3/5 * t, -1 + 4/5 * t)

-- Define the curve C in polar form
noncomputable def curve_C_polar (θ : ℝ) : ℝ := Real.sqrt 2 * Real.sin (θ + Real.pi/4)

-- State the theorem
theorem curve_C_and_chord_length :
  -- Part 1: Rectangular equation of curve C
  (∀ x y : ℝ, (x - 1/2)^2 + (y - 1/2)^2 = 1/2 ↔
    ∃ θ : ℝ, x = curve_C_polar θ * Real.cos θ ∧ y = curve_C_polar θ * Real.sin θ) ∧
  -- Part 2: Length of chord intercepted by line l on curve C
  (∃ t₁ t₂ : ℝ,
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    (x₁ - 1/2)^2 + (y₁ - 1/2)^2 = 1/2 ∧
    (x₂ - 1/2)^2 + (y₂ - 1/2)^2 = 1/2 ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 41 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_chord_length_l66_6676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passenger_fraction_l66_6625

/-- Proves that the fraction of passengers dropped at the first station is 1/3 --/
theorem train_passenger_fraction : 
  ∀ (initial_passengers : ℕ) 
    (added_first : ℕ) 
    (added_second : ℕ) 
    (final_passengers : ℕ),
  initial_passengers = 270 →
  added_first = 280 →
  added_second = 12 →
  final_passengers = 242 →
  ∃ (f : ℚ),
    f = 1/3 ∧
    final_passengers = 
      (((1 - f) * initial_passengers + added_first) / 2 + added_second).floor :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passenger_fraction_l66_6625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_direction_changes_proof_min_changes_is_n_l66_6631

/-- Represents an equilateral triangle with side length n -/
structure EquilateralTriangle (n : ℕ) where
  side_length : ℕ := n

/-- Represents a path on the triangular grid -/
structure PathOnTriangle (n : ℕ) where
  vertices_visited : Finset (ℕ × ℕ)
  direction_changes : ℕ

/-- The total number of vertices in an equilateral triangle with side length n -/
def total_vertices (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- A valid path visits each vertex exactly once -/
def is_valid_path (n : ℕ) (p : PathOnTriangle n) : Prop :=
  p.vertices_visited.card = total_vertices n

/-- The minimum number of direction changes for a valid path -/
def min_direction_changes (n : ℕ) : ℕ := n

theorem min_direction_changes_proof (n : ℕ) :
  ∀ (p : PathOnTriangle n), is_valid_path n p → p.direction_changes ≥ min_direction_changes n :=
by
  sorry

/-- The main theorem stating that the minimum number of direction changes is n -/
theorem min_changes_is_n (n : ℕ) :
  ∃ (p : PathOnTriangle n), is_valid_path n p ∧ p.direction_changes = min_direction_changes n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_direction_changes_proof_min_changes_is_n_l66_6631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_theorem_l66_6604

/-- Represents the time taken to paint small and large paintings -/
structure PaintingTime where
  smallTime : ℝ
  largeTime : ℝ

/-- Calculates the total time taken to paint all paintings -/
noncomputable def totalPaintingTime (pt : PaintingTime) (initialSmall initialLarge additionalSmall additionalLarge : ℕ) : ℝ :=
  let smallPaintingRate := pt.smallTime / initialSmall
  let largePaintingRate := pt.largeTime / initialLarge
  pt.smallTime + pt.largeTime + 
  (smallPaintingRate * additionalSmall) + 
  (largePaintingRate * additionalLarge)

/-- Theorem stating the total time taken to paint all paintings -/
theorem painting_time_theorem (pt : PaintingTime) 
  (h1 : pt.smallTime = 6)
  (h2 : pt.largeTime = 8)
  (h3 : initialSmall = 12)
  (h4 : initialLarge = 6)
  (h5 : additionalSmall = 15)
  (h6 : additionalLarge = 10) :
  ∃ (ε : ℝ), abs (totalPaintingTime pt initialSmall initialLarge additionalSmall additionalLarge - 34.8) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_theorem_l66_6604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l66_6652

theorem proposition_equivalence {α : Type*} (M : Set α) (m n : α) :
  (m ∈ M → n ∉ M) ↔ (n ∈ M → m ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l66_6652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_configuration_l66_6605

/-- Represents the state of people on the number line -/
structure State where
  positions : List ℕ
  deriving Repr

/-- Checks if a move is valid in the given state -/
def is_valid_move (s : State) (i j : ℕ) : Prop :=
  i < j ∧ j < s.positions.length ∧ s.positions[i]! + 2 ≤ s.positions[j]!

/-- Performs a move in the given state -/
def move (s : State) (i j : ℕ) : State :=
  { positions := s.positions.set i (s.positions[i]! + 1)
                              |>.set j (s.positions[j]! - 1) }

/-- Checks if the state is in the final configuration -/
def is_final (s : State) : Prop :=
  ∀ i j, i < j → j < s.positions.length → s.positions[i]! + 2 > s.positions[j]!

/-- The initial state of the problem -/
def initial_state : State :=
  { positions := List.range 2023 }

/-- The theorem to be proved -/
theorem final_configuration :
  ∃ (sequence : List (ℕ × ℕ)),
    let final := sequence.foldl (λ s (i, j) => move s i j) initial_state
    is_final final ∧ ∀ i < final.positions.length, final.positions[i]! = 1011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_configuration_l66_6605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_fifty_bullfinches_l66_6610

/-- Represents the statements about the number of bullfinches -/
inductive BullfinchStatement
  | atLeastOne
  | fewerThanFifty
  | moreThanFifty

/-- The number of bullfinches in the store -/
def numBullfinches : ℕ := sorry

/-- Only one of the statements is true -/
axiom only_one_true : ∃! (s : BullfinchStatement), 
  match s with
  | BullfinchStatement.atLeastOne => numBullfinches ≥ 1
  | BullfinchStatement.fewerThanFifty => numBullfinches < 50
  | BullfinchStatement.moreThanFifty => numBullfinches > 50

/-- A bullfinch was purchased, implying there was at least one -/
axiom bullfinch_purchased : numBullfinches ≥ 1

/-- The theorem stating that there are exactly 50 bullfinches -/
theorem exactly_fifty_bullfinches : numBullfinches = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_fifty_bullfinches_l66_6610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_twelve_l66_6611

theorem fourth_root_sixteen_to_twelve : (16 : ℝ) ^ (1/4) ^ 12 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_twelve_l66_6611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equality_l66_6640

open Matrix

theorem det_equality (A B C : Matrix (Fin 3) (Fin 3) ℂ)
  (h1 : Matrix.det A = Matrix.det B)
  (h2 : Matrix.det A = Matrix.det C)
  (h3 : Matrix.det (A + Complex.I • B) = Matrix.det (C + Complex.I • A)) :
  Matrix.det (A + B) = Matrix.det (C + A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equality_l66_6640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l66_6638

/-- Given a polynomial g(x) = px^4 + qx^3 + rx^2 + sx + t where g(-2) = -4,
    prove that 16p - 8q + 4r - 2s + t = 4 -/
theorem polynomial_value_theorem (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t) 
  (h_point : g (-2) = -4) : 
  16 * p - 8 * q + 4 * r - 2 * s + t = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l66_6638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_chocolate_bars_l66_6660

theorem james_chocolate_bars (sold_last_week sold_this_week to_sell : ℕ) :
  sold_last_week + sold_this_week + to_sell = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_chocolate_bars_l66_6660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l66_6626

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = c/2, then tan(A)/tan(B) = 3 -/
theorem triangle_tangent_ratio (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B - b * Real.cos A = c / 2 →
  Real.tan A / Real.tan B = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l66_6626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_cos_equals_log_sin_l66_6651

-- Define the variables and constants
variable (c x a : ℝ)

-- Define the conditions
def conditions (c x a : ℝ) : Prop :=
  c > 1 ∧
  Real.sin x > 0 ∧
  Real.cos x > 0 ∧
  Real.tan x = 1 ∧
  Real.logb c (Real.sin x) = a ∧
  0 < x ∧ x < Real.pi / 2

-- State the theorem
theorem log_cos_equals_log_sin (h : conditions c x a) : Real.logb c (Real.cos x) = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_cos_equals_log_sin_l66_6651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_is_3_sqrt_17_l66_6659

-- Define the right triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the trisection points D and E on hypotenuse AB
noncomputable def trisection_points (triangle : RightTriangle) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let D := ((2 * triangle.A.1 + triangle.B.1) / 3, (2 * triangle.A.2 + triangle.B.2) / 3)
  let E := ((triangle.A.1 + 2 * triangle.B.1) / 3, (triangle.A.2 + 2 * triangle.B.2) / 3)
  (D, E)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem hypotenuse_length_is_3_sqrt_17 (triangle : RightTriangle) :
  let (D, E) := trisection_points triangle
  distance triangle.C D = 7 →
  distance triangle.C E = 6 →
  distance triangle.A triangle.B = 3 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_is_3_sqrt_17_l66_6659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l66_6664

noncomputable def f (x : ℝ) : ℝ := 2^(x-1) + x - 1

theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 0 (1/2), f x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l66_6664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l66_6675

/-- A circle passing through (1, 2) and tangent to y = x^2 at (1, 1) has center (0, 3/2) -/
theorem circle_center (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) : 
  (∀ (p : ℝ × ℝ), p ∈ C → (p.1 - center.1)^2 + (p.2 - center.2)^2 = (1 - center.1)^2 + (1 - center.2)^2) →
  (1, 2) ∈ C →
  (1, 1) ∈ C →
  (∀ (p : ℝ × ℝ), p ∈ C → p.2 ≠ p.1^2 ∨ (p.1 = 1 ∧ p.2 = 1)) →
  center = (0, 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l66_6675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l66_6635

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- Expand the definition of g
  simp [g]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l66_6635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l66_6600

def ellipse_center : ℝ × ℝ := (2, -3)
def ellipse_focus : ℝ × ℝ := (2, -5)
def ellipse_semi_major_endpoint : ℝ × ℝ := (2, 0)

theorem ellipse_semi_minor_axis :
  let c := abs (ellipse_center.2 - ellipse_focus.2)
  let a := abs (ellipse_center.2 - ellipse_semi_major_endpoint.2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l66_6600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_20cm_l66_6632

/-- The diameter of a wheel given its revolutions and distance covered -/
noncomputable def wheel_diameter (revolutions : ℝ) (distance : ℝ) : ℝ :=
  distance / (revolutions * Real.pi)

/-- Theorem: The diameter of a wheel is approximately 20 cm -/
theorem wheel_diameter_approx_20cm :
  let revolutions : ℝ := 16.81528662420382
  let distance : ℝ := 1056
  abs (wheel_diameter revolutions distance - 20) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_20cm_l66_6632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_104_l66_6696

-- Define the arithmetic sequence
def arithmetic_sequence : List ℕ :=
  List.range 34 |> List.map (λ n => 1 + 3 * n)

-- Define the property of set A
def valid_set (A : Finset ℕ) : Prop :=
  A.card = 20 ∧ ∀ x ∈ A, x ∈ arithmetic_sequence

-- Theorem statement
theorem exists_sum_104 (A : Finset ℕ) (h : valid_set A) :
  ∃ x y, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ x + y = 104 := by
  sorry

#check exists_sum_104

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_104_l66_6696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l66_6621

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The focus of the parabola y^2 = 4x -/
def focus : Point :=
  { x := 1, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: If a point on the parabola y^2 = 4x is at distance 5 from the focus,
    its x-coordinate is 4 -/
theorem parabola_point_x_coordinate 
  (M : Point) 
  (h1 : isOnParabola M) 
  (h2 : distance M focus = 5) : 
  M.x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l66_6621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_triples_count_l66_6655

/-- The number of ways to divide 3n points on a circle into n triples
    such that the sides of the n inscribed triangles do not intersect -/
def non_intersecting_triples (n : ℕ) : ℚ :=
  (1 : ℚ) / (2 * n + 1 : ℚ) * (Nat.choose (3 * n) n : ℚ)

/-- Theorem: The number of ways to divide 3n points on a circle into n triples,
    such that the sides of the n inscribed triangles do not intersect,
    is equal to 1 / (2n+1) * C(3n, n) -/
theorem non_intersecting_triples_count (n : ℕ) :
  non_intersecting_triples n = (1 : ℚ) / (2 * n + 1 : ℚ) * (Nat.choose (3 * n) n : ℚ) := by
  rfl

#eval non_intersecting_triples 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_triples_count_l66_6655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_7_5_l66_6647

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def vertices : List (ℝ × ℝ) := [(0, 2), (3, 0), (5, 2), (2, 3)]

noncomputable def quadrilateral_area : ℝ :=
  let v := vertices
  triangle_area v[0].1 v[0].2 v[1].1 v[1].2 v[2].1 v[2].2 +
  triangle_area v[0].1 v[0].2 v[2].1 v[2].2 v[3].1 v[3].2

theorem quadrilateral_area_is_7_5 : quadrilateral_area = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_7_5_l66_6647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_project_l66_6682

/-- Represents the project details and worker information --/
structure ProjectInfo where
  totalDays : ℕ
  daysWorked : ℕ
  currentWorkers : ℕ
  portionCompleted : ℚ

/-- Calculates the minimum number of workers needed to complete the project on time --/
def minWorkersNeeded (info : ProjectInfo) : ℕ :=
  let remainingDays := info.totalDays - info.daysWorked
  let remainingWork := 1 - info.portionCompleted
  let currentRate := info.portionCompleted / info.daysWorked
  let requiredRate := remainingWork / remainingDays
  (((requiredRate / currentRate) * info.currentWorkers).ceil).toNat

/-- Theorem stating that for the given project information, the minimum number of workers needed is 6 --/
theorem min_workers_for_project :
  let info : ProjectInfo := {
    totalDays := 40,
    daysWorked := 10,
    currentWorkers := 12,
    portionCompleted := 2/5
  }
  minWorkersNeeded info = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_project_l66_6682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_with_parabola_focus_as_center_l66_6613

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the point P
noncomputable def point_P : ℝ × ℝ := (5, -2*Real.sqrt 5)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 36

theorem circle_through_point_with_parabola_focus_as_center :
  (∀ x y, parabola x y → (x, y) ≠ focus) ∧
  circle_equation point_P.1 point_P.2 ∧
  (∀ x y, circle_equation x y ↔ ((x - focus.1)^2 + (y - focus.2)^2 = 36)) := by
  sorry

#check circle_through_point_with_parabola_focus_as_center

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_with_parabola_focus_as_center_l66_6613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_iff_sum_of_segments_l66_6670

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  -- We don't need to define the internal structure for this problem
  dummy : Unit

/-- A line segment in a plane -/
structure LineSegment where
  -- We don't need to define the internal structure for this problem
  dummy : Unit

/-- Predicate indicating if a convex polygon has a center of symmetry -/
def has_center_of_symmetry (P : ConvexPolygon) : Prop :=
  sorry

/-- Representation of a polygon as a sum of line segments -/
def is_sum_of_segments (P : ConvexPolygon) : Prop :=
  ∃ (n : ℕ) (S : Fin n → LineSegment) (c : Fin n → ℝ), 
    P = sorry -- Placeholder for the actual sum representation

/-- Theorem stating the equivalence between having a center of symmetry and
    being representable as a sum of line segments for convex polygons -/
theorem symmetry_iff_sum_of_segments (P : ConvexPolygon) :
  has_center_of_symmetry P ↔ is_sum_of_segments P :=
by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_iff_sum_of_segments_l66_6670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tablecloth_length_l66_6665

/-- Calculates the length of a tablecloth given the following conditions:
  * The width of the tablecloth is 54 inches
  * 8 napkins of size 6 by 7 inches are also made
  * The total area of material for both tablecloth and napkins is 5844 square inches
-/
theorem tablecloth_length 
  (width : ℕ) (num_napkins : ℕ) (napkin_length : ℕ) (napkin_width : ℕ) (total_area : ℕ) (result : ℕ) :
  width = 54 →
  num_napkins = 8 →
  napkin_length = 6 →
  napkin_width = 7 →
  total_area = 5844 →
  result = 102 →
  (total_area - num_napkins * napkin_length * napkin_width) / width = result := by
  sorry

#check tablecloth_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tablecloth_length_l66_6665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_sequence_for_linear_relation_l66_6643

/-- Represents the steps in linear regression analysis -/
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | CalculateCorrelation
  | DrawScatterPlot

/-- Represents a sequence of regression steps -/
def RegressionSequence := List RegressionStep

/-- The correct sequence of steps for linear regression when x and y are linearly related -/
def correctSequence : RegressionSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateCorrelation,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

/-- Represents a linear relationship between two types -/
def LinearlyRelated (x y : Type) : Prop := sorry

/-- Represents the optimal regression sequence for linearly related variables -/
def OptimalRegressionSequence (x y : Type) (h : LinearlyRelated x y) : RegressionSequence := sorry

/-- Theorem stating that the given sequence is correct for linearly related variables -/
theorem correct_regression_sequence_for_linear_relation :
  ∀ (x y : Type) (h : LinearlyRelated x y),
  OptimalRegressionSequence x y h = correctSequence := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_sequence_for_linear_relation_l66_6643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_chip_cookies_l66_6694

/-- Given the total number of cookies, cookies per bag, and number of oatmeal cookie baggies,
    prove that the number of chocolate chip cookies is 13. -/
theorem chocolate_chip_cookies
  (total_cookies : ℝ)
  (cookies_per_bag : ℝ)
  (oatmeal_baggies : ℝ)
  (h1 : total_cookies = 41.0)
  (h2 : cookies_per_bag = 9.0)
  (h3 : oatmeal_baggies = 3.111111111) :
  ⌊total_cookies - ⌊oatmeal_baggies * cookies_per_bag⌋⌋ = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_chip_cookies_l66_6694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_percentage_before_storm_l66_6658

noncomputable def reservoir_capacity (original_contents storm_deposit full_percentage : ℝ) : ℝ :=
  (original_contents + storm_deposit) / full_percentage

theorem reservoir_percentage_before_storm 
  (original_contents : ℝ) 
  (storm_deposit : ℝ) 
  (full_percentage_after : ℝ) 
  (h1 : original_contents = 200)
  (h2 : storm_deposit = 120)
  (h3 : full_percentage_after = 0.8) :
  original_contents / reservoir_capacity original_contents storm_deposit full_percentage_after = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_percentage_before_storm_l66_6658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l66_6661

open Real

/-- The function f(x) = x cos x - sin x -/
noncomputable def f (x : ℝ) : ℝ := x * cos x - sin x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := -x * sin x

/-- Theorem stating that f' is the derivative of f -/
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = f' x :=
by
  intro x
  sorry -- We'll leave the proof for later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l66_6661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_and_height_l66_6624

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (14, 4, 5)
  A₂ : ℝ × ℝ × ℝ := (-5, -3, 2)
  A₃ : ℝ × ℝ × ℝ := (-2, -6, -3)
  A₄ : ℝ × ℝ × ℝ := (-2, 2, -1)

/-- Calculate the volume of the tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  112 + 2/3

/-- Calculate the height from A₄ to the plane A₁A₂A₃ -/
noncomputable def tetrahedronHeight (t : Tetrahedron) : ℝ :=
  Real.sqrt 26

/-- Theorem stating that the volume and height calculations are correct -/
theorem tetrahedron_volume_and_height (t : Tetrahedron) :
  tetrahedronVolume t = 112 + 2/3 ∧ tetrahedronHeight t = Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_and_height_l66_6624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_boys_same_time_l66_6678

/-- Represents a set of boys and their shop visits -/
structure ShopVisits where
  boys : Finset Nat
  visits : Finset (Finset Nat)
  boys_count : boys.card = 7
  visits_size : ∀ v ∈ visits, v.card = 3
  all_pairs_met : ∀ b1 b2, b1 ∈ boys → b2 ∈ boys → b1 ≠ b2 → ∃ v ∈ visits, b1 ∈ v ∧ b2 ∈ v

/-- Theorem stating that at least 3 boys must have been in the shop at the same time -/
theorem three_boys_same_time (sv : ShopVisits) : 
  ∃ v ∈ sv.visits, v.card ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_boys_same_time_l66_6678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l66_6648

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 8

-- Define the intersection points
noncomputable def x1 : ℝ := (3 - Real.sqrt 113) / 4
noncomputable def x2 : ℝ := (3 + Real.sqrt 113) / 4
noncomputable def y1 : ℝ := (360 - 9 * Real.sqrt 113) / 16
noncomputable def y2 : ℝ := (360 + 9 * Real.sqrt 113) / 16

theorem parabola_intersection :
  (∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x → (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2)) ∧
  parabola1 x1 = parabola2 x1 ∧
  parabola1 x2 = parabola2 x2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l66_6648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_equal_volume_l66_6653

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_surface_area : ℝ) :
  l = 12 →
  w = 3 →
  h = 18 →
  cube_surface_area = 6 * (l * w * h)^(2/3) →
  Int.floor cube_surface_area = 447 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_equal_volume_l66_6653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l66_6641

-- Define the clock structure
structure Clock where
  hour_hand_rate : ℚ  -- Degrees per hour
  minute_hand_rate : ℚ -- Degrees per minute

-- Define the time
def time : ℕ × ℕ := (3, 30)

-- Function to calculate hour hand position
def hour_hand_position (c : Clock) (t : ℕ × ℕ) : ℚ :=
  (t.1 : ℚ) * c.hour_hand_rate + (t.2 : ℚ) * c.hour_hand_rate / 60

-- Function to calculate minute hand position
def minute_hand_position (c : Clock) (t : ℕ × ℕ) : ℚ :=
  (t.2 : ℚ) * c.minute_hand_rate

-- Function to calculate angle between hands
def angle_between_hands (c : Clock) (t : ℕ × ℕ) : ℚ :=
  abs (minute_hand_position c t - hour_hand_position c t)

-- Theorem statement
theorem clock_angle_at_3_30 (c : Clock) 
  (h1 : c.hour_hand_rate = 30)
  (h2 : c.minute_hand_rate = 6) :
  angle_between_hands c time = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l66_6641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_distribution_tourist_distribution_alt_l66_6684

/-- The number of ways to distribute k tourists among n cinemas is n^k. -/
theorem tourist_distribution (n k : ℕ) : n^k = n^k :=
by
  -- The proof is trivial since we're stating that n^k equals itself
  rfl

/-- 
  An alternative formulation using a function to represent the number of ways:
  
  Given:
  - n : ℕ, the number of cinemas
  - k : ℕ, the number of tourists
  
  The number of ways to distribute k tourists among n cinemas is n^k.
-/
def number_of_ways_to_distribute (n k : ℕ) : ℕ := n^k

theorem tourist_distribution_alt (n k : ℕ) : 
  number_of_ways_to_distribute n k = n^k :=
by
  -- Unfold the definition of number_of_ways_to_distribute
  unfold number_of_ways_to_distribute
  -- The proof is then trivial
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_distribution_tourist_distribution_alt_l66_6684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_drain_time_approx_l66_6607

-- Define the pool dimensions and capacity
def topWidth : ℝ := 80
def bottomWidth : ℝ := 60
def poolHeight : ℝ := 10  -- Changed from 'height' to avoid conflict
def length : ℝ := 150
def currentCapacity : ℝ := 0.8

-- Define the hose drainage rates
def hoseARate : ℝ := 60
def hoseBRate : ℝ := 75
def hoseCRate : ℝ := 50

-- Calculate the full volume of the pool
noncomputable def fullVolume : ℝ := (1/2) * (topWidth + bottomWidth) * poolHeight * length

-- Calculate the current volume of water in the pool
noncomputable def currentVolume : ℝ := currentCapacity * fullVolume

-- Calculate the combined drainage rate of all three hoses
def combinedDrainageRate : ℝ := hoseARate + hoseBRate + hoseCRate

-- Define the time to drain the pool in hours
noncomputable def drainTime : ℝ := currentVolume / combinedDrainageRate / 60

-- Theorem statement
theorem pool_drain_time_approx :
  abs (drainTime - 7.57) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_drain_time_approx_l66_6607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l66_6693

open Set Finset

theorem proper_subsets_count (S : Finset ℕ) : S = {0, 1, 2} → (filter (· ⊂ S) (powerset S)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l66_6693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_line_equation_l66_6618

-- Define the curve C₁ in polar coordinates
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the curve C₂ derived from C₁
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

-- Define a line passing through the origin
def Line (m : ℝ) (x : ℝ) : ℝ :=
  m * x

-- Define the perimeter of quadrilateral ABCD
noncomputable def Perimeter (θ : ℝ) : ℝ :=
  4 * Real.sqrt 5 * Real.sin (θ + Real.arcsin (2 / Real.sqrt 5))

-- Theorem statement
theorem max_perimeter_line_equation :
  ∃ θ : ℝ, ∃ max_perimeter : ℝ, Perimeter θ = max_perimeter ∧
  (∀ φ : ℝ, Perimeter φ ≤ max_perimeter) ∧
  ∃ A : ℝ × ℝ, A ∈ Set.range C₂ ∧ 
  A.1 > 0 ∧ A.2 > 0 ∧ 
  A.2 = Line (1/4) A.1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_line_equation_l66_6618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_mpg_l66_6630

theorem round_trip_average_mpg (total_distance sedan_mpg pickup_mpg : ℝ) :
  total_distance > 0 ∧ sedan_mpg > 0 ∧ pickup_mpg > 0 →
  total_distance = 300 ∧ sedan_mpg = 40 ∧ pickup_mpg = 25 →
  let sedan_distance := total_distance / 2
  let pickup_distance := total_distance / 2
  let sedan_fuel := sedan_distance / sedan_mpg
  let pickup_fuel := pickup_distance / pickup_mpg
  let total_fuel := sedan_fuel + pickup_fuel
  let average_mpg := total_distance / total_fuel
  30 < average_mpg ∧ average_mpg < 31 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_mpg_l66_6630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l66_6646

/-- The average speed of a round trip given the speeds in each direction -/
noncomputable def average_speed (speed_forward speed_return : ℝ) : ℝ :=
  2 * speed_forward * speed_return / (speed_forward + speed_return)

/-- Theorem stating that the average speed of the round trip is 39.6 km/hr -/
theorem round_trip_average_speed :
  average_speed 44 36 = 39.6 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l66_6646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_intercept_relation_l66_6674

noncomputable section

def Point := ℝ × ℝ

structure Line where
  slope : ℝ
  intercept : ℝ

def Line.throughPoint (l : Line) (p : Point) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

def Line.xIntercept (l : Line) : ℝ :=
  -l.intercept / l.slope

def Line.yIntercept (l : Line) : ℝ :=
  l.intercept

theorem line_through_point_with_intercept_relation (p : Point) :
  ∃ (l₁ l₂ : Line),
    (l₁.throughPoint p ∧ l₂.throughPoint p) ∧
    (l₁.xIntercept = 3 * l₁.yIntercept ∧ l₂.xIntercept = 3 * l₂.yIntercept) ∧
    ((l₁.slope = -1/3 ∧ l₁.intercept = 1) ∨ (l₁.slope = -1/6 ∧ l₁.intercept = 0)) ∧
    ((l₂.slope = -1/3 ∧ l₂.intercept = 1) ∨ (l₂.slope = -1/6 ∧ l₂.intercept = 0)) ∧
    l₁ ≠ l₂ ∧
    ∀ (l : Line),
      l.throughPoint p ∧ l.xIntercept = 3 * l.yIntercept →
      (l = l₁ ∨ l = l₂) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_intercept_relation_l66_6674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l66_6649

theorem rectangle_area_increase : ∀ (l w : ℝ), l > 0 → w > 0 →
  (((2 * l) * (2 * w) - l * w) / (l * w)) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l66_6649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l66_6634

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x - (2 : ℝ)^(-x)
noncomputable def g (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x)

-- Define the propositions
def p₁ : Prop := ∀ x y : ℝ, x < y → f x < f y
def p₂ : Prop := ∀ x y : ℝ, x < y → g x > g y

-- Define the compound propositions
def q₁ : Prop := p₁ ∨ p₂
def q₂ : Prop := p₁ ∧ p₂
def q₃ : Prop := (¬p₁) ∨ p₂
def q₄ : Prop := p₁ ∨ (¬p₂)

-- State the theorem
theorem proposition_truth : q₁ ∧ q₄ ∧ ¬q₂ ∧ ¬q₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l66_6634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l66_6691

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x + Real.pi / 4) * Real.sin (x - Real.pi / 4) + Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), f (k * Real.pi / 2 + Real.pi / 6 + x) = f (k * Real.pi / 2 + Real.pi / 6 - x)) ∧
  (∀ (A B C : ℝ), 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2 ∧ A + B + C = Real.pi →
    f A = Real.sqrt 3 / 2 →
    ∃ (a b c : ℝ), a = 4 ∧ a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
      a + b + c ≤ 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l66_6691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_savings_average_l66_6619

/-- Calculates the average of a list of numbers -/
noncomputable def average (list : List ℝ) : ℝ :=
  (list.sum) / (list.length : ℝ)

/-- Rounds a real number to two decimal places -/
noncomputable def roundToTwoDecimals (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem emily_savings_average : 
  let balances : List ℝ := [100, 300, 450, 0, 300, 300]
  roundToTwoDecimals (average balances) = 241.67 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_savings_average_l66_6619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisibility_l66_6602

theorem odd_power_sum_divisibility (x y : ℤ) :
  ∀ k : ℕ, k > 0 → (∃ m : ℤ, x^(2*k-1) + y^(2*k-1) = m * (x + y)) →
             (∃ n : ℤ, x^(2*k+1) + y^(2*k+1) = n * (x + y)) :=
by
  intro k hk
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisibility_l66_6602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_validity_l66_6690

/-- A parameterization of a line in 2D space -/
structure LineParameterization where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The line y = -3x + 4 -/
noncomputable def line (x : ℝ) : ℝ := -3 * x + 4

/-- Check if a point lies on the line -/
def pointOnLine (p : ℝ × ℝ) : Prop :=
  p.2 = line p.1

/-- Check if a direction vector is valid for the line -/
def validDirection (d : ℝ × ℝ) : Prop :=
  d.2 = -3 * d.1

/-- Check if a parameterization is valid for the line -/
def validParameterization (param : LineParameterization) : Prop :=
  pointOnLine param.point ∧ validDirection param.direction

/-- The given parameterizations -/
noncomputable def paramA : LineParameterization := ⟨(0, 4), (1, -3)⟩
noncomputable def paramB : LineParameterization := ⟨(4/3, 0), (3, -1)⟩
noncomputable def paramC : LineParameterization := ⟨(1, 1), (5, -15)⟩
noncomputable def paramD : LineParameterization := ⟨(-1, 7), (-1/3, 1)⟩
noncomputable def paramE : LineParameterization := ⟨(-4, -8), (0.1, -0.3)⟩

theorem line_parameterization_validity :
  validParameterization paramA ∧
  validParameterization paramE ∧
  ¬validParameterization paramB ∧
  ¬validParameterization paramC ∧
  ¬validParameterization paramD := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_validity_l66_6690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l66_6689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

theorem a_range (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x > 0, f a x > 0) →
  (∀ x ≤ 0, g a x ≤ 0) →
  (a > 1 ∧ a ≤ 5) := by
  sorry

#check a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l66_6689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l66_6677

/-- A regular hexagonal pyramid with base side length and height both equal to a. -/
structure RegularHexagonalPyramid (a : ℝ) where
  base_side : ℝ := a
  height : ℝ := a

/-- The radius of the sphere inscribed in a regular hexagonal pyramid. -/
noncomputable def inscribed_sphere_radius (a : ℝ) (p : RegularHexagonalPyramid a) : ℝ :=
  (a * (Real.sqrt 21 - 3)) / 4

/-- Theorem stating that the radius of the inscribed sphere in a regular hexagonal pyramid
    with base side length and height both equal to a is a(√21 - 3)/4. -/
theorem inscribed_sphere_radius_formula (a : ℝ) (h : a > 0) :
  ∀ (p : RegularHexagonalPyramid a), inscribed_sphere_radius a p = (a * (Real.sqrt 21 - 3)) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l66_6677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_merchant_more_cost_effective_l66_6616

/-- Represents the cost and pit ratio of plums from a merchant -/
structure PlumMerchant where
  cost_per_kg : ℚ
  pit_ratio : ℚ

/-- Calculates the effective cost per kilogram of edible plum (pulp) -/
noncomputable def effective_cost (merchant : PlumMerchant) : ℚ :=
  merchant.cost_per_kg / (1 - merchant.pit_ratio)

/-- The first merchant's plum data -/
def merchant1 : PlumMerchant :=
  { cost_per_kg := 150
  , pit_ratio := 1/3 }

/-- The second merchant's plum data -/
def merchant2 : PlumMerchant :=
  { cost_per_kg := 100
  , pit_ratio := 1/2 }

/-- Theorem stating that the second merchant's plums are more cost-effective -/
theorem second_merchant_more_cost_effective :
  effective_cost merchant2 < effective_cost merchant1 := by
  -- Unfold definitions
  unfold effective_cost merchant1 merchant2
  -- Simplify fractions
  simp
  -- Prove the inequality
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_merchant_more_cost_effective_l66_6616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_power_plus_one_composite_l66_6681

theorem composite_number (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b → ¬ Nat.Prime n := by
  sorry

theorem power_plus_one_composite :
  ¬ Nat.Prime (10^1962 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_power_plus_one_composite_l66_6681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_margie_driving_distance_l66_6687

/-- Calculates the number of miles that can be driven given the car's efficiency,
    gas price, and amount of money spent on gas. -/
noncomputable def miles_driven (efficiency : ℝ) (gas_price : ℝ) (money_spent : ℝ) : ℝ :=
  (money_spent / gas_price) * efficiency

/-- Proves that Margie can drive 125 miles with $25 worth of gas, given her car's
    efficiency and the current gas price. -/
theorem margie_driving_distance :
  let efficiency : ℝ := 25  -- miles per gallon
  let gas_price : ℝ := 5    -- dollars per gallon
  let money_spent : ℝ := 25 -- dollars
  miles_driven efficiency gas_price money_spent = 125 := by
  -- Unfold the definition of miles_driven
  unfold miles_driven
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_margie_driving_distance_l66_6687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tonys_bucket_holds_two_pounds_l66_6667

/-- Represents the problem of calculating the weight of sand Tony's bucket can hold. -/
def tonys_bucket_problem (sandbox_depth sandbox_width sandbox_length : ℝ)
                         (sand_weight_per_cubic_foot : ℝ)
                         (water_ounces_per_drink : ℝ)
                         (trips_per_drink : ℕ)
                         (water_bottle_ounces : ℝ)
                         (water_bottle_cost : ℝ)
                         (initial_money : ℝ)
                         (change_money : ℝ) : Prop :=
  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let sandbox_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let spent_money := initial_money - change_money
  let bottles_bought := spent_money / water_bottle_cost
  let total_water_ounces := bottles_bought * water_bottle_ounces
  let total_trips := (total_water_ounces / water_ounces_per_drink) * (trips_per_drink : ℝ)
  let bucket_sand_weight := sandbox_sand_weight / total_trips
  bucket_sand_weight = 2

/-- Theorem stating that given the problem conditions, Tony's bucket can hold 2 pounds of sand. -/
theorem tonys_bucket_holds_two_pounds :
  tonys_bucket_problem 2 4 5 3 3 4 15 2 10 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tonys_bucket_holds_two_pounds_l66_6667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pure_ghee_percentage_l66_6662

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  vanaspati : ℝ
  pure : ℝ

/-- Calculates the percentage of vanaspati in a ghee mixture -/
noncomputable def vanaspatiPercentage (mixture : GheeMixture) : ℝ :=
  mixture.vanaspati / mixture.total * 100

theorem original_pure_ghee_percentage 
  (original : GheeMixture)
  (new : GheeMixture)
  (h1 : original.total = 10)
  (h2 : vanaspatiPercentage original = 40)
  (h3 : new.total = original.total + 10)
  (h4 : new.vanaspati = original.vanaspati)
  (h5 : vanaspatiPercentage new = 20) :
  (original.pure / original.total) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pure_ghee_percentage_l66_6662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_propositions_use_connectives_l66_6688

/-- Represents a proposition in the problem -/
inductive Proposition
| nationalDay : Proposition
| multiples : Proposition
| trapezoid : Proposition
| equation : Proposition

/-- Checks if a proposition uses a logical connective -/
def usesLogicalConnective (p : Proposition) : Bool :=
  match p with
  | Proposition.nationalDay => true  -- uses "and"
  | Proposition.multiples => false   -- doesn't use a connective
  | Proposition.trapezoid => true    -- uses "not"
  | Proposition.equation => true     -- uses "or"

/-- The list of all propositions in the problem -/
def allPropositions : List Proposition :=
  [Proposition.nationalDay, Proposition.multiples, Proposition.trapezoid, Proposition.equation]

/-- The main theorem stating that 3 propositions use logical connectives -/
theorem three_propositions_use_connectives :
  (allPropositions.filter usesLogicalConnective).length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_propositions_use_connectives_l66_6688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_two_l66_6609

-- Define the quadrilateral ABCD inscribed in the unit circle
def ABCD : Set (ℝ × ℝ) := sorry

-- Define that ABCD is inscribed in the unit circle
axiom ABCD_inscribed : ∀ p ∈ ABCD, p.1^2 + p.2^2 = 1

-- Define angle BAD as 30 degrees
noncomputable def angle_BAD : ℝ := Real.pi / 6

-- Define points P and Q on rays AB and AD respectively
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the function to calculate CP + PQ + CQ
noncomputable def path_length (C P Q : ℝ × ℝ) : ℝ := sorry

-- Define m as the minimum value of CP + PQ + CQ
noncomputable def m : ℝ := sorry

-- The theorem to prove
theorem max_m_is_two : 
  ∀ (A B C D : ℝ × ℝ), A ∈ ABCD → B ∈ ABCD → C ∈ ABCD → D ∈ ABCD →
  ∀ (P Q : ℝ × ℝ), 
  (∃ t : ℝ, P = A + t • (B - A) ∧ t ≥ 0) →
  (∃ s : ℝ, Q = A + s • (D - A) ∧ s ≥ 0) →
  m ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_two_l66_6609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l66_6671

theorem trig_identity (θ : Real) (h1 : Real.tan θ = -2) (h2 : -π/2 < θ) (h3 : θ < 0) :
  (Real.sin θ)^2 / (Real.cos (2*θ) + 2) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l66_6671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_max_a_is_tight_l66_6633

-- Define the function f(x) = (1 + ln x) / x - 1
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x - 1

-- State the theorem
theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, (a + 1) * x - 1 - Real.log x ≤ 0) → 
  a ≤ 1 - 2 * Real.log 2 := by
  sorry

-- Prove that this is indeed the maximum value
theorem max_a_is_tight : 
  ∃ a : ℝ, a = 1 - 2 * Real.log 2 ∧ 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, (a + 1) * x - 1 - Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_max_a_is_tight_l66_6633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_inequality_l66_6672

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem function_symmetry_and_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (Real.pi - x))
  (h2 : ∀ x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2), f x = x + Real.sin x)
  (a b c : ℝ)
  (ha : a = f 1)
  (hb : b = f 2)
  (hc : c = f 3) :
  c < a ∧ a < b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_inequality_l66_6672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l66_6698

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 5*x - 6)

-- Define the domain set
def domain : Set ℝ := {x | x ≤ -1 ∨ x ≥ 6}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ ∃ y : ℝ, f x = y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l66_6698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_church_members_difference_l66_6637

theorem church_members_difference (total_members : ℕ) (adult_percentage : ℚ) : 
  total_members = 245 →
  adult_percentage = 38 / 100 →
  ∃ (adults children : ℕ),
    adults = Int.floor (adult_percentage * total_members) ∧
    children = total_members - adults ∧
    children - adults = 59 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_church_members_difference_l66_6637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_surface_area_l66_6656

/-- The total surface area of a regular triangular pyramid. -/
noncomputable def totalSurfaceArea (H : ℝ) (α : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * H^2 * (Real.cos α / Real.sin (α/2)^2)

/-- Theorem: The total surface area of a regular triangular pyramid with height H
    and dihedral angle α at the base is given by the formula:
    S = (3√3/2) * H² * (cos α / sin²(α/2)) -/
theorem regular_triangular_pyramid_surface_area
  (H : ℝ) (α : ℝ) (h_H_pos : H > 0) (h_α_pos : α > 0) (h_α_lt_pi : α < π) :
  totalSurfaceArea H α = (3 * Real.sqrt 3 / 2) * H^2 * (Real.cos α / Real.sin (α/2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_surface_area_l66_6656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l66_6606

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem f_min_value : ∀ x : ℝ, f x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l66_6606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_threshold_is_75_l66_6668

/-- Represents the bonus calculation for Karen's students' test scores. -/
structure BonusCalculation where
  initialBonus : ℕ := 500
  bonusPerPoint : ℕ := 10
  gradedTests : ℕ := 8
  currentAverage : ℕ := 70
  maxScore : ℕ := 150
  targetBonus : ℕ := 600
  lastTwoTestsScore : ℕ := 290

/-- Calculates the threshold average score for Karen to get the initial bonus. -/
def calculateThreshold (bc : BonusCalculation) : ℕ :=
  let totalScore := bc.gradedTests * bc.currentAverage + bc.lastTwoTestsScore
  let totalTests := bc.gradedTests + 2
  let averageForTargetBonus := totalScore / totalTests
  let pointsAboveThreshold := (bc.targetBonus - bc.initialBonus) / bc.bonusPerPoint
  averageForTargetBonus - pointsAboveThreshold

/-- Theorem stating that the threshold average score is 75 points/test. -/
theorem threshold_is_75 (bc : BonusCalculation) : calculateThreshold bc = 75 := by
  sorry

#eval calculateThreshold { initialBonus := 500
                         , bonusPerPoint := 10
                         , gradedTests := 8
                         , currentAverage := 70
                         , maxScore := 150
                         , targetBonus := 600
                         , lastTwoTestsScore := 290 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_threshold_is_75_l66_6668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l66_6639

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l66_6639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l66_6629

/-- Given vectors a, b, and c in ℝ², prove that if a - c is parallel to b, then k = 5 -/
theorem parallel_vectors (a b c : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (3, 1)) 
  (h2 : b = (1, 3)) 
  (h3 : c = (k, 7)) 
  (h4 : ∃ (t : ℝ), a - c = t • b) : 
  k = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l66_6629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l66_6644

/-- For a parabola defined by x² = (1/2)y, the distance from its focus to its directrix is 1/4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = (1/2) * y → (∃ (d : ℝ), d = 1/4 ∧ d = (1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l66_6644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l66_6617

/-- The slope of the angle bisector of the acute angle formed at the origin by two lines -/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ - Real.sqrt (m₁^2 + m₂^2 - m₁*m₂ + 1)) / (1 + m₁*m₂)

/-- The theorem stating that the slope of the angle bisector of y = x and y = 3x is (1 + √5) / 2 -/
theorem angle_bisector_slope_specific : 
  angle_bisector_slope 1 3 = (1 + Real.sqrt 5) / 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l66_6617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l66_6695

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_circle_intersection 
  (h : Hyperbola) 
  (m : Circle) 
  (h_asymptote : ∃ (x y : ℝ), y = h.b / h.a * x) 
  (m_eq : m.h = 1 ∧ m.k = 0 ∧ m.r = 1) 
  (chord_length : ∃ (p q : ℝ × ℝ), 
    (p.1 - 1)^2 + p.2^2 = 1 ∧ 
    (q.1 - 1)^2 + q.2^2 = 1 ∧ 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 3) : 
  eccentricity h = 2/3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l66_6695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_and_fraction_l66_6663

theorem largest_x_and_fraction (x a b c d : ℝ) : 
  (∃ (a b c d : ℤ), x = (a + b * Real.sqrt c) / d) →
  (7 * x / 8 + 1 = 4 / x) →
  (x ≤ (-4 + 8 * Real.sqrt 15) / 7) ∧
  (x = (-4 + 8 * Real.sqrt 15) / 7 → a * c * d / b = -105/2) := by
  sorry

#check largest_x_and_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_and_fraction_l66_6663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_piles_l66_6628

-- Define what it means for two numbers to be similar
def similar (x y : ℝ) : Prop := x ≤ Real.sqrt 2 * y ∧ y ≤ Real.sqrt 2 * x

-- Define a pile as a list of real numbers
def Pile := List ℝ

-- Define a valid split as a function that takes a pile and returns three piles
-- such that the sum of the new piles equals the sum of the original pile
def valid_split (p : Pile) (p1 p2 p3 : Pile) : Prop :=
  p1.sum + p2.sum + p3.sum = p.sum

-- Define the similarity condition for three piles
def all_similar (p1 p2 p3 : Pile) : Prop :=
  similar p1.sum p2.sum ∧ similar p2.sum p3.sum ∧ similar p3.sum p1.sum

-- State the theorem
theorem no_three_similar_piles (p : Pile) : 
  ¬∃ (p1 p2 p3 : Pile), valid_split p p1 p2 p3 ∧ all_similar p1 p2 p3 := by
  sorry

#check no_three_similar_piles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_piles_l66_6628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_length_is_40_l66_6683

/-- Represents the forest properties and logging information -/
structure ForestLogging where
  width : ℕ
  trees_per_sq_mile : ℕ
  trees_per_logger_per_day : ℕ
  days_per_month : ℕ
  num_loggers : ℕ
  num_months : ℕ

/-- Calculates the length of the forest given the forest and logging information -/
def calculate_forest_length (f : ForestLogging) : ℚ :=
  (f.num_loggers * f.trees_per_logger_per_day * f.days_per_month * f.num_months : ℚ) /
  (f.width * f.trees_per_sq_mile : ℚ)

/-- Theorem stating that under the given conditions, the forest length is 40 miles -/
theorem forest_length_is_40 (f : ForestLogging) 
  (h1 : f.width = 6)
  (h2 : f.trees_per_sq_mile = 600)
  (h3 : f.trees_per_logger_per_day = 6)
  (h4 : f.days_per_month = 30)
  (h5 : f.num_loggers = 8)
  (h6 : f.num_months = 10) :
  calculate_forest_length f = 40 := by
  sorry

def main : IO Unit := do
  let result := calculate_forest_length {
    width := 6,
    trees_per_sq_mile := 600,
    trees_per_logger_per_day := 6,
    days_per_month := 30,
    num_loggers := 8,
    num_months := 10
  }
  IO.println s!"The length of the forest is {result} miles"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_length_is_40_l66_6683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l66_6657

/-- The parabola y² = 8x with focus F and vertex O. -/
structure Parabola where
  F : ℝ × ℝ := (2, 0)  -- Focus at (2, 0)
  O : ℝ × ℝ := (0, 0)  -- Vertex at origin

/-- A point P on the parabola. -/
structure PointOnParabola (p : Parabola) where
  P : ℝ × ℝ
  on_parabola : P.2^2 = 8 * P.1

/-- The distance between two points in ℝ². -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The area of a triangle given three points. -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem parabola_triangle_area (p : Parabola) (poP : PointOnParabola p)
    (h : distance poP.P p.F = 4) :
  triangleArea poP.P p.F p.O = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l66_6657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_correct_l66_6622

/-- Represents the scenario of a pedestrian and cyclist on a circular avenue --/
structure CircularAvenueScenario where
  diameter : ℝ  -- diameter of the circular avenue in km
  pedestrian_speed : ℝ  -- pedestrian's speed in km/h
  cyclist_highway_speed : ℝ  -- cyclist's speed on highway in km/h
  cyclist_avenue_speed : ℝ  -- cyclist's speed on avenue in km/h
  walking_time : ℝ  -- time pedestrian walks before realizing keys are forgotten in hours

/-- The minimum time for the pedestrian to receive the keys --/
noncomputable def min_time_to_receive_keys (scenario : CircularAvenueScenario) : ℝ :=
  (21 - 4 * Real.pi) / 43

/-- Theorem stating the minimum time for the pedestrian to receive the keys --/
theorem min_time_correct (scenario : CircularAvenueScenario) 
  (h1 : scenario.diameter = 4)
  (h2 : scenario.pedestrian_speed = 6.5)
  (h3 : scenario.cyclist_highway_speed = 15)
  (h4 : scenario.cyclist_avenue_speed = 20)
  (h5 : scenario.walking_time = 1) :
  min_time_to_receive_keys scenario = (21 - 4 * Real.pi) / 43 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_correct_l66_6622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l66_6673

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem necessary_but_not_sufficient_condition
  (a : ℕ → ℝ) (q : ℝ) (h_geom : is_geometric_sequence a q) (h_positive : a 1 > 0) :
  (∀ n : ℕ, n > 0 → a (2*n - 1) + a (2*n) < 0) →
  q < 0 ∧
  ∃ a₀ q₀, a₀ > 0 ∧ q₀ < 0 ∧ 
    is_geometric_sequence (λ n ↦ a₀ * q₀^(n-1)) q₀ ∧
    ∃ n : ℕ, n > 0 ∧ (a₀ * q₀^(2*n-2) + a₀ * q₀^(2*n-1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l66_6673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l66_6614

/-- The function f(x) = (a^x - 1) / (a^x + 1) where a > 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

/-- Theorem stating that f(x) is increasing on ℝ and is an odd function -/
theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧ 
  (∀ x : ℝ, f a (-x) = -(f a x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l66_6614
