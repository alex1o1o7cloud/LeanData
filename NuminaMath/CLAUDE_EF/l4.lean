import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l4_492

/-- The ratio of the volume of an inscribed right circular cone to its enclosing right rectangular prism -/
theorem cone_prism_volume_ratio (a h : ℝ) (ha : a > 0) (hh : h > 0) : 
  (1/3 * Real.pi * a^2 * h) / (6 * a^2 * h) = Real.pi / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l4_492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_l4_478

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The equation of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_eccentricity_m (m : ℝ) :
  (∃ x y : ℝ, is_ellipse x y (Real.sqrt 2) (Real.sqrt m) ∧ eccentricity (Real.sqrt 2) (Real.sqrt m) = Real.sqrt 6 / 3) ∨
  (∃ x y : ℝ, is_ellipse x y (Real.sqrt m) (Real.sqrt 2) ∧ eccentricity (Real.sqrt m) (Real.sqrt 2) = Real.sqrt 6 / 3) →
  m = 2/3 ∨ m = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_l4_478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_impossibility_l4_443

theorem grid_sum_impossibility : ¬ ∃ (grid : Fin 3 → Fin 3 → ℕ),
  (∀ i j, grid i j ∈ Finset.range 10 \ {0}) ∧
  (∀ i j k, i ≠ j → grid i k ≠ grid j k) ∧
  (∀ i j k, i ≠ j → grid k i ≠ grid k j) ∧
  (∀ i, (Finset.sum (Finset.range 3) (λ j ↦ grid i j)) ∈ Finset.range 21 \ Finset.range 13) ∧
  (∀ j, (Finset.sum (Finset.range 3) (λ i ↦ grid i j)) ∈ Finset.range 21 \ Finset.range 13) ∧
  ((Finset.sum (Finset.range 3) (λ i ↦ grid i i)) ∈ Finset.range 21 \ Finset.range 13) ∧
  ((Finset.sum (Finset.range 3) (λ i ↦ grid i (2 - i))) ∈ Finset.range 21 \ Finset.range 13) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_impossibility_l4_443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equivalence_l4_428

theorem expression_equivalence : (5 : Int) + (-3) - (-7) - 2 = 5 - 3 + 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equivalence_l4_428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_sqrt_two_l4_435

/-- Infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.exp (x * Real.log x)

theorem infinite_power_tower_sqrt_two :
  ∃! (x : ℝ), x > 0 ∧ infinitePowerTower x = 4 ∧ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_sqrt_two_l4_435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_f_and_g_l4_445

-- Define the domain (0, +∞)
def PositiveReals := {x : ℝ | x > 0}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1)

-- State the theorem
theorem strictly_increasing_f_and_g :
  ∀ (x₁ x₂ : PositiveReals), x₁ < x₂ → f x₁ < f x₂ ∧ g x₁ < g x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_f_and_g_l4_445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l4_405

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(4, 10) is 24 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (4, 10)
  let area := (1/2 : ℝ) * |B.1 - A.1| * |C.2 - A.2|
  area = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l4_405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l4_493

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle -/
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A line in the xy-plane represented by ax + by = 0 -/
structure Line where
  a : ℝ
  b : ℝ

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2) / Real.sqrt (l.a^2 + l.b^2)

/-- A circle is tangent to a line if the distance from its center to the line equals its radius -/
def is_tangent_to_line (c : Circle) (l : Line) : Prop :=
  distance_point_to_line c.center l = c.radius

/-- A circle is tangent to the x-axis if the y-coordinate of its center equals its radius -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- A point is in the first quadrant if both its x and y coordinates are positive -/
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

theorem circle_equation (c : Circle) 
  (h1 : c.radius = 1)
  (h2 : is_in_first_quadrant c.center)
  (h3 : is_tangent_to_line c (Line.mk 4 (-3)))
  (h4 : is_tangent_to_x_axis c) :
  ∀ x y : ℝ, standard_equation c x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l4_493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l4_481

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_sum_magnitude (a b : V) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a - b‖ = 2) : 
  ‖a + b‖ = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l4_481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l4_491

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- State the theorem
theorem range_of_f :
  Set.range f = { y : ℝ | -1/2 ≤ y ∧ y ≤ 1/2 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l4_491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_individual_score_l4_432

theorem max_individual_score 
  (num_players : ℕ) 
  (total_score : ℕ) 
  (min_score : ℕ) 
  (score : ℕ → ℕ)
  (h1 : num_players = 12)
  (h2 : total_score = 100)
  (h3 : min_score = 7)
  (h4 : ∀ p, p < num_players → min_score ≤ score p) :
  ∃ max_score : ℕ, max_score = 23 ∧ 
  (∀ p, p < num_players → score p ≤ max_score) ∧
  (∃ q, q < num_players ∧ score q = max_score) ∧
  (Finset.sum (Finset.range num_players) score = total_score) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_individual_score_l4_432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_in_pyramid_main_theorem_l4_421

/-- Represents a pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  base_is_square : base_side = Real.sqrt 3
  lateral_faces_are_equilateral : True

/-- Represents a cube that fits inside the pyramid -/
structure InsideCube where
  side_length : ℝ
  bottom_on_base : True
  top_vertices_touch_midpoints : True

/-- The theorem stating the volume of the cube inside the specific pyramid -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) : 
  c.side_length ^ 3 = 27 / 64 := by
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem : 
  ∃ (p : Pyramid) (c : InsideCube), c.side_length ^ 3 = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_in_pyramid_main_theorem_l4_421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_parents_program_parents_l4_437

/-- Given a program with pupils and parents, calculate the number of parents. -/
theorem number_of_parents (num_pupils : ℕ) (total_people : ℕ) (h : num_pupils + (total_people - num_pupils) = total_people) :
  total_people - num_pupils = total_people - num_pupils :=
by
  rfl

/-- The specific problem instance -/
theorem program_parents :
  238 - 177 = 61 :=
by
  norm_num

#eval 238 - 177  -- This will evaluate to 61

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_parents_program_parents_l4_437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rightPyramidHeight_l4_420

/-- A right pyramid with a rectangular base -/
structure RightPyramid where
  /-- Width of the rectangular base -/
  baseWidth : ℝ
  /-- Length of the rectangular base -/
  baseLength : ℝ
  /-- Distance from apex to any vertex of the base -/
  apexToVertex : ℝ
  /-- The length is twice the width -/
  lengthTwiceWidth : baseLength = 2 * baseWidth
  /-- The perimeter of the base is 40 -/
  perimeterIs40 : 2 * (baseWidth + baseLength) = 40
  /-- The apex is 10 inches from each vertex -/
  apexDistance : apexToVertex = 10

/-- The height of the pyramid from its peak to the center of its base -/
noncomputable def pyramidHeight (p : RightPyramid) : ℝ :=
  -- Definition of height, to be proven
  20 / 3

/-- Theorem stating that the height of the pyramid is 20/3 -/
theorem rightPyramidHeight (p : RightPyramid) : pyramidHeight p = 20 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rightPyramidHeight_l4_420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_45pi_l4_439

-- Define points A, B, and C
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (4, 6)
def C : ℝ × ℝ := (10, 9)

-- Define the circle
def myCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem circle_area_is_45pi :
  let diameter := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let radius := diameter / 2
  let center := B
  π * radius^2 = 45 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_45pi_l4_439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_per_tankful_l4_479

/-- Represents the miles per gallon for a car in different driving conditions -/
structure MilesPerGallon where
  city : ℚ
  highway : ℚ

/-- Represents the miles per tankful for a car in different driving conditions -/
structure MilesPerTankful where
  city : ℚ
  highway : ℚ

/-- Calculates the tank size based on city miles per tankful and city miles per gallon -/
def tankSize (cityMilesPerTankful : ℚ) (cityMilesPerGallon : ℚ) : ℚ :=
  cityMilesPerTankful / cityMilesPerGallon

/-- Theorem: Given the conditions, prove that the car traveled 462 miles per tankful on the highway -/
theorem highway_miles_per_tankful
  (mpg : MilesPerGallon)
  (mpt : MilesPerTankful)
  (h1 : mpt.city = 336)
  (h2 : mpg.city = mpg.highway - 3)
  (h3 : mpg.city = 8) :
  mpt.highway = 462 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_per_tankful_l4_479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_is_36_sqrt_3_l4_497

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse_eq : hypotenuse = 12
  shorter_leg_eq : shorter_leg = 6
  longer_leg_eq : longer_leg = 6 * Real.sqrt 3

/-- The area of the overlapping region of two congruent 30-60-90 triangles -/
noncomputable def overlapping_area (t : Triangle30_60_90) : ℝ :=
  (1 / 2) * t.hypotenuse * t.longer_leg

/-- Theorem stating that the overlapping area is 36√3 -/
theorem overlapping_area_is_36_sqrt_3 (t : Triangle30_60_90) :
  overlapping_area t = 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_is_36_sqrt_3_l4_497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sams_sitting_fee_l4_452

/-- 
Given two photo companies with different pricing structures, this theorem proves
that Sam's Picture Emporium's sitting fee is $140 when both companies charge
the same total amount for 12 sheets of pictures.
-/
theorem sams_sitting_fee (johns_per_sheet johns_sitting_fee sams_per_sheet : ℚ) 
  (sams_sitting_fee : ℚ)
  (h1 : johns_per_sheet = 2.75)
  (h2 : johns_sitting_fee = 125)
  (h3 : sams_per_sheet = 1.50)
  (h4 : johns_per_sheet * 12 + johns_sitting_fee = sams_per_sheet * 12 + sams_sitting_fee) :
  sams_sitting_fee = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sams_sitting_fee_l4_452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_singleton_subset_l4_449

theorem existence_of_singleton_subset (n : ℕ) (S : Finset ℕ) (S_i : ℕ → Finset ℕ) 
  (h1 : S = Finset.range n)
  (h2 : ∀ i, i ∈ S → S_i i ⊆ S ∧ (S_i i).Nonempty)
  (h3 : ∀ i j, i ∈ S → j ∈ S → j ∈ S_i i → i ∈ S_i j)
  (h4 : ∀ i j, i ∈ S → j ∈ S → (S_i i).card = (S_i j).card → Disjoint (S_i i) (S_i j)) :
  ∃ k, k ∈ S ∧ (S_i k).card = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_singleton_subset_l4_449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l4_495

/-- An arithmetic sequence with common difference d and first term a_1 -/
noncomputable def arithmetic_sequence (d a_1 : ℝ) (n : ℕ) : ℝ := a_1 + d * (n - 1)

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S_n (d a_1 : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_properties (d a_1 : ℝ) :
  (S_n d a_1 4 = 2 * S_n d a_1 2 + 4) ∧
  (∀ n : ℕ+, S_n d a_1 n ≥ S_n d a_1 8) →
  d = 1 ∧ -8 ≤ a_1 ∧ a_1 ≤ -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l4_495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_odd_condition_l4_466

/-- A function f is odd if f(-x) = -f(x) for all real x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The sine function with phase shift φ -/
noncomputable def SineWithPhase (φ : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (x + φ)

theorem sine_phase_shift_odd_condition :
  (IsOdd (SineWithPhase 0)) ∧
  (∃ φ ≠ 0, IsOdd (SineWithPhase φ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_odd_condition_l4_466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l4_456

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a perpendicular line from the right focus F₂ to an asymptote has length 2
    and slope -1/2, then b = 2, the hyperbola equation is x² - y²/4 = 1,
    and the foot of the perpendicular is at (√5/5, 2√5/5). -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let f2 := (c, 0)
  let asymptote_slope := b / a
  let perp_line := λ x : ℝ ↦ -1/asymptote_slope * (x - c)
  let p := (a^2/c, a*b/c)
  (p.2 - f2.2) / (p.1 - f2.1) = -1/2 →
  (p.1 - f2.1)^2 + (p.2 - f2.2)^2 = 4 →
  b = 2 ∧
  (∀ x y : ℝ, x^2 - y^2/4 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) ∧
  p = (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l4_456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_a_l4_457

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

theorem tangent_line_and_minimum_a :
  (∃ (m b : ℝ), ∀ x, m * x + b = (deriv f 1) * (x - 1) + f 1 ∧ m = -15 ∧ b = 29) ∧
  (∃ (a : ℤ), (∀ x > 0, f x ≤ (↑a - 3) * x^2 + (2 * ↑a - 13) * x + 1) ∧
              (∀ a' : ℤ, a' < a → ∃ x > 0, f x > (↑a' - 3) * x^2 + (2 * ↑a' - 13) * x + 1) ∧
              a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_a_l4_457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersection_locus_l4_484

open EuclideanGeometry

/-- The perpendicular line from a point P to a line PQ -/
noncomputable def perpendicularLine (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The intersection point of two lines -/
noncomputable def lineIntersection (l₁ l₂ : Set (EuclideanSpace ℝ (Fin 2))) : Option (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The circumcircle of a triangle -/
noncomputable def circumcircle (A B C : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The set of points M such that the perpendiculars from A to AM, B to BM, and C to CM intersect at a single point -/
noncomputable def perpIntersectionLocus (A B C : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  { M | ∃ P, lineIntersection (perpendicularLine A M) (perpendicularLine B M) = some P ∧
          lineIntersection (perpendicularLine A M) (perpendicularLine C M) = some P ∧
          lineIntersection (perpendicularLine B M) (perpendicularLine C M) = some P }

theorem perpendicular_intersection_locus (A B C : EuclideanSpace ℝ (Fin 2)) :
  perpIntersectionLocus A B C = circumcircle A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersection_locus_l4_484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_zhang_is_half_l4_426

-- Define the number of doctors and the number to be selected
def total_doctors : ℕ := 4
def doctors_to_select : ℕ := 2

-- Define a function to calculate combinations
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting Dr. Zhang
noncomputable def prob_select_zhang : ℚ :=
  (combinations (total_doctors - 1) (doctors_to_select - 1) : ℚ) / 
  (combinations total_doctors doctors_to_select : ℚ)

-- Theorem statement
theorem prob_select_zhang_is_half : prob_select_zhang = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_zhang_is_half_l4_426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_ge_one_over_e_l4_414

/-- The function f(x) defined as ae^(ax) - ln(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (a * x) - Real.log x

/-- Theorem stating the condition for f(x) ≥ 0 when x > 1 --/
theorem f_nonnegative_iff_a_ge_one_over_e :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x ≥ 0) ↔ a ≥ 1 / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_ge_one_over_e_l4_414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l4_464

-- Define the function f
noncomputable def f (a m x : ℝ) : ℝ := a^m / x

-- State the theorem
theorem problem_solution (a m : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc a (2*a), f a m x ∈ Set.Icc (a^2) (a^3)) →
  (a = 2 ∧ 
   ∀ s : ℝ, (∀ x t : ℝ, x ∈ Set.Icc 0 s → (x + t)^2 + 2*(x + t) ≤ (a + 1)*x) ↔ s ∈ Set.Ioo 0 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l4_464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_l4_401

theorem choir_average_age (num_females : ℕ) (num_males : ℕ) 
  (avg_age_females : ℚ) (avg_age_males : ℚ) :
  num_females = 10 →
  num_males = 18 →
  avg_age_females = 32 →
  avg_age_males = 35 →
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 33.92857 := by
  sorry

#eval (10 * 32 + 18 * 35) / (10 + 18)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_l4_401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l4_402

-- Definitions needed for the theorem
def IsoscelesTriangle (perimeter : ℝ) (angle : ℝ) : Prop :=
  sorry

def TriangleArea (perimeter : ℝ) (angle : ℝ) : ℝ :=
  sorry

theorem isosceles_triangle_area (p : ℝ) : 
  let perimeter : ℝ := 6 * p
  let angle_between_equal_sides : ℝ := 60
  let triangle_area : ℝ := (9 * Real.sqrt 3 * p^2) / 8
  IsoscelesTriangle perimeter angle_between_equal_sides → 
  TriangleArea perimeter angle_between_equal_sides = triangle_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l4_402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l4_427

noncomputable def f (x k a : ℝ) : ℝ :=
  if x ≥ 0 then x + k * (1 - a^2)
  else x^2 - 4*x + (3 - a)^2

theorem k_range (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f x₁ k a = f x₂ k a) →
  (k ≤ 0 ∨ k ≥ 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l4_427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_paint_intensity_problem_l4_413

/-- Calculates the intensity of replacement paint given the original paint intensity,
    new paint intensity, and fraction of original paint replaced. -/
def replacement_paint_intensity (original_intensity new_intensity fraction_replaced : ℚ) : ℚ :=
  (new_intensity - (1 - fraction_replaced) * original_intensity) / fraction_replaced

/-- Theorem stating that given the specific conditions of the problem,
    the replacement paint intensity is 30%. -/
theorem replacement_paint_intensity_problem :
  let original_intensity : ℚ := 60 / 100
  let new_intensity : ℚ := 40 / 100
  let fraction_replaced : ℚ := 2 / 3
  replacement_paint_intensity original_intensity new_intensity fraction_replaced = 30 / 100 := by
  sorry

#eval replacement_paint_intensity (60 / 100) (40 / 100) (2 / 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_paint_intensity_problem_l4_413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_of_equilateral_triangle_l4_469

-- Define the triangle ABC
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  side_length : ℝ
  is_equilateral : side_length = 24

-- Define the incenter I
noncomputable def incenter (t : EquilateralTriangle) : ℝ × ℝ := sorry

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem inradius_of_equilateral_triangle (t : EquilateralTriangle) :
  let I := incenter t
  distance I t.C = 12 * Real.sqrt 3 →
  (∃ r : ℝ, r = 4 * Real.sqrt 3 ∧
    r = distance I t.A ∧
    r = distance I t.B ∧
    r = distance I t.C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_of_equilateral_triangle_l4_469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l4_404

/-- The number of candy pieces caught by Tabitha -/
def tabitha_candy : ℕ → Prop := fun t => t > 0

/-- The total number of candy pieces caught by all friends -/
def total_candy : ℕ := 72

/-- The number of candy pieces caught by Stan -/
def stan_candy : ℕ := 13

/-- Proposition that Julie caught half as much candy as Tabitha -/
def julie_candy (t : ℕ) : ℕ := t / 2

/-- Proposition that Carlos caught twice as much candy as Stan -/
def carlos_candy (s : ℕ) : ℕ := 2 * s

theorem candy_distribution (t : ℕ) :
  total_candy = 72 ∧
  stan_candy = 13 ∧
  julie_candy t = t / 2 ∧
  carlos_candy stan_candy = 2 * stan_candy ∧
  t + stan_candy + (julie_candy t) + (carlos_candy stan_candy) = total_candy →
  tabitha_candy t ∧ t = 22 :=
by
  intro h
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l4_404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mileage_difference_l4_412

-- Define the given values
noncomputable def advertised_mileage : ℝ := 35
noncomputable def tank_capacity : ℝ := 12
noncomputable def total_miles : ℝ := 372

-- Define the actual mileage
noncomputable def actual_mileage : ℝ := total_miles / tank_capacity

-- State the theorem
theorem mileage_difference : advertised_mileage - actual_mileage = 4 := by
  -- Expand the definition of actual_mileage
  unfold actual_mileage
  -- Perform the calculation
  norm_num
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mileage_difference_l4_412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_original_price_l4_444

/-- The original selling price of a shirt given specific discounts and taxes. -/
noncomputable def original_price (final_price : ℝ) (discount1 : ℝ) (tax : ℝ) (discount2 : ℝ) : ℝ :=
  let price_before_discount2 := final_price / (1 - discount2)
  let price_before_tax := price_before_discount2 / (1 + tax)
  price_before_tax / (1 - discount1)

/-- Theorem stating the original price of the shirt given the problem conditions. -/
theorem shirt_original_price :
  ∃ (price : ℝ),
    0.99 * price ≤ original_price 650 0.32 0.15 0.10 ∧
    original_price 650 0.32 0.15 0.10 ≤ 1.01 * price ∧
    abs (price - 922.62) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval original_price 650 0.32 0.15 0.10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_original_price_l4_444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l4_411

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_lambda (l : ℝ) :
  let a : ℝ × ℝ := (2, 6)
  let b : ℝ × ℝ := (-1, l)
  are_parallel a b → l = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l4_411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_distance_l4_473

noncomputable def total_distance : ℝ := 129.9999999999999
noncomputable def train_fraction : ℝ := 3/5
noncomputable def bus_fraction : ℝ := 7/20

theorem walk_distance : 
  total_distance * (1 - train_fraction - bus_fraction) = 6.499999999999991 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_distance_l4_473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_AB_equals_target_l4_434

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point is on a circle -/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The area inside circle C but outside circles A and B -/
noncomputable def areaOutsideAB (A B C : Circle) : ℝ :=
  sorry

theorem area_outside_AB_equals_target (A B C : Circle) :
  A.radius = 1 →
  B.radius = 1 →
  C.radius = 2 →
  (∃ p : ℝ × ℝ, onCircle p A ∧ onCircle p B) →  -- A and B share a point of tangency
  (∃ m : ℝ × ℝ, onCircle m C ∧ m.1 = (A.center.1 + B.center.1) / 2 ∧ m.2 = (A.center.2 + B.center.2) / 2) →  -- C touches midpoint of AB
  areaOutsideAB A B C = (10 * Real.pi / 3) + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_AB_equals_target_l4_434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_rectangle_with_quarter_circles_l4_472

/-- The area of the shaded region in a rectangle with quarter circles at its corners -/
theorem shaded_area_rectangle_with_quarter_circles 
  (length : ℝ) 
  (width : ℝ) 
  (h_length : length = 15) 
  (h_width : width = 10) :
  length * width - π * (width / 2)^2 = 150 - 25 * π := by
  -- Substitute the given values
  have h1 : length * width = 150 := by rw [h_length, h_width]; norm_num
  have h2 : (width / 2)^2 = 25 := by rw [h_width]; norm_num
  
  -- Rewrite the goal using these facts
  rw [h1, h2]
  
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_rectangle_with_quarter_circles_l4_472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_theorem_l4_429

/-- The locus of points at a distance of 2 from the line 3x - 4y - 1 = 0 -/
def locus_of_points (x y : ℝ) : Prop :=
  (3*x - 4*y - 11 = 0) ∨ (3*x - 4*y + 9 = 0)

/-- The distance from a point (x, y) to the line 3x - 4y - 1 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (3*x - 4*y - 1) / Real.sqrt (3^2 + 4^2)

theorem locus_theorem :
  ∀ x y : ℝ, distance_to_line x y = 2 ↔ locus_of_points x y :=
by
  sorry

#check locus_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_theorem_l4_429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_l4_467

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if an ellipse contains a circle -/
def ellipseContainsCircle (e : Ellipse) (c : Circle) : Prop :=
  ∀ p : Point, isOnCircle p c → isOnEllipse p e

/-- The area of an ellipse -/
noncomputable def ellipseArea (e : Ellipse) : ℝ :=
  Real.pi * e.a * e.b

/-- The main theorem -/
theorem smallest_ellipse_area 
  (e : Ellipse) 
  (c1 c2 : Circle) 
  (h1 : c1.center = ⟨2, 0⟩ ∧ c1.radius = 2)
  (h2 : c2.center = ⟨-2, 0⟩ ∧ c2.radius = 2)
  (h3 : ellipseContainsCircle e c1)
  (h4 : ellipseContainsCircle e c2) :
  ellipseArea e ≥ 4 * Real.sqrt 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_l4_467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_is_one_l4_489

/-- The base number in the original problem -/
def base : ℝ := 10^4 * 3.456789

/-- The number of digits to the right of the decimal place in the result -/
def decimal_digits : ℕ := 18

/-- The exponent we need to find -/
def exponent : ℝ := 1

/-- Function to check if a real number has a specific number of digits after the decimal point -/
def has_decimal_digits (x : ℝ) (n : ℕ) : Prop := sorry

/-- Theorem stating that the exponent is 1 -/
theorem exponent_is_one : 
  ∃ (x : ℝ), (has_decimal_digits (base^x) decimal_digits) → x = exponent :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_is_one_l4_489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_time_is_one_point_five_l4_442

/-- Represents Alex's bike trip with given conditions --/
structure BikeTripData where
  flatSpeed : ℝ
  flatTime : ℝ
  uphillSpeed : ℝ
  uphillTime : ℝ
  downhillSpeed : ℝ
  totalDistance : ℝ
  walkedDistance : ℝ

/-- Calculates the time spent riding downhill --/
noncomputable def downhillTime (data : BikeTripData) : ℝ :=
  let flatDistance := data.flatSpeed * data.flatTime
  let uphillDistance := data.uphillSpeed * data.uphillTime
  let totalBeforePuncture := data.totalDistance - data.walkedDistance
  let downhillDistance := totalBeforePuncture - flatDistance - uphillDistance
  downhillDistance / data.downhillSpeed

/-- Theorem stating that the downhill time is 1.5 hours for the given conditions --/
theorem downhill_time_is_one_point_five (data : BikeTripData)
    (h1 : data.flatSpeed = 20)
    (h2 : data.flatTime = 4.5)
    (h3 : data.uphillSpeed = 12)
    (h4 : data.uphillTime = 2.5)
    (h5 : data.downhillSpeed = 24)
    (h6 : data.totalDistance = 164)
    (h7 : data.walkedDistance = 8) :
    downhillTime data = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_time_is_one_point_five_l4_442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_pieces_present_l4_496

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing pieces -/
def missing_pieces : ℕ := 10

/-- Theorem: The number of chess pieces present is 22 -/
theorem chess_pieces_present : standard_chess_pieces - missing_pieces = 22 := by
  rfl

#eval standard_chess_pieces - missing_pieces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_pieces_present_l4_496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_ln_quadratic_l4_488

/-- The function f(x) = ln(x^2 - ax - 3) is monotonically increasing on (1, +∞) if and only if a ∈ (-∞, -2] -/
theorem monotone_increasing_ln_quadratic (a : ℝ) :
  (∀ x > 1, Monotone (fun x => Real.log (x^2 - a*x - 3))) ↔ a ∈ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_ln_quadratic_l4_488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_each_attraction_has_visitors_l4_423

-- Define the number of tourists and attractions
def num_tourists : ℕ := 4
def num_attractions : ℕ := 3

-- Define the total number of possible outcomes
def total_outcomes : ℕ := num_attractions ^ num_tourists

-- Define the number of favorable outcomes (each attraction has visitors)
def favorable_outcomes : ℕ := (num_tourists.choose 2) * (Nat.factorial num_attractions)

-- Define the probability of each attraction having visitors
noncomputable def probability_all_visited : ℚ := 
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem statement
theorem probability_each_attraction_has_visitors :
  probability_all_visited = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_each_attraction_has_visitors_l4_423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_k_l4_450

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k / (x - 2)

theorem max_value_implies_k (k : ℝ) (h1 : k > 0) :
  (∀ x ∈ Set.Icc 4 6, f k x ≤ 1) ∧ (∃ x ∈ Set.Icc 4 6, f k x = 1) → k = 2 := by
  sorry

#check max_value_implies_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_k_l4_450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_inequality_range_l4_431

def sequence_a : ℕ → ℚ
  | 0 => 3/2  -- Add a case for 0 to avoid missing cases error
  | 1 => 3/2
  | n + 1 => 3 * sequence_a n - 1

def sequence_b (n : ℕ) : ℚ := sequence_a n - 1/2

theorem geometric_sequence_and_inequality_range :
  (∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) = 3 * sequence_b n) ∧
  (∀ m : ℚ, (∀ n : ℕ, n ≥ 1 → (sequence_b n + 1) / (sequence_b (n + 1) - 1) ≤ m) ↔ m ≥ 1) := by
  sorry

#check geometric_sequence_and_inequality_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_inequality_range_l4_431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_function_and_inverse_l4_459

def f (b : ℤ) : ℝ → ℝ := λ x ↦ 5 * x + b

theorem intersection_point_of_function_and_inverse (b a : ℤ) :
  (∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv (f b) ∧ Function.RightInverse f_inv (f b)) →
  f b (-3) = a →
  f b a = -3 →
  a = -3 := by
  intros h1 h2 h3
  sorry

#check intersection_point_of_function_and_inverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_function_and_inverse_l4_459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l4_460

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (3, 0)
def F₂ : ℝ × ℝ := (-3, 0)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the property that P satisfies
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  distance P F₁ + distance P F₂ = 10

-- Define the equation of the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 / 25 + P.2^2 / 16 = 1

-- State the theorem
theorem ellipse_trajectory :
  ∀ P : ℝ × ℝ, satisfies_condition P ↔ on_ellipse P :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l4_460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l4_468

theorem rectangular_to_polar_conversion :
  ∀ (x y ρ θ : ℝ),
    x = π / 2 →
    y = -Real.sqrt 3 * π / 2 →
    ρ > 0 →
    0 ≤ θ ∧ θ < 2 * π →
    ρ * Real.cos θ = x →
    ρ * Real.sin θ = y →
    ρ = π ∧ θ = 5 * π / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l4_468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cylinder_volumes_l4_409

/-- A cylinder with a lateral surface that unfolds to a rectangle --/
structure UnfoldedCylinder where
  length : ℝ  -- Length of the unfolded rectangle
  width : ℝ   -- Width of the unfolded rectangle

/-- The set of possible volumes for an unfolded cylinder --/
def possibleVolumes (c : UnfoldedCylinder) : Set ℝ :=
  {v | ∃ (r h : ℝ), 
    ((2 * Real.pi * r = c.length ∧ h = c.width) ∨ 
     (2 * Real.pi * r = c.width ∧ h = c.length)) ∧
    v = Real.pi * r^2 * h}

/-- The main theorem stating the possible volumes of the specific cylinder --/
theorem specific_cylinder_volumes :
  let c : UnfoldedCylinder := ⟨4, 2⟩
  possibleVolumes c = {4 / Real.pi, 8 / Real.pi} := by
  sorry

#check specific_cylinder_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cylinder_volumes_l4_409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l4_462

-- Define the triangle XYZ
structure RightTriangle where
  X : Real
  Y : Real
  Z : Real
  right_angle_at_Y : Y = 90
  sin_Z : Real.sin Z = 3/5
  YZ : Real

-- State the theorem
theorem right_triangle_properties (t : RightTriangle) (h : t.YZ = 10) :
  Real.cos t.X = 3/5 ∧ (t.YZ * Real.sin t.Z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l4_462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_circle_a_range_l4_480

-- Define the lines and circle
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - y + a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * x - y + a^2 + 1 = 0
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 = 0

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y, (line1 a x y ∧ my_circle x y) ∨ (line2 a x y ∧ my_circle x y)

-- Define the parallel condition
def are_parallel (a : ℝ) : Prop :=
  ∀ x y, line1 a x y ↔ line2 a (x + (a^2 - a + 1) / 2) y

-- Theorem statement
theorem tangent_lines_circle_a_range (a : ℝ) :
  (are_parallel a ∧ is_tangent a) →
  (-3 ≤ a ∧ a ≤ -Real.sqrt 6) ∨ (Real.sqrt 6 ≤ a ∧ a ≤ 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_circle_a_range_l4_480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_equality_l4_471

theorem complex_magnitude_equality (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  (∀ a b, Real.sqrt (a^2 + b^2) = (Complex.abs (↑a + ↑b * Complex.I))^2) ∧
  (∃ a b, Real.sqrt (a^2 + b^2) ≠ (a - b)^2) ∧
  (∃ a b, Real.sqrt (a^2 + b^2) ≠ (abs a + abs b)) ∧
  (∃ a b, Real.sqrt (a^2 + b^2) ≠ abs (a * b)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_equality_l4_471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_g_minimum_value_l4_436

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

-- Define the function g(x, a)
noncomputable def g (x a : ℝ) : ℝ := a * x + Real.exp x

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x, m * x + b = 2 * x - f 1 := by
  sorry

-- Theorem for the minimum value of g(x)
theorem g_minimum_value (a : ℝ) (h : a < -1) :
  ∃ x₀, ∀ x, g x a ≥ g x₀ a ∧ g x₀ a = -a + a * Real.log (-a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_g_minimum_value_l4_436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l4_485

-- Define the function f as noncomputable due to its dependency on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := (4 * x + 2) / Real.sqrt (x - 7)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 7}

-- Theorem stating that the domain of f is (7, ∞)
theorem domain_of_f : 
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ∈ domain_f :=
by
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l4_485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_line_intersecting_circle_l4_455

/-- Given a line passing through the point (-√3, -1) and intersecting the unit circle,
    prove that its slope k is between 0 and √3 inclusive. -/
theorem slope_range_for_line_intersecting_circle :
  ∀ k : ℝ,
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y + 1 = k * (x + Real.sqrt 3)) →
  0 ≤ k ∧ k ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_line_intersecting_circle_l4_455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_essay_time_is_one_hour_l4_422

/-- Represents the homework assignment and its components -/
structure HomeworkAssignment where
  totalTime : ℚ
  shortAnswerTime : ℚ
  paragraphTime : ℚ
  essayCount : ℕ
  paragraphCount : ℕ
  shortAnswerCount : ℕ

/-- Calculates the time for one essay given a homework assignment -/
noncomputable def timeForOneEssay (hw : HomeworkAssignment) : ℚ :=
  let shortAnswerTotalTime := hw.shortAnswerCount * hw.shortAnswerTime / 60
  let paragraphTotalTime := hw.paragraphCount * hw.paragraphTime / 60
  let remainingTime := hw.totalTime - shortAnswerTotalTime - paragraphTotalTime
  remainingTime / hw.essayCount

/-- Theorem stating that the time for one essay is 1 hour given the specific homework assignment -/
theorem essay_time_is_one_hour :
  let hw : HomeworkAssignment := {
    totalTime := 4
    shortAnswerTime := 3
    paragraphTime := 15
    essayCount := 2
    paragraphCount := 5
    shortAnswerCount := 15
  }
  timeForOneEssay hw = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_essay_time_is_one_hour_l4_422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_form_valid_set_l4_406

-- Define the properties of a valid set
structure ValidSet (α : Type) where
  elements : Set α
  is_definite : Bool
  has_specific_standard : Bool

-- Define the group of students
def students_2012_panzhihua : Set String := sorry

-- Define the properties of the student group
axiom students_are_definite : (ValidSet.mk students_2012_panzhihua true true).is_definite = true
axiom students_have_standard : (ValidSet.mk students_2012_panzhihua true true).has_specific_standard = true

-- Theorem stating that the student group forms a valid set
theorem students_form_valid_set : 
  ∃ (s : ValidSet String), s.elements = students_2012_panzhihua ∧ 
                            s.is_definite = true ∧ 
                            s.has_specific_standard = true := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_form_valid_set_l4_406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l4_458

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 2) : 
  let k := perimeter / (2 * (length_ratio + width_ratio : ℝ))
  let length := k * length_ratio
  let width := k * width_ratio
  let diagonal := Real.sqrt (length^2 + width^2)
  diagonal = 194 / 7 := by
  -- Unfold the let bindings
  simp only [h_perimeter, h_ratio]
  -- Simplify the expressions
  norm_num
  -- Skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l4_458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_in_interval_l4_417

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 2 * Real.pi / 3) + 1

-- State the theorem
theorem g_range_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi/3) (Real.pi/2),
  g x ∈ Set.Icc (-1) (Real.sqrt 3 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_in_interval_l4_417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_is_4_days_l4_490

/-- The time taken for workers a, b, and c to complete a job together -/
noncomputable def job_completion_time (b_speed : ℝ) : ℝ :=
  let a_speed := 3 * b_speed
  let c_initial_speed := 1.5 * b_speed
  let c_doubled_speed := 2 * c_initial_speed
  let work_rate_before := a_speed + b_speed + c_initial_speed
  let work_rate_after := a_speed + b_speed + c_doubled_speed
  let work_done_4_days := 4 * work_rate_before
  if work_done_4_days ≥ 1 then 4 else 4 + (1 - work_done_4_days) / work_rate_after

theorem job_completion_time_is_4_days (b_speed : ℝ) (h : b_speed > 0) :
  job_completion_time b_speed = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_is_4_days_l4_490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_equal_28_oranges_is_one_l4_408

/-- The price of one apple in rupees -/
def apple_price : ℝ := sorry

/-- The price of one orange in rupees -/
def orange_price : ℝ := sorry

/-- The number of apples with the same price as 28 oranges -/
def apples_equal_28_oranges : ℕ := sorry

/-- The price of some apples is equal to that of 28 oranges -/
axiom price_equality : apple_price = 28 * orange_price

/-- The price of 45 apples and 60 oranges together is Rs. 1350 -/
axiom total_price_1 : 45 * apple_price + 60 * orange_price = 1350

/-- The total price of 30 apples and 40 oranges is Rs. 900 -/
axiom total_price_2 : 30 * apple_price + 40 * orange_price = 900

/-- The number of apples with the same price as 28 oranges is 1 -/
theorem apples_equal_28_oranges_is_one : apples_equal_28_oranges = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_equal_28_oranges_is_one_l4_408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_subsets_l4_454

def A : Finset Nat := {1, 2, 3, 4, 5}

def even_subsets (S : Finset Nat) : Finset (Finset Nat) :=
  S.powerset.filter (fun s => ∃ x ∈ s, Even x)

theorem count_even_subsets :
  (even_subsets A).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_subsets_l4_454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_perp_beta_sufficient_not_necessary_l4_418

-- Define the basic structures
variable (α β : Type) (m : Type)

-- Define the relationships
def lies_in (l m : Type) : Prop := sorry
def perpendicular (p1 p2 : Type) : Prop := sorry
def line_perpendicular_to_plane (l p : Type) : Prop := sorry

-- State the theorem
theorem alpha_perp_beta_sufficient_not_necessary 
  (α β : Type) (m : Type)
  (h1 : lies_in m α) 
  (h2 : α ≠ β) : 
  (∀ (α β m : Type), perpendicular α β → line_perpendicular_to_plane m β) ∧ 
  (∃ (α β m : Type), line_perpendicular_to_plane m β ∧ ¬perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_perp_beta_sufficient_not_necessary_l4_418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_25_l4_476

def S : Set ℤ := {-25, -4, -1, 1, 3, 9}

theorem largest_quotient_is_25 : 
  ∀ a b : ℤ, a ∈ S → b ∈ S → a ≠ 0 → 
    (∃ c d : ℤ, c ∈ S ∧ d ∈ S ∧ c ≠ 0 ∧ |a / b| ≤ |c / d|) ∧ 
    (∃ e f : ℤ, e ∈ S ∧ f ∈ S ∧ e ≠ 0 ∧ |e / f| = 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_25_l4_476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gini_coefficient_change_l4_433

-- Define the regions
structure Region where
  population : ℕ
  ppc : ℝ → ℝ
  maxSets : ℝ

-- Define the country
structure Country where
  north : Region
  south : Region

-- Define the Gini coefficient calculation
noncomputable def giniCoefficient (poorer : ℝ) (richer : ℝ) : ℝ :=
  (poorer / (poorer + richer)) - (poorer * 2000 / ((poorer + richer) * 2000))

-- Define the problem
noncomputable def giniProblem (c : Country) : Prop :=
  let totalPopulation := (c.north.population + c.south.population : ℝ)
  let northIncome := c.north.maxSets * 2000
  let southIncome := c.south.maxSets * 2000
  let initialGini := giniCoefficient (c.north.population : ℝ) (c.south.population : ℝ)
  let newNorthIncome := northIncome + 661
  let newSouthIncome := (c.north.maxSets + c.south.maxSets) * 2000 - newNorthIncome
  let newGini := giniCoefficient (c.south.population : ℝ) (c.north.population : ℝ)
  initialGini = 0.2 ∧ newGini - initialGini = -0.001

-- Theorem statement
theorem gini_coefficient_change (c : Country) 
  (h1 : c.north.population = 24)
  (h2 : c.south.population = 6)
  (h3 : c.north.ppc = fun x => 13.5 - 9 * x)
  (h4 : c.south.ppc = fun x => 1.5 * x^2 - 24)
  (h5 : c.north.maxSets = 18)
  (h6 : c.south.maxSets = 12) :
  giniProblem c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gini_coefficient_change_l4_433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_PC_l4_400

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define an equilateral triangle
def is_equilateral_triangle (P A B : ℝ × ℝ) : Prop :=
  let d := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d P A = d A B ∧ d A B = d B P

-- Define a point on the circle
def point_on_circle (p : ℝ × ℝ) : Prop := circle_C p.1 p.2

-- Define a chord of the circle
def is_chord (A B : ℝ × ℝ) : Prop :=
  point_on_circle A ∧ point_on_circle B

-- Theorem statement
theorem max_length_PC (P A B : ℝ × ℝ) :
  is_equilateral_triangle P A B →
  is_chord A B →
  (∃ (C : ℝ × ℝ), point_on_circle C ∧
    ∀ (D : ℝ × ℝ), point_on_circle D →
      Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) ≥
      Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)) →
  ∃ (C : ℝ × ℝ), point_on_circle C ∧
    Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_PC_l4_400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_3x_plus_exp_x_l4_486

/-- The definite integral of 3x + e^x from 0 to 1 equals e + 1/2 -/
theorem integral_3x_plus_exp_x : ∫ (x : ℝ) in Set.Icc 0 1, (3 * x + Real.exp x) = Real.exp 1 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_3x_plus_exp_x_l4_486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_line_through_M_l4_407

-- Define the vectors
def AB : ℝ × ℝ × ℝ := (2, -1, 3)
def AC : ℝ × ℝ × ℝ := (-2, 1, 0)
def AP : ℝ × ℝ × ℝ := (3, -1, 4)

-- Define point M
def M : ℝ × ℝ × ℝ := (1, 1, 1)

-- Volume of the triangular pyramid
noncomputable def volume_P_ABC : ℝ := 1/2

-- Equation of the line
def line_equation (x y z : ℝ) : Prop :=
  (1 - x) / 2 = (1 - y) / (-1) ∧ (1 - x) / 2 = (1 - z) / 3

-- Theorem statements
theorem volume_pyramid : 
  volume_P_ABC = 1/2 := by sorry

theorem line_through_M :
  ∀ (x y z : ℝ), line_equation x y z ↔ 
    ∃ (t : ℝ), (x, y, z) = (1 + 2*t, 1 - t, 1 + 3*t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_line_through_M_l4_407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_tan_minus_x_over_x_minus_sin_l4_465

open Real

theorem limit_tan_minus_x_over_x_minus_sin : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ → |((tan x - x) / (x - sin x)) - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_tan_minus_x_over_x_minus_sin_l4_465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_person_share_approx_112_l4_499

/-- Represents the share distribution problem -/
structure ShareDistribution where
  total : ℕ
  num_people : ℕ
  ratios : Fin 5 → ℚ

/-- Calculates the share of the first person given a share distribution -/
def first_person_share (sd : ShareDistribution) : ℚ :=
  let sum_ratios := (Finset.range sd.num_people).sum (λ i => sd.ratios i)
  sd.total * sd.ratios 0 / sum_ratios

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (q : ℚ) : ℤ :=
  (q + 1/2).floor

/-- The theorem stating the approximate share of the first person -/
theorem first_person_share_approx_112 (sd : ShareDistribution) :
  sd.total = 950 ∧
  sd.num_people = 5 ∧
  sd.ratios 0 = 1 ∧
  sd.ratios 1 = 4/3 ∧
  sd.ratios 2 = 5/2 ∧
  sd.ratios 3 = 5/3 ∧
  sd.ratios 4 = 2 →
  roundToNearest (first_person_share sd) = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_person_share_approx_112_l4_499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5y2_in_expansion_l4_451

def binomial_expansion (a b : ℕ → ℕ) (n : ℕ) : ℕ → ℕ → ℕ :=
  sorry

theorem coefficient_x5y2_in_expansion :
  let expansion := binomial_expansion
    (fun k => if k = 2 then 1 else if k = 1 then 2 else if k = 0 then 3 else 0)
    (fun k => if k = 1 then 1 else 0)
    5
  expansion 5 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5y2_in_expansion_l4_451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l4_424

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then Real.exp (x * Real.log 2) else m - x^2

theorem function_properties :
  (∀ m < 0, ¬∃ x, f m x = 0) ∧
  f (1/4) (f (1/4) (-1)) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l4_424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_roots_l4_475

-- Define the logarithms
noncomputable def log_2_3 : ℝ := Real.log 3 / Real.log 2
noncomputable def log_3_5 : ℝ := Real.log 5 / Real.log 3
noncomputable def log_5_2 : ℝ := Real.log 2 / Real.log 5

-- Define the polynomial
noncomputable def polynomial (x : ℝ) : ℝ :=
  x^3 - (log_2_3 + log_3_5 + log_5_2) * x^2 +
  (log_2_3 * log_3_5 + log_3_5 * log_5_2 + log_5_2 * log_2_3) * x -
  log_2_3 * log_3_5 * log_5_2

-- Theorem statement
theorem log_roots :
  (polynomial log_2_3 = 0) ∧
  (polynomial log_3_5 = 0) ∧
  (polynomial log_5_2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_roots_l4_475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_area_bound_l4_470

/-- A point in a 2D plane with a color attribute -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Fin 3

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : ColoredPoint) : ℝ := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : ColoredPoint) : Prop := sorry

/-- A set of 18 points on a plane, with 6 of each color -/
def PointSet := { points : Finset ColoredPoint // points.card = 18 ∧ 
  (∀ c : Fin 3, (points.filter (λ p => p.color = c)).card = 6) ∧
  (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) }

/-- The sum of areas of all monochromatic triangles -/
noncomputable def sumMonochromaticAreas (ps : PointSet) : ℝ := sorry

/-- The sum of areas of all possible triangles -/
noncomputable def sumAllAreas (ps : PointSet) : ℝ := sorry

/-- The main theorem -/
theorem monochromatic_area_bound (ps : PointSet) : 
  sumMonochromaticAreas ps ≤ (1/4) * sumAllAreas ps := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_area_bound_l4_470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_ratio_is_six_l4_441

/-- The number of balls -/
def num_balls : Nat := 24

/-- The number of bins -/
def num_bins : Nat := 6

/-- Configuration type representing the number of balls in each bin -/
def Configuration := Fin num_bins → Nat

/-- The probability of a specific configuration occurring -/
noncomputable def prob (c : Configuration) : Real := sorry

/-- The configuration (2,4,4,5,5,5) -/
def config_p : Configuration := 
  fun i => if i.val = 0 then 2 else if i.val < 3 then 4 else 5

/-- The configuration (3,3,4,4,4,4) -/
def config_q : Configuration := 
  fun i => if i.val < 2 then 3 else 4

/-- The main theorem: the ratio of probabilities is 6 -/
theorem prob_ratio_is_six : 
  prob config_p / prob config_q = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_ratio_is_six_l4_441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponent_equality_l4_448

theorem like_terms_exponent_equality (m n : ℕ) : 
  (∃ (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0), ∀ (x y : ℝ), a * x^(m-1) * y^3 = b * x^2 * y^(n+1)) → 
  n^m = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponent_equality_l4_448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_cases_2015_l4_461

/-- Given a linear decrease in disease cases from 2000 to 2020, 
    calculate the number of cases in 2015 -/
theorem disease_cases_2015 (initial_year : ℕ) (final_year : ℕ) (middle_year : ℕ)
                            (initial_cases : ℕ) (final_cases : ℕ) 
                            (h1 : initial_year = 2000)
                            (h2 : final_year = 2020)
                            (h3 : middle_year = 2015)
                            (h4 : initial_cases = 600000)
                            (h5 : final_cases = 2000)
                            (h6 : final_year > initial_year)
                            (h7 : middle_year > initial_year)
                            (h8 : final_year > middle_year) :
  ∃ (middle_cases : ℕ), 
    middle_cases = initial_cases - 
      ((initial_cases - final_cases) * (middle_year - initial_year) / (final_year - initial_year)) ∧
    middle_cases = 151500 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_cases_2015_l4_461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_function_properties_l4_463

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 9*x^2 + 26*x + 24) / (x + 3)

/-- The theorem stating the properties of the simplified function -/
theorem simplified_function_properties :
  ∃ (A B C D : ℝ),
    (∀ x, x ≠ D → f x = A*x^2 + B*x + C) ∧
    (f D = 0/0) ∧
    A + B + C + D = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_function_properties_l4_463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_random_events_l4_487

-- Define the concept of a random event
def is_random_event (event : Prop) : Bool := sorry

-- Define the three events
def event1 : Prop := sorry
def event2 : Prop := sorry
def event3 : Prop := sorry

-- Define a list of the events
def events : List Prop := [event1, event2, event3]

-- Theorem stating that the number of random events is 3
theorem number_of_random_events : 
  (events.filter is_random_event).length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_random_events_l4_487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l4_425

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between points A(-1, 2) and B(-4, 6) is 5 -/
theorem distance_A_to_B : distance (-1) 2 (-4) 6 = 5 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l4_425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_47_l4_430

theorem floor_sqrt_47 : ⌊Real.sqrt 47⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_47_l4_430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l4_446

noncomputable def f (x : ℝ) : ℝ := if x > 1 then 2 else -1

theorem solution_set (x : ℝ) : 
  x + 2 * x * f (x + 1) > 5 ↔ x ∈ Set.Ioi (-5) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l4_446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_gamma_value_l4_477

/-- Given a point Q with positive coordinates and a line from the origin to Q,
    let α, β, and γ be the angles between this line and the x-, y-, and z-axes respectively.
    If cos α = 2/7 and cos β = 1/3, then cos γ = √356/21. -/
theorem cos_gamma_value (Q : ℝ × ℝ × ℝ) (α β γ : ℝ) 
    (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
    (h_angles : α = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
                β = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
                γ = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
    (h_cos_α : Real.cos α = 2/7)
    (h_cos_β : Real.cos β = 1/3) :
  Real.cos γ = Real.sqrt 356 / 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_gamma_value_l4_477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_square_all_valid_two_digit_numbers_l4_447

/-- A function that checks if a number is a four-digit perfect square with identical first two digits and identical last two digits -/
def is_valid_four_digit_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ k : ℕ, n = k^2 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

/-- A function that reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A function that checks if a two-digit number and its reverse sum to a perfect square -/
def is_valid_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ k : ℕ, n + reverse_two_digit n = k^2

/-- Theorem stating that 7744 is the only four-digit perfect square with identical first two digits and identical last two digits -/
theorem unique_four_digit_square :
  ∀ n : ℕ, is_valid_four_digit_square n ↔ n = 7744 :=
sorry

/-- Theorem stating that the given set contains all and only two-digit numbers where the sum of the number and its reverse is a perfect square -/
theorem all_valid_two_digit_numbers :
  ∀ n : ℕ, is_valid_two_digit n ↔ n ∈ ({29, 38, 47, 56, 65, 74, 83, 92} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_square_all_valid_two_digit_numbers_l4_447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_theorem_l4_419

def is_root (k : ℕ) : Prop :=
  Real.sin (k : ℝ) = Real.sin (334 * k : ℝ)

def smallest_root : ℕ := 36

def next_smallest_root : ℕ := 40

theorem root_theorem :
  (∀ k : ℕ, k < smallest_root → ¬ is_root k) ∧
  is_root smallest_root ∧
  (∀ k : ℕ, smallest_root < k ∧ k < next_smallest_root → ¬ is_root k) ∧
  is_root next_smallest_root :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_theorem_l4_419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l4_415

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The distance between foci of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ :=
  2 * h.a

/-- The equation of asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

theorem hyperbola_asymptotes (h : Hyperbola) (h_eq : h.b^2 = 16) 
  (h_focal : focal_distance h = 10) :
  asymptote_equation h x y ↔ y = 4/5 * x ∨ y = -4/5 * x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l4_415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l4_482

noncomputable def hour_hand_degrees_per_hour : ℝ := 30
noncomputable def minute_hand_degrees_per_minute : ℝ := 6

noncomputable def hour_hand_position (hours : ℝ) (minutes : ℝ) : ℝ :=
  (hours % 12) * hour_hand_degrees_per_hour + (minutes / 60) * hour_hand_degrees_per_hour

noncomputable def minute_hand_position (minutes : ℝ) : ℝ :=
  (minutes % 60) * minute_hand_degrees_per_minute

noncomputable def angle_between_hands (hours : ℝ) (minutes : ℝ) : ℝ :=
  abs (minute_hand_position minutes - hour_hand_position hours minutes)

theorem clock_angle_at_3_20 :
  angle_between_hands 3 20 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l4_482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_probability_l4_474

/-- Represents the probability of reaching (0,0) from a given point (a,b) -/
def P (a b : ℕ) : ℚ :=
  sorry

/-- The movement rules for the particle -/
axiom movement_rule (a b : ℕ) : a > 0 → b > 0 →
  P a b = (1/3 : ℚ) * P (a-1) b + (1/3 : ℚ) * P a (b-1) + (1/3 : ℚ) * P (a-1) (b-1)

/-- Base case: Probability of reaching (0,0) from (0,0) is 1 -/
axiom base_case : P 0 0 = 1

/-- Boundary conditions: Probability of reaching (0,0) from x-axis or y-axis (except origin) is 0 -/
axiom boundary_condition (x y : ℕ) :
  (x > 0 ∧ y = 0) ∨ (x = 0 ∧ y > 0) → P x y = 0

/-- The main theorem to prove -/
theorem particle_probability : P 5 3 = 1261 / 3^8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_probability_l4_474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_l4_453

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = 2 ∧ 
  (∀ (y : ℝ), (⌊y⌋ = 2 + 50 * (frac y) ∧ 0 ≤ frac y ∧ frac y < 1) → y ≥ x) ∧
  ⌊x⌋ = 2 + 50 * (frac x) ∧ 
  0 ≤ frac x ∧ 
  frac x < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_l4_453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_wrapping_paper_l4_440

/-- The number of shirt boxes that can be wrapped by one roll of wrapping paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- The number of shirt boxes Harold needs to wrap -/
def total_shirt_boxes : ℕ := 20

/-- The number of XL boxes Harold needs to wrap -/
def total_xl_boxes : ℕ := 12

/-- The cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- The total amount Harold will spend on wrapping paper in cents -/
def total_spent : ℕ := 3200

/-- The number of XL boxes that can be wrapped by one roll of wrapping paper -/
def xl_boxes_per_roll : ℕ := total_xl_boxes / (total_spent / cost_per_roll - total_shirt_boxes / shirt_boxes_per_roll)

theorem harold_wrapping_paper :
  xl_boxes_per_roll = 3 := by
  -- Unfold the definition of xl_boxes_per_roll
  unfold xl_boxes_per_roll
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

#eval xl_boxes_per_roll

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_wrapping_paper_l4_440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_correct_circle_statements_l4_483

-- Define the statements about circle geometry as a list of booleans
def circle_statements : List Bool := [
  false, -- The radius of a circle is perpendicular to the chord
  true,  -- The circumscribed parallelogram of a circle is a rhombus
  true,  -- The inscribed parallelogram of a circle is a rectangle
  true,  -- The opposite angles of a cyclic quadrilateral are supplementary
  false, -- Two arcs of equal length are equal arcs
  false  -- Arcs corresponding to equal central angles are equal
]

-- Define a function to count the number of true statements
def count_true_statements (statements : List Bool) : Nat :=
  statements.filter id |>.length

-- Theorem: Exactly 3 of the circle statements are correct
theorem three_correct_circle_statements :
  count_true_statements circle_statements = 3 := by
  -- Unfold the definitions and evaluate
  unfold count_true_statements
  unfold circle_statements
  -- The result is true by computation
  rfl

#eval count_true_statements circle_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_correct_circle_statements_l4_483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_enough_for_six_days_l4_498

/-- Represents the amount of cat food in a large package -/
noncomputable def B : ℝ := sorry

/-- Represents the amount of cat food in a small package -/
noncomputable def S : ℝ := sorry

/-- Represents the daily consumption of cat food -/
noncomputable def daily_consumption : ℝ := sorry

/-- A large package contains more food than a small one, but less than two small packages -/
axiom package_size : S < B ∧ B < 2 * S

/-- One large and two small packages of food are enough for the cat for exactly two days -/
axiom two_day_supply : B + 2 * S = 2 * daily_consumption

/-- Theorem stating that 4 large and 4 small packages are not enough for 6 days -/
theorem not_enough_for_six_days : 4 * B + 4 * S < 6 * daily_consumption := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_enough_for_six_days_l4_498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l4_438

/-- Parabola defined by parametric equations x = 4t^2 and y = 4t -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- Point P on the parabola -/
structure PointP where
  m : ℝ
  x : ℝ := 3
  y : ℝ := m

/-- Theorem: The distance |PF| from point P to the focus F of the parabola is 4 -/
theorem distance_to_focus (para : Parabola) (p : PointP) 
  (h : p.x = para.x ∧ p.y = para.y) : 
  ∃ (f : ℝ × ℝ), ‖(p.x, p.y) - f‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l4_438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_system_l4_494

/-- The system of equations has exactly 4 sets of real solutions -/
theorem four_solutions_system (x y z : ℝ) : 
  (x + y + z = 3 * x * y) ∧ 
  (x^2 + y^2 + z^2 = 3 * x * z) ∧ 
  (x^3 + y^3 + z^3 = 3 * y * z) → 
  ∃! (S : Finset (ℝ × ℝ × ℝ)), S.card = 4 ∧ ∀ (a b c : ℝ), (a, b, c) ∈ S ↔ 
    (a + b + c = 3 * a * b) ∧ 
    (a^2 + b^2 + c^2 = 3 * a * c) ∧ 
    (a^3 + b^3 + c^3 = 3 * b * c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_system_l4_494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l4_416

theorem triangle_properties :
  ∃ x : ℝ, 
    let a := 4
    let b := x
    let c := 12 - x
    (∀ y : ℝ, (a + b > c ∧ a + c > b ∧ b + c > a) → (4 < y ∧ y < 8)) ∧
    ((a = b ∨ a = c ∨ b = c) ∧ a + b + c = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l4_416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_7_and_10_days_day_less_than_0001_from_end_first_day_less_than_0001_from_end_l4_410

-- Define the path length
def path_length : ℝ := 10

-- Define the distance covered after n days
noncomputable def distance_covered (n : ℕ) : ℝ := path_length * (1 - 1 / 2^n)

-- Theorem for the distance covered after 7 and 10 days
theorem distance_after_7_and_10_days :
  (distance_covered 7 > 9.9 ∧ distance_covered 7 < 9.922) ∧
  (distance_covered 10 > 9.99 ∧ distance_covered 10 < 9.991) := by
  sorry

-- Theorem for the day when the flea is less than 0.001 meters from the end
theorem day_less_than_0001_from_end :
  ∀ n : ℕ, n ≥ 14 → path_length - distance_covered n < 0.001 := by
  sorry

-- Theorem for the first day when the flea is less than 0.001 meters from the end
theorem first_day_less_than_0001_from_end :
  ∀ n : ℕ, n < 14 → path_length - distance_covered n ≥ 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_7_and_10_days_day_less_than_0001_from_end_first_day_less_than_0001_from_end_l4_410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_negative_l4_403

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1 / x

-- Theorem statement
theorem tangent_perpendicular_implies_a_negative 
  (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ f_derivative a x = 0) → 
  a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_negative_l4_403
