import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_and_tv_labda_volumes_l663_66399

/-- The volume of a regular icosahedron with edge length 1 -/
noncomputable def regular_icosahedron_volume : ℝ := (5 * (3 + Real.sqrt 5)) / 12

/-- The volume of a "TV-labda" derived from an icosahedron with edge length 3 -/
noncomputable def tv_labda_volume : ℝ := (125 + 43 * Real.sqrt 5) / 4

/-- Theorem stating the volumes of the regular icosahedron and the TV-labda -/
theorem icosahedron_and_tv_labda_volumes :
  (regular_icosahedron_volume = (5 * (3 + Real.sqrt 5)) / 12) ∧
  (tv_labda_volume = (125 + 43 * Real.sqrt 5) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_and_tv_labda_volumes_l663_66399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l663_66396

def M : Set ℕ := {x : ℕ | x > 0}
def N : Set ℕ := {x : ℕ | x < 2}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l663_66396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l663_66321

/-- Predicate for an isosceles trapezoid -/
def IsoscelesTrapezoid (T : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate for perpendicular diagonals -/
def PerpendicularDiagonals (T : Set (ℝ × ℝ)) : Prop := sorry

/-- Function to calculate the area of a trapezoid -/
def Area (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- Function to calculate the height of a trapezoid -/
def Height (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- An isosceles trapezoid with perpendicular diagonals and area a² has height a -/
theorem isosceles_trapezoid_height (a : ℝ) (T : Set (ℝ × ℝ)) (h : ℝ) :
  IsoscelesTrapezoid T →
  PerpendicularDiagonals T →
  Area T = a^2 →
  h = Height T →
  h = a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l663_66321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l663_66301

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3) / Real.log 10

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l663_66301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cars_meet_l663_66372

/-- Represents a car on a circular racecourse -/
structure Car where
  speed : ℝ

/-- The state of the race -/
structure RaceState where
  cars : Fin 4 → Car
  track_length : ℝ

/-- Condition that any three cars meet at some point -/
def any_three_meet (rs : RaceState) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∃ (t : ℝ), t > 0 ∧ 
      (rs.cars i).speed * t % rs.track_length = 
      (rs.cars j).speed * t % rs.track_length ∧
      (rs.cars j).speed * t % rs.track_length = 
      (rs.cars k).speed * t % rs.track_length

/-- Theorem stating that all four cars will meet again -/
theorem all_cars_meet (rs : RaceState) 
  (h : any_three_meet rs) : 
  ∃ (t : ℝ), t > 0 ∧ 
    ∀ (i j : Fin 4), 
      (rs.cars i).speed * t % rs.track_length = 
      (rs.cars j).speed * t % rs.track_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cars_meet_l663_66372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_coefficient_third_term_coefficient_correct_l663_66327

/-- The coefficient of the third term in the expansion of (√2x - 1)^5 -/
theorem third_term_coefficient : ℝ :=
  let third_term_coeff := 20 * Real.sqrt 2
  third_term_coeff

/-- The third term coefficient is correct -/
theorem third_term_coefficient_correct : third_term_coefficient = 20 * Real.sqrt 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_coefficient_third_term_coefficient_correct_l663_66327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_half_trigonometric_simplification_l663_66336

theorem trigonometric_expression_equals_half : 
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) - 
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1/2 := by
  sorry

theorem trigonometric_simplification :
  ∀ α : Real,
  (Real.sin (α + π) ^ 2 * Real.cos (π + α)) / 
  (Real.tan (-α - 2*π) * Real.tan (π + α) * Real.cos (-α - π) ^ 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_half_trigonometric_simplification_l663_66336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validMenuCount_eq_729_l663_66339

/-- Represents the possible desserts --/
inductive Dessert
| Cake
| Pie
| IceCream
| Pudding

/-- Represents the days of the week --/
inductive Day
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to get the next day of the week --/
def nextDay : Day → Day
| Day.Sunday => Day.Monday
| Day.Monday => Day.Tuesday
| Day.Tuesday => Day.Wednesday
| Day.Wednesday => Day.Thursday
| Day.Thursday => Day.Friday
| Day.Friday => Day.Saturday
| Day.Saturday => Day.Sunday

/-- A valid dessert menu for a week --/
def WeekMenu := Day → Dessert

/-- Checks if a dessert menu is valid according to the rules --/
def isValidMenu (menu : WeekMenu) : Prop :=
  (∀ d : Day, d ≠ Day.Saturday → menu d ≠ menu (nextDay d)) ∧
  (menu Day.Friday = Dessert.Cake)

/-- The number of valid dessert menus for a week --/
def validMenuCount : ℕ := sorry

/-- The main theorem: proving that the number of valid dessert menus is 729 --/
theorem validMenuCount_eq_729 : validMenuCount = 729 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validMenuCount_eq_729_l663_66339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_large_class_l663_66382

/-- Represents a class in the kindergarten -/
structure KClass where
  size : ℕ

/-- Represents the kindergarten with three classes -/
structure Kindergarten where
  small : KClass
  medium : KClass
  large : KClass

/-- Calculates the total number of students in the kindergarten -/
def totalStudents (k : Kindergarten) : ℕ :=
  k.small.size + k.medium.size + k.large.size

/-- Calculates the proportion of students in the large class -/
def largeProportion (k : Kindergarten) : ℚ :=
  k.large.size / totalStudents k

/-- Theorem: In a stratified sampling of 50 students from a kindergarten
    with 90 small, 90 medium, and 120 large class students,
    the number of students sampled from the large class is 20 -/
theorem stratified_sample_large_class
  (k : Kindergarten)
  (h1 : k.small.size = 90)
  (h2 : k.medium.size = 90)
  (h3 : k.large.size = 120)
  (sample_size : ℕ)
  (h4 : sample_size = 50) :
  ⌊(sample_size : ℚ) * largeProportion k⌋ = 20 := by
  sorry

#check stratified_sample_large_class

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_large_class_l663_66382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_product_l663_66346

theorem square_root_product (a : ℝ) : 
  (a ^ 2 = 9 * 16 ∨ 16 * a = 9 ^ 2 ∨ 9 * a = 16 ^ 2) ↔ 
  a ∈ ({-12, 12, 81/16, 256/9} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_product_l663_66346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_sqrt_two_l663_66348

theorem sin_minus_cos_sqrt_two (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi → Real.sin x - Real.cos x = Real.sqrt 2 → x = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_sqrt_two_l663_66348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxiFare_correct_fareDistance_correct_taxiFare_fareDistance_inverse_l663_66319

-- Define the taxi fare function
noncomputable def taxiFare (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 3 then 10
  else if 3 < x ∧ x ≤ 18 then x + 7
  else 2*x - 11

-- Define the inverse function (distance given fare)
noncomputable def fareDistance (y : ℝ) : ℝ :=
  if 10 < y ∧ y ≤ 25 then 3 + (y - 10)
  else 18 + (y - 25)/2

-- Theorem: The taxiFare function is correct
theorem taxiFare_correct (x : ℝ) :
  taxiFare x = if 0 < x ∧ x ≤ 3 then 10
               else if 3 < x ∧ x ≤ 18 then x + 7
               else 2*x - 11 := by sorry

-- Theorem: The fareDistance function is correct
theorem fareDistance_correct (y : ℝ) :
  fareDistance y = if 10 < y ∧ y ≤ 25 then 3 + (y - 10)
                   else 18 + (y - 25)/2 := by sorry

-- Theorem: taxiFare and fareDistance are inverse functions
theorem taxiFare_fareDistance_inverse (x y : ℝ) :
  (0 < x ∧ taxiFare x = y) → fareDistance y = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxiFare_correct_fareDistance_correct_taxiFare_fareDistance_inverse_l663_66319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l663_66384

noncomputable def f (x : ℝ) : ℝ := min (2 - x) x

theorem max_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≤ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l663_66384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l663_66385

-- Define the triangle ABC
def triangle_ABC (a b C : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 3 ∧ C = 30 * Real.pi / 180

-- Define the area of a triangle
noncomputable def triangle_area (a b C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin C

-- Theorem statement
theorem area_of_triangle_ABC :
  ∀ a b C, triangle_ABC a b C →
  triangle_area a b C = Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l663_66385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l663_66304

noncomputable section

-- Define the curve C₂ in polar coordinates
def C₂ (θ : Real) : Real := 6 / Real.sqrt (9 - 3 * Real.sin θ ^ 2)

-- Define points A and B in polar coordinates
def A : Real × Real := (2 * Real.sqrt 3, -Real.pi / 6)
def B : Real × Real := (2 * Real.sqrt 3, 5 * Real.pi / 6)

-- Define the function to be maximized
def f (θ : Real) : Real :=
  let (ρ, φ) := (C₂ θ, θ)
  let (ρ_A, φ_A) := A
  let (ρ_B, φ_B) := B
  (ρ * Real.cos φ - ρ_A * Real.cos φ_A)^2 + (ρ * Real.sin φ - ρ_A * Real.sin φ_A)^2 +
  (ρ * Real.cos φ - ρ_B * Real.cos φ_B)^2 + (ρ * Real.sin φ - ρ_B * Real.sin φ_B)^2

-- Theorem statement
theorem max_sum_distances :
  ∃ (M : Real), M = 36 ∧ ∀ θ, f θ ≤ M := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l663_66304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_in_sequence_l663_66342

def is_valid_pair (m n : ℕ+) : Prop :=
  ∃ k : ℕ+, k * (m * n - 1) = m^2 + n^2

noncomputable def sequence_term (n : ℕ) : ℕ+ × ℕ+ :=
  match n with
  | 0 => (1, 2)
  | 1 => (1, 3)
  | 2 => (2, 1)
  | 3 => (3, 1)
  | n + 4 =>
    let (a, b) := sequence_term n
    (5 * a - b, a)

theorem valid_pairs_in_sequence :
  ∀ m n : ℕ+, is_valid_pair m n →
    ∃ k : ℕ, sequence_term k = (m, n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_in_sequence_l663_66342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_one_l663_66364

/-- The circle with equation x^2 + y^2 - 4x + 2y = 0 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

/-- The tangent line with equation x - 2y + 1 = 0 -/
def my_tangent_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- Theorem: The tangent line to the given circle at the point (1,1) has the equation x - 2y + 1 = 0 -/
theorem tangent_line_at_point_one_one :
  (my_circle 1 1) →
  ∃ (x y : ℝ), (x = 1 ∧ y = 1) ∧
  (my_tangent_line x y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_one_l663_66364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l663_66360

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2) * Real.cos x

-- State the theorem about the smallest positive period of f(x)
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l663_66360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l663_66391

-- Define the three functions
noncomputable def f1 (x : ℝ) : ℝ := 2 * x + 3
noncomputable def f2 (x : ℝ) : ℝ := (4 * x^2 + 12 * x + 9) / (2 * x + 3)
noncomputable def f3 (x : ℝ) : ℝ := (4 * x^2 + 12 * x + 9) / (2 * x + 3)

-- Define what it means for two functions to have the same graph
def same_graph (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (∃ y : ℝ, f x = y ∧ g x = y) ∨ (¬∃ y : ℝ, f x = y) ∧ (¬∃ y : ℝ, g x = y)

-- Theorem stating that all three functions have different graphs
theorem different_graphs : 
  ¬(same_graph f1 f2) ∧ ¬(same_graph f1 f3) ∧ ¬(same_graph f2 f3) := by
  sorry

-- Additional lemmas to support the main theorem
lemma f1_continuous : Continuous f1 := by
  sorry

lemma f2_discontinuous : ¬Continuous f2 := by
  sorry

lemma f3_vertical_line : ∃ x : ℝ, ∀ y : ℝ, f3 x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l663_66391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_l663_66392

/-- Regular octagon with apothem 3 -/
structure RegularOctagon where
  apothem : ℝ
  apothem_eq : apothem = 3

/-- Midpoint of a side of the octagon -/
def octagonMidpoint (R : RegularOctagon) (i : Fin 8) : ℝ × ℝ := sorry

/-- Hexagon formed by connecting every other midpoint -/
def midpointHexagon (R : RegularOctagon) : Set (ℝ × ℝ) :=
  {p | ∃ i : Fin 6, p = octagonMidpoint R ⟨2*i, sorry⟩}

/-- Area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem midpoint_hexagon_area (R : RegularOctagon) :
  area (midpointHexagon R) = 27 * Real.sqrt 3 * (3 - 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_l663_66392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_theorem_l663_66308

/-- Represents a car journey with two speed phases -/
structure CarJourney where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  reducedSpeed : ℝ

/-- Calculates the time spent at the initial speed given a car journey -/
noncomputable def timeAtInitialSpeed (j : CarJourney) : ℝ :=
  (j.totalDistance - j.reducedSpeed * j.totalTime) / (j.initialSpeed - j.reducedSpeed)

/-- Theorem stating the time spent at initial speed for the given journey -/
theorem car_journey_theorem (j : CarJourney) 
  (h1 : j.totalDistance = 156)
  (h2 : j.totalTime = 7)
  (h3 : j.initialSpeed = 30)
  (h4 : j.reducedSpeed = 18) :
  timeAtInitialSpeed j = 5/2 := by
  sorry

#check car_journey_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_theorem_l663_66308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_l663_66341

/-- An equilateral triangle inscribed in an ellipse -/
structure InscribedTriangle where
  /-- The ellipse equation: x^2 + 3y^2 = 3 -/
  ellipse : ℝ → ℝ → Prop
  /-- One vertex of the triangle is at (0, √3/3) -/
  vertex : ℝ × ℝ
  /-- One altitude of the triangle is along the y-axis -/
  altitude_on_y_axis : Prop
  /-- The triangle is equilateral -/
  is_equilateral : Prop

/-- The square of the length of each side of the triangle -/
noncomputable def side_length_squared (t : InscribedTriangle) : ℝ := (31 + 24 * Real.sqrt 141) / 50

/-- The main theorem -/
theorem inscribed_triangle_side_length 
  (t : InscribedTriangle) 
  (h1 : t.ellipse = fun x y => x^2 + 3*y^2 = 3)
  (h2 : t.vertex = (0, Real.sqrt 3 / 3))
  (h3 : t.altitude_on_y_axis)
  (h4 : t.is_equilateral) :
  side_length_squared t = (31 + 24 * Real.sqrt 141) / 50 := by
  sorry

#check inscribed_triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_l663_66341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l663_66387

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_y_axis_foci : a > b  -- Foci on y-axis implies a > b

/-- The value of m in the transformed ellipse equation x² + my² = 1 -/
noncomputable def m (e : Ellipse) : ℝ := (e.b / e.a) ^ 2

/-- Theorem stating that m = 1/4 when the major axis is twice the minor axis -/
theorem ellipse_m_value (e : Ellipse) 
  (h_axis_ratio : e.a = 2 * e.b) : m e = 1/4 := by
  sorry

#check ellipse_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l663_66387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_function_max_value_l663_66311

/-- The maximum value of the function y = cos²α + sinα + 3 is 17/4. -/
theorem cos_sin_function_max_value :
  (∀ α : ℝ, (Real.cos α) ^ 2 + Real.sin α + 3 ≤ 17 / 4) ∧
  (∃ α : ℝ, (Real.cos α) ^ 2 + Real.sin α + 3 = 17 / 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_function_max_value_l663_66311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l663_66332

theorem quadratic_function_proof (α β : ℂ) : 
  α^2 - α + 1 = 0 → 
  β^2 - β + 1 = 0 → 
  α ≠ β →
  (let f : ℂ → ℂ := λ x ↦ x^2 - x + 1
   f α = β ∧ f β = α ∧ f 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l663_66332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_divides_angle_bisector_C_l663_66312

-- Define the triangle
noncomputable def triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 15 ∧ BC = 12 ∧ AC = 18

-- Define the incenter
noncomputable def incenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  ((a * A.1 + b * B.1 + c * C.1) / (a + b + c),
   (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

-- Define the angle bisector of angle C
noncomputable def angle_bisector_C (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    P.1 = C.1 + t * (A.1 * BC / (AC + BC) + B.1 * AC / (AC + BC) - C.1) ∧
    P.2 = C.2 + t * (A.2 * BC / (AC + BC) + B.2 * AC / (AC + BC) - C.2)}

-- The theorem to prove
theorem incenter_divides_angle_bisector_C (A B C : ℝ × ℝ) :
  triangle A B C →
  ∃ O : ℝ × ℝ, O ∈ angle_bisector_C A B C ∧
    O = incenter A B C ∧
    let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2)) /
    (Real.sqrt ((A.1 * BC / (AC + BC) + B.1 * AC / (AC + BC) - O.1)^2 +
                (A.2 * BC / (AC + BC) + B.2 * AC / (AC + BC) - O.2)^2)) = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_divides_angle_bisector_C_l663_66312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_point_l663_66349

theorem tan_double_angle_specific_point :
  ∀ α : ℝ,
  (∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ Real.tan α = y / x) →
  Real.tan (2 * α) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_point_l663_66349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_passes_through_mean_l663_66356

noncomputable def data : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

noncomputable def mean_x : ℝ := (List.sum (List.map Prod.fst data)) / data.length
noncomputable def mean_y : ℝ := (List.sum (List.map Prod.snd data)) / data.length

theorem linear_regression_passes_through_mean :
  mean_y = mean_x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_passes_through_mean_l663_66356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_of_pentagon_l663_66323

/-- The side length of the regular pentagon -/
noncomputable def side_length : ℝ := 1

/-- The area of a regular pentagon with side length 1 -/
noncomputable def pentagon_area : ℝ := (5 * Real.sqrt 3) / 4

/-- The area removed around each vertex -/
noncomputable def removed_area_per_vertex : ℝ := Real.pi / 5

/-- The number of vertices in a pentagon -/
def num_vertices : ℕ := 5

/-- The theorem stating the area of the remaining part after removal -/
theorem remaining_area_of_pentagon : 
  let total_area := pentagon_area
  let total_removed_area := (num_vertices : ℝ) * removed_area_per_vertex
  total_area - total_removed_area = (5 * Real.sqrt 3) / 4 - Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_of_pentagon_l663_66323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_gap_after_speed_reduction_l663_66300

/-- Calculates the new gap between two cars after passing a speed limit sign -/
theorem new_gap_after_speed_reduction (v₁ v₂ : ℝ) (initial_gap : ℝ) : 
  v₁ = 80 * (1000 / 3600) →  -- Initial speed in m/s
  v₂ = 60 * (1000 / 3600) →  -- Reduced speed in m/s
  initial_gap = 10 →         -- Initial gap in meters
  v₂ * (initial_gap / v₁) = 7.5 := by
    sorry

#eval Float.toString (60 * (1000 / 3600))  -- Should output approximately 16.67
#eval Float.toString (80 * (1000 / 3600))  -- Should output approximately 22.22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_gap_after_speed_reduction_l663_66300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_Q_l663_66351

noncomputable section

-- Define the unit square ABCD
def unit_square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define point F on diagonal BC
def F : ℝ × ℝ :=
  (1 / Real.sqrt 2, 1 / Real.sqrt 2)

-- Define triangle ABF
def triangle_ABF : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 / Real.sqrt 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ p.1}

-- Define the strip
def strip : Set (ℝ × ℝ) :=
  {p | 1/4 ≤ p.2 ∧ p.2 ≤ 3/4 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1}

-- Define region Q
def region_Q : Set (ℝ × ℝ) :=
  strip \ triangle_ABF

-- Theorem statement
theorem area_of_region_Q :
  MeasureTheory.volume region_Q = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_Q_l663_66351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l663_66306

/-- The distance between two parallel lines given by their equations -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the parallel lines 3x - √7y + 2 = 0 and 6x - 2√7y + 3 = 0 is 1/8 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines 3 (-Real.sqrt 7) 2 (3/2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l663_66306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l663_66380

theorem complex_power_sum : Complex.I^8 + Complex.I^20 + (Complex.I^40)⁻¹ + 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l663_66380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l663_66357

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1
  h_not_vertex : x ≠ E.a ∧ x ≠ -E.a

/-- The statement of the problem -/
theorem ellipse_problem (E : Ellipse) (P : PointOnEllipse E) 
  (h_slopes : (P.y / (P.x + E.a)) * (P.y / (P.x - E.a)) = -1/4) :
  (∃ (e : ℝ), e^2 = 3/4 ∧ e = Real.sqrt (E.a^2 - E.b^2) / E.a) ∧
  (∃ (lambda : ℝ), lambda = 0 ∨ lambda = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l663_66357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_weight_l663_66340

theorem average_book_weight (sandy_books : ℕ) (sandy_weight : ℝ)
                             (benny_books : ℕ) (benny_weight : ℝ)
                             (tim_books : ℕ) (tim_weight : ℝ)
                             (lily_books : ℕ) (lily_weight : ℝ) :
  sandy_books = 10 ∧ sandy_weight = 1.5 ∧
  benny_books = 24 ∧ benny_weight = 1.2 ∧
  tim_books = 33 ∧ tim_weight = 1.8 ∧
  lily_books = 15 ∧ lily_weight = 1.3 →
  ∃ ε > 0, |((sandy_books * sandy_weight +
              benny_books * benny_weight +
              tim_books * tim_weight +
              lily_books * lily_weight) /
             (sandy_books + benny_books + tim_books + lily_books)) - 1.497| < ε :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_weight_l663_66340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l663_66368

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1/m)*x + 1

theorem f_inequality (m : ℝ) (hm : m > 0) :
  (m = 2 → ∀ x, f m x ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (0 < m ∧ m < 1 → ∀ x, f m x ≥ 0 ↔ x ≤ m ∨ x ≥ 1/m) ∧
  (m = 1 → ∀ x, f m x ≥ 0) ∧
  (m > 1 → ∀ x, f m x ≥ 0 ↔ x ≥ m ∨ x ≤ 1/m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l663_66368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l663_66389

theorem trig_identity (α : ℝ) 
  (h1 : Real.cos (65 * Real.pi / 180 + α) = 2/3)
  (h2 : Real.pi < α ∧ α < 3*Real.pi/2) :
  Real.cos (115 * Real.pi / 180 - α) + Real.sin (α - 115 * Real.pi / 180) = (Real.sqrt 5 - 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l663_66389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_sundae_cost_is_ten_l663_66354

/-- Calculates the cost of the fourth sundae given the prices of three sundaes, tip percentage, and final bill -/
noncomputable def fourth_sundae_cost (sundae1 sundae2 sundae3 : ℚ) (tip_percent : ℚ) (final_bill : ℚ) : ℚ :=
  let known_sundaes_cost := sundae1 + sundae2 + sundae3
  let before_tip := final_bill / (1 + tip_percent)
  before_tip - known_sundaes_cost

theorem fourth_sundae_cost_is_ten :
  fourth_sundae_cost 9 (15/2) (17/2) (1/5) 42 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_sundae_cost_is_ten_l663_66354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l663_66398

theorem log_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  Real.log (Real.log x) < (Real.log x)^2 ∧ (Real.log x)^2 < Real.log (x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l663_66398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_quadrilateral_existence_l663_66376

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point on a circle -/
def PointOnCircle (K : Circle) := { p : ℝ × ℝ // (p.1 - K.center.1)^2 + (p.2 - K.center.2)^2 = K.radius^2 }

/-- Extract the coordinates from a PointOnCircle -/
def getPoint (K : Circle) (p : PointOnCircle K) : ℝ × ℝ := p.val

/-- A quadrilateral is tangential if and only if the sum of one pair of opposite sides equals the sum of the other pair -/
def IsTangential (A B C D : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B + d C D = d B C + d D A

/-- Main theorem: For any circle and three distinct points on it, there exists a fourth point such that the quadrilateral formed is tangential -/
theorem tangential_quadrilateral_existence (K : Circle) (A B C : PointOnCircle K)
  (hAB : A ≠ B) (hBC : B ≠ C) (hCA : C ≠ A) :
  ∃ D : PointOnCircle K, IsTangential (getPoint K A) (getPoint K B) (getPoint K C) (getPoint K D) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_quadrilateral_existence_l663_66376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_line_l663_66320

/-- The curve defined by θ = π/4 in polar coordinates is a line -/
theorem polar_curve_is_line : ∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), (∃ (r : ℝ), x = r * Real.cos (π/4) ∧ y = r * Real.sin (π/4)) ↔ a*x + b*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_line_l663_66320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_diagonal_and_perimeter_l663_66388

/-- Represents a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The diagonal of a rectangle --/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.length^2 + r.width^2)

/-- The perimeter of a rectangle --/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- The area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem: A rectangle with diagonal 17 and perimeter 46 has an area of 120 --/
theorem rectangle_area_with_diagonal_and_perimeter :
  ∃ (r : Rectangle), r.diagonal = 17 ∧ r.perimeter = 46 ∧ r.area = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_diagonal_and_perimeter_l663_66388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_substance_ratio_of_hydrogen_content_ratio_is_6_4_3_l663_66343

/-- Given H₂O, NH₃, and CH₄ each contain 1 mol of hydrogen atoms,
    prove that the ratio of their amounts of substance is 6:4:3 -/
theorem substance_ratio_of_hydrogen_content
  (x y z : ℚ) -- Amounts of substance for H₂O, NH₃, and CH₄ respectively
  (h1 : 2 * x = 1) -- H₂O contains 1 mol of hydrogen atoms
  (h2 : 3 * y = 1) -- NH₃ contains 1 mol of hydrogen atoms
  (h3 : 4 * z = 1) -- CH₄ contains 1 mol of hydrogen atoms
  : (x, y, z) = (1/2, 1/3, 1/4) := by
  sorry

/-- The ratio of the amounts of substance is 6:4:3 -/
theorem ratio_is_6_4_3
  (x y z : ℚ)
  (h : (x, y, z) = (1/2, 1/3, 1/4))
  : (6 : ℚ) * x = 4 * y ∧ 4 * y = 3 * z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_substance_ratio_of_hydrogen_content_ratio_is_6_4_3_l663_66343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_binomial_l663_66386

/-- The number of items randomly selected for quality inspection -/
def n : ℕ := 5

/-- The probability of a product being qualified (1 - defect rate) -/
def p : ℝ := 0.95

/-- X represents the number of qualified products among the n items -/
def X : ℕ → ℝ := sorry

/-- The expected value of X -/
def E_X : ℝ := n * p

theorem expected_value_binomial :
  E_X = 4.75 := by
  unfold E_X n p
  norm_num

#eval E_X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_binomial_l663_66386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_calculation_l663_66374

/-- Represents the number of pages in a book -/
def BookPages (n : ℕ) : Prop := True

/-- Represents the daily reading rate in pages per day -/
def DailyReadingRate (n : ℕ) : Prop := True

/-- Represents the number of days Bill has been reading -/
def DaysReading (n : ℕ) : Prop := True

/-- Represents the fraction of the book covered -/
def FractionCovered (q : ℚ) : Prop := True

theorem book_pages_calculation 
  (daily_rate : DailyReadingRate 8)
  (days_read : DaysReading 12)
  (fraction : FractionCovered (2/3))
  : BookPages 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_calculation_l663_66374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l663_66325

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the directrix
def directrix (y₀ : ℝ) (x y : ℝ) : Prop := y = y₀

-- Define the focus of a parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

theorem parabola_focus_coordinates :
  ∀ p : ℝ,
  p > 0 →
  directrix (-2) (-1) (-2) →
  parabola p 0 2 →
  focus p = (0, 2) := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l663_66325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l663_66330

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (1/x - 1)

-- State the theorem
theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l663_66330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_theorem_l663_66390

noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

def line_l (x y : ℝ) : Prop :=
  x = 3/2 ∨ 8*x + 6*y = 39

noncomputable def angle_between (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

theorem circle_and_line_theorem :
  -- Circle C passes through O(0,0), A(1,1), and B(4,2)
  circle_C 0 0 ∧ circle_C 1 1 ∧ circle_C 4 2 →
  -- Line l passes through (3/2, 9/2)
  line_l (3/2) (9/2) →
  -- There exist points M and N on both C and l
  ∃ (mx my nx ny : ℝ),
    circle_C mx my ∧ circle_C nx ny ∧
    line_l mx my ∧ line_l nx ny ∧
    -- Angle MCN = 120°
    angle_between (mx - 4, my + 3) (nx - 4, ny + 3) = 2*Real.pi/3 →
  -- Then the equations of C and l are as stated
  (∀ x y, circle_C x y ↔ x^2 + y^2 - 8*x + 6*y = 0) ∧
  (∀ x y, line_l x y ↔ (x = 3/2 ∨ 8*x + 6*y = 39)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_theorem_l663_66390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_tangent_ellipse_l663_66345

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxesEllipse where
  center : ℝ × ℝ
  semimajor : ℝ
  semiminor : ℝ

/-- The ellipse is tangent to the x-axis at (6, 0) and to the y-axis at (0, 3) -/
def tangent_ellipse : ParallelAxesEllipse :=
  { center := (6, 3)
    semimajor := 6
    semiminor := 3 }

/-- The distance between the foci of an ellipse -/
noncomputable def focal_distance (e : ParallelAxesEllipse) : ℝ :=
  2 * Real.sqrt (e.semimajor ^ 2 - e.semiminor ^ 2)

/-- Theorem: The distance between the foci of the given ellipse is 6√3 -/
theorem focal_distance_of_tangent_ellipse :
  focal_distance tangent_ellipse = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_tangent_ellipse_l663_66345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l663_66329

noncomputable def f (x : ℝ) := Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1))

theorem sqrt_sum_equality (x : ℝ) :
  (f x = Real.sqrt 2 ↔ 1/2 ≤ x ∧ x ≤ 1) ∧
  (¬ ∃ x, f x = 1) ∧
  (f x = 2 ↔ x = 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l663_66329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_theorem_l663_66331

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle_Y : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define distances
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define angles
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_point_theorem (X Y Z : ℝ × ℝ) (h : Triangle X Y Z) :
  distance P X = 13 ∧
  distance P Y = 5 ∧
  angle X P Y = angle Y P Z ∧
  angle Y P Z = angle P Z X →
  distance P Z = 6.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_theorem_l663_66331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_exponential_l663_66316

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Define the set A (range of f)
def A (m : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Icc 1 2, f m x = y}

-- Define the set B (range of g)
def B (k : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Icc 1 2, g k x = y}

theorem power_function_and_exponential (m k : ℝ) : 
  (∀ x > 0, Monotone (f m)) →  -- f is monotonically increasing on (0, +∞)
  (B k ⊆ A m) →               -- B is a subset of A (necessary condition)
  (m = 0 ∧ 0 ≤ k ∧ k ≤ 1) :=  -- Conclusion to be proved
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_exponential_l663_66316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_zinc_ratio_approx_l663_66337

/-- The ratio of copper to zinc in an alloy -/
noncomputable def alloy_ratio (copper_mass : ℝ) (zinc_mass : ℝ) : ℝ :=
  copper_mass / zinc_mass

/-- Theorem stating that the ratio of copper to zinc in the given alloy is approximately 2.25 -/
theorem copper_zinc_ratio_approx :
  let copper_mass : ℝ := 24
  let zinc_mass : ℝ := 10.67
  abs (alloy_ratio copper_mass zinc_mass - 2.25) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_zinc_ratio_approx_l663_66337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l663_66322

/-- 
Given an angle α in the plane rectangular coordinate system xOy, where:
- The initial side of α is the non-negative half-axis of the x-axis
- The terminal side of α passes through the point P(-1,2)
Prove that cos(π - α) = √5/5
-/
theorem cos_pi_minus_alpha (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (-1, 2) ∧ 
    (∃ r : ℝ, r > 0 ∧ (r * Real.cos α = -1 ∧ r * Real.sin α = 2))) → 
  Real.cos (π - α) = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l663_66322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_amplitude_l663_66379

/-- The magnitude of the amplitude in a sine function given its graph --/
theorem sine_function_amplitude
  (a b c : ℝ) -- a, b, and c are real numbers
  (ha : a > 0) -- a is positive
  (hb : b > 0) -- b is positive
  (hmax : ∀ x, a * Real.sin (b * x + c) ≤ 3) -- maximum y-value is 3
  (hreach : ∃ x, a * Real.sin (b * x + c) = 3) -- the maximum is actually reached
  : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_amplitude_l663_66379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_identity_l663_66328

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2/x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := x^2 + 2/x^2

-- Define α as the slope of the tangent line at x = 1
noncomputable def α : ℝ := f' 1

open Real

-- State the theorem
theorem tangent_slope_identity : 
  (sin α * cos (2 * α)) / (sin α + cos α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_identity_l663_66328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l663_66317

theorem exponent_problem (h1 : 5568 / 87 = 64) (h2 : 72 * 2 = 144) :
  ∃ x : ℚ, (64 : ℝ) ^ (x : ℝ) + 12 = 16 ∧ x = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l663_66317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l663_66395

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 5*x - 14
  {x : ℝ | f x ≥ 0} = Set.Iic (-2) ∪ Set.Ici 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l663_66395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_exists_l663_66333

/-- Represents a 2011 x 2011 grid filled with numbers from 1 to 2011^2 -/
def Grid := Fin 2011 → Fin 2011 → Fin (2011^2)

/-- Checks if a grid is valid (increasing across rows and down columns) -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j₁ j₂, j₁ < j₂ → g i j₁ < g i j₂) ∧
  (∀ i₁ i₂ j, i₁ < i₂ → g i₁ j < g i₂ j)

/-- Checks if two grids differ by at most one swap of adjacent elements -/
def differs_by_one_swap (g₁ g₂ : Grid) : Prop :=
  ∃ i j, (g₁ i j = g₂ i (j+1) ∧ g₁ i (j+1) = g₂ i j) ∧
         (∀ i' j', (i' ≠ i ∨ (j' ≠ j ∧ j' ≠ j+1)) → g₁ i' j' = g₂ i' j')

/-- The main theorem stating that an optimal strategy exists -/
theorem optimal_strategy_exists :
  ∀ (grids : Fin 2010 → Grid),
    (∀ k, is_valid_grid (grids k)) →
    (∀ j k, j ≠ k → grids j ≠ grids k) →
    ∃ (optimal : Grid),
      is_valid_grid optimal ∧
      ∀ k, differs_by_one_swap optimal (grids k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_exists_l663_66333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_ratio_equality_l663_66347

/-- Given three equal segments on a line and a point not on the line, 
    with lines drawn to the endpoints and an arbitrary secant, 
    prove the equality of certain ratios. -/
theorem segment_ratio_equality 
  (A B C D S A₁ B₁ C₁ D₁ : EuclideanSpace ℝ (Fin 2)) 
  (h_collinear : Collinear ℝ ({A, B, C, D} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_equal_segments : dist A B = dist B C ∧ dist B C = dist C D)
  (h_S_not_collinear : ¬ Collinear ℝ ({A, B, C, D, S} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_secant : Collinear ℝ ({A₁, B₁, C₁, D₁} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_A₁_on_SA : Collinear ℝ ({S, A, A₁} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_B₁_on_SB : Collinear ℝ ({S, B, B₁} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_C₁_on_SC : Collinear ℝ ({S, C, C₁} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_D₁_on_SD : Collinear ℝ ({S, D, D₁} : Set (EuclideanSpace ℝ (Fin 2))))
  (h_S_not_on_secant : ¬ Collinear ℝ ({S, A₁, B₁, C₁, D₁} : Set (EuclideanSpace ℝ (Fin 2)))) :
  (dist A A₁ / dist A₁ S) + (dist D D₁ / dist D₁ S) = 
  (dist B B₁ / dist B₁ S) + (dist C C₁ / dist C₁ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_ratio_equality_l663_66347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cubic_polynomials_with_roots_equal_coefficients_l663_66361

theorem no_cubic_polynomials_with_roots_equal_coefficients :
  ¬∃ (a b c d : ℝ), a ≠ 0 ∧
  (let p : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d;
   let roots := {r : ℝ | p r = 0};
   let coeffs := {a, b, c, d};
   roots = coeffs) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cubic_polynomials_with_roots_equal_coefficients_l663_66361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2x_plus_exp_equals_e_l663_66352

open Real MeasureTheory

theorem integral_2x_plus_exp_equals_e :
  ∫ x in Set.Icc 0 1, (2 * x + exp x) = exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2x_plus_exp_equals_e_l663_66352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_interval_alpha_value_l663_66303

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_monotonic_interval (α : ℝ) :
  (f α 2 = 4) →
  (∀ x y, x ∈ Set.Ici (0 : ℝ) → y ∈ Set.Ici (0 : ℝ) → x ≤ y → f α x ≤ f α y) ∧
  (∀ a, a < 0 → ∃ b, a < b ∧ f α a > f α b) :=
by
  -- Proof goes here
  sorry

-- Additional theorem to show that α = 2
theorem alpha_value :
  ∃ α : ℝ, f α 2 = 4 ∧ α = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_interval_alpha_value_l663_66303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_range_l663_66313

-- Define the triangle operation
noncomputable def triangle (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

-- Define the main theorem
theorem triangle_function_range :
  ∀ k : ℝ, k > 0 → 
  (triangle 1 k = 3) →
  ∀ x : ℝ, x ≥ 0 →
  ∃ y : ℝ, y ≥ 1 ∧ triangle k x = y ∧
  ∀ z : ℝ, z ≥ 1 → ∃ w : ℝ, w ≥ 0 ∧ triangle k w = z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_range_l663_66313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_eq_solution_l663_66393

theorem arctan_tan_eq_solution :
  ∃! x : ℝ, x ∈ Set.Icc (-2 * Real.pi / 3) (2 * Real.pi / 3) ∧ Real.arctan (Real.tan x) = 3 * x / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_eq_solution_l663_66393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angles_in_regular_pentagon_l663_66375

/-- Definition of a regular pentagon -/
structure RegularPentagon where
  sides : ℕ
  angles_equal : Bool
  sides_eq_five : sides = 5

/-- Sum of interior angles in a polygon -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Definition of an obtuse angle -/
def is_obtuse (angle : ℝ) : Prop := angle > 90

/-- Theorem: The number of obtuse angles in a regular pentagon is 5 -/
theorem obtuse_angles_in_regular_pentagon (p : RegularPentagon) : 
  (sum_interior_angles p.sides / p.sides > 90) ∧ 
  (∃ (n : ℕ), n = 5 ∧ n = p.sides) :=
by
  sorry

#check obtuse_angles_in_regular_pentagon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angles_in_regular_pentagon_l663_66375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_and_equation_solution_l663_66334

theorem calculation_and_equation_solution :
  (((3 - Real.pi) ^ 0) - (2 * Real.cos (30 * π / 180)) + |1 - Real.sqrt 3| + (1 / 2)⁻¹ = 2) ∧
  (∀ x : ℝ, x^2 - 2*x - 9 = 0 ↔ x = 1 + Real.sqrt 10 ∨ x = 1 - Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_and_equation_solution_l663_66334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_play_area_l663_66302

/-- The area available to a llama tied to the corner of a rectangular shed --/
noncomputable def llama_play_area (shed_width : ℝ) (shed_length : ℝ) (leash_length : ℝ) : ℝ :=
  let full_circle_area := Real.pi * leash_length^2
  let arc_area := (3/4) * full_circle_area
  let additional_sector_radius := leash_length - shed_width
  let additional_sector_area := (1/4) * Real.pi * additional_sector_radius^2
  arc_area + additional_sector_area

/-- Theorem stating that the area available to Chuck the llama is 7π square meters --/
theorem chuck_play_area :
  llama_play_area 2 3 3 = 7 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_play_area_l663_66302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l663_66383

-- Define the ellipse
def myEllipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y = 0

-- Define the line
def myLine (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 2) + 1

-- Define the theorem
theorem ellipse_and_line_equations :
  ∀ (a b : ℝ) (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) (k : ℝ),
  a > b ∧ b > 0 ∧
  myEllipse a b x y ∧
  (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0 ∧
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (4/3)^2 ∧
  (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (14/3)^2 ∧
  myCircle (-2) 1 ∧
  myLine k x y ∧
  ∃ (A B : ℝ × ℝ), myEllipse a b A.1 A.2 ∧ myEllipse a b B.1 B.2 ∧
                    myLine k A.1 A.2 ∧ myLine k B.1 B.2 ∧
                    A.1 + B.1 = -4 ∧ A.2 + B.2 = 2 →
  myEllipse 3 2 x y ∧ (8*x - 9*y + 25 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l663_66383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_APB_obtuse_l663_66362

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The specific pentagon in the problem -/
noncomputable def problemPentagon : Pentagon := {
  A := (0, 2)
  B := (4, 0)
  C := (2 * Real.pi + 1, 0)
  D := (2 * Real.pi + 1, 4)
  E := (0, 4)
}

/-- Function to calculate the area of a pentagon -/
noncomputable def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Function to calculate the area of a semicircle -/
noncomputable def semicircleArea (radius : ℝ) : ℝ := sorry

/-- Function to check if a point is inside a semicircle -/
def isInsideSemicircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop := sorry

/-- Theorem stating the probability of ∠APB being obtuse -/
theorem probability_APB_obtuse :
  let p := problemPentagon
  let semicircleCenter := (2, 1)
  let semicircleRadius := Real.sqrt 5
  (semicircleArea semicircleRadius) / (pentagonArea p) = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_APB_obtuse_l663_66362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_lengths_right_triangle_l663_66371

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ 
  t.C = Real.pi / 3 ∧ 
  1 / 2 * t.a * t.b * Real.sin t.C = Real.sqrt 3

-- Theorem 1
theorem side_lengths (t : Triangle) (h : triangle_conditions t) : 
  t.a = 2 ∧ t.b = 2 := by
  sorry

-- Theorem 2
theorem right_triangle (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.C + Real.sin (t.B - t.A) = 2 * Real.sin (2 * t.A) → 
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_lengths_right_triangle_l663_66371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_hamiltonian_cycle_with_at_most_two_segments_l663_66318

/-- A complete graph with n vertices where each edge is colored with one of two colors -/
structure TwoColoredCompleteGraph (n : ℕ) where
  vertices : Fin n → Type
  edge_color : ∀ (i j : Fin n), i ≠ j → Bool

/-- A Hamiltonian cycle in the graph -/
def HamiltonianCycle (n : ℕ) (G : TwoColoredCompleteGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.Nodup ∧ cycle.head? = cycle.getLast? }

/-- The number of monochromatic segments in a Hamiltonian cycle -/
noncomputable def monochromaticSegments (n : ℕ) (G : TwoColoredCompleteGraph n) (cycle : HamiltonianCycle n G) : ℕ :=
  sorry

/-- Main theorem: There exists a Hamiltonian cycle with at most two monochromatic segments -/
theorem exists_hamiltonian_cycle_with_at_most_two_segments (n : ℕ) (G : TwoColoredCompleteGraph n) :
  ∃ (cycle : HamiltonianCycle n G), monochromaticSegments n G cycle ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_hamiltonian_cycle_with_at_most_two_segments_l663_66318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l663_66363

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2 ∧
  t.A + t.B + t.C = Real.pi

def satisfies_conditions (t : Triangle) : Prop :=
  is_valid_triangle t ∧
  Real.cos t.C + (Real.cos t.B - Real.sqrt 3 * Real.sin t.B) * Real.cos t.A = 0 ∧
  t.a = 2 * Real.sqrt 3 ∧
  t.b = 2 * Real.sqrt 2

-- Define the area function
noncomputable def triangle_area (t : Triangle) : Real :=
  1/2 * t.b * t.c * Real.sin t.A

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : satisfies_conditions t) :
  triangle_area t = 3 + Real.sqrt 3 ∧
  8 < 2 * t.b + t.c ∧ 2 * t.b + t.c ≤ 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l663_66363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_set_equals_complement_E_intersect_F_l663_66367

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set E
def E : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}

-- Define set F
def F : Set ℝ := {x | -1 < x ∧ x < 5}

-- Theorem statement
theorem target_set_equals_complement_E_intersect_F :
  {x : ℝ | -1 < x ∧ x < 2} = (Uᶜ ∩ E)ᶜ ∩ F := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_set_equals_complement_E_intersect_F_l663_66367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_dolls_discount_l663_66366

/-- Calculates the number of Russian dolls Daniel can buy at a discounted price -/
theorem russian_dolls_discount (original_price planned_dolls : ℕ) (new_price : ℚ) :
  original_price = 4 →
  planned_dolls = 15 →
  new_price = 3 →
  (original_price * planned_dolls : ℚ) / new_price = 20 := by
  sorry

#check russian_dolls_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_dolls_discount_l663_66366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3pi_minus_alpha_l663_66397

theorem sin_3pi_minus_alpha (α : Real) 
  (h1 : Real.cos (Real.pi + α) = -(1/2)) 
  (h2 : 3*Real.pi/2 < α) 
  (h3 : α < 2*Real.pi) : 
  Real.sin (3*Real.pi - α) = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3pi_minus_alpha_l663_66397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_RST_RSC_l663_66315

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem area_difference_RST_RSC : 
  let RST_base : ℝ := 5
  let RST_height : ℝ := 4
  let RSC_base : ℝ := 1
  let RSC_height : ℝ := 4
  (triangle_area RST_base RST_height) - (triangle_area RSC_base RSC_height) = 8 := by
  -- Unfold the definitions
  unfold triangle_area
  -- Perform the calculations
  simp [mul_comm, mul_assoc, add_comm, add_assoc]
  -- The proof is complete
  norm_num

#check area_difference_RST_RSC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_RST_RSC_l663_66315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l663_66314

def horner_polynomial (x : ℝ) : ℝ :=
  (((((5 * x - 3) * x + 3.6) * x - 7.2) * x - 10.1) * x + 7) * x - 3.5

def horner_operations : ℕ := 12

theorem horner_method_operations :
  ∀ x : ℝ, horner_operations = 12 := by
  intro x
  rfl

#eval horner_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l663_66314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_at_x1_l663_66370

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

-- State the theorem
theorem f_positive_at_x1 (x₀ x₁ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₀) (h3 : f x₀ = 0) : f x₁ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_at_x1_l663_66370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_l663_66365

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  /-- Radius of the cylinder -/
  r : ℝ
  /-- Height of the cylinder -/
  height : ℝ
  /-- Height of the cylinder equals its diameter -/
  height_eq_diameter : height = 2 * r
  /-- Diameter of the cone -/
  cone_diameter : ℝ
  /-- Altitude of the cone -/
  cone_altitude : ℝ
  /-- The cone's diameter is 8 -/
  cone_diameter_eq : cone_diameter = 8
  /-- The cone's altitude is 10 -/
  cone_altitude_eq : cone_altitude = 10
  /-- The axes of the cylinder and cone coincide -/
  axes_coincide : True

/-- The radius of the inscribed cylinder is 20/9 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) : c.r = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_l663_66365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_positive_integer_triples_sum_reciprocals_l663_66305

theorem finite_positive_integer_triples_sum_reciprocals : 
  Set.Finite { triple : ℕ × ℕ × ℕ | 
    let (a, b, c) := triple;
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1000 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_positive_integer_triples_sum_reciprocals_l663_66305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_of_f_l663_66344

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (3/2) * x^2 - Real.log x

-- State the theorem
theorem extreme_point_of_f :
  ∃! x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → (deriv f x = 0 ∧ (deriv f y = 0 → y = x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_of_f_l663_66344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l663_66338

-- Define the functions f and g
def f (x : ℝ) : ℝ := -x^2 - 3
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2*x*Real.log x - a*x

-- State the theorem
theorem tangent_and_inequality (a : ℝ) :
  -- Condition: Tangent lines of f and g are parallel at x = 1
  (deriv f 1 = deriv (g a) 1) →
  -- Claim 1: Equation of tangent line of g at (1, g(1))
  (∃ b c : ℝ, ∀ x y : ℝ, y = (g a 1) + (deriv (g a) 1) * (x - 1) ↔ 2*x + y + b = c) ∧
  -- Claim 2: Maximum value of a for which g(x) - f(x) ≥ 0 for all x > 0
  (∀ x : ℝ, x > 0 → g a x - f x ≥ 0) → a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l663_66338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_shop_purchase_l663_66381

noncomputable section

-- Define the variables
def total_weight : ℝ := 200
def apple_cost : ℝ := 15
def pear_cost : ℝ := 10
def apple_profit_rate : ℝ := 0.4
def pear_price_ratio : ℝ := 2/3
def total_profit : ℝ := 1020

-- Define the theorem
theorem fruit_shop_purchase :
  ∃ (apple_weight pear_weight : ℝ),
    apple_weight + pear_weight = total_weight ∧
    apple_weight * (apple_cost * apple_profit_rate) +
    pear_weight * (apple_cost * (1 + apple_profit_rate) * pear_price_ratio - pear_cost) = total_profit ∧
    apple_weight = 110 ∧
    pear_weight = 90 :=
by
  -- The proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_shop_purchase_l663_66381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_number_of_proper_subsets_M_l663_66335

-- Define M as a set of integers
def M : Set ℤ := {x | x^2 - 2*x - 3 < 0}

-- Define M as a finset
def M_finset : Finset ℤ := {0, 1, 2}

theorem M_eq : ∀ x : ℤ, x ∈ M ↔ x ∈ M_finset := by
  sorry

theorem number_of_proper_subsets_M : (Finset.powerset M_finset).card - 1 = 7 := by
  sorry

#eval (Finset.powerset M_finset).card - 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_number_of_proper_subsets_M_l663_66335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l663_66373

theorem right_triangle_side_length (Q : Real) (QP QR : Real) 
  (h1 : Real.cos Q = 0.6) (h2 : QP = 16) : QR = 80/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l663_66373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_calculation_l663_66358

/-- Given an article with an original price and a percentage decrease, 
    calculate the new price after the decrease. -/
noncomputable def new_price (original_price : ℝ) (percent_decrease : ℝ) : ℝ :=
  original_price * (1 - percent_decrease / 100)

/-- Theorem stating that for an article with original price 1100 
    and a 24% decrease, the new price is 836. -/
theorem price_decrease_calculation :
  new_price 1100 24 = 836 := by
  -- Unfold the definition of new_price
  unfold new_price
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_calculation_l663_66358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l663_66324

/-- Given a quadratic function g(x) with specific properties, 
    this theorem proves the values of its coefficients and 
    the range of k for a related inequality. -/
theorem quadratic_function_properties 
  (g : ℝ → ℝ) 
  (h : ∀ x, g x = a * x^2 - 2 * a * x + 1 + b) 
  (ha : a > 0) 
  (hmax : ∀ x ∈ Set.Icc 2 3, g x ≤ 4) 
  (hmin : ∀ x ∈ Set.Icc 2 3, g x ≥ 1) 
  (hmax_achieved : ∃ x ∈ Set.Icc 2 3, g x = 4) 
  (hmin_achieved : ∃ x ∈ Set.Icc 2 3, g x = 1) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, x ≠ 0 → f x = g x / x) : 
  (a = 1 ∧ b = 0) ∧ 
  (∀ k, (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (2^x) - k * 2^x ≥ 0) ↔ k ≤ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l663_66324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l663_66350

theorem tan_alpha_value (α β : ℝ) : 
  α ∈ Set.Ioo (-Real.pi/2) Real.pi → 
  Real.sin (α + 2*β) - 2 * Real.sin β * Real.cos (α + β) = -1/3 →
  Real.tan α = -Real.sqrt 2 / 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l663_66350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_on_curve_C_l663_66310

noncomputable section

/-- Curve C in polar coordinates -/
def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- The angle between OA and OB -/
def angle_AOB : ℝ := Real.pi / 3

/-- The sum of distances OA and OB -/
def sum_distances (θ : ℝ) : ℝ :=
  curve_C θ + curve_C (θ + angle_AOB)

/-- The maximum value of the sum of distances OA and OB -/
def max_sum_distances : ℝ := 2 * Real.sqrt 3

theorem max_sum_distances_on_curve_C :
  ∃ θ : ℝ, sum_distances θ ≤ max_sum_distances ∧
  ∀ φ : ℝ, sum_distances φ ≤ sum_distances θ :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_on_curve_C_l663_66310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_position_after_18_minutes_l663_66369

/-- Represents the position of a clock hand in degrees -/
def ClockPosition := ℝ

/-- Converts hours to degrees on a clock face -/
noncomputable def hoursToDegrees (hours : ℝ) : ClockPosition :=
  (hours * 360) % 360

/-- Converts minutes to degrees on a clock face -/
noncomputable def minutesToDegrees (minutes : ℝ) : ClockPosition :=
  (minutes * 6) % 360

/-- The number of degrees the hour hand moves in a given time -/
noncomputable def hourHandMovement (hours : ℝ) : ClockPosition :=
  hoursToDegrees (hours % 12)

/-- The number of degrees the minute hand moves in a given time -/
noncomputable def minuteHandMovement (hours : ℝ) : ClockPosition :=
  hoursToDegrees (hours * 12)

theorem clock_hands_position_after_18_minutes :
  let hours_passed : ℝ := 18 / 60
  let hour_hand_position := hourHandMovement hours_passed
  let minute_hand_position := minuteHandMovement hours_passed
  minute_hand_position = minutesToDegrees 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_position_after_18_minutes_l663_66369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_representation_l663_66309

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def f (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem f_representation (x : ℝ) : 
  ∃ k : ℤ, f x = x - (k : ℝ) ∧ (k : ℝ) ≤ x ∧ x < (k : ℝ) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_representation_l663_66309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l663_66307

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

-- State the theorem
theorem increasing_interval_of_f (φ : ℝ) 
  (h1 : π / 2 < |φ| ∧ |φ| < π) 
  (h2 : ∀ x : ℝ, f φ x ≤ |f φ (π / 6)|) :
  ∀ k : ℤ, StrictMonoOn (f φ) (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l663_66307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_max_area_l663_66394

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola : Set Point := {p : Point | p.y^2 = 4 * p.x}

/-- Represents a line y = x + m -/
def Line (m : ℝ) : Set Point := {p : Point | p.y = p.x + m}

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

theorem parabola_line_intersection_max_area :
  ∀ (m : ℝ), 0 ≤ m → m < 1 →
  ∃ (A B : Point),
    A ∈ Parabola ∧ A ∈ Line m ∧
    B ∈ Parabola ∧ B ∈ Line m ∧
    A ≠ B ∧
    (∀ (C D : Point),
      C ∈ Parabola → C ∈ Line m →
      D ∈ Parabola → D ∈ Line m →
      C ≠ D →
      triangleArea (Point.mk 1 0) C D ≤ triangleArea (Point.mk 1 0) A B) ∧
    triangleArea (Point.mk 1 0) A B = 8 * Real.sqrt 6 / 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_max_area_l663_66394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_equals_negative_one_l663_66378

theorem complex_square_equals_negative_one : 
  ((1 + Complex.I) / (1 - Complex.I)) ^ 2 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_equals_negative_one_l663_66378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l663_66377

def a : ℕ → ℚ
  | 0 => 3  -- Add this case for n = 0
  | 1 => 3
  | n + 1 => 3 * a n / (a n + 3)

theorem a_formula (n : ℕ) (h : n ≥ 1) : a n = 3 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l663_66377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_count_in_range_l663_66353

theorem prime_count_in_range : 
  (∃! p : Nat, 200 < p ∧ p < 220 ∧ Nat.Prime p) → 
  (Finset.filter (λ n : Nat ↦ 200 < n ∧ n < 220 ∧ Nat.Prime n) (Finset.range 220)).card = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_count_in_range_l663_66353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l663_66359

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the parallel relation for vectors
def parallel (v w : MyVector) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Define vector subtraction
def vsub (v w : MyVector) : MyVector :=
  (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem vector_subtraction (t : ℝ) :
  let a : MyVector := (-1, -3)
  let b : MyVector := (2, t)
  parallel a b → vsub a b = (-3, -9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l663_66359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_depends_on_d_and_n_l663_66355

noncomputable def arithmetic_sum (a d : ℝ) (k : ℕ) : ℝ := k / 2 * (2 * a + (k - 1) * d)

noncomputable def T (a d : ℝ) (n : ℕ) : ℝ :=
  arithmetic_sum a d (5 * n) - arithmetic_sum a d (4 * n) - arithmetic_sum a d n

theorem T_depends_on_d_and_n (a d : ℝ) (n : ℕ) :
  T a d n = 14 * d * n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_depends_on_d_and_n_l663_66355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_seven_ninths_l663_66326

-- Define the function f(x) = cos x sin 2x
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (2 * x)

-- Theorem statement
theorem no_solution_for_seven_ninths :
  ¬ ∃ x : ℝ, f x = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_seven_ninths_l663_66326
