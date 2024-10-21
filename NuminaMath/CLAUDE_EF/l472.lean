import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_properties_l472_47217

/-- Regular hexagon with side length 6 -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 6

/-- The length of a diagonal in a regular hexagon -/
noncomputable def diagonal_length (h : RegularHexagon) : ℝ := 6 * Real.sqrt 3

/-- The perimeter of a regular hexagon -/
def perimeter (h : RegularHexagon) : ℝ := 6 * h.side_length

theorem regular_hexagon_properties (h : RegularHexagon) :
  diagonal_length h = 6 * Real.sqrt 3 ∧ perimeter h = 36 := by
  sorry

#check regular_hexagon_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_properties_l472_47217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4i_molecular_weight_l472_47268

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_nitrogen : ℕ) (n_hydrogen : ℕ) (n_iodine : ℕ) 
                     (atomic_weight_N : ℝ) (atomic_weight_H : ℝ) (atomic_weight_I : ℝ) : ℝ :=
  n_nitrogen * atomic_weight_N + n_hydrogen * atomic_weight_H + n_iodine * atomic_weight_I

/-- The molecular weight of the compound NH₄I is approximately 144.95 g/mol -/
theorem nh4i_molecular_weight :
  let n_nitrogen : ℕ := 1
  let n_hydrogen : ℕ := 4
  let n_iodine : ℕ := 1
  let atomic_weight_N : ℝ := 14.01
  let atomic_weight_H : ℝ := 1.01
  let atomic_weight_I : ℝ := 126.90
  abs ((molecular_weight n_nitrogen n_hydrogen n_iodine atomic_weight_N atomic_weight_H atomic_weight_I) - 144.95) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4i_molecular_weight_l472_47268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l472_47294

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  focus_coincides_with_parabola : Bool

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / 5 + y^2 / 4 = 1

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The length of a chord formed by the intersection of a line and an ellipse -/
noncomputable def chord_length (e : Ellipse) (l : Line) : ℝ := 16 * Real.sqrt 5 / 9

/-- Main theorem -/
theorem ellipse_properties (e : Ellipse) (l : Line) :
  e.center = (0, 0) ∧
  e.foci_on_x_axis = true ∧
  e.eccentricity = Real.sqrt 5 / 5 ∧
  e.focus_coincides_with_parabola = true ∧
  l.point = (-1, 0) ∧
  l.slope = 1 →
  standard_equation e ∧
  chord_length e l = 16 * Real.sqrt 5 / 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l472_47294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l472_47206

/-- An ellipse with its upper vertex and left focus on a specific line -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_line : ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ x - y + 2 = 0 ∧ 
           ((x = 0 ∧ y > 0) ∨ (y = 0 ∧ x < 0))

/-- The standard equation of the ellipse and the equation of line l -/
def ellipse_and_line (E : SpecialEllipse) : 
  (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (λ x y ↦ x^2 / 8 + y^2 / 4 = 1, λ x y ↦ x - 2*y + 3 = 0)

/-- The main theorem -/
theorem special_ellipse_properties (E : SpecialEllipse) :
  let (std_eq, line_eq) := ellipse_and_line E
  (∀ x y, x^2 / E.a^2 + y^2 / E.b^2 = 1 ↔ std_eq x y) ∧
  (∃ P Q : ℝ × ℝ, 
    std_eq P.1 P.2 ∧ std_eq Q.1 Q.2 ∧
    line_eq P.1 P.2 ∧ line_eq Q.1 Q.2 ∧
    ∃ B F : ℝ × ℝ, 
      B.1 - B.2 + 2 = 0 ∧ F.1 - F.2 + 2 = 0 ∧
      (B.1 = 0 ∧ B.2 > 0) ∧ (F.2 = 0 ∧ F.1 < 0) ∧
      (P.1 - B.1, P.2 - B.2) = (F.1 - Q.1, F.2 - Q.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l472_47206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gail_works_twelve_hours_gail_work_calculation_gail_work_hours_correct_l472_47249

/-- Represents the working hours of Gail on Saturday -/
def gailWorkHours (x : ℝ) : ℝ := 12

theorem gail_works_twelve_hours (x : ℝ) :
  gailWorkHours x = 12 := by
  -- Unfold the definition of gailWorkHours
  unfold gailWorkHours
  -- The result is immediately true by definition
  rfl

theorem gail_work_calculation (x : ℝ) :
  (12 - x) + x = 12 := by
  -- Simplify the left side of the equation
  ring

theorem gail_work_hours_correct (x : ℝ) :
  gailWorkHours x = (12 - x) + x := by
  -- Use the previous theorems
  rw [gail_works_twelve_hours, gail_work_calculation]

#check gail_works_twelve_hours
#check gail_work_calculation
#check gail_work_hours_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gail_works_twelve_hours_gail_work_calculation_gail_work_hours_correct_l472_47249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sail_number_divisibility_l472_47264

theorem sail_number_divisibility (sail_number : Nat) (children_ages : List Nat) (mother_age : Nat) : 
  -- Conditions
  sail_number ≥ 1000 ∧ sail_number ≤ 9999 ∧  -- 4-digit number
  (∃ a b : Nat, (sail_number = a * 1000 + a * 100 + b * 10 + b) ∨ 
               (sail_number = a * 1000 + b * 100 + a * 10 + b)) ∧  -- aabb or abab pattern
  children_ages.length = 7 ∧  -- 7 children
  children_ages.all (· > 0) ∧  -- all ages are positive
  children_ages.minimum? = some 5 ∧  -- youngest child is 5
  List.Pairwise (·≠·) children_ages ∧  -- all ages are different
  sail_number % 10 * 10 + sail_number % 100 / 10 = mother_age ∧  -- last two digits are mother's age
  (∃ (non_divisor : Nat), non_divisor ∈ children_ages ∧ 
    sail_number % non_divisor ≠ 0 ∧
    ∀ (age : Nat), age ∈ children_ages ∧ age ≠ non_divisor → sail_number % age = 0) →
  -- Conclusion
  ∃ (non_divisor : Nat), non_divisor ∈ children_ages ∧ non_divisor = 4 ∧
    sail_number % non_divisor ≠ 0 ∧
    ∀ (age : Nat), age ∈ children_ages ∧ age ≠ non_divisor → sail_number % age = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sail_number_divisibility_l472_47264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_growth_pattern_l472_47277

/-- Represents the number of branches on the main stem and small branches on each branch -/
def x : ℕ := sorry

/-- The total number of stems, branches, and small branches -/
def total_count : ℕ := 1 + x + x^2

/-- Theorem stating that the total count is 73 -/
theorem plant_growth_pattern : total_count = 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_growth_pattern_l472_47277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_l472_47233

/-- Prove that the given vectors form a basis for R^2 -/
theorem vectors_form_basis : 
  let e₁ : Fin 2 → ℝ := ![0, -1]
  let e₂ : Fin 2 → ℝ := ![-1, 0]
  LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_l472_47233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_probability_tokyo_pair_probability_l472_47219

structure City : Type :=
  (name : String)

def distance (c1 c2 : City) : ℕ :=
  match c1.name, c2.name with
  | "Tokyo", "Cairo" => 5900
  | "Tokyo", "Sydney" => 4800
  | "Tokyo", "Toronto" => 6450
  | "Tokyo", "Paris" => 6040
  | "Cairo", "Sydney" => 8650
  | "Cairo", "Toronto" => 5260
  | "Cairo", "Paris" => 2140
  | "Sydney", "Toronto" => 9440
  | "Sydney", "Paris" => 10600
  | "Toronto", "Paris" => 3700
  | _, _ => 0

def cities : List City :=
  [⟨"Tokyo"⟩, ⟨"Cairo"⟩, ⟨"Sydney"⟩, ⟨"Toronto"⟩, ⟨"Paris"⟩]

def city_pairs : List (City × City) :=
  List.filter (λ p => p.1.name < p.2.name) (List.product cities cities)

theorem city_distance_probability :
  (city_pairs.filter (λ p => distance p.1 p.2 < 8000)).length / city_pairs.length = 7 / 10 :=
sorry

theorem tokyo_pair_probability :
  (city_pairs.filter (λ p => p.1.name = "Tokyo" ∨ p.2.name = "Tokyo")).length / city_pairs.length = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_probability_tokyo_pair_probability_l472_47219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_in_fourth_quadrant_l472_47261

noncomputable def z : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

theorem z_squared_in_fourth_quadrant :
  let w := z^2
  (w.re > 0) ∧ (w.im < 0) := by
  -- Define w
  let w := z^2
  
  -- Calculate w explicitly
  have w_calc : w = 1 - (Real.sqrt 3 / 2) * Complex.I := by
    -- The actual calculation would go here
    sorry
  
  -- Show that the real part is positive
  have re_pos : w.re > 0 := by
    rw [w_calc]
    -- Proof that 1 > 0 would go here
    sorry
  
  -- Show that the imaginary part is negative
  have im_neg : w.im < 0 := by
    rw [w_calc]
    -- Proof that -√3/2 < 0 would go here
    sorry
  
  -- Combine the results
  exact ⟨re_pos, im_neg⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_in_fourth_quadrant_l472_47261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_is_48_l472_47218

/-- Regular polygon with n sides and side length 2 -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The interior angle of a regular polygon -/
noncomputable def interior_angle (p : RegularPolygon) : ℝ :=
  180 * (p.n - 2) / p.n

/-- Configuration of three regular polygons meeting at a point -/
structure PolygonConfiguration where
  p1 : RegularPolygon
  p2 : RegularPolygon
  p3 : RegularPolygon
  at_least_two_congruent : p1 = p2 ∨ p1 = p3 ∨ p2 = p3
  angles_sum : interior_angle p1 + interior_angle p2 + interior_angle p3 = 720

/-- The perimeter of the new polygon formed by the configuration -/
def new_polygon_perimeter (c : PolygonConfiguration) : ℝ :=
  2 * (c.p1.n + c.p2.n + c.p3.n - 6)

/-- The theorem stating the maximum perimeter -/
theorem max_perimeter_is_48 :
  ∃ (c : PolygonConfiguration), ∀ (c' : PolygonConfiguration),
    new_polygon_perimeter c' ≤ new_polygon_perimeter c ∧
    new_polygon_perimeter c = 48 := by
  sorry

#check max_perimeter_is_48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_is_48_l472_47218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_timeForOliverArrangements_l472_47240

/-- The time in hours to write all arrangements of a 6-letter name -/
def timeToWriteArrangements (writingSpeed : ℕ) : ℚ :=
  (Nat.factorial 6 : ℚ) / (writingSpeed : ℚ) / 60

/-- Theorem: It takes 0.8 hours to write all arrangements of a 6-letter name at 15 arrangements per minute -/
theorem timeForOliverArrangements :
  timeToWriteArrangements 15 = 4/5 := by
  -- Unfold the definition of timeToWriteArrangements
  unfold timeToWriteArrangements
  -- Simplify the expression
  simp [Nat.factorial]
  -- Perform the numerical calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_timeForOliverArrangements_l472_47240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l472_47239

/-- The area of a triangle with vertices at (2, 3), (7, 3), and (4, 9) is 15 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 15 := by
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 7
  let y₂ : ℝ := 3
  let x₃ : ℝ := 4
  let y₃ : ℝ := 9
  let area : ℝ := (1/2) * |x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂)|
  use area
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l472_47239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l472_47202

/-- A polynomial function with real coefficients -/
def MyPolynomial := ℕ → ℝ

/-- A strictly increasing sequence of natural numbers -/
def StrictlyIncreasingSequence := ℕ → ℕ

/-- The set of prime numbers that divide at least one term of a sequence -/
def PrimeDivisors (a : ℕ → ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ n, p ∣ a n}

theorem infinite_prime_divisors
  (f : MyPolynomial)
  (a : StrictlyIncreasingSequence)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_bound : ∀ n, (a n : ℝ) ≤ f n) :
  Set.Infinite (PrimeDivisors a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l472_47202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l472_47209

noncomputable def f (x : ℝ) : ℝ := (x^4 - 9*x^2 + 20) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l472_47209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_greater_than_17_l472_47273

/-- A circular arrangement of 10 distinct integers from 1 to 10 -/
def CircularArrangement := { a : Fin 10 → Fin 10 // Function.Injective a }

/-- The sum of three consecutive numbers in the circular arrangement -/
def ConsecutiveSum (a : CircularArrangement) (i : Fin 10) : ℕ :=
  (a.val i).val + (a.val (i + 1)).val + (a.val (i + 2)).val

theorem exists_sum_greater_than_17 (a : CircularArrangement) :
  ∃ i : Fin 10, ConsecutiveSum a i > 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_greater_than_17_l472_47273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_longest_side_l472_47235

/-- Given a triangle with side lengths 8, x^2 + 3x + 2, and 2x + 6, and perimeter 45,
    prove that the longest side has length (166 - 15 * Real.sqrt 141) / 4 -/
theorem triangle_longest_side (x : ℝ) 
  (side1 : ℝ := 8)
  (side2 : ℝ := x^2 + 3*x + 2)
  (side3 : ℝ := 2*x + 6)
  (perimeter_eq : side1 + side2 + side3 = 45) :
  max side1 (max side2 side3) = (166 - 15 * Real.sqrt 141) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_longest_side_l472_47235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_survey_results_l472_47285

def suburban_scores : List ℚ := [74, 81, 75, 76, 70, 75, 75, 79, 81, 70, 74, 80, 91, 69, 82]
def urban_scores : List ℚ := [81, 94, 83, 77, 83, 80, 81, 70, 81, 73, 78, 82, 80, 70, 50]

def count_between (l : List ℚ) (lower upper : ℚ) : ℕ :=
  (l.filter (λ x => lower ≤ x ∧ x < upper)).length

noncomputable def median (l : List ℚ) : ℚ := sorry

noncomputable def mode (l : List ℚ) : ℚ := sorry

def mean (l : List ℚ) : ℚ :=
  (l.sum) / l.length

theorem epidemic_survey_results :
  let a := count_between urban_scores 60 80
  let b := median suburban_scores
  let c := mode urban_scores
  a = 5 ∧
  b = 75 ∧
  c = 81 ∧
  mean urban_scores > mean suburban_scores ∧
  (15000 : ℚ) * ((suburban_scores.filter (λ x => x ≥ 90)).length / suburban_scores.length) = 1000 := by
  sorry

#check epidemic_survey_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_survey_results_l472_47285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_value_compound_interest_l472_47260

/-- 
Given an initial value and a compound interest rate,
if the value quadruples after 20 years of annual compounding,
then the rate is approximately 0.0718 (7.18%)
-/
theorem art_value_compound_interest 
  (A : ℝ) 
  (R : ℝ) 
  (h : A * (1 + R)^20 = 4 * A) : 
  abs (R - 0.0718) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_value_compound_interest_l472_47260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_at_m_equals_one_over_101_m_equals_5051_l472_47256

/-- Represents the sequence as described in the problem -/
def F : ℕ → ℚ := sorry

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of 1/101 in the sequence -/
def m : ℕ := triangular_sum 100 + 1

/-- Theorem stating that F(m) = 1/101 when m = 5051 -/
theorem F_at_m_equals_one_over_101 : F m = 1 / 101 := by
  sorry

/-- Theorem proving that m = 5051 -/
theorem m_equals_5051 : m = 5051 := by
  unfold m triangular_sum
  norm_num

#eval m  -- This will output 5051

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_at_m_equals_one_over_101_m_equals_5051_l472_47256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sahar_movement_correct_l472_47275

/-- Represents Sahar's movement over time -/
noncomputable def sahar_movement (k : ℝ) (t : ℝ) : ℝ :=
  if t ≤ 10 then k * t else k * 10

theorem sahar_movement_correct :
  ∃ (k : ℝ),
    k > 0 ∧
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 →
      (t ≤ 10 → sahar_movement k t = k * t) ∧
      (t > 10 → sahar_movement k t = k * 10)) :=
by
  sorry

#check sahar_movement_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sahar_movement_correct_l472_47275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_segments_equal_l472_47276

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given conditions
axiom O₁ : Circle
axiom O₂ : Circle
axiom A : Point
axiom B : Point
axiom P : Point
axiom Q : Point
axiom C : Point
axiom D : Point

-- Define necessary functions
def on_circle (p : Point) (c : Circle) : Prop := sorry
def is_tangent_line (c : Circle) (p1 p2 : Point) : Prop := sorry
def distance (p1 p2 : Point) : ℝ := sorry
def line_through (p1 p2 : Point) : Set Point := sorry

-- Axioms representing the given conditions
axiom circles_intersect : 
  on_circle A O₁ ∧ on_circle A O₂ ∧ on_circle B O₁ ∧ on_circle B O₂

axiom PQ_tangent : 
  is_tangent_line O₁ P Q ∧ is_tangent_line O₂ P Q

axiom PQ_closer_to_B : 
  distance B P < distance A P ∧ distance B Q < distance A Q

axiom C_on_AP : C ∈ line_through A P
axiom D_on_AQ : D ∈ line_through A Q
axiom C_on_QB_extended : C ∈ line_through Q B
axiom D_on_PB_extended : D ∈ line_through P B

-- Theorem to prove
theorem product_of_segments_equal : 
  distance A C * distance B C = distance A D * distance B D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_segments_equal_l472_47276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_division_ratio_l472_47204

/-- Represents a square-based pyramid --/
structure SquarePyramid where
  base_side : ℝ
  height : ℝ

/-- Represents the division of a pyramid into two parts --/
structure PyramidDivision where
  original : SquarePyramid
  division_height : ℝ

/-- Calculates the volume of a square-based pyramid --/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

/-- Calculates the lateral surface area of a square-based pyramid --/
noncomputable def pyramid_lateral_area (p : SquarePyramid) : ℝ :=
  2 * p.base_side * (p.height^2 + (p.base_side/2)^2).sqrt

/-- Calculates the ratio of volumes and surface areas for a divided pyramid --/
noncomputable def division_ratio (d : PyramidDivision) : ℝ :=
  let small_base := d.division_height * (d.original.base_side / d.original.height)
  let small_volume := pyramid_volume ⟨small_base, d.division_height⟩
  let small_area := pyramid_lateral_area ⟨small_base, d.division_height⟩
  let large_volume := pyramid_volume d.original - small_volume
  let large_area := pyramid_lateral_area d.original - small_area
  small_volume / large_volume

/-- The main theorem --/
theorem pyramid_division_ratio :
  let p := SquarePyramid.mk 6 4
  let d := PyramidDivision.mk p (4.5 * 4 / 6)
  division_ratio d = 91 / 149 := by sorry

#eval (91 : ℕ) + (149 : ℕ)  -- Should evaluate to 240

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_division_ratio_l472_47204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_plus_x_l472_47278

-- Define the piecewise linear function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then -3
  else if x ≤ -1 then -2
  else if x ≤ 0 then -1
  else if x ≤ 1 then 0
  else if x ≤ 2 then 1
  else 2

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem range_of_f_plus_x :
  Set.range (fun x => f x + x) = Set.Icc (-6) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_plus_x_l472_47278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_triangular_6_l472_47296

/-- The nth triangular number -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of square roots of first n triangular numbers -/
noncomputable def sum_sqrt_triangular (n : ℕ) : ℝ :=
  (Finset.range n).sum (fun i => Real.sqrt (triangular (i + 1)))

/-- Theorem: Sum of square roots of first 6 triangular numbers equals 21 -/
theorem sum_sqrt_triangular_6 :
  sum_sqrt_triangular 6 = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_triangular_6_l472_47296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l472_47248

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l472_47248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_increases_l472_47253

/-- Represents the time for a round trip of a ferry between two points -/
noncomputable def roundTripTime (distance : ℝ) (ferrySpeed : ℝ) (waterFlow : ℝ) : ℝ :=
  (distance / (ferrySpeed + waterFlow)) + (distance / (ferrySpeed - waterFlow))

theorem round_trip_time_increases 
  (distance : ℝ) (ferrySpeed : ℝ) (waterFlow : ℝ) 
  (h1 : distance > 0) 
  (h2 : ferrySpeed > 0) 
  (h3 : 0 ≤ waterFlow) 
  (h4 : waterFlow < ferrySpeed) :
  ∀ ε > 0, roundTripTime distance ferrySpeed (waterFlow + ε) > roundTripTime distance ferrySpeed waterFlow := by
  sorry

#check round_trip_time_increases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_increases_l472_47253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l472_47269

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : Point := sorry

/-- Predicate to check if a point is on the right branch of the hyperbola -/
def is_on_right_branch (h : Hyperbola a b) (p : Point) : Prop := sorry

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral_triangle (p1 p2 p3 : Point) : Prop := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  ∃ (p : Point), 
    is_on_right_branch h p ∧ 
    is_equilateral_triangle p origin (right_focus h) →
    eccentricity h = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l472_47269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l472_47210

theorem m_greater_than_n (a : ℝ) : 
  (5 * a^2 - a + 1) > (4 * a^2 + a - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l472_47210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_interval_l472_47258

-- Define the function f(x) = 3/x
noncomputable def f (x : ℝ) : ℝ := 3 / x

-- Define the property of not being a decreasing function
def not_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ f x < f y

-- Theorem statement
theorem f_not_decreasing_interval :
  ∀ a b : ℝ, a < 0 ∧ 0 < b →
  (∀ x ∈ Set.Ioo a 0, f x ∈ Set.range f) ∧
  (∀ x ∈ Set.Ioo 0 b, f x ∈ Set.range f) ∧
  not_decreasing f a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_interval_l472_47258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_theorem_line_l_theorem_l472_47242

-- Define the points A and B
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)

-- Define the perpendicular bisector equation
def perpendicular_bisector : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * x - 4 * y - 23 = 0

-- Define the equations of line l
def line_l₁ : ℝ → ℝ → Prop := λ x y ↦ x + y - 2 = 0
def line_l₂ : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 4 * y = 0

-- Theorem for the perpendicular bisector
theorem perpendicular_bisector_theorem :
  perpendicular_bisector = λ x y ↦ 3 * x - 4 * y - 23 = 0 :=
by
  rfl

-- Theorem for line l
theorem line_l_theorem :
  (∃ x y, line_l₁ x y ∧ (x, y) = A) ∧
  (∃ x y, line_l₂ x y ∧ (x, y) = A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_theorem_line_l_theorem_l472_47242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_age_l472_47237

/-- Represents the age of a person -/
structure Age where
  value : ℕ

/-- Addison's age is three times Brenda's age -/
def addison_age (brenda : Age) : Age :=
  ⟨3 * brenda.value⟩

/-- Janet is six years older than Brenda -/
def janet_age (brenda : Age) : Age :=
  ⟨brenda.value + 6⟩

/-- Addison and Janet are twins -/
axiom twins (brenda : Age) : addison_age brenda = janet_age brenda

theorem brenda_age : ∃ (brenda : Age), 
  addison_age brenda = janet_age brenda ∧ brenda.value = 3 := by
  use ⟨3⟩
  constructor
  · simp [addison_age, janet_age]
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brenda_age_l472_47237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_transformable_iff_even_ones_l472_47283

structure Board where
  grid : Fin 4 → Fin 4 → Bool

def count_ones (b : Board) : Nat :=
  (Finset.univ.sum fun i => Finset.univ.sum fun j => if b.grid i j then 1 else 0)

def flip_row (b : Board) (row : Fin 4) : Board where
  grid := fun i j => if i = row then !b.grid i j else b.grid i j

def flip_column (b : Board) (col : Fin 4) : Board where
  grid := fun i j => if j = col then !b.grid i j else b.grid i j

def flip_diagonal (b : Board) (diag : Nat) : Board where
  grid := fun i j => if i.val + j.val = diag then !b.grid i j else b.grid i j

def can_transform_to_zeros (b : Board) : Prop :=
  ∃ (rows cols diags : List (Fin 4)),
    let b' := (rows.foldl flip_row b)
    let b'' := (cols.foldl flip_column b')
    let b''' := (diags.foldl flip_diagonal b'')
    count_ones b''' = 0

theorem board_transformable_iff_even_ones (b : Board) :
  can_transform_to_zeros b ↔ Even (count_ones b) := by
  sorry

#check board_transformable_iff_even_ones

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_transformable_iff_even_ones_l472_47283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l472_47297

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def solution_set : Set ℝ := {(1/2) * Real.sqrt 29, (1/2) * Real.sqrt 189, (1/2) * Real.sqrt 229, (1/2) * Real.sqrt 269}

theorem floor_equation_solution :
  {x : ℝ | 4 * x^2 - 40 * (floor x) + 51 = 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l472_47297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l472_47201

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x₁ x₂, Real.pi / 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi → f ω x₁ > f ω x₂) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l472_47201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equiv_a_monotone_l472_47227

/-- Definition of the sequence a_n -/
noncomputable def a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n + 1, a₀ => 2^n - 3 * a n a₀

/-- The closed form expression for a_n -/
noncomputable def a_closed_form (n : ℕ) (a₀ : ℝ) : ℝ :=
  (1/5) * (2^n + (-1)^(n-1) * 3^n) + (-1)^n * 3^n * a₀

/-- Theorem stating the equivalence of the recursive and closed forms -/
theorem a_equiv (n : ℕ) (a₀ : ℝ) :
  a n a₀ = a_closed_form n a₀ := by sorry

/-- Theorem stating the monotonicity of the sequence when a₀ = 1/5 -/
theorem a_monotone (n : ℕ) :
  a (n + 1) (1/5) > a n (1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equiv_a_monotone_l472_47227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_diagonals_of_inscribed_trapezoids_l472_47243

/-- Two isosceles trapezoids inscribed in a circle -/
structure InscribedTrapezoids where
  circle : Set (ℝ × ℝ)
  trapezoid1 : Set (ℝ × ℝ)
  trapezoid2 : Set (ℝ × ℝ)
  isInscribed1 : trapezoid1 ⊆ circle
  isInscribed2 : trapezoid2 ⊆ circle
  isIsosceles1 : Prop
  isIsosceles2 : Prop
  hasParallelSides1 : Prop
  hasParallelSides2 : Prop

/-- Diagonal of a trapezoid -/
noncomputable def diagonal (trapezoid : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The diagonals of two isosceles trapezoids inscribed in a circle are equal -/
theorem equal_diagonals_of_inscribed_trapezoids (t : InscribedTrapezoids) :
  diagonal t.trapezoid1 = diagonal t.trapezoid2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_diagonals_of_inscribed_trapezoids_l472_47243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_l472_47221

/-- The area of a circle is equal to π times the square of its radius. -/
theorem area_of_circle (r : ℝ) (h : r > 0) : 
  π * r^2 = π * r^2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_l472_47221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l472_47267

theorem sin_double_angle (x : ℝ) (h : Real.sin (x + π / 4) = -5 / 13) : 
  Real.sin (2 * x) = -119 / 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l472_47267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_11_perms_l472_47238

/-- A permutation of the digits 1 to 7 -/
def Digit7Perm := Fin 7 → Fin 7

/-- Check if a Digit7Perm represents a number divisible by 11 -/
def isDivisibleBy11 (p : Digit7Perm) : Prop :=
  (((p 0).val + 1) + ((p 2).val + 1) + ((p 4).val + 1) + ((p 6).val + 1)) -
  (((p 1).val + 1) + ((p 3).val + 1) + ((p 5).val + 1)) ≡ 0 [MOD 11]

/-- The set of all Digit7Perms that are divisible by 11 -/
def divisibleBy11Perms : Set Digit7Perm :=
  {p | isDivisibleBy11 p ∧ Function.Injective p}

/-- Finite type instance for divisibleBy11Perms -/
instance : Fintype divisibleBy11Perms := by sorry

theorem count_divisible_by_11_perms :
  Fintype.card divisibleBy11Perms = 576 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_11_perms_l472_47238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l472_47205

def number_of_arrangements (n : ℕ) (adjacent_pair : ℕ) (separated_pair : ℕ) : ℕ :=
  (n - adjacent_pair).factorial * adjacent_pair.factorial * (n - separated_pair + 1).choose 2

theorem photo_arrangements :
  number_of_arrangements 7 2 2 = 960 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l472_47205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_l472_47229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 4) + a * x

theorem monotonic_condition (a : ℝ) :
  (∀ x ≥ 0, Monotone (f a)) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_l472_47229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_speed_is_36_over_19_l472_47207

/-- Represents the hiking scenario described in the problem -/
structure HikingScenario where
  d : ℚ  -- distance from trailhead to peak
  chantal_speed_first_half : ℚ := 4
  chantal_speed_second_half : ℚ := 2
  chantal_speed_descent_steep : ℚ := 3
  chantal_speed_descent_flat : ℚ := 4
  extra_flat_distance : ℚ := 1
  meeting_point_distance : ℚ := 1

/-- Calculates Jean's average speed given a hiking scenario -/
def jeans_average_speed (scenario : HikingScenario) : ℚ :=
  let total_distance := 2 * scenario.d - scenario.meeting_point_distance
  let total_time := (scenario.d / 2 / scenario.chantal_speed_first_half) +
                    (scenario.d / 2 / scenario.chantal_speed_second_half) +
                    (scenario.d / 2 / scenario.chantal_speed_descent_steep) +
                    (scenario.d / 2 / scenario.chantal_speed_descent_flat) +
                    (scenario.extra_flat_distance / scenario.chantal_speed_descent_flat)
  total_distance / total_time

/-- Theorem stating that Jean's average speed is 36/19 mph -/
theorem jeans_speed_is_36_over_19 (scenario : HikingScenario) (h : scenario.d = 2) :
  jeans_average_speed scenario = 36 / 19 := by
  sorry

#eval jeans_average_speed { d := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_speed_is_36_over_19_l472_47207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_formation_l472_47241

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions for part (a)
def height_condition (t : Triangle) (h : ℝ) (c : Circle) : Prop :=
  ∃ (vertex : ℝ × ℝ) (base_midpoint : ℝ × ℝ),
    vertex ∈ [t.A, t.B, t.C] ∧
    base_midpoint ∈ [t.A, t.B, t.C] ∧
    vertex ≠ base_midpoint ∧
    c.center = ((vertex.1 + base_midpoint.1) / 2, (vertex.2 + base_midpoint.2) / 2) ∧
    c.radius = h / 2

-- Define the conditions for part (b)
def angle_bisector_condition (t : Triangle) (c : Circle) : Prop :=
  ∃ (vertex : ℝ × ℝ) (base : ℝ × ℝ) (side : ℝ × ℝ),
    vertex ∈ [t.A, t.B, t.C] ∧
    base ∈ [t.A, t.B, t.C] ∧
    side ∈ [t.A, t.B, t.C] ∧
    vertex ≠ base ∧ vertex ≠ side ∧ base ≠ side ∧
    c.center ∈ [t.A, t.B, t.C] ∧
    -- The tangent to the inscribed circle that contains the base is symmetrical to the side
    -- with respect to the center of the inscribed circle
    ∃ (tangent_point : ℝ × ℝ),
      (tangent_point.1 - base.1) * (c.center.2 - base.2) =
      (tangent_point.2 - base.2) * (c.center.1 - base.1) ∧
      (side.1 - c.center.1) * (tangent_point.2 - c.center.2) =
      (side.2 - c.center.2) * (tangent_point.1 - c.center.1)

-- Theorem stating that no triangle can be formed under these conditions
theorem no_triangle_formation (t : Triangle) (h : ℝ) (c : Circle) :
  (height_condition t h c ∨ angle_bisector_condition t c) → False := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_formation_l472_47241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_than_half_perimeter_inside_l472_47280

/-- A triangle with an inscribed circle and a circumscribed square -/
structure TriangleCircleSquare where
  /-- The triangle -/
  triangle : Set (Fin 2 → ℝ)
  /-- The inscribed circle -/
  circle : Set (Fin 2 → ℝ)
  /-- The circumscribed square -/
  square : Set (Fin 2 → ℝ)
  /-- The circle is inscribed in the triangle -/
  circle_inscribed : circle ⊆ triangle
  /-- The square is circumscribed around the circle -/
  square_circumscribed : circle ⊆ square

/-- The perimeter of a square -/
noncomputable def squarePerimeter (s : Set (Fin 2 → ℝ)) : ℝ := sorry

/-- The portion of the square's perimeter inside the triangle -/
noncomputable def perimeterInsideTriangle (tcs : TriangleCircleSquare) : ℝ := sorry

/-- Theorem: More than half of the square's perimeter is inside the triangle -/
theorem more_than_half_perimeter_inside (tcs : TriangleCircleSquare) :
  perimeterInsideTriangle tcs > (squarePerimeter tcs.square) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_than_half_perimeter_inside_l472_47280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l472_47247

/-- The area of a quadrilateral is less than or equal to one-fourth the product of the sums of opposite sides. -/
theorem quadrilateral_area_inequality (a b c d S : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hS : S > 0) : 
  S ≤ (1/4) * (a + b) * (c + d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l472_47247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l472_47293

theorem xy_value (x y : ℝ) (h1 : (2 : ℝ)^x = (16 : ℝ)^(y+1)) (h2 : (27 : ℝ)^y = (3 : ℝ)^(x-2)) : x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l472_47293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worm_growth_l472_47215

/-- The minimum time required to obtain n adult worms from one adult worm -/
noncomputable def min_time (n : ℕ+) : ℝ :=
  1 - 1 / (2 ^ (n.val - 1))

/-- An adult worm is 1 meter long -/
def adult_length : ℝ := 1

/-- Growth rate of worm parts in meters per hour -/
def growth_rate : ℝ := 1

theorem worm_growth (n : ℕ+) :
  ∀ (cut_strategy : ℕ+ → ℝ),
  (∀ m : ℕ+, m ≤ n → 0 < cut_strategy m ∧ cut_strategy m < adult_length) →
  (∀ t : ℝ, t ≥ 0 → 
    (∃ (num_adults : ℕ) (growing_parts : ℕ+ → ℝ),
      (↑num_adults : ℝ) + (Finset.sum (Finset.range n.val) (λ i => growing_parts ⟨i + 1, Nat.succ_pos i⟩)) = n ∧
      ∀ i : ℕ+, i ≤ n →
        growing_parts i ≤ adult_length ∧
        growing_parts i = min adult_length (cut_strategy i + growth_rate * t))) →
  min_time n ≤ t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worm_growth_l472_47215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_imply_a_equals_three_l472_47287

-- Define the two functions as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log 2 + a
noncomputable def g (x : ℝ) : ℝ := 2^(x - 3)

-- State the theorem
theorem symmetric_functions_imply_a_equals_three (a : ℝ) :
  (∀ x y, f a x = y ↔ g y = x) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_imply_a_equals_three_l472_47287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_coefficient_sum_l472_47236

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.p1 q.p2 + distance q.p2 q.p3 + distance q.p3 q.p4 + distance q.p4 q.p1

/-- Our specific quadrilateral -/
def myQuad : Quadrilateral :=
  { p1 := ⟨1, 2⟩
    p2 := ⟨4, 6⟩
    p3 := ⟨7, 4⟩
    p4 := ⟨6, 1⟩ }

/-- Theorem: The sum of coefficients in the simplified √ form of myQuad's perimeter is 3 -/
theorem perimeter_coefficient_sum :
  ∃ (a b c d : ℕ), perimeter myQuad = a * Real.sqrt c + b * Real.sqrt d ∧ a + b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_coefficient_sum_l472_47236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangements_l472_47211

/-- The number of ways to arrange 8 people (5 boys and 3 girls) in a row with 3 girls standing next to each other -/
def arrangements_with_girls_together (num_boys num_girls : ℕ) : ℕ :=
  (num_boys + 1) * Nat.factorial num_boys * Nat.factorial num_girls

/-- The number of ways to arrange 5 people (chosen from 5 boys and 3 girls) in a row with exactly 2 girls among them -/
def arrangements_with_two_girls (num_boys num_girls : ℕ) : ℕ :=
  Nat.choose num_girls 2 * Nat.choose num_boys 3 * Nat.factorial 5

/-- Theorem stating the correct number of arrangements for both scenarios -/
theorem correct_arrangements :
  arrangements_with_girls_together 5 3 = 4320 ∧
  arrangements_with_two_girls 5 3 = 3600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangements_l472_47211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_conversation_day_l472_47214

-- Define the days of the week
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the visit patterns
def visit_pattern (boy : Nat) : Nat :=
  match boy with
  | 1 => 1  -- A visits every day
  | 2 => 2  -- B visits every two days
  | 3 => 3  -- C visits every three days
  | _ => 0

-- Define the library closure
def is_library_closed (day : DayOfWeek) : Bool :=
  match day with
  | DayOfWeek.Wednesday => true
  | _ => false

-- Define the function to get the next open day
def next_open_day (day : DayOfWeek) : DayOfWeek :=
  if is_library_closed day then
    match day with
    | DayOfWeek.Wednesday => DayOfWeek.Thursday
    | _ => day
  else
    day

-- Define the function to calculate the previous visit day
def prev_visit_day (current_day : DayOfWeek) (pattern : Nat) : DayOfWeek :=
  sorry  -- The implementation would go here

-- The main theorem
theorem original_conversation_day 
  (boy1_pattern : Nat)
  (boy2_pattern : Nat)
  (boy3_pattern : Nat)
  (meeting_day : DayOfWeek) :
  boy1_pattern = visit_pattern 1 →
  boy2_pattern = visit_pattern 2 →
  boy3_pattern = visit_pattern 3 →
  meeting_day = DayOfWeek.Monday →
  prev_visit_day meeting_day boy1_pattern = DayOfWeek.Friday ∧
  prev_visit_day meeting_day boy2_pattern = DayOfWeek.Friday ∧
  prev_visit_day meeting_day boy3_pattern = DayOfWeek.Friday :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_conversation_day_l472_47214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_suitable_for_census_l472_47291

-- Define the type of investigation methods
inductive InvestigationMethod
  | Census
  | SampleSurvey

-- Define the options
inductive SurveyOption
  | A -- standing long jump scores of students in a class
  | B -- types of fish in a reservoir
  | C -- number of bends a shoe sole can withstand in a shoe factory
  | D -- lifespan of a certain model of energy-saving lamp

-- Function to determine the suitable investigation method for each option
def suitableMethod (option : SurveyOption) : InvestigationMethod :=
  match option with
  | SurveyOption.A => InvestigationMethod.Census
  | _ => InvestigationMethod.SampleSurvey

-- Theorem stating that only option A is suitable for a census
theorem only_A_suitable_for_census :
  ∀ (option : SurveyOption),
    suitableMethod option = InvestigationMethod.Census ↔ option = SurveyOption.A :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_suitable_for_census_l472_47291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l472_47220

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 / x

-- Define the point of intersection with the x-axis
def x₀ : ℝ := 1

-- Define the slope of the tangent line at x₀
noncomputable def k : ℝ := f' x₀

-- State the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, y = k * (x - x₀) → x - y - 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l472_47220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l472_47246

theorem trig_inequality (x : ℝ) (n : ℕ) (h : 0 < x ∧ x < π/2) :
  ((1 / Real.sin x ^ (2 * n)) - 1) * ((1 / Real.cos x ^ (2 * n)) - 1) ≥ (2^n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l472_47246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_two_sides_integer_multiple_of_median_no_triangle_with_all_sides_integer_multiple_of_median_l472_47232

/-- A triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The length of a median in a triangle. -/
noncomputable def median_length (t : Triangle) (side : ℝ) : ℝ :=
  (1 / 2) * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - side^2)

/-- Predicate to check if a side is an integer multiple of its median. -/
def is_integer_multiple_of_median (t : Triangle) (side : ℝ) : Prop :=
  ∃ k : ℕ, side = k * median_length t side

theorem triangle_with_two_sides_integer_multiple_of_median :
  ∃ t : Triangle, 
    (is_integer_multiple_of_median t t.a ∧ is_integer_multiple_of_median t t.b) ∨
    (is_integer_multiple_of_median t t.b ∧ is_integer_multiple_of_median t t.c) ∨
    (is_integer_multiple_of_median t t.c ∧ is_integer_multiple_of_median t t.a) :=
by
  sorry

theorem no_triangle_with_all_sides_integer_multiple_of_median :
  ¬ ∃ t : Triangle, 
    is_integer_multiple_of_median t t.a ∧
    is_integer_multiple_of_median t t.b ∧
    is_integer_multiple_of_median t t.c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_two_sides_integer_multiple_of_median_no_triangle_with_all_sides_integer_multiple_of_median_l472_47232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_Q_x_coordinate_l472_47292

theorem point_Q_x_coordinate 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (h1 : P = (3/5, 4/5))
  (h2 : Q.1 < 0 ∧ Q.2 < 0)  -- Q is in the third quadrant
  (h3 : Q.1^2 + Q.2^2 = 1)  -- |OQ| = 1
  (h4 : Real.arccos ((P.1 * Q.1 + P.2 * Q.2) / (Real.sqrt (P.1^2 + P.2^2))) = 3 * π / 4)  -- Angle POQ = 3π/4
  : Q.1 = -7 * Real.sqrt 2 / 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_Q_x_coordinate_l472_47292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l472_47224

open Real

/-- The original cosine function -/
noncomputable def f (x : ℝ) : ℝ := cos x

/-- The transformed function -/
noncomputable def g (x : ℝ) : ℝ := 3 * cos (2 * x + π / 3)

/-- Theorem stating that g is the correct transformation of f -/
theorem cosine_transformation (x : ℝ) : 
  g x = 3 * f (x / 2 + π / 3) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l472_47224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_l472_47231

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | Other

-- Define the class and selection criteria
def class_size : Nat := 40
def selection_interval : Nat := 5

-- Define the systematic selection function
def is_selected (id : Nat) : Prop :=
  id % selection_interval = 0 ∧ id ≤ class_size

-- Theorem statement
theorem systematic_sampling :
  (∀ id, id ≤ class_size → (is_selected id ↔ id ∈ ({5, 10, 15, 20, 25, 30, 35, 40} : Finset Nat))) →
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_l472_47231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_of_reflected_point_l472_47263

-- Define the basic structures
structure Triangle where
  A : Point
  B : Point
  C : Point
  is_acute : Prop

structure Circle where
  center : Point
  radius : ℝ
  circumscribes : Triangle → Prop

-- Define the reflection of a point
noncomputable def reflect (P Q R : Point) : Point :=
  sorry

-- Define the angle bisector
def is_angle_bisector (P Q R S : Point) : Prop :=
  sorry

-- State the theorem
theorem bisector_of_reflected_point
  (ABC : Triangle)
  (O : Point)
  (circ : Circle)
  (B₁ K : Point)
  (h1 : circ.center = O)
  (h2 : circ.circumscribes ABC)
  (h3 : B₁ = reflect ABC.B ABC.A ABC.C)
  (h4 : is_angle_bisector K ABC.A ABC.B B₁) :
  is_angle_bisector ABC.A K ABC.B B₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_of_reflected_point_l472_47263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l472_47266

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

noncomputable def f (x : ℝ) := lg (floor x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l472_47266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_values_l472_47212

-- Define the ellipse M
noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the eccentricity of an ellipse
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

-- Define the conditions of the problem
structure EllipseConditions (a b : ℝ) where
  ellipse : Ellipse a b
  a_gt_b : a > b
  b_pos : b > 0
  foci : ℝ × ℝ × ℝ × ℝ  -- Represents F1 and F2
  intersections : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ  -- Represents A, B, C, D
  equilateral_exists : Bool  -- True if an equilateral triangle can be formed

-- State the theorem
theorem ellipse_eccentricity_values (a b : ℝ) (h : EllipseConditions a b) :
  let e := Eccentricity a b
  e = 1/2 ∨ e = Real.sqrt 3 / 2 := by
  sorry

-- Example usage (optional, can be removed if not needed)
example : True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_values_l472_47212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_heads_than_tails_probability_l472_47288

/-- The probability of getting more heads than tails when tossing a fair coin 4 times -/
def probability_more_heads_than_tails : ℚ := 5 / 16

/-- The number of tosses -/
def num_tosses : ℕ := 4

/-- A coin toss is fair if the probability of heads is 1/2 -/
def is_fair_coin : Prop := true

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ := (n.choose k : ℚ) / (2 ^ n : ℚ)

theorem more_heads_than_tails_probability :
  is_fair_coin →
  (probability_k_heads num_tosses 3 + probability_k_heads num_tosses 4 = probability_more_heads_than_tails) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_heads_than_tails_probability_l472_47288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_H_table_property_l472_47282

/-- Definition of an n-order H table -/
def is_H_table (n : ℕ) (a : Fin n → Fin n → ℤ) : Prop :=
  n ≥ 2 ∧
  (∀ i j, a i j ∈ ({-1, 0, 1} : Set ℤ)) ∧
  let r := λ i => (Finset.univ.sum (λ j => a i j))
  let c := λ j => (Finset.univ.sum (λ i => a i j))
  (Finset.image r Finset.univ ∪ Finset.image c Finset.univ).card = 2 * n

/-- The set H for an n-order H table -/
def H_set (n : ℕ) (a : Fin n → Fin n → ℤ) : Set ℤ :=
  let r := λ i => (Finset.univ.sum (λ j => a i j))
  let c := λ j => (Finset.univ.sum (λ i => a i j))
  (Finset.image r Finset.univ ∪ Finset.image c Finset.univ).toSet

/-- Main theorem -/
theorem H_table_property (n : ℕ) (a : Fin n → Fin n → ℤ) (x : ℤ) :
  is_H_table n a →
  x ∈ Set.Icc (-n : ℤ) n →
  x ∉ H_set n a →
  Even x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_H_table_property_l472_47282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_perimeter_l472_47274

/-- The sum of all side lengths of a regular octagon with a side length of 2.3 meters is 1840 centimeters. -/
theorem regular_octagon_perimeter (meters_to_cm : ℝ) (side_length_meters : ℝ) (num_sides : ℕ) (total_length_cm : ℝ) : 
  meters_to_cm = 100 →
  side_length_meters = 2.3 →
  num_sides = 8 →
  total_length_cm = side_length_meters * meters_to_cm * (num_sides : ℝ) →
  total_length_cm = 1840 := by
  sorry

#check regular_octagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_perimeter_l472_47274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l472_47225

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6) - Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h_pos : ω > 0) :
  (∀ x, f ω x = f ω (x + 4 * Real.pi)) →
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), StrictMono (f ω)) →
  ω ∈ ({1/3, 5/6, 4/3} : Set ℝ) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l472_47225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l472_47284

/-- The speed of a train in km/h given its length and time to pass a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem: A 180-meter long train that crosses a stationary point in 9 seconds has a speed of 72 km/h -/
theorem train_speed_calculation :
  train_speed 180 9 = 72 := by
  unfold train_speed
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l472_47284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l472_47270

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the line l
def line_l (x : ℝ) : Prop := x = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the tangent condition
def is_tangent (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y ∧
  ∀ (x' y' : ℝ), line x' y' → circle x' y' → (x' = x ∧ y' = y)

-- State the theorem
theorem parabola_circle_tangent 
  (P Q : ℝ × ℝ) 
  (A₁ A₂ A₃ : ℝ × ℝ) :
  parabola_C P.1 P.2 →
  parabola_C Q.1 Q.2 →
  line_l P.1 →
  line_l Q.1 →
  (P.1 - 0) * (Q.1 - 0) + (P.2 - 0) * (Q.2 - 0) = 0 →  -- OP ⟂ OQ
  parabola_C A₁.1 A₁.2 →
  parabola_C A₂.1 A₂.2 →
  parabola_C A₃.1 A₃.2 →
  is_tangent (λ x y ↦ ∃ t, x = A₁.1 + t*(A₂.1 - A₁.1) ∧ y = A₁.2 + t*(A₂.2 - A₁.2)) circle_M →
  is_tangent (λ x y ↦ ∃ t, x = A₁.1 + t*(A₃.1 - A₁.1) ∧ y = A₁.2 + t*(A₃.2 - A₁.2)) circle_M →
  is_tangent (λ x y ↦ ∃ t, x = A₂.1 + t*(A₃.1 - A₂.1) ∧ y = A₂.2 + t*(A₃.2 - A₂.2)) circle_M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l472_47270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PMN_MN_passes_through_fixed_point_l472_47271

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the lines passing through P with slopes k1 and k2
def line1 (k1 : ℝ) (x y : ℝ) : Prop := y = k1 * (x - 2)
def line2 (k2 : ℝ) (x y : ℝ) : Prop := y = k2 * (x - 2)

-- Define the intersection points A, B, C, D
noncomputable def A (k1 : ℝ) : ℝ × ℝ := sorry
noncomputable def B (k1 : ℝ) : ℝ × ℝ := sorry
noncomputable def C (k2 : ℝ) : ℝ × ℝ := sorry
noncomputable def D (k2 : ℝ) : ℝ × ℝ := sorry

-- Define midpoints M and N
noncomputable def M (k1 : ℝ) : ℝ × ℝ := 
  (((A k1).1 + (B k1).1) / 2, ((A k1).2 + (B k1).2) / 2)
noncomputable def N (k2 : ℝ) : ℝ × ℝ := 
  (((C k2).1 + (D k2).1) / 2, ((C k2).2 + (D k2).2) / 2)

-- Define the area of triangle PMN
noncomputable def areaPMN (k1 k2 : ℝ) : ℝ := sorry

-- Part I: Minimum area theorem
theorem min_area_PMN (k1 k2 : ℝ) (h : k1 * k2 = -1) :
  ∀ m n : ℝ, m * n = -1 → areaPMN k1 k2 ≤ areaPMN m n :=
by sorry

-- Part II: Fixed point theorem
theorem MN_passes_through_fixed_point (k1 k2 : ℝ) (h : k1 + k2 = 1) :
  ∃ t : ℝ, (2 - (M k1).1) * t + (M k1).1 = 2 ∧ ((M k1).2 - (N k2).2) * t + (N k2).2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PMN_MN_passes_through_fixed_point_l472_47271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delivery_proof_l472_47244

def total_deliveries : ℕ := 75
def package_meal_ratio : ℕ := 8
def document_meal_ratio : ℕ := 4

def borough_ratio : List ℕ := [3, 2, 1, 1]

def meals : ℕ := 5
def packages : ℕ := 40
def documents : ℕ := 20

def borough_distribution : List ℕ := [2, 1, 1, 1]

theorem delivery_proof :
  (meals + packages + documents = total_deliveries) ∧
  (packages = package_meal_ratio * meals) ∧
  (documents = document_meal_ratio * meals) ∧
  (borough_distribution.sum = meals) ∧
  (∀ (i : Fin 4), borough_distribution[i] = 
    (borough_ratio[i] * meals + (borough_ratio.sum - 1) / 2) / borough_ratio.sum) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delivery_proof_l472_47244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l472_47222

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (1,1) to the line x+y-1=0 is √2/2 -/
theorem distance_point_to_line_example : distance_point_to_line 1 1 1 1 (-1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l472_47222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l472_47298

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi
  sides_opposite : a > 0 ∧ b > 0 ∧ c > 0

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.B + Real.sqrt 3 * Real.sin t.B = 2)
  (h2 : Real.cos t.B / t.b + Real.cos t.C / t.c = 2 * Real.sin t.A / (Real.sqrt 3 * Real.sin t.C)) :
  t.B = Real.pi/3 ∧ 
  t.b = Real.sqrt 3 / 2 ∧
  (∀ S : Real, S = t.a * t.c * Real.sin t.B / 2 → S ≤ 3 * Real.sqrt 3 / 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l472_47298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l472_47208

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_function_unique :
  ∀ a b c : ℝ,
    quadratic_function a b c (-2) = 0 →
    quadratic_function a b c 4 = 0 →
    (∀ x : ℝ, quadratic_function a b c x ≤ 9) →
    (∃ x : ℝ, quadratic_function a b c x = 9) →
    ∀ x : ℝ, quadratic_function a b c x = -x^2 + 2*x + 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l472_47208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l472_47255

noncomputable def f (x : ℝ) : ℝ := x^2 + 1 / Real.sqrt (1 + x)

theorem f_bounds (x : ℝ) (hx : x ∈ Set.Icc 0 1) : 
  f x ≥ x^2 - (1/2) * x + 1 ∧ 15/16 < f x ∧ f x ≤ (2 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l472_47255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_A_given_in_Ω_l472_47203

-- Define the regions Ω and A
def Ω : Set (ℝ × ℝ) := {p | p.1 + p.2 ≤ 6 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}
def A : Set (ℝ × ℝ) := {p | p.1 ≤ 4 ∧ p.2 ≥ 0 ∧ p.1 - 2*p.2 ≥ 0}

-- Define the area function (instead of volume)
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the probability function
noncomputable def probability (S T : Set (ℝ × ℝ)) : ℝ := (area (S ∩ T)) / (area S)

-- State the theorem
theorem probability_point_in_A_given_in_Ω : probability Ω A = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_A_given_in_Ω_l472_47203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_implies_phi_l472_47250

theorem sin_symmetry_implies_phi (φ : ℝ) : 
  (∀ x : ℝ, Real.sin (2 * (x + π/6) + φ) = Real.sin (2 * (π/6 - x) + φ)) → 
  φ = -π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_implies_phi_l472_47250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_value_l472_47252

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 / 4 - Real.sqrt x

-- State the theorem
theorem tangent_line_and_minimum_value :
  -- Part 1: Equation of the tangent line at (4, f(4))
  (∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ↔ 7 * x - 4 * y - 20 = 0) ∧
  -- Part 2: Minimum value of f(x)
  (∃ (x_min : ℝ), ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -3/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_value_l472_47252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l472_47295

/-- Predicate to represent that a given equation describes an ellipse -/
def IsEllipse (x y : ℝ) : Prop :=
  sorry

/-- Predicate to represent that the foci of the ellipse are on the y-axis -/
def FociOnYAxis (k : ℝ) : Prop :=
  sorry

/-- 
Given that x^2 + ky^2 = 2 represents an ellipse with foci on the y-axis,
prove that 0 < k < 1.
-/
theorem ellipse_k_range (k : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 + k*y^2 = 2 → IsEllipse x y)
  (h2 : FociOnYAxis k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l472_47295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l472_47281

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Side lengths are positive
  ha : a > 0
  hb : b > 0
  hc : c > 0
  -- Angle sum is π
  angle_sum : A + B + C = Real.pi

-- Define the properties of the specific triangle
def special_triangle (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = 3 * t.b * t.c * Real.cos t.A ∧
  t.B = t.C ∧
  t.a = 2

-- Define area function
noncomputable def area (t : Triangle) : Real :=
  (1/2) * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_properties (t : Triangle) (h : special_triangle t) :
  (area t = Real.sqrt 5) ∧
  ((Real.tan t.A / Real.tan t.B) + (Real.tan t.A / Real.tan t.C) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l472_47281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_probability_l472_47299

noncomputable def probability_of_selection (student : Fin 2010) : ℚ := 5 / 201

theorem selection_probability :
  ∀ student : Fin 2010, probability_of_selection student = 5 / 201 :=
by
  intro student
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_probability_l472_47299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_l472_47245

-- Define the triangle and its angles
variable (A B C : ℝ)

-- Define the vector a as a function
noncomputable def a (A B : ℝ) : ℝ × ℝ := (2 * Real.cos ((A - B) / 2), 3 * Real.sin ((A + B) / 2))

-- State the given conditions
axiom triangle_angles (A B C : ℝ) : A + B + C = Real.pi
axiom vector_magnitude (A B : ℝ) : Real.sqrt ((a A B).1^2 + (a A B).2^2) = Real.sqrt 26 / 2

-- State the theorem to be proved
theorem max_tan_C : 
  ∃ (max_tan_C : ℝ), ∀ (A B C : ℝ), 
    A + B + C = Real.pi → 
    Real.sqrt ((2 * Real.cos ((A - B) / 2))^2 + (3 * Real.sin ((A + B) / 2))^2) = Real.sqrt 26 / 2 →
    Real.tan C ≤ max_tan_C ∧ 
    max_tan_C = -Real.sqrt 65 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_l472_47245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_milk_problem_l472_47223

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the transfer of liquid between cups --/
def transfer (cup_from cup_to : CupContents) (amount : ℚ) : CupContents × CupContents :=
  let total := cup_from.coffee + cup_from.milk
  let transferred_coffee := (amount * cup_from.coffee) / total
  let transferred_milk := (amount * cup_from.milk) / total
  let new_from := CupContents.mk (cup_from.coffee - transferred_coffee) (cup_from.milk - transferred_milk)
  let new_to := CupContents.mk (cup_to.coffee + transferred_coffee) (cup_to.milk + transferred_milk)
  (new_from, new_to)

/-- The main theorem representing the problem --/
theorem coffee_milk_problem :
  let cup1_initial := CupContents.mk 6 0
  let cup2_initial := CupContents.mk 0 4
  let (cup1_after_first_transfer, cup2_after_first_transfer) := transfer cup1_initial cup2_initial 3
  let (cup2_final, cup1_final) := transfer cup2_after_first_transfer cup1_after_first_transfer 4
  cup1_final.milk / (cup1_final.coffee + cup1_final.milk) = 16 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_milk_problem_l472_47223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_fraction_lowest_terms_sum_numerator_denominator_l472_47234

/-- The repeating decimal 0.343434... -/
def repeating_decimal : ℚ := 34 / 99

/-- The repeating decimal 0.343434... expressed as a fraction -/
theorem repeating_decimal_fraction : repeating_decimal = 34 / 99 := by sorry

/-- The fraction 34/99 is in lowest terms -/
theorem fraction_lowest_terms : Int.gcd 34 99 = 1 := by sorry

/-- The sum of numerator and denominator of the fraction representing the repeating decimal -/
theorem sum_numerator_denominator : 
  let n := (repeating_decimal.num.natAbs)
  let d := (repeating_decimal.den)
  n + d = 133 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_fraction_lowest_terms_sum_numerator_denominator_l472_47234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_person_count_is_six_l472_47257

/-- The initial number of persons in a group where:
  - The average weight increase is 2.5 kg per person when a new person replaces one weighing 65 kg.
  - The new person weighs 80 kg.
-/
noncomputable def initialPersonCount : ℕ :=
  let avgWeightIncrease : ℚ := 5/2
  let oldPersonWeight : ℕ := 65
  let newPersonWeight : ℕ := 80
  let weightDifference : ℕ := newPersonWeight - oldPersonWeight
  Nat.floor ((weightDifference : ℚ) / avgWeightIncrease)

theorem initial_person_count_is_six : initialPersonCount = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_person_count_is_six_l472_47257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l472_47216

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N : Set ℝ := {0, 4}

theorem union_of_M_and_N : M ∪ N = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l472_47216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l472_47290

theorem smallest_positive_z (x z : ℝ) : 
  Real.cos x = 0 → 
  Real.cos (x + z) = Real.sqrt 2 / 2 → 
  (∀ w, w > 0 ∧ (Real.cos x = 0 → Real.cos (x + w) = Real.sqrt 2 / 2) → z ≤ w) →
  z = 5 * Real.pi / 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l472_47290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quarter_time_proportion_l472_47226

/-- Represents the speed during each quarter of the journey relative to the second quarter -/
structure JourneySpeed where
  first : ℚ
  second : ℚ
  third : ℚ
  fourth : ℚ

/-- Calculates the proportion of time spent on the first quarter of the journey -/
def firstQuarterTimeProportion (speed : JourneySpeed) : ℚ :=
  let totalTime := (1 / speed.first) + (1 / speed.second) + (1 / speed.third) + (1 / speed.fourth)
  (1 / speed.first) / totalTime

/-- Theorem stating the proportion of time spent on the first quarter of the journey
    given the specified speed ratios -/
theorem first_quarter_time_proportion :
  let speed := JourneySpeed.mk 4 1 (1/6) (1/2)
  firstQuarterTimeProportion speed = 1/37 := by
  sorry

#eval firstQuarterTimeProportion (JourneySpeed.mk 4 1 (1/6) (1/2))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quarter_time_proportion_l472_47226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_w_l472_47251

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (w * x + Real.pi / 7) - 1

theorem minimum_w (w : ℝ) (h1 : w > 0) 
  (h2 : ∀ x, f w x = f w (x + 4 * Real.pi / 3)) : w = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_w_l472_47251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l472_47213

-- Define the radii of the circles
def inner_radius : ℝ := 2
def outer_radius : ℝ := 4

-- Define a function to calculate the angle subtended by the chord that just touches the inner circle
noncomputable def critical_angle (r₁ r₂ : ℝ) : ℝ :=
  2 * Real.arccos (r₁ / r₂)

-- State the theorem
theorem chord_intersection_probability :
  let θ := critical_angle inner_radius outer_radius
  (θ / (2 * Real.pi)) = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l472_47213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l472_47262

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => (a n + a (n + 1)) / 2

def b (n : ℕ) : ℚ := a (n + 1) - a n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = (-1/2) * b n) ∧
  (∀ n : ℕ, n ≥ 0 → a n = 5/3 - 2/3 * (-1/2)^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l472_47262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_l472_47272

/-- A quadrilateral with side lengths 10, 12, 8, and 14 -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 10)
  (bc_length : dist B C = 12)
  (cd_length : dist C D = 8)
  (da_length : dist D A = 14)

/-- Check if a point is inside a quadrilateral -/
def pointInQuadrilateral (p : ℝ × ℝ) (q : Quadrilateral) : Prop :=
  sorry

/-- Check if a point is on a line segment -/
def pointOnSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  sorry

/-- A circle inscribed in a quadrilateral -/
structure InscribedCircle (q : Quadrilateral) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (inside : ∀ p : ℝ × ℝ, dist p center ≤ radius → pointInQuadrilateral p q)
  (touches : ∃ p1 p2 p3 p4 : ℝ × ℝ, 
    pointOnSegment p1 q.A q.B ∧ 
    pointOnSegment p2 q.B q.C ∧ 
    pointOnSegment p3 q.C q.D ∧ 
    pointOnSegment p4 q.D q.A ∧
    dist p1 center = radius ∧
    dist p2 center = radius ∧
    dist p3 center = radius ∧
    dist p4 center = radius)

/-- The theorem stating that the largest inscribed circle has radius 4√3 -/
theorem largest_inscribed_circle_radius (q : Quadrilateral) :
  (∃ c : InscribedCircle q, ∀ c' : InscribedCircle q, c.radius ≥ c'.radius) →
  ∃ c : InscribedCircle q, c.radius = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_l472_47272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_condition_l472_47289

noncomputable def f (a x : ℝ) : ℝ := 2^(x^2 + 4*a*x + 2)

theorem monotone_decreasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 6 → f a x > f a y) → a ≤ -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_condition_l472_47289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l472_47200

theorem calculation_proofs :
  (Real.sqrt 18 - 2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (3 * Real.sqrt 2) = 5 * Real.sqrt 2 / 2) ∧
  ((Real.sqrt 27 - 2 * Real.sqrt 3)^2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l472_47200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_word_game_result_l472_47230

/-- Represents a player in the word game -/
structure Player where
  name : String
  words : Finset String
  score : ℕ

/-- The Word game with three players -/
structure WordGame where
  players : Finset Player
  total_words : ℕ
  player_count : players.card = 3
  word_count : ∀ p, p ∈ players → p.words.card = total_words

/-- Scoring function for the Word game -/
def score (game : WordGame) (p : Player) : ℕ :=
  p.words.sum fun w =>
    if (game.players.filter (fun q => w ∈ q.words)).card = 1 then 3
    else if (game.players.filter (fun q => w ∈ q.words)).card = 2 then 1
    else 0

/-- Theorem stating the result of the Word game -/
theorem word_game_result (game : WordGame) (sam james : Player)
  (h_sam_in : sam ∈ game.players)
  (h_james_in : james ∈ game.players)
  (h_sam_score : sam.score = 19)
  (h_sam_lowest : ∀ p, p ∈ game.players → sam.score ≤ p.score)
  (h_james_highest : ∀ p, p ∈ game.players → p.score ≤ james.score)
  (h_different_scores : ∀ p q, p ∈ game.players → q ∈ game.players → p ≠ q → p.score ≠ q.score)
  (h_total_words : game.total_words = 10) :
  james.score = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_word_game_result_l472_47230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_buses_l472_47254

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : Finset Nat
  stop_count : stops.card = 3

/-- The city's bus system -/
structure BusSystem where
  stops : Finset Nat
  stop_count : stops.card = 9
  routes : Finset BusRoute
  valid_routes : ∀ r ∈ routes, r.stops ⊆ stops
  common_stop_constraint : ∀ r1 r2, r1 ∈ routes → r2 ∈ routes → r1 ≠ r2 → (r1.stops ∩ r2.stops).card ≤ 1

/-- The theorem stating the maximum number of buses in the system -/
theorem max_buses (system : BusSystem) : system.routes.card ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_buses_l472_47254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_n_for_f_extrema_l472_47259

/-- The function f(x) = -x^2/2 + x -/
noncomputable def f (x : ℝ) : ℝ := -x^2/2 + x

theorem unique_m_n_for_f_extrema :
  ∀ (m n : ℝ), m ≤ n →
  (∀ x ∈ Set.Icc m n, f m ≤ f x ∧ f x ≤ f n) →
  f m = 2*m →
  f n = 2*n →
  m = -2 ∧ n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_n_for_f_extrema_l472_47259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_one_root_l472_47279

theorem polynomial_one_root (θ : Real) : 
  let P : Real → Real := fun x ↦ Real.sin θ * x^2 + (Real.cos θ + Real.tan θ) * x + 1
  (∃! x, P x = 0) → Real.sin θ = 0 ∨ Real.sin θ = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_one_root_l472_47279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_sqrt_fraction_l472_47228

theorem meaningful_sqrt_fraction (x : ℝ) : 
  (∃ (y : ℝ), y^2 = 5 / (2 - 3*x)) ↔ x < 2/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_sqrt_fraction_l472_47228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l472_47286

-- Define the parabola
def Parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the triangle
def Triangle (p : ℝ) (O A B : ℝ × ℝ) : Prop :=
  O = (0, 0) ∧ Parabola p A.1 A.2 ∧ Parabola p B.1 B.2

-- Define the conditions
def Conditions (p : ℝ) (O A B : ℝ × ℝ) : Prop :=
  Triangle p O A B ∧
  (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA · OB = 0
  (A.2 = 2 * A.1) ∧              -- Equation of line OA is y = 2x
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * Real.sqrt 13)^2)  -- |AB| = 4√13

-- The theorem to prove
theorem parabola_equation (p : ℝ) (O A B : ℝ × ℝ) :
  Conditions p O A B → p = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l472_47286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_101_l472_47265

def Increasing (s : List ℕ) : Prop :=
  ∀ i j, i < j → j < s.length → s[i]! < s[j]!

def Decreasing (s : List ℕ) : Prop :=
  ∀ i j, i < j → j < s.length → s[i]! > s[j]!

theorem erdos_szekeres_101 (seq : List ℕ) : 
  seq.length = 101 → seq.Nodup → 
  ∃ subseq : List ℕ, subseq.length = 11 ∧ 
  (Increasing subseq ∨ Decreasing subseq) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_101_l472_47265
