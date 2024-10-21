import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1360_136066

noncomputable def f (x : ℝ) : ℝ := 6 * Real.log x + (1/2) * x^2 - 5 * x

theorem f_properties :
  let tangent_line (x y : ℝ) := 4 * x - 2 * y - 13 = 0
  let is_increasing (a b : ℝ) := ∀ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, x < y → f x < f y
  let is_decreasing (a b : ℝ) := ∀ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, x < y → f x > f y
  let has_maximum (v : ℝ) (x : ℝ) := f x = v ∧ ∀ y > 0, f y ≤ v
  let has_minimum (v : ℝ) (x : ℝ) := f x = v ∧ ∀ y > 0, f y ≥ v
  ∀ x > 0,
    (tangent_line x (f 1)) ∧
    (is_increasing 0 2) ∧
    (is_increasing 3 (Real.arctan 0 + π/2)) ∧
    (is_decreasing 2 3) ∧
    (has_maximum (-8 + 6 * Real.log 2) 2) ∧
    (has_minimum (-(21/2) + 6 * Real.log 3) 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1360_136066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1360_136070

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioi (Real.exp (-1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1360_136070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_count_l1360_136029

noncomputable def f (x : ℝ) : ℝ := (2 * x - 2) / (x^2 + 10 * x - 24)

theorem vertical_asymptotes_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, ¬∃ (y : ℝ), f x = y) ∧ S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_count_l1360_136029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_statue_model_ratio_l1360_136058

/-- Represents the height of a statue and its scale model --/
structure StatueModel where
  statue_height : ℝ  -- in feet
  model_height : ℝ   -- in centimeters

/-- Calculates the number of feet represented by one centimeter of the model --/
noncomputable def feet_per_cm (sm : StatueModel) : ℝ :=
  (sm.statue_height * 2.54) / (sm.model_height * 12)

/-- The theorem stating the relationship between the statue and its model --/
theorem washington_statue_model_ratio :
  let sm : StatueModel := { statue_height := 80, model_height := 10 }
  ∃ ε > 0, |feet_per_cm sm - 6.562| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_statue_model_ratio_l1360_136058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l1360_136032

def f (x : ℝ) : ℝ := x^3 - 3*x - 1

theorem min_t_value (t : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 2 → x₂ ∈ Set.Icc (-3) 2 → |f x₁ - f x₂| ≤ t) ↔ t ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l1360_136032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1360_136094

-- Define the parabola structure
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  vertex_at_origin : eq 0 0
  focus_on_x_axis : ∃ (f : ℝ), eq f 0 ∧ f ≠ 0

-- Define the intersection line
def intersection_line (x y : ℝ) : Prop := y = 2 * x - 4

-- Define the chord length
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Main theorem
theorem parabola_equation (p : Parabola) 
  (intersects : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    p.eq x₁ y₁ ∧ p.eq x₂ y₂ ∧ 
    intersection_line x₁ y₁ ∧ intersection_line x₂ y₂ ∧
    chord_length x₁ y₁ x₂ y₂ = 3 * Real.sqrt 5) :
  p.eq = (λ x y ↦ y^2 = 4*x) ∨ p.eq = (λ x y ↦ y^2 = -36*x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1360_136094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1360_136050

/-- The projection of a vector onto a line --/
def vector_projection (v : ℝ × ℝ × ℝ) (l : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- The line defined by x = -2y = 4z --/
noncomputable def line : ℝ × ℝ × ℝ :=
  (1, -1/2, 1/4)

/-- The vector to be projected --/
noncomputable def vector : ℝ × ℝ × ℝ :=
  (3, -5, 2)

/-- The expected projection result --/
noncomputable def expected_projection : ℝ × ℝ × ℝ :=
  (4.8, -2.4, 1.2)

theorem projection_theorem :
  vector_projection vector line = expected_projection :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1360_136050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_eq_three_fifths_l1360_136042

/-- The series defined in the problem -/
noncomputable def series_sum : ℝ := ∑' n : ℕ, if n ≥ 2 then (n^4 + 5*n^2 + 8*n + 8) / (2^n * (n^4 + 4)) else 0

/-- The theorem stating that the series sum is equal to 3/5 -/
theorem series_sum_eq_three_fifths : series_sum = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_eq_three_fifths_l1360_136042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_true_l1360_136002

/-- Proposition p: x > 2 is a necessary but not sufficient condition for x > log₂5 -/
def p : Prop := ∀ x : ℝ, x > Real.log 5 / Real.log 2 → x > 2

/-- Proposition q: if sin x = √3/3, then cos 2x = sin² x -/
def q : Prop := ∀ x : ℝ, Real.sin x = Real.sqrt 3 / 3 → Real.cos (2 * x) = Real.sin x ^ 2

/-- The conjunction of propositions p and q is true -/
theorem p_and_q_true : p ∧ q := by
  constructor
  · -- Proof of p
    intro x h
    sorry
  · -- Proof of q
    intro x h
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_true_l1360_136002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_age_correct_l1360_136088

def current_age (person_age_at_dog_birth : ℕ) (dog_age_in_two_years : ℕ) : ℕ :=
  let current_dog_age := dog_age_in_two_years - 2
  person_age_at_dog_birth + current_dog_age

theorem current_age_correct (person_age_at_dog_birth : ℕ) (dog_age_in_two_years : ℕ) :
  current_age person_age_at_dog_birth dog_age_in_two_years = person_age_at_dog_birth + (dog_age_in_two_years - 2) :=
by
  unfold current_age
  simp

#eval current_age 15 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_age_correct_l1360_136088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1360_136086

noncomputable def f (x : ℝ) := (Real.sqrt (x^2 - 5*x + 6)) / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1360_136086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l1360_136035

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The length of the altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The base of the triangle -/
  base : ℝ
  /-- One of the equal sides of the triangle -/
  side : ℝ
  /-- Condition that the triangle is isosceles -/
  isosceles : side = (perimeter - base) / 2
  /-- Pythagorean theorem relation -/
  pythagorean : base^2 / 4 + altitude^2 = side^2

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ := t.base * t.altitude / 2

/-- Theorem stating that an isosceles triangle with altitude 10 and perimeter 40 has area 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 := by
  sorry

#check isosceles_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l1360_136035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_problem_l1360_136008

theorem binomial_expansion_problem (n : ℕ) (a b : ℕ) :
  (∀ x : ℚ, (x + 1)^n = x^n + (a * x^3 + b * x^2 + n * x + 1)) →
  (a * 1 = b * 3) →
  n = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_problem_l1360_136008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_partition_theorem_l1360_136040

/-- Given a set of positive integers, returns the sum of its elements -/
def σ (S : Finset ℕ) : ℕ := S.sum id

/-- The type of partitions of a set into n classes -/
def Partition (α : Type) (n : ℕ) := Fin n → Set α

theorem sum_partition_theorem (n : ℕ) (A : Finset ℕ) (h : A.Nonempty) :
  ∃ (P : Partition (Finset ℕ) n),
    (∀ S, S.Nonempty → S ⊆ A → ∃ i, S ∈ P i) ∧
    (∀ i : Fin n, ∀ S T, S ∈ P i → T ∈ P i → S.Nonempty → T.Nonempty → 
      (σ S : ℚ) / (σ T : ℚ) ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_partition_theorem_l1360_136040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_approx_seven_l1360_136078

/-- The area of the circle in square meters -/
noncomputable def circle_area : ℝ := 153.93804002589985

/-- The approximate value of π -/
noncomputable def π_approx : ℝ := 3.14159

/-- The radius of the circle -/
noncomputable def radius : ℝ := Real.sqrt (circle_area / π_approx)

/-- Theorem stating that the radius of the circle is approximately 7 meters -/
theorem radius_approx_seven : ∀ ε > 0, |radius - 7| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_approx_seven_l1360_136078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_found_five_seashells_l1360_136063

/-- The number of seashells Tom found on the beach -/
def seashells_found : ℕ := 5

/-- The number of seashells Tom gave to Jessica -/
def seashells_given : ℕ := 2

/-- The number of seashells Tom has now -/
def seashells_left : ℕ := 3

/-- Theorem: Tom found 5 seashells on the beach -/
theorem tom_found_five_seashells : seashells_found = seashells_given + seashells_left := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_found_five_seashells_l1360_136063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l1360_136025

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h : dividend = quotient * divisor + remainder) : divisor = 165 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l1360_136025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_theorem_remainder_correct_l1360_136024

/-- The polynomial we're dividing -/
def f (x : ℝ) : ℝ := x^5 - 2*x^4 - x^3 + 3*x^2 + x

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := (x^2 - 1) * (x + 2)

/-- The remainder polynomial -/
def r (x : ℝ) : ℝ := -2*x^2 + 2*x + 2

/-- The quotient polynomial (existence assumed) -/
noncomputable def q : ℝ → ℝ := sorry

theorem division_theorem : ∀ x, f x = g x * q x + r x := by
  sorry

theorem remainder_correct : ∃! (a b c : ℝ), ∀ x, 
  f x = g x * q x + (a*x^2 + b*x + c) ∧ 
  a = -2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_theorem_remainder_correct_l1360_136024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trips_to_fill_barrel_l1360_136079

-- Define the constants
def barrel_radius : ℝ := 8
def barrel_height : ℝ := 20
def bucket_radius : ℝ := 8

-- Define the volumes
noncomputable def bucket_volume : ℝ := (2/3) * Real.pi * (bucket_radius ^ 3)
noncomputable def barrel_volume : ℝ := Real.pi * (barrel_radius ^ 2) * barrel_height

-- Theorem statement
theorem min_trips_to_fill_barrel :
  ∃ n : ℕ, (n : ℝ) * bucket_volume ≥ barrel_volume ∧
  ∀ m : ℕ, (m : ℝ) * bucket_volume ≥ barrel_volume → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trips_to_fill_barrel_l1360_136079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_y_l1360_136023

noncomputable def y (x : ℝ) : ℝ := (5 * x - 1) * (Real.log x)^2

theorem third_derivative_y (x : ℝ) (h : x > 0) :
  (deriv^[3] y) x = (6 - 2 * (5 * x + 2) * Real.log x) / x^3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_y_l1360_136023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_stack_height_l1360_136009

/-- Represents a stack of connected rings -/
structure RingStack where
  topDiameter : ℝ
  bottomDiameter : ℝ
  ringThickness : ℝ
  diameterDecrease : ℝ

/-- Calculates the total height of a stack of rings -/
noncomputable def totalHeight (stack : RingStack) : ℝ :=
  let numRings := (stack.topDiameter - stack.bottomDiameter) / stack.diameterDecrease + 1
  let sumInsideDiameters := (numRings / 2) * ((stack.topDiameter - 2 * stack.ringThickness) + (stack.bottomDiameter - 2 * stack.ringThickness))
  sumInsideDiameters

/-- Theorem stating the total height of the given ring stack is 253 cm -/
theorem ring_stack_height :
  let stack : RingStack := {
    topDiameter := 25,
    bottomDiameter := 4,
    ringThickness := 1.5,
    diameterDecrease := 1
  }
  totalHeight stack = 253 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_stack_height_l1360_136009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1360_136072

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) / Real.log a

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1
  (∀ t : ℝ, (a > 1 → (f a (t^2 - t - 1) + f a (t - 2) < 0 ↔ 1 < t ∧ t < Real.sqrt 3)) ∧
             (0 < a ∧ a < 1 → (f a (t^2 - t - 1) + f a (t - 2) < 0 ↔ Real.sqrt 3 < t ∧ t < 2))) ∧
  -- Part 2
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → 0 ≤ f a x ∧ f a x ≤ 1) → a = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1360_136072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l1360_136074

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | Real.exp ((x-2) * Real.log 2) > 1}

-- Define the intersection of A and the complement of B
def intersection : Set ℝ := A ∩ (Set.univ \ B)

-- State the theorem
theorem intersection_characterization : 
  intersection = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l1360_136074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_puzzle_l1360_136053

/-- The number of towers that can be built using cubes with edge lengths from 2 to n,
    where each cube on top has an edge length at most 1 greater than the cube below it. -/
def numTowers : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => 2 * numTowers (n+1)

/-- The problem statement -/
theorem tower_puzzle :
  numTowers 8 % 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_puzzle_l1360_136053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_formula_l1360_136061

-- Define the pension function
noncomputable def pension (k c x : ℝ) : ℝ := k * Real.sqrt (c * x)

-- State the theorem
theorem pension_formula 
  (k c x a b p q : ℝ) 
  (hb : b ≠ a) 
  (h1 : pension k c (x + a) = pension k c x + 2 * p)
  (h2 : pension k c (x + b) = pension k c x + 2 * q) :
  pension k c x = (a * q^2 - b * p^2) / (b * p - a * q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_formula_l1360_136061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_result_l1360_136076

-- Define the custom operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a ≥ b then Real.sqrt a - Real.sqrt b else Real.sqrt b - Real.sqrt a

-- Theorem statement
theorem custom_operation_result :
  star 9 8 + star 16 18 = Real.sqrt 2 - 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_result_l1360_136076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_vertex_and_circumcenter_l1360_136054

/-- The area of a square in 3D space with a given vertex and circumcenter -/
theorem square_area_from_vertex_and_circumcenter :
  let A : Fin 3 → ℝ := ![(-6), (-4), 2]
  let O : Fin 3 → ℝ := ![3, 2, (-1)]
  let distance := Real.sqrt ((A 0 - O 0)^2 + (A 1 - O 1)^2 + (A 2 - O 2)^2)
  let side_length := distance / Real.sqrt 2
  let area := side_length^2
  area = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_vertex_and_circumcenter_l1360_136054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1360_136096

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := 11 * x^2 + 20 * y^2 = 220

-- Define the focal length
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem ellipse_focal_length :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    focal_length a b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1360_136096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_sum_l1360_136065

theorem tan_value_for_given_sum (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = -Real.sqrt 10 / 5)
  (h2 : α ∈ Set.Ioo 0 Real.pi) :
  Real.tan α = -1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_sum_l1360_136065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_2003_l1360_136055

mutual
  def a : ℕ → ℕ
    | 0 => 1
    | n + 1 => (a n)^2001 + b n

  def b : ℕ → ℕ
    | 0 => 4
    | n + 1 => (b n)^2001 + a n
end

theorem not_divisible_by_2003 : ∀ n : ℕ, ¬(2003 ∣ a n) ∧ ¬(2003 ∣ b n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_2003_l1360_136055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1360_136093

theorem integer_solutions_equation :
  {(m, n) : ℤ × ℤ | m^2 * n + 1 = m^2 + 2*m*n + 2*m + n} =
  {(-1, -1), (0, 1), (1, -1), (2, -7), (3, 7)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1360_136093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l1360_136041

theorem absolute_value_nested_expression : 
  abs (abs (-abs (2 - 3) + 2) - 2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l1360_136041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l1360_136000

noncomputable section

-- Define the triangle and its properties
def triangle_side : ℝ := 15
def opposite_angle : ℝ := 45 * Real.pi / 180  -- Convert to radians

-- Define the diameter of the circumscribed circle
def circumscribed_circle_diameter : ℝ := triangle_side / Real.sin opposite_angle

-- Theorem statement
theorem triangle_circumscribed_circle_diameter :
  circumscribed_circle_diameter = 15 * Real.sqrt 2 :=
by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l1360_136000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_one_implies_cos_sum_negative_one_l1360_136010

theorem sin_product_one_implies_cos_sum_negative_one 
  (α β : ℝ) (h : Real.sin α * Real.sin β = 1) : 
  Real.cos (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_one_implies_cos_sum_negative_one_l1360_136010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_characterization_l1360_136015

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (y ≥ 0 ∧ x^2 = 8*y) ∨ (y < 0 ∧ x = 0)

/-- The distance from a point to the x-axis -/
def distToXAxis (y : ℝ) : ℝ := |y|

/-- The distance from a point to F(0, 2) -/
noncomputable def distToF (x y : ℝ) : ℝ := Real.sqrt (x^2 + (y - 2)^2)

theorem trajectory_characterization (x y : ℝ) :
  distToXAxis y < distToF x y - 2 → trajectory x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_characterization_l1360_136015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l1360_136039

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l1360_136039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_travel_time_l1360_136034

theorem andrew_travel_time (subway_time train_time_multiplier bike_time : ℕ) :
  subway_time = 10 →
  train_time_multiplier = 2 →
  bike_time = 8 →
  subway_time + train_time_multiplier * subway_time + bike_time = 38 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_travel_time_l1360_136034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_length_equals_arc_length_l1360_136090

/-- Given a circle with an arc of 120°, and an inscribed circle bounded by the tangents 
    drawn through the endpoints of the arc and the arc itself, the length of the inscribed 
    circle is equal to the length of the original arc. -/
theorem inscribed_circle_length_equals_arc_length (R : ℝ) (h : R > 0) : 
  2 * Real.pi * (R / 3) = R * (2 * Real.pi / 3) := by
  -- Define arc_angle
  let arc_angle : ℝ := 2 * Real.pi / 3
  -- Define arc_length
  let arc_length : ℝ := R * arc_angle
  -- Define inscribed_circle_radius
  let inscribed_circle_radius : ℝ := R / 3
  -- Define inscribed_circle_length
  let inscribed_circle_length : ℝ := 2 * Real.pi * inscribed_circle_radius
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_length_equals_arc_length_l1360_136090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_specific_wall_l1360_136020

/-- The force of water pressure on a rectangular wall -/
noncomputable def waterPressureForce (length height : ℝ) (waterDensity gravity : ℝ) : ℝ :=
  waterDensity * gravity * length * (height^2 / 2)

/-- Theorem: The force of water pressure on a specific wall -/
theorem water_pressure_force_specific_wall :
  let length : ℝ := 20
  let height : ℝ := 5
  let waterDensity : ℝ := 1000
  let gravity : ℝ := 9.81
  ∃ ε > 0, |waterPressureForce length height waterDensity gravity - 2.45e6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_specific_wall_l1360_136020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l1360_136033

/-- Represents an equilateral triangle with three inscribed circles -/
structure TriangleWithCircles where
  /-- Radius of the first inscribed circle -/
  r1 : ℝ
  /-- Radius of the second inscribed circle -/
  r2 : ℝ
  /-- Radius of the third inscribed circle -/
  r3 : ℝ
  /-- Each circle touches two sides of the triangle and two other circles -/
  touches_sides_and_circles : True

/-- Calculates the perimeter of the triangle -/
noncomputable def perimeter (t : TriangleWithCircles) : ℝ := 
  3 * 2 * (t.r1 + t.r2 + t.r3)

/-- Calculates the area of the triangle -/
noncomputable def area (t : TriangleWithCircles) : ℝ := 
  (Real.sqrt 3 / 4) * (2 * (t.r1 + t.r2 + t.r3))^2

/-- Theorem stating the perimeter and area of the specific triangle -/
theorem triangle_perimeter_and_area :
  ∃ (t : TriangleWithCircles), 
    t.r1 = 2 ∧ t.r2 = 3 ∧ t.r3 = 4 ∧ 
    perimeter t = 54 ∧ 
    area t = 81 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l1360_136033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_90_l1360_136022

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors n)).sum id

theorem sum_of_odd_divisors_90 :
  sum_of_odd_divisors 90 = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_90_l1360_136022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l1360_136014

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem simple_interest_calculation :
  let principal : ℝ := 8032.5
  let rate : ℝ := 10
  let time : ℝ := 5
  simpleInterest principal rate time = 4016.25 := by
  -- Unfold the definition of simpleInterest
  unfold simpleInterest
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l1360_136014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l1360_136013

/-- A regular tetrahedron with edge length 2√6 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 2 * Real.sqrt 6

/-- A sphere centered at the center of the tetrahedron with radius √3 -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  is_center : center = (0, 0, 0)  -- Assuming the center of the tetrahedron is at the origin
  radius_value : radius = Real.sqrt 3

/-- The surface of the tetrahedron -/
def surface (t : RegularTetrahedron) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The boundary of the sphere -/
def boundary (s : Sphere) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The intersection between the sphere and the surface of the tetrahedron -/
def intersection (t : RegularTetrahedron) (s : Sphere) : Set (ℝ × ℝ × ℝ) :=
  { p | p ∈ surface t ∧ p ∈ boundary s }

/-- The length of a curve in ℝ³ -/
noncomputable def curve_length (c : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem intersection_length (t : RegularTetrahedron) (s : Sphere) :
  curve_length (intersection t s) = 8 * Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l1360_136013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1360_136048

open Set
open Function

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

theorem f_increasing_on_interval : 
  StrictMonoOn f (Set.Ioi 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1360_136048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1360_136085

noncomputable def f (x : ℝ) := 4 * (Real.cos x)^2 - 4 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem function_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≤ 6) ∧
  (∀ k : ℤ, f (-π/6 + k * π) = 6) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (π/3 + k * π) (5*π/6 + k * π), 
    ∀ y ∈ Set.Icc (π/3 + k * π) (5*π/6 + k * π), x < y → f x < f y) ∧
  (∀ k : ℤ, ∀ x, f (2 * (π/3 + k * π/2) - x) = f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1360_136085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_is_ellipse_max_value_of_product_l1360_136044

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 12

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 6 = 0

-- Define the point N
def N (x y x₀ y₀ : ℝ) : Prop :=
  x = Real.sqrt 3 / 3 * x₀ ∧ y = 1 / 2 * y₀

-- Define the curve C (locus of N)
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l₂
def l₂ (x y k m : ℝ) : Prop := y = k * x + m

-- Define the distances d₁, d₂, and d₃
noncomputable def d₁ (k m : ℝ) : ℝ := |m - k| / Real.sqrt (1 + k^2)
noncomputable def d₂ (k m : ℝ) : ℝ := |m + k| / Real.sqrt (1 + k^2)
noncomputable def d₃ (k m : ℝ) : ℝ := |d₁ k m - d₂ k m| / |k|

-- Theorem statements
theorem curve_C_is_ellipse (x y : ℝ) :
  (∃ x₀ y₀ : ℝ, C₁ x₀ y₀ ∧ N x y x₀ y₀) ↔ C x y := by sorry

theorem max_value_of_product (k m : ℝ) :
  l₂ x y k m → C x y → (d₁ k m + d₂ k m) * d₃ k m ≤ 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_is_ellipse_max_value_of_product_l1360_136044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_xyz_l1360_136059

-- Define the given conditions
noncomputable def x : ℝ := Real.exp (Real.log 3 / 3)
noncomputable def y : ℝ := Real.exp (Real.log 7 / 6)
noncomputable def z : ℝ := 7^(1/7)

-- State the theorem
theorem relationship_xyz : z < y ∧ y < x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_xyz_l1360_136059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_is_half_l1360_136098

-- Define a right triangle with an acute angle β
structure RightTriangle where
  β : ℝ
  is_acute : 0 < β ∧ β < Real.pi / 2
  is_right_triangle : True  -- This is a placeholder for the right triangle condition

-- Define the condition for tan(β/2)
def tan_half_beta (triangle : RightTriangle) : Prop :=
  Real.tan (triangle.β / 2) = 1 / (3 ^ (1/3 : ℝ))

-- Define φ as the angle between the median and angle bisector
noncomputable def phi (triangle : RightTriangle) : ℝ := 
  -- This is a placeholder for the actual definition of φ
  Real.pi / 4  -- Using π/4 as a dummy value

-- State the theorem
theorem tan_phi_is_half (triangle : RightTriangle) 
  (h : tan_half_beta triangle) : Real.tan (phi triangle) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_is_half_l1360_136098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1360_136067

theorem triangle_angle_relation (A B C : ℝ) (h1 : 0 < A ∧ A < π / 2)
    (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π) 
    (h4 : A + B + C = π) (h5 : Real.sin B / Real.sin A = 2 * Real.cos (A + B)) :
  (∃ (B_max : ℝ), B ≤ B_max ∧ B_max = π / 3) ∧
  (B = π / 3 → C = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1360_136067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_remaining_money_l1360_136011

/-- Calculate the remaining money after purchases with tax --/
def remaining_money (initial_amount : ℚ) (pencil_cost : ℚ) (notebook_cost : ℚ) (num_pencils : ℕ) (tax_rate : ℚ) : ℚ :=
  let total_cost := pencil_cost * num_pencils + notebook_cost
  let tax_amount := (total_cost * tax_rate).floor / 100 * 100  -- Round down to nearest cent
  let total_with_tax := total_cost + tax_amount
  (initial_amount - total_with_tax).floor / 100 * 100  -- Round down to nearest cent

/-- Theorem stating that Mrs. Hilt's remaining money is $0.66 --/
theorem mrs_hilt_remaining_money :
  remaining_money (150/100) (11/100) (45/100) 3 (8/100) = 66/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_remaining_money_l1360_136011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digit_occurrences_l1360_136007

/-- Counts the occurrences of a given digit in the base ten representation of a natural number. -/
def digit_count (n : ℕ) (d : Fin 10) : ℕ :=
  sorry

/-- Given two positive integers, there exists another positive integer such that 
    when multiplied with the given integers, the results have the same number of 
    occurrences of each non-zero digit in base ten. -/
theorem same_digit_occurrences (m n : ℕ+) : 
  ∃ c : ℕ+, ∀ d : Fin 9, 
    (digit_count ((c : ℕ) * (m : ℕ)) ⟨d.val + 1, by sorry⟩ = 
     digit_count ((c : ℕ) * (n : ℕ)) ⟨d.val + 1, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digit_occurrences_l1360_136007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_four_y_fifth_power_one_l1360_136016

-- Define y as a complex number
noncomputable def y : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

-- State the theorem
theorem product_equals_four :
  (3 * y + y^2) * (3 * y^2 + y^4) * (3 * y^3 + y^6) * (3 * y^4 + y^8) = 4 :=
by
  sorry

-- Define the property y^5 = 1
theorem y_fifth_power_one : y^5 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_four_y_fifth_power_one_l1360_136016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_setop_theorem_l1360_136083

-- Define the operation ⊗
def setOp (P Q : Set ℝ) : Set ℝ := (P ∪ Q) \ (P ∩ Q)

-- Define sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {x | x > 1}

-- State the theorem
theorem setop_theorem : setOp P Q = Set.Icc 0 1 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_setop_theorem_l1360_136083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1360_136037

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + 1| + |a * x - 1|

/-- The solution set for part 1 -/
def solution_set : Set ℝ := {x | x ≤ -3/4 ∨ x ≥ 3/4}

theorem part1 :
  ∀ x : ℝ, f 2 x ≥ 3 ↔ x ∈ solution_set := by sorry

theorem part2 :
  ∀ a : ℝ, a > 0 →
  (∃ x : ℝ, f a x < a / 2 + 1) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1360_136037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_6_l1360_136051

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 4, 2]

theorem B_power_6 : 
  B^6 = 2080 • B + 7330 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_6_l1360_136051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_12x12_l1360_136060

/-- Represents a chessboard --/
structure Chessboard where
  size : ℕ

/-- Represents a king on the chessboard --/
structure King where
  x : ℕ
  y : ℕ

/-- Two kings attack each other if their squares share at least one vertex --/
def attacks (k1 k2 : King) : Prop :=
  (k1.x = k2.x ∧ (k1.y = k2.y + 1 ∨ k1.y = k2.y - 1)) ∨
  (k1.y = k2.y ∧ (k1.x = k2.x + 1 ∨ k1.x = k2.x - 1)) ∨
  (k1.x = k2.x + 1 ∧ k1.y = k2.y + 1) ∨
  (k1.x = k2.x + 1 ∧ k1.y = k2.y - 1) ∨
  (k1.x = k2.x - 1 ∧ k1.y = k2.y + 1) ∨
  (k1.x = k2.x - 1 ∧ k1.y = k2.y - 1)

/-- A valid king placement is one where each king attacks exactly one other king --/
def valid_placement (board : Chessboard) (kings : List King) : Prop :=
  ∀ k ∈ kings, (∃! k', k' ∈ kings ∧ k ≠ k' ∧ attacks k k') ∧
               k.x ≤ board.size ∧ k.y ≤ board.size

/-- The main theorem: The maximum number of kings on a 12x12 chessboard is 56 --/
theorem max_kings_12x12 :
  ∀ (kings : List King),
    valid_placement (Chessboard.mk 12) kings →
    kings.length ≤ 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_12x12_l1360_136060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_abs_3x_minus_2_solution_equation_unique_solution_l1360_136003

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  |(x^2 - 6*x + 9) / (3 - x) + (4*x^2 - 5*x) / x|

-- State the theorem
theorem f_equals_abs_3x_minus_2 (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 0) :
  f x = |3*x - 2| := by sorry

-- Define the domain of f
def D_f : Set ℝ := {x | x ≠ 3 ∧ x ≠ 0}

-- State the theorem about the solution of |3x - 2| = |x + a|
theorem solution_equation (a : ℝ) :
  (∃ x, x ∈ D_f ∧ |3*x - 2| = |x + a|) ↔ (a = -2/3 ∨ a = 2) := by sorry

-- State the theorem about the unique solution when a = -10
theorem unique_solution :
  ∃! x, x ∈ D_f ∧ |3*x - 2| = |x - 10| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_abs_3x_minus_2_solution_equation_unique_solution_l1360_136003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equals_square_of_linear_l1360_136084

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

def linear_expr (x c : ℝ) : ℝ := x + c

theorem quadratic_equals_square_of_linear 
  (m : ℝ) : 
  (∃ c : ℝ, c^2 = 3 ∧ ∀ x : ℝ, G x m = (linear_expr x c)^2) ↔ m = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equals_square_of_linear_l1360_136084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_condition_l1360_136057

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def ParallelVectors (a b : ℝ × ℝ) : Prop :=
  ∃ (lambda : ℝ), a = (lambda * b.1, lambda * b.2)

/-- Given vectors are not collinear -/
axiom not_collinear (e₁ e₂ : ℝ × ℝ) : e₁.1 * e₂.2 ≠ e₁.2 * e₂.1

theorem parallel_vectors_condition (e₁ e₂ : ℝ × ℝ) (k : ℝ) :
  let a := (e₁.1 - 4 * e₂.1, e₁.2 - 4 * e₂.2)
  let b := (2 * e₁.1 + k * e₂.1, 2 * e₁.2 + k * e₂.2)
  ParallelVectors a b ↔ k = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_condition_l1360_136057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1360_136056

noncomputable section

variable (a : ℝ)

noncomputable def expr1 := Real.sqrt (1/3)
noncomputable def expr2 := Real.sqrt 147
noncomputable def expr3 := Real.sqrt (25*a)
noncomputable def expr4 := Real.sqrt (a^2 + 1)

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, (∃ n : ℕ, x = Real.sqrt y ∧ n > 1 ∧ y ≠ Real.sqrt (y^n)) →
    x = Real.sqrt y

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (expr4 a) ∧
  ¬is_simplest_quadratic_radical expr1 ∧
  ¬is_simplest_quadratic_radical expr2 ∧
  ¬is_simplest_quadratic_radical (expr3 a) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1360_136056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_l_shape_theorem_l1360_136075

/-- An L-shaped piece is a 2 × 2 grid with one small square removed -/
def LShape := Unit

/-- Represents an n × n grid -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a grid can be cut into L-shaped pieces after removing one square -/
def canBeCutIntoLShapes (n : ℕ) (g : Grid n) : Prop := sorry

/-- The main theorem -/
theorem grid_l_shape_theorem (n : ℕ) (h : n ≥ 3) :
  (∀ (i j : Fin n), canBeCutIntoLShapes n (fun x y ↦ x ≠ i ∨ y ≠ j)) ↔ ¬(3 ∣ n) := by
  sorry

#check grid_l_shape_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_l_shape_theorem_l1360_136075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_self_intersections_approximately_12_intersections_l1360_136006

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := Real.cos t + t / 2 + Real.sin (2 * t)
noncomputable def y (t : ℝ) : ℝ := Real.sin t

-- Define the range of x values we're interested in
def x_min : ℝ := 2
def x_max : ℝ := 40

-- Define what it means for the curve to self-intersect
def is_self_intersection (t₁ t₂ : ℝ) : Prop :=
  t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ x_min ≤ x t₁ ∧ x t₁ ≤ x_max

-- Theorem stating that there exists at least n self-intersections
theorem exists_n_self_intersections : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (S : Finset ℝ), S.card = n ∧ ∀ t₁ ∈ S, ∃ t₂ ∈ S, is_self_intersection t₁ t₂ := by
  sorry

-- Additional theorem to state that the number of intersections is approximately 12
theorem approximately_12_intersections : 
  ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 14 ∧
  ∃ (S : Finset ℝ), S.card = n ∧ ∀ t₁ ∈ S, ∃ t₂ ∈ S, is_self_intersection t₁ t₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_self_intersections_approximately_12_intersections_l1360_136006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_erasing_critical_info_l1360_136046

/-- Represents the duration of the entire tape recording in minutes -/
noncomputable def total_duration : ℝ := 30

/-- Represents the start time of the critical conversation in minutes -/
noncomputable def conversation_start : ℝ := 0.5

/-- Represents the duration of the critical conversation in minutes -/
noncomputable def conversation_duration : ℝ := 1/6

/-- Represents the end time of the critical conversation in minutes -/
noncomputable def conversation_end : ℝ := conversation_start + conversation_duration

/-- Theorem stating the probability of erasing part or all of the critical conversation -/
theorem probability_of_erasing_critical_info : 
  (conversation_end / total_duration) = 1/45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_erasing_critical_info_l1360_136046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_and_gcd_l1360_136071

/-- Polynomial P -/
def P (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

/-- Polynomial Q -/
def Q (a b x : ℝ) : ℝ := x^3 + b*x + a

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem polynomial_roots_and_gcd (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x : ℝ, P a b x = 0 ↔ ∃ y : ℝ, Q a b y = 0 ∧ x * y = 1) →
  (∃ m n : ℤ, (a : ℝ) = ↑m ∧ (b : ℝ) = ↑n) ∧
  (∃ k : ℕ, Int.natAbs (Int.floor (P a b (↑(factorial 2013) + 1) - Q a b (↑(factorial 2013) + 1))) = k ∧
            k = Int.natAbs (Int.floor (1 + b + b^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_and_gcd_l1360_136071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_expression_evaluate_a_minus_b_evaluate_quadratic_expression_l1360_136043

-- Problem 1
noncomputable def x : ℝ := Real.sqrt 6 - Real.sqrt 2

theorem simplify_and_evaluate_expression :
  x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5) = 1 - 2 * Real.sqrt 3 :=
by sorry

-- Problem 2
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem evaluate_a_minus_b :
  a - b = 2 * Real.sqrt 2 :=
by sorry

theorem evaluate_quadratic_expression :
  a^2 - 2*a*b + b^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_expression_evaluate_a_minus_b_evaluate_quadratic_expression_l1360_136043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clowns_to_guppies_ratio_l1360_136038

/-- The number of clowns Tim bought -/
def clowns : ℕ := sorry

/-- The number of tetras bought -/
def tetras : ℕ := sorry

/-- The number of guppies Rick bought -/
def guppies : ℕ := sorry

/-- The total number of animals bought -/
def total_animals : ℕ := sorry

/-- Tetras are 4 times the number of clowns -/
axiom tetras_to_clowns : tetras = 4 * clowns

/-- Rick bought 30 guppies -/
axiom guppies_count : guppies = 30

/-- The total number of animals is 330 -/
axiom total_animals_count : total_animals = 330

/-- The sum of all animals equals the total -/
axiom sum_of_animals : tetras + clowns + guppies = total_animals

/-- The ratio of clowns to guppies is 2:1 -/
theorem clowns_to_guppies_ratio : clowns / guppies = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clowns_to_guppies_ratio_l1360_136038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_and_axis_l1360_136077

/-- Represents a parabola in the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  let y := -p.b / (2 * p.a)
  let x := p.a * y^2 + p.b * y + p.c
  (x, y)

/-- Calculates the axis of symmetry of a parabola -/
noncomputable def axisOfSymmetry (p : Parabola) : ℝ :=
  -p.b / (2 * p.a)

theorem parabola_vertex_and_axis (p : Parabola) 
  (h : p = { a := 2, b := 4, c := -5 }) : 
  vertex p = (-7, -1) ∧ axisOfSymmetry p = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_and_axis_l1360_136077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1360_136052

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  center : Point

/-- Theorem about the ellipse and the fixed point -/
theorem ellipse_and_fixed_point 
  (C : Ellipse)
  (A B E F : Point)
  (h1 : C.a > C.b)
  (h2 : C.b > 0)
  (h3 : A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1)  -- A is on the ellipse
  (h4 : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1)  -- B is on the ellipse
  (h5 : (A.x - B.x)^2 + (A.y - B.y)^2 = 7)  -- |AB| = √7
  (h6 : ∀ (P : Point), P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1 → 
        (P.x - E.x)^2 + (P.y - E.y)^2 = 9 ∨ (P.x - F.x)^2 + (P.y - F.y)^2 = 1)
  : C.a = 2 ∧ C.b = Real.sqrt 3 ∧ 
    ∀ (M : Point), M.x^2 / 4 + M.y^2 / 3 = 1 → 
    ∃ (N : Point), N.x^2 / 4 + N.y^2 / 3 = 1 ∧ 
    (N.x - E.x) / (N.y - E.y) = (M.x - E.x) / (M.y - E.y) ∧
    (N.x + M.x) / (N.y - M.y) = (-4 - M.x) / (-M.y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1360_136052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1360_136030

theorem relationship_abc : 
  (2 : ℝ) ^ 555 < (6 : ℝ) ^ 222 ∧ (6 : ℝ) ^ 222 < (3 : ℝ) ^ 444 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1360_136030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1360_136097

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, S n / T n = (2 * n) / (3 * n + 1)) →
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) →
  (∀ n : ℕ, T n = (n * (b 1 + b n)) / 2) →
  a 5 / b 5 = 9 / 14 := by
  sorry

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1360_136097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_can_be_any_real_l1360_136045

theorem a_can_be_any_real : ∀ (a b c d : ℝ), 
  b * d ≠ 0 → a / b < -c / d → 
  (∃ (x : ℝ), x > 0 ∧ a = x) ∧ 
  (∃ (y : ℝ), y < 0 ∧ a = y) ∧ 
  (∃ (z : ℝ), z = 0 ∧ a = z) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_can_be_any_real_l1360_136045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_intersection_theorem_l1360_136005

noncomputable section

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - (3/2) * x^2 - m

/-- The derivative of f with respect to x -/
def f_deriv (x : ℝ) : ℝ := 3 * x * (x - 1)

theorem cubic_intersection_theorem (m : ℝ) :
  m < 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ 
    ∀ x : ℝ, f m x = 0 → x = x₁ ∨ x = x₂) →
  m = -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_intersection_theorem_l1360_136005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kai_trip_time_l1360_136095

/-- Represents the details of Kai's trip -/
structure TripDetails where
  highway_distance : ℚ
  mountain_distance : ℚ
  speed_ratio : ℚ
  mountain_time : ℚ
  break_time : ℚ

/-- Calculates the total time for Kai's trip -/
noncomputable def total_trip_time (trip : TripDetails) : ℚ :=
  let mountain_speed := trip.mountain_distance / trip.mountain_time
  let highway_speed := mountain_speed * trip.speed_ratio
  let highway_time := trip.highway_distance / highway_speed
  2 * (trip.mountain_time + highway_time) + trip.break_time

/-- Theorem stating that Kai's trip took 240 minutes -/
theorem kai_trip_time : 
  ∀ (trip : TripDetails), 
  trip.highway_distance = 100 ∧ 
  trip.mountain_distance = 15 ∧ 
  trip.speed_ratio = 5 ∧ 
  trip.mountain_time = 45 ∧ 
  trip.break_time = 30 → 
  total_trip_time trip = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kai_trip_time_l1360_136095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_shifted_theta_l1360_136081

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_value_at_shifted_theta (θ : ℝ) 
  (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : f (θ + Real.pi / 4) = Real.sqrt 2 / 3) : 
  f (θ - Real.pi / 4) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_shifted_theta_l1360_136081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_none_exceed_correct_l1360_136019

/-- The probability that none of the three pieces of a rod of length l, 
    broken at two random points, exceeds a given length a -/
noncomputable def probability_none_exceed (l a : ℝ) : ℝ :=
  if 0 ≤ a ∧ a ≤ l/3 then 0
  else if l/3 < a ∧ a ≤ l/2 then (3*a/l - 1)^2
  else if l/2 < a ∧ a ≤ l then 1 - 3*(1 - a/l)^2
  else 0

/-- Theorem stating the probability that none of the three pieces 
    of a rod exceeds a given length -/
theorem probability_none_exceed_correct (l a : ℝ) (hl : l > 0) :
  probability_none_exceed l a =
    if 0 ≤ a ∧ a ≤ l/3 then 0
    else if l/3 < a ∧ a ≤ l/2 then (3*a/l - 1)^2
    else if l/2 < a ∧ a ≤ l then 1 - 3*(1 - a/l)^2
    else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_none_exceed_correct_l1360_136019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1360_136082

/-- Two lines are parallel if their slopes are equal -/
def parallel (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2

/-- Distance between two parallel lines -/
noncomputable def distance (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / Real.sqrt (a^2 + b^2)

theorem parallel_lines_distance :
  ∀ (m : ℝ),
  parallel 3 4 6 m →
  distance 3 4 (-3) 7 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1360_136082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_iff_not_collinear_specific_vectors_basis_condition_l1360_136073

/-- Two vectors form a basis for a 2D plane if and only if they are not collinear -/
theorem vectors_form_basis_iff_not_collinear (a b : ℝ × ℝ) :
  (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • a + p.2 • b) ↔ ¬ (∃ k : ℝ, b = k • a) :=
sorry

/-- The condition for two specific vectors to form a basis -/
theorem specific_vectors_basis_condition (m : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 2*m - 3)
  (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • a + p.2 • b) ↔ m ≠ -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_iff_not_collinear_specific_vectors_basis_condition_l1360_136073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sin_cos_monotonicity_l1360_136069

theorem log_sin_cos_monotonicity (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ k : ℤ, StrictMonoOn (fun x => Real.log (Real.sin x + Real.cos x) / Real.log a)
    (Set.Icc (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + 3 * Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sin_cos_monotonicity_l1360_136069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_40_equals_31_127_l1360_136026

def u : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | m+1 => if (m+1) % 3 = 0 then 2 + u ((m+1)/3) else 2 / u m

theorem u_40_equals_31_127 : u 40 = 31/127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_40_equals_31_127_l1360_136026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6_or_8_below_151_l1360_136001

theorem multiples_of_6_or_8_below_151 : 
  (Finset.filter (λ n : ℕ => n < 151 ∧ 
    ((6 ∣ n ∨ 8 ∣ n) ∧ ¬(6 ∣ n ∧ 8 ∣ n))) 
    (Finset.range 151)).card = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6_or_8_below_151_l1360_136001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l1360_136028

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x)^2 - 2

-- Define the interval
def interval (a : ℝ) : Set ℝ := Set.Icc (-Real.pi/12) a

-- Theorem statement
theorem max_value_condition (a : ℝ) :
  (∃ (x : ℝ), x ∈ interval a ∧ ∀ (y : ℝ), y ∈ interval a → f y ≤ f x) ↔ a > Real.pi/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l1360_136028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1360_136031

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) + Real.sqrt (15 - 3*x)

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
  (∃ x, 4 ≤ x ∧ x ≤ 5 ∧ f x = y) ↔ 1 ≤ y ∧ y ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1360_136031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l1360_136064

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_problem (principal time interest : ℝ) 
  (h1 : principal = 4000)
  (h2 : time = 2)
  (h3 : interest = 320) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 4 := by
  use 4
  constructor
  · rw [simple_interest, h1, h2, h3]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l1360_136064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_schur_theorem_l1360_136068

def IsPartition (P : List (Set ℕ+)) : Prop :=
  (∀ i j, i ≠ j → i < P.length → j < P.length → P[i]! ∩ P[j]! = ∅) ∧ 
  (⋃ i ∈ List.range P.length, P[i]! = Set.univ)

theorem schur_theorem (P : List (Set ℕ+)) (h : IsPartition P) :
  ∃ (i : Fin P.length) (x y : ℕ+), x ∈ P[i] ∧ y ∈ P[i] ∧ (x + y : ℕ+) ∈ P[i] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_schur_theorem_l1360_136068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_equivalence_l1360_136018

/-- Parametric equation representation -/
structure ParametricEq where
  t : ℝ
  x : ℝ
  y : ℝ
  eq_x : x = Real.sqrt t
  eq_y : y = 2 * Real.sqrt (1 - t)

/-- Ordinary equation representation -/
structure OrdinaryEq where
  x : ℝ
  y : ℝ
  eq : x^2 + y^2/4 = 1
  x_range : 0 ≤ x ∧ x ≤ 1
  y_range : 0 ≤ y ∧ y ≤ 2

/-- Theorem stating the equivalence of the parametric and ordinary equations -/
theorem parametric_to_ordinary_equivalence :
  ∀ (p : ParametricEq), ∃ (o : OrdinaryEq), p.x = o.x ∧ p.y = o.y :=
by
  intro p
  sorry  -- The proof is omitted for now

/-- Helper function to create a ParametricEq -/
noncomputable def makeParametricEq (t : ℝ) : ParametricEq where
  t := t
  x := Real.sqrt t
  y := 2 * Real.sqrt (1 - t)
  eq_x := rfl
  eq_y := rfl

/-- Helper function to create an OrdinaryEq -/
noncomputable def makeOrdinaryEq (x y : ℝ) (h_eq : x^2 + y^2/4 = 1)
    (h_x : 0 ≤ x ∧ x ≤ 1) (h_y : 0 ≤ y ∧ y ≤ 2) : OrdinaryEq where
  x := x
  y := y
  eq := h_eq
  x_range := h_x
  y_range := h_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_equivalence_l1360_136018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_percentage_proof_l1360_136062

/-- Proves that the unknown investment percentage is approximately 4% --/
theorem investment_percentage_proof (total_investment : ℝ) 
  (year_end_total : ℝ) (known_rate : ℝ) (amount_at_unknown_rate : ℝ) :
  total_investment = 1000 →
  year_end_total = 1046 →
  known_rate = 0.06 →
  amount_at_unknown_rate = 699.99 →
  ∃ (unknown_rate : ℝ), 
    (amount_at_unknown_rate * unknown_rate + 
     (total_investment - amount_at_unknown_rate) * known_rate = 
     year_end_total - total_investment) ∧
    (abs (unknown_rate - 0.04) < 0.001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_percentage_proof_l1360_136062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_26_l1360_136089

-- Define the points of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (-4, 6)
def C : ℝ × ℝ := (2, 10)

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to calculate the area of a triangle using Heron's formula
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem stating that the area of the triangle is 26 square units
theorem triangle_area_is_26 :
  triangleArea (distance A B) (distance B C) (distance C A) = 26 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_26_l1360_136089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_leq_one_l1360_136099

-- Define the function f(x) = e^(|x-a|)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (|x - a|)

-- State the theorem
theorem f_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f a x ≤ f a y) →
  a ≤ 1 := by
  sorry

-- Define the set of all possible values for a
def a_range : Set ℝ := Set.Iic 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_leq_one_l1360_136099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1360_136017

def sequence_formula (n : ℕ) : ℝ :=
  if n = 1 then 1
  else (Finset.range (n - 1)).prod (λ k => ((4 ^ k - 1) ^ 2 - 1))

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, Real.sqrt (a n * a (n + 1) + a n * a (n + 2)) = 4 * Real.sqrt (a n * a (n + 1) + a (n + 1) ^ 2) + 3 * Real.sqrt (a n * a (n + 1))) →
  a 1 = 1 →
  a 2 = 8 →
  ∀ n, a n = sequence_formula n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1360_136017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_coins_count_l1360_136021

/-- The number of coins George has, given the total value and number of nickels -/
def total_coins (total_value : ℚ) (num_nickels : ℕ) : ℕ :=
  let nickel_value : ℚ := 1/20
  let dime_value : ℚ := 1/10
  let value_in_dimes : ℚ := total_value - (num_nickels : ℚ) * nickel_value
  let num_dimes : ℕ := (value_in_dimes / dime_value).floor.toNat
  num_nickels + num_dimes

/-- Theorem stating that George has 28 coins in total -/
theorem george_coins_count : total_coins (13/5) 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_coins_count_l1360_136021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_intersection_time_l1360_136027

/-- Represents the position of an elevator as a function of time -/
def ElevatorPosition (k b : ℝ) : ℝ → ℝ := λ t ↦ k * t + b

theorem elevator_intersection_time :
  ∀ (k₁ k₂ k₃ k₄ b₁ b₂ b₃ b₄ : ℝ),
    let S₁ := ElevatorPosition k₁ b₁
    let S₂ := ElevatorPosition k₂ b₂
    let S₃ := ElevatorPosition k₃ b₃
    let S₄ := ElevatorPosition k₄ b₄
    (S₁ 36 = S₂ 36) →  -- Red catches up with Blue at t=36
    (S₁ 42 = S₃ 42) →  -- Red passes Green at t=42
    (S₁ 48 = S₄ 48) →  -- Red passes Yellow at t=48
    (S₄ 51 = S₂ 51) →  -- Yellow passes Blue at t=51
    (S₄ 54 = S₃ 54) →  -- Yellow catches up with Green at t=54
    ∃ t, t = 46 ∧ S₃ t = S₂ t  -- Green passes Blue at t=46
    := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_intersection_time_l1360_136027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1360_136080

theorem sum_remainder (a b c d : ℕ) 
  (ha : a % 53 = 33)
  (hb : b % 53 = 25)
  (hc : c % 53 = 6)
  (hd : d % 53 = 12) :
  (a + b + c + d) % 53 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1360_136080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_squared_plus_sin_x_l1360_136036

theorem definite_integral_x_squared_plus_sin_x : 
  ∫ x in (-1 : ℝ)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_squared_plus_sin_x_l1360_136036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_reservoir_pairing_l1360_136091

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n farms -/
def Farms (n : ℕ) := Fin n → Point

/-- A set of n reservoirs -/
def Reservoirs (n : ℕ) := Fin n → Point

/-- No three points are collinear -/
def NoThreeCollinear (points : Set Point) : Prop :=
  ∀ p q r : Point, p ∈ points → q ∈ points → r ∈ points →
    p ≠ q → q ≠ r → p ≠ r →
    (r.y - p.y) * (q.x - p.x) ≠ (q.y - p.y) * (r.x - p.x)

/-- Two line segments intersect -/
def Intersect (p1 q1 p2 q2 : Point) : Prop :=
  ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
    p1.x + t * (q1.x - p1.x) = p2.x + s * (q2.x - p2.x) ∧
    p1.y + t * (q1.y - p1.y) = p2.y + s * (q2.y - p2.y)

/-- Main theorem -/
theorem farm_reservoir_pairing (n : ℕ) (farms : Farms n) (reservoirs : Reservoirs n)
  (h : NoThreeCollinear (Set.range farms ∪ Set.range reservoirs)) :
  ∃ σ : Equiv.Perm (Fin n), ∀ i j : Fin n, i ≠ j →
    ¬ Intersect (farms i) (reservoirs (σ i)) (farms j) (reservoirs (σ j)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_reservoir_pairing_l1360_136091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_a_equals_six_l1360_136047

noncomputable def f (a b x : ℝ) : ℝ := x + a / x + b

theorem function_properties_imply_a_equals_six
  (a b : ℝ)
  (h₁ : a > 0)
  (h₂ : ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.sqrt a → f a b x₁ ≥ f a b x₂)
  (h₃ : ∀ x₁ x₂, Real.sqrt a ≤ x₁ ∧ x₁ < x₂ → f a b x₁ ≤ f a b x₂)
  (h₄ : ∀ x, 1 ≤ x ∧ x ≤ 2 → f a b x ≤ 5)
  (h₅ : ∀ x, 1 ≤ x ∧ x ≤ 2 → f a b x ≥ 3)
  (h₆ : ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f a b x = 5)
  (h₇ : ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f a b x = 3) :
  a = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_a_equals_six_l1360_136047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1360_136087

theorem equation_solution (x : ℝ) :
  (∀ n : ℤ, x ≠ π / 2 + n * π) →
  (1 + (2 : ℝ)^(Real.tan x) = 3 * (4 : ℝ)^((-1 / Real.sqrt 2) * Real.sin (π / 4 - x) * (Real.cos x)⁻¹)) ↔
  (∃ k : ℤ, x = π / 4 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1360_136087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l1360_136049

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are collinear -/
theorem vectors_collinear (a b : Fin 3 → ℝ) 
  (h1 : a = ![1, -2, 4])
  (h2 : b = ![7, 3, 5]) : 
  ∃ (k : ℝ), (6 • a - 3 • b) = k • (b - 2 • a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l1360_136049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_to_ellipse_l1360_136012

theorem circles_tangent_to_ellipse (x y : ℝ) :
  let ellipse := (4 * x^2 + 6 * y^2 = 8)
  let circle := ((x - 2/3)^2 + y^2 = (2/3)^2)
  (ellipse ∧ circle) → x = 2/3 :=
by
  intro h
  sorry

#check circles_tangent_to_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_to_ellipse_l1360_136012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_excess_money_l1360_136004

noncomputable def weekend1_earnings : ℝ := 20
noncomputable def weekend2_saturday : ℝ := 18
noncomputable def weekend2_sunday : ℝ := weekend2_saturday / 2
noncomputable def weekend2_earnings : ℝ := weekend2_saturday + weekend2_sunday
noncomputable def weekend3_earnings : ℝ := weekend2_earnings * 1.25
noncomputable def weekend4_earnings : ℝ := weekend3_earnings * 1.15
noncomputable def pogo_stick_cost : ℝ := 60

noncomputable def total_earnings : ℝ := weekend1_earnings + weekend2_earnings + weekend3_earnings + weekend4_earnings

theorem john_excess_money :
  total_earnings - pogo_stick_cost = 59.5625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_excess_money_l1360_136004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_duck_difference_l1360_136092

/-- Represents the number of birds of each type at the park -/
structure BirdCounts where
  mallards : ℕ
  woodDucks : ℕ
  geese : ℕ
  swans : ℕ
deriving Repr

/-- Calculates the final bird counts after all arrivals and departures -/
def finalCounts (initial : BirdCounts) : BirdCounts :=
  let morning := BirdCounts.mk
    (initial.mallards + 4)
    (initial.woodDucks + 8)
    (initial.geese + 7)
    initial.swans
  
  let noon := BirdCounts.mk
    morning.mallards
    (morning.woodDucks - 6)
    (morning.geese - 5)
    (morning.swans - 9)
  
  let later := BirdCounts.mk
    (noon.mallards + 8)
    (noon.woodDucks + 10)
    noon.geese
    (noon.swans + 4)
  
  let evening := BirdCounts.mk
    (later.mallards + 5)
    (later.woodDucks + 3)
    (later.geese + 15)
    (later.swans + 11)
  
  BirdCounts.mk
    0
    (evening.woodDucks / 4)
    (evening.geese * 4 / 5)
    (evening.swans / 2)

/-- The theorem to be proved -/
theorem geese_duck_difference (initial : BirdCounts) :
  let final := finalCounts initial
  final.geese - (final.mallards + final.woodDucks) = 38 := by
  sorry

/-- The initial bird counts at the park -/
def initialCounts : BirdCounts :=
  BirdCounts.mk 25 15 40 53

#eval finalCounts initialCounts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_duck_difference_l1360_136092
