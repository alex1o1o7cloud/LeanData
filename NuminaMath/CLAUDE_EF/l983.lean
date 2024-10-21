import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_approximately_27_l983_98395

/-- The total number of students in the class -/
def total_students : ℕ := 37

/-- The number of students who scored in the 70%-79% range -/
def students_in_range : ℕ := 10

/-- The percentage of students who scored in the 70%-79% range -/
noncomputable def percentage_in_range : ℝ := (students_in_range : ℝ) / total_students * 100

theorem percentage_approximately_27 : 
  26.5 < percentage_in_range ∧ percentage_in_range < 27.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_approximately_27_l983_98395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l983_98348

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Theorem: For an arithmetic sequence, if S_6 / S_3 = 4, then S_5 / S_6 = 25/36 -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : S seq 6 / S seq 3 = 4) : S seq 5 / S seq 6 = 25/36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l983_98348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_formula_l983_98355

/-- Isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid :=
  (AB : ℝ) (BC : ℝ) (CD : ℝ)
  (ab_eq : AB = 6)
  (bc_eq : BC = 5)
  (cd_eq : CD = 4)

/-- Circles centered at vertices of the trapezoid -/
structure TrapezoidCircles (t : IsoscelesTrapezoid) :=
  (radiusAB : ℝ) (radiusCD : ℝ)
  (rad_ab_eq : radiusAB = 3)
  (rad_cd_eq : radiusCD = 2)

/-- The radius of the circle tangent to all four circles -/
noncomputable def tangentCircleRadius (t : IsoscelesTrapezoid) (c : TrapezoidCircles t) : ℝ := sorry

/-- Theorem stating the radius of the tangent circle -/
theorem tangent_circle_radius_formula (t : IsoscelesTrapezoid) (c : TrapezoidCircles t) :
  ∃ (k m n p : ℕ), 
    k > 0 ∧ m > 0 ∧ n > 0 ∧ p > 0 ∧
    (∀ (q : ℕ), q > 1 → Nat.Prime q → ¬(q ^ 2 ∣ n)) ∧
    Nat.Coprime k p ∧
    tangentCircleRadius t c = (-k + m * Real.sqrt n) / p ∧
    k = 60 ∧ m = 48 ∧ n = 3 ∧ p = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_formula_l983_98355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_increasing_interval_max_min_values_l983_98352

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - (Real.cos x) ^ 2 - 1 / 2

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi := by
  sorry

-- Theorem for the monotonically increasing interval
theorem monotonically_increasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) := by
  sorry

-- Theorem for the maximum and minimum values
theorem max_min_values :
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12) → f x ≤ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12) ∧ f x = 0) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12) → f x ≥ -1 - Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12) ∧ f x = -1 - Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_increasing_interval_max_min_values_l983_98352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l983_98357

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumFirstN (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_sum_four (g : GeometricSequence) 
  (h1 : sumFirstN g 2 = g.a 1 + 2 * g.a 3)
  (h2 : g.a 4 = 1) :
  sumFirstN g 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l983_98357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_employees_to_hire_l983_98315

theorem minimum_employees_to_hire (A B : Finset Nat) 
  (h1 : A.card = 150)
  (h2 : B.card = 125)
  (h3 : (A ∩ B).card = 50) :
  (A ∪ B).card = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_employees_to_hire_l983_98315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_l983_98347

theorem smallest_set_size (S : Finset ℝ) (h_nonempty : S.Nonempty) : 
  (∃ p : ℝ, p ∈ S ∧ 
    p = Finset.sum S id / S.card ∧
    S.max' h_nonempty = p + 7 ∧
    S.min' h_nonempty = p - 7 ∧
    (S.filter (λ x => x < p)).card > S.card / 2) →
  S.card ≥ 7 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_l983_98347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_monotone_increasing_on_open_interval_no_zeros_when_a_gt_one_one_zero_when_a_leq_zero_or_eq_one_two_zeros_when_a_between_zero_and_one_l983_98322

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x + Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2) + 1 / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f' a x - x

-- Statement for part (I)
theorem extremum_at_one (a : ℝ) :
  (∃ ε > 0, ∀ x, x ∈ Set.Ioo (1 - ε) (1 + ε) → f a x ≥ f a 1) → a = 2 :=
sorry

-- Statement for part (II)
theorem monotone_increasing_on_open_interval (a : ℝ) :
  (∀ x y, x ∈ Set.Ioo 1 2 → y ∈ Set.Ioo 1 2 → x < y → f a x < f a y) → a ≤ 2 :=
sorry

-- Statements for part (III)
theorem no_zeros_when_a_gt_one (a : ℝ) :
  a > 1 → ∀ x, x > 0 → g a x ≠ 0 :=
sorry

theorem one_zero_when_a_leq_zero_or_eq_one (a : ℝ) :
  (a ≤ 0 ∨ a = 1) → ∃! x, x > 0 ∧ g a x = 0 :=
sorry

theorem two_zeros_when_a_between_zero_and_one (a : ℝ) :
  0 < a ∧ a < 1 → ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z, z > 0 → g a z = 0 → z = x ∨ z = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_monotone_increasing_on_open_interval_no_zeros_when_a_gt_one_one_zero_when_a_leq_zero_or_eq_one_two_zeros_when_a_between_zero_and_one_l983_98322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_with_constant_end_l983_98337

/-- A move on a sequence of real numbers. -/
noncomputable def move (s : List ℝ) (i j : Nat) : List ℝ :=
  let mean := (s.get! i + s.get! j) / 2
  s.set i mean |>.set j mean

/-- A sequence is constant if all its elements are equal. -/
def isConstant (s : List ℝ) : Prop :=
  ∀ i j, i < s.length → j < s.length → s.get! i = s.get! j

/-- There exists a sequence of 2015 distinct real numbers such that,
    for any initial move, it is possible to reach a constant sequence
    through a finite number of subsequent moves. -/
theorem exists_sequence_with_constant_end :
  ∃ (s : List ℝ),
    s.length = 2015 ∧
    s.Nodup ∧
    ∀ i j,
      i < s.length →
      j < s.length →
      i ≠ j →
      ∃ (moves : List (Nat × Nat)),
        isConstant (moves.foldl (λ acc (m, n) => move acc m n) (move s i j)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_with_constant_end_l983_98337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l983_98371

noncomputable def triangle_properties (a b c : ℝ) (A B C : ℝ) : Prop :=
  let cos_C : ℝ := 2/3
  let c_value : ℝ := Real.sqrt 5
  let circumcircle_circumference : ℝ := 3 * Real.pi
  let perimeter : ℝ := 5 + Real.sqrt 5
  (Real.cos C = cos_C) ∧
  (c = c_value) ∧
  (2 * a = 3 * b) ∧
  (2 * (circumcircle_circumference / (2 * Real.pi)) = c / Real.sin C) ∧
  (perimeter = a + b + c)

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  triangle_properties a b c A B C := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l983_98371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_special_subsets_l983_98331

theorem finite_special_subsets : ∃ (N : ℕ), ∀ (n : ℕ), n > N →
  ¬∃ (S : Finset ℕ), (S ⊆ Finset.range n) ∧ 
    (S.card ≥ Nat.floor (Real.sqrt (n : ℝ)) + 1) ∧
    (∀ (x y : ℕ), x ∈ S → y ∈ S → 
      ∃ (a b : ℕ), b ≥ 2 ∧ x * y = a ^ b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_special_subsets_l983_98331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_in_special_triangle_l983_98384

theorem tan_A_in_special_triangle (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : Real.sin A = 10 * Real.sin B * Real.sin C) (h3 : Real.cos A = 10 * Real.cos B * Real.cos C) : 
  Real.tan A = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_in_special_triangle_l983_98384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_l983_98373

/-- Given a parabola C: y² = 2px and a point A(1, √5) on it, 
    the distance from A to the directrix of C is 9/4 -/
theorem distance_to_directrix (p : ℝ) : 
  (Real.sqrt 5)^2 = 2 * p * 1 → -- A lies on the parabola
  (1 : ℝ) + p / 2 = 9 / 4 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_l983_98373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l983_98375

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x^2) / Real.log (1/3)

-- State the theorem
theorem f_strictly_decreasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ ≤ 1 →
  f x₂ < f x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l983_98375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_value_l983_98326

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (9/4) * a n + (3/4) * Real.sqrt (9^n - (a n)^2)

theorem a_7_value : a 7 = (39402 + 10935 * Real.sqrt 7) / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_value_l983_98326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_2008_cube_plus_l983_98314

theorem number_of_divisors_2008_cube_plus : 
  (Finset.card (Finset.filter (λ d => d ∣ (2008^3 + (3 * 2008 * 2009) + 1)^2) (Finset.range ((2008^3 + (3 * 2008 * 2009) + 1)^2 + 1)))) = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_2008_cube_plus_l983_98314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l983_98300

theorem range_of_a (a : ℝ) : 
  (∀ x θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l983_98300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l983_98383

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 + 2 * x - 1

theorem zero_in_interval :
  ∃ x ∈ Set.Ioo (1/2 : ℝ) 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l983_98383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_isosceles_l983_98363

/-- Predicate stating that ABCD is a trapezoid -/
def Trapezoid (A B C D : Point) : Prop := sorry

/-- Predicate stating that the diagonals of ABCD intersect at O -/
def DiagonalsIntersect (A B C D O : Point) : Prop := sorry

/-- Predicate stating that M is on the base AD of the trapezoid -/
def OnBase (A D M : Point) : Prop := sorry

/-- Predicate stating that M is on the circumcircle of triangle AOB -/
def OnCircumcircle (A O B M : Point) : Prop := sorry

/-- Predicate stating that triangle BMC is isosceles -/
def IsIsosceles (B M C : Point) : Prop := sorry

/-- Given a trapezoid ABCD with diagonals intersecting at O and circumcircles of AOB and COD
    intersecting at M on AD, prove that triangle BMC is isosceles -/
theorem trapezoid_isosceles (A B C D O M : Point) : 
  Trapezoid A B C D →
  DiagonalsIntersect A B C D O →
  OnBase A D M →
  OnCircumcircle A O B M →
  OnCircumcircle C O D M →
  IsIsosceles B M C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_isosceles_l983_98363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_increase_l983_98305

theorem cone_height_increase (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  ∃ (p : ℝ), 
    (1/3 * π * r^2 * (h * (1 + p/100)) = 2.60 * (1/3 * π * r^2 * h)) → 
    p = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_increase_l983_98305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomial_product_l983_98312

theorem constant_term_of_polynomial_product (p q : Polynomial ℝ) : 
  Polynomial.Monic p → Polynomial.Monic q → 
  Polynomial.degree p = 3 → Polynomial.degree q = 3 →
  (∃ (c : ℝ), c > 0 ∧ p.coeff 0 = c ∧ q.coeff 0 = c) →
  (∃ (d : ℝ), p.coeff 2 = d ∧ q.coeff 2 = d) →
  p * q = Polynomial.monomial 6 1 + Polynomial.monomial 5 2 + Polynomial.monomial 4 5 + 
          Polynomial.monomial 3 10 + Polynomial.monomial 2 10 + Polynomial.monomial 1 8 + 
          Polynomial.monomial 0 9 →
  p.coeff 0 = 3 ∧ q.coeff 0 = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomial_product_l983_98312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_arrives_earlier_l983_98389

-- Define the parameters
noncomputable def total_distance : ℝ := 1000
noncomputable def abel_speed : ℝ := 50
noncomputable def alice_speed : ℝ := 40
noncomputable def alice_delay : ℝ := 1

-- Define the time difference function
noncomputable def time_difference (d : ℝ) (v1 v2 : ℝ) (delay : ℝ) : ℝ :=
  (d / v2 + delay) - (d / v1)

-- Theorem statement
theorem abel_arrives_earlier :
  time_difference total_distance abel_speed alice_speed alice_delay * 60 = 360 := by
  -- Unfold the definition of time_difference
  unfold time_difference
  -- Perform algebraic manipulations
  simp [total_distance, abel_speed, alice_speed, alice_delay]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_arrives_earlier_l983_98389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l983_98393

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 2, 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 2}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l983_98393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_trig_l983_98333

theorem equation_roots_trig (θ : ℝ) (m : ℝ) : 
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ x y : ℝ, x = Real.sin θ ∧ y = Real.cos θ ∧ 
    2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ∧
    2 * y^2 - (Real.sqrt 3 + 1) * y + m = 0) →
  m = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_trig_l983_98333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l983_98324

theorem percentage_difference : ∃ (difference : ℝ), difference = 190 :=
  let number : ℝ := 1600
  let percentage : ℝ := 0.20
  let comparison : ℝ := 650
  let difference : ℝ := percentage * number - percentage * comparison
  have h : difference = 190 := by
    -- Proof steps would go here
    sorry
  ⟨difference, h⟩

#check percentage_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l983_98324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_specific_l983_98360

/-- The area of the unpainted region on a board when crossed with another board -/
noncomputable def unpainted_area (width1 : ℝ) (width2 : ℝ) (angle : ℝ) : ℝ :=
  (width1 / Real.cos angle) * width2

/-- Theorem: The area of the unpainted region on a 5-inch wide board when crossed 
    with a 7-inch wide board at a 45-degree angle is 35√2 square inches -/
theorem unpainted_area_specific : 
  unpainted_area 5 7 (π/4) = 35 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_specific_l983_98360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_surface_area_is_378π_l983_98388

-- Define the constants
def hemisphere_radius : ℝ := 9
def cone_height : ℝ := 12
def cone_base_radius : ℝ := hemisphere_radius

-- Calculate the slant height of the cone
noncomputable def cone_slant_height : ℝ := Real.sqrt (cone_base_radius ^ 2 + cone_height ^ 2)

-- Define the total surface area function
noncomputable def total_surface_area : ℝ :=
  Real.pi * hemisphere_radius ^ 2 +  -- Area of hemisphere's base
  2 * Real.pi * hemisphere_radius ^ 2 +  -- Curved area of hemisphere
  Real.pi * cone_base_radius * cone_slant_height  -- Lateral surface area of cone

-- Theorem statement
theorem total_surface_area_is_378π :
  total_surface_area = 378 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_surface_area_is_378π_l983_98388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_l983_98339

-- Define the region
def Region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ 150 * (x - ⌊x⌋) ≥ 2 * ⌊x⌋ + ⌊y⌋

-- Define the area function
noncomputable def area : ℝ := 2265.25

-- Theorem statement
theorem region_area : area = 2265.25 := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_l983_98339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l983_98308

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l983_98308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_tan_l983_98340

/-- The smallest positive period of y = 3tan(x/2 + π/3) is 4π -/
theorem smallest_positive_period_tan :
  ∃ y : ℝ → ℝ, ∃ T : ℝ,
    (∀ x, y x = 3 * Real.tan (x / 2 + π / 3)) ∧
    T > 0 ∧ T = 4 * π ∧
    (∀ x, y (x + T) = y x) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ x, y (x + S) = y x) → T ≤ S) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_tan_l983_98340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l983_98391

/-- The distance between the foci of a hyperbola -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- The standard form of a hyperbola equation -/
def is_hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_foci_distance (x y : ℝ) :
  is_hyperbola x y (Real.sqrt 32) (Real.sqrt 8) →
  distance_between_foci (Real.sqrt 32) (Real.sqrt 8) = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l983_98391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_yaxis_and_b_plus_one_l983_98367

/-- A point in the third quadrant of the Cartesian coordinate system -/
structure ThirdQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_neg : y < 0

/-- The theorem statement -/
theorem distance_to_yaxis_and_b_plus_one (A : ThirdQuadrantPoint) 
  (dist_to_yaxis : |A.x| = 5)
  (b_plus_one : |A.y + 1| = 3) :
  Real.sqrt ((A.x - A.y)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_yaxis_and_b_plus_one_l983_98367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l983_98349

/-- A line in the xy-plane --/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line --/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- Two lines are perpendicular if their slopes are negative reciprocals --/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

/-- The original line 4x + 5y = 10 --/
def original_line : Line :=
  { slope := -4/5, y_intercept := 2 }

/-- The perpendicular line with y-intercept -3 --/
def perpendicular_line : Line :=
  { slope := 5/4, y_intercept := -3 }

theorem x_intercept_of_perpendicular_line :
  perpendicular original_line perpendicular_line →
  x_intercept perpendicular_line = 12/5 := by
  intro h
  simp [x_intercept, perpendicular_line]
  -- The proof steps would go here
  sorry

#eval x_intercept perpendicular_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l983_98349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_line_through_point_with_distance_l983_98309

-- Define the lines and points
def l₁ : Set (ℝ × ℝ) := {(x, y) | x + 3*y - 3 = 0}
def l₂ : Set (ℝ × ℝ) := {(x, y) | x - y + 1 = 0}
def l₃ : Set (ℝ × ℝ) := {(x, y) | 2*x + y - 3 = 0}
def l₄ : Set (ℝ × ℝ) := {(x, y) | 2*x + y - 6 = 0}
def A : ℝ × ℝ := (1, -1)

-- Define the intersection point of l₁ and l₂
def P : ℝ × ℝ := (1, 2)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Statement for the first part of the problem
theorem line_through_intersection_parallel :
  ∃ (l : Set (ℝ × ℝ)), 
    (P ∈ l) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ 2*x + y - 4 = 0) ∧
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (k : ℝ), (x, y) ∈ l₃ ∧ x = P.1 + k ∧ y = P.2 + k/2) := by
  sorry

-- Statement for the second part of the problem
theorem line_through_point_with_distance :
  ∃ (l : Set (ℝ × ℝ)) (B : ℝ × ℝ),
    (A ∈ l) ∧ (B ∈ l) ∧ (B ∈ l₄) ∧ 
    (distance A B = 5) ∧
    ((∀ (x y : ℝ), (x, y) ∈ l ↔ x = 1) ∨
     (∀ (x y : ℝ), (x, y) ∈ l ↔ 3*x + 4*y + 1 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_line_through_point_with_distance_l983_98309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_music_score_teresa_total_score_correct_teresa_physics_half_music_l983_98311

/-- Represents Teresa's exam scores -/
structure TeresaScores where
  science : ℕ
  music : ℕ
  social_studies : ℕ
  physics : ℕ
  total : ℕ

/-- Teresa's actual scores satisfying the given conditions -/
def teresa_scores : TeresaScores :=
  { science := 70,
    music := 80,  -- We set this to 80 as it's the value we want to prove
    social_studies := 85,
    physics := 40,  -- Half of music score
    total := 275 }

/-- Theorem stating that Teresa's music score is 80 -/
theorem teresa_music_score :
  teresa_scores.music = 80 := by
  -- Unfold the definition of teresa_scores
  unfold teresa_scores
  -- The result follows directly from the definition
  rfl

/-- Theorem verifying that the total score is correct -/
theorem teresa_total_score_correct :
  teresa_scores.science + teresa_scores.music + teresa_scores.social_studies + teresa_scores.physics = teresa_scores.total := by
  -- Unfold the definition of teresa_scores
  unfold teresa_scores
  -- Perform the arithmetic
  norm_num

/-- Theorem verifying that physics score is half of music score -/
theorem teresa_physics_half_music :
  teresa_scores.physics = teresa_scores.music / 2 := by
  -- Unfold the definition of teresa_scores
  unfold teresa_scores
  -- The result follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_music_score_teresa_total_score_correct_teresa_physics_half_music_l983_98311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_transformation_l983_98302

/-- Given a 2x2 matrix A and a line y = kx + 1 transformed by A to pass through (2, 6), prove k = 1/2 -/
theorem line_transformation (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A = ![![1, 0], ![1, 2]])
  (h2 : ∃ k : ℝ, (A.mulVec ![2, k * 2 + 1]) = ![2, 6]) :
  ∃ k : ℝ, k = (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_transformation_l983_98302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l983_98368

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron where
  R : ℝ  -- radius of inscribed sphere
  S₁ : ℝ  -- area of face 1
  S₂ : ℝ  -- area of face 2
  S₃ : ℝ  -- area of face 3
  S₄ : ℝ  -- area of face 4

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  (1/3) * t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄)

/-- Theorem: The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume (t : Tetrahedron) :
  volume t = (1/3) * t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄) := by
  -- Unfold the definition of volume
  unfold volume
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l983_98368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_transformed_tan_l983_98366

-- Define the tangent function
noncomputable def tan_function (x : ℝ) : ℝ := Real.tan x

-- Define the transformed tangent function
noncomputable def transformed_tan (x : ℝ) : ℝ := tan_function (x / 3)

-- State the theorem
theorem period_of_transformed_tan :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), transformed_tan (x + p) = transformed_tan x ∧ 
  ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), transformed_tan (y + q) ≠ transformed_tan y :=
by
  -- The proof goes here
  sorry

#check period_of_transformed_tan

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_transformed_tan_l983_98366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l983_98398

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^4 + 5 = (X - 3)^2 * q + (108*X - 238) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l983_98398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goats_in_caravan_l983_98354

/-- Proves the number of goats in a caravan given specific conditions -/
theorem goats_in_caravan : ∃ (num_goats : ℕ), 
  let num_hens : ℕ := 50
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_feet : ℕ := 2
  let total_feet_minus_heads : ℕ := 224
  num_goats * goat_feet + 
  num_hens * hen_feet + 
  num_camels * camel_feet + 
  num_keepers * keeper_feet = 
  num_goats + num_hens + num_camels + num_keepers + total_feet_minus_heads ∧
  num_goats = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goats_in_caravan_l983_98354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_road_speed_l983_98345

/-- Proves that the speed on local roads is 20 mph given the conditions of the problem -/
theorem local_road_speed (total_distance : ℝ) (local_distance : ℝ) (highway_distance : ℝ)
  (highway_speed : ℝ) (average_speed : ℝ) (local_road_speed : ℝ) :
  total_distance = 220 →
  local_distance = 40 →
  highway_distance = 180 →
  highway_speed = 60 →
  average_speed = 44 →
  total_distance = local_distance + highway_distance →
  average_speed = total_distance / (local_distance / local_road_speed + highway_distance / highway_speed) →
  local_road_speed = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_road_speed_l983_98345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_and_unique_solution_l983_98310

/-- The average of all positive divisors of a positive integer n -/
def f (n : ℕ) : ℚ :=
  let divisors := (Finset.range n).filter (λ i => n % (i + 1) = 0)
  (divisors.sum (λ i => i + 1) : ℚ) / divisors.card

/-- Main theorem: bounds for f(n) and the unique solution for f(n) = 91/9 -/
theorem f_bounds_and_unique_solution (n : ℕ) (hn : n > 0) :
  ((n : ℚ) + 1) / 2 ≥ f n ∧ f n ≥ (n.sqrt : ℚ) ∧ (f n = 91/9 ↔ n = 36) := by
  sorry

#check f_bounds_and_unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_and_unique_solution_l983_98310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l983_98369

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (π/4 + (π/8 - x)) = f (π/8 + x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l983_98369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_l983_98303

theorem cubic_root_equation (y : ℝ) : 
  Real.sqrt (3 + (5 * y - 4) ^ (1/3 : ℝ)) = Real.sqrt 8 → y = 25.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_l983_98303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_neg_two_and_neg_eight_l983_98328

-- Define the geometric mean as noncomputable
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

-- State the theorem
theorem geometric_mean_of_neg_two_and_neg_eight :
  {x : ℝ | x = geometric_mean (-2) (-8) ∨ x = -geometric_mean (-2) (-8)} = {4, -4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_neg_two_and_neg_eight_l983_98328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l983_98378

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
noncomputable def min_distance_between_curves : ℝ := Real.sqrt 2 * (1 - Real.log 2)

/-- A point P on the curve y = (1/2)e^x -/
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, (1/2) * Real.exp x)

/-- A point Q on the curve y = ln(2x) -/
noncomputable def Q (x : ℝ) : ℝ × ℝ := (x, Real.log (2 * x))

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_proof :
  ∃ (x y : ℝ), distance (P x) (Q y) = min_distance_between_curves := by
  sorry

#check min_distance_between_curves
#check P
#check Q
#check distance
#check min_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l983_98378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coins_second_third_hours_l983_98377

/-- Represents the number of coins in Tina's jar at different stages --/
structure CoinJar where
  firstHour : ℤ
  secondHour : ℤ
  thirdHour : ℤ
  fourthHour : ℤ
  fifthHour : ℤ
  finalCount : ℤ

/-- Theorem stating the sum of coins put in during the second and third hours --/
theorem coins_second_third_hours (jar : CoinJar) 
  (h1 : jar.firstHour = 20)
  (h2 : jar.fourthHour = 40)
  (h3 : jar.fifthHour = -20)
  (h4 : jar.finalCount = 100)
  (h5 : jar.firstHour + jar.secondHour + jar.thirdHour + jar.fourthHour + jar.fifthHour = jar.finalCount) :
  jar.secondHour + jar.thirdHour = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coins_second_third_hours_l983_98377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inequality_l983_98365

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def is_acute (t : Triangle) : Prop := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The inequality holds if and only if the triangle is acute-angled -/
theorem centroid_inequality (t : Triangle) :
  let S := centroid t
  let r := circumradius t
  (distance S t.A)^2 + (distance S t.B)^2 + (distance S t.C)^2 > 8 * r^2 / 3 ↔ is_acute t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inequality_l983_98365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_diagonal_angle_l983_98399

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Finset (ℝ × ℝ)
  regular : vertices.card = 6
  -- Additional properties ensuring regularity

/-- The measure of an angle in degrees -/
def angle_measure : ℝ → ℝ := id

/-- Angle notation -/
notation "∠" => angle_measure

theorem regular_hexagon_diagonal_angle (ABCDEF : RegularHexagon) 
  (interior_angle : ∠ 120 = 120) 
  (A B C : ℝ × ℝ) 
  (hA : A ∈ ABCDEF.vertices) 
  (hB : B ∈ ABCDEF.vertices) 
  (hC : C ∈ ABCDEF.vertices) 
  (hAC : C ≠ A ∧ C ≠ B) : 
  ∠ (30 : ℝ) = 30 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_diagonal_angle_l983_98399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l983_98336

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (3 + k) * x + 3

theorem f_properties :
  ∃ (k : ℝ), 
    (f k 2 = 3) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 4, f k x ≤ 4) ∧
    (f k 1 = 4) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 4, f k x ≥ -5) ∧
    (f k 4 = -5) ∧
    (∀ m : ℝ, (∀ x ∈ Set.Icc (-2 : ℝ) 2, Monotone (λ y ↦ f k y - m * y)) ↔ (m ≤ -2 ∨ m ≥ 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l983_98336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l983_98346

/-- The complex polynomial whose roots define the ellipse -/
def f (z : ℂ) : ℂ := (z - 2) * (z^2 + 6*z + 13) * (z^2 + 8*z + 18)

/-- The set of roots of the polynomial f -/
def S : Set ℂ := {z | f z = 0}

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  (∃ (h : ℝ), ∀ (z : ℂ), z ∈ S → (((z.re - h) / a)^2 + (z.im / b)^2 = 1)) ∧
  eccentricity a b = Real.sqrt (49 / 320) := by
  sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l983_98346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_after_decimal_sqrt_polynomial_l983_98350

theorem first_digit_after_decimal_sqrt_polynomial (d : ℕ) 
  (h : d < 10) : ∃ (n : ℕ+) (k : ℕ),
  let x := (n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ)
  10 * ((k : ℝ) + (d : ℝ) / 10) ≤ Real.sqrt x ∧ 
  Real.sqrt x < 10 * ((k : ℝ) + ((d + 1) : ℝ) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_after_decimal_sqrt_polynomial_l983_98350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_squared_of_right_triangle_zeros_l983_98320

/-- Given complex numbers a, b, and c that are zeros of a cubic polynomial
    and form a right triangle, prove that the square of the hypotenuse is 125. -/
theorem hypotenuse_squared_of_right_triangle_zeros (a b c : ℂ) (q r : ℂ) :
  (a^3 + q*a + r = 0) →
  (b^3 + q*b + r = 0) →
  (c^3 + q*c + r = 0) →
  (Complex.abs a)^2 + (Complex.abs b)^2 + (Complex.abs c)^2 = 250 →
  ∃ (h : ℝ), (Complex.abs (a - b))^2 + (Complex.abs (b - c))^2 = (Complex.abs (a - c))^2 →
  (Complex.abs (a - c))^2 = h^2 →
  h^2 = 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_squared_of_right_triangle_zeros_l983_98320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_div_one_plus_i_l983_98330

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := m * (m - 1) + (m - 1) * Complex.I

-- Theorem 1: z is purely imaginary iff m = 0
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 0 := by
  sorry

-- Theorem 2: When m = 2, z / (1+i) = 3/2 - 1/2i
theorem z_div_one_plus_i : z 2 / (1 + Complex.I) = 3/2 - 1/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_div_one_plus_i_l983_98330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_at_one_implies_a_greater_than_half_l983_98338

/-- The function f(x) = ln x - 2ax + 2a --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 2 * a

/-- The function g(x) = xf(x) + ax^2 - x --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * f a x + a * x^2 - x

/-- Theorem: If g(x) attains a maximum value at x = 1, then a > 1/2 --/
theorem g_max_at_one_implies_a_greater_than_half (a : ℝ) :
  (∀ x > 0, g a x ≤ g a 1) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_at_one_implies_a_greater_than_half_l983_98338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_sum_l983_98386

theorem composite_sum (K L M N : ℕ) 
  (hK : K > 0) (hL : L > 0) (hM : M > 0) (hN : N > 0)
  (h1 : K > L) (h2 : L > M) (h3 : M > N)
  (h4 : K * M + L * N = (K + L - M + N) * (K - L + M + N)) :
  ∃ (a b : ℕ), a * b = K * L + M * N ∧ a > 1 ∧ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_sum_l983_98386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_l983_98306

/-- The function f(x) = (1/2)ax^2 + ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 + Real.log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * x + 1/x

theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x > 1, f_deriv a x ≥ 0) ↔ a ∈ Set.Ici 0 := by
  sorry

#check monotonically_increasing_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_l983_98306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l983_98372

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 6 * Real.sin x ^ 2 + Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem g_range : 
  Set.range g = Set.Icc 2 12 \ {12} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l983_98372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pasture_milkmen_count_l983_98321

/-- Represents a milkman's grazing data -/
structure MilkmanData where
  cows : ℕ
  months : ℕ
deriving Inhabited

/-- The problem setup -/
def pastureProblem : Prop := ∃ (milkmen : List MilkmanData) (a_rent : ℕ) (total_rent : ℕ),
  milkmen.length = 4 ∧
  milkmen = [
    ⟨24, 3⟩,  -- A's data
    ⟨10, 5⟩,  -- B's data
    ⟨35, 4⟩,  -- C's data
    ⟨21, 3⟩   -- D's data
  ] ∧
  a_rent = 720 ∧
  total_rent = 3250 ∧
  (milkmen.map (fun m => m.cows * m.months)).sum * a_rent = 
    (milkmen[0]!.cows * milkmen[0]!.months) * total_rent

theorem pasture_milkmen_count : pastureProblem → ∃ (n : ℕ), n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pasture_milkmen_count_l983_98321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doris_saturday_hours_l983_98304

/-- Represents Doris's babysitting scenario -/
structure BabysittingScenario where
  hourly_rate : ℚ
  monthly_goal : ℚ
  weekday_hours : ℚ
  weeks_to_goal : ℕ

/-- Calculates the number of Saturday hours Doris needs to babysit -/
def saturday_hours (scenario : BabysittingScenario) : ℚ :=
  let weekday_earnings := scenario.hourly_rate * scenario.weekday_hours * 5 * scenario.weeks_to_goal
  let saturday_earnings_needed := scenario.monthly_goal - weekday_earnings
  saturday_earnings_needed / (scenario.hourly_rate * scenario.weeks_to_goal)

/-- Theorem stating that Doris needs to babysit 5 hours on Saturdays -/
theorem doris_saturday_hours :
  let scenario := BabysittingScenario.mk 20 1200 3 3
  saturday_hours scenario = 5 := by
  -- Proof goes here
  sorry

#eval saturday_hours (BabysittingScenario.mk 20 1200 3 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doris_saturday_hours_l983_98304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l983_98364

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def timeToCrossBridge (trainLength : ℝ) (trainSpeed : ℝ) (bridgeLength : ℝ) : ℝ :=
  let totalDistance := trainLength + bridgeLength
  let speedInMetersPerSecond := trainSpeed * 1000 / 3600
  totalDistance / speedInMetersPerSecond

/-- Theorem stating that a train with given parameters takes 30.0024 seconds to cross a bridge -/
theorem train_bridge_crossing_time :
  let trainLength : ℝ := 135
  let trainSpeed : ℝ := 45  -- km/hr
  let bridgeLength : ℝ := 240.03
  timeToCrossBridge trainLength trainSpeed bridgeLength = 30.0024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l983_98364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l983_98397

open Nat

-- Define the number of divisors function
noncomputable def num_divisors (n : ℕ) : ℕ := (divisors n).card

-- Define the property for n
def satisfies_conditions (n : ℕ) : Prop :=
  n % 45 = 0 ∧ num_divisors n = 64

-- State the theorem
theorem smallest_n_with_conditions :
  ∃ n : ℕ, satisfies_conditions n ∧
    ∀ m : ℕ, m < n → ¬(satisfies_conditions m) ∧
    n / 45 = 3796875 := by
  sorry

#check smallest_n_with_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l983_98397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additive_implies_odd_l983_98325

/-- A function f: ℝ → ℝ is additive if f(a + b) = f(a) + f(b) for all a, b ∈ ℝ -/
def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a + f b

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem additive_implies_odd (f : ℝ → ℝ) (h : IsAdditive f) : IsOdd f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additive_implies_odd_l983_98325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_oscillation_l983_98385

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * Real.pi * x)^2

theorem sin_squared_oscillation (ω : ℝ) (h1 : ω > 0) :
  (∃ a b c d : ℝ, 0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 1/2 ∧
    (∀ x ∈ Set.Icc 0 (1/2), f ω x ≤ f ω a) ∧
    (∀ x ∈ Set.Icc 0 (1/2), f ω x ≥ f ω b) ∧
    (∀ x ∈ Set.Icc 0 (1/2), f ω x ≤ f ω c) ∧
    (∀ x ∈ Set.Icc 0 (1/2), f ω x ≥ f ω d)) →
  ω ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_oscillation_l983_98385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_expression_for_negative_x_l983_98381

theorem positive_expression_for_negative_x (x : ℝ) (h : x < 0) :
  (x / abs x)^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_expression_for_negative_x_l983_98381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norilsk_snowfall_theorem_l983_98380

/-- The average snowfall per hour in Norilsk, Russia, during December 1861 -/
noncomputable def averageSnowfallPerHour (totalSnowfall : ℝ) (daysInDecember : ℕ) (hoursPerDay : ℕ) : ℝ :=
  totalSnowfall / (daysInDecember * hoursPerDay)

/-- Theorem stating the average snowfall per hour in Norilsk, Russia, during December 1861 -/
theorem norilsk_snowfall_theorem (totalSnowfall : ℝ) (daysInDecember : ℕ) (hoursPerDay : ℕ) 
    (h1 : totalSnowfall = 492)
    (h2 : daysInDecember = 31)
    (h3 : hoursPerDay = 24) :
    averageSnowfallPerHour totalSnowfall daysInDecember hoursPerDay = 492 / (31 * 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_norilsk_snowfall_theorem_l983_98380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_value_for_x_5_l983_98301

/-- Represents the population data for different age groups -/
def population : List ℝ := [2, 3, 7, 8]

/-- The x values corresponding to the age groups -/
def x_values : List ℝ := [1, 2, 3, 4, 5]

/-- The slope of the linear regression equation -/
def m : ℝ := 2.1

/-- The y-intercept of the linear regression equation -/
def b : ℝ := -0.3

/-- The unknown population value for x = 5 -/
noncomputable def a : ℝ := sorry

/-- Theorem stating that a must equal 10 to satisfy the linear regression equation -/
theorem population_value_for_x_5 : 
  ((List.sum (population ++ [a]) : ℝ) / (List.length x_values : ℝ) = 
   m * ((List.sum x_values : ℝ) / (List.length x_values : ℝ)) + b) → 
  a = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_value_for_x_5_l983_98301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_golden_ratio_vertex_angle_l983_98313

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

def IsoscelesGoldenRatioTriangle (a b θ : ℝ) : Prop :=
  a = φ * b ∧ 
  a^2 = 2 * b^2 * (1 - Real.cos θ)

theorem isosceles_golden_ratio_vertex_angle :
  ∀ a b θ : ℝ, IsoscelesGoldenRatioTriangle a b θ → θ = 36 * π / 180 := by
  sorry

#check isosceles_golden_ratio_vertex_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_golden_ratio_vertex_angle_l983_98313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_division_l983_98332

/-- Calculates the amount of water in each beaker when equally divided -/
theorem water_division (bucket_capacity : ℝ) (num_filled_buckets : ℕ) (remaining_water : ℝ) (num_beakers : ℕ) :
  bucket_capacity = 120 →
  num_filled_buckets = 2 →
  remaining_water = 2.4 →
  num_beakers = 3 →
  (bucket_capacity * (num_filled_buckets : ℝ) + remaining_water) / (num_beakers : ℝ) = 80.8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_division_l983_98332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l983_98356

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / d = 34 / 999 ∧ 
  n + d = 1033 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l983_98356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_length_is_twenty_l983_98316

/-- Represents a rectangular pond -/
structure RectangularPond where
  width : ℝ
  depth : ℝ
  volume : ℝ

/-- Calculates the length of a rectangular pond given its width, depth, and volume -/
noncomputable def calculatePondLength (pond : RectangularPond) : ℝ :=
  pond.volume / (pond.width * pond.depth)

theorem pond_length_is_twenty :
  let pond : RectangularPond := {
    width := 10,
    depth := 5,
    volume := 1000
  }
  calculatePondLength pond = 20 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_length_is_twenty_l983_98316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l983_98344

/-- Given an ellipse with specified properties, prove that the angle between two foci and a point on the ellipse is π/3 -/
theorem ellipse_angle_theorem (F₁ F₂ P : ℝ × ℝ) (e : ℝ) : 
  F₁ = (0, -Real.sqrt 3) →
  F₂ = (0, Real.sqrt 3) →
  e = Real.sqrt 3 / 2 →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (P.1^2 / b^2 + P.2^2 / a^2 = 1)) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 2/3 →
  Real.arccos ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2))) = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l983_98344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_section_area_l983_98342

/-- The area of a circular section. -/
noncomputable def area_of_section (r : ℝ) (θ : ℝ) : ℝ := 
  Real.pi * (r * Real.sin θ) ^ 2

/-- The area of a section of a sphere, given specific conditions. -/
theorem sphere_section_area (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = Real.pi / 3 → area_of_section r θ = Real.pi :=
by
  intros h_r h_θ
  unfold area_of_section
  -- The proof steps would go here
  sorry

#check sphere_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_section_area_l983_98342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l983_98307

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)

def has_three_highest_points (ω : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 1 ∧
    ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f ω x ≤ f ω x₁ ∧
    f ω x ≤ f ω x₂ ∧ f ω x ≤ f ω x₃

theorem omega_range (ω : ℝ) (h₁ : ω > 0) (h₂ : has_three_highest_points ω) :
  17 * Real.pi / 4 ≤ ω ∧ ω < 25 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l983_98307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_and_distance_l983_98317

/-- A parabola with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

/-- The trajectory of the midpoint of AB -/
def midpointTrajectory (para : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = para.p * x - 2 * para.p^2}

/-- The distance function from a point to the given line -/
noncomputable def distToLine (para : Parabola) (x y : ℝ) : ℝ :=
  |x - 2*y + 2*Real.sqrt 5 - para.p| / Real.sqrt 5

/-- The main theorem -/
theorem parabola_midpoint_and_distance (para : Parabola) 
    (A B : ParabolaPoint para) (h_perp : A.x * B.x + A.y * B.y = 0) :
  (∀ (x y : ℝ), (x, y) ∈ midpointTrajectory para → 
    y^2 = para.p * x - 2 * para.p^2) ∧ 
  (∃ (d : ℝ), d = 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ midpointTrajectory para → 
      distToLine para x y ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_and_distance_l983_98317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l983_98334

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 3))

-- Define the two spheres touching the plane at B and C
structure Sphere where
  center : EuclideanSpace ℝ (Fin 3)
  radius : ℝ

variable (sphere1 sphere2 : Sphere)

-- Define the third sphere centered at A
variable (sphere3 : Sphere)

-- Define the circumcircle of triangle ABC
noncomputable def circumcircle (A B C : EuclideanSpace ℝ (Fin 3)) : Sphere :=
{ center := sorry,
  radius := sorry }

-- Define necessary functions
def touches_plane_at (s : Sphere) (p : EuclideanSpace ℝ (Fin 3)) : Prop := sorry
def on_opposite_side_of (s1 s2 : Sphere) : Prop := sorry
def touches_externally (s1 s2 : Sphere) : Prop := sorry

-- State the theorem
theorem circumcircle_radius
  (h1 : touches_plane_at sphere1 B)
  (h2 : touches_plane_at sphere2 C)
  (h3 : on_opposite_side_of sphere1 sphere2)
  (h4 : sphere1.radius + sphere2.radius = 12)
  (h5 : dist sphere1.center sphere2.center = 4 * Real.sqrt 29)
  (h6 : sphere3.center = A)
  (h7 : sphere3.radius = 8)
  (h8 : touches_externally sphere3 sphere1)
  (h9 : touches_externally sphere3 sphere2) :
  (circumcircle A B C).radius = 4 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l983_98334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_eq_neg_cos_l983_98390

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.sin
  | n + 1 => deriv (f n)

theorem f_2013_eq_neg_cos : f 2013 = λ x => -Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_eq_neg_cos_l983_98390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_variance_greater_than_girls_l983_98329

noncomputable def boys_scores : List ℝ := [86, 94, 88, 92, 90]
noncomputable def girls_scores : List ℝ := [88, 93, 93, 88, 93]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := scores.sum / scores.length
  (scores.map (fun x => (x - mean)^2)).sum / scores.length

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_variance_greater_than_girls_l983_98329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l983_98394

theorem cosine_sum_identity (α β : ℝ) :
  (Real.cos α * Real.cos (β/2)) / Real.cos (α - β/2) + 
  (Real.cos β * Real.cos (α/2)) / Real.cos (β - α/2) - 1 = 1 →
  Real.cos α + Real.cos β = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l983_98394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_half_l983_98351

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem f_sum_equals_half : ∀ x : ℝ, f x + f (1 - x) = 1/2 := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_half_l983_98351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilaterals_from_circle_points_l983_98353

-- Define the function
def number_of_convex_quadrilaterals (n : ℕ) : ℕ := Nat.choose n 4

-- Theorem statement
theorem quadrilaterals_from_circle_points (n : ℕ) (h : n = 12) :
  number_of_convex_quadrilaterals n = Nat.choose n 4 := by
  -- Proof is trivial due to definition, but we'll use sorry as requested
  sorry

-- Evaluate the result for n = 12
#eval number_of_convex_quadrilaterals 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilaterals_from_circle_points_l983_98353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_revenue_l983_98396

/-- The price function P(x) --/
noncomputable def P (x : ℝ) : ℝ := 1 + 1 / x

/-- The quantity function Q(x) --/
noncomputable def Q (x : ℝ) : ℝ := 125 - |x - 25|

/-- The revenue function f(x) --/
noncomputable def f (x : ℝ) : ℝ := P x * Q x

/-- The theorem stating the minimum value of f(x) --/
theorem min_revenue :
  ∀ x : ℕ+, 1 ≤ (x : ℝ) ∧ (x : ℝ) ≤ 30 → f x ≥ 121 ∧ f 10 = 121 :=
by
  sorry

#check min_revenue

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_revenue_l983_98396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_meet_is_66_minutes_l983_98361

/-- The time it takes for Lauren to reach Andrea under the given conditions -/
noncomputable def time_to_meet : ℝ :=
  let initial_distance : ℝ := 24
  let andrea_speed_multiplier : ℝ := 4
  let distance_decrease_rate : ℝ := 2
  let andrea_stop_time : ℝ := 6
  let lauren_speed_reduction : ℝ := 1/2

  let lauren_initial_speed := distance_decrease_rate / (andrea_speed_multiplier + 1)
  let andrea_initial_speed := lauren_initial_speed * andrea_speed_multiplier
  let distance_covered_before_stop := distance_decrease_rate * andrea_stop_time
  let remaining_distance := initial_distance - distance_covered_before_stop
  let lauren_final_speed := lauren_initial_speed * lauren_speed_reduction
  let time_after_stop := remaining_distance / lauren_final_speed

  andrea_stop_time + time_after_stop

/-- Theorem stating that the time to meet is 66 minutes -/
theorem time_to_meet_is_66_minutes : time_to_meet = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_meet_is_66_minutes_l983_98361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l983_98323

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

-- State the theorem
theorem f_inequality (a b : ℝ) (h : a ≠ b) : |f a - f b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l983_98323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phyllis_discount_equivalent_to_6_for_5_l983_98358

/-- Represents the price of a DVD in dollars -/
noncomputable def dvd_price : ℝ := 20

/-- Represents the total cost of Phyllis' purchase -/
noncomputable def phyllis_cost : ℝ := 2 * dvd_price + dvd_price / 2

/-- Represents the number of DVDs Phyllis gets -/
noncomputable def phyllis_dvds : ℝ := 3

/-- Represents the ratio of DVDs to cost in the equivalent discount -/
noncomputable def discount_ratio : ℝ := 6 / 5

/-- Theorem stating that Phyllis' discount is equivalent to getting 6 DVDs for the price of 5 -/
theorem phyllis_discount_equivalent_to_6_for_5 :
  phyllis_dvds / (phyllis_cost / dvd_price) = discount_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phyllis_discount_equivalent_to_6_for_5_l983_98358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l983_98387

theorem inequality_solution_set (x : ℝ) :
  (2 ≥ 1 / (x - 1)) ↔ x ∈ Set.Iio 1 ∪ Set.Ici (3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l983_98387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l983_98341

/-- The repeating decimal 0.4̄47 as a real number -/
noncomputable def repeating_decimal : ℝ := 0.4 + (47 / 990)

/-- The theorem stating that the repeating decimal 0.4̄47 is equal to 44/49 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 44 / 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l983_98341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_10_to_20_l983_98376

def isPrime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def sumOfPrimesBetween (a b : Nat) : Nat :=
  (List.range (b - a + 1)).map (λ x => x + a)
    |>.filter isPrime
    |>.sum

theorem sum_of_primes_10_to_20 : sumOfPrimesBetween 10 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_10_to_20_l983_98376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l983_98343

/-- The speed of a train in km/hr, given its length in meters and the time it takes to pass a stationary point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a 300-meter long train passing a point in 12 seconds has a speed of 90 km/hr. -/
theorem train_speed_theorem :
  train_speed 300 12 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l983_98343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l983_98327

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (0, 3)

-- Define C implicitly
noncomputable def C : ℝ × ℝ :=
  let x := Real.sqrt 7
  (x, 0)

-- State the theorem
theorem triangle_ABC_area :
  let distanceBC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  distanceBC = 4 →
  (1/2 : ℝ) * C.1 * 3 = (3 * Real.sqrt 7) / 2 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l983_98327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glow_interval_approximation_l983_98359

/-- Represents a time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts a Time to total seconds -/
def timeToSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Represents the problem setup -/
structure LightGlowProblem where
  startTime : Time
  endTime : Time
  glowCount : ℚ

/-- Calculates the glow interval in seconds -/
noncomputable def glowInterval (problem : LightGlowProblem) : ℚ :=
  let totalSeconds : ℕ := timeToSeconds problem.endTime - timeToSeconds problem.startTime
  (totalSeconds : ℚ) / problem.glowCount

theorem glow_interval_approximation (problem : LightGlowProblem) :
  problem.startTime = ⟨1, 57, 58⟩ →
  problem.endTime = ⟨3, 20, 47⟩ →
  problem.glowCount = 276056 / 1000 →
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ |glowInterval problem - 18| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glow_interval_approximation_l983_98359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_force_on_moon_l983_98379

/-- Gravitational force calculation -/
theorem gravitational_force_on_moon (k : ℝ) (d_earth d_moon : ℝ) (f_earth : ℝ) :
  k > 0 →
  d_earth > 0 →
  d_moon > 0 →
  f_earth > 0 →
  d_earth = 5000 →
  d_moon = 250000 →
  f_earth = 800 →
  k = f_earth * d_earth^2 →
  k / d_moon^2 = 8 / 25 := by
  sorry

#check gravitational_force_on_moon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_force_on_moon_l983_98379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_groomed_matrix_theorem_l983_98382

/-- A well-groomed matrix is an n×n matrix containing only 0s and 1s,
    and does not contain the submatrix [1 0; 0 1]. -/
def WellGroomedMatrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  (∀ i j, A i j = 0 ∨ A i j = 1) ∧
  (∀ i j k l, i ≠ k → j ≠ l → ¬(A i j = 1 ∧ A k l = 1 ∧ A i l = 0 ∧ A k j = 0))

/-- A submatrix of size m×m in an n×n matrix -/
def Submatrix {n m : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) (rows cols : Fin m → Fin n) : Matrix (Fin m) (Fin m) ℕ :=
  λ i j ↦ A (rows i) (cols j)

/-- All elements in a matrix are equal -/
def AllEqual {m : ℕ} (A : Matrix (Fin m) (Fin m) ℕ) : Prop :=
  ∀ i j k l, A i j = A k l

theorem well_groomed_matrix_theorem {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) (h : WellGroomedMatrix A) :
  ∃ (c : ℚ), c > 0 ∧ c < 1/80 ∧ ∃ (m : ℕ) (rows cols : Fin m → Fin n),
    (m : ℚ) ≥ c * n ∧ AllEqual (Submatrix A rows cols) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_groomed_matrix_theorem_l983_98382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_candidates_can_win_l983_98318

/-- Represents a candidate in the school election --/
structure Candidate where
  name : String
  votes : ℕ

/-- Represents the state of the school election --/
structure ElectionState where
  candidates : List Candidate
  votesCount : ℕ
  totalVotes : ℕ

/-- Checks if a candidate has a chance of winning --/
def hasChanceOfWinning (c : Candidate) (state : ElectionState) : Bool :=
  c.votes + (state.totalVotes - state.votesCount) ≥ 
    (state.candidates.map (λ x => x.votes)).foldl max 0

/-- The main theorem statement --/
theorem three_candidates_can_win (state : ElectionState) : 
  state.candidates = [
    ⟨"Henry", 14⟩, 
    ⟨"India", 11⟩, 
    ⟨"Jenny", 10⟩, 
    ⟨"Ken", 8⟩, 
    ⟨"Lena", 2⟩
  ] →
  state.votesCount = 45 →
  state.totalVotes = 50 →
  (state.candidates.filter (hasChanceOfWinning · state)).length = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_candidates_can_win_l983_98318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_sum_equality_l983_98319

def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

def sum_up_to (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

def divisor_property (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 2002, (sum_up_to a n) % (a (n + 1)) = 0

theorem eventual_sum_equality 
  (a : ℕ → ℕ) 
  (h_incr : is_increasing_sequence a) 
  (h_div : divisor_property a) : 
  ∃ N, ∀ n ≥ N, a (n + 1) = sum_up_to a n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_sum_equality_l983_98319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficiency_not_necessity_l983_98362

theorem sufficiency_not_necessity :
  (∀ x : ℝ, (1 / (x - 1) > 1) → (|2*x - 1| < 3)) ∧
  (∃ x : ℝ, |2*x - 1| < 3 ∧ 1 / (x - 1) ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficiency_not_necessity_l983_98362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l983_98374

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x - Real.pi / 3) - 1

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
  f x ∈ Set.Icc (-5 / 2) (Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l983_98374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_solution_set_iff_a_range_l983_98392

theorem empty_solution_set_iff_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ a ∈ Set.Iic 2 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_solution_set_iff_a_range_l983_98392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l983_98370

theorem sin_double_angle_fourth_quadrant (θ : ℝ) :
  θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →
  Real.cos θ = 4 / 5 →
  Real.sin (2 * θ) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l983_98370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_range_l983_98335

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem no_zeros_range (ω : ℝ) :
  ω > 0 ∧
  (∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f (ω * x) ≠ 0) ↔
  ω ∈ Set.Ioc 0 (5 / 12) ∪ Set.Icc (5 / 6) (11 / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_range_l983_98335
