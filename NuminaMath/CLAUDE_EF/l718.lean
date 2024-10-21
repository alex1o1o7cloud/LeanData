import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_progress_regress_theorem_l718_71839

/-- The ratio of progress to regress after 300 days -/
noncomputable def progress_regress_ratio : ℝ := (1 + 0.01)^300 / (1 - 0.01)^300

/-- The approximate value of the ratio -/
def approximate_ratio : ℝ := 407

/-- Theorem stating that the progress_regress_ratio is approximately equal to 407 -/
theorem progress_regress_theorem :
  ∃ ε > 0, |progress_regress_ratio - approximate_ratio| < ε :=
by
  sorry

#eval approximate_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_progress_regress_theorem_l718_71839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_cos_over_pi_l718_71882

theorem integral_sin_plus_cos_over_pi : 
  ∫ x in (-Real.pi/2)..(Real.pi/2), (Real.sin x + Real.cos x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_cos_over_pi_l718_71882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weekly_pay_is_528_l718_71845

/-- The weekly pay of employee Y in rupees -/
noncomputable def y_pay : ℚ := 240

/-- The weekly pay of employee X as a percentage of Y's pay -/
noncomputable def x_pay_percentage : ℚ := 120

/-- Calculates the weekly pay of employee X -/
noncomputable def x_pay : ℚ := (x_pay_percentage / 100) * y_pay

/-- Calculates the total weekly pay for both employees -/
noncomputable def total_pay : ℚ := x_pay + y_pay

/-- Theorem stating that the total weekly pay for both employees is 528 rupees -/
theorem total_weekly_pay_is_528 : total_pay = 528 := by
  -- Unfold the definitions
  unfold total_pay x_pay y_pay x_pay_percentage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weekly_pay_is_528_l718_71845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l718_71879

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 395/381 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l718_71879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l718_71808

noncomputable def sample_variance (s : Finset ℝ) (f : ℝ → ℝ) : ℝ := 
  (s.sum (λ x => (f x - s.sum f / s.card) ^ 2)) / s.card

theorem variance_transformation (k : Finset ℝ) 
  (h : sample_variance k id = 6) 
  (h_card : k.card = 10) :
  sample_variance k (λ x => 3 * (x - 1)) = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l718_71808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_return_simultaneously_l718_71869

/-- Two people walking towards each other with their dogs -/
structure WalkingScenario where
  L : ℝ  -- Distance between the two people
  u : ℝ  -- Speed of the slower person
  v : ℝ  -- Speed of the faster person
  V : ℝ  -- Speed of both dogs
  hu : 0 < u  -- Speed of slower person is positive
  hv : u < v  -- Speed of faster person is greater than slower person
  hV : 0 < V  -- Speed of dogs is positive

/-- Time taken for a dog to reach the other owner -/
noncomputable def time_to_reach (s : WalkingScenario) : ℝ :=
  s.L / (s.u + s.v + s.V)

/-- Time taken for a dog to complete its journey (to other owner and back) -/
noncomputable def total_time (s : WalkingScenario) : ℝ :=
  2 * time_to_reach s

/-- Theorem stating that both dogs return at the same time -/
theorem dogs_return_simultaneously (s : WalkingScenario) :
  total_time s = total_time s := by
  -- The proof is trivial as we're comparing the same expression
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_return_simultaneously_l718_71869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l718_71836

/-- The range of values for b such that the line y = x + b intersects 
    the semicircle x = √(1 - y²) at exactly one point -/
theorem intersection_range : 
  {b : ℝ | ∃! p : ℝ × ℝ, p.1 = Real.sqrt (1 - p.2^2) ∧ p.2 = p.1 + b ∧ p.1 ≥ 0} = 
  {-Real.sqrt 2} ∪ Set.Ioc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l718_71836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_n_gon_l718_71876

/-- A point on the circumcircle of an n-gon -/
structure CircumcirclePoint (n : ℕ) where
  point : ℝ × ℝ
  on_circumcircle : Bool

/-- An inscribed n-gon -/
structure InscribedNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_inscribed : Bool

/-- Projection of a point onto a line -/
noncomputable def project_point_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Simson line of an (n-1)-gon -/
noncomputable def simson_line (n : ℕ) (ngon : InscribedNGon n) (i : Fin n) : Set (ℝ × ℝ) := sorry

/-- Theorem: Simson line of an inscribed n-gon -/
theorem simson_line_n_gon (n : ℕ) (ngon : InscribedNGon n) (p : CircumcirclePoint n) :
  ∃ (l : Set (ℝ × ℝ)), ∀ (i : Fin n), 
    project_point_to_line p.point (simson_line n ngon i) ∈ l := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_n_gon_l718_71876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_proof_l718_71896

/-- Calculates the amount of water needed to dilute an alcohol mixture -/
noncomputable def water_needed (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  (initial_volume * initial_concentration / final_concentration) - initial_volume

/-- Proves that 12 ounces of water are needed to dilute 12 ounces of 50% alcohol to 25% alcohol -/
theorem dilution_proof :
  water_needed 12 0.5 0.25 = 12 := by
  -- Unfold the definition of water_needed
  unfold water_needed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- Use #eval with a computable function for demonstration
def water_needed_rat (initial_volume : ℚ) (initial_concentration : ℚ) (final_concentration : ℚ) : ℚ :=
  (initial_volume * initial_concentration / final_concentration) - initial_volume

#eval water_needed_rat 12 (1/2) (1/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_proof_l718_71896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_meets_alice_in_30_minutes_l718_71852

/-- The time in minutes it takes for two people to meet when walking towards each other -/
noncomputable def meeting_time (alice_speed : ℝ) (claire_speed : ℝ) (initial_distance : ℝ) : ℝ :=
  (initial_distance / (alice_speed + claire_speed)) * 60

/-- Theorem: Given the conditions, Claire meets Alice in 30 minutes -/
theorem claire_meets_alice_in_30_minutes :
  let alice_speed : ℝ := 4
  let claire_speed : ℝ := 6
  let initial_distance : ℝ := 5
  meeting_time alice_speed claire_speed initial_distance = 30 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_meets_alice_in_30_minutes_l718_71852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diane_harvest_increase_l718_71854

/-- Calculates the new harvest amount given the previous year's harvest and the percentage increase --/
noncomputable def new_harvest (previous_harvest : ℝ) (increase_percentage : ℝ) : ℝ :=
  previous_harvest * (1 + increase_percentage / 100)

/-- Proves that Diane's new harvest is 3346.65 pounds given the previous year's harvest and increase percentage --/
theorem diane_harvest_increase : new_harvest 2479 35 = 3346.65 := by
  -- Unfold the definition of new_harvest
  unfold new_harvest
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diane_harvest_increase_l718_71854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l718_71881

noncomputable def f (x : ℝ) := x^3 - (1/2) * x^2 - 2*x + 1

theorem f_properties :
  (∀ x, deriv f x = 3*x^2 - x - 2) ∧
  (deriv f 1 = 0) ∧
  (deriv f (-2/3) = 0) ∧
  (f (-1) = 3/2) ∧
  (∀ x, x < -2/3 ∨ x > 1 → deriv f x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → deriv f x < 0) ∧
  (f (-2/3) = 49/27) ∧
  (f 1 = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l718_71881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_original_number_l718_71818

noncomputable def ana_operations (x : ℝ) : ℝ :=
  (((3 * (x + 3) + 3) - 3) / 3)

theorem ana_original_number : ∃ x : ℝ, ana_operations x = 10 ∧ x = 7 := by
  use 7
  constructor
  · simp [ana_operations]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_original_number_l718_71818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_nonagon_area_l718_71829

/-- A nonagon with three pairs of equal and parallel sides -/
structure SpecialNonagon where
  vertices : Fin 9 → ℝ × ℝ
  parallel_sides : ∃ (i j k : Fin 9), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    (∃ v : ℝ × ℝ, (vertices i - vertices (i+1)) = v ∧ 
                  (vertices j - vertices (j+1)) = v ∧
                  (vertices k - vertices (k+1)) = v)

/-- The midpoints of the remaining three sides -/
def midpoints (n : SpecialNonagon) : Fin 3 → ℝ × ℝ :=
  sorry

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- The area of the entire nonagon -/
noncomputable def nonagonArea (n : SpecialNonagon) : ℝ :=
  sorry

theorem special_nonagon_area (n : SpecialNonagon) :
  triangleArea (midpoints n 0) (midpoints n 1) (midpoints n 2) = 12 →
  nonagonArea n = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_nonagon_area_l718_71829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_problem_l718_71873

theorem set_problem (U A B : Finset ℕ) : 
  (U.card = 193) →
  ((U \ (A ∪ B)).card = 59) →
  ((A ∩ B).card = 25) →
  (A.card = 110) →
  (B.card = 49) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_problem_l718_71873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l718_71856

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the x-intercept of a line -/
noncomputable def Line.xIntercept (l : Line) : ℝ :=
  -l.c / l.a

/-- Calculate the y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ :=
  -l.c / l.b

/-- Calculate the area of a triangle formed by a line and the positive x and y axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  (1/2) * l.xIntercept * l.yIntercept

theorem line_equation_proof (l : Line) (A : Point) :
  l.a = 1 ∧ l.b = 2 ∧ l.c = -2 →
  A.x = -2 ∧ A.y = 2 →
  l.contains A ∧
  l.xIntercept > 0 ∧
  l.yIntercept > 0 ∧
  triangleArea l = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l718_71856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_characterization_l718_71898

noncomputable def g (t : ℝ) : ℝ := (2 * t^2 - 5/2 * t + 1) / (t^2 + 3 * t + 2)

theorem g_range_characterization (y : ℝ) :
  (∃ t, g t = y) ↔ y^2 + 35 * y + 9/4 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_characterization_l718_71898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_is_6_l718_71884

/-- A function that checks if a two-digit number is divisible by either 12 or 15 -/
def isDivisibleBy12Or15 (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 12 = 0 ∨ n % 15 = 0)

/-- A function that checks if a list of digits satisfies the divisibility condition -/
def satisfiesDivisibilityCondition (digits : List Nat) : Prop :=
  ∀ i, i + 1 < digits.length → isDivisibleBy12Or15 (digits[i]! * 10 + digits[i+1]!)

/-- The main theorem -/
theorem largest_last_digit_is_6 :
  ∃ (digits : List Nat),
    digits.length = 2015 ∧
    digits.head? = some 2 ∧
    satisfiesDivisibilityCondition digits ∧
    digits.getLast? = some 6 ∧
    (∀ (other_digits : List Nat),
      other_digits.length = 2015 →
      other_digits.head? = some 2 →
      satisfiesDivisibilityCondition other_digits →
      (other_digits.getLast? = some n → n ≤ 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_is_6_l718_71884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sixth_power_sum_l718_71810

open Real BigOperators

theorem cos_sixth_power_sum :
  (∑ k in Finset.range 91, (cos (k * π / 180))^6) = 229/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sixth_power_sum_l718_71810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_officer_assignment_count_l718_71841

theorem officer_assignment_count : 
  let n : ℕ := 4  -- number of people and positions
  ∀ (people : Finset (Fin n)) (positions : Finset (Fin n)),
    Finset.card people = n →
    Finset.card positions = n →
    Fintype.card (Equiv.Perm (Fin n)) = 24 :=
by
  intro n people positions h1 h2
  rw [Fintype.card_perm]
  simp
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_officer_assignment_count_l718_71841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_difference_l718_71840

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 3
def g (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 3

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧
  (∀ x ∈ intersection_points, a ≤ x ∧ x ≤ c) ∧ c - a = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_difference_l718_71840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_greater_than_half_l718_71820

/-- The function f(x) = 2^x + a^2*x - 2a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + a^2*x - 2*a

/-- Theorem: If f(x) has a root in (0,1), then a > 1/2 -/
theorem root_implies_a_greater_than_half (a : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ f a x = 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_greater_than_half_l718_71820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_inscribed_square_properties_l718_71890

/-- Regular hexagon with side length 1 -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 1

/-- Square inscribed in a regular hexagon -/
structure InscribedSquare (h : RegularHexagon) where
  vertices : Fin 4 → ℝ × ℝ
  is_square : sorry
  on_perimeter : sorry

/-- Line segment with endpoints on opposite edges of the hexagon -/
structure OppositeEdgeSegment (h : RegularHexagon) where
  endpoints : Fin 2 → ℝ × ℝ
  on_opposite_edges : sorry

/-- Check if a segment passes through the center of the hexagon -/
def passes_through_center (h : RegularHexagon) (s : OppositeEdgeSegment h) : Prop := sorry

/-- Check if a segment divides opposite edges equally -/
def divides_edges_equally (h : RegularHexagon) (s : OppositeEdgeSegment h) : Prop := sorry

/-- Get the center of an inscribed square -/
def square_center (h : RegularHexagon) (sq : InscribedSquare h) : ℝ × ℝ := sorry

/-- Get the center of a hexagon -/
def hexagon_center (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Check if an inscribed square has sides parallel to the hexagon -/
def has_parallel_sides (h : RegularHexagon) (sq : InscribedSquare h) : Prop := sorry

/-- Get the side length of an inscribed square -/
def square_side_length (h : RegularHexagon) (sq : InscribedSquare h) : ℝ := sorry

/-- Main theorem -/
theorem hexagon_inscribed_square_properties (h : RegularHexagon) :
  -- 1. Line segment through center iff it divides edges in same ratio
  (∀ (s : OppositeEdgeSegment h), passes_through_center h s ↔ divides_edges_equally h s)
  ∧
  -- 2. Center of inscribed square coincides with hexagon center
  (∀ (sq : InscribedSquare h), square_center h sq = hexagon_center h)
  ∧
  -- 3. Side length of inscribed square with parallel sides
  (∃ (sq : InscribedSquare h), has_parallel_sides h sq ∧ 
    square_side_length h sq = (3 * (Real.sqrt 3 - 1)) / 2)
  ∧
  -- 4. Uniqueness of inscribed square up to rotation
  (∀ (sq1 sq2 : InscribedSquare h), ∃ (rotation : ℝ × ℝ → ℝ × ℝ), 
    (∀ (x : ℝ × ℝ), ‖rotation x - rotation (0, 0)‖ = ‖x - (0, 0)‖) ∧
    (rotation (square_center h sq1) = square_center h sq2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_inscribed_square_properties_l718_71890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_divided_in_half_l718_71874

-- Define the vertices of the shape
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (1, 2)
def F : ℝ × ℝ := (3, 2)
def G : ℝ × ℝ := (2, 4)

-- Define the shape as a set of points
def Shape : Set (ℝ × ℝ) := {p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2) ∨
  (1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 2 ≤ p.2 ∧ p.2 ≤ 4 ∧ p.2 ≤ -2 * p.1 + 8)}

-- Define the line y = 2x
def DividingLine (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem area_divided_in_half :
  ∃ (S₁ S₂ : Set (ℝ × ℝ)),
    S₁ ∪ S₂ = Shape ∧
    S₁ ∩ S₂ ⊆ {p : ℝ × ℝ | p.2 = DividingLine p.1} ∧
    (∀ m : MeasureTheory.Measure (ℝ × ℝ), m S₁ = m S₂) :=
sorry

#check area_divided_in_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_divided_in_half_l718_71874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_complex_numbers_result_as_rect_form_l718_71803

theorem product_of_complex_numbers :
  let z₁ : ℂ := Complex.exp (23 * π / 180 * I)
  let z₂ : ℂ := Complex.exp (37 * π / 180 * I)
  z₁ * z₂ = Complex.exp (60 * π / 180 * I) := by
  sorry

theorem result_as_rect_form :
  Complex.exp (60 * π / 180 * I) = Complex.ofReal (1 / 2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_complex_numbers_result_as_rect_form_l718_71803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l718_71804

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem f_properties :
  (∃ (M : ℝ), M = (1 + Real.sqrt 3) / 2 ∧ ∀ (x : ℝ), f x ≤ M) ∧
  (∃ (T : ℝ), T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (A B C : ℝ), 
    Real.cos B = 1/3 → 
    f (C/3) = -1/4 → 
    Real.sin A = 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l718_71804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l718_71801

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on the first line
  b : ℝ × ℝ  -- Point on the second line
  d : ℝ × ℝ  -- Direction vector

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  let v := (lines.b.1 - lines.a.1, lines.b.2 - lines.a.2)
  let proj_v_d := 
    let numerator := v.1 * lines.d.1 + v.2 * lines.d.2
    let denominator := lines.d.1 * lines.d.1 + lines.d.2 * lines.d.2
    (numerator / denominator * lines.d.1, numerator / denominator * lines.d.2)
  let orthogonal_component := (v.1 - proj_v_d.1, v.2 - proj_v_d.2)
  Real.sqrt (orthogonal_component.1^2 + orthogonal_component.2^2)

/-- Theorem: The distance between the given parallel lines is √10/5 -/
theorem distance_between_specific_lines : 
  distance_between_parallel_lines ⟨(3, -2), (4, -1), (2, 1)⟩ = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l718_71801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_point_seven_five_degrees_l718_71885

open Real

-- Define the angle in radians
noncomputable def angle : ℝ := 3.75 * (π / 180)

-- Define the function for the specific form of tangent
noncomputable def tan_form (x y z w : ℕ) : ℝ := sqrt (x : ℝ) - sqrt (y : ℝ) + sqrt (z : ℝ) - (w : ℝ)

-- Theorem statement
theorem tan_three_point_seven_five_degrees :
  ∃ (x y z w : ℕ), x ≥ y ∧ y ≥ z ∧ z ≥ w ∧
  tan_form x y z w = tan angle ∧
  x + y + z + w = 5 ∧
  tan_form x y z w = sqrt 3 - sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_point_seven_five_degrees_l718_71885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_correct_num_rows_l718_71867

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The increase in the number of desks for each subsequent row -/
def desk_increase : ℕ := 2

/-- The total number of desks in the classroom -/
def total_desks : ℕ := 136

/-- The number of rows in the classroom -/
def num_rows : ℕ := 8

/-- 
  Theorem stating that the number of rows is correct given the conditions
  of the problem.
-/
theorem correct_num_rows : 
  (first_row_desks + (num_rows - 1) * desk_increase) * num_rows / 2 = total_desks := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_correct_num_rows_l718_71867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_magnitude_l718_71823

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Given parallel vectors a and b, prove their sum has magnitude √5 -/
theorem parallel_vectors_sum_magnitude :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 1)
  parallel a b →
  magnitude (a.1 + b.1, a.2 + b.2) = Real.sqrt 5 :=
by
  sorry

#check parallel_vectors_sum_magnitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_magnitude_l718_71823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_of_specific_circles_l718_71814

/-- Circle represented by the equation x^2 + y^2 + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of the line formed by the intersection of two circles -/
noncomputable def intersection_slope (c1 c2 : Circle) : ℝ :=
  (c2.b - c1.b) / (c1.a - c2.a)

theorem intersection_slope_of_specific_circles :
  let c1 : Circle := { a := -6, b := 8, c := -20 }
  let c2 : Circle := { a := -10, b := -4, c := 40 }
  intersection_slope c1 c2 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_of_specific_circles_l718_71814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l718_71889

noncomputable def f (x : ℝ) : ℝ := |1/2 * x + 1| + |x|

theorem f_minimum_value :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f x ≥ m := by
  use 1
  constructor
  · rfl
  · intro x
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l718_71889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l718_71830

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The starting point of the journey -/
def start : Point := ⟨0, 0⟩

/-- The ending point of the journey -/
def finish : Point := ⟨24 - 16, -(10 - 8)⟩

/-- Theorem stating that the distance between start and finish points is 2√17 -/
theorem journey_distance : distance start finish = 2 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l718_71830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_O2_equation_l718_71807

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 6

-- Define the center of circle O2
def center_O2 : ℝ × ℝ := (2, 1)

-- Define the intersection points A and B
def intersect_points (A B : ℝ × ℝ) : Prop :=
  ∃ (x_A y_A x_B y_B : ℝ),
    A = (x_A, y_A) ∧ B = (x_B, y_B) ∧
    circle_O1 x_A y_A ∧ circle_O1 x_B y_B

-- Define the distance between A and B
noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem circle_O2_equation :
  ∀ (A B : ℝ × ℝ),
    intersect_points A B →
    distance_AB A B = 4 →
    (∀ (x y : ℝ),
      ((x - 2)^2 + (y - 1)^2 = 6) ∨ ((x - 2)^2 + (y - 1)^2 = 22)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_O2_equation_l718_71807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l718_71877

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x / Real.log 3 + m

noncomputable def g (x : ℝ) (m : ℝ) := (f x m)^2 - f (x^2) m

theorem range_of_g :
  ∃ m : ℝ, 
  (∀ x : ℝ, 1 ≤ x → x ≤ 9 → f x m = Real.log x / Real.log 3 + m) ∧
  f 1 m = 2 ∧
  Set.range (g · m) = Set.Icc 2 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l718_71877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_miles_l718_71811

/-- Represents Maria's car and trip details --/
structure CarTrip where
  /-- Miles per gallon the car can travel --/
  mpg : ℚ
  /-- Capacity of the gas tank in gallons --/
  tankCapacity : ℚ
  /-- Miles driven before refueling --/
  initialMiles : ℚ
  /-- Gallons of gas bought during refueling --/
  refuelGallons : ℚ
  /-- Fraction of tank full at the end of the trip --/
  endTankFraction : ℚ

/-- Calculates the total miles driven given the car trip details --/
def totalMilesDriven (trip : CarTrip) : ℚ :=
  trip.initialMiles + (trip.refuelGallons + trip.tankCapacity - trip.initialMiles / trip.mpg - trip.endTankFraction * trip.tankCapacity) * trip.mpg

/-- Theorem stating that Maria drove exactly 578 miles --/
theorem maria_trip_miles : 
  let trip : CarTrip := {
    mpg := 28
    tankCapacity := 16
    initialMiles := 420
    refuelGallons := 10
    endTankFraction := 1/3
  }
  totalMilesDriven trip = 578 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_miles_l718_71811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l718_71835

theorem power_equation_solution (x y : ℤ) 
  (h1 : (3 : ℝ)^x * (4 : ℝ)^y = 19683)
  (h2 : x - y = 9) : 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l718_71835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_l718_71858

noncomputable def evenPowerSum (a : ℝ) (k : ℕ) : ℝ := (1 - a^(2*k + 2)) / (1 - a^2)

noncomputable def oddPowerSum (a : ℝ) (k : ℕ) : ℝ := a * (1 - a^(2*k)) / (1 - a^2)

theorem induction_step (a : ℝ) (k : ℕ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : k > 0) 
  (hk : evenPowerSum a k / oddPowerSum a k > (k + 1) / k) :
  evenPowerSum a (k + 1) / oddPowerSum a (k + 1) > ((k + 1) + 1) / (k + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_l718_71858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l718_71817

noncomputable def f : ℝ → ℝ := fun y =>
  if y < 2 then 2 / (2 - y) else 0

theorem function_characterization :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 < x → x < 2 → f x ≠ 0) ∧
  (∀ y : ℝ, y ≥ 0 → y < 2 → f y = 2 / (2 - y)) ∧
  (∀ y : ℝ, y ≥ 2 → f y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l718_71817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_combinations_l718_71800

theorem ball_drawing_combinations : 
  (let n : ℕ := 15  -- Total number of balls
   let k : ℕ := 4   -- Number of balls drawn
   n * (n - 1) * (n - 2) * (n - 3)) = 32760 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_combinations_l718_71800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_division_l718_71837

theorem wine_division (m n : ℕ+) :
  (∃ (process : List (ℕ × ℕ)), 
    process.head? = some (0, 0) ∧ 
    process.getLast? = some ((m.val + n.val) / 2, (m.val + n.val) / 2) ∧
    ∀ (pair : ℕ × ℕ), pair ∈ process → pair.1 ≤ m.val ∧ pair.2 ≤ n.val) ↔
  (Even (m.val + n.val) ∧ (m.val + n.val) / 2 % Nat.gcd m.val n.val = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_division_l718_71837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l718_71894

-- Define vectors m and n
noncomputable def m (x : Real) : Real × Real := (Real.sin x, 3/4)
noncomputable def n (x : Real) : Real × Real := (Real.cos x, -1)

-- Define the function f
noncomputable def f (x : Real) : Real := 2 * ((m x).1 + (n x).1) * (n x).1 + 2 * ((m x).2 + (n x).2) * (n x).2

-- State the theorem
theorem vector_problem (A : Real) (h : Real.sin A + Real.cos A = Real.sqrt 2) :
  (∃ x : Real, (m x).1 * (n x).2 = (m x).2 * (n x).1 → Real.sin x ^ 2 + Real.sin (2 * x) = -3/5) ∧
  f A = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l718_71894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_douglas_vote_percent_Y_is_50_percent_l718_71880

/-- Represents the percentage of votes won by Douglas in county Y -/
noncomputable def douglas_vote_percent_Y (x y : ℝ) : ℝ :=
  (0.66 * (x + y) - 0.74 * x) / y

/-- Theorem stating the percentage of votes won by Douglas in county Y -/
theorem douglas_vote_percent_Y_is_50_percent
  (x y : ℝ) -- x represents votes in county X, y represents votes in county Y
  (h1 : x = 2 * y) -- ratio of voters in county X to county Y is 2:1
  (h2 : x > 0 ∧ y > 0) -- ensure positive number of voters
  : douglas_vote_percent_Y x y = 0.5 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_douglas_vote_percent_Y_is_50_percent_l718_71880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_color_ratio_l718_71822

theorem car_color_ratio (total purple red_excess : ℕ) 
  (h_total : total = 312)
  (h_purple : purple = 47)
  (h_red_excess : red_excess = 6) : 
  (total - (purple + (purple + red_excess)) : ℚ) / (purple + red_excess) = 212 / 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_color_ratio_l718_71822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l718_71844

/-- Given an ellipse with equation x^2 + my^2 = 1 and eccentricity √3/2, 
    the value of m is either 4 or 1/4 -/
theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2 + m*y^2 = 1 → 
    (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
      (x^2/a^2 + y^2/b^2 = 1) ∧ 
      (c^2 = a^2 - b^2) ∧ 
      (c/a = Real.sqrt 3/2))) → 
  (m = 4 ∨ m = 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l718_71844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l718_71859

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angle_sum : A + B + C = π
  side_angle_relation : (b^2 - a^2 - c^2) * Real.sin A * Real.cos A = a * c * Real.cos (A + C)
  side_a : a = Real.sqrt 2

noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

theorem triangle_properties (t : Triangle) :
  t.A = π/4 ∧ 
  ∃ (max_area : ℝ), max_area = (Real.sqrt 2 + 1) / 2 ∧ 
    ∀ (s : Triangle), s.a = t.a → triangle_area s ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l718_71859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_children_with_blue_flags_l718_71886

theorem percentage_of_children_with_blue_flags
  (total_flags : ℕ)
  (h_even : Even total_flags)
  (percentage_with_red_flags : ℚ)
  (percentage_with_both_colors : ℚ)
  (percentage_with_blue_flags : ℚ)
  (h_red_percentage : percentage_with_red_flags = 45)
  (h_both_percentage : percentage_with_both_colors = 5) :
  percentage_with_blue_flags = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_children_with_blue_flags_l718_71886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_l718_71893

-- Define the function f(x) = 2x / (x^2 - 1)
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x^2 - 1)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Theorem 1: f(x) is an odd function
theorem f_is_odd : ∀ x, domain x → f (-x) = -f x := by
  sorry

-- Theorem 2: f(x) is monotonically decreasing on its domain
theorem f_is_decreasing : ∀ x₁ x₂, domain x₁ → domain x₂ → x₁ < x₂ → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_l718_71893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_contest_score_difference_l718_71826

open List

-- Define median and mean for List ℝ
def median (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry

theorem math_contest_score_difference (scores : List ℝ) 
  (h1 : scores.count 50 = (scores.length * 5) / 100)
  (h2 : scores.count 60 = (scores.length * 20) / 100)
  (h3 : scores.count 70 = (scores.length * 25) / 100)
  (h4 : scores.count 80 = (scores.length * 30) / 100)
  (h5 : scores.count 90 = scores.length - (scores.count 50 + scores.count 60 + scores.count 70 + scores.count 80)) :
  median scores - mean scores = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_contest_score_difference_l718_71826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_to_grandma_distance_l718_71825

/-- Represents the distance from the bakery to grandmother's house -/
def bakery_to_grandma : ℝ := sorry

/-- Distance from apartment to bakery -/
def apartment_to_bakery : ℝ := 9

/-- Distance from grandmother's house to apartment -/
def grandma_to_apartment : ℝ := 27

/-- Theorem stating that the distance from bakery to grandmother's house is 30 miles -/
theorem bakery_to_grandma_distance : bakery_to_grandma = 30 := by
  have round_trip_difference : apartment_to_bakery + bakery_to_grandma + grandma_to_apartment = 2 * bakery_to_grandma + 6 := by sorry
  -- Proof steps
  sorry

#check bakery_to_grandma_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_to_grandma_distance_l718_71825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_count_theorem_l718_71850

noncomputable def jungkook_card : ℚ := 8/10
noncomputable def yoongi_card : ℚ := 1/2
noncomputable def yoojung_card : ℚ := 9/10

def threshold : ℚ := 3/10

def count_above_threshold (cards : List ℚ) : ℕ :=
  cards.filter (λ x => x ≥ threshold) |>.length

theorem card_count_theorem :
  count_above_threshold [jungkook_card, yoongi_card, yoojung_card] = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_count_theorem_l718_71850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equality_l718_71853

-- Define the domain for x ≤ 0
def NonPositive : Set ℝ := {x | x ≤ 0}

-- Define the reference function
def f (x : ℝ) : ℝ := x^2 - x + 2

-- Define the given functions
def f1 (x : ℝ) : ℝ := x^2 + |x| + 2
def f2 (t : ℝ) : ℝ := t^2 - t + 2
def f3 (x : ℝ) : ℝ := x^2 - |x| + 2

-- Use noncomputable for f4 due to the use of Real.sqrt
noncomputable def f4 (x : ℝ) : ℝ := (Real.sqrt (-x))^2 + Real.sqrt (x^4) + 2

theorem functions_equality :
  (∀ x ∈ NonPositive, f x = f1 x) ∧
  (∀ x ∈ NonPositive, f x = f2 x) ∧
  (∀ x ∈ NonPositive, f x = f4 x) ∧
  ¬(∀ x ∈ NonPositive, f x = f3 x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equality_l718_71853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_a_l718_71870

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos x)))))))

theorem f_derivative_at_a (a : ℝ) (h : a = Real.cos a) : 
  deriv f a = a^8 - 4*a^6 + 6*a^4 - 4*a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_a_l718_71870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_with_property_p_l718_71821

/-- A convex pentagon with vertices A, B, C, D, and E -/
structure ConvexPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Property P: All triangles formed by three consecutive vertices have area 1 -/
def hasPropertyP (p : ConvexPentagon) : Prop :=
  triangleArea p.A p.B p.C = 1 ∧
  triangleArea p.B p.C p.D = 1 ∧
  triangleArea p.C p.D p.E = 1 ∧
  triangleArea p.D p.E p.A = 1 ∧
  triangleArea p.E p.A p.B = 1

/-- The area of a pentagon given its five vertices -/
noncomputable def pentagonArea (p : ConvexPentagon) : ℝ := sorry

/-- Theorem: The area of a convex pentagon with property P is (5 + √5) / 2 -/
theorem pentagon_area_with_property_p (p : ConvexPentagon) (h : hasPropertyP p) :
  pentagonArea p = (5 + Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_with_property_p_l718_71821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l718_71834

theorem find_c :
  ∀ c : ℝ,
  let r := (λ x : ℝ ↦ 4 * x - 9)
  let s := (λ x : ℝ ↦ 5 * x - c)
  r (s 2) = 11 → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l718_71834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l718_71871

noncomputable def numPrimeFactors (n : ℕ+) : ℕ := (Nat.factors n.val).eraseDup.length

theorem max_prime_factors (a b : ℕ+) : 
  (∃ p : List ℕ, p.length = 10 ∧ (∀ q ∈ p, Nat.Prime q) ∧ (∀ q ∈ p, q ∣ a.gcd b)) →
  (∃ q : List ℕ, q.length = 35 ∧ (∀ r ∈ q, Nat.Prime r) ∧ (∀ r ∈ q, r ∣ a.lcm b)) →
  (numPrimeFactors a < numPrimeFactors b) →
  numPrimeFactors a ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l718_71871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_first_function_vertex_of_second_function_l718_71849

/-- Definition of a quadratic function -/
noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (a b c : ℝ) : ℝ := (4 * a * c - b^2) / (4 * a)

/-- Theorem: The vertex of y = 2x^2 - 4x - 1 is (1, -3) -/
theorem vertex_of_first_function :
  let f := quadratic_function 2 (-4) (-1)
  (vertex_x 2 (-4) = 1) ∧ (f (vertex_x 2 (-4)) = -3) := by
  sorry

/-- Theorem: The vertex of y = -3x^2 + 6x - 2 is (1, 1) -/
theorem vertex_of_second_function :
  let f := quadratic_function (-3) 6 (-2)
  (vertex_x (-3) 6 = 1) ∧ (vertex_y (-3) 6 (-2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_first_function_vertex_of_second_function_l718_71849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l718_71824

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line
def line_eq (x y : ℝ) : Prop := x + y - 5 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + y - 5| / Real.sqrt 2

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), circle_eq x y → 
  distance_to_line x y ≥ d ∧ 
  ∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧ distance_to_line x₀ y₀ = d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l718_71824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l718_71888

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |↑(ceiling x)| - |↑(floor (2 - x))|

-- State the theorem
theorem f_symmetry (x : ℝ) : f x = f (2 - x) := by
  -- Expand the definition of f
  unfold f
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l718_71888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_step_count_l718_71843

/-- The number of steps Petya counted while ascending the escalator -/
def steps_ascending : ℕ := 75

/-- The number of steps Petya counted while descending the escalator -/
def steps_descending : ℕ := 150

/-- The ratio of Petya's descending speed to ascending speed -/
def speed_ratio : ℚ := 3

/-- The speed of the escalator in steps per unit time -/
def escalator_speed : ℚ := 3/5

/-- The number of steps on the stopped escalator -/
def escalator_length : ℕ := 120

theorem escalator_step_count :
  (steps_ascending : ℚ) * (1 + escalator_speed) = 
  (steps_descending / speed_ratio) * (speed_ratio - escalator_speed) ∧
  escalator_length = (steps_ascending : ℕ) * (Nat.floor ((1 + escalator_speed))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_step_count_l718_71843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l718_71866

/-- The length of each train in meters -/
noncomputable def train_length : ℝ := 50

/-- The speed of the faster train in km/hr -/
noncomputable def faster_train_speed : ℝ := 46

/-- The speed of the slower train in km/hr -/
noncomputable def slower_train_speed : ℝ := 36

/-- The time taken for the faster train to pass the slower train in seconds -/
noncomputable def passing_time : ℝ := 36

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

theorem train_length_proof :
  let relative_speed := (faster_train_speed - slower_train_speed) * km_hr_to_m_s
  2 * train_length = relative_speed * passing_time :=
by sorry

#check train_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l718_71866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_sum_l718_71887

-- Define the triangles
def triangle_PQR : List (ℝ × ℝ) := [(0, 0), (0, 13), (17, 0)]
def triangle_PQR_prime : List (ℝ × ℝ) := [(34, 26), (46, 26), (34, 0)]

-- Define the rotation
noncomputable def clockwise_rotation (n : ℝ) (x y : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := point
  let angle := n * (Real.pi / 180)  -- Convert degrees to radians
  let new_x := x + (px - x) * Real.cos angle + (py - y) * Real.sin angle
  let new_y := y - (px - x) * Real.sin angle + (py - y) * Real.cos angle
  (new_x, new_y)

-- Theorem statement
theorem rotation_sum (n x y : ℝ) :
  (0 < n ∧ n < 180) →
  (∀ (p : ℝ × ℝ), p ∈ triangle_PQR →
    clockwise_rotation n x y p ∈ triangle_PQR_prime) →
  n + x + y = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_sum_l718_71887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_2_no_x_for_all_a_l718_71878

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + 2*a*x - a^2 + 1

-- Part 1: Solution set for a = 2
theorem solution_set_for_a_2 :
  Set.Icc (-3/2 : ℝ) (1/2) = {x : ℝ | f 2 x ≤ 0} :=
sorry

-- Part 2: Non-existence of x for all a in [-2, 2]
theorem no_x_for_all_a :
  ¬ ∃ x : ℝ, ∀ a : ℝ, a ∈ Set.Icc (-2) 2 → f a x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_2_no_x_for_all_a_l718_71878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_representation_l718_71806

theorem cosine_sum_product_representation :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos (2 * x) + Real.cos (4 * x) + Real.cos (8 * x) + Real.cos (10 * x) = 
      (a : ℝ) * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) ∧
    a + b + c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_representation_l718_71806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l718_71851

/-- A hyperbola with focus on the y-axis and eccentricity √3 has the standard equation y² - (x²/2) = 1 -/
theorem hyperbola_standard_equation :
  ∀ (a b c : ℝ),
    a > 0 → b > 0 → c > 0 →
    c^2 = a^2 + b^2 →
    c/a = Real.sqrt 3 →
    (fun (x y : ℝ) ↦ y^2/a^2 - x^2/b^2 = 1) =
    (fun (x y : ℝ) ↦ y^2 - x^2/2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l718_71851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_negative_two_l718_71891

def sequenceA (a₁ a₂ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | 1 => a₂
  | n + 2 => sequenceA a₁ a₂ (n + 1) + sequenceA a₁ a₂ n - a₁

theorem first_number_is_negative_two :
  ∃ (a₁ a₂ : ℤ), 
    sequenceA a₁ a₂ 7 = 27 ∧
    sequenceA a₁ a₂ 8 = 44 ∧
    sequenceA a₁ a₂ 9 = 71 →
    a₁ = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_negative_two_l718_71891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l718_71805

/-- A parabola with equation y^2 = 10x -/
structure Parabola where
  equation : ∀ x y : ℝ, y^2 = 10 * x

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (5/2, 0)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun x => x = -5/2

/-- The distance between a point and a line -/
noncomputable def distance_point_line (point : ℝ × ℝ) (line : ℝ → Prop) : ℝ :=
  sorry -- Definition of distance between a point and a line

/-- The distance from the focus to the directrix of the parabola y^2 = 10x is 5 -/
theorem parabola_focus_directrix_distance (p : Parabola) :
  distance_point_line (focus p) (directrix p) = 5 := by
  sorry

#check parabola_focus_directrix_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l718_71805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_2200_l718_71846

theorem least_factorial_divisible_by_2200 :
  (∀ k < 11, ¬(2200 ∣ Nat.factorial k)) ∧ (2200 ∣ Nat.factorial 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_2200_l718_71846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l718_71863

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (P Q R : Point) : ℝ := sorry

/-- Theorem about the ellipse properties and maximum area -/
theorem ellipse_properties_and_max_area (C : Ellipse) (M : Point) (l : Line) :
  C.a^2 / C.b^2 = 3 →  -- eccentricity condition
  (M.x^2 / C.a^2) + (M.y^2 / C.b^2) = 1 →  -- point M on ellipse
  M.x = -3 ∧ M.y = -1 →  -- coordinates of M
  l.a = 1 ∧ l.b = -1 ∧ l.c = -2 →  -- equation of line l
  ∃ (A B : Point),
    (A.x^2 / C.a^2) + (A.y^2 / C.b^2) = 1 ∧  -- A on ellipse
    (B.x^2 / C.a^2) + (B.y^2 / C.b^2) = 1 ∧  -- B on ellipse
    l.a * A.x + l.b * A.y + l.c = 0 ∧  -- A on line l
    l.a * B.x + l.b * B.y + l.c = 0 →  -- B on line l
  C.a^2 = 12 ∧ C.b^2 = 4 ∧  -- equation of ellipse C
  ∃ (P : Point),
    (P.x^2 / C.a^2) + (P.y^2 / C.b^2) = 1 ∧  -- P on ellipse
    P.x = -3 ∧ P.y = 1 ∧  -- coordinates of P
    ∀ (Q : Point),
      (Q.x^2 / C.a^2) + (Q.y^2 / C.b^2) = 1 →  -- Q on ellipse
      area_triangle P A B ≥ area_triangle Q A B ∧  -- P maximizes area
    area_triangle P A B = 9  -- maximum area
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l718_71863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_eq_5_sufficient_not_necessary_l718_71864

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line kx + 5y - 2 = 0 -/
noncomputable def slope1 (k : ℝ) : ℝ := -k / 5

/-- The slope of the line (4-k)x + y - 7 = 0 -/
def slope2 (k : ℝ) : ℝ := -(4 - k)

theorem k_eq_5_sufficient_not_necessary :
  (∀ k : ℝ, k = 5 → are_perpendicular (slope1 k) (slope2 k)) ∧
  (∃ k : ℝ, k ≠ 5 ∧ are_perpendicular (slope1 k) (slope2 k)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_eq_5_sufficient_not_necessary_l718_71864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l718_71875

theorem farm_fencing_cost (area : ℝ) (short_side : ℝ) (cost_per_meter : ℝ) :
  area = 1200 ∧ short_side = 30 ∧ cost_per_meter = 11 →
  (let long_side := area / short_side
   let diagonal := Real.sqrt (long_side^2 + short_side^2)
   let total_length := long_side + short_side + diagonal
   let total_cost := total_length * cost_per_meter
   total_cost = 1320) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l718_71875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l718_71832

/-- A line passing through (2,1) that intersects the x-axis and y-axis -/
structure IntersectingLine where
  slope : ℝ

/-- The x-coordinate of the intersection point with the x-axis -/
noncomputable def IntersectingLine.x_intercept (l : IntersectingLine) : ℝ := 2 - 1 / l.slope

/-- The y-coordinate of the intersection point with the y-axis -/
noncomputable def IntersectingLine.y_intercept (l : IntersectingLine) : ℝ := 1 - 2 * l.slope

/-- The area of the triangle formed by the line and the coordinate axes -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ :=
  (1/2) * |l.x_intercept| * |l.y_intercept|

/-- The theorem stating that there are exactly 3 lines satisfying the conditions -/
theorem exactly_three_lines :
  ∃! (s : Finset IntersectingLine),
    s.card = 3 ∧
    ∀ l ∈ s, triangle_area l = 4 := by
  sorry

#check exactly_three_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l718_71832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_periodic_f_extrema_on_interval_l718_71812

/-- A function satisfying the given conditions -/
noncomputable def f (x : ℝ) : ℝ :=
  sorry

/-- The positive constant a -/
noncomputable def a : ℝ :=
  sorry

/-- The domain of f -/
def f_domain (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ k * Real.pi

axiom f_diff (x y : ℝ) (hx : f_domain x) (hy : f_domain y) :
  f (x - y) = f x - f y

axiom f_a : f a = 1

axiom f_pos (x : ℝ) (h : 0 < x ∧ x < 2 * a) :
  f x > 0

axiom a_pos : a > 0

theorem f_odd (x : ℝ) (hx : f_domain x) :
  f (-x) = -f x := by
  sorry

theorem f_periodic (x : ℝ) (hx : f_domain x) :
  f (x + 4 * a) = f x := by
  sorry

theorem f_extrema_on_interval :
  (∀ x ∈ Set.Icc (2 * a) (3 * a), f x ≤ 0) ∧
  (∃ x ∈ Set.Icc (2 * a) (3 * a), f x = 0) ∧
  (∃ x ∈ Set.Icc (2 * a) (3 * a), f x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_periodic_f_extrema_on_interval_l718_71812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equality_and_inequality_l718_71868

theorem exponential_equality_and_inequality (a b c : ℝ) 
  (h1 : (3 : ℝ)^a = (4 : ℝ)^b) (h2 : (4 : ℝ)^b = (6 : ℝ)^c) 
  (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  (2/a + 1/b = 2/c) ∧
  (∀ m : ℝ, m^2 + Real.sqrt 2 ≤ (a + b) / c → -Real.sqrt 6 / 2 ≤ m ∧ m ≤ Real.sqrt 6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equality_and_inequality_l718_71868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amber_total_cost_l718_71872

noncomputable def base_cost : ℝ := 25
noncomputable def text_cost_first_120 : ℝ := 0.03
noncomputable def text_cost_additional : ℝ := 0.02
noncomputable def talk_time_limit : ℝ := 25
noncomputable def additional_talk_cost_per_minute : ℝ := 0.15
def amber_text_messages : ℕ := 140
noncomputable def amber_talk_hours : ℝ := 27

noncomputable def calculate_text_cost (messages : ℕ) : ℝ :=
  let first_120_cost := (min messages 120 : ℝ) * text_cost_first_120
  let additional_cost := (max (messages - 120) 0 : ℝ) * text_cost_additional
  first_120_cost + additional_cost

noncomputable def calculate_talk_cost (hours : ℝ) : ℝ :=
  max (hours - talk_time_limit) 0 * 60 * additional_talk_cost_per_minute

theorem amber_total_cost :
  base_cost + calculate_text_cost amber_text_messages + calculate_talk_cost amber_talk_hours = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amber_total_cost_l718_71872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l718_71883

open Real

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = Real.pi

theorem triangle_problem (A B C a b c : ℝ) 
  (h1 : triangle_ABC A B C a b c)
  (h2 : cos (A - Real.pi/3) = 2 * cos A)
  (h3 : b = 2)
  (h4 : 1/2 * b * c * sin A = 3 * sqrt 3)
  (h5 : cos (2*C) = 1 - a^2 / (6*b^2)) :
  a = 2 * sqrt 7 ∧ (B = Real.pi/12 ∨ B = 7*Real.pi/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l718_71883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_theorem_l718_71848

/-- The value of a for which the line y = ax + 3 intersects the circle (x - 1)² + (y - 2)² = 4 to form a chord of length 2√3 -/
def chord_intersection_parameter : ℝ := 0

/-- The line equation y = ax + 3 -/
def line_equation (a x : ℝ) : ℝ := a * x + 3

/-- The circle equation (x - 1)² + (y - 2)² = 4 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- The chord length formed by the intersection -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

/-- Theorem stating that the chord_intersection_parameter satisfies the given conditions -/
theorem chord_intersection_theorem :
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    line_equation chord_intersection_parameter x₁ = y₁ ∧
    line_equation chord_intersection_parameter x₂ = y₂ ∧
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_theorem_l718_71848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_five_l718_71897

-- Define the piecewise function g
noncomputable def g (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then 2 * (a : ℝ) * x + 4
  else if x = 0 then 2 * (a : ℝ) * (b : ℝ)
  else (b : ℝ) * x + 2 * (c : ℝ)

-- State the theorem
theorem sum_abc_equals_five (a b c : ℕ) :
  g a b c 1 = 8 ∧ g a b c 0 = 12 ∧ g a b c (-1) = -6 → a + b + c = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_five_l718_71897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_value_l718_71847

theorem xyz_value (x y z : ℝ) 
  (h1 : (2 : ℝ)^x = (16 : ℝ)^(y+3)) 
  (h2 : (27 : ℝ)^y = (3 : ℝ)^(z-2)) 
  (h3 : (256 : ℝ)^z = (4 : ℝ)^(x+4)) : 
  x * y * z = 24.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_value_l718_71847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l718_71860

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def water_height_in_cylinder (cone_radius cone_height cylinder_radius : ℝ) : ℝ :=
  (cone_volume cone_radius cone_height) / (Real.pi * cylinder_radius^2)

theorem water_height_theorem (cone_radius cone_height cylinder_radius : ℝ) 
  (h1 : cone_radius = 15)
  (h2 : cone_height = 20)
  (h3 : cylinder_radius = 30) :
  water_height_in_cylinder cone_radius cone_height cylinder_radius = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l718_71860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verandah_area_correct_l718_71827

/-- Represents the dimensions and specifications of a room with a verandah -/
structure RoomWithVerandah where
  room_length : ℝ
  room_width : ℝ
  verandah_width_long : ℝ
  verandah_width_short1 : ℝ
  verandah_width_short2 : ℝ
  semicircle_radius : ℝ

/-- Calculates the area of the verandah for a given room -/
noncomputable def verandah_area (r : RoomWithVerandah) : ℝ :=
  2 * r.room_length * r.verandah_width_long +
  (r.room_width + 2 * r.verandah_width_long) * r.verandah_width_short1 +
  (r.room_width + r.verandah_width_long + r.verandah_width_short2) * r.verandah_width_short2 +
  Real.pi * r.semicircle_radius^2 / 2

/-- Theorem stating that the verandah area for the given room specifications is correct -/
theorem verandah_area_correct (r : RoomWithVerandah) 
  (h1 : r.room_length = 15)
  (h2 : r.room_width = 12)
  (h3 : r.verandah_width_long = 2)
  (h4 : r.verandah_width_short1 = 3)
  (h5 : r.verandah_width_short2 = 1)
  (h6 : r.semicircle_radius = 2) :
  verandah_area r = 60 + 48 + 15 + 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_verandah_area_correct_l718_71827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l718_71833

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin (2 * x) - Real.cos x ^ 2 - 1/2

noncomputable def α : ℝ := Real.arctan (2 * Real.sqrt 3)

theorem function_properties :
  ∃ (m : ℝ),
    f m α = -3/26 ∧
    m = Real.sqrt 3 / 2 ∧
    (∀ x : ℝ, f m (x + π) = f m x) ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 (π/3) ∪ Set.Icc (5*π/6) π →
      ∀ y : ℝ, y ∈ Set.Icc 0 (π/3) ∪ Set.Icc (5*π/6) π →
        x < y → f m x < f m y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l718_71833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweet_numbers_count_l718_71828

def next_term (n : ℕ) : ℕ :=
  if n ≤ 20 then 3 * n else n - 10

def is_sweet (start : ℕ) : Bool :=
  let rec sequence (n : ℕ) (fuel : ℕ) : Bool :=
    if fuel = 0 then true  -- Assume sweet if we run out of fuel
    else if n = 20 then false
    else sequence (next_term n) (fuel - 1)
  sequence start 100  -- Use a reasonable fuel value

def count_sweet (upper_bound : ℕ) : ℕ :=
  (List.range upper_bound).filter (fun n => is_sweet (n + 1)) |>.length

theorem sweet_numbers_count : count_sweet 50 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweet_numbers_count_l718_71828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fourth_l718_71831

def spinner_numbers : List Nat := [1, 2, 5, 6]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

def probability_divisible_by_five : Rat :=
  let total_outcomes := (spinner_numbers.length : Rat) ^ 3
  let favorable_outcomes := ((spinner_numbers.length : Rat) ^ 2) * (spinner_numbers.filter is_divisible_by_five).length
  favorable_outcomes / total_outcomes

theorem probability_is_one_fourth :
  probability_divisible_by_five = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fourth_l718_71831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_l718_71809

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  (|A * x₀ + B * y₀ + C|) / Real.sqrt (A^2 + B^2)

def is_tangent_line (a b c r : ℝ) : Prop :=
  distance_point_to_line 0 0 a b c = r

theorem tangent_line_circle (m : ℝ) (h : m > 0) :
  is_tangent_line 1 1 (-2) (Real.sqrt m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_l718_71809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_on_body_diagonal_l718_71895

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  side_length : ℝ

/-- Check if three lines are mutually skew -/
def are_mutually_skew (l1 l2 l3 : Line3D) : Prop := sorry

/-- Get three mutually skew edges of a cube -/
def get_skew_edges (c : Cube) : (Line3D × Line3D × Line3D) := sorry

/-- Get the body diagonal of a cube that doesn't intersect the given skew edges -/
def get_body_diagonal (c : Cube) (e1 e2 e3 : Line3D) : Line3D := sorry

/-- Calculate the distance between a point and a line -/
noncomputable def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ := sorry

/-- Check if a point is on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- The main theorem -/
theorem equidistant_points_on_body_diagonal (c : Cube) : 
  let (e1, e2, e3) := get_skew_edges c
  let d := get_body_diagonal c e1 e2 e3
  ∀ p : Point3D, point_on_line p d → 
    distance_point_to_line p e1 = distance_point_to_line p e2 ∧
    distance_point_to_line p e2 = distance_point_to_line p e3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_on_body_diagonal_l718_71895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_cylinder_l718_71838

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 12

-- Define the longest segment
noncomputable def longest_segment : ℝ := 2 * Real.sqrt 61

-- Theorem statement
theorem longest_segment_in_cylinder :
  longest_segment = Real.sqrt (cylinder_height^2 + (2 * cylinder_radius)^2) :=
by
  -- Proof goes here
  sorry

#eval cylinder_radius
#eval cylinder_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_cylinder_l718_71838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_tan_sin_equation_l718_71862

theorem unique_solution_tan_sin_equation :
  ∃! (n : ℕ), n > 0 ∧ Real.tan (π / (2 * n)) + Real.sin (π / (2 * n)) = n / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_tan_sin_equation_l718_71862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_neg_three_l718_71855

noncomputable def f (x : ℝ) : ℝ := (x^3 + 2*x^2 + 3*x + 4) / (x + 3)

theorem vertical_asymptote_at_neg_three :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (x : ℝ), 0 < |x + 3| ∧ |x + 3| < δ → |f x| > M :=
by
  sorry

#check vertical_asymptote_at_neg_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_neg_three_l718_71855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_special_triangle_l718_71892

noncomputable section

open Real

theorem cosine_value_in_special_triangle (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  sin A / a = sin B / b ∧ sin B / b = sin C / c →
  -- Given conditions
  8 * b = 5 * c →
  C = 2 * B →
  -- Conclusion
  cos C = 7 / 25 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_special_triangle_l718_71892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l718_71815

/-- Definition of the hyperbola C -/
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

/-- Definition of the slope product condition -/
def slope_product (x y : ℝ) : Prop :=
  (y / (x + 3)) * (y / (x - 3)) = 16 / 9

/-- Definition of the foci -/
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

/-- Definition of the center of the inscribed circle of a triangle -/
def is_center_of_inscribed_circle_of_triangle (c : ℝ × ℝ) (a b d : ℝ × ℝ) : Prop :=
  sorry -- Actual definition would go here

/-- Main theorem -/
theorem hyperbola_properties :
  ∀ (x y : ℝ),
    slope_product x y →
    hyperbola x y →
    (∃ (cx cy : ℝ), x < 0 →
      (cx = -3 ∧
        is_center_of_inscribed_circle_of_triangle (cx, cy) F₁ (x, y) F₂)) :=
by
  sorry -- Proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l718_71815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_rate_l718_71899

theorem milk_production_rate (a b c d e : ℝ) (ha : a > 0) (hc : c > 0) :
  (d * e * (b / (a * c))) = (b * d * e) / (a * c) := by
  field_simp
  ring

#check milk_production_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_rate_l718_71899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l718_71802

/-- Given a hyperbola C and a parabola with specific properties, prove the equation of C -/
theorem hyperbola_equation (a : ℝ) : 
  -- Hyperbola C: x^2 - y^2 = a^2
  -- Center at the origin and foci on the x-axis (implied by the equation)
  -- C intersects the directrix of the parabola y^2 = 16x
  -- |AB| = 4√3, where A and B are intersection points
  (∃ (x y : ℝ), x^2 - y^2 = a^2 ∧ y^2 = 16*x ∧ x = -4) ∧
  (∃ (y : ℝ), |(-4 + y) - (-4 - y)| = 4 * Real.sqrt 3) →
  -- The equation of hyperbola C is x^2 - y^2 = 4
  a^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l718_71802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_multiple_of_three_l718_71861

theorem expression_multiple_of_three (n : ℕ) (h : n ≥ 9) :
  ∃ k : ℕ, (n + 1) * (n + 3) = 3 * k := by
  use ((n + 1) * (n + 3)) / 3
  have h1 : (n + 1) % 3 = 0 ∨ (n + 2) % 3 = 0 ∨ (n + 3) % 3 = 0 := by
    sorry -- This can be proved using properties of modular arithmetic
  cases h1 with
  | inl h2 => 
    sorry -- Prove that if n + 1 is divisible by 3, then (n + 1) * (n + 3) is divisible by 3
  | inr h2 => cases h2 with
    | inl h3 => 
      sorry -- Prove that if n + 2 is divisible by 3, then (n + 1) * (n + 3) is divisible by 3
    | inr h4 => 
      sorry -- Prove that if n + 3 is divisible by 3, then (n + 1) * (n + 3) is divisible by 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_multiple_of_three_l718_71861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l718_71865

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  (1.1 * L * 0.9 * W - L * W) / (L * W) = -0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l718_71865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l718_71813

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (1, 5)
def C : ℝ → ℝ × ℝ := λ m ↦ (m, 3)

-- Define vectors
def vec_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vec_OC (m : ℝ) : ℝ × ℝ := ((C m).1 - O.1, (C m).2 - O.2)

-- Define the perpendicularity condition
def perpendicular (m : ℝ) : Prop :=
  vec_AB.1 * (vec_OC m).1 + vec_AB.2 * (vec_OC m).2 = 0

-- Theorem statement
theorem find_m : ∃ m : ℝ, perpendicular m ∧ m = 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l718_71813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_fare_theorem_l718_71857

/-- Taxi fare calculation function -/
noncomputable def taxi_fare (distance : ℝ) : ℝ :=
  if distance ≤ 5 then 10.8 else 10.8 + 1.2 * (distance - 5)

theorem midpoint_fare_theorem (distance_AB : ℝ) : 
  (taxi_fare distance_AB = 24) →
  (taxi_fare (distance_AB - 0.46) = 24) →
  (15.46 < distance_AB) →
  (distance_AB ≤ 16) →
  (taxi_fare (distance_AB / 2) = 14.4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_fare_theorem_l718_71857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l718_71819

/-- The eccentricity of a hyperbola with equation 3x^2 - my^2 = 3m (m > 0) and focus at (3, 0) is √6/2 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) : 
  let hyperbola := {(x, y) : ℝ × ℝ | 3 * x^2 - m * y^2 = 3 * m}
  let focus : ℝ × ℝ := (3, 0)
  focus ∈ {f | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ hyperbola → 
    (abs ((p.1 - f.1)^2 + (p.2 - f.2)^2) - abs ((p.1 + f.1)^2 + (p.2 - f.2)^2) = 4 * a * b)}
  →
  ∃ (a c : ℝ), a > 0 ∧ c > 0 ∧ c / a = Real.sqrt 6 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l718_71819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_baseball_cards_l718_71842

theorem mary_baseball_cards (initial torn fred_gave bought : ℕ) :
  initial ≥ torn →
  (initial - torn) + fred_gave + bought = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_baseball_cards_l718_71842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l718_71816

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (π / 4) (π / 2) →
    f (x₀ - π / 12) = 6 / 5 →
    Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l718_71816
