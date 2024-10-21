import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l1017_101721

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_shift_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), f (x - φ) = -f (-x - φ)) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), f (x - ψ) = -f (-x - ψ)) → φ ≤ ψ) ∧
  φ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l1017_101721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_less_than_sin_neg_four_l1017_101737

theorem cos_less_than_sin_neg_four : Real.cos (-4) < Real.sin (-4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_less_than_sin_neg_four_l1017_101737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_area_l1017_101796

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  Real.sin t.A + Real.sin t.B = (Real.cos t.A + Real.cos t.B) * Real.sin t.C

def satisfies_condition_2 (t : Triangle) : Prop :=
  t.a + t.b + t.c = 1 + Real.sqrt 2

-- Define the theorems
theorem right_triangle (t : Triangle) (h : satisfies_condition_1 t) : 
  t.C = Real.pi / 2 := by sorry

theorem max_area (t : Triangle) (h1 : satisfies_condition_1 t) (h2 : satisfies_condition_2 t) :
  t.a * t.b / 2 ≤ 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_area_l1017_101796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_see_again_is_correct_sum_is_correct_l1017_101744

/-- The time until Jenny and Kenny can see each other again -/
noncomputable def time_to_see_again (
  path_distance : ℝ)  -- Distance between parallel paths
  (jenny_speed : ℝ)   -- Jenny's walking speed
  (kenny_speed : ℝ)   -- Kenny's walking speed
  (building_diameter : ℝ)  -- Diameter of the circular building
  (initial_distance : ℝ)   -- Initial distance between Jenny and Kenny
  : ℝ :=
  200 / 3

/-- Theorem stating that the time to see again is 200/3 seconds under given conditions -/
theorem time_to_see_again_is_correct 
  (h1 : path_distance = 300)
  (h2 : jenny_speed = 2)
  (h3 : kenny_speed = 4)
  (h4 : building_diameter = 100)
  (h5 : initial_distance = 200) :
  time_to_see_again path_distance jenny_speed kenny_speed building_diameter initial_distance = 200 / 3 :=
by
  -- Unfold the definition of time_to_see_again
  unfold time_to_see_again
  -- The result follows directly from the definition
  rfl

/-- The sum of the numerator and denominator of the fraction 200/3 -/
def sum_of_numerator_and_denominator : ℕ := 200 + 3

/-- Theorem stating that the sum of the numerator and denominator is 203 -/
theorem sum_is_correct : sum_of_numerator_and_denominator = 203 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_see_again_is_correct_sum_is_correct_l1017_101744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_cost_calculation_l1017_101780

/-- The cost of potatoes given a known price for a certain weight -/
noncomputable def potato_cost (known_weight : ℚ) (known_cost : ℚ) (desired_weight : ℚ) : ℚ :=
  (desired_weight / known_weight) * known_cost

/-- Theorem: The cost of 5 kg of potatoes is $15, given that 2 kg costs $6 -/
theorem potato_cost_calculation :
  potato_cost 2 6 5 = 15 := by
  -- Unfold the definition of potato_cost
  unfold potato_cost
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_cost_calculation_l1017_101780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1017_101761

theorem equation_solution :
  ∃ t : ℝ, 5 * (5 : ℝ)^t + Real.sqrt (25 * (25 : ℝ)^t) = 50 ∧ t = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1017_101761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sin_squared_l1017_101788

noncomputable def angle_sequence : List ℝ := 
  (List.range 29).map (fun n => (n + 1) * 6 * Real.pi / 180)

theorem sum_of_sin_squared : 
  (angle_sequence.map (fun θ => Real.sin θ ^ 2)).sum = 15.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sin_squared_l1017_101788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1017_101785

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- State the theorem
theorem f_domain : Set.Ici 1 = {x : ℝ | ∃ y, f x = y} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1017_101785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_sum_is_ten_l1017_101794

/-- The set of integers from -5 to 10, inclusive -/
def IntegerSet : Set Int := {x | -5 ≤ x ∧ x ≤ 10}

/-- A 4x4 matrix of integers -/
def Matrix4x4 : Type := Fin 4 → Fin 4 → Int

/-- The sum of a row in the matrix -/
def rowSum (m : Matrix4x4) (i : Fin 4) : Int :=
  Finset.sum (Finset.range 4) (λ j => m i j)

/-- The sum of a column in the matrix -/
def colSum (m : Matrix4x4) (j : Fin 4) : Int :=
  Finset.sum (Finset.range 4) (λ i => m i j)

/-- The sum of the main diagonal from top-left to bottom-right -/
def diagSum1 (m : Matrix4x4) : Int :=
  Finset.sum (Finset.range 4) (λ i => m i i)

/-- The sum of the main diagonal from top-right to bottom-left -/
def diagSum2 (m : Matrix4x4) : Int :=
  Finset.sum (Finset.range 4) (λ i => m i (3 - i))

/-- Predicate to check if all elements of the matrix are in the IntegerSet -/
def validMatrix (m : Matrix4x4) : Prop :=
  ∀ i j, m i j ∈ IntegerSet

/-- Predicate to check if all row sums, column sums, and diagonal sums are equal -/
def equalSums (m : Matrix4x4) : Prop :=
  ∃ s, (∀ i, rowSum m i = s) ∧
       (∀ j, colSum m j = s) ∧
       (diagSum1 m = s) ∧
       (diagSum2 m = s)

theorem common_sum_is_ten (m : Matrix4x4) 
  (h1 : validMatrix m) (h2 : equalSums m) : 
  ∃ s, s = 10 ∧ equalSums m ∧ 
  (∀ i j, m i j ∈ IntegerSet) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_sum_is_ten_l1017_101794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_repeat_achievable_distribution_l1017_101775

/-- Represents the state of the ball distribution in boxes -/
structure BallDistribution where
  boxes : List ℕ
  lastBox : Fin (boxes.length)

/-- Represents a move in the ball distribution system -/
def move (d : BallDistribution) : BallDistribution :=
  sorry

/-- Checks if two ball distributions are equal -/
def eqDistribution (d1 d2 : BallDistribution) : Prop :=
  sorry

/-- Theorem: The system will return to its initial state after a finite number of moves -/
theorem eventual_repeat (initial : BallDistribution) :
  ∃ n : ℕ, eqDistribution (Nat.iterate move n initial) initial := by
  sorry

/-- Theorem: It is possible to transform any initial state into any other state -/
theorem achievable_distribution (initial target : BallDistribution) :
  ∃ moves : List (Fin initial.boxes.length),
    eqDistribution (List.foldl (λ d i => move d) initial moves) target := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_repeat_achievable_distribution_l1017_101775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_consecutive_integers_divisibility_l1017_101727

theorem sum_of_four_consecutive_integers_divisibility (n : ℤ) : 
  ∃ (p : ℕ), Nat.Prime p ∧ (p : ℤ) ∣ ((n - 1) + n + (n + 1) + (n + 2)) ∧
  ∀ (q : ℕ), Nat.Prime q → (∀ (m : ℤ), (q : ℤ) ∣ ((m - 1) + m + (m + 1) + (m + 2))) → q = p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_consecutive_integers_divisibility_l1017_101727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101799

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := Real.sin (↑(floor (Real.cos x))) + Real.cos (↑(floor (Real.sin x)))

theorem f_properties :
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∃ x, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∃ x, f x > Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_limit_condition_l1017_101735

/-- Geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n ↦ a * q ^ (n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  (a * (1 - q^n)) / (1 - q)

/-- The condition a + q = 1 is necessary but not sufficient for the limit of geometric sum to be 1 -/
theorem geometric_sum_limit_condition (a q : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_sum a q n - 1| < ε) →
  (a + q = 1 ∧
  ¬(a + q = 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_sum a q n - 1| < ε)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_limit_condition_l1017_101735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_end_on_line_is_two_fifteenths_l1017_101791

-- Define the type for positions on the plane
def Position := ℤ × ℤ

-- Define the region
def InRegion (p : Position) : Prop :=
  p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 < 4

-- Define the stopping condition
def OnBoundary (p : Position) : Prop :=
  p.1 + p.2 = 4 ∨ p.1 = 0 ∨ p.2 = 0

-- Define the possible moves
def PossibleMoves (p : Position) : List Position :=
  let x := p.1
  let y := p.2
  [(x-1, y-1), (x-1, y), (x-1, y+1),
   (x, y-1),             (x, y+1),
   (x+1, y-1), (x+1, y), (x+1, y+1)]

-- Define the probability of ending on x+y=4
noncomputable def ProbEndOnLine (start : Position) : ℚ :=
  sorry  -- Proof to be implemented

-- The main theorem
theorem prob_end_on_line_is_two_fifteenths :
  ProbEndOnLine (1, 1) = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_end_on_line_is_two_fifteenths_l1017_101791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_three_equals_two_thirds_l1017_101743

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := 2 * x - 1

noncomputable def f (x : ℝ) : ℝ := 
  let y := (x + 1) / 2  -- This is g⁻¹(x)
  (1 + y^2) / (3 * y^2)

-- State the theorem
theorem f_negative_three_equals_two_thirds :
  f (-3) = 2/3 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_three_equals_two_thirds_l1017_101743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_tangent_l1017_101798

theorem angle_sum_tangent (α β : ℝ) (h_acute_α : 0 < α ∧ α < π/2)
                          (h_acute_β : 0 < β ∧ β < π/2)
                          (h_tan_α : Real.tan α = 1/8)
                          (h_sin_β : Real.sin β = 1/3) :
  Real.tan (α + 2*β) = 15/56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_tangent_l1017_101798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_omega_value_l1017_101729

/-- Given that f(x) = sin(ωx) + cos(ωx) and (π/8, 0) is a center of symmetry for the graph of f(x), prove that ω = 6 -/
theorem symmetry_implies_omega_value (ω : ℝ) : 
  (∀ x : ℝ, Real.sin (ω * x) + Real.cos (ω * x) = 
            Real.sin (ω * (π / 4 - x)) + Real.cos (ω * (π / 4 - x))) → 
  ω = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_omega_value_l1017_101729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101734

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∀ x : ℝ, f x = 0 ↔ ∃ k : ℤ, x = k * Real.pi / 2 - Real.pi / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangles_l1017_101708

/-- The maximum number of equilateral triangles that can be formed with 6 line segments of length 2 -/
theorem max_equilateral_triangles (num_segments : ℕ) (segment_length : ℝ) : 
  num_segments = 6 → segment_length = 2 → 
  ∃ (max_triangles : ℕ), max_triangles = 4 ∧ 
  (∀ n : ℕ, n > max_triangles → 
    ¬ ∃ (triangles : Finset (Finset ℕ)), 
      triangles.card = n ∧ 
      (∀ t ∈ triangles, t.card = 3 ∧ 
        (∀ i j : ℕ, i ∈ t → j ∈ t → i ≠ j → ∃ k : ℕ, k ∈ t ∧ k ≠ i ∧ k ≠ j ∧ 
          segment_length = 2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangles_l1017_101708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_M_bounds_l1017_101773

/-- Property M for a set of integers -/
def property_M (A : Set ℤ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x ≠ y → |x - y| > (x * y : ℚ) / 25

/-- A set with property M -/
structure SetWithPropertyM where
  A : Set ℤ
  elements : List ℤ
  elements_sorted : elements.Sorted (· < ·)
  elements_positive : ∀ a, a ∈ elements → 0 < a
  elements_distinct : elements.Nodup
  elements_in_A : ∀ a, a ∈ elements → a ∈ A
  A_in_elements : ∀ a, a ∈ A → a ∈ elements
  has_property_M : property_M A

theorem property_M_bounds (S : SetWithPropertyM) :
  (1 / S.elements.head! - 1 / S.elements.getLast! : ℚ) ≥ (S.elements.length - 1 : ℚ) / 25 ∧
  S.elements.length ≤ 9 := by
  sorry

#check property_M_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_M_bounds_l1017_101773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_cube_l1017_101742

/-- Helper definition for "x = a when y = b" -/
def equals_when (x y a b : ℝ) : Prop := y = b → x = a

/-- Given that x varies inversely with the cube of y, and x = 8 when y = 1,
    prove that x = 1 when y = 2 -/
theorem inverse_variation_cube (x y : ℝ) (h : ∃ k : ℝ, ∀ y, x * y^3 = k) 
  (h1 : equals_when x y 8 1) : equals_when x y 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_cube_l1017_101742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_satisfying_equation_l1017_101748

theorem ordered_pairs_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ p : ℕ × ℕ ↦ 
      p.1 * p.2 + 97 = 18 * Nat.lcm p.1 p.2 + 14 * Nat.gcd p.1 p.2 ∧ 
      p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_satisfying_equation_l1017_101748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_combinable_l1017_101765

-- Define the set of square roots
noncomputable def square_roots : Set ℝ := {Real.sqrt (1/2), Real.sqrt 8, Real.sqrt 12, Real.sqrt 18}

-- Define what it means for a real number to be combinable with √2
def combinable_with_sqrt2 (x : ℝ) : Prop :=
  ∃ (q : ℚ), x = q * Real.sqrt 2

-- Statement of the theorem
theorem unique_non_combinable :
  ∃! (x : ℝ), x ∈ square_roots ∧ ¬(combinable_with_sqrt2 x) :=
by
  sorry

#check unique_non_combinable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_combinable_l1017_101765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_polynomial_has_five_nonzero_terms_l1017_101783

/-- The number of nonzero terms in the expansion of a given polynomial expression -/
noncomputable def nonzeroTerms (p : Polynomial ℝ) : ℕ :=
  p.support.card

/-- The polynomial expression to be expanded -/
noncomputable def expandedPolynomial : Polynomial ℝ :=
  (Polynomial.X + 3) * (3 * Polynomial.X^2 + 2 * Polynomial.X + 8) +
  4 * (Polynomial.X^4 - 3 * Polynomial.X^3 + Polynomial.X^2) -
  2 * (Polynomial.X^3 - 3 * Polynomial.X^2 + 6 * Polynomial.X)

theorem expanded_polynomial_has_five_nonzero_terms :
  nonzeroTerms expandedPolynomial = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_polynomial_has_five_nonzero_terms_l1017_101783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l1017_101719

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The specific point in polar coordinates -/
noncomputable def polar_point : ℝ × ℝ := (10, 2 * Real.pi / 3)

/-- The expected rectangular coordinates -/
noncomputable def expected_rectangular : ℝ × ℝ := (-5, 5 * Real.sqrt 3)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = expected_rectangular := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l1017_101719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1017_101787

/-- An arithmetic sequence {a_n} -/
def a : ℕ → ℚ := sorry

/-- The common difference of the arithmetic sequence -/
def d : ℚ := sorry

/-- The sequence b_n defined in terms of a_n -/
def b : ℕ → ℚ := sorry

/-- The sum of the first n terms of b_n -/
def S : ℕ → ℚ := sorry

theorem arithmetic_sequence_problem (h1 : ∀ n, a (n + 1) > a n)
  (h2 : a 4 * a 7 = 15)
  (h3 : a 3 + a 8 = 8)
  (h4 : ∀ n, a n = a 1 + d * (n - 1))
  (h5 : ∀ n ≥ 2, b n = 1 / (9 * a (n - 1) * a n))
  (h6 : b 1 = 1 / 3) :
  (∀ n, a n = 2 / 3 * n + 1 / 3) ∧
  (∀ n, S n = n / (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1017_101787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_eq_pi_div_4_is_cone_l1017_101715

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the equation φ = π/4
def equationPhi (s : SphericalCoord) : Prop :=
  s.φ = Real.pi / 4

-- Define a cone
def isCone (shape : Set SphericalCoord) : Prop :=
  ∃ (α : ℝ), ∀ (s : SphericalCoord), s ∈ shape → s.φ = α

-- Theorem statement
theorem phi_eq_pi_div_4_is_cone :
  let shape := {s : SphericalCoord | equationPhi s}
  isCone shape := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_eq_pi_div_4_is_cone_l1017_101715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_arrangement_l1017_101757

/-- Represents the arrangement of numbers in the triangle -/
structure TriangleArrangement where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Checks if the given arrangement satisfies the conditions -/
def is_valid_arrangement (arr : TriangleArrangement) : Prop :=
  arr.A ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.B ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.C ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.D ∈ ({6, 7, 8, 9} : Set Nat) ∧
  arr.A ≠ arr.B ∧ arr.A ≠ arr.C ∧ arr.A ≠ arr.D ∧
  arr.B ≠ arr.C ∧ arr.B ≠ arr.D ∧
  arr.C ≠ arr.D

/-- Checks if the sums along each side of the triangle are equal -/
def has_equal_sums (arr : TriangleArrangement) : Prop :=
  arr.A + arr.C + 3 + 4 = arr.A + 5 + 1 + arr.B ∧
  arr.A + arr.C + 3 + 4 = 5 + arr.D + 2 + 4

/-- The main theorem stating that there's only one valid arrangement -/
theorem unique_triangle_arrangement :
  ∃! arr : TriangleArrangement,
    is_valid_arrangement arr ∧ has_equal_sums arr ∧
    arr.A = 6 ∧ arr.B = 8 ∧ arr.C = 7 ∧ arr.D = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_arrangement_l1017_101757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_partition_implies_equal_elements_l1017_101777

/-- A set with the property that removing any element allows equal-sum partitioning -/
def EqualSumPartitionable (E : Finset ℕ) (n : ℕ) : Prop :=
  (E.card = 2*n + 1) ∧ 
  ∀ x ∈ E, ∃ (A B : Finset ℕ), 
    A ∪ B = E \ {x} ∧ 
    A ∩ B = ∅ ∧ 
    A.card = n ∧ 
    B.card = n ∧ 
    (A.sum id = B.sum id)

/-- Theorem: If a set is EqualSumPartitionable, then all its elements are equal -/
theorem equal_sum_partition_implies_equal_elements 
  (E : Finset ℕ) (n : ℕ) (h : EqualSumPartitionable E n) : 
  ∀ x y, x ∈ E → y ∈ E → x = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_partition_implies_equal_elements_l1017_101777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_is_270_l1017_101747

/-- A function that checks if a 3-digit number satisfies the condition that its units digit is at least twice its tens digit -/
def satisfiesCondition (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 ≥ 2 * ((n / 10) % 10)

/-- The count of 3-digit numbers satisfying the condition -/
def countSatisfyingNumbers : ℕ :=
  (Finset.range 1000).filter (fun n => satisfiesCondition n) |>.card

/-- Theorem stating that the count of numbers satisfying the condition is 270 -/
theorem count_satisfying_numbers_is_270 : countSatisfyingNumbers = 270 := by
  sorry

#eval countSatisfyingNumbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_is_270_l1017_101747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1017_101746

noncomputable def m : ℝ := (Real.log 3)⁻¹

noncomputable def a : ℝ := Real.cos m
noncomputable def b : ℝ := 1 - (1/2) * m^2
noncomputable def c : ℝ := (Real.sin m) / m

theorem order_of_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1017_101746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_volume_ratio_l1017_101755

/-- The volume of a rectangular prism -/
noncomputable def volume (height width depth : ℝ) : ℝ := height * width * depth

/-- The ratio of volumes between two rectangular prisms -/
noncomputable def volume_ratio (h1 w1 d1 h2 w2 d2 : ℝ) : ℝ :=
  (volume h1 w1 d1) / (volume h2 w2 d2)

theorem lamp_volume_ratio :
  volume_ratio 2.33 1.25 1.25 1 0.5 0.5 = 14.578125 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_volume_ratio_l1017_101755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_award_winning_work_probability_l1017_101712

noncomputable def elderly_ratio : ℝ := 3/5
noncomputable def middle_aged_ratio : ℝ := 1/5
noncomputable def children_ratio : ℝ := 1/5

noncomputable def elderly_win_prob : ℝ := 0.6
noncomputable def middle_aged_win_prob : ℝ := 0.2
noncomputable def children_win_prob : ℝ := 0.1

theorem award_winning_work_probability :
  elderly_ratio * elderly_win_prob + 
  middle_aged_ratio * middle_aged_win_prob + 
  children_ratio * children_win_prob = 0.42 := by
  -- Calculation steps
  have h1 : elderly_ratio * elderly_win_prob = 3/5 * 0.6 := rfl
  have h2 : middle_aged_ratio * middle_aged_win_prob = 1/5 * 0.2 := rfl
  have h3 : children_ratio * children_win_prob = 1/5 * 0.1 := rfl
  
  -- Combine the steps
  calc
    elderly_ratio * elderly_win_prob + 
    middle_aged_ratio * middle_aged_win_prob + 
    children_ratio * children_win_prob
    = 3/5 * 0.6 + 1/5 * 0.2 + 1/5 * 0.1 := by rw [h1, h2, h3]
    _ = 0.36 + 0.04 + 0.02 := by norm_num
    _ = 0.42 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_award_winning_work_probability_l1017_101712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_root_l1017_101741

theorem cubic_equation_root (a b : ℚ) :
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 45 = 0 ∧ x = -2 - 5*Complex.I*Real.sqrt 3) →
  a = 239/71 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_root_l1017_101741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_strategy_l1017_101767

/-- Represents the cost and pricing structure of spicy strips -/
structure SpicyStrips where
  regular_cost : ℚ
  weilong_cost : ℚ
  regular_price : ℚ
  weilong_price : ℚ

/-- Represents the purchasing strategy for spicy strips -/
structure PurchaseStrategy where
  weilong_quantity : ℕ
  regular_quantity : ℕ

/-- Calculates the profit for a given purchase strategy -/
def calculate_profit (s : SpicyStrips) (p : PurchaseStrategy) : ℚ :=
  (s.weilong_price - s.weilong_cost) * p.weilong_quantity +
  (s.regular_price - s.regular_cost) * p.regular_quantity

/-- Theorem stating the optimal purchase strategy for maximum profit -/
theorem optimal_purchase_strategy (s : SpicyStrips)
  (h1 : s.weilong_cost = 2 * s.regular_cost)
  (h2 : 40 / s.weilong_cost - 10 / s.regular_cost = 10)
  (h3 : s.regular_price = 2)
  (h4 : s.weilong_price = 7/2) :
  ∃ (p : PurchaseStrategy),
    p.weilong_quantity = 200 ∧
    p.regular_quantity = 600 ∧
    s.regular_cost = 1 ∧
    s.weilong_cost = 2 ∧
    (∀ (q : PurchaseStrategy),
      q.weilong_quantity * s.weilong_cost + q.regular_quantity * s.regular_cost ≤ 1000 →
      q.regular_quantity ≤ 3 * q.weilong_quantity →
      calculate_profit s q ≤ calculate_profit s p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_strategy_l1017_101767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_extreme_values_f_monotonic_condition_f_range_condition_l1017_101762

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt x * |x - a|
def g (x : ℝ) : ℝ := 16 * x^3 - 24 * x^2 - 15 * x - 2

-- Theorem 1: Extreme values of g
theorem g_extreme_values :
  (∀ x, g x ≤ 0) ∧ (∃ x, g x = 0) ∧ (∃ x, g x = -27) ∧ (∀ x, g x ≥ -27) :=
by sorry

-- Theorem 2: Monotonicity condition for f
theorem f_monotonic_condition (a : ℝ) :
  (∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ↔ a ≤ 0 :=
by sorry

-- Theorem 3: Range of a for specific condition on f
theorem f_range_condition (a : ℝ) :
  a > 0 →
  (∃ t, t > a ∧ 
    (∀ x, 0 ≤ x ∧ x ≤ t → 0 ≤ f a x ∧ f a x ≤ t/2) ∧
    (∃ x, 0 ≤ x ∧ x ≤ t ∧ f a x = 0) ∧
    (∃ x, 0 ≤ x ∧ x ≤ t ∧ f a x = t/2)) →
  0 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_extreme_values_f_monotonic_condition_f_range_condition_l1017_101762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tip_percentage_approx_15_percent_l1017_101769

/-- Calculates the maximum tip percentage given the total allowed spending, sales tax rate, and meal cost. -/
noncomputable def max_tip_percentage (total_allowed : ℝ) (sales_tax_rate : ℝ) (meal_cost : ℝ) : ℝ :=
  let sales_tax := sales_tax_rate * meal_cost
  let max_tip := total_allowed - meal_cost - sales_tax
  (max_tip / meal_cost) * 100

/-- The maximum tip percentage is approximately 15% given the specified conditions. -/
theorem max_tip_percentage_approx_15_percent :
  let total_allowed := (75 : ℝ)
  let sales_tax_rate := (0.07 : ℝ)
  let meal_cost := (61.48 : ℝ)
  abs (max_tip_percentage total_allowed sales_tax_rate meal_cost - 15) < 0.01 := by
  sorry

-- Using #eval with noncomputable functions is not possible
-- Instead, we can use the following to check the result:
#check max_tip_percentage (75 : ℝ) (0.07 : ℝ) (61.48 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tip_percentage_approx_15_percent_l1017_101769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_ten_factorial_l1017_101718

open Nat

theorem divisors_of_ten_factorial (n : ℕ) : n = 9 ↔ 
  (∃ (s : Finset ℕ), s = {d : ℕ | d ∣ factorial 10 ∧ d > factorial 9} ∧ Finset.card s = n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_ten_factorial_l1017_101718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_white_given_popped_is_eight_thirteenths_l1017_101753

/-- Represents the probability that a randomly selected kernel that pops is white -/
noncomputable def prob_white_given_popped (
  total_kernels : ℝ
  ) (
  white_ratio : ℝ
  ) (
  yellow_ratio : ℝ
  ) (
  white_pop_ratio : ℝ
  ) (
  yellow_pop_ratio : ℝ
  ) : ℝ :=
  let white_kernels := total_kernels * white_ratio
  let yellow_kernels := total_kernels * yellow_ratio
  let white_popped := white_kernels * white_pop_ratio
  let yellow_popped := yellow_kernels * yellow_pop_ratio
  let total_popped := white_popped + yellow_popped
  white_popped / total_popped

theorem prob_white_given_popped_is_eight_thirteenths 
  (total_kernels : ℝ) 
  (h1 : total_kernels > 0) :
  prob_white_given_popped total_kernels (3/4) (1/4) (2/5) (3/4) = 8/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_white_given_popped_is_eight_thirteenths_l1017_101753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_has_one_element_l1017_101772

def U : Finset ℕ := {0, 1, 2, 3}

def A : Finset ℕ := U.filter (fun x => (x - 1) * (x - 3) ≤ 0)

theorem complement_A_has_one_element :
  Finset.card (U \ A) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_has_one_element_l1017_101772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101736

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6) - 4 * (Real.cos (ω * x / 2))^2 + 3

theorem f_properties (ω : ℝ) (h_ω : ω > 0) :
  -- The range of f is [-1, 3]
  (∀ x, -1 ≤ f ω x ∧ f ω x ≤ 3) ∧
  -- The distance between adjacent intersections with y = 1 is π/2
  (∃ x₁ x₂, x₁ < x₂ ∧ x₂ - x₁ = Real.pi / 2 ∧ f ω x₁ = 1 ∧ f ω x₂ = 1) →
  -- The intervals where f is monotonically decreasing
  (∃ k : ℤ, ∀ x, (k : ℝ) * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 6 → 
    ∀ y, x ≤ y → f ω y ≤ f ω x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101764

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.cos (2 * x)

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The maximum value in [0, π/2] is √2
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ Real.sqrt 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = Real.sqrt 2) ∧
  -- The minimum value in [0, π/2] is -1
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -1) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1017_101764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1017_101790

/-- The area of a rectangle given the coordinates of two opposite corners and a line equation for the other diagonal -/
theorem rectangle_area (B D : ℝ × ℝ) (a b c : ℝ) : 
  B = (4, 2) →
  D = (12, 8) →
  (∀ x y, x + 2*y - 18 = 0 → (x, y) ∈ Set.Icc B D) →
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) * Real.sqrt ((a^2 + b^2) / (a^2 + 4*b^2)) = 20 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1017_101790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1017_101739

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a * b ≤ 0 then a * b else -a / b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star x (Real.exp x)

-- Theorem statement
theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 / Real.exp 1 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1017_101739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_and_planes_relationships_l1017_101781

def vector_dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.fst * w.fst + v.snd.fst * w.snd.fst + v.snd.snd * w.snd.snd

def are_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.fst, k * w.snd.fst, k * w.snd.snd)

def are_perpendicular (v w : ℝ × ℝ × ℝ) : Prop :=
  vector_dot_product v w = 0

theorem lines_and_planes_relationships :
  let l₁_dir : ℝ × ℝ × ℝ := (2, 3, -1)
  let l₂_dir : ℝ × ℝ × ℝ := (-2, -3, 1)
  let α_normal : ℝ × ℝ × ℝ := (2, 2, -1)
  let β_normal : ℝ × ℝ × ℝ := (-3, 4, 2)
  (are_parallel l₁_dir l₂_dir) ∧
  (are_perpendicular α_normal β_normal) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_and_planes_relationships_l1017_101781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1017_101751

/-- The circle on which points A and B move -/
def circle_F (x y : ℝ) : Prop := (x - 1/2)^2 + y^2 = 4

/-- Point A is fixed at (-1/2, 0) -/
noncomputable def point_A : ℝ × ℝ := (-1/2, 0)

/-- The center of the circle -/
noncomputable def point_F : ℝ × ℝ := (1/2, 0)

/-- Point B moves on the circle -/
def point_B : ℝ × ℝ → Prop := λ p => circle_F p.1 p.2

/-- Point P is on the perpendicular bisector of AB and on BF -/
def point_P (b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  point_B b ∧ 
  (p.1 - point_A.1)^2 + (p.2 - point_A.2)^2 = (p.1 - b.1)^2 + (p.2 - b.2)^2 ∧
  ((p.1 - b.1) * (point_F.1 - b.1) + (p.2 - b.2) * (point_F.2 - b.2) = 0)

/-- The trajectory of P is an ellipse -/
theorem trajectory_of_P :
  ∀ p : ℝ × ℝ, (∃ b : ℝ × ℝ, point_P b p) → p.1^2 + (4/3) * p.2^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1017_101751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_ratio_l1017_101779

/-- Represents the cost and quantity of a rice variety -/
structure RiceVariety where
  cost : ℚ
  quantity : ℚ

/-- Calculates the total cost of a rice variety -/
def totalCost (r : RiceVariety) : ℚ := r.cost * r.quantity

/-- Calculates the cost per kg of a mixture of two rice varieties -/
def mixtureCost (r1 r2 : RiceVariety) : ℚ :=
  (totalCost r1 + totalCost r2) / (r1.quantity + r2.quantity)

theorem rice_mixture_ratio :
  let r1 : RiceVariety := { cost := 7, quantity := 5/2 }
  let r2 : RiceVariety := { cost := 35/4, quantity := 1 }
  mixtureCost r1 r2 = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_ratio_l1017_101779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_speed_with_dog_is_6_l1017_101725

-- Define the given constants
def johns_solo_speed : ℝ := 4
def total_time : ℝ := 1  -- 60 minutes = 1 hour
def time_with_dog : ℝ := 0.5  -- 30 minutes = 0.5 hours
def total_distance : ℝ := 5

-- Define John's speed with dog as a function
def johns_speed_with_dog (v : ℝ) : Prop :=
  v * time_with_dog + johns_solo_speed * (total_time - time_with_dog) = total_distance

-- Theorem statement
theorem johns_speed_with_dog_is_6 :
  johns_speed_with_dog 6 :=
by
  -- Unfold the definition of johns_speed_with_dog
  unfold johns_speed_with_dog
  -- Simplify the equation
  simp [johns_solo_speed, total_time, time_with_dog, total_distance]
  -- Check that the equation holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_speed_with_dog_is_6_l1017_101725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_illumination_range_l1017_101711

noncomputable def x (t : ℝ) : ℝ := 3 + Real.sin t * Real.cos t - Real.sin t - Real.cos t

def y (_t : ℝ) : ℝ := 1

def light_ray (c : ℝ) (x : ℝ) : ℝ := c * x

def illuminated (c : ℝ) (t : ℝ) : Prop :=
  y t = light_ray c (x t)

theorem particle_illumination_range :
  ∀ c : ℝ, c > 0 →
  (∃ t : ℝ, illuminated c t) ↔ 
  c ∈ Set.Icc ((2 * (7 - 2 * Real.sqrt 2)) / 41) (1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_illumination_range_l1017_101711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_T_l1017_101795

/-- Represents a circle in a plane -/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents the configuration of four circles tangent to a line -/
structure CircleConfiguration where
  circles : Fin 4 → Circle
  tangent_point : ℝ × ℝ
  radius_constraint : ∀ i : Fin 4, (circles i).radius ∈ ({2, 4, 6, 8} : Set ℝ)
  tangent_constraint : ∀ i : Fin 4, (circles i).center.1 = tangent_point.1

/-- The area of region T for a given configuration -/
noncomputable def area_T (config : CircleConfiguration) : ℝ :=
  sorry

/-- The theorem stating the maximum area of region T -/
theorem max_area_T :
  ∃ (config : CircleConfiguration),
    ∀ (other : CircleConfiguration), area_T config ≥ area_T other ∧ area_T config = 88 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_T_l1017_101795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l1017_101782

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

noncomputable def line_eq (x y : ℝ) (m : ℝ) : Prop := x + Real.sqrt 3 * y = 2 * m

def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y m ∧
  ∀ (x' y' : ℝ), circle_eq x' y' ∧ line_eq x' y' m → (x', y') = (x, y)

theorem tangent_line_m_value :
  ∀ m : ℝ, is_tangent m ↔ (m = -1/2 ∨ m = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l1017_101782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_angle_bisector_perpendicular_right_triangle_not_always_angle_bisector_perpendicular_l1017_101766

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₁ - x₃)^2 + (y₁ - y₃)^2

-- Define a right triangle
def IsRightTriangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∨
  (x₁ - x₂) * (x₃ - x₂) + (y₁ - y₂) * (y₃ - y₂) = 0 ∨
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = 0

-- Define the angle bisector property
def AngleBisectorPerpendicular (t : Triangle) : Prop :=
  ∃ (b : ℝ × ℝ), (b ∈ Set.Icc t.A t.B ∨ b ∈ Set.Icc t.B t.C ∨ b ∈ Set.Icc t.C t.A) ∧
    (let (x₁, y₁) := t.A
     let (x₂, y₂) := t.B
     let (x₃, y₃) := t.C
     let (xb, yb) := b
     ((xb - x₁) * (x₂ - x₃) + (yb - y₁) * (y₂ - y₃) = 0 ∨
      (xb - x₂) * (x₃ - x₁) + (yb - y₂) * (y₃ - y₁) = 0 ∨
      (xb - x₃) * (x₁ - x₂) + (yb - y₃) * (y₁ - y₂) = 0))

theorem isosceles_angle_bisector_perpendicular :
  ∀ t : Triangle, IsIsosceles t → AngleBisectorPerpendicular t := by
  sorry

theorem right_triangle_not_always_angle_bisector_perpendicular :
  ∃ t : Triangle, IsRightTriangle t ∧ ¬AngleBisectorPerpendicular t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_angle_bisector_perpendicular_right_triangle_not_always_angle_bisector_perpendicular_l1017_101766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1017_101709

noncomputable def z : ℂ := 5 / (Complex.I + 2)

theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1017_101709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1017_101758

-- Problem 1
theorem problem_1 : (1 * (-5)) - 6 + (-2) - (-9) = -4 := by sorry

-- Problem 2
theorem problem_2 : 12 * (-7) - (-4) / (2 / 37) = -10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1017_101758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_tree_max_profit_l1017_101722

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ := 64 - 48 / (x + 1) - 3 * x

-- State the theorem
theorem peach_tree_max_profit :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧
  L x = 43 ∧
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ 5 → L y ≤ L x) ∧
  x = 3 := by
  -- Proof goes here
  sorry

#check peach_tree_max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_tree_max_profit_l1017_101722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_different_sized_triangles_l1017_101776

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  side : ℝ
  side_pos : side > 0

/-- A similar triangle obtained by cutting an isosceles right triangle -/
structure SimilarTriangle where
  original : IsoscelesRightTriangle
  scale_factor : ℝ
  scale_pos : scale_factor > 0
  scale_lt_one : scale_factor < 1

/-- The set of all similar triangles obtained by cutting an isosceles right triangle -/
def cut_triangles (t : IsoscelesRightTriangle) : Set SimilarTriangle :=
  {st | st.original = t ∧ ∃ n : ℕ, st.scale_factor = (1 / Real.sqrt 2) ^ n}

/-- The theorem stating that an isosceles right triangle can be cut into infinitely many
    similar triangles of different sizes -/
theorem infinite_different_sized_triangles (t : IsoscelesRightTriangle) :
  (cut_triangles t).Infinite ∧ ∀ s₁ s₂, s₁ ∈ cut_triangles t → s₂ ∈ cut_triangles t → s₁ ≠ s₂ → s₁.scale_factor ≠ s₂.scale_factor :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_different_sized_triangles_l1017_101776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_axis_length_l1017_101706

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  a : ℝ  -- Length of the semi-major axis

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x

-- Define the directrix of the parabola
def directrix : ℝ := -1

-- Define the intersection points A and B
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

-- Function to calculate the length of the real axis
def length_of_real_axis (C : Hyperbola) : ℝ := 2 * C.a

-- Main theorem
theorem real_axis_length 
  (C : Hyperbola) 
  (intersection : IntersectionPoints) :
  C.center = (0, 0) →
  C.foci_on_x_axis = true →
  intersection.A.1 = directrix →
  intersection.B.1 = directrix →
  Real.sqrt ((intersection.A.1 - intersection.B.1)^2 + (intersection.A.2 - intersection.B.2)^2) = Real.sqrt 3 →
  length_of_real_axis C = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_axis_length_l1017_101706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_lines_l1017_101752

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the right focus of the hyperbola
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define a line passing through a point
def line_through_point (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem hyperbola_intersection_lines :
  ∃ (lines : Finset (ℝ × ℝ → ℝ × ℝ → Prop)),
    (Finset.card lines = 3) ∧
    (∀ l ∈ lines, ∃ A B : ℝ × ℝ,
      (hyperbola A.1 A.2) ∧
      (hyperbola B.1 B.2) ∧
      (∃ m : ℝ, line_through_point m right_focus A.1 A.2 ∧ line_through_point m right_focus B.1 B.2) ∧
      (distance A B = 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_lines_l1017_101752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l1017_101732

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

theorem min_value_of_f_on_interval :
  ∃ x₀ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f x₀ ≤ f x ∧ f x₀ = -4/3 := by
  -- We'll use x₀ = 2 as the minimum point
  use 2
  constructor
  · -- Prove that 2 is in the interval [0, 3]
    simp [Set.Icc]
    norm_num
  · intro x hx
    constructor
    · sorry -- Proof that f 2 ≤ f x for all x in [0, 3]
    · -- Prove that f 2 = -4/3
      unfold f
      norm_num
      

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l1017_101732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l1017_101750

theorem marble_probability (total : ℕ) (p_green : ℚ) (p_red_or_blue : ℚ) :
  total = 84 ∧ p_green = 1 / 7 ∧ p_red_or_blue = 3401 / 5600 →
  p_green + p_red_or_blue + (1 / 4 : ℚ) = 1 := by
  intro h
  sorry

#eval (3401 : ℚ) / 5600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l1017_101750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_bike_theorem_l1017_101756

def joe_bike_problem (x : ℝ) : Prop :=
  let friend_distance := x
  let store_distance := x
  let grandma_distance := x + 2
  let total_time := 1
  let helicopter_speed := 78
  
  friend_distance / 20 + store_distance / 20 + grandma_distance / 14 = total_time ∧
  let home_distance := Real.sqrt (friend_distance^2 + (store_distance + grandma_distance)^2)
  (home_distance / helicopter_speed) * 60 = 10

theorem joe_bike_theorem : ∃ x : ℝ, joe_bike_problem x := by
  sorry

#check joe_bike_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_bike_theorem_l1017_101756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_three_ways_l1017_101707

/-- Represents a block with its properties -/
structure Block where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  pattern : Fin 2
  shape : Fin 4
deriving Fintype, DecidableEq

/-- The total number of distinct blocks -/
def totalBlocks : Nat := 144

/-- The reference block (plastic medium red striped circle) -/
def referenceBlock : Block := {
  material := 0,
  size := 1,
  color := 2,
  pattern := 0,
  shape := 0
}

/-- Counts the number of differences between two blocks -/
def countDifferences (b1 b2 : Block) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.pattern ≠ b2.pattern then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The main theorem stating the number of blocks differing in exactly three ways -/
theorem blocks_differing_in_three_ways :
  (Finset.univ.filter (fun b => countDifferences b referenceBlock = 3)).card = 112 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_three_ways_l1017_101707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1017_101733

theorem right_triangle_hypotenuse (h : ℝ) : 
  let leg1 : ℝ := Real.log 125 / Real.log 8
  let leg2 : ℝ := Real.log 144 / Real.log 2
  h^2 = leg1^2 + leg2^2 →
  (8 : ℝ)^h = 5^13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1017_101733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_multiple_of_45_l1017_101705

def b : ℕ → ℕ
  | 0 => 20  -- Add this case to handle Nat.zero
  | 1 => 20  -- Add this case to handle n = 1 to 9
  | 2 => 20  -- Add cases for 2 to 9
  | 3 => 20
  | 4 => 20
  | 5 => 20
  | 6 => 20
  | 7 => 20
  | 8 => 20
  | 9 => 20
  | 10 => 20
  | n+11 => 50 * b n + 2 * (n+11)

def is_multiple_of_45 (n : ℕ) : Prop := ∃ k, b n = 45 * k

theorem least_n_multiple_of_45 :
  ∀ n, n > 10 → is_multiple_of_45 n → n ≥ 15 ∧
  is_multiple_of_45 15 :=
by
  sorry  -- Use 'by sorry' instead of just 'sorry'

#eval b 15  -- Add this line to check the value of b 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_multiple_of_45_l1017_101705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1017_101789

def sequence_a : ℕ → ℤ
  | 0 => -1  -- Adding the case for n = 0
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + 2^(n + 1) + 1

theorem sequence_a_closed_form (n : ℕ) : 
  sequence_a n = 2^n + n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1017_101789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1017_101770

/-- Represents a point on an infinite grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents the game state -/
structure GameState where
  marked_points : List GridPoint
  current_player : Nat
  turn : Nat

/-- Checks if a set of points forms a convex polygon -/
def is_convex_polygon (points : List GridPoint) : Bool :=
  sorry

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (state : GameState) (point : GridPoint) : Bool :=
  match state.turn with
  | 0 | 1 => true
  | _ => is_convex_polygon (point :: state.marked_points)

/-- Represents a player's strategy -/
def Strategy := GameState → Option GridPoint

/-- Simulates a game given two strategies -/
def simulate_game (strategy1 strategy2 : Strategy) : GameState :=
  sorry

/-- Theorem: The second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Strategy),
    ∀ (opponent_strategy : Strategy),
      let game_result := simulate_game opponent_strategy strategy
      (game_result.current_player = 0 ∧ game_result.turn % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1017_101770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1017_101703

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x)

-- State the theorem
theorem f_monotone_increasing_interval :
  ∀ a b : ℝ, (∀ x, a ≤ x ∧ x ≤ b → f x ∈ Set.range f) →
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) →
  a = 0 ∧ b = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1017_101703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQS_l1017_101778

-- Define the Point type
def Point := ℝ × ℝ

-- Define the IsTrapezoid predicate
def IsTrapezoid (P Q R S : Point) : Prop := sorry

-- Define the AreaOfTriangle function
def AreaOfTriangle (P Q S : Point) : ℝ := sorry

-- Define the trapezoid PQRS
structure Trapezoid :=
  (P Q R S : Point)
  (is_trapezoid : IsTrapezoid P Q R S)
  (area : ℝ)
  (rs_length : ℝ)
  (pq_length : ℝ)
  (rs_triple_pq : rs_length = 3 * pq_length)

-- Define the theorem
theorem area_of_triangle_PQS (pqrs : Trapezoid) (h : pqrs.area = 18) :
  AreaOfTriangle pqrs.P pqrs.Q pqrs.S = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQS_l1017_101778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_f_passes_through_fixed_point_l1017_101786

-- Define a power function
noncomputable def power_function (n : ℝ) (x : ℝ) : ℝ := x^n

-- Define the exponential function f(x) = a^(x+1) - 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 2

-- Statement 1: The graph of a power function never passes through the fourth quadrant
theorem power_function_not_in_fourth_quadrant (n : ℝ) (x : ℝ) :
  x > 0 → power_function n x > 0 := by
  sorry

-- Statement 2: For f(x) = a^(x+1) - 2 (a > 0, a ≠ 1), f(-1) = -1
theorem f_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_f_passes_through_fixed_point_l1017_101786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_characterization_angles_within_range_half_angle_quadrants_l1017_101714

open Real

noncomputable def α : ℝ := π / 3

def same_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = 2 * π * (k : ℝ) + α

def within_range (θ : ℝ) : Prop :=
  -4 * π < θ ∧ θ < 2 * π

def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ % (2 * π) ∧ θ % (2 * π) < π / 2) ∨
  (π < θ % (2 * π) ∧ θ % (2 * π) < 3 * π / 2)

theorem same_terminal_side_characterization :
  ∀ θ : ℝ, same_terminal_side θ ↔ ∃ k : ℤ, θ = 2 * π * (k : ℝ) + α := by sorry

theorem angles_within_range :
  {θ : ℝ | same_terminal_side θ ∧ within_range θ} = {-11 * π / 3, -5 * π / 3, π / 3} := by sorry

theorem half_angle_quadrants :
  ∀ β : ℝ, same_terminal_side β → in_first_or_third_quadrant (β / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_characterization_angles_within_range_half_angle_quadrants_l1017_101714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l1017_101754

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (4 * (x - 2) / 3) - 5

-- State the theorem
theorem h_fixed_point : ∃ (x : ℝ), h x = x ∧ x = 23 := by
  -- Use x = 23 as the witness
  use 23
  constructor
  · -- Prove h 23 = 23
    unfold h
    norm_num
  · -- Prove 23 = 23
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l1017_101754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_proof_l1017_101723

/-- The surface area of a cylinder with radius r and height h -/
noncomputable def cylinder_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_height_proof (r SA : ℝ) (h_r : r = 3) (h_SA : SA = 36 * Real.pi) :
  ∃ h : ℝ, h = 3 ∧ cylinder_surface_area r h = SA := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_proof_l1017_101723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_meal_cost_is_six_l1017_101702

/-- The cost of a burger meal, given upsizing and total cost information -/
def burger_meal_cost (upsized_cost : ℚ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  let regular_cost := upsized_cost - 1
  let daily_cost := upsized_cost
  let calculated_total := daily_cost * days
  if upsized_cost > 0 ∧ days = 5 ∧ total_cost = 35 ∧ calculated_total = total_cost
  then regular_cost
  else 0

#eval burger_meal_cost 7 5 35  -- Should evaluate to 6

theorem burger_meal_cost_is_six :
  burger_meal_cost 7 5 35 = 6 := by
  -- Unfold the definition of burger_meal_cost
  unfold burger_meal_cost
  -- Simplify the if-then-else expression
  simp
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_meal_cost_is_six_l1017_101702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1017_101759

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2

-- State the theorem
theorem unique_a_value (a : ℝ) (h1 : a > 0) (h2 : f a (f a (Real.sqrt 2)) = -2) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1017_101759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_factorial_sum_l1017_101713

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem greatest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), is_prime_factor p (factorial 15 + factorial 18) ∧
    (∀ (q : ℕ), is_prime_factor q (factorial 15 + factorial 18) → q ≤ p) ∧
    p = 17 := by
  sorry

#eval factorial 5  -- This line is added to test the factorial function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_factorial_sum_l1017_101713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l1017_101749

/-- The circle C in the Cartesian plane -/
noncomputable def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

/-- The line l passing through P(0, √3) with slope -1 -/
noncomputable def lineL (x y : ℝ) : Prop := y = Real.sqrt 3 - x

/-- The point P -/
noncomputable def P : ℝ × ℝ := (0, Real.sqrt 3)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_ratio : 
  ∃ (A B : ℝ × ℝ), 
    circleC A.1 A.2 ∧ circleC B.1 B.2 ∧ 
    lineL A.1 A.2 ∧ lineL B.1 B.2 ∧ 
    A ≠ B ∧
    (distance P A / distance P B + distance P B / distance P A = 8/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l1017_101749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1017_101726

/-- The function f(x) defined in the problem -/
noncomputable def f (k : ℤ) (x : ℝ) : ℝ := x^(-k^2 + k + 2)

/-- The function g(x) defined in the problem -/
noncomputable def g (p : ℝ) (k : ℤ) (x : ℝ) : ℝ := 1 - p * f k x + (2*p - 1) * x

/-- The theorem statement -/
theorem problem_solution (k : ℤ) (h : f k 2 < f k 3) :
  (k = 0 ∨ k = 1) ∧
  ∃! p : ℝ, p > 0 ∧ 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, g p k x ∈ Set.Icc (-4 : ℝ) (17/8)) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 2, g p k x = -4) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 2, g p k x = 17/8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1017_101726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_c_and_d_l1017_101738

/-- The time taken for two workers to complete a job together -/
noncomputable def time_to_complete (rate1 rate2 : ℝ) : ℝ := 1 / (rate1 + rate2)

/-- The work rate of a worker -/
noncomputable def work_rate (time : ℝ) : ℝ := 1 / time

theorem work_completion_time_c_and_d 
  (rate_a rate_b rate_c : ℝ)
  (hab : time_to_complete rate_a rate_b = 10)
  (hbc : time_to_complete rate_b rate_c = 15)
  (hca : time_to_complete rate_c rate_a = 12)
  (hd_with_c : ∀ rate_d, time_to_complete rate_c rate_d = time_to_complete rate_c (2 * rate_c))
  (hd_with_a : ∀ rate_d, time_to_complete rate_a rate_d = time_to_complete rate_a (rate_a / 2)) :
  time_to_complete rate_c (2 * rate_c) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_c_and_d_l1017_101738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l1017_101793

/-- Liam's current age -/
def L : ℕ := sorry

/-- Mia's current age -/
def M : ℕ := sorry

/-- The conditions given in the problem -/
axiom condition1 : L - 4 = 2 * (M - 4)
axiom condition2 : L - 10 = 3 * (M - 10)

/-- The theorem to prove -/
theorem age_ratio_years : ∃ x : ℕ, x = 8 ∧ (L + x : ℚ) / (M + x) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l1017_101793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_pages_theorem_l1017_101701

/-- Represents a person's reading habits --/
structure ReadingHabits where
  initial_hours : ℚ
  increase_percentage : ℚ
  current_pages_per_week : ℚ

/-- Calculates the initial pages read per day --/
noncomputable def initial_pages_per_day (habits : ReadingHabits) : ℚ :=
  let current_hours := habits.initial_hours * (1 + habits.increase_percentage / 100)
  let pages_per_hour := habits.current_pages_per_week / (7 * current_hours)
  habits.initial_hours * pages_per_hour

/-- Theorem stating that given the conditions, the initial pages read per day was 100 --/
theorem initial_pages_theorem (habits : ReadingHabits) 
  (h1 : habits.initial_hours = 2)
  (h2 : habits.increase_percentage = 150)
  (h3 : habits.current_pages_per_week = 1750) :
  initial_pages_per_day habits = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_pages_theorem_l1017_101701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tetrahedron_volume_is_512_div_3_l1017_101731

/-- A cube with side length 8 and alternately colored vertices (black and green) -/
structure ColoredCube where
  sideLength : ℝ
  sideLength_eq : sideLength = 8

/-- The volume of the tetrahedron formed by the green vertices of the colored cube -/
noncomputable def greenTetrahedronVolume (c : ColoredCube) : ℝ :=
  512 / 3

/-- Theorem stating that the volume of the tetrahedron formed by the green vertices is 512/3 -/
theorem green_tetrahedron_volume_is_512_div_3 (c : ColoredCube) :
  greenTetrahedronVolume c = 512 / 3 := by
  -- Unfold the definition of greenTetrahedronVolume
  unfold greenTetrahedronVolume
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tetrahedron_volume_is_512_div_3_l1017_101731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_length_is_nine_l1017_101717

/-- A triangle with vertices A, B, and C -/
structure Triangle (α : Type*) where
  A : α
  B : α
  C : α

/-- The perimeter of a triangle -/
def perimeter {α : Type*} (t : Triangle α) (length : α → α → ℝ) : ℝ :=
  length t.A t.B + length t.B t.C + length t.C t.A

/-- A triangle is isosceles if two of its sides are equal -/
def isIsosceles {α : Type*} (t : Triangle α) (length : α → α → ℝ) : Prop :=
  length t.A t.B = length t.A t.C ∨ length t.A t.B = length t.B t.C ∨ length t.A t.C = length t.B t.C

theorem ab_length_is_nine {α : Type*} 
  (ABC : Triangle α) (CBD : Triangle α) (length : α → α → ℝ) (D : α) :
  isIsosceles ABC length →
  isIsosceles CBD length →
  perimeter CBD length = 24 →
  perimeter ABC length = 23 →
  length CBD.B D = 10 →
  length ABC.A ABC.B = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_length_is_nine_l1017_101717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_favorite_number_l1017_101784

def is_product_of_three_distinct_primes (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r

def satisfies_claires_condition (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ ∀ a b c d, a ∈ s → b ∈ s → c ∈ s → d ∈ s → a + b - c = d

theorem cats_favorite_number :
  ∃! n : ℕ,
    n ≥ 10 ∧ n < 100 ∧
    is_product_of_three_distinct_primes n ∧
    (∃ s : Finset ℕ, satisfies_claires_condition s ∧
      (∀ m ∈ s, m ≥ 10 ∧ m < 100 ∧ is_product_of_three_distinct_primes m) ∧
      n ∉ s) ∧
    n = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_favorite_number_l1017_101784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l1017_101797

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Final amount calculation function -/
noncomputable def finalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simpleInterest principal rate time

theorem initial_amount_proof (rate : ℝ) (time : ℝ) (final : ℝ) 
  (h_rate : rate = 12)
  (h_time : time = 5)
  (h_final : final = 1200)
  : finalAmount 750 rate time = final :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l1017_101797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_sum_l1017_101771

theorem unique_factorial_sum (n : ℕ+) : 
  Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 1320 ↔ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_sum_l1017_101771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l1017_101763

theorem sin_2_cos_3_tan_4_negative :
  ∀ (two three four : ℝ),
  (π / 2 < two) ∧ (two < three) ∧ (three < π) ∧ (π < four) ∧ (four < 3 * π / 2) →
  Real.sin two * Real.cos three * Real.tan four < 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l1017_101763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_b_in_range_l1017_101700

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3/2) * x + b - 1
  else -x^2 + (2 - b) * x

-- State the theorem
theorem f_increasing_iff_b_in_range (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x < f b y) ↔ (3/2 < b ∧ b ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_b_in_range_l1017_101700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l1017_101740

/-- Define the operation ∘ for real numbers -/
def circ (x y : ℝ) : ℝ := 3*x - 2*y + 2*x*y

/-- Theorem statement -/
theorem unique_solution_exists :
  ∃! y : ℝ, circ 4 y = 20 ∧ y = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l1017_101740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l1017_101760

/-- The circle equation: x^2 - 10x + y^2 - 6y + 40 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 6*y + 40 = 0

/-- The shortest distance from the origin to the circle -/
noncomputable def shortest_distance : ℝ := Real.sqrt 34 - 2 * Real.sqrt 7

theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  Real.sqrt (x^2 + y^2) ≥ shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l1017_101760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1017_101704

-- Define the sets
def positive_fractions : Set ℚ := {x | 0 < x ∧ ∃ (a b : ℤ), x = a / b ∧ b ≠ 0}
def integers : Set ℝ := {x | ∃ (n : ℤ), x = n}
def irrationals : Set ℝ := {x | ¬∃ (a b : ℤ), x = a / b ∧ b ≠ 0}

-- Define the given numbers
noncomputable def given_numbers : List ℝ := [-1/10, (8:ℝ)^(1/3), 0.3, -Real.pi/3, -8, 0, (0.9:ℝ)^(1/2), 22/7]

-- Theorem statement
theorem number_categorization :
  (0.3 ∈ positive_fractions) ∧
  (22/7 ∈ positive_fractions) ∧
  ((8:ℝ)^(1/3) ∈ integers) ∧
  (-8 ∈ integers) ∧
  (0 ∈ integers) ∧
  (-Real.pi/3 ∈ irrationals) ∧
  ((0.9:ℝ)^(1/2) ∈ irrationals) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1017_101704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x0_range_l1017_101792

/-- Given a function f(x) = a*sin(x) + b*cos(x) where x ∈ ℝ and b/a ∈ (1, √3],
    if f(x) attains its maximum value at x = x₀, then the range of tan(2x₀) is [√3, +∞) -/
theorem tan_2x0_range (a b : ℝ) (h1 : a ≠ 0) (h2 : 1 < b/a ∧ b/a ≤ Real.sqrt 3) :
  let f := λ x : ℝ ↦ a * Real.sin x + b * Real.cos x
  ∃ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) →
    {y : ℝ | ∃ k : ℤ, y = Real.tan (2 * (x₀ + k * Real.pi))} = Set.Ici (Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x0_range_l1017_101792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1017_101768

/-- The inclination angle of a line given its equation -/
noncomputable def inclination_angle (a b : ℝ) : ℝ := Real.arctan (b / a)

/-- Theorem: The inclination angle of the line x - √3y - 2 = 0 is π/6 -/
theorem line_inclination_angle :
  inclination_angle 1 (-Real.sqrt 3) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1017_101768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_parallel_vectors_l1017_101728

-- Define the vectors
def a : Fin 2 → ℝ := ![4, 3]
def b : Fin 2 → ℝ := ![-1, 2]

-- Theorem for the cosine of the angle between vectors
theorem cosine_of_angle :
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  dot_product / (magnitude_a * magnitude_b) = 2 * Real.sqrt 5 / 25 := by
  sorry

-- Theorem for the value of λ
theorem parallel_vectors :
  ∃ k : ℝ, k ≠ 0 ∧ 
    (a 0 - (-1/2) * b 0 = k * (2 * a 0 + b 0)) ∧
    (a 1 - (-1/2) * b 1 = k * (2 * a 1 + b 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_parallel_vectors_l1017_101728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l1017_101774

theorem graph_shift (x : ℝ) : (3 : ℝ)^(x+1) = (3 : ℝ)^(x-(-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l1017_101774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kims_sweater_difference_l1017_101716

/-- Represents the number of sweaters knit on each day of the week --/
structure WeeklySweaters where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Represents the conditions of Kim's sweater knitting for the week --/
def KimsSweaters (w t : ℕ) : WeeklySweaters where
  monday := 8
  tuesday := 10
  wednesday := w
  thursday := t
  friday := 4

theorem kims_sweater_difference :
  ∀ w t : ℕ,
  -- Kim can knit up to 10 sweaters in a day
  (∀ d : ℕ, d ≤ 10) →
  -- On Tuesday, she knit 2 more sweaters than on Monday
  (KimsSweaters w t).tuesday = (KimsSweaters w t).monday + 2 →
  -- On Wednesday and Thursday, she knit fewer sweaters than on Tuesday
  w < (KimsSweaters w t).tuesday ∧ t < (KimsSweaters w t).tuesday →
  -- On Friday, she knit half the number of sweaters she had knit on Monday
  (KimsSweaters w t).friday * 2 = (KimsSweaters w t).monday →
  -- She knit a total of 34 sweaters that week
  (KimsSweaters w t).monday + (KimsSweaters w t).tuesday + w + t + (KimsSweaters w t).friday = 34 →
  -- The absolute difference between Tuesday and (Wednesday + Thursday) is 2
  |((KimsSweaters w t).tuesday : ℤ) - (w + t : ℤ)| = 2 := by
  sorry

#check kims_sweater_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kims_sweater_difference_l1017_101716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1017_101710

/-- The line l: 3x + y - 6 = 0 -/
def line (x y : ℝ) : Prop := 3 * x + y - 6 = 0

/-- The circle C: x^2 + y^2 - 2y - 4 = 0 -/
def circle' (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 4 = 0

/-- Two points are on the line -/
def on_line (A B : ℝ × ℝ) : Prop := line A.1 A.2 ∧ line B.1 B.2

/-- Two points are on the circle -/
def on_circle (A B : ℝ × ℝ) : Prop := circle' A.1 A.2 ∧ circle' B.1 B.2

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Theorem: The distance between the intersection points of the given line and circle is √10 -/
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, on_line A B → on_circle A B → distance A B = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1017_101710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1017_101730

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line
def line_eq (x y : ℝ) : Prop := x - y = Real.sqrt 2

-- Theorem statement
theorem chord_length :
  ∃ (a b c d : ℝ),
    circle_eq a b ∧ circle_eq c d ∧
    line_eq a b ∧ line_eq c d ∧
    ((a - c)^2 + (b - d)^2) = 32 := by
  sorry

#check chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1017_101730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foot_of_perpendicular_location_l1017_101720

-- Define a line segment
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a perpendicular line to a line segment
structure PerpendicularLine where
  segment : LineSegment
  point : Point

-- Define the foot of a perpendicular line
noncomputable def FootOfPerpendicular (perp : PerpendicularLine) : Point :=
  sorry

-- Theorem stating that the foot of the perpendicular can be in any of the three locations
theorem foot_of_perpendicular_location (segment : LineSegment) :
  ∃ (perp : PerpendicularLine),
    (FootOfPerpendicular perp ∈ Set.Icc segment.start segment.endpoint) ∨
    (FootOfPerpendicular perp = segment.start ∨ FootOfPerpendicular perp = segment.endpoint) ∨
    (FootOfPerpendicular perp ∉ Set.Icc segment.start segment.endpoint) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foot_of_perpendicular_location_l1017_101720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1017_101724

open Real Set

-- Define the equation
def equation (ω : ℝ) (x : ℝ) : Prop := sin (ω * x) + 1 = 0

-- Define the interval
def interval : Set ℝ := Ioo 0 (π / 2)

-- Define the condition of exactly one solution
def has_unique_solution (ω : ℝ) : Prop :=
  ∃! x, x ∈ interval ∧ equation ω x

-- Define the maximum value of ω
def max_omega : ℝ := 7

-- State the theorem
theorem max_omega_value :
  ∀ ω : ℝ, ω > 0 → has_unique_solution ω → ω ≤ max_omega :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1017_101724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_ideal_point_l1017_101745

/-- Represents a house in a 2D plane -/
structure House where
  x : ℝ
  y : ℝ

/-- Calculates the distance between a point and a house -/
noncomputable def distance (px py : ℝ) (h : House) : ℝ :=
  Real.sqrt ((px - h.x)^2 + (py - h.y)^2)

/-- Calculates the sum of distances from a point to all houses -/
noncomputable def sumDistances (px py : ℝ) (houses : List House) : ℝ :=
  houses.foldr (λ h acc => acc + distance px py h) 0

/-- Checks if all houses are collinear -/
def areCollinear (houses : List House) : Prop :=
  ∃ a b c : ℝ, ∀ h ∈ houses, a * h.x + b * h.y + c = 0

/-- Theorem: There exists at most one ideal point for resource generation -/
theorem at_most_one_ideal_point (n : ℕ) (houses : List House)
    (h1 : n > 2)
    (h2 : houses.length = n)
    (h3 : ¬ areCollinear houses) :
    ∃! p : ℝ × ℝ, ∀ q : ℝ × ℝ, sumDistances p.1 p.2 houses ≤ sumDistances q.1 q.2 houses := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_ideal_point_l1017_101745
