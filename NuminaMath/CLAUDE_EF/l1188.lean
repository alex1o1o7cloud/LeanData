import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_order_l1188_118836

/-- Represents a point on a parabola -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 - 2x + b -/
def Parabola (b : ℝ) : ℝ → ℝ := λ x ↦ x^2 - 2*x + b

theorem parabola_point_order (b : ℝ) (A B C : Point)
  (hA : A.y = Parabola b A.x) (hB : B.y = Parabola b B.x) (hC : C.y = Parabola b C.x)
  (hAx : A.x = -1) (hBx : B.x = 2) (hCx : C.x = 4) :
  C.y > A.y ∧ A.y > B.y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_order_l1188_118836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_exists_l1188_118838

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties of triangles
def IsIsoscelesRight (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 ∧
  (t.B.x - t.C.x) * (t.A.x - t.B.x) + (t.B.y - t.C.y) * (t.A.y - t.B.y) = 0

-- Define rotation
noncomputable def Rotate (p : Point) (θ : ℝ) : Point :=
  { x := p.x * Real.cos θ - p.y * Real.sin θ
  , y := p.x * Real.sin θ + p.y * Real.cos θ }

-- Theorem statement
theorem isosceles_right_triangle_exists 
  (ABC ADE : Triangle) 
  (hABC : IsIsoscelesRight ABC) 
  (hADE : IsIsoscelesRight ADE) 
  (hNonCong : ABC ≠ ADE) 
  (hRightA : ABC.A = ADE.A) : 
  ∀ θ : ℝ, ∃ M : Point, 
    (∃ t : ℝ, M = { x := t * (Rotate ADE.C θ).x + (1 - t) * ABC.C.x
                  , y := t * (Rotate ADE.C θ).y + (1 - t) * ABC.C.y }) ∧
    IsIsoscelesRight { A := ABC.B, B := M, C := Rotate ADE.B θ } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_exists_l1188_118838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_function_b_range_l1188_118875

/-- A function f is a storm function on an interval if |f(x₁) - f(x₂)| < 1 for any x₁, x₂ in the interval. -/
def IsStormFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Set.Icc a b → x₂ ∈ Set.Icc a b → |f x₁ - f x₂| < 1

/-- The quadratic function we're considering -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - b * x + 1

theorem storm_function_b_range :
  ∀ b : ℝ, IsStormFunction (f b) (-1) 1 ↔ b ∈ Set.Ioo (1 - Real.sqrt 2) (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_function_b_range_l1188_118875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_b_zero_l1188_118879

/-- A cubic function with roots at -2, 0, and 2, and f(-1) = 3 has b = 0 -/
theorem cubic_function_b_zero (a b c d : ℝ) : 
  (∀ x, (a * x^3 + b * x^2 + c * x + d = 0) ↔ (x = -2 ∨ x = 0 ∨ x = 2)) →
  (a * (-1)^3 + b * (-1)^2 + c * (-1) + d = 3) →
  b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_b_zero_l1188_118879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_difference_l1188_118849

theorem calculation_difference (harry_calc terry_calc : ℤ) : 
  let correct := 12 - (4 * 3)
  let incorrect := (12 - 4) * 3
  harry_calc = correct ∧ terry_calc = incorrect →
  harry_calc - terry_calc = -24 := by
  intros h
  simp [h]
  norm_num
  sorry

#check calculation_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_difference_l1188_118849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l1188_118876

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 1
  else if 2 < x ∧ x ≤ 3 then (1/2) * x^2 - 1
  else 0  -- We define f as 0 outside [1, 3] for completeness

-- Define h(a)
noncomputable def h (a : ℝ) : ℝ :=
  (⨆ x ∈ Set.Icc 1 3, f x - a * x) - (⨅ x ∈ Set.Icc 1 3, f x - a * x)

-- State the theorem
theorem h_properties :
  h 0 = 5/2 ∧
  ∃ (a_min : ℝ), a_min = 5/4 ∧ 
    (∀ a, h a_min ≤ h a) ∧
    h a_min = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l1188_118876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_difference_magnitude_l1188_118823

/-- Two parallel planar vectors have a specific difference magnitude -/
theorem parallel_vectors_difference_magnitude 
  (x : ℝ) 
  (a b : ℝ × ℝ) 
  (h1 : a = (1, x)) 
  (h2 : b = (2*x + 3, -x)) 
  (h3 : ∃ (k : ℝ), a = k • b) : 
  ‖a - b‖ = 2 ∨ ‖a - b‖ = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_difference_magnitude_l1188_118823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l1188_118847

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotation of a point by 90° counter-clockwise about the origin -/
def rotate90 (p : Point) : Point :=
  { x := -p.y, y := p.x }

/-- The original function y = log₂ x -/
noncomputable def originalFunc (x : ℝ) : ℝ :=
  Real.log x / Real.log 2

/-- The resulting function after rotation -/
noncomputable def rotatedFunc (x : ℝ) : ℝ :=
  2^(-x)

theorem rotation_result :
  ∀ (p : Point),
    p.y = originalFunc p.x →
    (rotate90 p).y = rotatedFunc (rotate90 p).x := by
  sorry

#check rotation_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l1188_118847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_filling_with_tetrahedra_and_octahedra_l1188_118845

-- Define a regular tetrahedron
structure RegularTetrahedron where
  edgeLength : ℝ
  height : ℝ
  heightFormula : height = Real.sqrt (2/3) * edgeLength

-- Define a regular octahedron
structure RegularOctahedron where
  edgeLength : ℝ

-- Define a space-filling configuration
structure SpaceFillingConfiguration where
  tetrahedra : Set RegularTetrahedron
  octahedra : Set RegularOctahedron

-- Helper functions (not implemented, just declared for the theorem statement)
def point_in_tetrahedron : (ℝ × ℝ × ℝ) → RegularTetrahedron → Prop := sorry
def point_in_octahedron : (ℝ × ℝ × ℝ) → RegularOctahedron → Prop := sorry
def tetrahedra_overlap : RegularTetrahedron → RegularTetrahedron → Prop := sorry
def octahedra_overlap : RegularOctahedron → RegularOctahedron → Prop := sorry
def tetrahedron_octahedron_overlap : RegularTetrahedron → RegularOctahedron → Prop := sorry

-- Theorem statement
theorem space_filling_with_tetrahedra_and_octahedra :
  ∃ (config : SpaceFillingConfiguration), 
    (∀ point : ℝ × ℝ × ℝ, 
      (∃ t ∈ config.tetrahedra, point_in_tetrahedron point t) ∨ 
      (∃ o ∈ config.octahedra, point_in_octahedron point o)) ∧
    (∀ t1 t2, t1 ∈ config.tetrahedra → t2 ∈ config.tetrahedra → t1 ≠ t2 → ¬ tetrahedra_overlap t1 t2) ∧
    (∀ o1 o2, o1 ∈ config.octahedra → o2 ∈ config.octahedra → o1 ≠ o2 → ¬ octahedra_overlap o1 o2) ∧
    (∀ t o, t ∈ config.tetrahedra → o ∈ config.octahedra → ¬ tetrahedron_octahedron_overlap t o) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_filling_with_tetrahedra_and_octahedra_l1188_118845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2010_equals_7948_l1188_118824

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ := sorry

/-- The number of terms in each group -/
def group_size (n : ℕ) : ℕ := n

/-- The cumulative sum of terms up to and including group n -/
def cumulative_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last term in group n -/
def last_term_in_group (n : ℕ) : ℕ := n * (4 * n + 1) / 2

/-- Theorem stating that the 2010th term of the sequence is 7948 -/
theorem v_2010_equals_7948 : v 2010 = 7948 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2010_equals_7948_l1188_118824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_prime_divisibility_l1188_118825

/-- Lucas sequence -/
def L : ℕ → ℕ
  | 0 => 2  -- Adding this case to handle the missing Nat.zero case
  | 1 => 1
  | 2 => 3
  | n + 3 => L (n + 2) + L (n + 1)

theorem lucas_prime_divisibility (p : ℕ) (hp : Nat.Prime p) : p ∣ (L p - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_prime_divisibility_l1188_118825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mica_sauce_jars_l1188_118867

/-- Represents the grocery shopping scenario --/
structure GroceryShopping where
  pasta_price : ℚ
  pasta_quantity : ℚ
  beef_price : ℚ
  beef_quantity : ℚ
  sauce_price : ℚ
  quesadilla_price : ℚ
  total_budget : ℚ

/-- Calculates the number of pasta sauce jars Mica can buy --/
def sauce_jars (shopping : GroceryShopping) : ℚ :=
  let pasta_cost := shopping.pasta_price * shopping.pasta_quantity
  let beef_cost := shopping.beef_price * shopping.beef_quantity
  let remaining_budget := shopping.total_budget - (pasta_cost + beef_cost + shopping.quesadilla_price)
  remaining_budget / shopping.sauce_price

/-- Theorem stating that Mica can buy 2 jars of pasta sauce --/
theorem mica_sauce_jars :
  let shopping : GroceryShopping := {
    pasta_price := 3/2,
    pasta_quantity := 2,
    beef_price := 8,
    beef_quantity := 1/4,
    sauce_price := 2,
    quesadilla_price := 6,
    total_budget := 15
  }
  sauce_jars shopping = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mica_sauce_jars_l1188_118867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_arts_arrangement_count_l1188_118808

theorem six_arts_arrangement_count :
  let n : ℕ := 6  -- Total number of arts
  let fixed_position : ℕ := 3  -- Position of "数" (mathematics)
  let adjacent_pair : ℕ := 2  -- Number of arts that must be adjacent ("射" and "御")
  
  (n - adjacent_pair + 1 - 1) *  -- Ways to place the adjacent pair
  (Nat.factorial adjacent_pair) *  -- Ways to arrange the adjacent pair internally
  (Nat.factorial (n - adjacent_pair - 1)) -- Ways to arrange the remaining elements
  = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_arts_arrangement_count_l1188_118808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identity_l1188_118878

theorem angle_identity (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α = 4/5) : 
  Real.cos (2*α + π/4) = 17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identity_l1188_118878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_4_l1188_118813

theorem cos_2alpha_plus_pi_4 (α : ℝ) 
  (h1 : Real.cos (α + Real.pi/4) = 3/5) 
  (h2 : Real.pi/2 < α) 
  (h3 : α < 3*Real.pi/2) : 
  Real.cos (2*α + Real.pi/4) = -31*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_4_l1188_118813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1188_118897

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- Define angles A and B
variable (A B : ℝ)

-- Theorem statement
theorem problem_statement 
  (hf' : ∀ x, HasDerivAt f (f' x) x)
  (hg' : ∀ x, HasDerivAt g (g' x) x)
  (hg_pos : ∀ x, g x > 0)
  (h_deriv : ∀ x, f' x * g x - f x * g' x > 0)
  (h_obtuse : π/2 < π - A - B)
  (h_triangle : 0 < A ∧ 0 < B ∧ A + B < π)
  : f (Real.cos A) * g (Real.sin B) > f (Real.sin B) * g (Real.cos A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1188_118897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_101_l1188_118827

/-- The sum of the alternating series from 1 to n -/
def alternating_sum (n : ℕ) : ℤ :=
  List.range n |> List.map (fun i => if i % 2 = 0 then (i + 1) else -(i + 1 : ℤ)) |> List.sum

/-- The sum of the series 1 - 2 + 3 - 4 + ... - 100 + 101 equals 51 -/
theorem alternating_sum_to_101 : alternating_sum 101 = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_101_l1188_118827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_polygon_comparison_l1188_118817

/-- Represents a cyclic polygon with ordered sides -/
structure CyclicPolygon (n : ℕ) where
  sides : Fin n → ℝ
  ordered : ∀ i j : Fin n, i ≤ j → sides i ≥ sides j

/-- Calculates the perimeter of a cyclic polygon -/
def perimeter {n : ℕ} (p : CyclicPolygon n) : ℝ :=
  (Finset.univ.sum fun i => p.sides i)

/-- Calculates the area of a cyclic polygon -/
noncomputable def area {n : ℕ} (p : CyclicPolygon n) : ℝ :=
  sorry

/-- Theorem: If the sides of one cyclic polygon are all smaller than the corresponding sides of another cyclic polygon,
    then both the perimeter and area of the first polygon are greater than those of the second polygon -/
theorem cyclic_polygon_comparison {k l : ℕ} (p₁ : CyclicPolygon k) (p₂ : CyclicPolygon l)
    (h : ∀ i : Fin (min k l), ∃ j₁ : Fin k, ∃ j₂ : Fin l, p₁.sides j₁ < p₂.sides j₂) :
    perimeter p₁ > perimeter p₂ ∧ area p₁ > area p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_polygon_comparison_l1188_118817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l1188_118822

theorem recurring_decimal_to_fraction : 
  (7/10 : ℚ) + (23/99 : ℚ) = 923/990 := by
  sorry

#eval (923 : ℚ) / 990

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l1188_118822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_proof_l1188_118894

-- Define the original function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (5 * x) + Real.sqrt x + 3/5

-- Define the proposed antiderivative F(x)
noncomputable def F (x : ℝ) : ℝ := -2/5 * Real.cos (5 * x) + 2/3 * x * Real.sqrt x + 3/5 * x + 1

-- State the theorem
theorem antiderivative_proof :
  (∀ x, deriv F x = f x) ∧ 
  (F 0 = f 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_proof_l1188_118894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_shapes_l1188_118826

/-- Given a right triangle ABC with legs AC = a and CB = b, prove the following:
    1. The side length of the largest square within the triangle is ab / (a + b)
    2. The dimensions of the largest rectangle within the triangle are a/2 and b/2 -/
theorem largest_inscribed_shapes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let triangle := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 / a + p.2 / b ≤ 1}
  ∃ (s : ℝ), s = a * b / (a + b) ∧ 
    (∀ (x : ℝ), x > 0 → {p : ℝ × ℝ | p.1 ∈ Set.Icc 0 x ∧ p.2 ∈ Set.Icc 0 x} ⊆ triangle → x ≤ s) ∧
  ∃ (l w : ℝ), l = a / 2 ∧ w = b / 2 ∧
    (∀ (x y : ℝ), x > 0 → y > 0 → {p : ℝ × ℝ | p.1 ∈ Set.Icc 0 x ∧ p.2 ∈ Set.Icc 0 y} ⊆ triangle → x * y ≤ l * w) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_shapes_l1188_118826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_permutations_with_repetition_l1188_118840

def digits : List Nat := [1, 3, 3, 3, 5]

def is_valid_permutation (perm : List Nat) : Bool :=
  perm.length = 5 && perm.head! ≠ 0 && perm.toFinset = digits.toFinset

def count_valid_permutations : Nat :=
  (List.permutations digits).filter is_valid_permutation |>.length

theorem count_permutations_with_repetition :
  count_valid_permutations = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_permutations_with_repetition_l1188_118840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_compensation_l1188_118889

/-- Calculates the total compensation for a bus driver given the following parameters:
  * regular_rate: The hourly rate for regular hours
  * regular_hours: The number of regular hours worked
  * overtime_rate_increase: The percentage increase for overtime hours
  * total_hours: The total number of hours worked
-/
noncomputable def calculate_compensation (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate_increase : ℚ) (total_hours : ℚ) : ℚ :=
  let overtime_hours := max (total_hours - regular_hours) 0
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Proves that the bus driver's total compensation for the week is $1340 -/
theorem bus_driver_compensation :
  calculate_compensation 16 40 (3/4) 65 = 1340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_compensation_l1188_118889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_equal_eccentricities_l1188_118893

-- Define the hyperbolas C₁ and C₂
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1
def C₂ (a : ℝ) (x y : ℝ) : Prop := y^2 / a^2 - x^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a : ℝ) : ℝ := (Real.sqrt (a^2 + 1)) / a

-- Theorem statement
theorem hyperbolas_equal_eccentricities (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ a x₁ y₁ → C₂ a x₂ y₂ → eccentricity a = eccentricity a :=
by
  intros x₁ y₁ x₂ y₂ _ _
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_equal_eccentricities_l1188_118893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_constants_for_sin_square_identity_l1188_118888

theorem no_constants_for_sin_square_identity :
  ¬ ∃ (c d : ℝ), ∀ θ : ℝ, (Real.sin θ)^2 = c * Real.sin (2 * θ) + d * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_constants_for_sin_square_identity_l1188_118888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l1188_118804

/-- The volume of a frustum formed by intersecting a cone with a plane parallel to its base -/
theorem frustum_volume (r₁ r₂ h₂ : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 2) (hh₂ : h₂ = 3) :
  let h₁ : ℝ := r₁ * h₂ / r₂
  (1/3 : ℝ) * Real.pi * (r₁^2 * h₁ - r₂^2 * h₂) = 28 * Real.pi :=
by
  -- Introduce the local definition of h₁
  let h₁ : ℝ := r₁ * h₂ / r₂
  
  -- Replace r₁, r₂, and h₂ with their given values
  rw [hr₁, hr₂, hh₂]
  
  -- Simplify the expression
  norm_num
  
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l1188_118804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_triples_l1188_118833

theorem prime_power_triples (p : ℕ) (x y : ℕ) :
  (Nat.Prime p ∧ 
   ∃ k l : ℕ, x^(p-1) + y = p^k ∧ x + y^(p-1) = p^l) ↔ 
  ((p = 3 ∧ ((x = 5 ∧ y = 2) ∨ (x = 2 ∧ y = 5))) ∨
   (p = 2 ∧ ∃ (α k : ℕ), (0 < α ∧ α < 2^k ∧ 
    ((x = α ∧ y = 2^k - α) ∨ (x = 2^k - α ∧ y = α))))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_triples_l1188_118833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_tangent_point_exists_constant_ratio_l1188_118891

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the foci
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the parameters as variables
variable (a b c : ℝ)

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom equilateral_triangle : ∃ (x y : ℝ), 
  (x^2 + y^2 = 16) ∧ 
  ((x+2)^2 + y^2 = 16) ∧ 
  ((x-2)^2 + y^2 = 16)

-- Theorem statements
theorem foci_distance : c = 2 ∧ a^2 - b^2 = 4 := by sorry

theorem tangent_point_exists : ∀ (P : ℝ × ℝ), ellipse a b P.1 P.2 → 
  ∃ (M : ℝ × ℝ), (M.1 - (-2))*(P.2 - M.2) = (P.1 - M.1)*(0 - M.2) := by sorry

theorem constant_ratio : ∃ (k : ℝ), ∀ (P M N : ℝ × ℝ),
  ellipse a b P.1 P.2 →
  (M.1 - (-2))*(P.2 - M.2) = (P.1 - M.1)*(0 - M.2) →
  N.1 = -2 →
  (N.1 - (-2))*(P.2 - N.2) = (P.1 - N.1)*(M.2 - N.2) →
  (N.1 - (-2))^2 + N.2^2 / ((M.1 - (-2))^2 + M.2^2) = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_tangent_point_exists_constant_ratio_l1188_118891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cube_diff_divisible_10_exists_cube_diff_divisible_8_l1188_118807

-- Define a function to check if the difference of cubes is divisible by 27
def cube_diff_divisible_by_27 (a b : ℤ) : Prop :=
  (a^3 - b^3) % 27 = 0

-- Theorem for 10 integers
theorem exists_cube_diff_divisible_10 (S : Finset ℤ) (h : S.card = 10) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ cube_diff_divisible_by_27 a b :=
sorry

-- Theorem for 8 integers
theorem exists_cube_diff_divisible_8 (S : Finset ℤ) (h : S.card = 8) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ cube_diff_divisible_by_27 a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cube_diff_divisible_10_exists_cube_diff_divisible_8_l1188_118807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_l1188_118805

-- Define the angle in degrees
noncomputable def angle : ℝ := 410

-- Define a function to determine the quadrant of an angle
noncomputable def quadrant (θ : ℝ) : ℕ :=
  let normalized_angle := θ % 360
  if 0 ≤ normalized_angle && normalized_angle < 90 then 1
  else if 90 ≤ normalized_angle && normalized_angle < 180 then 2
  else if 180 ≤ normalized_angle && normalized_angle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_quadrant :
  quadrant angle = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_l1188_118805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1188_118863

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (60 - x)) + Real.sqrt (x * (5 - x))

theorem max_value_of_f :
  ∃ (x₀ : ℝ) (M : ℝ),
    x₀ = 5 ∧
    M = Real.sqrt 275 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → f x ≤ M) ∧
    f x₀ = M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1188_118863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_pairs_satisfy_l1188_118872

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (abs x + 1)

-- Define the property for a pair (a, b) to satisfy M = N
def satisfies_M_equals_N (a b : ℝ) : Prop :=
  a < b ∧ Set.Icc a b = {y | ∃ x ∈ Set.Icc a b, f x = y}

-- Theorem statement
theorem exactly_three_pairs_satisfy :
  ∃! (s : Set (ℝ × ℝ)), s.Finite ∧ s.ncard = 3 ∧ 
  ∀ p : ℝ × ℝ, p ∈ s ↔ satisfies_M_equals_N p.1 p.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_pairs_satisfy_l1188_118872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1188_118830

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x + 2) / (3 - x) ≥ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}

-- Statement of the theorem
theorem complement_A_intersect_B : (U \ A) ∩ B = Set.Icc 3 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1188_118830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1188_118806

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The region consisting of all points within distance d of a line segment of length l -/
noncomputable def region_volume (d l : ℝ) : ℝ :=
  cylinder_volume d l + sphere_volume d

theorem line_segment_length (volume : ℝ) (d : ℝ) (h : d = 3) (h' : volume = 216 * Real.pi) :
  ∃ l : ℝ, region_volume d l = volume ∧ l = 20 := by
  sorry

#check line_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1188_118806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1188_118860

-- Define the points
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 10)
def C : ℝ × ℝ := (8, 6)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem triangle_perimeter :
  distance A B + distance B C + distance C A = 7 + 2 * Real.sqrt 13 + 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1188_118860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_work_cole_drive_time_l1188_118801

noncomputable def round_trip_time (speed_to_work speed_to_home total_time : ℝ) : ℝ :=
  let distance := (speed_to_work * speed_to_home * total_time) / (speed_to_work + speed_to_home)
  distance / speed_to_work

theorem time_to_work (speed_to_work speed_to_home total_time : ℝ) 
  (h1 : speed_to_work > 0)
  (h2 : speed_to_home > 0)
  (h3 : total_time > 0) :
  round_trip_time speed_to_work speed_to_home total_time = 
    (speed_to_home * total_time) / (speed_to_work + speed_to_home) :=
by
  sorry

-- For the specific problem
theorem cole_drive_time :
  round_trip_time 75 105 4 = 140 / 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_work_cole_drive_time_l1188_118801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_f_monotone_increasing_on_zero_pi_l1188_118811

noncomputable def f (x : ℝ) : ℝ := 2019 * Real.sin ((1/3) * x + Real.pi/6)

theorem monotone_increasing_interval (a b : ℝ) (h1 : 0 ≤ a) (h2 : a < b) (h3 : b ≤ Real.pi) :
  StrictMonoOn f (Set.Icc a b) := by sorry

theorem f_monotone_increasing_on_zero_pi :
  ∃ (a b : ℝ), a = 0 ∧ b = Real.pi ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 2*Real.pi → (StrictMonoOn f (Set.Icc a b) ↔ StrictMonoOn f (Set.Icc 0 Real.pi))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_f_monotone_increasing_on_zero_pi_l1188_118811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_is_36_l1188_118852

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0
  h_a1 : a 1 = 8
  h_geometric : (a 5) ^ 2 = (a 1) * (a 7)

/-- The sum of the first n terms of the arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1) + (n * (n - 1) / 2) * seq.d

/-- The theorem stating that the maximum value of S_n is 36 -/
theorem max_S_n_is_36 (seq : ArithmeticSequence) :
  ∃ n : ℕ, S_n seq n = 36 ∧ ∀ m : ℕ, S_n seq m ≤ 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_is_36_l1188_118852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l1188_118802

/-- A sinusoidal function with specific properties -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + 1

/-- The theorem stating the properties of the function and its range -/
theorem sinusoidal_function_properties :
  ∀ (A ω φ : ℝ),
    A > 0 →
    ω > 0 →
    -π/2 < φ ∧ φ < π/2 →
    (∀ x, f A ω φ x ≤ 3) →
    f A ω φ (π/3) = 3 →
    (∀ x, f A ω φ (x + π/ω) = f A ω φ x) →
    (∃ (B C : ℝ), ∀ x, f A ω φ x = 2 * Real.sin (2*x - π/6) + 1) ∧
    (∀ x, 0 ≤ x ∧ x ≤ π/2 → 0 ≤ f A ω φ x ∧ f A ω φ x ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l1188_118802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_83_congruence_l1188_118855

theorem base_83_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 19) 
  (h3 : (956812 : ℤ) ≡ b [ZMOD 17]) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_83_congruence_l1188_118855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skaters_meeting_distance_l1188_118841

/-- The distance between two skaters on a frozen lake -/
def distance_AB : ℝ := 200

/-- Allie's skating speed in meters per second -/
def allie_speed : ℝ := 10

/-- Billie's skating speed in meters per second -/
def billie_speed : ℝ := 7

/-- The angle between Allie's path and the line AB in radians -/
noncomputable def allie_angle : ℝ := Real.pi / 4

/-- The time at which Allie and Billie meet -/
def meeting_time : ℝ := 20

theorem skaters_meeting_distance :
  allie_speed * meeting_time = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skaters_meeting_distance_l1188_118841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l1188_118842

noncomputable def f (x : ℝ) : ℝ := 3^x - 8

theorem root_interval (m : ℕ) : 
  (∃ x ∈ Set.Icc (m : ℝ) ((m : ℝ) + 1), f x = 0) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l1188_118842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l1188_118881

/-- A power function passing through (1/2, √2/2) has k + α = 3/2 -/
theorem power_function_sum (k α : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ f : ℝ → ℝ, f x = k * x^α) →  -- f is a power function
  k * (1/2)^α = Real.sqrt 2/2 →                    -- f passes through (1/2, √2/2)
  k + α = 3/2 :=                                   -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l1188_118881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_p_and_q_l1188_118850

/-- Represents a number consisting of n consecutive 1's -/
def consecutive_ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Represents the number p as described in the problem -/
def p (n : ℕ) : ℕ := 
  (consecutive_ones n) * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

/-- Represents the number q as described in the problem -/
def q (n : ℕ) : ℕ := 
  (consecutive_ones (n+1)) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

/-- The main theorem to be proved -/
theorem divisibility_of_p_and_q (n : ℕ) 
  (h : ∃ (k : ℕ), 1987 ∣ consecutive_ones k) : 
  1987 ∣ p n ∧ 1987 ∣ q n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_p_and_q_l1188_118850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleArea_correct_l1188_118816

/-- The area of a triangle given the coordinates of its three vertices. -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs (x₁*y₂ - x₂*y₁ + x₂*y₃ - x₃*y₂ + x₃*y₁ - x₁*y₃)

/-- Theorem stating that triangleArea computes the correct area of a triangle. -/
theorem triangleArea_correct (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  triangleArea x₁ y₁ x₂ y₂ x₃ y₃ = 
    abs ((1/2) * (x₁*y₂ - x₂*y₁ + x₂*y₃ - x₃*y₂ + x₃*y₁ - x₁*y₃)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleArea_correct_l1188_118816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_at_6_or_7_l1188_118865

noncomputable def a (n : ℕ) : ℝ := (n + 1 : ℝ) * (7/8) ^ n

theorem sequence_max_at_6_or_7 :
  ∃ k ∈ ({6, 7} : Set ℕ), ∀ n : ℕ, a n ≤ a k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_at_6_or_7_l1188_118865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2x_increasing_no_extreme_l1188_118815

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

-- State the theorem
theorem f_2x_increasing_no_extreme :
  (∀ x y : ℝ, x < y → f (2 * x) < f (2 * y)) ∧
  (∀ c : ℝ, ¬ (∀ x : ℝ, f (2 * x) ≤ f (2 * c))) ∧
  (∀ c : ℝ, ¬ (∀ x : ℝ, f (2 * x) ≥ f (2 * c))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2x_increasing_no_extreme_l1188_118815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l1188_118871

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus and intersecting the parabola -/
structure IntersectingLine :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (h_A_on_parabola : A ∈ Parabola)
  (h_B_on_parabola : B ∈ Parabola)
  (h_line_through_focus : ∃ t : ℝ, A = Focus + t • (B - Focus) ∨ B = Focus + t • (A - Focus))

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_intersection_distance (l : IntersectingLine) :
  distance l.A Focus = 2 → distance l.B Focus = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l1188_118871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1188_118828

theorem sum_remainder_mod_15 (x y z : ℕ) 
  (hx : x % 15 = 8)
  (hy : y % 15 = 11)
  (hz : z % 15 = 13) :
  (x + y + z) % 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1188_118828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1188_118869

theorem sin_alpha_value (α β : Real) 
  (h1 : Real.cos (α - β) = 3/5)
  (h2 : Real.sin β = -5/13)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : -Real.pi/2 < β ∧ β < 0) :
  Real.sin α = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1188_118869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poles_intersection_height_l1188_118877

/-- Represents a line in 2D space --/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Calculate the intersection point of two lines --/
noncomputable def intersectionPoint (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.yIntercept - l1.yIntercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.yIntercept
  (x, y)

theorem poles_intersection_height :
  let line1 := Line.mk (-1/3) 40  -- Line from (0, 40) to (120, 0)
  let line2 := Line.mk (-1/2) 0   -- Line from (0, 0) to (120, 60)
  let (_, y) := intersectionPoint line1 line2
  y = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poles_intersection_height_l1188_118877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1188_118853

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 6)
  (h2 : (x : ℝ) / (y : ℝ) = 96.15) : 
  y = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1188_118853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_l1188_118859

/-- A bijective function from ℕ to ℕ × ℕ -/
noncomputable def f : ℕ → ℕ × ℕ :=
  sorry

/-- The inverse of f -/
noncomputable def f_inv : ℕ × ℕ → ℕ :=
  sorry

/-- The sequence a_k -/
def a : ℕ → ℕ
  | 0 => 1
  | k + 1 => 
    let (m, n) := f (k + 1)
    sorry  -- Definition of a_k for k ≥ 1

/-- Set A -/
def A : Set ℕ :=
  {x | ∃ k, a k = x}

/-- Set B -/
def B : Set ℕ :=
  {x | x ∉ A}

theorem partition_exists : 
  (∀ x y z, x ∈ A → y ∈ A → z ∈ A → x < y ∧ y < z → z - y ≠ y - x) ∧ 
  (∀ m n : ℕ, n > 0 → ∃ k, m + n * k ∉ B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_l1188_118859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_from_angles_l1188_118882

-- Define a structure for a quadrilateral
structure Quadrilateral :=
(a b c d : ℝ)

-- Define what it means for a quadrilateral to be a parallelogram
def IsParallelogram (q : Quadrilateral) : Prop :=
  q.a = q.c ∧ q.b = q.d

theorem parallelogram_from_angles (q : Quadrilateral) 
  (quad_sum : q.a + q.b + q.c + q.d = 360)
  (first_third_equal : q.a = q.c)
  (second_fourth_equal : q.b = q.d)
  (first_angle : q.a = 88)
  (second_angle : q.b = 92) : 
  IsParallelogram q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_from_angles_l1188_118882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_week_division_count_l1188_118884

def seconds_in_week : ℕ := 604800

theorem week_division_count : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = seconds_in_week)
    (Finset.range (seconds_in_week + 1) ×ˢ Finset.range (seconds_in_week + 1))).card = 144 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_week_division_count_l1188_118884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l1188_118858

/-- Represents a triangle in 2D space -/
structure Triangle (α : Type*) [LinearOrderedField α] where
  /-- The side length of the triangle -/
  sideLength : α
  /-- Predicate to check if the triangle is equilateral -/
  isEquilateral : Prop

/-- Theorem about equilateral triangle division -/
theorem equilateral_triangle_division :
  ¬ (∃ n : ℕ, n * (n + 1) / 2 = 2011) ∧
  ∃ (m : ℕ) (l₁ l₂ : ℝ), l₁ > 0 ∧ l₂ > 0 ∧ l₁ ≠ l₂ ∧
    ∃ (triangles : Finset (Triangle ℝ)),
      (∀ t ∈ triangles, t.isEquilateral) ∧
      (∀ t ∈ triangles, t.sideLength = l₁ ∨ t.sideLength = l₂) ∧
      triangles.card = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l1188_118858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stirling_constant_value_l1188_118837

/-- Stirling's formula constant -/
noncomputable def a : ℝ := Real.sqrt (2 * Real.pi)

/-- Stirling's formula -/
def stirling_formula (n : ℕ) : Prop :=
  Real.sqrt (2 * Real.pi * n) * (n / Real.exp 1) ^ n < Nat.factorial n ∧
  Nat.factorial n < Real.sqrt (2 * Real.pi * n) * (n / Real.exp 1) ^ n * Real.exp (1 / (12 * n))

/-- Theorem stating that Stirling's formula implies the value of the constant a -/
theorem stirling_constant_value :
  (∀ n : ℕ, stirling_formula n) → a = Real.sqrt (2 * Real.pi) :=
by
  intro h
  -- The proof is omitted for now
  sorry

#check stirling_constant_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stirling_constant_value_l1188_118837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarees_original_price_l1188_118873

/-- The original price of an item with two successive discounts -/
noncomputable def originalPrice (finalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  finalPrice / ((1 - discount1) * (1 - discount2))

/-- Theorem: The original price of sarees is approximately 390 -/
theorem sarees_original_price : 
  let finalPrice : ℝ := 285.09
  let discount1 : ℝ := 0.14
  let discount2 : ℝ := 0.15
  abs (originalPrice finalPrice discount1 discount2 - 390) < 1 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarees_original_price_l1188_118873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_l1188_118803

noncomputable section

/-- The region of the plate -/
def Ω (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ (p.1^2 / a^2) + (p.2^2 / b^2) ≤ 1}

/-- The surface density function -/
def P (x y : ℝ) : ℝ := x * y

/-- The mass of the plate -/
def M (a b : ℝ) : ℝ := ∫ p in Ω a b, P p.1 p.2

/-- The moment about the y-axis -/
def m_y (a b : ℝ) : ℝ := ∫ p in Ω a b, p.1 * P p.1 p.2

/-- The moment about the x-axis -/
def m_x (a b : ℝ) : ℝ := ∫ p in Ω a b, p.2 * P p.1 p.2

/-- The x-coordinate of the center of gravity -/
noncomputable def x_C (a b : ℝ) : ℝ := m_y a b / M a b

/-- The y-coordinate of the center of gravity -/
noncomputable def y_C (a b : ℝ) : ℝ := m_x a b / M a b

/-- The theorem stating the location of the center of gravity -/
theorem center_of_gravity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (x_C a b, y_C a b) = (8 * a / 15, 8 * b / 15) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_l1188_118803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_problem_l1188_118846

/-- Converts speed from km/h to m/s -/
noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

/-- Calculates the time for one train to cross another moving in the same direction -/
noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (kmph_to_mps speed1 - kmph_to_mps speed2)

/-- Theorem: The time for a train of length 200 m moving at 72 kmph to cross another train
    of length 300 m moving at 36 kmph in the same direction is 50 seconds -/
theorem train_crossing_problem :
  train_crossing_time 200 300 72 36 = 50 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_problem_l1188_118846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_prime_divisors_l1188_118861

def S (n : ℕ) : ℕ → ℕ
  | 0     => 2  -- S₁ > 1, we choose 2 as the initial value
  | (i+1) => let Si := S n i; Si * (Si + 1)

theorem hundredth_term_prime_divisors :
  ∃ (ps : Finset ℕ), ps.card = 100 ∧ 
    ∀ p ∈ ps, Nat.Prime p ∧ p ∣ S 100 99 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_prime_divisors_l1188_118861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_projection_not_two_points_l1188_118810

/-- Two skew lines in 3D space -/
structure SkewLines where
  line1 : Set (ℝ × ℝ × ℝ)
  line2 : Set (ℝ × ℝ × ℝ)
  skew : line1 ∩ line2 = ∅ ∧ ¬ ∃ (v : ℝ × ℝ × ℝ), ∀ (p1 p2 : ℝ × ℝ × ℝ), p1 ∈ line1 → p2 ∈ line2 → ∃ (t : ℝ), p2 - p1 = t • v

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Projection of a point onto a plane -/
noncomputable def project (p : ℝ × ℝ × ℝ) (plane : Plane) : ℝ × ℝ × ℝ :=
  sorry

/-- Projection of a line onto a plane -/
noncomputable def projectLine (line : Set (ℝ × ℝ × ℝ)) (plane : Plane) : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ q ∈ line, p = project q plane}

/-- Theorem: The projections of two skew lines onto a plane cannot be two distinct points -/
theorem skew_lines_projection_not_two_points (lines : SkewLines) (plane : Plane) :
  ¬∃ p1 p2 : ℝ × ℝ × ℝ, p1 ≠ p2 ∧ 
    projectLine lines.line1 plane = {p1} ∧ 
    projectLine lines.line2 plane = {p2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_projection_not_two_points_l1188_118810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1188_118868

noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := (3 - x) / 3

def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 = p.2}

theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
by sorry

#check intersection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1188_118868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_l1188_118864

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- Distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Slope between two points -/
noncomputable def slopeAB (a b : Point) : ℝ :=
  (b.y - a.y) / (b.x - a.x)

/-- Main theorem -/
theorem parabola_slope (c : Parabola) (a b : Point) :
  a.y^2 = 2 * c.p * a.x →
  b.y^2 = 2 * c.p * b.x →
  a.x > 0 →
  a.y > 0 →
  b.x > 0 →
  b.y > 0 →
  distance a (focus c) = 3 →
  distance b (focus c) = 7 →
  distance a b = 5 →
  slopeAB a b = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_l1188_118864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_k_value_l1188_118832

noncomputable section

-- Define the parabola
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2^2 = 2 * p * point.1}

-- Define the line
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2 = k * point.1 - 2}

-- Define the focus of the parabola
def Focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Define the distance between two points
noncomputable def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_equation_and_k_value
  (p : ℝ)
  (h1 : p > 0)
  (m : ℝ)
  (h2 : (4, m) ∈ Parabola p)
  (h3 : Distance (4, m) (Focus p) = 6)
  (k : ℝ)
  (h4 : ∃ A B : ℝ × ℝ, A ∈ Parabola p ∩ Line k ∧ B ∈ Parabola p ∩ Line k ∧ A ≠ B)
  (h5 : ∃ A B : ℝ × ℝ, A ∈ Parabola p ∩ Line k ∧ B ∈ Parabola p ∩ Line k ∧ (A.1 + B.1) / 2 = 2)
  : p = 4 ∧ k = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_k_value_l1188_118832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l1188_118856

noncomputable def z : ℂ := (3 - 4*Complex.I) / (2 - Complex.I)

theorem magnitude_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l1188_118856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_is_zero_l1188_118843

/-- Represents the time difference between two routes in minutes -/
noncomputable def route_time_difference 
  (route_x_distance : ℝ) 
  (route_x_speed : ℝ) 
  (route_y_total_distance : ℝ) 
  (route_y_construction_distance : ℝ) 
  (route_y_construction_speed : ℝ) 
  (route_y_normal_speed : ℝ) : ℝ :=
  let route_x_time := route_x_distance / route_x_speed * 60
  let route_y_construction_time := route_y_construction_distance / route_y_construction_speed * 60
  let route_y_normal_distance := route_y_total_distance - route_y_construction_distance
  let route_y_normal_time := route_y_normal_distance / route_y_normal_speed * 60
  let route_y_total_time := route_y_construction_time + route_y_normal_time
  route_x_time - route_y_total_time

/-- Theorem stating that Route Y saves 0 minutes compared to Route X -/
theorem route_time_difference_is_zero : 
  route_time_difference 7 35 6 1 10 50 = 0 := by
  -- Unfold the definition of route_time_difference
  unfold route_time_difference
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_is_zero_l1188_118843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_base_pyramid_properties_l1188_118870

/-- A pyramid with a rhombic base -/
structure RhombicBasePyramid where
  r : ℝ  -- radius of the inscribed circle in the base
  acute_angle : ℝ  -- acute angle of the rhombus base
  slant_angle : ℝ  -- angle of inclination of slant faces to the base

/-- Volume of the rhombic base pyramid -/
noncomputable def volume (p : RhombicBasePyramid) : ℝ := (8 * p.r^3 * Real.sqrt 3) / 3

/-- Total surface area of the rhombic base pyramid -/
def total_surface_area (p : RhombicBasePyramid) : ℝ := 24 * p.r^2

theorem rhombic_base_pyramid_properties (p : RhombicBasePyramid) 
    (h1 : p.acute_angle = π / 6)  -- 30°
    (h2 : p.slant_angle = π / 3)  -- 60°
    : volume p = (8 * p.r^3 * Real.sqrt 3) / 3 ∧ 
      total_surface_area p = 24 * p.r^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_base_pyramid_properties_l1188_118870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_fourth_vertex_l1188_118809

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rhombus defined by four vertices -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- The main theorem -/
theorem rhombus_fourth_vertex 
  (r : Rhombus) 
  (h1 : r.v1 = ⟨0, 3.5⟩) 
  (h2 : r.v2 = ⟨8, 0⟩) 
  (h3 : r.v3 = ⟨0, -3.5⟩) 
  (h4 : r.v4.y = 0) 
  (h5 : rhombusArea 7 (|8 - r.v4.x|) = 56) : 
  r.v4.x = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_fourth_vertex_l1188_118809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_star_b_equals_interval_l1188_118885

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = 2*x - x^2}
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define the operation A*B
def star (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- State the theorem
theorem a_star_b_equals_interval : star A B = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_star_b_equals_interval_l1188_118885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_profit_approx_nine_percent_l1188_118886

/-- Calculates the profit percentage for tomato sales given the purchase price,
    loss percentage, and selling price. -/
noncomputable def profit_percentage (purchase_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let remaining_percentage := 1 - loss_percentage
  let revenue := selling_price * remaining_percentage
  let profit := revenue - purchase_price
  (profit / purchase_price) * 100

/-- The profit percentage for the tomato sales is approximately 9% -/
theorem tomato_profit_approx_nine_percent :
  let purchase_price : ℝ := 0.80
  let loss_percentage : ℝ := 0.10
  let selling_price : ℝ := 0.968888888888889
  abs (profit_percentage purchase_price loss_percentage selling_price - 9) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_profit_approx_nine_percent_l1188_118886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kevin_finishes_first_l1188_118844

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the mowing rate of a lawn mower -/
structure MowingRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a person with their lawn and mower -/
structure Person where
  name : String
  lawn : LawnArea
  mower : MowingRate

noncomputable def mowing_time (p : Person) : ℝ :=
  p.lawn.size / p.mower.rate

theorem kevin_finishes_first (jenny kevin lana : Person)
  (h1 : jenny.lawn.size = 3 * kevin.lawn.size)
  (h2 : jenny.lawn.size = 4 * lana.lawn.size)
  (h3 : lana.mower.rate = (1 / 4) * jenny.mower.rate)
  (h4 : kevin.mower.rate = 2 * lana.mower.rate) :
  mowing_time kevin < min (mowing_time jenny) (mowing_time lana) := by
  sorry

#check kevin_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kevin_finishes_first_l1188_118844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l1188_118839

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a + 3) * x - 5 else 2 * a / x

theorem range_of_a_for_increasing_f (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ↔ a ∈ Set.Icc (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l1188_118839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_speed_l1188_118887

/-- Proves that given a circular track of 0.25 miles, if person A completes 12 laps in 0.5 hours,
    and person B's average speed is half of person A's, then person B's average speed is 3 miles per hour. -/
theorem gerald_speed (track_length : ℝ) (laps : ℕ) (time : ℝ) (speed_ratio : ℝ) :
  track_length = 0.25 →
  laps = 12 →
  time = 0.5 →
  speed_ratio = 0.5 →
  (track_length * (laps : ℝ) / time) * speed_ratio = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_speed_l1188_118887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_sum_distances_to_intersections_l1188_118835

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 - Real.sqrt 2 / 2 * t, Real.sqrt 5 + Real.sqrt 2 / 2 * t)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sqrt 5 * Real.sin θ

-- Define point P
noncomputable def point_P : ℝ × ℝ := (3, Real.sqrt 5)

-- Theorem for the distance from the center of C to line l
theorem distance_center_to_line :
  let center_C : ℝ × ℝ := (0, Real.sqrt 5)
  Real.sqrt ((center_C.1 - (line_l 0).1)^2 + (center_C.2 - (line_l 0).2)^2) = 3 * Real.sqrt 2 / 2 := by
  sorry

-- Theorem for the sum of distances from P to intersection points
theorem sum_distances_to_intersections :
  ∃ (t₁ t₂ : ℝ), 
    ((line_l t₁).1)^2 + ((line_l t₁).2 - Real.sqrt 5)^2 = 5 ∧
    ((line_l t₂).1)^2 + ((line_l t₂).2 - Real.sqrt 5)^2 = 5 ∧
    Real.sqrt ((point_P.1 - (line_l t₁).1)^2 + (point_P.2 - (line_l t₁).2)^2) +
    Real.sqrt ((point_P.1 - (line_l t₂).1)^2 + (point_P.2 - (line_l t₂).2)^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_sum_distances_to_intersections_l1188_118835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_no_purchase_days_l1188_118820

/-- Represents the number of days Vasya buys 9 marshmallows -/
def x : ℕ := sorry

/-- Represents the number of days Vasya buys 2 meat pies -/
def y : ℕ := sorry

/-- Represents the number of days Vasya buys 4 marshmallows and 1 meat pie -/
def z : ℕ := sorry

/-- Represents the number of days Vasya buys nothing -/
def w : ℕ := sorry

/-- The total number of school days -/
def total_days : ℕ := 15

/-- The total number of marshmallows bought -/
def total_marshmallows : ℕ := 30

/-- The total number of meat pies bought -/
def total_meat_pies : ℕ := 9

theorem vasya_no_purchase_days :
  (x + y + z + w = total_days) →
  (9 * x + 4 * z = total_marshmallows) →
  (2 * y + z = total_meat_pies) →
  w = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_no_purchase_days_l1188_118820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_distance_line_through_point_with_distance_proof_l1188_118854

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line passes through a point -/
def linePassesThroughPoint (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem line_through_point_with_distance (l : Line) : Prop :=
  (linePassesThroughPoint 2 (-4) l) ∧
  (distancePointToLine 0 0 l = 2) →
  ((l.a = 3 ∧ l.b = 4 ∧ l.c = 10) ∨ (l.a = 1 ∧ l.b = 0 ∧ l.c = -2))

-- The proof goes here
theorem line_through_point_with_distance_proof : ∀ l : Line, line_through_point_with_distance l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_distance_line_through_point_with_distance_proof_l1188_118854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_is_one_l1188_118874

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The eccentricity of a hyperbola with semi-major axis a and semi-minor axis b -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- Theorem: The product of eccentricities of ellipse M and hyperbola N is 1 -/
theorem eccentricity_product_is_one (m n : ℝ) :
  m > 0 → n > 0 →
  (∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are common foci of M and N
    -- M: x²/m² + y²/2 = 1
    -- N: x²/n² - y² = 1
    -- P is a common point of M and N
    -- PF₁ is perpendicular to F₁F₂
    True) →
  ellipse_eccentricity m (Real.sqrt 2) * hyperbola_eccentricity n 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_is_one_l1188_118874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1188_118848

/-- Define the sequence recursively -/
noncomputable def x : ℕ → ℝ → ℝ
  | 0, a => a
  | n + 1, a => (Real.sqrt (3 - 3 * (x n a)^2) - x n a) / 2

/-- Theorem statement -/
theorem sequence_properties (a : ℝ) (h : -1 < a ∧ a < 1) :
  (∀ n : ℕ, 0 < x n a) ↔ (0 < a ∧ a < Real.sqrt 3 / 2) ∧
  ∀ n : ℕ, x (n + 2) a = x n a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1188_118848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_half_l1188_118890

open BigOperators Real Nat

/-- The sum of the infinite series Σ(n^3 + 2n^2 - n - 1) / ((n + 3)!) for n from 1 to infinity equals 1/2 -/
theorem infinite_series_sum_equals_half :
  ∑' n : ℕ+, (n.val^3 + 2*n.val^2 - n.val - 1) / (factorial (n.val + 3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_half_l1188_118890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l1188_118880

/-- Multiplication of complex numbers -/
def complexMul (z w : Complex) : Complex :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

theorem complex_multiplication (x a b : ℝ) :
  let z₀ : Complex := ⟨Real.cos x, Real.sin x⟩
  let z₁ : Complex := ⟨a, b⟩
  let z : Complex := complexMul z₀ z₁
  (z.re = a * Real.cos x - b * Real.sin x) ∧
  (z.im = a * Real.sin x + b * Real.cos x) ∧
  (a = Real.sqrt 3 → b = -1 → z.im = 6/5 → Real.sin x = (3 * Real.sqrt 3 + 4) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l1188_118880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_point_distance_l1188_118892

-- Define the square sheet
noncomputable def square_area : ℝ := 18

-- Define the side length of the square
noncomputable def side_length : ℝ := Real.sqrt square_area

-- Define the length of the visible black triangle's side after folding
noncomputable def black_triangle_side : ℝ := Real.sqrt 12

-- Theorem statement
theorem folded_point_distance : 
  ∃ (distance : ℝ), 
    distance = 2 * Real.sqrt 6 ∧
    distance^2 = 2 * black_triangle_side^2 ∧
    (1/2) * black_triangle_side^2 = square_area - black_triangle_side^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_point_distance_l1188_118892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_remainder_theorem_smallest_prime_with_property_smallest_prime_value_l1188_118866

theorem prime_remainder_theorem (p : Nat) (h_prime : Nat.Prime p) (h_rem : p % 12 = 1) :
  (p^2 + 12) % 12 = 1 ∧ ∀ q < p, Nat.Prime q → q % 12 ≠ 1 :=
by sorry

theorem smallest_prime_with_property : 
  ∃ p, Nat.Prime p ∧ p % 12 = 1 ∧ ∀ q < p, Nat.Prime q → q % 12 ≠ 1 :=
by sorry

noncomputable def smallest_prime := Nat.find smallest_prime_with_property

theorem smallest_prime_value : 
  smallest_prime = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_remainder_theorem_smallest_prime_with_property_smallest_prime_value_l1188_118866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_invariant_impossible_all_black_l1188_118862

/-- Represents the state of a card (White or Black side up) -/
inductive CardState
| White
| Black
deriving Repr, DecidableEq

/-- Represents the circular arrangement of 10 cards -/
def CardArrangement := Fin 10 → CardState

/-- Checks if a set of indices represents 5 alternating cards in the circular arrangement -/
def isAlternatingSet (s : Finset (Fin 10)) : Prop :=
  s.card = 5 ∧ ∀ i ∈ s, (i + 2) % 10 ∈ s

/-- Counts the number of white cards in a given set of indices -/
def countWhiteCards (arr : CardArrangement) (s : Finset (Fin 10)) : Nat :=
  s.filter (fun i => arr i = CardState.White) |>.card

/-- Represents a valid move: either flipping 4 consecutive cards or 4 out of 5 consecutive cards -/
inductive Move
| FourConsecutive (start : Fin 10)
| FourOutOfFive (start : Fin 10)

/-- Applies a move to the card arrangement -/
def applyMove (arr : CardArrangement) (move : Move) : CardArrangement :=
  match move with
  | Move.FourConsecutive start =>
      fun i => if i ∈ [start, (start + 1) % 10, (start + 2) % 10, (start + 3) % 10] then
        match arr i with
        | CardState.White => CardState.Black
        | CardState.Black => CardState.White
      else arr i
  | Move.FourOutOfFive start =>
      fun i => if i ∈ [start, (start + 1) % 10, (start + 3) % 10, (start + 4) % 10] then
        match arr i with
        | CardState.White => CardState.Black
        | CardState.Black => CardState.White
      else arr i

/-- The main theorem: The parity of white cards in any alternating set remains unchanged after any move -/
theorem parity_invariant (arr : CardArrangement) (s : Finset (Fin 10)) (move : Move) :
  isAlternatingSet s →
  (countWhiteCards arr s) % 2 = (countWhiteCards (applyMove arr move) s) % 2 := by
  sorry

/-- Corollary: It's impossible to turn all cards black side up -/
theorem impossible_all_black (initial : CardArrangement) (moves : List Move) :
  (∀ i, initial i = CardState.White) →
  ∃ s, isAlternatingSet s ∧ ∃ i ∈ s, (moves.foldl applyMove initial) i = CardState.White := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_invariant_impossible_all_black_l1188_118862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_equation_l1188_118834

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 3, a; 0, 1, 5; 0, 0, 1]

theorem matrix_power_equation (a : ℝ) (n : ℕ) :
  (A a) ^ n = !![1, 27, 3060; 0, 1, 45; 0, 0, 1] →
  a + n = 289 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_equation_l1188_118834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1188_118800

theorem triangle_abc_properties 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a > 0 ∧ b > 0 ∧ c > 0)
  (h6 : a / Real.sin A = b / Real.sin B)
  (h7 : b / Real.sin B = c / Real.sin C)
  (h8 : Real.cos (2*B) - Real.cos (2*A) = 2 * Real.sin C * (Real.sin A - Real.sin C)) :
  B = π/3 ∧ 
  (b = Real.sqrt 3 → Real.sqrt 3 < 2*a + c ∧ 2*a + c ≤ 2 * Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1188_118800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_75_factorial_l1188_118818

theorem last_two_nonzero_digits_75_factorial : 
  ∃ k : ℕ, Nat.factorial 75 = 100 * k + 32 ∧ k % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_75_factorial_l1188_118818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l1188_118895

def n : ℕ := 2^4 * 3^2 * 5 * 7

def is_even_factor (d : ℕ) : Prop :=
  d ∣ n ∧ Even d

theorem count_even_factors : 
  (Finset.filter (fun d => d ∣ n ∧ Even d) (Finset.range (n + 1))).card = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l1188_118895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_calculation_l1188_118857

/-- Calculates the upstream distance swam given the downstream distance, time, and still water speed -/
noncomputable def upstream_distance (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let current_speed := downstream_distance / time / 2 - still_water_speed / 2
  (still_water_speed - current_speed) * time

theorem upstream_distance_calculation 
  (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ)
  (h1 : downstream_distance = 36)
  (h2 : time = 2)
  (h3 : still_water_speed = 15.5) :
  upstream_distance downstream_distance time still_water_speed = 26 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_calculation_l1188_118857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_properties_l1188_118819

/-- Given positive real numbers m and n, and function f(x) = |x + m| + |2x - n|,
    this theorem proves properties about the minimum value of f and a related expression. -/
theorem min_value_properties (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let f : ℝ → ℝ := fun x => |x + m| + |2*x - n|
  (∀ x, f x ≥ m + n/2) ∧
  (m + n/2 = 2 → m^2 + n^2/4 ≥ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_properties_l1188_118819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l1188_118812

noncomputable def hyperbola_center : ℝ × ℝ := (3, -1)
noncomputable def hyperbola_focus : ℝ × ℝ := (3 + Real.sqrt 20, -1)
noncomputable def hyperbola_vertex : ℝ × ℝ := (7, -1)

noncomputable def h : ℝ := hyperbola_center.1
noncomputable def k : ℝ := hyperbola_center.2
noncomputable def a : ℝ := hyperbola_vertex.1 - hyperbola_center.1
noncomputable def c : ℝ := hyperbola_focus.1 - hyperbola_center.1
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l1188_118812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1188_118896

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the hyperbola
def is_on_hyperbola (k : ℝ) (x y : ℝ) : Prop := x^2 / (k - 6) - y^2 / k = 1

-- State the theorem
theorem intersection_range (k : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ is_on_circle A.1 A.2 ∧ is_on_circle B.1 B.2 ∧ 
    line k A.1 = A.2 ∧ line k B.1 = B.2) →
  (∃ x y : ℝ, is_on_hyperbola k x y ∧ ∀ a b : ℝ, is_on_hyperbola k a b → (a = 0 ∨ b = 0)) →
  k < -2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1188_118896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OCD_l1188_118831

-- Define the quadrilateral CDEF as a parallelogram
structure Parallelogram

-- Define the trapezoid ABCD
structure Trapezoid

-- Define triangles AFH, GEB, and OCD
structure Triangle

-- Define an area function
def area {α : Type*} : α → ℝ := sorry

-- Define the quadrilateral CDEF as a parallelogram
def CDEF : Parallelogram := sorry

-- Define the trapezoid ABCD
def ABCD : Trapezoid := sorry

-- Define triangles AFH, GEB, and OCD
def AFH : Triangle := sorry
def GEB : Triangle := sorry
def OCD : Triangle := sorry

-- State the given areas
axiom area_ABCD : area ABCD = 320
axiom area_AFH : area AFH = 32
axiom area_GEB : area GEB = 48

-- Theorem to prove
theorem area_OCD : area OCD = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OCD_l1188_118831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1188_118829

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Define the two tangent lines
def line1 (x : ℝ) : Prop := x = 2
def line2 (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem tangent_lines_to_circle :
  (∀ x y, line1 x → (circle_eq x y → x = 2 ∧ y = 1)) ∧
  (∀ x y, line2 x y → (circle_eq x y → x = 5/3 ∧ y = 5/4)) ∧
  line1 P.1 ∧
  line2 P.1 P.2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1188_118829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1188_118821

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 1)
  let point : ℝ × ℝ := (2, 2)
  let equation := (x - center.1)^2 + (y - center.2)^2 = 2
  (∀ p : ℝ × ℝ, p = point → (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2) →
  equation
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1188_118821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_non_prime_x_l1188_118851

noncomputable def is_prime (n : ℤ) : Prop := 
  n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem greatest_non_prime_x : 
  ∃ (x : ℤ), x = 1 ∧ 
  (∀ y : ℤ, y > x → (6.1 * (10 : ℝ)^y ≥ 620 ∨ is_prime y)) ∧ 
  ¬ is_prime x ∧ 
  6.1 * (10 : ℝ)^x < 620 := by
  sorry

#check greatest_non_prime_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_non_prime_x_l1188_118851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_fixed_point_l1188_118899

/-- The linear function passing through a fixed point -/
theorem linear_function_fixed_point (a : ℝ) : 
  (λ x : ℝ => a * x - 3 * a + 1) 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_fixed_point_l1188_118899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_l1188_118883

/-- Proves that the number of members in a group is 44, given the total collection and contribution rule --/
theorem group_size (total_rupees : ℚ) (h1 : total_rupees = 19.36) :
  ∃ n : ℕ, n * n = (total_rupees * 100).floor ∧ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_l1188_118883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_must_ace_all_remaining_quizzes_l1188_118898

/-- Lisa's quiz performance tracker -/
structure QuizPerformance where
  totalQuizzes : ℕ
  goalPercentage : ℚ
  quizzesTaken : ℕ
  aScores : ℕ
  
/-- Calculate the maximum number of non-A scores allowed in remaining quizzes -/
def maxNonAScores (qp : QuizPerformance) : ℤ :=
  qp.totalQuizzes - qp.quizzesTaken - (qp.totalQuizzes * qp.goalPercentage.num / qp.goalPercentage.den - qp.aScores)

/-- Theorem: Lisa can't score lower than A on any remaining quiz -/
theorem lisa_must_ace_all_remaining_quizzes (qp : QuizPerformance)
  (h1 : qp.totalQuizzes = 60)
  (h2 : qp.goalPercentage = 9/10)
  (h3 : qp.quizzesTaken = 40)
  (h4 : qp.aScores = 30) :
  maxNonAScores qp = 0 := by
  sorry

#eval maxNonAScores { totalQuizzes := 60, goalPercentage := 9/10, quizzesTaken := 40, aScores := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_must_ace_all_remaining_quizzes_l1188_118898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l1188_118814

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the center of a circumscribed circle with radius r. -/
noncomputable def CenterOfCircumscribedCircle (r : ℝ) : Point :=
  ⟨0, r⟩  -- Placeholder implementation

/-- Represents the center of an inscribed circle with radius ρ. -/
noncomputable def CenterOfInscribedCircle (ρ : ℝ) : Point :=
  ⟨0, ρ⟩  -- Placeholder implementation

/-- Predicate to check if a real number represents the distance between two points. -/
def IsDistance (d : ℝ) (p q : Point) : Prop :=
  (p.x - q.x)^2 + (p.y - q.y)^2 = d^2

/-- For any isosceles triangle, given the radius of its circumscribed circle and inscribed circle,
    this theorem states the distance between the centers of these circles. -/
theorem distance_between_centers (r ρ : ℝ) (hr : r > 0) (hρ : ρ > 0) :
  ∃ (d : ℝ), d = Real.sqrt (r * (r - 2 * ρ)) ∧ 
  d ≥ 0 ∧
  IsDistance d (CenterOfCircumscribedCircle r) (CenterOfInscribedCircle ρ) := by
  sorry  -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l1188_118814
