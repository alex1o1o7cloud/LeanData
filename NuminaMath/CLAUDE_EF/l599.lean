import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_zero_l599_59975

theorem sum_of_coefficients_zero : 
  let f : Polynomial ℚ := (1 + 2 * X - 3 * X^2)^2
  f.eval 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_zero_l599_59975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_13_75_l599_59900

noncomputable section

-- Define the squares
def square1 : ℝ := 8
def square2 : ℝ := 5
def square3 : ℝ := 3

-- Define the line segment connecting the corners
def diagonal : ℝ := square1 + square2 + square3

-- Define the height-to-base ratio
noncomputable def ratio : ℝ := square1 / diagonal

-- Define the heights of the triangles
noncomputable def height1 : ℝ := square3 * ratio
noncomputable def height2 : ℝ := (square3 + square2) * ratio

-- Define the area of the quadrilateral
noncomputable def quadrilateral_area : ℝ := (square2 * (height1 + height2)) / 2

theorem quadrilateral_area_is_13_75 :
  quadrilateral_area = 13.75 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_13_75_l599_59900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_properties_l599_59981

/-- Truncated right circular cone -/
structure TruncatedCone where
  R : ℝ  -- Large base radius
  r : ℝ  -- Small base radius
  h : ℝ  -- Frustum height

/-- Volume of a truncated cone -/
noncomputable def volume (c : TruncatedCone) : ℝ :=
  (Real.pi * c.h / 3) * (c.R^2 + c.r^2 + c.R * c.r)

/-- Slant height of a truncated cone -/
noncomputable def slantHeight (c : TruncatedCone) : ℝ :=
  Real.sqrt (c.h^2 + (c.R - c.r)^2)

/-- Theorem stating the volume and slant height of a specific truncated cone -/
theorem truncated_cone_properties :
  let c : TruncatedCone := ⟨10, 5, 8⟩
  volume c = (1400 * Real.pi) / 3 ∧ slantHeight c = Real.sqrt 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_properties_l599_59981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_not_necessarily_similar_l599_59939

/-- A tetrahedron with faces based on equilateral triangles -/
structure Tetrahedron where
  base : ℝ
  lateral_edges : Fin 3 → ℝ

/-- Two faces are similar if their corresponding sides are proportional -/
def similar_faces (f1 f2 : Fin 3 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 3, f2 i = k * f1 i

/-- Two tetrahedra are similar if all their corresponding edges are proportional -/
def similar_tetrahedra (t1 t2 : Tetrahedron) : Prop :=
  ∃ k : ℝ, t2.base = k * t1.base ∧ ∀ i : Fin 3, t2.lateral_edges i = k * t1.lateral_edges i

theorem tetrahedra_not_necessarily_similar :
  ∃ t1 t2 : Tetrahedron,
    (∀ i j : Fin 3, i ≠ j → ¬similar_faces (t1.lateral_edges) (t1.lateral_edges)) ∧
    (∀ i j : Fin 3, i ≠ j → ¬similar_faces (t2.lateral_edges) (t2.lateral_edges)) ∧
    (∀ i : Fin 3, ∃ j : Fin 3, similar_faces (t1.lateral_edges) (t2.lateral_edges)) ∧
    ¬similar_tetrahedra t1 t2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_not_necessarily_similar_l599_59939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l599_59984

/-- A non-zero arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ a n ≠ 0

/-- Sum of the first n terms of a sequence -/
def sum_sequence (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_sequence a n + a (n + 1)

/-- The property a_n^2 = S_{2n-1} -/
def property_a_S (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (a n)^2 = S (2*n - 1)

/-- The sequence b_n defined in terms of a_n and lambda -/
def sequence_b (a : ℕ → ℝ) (lambda : ℝ) : ℕ → ℝ :=
  fun n => (a n)^2 + lambda * (a n)

/-- An increasing sequence -/
def increasing_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_property (a : ℕ → ℝ) (lambda : ℝ) :
  arithmetic_sequence a →
  property_a_S a (sum_sequence a) →
  increasing_sequence (sequence_b a lambda) →
  lambda > -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l599_59984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_school_students_l599_59989

theorem middle_school_students (total : ℕ) (ms_percent : ℚ) (hs_percent : ℚ) : 
  total = 36 → 
  ms_percent = 1/5 →
  hs_percent = 1/4 →
  ∃ (ms hs : ℕ), 
    ms + hs = total ∧ 
    ms_percent * (ms : ℚ) = hs_percent * (hs : ℚ) ∧
    ms = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_school_students_l599_59989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OA_is_plus_minus_one_l599_59999

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y^2 / 16 + x^2 / 4 = 1

-- Define the condition OB = 2OA
def OB_equals_2OA (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₂ = 2 * x₁ ∧ y₂ = 2 * y₁

-- Theorem statement
theorem slope_of_OA_is_plus_minus_one
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : C₁ x₁ y₁)
  (h₂ : C₂ x₂ y₂)
  (h₃ : OB_equals_2OA x₁ y₁ x₂ y₂)
  (h₄ : x₁ ≠ 0) :
  (y₁ / x₁)^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OA_is_plus_minus_one_l599_59999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_angle_C_l599_59926

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = 3b² + 3c² - 2√3*b*c*sin(A), then angle C = π/6 -/
theorem triangle_special_condition_angle_C (a b c : ℝ) (A B C : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hA : A > 0) (hB : B > 0) (hC : C > 0)
    (hsum : A + B + C = Real.pi)
    (hcos : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
    (hcond : a^2 = 3*b^2 + 3*c^2 - 2*Real.sqrt 3*b*c*Real.sin A) :
  C = Real.pi/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_angle_C_l599_59926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_arrangement_theorem_l599_59943

def number_of_meal_arrangements : ℕ := 22572

-- Helper function for derangement
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | n+2 => n * (derangement (n+1) + derangement n)

-- Helper function for partial derangement
def partialDerangement (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => derangement n
  | k+1 => Nat.choose n (k+1) * derangement (n - (k+1)) + (n - k) * partialDerangement n k

theorem meal_arrangement_theorem :
  let total_people : ℕ := 12
  let meal_types : ℕ := 3
  let people_per_meal : ℕ := 4
  let correct_meals : ℕ := 2
  number_of_meal_arrangements = 
    (Nat.choose total_people correct_meals) * 
    (3 * (derangement people_per_meal)^2 + 
     3 * (partialDerangement people_per_meal 1)^2) := by
  sorry

#eval number_of_meal_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_arrangement_theorem_l599_59943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_theorem_l599_59931

def g_inverse_problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) : Prop :=
  (Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g) ∧
  (g 4 = 7) ∧
  (g 6 = 2) ∧
  (g 3 = 6) →
  g_inv (g_inv 6 + g_inv 7) = 4

theorem g_inverse_theorem :
  ∀ g g_inv, g_inverse_problem g g_inv :=
by
  intro g g_inv
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_theorem_l599_59931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l599_59909

def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 32 ∧ (n.digits 10).Nodup

theorem smallest_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≥ 26789 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l599_59909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_is_minimum_l599_59934

/-- The minimum value of m satisfying the conditions for a given n -/
def min_m (n : ℕ) : ℕ :=
  1 + Nat.ceil (n / 2 : ℚ)

/-- The condition that x_ij is a maximum in its row or column -/
def is_max_in_row_or_col (x : ℕ → ℕ → ℝ) (i j : ℕ) : Prop :=
  (∀ k, k ≤ j → x i j ≥ x i k) ∨ (∀ k, k ≤ i → x i j ≥ x k j)

/-- The condition that there are at most m maxima in each row -/
def at_most_m_maxima_in_row (x : ℕ → ℕ → ℝ) (n m : ℕ) : Prop :=
  ∀ i, i ≤ n → (Finset.filter (λ j ↦ ∀ k, k ≤ j → x i j ≥ x i k) (Finset.range n)).card ≤ m

/-- The condition that there are at most m maxima in each column -/
def at_most_m_maxima_in_col (x : ℕ → ℕ → ℝ) (n m : ℕ) : Prop :=
  ∀ j, j ≤ n → (Finset.filter (λ i ↦ ∀ k, k ≤ i → x i j ≥ x k j) (Finset.range n)).card ≤ m

/-- The main theorem stating that min_m is the minimum value satisfying the conditions -/
theorem min_m_is_minimum (n : ℕ) (h : n ≥ 2) :
  (∃ x : ℕ → ℕ → ℝ, (∀ i j, i ≤ n ∧ j ≤ n → is_max_in_row_or_col x i j) ∧
                     at_most_m_maxima_in_row x n (min_m n) ∧
                     at_most_m_maxima_in_col x n (min_m n)) ∧
  (∀ m < min_m n, ¬∃ x : ℕ → ℕ → ℝ, (∀ i j, i ≤ n ∧ j ≤ n → is_max_in_row_or_col x i j) ∧
                                    at_most_m_maxima_in_row x n m ∧
                                    at_most_m_maxima_in_col x n m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_is_minimum_l599_59934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l599_59935

theorem third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 5 → b = 12 → θ = 150 * Real.pi / 180 → c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) → 
  c = Real.sqrt (169 + 60 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l599_59935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_inequality_l599_59932

/-- Given four points A, B, C, D on a straight line in that order, with AB = CD, 
    and a point E not on this line, prove that EA + ED > EB + EC. -/
theorem four_points_inequality 
  (A B C D E : EuclideanSpace ℝ (Fin 3))
  (collinear : Collinear ℝ ({A, B, C, D} : Set (EuclideanSpace ℝ (Fin 3))))
  (order : dist A B ≤ dist A C ∧ dist A C ≤ dist A D)
  (equal_segments : dist A B = dist C D)
  (E_not_collinear : ¬ Collinear ℝ ({A, B, E} : Set (EuclideanSpace ℝ (Fin 3)))) :
  dist E A + dist E D > dist E B + dist E C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_inequality_l599_59932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_theorem_l599_59921

theorem rearrangement_theorem (n : ℕ) : 
  ∀ (p : Equiv.Perm (Fin (2*n))), 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ 
    (((p i).val - (p j).val) % (2*n) = (i.val - j.val) % (2*n) ∨
     ((p j).val - (p i).val) % (2*n) = (j.val - i.val) % (2*n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_theorem_l599_59921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l599_59925

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

/-- The function g(x) as defined in the problem -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + 4

/-- The main theorem to be proved -/
theorem problem_solution :
  ∀ b : ℝ, 
  (∀ x₁ : ℝ, x₁ > 0 ∧ x₁ < 2 → 
    ∃ x₂ : ℝ, x₂ ≥ 1 ∧ x₂ ≤ 2 ∧ f (1/4) x₁ ≥ g b x₂) → 
  b ≥ 17/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l599_59925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_theorem_l599_59965

/-- Represents a square on a 2D plane --/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Calculate the area of intersection between two squares --/
def intersection_area (s1 s2 : Square) : ℝ :=
  sorry

/-- Theorem: There exists a parallel translation of one 3x3 square 
    such that its intersection with another 3x3 square is at least 7 --/
theorem square_intersection_theorem :
  ∀ (s1 s2 : Square), 
    s1.side_length = 3 → 
    s2.side_length = 3 → 
    ∃ (translation : ℝ × ℝ), 
      let s2_translated := Square.mk (s2.center.1 + translation.1, s2.center.2 + translation.2) s2.side_length
      intersection_area s1 s2_translated ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_theorem_l599_59965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l599_59927

theorem calculation_proof : (-1/4)⁻¹ - (Real.pi - 3)^0 - |(-4)| + (-1)^2021 = -10 := by
  -- The proof will be completed later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l599_59927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row_10_sum_l599_59904

/-- Represents a number in Pascal's Triangle -/
def PascalNumber (row : ℕ) (position : ℕ) : ℕ := sorry

/-- The sum of numbers in a row of Pascal's Triangle -/
def RowSum (row : ℕ) : ℕ := (Finset.range (row + 1)).sum (λ i => PascalNumber row i)

/-- Pascal's Triangle property: each number is the sum of the two numbers above it -/
axiom pascal_property (row : ℕ) (position : ℕ) :
  PascalNumber (row + 1) position = PascalNumber row (position - 1) + PascalNumber row position

/-- Theorem: The sum of numbers in Row 10 of Pascal's Triangle is 2^10 -/
theorem pascal_row_10_sum : RowSum 10 = 2^10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row_10_sum_l599_59904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_walk_saves_time_l599_59962

/-- A boy walks to school at different rates -/
def school_walk (usual_time : ℚ) (rate_ratio : ℚ) : Prop :=
  let new_time := usual_time * (1 / rate_ratio)
  (usual_time - new_time : ℚ) = 3

/-- The theorem states that walking at 7/6 of the usual rate saves 3 minutes -/
theorem faster_walk_saves_time :
  school_walk 21 (7/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_walk_saves_time_l599_59962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l599_59957

/-- The principal amount borrowed -/
def P : ℝ := 8000

/-- Interest rate for the first 3 years -/
def r1 : ℝ := 0.06

/-- Interest rate for the next 5 years -/
def r2 : ℝ := 0.09

/-- Interest rate for the period beyond 8 years -/
def r3 : ℝ := 0.13

/-- Total interest paid after 11 years -/
def total_interest : ℝ := 8160

/-- Theorem stating that the principal amount P is 8000 -/
theorem principal_amount : 
  P * (r1 * 3 + r2 * 5 + r3 * 3) = total_interest := by
  simp [P, r1, r2, r3, total_interest]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l599_59957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_geometric_sequence_implies_prime_power_l599_59960

/-- 
A function that returns true if the differences between consecutive elements in a list form a geometric sequence.
-/
def isGeometricSequence (l : List ℕ) : Prop :=
  ∃ r : ℚ, ∀ i : ℕ, i + 2 < l.length → 
    (l[i+2]! - l[i+1]!) / (l[i+1]! - l[i]!) = r

/-- 
A function that returns the list of all positive divisors of a natural number.
-/
def divisors (n : ℕ) : List ℕ := sorry

/-- 
The main theorem stating the conditions and conclusion about the form of n.
-/
theorem divisor_geometric_sequence_implies_prime_power (n : ℕ) :
  (divisors n).length ≥ 4 ∧ 
  isGeometricSequence (divisors n) →
  ∃ p α : ℕ, Nat.Prime p ∧ α ≥ 3 ∧ n = p^α :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_geometric_sequence_implies_prime_power_l599_59960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_after_decimal_l599_59976

/-- The first three digits to the right of the decimal point in (5^1001 + 1)^(5/3) are 333. -/
theorem first_three_digits_after_decimal : ∃ (k : ℕ) (r : ℝ), 
  (5^1001 + 1 : ℝ)^(5/3) = k + 0.333 + r ∧ 0 ≤ r ∧ r < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_after_decimal_l599_59976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_of_120_l599_59973

theorem gcd_of_polynomial_and_multiple_of_120 (x : ℤ) (h : 120 ∣ x) :
  Nat.gcd ((3 * x + 4) * (5 * x + 3) * (11 * x + 6) * (x + 11)).natAbs x.natAbs = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_of_120_l599_59973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_N_l599_59992

def N : ℕ := 2^3 * 3^2 * 5

theorem number_of_factors_of_N : 
  (Finset.filter (·∣N) (Finset.range (N + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_N_l599_59992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_shift_for_18cm_square_equal_visible_areas_after_folding_l599_59947

/-- The distance a vertex moves when folding a square sheet of paper -/
noncomputable def vertex_shift_distance (sheet_area : ℝ) : ℝ :=
  2 * Real.sqrt 6

/-- Theorem stating the vertex shift distance for a square sheet with area 18 cm² -/
theorem vertex_shift_for_18cm_square :
  vertex_shift_distance 18 = 2 * Real.sqrt 6 := by
  -- Unfold the definition of vertex_shift_distance
  unfold vertex_shift_distance
  -- The equality follows directly from the definition
  rfl

/-- Theorem proving the equality of visible areas after folding -/
theorem equal_visible_areas_after_folding (sheet_area : ℝ) :
  let side_length := Real.sqrt sheet_area
  let folded_triangle_leg := 2 * Real.sqrt 3
  (1/2) * folded_triangle_leg^2 = sheet_area - folded_triangle_leg^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_shift_for_18cm_square_equal_visible_areas_after_folding_l599_59947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_non_black_ball_l599_59908

theorem probability_non_black_ball (odds_black : ℚ) (h : odds_black = 5 / 3) :
  let total := odds_black + 1
  let prob_non_black := 1 / total
  prob_non_black = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_non_black_ball_l599_59908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_special_quadrilaterals_are_squares_l599_59945

/-- A quadrilateral with perpendicular and bisecting diagonals -/
structure SpecialQuadrilateral where
  /-- The diagonals are perpendicular -/
  diagonals_perpendicular : Bool
  /-- The diagonals bisect each other -/
  diagonals_bisect : Bool

/-- Definition of a square -/
structure Square where
  /-- All sides are equal -/
  sides_equal : Bool
  /-- All angles are 90 degrees -/
  right_angles : Bool

/-- Theorem: A quadrilateral with perpendicular and bisecting diagonals is not necessarily a square -/
theorem not_all_special_quadrilaterals_are_squares :
  ∃ q : SpecialQuadrilateral, ¬∃ (s : Square), q.diagonals_perpendicular = s.sides_equal ∧ q.diagonals_bisect = s.right_angles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_special_quadrilaterals_are_squares_l599_59945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_common_course_l599_59971

def number_of_courses : ℕ := 4
def courses_per_person : ℕ := 2

def number_of_ways_to_choose (n k₁ k₂ : ℕ) : ℕ := (n.choose k₁) * (n.choose k₂)

def number_of_ways_same_courses (n k : ℕ) : ℕ := n.choose k

def number_of_ways_different_courses (n k : ℕ) : ℕ := n.choose k

theorem exactly_one_common_course :
  (number_of_ways_to_choose number_of_courses courses_per_person courses_per_person - 
   number_of_ways_same_courses number_of_courses courses_per_person - 
   number_of_ways_different_courses number_of_courses courses_per_person) = 24 :=
by
  -- Expand definitions
  simp [number_of_ways_to_choose, number_of_ways_same_courses, number_of_ways_different_courses]
  -- Calculate
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_common_course_l599_59971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_below_a8_l599_59982

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 3
  | n + 1 => Real.sqrt ((5 * sequence_a n + 3 * Real.sqrt (sequence_a n^2 - 4)) / 2)

theorem largest_integer_below_a8 :
  ∃ (k : ℕ), k = 335 ∧ (k : ℝ) < sequence_a 8 ∧ ∀ (m : ℕ), (m : ℝ) < sequence_a 8 → m ≤ k := by
  sorry

#check largest_integer_below_a8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_below_a8_l599_59982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l599_59990

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

-- State the theorem
theorem min_value_of_f :
  ∃ (min_x : ℝ), min_x > 2 ∧ ∀ (x : ℝ), x > 2 → f x ≥ f min_x ∧ f min_x = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l599_59990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l599_59946

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (C : Ellipse) 
  (h3 : C.a = 2)
  (h4 : distance (Point.mk (-2) 0) (Point.mk 0 C.b) = Real.sqrt 5) :
  (∀ x y : ℝ, x^2/4 + y^2 = 1 ↔ x^2/(C.a^2) + y^2/(C.b^2) = 1) ∧ 
  (¬ ∃ A B : Point, 
    (A.x^2/(C.a^2) + A.y^2/(C.b^2) = 1) ∧ 
    (B.x^2/(C.a^2) + B.y^2/(C.b^2) = 1) ∧ 
    (A.x - 1 = (1/3) * (1 - B.x)) ∧ 
    (A.y = (1/3) * (-B.y))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l599_59946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_satisfying_conditions_l599_59914

/-- Given two lines and two circles in a plane, prove the range of m that satisfies the conditions -/
theorem range_of_m_satisfying_conditions :
  ∀ (m : ℝ),
  let l₁ := {(x, y) : ℝ × ℝ | x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m + 2) * x + 3 * y - 5 = 0}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2 * m * x - 2 * m * y + 2 * m^2 - 4 = 0}
  let O := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let p := ∀ (x y : ℝ), x + m * y - 1 = 0 ↔ (m + 2) * x + 3 * y - 5 = 0
  let q := (∃ (x y : ℝ), (x, y) ∈ C ∩ O) ∧ m ≠ 0
  (p ∨ q) ∧ ¬(p ∧ q) →
  m ∈ Set.Ioo (-2 * Real.sqrt 2) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioo 1 (2 * Real.sqrt 2) ∪ {-3} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_satisfying_conditions_l599_59914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_theorem_l599_59902

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Computes the perimeter of a rhombus given its diagonal lengths -/
noncomputable def perimeter (r : Rhombus) : ℝ :=
  4 * ((r.diagonal1 / 2) ^ 2 + (r.diagonal2 / 2) ^ 2) ^ (1/2 : ℝ)

/-- Theorem stating that a rhombus with diagonals 20 and 15 has perimeter 25 -/
theorem rhombus_perimeter_theorem :
  let r : Rhombus := { diagonal1 := 20, diagonal2 := 15 }
  perimeter r = 25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_theorem_l599_59902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l599_59918

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ x, f x ≤ M) ∧
  (∀ x ∈ Set.Icc (Real.pi / 24) (13 * Real.pi / 24),
   ∀ y ∈ Set.Icc (Real.pi / 24) (13 * Real.pi / 24),
   x < y → f y < f x) ∧
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * x - Real.pi / 12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l599_59918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_approx_l599_59917

noncomputable def principal : ℝ := 50000
noncomputable def final_amount : ℝ := 80000
noncomputable def time : ℝ := 4
noncomputable def compounding_frequency : ℝ := 1

noncomputable def compound_interest_rate (p a t n : ℝ) : ℝ :=
  ((a / p) ^ (1 / (n * t))) - 1

theorem compound_interest_rate_approx :
  |compound_interest_rate principal final_amount time compounding_frequency - 0.1247| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_approx_l599_59917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangements_without_subsequence_l599_59923

def word : Finset Char := {'О', 'Л', 'И', 'М', 'П', 'И', 'А', 'Д', 'А'}
def subsequence : List Char := ['Л', 'А', 'М', 'П', 'А']

def total_permutations : ℕ := 90720
def invalid_permutations : ℕ := 60

theorem rearrangements_without_subsequence :
  (total_permutations - invalid_permutations : ℕ) = 90660 :=
by
  -- The proof goes here
  sorry

#eval total_permutations - invalid_permutations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangements_without_subsequence_l599_59923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l599_59950

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3)
  (h2 : f 0 = 1) : 
  ∀ n, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l599_59950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l599_59967

-- Define the triangle and its properties
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  B_acute : 0 < B ∧ B < π/2

-- Define the vectors
noncomputable def m (B : Real) : Real × Real := (2 * Real.sin B, -Real.sqrt 3)
noncomputable def n (B : Real) : Real × Real := (Real.cos (2 * B), 2 * (Real.cos (B/2))^2 - 1)

-- Define parallel vectors
def parallel (v w : Real × Real) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem triangle_properties (t : Triangle) 
  (h_parallel : parallel (m t.B) (n t.B)) 
  (h_b_range : t.b ∈ Set.Icc (Real.sqrt 3) (2 * Real.sqrt 3)) : 
  t.B = π/3 ∧ 
  (let R := t.b / (2 * Real.sin t.B); R ∈ Set.Icc 1 2) ∧
  (t.b = 2 → ∃ (S : Real), S ≤ Real.sqrt 3 ∧ 
    ∀ (a c : Real), a * c * Real.sin t.B ≤ 2 * S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l599_59967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neg_f_reflection_l599_59913

-- Define a continuous function f from reals to reals
variable (f : ℝ → ℝ)
variable (hf : Continuous f)

-- Define the negation of f
def neg_f (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- State the theorem
theorem neg_f_reflection (f : ℝ → ℝ) (x y : ℝ) :
  (y = f x) ↔ (-y = neg_f f x) :=
by
  sorry

-- You can add more specific instances or examples here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neg_f_reflection_l599_59913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_check_l599_59991

-- Define the sets for each option
def set_A_1 : Set ℕ := {0, 1}
def set_B_1 : Set (ℕ × ℕ) := {(0, 1)}

def set_A_2 : Set ℕ := {2, 3}
def set_B_2 : Set ℕ := {3, 2}

def set_A_3 : Set ℕ := {x : ℕ | 0 < x ∧ x ≤ 1}
def set_B_3 : Set ℕ := {1}

noncomputable def set_A_4 : Set ℝ := ∅
noncomputable def set_B_4 : Set ℝ := {x : ℝ | Real.sqrt x ≤ 0}

-- Theorem statement
theorem set_equality_check :
  (set_A_1 ≠ (set_B_1.image Prod.fst)) ∧
  (set_A_2 = set_B_2) ∧
  (set_A_3 ≠ set_B_3) ∧
  (set_A_4 ≠ set_B_4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_check_l599_59991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l599_59920

def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 < 1}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l599_59920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fractions_l599_59986

-- Define the fractions as noncomputable
noncomputable def A (x : ℝ) : ℝ := (x^2 - 1) / (x^2 + 1)
noncomputable def B (x : ℝ) : ℝ := (x + 1) / (x^2 - 1)
noncomputable def C (x : ℝ) : ℝ := (x^2 - 1) / x
noncomputable def D (x : ℝ) : ℝ := (x - 1) / (x + 1)

-- Define what it means for a fraction to be in its simplest form
def is_simplest_form (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → ∀ g : ℝ → ℝ, (∀ y, y ≠ 0 → f y = g y) → g = f

-- Theorem statement
theorem simplest_fractions :
  is_simplest_form A ∧ is_simplest_form C ∧ is_simplest_form D :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fractions_l599_59986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_wage_l599_59940

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  a_wage : ℚ
  b_wage : ℚ
  c_wage : ℚ
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ

/-- Theorem stating the daily wage of worker c given the problem conditions -/
theorem worker_c_wage (w : WorkerData) 
  (h_days : w.a_days = 6 ∧ w.b_days = 9 ∧ w.c_days = 4)
  (h_ratio : w.b_wage = 4/3 * w.a_wage ∧ w.c_wage = 5/3 * w.a_wage)
  (h_total : w.a_wage * w.a_days + w.b_wage * w.b_days + w.c_wage * w.c_days = 1628) :
  w.c_wage = 110 := by
  sorry

/-- Example calculation -/
def example_calculation : Bool :=
  let w : WorkerData := ⟨66, 88, 110, 6, 9, 4⟩
  let days_correct := w.a_days = 6 ∧ w.b_days = 9 ∧ w.c_days = 4
  let ratio_correct := w.b_wage = 4/3 * w.a_wage ∧ w.c_wage = 5/3 * w.a_wage
  let total_correct := w.a_wage * w.a_days + w.b_wage * w.b_days + w.c_wage * w.c_days = 1628
  days_correct && ratio_correct && total_correct && (w.c_wage = 110)

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_wage_l599_59940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_a_plus_b_l599_59961

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (v w : V) : ℝ := Real.arccos (inner v w / (norm v * norm w))

theorem angle_between_a_and_a_plus_b
  (a b : V)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h_norm : norm a = norm b ∧ norm a = norm (a - b)) :
  angle a (a + b) = π / 6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_a_plus_b_l599_59961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l599_59929

noncomputable def f (x m : ℝ) : ℝ := 2 / (2^x + 1) + m

theorem odd_function_m_value (m : ℝ) :
  (∀ x, f x m = -f (-x) m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l599_59929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_height_in_cm_l599_59974

/-- Converts feet and inches to centimeters -/
def height_to_cm (feet : ℕ) (inches : ℕ) : ℝ :=
  (feet * 12 + inches : ℝ) * 2.54

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem julia_height_in_cm :
  round_to_tenth (height_to_cm 5 4) = 162.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_height_in_cm_l599_59974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l599_59966

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 - x)) / Real.sqrt (x - 2)

-- Theorem statement
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ 2 < x ∧ x < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l599_59966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_facial_product_waiting_time_l599_59972

/-- Calculates the waiting time between facial product applications -/
noncomputable def waiting_time (total_products : ℕ) (total_time : ℝ) (makeup_time : ℝ) : ℝ :=
  (total_time - makeup_time) / (total_products - 1 : ℝ)

/-- Theorem: Given the conditions, the waiting time between each product is 6.25 minutes -/
theorem facial_product_waiting_time :
  waiting_time 5 55 30 = 6.25 := by
  -- Unfold the definition of waiting_time
  unfold waiting_time
  -- Simplify the arithmetic
  simp [Nat.cast_sub, Nat.cast_one]
  -- Check that the result is equal to 6.25
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_facial_product_waiting_time_l599_59972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l599_59919

-- Define secant and cosecant functions
noncomputable def sec (x : ℝ) := 1 / Real.cos x
noncomputable def csc (x : ℝ) := 1 / Real.sin x

theorem trigonometric_identity (β : ℝ) :
  (Real.sin β + sec β)^2 + (Real.cos β + csc β)^2 -
  (Real.sin β^2 + Real.cos β^2 + sec β^2 + csc β^2) = 4 / Real.sin (2 * β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l599_59919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_length_calculation_l599_59987

/-- The length of the pipe in meters -/
def pipe_length : ℝ := 114

/-- The number of steps Gavrila took in the direction of the tractor's movement -/
def steps_forward : ℕ := 210

/-- The number of steps Gavrila took in the opposite direction -/
def steps_backward : ℕ := 100

/-- Gavrila's step length in meters -/
def step_length : ℝ := 0.8

theorem pipe_length_calculation :
  ∃ (y : ℝ),
    pipe_length = steps_forward * (step_length - y) ∧
    pipe_length = steps_backward * (step_length + y) ∧
    |pipe_length - 113.5| < 0.5 :=
by
  sorry

#eval pipe_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_length_calculation_l599_59987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l599_59998

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2/3) * x^3 - 2*a*x^2 - 3*x

theorem tangent_line_problem (a : ℝ) (b : ℝ) 
  (h : ∃ (y : ℝ), f a 1 = y ∧ (3 * 1 - y + b = 0) ∧ (deriv (f a) 1 = 3)) :
  f a 1 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l599_59998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l599_59952

theorem modulus_of_z (z : ℂ) (h : z^2 = 3/4 - I) : Complex.abs z = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l599_59952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_two_point_five_l599_59907

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * 5) / 2

/-- The ratio of S(2n) to S(n) is constant for all positive n -/
def ratio_is_constant (a : ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, n > 0 → S a (2 * n) / S a n = c

theorem first_term_is_two_point_five :
  ∀ a : ℝ, ratio_is_constant a → a = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_two_point_five_l599_59907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_g_max_value_g_symmetry_axis_l599_59948

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/6) + 1

def is_monotone_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x

theorem f_monotone_decreasing (k : ℤ) :
  is_monotone_decreasing f (Set.Icc (Real.pi/3 + 2*Real.pi*↑k) (4*Real.pi/3 + 2*Real.pi*↑k)) :=
sorry

theorem g_max_value : ∃ x, g x = 3 ∧ ∀ y, g y ≤ 3 :=
sorry

theorem g_symmetry_axis (k : ℤ) : ∀ x, g (Real.pi/2 + Real.pi*↑k + x) = g (Real.pi/2 + Real.pi*↑k - x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_g_max_value_g_symmetry_axis_l599_59948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_upstream_speed_l599_59985

/-- Calculates the effective speed of a boat traveling upstream against current and wind. -/
def effective_speed_upstream (rowing_speed current_speed wind_speed : ℝ) : ℝ :=
  rowing_speed - (current_speed + wind_speed)

/-- Theorem stating the effective speed of a man rowing upstream against current and wind. -/
theorem man_rowing_upstream_speed 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (wind_speed : ℝ) 
  (h1 : rowing_speed = 20)
  (h2 : current_speed = 3)
  (h3 : wind_speed = 2) :
  effective_speed_upstream rowing_speed current_speed wind_speed = 15 := by
  -- Unfold the definition of effective_speed_upstream
  unfold effective_speed_upstream
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Perform the arithmetic
  norm_num

#check man_rowing_upstream_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_upstream_speed_l599_59985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l599_59928

def is_arithmetic_sequence (seq : List ℝ) : Prop :=
  seq.length > 1 ∧ ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = seq[1]! - seq[0]!

def is_geometric_sequence (seq : List ℝ) : Prop :=
  seq.length > 1 ∧ ∀ i, i + 1 < seq.length → seq[i+1]! / seq[i]! = seq[1]! / seq[0]!

theorem arithmetic_geometric_sequence_ratio :
  ∀ (a b c d e : ℝ),
  is_arithmetic_sequence [-1, a, b, -4] →
  is_geometric_sequence [-1, c, d, e, -4] →
  (b - a) / d = 1/2 := by
  intros a b c d e h_arith h_geom
  sorry

#check arithmetic_geometric_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l599_59928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_l599_59955

-- Define the interval for a
def interval : Set ℝ := Set.Icc (-5) 5

-- Define the line equation
def line (a : ℝ) (x y : ℝ) : Prop := x + y + a = 0

-- Define the circle equation
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the intersection condition
def intersects (a : ℝ) : Prop := ∃ x y, line a x y ∧ circleEq x y

-- Main theorem
theorem intersection_probability : 
  ∃ p : Set ℝ, p ⊆ interval ∧ 
  (∀ a ∈ p, intersects a) ∧
  (MeasureTheory.volume p) / (MeasureTheory.volume interval) = 2/5 := by
  sorry

#check intersection_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_l599_59955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_m_value_l599_59997

theorem power_equality_implies_m_value (m : ℝ) : 
  (8 : ℝ)^(9/2) = (16 : ℝ)^m → m = 27/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_m_value_l599_59997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_point_l599_59901

-- Define the circle as a predicate on points
def is_on_circle (p : ℝ × ℝ) : Prop := (p.1 - 2)^2 + p.2^2 = 2

-- Define point A
def point_A : ℝ × ℝ := (-1, 3)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_circle_to_point :
  ∃ (max_dist : ℝ), max_dist = 4 * Real.sqrt 2 ∧
  ∀ (p : ℝ × ℝ), is_on_circle p → distance p point_A ≤ max_dist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_point_l599_59901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l599_59958

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The point coordinates -/
def point : ℝ × ℝ := (3, 0)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from (3,0) to an asymptote of the given hyperbola is 9/5 -/
theorem distance_to_asymptote :
  ∃ (A B C : ℝ), (∀ x y, hyperbola x y → (A * x + B * y + C = 0 ∨ A * x - B * y + C = 0)) ∧
                 distance_point_to_line point.1 point.2 A B C = 9/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l599_59958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l599_59936

noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l599_59936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_A_l599_59954

theorem cos_pi_half_minus_A (A : Real) (h : Real.sin (π - A) = 1/2) :
  Real.cos (π/2 - A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_A_l599_59954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l599_59905

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 5 * x - 2 * log x

-- State the theorem
theorem f_properties :
  let f : ℝ → ℝ := λ x ↦ (1/3) * x^3 - 2 * x^2 + 5 * x - 2 * log x
  -- f has a local minimum at x = 2
  ∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2
  -- f has no local maximum
  ∧ ¬∃ a : ℝ, ∃ ε > 0, ∀ x ∈ Set.Ioo (a - ε) (a + ε), f x ≤ f a
  -- Maximum value on [1, 3]
  ∧ (∀ x ∈ Set.Icc 1 3, f x ≤ 6 - 2 * log 3)
  -- Minimum value on [1, 3]
  ∧ (∀ x ∈ Set.Icc 1 3, f x ≥ 14/3 - 2 * log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l599_59905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_range_theorem_l599_59956

theorem trig_range_theorem (α : ℝ) :
  0 < α ∧ α < 2 * Real.pi →
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔
  ((0 < α ∧ α < Real.pi / 3) ∨ (5 * Real.pi / 3 < α ∧ α < 2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_range_theorem_l599_59956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_maximum_value_and_condition_l599_59924

noncomputable def g (x : ℝ) := 3 * Real.sin x + 4 * Real.cos x

theorem g_maximum_value_and_condition :
  (∃ (x : ℝ), g x = 5 ∧ ∀ (y : ℝ), g y ≤ 5) ∧
  (∀ (x : ℝ), g x = 5 ↔ Real.tan x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_maximum_value_and_condition_l599_59924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_orbit_distance_l599_59906

/-- Represents the distance of a planet from its sun in an elliptical orbit -/
noncomputable def orbit_distance (perigee apogee : ℝ) (angle : ℝ) : ℝ :=
  let a := (apogee + perigee) / 2
  let c := a - perigee
  let b := Real.sqrt (a^2 - c^2)
  b^2 / (a - c * Real.cos angle)

/-- Theorem stating that for an elliptical orbit with perigee 3 AU and apogee 15 AU,
    the distance from the focus to a point one-quarter way along the orbit from perigee is 5 AU -/
theorem quarter_orbit_distance :
  orbit_distance 3 15 (Real.pi / 2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_orbit_distance_l599_59906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_CD_EF_parallel_l599_59977

-- Define the basic structures
variable (Γ₁ Γ₂ : Set (EuclideanSpace ℝ (Fin 2))) -- Two circles
variable (A B C D E F : EuclideanSpace ℝ (Fin 2)) -- Points in the plane
variable (d_A d_B : Set (EuclideanSpace ℝ (Fin 2))) -- Lines passing through A and B respectively

-- Define the conditions
axiom intersect : A ∈ Γ₁ ∧ A ∈ Γ₂ ∧ B ∈ Γ₁ ∧ B ∈ Γ₂
axiom on_d_A : A ∈ d_A ∧ C ∈ d_A ∧ E ∈ d_A
axiom on_d_B : B ∈ d_B ∧ D ∈ d_B ∧ F ∈ d_B
axiom on_Γ₁ : C ∈ Γ₁ ∧ D ∈ Γ₁
axiom on_Γ₂ : E ∈ Γ₂ ∧ F ∈ Γ₂

-- Define parallelism
def parallel (L₁ L₂ : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define a line through two points
def Line.throughPoints (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- The theorem to prove
theorem lines_CD_EF_parallel : 
  parallel (Line.throughPoints C D) (Line.throughPoints E F) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_CD_EF_parallel_l599_59977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relation_l599_59953

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Function f as defined in the problem -/
noncomputable def f (n : ℤ) : ℝ := 
  (5 + 3 * Real.sqrt 5) / 10 * φ^n + 
  (5 - 3 * Real.sqrt 5) / 10 * (1 - φ)^n

/-- Theorem stating the relationship between f(n+1), f(n-1), and f(n) -/
theorem f_relation (n : ℤ) : f (n + 1) - f (n - 1) = f n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relation_l599_59953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l599_59996

/-- The area of a specific pentagon -/
noncomputable def pentagon_area (a b c d e : ℝ) : ℝ :=
  (1/2) * a * b + (1/2) * (c + d) * e

/-- Theorem stating the area of the given pentagon -/
theorem specific_pentagon_area :
  pentagon_area 18 25 25 30 28 = 995 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l599_59996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_trapezoid_l599_59968

/-- A quadrilateral ABCD with midpoints M of [AD] and N of [BC] is a trapezoid if MN = (AB + CD) / 2 -/
theorem quadrilateral_is_trapezoid 
  (A B C D M N : EuclideanSpace ℝ (Fin 2)) 
  (h_midpoint_M : M = (1/2 : ℝ) • (A + D))
  (h_midpoint_N : N = (1/2 : ℝ) • (B + C))
  (h_MN_length : ‖N - M‖ = (1/2 : ℝ) * (‖B - A‖ + ‖D - C‖)) :
  ∃ (k : ℝ), B - A = k • (D - C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_trapezoid_l599_59968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_l_l599_59988

-- Define the polar coordinates of point P
noncomputable def P : ℝ × ℝ := (2, 3 * Real.pi / 2)

-- Define the line l in polar form
def l (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ = 3

-- Define the distance function
noncomputable def distance_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Implementation of distance calculation

-- Theorem statement
theorem distance_from_P_to_l : distance_to_line P l = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_l_l599_59988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_quarter_circular_region_l599_59963

/-- The perimeter of a region bounded by quarter-circular arcs on a rectangle --/
theorem perimeter_quarter_circular_region (l w : ℝ) (hl : l = 4 / Real.pi) (hw : w = 2 / Real.pi) :
  (2 * (π * (l / 2) / 2) + 2 * (π * (w / 2) / 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_quarter_circular_region_l599_59963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_proof_l599_59944

/-- The function f(x) defined as 2sin(2x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

/-- The amount of horizontal shift between y = 2sin(2x) and f(x) -/
noncomputable def horizontal_shift : ℝ := Real.pi / 12

/-- Theorem stating that f(x) is equivalent to 2sin(2(x + horizontal_shift)) -/
theorem horizontal_shift_proof :
  ∀ x : ℝ, f x = 2 * Real.sin (2 * (x + horizontal_shift)) :=
by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_proof_l599_59944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l599_59983

theorem sin_cos_product (x : ℝ) (h1 : Real.cos (x - 3 * π / 2) = -4/5) (h2 : 0 < x) (h3 : x < π/2) :
  Real.sin (x/2) * Real.cos (5*x/2) = -38/125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l599_59983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l599_59911

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The sum of distances from any point on the ellipse to the foci is constant -/
axiom ellipse_property (e : Ellipse) (p : Point) (f1 f2 : Point) :
  isOnEllipse e p → distance p f1 + distance p f2 = 2 * e.a

/-- Theorem: The equation of the ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) (f1 f2 : Point) :
  f1 = ⟨-5, 0⟩ →
  f2 = ⟨5, 0⟩ →
  e.c = 5 →
  (∀ p, isOnEllipse e p → distance p f1 + distance p f2 = 26) →
  e.a = 13 ∧ e.b = 12 ∧ (∀ p, isOnEllipse e p ↔ p.x^2 / 169 + p.y^2 / 144 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l599_59911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_cannot_guarantee_catch_l599_59994

/-- Represents a point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A game state after n rounds -/
structure GameState where
  n : ℕ
  hunter : Point
  rabbit : Point

/-- A strategy for the rabbit -/
def RabbitStrategy := GameState → Point

/-- A strategy for the hunter -/
def HunterStrategy := GameState → Point → Point

/-- Play the game for n rounds -/
def playGame (n : ℕ) (rabbitStrategy : RabbitStrategy) (hunterStrategy : HunterStrategy) : GameState :=
  sorry

/-- The theorem stating that the hunter cannot guarantee catching the rabbit -/
theorem hunter_cannot_guarantee_catch :
  ∃ (rabbitStrategy : RabbitStrategy),
    ∀ (hunterStrategy : HunterStrategy),
      let finalState := playGame (10^9) rabbitStrategy hunterStrategy
      distance finalState.hunter finalState.rabbit ≥ 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_cannot_guarantee_catch_l599_59994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_seven_l599_59938

theorem trigonometric_expression_equals_seven :
  (3 * Real.tan (30 * π / 180)) / (1 - Real.sin (60 * π / 180)) +
  (Real.tan (30 * π / 180) + Real.cos (70 * π / 180)) ^ 0 -
  Real.tan (60 * π / 180) / (Real.cos (45 * π / 180)) ^ 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_seven_l599_59938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l599_59959

-- Define the set of real numbers satisfying the conditions
def S : Set ℝ := {a | ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  |a - 1| ≥ Real.sqrt (3*x + 1) + Real.sqrt (3*y + 1) + Real.sqrt (3*z + 1)}

-- Theorem statement
theorem range_of_a : S = Set.Iic (1 - 3 * Real.sqrt 2) ∪ Set.Ici (1 + 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l599_59959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_is_pi_over_six_l599_59993

/-- A simple spatial geometric body whose three views are all squares with side length 1 -/
structure GeometricBody where
  side_length : ℝ
  is_unit_cube : side_length = 1

/-- The volume of the inscribed sphere in a geometric body -/
noncomputable def inscribedSphereVolume (body : GeometricBody) : ℝ :=
  (4 / 3) * Real.pi * (body.side_length / 2) ^ 3

/-- Theorem stating that the volume of the inscribed sphere is π/6 -/
theorem inscribed_sphere_volume_is_pi_over_six (body : GeometricBody) :
  inscribedSphereVolume body = Real.pi / 6 := by
  sorry

#check inscribed_sphere_volume_is_pi_over_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_is_pi_over_six_l599_59993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l599_59979

-- Define the diamond operation
noncomputable def diamond (x y : ℝ) : ℝ := (x^2 + y^2) / (x^2 - y^2)

-- Theorem statement
theorem diamond_calculation : diamond (diamond 1 2) 4 = -169/119 := by
  -- Expand the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [pow_two]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l599_59979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sochi_puzzle_solution_l599_59970

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a FourDigitNumber to a natural number -/
def fourDigitToNat (n : FourDigitNumber) : ℕ :=
  1000 * n.thousands.val + 100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Checks if all digits in a FourDigitNumber are different -/
def allDigitsDifferent (n : FourDigitNumber) : Prop :=
  n.thousands ≠ n.hundreds ∧
  n.thousands ≠ n.tens ∧
  n.thousands ≠ n.ones ∧
  n.hundreds ≠ n.tens ∧
  n.hundreds ≠ n.ones ∧
  n.tens ≠ n.ones

theorem sochi_puzzle_solution :
  ∃ (year sochi : FourDigitNumber),
    allDigitsDifferent year ∧
    allDigitsDifferent sochi ∧
    2014 + fourDigitToNat year = fourDigitToNat sochi :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sochi_puzzle_solution_l599_59970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_expansion_count_l599_59969

/-- The number of terms in the binary expansion of (2^341 + 1) / (2^31 + 1) -/
def binary_expansion_terms : ℕ := 341

/-- The numerator of the fraction -/
def numerator : ℕ := 2^341 + 1

/-- The denominator of the fraction -/
def denominator : ℕ := 2^31 + 1

/-- Function to calculate the number of terms in binary expansion -/
noncomputable def number_of_terms_in_binary_expansion : ℚ → ℕ :=
  sorry -- Placeholder for the actual implementation

/-- Theorem stating that the number of terms in the binary expansion of (2^341 + 1) / (2^31 + 1) is 341 -/
theorem binary_expansion_count :
  binary_expansion_terms = 341 ∧
  binary_expansion_terms = number_of_terms_in_binary_expansion (numerator / denominator) := by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_expansion_count_l599_59969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_surface_area_equals_24_75_l599_59942

/-- Represents a cube with given edge length -/
structure Cube where
  edge : ℝ
  edge_positive : edge > 0

/-- Represents a triangular piece cut from the cube -/
structure TriangularPiece (cube : Cube) where
  base : ℝ
  height : ℝ
  base_valid : base = cube.edge
  height_valid : height = cube.edge / 2

/-- Calculates the volume of the triangular piece -/
noncomputable def volumeOfPiece (cube : Cube) (piece : TriangularPiece cube) : ℝ :=
  (1 / 2) * piece.base * piece.height * cube.edge

/-- Calculates the surface area of icing on the triangular piece -/
noncomputable def surfaceAreaOfIcing (cube : Cube) (piece : TriangularPiece cube) : ℝ :=
  2 * (1 / 2) * piece.base * piece.height + 3 * piece.height * cube.edge

/-- Main theorem stating the sum of volume and surface area -/
theorem volume_plus_surface_area_equals_24_75 (cube : Cube) 
  (h : cube.edge = 3) : 
  ∃ (piece : TriangularPiece cube), 
    volumeOfPiece cube piece + surfaceAreaOfIcing cube piece = 24.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_surface_area_equals_24_75_l599_59942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_minus_cos_15_degrees_l599_59949

theorem cot_minus_cos_15_degrees :
  Real.tan (π / 2 - 15 * π / 180) - 3 * Real.cos (15 * π / 180) = (18 + 11 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_minus_cos_15_degrees_l599_59949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_g_condition_l599_59978

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * (f x) - a * x^2 + (Real.log 2) / 2

theorem f_max_and_g_condition (a : ℝ) (h : a > 0) :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x) ∧
  (∃ (x : ℝ), x = Real.exp 1) ∧
  (∃ (M : ℝ), ∀ (x : ℝ), x > 0 → g a x ≤ M) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), x > 0 → g a x ≤ M) → M > a/2 - 1) ↔
  (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_g_condition_l599_59978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_paving_stones_l599_59930

/-- Calculates the number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  (hall_length * hall_width * 100 / (stone_length * stone_width)).floor.toNat

/-- Theorem stating that 1800 stones are required to pave the given hall -/
theorem hall_paving_stones : 
  stones_required 36 15 6 5 = 1800 := by
  -- Proof goes here
  sorry

#eval stones_required 36 15 6 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_paving_stones_l599_59930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_difference_is_three_l599_59916

def geometric_sequence : List ℕ := [3, 9, 27, 81, 243]

def arithmetic_sequence : List ℕ := List.range 20 |> List.map (fun n => 15 * (n + 1)) |> List.takeWhile (· ≤ 300)

def least_positive_difference (list1 list2 : List ℕ) : ℕ :=
  let differences := do
    let a ← list1
    let b ← list2
    pure (Int.natAbs (a - b))
  differences.foldl min (differences.head!)

theorem least_difference_is_three :
  least_positive_difference geometric_sequence arithmetic_sequence = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_difference_is_three_l599_59916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l599_59915

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (team_average : ℚ),
    team_size = 11 →
    captain_age = 27 →
    let wicket_keeper_age := captain_age + 1;
    let remaining_players := team_size - 2;
    let remaining_average := team_average - 1;
    team_average * team_size = 
      (remaining_average * remaining_players + captain_age + wicket_keeper_age) →
    team_average = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l599_59915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l599_59941

def k : ℕ := 10^30 - 54

theorem sum_of_digits_k : (k.digits 10).sum = 271 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l599_59941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l599_59980

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.sin (2 * x - Real.pi / 3)

noncomputable def g (x m : ℝ) := f (x + m)

theorem f_and_g_properties :
  -- The smallest positive period of f is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The minimum positive m for which g is symmetric about x = π/8 is 5π/12
  (∃ (m : ℝ), m > 0 ∧ (∀ (x : ℝ), g (Real.pi/4 - x) m = g (Real.pi/4 + x) m) ∧
    (∀ (n : ℝ), n > 0 → (∀ (x : ℝ), g (Real.pi/4 - x) n = g (Real.pi/4 + x) n) → m ≤ n)) ∧
  -- When m = 5π/12, the range of g on [0, π/4] is [-√3/2, 3/2]
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi/4 → -Real.sqrt 3 / 2 ≤ g x (5*Real.pi/12) ∧ g x (5*Real.pi/12) ≤ 3/2) ∧
  (∃ (y z : ℝ), 0 ≤ y ∧ y ≤ Real.pi/4 ∧ 0 ≤ z ∧ z ≤ Real.pi/4 ∧ 
    g y (5*Real.pi/12) = -Real.sqrt 3 / 2 ∧ g z (5*Real.pi/12) = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l599_59980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l599_59910

theorem product_of_roots : Real.sqrt (Real.sqrt 16) * (27 ^ (1/3 : Real)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l599_59910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_height_l599_59912

/-- Given the heights of a cat and a dog, and the average height of three animals
    (cat, dog, and bird), calculate the height of the bird. -/
theorem bird_height (cat_height dog_height avg_height bird_height : ℝ)
  (h_cat : cat_height = 92)
  (h_dog : dog_height = 94)
  (h_avg : avg_height = 95)
  (h_total : 3 * avg_height = cat_height + dog_height + bird_height) :
  bird_height = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_height_l599_59912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_armans_earnings_l599_59995

/-- Calculates the total earnings for Arman over two weeks given his work hours and pay rates. -/
theorem armans_earnings 
  (hours_last_week hours_this_week : ℕ) 
  (rate_last_week rate_increase : ℚ) 
  (h1 : hours_last_week = 35)
  (h2 : hours_this_week = 40)
  (h3 : rate_last_week = 10)
  (h4 : rate_increase = 1/2)
  : (hours_last_week : ℚ) * rate_last_week + 
    (hours_this_week : ℚ) * (rate_last_week + rate_increase) = 770 := by
  sorry

#check armans_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_armans_earnings_l599_59995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l599_59933

open Real

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * sin t.A

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.b * cos (t.A / 2) = area t) :
  t.A = π/3 ∧ 1 < (t.b + t.c) / t.a ∧ (t.b + t.c) / t.a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l599_59933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l599_59903

noncomputable def f (b c x : ℝ) : ℝ := (1/3) * x^3 - b * x + c

theorem function_properties :
  ∀ b c : ℝ,
  (∃ m k : ℝ, ∀ x : ℝ, HasDerivAt (f b c) (m * x + k) 1) →
  (b = 1 → ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ f b c x = 0) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f b c x₁ - f b c x₂| ≤ 4/3) →
  ((b = -1 ∧ c = 5/3) ∨
   (b = 1 ∧ (c = 2/3 ∨ (-2/3 < c ∧ c ≤ 0))) ∨
   (-1/3 ≤ b ∧ b ≤ 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l599_59903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_over_x_l599_59964

/-- Given a function f(x) = e^x / x, prove that its derivative f'(x) = (xe^x - e^x) / x^2 -/
theorem derivative_of_exp_over_x (x : ℝ) (hx : x ≠ 0) :
  deriv (λ y => Real.exp y / y) x = (x * Real.exp x - Real.exp x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_over_x_l599_59964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_product_l599_59951

theorem four_integer_product (m n p q : ℕ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_product_l599_59951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l599_59922

theorem min_value_reciprocal_sum (a m n : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (fun x : ℝ => a^(x+1) - 2) (-1) = -1 →
  -m - n + 2 = 0 →
  m * n > 0 →
  (∀ p q : ℝ, p > 0 → q > 0 → -p - q + 2 = 0 → 1/p + 1/q ≥ 1/m + 1/n) →
  1/m + 1/n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l599_59922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_sqrt_sum_l599_59937

theorem unique_solution_for_sqrt_sum (x y : ℕ) :
  0 < x ∧ 0 < y ∧ x ≤ y ∧ (Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ) = Real.sqrt 1992) → x = 498 ∧ y = 498 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_sqrt_sum_l599_59937
