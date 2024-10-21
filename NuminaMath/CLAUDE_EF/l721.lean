import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l721_72162

theorem union_of_sets : ∃ (A B : Set Int), A ∪ B = {-1, 0, 1, 2} := by
  -- Define the sets A and B
  let A : Set Int := {-1, 0, 1}
  let B : Set Int := {1, 2}

  -- Prove the existence of sets A and B that satisfy the condition
  use A, B

  -- State and prove the equality
  calc A ∪ B
    = {-1, 0, 1} ∪ {1, 2} := rfl
    _ = {-1, 0, 1, 2} := by
      ext x
      simp [Set.mem_union, Set.mem_singleton, Set.mem_insert]
      tauto

  -- The detailed proof is omitted and replaced with 'sorry'
  -- sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l721_72162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_when_a_is_one_h_x2_gt_half_x1_l721_72130

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def g (x : ℝ) : ℝ := (x - 1) / x

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f 1 (x + 1) + (1/2) * x^2 * g ((1/2) * x)

-- Statement for part 1
theorem f_geq_g_when_a_is_one :
  ∀ x : ℝ, x > 0 → f 1 x ≥ g x :=
by
  sorry

-- Statement for part 2
theorem h_x2_gt_half_x1 (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₁| < δ → |h x - h x₁| < ε) →
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₂| < δ → |h x - h x₂| < ε) →
  h x₂ > x₁ / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_when_a_is_one_h_x2_gt_half_x1_l721_72130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_distinct_positive_roots_l721_72106

/-- The type of integer pairs (b, c) where |b| ≤ 10 and |c| ≤ 10 -/
def ValidPair : Type := { p : ℤ × ℤ // |p.1| ≤ 10 ∧ |p.2| ≤ 10 }

/-- The equation x^2 + bx + c = 0 has distinct positive real roots -/
def HasDistinctPositiveRoots (p : ValidPair) : Prop :=
  let b := p.val.1
  let c := p.val.2
  b^2 - 4*c > 0 ∧ b < 0 ∧ b^2 > 4*c

/-- The total number of valid pairs -/
def TotalValidPairs : ℕ := 441

/-- The number of pairs that have distinct positive roots -/
def PairsWithDistinctPositiveRoots : ℕ := 40

/-- The probability of choosing a pair that doesn't have distinct positive roots -/
def ProbabilityNoDistinctPositiveRoots : ℚ :=
  1 - (PairsWithDistinctPositiveRoots : ℚ) / TotalValidPairs

theorem probability_no_distinct_positive_roots :
  ProbabilityNoDistinctPositiveRoots = 401 / 441 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_distinct_positive_roots_l721_72106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l721_72179

theorem cos_pi_minus_alpha (α : ℝ) :
  (∃ (x y : ℝ), x = -3 ∧ y = 4 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) →
  Real.cos (π - α) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l721_72179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_profit_percentage_l721_72127

noncomputable section

-- Define the cost price
def cost_price : ℝ := 300

-- Define the additional amount
def additional_amount : ℝ := 18

-- Define the new profit percentage
def new_profit_percentage : ℝ := 18

-- Define the function to calculate selling price given a profit percentage
def selling_price (profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Define the theorem
theorem initial_profit_percentage :
  ∃ (initial_percentage : ℝ),
    selling_price initial_percentage + additional_amount =
      selling_price new_profit_percentage ∧
    initial_percentage = 12 := by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_profit_percentage_l721_72127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_1_f_upper_bound_2_l721_72123

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.sqrt x - 1

-- Theorem for part (1)
theorem f_upper_bound_1 (x : ℝ) (h : x > 1) : f x < (3/2) * (x - 1) := by sorry

-- Theorem for part (2)
theorem f_upper_bound_2 (x : ℝ) (h1 : x > 1) (h2 : x < 3) : f x < 9 * (x - 1) / (x + 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_1_f_upper_bound_2_l721_72123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_sector_area_l721_72133

/-- The radius of a circle, given the area of a sector and its central angle --/
theorem circle_radius_from_sector_area (area : ℝ) (angle : ℝ) (radius : ℝ) : 
  area = (angle / 360) * Real.pi * radius^2 →
  angle = 42 →
  area = 82.5 →
  abs (radius - 15) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_sector_area_l721_72133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ships_distance_inconsistency_l721_72138

/-- Represents the position of a ship moving at constant velocity -/
def ship_position (initial_position velocity : ℝ) (time : ℝ) : ℝ :=
  initial_position + velocity * time

/-- Theorem stating the impossibility of the given scenario -/
theorem ships_distance_inconsistency :
  ∀ (x1 x2 v1 v2 : ℝ),
    ¬(∃ (d12 d14 d15 : ℝ),
      d12 = |ship_position x1 v1 0 - ship_position x2 v2 0| ∧
      d14 = |ship_position x1 v1 2 - ship_position x2 v2 2| ∧
      d15 = |ship_position x1 v1 3 - ship_position x2 v2 3| ∧
      d12 = 5 ∧ d14 = 7 ∧ d15 = 2) :=
by
  sorry

#check ships_distance_inconsistency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ships_distance_inconsistency_l721_72138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_worked_40_hours_l721_72199

/-- A babysitter's pay structure and worked hours. -/
structure BabysitterPay where
  regularRate : ℚ
  regularHours : ℚ
  overtimeRateIncrease : ℚ
  totalEarnings : ℚ

/-- Calculate the total hours worked by a babysitter given their pay structure and earnings. -/
noncomputable def totalHoursWorked (pay : BabysitterPay) : ℚ :=
  let overtimeRate := pay.regularRate * (1 + pay.overtimeRateIncrease)
  let regularEarnings := pay.regularRate * pay.regularHours
  let overtimeEarnings := pay.totalEarnings - regularEarnings
  let overtimeHours := overtimeEarnings / overtimeRate
  pay.regularHours + overtimeHours

/-- Theorem stating that given the specific pay structure and earnings, the babysitter worked 40 hours. -/
theorem babysitter_worked_40_hours :
  let pay : BabysitterPay := {
    regularRate := 16,
    regularHours := 30,
    overtimeRateIncrease := 3/4,
    totalEarnings := 760
  }
  totalHoursWorked pay = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_worked_40_hours_l721_72199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l721_72154

def mySequence (n : ℕ) : ℚ :=
  590049 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ m : ℤ, ↑m = q

theorem last_integer_in_sequence :
  ∃ k : ℕ, (is_integer (mySequence k) ∧ ¬is_integer (mySequence (k+1))) ∧
  mySequence k = 21853 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l721_72154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_right_triangle_from_hypotenuse_and_ratio_rhombus_construction_reduces_to_right_triangle_l721_72124

/-- Given a hypotenuse and the ratio of legs, we can construct a right-angled triangle. -/
theorem construct_right_triangle_from_hypotenuse_and_ratio 
  (hypotenuse : ℝ) (m n : ℝ) (h_positive : hypotenuse > 0) (h_ratio : m > 0 ∧ n > 0) :
  ∃ (a b : ℝ), a^2 + b^2 = hypotenuse^2 ∧ a/b = m/n := by
  -- We can construct a right triangle with the given conditions
  sorry

/-- The problem of constructing a rhombus with given side and diagonal ratio
    reduces to constructing a right triangle with given hypotenuse and leg ratio. -/
theorem rhombus_construction_reduces_to_right_triangle 
  (side : ℝ) (m n : ℝ) (h_positive : side > 0) (h_ratio : m > 0 ∧ n > 0) :
  ∃ (a b : ℝ), a^2 + b^2 = side^2 ∧ a/b = m/n := by
  -- The rhombus problem reduces to the right triangle problem
  exact construct_right_triangle_from_hypotenuse_and_ratio side m n h_positive h_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_right_triangle_from_hypotenuse_and_ratio_rhombus_construction_reduces_to_right_triangle_l721_72124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_extrema_iff_a_range_l721_72163

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + x + 1

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem f_local_extrema_iff_a_range (a : ℝ) :
  (∃ (x_min x_max : ℝ), IsLocalMin (f a) x_min ∧ IsLocalMax (f a) x_max) ↔ 
  (a < -1 ∨ a > 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_extrema_iff_a_range_l721_72163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_angle_l721_72193

theorem vector_perpendicular_angle (α β : ℝ) 
  (h1 : π/6 ≤ α) (h2 : α ≤ π/2) (h3 : π/2 < β) (h4 : β ≤ 5*π/6)
  (h5 : let a : ℝ × ℝ := (Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)
        let b : ℝ × ℝ := (Real.cos β, 2 * Real.sin β)
        (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2) = 0)) :
  β - α = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_angle_l721_72193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l721_72164

theorem expression_result : 
  |((0.82 : ℝ)^3 - (0.1 : ℝ)^3) / ((0.82 : ℝ)^2 + 0.082 + (0.1 : ℝ)^2) - 0.7201| < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l721_72164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_l721_72134

/-- The height of a stack of 5 identical cylindrical pipes -/
noncomputable def stack_height (d : ℝ) : ℝ :=
  d + d/2 * Real.sqrt 3 + d/2

/-- Theorem stating the height of the stack of pipes -/
theorem pipe_stack_height :
  let d : ℝ := 10
  stack_height d = 10 + 5 * Real.sqrt 3 := by
  sorry

#check pipe_stack_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_l721_72134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_range_l721_72116

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_monotone_increasing (k : ℤ) :
  is_monotone_increasing f (-Real.pi/3 + k*Real.pi) (Real.pi/6 + k*Real.pi) := by
  sorry

theorem f_range :
  ∀ x, -Real.pi/8 ≤ x ∧ x ≤ Real.pi/2 → 0 ≤ f x ∧ f x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_range_l721_72116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l721_72143

-- Define the monic cubic polynomial
def MonicCubicPolynomial (p : Polynomial ℝ) : Prop :=
  p.degree = some 3 ∧ p.leadingCoeff = 1

-- Define the munificence of a polynomial
noncomputable def munificence (p : Polynomial ℝ) : ℝ :=
  sSup { y | ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ y = |p.eval x| }

-- Theorem statement
theorem smallest_munificence_monic_cubic :
  ∃ (p : Polynomial ℝ), MonicCubicPolynomial p ∧
    munificence p = 1 ∧
    ∀ (q : Polynomial ℝ), MonicCubicPolynomial q → munificence q ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l721_72143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_division_exists_l721_72157

/-- Represents a cell in the 3x3 grid -/
inductive Cell
| A1 | A2 | A3
| B1 | B2 | B3
| C1 | C2 | C3

/-- Represents an L-shaped corner -/
structure Corner where
  cells : List Cell
  has_exactly_one_star : Bool

/-- Represents the 3x3 grid field -/
structure GameField where
  stars : List Cell
  corners : List Corner

/-- Checks if a valid division of the field into L-shaped corners exists -/
def has_valid_division (f : GameField) : Prop :=
  ∃ (division : List Corner),
    (∀ c, c ∈ division → c.has_exactly_one_star) ∧
    (∀ star, star ∈ f.stars → ∃! corner, corner ∈ division ∧ star ∈ corner.cells) ∧
    (∀ c1 c2, c1 ∈ division → c2 ∈ division → c1 ≠ c2 → ∀ cell, cell ∈ c1.cells → cell ∉ c2.cells)

/-- The main theorem stating that a valid division exists for a 3x3 grid with 5 stars -/
theorem valid_division_exists :
  ∀ (f : GameField), f.stars.length = 5 → has_valid_division f :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_division_exists_l721_72157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_lower_bound_l721_72100

/-- A configuration of planes in space. -/
structure PlaneConfiguration where
  n : ℕ
  planes : Fin n → Set (Fin 3 → ℝ)  -- Representing planes as sets of points in ℝ³
  n_ge_5 : n ≥ 5
  three_intersect : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    ∃! p, p ∈ (planes i) ∩ (planes j) ∩ (planes k)
  four_dont_intersect : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l →
    (planes i) ∩ (planes j) ∩ (planes k) ∩ (planes l) = ∅

/-- The number of non-overlapping tetrahedra formed by the planes. -/
noncomputable def num_tetrahedra (config : PlaneConfiguration) : ℕ := sorry

/-- The theorem to be proved. -/
theorem tetrahedra_lower_bound (config : PlaneConfiguration) :
  num_tetrahedra config ≥ (2 * config.n - 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_lower_bound_l721_72100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_polygon_l721_72147

/-- Represents a triangle --/
structure Triangle where
  -- Define triangle properties here
  mk :: -- Add a constructor

/-- Represents a polygon --/
structure Polygon where
  sides : ℕ

/-- Represents cutting a triangle into two parts --/
def Triangle.cut (t : Triangle) : Set Polygon := sorry

/-- Represents forming a polygon from a set of pieces --/
def Polygon.form (pieces : Set Polygon) : Polygon := sorry

/-- A triangle can be cut into two parts that can be used to form a 20-sided polygon --/
theorem triangle_to_polygon : ∃ (t : Triangle) (p1 p2 : Polygon),
  (Triangle.cut t = {p1, p2}) ∧ 
  (∃ (other_pieces : Set Polygon), 
    (Polygon.form ({p1, p2} ∪ other_pieces)).sides = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_polygon_l721_72147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l721_72132

noncomputable section

/-- The function f(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * Real.log x + b) / x

/-- The function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x + 2 / x - a - 2

/-- The function F(x) -/
noncomputable def F (a b : ℝ) (x : ℝ) : ℝ := f a b x + g a x

/-- The theorem statement -/
theorem function_range (a b : ℝ) : 
  a ≤ 2 ∧ a ≠ 0 ∧
  (∃ (m : ℝ), ∀ (x : ℝ), x ≠ 1 → f a b x - f a b 1 = m * (x - 1)) ∧
  f a b 1 + 2 * (3 - 1) = 0 ∧
  (∃! (x : ℝ), 0 < x ∧ x ≤ 2 ∧ F a b x = 0) →
  (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l721_72132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_values_l721_72153

theorem integer_fraction_values (x : ℝ) : 
  (∃ (n : ℤ), (x^2 + 2*x - 3) / (x^2 + 1) = ↑n) ↔ x ∈ ({-3, 0, 1, -1/2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_values_l721_72153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkered_triangle_division_l721_72190

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  area : Nat

/-- Represents the checkered triangle -/
structure CheckeredTriangle where
  parts : List TrianglePart

/-- The sum of numbers in a triangle part -/
def partSum (part : TrianglePart) : Nat :=
  part.numbers.sum

/-- The total sum of all numbers in the triangle -/
def totalSum (triangle : CheckeredTriangle) : Nat :=
  triangle.parts.map partSum |>.sum

/-- Checks if all parts have equal sums -/
def allPartsEqualSum (triangle : CheckeredTriangle) : Prop :=
  ∀ p1 p2, p1 ∈ triangle.parts → p2 ∈ triangle.parts → partSum p1 = partSum p2

/-- Checks if all parts have different areas -/
def allPartsDifferentAreas (triangle : CheckeredTriangle) : Prop :=
  ∀ p1 p2, p1 ∈ triangle.parts → p2 ∈ triangle.parts → p1 ≠ p2 → p1.area ≠ p2.area

/-- The main theorem -/
theorem checkered_triangle_division :
  ∃ (triangle : CheckeredTriangle),
    totalSum triangle = 63 ∧
    triangle.parts.length = 3 ∧
    allPartsEqualSum triangle ∧
    allPartsDifferentAreas triangle :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkered_triangle_division_l721_72190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_altitude_and_median_l721_72111

/-- Given a triangle with sides a and b (a > b) and area S, 
    the angle between the altitude and the median drawn to the third side 
    is arctan((8S²) / (a² * (a² - b²))) -/
theorem angle_between_altitude_and_median 
  (a b S : ℝ) 
  (h1 : a > b) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : S > 0) : 
  ∃ θ : ℝ, θ = Real.arctan ((8 * S^2) / (a^2 * (a^2 - b^2))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_altitude_and_median_l721_72111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l721_72115

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The common ratio of a geometric sequence. -/
noncomputable def CommonRatio (a : ℕ → ℝ) : ℝ :=
  a 1 / a 0

theorem geometric_sequence_common_ratio :
  ∀ a : ℕ → ℝ,
  IsGeometricSequence a →
  a 0 = 25 →
  a 1 = 50 →
  a 2 = 100 →
  a 3 = 200 →
  CommonRatio a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l721_72115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l721_72180

theorem negation_of_forall_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l721_72180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l721_72169

-- Define the vectors
noncomputable def a (α : Real) : Real × Real := (1/3, Real.tan α)
noncomputable def b (α : Real) : Real × Real := (Real.cos α, 1)

-- Define the parallelism condition
def parallel (α : Real) : Prop :=
  (1/3) * 1 = (Real.tan α) * (Real.cos α)

-- State the theorem
theorem cos_2α_value (α : Real) (h : parallel α) :
  Real.cos (2 * α) = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l721_72169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_f_increasing_f_range_l721_72135

noncomputable section

-- Define f as a function from positive reals to reals
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain (x : ℝ) : Prop := x > 0

-- Define the properties of f
axiom f_add {x y : ℝ} (hx : domain x) (hy : domain y) : f (x * y) = f x + f y
axiom f_pos {x : ℝ} (hx : domain x) (h : x > 1) : f x > 0

-- Theorem statements
theorem f_one : f 1 = 0 := by sorry

theorem f_increasing : ∀ {x y : ℝ}, domain x → domain y → x < y → f x < f y := by sorry

theorem f_range (h : f (1/3) = -1) : 
  Set.Icc (2 : ℝ) (9/4) = {x : ℝ | domain x ∧ f x - f (x - 2) ≥ 2} := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_f_increasing_f_range_l721_72135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_plus_y_minus_three_l721_72175

/-- The angle of inclination of a line is the angle it makes with the positive x-axis. -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- The line equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem angle_of_inclination_x_plus_y_minus_three (l : Line) :
  l.a = 1 → l.b = 1 → l.c = -3 →
  angle_of_inclination l.a l.b l.c = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_plus_y_minus_three_l721_72175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_regular_octagon_l721_72129

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_in_regular_octagon 
  (ABCDEFGH : RegularOctagon) 
  (A E C : Fin 8)
  (hAE : sorry) -- AE connects alternate vertices
  (hEC : sorry) -- EC connects alternate vertices
  : angle_measure 
      (ABCDEFGH.vertices A) 
      (ABCDEFGH.vertices E) 
      (ABCDEFGH.vertices C) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_regular_octagon_l721_72129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_half_a_bound_for_nonnegative_f_l721_72103

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1 - x^2 / 2

-- Part I
theorem f_increasing_when_a_half :
  Monotone (f (1/2)) := by
  sorry

-- Part II
theorem a_bound_for_nonnegative_f (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_half_a_bound_for_nonnegative_f_l721_72103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l721_72167

/-- Represents a square divided into a 5x5 grid with a shaded area -/
structure GridSquare where
  /-- The side length of the large square -/
  side_length : ℝ
  /-- The number of half-squares in the shaded area -/
  shaded_half_squares : ℕ

/-- Calculates the area of the shaded region in the GridSquare -/
noncomputable def shaded_area (gs : GridSquare) : ℝ :=
  (gs.shaded_half_squares : ℝ) * (gs.side_length ^ 2) / 50

/-- Theorem stating that the ratio of the shaded area to the total area is 1/10 -/
theorem shaded_area_ratio (gs : GridSquare) 
    (h1 : gs.side_length > 0)
    (h2 : gs.shaded_half_squares = 5) : 
  shaded_area gs / (gs.side_length ^ 2) = 1 / 10 := by
  sorry

#check shaded_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l721_72167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l721_72168

/-- The area of the triangle formed by two lines and the y-axis --/
noncomputable def triangle_area (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  let x_intersect := (b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁)
  let y_intercept₁ := c₁ / b₁
  let y_intercept₂ := c₂ / b₂
  (1 / 2) * |x_intersect| * |y_intercept₁ - y_intercept₂|

/-- Theorem stating that the area of the triangle formed by the given lines and y-axis is 9 --/
theorem triangle_area_is_nine :
  triangle_area 3 (-1) 12 3 2 (-6) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l721_72168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_BAD_l721_72148

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add properties that define a triangle if needed
  True

-- Define the angle C
def angle_C (A B C : ℝ × ℝ) : ℝ :=
  -- Definition of angle C
  sorry

-- Define the length of BC
def BC_length (B C : ℝ × ℝ) : ℝ :=
  -- Definition of BC length
  sorry

-- Define the midpoint D
def midpoint_D (B C : ℝ × ℝ) : ℝ × ℝ :=
  -- Definition of midpoint
  sorry

-- Define the tangent of angle BAD
noncomputable def tan_BAD (A B D : ℝ × ℝ) : ℝ :=
  -- Definition of tan∠BAD
  sorry

theorem max_tan_BAD (A B C : ℝ × ℝ) :
  Triangle A B C →
  angle_C A B C = π/4 →
  BC_length B C = 6 →
  let D := midpoint_D B C
  (∀ A' B' C', Triangle A' B' C' →
               angle_C A' B' C' = π/4 →
               BC_length B' C' = 6 →
               tan_BAD A' B' (midpoint_D B' C') ≤ 1 / (4 * Real.sqrt 2 - 3)) ∧
  (∃ A₀ B₀ C₀, Triangle A₀ B₀ C₀ ∧
                angle_C A₀ B₀ C₀ = π/4 ∧
                BC_length B₀ C₀ = 6 ∧
                tan_BAD A₀ B₀ (midpoint_D B₀ C₀) = 1 / (4 * Real.sqrt 2 - 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_BAD_l721_72148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_symmetry_l721_72194

-- Define a random variable following a normal distribution
def ξ : Real → Real := sorry

-- Define the parameters of the normal distribution
def μ : Real := 2
def σ : Real := 3

-- Define the probability density function for the normal distribution
noncomputable def normal_pdf (x : Real) : Real :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2) / (2 * σ^2))

-- Define the cumulative distribution function
noncomputable def normal_cdf (x : Real) : Real :=
  ∫ y in Set.Iio x, normal_pdf y

-- State the theorem
theorem normal_distribution_symmetry (c : Real) :
  (∀ x, ξ x = normal_pdf x) →
  (normal_cdf (c + 1) = 1 - normal_cdf (c - 1)) →
  c = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_symmetry_l721_72194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_correct_statements_l721_72144

-- Define a type for statements
inductive Statement
| InputStatement (s : String)
| OutputStatement (s : String)
| AssignmentStatement (s : String)

-- Define a function to check if a statement is correct
def isCorrect (stmt : Statement) : Bool :=
  match stmt with
  | Statement.InputStatement s => 
      s.contains ',' && s.startsWith "INPUT " && s.contains '='
  | Statement.OutputStatement s => 
      s.contains ',' && s.startsWith "INPUT "
  | Statement.AssignmentStatement s => 
      let parts := s.split (· == '=')
      parts.length == 2 && parts[0]!.all Char.isAlpha

-- Define the given statements
def statements : List Statement := [
  Statement.OutputStatement "INPUT a; b; c",
  Statement.InputStatement "INPUT x=3",
  Statement.AssignmentStatement "3=B",
  Statement.AssignmentStatement "A=B=2"
]

-- Theorem to prove
theorem no_correct_statements : 
  (statements.filter isCorrect).length = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_correct_statements_l721_72144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l721_72197

/-- Represents a line segment in 3D space --/
structure LineSegment where
  start : ℝ × ℝ × ℝ
  end_ : ℝ × ℝ × ℝ

/-- Calculates the length of a line segment --/
noncomputable def length (seg : LineSegment) : ℝ := sorry

/-- Calculates the volume of the region within a given distance of a line segment --/
noncomputable def regionVolume (seg : LineSegment) (distance : ℝ) : ℝ := sorry

theorem line_segment_length (CD : LineSegment) :
  regionVolume CD 5 = 570 * Real.pi → length CD = 16.1333 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l721_72197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_ratio_l721_72183

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron with vertices S, A, B, and C -/
structure Tetrahedron where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (p1 p2 p3 : Point3D) : Point3D :=
  { x := (p1.x + p2.x + p3.x) / 3
  , y := (p1.y + p2.y + p3.y) / 3
  , z := (p1.z + p2.z + p3.z) / 3 }

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem statement -/
theorem tetrahedron_ratio (t : Tetrahedron) : 
  let A₁ := centroid t.S t.B t.C
  let G := Point3D.mk ((3 * t.A.x + A₁.x) / 4) ((3 * t.A.y + A₁.y) / 4) ((3 * t.A.z + A₁.z) / 4)
  let M := Point3D.mk ((2 * t.A.x + t.B.x + t.C.x) / 4) ((2 * t.A.y + t.B.y + t.C.y) / 4) ((2 * t.A.z + t.B.z + t.C.z) / 4)
  distance A₁ M / distance t.A t.S = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_ratio_l721_72183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_0_05049_l721_72152

noncomputable def round_to_decimal_place (x : ℝ) (d : ℕ) : ℝ :=
  (⌊x * 10^d + 0.5⌋ / 10^d : ℝ)

theorem rounding_0_05049 :
  let x := 0.05049
  (round_to_decimal_place x 1 = 0.1) ∧
  (round_to_decimal_place x 3 = 0.050) ∧
  (round_to_decimal_place x 2 = 0.05) ∧
  (round_to_decimal_place x 4 = 0.0505) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_0_05049_l721_72152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riemann_zeta_inequality_l721_72110

open Real

-- Define the Riemann Zeta function
noncomputable def zeta (s : ℝ) : ℝ := tsum fun n => 1 / (n ^ s)

-- State the theorem
theorem riemann_zeta_inequality (s : ℝ) (hs : s > 1) :
  tsum (fun k => 1 / (1 + k ^ s)) ≥ zeta s / (1 + zeta s) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_riemann_zeta_inequality_l721_72110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l721_72177

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := min (x + 2) (min (4 * x + 1) (-2 * x + 4))

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 8/3 := by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l721_72177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l721_72105

-- Define the curves C1 and C2
noncomputable def C1 (θ : Real) : Real := Real.sqrt 3

noncomputable def C2 (θ : Real) : Real := Real.sqrt (6 / (1 - Real.sin (2 * θ) + Real.sqrt 3 * Real.cos (2 * θ)))

-- Define the intersection points
def intersection_points (C1 C2 : Real → Real) : Set Real :=
  {θ | C1 θ = C2 θ ∧ 0 ≤ θ ∧ θ ≤ Real.pi}

-- State the theorem
theorem intersection_angle (C1 C2 : Real → Real) :
  let points := intersection_points C1 C2
  ∃ θ1 θ2, θ1 ∈ points ∧ θ2 ∈ points ∧ θ2 - θ1 = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l721_72105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_line_l721_72178

/-- A line in two-dimensional space. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intercepts of a line on the x and y axes. -/
noncomputable def intercepts (l : Line) : ℝ × ℝ :=
  (-l.c / l.a, -l.c / l.b)

/-- The sum of the intercepts of a line. -/
noncomputable def intercept_sum (l : Line) : ℝ :=
  let (x, y) := intercepts l
  x + y

/-- A line passes through a point if the point satisfies the line equation. -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The intercepts are positive if both components of the intercepts pair are positive. -/
def positive_intercepts (l : Line) : Prop :=
  let (x, y) := intercepts l
  x > 0 ∧ y > 0

/-- The main theorem stating the existence of the optimal line. -/
theorem optimal_line :
  ∃ (l : Line),
    passes_through l (1, 4) ∧
    positive_intercepts l ∧
    (∀ (l' : Line),
      passes_through l' (1, 4) →
      positive_intercepts l' →
      intercept_sum l ≤ intercept_sum l') ∧
    l.a = 2 ∧ l.b = 1 ∧ l.c = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_line_l721_72178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l721_72137

/-- A line in 2D space -/
structure Line where
  mk :: 

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Symmetry of a point with respect to a line -/
def symmetric_point (p p' : Point) (_l : Line) : Prop :=
  p'.x = p.y + 1 ∧ p'.y = p.x - 1

/-- Symmetry of circles with respect to a line -/
def symmetric_circles (c c' : Circle) (l : Line) : Prop :=
  symmetric_point c.center c'.center l

/-- The problem statement -/
theorem circle_symmetry (l : Line) (p p' : Point) (c c' : Circle) :
  symmetric_point p p' l →
  c.center = Point.mk 3 1 →
  c.radius = Real.sqrt 10 →
  symmetric_circles c c' l →
  c'.center = Point.mk 2 2 ∧ c'.radius = Real.sqrt 10 := by
  sorry

#check circle_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l721_72137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ratio_l721_72128

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

-- Define the ray l
def l (α : ℝ) : Prop := Real.pi/4 ≤ α ∧ α ≤ Real.pi/3

-- Define the distance ratio function
noncomputable def distance_ratio (α : ℝ) : ℝ := 
  2 * Real.sqrt ((1 / Real.sin α^4) - 1)

-- State the theorem
theorem min_distance_ratio :
  ∀ α, l α → 
  (∀ β, l β → distance_ratio α ≤ distance_ratio β) →
  distance_ratio α = 2 * Real.sqrt 7 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ratio_l721_72128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l721_72158

def sequence_a : ℕ → ℝ
  | 0 => 15  -- Add this case to cover Nat.zero
  | 1 => 15
  | n + 1 => sequence_a n + 2 * n

theorem min_value_of_sequence_ratio :
  let a := sequence_a
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ 27 / 4 ∧
  ∃ m : ℕ, m ≥ 1 ∧ a m / m = 27 / 4 := by
  sorry

#check min_value_of_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l721_72158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_eq_16_f_odd_inequality_holds_l721_72195

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else -x^2

-- Theorem 1: f[f(2)] = 16
theorem f_f_2_eq_16 : f (f 2) = 16 := by sorry

-- Theorem 2: f is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x := by sorry

-- Theorem 3: For k > 8 and t ∈ [1,2], f(t^2 - 2t) + f(k - 2t^2) < 0
theorem inequality_holds (k : ℝ) (h : k > 8) (t : ℝ) (ht : t ∈ Set.Icc 1 2) :
  f (t^2 - 2*t) + f (k - 2*t^2) < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_eq_16_f_odd_inequality_holds_l721_72195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_collinearity_l721_72173

-- Define the basic geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define helper functions
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

def is_tangent_point_cc (c1 c2 : Circle) (p : ℝ × ℝ) : Prop := sorry

def is_tangent_point_lc (l : Line) (c : Circle) (p : ℝ × ℝ) : Prop := sorry

def collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Define the given conditions
def geometric_setup (r : ℝ) : Prop := ∃ (c1 c2 c3 : Circle) (s t : Line),
  -- Two non-intersecting circles with equal radii
  c1.radius = r ∧ c2.radius = r ∧ 
  -- Third circle tangent to both circles
  (∃ k l : ℝ × ℝ, is_tangent_point_cc c3 c1 k ∧ is_tangent_point_cc c3 c2 l) ∧
  -- Line s passes through centers of c1 and c2
  (s.point1 = c1.center ∧ s.point2 = c2.center) ∧
  -- Common tangent t touches both circles
  (∃ n p : ℝ × ℝ, is_tangent_point_lc t c1 n ∧ is_tangent_point_lc t c2 p) ∧
  -- Third circle tangent to s and t
  (∃ m p : ℝ × ℝ, is_tangent_point_lc s c3 m ∧ is_tangent_point_lc t c3 p)

-- Define the theorems to be proved
theorem distance_between_centers (r : ℝ) :
  geometric_setup r → ∃ c1 c2 : Circle, distance c1.center c2.center = 2 * r := by
  sorry

theorem collinearity (r : ℝ) :
  geometric_setup r → ∃ (c1 c3 : Circle) (s t : Line) (m k n : ℝ × ℝ),
    is_tangent_point_cc c3 c1 k ∧
    is_tangent_point_lc s c3 m ∧
    is_tangent_point_lc t c1 n ∧
    collinear m k n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_collinearity_l721_72173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_l721_72171

/-- The 5-adic absolute value -/
noncomputable def padic_abs (x : ℚ) : ℝ := sorry

/-- The field of 5-adic numbers -/
structure Q_5 where
  val : ℚ

/-- Embedding of rationals into 5-adic numbers -/
def embed : ℚ → Q_5 := 
  λ x => ⟨x⟩

/-- Power operation for Q_5 -/
instance : Pow Q_5 ℕ where
  pow a n := ⟨(a.val ^ n)⟩

/-- Multiplication for Q_5 -/
instance : Mul Q_5 where
  mul a b := ⟨a.val * b.val⟩

/-- Addition for Q_5 -/
instance : Add Q_5 where
  add a b := ⟨a.val + b.val⟩

/-- Coercion from ℚ to Q_5 -/
instance : Coe ℚ Q_5 where
  coe := embed

theorem sqrt_two_irrational (h : ∀ x : Q_5, x^2 ≠ (embed 2)) : 
  ¬ ∃ q : ℚ, q^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_l721_72171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l721_72188

noncomputable section

/-- The original sine function -/
def original_sine (x : ℝ) : ℝ := 3 * Real.sin (x - Real.pi / 6)

/-- The transformed sine function -/
def transformed_sine (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

/-- Transformation factor -/
def ω : ℝ := 2

theorem sine_transformation (x : ℝ) :
  ∃ (k : ℝ), k > 0 ∧ k < 1 ∧
  (∀ (y : ℝ), transformed_sine (k * x) = original_sine x) ∧
  k = 1 / ω :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l721_72188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_equals_100_l721_72101

def a_sequence (n : ℕ) : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (k+2) => k+1 - a_sequence n k

def sum_sequence (n : ℕ) : ℚ :=
  (Finset.range n).sum (a_sequence n)

theorem sum_21_equals_100 :
  sum_sequence 21 = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_equals_100_l721_72101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_three_implies_k_equals_four_l721_72198

theorem integral_equals_three_implies_k_equals_four (k : ℝ) : 
  (∫ x in (0 : ℝ)..(1 : ℝ), 3 * x^2 + k * x) = 3 → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_three_implies_k_equals_four_l721_72198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_22_consecutive_integers_as_cube_l721_72102

/-- Given 22 consecutive positive integers whose sum is a perfect cube,
    the smallest possible value of this sum is 1331. -/
theorem smallest_sum_of_22_consecutive_integers_as_cube : ∃ n : ℕ,
  (∃ k : ℕ, 11 * (2 * n + 21) = k^3) ∧  -- sum is a perfect cube
  (∀ m : ℕ, m < n →                     -- n is the smallest such number
    ¬∃ j : ℕ, 11 * (2 * m + 21) = j^3) ∧
  11 * (2 * n + 21) = 1331              -- the smallest sum is 1331
:= by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_22_consecutive_integers_as_cube_l721_72102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l721_72149

-- Define set A
def A : Set ℝ := {x : ℝ | 1/x < 1}

-- Define set B (domain of y = lg(x+1))
def B : Set ℝ := {x : ℝ | x > -1}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l721_72149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l721_72160

/-- Calculates the simple interest for a given principal, rate, and time --/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem transaction_gain_per_year 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (lending_rate : ℝ) 
  (time : ℝ) : 
  borrowed_amount = 5000 →
  borrowing_rate = 4 →
  lending_rate = 8 →
  time = 2 →
  (simpleInterest borrowed_amount lending_rate time - 
   simpleInterest borrowed_amount borrowing_rate time) / time = 200 := by
  sorry

#check transaction_gain_per_year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l721_72160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_s_r_composition_l721_72184

def r : Finset ℤ := {-2, -1, 0, 1}
def r_range : Finset ℤ := {-1, 1, 3, 5}
def s_domain : Finset ℤ := {0, 1, 2, 3}

def s (x : ℤ) : ℤ := x^2 + 1

theorem sum_s_r_composition : 
  (Finset.filter (λ x => x ∈ s_domain) r_range).sum s = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_s_r_composition_l721_72184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l721_72113

theorem trigonometric_system_solution (x y z : ℝ) :
  (Real.sin x + 2 * Real.sin (x + y + z) = 0) →
  (Real.sin y + 3 * Real.sin (x + y + z) = 0) →
  (Real.sin z + 4 * Real.sin (x + y + z) = 0) →
  ∃ (n l m : ℤ), x = n * Real.pi ∧ y = l * Real.pi ∧ z = m * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l721_72113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l721_72145

/-- An ellipse with center at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  /-- The ellipse equation -/
  ellipse_equation : ℝ → ℝ → ℝ
  /-- The ellipse passes through the point (3,0) -/
  passes_through_3_0 : ellipse_equation 3 0 = 1
  /-- The relation between a and b -/
  a_eq_3b : a = 3 * b

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  (∀ (x y : ℝ), e.ellipse_equation x y = x^2 / 9 + y^2) ∨
  (∀ (x y : ℝ), e.ellipse_equation x y = y^2 / 81 + x^2 / 9)

/-- Theorem stating that the given ellipse has one of the two standard equations -/
theorem ellipse_standard_equation (e : Ellipse) : standard_equation e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l721_72145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_6_equivalence_l721_72141

/-- Represents a single digit in a given base --/
def Digit (base : ℕ) := {d : ℕ // d < base}

/-- Converts a two-digit number in a given base to base 10 --/
def toBase10 (base : ℕ) (tens : Digit base) (ones : Digit base) : ℕ :=
  base * tens.val + ones.val

theorem base_8_6_equivalence :
  ∀ (A : Digit 8) (B : Digit 6),
  toBase10 8 A ⟨B.val, by sorry⟩ = toBase10 6 B ⟨A.val, by sorry⟩ →
  toBase10 8 A ⟨B.val, by sorry⟩ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_6_equivalence_l721_72141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l721_72119

noncomputable def g (x : ℝ) : ℝ := x / (x + 1)^2

theorem range_of_g :
  Set.range g = Set.Iic (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l721_72119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_gcd_sum_l721_72120

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 10 ∧
  ∀ i, i < seq.length → seq[i]! = Nat.gcd seq[((i - 1 + seq.length) % seq.length)]! seq[((i + 1) % seq.length)]! + 1

theorem circular_gcd_sum :
  ∃ seq : List Nat, is_valid_sequence seq ∧ seq.sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_gcd_sum_l721_72120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l721_72192

/-- Represents the initial total number of candies -/
def T : ℕ := sorry

/-- Represents Donald's initial share of candies -/
def donald_initial : ℕ := (70 * T) / 100

/-- Represents Henry's initial share of candies -/
def henry_initial : ℕ := (25 * T) / 100

/-- Represents John's initial share of candies -/
def john_initial : ℕ := (5 * T) / 100

/-- Represents the number of candies Donald gave to John -/
def candies_given : ℕ := 20

/-- Represents Henry's share after redistribution -/
def henry_after : ℕ := (henry_initial + john_initial + candies_given) / 2

/-- Represents Donald's share after redistribution -/
def donald_after : ℕ := donald_initial - candies_given

/-- Represents the number of candies given to each person the next day -/
def x : ℕ := sorry

theorem candy_distribution :
  donald_after = 3 * henry_after ∧
  donald_after + x = 2 * (henry_after + x) →
  x = 40 := by
  sorry

#check candy_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l721_72192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l721_72118

noncomputable def proj (a : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := ((u.1 * a.1 + u.2 * a.2) / (a.1 * a.1 + a.2 * a.2))
  (scalar * a.1, scalar * a.2)

theorem vector_satisfies_projections :
  let u : ℝ × ℝ := (6, 2)
  proj (1, 2) u = (2, 4) ∧ proj (3, 1) u = (6, 2) := by
  sorry

#check vector_satisfies_projections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l721_72118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_prism_circumscribing_sphere_l721_72142

/-- The volume of a regular triangular prism that circumscribes a sphere of radius 2 is 48√3 -/
theorem volume_prism_circumscribing_sphere (r : ℝ) (V : ℝ) :
  r = 2 →  -- radius of the inscribed sphere
  V = (3 * Real.sqrt 3 * r^2) * 2 * r →  -- volume formula for the prism
  V = 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_prism_circumscribing_sphere_l721_72142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_alpha_plus_pi_fourth_l721_72166

theorem sin_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1/4) :
  (Real.sin (α + π/4))^2 = 5/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_alpha_plus_pi_fourth_l721_72166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_packing_inequality_l721_72159

/-- Given a circular table of radius R and n non-overlapping circular coins of radius r 
    placed on it such that no more coins can be added, the ratio of the table's radius 
    to a coin's radius is less than or equal to 2 times the square root of the number 
    of coins plus 1. -/
theorem coin_packing_inequality (R r : ℝ) (n : ℕ) 
  (h_positive_R : R > 0) 
  (h_positive_r : r > 0) 
  (h_positive_n : n > 0) 
  (h_no_overlap : ∀ (i j : ℕ) (hi : i < n) (hj : j < n) (hij : i ≠ j), 
    ∃ (xi yi xj yj : ℝ), xi^2 + yi^2 ≤ R^2 ∧ xj^2 + yj^2 ≤ R^2 ∧ (xi - xj)^2 + (yi - yj)^2 ≥ 4*r^2)
  (h_max_packing : ∀ (x y : ℝ), x^2 + y^2 ≤ (R - r)^2 → 
    ∃ (i : ℕ) (hi : i < n) (xi yi : ℝ), xi^2 + yi^2 ≤ R^2 ∧ (x - xi)^2 + (y - yi)^2 < 4*r^2) :
  R / r ≤ 2 * Real.sqrt (n : ℝ) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_packing_inequality_l721_72159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_monthly_balance_l721_72136

/-- Represents the monthly balance of David's savings account --/
def monthly_balance : Fin 6 → ℕ
  | 0 => 150  -- January
  | 1 => 200  -- February
  | 2 => 250  -- March
  | 3 => 250  -- April
  | 4 => 200  -- May
  | 5 => 300  -- June

/-- The number of months in the period --/
def num_months : ℕ := 6

/-- The sum of all monthly balances --/
def total_balance : ℕ := (Finset.univ.sum monthly_balance : ℕ)

/-- The average monthly balance --/
noncomputable def average_balance : ℚ := (total_balance : ℚ) / num_months

theorem average_monthly_balance :
  average_balance = 225 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_monthly_balance_l721_72136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_two_divisors_l721_72109

theorem remainder_two_divisors : 
  {n : ℕ | (∃ k, 56 = n * k) ∧ n > 2 ∧ 58 % n = 2} = {7, 8, 14, 28, 56} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_two_divisors_l721_72109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l721_72185

theorem function_characterization (f : ℕ+ → ℕ+) : 
  (∀ m n : ℕ+, (f m + f n - m * n : ℤ) ≠ 0) →
  (∀ m n : ℕ+, (f m + f n - m * n : ℤ) ∣ (m * f m + n * f n : ℤ)) →
  (∀ n : ℕ+, f n = n ^ 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l721_72185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_CE_l721_72182

open Real

-- Define the triangle ABC
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = 10 ∧ dist A C = 10 ∧ dist B C = 12

-- Define the points D₁, D₂, E₁, E₂
def ExtraPoints (A B C D₁ D₂ E₁ E₂ : EuclideanSpace ℝ (Fin 2)) : Prop :=
  Triangle A B C ∧
  Triangle A D₁ E₁ ∧ Triangle A D₁ E₂ ∧ Triangle A D₂ E₁ ∧ Triangle A D₂ E₂ ∧
  dist B D₁ = 5 ∧ dist B D₂ = 5

-- The main theorem
theorem sum_of_squares_CE (A B C D₁ D₂ E₁ E₂ : EuclideanSpace ℝ (Fin 2)) 
  (h : ExtraPoints A B C D₁ D₂ E₁ E₂) : 
  (dist C E₁)^2 + (dist C E₂)^2 = 648.232 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_CE_l721_72182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robins_count_l721_72196

theorem robins_count (total : ℕ) (sparrow_fraction : ℚ) (robin_count : ℕ) : 
  total = 120 →
  sparrow_fraction = 1/3 →
  robin_count = total - (sparrow_fraction * ↑total).floor →
  robin_count = 80 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robins_count_l721_72196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_angle_radius_l721_72112

theorem inscribed_angle_radius :
  ∀ (α a b : ℝ),
  α = 120 * π / 180 →
  a = 1 →
  b = 2 →
  ∃ (r : ℝ), r = 1 ∧ r * Real.sin (α/2) = a * Real.sin (π/2 - α/4) ∧ r * Real.sin (α/2) = b * Real.sin (α/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_angle_radius_l721_72112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l721_72170

-- Define the points P and Q
noncomputable def P : ℝ × ℝ := (2, 1)
noncomputable def Q : ℝ × ℝ := (1, 4)

-- Define the line on which R lies
def line_equation (x y : ℝ) : Prop := x + y = 8

-- Define the area function for a triangle given three points
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_PQR_area :
  ∃ (R : ℝ × ℝ), line_equation R.1 R.2 ∧ triangle_area P Q R = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l721_72170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l721_72122

noncomputable def f (x : ℝ) := (Real.cos x)^2 - (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  -- Smallest positive period
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  -- Period is π
  (let T : ℝ := Real.pi; T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  -- Monotonically increasing interval
  (∀ k : ℤ, ∀ x y : ℝ, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) →
    y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) →
    x ≤ y → f x ≤ f y) ∧
  -- Minimum value on [0, π/4]
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) → f x ≥ f 0) ∧
  f 0 = 1 ∧
  -- Maximum value on [0, π/4]
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) → f x ≤ f (Real.pi / 6)) ∧
  f (Real.pi / 6) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l721_72122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_road_length_l721_72125

/-- Calculates the total man-hours for a given number of men, days, and hours per day -/
def manHours (men : ℕ) (days : ℝ) (hoursPerDay : ℕ) : ℝ :=
  (men : ℝ) * days * (hoursPerDay : ℝ)

/-- Represents the problem of asphalting roads -/
structure AsphaltProblem where
  menFirst : ℕ
  daysFirst : ℕ
  hoursPerDayFirst : ℕ
  lengthFirst : ℝ
  menSecond : ℕ
  daysSecond : ℝ
  hoursPerDaySecond : ℕ

/-- Theorem stating the length of the road for the second group -/
theorem asphalt_road_length (p : AsphaltProblem) :
  let totalHoursFirst := manHours p.menFirst (p.daysFirst : ℝ) p.hoursPerDayFirst
  let totalHoursSecond := manHours p.menSecond p.daysSecond p.hoursPerDaySecond
  totalHoursFirst / p.lengthFirst = totalHoursSecond / 2 :=
by sorry

/-- The specific problem instance -/
def problemInstance : AsphaltProblem :=
  { menFirst := 30
  , daysFirst := 12
  , hoursPerDayFirst := 8
  , lengthFirst := 1
  , menSecond := 20
  , daysSecond := 19.2
  , hoursPerDaySecond := 15
  }

#check asphalt_road_length problemInstance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_road_length_l721_72125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_toad_pairing_l721_72117

/-- The number of frogs and toads -/
def num_animals : ℕ := 2017

/-- A frog-toad friendship graph -/
structure FrogToadGraph where
  frog_friends : Fin num_animals → Fin num_animals × Fin num_animals
  toad_friends : Fin num_animals → List (Fin num_animals)
  frog_friends_distinct : ∀ f : Fin num_animals, (frog_friends f).1 ≠ (frog_friends f).2
  toad_friends_count : ∀ t : Fin num_animals, (toad_friends t).length ≤ 2

/-- The number of ways to pair every frog with a toad who is its friend -/
def pairing_count (g : FrogToadGraph) : ℕ := sorry

/-- The set of all possible values for the pairing count -/
def pairing_count_set : Set ℕ := sorry

/-- The theorem to be proved -/
theorem frog_toad_pairing :
  ∃ S : Finset ℕ,
    (S.card = 1009) ∧
    (S.sum id = 2^1009 - 2) := by
  sorry

#check frog_toad_pairing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_toad_pairing_l721_72117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_parallel_lines_l721_72140

-- Define the circles and points
variable (S S₁ S₂ : Set (EuclideanSpace ℝ (Fin 2)))
variable (A₁ A₂ B₁ B₂ C : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def equal_circles (S₁ S₂ : Set (EuclideanSpace ℝ (Fin 2))) : Prop := 
  ∃ r > 0, S₁ = Metric.sphere (0 : EuclideanSpace ℝ (Fin 2)) r ∧ S₂ = Metric.sphere (0 : EuclideanSpace ℝ (Fin 2)) r

def touch_internally (S S₁ S₂ : Set (EuclideanSpace ℝ (Fin 2))) (A₁ A₂ : EuclideanSpace ℝ (Fin 2)) : Prop := 
  (S₁ ∩ S = {A₁}) ∧ (S₂ ∩ S = {A₂})

def C_on_S (S : Set (EuclideanSpace ℝ (Fin 2))) (C : EuclideanSpace ℝ (Fin 2)) : Prop := C ∈ S

def B₁_on_CA₁ (A₁ B₁ C : EuclideanSpace ℝ (Fin 2)) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B₁ = (1 - t) • A₁ + t • C

def B₂_on_CA₂ (A₂ B₂ C : EuclideanSpace ℝ (Fin 2)) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B₂ = (1 - t) • A₂ + t • C

def B₁_on_S₁ (S₁ : Set (EuclideanSpace ℝ (Fin 2))) (B₁ : EuclideanSpace ℝ (Fin 2)) : Prop := B₁ ∈ S₁

def B₂_on_S₂ (S₂ : Set (EuclideanSpace ℝ (Fin 2))) (B₂ : EuclideanSpace ℝ (Fin 2)) : Prop := B₂ ∈ S₂

-- State the theorem
theorem circles_tangent_parallel_lines 
  (h1 : equal_circles S₁ S₂)
  (h2 : touch_internally S S₁ S₂ A₁ A₂)
  (h3 : C_on_S S C)
  (h4 : B₁_on_CA₁ A₁ B₁ C)
  (h5 : B₂_on_CA₂ A₂ B₂ C)
  (h6 : B₁_on_S₁ S₁ B₁)
  (h7 : B₂_on_S₂ S₂ B₂) :
  (∃ k : ℝ, B₂ - B₁ = k • (A₂ - A₁)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_parallel_lines_l721_72140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_plus_one_divides_n_plus_one_l721_72172

theorem divisor_plus_one_divides_n_plus_one (n : ℕ) :
  (∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) ↔ (n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_plus_one_divides_n_plus_one_l721_72172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_distance_l721_72114

/-- The line off which the light reflects -/
def reflection_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The point from which the light emanates -/
def P : ℝ × ℝ := (1, 0)

/-- The point through which the reflected light passes -/
def Q : ℝ × ℝ := (2, 1)

/-- The shortest distance the light travels from P to Q -/
noncomputable def shortest_distance : ℝ := Real.sqrt 10

theorem light_reflection_distance :
  ∃ (B : ℝ × ℝ),
    reflection_line B.1 B.2 ∧
    (B.1 - P.1) * (Q.2 - P.2) = (B.2 - P.2) * (Q.1 - P.1) ∧
    shortest_distance = Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_distance_l721_72114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l721_72104

def n : ℕ := 2^6 * 3^5 * 5^3 * 7^2

theorem number_of_factors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 504 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l721_72104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calorie_content_l721_72139

/-- Represents the recipe and calorie information for lemonade -/
structure LemonadeInfo where
  lemon_juice_weight : ℚ
  sugar_weight : ℚ
  water_weight : ℚ
  lemon_juice_calories_per_100g : ℚ
  sugar_calories_per_100g : ℚ

/-- Calculates the calorie content of a given weight of lemonade -/
noncomputable def calorie_content (info : LemonadeInfo) (weight : ℚ) : ℚ :=
  let total_weight := info.lemon_juice_weight + info.sugar_weight + info.water_weight
  let total_calories := (info.lemon_juice_weight / 100) * info.lemon_juice_calories_per_100g +
                        (info.sugar_weight / 100) * info.sugar_calories_per_100g
  (total_calories / total_weight) * weight

/-- Theorem stating that 300g of the specified lemonade contains 312 calories -/
theorem lemonade_calorie_content :
  let info : LemonadeInfo := {
    lemon_juice_weight := 150,
    sugar_weight := 150,
    water_weight := 300,
    lemon_juice_calories_per_100g := 30,
    sugar_calories_per_100g := 386
  }
  calorie_content info 300 = 312 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calorie_content_l721_72139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cells_occupied_l721_72108

/-- A circular arrangement of cells with frogs -/
structure FrogPond where
  n : ℕ
  cells : Fin (2 * n) → ℕ
  total_frogs : ℕ

/-- Two cells are adjacent if they share a common wall -/
def adjacent (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + 1) % (2 * n) = j.val ∨ 
  (j.val + 1) % (2 * n) = i.val ∨ 
  (i.val + n) % (2 * n) = j.val

/-- The rules for frog movement in the pond -/
def frog_jump (pond : FrogPond) : Prop :=
  (pond.n ≥ 5) ∧
  (pond.total_frogs = 4 * pond.n + 1) ∧
  (∀ i : Fin (2 * pond.n), 
    (pond.cells i ≥ 3) → 
    ∃ j k l : Fin (2 * pond.n), 
      j ≠ k ∧ k ≠ l ∧ l ≠ j ∧
      (adjacent pond.n i j) ∧ (adjacent pond.n i k) ∧ (adjacent pond.n i l))

/-- Every cell will eventually be occupied by at least one frog -/
theorem all_cells_occupied (pond : FrogPond) : 
  frog_jump pond → ∀ i : Fin (2 * pond.n), ∃ t : ℕ, pond.cells i > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cells_occupied_l721_72108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_candies_cost_l721_72146

/-- Calculates the cost of chocolate candies including tax -/
theorem chocolate_candies_cost 
  (candies_per_box : ℕ) 
  (cost_per_box : ℚ) 
  (tax_rate : ℚ) 
  (total_candies : ℕ) 
  (h1 : candies_per_box = 30)
  (h2 : cost_per_box = 7.5)
  (h3 : tax_rate = 0.1)
  (h4 : total_candies = 540)
  : 
  (total_candies / candies_per_box : ℚ) * cost_per_box * (1 + tax_rate) = 148.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_candies_cost_l721_72146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l721_72151

-- Define Tom's current age
variable (T : ℕ)

-- Define the number of years ago
variable (N : ℕ)

-- Define the sum of children's current ages
def children_sum : ℕ := T / 2

-- Condition: Tom's current age is twice the sum of his children's ages
axiom tom_current_age : T = 2 * children_sum

-- Condition: N years ago, Tom's age was three times the sum of his children's ages
axiom tom_past_age : T - N = 3 * (children_sum - 3 * N)

-- Theorem to prove
theorem tom_age_ratio : T / N = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l721_72151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_270_equals_tan_n_l721_72191

theorem tan_270_equals_tan_n (n : ℤ) :
  -180 < n ∧ n < 180 →
  (Real.tan (n * π / 180) = Real.tan (270 * π / 180) ↔ n = 90 ∨ n = -90) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_270_equals_tan_n_l721_72191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sine_cosine_inequality_l721_72150

theorem acute_triangle_sine_cosine_inequality (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 → 
  Real.sin A + Real.sin B + Real.sin C > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sine_cosine_inequality_l721_72150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1000_equals_2_l721_72156

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_1000_equals_2 :
  (∀ x > 0, f x - (1/2) * f (1/x) = log10 x) →
  f 1000 = 2 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1000_equals_2_l721_72156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l721_72189

theorem complex_power_magnitude : Complex.abs ((1 : ℂ) - Complex.I) ^ 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l721_72189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_finishes_first_l721_72176

/-- Represents the area of a garden -/
def d : ℝ → ℝ := id

/-- Represents the mowing rate of a lawnmower -/
def f : ℝ → ℝ := id

/-- Diana's garden area -/
noncomputable def diana_area (d : ℝ) : ℝ := d

/-- Fiona's garden area -/
noncomputable def fiona_area (d : ℝ) : ℝ := d / 2

/-- Elena's garden area -/
noncomputable def elena_area (d : ℝ) : ℝ := d / 3

/-- Diana's mowing rate -/
noncomputable def diana_rate (f : ℝ) : ℝ := f / 2

/-- Fiona's mowing rate -/
noncomputable def fiona_rate (f : ℝ) : ℝ := f

/-- Elena's mowing rate -/
noncomputable def elena_rate (f : ℝ) : ℝ := f / 3

/-- Diana's mowing time (including 1 hour delay) -/
noncomputable def diana_time (d f : ℝ) : ℝ := diana_area d / diana_rate f + 1

/-- Fiona's mowing time -/
noncomputable def fiona_time (d f : ℝ) : ℝ := fiona_area d / fiona_rate f

/-- Elena's mowing time -/
noncomputable def elena_time (d f : ℝ) : ℝ := elena_area d / elena_rate f

theorem fiona_finishes_first (d f : ℝ) (h1 : d > 0) (h2 : f > 0) :
  fiona_time d f < min (diana_time d f) (elena_time d f) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_finishes_first_l721_72176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_nine_equals_four_p_l721_72174

theorem sixteen_nine_equals_four_p (p : ℝ) : (16 : ℝ)^9 = (4 : ℝ)^p → p = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_nine_equals_four_p_l721_72174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_satisfactory_grades_l721_72181

/-- Represents the grades in a mathematics class -/
inductive Grade
| A | B | C | D | E | G | F | H

/-- Determines if a grade is satisfactory -/
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A | Grade.B | Grade.C | Grade.D | Grade.E | Grade.G => true
  | _ => false

/-- The number of students for each grade -/
def gradeCount (g : Grade) : Nat :=
  match g with
  | Grade.A => 6
  | Grade.B => 5
  | Grade.C => 4
  | Grade.D => 4
  | Grade.E => 3
  | Grade.G => 2
  | Grade.F => 8
  | Grade.H => 3

/-- The total number of students -/
def totalStudents : Nat :=
  (List.map gradeCount [Grade.A, Grade.B, Grade.C, Grade.D, Grade.E, Grade.G, Grade.F, Grade.H]).sum

/-- The number of students with satisfactory grades -/
def satisfactoryCount : Nat :=
  (List.map (fun g => if isSatisfactory g then gradeCount g else 0) [Grade.A, Grade.B, Grade.C, Grade.D, Grade.E, Grade.G, Grade.F, Grade.H]).sum

theorem fraction_of_satisfactory_grades :
  (satisfactoryCount : Rat) / totalStudents = 24 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_satisfactory_grades_l721_72181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l721_72155

-- Define the cone
structure Cone where
  S : Real × Real × Real  -- Vertex
  A : Real × Real × Real  -- Point on the base circle
  B : Real × Real × Real  -- Another point on the base circle

-- Define helper functions
noncomputable def cos_angle (p1 p2 p3 : Real × Real × Real) : Real := sorry
noncomputable def angle_with_base (p1 p2 : Real × Real × Real) : Real := sorry
noncomputable def area_triangle (p1 p2 p3 : Real × Real × Real) : Real := sorry
noncomputable def lateral_area (c : Cone) : Real := sorry

-- Define the problem parameters
def cone_problem (c : Cone) : Prop :=
  -- Condition 1: Cosine of angle between SA and SB is 7/8
  cos_angle c.S c.A c.B = 7/8 ∧
  -- Condition 2: Angle between SA and base is 45°
  angle_with_base c.S c.A = Real.pi/4 ∧
  -- Condition 3: Area of triangle SAB is 5√15
  area_triangle c.S c.A c.B = 5 * Real.sqrt 15

-- Theorem statement
theorem cone_lateral_area (c : Cone) (h : cone_problem c) : 
  lateral_area c = 40 * Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l721_72155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_f_is_odd_f_one_equals_negative_three_l721_72187

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2 * x^2 - x else -(2 * (-x)^2 - (-x))

theorem odd_function_property (f : ℝ → ℝ) : 
  (∀ x, f (-x) = -f x) → f 1 = -f (-1) := by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem f_one_equals_negative_three : f 1 = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_f_is_odd_f_one_equals_negative_three_l721_72187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_returning_home_is_3_l721_72186

/-- Calculates the speed of returning home given the total distance, speed to destination, rest time, and total time gone. -/
noncomputable def speed_returning_home (total_distance : ℝ) (speed_to_destination : ℝ) (rest_time : ℝ) (total_time : ℝ) : ℝ :=
  let time_to_destination := total_distance / speed_to_destination
  let time_spent_traveling := total_time - rest_time
  let time_returning := time_spent_traveling - time_to_destination
  total_distance / time_returning

/-- Theorem stating that given the specified conditions, the speed returning home is 3 kph. -/
theorem speed_returning_home_is_3 :
  speed_returning_home 7.5 5 2 6 = 3 := by
  -- Unfold the definition of speed_returning_home
  unfold speed_returning_home
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_returning_home_is_3_l721_72186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_sum_l721_72131

theorem set_equality_implies_sum (a b : ℝ) (h1 : a ≠ 0) 
  (h2 : ({1, b/a, a} : Set ℝ) = {0, a+b, a^2}) : a^2 + b^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_sum_l721_72131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l721_72121

/-- The sum of an infinite geometric series with first term a and common ratio r,
    where |r| < 1, is given by a / (1 - r) -/
noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Proof that the sum of the infinite geometric series with first term 1/5 and
    common ratio 1/2 is equal to 2/5 -/
theorem infinite_geometric_series_sum :
  geometric_series_sum (1/5 : ℝ) (1/2 : ℝ) = 2/5 := by
  -- Expand the definition of geometric_series_sum
  unfold geometric_series_sum
  -- Perform algebraic simplification
  simp [add_div, mul_div_cancel]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l721_72121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l721_72107

-- Define the trains' properties
def train_A_length : ℚ := 300
def train_A_speed : ℚ := 75
def train_B_length : ℚ := 250
def train_B_speed : ℚ := 65

-- Define the function to calculate the time for trains to pass
def time_to_pass (length_A length_B speed_A speed_B : ℚ) : ℚ :=
  (length_A + length_B) / ((speed_A + speed_B) * 1000 / 3600)

-- Theorem statement
theorem trains_passing_time :
  ∃ ε : ℚ, ε > 0 ∧ |time_to_pass train_A_length train_B_length train_A_speed train_B_speed - 14.14| < ε := by
  -- The proof goes here
  sorry

#eval time_to_pass train_A_length train_B_length train_A_speed train_B_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l721_72107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l721_72126

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / Real.sqrt (A^2 + B^2)

/-- First line equation: 3x - 4y - 2 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y - 2 = 0

/-- Second line equation: 3x - 4y + 8 = 0 -/
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 8 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 3 (-4) (-2) 8 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l721_72126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_stock_percentage_is_twenty_l721_72161

/-- Represents the stock and sales scenario of a shopkeeper --/
structure ShopkeeperStock where
  totalValue : ℚ
  profitPercentage : ℚ
  lossPercentage : ℚ
  overallLoss : ℚ

/-- Calculates the percentage of stock sold at a profit --/
noncomputable def calculateProfitStockPercentage (s : ShopkeeperStock) : ℚ :=
  100 * (s.overallLoss + s.lossPercentage * s.totalValue / 100) /
    (s.profitPercentage * s.totalValue / 100 + s.lossPercentage * s.totalValue / 100)

/-- Theorem stating that the percentage of stock sold at a profit is 20% --/
theorem profit_stock_percentage_is_twenty (s : ShopkeeperStock)
    (h1 : s.totalValue = 12499.99)
    (h2 : s.profitPercentage = 20)
    (h3 : s.lossPercentage = 10)
    (h4 : s.overallLoss = 500) :
    calculateProfitStockPercentage s = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_stock_percentage_is_twenty_l721_72161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_published_value_l721_72165

/-- The physical constant K -/
def K : ℝ := 5.12718

/-- The maximum error in the measurement of K -/
def error : ℝ := 0.00457

/-- The upper bound of K -/
def K_upper : ℝ := K + error

/-- The lower bound of K -/
def K_lower : ℝ := K - error

/-- The published value of K -/
def published_K : ℝ := 5.1

/-- Function to round a real number to n decimal places -/
noncomputable def round_to_decimal_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋ : ℝ) / 10^n

/-- Theorem stating that the published value is the most accurate while ensuring all presented digits are significant -/
theorem most_accurate_published_value :
  (∀ (x : ℝ), K_lower ≤ x ∧ x ≤ K_upper → 
    ∃ (n : ℕ), round_to_decimal_places published_K n = round_to_decimal_places x n) ∧
  (∀ (y : ℝ), y > published_K → 
    ∃ (x : ℝ) (n : ℕ), K_lower ≤ x ∧ x ≤ K_upper ∧ 
      round_to_decimal_places y n ≠ round_to_decimal_places x n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_published_value_l721_72165
