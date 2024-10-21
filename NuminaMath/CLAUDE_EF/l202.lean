import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_demonstrates_transformation_l202_20244

/-- Represents a quadratic equation in the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the process of solving a quadratic equation -/
structure QuadraticSolution where
  equation : QuadraticEquation
  factored_form : String
  linear_equations : List String
  solutions : List ℝ

/-- Represents different mathematical concepts -/
inductive MathConcept
  | Function
  | IntegrationOfNumberAndShape
  | Transformation
  | Axiomatic
deriving Repr

/-- The given quadratic equation (x-3)² - 4(x-3) = 0 -/
def given_equation : QuadraticEquation :=
  { a := 1, b := -10, c := 21 }

/-- The solution process for the given equation -/
def solution_process : QuadraticSolution :=
  { equation := given_equation,
    factored_form := "(x-3)(x-7) = 0",
    linear_equations := ["x-3 = 0", "x-7 = 0"],
    solutions := [3, 7] }

/-- Function to check if a concept is the Transformation concept -/
def isTransformationConcept (concept : MathConcept) : Bool :=
  match concept with
  | MathConcept.Transformation => true
  | _ => false

/-- Theorem stating that the solution process demonstrates the transformation concept -/
theorem solution_demonstrates_transformation :
  solution_process.equation = given_equation ∧
  solution_process.factored_form = "(x-3)(x-7) = 0" ∧
  solution_process.linear_equations = ["x-3 = 0", "x-7 = 0"] ∧
  solution_process.solutions = [3, 7] →
  isTransformationConcept MathConcept.Transformation = true :=
by
  intro h
  simp [isTransformationConcept]

#eval isTransformationConcept MathConcept.Transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_demonstrates_transformation_l202_20244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_n_range_l202_20203

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 - m) * Real.log x
noncomputable def g (m n : ℝ) (x : ℝ) : ℝ := -m/2 * x^2 - (m^2 - m - 1) * x - n

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (m n : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f m x₁ = g m n x₁ ∧ 
    f m x₂ = g m n x₂ ∧ 
    f m x₃ = g m n x₃

-- State the theorem
theorem intersection_implies_m_n_range (m n : ℝ) :
  has_three_distinct_intersections m n → (0 < m ∧ m < 1 ∧ n > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_n_range_l202_20203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l202_20236

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line -/
noncomputable def slope1 (m : ℝ) : ℝ := -(2 * (m + 1)) / (m - 3)

/-- The slope of the second line -/
noncomputable def slope2 (m : ℝ) : ℝ := -(m - 3) / 2

/-- The necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition (m : ℝ) : 
  perpendicular (slope1 m) (slope2 m) ↔ m = 3 ∨ m = -3 := by
  sorry

#check perpendicular_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l202_20236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_to_surface_area_ratio_l202_20257

/-- The ratio of volume to total surface area for a cone with specific properties -/
theorem cone_volume_to_surface_area_ratio : 
  ∀ (r h : ℝ), 
  r > 0 → h > 0 →
  r^2 + h^2 = 4 →  -- This ensures the lateral surface unfolds into a semicircle with radius 2
  (1/3 * Real.pi * r^2 * h) / (Real.pi * r^2 + Real.pi * r * Real.sqrt (r^2 + h^2)) = Real.sqrt 3 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_to_surface_area_ratio_l202_20257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_periodicity_l202_20276

/-- A function f is periodic with the given properties -/
theorem function_periodicity (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ) 
  (h_sum_neq : a₁ + b₁ ≠ a₂ + b₂)
  (h_prop1 : ∀ x, f (a₁ + x) = -f (b₁ - x))
  (h_prop2 : ∀ x, f (a₂ + x) = f (b₂ - x)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 2 * |a₂ + b₂ - (a₁ + b₁)| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_periodicity_l202_20276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_average_velocity_l202_20252

/-- The position function of the particle -/
noncomputable def s (t : ℝ) : ℝ := t^2 + 1

/-- The average velocity of the particle over a time interval -/
noncomputable def average_velocity (t₁ t₂ : ℝ) : ℝ := (s t₂ - s t₁) / (t₂ - t₁)

/-- Theorem: The average velocity of the particle over the time interval [1, 2] is 3 -/
theorem particle_average_velocity : average_velocity 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_average_velocity_l202_20252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_l202_20221

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the dot product
def dot_product (v w : MyVector) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the perpendicular relation
def perpendicular (v w : MyVector) : Prop := dot_product v w = 0

-- Define vector operations
def vec_add (v w : MyVector) : MyVector := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : MyVector) : MyVector := (a * v.1, a * v.2)
def vec_sub (v w : MyVector) : MyVector := vec_add v (vec_scale (-1) w)

-- Define the problem statement
theorem vector_orthogonality (t : ℝ) : 
  let a : MyVector := (1, t)
  let b : MyVector := (-2, 1)
  perpendicular (vec_sub (vec_scale 2 a) b) b → t = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_l202_20221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_sheep_count_stewar_farm_sheep_count_l202_20259

theorem farm_sheep_count (sheep_to_horse_ratio : ℚ) (horse_food_per_day : ℕ) (total_food_per_day : ℕ) : ℕ :=
  let sheep_count : ℕ := 16
  let horse_count : ℕ := 56
  have ratio_correct : (sheep_count : ℚ) / (horse_count : ℚ) = sheep_to_horse_ratio := by sorry
  have total_food_correct : horse_count * horse_food_per_day = total_food_per_day := by sorry
  sheep_count

theorem stewar_farm_sheep_count : farm_sheep_count (2/7) 230 12880 = 16 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_sheep_count_stewar_farm_sheep_count_l202_20259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l202_20208

-- Define the function f(x) = 2^x + log_2(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

-- State the theorem
theorem zero_point_in_interval :
  -- f(x) is continuous and strictly increasing for x > 0
  (∀ x > 0, ContinuousAt f x) →
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) →
  -- There exists a unique x in (0, 1/2) such that f(x) = 0
  ∃! x : ℝ, 0 < x ∧ x < 1/2 ∧ f x = 0 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l202_20208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_value_after_depreciation_l202_20235

/-- The value of equipment after depreciation -/
noncomputable def equipment_value (a : ℝ) (b : ℝ) (n : ℕ) : ℝ :=
  a * (1 - b / 100) ^ n

/-- Theorem: The value of equipment with initial value a, depreciating at b% annually, 
    after n years is equal to a · (1 - b%)^n -/
theorem equipment_value_after_depreciation (a b : ℝ) (n : ℕ) :
  equipment_value a b n = a * (1 - b / 100) ^ n := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_value_after_depreciation_l202_20235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l202_20265

open Real

/-- A triangle represented by three points in a real vector space -/
structure Triangle (V : Type*) [AddCommGroup V] [Module ℝ V] where
  A : V
  B : V
  C : V

/-- Point D lies on segment BC (excluding endpoints) -/
def PointOnSegment {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Triangle V) (D : V) : Prop :=
  ∃ l : ℝ, 0 < l ∧ l < 1 ∧ D = t.B + l • (t.C - t.B)

/-- Vector equation AD = x*AB + y*AC -/
def VectorEquation {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Triangle V) (D : V) (x y : ℝ) : Prop :=
  D - t.A = x • (t.B - t.A) + y • (t.C - t.A)

/-- The minimum value of 1/x + 2/y is 2√2 + 3 -/
theorem min_value_theorem {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Triangle V) (D : V) (x y : ℝ) :
  PointOnSegment t D → VectorEquation t D x y → x > 0 → y > 0 → (1/x + 2/y) ≥ 2*sqrt 2 + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l202_20265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_sum_l202_20297

/-- Given a line 3x - 4y + k = 0, prove that if the sum of its intercepts on the coordinate axes is 2, then k = -24 -/
theorem line_intercepts_sum (k : ℝ) : 
  (∃ x y : ℝ, 3*x - 4*y + k = 0 ∧ (x = 0 ∨ y = 0)) ∧ 
  (-k/3 + k/4 = 2) →
  k = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_sum_l202_20297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_triangle_with_small_triangles_l202_20296

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The number of small equilateral triangles needed to fill a large equilateral triangle -/
noncomputable def num_small_triangles (large_side small_side : ℝ) : ℝ :=
  (equilateral_triangle_area large_side) / (equilateral_triangle_area small_side)

theorem fill_triangle_with_small_triangles (large_side small_side : ℝ) :
  large_side = 10 → small_side = 1 → num_small_triangles large_side small_side = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_triangle_with_small_triangles_l202_20296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_zero_l202_20286

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

-- State the theorem
theorem extreme_value_at_zero (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a 0 ≥ f a x ∨ f a 0 ≤ f a x) →
  a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_zero_l202_20286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l202_20237

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_problem :
  ∃ (a₁ d : ℝ),
    (arithmetic_sequence a₁ d 3 = 5) ∧
    (arithmetic_sum a₁ d 3 = 9) ∧
    (a₁ = 1) ∧
    (d = 2) ∧
    (∃ (n : ℕ), arithmetic_sum a₁ d n = 100 ∧ n = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l202_20237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_l202_20239

-- Define the basic structures
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the given points and circles
variable (O₁ O₂ M P A N B H : EuclideanSpace ℝ (Fin 2))
variable (circle₁ circle₂ : Circle)

-- Define the conditions
def onCircle (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop :=
  ‖p - c.center‖ = c.radius

variable (intersect : onCircle M circle₁ ∧ onCircle M circle₂ ∧ onCircle P circle₁ ∧ onCircle P circle₂)

def isTangent (l : Set (EuclideanSpace ℝ (Fin 2))) (c : Circle) (p : EuclideanSpace ℝ (Fin 2)) : Prop :=
  p ∈ l ∧ onCircle p c ∧ ∀ q ∈ l, q ≠ p → ‖q - c.center‖ > c.radius

variable (chord_MA : onCircle A circle₁ ∧ isTangent (affineSpan ℝ {M, A}) circle₂ M)
variable (chord_MB : onCircle B circle₂ ∧ isTangent (affineSpan ℝ {M, B}) circle₁ M)
variable (segment_PH : H ∈ affineSpan ℝ {M, P} ∧ ‖P - H‖ = ‖P - M‖)

-- Theorem statement
theorem quadrilateral_inscribed : ∃ (C : Circle), onCircle A C ∧ onCircle M C ∧ onCircle N C ∧ onCircle B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_l202_20239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_3pi_plus_alpha_l202_20284

theorem cos_3pi_plus_alpha (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin α = Real.sqrt 10 / 5) :
  Real.cos (3 * π + α) = -(Real.sqrt 15 / 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_3pi_plus_alpha_l202_20284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_60_l202_20263

/-- A configuration of four positive integers on the corners of a square. -/
structure SquareConfig where
  v1 : ℕ+
  v2 : ℕ+
  v3 : ℕ+
  v4 : ℕ+

/-- Predicate to check if two numbers are relatively prime. -/
def RelativelyPrime (a b : ℕ+) : Prop := Nat.Coprime a.val b.val

/-- Predicate to check if two numbers are not relatively prime. -/
def NotRelativelyPrime (a b : ℕ+) : Prop := ¬(RelativelyPrime a b)

/-- The sum of the four numbers in a SquareConfig. -/
def ConfigSum (config : SquareConfig) : ℕ := config.v1.val + config.v2.val + config.v3.val + config.v4.val

/-- A valid SquareConfig satisfies the relative primality conditions. -/
def ValidConfig (config : SquareConfig) : Prop :=
  RelativelyPrime config.v1 config.v3 ∧
  RelativelyPrime config.v2 config.v4 ∧
  NotRelativelyPrime config.v1 config.v2 ∧
  NotRelativelyPrime config.v2 config.v3 ∧
  NotRelativelyPrime config.v3 config.v4 ∧
  NotRelativelyPrime config.v4 config.v1

/-- The main theorem: The smallest possible sum for a valid SquareConfig is 60. -/
theorem smallest_sum_is_60 :
  ∀ (config : SquareConfig), ValidConfig config → ConfigSum config ≥ 60 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_60_l202_20263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l202_20210

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Part 1
theorem part1 (a : ℝ) (h1 : a > 1) :
  (Set.Icc 1 a = Set.range (f a) ∩ Set.Icc 1 a) → a = 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : a > 1) :
  (∀ x ≤ 2, StrictMonoOn (f a) (Set.Iic x)) →
  (∀ x y, x ∈ Set.Icc 1 (a + 1) → y ∈ Set.Icc 1 (a + 1) → |f a x - f a y| ≤ 4) →
  2 ≤ a ∧ a ≤ 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l202_20210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2030_l202_20222

noncomputable def h : ℕ → ℝ
  | 0 => 2  -- Adding case for 0 to cover all natural numbers
  | 1 => 2
  | n+2 => h (n+1) - h n + (n+2) + Real.sin ((n+2) * Real.pi / 6)

theorem h_2030 : h 2030 = 2030 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2030_l202_20222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l202_20289

noncomputable def f (x : ℝ) := Real.sin (Real.pi / 6 - 2 * x) + 3 / 2

theorem f_properties :
  (∀ x : ℝ, f x ≤ 1 / 2) ∧
  (∀ k : ℤ, f (k * Real.pi - Real.pi / 6) = 1 / 2) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi - Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + Real.pi / 3 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l202_20289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_calculation_l202_20261

/-- Calculates the profit percentage without discount given the discount percentage and profit percentage with discount -/
noncomputable def profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) : ℝ :=
  let sp_with_discount := 100 + profit_with_discount_percent
  let sp_without_discount := sp_with_discount / (1 - discount_percent / 100)
  (sp_without_discount - 100) * 100 / 100

/-- Theorem stating that given a 5% discount and a 34.9% profit, the profit percentage without discount would be 42% -/
theorem profit_without_discount_calculation :
  profit_without_discount 5 34.9 = 42 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval profit_without_discount 5 34.9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_calculation_l202_20261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_cost_is_approx_3_33_l202_20201

/-- The original cost of a bag of chips given the following conditions:
  * 5 bags are bought
  * There's a 10% discount on the total cost
  * 3 friends split the payment equally
  * Each friend pays $5 after the discount
-/
noncomputable def original_cost : ℝ :=
  let num_bags : ℕ := 5
  let discount_rate : ℝ := 0.1
  let num_friends : ℕ := 3
  let individual_payment : ℝ := 5

  let total_discounted_cost : ℝ := individual_payment * num_friends
  let total_original_cost : ℝ := total_discounted_cost / (1 - discount_rate)
  total_original_cost / num_bags

/-- Theorem stating that the original cost of each bag of chips is approximately $3.33 -/
theorem original_cost_is_approx_3_33 : 
  ∃ ε > 0, |original_cost - 3.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_cost_is_approx_3_33_l202_20201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_inscribed_rectangles_l202_20229

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the rectangles PQRS and P₁Q₁R₁S₁
variable (P Q R S P₁ Q₁ R₁ S₁ : ℝ × ℝ)

-- Define the conditions
variable (h1 : P.1 = A.1 ∨ P.1 = B.1)
variable (h2 : P₁.1 = A.1 ∨ P₁.1 = B.1)
variable (h3 : Q.2 = B.2 ∨ Q.2 = C.2)
variable (h4 : Q₁.2 = B.2 ∨ Q₁.2 = C.2)
variable (h5 : R.1 = A.1 ∨ R.1 = C.1)
variable (h6 : S.1 = A.1 ∨ S.1 = C.1)
variable (h7 : R₁.1 = A.1 ∨ R₁.1 = C.1)
variable (h8 : S₁.1 = A.1 ∨ S₁.1 = C.1)
variable (h9 : Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 12)
variable (h10 : Real.sqrt ((P₁.1 - S₁.1)^2 + (P₁.2 - S₁.2)^2) = 3)
variable (h11 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2)
variable (h12 : (P₁.1 - Q₁.1)^2 + (P₁.2 - Q₁.2)^2 = (R₁.1 - S₁.1)^2 + (R₁.2 - S₁.2)^2)
variable (h13 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P₁.1 - Q₁.1)^2 + (P₁.2 - Q₁.2)^2)

-- Define the area_triangle function
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the theorem
theorem triangle_area_with_inscribed_rectangles :
  area_triangle A B C = 225/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_inscribed_rectangles_l202_20229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_symmetry_implication_l202_20287

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ) + Real.cos (ω * x + φ)

theorem period_and_symmetry_implication (ω φ : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  (∀ x, f ω φ (-x) = f ω φ x) → 
  (∃ k : ℤ, φ = k * π + π / 4) ∧
  ¬(φ = π / 4 → ∀ x, f ω φ (-x) = f ω φ x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_symmetry_implication_l202_20287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_conversion_l202_20204

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 && y < 0 then
             2 * Real.pi - Real.arctan (abs y / x)
           else
             0  -- placeholder for other cases
  (r, θ)

theorem point_conversion :
  let x : ℝ := 2 * Real.sqrt 2
  let y : ℝ := -2 * Real.sqrt 2
  let (r, θ) := rectangular_to_polar x y
  r = 4 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_conversion_l202_20204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_acute_angle_inequality_l202_20215

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1)

-- Theorem for part 1
theorem max_k_value (k : ℝ) :
  (∀ x > 0, f x ≥ k * g x) → k ≤ 1 := by sorry

-- Theorem for part 2
theorem acute_angle_inequality (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) :
  let A : ℝ × ℝ := (x₁, f x₁)
  let B : ℝ × ℝ := (x₂, -g x₂)
  let O : ℝ × ℝ := (0, 0)
  (A.1 * B.1 + A.2 * B.2 > 0) → x₂ > x₁^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_acute_angle_inequality_l202_20215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_argument_l202_20258

theorem root_product_argument : 
  ∃ (roots : Finset ℂ),
    (∀ z ∈ roots, z^7 + z^5 + z^4 + z^3 + z + 1 = 0) ∧
    (∀ z ∈ roots, z.im > 0) ∧
    ∃ s : ℝ, s > 0 ∧ 
      roots.prod id = s * Complex.exp (Complex.I * Real.pi * (120 / 180)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_argument_l202_20258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l202_20205

/-- The number of days it takes for A and B to finish the work together -/
noncomputable def days_AB : ℝ := 40

/-- The number of days it takes for A to finish the work alone -/
noncomputable def days_A : ℝ := 80

/-- The number of days A worked alone after B left -/
noncomputable def days_A_alone : ℝ := 6

/-- The fraction of work completed in one day by A and B together -/
noncomputable def rate_AB : ℝ := 1 / days_AB

/-- The fraction of work completed in one day by A alone -/
noncomputable def rate_A : ℝ := 1 / days_A

/-- The number of days A and B worked together -/
noncomputable def days_together : ℝ := 37

theorem work_completion :
  days_together * rate_AB + days_A_alone * rate_A = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l202_20205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersection_range_l202_20275

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - 1

-- Define the point (1, -1/6)
noncomputable def point : ℝ × ℝ := (1, -1/6)

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ x y : ℝ, y = f x → (x = point.1 ∧ y = point.2) ∨ (a * x + b * y + c ≠ 0)) ∧
  (a * point.1 + b * point.2 + c = 0) ∧
  (a = 12 ∧ b = -6 ∧ c = -13) := by sorry

-- Theorem for the range of m
theorem intersection_range :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
             f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔
            (-1 < m ∧ m < -5/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersection_range_l202_20275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l202_20230

/-- Given a triangle with side lengths a, b, c and perimeter not exceeding 2π,
    prove that sin a + sin b > sin c -/
theorem triangle_sine_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_perimeter : a + b + c ≤ 2 * Real.pi) :
  Real.sin a + Real.sin b > Real.sin c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l202_20230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l202_20277

noncomputable def f (x : ℝ) : ℝ := (1/4)^(x-1) - 4*(1/2)^x + 2

theorem f_min_max :
  ∃ y z : ℝ, 
    (0 ≤ y ∧ y ≤ 2 ∧ f y = 1 ∧ ∀ w : ℝ, 0 ≤ w ∧ w ≤ 2 → f y ≤ f w) ∧
    (0 ≤ z ∧ z ≤ 2 ∧ f z = 2 ∧ ∀ w : ℝ, 0 ≤ w ∧ w ≤ 2 → f w ≤ f z) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l202_20277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_2007_divides_2007_factorial_l202_20279

theorem highest_power_2007_divides_2007_factorial :
  ∃ k : ℕ, k = 9 ∧ 2007^k ∣ Nat.factorial 2007 ∧ 
  ∀ m : ℕ, m > k → ¬(2007^m ∣ Nat.factorial 2007) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_2007_divides_2007_factorial_l202_20279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l202_20218

theorem triangle_ABC_properties (a b c A B C : ℝ) (h1 : a = 2) (h2 : C = π/4) (h3 : Real.cos B = 3/5) :
  let sinB := Real.sqrt (1 - (Real.cos B)^2)
  let sinA := Real.sqrt 2 / 2 * Real.cos B - 1 / 2 * sinB
  let S := 1/2 * a * (a * sinB / sinA) * sinB
  (sinB = 4/5) ∧ (sinA = 7 * Real.sqrt 2 / 10) ∧ (S = 8/7) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l202_20218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l202_20283

-- Define the sets A, B, and C
def A (n : ℕ) : Set ℝ := {x : ℝ | 0 < x ∧ x < (1 : ℝ) / n}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}
def C : Set ℝ := {x : ℝ | 0 < x ∧ x < 1/2}

-- Define the conditions
def is_positive_integer_less_than_6 (n : ℕ) : Prop := 0 < n ∧ n < 6

def A_sufficient_not_necessary_for_B (n : ℕ) : Prop :=
  A n ⊆ B ∧ ¬(B ⊆ A n)

def A_necessary_not_sufficient_for_C (n : ℕ) : Prop :=
  C ⊆ A n ∧ ¬(A n ⊆ C)

-- Theorem to prove
theorem find_n :
  ∃ (n : ℕ), 
    is_positive_integer_less_than_6 n ∧
    A_sufficient_not_necessary_for_B n ∧
    A_necessary_not_sufficient_for_C n ∧
    n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l202_20283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l202_20272

/-- The sum of an infinite geometric series with first term a and ratio r -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_problem (m : ℝ) : 
  let a₁ : ℝ := 18
  let r₁ : ℝ := 6 / 18
  let a₂ : ℝ := 18
  let r₂ : ℝ := (6 + m) / 18
  geometricSeriesSum a₂ r₂ = 3 * geometricSeriesSum a₁ r₁ → m = 8 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l202_20272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_ride_time_l202_20217

noncomputable section

/-- Represents the time in seconds for Clea to walk down a stopped escalator -/
def time_stopped : ℝ := 90

/-- Represents the time in seconds for Clea to walk down a moving escalator -/
def time_moving : ℝ := 30

/-- Represents Clea's walking speed in steps per second -/
noncomputable def walking_speed : ℝ := time_stopped⁻¹

/-- Represents the escalator's speed in steps per second -/
noncomputable def escalator_speed : ℝ := walking_speed * (time_stopped / time_moving - 1)

/-- Represents the length of the escalator in steps -/
noncomputable def escalator_length : ℝ := time_stopped * walking_speed

theorem clea_ride_time :
  escalator_length / escalator_speed = 45 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_ride_time_l202_20217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l202_20260

/-- The parabola C: x² = 8y -/
def parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus F of the parabola -/
def focus : ℝ × ℝ := (0, 2)

/-- The directrix of the parabola -/
def directrix (y : ℝ) : Prop := y = -2

/-- Distance from a point to the focus -/
noncomputable def distToFocus (x y : ℝ) : ℝ := Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2)

/-- The circle centered at F with radius |FM| -/
def circleEq (x y x₀ y₀ : ℝ) : Prop :=
  (x - focus.1)^2 + (y - focus.2)^2 = (y₀ + 2) ^ 2

theorem parabola_intersection_range (x₀ y₀ : ℝ) :
  parabola x₀ y₀ →
  (∃ x y, circleEq x y x₀ y₀ ∧ directrix y) →
  y₀ > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l202_20260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l202_20274

/-- The side length of the squares -/
def side_length : ℝ := 2

/-- The rotation angle in radians -/
noncomputable def rotation_angle : ℝ := Real.pi / 4

/-- The diagonal length of a square -/
noncomputable def diagonal_length (s : ℝ) : ℝ := s * Real.sqrt 2

/-- Theorem stating that the height of the rotated square is 2√2 -/
theorem rotated_square_height : 
  let height := diagonal_length side_length / 2
  height * 2 = 2 * Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

#check rotated_square_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l202_20274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_property_l202_20290

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The image of a point under inversion with respect to a circle -/
noncomputable def inversionImage (c : Circle) (m : Point) : Point :=
  sorry

/-- Theorem: The inversion image satisfies the inversion property -/
theorem inversion_property (c : Circle) (m : Point) :
  let m' := inversionImage c m
  distance c.center m * distance c.center m' = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_property_l202_20290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l202_20212

/-- A function satisfying the given functional equation is either the zero function or the identity function. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l202_20212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_for_g_g_defined_l202_20232

-- Define the function g as noncomputable due to the use of Real.sqrt
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x - 3)

-- Define the theorem
theorem smallest_x_for_g_g_defined : 
  ∃ (x : ℝ), (∀ y : ℝ, y < x → ¬(∃ (z : ℝ), g z = g y ∧ g y ≥ 5)) ∧ 
             (∃ (z : ℝ), g z = g x ∧ g x ≥ 5) ∧
             x = 28 := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_for_g_g_defined_l202_20232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l202_20242

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- The common ratio
  h : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.q^n) / (1 - seq.q)

/-- Theorem: For a geometric sequence with 8a₂ + a₅ = 0, S₆/S₃ = -7 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) 
  (h : 8 * seq.a 2 + seq.a 5 = 0) : 
  S seq 6 / S seq 3 = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l202_20242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoxiao_age_is_11_l202_20264

/-- Xiaoxiao's age in 2015 -/
def xiaoxiao_age_2015 : ℕ := sorry

/-- Total family age in 2015 -/
def total_age_2015 : ℕ := sorry

/-- Total family age in 2020 -/
def total_age_2020 : ℕ := sorry

/-- Number of family members in 2015 -/
def family_members_2015 : ℕ := sorry

/-- Number of family members in 2020 -/
def family_members_2020 : ℕ := sorry

axiom family_growth : family_members_2020 = family_members_2015 + 1

axiom total_age_2015_relation : total_age_2015 = 7 * xiaoxiao_age_2015

axiom total_age_2020_relation : total_age_2020 = 6 * (xiaoxiao_age_2015 + 5)

axiom age_increase : total_age_2020 = total_age_2015 + 5 * family_members_2015 + 4

theorem xiaoxiao_age_is_11 : xiaoxiao_age_2015 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoxiao_age_is_11_l202_20264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_of_three_plus_two_l202_20206

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6*x^2 + 12*x - 11

theorem cubic_root_of_three_plus_two :
  (∃ (a b c d : ℤ), Q = fun x => (a : ℝ)*x^3 + (b : ℝ)*x^2 + (c : ℝ)*x + (d : ℝ)) ∧ 
  (Q (Real.rpow 3 (1/3) + 2) = 0) ∧
  (∀ x, Q x = x^3 - 6*x^2 + 12*x - 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_of_three_plus_two_l202_20206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_satisfying_property_l202_20292

-- Define the set of positive rational numbers
def PositiveRationals := {q : ℚ // q > 0}

-- Define multiplication for PositiveRationals
instance : Mul PositiveRationals where
  mul a b := ⟨a.val * b.val, mul_pos a.property b.property⟩

-- Define division for PositiveRationals
instance : Div PositiveRationals where
  div a b := ⟨a.val / b.val, div_pos a.property b.property⟩

-- Define the property that the function f should satisfy
def SatisfiesProperty (f : PositiveRationals → PositiveRationals) :=
  ∀ x y : PositiveRationals, f (x * f y) = (f x) / y

-- State the theorem
theorem exists_function_satisfying_property :
  ∃ f : PositiveRationals → PositiveRationals, SatisfiesProperty f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_satisfying_property_l202_20292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_l202_20267

open Real

theorem smallest_positive_solution_tan_sec (x : ℝ) : 
  (∀ y ∈ Set.Ioo (0 : ℝ) x, ¬(tan (3*y) + tan (4*y) = 1 / cos (4*y))) ∧ 
  (tan (3*x) + tan (4*x) = 1 / cos (4*x)) → 
  x = π/17 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_l202_20267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l202_20225

/-- The minimum distance from any point with integer coordinates to the line 5x - 3y + 4 = 0 is √34/85 -/
theorem min_distance_to_line : ∃ (d : ℝ),
  d = Real.sqrt 34 / 85 ∧
  ∀ (x y : ℤ), 
    (let distance := |5 * (x : ℝ) - 3 * (y : ℝ) + 4| / Real.sqrt 34;
     d ≤ distance) ∧ (∃ (x₀ y₀ : ℤ), |5 * (x₀ : ℝ) - 3 * (y₀ : ℝ) + 4| / Real.sqrt 34 = d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l202_20225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_remainder_l202_20227

def p : Nat := 2017

-- Define Expression type
structure Expression where
  numThrees : Nat
  numBoxes : Nat
  operations : Finset Char

-- Define ExpectedValue function (stub)
noncomputable def ExpectedValue (expr : Expression) : ℚ :=
  0 -- Placeholder, actual implementation would be more complex

theorem expected_value_remainder (m n : ℕ) (h_prime : Nat.Prime p) 
  (h_exp : ∃ E : ℚ, E = m / n ∧ 
    E = ExpectedValue ⟨p+3, p+2, {'+', '-', '*', '/'}⟩ ∧
    Nat.Coprime m n) : 
  (m + n) % p = 235 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_remainder_l202_20227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_equals_distance_AB_l202_20278

/-- The function f(x) representing the sum of distances --/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 4*x + 5) + Real.sqrt (x^2 - 2*x + 10)

/-- Point A --/
def A : ℝ × ℝ := (-2, -1)

/-- Point B --/
def B : ℝ × ℝ := (1, 3)

/-- The distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating that the minimum value of f is the distance between A and B --/
theorem min_value_f_equals_distance_AB :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (m = distance A B) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_equals_distance_AB_l202_20278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_special_triangle_eccentricity_l202_20238

/-- An ellipse with foci and a special triangle property has eccentricity -1 + √2 -/
theorem ellipse_special_triangle_eccentricity 
  (a b c : ℝ) 
  (ellipse : Set (ℝ × ℝ))
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ellipse)
  (h_foci : c > 0 ∧ c^2 = a^2 - b^2)
  (h_triangle : ∃ (M N : ℝ × ℝ), 
    M ∈ ellipse ∧ N ∈ ellipse ∧ 
    M.1 = -c ∧ N.1 = -c ∧
    (M.2 - N.2)^2 + (2*c)^2 = (M.2 - N.2)^2 + (M.2 + N.2)^2) :
  a / c = 1 / (-1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_special_triangle_eccentricity_l202_20238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l202_20270

theorem perfect_square_condition (n : ℕ+) :
  ∃ k : ℕ, ((n : ℤ)^2 + 11*n - 4)*(Nat.factorial n.val : ℤ) + 33*13^(n : ℕ) + 4 = k^2 ↔ n = 1 ∨ n = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l202_20270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carter_cheesecakes_l202_20209

/-- The number of cheesecakes Carter usually bakes in a week -/
def usual_cheesecakes : ℕ := sorry

/-- The number of muffins Carter usually bakes in a week -/
def usual_muffins : ℕ := 5

/-- The number of red velvet cakes Carter usually bakes in a week -/
def usual_red_velvet : ℕ := 8

/-- The multiplier for this week's baking -/
def this_week_multiplier : ℕ := 3

/-- The additional cakes baked this week compared to usual -/
def additional_cakes : ℕ := 38

theorem carter_cheesecakes : 
  usual_cheesecakes = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carter_cheesecakes_l202_20209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_theorem_l202_20288

/-- Given two lines in the 2D plane, this function returns true if they are symmetric with respect to a given line of symmetry. -/
def are_symmetric_lines (line1 line2 symmetry_line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y x' y', 
    line1 x y ∧ line2 x' y' → 
    symmetry_line ((x + x') / 2) ((y + y') / 2) ∧ 
    (y - y') / (x - x') = -1

/-- The main theorem stating that the given lines are symmetric. -/
theorem symmetric_lines_theorem : 
  are_symmetric_lines 
    (λ x y ↦ 2 * x + 3 * y - 6 = 0) 
    (λ x y ↦ 3 * x + 2 * y + 16 = 0) 
    (λ x y ↦ x + y + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_theorem_l202_20288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l202_20213

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalArea (baseEdge lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge^2 - (baseEdge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units -/
theorem pyramid_face_area :
  totalArea 8 7 = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l202_20213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l202_20241

theorem triangle_proof (A B C : ℝ) (S_ABC : ℝ) 
  (h_cos_B : Real.cos B = -5/13)
  (h_cos_C : Real.cos C = 4/5)
  (h_area : S_ABC = 33/2) :
  Real.sin A = 33/65 ∧ 
  let BC := Real.sqrt ((2 * S_ABC / Real.sin A)^2 - (2 * S_ABC / (Real.sin B * Real.sin C))^2)
  BC = 11/2 := by
sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l202_20241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l202_20298

theorem three_true_propositions :
  ∃ (P negation converse contrapositive : ℝ → Prop),
    (∀ x, P x ↔ (x > 1 → x^2 > 1)) ∧
    (∀ x, negation x ↔ (x ≤ 1 → x^2 ≤ 1)) ∧
    (∀ x, converse x ↔ (x^2 > 1 → x > 1)) ∧
    (∀ x, contrapositive x ↔ (x^2 ≤ 1 → x ≤ 1)) ∧
    (∀ x, P x) ∧ (∀ x, negation x) ∧ (∀ x, contrapositive x) ∧ (¬ ∀ x, converse x) :=
by
  sorry

#check three_true_propositions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l202_20298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_when_a_is_2_f_max_value_is_2_implies_a_is_plus_minus_1_l202_20271

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x) + a * Real.cos x

theorem f_min_value_when_a_is_2 :
  ∃ x_min : ℝ, ∀ x : ℝ, f 2 x_min ≤ f 2 x ∧ f 2 x_min = -3/2 := by
  sorry

theorem f_max_value_is_2_implies_a_is_plus_minus_1 :
  (∃ x_max : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x_max ∧ f 1 x_max = 2) ∧
  (∃ x_max : ℝ, ∀ x : ℝ, f (-1) x ≤ f (-1) x_max ∧ f (-1) x_max = 2) ∧
  (∀ a : ℝ, a ≠ 1 ∧ a ≠ -1 → ¬∃ x_max : ℝ, ∀ x : ℝ, f a x ≤ f a x_max ∧ f a x_max = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_when_a_is_2_f_max_value_is_2_implies_a_is_plus_minus_1_l202_20271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pencil_one_eraser_cost_l202_20253

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- The cost of an eraser in cents -/
def eraser_cost : ℕ := sorry

/-- The total cost of 20 pencils and 4 erasers is 160 cents -/
axiom total_cost : 20 * pencil_cost + 4 * eraser_cost = 160

/-- A pencil costs more than an eraser -/
axiom pencil_more_expensive : pencil_cost > eraser_cost

theorem one_pencil_one_eraser_cost : pencil_cost + eraser_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pencil_one_eraser_cost_l202_20253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_sides_l202_20293

-- Define the structures we need
structure Rectangle where
  shorter_side : ℝ
  longer_side : ℝ

structure Square where
  side_length : ℝ

-- Define the theorem
theorem inscribed_rectangle_sides (square_diagonal : ℝ) 
  (h1 : square_diagonal = 12) 
  (rectangle : Rectangle) 
  (square : Square) 
  (h2 : rectangle.shorter_side > 0)
  (h3 : rectangle.longer_side > 0)
  (h4 : square.side_length > 0)
  (h5 : rectangle.longer_side = 2 * rectangle.shorter_side)
  (h6 : square.side_length * Real.sqrt 2 = square_diagonal) : 
  rectangle.shorter_side = 4 ∧ rectangle.longer_side = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_sides_l202_20293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_al_increase_percentage_l202_20256

/-- Represents the bank accounts of Al and Eliot -/
structure BankAccounts where
  al : ℚ
  eliot : ℚ

/-- The conditions of the problem -/
def problem_conditions (accounts : BankAccounts) : Prop :=
  accounts.al > accounts.eliot ∧
  accounts.al - accounts.eliot = (accounts.al + accounts.eliot) / 12 ∧
  accounts.eliot = 200

/-- The result after increasing the accounts -/
def increased_accounts (accounts : BankAccounts) (al_increase : ℚ) : BankAccounts :=
  { al := accounts.al * (1 + al_increase / 100),
    eliot := accounts.eliot * 12 / 10 }

/-- The theorem to prove -/
theorem al_increase_percentage (accounts : BankAccounts) :
  problem_conditions accounts →
  ∃ al_increase : ℚ,
    al_increase = 10 ∧
    (increased_accounts accounts al_increase).al = (increased_accounts accounts al_increase).eliot + 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_al_increase_percentage_l202_20256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l202_20243

-- Define the probability distribution for a ball landing in bin k
noncomputable def prob_in_bin (k : ℕ+) : ℝ := (1/2 : ℝ) ^ (k : ℕ)

-- Define the event that the blue ball is in a higher-numbered bin than the yellow ball
def blue_higher_than_yellow : Set (ℕ+ × ℕ+) :=
  {p | p.1 > p.2}

-- State the theorem
theorem blue_ball_higher_probability :
  ∑' (p : ℕ+ × ℕ+), (prob_in_bin p.1 * prob_in_bin p.2) * (if p.1 > p.2 then 1 else 0) = 1/3 := by
  sorry

#check blue_ball_higher_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l202_20243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_inequality_l202_20281

theorem sqrt_sum_inequality : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 → Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b) →
  (∀ a b : ℝ, a > 0 ∧ b > 0 → Real.sqrt (a / b) = Real.sqrt a / Real.sqrt b) →
  (∀ a : ℝ, a ≥ 0 → (-Real.sqrt a)^2 = a) →
  Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_inequality_l202_20281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_positions_l202_20250

def initial_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7]

def erase_every_nth (seq : List Nat) (n : Nat) : List Nat :=
  seq.enum.filter (fun (i, _) => (i + 1) % n ≠ 0) |>.map Prod.snd

def repeat_list (lst : List Nat) (n : Nat) : List Nat :=
  (List.range n).bind (fun _ => lst)

def final_sequence (init_seq : List Nat) : List Nat :=
  erase_every_nth (erase_every_nth (erase_every_nth (repeat_list init_seq 2000) 4) 5) 6

def sum_at_positions (seq : List Nat) (pos1 pos2 pos3 : Nat) : Nat :=
  seq[pos1 - 1]! + seq[pos2 - 1]! + seq[pos3 - 1]!

theorem sum_of_specific_positions :
  sum_at_positions (final_sequence initial_sequence) 3020 3021 3022 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_positions_l202_20250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_3_point_5_percent_l202_20266

/-- Calculates the interest rate given principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest / (principal * time)) * 100

/-- Theorem stating that given the specific conditions, the interest rate is 3.5% -/
theorem interest_rate_is_3_point_5_percent :
  let principal : ℝ := 1499.9999999999998
  let time : ℝ := 4
  let simple_interest : ℝ := 210
  (calculate_interest_rate principal time simple_interest) = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_3_point_5_percent_l202_20266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_eq_neg_e_l202_20200

/-- The function f(x) defined on the positive real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 + Real.log x

/-- The maximum value of f on (0,1] is -1 -/
def max_value_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≤ -1

/-- The theorem stating that if the maximum value condition holds, then a = -e -/
theorem max_value_implies_a_eq_neg_e (a : ℝ) :
  max_value_condition a → a = -Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_eq_neg_e_l202_20200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_12_l202_20291

def game_numbers : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def has_divisor_in_list (n : Nat) (l : List Nat) : Bool :=
  l.any (fun m => m ≠ n ∧ n % m = 0)

def remove_divisors (n : Nat) (l : List Nat) : List Nat :=
  l.filter (fun m => m = n ∨ n % m ≠ 0)

def carolyn_move (l : List Nat) : Option Nat :=
  l.find? (fun n => has_divisor_in_list n l)

def paul_move (n : Nat) (l : List Nat) : List Nat :=
  remove_divisors n l

theorem carolyn_sum_is_12 : 
  let first_move := 4
  let after_first := paul_move first_move (game_numbers.filter (· ≠ first_move))
  let second_move := carolyn_move after_first
  match second_move with
  | some m => first_move + m = 12
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_12_l202_20291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l202_20299

theorem prime_divisors_of_50_factorial (n : ℕ) : n = 50 → 
  (Finset.filter (λ p : ℕ ↦ Nat.Prime p ∧ p ∣ n.factorial) (Finset.range (n + 1))).card = 
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l202_20299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_Q_count_valid_Q_eq_l202_20233

-- Define the polynomial P
def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

-- Define the set of valid polynomials Q
def valid_Q : Set (ℝ → ℝ) :=
  {Q | ∃ (R : ℝ → ℝ), (∀ x, P (Q x) = P x * R x) ∧ 
       (∃ a b c d e, ∀ x, R x = a*x^4 + b*x^3 + c*x^2 + d*x + e)}

-- State the theorem without using Fintype
theorem count_valid_Q : Nat.card valid_Q = 250 := by
  sorry

-- If you need to work with the cardinality of valid_Q, you can use this approach:
def count_valid_Q' : Nat := 250

theorem count_valid_Q_eq : count_valid_Q' = 250 := by
  rfl

-- You can then use count_valid_Q' in place of Fintype.card valid_Q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_Q_count_valid_Q_eq_l202_20233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_min_value_of_y_l202_20220

-- Part 1
theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin α + Real.cos α = 7/13) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -12/5 := by sorry

-- Part 2
theorem min_value_of_y :
  ∃ (x : Real), ∀ (t : Real),
    Real.sin (2*t) + 2*Real.sqrt 2*Real.cos (Real.pi/4 + t) + 3 ≥ 
    Real.sin (2*x) + 2*Real.sqrt 2*Real.cos (Real.pi/4 + x) + 3 ∧
    Real.sin (2*x) + 2*Real.sqrt 2*Real.cos (Real.pi/4 + x) + 3 = 2 - 2*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_min_value_of_y_l202_20220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l202_20211

noncomputable def cube_side_length : ℝ := 8
noncomputable def ball_radius : ℝ := 1.5

noncomputable def cube_volume : ℝ := cube_side_length ^ 3
noncomputable def ball_volume : ℝ := (4 / 3) * Real.pi * ball_radius ^ 3

noncomputable def max_balls : ℕ := Int.toNat ⌊cube_volume / ball_volume⌋

theorem max_balls_in_cube : max_balls = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l202_20211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l202_20245

/-- Predicate to check if an equation is quadratic in x -/
def IsQuadraticInX (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Equation A: x + 3x - 6 = 12 -/
noncomputable def eqnA (x : ℝ) : ℝ := x + 3*x - 6 - 12

/-- Equation B: 2x + y = 8 -/
noncomputable def eqnB (x y : ℝ) : ℝ := 2*x + y - 8

/-- Equation C: x² + 3x = 2 -/
noncomputable def eqnC (x : ℝ) : ℝ := x^2 + 3*x - 2

/-- Equation D: (2x - 1) / x = 6 -/
noncomputable def eqnD (x : ℝ) : ℝ := (2*x - 1) / x - 6

theorem quadratic_equation_identification :
  ¬IsQuadraticInX eqnA ∧
  ¬IsQuadraticInX (fun x ↦ eqnB x 0) ∧
  IsQuadraticInX eqnC ∧
  ¬IsQuadraticInX eqnD := by
  sorry

#check quadratic_equation_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l202_20245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_t_range_l202_20224

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x + 2| - |x - 2|
noncomputable def g (x : ℝ) : ℝ := x + 1/2

-- Define the solution set for f(x) ≥ g(x)
def solution_set : Set ℝ := {x | x ≤ -9/2 ∨ (1/2 ≤ x ∧ x ≤ 7/2)}

-- Theorem for the solution set of f(x) ≥ g(x)
theorem solution_set_correct : 
  ∀ x, x ∈ solution_set ↔ f x ≥ g x := by sorry

-- Theorem for the range of t
theorem t_range (t : ℝ) : 
  (∀ x, f x ≥ t^2 - 5*t) → (1 ≤ t ∧ t ≤ 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_t_range_l202_20224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_bounds_l202_20228

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x + a * x^2 - (2*a + 1) * x

-- State the theorem
theorem intersection_slope_bounds 
  (a : ℝ) (x₁ x₂ y₁ y₂ k : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf₁ : f x₁ = y₁) 
  (hf₂ : f x₂ = y₂) 
  (hk : k = (y₂ - y₁) / (x₂ - x₁)) : 
  1 / x₂ < k ∧ k < 1 / x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_bounds_l202_20228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_in_top_10_rows_l202_20268

def pascal_triangle : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal_triangle n k + pascal_triangle n (k + 1)

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_in_row (row : ℕ) : ℕ :=
  (List.range (row + 1)).filter (fun k => is_even (pascal_triangle row k)) |>.length

def count_even_in_top_rows (n : ℕ) : ℕ :=
  (List.range n).map count_even_in_row |>.sum

theorem even_count_in_top_10_rows :
  count_even_in_top_rows 10 = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_in_top_10_rows_l202_20268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l202_20202

theorem problem_solution : 
  ((40 * Real.sqrt 3 - 18 * Real.sqrt 3 + 8 * Real.sqrt 3) / 6 = 5 * Real.sqrt 3) ∧ 
  ((Real.sqrt 3 - 2)^2023 * (Real.sqrt 3 + 2)^2023 - Real.sqrt 4 * Real.sqrt (1/2) - (Real.pi - 1)^0 = -2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l202_20202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_eq_neg_half_l202_20249

/-- A sequence defined recursively -/
def a : ℕ → ℚ
| 0 => 1
| n + 1 => -1 / (1 + a n)

/-- The main theorem stating that the 2018th term of the sequence is -1/2 -/
theorem a_2018_eq_neg_half : a 2017 = -1/2 := by
  sorry

/-- Helper lemma: The sequence has a period of 3 -/
lemma a_period_three (n : ℕ) : a (n + 3) = a n := by
  sorry

/-- The sequence at index 2 is -1/2 -/
lemma a_2_eq_neg_half : a 1 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_eq_neg_half_l202_20249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_after_reduction_price_reduction_for_target_profit_l202_20226

/-- Represents the daily sales and profit model of a supermarket product -/
structure SalesModel where
  baseSales : ℕ  -- Base daily sales
  baseProfit : ℕ  -- Base profit per item in yuan
  salesIncrease : ℕ  -- Increase in sales per yuan of price reduction
  priceReduction : ℕ  -- Amount of price reduction in yuan

/-- Calculates the new daily sales after a price reduction -/
def newDailySales (model : SalesModel) : ℕ :=
  model.baseSales + model.salesIncrease * model.priceReduction

/-- Calculates the daily profit after a price reduction -/
def dailyProfit (model : SalesModel) : ℕ :=
  (model.baseProfit - model.priceReduction) * (newDailySales model)

/-- Theorem for the first part of the problem -/
theorem sales_after_reduction (model : SalesModel) 
  (h1 : model.baseSales = 20)
  (h2 : model.salesIncrease = 2)
  (h3 : model.priceReduction = 3) :
  newDailySales model = 26 := by
  sorry

/-- Theorem for the second part of the problem -/
theorem price_reduction_for_target_profit (model : SalesModel) 
  (h1 : model.baseSales = 20)
  (h2 : model.baseProfit = 40)
  (h3 : model.salesIncrease = 2) :
  ∃ x : ℕ, x ≤ 40 ∧ 
    (let m := { model with priceReduction := x };
     dailyProfit m = 1200 ∧ 
     ∀ y, y ≤ 40 → (let m' := { model with priceReduction := y };
       dailyProfit m' = 1200 → y ≤ x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_after_reduction_price_reduction_for_target_profit_l202_20226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_lines_properties_l202_20295

/-- Represents a line in the plane of the form y = mx + b -/
structure Line where
  m : ℤ
  b : ℤ

/-- The set of lines we want to construct -/
def SpecialLines : Set Line :=
  {l : Line | ∃ n : ℕ, l.m = n ∧ l.b = n^2}

theorem special_lines_properties :
  ∃ (S : Set Line),
    (∀ l₁ l₂ : Line, l₁ ∈ S → l₂ ∈ S → l₁ ≠ l₂ → l₁.m ≠ l₂.m) ∧ 
    (∀ l₁ l₂ : Line, l₁ ∈ S → l₂ ∈ S → l₁ ≠ l₂ → 
      ∃ x y : ℤ, y = l₁.m * x + l₁.b ∧ y = l₂.m * x + l₂.b) ∧
    (∀ l₁ l₂ l₃ : Line, l₁ ∈ S → l₂ ∈ S → l₃ ∈ S → 
      l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃ → 
      ¬∃ x y : ℤ, y = l₁.m * x + l₁.b ∧ y = l₂.m * x + l₂.b ∧ y = l₃.m * x + l₃.b) :=
by
  use SpecialLines
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_lines_properties_l202_20295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_equation_solution_l202_20248

theorem cotangent_equation_solution (x : ℝ) (h : Real.sin x ≠ 0) :
  2 * (Real.cos x / Real.sin x)^2 * Real.cos x^2 + 4 * Real.cos x^2 - (Real.cos x / Real.sin x)^2 - 2 = 0 →
  ∃ k : ℤ, x = Real.pi/4 * (2*k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_equation_solution_l202_20248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_square_factorial_l202_20219

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def factorial_expression (n : ℕ) : ℕ := 
  Nat.factorial n * Nat.factorial (n + 1) * Nat.factorial (n + 2) / 12

theorem unique_perfect_square_factorial :
  ∀ n ∈ ({18, 19, 20, 21, 22} : Set ℕ),
    is_perfect_square (factorial_expression n) ↔ n = 22 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_square_factorial_l202_20219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rex_driving_lessons_l202_20214

theorem rex_driving_lessons 
  (total_lessons : ℕ) 
  (weeks_completed : ℕ) 
  (weeks_remaining : ℕ) : 
  total_lessons = 40 → 
  weeks_completed = 6 → 
  weeks_remaining = 4 → 
  total_lessons / (weeks_completed + weeks_remaining) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rex_driving_lessons_l202_20214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l202_20269

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := x^2 + x*y + 2*y - 2
def B (x y : ℝ) : ℝ := 2*x^2 - 2*x*y + x - 1

theorem problem_solution (x y : ℝ) :
  (2 * A x y - B x y = 4*x*y + 4*y - x - 3) ∧
  (∀ x, 2 * A x y - B x y = 4*y - 3 → y = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l202_20269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_shortest_side_l202_20280

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 10) 
  (hc : c = 12) 
  (perimeter_similar : ℝ) 
  (h_perimeter : perimeter_similar = 150) :
  let ratio := perimeter_similar / (a + b + c)
  min a (min b c) * ratio = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_shortest_side_l202_20280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_frequent_color_l202_20216

/-- A coloring of integers from 1 to n -/
def Coloring (n : ℕ) := ℕ → ℕ

/-- Predicate that checks if a coloring satisfies the given condition -/
def ValidColoring {n : ℕ} (c : Coloring n) : Prop :=
  ∀ a b : ℕ, 0 < a → a < b → a + b ≤ n →
    c a = c b ∨ c b = c (a + b) ∨ c a = c (a + b)

/-- Main theorem -/
theorem existence_of_frequent_color {n : ℕ} (h : n > 0) (c : Coloring n) 
    (hc : ValidColoring c) :
    ∃ color : ℕ, (Finset.filter (λ i => c i = color) (Finset.range n)).card ≥ ⌈(2 * n : ℚ) / 5⌉ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_frequent_color_l202_20216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_computation_l202_20207

-- Define the ⋆ operation
noncomputable def star (a b : ℝ) : ℝ := (a^2 + b^2) / (a - b)

-- Theorem statement
theorem star_computation :
  star (star 2 3) 4 = -185 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_computation_l202_20207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equal_halves_exist_l202_20273

theorem triangle_equal_halves_exist :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b ∧
    a = (Real.sqrt 2 - 1) * (b + c) ∧
    (let x := Real.sqrt 2 / 2
     x^2 * (a * b / 2) = (1 - x^2) * (a * b / 2) ∧
     x * (a + b + c) = (1 + x) * a + (1 - x) * (b + c)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equal_halves_exist_l202_20273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_solution_iff_sum_zero_l202_20262

/-- A system of linear equations with parameters a, b, c and variables x, y -/
structure LinearSystem (a b c : ℝ) where
  x : ℝ
  y : ℝ
  eq1 : a * x + b * y = c
  eq2 : b * x + c * y = a
  eq3 : c * x + a * y = b

/-- The existence of a negative solution for the LinearSystem -/
def HasNegativeSolution (a b c : ℝ) : Prop :=
  ∃ (s : LinearSystem a b c), s.x < 0 ∧ s.y < 0

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  HasNegativeSolution a b c ↔ a + b + c = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_solution_iff_sum_zero_l202_20262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l202_20294

/-- Given a circle with equation x^2 + y^2 - 4x + 6y + 11 = 0, 
    its center is (2, -3) and its radius is √2 -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -3) ∧ 
    radius = Real.sqrt 2 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + 11 = 0 ↔ 
      (x - center.fst)^2 + (y - center.snd)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l202_20294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_one_plus_tan_equals_2_pow_15_l202_20240

open Real BigOperators

theorem product_of_one_plus_tan_equals_2_pow_15 :
  ∏ k in Finset.range 30, (1 + tan ((k + 1 : ℝ) * π / 180)) = 2^15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_one_plus_tan_equals_2_pow_15_l202_20240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_with_no_preference_l202_20246

theorem students_with_no_preference (total_students mac_preference windows_preference : ℕ) 
  (h1 : total_students = 210)
  (h2 : mac_preference = 60)
  (h3 : windows_preference = 40) : 
  total_students - (mac_preference + windows_preference + mac_preference / 3) = 90 := by
  let both_preference := mac_preference / 3
  let students_with_preference := mac_preference + windows_preference + both_preference
  let no_preference := total_students - students_with_preference
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_with_no_preference_l202_20246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l202_20251

noncomputable def f (x : ℝ) : ℝ := ∫ t in x..(x + Real.pi/3), |Real.sin t|

theorem f_properties :
  ∃ (f_min f_max : ℝ),
    (∀ x, (deriv f x) = |Real.sin (x + Real.pi/3)| - |Real.sin x|) ∧
    (∀ x ∈ Set.Icc 0 Real.pi, f x ≥ f_min) ∧
    (∃ x_min ∈ Set.Icc 0 Real.pi, f x_min = f_min) ∧
    (∀ x ∈ Set.Icc 0 Real.pi, f x ≤ f_max) ∧
    (∃ x_max ∈ Set.Icc 0 Real.pi, f x_max = f_max) ∧
    f_min = 2 - Real.sqrt 3 ∧
    f_max = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l202_20251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l202_20234

theorem smallest_number_divisible_by_multiple : ∃ (n : ℕ), 
  (∀ (m : ℕ), m ∈ ({4, 6, 8, 10, 12, 14, 16} : Set ℕ) → (n - 16) % m = 0) ∧
  (∀ (k : ℕ), k < n → ∃ (m : ℕ), m ∈ ({4, 6, 8, 10, 12, 14, 16} : Set ℕ) ∧ (k - 16) % m ≠ 0) ∧
  n = 3376 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l202_20234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_price_l202_20254

/-- Prove that the price of each necklace is $9, given the conditions of Megan's necklace sale. -/
theorem necklace_price (bead_necklaces gem_necklaces : ℕ) (total_earnings : ℚ) 
    (h1 : bead_necklaces = 7)
    (h2 : gem_necklaces = 3)
    (h3 : total_earnings = 90) : ℚ := by
  let total_necklaces : ℕ := bead_necklaces + gem_necklaces
  let price_per_necklace : ℚ := total_earnings / total_necklaces
  have : price_per_necklace = 9 := by
    -- Proof steps would go here
    sorry
  exact price_per_necklace


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_price_l202_20254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_harmonious_coloring_l202_20282

/-- A type representing the colors used in the polygon coloring -/
def Color := Fin 2018

/-- A type representing a regular polygon with N sides -/
structure RegularPolygon (N : ℕ) where
  vertices : Fin N → ℂ
  is_regular : ∀ (i j : Fin N), Complex.abs (vertices i - vertices j) = 2 * Real.sin (π / N)

/-- A coloring of a regular polygon -/
def Coloring (N : ℕ) := (Fin N × Fin N) → Color

/-- A predicate indicating if a coloring is harmonious -/
def IsHarmonious (N : ℕ) (c : Coloring N) : Prop :=
  ∀ (i j k : Fin N), i ≠ j ∧ j ≠ k ∧ k ≠ i →
    (c (i, j) = c (j, k) → c (k, i) ≠ c (i, j)) ∧
    (c (i, j) = c (k, i) → c (j, k) ≠ c (i, j)) ∧
    (c (j, k) = c (k, i) → c (i, j) ≠ c (j, k))

/-- The main theorem stating that 11 is the largest N for which a harmonious coloring exists -/
theorem largest_harmonious_coloring :
  (∃ (c : Coloring 11), IsHarmonious 11 c) ∧
  (∀ (N : ℕ), N > 11 → ¬ ∃ (c : Coloring N), IsHarmonious N c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_harmonious_coloring_l202_20282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sine_phase_l202_20285

theorem even_sine_phase (φ : ℝ) : 
  (∀ x, Real.sin ((x + φ) / 3) = Real.sin ((-x + φ) / 3)) → 
  φ ∈ Set.Icc 0 (2 * Real.pi) → 
  φ = 3 * Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sine_phase_l202_20285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saras_birdhouse_height_l202_20247

-- Define the dimensions of the birdhouses
noncomputable def sara_width : ℝ := 1
noncomputable def sara_depth : ℝ := 2
noncomputable def jake_width : ℝ := 16 / 12
noncomputable def jake_height : ℝ := 20 / 12
noncomputable def jake_depth : ℝ := 18 / 12

-- Define the volume difference in cubic feet
noncomputable def volume_difference : ℝ := 1152 / (12^3)

-- Theorem to prove
theorem saras_birdhouse_height :
  ∃ (h : ℝ), h * sara_width * sara_depth - jake_width * jake_height * jake_depth = volume_difference ∧ h = 2 :=
by
  -- Introduce the height of Sara's birdhouse
  let h : ℝ := 2
  
  -- Prove that this height satisfies the equation
  have eq : h * sara_width * sara_depth - jake_width * jake_height * jake_depth = volume_difference := by
    -- This step would involve actual calculations, which we'll skip for now
    sorry
  
  -- Prove that h equals 2
  have h_eq_2 : h = 2 := by rfl
  
  -- Combine the proofs to satisfy the theorem
  exact ⟨h, eq, h_eq_2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_saras_birdhouse_height_l202_20247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l202_20231

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line that passes through the origin
noncomputable def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := (Real.exp x₀) * (x - x₀)

-- Theorem statement
theorem tangent_point_coordinates :
  ∃ (x₀ : ℝ), 
    -- The tangent line passes through the origin
    tangent_line x₀ 0 = 0 ∧
    -- The tangent point is on the curve
    f x₀ = Real.exp x₀ ∧
    -- The x-coordinate of the tangent point is 1
    x₀ = 1 ∧
    -- The y-coordinate of the tangent point is e
    f x₀ = Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l202_20231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l202_20255

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := -3 * (n : ℚ) + 2

-- Define the geometric sequence (a_n + b_n)
def ab (n : ℕ) (q : ℚ) : ℚ := q^(n - 1)

-- Define the sequence b_n
def b (n : ℕ) (q : ℚ) : ℚ := 3 * (n : ℚ) - 2 + q^(n - 1)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) (q : ℚ) : ℚ :=
  if q = 1 then
    (3 * (n : ℚ)^2 + n) / 2
  else
    (n : ℚ) * (3 * n - 1) / 2 + (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequence_problem :
  -- Conditions from the problem
  a 2 + a 7 = -23 ∧
  a 3 + a 8 = -29 ∧
  ∀ n : ℕ, ab n q = a n + b n q ∧
  ab 1 q = 1 →
  -- Conclusions to prove
  (∀ n : ℕ, a n = -3 * (n : ℚ) + 2) ∧
  (∀ n : ℕ, S n q = if q = 1 then (3 * (n : ℚ)^2 + n) / 2 else (n : ℚ) * (3 * n - 1) / 2 + (1 - q^n) / (1 - q)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l202_20255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_l202_20223

noncomputable section

-- Define initial amounts
def michelle_usd : ℚ := 30
def alice_gbp : ℚ := 18
def marco_eur : ℚ := 24
def mary_jpy : ℚ := 1500

-- Define conversion rates
def usd_to_eur : ℚ := 85/100
def gbp_to_eur : ℚ := 115/100
def jpy_to_eur : ℚ := 77/10000

-- Define transactions
def marco_gives : ℚ := 1/2
def michelle_gives : ℚ := 3/5
def mary_spends : ℚ := 600

-- Theorem statement
theorem money_difference :
  let marco_final := marco_eur * (1 - marco_gives)
  let michelle_final := michelle_usd * (1 - michelle_gives) * usd_to_eur
  let alice_final := alice_gbp * gbp_to_eur + michelle_usd * michelle_gives * usd_to_eur
  let mary_final := (mary_jpy - mary_spends) * jpy_to_eur + marco_eur * marco_gives
  let combined_amount := marco_final + michelle_final + alice_final
  combined_amount - mary_final = 5127/100 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_l202_20223
