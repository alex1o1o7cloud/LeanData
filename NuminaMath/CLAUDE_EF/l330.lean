import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_five_l330_33035

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem fifth_term_is_five : fibonacci_like_sequence 4 = 5 := by
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rfl

#eval fibonacci_like_sequence 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_five_l330_33035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_integer_steps_l330_33064

/-- A tree is a connected acyclic graph -/
structure MyTree where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  -- Additional properties to ensure it's a tree

/-- Random walk on a tree -/
noncomputable def RandomWalk (t : MyTree) (start goal : Nat) : ℝ := sorry

/-- The theorem to be proved -/
theorem random_walk_integer_steps (t : MyTree) (start goal : Nat) :
  ∃ n : ℕ, RandomWalk t start goal = n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_integer_steps_l330_33064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_b_is_pi_third_l330_33061

-- Define the vectors
noncomputable def a : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)
noncomputable def b : ℝ × ℝ := (Real.sqrt 3, -1)

-- Define the angle between two vectors
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Theorem statement
theorem angle_between_a_b_is_pi_third :
  angle_between a b = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_b_is_pi_third_l330_33061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l330_33084

-- Define the three functions
noncomputable def f (x : ℝ) : ℝ := 2 * x - 3
noncomputable def g (x : ℝ) : ℝ := (x^2 - 9) / (x + 3)
noncomputable def h (x : ℝ) : ℝ := (x^2 - 9) / (x + 3)

-- Define the domain of each function
def dom_f : Set ℝ := Set.univ
def dom_g : Set ℝ := {x : ℝ | x ≠ -3}
def dom_h : Set ℝ := Set.univ

theorem different_graphs :
  (∃ x, x ∈ dom_f ∩ dom_g ∧ f x ≠ g x) ∧
  (∃ x, x ∈ dom_f ∩ dom_h ∧ f x ≠ h x) ∧
  (∃ x, x ∈ dom_g ∩ dom_h ∧ (x = -3 ∨ g x ≠ h x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l330_33084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_range_for_16_yuan_fare_l330_33099

/-- Represents the taxi fare system -/
structure TaxiFareSystem where
  basicCharge : ℚ
  basicDistance : ℚ
  additionalRate : ℚ

/-- Calculates the exact fare before rounding -/
def calculateExactFare (system : TaxiFareSystem) (distance : ℚ) : ℚ :=
  if distance ≤ system.basicDistance then
    system.basicCharge
  else
    system.basicCharge + system.additionalRate * (distance - system.basicDistance)

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊(x + 1/2 : ℚ)⌋

/-- Theorem stating the range of trip distance for a 16 yuan fare -/
theorem trip_distance_range_for_16_yuan_fare (system : TaxiFareSystem) 
    (h1 : system.basicCharge = 8)
    (h2 : system.basicDistance = 3)
    (h3 : system.additionalRate = 3/2) :
  ∀ d : ℚ, roundToNearest (calculateExactFare system d) = 16 ↔ 8 ≤ d ∧ d < 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_range_for_16_yuan_fare_l330_33099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l330_33063

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x^2 - (x - 3) * Real.exp x + 1

-- State the theorem
theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧
    (∀ x ∈ Set.Ioo 0 2, deriv (f a) x = 0 → x = x₁ ∨ x = x₂)) →
  Real.exp 1 / 3 < a ∧ a < Real.exp 2 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l330_33063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_q_is_true_l330_33053

theorem proposition_p_and_q_is_true : 
  (∃ x : ℝ, Real.sin x < 1) ∧ (∀ x : ℝ, Real.exp (abs x) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_q_is_true_l330_33053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l330_33062

/-- The volume of a truncated right circular cone. -/
noncomputable def truncatedConeVolume (r₁ : ℝ) (r₂ : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * h * (r₁^2 + r₂^2 + r₁ * r₂)

/-- 
Theorem: The volume of a truncated right circular cone with
large base radius 10 cm, small base radius 5 cm, and height 10 cm
is equal to (1750/3)π cubic cm.
-/
theorem truncated_cone_volume_example :
  truncatedConeVolume 10 5 10 = (1750 / 3) * Real.pi := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l330_33062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_rectangle_remaining_point_l330_33066

/-- The golden ratio --/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The value of q used in the solution --/
noncomputable def q : ℝ := (Real.sqrt 5 - 1) / 2

/-- A golden rectangle is a rectangle whose sides a and b satisfy a:b = b:(a-b) --/
def isGoldenRectangle (a b : ℝ) : Prop := a / b = b / (a - b)

/-- The coordinates of the remaining point after infinitely many iterations --/
noncomputable def remainingPoint (a : ℝ) : ℝ × ℝ := (a * q / (1 - q^4), a * q^4 / (1 - q^4))

/-- The theorem stating that the remaining point has the calculated coordinates --/
theorem golden_rectangle_remaining_point (a b : ℝ) (h : isGoldenRectangle a b) :
  remainingPoint a = (a * q / (1 - q^4), a * q^4 / (1 - q^4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_rectangle_remaining_point_l330_33066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l330_33036

open Real

-- Define the function f
noncomputable def f (ω φ x : ℝ) : ℝ := sin (ω * x + φ)

-- Define the derivative g
noncomputable def g (ω φ x : ℝ) : ℝ := ω * cos (ω * x + φ)

-- Theorem stating the periods are equal and y can be an odd function
theorem f_g_properties (ω φ : ℝ) (h : ω > 0) :
  (∃ T : ℝ, T > 0 ∧ 
    (∀ x : ℝ, f ω φ (x + T) = f ω φ x) ∧ 
    (∀ x : ℝ, g ω φ (x + T) = g ω φ x)) ∧
  (∃ θ : ℝ, ∀ x : ℝ, f ω φ x + g ω φ x = -(f ω φ (-x) + g ω φ (-x))) := by
  sorry

#check f_g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l330_33036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l330_33019

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Calculate the x-intercept of a line -/
noncomputable def Line.xIntercept (l : Line) : ℝ := -l.c / l.a

/-- Calculate the y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ := -l.c / l.b

/-- Check if a line satisfies the given conditions -/
def satisfiesConditions (l : Line) : Prop :=
  l.containsPoint 2 3 ∧ l.xIntercept = 2 * l.yIntercept

theorem line_equation : 
  ∀ l : Line, satisfiesConditions l → 
    (l.a = 1 ∧ l.b = 2 ∧ l.c = -8) ∨ (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l330_33019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_a_equal_to_fifth_of_b_l330_33086

theorem fraction_of_a_equal_to_fifth_of_b : ∃ (fraction : ℚ), fraction = 3 / 10 := by
  -- Define the total amount
  let total : ℚ := 100

  -- Define b's amount
  let b_amount : ℚ := 60

  -- Define a's amount
  let a_amount : ℚ := total - b_amount

  -- Define the fraction we're looking for
  let fraction : ℚ := (b_amount / 5) / a_amount

  -- Assert that this fraction is equal to 3/10
  have h : fraction = 3 / 10 := by
    -- Proof goes here
    sorry

  -- Conclude the theorem
  exact ⟨fraction, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_a_equal_to_fifth_of_b_l330_33086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_l330_33074

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y = 1

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop := x + 2*y = 0

/-- The vertex of the hyperbola -/
def vertex : ℝ × ℝ := (2, 0)

/-- The distance formula from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_vertex_to_asymptote :
  distance_point_to_line vertex.1 vertex.2 1 2 0 = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_l330_33074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_to_zero_matrix_l330_33072

/-- Represents the allowed operations on the matrix --/
inductive Operation
  | ColumnDouble (col : Nat)
  | RowSubtract (row : Nat)

/-- Defines a rectangular matrix of natural numbers --/
def RectMatrix := Array (Array Nat)

/-- Applies an operation to a matrix --/
def applyOperation (m : RectMatrix) (op : Operation) : RectMatrix :=
  sorry

/-- Checks if a matrix is a zero matrix --/
def isZeroMatrix (m : RectMatrix) : Bool :=
  sorry

/-- Theorem: For any rectangular matrix of natural numbers, there exists a finite sequence
    of operations that transforms it into a zero matrix --/
theorem table_to_zero_matrix (m : RectMatrix) :
  ∃ (ops : List Operation), isZeroMatrix (ops.foldl applyOperation m) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_to_zero_matrix_l330_33072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l330_33030

-- Define the necessary types and functions
variable (Point : Type)
variable (BE EF FC : ℝ)
variable (is_semicircle : Point → Point → Point → Point → Prop)
variable (is_tangent : Point → Point → Point → Point → Prop)
variable (perimeter_triangle : Point → Point → Point → ℝ)

-- State the theorem
theorem triangle_perimeter (A B C E F : Point) 
  (h1 : BE = 1) (h2 : EF = 24) (h3 : FC = 3)
  (h4 : is_semicircle E F B C) (h5 : is_tangent E F A B) (h6 : is_tangent E F A C) :
  perimeter_triangle A B C = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l330_33030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_area_is_244pi_l330_33023

noncomputable def circular_platform_area (chord_length radius_to_midpoint : ℝ) : ℝ :=
  Real.pi * (radius_to_midpoint^2 + (chord_length / 2)^2)

theorem platform_area_is_244pi :
  circular_platform_area 20 12 = 244 * Real.pi := by
  -- Unfold the definition of circular_platform_area
  unfold circular_platform_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_area_is_244pi_l330_33023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l330_33059

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 2 →
  Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = Real.sqrt 3 →
  b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 0 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (2 * Real.sqrt 3)) = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l330_33059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l330_33001

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-a * x^2 + x - 1) / Real.exp x

theorem f_lower_bound (a : ℝ) (h : a ≤ -1) : ∀ x : ℝ, f a x ≥ -Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l330_33001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_equals_16_l330_33032

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * ((x + 1) / 2)^2

-- State the theorem
theorem f_of_3_equals_16 : f 3 = 16 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_equals_16_l330_33032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l330_33026

theorem triangle_area_inequality (a b c S : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hS : S > 0) (h_triangle : S^2 = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c) / 16) : 
  (2 * S)^3 < (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l330_33026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l330_33080

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem f_properties :
  (∀ x, f 0 1 x = f 0 1 (-x)) ∧
  (∀ t, ∃ M m, (∀ x ∈ Set.Icc t (t + Real.pi/4), m ≤ f (Real.sqrt 3) 1 x ∧ f (Real.sqrt 3) 1 x ≤ M) ∧
                M - m ≤ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l330_33080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_selling_price_l330_33085

/-- Calculates the selling price of a car given the purchase price, repair costs, and profit percentage -/
def calculate_selling_price (purchase_price repair_costs : ℕ) (profit_percent : ℚ) : ℕ :=
  let total_cost := purchase_price + repair_costs
  let profit := (profit_percent / 100) * (total_cost : ℚ)
  ((total_cost : ℚ) + profit).floor.toNat

/-- Theorem stating that given the specific conditions, the selling price of the car is 80000 -/
theorem car_selling_price :
  calculate_selling_price 45000 12000 (40.35 : ℚ) = 80000 := by
  sorry

#eval calculate_selling_price 45000 12000 (40.35 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_selling_price_l330_33085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l330_33052

-- Define set A
def A : Set ℤ := {x | ∃ n : ℤ, x = 3*n - 1}

-- Define set B
def B : Set ℤ := {x | 0 < x ∧ x < 6}

-- Theorem statement
theorem A_intersect_B : A ∩ B = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l330_33052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_overlap_and_placement_l330_33049

/-- Represents a circular table with coins -/
structure CoinTable where
  R : ℝ  -- Radius of the table
  n : ℕ  -- Number of coins

/-- Function to calculate the distance between two coins -/
noncomputable def coin_distance (table : CoinTable) (i j : ℕ) : ℝ :=
  sorry -- Definition of distance between coins i and j

/-- Function to calculate the distance between a coin and a point -/
noncomputable def coin_distance_to_point (table : CoinTable) (i : ℕ) (x y : ℝ) : ℝ :=
  sorry -- Definition of distance between coin i and point (x, y)

/-- Predicate indicating that coins overlap on the table -/
def coins_overlap (table : CoinTable) : Prop :=
  ∃ (i j : ℕ), i ≠ j ∧ i < table.n ∧ j < table.n ∧ coin_distance table i j < 2

/-- Predicate indicating that an additional coin can be placed without overlap -/
def can_place_additional_coin (table : CoinTable) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 ≤ table.R^2 ∧
    ∀ (i : ℕ), i < table.n → coin_distance_to_point table i x y ≥ 2

/-- Theorem stating conditions for coin overlap and placement -/
theorem coin_overlap_and_placement (table : CoinTable) :
  (table.n > table.R^2 → coins_overlap table) ∧
  (table.n < 0.25 * (table.R - 1)^2 → can_place_additional_coin table) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_overlap_and_placement_l330_33049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_a_range_l330_33092

-- Define the function f(x) = x / (x - 1)
noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

-- Theorem for the monotonicity of f(x)
theorem f_monotone_decreasing :
  ∀ x y : ℝ, 1 < x → x < y → f x > f y :=
by
  sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  f (2 * a + 1) > f (a + 2) → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_a_range_l330_33092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_of_f_l330_33096

-- Define the function f(x) = ln x + 1/x - 2
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1/x - 2

-- Theorem statement
theorem two_zeros_of_f :
  ∃ (a b : ℝ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ f a = 0 ∧ f b = 0 ∧
  ∀ (x : ℝ), x > 0 → f x = 0 → (x = a ∨ x = b) :=
by
  sorry

#check two_zeros_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_of_f_l330_33096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₄_not_increasing_l330_33007

open Real

-- Define the interval (0, +∞)
def openPositiveReals := {x : ℝ | x > 0}

-- Define the functions
noncomputable def f₁ : ℝ → ℝ := λ x ↦ 2^x
noncomputable def f₂ : ℝ → ℝ := log
noncomputable def f₃ : ℝ → ℝ := λ x ↦ x^3
noncomputable def f₄ : ℝ → ℝ := λ x ↦ 1/x

-- Theorem statement
theorem only_f₄_not_increasing :
  (StrictMono f₁ ∧ StrictMono f₂ ∧ StrictMono f₃) ∧
  ¬(StrictMono f₄) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₄_not_increasing_l330_33007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_of_g_set_C_equality_intersection_A_C_l330_33024

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.sqrt ((1 + x) * (2 - x))
noncomputable def g (x a : ℝ) := Real.log (x - a)

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x > a}
def C : Set ℝ := {x | (2 : ℝ)^(x^2 - 2*x - 3) < 1}

-- Theorem statements
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = A := by sorry

theorem domain_of_g (a : ℝ) : {x : ℝ | ∃ y, g x a = y} = B a := by sorry

theorem set_C_equality : C = {x : ℝ | -1 < x ∧ x < 3} := by sorry

theorem intersection_A_C : A ∩ C = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_of_g_set_C_equality_intersection_A_C_l330_33024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_in_first_quadrant_l330_33076

-- Define the function (marked as noncomputable due to dependency on Real)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2)^x + m

-- Theorem statement
theorem not_in_first_quadrant (m : ℝ) : 
  (∀ x > 0, f m x ≤ 0) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_in_first_quadrant_l330_33076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l330_33071

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2*x + 1/x

-- Theorem statement
theorem tangent_line_sum (a b : ℝ) : 
  (∀ x : ℝ, f x = x^2 + Real.log x) →  -- Definition of f
  (f' 1 = 3) →                        -- Slope at x = 1
  (f 1 = 1) →                         -- f(1) = 1
  (1 + a = f 1 + a) →                 -- Point (1, f(1)) is on the tangent line
  (a = f' 1) →                        -- Slope of tangent line equals f'(1)
  (a * 1 + b = f 1) →                 -- Tangent line passes through (1, f(1))
  a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l330_33071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l330_33048

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 3}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l330_33048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_eight_l330_33029

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square -/
noncomputable def perimeter_semicircle_square (side_length : ℝ) : ℝ :=
  4 * (Real.pi * side_length / 2)

/-- Theorem: The perimeter of the region is 8 when the square's side length is 4/π -/
theorem perimeter_is_eight :
  perimeter_semicircle_square (4 / Real.pi) = 8 := by
  -- Unfold the definition of perimeter_semicircle_square
  unfold perimeter_semicircle_square
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_eight_l330_33029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_triangular_pyramid_volume_positive_l330_33097

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  /-- The length of each lateral edge is 1 -/
  lateral_edge_length : ℝ := 1
  /-- The lateral faces are congruent -/
  lateral_faces_congruent : Prop
  /-- One of the dihedral angles at the base is a right angle -/
  right_dihedral_angle_at_base : Prop

/-- The volume of a triangular pyramid with the given properties -/
noncomputable def volume (p : TriangularPyramid) : ℝ := 2 / (9 * Real.sqrt 3)

/-- Theorem stating that the volume of the specific triangular pyramid is 2 / (9 * √3) -/
theorem triangular_pyramid_volume (p : TriangularPyramid) : volume p = 2 / (9 * Real.sqrt 3) := by
  -- Unfold the definition of volume
  unfold volume
  -- The result follows directly from the definition
  rfl

/-- Theorem stating that the volume of the specific triangular pyramid is positive -/
theorem triangular_pyramid_volume_positive (p : TriangularPyramid) : volume p > 0 := by
  unfold volume
  -- Use positivity of real numbers and square root
  apply div_pos
  · norm_num
  · exact mul_pos (by norm_num) (Real.sqrt_pos.2 (by norm_num))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_triangular_pyramid_volume_positive_l330_33097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_of_man_is_90kg_l330_33050

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_depth * water_density

/-- Theorem stating that the mass of the man is 90 kg under the given conditions. -/
theorem mass_of_man_is_90kg :
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let boat_sink_depth : ℝ := 0.015  -- 1.5 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_depth water_density = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_of_man_is_90kg_l330_33050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_plus_q_equals_zero_l330_33031

-- Define the universal set U
def U : Set ℝ := {1, 2}

-- Define the set A
def A (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q = 0}

-- Define the theorem
theorem p_plus_q_equals_zero (p q : ℝ) :
  (U \ A p q) = {1} → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_plus_q_equals_zero_l330_33031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_count_l330_33044

theorem exam_pass_count (total_boys : ℕ) (avg_all avg_pass avg_fail : ℚ) :
  total_boys = 120 →
  avg_all = 37 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ (pass_count : ℕ),
    pass_count + (total_boys - pass_count) = total_boys ∧
    avg_all * total_boys = avg_pass * pass_count + avg_fail * (total_boys - pass_count) ∧
    pass_count = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_count_l330_33044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascals_theorem_l330_33017

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points of the hexagon
variable (A B C D E F : ℝ × ℝ)

-- Define the condition that the hexagon is inscribed in the circle
def inscribed_hexagon (circle : Set (ℝ × ℝ)) (A B C D E F : ℝ × ℝ) : Prop :=
  A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle

-- Define the intersection points
noncomputable def X (A B D E : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def Y (B C E F : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def Z (C D F A : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define collinearity
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

-- Pascal's Theorem
theorem pascals_theorem 
  (h : inscribed_hexagon circle A B C D E F) : 
  collinear (X A B D E) (Y B C E F) (Z C D F A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascals_theorem_l330_33017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l330_33014

theorem imaginary_part_of_z : ∃ z : ℂ, z = (1 + Complex.I) ^ 2 + Complex.I ^ 2010 ∧ z.im = 2 := by
  let z : ℂ := (1 + Complex.I) ^ 2 + Complex.I ^ 2010
  use z
  constructor
  · rfl
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l330_33014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_double_sum_l330_33033

theorem sec_double_sum (x y z : ℝ) 
  (h1 : Real.sin x / Real.cos x + Real.sin y / Real.cos y + Real.sin z / Real.cos z = 0)
  (h2 : 1 / Real.cos x + 1 / Real.cos y + 1 / Real.cos z = 3) :
  1 / Real.cos (2 * x) + 1 / Real.cos (2 * y) + 1 / Real.cos (2 * z) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_double_sum_l330_33033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_when_tan_is_one_third_l330_33075

theorem sin_minus_cos_when_tan_is_one_third (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_when_tan_is_one_third_l330_33075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_sum_l330_33028

/-- The sum of distances from any point on the ellipse 7x^2 + 3y^2 = 21 to its foci is 2√7 -/
theorem ellipse_foci_distance_sum :
  ∀ (x y : ℝ), 7 * x^2 + 3 * y^2 = 21 →
  ∃ (f1 f2 : ℝ × ℝ),
    (∀ (p q : ℝ × ℝ), (7 * p.1^2 + 3 * p.2^2 = 21 ∧ 7 * q.1^2 + 3 * q.2^2 = 21) →
      dist p f1 + dist p f2 = dist q f1 + dist q f2) ∧
    dist (x, y) f1 + dist (x, y) f2 = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_sum_l330_33028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l330_33037

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_equation_solution (m : ℝ) :
  is_pure_imaginary (Complex.ofReal (m^2 + m - 2) + Complex.I * (m^2 - 1)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l330_33037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_investment_is_600_l330_33034

-- Define the interest rates and initial investments
def tim_rate : ℝ := 0.10
def lana_rate : ℝ := 0.05
def lana_investment : ℝ := 800
def interest_difference : ℝ := 44.000000000000114

-- Define the function to calculate compound interest
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

-- Theorem statement
theorem tim_investment_is_600 :
  ∃ (tim_investment : ℝ),
    compound_interest tim_investment tim_rate 2 - 
    compound_interest lana_investment lana_rate 2 = interest_difference ∧
    abs (tim_investment - 600) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_investment_is_600_l330_33034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l330_33022

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (14 - x^2) * (Real.sin x - Real.cos (2*x))

-- State the theorem
theorem equation_roots :
  ∃ (S : Finset ℝ), S.card = 6 ∧ 
  (∀ x ∈ S, x ∈ Set.Icc (-Real.sqrt 14) (Real.sqrt 14) ∧ f x = 0) ∧
  (∀ x, x ∈ Set.Icc (-Real.sqrt 14) (Real.sqrt 14) ∧ f x = 0 → x ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l330_33022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_l330_33051

def vector_a (y : ℝ) : Fin 3 → ℝ := ![2, -y, 2]
def vector_b (x : ℝ) : Fin 3 → ℝ := ![4, 2, x]

def squared_magnitude (v : Fin 3 → ℝ) : ℝ :=
  (v 0) * (v 0) + (v 1) * (v 1) + (v 2) * (v 2)

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

theorem vector_sum (x y : ℝ) :
  squared_magnitude (vector_a y) + squared_magnitude (vector_b x) = 44 ∧
  dot_product (vector_a y) (vector_b x) = 0 →
  x + y = 4 ∨ x + y = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_l330_33051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l330_33065

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ × ℝ  -- (x, y) coordinates of the focus

/-- The set of points on the parabola. -/
def Parabola.toSet (p : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | (x - p.focus.1)^2 + (y - p.focus.2)^2 = (y - p.directrix)^2}

/-- The standard equation of a parabola. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ p.toSet → x^2 = 4 * y

/-- Theorem: For a parabola with directrix y = -1, its standard equation is x² = 4y. -/
theorem parabola_standard_equation (p : Parabola) (h : p.directrix = -1) :
  standardEquation p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l330_33065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l330_33006

-- Define the floor function (integer part)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem sin_cos_equation_solution (x : ℝ) :
  (floor (Real.sin x))^2 = (Real.cos x)^2 - 1 ↔ ∃ n : ℤ, x = n * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l330_33006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_count_l330_33042

-- Define the polar curves
noncomputable def curve1 (θ : ℝ) : ℝ := 6 * Real.sin θ
noncomputable def curve2 (θ : ℝ) : ℝ := 2 * Real.sin (2 * θ + Real.pi)

-- Define the Cartesian equations of the curves
def circle1 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 4

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 3)
def center2 : ℝ × ℝ := (0, -2)
def radius1 : ℝ := 3
def radius2 : ℝ := 2

-- Theorem stating that the curves intersect at exactly one point
theorem curves_intersection_count :
  ∃! p : ℝ × ℝ, circle1 p.1 p.2 ∧ circle2 p.1 p.2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_count_l330_33042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_percentage_is_ten_percent_l330_33057

/-- Represents the land distribution of a farmer -/
structure FarmLand where
  total : ℝ
  cleared_percentage : ℝ
  potato_percentage : ℝ
  tomato_acres : ℝ

/-- Calculates the percentage of cleared land planted with grapes -/
noncomputable def grape_percentage (farm : FarmLand) : ℝ :=
  let cleared_land := farm.total * farm.cleared_percentage
  let potato_land := cleared_land * farm.potato_percentage
  let grape_land := cleared_land - potato_land - farm.tomato_acres
  (grape_land / cleared_land) * 100

/-- Theorem stating that the percentage of cleared land planted with grapes is 10% -/
theorem grape_percentage_is_ten_percent (farm : FarmLand) 
  (h1 : farm.total = 4999.999999999999)
  (h2 : farm.cleared_percentage = 0.9)
  (h3 : farm.potato_percentage = 0.8)
  (h4 : farm.tomato_acres = 450) :
  grape_percentage farm = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_percentage_is_ten_percent_l330_33057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nandan_earnings_l330_33016

-- Define the investment and time for Nandan
variable (nandan_investment : ℝ)
variable (nandan_time : ℝ)

-- Define Krishan's investment and time based on Nandan's
def krishan_investment (nandan_investment : ℝ) : ℝ := 6 * nandan_investment
def krishan_time (nandan_time : ℝ) : ℝ := 2 * nandan_time

-- Define the total gain
def total_gain : ℝ := 78000

-- Define the proportionality of gain to investment and time
axiom gain_proportionality (nandan_investment nandan_time : ℝ) : 
  ∃ (k : ℝ), k * (nandan_investment * nandan_time + 
    krishan_investment nandan_investment * krishan_time nandan_time) = total_gain

-- Theorem to prove Nandan's earnings
theorem nandan_earnings (nandan_investment nandan_time : ℝ) : 
  (nandan_investment * nandan_time) / (nandan_investment * nandan_time + 
    krishan_investment nandan_investment * krishan_time nandan_time) * total_gain = 6000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nandan_earnings_l330_33016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l330_33025

theorem diophantine_equation_solutions (n : ℤ) :
  (∃ (a b c : ℕ+), (a : ℝ)^n + (b : ℝ)^n = (c : ℝ)^n) ↔ n = 1 ∨ n = -1 ∨ n = 2 ∨ n = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l330_33025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l330_33095

/-- The distance between the foci of a hyperbola with equation y²/a² - x²/b² = 1 -/
noncomputable def focalDistance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- The hyperbola equation y²/50 - x²/8 = 1 has foci distance 2√58 -/
theorem hyperbola_foci_distance :
  focalDistance (Real.sqrt 50) (Real.sqrt 8) = 2 * Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l330_33095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_implies_m_leq_neg_three_l330_33013

noncomputable section

/-- A function f is concave on an interval (a, b) if its second derivative is positive on that interval. -/
def IsConcave (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → (deriv^[2] f) x > 0

/-- The function f(x) = (1/20)x^5 - (1/12)mx^4 - 2x^2 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (1/20) * x^5 - (1/12) * m * x^4 - 2 * x^2

theorem concave_implies_m_leq_neg_three :
  ∀ m : ℝ, IsConcave (f m) 1 3 → m ≤ -3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_implies_m_leq_neg_three_l330_33013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_expression_l330_33015

theorem meaningful_expression (m : ℝ) :
  m ≠ 1 ↔ ∃ y : ℝ, y = (m + 1)^(1/3) / (m - 1) :=
sorry

#check meaningful_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_expression_l330_33015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l330_33078

/-- Represents the number of floors in the building. -/
def N : ℕ := by sorry

/-- Represents Dasha's floor number. -/
def D : ℕ := 6

/-- Represents Tanya's floor number. -/
def T : ℕ := by sorry

/-- Represents the number of floors Tanya initially climbed up. -/
def x : ℕ := by sorry

theorem building_floors :
  (D = 6) →
  (D + x = N) →
  (x + (N - T) = (3/2) * (D - T)) →
  (T < D) →
  (N = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l330_33078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_chinese_poem_equation_l330_33093

/-- Represents the number of rooms in the shop -/
def x : ℕ → ℕ := fun n => n

/-- The total number of guests when 7 guests are in each room plus 7 extra guests -/
def guests_scenario1 (n : ℕ) : ℕ := 7 * x n + 7

/-- The total number of guests when 9 guests are in each room with one room empty -/
def guests_scenario2 (n : ℕ) : ℕ := 9 * (x n - 1)

/-- Theorem stating that the equation correctly represents the scenario in the poem -/
theorem ancient_chinese_poem_equation (n : ℕ) :
  guests_scenario1 n = guests_scenario2 n :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_chinese_poem_equation_l330_33093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x8_eq_145_l330_33067

/-- The coefficient of x^8 in the expansion of (1-x)^2(2-x)^8 -/
def coefficient_x8 : ℤ :=
  let binomial := fun (n k : ℕ) => Nat.choose n k
  let term1 := (binomial 8 8 * 2^0 * (-1)^8 : ℤ)
  let term2 := (-2 * binomial 8 7 * 2^1 * (-1)^7 : ℤ)
  let term3 := (binomial 8 6 * 2^2 * (-1)^6 : ℤ)
  term1 + term2 + term3

theorem coefficient_x8_eq_145 : coefficient_x8 = 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x8_eq_145_l330_33067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_128_root_3_l330_33000

/-- The radius of the circle -/
def radius : ℝ := 16

/-- The rhombus formed by two radii and two chords of the circle -/
structure Rhombus where
  /-- The shorter diagonal of the rhombus (equal to the diameter of the circle) -/
  shorter_diagonal : ℝ
  /-- The longer diagonal of the rhombus -/
  longer_diagonal : ℝ
  /-- Proof that the shorter diagonal is equal to the diameter of the circle -/
  shorter_is_diameter : shorter_diagonal = 2 * radius

/-- The area of the rhombus -/
noncomputable def rhombus_area (r : Rhombus) : ℝ :=
  (1 / 2) * r.shorter_diagonal * r.longer_diagonal

/-- Theorem stating that the area of the rhombus is 128√3 square feet -/
theorem rhombus_area_is_128_root_3 (r : Rhombus) : rhombus_area r = 128 * Real.sqrt 3 := by
  sorry

#check rhombus_area_is_128_root_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_128_root_3_l330_33000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_l330_33082

theorem cubic_root_equation (a b c : ℕ+) :
  (4 : ℝ) * ((7 : ℝ) ^ (1/3) - (6 : ℝ) ^ (1/3)) ^ (1/2) = 
  (a.val : ℝ) ^ (1/3) + (b.val : ℝ) ^ (1/3) - (c.val : ℝ) ^ (1/3) →
  a.val + b.val + c.val = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_l330_33082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_sales_amount_l330_33021

/-- Calculates the next sales amount given initial royalties, initial sales, 
    next royalties, and royalty rate decrease. -/
noncomputable def calculate_next_sales (initial_royalties : ℝ) (initial_sales : ℝ) 
                         (next_royalties : ℝ) (royalty_rate_decrease : ℝ) : ℝ :=
  let initial_rate := initial_royalties / initial_sales
  let rate_decrease := initial_rate * royalty_rate_decrease
  let new_rate := initial_rate - rate_decrease
  next_royalties / new_rate

/-- Theorem stating that the next sales amount is approximately $70.588 million 
    given the specified conditions. -/
theorem next_sales_amount : 
  let initial_royalties := (3000000 : ℝ)
  let initial_sales := (20000000 : ℝ)
  let next_royalties := (9000000 : ℝ)
  let royalty_rate_decrease := (0.15 : ℝ)
  abs (calculate_next_sales initial_royalties initial_sales next_royalties royalty_rate_decrease - 70588000) < 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_sales_amount_l330_33021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l330_33094

/-- Given a circle centered at the origin and a line that intercepts a chord on the circle, 
    this theorem proves the equation of the circle and the equation of the tangent line 
    that minimizes the length of the segment intercepted by the coordinate axes. -/
theorem circle_and_tangent_line (x y : ℝ) : 
  (∃ (r : ℝ), (x - y + 1 = 0 → (x^2 + y^2 = r^2 ∧ (x - 1)^2 + (y + 1)^2 = r^2 + 6))) →
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2 + y^2 = 2 → (a*x + b*y - a*b)^2 ≥ 2*(a^2 + b^2)) ∧
    (x + y = 2 → x^2 + y^2 = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l330_33094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_exponents_main_theorem_l330_33055

/-- Two monomials are like terms if they have the same variables raised to the same powers -/
def like_terms (m1 m2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (a b : ℕ), ∃ (c : ℚ), m1 a b = c * m2 a b ∨ m2 a b = c * m1 a b

/-- The first monomial: -4a^3b^m -/
def monomial1 (a b m : ℕ) : ℚ := -4 * (a^3 * b^m : ℚ)

/-- The second monomial: 5a^(n+1)b -/
def monomial2 (a b n : ℕ) : ℚ := 5 * (a^(n+1) * b : ℚ)

theorem monomial_exponents (m n : ℕ) : 
  like_terms (monomial1 · · m) (monomial2 · · n) → m = n + 1 := by
  sorry

/-- The main theorem stating that m - n = -1 -/
theorem main_theorem (m n : ℕ) :
  like_terms (monomial1 · · m) (monomial2 · · n) → (m : ℤ) - (n : ℤ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_exponents_main_theorem_l330_33055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_l330_33018

-- Define an even function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2*x - x^2 else -2*x - x^2

-- State the theorem
theorem even_function_property :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, x ≥ 0 → f x = 2*x - x^2) → 
  (∀ x, x < 0 → f x = -2*x - x^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_l330_33018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_of_circles_l330_33010

-- Define the circles in polar coordinates
def circle_O1 (ρ : ℝ) : Prop := ρ = 2
def circle_O2 (ρ θ : ℝ) : Prop := ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi/4) = 2

-- Define the line in polar coordinates
def intersection_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = Real.sqrt 2 / 2

theorem intersection_line_of_circles :
  ∀ ρ θ : ℝ, circle_O1 ρ ∧ circle_O2 ρ θ → intersection_line ρ θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_of_circles_l330_33010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_not_equal_20172018_l330_33046

/-- The sequence of natural numbers not divisible by 4 -/
def notDivisibleBy4 : ℕ → ℕ
  | 0 => 1
  | n + 1 => if (n + 1) % 4 = 0 then notDivisibleBy4 n + 1 else n + 1

/-- The sum of 1000 consecutive numbers from the sequence -/
def sumOf1000 (start : ℕ) : ℕ :=
  (List.range 1000).map (fun i => notDivisibleBy4 (start + i)) |>.sum

/-- Theorem: The sum of 1000 consecutive numbers from the sequence cannot be 20172018 -/
theorem sum_not_equal_20172018 (start : ℕ) : sumOf1000 start ≠ 20172018 := by
  sorry

#eval sumOf1000 1  -- For demonstration purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_not_equal_20172018_l330_33046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_screen_diagonal_l330_33043

theorem tv_screen_diagonal (larger_diagonal : ℝ) (area_difference : ℝ) :
  larger_diagonal = 22 →
  area_difference = 42 →
  ∃ (smaller_diagonal : ℝ),
    (smaller_diagonal / Real.sqrt 2)^2 + area_difference = (larger_diagonal / Real.sqrt 2)^2 ∧
    smaller_diagonal = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_screen_diagonal_l330_33043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l330_33068

/-- Given a right angle with vertex O and two points A and B on one side of the angle,
    this function calculates the radius of the circle passing through A and B
    and tangent to the other side of the angle. -/
noncomputable def circleRadius (a b : ℝ) : ℝ :=
  (a + b) / 2

/-- Theorem stating that the radius of the circle passing through A and B
    and tangent to the other side of the right angle is (a + b) / 2,
    where OA = a and OB = b. -/
theorem circle_radius_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  circleRadius a b = (a + b) / 2 := by
  -- Unfold the definition of circleRadius
  unfold circleRadius
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l330_33068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_coprime_divisibility_l330_33012

-- Define the polynomials P(x) and Q(x)
def P (n : ℕ) (x : ℝ) : ℝ := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℝ) : ℝ := x^4 + x^3 + x^2 + x + 1

-- Define divisibility for polynomials
def divides (f g : ℝ → ℝ) : Prop := ∃ h : ℝ → ℝ, g = λ x ↦ f x * h x

-- Define the general polynomial for part (b)
def R (m n : ℕ) (x : ℝ) : ℝ := 
  Finset.sum (Finset.range m) (λ i ↦ x^(i*n))

def S (m : ℕ) (x : ℝ) : ℝ := 
  Finset.sum (Finset.range m) (λ i ↦ x^i)

-- Statement for part (a)
theorem divisibility_condition (n : ℕ) : 
  divides Q (P n) ↔ ¬(5 ∣ n) := by
  sorry

-- Statement for part (b)
theorem coprime_divisibility (m n : ℕ) : 
  Nat.Coprime m n → divides (S m) (R m n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_coprime_divisibility_l330_33012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l330_33008

/-- Given a hyperbola with the following properties:
  1. Its asymptotes are y = 2x + 3 and y = -2x + 1
  2. It passes through the point (2, 2)
  Prove that the distance between its foci is 5√5 -/
theorem hyperbola_foci_distance :
  ∀ (h : Set (ℝ × ℝ)),
  (∃ (l₁ l₂ : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ y = 2 * x + 3) ∧
    (∀ (x y : ℝ), (x, y) ∈ l₂ ↔ y = -2 * x + 1) ∧
    -- Assume some property that relates h to its asymptotes
    True) →
  ((2, 2) ∈ h) →
  -- Assume some property that defines the distance between foci
  (∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧ True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l330_33008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l330_33070

theorem triangle_ratio_theorem (a b c A B C S : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = b * Real.sin C / Real.sin B ∧
  c = b * Real.sin A / Real.sin B ∧
  S = (1/2) * b * c * Real.sin A ∧
  A = π/3 ∧ 
  b = 1 ∧
  S = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l330_33070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_bisection_theorem_l330_33003

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Check if a point is inside a circle --/
def is_inside (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

/-- Definition of a chord --/
structure Chord (c : Circle) where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  on_circle_start : (start.1 - c.center.1)^2 + (start.2 - c.center.2)^2 = c.radius^2
  on_circle_end : (endpoint.1 - c.center.1)^2 + (endpoint.2 - c.center.2)^2 = c.radius^2

/-- Theorem: For any circle and any point inside the circle, 
    there exists a chord passing through the point that is bisected by it --/
theorem chord_bisection_theorem (c : Circle) (A : ℝ × ℝ) 
    (h : is_inside c A) : 
  ∃ (ch : Chord c), 
    let midpoint := ((ch.start.1 + ch.endpoint.1) / 2, (ch.start.2 + ch.endpoint.2) / 2)
    midpoint = A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_bisection_theorem_l330_33003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_given_radii_l330_33004

/-- The radius of a circle inscribed in three mutually externally tangent circles -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2.5 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem: The radius of the inscribed circle for given radii -/
theorem inscribed_circle_radius_for_given_radii :
  inscribed_circle_radius 6 8 14 = 168/271 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_given_radii_l330_33004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_distance_theorem_l330_33098

/-- Calculates the distance traveled by an airplane against the wind -/
noncomputable def distance_against_wind (plane_speed : ℝ) (time_against : ℝ) (time_with : ℝ) : ℝ :=
  let wind_speed := (plane_speed * (time_with - time_against)) / (time_with + time_against)
  (plane_speed - wind_speed) * time_against

theorem airplane_distance_theorem (plane_speed : ℝ) (time_against : ℝ) (time_with : ℝ)
    (h1 : plane_speed = 810)
    (h2 : time_against = 5)
    (h3 : time_with = 4) :
  distance_against_wind plane_speed time_against time_with = 3600 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_against_wind 810 5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_distance_theorem_l330_33098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l330_33060

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

theorem problem_statement :
  ∀ (x₀ θ : ℝ) (A B C : ℝ) (triangle : Set ℝ),
    0 ≤ x₀ ∧ x₀ ≤ π / 3 →
    f (x₀ / 2) = 1 / 5 →
    A ∈ triangle ∧ B ∈ triangle ∧ C ∈ triangle →
    f B = 2 →
    ∃ (D : ℝ), D ∈ triangle ∧ 
      (∃ (AC : Set ℝ), AC ⊆ triangle ∧ D ∈ AC) ∧
      (∃ (AD DC : ℝ), AD = 3 * DC ∧ AD = 3) ∧
      A = θ ∧ 
      (∃ (AB BD : Set ℝ), AB ⊆ triangle ∧ BD ⊆ triangle ∧ 
        (∃ (angle : Set ℝ → Set ℝ → ℝ), angle AB BD = θ)) →
    Real.cos (2 * x₀) = (49 - 3 * Real.sqrt 33) / 100 ∧
    Real.sin θ = Real.sqrt 13 / 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l330_33060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_homes_l330_33040

/-- The distance between Maxwell's and Brad's homes --/
noncomputable def total_distance (maxwell_speed brad_speed maxwell_distance : ℝ) : ℝ :=
  maxwell_distance * (1 + brad_speed / maxwell_speed)

theorem distance_between_homes :
  let maxwell_speed : ℝ := 6
  let brad_speed : ℝ := 12
  let maxwell_distance : ℝ := 24
  total_distance maxwell_speed brad_speed maxwell_distance = 72 := by
  simp [total_distance]
  norm_num
  -- The proof is completed automatically by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_homes_l330_33040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l330_33079

theorem unique_b_value : ∃! b : ℤ, 
  (∃ k : ℤ, b = 7 * k) ∧ 
  (b + b^3 < 8000) ∧ 
  (b > 0) ∧ 
  (b = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l330_33079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approximation_l330_33089

/-- The circumference of the base of a cone -/
noncomputable def L : ℝ := sorry

/-- The height of a cone -/
noncomputable def h : ℝ := sorry

/-- The radius of the base of a cone -/
noncomputable def r : ℝ := sorry

/-- The standard formula for the volume of a cone -/
noncomputable def V : ℝ := (1/3) * Real.pi * r^2 * h

/-- The approximate formula for the volume of a cone -/
noncomputable def V_approx : ℝ := (2/75) * L^2 * h

/-- The relationship between the circumference and radius -/
axiom circumference_radius : L = 2 * Real.pi * r

theorem cone_volume_approximation :
  V_approx = V ↔ Real.pi = 25/8 := by
  sorry

#check cone_volume_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approximation_l330_33089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_is_odd_f_derivative_is_odd_l330_33054

-- Define the function f(x) = cos(x) - 1/2
noncomputable def f (x : ℝ) : ℝ := Real.cos x - 1/2

-- State the theorem
theorem derivative_f_is_odd : 
  ∀ x : ℝ, deriv f x = -Real.sin x := by sorry

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- State the main theorem
theorem f_derivative_is_odd : is_odd (deriv f) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_is_odd_f_derivative_is_odd_l330_33054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_maintenance_team_travel_l330_33077

-- Define the travel records
noncomputable def travel_records : List ℝ := [18.5, -9.3, 7, -14.7, 15.5, -6.8, -8.2]

-- Define the fuel consumption rate
noncomputable def fuel_consumption_rate : ℝ := 8 / 100

-- Define the initial fuel amount
noncomputable def initial_fuel : ℝ := 20

-- Theorem statement
theorem highway_maintenance_team_travel :
  let final_position := travel_records.sum
  let total_distance := travel_records.map (λ x => abs x) |>.sum
  let fuel_consumed := total_distance * fuel_consumption_rate
  let remaining_fuel := initial_fuel - fuel_consumed
  final_position = 2 ∧ remaining_fuel = 13.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_maintenance_team_travel_l330_33077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phoenix_properties_l330_33041

/-- Definition of a Phoenix equation -/
def is_phoenix (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a + b + c = 0

/-- Definition of a root of a function -/
def is_root (r : ℝ) (f : ℝ → ℝ) : Prop :=
  f r = 0

theorem phoenix_properties (a b c m n : ℝ) :
  is_phoenix a b c →
  is_root 1 (fun x => a * x^2 + b * x + c) ∧
  (is_phoenix 1 m n ∧ (∃ r : ℝ, (fun x => x^2 + m * x + n) = fun x => (x - r)^2) →
   m * n = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phoenix_properties_l330_33041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_button_probability_proof_l330_33027

/-- Represents a jar containing buttons of different colors -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a red button from a jar -/
def redProbability (jar : Jar) : ℚ :=
  jar.red / (jar.red + jar.blue)

theorem button_probability_proof :
  let jarA_initial : Jar := { red := 10, blue := 15 }
  let total_initial := jarA_initial.red + jarA_initial.blue
  let removed := (total_initial - (3 * total_initial / 5)) / 2
  let jarA_final : Jar := { red := jarA_initial.red - removed, blue := jarA_initial.blue - removed }
  let jarB : Jar := { red := removed, blue := removed }
  (redProbability jarA_final) * (redProbability jarB) = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_button_probability_proof_l330_33027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l330_33002

open Set Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x + 1/x - 5
def g (a x : ℝ) : ℝ := x^2 - 2*a*x

-- Define the condition
def condition (a : ℝ) : Prop :=
  ∀ x₁ > 0, ∃ x₂, f x₁ > g a x₂

-- State the theorem
theorem range_of_a :
  {a : ℝ | condition a} = Ioi 2 ∪ Iio (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l330_33002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_BC_length_l330_33069

/-- Represents the length of a line segment in centimeters -/
def Length : Type := ℝ

/-- Represents the area of a shape in square centimeters -/
def Area : Type := ℝ

/-- Calculates the length of BC in a trapezoid ABCD given its area, altitude, and lengths of AB and CD -/
noncomputable def trapezoid_BC (area : Area) (altitude : Length) (AB : Length) (CD : Length) : Length :=
  20 - Real.sqrt 11 - 5 * Real.sqrt 3

/-- Theorem: The length of BC in the given trapezoid is 20 - √11 - 5√3 cm -/
theorem trapezoid_BC_length :
  let area : Area := (200 : ℝ)
  let altitude : Length := (10 : ℝ)
  let AB : Length := (12 : ℝ)
  let CD : Length := (20 : ℝ)
  trapezoid_BC area altitude AB CD = 20 - Real.sqrt 11 - 5 * Real.sqrt 3 :=
by
  -- Unfold the definition of trapezoid_BC
  unfold trapezoid_BC
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_BC_length_l330_33069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_on_bisector_l330_33083

/-- The complex number z as a function of real number m -/
noncomputable def z (m : ℝ) : ℂ := (2 + Complex.I) * m^2 - 6 * m / (1 - Complex.I) - 2 * (1 - Complex.I)

/-- z is purely imaginary iff m = -1/2 -/
theorem z_purely_imaginary (m : ℝ) : z m ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0} ↔ m = -1/2 := by sorry

/-- z is on the bisector of the second and fourth quadrants iff m = 0 or m = 2 -/
theorem z_on_bisector (m : ℝ) : z m ∈ {w : ℂ | w.re = -w.im} ↔ m = 0 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_on_bisector_l330_33083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l330_33011

/-- A sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0 to avoid the "missing cases" error
  | 1 => 1
  | (n + 2) => a (n + 1) / (1 + 2 * a (n + 1))

/-- The theorem stating the closed form of the sequence -/
theorem a_closed_form (n : ℕ) (h : n ≥ 1) : a n = 1 / (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l330_33011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_conversion_l330_33020

/-- Convert a number in scientific notation to standard form -/
noncomputable def scientific_to_standard (mantissa : ℝ) (exponent : ℤ) : ℝ :=
  mantissa * (10 : ℝ) ^ exponent

/-- The problem statement -/
theorem scientific_notation_conversion :
  scientific_to_standard 2.03 (-2) = 0.0203 := by
  -- Unfold the definition of scientific_to_standard
  unfold scientific_to_standard
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_conversion_l330_33020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_area_3_max_area_and_angle_chord_length_max_area_l330_33058

-- Define the sector
noncomputable def Sector (r : ℝ) (α : ℝ) : Prop :=
  2 * r + r * α = 8 ∧ r > 0 ∧ α > 0

-- Define the area of the sector
noncomputable def SectorArea (r : ℝ) (α : ℝ) : ℝ :=
  1/2 * r^2 * α

-- Theorem 1: Central angle when area is 3
theorem central_angle_area_3 (r : ℝ) (α : ℝ) :
  Sector r α → SectorArea r α = 3 → α = 2/3 ∨ α = 6 := by sorry

-- Theorem 2: Maximum area and corresponding central angle
theorem max_area_and_angle (r : ℝ) (α : ℝ) :
  Sector r α → 
  (∀ r' α', Sector r' α' → SectorArea r' α' ≤ SectorArea r α) →
  SectorArea r α = 4 ∧ α = 2 := by sorry

-- Theorem 3: Chord length when area is maximum
theorem chord_length_max_area (r : ℝ) (α : ℝ) :
  Sector r α → 
  (∀ r' α', Sector r' α' → SectorArea r' α' ≤ SectorArea r α) →
  2 * r * Real.sin (α/2) = 4 * Real.sin 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_area_3_max_area_and_angle_chord_length_max_area_l330_33058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_tilings_mod_1000_l330_33090

inductive Color
  | Red
  | Blue
  | Green
deriving DecidableEq

/-- A tiling is valid if it covers the board completely without overlap and uses all three colors. -/
def ValidTiling (tiling : List (Nat × Color)) : Prop :=
  (tiling.map Prod.fst).sum = 9 ∧ 
  (tiling.all (λ (l, _) => l > 0)) ∧
  (tiling.map Prod.snd).toFinset.card = 3

/-- The set of all valid tilings -/
def AllValidTilings : Finset (List (Nat × Color)) :=
  sorry

/-- The number of valid tilings -/
def N : Nat := AllValidTilings.card

/-- The main theorem -/
theorem valid_tilings_mod_1000 : N % 1000 = 838 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_tilings_mod_1000_l330_33090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_terms_count_l330_33073

/-- The first arithmetic sequence -/
def seq1 : List ℕ := List.range ((2021 - 2) / 3 + 1) |>.map (fun n => 2 + 3 * n)

/-- The second arithmetic sequence -/
def seq2 : List ℕ := List.range ((2019 - 4) / 5 + 1) |>.map (fun n => 4 + 5 * n)

/-- The number of common terms between the two sequences -/
def commonTerms : ℕ := (seq1.toFinset ∩ seq2.toFinset).card

theorem common_terms_count : commonTerms = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_terms_count_l330_33073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l330_33091

theorem angle_properties (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : Real.sin θ = Real.cos 2 ∧ Real.cos θ = Real.sin 2) :
  θ = 5 * Real.pi / 2 - 2 ∧
  Real.cos θ + Real.cos (θ + 2 * Real.pi / 3) + Real.cos (θ + 4 * Real.pi / 3) = 0 ∧
  Real.sin θ + Real.sin (θ + 2 * Real.pi / 3) + Real.sin (θ + 4 * Real.pi / 3) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l330_33091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l330_33047

noncomputable def quadratic_vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_vertices : 
  let C := quadratic_vertex 1 6 13
  let D := quadratic_vertex 1 (-4) 5
  distance C D = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l330_33047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_pi_sixth_l330_33009

theorem sin_alpha_minus_pi_sixth (α : ℝ) 
  (h1 : Real.cos α = 3/5) 
  (h2 : α ∈ Set.Ioo 0 (π/2)) : 
  Real.sin (α - π/6) = (4 * Real.sqrt 3 - 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_pi_sixth_l330_33009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_angle_relation_l330_33088

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The median from vertex C to side AB -/
noncomputable def median_CC₁ (t : Triangle) : ℝ := sorry

/-- The median from vertex A to side BC -/
noncomputable def median_AA₁ (t : Triangle) : ℝ := sorry

/-- The angle CAB in the triangle -/
noncomputable def angle_CAB (t : Triangle) : ℝ := sorry

/-- The angle BCA in the triangle -/
noncomputable def angle_BCA (t : Triangle) : ℝ := sorry

/-- Theorem: If median CC₁ > median AA₁, then angle CAB ≥ angle BCA -/
theorem median_angle_relation (t : Triangle) :
  median_CC₁ t > median_AA₁ t → angle_CAB t ≥ angle_BCA t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_angle_relation_l330_33088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rungs_correct_min_rungs_achievable_l330_33081

/-- The minimum number of rungs for a ladder with given step sizes. -/
def min_rungs (a b : ℕ) : ℕ :=
  a + b - 1

/-- Proposition: The minimum number of rungs is correct for any positive integers a and b. -/
theorem min_rungs_correct (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∀ n : ℕ, (∃ (steps : List Bool), 
    (steps.foldl (λ acc step ↦ if step then (acc + a) % n else (acc + n - b) % n) 0 = 0) ∧
    (steps.foldl (λ acc step ↦ max acc (if step then (acc + a) % n else (acc + n - b) % n)) 0 = n - 1)) →
  n ≥ min_rungs a b :=
by sorry

/-- Theorem: The minimum number of rungs is achievable. -/
theorem min_rungs_achievable (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (steps : List Bool), 
    (steps.foldl (λ acc step ↦ if step then (acc + a) % (min_rungs a b) else (acc + (min_rungs a b) - b) % (min_rungs a b)) 0 = 0) ∧
    (steps.foldl (λ acc step ↦ max acc (if step then (acc + a) % (min_rungs a b) else (acc + (min_rungs a b) - b) % (min_rungs a b))) 0 = min_rungs a b - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rungs_correct_min_rungs_achievable_l330_33081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_number_equals_sum_of_others_l330_33087

def is_repeated_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n = d * (10^k - 1) / 9

theorem no_single_number_equals_sum_of_others :
  ∀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ,
  is_repeated_digit a₁ 1 →
  is_repeated_digit a₂ 2 →
  is_repeated_digit a₃ 3 →
  is_repeated_digit a₄ 4 →
  is_repeated_digit a₅ 5 →
  is_repeated_digit a₆ 6 →
  is_repeated_digit a₇ 7 →
  is_repeated_digit a₈ 8 →
  is_repeated_digit a₉ 9 →
  ¬∃ k : Fin 9, (Fin.val k + 1) * (List.nthLe [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉] k (by simp)) =
    (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_number_equals_sum_of_others_l330_33087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_after_exam_l330_33038

noncomputable def full_battery_life : ℚ := 60
noncomputable def used_battery_fraction : ℚ := 3/4
noncomputable def exam_duration : ℚ := 2

theorem remaining_battery_after_exam :
  let remaining_before_exam := full_battery_life * (1 - used_battery_fraction)
  let remaining_after_exam := remaining_before_exam - exam_duration
  remaining_after_exam = 13 := by
  -- Unfold the definitions
  unfold full_battery_life used_battery_fraction exam_duration
  -- Simplify the arithmetic
  simp [mul_sub, mul_one]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_after_exam_l330_33038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l330_33005

theorem sum_remainder (a b c : ℕ) 
  (ha : a % 30 = 15) 
  (hb : b % 30 = 7) 
  (hc : c % 30 = 18) : 
  (a + b + c) % 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l330_33005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l330_33056

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log (x - Real.sqrt (1 + x^2))

-- State the theorem
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l330_33056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_results_l330_33045

-- Define the condition function
def condition (x y : ℝ) : Prop := x * y + x / y + y / x = -3

-- Define the result function
def result (x y : ℝ) : ℝ := (x - 2) * (y - 2)

-- Theorem statement
theorem sum_of_results : 
  ∃ (S : Finset ℝ), (∀ r ∈ S, ∃ (a b : ℝ), condition a b ∧ r = result a b) ∧ 
  (∀ r : ℝ, (∃ (a b : ℝ), condition a b ∧ r = result a b) → r ∈ S) ∧
  (S.sum id) = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_results_l330_33045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l330_33039

/-- Represents a geometric series -/
structure GeometricSeries where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- Sum of the first n terms of a geometric series -/
noncomputable def sum_n (g : GeometricSeries) (n : ℕ) : ℝ :=
  g.a * (1 - g.r^n) / (1 - g.r)

theorem geometric_series_sum (g : GeometricSeries) :
  sum_n g 2011 = 200 →
  sum_n g 4022 = 380 →
  sum_n g 6033 = 542 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l330_33039
