import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_solution_satisfies_conditions_l1284_128414

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (5 * x - 28) / 3

-- State the theorem
theorem unique_fixed_point_of_h :
  ∃! x : ℝ, h x = x ∧ ∀ y : ℝ, h (3 * y + 2) = 5 * y - 6 :=
by
  -- The proof would go here
  sorry

-- Define the specific solution
def solution : ℝ := 14

-- State that the solution satisfies the conditions
theorem solution_satisfies_conditions :
  h solution = solution ∧ ∀ y : ℝ, h (3 * y + 2) = 5 * y - 6 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_solution_satisfies_conditions_l1284_128414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_sequences_count_l1284_128430

/-- Represents the number of tasks initially in the stack -/
def n : ℕ := 10

/-- Represents the number of tasks that have been confirmed as completed -/
def completed : ℕ := 2

/-- Calculates the number of possible task completion sequences -/
def task_sequences : ℕ := Finset.sum (Finset.range (n - completed + 1)) (fun k => (n - completed).choose k * (k + 1))

/-- Theorem stating that the number of possible task completion sequences is 1287 -/
theorem task_sequences_count : task_sequences = 1287 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_sequences_count_l1284_128430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_roots_l1284_128477

/-- A function with specific symmetry and root properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f (2 + x)) ∧
  (∀ x, f (5 - x) = f (5 + x)) ∧
  f 1 = 0 ∧ f 3 = 0

/-- The number of roots of a symmetric function in [-2020, 2020] -/
theorem symmetric_function_roots (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃ s : Finset ℝ, s.card = 1347 ∧ ∀ x ∈ s, x ∈ Set.Icc (-2020 : ℝ) 2020 ∧ f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_roots_l1284_128477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_difference_f_g_l1284_128405

/-- The phase difference between two trigonometric functions -/
noncomputable def phase_difference (f g : ℝ → ℝ) : ℝ := sorry

/-- The first function -/
noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

/-- The second function -/
noncomputable def g (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

/-- Theorem stating the phase difference between the two functions -/
theorem phase_difference_f_g : phase_difference f g = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_difference_f_g_l1284_128405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamps_for_light_envelope_is_9_l1284_128442

/-- The number of stamps needed for an envelope weighing less than 5 pounds -/
noncomputable def stamps_for_light_envelope : ℕ :=
  Nat.ceil ((52 : ℚ) / 6)

/-- The total number of envelopes Micah needed to buy -/
def total_envelopes : ℕ := 14

/-- The number of envelopes weighing less than 5 pounds -/
def light_envelopes : ℕ := 6

/-- The number of stamps needed for an envelope weighing more than 5 pounds -/
def stamps_for_heavy_envelope : ℕ := 5

/-- The total number of stamps Micah bought -/
def total_stamps : ℕ := 52

theorem stamps_for_light_envelope_is_9 :
  stamps_for_light_envelope = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamps_for_light_envelope_is_9_l1284_128442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sum_l1284_128404

theorem cube_root_of_sum (x y : ℝ) : 
  (x^2 = Real.sqrt 16) → 
  (y = ⌊Real.sqrt 40⌋) → 
  ((x + y)^(1/3) = 2 ∨ (x + y)^(1/3) = (4^(1/3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sum_l1284_128404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_determine_plane_l1284_128417

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define what it means for points to be non-collinear
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), (p3.x - p1.x) = t * (p2.x - p1.x) ∧ 
              (p3.y - p1.y) = t * (p2.y - p1.y) ∧ 
              (p3.z - p1.z) = t * (p2.z - p1.z)

-- Define membership for Point3D in Plane
def PointInPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- Instance for Membership Point3D Plane
instance : Membership Point3D Plane where
  mem := PointInPlane

-- Theorem: Three non-collinear points determine a unique plane
theorem three_points_determine_plane (p1 p2 p3 : Point3D) 
  (h : NonCollinear p1 p2 p3) : 
  ∃! (plane : Plane), p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_determine_plane_l1284_128417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l1284_128449

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 7 * x^2 + 4 * y^2 = 28

/-- The line equation -/
def line (x y : ℝ) : Prop := 3 * x - 2 * y - 16 = 0

/-- Distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3 * x - 2 * y - 16| / Real.sqrt 13

/-- The maximum distance from any point on the ellipse to the line -/
theorem max_distance_ellipse_to_line :
  ∃ (max_dist : ℝ), max_dist = (24 / 13) * Real.sqrt 13 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l1284_128449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_unreachable_l1284_128461

/-- Represents a configuration of 27 cubes -/
def CubeConfiguration := Fin 27 → Fin 27

/-- Checks if two positions in the 3x3x3 cube are neighbors -/
def are_neighbors (p q : Fin 27) : Prop := sorry

/-- Represents a single move in the game -/
inductive Move : CubeConfiguration → CubeConfiguration → Prop
| swap (c : CubeConfiguration) (p : Fin 27) :
    are_neighbors p 26 →
    Move c (Function.update (Function.update c p (c 26)) 26 (c p))

/-- Represents a sequence of moves -/
def Reachable : CubeConfiguration → CubeConfiguration → Prop :=
  Relation.ReflTransGen Move

/-- The initial configuration where each cube is in its numbered position -/
def initial_configuration : CubeConfiguration := id

/-- The target configuration where cube n has exchanged places with cube 27-n for 1 ≤ n ≤ 26 -/
def target_configuration : CubeConfiguration :=
  λ n => if n = 26 then 26 else (27 - n)

/-- The main theorem stating that the target configuration is unreachable -/
theorem target_unreachable :
  ¬ Reachable initial_configuration target_configuration := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_unreachable_l1284_128461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scholarship_sum_l1284_128462

/-- Calculates the total scholarship amount for Nina, Kelly, Jason, and Wendy -/
noncomputable def total_scholarship (wendy : ℚ) : ℚ :=
  let kelly := 2 * wendy
  let nina := kelly - 8000
  let jason := 3/4 * kelly
  wendy + kelly + nina + jason

/-- Proves that the total scholarship amount is $122,000 -/
theorem scholarship_sum : total_scholarship 20000 = 122000 := by
  -- Unfold the definition of total_scholarship
  unfold total_scholarship
  -- Simplify the arithmetic expressions
  simp [add_assoc, mul_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scholarship_sum_l1284_128462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noodles_and_pirates_count_l1284_128445

/-- The total number of noodles and pirates -/
def total_count (p : ℕ) (n : ℕ) : ℕ := p + n

/-- The relationship between the number of noodles and pirates -/
def noodle_count (p : ℕ) : ℝ := 2.5 * (p : ℝ) - 3

theorem noodles_and_pirates_count :
  ∃ (n : ℕ), total_count 45 n = 155 ∧ (n : ℝ) = noodle_count 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_noodles_and_pirates_count_l1284_128445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1284_128407

/-- Represents a number in base 8 --/
structure OctalNumber where
  value : ℕ

/-- Converts an OctalNumber to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : ℤ := sorry

/-- Converts a decimal number to its octal representation --/
def decimal_to_octal (n : ℤ) : OctalNumber := sorry

/-- Subtracts two OctalNumbers and returns the result as an OctalNumber --/
def octal_subtract (a b : OctalNumber) : OctalNumber := 
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Creates an OctalNumber from a natural number --/
def mk_octal (n : ℕ) : OctalNumber := ⟨n⟩

theorem octal_subtraction_theorem :
  octal_subtract (mk_octal 45) (mk_octal 76) = decimal_to_octal (-30) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1284_128407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_iff_a_in_range_l1284_128473

/-- Definition of the piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then -x^2 - 2*x else -x

/-- Theorem stating the relationship between the absence of a maximum value and the range of a --/
theorem no_max_iff_a_in_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x > y) ↔ a < -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_iff_a_in_range_l1284_128473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_sqrt3_over_2_l1284_128468

noncomputable def a : ℝ × ℝ := (Real.cos (23 * Real.pi / 180), Real.cos (67 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (53 * Real.pi / 180), Real.cos (37 * Real.pi / 180))

theorem dot_product_equals_sqrt3_over_2 :
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_sqrt3_over_2_l1284_128468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1284_128470

/-- A geometric sequence with positive terms where a₁, (1/2)a₃, 2a₂ form an arithmetic sequence -/
structure GeometricSequence (a : ℕ → ℝ) : Prop where
  positive : ∀ n, a n > 0
  geometric : ∃ q > 0, ∀ n, a (n + 1) = q * a n
  arithmetic : a 1 + 2 * a 2 = a 3

/-- The ratio of the sum of every other term starting from the 6th to the sum of every other term starting from the 7th -/
noncomputable def SequenceRatio (a : ℕ → ℝ) : ℝ :=
  (a 6 + a 8 + a 10) / (a 7 + a 9 + a 11)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) : 
  SequenceRatio a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1284_128470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1284_128456

theorem triangle_angle_range (A B C : ℝ) (h1 : A = π/4) (h2 : Real.sin B > Real.sqrt 2 * Real.cos C) :
  π/4 < C ∧ C < 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1284_128456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l1284_128429

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * k * x - 1

theorem function_monotonicity (k : ℝ) :
  (∀ x₁ x₂, 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    (x₁ - x₂) * (f k x₁ - f k x₂) < 0) ↔
  k ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l1284_128429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f1_domain_f2_domain_f3_domain_f4_l1284_128433

-- Function 1
noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (3 + 2*x)

theorem domain_f1 : 
  {x : ℝ | ∃ y, f1 x = y} = Set.Ici (-3/2) := by sorry

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ := 1 + Real.sqrt (9 - x^2)

theorem domain_f2 : 
  {x : ℝ | ∃ y, f2 x = y} = Set.Icc (-3) 3 := by sorry

-- Function 3
noncomputable def f3 (x : ℝ) : ℝ := Real.sqrt (Real.log ((5*x - x^2)/4) / Real.log 10)

theorem domain_f3 : 
  {x : ℝ | ∃ y, f3 x = y} = Set.Icc 1 4 := by sorry

-- Function 4
noncomputable def f4 (x : ℝ) : ℝ := Real.sqrt (3 - x) + Real.arccos ((x - 2)/3)

theorem domain_f4 : 
  {x : ℝ | ∃ y, f4 x = y} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f1_domain_f2_domain_f3_domain_f4_l1284_128433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_vector_sum_l1284_128411

/-- Circle C with center (2, 2) and radius 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 = 1}

/-- Line l with equation x + y = 1 -/
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1}

/-- Vector from origin to a point -/
def vecFromOrigin (p : ℝ × ℝ) : ℝ × ℝ := p

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- The minimum value of |OP + OQ| -/
noncomputable def minValue : ℝ := (5 * Real.sqrt 2 - 2) / 2

theorem min_value_of_vector_sum :
  ∀ P Q, P ∈ C → Q ∈ l →
    ∃ P' Q', P' ∈ C ∧ Q' ∈ l ∧
      magnitude (vecFromOrigin P' + vecFromOrigin Q') ≤ magnitude (vecFromOrigin P + vecFromOrigin Q) ∧
      magnitude (vecFromOrigin P' + vecFromOrigin Q') = minValue :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_vector_sum_l1284_128411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_f4_range_theorem_m_range_theorem_l1284_128415

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Statement 1
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hf2 : f a b 2 = 1) :
  (1 / a + 2 / b) ≥ 8 :=
sorry

-- Statement 2
theorem f4_range_theorem (a b : ℝ) (h : ∀ x ∈ Set.Icc 1 2, 0 ≤ f a b x ∧ f a b x ≤ 1) :
  f a b 4 ∈ Set.Icc (-2) 3 :=
sorry

-- Statement 3
theorem m_range_theorem (m : ℝ) :
  (∀ x > 2, g x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_f4_range_theorem_m_range_theorem_l1284_128415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1284_128469

/-- The function f(x) = ae^x - ln x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

/-- f is monotonically increasing on the interval (1,2) --/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f a x ≤ f a y

/-- The theorem stating the minimum value of a --/
theorem min_value_of_a :
  ∀ a : ℝ, is_monotone_increasing a → a ≥ Real.exp (-1) := by
  sorry

#check min_value_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1284_128469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_l1284_128465

noncomputable section

-- Define the dimensions of the rectangular prism
def a : ℝ := 12
def b : ℝ := 18
def c : ℝ := 15

-- Define the diagonal length
noncomputable def diagonal : ℝ := Real.sqrt (a^2 + b^2 + c^2)

-- Theorem statement
theorem rectangular_prism_diagonal :
  diagonal = 3 * Real.sqrt 77 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_l1284_128465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_in_bushel_l1284_128490

/-- Represents the number of apples in a bushel -/
def apples_per_bushel : ℕ := 48

/-- The cost of one bushel of apples in dollars -/
def cost_per_bushel : ℚ := 12

/-- The selling price of one apple in dollars -/
def price_per_apple : ℚ := 2/5

/-- The profit made after selling 100 apples in dollars -/
def profit_100_apples : ℚ := 15

/-- The number of apples sold -/
def apples_sold : ℕ := 100

theorem apples_in_bushel :
  (cost_per_bushel / apples_per_bushel) * apples_sold + profit_100_apples = price_per_apple * apples_sold →
  apples_per_bushel = 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_in_bushel_l1284_128490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beating_sets_count_l1284_128416

/-- Represents a basketball league with the given conditions -/
structure BasketballLeague where
  teams : Finset Nat
  games : Finset (Nat × Nat)
  wins : Nat → Nat
  losses : Nat → Nat
  played_all_others : ∀ t1 t2, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → (t1, t2) ∈ games ∨ (t2, t1) ∈ games
  no_ties : ∀ (g : Nat × Nat), g ∈ games → (g.2, g.1) ∉ games
  wins_count : ∀ t, t ∈ teams → wins t = 15
  losses_count : ∀ t, t ∈ teams → losses t = 6

/-- Counts the number of sets of four teams with the specified beating condition -/
def count_beating_sets (league : BasketballLeague) : Nat :=
  sorry

/-- The main theorem stating that the count of beating sets is 1643 -/
theorem beating_sets_count (league : BasketballLeague) : count_beating_sets league = 1643 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beating_sets_count_l1284_128416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_condition_l1284_128489

theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, (m - 1) * x^2 + 3 * x - 1 = 0 → (m - 1 ≠ 0)) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_condition_l1284_128489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_sufficient_not_necessary_l1284_128475

-- Define the lines l₁ and l₂ as functions of m
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + 2 * m * y - 1 = 0

def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (3 * m - 1) * x - m * y - 1 = 0

-- Define parallelism of two lines
def parallel (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)

-- Define the condition log₆(m) = -1
noncomputable def log_condition (m : ℝ) : Prop :=
  Real.log m / Real.log 6 = -1

-- Statement to prove
theorem log_condition_sufficient_not_necessary (m : ℝ) :
  (log_condition m → parallel (l₁ m) (l₂ m)) ∧
  ¬(parallel (l₁ m) (l₂ m) → log_condition m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_sufficient_not_necessary_l1284_128475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1284_128406

/-- Given that the terminal side of angle α passes through point P(-4a, 3a) with a < 0,
    prove that sin α = -3/5 -/
theorem sin_alpha_value (a : ℝ) (α : ℝ) (h1 : a < 0) :
  let P : ℝ × ℝ := (-4*a, 3*a)
  Real.sin α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1284_128406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_y_range_l1284_128435

/-- The parabola C: x² = 8y -/
def C (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus of the parabola C -/
def F : ℝ × ℝ := (0, 2)

/-- The directrix of the parabola C -/
def directrix : ℝ → ℝ := λ x ↦ -2

theorem parabola_point_y_range (x₀ y₀ : ℝ) :
  C x₀ y₀ → y₀ > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_y_range_l1284_128435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_l1284_128446

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) :
  n = 40 →
  original_mean = 100 →
  incorrect_value = 75 →
  correct_value = 50 →
  ((n : ℝ) * original_mean - (incorrect_value - correct_value)) / n = 99.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_l1284_128446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_count_l1284_128494

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the volume of a rectangular object -/
def volume (d : Dimensions) : ℚ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def mToCm (m : ℚ) : ℚ :=
  m * 100

/-- Calculates the number of bricks needed for a wall -/
def bricksNeeded (wallDim : Dimensions) (brickDim : Dimensions) : ℚ :=
  volume wallDim / volume brickDim

theorem wall_bricks_count (wallDim brickDim : Dimensions) :
  let wallDimCm : Dimensions := {
    length := mToCm wallDim.length,
    width := wallDim.width,
    height := mToCm wallDim.height
  }
  (bricksNeeded wallDimCm brickDim).floor = 6400 :=
by
  sorry

#eval (bricksNeeded
  { length := 800, width := 22.5, height := 600 }
  { length := 25, width := 11.25, height := 6 }).floor


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_count_l1284_128494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_expression_simplification_equation_solution_l1284_128410

-- Question 1
theorem factorization_problem (a : ℝ) : 2*a^3 - 12*a^2 + 8*a = 2*a*(a^2 - 6*a + 4) := by sorry

-- Question 2
theorem expression_simplification (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) :
  3/a - 6/(1-a) - (a+5)/(a^2-a) = 8/a := by sorry

-- Question 3
theorem equation_solution :
  ∃ x : ℝ, x = -1 ∧ (x-2)/(x+2) - 12/(x^2-4) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_expression_simplification_equation_solution_l1284_128410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l1284_128487

/-- The parabola y = 4x^2 -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2

/-- The focus of the parabola y = 4x^2 -/
noncomputable def focus : ℝ × ℝ := (0, 1/16)

/-- The directrix of the parabola y = 4x^2 -/
def directrix (y : ℝ) : Prop := y = -1/16

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_ordinate (x y : ℝ) :
  parabola x y →
  distance (x, y) focus = 1 →
  y = 15/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l1284_128487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_zero_l1284_128452

/-- A rectangle WXYZ with specified points and segments -/
structure Rectangle where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  O : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_WZ : W.2 = Z.2 ∧ Z.1 - W.1 = 6
  h_XY : X.1 = Y.1 ∧ W.2 - Y.2 = 3
  h_M : M.1 - W.1 = 2 ∧ M.2 = W.2
  h_N : N.1 = X.1 ∧ N.2 - X.2 = 1
  h_O : O.1 = Z.1 ∧ O.2 - Y.2 = 1
  h_R : R ∈ Set.range (fun t => (1 - t) • W + t • N) ∧
        R ∈ Set.range (fun t => (1 - t) • M + t • O)
  h_S : S ∈ Set.range (fun t => (1 - t) • W + t • Y) ∧
        S ∈ Set.range (fun t => (1 - t) • M + t • O)

/-- The ratio RS/MO is zero in the given rectangle -/
theorem ratio_is_zero (rect : Rectangle) : 
  dist rect.R rect.S / dist rect.M rect.O = 0 := by
  sorry

#check ratio_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_zero_l1284_128452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_extrema_condition_l1284_128459

def f (a : ℝ) : ℝ → ℝ := λ x => x^3 + 2*a*x^2 + 3*(a+2)*x + 1

theorem cubic_function_extrema_condition (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) → (a > 2 ∨ a < -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_extrema_condition_l1284_128459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_sixty_times_a_equals_100_l1284_128496

noncomputable def A : ℝ × ℝ := (1, 2)
noncomputable def B : ℝ × ℝ := (3, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem minimize_distance_sum :
  ∃ a : ℝ, ∀ x : ℝ,
    distance A (x, 0) + distance (x, 0) B ≥ distance A (a, 0) + distance (a, 0) B ∧
    a = 5/3 :=
by sorry

theorem sixty_times_a_equals_100 :
  let a := (5 : ℝ) / 3
  60 * a = 100 :=
by
  intro a
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_sixty_times_a_equals_100_l1284_128496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tangent_roots_l1284_128413

theorem acute_triangle_tangent_roots (A B C : ℝ) (p q : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Acute triangle condition
  A + B + C = π / 2 →  -- Sum of angles in a right-angled triangle
  q ≠ 1 →  -- Given condition
  (∀ x : ℝ, x^3 + p*x^2 + q*x + p = 0 ↔ x = Real.tan A ∨ x = Real.tan B ∨ x = Real.tan C) →  -- Roots condition
  p ≤ -3 * Real.sqrt 3 ∧ q > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tangent_roots_l1284_128413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1284_128403

/-- Represents the time it takes Taimour to paint the fence alone -/
noncomputable def taimour_time : ℝ := sorry

/-- Represents the time it takes Jamshid to paint the fence alone -/
noncomputable def jamshid_time : ℝ := taimour_time / 2

/-- Represents the time it takes Taimour and Jamshid to paint the fence together -/
def combined_time : ℝ := 7

/-- Theorem stating that Taimour takes 21 hours to paint the fence alone -/
theorem taimour_paint_time :
  (1 / taimour_time + 1 / jamshid_time) * combined_time = 1 →
  taimour_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1284_128403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_two_alpha_l1284_128457

theorem cos_pi_third_plus_two_alpha (α : ℝ) :
  Real.sin (π / 3 - α) = 1 / 3 → Real.cos (π / 3 + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_two_alpha_l1284_128457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1284_128492

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, and a distance of 15 cm between them, is 285 square centimeters -/
theorem trapezium_area_example : trapezium_area 20 18 15 = 285 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic expression
  simp [mul_add, mul_div_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1284_128492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_typing_time_l1284_128463

/-- Represents the time it takes for Jack to type the document alone -/
noncomputable def jack_time : ℝ := 24

/-- Represents the typing rate of Jonathan in pages per minute -/
noncomputable def jonathan_rate : ℝ := 40 / 40

/-- Represents the typing rate of Susan in pages per minute -/
noncomputable def susan_rate : ℝ := 40 / 30

/-- Represents the combined typing rate of Jonathan, Susan, and Jack in pages per minute -/
noncomputable def combined_rate : ℝ := 40 / 10

/-- Represents Jack's typing rate in pages per minute -/
noncomputable def jack_rate : ℝ := 40 / jack_time

theorem jack_typing_time :
  jonathan_rate + susan_rate + jack_rate = combined_rate := by
  sorry

#check jack_typing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_typing_time_l1284_128463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1284_128408

-- Define the custom operation
noncomputable def circleSlash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 3

-- State the theorem
theorem solve_equation (x : ℝ) : circleSlash 3 x = 64 → x = 7 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1284_128408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buy_thirteen_popsicles_l1284_128441

/-- The number of popsicles that can be bought with a given amount of money -/
def maxPopsicles (money : ℚ) : ℕ :=
  let regularPrice : ℚ := 2
  let discountedPrice : ℚ := 3/2
  let regularCount : ℕ := min 10 (Int.floor (money / regularPrice)).toNat
  let remainingMoney : ℚ := money - (regularCount : ℚ) * regularPrice
  let discountedCount : ℕ := (Int.floor (remainingMoney / discountedPrice)).toNat
  regularCount + discountedCount

/-- Theorem stating that $25.50 allows the purchase of exactly 13 popsicles -/
theorem buy_thirteen_popsicles :
  maxPopsicles (51/2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buy_thirteen_popsicles_l1284_128441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1284_128434

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * ((1 + rate / frequency) ^ (frequency * time) - 1)

/-- Theorem: Given conditions lead to 4% annual interest rate -/
theorem interest_rate_is_four_percent (principal : ℝ) (time : ℝ) (frequency : ℝ) (interest : ℝ) 
  (h1 : principal = 10000)
  (h2 : time = 2)
  (h3 : frequency = 2)
  (h4 : interest = 824.32) :
  ∃ (rate : ℝ), compound_interest principal rate time frequency = interest ∧ rate = 0.04 := by
  sorry

#check interest_rate_is_four_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1284_128434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_example_l1284_128419

-- Define the diamond operation as noncomputable
noncomputable def diamond (x y : ℝ) : ℝ := (x + y) / (x + y - 2 * x * y)

-- State the theorem
theorem diamond_example : diamond (diamond 3 4) 5 = 39 / 74 := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_example_l1284_128419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_root_vs_single_root_l1284_128448

theorem double_root_vs_single_root :
  ∃ (quadratic linear : ℝ → ℝ),
    (∀ x : ℝ, quadratic x = 0 ↔ x = 2) ∧
    (∀ x : ℝ, linear x = 0 ↔ x = 2) ∧
    (∃! r : ℝ, quadratic r = 0 ∧ (deriv quadratic r = 0)) ∧
    (∃! r : ℝ, linear r = 0) :=
by
  let quadratic := λ x : ℝ => x^2 - 4*x + 4
  let linear := λ x : ℝ => -4*x + 8
  exists quadratic, linear
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_root_vs_single_root_l1284_128448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1284_128493

-- Define the parabola on which P moves
def parabola (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define point A
def A : ℝ × ℝ := (0, -1)

-- State the theorem
theorem midpoint_trajectory :
  ∀ x y : ℝ,
  ∃ px : ℝ,
  let p : ℝ × ℝ := (px, parabola px)
  let m : ℝ × ℝ := ((px + A.1) / 2, (parabola px + A.2) / 2)
  m = (x, y) →
  y = 4 * x^2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1284_128493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_eq_neg_one_l1284_128453

theorem cos_2theta_eq_neg_one (θ : ℝ) :
  (2 : ℝ)^(-5/3 + Real.sin (2*θ)) + 2 = (2 : ℝ)^(1/3 + Real.sin θ) →
  Real.cos (2*θ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_eq_neg_one_l1284_128453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1284_128409

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h : S seq 5 > S seq 6) :
  ¬ (∀ seq : ArithmeticSequence, seq.a 3 + seq.a 6 + seq.a 12 < 2 * seq.a 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1284_128409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_2023_l1284_128424

-- Define the left-hand side of the equation
noncomputable def lhs (x : ℝ) : ℝ :=
  Real.sqrt (2 * x^2 - 2024 * x + 1023131) +
  Real.sqrt (3 * x^2 - 2025 * x + 1023132) +
  Real.sqrt (4 * x^2 - 2026 * x + 1023133)

-- Define the right-hand side of the equation
noncomputable def rhs (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - x + 1) +
  Real.sqrt (2 * x^2 - 2 * x + 2) +
  Real.sqrt (3 * x^2 - 3 * x + 3)

-- Define the equation
def equation (x : ℝ) : Prop := lhs x = rhs x

-- Theorem statement
theorem sum_of_roots_is_2023 :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ equation r₁ ∧ equation r₂ ∧ r₁ + r₂ = 2023 ∧
  ∀ (x : ℝ), equation x → x = r₁ ∨ x = r₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_2023_l1284_128424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_cost_theorem_l1284_128426

/-- Represents the cost and quantity of a candy type -/
structure CandyType where
  cost : ℚ
  quantity : ℕ

/-- Calculates the total cost for all classmates given candy types and number of classmates -/
def totalCost (candyTypes : List CandyType) (classmates : ℕ) : ℚ :=
  let oneCost := candyTypes.map (λ c => c.cost * (c.quantity : ℚ))
  (oneCost.sum) * (classmates : ℚ)

/-- The problem statement -/
theorem candy_cost_theorem :
  let candyTypes : List CandyType := [
    { cost := 1/10, quantity := 3 },
    { cost := 3/20, quantity := 2 },
    { cost := 1/5, quantity := 1 },
    { cost := 1/4, quantity := 4 }
  ]
  let classmates : ℕ := 35
  totalCost candyTypes classmates = 63/1 := by
  sorry

#eval totalCost [
    { cost := 1/10, quantity := 3 },
    { cost := 3/20, quantity := 2 },
    { cost := 1/5, quantity := 1 },
    { cost := 1/4, quantity := 4 }
  ] 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_cost_theorem_l1284_128426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_outside_circle_l1284_128421

/-- Point on a number line -/
structure Point where
  coordinate : ℝ

/-- Circle on a number line -/
structure Circle where
  center : Point
  radius : ℝ

/-- Define whether a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  |p.coordinate - c.center.coordinate| < c.radius

/-- The main theorem -/
theorem exists_point_outside_circle : 
  ∃ (a : ℝ), a < 5 ∧ ¬isInside (Point.mk a) (Circle.mk (Point.mk 3) 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_outside_circle_l1284_128421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_correct_l1284_128454

-- Define the water fee calculation function
noncomputable def water_fee (a : ℝ) : ℝ :=
  if a ≤ 15 then 2 * a else 2.5 * a - 7.5

-- State the theorem
theorem water_fee_correct :
  (∀ a : ℝ, a ≥ 0 → a ≤ 15 → water_fee a = 2 * a) ∧
  (∀ a : ℝ, a > 15 → water_fee a = 2.5 * a - 7.5) ∧
  (water_fee 20 = 42.5) ∧
  (∃ a : ℝ, water_fee a = 55 ∧ a = 25) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_correct_l1284_128454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l1284_128438

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) := ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ ∀ n, b (n + 1) = b n * q

theorem arithmetic_geometric_sequence_problem 
  (a b : ℕ → ℝ) (α β : ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 1 = 3 →
  b 1 = 1 →
  a 2 = b 2 →
  3 * a 5 = b 3 →
  (∀ n : ℕ, a n = Real.log (b n) / Real.log α + β) →
  α + β = Real.rpow 3 (1/3) + 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l1284_128438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probabilities_l1284_128432

/-- Represents the color of a ball -/
inductive Color where
  | White
  | Red
deriving Repr, DecidableEq

/-- Represents the contents of the box -/
def Box : Finset Color := {Color.White, Color.White, Color.White, Color.Red, Color.Red}

/-- The number of ways to choose 2 balls from the box -/
def TotalChoices : ℕ := Nat.choose Box.card 2

/-- The number of ways to choose 2 red balls -/
def RedChoices : ℕ := Nat.choose (Box.filter (· = Color.Red)).card 2

/-- The number of ways to choose 1 white ball and 1 red ball -/
def MixedChoices : ℕ := (Box.filter (· = Color.White)).card * (Box.filter (· = Color.Red)).card

theorem ball_probabilities :
  (RedChoices : ℚ) / TotalChoices = 1 / 10 ∧
  (MixedChoices : ℚ) / TotalChoices = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probabilities_l1284_128432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_property_l1284_128412

-- Define the polynomial type
def MyPolynomial (α : Type) := α → α

-- Define the property that P(2x) = P'(x) P''(x) for all real x
def SatisfiesProperty (P : MyPolynomial ℝ) : Prop :=
  ∀ x : ℝ, P (2 * x) = (deriv (fun y => P y) x) * (deriv (fun y => deriv (fun z => P z) y) x)

-- State the theorem
theorem unique_polynomial_property :
  ∃! P : MyPolynomial ℝ, SatisfiesProperty P ∧ 
    (∀ x : ℝ, P x = (4/9 : ℝ) * x^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_property_l1284_128412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_inequality_l1284_128400

/-- Represents the sum of the first n terms of an arithmetic sequence. -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- 
For an arithmetic sequence with first term a₁ and common difference d,
d > 0 is equivalent to S_n + S_3n > 2S_2n for all positive n.
-/
theorem arithmetic_sequence_sum_inequality (a₁ d : ℝ) :
  d > 0 ↔ ∀ n : ℕ+, S a₁ d n + S a₁ d (3*n) > 2 * S a₁ d (2*n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_inequality_l1284_128400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1284_128491

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/9 + y^2/3 = 1

/-- The circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x-1)^2 + y^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The minimum distance between C₁ and C₂ -/
theorem min_distance_C₁_C₂ :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
    distance x₁ y₁ x₂ y₂ ≥ Real.sqrt 10 / 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1284_128491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prism_less_than_20_shapes_l1284_128483

/-- Represents a shape made of 5 unit cubes as described in the problem -/
structure Shape where
  volume : ℕ := 5

/-- Represents a solid rectangular prism -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Function to check if a prism can be constructed from a given number of shapes -/
def can_construct_prism (n : ℕ) (p : RectangularPrism) (s : Shape) : Prop :=
  n * s.volume = p.length * p.width * p.height

/-- Theorem stating that there exists a rectangular prism that can be constructed from fewer than 20 shapes -/
theorem exists_prism_less_than_20_shapes :
  ∃ (n : ℕ) (p : RectangularPrism), n < 20 ∧ can_construct_prism n p { volume := 5 } := by
  -- We'll use 10 shapes to construct a 1 x 5 x 10 prism
  let n : ℕ := 10
  let p : RectangularPrism := { length := 1, width := 5, height := 10 }
  let s : Shape := { volume := 5 }
  
  have h1 : n < 20 := by norm_num
  have h2 : can_construct_prism n p s := by
    simp [can_construct_prism]
    norm_num
  
  exact ⟨n, p, h1, h2⟩

#check exists_prism_less_than_20_shapes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prism_less_than_20_shapes_l1284_128483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1284_128499

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 1

-- Theorem statement
theorem range_of_f :
  {y : ℝ | ∃ x ∈ domain, f x = y} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1284_128499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_implies_side_division_l1284_128418

/-- A triangle with a line parallel to its base -/
structure DividedTriangle where
  /-- The ratio of the area of the smaller triangle to the total area -/
  area_ratio : ℝ
  /-- The line is parallel to the base -/
  is_parallel : Bool

/-- The theorem stating the relationship between area ratio and side division -/
theorem area_ratio_implies_side_division (t : DividedTriangle) 
  (h1 : t.area_ratio = 2/3) 
  (h2 : t.is_parallel = true) : 
  ∃ (k : ℝ), k = Real.sqrt 6 + 2 ∧ 
  (∀ side : ℝ, ∃ (a b : ℝ), side = a + b ∧ a / b = k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_implies_side_division_l1284_128418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_cosine_l1284_128471

/-- Given a triangle ABC where sides a, b, c opposite to angles A, B, C respectively form an arithmetic sequence, and A - C = 90°, prove that cos B = 3/4 -/
theorem triangle_arithmetic_sequence_cosine (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧  -- sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- sides are positive
  2 * b = a + c ∧  -- arithmetic sequence condition
  A - C = π / 2 →  -- given angle condition
  Real.cos B = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_cosine_l1284_128471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l1284_128464

/-- Given a function f(x) = ax + ln(x) where one of its tangent lines is y = x, 
    prove that a = 1 - 1/e -/
theorem tangent_line_condition (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 
    (fun x ↦ a * x + Real.log x) x = x ∧ 
    (fun x ↦ a + 1/x) x = 1) → 
  a = 1 - 1/Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l1284_128464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_is_true_negation_of_p_l1284_128467

noncomputable def f (x : ℝ) : ℝ := (2/3) ^ x

theorem proposition_p_is_true : ∀ x : ℝ, x ≥ 0 → f x ≤ 1 := by sorry

theorem negation_of_p : ¬(∀ x : ℝ, x ≥ 0 → f x ≤ 1) ↔ ∃ x : ℝ, x ≥ 0 ∧ f x > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_is_true_negation_of_p_l1284_128467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_f_double_prime_at_1_l1284_128472

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the conditions
axiom derivative_is_second_derivative : ∀ x, deriv f x = deriv (deriv f) x
axiom tangent_line_at_1 : f 1 = 3 ∧ deriv f 1 = -1

-- State the theorem
theorem f_minus_f_double_prime_at_1 : f 1 - deriv (deriv f) 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_f_double_prime_at_1_l1284_128472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_not_power_of_3_and_5_l1284_128427

-- Define the sequence
def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

-- Define the property we want to prove
def not_power_of_3_and_5 (n : ℕ) : Prop :=
  ∀ α β : ℕ, v n ≠ (3 : ℤ) ^ α * (5 : ℤ) ^ β

-- State the theorem
theorem v_not_power_of_3_and_5 : ∀ n : ℕ, not_power_of_3_and_5 n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_not_power_of_3_and_5_l1284_128427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1284_128476

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) + 1 / (4 - x^2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > -2 ∧ x ≠ 2}

-- Theorem stating that domain_f is indeed the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1284_128476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_turnips_and_carrots_l1284_128466

structure Farmer :=
  (name : String)
  (week1_turnips : ℕ)
  (week1_carrots : ℕ)
  (turnip_growth_rate : ℚ)
  (carrot_growth_rate : ℚ)

def total_turnips (f : Farmer) : ℕ :=
  f.week1_turnips + Int.toNat ((f.week1_turnips : ℚ) * f.turnip_growth_rate).floor

def total_carrots (f : Farmer) : ℕ :=
  f.week1_carrots + Int.toNat ((f.week1_carrots : ℚ) * f.carrot_growth_rate).floor

def melanie : Farmer :=
  { name := "Melanie"
  , week1_turnips := 139
  , week1_carrots := 45
  , turnip_growth_rate := 2
  , carrot_growth_rate := 5/4 }

def benny : Farmer :=
  { name := "Benny"
  , week1_turnips := 113
  , week1_carrots := 75
  , turnip_growth_rate := 13/10
  , carrot_growth_rate := 13/10 }

def carol : Farmer :=
  { name := "Carol"
  , week1_turnips := 195
  , week1_carrots := 60
  , turnip_growth_rate := 6/5
  , carrot_growth_rate := 7/5 }

theorem most_turnips_and_carrots :
  (total_turnips carol > total_turnips melanie) ∧
  (total_turnips carol > total_turnips benny) ∧
  (total_carrots benny > total_carrots melanie) ∧
  (total_carrots benny > total_carrots carol) := by
  sorry

#eval total_turnips carol
#eval total_turnips melanie
#eval total_turnips benny
#eval total_carrots benny
#eval total_carrots melanie
#eval total_carrots carol

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_turnips_and_carrots_l1284_128466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1284_128497

/-- The circle equation: x^2 + y^2 + 2x - 2y - 2 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 2 = 0

/-- The line equation: 3x + 4y + 14 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  3*x + 4*y + 14 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 1)

/-- Distance from a point (x, y) to the line 3x + 4y + 14 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x + 4*y + 14| / Real.sqrt (3^2 + 4^2)

theorem distance_from_circle_center_to_line :
  distance_to_line circle_center.1 circle_center.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1284_128497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_bobbi_probability_correct_l1284_128455

/-- Calculates the probability that Billy and Bobbi selected the same number 
    given an upper bound and the LCM of their number's factors. -/
def billy_bobbi_probability (upper_bound lcm_value : ℕ) : ℚ :=
  let billy_multiples := (upper_bound - 1) / 18
  let bobbi_multiples := (upper_bound - 1) / 24
  let common_multiples := (upper_bound - 1) / lcm_value
  let total_combinations := billy_multiples * bobbi_multiples
  ↑common_multiples / ↑total_combinations

theorem billy_bobbi_probability_correct : 
  billy_bobbi_probability 200 72 = 1 / 44 := by
  -- Unfold the definition and simplify
  unfold billy_bobbi_probability
  simp
  -- The rest of the proof would go here
  sorry

#eval billy_bobbi_probability 200 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_bobbi_probability_correct_l1284_128455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_C_value_l1284_128437

theorem smallest_C_value : ∃ (C : ℝ),
  (∀ (x y : ℝ), x^2 * (1 + y) + y^2 * (1 + x) ≤ Real.sqrt ((x^4 + 4) * (y^4 + 4)) + C) ∧
  (∀ (C' : ℝ), C' < C → ∃ (x y : ℝ), x^2 * (1 + y) + y^2 * (1 + x) > Real.sqrt ((x^4 + 4) * (y^4 + 4)) + C') ∧
  C = 4 :=
by
  sorry

#check smallest_C_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_C_value_l1284_128437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l1284_128485

noncomputable def f (x : ℝ) := 2 * x / (x^2 + 1)

theorem f_monotonicity_and_inequality :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ y < -1 → f x > f y) ∧
  (∀ x y : ℝ, 1 < x ∧ x < y → f x > f y) ∧
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, 2 * m - 1 > f x) :=
by
  sorry

#check f_monotonicity_and_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l1284_128485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_most_relevant_for_restocking_l1284_128436

/-- Represents a refrigerator model with its capacity and sales quantity -/
structure RefrigeratorModel where
  capacity : ℕ
  sales : ℕ

/-- Calculates the mode of a list of natural numbers -/
def mode (list : List ℕ) : ℕ := sorry

/-- Determines if a given statistic is the most relevant for restocking decisions -/
def is_most_relevant_for_restocking (statistic : List RefrigeratorModel → ℕ) : Prop := sorry

/-- The main theorem stating that the mode of sales quantities is the most relevant statistic for restocking decisions -/
theorem mode_most_relevant_for_restocking 
  (models : List RefrigeratorModel) : 
  is_most_relevant_for_restocking (λ models => mode (models.map RefrigeratorModel.sales)) := by
  sorry

/-- Example data from the problem -/
def example_models : List RefrigeratorModel := [
  ⟨220, 6⟩,
  ⟨215, 30⟩,
  ⟨185, 14⟩,
  ⟨182, 8⟩
]

#check mode_most_relevant_for_restocking example_models

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_most_relevant_for_restocking_l1284_128436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coordinates_l1284_128481

/-- Given two vectors a and b in ℝ², prove that a = (2√5, √5) -/
theorem vector_coordinates (a b : ℝ × ℝ) (l : ℝ) : 
  (l > 0) →
  (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 5) →
  (b = (2, 1)) →
  (a = l • b) →
  (a = (2 * Real.sqrt 5, Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coordinates_l1284_128481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1284_128450

theorem cos_double_angle_special_case (α : ℝ) (h : Real.sin α = 3/5) : 
  Real.cos (2 * α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1284_128450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_eq_one_l1284_128431

/-- Given a non-zero constant a, define the function f(x) for x > a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 3) / (x - a)

/-- Theorem stating that if the minimum value of f(x) is 6, then a = 1 -/
theorem min_value_implies_a_eq_one (a : ℝ) (h_a : a ≠ 0) :
  (∀ x > a, f a x ≥ 6) ∧ (∃ x > a, f a x = 6) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_eq_one_l1284_128431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l1284_128425

theorem division_with_remainder : 
  ∃ (x : ℕ), (64 / x = 5 ∧ 64 % x = 4) ↔ x = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l1284_128425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_greater_than_three_fourths_perimeter_l1284_128447

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the perimeter of a triangle
noncomputable def perimeter (t : Triangle) : ℝ :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B + d t.B t.C + d t.C t.A

-- Define a median of a triangle
noncomputable def median (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  let midpoint (p q : ℝ × ℝ) := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  if vertex = t.A then d vertex (midpoint t.B t.C)
  else if vertex = t.B then d vertex (midpoint t.A t.C)
  else d vertex (midpoint t.A t.B)

-- Theorem: The sum of the lengths of the three medians is always greater than or equal to 3/4 of the perimeter
theorem median_sum_greater_than_three_fourths_perimeter (t : Triangle) :
  median t t.A + median t t.B + median t t.C ≥ (3/4) * perimeter t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_greater_than_three_fourths_perimeter_l1284_128447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_lines_l1284_128439

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the line
def line_L (x y : ℝ) : Prop := 2*x - y + 4 = 0

-- Define point M
def point_M : ℝ × ℝ := (3, 1)

-- Theorem for the length of the chord
theorem chord_length : 
  ∃ (a b c d : ℝ), 
    line_L a b ∧ circle_C a b ∧
    line_L c d ∧ circle_C c d ∧ 
    (a ≠ c ∨ b ≠ d) ∧
    ((a - c)^2 + (b - d)^2)^(1/2) = 4*5^(1/2)/5 :=
sorry

-- Theorem for the tangent lines
theorem tangent_lines :
  (∀ (x y : ℝ), circle_C x y → (x - 3)^2 + (y - 1)^2 ≥ 4) ∧
  (∃ (x y : ℝ), circle_C x y ∧ 3*x - 4*y - 5 = 0) ∧
  (∃ (y : ℝ), circle_C 3 y) ∧
  (∀ (x y : ℝ), circle_C x y → (3*x - 4*y - 5 = 0 ∨ x = 3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_lines_l1284_128439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_exists_l1284_128480

noncomputable def line (x : ℝ) : ℝ := (2 * x - 4) / 3

noncomputable def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_parameterization_exists :
  ∃ (v d : ℝ × ℝ),
    (∀ x ≥ 2, ∃ t,
      parameterization v d t = (x, line x) ∧
      distance (parameterization v d t) (2, 0) = t) →
    d = (3 / Real.sqrt 13, 2 / Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_exists_l1284_128480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_perpendicular_planes_l1284_128498

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Checks if a line is in a plane -/
def Line.inPlane (l : Line) (p : Plane) : Prop := sorry

/-- Checks if two planes are perpendicular -/
def Plane.perpendicular (p1 p2 : Plane) : Prop := sorry

theorem multiple_perpendicular_planes (P : Plane) (L : Line) 
  (h : L.inPlane P) : 
  ∃ (Q R : Plane), Q ≠ R ∧ 
    Plane.perpendicular P Q ∧ 
    Plane.perpendicular P R ∧ 
    L.inPlane Q ∧ 
    L.inPlane R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_perpendicular_planes_l1284_128498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l1284_128444

/-- A piecewise function f(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then 5 * x^2 + 4 else b * x + 1

/-- Theorem stating that if f is continuous, then b = 23/2 -/
theorem continuous_piecewise_function (b : ℝ) :
  Continuous (f b) → b = 23/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l1284_128444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1284_128486

theorem problem_solution : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((876954 * 876954 - 432196 * 432196) / (876954^2 - 432196^2)) * (Real.sqrt 2839 + (47108 : ℝ) ^ (1/3 : ℝ)) - 89.39| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1284_128486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_sequence_l1284_128440

/-- Represents a line in a 2D coordinate system -/
structure Line where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- Defines the sequence of lines -/
def line_sequence : ℕ → Line :=
  sorry  -- Implementation details omitted for brevity

/-- The point (1, 1) lies on all lines -/
axiom point_on_lines (n : ℕ) : 
  1 - (line_sequence n).slope = (line_sequence n).y_intercept - (line_sequence n).slope

/-- The slope of the (n+1)th line is equal to the difference between 
    the x-intercept and y-intercept of the nth line -/
axiom slope_relation (n : ℕ) : 
  (line_sequence (n + 1)).slope = (line_sequence n).x_intercept - (line_sequence n).y_intercept

/-- The product of the slopes of consecutive lines is non-negative -/
axiom slope_product_nonneg (n : ℕ) : 
  (line_sequence n).slope * (line_sequence (n + 1)).slope ≥ 0

/-- There does not exist an infinite sequence of lines satisfying the given conditions -/
theorem no_infinite_sequence : False := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_sequence_l1284_128440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_ten_like_apple_and_chocolate_not_blueberry_l1284_128478

/-- Represents the number of students who like a specific dessert or combination of desserts -/
structure DessertPreference where
  count : ℕ

/-- The total number of students in the class -/
def total_students : DessertPreference := ⟨50⟩

/-- The number of students who like apple pie -/
def like_apple_pie : DessertPreference := ⟨25⟩

/-- The number of students who like chocolate cake -/
def like_chocolate_cake : DessertPreference := ⟨20⟩

/-- The number of students who like blueberry tart -/
def like_blueberry_tart : DessertPreference := ⟨5⟩

/-- The number of students who don't like any of the desserts -/
def dislike_all : DessertPreference := ⟨15⟩

/-- Helper function to convert DessertPreference to ℕ -/
def to_nat (dp : DessertPreference) : ℕ := dp.count

/-- Theorem stating that at least 10 students like both apple pie and chocolate cake but not blueberry tart -/
theorem at_least_ten_like_apple_and_chocolate_not_blueberry :
  ∃ (n : ℕ), n ≥ 10 ∧ 
  n ≤ to_nat like_apple_pie ∧ 
  n ≤ to_nat like_chocolate_cake ∧
  n ≤ to_nat total_students - to_nat dislike_all - to_nat like_blueberry_tart :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_ten_like_apple_and_chocolate_not_blueberry_l1284_128478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_abcd_l1284_128458

-- Define the functions as axioms instead of recursive definitions
axiom a : ℝ
axiom b : ℝ
axiom c : ℝ
axiom d : ℝ

-- Define the properties of a, b, c, and d
axiom a_def : a = Real.sqrt (4 - Real.sqrt (5 - a))
axiom b_def : b = Real.sqrt (4 + Real.sqrt (5 - b))
axiom c_def : c = Real.sqrt (4 - Real.sqrt (5 + c))
axiom d_def : d = Real.sqrt (4 + Real.sqrt (5 + d))

theorem product_abcd : a * b * c * d = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_abcd_l1284_128458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_sum_is_twelve_l1284_128495

-- Define the card types
inductive Color
| Red
| Blue
deriving Inhabited

-- Define the card structure
structure Card where
  color : Color
  number : Nat
deriving Inhabited

-- Define the deck of cards
def redCards : List Card := [
  ⟨Color.Red, 1⟩, ⟨Color.Red, 2⟩, ⟨Color.Red, 3⟩, ⟨Color.Red, 4⟩, ⟨Color.Red, 5⟩
]

def blueCards : List Card := [
  ⟨Color.Blue, 3⟩, ⟨Color.Blue, 4⟩, ⟨Color.Blue, 5⟩, ⟨Color.Blue, 6⟩
]

-- Define the condition for valid arrangement
def isValidArrangement (arrangement : List Card) : Prop :=
  arrangement.length = 9 ∧
  (∀ i, i % 2 = 0 → (arrangement[i]!).color = Color.Red) ∧
  (∀ i, i % 2 = 1 → (arrangement[i]!).color = Color.Blue) ∧
  (∀ i, i % 2 = 0 ∧ i + 1 < arrangement.length →
    (arrangement[i + 1]!).number % (arrangement[i]!).number = 0)

-- Theorem statement
theorem middle_three_sum_is_twelve :
  ∀ arrangement : List Card,
    isValidArrangement arrangement →
    (arrangement[3]!).number + (arrangement[4]!).number + (arrangement[5]!).number = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_sum_is_twelve_l1284_128495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_and_root_condition_l1284_128428

open Real Set

noncomputable def f (x t : ℝ) : ℝ := 
  (sin (2 * x - π / 4))^2 - 2 * t * sin (2 * x - π / 4) + t^2 - 6 * t + 1

def X : Set ℝ := Icc (π / 24) (π / 2)

def T : Set ℝ := Icc (-1 / 2) 1

theorem minimum_value_and_root_condition :
  ∃ (g : ℝ → ℝ),
    (∀ t ∈ T, g t = -6 * t + 1) ∧
    (∀ t ∈ T, ∀ x ∈ X, g t ≤ f x t) ∧
    (∀ k : ℝ, (∃! t, t ∈ T ∧ g t = k * t) ↔ (k ≤ -8 ∨ k ≥ -5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_and_root_condition_l1284_128428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_theorem_l1284_128451

structure Pentagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ

def specificPentagon : Pentagon :=
  { v1 := (0, 0)
  , v2 := (2, 0)
  , v3 := (3, 2)
  , v4 := (1, 3)
  , v5 := (0, 2)
  }

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (p : Pentagon) : ℝ :=
  distance p.v1 p.v2 + distance p.v2 p.v3 + distance p.v3 p.v4 + 
  distance p.v4 p.v5 + distance p.v5 p.1

theorem pentagon_perimeter_theorem :
  perimeter specificPentagon = 4 + 2 * Real.sqrt 5 + Real.sqrt 2 ∧
  ∃ (a b c : ℕ), 
    perimeter specificPentagon = a + b * Real.sqrt c ∧
    a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_theorem_l1284_128451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_Q_zeros_l1284_128479

noncomputable section

/-- The polynomial Q(z) -/
def Q (z : ℂ) : ℂ := z^8 + (8 * Real.sqrt 2 + 12) * z^4 - (8 * Real.sqrt 2 + 14)

/-- The set of zeros of Q(z) -/
def zeros : Set ℂ := {z : ℂ | Q z = 0}

/-- An 8-sided polygon in the complex plane -/
structure Octagon where
  vertices : Finset ℂ
  card_eq : vertices.card = 8

/-- The perimeter of an octagon -/
def perimeter (o : Octagon) : ℝ := 
  (o.vertices.toList.map Complex.abs).sum

theorem min_perimeter_of_Q_zeros : 
  ∃ (o : Octagon), o.vertices = zeros ∧ 
    ∀ (o' : Octagon), o'.vertices = zeros → perimeter o ≤ perimeter o' ∧
    perimeter o = 8 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_Q_zeros_l1284_128479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_l1284_128488

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ := (x - 1/x)^n

def max_coeff_term : ℕ := 5

theorem coeff_x_squared : 
  (max_coeff_term = 5) → 
  (∃ (k : ℤ), ∃ (n : ℕ), binomial_expansion x n = k * x^2 + (binomial_expansion x n - k * x^2)) →
  k = -56 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_l1284_128488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_load_mass_range_l1284_128402

noncomputable section

/-- The range of the winch drum radius in meters -/
def radius_range : Set ℝ := {r | 0.20 ≤ r ∧ r ≤ 0.30}

/-- The constant force in Newton-meters -/
def constant_force : ℝ := 2000

/-- The gravitational acceleration in m/s² -/
def gravity : ℝ := 10

/-- The function to calculate the maximum load mass -/
noncomputable def max_load_mass (r : ℝ) : ℝ := constant_force / (gravity * r)

/-- The lower bound of the maximum load mass range -/
def lower_bound : ℝ := 640

/-- The upper bound of the maximum load mass range -/
def upper_bound : ℝ := 960

theorem max_load_mass_range :
  ∀ r ∈ radius_range, lower_bound ≤ max_load_mass r ∧ max_load_mass r ≤ upper_bound := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_load_mass_range_l1284_128402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_b_l1284_128420

/-- The weights of four friends satisfy certain conditions -/
structure FriendWeights where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  avg_all : (a + b + c + d) / 4 = 45
  avg_ab : (a + b) / 2 = 42
  avg_bc : (b + c) / 2 = 43
  ratio_da : d = (3/4) * a

/-- The weight of friend b is approximately 29.43 kg -/
theorem weight_of_b (w : FriendWeights) : ∃ ε > 0, |w.b - 29.43| < ε := by
  sorry

#check weight_of_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_b_l1284_128420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_prob_l1284_128460

noncomputable section

/-- The probability that a ball lands in bin k -/
def prob_in_bin (k : ℕ+) : ℝ := (3 : ℝ) ^ (-(k : ℝ))

/-- The probability that both balls land in the same bin k -/
def prob_same_bin (k : ℕ+) : ℝ := prob_in_bin k * prob_in_bin k

/-- The sum of probabilities of both balls landing in the same bin for all bins -/
noncomputable def prob_same_bin_total : ℝ := ∑' k, prob_same_bin k

/-- The probability that the blue ball is in a higher-numbered bin than the yellow ball -/
noncomputable def prob_blue_higher : ℝ := (1 - prob_same_bin_total) / 2

theorem blue_ball_higher_prob :
  prob_blue_higher = 7/16 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_prob_l1284_128460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_divisor_count_l1284_128443

/-- Given n = 2^33 * 3^21 * 5^7, this theorem proves that the number of positive integer 
    divisors of n^2 that are less than n but do not divide n is 15473. -/
theorem special_divisor_count (n : ℕ) (hn : n = 2^33 * 3^21 * 5^7) : 
  (Finset.filter (λ d ↦ d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 15473 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_divisor_count_l1284_128443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_division_ratio_l1284_128423

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the division of a cone into a smaller cone and a frustum -/
structure ConeDivision where
  cone : Cone
  smallerConeRadius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ :=
  (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Calculates the surface area of a cone -/
noncomputable def coneSurfaceArea (c : Cone) : ℝ :=
  Real.pi * c.baseRadius^2 + Real.pi * c.baseRadius * Real.sqrt (c.baseRadius^2 + c.height^2)

/-- Calculates the volume of the smaller cone in a division -/
noncomputable def smallerConeVolume (d : ConeDivision) : ℝ :=
  (1/3) * Real.pi * d.smallerConeRadius^2 * (d.smallerConeRadius * (4/3))

/-- Calculates the painted surface area of the smaller cone in a division -/
noncomputable def smallerConeSurfaceArea (d : ConeDivision) : ℝ :=
  (5/3) * Real.pi * d.smallerConeRadius^2

/-- Calculates the volume of the frustum in a division -/
noncomputable def frustumVolume (d : ConeDivision) : ℝ :=
  coneVolume d.cone - smallerConeVolume d

/-- Calculates the painted surface area of the frustum in a division -/
noncomputable def frustumSurfaceArea (d : ConeDivision) : ℝ :=
  coneSurfaceArea d.cone - smallerConeSurfaceArea d

/-- The main theorem -/
theorem cone_division_ratio (d : ConeDivision) :
  d.cone.height = 4 ∧ d.cone.baseRadius = 3 →
  (smallerConeSurfaceArea d) / (frustumSurfaceArea d) = 
  (smallerConeVolume d) / (frustumVolume d) →
  (smallerConeSurfaceArea d) / (frustumSurfaceArea d) = 125 / 387 := by
  sorry

#check cone_division_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_division_ratio_l1284_128423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_percentage_l1284_128474

noncomputable def net_salary : ℚ := 3600

noncomputable def discretionary_income : ℚ := net_salary / 5

def savings_percentage : ℚ := 20
def social_percentage : ℚ := 35
def gifts_amount : ℚ := 108

theorem vacation_fund_percentage :
  let vacation_percentage := 100 - (savings_percentage + social_percentage + (gifts_amount / discretionary_income) * 100)
  vacation_percentage = 30 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_percentage_l1284_128474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l1284_128484

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -1 then 5 * x + 10 else 3 * x - 9

theorem g_equation_solution :
  ∀ x : ℝ, g x = 6 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l1284_128484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_20_over_3_l1284_128482

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2
  else x + 1

-- State the theorem
theorem integral_f_equals_20_over_3 :
  ∫ x in (-2)..2, f x = 20/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_20_over_3_l1284_128482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l1284_128401

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.log (1 + x)
  else Real.log (1 / (1 - x))

-- Theorem statement
theorem f_increasing_and_odd : 
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l1284_128401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1284_128422

open Real

theorem problem_1 (θ : ℝ) (h : tan θ = 3) :
  (sin θ + cos θ) / (2 * sin θ + cos θ) = 4/7 := by sorry

theorem problem_2 (α β : ℝ) 
  (h1 : 0 < β) (h2 : β < π/2) (h3 : π/2 < α) (h4 : α < π)
  (h5 : cos (α - β/2) = -1/9) (h6 : sin (α/2 - β) = 2/3) :
  cos ((α + β)/2) = 7 * sqrt 5 / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1284_128422
