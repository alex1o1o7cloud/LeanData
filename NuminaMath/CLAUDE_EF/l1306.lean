import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1306_130666

/-- IsEllipse predicate -/
def IsEllipse (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Foci of an ellipse -/
def Foci (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (3,5) and (d,5) has d = 4/3 -/
theorem ellipse_foci_distance (d : ℝ) : 
  (∃ (e : Set (ℝ × ℝ)), 
    -- e is an ellipse
    IsEllipse e ∧ 
    -- e is in the first quadrant
    (∀ (x y : ℝ), (x, y) ∈ e → x ≥ 0 ∧ y ≥ 0) ∧
    -- e is tangent to x-axis
    (∃ (x : ℝ), (x, 0) ∈ e) ∧
    -- e is tangent to y-axis
    (∃ (y : ℝ), (0, y) ∈ e) ∧
    -- one focus is at (3,5)
    (3, 5) ∈ Foci e ∧
    -- other focus is at (d,5)
    (d, 5) ∈ Foci e) →
  d = 4/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1306_130666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_invalid_net_l1306_130625

/-- Represents a face of a cuboid -/
inductive Face
| Front
| Back
| Left
| Right
| Top
| Bottom

/-- Represents a net of a cuboid -/
structure CuboidNet where
  faces : List Face
  connections : List (Face × Face)

/-- Represents a path on a cuboid -/
structure CuboidPath where
  edges : List (Face × Face)

/-- Checks if a path is closed on a given net -/
def is_closed_path (net : CuboidNet) (path : CuboidPath) : Prop :=
  ∀ e ∈ path.edges, e ∈ net.connections ∨ (e.2, e.1) ∈ net.connections

/-- Theorem: There exists a net of a cuboid that does not allow for a closed path -/
theorem exists_invalid_net : ∃ (net : CuboidNet), ∀ (path : CuboidPath), ¬(is_closed_path net path) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_invalid_net_l1306_130625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_sequence_l1306_130623

theorem divisible_by_three_sequence (start : Nat) (count : Nat) (last : Nat) : 
  start = 10 → count = 13 → 
  (∃ seq : List Nat, seq.length = count ∧ 
    (∀ n ∈ seq, n ≥ start ∧ n ≤ last ∧ n % 3 = 0) ∧
    seq.getLast? = some last) →
  last = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_sequence_l1306_130623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l1306_130644

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the main theorem
theorem odd_function_sum (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 2)))
  (h_f1 : f 1 = a) :
  (Finset.range 505).sum (fun i ↦ f (2 * i + 1)) = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l1306_130644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1306_130673

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (5, -3) to the line x+2=0 is 7 -/
theorem distance_point_to_line_example : distance_point_to_line 5 (-3) 1 0 (-2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1306_130673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_congruence_l1306_130688

theorem square_sum_congruence (S : Finset ℕ) (h : S.card = 11) 
  (h_squares : ∀ n ∈ S, ∃ m : ℕ, n = m^2) : 
  ∃ a b c d e f, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
  (a + b + c) ≡ (d + e + f) [ZMOD 12] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_congruence_l1306_130688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_soda_problem_l1306_130698

/-- Represents the problem of calculating Liam's final amount of money after selling soda bottles -/
theorem liam_soda_problem (total_bottles : ℕ) (initial_price : ℚ) (sold_bottles : ℕ) (extra_money : ℚ) :
  total_bottles = 50 →
  initial_price = 1 →
  sold_bottles = 40 →
  extra_money = 10 →
  ∃ (selling_price : ℚ),
    selling_price > initial_price ∧
    selling_price * (sold_bottles : ℚ) = initial_price * (total_bottles : ℚ) + extra_money ∧
    selling_price * (total_bottles : ℚ) = 75 :=
by
  intros h_total h_init h_sold h_extra
  -- We use rationals (ℚ) instead of reals (ℝ) for exact arithmetic
  let selling_price : ℚ := 3/2 -- £1.50
  use selling_price
  constructor
  · -- Prove selling_price > initial_price
    rw [h_init]
    norm_num
  constructor
  · -- Prove selling_price * sold_bottles = initial_price * total_bottles + extra_money
    rw [h_total, h_init, h_sold, h_extra]
    norm_num
  · -- Prove selling_price * total_bottles = 75
    rw [h_total]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_soda_problem_l1306_130698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_C_to_D_l1306_130697

/-- The volume of a cone with radius r and height h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Cone C with radius 10 and height 20 -/
def cone_C : ℝ × ℝ := (10, 20)

/-- Cone D with radius 20 and height 10 -/
def cone_D : ℝ × ℝ := (20, 10)

theorem volume_ratio_C_to_D :
  cone_volume cone_C.1 cone_C.2 / cone_volume cone_D.1 cone_D.2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_C_to_D_l1306_130697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_square_roots_l1306_130612

theorem max_value_of_sum_square_roots (x y z : ℝ) 
  (hx : x ∈ Set.Icc (0 : ℝ) (1 : ℝ)) (hy : y ∈ Set.Icc (0 : ℝ) (1 : ℝ)) (hz : z ∈ Set.Icc (0 : ℝ) (1 : ℝ)) : 
  Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x)) ≤ Real.sqrt 2 + 1 ∧ 
  ∃ (a b c : ℝ), a ∈ Set.Icc (0 : ℝ) (1 : ℝ) ∧ b ∈ Set.Icc (0 : ℝ) (1 : ℝ) ∧ c ∈ Set.Icc (0 : ℝ) (1 : ℝ) ∧ 
    Real.sqrt (abs (a - b)) + Real.sqrt (abs (b - c)) + Real.sqrt (abs (c - a)) = Real.sqrt 2 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_square_roots_l1306_130612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1306_130653

/-- Given a line l that passes through the point (-2, 1) and is perpendicular to the line 2x - 3y + 5 = 0,
    prove that the equation of l is 3x + 2y + 4 = 0. -/
theorem perpendicular_line_equation :
  let l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (t : ℝ), p = (-2 + t * 3, 1 - t * 2)}
  let given_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 + 5 = 0}
  ((-2, 1) ∈ l) →
  (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (p.1 - q.1) * (2 / 3) + (p.2 - q.2) = 0) →
  (∀ p : ℝ × ℝ, p ∈ l → 3 * p.1 + 2 * p.2 + 4 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1306_130653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1306_130642

theorem cos_double_angle_special_case (α : Real) (h : Real.sin α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1306_130642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_inside_circle_probability_l1306_130638

/-- Circle C with radius 1 and center at the origin -/
def C : Set (ℝ × ℝ) := {p | Real.sqrt (p.1^2 + p.2^2) = 1}

/-- Point P on the circumference of C -/
def P : C := sorry

/-- Point Q in the interior of C -/
def Q : {q : ℝ × ℝ | Real.sqrt (q.1^2 + q.2^2) < 1} := sorry

/-- Rectangle with diagonal PQ and sides parallel to x and y axes -/
def rectangle (P : C) (Q : {q : ℝ × ℝ | Real.sqrt (q.1^2 + q.2^2) < 1}) : Set (ℝ × ℝ) := sorry

/-- The probability that the rectangle lies entirely inside or on C -/
noncomputable def probability : ℝ := sorry

/-- Theorem stating the probability is 2/π -/
theorem rectangle_inside_circle_probability :
  probability = 2 / Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_inside_circle_probability_l1306_130638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1306_130618

theorem log_equation_solution (r : ℝ) : 
  r = (Real.rpow 7 (-2/3) + 2) / 5 → Real.logb 49 (5*r - 2) = -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1306_130618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_solution_l1306_130679

/-- Represents a type of tank with its capacity, cost, and available quantity. -/
structure TankType where
  capacity : Nat
  cost : Nat
  available : Nat

/-- The problem setup for oil storage optimization. -/
structure OilStorageProblem where
  totalOil : Nat
  tankTypes : List TankType
  maxTanks : Nat

/-- A solution to the oil storage problem. -/
structure Solution where
  cost : Nat
  tankCounts : List Nat

/-- Checks if a solution is valid for a given problem. -/
def isSolutionValid (problem : OilStorageProblem) (solution : Solution) : Prop :=
  let totalCapacity := List.sum (List.zipWith (· * ·) (problem.tankTypes.map (·.capacity)) solution.tankCounts)
  let totalTanks := List.sum solution.tankCounts
  totalCapacity ≥ problem.totalOil ∧
  totalTanks ≤ problem.maxTanks ∧
  List.length solution.tankCounts = List.length problem.tankTypes ∧
  List.all₂ (· ≤ ·) solution.tankCounts (problem.tankTypes.map (·.available))

/-- Calculates the cost of a solution. -/
def solutionCost (problem : OilStorageProblem) (solution : Solution) : Nat :=
  List.sum (List.zipWith (· * ·) (problem.tankTypes.map (·.cost)) solution.tankCounts)

/-- The main theorem stating that the given solution is optimal. -/
theorem optimal_solution (problem : OilStorageProblem) (solution : Solution) :
  problem.totalOil = 728 ∧
  problem.tankTypes = [
    { capacity := 50, cost := 100, available := 10 },
    { capacity := 35, cost := 80, available := 15 },
    { capacity := 20, cost := 60, available := 20 }
  ] ∧
  problem.maxTanks = 20 ∧
  solution.tankCounts = [9, 8, 0] ∧
  isSolutionValid problem solution ∧
  solutionCost problem solution = 1540 ∧
  (∀ (otherSolution : Solution), 
    isSolutionValid problem otherSolution → 
    solutionCost problem otherSolution ≥ solutionCost problem solution) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_solution_l1306_130679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1306_130677

noncomputable def f (x : ℝ) : ℝ := 3 / (x - 2)

theorem function_properties :
  (∀ x₁ x₂ : ℝ, 2 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ m n : ℝ, 2 < m ∧ m < n ∧ (∀ x : ℝ, m ≤ x ∧ x ≤ n → 1 ≤ f x ∧ f x ≤ 3) → m + n = 8) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1306_130677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l1306_130628

-- Define the circle C
noncomputable def circle_C (α : ℝ) : ℝ × ℝ := (2 * (1 + Real.cos α), 2 * Real.sin α)

-- Define the line l
def line_l (θ₀ : ℝ) (θ : ℝ) : Prop := θ = θ₀

-- Theorem statement
theorem circle_and_line_intersection 
  (θ₀ : ℝ) 
  (h1 : 0 < θ₀ ∧ θ₀ < Real.pi / 2) 
  (h2 : Real.tan θ₀ = Real.sqrt 7 / 3) :
  (∃ ρ θ, ρ = 4 * Real.cos θ ∧ 
    (ρ * Real.cos θ, ρ * Real.sin θ) = circle_C (Real.arccos ((ρ / 2) - 1))) ∧
  (∃ ρ, ρ = 3 ∧ line_l θ₀ θ₀ ∧ 
    (ρ * Real.cos θ₀, ρ * Real.sin θ₀) = circle_C (Real.arccos ((ρ / 2) - 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l1306_130628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_one_l1306_130617

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * 2^x else 2^(-x)

-- State the theorem
theorem function_composition_equals_one (a : ℝ) :
  (f a (f a (-1)) = 1) → (a = 1/4) := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_one_l1306_130617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1306_130658

-- Define the function f as noncomputable
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

-- State the theorem
theorem f_property (a b : ℝ) : f a b (1/2015) = 4 → f a b 2015 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1306_130658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l1306_130629

/-- Circle C with equation x^2 + y^2 = 12 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 12}

/-- Line l with equation 4x + 3y = 25 -/
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 25}

/-- The center of circle C -/
def center : ℝ × ℝ := (0, 0)

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (|A * p.1 + B * p.2 + C|) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the center of C to line l is 5 -/
theorem distance_center_to_line :
  distanceToLine center 4 3 (-25) = 5 := by
  sorry

#check distance_center_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l1306_130629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l1306_130619

-- Define the cosine functions
noncomputable def f (x : ℝ) := Real.cos (2 * x - 1)
noncomputable def g (x : ℝ) := Real.cos (2 * x + 1)

-- Define the left shift transformation
def shift_left (h : ℝ → ℝ) (d : ℝ) (x : ℝ) : ℝ := h (x + d)

-- Theorem statement
theorem cos_graph_shift :
  ∀ x : ℝ, shift_left f 1 x = g x := by
  intro x
  simp [shift_left, f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l1306_130619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_even_function_l1306_130626

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3 * x
  else Real.log x - 3 * x

theorem tangent_line_of_even_function :
  (∀ x, f (-x) = f x) →  -- f is even
  (∀ x < 0, f x = Real.log (-x) + 3 * x) →  -- definition for x < 0
  ∃ a b : ℝ, a * 1 + b * f 1 + 1 = 0 ∧
             ∀ x y, y = f x → (a * x + b * y + 1 = 0 ↔ 2 * x + y + 1 = 0) :=
by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_even_function_l1306_130626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_value_l1306_130647

/-- Given a function y = sin(2x) translated left by φ units, 
    prove that the minimum value of φ is π/12, where φ > 0 and 
    the resulting graph is symmetric about the line x = π/6 -/
theorem min_translation_value (φ : ℝ) : 
  (φ > 0) →
  (∀ x, Real.sin (2*x + 2*φ) = Real.sin (2*(π/3 - x))) →
  φ ≥ π/12 ∧ ∃ φ₀, φ₀ > 0 ∧ φ₀ = π/12 ∧ 
    (∀ x, Real.sin (2*x + 2*φ₀) = Real.sin (2*(π/3 - x))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_value_l1306_130647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_triangle_hypotenuse_l1306_130674

noncomputable def triangle_hypotenuse (n : ℕ) : ℝ :=
  Real.sqrt (n : ℝ)

theorem ninth_triangle_hypotenuse : triangle_hypotenuse 9 = 3 := by
  unfold triangle_hypotenuse
  simp [Real.sqrt_eq_iff_sq_eq]
  norm_num

#check ninth_triangle_hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_triangle_hypotenuse_l1306_130674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_l1306_130650

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C
  angle_sum : A + B + C = Real.pi
  side_angle : a * Real.sin B = b * Real.sin A  -- Law of sines

-- Theorem statement
theorem cos_C_value (t : Triangle) 
  (h : t.a * Real.cos t.B + t.b * Real.cos t.A = 3 * t.c * Real.cos t.C) : 
  Real.cos t.C = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_l1306_130650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_october_temp_l1306_130665

/-- Temperature function -/
noncomputable def temp (a A x : ℝ) : ℝ := a + A * Real.cos (Real.pi / 6 * (x - 6))

theorem october_temp (a A : ℝ) :
  temp a A 6 = 28 →
  temp a A 12 = 18 →
  temp a A 10 = 20.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_october_temp_l1306_130665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_game_theorem_l1306_130691

/-- Represents the state of the card game -/
def GameState (n : ℕ) := Fin (n + 1) → Fin (n + 1)

/-- Defines a single move in the game -/
def move (n : ℕ) (state : GameState n) : Option (GameState n) :=
  sorry

/-- Checks if the game is in a winning state -/
def is_winning_state (n : ℕ) (state : GameState n) : Prop :=
  ∀ i : Fin (n + 1), state i = i

/-- The maximum number of moves required to end the game -/
def max_moves (n : ℕ) : ℕ := 2^n - 1

/-- There exists a unique initial configuration requiring the maximum number of moves -/
def unique_max_config (n : ℕ) (init_state : GameState n) : Prop :=
  (∃ (seq : Fin (max_moves n + 1) → GameState n), 
    seq 0 = init_state ∧ 
    (∀ i : Fin (max_moves n), move n (seq i) = some (seq (i + 1))) ∧
    is_winning_state n (seq (max_moves n))) ∧
  (∀ other_init : GameState n, other_init ≠ init_state → 
    ¬∃ (seq : Fin (max_moves n + 1) → GameState n),
      seq 0 = other_init ∧ 
      (∀ i : Fin (max_moves n), move n (seq i) = some (seq (i + 1))) ∧
      is_winning_state n (seq (max_moves n)))

theorem card_game_theorem (n : ℕ) :
  (∀ init_state : GameState n, ∃ k : ℕ, k ≤ max_moves n ∧ 
    (∃ (seq : Fin (k + 1) → GameState n),
      seq 0 = init_state ∧
      (∀ i : Fin k, move n (seq i) = some (seq (i + 1))) ∧
      is_winning_state n (seq k))) ∧
  (∃! init_state : GameState n, unique_max_config n init_state) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_game_theorem_l1306_130691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l1306_130631

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 2 / (3 * x + c)

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := (3 - 6 * x) / x

-- Theorem statement
theorem inverse_function_condition (c : ℝ) :
  (∀ x, f c (f_inv x) = x) → c = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l1306_130631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l1306_130614

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 2 * Real.sqrt 5 / 5) : Real.cos (2 * α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l1306_130614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1306_130630

def A : Set ℝ := {1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1306_130630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_and_roots_l1306_130615

/-- The function f(x) = ln x - ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

/-- The function g(x) = ln x - 3x + x^2 -/
noncomputable def g (x : ℝ) : ℝ := Real.log x - 3 * x + x^2

theorem tangent_line_parallel_and_roots (a m : ℝ) : 
  (f_deriv a 2 = -1/2) ∧ 
  (∃ x y : ℝ, 1/2 ≤ x ∧ x < y ∧ y ≤ 2 ∧ 
    f 1 x + m = 2 * x - x^2 ∧ 
    f 1 y + m = 2 * y - y^2 ∧
    ∀ z, 1/2 ≤ z ∧ z ≤ 2 ∧ f 1 z + m = 2 * z - z^2 → z = x ∨ z = y) →
  a = 1 ∧ Real.log 2 + 5/4 ≤ m ∧ m < 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_and_roots_l1306_130615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_divisible_l1306_130678

theorem polynomial_not_divisible (A : ℤ) (n m : ℕ) :
  ¬∃ (P : Polynomial ℤ), (3 * X^(2*n) + A * X^n + 2) = P * (2 * X^(2*m) + A * X^m + 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_divisible_l1306_130678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_subgraph_exists_l1306_130613

/-- A complete graph with 18 vertices where each edge is colored in one of two colors -/
structure TwoColorGraph :=
  (vertices : Finset (Fin 18))
  (edge_color : Fin 18 → Fin 18 → Option (Fin 2))
  (complete : ∀ i j, i ≠ j → edge_color i j ≠ none)

/-- A complete subgraph of 4 vertices -/
def CompleteSubgraph4 (g : TwoColorGraph) (vs : Finset (Fin 18)) : Prop :=
  vs.card = 4 ∧ ∀ i j, i ∈ vs → j ∈ vs → i ≠ j → g.edge_color i j ≠ none

/-- A monochromatic complete subgraph of 4 vertices -/
def MonochromaticSubgraph4 (g : TwoColorGraph) (vs : Finset (Fin 18)) (c : Fin 2) : Prop :=
  CompleteSubgraph4 g vs ∧ ∀ i j, i ∈ vs → j ∈ vs → i ≠ j → g.edge_color i j = some c

/-- Theorem: In a complete graph with 18 vertices where each edge is colored in one of two colors,
    there exists a monochromatic complete subgraph on 4 vertices -/
theorem monochromatic_subgraph_exists (g : TwoColorGraph) :
  ∃ (vs : Finset (Fin 18)) (c : Fin 2), MonochromaticSubgraph4 g vs c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_subgraph_exists_l1306_130613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_on_interval_extreme_values_on_specific_interval_l1306_130620

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 6

theorem extreme_values_on_interval (a b : ℝ) (h : a ≤ b) :
  let max_value := (34:ℝ)/3
  let min_value := (2:ℝ)/3
  (∀ x ∈ Set.Icc a b, f x ≤ max_value) ∧
  (∃ x ∈ Set.Icc a b, f x = max_value) ∧
  (∀ x ∈ Set.Icc a b, min_value ≤ f x) ∧
  (∃ x ∈ Set.Icc a b, f x = min_value) :=
by
  sorry

theorem extreme_values_on_specific_interval :
  let max_value := (34:ℝ)/3
  let min_value := (2:ℝ)/3
  (∀ x ∈ Set.Icc (-3) 4, f x ≤ max_value) ∧
  (∃ x ∈ Set.Icc (-3) 4, f x = max_value) ∧
  (∀ x ∈ Set.Icc (-3) 4, min_value ≤ f x) ∧
  (∃ x ∈ Set.Icc (-3) 4, f x = min_value) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_on_interval_extreme_values_on_specific_interval_l1306_130620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_savings_rate_theorem_savings_rate_theorem_l1306_130662

/-- Represents the financial situation of a person over two years --/
structure FinancialSituation where
  firstYearIncome : ℝ
  firstYearSavingsRate : ℝ
  secondYearIncomeIncrease : ℝ
  secondYearSavingsIncrease : ℝ
  secondYearTaxRate : ℝ
  inflationRate : ℝ

/-- Calculates the savings rate in the first year --/
def calculateSavingsRate (fs : FinancialSituation) : ℝ :=
  fs.firstYearSavingsRate

/-- Theorem stating the savings rate in the first year --/
theorem first_year_savings_rate_theorem (fs : FinancialSituation) 
  (h1 : fs.secondYearIncomeIncrease = 0.35)
  (h2 : fs.secondYearSavingsIncrease = 1.0)
  (h3 : fs.secondYearTaxRate = 0.10)
  (h4 : fs.inflationRate = 0.05)
  (h5 : let firstYearExpenditure := fs.firstYearIncome * (1 - fs.firstYearSavingsRate)
        let secondYearIncome := fs.firstYearIncome * (1 + fs.secondYearIncomeIncrease)
        let secondYearSavings := fs.firstYearIncome * fs.firstYearSavingsRate * (1 + fs.secondYearSavingsIncrease)
        let secondYearTax := secondYearIncome * fs.secondYearTaxRate
        let secondYearExpenditure := secondYearIncome - secondYearSavings + secondYearTax
        firstYearExpenditure + secondYearExpenditure / (1 + fs.inflationRate) = 2 * firstYearExpenditure) :
  ∃ ε > 0, |calculateSavingsRate fs - 0.4577| < ε := by
  sorry

/-- Main theorem combining all conditions and proving the savings rate --/
theorem savings_rate_theorem (fs : FinancialSituation) : 
  fs.secondYearIncomeIncrease = 0.35 →
  fs.secondYearSavingsIncrease = 1.0 →
  fs.secondYearTaxRate = 0.10 →
  fs.inflationRate = 0.05 →
  (let firstYearExpenditure := fs.firstYearIncome * (1 - fs.firstYearSavingsRate)
   let secondYearIncome := fs.firstYearIncome * (1 + fs.secondYearIncomeIncrease)
   let secondYearSavings := fs.firstYearIncome * fs.firstYearSavingsRate * (1 + fs.secondYearSavingsIncrease)
   let secondYearTax := secondYearIncome * fs.secondYearTaxRate
   let secondYearExpenditure := secondYearIncome - secondYearSavings + secondYearTax
   firstYearExpenditure + secondYearExpenditure / (1 + fs.inflationRate) = 2 * firstYearExpenditure) →
  ∃ ε > 0, |calculateSavingsRate fs - 0.4577| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_savings_rate_theorem_savings_rate_theorem_l1306_130662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1306_130672

/-- Represents the simple interest calculation for a loan --/
structure LoanInfo where
  principal : ℚ
  interest_paid : ℚ
  rate : ℚ

/-- Simple interest formula --/
def simple_interest (loan : LoanInfo) : ℚ :=
  loan.principal * loan.rate * loan.rate / 100

theorem interest_rate_calculation (loan : LoanInfo) 
  (h1 : loan.principal = 1500)
  (h2 : loan.interest_paid = 735)
  (h3 : simple_interest loan = loan.interest_paid) : 
  loan.rate = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1306_130672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_formula_l1306_130661

/-- The total time Martha spends on all activities related to her internet problem -/
noncomputable def total_time (x y z w : ℝ) : ℝ :=
  let router_time := x
  let hold_time := y * x
  let yelling_time := z * (y * x)
  let technician_time := w * (z * (y * x))
  let activities_time := router_time + hold_time + yelling_time + technician_time
  let email_time := (1 / 2) * activities_time
  activities_time + email_time

/-- Theorem stating that the total time can be expressed as (3/2)x(1 + y + zy + wzy) -/
theorem total_time_formula (x y z w : ℝ) :
  total_time x y z w = (3 / 2) * x * (1 + y + z * y + w * z * y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_formula_l1306_130661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l1306_130605

/-- Calculate the percentage of loss given cost price and selling price -/
noncomputable def percentageLoss (costPrice sellingPrice : ℝ) : ℝ :=
  ((costPrice - sellingPrice) / costPrice) * 100

/-- Theorem: The percentage of loss for an item with cost price 600 and selling price 480 is 20% -/
theorem loss_percentage_calculation (costPrice sellingPrice : ℝ) 
  (h1 : costPrice = 600) 
  (h2 : sellingPrice = 480) : 
  percentageLoss costPrice sellingPrice = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l1306_130605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_property_l1306_130621

open Real

theorem critical_point_property (m : ℝ) (h_m : 0 < m ∧ m < 1) :
  let f (x : ℝ) := x + log x
  let g (x : ℝ) := 3 - 2 / x
  let F (x : ℝ) := 3 * (x - m / 2) + (m / 2) * g x - 2 * f x
  let x₂ := (1 : ℝ) + sqrt (1 - m)
  ∀ x₁ : ℝ, (deriv F x₁ = 0 ∧ deriv F x₂ = 0 ∧ x₁ < x₂) → F x₂ < x₂ - 1 :=
by
  intros f g F x₂ x₁ h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_property_l1306_130621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_time_at_higher_speed_l1306_130668

/-- Represents a two-segment journey with different speeds -/
structure TwoSpeedJourney where
  speed1 : ℝ
  speed2 : ℝ
  time1 : ℝ
  time2 : ℝ

/-- Calculates the average speed of a two-segment journey -/
noncomputable def averageSpeed (j : TwoSpeedJourney) : ℝ :=
  (j.speed1 * j.time1 + j.speed2 * j.time2) / (j.time1 + j.time2)

/-- Calculates the fraction of time spent at the second speed -/
noncomputable def fractionAtSpeed2 (j : TwoSpeedJourney) : ℝ :=
  j.time2 / (j.time1 + j.time2)

/-- Theorem: If a journey has speeds 5 mph and 15 mph, and the average speed is 10 mph,
    then the fraction of time spent at 15 mph is 1/2 -/
theorem half_time_at_higher_speed (j : TwoSpeedJourney)
    (h1 : j.speed1 = 5)
    (h2 : j.speed2 = 15)
    (h3 : averageSpeed j = 10) :
    fractionAtSpeed2 j = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_time_at_higher_speed_l1306_130668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1306_130606

/-- A parabola is defined by its standard equation y^2 = 6x -/
structure Parabola where
  eq : ∀ x y : ℝ, y^2 = 6*x

/-- The focus of a parabola is a point (p/2, 0) where p is the coefficient of x in the standard equation -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (3/2, 0)

/-- Theorem: The focus of the parabola y^2 = 6x is (3/2, 0) -/
theorem parabola_focus :
  ∀ p : Parabola, focus p = (3/2, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1306_130606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_field_length_l1306_130641

theorem grass_field_length 
  (width : ℝ) 
  (path_width : ℝ) 
  (path_area : ℝ) 
  (h1 : width = 55) 
  (h2 : path_width = 2.8) 
  (h3 : path_area = 1518.72) : 
  ∃ length : ℝ, 
    (length + 2 * path_width) * (width + 2 * path_width) - length * width = path_area ∧ 
    abs (length - 210.6) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_field_length_l1306_130641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stereo_and_tv_spending_l1306_130611

theorem stereo_and_tv_spending (savings : ℚ) : 
  savings > 0 → 
  (1 / 4 * savings + (1 / 4 * savings - 2 / 3 * (1 / 4 * savings))) / savings = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stereo_and_tv_spending_l1306_130611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1306_130607

theorem taimour_paint_time (jamshid_time_ratio : Real) (combined_time : Real) : Real :=
  let taimour_time := 21
  let jamshid_time := jamshid_time_ratio * taimour_time
  let combined_rate := 1 / taimour_time + 1 / jamshid_time
  have h1 : jamshid_time_ratio = 0.5 := by sorry
  have h2 : combined_time = 7 := by sorry
  have h3 : combined_rate * combined_time = 1 := by sorry
  taimour_time

#check taimour_paint_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1306_130607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_is_20_sqrt_3_l1306_130645

/-- A rectangular box with given surface area and edge length sum -/
structure RectangularBox where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area : 2 * (a * b + b * c + c * a) = 150
  edge_sum : 4 * (a + b + c) = 60

/-- The sum of lengths of all interior diagonals of a rectangular box -/
noncomputable def interior_diagonals_sum (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.a^2 + box.b^2 + box.c^2)

theorem interior_diagonals_sum_is_20_sqrt_3 (box : RectangularBox) :
  interior_diagonals_sum box = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_is_20_sqrt_3_l1306_130645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_f_well_defined_f_not_defined_between_neg3_and_3_l1306_130616

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 9) / Real.log (1/3)

-- State the theorem
theorem f_monotone_increasing_interval :
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -3 → f x₁ < f x₂ := by
  sorry

-- Define the domain of the function
def f_domain : Set ℝ := {x | x < -3 ∨ x > 3}

-- State that the function is well-defined on its domain
theorem f_well_defined :
  ∀ x ∈ f_domain, (x^2 - 9 : ℝ) > 0 := by
  sorry

-- State that the function is not defined for x ∈ [-3, 3]
theorem f_not_defined_between_neg3_and_3 :
  ∀ x, -3 ≤ x ∧ x ≤ 3 → ¬(x ∈ f_domain) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_f_well_defined_f_not_defined_between_neg3_and_3_l1306_130616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitudes_sum_l1306_130634

/-- The line equation forming the triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 10 * x + 4 * y = 40

/-- The vertices of the triangle -/
def triangle_vertices : Set (ℝ × ℝ) := {(0, 0), (4, 0), (0, 10)}

/-- The sum of altitudes of the triangle -/
noncomputable def sum_of_altitudes : ℝ := 14 + 40 / Real.sqrt 116

theorem triangle_altitudes_sum :
  ∀ (t : Set (ℝ × ℝ)),
    t = triangle_vertices →
    (∀ (x y : ℝ), (x, y) ∈ t → line_equation x y ∨ (x = 0 ∧ y ≥ 0 ∧ y ≤ 10) ∨ (y = 0 ∧ x ≥ 0 ∧ x ≤ 4)) →
    sum_of_altitudes = 14 + 40 / Real.sqrt 116 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitudes_sum_l1306_130634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_drawn_is_random_variable_l1306_130694

/-- A bag containing balls of two colors -/
structure Bag where
  black : ℕ
  red : ℕ

/-- A draw from the bag -/
structure Draw where
  total : ℕ
  red : ℕ

/-- Definition of a random variable -/
def IsRandomVariable (X : Draw → ℝ) : Prop :=
  ∃ (Ω : Type) (P : Ω → Prop) (f : Ω → Draw), ∀ ω : Ω, P ω → X (f ω) ∈ Set.range X

/-- The number of red balls drawn is a random variable -/
theorem red_balls_drawn_is_random_variable (bag : Bag) (h1 : bag.black = 3) (h2 : bag.red = 7) :
  IsRandomVariable (fun d : Draw => d.red) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_drawn_is_random_variable_l1306_130694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_ten_percent_l1306_130627

-- Define the prices and quantities
def quiche_price : ℚ := 15
def quiche_quantity : ℕ := 2
def croissant_price : ℚ := 3
def croissant_quantity : ℕ := 6
def biscuit_price : ℚ := 2
def biscuit_quantity : ℕ := 6

-- Define the discounted total
def discounted_total : ℚ := 54

-- Calculate the original total
def original_total : ℚ := 
  quiche_price * quiche_quantity + 
  croissant_price * croissant_quantity + 
  biscuit_price * biscuit_quantity

-- Define the discount percentage
noncomputable def discount_percentage : ℚ := 
  (original_total - discounted_total) / original_total * 100

-- Theorem statement
theorem discount_is_ten_percent : discount_percentage = 10 := by
  -- Unfold definitions
  unfold discount_percentage
  unfold original_total
  unfold discounted_total
  -- Simplify the expression
  simp [quiche_price, quiche_quantity, croissant_price, croissant_quantity, biscuit_price, biscuit_quantity]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_ten_percent_l1306_130627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_k_range_l1306_130659

noncomputable section

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  k : ℝ
  m : ℝ
  h : k ≠ 0

theorem ellipse_equation_and_k_range (e : Ellipse) 
  (h1 : e.eccentricity = 1/2)
  (h2 : Point.mk 1 (3/2) ∈ {p : Point | p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1}) 
  (l : Line) 
  (h3 : ∃ (M N : Point), M ≠ N ∧ 
    M ∈ {p : Point | p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1} ∩ {p : Point | p.y = l.k * p.x + l.m} ∧
    N ∈ {p : Point | p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1} ∩ {p : Point | p.y = l.k * p.x + l.m})
  (P : Point)
  (h4 : P.x = 1/5 ∧ P.y = 0)
  (h5 : ∀ (M N : Point), M ≠ N → 
    M ∈ {p : Point | p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1} ∩ {p : Point | p.y = l.k * p.x + l.m} →
    N ∈ {p : Point | p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1} ∩ {p : Point | p.y = l.k * p.x + l.m} →
    (M.x - P.x)^2 + (M.y - P.y)^2 = (N.x - P.x)^2 + (N.y - P.y)^2) :
  e.a^2 = 4 ∧ e.b^2 = 3 ∧ (l.k < -Real.sqrt 7 / 7 ∨ l.k > Real.sqrt 7 / 7) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_k_range_l1306_130659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_has_card15_l1306_130601

-- Define the card type
inductive Card
  | card13 : Card
  | card15 : Card
  | card35 : Card

-- Define the people
inductive Person
  | A : Person
  | B : Person
  | C : Person

-- Define a function type to assign cards to people
def CardAssignment := Person → Card

-- Define A's statement
def A_statement (assignment : CardAssignment) : Prop :=
  (assignment Person.A = Card.card13 ∧ assignment Person.B = Card.card15) ∨
  (assignment Person.A = Card.card15 ∧ assignment Person.B = Card.card35)

-- Define B's statement
def B_statement (assignment : CardAssignment) : Prop :=
  (assignment Person.B = Card.card13 ∧ assignment Person.C = Card.card35) ∨
  (assignment Person.B = Card.card35 ∧ assignment Person.C = Card.card15)

-- Define C's statement
def C_statement (assignment : CardAssignment) : Prop :=
  assignment Person.C ≠ Card.card35

-- Theorem to prove
theorem A_has_card15 :
  ∀ assignment : CardAssignment,
    (∀ c : Card, ∃! p : Person, assignment p = c) →
    A_statement assignment →
    B_statement assignment →
    C_statement assignment →
    assignment Person.A = Card.card15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_has_card15_l1306_130601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1306_130610

theorem power_equation_solution (x : ℝ) : (1 / 8 : ℝ) * (2 : ℝ) ^ 36 = (8 : ℝ) ^ x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1306_130610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l1306_130636

def expression (n : ℕ) : ℚ :=
  (n^2 - 2) / (n^2 - n + 2)

def is_distinct (f : ℕ → ℚ) (a b : ℕ) : Prop :=
  f a ≠ f b

theorem distinct_values_count :
  ∃ (S : Finset ℕ),
    S.card = 98 ∧
    (∀ n, n ∈ S → n ≥ 1 ∧ n ≤ 100) ∧
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → is_distinct expression a b) ∧
    (∀ n, n ≥ 1 ∧ n ≤ 100 → ∃ m, m ∈ S ∧ expression n = expression m) :=
by sorry

#check distinct_values_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l1306_130636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_line_equation_l1306_130696

noncomputable section

-- Define the circles
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0
def circle_D (x y : ℝ) : Prop := (x - 5)^2 + (y - 4)^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define a line passing through a point with a given slope
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 5) + 4

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x0 y0 k : ℝ) : ℝ := 
  abs (k*x0 - y0 + 4 - 5*k) / Real.sqrt (k^2 + 1)

theorem circle_tangency_and_line_equation :
  -- Circles C and D are externally tangent
  (∃ x y : ℝ, circle_C x y ∧ circle_D x y) ∧
  distance 2 0 5 4 = 5 ∧
  -- The equation of the tangent line is either x = 5 or 7x - 24y + 61 = 0
  (∀ x y : ℝ, (x = 5 ∨ 7*x - 24*y + 61 = 0) → 
    (x = 5 ∧ circle_C x y) ∨ 
    (distance_point_to_line 2 0 (7/24) = 3 ∧ line (7/24) x y ∧ circle_C x y)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_line_equation_l1306_130696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_4_statement_5_l1306_130667

-- Define the function for statement 4
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin (Real.pi / 2 - x)

-- Define the function for statement 5
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

-- Theorem for statement 4
theorem statement_4 :
  (∀ x, f x = f (-x)) ∧ (∀ x, f x ≤ 2) ∧ (∃ x, f x = 2) := by sorry

-- Theorem for statement 5
theorem statement_5 (a b : ℝ) :
  g a b (-3) = 5 → g a b (Real.pi + 3) = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_4_statement_5_l1306_130667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preservation_interval_l1306_130603

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x - 1)^2 + 1

theorem preservation_interval :
  ∃! a b : ℝ, a < b ∧
  (∀ x, a ≤ x ∧ x ≤ b → a ≤ f x ∧ f x ≤ b) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y) ∧
  a = 1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preservation_interval_l1306_130603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_price_ratio_l1306_130602

theorem juice_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  (0.85 * p) / (1.3 * v) / (p / v) = 17 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_price_ratio_l1306_130602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_theorem_l1306_130686

-- Define the piecewise function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then x - 3 else Real.sqrt (x + 1)

-- State the theorem
theorem inverse_sum_theorem :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, f_inv (f x) = x) ∧ 
    (∀ y, f (f_inv y) = y) ∧
    (f_inv (-6) + f_inv (-5) + f_inv (-4) + f_inv (-3) + 
     f_inv 2 + f_inv 3 + f_inv 4 + f_inv 5 + f_inv 6 = 19) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_theorem_l1306_130686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_on_line_l1306_130687

/-- A parabola C is defined by the equation y² = 2px. Its focus lies on the line x + y - 2 = 0. -/
theorem parabola_focus_on_line (p : ℝ) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ x + y - 2 = 0 ∧ x = p/2) →
  p = 4 ∧ (∀ x y : ℝ, y^2 = 2*p*x → x = -2 → y^2 = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_on_line_l1306_130687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_naclO4_formation_l1306_130676

/-- Represents the number of moles of a substance -/
def Moles := ℝ

/-- Represents a chemical reaction with a 1:1:1 ratio -/
structure Reaction :=
  (reactant1 : Moles)
  (reactant2 : Moles)
  (product : Moles)
  (ratio : reactant1 = reactant2 ∧ reactant1 = product)

/-- Theorem: Given 3 moles of NaOH and 3 moles of HClO4 in a 1:1:1 ratio reaction,
    the number of moles of NaClO4 formed is 3 -/
theorem naclO4_formation 
  (naOH : Moles) 
  (hclO4 : Moles) 
  (reaction : Reaction) 
  (h1 : naOH = (3 : ℝ)) 
  (h2 : hclO4 = (3 : ℝ)) 
  (h3 : reaction.reactant1 = naOH) 
  (h4 : reaction.reactant2 = hclO4) : 
  reaction.product = (3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_naclO4_formation_l1306_130676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l1306_130671

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

-- State the theorem
theorem max_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  f x = 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l1306_130671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1306_130604

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (4, 5)
def c : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem triangle_area : |a.1 * c.2 - a.2 * c.1| / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1306_130604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_from_midpoints_l1306_130651

/-- Given a triangle XYZ in ℝ³, if the midpoints of its sides are known, 
    then the coordinates of vertex X can be determined. -/
theorem triangle_vertex_from_midpoints 
  (X Y Z : ℝ × ℝ × ℝ) 
  (M N P : ℝ × ℝ × ℝ) :
  M = (3, 2, -3) →  -- Midpoint of YZ
  N = (-1, 3, -5) → -- Midpoint of XZ
  P = (4, 0, 6) →   -- Midpoint of XY
  M = ((Y.1 + Z.1) / 2, (Y.2.1 + Z.2.1) / 2, (Y.2.2 + Z.2.2) / 2) →
  N = ((X.1 + Z.1) / 2, (X.2.1 + Z.2.1) / 2, (X.2.2 + Z.2.2) / 2) →
  P = ((X.1 + Y.1) / 2, (X.2.1 + Y.2.1) / 2, (X.2.2 + Y.2.2) / 2) →
  X = (8, -1, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_from_midpoints_l1306_130651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_living_room_width_l1306_130663

/-- The width of a rectangular room, given its length and total area -/
noncomputable def room_width (length : ℝ) (total_area : ℝ) : ℝ :=
  total_area / length

/-- Theorem: The width of Tom's living room is 20 feet -/
theorem toms_living_room_width :
  let length : ℝ := 16
  let total_area : ℝ := 320
  room_width length total_area = 20 := by
  -- Unfold the definition of room_width
  unfold room_width
  -- Simplify the expression
  simp
  -- Check that 320 / 16 = 20
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_living_room_width_l1306_130663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equation_l1306_130669

theorem cosine_sum_equation (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = π/6 + π/3 * (k : ℝ) ∨ x = π/2 + π * (k : ℝ) ∨ x = π/4 + π/2 * (k : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equation_l1306_130669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_and_properties_l1306_130684

noncomputable def f (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.sin (2*x + Real.pi/6) + 5/4

noncomputable def amplitude (h : ℝ → ℝ) : ℝ := sorry
noncomputable def period (h : ℝ → ℝ) : ℝ := sorry
noncomputable def phase_shift (h : ℝ → ℝ) : ℝ := sorry

theorem function_equivalence_and_properties :
  (∀ x, f x = g x) ∧
  (amplitude g = 1/2) ∧
  (period g = Real.pi) ∧
  (phase_shift g = Real.pi/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_and_properties_l1306_130684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_expression_l1306_130693

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

/-- The recursive definition of fₙ -/
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => f ∘ f_n n

/-- The theorem stating the expression for f₂₀₁₈ -/
theorem f_2018_expression (x : ℝ) (hx : x ≥ 0) :
  f_n 2018 x = x / (1 + 2018 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_expression_l1306_130693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_no_triangle_with_three_obtuse_angles_no_triangle_with_three_obtuse_angles_exists_l1306_130654

/-- An angle is obtuse if it is greater than 90 degrees -/
def IsObtuse (angle : ℝ) : Prop := angle > 90

/-- Predicate to represent a valid triangle -/
def IsTriangle (a b c : ℝ) : Prop := a + b + c = 180

/-- The sum of angles in a triangle is 180 degrees -/
theorem triangle_angle_sum (a b c : ℝ) : IsTriangle a b c ↔ a + b + c = 180 := by
  rfl

/-- Theorem: A triangle with three obtuse angles cannot exist -/
theorem no_triangle_with_three_obtuse_angles :
  ∀ a b c : ℝ, IsObtuse a → IsObtuse b → IsObtuse c → ¬ IsTriangle a b c := by
  intro a b c ha hb hc h
  have h1 : a > 90 := ha
  have h2 : b > 90 := hb
  have h3 : c > 90 := hc
  have h4 : a + b + c > 270 := by linarith
  have h5 : a + b + c ≠ 180 := by linarith
  exact h5 h

/-- Corollary: There exists no triangle with three obtuse angles -/
theorem no_triangle_with_three_obtuse_angles_exists :
  ¬ ∃ a b c : ℝ, IsObtuse a ∧ IsObtuse b ∧ IsObtuse c ∧ IsTriangle a b c := by
  intro h
  rcases h with ⟨a, b, c, ha, hb, hc, ht⟩
  exact no_triangle_with_three_obtuse_angles a b c ha hb hc ht

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_no_triangle_with_three_obtuse_angles_no_triangle_with_three_obtuse_angles_exists_l1306_130654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_with_inscribed_rectangle_l1306_130649

/-- Given a rectangle with dimensions 7 cm by 24 cm inscribed in a circle,
    prove that the circumference of the circle is 25π cm. -/
theorem circle_circumference_with_inscribed_rectangle :
  ∀ (circle : ℝ → ℝ → Prop) (rectangle : ℝ → ℝ → Prop),
  (∃ (x y : ℝ), rectangle x y ∧ x = 7 ∧ y = 24) →
  (∀ (x y : ℝ), rectangle x y → circle x y) →
  (∃ (r : ℝ), ∀ (x y : ℝ), circle x y ↔ x^2 + y^2 = r^2) →
  (∃ (C : ℝ), C = 25 * Real.pi ∧ 
    ∀ (x y : ℝ), circle x y → C = 2 * Real.pi * Real.sqrt (x^2 + y^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_with_inscribed_rectangle_l1306_130649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_common_number_proof_l1306_130609

/-- Checks if a number can be represented as a sum of distinct powers of a given base -/
def isSumOfDistinctPowers (n : ℕ) (base : ℕ) : Prop :=
  ∃ (s : Finset ℕ), n = s.sum (fun i => base ^ i)

/-- The 150th term in the sequence of powers of 3 and sums of distinct powers of 3 -/
def term150 : ℕ := 2280

/-- The smallest number that appears in both sequences and is at least the 150th term -/
def smallestCommonNumber : ℕ := 3125

theorem smallest_common_number_proof :
  (isSumOfDistinctPowers smallestCommonNumber 3) ∧
  (isSumOfDistinctPowers smallestCommonNumber 5) ∧
  (smallestCommonNumber ≥ term150) ∧
  (∀ m : ℕ, m < smallestCommonNumber →
    ¬(isSumOfDistinctPowers m 3 ∧ isSumOfDistinctPowers m 5 ∧ m ≥ term150)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_common_number_proof_l1306_130609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspector_meters_examined_l1306_130643

theorem inspector_meters_examined (rejection_rate : Real) (rejected_meters : Nat) (total_meters : Nat) : 
  rejection_rate = 0.08 / 100 →
  rejected_meters = 2 →
  (rejection_rate * (total_meters : Real) = rejected_meters) →
  total_meters = 2500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspector_meters_examined_l1306_130643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_b_zero_l1306_130689

noncomputable def f (x : ℝ) : ℝ := Real.cos x

def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

noncomputable def f_deriv (x : ℝ) : ℝ := -Real.sin x

def g_deriv (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem common_tangent_implies_b_zero (b : ℝ) (m : ℝ) :
  f 0 = g b 0 ∧ f_deriv 0 = g_deriv b 0 → b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_b_zero_l1306_130689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_girls_l1306_130682

theorem number_of_girls (total_people : ℕ) (total_saplings : ℕ) (teacher_saplings : ℕ) 
  (boy_saplings : ℕ) (girl_saplings : ℕ) (h1 : total_people = 13) 
  (h2 : total_saplings = 44) (h3 : teacher_saplings = 6) (h4 : boy_saplings = 4) 
  (h5 : girl_saplings = 2) : 
  ∃ (boys girls : ℕ), boys + girls = 12 ∧ 
  boys * boy_saplings + girls * girl_saplings = total_saplings - teacher_saplings ∧
  girls = 5 := by
  -- The proof goes here
  sorry

#check number_of_girls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_girls_l1306_130682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_mile_ride_cost_l1306_130675

/-- Calculates the cost of a taxi ride with given parameters -/
noncomputable def taxi_ride_cost (initial_fare : ℚ) (rate_per_mile : ℚ) (discount_rate : ℚ) (distance : ℚ) : ℚ :=
  if distance ≤ 10 then
    initial_fare + rate_per_mile * distance
  else
    let first_10_miles := initial_fare + rate_per_mile * 10
    let discounted_10_miles := first_10_miles * (1 - discount_rate)
    let remaining_distance := distance - 10
    discounted_10_miles + rate_per_mile * remaining_distance

/-- Theorem stating the cost of a 12-mile taxi ride with given parameters -/
theorem twelve_mile_ride_cost :
  taxi_ride_cost 2 (3/10) (1/10) 12 = 51/10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_mile_ride_cost_l1306_130675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_squared_greater_than_beta_squared_l1306_130657

theorem alpha_squared_greater_than_beta_squared 
  (α β : ℝ) 
  (h1 : α ∈ Set.Icc (-Real.pi/2) (Real.pi/2)) 
  (h2 : β ∈ Set.Icc (-Real.pi/2) (Real.pi/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_squared_greater_than_beta_squared_l1306_130657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_not_constant_function_of_d_l1306_130632

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Point C on the y-axis -/
def C (d : ℝ) : ℝ × ℝ := (0, d^2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Function t for a given chord AB passing through C -/
noncomputable def t (A B : ℝ × ℝ) (d : ℝ) : ℝ :=
  1 / distance A (C d) + 1 / distance B (C d)

/-- Theorem stating that t is not a constant function of d -/
theorem t_not_constant_function_of_d :
  ∀ d : ℝ, ∃ A B A' B' : ℝ × ℝ,
    parabola A.1 = A.2 ∧
    parabola B.1 = B.2 ∧
    parabola A'.1 = A'.2 ∧
    parabola B'.1 = B'.2 ∧
    (C d).2 = A.2 + (B.2 - A.2) / (B.1 - A.1) * ((C d).1 - A.1) ∧
    (C d).2 = A'.2 + (B'.2 - A'.2) / (B'.1 - A'.1) * ((C d).1 - A'.1) ∧
    t A B d ≠ t A' B' d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_not_constant_function_of_d_l1306_130632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l1306_130622

theorem simplify_square_roots : Real.sqrt (11 * 3) * Real.sqrt (3^3 * 11^3) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l1306_130622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1306_130660

theorem inequality_equivalence (l : ℝ) : 
  (∀ x ≤ l, ∀ n : ℕ+, x^2 + (1/2)*x ≥ (1/2)^(n : ℝ)) ↔ l ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1306_130660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l1306_130646

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Focal length of the ellipse -/
def focal_length (c : ℝ) : ℝ := 2 * c

/-- Point P on the ellipse -/
def point_P (c : ℝ) : ℝ × ℝ := (c, 1)

/-- Point Q on the ellipse -/
noncomputable def point_Q : ℝ × ℝ := (2 * Real.sqrt 3 / 3, -2 * Real.sqrt 3 / 3)

/-- Left focus F of the ellipse -/
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)

/-- Line l parallel to PF -/
noncomputable def line_l (x y m : ℝ) : Prop :=
  y = (Real.sqrt 2 / 4) * x + m

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ :=
  sorry -- Implementation of area calculation

/-- Theorem stating the equation of ellipse C and the equation of line l that maximizes the area of triangle AOB -/
theorem ellipse_and_max_area_line :
  ∃ (a b c : ℝ),
    ellipse_C c 1 a b ∧
    ellipse_C (2 * Real.sqrt 3 / 3) (-2 * Real.sqrt 3 / 3) a b ∧
    (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 2 = 1) ∧
    (∃ m : ℝ, m = Real.sqrt 5 / 2 ∧
      (∀ x y, line_l x y m ∨ line_l x y (-m) →
        ∃ A B : ℝ × ℝ,
          ellipse_C A.1 A.2 a b ∧
          ellipse_C B.1 B.2 a b ∧
          line_l A.1 A.2 m ∧
          line_l B.1 B.2 m ∧
          ∀ m' : ℝ, m' ≠ m ∧ m' ≠ -m →
            ∃ A' B' : ℝ × ℝ,
              ellipse_C A'.1 A'.2 a b ∧
              ellipse_C B'.1 B'.2 a b ∧
              line_l A'.1 A'.2 m' ∧
              line_l B'.1 B'.2 m' →
              area_triangle (0, 0) A B > area_triangle (0, 0) A' B')) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l1306_130646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l1306_130680

variable (x : ℝ)

/-- The least common multiple of 1/(4x), 1/(5x), and 1/(6x) is 1/(60x) for non-zero real x -/
theorem lcm_of_fractions (hx : x ≠ 0) :
  lcm (1 / (4 * x)) (lcm (1 / (5 * x)) (1 / (6 * x))) = 1 / (60 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l1306_130680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1306_130690

-- Define the circle C
noncomputable def circle_C (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 + Real.sin α)

-- Define the line l in polar form
noncomputable def line_l (θ : ℝ) : ℝ := 1 / Real.sin θ

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-1, 1), (1, 1)}

-- Theorem statement
theorem circle_line_intersection :
  ∀ (α θ : ℝ), 
    (∃ (p : ℝ), p * Real.sin θ = 1 ∧ circle_C α = (p * Real.cos θ, p * Real.sin θ)) 
    ↔ circle_C α ∈ intersection_points :=
by
  sorry

#check circle_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1306_130690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_three_l1306_130670

/-- The length of the chord cut by a circle from a line -/
noncomputable def chord_length (circle_eq : ℝ → ℝ → Prop) (line_eq : ℝ → ℝ → Prop) : ℝ :=
  2 * Real.sqrt (1 - (1 / 2) ^ 2)

/-- The circle equation x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The line equation √3x + y - 1 = 0 -/
def given_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 1 = 0

theorem chord_length_is_sqrt_three :
  chord_length unit_circle given_line = Real.sqrt 3 := by
  sorry

#check chord_length_is_sqrt_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_three_l1306_130670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_l1306_130624

/-- The opening price of stock M in dollars -/
def opening_price : ℚ := 25

/-- The percentage increase in the stock price -/
def percentage_increase : ℚ := 12.00000000000001

/-- The closing price of stock M in dollars -/
noncomputable def closing_price : ℚ := opening_price * (1 + percentage_increase / 100)

/-- Theorem stating that the closing price of stock M is $28.00 when rounded to two decimal places -/
theorem stock_price_increase : Int.floor (closing_price * 100) / 100 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_l1306_130624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_required_expenditure_calculation_l1306_130699

/-- The R&D expenditure required to increase the average labor productivity by 1 million rubles per person -/
noncomputable def required_expenditure (rd_expenditure : ℝ) (productivity_increase : ℝ) : ℝ :=
  rd_expenditure / productivity_increase

/-- Theorem stating the required R&D expenditure to increase productivity by 1 million rubles per person -/
theorem required_expenditure_calculation : 
  let rd_expenditure : ℝ := 3013.94
  let productivity_increase : ℝ := 3.29
  ∃ ε > 0, |required_expenditure rd_expenditure productivity_increase - 916| < ε := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_required_expenditure_calculation_l1306_130699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_change_notation_l1306_130695

/-- Represents a temperature change in degrees Celsius -/
structure TempChange where
  value : ℝ
  increase : Bool

/-- Notation for a temperature change -/
def tempNotation (t : TempChange) : ℝ :=
  if t.increase then t.value else -t.value

theorem temp_change_notation (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  tempNotation ⟨x, true⟩ = x ∧
  tempNotation ⟨y, false⟩ = -y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_change_notation_l1306_130695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_stable_numbers_l1306_130648

/-- A number is stable if the product of any two positive integers that end in it also ends in it. -/
def IsStable (k : ℕ) (a : ℕ) : Prop :=
  0 ≤ a ∧ a < 10^k ∧
  ∀ x y : ℕ, x > 0 → y > 0 → x % 10^k = a → y % 10^k = a → (x * y) % 10^k = a

/-- There are exactly four stable k-digit numbers for any positive integer k. -/
theorem four_stable_numbers (k : ℕ) (hk : k > 0) : ∃! (s : Finset ℕ), s.card = 4 ∧ ∀ a ∈ s, IsStable k a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_stable_numbers_l1306_130648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1306_130683

open Real

-- Define the curve function as noncomputable
noncomputable def f (x : ℝ) : ℝ := log (2 * x - 1)

-- Define the line function
def line (x y : ℝ) : ℝ := 2 * x - y + 3

-- Theorem statement
theorem shortest_distance_curve_to_line :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    (∀ (x y : ℝ), y = f x → 
      (x - x₀)^2 + (y - y₀)^2 ≥ (line x₀ y₀)^2 / 5) ∧
    (line x₀ y₀)^2 / 5 = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1306_130683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_phase_l1306_130652

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_phase (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : |φ| < π / 2) :
  (∃ x₀ : ℝ, f A ω φ x₀ = 0 ∧ 
    (∀ x : ℝ, f A ω φ x = 0 → |x + π / 4| ≥ π / 2) ∧
    (∀ x : ℝ, f A ω φ (x - π / 4) = f A ω φ (-x - π / 4))) →
  φ = -π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_phase_l1306_130652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_price_ratio_l1306_130635

/-- The ratio of a table's price to a chair's price --/
noncomputable def price_ratio (chair_price table_price : ℝ) : ℝ := table_price / chair_price

theorem furniture_price_ratio 
  (chair_price table_price couch_price total_price : ℝ)
  (h1 : couch_price = 5 * table_price)
  (h2 : total_price = chair_price + table_price + couch_price)
  (h3 : couch_price = 300)
  (h4 : total_price = 380) :
  price_ratio chair_price table_price = 3 := by
  sorry

#check furniture_price_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_price_ratio_l1306_130635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_implies_a_eq_two_perpendicular_implies_a_eq_neg_three_or_zero_l1306_130692

/-- Two lines l₁ and l₂ are defined by their equations -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0

def l₂ (a : ℝ) (x y : ℝ) : Prop := (a + 2) * x - a * y - 2 = 0

/-- The slope of l₁ -/
noncomputable def slope_l₁ (a : ℝ) : ℝ := a

/-- The slope of l₂ -/
noncomputable def slope_l₂ (a : ℝ) : ℝ := (a + 2) / a

/-- Definition of parallel lines -/
def parallel (a : ℝ) : Prop := slope_l₁ a = slope_l₂ a

/-- Definition of perpendicular lines -/
def perpendicular (a : ℝ) : Prop := slope_l₁ a * slope_l₂ a = -1

/-- Theorem: If l₁ is parallel to l₂, then a = 2 -/
theorem parallel_implies_a_eq_two :
  ∀ a : ℝ, parallel a → a = 2 := by sorry

/-- Theorem: If l₁ is perpendicular to l₂, then a = -3 or a = 0 -/
theorem perpendicular_implies_a_eq_neg_three_or_zero :
  ∀ a : ℝ, perpendicular a → a = -3 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_implies_a_eq_two_perpendicular_implies_a_eq_neg_three_or_zero_l1306_130692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l1306_130608

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ)

/-- The center of the circle in polar coordinates -/
noncomputable def circle_center : ℝ × ℝ := (1, Real.pi / 4)

/-- Theorem: The center of the circle given by the equation ρ = √2(cos θ + sin θ) 
    has polar coordinates (1, π/4) -/
theorem circle_center_coordinates :
  ∀ ρ θ : ℝ, circle_equation ρ θ → 
  ∃ r φ : ℝ, (r, φ) = circle_center ∧ 
  r * Real.cos φ = Real.sqrt 2 / 2 ∧ 
  r * Real.sin φ = Real.sqrt 2 / 2 := by
  sorry

#check circle_center_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l1306_130608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1306_130600

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x*y + y*z + z*x = 1) : 
  (27/4) * (x+y) * (y+z) * (z+x) ≥ (Real.sqrt (x+y) + Real.sqrt (y+z) + Real.sqrt (z+x))^2 ∧
  (Real.sqrt (x+y) + Real.sqrt (y+z) + Real.sqrt (z+x))^2 ≥ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1306_130600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_distance_l1306_130681

structure RightTriangle :=
  (AB AC BC : ℝ)
  (right_angle : AB^2 + AC^2 = BC^2)

noncomputable def inscribed_circle (t : RightTriangle) : ℝ := 
  (t.AB * t.AC) / (2 * (t.AB + t.AC + t.BC))

noncomputable def tangent_point_AC (t : RightTriangle) : ℝ := 
  2 * (inscribed_circle t)

noncomputable def tangent_point_AB (t : RightTriangle) : ℝ := 
  (t.AB * t.AC) / (t.AB + t.AC + t.BC)

noncomputable def radius_C2 (t : RightTriangle) : ℝ := 
  inscribed_circle t * (t.AC - tangent_point_AC t) / t.BC

noncomputable def radius_C3 (t : RightTriangle) : ℝ := 
  inscribed_circle t * (t.BC - tangent_point_AB t) / t.BC

noncomputable def center_distance (t : RightTriangle) : ℝ := 
  let x_diff := t.AC - radius_C2 t - tangent_point_AB t
  let y_diff := radius_C3 t - radius_C2 t
  Real.sqrt (x_diff^2 + y_diff^2)

theorem inscribed_circles_distance (t : RightTriangle) 
  (h1 : t.AB = 105) (h2 : t.AC = 140) (h3 : t.BC = 175) : 
  center_distance t = Real.sqrt (10 * 678.65) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_distance_l1306_130681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_over_12_l1306_130655

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem tangent_slope_at_pi_over_12 :
  (deriv f) (π / 12) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_over_12_l1306_130655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_properties_l1306_130640

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define the moving point P
def point_P (P : ℝ × ℝ) : Prop := P.1 = 4

-- Define the tangent line
def tangent_line (P A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ point_P P

-- Define the area of a triangle
noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_tangent_properties
  (P A B : ℝ × ℝ)
  (h_tangent : tangent_line P A B) :
  (∃ (t : ℝ), A.1 + t * A.2 = 1 ∧ B.1 + t * B.2 = 1 ∧ right_focus.1 + t * right_focus.2 = 1) ∧
  (∃ (S : ℝ), S = 9/2 ∧ ∀ (S' : ℝ), area_triangle P A B ≥ S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_properties_l1306_130640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1306_130639

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

noncomputable def triangle_perimeter (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) +
  Real.sqrt ((x₂ - x₃)^2 + (y₂ - y₃)^2) +
  Real.sqrt ((x₃ - x₁)^2 + (y₃ - y₁)^2)

noncomputable def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (y₂ - y₁) / (x₂ - x₁)

theorem ellipse_properties
  (a b : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : eccentricity a b = 1/2)
  (x₁ y₁ x₂ y₂ xf₁ yf₁ xf₂ yf₂ : ℝ)
  (h₄ : ellipse x₁ y₁ a b)
  (h₅ : ellipse x₂ y₂ a b)
  (h₆ : triangle_perimeter x₁ y₁ x₂ y₂ xf₁ yf₁ = 8)
  (h₇ : (x₂ - xf₂) * (y₁ - yf₂) = (x₁ - xf₂) * (y₂ - yf₂)) -- A, B, and F₂ are collinear
  :
  (∀ x y, ellipse x y 2 (Real.sqrt 3) ↔ ellipse x y a b) ∧
  (∀ x₃ y₃, ellipse x₃ y₃ a b →
    line_slope 4 0 x₁ y₁ + line_slope 4 0 x₂ y₂ = line_slope 4 0 x₃ y₃ + line_slope 4 0 x₃ y₃) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1306_130639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_after_adding_tiles_l1306_130664

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : Finset (ℕ × ℕ)
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (new_tiles : Finset (ℕ × ℕ)) : TileConfiguration :=
  { tiles := initial.tiles ∪ new_tiles, perimeter := initial.perimeter }

/-- Checks if two tiles are adjacent -/
def adjacent (t1 t2 : ℕ × ℕ) : Prop :=
  (t1.1 = t2.1 ∧ (t1.2 = t2.2 + 1 ∨ t1.2 + 1 = t2.2)) ∨
  (t1.2 = t2.2 ∧ (t1.1 = t2.1 + 1 ∨ t1.1 + 1 = t2.1))

theorem perimeter_after_adding_tiles
  (initial : TileConfiguration)
  (h_initial_tiles : initial.tiles.card = 12)
  (h_initial_perimeter : initial.perimeter = 18)
  (new_tiles : Finset (ℕ × ℕ))
  (h_new_tiles : new_tiles.card = 3)
  (h_shared_side : ∀ t ∈ new_tiles, ∃ s ∈ initial.tiles, adjacent t s) :
  ∃ (final : TileConfiguration),
    final = add_tiles initial new_tiles ∧
    final.perimeter = 22 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_after_adding_tiles_l1306_130664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_prism_l1306_130685

/-- A right prism with triangular bases -/
structure RightPrism where
  base_side : ℝ  -- Length of equal sides of the isosceles base triangle
  height : ℝ     -- Height of the prism

/-- The sum of areas of two lateral faces and one base -/
noncomputable def surface_area (p : RightPrism) : ℝ :=
  2 * p.base_side * p.height + (Real.sqrt 3 / 4) * p.base_side^2

/-- The volume of the prism -/
noncomputable def volume (p : RightPrism) : ℝ :=
  (Real.sqrt 3 / 4) * p.base_side^2 * p.height

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_of_prism :
  ∃ (max_vol : ℝ),
    max_vol = 432 ∧
    ∀ (p : RightPrism),
      surface_area p = 36 →
      volume p ≤ max_vol := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_prism_l1306_130685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l1306_130656

theorem quadratic_coefficient (c m : ℝ) : 
  (∀ x, x^2 + c*x + (1/4 : ℝ) = (x + m)^2 + (1/16 : ℝ)) →
  c < 0 →
  c = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l1306_130656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minutes_to_hour_ratio_l1306_130633

theorem minutes_to_hour_ratio : 
  (12 : ℚ) / (60 : ℚ) = 1 / 5 := by
  -- Convert the fraction to decimals
  have h1 : (12 : ℚ) / (60 : ℚ) = 0.2 := by norm_num
  have h2 : (1 : ℚ) / (5 : ℚ) = 0.2 := by norm_num
  -- Show equality
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minutes_to_hour_ratio_l1306_130633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seven_consecutive_even_integers_mod_12_l1306_130637

theorem sum_of_seven_consecutive_even_integers_mod_12 (n : ℕ) : 
  (List.sum (List.map (λ i => n + 2*i) (List.range 7))) % 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seven_consecutive_even_integers_mod_12_l1306_130637
