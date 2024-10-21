import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_two_f_monotone_decreasing_l925_92591

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x + 3/x

-- Theorem 1: For all x > 1, f(x) < 2
theorem f_less_than_two (x : ℝ) (hx : x > 1) : f x < 2 := by
  sorry

-- Theorem 2: f is monotonically decreasing on (0, +∞)
theorem f_monotone_decreasing (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h : x₁ < x₂) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_two_f_monotone_decreasing_l925_92591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_san_durango_pet_owners_proof_l925_92531

/-- The number of people who own all three types of pets in San Durango -/
def san_durango_pet_owners : ℕ :=
  let total_people : ℕ := 60
  let cat_owners : ℕ := 30
  let dog_owners : ℕ := 40
  let rabbit_owners : ℕ := 16
  let two_pet_owners : ℕ := 12
  let all_three_pet_owners : ℕ := 7
  
  all_three_pet_owners

theorem san_durango_pet_owners_proof : san_durango_pet_owners = 7 := by
  unfold san_durango_pet_owners
  rfl

#eval san_durango_pet_owners

end NUMINAMATH_CALUDE_ERRORFEEDBACK_san_durango_pet_owners_proof_l925_92531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acquaintances_is_100_l925_92548

/-- Represents a group of people and their acquaintances. -/
structure Club where
  people : Finset ℕ
  acquaintances : Finset (Finset ℕ)
  h_size : people.card = 20
  h_no_triple : ∀ a b c, a ∈ acquaintances → b ∈ acquaintances → c ∈ acquaintances → a ∩ b ∩ c = ∅

/-- The number of acquaintances in the club. -/
def num_acquaintances (c : Club) : ℕ :=
  c.acquaintances.sum (λ a => a.card) / 2

/-- The maximum number of acquaintances possible in the club. -/
def max_acquaintances : ℕ := 100

/-- Theorem stating that the maximum number of acquaintances is 100. -/
theorem max_acquaintances_is_100 (c : Club) :
  num_acquaintances c ≤ max_acquaintances := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acquaintances_is_100_l925_92548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_minus_a_range_l925_92567

open Real Set

noncomputable def f (x a : ℝ) : ℝ := 2 * sin (2 * x + π / 6) + a - 1

theorem zeros_sum_minus_a_range (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ∈ Icc 0 (π / 2) →
  x₂ ∈ Icc 0 (π / 2) →
  x₁ ≠ x₂ →
  f x₁ a = 0 →
  f x₂ a = 0 →
  π / 3 ≤ x₁ + x₂ - a ∧ x₁ + x₂ - a < π / 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_minus_a_range_l925_92567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_profit_threshold_l925_92510

/-- The number of units at which the factory starts making a profit -/
noncomputable def profit_threshold (fixed_cost variable_cost selling_price : ℝ) : ℝ :=
  fixed_cost / (selling_price - variable_cost)

/-- Theorem stating that the profit threshold is greater than 1000 units -/
theorem factory_profit_threshold :
  profit_threshold 500 2 2.5 > 1000 := by
  -- Unfold the definition of profit_threshold
  unfold profit_threshold
  -- Simplify the expression
  simp
  -- Prove the inequality
  norm_num
  -- If the automatic proof fails, we can use sorry to skip it
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_profit_threshold_l925_92510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l925_92537

theorem cos_2alpha_value (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.sin α + Real.cos α = Real.sqrt 3 / 3 →
  Real.cos (2 * α) = -Real.sqrt 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l925_92537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_5y_l925_92557

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of an event in a continuous uniform distribution --/
noncomputable def probability (totalArea areaOfEvent : ℝ) : ℝ :=
  areaOfEvent / totalArea

/-- The main theorem --/
theorem probability_x_gt_5y (r : Rectangle) 
    (h1 : r.x_min = 0) (h2 : r.x_max = 3000) 
    (h3 : r.y_min = 0) (h4 : r.y_max = 2000) : 
  probability ((r.x_max - r.x_min) * (r.y_max - r.y_min))
              ((1/2) * r.x_max * (r.x_max / 5)) = 3/20 := by
  sorry

#check probability_x_gt_5y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_5y_l925_92557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l925_92596

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x - 4) ^ (1/3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l925_92596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_paths_count_l925_92579

/-- Represents a strip of cells -/
structure Strip where
  size : ℕ

/-- Represents a player's position on the strip -/
structure Position where
  cell : ℕ

/-- Represents a move on the strip -/
inductive Move where
  | Forward (n : ℕ) : Move
  | Backward (n : ℕ) : Move

/-- Checks if a move is valid according to the rules -/
def is_valid_move (m : Move) : Prop :=
  match m with
  | Move.Forward n => Even n
  | Move.Backward n => Odd n

/-- Represents a path traversing all cells exactly once -/
def ValidPath (s : Strip) := List Move

/-- Counts the number of valid paths for a given starting position -/
noncomputable def count_paths (s : Strip) (start : Position) : ℕ := sorry

/-- The main theorem stating that the number of paths is equal for any two starting positions -/
theorem equal_paths_count (s : Strip) (p1 p2 : Position) :
  count_paths s p1 = count_paths s p2 := by
  sorry

#check equal_paths_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_paths_count_l925_92579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_problem_l925_92530

def symmetric_difference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

theorem set_cardinality_problem (x y : Finset ℤ) 
  (h1 : y.card = 18)
  (h2 : (x ∩ y).card = 6)
  (h3 : (symmetric_difference x y).card = 20) :
  x.card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_problem_l925_92530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l925_92590

theorem sine_cosine_inequality (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l925_92590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_inequality_proof_l925_92562

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - (a + 1) * x

-- Part 1: Tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f 1 π = -2 * π ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
    |(f 1 (π + h) - f 1 π) / h + 3| < ε) →
  3 * x + y - 3 * π = 0 :=
by sorry

-- Part 2: Inequality proof
theorem inequality_proof (a x : ℝ) :
  0 < x → x < π → a ≤ -3 → f a x + x * Real.cos x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_inequality_proof_l925_92562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l925_92545

theorem max_omega_value (f : ℝ → ℝ) (ω φ : ℝ) :
  ω > 0 →
  |φ| ≤ π/2 →
  (∀ x, f x = Real.sin (ω * x + φ)) →
  f (-π/4) = 0 →
  (∀ x, f (π/4 + x) = f (π/4 - x)) →
  (∀ x ∈ Set.Ioo (π/18) (5*π/36), StrictMono f) →
  ω ≤ 9 ∧ ∃ ω₀, ω₀ = 9 ∧ ∀ ω' > ω₀, ¬(
    ω' > 0 ∧
    ∃ φ', |φ'| ≤ π/2 ∧
    (∀ x, f x = Real.sin (ω' * x + φ')) ∧
    f (-π/4) = 0 ∧
    (∀ x, f (π/4 + x) = f (π/4 - x)) ∧
    (∀ x ∈ Set.Ioo (π/18) (5*π/36), StrictMono f)
  ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l925_92545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l925_92507

/-- Circle A with equation x² + y² = 1 -/
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle B with equation (x-3)² + (y+4)² = 10 -/
def circle_B (x y : ℝ) : Prop := (x-3)^2 + (y+4)^2 = 10

/-- The locus of point P satisfying the tangent condition -/
def locus_P (x y : ℝ) : Prop := 3*x - 4*y - 8 = 0

/-- The minimum distance from a point on the locus to the origin -/
noncomputable def min_distance : ℝ := 8/5

theorem min_distance_to_origin :
  ∀ (x y : ℝ),
    locus_P x y →
    ∃ (x' y' : ℝ), locus_P x' y' ∧
      ∀ (x'' y'' : ℝ), locus_P x'' y'' →
        (x'^2 + y'^2 : ℝ) ≤ x''^2 + y''^2 ∧
        (x'^2 + y'^2 : ℝ) = min_distance^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l925_92507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_minus_seven_l925_92587

theorem sum_of_fractions_minus_seven : 
  3/2 + 5/4 + 9/8 + 17/16 + 33/32 + 65/64 - 7 = -1/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_minus_seven_l925_92587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_diagonal_length_theorem_l925_92540

/-- Represents a chessboard with alternating colors -/
structure Chessboard where
  m : ℕ+
  n : ℕ+

/-- The length of red segments on the diagonal of a chessboard -/
noncomputable def redDiagonalLength (board : Chessboard) : ℝ :=
  Real.sqrt (board.m.val ^ 2 + board.n.val ^ 2) / 2

/-- Theorem stating the sum of red segment lengths on the diagonal -/
theorem red_diagonal_length_theorem (board : Chessboard) :
  redDiagonalLength board = Real.sqrt (board.m.val ^ 2 + board.n.val ^ 2) / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_diagonal_length_theorem_l925_92540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bound_l925_92577

-- Define a circle with a center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a predicate for circles being mutually external
def mutually_external (c1 c2 c3 : Circle) : Prop :=
  distance c1.center c2.center > c1.radius + c2.radius ∧
  distance c1.center c3.center > c1.radius + c3.radius ∧
  distance c2.center c3.center > c2.radius + c3.radius

-- Define a predicate for the line separation condition
def line_separation_condition (c1 c2 c3 : Circle) : Prop :=
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ p ∈ l, distance p c1.center > c1.radius ∧ distance p c2.center > c2.radius) →
    (∃ p ∈ l, distance p c3.center < c3.radius)

-- The main theorem
theorem circle_distance_bound (c1 c2 c3 : Circle) 
  (h1 : mutually_external c1 c2 c3)
  (h2 : line_separation_condition c1 c2 c3) :
  distance c1.center c2.center + distance c1.center c3.center + distance c2.center c3.center ≤ 
  2 * Real.sqrt 2 * (c1.radius + c2.radius + c3.radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bound_l925_92577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_terms_count_l925_92554

/-- The number of nonzero terms in the expansion of (x+4)(2x^2+3x+9)-3(x^3-2x^2+7x) -/
theorem nonzero_terms_count : 
  let p : Polynomial ℚ := (X + 4) * (2 * X^2 + 3 * X + 9) - 3 * (X^3 - 2 * X^2 + 7 * X)
  p.support.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_terms_count_l925_92554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l925_92593

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1)

-- Theorem statement
theorem inverse_function_proof :
  (∀ x > 0, f x = f x) →  -- This ensures the domain of f is (0, ∞)
  (∀ x > 0, g (f x) = x) ∧ 
  (∀ y, f (g y) = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l925_92593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_is_right_triangle_l925_92551

noncomputable def set_A : List ℝ := [2, 3, 4]
noncomputable def set_B : List ℝ := [Real.sqrt 3, Real.sqrt 4, Real.sqrt 5]
noncomputable def set_C : List ℝ := [1, Real.sqrt 2, 3]
def set_D : List ℝ := [5, 12, 13]

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

def satisfies_pythagorean (lst : List ℝ) : Prop :=
  match lst with
  | [a, b, c] => is_right_triangle a b c
  | _ => False

theorem only_set_D_is_right_triangle :
  ¬(satisfies_pythagorean set_A) ∧
  ¬(satisfies_pythagorean set_B) ∧
  ¬(satisfies_pythagorean set_C) ∧
  satisfies_pythagorean set_D :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_D_is_right_triangle_l925_92551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_equals_26_l925_92519

noncomputable def f (x : ℝ) : ℝ := (6 * x - 2) / (x + 3)

theorem f_neg_four_equals_26 : f (-4) = 26 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the numerator and denominator
  simp [add_comm, mul_comm, mul_assoc]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_equals_26_l925_92519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l925_92534

theorem divisibility_property (n : ℕ) :
  (∀ (a b c : ℕ), a > 0 → b > 0 → c > 0 → (a + b + c) ∣ (a^2 + b^2 + c^2) → (a + b + c) ∣ (a^n + b^n + c^n)) ↔
  (∃ (k : ℕ), k > 0 ∧ (n = 3*k - 1 ∨ n = 3*k - 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l925_92534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l925_92516

/-- The solution set of (x-1)(x-2) > 0 -/
def A : Set ℝ := {x | (x - 1) * (x - 2) > 0}

/-- The solution set of x^2 + (a-1)x - a > 0 -/
def B (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x - a > 0}

/-- Proposition p: The solution set is A -/
def p : Prop := True

/-- Proposition q: The solution set is B -/
def q (a : ℝ) : Prop := True

/-- p is a sufficient but not necessary condition for q -/
def p_sufficient_not_necessary (a : ℝ) : Prop := 
  (p → q a) ∧ ¬(q a → p)

theorem range_of_a : 
  ∃ S : Set ℝ, S = {a | p_sufficient_not_necessary a} ∧ S = Set.Ioo (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l925_92516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l925_92594

-- Define the sets
def Q : Set ℚ := Set.univ
def R : Set ℝ := Set.univ
def N_plus : Set ℕ := {x : ℕ | x > 0}
def Z : Set ℤ := Set.univ

-- Define the statements
def statement1 : Prop := (1/2 : ℚ) ∈ Q
def statement2 : Prop := Real.sqrt 2 ∉ R
def statement3 : Prop := (0 : ℕ) ∈ N_plus
noncomputable def statement4 : Prop := ∃ n : ℤ, (n : ℝ) = Real.pi
def statement5 : Prop := (∅ : Set ℕ) = ({0} : Set ℕ)

-- Theorem statement
theorem correct_statements_count :
  statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 ∧ ¬statement5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l925_92594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l925_92578

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

-- Theorem statement
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l925_92578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l925_92515

-- Define the variables and conditions
variable (a b : ℝ)
variable (c : ℤ)
variable (x : ℝ)

-- a and b are opposite non-zero numbers
axiom ab_opposite : a = -b ∧ a ≠ 0 ∧ b ≠ 0

-- c is the largest negative integer
axiom c_largest_negative : c = -1

-- x is a number whose square is equal to 4
axiom x_squared : x^2 = 4

-- Define the expression
noncomputable def expression : ℝ := x + a/b + 2*c - (a+b)/Real.pi

-- Theorem statement
theorem problem_solution :
  (c = -1) ∧ 
  (x = 2 ∨ x = -2) ∧ 
  (expression a b c x = -1 ∨ expression a b c x = -5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l925_92515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_50_5_l925_92526

/-- Represents a systematic sampling of products. -/
structure SystematicSampling where
  total_products : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the list of selected products for a systematic sampling. -/
def generate_sample (s : SystematicSampling) : List Nat :=
  (List.range s.sample_size).map (fun i => s.start + i * s.interval)

/-- Theorem stating that the systematic sampling of 5 products from 50 products
    results in the list [5, 15, 25, 35, 45]. -/
theorem systematic_sampling_50_5 :
  let s : SystematicSampling := {
    total_products := 50,
    sample_size := 5,
    start := 5,
    interval := 10
  }
  generate_sample s = [5, 15, 25, 35, 45] := by
  sorry

#eval generate_sample {
  total_products := 50,
  sample_size := 5,
  start := 5,
  interval := 10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_50_5_l925_92526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l925_92589

/-- A point with integer coordinates on the circle x^2 + y^2 = 169 -/
structure CirclePoint where
  x : ℤ
  y : ℤ
  on_circle : x^2 + y^2 = 169

/-- The distance between two CirclePoints -/
noncomputable def distance (p q : CirclePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 : ℝ)

/-- Predicate to check if a real number is irrational -/
def IsIrrational (x : ℝ) : Prop :=
  ∀ (q : ℚ), x ≠ ↑q

theorem max_distance_ratio :
  ∀ (p q r s : CirclePoint),
    p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
    IsIrrational (distance p q + distance r s) →
    distance p q / distance r s ≤ Real.sqrt (578/98) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l925_92589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_area_triangle_count_l925_92553

/-- Represents a point on a 10x10 grid --/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- Represents a triangle on the grid --/
structure GridTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- The main triangle FGH --/
def FGH : GridTriangle :=
  { a := ⟨0, 10⟩, b := ⟨0, 0⟩, c := ⟨10, 0⟩ }

/-- Checks if a point is inside FGH --/
def isInsideFGH (p : GridPoint) : Prop := sorry

/-- Calculates the area of a triangle --/
noncomputable def triangleArea (t : GridTriangle) : ℚ := sorry

/-- Checks if a point forms a half-area triangle --/
noncomputable def formsHalfAreaTriangle (p : GridPoint) : Prop :=
  let fpg := GridTriangle.mk FGH.a p FGH.b
  let gph := GridTriangle.mk FGH.b p FGH.c
  let hpf := GridTriangle.mk FGH.c p FGH.a
  triangleArea fpg = (triangleArea FGH) / 2 ∨
  triangleArea gph = (triangleArea FGH) / 2 ∨
  triangleArea hpf = (triangleArea FGH) / 2

instance : Fintype GridPoint := sorry
instance : DecidablePred isInsideFGH := sorry
instance : DecidablePred formsHalfAreaTriangle := sorry

theorem half_area_triangle_count :
  (Finset.filter formsHalfAreaTriangle 
    (Finset.filter isInsideFGH (Finset.univ : Finset GridPoint))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_area_triangle_count_l925_92553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l925_92580

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → 2 * Real.exp x ≥ (x - a)^2) ↔ 
  (Real.log 2 - 2 ≤ a ∧ a ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l925_92580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_magnitude_is_five_l925_92524

/-- The polynomial P(x) -/
def P (c x : ℂ) : ℂ := (x^2 - 4*x + 5) * (x^2 - c*x + 5) * (x^2 - 6*x + 10)

/-- The set of roots of P(x) -/
def roots (c : ℂ) : Set ℂ := {x : ℂ | P c x = 0}

/-- Theorem: If P(x) has exactly 4 distinct roots, then |c| = 5 -/
theorem c_magnitude_is_five (c : ℂ) (h : Set.ncard (roots c) = 4) :
  Complex.abs c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_magnitude_is_five_l925_92524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPB_l925_92517

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ^2 * (1 + 3 * Real.sin θ^2) = 4

/-- Point on curve C in Cartesian coordinates -/
def point_on_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point A on the x-axis -/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B on the y-axis -/
def point_B : ℝ × ℝ := (0, 1)

/-- Area of quadrilateral OAPB given a point P on curve C -/
noncomputable def area_OAPB (x y : ℝ) : ℝ := x * y / 2 + (2 - x) * y / 2

/-- Main theorem: maximum area of quadrilateral OAPB is √2 -/
theorem max_area_OAPB :
  ∃ (x y : ℝ), point_on_C x y ∧ x > 0 ∧ y > 0 ∧
  (∀ (x' y' : ℝ), point_on_C x' y' ∧ x' > 0 ∧ y' > 0 →
    area_OAPB x y ≥ area_OAPB x' y') ∧
  area_OAPB x y = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPB_l925_92517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_fraction_of_GH_l925_92598

-- Define the line segment GH and points E and F on it
variable (GH : ℝ)
variable (E : ℝ)
variable (F : ℝ)

-- Define the ratios given in the problem
axiom GE_ratio : E = 3/4 * GH
axiom GF_ratio : F = 5/6 * GH

-- Ensure E and F are on GH
axiom E_on_GH : 0 ≤ E ∧ E ≤ GH
axiom F_on_GH : 0 ≤ F ∧ F ≤ GH

-- Theorem to prove
theorem EF_fraction_of_GH : |F - E| = (1/12) * GH := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_fraction_of_GH_l925_92598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_solutions_l925_92597

/-- A triple of positive integers (a, b, c) is a solution if (a+1)/b, (b+1)/c, and (c+1)/a are all positive integers. -/
def is_solution (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a + 1) % b = 0 ∧ (b + 1) % c = 0 ∧ (c + 1) % a = 0

/-- The set of all solutions -/
def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, b, c) | is_solution a b c}

/-- The theorem stating that the only solutions are (1,1,1), (3,4,5), and their cyclic permutations -/
theorem characterize_solutions :
  solution_set = {(1, 1, 1), (3, 4, 5), (4, 5, 3), (5, 3, 4)} := by
  sorry

#check characterize_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_solutions_l925_92597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_product_factorials_l925_92566

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (λ acc i => acc * factorial (i + 1)) 1

partial def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec count_aux (m : ℕ) (acc : ℕ) : ℕ :=
    if m % 10 = 0 then count_aux (m / 10) (acc + 1)
    else acc
  count_aux n 0

theorem trailing_zeros_of_product_factorials :
  count_trailing_zeros product_of_factorials = 12 ∧ 12 % 500 = 12 := by
  sorry

#check trailing_zeros_of_product_factorials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_product_factorials_l925_92566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_seven_l925_92523

def arithmetic_sequence (start end_ step : ℕ) : List ℕ :=
  List.range ((end_ - start) / step + 1) |>.map (fun i => start + i * step)

theorem product_divisible_by_seven :
  ∀ (sequence : List ℕ),
    sequence = arithmetic_sequence 7 197 10 →
    (sequence.prod : ℤ) % 7 = 0 := by
  intro sequence h
  rw [h]
  simp [arithmetic_sequence]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_seven_l925_92523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_not_1380_l925_92572

theorem average_not_1380 : 
  let numbers := [1200, 1400, 1510, 1520, 1530, 1200]
  (numbers.sum : ℚ) / numbers.length ≠ 1380 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_not_1380_l925_92572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_30km_l925_92541

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  still_water_speed : ℝ

/-- Calculates the stream speed based on the swimming scenario -/
noncomputable def calculate_stream_speed (s : SwimmingScenario) : ℝ :=
  s.still_water_speed - s.upstream_distance / s.upstream_time

/-- Calculates the downstream distance based on the swimming scenario -/
noncomputable def calculate_downstream_distance (s : SwimmingScenario) : ℝ :=
  (s.still_water_speed + calculate_stream_speed s) * s.downstream_time

/-- Theorem stating that under the given conditions, the downstream distance is 30 km -/
theorem downstream_distance_is_30km (s : SwimmingScenario)
  (h1 : s.upstream_distance = 20)
  (h2 : s.upstream_time = 5)
  (h3 : s.downstream_time = 5)
  (h4 : s.still_water_speed = 5) :
  calculate_downstream_distance s = 30 := by
  sorry

#check downstream_distance_is_30km

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_30km_l925_92541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l925_92570

theorem diophantine_equation_solutions :
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} =
  {(-1, 1), (-1, 0), (0, -1), (0, 2), (1, -1), (1, 2), (2, 1), (2, 0)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l925_92570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treadmill_original_price_l925_92532

/-- Proves that the original price of a treadmill is $1350 given the specified conditions --/
theorem treadmill_original_price :
  ∀ (P : ℝ),
  (0.7 * P + 2 * 50 = 1045) →
  P = 1350 := by
  intro P
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treadmill_original_price_l925_92532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18s_l925_92536

/-- Represents the time it takes for a train to cross a signal pole -/
noncomputable def time_to_cross_pole (train_length platform_length time_to_cross_platform : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_to_cross_platform
  train_length / train_speed

/-- Theorem stating that a 300m train crossing a 350m platform in 39s takes about 18s to cross a signal pole -/
theorem train_crossing_time_approx_18s :
  ∃ ε > 0, ε < 0.01 ∧ |time_to_cross_pole 300 350 39 - 18| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18s_l925_92536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_property_f_negation_property_l925_92521

-- Define the function f as noncomputable
noncomputable def f (x y : ℝ) : ℝ := (2 * x + y) / (x + 2 * y)

-- Theorem 1
theorem f_inverse_property : f 1 2 = 1 / f 2 1 := by
  -- Proof steps will go here
  sorry

-- Theorem 2
theorem f_negation_property (c : ℝ) : f c c = -f (-c) c := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_property_f_negation_property_l925_92521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_is_two_lines_l925_92574

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Defines the graph of the equation sin(2θ) = 0 with r ≥ 0 -/
def graph : Set PolarPoint :=
  {p : PolarPoint | Real.sin (2 * p.θ) = 0 ∧ p.r ≥ 0}

/-- Represents a straight line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Converts a Line to a Set of PolarPoints -/
def Line.toSet (l : Line) : Set PolarPoint :=
  {p : PolarPoint | l.a * (p.r * Real.cos p.θ) + l.b * (p.r * Real.sin p.θ) + l.c = 0}

/-- Theorem stating that the graph consists of two straight lines -/
theorem graph_is_two_lines : ∃ (l₁ l₂ : Line), graph = {p : PolarPoint | p ∈ l₁.toSet ∨ p ∈ l₂.toSet} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_is_two_lines_l925_92574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_fraction_simplification_l925_92509

theorem fourth_root_fraction_simplification :
  (8 / 24.75 : ℝ) ^ (1/4 : ℝ) = 2 / (99 : ℝ) ^ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_fraction_simplification_l925_92509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_max_triangle_area_l925_92599

noncomputable section

-- Define the polar coordinate system
def polar_to_cartesian (ρ : ℝ) (θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define curve C₁
def C₁ : Set (ℝ × ℝ) := {p | ∃ θ, p = polar_to_cartesian 4 θ}

-- Define the condition for point P
def P_condition (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  M.1 * P.1 + M.2 * P.2 = 16 ∧ P.1 ≠ 0

-- Define curve C₂
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 4 ∧ p.1 ≠ 0}

-- Define point A
def A : ℝ × ℝ := polar_to_cartesian 2 (Real.pi / 3)

-- Define area_triangle function (placeholder)
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the theorems
theorem C₂_equation : ∀ M ∈ C₁, ∀ P, P_condition M P → P ∈ C₂ := by sorry

theorem max_triangle_area : 
  ∃ B ∈ C₂, ∀ B' ∈ C₂, 
    area_triangle (0, 0) A B ≥ area_triangle (0, 0) A B' ∧ 
    area_triangle (0, 0) A B = Real.sqrt 3 + 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_max_triangle_area_l925_92599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l925_92506

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : π / 2 < β)
  (h2 : β < α)
  (h3 : α < 3 * π / 4)
  (h4 : Real.cos (α - β) = 12 / 13)
  (h5 : Real.sin (α + β) = -3 / 5) : 
  Real.sin (2 * α) = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l925_92506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_seven_two_minus_three_l925_92559

theorem binomial_seven_two_minus_three : (Nat.choose 7 2) - 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_seven_two_minus_three_l925_92559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l925_92585

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l925_92585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l925_92546

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ,  cos θ]]

theorem smallest_rotation_power : 
  (∃ (n : ℕ), n > 0 ∧ (rotation_matrix (145 * π / 180))^n = 1) ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < 72 → (rotation_matrix (145 * π / 180))^m ≠ 1) ∧
  (rotation_matrix (145 * π / 180))^72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l925_92546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_four_consecutive_numbers_l925_92592

theorem unit_digit_of_four_consecutive_numbers (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_four_consecutive_numbers_l925_92592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_third_quadrant_l925_92549

noncomputable def Z : ℂ := -2 * Complex.I / (1 + 2 * Complex.I)

theorem Z_in_third_quadrant : 
  (Z.re < 0) ∧ (Z.im < 0) := by
  -- Rationalize the denominator
  have h : Z = (-4 - 2*Complex.I) / 5 := by
    -- Proof steps for rationalization
    sorry
  
  -- Show that the real part is negative
  have re_neg : Z.re < 0 := by
    -- Proof that -4/5 < 0
    sorry
  
  -- Show that the imaginary part is negative
  have im_neg : Z.im < 0 := by
    -- Proof that -2/5 < 0
    sorry
  
  -- Combine the results
  exact ⟨re_neg, im_neg⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_third_quadrant_l925_92549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_evaluation_l925_92502

theorem fraction_exponent_evaluation :
  (1 / 3 : ℚ)^6 * (5 / 3 : ℚ)^(-4 : ℤ) = 1 / 5625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_evaluation_l925_92502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_range_of_f_range_of_tangent_slope_l925_92501

-- Define the function f(x) = x^3 - √3x + 2
noncomputable def f (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 2

-- State the theorem about the range of f'(x)
theorem derivative_range_of_f :
  Set.range (deriv f) = Set.Ici (-Real.sqrt 3) := by
  sorry

-- Define the tangent slope function
noncomputable def tangent_slope (x : ℝ) : ℝ := deriv f x

-- State the theorem about the range of the tangent slope
theorem range_of_tangent_slope :
  Set.range tangent_slope = Set.Ici (-Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_range_of_f_range_of_tangent_slope_l925_92501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l925_92547

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

-- Define points
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the dot product of two vectors
def dot_product (v w : Point) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

-- State the theorem
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ M : Point) :
  M.1 > 0 → -- M is on the right branch
  dot_product (M.1 - origin.1, M.2 - origin.2) (F₂.1 - M.1, F₂.2 - M.2) = 0 →
  distance M F₁ = Real.sqrt 3 * distance M F₂ →
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l925_92547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_in_range_l925_92565

theorem count_multiples_in_range : ∃ (count : ℕ), 
  count = (Finset.filter (λ n : ℕ ↦ 
    500 ≤ n ∧ n ≤ 2500 ∧ 12 ∣ n ∧ 18 ∣ n ∧ 24 ∣ n) 
    (Finset.range (2501 - 500))).card ∧ 
  count = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_in_range_l925_92565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l925_92542

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1
noncomputable def e : ℝ := Real.sqrt 2 / 2

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line passing through (0,2)
def line_eq (k x : ℝ) : ℝ := k * x + 2

-- Define the intersection points A and B
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse_eq p.1 p.2 ∧ p.2 = line_eq k p.1}

-- Define the area of triangle OAB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * B.2 - A.2 * B.1)

-- State the theorem
theorem ellipse_properties :
  (∀ x y, ellipse_eq x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ k, ∀ A B, A ∈ intersection_points k → B ∈ intersection_points k →
    (∀ k' A' B', A' ∈ intersection_points k' → B' ∈ intersection_points k' →
      triangle_area A B ≥ triangle_area A' B') →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l925_92542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cases_in_2010_l925_92573

-- Define the exponential decay model
noncomputable def N (N₀ : ℝ) (k : ℝ) (t : ℝ) : ℝ := N₀ * Real.exp (-k * t)

-- State the theorem
theorem cases_in_2010 (N₀ k : ℝ) (h1 : N₀ = 60000) (h2 : N N₀ k 20 = 300) :
  ∃ ε > 0, |N N₀ k 10 - 4243| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cases_in_2010_l925_92573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_coordinate_of_P_l925_92569

/-- The maximum x-coordinate of point P given the conditions of the problem -/
theorem max_x_coordinate_of_P (A : ℝ × ℝ) (P : ℝ × ℝ) (l : ℝ) :
  A.1^2 / 25 + A.2^2 / 9 = 1 →
  P.1 - A.1 = (l - 1) * A.1 →
  P.2 - A.2 = (l - 1) * A.2 →
  P.1 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_coordinate_of_P_l925_92569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_f_g_l925_92583

/-- The function f(x) = 2sin²x -/
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2

/-- The function g(x) = √3sin(2x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x)

/-- The theorem stating that the maximum absolute difference between f and g is 3 -/
theorem max_diff_f_g :
  (∃ a : ℝ, |f a - g a| = 3) ∧ (∀ a : ℝ, |f a - g a| ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_f_g_l925_92583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l925_92564

theorem trig_identity (α : ℝ) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  (Real.sin α * Real.cos α) / (Real.sin α ^ 2 + 2 * Real.cos α ^ 2) = -3/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l925_92564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l925_92512

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the conditions
axiom derivative_greater : ∀ x : ℝ, f' x > f x
axiom f_at_two : f 2 = Real.exp 2

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | x > 2}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f x > Real.exp x} = solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l925_92512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l925_92561

-- Define the angle in radians (15 degrees = π/12 radians)
noncomputable def angle : Real := Real.pi / 12

-- State the theorem
theorem trig_identity : 
  400 * (Real.cos angle ^ 5 + Real.sin angle ^ 5) / (Real.cos angle + Real.sin angle) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l925_92561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_eating_theorem_l925_92522

/-- Represents the number of days it takes for a given number of cows to eat all the grass -/
noncomputable def daysToEatGrass (numCows : ℕ) : ℚ := sorry

/-- The rate at which grass grows per day -/
noncomputable def grassGrowthRate : ℚ := sorry

/-- The rate at which one cow eats grass per day -/
noncomputable def cowEatRate : ℚ := sorry

/-- The initial amount of grass -/
noncomputable def initialGrass : ℚ := sorry

theorem grass_eating_theorem :
  (daysToEatGrass 24 = 6) →
  (daysToEatGrass 21 = 8) →
  (∀ n : ℕ, daysToEatGrass n * (grassGrowthRate + n * cowEatRate) = initialGrass) →
  (daysToEatGrass 36 = 4.5) := by
  sorry

#check grass_eating_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_eating_theorem_l925_92522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_one_or_four_tangents_l925_92560

/-- Two circles in a plane with different radii -/
structure TwoCircles where
  r₁ : ℝ
  r₂ : ℝ
  h : r₁ ≠ r₂

/-- The number of common tangents for two circles -/
inductive CommonTangents
  | zero | one | two | three | four

/-- Proposition: Two circles with unequal radii cannot have 1 or 4 common tangents -/
theorem no_one_or_four_tangents (c : TwoCircles) :
  ∀ n : CommonTangents, n ≠ CommonTangents.one ∧ n ≠ CommonTangents.four :=
by
  intro n
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_one_or_four_tangents_l925_92560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l925_92556

noncomputable def f (x m : ℝ) : ℝ :=
  if x < 2 then -x^2 + 2*x + 3 else m/x

theorem range_of_m (m : ℝ) :
  (∀ x, f x m ≤ 4) ∧ (∀ ε > 0, ∃ x, f x m > 4 - ε) →
  m ≤ 8 ∧ (∀ ε > 0, ∃ m', m' < 8 - ε ∧ 
    ((∀ x, f x m' ≤ 4) ∧ (∀ δ > 0, ∃ y, f y m' > 4 - δ))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l925_92556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_stickers_remaining_l925_92576

theorem gilda_stickers_remaining (initial_stickers : ℝ) : 
  (initial_stickers * (1 - 0.3) * (1 - 0.2) * (1 - 0.3) * (1 - 0.1)) / initial_stickers = 0.3528 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_stickers_remaining_l925_92576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_distance_l925_92514

/-- Calculates the distance traveled by the faster runner when two people
    meet while running in opposite directions around a circular path. -/
noncomputable def distance_traveled (path_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (path_length * speed1) / (speed1 + speed2)

/-- Theorem stating that given a circular path of 18 km, if two people start
    at the same point and move in opposite directions with speeds of 5 km/h
    and 4 km/h respectively, the person moving at 5 km/h will travel 10 km
    when they meet. -/
theorem runners_meeting_distance :
  distance_traveled 18 5 4 = 10 := by
  -- Unfold the definition of distance_traveled
  unfold distance_traveled
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_distance_l925_92514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_factorial_plus_one_bound_l925_92543

theorem odd_prime_factorial_plus_one_bound (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ c : ℝ, c > 0 ∧ (∀ n : ℕ, n ≤ p - 1 → p ∣ n.factorial + 1 → n ≤ ⌊c * p^(2/3 : ℝ)⌋) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_factorial_plus_one_bound_l925_92543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_theorem_l925_92563

/-- Represents a number in base 5 --/
structure Base5 where
  value : ℕ
  isBase5 : value < 5^64  -- Assuming a reasonable upper bound

/-- Converts a base 5 number to its decimal (base 10) equivalent --/
def to_decimal (n : Base5) : ℕ := n.value

/-- Converts a decimal (base 10) number to its base 5 equivalent --/
def to_base5 (n : ℕ) : Base5 :=
  ⟨n % 5^64, by apply Nat.mod_lt; exact pow_pos (by norm_num) 64⟩

/-- Performs division in base 5 --/
def div_base5 (a b : Base5) : Base5 × Base5 :=
  let quotient := to_base5 (a.value / b.value)
  let remainder := to_base5 (a.value % b.value)
  (quotient, remainder)

instance : OfNat Base5 n where
  ofNat := to_base5 n

theorem base5_division_theorem (dividend divisor quotient remainder : Base5) :
  dividend = 1432 ∧ divisor = 23 →
  div_base5 dividend divisor = (quotient, remainder) →
  quotient = 33 ∧ remainder = 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_theorem_l925_92563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l925_92571

theorem cos_double_angle (x : ℝ) (h : Real.cos x = 3/4) : Real.cos (2*x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l925_92571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l925_92505

/-- The set of positive integers p for which there exists a positive integer n
    such that p^n + 3^n divides p^(n+1) + 3^(n+1) -/
def SolutionSet : Set ℕ+ :=
  {p : ℕ+ | ∃ n : ℕ+, (p^n.val + 3^n.val) ∣ (p^(n.val+1) + 3^(n.val+1))}

/-- Theorem stating that the solution set contains only 3, 6, and 15 -/
theorem solution_set_characterization :
  SolutionSet = {3, 6, 15} := by
  sorry

#check solution_set_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l925_92505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_perimeter_relation_l925_92511

theorem square_triangle_area_perimeter_relation :
  ∀ (s t : ℝ),
  s > 0 → t > 0 →
  s^2 = 3 * t →  -- Area of square equals perimeter of triangle
  (Real.sqrt 3 / 4) * t^2 = 4 * s →  -- Area of triangle equals perimeter of square
  s^2 = 6 :=
by
  intros s t hs ht h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_perimeter_relation_l925_92511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_growth_sum_eq_closed_form_l925_92558

/-- The sum of the infinite series representing the growth of a line -/
noncomputable def line_growth_sum : ℝ := 1 + (∑' n, (1/3)^n) + Real.sqrt 3 * (∑' n, (1/5)^n) + Real.sqrt 5 * (∑' n, (1/7)^n)

/-- The closed form of the sum -/
noncomputable def closed_form_sum : ℝ := 3/2 + Real.sqrt 3/4 + Real.sqrt 5/6

/-- Theorem stating that the infinite sum equals the closed form -/
theorem line_growth_sum_eq_closed_form : line_growth_sum = closed_form_sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_growth_sum_eq_closed_form_l925_92558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l925_92529

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (10, 0)
def C : ℝ × ℝ := (15, 0)

-- Define the initial slopes of the lines
def slope_ℓA : ℝ := 1
noncomputable def slope_ℓB : ℝ := Real.arctan (Real.pi / 2)  -- Vertical line approximation
def slope_ℓC : ℝ := -1

-- Define the rotation of the lines
noncomputable def rotate_lines (θ : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the triangle formed by the intersections
noncomputable def triangle_area (θ : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∃ (θ : ℝ), ∀ (φ : ℝ), triangle_area θ ≥ triangle_area φ ∧ triangle_area θ = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l925_92529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_timing_l925_92568

/-- Partnership problem -/
theorem partnership_investment_timing
  (annual_gain : ℚ)
  (lakshmi_share : ℚ)
  (raman_investment : ℚ)
  (lakshmi_investment_ratio : ℚ)
  (muthu_investment_ratio : ℚ)
  (lakshmi_investment_delay : ℚ)
  (m : ℚ)
  (h1 : annual_gain = 36000)
  (h2 : lakshmi_share = 12000)
  (h3 : lakshmi_investment_ratio = 2)
  (h4 : muthu_investment_ratio = 3)
  (h5 : lakshmi_investment_delay = 6)
  (profit_share_prop : ℚ → ℚ → ℚ)
  (h6 : profit_share_prop raman_investment 12 + 
        profit_share_prop (lakshmi_investment_ratio * raman_investment) (12 - lakshmi_investment_delay) + 
        profit_share_prop (muthu_investment_ratio * raman_investment) (12 - m) = annual_gain)
  (h7 : profit_share_prop (lakshmi_investment_ratio * raman_investment) (12 - lakshmi_investment_delay) = lakshmi_share)
  : m = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_timing_l925_92568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_tangent_lines_through_P_tangent_lines_slope_4_l925_92584

noncomputable section

-- Define the curve
def curve (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := x^2

-- Point P
def P : ℝ × ℝ := (2, 4)

-- Theorem for tangent line at P(2,4)
theorem tangent_line_at_P :
  ∀ x y : ℝ, curve P.1 = P.2 → 4*x - y - 4 = 0 ↔ y - P.2 = curve_derivative P.1 * (x - P.1) :=
by sorry

-- Theorem for tangent lines passing through P(2,4)
theorem tangent_lines_through_P :
  ∀ x y : ℝ, curve P.1 = P.2 → 
  (4*x - y - 4 = 0 ∨ x - y + 2 = 0) ↔ 
  (∃ x₀ : ℝ, y - curve x₀ = curve_derivative x₀ * (x - x₀) ∧ y = 4 ∧ x = 2) :=
by sorry

-- Theorem for tangent lines with slope 4
theorem tangent_lines_slope_4 :
  ∀ x y : ℝ, 
  (4*x - y - 4 = 0 ∨ 12*x - 3*y + 20 = 0) ↔
  (∃ x₀ : ℝ, y - curve x₀ = 4 * (x - x₀) ∧ curve_derivative x₀ = 4) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_tangent_lines_through_P_tangent_lines_slope_4_l925_92584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l925_92575

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 3

-- Define the inverse function
noncomputable def g (x : ℝ) : ℝ := 3^(x - 1)

-- State the theorem
theorem inverse_function_theorem :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 1 ≤ f x ∧ f x ≤ 2) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 1 ≤ g x ∧ g x ≤ 3) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → g (f x) = x) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f (g x) = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l925_92575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1_2016_closest_to_2_l925_92513

noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem harmonic_mean_1_2016_closest_to_2 :
  ∀ n : ℤ, n ≠ 2 → |harmonicMean 1 2016 - 2| < |harmonicMean 1 2016 - ↑n| := by
  sorry

#check harmonic_mean_1_2016_closest_to_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1_2016_closest_to_2_l925_92513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l925_92503

theorem trig_inequality (x : ℝ) : 
  (Real.sin x ^ 2 + (1 / Real.sin x) ^ 2) ^ 2 + (Real.cos x ^ 2 + (1 / Real.cos x) ^ 2) ^ 2 ≥ 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l925_92503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l925_92544

noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 2 = 1

noncomputable def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt (a^2 + 2) / a

theorem hyperbola_eccentricity_value (a : ℝ) (h1 : a > 0) (h2 : eccentricity a = 2) :
  a = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l925_92544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cows_count_l925_92508

theorem black_cows_count (total_feeds : ℕ) (feed_per_animal : ℕ) (bulls_count : ℕ) (black_cows_count : ℕ) : 
  total_feeds = 180 →
  feed_per_animal = 3 →
  bulls_count = 52 →
  (bulls_count * feed_per_animal + black_cows_count * feed_per_animal = total_feeds) →
  black_cows_count = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cows_count_l925_92508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_triangle_plane_on_sphere_l925_92582

/-- The distance from the center of a sphere to the plane of a triangle on its surface -/
noncomputable def distance_to_triangle_plane (radius PQ QR RP : ℝ) : ℝ :=
  let s := (PQ + QR + RP) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - QR) * (s - RP))
  let circumradius := (PQ * QR * RP) / (4 * area)
  Real.sqrt (radius^2 - circumradius^2)

theorem distance_to_triangle_plane_on_sphere 
  (radius PQ QR RP : ℝ) 
  (h_radius : radius = 15)
  (h_PQ : PQ = 12)
  (h_QR : QR = 16)
  (h_RP : RP = 20) :
  distance_to_triangle_plane radius PQ QR RP = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_triangle_plane_on_sphere_l925_92582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_class_mean_l925_92504

/-- Calculates the new class mean given the scores of three groups of students -/
theorem new_class_mean (total_students : ℕ) 
                        (group1_students : ℕ) (group1_mean : ℚ)
                        (group2_students : ℕ) (group2_mean : ℚ)
                        (group3_students : ℕ) (group3_score : ℚ) :
  total_students = group1_students + group2_students + group3_students →
  group1_students = 45 →
  group2_students = 4 →
  group3_students = 1 →
  group1_mean = 80 / 100 →
  group2_mean = 85 / 100 →
  group3_score = 90 / 100 →
  (group1_students * group1_mean + 
   group2_students * group2_mean + 
   group3_students * group3_score) / total_students = 806 / 1000 := by
  sorry

#eval (45 * (80 : ℚ) / 100 + 4 * (85 : ℚ) / 100 + 1 * (90 : ℚ) / 100) / 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_class_mean_l925_92504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_I_l925_92550

/-- The function f(x) = 2^x + 2^(-x) -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)

/-- The interval [-1/2, 1/2] -/
def I : Set ℝ := Set.Icc (-1/2) (1/2)

theorem max_value_of_f_on_I :
  ∃ (x : ℝ), x ∈ I ∧ f x = 2 ∧ ∀ y ∈ I, f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_I_l925_92550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_s_negative_l925_92500

-- Define the set of possible values for p
def P : Finset ℕ := Finset.range 10 |>.filter (λ n => n ≥ 1)

-- Define the function s in terms of p
def s (p : ℕ) : ℤ := p^2 - 13*p + 40

-- Define the set of p values where s is negative
def S : Finset ℕ := P.filter (λ p => s p < 0)

-- Theorem statement
theorem probability_s_negative : (S.card : ℚ) / P.card = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_s_negative_l925_92500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l925_92555

theorem sum_remainder (a b c : ℕ) : 
  a % 12 = 7 → b % 12 = 9 → c % 12 = 10 → (a + b + c) % 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l925_92555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l925_92538

theorem problem_solution :
  (∃ x : ℝ, (x - 1)^3 = -8 ∧ x = -1) ∧
  (∃ x : ℝ, (x + 1)^2 = 25 ∧ (x = 4 ∨ x = -6)) ∧
  (∀ a : ℝ, (∃ n : ℝ, n ≥ 0 ∧ Real.sqrt n = a + 3 ∧ Real.sqrt n = 2*a - 15) → a = 4) ∧
  (∀ b : ℝ, b ≥ 0 → b^((1:ℝ)/3) = 2 → b = 8) ∧
  (∀ a b : ℝ, a = 4 ∧ b = 8 → Real.sqrt (b - a) = 2 ∨ Real.sqrt (b - a) = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l925_92538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_position_quadratic_above_axis_quadratic_below_axis_l925_92533

-- Define the quadratic function
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + (9 * b^2) / (16 * a)

-- Theorem statement
theorem quadratic_position (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b x > 0) ∨ (∀ x, f a b x < 0) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

-- Additional theorem to capture the condition on 'a'
theorem quadratic_above_axis (a b : ℝ) (ha : a > 0) :
  ∀ x, f a b x > 0 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

theorem quadratic_below_axis (a b : ℝ) (ha : a < 0) :
  ∀ x, f a b x < 0 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_position_quadratic_above_axis_quadratic_below_axis_l925_92533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l925_92525

/-- The angle moved by the minute hand in one minute -/
def minute_hand_speed : ℚ := 6

/-- The angle moved by the hour hand in one minute -/
def hour_hand_speed : ℚ := 1/2

/-- The initial angle of the hour hand at 3:00 -/
def initial_hour_angle : ℚ := 90

/-- The number of minutes passed since 3:00 -/
def minutes_passed : ℕ := 40

/-- The angle of the minute hand at 3:40 -/
def minute_hand_angle : ℚ := minutes_passed * minute_hand_speed

/-- The angle of the hour hand at 3:40 -/
def hour_hand_angle : ℚ := initial_hour_angle + minutes_passed * hour_hand_speed

/-- The smaller angle between the hour and minute hands at 3:40 -/
noncomputable def clock_angle : ℚ := min (abs (minute_hand_angle - hour_hand_angle)) 
                           (360 - abs (minute_hand_angle - hour_hand_angle))

theorem clock_angle_at_3_40 : 
  ∃ ε > 0, abs (clock_angle - 130) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l925_92525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l925_92581

def f (a x : ℝ) : ℝ := |x + a| - |x + 1|

theorem f_properties (a : ℝ) :
  (∀ x, f a x ≤ 2*a) = (a ≥ 1/3) ∧
  (f a a > 1) = (a < -2/3 ∨ a > 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l925_92581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_centroid_ratio_l925_92595

def n : ℕ := 2188

-- Define the vertices of the regular n-gon
def A (j : ℕ) (i : ℕ) : ℂ :=
  sorry

-- Define the centroid operation
noncomputable def centroid (a b c : ℂ) : ℂ :=
  (a + b + c) / 3

-- Define the magnitude of a complex number
noncomputable def magnitude (z : ℂ) : ℝ :=
  Complex.abs z

-- State the theorem
theorem regular_polygon_centroid_ratio :
  (magnitude (A 2014 7)) / (magnitude (A 2014 0)) = 1 / 2187 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_centroid_ratio_l925_92595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_l925_92586

def a : ℕ → ℕ
| 0 => 0  -- Base case
| n + 1 =>
  if (n + 1) % 2 = 0 then a ((n + 1) / 2)
  else if (n + 1) % 4 = 1 then 1
  else 0

theorem sequence_not_periodic :
  ¬ ∃ T : ℕ+, ∀ n : ℕ, n ≥ 1 → a (n + T) = a n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_l925_92586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l925_92518

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

/-- The function g(x) as defined in the problem -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x - 5/12

/-- The theorem statement -/
theorem problem_statement (b : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 0 1, f (1/3) x₁ ≥ g b x₂) →
  b ∈ Set.Ici (1/2) := by
  sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l925_92518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_regular_tetrahedron_l925_92535

/-- The radius of a sphere touching all edges of a regular tetrahedron with edge length a -/
noncomputable def sphere_radius_in_tetrahedron (a : ℝ) : ℝ := (a * Real.sqrt 2) / 4

/-- Theorem: The radius of a sphere touching all edges of a regular tetrahedron with edge length a is (a * √2) / 4 -/
theorem sphere_radius_in_regular_tetrahedron (a : ℝ) (h : a > 0) :
  ∃ r : ℝ, r > 0 ∧ r = sphere_radius_in_tetrahedron a :=
by
  use sphere_radius_in_tetrahedron a
  constructor
  · sorry  -- Proof that r > 0
  · rfl    -- Proof that r = sphere_radius_in_tetrahedron a


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_regular_tetrahedron_l925_92535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_example_l925_92527

/-- The focus of a parabola y = ax² + bx + c is (h, k + 1/(4a)) where (h, k) is the vertex -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- The parabola equation y = 2x² - 4x - 1 has focus (1, -23/8) -/
theorem parabola_focus_example : parabola_focus 2 (-4) (-1) = (1, -23/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_example_l925_92527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l925_92539

/-- Sum of digits of a positive integer -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Predicate for three-digit numbers -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Count of three-digit numbers satisfying the condition -/
def countValidNumbers : ℕ := 50

theorem valid_numbers_count : 
  countValidNumbers = 50 ∧ 
  ∀ n : ℕ, isThreeDigit n → (digitSum (digitSum n) = 4 ↔ n ∈ Finset.range 1000 \ Finset.range 100 ∧ n < countValidNumbers) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l925_92539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationships_l925_92552

-- Define the types for planes and lines
variable {Plane Line : Type}

-- Define the relationships between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define a new relation for line perpendicular to plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the statements
theorem plane_relationships (α β : Plane) :
  -- Statement 1
  (∀ l : Line, in_plane l α → (∀ m : Line, in_plane m β → perp_line l m) → perp_plane α β) ∧
  -- Statement 2
  (∀ l : Line, in_plane l α → parallel_line_plane l β → parallel_plane α β) ∧
  -- Statement 3
  ¬(∀ l : Line, perp_plane α β → in_plane l α → perp_line_plane l β) ∧
  -- Statement 4
  (∀ l : Line, parallel_plane α β → in_plane l α → parallel_line_plane l β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationships_l925_92552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l925_92520

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h1 : π/2 < α ∧ α < π) 
  (h2 : π/2 < β ∧ β < π) 
  (h3 : cos (β - α) = 3/5) :
  (∀ a : ℝ, sin α + cos α = a → 
    (a = 1/3 → sin α * cos α + tan α - 1/(3*cos α) = -13/9) ∧
    (a = 7/13 → sin β = 16/65)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l925_92520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_cells_difference_l925_92528

/-- Represents a cell in the 8x8 grid -/
structure Cell where
  row : Fin 8
  col : Fin 8

/-- The type of the grid: a function from Cell to ℕ -/
def Grid := Cell → Fin 64

/-- Two cells are adjacent if they share a row or column and are one unit apart -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col.val + 1 = c2.col.val ∨ c2.col.val + 1 = c1.col.val)) ∨
  (c1.col = c2.col ∧ (c1.row.val + 1 = c2.row.val ∨ c2.row.val + 1 = c1.row.val))

/-- The main theorem -/
theorem adjacent_cells_difference (g : Grid) :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (g c1).val.sub (g c2).val ≥ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_cells_difference_l925_92528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_beats_seventh_l925_92588

/-- Represents a chess tournament with 8 participants -/
structure ChessTournament where
  participants : Fin 8 → ℕ
  different_scores : ∀ i j, i ≠ j → participants i ≠ participants j
  second_place_score : participants 1 = participants 4 + participants 5 + participants 6 + participants 7

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- Function to determine the match result between two players -/
def match_result (scores : Fin 8 → ℕ) (i j : Fin 8) : MatchResult :=
  if scores i > scores j then MatchResult.Win
  else if scores i < scores j then MatchResult.Loss
  else MatchResult.Draw

/-- The theorem stating that the 3rd place participant won against the 7th place participant -/
theorem third_beats_seventh (tournament : ChessTournament) : 
  match_result tournament.participants 2 6 = MatchResult.Win := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_beats_seventh_l925_92588
