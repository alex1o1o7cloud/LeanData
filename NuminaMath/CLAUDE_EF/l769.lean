import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l769_76919

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) →
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l769_76919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l769_76984

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def perpendicularLP (l : Line) (p : Plane) : Prop := sorry
def notContained (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_plane (m n : Line) (α : Plane) :
  perpendicular m n → perpendicularLP n α → notContained m α → parallel m α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l769_76984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l769_76987

/-- Hyperbola Γ: (x²/a²) - (y²/b²) = 1 (a > 0, b > 0) with focal distance 2c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_focal_distance : c^2 = a^2 + b^2

/-- Line l: y = kx - kc -/
noncomputable def line (k : ℝ) (c : ℝ) : ℝ → ℝ := λ x ↦ k * (x - c)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- Theorem stating the range of eccentricity for the given hyperbola -/
theorem eccentricity_range (h : Hyperbola) :
  (∃ (x₁ y₁ : ℝ), x₁^2 / h.a^2 - y₁^2 / h.b^2 = 1 ∧ y₁ = line (Real.sqrt 3) h.c x₁) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 / h.a^2 - y₂^2 / h.b^2 = 1 ∧ y₂ = line (Real.sqrt 3) h.c x₂) ∧
  (∃ (x₃ y₃ x₄ y₄ : ℝ), x₃^2 / h.a^2 - y₃^2 / h.b^2 = 1 ∧ y₃ = line (Real.sqrt 15) h.c x₃ ∧
                        x₄^2 / h.a^2 - y₄^2 / h.b^2 = 1 ∧ y₄ = line (Real.sqrt 15) h.c x₄ ∧
                        x₃ ≠ x₄ ∧ x₃ > 0 ∧ x₄ > 0) →
  2 < eccentricity h ∧ eccentricity h < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l769_76987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_negative_root_l769_76904

-- Define the equation (*)
def equation (t a : ℝ) : Prop := (t - 2/a)^2 + (t + 2/a)^2 = 6

-- Define the solution for x based on intervals of a
noncomputable def solution (a : ℝ) : ℝ :=
  if a < 0 then ((2 - Real.sqrt (4 - 6*a + 2*a^2))/a)^2
  else if a = 0 then 2.25
  else if 1 < a ∧ a < 2 then (2*a - 6)/a
  else if 3 < a then ((2 + Real.sqrt (4 - 6*a + 2*a^2))/a)^2
  else 0  -- For other cases

-- Theorem statement
theorem unique_non_negative_root (a : ℝ) : 
  (a < 0 ∨ a = 0 ∨ (1 < a ∧ a < 2) ∨ 3 < a) → 
  ∃! x : ℝ, x ≥ 0 ∧ equation (Real.sqrt x) a ∧ x = solution a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_negative_root_l769_76904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l769_76950

noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

def line_l_polar (ρ θ : ℝ) : Prop :=
  (Real.sqrt 2 / 2) * ρ * Real.cos (θ + Real.pi / 4) = -1

def line_l_cartesian (x y : ℝ) : Prop := x - y + 2 = 0

def point_M : ℝ × ℝ := (-1, 0)

theorem curve_and_line_properties :
  (∀ x y : ℝ, (∃ α : ℝ, curve_C α = (x, y)) ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ x y : ℝ, (∃ ρ θ : ℝ, line_l_polar ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ line_l_cartesian x y) ∧
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, A.1 = -1 + (Real.sqrt 2 / 2) * t ∧ A.2 = (Real.sqrt 2 / 2) * t) ∧
    (∃ t : ℝ, B.1 = -1 + (Real.sqrt 2 / 2) * t ∧ B.2 = (Real.sqrt 2 / 2) * t) ∧
    (∃ α : ℝ, curve_C α = A) ∧
    (∃ α : ℝ, curve_C α = B) ∧
    (A.1 - point_M.1)^2 + (A.2 - point_M.2)^2 * 
    (B.1 - point_M.1)^2 + (B.2 - point_M.2)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l769_76950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reporters_not_covering_politics_l769_76945

/-- The percentage of reporters covering local politics in country A -/
noncomputable def cover_A : ℚ := 20

/-- The percentage of reporters covering local politics in country B -/
noncomputable def cover_B : ℚ := 25

/-- The percentage of reporters covering local politics in country C -/
noncomputable def cover_C : ℚ := 15

/-- The percentage of reporters who cover politics but not local politics in A, B, or C -/
noncomputable def cover_other : ℚ := 20

/-- The total percentage of reporters covering local politics in A, B, or C -/
noncomputable def total_local : ℚ := cover_A + cover_B + cover_C

/-- The total percentage of reporters covering politics (local or otherwise) -/
noncomputable def total_politics : ℚ := total_local / (80 / 100)

theorem reporters_not_covering_politics :
  100 - total_politics = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reporters_not_covering_politics_l769_76945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_ratio_l769_76942

-- Define the original function g
noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x - 4)

-- Define the inverse function g_inv
noncomputable def g_inv (x : ℝ) : ℝ := (4 * x - 2) / (3 - x)

-- Theorem statement
theorem inverse_function_and_ratio :
  (∀ x : ℝ, x ≠ 4 → g (g_inv x) = x) ∧
  (∀ x : ℝ, x ≠ 3 → g_inv (g x) = x) ∧
  (4 / -1 : ℝ) = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_ratio_l769_76942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_weight_theorem_l769_76920

/-- The weight of soup after four days of reduction -/
def soupWeightAfterFourDays (initialWeight : ℝ) (reduction1 reduction2 reduction3 reduction4 : ℝ) : ℝ :=
  initialWeight * (1 - reduction1) * (1 - reduction2) * (1 - reduction3) * (1 - reduction4)

/-- Theorem stating the final weight of soup after four days of specific reductions -/
theorem soup_weight_theorem :
  let initialWeight : ℝ := 80
  let reduction1 : ℝ := 0.40
  let reduction2 : ℝ := 0.35
  let reduction3 : ℝ := 0.55
  let reduction4 : ℝ := 0.50
  abs (soupWeightAfterFourDays initialWeight reduction1 reduction2 reduction3 reduction4 - 7.02) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_weight_theorem_l769_76920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_rounds_sufficient_l769_76948

/-- Represents a swap operation between two positions -/
structure Swap where
  pos1 : Nat
  pos2 : Nat

/-- Represents a round of swaps -/
def Round := List Swap

/-- Checks if a list of swaps is valid (no athlete participates in more than one swap) -/
def validRound (r : Round) : Bool :=
  (r.map (fun s => [s.pos1, s.pos2])).join.Nodup

/-- Applies a round of swaps to a permutation -/
def applyRound (perm : List Nat) (r : Round) : List Nat :=
  sorry

/-- Checks if a permutation is in ascending order -/
def isSorted (perm : List Nat) : Bool :=
  sorry

theorem two_rounds_sufficient (n : Nat) :
  ∀ (perm : List Nat), perm.length = n → perm.Nodup →
  ∃ (r1 r2 : Round), validRound r1 ∧ validRound r2 ∧
    isSorted (applyRound (applyRound perm r1) r2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_rounds_sufficient_l769_76948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_balls_hypergeometric_prob_four_black_balls_l769_76964

open BigOperators Finset Real

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of black balls
def black_balls : ℕ := 6

-- Define the number of white balls
def white_balls : ℕ := 4

-- Define the number of balls drawn
def drawn_balls : ℕ := 4

-- Define the hypergeometric distribution
noncomputable def hypergeometric (population : ℕ) (success : ℕ) (sample : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose success k * Nat.choose (population - success) (sample - k)) / Nat.choose population sample

-- Theorem 1: The number of black balls drawn follows a hypergeometric distribution
theorem black_balls_hypergeometric :
  ∀ k : ℕ, k ≤ drawn_balls →
    hypergeometric total_balls black_balls drawn_balls k =
    (Nat.choose black_balls k * Nat.choose white_balls (drawn_balls - k)) / Nat.choose total_balls drawn_balls :=
by
  sorry

-- Theorem 2: The probability of drawing 4 black balls is 1/14
theorem prob_four_black_balls :
  hypergeometric total_balls black_balls drawn_balls drawn_balls = 1 / 14 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_balls_hypergeometric_prob_four_black_balls_l769_76964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straws_per_piglet_l769_76930

def total_straws : ℕ := 300
def adult_pig_fraction : ℚ := 3/5
def piglet_fraction : ℚ := 1/3
def num_piglets : ℕ := 20

theorem straws_per_piglet :
  (↑total_straws * piglet_fraction).num / (↑num_piglets * piglet_fraction.den) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_straws_per_piglet_l769_76930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l769_76954

open BigOperators

def binomial_expansion (n : ℕ) : ℕ → ℚ → ℚ
| r, k => (n.choose r) * (3^r) * k^(n-r)

def sum_of_coefficients (n : ℕ) : ℚ := 4^n

def sum_of_binomial_coefficients (n : ℕ) : ℚ := 2^n

theorem expansion_properties (n : ℕ) 
  (h : sum_of_coefficients n - sum_of_binomial_coefficients n = 992) :
  n = 5 ∧ 
  binomial_expansion n 2 (3 : ℚ) = 90 ∧
  (binomial_expansion n 2 (3 : ℚ) = 90 ∧ binomial_expansion n 3 (3 : ℚ) = 270) ∧
  binomial_expansion n 4 (3 : ℚ) = 405 ∧
  ∀ k : ℕ, binomial_expansion n k (3 : ℚ) ≤ binomial_expansion n 4 (3 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l769_76954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l769_76913

open Real

/-- The function f we're analyzing -/
noncomputable def f : ℝ → ℝ := sorry

/-- Condition 1: The tangent line equation of f at (1, f(1)) is y = x - 1 -/
axiom tangent_line : f 1 = 0 ∧ deriv f 1 = 1

/-- Condition 2: f'(x) = ln x + 1 for all x > 0 -/
axiom derivative_f : ∀ x > 0, deriv f x = log x + 1

/-- Theorem: The minimum value of f occurs at 1/e and equals -1/e -/
theorem min_value_f :
  ∃ x > 0, ∀ y > 0, f x ≤ f y ∧ f x = -1/exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l769_76913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_m_onto_n_l769_76955

def m : Fin 3 → ℝ := ![2, -4, 1]
def n : Fin 3 → ℝ := ![2, -1, 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

noncomputable def projection (v w : Fin 3 → ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

theorem projection_m_onto_n :
  projection m n = 10/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_m_onto_n_l769_76955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l769_76968

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | n + 2 => (1/16) * (1 + 4 * sequence_a (n + 1) + Real.sqrt (1 + 24 * sequence_a (n + 1)))

theorem sequence_a_general_term (n : ℕ) (h : n ≥ 1) :
  sequence_a n = (1/3) + (1/2)^n + (1/3) * ((1/2)^(2*n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l769_76968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_is_30_percent_l769_76918

/-- Represents the profit percentage of a dishonest dealer who uses a lighter weight. -/
noncomputable def dishonest_dealer_profit (actual_weight : ℝ) (standard_weight : ℝ) : ℝ :=
  (standard_weight - actual_weight) / standard_weight * 100

/-- Theorem stating that a dealer using 700 grams instead of 1 kg makes 30% profit. -/
theorem dealer_profit_is_30_percent :
  dishonest_dealer_profit 700 1000 = 30 := by
  -- Unfold the definition of dishonest_dealer_profit
  unfold dishonest_dealer_profit
  -- Simplify the arithmetic
  simp [sub_div, mul_div_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_is_30_percent_l769_76918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l769_76962

open Real

noncomputable section

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * t.a * t.c * sin t.B

theorem triangle_properties (t : Triangle) 
  (h : t.b * sin t.A = t.a * cos t.B * sqrt 3) :
  t.B = π/3 ∧ 
  ((t.b = 3 ∧ Triangle.area t = 9 * sqrt 3 / 4) → t.a + t.c = 6) ∧
  ((t.b = 3 ∧ t.a + t.c = 6) → Triangle.area t = 9 * sqrt 3 / 4) ∧
  ((Triangle.area t = 9 * sqrt 3 / 4 ∧ t.a + t.c = 6) → t.b = 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l769_76962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_not_always_true_prop_3_not_always_true_prop_4_l769_76905

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- Proposition ①
theorem prop_1 (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α := by sorry

-- Proposition ②
theorem prop_2_not_always_true : 
  ¬(∀ (h1 : plane_parallel α β) (h2 : ¬line_in_plane m α) (h3 : ¬line_in_plane n β), 
    perpendicular n α) := by sorry

-- Proposition ③
theorem prop_3_not_always_true : 
  ¬(∀ (h1 : parallel m n) (h2 : line_parallel_plane m α), line_parallel_plane n α) := by sorry

-- Proposition ④
theorem prop_4 (h1 : plane_parallel α β) (h2 : parallel m n) (h3 : perpendicular m α) : 
  perpendicular n β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_not_always_true_prop_3_not_always_true_prop_4_l769_76905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l769_76906

theorem inequality_proof (a b c : ℝ) : 
  a = 2 * Real.sqrt (Real.exp 1) ∧ 
  b = 2 / Real.log 2 ∧ 
  c = (Real.exp 1)^2 / (4 - Real.log 4) →
  c < b ∧ b < a := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l769_76906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l769_76958

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.tan A = 1/4 →
  Real.tan B = 3/5 →
  a = Real.sqrt 17 →
  a ≥ b ∧ a ≥ c →
  -- Side-angle relationships
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Conclusions
  C = 3*π/4 ∧ min b c = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l769_76958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_l769_76947

/-- The solution set of the inequality ax^2+(a-1)x+(a-2)<0 is (-∞, -1) ∪ (2, ∞) -/
def has_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 + (a-1)*x + (a-2) < 0 ↔ x < -1 ∨ x > 2

/-- The value of a that satisfies the given solution set is 1 -/
theorem solution_value : ∃ a : ℝ, has_solution_set a ∧ a = 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_l769_76947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_correctness_l769_76980

theorem equation_correctness : ∃ (S : Finset (Fin 4)), 
  S.card = 2 ∧ 
  (∀ i ∈ S, i = 0 → ∀ a : ℝ, a > 0 → 6 * a^(2/3) * 7 * a^(1/2) = 42 * a^(7/6)) ∧
  (∀ i ∈ S, i = 1 → ∀ a x : ℝ, (-a*x)^6 / (-a*x^3) = a^5 * x^3) ∧
  (∀ i ∈ S, i = 2 → (-1989^0 : ℝ)^1989 = -1) ∧
  (∀ i ∈ S, i = 3 → ∀ m : ℤ, ((-3 : ℝ)^m)^2 = 3^(m^2)) ∧
  (∀ i ∉ S, i = 0 → ∃ a : ℝ, a > 0 ∧ 6 * a^(2/3) * 7 * a^(1/2) ≠ 42 * a^(7/6)) ∧
  (∀ i ∉ S, i = 1 → ∃ a x : ℝ, (-a*x)^6 / (-a*x^3) ≠ a^5 * x^3) ∧
  (∀ i ∉ S, i = 2 → (-1989^0 : ℝ)^1989 ≠ -1) ∧
  (∀ i ∉ S, i = 3 → ∃ m : ℤ, ((-3 : ℝ)^m)^2 ≠ 3^(m^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_correctness_l769_76980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_forbidden_squares_for_win_l769_76990

/-- Represents the game board with size (2n+1) × (2n+1) -/
structure GameBoard (n : ℕ) where
  size : ℕ := 2 * n + 1

/-- Represents a player in the game -/
inductive Player
  | Andile
  | Zandre

/-- The result of the game -/
inductive GameResult
  | AndileWins
  | ZandreWins

/-- Represents the game state -/
structure GameState (n : ℕ) where
  board : GameBoard n
  forbidden : ℕ
  currentPlayer : Player

/-- Function to determine the game result -/
def determineGameResult (n : ℕ) (state : GameState n) : GameResult :=
  if state.forbidden ≥ 2 * n + 1 then
    GameResult.AndileWins
  else
    GameResult.ZandreWins

/-- The main theorem stating the minimum number of forbidden squares for Andile to win -/
theorem min_forbidden_squares_for_win (n : ℕ) :
  ∀ (state : GameState n),
    (state.forbidden < 2 * n + 1 → determineGameResult n state = GameResult.ZandreWins) ∧
    (state.forbidden ≥ 2 * n + 1 → determineGameResult n state = GameResult.AndileWins) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_forbidden_squares_for_win_l769_76990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l769_76979

-- Define the hyperbolas
def C₁ (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1
def C₂ (x y : ℝ) : Prop := y^2 / 2 - x^2 / 3 = 1

-- Define the asymptotes for C₁
noncomputable def asymptote_C₁ (x y : ℝ) : Prop := y = (Real.sqrt 6 / 3) * x ∨ y = -(Real.sqrt 6 / 3) * x

-- Define the asymptotes for C₂
noncomputable def asymptote_C₂ (x y : ℝ) : Prop := y = (Real.sqrt 6 / 3) * x ∨ y = -(Real.sqrt 6 / 3) * x

-- Define the focal length for C₁
noncomputable def focal_length_C₁ : ℝ := Real.sqrt 5

-- Define the focal length for C₂
noncomputable def focal_length_C₂ : ℝ := Real.sqrt 5

theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote_C₁ x y ↔ asymptote_C₂ x y) ∧
  focal_length_C₁ = focal_length_C₂ := by
  constructor
  · intro x y
    simp [asymptote_C₁, asymptote_C₂]
  · simp [focal_length_C₁, focal_length_C₂]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l769_76979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_unit_circle_l769_76982

theorem max_distance_on_unit_circle : 
  ∃ (max : ℝ), max = 3 ∧ 
  ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z + (Real.sqrt 3 : ℂ) + Complex.I) ≤ max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_unit_circle_l769_76982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_property_exists_l769_76916

theorem sum_property_exists (n : ℕ) (S : Finset ℕ) 
  (h1 : n > 1)
  (h2 : S.card = n + 1)
  (h3 : ∀ x, x ∈ S → x < 2 * n)
  (h4 : ∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b = c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_property_exists_l769_76916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_implies_equilateral_equilateral_implies_sixty_degrees_main_result_l769_76960

-- Define a triangle with rational side lengths
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + Real.sqrt 2)^2 = (t.b + Real.sqrt 2) * (t.c + Real.sqrt 2)

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem stating the main result
theorem triangle_condition_implies_equilateral (t : Triangle) :
  satisfiesCondition t → isEquilateral t :=
by sorry

-- Theorem connecting equilateral property to 60 degree angles
theorem equilateral_implies_sixty_degrees (t : Triangle) :
  isEquilateral t → 60 = 60 :=
by sorry

-- Main theorem combining the above results
theorem main_result (t : Triangle) :
  satisfiesCondition t → 60 = 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_implies_equilateral_equilateral_implies_sixty_degrees_main_result_l769_76960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_implies_b_bound_l769_76938

/-- The function f(x) with parameter b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (2*b + 1) * x^2 + b * (b + 1) * x

/-- Theorem stating that if f has a local minimum in (0,2), then -1 < b < 1 -/
theorem local_min_implies_b_bound (b : ℝ) :
  (∃ (c : ℝ), c ∈ Set.Ioo 0 2 ∧ IsLocalMin (f b) c) →
  b ∈ Set.Ioo (-1) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_implies_b_bound_l769_76938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_greater_than_quarter_l769_76926

noncomputable def R : ℕ → ℝ
  | 0 => Real.sqrt 2
  | n + 1 => Real.sqrt (2 + R n)

theorem fraction_greater_than_quarter (n : ℕ) (h : n ≥ 1) :
  (2 - R n) / (2 - R (n - 1)) > (1 / 4 : ℝ) := by
  sorry

#check fraction_greater_than_quarter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_greater_than_quarter_l769_76926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_equals_sqrt97_plus_sqrt85_l769_76992

/-- Given points A, B, and D in a 2D plane, prove that the sum of distances AD and BD is √97 + √85 -/
theorem distance_sum_equals_sqrt97_plus_sqrt85 :
  let A : ℝ × ℝ := (15, 0)
  let B : ℝ × ℝ := (0, 3)
  let D : ℝ × ℝ := (9, 7)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A D + distance B D = Real.sqrt 97 + Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_equals_sqrt97_plus_sqrt85_l769_76992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_result_l769_76976

def double_factorial : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => n * double_factorial n

def sum_ratio (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => (double_factorial (2 * i + 1) : ℚ) / (double_factorial (2 * i + 2) : ℚ))

theorem sum_ratio_result :
  ∃ (a b : ℕ), b % 2 = 1 ∧ sum_ratio 1011 = (1 : ℚ) / (2 ^ a * b) ∧ a * b / 10 = 1019 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_result_l769_76976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_permutations_l769_76928

/- Define the color type for chips -/
inductive Color : Type
  | Red : Color
  | Yellow : Color
  | Green : Color
  | Blue : Color

/- Define the circular arrangement of fields -/
def num_fields : Nat := 12

/- Define a function to represent valid moves -/
def is_valid_move (start : Nat) (finish : Nat) : Prop :=
  (finish - start) % num_fields = 4 ∨ (start - finish) % num_fields = 4

/- Define a type for chip arrangements -/
def ChipArrangement := Fin 4 → Color

/- Define the initial arrangement -/
def initial_arrangement : ChipArrangement :=
  fun i => match i with
    | 0 => Color.Red
    | 1 => Color.Yellow
    | 2 => Color.Green
    | 3 => Color.Blue

/- Define the set of possible final arrangements -/
def possible_arrangements : Set ChipArrangement :=
  { initial_arrangement,
    (fun i => [Color.Yellow, Color.Green, Color.Blue, Color.Red][i.val]),
    (fun i => [Color.Green, Color.Blue, Color.Red, Color.Yellow][i.val]),
    (fun i => [Color.Blue, Color.Red, Color.Yellow, Color.Green][i.val]) }

/- The main theorem -/
theorem chip_permutations :
  ∀ (final : ChipArrangement),
  (∃ (moves : List (Nat × Nat)),
    (∀ (move : Nat × Nat), move ∈ moves → is_valid_move move.fst move.snd) ∧
    (final ∈ possible_arrangements)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_permutations_l769_76928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l769_76977

noncomputable def options : List ℝ := [400, 1600, 1616, 2000, 3200]

noncomputable def target : ℝ := (404 + 1/4) / 0.25

theorem closest_to_target :
  ∀ x ∈ options, |target - 1616| ≤ |target - x| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l769_76977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_approx_l769_76993

/-- Calculates the cost price given the selling price and profit percentage -/
noncomputable def costPrice (sellingPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  sellingPrice / (1 + profitPercentage)

/-- The total cost price of three items -/
noncomputable def totalCostPrice (sp1 sp2 sp3 : ℝ) (profit1 profit2 profit3 : ℝ) : ℝ :=
  costPrice sp1 profit1 + costPrice sp2 profit2 + costPrice sp3 profit3

/-- Theorem stating that the total cost price of the three items is approximately $368.78 -/
theorem total_cost_price_approx :
  ∀ (ε : ℝ), ε > 0 →
  |totalCostPrice 100 120 200 0.15 0.20 0.10 - 368.78| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_approx_l769_76993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_l769_76991

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.log ((1 + x) / (1 - x)) / Real.log 10 + 5

-- State the theorem
theorem f_negative_a (a : ℝ) (ha : a ≠ 1) (ha2 : a ≠ -1) : 
  f a = 6 → f (-a) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_l769_76991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_income_theorem_l769_76903

/-- Represents Honey's financial data over two months -/
structure HoneyFinances where
  daily_income_range : Set ℝ
  exchange_rate : ℝ
  night_course_payment : ℝ
  savings : ℝ
  tax_rate : ℝ

/-- Calculates the total income after tax for Honey -/
noncomputable def total_income_after_tax (finances : HoneyFinances) : ℝ :=
  let night_course_euros := finances.night_course_payment / finances.exchange_rate
  let savings_euros := finances.savings / finances.exchange_rate
  let total_income_before_tax := (night_course_euros / 0.20 + savings_euros / 0.30) / 2
  let tax_amount := total_income_before_tax * finances.tax_rate
  let total_income_after_tax_euros := total_income_before_tax - tax_amount
  total_income_after_tax_euros * finances.exchange_rate

/-- Theorem stating that Honey's total income after tax is approximately $8504.91 -/
theorem honey_income_theorem (finances : HoneyFinances) 
  (h1 : finances.daily_income_range = {x : ℝ | 70 ≤ x ∧ x ≤ 90})
  (h2 : finances.exchange_rate = 1.10)
  (h3 : finances.night_course_payment = 1980)
  (h4 : finances.savings = 2700)
  (h5 : finances.tax_rate = 0.10) :
  abs (total_income_after_tax finances - 8504.91) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_income_theorem_l769_76903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vectors_collinear_with_MN_l769_76981

noncomputable def M : ℝ × ℝ := (1, 1)
noncomputable def N : ℝ × ℝ := (4, -3)

noncomputable def vector_MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

noncomputable def magnitude_MN : ℝ := Real.sqrt ((vector_MN.1)^2 + (vector_MN.2)^2)

noncomputable def unit_vector_positive : ℝ × ℝ := (vector_MN.1 / magnitude_MN, vector_MN.2 / magnitude_MN)
noncomputable def unit_vector_negative : ℝ × ℝ := (-vector_MN.1 / magnitude_MN, -vector_MN.2 / magnitude_MN)

theorem unit_vectors_collinear_with_MN :
  (unit_vector_positive = (3/5, -4/5) ∧ unit_vector_negative = (-3/5, 4/5)) ∧
  (∀ v : ℝ × ℝ, (∃ k : ℝ, v = (k * vector_MN.1, k * vector_MN.2)) →
    (v = unit_vector_positive ∨ v = unit_vector_negative)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vectors_collinear_with_MN_l769_76981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l769_76957

/-- A color, either red or black -/
inductive Color
| red
| black

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A coloring of the plane -/
def Coloring := Point → Color

theorem same_color_unit_distance (c : Coloring) : 
  ∃ (p q : Point), c p = c q ∧ distance p q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l769_76957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l769_76961

/-- Parabola C: y = 1/2 x^2 -/
def C : Set (ℝ × ℝ) := {p | p.2 = 1/2 * p.1^2}

/-- Line l: y = kx - 1 -/
def l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 - 1}

/-- C and l do not intersect -/
def no_intersection (k : ℝ) : Prop := C ∩ l k = ∅

/-- Point on line l -/
def point_on_l (k : ℝ) (P : ℝ × ℝ) : Prop := P ∈ l k

/-- Tangent line to C at point A -/
def tangent_line (A : ℝ × ℝ) : Set (ℝ × ℝ) := {p | p.2 - A.2 = A.1 * (p.1 - A.1)}

/-- A and B are points of tangency of tangents from P to C -/
def tangency_points (k : ℝ) (P A B : ℝ × ℝ) : Prop :=
  A ∈ C ∧ B ∈ C ∧ P ∈ tangent_line A ∧ P ∈ tangent_line B

/-- Fixed point Q -/
def Q (k : ℝ) : ℝ × ℝ := (k, 1)

/-- Line AB passes through Q -/
def AB_through_Q (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, Q k = A + t • (B - A)

/-- M and N are intersection points of PQ and C -/
def intersect_PQ_C (k : ℝ) (P M N : ℝ × ℝ) : Prop :=
  M ∈ C ∧ N ∈ C ∧ ∃ t s : ℝ, M = P + t • (Q k - P) ∧ N = P + s • (Q k - P)

/-- Main theorem -/
theorem parabola_tangent_theorem (k : ℝ) :
  no_intersection k →
  ∀ P A B : ℝ × ℝ, point_on_l k P →
  tangency_points k P A B →
  (AB_through_Q k A B ∧
   ∀ M N : ℝ × ℝ, intersect_PQ_C k P M N →
   |P.1 - M.1| / |P.1 - N.1| = |(Q k).1 - M.1| / |(Q k).1 - N.1|) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l769_76961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_in_first_and_last_game_l769_76997

/-- Represents a chess tournament with specific conditions -/
structure ChessTournament (n : ℕ) where
  /-- Total number of players in the tournament -/
  total_players : ℕ
  /-- Total number of games played in the tournament -/
  total_games : ℕ
  /-- Minimum number of games a player must rest after playing -/
  min_rest : ℕ
  /-- Condition: The total number of players is 2n + 3 -/
  player_count : total_players = 2*n + 3
  /-- Condition: Each player plays exactly once with every other player -/
  game_count : total_games = (total_players * (total_players - 1)) / 2
  /-- Condition: Minimum rest period is n games -/
  rest_period : min_rest = n

/-- Set of players in the first game -/
def set_of_players_in_first_game (n : ℕ) : Set ℕ :=
  {1, 2}  -- Assuming players are numbered from 1 to (2n + 3)

/-- Set of players in the last game -/
def set_of_players_in_last_game (n : ℕ) : Set ℕ :=
  {2*n + 2, 2*n + 3}  -- Assuming the last two players play the last game

/-- Theorem: In a chess tournament satisfying the given conditions, 
    there exists a player who played in both the first and last game -/
theorem player_in_first_and_last_game (n : ℕ) (tournament : ChessTournament n) :
  ∃ (player : ℕ), player ∈ set_of_players_in_first_game n ∧ player ∈ set_of_players_in_last_game n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_in_first_and_last_game_l769_76997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_point_theorem_l769_76935

/-- The point where a ray of light reflects off the x-axis -/
noncomputable def reflection_point (A B : ℝ × ℝ) : ℝ × ℝ :=
  let a := -(2 * A.1 * B.2) / (3 * B.2 - A.2)
  (a, 0)

/-- Theorem stating that the reflection point for the given conditions is (-2/3, 0) -/
theorem reflection_point_theorem (A B : ℝ × ℝ) (hA : A = (-2, 2)) (hB : B = (0, 1)) :
  reflection_point A B = (-2/3, 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_point_theorem_l769_76935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PQR_is_90_degrees_l769_76915

-- Define the point type
def Point := ℝ × ℝ

-- Define the distance function
def dist (A B : Point) : ℝ := sorry

-- Define the angle measure in degrees
def angle_measure (A B C : Point) : ℝ := sorry

-- Define the triangle structure
structure Triangle (P Q S : Point) :=
  (isIsosceles : dist P Q = dist S Q)

-- State the theorem
theorem angle_PQR_is_90_degrees 
  (P Q R S : Point) 
  (triangle : Triangle P Q S) 
  (straight_line : angle_measure R S P = 180) 
  (angle_QSP : angle_measure Q S P = 70) :
  angle_measure P Q R = 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PQR_is_90_degrees_l769_76915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pb_ac_is_90_degrees_l769_76989

-- Define the square ABCD
def Square (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define a point outside a plane
def PointOutsidePlane (P : EuclideanSpace ℝ (Fin 3)) (plane : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define perpendicularity of a line to a plane
def PerpendicularToPlane (line : Set (EuclideanSpace ℝ (Fin 3))) (plane : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define the angle between two lines
noncomputable def AngleBetweenLines (line1 line2 : Set (EuclideanSpace ℝ (Fin 3))) : ℝ := sorry

theorem angle_pb_ac_is_90_degrees 
  (A B C D P : EuclideanSpace ℝ (Fin 3)) 
  (h1 : Square A B C D) 
  (h2 : PointOutsidePlane P {A, B, C, D}) 
  (h3 : PerpendicularToPlane {P, A} {A, B, C, D}) 
  (h4 : ‖P - A‖ = ‖B - A‖) : 
  AngleBetweenLines {P, B} {A, C} = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pb_ac_is_90_degrees_l769_76989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_lateral_area_l769_76917

/-- The lateral surface area of a truncated cone. -/
noncomputable def lateralSurfaceArea (r R h : ℝ) : ℝ :=
  let l := Real.sqrt (h^2 + (R - r)^2)
  Real.pi * l * (r + R)

/-- Theorem: The lateral surface area of a truncated cone with upper base radius 1,
    lower base radius 4, and height 4 is equal to 25π. -/
theorem truncated_cone_lateral_area :
  lateralSurfaceArea 1 4 4 = 25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_lateral_area_l769_76917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l769_76909

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (∀ (x y z w : ℕ), x ∈ ({1, 3, 5, 7} : Set ℕ) → y ∈ ({1, 3, 5, 7} : Set ℕ) → 
    z ∈ ({1, 3, 5, 7} : Set ℕ) → w ∈ ({1, 3, 5, 7} : Set ℕ) →
    x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
    x * y + y * z + z * w ≤ a * b + b * c + c * d) →
  a * b + b * c + c * d = 53 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l769_76909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_monotonicity_intervals_l769_76967

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1 + a) / x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

-- Theorem for the tangent line
theorem tangent_line_at_e :
  let a := 2 * Real.exp 1
  let tangent_line (x y : ℝ) := x + y = 0
  tangent_line (Real.exp 1) (f a (Real.exp 1)) ∧
  ∀ x, tangent_line x (f a x + (deriv (f a)) (Real.exp 1) * (x - Real.exp 1)) := by
sorry

-- Theorem for the intervals of monotonicity
theorem monotonicity_intervals (a : ℝ) :
  (a ≤ -1 → ∀ x y, 0 < x ∧ x < y → (deriv (h a)) x > 0) ∧
  (a > -1 → 
    (∀ x y, 0 < x ∧ x < y ∧ y < 1 + a → (deriv (h a)) x < 0) ∧
    (∀ x y, 1 + a < x ∧ x < y → (deriv (h a)) x > 0)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_monotonicity_intervals_l769_76967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_7pi_over_6_l769_76959

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) :
  (Real.cos (α - π/6) + Real.sin α = 4/5 * Real.sqrt 3) →
  Real.sin (α + 7*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_7pi_over_6_l769_76959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_square_feet_per_person_approx_best_approximation_l769_76985

/-- The average number of square feet per person in Canada in 1980 -/
noncomputable def average_square_feet_per_person : ℝ :=
  let population : ℕ := 24343181
  let area_sq_miles : ℕ := 3855103
  let sq_feet_per_sq_mile : ℕ := 5280 * 5280
  (area_sq_miles * sq_feet_per_sq_mile : ℝ) / population

/-- The average number of square feet per person in Canada in 1980 is approximately 4,413,171 -/
theorem average_square_feet_per_person_approx :
  ‖average_square_feet_per_person - 4413171‖ < 1 := by
  sorry

/-- The best approximation from the given choices is 4,400,000 -/
theorem best_approximation :
  ‖average_square_feet_per_person - 4400000‖ < 
  ‖average_square_feet_per_person - 4000000‖ ∧
  ‖average_square_feet_per_person - 4400000‖ < 
  ‖average_square_feet_per_person - 5000000‖ ∧
  ‖average_square_feet_per_person - 4400000‖ < 
  ‖average_square_feet_per_person - 4500000‖ ∧
  ‖average_square_feet_per_person - 4400000‖ < 
  ‖average_square_feet_per_person - 4900000‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_square_feet_per_person_approx_best_approximation_l769_76985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l769_76910

noncomputable section

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := x^2

/-- The point of tangency -/
def P : ℝ × ℝ := (2, 4)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' P.1

theorem tangent_line_equation :
  ∀ x y : ℝ, (4 * x - y - 4 = 0) ↔ (y - P.2 = m * (x - P.1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l769_76910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_7_l769_76901

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y + 1 = 0

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptotes_eq (x y : ℝ) : Prop := y = x ∨ y = -x

/-- The chord length -/
noncomputable def chord_length : ℝ := Real.sqrt 7

/-- Theorem: The length of the chord cut by the circle on the asymptotes of the hyperbola is √7 -/
theorem chord_length_is_sqrt_7 :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
  asymptotes_eq x₁ y₁ ∧ asymptotes_eq x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_7_l769_76901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_theorem_l769_76966

/-- The circle with equation x^2 + 2x + y^2 = 0 -/
def my_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

/-- The line with equation x + y = 0 -/
def my_line (x y : ℝ) : Prop := x + y = 0

/-- The center of the circle -/
def my_center : ℝ × ℝ := (-1, 0)

/-- The perpendicular line passing through the center -/
def my_perpendicular_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem perpendicular_line_theorem :
  ∀ x y : ℝ, my_circle x y →
  my_perpendicular_line x y ↔ 
  (x = my_center.1 ∨ y = my_center.2) ∧ 
  (∀ a b : ℝ, my_line a b → (x - a) * (y - b) = -(a - x) * (b - y)) :=
by
  sorry

#check perpendicular_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_theorem_l769_76966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_tangent_to_circle_circle_through_points_tangent_to_line_l769_76983

-- Define the basic types
variable (Point Circle Line : Type)

-- Define the necessary operations and relations
variable (lies_on : Point → Circle → Prop)
variable (lies_on_line : Point → Line → Prop)
variable (invert : Point → Point → Point)
variable (invert_circle : Point → Circle → Circle)
variable (invert_line : Point → Line → Circle)
variable (is_inside : Point → Circle → Prop)
variable (is_outside : Point → Circle → Prop)

-- Define the theorem
theorem circle_through_points_tangent_to_circle 
  (A B : Point) (S : Circle) : 
  (∀ (B_star : Point) (S_star : Circle),
    B_star = invert A B →
    S_star = invert_circle A S →
    (((lies_on A S) ∧ (lies_on B S)) → 
      (∃! (n : Nat), n = 0)) ∧
    ((is_inside B_star S_star) → 
      (∃! (n : Nat), n = 0)) ∧
    ((lies_on B_star S_star) → 
      (∃! (n : Nat), n = 1)) ∧
    ((is_outside B_star S_star) → 
      (∃! (n : Nat), n = 2))) :=
by sorry

-- Define the theorem for the line case
theorem circle_through_points_tangent_to_line 
  (A B : Point) (S : Line) : 
  (∀ (B_star : Point) (S_star : Circle),
    B_star = invert A B →
    S_star = invert_line A S →
    (((lies_on_line A S) ∧ (lies_on_line B S)) → 
      (∃! (n : Nat), n = 0)) ∧
    ((is_inside B_star S_star) → 
      (∃! (n : Nat), n = 0)) ∧
    ((lies_on B_star S_star) → 
      (∃! (n : Nat), n = 1)) ∧
    ((is_outside B_star S_star) → 
      (∃! (n : Nat), n = 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_tangent_to_circle_circle_through_points_tangent_to_line_l769_76983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l769_76937

theorem solution_set_inequality (x : ℝ) : (2 : ℝ)^(x^2 - x) < 4 ↔ x ∈ Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l769_76937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alicia_john_efficiency_l769_76900

/-- Combined fuel efficiency of two cars -/
noncomputable def combined_efficiency (e1 e2 : ℝ) : ℝ :=
  2 / ((1 / e1) + (1 / e2))

theorem alicia_john_efficiency :
  let alicia_efficiency := (20 : ℝ)
  let john_efficiency := (15 : ℝ)
  ∃ ε > 0, |combined_efficiency alicia_efficiency john_efficiency - 120 / 7| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alicia_john_efficiency_l769_76900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_theorem_l769_76988

/-- Calculates the population after applying a percentage change --/
noncomputable def apply_percentage_change (population : ℝ) (percentage : ℝ) : ℝ :=
  population * (1 + percentage / 100)

/-- Represents the population changes over 5 years --/
noncomputable def population_after_changes (initial_population : ℝ) : ℝ :=
  apply_percentage_change
    (apply_percentage_change
      (apply_percentage_change
        (apply_percentage_change
          (apply_percentage_change initial_population 25)
          (-20))
        10)
      (-15))
    30

/-- Theorem stating the relationship between initial and final population --/
theorem population_growth_theorem (initial_population : ℝ) :
  initial_population = 24850 →
  population_after_changes initial_population + 150 = 25000 :=
by
  intro h
  sorry

#check population_growth_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_theorem_l769_76988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l769_76994

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  (Nat.choose total_members committee_size - (
    Nat.choose girls committee_size +
    Nat.choose boys 1 * Nat.choose girls 5 +
    Nat.choose boys committee_size +
    Nat.choose girls 1 * Nat.choose boys 5
  ) : ℚ) / Nat.choose total_members committee_size = 457215 / 593775 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l769_76994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l769_76933

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the equation
def satisfies_equation (x : ℤ) : Prop :=
  floor (-77.66 * (x : ℝ)) = -77 * x + 1

-- State the theorem
theorem exactly_three_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, satisfies_equation x) ∧ s.card = 3 := by
  sorry

#check exactly_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l769_76933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_EFGH_l769_76924

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of rectangles forming EFGH -/
structure Configuration where
  small_rectangle : Rectangle
  vertical_count : ℕ
  horizontal_count : ℕ

/-- The resulting large rectangle EFGH -/
def large_rectangle (config : Configuration) : Rectangle :=
  { width := config.small_rectangle.width * config.vertical_count,
    height := config.small_rectangle.height * config.horizontal_count }

theorem area_of_EFGH (config : Configuration) 
  (h1 : config.small_rectangle.width = 7)
  (h2 : config.small_rectangle.height = 14)
  (h3 : config.vertical_count = 3)
  (h4 : config.horizontal_count = 1) :
  (large_rectangle config).area = 294 := by
  -- Unfold definitions
  unfold large_rectangle
  unfold Rectangle.area
  -- Simplify
  simp [h1, h2, h3, h4]
  -- Perform the calculation
  norm_num

#check area_of_EFGH

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_EFGH_l769_76924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_derivative_range_l769_76970

theorem local_max_derivative_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, HasDerivAt f (a * (x + 1) * (x - a)) x) →
  (∃ δ > 0, ∀ x, |x - a| < δ → f x ≤ f a) →
  a ∈ Set.Ioo (-1) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_derivative_range_l769_76970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_range_l769_76963

-- Define the hyperbola
noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the condition for the distance comparison
noncomputable def distance_condition (a c : ℝ) : Prop := 3*c/2 - a > 3*a/2 + a^2/c

-- Define the asymptote angle
noncomputable def asymptote_angle (a b : ℝ) : ℝ := Real.arctan (b/a)

theorem hyperbola_asymptote_angle_range (a b : ℝ) :
  a > 0 →
  b > 0 →
  (∃ c : ℝ, c > 0 ∧ distance_condition a c) →
  0 < asymptote_angle a b ∧ asymptote_angle a b < Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_range_l769_76963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l769_76998

/-- The final price of an article after two successive discounts -/
noncomputable def final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  list_price * (1 - discount1 / 100) * (1 - discount2 / 100)

/-- Theorem stating that the final price of the article is approximately 61.11 rupees -/
theorem article_price_after_discounts :
  let list_price : ℝ := 70
  let discount1 : ℝ := 10
  let discount2 : ℝ := 3.000000000000001
  abs (final_price list_price discount1 discount2 - 61.11) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l769_76998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l769_76975

/-- A trapezoid ABCD with specific properties -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  abParallelCd : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1)
  abEqualsCd : dist A B = dist C D
  adRightAngle : (A.1 - D.1) * (A.2 - D.2) + (B.1 - C.1) * (B.2 - C.2) = 0
  bcRightAngle : (B.1 - C.1) * (C.1 - D.1) + (B.2 - C.2) * (C.2 - D.2) = 0
  adLength : dist A D = 5
  dcLength : dist D C = 7
  bcDivided : dist B C = 12

/-- The perimeter of the trapezoid ABCD is 41 units -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  dist t.A t.B + dist t.B t.C + dist t.C t.D + dist t.D t.A = 41 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l769_76975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_count_l769_76953

theorem petri_dish_count : ℝ := by
  let total_germs : ℝ := 5.4 * 10^6
  let germs_per_dish : ℝ := 500
  let num_dishes : ℝ := total_germs / germs_per_dish
  have h : num_dishes = 10800 := by
    -- Proof steps would go here
    sorry
  exact num_dishes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_count_l769_76953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l769_76969

noncomputable section

-- Define the functions and domain
def f (k : ℝ) (x : ℝ) : ℝ := k * x
def g (x : ℝ) : ℝ := 2 * Real.log x + 2 * Real.exp 1

def domain : Set ℝ := {x | 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 2}

-- Define the symmetry condition
def symmetric_points (k : ℝ) : Prop :=
  ∃ x ∈ domain, 2 * Real.exp 1 - f k x = g x

-- State the theorem
theorem k_range :
  ∀ k : ℝ, symmetric_points k → -2 / Real.exp 1 ≤ k ∧ k ≤ 2 * Real.exp 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l769_76969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_construction_cost_l769_76972

/-- Represents the construction cost function for the workshop -/
noncomputable def construction_cost (a b x : ℝ) : ℝ :=
  if x < 14 then
    7 * a * (x / 4 + 36 / x - 1) + b
  else
    7 * a / 2 + 2 * a * (x + 126 / x - 7) + b

/-- Theorem stating that the construction cost is minimized when x = 12 -/
theorem min_construction_cost (a b : ℝ) (ha : a > 0) (hb : b ≥ 0) :
  ∀ x > 0, construction_cost a b 12 ≤ construction_cost a b x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_construction_cost_l769_76972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_x_eq_neg_one_l769_76944

/-- The inclination angle of a vertical line -/
noncomputable def inclination_angle_vertical_line : ℝ := Real.pi / 2

/-- A line is vertical if it can be expressed as x = k for some constant k -/
def is_vertical_line (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, f x = y ↔ x = k

/-- The inclination angle of the line x = -1 is π/2 -/
theorem inclination_angle_x_eq_neg_one :
  let f : ℝ → ℝ := fun x ↦ -1
  is_vertical_line f → inclination_angle_vertical_line = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_x_eq_neg_one_l769_76944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l769_76914

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := -(x + 2) * (x - m)

noncomputable def g (x : ℝ) : ℝ := 2^x - 2

-- Part 1
theorem part1 (m : ℝ) (h : m > -2) :
  (∀ x, f m x ≥ 0 → g x < 0) ∧ 
  (∃ x, g x < 0 ∧ f m x < 0) ↔ 
  m < 1 :=
sorry

-- Part 2
theorem part2 (m : ℝ) (h : m > -2) :
  (∀ x, f m x < 0 ∨ g x < 0) ∧ 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 0, f m x * g x < 0) ↔ 
  -1 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l769_76914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_polynomial_satisfies_conditions_l769_76902

noncomputable def p (x : ℝ) : ℝ := -1/9 * x^4 + 40/9 * x^3 - 8 * x^2 + 10 * x + 2

theorem quartic_polynomial_satisfies_conditions :
  p 1 = -3 ∧ p 2 = -1 ∧ p 3 = 1 ∧ p 4 = -7 ∧ p 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_polynomial_satisfies_conditions_l769_76902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_correct_l769_76927

/-- Represents the monthly growth rate of factory production -/
def x : ℝ := 0  -- We define x as a real number, initially set to 0

/-- Initial production in January (in ten thousands) -/
def initial_production : ℝ := 50

/-- Total production for the first quarter (in ten thousands) -/
def total_quarterly_production : ℝ := 182

/-- The equation representing the production scenario -/
def production_equation (x : ℝ) : Prop :=
  initial_production * (1 + (1 + x) + (1 + x)^2) = total_quarterly_production

/-- Theorem stating that the production equation correctly represents the scenario -/
theorem production_equation_correct (x : ℝ) :
  production_equation x ↔ 
  initial_production * (1 + (1 + x) + (1 + x)^2) = total_quarterly_production :=
by
  -- The proof is trivial as the left-hand side is defined to be equal to the right-hand side
  rfl

-- Example to demonstrate the use of the theorem
example (x : ℝ) : production_equation x → 
  initial_production * (1 + (1 + x) + (1 + x)^2) = total_quarterly_production :=
by
  intro h
  exact (production_equation_correct x).mp h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_correct_l769_76927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_cosine_series_l769_76996

theorem sum_of_complex_cosine_series (i : ℂ) (h : i^2 = -1) :
  let series := (Finset.range 41).sum (λ n ↦ i^n * Real.cos ((45 + 90 * n : ℝ) * π / 180))
  series = (Real.sqrt 2 / 2) * (21 - 20 * i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_cosine_series_l769_76996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017_value_l769_76929

noncomputable def mySequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => (Real.sqrt 3 * mySequence n - 1) / (mySequence n + Real.sqrt 3)

theorem sequence_2017_value :
  mySequence 2017 = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017_value_l769_76929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_extreme_points_l769_76923

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/3) * x^3 + 4 * x^2 + 9 * x - 1

-- Define the geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define what it means for a point to be an extreme point of f
def isExtremePoint (x : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ ε > 0, ∀ y : ℝ, 0 < |y - x| ∧ |y - x| < ε → f y ≠ f x

-- State the theorem
theorem geometric_sequence_extreme_points
  (a : ℕ → ℝ)
  (h_geometric : isGeometric a)
  (h_extreme_3 : isExtremePoint (a 3) f)
  (h_extreme_7 : isExtremePoint (a 7) f)
  (h_distinct : a 3 ≠ a 7) :
  a 5 = -3 := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_extreme_points_l769_76923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_inequality_l769_76949

theorem solution_to_inequality : ∃! x : ℤ, x > 3 ∧ x ∈ ({-3, 0, 2, 4} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_inequality_l769_76949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincide_l769_76940

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2/3 - y^2/1 = 1

-- Define the focus of the parabola
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the right focus of the hyperbola
noncomputable def hyperbola_right_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem parabola_hyperbola_focus_coincide (p : ℝ) :
  (parabola_focus p = hyperbola_right_focus) → p = 4 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

#check parabola_hyperbola_focus_coincide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincide_l769_76940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bounds_l769_76922

theorem angle_bounds (w x y z : ℝ) 
  (h1 : -π/2 < w ∧ w < π/2)
  (h2 : -π/2 < x ∧ x < π/2)
  (h3 : -π/2 < y ∧ y < π/2)
  (h4 : -π/2 < z ∧ z < π/2)
  (h5 : Real.sin w + Real.sin x + Real.sin y + Real.sin z = 1)
  (h6 : Real.cos (2*w) + Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) ≥ 10/3) :
  0 ≤ w ∧ w ≤ π/6 ∧ 0 ≤ x ∧ x ≤ π/6 ∧ 0 ≤ y ∧ y ≤ π/6 ∧ 0 ≤ z ∧ z ≤ π/6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bounds_l769_76922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_values_l769_76995

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence (α : Type*) [Field α] where
  a₁ : α
  q : α

/-- Sum of the first n terms of a geometric sequence -/
def sumGeometric {α : Type*} [Field α] (s : GeometricSequence α) (n : ℕ) : α :=
  s.a₁ * (1 - s.q^n) / (1 - s.q)

/-- The third term of a geometric sequence -/
def thirdTerm {α : Type*} [Field α] (s : GeometricSequence α) : α :=
  s.a₁ * s.q^2

theorem geometric_sequence_ratio_values 
  (s : GeometricSequence ℝ) :
  sumGeometric s 3 = 3 * thirdTerm s →
  s.q = 1/2 ∨ s.q = (Real.sqrt 5 - 1)/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_values_l769_76995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_mean_variance_l769_76999

-- Define a sample type
def Sample := List ℝ

-- Define the mean of a sample
noncomputable def mean (s : Sample) : ℝ := (s.sum) / s.length

-- Define the variance of a sample
noncomputable def variance (s : Sample) : ℝ := (s.map (λ x => (x - mean s)^2)).sum / s.length

-- Define a function to transform the sample
def transform (s : Sample) : Sample := s.map (λ x => 2 * x + 1)

-- State the theorem
theorem transform_mean_variance (s : Sample) 
  (h1 : mean s = 4) 
  (h2 : variance s = 1) : 
  mean (transform s) = 9 ∧ variance (transform s) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_mean_variance_l769_76999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_C_subset_A_iff_l769_76921

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 1 < 2*x - 1 ∧ 2*x - 1 < 5}

-- Define set B
noncomputable def B : Set ℝ := {y | ∃ x : ℝ, x ≥ -2 ∧ y = (1/2)^x}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | a - 1 < x - a ∧ x - a < 1}

-- Theorem 1: Intersection of complement of A and B
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | (0 < x ∧ x ≤ 1/2) ∨ (5/2 ≤ x ∧ x ≤ 4)} :=
sorry

-- Theorem 2: Condition for C to be a subset of A
theorem C_subset_A_iff (a : ℝ) :
  C a ⊆ A ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_C_subset_A_iff_l769_76921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_value_l769_76939

theorem tan_two_alpha_value (α : ℝ) (h : Real.sin α = 2 * Real.sin (α + π / 2)) : 
  Real.tan (2 * α) = -(4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_value_l769_76939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l769_76941

noncomputable def a : ℝ := (2/3: ℝ) ^ (1/3 : ℝ)
noncomputable def b : ℝ := (2/3 : ℝ) ^ (1/2 : ℝ)
noncomputable def c : ℝ := (3/5 : ℝ) ^ (1/2 : ℝ)

theorem a_gt_b_gt_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l769_76941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_product_of_squares_l769_76956

theorem remainder_of_product_of_squares : 
  (Finset.range 10).prod (fun k => (10 * k + 3)^2) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_product_of_squares_l769_76956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_proof_l769_76952

/-- Represents a rectangle with given length and breadth -/
structure Rectangle where
  length : ℚ
  breadth : ℚ

/-- The ratio of a rectangle's length to breadth -/
def lengthToBreadthRatio (rect : Rectangle) : ℚ :=
  rect.length / rect.breadth

theorem rectangle_ratio_proof (rect : Rectangle) 
  (h1 : rect.length = 250)
  (h2 : rect.breadth = 160) :
  lengthToBreadthRatio rect = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_proof_l769_76952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_A_in_triangle_l769_76907

/-- Given a triangle ABC with vectors m and n defined as below, 
    and m perpendicular to n, the maximum value of sin A is 1/2 -/
theorem max_sin_A_in_triangle (A B C : ℝ × ℝ) : 
  let m : ℝ × ℝ := (C.1 - B.1, C.2 - B.2) - 2 • (A.1 - C.1, A.2 - C.2)
  let n : ℝ × ℝ := (A.1 - B.1, A.2 - B.2) - (A.1 - C.1, A.2 - C.2)
  m.1 * n.1 + m.2 * n.2 = 0 → 
  ∃ (sin_A : ℝ), sin_A = Real.sin (Real.arccos ((A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2)) / 
                 (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2))) ∧
                 sin_A ≤ 1/2 ∧
                 (∀ (other_sin_A : ℝ), other_sin_A = Real.sin (Real.arccos ((A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2)) / 
                 (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2))) →
                 other_sin_A ≤ sin_A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_A_in_triangle_l769_76907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sequence_mod_five_l769_76931

def sequenceList : List ℕ := List.range 20 |>.map (λ n => 7 + 10 * n)

theorem product_sequence_mod_five :
  (List.prod sequenceList : ℤ) ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sequence_mod_five_l769_76931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_l769_76912

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def secondDiagonal (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

/-- Theorem: For a rhombus with area 60 cm² and one diagonal 10 cm, the other diagonal is 12 cm -/
theorem rhombus_second_diagonal :
  let r : Rhombus := { area := 60, diagonal1 := 10 }
  secondDiagonal r = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_l769_76912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_five_l769_76908

/-- The sum of the infinite series ∑(n=1 to ∞) (2n^2 - n) / (n(n+1)(n+2)) -/
noncomputable def infiniteSeries : ℝ := ∑' n, (2 * n^2 - n) / (n * (n + 1) * (n + 2))

/-- The sum of the infinite series ∑(n=1 to ∞) (2n^2 - n) / (n(n+1)(n+2)) is equal to 5 -/
theorem infiniteSeries_eq_five : infiniteSeries = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_five_l769_76908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_is_rhombus_l769_76978

structure IsoscelesTrapezoid where
  diagonalLength : ℝ
  diagonalAngle : ℝ

structure Rhombus where
  sideLength : ℝ
  acuteAngle : ℝ

noncomputable def midpointQuadrilateral (t : IsoscelesTrapezoid) : Rhombus :=
  { sideLength := t.diagonalLength / 2,
    acuteAngle := t.diagonalAngle }

theorem midpoint_quadrilateral_is_rhombus (t : IsoscelesTrapezoid) 
  (h1 : t.diagonalLength = 10) 
  (h2 : t.diagonalAngle = 40) : 
  let r := midpointQuadrilateral t
  r.sideLength = 5 ∧ r.acuteAngle = 40 ∧ 180 - r.acuteAngle = 140 := by
  sorry

#check midpoint_quadrilateral_is_rhombus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_is_rhombus_l769_76978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l769_76951

def is_valid_number (n : ℕ) : Prop :=
  n > 36 ∧ n ≤ 45 ∧ n ≥ 10 ∧ n < 100 ∧
  (n / 10 ∈ ({1, 3, 4, 7} : Finset ℕ)) ∧ 
  (n % 10 ∈ ({1, 3, 4, 7} : Finset ℕ)) ∧
  (n / 10 ≠ n % 10)

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 43 :=
by
  intro n h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l769_76951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_boat_speed_is_n_div_3_l769_76943

/-- Two cities A and B are a km apart. A train travels from A to B at n km/hr.
    Two boats leave B in opposite directions at equal speeds. The train meets the oncoming boat
    in half the time it takes to catch up with the other boat. This theorem proves that
    the speed of the boats is n/3 km/hr. -/
theorem boat_speed (a n : ℝ) (ha : a > 0) (hn : n > 0) : 
  let x := n / 3  -- x is the speed of the boats
  let t₁ := a / (n + x)  -- time to meet oncoming boat
  let t₂ := a / (n - x)  -- time to catch up with other boat
  2 * t₁ = t₂ :=
by
  -- Define x as the speed of the boats
  let x := n / 3
  
  -- Show that 2 * t₁ = t₂
  have h1 : 2 * (a / (n + x)) = a / (n - x) := by
    -- Algebraic manipulation
    calc
      2 * (a / (n + x)) = 2 * (a / (n + n/3))   := by rfl
      _                 = 2 * (3*a / (3*n + n)) := by ring_nf
      _                 = 6*a / (4*n)           := by ring_nf
      _                 = 3*a / (2*n)           := by ring_nf
      _                 = a / (2*n/3)           := by ring_nf
      _                 = a / (n - n/3)         := by ring_nf
      _                 = a / (n - x)           := by rfl

  -- Apply the equality
  exact h1

/-- The speed of the boats is n/3 km/hr -/
theorem boat_speed_is_n_div_3 (a n : ℝ) (ha : a > 0) (hn : n > 0) : 
  let x := n / 3  -- x is the speed of the boats
  x = n / 3 :=
by
  -- Define x as the speed of the boats
  let x := n / 3
  -- Trivial equality
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_boat_speed_is_n_div_3_l769_76943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_largest_l769_76971

noncomputable section

open Real

-- Define the floor function
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the fractional part function
def frac (x : ℝ) : ℝ := x - (floor x)

-- Define a, b, c, and d
def a : ℝ := frac ((floor Real.pi) ^ 2)
def b : ℤ := floor ((frac Real.pi) ^ 2)
def c : ℤ := floor ((floor Real.pi) ^ 2)
def d : ℝ := frac ((frac Real.pi) ^ 2)

-- Theorem statement
theorem c_is_largest : 
  (a ≤ c) ∧ (↑b ≤ c) ∧ (c ≥ c) ∧ (d ≤ ↑c) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_largest_l769_76971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_one_plus_i_l769_76925

theorem complex_power_one_plus_i (n : ℤ) :
  (1 + Complex.I) ^ n = (Complex.abs (1 + Complex.I)) ^ n * (Complex.cos (n * Real.pi / 4) + Complex.I * Complex.sin (n * Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_one_plus_i_l769_76925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l769_76934

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Part 1
theorem part_one (x : ℝ) : 
  (lg ((1/x) + 2) > 1) ↔ (0 < x ∧ x < 1/8) := by sorry

-- Part 2
theorem part_two (lambda : ℝ) :
  (∃ x ∈ Set.Icc 2 3, lg (x + 10) = (1/Real.sqrt 2)^x + lambda) →
  (lg 12 - 1/2 ≤ lambda ∧ lambda ≤ lg 13 - Real.sqrt 2 / 4) := by sorry

-- Part 3
theorem part_three (x : ℝ) :
  (∀ n : ℕ, lg (Real.cos (2^n * x) + 2) < lg 2) →
  (∃ k n : ℕ, (π/2 + 2*k*π) / 2^n < x ∧ x < (3*π/2 + 2*k*π) / 2^n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l769_76934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_average_speed_l769_76936

/-- Calculates the average speed of a runner given track length, number of laps, and lap times. -/
noncomputable def averageSpeed (trackLength : ℝ) (numLaps : ℕ) (lapTimes : List ℝ) : ℝ :=
  let totalDistance := trackLength * (numLaps : ℝ)
  let totalTime := lapTimes.sum
  totalDistance / totalTime

/-- Theorem stating that for the given conditions, the average speed is 5 m/s. -/
theorem bobs_average_speed :
  let trackLength : ℝ := 400
  let numLaps : ℕ := 3
  let lapTimes : List ℝ := [70, 85, 85]
  averageSpeed trackLength numLaps lapTimes = 5 := by
  -- Unfold the definition of averageSpeed
  unfold averageSpeed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_average_speed_l769_76936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_equality_l769_76932

-- Define the pyramid parameters
noncomputable def baseEdgeLength : ℝ := 10
noncomputable def slantEdgeLength : ℝ := 12
noncomputable def lowerCutHeight : ℝ := 4
noncomputable def upperCutHeight : ℝ := 6

-- Define the pyramid height
noncomputable def pyramidHeight : ℝ := Real.sqrt (slantEdgeLength^2 - (baseEdgeLength/2)^2)

-- Define the frustum volume function
noncomputable def frustumVolume (h₁ h₂ : ℝ) : ℝ :=
  let a₁ := baseEdgeLength * (pyramidHeight - h₁) / pyramidHeight
  let a₂ := baseEdgeLength * (pyramidHeight - h₂) / pyramidHeight
  (2/3) * (a₁^2 + Real.sqrt (a₁^2 * a₂^2) + a₂^2)

-- Theorem statement
theorem frustum_volume_equality :
  frustumVolume lowerCutHeight upperCutHeight =
    (2/3) * ((10 * (Real.sqrt 119 - 4)/Real.sqrt 119)^2 +
    Real.sqrt ((10 * (Real.sqrt 119 - 4)/Real.sqrt 119)^2 * (10 * (Real.sqrt 119 - 6)/Real.sqrt 119)^2) +
    (10 * (Real.sqrt 119 - 6)/Real.sqrt 119)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_equality_l769_76932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l769_76986

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + π) = f x) ∧
  (∀ k, k > 0 → k < π → ∃ x, f (x + k) ≠ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l769_76986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_forms_triangle_l769_76946

/-- Triangle inequality theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ ∃ (t : Set ℝ), t = {a, b, c}

/-- Prove that the set (6, 8, 11) can form a triangle. -/
theorem set_forms_triangle :
  ∃ (t : Set ℝ), t = {6, 8, 11} ∧ (6 + 8 > 11 ∧ 8 + 11 > 6 ∧ 11 + 6 > 8) :=
by
  use {6, 8, 11}
  constructor
  · rfl
  · constructor
    · norm_num
    · constructor
      · norm_num
      · norm_num

#check set_forms_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_forms_triangle_l769_76946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l769_76965

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_to_cross : ℝ)
  (h1 : train_length = 100)
  (h2 : train_speed_kmph = 36)
  (h3 : time_to_cross = 29.997600191984642)
  : ∃ (bridge_length : ℝ), ∀ ε > 0, |bridge_length - 199.97600191984642| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l769_76965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l769_76911

/-- Calculates the speed of a train given the lengths of two trains, the speed of one train, 
    and the time taken for them to clear each other when moving in opposite directions. -/
noncomputable def calculate_train_speed (length1 length2 : ℝ) (speed2 : ℝ) (clear_time : ℝ) : ℝ :=
  let total_length := length1 + length2
  let total_length_km := total_length / 1000
  let clear_time_hours := clear_time / 3600
  let relative_speed := total_length_km / clear_time_hours
  relative_speed - speed2

/-- The theorem stating that given the specified conditions, 
    the speed of the first train is approximately 80.069 kmph. -/
theorem train_speed_calculation :
  let length1 : ℝ := 110
  let length2 : ℝ := 200
  let speed2 : ℝ := 65
  let clear_time : ℝ := 7.695936049253991
  let calculated_speed := calculate_train_speed length1 length2 speed2 clear_time
  ∃ ε > 0, |calculated_speed - 80.069| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l769_76911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_equals_4_f_prime_geq_ln_implies_a_equals_3_l769_76974

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.sin x - Real.cos x - (1/2) * a * x^2

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.cos x + Real.sin x - a * x

theorem tangent_line_parallel_implies_a_equals_4 (a : ℝ) : 
  (f_prime a (π/4) = Real.exp (π/4) - π) → a = 4 := by sorry

theorem f_prime_geq_ln_implies_a_equals_3 (a : ℝ) :
  (∀ x < 1, f_prime a x ≥ Real.log (1 - x)) → a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_equals_4_f_prime_geq_ln_implies_a_equals_3_l769_76974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l769_76973

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x - a) / Real.log (1/2)

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ∧ 
  (∀ x ∈ Set.Ioo (-3 : ℝ) (1 - Real.sqrt 3), StrictMono (f a)) →
  a ∈ Set.Icc 0 2 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l769_76973
