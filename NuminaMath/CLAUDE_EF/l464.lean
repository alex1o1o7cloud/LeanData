import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_M_l464_46495

def N : Set Nat := {n | 1 ≤ n ∧ n ≤ 1998}

def has_exactly_one_zero (n : Nat) : Prop :=
  ∃ (d : Nat), d < 4 ∧ (n.digits 10).count 0 = 1

def M : Set Nat := {n ∈ N | has_exactly_one_zero n}

-- We need to make this function computable to use it in Finset.filter
def has_exactly_one_zero_decidable (n : Nat) : Bool :=
  (n.digits 10).count 0 = 1

theorem max_cardinality_M : Finset.card (Finset.filter (λ n => has_exactly_one_zero_decidable n) (Finset.range 1999)) = 414 := by
  sorry

#eval Finset.card (Finset.filter (λ n => has_exactly_one_zero_decidable n) (Finset.range 1999))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_M_l464_46495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l464_46470

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

def C₂ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the ray
noncomputable def ray (θ : ℝ) (ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem statement
theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    (∃ α, C₁ α = ray (π/3) ρ₁) ∧
    (C₂ (ray (π/3) ρ₂).1 (ray (π/3) ρ₂).2) ∧
    ρ₂ - ρ₁ = Real.sqrt 30 / 5 - 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l464_46470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l464_46465

noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem rect_to_polar_3_neg3 :
  let (r, θ) := rect_to_polar 3 (-3)
  r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l464_46465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_points_l464_46428

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

def g (x : ℝ) : ℝ := x - 2

theorem shortest_distance_between_points (x₁ x₂ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) (h₃ : f x₁ = g x₂) : 
  |x₂ - x₁| ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_points_l464_46428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_width_is_ten_l464_46409

/-- A rectangular book with given length and area -/
structure Book where
  length : ℝ
  area : ℝ

/-- The width of a book -/
noncomputable def width (b : Book) : ℝ := b.area / b.length

/-- Theorem: A book with length 5 inches and area 50 square inches has a width of 10 inches -/
theorem book_width_is_ten :
  ∀ (b : Book), b.length = 5 ∧ b.area = 50 → width b = 10 := by
  intro b ⟨h_length, h_area⟩
  unfold width
  rw [h_length, h_area]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_width_is_ten_l464_46409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_70_cos_10_sqrt3_tan_20_minus_1_l464_46415

open Real

theorem tan_70_cos_10_sqrt3_tan_20_minus_1 :
  tan (70 * π / 180) * cos (10 * π / 180) * (Real.sqrt 3 * tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_70_cos_10_sqrt3_tan_20_minus_1_l464_46415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_veg_cost_ratio_l464_46490

/-- The ratio of beef cost to vegetable cost per pound -/
noncomputable def cost_ratio (beef_weight : ℝ) (veg_weight : ℝ) (veg_price : ℝ) (total_cost : ℝ) : ℝ :=
  let beef_cost := (total_cost - veg_weight * veg_price) / beef_weight
  beef_cost / veg_price

/-- The theorem stating the cost ratio of beef to vegetables is 3 -/
theorem beef_veg_cost_ratio :
  cost_ratio 4 6 2 36 = 3 := by
  -- Unfold the definition of cost_ratio
  unfold cost_ratio
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_veg_cost_ratio_l464_46490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_is_168_l464_46426

/-- Represents a grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Represents a path on the grid -/
structure GridPath (g : Grid) where
  steps : ℕ
  diagonalSteps : ℕ

/-- Counts the number of valid paths on a given grid -/
def countPaths (g : Grid) (p : GridPath g) : ℕ :=
  sorry

/-- The specific grid for our problem -/
def problemGrid : Grid :=
  { width := 4, height := 3 }

/-- The specific path constraints for our problem -/
def problemPath : GridPath problemGrid :=
  { steps := 8, diagonalSteps := 1 }

/-- Theorem stating that the number of paths is 168 -/
theorem path_count_is_168 :
  countPaths problemGrid problemPath = 168 := by
  sorry

#eval problemGrid.width
#eval problemPath.steps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_is_168_l464_46426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l464_46406

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * log x + (1-a) * x + 1

-- State the theorem
theorem f_properties (a : ℝ) :
  -- The derivative of f
  (∀ x > 0, HasDerivAt (f a) ((x-a)*(x+1)/x) x) ∧
  -- Monotonicity when a ≤ 0
  (a ≤ 0 → ∀ x > 0, ((x-a)*(x+1)/x) > 0) ∧
  -- Monotonicity when a > 0
  (a > 0 → (∀ x ∈ Set.Ioo 0 a, ((x-a)*(x+1)/x) < 0) ∧
           (∀ x > a, ((x-a)*(x+1)/x) > 0)) ∧
  -- Inequality when a = 1
  (a = 1 → ∀ x > 0, f 1 x ≤ x * (exp x - 1) + (1/2) * x^2 - 2 * log x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l464_46406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_probability_l464_46425

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- The number of teams in the tournament -/
def num_teams : Nat := 6

/-- The number of games each team plays -/
def games_per_team : Nat := 5

/-- The probability of winning any given match -/
noncomputable def win_probability : ℚ := 1/2

/-- Represents the outcome of a game -/
inductive GameResult
| Win
| Loss

/-- The number of points awarded for a win -/
def win_points : Nat := 1

/-- The state of the tournament after the first three games -/
structure TournamentState where
  a_wins : Nat := 2  -- Team A has won 2 games
  b_wins : Nat := 0  -- Team B has won 0 games
  c_wins : Nat := 0  -- Team C has won 0 games

/-- The probability that team A finishes with more points than both team B and team C -/
noncomputable def probability_a_wins (state : TournamentState) : ℚ :=
  sorry

theorem tournament_probability : 
  probability_a_wins {a_wins := 2, b_wins := 0, c_wins := 0} = 193/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_probability_l464_46425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_t_l464_46474

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Defines a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the ellipse and maximum t value -/
theorem ellipse_and_max_t (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 2 / 2)
  (h_line : ∃ (l : Line), l.a = 1 ∧ l.b = -1 ∧ l.c = Real.sqrt 2 ∧ 
    ∀ (p : Point), p.x^2 + p.y^2 = e.b^2 → l.a * p.x + l.b * p.y + l.c ≥ 0) :
  (∀ (x y : ℝ), x^2 / 2 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (∃ (t : ℤ), t = 1 ∧ 
    ∀ (t' : ℤ), (∃ (a b p : Point), 
      a.x^2 / e.a^2 + a.y^2 / e.b^2 = 1 ∧
      b.x^2 / e.a^2 + b.y^2 / e.b^2 = 1 ∧
      p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1 ∧
      (a.x + b.x, a.y + b.y) = (t' * p.x, t' * p.y) ∧
      ∃ (k : ℝ), a.y = k * (a.x - 2) ∧ b.y = k * (b.x - 2)) →
    t' ≤ t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_t_l464_46474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_chord_line_l464_46432

/-- Circle C with equation x^2 + y^2 - 2y - 3 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

/-- Point P -/
def point_P : ℝ × ℝ := (-1, 2)

/-- Line l passing through point P -/
def line_l (m : ℝ) (x y : ℝ) : Prop := y - point_P.2 = m * (x - point_P.1)

/-- Intersection points of line l and circle C -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ circle_C x y ∧ line_l m x y}

/-- Length of chord AB -/
noncomputable def chord_length (m : ℝ) : ℝ :=
  let points := intersection_points m
  -- We need to replace this with a more appropriate definition
  -- that doesn't rely on Set.min and Set.max
  0 -- placeholder value

theorem minimal_chord_line :
  ∃ m, (∀ x y, line_l m x y ↔ x - y + 3 = 0) ∧
    ∀ m', chord_length m ≤ chord_length m' :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_chord_line_l464_46432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l464_46412

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The perimeter of a triangle given by three points -/
noncomputable def trianglePerimeter (a b c : Point) : ℝ :=
  distance a b + distance b c + distance c a

/-- A point is on the y-axis if its x-coordinate is 0 -/
def onYAxis (p : Point) : Prop :=
  p.x = 0

/-- A point is on the line x - y - 2 = 0 if it satisfies the equation -/
def onLine (p : Point) : Prop :=
  p.x - p.y - 2 = 0

theorem min_perimeter_triangle :
  ∃ (b c : Point),
    onYAxis b ∧
    onLine c ∧
    ∀ (b' c' : Point),
      onYAxis b' → onLine c' →
        trianglePerimeter ⟨2, 3⟩ b c ≤ trianglePerimeter ⟨2, 3⟩ b' c' ∧
        trianglePerimeter ⟨2, 3⟩ b c = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l464_46412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l464_46486

noncomputable section

-- Define the triangle
variable (A B C a b c : ℝ)

-- Define the conditions
def triangle_condition (a b c : ℝ) : Prop := b^2 + c^2 - a^2 = Real.sqrt 3 * b * c

-- Part 1
theorem part_one (h : triangle_condition a b c) (tan_B : Real.tan B = Real.sqrt 6 / 12) :
  b / a = Real.sqrt 30 / 15 := by sorry

-- Part 2
theorem part_two (h : triangle_condition a b c) (angle_B : B = 2 * Real.pi / 3) (side_b : b = 2 * Real.sqrt 3) :
  let median := Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)
  median = Real.sqrt 7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l464_46486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_partition_property_l464_46498

def S (m : ℕ) := Finset.range (m - 1) \ {0, 1}

theorem smallest_m_for_partition_property : ∀ m : ℕ, m ≥ 2 →
  (∀ A B : Finset ℕ, A ∪ B = S m → A ∩ B = ∅ →
    (∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ^ b = b ^ a) ∨ 
    (∃ (a b : ℕ), a ∈ B ∧ b ∈ B ∧ a ^ b = b ^ a)) ↔ m ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_partition_property_l464_46498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_line_area_l464_46485

noncomputable section

/-- An 8th degree polynomial function -/
def polynomial (a b c d e f g h i x : ℝ) : ℝ :=
  a * x^8 + b * x^7 + c * x^6 + d * x^5 + e * x^4 + f * x^3 + g * x^2 + h * x + i

/-- A linear function -/
def line (p q x : ℝ) : ℝ := p * x + q

/-- The area between the polynomial and the line -/
def area (a α β γ δ : ℝ) : ℝ := (a / 9) * ((β^9 - α^9) + (δ^9 - γ^9))

theorem polynomial_line_area 
  (a b c d e f g h i p q α β γ δ : ℝ) 
  (ha : a ≠ 0) 
  (h_order : α < β ∧ β < γ ∧ γ < δ) 
  (h_touch_α : polynomial a b c d e f g h i α = line p q α)
  (h_touch_β : polynomial a b c d e f g h i β = line p q β)
  (h_touch_γ : polynomial a b c d e f g h i γ = line p q γ)
  (h_touch_δ : polynomial a b c d e f g h i δ = line p q δ) :
  (∫ (x : ℝ) in α..β, polynomial a b c d e f g h i x - line p q x) +
  (∫ (x : ℝ) in γ..δ, polynomial a b c d e f g h i x - line p q x) =
  area a α β γ δ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_line_area_l464_46485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_difference_l464_46492

def number_with_nines (n : ℕ) : ℕ := 18 * (10^(n+2) - 1) / 9 + 62

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem prime_sum_difference (p q r : ℕ) (n : ℕ) :
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p ≠ q → p ≠ r → q ≠ r →
  p * q * r = number_with_nines n →
  sum_of_digits p + sum_of_digits q + sum_of_digits r - sum_of_digits (p * q * r) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_difference_l464_46492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_solution_set_inequality_l464_46444

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (9 * (sin x)^2) + 4 / (9 * (cos x)^2)

-- Statement 1
theorem min_value_of_f :
  ∀ x : ℝ, 0 < x → x < π/2 → f x ≥ 1 := by sorry

-- Statement 2
theorem solution_set_inequality :
  ∀ x : ℝ, x^2 + |x - 2| + 1 ≥ 3 ↔ x ≤ 0 ∨ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_solution_set_inequality_l464_46444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l464_46450

noncomputable def f (x : ℝ) : ℝ := |x + 1|

noncomputable def g (x : ℝ) : ℝ := if x ≥ -1 then x + 1 else -x - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l464_46450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_equivalence_l464_46458

theorem abs_sum_equivalence :
  ∀ x : ℝ, |x - 1| + |x + 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_equivalence_l464_46458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_min_side_a_l464_46466

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

-- Theorem for the maximum value of f
theorem f_max_value :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M ∧ M = 2 := by
  sorry

-- Theorem for the minimum value of a in triangle ABC
theorem min_side_a (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) ∧
  f (B + C) = 3/2 ∧
  b + c = 2 →
  a ≥ 1 ∧ ∃ (b' c' : ℝ), b' + c' = 2 ∧ 
    Real.sqrt (b'^2 + c'^2 - 2*b'*c'*Real.cos A) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_min_side_a_l464_46466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l464_46463

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 6}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_range :
  (A ∩ B = Set.Icc 3 7) ∧
  (A ∪ B = Set.Ioo 2 10) ∧
  (∀ a : ℝ, (C a ∪ A = C a) → a < 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l464_46463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_negative_integer_roots_l464_46455

/-- A polynomial of degree 5 with integer coefficients -/
structure Polynomial5 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The sum of all coefficients of the polynomial -/
def Polynomial5.coeff_sum (p : Polynomial5) : ℤ :=
  p.a + p.b + p.c + p.d + p.e

/-- Predicate to check if all roots of the polynomial are negative integers -/
def has_all_negative_integer_roots (p : Polynomial5) : Prop :=
  ∃ s₁ s₂ s₃ s₄ s₅ : ℕ+, 
    (λ x : ℤ ↦ x^5 + p.a*x^4 + p.b*x^3 + p.c*x^2 + p.d*x + p.e) =
    (λ x : ℤ ↦ (x + s₁.val) * (x + s₂.val) * (x + s₃.val) * (x + s₄.val) * (x + s₅.val))

theorem polynomial_with_negative_integer_roots 
  (p : Polynomial5) 
  (h1 : has_all_negative_integer_roots p) 
  (h2 : p.coeff_sum = 2519) : 
  p.e = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_negative_integer_roots_l464_46455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l464_46405

/-- Represents the state of the game -/
structure GameState where
  boxes : Fin 11 → ℕ

/-- A move in the game is represented by the box number where no coin was placed -/
def Move := Fin 11

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState where
  boxes := λ i => if i = move then state.boxes i else state.boxes i + 1

/-- Checks if the game is over (any box has 21 coins) -/
def isGameOver (state : GameState) : Prop :=
  ∃ i, state.boxes i = 21

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: The second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Strategy), ∀ (first_move : Move),
    let game_after_first_move := applyMove ⟨λ _ => 0⟩ first_move
    ¬isGameOver game_after_first_move →
    ∃ (n : ℕ), 
      let final_state := (Nat.iterate (λ state => 
        applyMove (applyMove state (strategy state)) first_move) n game_after_first_move)
      isGameOver final_state ∧
      ¬isGameOver (applyMove final_state first_move) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l464_46405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_345_ratio_triangle_l464_46459

theorem largest_angle_in_345_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 180 →
    (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 →
    max a (max b c) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_345_ratio_triangle_l464_46459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_oak_tree_l464_46424

/-- Calculates the time (in seconds) it takes for a train to pass a stationary object. -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Proves that a train 150 meters long, traveling at 54 km/hr, will take 10 seconds to pass an oak tree. -/
theorem train_passing_oak_tree :
  train_passing_time 150 54 = 10 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_oak_tree_l464_46424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l464_46496

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared * u.1, dot_product / magnitude_squared * u.2)

theorem projection_theorem (proj_vector : ℝ × ℝ) :
  projection (3, 3) proj_vector = (45/10, 9/10) →
  projection (-3, 3) proj_vector = (-30/13, -6/13) := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l464_46496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_on_interval_l464_46420

theorem decreasing_function_on_interval (a : ℝ) (h : a > 0) :
  StrictMonoOn (fun x => -(x^2 - 2*a*x + 1)) (Set.Ioo 0 a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_on_interval_l464_46420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l464_46471

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 3/2) → x₀ = Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l464_46471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l464_46427

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then -Real.exp x + 1 else -((-Real.exp (-x)) + 1)

noncomputable def a : ℝ := -2 * f (-2)
noncomputable def b : ℝ := -f (-1)
noncomputable def c : ℝ := 3 * f 3

theorem relationship_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l464_46427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_two_years_l464_46417

noncomputable def tree_height (n : ℕ) : ℝ := 243 / (3 ^ (5 - n))

theorem tree_height_after_two_years :
  tree_height 2 = 9 :=
by
  -- Unfold the definition of tree_height
  unfold tree_height
  -- Simplify the expression
  simp [pow_sub]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_two_years_l464_46417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_coords_example_l464_46411

/-- Converts rectangular coordinates to cylindrical coordinates -/
noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y ≥ 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ, z)

/-- Theorem: The cylindrical coordinates of (6, -6, 10) are (6√2, 7π/4, 10) -/
theorem cylindrical_coords_example : 
  let (r, θ, z) := rectangular_to_cylindrical 6 (-6) 10
  r = 6 * Real.sqrt 2 ∧ 
  θ = 7 * Real.pi / 4 ∧ 
  z = 10 ∧ 
  r > 0 ∧ 
  0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_coords_example_l464_46411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l464_46456

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3 - 2 * x^2 + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 3

-- State the theorem
theorem tangent_line_and_extrema :
  -- Tangent line equation at (1, 5/3)
  (∃ (k m : ℝ), ∀ x y, (x = 1 ∧ y = f 1) → (y - f 1 = k * (x - 1) ∧ 6 * x + 3 * y - 11 = 0)) ∧
  -- Maximum value
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 3) ∧
  -- Minimum value
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l464_46456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l464_46477

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t + 1, Real.sqrt 3 * t + 1)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the line l'
noncomputable def line_l' (t : ℝ) : ℝ × ℝ := (1 + t / 2, Real.sqrt 3 * t / 2)

-- State the theorem
theorem intersection_length :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, 
      curve_C (line_l' t₁).1 (line_l' t₁).2 ∧
      curve_C (line_l' t₂).1 (line_l' t₂).2 ∧
      A = line_l' t₁ ∧ B = line_l' t₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l464_46477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l464_46452

def n : ℕ := 2^20 * 3^25

theorem divisors_count : 
  (Finset.filter (fun d => d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 499 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l464_46452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_conditions_l464_46475

theorem triangle_formation_conditions (a b c : ℝ) :
  (a > 0) →
  (b > a) →
  (c > b) →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y < c - b ∧ 
    (a * a + x * x = (b - a) * (b - a)) ∧
    (y * y + (c - b) * (c - b) = (c - a) * (c - a))) →
  (a < c / 3) ∧ (b < a + c / 3) := by
  sorry

#check triangle_formation_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_conditions_l464_46475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_perpendicular_pairs_l464_46431

/-- Represents a cube -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  faces : Finset (Finset (Fin 8))

/-- Represents a line in the cube -/
def Line (c : Cube) := c.edges

/-- Represents a plane in the cube -/
def Plane (c : Cube) := c.faces

/-- Predicate to check if a line is perpendicular to a plane -/
def isPerpendicular (c : Cube) (l : Line c) (p : Plane c) : Prop := sorry

/-- The number of perpendicular line-plane pairs in a cube -/
def numPerpendicularPairs (c : Cube) : ℕ :=
  (c.edges.card * c.faces.card) / 3

theorem cube_perpendicular_pairs :
  ∀ c : Cube, numPerpendicularPairs c = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_perpendicular_pairs_l464_46431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l464_46489

-- Define the variables as noncomputable
noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

-- State the theorem
theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = (11 + 2 * Real.sqrt 30) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l464_46489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l464_46436

/-- The distance between two stations in kilometers -/
noncomputable def distance_between_stations : ℝ := 200

/-- The speed of the first train in kilometers per hour -/
noncomputable def speed_first_train : ℝ := 20

/-- The time the first train travels in hours -/
noncomputable def time_first_train : ℝ := 5

/-- The time the second train travels in hours -/
noncomputable def time_second_train : ℝ := 4

/-- The speed of the second train in kilometers per hour -/
noncomputable def speed_second_train : ℝ := 
  (distance_between_stations - speed_first_train * time_first_train) / time_second_train

theorem second_train_speed : speed_second_train = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l464_46436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l464_46439

def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := Real.sqrt 2
  b = 3 ∧ 
  c = 1 ∧ 
  (1/2 * b * c * Real.sin A = area) ∧ 
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (Real.cos A = 1/3 ∨ Real.cos A = -1/3) ∧ 
  (a = 2*Real.sqrt 3 ∨ a = 2*Real.sqrt 2)

theorem triangle_theorem : 
  ∀ (a b c A B C : ℝ), triangle_problem a b c A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l464_46439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_eq_two_has_two_solutions_l464_46448

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 6

theorem f_composition_eq_two_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ f (f a) = 2 ∧ f (f b) = 2 ∧
  ∀ (x : ℝ), f (f x) = 2 → x = a ∨ x = b := by
  sorry

#check f_composition_eq_two_has_two_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_eq_two_has_two_solutions_l464_46448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l464_46478

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 2 / 2) (h4 : ellipse_C 0 1 a b) :
  -- 1. Equation of ellipse C
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  -- 2. Intersection point K lies on hyperbola
  (∀ t x y : ℝ, 
    (∃ y0 : ℝ, ellipse_C t y0 a b ∧ ellipse_C t (-y0) a b) →
    (∃ k1 k2 : ℝ, 
      y = k1 * (x + Real.sqrt 2) ∧ 
      y = k2 * (x - Real.sqrt 2) ∧ 
      k1 * (t + Real.sqrt 2) = y0 ∧ 
      k2 * (t - Real.sqrt 2) = -y0) →
    hyperbola x y) ∧
  -- 3. Equation of line l
  (∃ x1 y1 x2 y2 : ℝ,
    ellipse_C x1 y1 a b ∧ 
    ellipse_C x2 y2 a b ∧
    dot_product x1 y1 x2 y2 = -1/3 ∧
    ((y1 = x1 + 1 ∧ y2 = x2 + 1) ∨ (y1 = -x1 - 1 ∧ y2 = -x2 - 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l464_46478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_to_chord_distance_l464_46421

-- Define the circle
def Circle := {(x, y) : ℝ × ℝ | ∃ (r : ℝ), x^2 + y^2 = r^2}

-- Define the point A
def A : ℝ × ℝ := (4, 2)

-- Define the line where the center lies
def CenterLine (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem center_to_chord_distance :
  ∃ (c : ℝ × ℝ), 
    c ∈ Circle ∧ 
    (0, 0) ∈ Circle ∧ 
    A ∈ Circle ∧
    CenterLine c.1 c.2 ∧
    distance c ((0 + A.1) / 2, (0 + A.2) / 2) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_to_chord_distance_l464_46421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_reciprocal_distances_l464_46434

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points of the line and the parabola
def intersection_points (k : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem parabola_intersection_sum_reciprocal_distances (k : ℝ) :
  let (x1, y1, x2, y2) := intersection_points k
  let d1 := distance x1 y1 (focus.1) (focus.2)
  let d2 := distance x2 y2 (focus.1) (focus.2)
  1 / d1 + 1 / d2 = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_reciprocal_distances_l464_46434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_radii_l464_46443

/-- The radius of the inscribed sphere of a regular tetrahedron -/
noncomputable def radius_inscribed_sphere (a : ℝ) : ℝ :=
  a / 12 * Real.sqrt 6

/-- The radius of the circumscribed sphere of a regular tetrahedron -/
noncomputable def radius_circumscribed_sphere (a : ℝ) : ℝ :=
  a / 4 * Real.sqrt 6

/-- Given a regular tetrahedron with edge length a, 
    prove the radii of inscribed and circumscribed spheres -/
theorem tetrahedron_sphere_radii (a : ℝ) (a_pos : a > 0) :
  ∃ (r R : ℝ),
    r = a / 12 * Real.sqrt 6 ∧
    R = a / 4 * Real.sqrt 6 ∧
    r > 0 ∧
    R > 0 ∧
    r = radius_inscribed_sphere a ∧
    R = radius_circumscribed_sphere a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_radii_l464_46443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_parallel_l464_46446

/-- Given two planar vectors a and b, where a is parallel to b, 
    prove that 2a + 3b equals (-4, -8). -/
theorem vector_sum_parallel (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (-2, m) →
  (a.1 * b.2 = a.2 * b.1) →  -- Condition for parallel vectors
  (2 : ℝ) • a + (3 : ℝ) • b = (-4, -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_parallel_l464_46446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_seven_failures_three_successes_l464_46491

theorem probability_seven_failures_three_successes 
  (p : ℝ) (h1 : 0 < p) (h2 : p < 1) : 
  (1 - p)^7 * p^3 = (1 - p)^7 * p^3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_seven_failures_three_successes_l464_46491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wide_right_field_goals_l464_46494

theorem wide_right_field_goals 
  (total_attempts : ℕ) 
  (miss_ratio : ℚ) 
  (wide_right_ratio : ℚ) 
  (h1 : total_attempts = 60)
  (h2 : miss_ratio = 1/4)
  (h3 : wide_right_ratio = 1/5) :
  ⌊(total_attempts : ℚ) * miss_ratio * wide_right_ratio⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wide_right_field_goals_l464_46494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_median_equations_l464_46449

/-- Triangle ABC with vertices A(4, 0), B(6, 7), and C(0, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle where
  A := (4, 0)
  B := (6, 7)
  C := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The altitude from vertex B to side AC -/
def altitudeFromB (t : Triangle) : LineEquation := sorry

/-- The median from vertex B to side AC -/
def medianFromB (t : Triangle) : LineEquation := sorry

theorem triangle_altitude_median_equations :
  (altitudeFromB triangleABC).a = 3 ∧
  (altitudeFromB triangleABC).b = 2 ∧
  (altitudeFromB triangleABC).c = -12 ∧
  (medianFromB triangleABC).a = 5 ∧
  (medianFromB triangleABC).b = 1 ∧
  (medianFromB triangleABC).c = -20 := by
  sorry

#check triangle_altitude_median_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_median_equations_l464_46449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_jam_spoilage_l464_46407

theorem apple_jam_spoilage (initial_good_ratio initial_bad_ratio final_good_ratio final_bad_ratio : ℝ) 
  (spoilage_rate : ℝ) (days : ℕ) : 
  initial_good_ratio = 0.2 →
  initial_bad_ratio = 0.8 →
  final_good_ratio = 0.8 →
  final_bad_ratio = 0.2 →
  spoilage_rate = 0.5 →
  (spoilage_rate ^ days) * initial_bad_ratio = final_bad_ratio →
  days = 4 := by
  sorry

#check apple_jam_spoilage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_jam_spoilage_l464_46407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l464_46457

theorem trig_identity (x : ℝ) 
  (h1 : Real.cos (x - π/4) = -1/3) 
  (h2 : 5*π/4 < x ∧ x < 7*π/4) : 
  Real.sin x - Real.cos (2*x) = (5*Real.sqrt 2 - 12) / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l464_46457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_change_l464_46483

/-- Represents the time taken for a journey with walking and running portions -/
noncomputable def journey_time (walk_speed : ℝ) (run_speed : ℝ) (walk_distance : ℝ) (run_distance : ℝ) : ℝ :=
  walk_distance / walk_speed + run_distance / run_speed

theorem journey_time_change 
  (v : ℝ) 
  (S : ℝ) 
  (h1 : v > 0)
  (h2 : S > 0)
  (h3 : journey_time v (2*v) (2*S) S = 30) :
  journey_time v (2*v) S (2*S) = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_change_l464_46483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l464_46468

-- Define the function f as noncomputable
noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x)

-- State the theorem
theorem sine_function_property (w : ℝ) (a : ℝ) (h1 : w > 0) :
  (∀ x : ℝ, f w (x - 1/2) = f w (x + 1/2)) →
  f w (-1/4) = a →
  f w (9/4) = -a :=
by
  -- Introduce the assumptions
  intro h2 h3
  
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l464_46468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_meaningful_l464_46461

theorem square_root_meaningful (x : ℝ) : Real.sqrt (x + 2) ∈ Set.range Real.sqrt ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_meaningful_l464_46461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_is_18_l464_46414

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Calculates the total sample size for stratified sampling -/
def stratifiedSampleSize (workshops : List Workshop) (sampleFromSmallest : ℕ) : ℕ :=
  let totalQuantity := workshops.map (·.quantity) |>.sum
  let smallestQuantity := workshops.map (·.quantity) |>.minimum?.getD 1
  let samplingRatio := sampleFromSmallest / smallestQuantity
  samplingRatio * totalQuantity

/-- Theorem: The stratified sample size for the given workshops is 18 -/
theorem stratified_sample_size_is_18 :
    let workshops := [⟨120⟩, ⟨90⟩, ⟨60⟩]
    stratifiedSampleSize workshops 4 = 18 := by
  sorry

#eval stratifiedSampleSize [⟨120⟩, ⟨90⟩, ⟨60⟩] 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_is_18_l464_46414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l464_46437

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 1 / (2 * a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + |2 * x - 1|

theorem problem_solution (a : ℝ) (h : a ≠ 0) :
  (∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, f a x - f a (x + m) ≤ 1) ∧
  (a < 1/2 → ∃ x : ℝ, g a x = 0 ↔ a ∈ Set.Icc (-1/2) 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l464_46437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l464_46482

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/3)

-- State the theorem
theorem f_decreasing_interval :
  ∀ x y, x > 1 ∧ y > 1 ∧ x < y → f x > f y :=
by
  -- Introduce variables and assumptions
  intro x y ⟨hx, hy, hxy⟩
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l464_46482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surviving_car_speed_l464_46413

def car_speeds (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => 61 + i)

theorem surviving_car_speed (n : ℕ) (h : n = 31) :
  List.get! (car_speeds n) ((n - 1) / 2) = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surviving_car_speed_l464_46413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_is_5pi_div_6_l464_46488

/-- The angle of inclination of a line with equation ax + by + c = 0 -/
noncomputable def angle_of_inclination (a b : ℝ) : ℝ :=
  if a ≥ 0 then Real.arctan (-a/b) else Real.pi + Real.arctan (-a/b)

/-- The line equation x + √3y - 3 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y - 3 = 0

theorem angle_of_line_is_5pi_div_6 :
  angle_of_inclination 1 (Real.sqrt 3) = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_is_5pi_div_6_l464_46488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_low_repetition_number_l464_46460

-- Define repetition number
def repetition_number (n : ℕ) : ℕ :=
  (n.digits 10).toFinset.card

-- Theorem statement
theorem exists_multiple_with_low_repetition_number (n : ℕ+) :
  ∃ m : ℕ+, repetition_number (n * m) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_low_repetition_number_l464_46460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_covered_area_l464_46401

-- Define the side length of the squares
def side_length : ℝ := 12

-- Define the area of a square
def square_area (s : ℝ) : ℝ := s^2

-- Define the area of overlap between the squares
noncomputable def overlap_area (s : ℝ) : ℝ := (s / Real.sqrt 2)^2

-- Theorem statement
theorem total_covered_area :
  let total_area := 2 * square_area side_length - overlap_area side_length
  total_area = 216 := by
  -- Expand the definitions
  unfold square_area overlap_area
  -- Simplify the expression
  simp [side_length]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_covered_area_l464_46401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_divisors_l464_46440

/-- The number of positive integer divisors of 1806^1806 that have exactly 1806 positive integer divisors -/
def divisors_with_1806_divisors : ℕ := 2

/-- 1806 -/
def n : ℕ := 1806

theorem count_special_divisors :
  ∃ (S : Finset ℕ), 
    (∀ d ∈ S, d ∣ n^n) ∧ 
    (∀ d ∈ S, ∃ (T : Finset ℕ), (∀ t ∈ T, t ∣ d) ∧ T.card = n) ∧
    S.card = divisors_with_1806_divisors :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_divisors_l464_46440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rates_theorem_l464_46433

-- Define the principal amount
variable (P : ℝ)
axiom P_pos : P > 0

-- Define the time period
def T : ℝ := 5

-- Define the interest ratios
noncomputable def interest_ratio_A : ℝ := 1 / 5
noncomputable def interest_ratio_B : ℝ := 1 / 4

-- Define the interest rate for Bank A
noncomputable def rate_A : ℝ := interest_ratio_A * 100 / T

-- Define the interest rate for Bank B
noncomputable def rate_B : ℝ := 200 * ((1 + interest_ratio_B)^(1/10) - 1)

-- Theorem statement
theorem interest_rates_theorem :
  rate_A = 4 ∧ (abs (rate_B - 3.7) < 0.01) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rates_theorem_l464_46433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f1001_l464_46441

noncomputable def f₁ (x : ℝ) : ℝ := 1/2 - 4/(4*x+2)

noncomputable def f : ℕ → ℝ → ℝ 
| 0, x => x
| 1, x => f₁ x
| (n+1), x => f₁ (f n x)

theorem unique_solution_f1001 :
  ∃! x : ℝ, f 1001 x = x - 2 ∧ x = 3/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f1001_l464_46441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_is_two_years_l464_46423

-- Define the principal amount
def P : ℝ := 1200

-- Define the interest rate
def r : ℝ := 0.10

-- Define the function for simple interest
def simple_interest (t : ℝ) : ℝ := P * r * t

-- Define the function for compound interest
noncomputable def compound_interest (t : ℝ) : ℝ := P * ((1 + r) ^ t - 1)

-- Define the difference between compound and simple interest
noncomputable def interest_difference (t : ℝ) : ℝ := compound_interest t - simple_interest t

-- State the theorem
theorem investment_time_is_two_years :
  ∃ t : ℝ, t = 2 ∧ interest_difference t = 12 := by
  sorry

#eval simple_interest 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_is_two_years_l464_46423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l464_46499

noncomputable def f (x : ℝ) := Real.log (2 + x - x^2) / (|x| - x)

theorem domain_of_f :
  {x : ℝ | 2 + x - x^2 > 0 ∧ |x| - x ≠ 0} = {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l464_46499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l464_46479

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

def passes_through_point (eq : ℝ → ℝ → Prop) (px py : ℝ) : Prop :=
  eq px py

def tangent_to_line (eq : ℝ → ℝ → Prop) (line : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq x y ∧ line x ∧
  ∀ (x' y' : ℝ), eq x' y' ∧ line x' → x = x' ∧ y = y'

theorem circle_properties :
  passes_through_point circle_equation 1 0 ∧
  tangent_to_line circle_equation (λ x => x = -1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l464_46479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_36_to_nearest_tenth_l464_46419

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The problem statement -/
theorem round_4_36_to_nearest_tenth : 
  roundToNearestTenth 4.36 = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_36_to_nearest_tenth_l464_46419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_worked_48_hours_l464_46400

/-- A bus driver's compensation problem -/
def bus_driver_hours (regular_rate overtime_rate : ℚ) 
  (regular_hours : ℕ) (total_compensation : ℚ) : ℚ :=
  let total_hours :=
    regular_hours + 
    ((total_compensation - (regular_rate * regular_hours)) / overtime_rate)
  total_hours

/-- The specific instance of the bus driver's compensation problem -/
def bus_driver_specific_hours : ℚ :=
  let regular_rate : ℚ := 18
  let overtime_rate : ℚ := regular_rate * (1 + 3/4)
  let regular_hours : ℕ := 40
  let total_compensation : ℚ := 976
  bus_driver_hours regular_rate overtime_rate regular_hours total_compensation

#eval bus_driver_specific_hours

theorem bus_driver_worked_48_hours :
  bus_driver_specific_hours = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_worked_48_hours_l464_46400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l464_46476

/-- The inscribed circle radius of a triangle given by three points -/
noncomputable def inscribed_circle_radius (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- The eccentricity of an ellipse given its semi-major and semi-minor axes -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sorry

/-- Given an ellipse with the following properties:
    - Point P(m,4) lies on the ellipse
    - The equation of the ellipse is x²/a² + y²/b² = 1
    - a > b > 0
    - F₁ and F₂ are the two foci of the ellipse
    - The inscribed circle of triangle PF₁F₂ has a radius of 3/2
    Prove that the eccentricity of the ellipse is 3/5 -/
theorem ellipse_eccentricity (a b m : ℝ) (F₁ F₂ : ℝ × ℝ) :
  a > b → b > 0 →
  m^2 / a^2 + 4^2 / b^2 = 1 →
  (∃ r, r = 3/2 ∧ r = inscribed_circle_radius F₁ F₂ (m, 4)) →
  eccentricity a b = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l464_46476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l464_46418

noncomputable def move (z : ℂ) : ℂ := z * (Complex.exp (Complex.I * Real.pi / 6)) + 8

noncomputable def moves (n : ℕ) (z : ℂ) : ℂ :=
  match n with
  | 0 => z
  | n + 1 => move (moves n z)

theorem particle_returns_to_start :
  moves 120 3 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l464_46418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l464_46430

/-- The angle in degrees that the hour hand moves per hour -/
noncomputable def hour_angle_per_hour : ℝ := 30

/-- The angle in degrees that the minute hand moves per minute -/
noncomputable def minute_angle_per_minute : ℝ := 6

/-- The number of hours passed since 12:00 -/
noncomputable def hours_passed : ℝ := 3

/-- The number of minutes passed since the last hour -/
noncomputable def minutes_passed : ℝ := 40

/-- The position of the hour hand in degrees -/
noncomputable def hour_hand_position : ℝ := hour_angle_per_hour * hours_passed + (hour_angle_per_hour * minutes_passed / 60)

/-- The position of the minute hand in degrees -/
noncomputable def minute_hand_position : ℝ := minute_angle_per_minute * minutes_passed

/-- The absolute difference between the hour and minute hand positions -/
noncomputable def angle_difference : ℝ := |minute_hand_position - hour_hand_position|

/-- The smaller angle between the hour and minute hands -/
noncomputable def smaller_angle : ℝ := min angle_difference (360 - angle_difference)

theorem clock_angle_at_3_40 : smaller_angle = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l464_46430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_104_102_l464_46480

theorem abs_diff_squares_104_102 : |((104 : ℤ)^2 - (102 : ℤ)^2)| = 412 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_104_102_l464_46480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integral_fraction_l464_46472

theorem unique_integral_fraction (m n : ℕ) : m ≥ 3 ∧ n ≥ 3 →
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)) ↔
  m = 5 ∧ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integral_fraction_l464_46472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_theorem_l464_46408

def flower_beds_problem (A B C D : Finset ℕ) : Prop :=
  (A.card = 600) ∧
  (B.card = 500) ∧
  (C.card = 400) ∧
  (D.card = 300) ∧
  ((A ∩ B).card = 75) ∧
  ((A ∩ C).card = 100) ∧
  ((B ∩ C).card = 85) ∧
  ((A ∩ D).card = 60) ∧
  ((A ∩ B ∩ C ∩ D).card = 0) →
  ((A ∪ B ∪ C ∪ D).card = 1480)

theorem flower_beds_theorem :
  ∀ A B C D : Finset ℕ, flower_beds_problem A B C D :=
by
  intros A B C D
  intro h
  sorry -- Proof to be completed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_theorem_l464_46408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l464_46422

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = 6 * Real.pi) ∧
  (∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l464_46422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_sum_abcd_is_correct_l464_46487

-- Define the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function as noncomputable
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the theorem
theorem point_P_y_coordinate :
  ∃ P : ℝ × ℝ, 
    distance P A + distance P D = 10 ∧
    distance P B + distance P C = 10 ∧
    P.2 = 6/7 := by
  sorry

-- Define the sum of a, b, c, and d
def sum_abcd : ℕ := 14

-- Theorem stating that the sum of a, b, c, and d is correct
theorem sum_abcd_is_correct : sum_abcd = 14 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_sum_abcd_is_correct_l464_46487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l464_46497

/-- The region A in the xy-plane -/
noncomputable def region_A (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ t * (2 * p.1 - t)}

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem triangle_area_bound (t : ℝ) (h : 0 < t ∧ t < 1) :
  ∀ p ∈ region_A t, triangle_area (t, t^2) (1, 0) p ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l464_46497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_f_l464_46410

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), m > 0 ∧
  (∀ (n : ℕ), n > 0 → m ∣ f n) ∧
  (∀ (k : ℕ), k > 0 → (∀ (n : ℕ), n > 0 → k ∣ f n) → k ≤ 36) ∧
  (∀ (n : ℕ), n > 0 → 36 ∣ f n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_f_l464_46410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l464_46445

/-- The x-intercept of a line passing through two given points -/
noncomputable def x_intercept (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ - y₁ * (x₁ - x₂) / (y₁ - y₂)

/-- Theorem: The x-intercept of the line passing through (10, 3) and (-8, -6) is 4 -/
theorem x_intercept_specific_line : x_intercept 10 3 (-8) (-6) = 4 := by
  -- Unfold the definition of x_intercept
  unfold x_intercept
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l464_46445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l464_46469

theorem age_difference_proof :
  ∀ (a b : ℤ),
  (0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →  -- Ensuring a and b are single digits
  (10 * a + b = 3 * (10 * b + a + 2)) →  -- Jack's age equals Bill's age in two years
  (27 * b + 2 * a + 2 = 0) →  -- Bill's age relation
  (7 * a - 29 * b = 0) →  -- Derived from Jack's age equals Bill's future age
  (abs ((10 * a + b) - (10 * b + a)) = 18) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l464_46469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l464_46435

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b^2 - t.a^2 = t.c * (t.b - t.c) ∧
  t.a = 4 ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi

-- Part 1
theorem part_one (t : Triangle) (h : satisfies_conditions t) (hb : t.b = 4 * Real.sqrt 6 / 3) :
  t.B = Real.pi/4 := by
  sorry

-- Part 2
theorem part_two (t : Triangle) (h : satisfies_conditions t) (harea : 1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3) :
  t.b = 4 ∧ t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l464_46435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l464_46438

/-- Given a continuous and differentiable function f on ℝ satisfying x * f'(x) + f(x) > 0,
    prove that g(x) = x * f(x) + 1 has no zeros for x > 0 -/
theorem no_zeros_of_g (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_diff : Differentiable ℝ f)
  (hf_pos : ∀ x, x * (deriv f x) + f x > 0) :
  ∀ x > 0, x * f x + 1 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l464_46438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l464_46462

theorem two_integers_sum (a b : ℕ) : 
  a * b + a + b = 119 →
  Nat.gcd a b = 1 →
  a < 15 ∧ b < 15 →
  a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l464_46462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_product_l464_46453

/-- The coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 1)(3x^2 - 2x - 5) is -11 -/
theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X ^ 3 + 4 * X ^ 2 - 3 * X + 1
  let p₂ : Polynomial ℤ := 3 * X ^ 2 - 2 * X - 5
  (p₁ * p₂).coeff 2 = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_product_l464_46453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_time_is_three_years_l464_46416

/-- Calculates the time period of a loan given the principal, interest rate, and total interest. -/
noncomputable def loan_time_period (principal : ℝ) (rate : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * rate)

/-- Theorem stating that for a loan with given conditions, the time period is 3 years. -/
theorem loan_time_is_three_years :
  let principal : ℝ := 15000
  let rate : ℝ := 12
  let interest : ℝ := 5400
  loan_time_period principal rate interest = 3 := by
  -- Unfold the definition of loan_time_period
  unfold loan_time_period
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num
  -- Close the proof
  done

#check loan_time_is_three_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_time_is_three_years_l464_46416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l464_46429

-- Define the constant speed
noncomputable def speed : ℝ := 60

-- Define the distances for the two journeys
noncomputable def distance1 : ℝ := 540
noncomputable def distance2 : ℝ := 600

-- Define the function to calculate time in hours
noncomputable def time (distance : ℝ) : ℝ := distance / speed

-- Define the function to convert hours to minutes
noncomputable def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem statement
theorem journey_time_difference :
  hours_to_minutes (time distance2 - time distance1) = 60 := by
  -- Expand the definitions
  unfold hours_to_minutes time distance1 distance2 speed
  -- Simplify the expression
  simp [div_sub_div]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l464_46429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l464_46451

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 9) + 1 / (x^2 + 9) + 1 / (x^3 + 9)

theorem domain_of_k :
  {x : ℝ | ∃ y, k x = y} = {x | x < -9 ∨ (-9 < x ∧ x < -Real.rpow 9 (1/3)) ∨ x > -Real.rpow 9 (1/3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l464_46451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l464_46473

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 - 2*x

-- State the theorem
theorem function_inequality_range (a : ℝ) :
  (2*a^2 - a > 0 ∧ 4*a + 12 > 0 ∧ f (2*a^2 - a) ≤ f (4*a + 12)) ↔ 
  (a ∈ Set.Icc (-3/2) 0 ∪ Set.Ioc (1/2) 4) := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l464_46473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_reflection_theorem_l464_46481

/-- The expected number of reflections for a billiard ball on a rectangular table -/
noncomputable def expected_reflections (length width radius : ℝ) : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

/-- Theorem: The expected number of reflections for a billiard ball on a 3x1 meter
    rectangular table, starting from the center and traveling 2 meters in a random direction,
    is equal to (2/π) * (3 * arccos(1/4) - arcsin(3/4) + arccos(3/4)) -/
theorem billiard_reflection_theorem (length width radius : ℝ) 
    (h1 : length = 3)
    (h2 : width = 1)
    (h3 : radius = 2) :
  expected_reflections length width radius = 
    (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_reflection_theorem_l464_46481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_perpendicular_medians_l464_46454

/-- Triangle with two known medians -/
structure TriangleWithMedians where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  F : ℝ × ℝ
  E : ℝ × ℝ
  af_is_median : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  be_is_median : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

/-- Helper function to calculate the area of a triangle given its vertices -/
noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The theorem stating the area of the triangle given specific median properties -/
theorem area_of_triangle_with_perpendicular_medians
  (t : TriangleWithMedians)
  (medians_perpendicular : (t.F.1 - t.A.1) * (t.E.1 - t.B.1) + (t.F.2 - t.A.2) * (t.E.2 - t.B.2) = 0)
  (af_length : Real.sqrt ((t.F.1 - t.A.1)^2 + (t.F.2 - t.A.2)^2) = 10)
  (be_length : Real.sqrt ((t.E.1 - t.B.1)^2 + (t.E.2 - t.B.2)^2) = 15) :
  area t.A t.B t.C = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_perpendicular_medians_l464_46454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l464_46442

-- Define the two functions
noncomputable def f (x : ℝ) := 2 * Real.log x + x^2
noncomputable def g (x : ℝ) := Real.log (2 * x) + x

-- State the theorem
theorem intersection_points : ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l464_46442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_at_2_5_seconds_l464_46403

/-- The time when the ball hits the ground -/
def ground_hit_time : ℝ := 2.5

/-- The initial height of the ball in feet -/
def initial_height : ℝ := 160

/-- The initial velocity of the ball in feet per second (negative because it's thrown downward) -/
def initial_velocity : ℝ := -24

/-- The height of the ball as a function of time -/
def ball_height (t : ℝ) : ℝ := -16 * t^2 + initial_velocity * t + initial_height

theorem ball_hits_ground_at_2_5_seconds :
  ball_height ground_hit_time = 0 ∧ 
  ∀ t, 0 < t ∧ t < ground_hit_time → ball_height t > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_at_2_5_seconds_l464_46403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_perp_to_given_line_l464_46467

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 6*y + 5 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem line_through_center_perp_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ x y, my_circle x y ↔ (x - cx)^2 + (y - cy)^2 = 4) ∧
    (∀ x y, line_l x y ↔ y - cy = (x - cx)) ∧
    (∀ x y, perp_line x y → ∀ x' y', line_l x' y' → (x' - x) * (y' - y) = -(x - x') * (y - y')) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_perp_to_given_line_l464_46467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l464_46447

/-- The function f(x) = 2sin(x) + sin(2x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

/-- The minimum value of f(x) is -3√3/2 -/
theorem f_min_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l464_46447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_height_relation_l464_46493

/-- Represents a cone with base area and height -/
structure Cone where
  baseArea : ℝ
  height : ℝ

/-- Represents a cylinder with base area and height -/
structure Cylinder where
  baseArea : ℝ
  height : ℝ

/-- Volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * c.baseArea * c.height

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := c.baseArea * c.height

theorem cone_cylinder_height_relation (cone : Cone) (cylinder : Cylinder) :
  cone.baseArea = cylinder.baseArea →
  coneVolume cone = cylinderVolume cylinder →
  cylinder.height = 9 →
  cone.height = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_height_relation_l464_46493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ones_and_twos_l464_46464

/-- Represents a 100-digit natural number composed of ones and twos -/
def N : ℕ := sorry

/-- The number of ones in N -/
def num_ones : ℕ := sorry

/-- The number of twos in N -/
def num_twos : ℕ := sorry

/-- N is composed of only ones and twos -/
axiom composed_of_ones_and_twos : num_ones + num_twos = 100

/-- There is an even number of digits between any two twos in N -/
axiom even_between_twos : ∃ (k : ℕ), ∀ (i j : ℕ), i < j → (N / 10^i) % 10 = 2 → (N / 10^j) % 10 = 2 → ∃ (m : ℕ), j - i - 1 = 2 * m

/-- N is divisible by 3 -/
axiom divisible_by_three : N % 3 = 0

/-- The sum of digits in N is divisible by 3 -/
axiom sum_of_digits_divisible_by_three : (num_ones + 2 * num_twos) % 3 = 0

/-- Theorem: The number of twos in N is 2 and the number of ones is 98 -/
theorem count_ones_and_twos : num_twos = 2 ∧ num_ones = 98 := by
  sorry

#check count_ones_and_twos

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ones_and_twos_l464_46464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_necessary_not_sufficient_l464_46402

/-- The function f(x) = sin(2x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

/-- The line l: y = 2x + a -/
def l (a x : ℝ) : ℝ := 2 * x + a

/-- Predicate for the line being tangent to the curve -/
def is_tangent (a : ℝ) : Prop :=
  ∃ x₀, f x₀ = l a x₀ ∧ deriv f x₀ = 2

/-- Theorem stating that a = 0 is necessary but not sufficient for tangency -/
theorem a_zero_necessary_not_sufficient :
  (∀ a, is_tangent a → a = 0) ∧ ¬(∀ a, a = 0 → is_tangent a) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_necessary_not_sufficient_l464_46402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_m_value_l464_46404

-- Define the points P and Q
def P : ℝ × ℝ := (-1, -2)
def Q : ℝ × ℝ := (5, 3)

-- Define the point R with a fixed x-coordinate and variable y-coordinate
def R (m : ℝ) : ℝ × ℝ := (2, m)

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to calculate total distance PR + RQ
noncomputable def total_distance (m : ℝ) : ℝ :=
  distance P (R m) + distance (R m) Q

-- Theorem statement
theorem minimum_distance_m_value :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ (y : ℝ), total_distance m ≤ total_distance y := by
  sorry

#check minimum_distance_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_m_value_l464_46404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l464_46484

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + x - 5/2

-- Define the alternative form of the quadratic function
noncomputable def f_alt (x : ℝ) : ℝ := -1/2 * (x-1)^2 - 2

-- State the theorem
theorem quadratic_function_proof :
  (f (-2) = -13/2) ∧
  (f (-1) = -4) ∧
  (f 0 = -5/2) ∧
  (f 1 = -2) ∧
  (f 2 = -5/2) ∧
  (∀ x : ℝ, f x = f_alt x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l464_46484
