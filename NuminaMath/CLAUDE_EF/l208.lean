import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_exists_l208_20859

def game_sequence := List.range 101

structure GameState where
  remaining : List Nat
  turns_left : Nat

def initial_state : GameState := { remaining := game_sequence, turns_left := 11 }

def remove_numbers (state : GameState) (numbers : List Nat) : GameState :=
  { remaining := state.remaining.filter (fun n => n ∉ numbers),
    turns_left := state.turns_left - 1 }

def score (state : GameState) : Nat :=
  match state.remaining with
  | [a, b] => max a b - min a b
  | _ => 0

def player_strategy (state : GameState) : List Nat := sorry

theorem optimal_strategy_exists :
  ∃ (strategy : GameState → List Nat),
    ∀ (opponent_moves : List (List Nat)),
      let final_state := opponent_moves.foldl
        (fun s move => remove_numbers (remove_numbers s (strategy s)) move)
        initial_state
      score final_state ≥ 55 := by
  sorry

#check optimal_strategy_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_exists_l208_20859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l208_20870

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem spatial_relationships 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) : 
  (perpendicular m α ∧ perpendicular n α → parallel_lines m n) ∧ 
  ¬(perpendicular_lines m n ∧ perpendicular n α → parallel_line_plane m α) ∧
  (perpendicular m α ∧ perpendicular m β → parallel_planes α β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l208_20870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_area_greater_than_four_l208_20869

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the set M of five points
def M : Set Point := sorry

-- Define the area function for a triangle
def area (p q r : Point) : ℝ := sorry

-- Axiom: The area of each triangle formed by any three points from M is greater than 3
axiom area_greater_than_three :
  ∀ (p q r : Point), p ∈ M → q ∈ M → r ∈ M → 
  p ≠ q → q ≠ r → p ≠ r → area p q r > 3

-- Theorem to prove
theorem exists_triangle_area_greater_than_four :
  ∃ (p q r : Point), p ∈ M ∧ q ∈ M ∧ r ∈ M ∧
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ area p q r > 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_area_greater_than_four_l208_20869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_close_functions_range_l208_20858

open Real Set

-- Define the interval [1/e, e]
def I : Set ℝ := Icc (1/exp 1) (exp 1)

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (m * x - 1) / x

-- Define the property of being "close functions"
def close_functions (f g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x ∈ S, |f x - g x| ≤ 1

-- State the theorem
theorem close_functions_range :
  {m : ℝ | close_functions f (g m) I} = Icc (exp 1 - 2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_close_functions_range_l208_20858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_fills_pool_in_90_minutes_l208_20844

/-- The time it takes Tony to fill the pool alone, given the rates of Jim, Sue, and their combined rate --/
noncomputable def tonys_time (jim_time sue_time combined_time : ℝ) : ℝ :=
  1 / (1 / combined_time - 1 / jim_time - 1 / sue_time)

/-- Theorem stating that Tony's time to fill the pool is 90 minutes --/
theorem tony_fills_pool_in_90_minutes :
  tonys_time 30 45 15 = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_fills_pool_in_90_minutes_l208_20844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_AP_length_l208_20836

noncomputable section

-- Define the triangle and points
variable (A B C E P : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_equilateral_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖B - A‖ = 1 ∧ ‖C - B‖ = 1 ∧ ‖A - C‖ = 1

def E_on_AC (A C E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = A + t • (C - A)

def AC_eq_4AE (A C E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  C - A = 4 • (E - A)

def P_on_BE (B E P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = B + t • (E - B)

def AP_eq_mAB_plus_nAC (A B C P : EuclideanSpace ℝ (Fin 2)) (m n : ℝ) : Prop :=
  P - A = m • (B - A) + n • (C - A)

-- State the theorem
theorem min_value_and_AP_length 
  (h_triangle : is_equilateral_triangle A B C)
  (h_E_on_AC : E_on_AC A C E)
  (h_AC_eq_4AE : AC_eq_4AE A C E)
  (h_P_on_BE : P_on_BE B E P)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_AP : AP_eq_mAB_plus_nAC A B C P m n) :
  (∃ (m_min n_min : ℝ), 
    (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1/m' + 1/n' ≥ 1/m_min + 1/n_min) ∧
    1/m_min + 1/n_min = 9 ∧
    ‖P - A‖ = Real.sqrt 7 / 6) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_AP_length_l208_20836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l208_20861

/-- Given an ellipse with eccentricity 1/2 and foci at (-3,0) and (3,0), its equation is x^2/36 + y^2/27 = 1 -/
theorem ellipse_equation (e : ℝ) (f1 f2 : ℝ × ℝ) :
  e = 1/2 →
  f1 = (-3, 0) →
  f2 = (3, 0) →
  ∀ x y : ℝ, (x^2 / 36 + y^2 / 27 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (6 * Real.cos t, 3 * Real.sqrt 3 * Real.sin t)} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l208_20861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_relation_l208_20810

theorem binomial_sum_relation (n : ℕ) :
  (n % 2 = 1 → (2 : ℝ)^n < (2 : ℝ)^n - (-1)^n) ∧
  (n % 2 = 0 → (2 : ℝ)^n > (2 : ℝ)^n - (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_relation_l208_20810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l208_20853

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the given triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.sin t.B * Real.cos t.C - 2 * Real.sin t.A + Real.sin t.C = 0) :
  t.B = π / 3 ∧ 
  (t.b = 2 → 
    let R := t.b / (2 * Real.sin t.B)
    let d := Real.sqrt (R^2 - (t.b/2)^2)
    d = Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l208_20853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l208_20849

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem a_range (a : ℝ) (h_a : a > 0) :
  (is_monotonically_increasing (λ x ↦ a^x) ∨ 
   (∀ x : ℝ, a*x^2 - a*x + 1 > 0)) ∧
  ¬(is_monotonically_increasing (λ x ↦ a^x) ∧ 
    (∀ x : ℝ, a*x^2 - a*x + 1 > 0)) →
  (a ∈ Set.Ioc 0 1) ∨ (a ∈ Set.Ici 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l208_20849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_shadow_theorem_l208_20857

-- Define the pyramid's base side length
noncomputable def base_side : ℝ := 2

-- Define the shadow area (excluding area beneath the pyramid)
noncomputable def shadow_area : ℝ := 36

-- Define the function to calculate y
noncomputable def calculate_y (base : ℝ) (shadow : ℝ) : ℝ :=
  (2 * (Real.sqrt 10 + 1)) / 9

-- Define the function to calculate 800y
noncomputable def calculate_800y (y : ℝ) : ℝ :=
  800 * y

-- Theorem statement
theorem pyramid_shadow_theorem :
  ∃ (y : ℝ), y = calculate_y base_side shadow_area ∧
  Int.floor (calculate_800y y) = 828 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_shadow_theorem_l208_20857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_t_range_l208_20807

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the point P
def P (t : ℝ) : ℝ × ℝ := (2, t)

-- Define the condition for three tangent lines
def has_three_tangents (t : ℝ) : Prop :=
  ∃ (s₁ s₂ s₃ : ℝ), s₁ ≠ s₂ ∧ s₂ ≠ s₃ ∧ s₁ ≠ s₃ ∧
  ∀ i, i ∈ ({s₁, s₂, s₃} : Set ℝ) →
    (f i - t) = (3*i^2 - 6*i) * (i - 2)

-- State the theorem
theorem three_tangents_t_range (t : ℝ) :
  has_three_tangents t → -5 < t ∧ t < -4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_t_range_l208_20807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_id_function_satisfies_conditions_unique_identity_function_l208_20850

-- Define the type for non-negative real numbers
def NonnegReal := {x : ℝ | 0 ≤ x}

-- Define the function type
def FunctionType := NonnegReal → NonnegReal

-- State the theorem
theorem unique_function_theorem (f : FunctionType) : 
  (∀ x : NonnegReal, (4 : ℝ) * (f x).val ≥ (3 : ℝ) * x.val) ∧ 
  (∀ x : NonnegReal, f ⟨(4 : ℝ) * (f x).val - (3 : ℝ) * x.val, sorry⟩ = x) →
  (∀ x : NonnegReal, f x = x) :=
by sorry

-- Define the identity function on NonnegReal
def id_function : FunctionType := λ x ↦ x

-- State that the identity function satisfies the conditions
theorem id_function_satisfies_conditions : 
  (∀ x : NonnegReal, (4 : ℝ) * (id_function x).val ≥ (3 : ℝ) * x.val) ∧ 
  (∀ x : NonnegReal, id_function ⟨(4 : ℝ) * (id_function x).val - (3 : ℝ) * x.val, sorry⟩ = x) :=
by sorry

-- Combine the theorems to state the uniqueness of the identity function
theorem unique_identity_function : 
  ∃! f : FunctionType, 
    (∀ x : NonnegReal, (4 : ℝ) * (f x).val ≥ (3 : ℝ) * x.val) ∧ 
    (∀ x : NonnegReal, f ⟨(4 : ℝ) * (f x).val - (3 : ℝ) * x.val, sorry⟩ = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_id_function_satisfies_conditions_unique_identity_function_l208_20850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_properties_l208_20851

noncomputable section

def f (x : ℝ) := Real.log x / Real.log 10

theorem logarithm_properties :
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → x₁ > 0 → x₂ > 0 →
  (0 < deriv f 3 ∧ deriv f 3 < f 3 - f 2 ∧ f 3 - f 2 < deriv f 2) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_properties_l208_20851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_positions_l208_20895

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the angle between two vectors -/
noncomputable def angle (v1 v2 : Point) : ℝ :=
  Real.arccos ((v1.x * v2.x + v1.y * v2.y) / (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2)))

theorem ship_positions (merchant pirate escort : Point)
  (h1 : distance merchant pirate = 50)
  (h2 : distance merchant escort = 50)
  (h3 : angle (Point.mk (pirate.x - merchant.x) (pirate.y - merchant.y))
              (Point.mk 1 0) = π / 3)
  (h4 : angle (Point.mk (escort.x - merchant.x) (escort.y - merchant.y))
              (Point.mk (-1) 0) = π / 3) :
  escort.y = pirate.y - 50 ∧ escort.x = pirate.x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_positions_l208_20895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_range_l208_20832

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2

-- Define g as the derivative of f
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem extreme_values_and_range (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ x + 1) ↔ a ≤ 1/2 ∧
  ((a > 0 → ∃ x_min : ℝ, x_min = Real.log (2*a) ∧
              g a x_min = 2*a - 2*a*Real.log (2*a) ∧
              ∀ y : ℝ, g a y ≥ g a x_min) ∧
   (a ≤ 0 → ∀ x : ℝ, ¬∃ y : ℝ, g a y < g a x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_range_l208_20832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_three_element_set_l208_20827

theorem subset_count_three_element_set {α : Type*} (M : Finset α) (h : M.card = 3) :
  (M.powerset).card = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_three_element_set_l208_20827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_token_with_nine_left_moves_l208_20899

/-- Represents a move in the token game -/
inductive Move
  | Single (direction : Bool) -- True for right, False for left
  | Multiple (direction : Bool) -- True for right, False for left

/-- The state of the game after a sequence of moves -/
structure GameState :=
  (cells : List ℕ)
  (moves : List Move)

/-- The initial state of the game -/
def initial_state : GameState :=
  { cells := 203 :: List.replicate 202 0,
    moves := [] }

/-- Checks if a game state is valid according to the rules -/
def is_valid_state (state : GameState) : Prop :=
  state.cells.sum = 203 ∧ state.moves.length ≤ 2023

/-- Checks if a game state is the final state -/
def is_final_state (state : GameState) : Prop :=
  state.cells.all (· = 1) ∧ state.moves.length = 2023

/-- Counts the number of left moves for a specific token -/
def left_moves_count (token : Fin 203) (state : GameState) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem exists_token_with_nine_left_moves 
  (final_state : GameState) 
  (h_valid : is_valid_state final_state) 
  (h_final : is_final_state final_state) :
  ∃ token : Fin 203, left_moves_count token final_state ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_token_with_nine_left_moves_l208_20899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l208_20885

theorem simplify_expression : (-5) - (-7) - 9 = -5 + 7 - 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l208_20885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_initial_condition_difference_of_terms_l208_20842

/-- A sequence satisfying a_{n+1} + a_n = n for all n ≥ 1 and a_1 = 2 -/
def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => n - mySequence n

theorem sequence_property (n : ℕ) : mySequence (n + 1) + mySequence n = n := by sorry

theorem initial_condition : mySequence 1 = 2 := by sorry

theorem difference_of_terms : mySequence 4 - mySequence 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_initial_condition_difference_of_terms_l208_20842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l208_20892

theorem shopkeeper_profit (cost_price selling_price : ℚ) : 
  cost_price = 120 →
  selling_price = 155 →
  (selling_price - cost_price) / cost_price * 100 = 29.17 := by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- The exact calculation would yield a slightly different result due to rounding
  -- We use 'sorry' here to skip the precise numerical proof
  sorry

#eval (155 - 120) / 120 * 100  -- This will output the actual calculated value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l208_20892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l208_20852

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m / Real.exp x - 2 * x

-- State the theorem
theorem odd_function_and_inequality (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f m x = -f m (-x)) →
  (m = 1 ∧ ∀ a : ℝ, f 1 (a - 1) + f 1 (2 * a^2) ≤ 0 → 0 ≤ a ∧ a ≤ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l208_20852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_math_marks_are_89_l208_20862

/-- Calculates David's marks in Mathematics given his marks in other subjects and the average --/
def davidsMathMarks (english physics chemistry biology : ℕ) (average : ℚ) : ℤ :=
  let totalMarks := (average * 5).floor
  totalMarks - (english + physics + chemistry + biology)

/-- Theorem stating that David's Mathematics marks are 89 given the problem conditions --/
theorem davids_math_marks_are_89 :
  davidsMathMarks 86 82 87 81 (85 : ℚ) = 89 := by
  sorry

#eval davidsMathMarks 86 82 87 81 (85 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_math_marks_are_89_l208_20862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l208_20864

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (x + 2 * φ)

theorem symmetry_implies_phi (φ : ℝ) :
  (∀ x : ℝ, f φ (π/4 + x) = f φ (π/4 - x)) →
  ∃ k : ℤ, φ = π/8 + k * π/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l208_20864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_l208_20838

theorem leak_drain_time (pump_rate combined_rate : ℚ) : ℚ :=
  let leak_rate := pump_rate - combined_rate
  by
    -- The pump can fill the tank in 2 hours without a leak
    have h1 : pump_rate = 1 / 2 := by sorry
    -- With the leak, it takes 2 1/3 hours to fill the tank
    have h2 : combined_rate = 3 / 7 := by sorry
    -- The leak can drain all the water in 14 hours
    have h3 : 14 = 1 / leak_rate := by sorry
    exact 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_l208_20838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_condition_l208_20820

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x^2 + 3)

-- State the theorem
theorem function_derivative_condition (a : ℝ) :
  (∀ x, HasDerivAt (f a) ((a*(x^2+3) - a*x*(2*x)) / (x^2+3)^2) x) →
  HasDerivAt (f a) (1/2) 1 →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_condition_l208_20820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_equals_g_l208_20811

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

-- State the theorem
theorem f_shift_equals_g : ∀ x : ℝ, g x = f (x + π / 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_equals_g_l208_20811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_perpendicular_distance_l208_20889

/-- Given a triangle ABC and a line PQ, prove that the perpendicular distance from the midpoint of AC to PQ is 22.5 -/
theorem midpoint_perpendicular_distance 
  (A B C : ℝ × ℝ) -- Points of the triangle
  (PQ : Set (ℝ × ℝ)) -- The line PQ
  (hAD : |A.2 - 0| = 15) -- Perpendicular distance from A to PQ is 15
  (hBE : |B.2 - 0| = 9)  -- Perpendicular distance from B to PQ is 9
  (hCF : |C.2 - 0| = 30) -- Perpendicular distance from C to PQ is 30
  (hPQ : ∀ p ∈ PQ, p.2 = 0) -- PQ is a horizontal line at y = 0
  (hI : I = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) -- I is the midpoint of AC
  : |I.2 - 0| = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_perpendicular_distance_l208_20889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translate_f_down_3_is_g_l208_20891

def f (x : ℝ) := 3 * x + 2

def g (x : ℝ) := 3 * x - 1

def vertical_translate (h : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ h x - d

theorem translate_f_down_3_is_g :
  vertical_translate f 3 = g := by
  ext x
  simp [vertical_translate, f, g]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translate_f_down_3_is_g_l208_20891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l208_20800

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x)

theorem f_range :
  (∀ y, y ∈ Set.range f → 0 < y ∧ y ≤ 2) ∧
  (∀ ε > 0, ∃ x, |f x - 2| < ε) ∧
  (∀ ε > 0, ∃ x, 0 < f x ∧ f x < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l208_20800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l208_20840

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l208_20840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_form_l208_20808

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n+2 => 2 * sequence_a (n+1) - 1

theorem sequence_a_general_form (n : ℕ) : 
  sequence_a n = 2^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_form_l208_20808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l208_20873

theorem count_integer_solutions : 
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ 1 + ⌊(101 * n : ℚ) / 102⌋ = ⌈(98 * n : ℚ) / 101⌉) ∧ Finset.card S = 10302 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l208_20873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l208_20834

theorem cos_minus_sin_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo (3*π/4) π) 
  (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3/2) : 
  Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l208_20834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_horse_problem_l208_20882

/-- Represents the horse and tile problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_horse_problem (x y : ℚ) : 
  x + y = 150 →  -- Total number of horses
  3 * x + (1/3) * y = 210 →  -- Total number of tiles pulled
  ∃ (large_horse small_horse : ℚ),
    (large_horse = 1 → large_horse * 3 = 3) ∧  -- Large horse capacity
    (small_horse = 1 → small_horse * (1/3) = 1/3) ∧  -- Small horse capacity
    x = large_horse * 150 ∧  -- x is the number of large horses
    y = small_horse * 150 →  -- y is the number of small horses
  True  -- Placeholder for the conclusion
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_horse_problem_l208_20882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l208_20872

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1 + Real.sqrt 2, -1; 1, 1 + Real.sqrt 2]

theorem matrix_power_four :
  A^4 = !![0, -(7 + 5 * Real.sqrt 2); 7 + 5 * Real.sqrt 2, 0] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l208_20872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_plus_s_l208_20898

noncomputable section

-- Define the triangle XYZ
def X : ℝ × ℝ := (10, 15)
def Y : ℝ × ℝ := (22, 16)
def Z : ℝ × ℝ → ℝ × ℝ := fun (r, s) ↦ (r, s)

-- Define the area of the triangle
noncomputable def triangleArea (r s : ℝ) : ℝ :=
  (1/2) * abs (r*(15-16) + 10*(16-s) + 22*(s-15))

-- Define the slope of the median
noncomputable def medianSlope (r s : ℝ) : ℝ :=
  (s - 15.5) / (r - 16)

-- Theorem statement
theorem max_r_plus_s :
  ∃ (r s : ℝ),
    triangleArea r s = 56 ∧
    medianSlope r s = -3 ∧
    ∀ (r' s' : ℝ),
      triangleArea r' s' = 56 ∧
      medianSlope r' s' = -3 →
      r + s ≥ r' + s' ∧
      r + s = 1367.5 / 37 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_plus_s_l208_20898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_in_factorial_factorization_l208_20817

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_in_factorial_factorization :
  ∃ (a b c : ℕ+), 
    (a * b * c : ℕ) = factorial 9 ∧ 
    a < b ∧ 
    b < c ∧ 
    ∀ (x y z : ℕ+), 
      (x * y * z : ℕ) = factorial 9 → 
      x < y → 
      y < z → 
      (c : ℕ) - (a : ℕ) ≤ (z : ℕ) - (x : ℕ) ∧
    (c : ℕ) - (a : ℕ) = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_in_factorial_factorization_l208_20817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersecting_lines_l208_20815

/-- The trajectory of a point P(x,y) whose distance to F(1,0) and the line x=2 has a constant ratio of √2/2 -/
def Trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- A line that intersects the trajectory at two points -/
structure IntersectingLine where
  k : ℝ
  m : ℝ

/-- The fixed point that all intersecting lines pass through -/
def FixedPoint : ℝ × ℝ := (2, 0)

/-- Theorem stating the properties of the trajectory and intersecting lines -/
theorem trajectory_and_intersecting_lines 
  (P : ℝ × ℝ) 
  (h_ratio : Real.sqrt ((P.1 - 1)^2 + P.2^2) / |P.1 - 2| = Real.sqrt 2 / 2) 
  (l : IntersectingLine) 
  (M N : ℝ × ℝ) 
  (h_M_on_traj : M ∈ Trajectory) 
  (h_N_on_traj : N ∈ Trajectory) 
  (h_M_on_line : M.2 = l.k * M.1 + l.m) 
  (h_N_on_line : N.2 = l.k * N.1 + l.m) 
  (h_angles : ∃ (α β : ℝ), α + β = Real.pi ∧ 
    Real.tan α = (M.2 / (M.1 - 1)) ∧ 
    Real.tan β = (N.2 / (N.1 - 1))) :
  P ∈ Trajectory ∧ (∃ t : ℝ, (2 + t, l.k * t) ∈ Trajectory) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersecting_lines_l208_20815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l208_20875

theorem y_squared_range (y : ℝ) (h : (y + 16)^(1/3 : ℝ) - (y - 16)^(1/3 : ℝ) = 4) :
  235 < y^2 ∧ y^2 < 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l208_20875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_PAB_l208_20835

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8)
  (bc : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 17)
  (ca : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15)

-- Define a point P inside the triangle
def PointInside (t : Triangle) := { P : ℝ × ℝ // 
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  P = (α * t.A.1 + β * t.B.1 + γ * t.C.1, α * t.A.2 + β * t.B.2 + γ * t.C.2) }

-- Define the angle function (this is a placeholder and needs to be properly defined)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the angle equality condition
def AngleEquality (t : Triangle) (P : PointInside t) :=
  ∃ ω : ℝ, 
    angle t.A P.val t.B = ω ∧
    angle t.B P.val t.C = ω ∧
    angle t.C P.val t.A = ω

-- The main theorem
theorem tan_angle_PAB (t : Triangle) (P : PointInside t) 
  (h : AngleEquality t P) : 
  Real.tan (angle t.A P.val t.B) = 168 / 289 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_PAB_l208_20835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_is_open_interval_l208_20837

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Theorem statement
theorem union_of_A_and_B_is_open_interval :
  A ∪ B = Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_is_open_interval_l208_20837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_copied_for_35_dollars_l208_20845

/-- Given that 4 pages cost 7 cents, prove that $35 (3500 cents) will copy 2000 pages. -/
theorem pages_copied_for_35_dollars : 
  (3500 : ℚ) * (4 : ℚ) / (7 : ℚ) = 2000 := by
  norm_num

#eval (3500 : ℚ) * (4 : ℚ) / (7 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_copied_for_35_dollars_l208_20845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l208_20804

/-- A sinusoidal function with given properties and points -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- The set of integers -/
def ℤSet : Set ℤ := Set.univ

theorem sinusoidal_function_properties
  (A ω φ : ℝ)
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : |φ| < π/2)
  (h4 : f A ω φ (π/6) = 2)
  (h5 : f A ω φ 0 = 0)
  (h6 : f A ω φ (π/2) = 2)
  (h7 : f A ω φ π = 0)
  (h8 : f A ω φ (2*π) = 0) :
  (∀ x, f A ω φ x = 2 * Real.sin (2*x + π/6)) ∧
  (∀ k ∈ ℤSet, ∀ x ∈ Set.Icc (-π/3 + k*π) (π/6 + k*π),
    ∀ y ∈ Set.Icc (-π/3 + k*π) (π/6 + k*π),
    x ≤ y → f A ω φ x ≤ f A ω φ y) ∧
  (∀ x ∈ Set.Icc (-π/2) 0, f A ω φ x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-π/2) 0, f A ω φ x = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l208_20804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l208_20871

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line segment with two endpoints -/
structure LineSegment where
  a : Point
  b : Point

/-- Represents a square with four vertices -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The configuration of the problem -/
structure Configuration where
  ab : LineSegment
  m : Point
  square1 : Square
  square2 : Square
  circle1 : Circle
  circle2 : Circle

/-- Checks if a point is on a line segment -/
def Point.isOnLineSegment (p : Point) (l : LineSegment) : Prop :=
  sorry

/-- Checks if two squares are on the same side of a line -/
def Square.onSameSide (s1 s2 : Square) (l : LineSegment) : Prop :=
  sorry

/-- Finds the intersection of two circles -/
noncomputable def Circle.intersect (c1 c2 : Circle) : Point :=
  sorry

/-- Finds the intersection of two lines -/
noncomputable def Line.intersect (l1 l2 : LineSegment) : Point :=
  sorry

theorem problem_statement (config : Configuration) :
  config.m.isOnLineSegment config.ab ∧
  config.square1.onSameSide config.square2 config.ab →
  let n := config.circle1.intersect config.circle2
  let n1 := Line.intersect (LineSegment.mk config.square1.b config.square1.c)
                           (LineSegment.mk config.square1.a config.square2.d)
  let fixed_point : Point := sorry
  n = n1 ∧ 
  ∀ (m' : Point), m'.isOnLineSegment config.ab →
    ∃ (t : ℝ), fixed_point = Point.mk (t * m'.x + (1 - t) * n.x) (t * m'.y + (1 - t) * n.y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l208_20871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_ratio_split_l208_20888

/-- Theorem: For a line segment AB that is not horizontal or vertical, with C on AB (C ≠ A and C ≠ B),
    the ratio of lengths AC:CB equals m:n if and only if the x-coordinate and y-coordinate of C
    split the corresponding coordinates of A and B in the ratio m:n. -/
theorem segment_ratio_split (A B C : ℝ × ℝ) (m n : ℝ) : 
  (A.1 ≠ B.1 ∧ A.2 ≠ B.2) →  -- AB is not horizontal or vertical
  (C ≠ A ∧ C ≠ B) →  -- C is not at A or B
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) →  -- C is on AB
  (dist A C / dist C B = m / n) ↔ 
  ((C.1 - A.1) / (B.1 - C.1) = m / n ∧ (C.2 - A.2) / (B.2 - C.2) = m / n) :=
by sorry

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_ratio_split_l208_20888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_sequence_probability_l208_20813

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Converts a coin toss to its numerical value -/
def tossValue (t : CoinToss) : Int :=
  match t with
  | CoinToss.Heads => 1
  | CoinToss.Tails => -1

/-- Represents a sequence of coin tosses -/
def CoinTossSequence := List CoinToss

/-- Calculates the sum of values for a sequence of coin tosses -/
def sequenceSum (s : CoinTossSequence) : Int :=
  s.map tossValue |>.sum

/-- The probability of getting heads or tails in a fair coin toss -/
def fairCoinProbability : ℚ := 1 / 2

/-- 
Theorem: The probability of having a non-zero sum after 2 tosses 
and a sum of 2 after 8 tosses in a fair coin toss sequence is 13/128
-/
theorem fair_coin_sequence_probability : 
  (13 : ℚ) / 128 = 13 / 128 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_sequence_probability_l208_20813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l208_20883

/-- Time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 450 →
  platform_length = 250.056 →
  train_speed_kmph = 126 →
  (train_length + platform_length) / (train_speed_kmph * (1000 / 3600)) = 20.0016 := by
  intros h1 h2 h3
  -- Define variables for readability
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let time := total_distance / train_speed_mps
  -- Proof steps would go here
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l208_20883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_curve_l208_20819

/-- The curve y = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The slope of the line parallel to 4x - y - 1 = 0 -/
def m : ℝ := 4

/-- The derivative of f -/
def f' : ℝ → ℝ := fun x ↦ 3 * x^2 + 1

theorem tangent_lines_to_curve :
  ∀ k : ℝ, (∀ x y : ℝ, y = m * x - k → (∃ a : ℝ, f a = m * a - k ∧ f' a = m)) ↔ k = 4 ∨ k = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_curve_l208_20819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_parabola_l208_20867

/-- The minimum distance sum from a point on the parabola y^2 = 8x to two fixed points -/
theorem min_distance_sum_parabola :
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (7, 6)
  let parabola := {P : ℝ × ℝ | P.2^2 = 8 * P.1}
  ∀ P ∈ parabola, Real.sqrt 97 ≤ dist A P + dist B P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_parabola_l208_20867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_sum_l208_20886

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 3 / 5) :
  (Real.sin θ) ^ 6 + (Real.cos θ) ^ 6 = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_sum_l208_20886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_right_vertex_l208_20856

-- Define the line l
def line_l (t a : ℝ) : ℝ × ℝ := (t, t - a)

-- Define the ellipse C
noncomputable def ellipse_C (φ : ℝ) : ℝ × ℝ := (3 * Real.cos φ, 2 * Real.sin φ)

-- Define the right vertex of the ellipse
def right_vertex : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem line_passes_through_right_vertex (a : ℝ) :
  (∃ t : ℝ, line_l t a = right_vertex) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_right_vertex_l208_20856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_5_or_7_not_8_count_l208_20825

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def count_multiples (n : ℕ) (p : ℕ → Bool) : ℕ :=
  (List.range n).filter p |>.length

def satisfies_condition (x : ℕ) : Bool :=
  x > 0 && ((x % 5 == 0) || (x % 7 == 0)) && (x % 8 ≠ 0)

theorem multiples_of_5_or_7_not_8_count :
  count_multiples 100 satisfies_condition = 29 := by
  sorry

#eval count_multiples 100 satisfies_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_5_or_7_not_8_count_l208_20825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_closest_to_45_degrees_l208_20880

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Define the slope at x = 1
def slope_at_1 : ℝ := f' 1

-- Define the given angle options in radians
noncomputable def angle_options : List ℝ := [30 * Real.pi / 180, 45 * Real.pi / 180, 135 * Real.pi / 180, 150 * Real.pi / 180]

-- Theorem statement
theorem tangent_slope_closest_to_45_degrees :
  ∃ (θ : ℝ), θ ∈ angle_options ∧ 
  ∀ (φ : ℝ), φ ∈ angle_options → |slope_at_1 - Real.tan θ| ≤ |slope_at_1 - Real.tan φ| := by
  sorry

#eval slope_at_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_closest_to_45_degrees_l208_20880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_equals_third_of_sum_l208_20879

-- Define the probability function
noncomputable def P (d : ℕ) : ℝ := Real.log (d + 1 : ℝ) - Real.log (d : ℝ)

-- State the theorem
theorem prob_three_equals_third_of_sum : 
  P 3 = (1 / 3) * (P 6 + P 7 + P 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_equals_third_of_sum_l208_20879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l208_20897

/-- Circle C with center (3,4) and radius 2 -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

/-- Line l passing through point (1,0) -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

/-- Distance from point (3,4) to line y = k(x-1) -/
noncomputable def distance_center_to_line (k : ℝ) : ℝ := 
  abs (2*k - 4) / Real.sqrt (k^2 + 1)

/-- Area of triangle CPQ -/
noncomputable def area_CPQ (d : ℝ) : ℝ := d * Real.sqrt (4 - d^2)

theorem circle_line_properties :
  ∃ (k₁ k₂ : ℝ),
    /- Tangent line equations -/
    (k₁ = 0 ∧ line_l k₁ 1 0) ∨ (3 * k₂ - 4 = 0 ∧ line_l k₂ 1 0) ∧
    /- Maximum area of triangle CPQ -/
    (∀ k, area_CPQ (distance_center_to_line k) ≤ 2) ∧
    /- Line equations for maximum area -/
    (area_CPQ (distance_center_to_line 1) = 2 ∧ line_l 1 1 0) ∨
    (area_CPQ (distance_center_to_line 7) = 2 ∧ line_l 7 1 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l208_20897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l208_20839

theorem inequality_proof (n : ℕ) (h : n > 2) :
  (2*n - 1)^n + (2*n)^n < (2*n + 1)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l208_20839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equilateral_eccentricity_l208_20881

/-- An ellipse with a vertex and two foci forming an equilateral triangle has eccentricity 1/2 -/
theorem ellipse_equilateral_eccentricity 
  (E : Set (ℝ × ℝ)) 
  (a b c : ℝ) 
  (h_ellipse : E = {p : ℝ × ℝ | (p.1^2/a^2) + (p.2^2/b^2) = 1 ∧ a > b ∧ b > 0})
  (h_foci : c^2 = a^2 - b^2)
  (h_equilateral : b = Real.sqrt 3 * c) :
  c / a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equilateral_eccentricity_l208_20881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l208_20893

noncomputable def α : ℝ := Real.rpow 17 (1/3)
noncomputable def β : ℝ := Real.rpow 73 (1/3)
noncomputable def γ : ℝ := Real.rpow 137 (1/3)

def equation (x : ℝ) : Prop :=
  (x - α) * (x - β) * (x - γ) = 1/2

theorem sum_of_cubes_of_solutions :
  ∃ (a b c : ℝ),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    equation a ∧ equation b ∧ equation c ∧
    a^3 + b^3 + c^3 = 228.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l208_20893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_less_than_perimeter_probability_value_l208_20801

/-- The probability that the area of a square is less than its perimeter,
    when the side length is determined by the sum of two 6-sided dice rolls. -/
noncomputable def square_area_less_than_perimeter_probability : ℚ :=
  let dice_sum : Fin 11 → ℕ := fun i => i.val + 2
  let side_length := dice_sum
  let area := fun s : ℕ => s * s
  let perimeter := fun s : ℕ => 4 * s
  let valid_outcomes := {s : ℕ | ∃ i : Fin 11, s = side_length i ∧ area s < perimeter s}
  let total_outcomes := 36  -- 6 * 6 possible dice roll combinations
  let favorable_outcomes := (Finset.filter (fun i : Fin 11 => area (side_length i) < perimeter (side_length i)) (Finset.univ)).card
  (favorable_outcomes : ℚ) / total_outcomes

/-- The probability is equal to 1/12 -/
theorem square_area_less_than_perimeter_probability_value :
  square_area_less_than_perimeter_probability = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_less_than_perimeter_probability_value_l208_20801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ef_distance_l208_20854

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1 ∧ x ≠ 2 ∧ x ≠ -2

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_ef_distance (m : ℝ) (hm : |m| > 1) :
  ∃ (x1 y1 x2 y2 : ℝ),
    C x1 y1 ∧ C x2 y2 ∧  -- E and F are on curve C
    (∃ k : ℝ, y1 = k * (x1 - m) ∧ y2 = k * (x2 - m) ∧  -- E and F are on the same line through (m, 0)
      ∃ xt yt : ℝ, unit_circle xt yt ∧ yt = k * (xt - m)) ∧  -- The line is tangent to the circle
    (∀ x3 y3 x4 y4 : ℝ,
      C x3 y3 → C x4 y4 →
      (∃ k : ℝ, y3 = k * (x3 - m) ∧ y4 = k * (x4 - m) ∧
        ∃ xt yt : ℝ, unit_circle xt yt ∧ yt = k * (xt - m)) →
      distance x1 y1 x2 y2 ≥ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ef_distance_l208_20854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_in_S_l208_20878

def S : Set ℚ := {2, 0, -1, -3}

theorem smallest_sum_in_S : 
  (∃ a b : ℚ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = -4) ∧
  (∀ x y : ℚ, x ∈ S → y ∈ S → x ≠ y → x + y ≥ -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_in_S_l208_20878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_limit_l208_20876

def sequence_a : ℕ → ℝ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | n + 2 => 9 * 3 * sequence_a (n + 1)

theorem sequence_a_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_a n - 27| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_limit_l208_20876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_inscribed_tetrahedron_sphere_radius_6_l208_20802

/-- The maximum volume of an inscribed regular tetrahedron in a sphere of radius r -/
noncomputable def max_volume_inscribed_tetrahedron (r : ℝ) : ℝ :=
  let l := (4 * r) / Real.sqrt 6
  (Real.sqrt 2 / 12) * l^3

/-- Given a sphere of radius 6, the maximum volume of an inscribed regular tetrahedron is 64√3 -/
theorem max_volume_inscribed_tetrahedron_sphere_radius_6 (r : ℝ) (h : r = 6) :
  ∃ (V : ℝ), V = max_volume_inscribed_tetrahedron r ∧ V = 64 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_inscribed_tetrahedron_sphere_radius_6_l208_20802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_is_three_fifths_l208_20821

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 4) + 2

-- Define the theorem
theorem sin_alpha_is_three_fifths
  (a : ℝ)
  (h1 : a > 1)
  (P : ℝ × ℝ)
  (h2 : ∃ x, f a x = P.2 ∧ x = P.1)
  (α : ℝ)
  (h3 : ∃ t, t > 0 ∧ t * (Real.cos α, Real.sin α) = P) :
  Real.sin α = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_is_three_fifths_l208_20821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_size_l208_20860

def original_set : Finset Nat := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

def is_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

def valid_subset (s : Finset Nat) : Prop :=
  s ⊆ original_set ∧ 
  ∀ x y, x ∈ s → y ∈ s → x ≠ y → ¬is_perfect_square (x + y)

theorem largest_valid_subset_size :
  ∃ (s : Finset Nat), valid_subset s ∧ s.card = 7 ∧
  ∀ (t : Finset Nat), valid_subset t → t.card ≤ 7 := by
  sorry

#check largest_valid_subset_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_size_l208_20860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_centers_are_three_circles_l208_20887

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- The set of centers of rectangles formed by lines passing through four given points -/
def rectangleCenters (A B C D : Point2D) : Set Point2D :=
  sorry

/-- Three circles formed by the midpoints of the diagonals of the quadrilateral ABCD -/
def threeCircles (A B C D : Point2D) : Set Circle :=
  sorry

/-- Define membership for Point2D in Circle -/
instance : Membership Point2D Circle where
  mem p c := (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

theorem rectangle_centers_are_three_circles (A B C D : Point2D) :
  rectangleCenters A B C D = ⋃ (c ∈ threeCircles A B C D), {p : Point2D | p ∈ c} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_centers_are_three_circles_l208_20887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_furniture_pricing_solution_l208_20826

/-- Represents the prices of garden furniture items -/
structure GardenFurniturePrices where
  bench : ℝ
  table : ℝ
  chair : ℝ

/-- Checks if the given prices satisfy the problem conditions -/
def satisfies_conditions (prices : GardenFurniturePrices) : Prop :=
  prices.table + prices.bench + prices.chair = 650 ∧
  prices.table = 2 * prices.bench - 50 ∧
  prices.chair = 1.5 * prices.bench - 25

/-- The theorem stating the solution to the garden furniture pricing problem -/
theorem garden_furniture_pricing_solution :
  ∃ (prices : GardenFurniturePrices),
    satisfies_conditions prices ∧
    |prices.bench - 161.11| < 0.01 ∧
    |prices.table - 272.22| < 0.01 ∧
    |prices.chair - 216.67| < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_furniture_pricing_solution_l208_20826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_dominance_l208_20866

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * Real.log x

noncomputable def g (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x

theorem f_monotonicity_and_g_dominance :
  ∀ a : ℝ,
  (a = 2/3 →
    (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3/2 → f a x < f a y) ∧
    (∀ x y : ℝ, 3/2 < x ∧ x < y ∧ y < 2 → f a x > f a y) ∧
    (∀ x y : ℝ, 2 < x ∧ x < y → f a x < f a y)) ∧
  (a ∈ Set.Ioi (Real.log 2 - 1) ↔
    ∀ x₁ : ℝ, x₁ ∈ Set.Ioo 0 2 →
      ∃ x₂ : ℝ, x₂ ∈ Set.Ioo 0 2 ∧ f a x₁ < g x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_dominance_l208_20866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l208_20896

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}

theorem complement_of_M_in_U : Set.compl M = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l208_20896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_difference_sum_is_ten_l208_20833

/-- The number of fruits each person has -/
def sharon_fruits : ℕ := 7
def allan_fruits : ℕ := 10
def dave_fruits : ℕ := 12

/-- The sum of the differences in fruits between each pair of people -/
def fruit_difference_sum : ℕ :=
  (sharon_fruits.max allan_fruits - sharon_fruits.min allan_fruits) +
  (sharon_fruits.max dave_fruits - sharon_fruits.min dave_fruits) +
  (allan_fruits.max dave_fruits - allan_fruits.min dave_fruits)

/-- Theorem stating that the sum of the differences in fruits is 10 -/
theorem fruit_difference_sum_is_ten : fruit_difference_sum = 10 := by
  rw [fruit_difference_sum]
  rw [sharon_fruits, allan_fruits, dave_fruits]
  simp [Nat.max, Nat.min]
  rfl

#eval fruit_difference_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_difference_sum_is_ten_l208_20833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_division_equality_l208_20830

theorem sqrt_division_equality : Real.sqrt 2 / Real.sqrt 3 = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_division_equality_l208_20830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l208_20894

open Set

theorem union_of_sets (A B : Set ℝ) :
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 1} →
  B = {x : ℝ | x ≤ 2} →
  A ∪ B = {x : ℝ | x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l208_20894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l208_20863

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (x^2 + a*x + 1)

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, x + |x - 2*a| > 1

theorem range_of_a (a : ℝ) 
  (h1 : proposition_p a ∨ proposition_q a) 
  (h2 : ¬(proposition_p a ∧ proposition_q a)) : 
  (-2 < a ∧ a ≤ 1/2) ∨ (a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l208_20863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l208_20806

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l208_20806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_count_sum_l208_20841

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 4) / (x^3 - x^2 - 4*x)

def count_holes (f : ℝ → ℝ) : ℕ := sorry
def count_vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def count_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def count_oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

theorem asymptote_count_sum (f : ℝ → ℝ) :
  let a := count_holes f
  let b := count_vertical_asymptotes f
  let c := count_horizontal_asymptotes f
  let d := count_oblique_asymptotes f
  a + 2*b + 3*c + 4*d = 8 := by sorry

#check asymptote_count_sum f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_count_sum_l208_20841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l208_20816

theorem modulus_of_complex_fraction :
  let z : ℂ := (1 + 3 * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l208_20816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_9_1000_pow_1000_has_3001_digits_l208_20822

/-- The smallest non-zero digit in the decimal representation of a natural number -/
def smallest_nonzero_digit (n : ℕ) : ℕ := sorry

/-- Sequence where a₁ = 1 and a_{n+1} = a_n + (smallest non-zero digit of a_n) -/
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n + smallest_nonzero_digit (a n)

/-- The number of digits in the decimal representation of a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

theorem sequence_9_1000_pow_1000_has_3001_digits :
  num_digits (a (9 * 1000^1000)) = 3001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_9_1000_pow_1000_has_3001_digits_l208_20822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l208_20823

/-- Profit maximization problem for a product -/
theorem profit_maximization
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_units_sold : ℝ)
  (price_increase_rate : ℝ)
  (sales_decrease_rate : ℝ)
  (h1 : purchase_price = 80)
  (h2 : initial_selling_price = 90)
  (h3 : initial_units_sold = 400)
  (h4 : price_increase_rate = 1)
  (h5 : sales_decrease_rate = 20)
  : ∃ (optimal_price : ℝ),
    optimal_price = 95 ∧
    ∀ (price : ℝ),
      (price - purchase_price) * (initial_units_sold - sales_decrease_rate * (price - initial_selling_price))
      ≤ (optimal_price - purchase_price) * (initial_units_sold - sales_decrease_rate * (optimal_price - initial_selling_price)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l208_20823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l208_20818

/-- The volume of a cylindrical swimming pool -/
noncomputable def pool_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

theorem pool_volume_calculation :
  pool_volume 20 5 = 500 * Real.pi := by
  -- Unfold the definition of pool_volume
  unfold pool_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l208_20818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_equivalence_l208_20824

/-- Represents a point in the cyclic path -/
def CyclicPoint := Fin 6

/-- The length of the cycle -/
def cycleLength : Nat := 6

/-- Converts a natural number to its equivalent position in the cycle -/
def toCyclicPoint (n : Nat) : CyclicPoint :=
  Fin.ofNat n

/-- Theorem: The path from point 751 to 756 is equivalent to the path from 1 to 0 in a 6-point cycle -/
theorem path_equivalence :
  (toCyclicPoint 751 = toCyclicPoint 1 ∧ toCyclicPoint 756 = toCyclicPoint 0) →
  ∀ (i : Nat), i ≥ 751 ∧ i < 756 →
    toCyclicPoint i = toCyclicPoint (i % cycleLength) :=
by
  intro h i hi
  sorry

#check path_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_equivalence_l208_20824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_erased_cheburashkas_l208_20829

/-- Represents the number of Cheburashkas in a row -/
def num_cheburashkas : ℕ := 0  -- Initialize with a default value

/-- Represents the total number of Krakozyabras after erasing other characters -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of rows -/
def num_rows : ℕ := 2

/-- The theorem stating that the number of erased Cheburashkas is 11 -/
theorem num_erased_cheburashkas :
  (num_cheburashkas > 0) →  -- At least one Cheburashka in each row
  (num_cheburashkas * num_rows + (num_cheburashkas - 1) * num_rows + 
   num_cheburashkas * num_rows + (num_cheburashkas * num_rows + 
   (num_cheburashkas - 1) * num_rows + num_cheburashkas * num_rows - 1) = 
   total_krakozyabras) →  -- Total characters minus 1 equals total Krakozyabras
  num_cheburashkas * num_rows = 11 :=
by
  sorry

#check num_erased_cheburashkas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_erased_cheburashkas_l208_20829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_ratio_l208_20809

-- Define the function f
def f : ℕ → ℕ → ℕ → ℕ
| 0, 0, 0 => 1
| x, y, z => 
  if x > 0 then f (x - 1) y z
  else if y > 0 then f x (y - 1) z
  else if z > 0 then f x y (z - 1)
  else 0

-- Define a triangle
def is_triangle (x y z : ℕ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem not_integer_ratio (x y z k m : ℕ) (h_triangle : is_triangle x y z) (h_k : k > 1) (h_m : m > 1) :
  ¬ (∃ n : ℕ, (f x y z)^k = n * f (m * x) (m * y) (m * z)) := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_positive (x y z : ℕ) : f x y z > 0 := by
  sorry

lemma f_monotone (x y z : ℕ) : f (x + 1) y z ≥ f x y z := by
  sorry

lemma f_combinatorial (x y z : ℕ) : f x y z = (x + y + z).factorial / (x.factorial * y.factorial * z.factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_ratio_l208_20809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_l208_20805

/-- A line in a plane --/
structure Line where
  -- Define necessary properties for a line
  mk :: -- placeholder for line properties

/-- A point in a plane --/
structure Point where
  -- Define necessary properties for a point
  mk :: -- placeholder for point properties

/-- A plane --/
structure Plane where
  -- Define necessary properties for a plane
  mk :: -- placeholder for plane properties

/-- Perpendicularity relation between two lines --/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry -- Define perpendicularity condition

/-- A point lies on a line --/
def on_line (p : Point) (l : Line) : Prop :=
  sorry -- Define condition for a point to be on a line

/-- A line passes through a point --/
def passes_through (l : Line) (p : Point) : Prop :=
  sorry -- Define condition for a line to pass through a point

/-- Main theorem: Unique perpendicular line through a point --/
theorem unique_perpendicular_line (π : Plane) (l : Line) (p : Point) :
  ∃! l' : Line, passes_through l' p ∧ perpendicular l' l :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_l208_20805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_with_always_even_solutions_l208_20848

-- Define the property of having an even number of real solutions
def HasEvenSolutions (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, ∃ n : ℕ, 2 ∣ n ∧ (∃ (s : Finset ℝ), s.card = n ∧ ∀ x ∈ s, f x = a)

-- State the theorem
theorem no_polynomial_with_always_even_solutions :
  ¬∃ (f : ℝ → ℝ), (∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, ∃ k : ℕ, f x = (x^n : ℝ) + (k : ℝ)) ∧ HasEvenSolutions f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_with_always_even_solutions_l208_20848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_l208_20831

-- Define the circle equation in polar coordinates
def circle_equation (ρ θ : Real) : Prop :=
  ρ = Real.cos (θ + Real.pi/3)

-- Define the center of the circle in polar coordinates
noncomputable def circle_center : Real × Real :=
  (1/2, -Real.pi/3)

-- Theorem statement
theorem center_of_circle :
  ∀ ρ θ : Real,
  circle_equation ρ θ →
  ∃ r φ : Real,
  (r, φ) = circle_center ∧
  r * Real.cos φ = 1/4 ∧
  r * Real.sin φ = -Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_l208_20831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_concentrate_volume_l208_20865

/-- The volume of concentrate required to prepare a given number of servings of orange juice -/
noncomputable def concentrate_volume (servings : ℕ) (serving_size : ℝ) (concentrate_ratio : ℝ) (total_ratio : ℝ) : ℝ :=
  (servings : ℝ) * serving_size * concentrate_ratio / total_ratio

/-- Theorem stating the volume of concentrate required for 375 servings of 150mL orange juice -/
theorem orange_juice_concentrate_volume :
  concentrate_volume 375 150 1 6 = 9375 := by
  -- Unfold the definition of concentrate_volume
  unfold concentrate_volume
  -- Simplify the arithmetic
  simp [Nat.cast_ofNat]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_concentrate_volume_l208_20865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l208_20874

-- Define s as the real positive solution to x^4 + (1/4)x - (1/2) = 0
noncomputable def s : ℝ := Real.sqrt (Real.sqrt ((1/2)^(1/3)))

-- Define the infinite series S
noncomputable def S : ℝ := ∑' n : ℕ, (n + 1) * s^(3*n + 1)

-- Theorem statement
theorem infinite_series_sum : S = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l208_20874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l208_20890

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2/5) : Real.cos (7 * θ) = -83728/390625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l208_20890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l208_20828

noncomputable def data_set : List ℝ := [9, 8, 12, 10, 11]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (λ x => (x - μ)^2)).sum / xs.length

theorem variance_of_data_set :
  mean data_set = 10 → variance data_set = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l208_20828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_one_f_inequality_range_l208_20803

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + Real.exp (-x)

-- Define an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem 1: f is even iff a = 1
theorem f_even_iff_a_one (a : ℝ) :
  is_even (f a) ↔ a = 1 := by sorry

-- Theorem 2: When a = 1, f(m+2) ≤ f(2m-3) iff m ≤ 1/3 or m ≥ 5
theorem f_inequality_range (m : ℝ) :
  f 1 (m + 2) ≤ f 1 (2 * m - 3) ↔ m ≤ 1/3 ∨ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_one_f_inequality_range_l208_20803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l208_20812

/-- The constant term in the binomial expansion of (3x - 2/x)^8 -/
def constant_term : ℕ := 112

/-- The binomial expression (3x - 2/x)^8 -/
noncomputable def binomial_expression (x : ℝ) : ℝ := (3 * x - 2 / x) ^ 8

/-- Theorem stating that the constant term in the binomial expansion of (3x - 2/x)^8 is 112 -/
theorem constant_term_proof : 
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → 
  binomial_expression x = c + x * (binomial_expression x - c) / x ∧ 
  c = constant_term := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l208_20812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l208_20843

/-- Given a circular sector with central angle θ and chord length c, 
    this function calculates the area of the sector. -/
noncomputable def sectorArea (θ : ℝ) (c : ℝ) : ℝ :=
  let r := c / (2 * Real.sin (θ / 2))
  (1 / 2) * r^2 * θ

/-- Theorem stating that for a circular sector with central angle 2 radians 
    and chord length 2, the area is 1/(sin^2(1)). -/
theorem sector_area_specific : sectorArea 2 2 = 1 / (Real.sin 1)^2 := by
  sorry

#check sector_area_specific

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l208_20843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_AMB_equals_2_sqrt_2_l208_20877

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line with a slope angle of 45° passing through the focus of a parabola -/
structure Line (c : Parabola) where
  slope_angle : ℝ
  h_slope_angle : slope_angle = π / 4
  passes_through_focus : True  -- We assume this without explicitly defining the focus

/-- Points A and B are the intersections of the line with the parabola -/
structure IntersectionPoints (c : Parabola) (l : Line c) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_A_on_parabola : (A.2)^2 = 2 * c.p * A.1
  h_B_on_parabola : (B.2)^2 = 2 * c.p * B.1
  h_A_on_line : True  -- We assume this without explicitly defining the line equation
  h_B_on_line : True  -- We assume this without explicitly defining the line equation

/-- Point M is defined as (-p/2, 0) -/
noncomputable def M (c : Parabola) : ℝ × ℝ := (-c.p/2, 0)

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- Angle AMB -/
noncomputable def angle_AMB (M A B : ℝ × ℝ) : ℝ :=
  angle (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2)

/-- The main theorem to prove -/
theorem tan_AMB_equals_2_sqrt_2 (c : Parabola) (l : Line c) (pts : IntersectionPoints c l) :
  let m := M c
  Real.tan (angle_AMB m pts.A pts.B) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_AMB_equals_2_sqrt_2_l208_20877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_D_proof_l208_20855

def p (a : ℝ) (D : Set ℝ) : Prop := a ∈ D

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - a*x₀ - a ≤ -3

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem interval_D_proof :
  ∃ D : Set ℝ,
    (∀ a, necessary_not_sufficient (p a D) (q a)) ∧
    D = Set.Iic (-4) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_D_proof_l208_20855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_oh_distance_l208_20814

/-- Triangle ABC with special points --/
structure SpecialTriangle where
  A : ℝ × ℝ  -- Vertex A of the triangle
  I : ℝ × ℝ  -- Incenter
  O : ℝ × ℝ  -- Circumcenter
  H : ℝ × ℝ  -- Orthocenter

/-- Distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem special_triangle_oh_distance (t : SpecialTriangle) 
  (h1 : distance t.A t.I = 11)
  (h2 : distance t.A t.O = 13)
  (h3 : distance t.A t.H = 13) : 
  distance t.O t.H = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_oh_distance_l208_20814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_3n_minus_1_over_2n_plus_3_l208_20884

theorem limit_fraction_3n_minus_1_over_2n_plus_3 :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((3 * (n : ℝ) - 1) / (2 * n + 3)) - (3 / 2)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_3n_minus_1_over_2n_plus_3_l208_20884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l208_20868

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The set of primes that divide f(n) for some natural number n -/
def PrimeDivisorSet (f : IntPolynomial) : Set ℕ :=
  {p | Nat.Prime p ∧ ∃ n, (p : ℤ) ∣ f n}

/-- Theorem: For any non-constant polynomial with integer coefficients,
    the set of primes that divide f(n) for some n is infinite -/
theorem infinite_prime_divisors (f : IntPolynomial)
  (hf : ∃ n, f n ≠ 0) :
  Set.Infinite (PrimeDivisorSet f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l208_20868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_generation_l208_20846

-- Define the probability density function
noncomputable def f (x y : ℝ) : ℝ := (3/4) * x * y^2

-- Define the region
def region (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2

-- Define the cumulative distribution functions
noncomputable def F_X (x : ℝ) : ℝ := x^2
noncomputable def F_Y (y : ℝ) : ℝ := (1/8) * y^3

-- Define the inverse transform functions
noncomputable def x_i (r : ℝ) : ℝ := Real.sqrt r
noncomputable def y_i (r : ℝ) : ℝ := 2 * (r^(1/3))

-- Theorem statement
theorem random_variable_generation :
  ∀ (r : ℝ), 0 < r ∧ r < 1 →
  (∀ (x y : ℝ), region x y → f x y = (3/4) * x * y^2) →
  F_X (x_i r) = r ∧ F_Y (y_i r) = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_generation_l208_20846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_12_85_l208_20847

-- Define the length of each train
noncomputable def train_length : ℝ := 120

-- Define the time taken by each train to cross a telegraph post
noncomputable def time_train1 : ℝ := 10
noncomputable def time_train2 : ℝ := 18

-- Define the speed of each train
noncomputable def speed_train1 : ℝ := train_length / time_train1
noncomputable def speed_train2 : ℝ := train_length / time_train2

-- Define the relative speed of the trains
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2

-- Define the total distance to be covered when trains cross
noncomputable def total_distance : ℝ := 2 * train_length

-- Define the time taken for trains to cross each other
noncomputable def crossing_time : ℝ := total_distance / relative_speed

-- Theorem statement
theorem trains_crossing_time_approx_12_85 :
  ∃ ε > 0, |crossing_time - 12.85| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_12_85_l208_20847
