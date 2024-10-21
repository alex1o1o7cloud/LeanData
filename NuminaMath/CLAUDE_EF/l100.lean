import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_real_l100_10065

/-- The function f(x) defined in terms of m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log ((m^2 - 3*m + 2)*x^2 + 2*(m - 1)*x + 5)

/-- The theorem stating the condition for f to have a range of all real numbers -/
theorem f_range_is_real (m : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f m x = y) ↔ (m = 1 ∨ (2 < m ∧ m ≤ 9/4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_real_l100_10065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_Y_l100_10077

/-- Random variable X taking values in positive integers -/
def X : Type := ℕ+

/-- Random variable Y taking values in {0, 1, 2} -/
def Y : Type := Fin 3

/-- Probability mass function for X -/
noncomputable def P (k : X) : ℝ := 1 / (2 * k.val)

/-- Y is congruent to 3 modulo X -/
def Y_mod_X (x : X) (y : Y) : Prop := y.val ≡ 3 [MOD x.val]

/-- Expected value of Y -/
noncomputable def E_Y : ℝ := sorry

/-- Theorem stating that E(Y) = 8/7 -/
theorem expected_value_Y : E_Y = 8/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_Y_l100_10077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l100_10066

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x - Real.pi/3)

-- Define the domain of x
def domain : Set ℝ := {x : ℝ | x ≥ 0}

-- Theorem stating the properties of the function
theorem function_properties :
  let amplitude : ℝ := 1/2
  let period : ℝ := Real.pi
  let initial_phase : ℝ := -Real.pi/3
  (∀ x ∈ domain, f x = amplitude * Real.sin (2*Real.pi/period * x + initial_phase)) ∧
  (∀ x ∈ domain, f (x + period) = f x) ∧
  (f 0 = amplitude * Real.sin initial_phase) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l100_10066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l100_10000

-- Define a function that represents a horizontal shift
def horizontalShift (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x - h)

-- Define a function that represents a vertical shift
def verticalShift (f : ℝ → ℝ) (v : ℝ) : ℝ → ℝ := fun x ↦ f x + v

-- Define the exponential function base 2
noncomputable def exp2 : ℝ → ℝ := fun x ↦ 2^x

-- State the theorem
theorem function_transformation (f : ℝ → ℝ) :
  (horizontalShift (verticalShift f (-2)) 2 = exp2) →
  (f = fun x ↦ 2^(x-2) + 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l100_10000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wikipedia_error_free_probability_l100_10095

/-- Represents the probability of catching an error on a single day -/
def catch_probability : ℚ := 2/3

/-- Represents the number of days the process continues -/
def days : ℕ := 3

/-- Calculates the probability of an error being caught over a given number of days -/
def error_caught_probability (n : ℕ) : ℚ :=
  1 - (1 - catch_probability) ^ n

/-- Calculates the probability of the article being error-free after a given number of days -/
def error_free_probability (n : ℕ) : ℚ :=
  Finset.prod (Finset.range n) (λ i => error_caught_probability (n - i))

/-- The main theorem stating that the probability of the article being error-free after 3 days is 416/729 -/
theorem wikipedia_error_free_probability :
  error_free_probability days = 416/729 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wikipedia_error_free_probability_l100_10095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_positive_l100_10028

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + 2*m - 5)

-- State the theorem
theorem f_sum_positive (m : ℝ) (a b : ℝ) 
  (h1 : ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) > 0)
  (h2 : a + b > 0) :
  f m a + f m b > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_positive_l100_10028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_for_natural_numbers_l100_10054

theorem inequality_for_natural_numbers (n : ℕ) (hn : n > 0) :
  (n - 1 : ℝ)^(n + 1) * (n + 1 : ℝ)^(n - 1) < n^(2 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_for_natural_numbers_l100_10054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_minimum_l100_10073

theorem triangle_cosine_minimum (A B C : Real) (h : Real.sin A + Real.sqrt 2 * Real.sin B = 2 * Real.sin C) :
  Real.cos C ≥ (Real.sqrt 6 - Real.sqrt 2) / 4 ∧
  ∃ (A' B' C' : Real), Real.sin A' + Real.sqrt 2 * Real.sin B' = 2 * Real.sin C' ∧
                       Real.cos C' = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_minimum_l100_10073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_arrangements_l100_10094

def word : String := "MATHEMATICS"

theorem mathematics_arrangements :
  (Nat.factorial 11) / ((Nat.factorial 2) * (Nat.factorial 2) * (Nat.factorial 2)) = 4989600 := by
  norm_num
  ring
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_arrangements_l100_10094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_a_general_term_l100_10031

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => x^2

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
  | 0 => 1  -- a_1 = 1 (we use 0-based indexing)
  | n + 1 => a n - 1/2

-- State the theorem
theorem a_8_value :
  (∀ x : ℝ, f x = 2 * f (2 - x) - x^2 + 8*x - 8) →
  (∀ n : ℕ, (a n, 2 * a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 - 1}) →
  a 0 ≠ 1 →
  a 7 = -5/2 := by
  sorry

-- Additional lemma to show the general term of the sequence
theorem a_general_term (n : ℕ) :
  a n = -1/2 * (n : ℝ) + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_a_general_term_l100_10031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l100_10045

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (2 - t, 2 * t)

-- Define the curve C₁
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the curve C₂ in polar coordinates
noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the line θ = α
def line_alpha (α : ℝ) (ρ : ℝ) : Prop := ρ > 0 ∧ Real.pi/4 < α ∧ α < Real.pi/2

-- Define the midpoint of AB
noncomputable def midpoint_AB (ρA ρB : ℝ) (α : ℝ) : ℝ × ℝ := 
  ((ρA + ρB)/2 * Real.cos α, (ρA + ρB)/2 * Real.sin α)

-- Theorem statement
theorem length_of_AB (α : ℝ) (ρA ρB : ℝ) :
  line_alpha α ρA ∧ line_alpha α ρB ∧
  ρA = 4 * Real.cos α ∧
  ρB = 4 * Real.sin α ∧
  (∃ t : ℝ, midpoint_AB ρA ρB α = line_l t) →
  |ρA - ρB| = 4 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l100_10045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_existence_l100_10091

theorem square_division_existence :
  ∃ (c a b n : ℕ), 
    c > 0 ∧ a > 0 ∧ b > 0 ∧ n > 0 ∧
    a ≠ b ∧ 
    c^2 = n * (a^2 + b^2) ∧
    2 * n > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_existence_l100_10091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_condition_l100_10082

theorem unique_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n.val * 2^(n.val + 1) + 1 = m^2) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_condition_l100_10082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_70_eq_one_minus_two_k_squared_l100_10090

-- Define k as the sine of 10 degrees
noncomputable def k : ℝ := Real.sin (10 * Real.pi / 180)

-- State the theorem
theorem sin_70_eq_one_minus_two_k_squared :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_70_eq_one_minus_two_k_squared_l100_10090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nandan_earnings_l100_10048

/-- Represents the investment and earnings of a person in the business --/
structure Investor where
  name : String
  investment_multiplier : ℝ
  time_multiplier : ℝ
  market_impact : ℝ

/-- Calculates the gain for an investor --/
def gain (i : Investor) (base_investment : ℝ) (base_time : ℝ) : ℝ :=
  i.investment_multiplier * base_investment * i.time_multiplier * base_time * i.market_impact

/-- Theorem stating Nandan's earnings given the business conditions --/
theorem nandan_earnings 
  (nandan : Investor)
  (krishan : Investor)
  (arjun : Investor)
  (rohit : Investor)
  (base_investment : ℝ)
  (base_time : ℝ)
  (total_gain : ℝ)
  (h1 : nandan.name = "Nandan" ∧ nandan.investment_multiplier = 1 ∧ nandan.time_multiplier = 1 ∧ nandan.market_impact = 0.15)
  (h2 : krishan.name = "Krishan" ∧ krishan.investment_multiplier = 4 ∧ krishan.time_multiplier = 3 ∧ krishan.market_impact = 0.10)
  (h3 : arjun.name = "Arjun" ∧ arjun.investment_multiplier = 2 ∧ arjun.time_multiplier = 2 ∧ arjun.market_impact = 0.12)
  (h4 : rohit.name = "Rohit" ∧ rohit.investment_multiplier = 5 ∧ rohit.time_multiplier = 1 ∧ rohit.market_impact = 0)
  (h5 : total_gain = 26000)
  (h6 : total_gain = gain krishan base_investment base_time + gain nandan base_investment base_time + 
                     gain arjun base_investment base_time + gain rohit base_investment base_time) :
  ∃ ε > 0, |gain nandan base_investment base_time - 2131.15| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nandan_earnings_l100_10048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_l100_10041

theorem quadruple_count (n : ℕ) : 
  Finset.card (Finset.filter (fun q : ℕ × ℕ × ℕ × ℕ => 
    let (i, j, k, h) := q
    1 ≤ i ∧ i < j ∧ j ≤ k ∧ k < h ∧ h ≤ n + 1) (Finset.product (Finset.range (n+1)) (Finset.product (Finset.range (n+1)) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))))) = Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_l100_10041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l100_10069

/-- Represents the game state -/
structure GameState where
  cells : Fin 13 → ℕ

/-- Represents a move in the game -/
inductive Move where
  | one  : Move  -- Move 1 cell to the right
  | two  : Move  -- Move 2 cells to the right

/-- Defines a valid move in the game -/
def is_valid_move (gs : GameState) (m : Move) : Prop :=
  match m with
  | Move.one => ∃ i : Fin 12, gs.cells i > 0
  | Move.two => ∃ i : Fin 11, gs.cells i > 0

/-- Defines the initial game state -/
def initial_state : GameState :=
  { cells := λ i => if i = 0 then 2023 else 0 }

/-- Defines the winning condition -/
def is_winning_state (gs : GameState) : Prop :=
  gs.cells 12 > 0

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Applies a move to the game state -/
def apply_move (gs : GameState) (m : Move) : GameState :=
  sorry -- Implementation details omitted for brevity

/-- Plays the game to completion given two strategies -/
def play_game (initial : GameState) (s1 s2 : Strategy) : GameState :=
  sorry -- Implementation details omitted for brevity

/-- Theorem: The second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Strategy), 
    ∀ (first_player_strategy : Strategy),
      is_winning_state (play_game initial_state first_player_strategy strategy) := by
  sorry -- Proof details omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l100_10069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_points_theorem_l100_10047

noncomputable section

/-- The curve y^2 = x --/
def Curve (x y : ℝ) : Prop := y^2 = x

/-- The slope of the tangent line at a point (x, y) on the curve y^2 = x --/
noncomputable def TangentSlope (x y : ℝ) : ℝ := 1 / (2 * y)

/-- The slope of the line passing through two points --/
noncomputable def LineSlope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

/-- Two lines are perpendicular if the product of their slopes is -1 --/
def Perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The angle between two lines with slopes m1 and m2 --/
noncomputable def AngleBetweenLines (m1 m2 : ℝ) : ℝ := Real.arctan ((m2 - m1) / (1 + m1 * m2))

theorem curve_points_theorem (a b : ℝ) :
  Curve (a^2) a ∧ Curve (b^2) b ∧
  Perpendicular (TangentSlope (a^2) a) (LineSlope (a^2) a (b^2) b) ∧
  AngleBetweenLines (LineSlope (a^2) a (b^2) b) (TangentSlope (b^2) b) = π/4 →
  ((a^2 = 1 ∧ a = -1 ∧ b^2 = 9/4 ∧ b = 3/2) ∨ (a^2 = 1 ∧ a = 1 ∧ b^2 = 9/4 ∧ b = -3/2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_points_theorem_l100_10047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AM_l100_10071

noncomputable section

open Real

theorem length_AM (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →
  -- AB = 60
  sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 60 →
  -- AC = 160
  sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 160 →
  -- M is the midpoint of BC
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  -- The length of AM is approximately 56.3
  abs (sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) - 56.3) < 0.1 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AM_l100_10071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l100_10007

theorem trigonometric_problem (α β : ℝ) 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α - β) = 13 / 14)
  (h3 : 0 < β)
  (h4 : β < α)
  (h5 : α < π / 2) :
  Real.tan (2 * α) = - (8 * Real.sqrt 3) / 47 ∧ β = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l100_10007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_line_slope_l100_10059

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x + 2 * y = 1

-- Define the slope-intercept form of a line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Theorem: The line 4x + 2y = 1 can be written in slope-intercept form with slope -2
theorem line_slope_intercept :
  ∀ x y : ℝ, line_equation x y → slope_intercept_form (-2) (1/2) x y :=
by
  intros x y h
  unfold line_equation at h
  unfold slope_intercept_form
  -- Algebraic manipulation to isolate y
  have h1 : 2 * y = -4 * x + 1 := by linarith
  have h2 : y = -2 * x + 1/2 := by
    field_simp at h1
    linarith
  exact h2

-- Theorem: The slope of the line 4x + 2y = 1 is -2
theorem line_slope :
  ∃ m : ℝ, ∀ x y : ℝ, line_equation x y → slope_intercept_form m (1/2) x y ∧ m = -2 :=
by
  use -2
  intros x y h
  constructor
  · exact line_slope_intercept x y h
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_line_slope_l100_10059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_2012_l100_10021

/-- Given a function f: ℕ → ℕ satisfying f(f(n)) + f(n) = 2n + 3 for all n,
    and f(0) = 1, prove that f(2012) = 2013 -/
theorem function_value_at_2012 (f : ℕ → ℕ) 
    (h1 : ∀ n, f (f n) + f n = 2*n + 3)
    (h2 : f 0 = 1) : 
  f 2012 = 2013 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_2012_l100_10021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_theorem_l100_10030

/-- Represents the outcome of a single shot -/
inductive Shot
| Hit
| Miss

/-- Represents the state of the game -/
inductive GameState
| InProgress
| Cleared
| Failed

/-- Represents the possible coupon amounts -/
inductive CouponAmount
| Three
| Six
| Nine

/-- The shooting accuracy of Xiao Ming -/
def shooting_accuracy : ℚ := 2/3

/-- The probability of ending the game after five shots -/
def prob_end_after_five : ℚ := 8/81

/-- The expected coupon amount -/
def expected_coupon : ℚ := 1609/243

/-- Updates the game state based on the current state and the new shot -/
def update_state (state : GameState) (shot : Shot) : GameState :=
  sorry

/-- Calculates the probability of a specific sequence of shots -/
def prob_sequence (shots : List Shot) : ℚ :=
  sorry

/-- Determines the coupon amount based on the final game state -/
def determine_coupon (state : GameState) (shots : Nat) : CouponAmount :=
  sorry

/-- Calculates the probability distribution of coupon amounts -/
def coupon_distribution : List (CouponAmount × ℚ) :=
  sorry

/-- The main theorem to prove -/
theorem basketball_game_theorem :
  (∃ (sequences : List (List Shot)), 
    (sequences.all (λ seq => seq.length = 5)) ∧
    (sequences.map prob_sequence).sum = prob_end_after_five) ∧
  (let dist := coupon_distribution
   (dist.map (λ (amount, prob) => 
     match amount with
     | CouponAmount.Three => 3 * prob
     | CouponAmount.Six => 6 * prob
     | CouponAmount.Nine => 9 * prob
   )).sum = expected_coupon) := by
  sorry

#eval shooting_accuracy
#eval prob_end_after_five
#eval expected_coupon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_theorem_l100_10030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonnegative_l100_10093

-- Define the function f(x) = log₂(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the interval [1/2, 2]
def interval : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 2 }

-- Define the subset where f(x) ≥ 0
def subset : Set ℝ := { x ∈ interval | f x ≥ 0 }

-- State the theorem
theorem probability_f_nonnegative :
  (MeasureTheory.volume subset) / (MeasureTheory.volume interval) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonnegative_l100_10093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l100_10046

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  (∀ m : ℝ, x + y/4 < m^2 - 3*m → m ∈ Set.Ioi 4 ∪ Set.Iic (-1)) ∧
  (∀ m : ℝ, m ∈ Set.Ioi 4 ∪ Set.Iic (-1) → x + y/4 < m^2 - 3*m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l100_10046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_speed_calculation_l100_10092

/-- Calculates the wind speed given the plane's speed and distances traveled with and against the wind -/
noncomputable def wind_speed (plane_speed : ℝ) (distance_with_wind : ℝ) (distance_against_wind : ℝ) : ℝ :=
  (distance_with_wind - distance_against_wind) * plane_speed / (distance_with_wind + distance_against_wind)

theorem wind_speed_calculation (plane_speed : ℝ) (distance_with_wind : ℝ) (distance_against_wind : ℝ)
    (h1 : plane_speed = 253)
    (h2 : distance_with_wind = 420)
    (h3 : distance_against_wind = 350) :
    wind_speed plane_speed distance_with_wind distance_against_wind = 23 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval wind_speed 253 420 350

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_speed_calculation_l100_10092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l100_10017

-- Define the rates of the pipes
noncomputable def rate_a : ℝ := 1 / 34
noncomputable def rate_b : ℝ := 2 * rate_a
noncomputable def rate_c : ℝ := 2 * rate_b
noncomputable def rate_d : ℝ := 1.5 * rate_a

-- Define the time it takes for all pipes to fill the tank
def total_time : ℝ := 4

-- Theorem statement
theorem pipe_a_fill_time :
  (rate_a + rate_b + rate_c + rate_d) * total_time = 1 ∧
  (1 / rate_a : ℝ) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l100_10017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l100_10006

/-- Regular octagon with apothem 3 -/
structure RegularOctagon :=
  (apothem : ℝ)
  (is_regular : apothem = 3)

/-- Midpoints of alternating sides of the octagon -/
def midpoints (octagon : RegularOctagon) : Fin 4 → ℝ × ℝ := sorry

/-- The quadrilateral formed by the midpoints -/
def midpoint_quadrilateral (octagon : RegularOctagon) : Set (ℝ × ℝ) :=
  {p | ∃ i : Fin 4, p = midpoints octagon i}

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating the area of the midpoint quadrilateral -/
theorem midpoint_quadrilateral_area (octagon : RegularOctagon) :
  area (midpoint_quadrilateral octagon) = 72 * (3 - 2 * Real.sqrt 2) := by
  sorry

#check midpoint_quadrilateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l100_10006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_symmetric_about_x_axis_l100_10070

-- Define the curve C using its parametric equations
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)

-- Define symmetry about x-axis
def symmetric_about_x_axis (C : ℝ → ℝ × ℝ) : Prop :=
  ∀ θ : ℝ, ∃ φ : ℝ, C θ = (Prod.fst (C φ), -Prod.snd (C φ))

-- Theorem statement
theorem curve_C_symmetric_about_x_axis :
  symmetric_about_x_axis curve_C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_symmetric_about_x_axis_l100_10070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_28000_l100_10042

/-- Represents a position title with its count and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company -/
def positions : List Position := [
  ⟨"CEO", 1, 150000⟩,
  ⟨"Senior Vice-President", 4, 110000⟩,
  ⟨"Manager", 15, 80000⟩,
  ⟨"Assistant Manager", 10, 55000⟩,
  ⟨"Clerk", 45, 28000⟩
]

/-- The total number of employees -/
def totalEmployees : Nat := (positions.map Position.count).sum

/-- The index of the median salary -/
def medianIndex : Nat := (totalEmployees + 1) / 2

/-- Theorem stating that the median salary is $28,000 -/
theorem median_salary_is_28000 :
  ∃ (p : Position), p ∈ positions ∧ p.salary = 28000 ∧
  ((positions.filter (fun q ↦ q.salary ≤ p.salary)).map Position.count).sum ≥ medianIndex :=
by
  -- We'll use the Clerk position as our witness
  let clerk : Position := ⟨"Clerk", 45, 28000⟩
  use clerk
  apply And.intro
  · simp [positions]  -- Show that clerk ∈ positions
  apply And.intro
  · rfl  -- Prove that clerk.salary = 28000
  · sorry  -- Prove that the sum of counts for positions with salary ≤ 28000 is ≥ medianIndex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_28000_l100_10042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l100_10034

noncomputable def f (a x : ℝ) : ℝ := x + a^2 / x

noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem min_a_value (a : ℝ) (h₁ : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 (Real.exp 1) → x₂ ∈ Set.Icc 1 (Real.exp 1) → f a x₁ ≥ g x₂) ↔
  a ≥ Real.sqrt (Real.exp 1 - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l100_10034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l100_10049

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) / (x - 1)

-- Define the domain of the function
def domain : Set ℝ := {x | x ≤ 2 ∧ x ≠ 1}

-- Theorem statement
theorem function_domain : 
  {x : ℝ | ∃ y : ℝ, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l100_10049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_car_reaches_destination_l100_10013

/-- Represents the direction of the car -/
inductive Direction
| North
| East
| South
| West

/-- Represents a turn -/
inductive Turn
| Left
| Right
| None

/-- Represents the state of the car -/
structure CarState where
  x : ℤ
  y : ℤ
  direction : Direction

/-- Perform a single move based on the current distance traveled -/
def single_move (ℓ r : ℕ) (distance : ℕ) (state : CarState) : CarState :=
sorry

/-- Iterate moves for a given number of steps -/
def iterate_moves (ℓ r : ℕ) (n m : ℕ) (initialState : CarState) : CarState :=
sorry

/-- The robot car problem -/
theorem robot_car_reaches_destination (ℓ r : ℕ) 
  (hℓ : ℓ > 0) 
  (hr : r > 0) 
  (hcoprime : Nat.Coprime ℓ r) :
  (∃ d : ℕ, ∀ n : ℕ, ∃ m : ℕ, 
    let finalState := iterate_moves ℓ r n m (CarState.mk 0 0 Direction.East)
    finalState.x = d ∧ finalState.y = 0) ↔ 
  (ℓ % 4 = 1 ∧ r % 4 = 1) ∨ (ℓ % 4 = 3 ∧ r % 4 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_car_reaches_destination_l100_10013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_f_greater_g_iff_p_less_neg_eight_l100_10083

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 6 * Real.log x + x^2 - 8*x
noncomputable def g (p : ℝ) (x : ℝ) : ℝ := p/x + x^2

-- State the theorem
theorem exists_point_f_greater_g_iff_p_less_neg_eight :
  ∀ p : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f x₀ > g p x₀) ↔ p < -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_f_greater_g_iff_p_less_neg_eight_l100_10083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_sides_area_of_triangle_l100_10086

noncomputable section

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  (Real.sin (2 * t.A + t.B)) / (Real.sin t.A) = 2 + 2 * Real.cos (t.A + t.B)

-- Theorem 1
theorem ratio_of_sides (t : Triangle) (h : triangle_condition t) : t.b / t.a = 2 := by
  sorry

-- Theorem 2
theorem area_of_triangle (t : Triangle) (h1 : t.b / t.a = 2) (h2 : t.a = 1) (h3 : t.c = Real.sqrt 7) :
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_sides_area_of_triangle_l100_10086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_all_activities_lower_bound_l100_10050

/-- Represents a class with students who can perform different activities -/
structure MyClass where
  total : ℕ
  swim : ℕ
  bike : ℕ
  skate : ℕ
  tableTennis : ℕ

/-- The minimum number of students who can perform all activities -/
def minAllActivities (c : MyClass) : ℕ :=
  max 0 (c.swim + c.bike - c.total) + max 0 (c.skate + c.tableTennis - c.total) - c.total

/-- Theorem stating the minimum number of students who can do all activities -/
theorem min_all_activities_lower_bound (c : MyClass) 
  (h_total : c.total = 60)
  (h_swim : c.swim = 42)
  (h_bike : c.bike = 46)
  (h_skate : c.skate = 50)
  (h_tableTennis : c.tableTennis = 55) :
  minAllActivities c ≥ 13 := by
  sorry

#eval minAllActivities { total := 60, swim := 42, bike := 46, skate := 50, tableTennis := 55 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_all_activities_lower_bound_l100_10050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_GHI_JKL_l100_10018

-- Define the triangles
def triangle_GHI : (ℝ × ℝ × ℝ) := (10, 24, 26)
def triangle_JKL : (ℝ × ℝ × ℝ) := (16, 34, 40)

-- Define a function to calculate the area of a triangle using Heron's formula
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem area_ratio_GHI_JKL :
  (area triangle_GHI.1 triangle_GHI.2.1 triangle_GHI.2.2) /
  (area triangle_JKL.1 triangle_JKL.2.1 triangle_JKL.2.2) =
  8 * Real.sqrt 319 / 319 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_GHI_JKL_l100_10018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l100_10022

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 6 = 0

-- Define the distance function from a point (x, y) to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 6| / Real.sqrt 2

-- State the theorem
theorem min_distance_curve_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), curve_C x y →
    d ≤ distance_to_line x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l100_10022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_to_negative_third_equals_two_l100_10026

theorem eighth_to_negative_third_equals_two :
  (1 / 8 : ℝ) ^ (-(1/3) : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_to_negative_third_equals_two_l100_10026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_fluctuation_l100_10024

noncomputable def price_change (initial_price : ℝ) (percent_change : ℝ) : ℝ :=
  initial_price * (1 + percent_change / 100)

theorem price_fluctuation (P : ℝ) (y : ℝ) (h : P > 0) :
  let P1 := price_change P 30
  let P2 := price_change P1 (-30)
  let P3 := price_change P2 40
  let P4 := price_change P3 (-y)
  P4 = P →
  ⌊y + 0.5⌋ = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_fluctuation_l100_10024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l100_10015

noncomputable def a (n : ℕ) : ℝ := 1 / (4 ^ (n - 1))

def T (n : ℕ) : ℝ := 2^(n * (1 - n))

noncomputable def S (n : ℕ) : ℝ := (4 / 3) * (1 - (1 / 4) ^ n)

theorem sequence_properties :
  (a 1 = 1) ∧
  (∀ n > 1, a n / a (n - 1) = 1 / 4) ∧
  (∀ n : ℕ, n > 0 → (S (n + 1) - 4 / 3)^2 = (S (n + 2) - 4 / 3) * (S n - 4 / 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l100_10015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l100_10035

/-- Represents a circular arrangement of 100 numbers -/
def CircularArrangement := Fin 100 → ℤ

/-- The sum of all numbers in the arrangement is 100 -/
def SumIs100 (arr : CircularArrangement) : Prop :=
  Finset.sum Finset.univ arr = 100

/-- The sum of any 6 consecutive numbers is 6 -/
def ConsecutiveSumIs6 (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, Finset.sum (Finset.range 6) (fun j => arr ((i + j) % 100)) = 6

/-- One of the numbers in the arrangement is 6 -/
def Contains6 (arr : CircularArrangement) : Prop :=
  ∃ i : Fin 100, arr i = 6

/-- All numbers with odd indices are 6 and all numbers with even indices are -4 -/
def AlternatingArrangement (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, if i.val % 2 = 0 then arr i = -4 else arr i = 6

theorem circular_arrangement_theorem (arr : CircularArrangement) :
  SumIs100 arr → ConsecutiveSumIs6 arr → Contains6 arr →
  AlternatingArrangement arr := by
  sorry

#check circular_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l100_10035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l100_10088

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + (x - 2) ^ 0

-- Define the domain
def domain : Set ℝ := {x | x ≥ -1 ∧ x ≠ 2}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ (∃ y : ℝ, f x = y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l100_10088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l100_10068

/-- The speed of a bus including stoppages, given its speed excluding stoppages and stoppage time per hour. -/
noncomputable def bus_speed_with_stoppages (speed_without_stoppages : ℝ) (stoppage_time_minutes : ℝ) : ℝ :=
  let stoppage_time_hours := stoppage_time_minutes / 60
  let moving_time_hours := 1 - stoppage_time_hours
  speed_without_stoppages * moving_time_hours

/-- Theorem stating that a bus with a speed of 54 kmph excluding stoppages and stopping for 14.444444444444443 minutes per hour has a speed of approximately 41 kmph including stoppages. -/
theorem bus_speed_theorem :
  let speed_without_stoppages := (54 : ℝ)
  let stoppage_time_minutes := (14.444444444444443 : ℝ)
  let speed_with_stoppages := bus_speed_with_stoppages speed_without_stoppages stoppage_time_minutes
  ∃ ε > 0, |speed_with_stoppages - 41| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l100_10068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptex_word_count_l100_10057

/-- The number of letters in the Cryptex alphabet -/
def alphabet_size : ℕ := 15

/-- The maximum word length in the Cryptex language -/
def max_word_length : ℕ := 5

/-- Calculates the number of words of a given length using all letters -/
def words_of_length (n : ℕ) : ℕ := alphabet_size ^ n

/-- Calculates the number of words of a given length not using the letter A -/
def words_without_a_of_length (n : ℕ) : ℕ := (alphabet_size - 1) ^ n

/-- The total number of words in the Cryptex language -/
def total_words : ℕ := Finset.sum (Finset.range (max_word_length + 1)) words_of_length

/-- The total number of words without the letter A -/
def total_words_without_a : ℕ := Finset.sum (Finset.range (max_word_length + 1)) words_without_a_of_length

/-- The number of valid words in the Cryptex language -/
def valid_words : ℕ := total_words - total_words_without_a

theorem cryptex_word_count : valid_words = 228421 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptex_word_count_l100_10057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_39_492_to_nearest_tenth_l100_10097

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The problem statement -/
theorem round_39_492_to_nearest_tenth :
  roundToNearestTenth 39.492 = 39.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_39_492_to_nearest_tenth_l100_10097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_l100_10089

-- Define the average cost-profit functions
noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x) / x + 5 / x - b
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x / x

-- Define the total profit function
noncomputable def total_profit (a b x : ℝ) : ℝ := x * f a b x + (50 - x) * g (50 - x)

-- State the theorem
theorem optimal_investment (a b : ℝ) :
  (f a b 1 = 5) →
  (10 * f a b 10 = 16.515) →
  (∀ x : ℝ, 10 ≤ x ∧ x ≤ 40 → 
    total_profit a b x ≤ total_profit a b 25) ∧
  (total_profit a b 25 = 31.09) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_l100_10089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_height_is_90_l100_10036

/-- The rebound factor of the ball -/
noncomputable def rebound_factor : ℝ := 1/2

/-- The total travel distance when the ball touches the floor for the third time -/
noncomputable def total_travel : ℝ := 225

/-- Calculates the total travel distance for a ball dropped from height h -/
noncomputable def calculate_total_travel (h : ℝ) : ℝ :=
  h + 2 * rebound_factor * h + 2 * rebound_factor^2 * h

/-- Theorem stating that the original height is 90 cm -/
theorem original_height_is_90 :
  ∃ (h : ℝ), h > 0 ∧ calculate_total_travel h = total_travel ∧ h = 90 :=
by
  -- We'll use 90 as our witness for h
  use 90
  
  -- Split the goal into three parts
  constructor
  · -- Prove h > 0
    norm_num
  
  constructor
  · -- Prove calculate_total_travel h = total_travel
    unfold calculate_total_travel total_travel rebound_factor
    norm_num
    
  · -- Prove h = 90
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_height_is_90_l100_10036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_greater_when_f_greater_l100_10055

noncomputable def f (x : ℝ) := 2023 * x^2 + 2024 * x * Real.sin x

theorem square_greater_when_f_greater
  (x₁ x₂ : ℝ)
  (h₁ : x₁ ∈ Set.Ioo (-Real.pi) Real.pi)
  (h₂ : x₂ ∈ Set.Ioo (-Real.pi) Real.pi)
  (h₃ : f x₁ > f x₂) :
  x₁^2 > x₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_greater_when_f_greater_l100_10055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_frequency_l100_10067

/-- The process of writing numbers on a segment as described in the problem -/
def SegmentProcess : Type := Unit

/-- The number of times the process is repeated -/
def iterations : Nat := 1000000

/-- Euler's totient function -/
noncomputable def φ : Nat → Nat := sorry

/-- The target number we're counting occurrences of -/
def target : Nat := 1978

/-- The frequency of a number's appearance in the process -/
noncomputable def frequency (n : Nat) (_ : SegmentProcess) : Nat :=
  φ n

theorem target_frequency :
  ∀ (process : SegmentProcess),
  frequency target process = 924 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_frequency_l100_10067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l100_10096

-- Define the painting rates and time
noncomputable def taimour_rate : ℝ → ℝ := λ t => 1 / t
noncomputable def jamshid_rate : ℝ → ℝ := λ t => 2 / t
def combined_time : ℝ := 3

-- State the theorem
theorem taimour_paint_time :
  (∀ t : ℝ, jamshid_rate t = 2 * taimour_rate t) →
  (taimour_rate 9 + jamshid_rate 9 = 1 / combined_time) →
  9 = combined_time * 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l100_10096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_is_25_hours_l100_10029

/-- Represents the pool filling scenario -/
structure PoolFilling where
  pool_volume : ℚ
  hose_rate_1 : ℚ
  hose_count_1 : ℕ
  hose_rate_2 : ℚ
  hose_count_2 : ℕ

/-- Calculates the time in hours to fill the pool -/
def fill_time (pf : PoolFilling) : ℚ :=
  pf.pool_volume / (pf.hose_rate_1 * pf.hose_count_1 + pf.hose_rate_2 * pf.hose_count_2) / 60

/-- Theorem stating that the pool filling time is 25 hours -/
theorem pool_fill_time_is_25_hours (pf : PoolFilling) 
    (h1 : pf.pool_volume = 15000)
    (h2 : pf.hose_rate_1 = 2)
    (h3 : pf.hose_count_1 = 2)
    (h4 : pf.hose_rate_2 = 3)
    (h5 : pf.hose_count_2 = 2) : 
  fill_time pf = 25 := by
  sorry

#eval fill_time { pool_volume := 15000, hose_rate_1 := 2, hose_count_1 := 2, hose_rate_2 := 3, hose_count_2 := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_is_25_hours_l100_10029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_increasing_domain_f_is_closed_ray_l100_10087

-- Define the function f(x) = √(x-1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- State the theorem
theorem f_monotonically_increasing :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 1 → x₂ ≥ 1 → x₂ > x₁ → f x₂ > f x₁ := by
  sorry

-- Define the domain of f
def domain_f : Set ℝ := { x : ℝ | x ≥ 1 }

-- State that the domain is [1, +∞)
theorem domain_f_is_closed_ray :
  domain_f = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_increasing_domain_f_is_closed_ray_l100_10087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_marks_l100_10003

theorem candidate_marks (max_marks : ℕ) (passing_percentage : ℚ) (failing_margin : ℕ) (secured_marks : ℕ) : 
  max_marks = 150 →
  passing_percentage = 40 / 100 →
  failing_margin = 20 →
  secured_marks = (passing_percentage * max_marks).floor - failing_margin →
  secured_marks = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_marks_l100_10003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_of_each_color_l100_10053

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 5

/-- The probability of selecting at least one marble of each color -/
theorem probability_at_least_one_of_each_color : 
  (Nat.choose total_marbles selected_marbles : ℚ)⁻¹ * 
  ((Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2 +
    Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 2 +
    Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1) : ℚ) = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_of_each_color_l100_10053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_a2_b2_is_simplest_l100_10032

noncomputable def sqrt_16a (a : ℝ) : ℝ := Real.sqrt (16 * a)
noncomputable def sqrt_a2_b2 (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)
noncomputable def sqrt_b_div_a (a b : ℝ) : ℝ := Real.sqrt (b / a)
noncomputable def sqrt_45 : ℝ := Real.sqrt 45

-- Define a predicate for simplest form
def is_simplest_form (x : ℝ) : Prop := ∀ y : ℝ, y ≠ x → ¬(∃ k : ℝ, k * y = x)

-- Theorem statement
theorem sqrt_a2_b2_is_simplest (a b : ℝ) : 
  is_simplest_form (sqrt_a2_b2 a b) ∧ 
  ¬(is_simplest_form (sqrt_16a a)) ∧ 
  ¬(is_simplest_form (sqrt_b_div_a a b)) ∧ 
  ¬(is_simplest_form sqrt_45) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_a2_b2_is_simplest_l100_10032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_in_terms_of_B_l100_10056

theorem find_A_in_terms_of_B (B : ℝ) (hB : B ≠ 0) :
  ∃ A : ℝ, ∃ f g : ℝ → ℝ, 
    (∀ x, f x = A * x - 3 * B^2) ∧ 
    (∀ x, g x = B * x) ∧ 
    f (g 2) = 0 → 
    A = (3 * B) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_in_terms_of_B_l100_10056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enlarged_circles_cover_triangle_l100_10063

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle structure
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Define a function to check if a point is inside a circle
def isPointInCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

-- Define a function to check if circles are externally tangent
def areCirclesExternallyTangent (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d = c1.radius + c2.radius

-- Define a function to check if a point is inside a triangle
def isPointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop :=
  sorry -- Implementation omitted for brevity

-- Main theorem
theorem enlarged_circles_cover_triangle 
  (c1 c2 c3 : Circle) 
  (t : Triangle) 
  (h1 : areCirclesExternallyTangent c1 c2)
  (h2 : areCirclesExternallyTangent c2 c3)
  (h3 : areCirclesExternallyTangent c3 c1)
  (h4 : t.a = ((c2.center.1 + c3.center.1) / 2, (c2.center.2 + c3.center.2) / 2))
  (h5 : t.b = ((c1.center.1 + c3.center.1) / 2, (c1.center.2 + c3.center.2) / 2))
  (h6 : t.c = ((c1.center.1 + c2.center.1) / 2, (c1.center.2 + c2.center.2) / 2))
  (k : ℝ)
  (hk : k > 2 / Real.sqrt 3)
  : ∀ p : ℝ × ℝ, isPointInTriangle p t → 
    (isPointInCircle p { center := c1.center, radius := k * c1.radius } ∨
     isPointInCircle p { center := c2.center, radius := k * c2.radius } ∨
     isPointInCircle p { center := c3.center, radius := k * c3.radius }) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enlarged_circles_cover_triangle_l100_10063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l100_10014

-- Define the number of people and chairs
def num_people : ℕ := 5
def num_chairs : ℕ := 7

-- Define the number of people who must sit together
def num_fixed_pair : ℕ := 2

-- Theorem statement
theorem seating_arrangements_count :
  (num_chairs - num_fixed_pair + 1) *     -- Positions for the pair
  2 *                                     -- Arrangements of the pair
  Nat.factorial (num_people - num_fixed_pair) =  -- Arrangements of remaining people
  720 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l100_10014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_semicircle_l100_10010

/-- The volume of a cone-shaped container made from a semi-circular sheet of thin iron -/
theorem cone_volume_from_semicircle (r : ℝ) (h : r = 1) : 
  (1 / 3) * π * (r / 2)^2 * (Real.sqrt 3 / 2) = (Real.sqrt 3 * π) / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_semicircle_l100_10010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l100_10038

-- Define the ⊗ operation
noncomputable def otimes (a b c : ℝ) : ℝ := a / (b - c)

-- State the theorem
theorem otimes_calculation :
  otimes (otimes 2 5 4) (otimes 7 10 4) (otimes 3 7 10) = 12 / 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l100_10038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_construction_l100_10023

-- Define the basic geometric objects
structure Plane where

structure Line where

structure Point where

-- Define the relationships between geometric objects
def Point.onPlane (p : Point) (π : Plane) : Prop := sorry

def Point.onLine (p : Point) (l : Line) : Prop := sorry

def Line.onPlane (l : Line) (π : Plane) : Prop := sorry

def Plane.intersect (π₁ π₂ : Plane) : Line := sorry

-- Define the trapezoid and its properties
structure Trapezoid (A B C D : Point) where

def Trapezoid.isIsosceles (t : Trapezoid A B C D) : Prop := sorry

def Trapezoid.hasParallelSides (t : Trapezoid A B C D) : Prop := sorry

def Trapezoid.hasInscribedCircle (t : Trapezoid A B C D) : Prop := sorry

-- Theorem statement
theorem isosceles_trapezoid_construction
  (P Q : Plane) (p : Line) (A B : Point) :
  p = Plane.intersect P Q →
  Point.onPlane A P →
  Point.onPlane B Q →
  ¬ Point.onLine A p →
  ¬ Point.onLine B p →
  ∃ (C D : Point) (t : Trapezoid A B C D),
    Point.onPlane B P ∧
    Point.onPlane D Q ∧
    Trapezoid.isIsosceles t ∧
    Trapezoid.hasParallelSides t ∧
    Trapezoid.hasInscribedCircle t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_construction_l100_10023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_division_l100_10084

-- Define the right angle and the rays
def angle_AOB : ℝ := 90
def angle_COD : ℝ := 10

-- Define the unknown angles as variables
variable (angle_AOC angle_DOB : ℝ)

-- State the theorem
theorem right_angle_division :
  angle_AOB = 90 ∧
  angle_COD = 10 ∧
  (max (max angle_AOC angle_COD) (max angle_DOB (angle_AOC + angle_COD))) +
  (min (min angle_AOC angle_COD) (min angle_DOB (angle_AOC + angle_DOB))) = 85 →
  (angle_AOC = 15 ∧ angle_DOB = 65 ∧ angle_COD = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_division_l100_10084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l100_10074

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * (1 - seq.r^n) / (1 - seq.r)

theorem geometric_sequence_sum_five (seq : GeometricSequence) :
  geometricSum seq 3 = 13 →
  geometricSum seq 9 = 295 →
  geometricSum seq 5 = 52 := by
  sorry

#check geometric_sequence_sum_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l100_10074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_triangle_perimeter_l100_10019

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y k m : ℝ) : Prop := y = k * x + m

-- Define the right focus F of ellipse C
noncomputable def focus_F : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the maximum distance between points on C and O
noncomputable def max_distance : ℝ := Real.sqrt 3 + 1

-- Define the perimeter of triangle FPQ
noncomputable def perimeter_FPQ : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem ellipse_circle_triangle_perimeter 
  (b k m : ℝ) 
  (h_b : b > 0) 
  (h_k : k < 0) 
  (h_m : m > 0) 
  (h_tangent : ∃ (x y : ℝ), circle_O x y ∧ line_l x y k m)
  (h_not_through_F : ¬ line_l (focus_F.1) (focus_F.2) k m)
  (h_intersect : ∃ (x1 y1 x2 y2 : ℝ), 
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧ 
    line_l x1 y1 k m ∧ line_l x2 y2 k m ∧ 
    (x1 ≠ x2 ∨ y1 ≠ y2)) :
  perimeter_FPQ = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_triangle_perimeter_l100_10019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_expenditure_l100_10080

/-- Represents the speed of the bus in km/h -/
def speed : ℝ → ℝ := λ _ => 0

/-- Represents the diesel consumption in litres per hour -/
noncomputable def consumption (v : ℝ) : ℝ := (1 / 2500) * v^2

/-- Represents the total expenditure for a trip -/
noncomputable def expenditure (v : ℝ) (d : ℝ) : ℝ := (50 / v + (1 / 2500) * v * 50) * d

/-- The problem statement -/
theorem minimum_expenditure :
  ∃ (v : ℝ), 
    v > 0 ∧ 
    consumption 50 = 1 ∧ 
    (∀ (u : ℝ), u > 0 → expenditure v 500 ≤ expenditure u 500) ∧
    expenditure v 500 = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_expenditure_l100_10080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l100_10039

/-- Circle C -/
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Point M -/
def M : ℝ × ℝ := (2, 4)

/-- Ellipse T -/
def T (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Line l -/
def l (x y k : ℝ) : Prop := y = k * x + Real.sqrt 3 ∧ k > 0

/-- Area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The theorem statement -/
theorem ellipse_and_triangle_area 
  (A B : ℝ × ℝ) -- Points where tangents touch circle C
  (P Q : ℝ × ℝ) -- Intersection points of line l with ellipse T
  (h1 : C A.1 A.2) 
  (h2 : C B.1 B.2)
  (h3 : ∃ (k : ℝ), l P.1 P.2 k ∧ l Q.1 Q.2 k)
  (h4 : ∃ (a b : ℝ), T 2 0 a b ∧ T 0 1 a b) -- Right and top vertices of T
  (h5 : (B.2 - A.2) * 2 = (B.1 - A.1) ∧ A.1 + 2 * A.2 = 2) -- Line AB passes through (2,0) and (0,1)
  : 
  (∃ (x y : ℝ), T x y 2 1) ∧ -- Ellipse T equation
  (∃ (S : ℝ), S = 1 ∧ ∀ (k : ℝ), k > 0 → 
    let O : ℝ × ℝ := (0, 0)
    area_triangle O P Q ≤ S) -- Maximum area of triangle OPQ
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l100_10039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l100_10002

/-- Represents the travel details of a person -/
structure TravelDetails where
  distance : ℝ
  time : ℝ

/-- Calculates the average speed given travel details -/
noncomputable def averageSpeed (td : TravelDetails) : ℝ :=
  td.distance / td.time

theorem eddy_travel_time :
  ∀ (eddy freddy : TravelDetails),
    freddy.distance = 300 →
    freddy.time = 4 →
    eddy.distance = 450 →
    averageSpeed eddy = 2 * averageSpeed freddy →
    eddy.time = 3 := by
  intro eddy freddy h1 h2 h3 h4
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l100_10002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_european_postcards_cost_l100_10075

/-- Represents a country --/
inductive Country
  | Germany
  | Italy
  | Canada
  | Mexico

/-- Represents a decade --/
inductive Decade
  | Fifties
  | Sixties
  | Seventies

/-- Price of a postcard in cents --/
def price (c : Country) : Nat :=
  match c with
  | Country.Germany => 7
  | Country.Italy => 7
  | Country.Canada => 5
  | Country.Mexico => 6

/-- Number of postcards for a given country and decade --/
def postcardCount (c : Country) (d : Decade) : Nat :=
  match c, d with
  | Country.Germany, Decade.Fifties => 5
  | Country.Germany, Decade.Sixties => 6
  | Country.Germany, Decade.Seventies => 9
  | Country.Italy, Decade.Fifties => 9
  | Country.Italy, Decade.Sixties => 8
  | Country.Italy, Decade.Seventies => 10
  | Country.Canada, Decade.Fifties => 8
  | Country.Canada, Decade.Sixties => 10
  | Country.Canada, Decade.Seventies => 12
  | Country.Mexico, Decade.Fifties => 7
  | Country.Mexico, Decade.Sixties => 8
  | Country.Mexico, Decade.Seventies => 11

/-- Whether a country is European --/
def isEuropean (c : Country) : Bool :=
  match c with
  | Country.Germany => true
  | Country.Italy => true
  | _ => false

/-- Total cost of postcards for a given country and decade in cents --/
def totalCost (c : Country) (d : Decade) : Nat :=
  price c * postcardCount c d

/-- Theorem: The total cost of European postcards issued before the 70's is 196 cents --/
theorem european_postcards_cost : 
  (totalCost Country.Germany Decade.Fifties + 
   totalCost Country.Germany Decade.Sixties + 
   totalCost Country.Italy Decade.Fifties + 
   totalCost Country.Italy Decade.Sixties) = 196 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_european_postcards_cost_l100_10075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l100_10020

noncomputable def f (x : ℝ) : ℝ := (10 * x + 10) / (x^2 + 2*x + 2)

theorem f_extrema :
  let a := -1
  let b := 2
  (∀ x ∈ Set.Icc a b, f x ≥ 0) ∧
  (∃ x ∈ Set.Icc a b, f x = 0) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) ∧
  f a = 0 ∧
  f 0 = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l100_10020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_equation_l100_10027

theorem product_of_logarithmic_equation (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  ∃ (a b c d : ℕ+), 
    (Real.sqrt (Real.log x) : ℝ) = a ∧
    (Real.sqrt (Real.log y) : ℝ) = b ∧
    (Real.log (Real.sqrt x) : ℝ) = c ∧
    (Real.log (Real.sqrt y) : ℝ) = d ∧
    a + b + c + d = 200 →
  x * y = (10 : ℝ)^202 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_equation_l100_10027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l100_10037

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 1/2
  h_intersect : ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2/a^2 + 1/b^2 = 1 ∧ x₂^2/a^2 + 1/b^2 = 1 ∧ (x₂ - x₁)^2 = 8

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) : Prop :=
  ∀ x y, x^2/4 + y^2/2 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- Area of quadrilateral OADB -/
noncomputable def area_OADB (e : Ellipse) (l : ℝ → ℝ) : ℝ :=
  let A := (Real.sqrt ((4*l 0^2 + 2 - (l 0)^2) / (1 + 2*l 0^2)), l 0 * Real.sqrt ((4*l 0^2 + 2 - (l 0)^2) / (1 + 2*l 0^2)))
  let B := (-Real.sqrt ((4*l 0^2 + 2 - (l 0)^2) / (1 + 2*l 0^2)), -l 0 * Real.sqrt ((4*l 0^2 + 2 - (l 0)^2) / (1 + 2*l 0^2)))
  let D := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  Real.sqrt 6

theorem ellipse_properties (e : Ellipse) :
  ellipse_equation e ∧
  (∀ l : ℝ → ℝ, area_OADB e l = Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l100_10037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_sums_not_equal_l100_10079

/-- Represents a point on the line -/
structure Point where
  position : ℕ
  color : Bool  -- true for red, false for blue

/-- The total number of points on the line -/
def total_points : ℕ := 2022

/-- Checks if the coloring is valid (equal number of red and blue points) -/
def valid_coloring (points : List Point) : Prop :=
  points.length = total_points ∧
  (points.filter (λ p => p.color)).length = total_points / 2

/-- Calculates the length of a segment between two points -/
def segment_length (p1 p2 : Point) : ℕ :=
  Int.natAbs (p2.position - p1.position)

/-- Calculates the sum of lengths of segments with specified left and right colors -/
def sum_segment_lengths (points : List Point) (left_color right_color : Bool) : ℕ :=
  points.foldl (λ acc p1 =>
    acc + (points.filter (λ p2 => p2.position > p1.position ∧ p1.color = left_color ∧ p2.color = right_color)
      |>.foldl (λ inner_acc p2 => inner_acc + segment_length p1 p2) 0)
  ) 0

/-- The main theorem to be proved -/
theorem segment_sums_not_equal (points : List Point) :
  valid_coloring points →
  sum_segment_lengths points true false ≠ sum_segment_lengths points false true := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_sums_not_equal_l100_10079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l100_10052

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to a line Ax + By + C = 0 -/
noncomputable def distanceToLine (p : Point) (A B C : ℝ) : ℝ :=
  |A * p.x + B * p.y + C| / Real.sqrt (A^2 + B^2)

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates :
  ∀ (P : Point),
    (2 * P.x + P.y - 3 = 0) →  -- P lies on the line 2x + y - 3 = 0
    (P.x > 0 ∧ P.y > 0) →  -- P is in the first quadrant
    (distanceToLine P 1 (-2) (-4) = Real.sqrt 5) →  -- Distance to x - 2y - 4 = 0 is √5
    (P.x = 1 ∧ P.y = 1) :=
by
  sorry

#check point_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l100_10052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_equals_sqrt_3_l100_10025

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Define the base case for n = 0 (corresponding to a₁)
  | (n + 1) => (sequence_a n - Real.sqrt 3) / (1 + Real.sqrt 3 * sequence_a n)

theorem a_6_equals_sqrt_3 : sequence_a 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_equals_sqrt_3_l100_10025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_school_time_l100_10008

/-- Represents the walking characteristics of a person -/
structure Walker where
  steps_per_minute : ℚ
  step_length : ℚ

/-- Calculates the time taken to cover a distance at a given speed -/
def time_taken (distance speed : ℚ) : ℚ := distance / speed

theorem jill_school_time (dave : Walker) (jill : Walker) (dave_time : ℚ) : 
  dave.steps_per_minute = 80 →
  dave.step_length = 70 →
  dave_time = 20 →
  jill.steps_per_minute = 120 →
  jill.step_length = 50 →
  time_taken (dave.steps_per_minute * dave.step_length * dave_time) 
             (jill.steps_per_minute * jill.step_length) = 56/3 := by
  sorry

#eval time_taken 112000 6000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_school_time_l100_10008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ABC_collinear_ACD_collinear_implies_k_value_l100_10085

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)
variable (O A B C D : V)
variable (k : ℝ)

-- Non-collinearity and non-zero conditions
axiom a_b_noncollinear : ¬ ∃ (r : ℝ), a = r • b
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0

-- Given vector relationships
axiom OA_def : A - O = 2 • a - b
axiom OB_def : B - O = 3 • a + b
axiom OC_def : C - O = a - 3 • b
axiom AB_def : B - A = a + b
axiom BC_def : C - B = 2 • a - 3 • b
axiom CD_def : D - C = 2 • a - k • b

-- Define collinearity
def collinear (P Q R : V) : Prop :=
  ∃ (t : ℝ), Q - P = t • (R - P)

-- Theorem 1
theorem ABC_collinear :
  collinear V A B C :=
sorry

-- Theorem 2
theorem ACD_collinear_implies_k_value :
  collinear V A C D → k = 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ABC_collinear_ACD_collinear_implies_k_value_l100_10085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unit_distance_partition_l100_10012

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A unit disc centered at the origin -/
def UnitDisc : Set Point :=
  {p : Point | p.x^2 + p.y^2 ≤ 1}

/-- A partition of a set into three subsets -/
structure Partition (S : Set Point) where
  A : Set Point
  B : Set Point
  C : Set Point
  partitionA : A ⊆ S
  partitionB : B ⊆ S
  partitionC : C ⊆ S
  disjointAB : A ∩ B = ∅
  disjointBC : B ∩ C = ∅
  disjointCA : C ∩ A = ∅
  cover : A ∪ B ∪ C = S

/-- No two points in a set are at distance 1 -/
def NoUnitDistance (S : Set Point) : Prop :=
  ∀ p q : Point, p ∈ S → q ∈ S → distance p q = 1 → p = q

theorem no_unit_distance_partition :
  ¬∃ (π : Partition UnitDisc), NoUnitDistance π.A ∧ NoUnitDistance π.B ∧ NoUnitDistance π.C := by
  sorry

#check no_unit_distance_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unit_distance_partition_l100_10012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forty_seventh_digit_of_one_seventeenth_l100_10078

/-- Represents a decimal digit at a specific position after the decimal point in a rational number. -/
def decimal_digit_at (q : ℚ) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 47th digit after the decimal point in 1/17 is 4. -/
theorem forty_seventh_digit_of_one_seventeenth : 
  decimal_digit_at (1/17) 47 = 4 :=
by
  sorry

#check forty_seventh_digit_of_one_seventeenth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forty_seventh_digit_of_one_seventeenth_l100_10078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l100_10001

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x^2

-- State the theorem
theorem tangent_function_properties :
  ∀ a b : ℝ,
  (∀ x : ℝ, x > 0 → DifferentiableAt ℝ (f a b) x) →
  (∃ x : ℝ, x > 0 ∧ f a b x = -1/2 ∧ deriv (f a b) x = 0) →
  (a = 1 ∧ b = 1/2) ∧
  (∀ x : ℝ, 1/exp 1 ≤ x ∧ x ≤ exp 1 → f 1 (1/2) x ≤ -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l100_10001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_SO2_produced_l100_10072

/-- Atomic weight of aluminum in g/mol -/
noncomputable def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of sulfur in g/mol -/
noncomputable def atomic_weight_S : ℝ := 32.06

/-- Atomic weight of oxygen in g/mol -/
noncomputable def atomic_weight_O : ℝ := 15.999

/-- Molecular weight of Al2S3 in g/mol -/
noncomputable def molecular_weight_Al2S3 : ℝ := 2 * atomic_weight_Al + 3 * atomic_weight_S

/-- Molecular weight of SO2 in g/mol -/
noncomputable def molecular_weight_SO2 : ℝ := atomic_weight_S + 2 * atomic_weight_O

/-- Number of moles of Al2S3 -/
noncomputable def moles_Al2S3 : ℝ := 4

/-- Balanced chemical reaction 1: 2 Al2S3 produces 6 H2S -/
noncomputable def reaction1_ratio : ℝ := 6 / 2

/-- Balanced chemical reaction 2: 1 H2S produces 1 SO2 -/
noncomputable def reaction2_ratio : ℝ := 1

theorem mass_SO2_produced :
  moles_Al2S3 * reaction1_ratio * reaction2_ratio * molecular_weight_SO2 = 768.696 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_SO2_produced_l100_10072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_upper_bound_l100_10005

theorem subset_implies_upper_bound (a : ℝ) :
  let A := {x : ℝ | x < a}
  let B := {x : ℝ | Real.rpow 2 x < 4}
  A ⊆ B → a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_upper_bound_l100_10005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_buses_l100_10009

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : Finset Nat
  stop_count : stops.card = 3

/-- The city's bus system -/
structure BusSystem where
  stops : Finset Nat
  routes : Finset BusRoute
  total_stops : stops.card = 9
  common_stop : ∀ r1 r2 : BusRoute, r1 ∈ routes → r2 ∈ routes → r1 ≠ r2 → (r1.stops ∩ r2.stops).card ≤ 1

/-- The theorem stating the maximum number of buses possible -/
theorem max_buses (system : BusSystem) : system.routes.card ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_buses_l100_10009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_specific_l100_10033

/-- The area of a lune formed by two semicircles -/
noncomputable def lune_area (d1 d2 : ℝ) : ℝ :=
  (1/2 - (d2^2/4) * Real.arcsin (d1/d2)) * Real.pi

/-- Theorem: The area of a lune formed by semicircles with diameters 2 and 3 -/
theorem lune_area_specific : lune_area 2 3 = (1/2 - 2.25 * Real.arcsin (2/3)) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_specific_l100_10033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_travel_time_l100_10016

/-- The time taken by a raft to float between two points given boat travel times -/
theorem raft_travel_time (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time > 0) (h2 : upstream_time > 0) (h3 : downstream_time ≠ upstream_time) :
  let boat_speed := (downstream_time + upstream_time) / (2 * downstream_time * upstream_time)
  let current_speed := (upstream_time - downstream_time) / (2 * downstream_time * upstream_time)
  let distance := boat_speed * downstream_time + current_speed * downstream_time
  distance / current_speed = downstream_time * upstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_travel_time_l100_10016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_distance_l100_10051

/-- Terese's running schedule for a week -/
structure RunningSchedule where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- The average distance Terese runs each day -/
noncomputable def average_distance (s : RunningSchedule) : ℝ :=
  (s.monday + s.tuesday + s.wednesday + s.thursday) / 4

/-- Theorem: Given Terese's running schedule for Monday to Wednesday
    and the average distance, prove that she runs 4.4 miles on Thursday -/
theorem thursday_distance (s : RunningSchedule) 
    (h1 : s.monday = 4.2)
    (h2 : s.tuesday = 3.8)
    (h3 : s.wednesday = 3.6)
    (h4 : average_distance s = 4) :
    s.thursday = 4.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_distance_l100_10051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l100_10099

/- Define the function f -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x - Real.pi/6) - 1/2

/- Define the triangle ABC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/- State the theorem -/
theorem triangle_and_function_properties 
  (t : Triangle) 
  (h1 : t.c = Real.sqrt 3)
  (h2 : f t.C = 0)
  (h3 : Real.sin t.B = 2 * Real.sin t.A) : 
  (∀ x, f x ≥ -2) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧ 
  (t.a = 1) ∧ 
  (t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l100_10099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l100_10076

/-- Represents the taxi fare structure in a certain city -/
structure TaxiFare where
  initialFare : ℚ
  additionalCharge : ℚ
  initialDistance : ℚ := 3

/-- Calculates the total fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℚ) : ℚ :=
  tf.initialFare + max 0 (distance - tf.initialDistance) * tf.additionalCharge

theorem taxi_fare_theorem (tf : TaxiFare) : 
  (calculateFare tf 10 = 33/2 ∧ calculateFare tf 14 = 45/2) → 
  (tf.initialFare = 6 ∧ tf.additionalCharge = 3/2 ∧ calculateFare tf 7 = 12) := by
  sorry

#check taxi_fare_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l100_10076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_case_l100_10062

/-- The length of a wire stretched between the tops of two vertical poles -/
noncomputable def wire_length (distance_between_poles height_pole1 height_pole2 : ℝ) : ℝ :=
  Real.sqrt (distance_between_poles^2 + (height_pole2 - height_pole1)^2)

/-- Theorem stating the length of the wire in the given scenario -/
theorem wire_length_specific_case : 
  wire_length 20 8 18 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_case_l100_10062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_t_value_l100_10040

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given two non-zero vectors a and b that are not collinear, and points A, B, C defined by
    OA = a, OB = t * b, OC = (1/3) * (a + b), prove that when A, B, and C are collinear, t = 1/2. -/
theorem collinear_points_t_value
  (a b : V) (t : ℝ)
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b)
  (h_collinear : ∃ (lambda : ℝ), (1/3) • (a + b) = lambda • a + (1 - lambda) • (t • b)) :
  t = 1/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_t_value_l100_10040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sides_of_dissected_polygons_l100_10044

/-- Represents a polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℚ × ℚ)
  convex : Bool

/-- Represents a line segment with rational endpoints -/
structure Segment where
  start : ℚ × ℚ
  stop : ℚ × ℚ

/-- A function to check if a segment has rational length -/
def has_rational_length (s : Segment) : Prop :=
  ∃ q : ℚ, q^2 = (s.stop.1 - s.start.1)^2 + (s.stop.2 - s.start.2)^2

/-- A function to get all sides of a polygon -/
def sides (p : Polygon) : List Segment :=
  sorry

/-- A function to get all diagonals of a polygon -/
def diagonals (p : Polygon) : List Segment :=
  sorry

/-- A function to dissect a polygon by its diagonals -/
def dissect (p : Polygon) : List Polygon :=
  sorry

/-- The main theorem -/
theorem rational_sides_of_dissected_polygons 
  (p : Polygon) 
  (h_convex : p.convex)
  (h_rational_sides : ∀ s ∈ sides p, has_rational_length s)
  (h_rational_diagonals : ∀ d ∈ diagonals p, has_rational_length d) :
  ∀ q ∈ dissect p, ∀ s ∈ sides q, has_rational_length s :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sides_of_dissected_polygons_l100_10044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_simplification_l100_10060

/-- The main theorem to prove -/
theorem complex_power_simplification :
  ((1 + Complex.I) / (1 - Complex.I))^2006 = (-1 : ℂ) := by
  -- Proof steps would go here
  sorry

/-- A helper lemma that might be useful in the proof -/
lemma complex_fraction_equals_i :
  (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_simplification_l100_10060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_probabilities_l100_10058

theorem chess_probabilities (draw_prob : ℚ) (b_win_prob : ℚ)
  (h1 : draw_prob = 1/2)
  (h2 : b_win_prob = 1/3) :
  let a_win_prob := 1 - draw_prob - b_win_prob
  let a_not_lose_prob := draw_prob + a_win_prob
  let b_lose_prob := a_win_prob
  let b_not_lose_prob := draw_prob + b_win_prob
  (a_win_prob = 1/6) ∧
  (a_not_lose_prob = 2/3) ∧
  (b_lose_prob = 1/6) ∧
  (b_not_lose_prob = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_probabilities_l100_10058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_interest_rate_l100_10011

/-- Proves that given a bill with a face value of 1680 and a true discount of 180 for a 9-month period, the annual interest rate is 16% -/
theorem bill_interest_rate (face_value : ℝ) (true_discount : ℝ) (months : ℝ) :
  face_value = 1680 →
  true_discount = 180 →
  months = 9 →
  (let present_value := face_value - true_discount
   let time := months / 12
   let interest_rate := (true_discount * 100) / (present_value * time)
   interest_rate = 16) := by
  intro h1 h2 h3
  -- The proof goes here
  sorry

#check bill_interest_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_interest_rate_l100_10011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l100_10098

theorem cos_alpha_for_point (P : ℝ × ℝ) (α : ℝ) : 
  P = (-6, 8) → Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l100_10098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_7_l100_10043

def isDivisibleBy7 (n : ℕ) : Bool := n % 7 = 0

def numbersInRange : List ℕ := (List.range 55).filter (fun n => 6 < n ∧ n < 55 ∧ isDivisibleBy7 n)

theorem average_of_numbers_divisible_by_7 :
  let sum := numbersInRange.sum
  let count := numbersInRange.length
  sum / count = 28 := by
    sorry

#eval numbersInRange
#eval numbersInRange.sum
#eval numbersInRange.length
#eval numbersInRange.sum / numbersInRange.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_7_l100_10043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l100_10081

/-- Given an ellipse with equation x²/(m+1) + y² = 1 where m > 0,
    with foci F₁ and F₂, and a point E that is the intersection of
    the line y = x + 2 and the ellipse, when the sum of distances
    |EF₁| + |EF₂| is minimized, the eccentricity of the ellipse
    is √6/3. -/
theorem ellipse_eccentricity (m : ℝ) (F₁ F₂ E : ℝ × ℝ) 
  (ellipse : Set (ℝ × ℝ)) (dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ) 
  (eccentricity : Set (ℝ × ℝ) → ℝ) :
  m > 0 →
  (∀ x y, (x^2 / (m + 1) + y^2 = 1) ↔ (x, y) ∈ ellipse) →
  E ∈ ellipse →
  E.2 = E.1 + 2 →
  (∀ P ∈ ellipse, dist E F₁ + dist E F₂ ≤ dist P F₁ + dist P F₂) →
  eccentricity ellipse = Real.sqrt 6 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l100_10081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_first_exponent_l100_10061

theorem base_of_first_exponent (x b : ℕ) : 
  (18 ^ 6) * 9 ^ (3 * 6 - 1) = (x ^ 6) * (3 ^ b) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_first_exponent_l100_10061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l100_10004

/-- An isosceles trapezoid with given side lengths -/
structure IsoscelesTrapezoid where
  leg : ℝ
  base1 : ℝ
  base2 : ℝ

/-- The area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  let height := Real.sqrt (t.leg ^ 2 - ((t.base2 - t.base1) / 2) ^ 2)
  (t.base1 + t.base2) * height / 2

/-- Theorem: The area of the specified isosceles trapezoid is 40 -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := ⟨5, 7, 13⟩
  area t = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l100_10004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_y_l100_10064

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def FinalMixture (x y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xWeight + y.ryegrass * (1 - xWeight),
    bluegrass := x.bluegrass * xWeight + y.bluegrass * (1 - xWeight),
    fescue := x.fescue * xWeight + y.fescue * (1 - xWeight) }

theorem ryegrass_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.bluegrass = 0.6)
  (h3 : y.fescue = 0.75)
  (h4 : (FinalMixture x y 0.6667).ryegrass = 0.35) :
  y.ryegrass = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_y_l100_10064
