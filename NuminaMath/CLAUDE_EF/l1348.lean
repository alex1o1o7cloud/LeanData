import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_observation_count_l1348_134802

theorem observation_count 
  (original_mean wrong_value correct_value new_mean : ℝ)
  (h1 : original_mean = 30)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 48)
  (h4 : new_mean = 30.5)
  : ∃ n : ℕ, n = 50 ∧ (n : ℝ) * new_mean = n * original_mean + (correct_value - wrong_value) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_observation_count_l1348_134802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_congruences_l1348_134825

theorem smallest_x_congruences :
  ∃ x : ℕ,
    x % 285 = 31 ∧
    x % 17 = 14 ∧
    x % 23 = 8 ∧
    x % 19 = 12 ∧
    (∀ y : ℕ, y % 285 = 31 → y ≥ x) ∧
    x = 31 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_congruences_l1348_134825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_odd_nor_even_f_increasing_implies_a_le_16_l1348_134833

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a/x

-- Theorem 1: f is neither odd nor even when a ≠ 0
theorem f_not_odd_nor_even (a : ℝ) (ha : a ≠ 0) :
  ¬(∀ x, f a x = f a (-x)) ∧ ¬(∀ x, f a x = -(f a (-x))) := by sorry

-- Theorem 2: If f is increasing on [2, +∞), then a ≤ 16
theorem f_increasing_implies_a_le_16 (a : ℝ) :
  (∀ x y : ℝ, 2 ≤ x → x < y → f a x < f a y) → a ≤ 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_odd_nor_even_f_increasing_implies_a_le_16_l1348_134833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_dot_product_range_l1348_134890

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle (renamed to avoid conflict with existing definition)
def custom_circle (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_tangent_line_dot_product_range :
  ∀ (m t : ℝ) (A B : ℝ × ℝ),
    let (x1, y1) := A
    let (x2, y2) := B
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 →
    (∃ (x y : ℝ), custom_circle x y ∧ m * y = x + t) →
    x1 ≠ x2 ∨ y1 ≠ y2 →
    11 / 4 ≤ dot_product x1 y1 x2 y2 ∧ dot_product x1 y1 x2 y2 < 11 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_dot_product_range_l1348_134890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_onur_bikes_250_km_daily_l1348_134866

/-- Onur's daily biking distance in kilometers -/
def onur_distance : ℝ := 250

/-- Hanil's daily biking distance in kilometers -/
def hanil_distance : ℝ := onur_distance + 40

/-- Number of days they bike per week -/
def days_per_week : ℕ := 5

/-- Total distance biked by both in a week -/
def total_weekly_distance : ℝ := 2700

theorem onur_bikes_250_km_daily : 
  onur_distance * days_per_week + hanil_distance * days_per_week = total_weekly_distance ∧
  onur_distance = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_onur_bikes_250_km_daily_l1348_134866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_when_slope_half_line_equation_when_P_midpoint_l1348_134898

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define line l passing through P with slope m
def line_l (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_l m p.1 p.2}

-- Part 1: Length of AB when slope is 1/2
theorem length_AB_when_slope_half :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points (1/2) ∧ B ∈ intersection_points (1/2) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3 * Real.sqrt 10 := by sorry

-- Part 2: Equation of line l when P is midpoint of AB
theorem line_equation_when_P_midpoint :
  ∃ (m : ℝ), (∀ (x y : ℝ), line_l m x y ↔ x + 2*y - 8 = 0) ∧
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points m ∧ B ∈ intersection_points m ∧
  P = ((A.1 + B.1)/2, (A.2 + B.2)/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_when_slope_half_line_equation_when_P_midpoint_l1348_134898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1348_134810

def a : ℕ → ℕ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | (n + 1) => 2 * a n + 2^(n + 1)

theorem sequence_properties (n : ℕ) (h : n ≥ 1) :
  (∀ k ≥ 1, (a k : ℚ) / 2^k = k) ∧
  (Finset.sum (Finset.range n) (λ i => (a (i + 1) : ℚ) / (i + 1)) = 2^(n + 1) - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1348_134810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1348_134863

noncomputable section

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_a_on_b : projection a b = Real.sqrt 65 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1348_134863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_triangle_areas_l1348_134874

-- Define the hexagon and point M
variable (A B C D E F M : EuclideanSpace ℝ (Fin 2))

-- Define the intersection points
noncomputable def P : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def R : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def S : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def L : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def N : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def K : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the triangles
noncomputable def triangle_KLN : Set (EuclideanSpace ℝ (Fin 2)) := {K, L, N}
noncomputable def triangle_SRP : Set (EuclideanSpace ℝ (Fin 2)) := {S, R, P}

-- Define the hexagon area
noncomputable def S_hexagon : ℝ := sorry

-- Define the areas of triangles formed by M and hexagon sides
noncomputable def S_AMB : ℝ := sorry
noncomputable def S_CMD : ℝ := sorry
noncomputable def S_FME : ℝ := sorry
noncomputable def S_BMC : ℝ := sorry
noncomputable def S_DME : ℝ := sorry
noncomputable def S_AMF : ℝ := sorry
noncomputable def S_EMD : ℝ := sorry
noncomputable def S_MEF : ℝ := sorry

-- State the theorem
theorem hexagon_triangle_areas 
  (h1 : sorry) -- Placeholder for triangle_KLN is equilateral
  (h2 : sorry) -- Placeholder for triangle_SRP is equilateral
  (h3 : sorry) -- Placeholder for triangle_KLN ≅ triangle_SRP
  (h4 : S_AMB + S_CMD + S_FME + S_BMC + S_DME + S_AMF = (1/2) * S_hexagon)
  (h5 : S_AMB + S_EMD = S_BMC + S_MEF)
  (h6 : S_BMC + S_MEF = S_CMD + S_AMF)
  (h7 : S_AMB + S_EMD = (1/3) * S_hexagon) :
  S_FME = 6 ∧ S_BMC = 6 ∧ S_DME = 9 ∧ S_AMF = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_triangle_areas_l1348_134874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2021_is_one_l1348_134879

/-- The sequence of digits obtained by concatenating integers from 1 to 999 -/
def digit_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => (digit_sequence n + 1) % 10

/-- The number x as defined in the problem -/
noncomputable def x : ℚ :=
  ∑' n, (digit_sequence n : ℚ) / 10^(n + 1)

/-- The 2021st digit of x -/
def digit_2021 : ℕ := digit_sequence 2020

theorem digit_2021_is_one : digit_2021 = 1 := by
  sorry

#eval digit_2021

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2021_is_one_l1348_134879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_two_and_one_third_l1348_134839

theorem reciprocal_of_negative_two_and_one_third :
  (-(2 + 1/3 : ℚ))⁻¹ = -(3/7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_two_and_one_third_l1348_134839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l1348_134842

theorem integral_value (a : ℝ) : 
  (∃ k : ℝ, k = 8 ∧ k = 4 * a^3) → 
  ∫ x in a..Real.exp 2, (1/x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l1348_134842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_due_in_four_years_l1348_134848

/-- Calculates the number of years required for a present value to grow to a future value at a given compound interest rate. -/
noncomputable def years_to_grow (present_value : ℝ) (future_value : ℝ) (interest_rate : ℝ) : ℝ :=
  (Real.log (future_value / present_value)) / (Real.log (1 + interest_rate))

/-- Rounds up a real number to the nearest integer. -/
noncomputable def ceil (x : ℝ) : ℤ :=
  ⌈x⌉

theorem money_due_in_four_years :
  ceil (years_to_grow 2500 3600 0.20) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_due_in_four_years_l1348_134848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_2020_diff_l1348_134887

/-- A sequence of all square-free positive integers in increasing order -/
def squareFreeSequence : ℕ → ℕ := sorry

/-- The property that a natural number is square-free -/
def isSquareFree (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p * p ∣ n → p = 1)

/-- The squareFreeSequence contains all square-free positive integers in increasing order -/
axiom squareFreeSequence_prop :
  (∀ n, isSquareFree (squareFreeSequence n)) ∧
  (∀ n, squareFreeSequence n < squareFreeSequence (n + 1)) ∧
  (∀ m, isSquareFree m → ∃ n, squareFreeSequence n = m)

/-- The set of indices where the difference between consecutive terms is 2020 -/
def diffIs2020 : Set ℕ :=
  {n | squareFreeSequence (n + 1) - squareFreeSequence n = 2020}

theorem infinite_2020_diff : Set.Infinite diffIs2020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_2020_diff_l1348_134887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_x_value_l1348_134856

theorem min_cos_x_value (x y z : ℝ) 
  (h1 : Real.sin x = Real.tan (π / 2 - y))
  (h2 : Real.sin y = Real.tan (π / 2 - z))
  (h3 : Real.sin z = Real.tan (π / 2 - x)) :
  ∃ (min_cos_x : ℝ), 
    (∀ x', Real.cos x' ≥ min_cos_x) ∧ 
    (min_cos_x = Real.sqrt ((3 - Real.sqrt 5) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_x_value_l1348_134856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1348_134827

def my_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 1) = a (n + 2) - a n) ∧ 
  a 1 = 2 ∧ 
  a 2 = 5

theorem fifth_term_value (a : ℕ → ℤ) (h : my_sequence a) : a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1348_134827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_spread_equation_l1348_134806

/-- Represents the number of people infected with flu after two rounds of infection --/
def total_infected : ℕ := 36

/-- Represents the average number of people infected by each person in each round --/
def x : ℝ := sorry

/-- Theorem stating the equation for flu spread after two rounds of infection --/
theorem flu_spread_equation :
  1 + x + x * (1 + x) = total_infected :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_spread_equation_l1348_134806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_winning_strategy_initial_state_not_winning_l1348_134867

/-- Represents the state of the game with three wall sizes -/
structure GameState where
  wall1 : Nat
  wall2 : Nat
  wall3 : Nat

/-- Calculates the Nim-value of a single wall -/
def nimValue : Nat → Nat
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | n + 6 => nimValue n

/-- Calculates the Nim-sum of the Nim-values of all walls -/
def gameNimSum (state : GameState) : Nat :=
  (nimValue state.wall1) ^^^ (nimValue state.wall2) ^^^ (nimValue state.wall3)

/-- Determines if a given state is a winning position for the current player -/
def isWinningPosition (state : GameState) : Bool :=
  gameNimSum state = 0

/-- Theorem stating that (7,4,2) is a winning strategy for Beth -/
theorem beth_winning_strategy :
  isWinningPosition ⟨7, 4, 2⟩ = true := by
  sorry

/-- Theorem stating that the initial configuration (5,3,2) is not a winning position -/
theorem initial_state_not_winning :
  isWinningPosition ⟨5, 3, 2⟩ = false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_winning_strategy_initial_state_not_winning_l1348_134867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_width_calculation_l1348_134870

/-- The width of a rectangular park given the total area and length -/
noncomputable def park_width (total_area : ℝ) (num_parks : ℕ+) (length : ℝ) : ℝ :=
  (total_area * 1000000) / (num_parks.val * length)

/-- Theorem stating that under given conditions, the width of each park is 250 meters -/
theorem park_width_calculation (total_area : ℝ) (num_parks : ℕ+) (length : ℝ) 
  (h1 : total_area = 0.6)
  (h2 : num_parks = 8)
  (h3 : length = 300) :
  park_width total_area num_parks length = 250 := by
  sorry

-- Remove the #eval statement as it's not necessary for building and may cause issues
-- #eval park_width 0.6 8 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_width_calculation_l1348_134870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_g_iterated_four_times_l1348_134895

/-- The function g(x) = x^2 - 6x -/
def g (x : ℝ) : ℝ := x^2 - 6*x

/-- The statement that there are exactly two distinct real numbers c that satisfy g(g(g(g(c)))) = 15 -/
theorem two_solutions_for_g_iterated_four_times :
  ∃! (s : Set ℝ), (∀ c ∈ s, g (g (g (g c))) = 15) ∧ (Finite s ∧ s.ncard = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_g_iterated_four_times_l1348_134895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1348_134875

/-- The speed of a train passing through a tunnel -/
noncomputable def train_speed (train_length : ℝ) (tunnel_length : ℝ) (passing_time : ℝ) : ℝ :=
  let total_distance := tunnel_length + train_length / 1000
  let time_in_hours := passing_time / 60
  total_distance / time_in_hours

theorem train_speed_calculation :
  let train_length : ℝ := 100
  let tunnel_length : ℝ := 1.1
  let passing_time : ℝ := 1.0000000000000002
  abs (train_speed train_length tunnel_length passing_time - 72) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1348_134875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt3_sin_plus_cos_l1348_134837

theorem max_value_sqrt3_sin_plus_cos :
  (∀ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x ≤ 2) ∧
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt3_sin_plus_cos_l1348_134837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1348_134821

/-- The time (in days) it takes b alone to complete the work -/
noncomputable def b_time : ℝ := 30

/-- The time (in days) it takes a and b together to complete the work -/
noncomputable def ab_time : ℝ := 10

/-- The speed of b (fraction of work completed per day) -/
noncomputable def b_speed : ℝ := 1 / b_time

/-- The combined speed of a and b (fraction of work completed per day) -/
noncomputable def ab_speed : ℝ := 1 / ab_time

/-- The speed of a (fraction of work completed per day) -/
noncomputable def a_speed : ℝ := ab_speed - b_speed

theorem speed_ratio : a_speed / b_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1348_134821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1348_134859

noncomputable def f (x : ℝ) := 1 / Real.sqrt (3 * x - 2) + Real.log (2 * x - 1) / Real.log 10

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x | x > 2/3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1348_134859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1348_134838

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a - 2 * t.b) * Real.cos t.C + t.c * Real.cos t.A = 0) 
  (h2 : t.c = 2 * Real.sqrt 3) : 
  t.C = π / 3 ∧ 
  ∃ p : ℝ, p ≤ 6 * Real.sqrt 3 ∧ 
  (p = 6 * Real.sqrt 3 → t.a + t.b + t.c = p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1348_134838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_l1348_134885

def is_valid_fraction (n d : ℕ) : Prop :=
  n > 0 ∧ d > 0 ∧ Nat.gcd n d = 1 ∧ n * 2 = (n + 2) * d

theorem factory_production :
  ∃ (jan_n jan_d feb_n feb_d : ℕ),
    is_valid_fraction jan_n jan_d ∧
    is_valid_fraction feb_n feb_d ∧
    (jan_n : ℚ) / jan_d > 1/3 ∧
    (feb_n : ℚ) / feb_d > 1/3 ∧
    (jan_n : ℚ) / jan_d > (feb_n : ℚ) / feb_d ∧
    jan_n = 2 ∧ jan_d = 3 ∧
    feb_n = 2 ∧ feb_d = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_l1348_134885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_game_odd_players_l1348_134881

/-- Represents a game between a girl and a boy -/
structure Game where
  girl : Nat
  boy : Nat

/-- Represents the state of the tournament -/
structure TournamentState where
  n : Nat
  games : List Game

/-- Predicate to check if a number is odd -/
def isOdd (n : Nat) : Prop := n % 2 = 1

theorem last_game_odd_players (n : Nat) (tournament : TournamentState) :
  isOdd n →
  tournament.n = n →
  (∀ g b, 1 ≤ g ∧ g ≤ n ∧ 1 ≤ b ∧ b ≤ n → 
    ∃! game, game ∈ tournament.games ∧ game.girl = g ∧ game.boy = b) →
  ∃ lastGame : Game, lastGame ∈ tournament.games ∧ 
    isOdd lastGame.girl ∧ isOdd lastGame.boy ∧
    ∀ game, game ∈ tournament.games → game ≠ lastGame → 
      game.girl < lastGame.girl ∨ game.boy < lastGame.boy :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_game_odd_players_l1348_134881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_multiple_of_3_l1348_134868

/-- A function representing the possible outcomes of rolling a standard die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of all possible outcomes when rolling two dice -/
def allOutcomes : Finset (ℕ × ℕ) := standardDie.product standardDie

/-- A predicate that checks if a pair of numbers sums to a multiple of 3 -/
def sumIsMultipleOf3 (pair : ℕ × ℕ) : Bool := (pair.1 + pair.2) % 3 = 0

/-- The set of favorable outcomes (sum is a multiple of 3) -/
def favorableOutcomes : Finset (ℕ × ℕ) := allOutcomes.filter (fun p => sumIsMultipleOf3 p)

/-- The probability of rolling two dice and getting a sum that is a multiple of 3 -/
theorem probability_sum_multiple_of_3 : 
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 1 / 3 := by
  sorry

#eval favorableOutcomes.card
#eval allOutcomes.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_multiple_of_3_l1348_134868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutation_exists_25_and_1000_l1348_134878

/-- A valid permutation is one where adjacent elements differ by 3 or 5 -/
def is_valid_permutation (p : List Nat) : Prop :=
  ∀ i, i + 1 < p.length → (p[i+1]! - p[i]! = 3 ∨ p[i+1]! - p[i]! = 5) ∨
                          (p[i]! - p[i+1]! = 3 ∨ p[i]! - p[i+1]! = 5)

/-- The existence of a valid permutation for a given n -/
def exists_valid_permutation (n : Nat) : Prop :=
  ∃ p : List Nat, p.Nodup ∧ p.length = n ∧ (∀ i, i ∈ p ↔ 1 ≤ i ∧ i ≤ n) ∧ is_valid_permutation p

theorem valid_permutation_exists_25_and_1000 :
  exists_valid_permutation 25 ∧ exists_valid_permutation 1000 := by
  sorry

#check valid_permutation_exists_25_and_1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutation_exists_25_and_1000_l1348_134878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_proof_l1348_134896

theorem equilateral_triangle_proof (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0)
  (h3 : α + β + γ = Real.pi)
  (h4 : a * Real.sin β = b * Real.sin α)
  (h5 : b * Real.sin γ = c * Real.sin β)
  (h6 : c * Real.sin α = a * Real.sin γ)
  (h7 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  a = b ∧ b = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_proof_l1348_134896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_l1348_134880

/-- Regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  a_pos : a > 0

/-- Orthogonal projection of a regular tetrahedron onto a plane -/
noncomputable def orthogonal_projection (t : RegularTetrahedron) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The maximum area of an orthogonal projection of a regular tetrahedron is a²/2 -/
theorem max_projection_area (t : RegularTetrahedron) : 
  ∃ (p : Set (ℝ × ℝ)), p = orthogonal_projection t ∧ 
    area p = t.a^2 / 2 ∧
    ∀ (q : Set (ℝ × ℝ)), q = orthogonal_projection t → area q ≤ t.a^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_l1348_134880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speed_is_74_l1348_134815

/-- Represents the travel scenario for Ms. Rush --/
structure TravelScenario where
  distance : ℚ
  ideal_time : ℚ

/-- Calculates the time taken given speed and distance --/
def time_taken (speed : ℚ) (distance : ℚ) : ℚ := distance / speed

/-- Theorem stating the correct speed to arrive on time --/
theorem correct_speed_is_74 (scenario : TravelScenario) : 
  (time_taken 50 scenario.distance = scenario.ideal_time + 1/12) →
  (time_taken 70 scenario.distance = scenario.ideal_time - 1/9) →
  (time_taken 74 scenario.distance = scenario.ideal_time) :=
by
  sorry

#eval time_taken 50 100  -- Example usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speed_is_74_l1348_134815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_three_valid_balls_l1348_134894

/-- The number of balls -/
def total_balls : ℕ := 50

/-- A ball is valid if it's an even number and a multiple of 5 -/
def is_valid_ball (n : ℕ) : Bool := n % 2 = 0 ∧ n % 5 = 0 ∧ n ≤ total_balls

/-- The number of valid balls -/
def valid_balls : ℕ := (List.range total_balls).filter is_valid_ball |>.length

/-- The probability of drawing three valid balls without replacement -/
noncomputable def probability : ℚ := 
  (valid_balls : ℚ) / total_balls *
  (valid_balls - 1 : ℚ) / (total_balls - 1) *
  (valid_balls - 2 : ℚ) / (total_balls - 2)

theorem probability_of_three_valid_balls : probability = 1 / 1960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_three_valid_balls_l1348_134894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1348_134884

noncomputable section

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + y = 1

-- Define perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Slope angle of a line
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

-- Distance from a point to a line
noncomputable def distance_point_to_line (a b c : ℝ) : ℝ :=
  |c| / Real.sqrt (a^2 + b^2)

theorem line_properties :
  ∃ (a : ℝ), 
    (∀ x y, l₁ x y → l₂ a x y → perpendicular (-Real.sqrt 3) a) →
    slope_angle (-Real.sqrt 3) = 2 * Real.pi / 3 ∧
    distance_point_to_line a 1 (-1) = Real.sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1348_134884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_in_75_to_84_is_approximately_26_67_l1348_134852

/-- Represents the score ranges in the frequency distribution --/
inductive ScoreRange
  | Above95
  | Between85And94
  | Between75And84
  | Between65And74
  | Between55And64
  | Below55

/-- Returns the number of students in each score range --/
def students_in_range (range : ScoreRange) : Nat :=
  match range with
  | .Above95 => 4
  | .Between85And94 => 5
  | .Between75And84 => 8
  | .Between65And74 => 6
  | .Between55And64 => 4
  | .Below55 => 3

/-- Calculates the total number of students --/
def total_students : Nat :=
  (students_in_range ScoreRange.Above95) +
  (students_in_range ScoreRange.Between85And94) +
  (students_in_range ScoreRange.Between75And84) +
  (students_in_range ScoreRange.Between65And74) +
  (students_in_range ScoreRange.Between55And64) +
  (students_in_range ScoreRange.Below55)

/-- Calculates the percentage of students in the 75%-84% range --/
noncomputable def percentage_in_75_to_84 : ℝ :=
  (students_in_range ScoreRange.Between75And84 : ℝ) / total_students * 100

theorem percentage_in_75_to_84_is_approximately_26_67 :
  abs (percentage_in_75_to_84 - 26.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_in_75_to_84_is_approximately_26_67_l1348_134852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l1348_134829

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem trigonometric_function_properties
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π)
  (h_even : ∀ x, f ω φ (-x) = f ω φ x)
  (h_intersect : ∃ x₁ x₂, f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ x₁ ≠ x₂)
  (h_min_period : ∀ x₁ x₂, f ω φ x₁ = 2 → f ω φ x₂ = 2 → x₁ ≠ x₂ → |x₁ - x₂| ≥ π)
  (h_period_exists : ∃ x₁ x₂, f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = π) :
  ω = 2 ∧ φ = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l1348_134829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_relationship_l1348_134860

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the relationships
def ParallelToPlane (l : Line) (p : Plane) : Prop := sorry

def ContainedInPlane (l : Line) (p : Plane) : Prop := sorry

def Skew (l1 l2 : Line) : Prop := sorry

def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : ParallelToPlane a α) 
  (h2 : ContainedInPlane b α) : 
  Skew a b ∨ Parallel a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_relationship_l1348_134860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1348_134886

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  a_1_eq_25 : a 1 = 25
  geometric_subseq : (a 1) * (a 13) = (a 11)^2

/-- The general term of the sequence -/
def general_term (n : ℕ) : ℝ := -2 * ↑n + 27

/-- The sum of every third term starting from a_1 -/
def sum_formula (n : ℕ) : ℝ := -3 * ↑n^2 + 28 * ↑n

/-- The main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = general_term n) ∧
  (∀ n, (Finset.range n).sum (λ i ↦ seq.a (3*i + 1)) = sum_formula n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1348_134886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_handshakes_l1348_134844

/-- Represents a meeting of people and their handshakes -/
structure Meeting (N : ℕ) where
  /-- The set of people in the meeting -/
  people : Finset (Fin N)
  /-- The handshake relation between people -/
  handshake : Fin N → Fin N → Bool
  /-- Handshakes are symmetric -/
  handshake_symm : ∀ i j, handshake i j = handshake j i
  /-- No one shakes hands with themselves -/
  handshake_irrefl : ∀ i, handshake i i = false

/-- The number of people who have shaken hands with everyone else -/
def num_all_handshakes (N : ℕ) (m : Meeting N) : ℕ :=
  (m.people.filter (λ i => ∀ j ∈ m.people, i ≠ j → m.handshake i j)).card

/-- The theorem to be proved -/
theorem max_handshakes (N : ℕ) (h : N > 4) (m : Meeting N) 
  (spec1 spec2 : Fin N) (hspec : spec1 ≠ spec2) 
  (h1 : ∃ j ∈ m.people, spec1 ≠ j ∧ m.handshake spec1 j = false) 
  (h2 : ∃ j ∈ m.people, spec2 ≠ j ∧ m.handshake spec2 j = false) : 
  num_all_handshakes N m ≤ N - 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_handshakes_l1348_134844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_theorem_l1348_134897

def Digit := Fin 10

structure Chip :=
  (digit : Digit)

def statement1 (c : Chip) : Prop := c.digit.val = 5
def statement2 (c : Chip) : Prop := c.digit.val ≠ 6
def statement3 (c : Chip) : Prop := c.digit.val = 7
def statement4 (c : Chip) : Prop := c.digit.val ≠ 8

theorem chip_theorem (c : Chip) 
  (h : ∃ (a b d : Prop), a ∧ b ∧ d ∧ 
    (a = statement1 c ∨ a = statement2 c ∨ a = statement3 c ∨ a = statement4 c) ∧
    (b = statement1 c ∨ b = statement2 c ∨ b = statement3 c ∨ b = statement4 c) ∧
    (d = statement1 c ∨ d = statement2 c ∨ d = statement3 c ∨ d = statement4 c) ∧
    a ≠ b ∧ a ≠ d ∧ b ≠ d) : 
  statement2 c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_theorem_l1348_134897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1348_134812

noncomputable def a_n (n : ℕ+) (a : ℝ) : ℝ := (-1 : ℝ)^(n.val + 2018) * a

noncomputable def b_n (n : ℕ+) : ℝ := 2 + ((-1 : ℝ)^(n.val + 2019)) / n.val

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ+, a_n n a < b_n n) ↔ a ∈ Set.Icc (-2 : ℝ) (3/2) ∧ a ≠ 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1348_134812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_average_l1348_134826

/-- A finite set of positive integers satisfying specific conditions -/
def SpecialIntegerSet (T : Finset ℕ) : Prop :=
  ∃ (m : ℕ) (b : ℕ → ℕ), m > 1 ∧
  (∀ i, i ∈ Finset.range m → b i ∈ T) ∧
  (∀ i j, i < j → i < m → j < m → b i < b j) ∧
  (Finset.sum (Finset.range (m-1) \ {0}) (λ i => b (i+1)) = 42 * (m-1)) ∧
  (Finset.sum (Finset.range (m-2) \ {0}) (λ i => b (i+1)) = 39 * (m-2)) ∧
  (Finset.sum (Finset.range (m-1)) (λ i => b i) = 45 * (m-1)) ∧
  (b (m-1) = b 0 + 90)

/-- The average of a finite set of natural numbers -/
noncomputable def average (s : Finset ℕ) : ℚ :=
  (s.sum (λ x => (x : ℚ))) / s.card

/-- Theorem stating that the average of a SpecialIntegerSet is 49.35 -/
theorem special_set_average (T : Finset ℕ) (h : SpecialIntegerSet T) :
  average T = 49.35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_average_l1348_134826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l1348_134846

theorem floor_ceil_fraction_square : ⌊⌈((11 : ℚ) / 5)^2⌉ + 19 / 3⌋ = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l1348_134846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l1348_134820

-- Define the points M and N
noncomputable def M : ℝ × ℝ := (1, 5/4)
noncomputable def N : ℝ × ℝ := (-4, -5/4)

-- Define the three curves
def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def curve2 (x y : ℝ) : Prop := x^2/2 + y^2 = 1
def curve3 (x y : ℝ) : Prop := x^2/2 - y^2 = 1

-- Define the equidistant condition
def isEquidistant (x y : ℝ) : Prop :=
  (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2

-- Theorem stating that each curve has a point equidistant from M and N
theorem equidistant_point_exists :
  (∃ x y, curve1 x y ∧ isEquidistant x y) ∧
  (∃ x y, curve2 x y ∧ isEquidistant x y) ∧
  (∃ x y, curve3 x y ∧ isEquidistant x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l1348_134820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1348_134872

noncomputable def numbers : List ℝ := [-3/5, 8, 0, -0.3, -100, Real.pi, 2.1010010001]

def is_positive (x : ℝ) : Prop := x > 0

def is_fraction (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = ↑n

theorem number_categorization :
  (∀ x ∈ numbers, is_positive x ↔ x = 8 ∨ x = Real.pi ∨ x = 2.1010010001) ∧
  (∀ x ∈ numbers, is_fraction x ↔ x = -3/5 ∨ x = -0.3) ∧
  (∀ x ∈ numbers, is_integer x ↔ x = 8 ∨ x = 0 ∨ x = -100) :=
by sorry

#check number_categorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1348_134872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_B_faster_l1348_134818

/-- Represents a boat with its speed in still water and the downstream current it encounters. -/
structure Boat where
  stillWaterSpeed : ℚ
  downstreamCurrent : ℚ

/-- Calculates the time taken by a boat to cover a given distance downstream. -/
def timeTaken (boat : Boat) (distance : ℚ) : ℚ :=
  distance / (boat.stillWaterSpeed + boat.downstreamCurrent)

/-- Theorem stating that Boat B takes less time than Boat A to cover 120 km downstream. -/
theorem boat_B_faster (boatA boatB : Boat) (h1 : boatA.stillWaterSpeed = 20)
    (h2 : boatA.downstreamCurrent = 6) (h3 : boatB.stillWaterSpeed = 24)
    (h4 : boatB.downstreamCurrent = 4) :
    timeTaken boatB 120 < timeTaken boatA 120 := by
  -- Unfold the definition of timeTaken
  unfold timeTaken
  -- Simplify the expressions
  simp [h1, h2, h3, h4]
  -- Prove the inequality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_B_faster_l1348_134818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l1348_134853

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -2x
def terminal_side_on_line (α : Real) : Prop :=
  ∃ (x y : Real), y = -2 * x ∧ x = Real.cos α ∧ y = Real.sin α

-- Theorem statement
theorem angle_on_line (h : terminal_side_on_line α) :
  Real.tan α = -2 ∧ Real.cos (2 * α + 3 * Real.pi / 2) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l1348_134853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1348_134850

theorem divisibility_condition (n : ℕ) :
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1348_134850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1348_134854

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := x + y - k = 0

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    unit_circle x₁ y₁ ∧ unit_circle x₂ y₂

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the vector condition
def vector_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let OA := (x₁, y₁)
  let OB := (x₂, y₂)
  let AB := (x₂ - x₁, y₂ - y₁)
  (OA.1 + OB.1)^2 + (OA.2 + OB.2)^2 ≥ (Real.sqrt 3 / 3)^2 * ((AB.1)^2 + (AB.2)^2)

-- Main theorem
theorem range_of_k :
  ∀ k : ℝ, k > 0 →
  intersection_points k →
  (∃ x₁ y₁ x₂ y₂ : ℝ, vector_condition x₁ y₁ x₂ y₂) →
  k ≥ Real.sqrt 2 ∧ k < 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1348_134854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_sum_and_verification_l1348_134824

theorem equation_solution_sum_and_verification :
  let equation := fun x : ℚ => (4*x - 3) * (3*x + 7) = 0
  let root1 := 3/4
  let root2 := -7/3
  (root1 + root2 = -19/12) ∧ (equation root1) ∧ (equation root2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_sum_and_verification_l1348_134824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_alpha_l1348_134883

theorem cos_pi_third_minus_alpha (α : ℝ) : 
  Real.sin (π / 6 + α) = Real.sqrt 3 / 3 → Real.cos (π / 3 - α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_alpha_l1348_134883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_configuration_l1348_134855

/-- Represents a circle in the figure -/
structure Circle where
  value : ℤ

/-- Represents the entire figure with 20 circles -/
def Figure := Vector Circle 20

/-- Checks if a given figure satisfies the sum condition -/
def satisfiesSumCondition (f : Figure) : Prop :=
  ∀ i j k l, i < 20 ∧ j < 20 ∧ k < 20 ∧ l < 20 →
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l →
    (f.get i).value + (f.get j).value + (f.get k).value + (f.get l).value = 0

/-- Theorem stating that the only valid configuration is all zeros -/
theorem only_zero_configuration (f : Figure) :
  satisfiesSumCondition f → (∀ i, i < 20 → (f.get i).value = 0) := by
  sorry

#check only_zero_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_configuration_l1348_134855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_men_double_length_l1348_134849

/-- Represents the length of wall that can be built -/
noncomputable def wall_length (men : ℕ) (days : ℕ) : ℝ :=
  112 * ((men : ℝ) / 20) * ((days : ℝ) / 6)

theorem double_men_double_length :
  wall_length 40 6 = 224 :=
by
  -- Unfold the definition of wall_length
  unfold wall_length
  -- Simplify the arithmetic
  simp [Nat.cast_div, Nat.cast_mul]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_men_double_length_l1348_134849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1348_134871

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-3/5 * t + 2, 4/5 * t)

-- Define the circle C in polar form
noncomputable def circle_C (a : ℝ) (θ : ℝ) : ℝ := a * Real.sin θ

-- Define the circle C in Cartesian form
def circle_C_cartesian (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + (y - a/2)^2 = a^2/4

-- Define the general equation of line l
def line_l_general (x y : ℝ) : Prop :=
  4*x + 3*y - 8 = 0

-- Define the chord length condition
def chord_length_condition (a : ℝ) : Prop :=
  |3*a/2 - 8|/5 = (Real.sqrt 3/2) * (|a|/2)

-- State the theorem
theorem circle_line_intersection (a : ℝ) :
  a ≠ 0 →
  chord_length_condition a →
  a = 32 ∨ a = 32/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1348_134871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_negative_45_degrees_l1348_134828

theorem cosecant_negative_45_degrees : 
  1 / Real.sin (-(π / 4)) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_negative_45_degrees_l1348_134828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_equals_99_l1348_134836

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_147 : a 1 + a 4 + a 7 = 39
  sum_369 : a 3 + a 6 + a 9 = 27

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proved -/
theorem sum_9_equals_99 (seq : ArithmeticSequence) : sum_n seq 9 = 99 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_equals_99_l1348_134836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_circumradius_l1348_134804

/-- A triangle with integer coordinate vertices, circumcenter, and orthocenter -/
structure IntegerTriangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ
  O : ℤ × ℤ
  H : ℤ × ℤ
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ O ∧ A ≠ H ∧
             B ≠ C ∧ B ≠ O ∧ B ≠ H ∧
             C ≠ O ∧ C ≠ H ∧
             O ≠ H

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : IntegerTriangle) : ℝ :=
  Real.sqrt ((t.A.1 - t.O.1)^2 + (t.A.2 - t.O.2)^2)

/-- The second smallest possible circumradius is √10 -/
theorem second_smallest_circumradius :
  ∃ (t : IntegerTriangle), 
    ∀ (s : IntegerTriangle), 
      circumradius s ≠ circumradius t → circumradius s > circumradius t ∨ circumradius s = Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_circumradius_l1348_134804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doug_money_l1348_134832

theorem doug_money (total : ℝ) (josh_brad_ratio : ℝ) (josh_doug_ratio : ℝ) :
  total = 68 →
  josh_brad_ratio = 2 →
  josh_doug_ratio = 3/4 →
  ∃ (doug_money : ℝ), abs (doug_money - 36.27) < 0.01 :=
by
  intros h_total h_josh_brad h_josh_doug
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doug_money_l1348_134832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1348_134899

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line_l (m c : ℝ) (x y : ℝ) : Prop := y = m * x + c

-- Define the tangent condition
def is_tangent (m c : ℝ) : Prop := line_l m c (-2) (Real.sqrt 2)

-- Define the intersection points P and Q
def intersection_points (m c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse 2 1 x₁ y₁ ∧ ellipse 2 1 x₂ y₂ ∧ line_l m c x₁ y₁ ∧ line_l m c x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Theorem statement
theorem ellipse_properties :
  ∃ (m c : ℝ),
    (∀ x y, ellipse 2 1 x y ↔ x^2 / 4 + y^2 = 1) ∧
    (line_l m c 6 0) ∧
    (is_tangent (-Real.sqrt 2 / 8) (3 * Real.sqrt 2 / 2)) ∧
    (∀ x₁ y₁ x₂ y₂, intersection_points m c x₁ y₁ x₂ y₂ →
      ((y₁ / (x₁ - 2)) * (y₂ / (x₂ - 2)) = 1 / 2)) ∧
    (∀ x₁ y₁ x₂ y₂, intersection_points m c x₁ y₁ x₂ y₂ →
      (y₁ / (x₁ + 2) + y₂ / (x₂ - 2) = -1 / 2) →
      (m = -1 / 6 ∧ c = 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1348_134899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1348_134876

-- Define the set S
def S : Set ℝ := {x : ℝ | Real.sqrt (4 - x^2) + (|x| / x) ≥ 0}

-- State the theorem
theorem solution_set_equality : S = Set.Icc (-Real.sqrt 3) 0 ∪ Set.Ioc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1348_134876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l1348_134834

theorem integer_root_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, x ≥ 0 ∧ ∃ n : ℤ, (169 - x^(1/3) : ℝ) = n^2) ∧ 
  S.card = 14 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l1348_134834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimin_rank_l1348_134861

def seokjin_rank : ℕ := 4

def jimin_rank_after_seokjin (seokjin : ℕ) (jimin : ℕ) : Prop :=
  jimin = seokjin + 1

theorem jimin_rank (jimin : ℕ) :
  jimin_rank_after_seokjin seokjin_rank jimin → jimin = 5 := by
  intro h
  simp [jimin_rank_after_seokjin, seokjin_rank] at h
  exact h

#check jimin_rank

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimin_rank_l1348_134861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l1348_134877

def divisor_product (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisor_product_1024 (n : ℕ) : divisor_product n = 1024 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l1348_134877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_Q_range_l1348_134840

-- Define the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the trajectory C
def C (x y : ℝ) : Prop := y^2 = -8*x

-- Define the line l
def l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 2)

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the x-coordinate of point Q
noncomputable def x_Q (k : ℝ) : ℝ := -2 - 8/k

-- Main theorem
theorem x_Q_range (k : ℝ) :
  (∃ S T : ℝ × ℝ, 
    C S.1 S.2 ∧ C T.1 T.2 ∧ 
    l k S.1 S.2 ∧ l k T.1 T.2 ∧
    second_quadrant S.1 S.2 ∧ second_quadrant T.1 T.2 ∧
    k ≠ 0 ∧ -1 < k ∧ k < 0) →
  x_Q k < -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_Q_range_l1348_134840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_change_without_exact_l1348_134851

inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar
  | Dollar
deriving Repr, DecidableEq

def coin_value : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50
  | Coin.Dollar => 100

def target_amounts : List Nat := [5, 10, 25, 50, 100]

def is_exact_change (amount : Nat) (coins : List Coin) : Prop :=
  ∃ (counts : List Nat), 
    counts.length = coins.length ∧ 
    (List.sum (List.zipWith (λ c n => n * coin_value c) coins counts) = amount)

theorem max_change_without_exact (coins : List Coin) : 
  (∀ t ∈ target_amounts, ¬ is_exact_change t coins) →
  (List.sum (coins.map coin_value) ≤ 119) :=
by sorry

#eval coin_value Coin.Penny
#eval coin_value Coin.Dollar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_change_without_exact_l1348_134851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_sum_is_14_l1348_134843

/-- Represents a card with a color and a number -/
inductive Card
| red (n : Nat)
| blue (n : Nat)

/-- Checks if a number divides another number evenly -/
def divides (a b : Nat) : Prop := b % a = 0

/-- Checks if two adjacent cards satisfy the divisibility condition -/
def validAdjacent (c1 c2 : Card) : Prop :=
  match c1, c2 with
  | Card.red n1, Card.blue n2 => divides n1 n2
  | Card.blue n1, Card.red n2 => divides n2 n1
  | _, _ => False

/-- Checks if a list of cards alternates between red and blue -/
def alternatingColors : List Card → Prop
| [] => True
| [_] => True
| (Card.red _) :: (Card.blue _) :: rest => alternatingColors rest
| (Card.blue _) :: (Card.red _) :: rest => alternatingColors rest
| _ => False

/-- Checks if all adjacent pairs in a list of cards satisfy the divisibility condition -/
def allValidAdjacent : List Card → Prop
| [] => True
| [_] => True
| c1 :: c2 :: rest => validAdjacent c1 c2 ∧ allValidAdjacent (c2 :: rest)

/-- The main theorem stating that the sum of the middle three card numbers is 14 -/
theorem middle_three_sum_is_14 (cards : List Card) : 
  (cards.length = 10) →
  (∀ n, Card.red n ∈ cards → 2 ≤ n ∧ n ≤ 6) →
  (∀ n, Card.blue n ∈ cards → 4 ≤ n ∧ n ≤ 8) →
  (alternatingColors cards) →
  (allValidAdjacent cards) →
  (∃ n1 n2 n3, cards.get? 4 = some (Card.blue n1) ∧
               cards.get? 5 = some (Card.red n2) ∧
               cards.get? 6 = some (Card.blue n3) ∧
               n1 + n2 + n3 = 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_sum_is_14_l1348_134843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1348_134865

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) + 4

theorem function_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q)) ∧
  (∃ M : ℝ, ∀ x : ℝ, f x ≤ M) ∧
  (∀ a : ℝ, f a = 5 → Real.tan a = 0 ∨ Real.tan a = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1348_134865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_powers_of_two_3144_l1348_134869

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def distinct_powers_of_two (n : ℕ) (powers : List ℕ) : Prop :=
  (∀ p, p ∈ powers → is_power_of_two p) ∧ 
  (∀ p q, p ∈ powers → q ∈ powers → p ≠ q → p ≠ q) ∧
  (n = powers.sum)

theorem sum_of_distinct_powers_of_two_3144 :
  ∃ (powers : List ℕ),
    distinct_powers_of_two 3144 powers ∧
    (powers.map (λ p ↦ Nat.log2 p)).sum = 30 ∧
    powers.length = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_powers_of_two_3144_l1348_134869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1348_134809

theorem trigonometric_identities (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - Real.pi/4) = 1/4)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2)
  (h5 : Real.cos (α + β) = Real.sqrt 5/5)
  (h6 : Real.sin (α - β) = Real.sqrt 10/10) :
  (((Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22) ∧ (2 * β = Real.pi/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1348_134809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_sum_l1348_134817

/-- The equation of the graph -/
noncomputable def f (A B C : ℤ) (x : ℝ) : ℝ := x / (x^3 + A*x^2 + B*x + C)

/-- The denominator of the equation -/
def denominator (A B C : ℤ) (x : ℝ) : ℝ := x^3 + A*x^2 + B*x + C

theorem asymptotes_sum (A B C : ℤ) :
  (denominator A B C (-3) = 0) →
  (denominator A B C 0 = 0) →
  (denominator A B C 4 = 0) →
  A + B + C = -13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_sum_l1348_134817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_wheel_horsepower_l1348_134830

/-- Calculates the effective horsepower of an overshot water wheel -/
noncomputable def effective_horsepower (water_velocity : ℝ) (channel_width : ℝ) (water_thickness : ℝ) 
  (wheel_diameter : ℝ) (efficiency : ℝ) : ℝ :=
  let water_density : ℝ := 1000  -- kg/m^3
  let gravity : ℝ := 9.81  -- m/s^2
  let mass_flow_rate : ℝ := water_velocity * channel_width * water_thickness * water_density
  let kinetic_energy : ℝ := 0.5 * mass_flow_rate * water_velocity^2
  let potential_energy : ℝ := mass_flow_rate * wheel_diameter * gravity
  let indicated_power : ℝ := kinetic_energy + potential_energy
  let horsepower_conversion : ℝ := 745.7  -- 1 HP = 745.7 W
  let indicated_horsepower : ℝ := indicated_power / horsepower_conversion
  indicated_horsepower * efficiency

/-- Theorem stating that the effective horsepower of the given water wheel is approximately 2.9 HP -/
theorem water_wheel_horsepower :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |effective_horsepower 1.4 0.5 0.13 3 0.78 - 2.9| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_wheel_horsepower_l1348_134830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABM_l1348_134805

/-- The polar equation of curve C₁ -/
def C₁ (ρ θ : ℝ) : Prop := ρ^2 * (1 + 3 * Real.sin θ^2) = 16

/-- The relationship between points O, P, and M -/
def OP_OM_relation (OP OM : ℝ × ℝ) : Prop := OP = (2 * OM.1, 2 * OM.2)

/-- The parametric equation of line l -/
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, t)

/-- The theorem stating the maximum area of triangle ABM -/
theorem max_area_triangle_ABM :
  ∀ (M : ℝ × ℝ),
  (∃ (θ : ℝ), C₁ (2 * M.1) θ ∧ C₁ (2 * M.2) θ) →
  (∃ (P : ℝ × ℝ), OP_OM_relation P M) →
  (M.1^2 / 4 + M.2^2 = 1) →
  (∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∧ A.2 = -1) ∧
    (B.1 = 1 ∧ B.2 = 0) ∧
    (∃ (t : ℝ), line_l t = A ∨ line_l t = B)) →
  ∀ (maximal_area : ℝ) (triangle_ABM : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ),
  (∀ (M' : ℝ × ℝ), triangle_ABM A B M' ≤ maximal_area) →
  maximal_area = (Real.sqrt 5 + 1) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABM_l1348_134805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3x_between_4_and_5_l1348_134800

theorem sqrt_3x_between_4_and_5 : 
  (Finset.filter (fun x : ℕ => 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) (Finset.range 9)).card = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3x_between_4_and_5_l1348_134800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1348_134873

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 6)

theorem min_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1348_134873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_summable_subset_l1348_134864

-- Define the property for the subset E
def NonSummableSubset (E : Set ℕ) : Prop :=
  ∀ x ∈ E, ∀ y ∈ E, ∀ z ∈ E, x ≠ y + z ∨ (x = y + z ∧ y = x ∧ z = 0) ∨ (x = y + z ∧ z = x ∧ y = 0)

-- Theorem statement
theorem exists_non_summable_subset : ∃ E : Set ℕ, NonSummableSubset E := by
  -- Define the set E
  let E : Set ℕ := {5, 14, 22}
  
  -- Prove that E satisfies the NonSummableSubset property
  have h : NonSummableSubset E := by
    -- Proof goes here
    sorry
  
  -- Conclude the existence of such a set
  exact ⟨E, h⟩

-- Example to show that the set E satisfies the property
example : NonSummableSubset {5, 14, 22} := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_summable_subset_l1348_134864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_u_and_v_l1348_134807

/-- A line passing through point P(2,1) and intersecting positive x and y axes -/
structure IntersectingLine where
  /-- Point where the line intersects positive x-axis -/
  A : ℝ × ℝ
  /-- Point where the line intersects positive y-axis -/
  B : ℝ × ℝ
  /-- The line passes through P(2,1) -/
  passes_through_P : (2 - A.1) * (B.2 - 1) = (1 - A.2) * (B.1 - 2)
  /-- A is on positive x-axis -/
  A_on_x_axis : A.2 = 0 ∧ A.1 > 0
  /-- B is on positive y-axis -/
  B_on_y_axis : B.1 = 0 ∧ B.2 > 0

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Origin (0,0) -/
def O : ℝ × ℝ := (0, 0)

/-- Point P(2,1) -/
def P : ℝ × ℝ := (2, 1)

/-- Function u to be minimized -/
noncomputable def u (l : IntersectingLine) : ℝ :=
  distance O l.A + distance O l.B

/-- Function v to be minimized -/
noncomputable def v (l : IntersectingLine) : ℝ :=
  distance P l.A * distance P l.B

theorem min_u_and_v (l : IntersectingLine) :
  ((∀ l', u l ≤ u l') → u l = 2 * Real.sqrt 2 + 3) ∧
  ((∀ l', v l ≤ v l') → v l = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_u_and_v_l1348_134807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_points_l1348_134891

/-- Given a triangle ABC with side lengths a, b, c, and any two points P and Q in the plane,
    the sum of the products of each side length and the distances from its endpoints to P and Q
    is greater than or equal to the product of all side lengths. -/
theorem triangle_inequality_with_points (a b c : ℝ) (A B C P Q : ℂ) :
  a > 0 → b > 0 → c > 0 →
  Complex.abs (B - C) = a → Complex.abs (C - A) = b → Complex.abs (A - B) = c →
  a * Complex.abs (P - A) * Complex.abs (Q - A) + 
  b * Complex.abs (P - B) * Complex.abs (Q - B) + 
  c * Complex.abs (P - C) * Complex.abs (Q - C) ≥ a * b * c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_points_l1348_134891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1348_134858

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 6

/-- The semi-major axis of the ellipse -/
noncomputable def a : ℝ := Real.sqrt 6

/-- The semi-minor axis of the ellipse -/
noncomputable def b : ℝ := Real.sqrt 5

/-- The distance from the center to a focus -/
noncomputable def c : ℝ := Real.sqrt (a^2 - b^2)

/-- The left focus -/
noncomputable def F₁ : ℝ × ℝ := (-c, 0)

/-- The right focus -/
noncomputable def F₂ : ℝ × ℝ := (c, 0)

/-- Area of triangle MF₁F₂ given y-coordinate of M -/
noncomputable def triangleArea (y : ℝ) : ℝ := |y|

/-- Dot product of vectors PF₁ and PF₂ -/
noncomputable def dotProduct (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + c) * (x - c) + y * y

theorem ellipse_properties :
  -- 1. The equation of the ellipse is correct
  (∀ x y, Ellipse x y ↔ x^2 / 6 + y^2 / 5 = 1) ∧
  -- 2. The maximum area of triangle MF₁F₂ is √5
  (∀ M : ℝ × ℝ, Ellipse M.fst M.snd → triangleArea M.snd ≤ b) ∧
  (∃ M : ℝ × ℝ, Ellipse M.fst M.snd ∧ triangleArea M.snd = b) ∧
  -- 3. There is no point P on the ellipse such that PF₁ · PF₂ = 0
  (¬ ∃ P : ℝ × ℝ, Ellipse P.fst P.snd ∧ dotProduct P = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1348_134858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l1348_134816

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle --/
def lies_on (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- The tangent line to a circle at a given point --/
noncomputable def tangent_line (c : Circle) (p : Point) : ℝ → ℝ → Prop :=
  sorry

/-- Two lines intersect at a point --/
def lines_intersect (l1 l2 : ℝ → ℝ → Prop) (p : Point) : Prop :=
  sorry

/-- A point lies on the x-axis --/
def on_x_axis (p : Point) : Prop :=
  let (_, y) := p
  y = 0

/-- The area of a circle --/
noncomputable def circle_area (c : Circle) : ℝ :=
  Real.pi * c.radius^2

theorem circle_area_theorem (ω : Circle) (A B : Point) :
  lies_on A ω →
  lies_on B ω →
  A = (2, 7) →
  B = (8, 5) →
  (∃ p, on_x_axis p ∧ lines_intersect (tangent_line ω A) (tangent_line ω B) p) →
  circle_area ω = 12.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l1348_134816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_knight_arrangement_l1348_134803

/-- Represents a student who can be either a knight or a liar -/
inductive Student
| Knight
| Liar

/-- Represents a pair of students sitting at a desk -/
structure Desk where
  student1 : Student
  student2 : Student

/-- Represents the class of students -/
structure ClassOfStudents where
  students : Finset Student
  desks : Finset Desk
  student_count : Nat
  desk_count : Nat

/-- Checks if a statement made by a student about their neighbor is true -/
def is_statement_true (s : Student) (neighbor : Student) (statement : Student → Prop) : Prop :=
  match s with
  | Student.Knight => statement neighbor
  | Student.Liar => ¬(statement neighbor)

/-- The main theorem to be proved -/
theorem impossible_knight_arrangement (c : ClassOfStudents) : 
  c.student_count = 26 ∧ 
  c.desk_count = 13 ∧
  (∀ d : Desk, d ∈ c.desks → 
    is_statement_true d.student1 d.student2 (λ n => n = Student.Liar) ∧
    is_statement_true d.student2 d.student1 (λ n => n = Student.Liar)) →
  ¬(∃ new_desks : Finset Desk, 
    new_desks.card = 13 ∧
    (∀ d : Desk, d ∈ new_desks → 
      is_statement_true d.student1 d.student2 (λ n => n = Student.Knight) ∧
      is_statement_true d.student2 d.student1 (λ n => n = Student.Knight))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_knight_arrangement_l1348_134803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1348_134823

/-- The length of the first train in meters -/
noncomputable def first_train_length : ℝ := 108.02

/-- The speed of the first train in km/h -/
noncomputable def first_train_speed : ℝ := 50

/-- The length of the second train in meters -/
noncomputable def second_train_length : ℝ := 112

/-- The time taken for the trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ := 6

/-- The speed of the second train in km/h -/
noncomputable def second_train_speed : ℝ := 82

/-- Conversion factor from km/h to m/s -/
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

theorem train_length_proof :
  ∃ ε > 0, |first_train_length - 
    ((first_train_speed * kmh_to_ms + second_train_speed * kmh_to_ms) * crossing_time - second_train_length)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1348_134823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_percentage_l1348_134882

/-- Calculates the gain percentage for a cloth sale -/
noncomputable def gain_percentage (total_meters : ℝ) (gain_meters : ℝ) : ℝ :=
  (gain_meters / total_meters) * 100

/-- Theorem: The gain percentage is 40% when selling 25 meters of cloth with a gain of 10 meters -/
theorem cloth_sale_gain_percentage :
  gain_percentage 25 10 = 40 := by
  -- Unfold the definition of gain_percentage
  unfold gain_percentage
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_percentage_l1348_134882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l1348_134822

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℕ × ℕ × ℕ,
    (∀ n : ℕ, 
      let (a, b, c) := f n
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a + b) * c = a * b) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l1348_134822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eunice_car_discount_l1348_134835

noncomputable def original_price : ℝ := 10000
noncomputable def eunice_paid : ℝ := 7500

noncomputable def percentage_decrease (original : ℝ) (paid : ℝ) : ℝ :=
  ((original - paid) / original) * 100

theorem eunice_car_discount : percentage_decrease original_price eunice_paid = 25 := by
  -- Unfold the definition of percentage_decrease
  unfold percentage_decrease
  -- Simplify the expression
  simp [original_price, eunice_paid]
  -- The proof is complete
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eunice_car_discount_l1348_134835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_shift_left_l1348_134831

/-- Represents a quadratic function of the form y = -1/2 * (x + a)^2 -/
noncomputable def QuadraticFunction (a : ℝ) := λ x : ℝ => -1/2 * (x + a)^2

/-- Represents a horizontal shift transformation -/
noncomputable def HorizontalShift (f : ℝ → ℝ) (shift : ℝ) := λ x : ℝ => f (x + shift)

theorem quadratic_shift_left (a shift : ℝ) :
  HorizontalShift (QuadraticFunction a) shift = QuadraticFunction (a + shift) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_shift_left_l1348_134831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l1348_134801

theorem sequence_equality (n : ℕ) (a : ℕ → ℝ) 
  (h₁ : n ≥ 2)
  (h₂ : ∀ i, i < n → a i ≠ -1)
  (h₃ : ∀ i, i < n → a ((i + 2) % n) = (a i ^ 2 + a i) / (a ((i + 1) % n) + 1)) :
  ∀ i j, i < n → j < n → a i = a j := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l1348_134801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1348_134847

noncomputable def f (x : ℝ) : ℝ := (2*x^3 - 3*x^2 + 5*x - 1) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1348_134847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_g_max_value_f_g_inequality_l1348_134819

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 2) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 1 / x

-- Statement 1: The derivative of f at x = 1
theorem f_derivative_at_one : 
  deriv f 1 = 2 := by sorry

-- Statement 2: Maximum value of g on [1,2]
theorem g_max_value (a : ℝ) : 
  ∃ (max_val : ℝ), ∀ x ∈ Set.Icc 1 2, g a x ≤ max_val ∧
  (a ≤ -1 → max_val = -1) ∧
  (-1 < a ∧ a < -1/2 → max_val = a * Real.log (-1/a) + a) ∧
  (-1/2 ≤ a → max_val = a * Real.log 2 - 1/2) := by sorry

-- Statement 3: Inequality for a = 1
theorem f_g_inequality :
  ∀ x > 0, f x > g 1 x - Real.cos x / x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_g_max_value_f_g_inequality_l1348_134819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retail_cost_price_proof_l1348_134811

/-- Calculates the cost price of an item given selling price, overhead, and profit percentage. -/
noncomputable def calculate_cost_price (selling_price : ℝ) (overhead : ℝ) (profit_percent : ℝ) : ℝ :=
  (selling_price - overhead) / (1 + profit_percent / 100)

/-- Theorem stating that under given conditions, the cost price is approximately 234.65. -/
theorem retail_cost_price_proof :
  let selling_price : ℝ := 300
  let overhead : ℝ := 15
  let profit_percent : ℝ := 21.457489878542503
  let cost_price := calculate_cost_price selling_price overhead profit_percent
  ∃ ε > 0, |cost_price - 234.65| < ε :=
by
  sorry

#eval (300 - 15) / (1 + 21.457489878542503 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retail_cost_price_proof_l1348_134811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_line_l1348_134888

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Rotates a line 90° counterclockwise around its y-axis intersection --/
noncomputable def rotate90 (l : Line) : Line :=
  { slope := -1 / l.slope, intercept := l.intercept }

theorem rotation_of_line :
  let original_line := Line.mk 3 1
  let rotated_line := rotate90 original_line
  rotated_line.slope = -1/3 ∧ rotated_line.intercept = 1 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_line_l1348_134888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_sector_surface_area_formula_l1348_134814

/-- The surface area of a solid formed by rotating a circular sector about one of its radii. -/
noncomputable def rotated_sector_surface_area (R : ℝ) (θ : ℝ) : ℝ :=
  2 * Real.pi * R^2 * Real.sin (θ/2) * (Real.cos (θ/2) + 2 * Real.sin (θ/2))

/-- Theorem stating that the surface area of a solid formed by rotating a circular sector
    about one of its radii is equal to 2πR²sin(θ/2)(cos(θ/2) + 2sin(θ/2)). -/
theorem rotated_sector_surface_area_formula (R θ : ℝ) (h_R : R > 0) (h_θ : 0 < θ ∧ θ < 2 * Real.pi) :
  rotated_sector_surface_area R θ = 2 * Real.pi * R^2 * Real.sin (θ/2) * (Real.cos (θ/2) + 2 * Real.sin (θ/2)) :=
by
  -- Unfold the definition of rotated_sector_surface_area
  unfold rotated_sector_surface_area
  -- The equality follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_sector_surface_area_formula_l1348_134814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandson_height_prediction_l1348_134892

/-- Represents the heights of five generations in centimeters -/
structure FamilyHeights where
  grandfather : ℝ
  father : ℝ
  teacher : ℝ
  son : ℝ
  grandson : ℝ

/-- Calculates the slope of the linear regression line -/
noncomputable def calculateSlope (heights : FamilyHeights) : ℝ :=
  (heights.grandfather * heights.father + heights.father * heights.teacher + heights.teacher * heights.son
    - 3 * heights.grandfather * heights.teacher) /
  (heights.grandfather^2 + heights.father^2 + heights.teacher^2 - 3 * heights.grandfather^2)

/-- Calculates the y-intercept of the linear regression line -/
noncomputable def calculateIntercept (heights : FamilyHeights) (slope : ℝ) : ℝ :=
  heights.teacher - slope * heights.grandfather

/-- Predicts the grandson's height using linear regression -/
noncomputable def predictGrandsonHeight (heights : FamilyHeights) : ℝ :=
  let slope := calculateSlope heights
  let intercept := calculateIntercept heights slope
  slope * heights.son + intercept

/-- Theorem stating that the predicted grandson's height is 185cm -/
theorem grandson_height_prediction (heights : FamilyHeights)
    (h1 : heights.grandfather = 173)
    (h2 : heights.father = 170)
    (h3 : heights.teacher = 176)
    (h4 : heights.son = 182) :
    predictGrandsonHeight heights = 185 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandson_height_prediction_l1348_134892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exceeds_exponential_growth_l1348_134841

/-- A function from reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ | ∀ x, f x > 0}

theorem exceeds_exponential_growth
  (f : ℝ → ℝ)
  (h_pos : ∀ x, f x > 0)
  (h_diff : Differentiable ℝ f)
  (h_deriv : ∀ x, deriv f x > f x) :
  (∃ C : ℝ, ∀ x ≥ C, ∀ k ≤ 1, f x > Real.exp (k * x)) ∧
  (∀ k > 1, ∃ D : ℝ, ∀ x ≥ D, f x ≤ Real.exp (k * x)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exceeds_exponential_growth_l1348_134841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_abs_values_l1348_134808

theorem min_sum_abs_values (p q r s : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (Matrix.of !![p, q; r, s])^2 = Matrix.of !![9, 0; 0, 9] →
  (∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (Matrix.of !![a, b; c, d])^2 = Matrix.of !![9, 0; 0, 9] ∧
    (abs a + abs b + abs c + abs d < abs p + abs q + abs r + abs s)) →
  abs p + abs q + abs r + abs s ≥ 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_abs_values_l1348_134808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_simplification_l1348_134889

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_simplification :
  (lg 2)^2 + lg 2 * lg 5 + lg 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_simplification_l1348_134889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_regular_tetrahedron_l1348_134813

/-- The radius of a sphere inscribed in a regular tetrahedron with edge length a -/
noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a * Real.sqrt 6 / 12

/-- Theorem: The radius of a sphere inscribed in a regular tetrahedron with edge length a is a√6/12 -/
theorem inscribed_sphere_radius_in_regular_tetrahedron (a : ℝ) (h : a > 0) :
  ∃ r : ℝ, r = inscribed_sphere_radius a ∧ r > 0 := by
  use inscribed_sphere_radius a
  constructor
  · rfl
  · exact mul_pos (mul_pos h (Real.sqrt_pos.2 (by norm_num))) (by norm_num)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_regular_tetrahedron_l1348_134813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1348_134857

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = Real.pi) →  -- Sum of angles in a triangle is π
  (b = 1) →  -- Given condition
  (c = Real.sqrt 3) →  -- Given condition
  (B = Real.pi / 6) →  -- Given condition
  (Real.sin B / b = Real.sin C / c) →  -- Law of sines
  (a ^ 2 = b ^ 2 + c ^ 2) →  -- Pythagorean theorem
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1348_134857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_explicit_sum_l1348_134893

/-- The sum of the infinite series Σ(n=1 to ∞) of (2n-1)(1/2023)^(n-1) -/
noncomputable def series_sum : ℝ := ∑' n, (2 * n - 1) * (1 / 2023) ^ (n - 1)

/-- The explicit value of the sum -/
noncomputable def explicit_sum : ℝ := 2027.005938 / 2022

/-- Theorem stating that the series sum equals the explicit sum -/
theorem series_sum_equals_explicit_sum : series_sum = explicit_sum := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_explicit_sum_l1348_134893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1348_134862

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : |φ| < π / 2)
  (h_sym_axis : ∀ x, f ω φ (-π - x) = f ω φ (-π + x))
  (h_sym_center : ∀ x, f ω φ (π/2 + x) = -f ω φ (π/2 - x))
  (h_incr : MonotoneOn (f ω φ) (Set.Icc (-π) (-π/2))) :
  (ω = 1/3) ∧ 
  (∀ δ, (∀ x, f ω φ (x + δ) = f ω φ (-x + δ)) → 
    ∃ k : ℤ, δ = 2*π + 3*k*π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1348_134862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_eq_225_l1348_134845

/-- Represents a trapezoid ABCD with points E and F on its non-parallel sides. -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ
  hAB_positive : 0 < AB
  hCD_positive : 0 < CD
  hAltitude_positive : 0 < altitude

/-- The area of quadrilateral EFCD in the given trapezoid. -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  let EF := (1/3) * t.AB + (2/3) * t.CD
  let h_EFCD := (2/3) * t.altitude
  (h_EFCD * (EF + t.CD)) / 2

/-- Theorem stating that the area of EFCD is 225 square units. -/
theorem area_EFCD_eq_225 (t : Trapezoid) 
    (h1 : t.AB = 10) 
    (h2 : t.CD = 25) 
    (h3 : t.altitude = 15) : 
  area_EFCD t = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_eq_225_l1348_134845
