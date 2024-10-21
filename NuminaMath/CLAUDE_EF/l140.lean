import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_squared_l140_14009

theorem sin_plus_cos_squared (θ : Real) (b : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) -- θ is acute
  (h2 : Real.cos (2 * θ) = b) : 
  (Real.sin θ + Real.cos θ)^2 = 2 - b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_squared_l140_14009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l140_14062

theorem sequence_bound (a : ℕ → ℝ) (N : ℕ) 
  (h1 : ∀ n ≥ N, a n = 1)
  (h2 : ∀ n ≥ 2, a n ≤ a (n-1) + 2^(-(n : ℝ)) * a (2*n)) :
  ∀ k : ℕ, a k > 1 - 2^(-(k : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l140_14062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_play_win_probability_best_play_always_wins_multiple_plays_l140_14061

/-- The probability of the best option winning in a voting scenario -/
theorem best_play_win_probability (n : ℕ) : 
  (1 - (n.factorial^2 : ℚ) / (2*n).factorial : ℚ) = 
  ((2*n).factorial - n.factorial^2 : ℚ) / (2*n).factorial := by
  sorry

/-- The probability of the best play winning when there are more than two plays -/
theorem best_play_always_wins_multiple_plays (n m : ℕ) (h : m > 2) :
  (1 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_play_win_probability_best_play_always_wins_multiple_plays_l140_14061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circleplus_property_l140_14066

-- Define the ⊕ operation
noncomputable def circleplus (x y : ℝ) : ℝ := 1 / (x * y)

-- State the theorem
theorem circleplus_property :
  ∀ a : ℝ, circleplus 4 (circleplus a 1540) = 385 := by
  intro a
  -- Unfold the definition of circleplus
  unfold circleplus
  -- Simplify the expression
  simp [mul_inv_cancel, inv_mul_cancel]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circleplus_property_l140_14066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_15_l140_14002

/-- An arithmetic sequence with a₈ = 8 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 8 = 8

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sum_15 (a : ℕ → ℚ) :
  arithmetic_sequence a → arithmetic_sum a 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_15_l140_14002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_altitude_perpendicular_l140_14048

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def Quadrilateral (A B C D : Point) := True

def convex (q : Quadrilateral A B C D) := True

noncomputable def intersection_of_diagonals (A B C D : Point) : Point := sorry

noncomputable def median_intersection (A B C : Point) : Point := sorry

noncomputable def altitude_intersection (A B C : Point) : Point := sorry

noncomputable def line_through (P Q : Point) : Set Point := sorry

def perpendicular (l1 l2 : Set Point) : Prop := sorry

-- State the theorem
theorem median_altitude_perpendicular 
  (A B C D O : Point) 
  (q : Quadrilateral A B C D) 
  (h_convex : convex q)
  (h_O : O = intersection_of_diagonals A B C D) :
  let K := median_intersection A O B
  let L := median_intersection C O D
  let M := altitude_intersection B O C
  let N := altitude_intersection A O D
  perpendicular (line_through K L) (line_through M N) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_altitude_perpendicular_l140_14048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_piece_at_least_quarter_l140_14007

/-- A point inside a square -/
structure PointInSquare where
  x : ℝ
  y : ℝ
  h_x : 0 < x ∧ x < 1
  h_y : 0 < y ∧ y < 1

/-- The area of a piece of the square after two perpendicular cuts -/
def piece_area (p : PointInSquare) : ℝ × ℝ × ℝ × ℝ :=
  (p.x * p.y, p.x * (1 - p.y), (1 - p.x) * p.y, (1 - p.x) * (1 - p.y))

/-- The smallest piece is at least one-quarter of the total area -/
theorem smallest_piece_at_least_quarter (p : PointInSquare) :
  min (min (min (piece_area p).1 (piece_area p).2.1) (piece_area p).2.2.1) (piece_area p).2.2.2 ≥ 1/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_piece_at_least_quarter_l140_14007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l140_14026

/-- The focus of a parabola y² = 2px passing through (1, √3) is at (3/4, 0) -/
theorem parabola_focus (p : ℝ) : 
  (∃ (y : ℝ), y^2 = 2*p*1 ∧ y^2 = 3) →  -- parabola passes through (1, √3)
  (∀ (x y : ℝ), y^2 = 2*p*x) →         -- general equation of the parabola
  (p/2, 0) = (3/4, 0) :=                -- focus of the parabola
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l140_14026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_when_k_10_first_player_wins_when_k_15_l140_14054

/-- Represents a player in the game -/
inductive Player : Type
| First : Player
| Second : Player

/-- Represents a digit in the game -/
def Digit : Type := Fin 5

/-- Represents the game state -/
structure GameState :=
  (digits : List Digit)
  (currentPlayer : Player)

/-- Checks if a list of digits sums to a multiple of 9 -/
def isSumDivisibleBy9 (digits : List Digit) : Prop :=
  (digits.map Fin.val).sum % 9 = 0

/-- Theorem for the case when k = 10 -/
theorem second_player_wins_when_k_10 :
  ∃ (strategy : GameState → Digit),
    ∀ (game : List Digit),
      game.length = 20 →
      isSumDivisibleBy9 (game ++ [strategy (GameState.mk game Player.Second)]) := by
  sorry

/-- Theorem for the case when k = 15 -/
theorem first_player_wins_when_k_15 :
  ∃ (strategy : GameState → Digit),
    ∀ (game : List Digit),
      game.length < 30 →
      ∃ (finalGame : List Digit),
        finalGame.length = 30 ∧
        (∀ i, i % 2 = 0 → i < finalGame.length → 
          finalGame.get ⟨i, by sorry⟩ = strategy (GameState.mk (finalGame.take i) Player.First)) ∧
        isSumDivisibleBy9 finalGame := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_when_k_10_first_player_wins_when_k_15_l140_14054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maurice_late_prob_converges_maurice_late_prob_467_l140_14025

/-- Probability of Maurice being late for work -/
noncomputable def prob_late (p : ℝ) : ℝ := 1/2 * p + 1/4 * (1 - p)

/-- Recurrence relation for the probability of driving a car -/
noncomputable def next_prob (p : ℝ) : ℝ := 1/4 * (p + 1)

/-- Theorem: The probability of Maurice being late converges to 2/3 -/
theorem maurice_late_prob_converges :
  ∃ (p : ℝ), p = next_prob p ∧ prob_late p = 2/3 := by
  sorry

/-- Corollary: The probability of Maurice being late on the 467th trip is approximately 2/3 -/
theorem maurice_late_prob_467 :
  ∃ (p : ℝ), p = next_prob p ∧ prob_late p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maurice_late_prob_converges_maurice_late_prob_467_l140_14025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_calculation_l140_14052

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * (Int.floor (x / y))

-- State the theorem
theorem rem_calculation : rem ((5 : ℚ) / 7 + (1 : ℚ) / 3) (-(3 : ℚ) / 4) = -(19 : ℚ) / 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_calculation_l140_14052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l140_14006

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * Real.arcsin x + 3

-- State the theorem
theorem min_value_of_f (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : ∃ x, ∀ y, f a b y ≤ f a b x ∧ f a b x = 10) : 
  ∃ x, ∀ y, f a b x ≤ f a b y ∧ f a b x = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l140_14006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_18_divisors_l140_14064

/-- The number of positive integer divisors of 180^180 that are divisible by exactly 18 positive integers -/
noncomputable def divisors_with_18_divisors : ℕ :=
  let base := 180
  let exponent := 180
  let n := base ^ exponent
  (Finset.filter (fun d => (Nat.divisors d).card = 18) (Nat.divisors n)).card

/-- The main theorem stating that there are 24 such divisors -/
theorem count_divisors_with_18_divisors : divisors_with_18_divisors = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_18_divisors_l140_14064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inflation_and_interest_rates_l140_14047

/-- Calculates the total inflation over two years given an annual inflation rate -/
noncomputable def total_inflation (annual_rate : ℝ) : ℝ :=
  ((1 + annual_rate) ^ 2 - 1) * 100

/-- Calculates the real interest rate of a bank deposit over two years
    given the nominal interest rate and the total inflation rate -/
noncomputable def real_interest_rate (nominal_rate : ℝ) (total_inflation : ℝ) : ℝ :=
  ((1 + nominal_rate) ^ 2 / (1 + total_inflation / 100) - 1) * 100

theorem inflation_and_interest_rates :
  let annual_inflation_rate := 0.025
  let nominal_interest_rate := 0.06
  let calculated_total_inflation := total_inflation annual_inflation_rate
  let calculated_real_interest_rate := real_interest_rate nominal_interest_rate calculated_total_inflation
  calculated_total_inflation = 5.0625 ∧
  abs (calculated_real_interest_rate - 6.95) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inflation_and_interest_rates_l140_14047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_conversion_l140_14035

/-- Converts CAD to USD given the exchange rate -/
noncomputable def cad_to_usd (cad : ℝ) (exchange_rate : ℝ) : ℝ :=
  cad / exchange_rate

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem book_cost_conversion :
  let book_cost_cad : ℝ := 30
  let exchange_rate : ℝ := 1.25
  round_to_hundredth (cad_to_usd book_cost_cad exchange_rate) = 24.00 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_conversion_l140_14035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_when_a_2_max_chord_length_when_m_2_m_range_when_tangent_below_center_l140_14021

-- Define the circle and line
def circle_eq (a x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 6*a*y + 10*a^2 - 4*a = 0

def line_eq (m x y : ℝ) : Prop :=
  y = x + m

-- Define the center and radius of the circle
def center (a : ℝ) : ℝ × ℝ :=
  (a, 3*a)

noncomputable def radius (a : ℝ) : ℝ :=
  2 * Real.sqrt a

-- Theorem for part (1)
theorem max_chord_length_when_a_2 (a : ℝ) :
  a = 2 → ∃ (max_length : ℝ), max_length = 4 * Real.sqrt 2 ∧
  ∀ (x y : ℝ) (m : ℝ), circle_eq a x y → line_eq m x y →
  ∃ (chord_length : ℝ), chord_length ≤ max_length :=
by sorry

-- Theorem for part (2)
theorem max_chord_length_when_m_2 (a : ℝ) :
  0 < a ∧ a ≤ 4 →
  ∃ (max_length : ℝ), max_length = 2 * Real.sqrt 6 ∧
  ∀ (x y : ℝ), circle_eq a x y → line_eq 2 x y →
  ∃ (chord_length : ℝ), chord_length ≤ max_length :=
by sorry

-- Theorem for part (3)
theorem m_range_when_tangent_below_center (a : ℝ) :
  0 < a ∧ a ≤ 4 →
  ∃ (m : ℝ), -1 ≤ m ∧ m ≤ 8 - 4 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), circle_eq a x y → line_eq m x y →
  ∃ (t : ℝ), x = t ∧ y = t + m ∧
  (∀ (x' y' : ℝ), x' ≠ x ∨ y' ≠ y → circle_eq a x' y' → ¬ line_eq m x' y') ∧
  y < (center a).2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_when_a_2_max_chord_length_when_m_2_m_range_when_tangent_below_center_l140_14021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_area_formula_l140_14063

/-- The area of a circular segment with perimeter p and arc angle 120° -/
noncomputable def segmentArea (p : ℝ) : ℝ :=
  (3 * p^2 * (4 * Real.pi - 3 * Real.sqrt 3)) / (4 * (2 * Real.pi + 3 * Real.sqrt 3)^2)

/-- Theorem: The area of a circular segment with perimeter p and arc angle 120° 
    is equal to (3p^2 * (4π - 3√3)) / (4 * (2π + 3√3)^2) -/
theorem segment_area_formula (p : ℝ) (h : p > 0) : 
  let R := (3 * p) / (2 * Real.pi + 3 * Real.sqrt 3)
  let arcLength := (2 * Real.pi * R) / 3
  let chordLength := R * Real.sqrt 3
  let sectorArea := (Real.pi * R^2) / 3
  let triangleArea := (R^2 * Real.sqrt 3) / 4
  arcLength + chordLength = p ∧ 
  sectorArea - triangleArea = segmentArea p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_area_formula_l140_14063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_3_of_2_power_le_1_l140_14019

theorem log_base_3_of_2_power_le_1 :
  ∀ x : ℝ, x ≥ 0 → (Real.log 2 / Real.log 3) ^ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_3_of_2_power_le_1_l140_14019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l140_14096

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.sin x - 1/2

theorem min_value_of_f :
  ∃ (m : ℝ), m = 1 ∧ ∀ x ∈ Set.Icc (π/6) (5*π/6), f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l140_14096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l140_14033

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def all_digits_perfect_squares (n : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 n) → is_perfect_square d

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  all_digits_perfect_squares n ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧
  n = 4410 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l140_14033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_calculation_l140_14016

noncomputable def x (t : ℝ) : ℝ := Real.exp t * (Real.cos t + Real.sin t)
noncomputable def y (t : ℝ) : ℝ := Real.exp t * (Real.cos t - Real.sin t)

noncomputable def arc_length (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

theorem arc_length_calculation :
  arc_length (π/6) (π/4) = 2 * (Real.exp (π/4) - Real.exp (π/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_calculation_l140_14016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sequence_is_32_l140_14030

def mySequence : List ℚ := [1/4, 8/1, 1/32, 64/1, 1/256, 512/1, 1/2048, 4096/1]

theorem product_of_sequence_is_32 : 
  mySequence.prod = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sequence_is_32_l140_14030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l140_14084

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 - 2*x + 2)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l140_14084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_equality_trig_values_second_quadrant_l140_14078

-- Part 1
theorem logarithm_sum_equality : 
  (Real.log 6.25) / (Real.log 2.5) + Real.log 0.01 / Real.log 10 + Real.log (Real.sqrt (Real.exp 1)) / Real.log (Real.exp 1) - 2^(1 + Real.log 3 / Real.log 2) = -11/2 := by sorry

-- Part 2
theorem trig_values_second_quadrant (α : Real) (h1 : Real.tan α = -3) (h2 : π/2 < α ∧ α < π) :
  Real.sin α = 3 * (Real.sqrt 10) / 10 ∧ Real.cos α = -(Real.sqrt 10) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_equality_trig_values_second_quadrant_l140_14078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_k_bound_l140_14042

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x / x^2 + 2 * k * Real.log x - k * x

-- Define the derivative of f with respect to x
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := 
  (Real.exp x * (x - 2)) / x^3 + 2 * k / x - k

-- Theorem statement
theorem extreme_point_implies_k_bound :
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → x ≠ 2 → f_deriv k x ≠ 0) → k ≤ Real.exp 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_k_bound_l140_14042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_income_main_result_l140_14012

/-- Represents the tax structure and Samantha's income --/
structure TaxInfo where
  q : ℝ  -- Base tax rate in decimal form (e.g., 0.05 for 5%)
  income : ℝ  -- Samantha's annual income

/-- Calculates the total tax based on the given tax structure --/
noncomputable def calculateTax (info : TaxInfo) : ℝ :=
  if info.income ≤ 30000 then
    info.q * info.income
  else
    info.q * 30000 + (info.q + 0.01) * (info.income - 30000)

/-- Theorem stating that Samantha's income is $60000 --/
theorem samantha_income (info : TaxInfo) :
  (calculateTax info = (info.q + 0.005) * info.income) →
  info.income = 60000 := by
  sorry

/-- The main result --/
theorem main_result : ∃ (info : TaxInfo), 
  (calculateTax info = (info.q + 0.005) * info.income) ∧ 
  info.income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_income_main_result_l140_14012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_60_l140_14050

/-- Represents the boat's journey with given parameters. -/
structure BoatJourney where
  downstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the upstream distance given a boat journey. -/
noncomputable def upstream_distance (j : BoatJourney) : ℝ :=
  let downstream_speed := j.downstream_distance / j.downstream_time
  let boat_speed := downstream_speed - j.stream_speed
  let upstream_speed := boat_speed - j.stream_speed
  upstream_speed * j.upstream_time

/-- Theorem stating that for the given journey parameters, the upstream distance is 60 km. -/
theorem upstream_distance_is_60 (j : BoatJourney) 
    (h1 : j.downstream_distance = 100)
    (h2 : j.downstream_time = 10)
    (h3 : j.upstream_time = 15)
    (h4 : j.stream_speed = 3) :
  upstream_distance j = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_60_l140_14050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_one_tenth_l140_14082

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 5

theorem inverse_f_at_negative_one_tenth :
  ∃ y : ℝ, f y = -1/10 ∧ y = (1/2 : ℝ)^(1/7 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_one_tenth_l140_14082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l140_14041

noncomputable section

/-- The function f(x) as described in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.sin (a * x - Real.pi / 4) * Real.cos (a * x - Real.pi / 4) +
  2 * (Real.cos (a * x - Real.pi / 4))^2

/-- The theorem stating the properties of the function f -/
theorem function_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, f a (x + Real.pi / (2 * a)) = f a x) ∧
  (∀ x : ℝ, ∀ T : ℝ, T > 0 ∧ T < Real.pi / (2 * a) → f a (x + T) ≠ f a x) ∧
  a = 2 ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) → f a x ≤ 3) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) ∧ f a x = 3) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) → f a x ≥ 1 - Real.sqrt 3) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) ∧ f a x = 1 - Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l140_14041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l140_14087

open Real

-- Define the function f(x) = 2ln x - x^2
noncomputable def f (x : ℝ) : ℝ := 2 * log x - x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x, 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 ∧ a - x^2 = -2 * log x) ↔ 1 ≤ a ∧ a ≤ (Real.exp 1)^2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l140_14087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_no_solution_l140_14020

-- Define the points M, N, A, B, C
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (4, 0)
def A : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (-4, 2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the distance ratio condition
def S : Set (ℝ × ℝ) := {p | 2 * distance p M = distance p N}

-- Theorem 1: The set S is a circle with equation x^2 + y^2 = 4
theorem trajectory_equation : S = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} := by sorry

-- Theorem 2: There is no point in S satisfying the sum of squared distances condition
theorem no_solution : ∀ p ∈ S, (distance p A)^2 + (distance p B)^2 + (distance p C)^2 ≠ 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_no_solution_l140_14020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_arrangements_l140_14088

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the number of arrangements when A is first or last -/
def arrangements_with_A_fixed : ℕ := 2 * 3 * 2 * 2

theorem experiment_arrangements :
  arrangements_with_A_fixed = 24 := by
  -- Proof goes here
  sorry

#eval arrangements_with_A_fixed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_arrangements_l140_14088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l140_14051

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x - y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define point M
def M : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem intersection_ratio :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    let MA := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
    let MB := Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2)
    let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    MA * MB / AB = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l140_14051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastrami_price_is_five_l140_14086

/-- The price of a Reuben sandwich -/
def reuben_price : ℝ := sorry

/-- The price of a pastrami sandwich -/
def pastrami_price : ℝ := sorry

/-- Pastrami costs $2 more than Reuben -/
axiom price_difference : pastrami_price = reuben_price + 2

/-- Total revenue from selling 10 Reubens and 5 Pastramis -/
axiom total_revenue : 10 * reuben_price + 5 * pastrami_price = 55

/-- Theorem: The price of a pastrami sandwich is $5 -/
theorem pastrami_price_is_five : pastrami_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastrami_price_is_five_l140_14086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l140_14040

/-- A hyperbola with the same asymptotes as x^2/3 - y^2/2 = 1 -/
def same_asymptotes (a b c : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), a * x^2 - b * y^2 = c * k

/-- The point A(√3, 2√5) -/
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 2 * Real.sqrt 5)

/-- The equation passes through a given point -/
def passes_through (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  b * y^2 - a * x^2 = c

theorem hyperbola_equation : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    same_asymptotes a b c ∧
    passes_through a b c point_A ∧
    ∀ (x y : ℝ), b * y^2 - a * x^2 = c :=
sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l140_14040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_radius_is_35_l140_14018

/-- The radius of a circular lawn with a path around it. -/
noncomputable def lawn_radius (path_width : ℝ) (path_area : ℝ) : ℝ :=
  (path_area / Real.pi + path_width ^ 2) / (2 * path_width)

/-- Theorem stating that for a circular lawn with a 7-meter wide path around it,
    if the area of the path is 1693.3184402848983 square meters,
    then the radius of the lawn is 35 meters. -/
theorem lawn_radius_is_35 :
  lawn_radius 7 1693.3184402848983 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_radius_is_35_l140_14018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_parallel_length_is_150_optimal_length_maximizes_area_l140_14056

/-- Represents the configuration of a rectangular cow pasture --/
structure CowPasture where
  barn_length : ℝ
  fence_cost_per_foot : ℝ
  fence_budget : ℝ

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn --/
noncomputable def pasture_area (cp : CowPasture) (x : ℝ) : ℝ :=
  x * (cp.fence_budget / cp.fence_cost_per_foot - 2 * x)

/-- Finds the length of the side perpendicular to the barn that maximizes the area --/
noncomputable def optimal_perpendicular_length (cp : CowPasture) : ℝ :=
  (cp.fence_budget / cp.fence_cost_per_foot) / 4

/-- Calculates the length of the side parallel to the barn that maximizes the area --/
noncomputable def optimal_parallel_length (cp : CowPasture) : ℝ :=
  cp.fence_budget / cp.fence_cost_per_foot - 2 * optimal_perpendicular_length cp

/-- Theorem stating that for the given pasture configuration, the optimal length of the side parallel to the barn is 150 feet --/
theorem optimal_parallel_length_is_150 (cp : CowPasture) 
    (h1 : cp.barn_length = 400)
    (h2 : cp.fence_cost_per_foot = 5)
    (h3 : cp.fence_budget = 1500) : 
  optimal_parallel_length cp = 150 := by
  sorry

/-- Theorem stating that the calculated optimal length indeed maximizes the area --/
theorem optimal_length_maximizes_area (cp : CowPasture) (x : ℝ) :
  pasture_area cp (optimal_perpendicular_length cp) ≥ pasture_area cp x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_parallel_length_is_150_optimal_length_maximizes_area_l140_14056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l140_14076

theorem factorial_equation_solution : ∃ m : ℕ, 3 * 4 * 5 * m = Nat.factorial 8 ∧ m = 672 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l140_14076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_obtuse_angle_l140_14085

-- Define the pentagon vertices
noncomputable def A : ℝ × ℝ := (-1, 3)
noncomputable def B : ℝ × ℝ := (5, -1)
noncomputable def C : ℝ × ℝ := (2 * Real.pi + 2, -1)
noncomputable def D : ℝ × ℝ := (2 * Real.pi + 2, 5)
noncomputable def E : ℝ × ℝ := (-1, 5)

-- Define the pentagon
def pentagon : Set (ℝ × ℝ) := sorry

-- Define the area of the pentagon
noncomputable def pentagon_area : ℝ := 5 * Real.pi + 6

-- Define the region where ∠AQB is obtuse
def obtuse_region : Set (ℝ × ℝ) := sorry

-- Define the area of the obtuse region
noncomputable def obtuse_area : ℝ := 5 * Real.pi

-- Theorem statement
theorem probability_obtuse_angle :
  (obtuse_area / pentagon_area) = (5 * Real.pi) / (5 * Real.pi + 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_obtuse_angle_l140_14085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_2i_in_second_quadrant_l140_14094

noncomputable def euler_formula (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem e_2i_in_second_quadrant : second_quadrant (euler_formula 2) := by
  -- Expand the definition of euler_formula
  unfold euler_formula
  -- Expand the definition of second_quadrant
  unfold second_quadrant
  -- Split the conjunction into two goals
  apply And.intro
  -- Prove that the real part is negative
  · sorry
  -- Prove that the imaginary part is positive
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_2i_in_second_quadrant_l140_14094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_c_l140_14092

theorem max_prime_factors_c (c d : ℕ) 
  (hc : c > 0) (hd : d > 0)
  (h1 : (Nat.gcd c d).factors.length = 8)
  (h2 : (Nat.lcm c d).factors.length = 36)
  (h3 : c.factors.length < d.factors.length) : 
  c.factors.length ≤ 22 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_c_l140_14092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l140_14022

-- Part 1
theorem part_one : (Real.pi - 2 : ℝ) ^ (0 : ℝ) - (1 / 2 : ℝ) ^ (-2 : ℝ) + 3 ^ (2 : ℝ) = 6 := by sorry

-- Part 2
theorem part_two (x : ℝ) : (-2 * x^2)^2 + x^3 * x - x^5 / x = 4 * x^4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l140_14022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fraction_increase_l140_14046

theorem unique_fraction_increase : ∃! (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  (Nat.gcd x y = 1) ∧ 
  ((x + 1 : ℚ) / (y + 1) = 1.2 * (x : ℚ) / y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fraction_increase_l140_14046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_foci_to_asymptotes_l140_14091

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) := {(x, y) | x = Real.sqrt 7 ∨ x = -Real.sqrt 7 ∧ y = 0}

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := Real.sqrt 3 * x + 2 * y = 0 ∨ Real.sqrt 3 * x - 2 * y = 0

-- Theorem statement
theorem distance_foci_to_asymptotes :
  ∀ (f : ℝ × ℝ) (x y : ℝ),
    f ∈ foci →
    asymptotes x y →
    hyperbola x y →
    ∃ (d : ℝ), d = Real.sqrt 3 ∧ d = |Real.sqrt 3 * f.1| / Real.sqrt (3 + 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_foci_to_asymptotes_l140_14091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_ratio_is_three_l140_14058

/-- Represents the circular track where the cyclists ride. -/
structure Track where
  length : ℝ
  length_pos : length > 0

/-- Represents a cyclist. -/
structure Cyclist where
  speed : ℝ
  speed_pos : speed > 0

/-- Calculates the time between meetings of two cyclists moving in opposite directions. -/
noncomputable def time_between_meetings_opposite (track : Track) (c1 c2 : Cyclist) : ℝ :=
  track.length / (c1.speed + c2.speed)

/-- Calculates the time between meetings of two cyclists moving in the same direction. -/
noncomputable def time_between_meetings_same (track : Track) (c1 c2 : Cyclist) : ℝ :=
  track.length / |c1.speed - c2.speed|

/-- The main theorem stating that the ratio of meeting times is 3. -/
theorem meeting_time_ratio_is_three (track : Track) : 
  let petya_initial : Cyclist := ⟨8, by norm_num⟩
  let vasya : Cyclist := ⟨10, by norm_num⟩
  let petya_changed : Cyclist := ⟨16, by norm_num⟩
  let time_initial := time_between_meetings_opposite track petya_initial vasya
  let time_changed := time_between_meetings_same track petya_changed vasya
  time_changed / time_initial = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_ratio_is_three_l140_14058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_squared_plus_x_l140_14001

noncomputable section

-- Define the complex number i
def i : ℂ := Complex.I

-- Define x as given in the problem
noncomputable def x : ℂ := (2 + i * Real.sqrt 2) / 3

-- State the theorem
theorem inverse_x_squared_plus_x :
  1 / (x^2 + x) = (72 - 63 * i * Real.sqrt 2) / 113 := by sorry

end noncomputable section

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_squared_plus_x_l140_14001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l140_14032

def set_A (a : ℝ) : Set ℝ := {x | |x - 2| < a}

def set_B : Set ℝ := {x | (2*x - 1)/(x + 2) ≤ 1}

theorem range_of_a :
  {a : ℝ | set_A a ⊆ set_B} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l140_14032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l140_14097

/-- The ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + p.2^2 = 1}

/-- The foci of the ellipse C -/
noncomputable def foci (C : Set (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A point on the ellipse C -/
noncomputable def point_on_ellipse (C : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The vector from one point to another -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- The magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  let C := ellipse_C
  let (F₁, F₂) := foci C
  let M := point_on_ellipse C
  magnitude (vector M F₁ + vector M F₂) = 2 * Real.sqrt 3 →
  triangle_area M F₁ F₂ = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l140_14097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_a_range_l140_14017

/-- The quadratic function f(x) = ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- Theorem stating the range of a for the given conditions -/
theorem quadratic_function_a_range (a b : ℝ) :
  (f a b (-1) = 1) →
  (∀ x, f a b x < 2) →
  a ∈ Set.Ioc (-4) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_a_range_l140_14017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_center_and_line_l140_14071

/-- Line represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle represented by polar equation -/
structure PolarCircle where
  ρ : ℝ → ℝ

/-- Calculate the distance between a point and a line -/
noncomputable def distancePointToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- The main theorem -/
theorem distance_between_circle_center_and_line
  (l : ParametricLine)
  (c : PolarCircle)
  (h1 : l.x = λ t => 2 * t)
  (h2 : l.y = λ t => 1 + 4 * t)
  (h3 : c.ρ = λ θ => 2 * Real.cos θ) :
  ∃ d : ℝ, d = (3 * Real.sqrt 5) / 5 ∧
    d = distancePointToLine 1 0 2 (-1) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_center_and_line_l140_14071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_neg_necessary_not_sufficient_l140_14023

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- A complex number z defined as i(a+bi) -/
noncomputable def z (a b : ℝ) : ℂ := i * (a + i * b)

/-- Predicate for a complex number being in the first quadrant -/
def is_in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- Statement that "ab < 0" is a necessary but not sufficient condition for z being in the first quadrant -/
theorem ab_neg_necessary_not_sufficient (a b : ℝ) :
  (is_in_first_quadrant (z a b) → a * b < 0) ∧
  ¬(a * b < 0 → is_in_first_quadrant (z a b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_neg_necessary_not_sufficient_l140_14023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l140_14073

noncomputable section

-- Define the parabola
def Parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 2*p*x ∧ p > 0}

-- Define the focus of the parabola
def Focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the directrix of the parabola
def Directrix (p : ℝ) := {(x, _) : ℝ × ℝ | x = -p/2}

-- Define a point in the first quadrant
def FirstQuadrant (A : ℝ × ℝ) := A.1 > 0 ∧ A.2 > 0

-- Define the dot product of two 2D vectors
def DotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem parabola_theorem (p : ℝ) (A B C : ℝ × ℝ) :
  p > 0 →
  A ∈ Parabola p →
  FirstQuadrant A →
  B ∈ Directrix p →
  C.1 = B.1 ∧ C.2 = A.2 →
  A.1 - (Focus p).1 = (Focus p).1 - B.1 ∧
  A.2 - (Focus p).2 = (Focus p).2 - B.2 →
  DotProduct (B.1 - A.1, B.2 - A.2) (B.1 - C.1, B.2 - C.2) = 48 →
  p = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l140_14073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_round_trip_l140_14031

/-- Calculates the average speed of a round trip journey -/
theorem average_speed_round_trip 
  (distance_one_way : ℝ) 
  (time_uphill : ℝ) 
  (time_downhill : ℝ) 
  (h1 : distance_one_way = 2) 
  (h2 : time_uphill = 45 / 60) 
  (h3 : time_downhill = 15 / 60) : 
  (2 * distance_one_way) / (time_uphill + time_downhill) = 4 := by
  sorry

#check average_speed_round_trip

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_round_trip_l140_14031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_thirds_l140_14027

noncomputable section

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 20/9

-- Define the roots of the polynomial
def roots : Set ℝ := {x | f x = 0}

-- Assume there are exactly three distinct roots
axiom three_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ roots = {a, b, c}

-- Define the area of a triangle given its side lengths
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_is_five_thirds :
  ∀ (a b c : ℝ), a ∈ roots → b ∈ roots → c ∈ roots → a ≠ b → b ≠ c → a ≠ c →
  triangle_area a b c = 5/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_thirds_l140_14027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_plates_left_l140_14011

/-- Calculates the number of plates Jack has left after buying new plates and smashing some. -/
theorem jack_plates_left 
  /- Initial number of plates for each pattern -/
  (flower : ℕ) (checked : ℕ) (striped : ℕ)
  /- Hypothesis: Jack initially has 6 flower-patterned, 9 checked-patterned, and 3 striped-patterned plates -/
  (h_flower : flower = 6)
  (h_checked : checked = 9)
  (h_striped : striped = 3)
  /- Hypothesis: Jack smashes 2 flowered plates and 1 striped plate -/
  (h_smash_flower : 2 ≤ flower)
  (h_smash_striped : 1 ≤ striped)
  : flower - 2 + checked + (striped - 1) + checked ^ 2 = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_plates_left_l140_14011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l140_14004

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_and_range_proof 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π / 2) 
  (h_period : ∀ x, f A ω φ (x + π) = f A ω φ x) 
  (h_symmetry : ∀ x, f A ω φ (π / 6 + x) = f A ω φ (π / 6 - x)) 
  (h_point : f A ω φ (π / 2) = 1) :
  (∀ x, f A ω φ x = 2 * Real.sin (2 * x - π / 6)) ∧
  (Set.Icc (-1 : ℝ) 5 = { a | ∃ x ∈ Set.Icc 0 (π / 2), 2 * f A ω φ x - a + 1 = 0 }) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l140_14004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_circumscribed_sphere_l140_14089

/-- The surface area of a circumscribed sphere of a triangular pyramid with
    mutually perpendicular lateral edges of lengths 1, √2, and √3 is 6π. -/
theorem surface_area_circumscribed_sphere (a b c : ℝ) : 
  a = 1 → b = Real.sqrt 2 → c = Real.sqrt 3 → 
  (∀ (i j : Fin 3), i ≠ j → Vector.get ⟨[a, b, c], by simp⟩ i * Vector.get ⟨[a, b, c], by simp⟩ j = 0) →
  4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2)) / 2)^2 = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_circumscribed_sphere_l140_14089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l140_14038

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
noncomputable def selling_price (cost_price : ℝ) (loss_percentage : ℝ) : ℝ :=
  cost_price * (1 - loss_percentage / 100)

/-- Theorem stating that for a radio with a cost price of 1500 and a loss percentage of 13,
    the selling price is 1305. -/
theorem radio_selling_price :
  selling_price 1500 13 = 1305 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the arithmetic expression
  simp [mul_sub, mul_div_cancel]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l140_14038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_bounded_sequence_for_a_greater_than_one_l140_14013

theorem no_bounded_sequence_for_a_greater_than_one :
  ∀ (a : ℝ), a > 1 →
  ¬∃ (x : ℕ → ℝ), 
    (∃ (C : ℝ), ∀ (i : ℕ), |x i| ≤ C) ∧ 
    (∀ (i j : ℕ), i ≠ j → |x i - x j| * (|i - j| : ℝ)^a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_bounded_sequence_for_a_greater_than_one_l140_14013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volume_l140_14028

/-- Regular truncated quadrangular pyramid -/
structure TruncatedPyramid where
  diagonal : ℝ
  base1_side : ℝ
  base2_side : ℝ

/-- Volume of a truncated pyramid -/
noncomputable def volume (p : TruncatedPyramid) : ℝ :=
  let h := Real.sqrt (p.diagonal^2 - (p.base1_side - p.base2_side)^2 / 2)
  let s1 := p.base1_side^2
  let s2 := p.base2_side^2
  h / 3 * (s1 + s2 + Real.sqrt (s1 * s2))

/-- Theorem: The volume of the specified truncated pyramid is 872 -/
theorem truncated_pyramid_volume :
  let p := TruncatedPyramid.mk 18 14 10
  volume p = 872 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volume_l140_14028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_sleeper_coach_l140_14044

def total_passengers : ℕ := 300
def student_percentage : ℚ := 80 / 100
def sleeper_percentage : ℚ := 15 / 100

theorem students_in_sleeper_coach : 
  (↑total_passengers * student_percentage * sleeper_percentage).floor = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_sleeper_coach_l140_14044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_becomes_negative_l140_14057

noncomputable def a : ℕ → ℚ
  | 0 => 56
  | n + 1 => a n - 1 / a n

theorem sequence_becomes_negative :
  ∃ n : ℕ, 0 < n ∧ n < 2002 ∧ a n < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_becomes_negative_l140_14057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phosphorus_symbol_is_P_l140_14059

-- Define the atomic weights
def atomic_weight_Al : Float := 26.98
def atomic_weight_P : Float := 30.97
def atomic_weight_O : Float := 16.00

-- Define the compound's composition
def Al_count : Nat := 1
def P_count : Nat := 1
def O_count : Nat := 4

-- Define the compound's molecular weight
def compound_weight : Float := 122.0

-- Define a function to calculate the molecular weight
def calculate_molecular_weight (al_count p_count o_count : Nat) : Float :=
  al_count.toFloat * atomic_weight_Al +
  p_count.toFloat * atomic_weight_P +
  o_count.toFloat * atomic_weight_O

-- Theorem to prove
theorem phosphorus_symbol_is_P :
  (calculate_molecular_weight Al_count P_count O_count - compound_weight).abs < 0.1 →
  (atomic_weight_P - 30.97).abs < 0.01 →
  (String.mk ['P'] = "P") := by
  intro h1 h2
  rfl

#check phosphorus_symbol_is_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phosphorus_symbol_is_P_l140_14059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tails_one_head_prob_l140_14080

/-- The number of coins being tossed -/
def num_coins : ℕ := 4

/-- The number of possible outcomes for each coin -/
def outcomes_per_coin : ℕ := 2

/-- The probability of getting heads or tails on a single coin -/
def single_outcome_prob : ℚ := 1 / 2

/-- The number of tails we want in the outcome -/
def desired_tails : ℕ := 3

/-- The number of heads we want in the outcome -/
def desired_heads : ℕ := 1

/-- The probability of getting three tails and one head when tossing four coins -/
theorem three_tails_one_head_prob : 
  (Nat.choose num_coins desired_tails : ℚ) * single_outcome_prob ^ num_coins = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tails_one_head_prob_l140_14080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_inverse_final_result_l140_14072

theorem inverse_sum_inverse : (5⁻¹ : ℚ) + (2⁻¹ : ℚ) = 7 / 10 := by
  sorry

theorem final_result : ((5⁻¹ : ℚ) + (2⁻¹ : ℚ))⁻¹ = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_inverse_final_result_l140_14072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_probability_l140_14098

/-- The probability of selecting exactly 2 females and 1 male when randomly choosing 3 contestants
    from a group of 7 contestants (4 females and 3 males) is 18/35. -/
theorem game_show_probability (total : ℕ) (females : ℕ) (males : ℕ) (chosen : ℕ) :
  total = 7 →
  females = 4 →
  males = 3 →
  chosen = 3 →
  (Nat.choose females 2 * Nat.choose males 1 : ℚ) / Nat.choose total chosen = 18/35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_probability_l140_14098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l140_14008

-- Define the points
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 3)

-- Define the reflection point C on the x-axis
def C : ℝ × ℝ := (B.1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem light_path_length : distance A C + distance C B = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l140_14008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l140_14037

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := ⌊x⌋

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ := x - intPart x

-- Theorem 1
theorem part_one :
  decPart 1.6 = 0.6 ∧ intPart (-1.6) = -2 := by sorry

-- Theorem 2
theorem part_two :
  ∀ x : ℝ, intPart x = 2 * decPart x → x = 0 ∨ x = 3/2 := by sorry

-- Theorem 3
theorem part_three :
  ∀ x : ℝ, 3 * intPart x + 1 = 2 * decPart x + x → x = 1/3 := by sorry

-- Axioms
axiom intPart_def : ∀ x : ℝ, x = intPart x + decPart x
axiom decPart_range : ∀ x : ℝ, 0 ≤ decPart x ∧ decPart x < 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l140_14037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_grid_coloring_l140_14043

/-- A coloring of the integer grid -/
def GridColoring := ℤ → ℤ → Bool

/-- The number of points with a specific color on a line -/
def ColoredPointsOnLine (c : GridColoring) (isVertical : Bool) (lineCoord : ℤ) (color : Bool) : Set ℤ :=
  if isVertical then
    {y : ℤ | c lineCoord y = color}
  else
    {x : ℤ | c x lineCoord = color}

/-- Theorem: There exists a grid coloring with finitely many red points on vertical lines
    and finitely many blue points on horizontal lines -/
theorem exists_valid_grid_coloring : ∃ (c : GridColoring),
  (∀ x : ℤ, Set.Finite (ColoredPointsOnLine c true x true)) ∧
  (∀ y : ℤ, Set.Finite (ColoredPointsOnLine c false y false)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_grid_coloring_l140_14043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l140_14099

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 3) :
  4 * (Real.arctan 2) * (Real.arctan ((a * b * c) ^ (1/3 : ℝ))) ≤ 
  π * Real.arctan (1 + (a * b * c) ^ (1/3 : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l140_14099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_and_odd_shift_l140_14034

noncomputable def f (ω θ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + θ - Real.pi / 6)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem sin_period_and_odd_shift (ω θ : ℝ) : 
  (∀ x, f ω θ (x + Real.pi / ω) = f ω θ x) →  -- minimum positive period is π
  (is_odd (λ x ↦ f ω θ (x + Real.pi / 6))) →  -- shifting left by π/6 results in odd function
  ∃ θ', θ' = -Real.pi / 6 ∧ f ω θ' = f ω θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_and_odd_shift_l140_14034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subgraph_iff_topological_subgraph_l140_14015

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type*) where
  edge : V → V → Prop

/-- A subgraph of a graph is a graph with a subset of vertices and edges. -/
def Subgraph {V : Type*} (G H : Graph V) : Prop :=
  ∀ u v, H.edge u v → G.edge u v

/-- A topological subgraph is a subgraph that can be obtained by subdividing edges. -/
def TopologicalSubgraph {V : Type*} (G H : Graph V) : Prop :=
  ∃ (F : Graph V), Subgraph F G ∧ H.edge = F.edge

/-- K^5 is the complete graph on 5 vertices. -/
def K5 (V : Type*) : Graph V where
  edge := λ u v ↦ u ≠ v

/-- K_{3,3} is the complete bipartite graph with 3 vertices on each side. -/
noncomputable def K33 (V : Type*) [Fintype V] [DecidableEq V] : Graph V where
  edge := λ u v ↦ (u ∈ Part1 ∧ v ∈ Part2) ∨ (u ∈ Part2 ∧ v ∈ Part1)
where
  Part1 : Finset V := sorry
  Part2 : Finset V := sorry

/-- The main theorem: A graph contains K^5 or K_{3,3} as a subgraph if and only if
    it contains K^5 or K_{3,3} as a topological subgraph. -/
theorem subgraph_iff_topological_subgraph {V : Type*} [Fintype V] [DecidableEq V] (G : Graph V) :
  (∃ H, (H = K5 V ∨ H = K33 V) ∧ Subgraph H G) ↔
  (∃ H, (H = K5 V ∨ H = K33 V) ∧ TopologicalSubgraph H G) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subgraph_iff_topological_subgraph_l140_14015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l140_14049

theorem perpendicular_vector_scalar (a b : ℝ × ℝ) :
  a = (1, 3) →
  b = (3, 4) →
  (∃ l : ℝ, (a.1 - l * b.1, a.2 - l * b.2) • b = 0) →
  ∃ l : ℝ, (a.1 - l * b.1, a.2 - l * b.2) • b = 0 ∧ l = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l140_14049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l140_14065

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the focus coordinates
noncomputable def focus_coord : ℝ × ℝ := (2, -3 + Real.sqrt (70/3))

-- Theorem statement
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ (x, y) = focus_coord := by
  sorry

#check hyperbola_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l140_14065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l140_14093

-- Define the sets A and C
def A : Set ℝ := {x | (8*x - 1)*(x - 1) ≤ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 5}

-- Define the condition (1/4)^t ∈ A
def condition_t (t : ℝ) : Prop := (1/4)^t ∈ A

-- Define the set B
def B : Set ℝ := {t | condition_t t}

-- State the theorem
theorem problem_solution (a : ℝ) :
  (∀ t : ℝ, condition_t t → t ∈ Set.Icc 0 (3/2)) ∧
  ((A ∪ B) ⊆ C a → a ∈ Set.Icc (-7/4) 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l140_14093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jess_calculation_l140_14090

theorem jess_calculation (y : ℝ) (h : (y - 11) / 5 = 31) : 
  ∃ n : ℚ, n ≥ 14.6 ∧ n < 14.7 ∧ (y - 5) / 11 = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jess_calculation_l140_14090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l140_14095

-- Define the ⊕ operation
noncomputable def circplus (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Define the ⊗ operation
noncomputable def circtimes (a b : ℝ) : ℝ := Real.sqrt ((a - b)^2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (circplus 2 x) / (circtimes x 2 - 2)

-- Statement to prove
theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l140_14095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_difference_l140_14036

theorem integer_difference (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 14) (h4 : x * y = 45) :
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_difference_l140_14036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_log_diffs_l140_14055

def numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def log_diff (a b : ℕ) : ℝ := Real.log (a : ℝ) - Real.log (b : ℝ)

theorem distinct_log_diffs : 
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => log_diff p.1 p.2) 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 ∈ numbers ∧ p.2 ∈ numbers ∧ p.1 ≠ p.2) (Finset.product numbers numbers))) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_log_diffs_l140_14055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l140_14074

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) (D : Real) :
  -- Given conditions
  t.c * Real.tan t.C = Real.sqrt 3 * (t.a * Real.cos t.B + t.b * Real.cos t.A) →
  D ∈ Set.Icc 0 t.c →
  4 = Real.sqrt ((t.c - D)^2 + t.a^2 - 2 * t.a * (t.c - D) * Real.cos t.B) →
  4 = Real.sqrt (D^2 + t.b^2 - 2 * t.b * D * Real.cos t.A) →
  8 * Real.sqrt 3 = 1/2 * 4 * D * Real.sin (t.A + t.B) →
  -- Conclusions
  t.C = π/3 ∧ t.c = 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l140_14074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_coefficient_l140_14053

/-- Triangle ABC with side lengths and inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Ensure side lengths form a valid triangle
  triangle_inequality : AB + BC > CA ∧ BC + CA > AB ∧ CA + AB > BC
  -- Rectangle PQRS inscribed in triangle ABC
  -- P on AB, Q on AC, R and S on BC
  PQRS : Rectangle

/-- Area of rectangle PQRS as a function of its width w --/
def rectangle_area (T : TriangleWithRectangle) (w : ℝ) : ℝ :=
  α * w - β * w^2
  where
    α : ℝ := sorry  -- Definition of α
    β : ℝ := sorry  -- Definition of β

/-- Main theorem --/
theorem inscribed_rectangle_coefficient
  (T : TriangleWithRectangle)
  (h1 : T.AB = 12)
  (h2 : T.BC = 25)
  (h3 : T.CA = 17) :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ β = m / n ∧ m + n = 161 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_coefficient_l140_14053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l140_14014

/-- The area of a parallelogram with base 15 ft and height 3 ft is 45 square feet. -/
theorem parallelogram_area (base height : Real) 
  (h1 : base = 15) (h2 : height = 3) : base * height = 45 := by
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l140_14014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_depends_on_k_l140_14068

noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  (|a * x + b * y + c|) / Real.sqrt (a^2 + b^2)

inductive PositionalRelationship
  | Tangent
  | Intersect
  | Separate

noncomputable def lineCircleRelationship (k b cx cy r : ℝ) : PositionalRelationship :=
  let d := distancePointToLine cx cy k (-1) b
  if d < r then PositionalRelationship.Intersect
  else if d = r then PositionalRelationship.Tangent
  else PositionalRelationship.Separate

theorem line_circle_relationship_depends_on_k :
  ∀ k : ℝ, k ≠ 0 →
  ∃ k₁ k₂ k₃ : ℝ, k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ k₃ ≠ 0 ∧
  lineCircleRelationship k₁ 2 0 3 3 = PositionalRelationship.Tangent ∧
  lineCircleRelationship k₂ 2 0 3 3 = PositionalRelationship.Intersect ∧
  lineCircleRelationship k₃ 2 0 3 3 = PositionalRelationship.Separate :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_depends_on_k_l140_14068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_rational_check_l140_14070

theorem sqrt_two_irrational : 
  ¬ ∃ (q : ℚ), q * q = 2 := by
  sorry

theorem rational_check (a b c : ℚ) (d : ℝ) 
  (h1 : a = -4) (h2 : b = 3/10) (h3 : c = 7/5) (h4 : d = Real.sqrt 2) : 
  ¬ ∃ (q : ℚ), (q : ℝ) = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_rational_check_l140_14070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_asleep_simultaneously_l140_14000

/-- Represents a mathematician in the sleeping problem -/
structure Mathematician where
  id : Nat
  sleep_times : Finset Nat

/-- The sleeping problem setup -/
structure SleepingProblem where
  mathematicians : Finset Mathematician
  total_events : Nat
  pair_asleep : Mathematician → Mathematician → Prop

/-- Main theorem: In any valid sleeping problem, 
    there exists a moment when at least three mathematicians are asleep simultaneously -/
theorem three_asleep_simultaneously 
  (sp : SleepingProblem)
  (h_count : sp.mathematicians.card = 5)
  (h_sleep_twice : ∀ m, m ∈ sp.mathematicians → m.sleep_times.card = 2)
  (h_pairs_asleep : ∀ m1 m2, m1 ∈ sp.mathematicians → m2 ∈ sp.mathematicians → m1 ≠ m2 → sp.pair_asleep m1 m2)
  : ∃ t : Nat, ∃ m1 m2 m3 : Mathematician, 
    m1 ∈ sp.mathematicians ∧ 
    m2 ∈ sp.mathematicians ∧ 
    m3 ∈ sp.mathematicians ∧ 
    m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
    t ∈ m1.sleep_times ∧ 
    t ∈ m2.sleep_times ∧ 
    t ∈ m3.sleep_times :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_asleep_simultaneously_l140_14000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l140_14077

/-- The distance from a point (x₀, y₀) to the line ax + by + c = 0 -/
noncomputable def distancePointToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  (abs (a * x₀ + b * y₀ + c)) / Real.sqrt (a^2 + b^2)

/-- Determines if a line is tangent to a circle -/
def isTangent (a b c x₀ y₀ r : ℝ) : Prop :=
  distancePointToLine x₀ y₀ a b c = r

theorem line_tangent_to_circle :
  isTangent 3 4 (-13) 2 3 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l140_14077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_l140_14039

/-- Given two vectors a and b in ℝ², where a = (sin x, cos x) and b = (2, -3),
    if a is parallel to b, then tan x = -2/3 -/
theorem parallel_vectors_tan (x : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.sin x, Real.cos x]
  let b : Fin 2 → ℝ := ![2, -3]
  (∃ (k : ℝ), a = k • b) → Real.tan x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_l140_14039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fast_painter_time_l140_14067

/-- Represents the time taken by a painter to complete their work -/
structure PainterTime where
  hours : ℚ
  deriving Repr

/-- Represents a time of day -/
structure TimeOfDay where
  hours : ℚ
  deriving Repr

/-- Converts hours past midnight to TimeOfDay -/
noncomputable def hoursToTimeOfDay (h : ℚ) : TimeOfDay :=
  ⟨h % 24⟩

/-- Calculates the time difference between two TimeOfDay values -/
noncomputable def timeDifference (t1 t2 : TimeOfDay) : ℚ :=
  (t2.hours - t1.hours + 24) % 24

theorem fast_painter_time :
  let slow_painter_time : PainterTime := ⟨6⟩
  let slow_start : TimeOfDay := ⟨14⟩  -- 2:00 PM
  let fast_start : TimeOfDay := hoursToTimeOfDay (slow_start.hours + 3)
  let finish_time : TimeOfDay := ⟨(24 + 36/60)⟩  -- 12:36 AM
  let fast_painter_time : PainterTime := ⟨timeDifference fast_start finish_time⟩
  fast_painter_time.hours = 38/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fast_painter_time_l140_14067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_base_sides_correct_l140_14010

/-- Represents a regular quadrilateral truncated pyramid -/
structure TruncatedPyramid where
  α : Real  -- Angle between lateral face and base
  β : Real  -- Angle between parallel plane and base
  S : Real  -- Lateral surface area

/-- Calculates the side length of the lower base of the truncated pyramid -/
noncomputable def lower_base_side (p : TruncatedPyramid) : Real :=
  Real.sqrt ((p.S * Real.cos p.α * Real.sin p.α * Real.sin p.β) / (Real.sin (p.α + p.β) * Real.sin (2 * p.β)))

/-- Calculates the side length of the upper base of the truncated pyramid -/
noncomputable def upper_base_side (p : TruncatedPyramid) : Real :=
  Real.sin (p.α - p.β) * Real.sqrt (p.S / (2 * Real.sin p.α * Real.sin (2 * p.β)))

/-- Theorem stating the correctness of the calculated base sides -/
theorem truncated_pyramid_base_sides_correct (p : TruncatedPyramid) :
  let a := lower_base_side p
  let b := upper_base_side p
  p.S = (a^2 - b^2) / Real.cos p.α ∧
  (a - b) / (a + b) = Real.tan p.β / Real.tan p.α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_base_sides_correct_l140_14010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l140_14045

/-- Pete's current age -/
def p : ℕ := sorry

/-- Claire's current age -/
def c : ℕ := sorry

/-- The number of years until the ratio of Pete's age to Claire's age is 2:1 -/
def x : ℕ := sorry

/-- Pete's age two years ago was three times Claire's age two years ago -/
axiom past_condition_1 : p - 2 = 3 * (c - 2)

/-- Pete's age four years ago was four times Claire's age four years ago -/
axiom past_condition_2 : p - 4 = 4 * (c - 4)

/-- The ratio of their ages will be 2:1 after x years -/
axiom future_ratio : (p + x) = 2 * (c + x)

theorem age_ratio_years : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l140_14045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_two_l140_14069

/-- The function f(x) = (2x^2 + 3x + 10) / (x + 2) has a vertical asymptote at x = -2 -/
theorem vertical_asymptote_at_negative_two :
  ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x' : ℝ, 
    0 < |x' - (-2)| ∧ |x' - (-2)| < δ → 
    |(2 * x'^2 + 3 * x' + 10) / (x' + 2)| > L :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_two_l140_14069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l140_14005

noncomputable def f (a : ℝ) (x : ℝ) := Real.cos x ^ 2 + a * Real.sin x + 2 * a - 1

theorem f_properties :
  ∀ a : ℝ,
  (a = 1 →
    (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f a x ≤ 9/4) ∧
    (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f a x ≥ 0) ∧
    (∃ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧ f a x = 9/4) ∧
    (∃ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧ f a x = 0)) ∧
  ((∀ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f a x ≤ 5) ↔ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l140_14005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_red_or_purple_l140_14024

def total_balls : ℕ := 100
def red_balls : ℕ := 37
def purple_balls : ℕ := 3

theorem probability_not_red_or_purple :
  (total_balls - (red_balls + purple_balls)) / total_balls = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_red_or_purple_l140_14024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l140_14060

theorem angle_terminal_side_point (α : ℝ) (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (m, -1) ∧ P.1 = m * Real.cos α ∧ P.2 = m * Real.sin α) →
  Real.cos α = (2 * Real.sqrt 5) / 5 →
  m > 0 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l140_14060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_in_sliced_cone_l140_14079

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a slice of a cone -/
structure ConeSlice where
  height : ℝ
  baseRadius : ℝ

/-- Calculates the volume of a cone slice -/
noncomputable def volumeOfConeSlice (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.baseRadius^2 * slice.height

/-- Theorem: The ratio of volumes in a sliced cone -/
theorem volume_ratio_in_sliced_cone (cone : RightCircularCone) 
  (h : cone.height > 0) :
  let smallestSlice : ConeSlice := { height := cone.height / 3, baseRadius := cone.baseRadius / 3 }
  let middleSlice : ConeSlice := { height := cone.height / 3, baseRadius := 2 * cone.baseRadius / 3 }
  (volumeOfConeSlice smallestSlice) / (volumeOfConeSlice middleSlice) = 19 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_in_sliced_cone_l140_14079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_areas_l140_14081

/-- Surface area of a sphere --/
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_surface_areas
  (R₁ R₂ R₃ : ℝ)
  (h_radii : R₁ + R₃ = 2 * R₂)
  (S₁ S₂ S₃ : ℝ)
  (h_S₁ : S₁ = surface_area R₁)
  (h_S₂ : S₂ = surface_area R₂)
  (h_S₃ : S₃ = surface_area R₃)
  (h_area₁ : S₁ = 1)
  (h_area₃ : S₃ = 9) :
  S₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_areas_l140_14081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_l140_14029

-- Define the total number of cookies
variable (x : ℝ)

-- Define the fractions of cookies in each tin
noncomputable def tin_A : ℝ := 8/15 * x
noncomputable def tin_B : ℝ := 1/3 * x
noncomputable def tins_ABC : ℝ := 3/5 * x
noncomputable def tins_DE : ℝ := 2/5 * x

-- Define the fraction we want to prove
noncomputable def q : ℝ := tin_B x / (x - tin_A x)

theorem cookie_distribution (x : ℝ) (h : x > 0) : q x = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_l140_14029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_theorem_l140_14003

/-- The area of a regular hexagon with two vertices at (1,2) and (8,5) -/
noncomputable def regular_hexagon_area : ℝ := 29 * Real.sqrt 3

/-- The coordinates of vertex A -/
def vertex_A : ℝ × ℝ := (1, 2)

/-- The coordinates of vertex C -/
def vertex_C : ℝ × ℝ := (8, 5)

/-- Theorem: The area of a regular hexagon with vertices A(1,2) and C(8,5) is 29√3 -/
theorem regular_hexagon_area_theorem :
  let hexagon_area := regular_hexagon_area
  let A := vertex_A
  let C := vertex_C
  hexagon_area = 29 * Real.sqrt 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_theorem_l140_14003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_is_seven_l140_14075

-- Define variables as natural numbers
variable (d e f : ℕ)

-- Define the expression
def expression : ℕ → ℕ → ℕ → ℕ := λ d e f => 40 * d^5 * e^8 * f^15

-- Define the function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (expr : ℕ → ℕ → ℕ → ℕ) : ℕ := 7

-- Theorem statement
theorem sum_of_exponents_is_seven :
  sum_of_exponents_outside_radical expression = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_is_seven_l140_14075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_g_implies_a_range_l140_14083

open Real

/-- The function f(x) = ln x - e^(1-x) -/
noncomputable def f (x : ℝ) : ℝ := log x - exp (1 - x)

/-- The function g(x) = a(x^2 - 1) - 1/x -/
noncomputable def g (a x : ℝ) : ℝ := a * (x^2 - 1) - 1 / x

/-- The theorem stating the range of a for which f(x) < g(x) holds in (1, +∞) -/
theorem f_less_than_g_implies_a_range (a : ℝ) :
  (∀ x, x > 1 → f x < g a x) → a ≥ (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_g_implies_a_range_l140_14083
