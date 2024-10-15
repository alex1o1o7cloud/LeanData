import Mathlib

namespace NUMINAMATH_CALUDE_gcd_7854_13843_l1341_134176

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7854_13843_l1341_134176


namespace NUMINAMATH_CALUDE_coordinates_of_P_l1341_134192

-- Define points M and N
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (-5, -5)

-- Define vector from M to N
def MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define point P
def P : ℝ × ℝ := (x, y) where
  x : ℝ := sorry
  y : ℝ := sorry

-- Define vector from M to P
def MP : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

-- Theorem statement
theorem coordinates_of_P :
  MP = (1/2 : ℝ) • MN → P = (-1, -3/2) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_P_l1341_134192


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l1341_134186

/-- The area of a regular hexagon inscribed in a circle with area 100π square units -/
theorem inscribed_hexagon_area :
  let circle_area : ℝ := 100 * Real.pi
  let hexagon_area : ℝ := 150 * Real.sqrt 3
  (∃ (r : ℝ), r > 0 ∧ circle_area = Real.pi * r^2) →
  (∃ (s : ℝ), s > 0 ∧ hexagon_area = 6 * (s^2 * Real.sqrt 3 / 4)) →
  hexagon_area = 150 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l1341_134186


namespace NUMINAMATH_CALUDE_triangle_relation_angle_C_measure_l1341_134152

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

theorem triangle_relation (t : Triangle)
  (h : Real.sin (2 * t.A + t.B) / Real.sin t.A = 2 + 2 * Real.cos (t.A + t.B)) :
  t.b = 2 * t.a := by sorry

theorem angle_C_measure (t : Triangle)
  (h1 : t.b = 2 * t.a)
  (h2 : t.c = Real.sqrt 7 * t.a) :
  t.C = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_relation_angle_C_measure_l1341_134152


namespace NUMINAMATH_CALUDE_third_range_is_56_prove_third_range_l1341_134134

/-- The minimum possible range of scores -/
def min_range : ℕ := 30

/-- The first given range -/
def range1 : ℕ := 18

/-- The second given range -/
def range2 : ℕ := 26

/-- The theorem stating that the third range is 56 -/
theorem third_range_is_56 : ℕ :=
  min_range + range2

/-- The main theorem to prove -/
theorem prove_third_range :
  third_range_is_56 = 56 :=
by sorry

end NUMINAMATH_CALUDE_third_range_is_56_prove_third_range_l1341_134134


namespace NUMINAMATH_CALUDE_circle_C_equation_circle_C_fixed_point_l1341_134104

-- Define the circle C
def circle_C (t x y : ℝ) : Prop :=
  x^2 + y^2 - 2*t*x - 2*t^2*y + 4*t - 4 = 0

-- Define the line on which the center of C lies
def center_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Theorem 1: Equation of circle C
theorem circle_C_equation (t : ℝ) :
  (∃ x y : ℝ, circle_C t x y ∧ center_line x y) →
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 2*y - 8 = 0) ∨
  (∃ x y : ℝ, x^2 + y^2 - 4*x - 8*y + 4 = 0) :=
sorry

-- Theorem 2: Fixed point of circle C
theorem circle_C_fixed_point (t : ℝ) :
  circle_C t 2 0 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_circle_C_fixed_point_l1341_134104


namespace NUMINAMATH_CALUDE_worker_b_completion_time_l1341_134110

/-- Given workers A, B, and C, and their work rates, prove that B can complete the work alone in 5 days -/
theorem worker_b_completion_time 
  (total_work : ℝ) 
  (rate_a : ℝ) (rate_b : ℝ) (rate_c : ℝ) 
  (time_a : ℝ) (time_b : ℝ) (time_c : ℝ) (time_abc : ℝ) 
  (h1 : rate_a = total_work / time_a)
  (h2 : rate_b = total_work / time_b)
  (h3 : rate_c = total_work / time_c)
  (h4 : rate_a + rate_b + rate_c = total_work / time_abc)
  (h5 : time_a = 4)
  (h6 : time_c = 20)
  (h7 : time_abc = 2)
  (h8 : total_work > 0) :
  time_b = 5 := by
  sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_l1341_134110


namespace NUMINAMATH_CALUDE_circle_equation_shortest_chord_line_l1341_134160

-- Define the circle
def circle_center : ℝ × ℝ := (1, -2)
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, -2.5)

-- Function to check if a point is inside the circle
def is_inside_circle (p : ℝ × ℝ) : Prop := sorry

-- Theorem for the standard equation of the circle
theorem circle_equation (x y : ℝ) : 
  is_inside_circle point_P →
  ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 2) ↔ 
  (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ (is_inside_circle p ∨ tangent_line p.1 p.2)) :=
sorry

-- Theorem for the equation of the line containing the shortest chord
theorem shortest_chord_line (x y : ℝ) :
  is_inside_circle point_P →
  (4*x - 2*y - 13 = 0) ↔ 
  (∃ (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    is_inside_circle p ∧ 
    is_inside_circle q ∧ 
    p.1 = x ∧ p.2 = y ∧
    (q.1 - point_P.1) * (p.2 - point_P.2) = (q.2 - point_P.2) * (p.1 - point_P.1) ∧
    ∀ (r s : ℝ × ℝ), 
      r ≠ s → 
      is_inside_circle r → 
      is_inside_circle s → 
      (r.1 - point_P.1) * (s.2 - point_P.2) = (r.2 - point_P.2) * (s.1 - point_P.1) →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ (r.1 - s.1)^2 + (r.2 - s.2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_shortest_chord_line_l1341_134160


namespace NUMINAMATH_CALUDE_charlotte_overall_score_l1341_134196

/-- Charlotte's test scores -/
def charlotte_scores : Fin 3 → ℚ
  | 0 => 60 / 100
  | 1 => 75 / 100
  | 2 => 85 / 100

/-- Number of problems in each test -/
def test_problems : Fin 3 → ℕ
  | 0 => 15
  | 1 => 20
  | 2 => 25

/-- Total number of problems in the combined test -/
def total_problems : ℕ := 60

/-- Charlotte's overall score on the combined test -/
def overall_score : ℚ := (charlotte_scores 0 * test_problems 0 +
                          charlotte_scores 1 * test_problems 1 +
                          charlotte_scores 2 * test_problems 2) / total_problems

theorem charlotte_overall_score :
  overall_score = 75 / 100 := by sorry

end NUMINAMATH_CALUDE_charlotte_overall_score_l1341_134196


namespace NUMINAMATH_CALUDE_triangle_area_approximation_l1341_134118

theorem triangle_area_approximation (α β : Real) (k l m : Real) :
  α = π / 6 →
  β = π / 4 →
  k = 3 →
  l = 2 →
  m = 4 →
  let γ : Real := π - α - β
  let S := ((k * Real.sin α + l * Real.sin β + m * Real.sin γ) ^ 2) / (2 * Real.sin α * Real.sin β * Real.sin γ)
  |S - 67| < 0.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_approximation_l1341_134118


namespace NUMINAMATH_CALUDE_max_pieces_is_nine_l1341_134191

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 15

/-- The size of a small piece in inches -/
def small_piece_size : ℕ := 5

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_pieces_is_nine : max_pieces = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_nine_l1341_134191


namespace NUMINAMATH_CALUDE_f_fixed_point_exists_f_fixed_point_19_pow_86_l1341_134133

def f (A : ℕ) : ℕ :=
  let digits := Nat.digits 10 A
  List.sum (List.zipWith (·*·) (List.reverse digits) (List.map (2^·) (List.range digits.length)))

theorem f_fixed_point_exists (A : ℕ) : ∃ k : ℕ, f (f^[k] A) = f^[k] A :=
sorry

theorem f_fixed_point_19_pow_86 : ∃ k : ℕ, f^[k] (19^86) = 19 :=
sorry

end NUMINAMATH_CALUDE_f_fixed_point_exists_f_fixed_point_19_pow_86_l1341_134133


namespace NUMINAMATH_CALUDE_min_value_of_4a_plus_b_l1341_134100

theorem min_value_of_4a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (1 : ℝ) / a + (1 : ℝ) / b = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1 : ℝ) / a' + (1 : ℝ) / b' = 1 → 4 * a' + b' ≥ 4 * a + b) ∧ 
  4 * a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_4a_plus_b_l1341_134100


namespace NUMINAMATH_CALUDE_jasons_lawn_cutting_l1341_134175

/-- The number of lawns Jason can cut in 8 hours, given that it takes 30 minutes to cut one lawn -/
theorem jasons_lawn_cutting (time_per_lawn : ℕ) (total_time_hours : ℕ) : 
  time_per_lawn = 30 → total_time_hours = 8 → (total_time_hours * 60) / time_per_lawn = 16 := by
  sorry

end NUMINAMATH_CALUDE_jasons_lawn_cutting_l1341_134175


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l1341_134158

/-- Conversion rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 30 / 20

/-- Conversion rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 20 / 12

/-- The number of yahs we want to convert -/
def target_yahs : ℕ := 500

/-- The equivalent number of bahs -/
def equivalent_bahs : ℕ := 200

theorem yah_to_bah_conversion :
  (target_yahs : ℚ) / (rah_to_yah_rate * bah_to_rah_rate) = equivalent_bahs := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l1341_134158


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l1341_134121

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine configuration,
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_color_is_13 (machine : GumballMachine)
    (h : machine = { red := 12, white := 15, blue := 10, green := 7 }) :
    minGumballsForFourSameColor machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l1341_134121


namespace NUMINAMATH_CALUDE_largest_number_l1341_134132

def A : ℕ := 27

def B (A : ℕ) : ℕ := A + 7

def C (B : ℕ) : ℕ := B - 9

def D (C : ℕ) : ℕ := 2 * C

theorem largest_number (A B C D : ℕ) (hA : A = 27) (hB : B = A + 7) (hC : C = B - 9) (hD : D = 2 * C) :
  D = max A (max B (max C D)) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1341_134132


namespace NUMINAMATH_CALUDE_max_profit_at_18_profit_maximized_at_18_l1341_134157

-- Define the profit function
def profit (x : ℝ) : ℝ := -0.5 * x^2 + 18 * x - 20

-- Theorem statement
theorem max_profit_at_18 :
  ∃ (x_max : ℝ), x_max > 0 ∧ 
  (∀ (x : ℝ), x > 0 → profit x ≤ profit x_max) ∧
  x_max = 18 ∧ profit x_max = 142 := by
  sorry

-- Additional theorem to show that 18 is indeed the maximizer
theorem profit_maximized_at_18 :
  ∀ (x : ℝ), x > 0 → profit x ≤ profit 18 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_18_profit_maximized_at_18_l1341_134157


namespace NUMINAMATH_CALUDE_arcsin_of_one_l1341_134181

theorem arcsin_of_one (π : Real) : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_one_l1341_134181


namespace NUMINAMATH_CALUDE_monitor_length_is_14_l1341_134198

/-- Represents the dimensions of a rectangular monitor. -/
structure Monitor where
  width : ℝ
  length : ℝ
  circumference : ℝ

/-- The circumference of a rectangle is equal to twice the sum of its length and width. -/
def circumference_formula (m : Monitor) : Prop :=
  m.circumference = 2 * (m.length + m.width)

/-- Theorem: A monitor with width 9 cm and circumference 46 cm has a length of 14 cm. -/
theorem monitor_length_is_14 :
  ∃ (m : Monitor), m.width = 9 ∧ m.circumference = 46 ∧ circumference_formula m → m.length = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_monitor_length_is_14_l1341_134198


namespace NUMINAMATH_CALUDE_total_tips_proof_l1341_134112

/-- Calculates the total tips earned over 3 days for a food truck --/
def total_tips (tips_per_customer : ℚ) (friday_customers : ℕ) (sunday_customers : ℕ) : ℚ :=
  let saturday_customers := 3 * friday_customers
  tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

/-- Proves that the total tips earned over 3 days is $296.00 --/
theorem total_tips_proof :
  total_tips 2 28 36 = 296 := by
  sorry

end NUMINAMATH_CALUDE_total_tips_proof_l1341_134112


namespace NUMINAMATH_CALUDE_sum_x_y_equals_eight_l1341_134172

theorem sum_x_y_equals_eight (x y : ℝ) 
  (h1 : |x| + x + y = 14)
  (h2 : x + |y| - y = 10)
  (h3 : |x| - |y| + x - y = 8) :
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_eight_l1341_134172


namespace NUMINAMATH_CALUDE_sum_of_primes_less_than_20_is_77_l1341_134135

def is_prime (n : ℕ) : Prop := sorry

def sum_of_primes_less_than_20 : ℕ := sorry

theorem sum_of_primes_less_than_20_is_77 :
  sum_of_primes_less_than_20 = 77 := by sorry

end NUMINAMATH_CALUDE_sum_of_primes_less_than_20_is_77_l1341_134135


namespace NUMINAMATH_CALUDE_triangle_inequality_l1341_134146

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1341_134146


namespace NUMINAMATH_CALUDE_sawyer_coaching_fee_l1341_134188

/-- Calculate the total coaching fee for Sawyer --/
theorem sawyer_coaching_fee :
  let start_date : Nat := 1  -- January 1
  let end_date : Nat := 307  -- November 3
  let daily_fee : ℚ := 39
  let discount_days : Nat := 50
  let discount_rate : ℚ := 0.1

  let full_price_days : Nat := min discount_days (end_date - start_date + 1)
  let discounted_days : Nat := (end_date - start_date + 1) - full_price_days
  let discounted_fee : ℚ := daily_fee * (1 - discount_rate)

  let total_fee : ℚ := (full_price_days : ℚ) * daily_fee + (discounted_days : ℚ) * discounted_fee

  total_fee = 10967.7 := by
    sorry

end NUMINAMATH_CALUDE_sawyer_coaching_fee_l1341_134188


namespace NUMINAMATH_CALUDE_train_speed_l1341_134120

theorem train_speed 
  (n : ℝ) 
  (a : ℝ) 
  (b : ℝ) 
  (c : ℝ) 
  (h1 : n > 0) 
  (h2 : a > c) 
  (h3 : b > 0) : 
  ∃ (speed : ℝ), speed = (b * (n + 1)) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1341_134120


namespace NUMINAMATH_CALUDE_total_bears_is_98_l1341_134150

/-- The maximum number of teddy bears that can be placed on each shelf. -/
def max_bears_per_shelf : ℕ := 7

/-- The number of filled shelves. -/
def filled_shelves : ℕ := 14

/-- The total number of teddy bears. -/
def total_bears : ℕ := max_bears_per_shelf * filled_shelves

/-- Theorem stating that the total number of teddy bears is 98. -/
theorem total_bears_is_98 : total_bears = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_bears_is_98_l1341_134150


namespace NUMINAMATH_CALUDE_at_least_one_correct_guess_l1341_134125

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Blue
| Green

/-- Converts HatColor to its corresponding integer representation -/
def hatColorToInt (color : HatColor) : Fin 3 :=
  match color with
  | HatColor.Red => 0
  | HatColor.Blue => 1
  | HatColor.Green => 2

/-- Represents the configuration of hats on the four sages -/
structure HatConfiguration where
  a : HatColor
  b : HatColor
  c : HatColor
  d : HatColor

/-- Represents a sage's guess -/
def SageGuess := Fin 3

/-- The strategy for Sage A -/
def guessA (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.b + hatColorToInt config.d) % 3

/-- The strategy for Sage B -/
def guessB (config : HatConfiguration) : SageGuess :=
  (-(hatColorToInt config.a + hatColorToInt config.c)) % 3

/-- The strategy for Sage C -/
def guessC (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.b - hatColorToInt config.d) % 3

/-- The strategy for Sage D -/
def guessD (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.c - hatColorToInt config.a) % 3

/-- Theorem stating that the strategy guarantees at least one correct guess -/
theorem at_least_one_correct_guess (config : HatConfiguration) :
  (guessA config = hatColorToInt config.a) ∨
  (guessB config = hatColorToInt config.b) ∨
  (guessC config = hatColorToInt config.c) ∨
  (guessD config = hatColorToInt config.d) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_correct_guess_l1341_134125


namespace NUMINAMATH_CALUDE_fraction_simplification_l1341_134128

theorem fraction_simplification (x : ℝ) : 
  (x + 2) / 4 + (3 - 4*x) / 5 + (7*x - 1) / 10 = (3*x + 20) / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1341_134128


namespace NUMINAMATH_CALUDE_last_remaining_number_l1341_134137

/-- Represents the state of the number sequence --/
structure SequenceState where
  numbers : List Nat
  markStart : Nat

/-- Marks every third number in the sequence --/
def markEveryThird (state : SequenceState) : SequenceState := sorry

/-- Reverses the remaining numbers in the sequence --/
def reverseRemaining (state : SequenceState) : SequenceState := sorry

/-- Performs one round of marking and reversing --/
def performRound (state : SequenceState) : SequenceState := sorry

/-- Continues the process until only one number remains --/
def processUntilOne (state : SequenceState) : Nat := sorry

/-- The main theorem to be proved --/
theorem last_remaining_number :
  processUntilOne { numbers := List.range 120, markStart := 1 } = 57 := by sorry

end NUMINAMATH_CALUDE_last_remaining_number_l1341_134137


namespace NUMINAMATH_CALUDE_solve_equation_l1341_134103

theorem solve_equation (x y : ℚ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1341_134103


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_l1341_134187

theorem sum_of_absolute_values (a b : ℤ) : 
  (abs a = 2023) → (abs b = 2022) → (a > b) → ((a + b = 1) ∨ (a + b = 4045)) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_l1341_134187


namespace NUMINAMATH_CALUDE_scooter_profit_l1341_134140

theorem scooter_profit (original_cost repair_cost profit_percentage : ℝ) : 
  repair_cost = 0.1 * original_cost → 
  repair_cost = 500 → 
  profit_percentage = 0.2 → 
  original_cost * profit_percentage = 1000 := by
sorry

end NUMINAMATH_CALUDE_scooter_profit_l1341_134140


namespace NUMINAMATH_CALUDE_stable_poly_characterization_l1341_134101

-- Define the set K of positive integers not containing the digit 7
def K : Set Nat := {n : Nat | n > 0 ∧ ∀ d, d ∈ n.digits 10 → d ≠ 7}

-- Define a polynomial with nonnegative coefficients
def NonNegativePoly (f : Nat → Nat) : Prop :=
  ∃ (coeffs : List Nat), ∀ x, f x = (coeffs.enum.map (λ (i, a) => a * x^i)).sum

-- Define the stable property for a polynomial
def Stable (f : Nat → Nat) : Prop :=
  ∀ x, x ∈ K → f x ∈ K

-- Theorem statement
theorem stable_poly_characterization (f : Nat → Nat) 
  (h_nonneg : NonNegativePoly f) (h_stable : Stable f) :
  (∃ e k, k ∈ K ∧ ∀ x, f x = 10^e * x + k) ∨
  (∃ e, ∀ x, f x = 10^e * x) ∨
  (∃ k, k ∈ K ∧ ∀ x, f x = k) :=
sorry

end NUMINAMATH_CALUDE_stable_poly_characterization_l1341_134101


namespace NUMINAMATH_CALUDE_square_sum_value_l1341_134189

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -9) :
  x^2 + y^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1341_134189


namespace NUMINAMATH_CALUDE_mango_rate_is_75_l1341_134197

/-- The rate of mangoes per kg given the purchase details -/
def mango_rate (apple_weight : ℕ) (apple_rate : ℕ) (mango_weight : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_weight * apple_rate) / mango_weight

/-- Theorem stating that the rate of mangoes is 75 per kg -/
theorem mango_rate_is_75 :
  mango_rate 8 70 9 1235 = 75 := by
  sorry

#eval mango_rate 8 70 9 1235

end NUMINAMATH_CALUDE_mango_rate_is_75_l1341_134197


namespace NUMINAMATH_CALUDE_candy_distribution_l1341_134179

/-- Calculates the number of candy pieces each student receives -/
def candy_per_student (total : ℕ) (reserved : ℕ) (students : ℕ) : ℕ :=
  (total - reserved) / students

/-- Proves that each student receives 6 pieces of candy -/
theorem candy_distribution (total : ℕ) (reserved : ℕ) (students : ℕ) 
  (h1 : total = 344) 
  (h2 : reserved = 56) 
  (h3 : students = 43) : 
  candy_per_student total reserved students = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1341_134179


namespace NUMINAMATH_CALUDE_remainder_polynomial_l1341_134141

theorem remainder_polynomial (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 8) (h3 : p 0 = 6) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 2) * (x - 5) + ((1/3) * x + 19/3) := by
sorry

end NUMINAMATH_CALUDE_remainder_polynomial_l1341_134141


namespace NUMINAMATH_CALUDE_equation_solutions_l1341_134126

theorem equation_solutions : 
  {x : ℝ | (x^3 - 3*x^2)/(x^2 - 4*x + 4) + x = -3} = {-2, 3/2} :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1341_134126


namespace NUMINAMATH_CALUDE_regular_implies_all_equal_regular_implies_rotational_symmetry_rotational_symmetry_implies_regular_regular_implies_topologically_regular_l1341_134113

-- Define a structure for a polyhedron
structure Polyhedron where
  vertices : Set Point
  edges : Set (Point × Point)
  faces : Set (Set Point)

-- Define properties of a regular polyhedron
def is_regular (P : Polyhedron) : Prop := sorry

-- Define equality of geometric elements
def all_elements_equal (P : Polyhedron) : Prop := sorry

-- Define rotational symmetry property
def has_rotational_symmetry (P : Polyhedron) : Prop := sorry

-- Define topological regularity
def is_topologically_regular (P : Polyhedron) : Prop := sorry

-- Theorem 1
theorem regular_implies_all_equal (P : Polyhedron) :
  is_regular P → all_elements_equal P := by sorry

-- Theorem 2
theorem regular_implies_rotational_symmetry (P : Polyhedron) :
  is_regular P → has_rotational_symmetry P := by sorry

-- Theorem 3
theorem rotational_symmetry_implies_regular (P : Polyhedron) :
  has_rotational_symmetry P → is_regular P := by sorry

-- Theorem 4
theorem regular_implies_topologically_regular (P : Polyhedron) :
  is_regular P → is_topologically_regular P := by sorry

end NUMINAMATH_CALUDE_regular_implies_all_equal_regular_implies_rotational_symmetry_rotational_symmetry_implies_regular_regular_implies_topologically_regular_l1341_134113


namespace NUMINAMATH_CALUDE_emily_shopping_expense_l1341_134108

def total_spent (art_supplies_cost skirt_cost number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + skirt_cost * number_of_skirts

theorem emily_shopping_expense :
  let art_supplies_cost : ℕ := 20
  let skirt_cost : ℕ := 15
  let number_of_skirts : ℕ := 2
  total_spent art_supplies_cost skirt_cost number_of_skirts = 50 := by
  sorry

end NUMINAMATH_CALUDE_emily_shopping_expense_l1341_134108


namespace NUMINAMATH_CALUDE_trajectory_circle_fixed_points_l1341_134139

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

/-- The distance condition for point M -/
def distance_condition (x y : ℝ) : Prop :=
  ((x - 1)^2 + y^2)^(1/2) = x + 1

/-- The line passing through F(1,0) and intersecting the trajectory -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- The circle with diameter AB -/
def circle_AB (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 + 4 * y = 4

/-- The main theorem -/
theorem trajectory_circle_fixed_points :
  ∀ x y m,
  trajectory x y →
  distance_condition x y →
  intersecting_line m x y →
  (circle_AB (-1) 0 ∧ circle_AB 3 0) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_circle_fixed_points_l1341_134139


namespace NUMINAMATH_CALUDE_cube_inequality_iff_inequality_l1341_134155

theorem cube_inequality_iff_inequality (a b : ℝ) : a^3 > b^3 ↔ a > b := by sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_inequality_l1341_134155


namespace NUMINAMATH_CALUDE_building_stories_l1341_134115

theorem building_stories (apartments_per_floor : ℕ) (people_per_apartment : ℕ) (total_people : ℕ) :
  apartments_per_floor = 4 →
  people_per_apartment = 2 →
  total_people = 200 →
  total_people / (apartments_per_floor * people_per_apartment) = 25 :=
by sorry

end NUMINAMATH_CALUDE_building_stories_l1341_134115


namespace NUMINAMATH_CALUDE_new_rectangle_area_l1341_134162

theorem new_rectangle_area (x y : ℝ) (h : 0 < x ∧ x ≤ y) :
  let base := Real.sqrt (x^2 + y^2) + y
  let altitude := Real.sqrt (x^2 + y^2) - y
  base * altitude = x^2 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l1341_134162


namespace NUMINAMATH_CALUDE_augmented_matrix_of_system_l1341_134199

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x + 5 * y + 6 = 0
def equation2 (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Define the augmented matrix
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ :=
  !![3, 5, -6;
     4, -3, 7]

-- Theorem statement
theorem augmented_matrix_of_system :
  ∀ (x y : ℝ), equation1 x y ∧ equation2 x y →
  augmented_matrix = !![3, 5, -6; 4, -3, 7] := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_of_system_l1341_134199


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1341_134159

theorem unique_integer_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1341_134159


namespace NUMINAMATH_CALUDE_inequality_problem_l1341_134193

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c < b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1341_134193


namespace NUMINAMATH_CALUDE_point_on_line_l1341_134194

/-- Given a line L with equation Ax + By + C = 0 that can be rewritten as A(x - x₀) + B(y - y₀) = 0,
    prove that the point (x₀, y₀) lies on the line L. -/
theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (∀ x y, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0) →
  A * x₀ + B * y₀ + C = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1341_134194


namespace NUMINAMATH_CALUDE_linear_function_value_l1341_134156

theorem linear_function_value (k b : ℝ) :
  ((-1 : ℝ) * k + b = 1) →
  (2 * k + b = -2) →
  (1 : ℝ) * k + b = -1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l1341_134156


namespace NUMINAMATH_CALUDE_cricket_team_size_l1341_134123

theorem cricket_team_size :
  ∀ (n : ℕ) (initial_avg final_avg : ℝ),
  initial_avg = 29 →
  final_avg = 26 →
  (n * final_avg = (n - 2) * (initial_avg - 1) + (initial_avg + 3) + initial_avg) →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l1341_134123


namespace NUMINAMATH_CALUDE_solve_equation_l1341_134145

theorem solve_equation (x : ℝ) :
  3 * (x - 5) = 3 * (18 - 5) → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1341_134145


namespace NUMINAMATH_CALUDE_riku_stickers_comparison_l1341_134169

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := 85

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := 2210

/-- The number of times Riku has more stickers than Kristoff -/
def times_more_stickers : ℚ := riku_stickers / kristoff_stickers

/-- Theorem stating that Riku has 26 times more stickers than Kristoff -/
theorem riku_stickers_comparison : times_more_stickers = 26 := by
  sorry

end NUMINAMATH_CALUDE_riku_stickers_comparison_l1341_134169


namespace NUMINAMATH_CALUDE_harry_snakes_l1341_134106

/-- The number of snakes Harry owns -/
def num_snakes : ℕ := sorry

/-- The number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- The number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- Monthly feeding cost per snake in dollars -/
def snake_cost : ℕ := 10

/-- Monthly feeding cost per iguana in dollars -/
def iguana_cost : ℕ := 5

/-- Monthly feeding cost per gecko in dollars -/
def gecko_cost : ℕ := 15

/-- Total yearly feeding cost for all pets in dollars -/
def total_yearly_cost : ℕ := 1140

theorem harry_snakes :
  num_snakes = 4 ∧
  (12 * (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost) = total_yearly_cost) :=
by sorry

end NUMINAMATH_CALUDE_harry_snakes_l1341_134106


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_45_4095_l1341_134105

theorem gcd_lcm_sum_45_4095 : 
  (Nat.gcd 45 4095) + (Nat.lcm 45 4095) = 4140 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_45_4095_l1341_134105


namespace NUMINAMATH_CALUDE_volume_ratio_l1341_134180

theorem volume_ratio (A B C : ℝ) 
  (h1 : 2 * A = B + C) 
  (h2 : 5 * B = A + C) : 
  C / (A + B) = 1 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_l1341_134180


namespace NUMINAMATH_CALUDE_subtract_negatives_l1341_134119

theorem subtract_negatives : (-7) - (-5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l1341_134119


namespace NUMINAMATH_CALUDE_david_math_homework_time_l1341_134164

/-- Given David's homework times, prove he spent 15 minutes on math. -/
theorem david_math_homework_time :
  ∀ (total_time spelling_time reading_time math_time : ℕ),
    total_time = 60 →
    spelling_time = 18 →
    reading_time = 27 →
    math_time = total_time - spelling_time - reading_time →
    math_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_david_math_homework_time_l1341_134164


namespace NUMINAMATH_CALUDE_diamond_value_l1341_134130

/-- Given a digit d, this function returns the value of d3 in base 5 -/
def base5_value (d : ℕ) : ℕ := d * 5 + 3

/-- Given a digit d, this function returns the value of d2 in base 6 -/
def base6_value (d : ℕ) : ℕ := d * 6 + 2

/-- The theorem states that the digit d satisfying d3 in base 5 equals d2 in base 6 is 1 -/
theorem diamond_value :
  ∃ (d : ℕ), d < 10 ∧ base5_value d = base6_value d ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_diamond_value_l1341_134130


namespace NUMINAMATH_CALUDE_orange_distribution_ratio_l1341_134178

/-- Proves the ratio of oranges given to the brother to the total number of oranges --/
theorem orange_distribution_ratio :
  let total_oranges : ℕ := 12
  let friend_oranges : ℕ := 2
  ∀ brother_fraction : ℚ,
    (1 / 4 : ℚ) * ((1 : ℚ) - brother_fraction) * total_oranges = friend_oranges →
    (brother_fraction * total_oranges : ℚ) / total_oranges = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_ratio_l1341_134178


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_one_l1341_134122

theorem complex_magnitude_equals_one : ∀ (z : ℂ), z = (2 * Complex.I + 1) / (Complex.I - 2) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_one_l1341_134122


namespace NUMINAMATH_CALUDE_cos_to_sin_shift_l1341_134143

open Real

theorem cos_to_sin_shift (x : ℝ) : 
  cos (2*x) = sin (2*(x - π/6)) :=
by sorry

end NUMINAMATH_CALUDE_cos_to_sin_shift_l1341_134143


namespace NUMINAMATH_CALUDE_problem_solution_l1341_134177

theorem problem_solution (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  w = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1341_134177


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l1341_134127

def a (x : ℝ) : Fin 3 → ℝ := ![x, 4, 1]
def b (y : ℝ) : Fin 3 → ℝ := ![-2, y, -1]

theorem parallel_vectors_solution (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ a x = k • b y) → x = 2 ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l1341_134127


namespace NUMINAMATH_CALUDE_intersection_A_B_l1341_134148

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Define set B
def B : Set ℝ := {x | x^2 - x < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1341_134148


namespace NUMINAMATH_CALUDE_carolyn_unicorns_l1341_134184

/-- Calculates the number of unicorns Carolyn wants to embroider --/
def number_of_unicorns (stitches_per_minute : ℕ) (flower_stitches : ℕ) (unicorn_stitches : ℕ) 
  (godzilla_stitches : ℕ) (total_minutes : ℕ) (number_of_flowers : ℕ) : ℕ :=
  let total_stitches := stitches_per_minute * total_minutes
  let flower_total_stitches := flower_stitches * number_of_flowers
  let remaining_stitches := total_stitches - flower_total_stitches - godzilla_stitches
  remaining_stitches / unicorn_stitches

/-- Theorem stating that Carolyn wants to embroider 3 unicorns --/
theorem carolyn_unicorns : 
  number_of_unicorns 4 60 180 800 1085 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_unicorns_l1341_134184


namespace NUMINAMATH_CALUDE_polynomial_sum_l1341_134185

theorem polynomial_sum (h k : ℝ → ℝ) :
  (∀ x, h x + k x = -3 + 2 * x) →
  (∀ x, h x = x^3 - 3 * x^2 - 2) →
  (∀ x, k x = -x^3 + 3 * x^2 + 2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1341_134185


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1341_134167

/-- Given two quadratic equations, where the roots of the second are each three less than
    the roots of the first, this theorem proves that the constant term of the second
    equation is 3.5. -/
theorem quadratic_root_relation (d : ℝ) :
  (∃ r s : ℝ, r + s = 2 ∧ r * s = 1/2 ∧ 
   ∀ x : ℝ, 4 * x^2 - 8 * x + 2 = 0 ↔ (x = r ∨ x = s)) →
  (∃ e : ℝ, ∀ x : ℝ, x^2 + d * x + e = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  ∃ e : ℝ, e = 3.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1341_134167


namespace NUMINAMATH_CALUDE_simplify_fraction_l1341_134163

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1341_134163


namespace NUMINAMATH_CALUDE_manicure_cost_l1341_134144

/-- The cost of a manicure before tip, given the total amount paid and tip percentage. -/
theorem manicure_cost (total_paid : ℝ) (tip_percentage : ℝ) (cost : ℝ) : 
  total_paid = 39 → 
  tip_percentage = 0.30 → 
  cost * (1 + tip_percentage) = total_paid → 
  cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_manicure_cost_l1341_134144


namespace NUMINAMATH_CALUDE_isabelle_bubble_bath_amount_l1341_134124

/-- Represents the configuration of a hotel --/
structure HotelConfig where
  double_suites : Nat
  couple_rooms : Nat
  single_rooms : Nat
  family_rooms : Nat
  double_suite_capacity : Nat
  couple_room_capacity : Nat
  single_room_capacity : Nat
  family_room_capacity : Nat
  bubble_bath_per_guest : Nat

/-- Calculates the total bubble bath needed for a given hotel configuration --/
def total_bubble_bath (config : HotelConfig) : Nat :=
  (config.double_suites * config.double_suite_capacity +
   config.couple_rooms * config.couple_room_capacity +
   config.single_rooms * config.single_room_capacity +
   config.family_rooms * config.family_room_capacity) *
  config.bubble_bath_per_guest

/-- The specific hotel configuration from the problem --/
def isabelle_hotel : HotelConfig :=
  { double_suites := 5
  , couple_rooms := 13
  , single_rooms := 14
  , family_rooms := 3
  , double_suite_capacity := 4
  , couple_room_capacity := 2
  , single_room_capacity := 1
  , family_room_capacity := 6
  , bubble_bath_per_guest := 25
  }

/-- Theorem stating that the total bubble bath needed for Isabelle's hotel is 1950 ml --/
theorem isabelle_bubble_bath_amount :
  total_bubble_bath isabelle_hotel = 1950 := by
  sorry

end NUMINAMATH_CALUDE_isabelle_bubble_bath_amount_l1341_134124


namespace NUMINAMATH_CALUDE_tile_difference_l1341_134147

/-- Given an initial figure with blue and red hexagonal tiles, and adding a border of red tiles,
    calculate the difference between the total number of red tiles and blue tiles in the new figure. -/
theorem tile_difference (initial_blue : ℕ) (initial_red : ℕ) (border_red : ℕ) : 
  initial_blue = 17 → initial_red = 8 → border_red = 24 →
  (initial_red + border_red) - initial_blue = 15 := by
sorry

end NUMINAMATH_CALUDE_tile_difference_l1341_134147


namespace NUMINAMATH_CALUDE_course_selection_theorem_l1341_134131

/-- The number of ways for students to select courses --/
def selectCourses (numCourses numStudents coursesPerStudent : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of selection methods --/
theorem course_selection_theorem :
  selectCourses 4 3 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l1341_134131


namespace NUMINAMATH_CALUDE_total_prom_cost_is_correct_l1341_134138

/-- Calculates the total cost of prom services for Keesha -/
def total_prom_cost : ℝ :=
  let updo_cost : ℝ := 50
  let updo_discount : ℝ := 0.1
  let manicure_cost : ℝ := 30
  let pedicure_cost : ℝ := 35
  let pedicure_discount : ℝ := 0.5
  let makeup_cost : ℝ := 40
  let makeup_tax : ℝ := 0.07
  let facial_cost : ℝ := 60
  let facial_discount : ℝ := 0.15
  let tip_rate : ℝ := 0.2

  let hair_total : ℝ := (updo_cost * (1 - updo_discount)) * (1 + tip_rate)
  let nails_total : ℝ := (manicure_cost + pedicure_cost * pedicure_discount) * (1 + tip_rate)
  let makeup_total : ℝ := (makeup_cost * (1 + makeup_tax)) * (1 + tip_rate)
  let facial_total : ℝ := (facial_cost * (1 - facial_discount)) * (1 + tip_rate)

  hair_total + nails_total + makeup_total + facial_total

/-- Theorem stating that the total cost of prom services for Keesha is $223.56 -/
theorem total_prom_cost_is_correct : total_prom_cost = 223.56 := by
  sorry

end NUMINAMATH_CALUDE_total_prom_cost_is_correct_l1341_134138


namespace NUMINAMATH_CALUDE_pascal_triangle_row17_element5_l1341_134102

theorem pascal_triangle_row17_element5 : Nat.choose 17 4 = 2380 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row17_element5_l1341_134102


namespace NUMINAMATH_CALUDE_smallest_valid_purchase_l1341_134117

def is_valid_purchase (n : ℕ) : Prop :=
  n % 12 = 0 ∧ n % 10 = 0 ∧ n % 9 = 0 ∧ n % 8 = 0 ∧
  n % 18 = 0 ∧ n % 24 = 0 ∧ n % 20 = 0 ∧ n % 30 = 0

theorem smallest_valid_purchase :
  ∃ (n : ℕ), is_valid_purchase n ∧ ∀ (m : ℕ), is_valid_purchase m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_purchase_l1341_134117


namespace NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_sum_of_fraction_parts_l1341_134136

/-- Recurrence relation for sequences without three consecutive 1s -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element binary sequence not containing three consecutive 1s -/
theorem probability_no_three_consecutive_ones : 
  (b 12 : ℚ) / 2^12 = 927 / 4096 := by sorry

/-- The sum of numerator and denominator of the probability fraction -/
theorem sum_of_fraction_parts : 927 + 4096 = 5023 := by sorry

end NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_sum_of_fraction_parts_l1341_134136


namespace NUMINAMATH_CALUDE_sin_cos_derivative_l1341_134114

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos x ^ 2 - Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_l1341_134114


namespace NUMINAMATH_CALUDE_factorization_3y_squared_minus_12_l1341_134182

theorem factorization_3y_squared_minus_12 (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3y_squared_minus_12_l1341_134182


namespace NUMINAMATH_CALUDE_function_determination_l1341_134111

/-- Given a function f(x) = a^x + k, if f(1) = 3 and f(0) = 2, then f(x) = 2^x + 1 -/
theorem function_determination (a k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a^x + k) 
  (h2 : f 1 = 3) 
  (h3 : f 0 = 2) : 
  ∀ x, f x = 2^x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l1341_134111


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1341_134142

theorem rotation_equivalence (y : ℝ) : 
  (450 % 360 : ℝ) = (360 - y) % 360 → y < 360 → y = 270 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1341_134142


namespace NUMINAMATH_CALUDE_substitution_elimination_l1341_134129

theorem substitution_elimination (x y : ℝ) : 
  (y = x - 5 ∧ 3*x - y = 8) → (3*x - x + 5 = 8) := by
  sorry

end NUMINAMATH_CALUDE_substitution_elimination_l1341_134129


namespace NUMINAMATH_CALUDE_apple_ratio_is_one_to_two_l1341_134170

/-- Represents the number of golden delicious apples needed for one pint of cider -/
def golden_delicious_per_pint : ℕ := 20

/-- Represents the number of pink lady apples needed for one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- Represents the number of farmhands -/
def num_farmhands : ℕ := 6

/-- Represents the number of apples a farmhand can pick per hour -/
def apples_per_hour : ℕ := 240

/-- Represents the number of hours worked -/
def hours_worked : ℕ := 5

/-- Represents the number of pints of cider that can be made -/
def pints_of_cider : ℕ := 120

/-- Theorem stating that the ratio of golden delicious apples to pink lady apples gathered is 1:2 -/
theorem apple_ratio_is_one_to_two :
  (golden_delicious_per_pint * pints_of_cider) / (pink_lady_per_pint * pints_of_cider) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_is_one_to_two_l1341_134170


namespace NUMINAMATH_CALUDE_ellipse_equation_l1341_134174

/-- Given an ellipse with foci on the x-axis, sum of major and minor axes equal to 10,
    and focal distance equal to 4√5, prove that its equation is x²/36 + y²/16 = 1. -/
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a + b = 10) (h4 : 2 * c = 4 * Real.sqrt 5) (h5 : a^2 - b^2 = c^2) :
  ∀ x y : ℝ, (x^2 / 36 + y^2 / 16 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1341_134174


namespace NUMINAMATH_CALUDE_wire_length_equals_49_l1341_134165

/-- The total length of a wire cut into two pieces forming a square and a regular octagon -/
def wire_length (square_side : ℝ) : ℝ :=
  4 * square_side

theorem wire_length_equals_49 (square_side : ℝ) (h1 : square_side = 7) :
  let octagon_side := (3 * wire_length square_side) / (8 * 4)
  let square_area := square_side ^ 2
  let octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side ^ 2
  square_area = octagon_area →
  wire_length square_side = 49 := by sorry

end NUMINAMATH_CALUDE_wire_length_equals_49_l1341_134165


namespace NUMINAMATH_CALUDE_min_area_rectangle_l1341_134183

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 84 → l * w ≥ 41 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l1341_134183


namespace NUMINAMATH_CALUDE_pony_discount_rate_l1341_134171

/-- Represents the discount rate for Fox jeans -/
def F : ℝ := sorry

/-- Represents the discount rate for Pony jeans -/
def P : ℝ := sorry

/-- Regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- Regular price of Pony jeans -/
def pony_price : ℝ := 20

/-- Number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- Number of Pony jeans purchased -/
def pony_count : ℕ := 2

/-- Total savings -/
def total_savings : ℝ := 9

/-- Sum of discount rates -/
def discount_sum : ℝ := 22

theorem pony_discount_rate :
  F + P = discount_sum ∧
  (fox_count * fox_price * F / 100 + pony_count * pony_price * P / 100 = total_savings) →
  P = 18 := by sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l1341_134171


namespace NUMINAMATH_CALUDE_sam_goal_impossible_l1341_134168

theorem sam_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (a_grades : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 40 →
  a_grades = 26 →
  ¬∃ (remaining_non_a : ℕ), 
    (a_grades + (total_quizzes - completed_quizzes - remaining_non_a) : ℚ) / total_quizzes ≥ goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_sam_goal_impossible_l1341_134168


namespace NUMINAMATH_CALUDE_spaghetti_cost_l1341_134109

def hamburger_cost : ℝ := 3
def fries_cost : ℝ := 1.20
def soda_cost : ℝ := 0.5
def num_hamburgers : ℕ := 5
def num_fries : ℕ := 4
def num_sodas : ℕ := 5
def num_friends : ℕ := 5
def individual_payment : ℝ := 5

theorem spaghetti_cost : 
  ∃ (spaghetti_price : ℝ),
    spaghetti_price = 
      num_friends * individual_payment - 
      (num_hamburgers * hamburger_cost + 
       num_fries * fries_cost + 
       num_sodas * soda_cost) ∧
    spaghetti_price = 2.70 :=
sorry

end NUMINAMATH_CALUDE_spaghetti_cost_l1341_134109


namespace NUMINAMATH_CALUDE_ice_water_masses_l1341_134151

/-- Proof of initial ice and water masses in a cylindrical vessel --/
theorem ice_water_masses
  (S : ℝ) (ρw ρi : ℝ) (hf Δh : ℝ)
  (h_S : S = 15)
  (h_ρw : ρw = 1)
  (h_ρi : ρi = 0.9)
  (h_hf : hf = 115)
  (h_Δh : Δh = 5) :
  ∃ (m_ice m_water : ℝ),
    m_ice = 675 ∧
    m_water = 1050 ∧
    m_ice / ρi - m_ice / ρw = S * Δh ∧
    m_water = ρw * S * hf - m_ice :=
by sorry

end NUMINAMATH_CALUDE_ice_water_masses_l1341_134151


namespace NUMINAMATH_CALUDE_bellas_bistro_purchase_l1341_134161

/-- The cost of a sandwich at Bella's Bistro -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Bella's Bistro -/
def soda_cost : ℕ := 1

/-- The number of sandwiches to be purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas to be purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase at Bella's Bistro -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem bellas_bistro_purchase :
  total_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_bellas_bistro_purchase_l1341_134161


namespace NUMINAMATH_CALUDE_heartsuit_calculation_l1341_134149

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calculation : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_calculation_l1341_134149


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1341_134107

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 4 →
  c = Real.sqrt 6 →
  C = π / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = 5 * π / 12 ∧
  a = 1 + Real.sqrt 3 ∧
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1341_134107


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1341_134116

open Set
open Real

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the modified quadratic function
def g (a c x : ℝ) := a * x^2 + 2*x + 4*c

-- Define the linear function
def h (m x : ℝ) := x + m

theorem quadratic_inequality_solution (a c : ℝ) :
  (∀ x, x ∈ solution_set ↔ f a c x > 0) →
  (∀ x, g a c x > 0 → h m x > 0) →
  (∃ x, h m x > 0 ∧ g a c x ≤ 0) →
  (a = -1/4 ∧ c = -3/4) ∧ (∀ m', m' ≥ -2 ↔ m' ≥ m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1341_134116


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1341_134153

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a * b = 3) : 
  a * b^2 - a^2 * b = -15 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1341_134153


namespace NUMINAMATH_CALUDE_mode_is_131_l1341_134173

/- Define the structure of a stem-and-leaf plot entry -/
structure StemLeafEntry :=
  (stem : ℕ)
  (leaves : List ℕ)

/- Define the stem-and-leaf plot -/
def stemLeafPlot : List StemLeafEntry := [
  ⟨9, [5, 5, 6]⟩,
  ⟨10, [4, 8]⟩,
  ⟨11, [2, 2, 2, 6, 6, 7]⟩,
  ⟨12, [0, 0, 3, 7, 7, 7]⟩,
  ⟨13, [1, 1, 1, 1]⟩,
  ⟨14, [5, 9]⟩
]

/- Define a function to calculate the mode -/
def calculateMode (plot : List StemLeafEntry) : ℕ :=
  sorry

/- Theorem stating that the mode of the given stem-and-leaf plot is 131 -/
theorem mode_is_131 : calculateMode stemLeafPlot = 131 :=
  sorry

end NUMINAMATH_CALUDE_mode_is_131_l1341_134173


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1341_134195

-- Define the inequality
def inequality (x : ℝ) : Prop := |2*x - 1| < 1

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1341_134195


namespace NUMINAMATH_CALUDE_max_points_is_168_l1341_134190

/-- Represents the number of cards of each color chosen by Vasya -/
structure CardChoice where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total points for a given card choice -/
def calculatePoints (choice : CardChoice) : ℕ :=
  choice.red + 2 * choice.red * choice.blue + 3 * choice.blue * choice.yellow

/-- Theorem: The maximum number of points Vasya can earn is 168 -/
theorem max_points_is_168 : 
  ∃ (choice : CardChoice), 
    choice.red + choice.blue + choice.yellow = 15 ∧ 
    choice.red ≤ 15 ∧ choice.blue ≤ 15 ∧ choice.yellow ≤ 15 ∧
    calculatePoints choice = 168 ∧
    ∀ (other : CardChoice), 
      other.red + other.blue + other.yellow = 15 → 
      other.red ≤ 15 ∧ other.blue ≤ 15 ∧ other.yellow ≤ 15 →
      calculatePoints other ≤ 168 := by
  sorry


end NUMINAMATH_CALUDE_max_points_is_168_l1341_134190


namespace NUMINAMATH_CALUDE_product_abcd_is_zero_l1341_134166

theorem product_abcd_is_zero 
  (a b c d : ℤ) 
  (eq1 : 2*a + 3*b + 5*c + 7*d = 34)
  (eq2 : 3*(d + c) = b)
  (eq3 : 3*b + c = a)
  (eq4 : c - 1 = d) :
  a * b * c * d = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_abcd_is_zero_l1341_134166


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l1341_134154

theorem sum_of_fifth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 4)
  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l1341_134154
