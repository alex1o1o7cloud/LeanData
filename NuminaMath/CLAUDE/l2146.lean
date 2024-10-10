import Mathlib

namespace soup_feeding_theorem_l2146_214606

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (totalCans : ℕ) (canCapacity : SoupCan) (childrenFed : ℕ) : ℕ :=
  let cansUsedForChildren := (childrenFed + canCapacity.children - 1) / canCapacity.children
  let remainingCans := totalCans - cansUsedForChildren
  remainingCans * canCapacity.adults

/-- Theorem stating that given 10 cans of soup, where each can feeds 4 adults or 6 children,
    if 30 children are fed, the remaining soup can feed 20 adults -/
theorem soup_feeding_theorem (totalCans : ℕ) (canCapacity : SoupCan) (childrenFed : ℕ) :
  totalCans = 10 →
  canCapacity.adults = 4 →
  canCapacity.children = 6 →
  childrenFed = 30 →
  remainingAdults totalCans canCapacity childrenFed = 20 := by
  sorry

end soup_feeding_theorem_l2146_214606


namespace smallest_fraction_divides_exactly_l2146_214625

def fraction1 : Rat := 6 / 7
def fraction2 : Rat := 5 / 14
def fraction3 : Rat := 10 / 21
def smallestFraction : Rat := 1 / 42

theorem smallest_fraction_divides_exactly :
  (∃ (n1 n2 n3 : ℕ), fraction1 * n1 = smallestFraction ∧
                     fraction2 * n2 = smallestFraction ∧
                     fraction3 * n3 = smallestFraction) ∧
  (∀ (f : Rat), f > 0 ∧ (∃ (m1 m2 m3 : ℕ), fraction1 * m1 = f ∧
                                           fraction2 * m2 = f ∧
                                           fraction3 * m3 = f) →
                f ≥ smallestFraction) :=
by sorry

end smallest_fraction_divides_exactly_l2146_214625


namespace coefficient_x_squared_proof_l2146_214683

/-- The coefficient of x² in the expansion of (2x³ + 5x² - 3x)(3x² - 5x + 1) -/
def coefficient_x_squared : ℤ := 20

/-- The first polynomial in the product -/
def poly1 (x : ℚ) : ℚ := 2 * x^3 + 5 * x^2 - 3 * x

/-- The second polynomial in the product -/
def poly2 (x : ℚ) : ℚ := 3 * x^2 - 5 * x + 1

theorem coefficient_x_squared_proof :
  ∃ (a b c d e f : ℚ),
    poly1 x * poly2 x = a * x^5 + b * x^4 + c * x^3 + coefficient_x_squared * x^2 + e * x + f :=
by sorry

end coefficient_x_squared_proof_l2146_214683


namespace students_history_not_statistics_l2146_214657

/-- Given a group of students, prove the number taking history but not statistics -/
theorem students_history_not_statistics 
  (total : ℕ) 
  (history : ℕ) 
  (statistics : ℕ) 
  (history_or_statistics : ℕ) 
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 30)
  (h_history_or_statistics : history_or_statistics = 59) :
  history - (history + statistics - history_or_statistics) = 29 := by
  sorry

end students_history_not_statistics_l2146_214657


namespace binary_addition_subtraction_l2146_214613

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits. -/
def binary (bits : List Bool) : ℕ := binaryToNat bits

theorem binary_addition_subtraction :
  let a := binary [true, true, true, false, true]  -- 11101₂
  let b := binary [true, true, false, true]        -- 1101₂
  let c := binary [true, false, true, true, false] -- 10110₂
  let d := binary [true, false, true, true]        -- 1011₂
  let result := binary [true, true, false, true, true] -- 11011₂
  a + b - c + d = result := by sorry

end binary_addition_subtraction_l2146_214613


namespace fruit_eating_permutations_l2146_214602

theorem fruit_eating_permutations :
  let total_fruits : ℕ := 4 + 2 + 1
  let apple_count : ℕ := 4
  let orange_count : ℕ := 2
  let banana_count : ℕ := 1
  (Nat.factorial total_fruits) / 
  (Nat.factorial apple_count * Nat.factorial orange_count * Nat.factorial banana_count) = 105 := by
sorry

end fruit_eating_permutations_l2146_214602


namespace tangent_circles_m_values_l2146_214664

/-- Definition of circle C1 -/
def C1 (m x y : ℝ) : Prop := (x - m)^2 + (y + 2)^2 = 9

/-- Definition of circle C2 -/
def C2 (m x y : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 4

/-- C1 is tangent to C2 from the inside -/
def is_tangent_inside (m : ℝ) : Prop :=
  ∃ x y : ℝ, C1 m x y ∧ C2 m x y ∧
  ∀ x' y' : ℝ, C1 m x' y' → C2 m x' y' → (x = x' ∧ y = y')

/-- The theorem to be proved -/
theorem tangent_circles_m_values :
  ∀ m : ℝ, is_tangent_inside m ↔ (m = -2 ∨ m = -1) :=
sorry

end tangent_circles_m_values_l2146_214664


namespace complex_sum_equals_z_l2146_214615

theorem complex_sum_equals_z (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 = z := by
  sorry

end complex_sum_equals_z_l2146_214615


namespace complex_multiplication_l2146_214693

/-- Given that i is the imaginary unit, prove that i(3-4i) = 4 + 3i -/
theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (3 - 4*i) = 4 + 3*i := by
  sorry

end complex_multiplication_l2146_214693


namespace sally_initial_cards_l2146_214661

def initial_cards : ℕ := 27
def cards_from_dan : ℕ := 41
def cards_bought : ℕ := 20
def total_cards : ℕ := 88

theorem sally_initial_cards : 
  initial_cards + cards_from_dan + cards_bought = total_cards :=
by sorry

end sally_initial_cards_l2146_214661


namespace other_coin_denomination_l2146_214653

theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h1 : total_coins = 342)
  (h2 : total_value = 7100)  -- 71 Rs in paise
  (h3 : twenty_paise_coins = 290) :
  (total_value - 20 * twenty_paise_coins) / (total_coins - twenty_paise_coins) = 25 := by
sorry

end other_coin_denomination_l2146_214653


namespace point_in_fourth_quadrant_l2146_214609

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the point P
structure Point where
  x : Real
  y : Real

-- Define the theorem
theorem point_in_fourth_quadrant (abc : Triangle) (p : Point) :
  abc.A > Real.pi / 2 →  -- Angle A is obtuse
  p.x = Real.tan abc.B →  -- x-coordinate is tan B
  p.y = Real.cos abc.A →  -- y-coordinate is cos A
  p.x > 0 ∧ p.y < 0  -- Point is in fourth quadrant
:= by sorry

end point_in_fourth_quadrant_l2146_214609


namespace smallest_even_with_repeated_seven_l2146_214627

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_repeated_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ p ^ k ∣ n

theorem smallest_even_with_repeated_seven :
  ∀ n : ℕ, 
    is_even n ∧ 
    has_repeated_prime_factor n 7 → 
    n ≥ 98 :=
by
  sorry

end smallest_even_with_repeated_seven_l2146_214627


namespace max_min_difference_l2146_214647

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Define the interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧
               (∀ x ∈ I, m ≤ f x) ∧
               (M - m = 32) :=
sorry

end max_min_difference_l2146_214647


namespace geometric_sum_first_eight_l2146_214639

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end geometric_sum_first_eight_l2146_214639


namespace sum_of_angles_less_than_90_degrees_l2146_214687

/-- A line intersecting two perpendicular planes forms angles α and β with these planes. -/
structure LineIntersectingPerpendicularPlanes where
  α : Real
  β : Real

/-- The theorem states that the sum of angles α and β is always less than 90 degrees. -/
theorem sum_of_angles_less_than_90_degrees (l : LineIntersectingPerpendicularPlanes) :
  l.α + l.β < 90 * Real.pi / 180 := by
  sorry

end sum_of_angles_less_than_90_degrees_l2146_214687


namespace sum_of_consecutive_integers_l2146_214614

theorem sum_of_consecutive_integers (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (c = 7) → (a + b + c = 18) := by
  sorry

end sum_of_consecutive_integers_l2146_214614


namespace radio_selling_price_l2146_214651

def purchase_price : ℚ := 232
def overhead_expenses : ℚ := 15
def profit_percent : ℚ := 21.457489878542503

def total_cost_price : ℚ := purchase_price + overhead_expenses

def profit_amount : ℚ := (profit_percent / 100) * total_cost_price

def selling_price : ℚ := total_cost_price + profit_amount

theorem radio_selling_price : 
  ∃ (sp : ℚ), sp = selling_price ∧ round sp = 300 := by sorry

end radio_selling_price_l2146_214651


namespace strawberry_pies_l2146_214638

def christine_strawberries : ℕ := 10
def rachel_strawberries : ℕ := 2 * christine_strawberries
def strawberries_per_pie : ℕ := 3

theorem strawberry_pies : 
  (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 :=
by sorry

end strawberry_pies_l2146_214638


namespace probability_three_of_a_kind_after_reroll_l2146_214695

/-- The probability of getting at least three dice showing the same value after re-rolling the unmatched die in a specific dice configuration. -/
theorem probability_three_of_a_kind_after_reroll (n : ℕ) (p : ℚ) : 
  n = 5 → -- number of dice
  p = 1 / 3 → -- probability we want to prove
  ∃ (X Y : ℕ), -- the two pair values
    X ≠ Y ∧ 
    X ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧ 
    Y ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
  p = (1 : ℚ) / 6 + (1 : ℚ) / 6 := by
  sorry


end probability_three_of_a_kind_after_reroll_l2146_214695


namespace morse_alphabet_size_l2146_214678

/-- The number of signals in each letter -/
def signal_length : Nat := 7

/-- The number of possible signals (dot and dash) -/
def signal_types : Nat := 2

/-- The number of possible alterations for each sequence (including the original) -/
def alterations_per_sequence : Nat := signal_length + 1

/-- The total number of possible sequences -/
def total_sequences : Nat := signal_types ^ signal_length

/-- The maximum number of unique letters in the alphabet -/
def max_letters : Nat := total_sequences / alterations_per_sequence

theorem morse_alphabet_size :
  max_letters = 16 := by sorry

end morse_alphabet_size_l2146_214678


namespace democrat_ratio_l2146_214660

/-- Proves that the ratio of democrats to total participants is 1:3 given the specified conditions -/
theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 990 →
  female_democrats = 165 →
  (∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    2 * female_democrats = female_participants ∧
    4 * female_democrats = male_participants) →
  (3 : ℚ) * (female_democrats + female_democrats) = total_participants := by
  sorry


end democrat_ratio_l2146_214660


namespace average_problem_l2146_214667

theorem average_problem (x : ℝ) : (15 + 25 + 35 + x) / 4 = 30 → x = 45 := by
  sorry

end average_problem_l2146_214667


namespace inequality_solution_l2146_214672

theorem inequality_solution : 
  ∀ x y : ℝ, y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y → 
  ((x = 0 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2)) ∧
  (x = 0 ∧ y = 0 → y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y) ∧
  (x = 1/2 ∧ y = 1/2 → y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y) :=
by sorry

end inequality_solution_l2146_214672


namespace macaroon_problem_l2146_214688

/-- Proves that the initial number of macaroons is 12 given the problem conditions -/
theorem macaroon_problem (weight_per_macaroon : ℕ) (num_bags : ℕ) (remaining_weight : ℕ) : 
  weight_per_macaroon = 5 →
  num_bags = 4 →
  remaining_weight = 45 →
  ∃ (initial_macaroons : ℕ),
    initial_macaroons = 12 ∧
    initial_macaroons % num_bags = 0 ∧
    (initial_macaroons / num_bags) * weight_per_macaroon * (num_bags - 1) = remaining_weight :=
by sorry

end macaroon_problem_l2146_214688


namespace correct_mark_calculation_l2146_214634

theorem correct_mark_calculation (n : ℕ) (initial_avg final_avg wrong_mark : ℚ) :
  n = 30 →
  initial_avg = 60 →
  wrong_mark = 90 →
  final_avg = 57.5 →
  (n : ℚ) * initial_avg - wrong_mark + ((n : ℚ) * final_avg - (n : ℚ) * initial_avg + wrong_mark) = 15 :=
by sorry

end correct_mark_calculation_l2146_214634


namespace fraction_17_39_415th_digit_l2146_214663

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

def nth_digit (n d k : ℕ) : ℕ :=
  sorry

theorem fraction_17_39_415th_digit :
  nth_digit 17 39 415 = 4 := by
  sorry

end fraction_17_39_415th_digit_l2146_214663


namespace set_condition_l2146_214676

theorem set_condition (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ↔ x^2 - 2*x ≠ 3 := by
  sorry

end set_condition_l2146_214676


namespace solution_set_inequality_l2146_214674

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | 3 * x - x^2 ≥ 0} = Set.Icc 0 3 := by sorry

end solution_set_inequality_l2146_214674


namespace sector_area_l2146_214633

/-- Given a circular sector with central angle 60° and arc length 4, its area is 24/π -/
theorem sector_area (r : ℝ) : 
  (π / 3 : ℝ) = 4 / r →   -- Central angle in radians = Arc length / radius
  (1 / 2) * r^2 * (π / 3) = 24 / π := by
sorry

end sector_area_l2146_214633


namespace simon_fraction_of_alvin_age_l2146_214668

/-- Given that Alvin is 30 years old and Simon is 10 years old, prove that Simon will be 3/7 of Alvin's age in 5 years. -/
theorem simon_fraction_of_alvin_age (alvin_age : ℕ) (simon_age : ℕ) : 
  alvin_age = 30 → simon_age = 10 → (simon_age + 5 : ℚ) / (alvin_age + 5 : ℚ) = 3/7 := by
  sorry

end simon_fraction_of_alvin_age_l2146_214668


namespace recurrence_is_geometric_iff_first_two_equal_l2146_214628

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℝ) : Prop :=
  (∀ n, b n > 0) ∧ (∀ n, b (n + 2) = 3 * b n * b (n + 1))

/-- A geometric progression -/
def IsGeometricProgression (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, b (n + 1) = r * b n

/-- The main theorem -/
theorem recurrence_is_geometric_iff_first_two_equal
    (b : ℕ → ℝ) (h : RecurrenceSequence b) :
    IsGeometricProgression b ↔ b 1 = b 2 := by
  sorry

end recurrence_is_geometric_iff_first_two_equal_l2146_214628


namespace equation_solution_l2146_214692

theorem equation_solution :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^3 - 3*x₁*y₁^2 = 2007) ∧ (y₁^3 - 3*x₁^2*y₁ = 2006) ∧
    (x₂^3 - 3*x₂*y₂^2 = 2007) ∧ (y₂^3 - 3*x₂^2*y₂ = 2006) ∧
    (x₃^3 - 3*x₃*y₃^2 = 2007) ∧ (y₃^3 - 3*x₃^2*y₃ = 2006) →
    (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1003 := by
  sorry

end equation_solution_l2146_214692


namespace difference_of_numbers_l2146_214641

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) :
  |x - y| = 2 * Real.sqrt 65 := by
sorry

end difference_of_numbers_l2146_214641


namespace tangent_line_equation_l2146_214648

/-- The equation of a line passing through (3,4) and tangent to x^2 + y^2 = 25 is 3x + 4y - 25 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (x^2 + y^2 = 25) →  -- Circle equation
  ((3:ℝ)^2 + 4^2 = 25) →  -- Point (3,4) lies on the circle
  (∃ k : ℝ, y - 4 = k * (x - 3)) →  -- Line passes through (3,4)
  (∀ p : ℝ × ℝ, p.1^2 + p.2^2 = 25 → (3 * p.1 + 4 * p.2 - 25 = 0 → p = (3, 4))) →  -- Line touches circle at only one point
  (3 * x + 4 * y - 25 = 0) -- Equation of the tangent line
:= by sorry

end tangent_line_equation_l2146_214648


namespace max_value_quadratic_l2146_214608

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (z : ℝ), z = x^2 + 3*x*y + 2*y^2 ∧ z ≤ 120 - 30*Real.sqrt 3 ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 - 2*x'*y' + 3*y'^2 = 10 ∧
  x'^2 + 3*x'*y' + 2*y'^2 = 120 - 30*Real.sqrt 3 := by
  sorry

end max_value_quadratic_l2146_214608


namespace quadratic_minimum_l2146_214626

theorem quadratic_minimum : 
  (∃ (x : ℝ), x^2 + 12*x + 9 = -27) ∧ 
  (∀ (x : ℝ), x^2 + 12*x + 9 ≥ -27) := by
sorry

end quadratic_minimum_l2146_214626


namespace min_tries_for_blue_and_yellow_is_nine_l2146_214654

/-- Represents the number of balls of each color in the box -/
structure BoxContents where
  purple : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the minimum number of tries to get one blue and one yellow ball -/
def minTriesForBlueAndYellow (box : BoxContents) : Nat :=
  box.purple + 2

/-- Theorem stating the minimum number of tries for the given box contents -/
theorem min_tries_for_blue_and_yellow_is_nine :
  let box : BoxContents := { purple := 7, blue := 5, yellow := 11 }
  minTriesForBlueAndYellow box = 9 := by
  sorry


end min_tries_for_blue_and_yellow_is_nine_l2146_214654


namespace correct_average_l2146_214652

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) 
  (misread1 misread2 misread3 : ℚ) 
  (correct1 correct2 correct3 : ℚ) :
  n = 15 ∧ 
  incorrect_avg = 62 ∧
  misread1 = 30 ∧ correct1 = 90 ∧
  misread2 = 60 ∧ correct2 = 120 ∧
  misread3 = 25 ∧ correct3 = 75 →
  (n : ℚ) * incorrect_avg + (correct1 - misread1) + (correct2 - misread2) + (correct3 - misread3) = n * (73 + 1/3) :=
by sorry

end correct_average_l2146_214652


namespace f_of_four_equals_thirteen_l2146_214690

/-- Given a function f where f(2x) = 3x^2 + 1 for all x, prove that f(4) = 13 -/
theorem f_of_four_equals_thirteen (f : ℝ → ℝ) (h : ∀ x, f (2 * x) = 3 * x^2 + 1) : 
  f 4 = 13 := by
  sorry

end f_of_four_equals_thirteen_l2146_214690


namespace area_ratio_is_nine_thirtytwo_l2146_214691

/-- Triangle XYZ with points G, H, I on its sides -/
structure TriangleXYZ where
  /-- Length of side XY -/
  xy : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- Length of side ZX -/
  zx : ℝ
  /-- Ratio of XG to XY -/
  s : ℝ
  /-- Ratio of YH to YZ -/
  t : ℝ
  /-- Ratio of ZI to ZX -/
  u : ℝ
  /-- XY length is 14 -/
  xy_eq : xy = 14
  /-- YZ length is 16 -/
  yz_eq : yz = 16
  /-- ZX length is 18 -/
  zx_eq : zx = 18
  /-- s is positive -/
  s_pos : s > 0
  /-- t is positive -/
  t_pos : t > 0
  /-- u is positive -/
  u_pos : u > 0
  /-- Sum of s, t, u is 3/4 -/
  sum_stu : s + t + u = 3/4
  /-- Sum of squares of s, t, u is 3/8 -/
  sum_sq_stu : s^2 + t^2 + u^2 = 3/8

/-- The ratio of the area of triangle GHI to the area of triangle XYZ -/
def areaRatio (T : TriangleXYZ) : ℝ :=
  1 - T.s * (1 - T.u) - T.t * (1 - T.s) - T.u * (1 - T.t)

theorem area_ratio_is_nine_thirtytwo (T : TriangleXYZ) : 
  areaRatio T = 9/32 := by
  sorry

end area_ratio_is_nine_thirtytwo_l2146_214691


namespace best_play_wins_probability_best_play_always_wins_more_than_two_plays_l2146_214622

/-- The probability that the best play wins in a competition with two plays -/
def probability_best_play_wins (n : ℕ) : ℚ :=
  1 - (n.factorial * n.factorial : ℚ) / ((2 * n).factorial : ℚ)

/-- The setup of the competition -/
structure Competition :=
  (n : ℕ)  -- number of students in each play
  (honest_mothers : ℕ)  -- number of mothers voting honestly
  (biased_mothers : ℕ)  -- number of mothers voting for their child's play

/-- The conditions of the competition -/
def competition_conditions (c : Competition) : Prop :=
  c.honest_mothers = c.n ∧ c.biased_mothers = c.n

/-- The theorem stating the probability of the best play winning -/
theorem best_play_wins_probability (c : Competition) 
  (h : competition_conditions c) : 
  probability_best_play_wins c.n = 1 - (c.n.factorial * c.n.factorial : ℚ) / ((2 * c.n).factorial : ℚ) :=
sorry

/-- For more than two plays, the best play always wins -/
theorem best_play_always_wins_more_than_two_plays (c : Competition) (s : ℕ) 
  (h1 : competition_conditions c) (h2 : s > 2) : 
  probability_best_play_wins c.n = 1 :=
sorry

end best_play_wins_probability_best_play_always_wins_more_than_two_plays_l2146_214622


namespace kaleb_chocolate_bars_l2146_214686

/-- The number of chocolate bars Kaleb needs to sell -/
def total_chocolate_bars (bars_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  bars_per_box * num_boxes

/-- Theorem stating the total number of chocolate bars Kaleb needs to sell -/
theorem kaleb_chocolate_bars :
  total_chocolate_bars 5 142 = 710 := by
  sorry

end kaleb_chocolate_bars_l2146_214686


namespace dan_picked_nine_limes_l2146_214684

/-- The number of limes Dan has now -/
def total_limes : ℕ := 13

/-- The number of limes Sara gave to Dan -/
def sara_limes : ℕ := 4

/-- The number of limes Dan picked -/
def dan_picked_limes : ℕ := total_limes - sara_limes

theorem dan_picked_nine_limes : dan_picked_limes = 9 := by sorry

end dan_picked_nine_limes_l2146_214684


namespace root_in_interval_l2146_214682

-- Define the function
def f (x : ℝ) := x^3 - 2*x - 1

-- State the theorem
theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) →
  (∃ x ∈ Set.Ioo (3/2) 2, f x = 0) :=
by sorry

end root_in_interval_l2146_214682


namespace lions_escaped_l2146_214679

/-- The number of rhinos that escaped -/
def rhinos : ℕ := 2

/-- The time (in hours) to recover each animal -/
def recovery_time : ℕ := 2

/-- The total time (in hours) spent recovering all animals -/
def total_time : ℕ := 10

/-- The number of lions that escaped -/
def lions : ℕ := (total_time - rhinos * recovery_time) / recovery_time

theorem lions_escaped :
  lions = 3 :=
by sorry

end lions_escaped_l2146_214679


namespace sin_210_degrees_l2146_214629

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_degrees_l2146_214629


namespace batsman_average_l2146_214665

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  (previous_total = 16 * previous_average) →
  (previous_total + 85 = 17 * (previous_average + 3)) →
  (previous_average + 3 = 37) := by
  sorry

end batsman_average_l2146_214665


namespace sin_585_degrees_l2146_214635

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l2146_214635


namespace business_school_class_l2146_214698

theorem business_school_class (p q r s : ℕ+) 
  (h_product : p * q * r * s = 1365)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s) : 
  p + q + r + s = 28 := by
  sorry

end business_school_class_l2146_214698


namespace nonzero_terms_count_l2146_214685

/-- The number of nonzero terms in the expansion of (x^2+2)(3x^3+2x^2+4)-4(x^4+x^3-3x) -/
theorem nonzero_terms_count : ∃ (p : Polynomial ℝ), 
  p = (X^2 + 2) * (3*X^3 + 2*X^2 + 4) - 4*(X^4 + X^3 - 3*X) ∧ 
  p.support.card = 6 := by
  sorry

end nonzero_terms_count_l2146_214685


namespace reversed_digits_multiple_l2146_214650

/-- Given a two-digit number that is k times the sum of its digits, 
    prove that the number formed by reversing its digits is (11 - k) times the sum of its digits. -/
theorem reversed_digits_multiple (k : ℕ) (u v : ℕ) : 
  (u ≤ 9 ∧ v ≤ 9 ∧ u ≠ 0) → 
  (10 * u + v = k * (u + v)) → 
  (10 * v + u = (11 - k) * (u + v)) :=
by sorry

end reversed_digits_multiple_l2146_214650


namespace tangent_line_equation_minimum_value_maximum_value_l2146_214681

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ y - f 1 = f' 1 * (x - 1) := by sorry

-- Theorem for the minimum value
theorem minimum_value : 
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -14 ∧ ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≥ f x := by sorry

-- Theorem for the maximum value
theorem maximum_value : 
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 18 ∧ ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≤ f x := by sorry

end tangent_line_equation_minimum_value_maximum_value_l2146_214681


namespace inverse_composition_equals_one_third_l2146_214605

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x - 2) / 3

-- Theorem statement
theorem inverse_composition_equals_one_third :
  g_inv (g_inv 11) = 1/3 := by
  sorry

end inverse_composition_equals_one_third_l2146_214605


namespace alphabet_letter_count_l2146_214697

/-- The number of letters in an alphabet with specific dot and line properties -/
theorem alphabet_letter_count :
  let dot_and_line : ℕ := 13  -- Letters with both dot and line
  let line_only : ℕ := 24     -- Letters with line but no dot
  let dot_only : ℕ := 3       -- Letters with dot but no line
  let total : ℕ := dot_and_line + line_only + dot_only
  total = 40 := by sorry

end alphabet_letter_count_l2146_214697


namespace kickball_total_players_kickball_problem_l2146_214696

theorem kickball_total_players (wed_morning : ℕ) (wed_afternoon_increase : ℕ) 
  (thu_morning_decrease : ℕ) (thu_lunchtime_decrease : ℕ) : ℕ :=
  let wed_afternoon := wed_morning + wed_afternoon_increase
  let thu_morning := wed_morning - thu_morning_decrease
  let thu_afternoon := thu_morning - thu_lunchtime_decrease
  let wed_total := wed_morning + wed_afternoon
  let thu_total := thu_morning + thu_afternoon
  wed_total + thu_total

theorem kickball_problem :
  kickball_total_players 37 15 9 7 = 138 := by
  sorry

end kickball_total_players_kickball_problem_l2146_214696


namespace intersection_A_B_l2146_214659

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ t : ℝ, x = t^2 + 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x * (x - 1) = 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l2146_214659


namespace limit_of_f_is_one_fourth_l2146_214617

def C (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

def f (n : ℕ) : ℚ := (C n 2 : ℚ) / (2 * n^2 + n : ℚ)

theorem limit_of_f_is_one_fourth :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n - 1/4| < ε :=
sorry

end limit_of_f_is_one_fourth_l2146_214617


namespace quadratic_square_solutions_l2146_214623

theorem quadratic_square_solutions (n : ℕ) : 
  ∃ (p q : ℤ), ∃ (S : Finset ℤ), 
    (Finset.card S = n) ∧ 
    (∀ x : ℤ, x ∈ S ↔ ∃ y : ℕ, x^2 + p * x + q = y^2) ∧
    (∀ x y : ℤ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) :=
by sorry

end quadratic_square_solutions_l2146_214623


namespace unique_solution_l2146_214658

-- Define the range of numbers
def valid_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

-- Define primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the conditions of the problem
structure DrawResult where
  alice : ℕ
  bob : ℕ
  alice_valid : valid_number alice
  bob_valid : valid_number bob
  alice_uncertain : ∀ n, valid_number n → n ≠ alice → (n < alice ∨ n > alice)
  bob_certain : bob < alice ∨ bob > alice
  bob_prime : is_prime bob
  product_multiple_of_10 : (alice * bob) % 10 = 0
  perfect_square : ∃ k : ℕ, 100 * bob + alice = k * k

-- Theorem statement
theorem unique_solution (d : DrawResult) : d.alice = 29 ∧ d.bob = 5 ∧ d.alice + d.bob = 34 := by
  sorry

end unique_solution_l2146_214658


namespace triangle_square_perimeter_difference_l2146_214612

theorem triangle_square_perimeter_difference (d : ℕ) : 
  (∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    3 * a - 4 * b = 1989 ∧ 
    a - b = d ∧ 
    4 * b > 0) ↔ 
  d > 663 :=
sorry

end triangle_square_perimeter_difference_l2146_214612


namespace max_xy_value_l2146_214669

theorem max_xy_value (x y : ℕ+) (h : 5 * x + 3 * y = 100) : x * y ≤ 165 := by
  sorry

end max_xy_value_l2146_214669


namespace value_of_x_l2146_214675

theorem value_of_x (x y z : ℝ) : 
  x = (1/2) * y → 
  y = (1/4) * z → 
  z = 80 → 
  x = 10 := by
sorry

end value_of_x_l2146_214675


namespace cube_with_holes_surface_area_l2146_214640

/-- Calculates the total surface area of a cube with holes --/
def totalSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let internalSurface := 6 * 4 * holeEdge^2
  originalSurface - holeArea + internalSurface

/-- The problem statement --/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 5 2 = 222 := by
  sorry

end cube_with_holes_surface_area_l2146_214640


namespace tangent_line_slope_positive_l2146_214637

/-- Given a function f: ℝ → ℝ, if the tangent line to the curve y = f(x) at the point (2, f(2)) 
    passes through the point (-1, 2), then f'(2) > 0. -/
theorem tangent_line_slope_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (deriv f 2) * (2 - (-1)) = f 2 - 2) : 
  deriv f 2 > 0 := by
  sorry


end tangent_line_slope_positive_l2146_214637


namespace largest_digit_divisible_by_six_l2146_214604

theorem largest_digit_divisible_by_six :
  ∀ M : ℕ, M ≤ 9 →
    (54320 + M).mod 6 = 0 →
    ∀ N : ℕ, N ≤ 9 → N > M →
      (54320 + N).mod 6 ≠ 0 →
    M = 4 :=
by sorry

end largest_digit_divisible_by_six_l2146_214604


namespace problem_solution_l2146_214642

theorem problem_solution : (3/4)^2017 * (-1-1/3)^2018 = 4/3 := by
  sorry

end problem_solution_l2146_214642


namespace smallest_in_row_10_n_squared_minus_n_and_2n_in_row_largest_n_not_including_n_squared_minus_10n_l2146_214666

/-- Predicate defining whether an integer m is in Row n -/
def in_row (n : ℕ) (m : ℕ) : Prop :=
  m % n = 0 ∧ m ≤ n^2 ∧ ∀ k < n, ¬in_row k m

theorem smallest_in_row_10 :
  ∀ m, in_row 10 m → m ≥ 10 :=
sorry

theorem n_squared_minus_n_and_2n_in_row (n : ℕ) (h : n ≥ 3) :
  in_row n (n^2 - n) ∧ in_row n (n^2 - 2*n) :=
sorry

theorem largest_n_not_including_n_squared_minus_10n :
  ∀ n > 9, in_row n (n^2 - 10*n) :=
sorry

end smallest_in_row_10_n_squared_minus_n_and_2n_in_row_largest_n_not_including_n_squared_minus_10n_l2146_214666


namespace jim_bike_shop_profit_l2146_214600

/-- Represents Jim's bike shop financials for a month -/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repair_cost : ℕ
  complex_repairs_count : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ

/-- Calculates the total profit of the bike shop -/
def total_profit (shop : BikeShop) : ℤ :=
  (shop.tire_repair_price - shop.tire_repair_cost) * shop.tire_repairs_count +
  (shop.complex_repair_price - shop.complex_repair_cost) * shop.complex_repairs_count +
  shop.retail_profit - shop.fixed_expenses

/-- Theorem stating that Jim's bike shop profit is $3000 -/
theorem jim_bike_shop_profit :
  ∃ (shop : BikeShop),
    shop.tire_repair_price = 20 ∧
    shop.tire_repair_cost = 5 ∧
    shop.tire_repairs_count = 300 ∧
    shop.complex_repair_price = 300 ∧
    shop.complex_repair_cost = 50 ∧
    shop.complex_repairs_count = 2 ∧
    shop.retail_profit = 2000 ∧
    shop.fixed_expenses = 4000 ∧
    total_profit shop = 3000 := by
  sorry

end jim_bike_shop_profit_l2146_214600


namespace linear_dependency_condition_l2146_214646

-- Define the vectors
def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (k : ℝ) : Fin 2 → ℝ := ![1, k]

-- Define linear dependency
def is_linearly_dependent (v1 v2 : Fin 2 → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a • v1 + b • v2 = 0)

-- Theorem statement
theorem linear_dependency_condition (k : ℝ) :
  is_linearly_dependent v1 (v2 k) ↔ k = 2 :=
sorry

end linear_dependency_condition_l2146_214646


namespace A_three_times_faster_than_B_l2146_214645

/-- The work rate of A -/
def work_rate_A : ℚ := 1 / 16

/-- The work rate of B -/
def work_rate_B : ℚ := 1 / 12 - 1 / 16

/-- The theorem stating that A is 3 times faster than B -/
theorem A_three_times_faster_than_B : work_rate_A / work_rate_B = 3 := by
  sorry

end A_three_times_faster_than_B_l2146_214645


namespace box_height_proof_l2146_214601

/-- Proves that the height of boxes is 12 inches given the specified conditions. -/
theorem box_height_proof (box_length : ℝ) (box_width : ℝ) (total_volume : ℝ) 
  (cost_per_box : ℝ) (min_spend : ℝ) (h : ℝ) : 
  box_length = 20 → 
  box_width = 20 → 
  total_volume = 2160000 → 
  cost_per_box = 0.4 → 
  min_spend = 180 → 
  (total_volume / (box_length * box_width * h)) * cost_per_box = min_spend → 
  h = 12 := by
  sorry

end box_height_proof_l2146_214601


namespace five_pq_is_odd_l2146_214610

theorem five_pq_is_odd (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) : 
  Odd (5 * p * q) := by
  sorry

end five_pq_is_odd_l2146_214610


namespace chicken_rabbit_problem_has_unique_solution_l2146_214689

/-- Represents the number of chickens and rabbits in a cage. -/
structure AnimalCount where
  chickens : ℕ
  rabbits : ℕ

/-- Checks if the given animal count satisfies the problem conditions. -/
def satisfiesConditions (count : AnimalCount) : Prop :=
  count.chickens = 2 * (4 * count.rabbits) - 5 ∧
  2 * count.chickens + count.rabbits = 92

/-- There exists a unique solution to the chicken and rabbit problem. -/
theorem chicken_rabbit_problem_has_unique_solution :
  ∃! count : AnimalCount, satisfiesConditions count :=
sorry

end chicken_rabbit_problem_has_unique_solution_l2146_214689


namespace average_weight_of_class_class_average_weight_l2146_214649

theorem average_weight_of_class (group1_count : ℕ) (group1_avg : ℚ) 
                                 (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count : ℕ := group1_count + group2_count
  let total_weight : ℚ := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count

theorem class_average_weight :
  average_weight_of_class 26 (50.25 : ℚ) 8 (45.15 : ℚ) = (49.05 : ℚ) := by
  sorry

end average_weight_of_class_class_average_weight_l2146_214649


namespace last_digit_of_one_over_two_to_twelve_l2146_214632

theorem last_digit_of_one_over_two_to_twelve (n : ℕ) : 
  n = 12 → (1 : ℚ) / (2 ^ n) * 10^n % 10 = 5 := by sorry

end last_digit_of_one_over_two_to_twelve_l2146_214632


namespace banana_arrangements_l2146_214630

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- Theorem: The number of unique arrangements of "BANANA" is 60 -/
theorem banana_arrangements :
  uniqueArrangements 6 [3, 2, 1] = 60 := by
  sorry

end banana_arrangements_l2146_214630


namespace unique_reverse_multiple_of_nine_l2146_214656

/-- A function that checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

/-- A function that reverses the digits of a number -/
def reverseDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that 10989 is the only five-digit number
    that when multiplied by 9, results in its reverse -/
theorem unique_reverse_multiple_of_nine :
  ∀ n : ℕ, isFiveDigit n → (9 * n = reverseDigits n) → n = 10989 :=
sorry

end unique_reverse_multiple_of_nine_l2146_214656


namespace field_length_calculation_l2146_214616

theorem field_length_calculation (w : ℝ) (l : ℝ) : 
  l = 2 * w →  -- length is double the width
  25 = (1/8) * (l * w) →  -- pond area (5^2) is 1/8 of field area
  l = 20 := by
  sorry

end field_length_calculation_l2146_214616


namespace age_difference_l2146_214621

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : a = c + 10 :=
by sorry

end age_difference_l2146_214621


namespace kimberly_skittles_bought_l2146_214620

/-- The number of Skittles Kimberly bought -/
def skittles_bought : ℕ := sorry

/-- Kimberly's initial number of Skittles -/
def initial_skittles : ℕ := 5

/-- Kimberly's total number of Skittles after buying more -/
def total_skittles : ℕ := 12

theorem kimberly_skittles_bought :
  skittles_bought = total_skittles - initial_skittles :=
sorry

end kimberly_skittles_bought_l2146_214620


namespace triangle_is_right_angled_l2146_214677

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def complementary (t : Triangle) : Prop :=
  t.A + t.B = 90

def pythagorean (t : Triangle) : Prop :=
  (t.a + t.b) * (t.a - t.b) = t.c^2

def angle_ratio (t : Triangle) : Prop :=
  t.A / t.B = 1 / 2 ∧ t.A / t.C = 1

-- Theorem statement
theorem triangle_is_right_angled (t : Triangle) 
  (h1 : complementary t) 
  (h2 : pythagorean t) 
  (h3 : angle_ratio t) : 
  t.A = 45 ∧ t.B = 90 ∧ t.C = 45 :=
sorry

end triangle_is_right_angled_l2146_214677


namespace factorization_xy_squared_minus_16x_l2146_214619

theorem factorization_xy_squared_minus_16x (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := by
  sorry

end factorization_xy_squared_minus_16x_l2146_214619


namespace fixed_point_power_function_l2146_214670

theorem fixed_point_power_function (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 4 = 2 →
  f 16 = 4 := by
sorry

end fixed_point_power_function_l2146_214670


namespace lg_meaningful_iff_first_or_second_quadrant_l2146_214618

open Real

-- Define the meaningful condition for lg(cos θ · tan θ)
def is_meaningful (θ : ℝ) : Prop :=
  sin θ > 0 ∧ sin θ ≠ 1

-- Define the first and second quadrants
def in_first_or_second_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π

-- Theorem statement
theorem lg_meaningful_iff_first_or_second_quadrant (θ : ℝ) :
  is_meaningful θ ↔ in_first_or_second_quadrant θ :=
sorry

end lg_meaningful_iff_first_or_second_quadrant_l2146_214618


namespace normal_symmetry_l2146_214631

/-- A random variable with normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a normal random variable is less than or equal to a given value -/
def normalCDF (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

theorem normal_symmetry (X : NormalRandomVariable) (a : ℝ) :
  normalCDF X 0 = 1 - normalCDF X (a - 2) → a = 6 := by
  sorry

end normal_symmetry_l2146_214631


namespace tank_volume_ratio_l2146_214636

theorem tank_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end tank_volume_ratio_l2146_214636


namespace min_distinct_values_l2146_214644

/-- Given a list of 2023 positive integers with a unique mode occurring 15 times,
    the minimum number of distinct values is 146 -/
theorem min_distinct_values (l : List ℕ+) : 
  l.length = 2023 →
  ∃! m : ℕ+, (l.count m = 15 ∧ ∀ n : ℕ+, l.count n ≤ 15) →
  (∀ k : ℕ+, l.count k = 15 → k = m) →
  (Finset.card l.toFinset : ℕ) ≥ 146 ∧ 
  ∃ l' : List ℕ+, l'.length = 2023 ∧ 
    (∃! m' : ℕ+, (l'.count m' = 15 ∧ ∀ n : ℕ+, l'.count n ≤ 15)) ∧
    (Finset.card l'.toFinset : ℕ) = 146 :=
by
  sorry

end min_distinct_values_l2146_214644


namespace abs_neg_three_halves_l2146_214643

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end abs_neg_three_halves_l2146_214643


namespace fraction_integer_iff_p_range_l2146_214655

theorem fraction_integer_iff_p_range (p : ℕ+) :
  (∃ (k : ℕ+), (3 * p.val + 25 : ℤ) = k.val * (2 * p.val - 5)) ↔ 3 ≤ p.val ∧ p.val ≤ 35 := by
  sorry

end fraction_integer_iff_p_range_l2146_214655


namespace disease_cases_2005_2015_l2146_214603

/-- Calculates the number of disease cases in a given year, assuming a linear decrease. -/
def cases_in_year (initial_year initial_cases final_year final_cases target_year : ℕ) : ℕ :=
  initial_cases - (initial_cases - final_cases) * (target_year - initial_year) / (final_year - initial_year)

/-- Theorem stating the number of disease cases in 2005 and 2015 given the conditions. -/
theorem disease_cases_2005_2015 :
  cases_in_year 1970 300000 2020 100 2005 = 90070 ∧
  cases_in_year 1970 300000 2020 100 2015 = 30090 :=
by
  sorry

#eval cases_in_year 1970 300000 2020 100 2005
#eval cases_in_year 1970 300000 2020 100 2015

end disease_cases_2005_2015_l2146_214603


namespace point_P_quadrants_l2146_214671

def is_root (x : ℝ) : Prop := (2 * x - 1) * (x + 1) = 0

def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_P_quadrants :
  ∃ (x y : ℝ), (is_root x ∧ is_root y) →
    (in_second_quadrant x y ∨ in_fourth_quadrant x y) ∧
    ¬(in_second_quadrant x y ∧ in_fourth_quadrant x y) :=
sorry

end point_P_quadrants_l2146_214671


namespace geometric_sequence_common_ratio_l2146_214624

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_incr : increasing_sequence a)
  (h_first : a 1 = -2)
  (h_relation : ∀ n : ℕ, 3 * (a n + a (n + 2)) = 10 * a (n + 1)) :
  ∃ q : ℝ, q = 1/3 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end geometric_sequence_common_ratio_l2146_214624


namespace area_between_concentric_circles_l2146_214607

theorem area_between_concentric_circles 
  (r : Real) -- radius of inner circle
  (h1 : r > 0) -- inner radius is positive
  (h2 : 3 * r - r = 4) -- difference between outer and inner radii is 4
  : π * (3 * r)^2 - π * r^2 = 32 * π := by
  sorry

end area_between_concentric_circles_l2146_214607


namespace equation_proof_l2146_214699

theorem equation_proof : 300 * 2 + (12 + 4) * (1 / 8) = 602 := by
  sorry

end equation_proof_l2146_214699


namespace asterisk_value_l2146_214662

theorem asterisk_value : ∃ x : ℚ, (63 / 21) * (x / 189) = 1 ∧ x = 63 := by
  sorry

end asterisk_value_l2146_214662


namespace patsy_guests_l2146_214611

-- Define the problem parameters
def appetizers_per_guest : ℕ := 6
def initial_dozens : ℕ := 3 + 2 + 2
def additional_dozens : ℕ := 8

-- Define the theorem
theorem patsy_guests :
  (((initial_dozens + additional_dozens) * 12) / appetizers_per_guest) = 30 := by
  sorry

end patsy_guests_l2146_214611


namespace five_digit_number_divisible_by_37_and_173_l2146_214694

theorem five_digit_number_divisible_by_37_and_173 (n : ℕ) : 
  (n ≥ 10000 ∧ n < 100000) →  -- five-digit number
  n % 37 = 0 →  -- divisible by 37
  n % 173 = 0 →  -- divisible by 173
  (n / 1000) % 10 = 3 →  -- thousands digit is 3
  (n / 100) % 10 = 2  -- hundreds digit is 2
  := by sorry

end five_digit_number_divisible_by_37_and_173_l2146_214694


namespace function_satisfying_conditions_l2146_214673

theorem function_satisfying_conditions (f : ℚ → ℚ) 
  (h1 : f 0 = 0) 
  (h2 : ∀ x y : ℚ, f (f x + f y) = x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
  sorry

end function_satisfying_conditions_l2146_214673


namespace smallest_nonprime_with_conditions_l2146_214680

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_nonprime_with_conditions (n : ℕ) : 
  n = 289 ↔ 
    (¬is_prime n ∧ 
     n > 25 ∧ 
     has_no_prime_factor_less_than n 15 ∧ 
     sum_of_digits n > 10 ∧
     ∀ m : ℕ, m < n → 
       (¬is_prime m → 
        m ≤ 25 ∨ 
        ¬has_no_prime_factor_less_than m 15 ∨ 
        sum_of_digits m ≤ 10)) :=
by sorry

end smallest_nonprime_with_conditions_l2146_214680
