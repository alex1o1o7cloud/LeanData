import Mathlib

namespace NUMINAMATH_CALUDE_parabola_directrix_l1202_120235

/-- The directrix of the parabola x = -1/4 * y^2 is the line x = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (x = -1/4 * y^2) → (∃ (p : ℝ), p = 1 ∧ 
    (∀ (x₀ y₀ : ℝ), x₀ = -1/4 * y₀^2 → 
      ((x₀ + 1)^2 + y₀^2 = (x₀ - p)^2))) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1202_120235


namespace NUMINAMATH_CALUDE_player_a_wins_with_9_balls_l1202_120223

/-- A game where two players take turns picking 1 to 3 balls, and the player who picks the last ball wins. -/
def BallGame (n : ℕ) :=
  { strategy : ℕ → ℕ // 
    (∀ k, 1 ≤ strategy k ∧ strategy k ≤ 3) ∧ 
    (∀ m, m ≤ n → m > 0 → ∃ k, m - strategy k = 0) }

/-- Player A has a winning strategy when there are 9 balls. -/
theorem player_a_wins_with_9_balls : 
  ∃ (strategy : BallGame 9), True :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_with_9_balls_l1202_120223


namespace NUMINAMATH_CALUDE_total_pens_bought_l1202_120233

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) 
  (h1 : pen_cost > 10)
  (h2 : masha_spent = 357)
  (h3 : olya_spent = 441)
  (h4 : masha_spent % pen_cost = 0)
  (h5 : olya_spent % pen_cost = 0) :
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
sorry

end NUMINAMATH_CALUDE_total_pens_bought_l1202_120233


namespace NUMINAMATH_CALUDE_percentage_exceeding_speed_limit_l1202_120247

/-- Given a road where:
  * 10% of motorists receive speeding tickets
  * 60% of motorists who exceed the speed limit do not receive tickets
  Prove that 25% of motorists exceed the speed limit -/
theorem percentage_exceeding_speed_limit
  (total_motorists : ℝ)
  (h_positive : total_motorists > 0)
  (ticketed_percentage : ℝ)
  (h_ticketed : ticketed_percentage = 0.1)
  (non_ticketed_speeders_percentage : ℝ)
  (h_non_ticketed : non_ticketed_speeders_percentage = 0.6)
  : (ticketed_percentage * total_motorists) / (1 - non_ticketed_speeders_percentage) / total_motorists = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_exceeding_speed_limit_l1202_120247


namespace NUMINAMATH_CALUDE_coordinates_of_C_l1202_120289

-- Define the points
def A : ℝ × ℝ := (7, 2)
def B : ℝ × ℝ := (-1, 9)
def D : ℝ × ℝ := (2, 7)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC (isosceles triangle)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on line BC
  (D.1 - B.1) * (C.2 - B.2) = (D.2 - B.2) * (C.1 - B.1) ∧
  -- AD is perpendicular to BC (altitude condition)
  (A.1 - D.1) * (C.1 - B.1) + (A.2 - D.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem coordinates_of_C :
  ∃ (C : ℝ × ℝ), triangle_ABC C ∧ C = (5, 5) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_C_l1202_120289


namespace NUMINAMATH_CALUDE_final_prob_is_three_fourths_l1202_120230

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the probability of drawing a black ball from the bag -/
def probBlack (bag : BagContents) : ℚ :=
  bag.black / (bag.white + bag.black + bag.red)

/-- The initial contents of the bag -/
def initialBag : BagContents := ⟨2, 3, 5⟩

/-- The number of additional black balls added -/
def additionalBlackBalls : ℕ := 18

/-- The contents of the bag after adding additional black balls -/
def finalBag : BagContents := ⟨initialBag.white, initialBag.black + additionalBlackBalls, initialBag.red⟩

/-- Theorem stating that the probability of drawing a black ball from the final bag is 3/4 -/
theorem final_prob_is_three_fourths : probBlack finalBag = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_final_prob_is_three_fourths_l1202_120230


namespace NUMINAMATH_CALUDE_work_left_after_nine_days_l1202_120274

/-- The fraction of work left after 9 days given the work rates of A, B, and C -/
theorem work_left_after_nine_days (a_rate b_rate c_rate : ℚ) : 
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  c_rate = 1 / 25 →
  let combined_rate := a_rate + b_rate + c_rate
  let work_done_first_four_days := 4 * combined_rate
  let ac_rate := a_rate + c_rate
  let work_done_next_five_days := 5 * ac_rate
  let total_work_done := work_done_first_four_days + work_done_next_five_days
  total_work_done ≥ 1 := by sorry

#check work_left_after_nine_days

end NUMINAMATH_CALUDE_work_left_after_nine_days_l1202_120274


namespace NUMINAMATH_CALUDE_anthony_pencils_count_l1202_120222

/-- Given Anthony's initial pencils and Kathryn's gift, calculate Anthony's total pencils -/
def anthonyTotalPencils (initialPencils giftedPencils : ℕ) : ℕ :=
  initialPencils + giftedPencils

/-- Theorem: Anthony's total pencils is 65 given the initial conditions -/
theorem anthony_pencils_count :
  anthonyTotalPencils 9 56 = 65 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_count_l1202_120222


namespace NUMINAMATH_CALUDE_concert_ticket_price_l1202_120268

/-- Proves that the cost of each ticket is $30 given the concert conditions --/
theorem concert_ticket_price :
  ∀ (ticket_price : ℝ),
    (500 : ℝ) * ticket_price * 0.7 = (4 : ℝ) * 2625 →
    ticket_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l1202_120268


namespace NUMINAMATH_CALUDE_karabases_more_numerous_l1202_120259

/-- Represents the inhabitants of Perra-Terra -/
inductive Inhabitant
  | Karabas
  | Barabas

/-- The number of acquaintances each type of inhabitant has -/
def acquaintances (i : Inhabitant) : Nat × Nat :=
  match i with
  | Inhabitant.Karabas => (6, 9)  -- (Karabases, Barabases)
  | Inhabitant.Barabas => (10, 7) -- (Karabases, Barabases)

theorem karabases_more_numerous :
  ∃ (K B : Nat), K > B ∧
  K * (acquaintances Inhabitant.Karabas).2 = B * (acquaintances Inhabitant.Barabas).1 :=
by sorry

end NUMINAMATH_CALUDE_karabases_more_numerous_l1202_120259


namespace NUMINAMATH_CALUDE_juice_bar_solution_l1202_120298

/-- Represents the juice bar problem --/
def juice_bar_problem (total_spent : ℕ) (mango_price : ℕ) (pineapple_price : ℕ) (pineapple_spent : ℕ) : Prop :=
  ∃ (mango_count pineapple_count : ℕ),
    mango_count * mango_price + pineapple_count * pineapple_price = total_spent ∧
    pineapple_count * pineapple_price = pineapple_spent ∧
    mango_count + pineapple_count = 17

/-- The theorem stating the solution to the juice bar problem --/
theorem juice_bar_solution :
  juice_bar_problem 94 5 6 54 :=
sorry

end NUMINAMATH_CALUDE_juice_bar_solution_l1202_120298


namespace NUMINAMATH_CALUDE_parabola_focus_l1202_120204

/-- A parabola is defined by the equation y² = -16x + 64. -/
def parabola (x y : ℝ) : Prop := y^2 = -16*x + 64

/-- The focus of a parabola is a point on its axis of symmetry. -/
def is_focus (x y : ℝ) : Prop := sorry

/-- The focus of the parabola y² = -16x + 64 is at (0, 0). -/
theorem parabola_focus :
  is_focus 0 0 ∧ ∀ x y, parabola x y → is_focus x y → x = 0 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1202_120204


namespace NUMINAMATH_CALUDE_distance_O_to_J_l1202_120275

/-- A right triangle with its circumcircle and incircle -/
structure RightTriangleWithCircles where
  /-- The center of the circumcircle -/
  O : ℝ × ℝ
  /-- The center of the incircle -/
  I : ℝ × ℝ
  /-- The radius of the circumcircle -/
  R : ℝ
  /-- The radius of the incircle -/
  r : ℝ
  /-- The vertex of the right angle -/
  C : ℝ × ℝ
  /-- The point symmetric to C with respect to I -/
  J : ℝ × ℝ
  /-- Ensure that C is the right angle vertex -/
  right_angle : (C.1 - O.1)^2 + (C.2 - O.2)^2 = R^2
  /-- Ensure that J is symmetric to C with respect to I -/
  symmetry : J.1 - I.1 = I.1 - C.1 ∧ J.2 - I.2 = I.2 - C.2

/-- The theorem to be proved -/
theorem distance_O_to_J (t : RightTriangleWithCircles) : 
  ((t.O.1 - t.J.1)^2 + (t.O.2 - t.J.2)^2)^(1/2) = t.R - 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_distance_O_to_J_l1202_120275


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_two_l1202_120250

theorem trigonometric_expression_equals_two (α : ℝ) : 
  (Real.sin (π + α))^2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_two_l1202_120250


namespace NUMINAMATH_CALUDE_unique_common_term_l1202_120221

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem unique_common_term : ∀ n : ℕ, x n = y n → x n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_common_term_l1202_120221


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l1202_120277

theorem simplify_and_ratio (m : ℝ) : ∃ (c d : ℝ), 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l1202_120277


namespace NUMINAMATH_CALUDE_trundic_word_count_l1202_120206

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 15

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of required letters (A and B) -/
def required_letters : ℕ := 2

/-- Calculates the number of valid words in the Trundic language -/
def count_valid_words (alphabet_size : ℕ) (max_word_length : ℕ) (required_letters : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid words in the Trundic language -/
theorem trundic_word_count :
  count_valid_words alphabet_size max_word_length required_letters = 35180 :=
sorry

end NUMINAMATH_CALUDE_trundic_word_count_l1202_120206


namespace NUMINAMATH_CALUDE_interest_problem_l1202_120254

/-- Given a sum P put at simple interest for 10 years, if increasing the interest rate
    by 5% results in Rs. 150 more interest, then P = 300. -/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 150 → P = 300 := by
  sorry


end NUMINAMATH_CALUDE_interest_problem_l1202_120254


namespace NUMINAMATH_CALUDE_base_27_to_3_conversion_l1202_120282

/-- Converts a single digit from base 27 to its three-digit representation in base 3 -/
def convert_digit_27_to_3 (d : Nat) : Nat × Nat × Nat :=
  (d / 9, (d % 9) / 3, d % 3)

/-- Converts a number from base 27 to base 3 -/
def convert_27_to_3 (n : Nat) : List Nat :=
  let digits := n.digits 27
  List.join (digits.map (fun d => let (a, b, c) := convert_digit_27_to_3 d; [a, b, c]))

theorem base_27_to_3_conversion :
  convert_27_to_3 652 = [0, 2, 0, 0, 1, 2, 0, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_27_to_3_conversion_l1202_120282


namespace NUMINAMATH_CALUDE_playground_children_count_l1202_120216

theorem playground_children_count :
  ∀ (girls boys : ℕ),
    girls = 28 →
    boys = 35 →
    girls + boys = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l1202_120216


namespace NUMINAMATH_CALUDE_magnitude_of_2a_plus_b_l1202_120200

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (0, -1, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 1)

-- Define the operation 2a + b
def result : ℝ × ℝ × ℝ := (2 * a.1 + b.1, 2 * a.2.1 + b.2.1, 2 * a.2.2 + b.2.2)

-- Theorem statement
theorem magnitude_of_2a_plus_b : 
  Real.sqrt ((result.1)^2 + (result.2.1)^2 + (result.2.2)^2) = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_2a_plus_b_l1202_120200


namespace NUMINAMATH_CALUDE_train_length_l1202_120288

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 27 ∧ time = 20 → speed * time * (5 / 18) = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1202_120288


namespace NUMINAMATH_CALUDE_calculation_21_implies_72_l1202_120284

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The calculation process described in the problem -/
def calculation (n : TwoDigitNumber) : Nat :=
  2 * (5 * n.units - 3) + n.tens

/-- Theorem stating that if the calculation result is 21, the original number is 72 -/
theorem calculation_21_implies_72 (n : TwoDigitNumber) :
  calculation n = 21 → n.tens = 7 ∧ n.units = 2 := by
  sorry

#eval calculation ⟨7, 2, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_calculation_21_implies_72_l1202_120284


namespace NUMINAMATH_CALUDE_smallest_k_proof_l1202_120208

/-- The smallest integer k for which x^2 - x + 2 - k = 0 has two distinct real roots -/
def smallest_k : ℕ := 2

/-- The quadratic equation x^2 - x + 2 - k = 0 -/
def quadratic (x k : ℝ) : Prop := x^2 - x + 2 - k = 0

theorem smallest_k_proof :
  (∀ k < smallest_k, ¬∃ x y : ℝ, x ≠ y ∧ quadratic x k ∧ quadratic y k) ∧
  (∃ x y : ℝ, x ≠ y ∧ quadratic x smallest_k ∧ quadratic y smallest_k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_proof_l1202_120208


namespace NUMINAMATH_CALUDE_parabola_equation_l1202_120232

/-- A parabola with vertex at the origin and axis of symmetry x = -4 has the standard equation y^2 = 16x -/
theorem parabola_equation (y x : ℝ) : 
  (∀ p : ℝ, p > 0 → y^2 = 2*p*x) → -- Standard form of parabola equation
  (∀ p : ℝ, -p/2 = -4) →           -- Axis of symmetry condition
  y^2 = 16*x :=                    -- Conclusion: standard equation
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1202_120232


namespace NUMINAMATH_CALUDE_quotient_calculation_l1202_120261

theorem quotient_calculation (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) 
  (h1 : dividend = 149)
  (h2 : divisor = 16)
  (h3 : remainder = 5)
  (h4 : dividend = divisor * 9 + remainder) :
  9 = dividend / divisor := by
sorry

end NUMINAMATH_CALUDE_quotient_calculation_l1202_120261


namespace NUMINAMATH_CALUDE_max_value_k_l1202_120234

theorem max_value_k (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧ 
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) := by
sorry

end NUMINAMATH_CALUDE_max_value_k_l1202_120234


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l1202_120212

theorem fourth_circle_radius (r₁ r₂ r : ℝ) (h₁ : r₁ = 17) (h₂ : r₂ = 27) :
  π * r^2 = π * (r₂^2 - r₁^2) → r = 2 * Real.sqrt 110 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l1202_120212


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1202_120238

theorem sum_of_x_and_y (x y S : ℝ) 
  (h1 : x + y = S) 
  (h2 : y - 3 * x = 7) 
  (h3 : y - x = 7.5) : 
  S = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1202_120238


namespace NUMINAMATH_CALUDE_inverse_function_property_l1202_120211

/-- Given two real-valued functions f and f_inv that are inverses of each other,
    prove that a + 2b = -3 --/
theorem inverse_function_property (a b : ℝ) 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + 2 * b)
  (h2 : ∀ x, f_inv x = b * x + 2 * a)
  (h3 : Function.LeftInverse f_inv f)
  (h4 : Function.RightInverse f_inv f) :
  a + 2 * b = -3 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1202_120211


namespace NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l1202_120278

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line equation
def given_line_equation (x y : ℝ) : Prop := x + y = 0

-- Define the resulting line equation
def result_line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of a circle
def circle_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - c.1)^2 + (y - c.2)^2 = 1

-- Define perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem line_through_circle_center_perpendicular_to_given_line :
  ∃ c : ℝ × ℝ,
    circle_center c circle_equation ∧
    (∃ m₁ m₂ : ℝ,
      (∀ x y, given_line_equation x y ↔ y = m₁ * x) ∧
      (∀ x y, result_line_equation x y ↔ y = m₂ * x + c.2) ∧
      perpendicular m₁ m₂) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l1202_120278


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1202_120256

theorem decimal_point_problem : ∃! (x : ℝ), x > 0 ∧ 10000 * x = 9 / x ∧ x = 0.03 := by sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1202_120256


namespace NUMINAMATH_CALUDE_existence_of_representation_l1202_120218

theorem existence_of_representation (m : ℤ) :
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_representation_l1202_120218


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l1202_120271

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧
  N = 23 ∧
  (∀ (k : ℕ), k > N → 
    (1743 % k = 2019 % k ∧ 2019 % k = 3008 % k) → false) ∧
  1743 % N = 2019 % N ∧ 2019 % N = 3008 % N :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l1202_120271


namespace NUMINAMATH_CALUDE_ticket_cost_after_30_years_l1202_120287

/-- The cost of a ticket to Mars after a given number of years, given an initial cost and a halving period --/
def ticket_cost (initial_cost : ℕ) (halving_period : ℕ) (years : ℕ) : ℕ :=
  initial_cost / (2 ^ (years / halving_period))

/-- Theorem stating that the cost of a ticket to Mars after 30 years is $125,000 --/
theorem ticket_cost_after_30_years :
  ticket_cost 1000000 10 30 = 125000 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_after_30_years_l1202_120287


namespace NUMINAMATH_CALUDE_symmetric_circle_l1202_120217

/-- Given a circle C and a line l, find the equation of the circle symmetric to C with respect to l -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x - y - 3 = 0 → 
    ∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ 
    (x - a)^2 + (y - b)^2 = x^2 + y^2 - 6*x + 6*y + 14) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_l1202_120217


namespace NUMINAMATH_CALUDE_number_equals_five_times_difference_l1202_120262

theorem number_equals_five_times_difference : ∃! x : ℝ, x = 5 * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_number_equals_five_times_difference_l1202_120262


namespace NUMINAMATH_CALUDE_height_growth_l1202_120228

theorem height_growth (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) : 
  current_height = 126 ∧ 
  growth_rate = 0.05 ∧ 
  current_height = previous_height * (1 + growth_rate) → 
  previous_height = 120 := by
sorry

end NUMINAMATH_CALUDE_height_growth_l1202_120228


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1202_120213

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (2 * m^2 + m - 1) (-m^2 - 2*m - 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1202_120213


namespace NUMINAMATH_CALUDE_complex_power_equality_l1202_120294

theorem complex_power_equality : (((1 + Complex.I) / (1 - Complex.I)) ^ 2016 = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l1202_120294


namespace NUMINAMATH_CALUDE_original_number_proof_l1202_120243

theorem original_number_proof : ∃ x : ℤ, (x + 24) % 27 = 0 ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1202_120243


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l1202_120231

theorem positive_numbers_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hnot_all_equal : ¬(a = b ∧ b = c)) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l1202_120231


namespace NUMINAMATH_CALUDE_exponent_division_l1202_120242

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1202_120242


namespace NUMINAMATH_CALUDE_intersects_x_axis_once_iff_m_range_l1202_120245

/-- The function f(x) = x³ - x² - x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + m

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem intersects_x_axis_once_iff_m_range (m : ℝ) :
  (∃! x, f m x = 0) ↔ (m < -5/27 ∨ m > 0) := by sorry

end NUMINAMATH_CALUDE_intersects_x_axis_once_iff_m_range_l1202_120245


namespace NUMINAMATH_CALUDE_stratified_sampling_correctness_problem_case_proof_l1202_120210

/-- Represents the number of students in each year and the total sample size. -/
structure SchoolData where
  totalStudents : ℕ
  freshmanStudents : ℕ
  sophomoreStudents : ℕ
  juniorStudents : ℕ
  sampleSize : ℕ

/-- Calculates the number of students to be sampled from a specific year. -/
def sampledStudents (data : SchoolData) (yearStudents : ℕ) : ℕ :=
  (yearStudents * data.sampleSize) / data.totalStudents

/-- Theorem stating that the sum of sampled students from each year equals the total sample size. -/
theorem stratified_sampling_correctness (data : SchoolData) 
    (h1 : data.totalStudents = data.freshmanStudents + data.sophomoreStudents + data.juniorStudents)
    (h2 : data.sampleSize ≤ data.totalStudents) :
  sampledStudents data data.freshmanStudents +
  sampledStudents data data.sophomoreStudents +
  sampledStudents data data.juniorStudents = data.sampleSize := by
  sorry

/-- Verifies the specific case given in the problem. -/
def verifyProblemCase : Prop :=
  let data : SchoolData := {
    totalStudents := 1200,
    freshmanStudents := 300,
    sophomoreStudents := 400,
    juniorStudents := 500,
    sampleSize := 60
  }
  sampledStudents data data.freshmanStudents = 15 ∧
  sampledStudents data data.sophomoreStudents = 20 ∧
  sampledStudents data data.juniorStudents = 25

/-- Proves the specific case given in the problem. -/
theorem problem_case_proof : verifyProblemCase := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correctness_problem_case_proof_l1202_120210


namespace NUMINAMATH_CALUDE_three_digit_permutation_sum_l1202_120207

/-- A three-digit number with no zeros -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ d, d ∣ n → d ≠ 0

/-- Sum of all distinct permutations of the digits of a number -/
def SumOfPermutations (n : ℕ) : ℕ := sorry

theorem three_digit_permutation_sum (n : ℕ) :
  ThreeDigitNumber n → SumOfPermutations n = 2775 → n = 889 ∨ n = 997 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_permutation_sum_l1202_120207


namespace NUMINAMATH_CALUDE_sector_area_l1202_120240

/-- Theorem: Area of a circular sector with central angle 2π/3 and arc length 2 --/
theorem sector_area (r : ℝ) (h1 : (2 * π / 3) * r = 2) : 
  (1 / 2) * r^2 * (2 * π / 3) = 3 / π := by
  sorry


end NUMINAMATH_CALUDE_sector_area_l1202_120240


namespace NUMINAMATH_CALUDE_apples_left_l1202_120249

/-- The number of bags with 20 apples each -/
def bags_20 : ℕ := 4

/-- The number of apples in each of the first type of bags -/
def apples_per_bag_20 : ℕ := 20

/-- The number of bags with 25 apples each -/
def bags_25 : ℕ := 6

/-- The number of apples in each of the second type of bags -/
def apples_per_bag_25 : ℕ := 25

/-- The number of apples Ella sells -/
def apples_sold : ℕ := 200

/-- The theorem stating that Ella has 30 apples left -/
theorem apples_left : 
  bags_20 * apples_per_bag_20 + bags_25 * apples_per_bag_25 - apples_sold = 30 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l1202_120249


namespace NUMINAMATH_CALUDE_unique_last_digit_for_multiple_of_6_l1202_120224

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem unique_last_digit_for_multiple_of_6 :
  ∃! d : ℕ, d < 10 ∧ is_multiple_of_6 (64310 + d) :=
by sorry

end NUMINAMATH_CALUDE_unique_last_digit_for_multiple_of_6_l1202_120224


namespace NUMINAMATH_CALUDE_second_number_value_l1202_120260

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 660 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a →
  b = 180 := by sorry

end NUMINAMATH_CALUDE_second_number_value_l1202_120260


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l1202_120237

def a : ℝ × ℝ × ℝ := (-1, 2, -3)
def b : ℝ × ℝ × ℝ := (-4, -1, 2)

theorem vector_b_magnitude : ‖b‖ = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l1202_120237


namespace NUMINAMATH_CALUDE_seven_twelfths_decimal_l1202_120267

theorem seven_twelfths_decimal : 7 / 12 = 0.5833333333333333 := by sorry

end NUMINAMATH_CALUDE_seven_twelfths_decimal_l1202_120267


namespace NUMINAMATH_CALUDE_square_difference_equality_l1202_120269

theorem square_difference_equality : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1202_120269


namespace NUMINAMATH_CALUDE_vector_b_components_l1202_120265

def vector_a : ℝ × ℝ := (2, -1)

theorem vector_b_components :
  ∀ (b : ℝ × ℝ),
  (∃ (k : ℝ), k < 0 ∧ b = (k * vector_a.1, k * vector_a.2)) →
  (b.1 * b.1 + b.2 * b.2 = 20) →
  b = (-4, 2) := by sorry

end NUMINAMATH_CALUDE_vector_b_components_l1202_120265


namespace NUMINAMATH_CALUDE_existence_of_composite_nx_plus_one_l1202_120225

theorem existence_of_composite_nx_plus_one (n : ℤ) : ∃ x : ℤ, ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n * x + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_composite_nx_plus_one_l1202_120225


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l1202_120270

/-- Given a 6-8-10 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 56π. -/
theorem sum_of_circle_areas (a b c : ℝ) (r s t : ℝ) : 
  a = 6 ∧ b = 8 ∧ c = 10 ∧  -- Triangle side lengths
  a^2 + b^2 = c^2 ∧         -- Right triangle condition
  r + s = a ∧              -- Circles are externally tangent
  r + t = b ∧
  s + t = c ∧
  r > 0 ∧ s > 0 ∧ t > 0 →   -- Radii are positive
  π * (r^2 + s^2 + t^2) = 56 * π :=
by sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l1202_120270


namespace NUMINAMATH_CALUDE_bert_spending_l1202_120296

theorem bert_spending (n : ℚ) : 
  (1/2 * ((3/4 * n) - 9)) = 15 → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_bert_spending_l1202_120296


namespace NUMINAMATH_CALUDE_intersection_range_l1202_120239

theorem intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ 
  -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1202_120239


namespace NUMINAMATH_CALUDE_range_of_a_for_full_range_l1202_120266

/-- Piecewise function f(x) defined by a real parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x - 1 else x^2 - 2 * a * x

/-- The range of f(x) is all real numbers -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The range of a for which f(x) has a full range is [2/3, +∞) -/
theorem range_of_a_for_full_range :
  {a : ℝ | has_full_range a} = {a : ℝ | a ≥ 2/3} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_full_range_l1202_120266


namespace NUMINAMATH_CALUDE_garage_sale_pants_price_l1202_120273

/-- Proves that the price of each pair of pants is $3 in Kekai's garage sale scenario --/
theorem garage_sale_pants_price (shirt_price : ℚ) (num_shirts num_pants : ℕ) (remaining_money : ℚ) :
  shirt_price = 1 →
  num_shirts = 5 →
  num_pants = 5 →
  remaining_money = 10 →
  ∃ (pants_price : ℚ),
    pants_price = 3 ∧
    remaining_money = (shirt_price * num_shirts + pants_price * num_pants) / 2 := by
  sorry


end NUMINAMATH_CALUDE_garage_sale_pants_price_l1202_120273


namespace NUMINAMATH_CALUDE_equation_decomposition_l1202_120214

-- Define the original equation
def original_equation (y z : ℝ) : Prop :=
  z^4 - 6*y^4 = 3*z^2 - 2

-- Define the hyperbola equation
def hyperbola_equation (y z : ℝ) : Prop :=
  z^2 - 3*y^2 = 2

-- Define the ellipse equation
def ellipse_equation (y z : ℝ) : Prop :=
  z^2 - 2*y^2 = 1

-- Theorem stating that the original equation can be decomposed into a hyperbola and an ellipse
theorem equation_decomposition :
  ∀ y z : ℝ, original_equation y z ↔ (hyperbola_equation y z ∨ ellipse_equation y z) :=
by sorry


end NUMINAMATH_CALUDE_equation_decomposition_l1202_120214


namespace NUMINAMATH_CALUDE_raccoon_lock_problem_l1202_120257

theorem raccoon_lock_problem :
  ∀ (x : ℝ),
  let first_lock_time := 5
  let second_lock_time := x * first_lock_time - 3
  let both_locks_time := 5 * second_lock_time
  both_locks_time = 60 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_raccoon_lock_problem_l1202_120257


namespace NUMINAMATH_CALUDE_lyndee_friends_l1202_120241

theorem lyndee_friends (total_pieces : ℕ) (lyndee_ate : ℕ) (friend_ate : ℕ) : 
  total_pieces = 11 → lyndee_ate = 1 → friend_ate = 2 → 
  (total_pieces - lyndee_ate) / friend_ate = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_lyndee_friends_l1202_120241


namespace NUMINAMATH_CALUDE_complex_number_location_l1202_120292

/-- Given a complex number z satisfying z * (-1 + 3*I) = 1 + 7*I,
    prove that z is located in the fourth quadrant of the complex plane. -/
theorem complex_number_location (z : ℂ) (h : z * (-1 + 3*I) = 1 + 7*I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1202_120292


namespace NUMINAMATH_CALUDE_contractor_problem_l1202_120201

/-- The number of days initially planned to complete the work -/
def initial_days : ℕ := 15

/-- The number of absent laborers -/
def absent_laborers : ℕ := 5

/-- The number of days taken to complete the work with reduced laborers -/
def actual_days : ℕ := 20

/-- The original number of laborers employed -/
def original_laborers : ℕ := 15

theorem contractor_problem :
  (original_laborers - absent_laborers) * initial_days = original_laborers * actual_days :=
sorry

end NUMINAMATH_CALUDE_contractor_problem_l1202_120201


namespace NUMINAMATH_CALUDE_problem_statement_l1202_120229

theorem problem_statement (a b m n : ℝ) : 
  a * m^2001 + b * n^2001 = 3 →
  a * m^2002 + b * n^2002 = 7 →
  a * m^2003 + b * n^2003 = 24 →
  a * m^2004 + b * n^2004 = 102 →
  m^2 * (n - 1) = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1202_120229


namespace NUMINAMATH_CALUDE_partner_capital_l1202_120236

/-- Given the profit distribution and profit rate change, calculate A's capital -/
theorem partner_capital (total_profit : ℝ) (a_profit_share : ℝ) (a_income_increase : ℝ) 
  (initial_rate : ℝ) (final_rate : ℝ) :
  (a_profit_share = 2/3) →
  (a_income_increase = 300) →
  (initial_rate = 0.05) →
  (final_rate = 0.07) →
  (a_income_increase = a_profit_share * total_profit * (final_rate - initial_rate)) →
  (∃ (a_capital : ℝ), a_capital = 300000 ∧ a_profit_share * total_profit = initial_rate * a_capital) :=
by sorry

end NUMINAMATH_CALUDE_partner_capital_l1202_120236


namespace NUMINAMATH_CALUDE_apple_value_in_cake_slices_l1202_120219

/-- Represents the value of one apple in terms of cake slices -/
def apple_value : ℚ := 15 / 4

/-- Represents the number of apples that can be traded for juice bottles -/
def apples_per_juice_trade : ℕ := 4

/-- Represents the number of juice bottles received in trade for apples -/
def juice_bottles_per_apple_trade : ℕ := 3

/-- Represents the number of cake slices that can be traded for one juice bottle -/
def cake_slices_per_juice_bottle : ℕ := 5

theorem apple_value_in_cake_slices :
  apple_value = (juice_bottles_per_apple_trade * cake_slices_per_juice_bottle : ℚ) / apples_per_juice_trade :=
sorry

#eval apple_value -- Should output 3.75

end NUMINAMATH_CALUDE_apple_value_in_cake_slices_l1202_120219


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1202_120276

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ (n : ℕ), n > 2 ∧ D = n * (n - 3) / 2 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1202_120276


namespace NUMINAMATH_CALUDE_fib_divisibility_implies_fib_number_l1202_120281

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Condition: For every positive integer m, there exists a positive integer n such that m | Fₙ - k -/
def condition (k : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → ∃ n : ℕ, n > 0 ∧ (fib n - k) % m = 0

/-- Main theorem: If the condition holds, then k is a Fibonacci number -/
theorem fib_divisibility_implies_fib_number (k : ℕ) (h : condition k) :
  ∃ n : ℕ, fib n = k :=
sorry

end NUMINAMATH_CALUDE_fib_divisibility_implies_fib_number_l1202_120281


namespace NUMINAMATH_CALUDE_ellipse_equation_l1202_120251

/-- An ellipse with a = 2b passing through point (2, 0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  a_eq_2b : a = 2 * b
  passes_through_2_0 : (2 : ℝ)^2 / (a^2) + 0^2 / (b^2) = 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / (e.a^2) + y^2 / (e.b^2) = 1

theorem ellipse_equation (e : Ellipse) : standard_equation e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1202_120251


namespace NUMINAMATH_CALUDE_almond_butter_servings_l1202_120248

/-- Represents the number of tablespoons in the container -/
def container_amount : ℚ := 35 + 2/3

/-- Represents the number of tablespoons in one serving -/
def serving_size : ℚ := 2 + 1/2

/-- Represents the number of servings in the container -/
def number_of_servings : ℚ := container_amount / serving_size

theorem almond_butter_servings : 
  ∃ (n : ℕ) (r : ℚ), 0 ≤ r ∧ r < 1 ∧ number_of_servings = n + r ∧ n = 14 ∧ r = 4/15 :=
sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l1202_120248


namespace NUMINAMATH_CALUDE_square_on_circle_radius_l1202_120263

theorem square_on_circle_radius (S : ℝ) (x : ℝ) (R : ℝ) : 
  S = 256 → -- Square area is 256 cm²
  x^2 = S → -- Side length of the square
  (x - R)^2 = R^2 - (x/2)^2 → -- Pythagoras theorem application
  R = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_on_circle_radius_l1202_120263


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_is_330_75_l1202_120264

/-- Triangle ABC with given side lengths and parallel lines forming a new triangle -/
structure TriangleWithParallelLines where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Lengths of segments formed by parallel lines
  ℓA_length : ℝ
  ℓB_length : ℝ
  ℓC_length : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  ℓA_positive : ℓA_length > 0
  ℓB_positive : ℓB_length > 0
  ℓC_positive : ℓC_length > 0
  triangle_inequality : AB + BC > AC ∧ BC + AC > AB ∧ AC + AB > BC
  ℓA_inside : ℓA_length < BC
  ℓB_inside : ℓB_length < AC
  ℓC_inside : ℓC_length < AB

/-- The perimeter of the triangle formed by parallel lines -/
def innerTrianglePerimeter (t : TriangleWithParallelLines) : ℝ :=
  sorry

/-- Theorem stating that for the given triangle and parallel lines, the inner triangle perimeter is 330.75 -/
theorem inner_triangle_perimeter_is_330_75 
  (t : TriangleWithParallelLines) 
  (h1 : t.AB = 150) 
  (h2 : t.BC = 270) 
  (h3 : t.AC = 210) 
  (h4 : t.ℓA_length = 65) 
  (h5 : t.ℓB_length = 60) 
  (h6 : t.ℓC_length = 20) : 
  innerTrianglePerimeter t = 330.75 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_is_330_75_l1202_120264


namespace NUMINAMATH_CALUDE_intersection_range_l1202_120279

theorem intersection_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ - 1 ∧ 
    y₂ = k * x₂ - 1 ∧ 
    x₁^2 - y₁^2 = 4 ∧ 
    x₂^2 - y₂^2 = 4 ∧ 
    x₁ > 0 ∧ 
    x₂ > 0) ↔ 
  (1 < k ∧ k < Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1202_120279


namespace NUMINAMATH_CALUDE_exponent_rule_l1202_120244

theorem exponent_rule (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l1202_120244


namespace NUMINAMATH_CALUDE_real_number_line_bijection_l1202_120209

-- Define the set of points on a number line
def NumberLine : Type := ℝ

-- State the theorem
theorem real_number_line_bijection : 
  ∃ f : ℝ → NumberLine, Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_l1202_120209


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1202_120252

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: Given a man's speed with the current of 21 km/hr and a current speed of 4.3 km/hr,
    the man's speed against the current is 12.4 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 21 4.3 = 12.4 := by
  sorry

#eval speed_against_current 21 4.3

end NUMINAMATH_CALUDE_mans_speed_against_current_l1202_120252


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1202_120280

/-- Given a quadratic equation px^2 + qx + r = 0 with roots u and v,
    prove that qu + r and qv + r are roots of px^2 - (2pr-q)x + (pr-q^2+qr) = 0 -/
theorem quadratic_root_transformation (p q r u v : ℝ) 
  (hu : p * u^2 + q * u + r = 0)
  (hv : p * v^2 + q * v + r = 0) :
  p * (q * u + r)^2 - (2 * p * r - q) * (q * u + r) + (p * r - q^2 + q * r) = 0 ∧
  p * (q * v + r)^2 - (2 * p * r - q) * (q * v + r) + (p * r - q^2 + q * r) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1202_120280


namespace NUMINAMATH_CALUDE_base_conversion_sum_approx_l1202_120205

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 6, 2]  -- 263 in base 8
def num2 : List Nat := [3, 1]     -- 13 in base 3
def num3 : List Nat := [3, 4, 2]  -- 243 in base 7
def num4 : List Nat := [5, 3]     -- 35 in base 6

-- State the theorem
theorem base_conversion_sum_approx :
  let x1 := baseToDecimal num1 8
  let x2 := baseToDecimal num2 3
  let x3 := baseToDecimal num3 7
  let x4 := baseToDecimal num4 6
  abs ((x1 / x2 + x3 / x4 : ℚ) - 35.442) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_approx_l1202_120205


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1202_120285

def A : Set Int := {1, 2}
def B : Set Int := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1202_120285


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1202_120297

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1202_120297


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1202_120253

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1202_120253


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1202_120202

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = 3 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1202_120202


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1202_120203

theorem cubic_roots_sum (u v w : ℝ) : 
  (u - Real.rpow 17 (1/3 : ℝ)) * (u - Real.rpow 67 (1/3 : ℝ)) * (u - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  (v - Real.rpow 17 (1/3 : ℝ)) * (v - Real.rpow 67 (1/3 : ℝ)) * (v - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  (w - Real.rpow 17 (1/3 : ℝ)) * (w - Real.rpow 67 (1/3 : ℝ)) * (w - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  u^3 + v^3 + w^3 = 221 + 6/5 - 3 * 1549 :=
by sorry


end NUMINAMATH_CALUDE_cubic_roots_sum_l1202_120203


namespace NUMINAMATH_CALUDE_gumball_solution_l1202_120295

/-- Represents the gumball distribution problem --/
def gumball_problem (total : ℕ) (todd : ℕ) (alisha : ℕ) (bobby : ℕ) : Prop :=
  total = 45 ∧
  todd = 4 ∧
  alisha = 2 * todd ∧
  bobby = 4 * alisha - 5 ∧
  total - (todd + alisha + bobby) = 6

/-- Theorem stating that the gumball problem has a solution --/
theorem gumball_solution : ∃ (total todd alisha bobby : ℕ), gumball_problem total todd alisha bobby :=
sorry

end NUMINAMATH_CALUDE_gumball_solution_l1202_120295


namespace NUMINAMATH_CALUDE_triangle_square_ratio_l1202_120215

/-- The ratio of the combined area to the combined perimeter of an equilateral triangle and a square -/
theorem triangle_square_ratio : 
  let triangle_side : ℝ := 10
  let triangle_altitude : ℝ := triangle_side * (Real.sqrt 3 / 2)
  let square_side : ℝ := triangle_altitude / 2
  let triangle_area : ℝ := (1 / 2) * triangle_side * triangle_altitude
  let square_area : ℝ := square_side ^ 2
  let combined_area : ℝ := triangle_area + square_area
  let triangle_perimeter : ℝ := 3 * triangle_side
  let square_perimeter : ℝ := 4 * square_side
  let combined_perimeter : ℝ := triangle_perimeter + square_perimeter
  combined_area / combined_perimeter = (25 * Real.sqrt 3 + 18.75) / (30 + 10 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_square_ratio_l1202_120215


namespace NUMINAMATH_CALUDE_settlement_area_theorem_l1202_120286

/-- Represents the lengths of the sides of the fields and forest -/
structure SettlementGeometry where
  r : ℝ  -- Length of the side of the square field
  p : ℝ  -- Length of the shorter side of the rectangular field
  q : ℝ  -- Length of the longer side of the rectangular forest

/-- The total area of the forest and fields given the geometry -/
def totalArea (g : SettlementGeometry) : ℝ :=
  g.r^2 + 4*g.p^2 + 12*g.q

/-- The conditions given in the problem -/
def satisfiesConditions (g : SettlementGeometry) : Prop :=
  12*g.q = g.r^2 + 4*g.p^2 + 45 ∧
  g.r > 0 ∧ g.p > 0 ∧ g.q > 0

theorem settlement_area_theorem (g : SettlementGeometry) 
  (h : satisfiesConditions g) : totalArea g = 135 := by
  sorry

#check settlement_area_theorem

end NUMINAMATH_CALUDE_settlement_area_theorem_l1202_120286


namespace NUMINAMATH_CALUDE_basil_pots_l1202_120293

theorem basil_pots (rosemary_pots thyme_pots : ℕ)
  (basil_leaves rosemary_leaves thyme_leaves total_leaves : ℕ) :
  rosemary_pots = 9 →
  thyme_pots = 6 →
  basil_leaves = 4 →
  rosemary_leaves = 18 →
  thyme_leaves = 30 →
  total_leaves = 354 →
  ∃ basil_pots : ℕ,
    basil_pots * basil_leaves +
    rosemary_pots * rosemary_leaves +
    thyme_pots * thyme_leaves = total_leaves ∧
    basil_pots = 3 :=
by sorry

end NUMINAMATH_CALUDE_basil_pots_l1202_120293


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1202_120258

theorem complex_fraction_calculation : 
  (((11 + 1/9 - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / 3.6) / (2 + 6/25)) = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1202_120258


namespace NUMINAMATH_CALUDE_inequality_solution_and_abs_inequality_l1202_120220

def f (x : ℝ) := |x - 1|

theorem inequality_solution_and_abs_inequality (a b : ℝ) :
  (∀ x, f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3) ∧
  (|a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_and_abs_inequality_l1202_120220


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l1202_120290

theorem probability_four_twos_in_five_rolls (p : ℝ) :
  p = 1 / 8 →
  (5 : ℝ) * p^4 * (1 - p) = 35 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l1202_120290


namespace NUMINAMATH_CALUDE_only_setB_proportional_l1202_120226

/-- A set of four line segments --/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a set of line segments is proportional --/
def isProportional (s : LineSegmentSet) : Prop :=
  s.a * s.d = s.b * s.c

/-- The given sets of line segments --/
def setA : LineSegmentSet := ⟨3, 4, 5, 6⟩
def setB : LineSegmentSet := ⟨5, 15, 2, 6⟩
def setC : LineSegmentSet := ⟨4, 8, 3, 5⟩
def setD : LineSegmentSet := ⟨8, 4, 1, 3⟩

/-- Theorem stating that only set B is proportional --/
theorem only_setB_proportional :
  ¬ isProportional setA ∧
  isProportional setB ∧
  ¬ isProportional setC ∧
  ¬ isProportional setD :=
sorry

end NUMINAMATH_CALUDE_only_setB_proportional_l1202_120226


namespace NUMINAMATH_CALUDE_product_of_geometric_terms_l1202_120291

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

/-- The main theorem -/
theorem product_of_geometric_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n ≠ 0) →
  a 6 - a 7 ^ 2 + a 8 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 2 * b 8 * b 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_geometric_terms_l1202_120291


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l1202_120246

open Real

theorem triangle_perimeter_range (A B C a b c : ℝ) : 
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧
  -- Sum of angles in a triangle
  A + B + C = π ∧
  -- Given equation
  cos B^2 + cos B * cos (A - C) = sin A * sin C ∧
  -- Side length a
  a = 2 * Real.sqrt 3 ∧
  -- Derived value of B
  B = π/3 ∧
  -- Definition of sides using sine rule
  b = a * sin B / sin A ∧
  c = a * sin C / sin A
  →
  -- Perimeter range
  3 + 3 * Real.sqrt 3 < a + b + c ∧ a + b + c < 6 + 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l1202_120246


namespace NUMINAMATH_CALUDE_carol_rectangle_length_l1202_120272

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem carol_rectangle_length 
  (jordan : Rectangle) 
  (carol : Rectangle) 
  (h1 : jordan.length = 3) 
  (h2 : jordan.width = 40) 
  (h3 : carol.width = 24) 
  (h4 : area jordan = area carol) : 
  carol.length = 5 := by
sorry

end NUMINAMATH_CALUDE_carol_rectangle_length_l1202_120272


namespace NUMINAMATH_CALUDE_count_ordered_pairs_eq_six_l1202_120255

/-- The number of ordered pairs of positive integers (M, N) satisfying M/8 = 4/N -/
def count_ordered_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 32) (Finset.product (Finset.range 33) (Finset.range 33))).card

theorem count_ordered_pairs_eq_six : count_ordered_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_eq_six_l1202_120255


namespace NUMINAMATH_CALUDE_min_value_theorem_l1202_120227

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (1 / a) + (1 / b) = 1) :
  6 ≤ (4 / (a - 1)) + (9 / (b - 1)) ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ (1 / a₀) + (1 / b₀) = 1 ∧ (4 / (a₀ - 1)) + (9 / (b₀ - 1)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1202_120227


namespace NUMINAMATH_CALUDE_perfect_squares_theorem_l1202_120299

theorem perfect_squares_theorem :
  -- Part 1: Infinitely many n such that 2n+1 and 3n+1 are perfect squares
  (∃ f : ℕ → ℤ, ∀ k, ∃ a b : ℤ, 2 * f k + 1 = a^2 ∧ 3 * f k + 1 = b^2) ∧
  -- Part 2: Such n are multiples of 40
  (∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → ∃ k : ℤ, n = 40 * k) ∧
  -- Part 3: Generalization for any positive integer m
  (∀ m : ℕ, m > 0 →
    ∃ g : ℕ → ℤ, ∀ k, ∃ a b : ℤ, m * g k + 1 = a^2 ∧ (m + 1) * g k + 1 = b^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_theorem_l1202_120299


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1202_120283

/-- Proves that the speed of the first train is approximately 120.016 kmph given the conditions -/
theorem train_speed_calculation (length1 length2 speed2 time : ℝ) 
  (h1 : length1 = 290) 
  (h2 : length2 = 210.04)
  (h3 : speed2 = 80)
  (h4 : time = 9)
  : ∃ speed1 : ℝ, abs (speed1 - 120.016) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1202_120283
