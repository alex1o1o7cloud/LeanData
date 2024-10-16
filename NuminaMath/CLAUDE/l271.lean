import Mathlib

namespace NUMINAMATH_CALUDE_cos_210_degrees_l271_27199

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l271_27199


namespace NUMINAMATH_CALUDE_special_triangle_area_property_l271_27182

/-- A triangle with side length PQ = 30 and its incircle trisecting the median PS in ratio 1:2 -/
structure SpecialTriangle where
  -- Points of the triangle
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- Incircle center
  I : ℝ × ℝ
  -- Point where median PS intersects QR
  S : ℝ × ℝ
  -- Points where incircle touches the sides
  T : ℝ × ℝ  -- on QR
  U : ℝ × ℝ  -- on RP
  V : ℝ × ℝ  -- on PQ
  -- Properties
  pq_length : dist P Q = 30
  trisect_median : dist P T = (1/3) * dist P S ∧ dist T S = (2/3) * dist P S
  incircle_tangent : dist I T = dist I U ∧ dist I U = dist I V

/-- The area of the special triangle can be expressed as x√y where x and y are integers -/
def area_expression (t : SpecialTriangle) : ℕ × ℕ :=
  sorry

/-- Predicate to check if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

theorem special_triangle_area_property (t : SpecialTriangle) :
  let (x, y) := area_expression t
  (x > 0 ∧ y > 0) ∧ not_divisible_by_prime_square y ∧ ∃ (k : ℕ), x + y = k :=
sorry

end NUMINAMATH_CALUDE_special_triangle_area_property_l271_27182


namespace NUMINAMATH_CALUDE_function_equality_l271_27173

theorem function_equality (f g : ℕ → ℕ) 
  (h_surj : Function.Surjective f)
  (h_inj : Function.Injective g)
  (h_ge : ∀ n : ℕ, f n ≥ g n) :
  ∀ n : ℕ, f n = g n := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l271_27173


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l271_27190

-- Define the universe U
def U : Finset ℕ := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset ℕ := {1, 4}

-- Define set N
def N : Finset ℕ := {1, 3, 5}

-- Theorem statement
theorem intersection_N_complement_M : N ∩ (U \ M) = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l271_27190


namespace NUMINAMATH_CALUDE_polynomial_factorization_l271_27135

theorem polynomial_factorization (x : ℤ) :
  x^12 + x^9 + 1 = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l271_27135


namespace NUMINAMATH_CALUDE_tartar_arrangements_l271_27192

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : ℕ) (duplicateSets : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (duplicateSets.map Nat.factorial).prod

/-- The word TARTAR has 6 letters with T, A, and R each appearing twice -/
theorem tartar_arrangements :
  uniqueArrangements 6 [2, 2, 2] = 90 := by
  sorry

end NUMINAMATH_CALUDE_tartar_arrangements_l271_27192


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l271_27118

theorem fraction_product_simplification :
  (21 : ℚ) / 28 * 14 / 33 * 99 / 42 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l271_27118


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l271_27108

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x + 1))| > 3 ↔ x < -1 ∨ (-5/6 < x ∧ x < -1) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l271_27108


namespace NUMINAMATH_CALUDE_frog_jump_probability_l271_27160

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the rectangular garden -/
structure Garden where
  bottomLeft : Point
  topRight : Point

/-- Represents the possible jump directions -/
inductive JumpDirection
  | Up
  | Down
  | Left
  | Right
  | NorthEast
  | NorthWest
  | SouthEast
  | SouthWest

/-- Represents the possible jump lengths -/
inductive JumpLength
  | One
  | Two

/-- Function to calculate the probability of ending on a horizontal side -/
def probabilityHorizontalEnd (garden : Garden) (start : Point) : ℝ :=
  sorry

/-- Theorem stating the probability of ending on a horizontal side is 0.4 -/
theorem frog_jump_probability (garden : Garden) (start : Point) :
  garden.bottomLeft = ⟨1, 1⟩ ∧
  garden.topRight = ⟨5, 6⟩ ∧
  start = ⟨2, 3⟩ →
  probabilityHorizontalEnd garden start = 0.4 :=
sorry


end NUMINAMATH_CALUDE_frog_jump_probability_l271_27160


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_ratio_l271_27126

theorem dividend_divisor_quotient_ratio 
  (dividend : ℚ) (divisor : ℚ) (quotient : ℚ) 
  (h : dividend / divisor = 9 / 2) : 
  divisor / quotient = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_ratio_l271_27126


namespace NUMINAMATH_CALUDE_square_difference_equals_two_l271_27163

theorem square_difference_equals_two (x y : ℝ) 
  (h1 : 1/x + 1/y = 2) 
  (h2 : x*y + x - y = 6) : 
  x^2 - y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_two_l271_27163


namespace NUMINAMATH_CALUDE_hat_price_after_discounts_l271_27157

def initial_price : ℝ := 15
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.50

theorem hat_price_after_discounts :
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 5.625 := by sorry

end NUMINAMATH_CALUDE_hat_price_after_discounts_l271_27157


namespace NUMINAMATH_CALUDE_sports_event_distribution_l271_27102

/-- Represents the number of medals remaining after k days -/
def remaining_medals (k : ℕ) (m : ℕ) : ℚ :=
  if k = 0 then m
  else (6/7) * remaining_medals (k-1) m - (6/7) * k

/-- The sports event distribution problem -/
theorem sports_event_distribution (n m : ℕ) : 
  (∀ k, 1 ≤ k ∧ k < n → 
    remaining_medals k m = remaining_medals (k-1) m - (k + (1/7) * (remaining_medals (k-1) m - k))) ∧
  remaining_medals (n-1) m = n ∧
  remaining_medals n m = 0 →
  n = 6 ∧ m = 36 := by
sorry

end NUMINAMATH_CALUDE_sports_event_distribution_l271_27102


namespace NUMINAMATH_CALUDE_mini_train_length_l271_27194

/-- The length of a mini-train given its speed and time to cross a pole -/
theorem mini_train_length (speed_kmph : ℝ) (time_seconds : ℝ) : 
  speed_kmph = 75 → time_seconds = 3 → 
  ∃ length_meters : ℝ, abs (length_meters - 62.5) < 0.1 ∧ 
  length_meters = speed_kmph * (1000 / 3600) * time_seconds :=
sorry

end NUMINAMATH_CALUDE_mini_train_length_l271_27194


namespace NUMINAMATH_CALUDE_complex_multiplication_l271_27150

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 + 3 * i) = -3 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l271_27150


namespace NUMINAMATH_CALUDE_ball_box_probabilities_l271_27186

/-- The number of different balls -/
def num_balls : ℕ := 4

/-- The number of different boxes -/
def num_boxes : ℕ := 4

/-- The total number of possible outcomes when placing balls into boxes -/
def total_outcomes : ℕ := num_boxes ^ num_balls

/-- The probability of no empty boxes when placing balls into boxes -/
def prob_no_empty_boxes : ℚ := 3 / 32

/-- The probability of exactly one empty box when placing balls into boxes -/
def prob_one_empty_box : ℚ := 9 / 16

/-- Theorem stating the probabilities for different scenarios when placing balls into boxes -/
theorem ball_box_probabilities :
  (prob_no_empty_boxes = 3 / 32) ∧ (prob_one_empty_box = 9 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ball_box_probabilities_l271_27186


namespace NUMINAMATH_CALUDE_imaginary_sum_l271_27155

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum : i^55 + i^555 + i^5 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_l271_27155


namespace NUMINAMATH_CALUDE_profit_share_ratio_l271_27162

theorem profit_share_ratio (total_profit : ℚ) (difference : ℚ) 
  (h1 : total_profit = 1000)
  (h2 : difference = 200) :
  ∃ (x y : ℚ), x + y = total_profit ∧ x - y = difference ∧ y / total_profit = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l271_27162


namespace NUMINAMATH_CALUDE_hilton_marbles_l271_27198

/-- Calculates the final number of marbles Hilton has -/
def final_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

theorem hilton_marbles : final_marbles 26 6 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_hilton_marbles_l271_27198


namespace NUMINAMATH_CALUDE_solve_gumball_problem_l271_27177

def gumball_problem (alicia_gumballs : ℕ) (remaining_gumballs : ℕ) : Prop :=
  let pedro_gumballs := alicia_gumballs + 3 * alicia_gumballs
  let total_gumballs := alicia_gumballs + pedro_gumballs
  let taken_gumballs := total_gumballs - remaining_gumballs
  (taken_gumballs : ℚ) / (total_gumballs : ℚ) = 2/5

theorem solve_gumball_problem :
  gumball_problem 20 60 := by sorry

end NUMINAMATH_CALUDE_solve_gumball_problem_l271_27177


namespace NUMINAMATH_CALUDE_part_one_part_two_l271_27116

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, q x → (∀ a : ℝ, a > 0 → p a x)) ∧ 
  (∃ x a : ℝ, a > 0 ∧ p a x ∧ ¬q x) → 
  ∀ a : ℝ, 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l271_27116


namespace NUMINAMATH_CALUDE_real_estate_transaction_result_l271_27153

/-- Represents the result of a transaction -/
inductive TransactionResult
  | Loss (amount : ℚ)
  | Gain (amount : ℚ)
  | NoChange

/-- Calculates the result of a real estate transaction -/
def calculateTransactionResult (houseSalePrice storeSalePrice : ℚ) 
                               (houseLossPercentage storeGainPercentage : ℚ) : TransactionResult :=
  let houseCost := houseSalePrice / (1 - houseLossPercentage)
  let storeCost := storeSalePrice / (1 + storeGainPercentage)
  let totalCost := houseCost + storeCost
  let totalSale := houseSalePrice + storeSalePrice
  let difference := totalCost - totalSale
  if difference > 0 then TransactionResult.Loss difference
  else if difference < 0 then TransactionResult.Gain (-difference)
  else TransactionResult.NoChange

/-- Theorem stating the result of the specific real estate transaction -/
theorem real_estate_transaction_result :
  calculateTransactionResult 15000 15000 (30/100) (25/100) = TransactionResult.Loss (3428.57/100) := by
  sorry

end NUMINAMATH_CALUDE_real_estate_transaction_result_l271_27153


namespace NUMINAMATH_CALUDE_simplify_fraction_l271_27168

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l271_27168


namespace NUMINAMATH_CALUDE_grandson_age_l271_27141

theorem grandson_age (grandson_age grandfather_age : ℕ) : 
  grandfather_age = 6 * grandson_age →
  (grandson_age + 4) + (grandfather_age + 4) = 78 →
  grandson_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_grandson_age_l271_27141


namespace NUMINAMATH_CALUDE_product_and_ratio_implies_y_value_l271_27145

theorem product_and_ratio_implies_y_value 
  (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = 9) 
  (h4 : x / y = 36) : 
  y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_product_and_ratio_implies_y_value_l271_27145


namespace NUMINAMATH_CALUDE_similar_triangles_area_l271_27148

/-- Given two similar triangles with corresponding sides of 1 cm and 2 cm, 
    and a total area of 25 cm², the area of the larger triangle is 20 cm². -/
theorem similar_triangles_area (A B : ℝ) : 
  A > 0 → B > 0 →  -- Areas are positive
  A + B = 25 →     -- Sum of areas is 25 cm²
  B / A = 4 →      -- Ratio of areas is 4 (square of the ratio of sides)
  B = 20 := by 
sorry

end NUMINAMATH_CALUDE_similar_triangles_area_l271_27148


namespace NUMINAMATH_CALUDE_sin_period_l271_27152

/-- The period of y = sin(4x + π) is π/2 -/
theorem sin_period (x : ℝ) : 
  (∀ y, y = Real.sin (4 * x + π)) → 
  (∃ p, p > 0 ∧ ∀ x, Real.sin (4 * x + π) = Real.sin (4 * (x + p) + π) ∧ p = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_sin_period_l271_27152


namespace NUMINAMATH_CALUDE_euclidean_division_remainder_l271_27138

theorem euclidean_division_remainder 
  (P : Polynomial ℝ) 
  (D : Polynomial ℝ) 
  (h1 : P = X^100 - 2*X^51 + 1)
  (h2 : D = X^2 - 1) :
  ∃ (Q R : Polynomial ℝ), 
    P = D * Q + R ∧ 
    R.degree < D.degree ∧ 
    R = -2*X + 2 := by
sorry

end NUMINAMATH_CALUDE_euclidean_division_remainder_l271_27138


namespace NUMINAMATH_CALUDE_no_x_axis_intersection_implies_m_bound_l271_27100

/-- A quadratic function of the form f(x) = x^2 - x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - x + m

/-- The discriminant of the quadratic function f(x) = x^2 - x + m -/
def discriminant (m : ℝ) : ℝ := 1 - 4*m

theorem no_x_axis_intersection_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, f m x ≠ 0) → m > (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_no_x_axis_intersection_implies_m_bound_l271_27100


namespace NUMINAMATH_CALUDE_taxi_charge_per_segment_l271_27105

/-- Calculates the additional charge per 2/5 of a mile for a taxi service -/
theorem taxi_charge_per_segment (initial_fee : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_fee = 2.25 →
  total_distance = 3.6 →
  total_charge = 3.60 →
  (total_charge - initial_fee) / (total_distance / (2/5)) = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_per_segment_l271_27105


namespace NUMINAMATH_CALUDE_circle_tangent_k_range_l271_27131

/-- Represents a circle in the 2D plane --/
structure Circle where
  k : ℝ

/-- Represents a point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is outside the circle --/
def isOutside (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + 2*p.x + 2*p.y + c.k > 0

/-- Checks if two tangents can be drawn from a point to the circle --/
def hasTwoTangents (p : Point) (c : Circle) : Prop :=
  isOutside p c

/-- The main theorem --/
theorem circle_tangent_k_range (c : Circle) :
  let p : Point := ⟨1, -1⟩
  hasTwoTangents p c → -2 < c.k ∧ c.k < 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_k_range_l271_27131


namespace NUMINAMATH_CALUDE_tickets_left_to_sell_l271_27114

theorem tickets_left_to_sell (total : ℕ) (first_week : ℕ) (second_week : ℕ) 
  (h1 : total = 90) 
  (h2 : first_week = 38) 
  (h3 : second_week = 17) :
  total - (first_week + second_week) = 35 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_to_sell_l271_27114


namespace NUMINAMATH_CALUDE_angle_trisection_for_non_multiple_of_three_l271_27164

/-- An angle is constructible if it can be constructed using only a ruler and compass. -/
def AngleConstructible (θ : ℝ) : Prop := sorry

/-- An angle is trisectable if it can be divided into three equal parts using only a ruler and compass. -/
def AngleTrisectable (θ : ℝ) : Prop := sorry

/-- The theorem stating that if n is not a multiple of 3, then the angle π/n can be trisected with ruler and compasses. -/
theorem angle_trisection_for_non_multiple_of_three (n : ℕ) (h : ¬ 3 ∣ n) : 
  AngleTrisectable (π / n) := by sorry

end NUMINAMATH_CALUDE_angle_trisection_for_non_multiple_of_three_l271_27164


namespace NUMINAMATH_CALUDE_determinant_positive_range_l271_27136

def second_order_determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_positive_range (x : ℝ) :
  second_order_determinant 2 (3 - x) 1 x > 0 ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_determinant_positive_range_l271_27136


namespace NUMINAMATH_CALUDE_negative_expression_l271_27154

theorem negative_expression : 
  let expr1 := -(-1)
  let expr2 := (-1)^2
  let expr3 := |-1|
  let expr4 := -|-1|
  (expr1 ≥ 0 ∧ expr2 ≥ 0 ∧ expr3 ≥ 0 ∧ expr4 < 0) := by sorry

end NUMINAMATH_CALUDE_negative_expression_l271_27154


namespace NUMINAMATH_CALUDE_mary_gave_three_green_crayons_l271_27101

/-- The number of green crayons Mary gave to Becky -/
def green_crayons_given : ℕ := 3

/-- The initial number of green crayons Mary had -/
def initial_green : ℕ := 5

/-- The initial number of blue crayons Mary had -/
def initial_blue : ℕ := 8

/-- The number of blue crayons Mary gave away -/
def blue_given : ℕ := 1

/-- The number of crayons Mary has left -/
def crayons_left : ℕ := 9

theorem mary_gave_three_green_crayons :
  green_crayons_given = initial_green - (initial_green + initial_blue - blue_given - crayons_left) :=
by sorry

end NUMINAMATH_CALUDE_mary_gave_three_green_crayons_l271_27101


namespace NUMINAMATH_CALUDE_restaurant_cooks_l271_27144

theorem restaurant_cooks (initial_cooks : ℕ) (initial_waiters : ℕ) : 
  initial_cooks / initial_waiters = 3 / 11 →
  initial_cooks / (initial_waiters + 12) = 1 / 5 →
  initial_cooks = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_l271_27144


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_touching_circle_l271_27121

/-- A triangle with sides a, b, c and medians m_a, m_b, m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ

/-- A circle touching two sides and two medians of a triangle -/
structure TouchingCircle (T : Triangle) where
  touches_side_a : Bool
  touches_side_b : Bool
  touches_median_a : Bool
  touches_median_b : Bool

/-- 
If a circle touches two sides of a triangle and their corresponding medians,
then the triangle is isosceles.
-/
theorem isosceles_triangle_from_touching_circle (T : Triangle) 
  (C : TouchingCircle T) (h1 : C.touches_side_a) (h2 : C.touches_side_b) 
  (h3 : C.touches_median_a) (h4 : C.touches_median_b) : 
  T.a = T.b := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_from_touching_circle_l271_27121


namespace NUMINAMATH_CALUDE_sin_15_minus_sin_75_fourth_power_l271_27142

theorem sin_15_minus_sin_75_fourth_power :
  Real.sin (15 * π / 180) ^ 4 - Real.sin (75 * π / 180) ^ 4 = -(Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_minus_sin_75_fourth_power_l271_27142


namespace NUMINAMATH_CALUDE_final_digit_mod_seven_l271_27137

/-- Represents the allowed operations on the number --/
inductive Operation
  | increaseDecrease : Operation
  | subtractAdd : Operation
  | decreaseBySeven : Operation

/-- The initial number as a list of digits --/
def initialNumber : List Nat := List.replicate 100 8

/-- A function that applies an operation to a list of digits --/
def applyOperation (digits : List Nat) (op : Operation) : List Nat :=
  sorry

/-- A function that removes leading zeros from a list of digits --/
def removeLeadingZeros (digits : List Nat) : List Nat :=
  sorry

/-- A function that applies operations until a single digit remains --/
def applyOperationsUntilSingleDigit (digits : List Nat) : Nat :=
  sorry

/-- Theorem stating that the final single digit is equivalent to 3 modulo 7 --/
theorem final_digit_mod_seven (ops : List Operation) :
  (applyOperationsUntilSingleDigit initialNumber) % 7 = 3 :=
sorry

end NUMINAMATH_CALUDE_final_digit_mod_seven_l271_27137


namespace NUMINAMATH_CALUDE_trailing_zeros_of_product_sum_of_digits_l271_27176

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Product of sum of digits from 1 to n -/
def product_of_sum_of_digits (n : ℕ) : ℕ := sorry

/-- Number of trailing zeros in a natural number -/
def trailing_zeros (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in the product of sum of digits from 1 to 100 is 19 -/
theorem trailing_zeros_of_product_sum_of_digits : 
  trailing_zeros (product_of_sum_of_digits 100) = 19 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_product_sum_of_digits_l271_27176


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solution_l271_27117

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  9 * x^2 - (x - 1)^2 = 0 ↔ x = -0.5 ∨ x = 0.25 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x * (x - 3) = 10 ↔ x = 5 ∨ x = -2 := by sorry

-- Equation 3
theorem equation_three_solution (x : ℝ) :
  (x + 3)^2 = 2 * x + 5 ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solution_l271_27117


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l271_27184

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 32) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l271_27184


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l271_27128

theorem least_positive_integer_with_remainders : ∃! b : ℕ+, 
  (b : ℕ) % 3 = 2 ∧ 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 5 = 4 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ x : ℕ+, 
    (x : ℕ) % 3 = 2 → 
    (x : ℕ) % 4 = 3 → 
    (x : ℕ) % 5 = 4 → 
    (x : ℕ) % 6 = 5 → 
    b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l271_27128


namespace NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l271_27174

theorem four_digit_perfect_cubes_divisible_by_16 :
  (Finset.filter (fun n : ℕ => 
    1000 ≤ 8 * n^3 ∧ 8 * n^3 ≤ 9999) (Finset.range 1000)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l271_27174


namespace NUMINAMATH_CALUDE_system_solution_range_l271_27196

theorem system_solution_range (a x y : ℝ) : 
  x - y = a + 3 →
  2 * x + y = 5 * a →
  x < y →
  a < -3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l271_27196


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l271_27147

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)).im ≠ 0 ∧ 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)).re = 0 → 
  m = 5 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l271_27147


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_greatest_integer_value_l271_27122

theorem greatest_integer_for_all_real_domain (b : ℤ) : 
  (∀ x : ℝ, (x^2 + b*x + 5 ≠ 0)) ↔ b^2 < 20 :=
by sorry

theorem greatest_integer_value : 
  ∃ b : ℤ, b = 4 ∧ (∀ x : ℝ, (x^2 + b*x + 5 ≠ 0)) ∧ 
  (∀ c : ℤ, c > b → ∃ x : ℝ, (x^2 + c*x + 5 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_greatest_integer_value_l271_27122


namespace NUMINAMATH_CALUDE_cosine_sum_l271_27158

theorem cosine_sum (α β : Real) : 
  α ∈ Set.Ioo 0 (π/3) →
  β ∈ Set.Ioo (π/6) (π/2) →
  5 * Real.sqrt 3 * Real.sin α + 5 * Real.cos α = 8 →
  Real.sqrt 2 * Real.sin β + Real.sqrt 6 * Real.cos β = 2 →
  Real.cos (α + β) = -(Real.sqrt 2) / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_l271_27158


namespace NUMINAMATH_CALUDE_at_least_one_composite_l271_27171

theorem at_least_one_composite (a b c : ℕ) 
  (h_odd_a : Odd a) (h_odd_b : Odd b) (h_odd_c : Odd c)
  (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c)
  (h_not_square : ¬∃k, a = k^2)
  (h_equation : a^2 + a + 1 = 3 * (b^2 + b + 1) * (c^2 + c + 1)) :
  (∃k > 1, k ∣ (b^2 + b + 1)) ∨ (∃k > 1, k ∣ (c^2 + c + 1)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_composite_l271_27171


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l271_27179

theorem trigonometric_equation_solutions :
  ∃ (S : Finset ℝ), 
    (∀ X ∈ S, 0 < X ∧ X < 2 * Real.pi) ∧
    (∀ X ∈ S, 1 + 2 * Real.sin X - 4 * (Real.sin X)^2 - 8 * (Real.sin X)^3 = 0) ∧
    S.card = 4 ∧
    (∀ Y, 0 < Y ∧ Y < 2 * Real.pi → 
      (1 + 2 * Real.sin Y - 4 * (Real.sin Y)^2 - 8 * (Real.sin Y)^3 = 0) → 
      Y ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l271_27179


namespace NUMINAMATH_CALUDE_min_c_value_l271_27180

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c) :
  (∃! x y : ℝ, 2 * x + y = 2031 ∧ y = |x - a| + |x - b| + |x - c|) →
  c ≥ 1016 ∧ ∃ a' b' : ℕ, a' < b' ∧ b' < 1016 ∧
    (∃! x y : ℝ, 2 * x + y = 2031 ∧ y = |x - a'| + |x - b'| + |x - 1016|) :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l271_27180


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l271_27165

theorem sqrt_five_irrational_and_greater_than_two :
  ∃ x : ℝ, Irrational x ∧ x > 2 ∧ x ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l271_27165


namespace NUMINAMATH_CALUDE_consecutive_cube_product_divisible_l271_27167

theorem consecutive_cube_product_divisible (a : ℤ) : 
  504 ∣ ((a^3 - 1) * a^3 * (a^3 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cube_product_divisible_l271_27167


namespace NUMINAMATH_CALUDE_container_filling_l271_27115

theorem container_filling (initial_percentage : Real) (added_amount : Real) (capacity : Real) :
  initial_percentage = 0.4 →
  added_amount = 14 →
  capacity = 40 →
  (initial_percentage * capacity + added_amount) / capacity = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_container_filling_l271_27115


namespace NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l271_27187

-- Define a type for our functions
def Function2D := ℝ → ℝ

-- Define the horizontal line test
def passes_horizontal_line_test (f : Function2D) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- Define what it means for a function to have an inverse
def has_inverse (f : Function2D) : Prop :=
  ∃ g : Function2D, (∀ x : ℝ, g (f x) = x) ∧ (∀ y : ℝ, f (g y) = y)

-- Theorem statement
theorem inverse_iff_horizontal_line_test (f : Function2D) :
  has_inverse f ↔ passes_horizontal_line_test f :=
sorry

end NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l271_27187


namespace NUMINAMATH_CALUDE_marys_thursday_payment_l271_27156

theorem marys_thursday_payment 
  (credit_limit : ℕ) 
  (tuesday_payment : ℕ) 
  (remaining_balance : ℕ) 
  (h1 : credit_limit = 100)
  (h2 : tuesday_payment = 15)
  (h3 : remaining_balance = 62) :
  credit_limit - tuesday_payment - remaining_balance = 23 := by
sorry

end NUMINAMATH_CALUDE_marys_thursday_payment_l271_27156


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l271_27161

theorem least_addition_for_divisibility (n : ℕ) (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  ∃! x : ℕ, x < a * b ∧ (n + x) % a = 0 ∧ (n + x) % b = 0 ∧
  ∀ y : ℕ, y < x → ((n + y) % a ≠ 0 ∨ (n + y) % b ≠ 0) :=
by sorry

theorem problem_solution : 
  let n := 1056
  let a := 27
  let b := 31
  ∃! x : ℕ, x < a * b ∧ (n + x) % a = 0 ∧ (n + x) % b = 0 ∧
  ∀ y : ℕ, y < x → ((n + y) % a ≠ 0 ∨ (n + y) % b ≠ 0) ∧
  x = 618 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l271_27161


namespace NUMINAMATH_CALUDE_cloud9_diving_company_revenue_l271_27189

/-- Cloud 9 Diving Company's financial calculation -/
theorem cloud9_diving_company_revenue 
  (individual_bookings : ℕ) 
  (group_bookings : ℕ) 
  (cancellations : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : cancellations = 1600) :
  individual_bookings + group_bookings - cancellations = 26400 :=
by sorry

end NUMINAMATH_CALUDE_cloud9_diving_company_revenue_l271_27189


namespace NUMINAMATH_CALUDE_cube_labeling_impossible_l271_27178

/-- Represents a cube with vertices labeled by natural numbers -/
structure LabeledCube :=
  (vertices : Fin 8 → ℕ)
  (is_permutation : Function.Bijective vertices)

/-- The set of edges in a cube -/
def cube_edges : Finset (Fin 8 × Fin 8) := sorry

/-- The sum of labels at the ends of an edge -/
def edge_sum (c : LabeledCube) (e : Fin 8 × Fin 8) : ℕ :=
  c.vertices e.1 + c.vertices e.2

/-- Theorem: It's impossible to label a cube's vertices with 1 to 8 such that all edge sums are different -/
theorem cube_labeling_impossible : 
  ¬ ∃ (c : LabeledCube), (∀ v : Fin 8, c.vertices v ∈ Finset.range 9 \ {0}) ∧ 
    (∀ e₁ e₂ : Fin 8 × Fin 8, e₁ ∈ cube_edges → e₂ ∈ cube_edges → e₁ ≠ e₂ → 
      edge_sum c e₁ ≠ edge_sum c e₂) :=
sorry

end NUMINAMATH_CALUDE_cube_labeling_impossible_l271_27178


namespace NUMINAMATH_CALUDE_regular_tire_usage_l271_27123

theorem regular_tire_usage
  (total_miles : ℕ)
  (spare_miles : ℕ)
  (regular_tires : ℕ)
  (h1 : total_miles = 50000)
  (h2 : spare_miles = 2000)
  (h3 : regular_tires = 4) :
  (total_miles - spare_miles) / regular_tires = 12000 :=
by sorry

end NUMINAMATH_CALUDE_regular_tire_usage_l271_27123


namespace NUMINAMATH_CALUDE_family_age_calculation_l271_27197

theorem family_age_calculation (initial_members : ℕ) (initial_avg_age : ℝ) 
  (current_members : ℕ) (current_avg_age : ℝ) (baby_age : ℝ) : ℝ :=
by
  -- Define the conditions
  have h1 : initial_members = 5 := by sorry
  have h2 : initial_avg_age = 17 := by sorry
  have h3 : current_members = 6 := by sorry
  have h4 : current_avg_age = 17 := by sorry
  have h5 : baby_age = 2 := by sorry

  -- Define the function to calculate the time elapsed
  let time_elapsed := 
    (current_members * current_avg_age - initial_members * initial_avg_age - baby_age) / 
    (initial_members : ℝ)

  -- Prove that the time elapsed is 3 years
  have : time_elapsed = 3 := by sorry

  -- Return the result
  exact time_elapsed

end NUMINAMATH_CALUDE_family_age_calculation_l271_27197


namespace NUMINAMATH_CALUDE_updated_p_value_l271_27159

/-- Given the equation fp - w = 20000, where f = 10 and w = 10 + 250i, prove that p = 2001 + 25i -/
theorem updated_p_value (f w p : ℂ) : 
  f = 10 → w = 10 + 250 * Complex.I → f * p - w = 20000 → p = 2001 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_updated_p_value_l271_27159


namespace NUMINAMATH_CALUDE_pencil_boxes_count_l271_27109

theorem pencil_boxes_count (book_boxes : ℕ) (books_per_box : ℕ) (pencils_per_box : ℕ) (total_items : ℕ) :
  book_boxes = 19 →
  books_per_box = 46 →
  pencils_per_box = 170 →
  total_items = 1894 →
  (total_items - book_boxes * books_per_box) / pencils_per_box = 6 :=
by
  sorry

#check pencil_boxes_count

end NUMINAMATH_CALUDE_pencil_boxes_count_l271_27109


namespace NUMINAMATH_CALUDE_equation_solution_l271_27193

theorem equation_solution : ∃ x : ℝ, 2*x + 5 = 3*x - 2 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l271_27193


namespace NUMINAMATH_CALUDE_cubic_equation_roots_relation_l271_27125

theorem cubic_equation_roots_relation (a b c : ℝ) (s₁ s₂ s₃ : ℂ) :
  (s₁^3 + a*s₁^2 + b*s₁ + c = 0) →
  (s₂^3 + a*s₂^2 + b*s₂ + c = 0) →
  (s₃^3 + a*s₃^2 + b*s₃ + c = 0) →
  (∃ p q r : ℝ, (s₁^2)^3 + p*(s₁^2)^2 + q*(s₁^2) + r = 0 ∧
               (s₂^2)^3 + p*(s₂^2)^2 + q*(s₂^2) + r = 0 ∧
               (s₃^2)^3 + p*(s₃^2)^2 + q*(s₃^2) + r = 0) →
  (∃ p q r : ℝ, p = a^2 - 2*b ∧ q = b^2 + 2*a*c ∧ r = c^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_relation_l271_27125


namespace NUMINAMATH_CALUDE_inequality_not_hold_l271_27104

theorem inequality_not_hold (m n a : Real) 
  (h1 : m > n) (h2 : n > 1) (h3 : 0 < a) (h4 : a < 1) : 
  ¬(a^m > a^n) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l271_27104


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l271_27111

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numDays : Nat
  numFridays : Nat

/-- Given a starting day and a number of days, calculates the resulting day of the week -/
def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that under given conditions, the 12th day of the month is a Monday -/
theorem twelfth_day_is_monday (m : Month) 
  (h1 : m.numFridays = 5)
  (h2 : m.firstDay ≠ DayOfWeek.Friday)
  (h3 : m.lastDay ≠ DayOfWeek.Friday)
  (h4 : m.numDays ≥ 12) :
  advanceDays m.firstDay 11 = DayOfWeek.Monday :=
  sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l271_27111


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l271_27130

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l271_27130


namespace NUMINAMATH_CALUDE_exists_n_congruence_l271_27139

theorem exists_n_congruence (l : ℕ+) : ∃ n : ℕ, (n^n + 47) % (2^l.val) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_congruence_l271_27139


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l271_27151

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 8 → 
  (10 * x + y) - (10 * y + x) = 72 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l271_27151


namespace NUMINAMATH_CALUDE_beads_per_necklace_l271_27143

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 18) (h2 : num_necklaces = 6) :
  total_beads / num_necklaces = 3 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l271_27143


namespace NUMINAMATH_CALUDE_students_in_school_l271_27169

theorem students_in_school (total_students : ℕ) (trip_fraction : ℚ) (home_fraction : ℚ) : 
  total_students = 1000 →
  trip_fraction = 1/2 →
  home_fraction = 1/2 →
  (total_students - (trip_fraction * total_students).floor - 
   (home_fraction * (total_students - (trip_fraction * total_students).floor)).floor) = 250 := by
  sorry

#check students_in_school

end NUMINAMATH_CALUDE_students_in_school_l271_27169


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l271_27112

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = -5 / 17 - 14 / 17 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l271_27112


namespace NUMINAMATH_CALUDE_smallest_k_for_f_iteration_zero_l271_27185

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

def iterateF (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (iterateF a b M k n)

theorem smallest_k_for_f_iteration_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k : ℕ, k = (a + b) / Nat.gcd a b ∧ 
    iterateF a b M k 0 = 0 ∧ 
    ∀ j : ℕ, j < k → iterateF a b M j 0 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_f_iteration_zero_l271_27185


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l271_27191

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℚ
  common_ratio : ℚ

/-- Get the nth term of a geometric sequence -/
def GeometricSequence.nth_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 48 and the 6th term is 72, the 2nd term is 1152/81 -/
theorem geometric_sequence_second_term
  (seq : GeometricSequence)
  (h5 : seq.nth_term 5 = 48)
  (h6 : seq.nth_term 6 = 72) :
  seq.nth_term 2 = 1152 / 81 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_second_term_l271_27191


namespace NUMINAMATH_CALUDE_special_haircut_price_l271_27140

/-- Represents the cost of different types of haircuts and the hairstylist's earnings --/
structure HaircutPrices where
  normal : ℝ
  special : ℝ
  trendy : ℝ
  daily_normal : ℕ
  daily_special : ℕ
  daily_trendy : ℕ
  weekly_earnings : ℝ
  days_per_week : ℕ

/-- Theorem stating that the special haircut price is $6 given the conditions --/
theorem special_haircut_price (h : HaircutPrices) 
    (h_normal : h.normal = 5)
    (h_trendy : h.trendy = 8)
    (h_daily_normal : h.daily_normal = 5)
    (h_daily_special : h.daily_special = 3)
    (h_daily_trendy : h.daily_trendy = 2)
    (h_weekly_earnings : h.weekly_earnings = 413)
    (h_days_per_week : h.days_per_week = 7) :
  h.special = 6 := by
  sorry

#check special_haircut_price

end NUMINAMATH_CALUDE_special_haircut_price_l271_27140


namespace NUMINAMATH_CALUDE_cube_sum_equality_l271_27133

theorem cube_sum_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (square_fourth_equality : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) :
  a^3 + b^3 + c^3 = -3*a*b*(a+b) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l271_27133


namespace NUMINAMATH_CALUDE_expression_evaluation_l271_27127

theorem expression_evaluation :
  let x : ℚ := -1/2
  (3 * x^4 - 2 * x^3) / (-x) - (x - x^2) * 3 * x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l271_27127


namespace NUMINAMATH_CALUDE_product_xyz_is_negative_two_l271_27113

theorem product_xyz_is_negative_two 
  (x y z : ℝ) 
  (h1 : x + 2 / y = 2) 
  (h2 : y + 2 / z = 2) : 
  x * y * z = -2 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_negative_two_l271_27113


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l271_27183

theorem real_roots_of_polynomial (x : ℝ) : 
  (x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0) ↔ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l271_27183


namespace NUMINAMATH_CALUDE_f_composition_half_f_composition_eq_one_solutions_l271_27103

noncomputable section

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by sorry

theorem f_composition_eq_one_solutions : 
  {x : ℝ | f (f x) = 1} = {1, Real.exp (Real.exp 1)} := by sorry

end NUMINAMATH_CALUDE_f_composition_half_f_composition_eq_one_solutions_l271_27103


namespace NUMINAMATH_CALUDE_age_ratio_solution_l271_27134

/-- Represents the age ratio problem of Mandy and her siblings -/
def age_ratio_problem (mandy_age brother_age sister_age : ℚ) : Prop :=
  mandy_age = 3 ∧
  sister_age = brother_age - 5 ∧
  mandy_age - sister_age = 4 ∧
  brother_age / mandy_age = 4 / 3

/-- Theorem stating that there exists a unique solution to the age ratio problem -/
theorem age_ratio_solution :
  ∃! (mandy_age brother_age sister_age : ℚ),
    age_ratio_problem mandy_age brother_age sister_age :=
by
  sorry

#check age_ratio_solution

end NUMINAMATH_CALUDE_age_ratio_solution_l271_27134


namespace NUMINAMATH_CALUDE_austin_picked_24_bags_l271_27146

/-- The number of bags of fruit Austin picked in total -/
def austin_total (dallas_apples dallas_pears austin_apples_diff austin_pears_diff : ℕ) : ℕ :=
  (dallas_apples + austin_apples_diff) + (dallas_pears - austin_pears_diff)

/-- Theorem stating that Austin picked 24 bags of fruit in total -/
theorem austin_picked_24_bags
  (dallas_apples : ℕ)
  (dallas_pears : ℕ)
  (austin_apples_diff : ℕ)
  (austin_pears_diff : ℕ)
  (h1 : dallas_apples = 14)
  (h2 : dallas_pears = 9)
  (h3 : austin_apples_diff = 6)
  (h4 : austin_pears_diff = 5) :
  austin_total dallas_apples dallas_pears austin_apples_diff austin_pears_diff = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_austin_picked_24_bags_l271_27146


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l271_27181

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.repeatingPart / (99 : ℚ)

/-- The repeating decimal 0.72̅ -/
def zero_point_72_repeating : RepeatingDecimal :=
  ⟨0, 72⟩

/-- The repeating decimal 2.09̅ -/
def two_point_09_repeating : RepeatingDecimal :=
  ⟨2, 9⟩

/-- Theorem stating that the division of the two given repeating decimals equals 8/23 -/
theorem repeating_decimal_division :
    (toRational zero_point_72_repeating) / (toRational two_point_09_repeating) = 8 / 23 := by
  sorry


end NUMINAMATH_CALUDE_repeating_decimal_division_l271_27181


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l271_27107

/-- Given vectors a and b in ℝ², if a is perpendicular to (a + 2b), then the second component of b is -3/4 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (-1, 2)) (h' : b.1 = 1) 
    (h'' : a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) : 
    b.2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l271_27107


namespace NUMINAMATH_CALUDE_arun_remaining_work_days_arun_remaining_work_days_proof_l271_27166

-- Define the work rates and time
def arun_tarun_rate : ℚ := 1 / 10
def arun_rate : ℚ := 1 / 60
def initial_work_days : ℕ := 4
def total_work : ℚ := 1

-- Theorem statement
theorem arun_remaining_work_days : ℕ :=
  let remaining_work : ℚ := total_work - (arun_tarun_rate * initial_work_days)
  let arun_remaining_days : ℚ := remaining_work / arun_rate
  36

-- Proof
theorem arun_remaining_work_days_proof :
  arun_remaining_work_days = 36 := by
  sorry

end NUMINAMATH_CALUDE_arun_remaining_work_days_arun_remaining_work_days_proof_l271_27166


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l271_27129

/-- 
For a quadratic equation x^2 - 2x + k = 0 to have two real roots, 
k must satisfy k ≤ 1
-/
theorem quadratic_two_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) →
  k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l271_27129


namespace NUMINAMATH_CALUDE_solve_star_equation_l271_27110

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

-- Theorem statement
theorem solve_star_equation :
  ∀ y : ℝ, star 7 y = 47 → y = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l271_27110


namespace NUMINAMATH_CALUDE_emily_income_l271_27120

/-- Represents the tax structure and Emily's tax payment --/
structure TaxSystem where
  q : ℝ  -- Base tax rate
  income : ℝ  -- Emily's annual income
  total_tax : ℝ  -- Total tax paid by Emily

/-- The tax system satisfies the given conditions --/
def valid_tax_system (ts : TaxSystem) : Prop :=
  ts.total_tax = 
    (0.01 * ts.q * 35000 + 
     0.01 * (ts.q + 3) * 15000 + 
     0.01 * (ts.q + 5) * (ts.income - 50000)) *
    (if ts.income > 50000 then 1 else 0) +
    (0.01 * ts.q * 35000 + 
     0.01 * (ts.q + 3) * (ts.income - 35000)) *
    (if ts.income > 35000 ∧ ts.income ≤ 50000 then 1 else 0) +
    (0.01 * ts.q * ts.income) *
    (if ts.income ≤ 35000 then 1 else 0)

/-- Emily's total tax is (q + 0.75)% of her income --/
def emily_tax_condition (ts : TaxSystem) : Prop :=
  ts.total_tax = 0.01 * (ts.q + 0.75) * ts.income

/-- Theorem: Emily's income is $48235 --/
theorem emily_income (ts : TaxSystem) 
  (h1 : valid_tax_system ts) 
  (h2 : emily_tax_condition ts) : 
  ts.income = 48235 :=
sorry

end NUMINAMATH_CALUDE_emily_income_l271_27120


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l271_27149

/-- Given four points on a plane, the distance between any two points
    is less than or equal to the sum of the distances along a path
    through the other two points. -/
theorem quadrilateral_inequality (A B C D : EuclideanSpace ℝ (Fin 2)) :
  dist A D ≤ dist A B + dist B C + dist C D := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l271_27149


namespace NUMINAMATH_CALUDE_triangle_property_and_function_value_l271_27175

theorem triangle_property_and_function_value (a b c A : ℝ) :
  0 < A ∧ A < π →
  b^2 + c^2 = a^2 + Real.sqrt 3 * b * c →
  let m : ℝ × ℝ := (Real.sin A, Real.cos A)
  let n : ℝ × ℝ := (Real.cos A, Real.sqrt 3 * Real.cos A)
  let f : ℝ → ℝ := fun x => m.1 * n.1 + m.2 * n.2 - Real.sqrt 3 / 2
  f A = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_and_function_value_l271_27175


namespace NUMINAMATH_CALUDE_not_divisible_by_81_l271_27170

theorem not_divisible_by_81 (n : ℤ) : ¬(81 ∣ (n^3 - 9*n + 27)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_81_l271_27170


namespace NUMINAMATH_CALUDE_cos_150_degrees_l271_27195

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l271_27195


namespace NUMINAMATH_CALUDE_fraction_unchanged_when_multiplied_by_two_l271_27124

theorem fraction_unchanged_when_multiplied_by_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / (x + y) = (2 * x) / (2 * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_when_multiplied_by_two_l271_27124


namespace NUMINAMATH_CALUDE_cunningham_white_lambs_l271_27132

/-- The number of white lambs owned by farmer Cunningham -/
def white_lambs (total : ℕ) (black : ℕ) : ℕ := total - black

theorem cunningham_white_lambs :
  white_lambs 6048 5855 = 193 :=
by sorry

end NUMINAMATH_CALUDE_cunningham_white_lambs_l271_27132


namespace NUMINAMATH_CALUDE_inequality_solution_set_l271_27106

theorem inequality_solution_set :
  {x : ℝ | (1 : ℝ) / x < (1 : ℝ) / 2} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l271_27106


namespace NUMINAMATH_CALUDE_expression_value_l271_27119

theorem expression_value (a b m n x : ℝ) : 
  (a = -b) →                   -- a and b are opposite numbers
  (m * n = 1) →                -- m and n are reciprocal numbers
  (m - n ≠ 0) →                -- given condition
  (abs x = 2) →                -- absolute value of x is 2
  (-2 * m * n + (b + a) / (m - n) - x = -4 ∨ 
   -2 * m * n + (b + a) / (m - n) - x = 0) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l271_27119


namespace NUMINAMATH_CALUDE_selling_price_achieves_target_profit_selling_price_minimizes_inventory_l271_27188

/-- Represents the selling price of a helmet -/
def selling_price : ℝ := 50

/-- Represents the cost price of a helmet -/
def cost_price : ℝ := 30

/-- Represents the initial selling price -/
def initial_price : ℝ := 40

/-- Represents the initial monthly sales volume -/
def initial_sales : ℝ := 600

/-- Represents the rate of decrease in sales volume per dollar increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Represents the target monthly profit -/
def target_profit : ℝ := 10000

/-- Calculates the monthly sales volume based on the selling price -/
def monthly_sales (price : ℝ) : ℝ := initial_sales - sales_decrease_rate * (price - initial_price)

/-- Calculates the monthly profit based on the selling price -/
def monthly_profit (price : ℝ) : ℝ := (price - cost_price) * monthly_sales price

/-- Theorem stating that the selling price achieves the target monthly profit -/
theorem selling_price_achieves_target_profit : 
  monthly_profit selling_price = target_profit :=
sorry

/-- Theorem stating that the selling price minimizes inventory -/
theorem selling_price_minimizes_inventory :
  ∀ (price : ℝ), monthly_profit price = target_profit → price ≥ selling_price :=
sorry

end NUMINAMATH_CALUDE_selling_price_achieves_target_profit_selling_price_minimizes_inventory_l271_27188


namespace NUMINAMATH_CALUDE_bailey_dog_treats_l271_27172

theorem bailey_dog_treats :
  let total_items : ℕ := 4 * 5
  let chew_toys : ℕ := 2
  let rawhide_bones : ℕ := 10
  let dog_treats : ℕ := total_items - (chew_toys + rawhide_bones)
  dog_treats = 8 := by
sorry

end NUMINAMATH_CALUDE_bailey_dog_treats_l271_27172
