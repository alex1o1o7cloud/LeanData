import Mathlib

namespace broken_flagpole_tip_height_l2692_269257

/-- Represents a broken flagpole -/
structure BrokenFlagpole where
  initial_height : ℝ
  break_height : ℝ
  folds_in_half : Bool

/-- Calculates the height of the tip of a broken flagpole from the ground -/
def tip_height (f : BrokenFlagpole) : ℝ :=
  if f.folds_in_half then f.break_height else f.initial_height

/-- Theorem stating that the height of the tip of a broken flagpole is equal to the break height -/
theorem broken_flagpole_tip_height 
  (f : BrokenFlagpole) 
  (h1 : f.initial_height = 12)
  (h2 : f.break_height = 7)
  (h3 : f.folds_in_half = true) :
  tip_height f = 7 := by
  sorry

end broken_flagpole_tip_height_l2692_269257


namespace modified_ohara_triple_solution_l2692_269236

/-- Definition of a Modified O'Hara Triple -/
def is_modified_ohara_triple (a b x k : ℕ+) : Prop :=
  k * (a : ℝ).sqrt + (b : ℝ).sqrt = x

/-- Theorem: If (49, 16, x, 2) is a Modified O'Hara Triple, then x = 18 -/
theorem modified_ohara_triple_solution :
  ∀ x : ℕ+, is_modified_ohara_triple 49 16 x 2 → x = 18 := by
  sorry

end modified_ohara_triple_solution_l2692_269236


namespace integer_division_implication_l2692_269235

theorem integer_division_implication (n : ℕ) (m : ℤ) :
  2^n - 2 = m * n →
  ∃ k : ℤ, (2^(2^n - 1) - 2) / (2^n - 1) = 2 * k :=
sorry

end integer_division_implication_l2692_269235


namespace square_of_105_l2692_269273

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end square_of_105_l2692_269273


namespace least_number_divisible_by_all_l2692_269218

def divisors : List Nat := [24, 32, 36, 54, 72, 81, 100]

theorem least_number_divisible_by_all (n : Nat) :
  (∀ d ∈ divisors, (n + 21) % d = 0) →
  (∀ m < n, ∃ d ∈ divisors, (m + 21) % d ≠ 0) →
  n = 64779 := by
  sorry

end least_number_divisible_by_all_l2692_269218


namespace smallest_triangle_area_l2692_269222

/-- The smallest area of a triangle with given vertices -/
theorem smallest_triangle_area :
  let A : ℝ × ℝ × ℝ := (-1, 1, 2)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ → ℝ → ℝ × ℝ × ℝ := fun t s ↦ (t, s, 1)
  let triangle_area (t s : ℝ) : ℝ :=
    (1 / 2) * Real.sqrt ((s^2) + ((-t-3)^2) + ((2*s-t-2)^2))
  ∃ (min_area : ℝ), min_area = Real.sqrt 58 / 2 ∧
    ∀ (t s : ℝ), triangle_area t s ≥ min_area :=
by
  sorry

end smallest_triangle_area_l2692_269222


namespace expand_expression_l2692_269246

theorem expand_expression (y : ℝ) : 5 * (4 * y^3 - 3 * y^2 + y - 7) = 20 * y^3 - 15 * y^2 + 5 * y - 35 := by
  sorry

end expand_expression_l2692_269246


namespace number_division_problem_l2692_269283

theorem number_division_problem :
  ∃ x : ℝ, x / 5 = 30 + x / 6 ∧ x = 900 := by
  sorry

end number_division_problem_l2692_269283


namespace min_abs_z_complex_l2692_269205

/-- Given a complex number z satisfying |z - 5i| + |z - 3| = 7, 
    the minimum value of |z| is 15/7 -/
theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 5*Complex.I) + Complex.abs (z - 3) = 7) :
  ∃ (w : ℂ), Complex.abs w = 15/7 ∧ ∀ (v : ℂ), Complex.abs (v - 5*Complex.I) + Complex.abs (v - 3) = 7 → Complex.abs w ≤ Complex.abs v :=
sorry

end min_abs_z_complex_l2692_269205


namespace mark_to_jaydon_ratio_l2692_269226

/-- Represents the number of cans brought by each person -/
structure Cans where
  rachel : ℕ
  jaydon : ℕ
  mark : ℕ

/-- The conditions of the food drive problem -/
def FoodDrive (c : Cans) : Prop :=
  c.mark = 100 ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  c.rachel + c.jaydon + c.mark = 135

/-- The theorem to be proved -/
theorem mark_to_jaydon_ratio (c : Cans) (h : FoodDrive c) : 
  c.mark / c.jaydon = 4 := by
  sorry

#check mark_to_jaydon_ratio

end mark_to_jaydon_ratio_l2692_269226


namespace seven_points_non_isosceles_l2692_269299

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define an isosceles triangle
def IsIsosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p3.x - p1.x)^2 + (p3.y - p1.y)^2 = (p3.x - p2.x)^2 + (p3.y - p2.y)^2

-- Main theorem
theorem seven_points_non_isosceles (points : Fin 7 → Point) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ¬IsIsosceles (points i) (points j) (points k) := by
  sorry

end seven_points_non_isosceles_l2692_269299


namespace largest_four_digit_sum_19_l2692_269230

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is four-digit -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Theorem stating that 9730 is the largest four-digit number whose digits add up to 19 -/
theorem largest_four_digit_sum_19 : 
  (∀ n : ℕ, is_four_digit n → sum_of_digits n = 19 → n ≤ 9730) ∧ 
  is_four_digit 9730 ∧ 
  sum_of_digits 9730 = 19 := by sorry

end largest_four_digit_sum_19_l2692_269230


namespace minimum_fourth_round_score_l2692_269223

def minimum_average_score : ℝ := 96
def number_of_rounds : ℕ := 4
def first_round_score : ℝ := 95
def second_round_score : ℝ := 97
def third_round_score : ℝ := 94

theorem minimum_fourth_round_score :
  let total_required_score := minimum_average_score * number_of_rounds
  let sum_of_first_three_rounds := first_round_score + second_round_score + third_round_score
  let minimum_fourth_round_score := total_required_score - sum_of_first_three_rounds
  minimum_fourth_round_score = 98 := by sorry

end minimum_fourth_round_score_l2692_269223


namespace parabola_maximum_l2692_269255

/-- The quadratic function f(x) = -x^2 - 1 -/
def f (x : ℝ) : ℝ := -x^2 - 1

theorem parabola_maximum :
  (∀ x : ℝ, f x ≤ f 0) ∧ f 0 = -1 :=
sorry

end parabola_maximum_l2692_269255


namespace E_72_with_4_equals_9_l2692_269212

/-- The number of ways to express an integer as a product of integers greater than 1 -/
def E (n : ℕ) : ℕ := sorry

/-- The number of ways to express 72 as a product of integers greater than 1,
    including at least one factor of 4, where the order of factors matters -/
def E_72_with_4 : ℕ := sorry

/-- The prime factorization of 72 -/
def prime_factorization_72 : List ℕ := [2, 2, 2, 3, 3]

theorem E_72_with_4_equals_9 : E_72_with_4 = 9 := by sorry

end E_72_with_4_equals_9_l2692_269212


namespace isosceles_triangle_perimeter_l2692_269264

/-- An isosceles triangle PQR with given side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  pq : ℝ
  qr : ℝ
  pr : ℝ
  -- Isosceles condition
  isIsosceles : pq = pr
  -- Given side lengths
  qr_eq : qr = 8
  pr_eq : pr = 10

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.pq + t.qr + t.pr

/-- Theorem: The perimeter of the given isosceles triangle is 28 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  perimeter t = 28 := by
  sorry


end isosceles_triangle_perimeter_l2692_269264


namespace roots_and_inequality_l2692_269298

/-- Given the equation ln x - (2a)/(x-1) = a with two distinct real roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ Real.log x₁ - (2*a)/(x₁-1) = a ∧ Real.log x₂ - (2*a)/(x₂-1) = a

theorem roots_and_inequality (a : ℝ) (h : has_two_distinct_roots a) :
  a > 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1/(Real.log x₁ + a) + 1/(Real.log x₂ + a) < 0 :=
sorry

end roots_and_inequality_l2692_269298


namespace min_horizontal_distance_l2692_269234

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - x - 6

/-- Theorem stating the minimum horizontal distance between points P and Q -/
theorem min_horizontal_distance :
  ∃ (xp xq : ℝ),
    f xp = 6 ∧
    f xq = -6 ∧
    ∀ (yp yq : ℝ),
      f yp = 6 → f yq = -6 →
      |xp - xq| ≤ |yp - yq| ∧
      |xp - xq| = 4 :=
sorry

end min_horizontal_distance_l2692_269234


namespace problem_solution_l2692_269281

-- Define the variables and functions
def f (x : ℝ) := 2 * x + 1
def g (x : ℝ) := x^2 + 2 * x

-- State the theorem
theorem problem_solution :
  ∃ (a b n : ℝ),
    f 2 = 5 ∧
    g 2 = a ∧
    f n = b ∧
    g n = -1 ∧
    a = 8 ∧
    b = 3 := by
  sorry

end problem_solution_l2692_269281


namespace fraction_equality_l2692_269242

theorem fraction_equality (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 := by
  sorry

end fraction_equality_l2692_269242


namespace rectangle_area_is_twelve_l2692_269287

/-- Represents a rectangle with given properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ
  perimeter : ℝ
  length_eq : length = 3 * width
  perimeter_eq : perimeter = 2 * (length + width)
  diagonal_eq : diagonal^2 = width^2 + length^2

/-- The area of a rectangle with specific properties is 12 -/
theorem rectangle_area_is_twelve (rect : Rectangle) (h : rect.perimeter = 16) : 
  rect.width * rect.length = 12 := by
  sorry

#check rectangle_area_is_twelve

end rectangle_area_is_twelve_l2692_269287


namespace intersection_of_lines_l2692_269271

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (3, -2, 6) → 
  B = (13, -12, 11) → 
  C = (1, 5, -3) → 
  D = (3, -1, 9) → 
  ∃ t s : ℝ, 
    (3 + 10*t, -2 - 10*t, 6 + 5*t) = 
    (1 + 2*s, 5 - 6*s, -3 + 12*s) ∧
    (3 + 10*t, -2 - 10*t, 6 + 5*t) = (7.5, -6.5, 8.25) := by
  sorry

#check intersection_of_lines

end intersection_of_lines_l2692_269271


namespace oil_barrels_problem_l2692_269269

theorem oil_barrels_problem (a b : ℝ) : 
  a > 0 ∧ b > 0 →  -- Initial amounts are positive
  (2/3 * a + 1/5 * (b + 1/3 * a) = 24) ∧  -- Amount in A after transfers
  ((b + 1/3 * a) * 4/5 = 24) →  -- Amount in B after transfers
  a - b = 6 := by sorry

end oil_barrels_problem_l2692_269269


namespace square_difference_49_50_l2692_269284

theorem square_difference_49_50 : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end square_difference_49_50_l2692_269284


namespace heesu_has_greatest_sum_l2692_269228

-- Define the card numbers for each player
def sora_cards : List Nat := [4, 6]
def heesu_cards : List Nat := [7, 5]
def jiyeon_cards : List Nat := [3, 8]

-- Define a function to calculate the sum of a player's cards
def sum_cards (cards : List Nat) : Nat :=
  cards.sum

-- Theorem statement
theorem heesu_has_greatest_sum :
  sum_cards heesu_cards > sum_cards sora_cards ∧
  sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  sorry


end heesu_has_greatest_sum_l2692_269228


namespace theo_cookie_eating_frequency_l2692_269262

/-- The number of cookies Theo eats each time -/
def cookies_per_time : ℕ := 13

/-- The number of days Theo eats cookies each month -/
def days_per_month : ℕ := 20

/-- The number of cookies Theo eats in 3 months -/
def cookies_in_three_months : ℕ := 2340

/-- The number of times Theo eats cookies per day -/
def times_per_day : ℕ := 3

theorem theo_cookie_eating_frequency :
  times_per_day * cookies_per_time * days_per_month * 3 = cookies_in_three_months :=
by sorry

end theo_cookie_eating_frequency_l2692_269262


namespace p_and_q_implies_m_leq_1_l2692_269248

/-- Proposition p: For all x ∈ ℝ, the function y = log₂(2ˣ - m + 1) is defined. -/
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, 2^x - m + 1 > 0

/-- Proposition q: The function f(x) = (5 - 2m)ˣ is increasing. -/
def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- If propositions p and q are true, then m ≤ 1. -/
theorem p_and_q_implies_m_leq_1 (m : ℝ) :
  proposition_p m ∧ proposition_q m → m ≤ 1 := by
  sorry

end p_and_q_implies_m_leq_1_l2692_269248


namespace coefficient_is_nine_l2692_269293

/-- The coefficient of x^2 in the expansion of (1+x)^10 - (1-x)^9 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 10 2) - (Nat.choose 9 2)

/-- Theorem stating that the coefficient of x^2 in the expansion of (1+x)^10 - (1-x)^9 is 9 -/
theorem coefficient_is_nine : coefficient_x_squared = 9 := by
  sorry

end coefficient_is_nine_l2692_269293


namespace boys_left_bakery_l2692_269220

/-- The number of boys who left the bakery --/
def boys_who_left (initial_children : ℕ) (girls_came_in : ℕ) (final_children : ℕ) : ℕ :=
  initial_children + girls_came_in - final_children

theorem boys_left_bakery (initial_children : ℕ) (girls_came_in : ℕ) (final_children : ℕ)
  (h1 : initial_children = 85)
  (h2 : girls_came_in = 24)
  (h3 : final_children = 78) :
  boys_who_left initial_children girls_came_in final_children = 31 := by
  sorry

#eval boys_who_left 85 24 78

end boys_left_bakery_l2692_269220


namespace ellipse_standard_form_l2692_269200

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    passing through the point (√6/2, 1/2), and having an eccentricity of √2/2,
    prove that the standard form of the ellipse equation is (x²/a²) + (y²/b²) = 1 -/
theorem ellipse_standard_form 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (Real.sqrt 6 / 2)^2 / a^2 + (1/2)^2 / b^2 = 1)
  (h4 : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2) :
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
by sorry

end ellipse_standard_form_l2692_269200


namespace geometric_sequence_ratio_l2692_269258

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  third_term : a 3 = 2
  product_46 : a 4 * a 6 = 16

/-- The main theorem -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence) :
  (seq.a 9 - seq.a 11) / (seq.a 5 - seq.a 7) = 4 := by
  sorry


end geometric_sequence_ratio_l2692_269258


namespace constant_term_binomial_expansion_l2692_269291

theorem constant_term_binomial_expansion :
  let f := fun (x : ℝ) => (x - 1 / (2 * Real.sqrt x)) ^ 6
  ∃ (c : ℝ), (∀ (x : ℝ), x > 0 → f x = c + x * (f x - c) / x) ∧ c = 15/16 :=
by sorry

end constant_term_binomial_expansion_l2692_269291


namespace equal_selection_probability_l2692_269254

/-- Represents the probability of a student being selected -/
def probability_of_selection (n : ℕ) (total : ℕ) : ℚ := n / total

theorem equal_selection_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (eliminated_students : ℕ) 
  (h1 : total_students = 54) 
  (h2 : selected_students = 5) 
  (h3 : eliminated_students = 4) :
  ∀ (student : ℕ), student ≤ total_students → 
    probability_of_selection selected_students total_students = 5 / 54 :=
by sorry

end equal_selection_probability_l2692_269254


namespace division_of_decimals_l2692_269201

theorem division_of_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by
  sorry

end division_of_decimals_l2692_269201


namespace james_touchdown_points_l2692_269263

/-- The number of points per touchdown in James' football season -/
def points_per_touchdown : ℕ := by sorry

/-- The number of touchdowns James scores per game -/
def touchdowns_per_game : ℕ := 4

/-- The number of games in the season -/
def games_in_season : ℕ := 15

/-- The number of 2-point conversions James scores in the season -/
def two_point_conversions : ℕ := 6

/-- The total points James scores in the season -/
def total_points : ℕ := 372

theorem james_touchdown_points :
  points_per_touchdown * touchdowns_per_game * games_in_season +
  2 * two_point_conversions = total_points ∧
  points_per_touchdown = 6 := by sorry

end james_touchdown_points_l2692_269263


namespace binomial_12_choose_3_l2692_269272

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by sorry

end binomial_12_choose_3_l2692_269272


namespace ivans_bird_feeder_feeds_21_l2692_269297

/-- Calculates the number of birds fed weekly by a bird feeder --/
def birds_fed_weekly (feeder_capacity : ℝ) (birds_per_cup : ℝ) (stolen_amount : ℝ) : ℝ :=
  (feeder_capacity - stolen_amount) * birds_per_cup

/-- Theorem: Ivan's bird feeder feeds 21 birds weekly --/
theorem ivans_bird_feeder_feeds_21 :
  birds_fed_weekly 2 14 0.5 = 21 := by
  sorry

end ivans_bird_feeder_feeds_21_l2692_269297


namespace abs_opposite_equal_l2692_269238

theorem abs_opposite_equal (x : ℝ) : |x| = |-x| := by sorry

end abs_opposite_equal_l2692_269238


namespace base4_division_l2692_269280

/-- Convert a number from base 4 to decimal --/
def base4ToDecimal (n : List Nat) : Nat :=
  n.foldr (fun digit acc => acc * 4 + digit) 0

/-- Convert a number from decimal to base 4 --/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem: 12345₄ divided by 23₄ equals 535₄ in base 4 --/
theorem base4_division :
  let dividend := base4ToDecimal [1, 2, 3, 4, 5]
  let divisor := base4ToDecimal [2, 3]
  let quotient := base4ToDecimal [5, 3, 5]
  decimalToBase4 (dividend / divisor) = [5, 3, 5] :=
by sorry

end base4_division_l2692_269280


namespace calculator_key_functions_l2692_269214

/-- Represents the keys on a calculator --/
inductive CalculatorKey
  | ON_C
  | OFF
  | Other

/-- Represents the functions of calculator keys --/
inductive KeyFunction
  | ClearScreen
  | PowerOff
  | Other

/-- Maps calculator keys to their functions --/
def key_function : CalculatorKey → KeyFunction
  | CalculatorKey.ON_C => KeyFunction.ClearScreen
  | CalculatorKey.OFF => KeyFunction.PowerOff
  | CalculatorKey.Other => KeyFunction.Other

theorem calculator_key_functions :
  (key_function CalculatorKey.ON_C = KeyFunction.ClearScreen) ∧
  (key_function CalculatorKey.OFF = KeyFunction.PowerOff) :=
by sorry

end calculator_key_functions_l2692_269214


namespace first_customer_boxes_l2692_269245

def cookie_problem (x : ℚ) : Prop :=
  let second_customer := 4 * x
  let third_customer := second_customer / 2
  let fourth_customer := 3 * third_customer
  let final_customer := 10
  let total_sold := x + second_customer + third_customer + fourth_customer + final_customer
  let goal := 150
  let left_to_sell := 75
  total_sold + left_to_sell = goal

theorem first_customer_boxes : ∃ x : ℚ, cookie_problem x ∧ x = 5 := by
  sorry

end first_customer_boxes_l2692_269245


namespace polynomial_condition_implies_linear_l2692_269227

/-- A polynomial with real coefficients -/
def RealPolynomial : Type := ℝ → ℝ

/-- The condition that P(x + y) is rational when P(x) and P(y) are rational -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, (∃ q₁ q₂ : ℚ, P x = q₁ ∧ P y = q₂) → ∃ q : ℚ, P (x + y) = q

/-- The theorem stating that polynomials satisfying the condition must be linear with rational coefficients -/
theorem polynomial_condition_implies_linear
  (P : RealPolynomial)
  (h : SatisfiesCondition P) :
  ∃ a b : ℚ, ∀ x : ℝ, P x = a * x + b :=
sorry

end polynomial_condition_implies_linear_l2692_269227


namespace sphere_intersection_ratio_l2692_269249

/-- Two spheres with radii R₁ and R₂ are intersected by a plane P perpendicular to the line
    connecting their centers and passing through its midpoint. If P divides the surface area
    of the first sphere in ratio m:1 and the second sphere in ratio n:1 (where m > 1 and n > 1),
    then R₂/R₁ = ((m - 1)(n + 1)) / ((m + 1)(n - 1)). -/
theorem sphere_intersection_ratio (R₁ R₂ m n : ℝ) (hm : m > 1) (hn : n > 1) :
  let h₁ := (2 * R₁) / (m + 1)
  let h₂ := (2 * R₂) / (n + 1)
  R₁ - h₁ = R₂ - h₂ →
  R₂ / R₁ = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := by
  sorry

end sphere_intersection_ratio_l2692_269249


namespace sin_4x_eq_sin_2x_solutions_l2692_269221

open Set
open Real

theorem sin_4x_eq_sin_2x_solutions :
  let S := {x : ℝ | 0 < x ∧ x < (3/2)*π ∧ sin (4*x) = sin (2*x)}
  S = {π/6, π/2, π, 5*π/6, 7*π/6} := by sorry

end sin_4x_eq_sin_2x_solutions_l2692_269221


namespace smallest_visible_sum_l2692_269296

/-- Represents a small die in the 4x4x4 cube -/
structure SmallDie where
  /-- The value on each face of the die -/
  faces : Fin 6 → ℕ
  /-- The property that opposite sides sum to 7 -/
  opposite_sum_seven : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the 4x4x4 cube made of small dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → SmallDie

/-- Calculates the sum of visible values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ := sorry

/-- Theorem stating the smallest possible sum of visible values -/
theorem smallest_visible_sum (cube : LargeCube) :
  visible_sum cube ≥ 144 ∧ ∃ (optimal_cube : LargeCube), visible_sum optimal_cube = 144 := by sorry

end smallest_visible_sum_l2692_269296


namespace greatest_n_value_exists_greatest_n_l2692_269211

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

theorem exists_greatest_n :
  ∃ n : ℤ, n = 8 ∧ 101 * n^2 ≤ 8100 ∧ ∀ m : ℤ, 101 * m^2 ≤ 8100 → m ≤ n :=
sorry

end greatest_n_value_exists_greatest_n_l2692_269211


namespace problem_statement_l2692_269259

theorem problem_statement : (36 / (7 + 2 - 5)) * 4 = 36 := by
  sorry

end problem_statement_l2692_269259


namespace no_binomial_arithmetic_progression_l2692_269241

theorem no_binomial_arithmetic_progression :
  ∀ (n k : ℕ+), k ≤ n →
    ¬∃ (d : ℚ), 
      (Nat.choose n (k + 1) : ℚ) - (Nat.choose n k : ℚ) = d ∧
      (Nat.choose n (k + 2) : ℚ) - (Nat.choose n (k + 1) : ℚ) = d ∧
      (Nat.choose n (k + 3) : ℚ) - (Nat.choose n (k + 2) : ℚ) = d :=
by sorry

end no_binomial_arithmetic_progression_l2692_269241


namespace equation_solutions_l2692_269243

/-- The equation x^4 * y^4 - 10 * x^2 * y^2 + 9 = 0 -/
def equation (x y : ℕ+) : Prop :=
  (x.val : ℝ)^4 * (y.val : ℝ)^4 - 10 * (x.val : ℝ)^2 * (y.val : ℝ)^2 + 9 = 0

/-- The set of all ordered pairs (x,y) of positive integers satisfying the equation -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | equation p.1 p.2}

theorem equation_solutions :
  ∃ (s : Finset (ℕ+ × ℕ+)), s.card = 3 ∧ ↑s = solution_set := by sorry

end equation_solutions_l2692_269243


namespace video_votes_l2692_269250

theorem video_votes (score : ℕ) (like_percent : ℚ) (dislike_percent : ℚ) (neutral_percent : ℚ) :
  score = 180 →
  like_percent = 60 / 100 →
  dislike_percent = 20 / 100 →
  neutral_percent = 20 / 100 →
  like_percent + dislike_percent + neutral_percent = 1 →
  ∃ (total_votes : ℕ), 
    (↑score : ℚ) = (like_percent - dislike_percent) * ↑total_votes ∧
    total_votes = 450 := by
  sorry

end video_votes_l2692_269250


namespace sequence_general_term_l2692_269294

/-- Given a sequence {a_n} with the sum of the first n terms S_n satisfying
    S_n + a_n = (n-1) / (n(n+1)) for n = 1, 2, ..., 
    prove that the general term a_n = 1/(2^n) - 1/(n(n+1)). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n + a n = (n - 1) / (n * (n + 1))) →
  (∀ n : ℕ, n ≥ 1 → a n = 1 / (2^n) - 1 / (n * (n + 1))) :=
by sorry

end sequence_general_term_l2692_269294


namespace min_crystals_to_kill_120_l2692_269279

structure Skill where
  name : String
  crystalCost : ℕ
  damage : ℕ
  specialEffect : Bool

def applySkill (health : ℕ) (skill : Skill) (prevWindUsed : Bool) : ℕ × ℕ :=
  let actualCost := if prevWindUsed then skill.crystalCost / 2 else skill.crystalCost
  let newHealth := 
    if skill.name = "Earth" then
      if health % 2 = 1 then (health + 1) / 2 else health / 2
    else
      if health > skill.damage then health - skill.damage else 0
  (newHealth, actualCost)

def minCrystalsToKill (initialHealth : ℕ) (water fire wind earth : Skill) : ℕ :=
  sorry

theorem min_crystals_to_kill_120 :
  let water : Skill := ⟨"Water", 4, 4, false⟩
  let fire : Skill := ⟨"Fire", 10, 11, false⟩
  let wind : Skill := ⟨"Wind", 10, 5, true⟩
  let earth : Skill := ⟨"Earth", 18, 0, false⟩
  minCrystalsToKill 120 water fire wind earth = 68 := by
  sorry

end min_crystals_to_kill_120_l2692_269279


namespace lemonade_juice_requirement_l2692_269268

/-- The amount of lemon juice required for a lemonade mixture -/
def lemon_juice_required (total_volume : ℚ) (water_parts : ℕ) (juice_parts : ℕ) : ℚ :=
  (total_volume * juice_parts) / (water_parts + juice_parts)

/-- Conversion from gallons to quarts -/
def gallons_to_quarts (gallons : ℚ) : ℚ := 4 * gallons

theorem lemonade_juice_requirement :
  let total_volume := (3 : ℚ) / 2  -- 1.5 gallons
  let water_parts := 5
  let juice_parts := 3
  lemon_juice_required (gallons_to_quarts total_volume) water_parts juice_parts = (9 : ℚ) / 4 := by
  sorry

end lemonade_juice_requirement_l2692_269268


namespace park_tree_removal_l2692_269206

/-- The number of trees removed from a park -/
def trees_removed (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem: Given 6 initial trees and 2 remaining trees, 4 trees are removed -/
theorem park_tree_removal :
  trees_removed 6 2 = 4 := by
  sorry

end park_tree_removal_l2692_269206


namespace percentage_60to69_is_20_percent_l2692_269210

/-- Represents the score ranges in the class --/
inductive ScoreRange
  | Below60
  | Range60to69
  | Range70to79
  | Range80to89
  | Range90to100

/-- The frequency of students for each score range --/
def frequency (range : ScoreRange) : Nat :=
  match range with
  | .Below60 => 2
  | .Range60to69 => 5
  | .Range70to79 => 6
  | .Range80to89 => 8
  | .Range90to100 => 4

/-- The total number of students in the class --/
def totalStudents : Nat :=
  frequency ScoreRange.Below60 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range90to100

/-- The percentage of students in the 60%-69% range --/
def percentageIn60to69Range : Rat :=
  (frequency ScoreRange.Range60to69 : Rat) / (totalStudents : Rat) * 100

theorem percentage_60to69_is_20_percent :
  percentageIn60to69Range = 20 := by
  sorry

#eval percentageIn60to69Range

end percentage_60to69_is_20_percent_l2692_269210


namespace odd_number_as_difference_of_squares_l2692_269203

theorem odd_number_as_difference_of_squares (n : ℤ) : 
  2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end odd_number_as_difference_of_squares_l2692_269203


namespace smallest_n_for_inequality_l2692_269277

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z : ℝ), (x^2 + 2*y^2 + z^2)^2 ≤ n*(x^4 + 3*y^4 + z^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x^2 + 2*y^2 + z^2)^2 > m*(x^4 + 3*y^4 + z^4)) :=
by sorry

end smallest_n_for_inequality_l2692_269277


namespace parabola_focus_distance_l2692_269267

/-- Given a parabola y² = -2x with focus F, and a point A(x₀, y₀) on the parabola,
    if |AF| = 3/2, then x₀ = -1 -/
theorem parabola_focus_distance (x₀ y₀ : ℝ) :
  y₀^2 = -2*x₀ →  -- A is on the parabola
  ∃ F : ℝ × ℝ, (F.1 = 1/2 ∧ F.2 = 0) →  -- Focus coordinates
  (x₀ - F.1)^2 + (y₀ - F.2)^2 = (3/2)^2 →  -- |AF| = 3/2
  x₀ = -1 := by
sorry

end parabola_focus_distance_l2692_269267


namespace intersection_A_B_when_a_is_3_A_subset_B_iff_a_greater_than_5_l2692_269217

-- Define set A
def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 4}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part I
theorem intersection_A_B_when_a_is_3 :
  A ∩ B 3 = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for part II
theorem A_subset_B_iff_a_greater_than_5 :
  ∀ a : ℝ, A ⊆ B a ↔ a > 5 := by sorry

end intersection_A_B_when_a_is_3_A_subset_B_iff_a_greater_than_5_l2692_269217


namespace sum_of_four_integers_l2692_269274

theorem sum_of_four_integers (a b c d : ℕ+) : 
  (a > 1) → (b > 1) → (c > 1) → (d > 1) →
  (a * b * c * d = 1000000) →
  (Nat.gcd a.val b.val = 1) → (Nat.gcd a.val c.val = 1) → (Nat.gcd a.val d.val = 1) →
  (Nat.gcd b.val c.val = 1) → (Nat.gcd b.val d.val = 1) →
  (Nat.gcd c.val d.val = 1) →
  (a + b + c + d = 15698) := by
sorry

end sum_of_four_integers_l2692_269274


namespace gloria_turtle_time_l2692_269256

/-- The time it took for Gloria's turtle to finish the race -/
def glorias_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

theorem gloria_turtle_time : ∃ (gretas_time georges_time : ℕ),
  gretas_time = 6 ∧
  georges_time = gretas_time - 2 ∧
  glorias_time gretas_time georges_time = 8 := by
  sorry

end gloria_turtle_time_l2692_269256


namespace cost_of_paints_paint_set_cost_l2692_269204

theorem cost_of_paints (total_spent : ℕ) (num_classes : ℕ) (folders_per_class : ℕ) 
  (pencils_per_class : ℕ) (pencils_per_eraser : ℕ) (folder_cost : ℕ) (pencil_cost : ℕ) 
  (eraser_cost : ℕ) : ℕ :=
  let num_folders := num_classes * folders_per_class
  let num_pencils := num_classes * pencils_per_class
  let num_erasers := num_pencils / pencils_per_eraser
  let folders_total_cost := num_folders * folder_cost
  let pencils_total_cost := num_pencils * pencil_cost
  let erasers_total_cost := num_erasers * eraser_cost
  let supplies_cost := folders_total_cost + pencils_total_cost + erasers_total_cost
  total_spent - supplies_cost

theorem paint_set_cost : cost_of_paints 80 6 1 3 6 6 2 1 = 5 := by
  sorry

end cost_of_paints_paint_set_cost_l2692_269204


namespace last_two_nonzero_digits_75_factorial_l2692_269253

theorem last_two_nonzero_digits_75_factorial (n : ℕ) : n = 75 → 
  ∃ k : ℕ, n.factorial = 100 * k + 76 ∧ k % 10 ≠ 0 :=
by sorry

end last_two_nonzero_digits_75_factorial_l2692_269253


namespace customized_packaging_combinations_l2692_269285

def wrapping_papers : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

theorem customized_packaging_combinations : 
  wrapping_papers * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end customized_packaging_combinations_l2692_269285


namespace three_numbers_in_unit_interval_l2692_269251

theorem three_numbers_in_unit_interval (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) :=
by sorry

end three_numbers_in_unit_interval_l2692_269251


namespace largest_in_set_l2692_269225

def S : Set ℝ := {0.109, 0.2, 0.111, 0.114, 0.19}

theorem largest_in_set : ∀ x ∈ S, x ≤ 0.2 := by sorry

end largest_in_set_l2692_269225


namespace quadratic_roots_l2692_269265

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop :=
  (a - 1) * x^2 + 2 * x + a - 1 = 0

theorem quadratic_roots :
  -- Part 1
  (∃ a : ℝ, quadratic_equation a 2 ∧ 
    ∃ x : ℝ, x ≠ 2 ∧ quadratic_equation a x) →
  (quadratic_equation (1/5) 2 ∧ quadratic_equation (1/5) (1/2)) ∧
  -- Part 2
  (∃ x : ℝ, quadratic_equation 1 x ↔ x = 0) ∧
  (∃ x : ℝ, quadratic_equation 2 x ↔ x = -1) ∧
  (∃ x : ℝ, quadratic_equation 0 x ↔ x = 1) :=
sorry

end quadratic_roots_l2692_269265


namespace geometric_series_relation_l2692_269229

/-- Given real numbers x and y satisfying an infinite geometric series equation,
    prove that another related infinite geometric series has a specific value. -/
theorem geometric_series_relation (x y : ℝ) 
  (h : (x / y) / (1 - 1 / y) = 3) :
  (x / (x + 2 * y)) / (1 - 1 / (x + 2 * y)) = 3 * (y - 1) / (5 * y - 4) := by
  sorry

end geometric_series_relation_l2692_269229


namespace gathering_handshakes_l2692_269209

/-- Represents a gathering of couples -/
structure Gathering where
  couples : Nat
  people : Nat
  men : Nat
  women : Nat

/-- Calculates the number of handshakes in a gathering -/
def handshakes (g : Gathering) : Nat :=
  g.men * (g.women - 1)

/-- Theorem: In a gathering of 7 couples with the given handshake rules, 
    the total number of handshakes is 42 -/
theorem gathering_handshakes :
  ∀ g : Gathering, 
    g.couples = 7 →
    g.people = 2 * g.couples →
    g.men = g.couples →
    g.women = g.couples →
    handshakes g = 42 := by
  sorry

end gathering_handshakes_l2692_269209


namespace cos_five_pi_sixth_minus_alpha_l2692_269216

theorem cos_five_pi_sixth_minus_alpha (α : ℝ) (h : Real.cos (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 - α) = -(Real.sqrt 3 / 3) := by
  sorry

end cos_five_pi_sixth_minus_alpha_l2692_269216


namespace probability_theorem_l2692_269270

def is_valid (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧
  1 ≤ b ∧ b ≤ 60 ∧
  1 ≤ c ∧ c ≤ 60 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def satisfies_condition (a b c : ℕ) : Prop :=
  ∃ m : ℕ, (a * b * c + a + b + c) = 6 * m - 2

def total_combinations : ℕ := Nat.choose 60 3

def valid_combinations : ℕ := 14620

theorem probability_theorem :
  (valid_combinations : ℚ) / total_combinations = 2437 / 5707 := by sorry

end probability_theorem_l2692_269270


namespace rectangle_area_l2692_269213

theorem rectangle_area (a c : ℝ) (ha : a = 15) (hc : c = 17) : 
  ∃ b : ℝ, a * b = 120 ∧ a^2 + b^2 = c^2 := by
  sorry

end rectangle_area_l2692_269213


namespace exists_consecutive_numbers_with_54_digit_product_ratio_l2692_269239

/-- Given a natural number, return the product of its non-zero digits -/
def productOfNonZeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that there exist two consecutive natural numbers
    such that the product of all non-zero digits of the larger number
    multiplied by 54 equals the product of all non-zero digits of the smaller number -/
theorem exists_consecutive_numbers_with_54_digit_product_ratio :
  ∃ n : ℕ, productOfNonZeroDigits n = 54 * productOfNonZeroDigits (n + 1) := by
  sorry

end exists_consecutive_numbers_with_54_digit_product_ratio_l2692_269239


namespace inverse_square_relation_l2692_269240

/-- Given that x varies inversely as the square of y, and y = 3 when x = 1,
    prove that x = 0.5625 when y = 4. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y ^ 2)) →  -- x varies inversely as the square of y
  (1 = k / (3 ^ 2)) →               -- y = 3 when x = 1
  (k = 9) →                         -- derived from the previous condition
  (x = 9 / (4 ^ 2)) →               -- x when y = 4
  x = 0.5625 := by
sorry

end inverse_square_relation_l2692_269240


namespace gold_alloy_composition_l2692_269286

/-- Proves that adding 24 ounces of pure gold to a 16-ounce alloy that is 50% gold
    will result in an alloy that is 80% gold. -/
theorem gold_alloy_composition (initial_weight : ℝ) (initial_purity : ℝ) 
    (added_gold : ℝ) (final_purity : ℝ) : 
  initial_weight = 16 →
  initial_purity = 0.5 →
  added_gold = 24 →
  final_purity = 0.8 →
  (initial_weight * initial_purity + added_gold) / (initial_weight + added_gold) = final_purity :=
by
  sorry

#check gold_alloy_composition

end gold_alloy_composition_l2692_269286


namespace fifteen_sided_polygon_diagonals_l2692_269260

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end fifteen_sided_polygon_diagonals_l2692_269260


namespace jennifer_spending_l2692_269282

theorem jennifer_spending (initial_amount : ℚ) : 
  (initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 16) → 
  initial_amount = 120 := by
sorry

end jennifer_spending_l2692_269282


namespace unique_solution_l2692_269275

theorem unique_solution : ∃! (n : ℕ+), 
  Real.sin (π / (3 * n.val : ℝ)) + Real.cos (π / (3 * n.val : ℝ)) = Real.sqrt (2 * n.val : ℝ) / 2 :=
by sorry

end unique_solution_l2692_269275


namespace fourth_guard_distance_theorem_l2692_269276

/-- Represents a rectangular classified area with guards -/
structure ClassifiedArea where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  guard_count : ℕ
  three_guards_distance : ℝ

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (area : ClassifiedArea) : ℝ :=
  area.perimeter - area.three_guards_distance

/-- Theorem stating the distance run by the fourth guard -/
theorem fourth_guard_distance_theorem (area : ClassifiedArea) 
  (h1 : area.length = 200)
  (h2 : area.width = 300)
  (h3 : area.perimeter = 2 * (area.length + area.width))
  (h4 : area.guard_count = 4)
  (h5 : area.three_guards_distance = 850)
  : fourth_guard_distance area = 150 := by
  sorry

end fourth_guard_distance_theorem_l2692_269276


namespace fractional_sum_equality_l2692_269224

theorem fractional_sum_equality (n : ℕ) (h : n > 1) :
  ∃ i j : ℕ, (1 : ℚ) / n = 
    Finset.sum (Finset.range (j - i + 1)) (λ k => 1 / ((i + k) * (i + k + 1))) := by
  sorry

end fractional_sum_equality_l2692_269224


namespace prime_pair_product_l2692_269233

theorem prime_pair_product : ∃ p q : ℕ, 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Odd (p + q) ∧ 
  p + q < 100 ∧ 
  (∃ k : ℕ, p + q = 17 * k) ∧ 
  p * q = 166 := by
  sorry

end prime_pair_product_l2692_269233


namespace shaded_area_is_60_l2692_269208

/-- Represents a point in a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangular grid -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Represents the shaded region in the grid -/
structure ShadedRegion where
  grid : Grid
  points : List Point

/-- Calculates the area of the shaded region -/
def shadedArea (region : ShadedRegion) : ℕ :=
  sorry

/-- The specific grid and shaded region from the problem -/
def problemGrid : Grid :=
  { width := 15, height := 5 }

def problemShadedRegion : ShadedRegion :=
  { grid := problemGrid,
    points := [
      { x := 0, y := 0 },   -- bottom left corner
      { x := 4, y := 3 },   -- first point
      { x := 9, y := 5 },   -- second point
      { x := 15, y := 5 }   -- top right corner
    ] }

/-- The main theorem to prove -/
theorem shaded_area_is_60 :
  shadedArea problemShadedRegion = 60 :=
sorry

end shaded_area_is_60_l2692_269208


namespace solution_set_transformation_l2692_269290

theorem solution_set_transformation (k a b c : ℝ) :
  (∀ x, (x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 2 3) ↔ 
    (k * x / (a * x - 1) + (b * x - 1) / (c * x - 1) < 0)) →
  (∀ x, (x ∈ Set.Ioo (-1/2 : ℝ) (-1/3) ∪ Set.Ioo (1/2) 1) ↔ 
    (k / (x + a) + (x + b) / (x + c) < 0)) :=
by sorry

end solution_set_transformation_l2692_269290


namespace polynomial_divisibility_l2692_269202

theorem polynomial_divisibility (k l m n : ℕ) : 
  ∃ q : Polynomial ℤ, (X^4*k + X^(4*l+1) + X^(4*m+2) + X^(4*n+3)) = (X^3 + X^2 + X + 1) * q := by
  sorry

end polynomial_divisibility_l2692_269202


namespace additive_function_characterization_l2692_269219

/-- A function satisfying the given functional equation -/
def AdditiveFunctionQ (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

/-- The main theorem characterizing additive functions on rationals -/
theorem additive_function_characterization :
  ∀ f : ℚ → ℚ, AdditiveFunctionQ f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x := by
  sorry

end additive_function_characterization_l2692_269219


namespace max_sum_of_sides_l2692_269252

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  (2 * t.c - t.b) / t.a = (Real.cos t.B) / (Real.cos t.A)

def side_a_condition (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 5

-- Theorem statement
theorem max_sum_of_sides (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : side_a_condition t) : 
  ∃ (max : Real), ∀ (t' : Triangle), 
    satisfies_condition t' → side_a_condition t' → 
    t'.b + t'.c ≤ max ∧ 
    ∃ (t'' : Triangle), satisfies_condition t'' ∧ side_a_condition t'' ∧ t''.b + t''.c = max ∧
    max = 4 * Real.sqrt 5 :=
sorry

end max_sum_of_sides_l2692_269252


namespace product_modulo_remainder_1491_2001_mod_250_l2692_269244

theorem product_modulo (a b m : ℕ) (h : m > 0) :
  (a * b) % m = ((a % m) * (b % m)) % m :=
by sorry

theorem remainder_1491_2001_mod_250 :
  (1491 * 2001) % 250 = 241 :=
by sorry

end product_modulo_remainder_1491_2001_mod_250_l2692_269244


namespace divisibility_by_ten_l2692_269278

theorem divisibility_by_ten (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a * b * c) % 10 = 0 ∧
  (a * b * d) % 10 = 0 ∧
  (a * b * e) % 10 = 0 ∧
  (a * c * d) % 10 = 0 ∧
  (a * c * e) % 10 = 0 ∧
  (a * d * e) % 10 = 0 ∧
  (b * c * d) % 10 = 0 ∧
  (b * c * e) % 10 = 0 ∧
  (b * d * e) % 10 = 0 ∧
  (c * d * e) % 10 = 0 →
  a % 10 = 0 ∨ b % 10 = 0 ∨ c % 10 = 0 ∨ d % 10 = 0 ∨ e % 10 = 0 :=
by sorry

end divisibility_by_ten_l2692_269278


namespace points_on_line_procedure_l2692_269289

theorem points_on_line_procedure (x : ℕ) : ∃ x, 9 * x - 8 = 82 := by
  sorry

end points_on_line_procedure_l2692_269289


namespace yanna_purchase_l2692_269261

def shirts_cost : ℕ := 10 * 5
def sandals_cost : ℕ := 3 * 3
def hats_cost : ℕ := 5 * 8
def bags_cost : ℕ := 7 * 14
def sunglasses_cost : ℕ := 2 * 12

def total_cost : ℕ := shirts_cost + sandals_cost + hats_cost + bags_cost + sunglasses_cost
def payment : ℕ := 200

theorem yanna_purchase :
  total_cost = payment + 21 :=
by sorry

end yanna_purchase_l2692_269261


namespace inequality_proof_l2692_269266

theorem inequality_proof (x y : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y := by
  sorry

end inequality_proof_l2692_269266


namespace set_union_implies_a_zero_l2692_269207

theorem set_union_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {2^a, 3}
  let B : Set ℝ := {2, 3}
  A ∪ B = {1, 2, 3} → a = 0 :=
by
  sorry

end set_union_implies_a_zero_l2692_269207


namespace only_solutions_l2692_269215

/-- A four-digit number is composed of two two-digit numbers x and y -/
def is_valid_four_digit (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 10 ∧ x < 100 ∧ y ≥ 10 ∧ y < 100 ∧ n = 100 * x + y

/-- The condition that the square of the sum of x and y equals the four-digit number -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (x y : ℕ), is_valid_four_digit n ∧ (x + y)^2 = n

/-- The theorem stating that 3025 and 2025 are the only solutions -/
theorem only_solutions : ∀ (n : ℕ), satisfies_condition n ↔ (n = 3025 ∨ n = 2025) :=
sorry

end only_solutions_l2692_269215


namespace number_of_points_l2692_269288

theorem number_of_points (initial_sum : ℝ) (shift : ℝ) (final_sum : ℝ) : 
  initial_sum = -1.5 → 
  shift = -2 → 
  final_sum = -15.5 → 
  (final_sum - initial_sum) / shift = 7 := by
sorry

end number_of_points_l2692_269288


namespace total_balloons_l2692_269232

theorem total_balloons (tom_balloons sara_balloons : ℕ) 
  (h1 : tom_balloons = 9) 
  (h2 : sara_balloons = 8) : 
  tom_balloons + sara_balloons = 17 := by
sorry

end total_balloons_l2692_269232


namespace function_characterization_l2692_269292

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : f 0 = 1) 
  (h2 : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x : ℝ, f x = 1 - 2 * x := by
sorry

end function_characterization_l2692_269292


namespace logarithmic_equation_proof_l2692_269295

theorem logarithmic_equation_proof : 2 * (Real.log 10 / Real.log 5) + Real.log (1/4) / Real.log 5 + 2^(Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end logarithmic_equation_proof_l2692_269295


namespace root_product_theorem_l2692_269231

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 5 = 0) → 
  (b^2 - m*b + 5 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 36/5 := by
sorry

end root_product_theorem_l2692_269231


namespace quadratic_rewrite_sum_l2692_269237

/-- Given a quadratic expression x^2 - 24x + 50 that can be rewritten as (x+d)^2 + e,
    this theorem states that d + e = -106. -/
theorem quadratic_rewrite_sum (d e : ℝ) : 
  (∀ x, x^2 - 24*x + 50 = (x + d)^2 + e) → d + e = -106 := by
  sorry

end quadratic_rewrite_sum_l2692_269237


namespace cookie_making_time_l2692_269247

/-- Given the total time to make cookies, baking time, and icing hardening times,
    prove that the time for making dough and cooling cookies is 45 minutes. -/
theorem cookie_making_time (total_time baking_time white_icing_time chocolate_icing_time : ℕ)
  (h1 : total_time = 120)
  (h2 : baking_time = 15)
  (h3 : white_icing_time = 30)
  (h4 : chocolate_icing_time = 30) :
  total_time - (baking_time + white_icing_time + chocolate_icing_time) = 45 := by
  sorry

#check cookie_making_time

end cookie_making_time_l2692_269247
