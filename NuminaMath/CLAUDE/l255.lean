import Mathlib

namespace petes_flag_shapes_l255_25504

/-- Given a flag with circles and squares, calculate the total number of shapes --/
def total_shapes (stars : ℕ) (stripes : ℕ) : ℕ :=
  let circles := stars / 2 - 3
  let squares := stripes * 2 + 6
  circles + squares

/-- Theorem: The total number of shapes on Pete's flag is 54 --/
theorem petes_flag_shapes :
  total_shapes 50 13 = 54 := by
  sorry

end petes_flag_shapes_l255_25504


namespace square_perimeter_l255_25571

theorem square_perimeter (t : ℝ) (h1 : t > 0) : 
  (5 / 2 * t = 40) → (4 * t = 64) := by
  sorry

end square_perimeter_l255_25571


namespace binary_linear_equation_l255_25502

theorem binary_linear_equation (x y : ℝ) : x + y = 5 → x = 3 → y = 2 := by
  sorry

end binary_linear_equation_l255_25502


namespace division_remainder_problem_l255_25555

theorem division_remainder_problem (N : ℕ) 
  (h1 : N / 8 = 8) 
  (h2 : N % 5 = 4) : 
  N % 8 = 6 := by
  sorry

end division_remainder_problem_l255_25555


namespace min_value_x_plus_81_over_x_l255_25595

theorem min_value_x_plus_81_over_x (x : ℝ) (h : x > 0) :
  x + 81 / x ≥ 18 ∧ (x + 81 / x = 18 ↔ x = 9) := by
  sorry

end min_value_x_plus_81_over_x_l255_25595


namespace little_john_initial_money_l255_25553

def sweets_cost : ℝ := 1.05
def friend_gift : ℝ := 1.00
def num_friends : ℕ := 2
def money_left : ℝ := 17.05

theorem little_john_initial_money :
  sweets_cost + friend_gift * num_friends + money_left = 20.10 := by
  sorry

end little_john_initial_money_l255_25553


namespace log_equation_solution_l255_25550

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = 3^(10/3) := by
sorry

end log_equation_solution_l255_25550


namespace intersect_point_m_bisecting_line_equation_l255_25512

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l₂ (x y : ℝ) : Prop := x - y + 2 = 0
def l₃ (m x y : ℝ) : Prop := 3 * x + m * y - 6 = 0

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Theorem for part (1)
theorem intersect_point_m : 
  ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ (∃ (m : ℝ), l₃ m x y) → 
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ l₃ (21/5) x y) :=
sorry

-- Theorem for part (2)
theorem bisecting_line_equation :
  ∃ (A B : ℝ × ℝ) (k : ℝ), 
    l₁ A.1 A.2 ∧ l₂ B.1 B.2 ∧ 
    ((A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) →
    (∀ (x y : ℝ), 11 * x + y - 22 = 0 ↔ 
      ∃ (t : ℝ), x = A.1 + t * (M.1 - A.1) ∧ y = A.2 + t * (M.2 - A.2)) :=
sorry

end intersect_point_m_bisecting_line_equation_l255_25512


namespace present_worth_from_discounts_l255_25554

/-- Present worth of a bill given true discount and banker's discount -/
theorem present_worth_from_discounts (TD BD : ℚ) : 
  TD = 36 → BD = 37.62 → 
  ∃ P : ℚ, P = 800 ∧ BD = (TD * (P + TD)) / P := by
  sorry

#check present_worth_from_discounts

end present_worth_from_discounts_l255_25554


namespace slightly_used_crayons_l255_25560

theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
sorry

end slightly_used_crayons_l255_25560


namespace not_perfect_square_l255_25547

theorem not_perfect_square (n : ℕ+) : 
  (n^2 + n)^2 < n^4 + 2*n^3 + 2*n^2 + 2*n + 1 ∧ 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 < (n^2 + n + 1)^2 :=
by sorry

end not_perfect_square_l255_25547


namespace ice_cream_bill_l255_25566

/-- Calculate the final bill for ice cream sundaes with tip -/
theorem ice_cream_bill (price1 price2 price3 price4 : ℝ) :
  let total_price := price1 + price2 + price3 + price4
  let tip_percentage := 0.20
  let tip := total_price * tip_percentage
  let final_bill := total_price + tip
  final_bill = total_price * (1 + tip_percentage) :=
by sorry

end ice_cream_bill_l255_25566


namespace problem_curve_is_ray_l255_25541

/-- A curve defined by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of a ray -/
def IsRay (c : ParametricCurve) : Prop :=
  ∃ (a b m : ℝ), ∀ t : ℝ, 
    c.x t = m * (c.y t) + b ∧ 
    c.x t ≥ a ∧ 
    c.y t ≥ -1

/-- The specific curve from the problem -/
def problemCurve : ParametricCurve :=
  { x := λ t : ℝ => 3 * t^2 + 2
    y := λ t : ℝ => t^2 - 1 }

/-- Theorem stating that the problem curve is a ray -/
theorem problem_curve_is_ray : IsRay problemCurve := by
  sorry


end problem_curve_is_ray_l255_25541


namespace smallest_s_is_six_l255_25563

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 13

-- Define the property of s being the smallest whole number that forms a valid triangle
def is_smallest_valid_s (s : ℕ) : Prop :=
  (s : ℝ) + a > b ∧ 
  (s : ℝ) + b > a ∧ 
  a + b > (s : ℝ) ∧
  ∀ t : ℕ, t < s → ¬((t : ℝ) + a > b ∧ (t : ℝ) + b > a ∧ a + b > (t : ℝ))

-- Theorem statement
theorem smallest_s_is_six : is_smallest_valid_s 6 :=
sorry

end smallest_s_is_six_l255_25563


namespace hubei_population_scientific_notation_l255_25540

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The population of Hubei Province -/
def hubei_population : ℕ := 57000000

/-- Scientific notation for Hubei population -/
def hubei_scientific : ScientificNotation :=
  { coefficient := 5.7
  , exponent := 7
  , h1 := by sorry }

/-- Theorem stating that the scientific notation correctly represents the population -/
theorem hubei_population_scientific_notation :
  (hubei_scientific.coefficient * (10 : ℝ) ^ hubei_scientific.exponent) = hubei_population := by
  sorry

end hubei_population_scientific_notation_l255_25540


namespace amy_baskets_l255_25590

/-- The number of baskets Amy can fill with candies -/
def num_baskets (chocolate_bars : ℕ) (m_and_ms_ratio : ℕ) (marshmallow_ratio : ℕ) (candies_per_basket : ℕ) : ℕ :=
  let m_and_ms := chocolate_bars * m_and_ms_ratio
  let marshmallows := m_and_ms * marshmallow_ratio
  let total_candies := chocolate_bars + m_and_ms + marshmallows
  total_candies / candies_per_basket

/-- Theorem stating that Amy will fill 25 baskets given the conditions -/
theorem amy_baskets :
  num_baskets 5 7 6 10 = 25 := by
  sorry

end amy_baskets_l255_25590


namespace specific_cards_probability_l255_25516

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (kings_per_suit : Nat)
  (queens_per_suit : Nat)
  (jacks_per_suit : Nat)

/-- Calculates the probability of drawing specific cards from a deck -/
def draw_probability (d : Deck) : Rat :=
  1 / (d.cards * (d.cards - 1) * (d.cards - 2) / (4 * d.queens_per_suit))

theorem specific_cards_probability :
  let standard_deck : Deck := {
    cards := 52,
    suits := 4,
    cards_per_suit := 13,
    kings_per_suit := 1,
    queens_per_suit := 1,
    jacks_per_suit := 1
  }
  draw_probability standard_deck = 1 / 33150 := by
  sorry

end specific_cards_probability_l255_25516


namespace sum_of_digits_1_to_9999_l255_25534

/-- Sum of digits for numbers from 1 to n -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for numbers from 1 to 4999 -/
def sumTo4999 : ℕ := sumOfDigits 4999

/-- Sum of digits for numbers from 5000 to 9999, considering mirroring and additional 5 -/
def sum5000To9999 : ℕ := sumTo4999 + 5000 * 5

/-- The total sum of digits for all numbers from 1 to 9999 -/
def totalSum : ℕ := sumTo4999 + sum5000To9999

theorem sum_of_digits_1_to_9999 : totalSum = 474090 := by sorry

end sum_of_digits_1_to_9999_l255_25534


namespace tom_and_mary_ages_l255_25564

theorem tom_and_mary_ages :
  ∃ (tom_age mary_age : ℕ),
    tom_age^2 + mary_age = 62 ∧
    mary_age^2 + tom_age = 176 ∧
    tom_age = 7 ∧
    mary_age = 13 := by
  sorry

end tom_and_mary_ages_l255_25564


namespace stratified_sample_size_l255_25599

/-- Represents the sample sizes of sedan models A, B, and C in a stratified sample. -/
structure SedanSample where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The production ratio of sedan models A, B, and C. -/
def productionRatio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 4

/-- The total of the production ratio values. -/
def ratioTotal : ℕ := (productionRatio 0) + (productionRatio 1) + (productionRatio 2)

/-- Theorem stating that if the number of model A sedans is 8 fewer than model B sedans
    in a stratified sample with the given production ratio, then the total sample size is 72. -/
theorem stratified_sample_size
  (sample : SedanSample)
  (h1 : sample.a + 8 = sample.b)
  (h2 : (sample.a : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 0 : ℚ) / ratioTotal)
  (h3 : (sample.b : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 1 : ℚ) / ratioTotal)
  (h4 : (sample.c : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 2 : ℚ) / ratioTotal) :
  sample.a + sample.b + sample.c = 72 := by
  sorry


end stratified_sample_size_l255_25599


namespace arithmetic_sequence_common_difference_l255_25556

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference d is 2 when (S_2020 / 2020) - (S_20 / 20) = 2000 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence property
  (h_sum : ∀ n, S n = n * a 0 + n * (n - 1) / 2 * (a 1 - a 0))  -- Sum formula
  (h_condition : S 2020 / 2020 - S 20 / 20 = 2000)  -- Given condition
  : a 1 - a 0 = 2 :=
sorry

end arithmetic_sequence_common_difference_l255_25556


namespace expression_evaluation_l255_25559

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -1
  (x + y)^2 - 3*x*(x + y) + (x + 2*y)*(x - 2*y) = -3 := by
  sorry

end expression_evaluation_l255_25559


namespace log_sum_approximation_l255_25558

open Real

theorem log_sum_approximation : 
  ∃ ε > 0, abs (log 9 / log 10 + 3 * log 2 / log 10 + 2 * log 3 / log 10 + 
               4 * log 5 / log 10 + log 4 / log 10 - 6.21) < ε :=
by
  sorry

end log_sum_approximation_l255_25558


namespace roundness_of_1728000_l255_25544

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 1,728,000 is 19 -/
theorem roundness_of_1728000 : roundness 1728000 = 19 := by sorry

end roundness_of_1728000_l255_25544


namespace equation_solution_l255_25526

theorem equation_solution : ∃! x : ℝ, 3 * (5 - x) = 9 ∧ x = 2 := by sorry

end equation_solution_l255_25526


namespace jerry_payment_l255_25514

/-- Calculates the total amount paid for Jerry's work given the following conditions:
  * Jerry's hourly rate
  * Time spent painting the house
  * Time spent fixing the kitchen counter (3 times the painting time)
  * Time spent mowing the lawn
-/
def total_amount_paid (rate : ℕ) (painting_time : ℕ) (mowing_time : ℕ) : ℕ :=
  rate * (painting_time + 3 * painting_time + mowing_time)

/-- Theorem stating that given the specific conditions of Jerry's work,
    the total amount paid is $570 -/
theorem jerry_payment : total_amount_paid 15 8 6 = 570 := by
  sorry

end jerry_payment_l255_25514


namespace polygon_rotation_theorem_l255_25583

theorem polygon_rotation_theorem (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → Fin n) (h_perm : Function.Bijective a) 
  (h_initial : ∀ i : Fin n, a i ≠ i) :
  ∃ (r : ℕ) (i j : Fin n), i ≠ j ∧ 
    (a i).val - i.val ≡ r [MOD n] ∧
    (a j).val - j.val ≡ r [MOD n] :=
sorry

end polygon_rotation_theorem_l255_25583


namespace marbles_remaining_l255_25536

/-- The number of marbles remaining in a pile after Chris and Ryan combine their marbles and each takes away 1/4 of the total. -/
theorem marbles_remaining (chris_marbles ryan_marbles : ℕ) 
  (h_chris : chris_marbles = 12)
  (h_ryan : ryan_marbles = 28) : 
  (chris_marbles + ryan_marbles) - 2 * ((chris_marbles + ryan_marbles) / 4) = 20 := by
  sorry

end marbles_remaining_l255_25536


namespace debate_club_next_meeting_l255_25535

theorem debate_club_next_meeting (anthony bethany casey dana : ℕ) 
  (h1 : anthony = 5)
  (h2 : bethany = 6)
  (h3 : casey = 8)
  (h4 : dana = 10) :
  Nat.lcm (Nat.lcm (Nat.lcm anthony bethany) casey) dana = 120 := by
  sorry

end debate_club_next_meeting_l255_25535


namespace op_times_oq_equals_10_l255_25507

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 10 = 0

-- Define line l₁
def line_l₁ (x y k : ℝ) : Prop := y = k * x

-- Define line l₂
def line_l₂ (x y : ℝ) : Prop := 3*x + 2*y + 10 = 0

-- Define the intersection points A and B of circle C and line l₁
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_l₁ A.1 A.2 k ∧ line_l₁ B.1 B.2 k ∧
  A ≠ B

-- Define point P as the midpoint of AB
def midpoint_P (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define point Q as the intersection of l₁ and l₂
def intersection_Q (Q : ℝ × ℝ) (k : ℝ) : Prop :=
  line_l₁ Q.1 Q.2 k ∧ line_l₂ Q.1 Q.2

-- State the theorem
theorem op_times_oq_equals_10 (k : ℝ) (A B P Q : ℝ × ℝ) :
  intersection_points A B k →
  midpoint_P P A B →
  intersection_Q Q k →
  ‖(P.1, P.2)‖ * ‖(Q.1, Q.2)‖ = 10 :=
sorry

end op_times_oq_equals_10_l255_25507


namespace geometric_sequence_y_value_l255_25503

/-- Given that 2, x, y, z, 18 form a geometric sequence, prove that y = 6 -/
theorem geometric_sequence_y_value 
  (x y z : ℝ) 
  (h : ∃ (q : ℝ), q ≠ 0 ∧ x = 2 * q ∧ y = 2 * q^2 ∧ z = 2 * q^3 ∧ 18 = 2 * q^4) : 
  y = 6 := by
sorry

end geometric_sequence_y_value_l255_25503


namespace max_product_value_l255_25529

-- Define the functions f and h
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value :
  (∀ x, -3 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -1 ≤ h x ∧ h x ≤ 3) →
  (∃ d, ∀ x, f x * h x ≤ d) ∧
  ∀ d', (∀ x, f x * h x ≤ d') → d' ≥ 12 :=
by sorry

end max_product_value_l255_25529


namespace coupon1_best_in_range_best_price_is_209_95_l255_25527

def coupon1_discount (x : ℝ) : ℝ := 0.12 * x

def coupon2_discount : ℝ := 25

def coupon3_discount (x : ℝ) : ℝ := 0.15 * (x - 120)

theorem coupon1_best_in_range (x : ℝ) 
  (h1 : 208.33 < x) (h2 : x < 600) : 
  coupon1_discount x > coupon2_discount ∧ 
  coupon1_discount x > coupon3_discount x := by
  sorry

def listed_prices : List ℝ := [189.95, 209.95, 229.95, 249.95, 269.95]

theorem best_price_is_209_95 : 
  ∃ p ∈ listed_prices, p > 208.33 ∧ p < 600 ∧ 
  ∀ q ∈ listed_prices, q > 208.33 ∧ q < 600 → p ≤ q := by
  sorry

end coupon1_best_in_range_best_price_is_209_95_l255_25527


namespace coloring_book_shelves_l255_25520

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) :
  initial_stock = 120 →
  books_sold = 39 →
  books_per_shelf = 9 →
  (initial_stock - books_sold) / books_per_shelf = 9 :=
by
  sorry

end coloring_book_shelves_l255_25520


namespace power_multiplication_l255_25539

theorem power_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 := by
  sorry

end power_multiplication_l255_25539


namespace remainder_theorem_polynomial_division_remainder_l255_25506

def P (x : ℝ) : ℝ := 8*x^5 - 10*x^4 + 6*x^3 - 2*x^2 + 3*x - 35

theorem remainder_theorem (P : ℝ → ℝ) (a : ℝ) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - a) * Q x + P a :=
sorry

theorem polynomial_division_remainder :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (2*x - 8) * Q x + 5961 :=
sorry

end remainder_theorem_polynomial_division_remainder_l255_25506


namespace min_value_of_expression_l255_25572

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + x/y ≥ 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1/x + x/y = 3 :=
by sorry

end min_value_of_expression_l255_25572


namespace alice_ice_cream_l255_25515

/-- The number of pints of ice cream Alice had on Wednesday -/
def ice_cream_pints : ℕ → ℕ
  | 0 => 4  -- Sunday
  | 1 => 3 * ice_cream_pints 0  -- Monday
  | 2 => ice_cream_pints 1 / 3  -- Tuesday
  | 3 => ice_cream_pints 0 + ice_cream_pints 1 + ice_cream_pints 2 - ice_cream_pints 2 / 2  -- Wednesday
  | _ => 0  -- Other days (not relevant to the problem)

theorem alice_ice_cream : ice_cream_pints 3 = 18 := by
  sorry

end alice_ice_cream_l255_25515


namespace total_lines_for_given_conditions_l255_25513

/-- Given a number of intersections, crosswalks per intersection, and lines per crosswalk,
    calculate the total number of lines across all crosswalks in all intersections. -/
def total_lines (intersections : ℕ) (crosswalks_per_intersection : ℕ) (lines_per_crosswalk : ℕ) : ℕ :=
  intersections * crosswalks_per_intersection * lines_per_crosswalk

/-- Prove that for 10 intersections, each with 8 crosswalks, and each crosswalk having 30 lines,
    the total number of lines is 2400. -/
theorem total_lines_for_given_conditions :
  total_lines 10 8 30 = 2400 := by
  sorry

end total_lines_for_given_conditions_l255_25513


namespace max_price_reduction_l255_25576

/-- The maximum price reduction for a product while maintaining a minimum profit margin -/
theorem max_price_reduction (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 1000 →
  selling_price = 1500 →
  min_profit_margin = 0.05 →
  ∃ (max_reduction : ℝ),
    max_reduction = 450 ∧
    selling_price - max_reduction = cost_price * (1 + min_profit_margin) :=
by sorry

end max_price_reduction_l255_25576


namespace bd_squared_equals_four_l255_25585

theorem bd_squared_equals_four (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 9) : 
  (b - d)^2 = 4 := by sorry

end bd_squared_equals_four_l255_25585


namespace polynomial_divisibility_sum_A_B_l255_25578

-- Define the polynomial
def p (A B : ℂ) (x : ℂ) : ℂ := x^103 + A*x + B

-- Define the divisor polynomial
def d (x : ℂ) : ℂ := x^2 + x + 1

-- State the theorem
theorem polynomial_divisibility (A B : ℂ) :
  (∀ x, d x = 0 → p A B x = 0) →
  A = -1 ∧ B = 0 := by
  sorry

-- Corollary for A + B
theorem sum_A_B (A B : ℂ) :
  (∀ x, d x = 0 → p A B x = 0) →
  A + B = -1 := by
  sorry

end polynomial_divisibility_sum_A_B_l255_25578


namespace sum_of_squares_of_exponents_992_l255_25542

-- Define a function to express a number as a sum of distinct powers of 2
def expressAsPowersOfTwo (n : ℕ) : List ℕ := sorry

-- Define a function to calculate the sum of squares of a list of numbers
def sumOfSquares (l : List ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_squares_of_exponents_992 :
  sumOfSquares (expressAsPowersOfTwo 992) = 255 := by sorry

end sum_of_squares_of_exponents_992_l255_25542


namespace prime_equation_solution_l255_25510

theorem prime_equation_solution (p : ℕ) (hp : Prime p) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 :=
by sorry

end prime_equation_solution_l255_25510


namespace library_books_theorem_l255_25565

variable (Library : Type)
variable (is_new_edition : Library → Prop)

theorem library_books_theorem :
  (¬ ∀ (book : Library), is_new_edition book) →
  (∃ (book : Library), ¬ is_new_edition book) ∧
  (¬ ∀ (book : Library), is_new_edition book) :=
by sorry

end library_books_theorem_l255_25565


namespace complex_expression_equality_l255_25517

theorem complex_expression_equality (c d : ℂ) (h1 : c = 3 - 2*I) (h2 : d = 2 + 3*I) :
  3*c + 4*d + 2 = 19 + 6*I :=
by sorry

end complex_expression_equality_l255_25517


namespace johns_money_l255_25580

/-- Given that John needs a total amount of money and still needs some more,
    prove that the amount he already has is the difference between the total needed and the amount still needed. -/
theorem johns_money (total_needed : ℚ) (still_needed : ℚ) (already_has : ℚ) :
  total_needed = 2.5 →
  still_needed = 1.75 →
  already_has = total_needed - still_needed →
  already_has = 0.75 := by
  sorry

end johns_money_l255_25580


namespace triangle_properties_l255_25557

theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (2 * a * (Real.sin (2 * B) - Real.sin A * Real.cos C) = c * Real.sin (2 * A)) →
  (3 : Real) = 3 →
  (Real.sin (π / 3 : Real) = Real.sin (Real.pi / 3)) →
  ((1 / 2 : Real) * a * c * Real.sin B = 3 * Real.sqrt 3) →
  (B = π / 3) ∧
  (a + b + c = 2 * Real.sqrt 13 + 4) :=
by sorry

end triangle_properties_l255_25557


namespace max_value_a_l255_25552

open Real

theorem max_value_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a ≤ (1-x)/x + log x) → 
  a ≤ 0 := by sorry

end max_value_a_l255_25552


namespace power_function_through_point_l255_25551

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x^α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 :=
by sorry

end power_function_through_point_l255_25551


namespace distance_to_focus_l255_25598

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (y : ℝ) : 
  y^2 = 8 * 2 →  -- Point M(2, y) is on the parabola y^2 = 8x
  4 = (2 - (-2)) -- Distance from M to the directrix (x = -2)
    + (2 - 0)    -- Distance from M to the x-coordinate of the focus (which is at x = 0 for this parabola)
  := by sorry

end distance_to_focus_l255_25598


namespace complex_number_problem_l255_25549

theorem complex_number_problem (α β : ℂ) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ α + β = x ∧ Complex.I * (2 * α - β) = y) →
  β = 4 + 3 * Complex.I →
  α = 2 - 3 * Complex.I :=
by sorry

end complex_number_problem_l255_25549


namespace ellipse_dot_product_min_l255_25561

/-- An ellipse with center at origin and left focus at (-1, 0) -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2 / 4 + y^2 / 3 = 1

/-- The dot product of OP and FP is always greater than or equal to 2 -/
theorem ellipse_dot_product_min (P : Ellipse) : 
  P.x * (P.x + 1) + P.y * P.y ≥ 2 := by
  sorry

#check ellipse_dot_product_min

end ellipse_dot_product_min_l255_25561


namespace original_number_exists_and_unique_l255_25573

theorem original_number_exists_and_unique :
  ∃! x : ℕ, 
    Odd (3 * x) ∧ 
    (∃ k : ℕ, 3 * x = 9 * k) ∧ 
    4 * x = 108 := by
  sorry

end original_number_exists_and_unique_l255_25573


namespace max_value_product_l255_25505

theorem max_value_product (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (hsum : a + b + c = 3) : 
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 729/432 ∧ 
  ∃ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 3 ∧ 
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 729/432 :=
by sorry

end max_value_product_l255_25505


namespace work_completion_time_l255_25500

/-- Given workers A, B, and C who can complete a work individually in 4, 8, and 8 days respectively,
    prove that they can complete the work together in 2 days. -/
theorem work_completion_time (work : ℝ) (days_A days_B days_C : ℝ) 
    (h_work : work > 0)
    (h_A : days_A = 4)
    (h_B : days_B = 8)
    (h_C : days_C = 8) :
    work / (work / days_A + work / days_B + work / days_C) = 2 := by
  sorry


end work_completion_time_l255_25500


namespace debby_messages_before_noon_l255_25533

/-- The number of text messages Debby received before noon -/
def messages_before_noon : ℕ := sorry

/-- The number of text messages Debby received after noon -/
def messages_after_noon : ℕ := 18

/-- The total number of text messages Debby received -/
def total_messages : ℕ := 39

/-- Theorem stating that Debby received 21 text messages before noon -/
theorem debby_messages_before_noon :
  messages_before_noon = 21 :=
by
  sorry

end debby_messages_before_noon_l255_25533


namespace harrys_pizza_toppings_l255_25570

/-- Calculates the number of toppings per pizza given the conditions of Harry's pizza order --/
theorem harrys_pizza_toppings : ∀ (toppings_per_pizza : ℕ),
  (14 : ℚ) * 2 + -- Cost of two large pizzas
  (2 : ℚ) * (2 * toppings_per_pizza) + -- Cost of toppings
  (((14 : ℚ) * 2 + (2 : ℚ) * (2 * toppings_per_pizza)) * (1 / 4)) -- 25% tip
  = 50 →
  toppings_per_pizza = 3 := by
  sorry

#check harrys_pizza_toppings

end harrys_pizza_toppings_l255_25570


namespace sum_of_reciprocals_equals_five_l255_25538

theorem sum_of_reciprocals_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end sum_of_reciprocals_equals_five_l255_25538


namespace jacksons_grade_l255_25543

/-- Calculates Jackson's grade based on his study time and point increase rate. -/
def calculate_grade (gaming_hours : ℝ) (study_ratio : ℝ) (points_per_hour : ℝ) : ℝ :=
  gaming_hours * study_ratio * points_per_hour

/-- Theorem stating that Jackson's grade is 45 points given the problem conditions. -/
theorem jacksons_grade :
  let gaming_hours : ℝ := 9
  let study_ratio : ℝ := 1/3
  let points_per_hour : ℝ := 15
  calculate_grade gaming_hours study_ratio points_per_hour = 45 := by
  sorry


end jacksons_grade_l255_25543


namespace joan_bought_six_dozens_l255_25545

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := 72

/-- The number of dozens of eggs Joan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem joan_bought_six_dozens : dozens_bought = 6 := by
  sorry

end joan_bought_six_dozens_l255_25545


namespace promotion_savings_l255_25596

/-- Represents a promotion offered by the department store -/
structure Promotion where
  name : String
  first_pair_price : ℝ
  second_pair_price : ℝ
  additional_discount : ℝ

/-- Calculates the total cost for a given promotion -/
def total_cost (p : Promotion) (handbag_price : ℝ) : ℝ :=
  p.first_pair_price + p.second_pair_price + handbag_price - p.additional_discount

/-- The main theorem stating that Promotion A saves $19.5 more than Promotion B -/
theorem promotion_savings :
  let shoe_price : ℝ := 50
  let handbag_price : ℝ := 20
  let promotion_a : Promotion := {
    name := "A",
    first_pair_price := shoe_price,
    second_pair_price := shoe_price / 2,
    additional_discount := (shoe_price + shoe_price / 2 + handbag_price) * 0.1
  }
  let promotion_b : Promotion := {
    name := "B",
    first_pair_price := shoe_price,
    second_pair_price := shoe_price - 15,
    additional_discount := 0
  }
  total_cost promotion_b handbag_price - total_cost promotion_a handbag_price = 19.5 := by
  sorry


end promotion_savings_l255_25596


namespace baseball_card_pages_l255_25523

theorem baseball_card_pages (cards_per_page : ℕ) (new_cards : ℕ) (old_cards : ℕ) :
  cards_per_page = 3 →
  new_cards = 2 →
  old_cards = 10 →
  (new_cards + old_cards) / cards_per_page = 4 :=
by sorry

end baseball_card_pages_l255_25523


namespace negative_x_squared_times_x_cubed_l255_25575

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x)^2 * x^3 = x^5 := by
  sorry

end negative_x_squared_times_x_cubed_l255_25575


namespace expression_value_l255_25574

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 4)  -- The absolute value of m is 4
  : m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end expression_value_l255_25574


namespace sandy_earnings_l255_25522

/-- Calculates the total earnings for Sandy given her hourly rate and hours worked each day -/
def total_earnings (hourly_rate : ℕ) (hours_friday : ℕ) (hours_saturday : ℕ) (hours_sunday : ℕ) : ℕ :=
  hourly_rate * (hours_friday + hours_saturday + hours_sunday)

/-- Theorem stating that Sandy's total earnings for the three days is $450 -/
theorem sandy_earnings : 
  total_earnings 15 10 6 14 = 450 := by
  sorry

end sandy_earnings_l255_25522


namespace parallel_angles_theorem_l255_25567

/-- Two angles in space with parallel corresponding sides -/
structure ParallelAngles where
  a : Real
  b : Real
  parallel : Bool

/-- Theorem: If two angles with parallel corresponding sides have one angle of 60°, 
    then the other angle is either 60° or 120° -/
theorem parallel_angles_theorem (angles : ParallelAngles) 
  (h1 : angles.parallel = true) 
  (h2 : angles.a = 60) : 
  angles.b = 60 ∨ angles.b = 120 := by
  sorry

end parallel_angles_theorem_l255_25567


namespace intersecting_circles_values_l255_25594

/-- Two circles intersecting at points A and B, with centers on a line -/
structure IntersectingCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  c : ℝ
  centers_on_line : ∀ (center : ℝ × ℝ), center.1 + center.2 + c = 0

/-- The theorem stating the values of m and c for the given configuration -/
theorem intersecting_circles_values (circles : IntersectingCircles) 
  (h1 : circles.A = (-1, 3))
  (h2 : circles.B.1 = -6) : 
  circles.B.2 = 3 ∧ circles.c = -2 := by
  sorry

end intersecting_circles_values_l255_25594


namespace population_net_increase_l255_25584

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 7

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 2

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Theorem stating the net increase in population size in one day -/
theorem population_net_increase : 
  (birth_rate - death_rate) / 2 * seconds_per_day = 216000 := by sorry

end population_net_increase_l255_25584


namespace prism_with_ten_diagonals_has_five_sides_l255_25562

/-- A right prism with n sides and d diagonals. -/
structure RightPrism where
  n : ℕ
  d : ℕ

/-- The number of diagonals in a right n-sided prism is 2n. -/
axiom diagonals_count (p : RightPrism) : p.d = 2 * p.n

/-- For a right prism with 10 diagonals, the number of sides is 5. -/
theorem prism_with_ten_diagonals_has_five_sides (p : RightPrism) (h : p.d = 10) : p.n = 5 := by
  sorry

end prism_with_ten_diagonals_has_five_sides_l255_25562


namespace jonessas_take_home_pay_l255_25592

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jonessa's take-home pay is $450 -/
theorem jonessas_take_home_pay :
  let totalPay : ℝ := 500
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 450 := by sorry

end jonessas_take_home_pay_l255_25592


namespace books_for_girls_l255_25519

theorem books_for_girls (num_girls num_boys total_books : ℕ) : 
  num_girls = 15 → 
  num_boys = 10 → 
  total_books = 375 → 
  (num_girls * (total_books / (num_girls + num_boys))) = 225 := by
  sorry

end books_for_girls_l255_25519


namespace square_of_linear_expression_l255_25568

theorem square_of_linear_expression (n : ℚ) :
  (∃ a b : ℚ, ∀ x : ℚ, (7 * x^2 + 21 * x + 5 * n) / 7 = (a * x + b)^2) →
  n = 63/20 := by
sorry

end square_of_linear_expression_l255_25568


namespace remainder_3_pow_2023_mod_5_l255_25582

theorem remainder_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l255_25582


namespace nancy_files_problem_l255_25509

theorem nancy_files_problem (deleted_files : ℕ) (files_per_folder : ℕ) (final_folders : ℕ) :
  deleted_files = 31 →
  files_per_folder = 6 →
  final_folders = 2 →
  deleted_files + (files_per_folder * final_folders) = 43 :=
by sorry

end nancy_files_problem_l255_25509


namespace man_son_age_ratio_l255_25532

/-- 
Given a man who is 20 years older than his son, and the son's present age is 18,
prove that the ratio of the man's age to his son's age in two years will be 2:1.
-/
theorem man_son_age_ratio : 
  ∀ (son_age man_age : ℕ),
  son_age = 18 →
  man_age = son_age + 20 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end man_son_age_ratio_l255_25532


namespace interest_equality_theorem_l255_25524

theorem interest_equality_theorem (total : ℝ) (x : ℝ) : 
  total = 2665 →
  (x * 3 * 8) / 100 = ((total - x) * 5 * 3) / 100 →
  total - x = 1640 := by
  sorry

end interest_equality_theorem_l255_25524


namespace floor_sqrt_50_squared_l255_25569

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end floor_sqrt_50_squared_l255_25569


namespace parallelogram_count_in_triangle_grid_l255_25537

/-- Given an equilateral triangle with sides divided into n parts, 
    calculates the number of parallelograms formed by parallel lines --/
def parallelogramCount (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- Theorem stating the number of parallelograms in the grid --/
theorem parallelogram_count_in_triangle_grid (n : ℕ) :
  parallelogramCount n = 3 * Nat.choose (n + 2) 4 := by
  sorry

end parallelogram_count_in_triangle_grid_l255_25537


namespace race_time_theorem_l255_25593

/-- The time taken by this year's winner to complete the race around the town square. -/
def this_year_time (laps : ℕ) (square_length : ℚ) (last_year_time : ℚ) (time_improvement : ℚ) : ℚ :=
  let total_distance := laps * square_length
  let last_year_pace := last_year_time / total_distance
  let this_year_pace := last_year_pace - time_improvement
  this_year_pace * total_distance

/-- Theorem stating that this year's winner completed the race in 42 minutes. -/
theorem race_time_theorem :
  this_year_time 7 (3/4) 47.25 1 = 42 := by
  sorry

end race_time_theorem_l255_25593


namespace complement_and_union_when_m_3_subset_condition_disjoint_condition_l255_25579

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ m + 2}

-- Theorem 1
theorem complement_and_union_when_m_3 :
  (Set.univ \ B 3) = {x : ℝ | x < 3 ∨ x > 5} ∧
  A ∪ B 3 = {x : ℝ | 0 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem subset_condition :
  ∀ m : ℝ, B m ⊆ A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

-- Theorem 3
theorem disjoint_condition :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m < -2 ∨ m > 4 := by sorry

end complement_and_union_when_m_3_subset_condition_disjoint_condition_l255_25579


namespace correlation_strength_increases_l255_25586

-- Define the correlation coefficient as a real number between -1 and 1
def correlation_coefficient : Type := {r : ℝ // -1 ≤ r ∧ r ≤ 1}

-- Define a measure of linear correlation strength
def linear_correlation_strength (r : correlation_coefficient) : ℝ := |r.val|

-- Define a notion of "closer to 1"
def closer_to_one (r1 r2 : correlation_coefficient) : Prop :=
  |r1.val - 1| < |r2.val - 1|

-- Statement: As |r| approaches 1, the linear correlation becomes stronger
theorem correlation_strength_increases (r1 r2 : correlation_coefficient) :
  closer_to_one r1 r2 → linear_correlation_strength r1 > linear_correlation_strength r2 :=
sorry

end correlation_strength_increases_l255_25586


namespace hiking_problem_l255_25591

/-- A hiking problem with two trails -/
theorem hiking_problem (trail1_length trail1_speed trail2_speed : ℝ)
  (break_time time_difference : ℝ) :
  trail1_length = 20 ∧
  trail1_speed = 5 ∧
  trail2_speed = 3 ∧
  break_time = 1 ∧
  time_difference = 1 ∧
  (trail1_length / trail1_speed = 
    (trail1_length / trail1_speed + time_difference)) →
  ∃ trail2_length : ℝ,
    trail2_length / trail2_speed / 2 + break_time + 
    trail2_length / trail2_speed / 2 = 
    trail1_length / trail1_speed + time_difference ∧
    trail2_length = 12 :=
by sorry


end hiking_problem_l255_25591


namespace quadratic_inequality_solution_l255_25501

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 := by
  sorry

end quadratic_inequality_solution_l255_25501


namespace not_sum_of_three_squares_2015_l255_25530

theorem not_sum_of_three_squares_2015 : ¬∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2015 := by
  sorry

end not_sum_of_three_squares_2015_l255_25530


namespace min_cost_at_optimal_distance_l255_25597

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 5)^2 + 1000 / (x + 5)

theorem min_cost_at_optimal_distance :
  ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 8 ∧
  (∀ y : ℝ, 2 ≤ y ∧ y ≤ 8 → f y ≥ f x) ∧
  x = 5 ∧ f x = 150 := by
sorry

end min_cost_at_optimal_distance_l255_25597


namespace third_term_is_seven_l255_25511

/-- An arithmetic sequence with general term aₙ = 2n + 1 -/
def a (n : ℕ) : ℝ := 2 * n + 1

/-- The third term of the sequence is 7 -/
theorem third_term_is_seven : a 3 = 7 := by
  sorry

end third_term_is_seven_l255_25511


namespace x_value_at_stop_l255_25528

/-- Represents the state of the computation at each step -/
structure State where
  x : ℕ
  s : ℕ

/-- Computes the next state given the current state -/
def nextState (state : State) : State :=
  { x := state.x + 3,
    s := state.s + state.x + 3 }

/-- Checks if the stopping condition is met -/
def isStoppingState (state : State) : Prop :=
  state.s ≥ 15000

/-- Represents the sequence of states -/
def stateSequence : ℕ → State
  | 0 => { x := 5, s := 0 }
  | n + 1 => nextState (stateSequence n)

theorem x_value_at_stop :
  ∃ n : ℕ, isStoppingState (stateSequence n) ∧
    ¬isStoppingState (stateSequence (n - 1)) ∧
    (stateSequence n).x = 368 :=
  sorry

end x_value_at_stop_l255_25528


namespace cards_per_box_l255_25587

theorem cards_per_box (total_cards : ℕ) (unboxed_cards : ℕ) (boxes_given : ℕ) (boxes_left : ℕ) :
  total_cards = 75 →
  unboxed_cards = 5 →
  boxes_given = 2 →
  boxes_left = 5 →
  (total_cards - unboxed_cards) % (boxes_given + boxes_left) = 0 →
  (total_cards - unboxed_cards) / (boxes_given + boxes_left) = 10 := by
  sorry

end cards_per_box_l255_25587


namespace sufficient_but_not_necessary_l255_25581

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x^2 > 2012 → x^2 > 2011) ∧
  (∃ x : ℝ, x^2 > 2011 ∧ x^2 ≤ 2012) →
  (∀ x : ℝ, x^2 > 2012 → x^2 > 2011) ∧
  ¬(∀ x : ℝ, x^2 > 2011 → x^2 > 2012) :=
by sorry

end sufficient_but_not_necessary_l255_25581


namespace sequence_properties_l255_25525

/-- Given a sequence {a_n} where n ∈ ℕ* and S_n = n^2 + n, prove:
    1) a_n = 2n for all n ∈ ℕ*
    2) The sum of the first n terms of {1/(n+1)a_n} equals n/(2n+2) -/
theorem sequence_properties (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, S n = (n : ℚ)^2 + n) :
  (∀ n : ℕ+, a n = 2 * n) ∧ 
  (∀ n : ℕ+, (Finset.range n.val).sum (λ i => 1 / ((i + 2 : ℚ) * a (⟨i + 1, Nat.succ_pos i⟩))) = n / (2 * n + 2)) :=
by sorry

end sequence_properties_l255_25525


namespace sequence_sum_bounded_l255_25521

theorem sequence_sum_bounded (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_a1 : 0 ≤ a 1)
  (h_a : ∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1) ∧ a (i + 1) ≤ 2 * a i) :
  ∃ ε : ℕ → ℝ, (∀ i ∈ Finset.range n, ε i = 1 ∨ ε i = -1) ∧ 
    0 ≤ (Finset.range n).sum (λ i => ε i * a (i + 1)) ∧
    (Finset.range n).sum (λ i => ε i * a (i + 1)) ≤ a 1 := by
  sorry

end sequence_sum_bounded_l255_25521


namespace sum_of_roots_eq_seven_halves_l255_25546

theorem sum_of_roots_eq_seven_halves :
  let f : ℝ → ℝ := λ x => (2*x + 3)*(x - 5) - 27
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 7/2 := by
  sorry

end sum_of_roots_eq_seven_halves_l255_25546


namespace y_squared_plus_reciprocal_l255_25508

theorem y_squared_plus_reciprocal (x : ℝ) (a : ℕ) (h1 : x + 1/x = 3) (h2 : a ≠ 1) (h3 : a > 0) :
  let y := x^a
  y^2 + 1/y^2 = (x^2 + 1/x^2)^a - 2*a := by
sorry

end y_squared_plus_reciprocal_l255_25508


namespace floor_sqrt_101_l255_25548

theorem floor_sqrt_101 : ⌊Real.sqrt 101⌋ = 10 := by
  sorry

end floor_sqrt_101_l255_25548


namespace smallest_integer_x_zero_satisfies_inequality_zero_is_smallest_l255_25588

theorem smallest_integer_x (x : ℤ) : (3 - 2 * x^2 < 21) → x ≥ 0 :=
by sorry

theorem zero_satisfies_inequality : 3 - 2 * 0^2 < 21 :=
by sorry

theorem zero_is_smallest (x : ℤ) :
  x < 0 → ¬(3 - 2 * x^2 < 21) :=
by sorry

end smallest_integer_x_zero_satisfies_inequality_zero_is_smallest_l255_25588


namespace micah_fish_count_l255_25589

/-- Proves that Micah has 7 fish given the problem conditions -/
theorem micah_fish_count :
  ∀ (m k t : ℕ),
  k = 3 * m →                -- Kenneth has three times as many fish as Micah
  t = k - 15 →                -- Matthias has 15 less fish than Kenneth
  m + k + t = 34 →            -- The total number of fish for all three boys is 34
  m = 7 :=                    -- Micah has 7 fish
by
  sorry

end micah_fish_count_l255_25589


namespace difference_of_squares_401_399_l255_25577

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 := by
  sorry

end difference_of_squares_401_399_l255_25577


namespace lisa_challenge_time_l255_25518

/-- The time remaining for Lisa to complete the hotdog-eating challenge -/
def timeRemaining (totalHotdogs : ℕ) (hotdogsEaten : ℕ) (eatingRate : ℕ) : ℚ :=
  (totalHotdogs - hotdogsEaten : ℚ) / eatingRate

/-- Theorem stating that Lisa has 5 minutes to complete the challenge -/
theorem lisa_challenge_time : 
  timeRemaining 75 20 11 = 5 := by sorry

end lisa_challenge_time_l255_25518


namespace z_in_second_quadrant_l255_25531

def i : ℂ := Complex.I

def z : ℂ := i * (1 + i)

theorem z_in_second_quadrant :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
by
  sorry

end z_in_second_quadrant_l255_25531
