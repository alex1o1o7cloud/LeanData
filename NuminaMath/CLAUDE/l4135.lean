import Mathlib

namespace NUMINAMATH_CALUDE_min_value_sum_product_l4135_413542

theorem min_value_sum_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 20) :
  a + 2 * b ≥ 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l4135_413542


namespace NUMINAMATH_CALUDE_exists_k_composite_for_all_n_l4135_413550

theorem exists_k_composite_for_all_n : ∃ k : ℕ, ∀ n : ℕ, ∃ m : ℕ, m > 1 ∧ m ∣ (k * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_k_composite_for_all_n_l4135_413550


namespace NUMINAMATH_CALUDE_fraction_product_equals_fifteen_thirty_seconds_l4135_413537

theorem fraction_product_equals_fifteen_thirty_seconds :
  (3 + 5 + 7) / (2 + 4 + 6) * (1 + 3 + 5) / (6 + 8 + 10) = 15 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_fifteen_thirty_seconds_l4135_413537


namespace NUMINAMATH_CALUDE_matrix_product_equality_l4135_413509

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -4; 6, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; -2, 1]

theorem matrix_product_equality :
  A * B = !![8, 5; -4, 20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l4135_413509


namespace NUMINAMATH_CALUDE_project_hours_difference_l4135_413592

theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 216 ∧
  pat_hours = 2 * kate_hours ∧
  pat_hours * 3 = mark_hours ∧
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours - kate_hours = 120 :=
by sorry

end NUMINAMATH_CALUDE_project_hours_difference_l4135_413592


namespace NUMINAMATH_CALUDE_adult_ticket_price_l4135_413568

/-- Represents the price of tickets and sales data for a theater --/
structure TheaterSales where
  adult_price : ℚ
  child_price : ℚ
  total_revenue : ℚ
  total_tickets : ℕ
  adult_tickets : ℕ

/-- Theorem stating that the adult ticket price is $10.50 given the conditions --/
theorem adult_ticket_price (sale : TheaterSales)
  (h1 : sale.child_price = 5)
  (h2 : sale.total_revenue = 236)
  (h3 : sale.total_tickets = 34)
  (h4 : sale.adult_tickets = 12)
  : sale.adult_price = 21/2 := by
  sorry

#eval (21 : ℚ) / 2  -- To verify that 21/2 is indeed 10.50

end NUMINAMATH_CALUDE_adult_ticket_price_l4135_413568


namespace NUMINAMATH_CALUDE_stock_trade_profit_l4135_413566

/-- Represents the stock trading scenario --/
structure StockTrade where
  initial_price : ℝ
  price_changes : List ℝ
  num_shares : ℕ
  buying_fee : ℝ
  selling_fee : ℝ
  transaction_tax : ℝ

/-- Calculates the final price of the stock --/
def final_price (trade : StockTrade) : ℝ :=
  trade.initial_price + trade.price_changes.sum

/-- Calculates the profit from the stock trade --/
def calculate_profit (trade : StockTrade) : ℝ :=
  let cost := trade.initial_price * trade.num_shares * (1 + trade.buying_fee)
  let revenue := (final_price trade) * trade.num_shares * (1 - trade.selling_fee - trade.transaction_tax)
  revenue - cost

/-- Theorem stating that the profit from the given stock trade is 889.5 yuan --/
theorem stock_trade_profit (trade : StockTrade) 
  (h1 : trade.initial_price = 27)
  (h2 : trade.price_changes = [4, 4.5, -1, -2.5, -6, 2])
  (h3 : trade.num_shares = 1000)
  (h4 : trade.buying_fee = 0.0015)
  (h5 : trade.selling_fee = 0.0015)
  (h6 : trade.transaction_tax = 0.001) :
  calculate_profit trade = 889.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_trade_profit_l4135_413566


namespace NUMINAMATH_CALUDE_triangle_properties_l4135_413571

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side length b
  c : ℝ  -- Side length c
  h1 : A = π / 3  -- A = 60° in radians
  h2 : b = 5
  h3 : c = 4

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) :
  ∃ (a : ℝ), 
    a ^ 2 = 21 ∧ 
    Real.sin (Real.arcsin (t.b / a)) * Real.sin (Real.arcsin (t.c / a)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4135_413571


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4135_413502

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4135_413502


namespace NUMINAMATH_CALUDE_four_common_tangents_l4135_413536

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Define the number of common tangent lines
def num_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem four_common_tangents :
  num_common_tangents circle_C1 circle_C2 = 4 := by sorry

end NUMINAMATH_CALUDE_four_common_tangents_l4135_413536


namespace NUMINAMATH_CALUDE_spending_percentage_l4135_413541

/-- Represents Roger's entertainment budget and spending --/
structure Entertainment where
  budget : ℝ
  movie_cost : ℝ
  soda_cost : ℝ
  popcorn_cost : ℝ
  tax_rate : ℝ

/-- Calculates the total spending including tax --/
def total_spending (e : Entertainment) : ℝ :=
  (e.movie_cost + e.soda_cost + e.popcorn_cost) * (1 + e.tax_rate)

/-- Theorem stating that the total spending is approximately 28% of the budget --/
theorem spending_percentage (e : Entertainment) 
  (h1 : e.movie_cost = 0.25 * (e.budget - e.soda_cost))
  (h2 : e.soda_cost = 0.10 * (e.budget - e.movie_cost))
  (h3 : e.popcorn_cost = 5)
  (h4 : e.tax_rate = 0.10)
  (h5 : e.budget > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_spending e / e.budget - 0.28| < ε :=
sorry

end NUMINAMATH_CALUDE_spending_percentage_l4135_413541


namespace NUMINAMATH_CALUDE_seating_arrangements_special_guest_seating_l4135_413532

theorem seating_arrangements (n : Nat) (k : Nat) (h : n > k) :
  (n : Nat) * (n - 1).factorial = n * (n - 1 : Nat).factorial :=
by sorry

theorem special_guest_seating :
  8 * 7 * 6 * 5 * 4 * 3 * 2 = 20160 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_special_guest_seating_l4135_413532


namespace NUMINAMATH_CALUDE_triangle_problem_l4135_413575

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b = 2) :
  t.A = Real.pi / 3 ∧ t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l4135_413575


namespace NUMINAMATH_CALUDE_paint_cost_per_liter_l4135_413510

/-- Calculates the cost of paint per liter given the costs of materials and profit --/
theorem paint_cost_per_liter 
  (brush_cost : ℚ) 
  (canvas_cost_multiplier : ℚ) 
  (min_paint_liters : ℚ) 
  (selling_price : ℚ) 
  (profit : ℚ) 
  (h1 : brush_cost = 20)
  (h2 : canvas_cost_multiplier = 3)
  (h3 : min_paint_liters = 5)
  (h4 : selling_price = 200)
  (h5 : profit = 80) :
  (selling_price - profit - (brush_cost + canvas_cost_multiplier * brush_cost)) / min_paint_liters = 8 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_liter_l4135_413510


namespace NUMINAMATH_CALUDE_problem_2015_l4135_413518

theorem problem_2015 : (2015^2 + 2015 - 1) / 2015 = 2016 - 1/2015 := by
  sorry

end NUMINAMATH_CALUDE_problem_2015_l4135_413518


namespace NUMINAMATH_CALUDE_cellphone_cost_correct_l4135_413574

/-- The cost of a single cellphone before discount -/
def cellphone_cost : ℝ := 800

/-- The number of cellphones purchased -/
def num_cellphones : ℕ := 2

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.05

/-- The final price paid after the discount -/
def final_price : ℝ := 1520

/-- Theorem stating that the given cellphone cost satisfies the conditions -/
theorem cellphone_cost_correct : 
  (num_cellphones : ℝ) * cellphone_cost * (1 - discount_rate) = final_price := by
  sorry

end NUMINAMATH_CALUDE_cellphone_cost_correct_l4135_413574


namespace NUMINAMATH_CALUDE_linear_inequality_equivalence_l4135_413514

theorem linear_inequality_equivalence :
  ∀ x : ℝ, (2 * x - 4 > 0) ↔ (x > 2) := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_equivalence_l4135_413514


namespace NUMINAMATH_CALUDE_inequality_solution_l4135_413505

theorem inequality_solution (a x : ℝ) :
  (a < 0 ∨ a > 1 → (((x - a) / (x - a^2) < 0) ↔ (a < x ∧ x < a^2))) ∧
  (0 < a ∧ a < 1 → (((x - a) / (x - a^2) < 0) ↔ (a^2 < x ∧ x < a))) ∧
  (a = 0 ∨ a = 1 → ¬∃x, (x - a) / (x - a^2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4135_413505


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repetition_l4135_413544

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of digits in the numbers we're forming -/
def num_places : ℕ := 3

/-- The number of non-zero digits available for the first place -/
def non_zero_digits : ℕ := num_digits - 1

/-- The total number of three-digit numbers (including those without repetition) -/
def total_numbers : ℕ := non_zero_digits * num_digits * num_digits

/-- The number of three-digit numbers without repetition -/
def numbers_without_repetition : ℕ := non_zero_digits * (num_digits - 1) * (num_digits - 2)

theorem three_digit_numbers_with_repetition :
  total_numbers - numbers_without_repetition = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repetition_l4135_413544


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_3_and_7_l4135_413587

/-- Represents a three-digit number in the form A3B -/
def ThreeDigitNumber (a b : Nat) : Nat :=
  100 * a + 30 + b

theorem unique_three_digit_number_divisible_by_3_and_7 :
  ∀ a b : Nat,
    (300 < ThreeDigitNumber a b) →
    (ThreeDigitNumber a b < 400) →
    (ThreeDigitNumber a b % 3 = 0) →
    (ThreeDigitNumber a b % 7 = 0) →
    b = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_3_and_7_l4135_413587


namespace NUMINAMATH_CALUDE_distribution_ratio_l4135_413503

/-- Represents the distribution of money among four people --/
structure Distribution where
  p : ℚ  -- Amount received by P
  q : ℚ  -- Amount received by Q
  r : ℚ  -- Amount received by R
  s : ℚ  -- Amount received by S

/-- Theorem stating the ratio of P's amount to Q's amount --/
theorem distribution_ratio (d : Distribution) : 
  d.p + d.q + d.r + d.s = 1000 →  -- Total amount condition
  d.s = 4 * d.r →                 -- S gets 4 times R's amount
  d.q = d.r →                     -- Q and R receive equal amounts
  d.s - d.p = 250 →               -- Difference between S and P
  d.p / d.q = 2 / 1 := by          -- Ratio of P's amount to Q's amount
sorry


end NUMINAMATH_CALUDE_distribution_ratio_l4135_413503


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l4135_413599

/-- Represents a triangle with angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

/-- An isosceles, obtuse triangle with one angle 50% larger than a right angle -/
def special_triangle : Triangle :=
  { angle1 := 135
    angle2 := 22.5
    angle3 := 22.5
    sum_180 := by sorry
    all_positive := by sorry }

theorem smallest_angle_measure :
  ∃ (t : Triangle), 
    (t.angle1 = 90 * 1.5) ∧  -- One angle is 50% larger than right angle
    (t.angle2 = t.angle3) ∧  -- Isosceles property
    (t.angle1 > 90) ∧        -- Obtuse triangle
    (t.angle2 = 22.5 ∧ t.angle3 = 22.5) -- The two smallest angles
    := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l4135_413599


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l4135_413525

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l4135_413525


namespace NUMINAMATH_CALUDE_lottery_tax_percentage_l4135_413594

/-- Proves that the percentage of lottery winnings paid for tax is 20% given the specified conditions --/
theorem lottery_tax_percentage (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ) : 
  winnings = 50 → processing_fee = 5 → take_home = 35 → 
  (winnings - (take_home + processing_fee)) / winnings * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_lottery_tax_percentage_l4135_413594


namespace NUMINAMATH_CALUDE_no_double_application_function_exists_l4135_413582

theorem no_double_application_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1987 := by
sorry

end NUMINAMATH_CALUDE_no_double_application_function_exists_l4135_413582


namespace NUMINAMATH_CALUDE_binomial_sum_equals_240_l4135_413578

theorem binomial_sum_equals_240 : Nat.choose 10 3 + Nat.choose 10 7 = 240 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_240_l4135_413578


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l4135_413545

theorem min_value_trig_expression (x : ℝ) : 
  Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.sin x ^ 2 * Real.cos x ^ 2 ≥ 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l4135_413545


namespace NUMINAMATH_CALUDE_cosine_triple_angle_identity_l4135_413500

theorem cosine_triple_angle_identity (x : ℝ) : 
  4 * Real.cos x * Real.cos (x + π/3) * Real.cos (x - π/3) = Real.cos (3*x) := by
  sorry

end NUMINAMATH_CALUDE_cosine_triple_angle_identity_l4135_413500


namespace NUMINAMATH_CALUDE_ending_number_proof_l4135_413533

def starting_number : ℕ := 100
def multiples_count : ℚ := 13.5

theorem ending_number_proof :
  ∃ (n : ℕ), n ≥ starting_number ∧ 
  (n - starting_number) / 8 + 1 = multiples_count ∧
  n = 204 :=
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l4135_413533


namespace NUMINAMATH_CALUDE_kaleb_books_l4135_413596

theorem kaleb_books (initial_books sold_books new_books : ℕ) :
  initial_books ≥ sold_books →
  initial_books - sold_books + new_books = initial_books + new_books - sold_books :=
by
  sorry

#check kaleb_books 34 17 7

end NUMINAMATH_CALUDE_kaleb_books_l4135_413596


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l4135_413586

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : Nat.choose n 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l4135_413586


namespace NUMINAMATH_CALUDE_tan_equality_implies_negative_thirty_l4135_413562

theorem tan_equality_implies_negative_thirty
  (n : ℤ)
  (h1 : -90 < n ∧ n < 90)
  (h2 : Real.tan (n * π / 180) = Real.tan (1230 * π / 180)) :
  n = -30 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_implies_negative_thirty_l4135_413562


namespace NUMINAMATH_CALUDE_shuai_fen_solution_l4135_413546

/-- Represents the "Shuai Fen" distribution system -/
structure ShuaiFen where
  a : ℝ
  x : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_a_pos : a > 0
  h_c : c = 36
  h_bd : b + d = 75
  h_shuai_fen_b : (b - c) / b = x
  h_shuai_fen_c : (c - d) / c = x
  h_shuai_fen_a : (a - b) / a = x
  h_total : a = b + c + d

/-- The "Shuai Fen" problem solution -/
theorem shuai_fen_solution (sf : ShuaiFen) : sf.x = 0.25 ∧ sf.a = 175 := by
  sorry

end NUMINAMATH_CALUDE_shuai_fen_solution_l4135_413546


namespace NUMINAMATH_CALUDE_victors_initial_money_l4135_413560

/-- Victor's money problem -/
theorem victors_initial_money (initial_amount allowance total : ℕ) : 
  allowance = 8 → total = 18 → initial_amount + allowance = total → initial_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_victors_initial_money_l4135_413560


namespace NUMINAMATH_CALUDE_reps_before_high_elevation_pushups_l4135_413524

/-- Calculates the number of reps reached before moving to the next push-up type -/
def repsBeforeNextType (totalWeeks : ℕ) (typesOfPushups : ℕ) (daysPerWeek : ℕ) (repsAddedPerDay : ℕ) (initialReps : ℕ) : ℕ :=
  let weeksPerType : ℕ := totalWeeks / typesOfPushups
  let totalDays : ℕ := weeksPerType * daysPerWeek
  initialReps + (totalDays * repsAddedPerDay)

theorem reps_before_high_elevation_pushups :
  repsBeforeNextType 9 4 5 1 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_reps_before_high_elevation_pushups_l4135_413524


namespace NUMINAMATH_CALUDE_g_zero_at_three_l4135_413565

/-- The polynomial function g(x) -/
def g (x s : ℝ) : ℝ := 3*x^5 - 2*x^4 + x^3 - 4*x^2 + 5*x + s

/-- Theorem stating that g(3) = 0 when s = -573 -/
theorem g_zero_at_three : g 3 (-573) = 0 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l4135_413565


namespace NUMINAMATH_CALUDE_probability_of_snow_l4135_413569

/-- The probability of snow on at least one day out of four, given specific conditions --/
theorem probability_of_snow (p : ℝ) (q : ℝ) : 
  p = 3/4 →  -- probability of snow on each of the first three days
  q = 4/5 →  -- probability of snow on the last day if it snowed before
  (1 - (1 - p)^3 * (1 - p) - (1 - (1 - p)^3) * (1 - q)) = 1023/1280 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_snow_l4135_413569


namespace NUMINAMATH_CALUDE_bank_withdrawal_bill_value_l4135_413507

theorem bank_withdrawal_bill_value (x n : ℕ) (h1 : x = 300) (h2 : n = 30) :
  (2 * x) / n = 20 := by
  sorry

end NUMINAMATH_CALUDE_bank_withdrawal_bill_value_l4135_413507


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l4135_413516

/-- Given a triangle with vertices (0, 0), (x, 3x), and (2x, 0), 
    if its area is 150 square units and x > 0, then x = 5√2 -/
theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * (3*x) = 150 → x = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l4135_413516


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l4135_413567

theorem factor_difference_of_squares (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l4135_413567


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l4135_413591

theorem complex_magnitude_equation (n : ℝ) :
  n > 0 → (Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 6 ↔ n = 5 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l4135_413591


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l4135_413519

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l4135_413519


namespace NUMINAMATH_CALUDE_coffee_mug_price_l4135_413513

/-- The cost of a personalized coffee mug -/
def coffee_mug_cost : ℕ := sorry

/-- The price of a bracelet -/
def bracelet_price : ℕ := 15

/-- The price of a gold heart necklace -/
def necklace_price : ℕ := 10

/-- The number of bracelets bought -/
def num_bracelets : ℕ := 3

/-- The number of gold heart necklaces bought -/
def num_necklaces : ℕ := 2

/-- The amount paid with -/
def amount_paid : ℕ := 100

/-- The change received -/
def change_received : ℕ := 15

theorem coffee_mug_price : coffee_mug_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_mug_price_l4135_413513


namespace NUMINAMATH_CALUDE_max_value_on_circle_l4135_413508

theorem max_value_on_circle (x y z : ℝ) (h : x^2 + y^2 - 2*x + 2*y - 1 = 0) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 + Real.sqrt 3 ∧ 
  ∀ (w : ℝ), w = (x + 1) * Real.sin z + (y - 1) * Real.cos z → w ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l4135_413508


namespace NUMINAMATH_CALUDE_bounce_height_theorem_l4135_413577

/-- The number of bounces required for a ball to reach a height less than 3 meters -/
def number_of_bounces : ℕ := 22

/-- The initial height of the ball in meters -/
def initial_height : ℝ := 500

/-- The bounce ratio (percentage of height retained after each bounce) -/
def bounce_ratio : ℝ := 0.6

/-- The target height in meters -/
def target_height : ℝ := 3

/-- Theorem stating that the number of bounces is correct -/
theorem bounce_height_theorem :
  (∀ k : ℕ, k < number_of_bounces → initial_height * bounce_ratio ^ k ≥ target_height) ∧
  (initial_height * bounce_ratio ^ number_of_bounces < target_height) :=
sorry

end NUMINAMATH_CALUDE_bounce_height_theorem_l4135_413577


namespace NUMINAMATH_CALUDE_mat_weaving_problem_l4135_413589

/-- Given that 4 mat-weaves can weave 4 mats in 4 days, 
    prove that 8 mat-weaves will weave 16 mats in 8 days. -/
theorem mat_weaving_problem (weave_rate : ℕ → ℕ → ℕ → ℕ) :
  weave_rate 4 4 4 = 4 →
  weave_rate 8 16 8 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_mat_weaving_problem_l4135_413589


namespace NUMINAMATH_CALUDE_prob_six_given_hugo_wins_l4135_413553

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The probability of rolling a 6 on a single die -/
def prob_roll_six : ℚ := 1 / die_sides

/-- The probability of Hugo winning the game -/
def prob_hugo_wins : ℚ := 1 / num_players

/-- The probability that Hugo wins given his first roll was a 6 -/
noncomputable def prob_hugo_wins_given_six : ℚ := 875 / 1296

/-- Theorem: The probability that Hugo's first roll was 6, given that he won the game -/
theorem prob_six_given_hugo_wins :
  (prob_roll_six * prob_hugo_wins_given_six) / prob_hugo_wins = 4375 / 7776 := by sorry

end NUMINAMATH_CALUDE_prob_six_given_hugo_wins_l4135_413553


namespace NUMINAMATH_CALUDE_new_rectangle_perimeter_l4135_413595

/-- Given a rectangle ABCD composed of four congruent triangles -/
structure Rectangle :=
  (AB BC : ℝ)
  (AK : ℝ)
  (perimeter : ℝ)
  (h1 : perimeter = 4 * (AB + BC / 2 + AK))
  (h2 : AK = 17)
  (h3 : perimeter = 180)

/-- The perimeter of a new rectangle with sides 2*AB and BC -/
def new_perimeter (r : Rectangle) : ℝ :=
  2 * (2 * r.AB + r.BC)

/-- Theorem stating the perimeter of the new rectangle is 112 cm -/
theorem new_rectangle_perimeter (r : Rectangle) : new_perimeter r = 112 :=
sorry

end NUMINAMATH_CALUDE_new_rectangle_perimeter_l4135_413595


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l4135_413548

theorem polynomial_coefficient_bound (a b c d : ℝ) : 
  (∀ x : ℝ, |x| < 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l4135_413548


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l4135_413512

/-- Given an isosceles right triangle with hypotenuse 6√2, prove its area is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (is_hypotenuse : h = 6 * Real.sqrt 2) :
  let a : ℝ := h / Real.sqrt 2
  let area : ℝ := (1 / 2) * a ^ 2
  area = 18 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l4135_413512


namespace NUMINAMATH_CALUDE_paint_usage_l4135_413557

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : initial_paint = 360)
  (h2 : first_week_fraction = 1/6)
  (h3 : second_week_fraction = 1/5) :
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 120 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l4135_413557


namespace NUMINAMATH_CALUDE_parentheses_placement_l4135_413556

theorem parentheses_placement :
  (1 : ℚ) / (2 / (3 / (4 / (5 / (6 / (7 / (8 / (9 / 10)))))))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_placement_l4135_413556


namespace NUMINAMATH_CALUDE_largest_four_digit_product_of_primes_l4135_413540

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a four-digit positive integer -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_product_of_primes :
  ∃ (n x y : ℕ),
    isFourDigit n ∧
    isPrime x ∧
    isPrime y ∧
    x < 10 ∧
    y < 10 ∧
    isPrime (10 * y + x) ∧
    n = x * y * (10 * y + x) ∧
    (∀ (m a b : ℕ),
      isFourDigit m →
      isPrime a →
      isPrime b →
      a < 10 →
      b < 10 →
      isPrime (10 * b + a) →
      m = a * b * (10 * b + a) →
      m ≤ n) ∧
    n = 1533 :=
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_product_of_primes_l4135_413540


namespace NUMINAMATH_CALUDE_words_removed_during_editing_l4135_413539

theorem words_removed_during_editing 
  (yvonne_words : ℕ)
  (janna_words : ℕ)
  (words_removed : ℕ)
  (words_added : ℕ)
  (h1 : yvonne_words = 400)
  (h2 : janna_words = yvonne_words + 150)
  (h3 : words_added = 2 * words_removed)
  (h4 : yvonne_words + janna_words - words_removed + words_added + 30 = 1000) :
  words_removed = 20 := by
  sorry

end NUMINAMATH_CALUDE_words_removed_during_editing_l4135_413539


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l4135_413564

theorem cos_arcsin_three_fifths : Real.cos (Real.arcsin (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l4135_413564


namespace NUMINAMATH_CALUDE_base_conversion_equality_l4135_413554

theorem base_conversion_equality (b : ℕ) : b > 0 ∧ (4 * 6 + 2 = 1 * b^2 + 2 * b + 1) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l4135_413554


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l4135_413551

/-- Given a rectangle with an inscribed circle of radius 6 and a length-to-width ratio of 3:1,
    prove that the area of the rectangle is 432. -/
theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) :
  r = 6 →
  ratio = 3 →
  let width := 2 * r
  let length := ratio * width
  width * length = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l4135_413551


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l4135_413506

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Check if a ScientificNotation represents a given real number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  x = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number we want to represent in scientific notation -/
def target_number : ℝ := 37000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.7
    exponent := 7
    h_coeff_range := by sorry }

theorem scientific_notation_correct :
  represents proposed_notation target_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l4135_413506


namespace NUMINAMATH_CALUDE_min_value_of_function_l4135_413590

theorem min_value_of_function (m n : ℝ) : 
  m > 0 → n > 0 →  -- point in first quadrant
  ∃ (a b : ℝ), (m + a) / 2 + (n + b) / 2 - 2 = 0 →  -- symmetry condition
  2 * a + b + 3 = 0 →  -- (a,b) lies on 2x+y+3=0
  (n - b) / (m - a) = 1 →  -- slope of line of symmetry
  2 * m + n + 3 = 0 →  -- (m,n) lies on 2x+y+3=0
  (1 / m + 8 / n) ≥ 25 / 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4135_413590


namespace NUMINAMATH_CALUDE_marie_magazines_sold_l4135_413543

/-- The number of magazines Marie sold -/
def magazines_sold : ℕ := 700 - 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := 700

/-- The number of newspapers Marie sold -/
def newspapers_sold : ℕ := 275

theorem marie_magazines_sold :
  magazines_sold = 425 ∧
  magazines_sold + newspapers_sold = total_reading_materials :=
sorry

end NUMINAMATH_CALUDE_marie_magazines_sold_l4135_413543


namespace NUMINAMATH_CALUDE_xiao_hong_age_expression_dad_age_when_xiao_hong_is_seven_l4135_413501

-- Define Dad's age
def dad_age : ℕ → ℕ := λ a => a

-- Define Xiao Hong's age as a function of Dad's age
def xiao_hong_age : ℕ → ℚ := λ a => (a - 3) / 4

-- Theorem for Xiao Hong's age expression
theorem xiao_hong_age_expression (a : ℕ) :
  xiao_hong_age a = (a - 3) / 4 :=
sorry

-- Theorem for Dad's age when Xiao Hong is 7
theorem dad_age_when_xiao_hong_is_seven :
  ∃ a : ℕ, xiao_hong_age a = 7 ∧ dad_age a = 31 :=
sorry

end NUMINAMATH_CALUDE_xiao_hong_age_expression_dad_age_when_xiao_hong_is_seven_l4135_413501


namespace NUMINAMATH_CALUDE_max_balloons_is_400_l4135_413523

def small_bag_cost : ℕ := 4
def small_bag_balloons : ℕ := 50
def medium_bag_cost : ℕ := 6
def medium_bag_balloons : ℕ := 75
def large_bag_cost : ℕ := 12
def large_bag_balloons : ℕ := 200
def budget : ℕ := 24

def max_balloons (budget small_cost small_balloons medium_cost medium_balloons large_cost large_balloons : ℕ) : ℕ := 
  sorry

theorem max_balloons_is_400 : 
  max_balloons budget small_bag_cost small_bag_balloons medium_bag_cost medium_bag_balloons large_bag_cost large_bag_balloons = 400 :=
by sorry

end NUMINAMATH_CALUDE_max_balloons_is_400_l4135_413523


namespace NUMINAMATH_CALUDE_parallelepiped_length_l4135_413563

theorem parallelepiped_length (n : ℕ) : 
  n > 6 →
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_length_l4135_413563


namespace NUMINAMATH_CALUDE_hockey_league_games_l4135_413530

structure Team where
  games_played : ℕ
  games_won : ℕ
  win_ratio : ℚ

def team_X : Team → Prop
| t => t.win_ratio = 3/4

def team_Y : Team → Prop
| t => t.win_ratio = 2/3

theorem hockey_league_games (X Y : Team) : 
  team_X X → team_Y Y → 
  Y.games_played = X.games_played + 12 →
  Y.games_won = X.games_won + 4 →
  X.games_played = 48 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l4135_413530


namespace NUMINAMATH_CALUDE_function_property_implies_f3_values_l4135_413511

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionType) : Prop :=
  ∀ x y : ℝ, f (x * f y - x) = x * y - f x

-- State the theorem
theorem function_property_implies_f3_values (f : FunctionType) 
  (h : SatisfiesProperty f) : 
  ∃ (a b : ℝ), (∀ z : ℝ, f 3 = z → (z = a ∨ z = b)) ∧ a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_f3_values_l4135_413511


namespace NUMINAMATH_CALUDE_injury_point_is_20_l4135_413597

/-- Represents the runner's journey from Marathon to Athens -/
structure RunnerJourney where
  totalDistance : ℝ
  injuryPoint : ℝ
  initialSpeed : ℝ
  secondPartTime : ℝ
  timeDifference : ℝ

/-- The conditions of the runner's journey -/
def journeyConditions (j : RunnerJourney) : Prop :=
  j.totalDistance = 40 ∧
  j.secondPartTime = 22 ∧
  j.timeDifference = 11 ∧
  j.initialSpeed > 0 ∧
  j.injuryPoint > 0 ∧
  j.injuryPoint < j.totalDistance ∧
  (j.totalDistance - j.injuryPoint) / (j.initialSpeed / 2) = j.secondPartTime ∧
  (j.totalDistance - j.injuryPoint) / (j.initialSpeed / 2) = j.injuryPoint / j.initialSpeed + j.timeDifference

/-- Theorem stating that given the journey conditions, the injury point is at 20 miles -/
theorem injury_point_is_20 (j : RunnerJourney) (h : journeyConditions j) : j.injuryPoint = 20 := by
  sorry

#check injury_point_is_20

end NUMINAMATH_CALUDE_injury_point_is_20_l4135_413597


namespace NUMINAMATH_CALUDE_jacket_discount_percentage_l4135_413520

/-- Proves that the discount percentage is 20% given the specified conditions --/
theorem jacket_discount_percentage (purchase_price selling_price discount_price : ℝ) 
  (h1 : purchase_price = 54)
  (h2 : selling_price = purchase_price + 0.4 * selling_price)
  (h3 : discount_price - purchase_price = 18) : 
  (selling_price - discount_price) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_jacket_discount_percentage_l4135_413520


namespace NUMINAMATH_CALUDE_amoeba_count_day_10_l4135_413528

def amoeba_count (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % 3 = 0 then (3 * amoeba_count (n - 1)) / 2
  else 2 * amoeba_count (n - 1)

theorem amoeba_count_day_10 : amoeba_count 10 = 432 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_day_10_l4135_413528


namespace NUMINAMATH_CALUDE_molly_has_three_brothers_l4135_413547

/-- Represents the problem of determining Molly's number of brothers --/
def MollysBrothers (cost_per_package : ℕ) (num_parents : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (num_brothers : ℕ),
    cost_per_package * (num_parents + num_brothers + num_brothers + 2 * num_brothers) = total_cost

/-- Theorem stating that Molly has 3 brothers given the problem conditions --/
theorem molly_has_three_brothers :
  MollysBrothers 5 2 70 → ∃ (num_brothers : ℕ), num_brothers = 3 := by
  sorry

end NUMINAMATH_CALUDE_molly_has_three_brothers_l4135_413547


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l4135_413572

theorem sum_of_cyclic_equations (p q r : ℝ) 
  (distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (eq1 : q = p * (4 - p))
  (eq2 : r = q * (4 - q))
  (eq3 : p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l4135_413572


namespace NUMINAMATH_CALUDE_f_extreme_value_and_negative_range_l4135_413555

/-- The function f(x) defined on (0, +∞) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x + (Real.log x - 2) / x + 1

theorem f_extreme_value_and_negative_range :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x) ∧
  f 0 (Real.exp 3) = 1 / Real.exp 3 + 1 ∧
  ∀ (m : ℝ), (∀ (x : ℝ), x > 0 → f m x < 0) ↔ m < -1 / Real.exp 3 :=
by sorry

end NUMINAMATH_CALUDE_f_extreme_value_and_negative_range_l4135_413555


namespace NUMINAMATH_CALUDE_removed_carrots_average_weight_l4135_413573

/-- Proves that the average weight of 4 removed carrots is 190 grams -/
theorem removed_carrots_average_weight
  (total_weight : ℝ)
  (remaining_carrots : ℕ)
  (removed_carrots : ℕ)
  (remaining_average : ℝ)
  (h1 : total_weight = 3.64)
  (h2 : remaining_carrots = 16)
  (h3 : removed_carrots = 4)
  (h4 : remaining_average = 180)
  (h5 : remaining_carrots + removed_carrots = 20) :
  (total_weight * 1000 - remaining_carrots * remaining_average) / removed_carrots = 190 :=
by sorry

end NUMINAMATH_CALUDE_removed_carrots_average_weight_l4135_413573


namespace NUMINAMATH_CALUDE_common_difference_is_fifteen_exists_valid_prism_l4135_413504

/-- Represents a rectangular prism with sides that are consecutive multiples of a certain number -/
structure RectangularPrism where
  base_number : ℕ
  common_difference : ℕ

/-- The base area of the rectangular prism -/
def base_area (prism : RectangularPrism) : ℕ :=
  prism.base_number * (prism.base_number + prism.common_difference)

/-- Theorem stating that for a rectangular prism with base area 450,
    the common difference between consecutive multiples is 15 -/
theorem common_difference_is_fifteen (prism : RectangularPrism) 
    (h : base_area prism = 450) : prism.common_difference = 15 := by
  sorry

/-- Proof of the existence of a rectangular prism satisfying the conditions -/
theorem exists_valid_prism : ∃ (prism : RectangularPrism), base_area prism = 450 ∧ prism.common_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_fifteen_exists_valid_prism_l4135_413504


namespace NUMINAMATH_CALUDE_problem_solution_l4135_413570

theorem problem_solution : 
  ((-1)^3 + |1 - Real.sqrt 2| + (8 : ℝ)^(1/3) = Real.sqrt 2) ∧
  (((-5 : ℝ)^3)^(1/3) + (-3)^2 - Real.sqrt 25 + |Real.sqrt 3 - 2| + (Real.sqrt 3)^2 = 4 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4135_413570


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_65_l4135_413529

theorem perfect_square_power_of_two_plus_65 (n : ℕ+) :
  (∃ (x : ℕ), 2^n.val + 65 = x^2) ↔ n.val = 4 ∨ n.val = 10 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_65_l4135_413529


namespace NUMINAMATH_CALUDE_equation_solution_l4135_413534

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 90 + 5 * 12 / (180 / x) = 91 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4135_413534


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l4135_413527

theorem max_value_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  x + 2*y ≤ Real.sqrt (5/18) + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l4135_413527


namespace NUMINAMATH_CALUDE_solution_set_when_t_3_non_negative_for_all_x_iff_t_1_l4135_413579

-- Define the function f
def f (t x : ℝ) : ℝ := x^2 - (t+1)*x + t

-- Theorem for part 1
theorem solution_set_when_t_3 :
  {x : ℝ | f 3 x > 0} = {x : ℝ | x < 1 ∨ x > 3} := by sorry

-- Theorem for part 2
theorem non_negative_for_all_x_iff_t_1 :
  (∀ x : ℝ, f t x ≥ 0) ↔ t = 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_t_3_non_negative_for_all_x_iff_t_1_l4135_413579


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4135_413558

theorem min_value_quadratic (x y : ℝ) :
  y = x^2 + 16*x + 20 → ∀ z : ℝ, y ≥ -44 ∧ (∃ x₀ : ℝ, x₀^2 + 16*x₀ + 20 = -44) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4135_413558


namespace NUMINAMATH_CALUDE_subtract_sum_digits_100_times_is_zero_l4135_413561

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Computes the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Performs one iteration of subtracting the sum of digits -/
def subtract_sum_of_digits (n : ThreeDigitNumber) : ℕ := 
  n.value - sum_of_digits n.value

/-- Performs the subtraction process n times -/
def iterate_subtraction (n : ThreeDigitNumber) (iterations : ℕ) : ℕ := sorry

/-- Theorem: After 100 iterations of subtracting the sum of digits from any three-digit number, the result is zero -/
theorem subtract_sum_digits_100_times_is_zero (n : ThreeDigitNumber) : 
  iterate_subtraction n 100 = 0 := by sorry

end NUMINAMATH_CALUDE_subtract_sum_digits_100_times_is_zero_l4135_413561


namespace NUMINAMATH_CALUDE_system_ratio_value_l4135_413535

/-- Given a system of linear equations with a nontrivial solution,
    prove that the ratio xy/z^2 has a specific value. -/
theorem system_ratio_value (x y z k : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  z ≠ 0 →
  x + k*y + 4*z = 0 →
  3*x + k*y - 3*z = 0 →
  2*x + 5*y - 3*z = 0 →
  -- The condition for nontrivial solution is implicitly included in the equations
  ∃ (c : ℝ), x*y / (z^2) = c :=
by
  sorry


end NUMINAMATH_CALUDE_system_ratio_value_l4135_413535


namespace NUMINAMATH_CALUDE_action_figure_collection_l4135_413576

/-- The problem of calculating the total number of action figures needed for a complete collection. -/
theorem action_figure_collection
  (jerry_has : ℕ)
  (cost_per_figure : ℕ)
  (total_cost_to_finish : ℕ)
  (h1 : jerry_has = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost_to_finish = 72) :
  jerry_has + total_cost_to_finish / cost_per_figure = 16 :=
by sorry

end NUMINAMATH_CALUDE_action_figure_collection_l4135_413576


namespace NUMINAMATH_CALUDE_runs_by_running_percentage_l4135_413580

def total_runs : ℕ := 120
def boundaries : ℕ := 6
def sixes : ℕ := 4
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem runs_by_running_percentage :
  let runs_from_boundaries := boundaries * runs_per_boundary
  let runs_from_sixes := sixes * runs_per_six
  let runs_without_running := runs_from_boundaries + runs_from_sixes
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℚ) / total_runs * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_runs_by_running_percentage_l4135_413580


namespace NUMINAMATH_CALUDE_number_whose_quarter_is_nine_more_l4135_413588

theorem number_whose_quarter_is_nine_more (x : ℚ) : (x / 4 = x + 9) → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_number_whose_quarter_is_nine_more_l4135_413588


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l4135_413583

theorem min_value_of_function (x : ℝ) (h : x > 0) : (x^2 + 1) / x ≥ 2 :=
  sorry

theorem equality_condition (x : ℝ) (h : x > 0) : (x^2 + 1) / x = 2 ↔ x = 1 :=
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l4135_413583


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l4135_413598

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l4135_413598


namespace NUMINAMATH_CALUDE_ship_passengers_heads_l4135_413585

/-- Represents the number of heads and legs on a ship with cats, crew, and a one-legged captain. -/
structure ShipPassengers where
  cats : ℕ
  crew : ℕ
  captain : ℕ := 1

/-- Calculates the total number of heads on the ship. -/
def totalHeads (p : ShipPassengers) : ℕ :=
  p.cats + p.crew + p.captain

/-- Calculates the total number of legs on the ship. -/
def totalLegs (p : ShipPassengers) : ℕ :=
  p.cats * 4 + p.crew * 2 + 1

/-- Theorem stating that given the conditions, the total number of heads on the ship is 14. -/
theorem ship_passengers_heads :
  ∃ (p : ShipPassengers),
    p.cats = 7 ∧
    totalLegs p = 41 ∧
    totalHeads p = 14 :=
sorry

end NUMINAMATH_CALUDE_ship_passengers_heads_l4135_413585


namespace NUMINAMATH_CALUDE_cube_pyramid_volume_equality_l4135_413584

theorem cube_pyramid_volume_equality (h : ℝ) : 
  let cube_edge : ℝ := 6
  let pyramid_base : ℝ := 12
  let cube_volume : ℝ := cube_edge^3
  let pyramid_volume : ℝ := (1/3) * pyramid_base^2 * h
  cube_volume = pyramid_volume → h = 4.5 := by
sorry

end NUMINAMATH_CALUDE_cube_pyramid_volume_equality_l4135_413584


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l4135_413517

theorem interest_rate_calculation (principal : ℝ) (difference : ℝ) (time : ℕ) (rate : ℝ) : 
  principal = 15000 →
  difference = 150 →
  time = 2 →
  principal * ((1 + rate)^time - 1) - principal * rate * time = difference →
  rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l4135_413517


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4135_413552

theorem arithmetic_mean_problem (y : ℚ) : 
  ((y + 10) + 20 + (3 * y) + 18 + (3 * y + 6) + 12) / 6 = 30 → y = 114 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4135_413552


namespace NUMINAMATH_CALUDE_solution_correctness_l4135_413581

-- First system of equations
def system1 (x y : ℝ) : Prop :=
  3 * x + 2 * y = 6 ∧ y = x - 2

-- Second system of equations
def system2 (m n : ℝ) : Prop :=
  m + 2 * n = 7 ∧ -3 * m + 5 * n = 1

theorem solution_correctness :
  (∃ x y, system1 x y) ∧ (∃ m n, system2 m n) ∧
  (∀ x y, system1 x y → x = 2 ∧ y = 0) ∧
  (∀ m n, system2 m n → m = 3 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l4135_413581


namespace NUMINAMATH_CALUDE_sleep_deficit_l4135_413522

def weeknights : ℕ := 5
def weekendNights : ℕ := 2
def actualWeekdaySleep : ℕ := 5
def actualWeekendSleep : ℕ := 6
def idealSleep : ℕ := 8

theorem sleep_deficit :
  (weeknights * idealSleep + weekendNights * idealSleep) -
  (weeknights * actualWeekdaySleep + weekendNights * actualWeekendSleep) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sleep_deficit_l4135_413522


namespace NUMINAMATH_CALUDE_equal_gumball_share_l4135_413593

def gumball_distribution (joanna_initial : ℕ) (jacques_initial : ℕ) (purchase_multiplier : ℕ) : ℕ :=
  let joanna_total := joanna_initial + joanna_initial * purchase_multiplier
  let jacques_total := jacques_initial + jacques_initial * purchase_multiplier
  let combined_total := joanna_total + jacques_total
  combined_total / 2

theorem equal_gumball_share :
  gumball_distribution 40 60 4 = 250 := by sorry

end NUMINAMATH_CALUDE_equal_gumball_share_l4135_413593


namespace NUMINAMATH_CALUDE_equal_probability_events_l4135_413515

/-- Given a jar with 'a' white balls and 'b' black balls, where a ≠ b, this theorem proves that
    the probability of Event A (at some point, the number of drawn white balls equals the number
    of drawn black balls) is equal to the probability of Event B (at some point, the number of
    white balls remaining in the jar equals the number of black balls remaining in the jar),
    and that this probability is (2 * min(a, b)) / (a + b). -/
theorem equal_probability_events (a b : ℕ) (h : a ≠ b) :
  let total := a + b
  let prob_A := (2 * min a b) / total
  let prob_B := (2 * min a b) / total
  prob_A = prob_B ∧ prob_A = (2 * min a b) / total := by
  sorry

#check equal_probability_events

end NUMINAMATH_CALUDE_equal_probability_events_l4135_413515


namespace NUMINAMATH_CALUDE_dog_food_duration_l4135_413531

/-- Given a dog's feeding schedule and a bag of dog food, calculate how many days the food will last. -/
theorem dog_food_duration (morning_food evening_food bag_size : ℕ) : 
  morning_food = 1 → 
  evening_food = 1 → 
  bag_size = 32 → 
  (bag_size / (morning_food + evening_food) : ℕ) = 16 := by
sorry

end NUMINAMATH_CALUDE_dog_food_duration_l4135_413531


namespace NUMINAMATH_CALUDE_sample_for_x_24_possible_x_for_87_l4135_413538

/-- Represents a systematic sampling method for a population of 1000 individuals. -/
def systematicSample (x : Nat) : List Nat :=
  List.range 10
    |>.map (fun k => (x + 33 * k) % 1000)

/-- Checks if a number ends with given digits. -/
def endsWithDigits (n : Nat) (digits : Nat) : Bool :=
  n % 100 = digits

/-- Theorem for the first part of the problem. -/
theorem sample_for_x_24 :
    systematicSample 24 = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921] := by
  sorry

/-- Theorem for the second part of the problem. -/
theorem possible_x_for_87 :
    {x : Nat | ∃ n ∈ systematicSample x, endsWithDigits n 87} =
    {87, 54, 21, 88, 55, 22, 89, 56, 23, 90} := by
  sorry

end NUMINAMATH_CALUDE_sample_for_x_24_possible_x_for_87_l4135_413538


namespace NUMINAMATH_CALUDE_money_difference_l4135_413549

/-- Given Eliza has 7q + 3 quarters and Tom has 2q + 8 quarters, where every 5 quarters
    over the count of the other person are converted into nickels, the difference in
    their money is 5(q - 1) cents. -/
theorem money_difference (q : ℤ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := eliza_quarters - tom_quarters
  let nickel_groups := quarter_difference / 5
  nickel_groups * 5 = 5 * (q - 1) := by sorry

end NUMINAMATH_CALUDE_money_difference_l4135_413549


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l4135_413526

/-- The curve W defined by y = x^2 + 1/4 -/
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + 1/4}

/-- A rectangle with vertices as points in ℝ × ℝ -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (dist r.A r.B + dist r.B r.C)

/-- Three vertices of the rectangle are on W -/
def three_vertices_on_W (r : Rectangle) : Prop :=
  (r.A ∈ W ∧ r.B ∈ W ∧ r.C ∈ W) ∨
  (r.A ∈ W ∧ r.B ∈ W ∧ r.D ∈ W) ∨
  (r.A ∈ W ∧ r.C ∈ W ∧ r.D ∈ W) ∨
  (r.B ∈ W ∧ r.C ∈ W ∧ r.D ∈ W)

theorem rectangle_perimeter_bound (r : Rectangle) 
  (h : three_vertices_on_W r) : 
  perimeter r > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l4135_413526


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l4135_413559

/-- The trajectory of the midpoint of a line segment from a point on a hyperbola to its perpendicular projection on a line -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₁ y₁ : ℝ, 
    -- Q(x₁, y₁) is on the hyperbola x^2 - y^2 = 1
    x₁^2 - y₁^2 = 1 ∧ 
    -- N(2x - x₁, 2y - y₁) is on the line x + y = 2
    (2*x - x₁) + (2*y - y₁) = 2 ∧ 
    -- PQ is perpendicular to the line x + y = 2
    (y - y₁) = (x - x₁) ∧ 
    -- P(x, y) is the midpoint of QN
    x = (x₁ + (2*x - x₁)) / 2 ∧ 
    y = (y₁ + (2*y - y₁)) / 2) →
  -- The trajectory equation of P(x, y)
  2*x^2 - 2*y^2 - 2*x + 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l4135_413559


namespace NUMINAMATH_CALUDE_shaded_square_area_fraction_l4135_413521

/-- The area of a square with vertices at (3,2), (5,4), (3,6), and (1,4) on a 6x6 grid is 2/9 of the total grid area. -/
theorem shaded_square_area_fraction :
  let grid_size : ℕ := 6
  let total_area : ℝ := (grid_size : ℝ) ^ 2
  let shaded_square_vertices : List (ℕ × ℕ) := [(3, 2), (5, 4), (3, 6), (1, 4)]
  let shaded_square_side : ℝ := 2 * Real.sqrt 2
  let shaded_square_area : ℝ := shaded_square_side ^ 2
  shaded_square_area / total_area = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_shaded_square_area_fraction_l4135_413521
