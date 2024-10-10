import Mathlib

namespace equation_simplification_l2377_237717

theorem equation_simplification :
  120 + (150 / 10) + (35 * 9) - 300 - (420 / 7) + 2^3 = 98 := by
  sorry

end equation_simplification_l2377_237717


namespace chocolates_in_boxes_l2377_237790

theorem chocolates_in_boxes (total_chocolates : ℕ) (filled_boxes : ℕ) (loose_chocolates : ℕ) (friend_chocolates : ℕ) (box_capacity : ℕ) : 
  total_chocolates = 50 →
  filled_boxes = 3 →
  loose_chocolates = 5 →
  friend_chocolates = 25 →
  box_capacity = 15 →
  (total_chocolates - loose_chocolates) / filled_boxes = box_capacity →
  (loose_chocolates + friend_chocolates) / box_capacity = 2 := by
sorry

end chocolates_in_boxes_l2377_237790


namespace complex_modulus_product_l2377_237708

theorem complex_modulus_product : 
  Complex.abs ((10 - 5 * Complex.I) * (7 + 24 * Complex.I)) = 125 * Real.sqrt 5 := by
  sorry

end complex_modulus_product_l2377_237708


namespace inverse_g_at_19_128_l2377_237754

noncomputable def g (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_g_at_19_128 :
  g⁻¹ (19/128) = (51/32)^(1/7) := by
sorry

end inverse_g_at_19_128_l2377_237754


namespace fraction_always_defined_l2377_237737

theorem fraction_always_defined (x : ℝ) : (x^2 + 2 ≠ 0) := by
  sorry

end fraction_always_defined_l2377_237737


namespace sum_of_two_arithmetic_sequences_l2377_237740

/-- Sum of two arithmetic sequences with specific properties -/
theorem sum_of_two_arithmetic_sequences : 
  let seq1 := [2, 14, 26, 38, 50]
  let seq2 := [6, 18, 30, 42, 54]
  (seq1.sum + seq2.sum) = 280 := by sorry

end sum_of_two_arithmetic_sequences_l2377_237740


namespace valid_number_count_l2377_237789

/-- Represents a valid seven-digit number configuration --/
structure ValidNumber :=
  (digits : Fin 7 → Fin 7)
  (injective : Function.Injective digits)
  (no_6_7_at_ends : digits 0 ≠ 5 ∧ digits 0 ≠ 6 ∧ digits 6 ≠ 5 ∧ digits 6 ≠ 6)
  (one_adjacent_six : ∃ i, (digits i = 0 ∧ digits (i+1) = 5) ∨ (digits i = 5 ∧ digits (i+1) = 0))

/-- The number of valid seven-digit numbers --/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the count of valid numbers --/
theorem valid_number_count : count_valid_numbers = 768 := by sorry

end valid_number_count_l2377_237789


namespace tangent_slope_ratio_l2377_237739

-- Define the function f(x) = ax² + b
def f (a b x : ℝ) : ℝ := a * x^2 + b

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 2 * a * x

theorem tangent_slope_ratio (a b : ℝ) :
  f_derivative a b 1 = 2 ∧ f a b 1 = 3 → a / b = 1 / 2 := by
  sorry

end tangent_slope_ratio_l2377_237739


namespace certain_amount_calculation_l2377_237711

theorem certain_amount_calculation (x : ℝ) (A : ℝ) (h1 : x = 190) (h2 : 0.65 * x = 0.20 * A) : A = 617.5 := by
  sorry

end certain_amount_calculation_l2377_237711


namespace remaining_macaroons_weight_l2377_237750

def macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let total_weight := total_macaroons * weight_per_macaroon
  let macaroons_per_bag := total_macaroons / num_bags
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon
  total_weight - (bags_eaten * weight_per_bag)

theorem remaining_macaroons_weight :
  macaroon_problem 12 5 4 1 = 45 := by
  sorry

end remaining_macaroons_weight_l2377_237750


namespace lcm_from_hcf_and_product_l2377_237735

theorem lcm_from_hcf_and_product (x y : ℕ+) : 
  Nat.gcd x y = 12 → x * y = 2460 → Nat.lcm x y = 205 := by
  sorry

end lcm_from_hcf_and_product_l2377_237735


namespace rectangle_formations_l2377_237755

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of horizontal lines -/
def horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def vertical_lines : ℕ := 5

/-- The number of horizontal lines needed to form a rectangle -/
def horizontal_lines_needed : ℕ := 2

/-- The number of vertical lines needed to form a rectangle -/
def vertical_lines_needed : ℕ := 2

/-- The theorem stating the number of ways to form a rectangle -/
theorem rectangle_formations :
  (choose horizontal_lines horizontal_lines_needed) *
  (choose vertical_lines vertical_lines_needed) = 100 := by
  sorry

end rectangle_formations_l2377_237755


namespace inequality_system_solution_l2377_237716

theorem inequality_system_solution (x : ℝ) : 
  (x - 2 < 0 ∧ 5 * x + 1 > 2 * (x - 1)) ↔ -1/3 < x ∧ x < 2 := by
  sorry

end inequality_system_solution_l2377_237716


namespace initial_fee_equals_65_l2377_237729

/-- The initial fee of the first car rental plan -/
def initial_fee : ℝ := 65

/-- The cost per mile for the first plan -/
def cost_per_mile_plan1 : ℝ := 0.40

/-- The cost per mile for the second plan -/
def cost_per_mile_plan2 : ℝ := 0.60

/-- The number of miles driven -/
def miles_driven : ℝ := 325

/-- Theorem stating that the initial fee makes both plans cost the same for the given miles -/
theorem initial_fee_equals_65 :
  initial_fee + cost_per_mile_plan1 * miles_driven = cost_per_mile_plan2 * miles_driven :=
by sorry

end initial_fee_equals_65_l2377_237729


namespace book_cost_problem_l2377_237769

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) 
  (h1 : total_cost = 420)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.19)
  (h4 : ∃ (sell_price : ℝ), 
    sell_price = (1 - loss_percent) * (total_cost - x) ∧ 
    sell_price = (1 + gain_percent) * x) : 
  ∃ (x : ℝ), x = 245 ∧ x + (total_cost - x) = total_cost := by
sorry

end book_cost_problem_l2377_237769


namespace optimal_sale_info_l2377_237764

/-- Represents the selling prices and quantities of notebooks and sticky notes -/
structure SaleInfo where
  notebook_price : ℝ
  sticky_note_price : ℝ
  notebook_quantity : ℕ
  sticky_note_quantity : ℕ

/-- Calculates the total income given the sale information -/
def total_income (s : SaleInfo) : ℝ :=
  s.notebook_price * s.notebook_quantity + s.sticky_note_price * s.sticky_note_quantity

/-- Theorem stating the optimal selling prices and quantities for maximum income -/
theorem optimal_sale_info :
  ∃ (s : SaleInfo),
    -- Total number of items is 100
    s.notebook_quantity + s.sticky_note_quantity = 100 ∧
    -- 3 notebooks and 2 sticky notes sold for 65 yuan
    3 * s.notebook_price + 2 * s.sticky_note_price = 65 ∧
    -- 4 notebooks and 3 sticky notes sold for 90 yuan
    4 * s.notebook_price + 3 * s.sticky_note_price = 90 ∧
    -- Number of notebooks does not exceed 3 times the number of sticky notes
    s.notebook_quantity ≤ 3 * s.sticky_note_quantity ∧
    -- Notebook price is 15 yuan
    s.notebook_price = 15 ∧
    -- Sticky note price is 10 yuan
    s.sticky_note_price = 10 ∧
    -- Optimal quantities are 75 notebooks and 25 sticky notes
    s.notebook_quantity = 75 ∧
    s.sticky_note_quantity = 25 ∧
    -- Maximum total income is 1375 yuan
    total_income s = 1375 ∧
    -- This is the maximum income
    ∀ (t : SaleInfo),
      t.notebook_quantity + t.sticky_note_quantity = 100 →
      t.notebook_quantity ≤ 3 * t.sticky_note_quantity →
      total_income t ≤ total_income s := by
  sorry

end optimal_sale_info_l2377_237764


namespace digit_equation_solution_l2377_237733

theorem digit_equation_solution :
  ∀ (A M C : ℕ),
    A ≤ 9 → M ≤ 9 → C ≤ 9 →
    (100 * A + 10 * M + C) * (2 * (A + M + C + 1)) = 4010 →
    A = 4 := by
  sorry

end digit_equation_solution_l2377_237733


namespace right_triangle_legs_l2377_237785

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the first segment of the hypotenuse -/
  a : ℝ
  /-- Length of the second segment of the hypotenuse -/
  b : ℝ
  /-- The first leg of the triangle -/
  leg1 : ℝ
  /-- The second leg of the triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The first segment plus radius equals the first leg -/
  h1 : a + r = leg1
  /-- The second segment plus radius equals the second leg -/
  h2 : b + r = leg2
  /-- The Pythagorean theorem holds -/
  pythagoras : leg1^2 + leg2^2 = (a + b)^2

/-- The main theorem -/
theorem right_triangle_legs (t : RightTriangleWithInscribedCircle)
  (ha : t.a = 5) (hb : t.b = 12) : t.leg1 = 8 ∧ t.leg2 = 15 := by
  sorry

end right_triangle_legs_l2377_237785


namespace power_set_of_S_l2377_237793

def S : Set ℕ := {0, 1}

theorem power_set_of_S :
  𝒫 S = {∅, {0}, {1}, {0, 1}} := by
  sorry

end power_set_of_S_l2377_237793


namespace bouquet_39_roses_cost_l2377_237759

/-- Represents the cost of a bouquet of roses -/
structure BouquetCost where
  baseCost : ℝ
  additionalCostPerRose : ℝ

/-- Calculates the total cost of a bouquet given the number of roses -/
def totalCost (bc : BouquetCost) (numRoses : ℕ) : ℝ :=
  bc.baseCost + bc.additionalCostPerRose * numRoses

/-- Theorem: Given the conditions, a bouquet of 39 roses costs $58.75 -/
theorem bouquet_39_roses_cost
  (bc : BouquetCost)
  (h1 : bc.baseCost = 10)
  (h2 : totalCost bc 12 = 25) :
  totalCost bc 39 = 58.75 := by
  sorry

#check bouquet_39_roses_cost

end bouquet_39_roses_cost_l2377_237759


namespace triangle_perimeter_range_l2377_237745

/-- Given a triangle ABC with sides a, b, c, where a = 1 and 2cos(C) + c = 2b,
    the perimeter p satisfies 2 < p ≤ 3 -/
theorem triangle_perimeter_range (b c : ℝ) (C : ℝ) : 
  let a : ℝ := 1
  let p := a + b + c
  2 * Real.cos C + c = 2 * b →
  2 < p ∧ p ≤ 3 := by
sorry

end triangle_perimeter_range_l2377_237745


namespace simplify_expression_l2377_237705

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+4)) = 29 / 48 := by
  sorry

end simplify_expression_l2377_237705


namespace parallel_intersecting_lines_c_is_zero_l2377_237797

/-- Two lines that are parallel and intersect at a specific point -/
structure ParallelIntersectingLines where
  a : ℝ
  b : ℝ
  c : ℝ
  parallel : a / 2 = -2 / b
  intersect_x : 2 * a - 2 * (-4) = c
  intersect_y : 2 * 2 + b * (-4) = c

/-- The theorem stating that for such lines, c must be 0 -/
theorem parallel_intersecting_lines_c_is_zero (lines : ParallelIntersectingLines) : lines.c = 0 := by
  sorry

end parallel_intersecting_lines_c_is_zero_l2377_237797


namespace intersection_X_complement_Y_l2377_237798

def U : Set ℝ := Set.univ

def X : Set ℝ := {x | x^2 - x = 0}

def Y : Set ℝ := {x | x^2 + x = 0}

theorem intersection_X_complement_Y : X ∩ (U \ Y) = {1} := by sorry

end intersection_X_complement_Y_l2377_237798


namespace exam_failure_percentage_l2377_237775

theorem exam_failure_percentage 
  (total_candidates : ℕ) 
  (hindi_failure_rate : ℚ)
  (both_failure_rate : ℚ)
  (english_only_pass : ℕ) :
  total_candidates = 3000 →
  hindi_failure_rate = 36/100 →
  both_failure_rate = 15/100 →
  english_only_pass = 630 →
  ∃ (english_failure_rate : ℚ),
    english_failure_rate = 85/100 ∧
    english_only_pass = total_candidates * ((1 - english_failure_rate) + (hindi_failure_rate - both_failure_rate)) :=
by sorry

end exam_failure_percentage_l2377_237775


namespace imaginary_part_of_z_l2377_237738

theorem imaginary_part_of_z (z : ℂ) (h : 1 + (1 + 2 * z) * Complex.I = 0) :
  z.im = 1 / 2 := by
  sorry

end imaginary_part_of_z_l2377_237738


namespace discount_percentages_l2377_237724

theorem discount_percentages :
  ∃ (x y : ℕ), 0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10 ∧
  69000 * (100 - x) * (100 - y) / 10000 = 60306 ∧
  ((x = 5 ∧ y = 8) ∨ (x = 8 ∧ y = 5)) := by
  sorry

end discount_percentages_l2377_237724


namespace min_value_inequality_l2377_237766

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end min_value_inequality_l2377_237766


namespace shanghai_expo_2010_l2377_237744

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

/-- Calculates the number of days in a year -/
def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

/-- Calculates the day of the week for a given date -/
def dayOfWeek (year month day : Nat) : DayOfWeek := sorry

/-- Calculates the number of days between two dates in the same year -/
def daysBetween (year startMonth startDay endMonth endDay : Nat) : Nat := sorry

theorem shanghai_expo_2010 :
  let year := 2010
  let mayFirst := DayOfWeek.Saturday
  ¬isLeapYear year ∧
  daysInYear year = 365 ∧
  dayOfWeek year 5 31 = DayOfWeek.Monday ∧
  daysBetween year 5 1 10 31 = 184 := by sorry

end shanghai_expo_2010_l2377_237744


namespace profit_share_b_profit_share_b_is_1500_l2377_237743

theorem profit_share_b (capital_a capital_b capital_c : ℕ) 
  (profit_diff_ac : ℚ) (profit_share_b : ℚ) : Prop :=
  capital_a = 8000 ∧ 
  capital_b = 10000 ∧ 
  capital_c = 12000 ∧ 
  profit_diff_ac = 600 ∧
  profit_share_b = 1500 ∧
  ∃ (total_profit : ℚ),
    total_profit * (capital_b : ℚ) / (capital_a + capital_b + capital_c : ℚ) = profit_share_b ∧
    total_profit * (capital_c - capital_a : ℚ) / (capital_a + capital_b + capital_c : ℚ) = profit_diff_ac

-- Proof
theorem profit_share_b_is_1500 : 
  profit_share_b 8000 10000 12000 600 1500 := by
  sorry

end profit_share_b_profit_share_b_is_1500_l2377_237743


namespace max_value_cos_squared_minus_sin_l2377_237736

open Real

theorem max_value_cos_squared_minus_sin (x : ℝ) : 
  ∃ (M : ℝ), M = (5 : ℝ) / 4 ∧ ∀ x, cos x ^ 2 - sin x ≤ M :=
sorry

end max_value_cos_squared_minus_sin_l2377_237736


namespace cube_of_negative_l2377_237771

theorem cube_of_negative (x : ℝ) : (-x)^3 = -x^3 := by
  sorry

end cube_of_negative_l2377_237771


namespace tenfold_largest_two_digit_l2377_237720

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit : 10 * largest_two_digit_number = 990 := by
  sorry

end tenfold_largest_two_digit_l2377_237720


namespace ball_purchase_theorem_l2377_237796

/-- Represents the cost and quantity of balls in two purchases -/
structure BallPurchase where
  soccer_price : ℝ
  volleyball_price : ℝ
  soccer_quantity1 : ℕ
  volleyball_quantity1 : ℕ
  total_cost1 : ℝ
  total_quantity2 : ℕ
  soccer_price_increase : ℝ
  volleyball_price_decrease : ℝ
  total_cost2_ratio : ℝ

/-- Theorem stating the prices of balls and the quantity of volleyballs in the second purchase -/
theorem ball_purchase_theorem (bp : BallPurchase)
  (h1 : bp.soccer_quantity1 * bp.soccer_price + bp.volleyball_quantity1 * bp.volleyball_price = bp.total_cost1)
  (h2 : bp.soccer_price = bp.volleyball_price + 30)
  (h3 : bp.soccer_quantity1 = 40)
  (h4 : bp.volleyball_quantity1 = 30)
  (h5 : bp.total_cost1 = 4000)
  (h6 : bp.total_quantity2 = 50)
  (h7 : bp.soccer_price_increase = 0.1)
  (h8 : bp.volleyball_price_decrease = 0.1)
  (h9 : bp.total_cost2_ratio = 0.86) :
  bp.soccer_price = 70 ∧ bp.volleyball_price = 40 ∧
  ∃ m : ℕ, m = 10 ∧ 
    (bp.total_quantity2 - m) * (bp.soccer_price * (1 + bp.soccer_price_increase)) +
    m * (bp.volleyball_price * (1 - bp.volleyball_price_decrease)) =
    bp.total_cost1 * bp.total_cost2_ratio :=
by sorry

end ball_purchase_theorem_l2377_237796


namespace selene_and_tanya_spend_16_l2377_237767

/-- Represents the prices of items in the school canteen -/
structure CanteenPrices where
  sandwich : ℕ
  hamburger : ℕ
  hotdog : ℕ
  fruitJuice : ℕ

/-- Represents an order in the canteen -/
structure Order where
  sandwiches : ℕ
  hamburgers : ℕ
  hotdogs : ℕ
  fruitJuices : ℕ

/-- Calculates the total cost of an order given the prices -/
def orderCost (prices : CanteenPrices) (order : Order) : ℕ :=
  prices.sandwich * order.sandwiches +
  prices.hamburger * order.hamburgers +
  prices.hotdog * order.hotdogs +
  prices.fruitJuice * order.fruitJuices

/-- The main theorem stating that Selene and Tanya spend $16 together -/
theorem selene_and_tanya_spend_16 (prices : CanteenPrices) 
    (seleneOrder : Order) (tanyaOrder : Order) : 
    prices.sandwich = 2 → 
    prices.hamburger = 2 → 
    prices.hotdog = 1 → 
    prices.fruitJuice = 2 → 
    seleneOrder.sandwiches = 3 → 
    seleneOrder.fruitJuices = 1 → 
    seleneOrder.hamburgers = 0 → 
    seleneOrder.hotdogs = 0 → 
    tanyaOrder.hamburgers = 2 → 
    tanyaOrder.fruitJuices = 2 → 
    tanyaOrder.sandwiches = 0 → 
    tanyaOrder.hotdogs = 0 → 
    orderCost prices seleneOrder + orderCost prices tanyaOrder = 16 := by
  sorry

end selene_and_tanya_spend_16_l2377_237767


namespace bakery_combinations_l2377_237753

/-- The number of ways to distribute n items among k categories, 
    with at least m items in each of the first two categories -/
def distribute (n k m : ℕ) : ℕ :=
  -- We don't provide the implementation, just the type signature
  sorry

/-- The specific case for the bakery problem -/
theorem bakery_combinations : distribute 8 5 2 = 70 := by
  sorry

end bakery_combinations_l2377_237753


namespace debugging_time_l2377_237742

theorem debugging_time (total_hours : ℝ) (flow_chart_frac : ℝ) (coding_frac : ℝ) (meeting_frac : ℝ)
  (h1 : total_hours = 192)
  (h2 : flow_chart_frac = 3 / 10)
  (h3 : coding_frac = 3 / 8)
  (h4 : meeting_frac = 1 / 5)
  (h5 : flow_chart_frac + coding_frac + meeting_frac < 1) :
  total_hours - (flow_chart_frac + coding_frac + meeting_frac) * total_hours = 24 := by
  sorry

end debugging_time_l2377_237742


namespace tony_winnings_l2377_237782

/-- Calculates the winnings for a single lottery ticket -/
def ticket_winnings (winning_numbers : ℕ) : ℕ :=
  if winning_numbers ≤ 2 then
    15 * winning_numbers
  else
    30 + 20 * (winning_numbers - 2)

/-- Represents Tony's lottery tickets and calculates total winnings -/
def total_winnings : ℕ :=
  ticket_winnings 3 + ticket_winnings 5 + ticket_winnings 2 + ticket_winnings 4

/-- Theorem stating that Tony's total winnings are $240 -/
theorem tony_winnings : total_winnings = 240 := by
  sorry

end tony_winnings_l2377_237782


namespace toy_cost_l2377_237721

/-- The cost of each toy given Paul's savings and allowance -/
theorem toy_cost (initial_savings : ℕ) (allowance : ℕ) (num_toys : ℕ) 
  (h1 : initial_savings = 3)
  (h2 : allowance = 7)
  (h3 : num_toys = 2)
  (h4 : num_toys > 0) :
  (initial_savings + allowance) / num_toys = 5 := by
  sorry


end toy_cost_l2377_237721


namespace four_digit_odd_divisible_by_digits_l2377_237792

def is_odd (n : Nat) : Prop := ∃ k, n = 2 * k + 1

def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_of (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem four_digit_odd_divisible_by_digits :
  ∀ n : Nat,
  is_four_digit n →
  (let d := digits_of n
   d.length = 4 ∧
   (∀ x ∈ d, is_odd x) ∧
   d.toFinset.card = 4 ∧
   (∀ x ∈ d, n % x = 0)) →
  n ∈ [1395, 1935, 3195, 3915, 9135, 9315] := by
sorry

end four_digit_odd_divisible_by_digits_l2377_237792


namespace min_value_and_max_product_l2377_237741

def f (x : ℝ) : ℝ := 2 * abs (x + 1) - abs (x - 1)

theorem min_value_and_max_product :
  (∃ k : ℝ, ∀ x : ℝ, f x ≥ k ∧ ∃ x₀ : ℝ, f x₀ = k) ∧
  (∀ a b c : ℝ, a^2 + c^2 + b^2/2 = 2 → b*(a+c) ≤ 2) ∧
  (∃ a b c : ℝ, a^2 + c^2 + b^2/2 = 2 ∧ b*(a+c) = 2) :=
by sorry

end min_value_and_max_product_l2377_237741


namespace quadratic_intersection_l2377_237706

/-- 
Given two functions f(x) = bx² + 5x + 2 and g(x) = -2x - 2,
prove that they intersect at exactly one point when b = 49/16.
-/
theorem quadratic_intersection (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 5 * x + 2 = -2 * x - 2) ↔ b = 49/16 := by
sorry

end quadratic_intersection_l2377_237706


namespace sin_2alpha_value_l2377_237799

theorem sin_2alpha_value (α : Real) (h : Real.sin α - Real.cos α = 1/5) : 
  Real.sin (2 * α) = 24/25 := by
  sorry

end sin_2alpha_value_l2377_237799


namespace point_and_tangent_line_l2377_237770

def f (a t x : ℝ) : ℝ := x^3 + a*x
def g (b c t x : ℝ) : ℝ := b*x^2 + c
def h (a b c t x : ℝ) : ℝ := f a t x - g b c t x

theorem point_and_tangent_line (t : ℝ) (h_t : t ≠ 0) :
  ∃ (a b c : ℝ),
    (f a t t = 0) ∧
    (g b c t t = 0) ∧
    (∀ x, (deriv (f a t)) x = (deriv (g b c t)) x) ∧
    (∀ x ∈ Set.Ioo (-1) 3, StrictMonoOn (h a b c t) (Set.Ioo (-1) 3)) →
    (a = -t^2 ∧ b = t ∧ c = -t^3 ∧ (t ≤ -9 ∨ t ≥ 3)) :=
by sorry

end point_and_tangent_line_l2377_237770


namespace wedding_couples_theorem_l2377_237712

/-- The number of couples invited by the bride and groom to their wedding reception --/
def couples_invited (total_guests : ℕ) (friends : ℕ) : ℕ :=
  (total_guests - friends) / 2

theorem wedding_couples_theorem (total_guests : ℕ) (friends : ℕ) 
  (h1 : total_guests = 180) 
  (h2 : friends = 100) :
  couples_invited total_guests friends = 40 := by
  sorry

end wedding_couples_theorem_l2377_237712


namespace probability_no_shaded_l2377_237780

/-- Represents a rectangle in the 2 by 1001 grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The total number of possible rectangles in the grid --/
def total_rectangles : Nat := 501501

/-- The number of rectangles containing at least one shaded square --/
def shaded_rectangles : Nat := 252002

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left = 1 ∧ r.right ≥ 1) ∨ 
  (r.left ≤ 501 ∧ r.right ≥ 501) ∨ 
  (r.left ≤ 1001 ∧ r.right = 1001)

/-- The main theorem stating the probability of choosing a rectangle without a shaded square --/
theorem probability_no_shaded : 
  (total_rectangles - shaded_rectangles) / total_rectangles = 249499 / 501501 := by
  sorry

end probability_no_shaded_l2377_237780


namespace max_area_rectangular_pen_l2377_237731

/-- Given a rectangular pen with perimeter 60 feet and one side length at least 15 feet,
    the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen :
  ∀ (x y : ℝ),
    x > 0 ∧ y > 0 →
    x + y = 30 →
    (x ≥ 15 ∨ y ≥ 15) →
    x * y ≤ 225 :=
by sorry

end max_area_rectangular_pen_l2377_237731


namespace largest_two_digit_prime_factor_of_binomial_200_100_l2377_237763

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := sorry

/-- The largest 2-digit prime factor of (200 choose 100) -/
def largestTwoDigitPrimeFactor : ℕ := 61

theorem largest_two_digit_prime_factor_of_binomial_200_100 :
  ∀ p : ℕ, 
    10 ≤ p → p < 100 → isPrime p → 
    p ∣ binomial 200 100 →
    p ≤ largestTwoDigitPrimeFactor ∧
    isPrime largestTwoDigitPrimeFactor ∧
    largestTwoDigitPrimeFactor ∣ binomial 200 100 := by
  sorry

end largest_two_digit_prime_factor_of_binomial_200_100_l2377_237763


namespace gumball_machine_total_l2377_237751

/-- Represents the number of gumballs of each color in a gumball machine. -/
structure GumballMachine where
  red : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ
  orange : ℕ

/-- Represents the conditions of the gumball machine problem. -/
def gumball_machine_conditions (m : GumballMachine) : Prop :=
  m.blue = m.red / 2 ∧
  m.green = 4 * m.blue ∧
  m.yellow = (7 * m.blue) / 2 ∧
  m.orange = (2 * (m.red + m.blue)) / 3 ∧
  m.red = (3 * m.yellow) / 2 ∧
  m.yellow = 24

/-- The theorem stating that a gumball machine satisfying the given conditions has 186 gumballs. -/
theorem gumball_machine_total (m : GumballMachine) 
  (h : gumball_machine_conditions m) : 
  m.red + m.green + m.blue + m.yellow + m.orange = 186 := by
  sorry


end gumball_machine_total_l2377_237751


namespace percent_of_y_l2377_237760

theorem percent_of_y (y : ℝ) (h : y > 0) : ((6 * y) / 20 + (3 * y) / 10) / y = 0.6 := by
  sorry

end percent_of_y_l2377_237760


namespace two_lines_perpendicular_to_plane_are_parallel_two_planes_perpendicular_to_line_are_parallel_l2377_237726

-- Define the basic geometric objects
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relationships
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Statement ②
theorem two_lines_perpendicular_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) :
  perpendicular l1 p → perpendicular l2 p → parallel l1 l2 := by sorry

-- Statement ③
theorem two_planes_perpendicular_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) :
  perpendicular_plane p1 l → perpendicular_plane p2 l → parallel_plane p1 p2 := by sorry

end two_lines_perpendicular_to_plane_are_parallel_two_planes_perpendicular_to_line_are_parallel_l2377_237726


namespace book_price_reduction_l2377_237723

theorem book_price_reduction : 
  let initial_discount : ℝ := 0.3
  let price_increase : ℝ := 0.2
  let final_discount : ℝ := 0.5
  let original_price : ℝ := 1
  let discounted_price := original_price * (1 - initial_discount)
  let increased_price := discounted_price * (1 + price_increase)
  let final_price := increased_price * (1 - final_discount)
  let total_reduction := (original_price - final_price) / original_price
  total_reduction = 0.58
:= by sorry

end book_price_reduction_l2377_237723


namespace drink_ticket_cost_l2377_237701

/-- Proves that the cost of each drink ticket is $7 given Jenna's income and spending constraints -/
theorem drink_ticket_cost 
  (concert_ticket_cost : ℕ)
  (hourly_wage : ℕ)
  (weekly_hours : ℕ)
  (spending_percentage : ℚ)
  (num_drink_tickets : ℕ)
  (h1 : concert_ticket_cost = 181)
  (h2 : hourly_wage = 18)
  (h3 : weekly_hours = 30)
  (h4 : spending_percentage = 1/10)
  (h5 : num_drink_tickets = 5) :
  (((hourly_wage * weekly_hours * 4) * spending_percentage - concert_ticket_cost : ℚ) / num_drink_tickets : ℚ) = 7 := by
  sorry

end drink_ticket_cost_l2377_237701


namespace joey_study_time_l2377_237709

/-- Calculates the total study time for Joey's SAT exam preparation --/
theorem joey_study_time (weekday_hours : ℕ) (weekday_nights : ℕ) (weekend_hours : ℕ) (weekend_days : ℕ) (weeks : ℕ) : 
  weekday_hours = 2 →
  weekday_nights = 5 →
  weekend_hours = 3 →
  weekend_days = 2 →
  weeks = 6 →
  (weekday_hours * weekday_nights + weekend_hours * weekend_days) * weeks = 96 := by
  sorry

#check joey_study_time

end joey_study_time_l2377_237709


namespace calculate_expression_l2377_237788

theorem calculate_expression : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end calculate_expression_l2377_237788


namespace rectangle_area_error_percent_l2377_237730

/-- Given a rectangle with sides measured with errors, calculate the error percent in the area --/
theorem rectangle_area_error_percent (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percent := (error / actual_area) * 100
  error_percent = 0.8 := by
  sorry

end rectangle_area_error_percent_l2377_237730


namespace angle_triple_complement_l2377_237718

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end angle_triple_complement_l2377_237718


namespace same_color_probability_is_five_eighteenths_l2377_237713

/-- Represents the number of jelly beans of each color that Abe has -/
structure AbeJellyBeans where
  green : Nat
  blue : Nat

/-- Represents the number of jelly beans of each color that Bob has -/
structure BobJellyBeans where
  green : Nat
  blue : Nat
  red : Nat

/-- Calculates the probability of both Abe and Bob showing the same color jelly bean -/
def probability_same_color (abe : AbeJellyBeans) (bob : BobJellyBeans) : Rat :=
  sorry

/-- The main theorem stating the probability of Abe and Bob showing the same color jelly bean -/
theorem same_color_probability_is_five_eighteenths 
  (abe : AbeJellyBeans) 
  (bob : BobJellyBeans) 
  (h1 : abe.green = 2)
  (h2 : abe.blue = 1)
  (h3 : bob.green = 2)
  (h4 : bob.blue = 1)
  (h5 : bob.red = 3) :
  probability_same_color abe bob = 5 / 18 := by
  sorry

end same_color_probability_is_five_eighteenths_l2377_237713


namespace yuan_equality_l2377_237702

theorem yuan_equality : (3.00 : ℝ) = (3 : ℝ) := by
  sorry

end yuan_equality_l2377_237702


namespace batsman_highest_score_l2377_237714

theorem batsman_highest_score 
  (total_innings : ℕ)
  (average : ℚ)
  (score_difference : ℕ)
  (average_excluding_extremes : ℚ)
  (h : total_innings = 46)
  (h1 : average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * average = 
      ((total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score) ∧
    highest_score = 202 := by
  sorry

end batsman_highest_score_l2377_237714


namespace bella_needs_twelve_beads_l2377_237783

/-- Given the number of friends, beads per bracelet, and beads on hand,
    calculate the number of additional beads needed. -/
def additional_beads_needed (friends : ℕ) (beads_per_bracelet : ℕ) (beads_on_hand : ℕ) : ℕ :=
  max 0 (friends * beads_per_bracelet - beads_on_hand)

/-- Proof that Bella needs 12 more beads to make bracelets for her friends. -/
theorem bella_needs_twelve_beads :
  additional_beads_needed 6 8 36 = 12 := by
  sorry

end bella_needs_twelve_beads_l2377_237783


namespace tim_weekly_earnings_l2377_237746

/-- Tim's daily task count -/
def daily_tasks : ℕ := 100

/-- Tim's working days per week -/
def working_days : ℕ := 6

/-- Number of tasks paying $1.2 each -/
def tasks_1_2 : ℕ := 40

/-- Number of tasks paying $1.5 each -/
def tasks_1_5 : ℕ := 30

/-- Number of tasks paying $2 each -/
def tasks_2 : ℕ := 30

/-- Payment rate for the first group of tasks -/
def rate_1_2 : ℚ := 1.2

/-- Payment rate for the second group of tasks -/
def rate_1_5 : ℚ := 1.5

/-- Payment rate for the third group of tasks -/
def rate_2 : ℚ := 2

/-- Tim's weekly earnings -/
def weekly_earnings : ℚ := 918

theorem tim_weekly_earnings :
  daily_tasks = tasks_1_2 + tasks_1_5 + tasks_2 →
  working_days * (tasks_1_2 * rate_1_2 + tasks_1_5 * rate_1_5 + tasks_2 * rate_2) = weekly_earnings :=
by sorry

end tim_weekly_earnings_l2377_237746


namespace geometric_sequence_constant_l2377_237722

/-- A geometric sequence with sum S_n = 3 · 2^n + k -/
def geometric_sequence (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ+, S n = 3 * 2^(n : ℝ) + k

theorem geometric_sequence_constant (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  geometric_sequence a S (-3) →
  (∀ n : ℕ+, a n = S n - S (n - 1)) →
  a 1 = 3 :=
by sorry

end geometric_sequence_constant_l2377_237722


namespace joseph_decks_l2377_237784

/-- The number of complete decks given a total number of cards and cards per deck -/
def number_of_decks (total_cards : ℕ) (cards_per_deck : ℕ) : ℕ :=
  total_cards / cards_per_deck

/-- Proof that Joseph has 4 complete decks of cards -/
theorem joseph_decks :
  number_of_decks 208 52 = 4 := by
  sorry

end joseph_decks_l2377_237784


namespace complex_power_sum_l2377_237752

/-- If z is a complex number satisfying z + 1/z = 2 cos 5°, then z^1500 + 1/z^1500 = 1 -/
theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + 1/z^1500 = 1 := by
  sorry

end complex_power_sum_l2377_237752


namespace relationship_equation_l2377_237756

/-- Given a relationship "a number that is 3 more than half of x is equal to twice y",
    prove that the equation (1/2)x + 3 = 2y correctly represents this relationship. -/
theorem relationship_equation (x y : ℝ) :
  (∃ (n : ℝ), n = (1/2) * x + 3 ∧ n = 2 * y) ↔ (1/2) * x + 3 = 2 * y :=
by sorry

end relationship_equation_l2377_237756


namespace minimum_point_of_transformed_graph_l2377_237732

-- Define the function representing the transformed graph
def f (x : ℝ) : ℝ := 2 * abs (x + 3) - 7

-- State the theorem
theorem minimum_point_of_transformed_graph :
  ∃ (x : ℝ), f x = f (-3) ∧ ∀ (y : ℝ), f y ≥ f (-3) ∧ f (-3) = -7 :=
sorry

end minimum_point_of_transformed_graph_l2377_237732


namespace geometric_sequence_property_l2377_237774

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property
  (b : ℕ → ℝ)
  (h_geometric : geometric_sequence b)
  (h_b1 : b 1 = 1)
  (s t : ℕ)
  (h_distinct : s ≠ t)
  (h_positive : s > 0 ∧ t > 0) :
  (b t) ^ (s - 1) / (b s) ^ (t - 1) = 1 := by
  sorry

end geometric_sequence_property_l2377_237774


namespace age_difference_l2377_237765

theorem age_difference (age1 age2 : ℕ) : 
  age1 + age2 = 27 → age1 = 13 → age2 = 14 → age2 - age1 = 1 := by
  sorry

end age_difference_l2377_237765


namespace gcd_of_specific_numbers_l2377_237719

theorem gcd_of_specific_numbers : Nat.gcd 333333333 666666666 = 333333333 := by
  sorry

end gcd_of_specific_numbers_l2377_237719


namespace acid_mixture_concentration_exists_l2377_237779

theorem acid_mixture_concentration_exists :
  ∃! P : ℝ, ∃ a w : ℝ,
    a > 0 ∧ w > 0 ∧
    (a / (a + w + 2)) * 100 = 30 ∧
    ((a + 1) / (a + w + 3)) * 100 = 40 ∧
    (a / (a + w)) * 100 = P ∧
    (P = 50 ∨ P = 52 ∨ P = 55 ∨ P = 57 ∨ P = 60) :=
by sorry

end acid_mixture_concentration_exists_l2377_237779


namespace complex_arithmetic_result_l2377_237768

def A : ℂ := 3 - 4 * Complex.I
def M : ℂ := -3 + 2 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := -1

theorem complex_arithmetic_result : A - M + S + P = 5 - 4 * Complex.I := by
  sorry

end complex_arithmetic_result_l2377_237768


namespace bus_driver_rate_l2377_237776

/-- Represents the bus driver's compensation structure and work details -/
structure BusDriverCompensation where
  regularHours : ℕ := 40
  totalHours : ℕ
  overtimeMultiplier : ℚ
  totalCompensation : ℚ

/-- Calculates the regular hourly rate given the compensation structure -/
def calculateRegularRate (bdc : BusDriverCompensation) : ℚ :=
  let overtimeHours := bdc.totalHours - bdc.regularHours
  bdc.totalCompensation / (bdc.regularHours + overtimeHours * bdc.overtimeMultiplier)

/-- Theorem stating that the bus driver's regular rate is $16 per hour -/
theorem bus_driver_rate : 
  let bdc : BusDriverCompensation := {
    totalHours := 65,
    overtimeMultiplier := 1.75,
    totalCompensation := 1340
  }
  calculateRegularRate bdc = 16 := by sorry

end bus_driver_rate_l2377_237776


namespace two_days_saved_l2377_237728

/-- Represents the work scenario with original and additional workers --/
structure WorkScenario where
  originalMen : ℕ
  originalDays : ℕ
  additionalMen : ℕ
  totalWork : ℕ

/-- Calculates the number of days saved when additional workers join --/
def daysSaved (w : WorkScenario) : ℕ :=
  w.originalDays - (w.totalWork / (w.originalMen + w.additionalMen))

/-- Theorem stating that in the given scenario, 2 days are saved --/
theorem two_days_saved (w : WorkScenario) 
  (h1 : w.originalMen = 30)
  (h2 : w.originalDays = 8)
  (h3 : w.additionalMen = 10)
  (h4 : w.totalWork = w.originalMen * w.originalDays) :
  daysSaved w = 2 := by
  sorry

#eval daysSaved { originalMen := 30, originalDays := 8, additionalMen := 10, totalWork := 240 }

end two_days_saved_l2377_237728


namespace parabola_directrix_l2377_237762

/-- The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), x = -(1/4) * y^2 → 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (p : ℝ × ℝ), p.1 = -(1/4) * p.2^2 → 
  (p.1 - d)^2 = (p.1 - (-d))^2 + p.2^2 := by
sorry

end parabola_directrix_l2377_237762


namespace temperature_difference_l2377_237795

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 11) 
  (h2 : lowest = -11) : 
  highest - lowest = 22 := by
  sorry

end temperature_difference_l2377_237795


namespace pen_ratio_is_one_l2377_237786

theorem pen_ratio_is_one (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 25)
  (h2 : mike_pens = 22)
  (h3 : sharon_pens = 19)
  (h4 : final_pens = 75) :
  (final_pens + sharon_pens - (initial_pens + mike_pens)) / (initial_pens + mike_pens) = 1 :=
by
  sorry

end pen_ratio_is_one_l2377_237786


namespace unique_four_digit_prime_product_l2377_237787

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_four_digit_prime_product :
  ∃! n : ℕ,
    1000 ≤ n ∧ n ≤ 9999 ∧
    ∃ (p q r s : ℕ),
      is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
      p < q ∧ q < r ∧
      n = p * q * r ∧
      p + q = r - q ∧
      p + q + r = s^2 ∧
      n = 2015 := by
  sorry

end unique_four_digit_prime_product_l2377_237787


namespace abs_five_minus_e_l2377_237703

theorem abs_five_minus_e (e : ℝ) (h : e < 5) : |5 - e| = 5 - e := by sorry

end abs_five_minus_e_l2377_237703


namespace b_equals_one_b_non_negative_l2377_237700

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem 1
theorem b_equals_one (b c : ℝ) :
  c = -3 →
  quadratic 2 b c (-1) = -2 →
  b = 1 := by sorry

-- Theorem 2
theorem b_non_negative (b c p : ℝ) :
  b + c = -2 →
  b > c →
  quadratic 2 b c p = -2 →
  b ≥ 0 := by sorry

end b_equals_one_b_non_negative_l2377_237700


namespace min_sum_three_integers_l2377_237734

theorem min_sum_three_integers (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃ (k₁ k₂ k₃ : ℕ), 
    (1 / a + 1 / b : ℚ) = k₁ * (1 / c : ℚ) ∧
    (1 / a + 1 / c : ℚ) = k₂ * (1 / b : ℚ) ∧
    (1 / b + 1 / c : ℚ) = k₃ * (1 / a : ℚ)) →
  a + b + c ≥ 11 :=
sorry

end min_sum_three_integers_l2377_237734


namespace leas_purchases_total_cost_l2377_237725

/-- The total cost of Léa's purchases is $28, given that she bought one book for $16, 
    three binders for $2 each, and six notebooks for $1 each. -/
theorem leas_purchases_total_cost : 
  let book_cost : ℕ := 16
  let binder_cost : ℕ := 2
  let notebook_cost : ℕ := 1
  let num_binders : ℕ := 3
  let num_notebooks : ℕ := 6
  book_cost + num_binders * binder_cost + num_notebooks * notebook_cost = 28 :=
by sorry

end leas_purchases_total_cost_l2377_237725


namespace at_least_thirty_percent_have_all_colors_l2377_237778

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  total_children : ℕ
  blue_percentage : ℚ
  red_percentage : ℚ
  green_percentage : ℚ

/-- Conditions for the flag distribution problem -/
def valid_distribution (d : FlagDistribution) : Prop :=
  d.blue_percentage = 55 / 100 ∧
  d.red_percentage = 45 / 100 ∧
  d.green_percentage = 30 / 100 ∧
  (d.total_children * 3) % 2 = 0 ∧
  d.blue_percentage + d.red_percentage + d.green_percentage ≥ 1

/-- The main theorem stating that at least 30% of children have all three colors -/
theorem at_least_thirty_percent_have_all_colors (d : FlagDistribution) 
  (h : valid_distribution d) : 
  ∃ (all_colors_percentage : ℚ), 
    all_colors_percentage ≥ 30 / 100 ∧ 
    all_colors_percentage ≤ d.blue_percentage ∧
    all_colors_percentage ≤ d.red_percentage ∧
    all_colors_percentage ≤ d.green_percentage :=
sorry

end at_least_thirty_percent_have_all_colors_l2377_237778


namespace pet_shop_total_cost_l2377_237757

/-- Calculates the total cost of purchasing all pets with discounts -/
def total_cost_with_discounts (puppy1_price puppy2_price kitten1_price kitten2_price 
                               parakeet1_price parakeet2_price parakeet3_price : ℚ) : ℚ :=
  let puppy_total := puppy1_price + puppy2_price
  let puppy_discount := puppy_total * (5 / 100)
  let puppy_cost := puppy_total - puppy_discount

  let kitten_total := kitten1_price + kitten2_price
  let kitten_discount := kitten_total * (10 / 100)
  let kitten_cost := kitten_total - kitten_discount

  let parakeet_total := parakeet1_price + parakeet2_price + parakeet3_price
  let parakeet_discount := min parakeet1_price (min parakeet2_price parakeet3_price) / 2
  let parakeet_cost := parakeet_total - parakeet_discount

  puppy_cost + kitten_cost + parakeet_cost

/-- The theorem stating the total cost of purchasing all pets with discounts -/
theorem pet_shop_total_cost :
  total_cost_with_discounts 72 78 48 52 10 12 14 = 263.5 := by
  sorry

end pet_shop_total_cost_l2377_237757


namespace lcm_factor_is_one_l2377_237748

/-- Given two positive integers with specific properties, prove that a certain factor of their LCM is 1. -/
theorem lcm_factor_is_one (A B : ℕ+) (X : ℕ) 
  (hcf : Nat.gcd A B = 10)
  (a_val : A = 150)
  (lcm_fact : Nat.lcm A B = 10 * X * 15) : X = 1 := by
  sorry

end lcm_factor_is_one_l2377_237748


namespace union_A_complement_B_l2377_237772

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Theorem statement
theorem union_A_complement_B : A ∪ (U \ B) = {x | x < 2} := by sorry

end union_A_complement_B_l2377_237772


namespace complex_fraction_simplification_l2377_237758

theorem complex_fraction_simplification :
  (5 + 12 * Complex.I) / (2 - 3 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end complex_fraction_simplification_l2377_237758


namespace fifteen_fishers_tomorrow_l2377_237777

/-- Represents the fishing schedule in a coastal village -/
structure FishingSchedule where
  daily : ℕ              -- Number of people fishing daily
  everyOtherDay : ℕ      -- Number of people fishing every other day
  everyThreeDay : ℕ      -- Number of people fishing every three days
  yesterday : ℕ          -- Number of people who fished yesterday
  today : ℕ              -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow given a FishingSchedule -/
def tomorrowFishers (schedule : FishingSchedule) : ℕ :=
  schedule.daily +
  schedule.everyThreeDay +
  (schedule.everyOtherDay - (schedule.yesterday - schedule.daily))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow 
  (schedule : FishingSchedule)
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { daily := 7, everyOtherDay := 8, everyThreeDay := 3, yesterday := 12, today := 10 }

end fifteen_fishers_tomorrow_l2377_237777


namespace tan_seventeen_pi_fourths_l2377_237707

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end tan_seventeen_pi_fourths_l2377_237707


namespace sarah_initial_money_l2377_237761

def toy_car_price : ℕ := 11
def toy_car_quantity : ℕ := 2
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7

theorem sarah_initial_money :
  ∃ (initial_money : ℕ),
    initial_money = 
      remaining_money + beanie_price + scarf_price + (toy_car_price * toy_car_quantity) ∧
    initial_money = 53 := by
  sorry

end sarah_initial_money_l2377_237761


namespace shanmukham_purchase_l2377_237781

/-- Calculates the final amount to pay for goods given the original price, rebate percentage, and sales tax percentage. -/
def finalAmount (originalPrice rebatePercentage salesTaxPercentage : ℚ) : ℚ :=
  let priceAfterRebate := originalPrice * (1 - rebatePercentage / 100)
  let salesTax := priceAfterRebate * (salesTaxPercentage / 100)
  priceAfterRebate + salesTax

/-- Theorem stating that given the specific conditions, the final amount to pay is 6876.10 -/
theorem shanmukham_purchase :
  finalAmount 6650 6 10 = 6876.1 := by
  sorry

end shanmukham_purchase_l2377_237781


namespace average_rainfall_leap_year_february_l2377_237747

/-- Calculates the average rainfall per hour in February of a leap year -/
theorem average_rainfall_leap_year_february (total_rainfall : ℝ) :
  total_rainfall = 420 →
  (35 : ℝ) / 58 = total_rainfall / (29 * 24) := by
  sorry

end average_rainfall_leap_year_february_l2377_237747


namespace correct_num_friends_l2377_237710

/-- The number of friends Jeremie is going with to the amusement park. -/
def num_friends : ℕ := 3

/-- The cost of a ticket in dollars. -/
def ticket_cost : ℕ := 18

/-- The cost of a snack set in dollars. -/
def snack_cost : ℕ := 5

/-- The total cost for Jeremie and her friends in dollars. -/
def total_cost : ℕ := 92

/-- Theorem stating that the number of friends Jeremie is going with is correct. -/
theorem correct_num_friends :
  (num_friends + 1) * (ticket_cost + snack_cost) = total_cost :=
by sorry

end correct_num_friends_l2377_237710


namespace power_function_m_values_l2377_237749

/-- A function is a power function if it's of the form f(x) = ax^n, where a ≠ 0 and n is a real number -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The given function f(x) = (m^2 - m - 1)x^3 -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * x^3

/-- Theorem: If f(x) = (m^2 - m - 1)x^3 is a power function, then m = -1 or m = 2 -/
theorem power_function_m_values (m : ℝ) : IsPowerFunction (f m) → m = -1 ∨ m = 2 := by
  sorry


end power_function_m_values_l2377_237749


namespace sum_of_coefficients_l2377_237794

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₂ + a₄ = 41 := by
  sorry

end sum_of_coefficients_l2377_237794


namespace car_distance_theorem_l2377_237704

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours -/
def totalDistance (initialDistance : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initialDistance + (hours - 1) * speedIncrease) / 2

theorem car_distance_theorem :
  totalDistance 55 2 12 = 792 := by
  sorry

end car_distance_theorem_l2377_237704


namespace odd_function_solution_set_l2377_237773

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 2 = 0) ∧ 
  (∀ x > 0, x * (deriv f x) - f x < 0)

/-- The solution set for f(x)/x > 0 given the conditions on f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2}

theorem odd_function_solution_set (f : ℝ → ℝ) (hf : OddFunction f) :
  {x : ℝ | f x / x > 0} = SolutionSet f :=
sorry

end odd_function_solution_set_l2377_237773


namespace roots_of_equation_l2377_237791

theorem roots_of_equation (x : ℝ) :
  (x^2 - 5*x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 := by
sorry

end roots_of_equation_l2377_237791


namespace initial_men_count_l2377_237715

/-- Represents the initial number of men -/
def initialMen : ℕ := 200

/-- Represents the initial food duration in days -/
def initialDuration : ℕ := 20

/-- Represents the number of days after which some men leave -/
def daysBeforeLeaving : ℕ := 15

/-- Represents the number of men who leave -/
def menWhoLeave : ℕ := 100

/-- Represents the remaining food duration after some men leave -/
def remainingDuration : ℕ := 10

theorem initial_men_count :
  initialMen * daysBeforeLeaving = (initialMen - menWhoLeave) * remainingDuration ∧
  initialMen * initialDuration = initialMen * daysBeforeLeaving + (initialMen - menWhoLeave) * remainingDuration :=
by sorry

end initial_men_count_l2377_237715


namespace lucy_doll_collection_l2377_237727

/-- Represents Lucy's doll collection problem -/
theorem lucy_doll_collection (X : ℕ) (Z : ℕ) : 
  (X : ℚ) * (1 + 1/5) = X + 5 → -- 20% increase after adding 5 dolls
  Z = (X + 5 + (X + 5) / 10 : ℚ).floor → -- 10% more dolls from updated collection
  X = 25 ∧ Z = 33 := by
  sorry

end lucy_doll_collection_l2377_237727
