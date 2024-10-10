import Mathlib

namespace equation_unique_solution_l2691_269178

theorem equation_unique_solution :
  ∃! x : ℝ, Real.sqrt (3 + Real.sqrt (4 + Real.sqrt x)) = (3 + Real.sqrt x) ^ (1/3) ∧ x = 576 := by
  sorry

end equation_unique_solution_l2691_269178


namespace equation_root_l2691_269122

theorem equation_root (a b c d x : ℝ) 
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) ↔ x = 1007.5 := by
sorry

end equation_root_l2691_269122


namespace positive_difference_of_solutions_l2691_269194

theorem positive_difference_of_solutions : ∃ (x₁ x₂ : ℝ), 
  (|2 * x₁ - 3| = 15 ∧ |2 * x₂ - 3| = 15) ∧ |x₁ - x₂| = 15 := by
  sorry

end positive_difference_of_solutions_l2691_269194


namespace hyperbola_eccentricity_l2691_269142

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (c x y : ℝ),
    (x^2 / a^2 - y^2 / b^2 = 1) ∧  -- P is on the hyperbola
    (x = c) ∧  -- PF is perpendicular to x-axis
    ((c - b) / (c + b) = 1/3) →  -- ratio of distances to asymptotes
    c^2 / a^2 = 4/3 :=
by sorry

end hyperbola_eccentricity_l2691_269142


namespace greatest_three_digit_divisible_by_8_ending_4_l2691_269130

theorem greatest_three_digit_divisible_by_8_ending_4 : ∃ n : ℕ, 
  n = 984 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 8 = 0 ∧ 
  n % 10 = 4 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ m % 10 = 4 → m ≤ n :=
by sorry

end greatest_three_digit_divisible_by_8_ending_4_l2691_269130


namespace intersection_of_M_and_N_l2691_269100

open Set Real

theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x^2 < 3*x}
  let N : Set ℝ := {x | log x < 0}
  M ∩ N = Ioo 0 1 := by
  sorry

end intersection_of_M_and_N_l2691_269100


namespace total_jeans_purchased_l2691_269123

-- Define the regular prices and quantities
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def fox_quantity : ℕ := 3
def pony_quantity : ℕ := 2

-- Define the total savings and discount rates
def total_savings : ℝ := 8.55
def total_discount_rate : ℝ := 0.22
def pony_discount_rate : ℝ := 0.15

-- Define the theorem
theorem total_jeans_purchased :
  fox_quantity + pony_quantity = 5 := by sorry

end total_jeans_purchased_l2691_269123


namespace least_N_mod_1000_l2691_269171

/-- Sum of digits in base-five representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-seven representation -/
def g (n : ℕ) : ℕ := sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ := sorry

theorem least_N_mod_1000 : N % 1000 = 781 := by sorry

end least_N_mod_1000_l2691_269171


namespace invisible_square_exists_l2691_269193

/-- A point (x, y) is invisible if gcd(x, y) > 1 -/
def invisible (x y : ℤ) : Prop := Nat.gcd x.natAbs y.natAbs > 1

/-- For any natural number L, there exists integers a and b such that
    for all integers i and j where 0 ≤ i, j ≤ L, the point (a+i, b+j) is invisible -/
theorem invisible_square_exists (L : ℕ) :
  ∃ a b : ℤ, ∀ i j : ℤ, 0 ≤ i ∧ i ≤ L ∧ 0 ≤ j ∧ j ≤ L →
    invisible (a + i) (b + j) := by
  sorry

end invisible_square_exists_l2691_269193


namespace ink_needed_per_whiteboard_l2691_269143

-- Define the given conditions
def num_classes : ℕ := 5
def whiteboards_per_class : ℕ := 2
def ink_cost_per_ml : ℚ := 50 / 100  -- 50 cents = 0.5 dollars
def total_daily_cost : ℚ := 100

-- Define the function to calculate ink needed per whiteboard
def ink_per_whiteboard : ℚ :=
  let total_whiteboards : ℕ := num_classes * whiteboards_per_class
  let total_ink_ml : ℚ := total_daily_cost / ink_cost_per_ml
  total_ink_ml / total_whiteboards

-- Theorem to prove
theorem ink_needed_per_whiteboard : ink_per_whiteboard = 20 := by
  sorry

end ink_needed_per_whiteboard_l2691_269143


namespace parallel_lines_a_value_l2691_269174

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₂ (a x y : ℝ) : Prop := 2*x - a*y + 3 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ x y → l₂ a x y → x = x

-- Theorem statement
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel a → a = -4 := by sorry

end parallel_lines_a_value_l2691_269174


namespace shirt_cost_calculation_l2691_269137

theorem shirt_cost_calculation :
  let discounted_shirts := 3
  let discounted_shirt_price := 15
  let first_discount := 0.1
  let second_discount := 0.05
  let taxed_shirts := 2
  let taxed_shirt_price := 20
  let first_tax := 0.05
  let second_tax := 0.03

  let discounted_price := discounted_shirt_price * (1 - first_discount) * (1 - second_discount)
  let taxed_price := taxed_shirt_price * (1 + first_tax) * (1 + second_tax)

  let total_cost := discounted_shirts * discounted_price + taxed_shirts * taxed_price

  total_cost = 81.735 := by sorry

end shirt_cost_calculation_l2691_269137


namespace diamond_equation_solution_l2691_269111

/-- Diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a ◇ 4 = 21, then a = 53/3 -/
theorem diamond_equation_solution :
  ∀ a : ℝ, diamond a 4 = 21 → a = 53/3 := by
  sorry

end diamond_equation_solution_l2691_269111


namespace equation_solution_l2691_269117

theorem equation_solution : ∃ x : ℝ, (6000 - (105 / x) = 5995) ∧ x = 21 := by sorry

end equation_solution_l2691_269117


namespace total_cards_after_addition_l2691_269163

theorem total_cards_after_addition (initial_playing_cards initial_id_cards additional_playing_cards additional_id_cards : ℕ) :
  initial_playing_cards = 9 →
  initial_id_cards = 4 →
  additional_playing_cards = 6 →
  additional_id_cards = 3 →
  initial_playing_cards + initial_id_cards + additional_playing_cards + additional_id_cards = 22 :=
by
  sorry

end total_cards_after_addition_l2691_269163


namespace proposition_p_q_equivalence_l2691_269155

theorem proposition_p_q_equivalence (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∧
  (∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0) ↔
  2 < m ∧ m < 3 :=
sorry

end proposition_p_q_equivalence_l2691_269155


namespace move_point_left_l2691_269159

/-- Given a point A in a 2D Cartesian coordinate system, moving it
    3 units to the left results in a new point A' -/
def move_left (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - 3, A.2)

/-- Theorem: Moving point A(1, -1) 3 units to the left results in A'(-2, -1) -/
theorem move_point_left : 
  let A : ℝ × ℝ := (1, -1)
  move_left A = (-2, -1) := by
sorry

end move_point_left_l2691_269159


namespace max_profit_at_one_MP_decreasing_max_profit_x_eq_one_l2691_269153

-- Define the profit function
def P (x : ℕ) : ℚ := -0.2 * x^2 + 25 * x - 40

-- Define the marginal profit function
def MP (x : ℕ) : ℚ := P (x + 1) - P x

-- State the theorem
theorem max_profit_at_one :
  ∀ x : ℕ, 1 ≤ x → x ≤ 100 → P 1 ≥ P x ∧ P 1 = 24.4 := by
  sorry

-- Prove that MP is decreasing
theorem MP_decreasing :
  ∀ x y : ℕ, x < y → MP x > MP y := by
  sorry

-- Prove that maximum profit occurs at x = 1
theorem max_profit_x_eq_one :
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 100 ∧ ∀ y : ℕ, 1 ≤ y ∧ y ≤ 100 → P x ≥ P y := by
  sorry

end max_profit_at_one_MP_decreasing_max_profit_x_eq_one_l2691_269153


namespace library_book_loans_l2691_269146

theorem library_book_loans (initial_A initial_B initial_C final_A final_B final_C : ℕ)
  (return_rate_A return_rate_B return_rate_C : ℚ) :
  initial_A = 75 →
  initial_B = 100 →
  initial_C = 150 →
  final_A = 54 →
  final_B = 82 →
  final_C = 121 →
  return_rate_A = 65/100 →
  return_rate_B = 1/2 →
  return_rate_C = 7/10 →
  ∃ (loaned_A loaned_B loaned_C : ℕ),
    loaned_A + loaned_B + loaned_C = 420 ∧
    loaned_A ≤ loaned_B ∧
    loaned_B ≤ loaned_C ∧
    (↑loaned_A : ℚ) * return_rate_A = final_A ∧
    (↑loaned_B : ℚ) * return_rate_B = final_B ∧
    (↑loaned_C : ℚ) * return_rate_C = final_C :=
by sorry

end library_book_loans_l2691_269146


namespace linda_needs_one_train_l2691_269192

/-- The number of trains Linda currently has -/
def current_trains : ℕ := 31

/-- The number of trains Linda wants in each row -/
def trains_per_row : ℕ := 8

/-- The function to calculate the smallest number of additional trains needed -/
def additional_trains_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - current % per_row) % per_row

/-- The theorem stating that Linda needs 1 additional train -/
theorem linda_needs_one_train : 
  additional_trains_needed current_trains trains_per_row = 1 := by
  sorry

end linda_needs_one_train_l2691_269192


namespace proportion_sum_l2691_269156

theorem proportion_sum (x y : ℝ) : 
  (31.25 : ℝ) / x = 100 / (9.6 : ℝ) ∧ x / 13.75 = (9.6 : ℝ) / y → x + y = 47 := by
  sorry

end proportion_sum_l2691_269156


namespace lion_king_earnings_l2691_269190

/-- Represents movie financial data in millions of dollars -/
structure MovieData where
  productionCost : ℝ
  boxOfficeEarnings : ℝ
  profit : ℝ

/-- The Lion King's box office earnings -/
def lionKingEarnings : ℝ := 200

theorem lion_king_earnings (starWars lionKing : MovieData) :
  starWars.productionCost = 25 →
  starWars.boxOfficeEarnings = 405 →
  lionKing.productionCost = 10 →
  lionKing.profit = (starWars.boxOfficeEarnings - starWars.productionCost) / 2 →
  lionKing.boxOfficeEarnings = lionKingEarnings := by
  sorry

end lion_king_earnings_l2691_269190


namespace total_stocking_stuffers_l2691_269196

def num_kids : ℕ := 3
def candy_canes_per_stocking : ℕ := 4
def beanie_babies_per_stocking : ℕ := 2
def books_per_stocking : ℕ := 1
def small_toys_per_stocking : ℕ := 3
def gift_cards_per_stocking : ℕ := 1

def items_per_stocking : ℕ := 
  candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking + 
  small_toys_per_stocking + gift_cards_per_stocking

theorem total_stocking_stuffers : 
  num_kids * items_per_stocking = 33 := by
  sorry

end total_stocking_stuffers_l2691_269196


namespace arcade_spending_amount_l2691_269133

def weekly_allowance : ℚ := 345/100

def arcade_spending (x : ℚ) : Prop :=
  let remaining_after_arcade := weekly_allowance - x
  let toy_store_spending := (1/3) * remaining_after_arcade
  let candy_store_spending := 92/100
  remaining_after_arcade - toy_store_spending = candy_store_spending

theorem arcade_spending_amount :
  ∃ (x : ℚ), arcade_spending x ∧ x = 207/100 := by sorry

end arcade_spending_amount_l2691_269133


namespace complex_ratio_max_value_l2691_269164

theorem complex_ratio_max_value (z : ℂ) (h : Complex.abs z = 2) :
  (Complex.abs (z^2 - z + 1)) / (Complex.abs (2*z - 1 - Complex.I * Real.sqrt 3)) ≤ 3/2 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧
    (Complex.abs (w^2 - w + 1)) / (Complex.abs (2*w - 1 - Complex.I * Real.sqrt 3)) = 3/2 :=
by sorry

end complex_ratio_max_value_l2691_269164


namespace add_12345_seconds_to_1045am_l2691_269102

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time: 10:45:00 -/
def initialTime : Time :=
  { hours := 10, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The expected final time: 13:45:45 -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 45, seconds := 45 }

theorem add_12345_seconds_to_1045am :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end add_12345_seconds_to_1045am_l2691_269102


namespace remaining_digits_count_l2691_269135

theorem remaining_digits_count (total_count : ℕ) (total_avg : ℚ) (subset_count : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total_count = 8 →
  total_avg = 20 →
  subset_count = 5 →
  subset_avg = 12 →
  remaining_avg = 33333333333333336 / 1000000000000000 →
  total_count - subset_count = 3 :=
by sorry

end remaining_digits_count_l2691_269135


namespace ben_baseball_cards_l2691_269110

/-- The number of baseball cards in each box given to Ben by his mother -/
def baseball_cards_per_box : ℕ := sorry

theorem ben_baseball_cards :
  let basketball_boxes : ℕ := 4
  let basketball_cards_per_box : ℕ := 10
  let baseball_boxes : ℕ := 5
  let cards_given_away : ℕ := 58
  let cards_remaining : ℕ := 22
  
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box = 
  cards_given_away + cards_remaining →
  
  baseball_cards_per_box = 8 := by sorry

end ben_baseball_cards_l2691_269110


namespace polynomial_evaluation_l2691_269107

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

def nonzero_coeff_product (P : ℝ → ℝ) : ℝ :=
  3 * (-5) * 4

def coeff_abs_sum (P : ℝ → ℝ) : ℝ :=
  |3| + |-5| + |4|

def Q (x : ℝ) : ℝ :=
  (nonzero_coeff_product P) * x^3 + (nonzero_coeff_product P) * x + (nonzero_coeff_product P)

def R (x : ℝ) : ℝ :=
  (coeff_abs_sum P) * x^3 - (coeff_abs_sum P) * x + (coeff_abs_sum P)

theorem polynomial_evaluation :
  Q 1 = -180 ∧ R 1 = 12 ∧ Q 1 ≠ R 1 := by
  sorry

end polynomial_evaluation_l2691_269107


namespace recipe_flour_amount_l2691_269191

theorem recipe_flour_amount (flour_added : ℕ) (flour_needed : ℕ) : 
  flour_added = 4 → flour_needed = 4 → flour_added + flour_needed = 8 := by
  sorry

end recipe_flour_amount_l2691_269191


namespace special_sequence_a10_l2691_269172

/-- A sequence of positive real numbers satisfying aₚ₊ₖ = aₚ · aₖ for all positive integers p and q -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)

theorem special_sequence_a10 (a : ℕ → ℝ) (h : SpecialSequence a) (h8 : a 8 = 16) : 
  a 10 = 32 := by
  sorry

end special_sequence_a10_l2691_269172


namespace product_equality_implies_sum_l2691_269161

theorem product_equality_implies_sum (g h a b : ℝ) :
  (∀ d : ℝ, (8 * d^2 - 4 * d + g) * (2 * d^2 + h * d - 7) = 16 * d^4 - 28 * d^3 + a * h^2 * d^2 - b * d + 49) →
  g + h = -3 := by
sorry

end product_equality_implies_sum_l2691_269161


namespace sector_central_angle_l2691_269173

/-- Given a circle with circumference 2π + 2 and a sector of that circle with arc length 2π - 2,
    the central angle of the sector is π - 1. -/
theorem sector_central_angle (r : ℝ) (α : ℝ) 
    (h_circumference : 2 * π * r = 2 * π + 2)
    (h_arc_length : r * α = 2 * π - 2) : 
  α = π - 1 := by
  sorry

end sector_central_angle_l2691_269173


namespace palindrome_percentage_l2691_269126

/-- A palindrome between 1000 and 2000 -/
structure Palindrome :=
  (a b c : Fin 10)

/-- The set of all palindromes between 1000 and 2000 -/
def all_palindromes : Finset Palindrome :=
  sorry

/-- The set of palindromes containing at least one 3 or 5 (except in the first digit) -/
def palindromes_with_3_or_5 : Finset Palindrome :=
  sorry

/-- The percentage of palindromes with 3 or 5 -/
def percentage_with_3_or_5 : ℚ :=
  (palindromes_with_3_or_5.card : ℚ) / (all_palindromes.card : ℚ) * 100

theorem palindrome_percentage :
  percentage_with_3_or_5 = 36 :=
sorry

end palindrome_percentage_l2691_269126


namespace blue_balls_removed_l2691_269149

theorem blue_balls_removed (total_balls : Nat) (initial_blue : Nat) (final_probability : Rat) :
  total_balls = 15 →
  initial_blue = 7 →
  final_probability = 1/3 →
  ∃ (removed : Nat), removed = 3 ∧
    (initial_blue - removed : Rat) / (total_balls - removed : Rat) = final_probability :=
by sorry

end blue_balls_removed_l2691_269149


namespace arithmetic_sequence_product_l2691_269189

-- Define an arithmetic sequence of integers
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

-- Theorem statement
theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  is_arithmetic_sequence a →
  is_increasing_sequence a →
  a 4 * a 5 = 45 →
  a 3 * a 6 = 13 := by
sorry

end arithmetic_sequence_product_l2691_269189


namespace columbus_discovery_year_l2691_269140

def is_valid_year (year : ℕ) : Prop :=
  1000 ≤ year ∧ year < 2000 ∧
  (year / 1000 = 1) ∧
  (year / 100 % 10 ≠ year / 10 % 10) ∧
  (year / 100 % 10 ≠ year % 10) ∧
  (year / 10 % 10 ≠ year % 10) ∧
  (year / 1000 + year / 100 % 10 + year / 10 % 10 + year % 10 = 16) ∧
  (year / 10 % 10 + 1 = 5 * (year % 10))

theorem columbus_discovery_year :
  ∀ year : ℕ, is_valid_year year ↔ year = 1492 :=
by sorry

end columbus_discovery_year_l2691_269140


namespace triangle_abc_properties_l2691_269132

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove the measure of angle A
    and the perimeter of the triangle under specific conditions. -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a * Real.sin B = -Real.sqrt 3 * b * Real.cos A →
  b = 4 →
  S = 2 * Real.sqrt 3 →
  S = (1/2) * b * c * Real.sin A →
  (A = (2/3) * Real.pi ∧ a + b + c = 6 + 2 * Real.sqrt 7) := by
  sorry


end triangle_abc_properties_l2691_269132


namespace tina_remaining_money_l2691_269116

def monthly_income : ℝ := 1000

def june_bonus_rate : ℝ := 0.1
def investment_return_rate : ℝ := 0.05
def tax_rate : ℝ := 0.1

def june_savings_rate : ℝ := 0.25
def july_savings_rate : ℝ := 0.2
def august_savings_rate : ℝ := 0.3

def june_rent : ℝ := 200
def june_groceries : ℝ := 100
def june_book_rate : ℝ := 0.05

def july_rent : ℝ := 250
def july_groceries : ℝ := 150
def july_shoes_rate : ℝ := 0.15

def august_rent : ℝ := 300
def august_groceries : ℝ := 175
def august_misc_rate : ℝ := 0.1

theorem tina_remaining_money :
  let june_income := monthly_income * (1 + june_bonus_rate)
  let june_expenses := june_rent + june_groceries + (june_income * june_book_rate)
  let june_savings := june_income * june_savings_rate
  let june_remaining := june_income - june_savings - june_expenses

  let july_investment_return := june_savings * investment_return_rate
  let july_income := monthly_income + july_investment_return
  let july_expenses := july_rent + july_groceries + (monthly_income * july_shoes_rate)
  let july_savings := july_income * july_savings_rate
  let july_remaining := july_income - july_savings - july_expenses

  let august_investment_return := july_savings * investment_return_rate
  let august_income := monthly_income + august_investment_return
  let august_expenses := august_rent + august_groceries + (monthly_income * august_misc_rate)
  let august_savings := august_income * august_savings_rate
  let august_remaining := august_income - august_savings - august_expenses

  let total_investment_return := july_investment_return + august_investment_return
  let total_tax := total_investment_return * tax_rate
  let total_remaining := june_remaining + july_remaining + august_remaining - total_tax

  total_remaining = 860.7075 := by sorry

end tina_remaining_money_l2691_269116


namespace geometric_series_ratio_l2691_269115

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (∃ S : ℝ, S = a / (1 - r) ∧ S = 81 * (a * r^4 / (1 - r))) → r = 1/3 := by
  sorry

end geometric_series_ratio_l2691_269115


namespace books_movies_difference_l2691_269147

theorem books_movies_difference (total_books total_movies : ℕ) 
  (h1 : total_books = 10) 
  (h2 : total_movies = 6) : 
  total_books - total_movies = 4 := by
  sorry

end books_movies_difference_l2691_269147


namespace inequality_proof_l2691_269177

theorem inequality_proof (K x : ℝ) (hK : K > 1) (hx_pos : x > 0) (hx_bound : x < π / K) :
  (Real.sin (K * x) / Real.sin x) < K * Real.exp (-(K^2 - 1) * x^2 / 6) := by
  sorry

end inequality_proof_l2691_269177


namespace equal_roots_quadratic_l2691_269175

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (1/2 : ℝ) * x^2 - 2*m*x + 4*m + 1 = 0 ∧ 
   ∀ y : ℝ, (1/2 : ℝ) * y^2 - 2*m*y + 4*m + 1 = 0 → y = x) → 
  m^2 - 2*m = 1/2 := by
sorry

end equal_roots_quadratic_l2691_269175


namespace time_equation_l2691_269101

-- Define variables
variable (g V V₀ c S t : ℝ)

-- State the theorem
theorem time_equation (eq1 : V = g * t + V₀ + c) (eq2 : S = (1/2) * g * t^2 + V₀ * t + c * t^2) :
  t = 2 * S / (V + V₀ - c) := by
  sorry

end time_equation_l2691_269101


namespace no_solution_equation1_solutions_equation2_l2691_269129

-- Define the equations
def equation1 (x : ℝ) : Prop := 1 + (3 * x) / (x - 2) = 6 / (x - 2)
def equation2 (x : ℝ) : Prop := x^2 + x - 6 = 0

-- Theorem for the first equation
theorem no_solution_equation1 : ¬ ∃ x : ℝ, equation1 x := by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  (equation2 (-3) ∧ equation2 2) ∧ 
  (∀ x : ℝ, equation2 x → (x = -3 ∨ x = 2)) := by sorry

end no_solution_equation1_solutions_equation2_l2691_269129


namespace consecutive_integers_with_properties_l2691_269106

def sumOfDigits (n : ℕ) : ℕ := sorry

def isPrime (n : ℕ) : Prop := sorry

def isPerfect (n : ℕ) : Prop := sorry

def isSquareFree (n : ℕ) : Prop := sorry

def numberOfDivisors (n : ℕ) : ℕ := sorry

def hasOnePrimeDivisorLessThan10 (n : ℕ) : Prop := sorry

def atMostTwoDigitsEqualOne (n : ℕ) : Prop := sorry

theorem consecutive_integers_with_properties :
  ∃ (n : ℕ),
    (isPrime (sumOfDigits n) ∨ isPrime (sumOfDigits (n + 1)) ∨ isPrime (sumOfDigits (n + 2))) ∧
    (isPerfect (sumOfDigits n) ∨ isPerfect (sumOfDigits (n + 1)) ∨ isPerfect (sumOfDigits (n + 2))) ∧
    (sumOfDigits n = numberOfDivisors n ∨ sumOfDigits (n + 1) = numberOfDivisors (n + 1) ∨ sumOfDigits (n + 2) = numberOfDivisors (n + 2)) ∧
    (atMostTwoDigitsEqualOne n ∧ atMostTwoDigitsEqualOne (n + 1) ∧ atMostTwoDigitsEqualOne (n + 2)) ∧
    (∃ (m : ℕ), (n + 11 = m^2) ∨ (n + 12 = m^2) ∨ (n + 13 = m^2)) ∧
    (hasOnePrimeDivisorLessThan10 n ∧ hasOnePrimeDivisorLessThan10 (n + 1) ∧ hasOnePrimeDivisorLessThan10 (n + 2)) ∧
    (isSquareFree n ∧ isSquareFree (n + 1) ∧ isSquareFree (n + 2)) ∧
    n = 2013 :=
by
  sorry

end consecutive_integers_with_properties_l2691_269106


namespace solution_is_one_l2691_269144

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  (7 / (x^2 + x)) - (3 / (x - x^2)) = 1 + ((7 - x^2) / (x^2 - 1))

/-- Theorem stating that x = 1 is the solution to the equation -/
theorem solution_is_one : equation 1 := by
  sorry

end solution_is_one_l2691_269144


namespace prime_square_product_theorem_l2691_269128

theorem prime_square_product_theorem :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℕ),
    Prime x₁ ∧ Prime x₂ ∧ Prime x₃ ∧ Prime x₄ ∧
    Prime x₅ ∧ Prime x₆ ∧ Prime x₇ ∧ Prime x₈ →
    4 * (x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈) -
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 + x₆^2 + x₇^2 + x₈^2) = 992 →
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧
    x₅ = 2 ∧ x₆ = 2 ∧ x₇ = 2 ∧ x₈ = 2 :=
by sorry

end prime_square_product_theorem_l2691_269128


namespace player_time_on_field_l2691_269127

/-- Proves that each player in a team of 10 will play for 36 minutes in a 45-minute match with 8 players always on the field. -/
theorem player_time_on_field
  (team_size : ℕ)
  (players_on_field : ℕ)
  (match_duration : ℕ)
  (h1 : team_size = 10)
  (h2 : players_on_field = 8)
  (h3 : match_duration = 45)
  : (players_on_field * match_duration) / team_size = 36 := by
  sorry

#eval (8 * 45) / 10  -- Should output 36

end player_time_on_field_l2691_269127


namespace brians_books_l2691_269179

theorem brians_books (x : ℕ) : 
  x + 2 * 15 + (x + 2 * 15) / 2 = 75 → x = 20 := by
  sorry

end brians_books_l2691_269179


namespace second_company_daily_rate_l2691_269113

/-- Represents the daily rate and per-mile rate for a car rental company -/
structure RentalRate where
  daily : ℝ
  perMile : ℝ

/-- Calculates the total cost for a rental given the rate and miles driven -/
def totalCost (rate : RentalRate) (miles : ℝ) : ℝ :=
  rate.daily + rate.perMile * miles

theorem second_company_daily_rate :
  let sunshine := RentalRate.mk 17.99 0.18
  let other := RentalRate.mk x 0.16
  let miles := 48.0
  totalCost sunshine miles = totalCost other miles →
  x = 18.95 := by
  sorry

end second_company_daily_rate_l2691_269113


namespace geometric_sequence_formula_l2691_269114

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a2 : a 2 = 4) :
  ∀ n : ℕ, a n = 2^n := by
sorry

end geometric_sequence_formula_l2691_269114


namespace correct_calculation_l2691_269185

theorem correct_calculation (a b : ℝ) : 3 * a * b + 2 * a * b = 5 * a * b := by
  sorry

end correct_calculation_l2691_269185


namespace sector_perimeter_l2691_269198

theorem sector_perimeter (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 8) :
  2 * r + 2 * area / r = 12 := by
  sorry

end sector_perimeter_l2691_269198


namespace sean_has_45_whistles_l2691_269145

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := 13

/-- The number of additional whistles Sean has compared to Charles -/
def sean_additional_whistles : ℕ := 32

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := charles_whistles + sean_additional_whistles

theorem sean_has_45_whistles : sean_whistles = 45 := by
  sorry

end sean_has_45_whistles_l2691_269145


namespace geometric_sequence_sum_l2691_269188

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 3 + a 2 * a 4 + 2 * a 2 * a 3 = 49 →
  a 2 + a 3 = 7 := by
sorry

end geometric_sequence_sum_l2691_269188


namespace g_negative_three_l2691_269169

def g (x : ℝ) : ℝ := 3*x^5 - 5*x^4 + 9*x^3 - 6*x^2 + 15*x - 210

theorem g_negative_three : g (-3) = -1686 := by
  sorry

end g_negative_three_l2691_269169


namespace inequality_proof_l2691_269138

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^4 + b^4 + c^4 + d^4 - 4*a*b*c*d ≥ 4*(a - b)^2 * Real.sqrt (a*b*c*d) := by
  sorry

end inequality_proof_l2691_269138


namespace unique_solution_for_difference_of_squares_l2691_269119

theorem unique_solution_for_difference_of_squares : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 204 := by sorry

end unique_solution_for_difference_of_squares_l2691_269119


namespace debate_pairs_l2691_269158

theorem debate_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end debate_pairs_l2691_269158


namespace opposite_of_negative_two_thirds_l2691_269187

theorem opposite_of_negative_two_thirds :
  let x : ℚ := -2/3
  let opposite (y : ℚ) := -y
  opposite x = 2/3 := by sorry

end opposite_of_negative_two_thirds_l2691_269187


namespace ellipse_and_triangle_properties_l2691_269134

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about the ellipse and triangle properties -/
theorem ellipse_and_triangle_properties
  (e : Ellipse)
  (focus : Point)
  (pass_through : Point)
  (p : Point)
  (h_focus : focus = ⟨2 * Real.sqrt 2, 0⟩)
  (h_pass : pass_through = ⟨3, 1⟩)
  (h_p : p = ⟨-3, 2⟩)
  (h_on_ellipse : pass_through.x^2 / e.a^2 + pass_through.y^2 / e.b^2 = 1)
  (h_focus_prop : e.a^2 - e.b^2 = 8)
  (h_intersect : ∃ (a b : Point), a ≠ b ∧
    a.x^2 / e.a^2 + a.y^2 / e.b^2 = 1 ∧
    b.x^2 / e.a^2 + b.y^2 / e.b^2 = 1 ∧
    a.y - b.y = a.x - b.x)
  (h_isosceles : ∃ (a b : Point), 
    (a.x - p.x)^2 + (a.y - p.y)^2 = (b.x - p.x)^2 + (b.y - p.y)^2) :
  e.a^2 = 12 ∧ e.b^2 = 4 ∧
  (∃ (a b : Point), 
    (1/2) * Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2) * 
    (3 / Real.sqrt 2) = 9/2) := by
  sorry

end ellipse_and_triangle_properties_l2691_269134


namespace expression_simplification_l2691_269136

theorem expression_simplification (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3*a) (h3 : b ≠ a) (h4 : b ≠ -a) :
  ((2*b + a - (4*a^2 - b^2)/a) / (b^3 + 2*a*b^2 - 3*a^2*b)) * 
  ((a^3*b - 2*a^2*b^2 + a*b^3) / (a^2 - b^2)) = (a - b) / (a + b) := by
  sorry

end expression_simplification_l2691_269136


namespace coin_toss_probability_l2691_269182

theorem coin_toss_probability : 
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of heads we want
  let p : ℚ := 1/2  -- Probability of getting heads on a single toss
  Nat.choose n k * p^n = 5/16 := by
  sorry

end coin_toss_probability_l2691_269182


namespace diana_earnings_ratio_l2691_269160

/-- Diana's earnings over three months --/
def DianaEarnings (july : ℕ) (august_multiple : ℕ) : Prop :=
  let august := july * august_multiple
  let september := 2 * august
  july + august + september = 1500

theorem diana_earnings_ratio : 
  DianaEarnings 150 3 ∧ 
  ∀ x : ℕ, DianaEarnings 150 x → x = 3 :=
by sorry

end diana_earnings_ratio_l2691_269160


namespace cistern_filling_time_l2691_269183

theorem cistern_filling_time (capacity : ℝ) (fill_time : ℝ) (empty_time : ℝ) :
  fill_time = 10 →
  empty_time = 15 →
  (capacity / fill_time - capacity / empty_time) * (fill_time * empty_time / (empty_time - fill_time)) = capacity :=
by
  sorry

#check cistern_filling_time

end cistern_filling_time_l2691_269183


namespace divisibility_of_quadratic_form_l2691_269112

theorem divisibility_of_quadratic_form (n : ℕ) (h : 0 < n) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (n ∣ 4 * a^2 + 9 * b^2 - 1) := by
  sorry

end divisibility_of_quadratic_form_l2691_269112


namespace contrapositive_quadratic_inequality_l2691_269151

theorem contrapositive_quadratic_inequality :
  (∀ x : ℝ, x^2 + x - 6 > 0 → x < -3 ∨ x > 2) ↔
  (∀ x : ℝ, x ≥ -3 ∧ x ≤ 2 → x^2 + x - 6 ≤ 0) :=
by sorry

end contrapositive_quadratic_inequality_l2691_269151


namespace matrix_product_equals_A_l2691_269125

variable {R : Type*} [Field R]
variable (d e f x y z : R)

def A : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def B : Matrix (Fin 3) (Fin 3) R :=
  ![![x^2 + 1, x*y, x*z],
    ![x*y, y^2 + 1, y*z],
    ![x*z, y*z, z^2 + 1]]

theorem matrix_product_equals_A :
  A d e f * B x y z = A d e f := by sorry

end matrix_product_equals_A_l2691_269125


namespace solution_product_l2691_269120

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 8) = p^2 - 15 * p + 54 →
  (q - 6) * (3 * q + 8) = q^2 - 15 * q + 54 →
  p ≠ q →
  (p + 4) * (q + 4) = 130 := by
sorry

end solution_product_l2691_269120


namespace unique_solution_l2691_269197

def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

theorem unique_solution : ∃! A : ℝ, clubsuit A 5 = 80 ∧ A = 14.5 := by
  sorry

end unique_solution_l2691_269197


namespace min_value_quadratic_l2691_269195

-- Define the function f(x) = x^2 + px + q
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Theorem statement
theorem min_value_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p q x_min ≤ f p q x ∧ x_min = -p/2 :=
sorry

end min_value_quadratic_l2691_269195


namespace nancy_books_count_l2691_269154

/-- Given that Alyssa has 36 books and Nancy has 7 times more books than Alyssa,
    prove that Nancy has 252 books. -/
theorem nancy_books_count (alyssa_books : ℕ) (nancy_books : ℕ) 
    (h1 : alyssa_books = 36)
    (h2 : nancy_books = 7 * alyssa_books) : 
  nancy_books = 252 := by
  sorry

end nancy_books_count_l2691_269154


namespace milk_remaining_l2691_269168

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) : 
  initial = 4 → given_away = 16/3 → remaining = initial - given_away → remaining = -4/3 := by
  sorry

end milk_remaining_l2691_269168


namespace mojave_population_increase_l2691_269199

/-- The factor by which the population of Mojave has increased over a decade -/
def population_increase_factor (initial_population : ℕ) (future_population : ℕ) (future_increase_percent : ℕ) : ℚ :=
  let current_population := (100 : ℚ) / (100 + future_increase_percent) * future_population
  current_population / initial_population

/-- Theorem stating that the population increase factor is 3 -/
theorem mojave_population_increase : 
  population_increase_factor 4000 16800 40 = 3 := by sorry

end mojave_population_increase_l2691_269199


namespace opposite_of_blue_is_white_l2691_269180

/-- Represents the colors of the squares --/
inductive Color
| Red | Blue | Orange | Purple | Green | Yellow | White

/-- Represents the positions on the cube --/
inductive Position
| Top | Bottom | Front | Back | Left | Right

/-- Represents a cube configuration --/
structure CubeConfig where
  top : Color
  bottom : Color
  front : Color
  back : Color
  left : Color
  right : Color

/-- Defines the property of opposite faces --/
def isOpposite (p1 p2 : Position) : Prop :=
  (p1 = Position.Top ∧ p2 = Position.Bottom) ∨
  (p1 = Position.Bottom ∧ p2 = Position.Top) ∨
  (p1 = Position.Front ∧ p2 = Position.Back) ∨
  (p1 = Position.Back ∧ p2 = Position.Front) ∨
  (p1 = Position.Left ∧ p2 = Position.Right) ∨
  (p1 = Position.Right ∧ p2 = Position.Left)

/-- The main theorem --/
theorem opposite_of_blue_is_white 
  (cube : CubeConfig)
  (top_is_purple : cube.top = Color.Purple)
  (front_is_green : cube.front = Color.Green)
  (blue_on_side : cube.left = Color.Blue ∨ cube.right = Color.Blue) :
  (cube.left = Color.Blue ∧ cube.right = Color.White) ∨ 
  (cube.right = Color.Blue ∧ cube.left = Color.White) :=
sorry

end opposite_of_blue_is_white_l2691_269180


namespace bryden_receives_20_dollars_l2691_269108

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_rate : ℚ := 2000

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 4

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_receives : ℚ := (collector_rate / 100) * (bryden_quarters : ℚ) * quarter_value

theorem bryden_receives_20_dollars : bryden_receives = 20 := by
  sorry

end bryden_receives_20_dollars_l2691_269108


namespace last_digit_of_2_to_20_l2691_269141

theorem last_digit_of_2_to_20 (n : ℕ) :
  n ≥ 1 → (2^n : ℕ) % 10 = ((2^(n % 4)) : ℕ) % 10 →
  (2^20 : ℕ) % 10 = 6 := by
  sorry

end last_digit_of_2_to_20_l2691_269141


namespace select_computers_l2691_269118

theorem select_computers (type_a : ℕ) (type_b : ℕ) : 
  type_a = 4 → type_b = 5 → 
  (Nat.choose type_a 2 * Nat.choose type_b 1) + (Nat.choose type_a 1 * Nat.choose type_b 2) = 70 := by
  sorry

end select_computers_l2691_269118


namespace remainder_256_div_13_l2691_269139

theorem remainder_256_div_13 : ∃ q r : ℤ, 256 = 13 * q + r ∧ 0 ≤ r ∧ r < 13 ∧ r = 9 := by
  sorry

end remainder_256_div_13_l2691_269139


namespace least_number_divisibility_l2691_269181

theorem least_number_divisibility (x : ℕ) : x = 10315 ↔ 
  (∀ y : ℕ, y < x → ¬((1024 + y) % (17 * 23 * 29) = 0)) ∧ 
  ((1024 + x) % (17 * 23 * 29) = 0) := by
  sorry

end least_number_divisibility_l2691_269181


namespace c_7_equals_448_l2691_269162

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem c_7_equals_448 : c 7 = 448 := by
  sorry

end c_7_equals_448_l2691_269162


namespace factor_implies_d_value_l2691_269184

theorem factor_implies_d_value (d : ℚ) :
  (∀ x : ℚ, (3 * x + 4) ∣ (4 * x^3 + 17 * x^2 + d * x + 28)) →
  d = 155 / 9 := by
sorry

end factor_implies_d_value_l2691_269184


namespace quadratic_inequality_always_true_l2691_269165

theorem quadratic_inequality_always_true (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m ≤ 0 := by
  sorry

end quadratic_inequality_always_true_l2691_269165


namespace line_point_order_l2691_269186

/-- Given a line y = mx + n where m < 0 and n > 0, if points A(-2, y₁), B(-3, y₂), and C(1, y₃) 
    are on the line, then y₃ < y₁ < y₂. -/
theorem line_point_order (m n y₁ y₂ y₃ : ℝ) 
    (hm : m < 0) (hn : n > 0)
    (hA : y₁ = m * (-2) + n)
    (hB : y₂ = m * (-3) + n)
    (hC : y₃ = m * 1 + n) :
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end line_point_order_l2691_269186


namespace function_with_bounded_difference_is_constant_l2691_269105

/-- A function f: ℝ → ℝ that satisfies |f(x) - f(y)| ≤ (x - y)² for all x, y ∈ ℝ is constant. -/
theorem function_with_bounded_difference_is_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, |f x - f y| ≤ (x - y)^2) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end function_with_bounded_difference_is_constant_l2691_269105


namespace line_intercept_form_l2691_269121

/-- Given a line with equation 3x - 2y = 4, its intercept form is x/(4/3) + y/(-2) = 1 -/
theorem line_intercept_form :
  ∀ (x y : ℝ), 3*x - 2*y = 4 → x/(4/3) + y/(-2) = 1 := by sorry

end line_intercept_form_l2691_269121


namespace distance_difference_l2691_269157

/-- The difference in distances from Q to the intersection points of a line and a parabola -/
theorem distance_difference (Q : ℝ × ℝ) (C D : ℝ × ℝ) : 
  Q.1 = 2 ∧ Q.2 = 0 →
  C.2 - 2 * C.1 + 4 = 0 →
  D.2 - 2 * D.1 + 4 = 0 →
  C.2^2 = 3 * C.1 + 4 →
  D.2^2 = 3 * D.1 + 4 →
  |((C.1 - Q.1)^2 + (C.2 - Q.2)^2).sqrt - ((D.1 - Q.1)^2 + (D.2 - Q.2)^2).sqrt| = 
  |2 * (5 : ℝ).sqrt - (8.90625 : ℝ).sqrt| :=
by sorry

end distance_difference_l2691_269157


namespace sqrt_twelve_minus_sqrt_three_l2691_269148

theorem sqrt_twelve_minus_sqrt_three : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_three_l2691_269148


namespace quadratic_function_property_l2691_269167

/-- Given a quadratic function f(x) = ax^2 + bx + 5 where a ≠ 0,
    if there exist two distinct points (x₁, 2002) and (x₂, 2002) on the graph of f,
    then f(x₁ + x₂) = 5. -/
theorem quadratic_function_property (a b x₁ x₂ : ℝ) (ha : a ≠ 0) :
  let f := λ x : ℝ => a * x^2 + b * x + 5
  (f x₁ = 2002) → (f x₂ = 2002) → (x₁ ≠ x₂) → f (x₁ + x₂) = 5 := by
  sorry

end quadratic_function_property_l2691_269167


namespace range_of_a_l2691_269109

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → -1 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l2691_269109


namespace average_problem_l2691_269166

theorem average_problem (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄ + x₅ + 3) / 6 = 3) : 
  (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3 := by
  sorry

end average_problem_l2691_269166


namespace inequality_solution_expression_value_l2691_269170

-- Problem 1
theorem inequality_solution (x : ℝ) : 2*x - 3 > x + 1 ↔ x > 4 := by sorry

-- Problem 2
theorem expression_value (a b : ℝ) (h : a^2 + 3*a*b = 5) : 
  (a + b) * (a + 2*b) - 2*b^2 = 5 := by sorry

end inequality_solution_expression_value_l2691_269170


namespace x_value_proof_l2691_269131

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem x_value_proof (x y : ℕ) : 
  x = sum_integers 10 20 → 
  y = count_even_integers 10 20 → 
  x + y = 171 → 
  x = 165 := by sorry

end x_value_proof_l2691_269131


namespace no_rational_roots_l2691_269104

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3

theorem no_rational_roots :
  ∀ x : ℚ, polynomial x ≠ 0 := by sorry

end no_rational_roots_l2691_269104


namespace intersection_of_A_and_B_l2691_269152

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 10}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 3 ≤ x ∧ x < 7} := by sorry

end intersection_of_A_and_B_l2691_269152


namespace three_digit_numbers_equation_l2691_269150

theorem three_digit_numbers_equation : 
  ∃! (A B : ℕ), 
    100 ≤ A ∧ A < 1000 ∧
    100 ≤ B ∧ B < 1000 ∧
    1000 * A + B = 3 * A * B := by
  sorry

end three_digit_numbers_equation_l2691_269150


namespace smallest_solution_quadratic_l2691_269176

theorem smallest_solution_quadratic (x : ℝ) :
  (6 * x^2 - 29 * x + 35 = 0) → (x ≥ 7/3) :=
by sorry

end smallest_solution_quadratic_l2691_269176


namespace prime_pairs_satisfying_equation_l2691_269103

theorem prime_pairs_satisfying_equation :
  ∀ (p q : ℕ), Prime p → Prime q →
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end prime_pairs_satisfying_equation_l2691_269103


namespace sum_of_y_values_l2691_269124

theorem sum_of_y_values (y : ℝ) : 
  (∃ (y₁ y₂ : ℝ), 
    (Real.sqrt ((y₁ - 2)^2) = 9 ∧ 
     Real.sqrt ((y₂ - 2)^2) = 9 ∧ 
     y₁ ≠ y₂ ∧
     (∀ y', Real.sqrt ((y' - 2)^2) = 9 → y' = y₁ ∨ y' = y₂)) →
    y₁ + y₂ = 4) :=
sorry

end sum_of_y_values_l2691_269124
