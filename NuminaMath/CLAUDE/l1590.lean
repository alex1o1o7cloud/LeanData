import Mathlib

namespace quadratic_inequality_solution_set_l1590_159082

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ 2 < x ∧ x < 3) 
  (h2 : a < 0) :
  ∀ x, c*x^2 - b*x + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by sorry

end quadratic_inequality_solution_set_l1590_159082


namespace cos_18_minus_cos_54_l1590_159070

theorem cos_18_minus_cos_54 :
  Real.cos (18 * π / 180) - Real.cos (54 * π / 180) =
  -16 * (Real.cos (9 * π / 180))^4 + 24 * (Real.cos (9 * π / 180))^2 - 4 := by
sorry

end cos_18_minus_cos_54_l1590_159070


namespace fraction_of_one_third_is_one_eighth_l1590_159076

theorem fraction_of_one_third_is_one_eighth (a b c d : ℚ) : 
  a = 1/3 → b = 1/8 → (b/a = c/d) → (c = 3 ∧ d = 8) := by
  sorry

end fraction_of_one_third_is_one_eighth_l1590_159076


namespace greatest_five_digit_integer_l1590_159051

def reverse_digits (n : Nat) : Nat :=
  -- Implementation of reverse_digits function
  sorry

def is_five_digit (n : Nat) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem greatest_five_digit_integer (p : Nat) : 
  is_five_digit p ∧ 
  is_five_digit (reverse_digits p) ∧
  p % 63 = 0 ∧
  (reverse_digits p) % 63 = 0 ∧
  p % 11 = 0 →
  p ≤ 99729 :=
sorry

end greatest_five_digit_integer_l1590_159051


namespace cubic_root_sum_l1590_159032

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  (a / ((1/a) + b*c)) + (b / ((1/b) + c*a)) + (c / ((1/c) + a*b)) = 181/9 := by
sorry

end cubic_root_sum_l1590_159032


namespace video_game_marathon_points_l1590_159056

theorem video_game_marathon_points : 
  ∀ (jack_points alex_bella_points : ℕ),
    jack_points = 8972 →
    alex_bella_points = 21955 →
    jack_points + alex_bella_points = 30927 := by
  sorry

end video_game_marathon_points_l1590_159056


namespace charge_with_interest_after_one_year_l1590_159033

/-- Calculates the amount owed after one year given an initial charge and simple annual interest rate -/
def amount_owed_after_one_year (initial_charge : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_charge * (1 + interest_rate)

/-- Theorem stating that a $35 charge with 7% simple annual interest results in $37.45 owed after one year -/
theorem charge_with_interest_after_one_year :
  let initial_charge : ℝ := 35
  let interest_rate : ℝ := 0.07
  amount_owed_after_one_year initial_charge interest_rate = 37.45 := by
  sorry

#eval amount_owed_after_one_year 35 0.07

end charge_with_interest_after_one_year_l1590_159033


namespace ellipse_touches_hyperbola_l1590_159087

/-- An ellipse touches a hyperbola if they share a common point and have the same tangent at that point -/
def touches (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x ∧
    (-(b / a) * (x / Real.sqrt (a ^ 2 - x ^ 2))) = -1 / x ^ 2

/-- If an ellipse with equation (x/a)^2 + (y/b)^2 = 1 touches a hyperbola with equation y = 1/x, then ab = 2 -/
theorem ellipse_touches_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  touches a b → a * b = 2 := by
  sorry

#check ellipse_touches_hyperbola

end ellipse_touches_hyperbola_l1590_159087


namespace max_min_triangle_area_l1590_159048

/-- A point on the 10x10 grid -/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The configuration of three pieces on the grid -/
structure Configuration where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  sorry

/-- Check if two grid points are adjacent -/
def isAdjacent (p1 p2 : GridPoint) : Prop :=
  sorry

/-- A valid move between two configurations -/
def validMove (c1 c2 : Configuration) : Prop :=
  sorry

/-- A sequence of configurations representing a valid solution -/
def ValidSolution : Type :=
  sorry

/-- The minimum triangle area over all configurations in a solution -/
def minTriangleArea (sol : ValidSolution) : ℚ :=
  sorry

theorem max_min_triangle_area :
  (∃ (sol : ValidSolution), minTriangleArea sol = 5/2) ∧
  (∀ (sol : ValidSolution), minTriangleArea sol ≤ 5/2) := by
  sorry

end max_min_triangle_area_l1590_159048


namespace triangle_ABC_properties_l1590_159012

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  2 * Real.sqrt 3 * (Real.sin ((A + B) / 2))^2 - Real.sin C = Real.sqrt 3 ∧
  c = Real.sqrt 3 ∧
  a = Real.sqrt 2

theorem triangle_ABC_properties {A B C a b c : ℝ} 
  (h : triangle_ABC A B C a b c) : 
  C = π / 3 ∧ 
  (1/2 * a * b * Real.sin C) = (Real.sqrt 3 + 3) / 4 := by
sorry

end triangle_ABC_properties_l1590_159012


namespace sufficient_not_necessary_l1590_159052

theorem sufficient_not_necessary (a b : ℝ) (h : a ≠ b) :
  (a > abs b → a^3 + b^3 > a^2*b + a*b^2) ∧
  ¬(a^3 + b^3 > a^2*b + a*b^2 → a > abs b) :=
sorry

end sufficient_not_necessary_l1590_159052


namespace inscribed_cube_volume_l1590_159044

/-- The volume of a cube inscribed in a specific pyramid -/
theorem inscribed_cube_volume (base_side : ℝ) (h : base_side = 2) :
  let pyramid_height := 2 * Real.sqrt 3 / 3
  let cube_side := 2 * Real.sqrt 3 / 9
  let cube_volume := cube_side ^ 3
  cube_volume = 8 * Real.sqrt 3 / 243 := by
  sorry

end inscribed_cube_volume_l1590_159044


namespace gcd_of_35_91_840_l1590_159067

theorem gcd_of_35_91_840 : Nat.gcd 35 (Nat.gcd 91 840) = 7 := by
  sorry

end gcd_of_35_91_840_l1590_159067


namespace pizza_cost_distribution_l1590_159005

theorem pizza_cost_distribution (total_cost : ℚ) (num_students : ℕ) 
  (price1 price2 : ℚ) (h1 : total_cost = 26) (h2 : num_students = 7) 
  (h3 : price1 = 371/100) (h4 : price2 = 372/100) : 
  ∃ (x y : ℕ), x + y = num_students ∧ 
  x * price1 + y * price2 = total_cost ∧ 
  y = 3 := by sorry

end pizza_cost_distribution_l1590_159005


namespace max_annual_profit_l1590_159027

/-- Represents the annual production quantity -/
def x : Type := { n : ℕ // n > 0 }

/-- Calculates the annual sales revenue in million yuan -/
def salesRevenue (x : x) : ℝ :=
  if x.val ≤ 20 then 33 * x.val - x.val^2 else 260

/-- Calculates the total annual investment in million yuan -/
def totalInvestment (x : x) : ℝ := 1 + 0.01 * x.val

/-- Calculates the annual profit in million yuan -/
def annualProfit (x : x) : ℝ := salesRevenue x - totalInvestment x

/-- Theorem stating the maximum annual profit and the production quantity that achieves it -/
theorem max_annual_profit :
  ∃ (x_max : x), 
    (∀ (x : x), annualProfit x ≤ annualProfit x_max) ∧
    (x_max.val = 16) ∧
    (annualProfit x_max = 156) := by sorry

end max_annual_profit_l1590_159027


namespace run_difference_is_240_l1590_159035

/-- The width of the street in feet -/
def street_width : ℝ := 30

/-- The side length of the square block in feet -/
def block_side : ℝ := 500

/-- The perimeter of Sarah's run (inner side of the block) -/
def sarah_perimeter : ℝ := 4 * block_side

/-- The perimeter of Sam's run (outer side of the block) -/
def sam_perimeter : ℝ := 4 * (block_side + 2 * street_width)

/-- The difference in distance run by Sam and Sarah -/
def run_difference : ℝ := sam_perimeter - sarah_perimeter

theorem run_difference_is_240 : run_difference = 240 := by
  sorry

end run_difference_is_240_l1590_159035


namespace black_dogs_count_l1590_159001

def total_dogs : Nat := 45
def brown_dogs : Nat := 20
def white_dogs : Nat := 10

theorem black_dogs_count : total_dogs - (brown_dogs + white_dogs) = 15 := by
  sorry

end black_dogs_count_l1590_159001


namespace josh_work_hours_l1590_159062

/-- Proves that Josh works 8 hours a day given the problem conditions -/
theorem josh_work_hours :
  ∀ (h : ℝ),
  (20 * h * 9 + (20 * h - 40) * 4.5 = 1980) →
  h = 8 :=
by sorry

end josh_work_hours_l1590_159062


namespace prime_sum_of_composites_l1590_159080

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem prime_sum_of_composites :
  (∃ p : ℕ, Nat.Prime p ∧ p = 13 ∧ 
    ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b) ∧
  (∀ p : ℕ, Nat.Prime p → p > 13 → 
    ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b) :=
sorry

end prime_sum_of_composites_l1590_159080


namespace negative_division_equality_l1590_159025

theorem negative_division_equality : (-81) / (-9) = 9 := by
  sorry

end negative_division_equality_l1590_159025


namespace tan_315_degrees_l1590_159072

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by sorry

end tan_315_degrees_l1590_159072


namespace remaining_plums_l1590_159022

def gyuris_plums (initial : ℝ) (given_to_sungmin : ℝ) (given_to_dongju : ℝ) : ℝ :=
  initial - given_to_sungmin - given_to_dongju

theorem remaining_plums :
  gyuris_plums 1.6 0.8 0.3 = 0.5 := by
  sorry

end remaining_plums_l1590_159022


namespace courtyard_width_main_theorem_l1590_159073

/-- Proves that the width of a rectangular courtyard is 16 meters -/
theorem courtyard_width : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (length width brick_length brick_width : ℝ) =>
    length = 30 ∧
    brick_length = 0.2 ∧
    brick_width = 0.1 ∧
    (length * width) / (brick_length * brick_width) = 24000 →
    width = 16

/-- Main theorem proof -/
theorem main_theorem : courtyard_width 30 16 0.2 0.1 := by
  sorry

end courtyard_width_main_theorem_l1590_159073


namespace equation_solution_l1590_159007

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (x + 2 = 1 / (x - 2)) ↔ (x = Real.sqrt 5 ∨ x = -Real.sqrt 5) :=
by sorry

end equation_solution_l1590_159007


namespace mean_ice_cream_sales_l1590_159017

def ice_cream_sales : List ℕ := [100, 92, 109, 96, 103, 96, 105]

theorem mean_ice_cream_sales :
  (ice_cream_sales.sum : ℚ) / ice_cream_sales.length = 100.14 := by
  sorry

end mean_ice_cream_sales_l1590_159017


namespace equation_solution_l1590_159043

theorem equation_solution : ∃ x : ℝ, (6*x + 7)^2 * (3*x + 4) * (x + 1) = 6 :=
  have h1 : (6 * (-2/3) + 7)^2 * (3 * (-2/3) + 4) * (-2/3 + 1) = 6 := by sorry
  have h2 : (6 * (-5/3) + 7)^2 * (3 * (-5/3) + 4) * (-5/3 + 1) = 6 := by sorry
  ⟨-2/3, h1⟩

#check equation_solution

end equation_solution_l1590_159043


namespace allocation_five_to_three_l1590_159038

/-- The number of ways to allocate n identical objects to k distinct groups,
    with each group receiving at least one object -/
def allocations (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 6 ways to allocate 5 identical objects to 3 distinct groups,
    with each group receiving at least one object -/
theorem allocation_five_to_three :
  allocations 5 3 = 6 := by
  sorry

end allocation_five_to_three_l1590_159038


namespace expand_product_l1590_159014

theorem expand_product (y : ℝ) : 3 * (y - 4) * (y + 9) = 3 * y^2 + 15 * y - 108 := by
  sorry

end expand_product_l1590_159014


namespace parallel_lines_k_equals_two_l1590_159009

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of the line 2x - y + 2 = 0 -/
def line1_slope_intercept (x y : ℝ) : Prop := y = 2 * x + 2

/-- The slope-intercept form of the line y = kx + 1 -/
def line2_slope_intercept (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

theorem parallel_lines_k_equals_two :
  (∀ x y : ℝ, 2 * x - y + 2 = 0 ↔ y = k * x + 1) → k = 2 :=
by sorry

end parallel_lines_k_equals_two_l1590_159009


namespace sqrt_18_greater_than_pi_l1590_159042

theorem sqrt_18_greater_than_pi : Real.sqrt 18 > Real.pi := by
  sorry

end sqrt_18_greater_than_pi_l1590_159042


namespace valid_sequence_count_is_840_l1590_159085

/-- Represents a coin toss sequence --/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a given subsequence in a coin sequence --/
def countSubsequence (seq : CoinSequence) (subseq : CoinSequence) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions --/
def satisfiesConditions (seq : CoinSequence) : Prop :=
  seq.length = 16 ∧
  countSubsequence seq [true, true] = 2 ∧
  countSubsequence seq [false, false] = 6 ∧
  countSubsequence seq [true, false] = 4 ∧
  countSubsequence seq [false, true] = 4

/-- The number of valid coin sequences --/
def validSequenceCount : Nat :=
  sorry

theorem valid_sequence_count_is_840 :
  validSequenceCount = 840 :=
sorry

end valid_sequence_count_is_840_l1590_159085


namespace problem_1_l1590_159006

theorem problem_1 : 23 * (-5) - (-3) / (3/108) = -7 := by sorry

end problem_1_l1590_159006


namespace dog_bones_found_l1590_159064

/-- Given a dog initially has 15 bones and ends up with 23 bones, 
    prove that the number of bones found is 23 - 15. -/
theorem dog_bones_found (initial_bones final_bones : ℕ) 
  (h1 : initial_bones = 15) 
  (h2 : final_bones = 23) : 
  final_bones - initial_bones = 23 - 15 := by
  sorry

end dog_bones_found_l1590_159064


namespace kathryn_salary_l1590_159071

/-- Calculates Kathryn's monthly salary given her expenses and remaining money --/
def monthly_salary (rent : ℕ) (remaining : ℕ) : ℕ :=
  let food_travel := 2 * rent
  let total_expenses := rent + food_travel
  let shared_rent := rent / 2
  let adjusted_expenses := total_expenses - (rent - shared_rent)
  adjusted_expenses + remaining

/-- Proves that Kathryn's monthly salary is $5000 given the problem conditions --/
theorem kathryn_salary :
  let rent : ℕ := 1200
  let remaining : ℕ := 2000
  monthly_salary rent remaining = 5000 := by
  sorry

#eval monthly_salary 1200 2000

end kathryn_salary_l1590_159071


namespace carmela_money_distribution_l1590_159020

/-- Proves that Carmela needs to give $1 to each cousin for equal distribution -/
theorem carmela_money_distribution (carmela_money : ℕ) (cousin_money : ℕ) (num_cousins : ℕ) :
  carmela_money = 7 →
  cousin_money = 2 →
  num_cousins = 4 →
  let total_money := carmela_money + num_cousins * cousin_money
  let num_people := num_cousins + 1
  let equal_share := total_money / num_people
  let carmela_gives := carmela_money - equal_share
  carmela_gives / num_cousins = 1 := by
  sorry

end carmela_money_distribution_l1590_159020


namespace complex_fraction_problem_l1590_159024

theorem complex_fraction_problem (x y : ℂ) (k : ℝ) 
  (h : (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41 / 20 := by
  sorry

end complex_fraction_problem_l1590_159024


namespace equation_solutions_count_l1590_159063

theorem equation_solutions_count : 
  let count := Finset.filter (fun k => 
    k % 2 = 1 ∧ 
    (Finset.filter (fun p : ℕ × ℕ => 
      let (m, n) := p
      2^(4*m^2) + 2^(m^2 - n^2 + 4) = 2^(k + 4) + 2^(3*m^2 + n^2 + k) ∧ 
      m > 0 ∧ n > 0
    ) (Finset.product (Finset.range 101) (Finset.range 101))).card = 2
  ) (Finset.range 101)
  count.card = 18 := by sorry

end equation_solutions_count_l1590_159063


namespace dollar_sum_squared_zero_l1590_159075

/-- Definition of the $ operation for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem: For real numbers x and y, (x + y)^2 $ (y + x)^2 = 0 -/
theorem dollar_sum_squared_zero (x y : ℝ) : dollar ((x + y)^2) ((y + x)^2) = 0 := by
  sorry

end dollar_sum_squared_zero_l1590_159075


namespace book_price_calculation_l1590_159065

/-- Represents the price of a single book -/
def book_price : ℝ := 20

/-- Represents the number of books bought per month -/
def books_per_month : ℕ := 3

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the total sale price of all books at the end of the year -/
def total_sale_price : ℝ := 500

/-- Represents the total loss incurred -/
def total_loss : ℝ := 220

theorem book_price_calculation : 
  book_price * (books_per_month * months_in_year) - total_sale_price = total_loss :=
sorry

end book_price_calculation_l1590_159065


namespace cube_difference_divisibility_l1590_159074

theorem cube_difference_divisibility (a b : ℤ) :
  24 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) ↔ 3 ∣ (a - b) := by
  sorry

end cube_difference_divisibility_l1590_159074


namespace connie_marbles_l1590_159094

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℝ := 183.0

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.0

/-- The initial number of marbles Connie had -/
def initial_marbles : ℝ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776.0 := by
  sorry

end connie_marbles_l1590_159094


namespace parabola_properties_l1590_159050

-- Define the parabola function
def f (a x : ℝ) : ℝ := x^2 + (a + 2) * x - 2 * a + 1

-- State the theorem
theorem parabola_properties (a : ℝ) :
  -- 1. The parabola always passes through the point (2, 9)
  (f a 2 = 9) ∧
  -- 2. The vertex of the parabola lies on the curve y = -x^2 + 4x + 5
  (∃ x y : ℝ, y = f a x ∧ y = -x^2 + 4*x + 5 ∧ 
    ∀ t : ℝ, f a t ≥ f a x) ∧
  -- 3. When the quadratic equation has two distinct real roots,
  --    the range of the larger root is (-1, 2) ∪ (5, +∞)
  (∀ x : ℝ, (f a x = 0 ∧ 
    (∃ y : ℝ, y ≠ x ∧ f a y = 0)) →
    ((x > -1 ∧ x < 2) ∨ x > 5)) :=
by sorry

end parabola_properties_l1590_159050


namespace hilt_garden_border_rocks_l1590_159068

/-- The number of rocks Mrs. Hilt needs to complete her garden border -/
def total_rocks_needed (rocks_on_hand : ℕ) (additional_rocks_needed : ℕ) : ℕ :=
  rocks_on_hand + additional_rocks_needed

/-- Theorem: Mrs. Hilt needs 125 rocks in total to complete her garden border -/
theorem hilt_garden_border_rocks : 
  total_rocks_needed 64 61 = 125 := by
  sorry

end hilt_garden_border_rocks_l1590_159068


namespace inequality_proof_l1590_159031

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by sorry

end inequality_proof_l1590_159031


namespace arithmetic_sequence_ninth_term_l1590_159083

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def sequence_condition (a : ℕ → ℤ) : Prop :=
  a 2 - a 3 - a 7 - a 11 - a 13 + a 16 = 8

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sequence_condition a) :
  a 9 = -4 :=
sorry

end arithmetic_sequence_ninth_term_l1590_159083


namespace prob_B_wins_at_least_one_l1590_159026

/-- The probability of player A winning against player B in a single match. -/
def prob_A_win : ℝ := 0.5

/-- The probability of player B winning against player A in a single match. -/
def prob_B_win : ℝ := 0.3

/-- The probability of a tie between players A and B in a single match. -/
def prob_tie : ℝ := 0.2

/-- The number of matches played between A and B. -/
def num_matches : ℕ := 2

/-- Theorem: The probability of B winning at least one match against A in two independent matches. -/
theorem prob_B_wins_at_least_one (h1 : prob_A_win + prob_B_win + prob_tie = 1) :
  1 - (1 - prob_B_win) ^ num_matches = 0.51 := by
  sorry

end prob_B_wins_at_least_one_l1590_159026


namespace cube_of_negative_double_l1590_159088

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end cube_of_negative_double_l1590_159088


namespace smallest_number_divisible_by_225_with_ones_and_zeros_l1590_159060

def is_composed_of_ones_and_zeros (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 0

def smallest_divisible_by_225_with_ones_and_zeros : ℕ := 11111111100

theorem smallest_number_divisible_by_225_with_ones_and_zeros :
  (smallest_divisible_by_225_with_ones_and_zeros % 225 = 0) ∧
  is_composed_of_ones_and_zeros smallest_divisible_by_225_with_ones_and_zeros ∧
  ∀ n : ℕ, n < smallest_divisible_by_225_with_ones_and_zeros →
    ¬(n % 225 = 0 ∧ is_composed_of_ones_and_zeros n) :=
by sorry

#eval smallest_divisible_by_225_with_ones_and_zeros

end smallest_number_divisible_by_225_with_ones_and_zeros_l1590_159060


namespace remainder_problem_l1590_159041

theorem remainder_problem (y : ℤ) : 
  y % 23 = 19 → y % 276 = 180 := by
sorry

end remainder_problem_l1590_159041


namespace problem_statement_l1590_159010

theorem problem_statement (p q r s : ℝ) (ω : ℂ) 
  (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1/(p + ω) + 1/(q + ω) + 1/(r + ω) + 1/(s + ω) = 3/ω^2) :
  1/(p + 1) + 1/(q + 1) + 1/(r + 1) + 1/(s + 1) = 3 := by
  sorry

end problem_statement_l1590_159010


namespace valid_triangulations_l1590_159066

/-- A triangulation of a triangle is a division of the triangle into n smaller triangles
    such that no three vertices are collinear and each vertex belongs to the same number of segments -/
structure Triangulation :=
  (n : ℕ)  -- number of smaller triangles
  (no_collinear : Bool)  -- no three vertices are collinear
  (equal_vertex_degree : Bool)  -- each vertex belongs to the same number of segments

/-- The set of valid n values for triangulations -/
def ValidTriangulations : Set ℕ := {1, 3, 7, 19}

/-- Theorem stating that the only valid triangulations are those with n in ValidTriangulations -/
theorem valid_triangulations (t : Triangulation) :
  t.no_collinear ∧ t.equal_vertex_degree → t.n ∈ ValidTriangulations := by
  sorry

end valid_triangulations_l1590_159066


namespace carolyn_shared_marbles_l1590_159059

/-- The number of marbles Carolyn shared with Diana -/
def marbles_shared (initial_marbles final_marbles : ℕ) : ℕ :=
  initial_marbles - final_marbles

/-- Theorem stating that Carolyn shared 42 marbles with Diana -/
theorem carolyn_shared_marbles :
  let initial_marbles : ℕ := 47
  let final_marbles : ℕ := 5
  marbles_shared initial_marbles final_marbles = 42 := by
  sorry

end carolyn_shared_marbles_l1590_159059


namespace f_min_at_300_l1590_159078

/-- The quadratic expression we're minimizing -/
def f (x : ℝ) : ℝ := x^2 - 600*x + 369

/-- The theorem stating that f(x) takes its minimum value when x = 300 -/
theorem f_min_at_300 : 
  ∀ x : ℝ, f x ≥ f 300 := by sorry

end f_min_at_300_l1590_159078


namespace unique_triple_product_sum_l1590_159077

theorem unique_triple_product_sum : 
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a * b = c ∧ b * c = a ∧ c * a = b ∧ a + b + c = 2 := by
  sorry

end unique_triple_product_sum_l1590_159077


namespace ps_length_l1590_159086

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (PQRS : Quadrilateral) : Prop :=
  let (px, py) := PQRS.P
  let (qx, qy) := PQRS.Q
  let (rx, ry) := PQRS.R
  let (sx, sy) := PQRS.S
  -- PQ = 6
  (px - qx)^2 + (py - qy)^2 = 36 ∧
  -- QR = 10
  (qx - rx)^2 + (qy - ry)^2 = 100 ∧
  -- RS = 25
  (rx - sx)^2 + (ry - sy)^2 = 625 ∧
  -- Angle Q is right angle
  (px - qx) * (rx - qx) + (py - qy) * (ry - qy) = 0 ∧
  -- Angle R is right angle
  (qx - rx) * (sx - rx) + (qy - ry) * (sy - ry) = 0

-- Theorem statement
theorem ps_length (PQRS : Quadrilateral) (h : is_valid_quadrilateral PQRS) :
  (PQRS.P.1 - PQRS.S.1)^2 + (PQRS.P.2 - PQRS.S.2)^2 = 461 :=
by sorry

end ps_length_l1590_159086


namespace negation_of_universal_proposition_l1590_159018

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 1 < 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≥ 0 := by sorry

end negation_of_universal_proposition_l1590_159018


namespace linear_equation_condition_l1590_159004

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k l m : ℝ, (a^2 - 4) * x^2 + (2 - 3*a) * x + (a + 1) * y + 3*a = k * x + l * y + m) → 
  (a = 2 ∨ a = -2) :=
by sorry

end linear_equation_condition_l1590_159004


namespace fraction_simplification_l1590_159011

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) = 4 / 5 := by
  sorry

end fraction_simplification_l1590_159011


namespace ninth_term_of_arithmetic_sequence_l1590_159069

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem ninth_term_of_arithmetic_sequence :
  ∀ (a₁ d : ℚ),
    a₁ = 2/3 →
    arithmetic_sequence a₁ d 17 = 3/2 →
    arithmetic_sequence a₁ d 9 = 13/12 :=
by
  sorry

end ninth_term_of_arithmetic_sequence_l1590_159069


namespace unique_good_days_count_l1590_159061

/-- Represents the change factor for an ingot on a given day type -/
structure IngotFactor where
  good : ℝ
  bad : ℝ

/-- Calculates the final value of an ingot after a week -/
def finalValue (factor : IngotFactor) (goodDays : ℕ) : ℝ :=
  factor.good ^ goodDays * factor.bad ^ (7 - goodDays)

/-- The problem statement -/
theorem unique_good_days_count :
  ∃! goodDays : ℕ,
    goodDays ≤ 7 ∧
    let goldFactor : IngotFactor := { good := 1.3, bad := 0.7 }
    let silverFactor : IngotFactor := { good := 1.2, bad := 0.8 }
    (finalValue goldFactor goodDays < 1 ∧ finalValue silverFactor goodDays > 1) :=
by sorry

end unique_good_days_count_l1590_159061


namespace unique_satisfying_function_l1590_159013

/-- A continuous monotonous function satisfying the given inequality -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  Monotone f ∧
  f 0 = 1 ∧
  ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1

/-- The main theorem stating that any function satisfying the conditions must be f(x) = x + 1 -/
theorem unique_satisfying_function (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end unique_satisfying_function_l1590_159013


namespace perimeter_of_square_d_l1590_159084

/-- Given two squares C and D, where C has a perimeter of 32 cm and D has an area that is half the area of C,
    prove that the perimeter of D is 16√2 cm. -/
theorem perimeter_of_square_d (C D : Real) : 
  (C = 32) →  -- Perimeter of square C is 32 cm
  (D^2 = (C/4)^2 / 2) →  -- Area of D is half the area of C
  (4 * D = 16 * Real.sqrt 2) :=  -- Perimeter of D is 16√2 cm
by sorry

end perimeter_of_square_d_l1590_159084


namespace correct_divisor_l1590_159003

theorem correct_divisor (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 63 →
  incorrect_quotient = 24 →
  correct_quotient = 42 →
  (incorrect_divisor * incorrect_quotient) / correct_quotient = 36 :=
by
  sorry

end correct_divisor_l1590_159003


namespace dihedral_angle_range_l1590_159058

/-- The dihedral angle between two adjacent lateral faces of a regular n-sided pyramid -/
def dihedral_angle (n : ℕ) (h : ℝ) : ℝ :=
  sorry

/-- The internal angle of a regular n-sided polygon -/
def internal_angle (n : ℕ) : ℝ :=
  sorry

theorem dihedral_angle_range (n : ℕ) (h : ℝ) :
  0 < dihedral_angle n h ∧ dihedral_angle n h < π :=
by sorry

end dihedral_angle_range_l1590_159058


namespace workers_paid_four_fifties_is_31_l1590_159016

/-- Represents the payment structure for workers -/
structure PaymentStructure where
  total_workers : Nat
  payment_per_worker : Nat
  hundred_bills : Nat
  fifty_bills : Nat
  workers_paid_two_hundreds : Nat

/-- Calculates the number of workers paid with four $50 bills -/
def workers_paid_four_fifties (p : PaymentStructure) : Nat :=
  let remaining_hundreds := p.hundred_bills - 2 * p.workers_paid_two_hundreds
  let workers_paid_mixed := remaining_hundreds
  let fifties_for_mixed := 2 * workers_paid_mixed
  let remaining_fifties := p.fifty_bills - fifties_for_mixed
  remaining_fifties / 4

/-- Theorem stating that given the specific payment structure, 31 workers are paid with four $50 bills -/
theorem workers_paid_four_fifties_is_31 :
  let p : PaymentStructure := {
    total_workers := 108,
    payment_per_worker := 200,
    hundred_bills := 122,
    fifty_bills := 188,
    workers_paid_two_hundreds := 45
  }
  workers_paid_four_fifties p = 31 := by
  sorry

end workers_paid_four_fifties_is_31_l1590_159016


namespace number_of_female_democrats_l1590_159021

theorem number_of_female_democrats
  (total : ℕ)
  (h_total : total = 780)
  (female : ℕ)
  (male : ℕ)
  (h_sum : female + male = total)
  (female_democrats : ℕ)
  (male_democrats : ℕ)
  (h_female_dem : female_democrats = female / 2)
  (h_male_dem : male_democrats = male / 4)
  (h_total_dem : female_democrats + male_democrats = total / 3) :
  female_democrats = 130 := by
sorry

end number_of_female_democrats_l1590_159021


namespace max_squares_covered_2inch_card_l1590_159099

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard -/
structure Checkerboard where
  square_size : ℝ

/-- Calculates the maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered_2inch_card (card : Card) (board : Checkerboard) :
  card.side_length = 2 → board.square_size = 1 → max_squares_covered card board = 9 :=
by sorry

end max_squares_covered_2inch_card_l1590_159099


namespace frances_towel_weight_frances_towel_weight_is_240_ounces_l1590_159047

/-- Calculates the weight of Frances's towels in ounces given the conditions of the beach towel problem -/
theorem frances_towel_weight (mary_towel_count : ℕ) (total_weight_pounds : ℕ) : ℕ :=
  let frances_towel_count := mary_towel_count / 4
  let total_weight_ounces := total_weight_pounds * 16
  let mary_towel_weight_ounces := (total_weight_ounces / mary_towel_count) * mary_towel_count
  let frances_towel_weight_ounces := total_weight_ounces - mary_towel_weight_ounces
  frances_towel_weight_ounces

/-- Proves that Frances's towels weigh 240 ounces given the conditions of the beach towel problem -/
theorem frances_towel_weight_is_240_ounces : frances_towel_weight 24 60 = 240 := by
  sorry

end frances_towel_weight_frances_towel_weight_is_240_ounces_l1590_159047


namespace algebraic_expression_proof_l1590_159081

theorem algebraic_expression_proof (a b : ℝ) : 
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2) = 2 * b^2 ∧
  3 * a^2 + (4 * a * (-2) - a^2) - 2 * (a^2 + 2 * a * (-2) - (-2)^2) = 8 :=
by sorry

end algebraic_expression_proof_l1590_159081


namespace tan_pi_plus_alpha_eq_two_implies_fraction_eq_three_l1590_159008

theorem tan_pi_plus_alpha_eq_two_implies_fraction_eq_three (α : Real) 
  (h : Real.tan (π + α) = 2) : 
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π - α)) = 3 := by
  sorry

end tan_pi_plus_alpha_eq_two_implies_fraction_eq_three_l1590_159008


namespace ball_hit_ground_time_l1590_159090

/-- The time when a ball thrown upward hits the ground -/
theorem ball_hit_ground_time : ∃ t : ℚ, t = 10 / 7 ∧ -4.9 * t^2 + 4 * t + 6 = 0 := by
  sorry

end ball_hit_ground_time_l1590_159090


namespace expression_value_l1590_159057

theorem expression_value (a b c : ℤ) (ha : a = 8) (hb : b = 10) (hc : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 := by
  sorry

end expression_value_l1590_159057


namespace min_sum_squares_l1590_159049

theorem min_sum_squares (a b c d e f g h : Int) : 
  a ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  b ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  c ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  d ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  e ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  f ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  g ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  h ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h →
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 98 := by
sorry

end min_sum_squares_l1590_159049


namespace correct_ages_l1590_159030

/-- Represents the ages of a family -/
structure FamilyAges where
  kareem : ℕ
  son : ℕ
  daughter : ℕ
  wife : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.kareem = 3 * ages.son ∧
  ages.daughter = ages.son / 2 ∧
  ages.kareem + 10 + ages.son + 10 + ages.daughter + 10 = 120 ∧
  ages.wife = ages.kareem - 8

/-- Theorem stating that the given ages satisfy the problem conditions -/
theorem correct_ages : 
  let ages : FamilyAges := ⟨60, 20, 10, 52⟩
  satisfiesConditions ages :=
by sorry

end correct_ages_l1590_159030


namespace product_of_binary_and_ternary_l1590_159079

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, x) acc => acc + if x then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldr (fun (i, x) acc => acc + x * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary_num := [true, false, true, true]
  let ternary_num := [2, 1, 2]
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 299 := by
sorry

end product_of_binary_and_ternary_l1590_159079


namespace road_trip_distance_l1590_159037

/-- Road trip problem -/
theorem road_trip_distance (total_time hours_driving friend_distance jenna_speed friend_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : hours_driving = total_time - 1)
  (h3 : friend_distance = 100)
  (h4 : jenna_speed = 50)
  (h5 : friend_speed = 20) :
  jenna_speed * (hours_driving - friend_distance / friend_speed) = 200 :=
by sorry

end road_trip_distance_l1590_159037


namespace not_divisible_by_100_l1590_159092

theorem not_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by sorry

end not_divisible_by_100_l1590_159092


namespace solve_magazine_problem_l1590_159045

def magazine_problem (cost_price selling_price gain : ℚ) : Prop :=
  ∃ (num_magazines : ℕ), 
    (selling_price - cost_price) * num_magazines = gain ∧
    num_magazines > 0

theorem solve_magazine_problem : 
  magazine_problem 3 3.5 5 → ∃ (num_magazines : ℕ), num_magazines = 10 :=
by
  sorry

end solve_magazine_problem_l1590_159045


namespace total_apples_is_340_l1590_159046

/-- The number of apples Kylie picked -/
def kylie_apples : ℕ := 66

/-- The number of apples Kayla picked -/
def kayla_apples : ℕ := 274

/-- The relationship between Kayla's and Kylie's apples -/
axiom kayla_kylie_relation : kayla_apples = 4 * kylie_apples + 10

/-- The total number of apples picked by Kylie and Kayla -/
def total_apples : ℕ := kylie_apples + kayla_apples

/-- Theorem: The total number of apples picked by Kylie and Kayla is 340 -/
theorem total_apples_is_340 : total_apples = 340 := by
  sorry

end total_apples_is_340_l1590_159046


namespace count_solutions_eq_4n_l1590_159029

/-- The number of integer solutions (x, y) for |x| + |y| = n -/
def count_solutions (n : ℕ) : ℕ :=
  4 * n

/-- Theorem: For any positive integer n, the number of integer solutions (x, y) 
    satisfying |x| + |y| = n is equal to 4n -/
theorem count_solutions_eq_4n (n : ℕ) (hn : n > 0) : 
  count_solutions n = 4 * n := by sorry

end count_solutions_eq_4n_l1590_159029


namespace gcd_84_210_l1590_159040

theorem gcd_84_210 : Nat.gcd 84 210 = 42 := by
  sorry

end gcd_84_210_l1590_159040


namespace boys_in_class_l1590_159054

/-- Proves that in a class with a 3:4 ratio of girls to boys and 35 total students, the number of boys is 20. -/
theorem boys_in_class (total_students : ℕ) (girls_to_boys_ratio : ℚ) : total_students = 35 → girls_to_boys_ratio = 3 / 4 → ∃ (boys : ℕ), boys = 20 := by
  sorry

end boys_in_class_l1590_159054


namespace m_range_l1590_159028

theorem m_range : 
  let m : ℝ := (-Real.sqrt 3 / 3) * (-2 * Real.sqrt 21)
  5 < m ∧ m < 6 := by
sorry

end m_range_l1590_159028


namespace king_midas_gold_l1590_159023

theorem king_midas_gold (x : ℝ) (h : x > 1) : 
  let initial_gold := 1
  let spent_fraction := 1 / x
  let remaining_gold := initial_gold - spent_fraction * initial_gold
  let needed_fraction := (initial_gold - remaining_gold) / remaining_gold
  needed_fraction = 1 / (x - 1) := by
sorry

end king_midas_gold_l1590_159023


namespace average_of_numbers_l1590_159096

def numbers : List ℕ := [12, 13, 14, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 125781 := by
  sorry

end average_of_numbers_l1590_159096


namespace not_equal_to_eighteen_fifths_other_options_equal_l1590_159089

theorem not_equal_to_eighteen_fifths : (18 + 1) / (5 + 1) ≠ 18 / 5 := by
  sorry

theorem other_options_equal :
  6^2 / 10 = 18 / 5 ∧
  (1 / 5) * (6 * 3) = 18 / 5 ∧
  3.6 = 18 / 5 ∧
  Real.sqrt (324 / 25) = 18 / 5 := by
  sorry

end not_equal_to_eighteen_fifths_other_options_equal_l1590_159089


namespace power_sum_equality_l1590_159019

theorem power_sum_equality : (-1 : ℤ) ^ (6^2) + (1 : ℤ) ^ (3^3) = 2 := by sorry

end power_sum_equality_l1590_159019


namespace complex_equation_solution_l1590_159053

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + 3 * Complex.I → z = 3 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l1590_159053


namespace workbook_arrangement_count_l1590_159097

/-- The number of ways to arrange 2 Korean and 2 English workbooks in a row with English workbooks side by side -/
def arrange_workbooks : ℕ :=
  let korean_books := 2
  let english_books := 2
  let total_units := korean_books + 1  -- English books count as one unit
  let unit_arrangements := Nat.factorial total_units
  let english_arrangements := Nat.factorial english_books
  unit_arrangements * english_arrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem workbook_arrangement_count : arrange_workbooks = 12 := by
  sorry

end workbook_arrangement_count_l1590_159097


namespace eraser_cost_tyler_eraser_cost_l1590_159000

/-- Calculates the cost of each eraser given Tyler's shopping scenario -/
theorem eraser_cost (initial_amount : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
  (eraser_count : ℕ) (remaining_amount : ℕ) : ℕ :=
  by
  sorry

/-- Proves that each eraser costs $4 in Tyler's specific scenario -/
theorem tyler_eraser_cost : eraser_cost 100 8 5 10 20 = 4 := by
  sorry

end eraser_cost_tyler_eraser_cost_l1590_159000


namespace roger_lawn_mowing_l1590_159093

/-- The number of lawns Roger had to mow -/
def total_lawns : ℕ := 14

/-- The amount Roger earns per lawn -/
def earnings_per_lawn : ℕ := 9

/-- The number of lawns Roger forgot to mow -/
def forgotten_lawns : ℕ := 8

/-- The total amount Roger actually earned -/
def actual_earnings : ℕ := 54

/-- Theorem stating that the total number of lawns Roger had to mow is 14 -/
theorem roger_lawn_mowing :
  total_lawns = (actual_earnings / earnings_per_lawn) + forgotten_lawns :=
by sorry

end roger_lawn_mowing_l1590_159093


namespace positive_rational_solution_condition_l1590_159095

theorem positive_rational_solution_condition 
  (a b : ℚ) (x y : ℚ) 
  (h_product : x * y = a) 
  (h_sum : x + y = b) : 
  (∃ (k : ℚ), k > 0 ∧ b^2 / 4 - a = k^2) ↔ 
  (x > 0 ∧ y > 0 ∧ ∃ (m n : ℕ), x = m / n) :=
sorry

end positive_rational_solution_condition_l1590_159095


namespace subtraction_inequality_l1590_159091

theorem subtraction_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : a - b < 0 := by
  sorry

end subtraction_inequality_l1590_159091


namespace bike_ride_time_l1590_159015

/-- Given a consistent bike riding speed where 1 mile takes 4 minutes,
    prove that the time required to ride 4.5 miles is 18 minutes. -/
theorem bike_ride_time (speed : ℝ) (distance_to_park : ℝ) : 
  speed = 1 / 4 →  -- Speed in miles per minute
  distance_to_park = 4.5 → -- Distance to park in miles
  distance_to_park / speed = 18 := by
  sorry

end bike_ride_time_l1590_159015


namespace club_truncator_season_probability_l1590_159055

/-- Represents the possible outcomes of a soccer match -/
inductive MatchResult
| Win
| Lose
| Tie

/-- Represents the season results for Club Truncator -/
structure SeasonResult :=
  (wins : ℕ)
  (losses : ℕ)
  (ties : ℕ)

/-- The number of teams in the league -/
def numTeams : ℕ := 8

/-- The number of matches Club Truncator plays -/
def numMatches : ℕ := 7

/-- The probability of winning a single match -/
def winProb : ℚ := 2/5

/-- The probability of losing a single match -/
def loseProb : ℚ := 1/5

/-- The probability of tying a single match -/
def tieProb : ℚ := 2/5

/-- Checks if a season result has more wins than losses -/
def moreWinsThanLosses (result : SeasonResult) : Prop :=
  result.wins > result.losses

/-- The probability of Club Truncator finishing with more wins than losses -/
def probMoreWinsThanLosses : ℚ := 897/2187

theorem club_truncator_season_probability :
  probMoreWinsThanLosses = 897/2187 := by sorry

end club_truncator_season_probability_l1590_159055


namespace new_barbell_cost_l1590_159036

theorem new_barbell_cost (old_cost : ℝ) (percentage_increase : ℝ) : 
  old_cost = 250 → percentage_increase = 0.3 → 
  old_cost + old_cost * percentage_increase = 325 := by
  sorry

end new_barbell_cost_l1590_159036


namespace quartic_sum_l1590_159002

theorem quartic_sum (f : ℝ → ℝ) :
  (∃ (a b c d : ℝ), ∀ x, f x = a*x^4 + b*x^3 + c*x^2 + d*x + (f 0)) →
  (f 1 = 10) →
  (f 2 = 20) →
  (f 3 = 30) →
  (f 10 + f (-6) = 8104) :=
by sorry

end quartic_sum_l1590_159002


namespace roger_actual_earnings_l1590_159098

/-- Calculates Roger's earnings from mowing lawns --/
def roger_earnings (small_price medium_price large_price : ℕ)
                   (total_small total_medium total_large : ℕ)
                   (forgot_small forgot_medium forgot_large : ℕ) : ℕ :=
  (small_price * (total_small - forgot_small)) +
  (medium_price * (total_medium - forgot_medium)) +
  (large_price * (total_large - forgot_large))

/-- Theorem: Roger's actual earnings are $69 --/
theorem roger_actual_earnings :
  roger_earnings 9 12 15 5 4 5 2 3 3 = 69 := by
  sorry

end roger_actual_earnings_l1590_159098


namespace best_approximation_l1590_159039

-- Define the function f(x) = x^2 - 3x - 4.6
def f (x : ℝ) : ℝ := x^2 - 3*x - 4.6

-- Define the table of values
def table : List (ℝ × ℝ) := [
  (-1.13, 4.67),
  (-1.12, 4.61),
  (-1.11, 4.56),
  (-1.10, 4.51),
  (-1.09, 4.46),
  (-1.08, 4.41),
  (-1.07, 4.35)
]

-- Define the given options
def options : List ℝ := [-1.073, -1.089, -1.117, -1.123]

-- Theorem statement
theorem best_approximation :
  ∃ (x : ℝ), x ∈ options ∧
  ∀ (y : ℝ), y ∈ options → |f x| ≤ |f y| ∧
  x = -1.117 := by
  sorry

end best_approximation_l1590_159039


namespace safety_gear_to_test_tube_ratio_l1590_159034

def total_budget : ℚ := 325
def flask_cost : ℚ := 150
def remaining_budget : ℚ := 25

def test_tube_cost : ℚ := (2/3) * flask_cost

def total_spent : ℚ := total_budget - remaining_budget

def safety_gear_cost : ℚ := total_spent - flask_cost - test_tube_cost

theorem safety_gear_to_test_tube_ratio :
  safety_gear_cost / test_tube_cost = 1/2 := by
  sorry

end safety_gear_to_test_tube_ratio_l1590_159034
