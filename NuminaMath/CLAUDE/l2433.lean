import Mathlib

namespace equation_solutions_l2433_243377

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5/2 ∧ x₂ = 4 ∧ 
  (∀ x : ℝ, 3*(2*x - 5) = (2*x - 5)^2 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solutions_l2433_243377


namespace equation_is_linear_l2433_243399

/-- A linear equation in two variables has the form ax + by = c, where a, b, and c are constants, and x and y are variables. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The equation 4x - y = 3 -/
def equation (x y : ℝ) : ℝ := 4 * x - y - 3

theorem equation_is_linear : IsLinearEquationInTwoVariables equation := by
  sorry


end equation_is_linear_l2433_243399


namespace john_walking_speed_l2433_243330

/-- The walking speed of John in km/h -/
def john_speed : ℝ := 6

/-- The biking speed of Joan in km/h -/
def joan_speed (js : ℝ) : ℝ := 2 * js

/-- The distance between home and school in km -/
def distance : ℝ := 3

/-- The time difference between John's and Joan's departure in hours -/
def time_difference : ℝ := 0.25

theorem john_walking_speed :
  ∃ (js : ℝ),
    js = john_speed ∧
    joan_speed js = 2 * js ∧
    distance / js = distance / (joan_speed js) + time_difference :=
by sorry

end john_walking_speed_l2433_243330


namespace sum_of_digits_of_3_to_17_l2433_243350

/-- The sum of the tens digit and the ones digit of (7-4)^17 is 9. -/
theorem sum_of_digits_of_3_to_17 : 
  (((7 - 4)^17 / 10) % 10 + (7 - 4)^17 % 10) = 9 := by
  sorry

end sum_of_digits_of_3_to_17_l2433_243350


namespace class_size_problem_l2433_243390

/-- Given information about class sizes, prove the size of Class C -/
theorem class_size_problem (class_b : ℕ) (h1 : class_b = 20) : ∃ (class_c : ℕ), class_c = 170 ∧
  ∃ (class_a class_d : ℕ),
    class_a = 2 * class_b ∧
    3 * class_a = class_c ∧
    class_d = 4 * class_a ∧
    class_c = class_d + 10 := by
  sorry

end class_size_problem_l2433_243390


namespace quadratic_real_roots_implies_m_leq_3_l2433_243381

theorem quadratic_real_roots_implies_m_leq_3 (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := by
  sorry

end quadratic_real_roots_implies_m_leq_3_l2433_243381


namespace y_value_l2433_243396

theorem y_value (x y : ℝ) (h1 : x^2 = y - 7) (h2 : x = 6) : y = 43 := by
  sorry

end y_value_l2433_243396


namespace sum_of_digits_M_l2433_243332

/-- A function that returns true if a number is a five-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The greatest five-digit number whose digits have a product of 72 -/
def M : ℕ := sorry

theorem sum_of_digits_M :
  is_five_digit M ∧
  digit_product M = 72 ∧
  (∀ n : ℕ, is_five_digit n → digit_product n = 72 → n ≤ M) →
  digit_sum M = 20 := by sorry

end sum_of_digits_M_l2433_243332


namespace train_crossing_contradiction_l2433_243394

theorem train_crossing_contradiction (V₁ V₂ L₁ L₂ T₂ : ℝ) : 
  V₁ > 0 → V₂ > 0 → L₁ > 0 → L₂ > 0 → T₂ > 0 →
  (L₁ / V₁ = 20) →  -- First train crosses man in 20 seconds
  (L₂ / V₂ = T₂) →  -- Second train crosses man in T₂ seconds
  ((L₁ + L₂) / (V₁ + V₂) = 19) →  -- Trains cross each other in 19 seconds
  (V₁ = V₂) →  -- Ratio of speeds is 1
  False :=  -- This leads to a contradiction
by
  sorry

#check train_crossing_contradiction

end train_crossing_contradiction_l2433_243394


namespace hannah_easter_eggs_l2433_243308

theorem hannah_easter_eggs (total : ℕ) (h : total = 63) :
  ∃ (helen : ℕ) (hannah : ℕ),
    hannah = 2 * helen ∧
    hannah + helen = total ∧
    hannah = 42 := by
  sorry

end hannah_easter_eggs_l2433_243308


namespace olympic_triathlon_distance_l2433_243316

theorem olympic_triathlon_distance :
  ∀ (cycling running swimming : ℝ),
  cycling = 4 * running →
  swimming = (3 / 80) * cycling →
  running - swimming = 8.5 →
  cycling + running + swimming = 51.5 := by
sorry

end olympic_triathlon_distance_l2433_243316


namespace digital_earth_capabilities_l2433_243331

-- Define the capabilities of Digital Earth
def can_simulate_environmental_impact : Prop := True
def can_monitor_crop_pests : Prop := True
def can_predict_submerged_areas : Prop := True
def can_simulate_past_environments : Prop := True

-- Define the statement to be proven false
def incorrect_statement : Prop :=
  ∃ (can_predict_future : Prop),
    can_predict_future ∧ ¬can_simulate_past_environments

-- Theorem statement
theorem digital_earth_capabilities :
  can_simulate_environmental_impact →
  can_monitor_crop_pests →
  can_predict_submerged_areas →
  can_simulate_past_environments →
  ¬incorrect_statement :=
by
  sorry

end digital_earth_capabilities_l2433_243331


namespace total_amount_is_correct_l2433_243340

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55
def apples_quantity : ℕ := 12
def apples_rate : ℕ := 80
def papayas_quantity : ℕ := 7
def papayas_rate : ℕ := 45
def oranges_quantity : ℕ := 15
def oranges_rate : ℕ := 30
def bananas_quantity : ℕ := 5
def bananas_rate : ℕ := 25

def total_amount : ℕ := 
  grapes_quantity * grapes_rate +
  mangoes_quantity * mangoes_rate +
  apples_quantity * apples_rate +
  papayas_quantity * papayas_rate +
  oranges_quantity * oranges_rate +
  bananas_quantity * bananas_rate

theorem total_amount_is_correct : total_amount = 3045 := by
  sorry

end total_amount_is_correct_l2433_243340


namespace cloth_worth_calculation_l2433_243391

/-- Represents the commission rate as a percentage -/
def commission_rate : ℚ := 25/10

/-- Represents the commission earned in rupees -/
def commission_earned : ℚ := 15

/-- Represents the worth of cloth sold -/
def cloth_worth : ℚ := 600

/-- Theorem stating that given the commission rate and earned commission, 
    the worth of cloth sold is 600 rupees -/
theorem cloth_worth_calculation : 
  commission_earned = (commission_rate / 100) * cloth_worth :=
sorry

end cloth_worth_calculation_l2433_243391


namespace f_4cos2alpha_equals_4_l2433_243373

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem f_4cos2alpha_equals_4 
  (f : ℝ → ℝ) (α : ℝ) 
  (h_odd : IsOdd f) 
  (h_period : HasPeriod f 5) 
  (h_f_neg3 : f (-3) = 4) 
  (h_sin_alpha : Real.sin α = Real.sqrt 3 / 2) : 
  f (4 * Real.cos (2 * α)) = 4 := by
sorry

end f_4cos2alpha_equals_4_l2433_243373


namespace solution_in_interval_monotonic_decreasing_range_two_roots_range_l2433_243321

-- Define the function f(x)
def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- Theorem 1
theorem solution_in_interval (k : ℝ) :
  ∃ x : ℝ, x ∈ Set.Ioo 0 2 ∧ f x k = k*x + 3 → x = Real.sqrt 2 :=
sorry

-- Theorem 2
theorem monotonic_decreasing_range (k : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Ioo 0 2 → y ∈ Set.Ioo 0 2 → x < y → f x k > f y k) →
  k ∈ Set.Iic (-8) :=
sorry

-- Theorem 3
theorem two_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Ioo 0 2 ∧ y ∈ Set.Ioo 0 2 ∧ x ≠ y ∧ f x k = 0 ∧ f y k = 0) →
  k ∈ Set.Ioo (-7/2) (-1) :=
sorry

end solution_in_interval_monotonic_decreasing_range_two_roots_range_l2433_243321


namespace min_value_product_l2433_243310

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 9) :
  x^4 * y^3 * z^2 ≥ 1/3456 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    1/x₀ + 1/y₀ + 1/z₀ = 9 ∧ 
    x₀^4 * y₀^3 * z₀^2 = 1/3456 := by
  sorry

end min_value_product_l2433_243310


namespace union_of_A_and_B_l2433_243392

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | (x - 1)^2 < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 3} := by sorry

end union_of_A_and_B_l2433_243392


namespace polynomial_identity_l2433_243309

theorem polynomial_identity : ∀ x : ℝ, 
  (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := by
  sorry

end polynomial_identity_l2433_243309


namespace total_shells_l2433_243382

/-- The amount of shells in Jovana's bucket -/
def shells_in_bucket (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating the total amount of shells in Jovana's bucket -/
theorem total_shells : shells_in_bucket 5 12 = 17 := by
  sorry

end total_shells_l2433_243382


namespace pen_purchase_problem_l2433_243370

/-- The problem of calculating the total number of pens purchased --/
theorem pen_purchase_problem (price_x price_y total_spent : ℚ) (num_x : ℕ) : 
  price_x = 4 → 
  price_y = (14/5 : ℚ) → 
  total_spent = 40 → 
  num_x = 8 → 
  ∃ (num_y : ℕ), num_x * price_x + num_y * price_y = total_spent ∧ num_x + num_y = 10 :=
by
  sorry


end pen_purchase_problem_l2433_243370


namespace sum_75_odd_numbers_l2433_243385

-- Define a function for the sum of first n odd numbers
def sum_odd_numbers (n : ℕ) : ℕ := n^2

-- State the theorem
theorem sum_75_odd_numbers :
  (sum_odd_numbers 50 = 2500) → (sum_odd_numbers 75 = 5625) :=
by
  sorry

end sum_75_odd_numbers_l2433_243385


namespace quadratic_equations_solutions_l2433_243363

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 8*x + 12 = 0 ↔ x = 6 ∨ x = 2) ∧
  (∀ x : ℝ, (x - 3)^2 = 2*x*(x - 3) ↔ x = 3 ∨ x = -3) := by
sorry

end quadratic_equations_solutions_l2433_243363


namespace bryan_has_more_candies_l2433_243301

-- Define the number of candies for Bryan and Ben
def bryan_skittles : ℕ := 50
def ben_mms : ℕ := 20

-- Theorem to prove Bryan has more candies and the difference is 30
theorem bryan_has_more_candies : 
  bryan_skittles > ben_mms ∧ bryan_skittles - ben_mms = 30 := by
  sorry

end bryan_has_more_candies_l2433_243301


namespace solution_set_l2433_243334

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
def condition1 : Prop := -Real.sqrt (1 / ((a - b)^2)) * (b - a) = 1
def condition2 : Prop := 3*x - 4*a ≤ a - 2*x
def condition3 : Prop := (3*x + 2*b) / 5 > b

-- State the theorem
theorem solution_set (h1 : condition1 a b) (h2 : condition2 a x) (h3 : condition3 b x) :
  b < x ∧ x ≤ a :=
sorry

end solution_set_l2433_243334


namespace triangle_height_l2433_243337

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 576 →
  base = 32 →
  area = (base * height) / 2 →
  height = 36 := by
sorry

end triangle_height_l2433_243337


namespace digit_squaring_l2433_243346

theorem digit_squaring (A B C : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ B ≠ C →
  (A + 1 > 1) →
  (A * (A + 1)^3 + A * (A + 1)^2 + A * (A + 1) + A)^2 = 
    A * (A + 1)^7 + A * (A + 1)^6 + A * (A + 1)^5 + B * (A + 1)^4 + C * (A + 1)^3 + C * (A + 1)^2 + C * (A + 1) + B →
  A = 2 ∧ B = 1 ∧ C = 0 := by
sorry

end digit_squaring_l2433_243346


namespace pizza_total_slices_l2433_243326

def pizza_problem (john_slices sam_slices remaining_slices : ℕ) : Prop :=
  john_slices = 3 ∧
  sam_slices = 2 * john_slices ∧
  remaining_slices = 3

theorem pizza_total_slices 
  (john_slices sam_slices remaining_slices : ℕ) 
  (h : pizza_problem john_slices sam_slices remaining_slices) : 
  john_slices + sam_slices + remaining_slices = 12 :=
by
  sorry

#check pizza_total_slices

end pizza_total_slices_l2433_243326


namespace problem_statement_l2433_243376

theorem problem_statement (x y : ℝ) 
  (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) :
  |x| + |y| = 3/2 := by
sorry

end problem_statement_l2433_243376


namespace area_BPQ_is_six_l2433_243333

/-- Rectangle ABCD with length 8 and width 6, diagonal AC divided into 4 equal segments by P, Q, R -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (diagonal_segments : ℕ)

/-- The area of triangle BPQ in the given rectangle -/
def area_BPQ (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle BPQ is 6 square inches -/
theorem area_BPQ_is_six (rect : Rectangle) 
  (h1 : rect.length = 8)
  (h2 : rect.width = 6)
  (h3 : rect.diagonal_segments = 4) : 
  area_BPQ rect = 6 :=
sorry

end area_BPQ_is_six_l2433_243333


namespace parabola_directrix_l2433_243365

/-- Definition of a parabola with equation y^2 = 6x -/
def parabola (x y : ℝ) : Prop := y^2 = 6*x

/-- Definition of the directrix of a parabola -/
def directrix (x : ℝ) : Prop := x = -3/2

/-- Theorem: The directrix of the parabola y^2 = 6x is x = -3/2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → directrix x :=
by sorry

end parabola_directrix_l2433_243365


namespace circles_have_three_common_tangents_l2433_243389

/-- Circle C₁ with equation x² + y² + 2x + 4y + 1 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- The number of common tangents between C₁ and C₂ -/
def num_common_tangents : ℕ := 3

theorem circles_have_three_common_tangents :
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  (∀ (x y : ℝ), C₁ x y ∨ C₂ x y → n = 3) :=
sorry

end circles_have_three_common_tangents_l2433_243389


namespace break_even_items_l2433_243338

/-- The cost of producing each item is inversely proportional to the square root of the number of items produced. -/
def cost_production_relation (C N : ℝ) : Prop :=
  ∃ k : ℝ, C * (N^(1/2 : ℝ)) = k

/-- The cost of producing 10 items is $2100. -/
def cost_10_items : ℝ := 2100

/-- The selling price per item is $30. -/
def selling_price : ℝ := 30

/-- The break-even condition: total revenue equals total cost. -/
def break_even (N : ℝ) : Prop :=
  selling_price * N = cost_10_items * (10^(1/2 : ℝ)) / (N^(1/2 : ℝ))

/-- The number of items needed to break even is 10 * ∛49. -/
theorem break_even_items :
  ∃ N : ℝ, break_even N ∧ N = 10 * (49^(1/3 : ℝ)) :=
sorry

end break_even_items_l2433_243338


namespace probability_at_least_one_correct_l2433_243358

theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 6 → choices = 6 → 
  1 - (1 - 1 / choices : ℚ) ^ n = 31031 / 46656 := by
  sorry

end probability_at_least_one_correct_l2433_243358


namespace arithmetic_mean_of_fractions_l2433_243322

theorem arithmetic_mean_of_fractions : 
  (3 / 8 + 5 / 12) / 2 = 19 / 48 := by sorry

end arithmetic_mean_of_fractions_l2433_243322


namespace problem_solution_l2433_243319

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^5 + (Real.log y / Real.log 5)^5 + 10 = 
       10 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 4^(2*5^(1/5)) + 5^(2*5^(1/5)) := by
  sorry

end problem_solution_l2433_243319


namespace min_value_of_expression_existence_of_minimum_l2433_243323

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 ≥ 2039 :=
sorry

theorem existence_of_minimum : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end min_value_of_expression_existence_of_minimum_l2433_243323


namespace picture_books_count_l2433_243328

theorem picture_books_count (total : ℕ) (fiction : ℕ) (non_fiction : ℕ) (autobiographies : ℕ) (picture : ℕ) : 
  total = 35 →
  fiction = 5 →
  non_fiction = fiction + 4 →
  autobiographies = 2 * fiction →
  total = fiction + non_fiction + autobiographies + picture →
  picture = 11 := by
sorry

end picture_books_count_l2433_243328


namespace cyclist_speed_problem_l2433_243349

theorem cyclist_speed_problem (distance : ℝ) (speed_diff : ℝ) (time_diff : ℝ) :
  distance = 195 →
  speed_diff = 4 →
  time_diff = 1 →
  ∃ (v : ℝ),
    v > 0 ∧
    distance / v = distance / (v - speed_diff) - time_diff ∧
    v = 30 :=
by sorry

end cyclist_speed_problem_l2433_243349


namespace least_addition_for_divisibility_l2433_243302

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ k : ℕ, k < 9 → ¬(11 ∣ (11002 + k))) ∧ (11 ∣ (11002 + 9)) := by
  sorry

end least_addition_for_divisibility_l2433_243302


namespace log_expression_equals_negative_four_l2433_243320

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_four :
  (lg 8 + lg 125 - lg 2 - lg 5) / (lg (Real.sqrt 10) * lg 0.1) = -4 := by
  sorry

end log_expression_equals_negative_four_l2433_243320


namespace school_club_profit_l2433_243324

/-- Calculates the profit for a school club selling granola bars -/
theorem school_club_profit : 
  ∀ (total_bars : ℕ) 
    (buy_price : ℚ) 
    (buy_quantity : ℕ) 
    (sell_price : ℚ) 
    (sell_quantity : ℕ),
  total_bars = 1200 →
  buy_price = 3/2 →
  buy_quantity = 3 →
  sell_price = 12/5 →
  sell_quantity = 4 →
  (total_bars : ℚ) * (sell_price / sell_quantity) - 
  (total_bars : ℚ) * (buy_price / buy_quantity) = 120 := by
sorry


end school_club_profit_l2433_243324


namespace function_inequality_relation_l2433_243354

theorem function_inequality_relation (f : ℝ → ℝ) (a b : ℝ) :
  (f = λ x => 3 * x + 1) →
  (a > 0 ∧ b > 0) →
  (∀ x, |x - 1| < b → |f x - 4| < a) →
  a - 3 * b ≥ 0 := by sorry

end function_inequality_relation_l2433_243354


namespace rationalize_denominator_l2433_243345

theorem rationalize_denominator : 
  1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by sorry

end rationalize_denominator_l2433_243345


namespace product_of_divisors_1024_l2433_243357

/-- The product of divisors of a positive integer -/
def product_of_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem: If the product of divisors of n is 1024, then n = 16 -/
theorem product_of_divisors_1024 (n : ℕ+) :
  product_of_divisors n = 1024 → n = 16 := by sorry

end product_of_divisors_1024_l2433_243357


namespace blue_marble_probability_l2433_243371

theorem blue_marble_probability : 
  ∀ (total yellow green red blue : ℕ),
    total = 60 →
    yellow = 20 →
    green = yellow / 2 →
    red = blue →
    total = yellow + green + red + blue →
    (blue : ℚ) / total * 100 = 25 :=
by
  sorry

end blue_marble_probability_l2433_243371


namespace max_individual_points_is_23_l2433_243374

/-- Represents a basketball team -/
structure BasketballTeam where
  players : Nat
  totalPoints : Nat
  minPointsPerPlayer : Nat

/-- Calculates the maximum points a single player could have scored -/
def maxIndividualPoints (team : BasketballTeam) : Nat :=
  team.totalPoints - (team.players - 1) * team.minPointsPerPlayer

/-- Theorem: The maximum points an individual player could have scored is 23 -/
theorem max_individual_points_is_23 (team : BasketballTeam) 
  (h1 : team.players = 12)
  (h2 : team.totalPoints = 100)
  (h3 : team.minPointsPerPlayer = 7) :
  maxIndividualPoints team = 23 := by
  sorry

#eval maxIndividualPoints ⟨12, 100, 7⟩

end max_individual_points_is_23_l2433_243374


namespace quadratic_equation_root_zero_l2433_243329

theorem quadratic_equation_root_zero (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + x + a^2 - 1 = 0) ∧
  ((a - 1) * 0^2 + 0 + a^2 - 1 = 0) ∧
  (a - 1 ≠ 0) →
  a = -1 := by
sorry

end quadratic_equation_root_zero_l2433_243329


namespace simplify_and_evaluate_l2433_243313

theorem simplify_and_evaluate (x y : ℤ) (A B : ℤ) (h1 : A = 2*x + y) (h2 : B = 2*x - y) (h3 : x = -1) (h4 : y = 2) :
  (A^2 - B^2) * (x - 2*y) = 80 := by
  sorry

end simplify_and_evaluate_l2433_243313


namespace parallel_vectors_iff_m_values_l2433_243348

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The vector a as a function of m -/
def a (m : ℝ) : ℝ × ℝ := (2*m + 1, 3)

/-- The vector b as a function of m -/
def b (m : ℝ) : ℝ × ℝ := (2, m)

/-- Theorem stating that vectors a and b are parallel if and only if m = 3/2 or m = -2 -/
theorem parallel_vectors_iff_m_values :
  ∀ m : ℝ, are_parallel (a m) (b m) ↔ m = 3/2 ∨ m = -2 := by sorry

end parallel_vectors_iff_m_values_l2433_243348


namespace x_value_l2433_243347

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.1 * 500 - 5) ∧ (x = 180) := by
  sorry

end x_value_l2433_243347


namespace samantha_birthday_next_monday_l2433_243367

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Calculates the day of the week for June 18 in a given year, 
    given the day of the week for June 18 in the previous year -/
def nextJune18 (prevDay : DayOfWeek) (year : Nat) : DayOfWeek :=
  sorry

/-- Finds the next year when June 18 falls on a Monday, given a starting year and day -/
def nextMondayJune18 (startYear : Nat) (startDay : DayOfWeek) : Nat :=
  sorry

theorem samantha_birthday_next_monday (startYear : Nat) (startDay : DayOfWeek) :
  startYear = 2009 →
  startDay = DayOfWeek.Friday →
  ¬isLeapYear startYear →
  nextMondayJune18 startYear startDay = 2017 :=
sorry

end samantha_birthday_next_monday_l2433_243367


namespace total_sheets_is_114_l2433_243372

/-- The number of bundles of colored paper -/
def coloredBundles : ℕ := 3

/-- The number of bunches of white paper -/
def whiteBunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrapHeaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheetsPerBunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheetsPerBundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheetsPerHeap : ℕ := 20

/-- The total number of sheets of paper removed from the chest of drawers -/
def totalSheets : ℕ := coloredBundles * sheetsPerBundle + whiteBunches * sheetsPerBunch + scrapHeaps * sheetsPerHeap

theorem total_sheets_is_114 : totalSheets = 114 := by
  sorry

end total_sheets_is_114_l2433_243372


namespace sum_sqrt_inequality_l2433_243387

theorem sum_sqrt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (a / (a + 8)) + Real.sqrt (b / (b + 8)) + Real.sqrt (c / (c + 8)) ≥ 1 := by
  sorry

end sum_sqrt_inequality_l2433_243387


namespace experiments_to_target_reduction_l2433_243379

/-- The factor by which the range is reduced after each experiment -/
def reduction_factor : ℝ := 0.618

/-- The target reduction of the range -/
def target_reduction : ℝ := 0.618^4

/-- The number of experiments needed to reach the target reduction -/
def num_experiments : ℕ := 4

/-- Theorem stating that the number of experiments needed to reach the target reduction is correct -/
theorem experiments_to_target_reduction :
  (reduction_factor ^ num_experiments) = target_reduction :=
by sorry

end experiments_to_target_reduction_l2433_243379


namespace triangle_area_l2433_243388

/-- The area of a triangle with perimeter 32 and inradius 2.5 is 40 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h1 : perimeter = 32) 
  (h2 : inradius = 2.5) 
  (h3 : area = inradius * (perimeter / 2)) : 
  area = 40 := by
sorry

end triangle_area_l2433_243388


namespace arithmetic_sequence_ninth_term_l2433_243318

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 5 and the 6th term is 11, the 9th term is 17. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third : a 3 = 5)
  (h_sixth : a 6 = 11) :
  a 9 = 17 := by
  sorry


end arithmetic_sequence_ninth_term_l2433_243318


namespace guitar_price_theorem_l2433_243398

-- Define the suggested retail price
variable (P : ℝ)

-- Define the prices at Guitar Center and Sweetwater
def guitar_center_price (P : ℝ) : ℝ := 0.85 * P + 100
def sweetwater_price (P : ℝ) : ℝ := 0.90 * P

-- State the theorem
theorem guitar_price_theorem (h : abs (guitar_center_price P - sweetwater_price P) = 50) : 
  P = 1000 := by
  sorry

end guitar_price_theorem_l2433_243398


namespace chipped_marbles_are_36_l2433_243361

def marble_bags : List Nat := [16, 18, 22, 24, 26, 30, 36]

structure MarbleDistribution where
  jane_bags : List Nat
  george_bags : List Nat
  chipped_bag : Nat

def is_valid_distribution (d : MarbleDistribution) : Prop :=
  d.jane_bags.length = 4 ∧
  d.george_bags.length = 2 ∧
  d.chipped_bag ∈ marble_bags ∧
  d.jane_bags.sum = 3 * d.george_bags.sum ∧
  (∀ b ∈ d.jane_bags ++ d.george_bags, b ≠ d.chipped_bag) ∧
  (∀ b ∈ marble_bags, b ∉ d.jane_bags → b ∉ d.george_bags → b = d.chipped_bag)

theorem chipped_marbles_are_36 :
  ∀ d : MarbleDistribution, is_valid_distribution d → d.chipped_bag = 36 := by
  sorry

end chipped_marbles_are_36_l2433_243361


namespace real_part_of_inverse_l2433_243315

theorem real_part_of_inverse (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1 / 4 := by
  sorry

end real_part_of_inverse_l2433_243315


namespace sin_300_degrees_l2433_243369

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l2433_243369


namespace min_perimeter_isosceles_triangles_l2433_243339

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- Checks if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.side ≠ t2.side ∨ t1.base ≠ t2.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 4 : ℝ) * Real.sqrt (4 * t.side^2 - t.base^2)

/-- Theorem: Minimum perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 1180 :=
  sorry

end min_perimeter_isosceles_triangles_l2433_243339


namespace A_intersect_B_l2433_243362

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {y | ∃ x ∈ A, y = Real.exp x}

theorem A_intersect_B : A ∩ B = {1} := by
  sorry

end A_intersect_B_l2433_243362


namespace zhong_is_symmetrical_l2433_243304

/-- A Chinese character is represented as a structure with left and right sides -/
structure ChineseCharacter where
  left : String
  right : String

/-- A function to check if a character is symmetrical -/
def isSymmetrical (c : ChineseCharacter) : Prop :=
  c.left = c.right

/-- The Chinese character "中" -/
def zhong : ChineseCharacter :=
  { left := "|", right := "|" }

/-- Theorem stating that "中" is symmetrical -/
theorem zhong_is_symmetrical : isSymmetrical zhong := by
  sorry


end zhong_is_symmetrical_l2433_243304


namespace least_three_digit_multiple_of_eight_l2433_243360

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, 100 ≤ n ∧ n < 104 → ¬(n % 8 = 0)) ∧ 
  104 % 8 = 0 ∧ 
  104 ≥ 100 ∧ 
  104 < 1000 := by
sorry

end least_three_digit_multiple_of_eight_l2433_243360


namespace medical_team_selection_l2433_243393

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of internists available. -/
def num_internists : ℕ := 5

/-- The number of surgeons available. -/
def num_surgeons : ℕ := 6

/-- The total number of doctors needed for the team. -/
def team_size : ℕ := 4

/-- The number of ways to select the medical team. -/
def select_team : ℕ :=
  choose num_internists 1 * choose num_surgeons 3 +
  choose num_internists 2 * choose num_surgeons 2 +
  choose num_internists 3 * choose num_surgeons 1

theorem medical_team_selection :
  select_team = 310 := by sorry

end medical_team_selection_l2433_243393


namespace count_distinct_tetrahedrons_l2433_243356

/-- The number of distinct tetrahedrons that can be painted with n colors, 
    where each face is painted with exactly one color. -/
def distinctTetrahedrons (n : ℕ) : ℕ :=
  n * ((n-1)*(n-2)*(n-3)/12 + (n-1)*(n-2)/3 + (n-1) + 1)

/-- Theorem stating the number of distinct tetrahedrons that can be painted 
    with n colors, where n ≥ 4 and each face is painted with exactly one color. -/
theorem count_distinct_tetrahedrons (n : ℕ) (h : n ≥ 4) : 
  distinctTetrahedrons n = n * ((n-1)*(n-2)*(n-3)/12 + (n-1)*(n-2)/3 + (n-1) + 1) :=
by
  sorry

#check count_distinct_tetrahedrons

end count_distinct_tetrahedrons_l2433_243356


namespace plan_y_more_cost_effective_l2433_243300

/-- Represents the cost of Plan X in cents for y gigabytes of data -/
def plan_x_cost (y : ℝ) : ℝ := 25 * y

/-- Represents the cost of Plan Y in cents for y gigabytes of data -/
def plan_y_cost (y : ℝ) : ℝ := 1500 + 15 * y

/-- The minimum number of gigabytes for Plan Y to be more cost-effective -/
def min_gb_for_plan_y : ℝ := 150

theorem plan_y_more_cost_effective :
  ∀ y : ℝ, y ≥ min_gb_for_plan_y → plan_y_cost y < plan_x_cost y :=
by sorry

end plan_y_more_cost_effective_l2433_243300


namespace power_equality_l2433_243384

theorem power_equality (x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) :
  3^x * 4^y = 531441 := by
sorry

end power_equality_l2433_243384


namespace unique_solution_power_equation_l2433_243380

theorem unique_solution_power_equation :
  ∃! (n k l m : ℕ), l > 1 ∧ (1 + n^k)^l = 1 + n^m ∧ n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 := by
  sorry

end unique_solution_power_equation_l2433_243380


namespace max_take_home_pay_l2433_243386

-- Define the income function
def income (y : ℝ) : ℝ := 100 * y^2

-- Define the tax function
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay function
def takeHomePay (y : ℝ) : ℝ := income y - tax y

-- Theorem statement
theorem max_take_home_pay :
  ∃ y : ℝ, y > 0 ∧ 
    (∀ z : ℝ, z > 0 → takeHomePay z ≤ takeHomePay y) ∧
    income y = 250000 := by sorry

end max_take_home_pay_l2433_243386


namespace tan_alpha_value_l2433_243364

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - Real.cos α) / (3 * Real.sin α + Real.cos α) = 1/7) : 
  Real.tan α = 2 := by
  sorry

end tan_alpha_value_l2433_243364


namespace stating_cat_purchase_possible_l2433_243378

/-- Represents the available denominations of rubles --/
def denominations : List ℕ := [1, 5, 10, 50, 100, 500, 1000]

/-- Represents the total amount of money available --/
def total_money : ℕ := 1999

/-- 
Theorem stating that for any price of the cat, 
the buyer can make the purchase and receive correct change
--/
theorem cat_purchase_possible :
  ∀ (price : ℕ), price ≤ total_money →
  ∃ (buyer_money seller_money : List ℕ),
    (buyer_money.sum = price) ∧
    (seller_money.sum = total_money - price) ∧
    (∀ x ∈ buyer_money ∪ seller_money, x ∈ denominations) :=
by sorry

end stating_cat_purchase_possible_l2433_243378


namespace base7_to_base10_conversion_l2433_243352

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [4, 6, 5, 7, 3]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 9895 := by
  sorry

end base7_to_base10_conversion_l2433_243352


namespace betty_calculation_l2433_243375

theorem betty_calculation : ∀ (x y : ℚ),
  x = 8/100 →
  y = 325/100 →
  (x * y : ℚ) = 26/100 :=
by
  sorry

end betty_calculation_l2433_243375


namespace num_lines_eq_60_l2433_243312

def coefficients : Finset ℕ := {1, 3, 5, 7, 9}

/-- The number of different lines formed by the equation Ax + By + C = 0,
    where A, B, and C are distinct elements from the set {1, 3, 5, 7, 9} -/
def num_lines : ℕ :=
  (coefficients.card) * (coefficients.card - 1) * (coefficients.card - 2)

theorem num_lines_eq_60 : num_lines = 60 := by
  sorry

end num_lines_eq_60_l2433_243312


namespace complex_equation_solution_l2433_243395

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2) : z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l2433_243395


namespace polynomial_invariant_under_increment_l2433_243368

def P (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x

theorem polynomial_invariant_under_increment :
  ∀ x : ℝ, P x = P (x + 1) ↔ x = 1 ∨ x = 4/3 := by sorry

end polynomial_invariant_under_increment_l2433_243368


namespace product_of_specific_integers_l2433_243305

theorem product_of_specific_integers : 
  ∃ (a b : ℤ), 
    a = 32 ∧ 
    b = 3125 ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    a * b = 100000 := by
  sorry

end product_of_specific_integers_l2433_243305


namespace expression_equals_one_l2433_243366

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  let x := b^2 + c^2 - b - c + 1 + b*c
  (a^2*b^2) / (x^2) + (a^2*c^2) / (x^2) + (b^2*c^2) / (x^2) = 1 := by
sorry

end expression_equals_one_l2433_243366


namespace factorization_x3_plus_5x_l2433_243343

theorem factorization_x3_plus_5x (x : ℂ) : x^3 + 5*x = x * (x - Complex.I * Real.sqrt 5) * (x + Complex.I * Real.sqrt 5) := by
  sorry

end factorization_x3_plus_5x_l2433_243343


namespace two_digit_number_difference_l2433_243314

def digits : Finset Nat := {1, 4, 7, 9}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

def largest_number : Nat := 97
def smallest_number : Nat := 14

theorem two_digit_number_difference :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number - smallest_number = 83 := by
  sorry

end two_digit_number_difference_l2433_243314


namespace four_squares_power_of_two_l2433_243307

def count_four_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 0

theorem four_squares_power_of_two (n : ℕ) :
  count_four_squares n = (Nat.card {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a^2 + b^2 + c^2 + d^2 = 2^n}) :=
sorry

end four_squares_power_of_two_l2433_243307


namespace quadrilateral_equal_sides_is_rhombus_l2433_243353

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem: A quadrilateral with all sides equal is a rhombus
theorem quadrilateral_equal_sides_is_rhombus (q : Quadrilateral) :
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d → is_rhombus q :=
by
  sorry

end quadrilateral_equal_sides_is_rhombus_l2433_243353


namespace complex_modulus_range_l2433_243383

theorem complex_modulus_range (z : ℂ) (a : ℝ) :
  z = 3 + a * Complex.I ∧ Complex.abs z < 4 →
  a ∈ Set.Ioo (-Real.sqrt 7) (Real.sqrt 7) := by
sorry

end complex_modulus_range_l2433_243383


namespace jellybean_guess_difference_l2433_243344

/-- The jellybean guessing problem -/
theorem jellybean_guess_difference :
  ∀ (guess1 guess2 guess3 guess4 : ℕ),
  guess1 = 100 →
  guess2 = 8 * guess1 →
  guess3 < guess2 →
  guess4 = (guess1 + guess2 + guess3) / 3 + 25 →
  guess4 = 525 →
  guess2 - guess3 = 200 :=
by
  sorry

end jellybean_guess_difference_l2433_243344


namespace reciprocal_of_point_B_is_one_l2433_243327

-- Define the position of point A on the number line
def point_A : ℝ := -3

-- Define the distance between point A and point B
def distance_AB : ℝ := 4

-- Define the position of point B on the number line
def point_B : ℝ := point_A + distance_AB

-- Theorem to prove
theorem reciprocal_of_point_B_is_one : 
  (1 : ℝ) / point_B = 1 := by sorry

end reciprocal_of_point_B_is_one_l2433_243327


namespace cube_third_times_eighth_equals_one_over_216_l2433_243351

theorem cube_third_times_eighth_equals_one_over_216 :
  (1 / 3 : ℚ)^3 * (1 / 8 : ℚ) = 1 / 216 := by
  sorry

end cube_third_times_eighth_equals_one_over_216_l2433_243351


namespace min_dot_product_l2433_243342

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

/-- The circle C equation -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- A point on the circle C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : circle_C x y

/-- The dot product of tangent vectors PA and PB -/
def dot_product (P : PointOnC) (A B : ℝ × ℝ) : ℝ :=
  let PA := (A.1 - P.x, A.2 - P.y)
  let PB := (B.1 - P.x, B.2 - P.y)
  PA.1 * PB.1 + PA.2 * PB.2

/-- The theorem statement -/
theorem min_dot_product :
  ∀ P : PointOnC, ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    dot_product P A B ≥ 18 * Real.sqrt 2 - 27 :=
sorry

end min_dot_product_l2433_243342


namespace comparison_inequality_l2433_243341

theorem comparison_inequality : ∀ x : ℝ, (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry

end comparison_inequality_l2433_243341


namespace last_round_probability_l2433_243311

/-- A tournament with the given conditions -/
structure Tournament (n : ℕ) where
  num_players : ℕ := 2^(n+1)
  num_rounds : ℕ := n+1
  pairing : Unit  -- Represents the pairing process
  pushover_game : Unit  -- Represents the Pushover game

/-- The probability of two specific players facing each other in the last round -/
def face_probability (t : Tournament n) : ℚ :=
  (2^n - 1) / 8^n

/-- Theorem stating the probability of players 1 and 2^n facing each other in the last round -/
theorem last_round_probability (n : ℕ) (h : n > 0) :
  ∀ (t : Tournament n), face_probability t = (2^n - 1) / 8^n :=
sorry

end last_round_probability_l2433_243311


namespace hair_cut_first_day_l2433_243355

/-- The amount of hair cut off on the first day, given the total amount cut off and the amount cut off on the second day. -/
theorem hair_cut_first_day (total : ℚ) (second_day : ℚ) (h1 : total = 0.875) (h2 : second_day = 0.5) :
  total - second_day = 0.375 := by
  sorry

end hair_cut_first_day_l2433_243355


namespace base8_to_base7_conversion_l2433_243397

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The given number in base 8 -/
def givenNumber : ℕ := 653

/-- The expected result in base 7 -/
def expectedResult : ℕ := 1150

theorem base8_to_base7_conversion :
  base10ToBase7 (base8ToBase10 givenNumber) = expectedResult := by
  sorry

end base8_to_base7_conversion_l2433_243397


namespace max_quotient_value_l2433_243336

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 1200 ≤ b ∧ b ≤ 2400) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 1200 ≤ y ∧ y ≤ 2400 → y / x ≤ b / a) →
  b / a = 24 :=
by sorry

end max_quotient_value_l2433_243336


namespace origin_inside_ellipse_iff_k_range_l2433_243303

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- A point (x,y) is inside the ellipse if the left side of the equation is negative -/
def inside_ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 < 0

theorem origin_inside_ellipse_iff_k_range :
  ∀ k : ℝ, inside_ellipse k 0 0 ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end origin_inside_ellipse_iff_k_range_l2433_243303


namespace speeds_satisfy_conditions_l2433_243317

/-- The speed of person A in km/h -/
def speed_A : ℝ := 3.6

/-- The speed of person B in km/h -/
def speed_B : ℝ := 6

/-- The total distance between the starting points of person A and person B in km -/
def total_distance : ℝ := 36

/-- Theorem stating that the given speeds satisfy the conditions of the problem -/
theorem speeds_satisfy_conditions :
  (5 * speed_A + 3 * speed_B = total_distance) ∧
  (2.5 * speed_A + 4.5 * speed_B = total_distance) :=
by sorry

end speeds_satisfy_conditions_l2433_243317


namespace inequality_equivalence_l2433_243325

theorem inequality_equivalence (x : ℝ) :
  (1 / Real.sqrt (1 - x) - 1 / Real.sqrt (1 + x) ≥ 1) ↔ 
  (Real.sqrt (2 * Real.sqrt 3 - 3) ≤ x ∧ x < 1) :=
by sorry

end inequality_equivalence_l2433_243325


namespace total_spent_is_correct_l2433_243359

def original_cost : ℝ := 1200
def discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.08

def total_spent : ℝ :=
  let discounted_cost := original_cost * (1 - discount_rate)
  let other_toys_with_tax := discounted_cost * (1 + tax_rate)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_with_tax := lightsaber_cost * (1 + tax_rate)
  other_toys_with_tax + lightsaber_with_tax

theorem total_spent_is_correct :
  total_spent = 3628.80 := by sorry

end total_spent_is_correct_l2433_243359


namespace certain_number_value_l2433_243306

theorem certain_number_value (x p n : ℕ) (h1 : x > 0) (h2 : Prime p) 
  (h3 : ∃ k : ℕ, Prime k ∧ Even k ∧ x = k * n * p) (h4 : x ≥ 44) 
  (h5 : ∀ y, y > 0 → y < x → ¬∃ k : ℕ, Prime k ∧ Even k ∧ y = k * n * p) : n = 2 := by
  sorry

end certain_number_value_l2433_243306


namespace money_distribution_l2433_243335

/-- Given that p, q, and r have $9000 among themselves, and r has two-thirds of the total amount with p and q, prove that r has $3600. -/
theorem money_distribution (p q r : ℝ) 
  (total : p + q + r = 9000)
  (r_proportion : r = (2/3) * (p + q)) :
  r = 3600 := by
  sorry

end money_distribution_l2433_243335
