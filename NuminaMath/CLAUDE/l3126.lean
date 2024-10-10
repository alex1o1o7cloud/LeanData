import Mathlib

namespace fractional_equation_solution_l3126_312616

theorem fractional_equation_solution :
  ∃ x : ℝ, (x * (x - 2) ≠ 0) ∧ (4 / (x - 2) = 2 / x) ∧ (x = -2) := by
  sorry

end fractional_equation_solution_l3126_312616


namespace not_R_intersection_A_B_l3126_312686

def set_A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def set_B : Set ℝ := {x | x - 2 > 0}

theorem not_R_intersection_A_B :
  (set_A ∩ set_B)ᶜ = {x : ℝ | x ≤ 2 ∨ x > 3} := by sorry

end not_R_intersection_A_B_l3126_312686


namespace isosceles_triangle_area_l3126_312675

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 → 
  2 * s + 2 * b = 40 → -- perimeter condition
  b ^ 2 + 10 ^ 2 = s ^ 2 → -- Pythagorean theorem
  (2 * b) * 10 / 2 = 75 := by 
sorry

end isosceles_triangle_area_l3126_312675


namespace transformed_area_is_450_l3126_312698

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -8, 6]

-- Define the original area
def original_area : ℝ := 9

-- Theorem statement
theorem transformed_area_is_450 :
  let det := A.det
  let new_area := original_area * |det|
  new_area = 450 := by sorry

end transformed_area_is_450_l3126_312698


namespace inequality_problem_l3126_312648

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) 
  (h4 : c < d) (h5 : d < 0) : 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end inequality_problem_l3126_312648


namespace days_to_pay_cash_register_l3126_312663

/-- Represents the financial data for Marie's bakery --/
structure BakeryFinances where
  cash_register_cost : ℕ
  bread_price : ℕ
  bread_quantity : ℕ
  cake_price : ℕ
  cake_quantity : ℕ
  daily_rent : ℕ
  daily_electricity : ℕ

/-- Calculates the number of days required to pay for the cash register --/
def days_to_pay (finances : BakeryFinances) : ℕ :=
  let daily_income := finances.bread_price * finances.bread_quantity + finances.cake_price * finances.cake_quantity
  let daily_expenses := finances.daily_rent + finances.daily_electricity
  let daily_profit := daily_income - daily_expenses
  finances.cash_register_cost / daily_profit

/-- Theorem stating that it takes 8 days to pay for the cash register --/
theorem days_to_pay_cash_register :
  let maries_finances : BakeryFinances := {
    cash_register_cost := 1040,
    bread_price := 2,
    bread_quantity := 40,
    cake_price := 12,
    cake_quantity := 6,
    daily_rent := 20,
    daily_electricity := 2
  }
  days_to_pay maries_finances = 8 := by
  sorry


end days_to_pay_cash_register_l3126_312663


namespace solution_to_system_of_equations_l3126_312637

theorem solution_to_system_of_equations :
  let x : ℚ := -49/3
  let y : ℚ := -17/6
  (3 * x - 18 * y = 2) ∧ (4 * y - x = 5) := by
sorry

end solution_to_system_of_equations_l3126_312637


namespace cos_equality_exists_l3126_312662

theorem cos_equality_exists (n : ℤ) : ∃ n, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (315 * π / 180) ∧ n = 45 := by
  sorry

end cos_equality_exists_l3126_312662


namespace repeating_decimal_to_fraction_l3126_312606

/-- Proves that the repeating decimal 0.3̄03 is equal to the fraction 109/330 -/
theorem repeating_decimal_to_fraction : 
  (0.3 : ℚ) + (3 : ℚ) / 100 / (1 - 1 / 100) = 109 / 330 := by sorry

end repeating_decimal_to_fraction_l3126_312606


namespace sin_double_angle_for_specific_tan_l3126_312647

theorem sin_double_angle_for_specific_tan (α : Real) (h : Real.tan α = -1/3) :
  Real.sin (2 * α) = -3/5 := by sorry

end sin_double_angle_for_specific_tan_l3126_312647


namespace roots_sum_reciprocal_l3126_312621

theorem roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 + 8*x₁ + 4 = 0 → x₂^2 + 8*x₂ + 4 = 0 → x₁ ≠ x₂ → 
  (1 / x₁) + (1 / x₂) = -2 := by
  sorry

end roots_sum_reciprocal_l3126_312621


namespace max_value_of_rational_function_l3126_312658

theorem max_value_of_rational_function : 
  (∀ x : ℝ, (5*x^2 + 10*x + 12) / (5*x^2 + 10*x + 2) ≤ 5) ∧ 
  (∀ ε > 0, ∃ x : ℝ, (5*x^2 + 10*x + 12) / (5*x^2 + 10*x + 2) > 5 - ε) :=
by sorry

end max_value_of_rational_function_l3126_312658


namespace min_value_theorem_l3126_312629

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The first circle equation -/
def circle1 (x y : ℝ) : Prop := (x+2)^2 + y^2 = 4

/-- The second circle equation -/
def circle2 (x y : ℝ) : Prop := (x-2)^2 + y^2 = 1

/-- The expression |PM|^2 - |PN|^2 -/
def expr (x : ℝ) : ℝ := 8*x - 3

/-- The theorem stating the minimum value of |PM|^2 - |PN|^2 -/
theorem min_value_theorem (x y : ℝ) (h1 : hyperbola x y) (h2 : x ≥ 1) :
  ∃ (m : ℝ), m = 5 ∧ ∀ (x' y' : ℝ), hyperbola x' y' → x' ≥ 1 → expr x' ≥ m :=
sorry

end min_value_theorem_l3126_312629


namespace triangle_inequality_with_interior_point_l3126_312624

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define a point inside the triangle
def insidePoint (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality_with_interior_point (t : Triangle) :
  let P := perimeter t
  let O := insidePoint t
  P / 2 < distance O t.A + distance O t.B + distance O t.C ∧
  distance O t.A + distance O t.B + distance O t.C < P :=
sorry

end triangle_inequality_with_interior_point_l3126_312624


namespace blood_expiration_date_l3126_312672

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the number of days in January -/
def days_in_january : ℕ := 31

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

/-- Represents the expiration time of blood in seconds -/
def blood_expiration_time : ℕ := factorial 10

theorem blood_expiration_date :
  blood_expiration_time / seconds_per_day = days_in_january + 11 :=
sorry

end blood_expiration_date_l3126_312672


namespace malvina_money_l3126_312641

theorem malvina_money (m n : ℕ) : 
  m + n < 40 →
  n < 8 * m →
  n ≥ 4 * m + 15 →
  n = 31 :=
by sorry

end malvina_money_l3126_312641


namespace brians_breath_holding_l3126_312684

theorem brians_breath_holding (T : ℝ) : 
  T > 0 → (T * 2 * 2 * 1.5 = 60) → T = 10 := by
  sorry

end brians_breath_holding_l3126_312684


namespace sqrt_x_minus_one_meaningful_l3126_312678

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
sorry

end sqrt_x_minus_one_meaningful_l3126_312678


namespace chicken_feed_bag_weight_l3126_312671

-- Define the constants from the problem
def chicken_price : ℚ := 3/2
def feed_bag_cost : ℚ := 2
def feed_per_chicken : ℚ := 2
def num_chickens : ℕ := 50
def total_profit : ℚ := 65

-- Define the theorem
theorem chicken_feed_bag_weight :
  ∃ (bag_weight : ℚ),
    bag_weight * (feed_bag_cost / feed_per_chicken) = num_chickens * chicken_price - total_profit ∧
    bag_weight > 0 :=
by
  -- The proof goes here
  sorry

end chicken_feed_bag_weight_l3126_312671


namespace fraction_invariance_l3126_312627

theorem fraction_invariance (x y : ℝ) (square : ℝ) :
  (2 * x * y) / (x^2 + square) = (2 * (3*x) * (3*y)) / ((3*x)^2 + square) →
  square = y^2 :=
by sorry

end fraction_invariance_l3126_312627


namespace abc_inequality_l3126_312674

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_prod : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end abc_inequality_l3126_312674


namespace coefficient_x3y3_equals_15_l3126_312650

/-- The coefficient of x^3 * y^3 in the expansion of (x + y^2/x)(x + y)^5 -/
def coefficient_x3y3 (x y : ℝ) : ℝ :=
  let expanded := (x + y^2/x) * (x + y)^5
  sorry

theorem coefficient_x3y3_equals_15 :
  ∀ x y, x ≠ 0 → coefficient_x3y3 x y = 15 := by
  sorry

end coefficient_x3y3_equals_15_l3126_312650


namespace stating_special_multiples_count_l3126_312694

/-- 
The count of positive integers less than 500 that are multiples of 3 but not multiples of 9.
-/
def count_special_multiples : ℕ := 
  (Finset.filter (fun n => n % 3 = 0 ∧ n % 9 ≠ 0) (Finset.range 500)).card

/-- 
Theorem stating that the count of positive integers less than 500 
that are multiples of 3 but not multiples of 9 is equal to 111.
-/
theorem special_multiples_count : count_special_multiples = 111 := by
  sorry

end stating_special_multiples_count_l3126_312694


namespace smallest_four_digit_congruence_l3126_312668

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    (75 * n) % 375 = 225 ∧
    (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → (75 * m) % 375 = 225 → m ≥ n) ∧
    n = 1003 := by
  sorry

end smallest_four_digit_congruence_l3126_312668


namespace series_sum_equals_first_term_l3126_312610

def decreasing_to_zero (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a n ≥ a (n + 1)) ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, a n < ε)

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a n - 2 * a (n + 1) + a (n + 2)

theorem series_sum_equals_first_term (a : ℕ → ℝ) :
  decreasing_to_zero a →
  (∀ n, b a n ≥ 0) →
  (∑' n, n * b a n) = a 1 :=
sorry

end series_sum_equals_first_term_l3126_312610


namespace hyperbola_asymptotes_from_parabola_focus_l3126_312670

/-- Given a parabola and a hyperbola with shared focus, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes_from_parabola_focus 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (b : ℝ) :
  (∀ x y, parabola x y ↔ y^2 = 16*x) →
  (∀ x y, hyperbola x y ↔ x^2/12 - y^2/b^2 = 1) →
  (∃ x₀, x₀ = 4 ∧ parabola x₀ 0 ∧ ∀ y, hyperbola x₀ y → y = 0) →
  (∀ x y, hyperbola x y → y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x) :=
by sorry

end hyperbola_asymptotes_from_parabola_focus_l3126_312670


namespace base7_351_to_base6_l3126_312619

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10ToBase6 (n : ℕ) : ℕ := sorry

theorem base7_351_to_base6 :
  base10ToBase6 (base7ToBase10 351) = 503 := by sorry

end base7_351_to_base6_l3126_312619


namespace number_subtraction_l3126_312646

theorem number_subtraction (x : ℤ) : x + 30 = 55 → x - 23 = 2 := by
  sorry

end number_subtraction_l3126_312646


namespace factorial_ratio_45_43_l3126_312640

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio_45_43 : factorial 45 / factorial 43 = 1980 := by
  sorry

end factorial_ratio_45_43_l3126_312640


namespace expand_and_simplify_polynomial_l3126_312601

theorem expand_and_simplify_polynomial (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_polynomial_l3126_312601


namespace positive_number_equality_l3126_312613

theorem positive_number_equality (x : ℝ) (h1 : x > 0) :
  (2 / 3) * x = (25 / 216) * (1 / x) → x = 5 / 12 := by sorry

end positive_number_equality_l3126_312613


namespace smallest_five_digit_mod_13_l3126_312626

theorem smallest_five_digit_mod_13 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n ≡ 11 [MOD 13] → n ≥ 10009 :=
by sorry

end smallest_five_digit_mod_13_l3126_312626


namespace lettuce_types_count_l3126_312638

/-- The number of lunch combo options given the number of lettuce types -/
def lunch_combos (lettuce_types : ℕ) : ℕ :=
  lettuce_types * 3 * 4 * 2

/-- Theorem stating that there are 2 types of lettuce -/
theorem lettuce_types_count : ∃ (n : ℕ), n = 2 ∧ lunch_combos n = 48 := by
  sorry

end lettuce_types_count_l3126_312638


namespace chicken_to_beef_ratio_is_two_to_one_l3126_312667

/-- Represents the order of beef and chicken --/
structure FoodOrder where
  beef_pounds : ℕ
  beef_price_per_pound : ℕ
  chicken_price_per_pound : ℕ
  total_cost : ℕ

/-- Calculates the ratio of chicken to beef in the order --/
def chicken_to_beef_ratio (order : FoodOrder) : ℚ :=
  let beef_cost := order.beef_pounds * order.beef_price_per_pound
  let chicken_cost := order.total_cost - beef_cost
  let chicken_pounds := chicken_cost / order.chicken_price_per_pound
  chicken_pounds / order.beef_pounds

/-- Theorem stating that the ratio of chicken to beef is 2:1 for the given order --/
theorem chicken_to_beef_ratio_is_two_to_one (order : FoodOrder) 
  (h1 : order.beef_pounds = 1000)
  (h2 : order.beef_price_per_pound = 8)
  (h3 : order.chicken_price_per_pound = 3)
  (h4 : order.total_cost = 14000) : 
  chicken_to_beef_ratio order = 2 := by
  sorry

#eval chicken_to_beef_ratio { 
  beef_pounds := 1000, 
  beef_price_per_pound := 8, 
  chicken_price_per_pound := 3, 
  total_cost := 14000 
}

end chicken_to_beef_ratio_is_two_to_one_l3126_312667


namespace rectangle_max_area_l3126_312682

theorem rectangle_max_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  2 * (a + b) = 60 → a * b ≤ 225 := by
  sorry

end rectangle_max_area_l3126_312682


namespace third_place_winnings_value_l3126_312681

/-- The amount of money in the pot -/
def pot_total : ℝ := 210

/-- The percentage of the pot that the third place winner receives -/
def third_place_percentage : ℝ := 0.15

/-- The amount of money the third place winner receives -/
def third_place_winnings : ℝ := pot_total * third_place_percentage

theorem third_place_winnings_value : third_place_winnings = 31.5 := by
  sorry

end third_place_winnings_value_l3126_312681


namespace product_congruence_l3126_312631

theorem product_congruence : ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (93 * 59 * 84) % 25 = m ∧ m = 8 := by
  sorry

end product_congruence_l3126_312631


namespace days_for_C_alone_is_8_l3126_312620

/-- The number of days it takes for C to finish the work alone, given that:
    - A, B, and C together can finish the work in 4 days
    - A alone can finish the work in 12 days
    - B alone can finish the work in 24 days
-/
def days_for_C_alone (days_together days_A_alone days_B_alone : ℚ) : ℚ :=
  let work_rate_together := 1 / days_together
  let work_rate_A := 1 / days_A_alone
  let work_rate_B := 1 / days_B_alone
  let work_rate_C := work_rate_together - work_rate_A - work_rate_B
  1 / work_rate_C

theorem days_for_C_alone_is_8 :
  days_for_C_alone 4 12 24 = 8 := by
  sorry

end days_for_C_alone_is_8_l3126_312620


namespace triangle_median_inequality_l3126_312655

theorem triangle_median_inequality (a b c ma mb mc : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hma : ma > 0) (hmb : mb > 0) (hmc : mc > 0)
  (h_ma : 4 * ma^2 = 2 * (b^2 + c^2) - a^2)
  (h_mb : 4 * mb^2 = 2 * (c^2 + a^2) - b^2)
  (h_mc : 4 * mc^2 = 2 * (a^2 + b^2) - c^2) :
  ma^2 / a^2 + mb^2 / b^2 + mc^2 / c^2 ≥ 9/4 := by
sorry

end triangle_median_inequality_l3126_312655


namespace simplify_expression_1_simplify_expression_2_l3126_312649

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  3/2 * (a^2 * b - 2 * a * b^2) - 1/2 * (a * b^2 - 4 * a^2 * b) + 1/2 * a * b^2 =
  7/2 * a^2 * b - 3 * a * b^2 := by sorry

end simplify_expression_1_simplify_expression_2_l3126_312649


namespace circular_lid_area_l3126_312673

/-- The area of a circular lid with diameter 2.75 inches is approximately 5.9375 square inches. -/
theorem circular_lid_area :
  let diameter : ℝ := 2.75
  let radius : ℝ := diameter / 2
  let area : ℝ := Real.pi * radius^2
  ∃ ε > 0, abs (area - 5.9375) < ε :=
by sorry

end circular_lid_area_l3126_312673


namespace unique_prime_p_l3126_312639

theorem unique_prime_p : ∃! p : ℕ, Prime p ∧ Prime (5 * p + 1) :=
by
  -- The proof goes here
  sorry

end unique_prime_p_l3126_312639


namespace heating_rate_at_10_seconds_l3126_312665

-- Define the temperature function
def temperature (t : ℝ) : ℝ := 0.2 * t^2

-- Define the rate of heating function (derivative of temperature)
def rateOfHeating (t : ℝ) : ℝ := 0.4 * t

-- Theorem statement
theorem heating_rate_at_10_seconds :
  rateOfHeating 10 = 4 := by sorry

end heating_rate_at_10_seconds_l3126_312665


namespace original_jellybean_count_l3126_312651

-- Define the daily reduction rate
def daily_reduction_rate : ℝ := 0.8

-- Define the function that calculates the remaining quantity after n days
def remaining_after_days (initial : ℝ) (days : ℕ) : ℝ :=
  initial * (daily_reduction_rate ^ days)

-- State the theorem
theorem original_jellybean_count :
  ∃ (initial : ℝ), remaining_after_days initial 2 = 32 ∧ initial = 50 := by
  sorry

end original_jellybean_count_l3126_312651


namespace find_number_l3126_312679

theorem find_number : ∃ n : ℕ, n + 3427 = 13200 ∧ n = 9773 := by
  sorry

end find_number_l3126_312679


namespace sane_person_identified_l3126_312643

/-- Represents the types of individuals in Transylvania -/
inductive PersonType
| Sane
| Transylvanian

/-- Represents possible answers to a question -/
inductive Answer
| Yes
| No

/-- A function that determines how a person of a given type would answer the question -/
def wouldAnswer (t : PersonType) : Answer :=
  match t with
  | PersonType.Sane => Answer.No
  | PersonType.Transylvanian => Answer.Yes

/-- Theorem stating that if an answer allows immediate identification, the person must be sane -/
theorem sane_person_identified
  (answer : Answer)
  (h_immediate : ∃ (t : PersonType), wouldAnswer t = answer) :
  answer = Answer.No ∧ wouldAnswer PersonType.Sane = answer :=
sorry

end sane_person_identified_l3126_312643


namespace overhead_percentage_problem_l3126_312680

/-- Calculates the percentage of cost for overhead given the purchase price, markup, and net profit. -/
def overhead_percentage (purchase_price markup net_profit : ℚ) : ℚ :=
  let overhead := markup - net_profit
  (overhead / purchase_price) * 100

/-- Theorem stating that given the specific values in the problem, the overhead percentage is 37.5% -/
theorem overhead_percentage_problem :
  let purchase_price : ℚ := 48
  let markup : ℚ := 30
  let net_profit : ℚ := 12
  overhead_percentage purchase_price markup net_profit = 37.5 := by
  sorry

#eval overhead_percentage 48 30 12

end overhead_percentage_problem_l3126_312680


namespace nicks_chocolate_oranges_l3126_312633

/-- Proves the number of chocolate oranges Nick had initially -/
theorem nicks_chocolate_oranges 
  (candy_bar_price : ℕ) 
  (chocolate_orange_price : ℕ) 
  (fundraising_goal : ℕ) 
  (candy_bars_to_sell : ℕ) 
  (h1 : candy_bar_price = 5)
  (h2 : chocolate_orange_price = 10)
  (h3 : fundraising_goal = 1000)
  (h4 : candy_bars_to_sell = 160)
  (h5 : candy_bar_price * candy_bars_to_sell + chocolate_orange_price * chocolate_oranges = fundraising_goal) :
  chocolate_oranges = 20 := by
  sorry

end nicks_chocolate_oranges_l3126_312633


namespace inequality_preservation_l3126_312600

theorem inequality_preservation (m n : ℝ) (h : m > n) : m - 6 > n - 6 := by
  sorry

end inequality_preservation_l3126_312600


namespace sum_x_y_equals_nine_fifths_l3126_312622

theorem sum_x_y_equals_nine_fifths (x y : ℝ) 
  (eq1 : x + |x| + y = 5)
  (eq2 : x + |y| - y = 6) : 
  x + y = 9/5 := by
  sorry

end sum_x_y_equals_nine_fifths_l3126_312622


namespace hyperbola_asymptotes_tangent_to_circle_l3126_312645

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), y^2 - x^2/m^2 = 1) →
  (∃ (x y : ℝ), x^2 + y^2 - 4*y + 3 = 0) →
  (∀ (x y : ℝ), (y^2 - x^2/m^2 = 0) → (x^2 + (y-2)^2 = 1)) →
  m = Real.sqrt 3 / 3 :=
by sorry

end hyperbola_asymptotes_tangent_to_circle_l3126_312645


namespace blue_paint_cans_l3126_312608

theorem blue_paint_cans (total_cans : ℕ) (blue_ratio yellow_ratio : ℕ) 
  (h1 : total_cans = 42)
  (h2 : blue_ratio = 4)
  (h3 : yellow_ratio = 3) : 
  (blue_ratio * total_cans) / (blue_ratio + yellow_ratio) = 24 := by
  sorry

end blue_paint_cans_l3126_312608


namespace trig_expression_equality_l3126_312692

theorem trig_expression_equality : 
  (Real.sin (40 * π / 180) - Real.sqrt 3 * Real.cos (20 * π / 180)) / Real.cos (10 * π / 180) = -1 := by
  sorry

end trig_expression_equality_l3126_312692


namespace balls_removed_by_other_students_l3126_312688

theorem balls_removed_by_other_students (tennis_balls soccer_balls baskets students_removed_8 remaining_balls : ℕ) 
  (h1 : tennis_balls = 15)
  (h2 : soccer_balls = 5)
  (h3 : baskets = 5)
  (h4 : students_removed_8 = 3)
  (h5 : remaining_balls = 56) : 
  ((baskets * (tennis_balls + soccer_balls)) - (students_removed_8 * 8) - remaining_balls) / 2 = 10 := by
sorry

end balls_removed_by_other_students_l3126_312688


namespace sport_participation_l3126_312695

theorem sport_participation (total : ℕ) (cyclists : ℕ) (swimmers : ℕ) (skiers : ℕ) (unsatisfactory : ℕ)
  (h1 : total = 25)
  (h2 : cyclists = 17)
  (h3 : swimmers = 13)
  (h4 : skiers = 8)
  (h5 : unsatisfactory = 6)
  (h6 : ∀ s : ℕ, s ≤ total → s ≤ cyclists + swimmers + skiers - 2)
  (h7 : cyclists + swimmers + skiers = 2 * (total - unsatisfactory)) :
  ∃ swim_and_ski : ℕ, swim_and_ski = 2 ∧ swim_and_ski ≤ swimmers ∧ swim_and_ski ≤ skiers :=
by sorry

end sport_participation_l3126_312695


namespace perpendicular_vector_proof_l3126_312687

/-- Given two parallel lines with direction vector (5, 4), prove that the vector (v₁, v₂) 
    perpendicular to (5, 4) satisfying v₁ + v₂ = 7 is (-28, 35). -/
theorem perpendicular_vector_proof (v₁ v₂ : ℝ) : 
  (5 * 4 + 4 * (-5) = 0) →  -- Lines are parallel with direction vector (5, 4)
  (5 * v₁ + 4 * v₂ = 0) →   -- (v₁, v₂) is perpendicular to (5, 4)
  (v₁ + v₂ = 7) →           -- Sum of v₁ and v₂ is 7
  (v₁ = -28 ∧ v₂ = 35) :=   -- Conclusion: v₁ = -28 and v₂ = 35
by
  sorry


end perpendicular_vector_proof_l3126_312687


namespace abie_spent_64_dollars_l3126_312632

def initial_bags : ℕ := 20
def original_price : ℚ := 2
def shared_fraction : ℚ := 2/5
def half_price_bags : ℕ := 18
def coupon_bags : ℕ := 4
def coupon_price_fraction : ℚ := 3/4

def total_spent : ℚ :=
  initial_bags * original_price +
  half_price_bags * (original_price / 2) +
  coupon_bags * (original_price * coupon_price_fraction)

theorem abie_spent_64_dollars : total_spent = 64 := by
  sorry

end abie_spent_64_dollars_l3126_312632


namespace exists_n_with_totient_inequality_l3126_312607

open Nat

theorem exists_n_with_totient_inequality : 
  ∃ (n : ℕ), n > 0 ∧ totient (2*n - 1) + totient (2*n + 1) < (1 : ℚ) / 1000 * totient (2*n) :=
by sorry

end exists_n_with_totient_inequality_l3126_312607


namespace min_value_x_plus_y_l3126_312690

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 2 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 2 ∧ x + y = 8 :=
sorry

end min_value_x_plus_y_l3126_312690


namespace hyperbola_parabola_relation_l3126_312603

theorem hyperbola_parabola_relation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∃ (c : ℝ), c = 2 * a ∧ c^2 = a^2 + b^2) →
  (2 = (p / 2 / b) / Real.sqrt ((1 / a^2) + (1 / b^2))) →
  p = 8 := by sorry

end hyperbola_parabola_relation_l3126_312603


namespace polynomial_coefficient_sum_l3126_312612

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end polynomial_coefficient_sum_l3126_312612


namespace square_side_ratio_l3126_312677

theorem square_side_ratio (area_ratio : ℚ) (h : area_ratio = 50 / 98) :
  ∃ (p q r : ℕ), 
    (Real.sqrt (area_ratio) = p * Real.sqrt q / r) ∧
    (p + q + r = 13) := by
  sorry

end square_side_ratio_l3126_312677


namespace no_integer_solution_for_trig_equation_l3126_312691

theorem no_integer_solution_for_trig_equation : 
  ¬ ∃ (a b : ℤ), Real.sqrt (4 - 3 * Real.sin (30 * π / 180)) = a + b * (1 / Real.sin (30 * π / 180)) := by
  sorry

end no_integer_solution_for_trig_equation_l3126_312691


namespace cards_taken_away_l3126_312628

theorem cards_taken_away (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 76)
  (h2 : final_cards = 17) :
  initial_cards - final_cards = 59 := by
  sorry

end cards_taken_away_l3126_312628


namespace max_sum_of_a_and_b_l3126_312659

theorem max_sum_of_a_and_b : ∀ a b : ℕ+,
  b > 2 →
  a^(b:ℕ) < 600 →
  a + b ≤ 11 :=
by sorry

end max_sum_of_a_and_b_l3126_312659


namespace certain_fraction_proof_l3126_312635

theorem certain_fraction_proof : 
  ∃ (x y : ℚ), (3 / 7) / (x / y) = (2 / 5) / (1 / 7) ∧ x / y = 15 / 98 :=
by sorry

end certain_fraction_proof_l3126_312635


namespace actual_distance_travelled_l3126_312634

/-- The actual distance travelled by a person under specific conditions -/
theorem actual_distance_travelled (normal_speed fast_speed additional_distance : ℝ) 
  (h1 : normal_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : additional_distance = 20)
  (h4 : (actual_distance / normal_speed) = ((actual_distance + additional_distance) / fast_speed)) :
  actual_distance = 50 := by
  sorry

end actual_distance_travelled_l3126_312634


namespace complement_of_union_l3126_312656

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2}

-- Define set N
def N : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_of_union (u : Finset Nat) (m n : Finset Nat) 
  (hU : u = U) (hM : m = M) (hN : n = N) : 
  u \ (m ∪ n) = {5} := by
  sorry

end complement_of_union_l3126_312656


namespace soda_price_theorem_l3126_312676

/-- Calculates the price of soda cans given specific discount conditions -/
def sodaPrice (regularPrice : ℝ) (caseDiscount : ℝ) (bulkDiscount : ℝ) (caseSize : ℕ) (numCans : ℕ) : ℝ :=
  let discountedPrice := regularPrice * (1 - caseDiscount)
  let fullCases := numCans / caseSize
  let remainingCans := numCans % caseSize
  let fullCasePrice := if fullCases ≥ 3
                       then (fullCases * caseSize * discountedPrice) * (1 - bulkDiscount)
                       else fullCases * caseSize * discountedPrice
  let remainingPrice := remainingCans * discountedPrice
  fullCasePrice + remainingPrice

/-- The price of 70 cans of soda under given discount conditions is $26.895 -/
theorem soda_price_theorem :
  sodaPrice 0.55 0.25 0.10 24 70 = 26.895 := by
  sorry

end soda_price_theorem_l3126_312676


namespace abs_x_minus_one_equals_one_minus_x_implies_x_leq_one_l3126_312666

theorem abs_x_minus_one_equals_one_minus_x_implies_x_leq_one (x : ℝ) : 
  |x - 1| = 1 - x → x ≤ 1 := by
sorry

end abs_x_minus_one_equals_one_minus_x_implies_x_leq_one_l3126_312666


namespace triangle_construction_l3126_312605

-- Define the necessary structures
structure Line where
  -- Add necessary fields for a line

structure Point where
  -- Add necessary fields for a point

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the necessary functions
def is_on_line (p : Point) (l : Line) : Prop :=
  sorry

def is_foot_of_altitude (m : Point) (v : Point) (s : Point) (t : Point) : Prop :=
  sorry

-- Main theorem
theorem triangle_construction (L : Line) (M₁ M₂ : Point) :
  ∃ (ABC A'B'C' : Triangle),
    (is_on_line ABC.C L ∧ is_on_line ABC.B L) ∧
    (is_on_line A'B'C'.C L ∧ is_on_line A'B'C'.B L) ∧
    (is_foot_of_altitude M₁ ABC.A ABC.B ABC.C) ∧
    (is_foot_of_altitude M₂ ABC.B ABC.A ABC.C) ∧
    (is_foot_of_altitude M₁ A'B'C'.A A'B'C'.B A'B'C'.C) ∧
    (is_foot_of_altitude M₂ A'B'C'.B A'B'C'.A A'B'C'.C) :=
  sorry


end triangle_construction_l3126_312605


namespace purely_imaginary_iff_a_eq_two_l3126_312618

/-- For a complex number z = (a^2 - 4) + (a + 2)i where a is real,
    z is purely imaginary if and only if a = 2 -/
theorem purely_imaginary_iff_a_eq_two (a : ℝ) :
  let z : ℂ := (a^2 - 4) + (a + 2)*I
  (z.re = 0 ∧ z.im ≠ 0) ↔ a = 2 := by
  sorry

end purely_imaginary_iff_a_eq_two_l3126_312618


namespace mara_pink_crayons_percentage_l3126_312653

/-- The percentage of Mara's crayons that are pink -/
def mara_pink_percentage : ℝ := 10

theorem mara_pink_crayons_percentage 
  (mara_total : ℕ) 
  (luna_total : ℕ) 
  (luna_pink_percentage : ℝ) 
  (total_pink : ℕ) 
  (h1 : mara_total = 40)
  (h2 : luna_total = 50)
  (h3 : luna_pink_percentage = 20)
  (h4 : total_pink = 14)
  : mara_pink_percentage = 10 := by
  sorry

end mara_pink_crayons_percentage_l3126_312653


namespace fraction_subtraction_l3126_312696

theorem fraction_subtraction : 
  (5 + 7 + 9) / (2 + 4 + 6) - (4 + 6 + 8) / (3 + 5 + 7) = 11 / 20 := by
  sorry

end fraction_subtraction_l3126_312696


namespace semicircle_rectangle_property_l3126_312685

-- Define the semicircle and its properties
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define a point on the semicircle
def PointOnSemicircle (s : Semicircle) := { p : ℝ × ℝ // (p.1 - s.center.1)^2 + (p.2 - s.center.2)^2 = s.radius^2 ∧ p.2 ≥ s.center.2 }

-- Theorem statement
theorem semicircle_rectangle_property
  (s : Semicircle)
  (r : Rectangle)
  (h_square : r.height = s.radius / Real.sqrt 2)  -- Height equals side of inscribed square
  (h_base : r.base = 2 * s.radius)  -- Base is diameter
  (M : PointOnSemicircle s)
  (E F : ℝ)  -- E and F are x-coordinates on the diameter
  (h_E : E ∈ Set.Icc s.center.1 (s.center.1 + s.radius))
  (h_F : F ∈ Set.Icc s.center.1 (s.center.1 + s.radius))
  : (F - s.center.1)^2 + (s.center.1 + 2*s.radius - E)^2 = (2*s.radius)^2 := by
  sorry


end semicircle_rectangle_property_l3126_312685


namespace system_solution_l3126_312699

theorem system_solution : ∃ (x y : ℝ), 2 * x - y = 8 ∧ 3 * x + 2 * y = 5 := by
  use 3, -2
  sorry

end system_solution_l3126_312699


namespace odd_function_properties_l3126_312615

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   ∀ x, f a x = 1 - 2 / (2^x + 1) ∧
   StrictMono (f a)) :=
by sorry

end odd_function_properties_l3126_312615


namespace fib_like_seq_a9_l3126_312697

/-- An increasing sequence of positive integers with a Fibonacci-like recurrence relation -/
def FibLikeSeq (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a (n + 1) + a n)

theorem fib_like_seq_a9 (a : ℕ → ℕ) (h : FibLikeSeq a) (h7 : a 7 = 210) : 
  a 9 = 550 := by
  sorry

end fib_like_seq_a9_l3126_312697


namespace inequality_proof_l3126_312689

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) ≤ (Real.sqrt 33 + 1) / 4 := by
  sorry

end inequality_proof_l3126_312689


namespace equation_solutions_l3126_312683

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = 4/3 ∧
    (x1 + 1)^2 = (2*x1 - 3)^2 ∧ (x2 + 1)^2 = (2*x2 - 3)^2) :=
by sorry

end equation_solutions_l3126_312683


namespace base_eight_satisfies_equation_unique_base_satisfies_equation_l3126_312693

/-- Given a base b, converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 245_b + 132_b = 400_b holds for a given base b --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 4, 5] b + toDecimal [1, 3, 2] b = toDecimal [4, 0, 0] b

theorem base_eight_satisfies_equation :
  equationHolds 8 := by sorry

theorem unique_base_satisfies_equation :
  ∀ b : Nat, b > 1 → equationHolds b → b = 8 := by sorry

end base_eight_satisfies_equation_unique_base_satisfies_equation_l3126_312693


namespace lottery_probability_l3126_312654

theorem lottery_probability : 
  (1 : ℚ) / 30 * (1 / 50 * 1 / 49 * 1 / 48 * 1 / 47 * 1 / 46) = 1 / 7627536000 := by
  sorry

end lottery_probability_l3126_312654


namespace a_equals_two_l3126_312604

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 4

-- State the theorem
theorem a_equals_two (a : ℝ) : f a a = f a 1 + 2 * (1 - 1) → a = 2 := by
  sorry

end a_equals_two_l3126_312604


namespace factorial_equation_l3126_312636

theorem factorial_equation : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_equation_l3126_312636


namespace y_in_terms_of_x_l3126_312614

theorem y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3*x) : y = -2*x - 2 := by
  sorry

end y_in_terms_of_x_l3126_312614


namespace allan_bought_three_balloons_l3126_312660

/-- The number of balloons Allan bought at the park -/
def balloons_bought_by_allan (allan_initial : ℕ) (jake_total : ℕ) (jake_difference : ℕ) : ℕ :=
  (jake_total - jake_difference) - allan_initial

/-- Theorem stating that Allan bought 3 balloons at the park -/
theorem allan_bought_three_balloons :
  balloons_bought_by_allan 2 6 1 = 3 := by
  sorry

end allan_bought_three_balloons_l3126_312660


namespace perfect_square_quadratic_l3126_312652

theorem perfect_square_quadratic (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*(k+1)*x + 4 = (x - a)^2) → (k = -3 ∨ k = 1) := by
  sorry

end perfect_square_quadratic_l3126_312652


namespace no_rational_solution_to_5x2_plus_3y2_eq_1_l3126_312625

theorem no_rational_solution_to_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := by
sorry

end no_rational_solution_to_5x2_plus_3y2_eq_1_l3126_312625


namespace root_conditions_l3126_312657

/-- The equation x^4 + px^2 + q = 0 has real roots satisfying x₂/x₁ = x₃/x₂ = x₄/x₃ 
    if and only if p < 0 and q = p^2/4 -/
theorem root_conditions (p q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0 ∧
    x₁^4 + p*x₁^2 + q = 0 ∧
    x₂^4 + p*x₂^2 + q = 0 ∧
    x₃^4 + p*x₃^2 + q = 0 ∧
    x₄^4 + p*x₄^2 + q = 0 ∧
    x₂/x₁ = x₃/x₂ ∧ x₃/x₂ = x₄/x₃) ↔
  (p < 0 ∧ q = p^2/4) :=
by sorry


end root_conditions_l3126_312657


namespace simplify_expression_solve_equation_solve_system_l3126_312630

-- Part 1
theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem solve_equation (x y : ℝ) (h : x^2 - 2*y = 4) :
  23 - 3*x^2 + 6*y = 11 := by sorry

-- Part 3
theorem solve_system (a b c d : ℝ) 
  (h1 : a - 2*b = 3) (h2 : 2*b - c = -5) (h3 : c - d = -9) :
  (a - c) + (2*b - d) - (2*b - c) = -11 := by sorry

end simplify_expression_solve_equation_solve_system_l3126_312630


namespace shoe_repair_time_calculation_l3126_312609

/-- Given the total time spent on repairing shoes and the time required to replace buckles,
    calculate the time needed to even out the heel for each shoe. -/
theorem shoe_repair_time_calculation 
  (total_time : ℝ)
  (buckle_time : ℝ)
  (h_total : total_time = 30)
  (h_buckle : buckle_time = 5)
  : (total_time - buckle_time) / 2 = 12.5 := by
  sorry

end shoe_repair_time_calculation_l3126_312609


namespace square_sum_identity_l3126_312623

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end square_sum_identity_l3126_312623


namespace difference_of_squares_example_l3126_312661

theorem difference_of_squares_example : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_example_l3126_312661


namespace library_books_count_l3126_312617

theorem library_books_count : ∃ (n : ℕ), 
  500 < n ∧ n < 650 ∧ 
  ∃ (r : ℕ), n = 12 * r + 7 ∧
  ∃ (l : ℕ), n = 25 * l - 5 ∧
  n = 595 := by
  sorry

end library_books_count_l3126_312617


namespace triangle_properties_l3126_312602

theorem triangle_properties (A B C : Real) (a : Real × Real) :
  -- A, B, C are angles of a triangle
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π →
  -- Definition of vector a
  a = (Real.sqrt 2 * Real.cos ((A + B) / 2), Real.sin ((A - B) / 2)) →
  -- Magnitude of a
  Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 6 / 2 →
  -- Conclusions
  (Real.tan A * Real.tan B = 1 / 3) ∧
  (∀ C', C' = π - A - B → Real.tan C' ≤ -Real.sqrt 3) ∧
  (∃ C', C' = π - A - B ∧ Real.tan C' = -Real.sqrt 3) :=
by sorry

end triangle_properties_l3126_312602


namespace class_average_score_l3126_312611

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percentage : ℚ) (makeup_day_percentage : ℚ)
  (assigned_day_average : ℚ) (makeup_day_average : ℚ) :
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_day_percentage = 30 / 100 →
  assigned_day_average = 65 / 100 →
  makeup_day_average = 95 / 100 →
  (assigned_day_percentage * assigned_day_average + 
   makeup_day_percentage * makeup_day_average) = 74 / 100 := by
sorry

end class_average_score_l3126_312611


namespace roof_area_l3126_312664

theorem roof_area (width length : ℝ) (h1 : length = 5 * width) (h2 : length - width = 48) :
  width * length = 720 := by
  sorry

end roof_area_l3126_312664


namespace max_sin_a_value_l3126_312644

theorem max_sin_a_value (a b c : Real) 
  (h1 : Real.cos a = Real.tan b)
  (h2 : Real.cos b = Real.tan c)
  (h3 : Real.cos c = Real.tan a) :
  ∃ (max_sin_a : Real), 
    (∀ a' b' c' : Real, 
      Real.cos a' = Real.tan b' → 
      Real.cos b' = Real.tan c' → 
      Real.cos c' = Real.tan a' → 
      Real.sin a' ≤ max_sin_a) ∧
    max_sin_a = Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by sorry

end max_sin_a_value_l3126_312644


namespace inequality_proof_l3126_312669

theorem inequality_proof (x y z : ℝ) : x^4 + y^4 + z^2 + 1 ≥ 2*x*(x*y^2 - x + z + 1) := by
  sorry

end inequality_proof_l3126_312669


namespace sum_product_uniqueness_l3126_312642

theorem sum_product_uniqueness (S P : ℝ) (x y : ℝ) 
  (h_sum : x + y = S) (h_product : x * y = P) :
  (x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
  (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2) :=
by sorry

end sum_product_uniqueness_l3126_312642
