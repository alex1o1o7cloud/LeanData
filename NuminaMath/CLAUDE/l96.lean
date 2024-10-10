import Mathlib

namespace ammonia_formation_l96_9683

-- Define the chemical reaction
structure Reaction where
  koh : ℝ  -- moles of Potassium hydroxide
  nh4i : ℝ  -- moles of Ammonium iodide
  nh3 : ℝ  -- moles of Ammonia formed

-- Define the balanced equation
axiom balanced_equation (r : Reaction) : r.koh = r.nh4i ∧ r.koh = r.nh3

-- Theorem: Given 3 moles of KOH, the number of moles of NH3 formed is 3
theorem ammonia_formation (r : Reaction) (h : r.koh = 3) : r.nh3 = 3 := by
  sorry

-- The proof is omitted as per instructions

end ammonia_formation_l96_9683


namespace monthly_salary_calculation_l96_9619

/-- Proves that a man's monthly salary is 5750 Rs. given the specified conditions -/
theorem monthly_salary_calculation (savings_rate : ℝ) (expense_increase : ℝ) (new_savings : ℝ) : 
  savings_rate = 0.20 →
  expense_increase = 0.20 →
  new_savings = 230 →
  ∃ (salary : ℝ), salary = 5750 ∧ 
    (1 - savings_rate - expense_increase * (1 - savings_rate)) * salary = new_savings :=
by sorry

end monthly_salary_calculation_l96_9619


namespace probability_of_specific_dice_outcome_l96_9640

def num_dice : ℕ := 5
def num_sides : ℕ := 5
def target_number : ℕ := 3
def num_target : ℕ := 2

theorem probability_of_specific_dice_outcome :
  (num_dice.choose num_target *
   (1 / num_sides) ^ num_target *
   ((num_sides - 1) / num_sides) ^ (num_dice - num_target) : ℚ) =
  640 / 3125 := by
  sorry

end probability_of_specific_dice_outcome_l96_9640


namespace westville_summer_retreat_soccer_percentage_l96_9689

theorem westville_summer_retreat_soccer_percentage 
  (total : ℝ) 
  (soccer_percentage : ℝ) 
  (swim_percentage : ℝ) 
  (soccer_and_swim_percentage : ℝ) 
  (basketball_percentage : ℝ) 
  (basketball_soccer_no_swim_percentage : ℝ) 
  (h1 : soccer_percentage = 0.7) 
  (h2 : swim_percentage = 0.5) 
  (h3 : soccer_and_swim_percentage = 0.3 * soccer_percentage) 
  (h4 : basketball_percentage = 0.2) 
  (h5 : basketball_soccer_no_swim_percentage = 0.25 * basketball_percentage) : 
  (soccer_percentage * total - soccer_and_swim_percentage * total - basketball_soccer_no_swim_percentage * total) / 
  ((1 - swim_percentage) * total) = 0.8 := by
  sorry

end westville_summer_retreat_soccer_percentage_l96_9689


namespace quadratic_sum_l96_9677

/-- A quadratic function passing through (1,0) and (-5,0) with minimum value 25 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≥ 25) ∧
  QuadraticFunction a b c 1 = 0 ∧
  QuadraticFunction a b c (-5) = 0 →
  a + b + c = 25 := by
sorry

end quadratic_sum_l96_9677


namespace curve_self_intersection_l96_9660

-- Define the curve
def x (t : ℝ) : ℝ := t^3 - t - 2
def y (t : ℝ) : ℝ := t^3 - t^2 - 9*t + 5

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (22, -4)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ 
    x a = x b ∧ 
    y a = y b ∧ 
    (x a, y a) = intersection_point :=
sorry

end curve_self_intersection_l96_9660


namespace value_of_y_l96_9630

theorem value_of_y (y : ℝ) (h : 2/3 - 1/4 = 4/y) : y = 9.6 := by
  sorry

end value_of_y_l96_9630


namespace cost_per_serving_is_one_dollar_l96_9676

/-- The cost of a serving of spaghetti and meatballs -/
def cost_per_serving (pasta_cost sauce_cost meatballs_cost : ℚ) (num_servings : ℕ) : ℚ :=
  (pasta_cost + sauce_cost + meatballs_cost) / num_servings

/-- Theorem: The cost per serving is $1.00 -/
theorem cost_per_serving_is_one_dollar :
  cost_per_serving 1 2 5 8 = 1 := by sorry

end cost_per_serving_is_one_dollar_l96_9676


namespace tan_alpha_implies_c_equals_five_l96_9657

theorem tan_alpha_implies_c_equals_five (α : Real) (c : Real) 
  (h1 : Real.tan α = -1/2) 
  (h2 : c = (2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α)) : 
  c = 5 := by
  sorry

end tan_alpha_implies_c_equals_five_l96_9657


namespace article_price_reduction_l96_9668

/-- Proves that given an article with an original cost of 50, sold at a 25% profit,
    if the selling price is reduced by 10.50 and the profit becomes 30%,
    then the reduction in the buying price is 20%. -/
theorem article_price_reduction (original_cost : ℝ) (original_profit_percent : ℝ)
  (price_reduction : ℝ) (new_profit_percent : ℝ) :
  original_cost = 50 →
  original_profit_percent = 25 →
  price_reduction = 10.50 →
  new_profit_percent = 30 →
  ∃ (buying_price_reduction : ℝ),
    buying_price_reduction = 20 ∧
    (original_cost * (1 + original_profit_percent / 100) - price_reduction) =
    (original_cost * (1 - buying_price_reduction / 100)) * (1 + new_profit_percent / 100) :=
by sorry

end article_price_reduction_l96_9668


namespace gcd_lcm_sum_l96_9628

theorem gcd_lcm_sum : Nat.gcd 45 125 + Nat.lcm 50 15 = 155 := by
  sorry

end gcd_lcm_sum_l96_9628


namespace ellipse_locus_l96_9618

/-- Given two fixed points F₁ and F₂ on the x-axis, and a point M such that
    the sum of its distances to F₁ and F₂ is constant, prove that the locus of M
    is an ellipse with F₁ and F₂ as foci. -/
theorem ellipse_locus (F₁ F₂ M : ℝ × ℝ) (d : ℝ) :
  F₁ = (-4, 0) →
  F₂ = (4, 0) →
  d = 10 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) +
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = d →
  M.1^2 / 25 + M.2^2 / 9 = 1 :=
by sorry

end ellipse_locus_l96_9618


namespace arithmetic_sequence_m_value_l96_9699

/-- An arithmetic sequence with sum of first n terms Sn -/
structure ArithmeticSequence where
  S : ℕ → ℤ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_{m-1} = -2, S_m = 0, and S_{m+1} = 3, then m = 5 -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
  (h1 : seq.S (m - 1) = -2)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 1) = 3) :
  m = 5 := by
  sorry


end arithmetic_sequence_m_value_l96_9699


namespace not_perfect_square_l96_9608

theorem not_perfect_square : 
  (∃ x : ℕ, 6^2040 = x^2) ∧ 
  (∀ y : ℕ, 7^2041 ≠ y^2) ∧ 
  (∃ z : ℕ, 8^2042 = z^2) ∧ 
  (∃ w : ℕ, 9^2043 = w^2) ∧ 
  (∃ v : ℕ, 10^2044 = v^2) :=
by sorry

end not_perfect_square_l96_9608


namespace exists_x0_implies_a_value_l96_9646

noncomputable section

open Real

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x + exp (x - a)
def g (a x : ℝ) : ℝ := log (x + 2) - 4 * exp (a - x)

-- State the theorem
theorem exists_x0_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) →
  a = -log 2 - 1 := by
sorry

end

end exists_x0_implies_a_value_l96_9646


namespace abc_remainder_mod_9_l96_9696

theorem abc_remainder_mod_9 (a b c : ℕ) 
  (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (cong1 : a + 2*b + 3*c ≡ 0 [ZMOD 9])
  (cong2 : 2*a + 3*b + c ≡ 5 [ZMOD 9])
  (cong3 : 3*a + b + 2*c ≡ 5 [ZMOD 9]) :
  a * b * c ≡ 0 [ZMOD 9] := by
sorry

end abc_remainder_mod_9_l96_9696


namespace greatest_3digit_base9_div_by_7_l96_9670

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

/-- Checks if a number is a 3-digit base 9 number -/
def isThreeDigitBase9 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_div_by_7 :
  ∀ n : ℕ, isThreeDigitBase9 n → (base9ToBase10 n) % 7 = 0 → n ≤ 888 :=
by sorry

end greatest_3digit_base9_div_by_7_l96_9670


namespace gcd_of_4557_1953_5115_l96_9639

theorem gcd_of_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end gcd_of_4557_1953_5115_l96_9639


namespace star_inequality_equivalence_l96_9632

-- Define the * operation
def star (a b : ℝ) : ℝ := (a + 3*b) - a*b

-- State the theorem
theorem star_inequality_equivalence :
  ∀ x : ℝ, star 5 x < 13 ↔ x > -4 :=
by sorry

end star_inequality_equivalence_l96_9632


namespace final_pens_count_l96_9642

def calculate_final_pens (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (alex_takes_percent : ℕ) (sharon_gets : ℕ) : ℕ :=
  let after_mike := initial + mike_gives
  let after_cindy := after_mike * cindy_multiplier
  let alex_takes := (after_cindy * alex_takes_percent + 99) / 100  -- Rounding up
  let after_alex := after_cindy - alex_takes
  after_alex - sharon_gets

theorem final_pens_count :
  calculate_final_pens 20 22 2 15 19 = 52 := by
  sorry

end final_pens_count_l96_9642


namespace pizza_cost_is_9_60_l96_9626

/-- The cost of a single box of pizza -/
def pizza_cost : ℝ := sorry

/-- The cost of a single can of soft drink -/
def soft_drink_cost : ℝ := 2

/-- The cost of a single hamburger -/
def hamburger_cost : ℝ := 3

/-- The number of pizza boxes Robert buys -/
def robert_pizza_boxes : ℕ := 5

/-- The number of soft drink cans Robert buys -/
def robert_soft_drinks : ℕ := 10

/-- The number of hamburgers Teddy buys -/
def teddy_hamburgers : ℕ := 6

/-- The number of soft drink cans Teddy buys -/
def teddy_soft_drinks : ℕ := 10

/-- The total amount spent by Robert and Teddy -/
def total_spent : ℝ := 106

theorem pizza_cost_is_9_60 :
  pizza_cost = 9.60 ∧
  (robert_pizza_boxes : ℝ) * pizza_cost +
  (robert_soft_drinks : ℝ) * soft_drink_cost +
  (teddy_hamburgers : ℝ) * hamburger_cost +
  (teddy_soft_drinks : ℝ) * soft_drink_cost = total_spent :=
sorry

end pizza_cost_is_9_60_l96_9626


namespace exam_score_problem_l96_9612

theorem exam_score_problem (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) : 
  total_questions = 80 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers wrong_answers : ℕ),
    correct_answers + wrong_answers = total_questions ∧
    correct_score * correct_answers + wrong_score * wrong_answers = total_score ∧
    correct_answers = 42 :=
by sorry

end exam_score_problem_l96_9612


namespace sales_increase_percentage_l96_9690

theorem sales_increase_percentage (price_reduction : ℝ) (receipts_increase : ℝ) : 
  price_reduction = 30 → receipts_increase = 5 → 
  ∃ (sales_increase : ℝ), sales_increase = 50 ∧ 
  (100 - price_reduction) / 100 * (1 + sales_increase / 100) = 1 + receipts_increase / 100 :=
by sorry

end sales_increase_percentage_l96_9690


namespace f_of_2_eq_0_l96_9638

def f (x : ℝ) : ℝ := (x - 1)^2 - (x - 1)

theorem f_of_2_eq_0 : f 2 = 0 := by
  sorry

end f_of_2_eq_0_l96_9638


namespace fraction_subtraction_l96_9666

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_l96_9666


namespace hyperbola_asymptote_slopes_l96_9606

/-- Given a hyperbola defined by the equation x²/16 - y²/25 = 1, 
    the slopes of its asymptotes are ±5/4. -/
theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), x^2/16 - y^2/25 = 1 →
  ∃ (m : ℝ), m = 5/4 ∧ (y = m*x ∨ y = -m*x) := by
sorry

end hyperbola_asymptote_slopes_l96_9606


namespace quadratic_triple_root_relation_l96_9663

/-- Given a quadratic equation ax^2 + bx + c = 0 where one root is triple the other,
    prove that 3b^2 = 16ac -/
theorem quadratic_triple_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end quadratic_triple_root_relation_l96_9663


namespace complex_arithmetic_proof_l96_9653

theorem complex_arithmetic_proof :
  let A : ℂ := 3 + 2*Complex.I
  let B : ℂ := -5
  let C : ℂ := 2*Complex.I
  let D : ℂ := 1 + 3*Complex.I
  A - B + C - D = 7 + Complex.I :=
by sorry

end complex_arithmetic_proof_l96_9653


namespace nail_polish_drying_time_l96_9654

theorem nail_polish_drying_time (total_time color_coat_time top_coat_time : ℕ) 
  (h1 : total_time = 13)
  (h2 : color_coat_time = 3)
  (h3 : top_coat_time = 5) :
  total_time - (2 * color_coat_time + top_coat_time) = 2 := by
  sorry

end nail_polish_drying_time_l96_9654


namespace point_on_parabola_l96_9601

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 - 3 * x + 1

/-- Theorem: The point (1/2, 0) lies on the parabola y = 2x^2 - 3x + 1 -/
theorem point_on_parabola : parabola (1/2) 0 := by sorry

end point_on_parabola_l96_9601


namespace parallelogram_angle_difference_l96_9603

theorem parallelogram_angle_difference (a b : ℝ) : 
  a = 65 → -- smaller angle is 65 degrees
  a + b = 180 → -- adjacent angles in a parallelogram are supplementary
  b - a = 50 := by
sorry

end parallelogram_angle_difference_l96_9603


namespace shares_to_buy_l96_9643

def wife_weekly_savings : ℕ := 100
def husband_monthly_savings : ℕ := 225
def weeks_per_month : ℕ := 4
def savings_period_months : ℕ := 4
def stock_price : ℕ := 50

def total_savings : ℕ :=
  (wife_weekly_savings * weeks_per_month + husband_monthly_savings) * savings_period_months

def investment_amount : ℕ := total_savings / 2

theorem shares_to_buy : investment_amount / stock_price = 25 := by
  sorry

end shares_to_buy_l96_9643


namespace alpha_value_l96_9688

theorem alpha_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan (α + β) = 3 →
  Real.tan β = 1/2 →
  α = π/4 := by
  sorry

end alpha_value_l96_9688


namespace curve_C_parametric_equations_l96_9694

/-- Given a curve C with polar equation ρ = 2cosθ, prove that its parametric equations are x = 1 + cosθ and y = sinθ -/
theorem curve_C_parametric_equations (θ : ℝ) :
  let ρ := 2 * Real.cos θ
  let x := 1 + Real.cos θ
  let y := Real.sin θ
  (x, y) ∈ {(x, y) : ℝ × ℝ | x^2 + y^2 = ρ^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ} :=
by sorry

end curve_C_parametric_equations_l96_9694


namespace largest_four_digit_sum_20_distinct_l96_9614

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 20) ∧
  (n / 1000 ≠ n / 100 % 10) ∧ (n / 1000 ≠ n / 10 % 10) ∧ (n / 1000 ≠ n % 10) ∧
  (n / 100 % 10 ≠ n / 10 % 10) ∧ (n / 100 % 10 ≠ n % 10) ∧
  (n / 10 % 10 ≠ n % 10)

theorem largest_four_digit_sum_20_distinct : 
  ∀ n : ℕ, is_valid_number n → n ≤ 9821 :=
sorry

end largest_four_digit_sum_20_distinct_l96_9614


namespace three_zeros_range_of_a_l96_9623

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a^2*x - 4*a

-- State the theorem
theorem three_zeros_range_of_a (a : ℝ) :
  a > 0 ∧ (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > Real.sqrt 2 :=
by sorry

end three_zeros_range_of_a_l96_9623


namespace factorization_problem_triangle_shape_l96_9616

-- Problem 1
theorem factorization_problem (a b : ℝ) :
  a^2 - 6*a*b + 9*b^2 - 36 = (a - 3*b - 6) * (a - 3*b + 6) := by sorry

-- Problem 2
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^2 + c^2 + 2*b^2 - 2*a*b - 2*b*c = 0) :
  a = b ∧ b = c := by sorry

end factorization_problem_triangle_shape_l96_9616


namespace car_speed_problem_l96_9698

/-- Proves that given a car traveling for two hours with a speed of 90 km/h in the first hour
    and an average speed of 72.5 km/h over the two hours, the speed in the second hour must be 55 km/h. -/
theorem car_speed_problem (speed_first_hour : ℝ) (average_speed : ℝ) (speed_second_hour : ℝ) :
  speed_first_hour = 90 →
  average_speed = 72.5 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_second_hour = 55 := by
  sorry

end car_speed_problem_l96_9698


namespace hilton_final_marbles_l96_9685

/-- Calculates the final number of marbles Hilton has after a series of events -/
def hiltons_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

/-- Theorem stating that given the initial conditions, Hilton ends up with 42 marbles -/
theorem hilton_final_marbles :
  hiltons_marbles 26 6 10 = 42 := by
  sorry

end hilton_final_marbles_l96_9685


namespace football_players_count_l96_9650

/-- Calculates the number of students playing football given the total number of students,
    the number of students playing cricket, the number of students playing neither sport,
    and the number of students playing both sports. -/
def students_playing_football (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) : ℕ :=
  total - neither - cricket + both

/-- Theorem stating that the number of students playing football is 325 -/
theorem football_players_count :
  students_playing_football 450 175 50 100 = 325 := by
  sorry

end football_players_count_l96_9650


namespace sufficient_not_necessary_l96_9649

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧
  (∃ a, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end sufficient_not_necessary_l96_9649


namespace tangent_lines_through_point_l96_9684

-- Define the curve
def f (x : ℝ) : ℝ := x^2

-- Define the point P
def P : ℝ × ℝ := (3, 5)

-- Define the two lines
def line1 (x : ℝ) : ℝ := 2*x - 1
def line2 (x : ℝ) : ℝ := 10*x - 25

theorem tangent_lines_through_point :
  ∀ m b : ℝ,
  (∃ x₀ : ℝ, 
    -- The line y = mx + b passes through P(3, 5)
    m * 3 + b = 5 ∧
    -- The line is tangent to the curve at some point (x₀, f(x₀))
    m * x₀ + b = f x₀ ∧
    m = 2 * x₀) →
  ((∀ x, m * x + b = line1 x) ∨ (∀ x, m * x + b = line2 x)) :=
sorry

end tangent_lines_through_point_l96_9684


namespace initial_fee_correct_l96_9667

/-- The initial fee for Jim's taxi service -/
def initial_fee : ℝ := 2.25

/-- The charge per 2/5 mile segment -/
def charge_per_segment : ℝ := 0.35

/-- The length of a trip in miles -/
def trip_length : ℝ := 3.6

/-- The total charge for the trip -/
def total_charge : ℝ := 5.4

/-- Theorem stating that the initial fee is correct given the conditions -/
theorem initial_fee_correct : 
  initial_fee + (trip_length / (2/5) * charge_per_segment) = total_charge :=
by sorry

end initial_fee_correct_l96_9667


namespace percentage_problem_l96_9615

theorem percentage_problem : ∃ x : ℝ, 
  (x / 100) * 150 - (20 / 100) * 250 = 43 ∧ 
  x = 62 := by
  sorry

end percentage_problem_l96_9615


namespace root_product_zero_l96_9624

theorem root_product_zero (α β c : ℝ) : 
  (α^2 - 4*α + c = 0) → 
  (β^2 - 4*β + c = 0) → 
  ((-α)^2 + 4*(-α) - c = 0) → 
  α * β = 0 := by
sorry

end root_product_zero_l96_9624


namespace cubic_sum_identity_l96_9622

theorem cubic_sum_identity 
  (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) := by
  sorry

end cubic_sum_identity_l96_9622


namespace remaining_speed_calculation_l96_9627

/-- Given a trip with the following characteristics:
  * Total distance of 80 miles
  * First 30 miles traveled at 30 mph
  * Average speed for the entire trip is 40 mph
  Prove that the speed for the remaining part of the trip is 50 mph -/
theorem remaining_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) 
  (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 80 ∧ 
  first_part_distance = 30 ∧ 
  first_part_speed = 30 ∧ 
  average_speed = 40 →
  (total_distance - first_part_distance) / 
    (total_distance / average_speed - first_part_distance / first_part_speed) = 50 :=
by sorry

end remaining_speed_calculation_l96_9627


namespace valid_pairs_l96_9607

def is_valid_pair (m n : ℕ+) : Prop :=
  (m^2 - n) ∣ (m + n^2) ∧ (n^2 - m) ∣ (n + m^2)

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ 
    ((m = 2 ∧ n = 2) ∨ 
     (m = 3 ∧ n = 3) ∨ 
     (m = 1 ∧ n = 2) ∨ 
     (m = 2 ∧ n = 1) ∨ 
     (m = 3 ∧ n = 2) ∨ 
     (m = 2 ∧ n = 3)) :=
by sorry

end valid_pairs_l96_9607


namespace common_roots_product_l96_9652

theorem common_roots_product (C D E : ℝ) : 
  ∃ (u v w t : ℂ), 
    (u^3 + C*u^2 + D*u + 20 = 0) ∧ 
    (v^3 + C*v^2 + D*v + 20 = 0) ∧ 
    (w^3 + C*w^2 + D*w + 20 = 0) ∧
    (u^3 + E*u^2 + 70 = 0) ∧ 
    (v^3 + E*v^2 + 70 = 0) ∧ 
    (t^3 + E*t^2 + 70 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 2 * Real.rpow 175 (1/3) :=
sorry

end common_roots_product_l96_9652


namespace arithmetic_mean_problem_l96_9673

theorem arithmetic_mean_problem (x : ℝ) : 
  (x + 8 + 15 + 2*x + 13 + 2*x + 4) / 5 = 24 → x = 16 := by
  sorry

end arithmetic_mean_problem_l96_9673


namespace min_sum_squares_l96_9680

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 2*x₂ + 3*x₃ = 120) : 
  ∀ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + 2*y₂ + 3*y₃ = 120 → 
  x₁^2 + x₂^2 + x₃^2 ≤ y₁^2 + y₂^2 + y₃^2 ∧ 
  ∃ x₁' x₂' x₃' : ℝ, x₁'^2 + x₂'^2 + x₃'^2 = 1400 ∧ 
                    x₁' > 0 ∧ x₂' > 0 ∧ x₃' > 0 ∧ 
                    x₁' + 2*x₂' + 3*x₃' = 120 := by
  sorry

end min_sum_squares_l96_9680


namespace sum_of_digits_5mul_permutation_l96_9641

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if two natural numbers are permutations of each other's digits -/
def isDigitPermutation (a b : ℕ) : Prop := sorry

/-- Theorem: If A is a permutation of B's digits, then sum of digits of 5A equals sum of digits of 5B -/
theorem sum_of_digits_5mul_permutation (A B : ℕ) :
  isDigitPermutation A B → sumOfDigits (5 * A) = sumOfDigits (5 * B) := by sorry

end sum_of_digits_5mul_permutation_l96_9641


namespace x_over_z_equals_five_l96_9631

theorem x_over_z_equals_five (x y z : ℚ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21)
  (eq4 : y / z = 6) :
  x / z = 5 := by sorry

end x_over_z_equals_five_l96_9631


namespace domain_equivalence_l96_9625

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ := {x | -2 < x ∧ x < 0}

-- Define the domain of f(2x-1)
def domain_f_2x_minus_1 (f : ℝ → ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem domain_equivalence (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ≠ 0) →
  (∀ x, x ∈ domain_f_2x_minus_1 f ↔ f (2 * x - 1) ≠ 0) :=
sorry

end domain_equivalence_l96_9625


namespace new_average_calculation_l96_9647

theorem new_average_calculation (num_students : ℕ) (original_avg : ℝ) 
  (increase_percent : ℝ) (bonus : ℝ) (new_avg : ℝ) : 
  num_students = 37 → 
  original_avg = 73 → 
  increase_percent = 65 → 
  bonus = 15 → 
  new_avg = original_avg * (1 + increase_percent / 100) + bonus →
  new_avg = 135.45 := by
sorry

end new_average_calculation_l96_9647


namespace essay_time_theorem_l96_9695

/-- Represents the time spent on various activities during essay writing -/
structure EssayWritingTime where
  wordsPerPage : ℕ
  timePerPageFirstDraft : ℕ
  researchTime : ℕ
  outlineTime : ℕ
  brainstormTime : ℕ
  firstDraftPages : ℕ
  timePerPageSecondDraft : ℕ
  breakTimePerPage : ℕ
  editingTime : ℕ
  proofreadingTime : ℕ

/-- Calculates the total time spent on writing the essay -/
def totalEssayTime (t : EssayWritingTime) : ℕ :=
  t.researchTime +
  t.outlineTime * 60 +
  t.brainstormTime +
  t.firstDraftPages * t.timePerPageFirstDraft +
  (t.firstDraftPages - 1) * t.breakTimePerPage +
  t.firstDraftPages * t.timePerPageSecondDraft +
  t.editingTime +
  t.proofreadingTime

/-- Theorem stating that the total time spent on the essay is 34900 seconds -/
theorem essay_time_theorem (t : EssayWritingTime)
  (h1 : t.wordsPerPage = 500)
  (h2 : t.timePerPageFirstDraft = 1800)
  (h3 : t.researchTime = 2700)
  (h4 : t.outlineTime = 15)
  (h5 : t.brainstormTime = 1200)
  (h6 : t.firstDraftPages = 6)
  (h7 : t.timePerPageSecondDraft = 1500)
  (h8 : t.breakTimePerPage = 600)
  (h9 : t.editingTime = 4500)
  (h10 : t.proofreadingTime = 1800) :
  totalEssayTime t = 34900 := by
  sorry

#eval totalEssayTime {
  wordsPerPage := 500,
  timePerPageFirstDraft := 1800,
  researchTime := 2700,
  outlineTime := 15,
  brainstormTime := 1200,
  firstDraftPages := 6,
  timePerPageSecondDraft := 1500,
  breakTimePerPage := 600,
  editingTime := 4500,
  proofreadingTime := 1800
}

end essay_time_theorem_l96_9695


namespace circle_condition_l96_9634

def is_circle (m : ℤ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), 
    x^2 + y^2 + m*x - m*y + 2 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

theorem circle_condition (m : ℤ) : 
  m ∈ ({0, 1, 2, 3} : Set ℤ) →
  (is_circle m ↔ m = 3) :=
by sorry

end circle_condition_l96_9634


namespace rachel_essay_time_l96_9655

/-- Represents the time spent on various activities of essay writing -/
structure EssayTime where
  research_time : ℕ  -- in minutes
  writing_rate : ℕ  -- pages per 30 minutes
  total_pages : ℕ
  editing_time : ℕ  -- in minutes

/-- Calculates the total time spent on an essay in hours -/
def total_essay_time (et : EssayTime) : ℚ :=
  let writing_time := (et.total_pages * 30) / 60  -- convert to hours
  let other_time := (et.research_time + et.editing_time) / 60  -- convert to hours
  writing_time + other_time

/-- Theorem stating that Rachel's total essay time is 5 hours -/
theorem rachel_essay_time :
  let rachel_essay := EssayTime.mk 45 1 6 75
  total_essay_time rachel_essay = 5 := by
  sorry

end rachel_essay_time_l96_9655


namespace hoseok_has_least_paper_l96_9620

def jungkook_paper : ℕ := 10
def hoseok_paper : ℕ := 7
def seokjin_paper : ℕ := jungkook_paper - 2

theorem hoseok_has_least_paper : 
  hoseok_paper < jungkook_paper ∧ hoseok_paper < seokjin_paper := by
sorry

end hoseok_has_least_paper_l96_9620


namespace loop_execution_l96_9678

theorem loop_execution (n α : ℕ) (β : ℚ) : 
  β = (n - 1) / 2^α →
  ∃ (ℓ m : ℕ → ℚ),
    (ℓ 0 = 0 ∧ m 0 = n - 1) ∧
    (∀ k, k < α → ℓ (k + 1) = ℓ k + 1 ∧ m (k + 1) = m k / 2) →
    ℓ α = α ∧ m α = β :=
sorry

end loop_execution_l96_9678


namespace jelly_bean_probability_l96_9637

theorem jelly_bean_probability (p_red p_orange p_green : ℝ) 
  (h_red : p_red = 0.1)
  (h_orange : p_orange = 0.4)
  (h_green : p_green = 0.2)
  (h_sum : p_red + p_orange + p_green + p_yellow = 1)
  (h_nonneg : p_yellow ≥ 0) :
  p_yellow = 0.3 := by
sorry

end jelly_bean_probability_l96_9637


namespace robert_ride_time_l96_9609

/-- The time taken for Robert to ride along a semicircular path on a highway section -/
theorem robert_ride_time :
  let highway_length : ℝ := 1 -- mile
  let highway_width : ℝ := 40 -- feet
  let robert_speed : ℝ := 5 -- miles per hour
  let feet_per_mile : ℝ := 5280
  let path_shape := Semicircle
  let time_taken := 
    (highway_length * feet_per_mile / highway_width) * (π * highway_width / 2) / 
    (robert_speed * feet_per_mile)
  time_taken = π / 10
  := by sorry

end robert_ride_time_l96_9609


namespace pythagorean_triple_l96_9648

theorem pythagorean_triple (n : ℕ) (h1 : n ≥ 3) (h2 : Odd n) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end pythagorean_triple_l96_9648


namespace prime_square_mod_180_l96_9651

theorem prime_square_mod_180 (p : Nat) (h_prime : Prime p) (h_gt_5 : p > 5) :
  p ^ 2 % 180 = 1 := by
  sorry

end prime_square_mod_180_l96_9651


namespace triangle_problem_l96_9671

-- Define a triangle with interior angles a, b, and x
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the property of being an acute triangle
def isAcute (t : Triangle) : Prop :=
  t.a < 90 ∧ t.b < 90 ∧ t.x < 90

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 60)
  (h2 : t.b = 70)
  (h3 : t.a + t.b + t.x = 180) : 
  t.x = 50 ∧ isAcute t := by
  sorry


end triangle_problem_l96_9671


namespace converse_statement_l96_9610

theorem converse_statement (x : ℝ) : 
  (∀ x, x ≥ 1 → x^2 + 3*x - 2 ≥ 0) →
  (∀ x, x^2 + 3*x - 2 < 0 → x < 1) :=
by sorry

end converse_statement_l96_9610


namespace f_one_lower_bound_l96_9681

/-- A function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x → x < y → f m x < f m y

theorem f_one_lower_bound (m : ℝ) (h : is_increasing_on_interval m) : f m 1 ≥ 25 := by
  sorry

end f_one_lower_bound_l96_9681


namespace tagged_fish_in_second_catch_l96_9662

theorem tagged_fish_in_second_catch 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (total_fish : ℕ) 
  (h1 : initial_tagged = 40)
  (h2 : second_catch = 40)
  (h3 : total_fish = 800) :
  ∃ (tagged_in_second : ℕ), 
    tagged_in_second = 2 ∧ 
    (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish :=
by
  sorry

end tagged_fish_in_second_catch_l96_9662


namespace rectangle_area_ratio_l96_9675

/-- Given two rectangles S₁ and S₂ with specific vertices and equal areas, prove that 360x/y = 810 --/
theorem rectangle_area_ratio (x y : ℝ) (hx : x < 9) (hy : y < 4) 
  (h_equal_area : x * (4 - y) = y * (9 - x)) : 
  360 * x / y = 810 := by
  sorry

end rectangle_area_ratio_l96_9675


namespace polynomial_evaluation_l96_9645

theorem polynomial_evaluation : 
  ∃ (x : ℝ), x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^4 - 3*x^3 + 2*x^2 + 5*x - 7 = 318 := by
  sorry

end polynomial_evaluation_l96_9645


namespace arithmetic_sequence_problem_l96_9659

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_3 = 15 and a_3 + a_4 = 20,
    prove that a_4 + a_5 = 25. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum1 : a 2 + a 3 = 15)
    (h_sum2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 := by
  sorry

end arithmetic_sequence_problem_l96_9659


namespace intersection_of_A_and_B_l96_9693

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l96_9693


namespace vector_equation_l96_9697

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)

theorem vector_equation : A - D + D - C - (A - B) = B - C := by sorry

end vector_equation_l96_9697


namespace survey_analysis_l96_9682

/-- Data from the survey --/
structure SurveyData where
  a : Nat  -- Females who understand
  b : Nat  -- Females who do not understand
  c : Nat  -- Males who understand
  d : Nat  -- Males who do not understand

/-- Chi-square calculation function --/
def chiSquare (data : SurveyData) : Rat :=
  let n := data.a + data.b + data.c + data.d
  n * (data.a * data.d - data.b * data.c)^2 / 
    ((data.a + data.b) * (data.c + data.d) * (data.a + data.c) * (data.b + data.d))

/-- Binomial probability calculation function --/
def binomialProb (n k : Nat) (p : Rat) : Rat :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Main theorem --/
theorem survey_analysis (data : SurveyData) 
    (h_data : data.a = 140 ∧ data.b = 60 ∧ data.c = 180 ∧ data.d = 20) :
    chiSquare data = 25 ∧ 
    chiSquare data > (10828 : Rat) / 1000 ∧
    binomialProb 5 3 (4/5) = 128/625 := by
  sorry

#eval chiSquare ⟨140, 60, 180, 20⟩
#eval binomialProb 5 3 (4/5)

end survey_analysis_l96_9682


namespace correct_num_arrangements_l96_9656

/-- The number of different arrangements of 5 boys and 2 girls in a row,
    where one boy (A) must stand in the center and the two girls must stand next to each other. -/
def num_arrangements : ℕ :=
  Nat.choose 4 1 * Nat.factorial 2 * Nat.factorial 4

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_num_arrangements :
  num_arrangements = Nat.choose 4 1 * Nat.factorial 2 * Nat.factorial 4 := by
  sorry

end correct_num_arrangements_l96_9656


namespace circle_radius_squared_l96_9665

theorem circle_radius_squared (r : ℝ) 
  (AB CD : ℝ) (angle_APD : ℝ) (BP : ℝ) : 
  AB = 10 → 
  CD = 7 → 
  angle_APD = 60 * π / 180 → 
  BP = 8 → 
  r^2 = 73 := by
sorry

end circle_radius_squared_l96_9665


namespace no_hexagon_with_special_point_l96_9605

-- Define a hexagon as a set of 6 points in 2D space
def Hexagon := Fin 6 → ℝ × ℝ

-- Define convexity for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define a point being inside a hexagon
def is_inside (p : ℝ × ℝ) (h : Hexagon) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem no_hexagon_with_special_point :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    is_inside m h ∧
    (∀ i j : Fin 6, i ≠ j → distance (h i) (h j) > 1) ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
by sorry

end no_hexagon_with_special_point_l96_9605


namespace power_function_through_point_l96_9604

/-- A power function that passes through the point (9, 3) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem stating that f(9) = 3 -/
theorem power_function_through_point : f 9 = 3 := by
  sorry

end power_function_through_point_l96_9604


namespace als_original_portion_l96_9602

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1000 →
  a - 100 + 2*b + 2*c = 1500 →
  a = 400 :=
by sorry

end als_original_portion_l96_9602


namespace quadratic_discriminant_with_specific_roots_l96_9613

/-- The discriminant of a quadratic polynomial with specific root conditions -/
theorem quadratic_discriminant_with_specific_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) ∧
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4 * a * c = -1/2 := by
sorry

end quadratic_discriminant_with_specific_roots_l96_9613


namespace simplify_polynomial_subtraction_l96_9679

theorem simplify_polynomial_subtraction (r : ℝ) : 
  (r^2 + 3*r - 2) - (r^2 + 7*r - 5) = -4*r + 3 := by
  sorry

end simplify_polynomial_subtraction_l96_9679


namespace four_point_theorem_l96_9658

/-- Given four points A, B, C, D in a plane, if for any point P the inequality 
    PA + PD ≥ PB + PC holds, then B and C lie on the segment AD and AB = CD. -/
theorem four_point_theorem (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (∀ P : EuclideanSpace ℝ (Fin 2), dist P A + dist P D ≥ dist P B + dist P C) →
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 
    B = (1 - t₁) • A + t₁ • D ∧ 
    C = (1 - t₂) • A + t₂ • D) ∧
  dist A B = dist C D := by
  sorry

end four_point_theorem_l96_9658


namespace greatest_prime_factor_of_210_l96_9617

theorem greatest_prime_factor_of_210 : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ 210 ∧ ∀ (q : ℕ), q.Prime → q ∣ 210 → q ≤ p :=
by sorry

end greatest_prime_factor_of_210_l96_9617


namespace intersection_complement_equality_l96_9686

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Finset Nat := {3, 4, 5}
def B : Finset Nat := {1, 3, 6}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {4, 5} := by sorry

end intersection_complement_equality_l96_9686


namespace negative_twenty_seven_to_five_thirds_l96_9635

theorem negative_twenty_seven_to_five_thirds :
  (-27 : ℝ) ^ (5/3) = -243 := by
  sorry

end negative_twenty_seven_to_five_thirds_l96_9635


namespace abs_neg_2023_l96_9672

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l96_9672


namespace peach_difference_l96_9644

theorem peach_difference (audrey_peaches paul_peaches : ℕ) 
  (h1 : audrey_peaches = 26) 
  (h2 : paul_peaches = 48) : 
  paul_peaches - audrey_peaches = 22 := by
  sorry

end peach_difference_l96_9644


namespace find_x_l96_9611

-- Define the variables
variable (a b x : ℝ)
variable (r : ℝ)

-- State the theorem
theorem find_x (h1 : b ≠ 0) (h2 : r = (3 * a) ^ (2 * b)) (h3 : r = a ^ b * x ^ (2 * b)) : x = 3 * Real.sqrt a := by
  sorry

end find_x_l96_9611


namespace engagement_treats_ratio_l96_9633

def total_value : ℕ := 158000
def hotel_cost_per_night : ℕ := 4000
def nights_stayed : ℕ := 2
def car_value : ℕ := 30000

theorem engagement_treats_ratio :
  let hotel_total := hotel_cost_per_night * nights_stayed
  let non_house_total := hotel_total + car_value
  let house_value := total_value - non_house_total
  house_value / car_value = 4 := by
sorry

end engagement_treats_ratio_l96_9633


namespace sugar_calculation_l96_9600

/-- Given a recipe with a sugar to flour ratio and an amount of flour,
    calculate the amount of sugar needed. -/
def sugar_amount (sugar_flour_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  sugar_flour_ratio * flour_amount

theorem sugar_calculation (sugar_flour_ratio flour_amount : ℚ) :
  sugar_flour_ratio = 10 / 1 →
  flour_amount = 5 →
  sugar_amount sugar_flour_ratio flour_amount = 50 := by
sorry

end sugar_calculation_l96_9600


namespace concentric_circles_ratio_l96_9669

theorem concentric_circles_ratio (s S : ℝ) (h : s > 0) (H : S > s) :
  (π * S^2 = 3/2 * (π * S^2 - π * s^2)) → S/s = Real.sqrt 3 := by
  sorry

end concentric_circles_ratio_l96_9669


namespace valid_words_count_l96_9621

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum length of a word -/
def max_word_length : ℕ := 5

/-- The number of words of length n that do not contain the letter A -/
def words_without_a (n : ℕ) : ℕ := (alphabet_size - 1) ^ n

/-- The total number of possible words of length n -/
def total_words (n : ℕ) : ℕ := alphabet_size ^ n

/-- The number of words of length n that contain the letter A at least once -/
def words_with_a (n : ℕ) : ℕ := total_words n - words_without_a n

/-- The total number of valid words -/
def total_valid_words : ℕ :=
  words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5

theorem valid_words_count : total_valid_words = 1863701 := by
  sorry

end valid_words_count_l96_9621


namespace max_value_is_110003_l96_9661

/-- The set of given integers --/
def given_integers : Finset ℤ := {100004, 110003, 102002, 100301, 100041}

/-- Theorem stating that 110003 is the maximum value in the given set of integers --/
theorem max_value_is_110003 : 
  ∀ x ∈ given_integers, x ≤ 110003 ∧ 110003 ∈ given_integers := by
  sorry

#check max_value_is_110003

end max_value_is_110003_l96_9661


namespace min_value_of_reciprocal_sum_l96_9674

-- Define the function f
def f (b c x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- State the theorem
theorem min_value_of_reciprocal_sum (b c : ℝ) (x₁ x₂ : ℝ) :
  (∃ (b c : ℝ), f b c (-10) = f b c 12) →  -- f(-10) = f(12)
  (x₁ > 0 ∧ x₂ > 0) →  -- x₁ and x₂ are positive
  (f b c x₁ = 0 ∧ f b c x₂ = 0) →  -- x₁ and x₂ are roots of f(x) = 0
  (∀ y z : ℝ, y > 0 ∧ z > 0 ∧ f b c y = 0 ∧ f b c z = 0 → 1/y + 1/z ≥ 1/x₁ + 1/x₂) →  -- x₁ and x₂ give the minimum value
  1/x₁ + 1/x₂ = 2 :=
by sorry

end min_value_of_reciprocal_sum_l96_9674


namespace sum_25887_2014_not_even_l96_9664

theorem sum_25887_2014_not_even : ¬ Even (25887 + 2014) := by
  sorry

end sum_25887_2014_not_even_l96_9664


namespace arithmetic_sequence_sum_remainder_l96_9629

/-- The sum of an arithmetic sequence with first term a, last term l, and common difference d -/
def arithmetic_sum (a l d : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

/-- The theorem stating that the remainder of the sum of the given arithmetic sequence when divided by 8 is 2 -/
theorem arithmetic_sequence_sum_remainder :
  (arithmetic_sum 3 299 8) % 8 = 2 := by sorry

end arithmetic_sequence_sum_remainder_l96_9629


namespace tax_revenue_change_l96_9636

theorem tax_revenue_change (T C : ℝ) (T_new C_new R_new : ℝ) : 
  T_new = T * 0.9 →
  C_new = C * 1.1 →
  R_new = T_new * C_new →
  R_new = T * C * 0.99 := by
sorry

end tax_revenue_change_l96_9636


namespace fraction_invariance_l96_9692

theorem fraction_invariance (a b m n : ℚ) (h : b ≠ 0) (h' : b + n ≠ 0) :
  a / b = m / n → (a + m) / (b + n) = a / b := by
  sorry

end fraction_invariance_l96_9692


namespace no_blonde_girls_added_l96_9687

/-- The number of blonde girls added to a choir -/
def blonde_girls_added (initial_total : ℕ) (initial_blonde : ℕ) (black_haired : ℕ) : ℕ :=
  initial_total - initial_blonde - black_haired

/-- Theorem: Given the initial conditions, no blonde girls were added to the choir -/
theorem no_blonde_girls_added :
  blonde_girls_added 80 30 50 = 0 := by
  sorry

end no_blonde_girls_added_l96_9687


namespace floor_with_57_diagonal_tiles_has_841_total_tiles_l96_9691

/-- Represents a rectangular floor covered with square tiles -/
structure TiledFloor where
  length : ℕ
  width : ℕ
  diagonal_tiles : ℕ

/-- The total number of tiles on a rectangular floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.length * floor.width

/-- Theorem stating that a rectangular floor with 57 tiles on its diagonals has 841 tiles in total -/
theorem floor_with_57_diagonal_tiles_has_841_total_tiles :
  ∃ (floor : TiledFloor), floor.diagonal_tiles = 57 ∧ total_tiles floor = 841 := by
  sorry


end floor_with_57_diagonal_tiles_has_841_total_tiles_l96_9691
