import Mathlib

namespace equidistant_from_axes_l3075_307579

/-- A point in the 2D plane is equidistant from both coordinate axes if and only if the square of its x-coordinate equals the square of its y-coordinate. -/
theorem equidistant_from_axes (x y : ℝ) : (|x| = |y|) ↔ (x^2 = y^2) := by sorry

end equidistant_from_axes_l3075_307579


namespace max_value_of_sum_products_l3075_307557

theorem max_value_of_sum_products (a b c : ℝ) (h : a + b + 3 * c = 6) :
  ∃ (max : ℝ), max = 516 / 49 ∧ ∀ (x y z : ℝ), x + y + 3 * z = 6 → x * y + x * z + y * z ≤ max :=
by sorry

end max_value_of_sum_products_l3075_307557


namespace arithmetic_progression_sum_3n_l3075_307541

/-- An arithmetic progression with partial sums S_n -/
structure ArithmeticProgression where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- Given S_n = a and S_{2n} = b, prove S_{3n} = 3b - 2a -/
theorem arithmetic_progression_sum_3n 
  (ap : ArithmeticProgression) (n : ℕ) (a b : ℝ) 
  (h1 : ap.S n = a) 
  (h2 : ap.S (2 * n) = b) : 
  ap.S (3 * n) = 3 * b - 2 * a := by
  sorry

end arithmetic_progression_sum_3n_l3075_307541


namespace inequality_proof_l3075_307513

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l3075_307513


namespace remainder_of_factorial_sum_l3075_307523

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_of_factorial_sum (n : ℕ) (h : n ≥ 100) :
  (sum_factorials n) % 30 = (sum_factorials 4) % 30 := by
  sorry

end remainder_of_factorial_sum_l3075_307523


namespace initial_disappearance_percentage_l3075_307518

/-- Proof of the initial percentage of inhabitants that disappeared from a village --/
theorem initial_disappearance_percentage 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (initial_population_eq : initial_population = 7600)
  (final_population_eq : final_population = 5130) :
  ∃ (p : ℝ), 
    p = 10 ∧ 
    (initial_population : ℝ) * (1 - p / 100) * 0.75 = final_population := by
  sorry

end initial_disappearance_percentage_l3075_307518


namespace range_of_a_l3075_307511

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + abs x)

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | f (x^2 + 1) > f (a * x)}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → x ∈ A a) →
  a ∈ Set.Ioo (-5/2) (5/2) :=
sorry

end range_of_a_l3075_307511


namespace multiple_of_nine_is_multiple_of_three_l3075_307596

theorem multiple_of_nine_is_multiple_of_three (n : ℤ) : 
  (∃ k : ℤ, n = 9 * k) → (∃ m : ℤ, n = 3 * m) := by
  sorry

end multiple_of_nine_is_multiple_of_three_l3075_307596


namespace scout_weekend_earnings_280_l3075_307563

/-- Calculates Scout's earnings for the weekend given the specified conditions --/
def scout_weekend_earnings (base_pay : ℕ) (sat_hours : ℕ) (sat_deliveries : ℕ) (sat_tip : ℕ)
  (sun_hours : ℕ) (sun_deliveries : ℕ) (sun_low_tip : ℕ) (sun_high_tip : ℕ)
  (transport_cost : ℕ) : ℕ :=
  let sat_earnings := base_pay * sat_hours + sat_deliveries * sat_tip - sat_deliveries * transport_cost
  let sun_earnings := 2 * base_pay * sun_hours + (sun_deliveries / 2) * (sun_low_tip + sun_high_tip) - sun_deliveries * transport_cost
  sat_earnings + sun_earnings

/-- Theorem stating that Scout's weekend earnings are $280.00 --/
theorem scout_weekend_earnings_280 :
  scout_weekend_earnings 10 6 5 5 8 10 3 7 1 = 280 := by
  sorry

#eval scout_weekend_earnings 10 6 5 5 8 10 3 7 1

end scout_weekend_earnings_280_l3075_307563


namespace average_of_combined_data_points_l3075_307598

theorem average_of_combined_data_points (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  let total_points := n1 + n2
  let combined_avg := (n1 * avg1 + n2 * avg2) / total_points
  combined_avg = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

end average_of_combined_data_points_l3075_307598


namespace overbridge_length_l3075_307519

/-- Calculates the length of an overbridge given train parameters --/
theorem overbridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) : 
  train_length = 600 →
  train_speed_kmh = 36 →
  crossing_time = 70 →
  (train_length + (train_speed_kmh * 1000 / 3600 * crossing_time)) - train_length = 100 :=
by
  sorry

end overbridge_length_l3075_307519


namespace bicycle_price_adjustment_l3075_307551

theorem bicycle_price_adjustment (original_price : ℝ) 
  (wednesday_discount : ℝ) (friday_increase : ℝ) (saturday_discount : ℝ) : 
  original_price = 200 →
  wednesday_discount = 0.40 →
  friday_increase = 0.20 →
  saturday_discount = 0.25 →
  original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount) = 108 := by
  sorry

end bicycle_price_adjustment_l3075_307551


namespace family_birth_years_l3075_307527

def current_year : ℕ := 1967

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def satisfies_condition (birth_year : ℕ) (multiplier : ℕ) : Prop :=
  current_year - birth_year = multiplier * sum_of_digits birth_year

theorem family_birth_years :
  ∃ (grandpa eldest_son father pali brother mother grandfather grandmother : ℕ),
    satisfies_condition grandpa 3 ∧
    satisfies_condition eldest_son 3 ∧
    satisfies_condition father 3 = false ∧
    satisfies_condition (father - 1) 3 ∧
    satisfies_condition grandfather 3 = false ∧
    satisfies_condition (grandfather - 1) 3 ∧
    satisfies_condition grandmother 3 = false ∧
    satisfies_condition (grandmother + 1) 3 ∧
    satisfies_condition mother 2 = false ∧
    satisfies_condition (mother - 1) 2 ∧
    satisfies_condition pali 1 ∧
    satisfies_condition brother 1 = false ∧
    satisfies_condition (brother - 1) 1 ∧
    grandpa = 1889 ∧
    eldest_son = 1916 ∧
    father = 1928 ∧
    pali = 1951 ∧
    brother = 1947 ∧
    mother = 1934 ∧
    grandfather = 1896 ∧
    grandmother = 1909 :=
by
  sorry

end family_birth_years_l3075_307527


namespace triangle_radii_inequality_l3075_307590

theorem triangle_radii_inequality (r R α β γ : Real) : 
  r > 0 → R > 0 → 
  0 < α ∧ α < π → 0 < β ∧ β < π → 0 < γ ∧ γ < π →
  α + β + γ = π →
  r / R ≤ 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) := by
  sorry

end triangle_radii_inequality_l3075_307590


namespace parabola_through_point_2_4_l3075_307555

-- Define a parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a function to check if a point is on the parabola
def on_parabola (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Theorem statement
theorem parabola_through_point_2_4 :
  ∃ (p : Parabola), 
    (on_parabola p 2 4) ∧ 
    ((∀ x y : ℝ, p.equation x y ↔ y^2 = 8*x) ∨ 
     (∀ x y : ℝ, p.equation x y ↔ x^2 = y)) :=
sorry

end parabola_through_point_2_4_l3075_307555


namespace A_power_150_is_identity_l3075_307542

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_150_is_identity :
  A ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end A_power_150_is_identity_l3075_307542


namespace ratio_problem_l3075_307581

theorem ratio_problem (antecedent consequent : ℚ) : 
  antecedent / consequent = 4 / 6 → antecedent = 20 → consequent = 30 := by
  sorry

end ratio_problem_l3075_307581


namespace second_gym_signup_fee_covers_four_months_l3075_307547

-- Define the given constants
def cheap_monthly_fee : ℤ := 10
def cheap_signup_fee : ℤ := 50
def total_paid_first_year : ℤ := 650
def months_in_year : ℕ := 12

-- Define the relationships
def second_monthly_fee : ℤ := 3 * cheap_monthly_fee

-- State the theorem
theorem second_gym_signup_fee_covers_four_months :
  ∃ (second_signup_fee : ℤ),
    (cheap_monthly_fee * months_in_year + cheap_signup_fee +
     second_monthly_fee * months_in_year + second_signup_fee = total_paid_first_year) ∧
    (second_signup_fee / second_monthly_fee = 4) := by
  sorry

end second_gym_signup_fee_covers_four_months_l3075_307547


namespace cube_and_square_root_problem_l3075_307502

theorem cube_and_square_root_problem (a b : ℝ) 
  (h1 : (2*b - 2*a)^(1/3 : ℝ) = -2)
  (h2 : (4*a + 3*b)^(1/2 : ℝ) = 3) :
  a = 3 ∧ b = -1 ∧ (5*a - b)^(1/2 : ℝ) = 4 ∨ (5*a - b)^(1/2 : ℝ) = -4 :=
by sorry

end cube_and_square_root_problem_l3075_307502


namespace white_ball_count_l3075_307529

/-- Given a bag with red and white balls, if the probability of drawing a red ball
    is 1/4 and there are 5 red balls, prove that there are 15 white balls. -/
theorem white_ball_count (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
    (h1 : red_balls = 5)
    (h2 : total_balls = red_balls + white_balls)
    (h3 : (red_balls : ℚ) / total_balls = 1 / 4) :
  white_balls = 15 := by
  sorry

end white_ball_count_l3075_307529


namespace radio_station_survey_l3075_307540

theorem radio_station_survey (total_listeners total_non_listeners male_non_listeners female_listeners : ℕ) 
  (h1 : total_listeners = 180)
  (h2 : total_non_listeners = 160)
  (h3 : male_non_listeners = 85)
  (h4 : female_listeners = 75) :
  total_listeners - female_listeners = 105 := by
  sorry

end radio_station_survey_l3075_307540


namespace bobs_walking_rate_l3075_307516

/-- Proves that Bob's walking rate is 5 miles per hour given the problem conditions -/
theorem bobs_walking_rate
  (total_distance : ℝ)
  (yolanda_rate : ℝ)
  (bob_start_delay : ℝ)
  (bob_distance : ℝ)
  (h1 : total_distance = 60)
  (h2 : yolanda_rate = 5)
  (h3 : bob_start_delay = 1)
  (h4 : bob_distance = 30) :
  bob_distance / (total_distance / yolanda_rate - bob_start_delay) = 5 :=
by sorry

end bobs_walking_rate_l3075_307516


namespace modified_fibonacci_sum_l3075_307546

def G : ℕ → ℚ
  | 0 => 2
  | 1 => 1
  | (n + 2) => G (n + 1) + G n

theorem modified_fibonacci_sum :
  (∑' n, G n / 5^n) = 280 / 99 := by
  sorry

end modified_fibonacci_sum_l3075_307546


namespace article_cost_l3075_307505

/-- Represents the cost and selling price of an article -/
structure Article where
  cost : ℝ
  sellingPrice : ℝ

/-- The original article with 25% profit -/
def originalArticle : Article → Prop := fun a => 
  a.sellingPrice = 1.25 * a.cost

/-- The new article with reduced cost and selling price -/
def newArticle : Article → Prop := fun a => 
  (0.8 * a.cost) * 1.3 = a.sellingPrice - 16.8

/-- Theorem stating that the cost of the article is 80 -/
theorem article_cost : ∃ a : Article, originalArticle a ∧ newArticle a ∧ a.cost = 80 := by
  sorry

end article_cost_l3075_307505


namespace sheets_colored_l3075_307573

/-- Given 2450 sheets of paper evenly split into 5 binders,
    prove that coloring one-half of the sheets in one binder uses 245 sheets. -/
theorem sheets_colored (total_sheets : ℕ) (num_binders : ℕ) (sheets_per_binder : ℕ) :
  total_sheets = 2450 →
  num_binders = 5 →
  total_sheets = num_binders * sheets_per_binder →
  sheets_per_binder / 2 = 245 := by
  sorry

#check sheets_colored

end sheets_colored_l3075_307573


namespace geometric_series_first_term_l3075_307520

/-- Given an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 30 and the sum of the squares of its terms is 120,
    then the first term a is equal to 120/17. -/
theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) :
  a = 120 / 17 := by
  sorry

end geometric_series_first_term_l3075_307520


namespace sum_of_coefficients_l3075_307564

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (x - 2)^6 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64 := by
sorry

end sum_of_coefficients_l3075_307564


namespace sufficient_not_necessary_condition_l3075_307586

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a > b ∧ b > 0 → 1/a < 1/b) ∧
  (∃ a b, 1/a < 1/b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end sufficient_not_necessary_condition_l3075_307586


namespace arithmetic_square_root_of_sqrt_16_l3075_307509

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l3075_307509


namespace area_triangle_abc_l3075_307560

/-- Given a point A(x, y) where x ≠ 0 and y ≠ 0, with B symmetric to A with respect to the x-axis,
    C symmetric to A with respect to the y-axis, and the area of triangle AOB equal to 4,
    prove that the area of triangle ABC is equal to 8. -/
theorem area_triangle_abc (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  let A : ℝ × ℝ := (x, y)
  let B : ℝ × ℝ := (x, -y)
  let C : ℝ × ℝ := (-x, y)
  let O : ℝ × ℝ := (0, 0)
  let area_AOB := abs (x * y)
  area_AOB = 4 → abs (2 * x * y) = 8 :=
by sorry

end area_triangle_abc_l3075_307560


namespace smallest_n_for_sqrt_20n_integer_l3075_307501

theorem smallest_n_for_sqrt_20n_integer (n : ℕ) : 
  (∃ k : ℕ, k ^ 2 = 20 * n) → (∀ m : ℕ, m > 0 ∧ m < n → ¬∃ k : ℕ, k ^ 2 = 20 * m) → n = 5 := by
  sorry

end smallest_n_for_sqrt_20n_integer_l3075_307501


namespace midpoint_locus_l3075_307526

/-- The locus of midpoints of line segments from P(4, -2) to points on x^2 + y^2 = 4 -/
theorem midpoint_locus (x y u v : ℝ) : 
  (u^2 + v^2 = 4) →  -- Point (u, v) is on the circle
  (x = (u + 4) / 2 ∧ y = (v - 2) / 2) →  -- (x, y) is the midpoint
  (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry

end midpoint_locus_l3075_307526


namespace cube_order_preserving_l3075_307510

theorem cube_order_preserving (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 < b^3 := by
  sorry

end cube_order_preserving_l3075_307510


namespace janice_office_floor_l3075_307572

/-- The floor number of Janice's office -/
def office_floor : ℕ := 3

/-- The number of times Janice goes up the stairs per day -/
def up_times : ℕ := 5

/-- The number of times Janice goes down the stairs per day -/
def down_times : ℕ := 3

/-- The total number of flights of stairs Janice walks in a day -/
def total_flights : ℕ := 24

theorem janice_office_floor :
  office_floor * (up_times + down_times) = total_flights :=
sorry

end janice_office_floor_l3075_307572


namespace nonagon_ribbon_theorem_l3075_307574

def nonagon_ribbon_length (a b c d e f g h i : ℝ) : Prop :=
  a + b + c + d + e + f + g + h + i = 62 →
  1.5 * (a + b + c + d + e + f + g + h + i) = 93

theorem nonagon_ribbon_theorem :
  ∀ a b c d e f g h i : ℝ, nonagon_ribbon_length a b c d e f g h i :=
by
  sorry

end nonagon_ribbon_theorem_l3075_307574


namespace possible_theta_value_l3075_307594

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

theorem possible_theta_value :
  ∃ θ : ℝ,
    (∀ x : ℝ, (2015 : ℝ) ^ (f θ (-x)) = 1 / ((2015 : ℝ) ^ (f θ x))) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/4 → f θ y < f θ x) ∧
    θ = 2 * π / 3 :=
by sorry

end possible_theta_value_l3075_307594


namespace intersection_point_y_coordinate_l3075_307504

/-- Given that point A is an intersection point of y = ax and y = (4-a)/x with x-coordinate 1,
    prove that the y-coordinate of A is 2. -/
theorem intersection_point_y_coordinate (a : ℝ) :
  (∃ A : ℝ × ℝ, A.1 = 1 ∧ A.2 = a * A.1 ∧ A.2 = (4 - a) / A.1) →
  (∃ A : ℝ × ℝ, A.1 = 1 ∧ A.2 = a * A.1 ∧ A.2 = (4 - a) / A.1 ∧ A.2 = 2) :=
by sorry


end intersection_point_y_coordinate_l3075_307504


namespace exam_score_calculation_l3075_307506

/-- Given an exam with mean score and a score below the mean, calculate the score above the mean -/
theorem exam_score_calculation (mean : ℝ) (below_score : ℝ) (below_sd : ℝ) (above_sd : ℝ)
  (h1 : mean = 76)
  (h2 : below_score = 60)
  (h3 : below_sd = 2)
  (h4 : above_sd = 3)
  (h5 : below_score = mean - below_sd * ((mean - below_score) / below_sd)) :
  mean + above_sd * ((mean - below_score) / below_sd) = 100 := by
sorry

end exam_score_calculation_l3075_307506


namespace product_seventeen_reciprocal_squares_sum_l3075_307514

theorem product_seventeen_reciprocal_squares_sum (x y : ℕ) :
  x * y = 17 → (1 : ℚ) / x^2 + 1 / y^2 = 290 / 289 := by
  sorry

end product_seventeen_reciprocal_squares_sum_l3075_307514


namespace min_value_of_P_l3075_307522

/-- The polynomial P as a function of a real number a -/
def P (a : ℝ) : ℝ := a^2 + 4*a + 2014

/-- Theorem stating that the minimum value of P is 2010 -/
theorem min_value_of_P :
  ∃ (min : ℝ), min = 2010 ∧ ∀ (a : ℝ), P a ≥ min :=
sorry

end min_value_of_P_l3075_307522


namespace train_speed_l3075_307525

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 6) :
  length / time = 140 / 6 := by
  sorry

end train_speed_l3075_307525


namespace inequality_solution_l3075_307589

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(x + 5) < a^(4*x - 1) ↔ (0 < a ∧ a < 1 ∧ x < 2) ∨ (a > 1 ∧ x > 2)) :=
by sorry

end inequality_solution_l3075_307589


namespace horner_v2_value_l3075_307556

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def v0 : ℝ := 3
def v1 (x : ℝ) : ℝ := v0 * x + 5
def v2 (x : ℝ) : ℝ := v1 x * x + 6

theorem horner_v2_value :
  v2 (-4) = 34 := by sorry

end horner_v2_value_l3075_307556


namespace liquid_x_percentage_in_mixed_solution_l3075_307536

/-- Given two solutions P and Q, where liquid X makes up 0.5% of P and 1.5% of Q,
    prove that mixing 200g of P with 800g of Q results in a solution containing 1.3% liquid X. -/
theorem liquid_x_percentage_in_mixed_solution :
  let p_weight : ℝ := 200
  let q_weight : ℝ := 800
  let p_percentage : ℝ := 0.5
  let q_percentage : ℝ := 1.5
  let x_in_p : ℝ := p_weight * (p_percentage / 100)
  let x_in_q : ℝ := q_weight * (q_percentage / 100)
  let total_x : ℝ := x_in_p + x_in_q
  let total_weight : ℝ := p_weight + q_weight
  let result_percentage : ℝ := (total_x / total_weight) * 100
  result_percentage = 1.3 := by sorry

end liquid_x_percentage_in_mixed_solution_l3075_307536


namespace expression_equals_zero_l3075_307524

theorem expression_equals_zero (x : ℚ) (h : x = 1/3) :
  (2*x + 1) * (2*x - 1) + x * (3 - 4*x) = 0 := by
  sorry

end expression_equals_zero_l3075_307524


namespace kevin_bought_two_watermelons_l3075_307512

-- Define the weights of the watermelons and the total weight
def weight1 : ℝ := 9.91
def weight2 : ℝ := 4.11
def totalWeight : ℝ := 14.02

-- Define the number of watermelons Kevin bought
def numberOfWatermelons : ℕ := 2

-- Theorem to prove
theorem kevin_bought_two_watermelons :
  weight1 + weight2 = totalWeight ∧ numberOfWatermelons = 2 :=
by sorry

end kevin_bought_two_watermelons_l3075_307512


namespace totalDays_is_25_l3075_307576

/-- Calculates the total number of days in a work period given the following conditions:
  * A woman is paid $20 for each day she works
  * She forfeits $5 for each day she is idle
  * She nets $450
  * She worked for 23 days
-/
def totalDaysInPeriod (dailyPay : ℕ) (dailyForfeit : ℕ) (netEarnings : ℕ) (daysWorked : ℕ) : ℕ :=
  sorry

/-- Proves that the total number of days in the period is 25 -/
theorem totalDays_is_25 :
  totalDaysInPeriod 20 5 450 23 = 25 := by
  sorry

end totalDays_is_25_l3075_307576


namespace inequality_implication_l3075_307561

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_implication_l3075_307561


namespace snail_well_depth_l3075_307549

/-- The minimum depth of a well that allows a snail to reach the top during the day on the fifth day,
    given its daily climbing and nightly sliding distances. -/
def min_well_depth (day_climb : ℕ) (night_slide : ℕ) : ℕ :=
  (day_climb - night_slide) * 3 + day_climb + 1

/-- Theorem stating the minimum well depth for a snail with specific climbing characteristics. -/
theorem snail_well_depth :
  min_well_depth 110 40 = 321 := by
  sorry

#eval min_well_depth 110 40

end snail_well_depth_l3075_307549


namespace fraction_value_l3075_307578

theorem fraction_value : (150 + (150 / 10)) / (15 - 5) = 16.5 := by
  sorry

end fraction_value_l3075_307578


namespace kConnectedSubgraph_l3075_307569

/-- A graph G is a pair (V, E) where V is a finite set of vertices and E is a set of edges. -/
structure Graph (α : Type*) where
  V : Finset α
  E : Finset (α × α)

/-- The minimum degree of a graph G. -/
def minDegree {α : Type*} (G : Graph α) : ℕ :=
  sorry

/-- A graph G is k-connected if it remains connected after removing any k-1 vertices. -/
def isKConnected {α : Type*} (G : Graph α) (k : ℕ) : Prop :=
  sorry

/-- A subgraph H of G is a graph whose vertices and edges are subsets of G's vertices and edges. -/
def isSubgraph {α : Type*} (H G : Graph α) : Prop :=
  sorry

/-- The main theorem stating that if δ(G) ≥ 8k and |G| ≤ 16k, then G contains a k-connected subgraph. -/
theorem kConnectedSubgraph {α : Type*} (G : Graph α) (k : ℕ) :
  minDegree G ≥ 8 * k →
  G.V.card ≤ 16 * k →
  ∃ H : Graph α, isSubgraph H G ∧ isKConnected H k :=
sorry

end kConnectedSubgraph_l3075_307569


namespace power_zero_simplify_expression_l3075_307565

-- Theorem 1: For any real number x ≠ 0, x^0 = 1
theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Theorem 2: For any real numbers a and b, (-2a^2)^2 * 3ab^2 = 12a^5b^2
theorem simplify_expression (a b : ℝ) : (-2*a^2)^2 * 3*a*b^2 = 12*a^5*b^2 := by sorry

end power_zero_simplify_expression_l3075_307565


namespace prop_2_prop_3_prop_4_l3075_307558

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the existence of two distinct lines and two distinct planes
variable (a b : Line)
variable (α β : Plane)
variable (h_distinct_lines : a ≠ b)
variable (h_distinct_planes : α ≠ β)

-- Proposition ②
theorem prop_2 : 
  (perpendicular a α ∧ perpendicular a β) → parallel_planes α β :=
sorry

-- Proposition ③
theorem prop_3 :
  perpendicular_planes α β → 
  ∃ γ : Plane, perpendicular_planes γ α ∧ perpendicular_planes γ β :=
sorry

-- Proposition ④
theorem prop_4 :
  perpendicular_planes α β → 
  ∃ l : Line, perpendicular l α ∧ parallel l β :=
sorry

end prop_2_prop_3_prop_4_l3075_307558


namespace coefficient_of_x_squared_l3075_307595

def polynomial (x : ℝ) : ℝ := 5*(x^2 - 2*x^3) + 3*(2*x - 3*x^2 + x^4) - (6*x^3 - 2*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), ∀ x, polynomial x = a*x^4 + b*x^3 + (-2)*x^2 + c*x + d :=
by sorry

end coefficient_of_x_squared_l3075_307595


namespace abcdef_hex_bit_length_l3075_307553

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- This case should not occur for valid hex digits

/-- Converts a hexadecimal number (as a string) to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- Calculates the number of bits needed to represent a natural number -/
def bit_length (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem abcdef_hex_bit_length :
  bit_length (hex_to_decimal "ABCDEF") = 24 := by
  sorry

#eval bit_length (hex_to_decimal "ABCDEF")

end abcdef_hex_bit_length_l3075_307553


namespace parallelogram_formation_condition_l3075_307552

/-- Represents a point in a one-dimensional space -/
structure Point :=
  (x : ℝ)

/-- Represents a line segment between two points -/
def LineSegment (P Q : Point) : ℝ :=
  |Q.x - P.x|

/-- Condition for forming a parallelogram when rotating line segments -/
def ParallelogramCondition (P Q R S T : Point) (a b c : ℝ) : Prop :=
  P.x < Q.x ∧ Q.x < R.x ∧ R.x < S.x ∧ S.x < T.x ∧
  LineSegment P Q = a ∧
  LineSegment P R = b ∧
  LineSegment P T = c ∧
  b = c - a

theorem parallelogram_formation_condition 
  (P Q R S T : Point) (a b c : ℝ) :
  ParallelogramCondition P Q R S T a b c →
  ∃ (P' T' : Point),
    LineSegment Q P' = a ∧
    LineSegment R T' = c - b ∧
    LineSegment P' T' = b - a ∧
    LineSegment S P' = LineSegment S T' :=
sorry

end parallelogram_formation_condition_l3075_307552


namespace quadratic_function_property_l3075_307591

/-- Given a quadratic function f(x) = ax² + bx + c with specific properties, 
    prove that a > 0 and 4a + b = 0 -/
theorem quadratic_function_property (a b c : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  (f 0 = f 4) ∧ (f 0 > f 1) → a > 0 ∧ 4 * a + b = 0 := by
  sorry

end quadratic_function_property_l3075_307591


namespace truncated_cone_base_area_l3075_307533

-- Define the radii of the three cones
def r₁ : ℝ := 10
def r₂ : ℝ := 15
def r₃ : ℝ := 15

-- Define the radius of the smaller base of the truncated cone
def r : ℝ := 2

-- Theorem statement
theorem truncated_cone_base_area 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 15)
  (h₃ : r₃ = 15)
  (h₄ : (r₁ + r)^2 = r₁^2 + (r₂ + r - r₁)^2)
  (h₅ : (r₂ + r)^2 = r₂^2 + (r₁ + r₂ - r)^2)
  (h₆ : (r₃ + r)^2 = r₃^2 + (r₁ + r₃ - r)^2) :
  π * r^2 = 4 * π := by sorry

end truncated_cone_base_area_l3075_307533


namespace ab_equals_one_l3075_307559

theorem ab_equals_one (a b : ℝ) (ha : a = Real.sqrt 3 / 3) (hb : b = Real.sqrt 3) : a * b = 1 := by
  sorry

end ab_equals_one_l3075_307559


namespace randy_blocks_total_l3075_307521

theorem randy_blocks_total (house_blocks tower_blocks : ℕ) 
  (house_tower_diff : ℕ) (total_blocks : ℕ) : 
  house_blocks = 20 →
  tower_blocks = 50 →
  tower_blocks = house_blocks + house_tower_diff →
  house_tower_diff = 30 →
  total_blocks = house_blocks + tower_blocks →
  total_blocks = 70 := by
sorry

end randy_blocks_total_l3075_307521


namespace small_bottles_sold_percentage_l3075_307562

/-- Given the initial number of small and big bottles, the percentage of big bottles sold,
    and the total number of bottles remaining, prove that 15% of small bottles were sold. -/
theorem small_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (big_bottles_sold_percent : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 5000)
  (h2 : initial_big = 12000)
  (h3 : big_bottles_sold_percent = 18/100)
  (h4 : total_remaining = 14090)
  (h5 : total_remaining = initial_small + initial_big -
        (initial_small * small_bottles_sold_percent / 100 +
         initial_big * big_bottles_sold_percent).floor) :
  small_bottles_sold_percent = 15/100 :=
sorry

end small_bottles_sold_percentage_l3075_307562


namespace trajectory_equation_l3075_307508

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation
def vector_equation (C : ℝ × ℝ) (s t : ℝ) : Prop :=
  C = (s * A.1 + t * B.1, s * A.2 + t * B.2)

-- Define the constraint
def constraint (s t : ℝ) : Prop := s + t = 1

-- Theorem statement
theorem trajectory_equation :
  ∀ (C : ℝ × ℝ) (s t : ℝ),
  vector_equation C s t → constraint s t →
  C.1 - C.2 - 1 = 0 :=
sorry

end trajectory_equation_l3075_307508


namespace students_in_diligence_l3075_307588

/-- Represents the number of students in a section before transfers -/
structure SectionCount where
  diligence : ℕ
  industry : ℕ
  progress : ℕ

/-- Represents the transfers between sections -/
structure Transfers where
  industry_to_diligence : ℕ
  progress_to_industry : ℕ

/-- The problem statement -/
theorem students_in_diligence 
  (initial : SectionCount) 
  (transfers : Transfers) 
  (total_students : ℕ) :
  (initial.diligence + initial.industry + initial.progress = total_students) →
  (initial.diligence + transfers.industry_to_diligence = 
   initial.industry - transfers.industry_to_diligence + transfers.progress_to_industry) →
  (initial.diligence + transfers.industry_to_diligence = 
   initial.progress - transfers.progress_to_industry) →
  (transfers.industry_to_diligence = 2) →
  (transfers.progress_to_industry = 3) →
  (total_students = 75) →
  initial.diligence = 23 := by
  sorry

end students_in_diligence_l3075_307588


namespace count_odd_numbers_between_215_and_500_l3075_307545

theorem count_odd_numbers_between_215_and_500 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n > 215 ∧ n < 500) (Finset.range 500)).card = 142 :=
by sorry

end count_odd_numbers_between_215_and_500_l3075_307545


namespace tan_405_degrees_l3075_307500

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end tan_405_degrees_l3075_307500


namespace sum_even_positive_lt_100_eq_2450_l3075_307532

/-- The sum of all even, positive integers less than 100 -/
def sum_even_positive_lt_100 : ℕ :=
  (Finset.range 50).sum (fun i => 2 * i)

/-- Theorem stating that the sum of all even, positive integers less than 100 is 2450 -/
theorem sum_even_positive_lt_100_eq_2450 : sum_even_positive_lt_100 = 2450 := by
  sorry

end sum_even_positive_lt_100_eq_2450_l3075_307532


namespace garden_width_l3075_307593

/-- A rectangular garden with given length and area has a specific width. -/
theorem garden_width (length area : ℝ) (h1 : length = 12) (h2 : area = 60) :
  area / length = 5 := by
  sorry

end garden_width_l3075_307593


namespace square_diff_sqrt_l3075_307583

theorem square_diff_sqrt : (Real.sqrt 81 - Real.sqrt 144)^2 = 9 := by
  sorry

end square_diff_sqrt_l3075_307583


namespace club_enrollment_l3075_307539

theorem club_enrollment (total : ℕ) (math : ℕ) (chem : ℕ) (both : ℕ) :
  total = 150 →
  math = 90 →
  chem = 70 →
  both = 20 →
  total - (math + chem - both) = 10 :=
by sorry

end club_enrollment_l3075_307539


namespace g_over_log16_2_eq_4n_l3075_307599

/-- Sum of squares of elements in nth row of Pascal's triangle -/
def pascal_row_sum_squares (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Base-16 logarithm function -/
noncomputable def log16 (x : ℝ) : ℝ := Real.log x / Real.log 16

/-- Function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ := log16 (pascal_row_sum_squares n)

/-- Theorem stating the relationship between g(n) and n -/
theorem g_over_log16_2_eq_4n (n : ℕ) : g n / log16 2 = 4 * n := by sorry

end g_over_log16_2_eq_4n_l3075_307599


namespace sum_greater_than_three_l3075_307531

theorem sum_greater_than_three (a b c : ℝ) 
  (h1 : a * b + b * c + c * a > a + b + c) 
  (h2 : a + b + c > 0) : 
  a + b + c > 3 := by
  sorry

end sum_greater_than_three_l3075_307531


namespace present_value_log_formula_l3075_307528

theorem present_value_log_formula (c s P k n : ℝ) (h_pos : 0 < 1 + k) :
  P = c * s / (1 + k) ^ n →
  n = (Real.log (c * s / P)) / (Real.log (1 + k)) :=
by sorry

end present_value_log_formula_l3075_307528


namespace curve_equation_l3075_307582

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t + 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation (t : ℝ) :
  let a : ℝ := 1/9
  let b : ℝ := -4/15
  let c : ℝ := 19/375
  a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 := by
  sorry

end curve_equation_l3075_307582


namespace floral_shop_sale_total_l3075_307548

/-- Represents the total number of bouquets sold during a three-day sale at a floral shop. -/
def total_bouquets_sold (monday_sales : ℕ) : ℕ :=
  let tuesday_sales := 3 * monday_sales
  let wednesday_sales := tuesday_sales / 3
  monday_sales + tuesday_sales + wednesday_sales

/-- Theorem stating that given the conditions of the sale, the total number of bouquets sold is 60. -/
theorem floral_shop_sale_total (h : total_bouquets_sold 12 = 60) : 
  total_bouquets_sold 12 = 60 := by
  sorry

end floral_shop_sale_total_l3075_307548


namespace parallelogram_base_length_l3075_307570

/-- A parallelogram with an area of 200 sq m and an altitude that is twice the corresponding base has a base length of 10 meters. -/
theorem parallelogram_base_length (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 200 →
  altitude = 2 * base →
  area = base * altitude →
  base = 10 := by
  sorry

end parallelogram_base_length_l3075_307570


namespace michelle_initial_crayons_l3075_307535

theorem michelle_initial_crayons :
  ∀ (michelle_initial janet : ℕ),
    janet = 2 →
    michelle_initial + janet = 4 →
    michelle_initial = 2 := by
  sorry

end michelle_initial_crayons_l3075_307535


namespace problem_solution_l3075_307534

theorem problem_solution (x y : ℝ) : 
  (2*x - 3*y + 5)^2 + |x + y - 2| = 0 → 3*x - 2*y = -3 := by
sorry

end problem_solution_l3075_307534


namespace geometric_sum_of_powers_of_five_l3075_307515

theorem geometric_sum_of_powers_of_five : 
  (Finset.range 6).sum (fun i => 5^(i+1)) = 19530 := by
  sorry

end geometric_sum_of_powers_of_five_l3075_307515


namespace arctangent_sum_equals_pi_over_four_l3075_307537

theorem arctangent_sum_equals_pi_over_four :
  ∃ (n : ℕ+), (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/n) = π/4) ∧ n = 113 := by
  sorry

end arctangent_sum_equals_pi_over_four_l3075_307537


namespace garden_perimeter_l3075_307592

/-- The perimeter of a rectangular garden with the same area as a given playground -/
theorem garden_perimeter (garden_width playground_length playground_width : ℝ) :
  garden_width = 4 ∧
  playground_length = 16 ∧
  playground_width = 12 →
  (garden_width * (playground_length * playground_width / garden_width) + garden_width) * 2 = 104 := by
  sorry

end garden_perimeter_l3075_307592


namespace simultaneous_ringing_l3075_307538

/-- The least common multiple of the bell ringing periods -/
def bell_lcm : ℕ := sorry

/-- The time difference in minutes between the first and next simultaneous ringing -/
def time_difference : ℕ := sorry

theorem simultaneous_ringing :
  bell_lcm = lcm 18 (lcm 24 (lcm 30 36)) ∧
  time_difference = bell_lcm ∧
  time_difference = 360 := by sorry

end simultaneous_ringing_l3075_307538


namespace combined_tennis_percentage_l3075_307554

def north_students : ℕ := 1800
def south_students : ℕ := 2700
def north_tennis_percent : ℚ := 25 / 100
def south_tennis_percent : ℚ := 35 / 100

theorem combined_tennis_percentage :
  let total_students := north_students + south_students
  let north_tennis := (north_students : ℚ) * north_tennis_percent
  let south_tennis := (south_students : ℚ) * south_tennis_percent
  let total_tennis := north_tennis + south_tennis
  (total_tennis / total_students) * 100 = 31 := by
sorry

end combined_tennis_percentage_l3075_307554


namespace sum_of_positive_numbers_l3075_307517

theorem sum_of_positive_numbers (a b : ℝ) : 
  a > 0 → b > 0 → (a + b) / (a^2 + a*b + b^2) = 4/49 → a + b = 16 := by
  sorry

end sum_of_positive_numbers_l3075_307517


namespace at_least_one_divisible_by_three_l3075_307550

theorem at_least_one_divisible_by_three (a b : ℤ) : 
  (3 ∣ a) ∨ (3 ∣ b) ∨ (3 ∣ (a + b)) ∨ (3 ∣ (a - b)) := by
  sorry

end at_least_one_divisible_by_three_l3075_307550


namespace fathers_sons_age_sum_l3075_307575

theorem fathers_sons_age_sum (father_age son_age : ℕ) : 
  father_age = 40 → 
  son_age = 15 → 
  2 * son_age + father_age = 70 → 
  2 * father_age + son_age = 95 :=
by sorry

end fathers_sons_age_sum_l3075_307575


namespace remainder_s_15_plus_1_l3075_307571

theorem remainder_s_15_plus_1 (s : ℤ) : (s^15 + 1) % (s - 1) = 2 := by
  sorry

end remainder_s_15_plus_1_l3075_307571


namespace walking_time_equals_early_arrival_l3075_307566

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure WalkingScenario where
  D : ℝ  -- Total distance from station to home
  Vw : ℝ  -- Wife's driving speed
  Vm : ℝ  -- Man's walking speed
  T : ℝ  -- Usual time for wife to drive from station to home
  t : ℝ  -- Time man spent walking before being picked up
  early_arrival : ℝ  -- Time they arrived home earlier than usual

/-- The time the man spent walking is equal to the time they arrived home earlier --/
theorem walking_time_equals_early_arrival (scenario : WalkingScenario) 
  (h1 : scenario.D = scenario.Vw * scenario.T)
  (h2 : scenario.D - scenario.Vm * scenario.t = scenario.Vw * (scenario.T - scenario.t))
  (h3 : scenario.early_arrival = scenario.t) :
  scenario.t = scenario.early_arrival :=
by
  sorry

#check walking_time_equals_early_arrival

end walking_time_equals_early_arrival_l3075_307566


namespace third_number_proof_l3075_307530

theorem third_number_proof (A B C : ℕ+) : 
  A = 24 → B = 36 → Nat.gcd A (Nat.gcd B C) = 32 → Nat.lcm A (Nat.lcm B C) = 1248 → C = 32 := by
  sorry

end third_number_proof_l3075_307530


namespace lauryn_earnings_l3075_307507

theorem lauryn_earnings (x : ℝ) : 
  x + 0.7 * x = 3400 → x = 2000 := by
  sorry

end lauryn_earnings_l3075_307507


namespace stratified_sampling_female_athletes_l3075_307580

theorem stratified_sampling_female_athletes 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (female_athletes : ℕ) 
  (h1 : total_population = 224) 
  (h2 : sample_size = 32) 
  (h3 : female_athletes = 84) : 
  ↑sample_size * female_athletes / total_population = 12 :=
by
  sorry

end stratified_sampling_female_athletes_l3075_307580


namespace extreme_points_condition_l3075_307597

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a*x + Real.log x

theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ x > 0, f a x ≥ f a x₁ ∨ f a x ≥ f a x₂) ∧
   |f a x₁ - f a x₂| ≥ 3/4 - Real.log 2) →
  a ≥ 3 * Real.sqrt 2 / 2 := by
sorry

end extreme_points_condition_l3075_307597


namespace fermat_number_prime_count_l3075_307503

/-- Fermat number defined as F_n = 2^(2^n) + 1 -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- There are at least n+1 distinct prime numbers less than or equal to F_n -/
theorem fermat_number_prime_count (n : ℕ) :
  ∃ (S : Finset ℕ), S.card = n + 1 ∧ (∀ p ∈ S, Nat.Prime p ∧ p ≤ fermat_number n) :=
sorry

end fermat_number_prime_count_l3075_307503


namespace polynomial_product_sum_l3075_307543

theorem polynomial_product_sum (g h : ℚ) : 
  (∀ d : ℚ, (5 * d^2 - 2 * d + g) * (4 * d^2 + h * d - 6) = 
             20 * d^4 - 18 * d^3 + 7 * d^2 + 10 * d - 18) →
  g + h = 7/3 := by
sorry

end polynomial_product_sum_l3075_307543


namespace triangle_problem_l3075_307568

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
    (h1 : Real.cos t.B * (Real.sqrt 3 * t.a - t.b * Real.sin t.C) - t.b * Real.sin t.B * Real.cos t.C = 0)
    (h2 : t.c = 2 * t.a)
    (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
    t.B = π / 3 ∧ t.a + t.b + t.c = 3 * Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end triangle_problem_l3075_307568


namespace p_recurrence_l3075_307544

/-- Probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ :=
  sorry

/-- The recurrence relation for p(n, k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
sorry

end p_recurrence_l3075_307544


namespace range_of_f_l3075_307584

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 2*x)

theorem range_of_f :
  Set.range f = Set.Ioo (1/2) (Real.pi) :=
sorry

end range_of_f_l3075_307584


namespace subtract_negative_l3075_307567

theorem subtract_negative : -3 - 1 = -4 := by
  sorry

end subtract_negative_l3075_307567


namespace complex_number_in_third_quadrant_l3075_307587

theorem complex_number_in_third_quadrant (z : ℂ) (h : (z + Complex.I) * Complex.I = 1 + z) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_in_third_quadrant_l3075_307587


namespace root_equation_problem_l3075_307577

theorem root_equation_problem (c d : ℝ) : 
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ 
    (∀ x : ℝ, (x + c) * (x + d) * (x + 10) / (x + 2)^2 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)) ∧
  (∃! (r : ℝ), ∀ x : ℝ, (x + 2*c) * (x + 4) * (x + 8) / ((x + d) * (x + 10)) = 0 ↔ x = r) →
  200 * c + d = 392 :=
by sorry

end root_equation_problem_l3075_307577


namespace larger_integer_proof_l3075_307585

theorem larger_integer_proof (A B : ℤ) (h1 : A + B = 2010) (h2 : Nat.lcm A.natAbs B.natAbs = 14807) : 
  max A B = 1139 := by
sorry

end larger_integer_proof_l3075_307585
