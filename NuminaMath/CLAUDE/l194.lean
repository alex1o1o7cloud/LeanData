import Mathlib

namespace NUMINAMATH_CALUDE_charles_pictures_l194_19465

theorem charles_pictures (initial_papers : ℕ) (today_pictures : ℕ) (yesterday_before_work : ℕ) (papers_left : ℕ) :
  initial_papers = 20 →
  today_pictures = 6 →
  yesterday_before_work = 6 →
  papers_left = 2 →
  initial_papers - today_pictures - yesterday_before_work - papers_left = 6 := by
  sorry

end NUMINAMATH_CALUDE_charles_pictures_l194_19465


namespace NUMINAMATH_CALUDE_three_x_plus_five_y_equals_six_l194_19474

theorem three_x_plus_five_y_equals_six 
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 
  3 * x + 5 * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_x_plus_five_y_equals_six_l194_19474


namespace NUMINAMATH_CALUDE_call_center_efficiency_l194_19452

/-- Represents the efficiency and size of call center teams relative to Team B -/
structure CallCenterTeams where
  team_a_efficiency : ℚ  -- Efficiency of Team A relative to Team B
  team_c_efficiency : ℚ  -- Efficiency of Team C relative to Team B
  team_a_size : ℚ        -- Size of Team A relative to Team B
  team_c_size : ℚ        -- Size of Team C relative to Team B

/-- Calculates the fraction of total calls processed by all three teams combined -/
def fraction_of_total_calls (teams : CallCenterTeams) : ℚ :=
  sorry

/-- Theorem stating that the fraction of total calls processed is 19/32 -/
theorem call_center_efficiency (teams : CallCenterTeams) 
  (h1 : teams.team_a_efficiency = 1/5)
  (h2 : teams.team_c_efficiency = 7/8)
  (h3 : teams.team_a_size = 5/8)
  (h4 : teams.team_c_size = 3/4) :
  fraction_of_total_calls teams = 19/32 :=
  sorry

end NUMINAMATH_CALUDE_call_center_efficiency_l194_19452


namespace NUMINAMATH_CALUDE_a_45_value_l194_19464

def a : ℕ → ℤ
  | 0 => 11
  | 1 => 11
  | n + 2 => sorry  -- This will be defined using the recurrence relation

-- Define the recurrence relation
axiom a_rec : ∀ (m n : ℕ), a (m + n) = (1/2) * (a (2*m) + a (2*n)) - (m - n)^2

theorem a_45_value : a 45 = 1991 := by
  sorry

end NUMINAMATH_CALUDE_a_45_value_l194_19464


namespace NUMINAMATH_CALUDE_factorization_equality_l194_19436

theorem factorization_equality (y : ℝ) : 5*y*(y+2) + 8*(y+2) + 15 = (5*y+8)*(y+2) + 15 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l194_19436


namespace NUMINAMATH_CALUDE_absolute_value_integral_l194_19433

theorem absolute_value_integral : ∫ x in (0:ℝ)..2, |1 - x| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l194_19433


namespace NUMINAMATH_CALUDE_sum_mod_ten_l194_19472

theorem sum_mod_ten : (17145 + 17146 + 17147 + 17148 + 17149) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_ten_l194_19472


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_40_l194_19418

/-- The parabola P defined by y = x^2 + 4 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 4

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The slope-intercept form of a line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- The quadratic equation representing the intersection of the line and parabola -/
def intersection_eq (m : ℝ) : ℝ → ℝ := λ x ↦ x^2 - m * x + (10 * m - 2)

/-- The discriminant of the intersection equation -/
def discriminant (m : ℝ) : ℝ := m^2 - 4 * (10 * m - 2)

theorem sum_of_roots_eq_40 :
  ∃ r s : ℝ, r + s = 40 ∧
    ∀ m : ℝ, discriminant m < 0 ↔ r < m ∧ m < s :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_40_l194_19418


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l194_19419

theorem arithmetic_sqrt_of_nine (x : ℝ) : x ≥ 0 ∧ x ^ 2 = 9 → x = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l194_19419


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_neg_five_thirds_l194_19414

theorem trigonometric_expression_equals_neg_five_thirds :
  (Real.tan (30 * π / 180))^2 - (Real.cos (30 * π / 180))^2
  / ((Real.tan (30 * π / 180))^2 * (Real.cos (30 * π / 180))^2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_neg_five_thirds_l194_19414


namespace NUMINAMATH_CALUDE_tigers_wins_l194_19437

def total_games : ℕ := 56
def losses : ℕ := 12

theorem tigers_wins : 
  let ties := losses / 2
  let wins := total_games - (losses + ties)
  wins = 38 := by sorry

end NUMINAMATH_CALUDE_tigers_wins_l194_19437


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_each_l194_19458

/-- The probability of having a boy or a girl -/
def p_boy_or_girl : ℚ := 1/2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having all boys or all girls -/
def p_all_same : ℚ := 2 * (p_boy_or_girl ^ num_children)

/-- The probability of having at least one boy and one girl -/
def p_at_least_one_of_each : ℚ := 1 - p_all_same

theorem prob_at_least_one_of_each :
  p_at_least_one_of_each = 7/8 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_each_l194_19458


namespace NUMINAMATH_CALUDE_milkman_profit_is_90_l194_19490

/-- Calculates the profit of a milkman selling a milk-water mixture --/
def milkman_profit (total_milk : ℕ) (milk_in_mixture : ℕ) (water_in_mixture : ℕ) (cost_per_liter : ℕ) : ℕ :=
  let total_mixture := milk_in_mixture + water_in_mixture
  let selling_price := total_mixture * cost_per_liter
  let cost_of_milk_used := milk_in_mixture * cost_per_liter
  selling_price - cost_of_milk_used

/-- Proves that the milkman's profit is 90 under given conditions --/
theorem milkman_profit_is_90 :
  milkman_profit 30 20 5 18 = 90 := by
  sorry

#eval milkman_profit 30 20 5 18

end NUMINAMATH_CALUDE_milkman_profit_is_90_l194_19490


namespace NUMINAMATH_CALUDE_divisor_sum_representation_l194_19430

theorem divisor_sum_representation (n : ℕ) :
  ∀ k : ℕ, k ≤ n! → ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x ∣ n!) ∧ 
    S.card ≤ n ∧ 
    k = S.sum id :=
by sorry

end NUMINAMATH_CALUDE_divisor_sum_representation_l194_19430


namespace NUMINAMATH_CALUDE_charity_arrangements_l194_19427

/-- The number of people selected from the class -/
def total_people : ℕ := 6

/-- The maximum number of people that can participate in each activity -/
def max_per_activity : ℕ := 4

/-- The number of charity activities -/
def num_activities : ℕ := 2

/-- The function to calculate the number of different arrangements -/
def num_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ := sorry

theorem charity_arrangements :
  num_arrangements total_people max_per_activity num_activities = 50 := by sorry

end NUMINAMATH_CALUDE_charity_arrangements_l194_19427


namespace NUMINAMATH_CALUDE_shaded_area_of_folded_rectangle_l194_19442

/-- The area of the shaded region formed by folding a rectangular sheet along its diagonal -/
theorem shaded_area_of_folded_rectangle (length width : ℝ) (h_length : length = 12) (h_width : width = 18) :
  let rectangle_area := length * width
  let diagonal := Real.sqrt (length^2 + width^2)
  let triangle_area := (1 / 2) * diagonal * diagonal * (2 / 3)
  rectangle_area - triangle_area = 138 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_folded_rectangle_l194_19442


namespace NUMINAMATH_CALUDE_ellipse_equation_l194_19455

/-- The equation √(x² + (y-3)²) + √(x² + (y+3)²) = 10 represents an ellipse. -/
theorem ellipse_equation (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt (x^2 + (y + 3)^2) = 10) ↔
  (y^2 / 25 + x^2 / 16 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l194_19455


namespace NUMINAMATH_CALUDE_real_condition_implies_a_equals_one_l194_19403

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that a complex number is real
def is_real (z : ℂ) : Prop := z.im = 0

-- Theorem statement
theorem real_condition_implies_a_equals_one (a : ℝ) :
  is_real ((1 + i) * (1 - a * i)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_condition_implies_a_equals_one_l194_19403


namespace NUMINAMATH_CALUDE_maggies_earnings_proof_l194_19484

/-- Calculates Maggie's earnings from selling magazine subscriptions -/
def maggies_earnings (price_per_subscription : ℕ) 
                     (parents_subscriptions : ℕ)
                     (grandfather_subscriptions : ℕ)
                     (neighbor1_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := parents_subscriptions + 
                             grandfather_subscriptions + 
                             neighbor1_subscriptions + 
                             (2 * neighbor1_subscriptions)
  price_per_subscription * total_subscriptions

theorem maggies_earnings_proof : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

#eval maggies_earnings 5 4 1 2

end NUMINAMATH_CALUDE_maggies_earnings_proof_l194_19484


namespace NUMINAMATH_CALUDE_median_to_longest_side_l194_19400

/-- Given a triangle with side lengths 10, 24, and 26, the length of the median to the longest side is 13. -/
theorem median_to_longest_side (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  let m := (1/2) * Real.sqrt (2 * a^2 + 2 * b^2 - c^2)
  m = 13 := by sorry

end NUMINAMATH_CALUDE_median_to_longest_side_l194_19400


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l194_19469

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l194_19469


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l194_19486

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 10, then x = -25/2 when y = -4 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * 10 = k) :
  -4 * x = k → x = -25/2 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l194_19486


namespace NUMINAMATH_CALUDE_polynomial_factor_l194_19471

theorem polynomial_factor (a : ℚ) : 
  (∀ x : ℚ, (x + 5) ∣ (a * x^4 + 12 * x^2 - 5 * a * x + 42)) → 
  a = -57/100 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_l194_19471


namespace NUMINAMATH_CALUDE_sum_first_49_primes_l194_19448

def first_n_primes (n : ℕ) : List ℕ := sorry

theorem sum_first_49_primes :
  (first_n_primes 49).sum = 10787 := by sorry

end NUMINAMATH_CALUDE_sum_first_49_primes_l194_19448


namespace NUMINAMATH_CALUDE_cone_volume_l194_19481

/-- Given a cone with slant height 15 cm and height 9 cm, its volume is 432π cubic centimeters. -/
theorem cone_volume (s h r : ℝ) (hs : s = 15) (hh : h = 9) 
  (hr : r^2 = s^2 - h^2) : (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l194_19481


namespace NUMINAMATH_CALUDE_road_length_proof_l194_19424

/-- The length of a road given round trip conditions -/
theorem road_length_proof (total_time : ℝ) (walking_speed : ℝ) (bus_speed : ℝ)
  (h1 : total_time = 2)
  (h2 : walking_speed = 5)
  (h3 : bus_speed = 20) :
  ∃ (road_length : ℝ), road_length / walking_speed + road_length / bus_speed = total_time ∧ road_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_road_length_proof_l194_19424


namespace NUMINAMATH_CALUDE_trigonometric_identity_l194_19438

theorem trigonometric_identity (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l194_19438


namespace NUMINAMATH_CALUDE_integer_solutions_for_mn_squared_equation_l194_19496

theorem integer_solutions_for_mn_squared_equation : 
  ∀ (m n : ℤ), m * n^2 = 2009 * (n + 1) ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_for_mn_squared_equation_l194_19496


namespace NUMINAMATH_CALUDE_tangent_line_equation_l194_19473

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

/-- The point of tangency -/
def p : ℝ × ℝ := (-1, -3)

/-- Theorem: The equation of the tangent line to the curve y = x³ + 3x² - 5
    at the point (-1, -3) is 3x + y + 6 = 0 -/
theorem tangent_line_equation :
  ∀ (x y : ℝ), y = f' p.1 * (x - p.1) + p.2 ↔ 3*x + y + 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l194_19473


namespace NUMINAMATH_CALUDE_other_endpoint_coordinates_l194_19489

/-- Given a line segment with midpoint (3, 7) and one endpoint at (0, 11),
    prove that the other endpoint is at (6, 3). -/
theorem other_endpoint_coordinates :
  ∀ (x y : ℝ),
  (3 = (0 + x) / 2) →
  (7 = (11 + y) / 2) →
  (x = 6 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinates_l194_19489


namespace NUMINAMATH_CALUDE_spider_web_production_l194_19413

def spider_webs (num_spiders : ℕ) (num_webs : ℕ) (days : ℕ) : Prop :=
  num_spiders = num_webs ∧ days > 0

theorem spider_web_production 
  (h1 : spider_webs 7 7 (7 : ℕ)) 
  (h2 : spider_webs 1 1 7) : 
  ∀ s, s ≤ 7 → spider_webs 1 1 7 :=
sorry

end NUMINAMATH_CALUDE_spider_web_production_l194_19413


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l194_19468

theorem six_digit_divisibility (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  let n := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c
  n % 7 = 0 ∧ n % 11 = 0 ∧ n % 13 = 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l194_19468


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l194_19423

def range_start : ℕ := 1
def range_end : ℕ := 60
def multiples_of_four : ℕ := 15

theorem probability_at_least_one_multiple_of_four :
  let total_numbers := range_end - range_start + 1
  let non_multiples := total_numbers - multiples_of_four
  let prob_neither_multiple := (non_multiples / total_numbers) ^ 2
  1 - prob_neither_multiple = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l194_19423


namespace NUMINAMATH_CALUDE_fruit_salad_weight_l194_19498

theorem fruit_salad_weight (melon_weight berries_weight : ℝ) 
  (h1 : melon_weight = 0.25)
  (h2 : berries_weight = 0.38) : 
  melon_weight + berries_weight = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_weight_l194_19498


namespace NUMINAMATH_CALUDE_shopkeeper_pricing_l194_19454

/-- Proves that the original selling price is 800 given the conditions of the problem -/
theorem shopkeeper_pricing (cost_price : ℝ) : 
  (1.25 * cost_price = 800) ∧ (0.8 * cost_price = 512) := by
  sorry

#check shopkeeper_pricing

end NUMINAMATH_CALUDE_shopkeeper_pricing_l194_19454


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l194_19410

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l194_19410


namespace NUMINAMATH_CALUDE_apples_in_basket_l194_19440

theorem apples_in_basket (initial_apples : ℕ) : 
  let ricki_removed : ℕ := 14
  let samson_removed : ℕ := 2 * ricki_removed
  let apples_left : ℕ := 32
  initial_apples = apples_left + ricki_removed + samson_removed :=
by sorry

end NUMINAMATH_CALUDE_apples_in_basket_l194_19440


namespace NUMINAMATH_CALUDE_inequality_proof_l194_19421

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄)
  (h4 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l194_19421


namespace NUMINAMATH_CALUDE_apples_sold_per_day_l194_19494

/-- Calculates the average number of apples sold per day given the total number of boxes,
    days, and apples per box. -/
def average_apples_per_day (boxes : ℕ) (days : ℕ) (apples_per_box : ℕ) : ℚ :=
  (boxes * apples_per_box : ℚ) / days

/-- Theorem stating that given 12 boxes of apples sold in 4 days,
    with 25 apples per box, the average number of apples sold per day is 75. -/
theorem apples_sold_per_day :
  average_apples_per_day 12 4 25 = 75 := by
  sorry

end NUMINAMATH_CALUDE_apples_sold_per_day_l194_19494


namespace NUMINAMATH_CALUDE_marbles_ratio_l194_19405

def marbles_problem (initial_marbles : ℕ) (current_marbles : ℕ) (brother_marbles : ℕ) : Prop :=
  let savanna_marbles := 3 * current_marbles
  let sister_marbles := initial_marbles - current_marbles - brother_marbles - savanna_marbles
  (sister_marbles : ℚ) / brother_marbles = 2

theorem marbles_ratio :
  marbles_problem 300 30 60 := by
  sorry

end NUMINAMATH_CALUDE_marbles_ratio_l194_19405


namespace NUMINAMATH_CALUDE_sphere_prism_area_difference_l194_19467

theorem sphere_prism_area_difference :
  let r : ℝ := 2  -- radius of the sphere
  let a : ℝ := 2  -- base edge length of the prism
  let sphere_surface_area : ℝ := 4 * π * r^2
  let max_prism_lateral_area : ℝ := 16 * Real.sqrt 2
  sphere_surface_area - max_prism_lateral_area = 16 * (π - Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_sphere_prism_area_difference_l194_19467


namespace NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l194_19470

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l194_19470


namespace NUMINAMATH_CALUDE_min_value_of_expression_l194_19431

theorem min_value_of_expression (x y : ℝ) :
  2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 ≥ -2 ∧
  ∃ (a b : ℝ), 2 * a^2 + 2 * a * b + b^2 - 2 * a - 1 = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l194_19431


namespace NUMINAMATH_CALUDE_distribute_4_3_l194_19497

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 36 ways to distribute 4 distinct objects
    into 3 distinct containers, with each container having at least one object. -/
theorem distribute_4_3 : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_4_3_l194_19497


namespace NUMINAMATH_CALUDE_oil_price_reduction_l194_19420

/-- Calculates the percentage reduction in oil price given the conditions --/
theorem oil_price_reduction (total_cost : ℝ) (additional_kg : ℝ) (reduced_price : ℝ) : 
  total_cost = 1100 ∧ 
  additional_kg = 5 ∧ 
  reduced_price = 55 →
  (((total_cost / (total_cost / reduced_price - additional_kg)) - reduced_price) / 
   (total_cost / (total_cost / reduced_price - additional_kg))) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_oil_price_reduction_l194_19420


namespace NUMINAMATH_CALUDE_triangle_abc_area_l194_19432

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- The line to which the circles are tangent -/
def line_m : Set Point := sorry

theorem triangle_abc_area :
  let circle_a : Circle := { center := { x := -5, y := 2 }, radius := 2 }
  let circle_b : Circle := { center := { x := 0, y := 3 }, radius := 3 }
  let circle_c : Circle := { center := { x := 7, y := 4 }, radius := 4 }
  let point_a' : Point := sorry
  let point_b' : Point := sorry
  let point_c' : Point := sorry

  -- Circles are tangent to line m
  (point_a' ∈ line_m) ∧
  (point_b' ∈ line_m) ∧
  (point_c' ∈ line_m) →

  -- Circle B is externally tangent to circles A and C
  (circle_b.center.x - circle_a.center.x)^2 + (circle_b.center.y - circle_a.center.y)^2 = (circle_b.radius + circle_a.radius)^2 ∧
  (circle_b.center.x - circle_c.center.x)^2 + (circle_b.center.y - circle_c.center.y)^2 = (circle_b.radius + circle_c.radius)^2 →

  -- B' is between A' and C' on line m
  (point_b'.x > point_a'.x ∧ point_b'.x < point_c'.x) →

  -- Centers A and C are aligned horizontally
  circle_a.center.y = circle_c.center.y →

  -- The area of triangle ABC is 6
  abs ((circle_a.center.x * (circle_b.center.y - circle_c.center.y) +
        circle_b.center.x * (circle_c.center.y - circle_a.center.y) +
        circle_c.center.x * (circle_a.center.y - circle_b.center.y)) / 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l194_19432


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l194_19404

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) is on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line is tangent to a circle -/
def Line.isTangentTo (l : Line) (c : Circle) : Prop :=
  (c.h - l.a * (c.h * l.a + c.k * l.b - l.c) / (l.a^2 + l.b^2))^2 +
  (c.k - l.b * (c.h * l.a + c.k * l.b - l.c) / (l.a^2 + l.b^2))^2 = c.r^2

theorem tangent_lines_to_circle (c : Circle) :
  c.h = 1 ∧ c.k = 0 ∧ c.r = 2 →
  ∃ (l₁ l₂ : Line),
    (l₁.a = 3 ∧ l₁.b = 4 ∧ l₁.c = -13) ∧
    (l₂.a = 1 ∧ l₂.b = 0 ∧ l₂.c = -3) ∧
    l₁.contains 3 1 ∧
    l₂.contains 3 1 ∧
    l₁.isTangentTo c ∧
    l₂.isTangentTo c ∧
    ∀ (l : Line), l.contains 3 1 ∧ l.isTangentTo c → l = l₁ ∨ l = l₂ :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l194_19404


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l194_19445

def i : ℂ := Complex.I

theorem imaginary_part_of_complex_fraction :
  ((-1 + i) / (2 - i)).im = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l194_19445


namespace NUMINAMATH_CALUDE_five_ruble_coins_l194_19422

theorem five_ruble_coins (total_coins : ℕ) 
  (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ) :
  total_coins = 25 →
  not_two_ruble = 19 →
  not_ten_ruble = 20 →
  not_one_ruble = 16 →
  total_coins - (total_coins - not_two_ruble + total_coins - not_ten_ruble + total_coins - not_one_ruble) = 5 :=
by sorry

end NUMINAMATH_CALUDE_five_ruble_coins_l194_19422


namespace NUMINAMATH_CALUDE_horner_method_f_2_l194_19426

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x
  let v2 := (v1 - 3) * x + 2
  (v2 * x + 1) * x - 3

theorem horner_method_f_2 :
  horner_v3 f 2 = 12 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l194_19426


namespace NUMINAMATH_CALUDE_mobius_trip_time_l194_19446

-- Define the constants from the problem
def distance : ℝ := 143
def speed_with_load : ℝ := 11
def speed_without_load : ℝ := 13
def rest_time_per_stop : ℝ := 0.5
def num_rest_stops : ℕ := 4

-- Define the theorem
theorem mobius_trip_time :
  let time_with_load := distance / speed_with_load
  let time_without_load := distance / speed_without_load
  let total_rest_time := rest_time_per_stop * num_rest_stops
  time_with_load + time_without_load + total_rest_time = 26 := by
sorry


end NUMINAMATH_CALUDE_mobius_trip_time_l194_19446


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l194_19425

theorem consecutive_integers_product_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = 103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l194_19425


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l194_19475

def I : Set ℕ := {x | 0 < x ∧ x ≤ 6}
def P : Set ℕ := {x | 6 % x = 0}
def Q : Set ℕ := {1, 3, 4, 5}

theorem complement_intersection_theorem :
  (I \ P) ∩ Q = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l194_19475


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l194_19488

def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x

theorem monotonic_increasing_range (a : ℝ) :
  Monotone (f a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l194_19488


namespace NUMINAMATH_CALUDE_least_addend_for_divisibility_problem_solution_l194_19492

theorem least_addend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (x : Nat), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : Nat), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem problem_solution :
  ∃ (x : Nat), x = 19 ∧ (1156 + x) % 25 = 0 ∧ ∀ (y : Nat), y < x → (1156 + y) % 25 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addend_for_divisibility_problem_solution_l194_19492


namespace NUMINAMATH_CALUDE_car_speed_problem_l194_19447

theorem car_speed_problem (speed_A : ℝ) (time_A : ℝ) (time_B : ℝ) (distance_ratio : ℝ) :
  speed_A = 70 →
  time_A = 10 →
  time_B = 10 →
  distance_ratio = 2 →
  ∃ speed_B : ℝ, speed_B = 35 ∧ speed_A * time_A = distance_ratio * (speed_B * time_B) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l194_19447


namespace NUMINAMATH_CALUDE_steves_emails_l194_19443

theorem steves_emails (initial_emails : ℕ) : 
  (initial_emails / 2 : ℚ) * (1 - 0.4) = 120 → initial_emails = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_steves_emails_l194_19443


namespace NUMINAMATH_CALUDE_evaluate_expression_l194_19493

theorem evaluate_expression : 4 * (8 - 3) - 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l194_19493


namespace NUMINAMATH_CALUDE_quartic_roots_product_l194_19411

theorem quartic_roots_product (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_product_l194_19411


namespace NUMINAMATH_CALUDE_relative_speed_calculation_l194_19412

/-- Convert meters per second to kilometers per hour -/
def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * 3.6

/-- Convert centimeters per minute to kilometers per hour -/
def cmpm_to_kmh (speed_cmpm : ℝ) : ℝ :=
  speed_cmpm * 0.0006

/-- Calculate the relative speed of two objects moving in opposite directions -/
def relative_speed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  speed1 + speed2

theorem relative_speed_calculation (speed1_ms : ℝ) (speed2_cmpm : ℝ) 
  (h1 : speed1_ms = 12.5)
  (h2 : speed2_cmpm = 1800) :
  relative_speed (ms_to_kmh speed1_ms) (cmpm_to_kmh speed2_cmpm) = 46.08 := by
  sorry

#check relative_speed_calculation

end NUMINAMATH_CALUDE_relative_speed_calculation_l194_19412


namespace NUMINAMATH_CALUDE_equation_proof_l194_19439

theorem equation_proof : 578 - 214 = 364 := by sorry

end NUMINAMATH_CALUDE_equation_proof_l194_19439


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l194_19485

/-- The hyperbola equation -/
def hyperbola_equation (x y a : ℝ) : Prop :=
  y^2 / (2 * a^2) - x^2 / a^2 = 1

/-- The asymptote equation -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±√2x -/
theorem hyperbola_asymptotes (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, hyperbola_equation x y a → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l194_19485


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l194_19463

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4) = 8) ↔ (x = 29 / 5 ∨ x = 13 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l194_19463


namespace NUMINAMATH_CALUDE_glove_selection_count_l194_19459

def num_glove_pairs : ℕ := 6
def num_gloves_to_choose : ℕ := 4
def num_paired_gloves : ℕ := 2

theorem glove_selection_count :
  (num_glove_pairs.choose 1) * ((2 * num_glove_pairs - 2).choose (num_gloves_to_choose - num_paired_gloves) - (num_glove_pairs - 1)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_glove_selection_count_l194_19459


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l194_19480

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 600 → s^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l194_19480


namespace NUMINAMATH_CALUDE_smallest_possible_student_count_l194_19478

/-- The smallest possible number of students in a classroom with the given seating arrangement --/
def smallest_student_count : ℕ := 42

/-- The number of rows in the classroom --/
def num_rows : ℕ := 5

/-- Represents the number of students in each of the first four rows --/
def students_per_row : ℕ := 8

theorem smallest_possible_student_count :
  (num_rows - 1) * students_per_row + (students_per_row + 2) = smallest_student_count ∧
  smallest_student_count > 40 ∧
  ∀ n : ℕ, n < smallest_student_count →
    (num_rows - 1) * (n / num_rows) + (n / num_rows + 2) ≠ n ∨ n ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_student_count_l194_19478


namespace NUMINAMATH_CALUDE_parabola_intersection_l194_19444

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x + 15
  let g (x : ℝ) := 2 * x^2 - 8 * x + 12
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 1 ∧ y = 6) ∨ (x = 3 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l194_19444


namespace NUMINAMATH_CALUDE_valid_street_distances_l194_19407

/-- Represents the position of a house on a street. -/
structure House where
  position : ℝ

/-- The street with four houses. -/
structure Street where
  andrei : House
  borya : House
  vova : House
  gleb : House

/-- The distance between two houses. -/
def distance (h1 h2 : House) : ℝ :=
  |h1.position - h2.position|

/-- A street satisfying the given conditions. -/
def validStreet (s : Street) : Prop :=
  distance s.andrei s.borya = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrei s.gleb = 3 * distance s.borya s.vova

theorem valid_street_distances (s : Street) (h : validStreet s) :
  distance s.andrei s.gleb = 900 ∨ distance s.andrei s.gleb = 1800 :=
sorry

end NUMINAMATH_CALUDE_valid_street_distances_l194_19407


namespace NUMINAMATH_CALUDE_linear_system_solution_l194_19461

theorem linear_system_solution (x y z : ℝ) 
  (eq1 : x + 2*y - z = 8) 
  (eq2 : 2*x - y + z = 18) : 
  8*x + y + z = 70 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l194_19461


namespace NUMINAMATH_CALUDE_total_laundry_loads_l194_19417

/-- The number of families sharing the vacation rental -/
def num_families : ℕ := 7

/-- The number of days of the vacation -/
def num_days : ℕ := 12

/-- The number of adults in each family -/
def adults_per_family : ℕ := 2

/-- The number of children in each family -/
def children_per_family : ℕ := 4

/-- The number of towels used by each adult per day -/
def towels_per_adult : ℕ := 2

/-- The number of towels used by each child per day -/
def towels_per_child : ℕ := 1

/-- The washing machine capacity for the first half of the vacation -/
def machine_capacity_first_half : ℕ := 8

/-- The washing machine capacity for the second half of the vacation -/
def machine_capacity_second_half : ℕ := 6

/-- The number of days in each half of the vacation -/
def days_per_half : ℕ := 6

/-- Theorem stating that the total number of loads of laundry is 98 -/
theorem total_laundry_loads : 
  let towels_per_family := adults_per_family * towels_per_adult + children_per_family * towels_per_child
  let total_towels_per_day := num_families * towels_per_family
  let total_towels := total_towels_per_day * num_days
  let loads_first_half := (total_towels_per_day * days_per_half) / machine_capacity_first_half
  let loads_second_half := (total_towels_per_day * days_per_half) / machine_capacity_second_half
  loads_first_half + loads_second_half = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_loads_l194_19417


namespace NUMINAMATH_CALUDE_archer_weekly_expenditure_l194_19499

def archer_expenditure (shots_per_day : ℕ) (days_per_week : ℕ) (recovery_rate : ℚ) 
  (arrow_cost : ℚ) (team_contribution_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let recovered_arrows := total_shots * recovery_rate
  let net_arrows_used := total_shots - recovered_arrows
  let total_cost := net_arrows_used * arrow_cost
  let archer_cost := total_cost * (1 - team_contribution_rate)
  archer_cost

theorem archer_weekly_expenditure :
  archer_expenditure 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end NUMINAMATH_CALUDE_archer_weekly_expenditure_l194_19499


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_intersection_l194_19406

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Theorem: Given a parabola and a circle with specific properties, 
    prove the equation of the circle -/
theorem circle_equation_from_parabola_intersection 
  (C₁ : Parabola) 
  (C₂ : Circle) 
  (F : ℝ × ℝ) 
  (A B C D : ℝ × ℝ) :
  C₁.equation = fun x y ↦ x^2 = 2*y →  -- Parabola equation
  C₂.center = F →                      -- Circle center at focus
  C₂.equation A.1 A.2 →                -- Circle intersects parabola at A
  C₂.equation B.1 B.2 →                -- Circle intersects parabola at B
  C₂.equation C.1 C.2 →                -- Circle intersects directrix at C
  C₂.equation D.1 D.2 →                -- Circle intersects directrix at D
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2 →  -- ABCD is rectangle
  C₂.equation = fun x y ↦ x^2 + (y - 1/2)^2 = 4 :=  -- Conclusion: Circle equation
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_intersection_l194_19406


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l194_19487

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_condition : a 1 + a 3 + a 8 = 99
  fifth_term : a 5 = 31

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proven -/
theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) :
  ∃ k : ℕ+, ∀ n : ℕ+, S seq n ≤ S seq k ∧ k = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l194_19487


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l194_19483

def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem ratio_of_arithmetic_sequences : 
  let seq1_sum := arithmetic_sequence_sum 4 4 48
  let seq2_sum := arithmetic_sequence_sum 2 3 35
  seq1_sum / seq2_sum = 52 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l194_19483


namespace NUMINAMATH_CALUDE_rectangle_area_18_l194_19428

def rectangle_pairs : Set (Nat × Nat) :=
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}

theorem rectangle_area_18 :
  ∀ (w l : Nat), w > 0 ∧ l > 0 →
  (w * l = 18 ↔ (w, l) ∈ rectangle_pairs) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l194_19428


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l194_19477

/-- The area of the shaded region formed by the overlap of two 30° sectors in a circle with radius 10 is equal to the area of a single 30° sector. -/
theorem overlapping_sectors_area (r : ℝ) (angle : ℝ) : 
  r = 10 → angle = 30 * (π / 180) → 
  let sector_area := (angle / (2 * π)) * π * r^2
  let shaded_area := sector_area
  ∀ ε > 0, |shaded_area - sector_area| < ε :=
sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l194_19477


namespace NUMINAMATH_CALUDE_max_planes_is_six_l194_19450

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A configuration of 6 points in 3D space -/
def Configuration := Fin 6 → Point3D

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Check if a plane contains at least 4 points from the configuration -/
def planeContainsAtLeast4Points (plane : Plane3D) (config : Configuration) : Prop :=
  ∃ (p1 p2 p3 p4 : Fin 6), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    pointOnPlane (config p1) plane ∧ pointOnPlane (config p2) plane ∧
    pointOnPlane (config p3) plane ∧ pointOnPlane (config p4) plane

/-- Check if no line passes through 4 points in the configuration -/
def noLinePasses4Points (config : Configuration) : Prop :=
  ∀ (p1 p2 p3 p4 : Fin 6), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    ¬∃ (a b c : ℝ), ∀ (p : Fin 6), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 →
      a * (config p).x + b * (config p).y + c = (config p).z

/-- The main theorem: The maximum number of planes satisfying the conditions is 6 -/
theorem max_planes_is_six (config : Configuration) 
    (h_no_line : noLinePasses4Points config) : 
    (∃ (planes : Fin 6 → Plane3D), ∀ (i : Fin 6), planeContainsAtLeast4Points (planes i) config) ∧
    (∀ (n : ℕ) (planes : Fin (n + 1) → Plane3D), 
      (∀ (i : Fin (n + 1)), planeContainsAtLeast4Points (planes i) config) → n ≤ 5) :=
  sorry


end NUMINAMATH_CALUDE_max_planes_is_six_l194_19450


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l194_19451

theorem cylinder_radius_problem (r : ℝ) : 
  let h : ℝ := 3
  let volume_decrease_radius : ℝ := 3 * Real.pi * ((r - 4)^2 - r^2)
  let volume_decrease_height : ℝ := Real.pi * r^2 * (h - (h - 4))
  volume_decrease_radius = volume_decrease_height →
  (r = 6 + 2 * Real.sqrt 3 ∨ r = 6 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l194_19451


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l194_19415

/-- Given two vectors in ℝ², prove that if k * a + b is perpendicular to a - 3 * b, then k = 19 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) :
  k = 19 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l194_19415


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l194_19408

theorem root_sum_reciprocal (α β γ : ℂ) : 
  (α^3 - 2*α^2 - α + 2 = 0) → 
  (β^3 - 2*β^2 - β + 2 = 0) → 
  (γ^3 - 2*γ^2 - γ + 2 = 0) → 
  (1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2) = -19 / 14) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l194_19408


namespace NUMINAMATH_CALUDE_purple_book_pages_purple_book_pages_proof_l194_19401

theorem purple_book_pages : ℕ → Prop :=
  fun p =>
    let orange_pages : ℕ := 510
    let purple_books_read : ℕ := 5
    let orange_books_read : ℕ := 4
    let page_difference : ℕ := 890
    orange_books_read * orange_pages - purple_books_read * p = page_difference →
    p = 230

-- The proof goes here
theorem purple_book_pages_proof : purple_book_pages 230 := by
  sorry

end NUMINAMATH_CALUDE_purple_book_pages_purple_book_pages_proof_l194_19401


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l194_19462

theorem absolute_value_theorem (x y : ℝ) (hx : x > 0) :
  |x + 1 - Real.sqrt ((x + y)^2)| = 
    if x + y ≥ 0 then |1 - y| else |2*x + y + 1| := by sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l194_19462


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_l194_19456

/-- Represents a sequence of five natural numbers with a constant difference -/
structure ArithmeticSequence :=
  (first : ℕ)
  (diff : ℕ)

/-- Converts a natural number to a string representation -/
def toLetterRepresentation (n : ℕ) : String :=
  match n with
  | 5 => "T"
  | 12 => "EL"
  | 19 => "EK"
  | 26 => "LA"
  | 33 => "SS"
  | _ => ""

/-- The main theorem to be proved -/
theorem arithmetic_sequence_unique :
  ∀ (seq : ArithmeticSequence),
    (seq.first = 5 ∧ seq.diff = 7) ↔
    (toLetterRepresentation seq.first = "T" ∧
     toLetterRepresentation (seq.first + seq.diff) = "EL" ∧
     toLetterRepresentation (seq.first + 2 * seq.diff) = "EK" ∧
     toLetterRepresentation (seq.first + 3 * seq.diff) = "LA" ∧
     toLetterRepresentation (seq.first + 4 * seq.diff) = "SS") :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_l194_19456


namespace NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l194_19409

theorem no_positive_a_satisfies_inequality :
  ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l194_19409


namespace NUMINAMATH_CALUDE_three_non_congruent_triangles_l194_19453

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 11 -/
def triangles_with_perimeter_11 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 11}

/-- The theorem to be proved -/
theorem three_non_congruent_triangles : 
  ∃ (t1 t2 t3 : IntTriangle), 
    t1 ∈ triangles_with_perimeter_11 ∧
    t2 ∈ triangles_with_perimeter_11 ∧
    t3 ∈ triangles_with_perimeter_11 ∧
    ¬(are_congruent t1 t2) ∧
    ¬(are_congruent t1 t3) ∧
    ¬(are_congruent t2 t3) ∧
    ∀ (t : IntTriangle), t ∈ triangles_with_perimeter_11 → 
      (are_congruent t t1 ∨ are_congruent t t2 ∨ are_congruent t t3) :=
by
  sorry

end NUMINAMATH_CALUDE_three_non_congruent_triangles_l194_19453


namespace NUMINAMATH_CALUDE_smallest_x_for_fraction_l194_19457

theorem smallest_x_for_fraction (x : ℕ) (y : ℤ) : 
  (3 : ℚ) / 4 = y / (256 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_fraction_l194_19457


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l194_19479

def cost_per_5_pages : ℚ := 7
def pages_per_5 : ℚ := 5
def total_money : ℚ := 1500  -- in cents

def pages_copied : ℕ := 1071

theorem copy_pages_theorem :
  ⌊(total_money / cost_per_5_pages) * pages_per_5⌋ = pages_copied :=
by sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l194_19479


namespace NUMINAMATH_CALUDE_larger_integer_value_l194_19482

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 5 / 2)
  (h_product : (a : ℕ) * b = 160) : 
  max a b = 20 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l194_19482


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_2beta_l194_19491

theorem cos_2alpha_plus_2beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_2beta_l194_19491


namespace NUMINAMATH_CALUDE_nickels_count_l194_19435

/-- Represents the number of coins of each type found by Harriett --/
structure CoinCounts where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in cents for a given set of coin counts --/
def totalValue (coins : CoinCounts) : Nat :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem stating that given the number of quarters, dimes, pennies, and the total value,
    the number of nickels must be 3 to make the total $3.00 --/
theorem nickels_count (coins : CoinCounts) :
  coins.quarters = 10 ∧ coins.dimes = 3 ∧ coins.pennies = 5 ∧ totalValue coins = 300 →
  coins.nickels = 3 := by
  sorry


end NUMINAMATH_CALUDE_nickels_count_l194_19435


namespace NUMINAMATH_CALUDE_josh_lost_marbles_l194_19429

/-- Represents the number of marbles Josh lost -/
def marbles_lost (initial current : ℕ) : ℕ := initial - current

/-- Theorem stating that Josh lost 5 marbles -/
theorem josh_lost_marbles : marbles_lost 9 4 = 5 := by sorry

end NUMINAMATH_CALUDE_josh_lost_marbles_l194_19429


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l194_19441

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) (hd : ∃ n : ℤ, d = 9 * n) : 
  ∃ k : ℤ, c + d = 3 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l194_19441


namespace NUMINAMATH_CALUDE_abs_two_i_over_one_minus_i_l194_19434

/-- The absolute value of the complex number 2i / (1-i) is √2 -/
theorem abs_two_i_over_one_minus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i / (1 - i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_i_over_one_minus_i_l194_19434


namespace NUMINAMATH_CALUDE_most_likely_parent_genotypes_l194_19476

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Determines if a rabbit has hairy fur based on its genotype -/
def isHairy (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | Allele.h, Allele.h => true
  | _, _ => false

/-- The probability of the hairy allele in the population -/
def p : ℝ := 0.1

/-- Theorem: The most likely genotype combination for parents resulting in all hairy offspring -/
theorem most_likely_parent_genotypes :
  ∃ (parent1 parent2 : Genotype),
    isHairy parent1 ∧
    ¬isHairy parent2 ∧
    (∀ (offspring : Genotype),
      (offspring.allele1 = parent1.allele1 ∨ offspring.allele1 = parent1.allele2) ∧
      (offspring.allele2 = parent2.allele1 ∨ offspring.allele2 = parent2.allele2) →
      isHairy offspring) ∧
    parent1 = ⟨Allele.H, Allele.H⟩ ∧
    parent2 = ⟨Allele.S, Allele.h⟩ :=
by sorry


end NUMINAMATH_CALUDE_most_likely_parent_genotypes_l194_19476


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l194_19416

/-- Given a line y = mx + 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    prove that the possible slopes m satisfy m^2 ≥ 1/55. -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 3) → m^2 ≥ 1/55 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l194_19416


namespace NUMINAMATH_CALUDE_special_parallelogram_sides_prove_special_parallelogram_sides_l194_19495

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The perimeter of the parallelogram
  perimeter : ℝ
  -- The measure of the acute angle in degrees
  acuteAngle : ℝ
  -- The ratio in which the diagonal divides the obtuse angle
  diagonalRatio : ℝ × ℝ
  -- The sides of the parallelogram
  sides : ℝ × ℝ × ℝ × ℝ

/-- The theorem stating the properties of the special parallelogram -/
theorem special_parallelogram_sides (p : SpecialParallelogram) :
  p.perimeter = 90 ∧ 
  p.acuteAngle = 60 ∧ 
  p.diagonalRatio = (1, 3) →
  p.sides = (15, 15, 30, 30) := by
  sorry

/-- Proof that the sides of the special parallelogram are 15, 15, 30, and 30 -/
theorem prove_special_parallelogram_sides : 
  ∃ (p : SpecialParallelogram), 
    p.perimeter = 90 ∧ 
    p.acuteAngle = 60 ∧ 
    p.diagonalRatio = (1, 3) ∧ 
    p.sides = (15, 15, 30, 30) := by
  sorry

end NUMINAMATH_CALUDE_special_parallelogram_sides_prove_special_parallelogram_sides_l194_19495


namespace NUMINAMATH_CALUDE_profit_growth_rate_l194_19449

/-- The average monthly growth rate that achieves the target profit -/
def average_growth_rate : ℝ := 0.2

/-- The initial profit in June -/
def initial_profit : ℝ := 2500

/-- The target profit in August -/
def target_profit : ℝ := 3600

/-- The number of months between June and August -/
def months : ℕ := 2

theorem profit_growth_rate :
  initial_profit * (1 + average_growth_rate) ^ months = target_profit :=
sorry

end NUMINAMATH_CALUDE_profit_growth_rate_l194_19449


namespace NUMINAMATH_CALUDE_max_value_of_f_l194_19466

/-- A cubic function with a constant term -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

/-- The minimum value of f on the interval [1,3] -/
def min_value : ℝ := 2

/-- The interval on which we're considering the function -/
def interval : Set ℝ := Set.Icc 1 3

theorem max_value_of_f (m : ℝ) (h : ∃ x ∈ interval, ∀ y ∈ interval, f m y ≥ f m x ∧ f m x = min_value) :
  ∃ x ∈ interval, ∀ y ∈ interval, f m y ≤ f m x ∧ f m x = 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l194_19466


namespace NUMINAMATH_CALUDE_duty_arrangements_l194_19460

/-- Represents the number of teachers -/
def num_teachers : ℕ := 3

/-- Represents the number of days in a week -/
def num_days : ℕ := 5

/-- Represents the number of teachers required on Monday -/
def teachers_on_monday : ℕ := 2

/-- Represents the number of duty days per teacher -/
def duty_days_per_teacher : ℕ := 2

/-- Theorem stating the number of possible duty arrangements -/
theorem duty_arrangements :
  (num_teachers.choose teachers_on_monday) * ((num_days - 1).choose (num_teachers - 1)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_duty_arrangements_l194_19460


namespace NUMINAMATH_CALUDE_cost_price_calculation_l194_19402

/-- Given a sale price including tax, sales tax rate, and profit rate,
    calculate the approximate cost price of an article. -/
theorem cost_price_calculation (sale_price_with_tax : ℝ)
                                (sales_tax_rate : ℝ)
                                (profit_rate : ℝ)
                                (h1 : sale_price_with_tax = 616)
                                (h2 : sales_tax_rate = 0.1)
                                (h3 : profit_rate = 0.17) :
  ∃ (cost_price : ℝ), 
    (cost_price * (1 + profit_rate) * (1 + sales_tax_rate) = sale_price_with_tax) ∧
    (abs (cost_price - 478.77) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l194_19402
