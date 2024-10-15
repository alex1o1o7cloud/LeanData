import Mathlib

namespace NUMINAMATH_CALUDE_half_of_large_number_l2006_200691

theorem half_of_large_number : (1.2 * 10^30) / 2 = 6.0 * 10^29 := by
  sorry

end NUMINAMATH_CALUDE_half_of_large_number_l2006_200691


namespace NUMINAMATH_CALUDE_isosceles_triangle_min_ratio_l2006_200654

/-- The minimum value of (2a + b) / a for an isosceles triangle with two equal sides of length a and base of length b is 3 -/
theorem isosceles_triangle_min_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a ≥ b) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x ≥ y → (2 * x + y) / x ≥ (2 * a + b) / a ∧ (2 * a + b) / a = 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_min_ratio_l2006_200654


namespace NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_perimeter_6_l2006_200603

/-- The minimum value of the hypotenuse of a right triangle with perimeter 6 -/
theorem min_hypotenuse_right_triangle_perimeter_6 :
  ∃ (c : ℝ), c > 0 ∧ c = 6 * (Real.sqrt 2 - 1) ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = c^2 → a + b + c = 6 →
  c ≤ 6 * (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_perimeter_6_l2006_200603


namespace NUMINAMATH_CALUDE_value_added_to_doubled_number_l2006_200695

theorem value_added_to_doubled_number (initial_number : ℕ) (added_value : ℕ) : 
  initial_number = 10 → 
  3 * (2 * initial_number + added_value) = 84 → 
  added_value = 8 := by
sorry

end NUMINAMATH_CALUDE_value_added_to_doubled_number_l2006_200695


namespace NUMINAMATH_CALUDE_temperature_function_correct_and_linear_l2006_200636

/-- Represents the temperature change per kilometer of altitude increase -/
def temperature_change_per_km : ℝ := -6

/-- Represents the ground temperature in Celsius -/
def ground_temperature : ℝ := 20

/-- Represents the temperature y in Celsius at a height of x kilometers above the ground -/
def temperature (x : ℝ) : ℝ := temperature_change_per_km * x + ground_temperature

theorem temperature_function_correct_and_linear :
  (∀ x : ℝ, temperature x = temperature_change_per_km * x + ground_temperature) ∧
  (∃ m b : ℝ, ∀ x : ℝ, temperature x = m * x + b) :=
by sorry

end NUMINAMATH_CALUDE_temperature_function_correct_and_linear_l2006_200636


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2006_200608

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 257 → ¬(((m + 7) % 8 = 0) ∧ ((m + 7) % 11 = 0) ∧ ((m + 7) % 24 = 0))) ∧
  ((257 + 7) % 8 = 0) ∧ ((257 + 7) % 11 = 0) ∧ ((257 + 7) % 24 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2006_200608


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_23_l2006_200696

theorem greatest_four_digit_multiple_of_23 : ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 23 ∣ n → n ≤ 9978 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_23_l2006_200696


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l2006_200689

/-- Given a polynomial function g(x) = ax^7 + bx^3 + dx^2 + cx - 8,
    prove that if g(-7) = 3 and d = 0, then g(7) = -19 -/
theorem polynomial_symmetry (a b c d : ℝ) :
  let g := λ x : ℝ => a * x^7 + b * x^3 + d * x^2 + c * x - 8
  (g (-7) = 3) → (d = 0) → (g 7 = -19) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l2006_200689


namespace NUMINAMATH_CALUDE_ball_probability_after_swap_l2006_200678

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.red + bag.yellow + bag.blue)

/-- The initial contents of the bag -/
def initialBag : BagContents :=
  { red := 10, yellow := 2, blue := 8 }

/-- The contents of the bag after removing red balls and adding yellow balls -/
def finalBag (n : ℕ) : BagContents :=
  { red := initialBag.red - n, yellow := initialBag.yellow + n, blue := initialBag.blue }

theorem ball_probability_after_swap :
  probability (finalBag 6) (finalBag 6).yellow = 2/5 :=
sorry

end NUMINAMATH_CALUDE_ball_probability_after_swap_l2006_200678


namespace NUMINAMATH_CALUDE_velvet_for_cloak_l2006_200692

/-- The number of hats that can be made from one yard of velvet -/
def hats_per_yard : ℕ := 4

/-- The total number of yards of velvet needed for 6 cloaks and 12 hats -/
def total_yards : ℕ := 21

/-- The number of cloaks made with the total yards -/
def num_cloaks : ℕ := 6

/-- The number of hats made with the total yards -/
def num_hats : ℕ := 12

/-- The number of yards needed to make one cloak -/
def yards_per_cloak : ℚ := 3

theorem velvet_for_cloak :
  yards_per_cloak = (total_yards - (num_hats / hats_per_yard : ℚ)) / num_cloaks := by
  sorry

end NUMINAMATH_CALUDE_velvet_for_cloak_l2006_200692


namespace NUMINAMATH_CALUDE_initial_workers_count_l2006_200612

/-- The work rate of workers (depth dug per worker per hour) -/
def work_rate : ℝ := sorry

/-- The initial number of workers -/
def initial_workers : ℕ := sorry

/-- The depth of the hole in meters -/
def hole_depth : ℝ := 30

theorem initial_workers_count : initial_workers = 45 := by
  have h1 : initial_workers * 8 * work_rate = hole_depth := sorry
  have h2 : (initial_workers + 15) * 6 * work_rate = hole_depth := sorry
  sorry


end NUMINAMATH_CALUDE_initial_workers_count_l2006_200612


namespace NUMINAMATH_CALUDE_work_completion_time_l2006_200609

theorem work_completion_time (a_total_days b_remaining_days : ℚ) 
  (h1 : a_total_days = 15)
  (h2 : b_remaining_days = 10) : 
  let a_work_days : ℚ := 5
  let a_work_fraction : ℚ := a_work_days / a_total_days
  let b_work_fraction : ℚ := 1 - a_work_fraction
  b_remaining_days / b_work_fraction = 15 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2006_200609


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2006_200690

theorem smallest_solution_of_equation (y : ℝ) : 
  (3 * y^2 + 36 * y - 90 = y * (y + 18)) → y ≥ -15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2006_200690


namespace NUMINAMATH_CALUDE_commercial_break_length_is_47_l2006_200661

/-- Calculates the total length of a commercial break given the following conditions:
    - Three commercials of 5, 6, and 7 minutes
    - Eleven 2-minute commercials
    - Two of the 2-minute commercials overlap with a 3-minute interruption and restart after
-/
def commercial_break_length : ℕ :=
  let long_commercials := 5 + 6 + 7
  let short_commercials := 11 * 2
  let interruption := 3
  let restarted_commercials := 2 * 2
  long_commercials + short_commercials + interruption + restarted_commercials

/-- Theorem stating that the commercial break length is 47 minutes -/
theorem commercial_break_length_is_47 : commercial_break_length = 47 := by
  sorry

end NUMINAMATH_CALUDE_commercial_break_length_is_47_l2006_200661


namespace NUMINAMATH_CALUDE_circle_center_l2006_200621

/-- The center of the circle with equation x^2 - 10x + y^2 - 4y = -4 is (5, 2) -/
theorem circle_center (x y : ℝ) :
  (x^2 - 10*x + y^2 - 4*y = -4) ↔ ((x - 5)^2 + (y - 2)^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l2006_200621


namespace NUMINAMATH_CALUDE_no_intersection_l2006_200617

theorem no_intersection : ¬∃ x : ℝ, |3*x + 6| = -|4*x - 3| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l2006_200617


namespace NUMINAMATH_CALUDE_triangle_inequality_l2006_200656

theorem triangle_inequality (a b c : ℝ) (h_area : (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = 1/4) (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2006_200656


namespace NUMINAMATH_CALUDE_f_negative_expression_l2006_200655

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function f for x > 0
def f_positive (x : ℝ) : ℝ :=
  x^3 + x + 1

-- Theorem statement
theorem f_negative_expression 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = -x^3 - x + 1 := by
sorry

end NUMINAMATH_CALUDE_f_negative_expression_l2006_200655


namespace NUMINAMATH_CALUDE_equal_population_in_17_years_l2006_200614

/-- The number of years it takes for two village populations to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_rate + x_rate)

/-- Theorem: Given the initial populations and rates of change, it takes 17 years for the populations to be equal -/
theorem equal_population_in_17_years :
  years_to_equal_population 76000 1200 42000 800 = 17 := by
  sorry

end NUMINAMATH_CALUDE_equal_population_in_17_years_l2006_200614


namespace NUMINAMATH_CALUDE_delivery_pay_difference_l2006_200681

/-- Calculate the difference in pay between two workers --/
theorem delivery_pay_difference (deliveries_worker1 : ℕ) 
  (pay_per_delivery : ℕ) : 
  deliveries_worker1 = 96 →
  pay_per_delivery = 100 →
  (deliveries_worker1 * pay_per_delivery : ℕ) - 
  ((deliveries_worker1 * 3 / 4) * pay_per_delivery : ℕ) = 2400 := by
sorry

end NUMINAMATH_CALUDE_delivery_pay_difference_l2006_200681


namespace NUMINAMATH_CALUDE_debate_schedule_ways_l2006_200675

/-- Number of debaters from each school -/
def num_debaters : ℕ := 4

/-- Total number of debates -/
def total_debates : ℕ := num_debaters * num_debaters

/-- Maximum number of debates per session -/
def max_debates_per_session : ℕ := 3

/-- Number of ways to schedule debates -/
def schedule_ways : ℕ := 20922789888000

/-- Theorem stating the number of ways to schedule debates -/
theorem debate_schedule_ways :
  (total_debates.factorial) / (max_debates_per_session.factorial ^ 5 * 1) = schedule_ways := by
  sorry

end NUMINAMATH_CALUDE_debate_schedule_ways_l2006_200675


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2006_200616

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2006_200616


namespace NUMINAMATH_CALUDE_square_of_negative_triple_l2006_200657

theorem square_of_negative_triple (a : ℝ) : (-3 * a)^2 = 9 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_of_negative_triple_l2006_200657


namespace NUMINAMATH_CALUDE_evaluate_expression_l2006_200627

theorem evaluate_expression : 
  2011 * 20122012 * 201320132013 - 2013 * 20112011 * 201220122012 = 
  -2 * 2012 * 2013 * 10001 * 100010001 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2006_200627


namespace NUMINAMATH_CALUDE_weight_of_BaO_l2006_200632

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of barium oxide (BaO) in g/mol -/
def molecular_weight_BaO : ℝ := atomic_weight_Ba + atomic_weight_O

/-- The number of moles of barium oxide -/
def moles_BaO : ℝ := 6

/-- Theorem: The weight of 6 moles of barium oxide (BaO) is 919.98 grams -/
theorem weight_of_BaO : moles_BaO * molecular_weight_BaO = 919.98 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaO_l2006_200632


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2006_200669

-- Problem 1
theorem problem_1 : (-12) + 13 + (-18) + 16 = -1 := by sorry

-- Problem 2
theorem problem_2 : 19.5 + (-6.9) + (-3.1) + (-9.5) = 0 := by sorry

-- Problem 3
theorem problem_3 : (6/5 : ℚ) * (-1/3 - 1/2) / (5/4 : ℚ) = -4/5 := by sorry

-- Problem 4
theorem problem_4 : 18 + 32 * (-1/2)^5 - (1/2)^4 * (-2)^5 = 19 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2006_200669


namespace NUMINAMATH_CALUDE_scalper_ticket_percentage_l2006_200638

theorem scalper_ticket_percentage :
  let normal_price : ℝ := 50
  let website_tickets : ℕ := 2
  let scalper_tickets : ℕ := 2
  let discounted_tickets : ℕ := 1
  let discounted_percentage : ℝ := 60
  let total_paid : ℝ := 360
  let scalper_discount : ℝ := 10

  ∃ P : ℝ,
    website_tickets * normal_price +
    scalper_tickets * (P / 100 * normal_price) - scalper_discount +
    discounted_tickets * (discounted_percentage / 100 * normal_price) = total_paid ∧
    P = 480 :=
by sorry

end NUMINAMATH_CALUDE_scalper_ticket_percentage_l2006_200638


namespace NUMINAMATH_CALUDE_candy_count_proof_l2006_200687

/-- Calculates the total number of candy pieces given the number of packages and pieces per package -/
def total_candy_pieces (packages : ℕ) (pieces_per_package : ℕ) : ℕ :=
  packages * pieces_per_package

/-- Proves that 45 packages of candy with 9 pieces each results in 405 total pieces -/
theorem candy_count_proof :
  total_candy_pieces 45 9 = 405 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_proof_l2006_200687


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2006_200671

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2006_200671


namespace NUMINAMATH_CALUDE_pen_profit_percentage_pen_profit_percentage_result_l2006_200670

/-- Given a purchase of pens with specific pricing and discount conditions, 
    calculate the profit percentage. -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (marked_price_ratio : ℚ) 
  (discount_percent : ℚ) : ℚ :=
  let cost_price := marked_price_ratio
  let selling_price := num_pens * (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  by
    -- Assuming num_pens = 50, marked_price_ratio = 46/50, discount_percent = 1
    sorry

/-- The profit percentage for the given pen sale scenario is 7.61%. -/
theorem pen_profit_percentage_result : 
  pen_profit_percentage 50 (46/50) 1 = 761/100 :=
by sorry

end NUMINAMATH_CALUDE_pen_profit_percentage_pen_profit_percentage_result_l2006_200670


namespace NUMINAMATH_CALUDE_specific_rectangle_burning_time_l2006_200647

/-- Represents a rectangular structure made of toothpicks -/
structure ToothpickRectangle where
  rows : ℕ
  columns : ℕ
  toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  burn_time_per_toothpick : ℕ
  start_corners : ℕ

/-- Calculates the total burning time for a toothpick rectangle -/
def total_burning_time (rect : ToothpickRectangle) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem statement for the burning time of the specific rectangle -/
theorem specific_rectangle_burning_time :
  let rect := ToothpickRectangle.mk 3 5 38
  let props := BurningProperties.mk 10 2
  total_burning_time rect props = 65 :=
by sorry

end NUMINAMATH_CALUDE_specific_rectangle_burning_time_l2006_200647


namespace NUMINAMATH_CALUDE_function_difference_l2006_200694

/-- Given a function f(x) = 3x^2 + 5x - 4, prove that f(x+h) - f(x) = h(3h + 6x + 5) for all real x and h. -/
theorem function_difference (x h : ℝ) : 
  let f : ℝ → ℝ := λ t => 3 * t^2 + 5 * t - 4
  f (x + h) - f x = h * (3 * h + 6 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l2006_200694


namespace NUMINAMATH_CALUDE_door_ticket_cost_l2006_200645

/-- Proves the cost of tickets purchased at the door given ticket sales information -/
theorem door_ticket_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advance_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_tickets = 140)
  (h2 : total_revenue = 1720)
  (h3 : advance_ticket_cost = 8)
  (h4 : advance_tickets_sold = 100) :
  (total_revenue - advance_ticket_cost * advance_tickets_sold) / (total_tickets - advance_tickets_sold) = 23 := by
  sorry


end NUMINAMATH_CALUDE_door_ticket_cost_l2006_200645


namespace NUMINAMATH_CALUDE_equation_with_operations_l2006_200664

theorem equation_with_operations : ∃ (op1 op2 op3 : ℕ → ℕ → ℕ), 
  op1 6 (op2 3 (op3 4 2)) = 24 :=
sorry

end NUMINAMATH_CALUDE_equation_with_operations_l2006_200664


namespace NUMINAMATH_CALUDE_curve_transformation_l2006_200637

theorem curve_transformation (x : ℝ) : 
  Real.sin (x + π / 2) = Real.sin (2 * (x + π / 12) + 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l2006_200637


namespace NUMINAMATH_CALUDE_beijing_spirit_max_l2006_200653

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- The equation: Patriotism × Innovation × Inclusiveness + Integrity = Beijing Spirit -/
def equation (patriotism innovation inclusiveness integrity : Digit) (beijingSpirit : Nat) :=
  (patriotism.val * innovation.val * inclusiveness.val + integrity.val = beijingSpirit)

/-- All digits are different -/
def all_different (patriotism nation creation new inclusiveness tolerance integrity virtue : Digit) :=
  patriotism ≠ nation ∧ patriotism ≠ creation ∧ patriotism ≠ new ∧ patriotism ≠ inclusiveness ∧
  patriotism ≠ tolerance ∧ patriotism ≠ integrity ∧ patriotism ≠ virtue ∧
  nation ≠ creation ∧ nation ≠ new ∧ nation ≠ inclusiveness ∧ nation ≠ tolerance ∧
  nation ≠ integrity ∧ nation ≠ virtue ∧ creation ≠ new ∧ creation ≠ inclusiveness ∧
  creation ≠ tolerance ∧ creation ≠ integrity ∧ creation ≠ virtue ∧ new ≠ inclusiveness ∧
  new ≠ tolerance ∧ new ≠ integrity ∧ new ≠ virtue ∧ inclusiveness ≠ tolerance ∧
  inclusiveness ≠ integrity ∧ inclusiveness ≠ virtue ∧ tolerance ≠ integrity ∧ tolerance ≠ virtue ∧
  integrity ≠ virtue

theorem beijing_spirit_max (patriotism nation creation new inclusiveness tolerance integrity virtue : Digit) :
  all_different patriotism nation creation new inclusiveness tolerance integrity virtue →
  equation patriotism creation inclusiveness integrity 9898 →
  integrity.val = 98 := by
  sorry

end NUMINAMATH_CALUDE_beijing_spirit_max_l2006_200653


namespace NUMINAMATH_CALUDE_bridge_length_l2006_200602

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2006_200602


namespace NUMINAMATH_CALUDE_hash_one_two_three_l2006_200649

/-- The operation # defined for real numbers a, b, and c -/
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem stating that #(1, 2, 3) = -8 -/
theorem hash_one_two_three : hash 1 2 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_one_two_three_l2006_200649


namespace NUMINAMATH_CALUDE_cookie_sales_revenue_l2006_200613

theorem cookie_sales_revenue : 
  let chocolate_cookies : ℕ := 220
  let chocolate_price : ℕ := 1
  let vanilla_cookies : ℕ := 70
  let vanilla_price : ℕ := 2
  chocolate_cookies * chocolate_price + vanilla_cookies * vanilla_price = 360 := by
sorry

end NUMINAMATH_CALUDE_cookie_sales_revenue_l2006_200613


namespace NUMINAMATH_CALUDE_lobachevsky_angle_existence_l2006_200677

theorem lobachevsky_angle_existence (A B C : Real) 
  (hB : 0 < B ∧ B < Real.pi / 2) 
  (hC : 0 < C ∧ C < Real.pi / 2) : 
  ∃ X, Real.sin X = (Real.sin B * Real.sin C) / (1 - Real.cos A * Real.cos B * Real.cos C) := by
  sorry

end NUMINAMATH_CALUDE_lobachevsky_angle_existence_l2006_200677


namespace NUMINAMATH_CALUDE_same_value_point_m_two_distinct_same_value_points_l2006_200644

/-- Quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + m

theorem same_value_point_m (m : ℝ) :
  f m 2 = 2 → m = -8 := by sorry

theorem two_distinct_same_value_points (m : ℝ) (a b : ℝ) :
  (∃ (a b : ℝ), a < 1 ∧ 1 < b ∧ f m a = a ∧ f m b = b) →
  m < -3 := by sorry

end NUMINAMATH_CALUDE_same_value_point_m_two_distinct_same_value_points_l2006_200644


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2006_200673

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  3 * a 9 - a 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2006_200673


namespace NUMINAMATH_CALUDE_equation_solution_l2006_200622

theorem equation_solution (n : ℤ) : 
  (5 : ℚ) / 4 * n + (5 : ℚ) / 4 = n ↔ ∃ k : ℤ, n = -5 + 1024 * k := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2006_200622


namespace NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_1_l2006_200659

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

-- State the theorem
theorem monotonically_decreasing_implies_a_leq_1 :
  ∀ a : ℝ, (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_1_l2006_200659


namespace NUMINAMATH_CALUDE_simplify_expression_l2006_200688

theorem simplify_expression : (8 * 10^7) / (4 * 10^2) = 200000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2006_200688


namespace NUMINAMATH_CALUDE_area_between_tangent_circles_l2006_200629

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles
  (r₁ : ℝ) (r₂ : ℝ) (d : ℝ)
  (h₁ : r₁ = 4)
  (h₂ : r₂ = 7)
  (h₃ : d = 3)
  (h₄ : d = r₂ - r₁) :
  π * (r₂^2 - r₁^2) = 33 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_tangent_circles_l2006_200629


namespace NUMINAMATH_CALUDE_banana_arrangements_l2006_200633

def banana_length : ℕ := 6
def num_a : ℕ := 3
def num_n : ℕ := 2

theorem banana_arrangements : 
  (banana_length.factorial) / (num_a.factorial * num_n.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2006_200633


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2006_200610

theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  (((x^2 : ℝ) / 49) - ((y^2 : ℝ) / 36) = 1) →
  (∃ (k : ℝ), y = k * m * x ∧ y = -k * m * x) →
  (m > 0) →
  (m = 6/7) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2006_200610


namespace NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_9_and_9_1_l2006_200601

theorem unique_number_divisible_by_24_with_cube_root_between_9_and_9_1 :
  ∃! (n : ℕ), n > 0 ∧ 24 ∣ n ∧ 9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_9_and_9_1_l2006_200601


namespace NUMINAMATH_CALUDE_evaluate_expression_l2006_200672

theorem evaluate_expression : (2 : ℕ) ^ (3 ^ 2) + 3 ^ (2 ^ 3) = 7073 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2006_200672


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2006_200619

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_sequence (x y : ℝ) :
  ∀ a : ℕ → ℝ, arithmetic_sequence a →
  a 1 = x - y → a 2 = x → a 3 = x + y → a 4 = x + 2*y →
  a 5 = x + 3*y := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2006_200619


namespace NUMINAMATH_CALUDE_prism_length_l2006_200668

/-- A regular rectangular prism with given edge sum and proportions -/
structure RegularPrism where
  width : ℝ
  length : ℝ
  height : ℝ
  edge_sum : ℝ
  length_prop : length = 4 * width
  height_prop : height = 3 * width
  sum_prop : 4 * length + 4 * width + 4 * height = edge_sum

/-- The length of a regular rectangular prism with edge sum 256 cm is 32 cm -/
theorem prism_length (p : RegularPrism) (h : p.edge_sum = 256) : p.length = 32 := by
  sorry

end NUMINAMATH_CALUDE_prism_length_l2006_200668


namespace NUMINAMATH_CALUDE_mike_ride_length_l2006_200611

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startFee : ℝ
  perMileFee : ℝ
  bridgeToll : ℝ

/-- Calculates the total fare for a given ride -/
def calculateFare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.startFee + fare.perMileFee * miles + fare.bridgeToll

theorem mike_ride_length :
  let mikeFare : TaxiFare := { startFee := 2.5, perMileFee := 0.25, bridgeToll := 0 }
  let annieFare : TaxiFare := { startFee := 2.5, perMileFee := 0.25, bridgeToll := 5 }
  let annieMiles : ℝ := 26
  ∃ mikeMiles : ℝ, mikeMiles = 36 ∧ 
    calculateFare mikeFare mikeMiles = calculateFare annieFare annieMiles :=
by sorry

end NUMINAMATH_CALUDE_mike_ride_length_l2006_200611


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisector_segments_l2006_200686

/-- Given a right triangle where an acute angle bisector divides the adjacent leg into segments m and n,
    prove the lengths of the other leg and hypotenuse. -/
theorem right_triangle_angle_bisector_segments (m n : ℝ) (h : m > n) :
  ∃ (other_leg hypotenuse : ℝ),
    other_leg = n * Real.sqrt ((m + n) / (m - n)) ∧
    hypotenuse = m * Real.sqrt ((m + n) / (m - n)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_bisector_segments_l2006_200686


namespace NUMINAMATH_CALUDE_probability_sum_six_l2006_200652

-- Define a die as having 6 faces
def die : ℕ := 6

-- Define the favorable outcomes (combinations that sum to 6)
def favorable_outcomes : ℕ := 5

-- Define the total number of possible outcomes
def total_outcomes : ℕ := die * die

-- State the theorem
theorem probability_sum_six (d : ℕ) (h : d = die) : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_l2006_200652


namespace NUMINAMATH_CALUDE_probability_AR55_l2006_200698

/-- Represents the set of possible symbols for each position in the license plate -/
def LicensePlateSymbols : Fin 4 → Type
  | 0 => Fin 5  -- Vowels (A, E, I, O, U)
  | 1 => Fin 21 -- Non-vowels (consonants)
  | 2 => Fin 10 -- Digits (0-9)
  | 3 => Fin 10 -- Digits (0-9)

/-- The total number of possible license plates -/
def totalLicensePlates : ℕ := 5 * 21 * 10 * 10

/-- Represents a specific license plate -/
def SpecificPlate : Fin 4 → ℕ
  | 0 => 0  -- 'A' (first vowel)
  | 1 => 17 -- 'R' (18th consonant, 0-indexed)
  | 2 => 5  -- '5'
  | 3 => 5  -- '5'

/-- The probability of randomly selecting the license plate "AR55" -/
theorem probability_AR55 : 
  (1 : ℚ) / totalLicensePlates = 1 / 10500 :=
sorry

end NUMINAMATH_CALUDE_probability_AR55_l2006_200698


namespace NUMINAMATH_CALUDE_divisor_problem_l2006_200674

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a number is divisible by another -/
def is_divisible_by (a b : ℕ) : Prop := sorry

theorem divisor_problem (n : ℕ+) (k : ℕ) :
  num_divisors n = 72 →
  num_divisors (5 * n) = 120 →
  (∀ m : ℕ, m > k → ¬ is_divisible_by n (5^m)) →
  is_divisible_by n (5^k) →
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2006_200674


namespace NUMINAMATH_CALUDE_six_partitions_into_three_or_fewer_l2006_200604

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty parts -/
def partitions_into_k_or_fewer (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to partition 6 indistinguishable objects into 3 or fewer non-empty parts -/
theorem six_partitions_into_three_or_fewer : partitions_into_k_or_fewer 6 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_partitions_into_three_or_fewer_l2006_200604


namespace NUMINAMATH_CALUDE_bird_population_theorem_l2006_200646

/-- Represents the bird population in the nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawks : ℝ
  paddyfield_warblers : ℝ
  kingfishers : ℝ

/-- Conditions for the bird population -/
def ValidBirdPopulation (bp : BirdPopulation) : Prop :=
  bp.total > 0 ∧
  bp.hawks = 0.3 * bp.total ∧
  bp.paddyfield_warblers = 0.4 * (bp.total - bp.hawks) ∧
  bp.kingfishers = 0.25 * bp.paddyfield_warblers

/-- Theorem stating the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
theorem bird_population_theorem (bp : BirdPopulation) 
  (h : ValidBirdPopulation bp) : 
  (bp.total - (bp.hawks + bp.paddyfield_warblers + bp.kingfishers)) / bp.total = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_bird_population_theorem_l2006_200646


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l2006_200642

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 overall_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 35 / 100 →
  overall_germination_rate = 26 / 100 →
  ∃ (germination_rate_plot1 : ℚ),
    germination_rate_plot1 = 20 / 100 ∧
    germination_rate_plot1 * seeds_plot1 + germination_rate_plot2 * seeds_plot2 = 
    overall_germination_rate * (seeds_plot1 + seeds_plot2) := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l2006_200642


namespace NUMINAMATH_CALUDE_total_collection_l2006_200634

def friend_payment : ℕ := 5
def brother_payment : ℕ := 8
def cousin_payment : ℕ := 4
def num_days : ℕ := 7

theorem total_collection :
  (friend_payment * num_days) + (brother_payment * num_days) + (cousin_payment * num_days) = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_collection_l2006_200634


namespace NUMINAMATH_CALUDE_fathers_cookies_l2006_200640

theorem fathers_cookies (total cookies_charlie cookies_mother : ℕ) 
  (h1 : total = 30)
  (h2 : cookies_charlie = 15)
  (h3 : cookies_mother = 5) :
  total - cookies_charlie - cookies_mother = 10 := by
sorry

end NUMINAMATH_CALUDE_fathers_cookies_l2006_200640


namespace NUMINAMATH_CALUDE_hole_empty_time_l2006_200635

/-- Given a pipe that can fill a tank in 15 hours, and with a hole causing
    the tank to fill in 20 hours instead, prove that the time it takes for
    the hole to empty a full tank is 60 hours. -/
theorem hole_empty_time (fill_time_no_hole fill_time_with_hole : ℝ)
  (h1 : fill_time_no_hole = 15)
  (h2 : fill_time_with_hole = 20) :
  (fill_time_no_hole * fill_time_with_hole) /
    (fill_time_with_hole - fill_time_no_hole) = 60 := by
  sorry

end NUMINAMATH_CALUDE_hole_empty_time_l2006_200635


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2006_200648

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) ↔ m = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2006_200648


namespace NUMINAMATH_CALUDE_sour_candy_percentage_l2006_200631

theorem sour_candy_percentage (total_candies : ℕ) (num_people : ℕ) (good_candies_per_person : ℕ) :
  total_candies = 300 →
  num_people = 3 →
  good_candies_per_person = 60 →
  (total_candies - num_people * good_candies_per_person) / total_candies = 2/5 :=
by
  sorry

end NUMINAMATH_CALUDE_sour_candy_percentage_l2006_200631


namespace NUMINAMATH_CALUDE_range_of_a_l2006_200680

open Set

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, x^2 - a ≥ 0) ∨ 
  (∃ x₀ : ℝ, ∀ x : ℝ, x + (a - 1) * x₀ + 1 < 0) ∧
  ¬((∀ x ∈ Icc 1 2, x^2 - a ≥ 0) ∧ 
    (∃ x₀ : ℝ, ∀ x : ℝ, x + (a - 1) * x₀ + 1 < 0)) →
  a > 3 ∨ a ∈ Icc (-1) 1 :=
by sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l2006_200680


namespace NUMINAMATH_CALUDE_unique_pair_power_sum_l2006_200665

theorem unique_pair_power_sum : 
  ∃! (a b : ℕ), ∀ (n : ℕ), ∃ (c : ℕ), a^n + b^n = c^(n+1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_pair_power_sum_l2006_200665


namespace NUMINAMATH_CALUDE_largest_triangle_area_21_points_l2006_200626

/-- A configuration of points where every three adjacent points form an equilateral triangle --/
structure TriangleConfiguration where
  num_points : ℕ
  small_triangle_area : ℝ

/-- The area of the largest triangle formed by the configuration --/
def largest_triangle_area (config : TriangleConfiguration) : ℝ :=
  sorry

/-- Theorem stating that for a configuration of 21 points with unit area small triangles,
    the largest triangle has an area of 13 --/
theorem largest_triangle_area_21_points :
  let config : TriangleConfiguration := { num_points := 21, small_triangle_area := 1 }
  largest_triangle_area config = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_area_21_points_l2006_200626


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2006_200628

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 9) (h2 : x * y = 10) : x^3 + y^3 = 459 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2006_200628


namespace NUMINAMATH_CALUDE_book_arrangement_l2006_200676

theorem book_arrangement (n m : ℕ) (hn : n = 3) (hm : m = 4) :
  (Nat.choose (n + m) n) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l2006_200676


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2006_200630

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (4 * x - 7 * y = -20) ∧ 
    (9 * x + 3 * y = -21) ∧ 
    (x = -69/25) ∧ 
    (y = 32/25) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2006_200630


namespace NUMINAMATH_CALUDE_sqrt_529_squared_l2006_200643

theorem sqrt_529_squared : (Real.sqrt 529)^2 = 529 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_529_squared_l2006_200643


namespace NUMINAMATH_CALUDE_multiply_by_99999_l2006_200658

theorem multiply_by_99999 (x : ℝ) : x * 99999 = 58293485180 → x = 582.935 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_99999_l2006_200658


namespace NUMINAMATH_CALUDE_lcm_1640_1020_l2006_200618

theorem lcm_1640_1020 : Nat.lcm 1640 1020 = 83640 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1640_1020_l2006_200618


namespace NUMINAMATH_CALUDE_screen_area_difference_screen_area_difference_is_152_l2006_200620

/-- The difference in area between two square screens with diagonal lengths 21 and 17 inches -/
theorem screen_area_difference : Int :=
  let screen1_diagonal : Int := 21
  let screen2_diagonal : Int := 17
  let screen1_area : Int := screen1_diagonal ^ 2
  let screen2_area : Int := screen2_diagonal ^ 2
  screen1_area - screen2_area

/-- Proof that the difference in area is 152 square inches -/
theorem screen_area_difference_is_152 : screen_area_difference = 152 := by
  sorry

end NUMINAMATH_CALUDE_screen_area_difference_screen_area_difference_is_152_l2006_200620


namespace NUMINAMATH_CALUDE_Q_roots_nature_l2006_200685

def Q (x : ℝ) : ℝ := x^7 - 2*x^6 - 6*x^4 - 4*x + 16

theorem Q_roots_nature :
  (∀ x < 0, Q x > 0) ∧ 
  (∃ x > 0, Q x < 0) ∧ 
  (∃ x > 0, Q x > 0) :=
by sorry

end NUMINAMATH_CALUDE_Q_roots_nature_l2006_200685


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l2006_200683

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(x^2 + x ≤ -8) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l2006_200683


namespace NUMINAMATH_CALUDE_root_ratio_to_power_l2006_200679

theorem root_ratio_to_power (x : ℝ) (h : x > 0) :
  (x^(1/3)) / (x^(1/5)) = x^(2/15) :=
by sorry

end NUMINAMATH_CALUDE_root_ratio_to_power_l2006_200679


namespace NUMINAMATH_CALUDE_firefly_count_l2006_200641

/-- The number of fireflies remaining after a series of events --/
def remaining_fireflies (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the number of remaining fireflies in the given scenario --/
theorem firefly_count : remaining_fireflies 3 8 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_firefly_count_l2006_200641


namespace NUMINAMATH_CALUDE_boat_current_speed_l2006_200660

/-- Proves that given a boat with a speed of 18 km/hr in still water, traveling downstream
    for 14 minutes and covering a distance of 5.133333333333334 km, the rate of the current
    is 4 km/hr. -/
theorem boat_current_speed
  (boat_speed : ℝ)
  (travel_time : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 18)
  (h2 : travel_time = 14 / 60)
  (h3 : distance = 5.133333333333334) :
  let current_speed := (distance / travel_time) - boat_speed
  current_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_boat_current_speed_l2006_200660


namespace NUMINAMATH_CALUDE_fibonacci_matrix_power_fibonacci_determinant_l2006_200651

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fibonacci (n+1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ := 
  ![![fibonacci (n+1), fibonacci n],
    ![fibonacci n, fibonacci (n-1)]]

theorem fibonacci_matrix_power (n : ℕ) :
  (Matrix.of ![![1, 1], ![1, 0]] : Matrix (Fin 2) (Fin 2) ℕ) ^ n = fibonacci_matrix n := by
  sorry

theorem fibonacci_determinant (n : ℕ) :
  fibonacci (n+1) * fibonacci (n-1) - fibonacci n ^ 2 = (-1 : ℤ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_matrix_power_fibonacci_determinant_l2006_200651


namespace NUMINAMATH_CALUDE_yoongi_has_smallest_number_l2006_200606

def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem yoongi_has_smallest_number : 
  yoongi_number ≤ jungkook_number ∧ yoongi_number ≤ yuna_number :=
sorry

end NUMINAMATH_CALUDE_yoongi_has_smallest_number_l2006_200606


namespace NUMINAMATH_CALUDE_trolley_theorem_l2006_200666

def trolley_problem (X : ℕ) : Prop :=
  let initial_passengers := 10
  let second_stop_off := 3
  let second_stop_on := 2 * initial_passengers
  let third_stop_off := 18
  let third_stop_on := 2
  let fourth_stop_off := 5
  let fourth_stop_on := X
  let final_passengers := 
    initial_passengers - second_stop_off + second_stop_on - 
    third_stop_off + third_stop_on - fourth_stop_off + fourth_stop_on
  final_passengers = 6 + X

theorem trolley_theorem (X : ℕ) : 
  trolley_problem X :=
sorry

end NUMINAMATH_CALUDE_trolley_theorem_l2006_200666


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l2006_200623

/-- The number of ways to distribute n indistinguishable balls into 2 distinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ := n + 1

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 2 distinguishable boxes -/
theorem six_balls_two_boxes : distribute_balls 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l2006_200623


namespace NUMINAMATH_CALUDE_section_formula_vector_form_l2006_200697

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q⃗ = (5/8)C⃗ + (3/8)D⃗ --/
theorem section_formula_vector_form (C D Q : EuclideanSpace ℝ (Fin 3)) :
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) →  -- Q is on line segment CD
  (∃ s : ℝ, s > 0 ∧ dist C Q = (3 / (3 + 5)) * s ∧ dist Q D = (5 / (3 + 5)) * s) →  -- CQ:QD = 3:5
  Q = (5 / 8) • C + (3 / 8) • D :=
by sorry

end NUMINAMATH_CALUDE_section_formula_vector_form_l2006_200697


namespace NUMINAMATH_CALUDE_max_dimes_possible_l2006_200625

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The total amount in cents -/
def total_amount : ℕ := 550

/-- Theorem stating the maximum number of dimes possible -/
theorem max_dimes_possible (quarters nickels dimes : ℕ) 
  (h1 : quarters = nickels)
  (h2 : dimes ≥ 3 * quarters)
  (h3 : quarters * coin_value "quarter" + 
        nickels * coin_value "nickel" + 
        dimes * coin_value "dime" = total_amount) :
  dimes ≤ 28 :=
sorry

end NUMINAMATH_CALUDE_max_dimes_possible_l2006_200625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_equals_5_l2006_200650

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that in an arithmetic sequence, a_3 = 5 given the conditions -/
theorem arithmetic_sequence_a3_equals_5 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 3 + a 5 = 15) : 
  a 3 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_equals_5_l2006_200650


namespace NUMINAMATH_CALUDE_previous_largest_spider_weight_l2006_200682

/-- Proves the weight of the previous largest spider given the characteristics of a giant spider. -/
theorem previous_largest_spider_weight
  (weight_ratio : ℝ)
  (leg_count : ℕ)
  (leg_area : ℝ)
  (leg_pressure : ℝ)
  (h1 : weight_ratio = 2.5)
  (h2 : leg_count = 8)
  (h3 : leg_area = 0.5)
  (h4 : leg_pressure = 4) :
  let giant_spider_weight := leg_count * leg_area * leg_pressure
  giant_spider_weight / weight_ratio = 6.4 := by
sorry

end NUMINAMATH_CALUDE_previous_largest_spider_weight_l2006_200682


namespace NUMINAMATH_CALUDE_distance_for_given_point_l2006_200662

/-- The distance between a point and its symmetric point about the x-axis --/
def distance_to_symmetric_point (x y : ℝ) : ℝ := 2 * |y|

/-- Theorem: The distance between (2, -3) and its symmetric point about the x-axis is 6 --/
theorem distance_for_given_point : distance_to_symmetric_point 2 (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_given_point_l2006_200662


namespace NUMINAMATH_CALUDE_power_two_305_mod_9_l2006_200667

theorem power_two_305_mod_9 : 2^305 % 9 = 5 := by sorry

end NUMINAMATH_CALUDE_power_two_305_mod_9_l2006_200667


namespace NUMINAMATH_CALUDE_f_minimum_value_l2006_200699

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 2| + |x - a|

-- State the theorem
theorem f_minimum_value (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3) ↔ (a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2006_200699


namespace NUMINAMATH_CALUDE_inequality_for_positive_integers_l2006_200684

theorem inequality_for_positive_integers (n : ℕ+) :
  (n : ℝ)^(n : ℕ) ≤ ((n : ℕ).factorial : ℝ)^2 ∧ 
  ((n : ℕ).factorial : ℝ)^2 ≤ (((n + 1) * (n + 2) : ℝ) / 6)^(n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_integers_l2006_200684


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_four_l2006_200639

theorem same_solution_implies_k_equals_four (x k : ℝ) :
  (8 * x - k = 2 * (x + 1)) ∧ 
  (2 * (2 * x - 3) = 1 - 3 * x) ∧ 
  (∃ x, (8 * x - k = 2 * (x + 1)) ∧ (2 * (2 * x - 3) = 1 - 3 * x)) →
  k = 4 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_four_l2006_200639


namespace NUMINAMATH_CALUDE_language_courses_enrollment_l2006_200605

theorem language_courses_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (german_spanish : ℕ) (spanish_french : ℕ) (all_three : ℕ) :
  total = 180 →
  french = 60 →
  german = 50 →
  spanish = 35 →
  french_german = 20 →
  german_spanish = 15 →
  spanish_french = 10 →
  all_three = 5 →
  total - (french + german + spanish - french_german - german_spanish - spanish_french + all_three) = 80 := by
sorry

end NUMINAMATH_CALUDE_language_courses_enrollment_l2006_200605


namespace NUMINAMATH_CALUDE_inequality_proof_l2006_200607

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (b * c / a) + (a * c / b) + (a * b / c) > a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2006_200607


namespace NUMINAMATH_CALUDE_play_role_assignment_l2006_200663

theorem play_role_assignment (men : ℕ) (women : ℕ) : men = 7 ∧ women = 5 →
  (men * women * (Nat.choose (men + women - 2) 4)) = 7350 :=
by sorry

end NUMINAMATH_CALUDE_play_role_assignment_l2006_200663


namespace NUMINAMATH_CALUDE_ratio_equality_l2006_200615

theorem ratio_equality (x y z : ℝ) 
  (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (x + 4) / 2 = (x + 5) / (z - 5)) : 
  x / y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2006_200615


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l2006_200600

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed in a given interval -/
def changeObservationWindow (cycle : TrafficLightCycle) (interval : ℕ) : ℕ :=
  3 * interval  -- There are 3 color changes in a cycle

/-- Calculates the probability of observing a color change during a given interval -/
def probabilityOfChange (cycle : TrafficLightCycle) (interval : ℕ) : ℚ :=
  (changeObservationWindow cycle interval : ℚ) / (cycleDuration cycle : ℚ)

theorem traffic_light_change_probability :
  ∀ (cycle : TrafficLightCycle),
    cycle.green = 45 →
    cycle.yellow = 5 →
    cycle.red = 40 →
    probabilityOfChange cycle 4 = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l2006_200600


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l2006_200693

theorem factorization_of_difference_of_squares (a b : ℝ) : 
  -a^2 + 4*b^2 = (2*b + a) * (2*b - a) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l2006_200693


namespace NUMINAMATH_CALUDE_phillips_remaining_money_l2006_200624

/-- Calculates the remaining money after purchases --/
def remaining_money (initial : ℕ) (orange_cost apple_cost candy_cost : ℕ) : ℕ :=
  initial - (orange_cost + apple_cost + candy_cost)

/-- Theorem stating that given the specific amounts, the remaining money is $50 --/
theorem phillips_remaining_money :
  remaining_money 95 14 25 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_phillips_remaining_money_l2006_200624
