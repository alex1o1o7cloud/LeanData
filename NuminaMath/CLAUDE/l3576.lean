import Mathlib

namespace random_walk_properties_l3576_357662

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- Number of right steps
  b : ℕ  -- Number of left steps
  h : a > b

/-- The maximum possible range of the random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences achieving the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Main theorem about the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) : 
  (max_range w = w.a) ∧ 
  (min_range w = w.a - w.b) ∧ 
  (max_range_sequences w = w.b + 1) := by
  sorry


end random_walk_properties_l3576_357662


namespace reggie_bought_five_books_l3576_357657

/-- The number of books Reggie bought -/
def number_of_books (initial_amount remaining_amount cost_per_book : ℕ) : ℕ :=
  (initial_amount - remaining_amount) / cost_per_book

/-- Theorem: Reggie bought 5 books -/
theorem reggie_bought_five_books :
  number_of_books 48 38 2 = 5 := by
  sorry

end reggie_bought_five_books_l3576_357657


namespace overlap_area_is_half_unit_l3576_357693

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of overlap between two triangles on a 4x4 grid -/
def triangleOverlapArea (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the overlap area is 0.5 square units -/
theorem overlap_area_is_half_unit : 
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 3 2) (Point.mk 2 3)
  let t2 := Triangle.mk (Point.mk 0 3) (Point.mk 3 3) (Point.mk 3 0)
  triangleOverlapArea t1 t2 = 0.5 := by
  sorry

end overlap_area_is_half_unit_l3576_357693


namespace expand_polynomial_l3576_357628

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_polynomial_l3576_357628


namespace complex_modulus_problem_l3576_357647

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end complex_modulus_problem_l3576_357647


namespace hyperbola_center_is_3_6_l3576_357645

/-- The equation of a hyperbola in its general form -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 432 * y - 1017 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 6)

/-- Theorem: The center of the given hyperbola is (3, 6) -/
theorem hyperbola_center_is_3_6 :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ h k : ℝ, h = hyperbola_center.1 ∧ k = hyperbola_center.2 ∧
  ∀ t : ℝ, hyperbola_equation (t + h) (t + k) ↔ hyperbola_equation (t + x) (t + y) :=
by sorry

end hyperbola_center_is_3_6_l3576_357645


namespace derivative_equals_negative_function_l3576_357669

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- State the theorem
theorem derivative_equals_negative_function (x₀ : ℝ) :
  x₀ ≠ 0 → -- Ensure x₀ is not zero to avoid division by zero
  (deriv f) x₀ = -f x₀ →
  x₀ = 1/2 :=
by
  sorry


end derivative_equals_negative_function_l3576_357669


namespace expression_evaluation_l3576_357643

theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/2
  (2*a + b) * (2*a - b) + (3*a - b)^2 - ((12*a*b^2 - 16*a^2*b + 4*b) / (2*b)) = 104 :=
by sorry

end expression_evaluation_l3576_357643


namespace linear_equation_implies_m_eq_neg_three_l3576_357609

/-- Given that the equation (|m|-3)x^2 + (-m+3)x - 4 = 0 is linear in x with respect to m, prove that m = -3 -/
theorem linear_equation_implies_m_eq_neg_three (m : ℝ) 
  (h1 : ∀ x, (|m| - 3) * x^2 + (-m + 3) * x - 4 = 0 → (|m| - 3 = 0 ∧ -m + 3 ≠ 0)) : 
  m = -3 := by
  sorry

end linear_equation_implies_m_eq_neg_three_l3576_357609


namespace cubic_equation_root_l3576_357640

theorem cubic_equation_root (h : ℝ) : 
  (2 : ℝ)^3 + h * 2 + 10 = 0 → h = -9 := by
  sorry

end cubic_equation_root_l3576_357640


namespace average_equation_solution_l3576_357635

theorem average_equation_solution (x : ℝ) : 
  ((x + 3) + (4 * x + 1) + (3 * x + 6)) / 3 = 3 * x - 8 → x = 34 := by
sorry

end average_equation_solution_l3576_357635


namespace sandys_hourly_wage_l3576_357651

theorem sandys_hourly_wage (hours_friday hours_saturday hours_sunday : ℕ) 
  (total_earnings : ℕ) (hourly_wage : ℚ) :
  hours_friday = 10 →
  hours_saturday = 6 →
  hours_sunday = 14 →
  total_earnings = 450 →
  hourly_wage * (hours_friday + hours_saturday + hours_sunday) = total_earnings →
  hourly_wage = 15 := by
sorry

end sandys_hourly_wage_l3576_357651


namespace rectangle_to_square_cut_l3576_357659

theorem rectangle_to_square_cut (width : ℕ) (height : ℕ) : 
  width = 4 ∧ height = 9 → 
  ∃ (s : ℕ) (w1 w2 h1 h2 : ℕ),
    s * s = width * height ∧
    w1 + w2 = width ∧
    h1 = height ∧ h2 = height ∧
    (w1 * h1 + w2 * h2 = s * s) :=
by sorry

end rectangle_to_square_cut_l3576_357659


namespace sqrt_equation_solution_l3576_357602

theorem sqrt_equation_solution :
  ∃ t : ℝ, t = 3.7 ∧ Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) :=
by sorry

end sqrt_equation_solution_l3576_357602


namespace regression_and_probability_theorem_l3576_357652

/-- Data point representing year and sales volume -/
structure DataPoint where
  year : ℕ
  sales : ℕ

/-- Linear regression coefficients -/
structure RegressionCoefficients where
  b : ℚ
  a : ℚ

def data : List DataPoint := [
  ⟨1, 5⟩, ⟨2, 5⟩, ⟨3, 6⟩, ⟨4, 7⟩, ⟨5, 7⟩
]

def calculateRegressionCoefficients (data : List DataPoint) : RegressionCoefficients :=
  sorry

def probabilityConsecutiveYears (data : List DataPoint) : ℚ :=
  sorry

theorem regression_and_probability_theorem :
  let coeffs := calculateRegressionCoefficients data
  coeffs.b = 3/5 ∧ coeffs.a = 21/5 ∧ probabilityConsecutiveYears data = 2/5 := by
  sorry

#check regression_and_probability_theorem

end regression_and_probability_theorem_l3576_357652


namespace exponent_simplification_l3576_357636

theorem exponent_simplification :
  (5^6 * 5^9 * 5) / 5^3 = 5^13 := by sorry

end exponent_simplification_l3576_357636


namespace cos_alpha_sin_beta_range_l3576_357689

theorem cos_alpha_sin_beta_range (α β : Real) (h : Real.sin α * Real.cos β = -1/2) :
  ∃ (x : Real), Real.cos α * Real.sin β = x ∧ -1/2 ≤ x ∧ x ≤ 1/2 :=
sorry

end cos_alpha_sin_beta_range_l3576_357689


namespace papaya_problem_l3576_357682

/-- The number of fruits that turned yellow on Friday -/
def friday_yellow : ℕ := 2

theorem papaya_problem (initial_green : ℕ) (final_green : ℕ) :
  initial_green = 14 →
  final_green = 8 →
  initial_green - final_green = friday_yellow + 2 * friday_yellow →
  friday_yellow = 2 := by
  sorry

#check papaya_problem

end papaya_problem_l3576_357682


namespace collinear_dots_probability_l3576_357673

/-- Represents a 5x5 grid of dots -/
def Grid : Type := Unit

/-- The number of dots in the grid -/
def num_dots : ℕ := 25

/-- The number of ways to choose 4 collinear dots from the grid -/
def num_collinear_sets : ℕ := 54

/-- The total number of ways to choose 4 dots from the grid -/
def total_combinations : ℕ := 12650

/-- The probability of selecting 4 collinear dots when choosing 4 dots at random -/
def collinear_probability (g : Grid) : ℚ := 6 / 1415

theorem collinear_dots_probability (g : Grid) : 
  collinear_probability g = num_collinear_sets / total_combinations :=
by sorry

end collinear_dots_probability_l3576_357673


namespace ride_cost_is_factor_of_remaining_tickets_l3576_357604

def total_tickets : ℕ := 40
def spent_tickets : ℕ := 28
def remaining_tickets : ℕ := total_tickets - spent_tickets

def is_factor (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem ride_cost_is_factor_of_remaining_tickets :
  ∀ (num_rides cost_per_ride : ℕ),
    num_rides > 0 →
    cost_per_ride > 0 →
    num_rides * cost_per_ride = remaining_tickets →
    is_factor remaining_tickets cost_per_ride :=
by sorry

end ride_cost_is_factor_of_remaining_tickets_l3576_357604


namespace onion_bag_weight_l3576_357622

/-- Proves that the weight of each bag of onions is 50 kgs given the specified conditions -/
theorem onion_bag_weight 
  (bags_per_trip : ℕ) 
  (num_trips : ℕ) 
  (total_weight : ℕ) 
  (h1 : bags_per_trip = 10)
  (h2 : num_trips = 20)
  (h3 : total_weight = 10000) :
  total_weight / (bags_per_trip * num_trips) = 50 := by
  sorry

end onion_bag_weight_l3576_357622


namespace line_ellipse_intersection_slope_range_l3576_357625

/-- The slope range for a line intersecting an ellipse --/
theorem line_ellipse_intersection_slope_range :
  ∀ m : ℝ,
  (∃ x y : ℝ, y = m * x + 7 ∧ 4 * x^2 + 25 * y^2 = 100) →
  -Real.sqrt (9/5) ≤ m ∧ m ≤ Real.sqrt (9/5) := by
sorry

end line_ellipse_intersection_slope_range_l3576_357625


namespace intersection_when_a_is_two_union_equals_A_iff_l3576_357687

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1) - Real.sqrt (x + 2)}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3*x + a = 0}

-- Statement 1: When a = 2, A ∩ B = {2}
theorem intersection_when_a_is_two :
  A ∩ B 2 = {2} := by sorry

-- Statement 2: A ∪ B = A if and only if a ∈ (2, +∞)
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a > 2 := by sorry

end intersection_when_a_is_two_union_equals_A_iff_l3576_357687


namespace relative_prime_linear_forms_l3576_357627

theorem relative_prime_linear_forms (a b : ℤ) : 
  ∃ c d : ℤ, ∀ n : ℤ, Int.gcd (a * n + c) (b * n + d) = 1 := by
  sorry

end relative_prime_linear_forms_l3576_357627


namespace principal_is_800_l3576_357630

/-- Calculates the principal amount given the simple interest rate, final amount, and time period. -/
def calculate_principal (rate : ℚ) (final_amount : ℚ) (time : ℕ) : ℚ :=
  (final_amount * 100) / (rate * time)

/-- Theorem stating that the principal amount is 800 given the specified conditions. -/
theorem principal_is_800 (rate : ℚ) (final_amount : ℚ) (time : ℕ) 
  (h_rate : rate = 25/400)  -- 6.25% as a rational number
  (h_final_amount : final_amount = 200)
  (h_time : time = 4) :
  calculate_principal rate final_amount time = 800 := by
  sorry

#eval calculate_principal (25/400) 200 4  -- This should evaluate to 800

end principal_is_800_l3576_357630


namespace h_of_negative_one_l3576_357632

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x)^2 - 3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_negative_one : h (-1) = 298 := by
  sorry

end h_of_negative_one_l3576_357632


namespace mountain_dew_to_coke_ratio_l3576_357612

/-- Represents the composition of a drink -/
structure DrinkComposition where
  coke : ℝ
  sprite : ℝ
  mountainDew : ℝ

/-- Proves that the ratio of Mountain Dew to Coke is 3:2 given the conditions -/
theorem mountain_dew_to_coke_ratio 
  (drink : DrinkComposition)
  (coke_sprite_ratio : drink.coke = 2 * drink.sprite)
  (coke_amount : drink.coke = 6)
  (total_amount : drink.coke + drink.sprite + drink.mountainDew = 18) :
  drink.mountainDew / drink.coke = 3 / 2 := by
  sorry

end mountain_dew_to_coke_ratio_l3576_357612


namespace first_number_proof_l3576_357663

theorem first_number_proof (x : ℕ) : 
  (∃ k : ℤ, x = 2 * k + 7) ∧ 
  (∃ l : ℤ, 2037 = 2 * l + 5) ∧ 
  (∀ y : ℕ, y < x → ¬(∃ m : ℤ, y = 2 * m + 7)) → 
  x = 7 := by
sorry

end first_number_proof_l3576_357663


namespace saucers_per_pitcher_l3576_357638

/-- The weight of a cup -/
def cup_weight : ℝ := sorry

/-- The weight of a pitcher -/
def pitcher_weight : ℝ := sorry

/-- The weight of a saucer -/
def saucer_weight : ℝ := sorry

/-- Two cups and two pitchers weigh the same as 14 saucers -/
axiom weight_equation : 2 * cup_weight + 2 * pitcher_weight = 14 * saucer_weight

/-- One pitcher weighs the same as one cup and one saucer -/
axiom pitcher_cup_saucer : pitcher_weight = cup_weight + saucer_weight

/-- The number of saucers that balance with a pitcher is 4 -/
theorem saucers_per_pitcher : pitcher_weight = 4 * saucer_weight := by sorry

end saucers_per_pitcher_l3576_357638


namespace negation_of_proposition_negation_of_specific_proposition_l3576_357676

theorem negation_of_proposition (P : (ℝ → Prop)) :
  (¬ (∀ x : ℝ, x > 0 → P x)) ↔ (∃ x : ℝ, x > 0 ∧ ¬(P x)) :=
by sorry

-- Define the specific proposition
def Q (x : ℝ) : Prop := x^2 + 2*x - 3 ≥ 0

theorem negation_of_specific_proposition :
  (¬ (∀ x : ℝ, x > 0 → Q x)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + 2*x - 3 < 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l3576_357676


namespace sheila_hourly_wage_l3576_357623

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  full_days : Nat        -- Number of days working 8 hours
  partial_days : Nat     -- Number of days working 6 hours
  weekly_earnings : Nat  -- Total earnings per week in dollars

/-- Calculate Sheila's hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (8 * schedule.full_days + 6 * schedule.partial_days)

/-- Theorem: Sheila's hourly wage is $6 -/
theorem sheila_hourly_wage :
  let schedule : WorkSchedule := {
    full_days := 3,
    partial_days := 2,
    weekly_earnings := 216
  }
  hourly_wage schedule = 6 := by sorry

end sheila_hourly_wage_l3576_357623


namespace max_product_sum_l3576_357644

theorem max_product_sum (a b M : ℝ) : 
  a > 0 → b > 0 → (a + b = M) → (∀ x y : ℝ, x > 0 → y > 0 → x + y = M → x * y ≤ 2) → M = 2 * Real.sqrt 2 := by
  sorry

end max_product_sum_l3576_357644


namespace lychee_ratio_proof_l3576_357688

theorem lychee_ratio_proof (total : ℕ) (remaining : ℕ) : 
  total = 500 →
  remaining = 100 →
  ∃ (sold : ℕ) (taken_home : ℕ),
    sold + taken_home = total ∧
    (2 * remaining : ℚ) = (2 / 5 : ℚ) * taken_home ∧
    2 * sold = total :=
by sorry

end lychee_ratio_proof_l3576_357688


namespace mooney_ate_four_brownies_l3576_357624

/-- The number of brownies in a dozen -/
def dozen : ℕ := 12

/-- The number of brownies Mother initially made -/
def initial_brownies : ℕ := 2 * dozen

/-- The number of brownies Father ate -/
def father_ate : ℕ := 8

/-- The number of brownies Mother made the next morning -/
def new_batch : ℕ := 2 * dozen

/-- The total number of brownies after adding the new batch -/
def final_count : ℕ := 36

/-- Theorem: Mooney ate 4 brownies -/
theorem mooney_ate_four_brownies :
  initial_brownies - father_ate - (final_count - new_batch) = 4 := by
  sorry

end mooney_ate_four_brownies_l3576_357624


namespace zoo_feeding_sequences_l3576_357692

def number_of_animal_pairs : ℕ := 5

def alternating_feeding_sequences (n : ℕ) : ℕ :=
  (Nat.factorial n) * (Nat.factorial n)

theorem zoo_feeding_sequences :
  alternating_feeding_sequences number_of_animal_pairs = 14400 :=
by sorry

end zoo_feeding_sequences_l3576_357692


namespace bicycle_price_problem_l3576_357656

theorem bicycle_price_problem (profit_a_to_b : ℝ) (profit_b_to_c : ℝ) (final_price : ℝ) :
  profit_a_to_b = 0.25 →
  profit_b_to_c = 0.5 →
  final_price = 225 →
  ∃ (cost_price_a : ℝ),
    cost_price_a * (1 + profit_a_to_b) * (1 + profit_b_to_c) = final_price ∧
    cost_price_a = 120 :=
by sorry

end bicycle_price_problem_l3576_357656


namespace b_work_time_l3576_357683

/-- Represents the time in days it takes for a person to complete a task alone. -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents the rate at which a person completes a task, as a fraction of the task per day. -/
def workRate (wt : WorkTime) : ℚ := 1 / wt.days

/-- The combined work rate of multiple people working together. -/
def combinedWorkRate (rates : List ℚ) : ℚ := rates.sum

theorem b_work_time (a_time : WorkTime) (c_time : WorkTime) (abc_time : WorkTime) 
  (ha : a_time.days = 8)
  (hc : c_time.days = 24)
  (habc : abc_time.days = 4) :
  ∃ (b_time : WorkTime), b_time.days = 12 := by
  sorry

end b_work_time_l3576_357683


namespace max_leap_years_in_period_l3576_357639

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leapYearFrequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def maxLeapYears : ℕ := period / leapYearFrequency

/-- Theorem: The maximum number of leap years in a 200-year period
    with leap years occurring every five years is 40 -/
theorem max_leap_years_in_period :
  maxLeapYears = 40 := by sorry

end max_leap_years_in_period_l3576_357639


namespace sphere_radius_equals_eight_l3576_357655

-- Define constants for the cylinder dimensions
def cylinder_height : ℝ := 16
def cylinder_diameter : ℝ := 16

-- Define the theorem
theorem sphere_radius_equals_eight :
  ∀ r : ℝ,
  (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) →
  r = 8 := by
  sorry


end sphere_radius_equals_eight_l3576_357655


namespace ordinary_day_probability_l3576_357690

/-- Probability of shark appearance on any given day -/
def P_shark_appearance : ℚ := 1 / 30

/-- Probability of system detecting a shark when present -/
def P_detection_given_shark : ℚ := 3 / 4

/-- Probability of false alarm given no shark -/
def P_false_alarm_given_no_shark : ℚ := 10 * P_shark_appearance

/-- Theorem: The probability of an "ordinary" day (no sharks and no false alarms) is 29/45 -/
theorem ordinary_day_probability : 
  let P_no_shark : ℚ := 1 - P_shark_appearance
  let P_no_alarm_given_no_shark : ℚ := 1 - P_false_alarm_given_no_shark
  P_no_shark * P_no_alarm_given_no_shark = 29 / 45 := by
  sorry

end ordinary_day_probability_l3576_357690


namespace mary_baking_cake_l3576_357684

/-- Given a recipe that requires a certain amount of flour and an amount already added,
    calculate the remaining amount to be added. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Prove that for a recipe requiring 7 cups of flour, with 2 cups already added,
    the remaining amount to be added is 5 cups. -/
theorem mary_baking_cake :
  remaining_flour 7 2 = 5 := by
  sorry

end mary_baking_cake_l3576_357684


namespace circle_tangent_triangle_division_l3576_357614

/-- Given a triangle with sides a, b, and c, where c is the longest side,
    and a circle touching sides a and b with its center on side c,
    prove that the center divides c into segments of length x and y. -/
theorem circle_tangent_triangle_division (a b c x y : ℝ) : 
  a = 12 → b = 15 → c = 18 → c > a ∧ c > b →
  x + y = c → x / y = a / b →
  x = 8 ∧ y = 10 := by sorry

end circle_tangent_triangle_division_l3576_357614


namespace smallest_value_of_quadratic_l3576_357664

theorem smallest_value_of_quadratic :
  (∀ x : ℝ, x^2 + 6*x + 9 ≥ 0) ∧ (∃ x : ℝ, x^2 + 6*x + 9 = 0) := by
  sorry

end smallest_value_of_quadratic_l3576_357664


namespace comparison_of_trigonometric_expressions_l3576_357629

theorem comparison_of_trigonometric_expressions :
  let a := (1/2) * Real.cos (4 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * π / 180)
  let b := Real.cos (13 * π / 180)^2 - Real.sin (13 * π / 180)^2
  let c := (2 * Real.tan (23 * π / 180)) / (1 - Real.tan (23 * π / 180)^2)
  a < b ∧ b < c := by sorry

end comparison_of_trigonometric_expressions_l3576_357629


namespace family_race_problem_l3576_357618

/-- Represents the driving data for the family race -/
structure DrivingData where
  cory_time : ℝ
  cory_speed : ℝ
  mira_time : ℝ
  mira_speed : ℝ
  tia_time : ℝ
  tia_speed : ℝ

/-- The theorem statement for the family race problem -/
theorem family_race_problem (data : DrivingData) 
  (h1 : data.mira_time = data.cory_time + 3)
  (h2 : data.mira_speed = data.cory_speed + 8)
  (h3 : data.mira_speed * data.mira_time = data.cory_speed * data.cory_time + 120)
  (h4 : data.tia_time = data.cory_time + 4)
  (h5 : data.tia_speed = data.cory_speed + 12) :
  data.tia_speed * data.tia_time - data.cory_speed * data.cory_time = 192 := by
  sorry

end family_race_problem_l3576_357618


namespace factory_produces_4000_candies_l3576_357668

/-- Represents a candy factory with its production rate and work schedule. -/
structure CandyFactory where
  production_rate : ℕ  -- candies per hour
  work_hours_per_day : ℕ
  work_days : ℕ

/-- Calculates the total number of candies produced by a factory. -/
def total_candies_produced (factory : CandyFactory) : ℕ :=
  factory.production_rate * factory.work_hours_per_day * factory.work_days

/-- Theorem stating that a factory with the given parameters produces 4000 candies. -/
theorem factory_produces_4000_candies 
  (factory : CandyFactory) 
  (h1 : factory.production_rate = 50)
  (h2 : factory.work_hours_per_day = 10)
  (h3 : factory.work_days = 8) : 
  total_candies_produced factory = 4000 := by
  sorry

#eval total_candies_produced { production_rate := 50, work_hours_per_day := 10, work_days := 8 }

end factory_produces_4000_candies_l3576_357668


namespace macaron_ratio_l3576_357654

theorem macaron_ratio (mitch joshua miles renz : ℕ) : 
  mitch = 20 →
  joshua = mitch + 6 →
  (∃ k : ℚ, joshua = k * miles) →
  renz = (3 * miles) / 4 - 1 →
  mitch + joshua + miles + renz = 68 * 2 →
  joshua * 2 = miles * 1 := by
  sorry

end macaron_ratio_l3576_357654


namespace abs_sum_minimum_l3576_357600

theorem abs_sum_minimum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end abs_sum_minimum_l3576_357600


namespace arithmetic_geometric_sum_l3576_357646

/-- An arithmetic sequence with first term 2 and last term 3 -/
def is_arithmetic_sequence (x y : ℝ) : Prop :=
  x - 2 = 3 - y ∧ y - x = 3 - y

/-- A geometric sequence with first term 2 and last term 3 -/
def is_geometric_sequence (m n : ℝ) : Prop :=
  m / 2 = 3 / n ∧ n / m = 3 / n

theorem arithmetic_geometric_sum (x y m n : ℝ) 
  (h1 : is_arithmetic_sequence x y) 
  (h2 : is_geometric_sequence m n) : 
  x + y + m * n = 11 := by
  sorry

end arithmetic_geometric_sum_l3576_357646


namespace seventh_term_of_geometric_sequence_l3576_357666

/-- Given a geometric sequence with first term √3 and second term 3√3, 
    the seventh term is 729√3 -/
theorem seventh_term_of_geometric_sequence 
  (a₁ : ℝ) 
  (a₂ : ℝ) 
  (h₁ : a₁ = Real.sqrt 3)
  (h₂ : a₂ = 3 * Real.sqrt 3) :
  (a₁ * (a₂ / a₁)^6 : ℝ) = 729 * Real.sqrt 3 := by
  sorry

#check seventh_term_of_geometric_sequence

end seventh_term_of_geometric_sequence_l3576_357666


namespace quadratic_root_l3576_357681

theorem quadratic_root (b : ℝ) : 
  (1 : ℝ) ^ 2 + b * 1 + 2 = 0 → ∃ x : ℝ, x ≠ 1 ∧ x ^ 2 + b * x + 2 = 0 ∧ x = 2 := by
  sorry

end quadratic_root_l3576_357681


namespace rectangular_garden_width_l3576_357621

theorem rectangular_garden_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end rectangular_garden_width_l3576_357621


namespace power_of_power_three_l3576_357675

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_l3576_357675


namespace cube_of_square_of_third_smallest_prime_l3576_357678

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l3576_357678


namespace fraction_equality_l3576_357633

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) :
  (x + y) / (x - y) = -1001 := by sorry

end fraction_equality_l3576_357633


namespace margaret_mean_score_l3576_357619

def scores : List ℕ := [84, 86, 90, 92, 93, 95, 97, 96, 99]

def cyprian_count : ℕ := 5
def margaret_count : ℕ := 4
def cyprian_mean : ℕ := 92

theorem margaret_mean_score :
  let total_sum := scores.sum
  let cyprian_sum := cyprian_count * cyprian_mean
  let margaret_sum := total_sum - cyprian_sum
  (margaret_sum : ℚ) / margaret_count = 93 := by sorry

end margaret_mean_score_l3576_357619


namespace diamond_inequality_l3576_357637

def diamond (x y : ℝ) : ℝ := |x^2 - y^2|

theorem diamond_inequality : ∃ x y : ℝ, diamond (x + y) (x - y) ≠ diamond x y := by
  sorry

end diamond_inequality_l3576_357637


namespace min_value_at_angle_l3576_357653

def minimizing_angle (k : ℤ) : ℝ := 660 + 720 * k

theorem min_value_at_angle (A : ℝ) :
  (∃ k : ℤ, A = minimizing_angle k) ↔
  ∀ B : ℝ, Real.sin (A / 2) - Real.sqrt 3 * Real.cos (A / 2) ≤ 
           Real.sin (B / 2) - Real.sqrt 3 * Real.cos (B / 2) :=
by sorry

#check min_value_at_angle

end min_value_at_angle_l3576_357653


namespace joans_remaining_practice_time_l3576_357679

/-- Given Joan's music practice schedule, calculate the remaining time for finger exercises. -/
theorem joans_remaining_practice_time :
  let total_time : ℕ := 2 * 60  -- 2 hours in minutes
  let piano_time : ℕ := 30
  let writing_time : ℕ := 25
  let reading_time : ℕ := 38
  let used_time : ℕ := piano_time + writing_time + reading_time
  total_time - used_time = 27 := by
  sorry

#check joans_remaining_practice_time

end joans_remaining_practice_time_l3576_357679


namespace bottle_filling_proportion_l3576_357698

/-- Given two bottles with capacities of 4 and 8 cups, and a total of 8 cups of milk,
    prove that the proportion of capacity each bottle should be filled to is 2/3,
    when the 8-cup bottle contains 5.333333333333333 cups of milk. -/
theorem bottle_filling_proportion :
  let total_milk : ℚ := 8
  let bottle1_capacity : ℚ := 4
  let bottle2_capacity : ℚ := 8
  let milk_in_bottle2 : ℚ := 5.333333333333333
  let proportion : ℚ := milk_in_bottle2 / bottle2_capacity
  proportion = 2/3 ∧ 
  bottle1_capacity * proportion + bottle2_capacity * proportion = total_milk :=
by sorry

end bottle_filling_proportion_l3576_357698


namespace product_of_three_integers_l3576_357607

theorem product_of_three_integers (A B C : ℤ) 
  (sum_eq : A + B + C = 33)
  (largest_eq : C = 3 * B)
  (smallest_eq : A = C - 23) :
  A * B * C = 192 := by
sorry

end product_of_three_integers_l3576_357607


namespace x_squared_in_set_l3576_357686

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end x_squared_in_set_l3576_357686


namespace waiter_customers_l3576_357613

/-- Calculates the final number of customers for a waiter given the initial number,
    the number who left, and the number of new customers. -/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that for the given scenario, the final number of customers is 28. -/
theorem waiter_customers : final_customers 33 31 26 = 28 := by
  sorry

end waiter_customers_l3576_357613


namespace log_sum_simplification_l3576_357650

theorem log_sum_simplification :
  1 / (Real.log 2 / Real.log 15 + 1) + 
  1 / (Real.log 3 / Real.log 10 + 1) + 
  1 / (Real.log 5 / Real.log 6 + 1) = 2 :=
by sorry

end log_sum_simplification_l3576_357650


namespace shoes_theorem_l3576_357696

/-- Given an initial number of shoe pairs and a number of lost individual shoes,
    calculate the maximum number of complete pairs remaining. -/
def maxRemainingPairs (initialPairs : ℕ) (lostShoes : ℕ) : ℕ :=
  initialPairs - lostShoes

/-- Theorem: Given 26 initial pairs of shoes and losing 9 individual shoes,
    the maximum number of complete pairs remaining is 17. -/
theorem shoes_theorem :
  maxRemainingPairs 26 9 = 17 := by
  sorry

#eval maxRemainingPairs 26 9

end shoes_theorem_l3576_357696


namespace S_equality_l3576_357606

/-- S_k(n) function (not defined, assumed to exist) -/
noncomputable def S_k (k n : ℕ) : ℕ := sorry

/-- The sum S as defined in the problem -/
noncomputable def S (n k : ℕ) : ℚ :=
  (Finset.range ((k + 1) / 2)).sum (λ i =>
    Nat.choose (k + 1) (2 * i + 1) * S_k (k - 2 * i) n)

/-- Theorem stating the equality to be proved -/
theorem S_equality (n k : ℕ) :
  S n k = ((n + 1)^(k + 1) + n^(k + 1) - 1) / 2 := by sorry

end S_equality_l3576_357606


namespace system_solution_l3576_357605

theorem system_solution (x y : ℝ) (hx : x = 4) (hy : y = -1) : x - 2*y = 6 := by
  sorry

end system_solution_l3576_357605


namespace max_rectangles_equals_black_squares_l3576_357697

/-- Represents a figure that can be cut into squares and rectangles -/
structure Figure where
  shape : Set (ℕ × ℕ)  -- Set of coordinates representing the shape

/-- Counts the number of black squares when coloring the middle diagonal -/
def count_black_squares (f : Figure) : ℕ :=
  sorry

/-- Represents the specific figure given in the problem -/
def given_figure : Figure :=
  { shape := sorry }

/-- The maximum number of 1×2 rectangles that can be obtained -/
def max_rectangles (f : Figure) : ℕ :=
  sorry

theorem max_rectangles_equals_black_squares :
  max_rectangles given_figure = count_black_squares given_figure ∧
  count_black_squares given_figure = 5 := by
  sorry

end max_rectangles_equals_black_squares_l3576_357697


namespace all_graphs_different_l3576_357610

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x - 1
def eq2 (x y : ℝ) : Prop := y = (x^2 - 1) / (x + 1)
def eq3 (x y : ℝ) : Prop := (x + 1) * y = x^2 - 1

-- Define what it means for two equations to have the same graph
def same_graph (eq_a eq_b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq_a x y ↔ eq_b x y

-- Theorem stating that all equations have different graphs
theorem all_graphs_different :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end all_graphs_different_l3576_357610


namespace mirror_area_l3576_357671

/-- Given a rectangular mirror centered within two frames, where the outermost frame measures
    100 cm by 140 cm, and both frames have a width of 15 cm on each side, the area of the mirror
    is 3200 cm². -/
theorem mirror_area (outer_length outer_width frame_width : ℕ) 
  (h1 : outer_length = 100)
  (h2 : outer_width = 140)
  (h3 : frame_width = 15) : 
  (outer_length - 2 * frame_width - 2 * frame_width) * 
  (outer_width - 2 * frame_width - 2 * frame_width) = 3200 :=
by sorry

end mirror_area_l3576_357671


namespace tammy_running_schedule_l3576_357660

/-- Calculates the number of loops per day given weekly distance goal, track length, and days per week -/
def loops_per_day (weekly_goal : ℕ) (track_length : ℕ) (days_per_week : ℕ) : ℕ :=
  (weekly_goal / track_length) / days_per_week

theorem tammy_running_schedule :
  loops_per_day 3500 50 7 = 10 := by
  sorry

end tammy_running_schedule_l3576_357660


namespace pentagon_to_squares_ratio_l3576_357677

-- Define the square structure
structure Square :=
  (side : ℝ)

-- Define the pentagon structure
structure Pentagon :=
  (area : ℝ)

-- Define the theorem
theorem pentagon_to_squares_ratio 
  (s : Square) 
  (p : Pentagon) 
  (h1 : s.side > 0)
  (h2 : p.area = s.side * s.side)
  : p.area / (3 * s.side * s.side) = 1 / 3 :=
sorry

end pentagon_to_squares_ratio_l3576_357677


namespace min_a_value_l3576_357620

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by sorry

end min_a_value_l3576_357620


namespace jessie_current_weight_l3576_357674

def jessie_weight_problem (initial_weight lost_weight : ℕ) : Prop :=
  initial_weight = 69 ∧ lost_weight = 35 →
  initial_weight - lost_weight = 34

theorem jessie_current_weight : jessie_weight_problem 69 35 := by
  sorry

end jessie_current_weight_l3576_357674


namespace function_period_l3576_357603

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_period (f : ℝ → ℝ) (h : ∀ x, f (x + 3) = -f x) :
  is_periodic f 6 :=
sorry

end function_period_l3576_357603


namespace additional_toothpicks_for_extension_l3576_357626

/-- The number of toothpicks required for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n ≤ 2 then 2 * n + 2 else 2 * n + (n - 1) * (n - 2)

theorem additional_toothpicks_for_extension :
  toothpicks 4 = 26 →
  toothpicks 6 - toothpicks 4 = 22 := by
  sorry

end additional_toothpicks_for_extension_l3576_357626


namespace pentagon_diagonals_l3576_357695

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The number of diagonals in a pentagon is 5 -/
theorem pentagon_diagonals : num_diagonals pentagon_sides = 5 := by
  sorry

end pentagon_diagonals_l3576_357695


namespace daughter_age_in_three_years_l3576_357680

/-- Given that 5 years ago a mother was twice as old as her daughter,
    and the mother is 41 years old now, prove that the daughter
    will be 26 years old in 3 years. -/
theorem daughter_age_in_three_years
  (mother_age_now : ℕ)
  (h1 : mother_age_now = 41)
  (h2 : mother_age_now - 5 = 2 * ((mother_age_now - 5) / 2)) :
  ((mother_age_now - 5) / 2) + 8 = 26 := by
  sorry

end daughter_age_in_three_years_l3576_357680


namespace expression_evaluation_l3576_357601

theorem expression_evaluation :
  (5^1003 + 6^1004)^2 - (5^1003 - 6^1004)^2 = 24 * 30^1003 :=
by sorry

end expression_evaluation_l3576_357601


namespace arithmetic_mean_of_first_four_primes_reciprocals_l3576_357670

-- Define the first four prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define a function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (lst : List Nat) : ℚ :=
  (lst.map (λ x => (1 : ℚ) / x)).sum / lst.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_primes_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end arithmetic_mean_of_first_four_primes_reciprocals_l3576_357670


namespace disinfectant_sales_problem_l3576_357661

-- Define the range of x
def valid_x (x : ℤ) : Prop := 8 ≤ x ∧ x ≤ 15

-- Define the linear function
def y (x : ℤ) : ℤ := -5 * x + 150

-- Define the profit function
def w (x : ℤ) : ℤ := (x - 8) * (-5 * x + 150)

theorem disinfectant_sales_problem :
  (∀ x : ℤ, valid_x x → 
    (x = 9 → y x = 105) ∧ 
    (x = 11 → y x = 95) ∧ 
    (x = 13 → y x = 85)) ∧
  (∃ x : ℤ, valid_x x ∧ w x = 425 ∧ x = 13) ∧
  (∃ x : ℤ, valid_x x ∧ w x = 525 ∧ x = 15 ∧ ∀ x' : ℤ, valid_x x' → w x' ≤ w x) :=
by sorry

end disinfectant_sales_problem_l3576_357661


namespace fixed_point_power_function_l3576_357691

theorem fixed_point_power_function (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 / 2 →
  f 9 = 1 / 3 := by
  sorry

end fixed_point_power_function_l3576_357691


namespace zeros_sum_inequality_l3576_357649

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * log x

theorem zeros_sum_inequality (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  f (1 / exp 2) x₁ = 0 → f (1 / exp 2) x₂ = 0 →
  log (x₁ + x₂) > log 2 + 1 := by
  sorry

end zeros_sum_inequality_l3576_357649


namespace non_intersecting_probability_is_two_thirds_l3576_357699

/-- Two persons start from opposite corners of a rectangular grid and can only move up or right one step at a time. -/
structure GridWalk where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- The probability that the routes of two persons do not intersect -/
def non_intersecting_probability (g : GridWalk) : ℚ :=
  2/3

/-- Theorem stating that the probability of non-intersecting routes is 2/3 -/
theorem non_intersecting_probability_is_two_thirds (g : GridWalk) :
  non_intersecting_probability g = 2/3 := by
  sorry

end non_intersecting_probability_is_two_thirds_l3576_357699


namespace probability_all_co_captains_value_l3576_357685

def team_sizes : List Nat := [6, 9, 10]
def co_captains_per_team : Nat := 3
def num_teams : Nat := 3

def probability_all_co_captains : ℚ :=
  (1 : ℚ) / num_teams *
  (team_sizes.map (λ n => (co_captains_per_team : ℚ) / (n * (n - 1) * (n - 2)))).sum

theorem probability_all_co_captains_value :
  probability_all_co_captains = 59 / 2520 := by
  sorry

#eval probability_all_co_captains

end probability_all_co_captains_value_l3576_357685


namespace integer_solutions_count_l3576_357611

theorem integer_solutions_count (m : ℤ) : 
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x ∈ s, x - m < 0 ∧ 5 - 2*x ≤ 1) ↔ m = 4 := by
  sorry

end integer_solutions_count_l3576_357611


namespace completing_square_quadratic_equation_l3576_357617

theorem completing_square_quadratic_equation :
  ∀ x : ℝ, x^2 + 4*x + 3 = 0 ↔ (x + 2)^2 = 1 := by
sorry

end completing_square_quadratic_equation_l3576_357617


namespace sports_club_overlap_l3576_357634

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : tennis = 18)
  (h4 : neither = 5)
  (h5 : badminton + tennis - (badminton + tennis - total + neither) = total - neither) :
  badminton + tennis - total + neither = 3 := by
  sorry

end sports_club_overlap_l3576_357634


namespace hyperbola_line_intersection_property_l3576_357648

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a straight line -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a point lies on a line -/
def on_line (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.c

/-- Checks if a point lies on an asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

/-- The main theorem -/
theorem hyperbola_line_intersection_property
  (h : Hyperbola) (l : Line)
  (p q p' q' : Point)
  (hp : on_hyperbola h p)
  (hq : on_hyperbola h q)
  (hp' : on_asymptote h p')
  (hq' : on_asymptote h q')
  (hlp : on_line l p)
  (hlq : on_line l q)
  (hlp' : on_line l p')
  (hlq' : on_line l q') :
  |p.x - p'.x| = |q.x - q'.x| ∧ |p.y - p'.y| = |q.y - q'.y| :=
sorry

end hyperbola_line_intersection_property_l3576_357648


namespace equal_domain_function_iff_a_range_l3576_357608

/-- A function that maps a set onto itself --/
def EqualDomainFunction (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ x ∈ A, f x ∈ A ∧ ∀ y ∈ A, ∃ x ∈ A, f x = y

/-- The quadratic function f(x) = a(x-1)^2 - 2 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - 2

theorem equal_domain_function_iff_a_range :
  ∀ a < 0, (∃ m n : ℝ, m < n ∧ EqualDomainFunction (f a) (Set.Icc m n)) ↔ -1/12 < a ∧ a < 0 := by
  sorry

#check equal_domain_function_iff_a_range

end equal_domain_function_iff_a_range_l3576_357608


namespace prob_king_then_ten_l3576_357665

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of 10s in a standard deck -/
def TensInDeck : ℕ := 4

/-- Probability of drawing a King first and then a 10 from a standard deck -/
theorem prob_king_then_ten (deck : ℕ) (kings : ℕ) (tens : ℕ) :
  deck = StandardDeck → kings = KingsInDeck → tens = TensInDeck →
  (kings : ℚ) / deck * tens / (deck - 1) = 4 / 663 := by
  sorry

end prob_king_then_ten_l3576_357665


namespace three_digit_subtraction_l3576_357616

/-- Represents a three-digit number with digits a, b, c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

theorem three_digit_subtraction
  (n₁ n₂ : ThreeDigitNumber)
  (h_reverse : n₂.a = n₁.c ∧ n₂.b = n₁.b ∧ n₂.c = n₁.a)
  (h_result_units : (n₁.toNat - n₂.toNat) % 10 = 2)
  (h_result_tens : ((n₁.toNat - n₂.toNat) / 10) % 10 = 9)
  (h_borrow : n₁.c < n₂.c) :
  n₁.a = 9 ∧ n₁.b = 9 ∧ n₁.c = 1 := by
  sorry

end three_digit_subtraction_l3576_357616


namespace freshmen_in_liberal_arts_l3576_357642

theorem freshmen_in_liberal_arts (total_students : ℝ) 
  (freshmen_percent : ℝ) 
  (psych_majors_percent : ℝ) 
  (freshmen_psych_liberal_arts_percent : ℝ) :
  freshmen_percent = 0.6 →
  psych_majors_percent = 0.2 →
  freshmen_psych_liberal_arts_percent = 0.048 →
  (freshmen_psych_liberal_arts_percent * total_students) / 
  (psych_majors_percent * freshmen_percent * total_students) = 0.4 := by
sorry

end freshmen_in_liberal_arts_l3576_357642


namespace complex_root_of_unity_sum_l3576_357631

theorem complex_root_of_unity_sum (ω : ℂ) : 
  ω = -1/2 + (Complex.I * Real.sqrt 3) / 2 → ω^4 + ω^2 + 1 = 0 := by
  sorry

end complex_root_of_unity_sum_l3576_357631


namespace intersection_distance_l3576_357672

/-- The distance between intersections of x = y³ and x + y² = 1 -/
theorem intersection_distance (a : ℝ) : 
  (a^4 + a^3 + a^2 - 1 = 0) →
  ∃ (u v p : ℝ), 
    (2 * Real.sqrt (a^6 + a^2) = Real.sqrt (u + v * Real.sqrt p)) ∧
    ((a^3)^2 + a^2)^2 = (((-a)^3)^2 + (-a)^2)^2 := by
  sorry

end intersection_distance_l3576_357672


namespace depak_money_problem_l3576_357615

theorem depak_money_problem :
  ∀ x : ℕ, 
    (x + 1) % 6 = 0 ∧ 
    x % 6 ≠ 0 ∧
    ∀ y : ℕ, y > x → (y + 1) % 6 ≠ 0 ∨ y % 6 = 0
    → x = 5 := by
  sorry

end depak_money_problem_l3576_357615


namespace z_in_first_quadrant_l3576_357658

def complex_condition (z : ℂ) : Prop := z * (1 + Complex.I) = 2 * Complex.I

theorem z_in_first_quadrant (z : ℂ) (h : complex_condition z) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end z_in_first_quadrant_l3576_357658


namespace quadratic_roots_problem_l3576_357641

theorem quadratic_roots_problem (α β b : ℝ) : 
  (∀ x, x^2 + b*x - 1 = 0 ↔ x = α ∨ x = β) →
  α * β - 2*α - 2*β = -11 →
  b = -5 := by
sorry

end quadratic_roots_problem_l3576_357641


namespace binary_21_l3576_357694

/-- The binary representation of a natural number. -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Proposition: The binary representation of 21 is [true, false, true, false, true] -/
theorem binary_21 : toBinary 21 = [true, false, true, false, true] := by
  sorry

end binary_21_l3576_357694


namespace factorial_equation_l3576_357667

theorem factorial_equation : (Nat.factorial 6 - Nat.factorial 4) / Nat.factorial 5 = 29/5 := by
  sorry

end factorial_equation_l3576_357667
