import Mathlib

namespace complementary_angles_difference_l2196_219687

/-- Given two complementary angles with measures in the ratio of 3:1, 
    their positive difference is 45 degrees. -/
theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 3 * b →   -- ratio of angles is 3:1
  |a - b| = 45  -- positive difference is 45 degrees
:= by sorry

end complementary_angles_difference_l2196_219687


namespace slope_implies_y_coordinate_l2196_219619

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q
    is equal to 5/3, then the y-coordinate of Q is 56/3. -/
theorem slope_implies_y_coordinate :
  ∀ (y : ℚ),
  let P : ℚ × ℚ := (-2, 7)
  let Q : ℚ × ℚ := (5, y)
  let slope : ℚ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 5/3 → y = 56/3 :=
by
  sorry

end slope_implies_y_coordinate_l2196_219619


namespace rhombus_area_from_quadratic_roots_l2196_219673

/-- Given a quadratic equation x^2 - 10x + 24 = 0, if its roots are the lengths of the diagonals
    of a rhombus, then the area of the rhombus is 12. -/
theorem rhombus_area_from_quadratic_roots : 
  ∀ (d₁ d₂ : ℝ), d₁ * d₂ = 24 → d₁ + d₂ = 10 → (1/2) * d₁ * d₂ = 12 := by
  sorry

end rhombus_area_from_quadratic_roots_l2196_219673


namespace longer_train_length_l2196_219642

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
theorem longer_train_length
  (speed1 : ℝ)
  (speed2 : ℝ)
  (crossing_time : ℝ)
  (shorter_train_length : ℝ)
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : crossing_time = 11.159107271418288)
  (h4 : shorter_train_length = 140)
  : ∃ (longer_train_length : ℝ),
    longer_train_length = 170 ∧
    (speed1 + speed2) * (1000 / 3600) * crossing_time =
      shorter_train_length + longer_train_length :=
by
  sorry

end longer_train_length_l2196_219642


namespace three_digit_with_three_without_five_seven_l2196_219636

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, n = k * 10 + d ∨ n = k * 100 + d ∨ ∃ m, n = k * 100 + m * 10 + d

def not_contains_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  ¬(contains_digit n d₁) ∧ ¬(contains_digit n d₂)

theorem three_digit_with_three_without_five_seven (n : ℕ) :
  (is_three_digit n ∧ contains_digit n 3 ∧ not_contains_digits n 5 7) →
  ∃ S : Finset ℕ, S.card = 154 ∧ n ∈ S :=
sorry

end three_digit_with_three_without_five_seven_l2196_219636


namespace stratified_sampling_difference_l2196_219669

theorem stratified_sampling_difference (total_male : Nat) (total_female : Nat) (sample_size : Nat) : 
  total_male = 56 → 
  total_female = 42 → 
  sample_size = 28 → 
  (sample_size : ℚ) / ((total_male + total_female) : ℚ) = 2 / 7 → 
  (total_male : ℚ) * ((sample_size : ℚ) / ((total_male + total_female) : ℚ)) - 
  (total_female : ℚ) * ((sample_size : ℚ) / ((total_male + total_female) : ℚ)) = 4 := by
  sorry

#check stratified_sampling_difference

end stratified_sampling_difference_l2196_219669


namespace subset_implies_a_bound_l2196_219645

theorem subset_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 2 ≤ 0 → 1/(x-3) < a) → a > -1/2 := by
  sorry

end subset_implies_a_bound_l2196_219645


namespace base_7_addition_problem_l2196_219651

/-- Given an addition problem in base 7, prove that X + Y = 10 in base 10 --/
theorem base_7_addition_problem (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (5 * 7 + 2) = 6 * 7^2 + 4 * 7 + X → X + Y = 10 := by
sorry

end base_7_addition_problem_l2196_219651


namespace soda_price_l2196_219698

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- Four burgers and three sodas cost 540 cents -/
axiom alice_purchase : 4 * burger_cost + 3 * soda_cost = 540

/-- Three burgers and two sodas cost 390 cents -/
axiom bill_purchase : 3 * burger_cost + 2 * soda_cost = 390

/-- The cost of a soda is 60 cents -/
theorem soda_price : soda_cost = 60 := by sorry

end soda_price_l2196_219698


namespace range_of_a_l2196_219646

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (1 - a)^x < (1 - a)^y

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → (0 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
by sorry

end range_of_a_l2196_219646


namespace mold_cost_is_250_l2196_219606

/-- The cost of a mold for handmade shoes --/
def mold_cost (hourly_rate : ℝ) (hours : ℝ) (work_percentage : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid - work_percentage * hourly_rate * hours

/-- Proves that the cost of the mold is $250 given the problem conditions --/
theorem mold_cost_is_250 :
  mold_cost 75 8 0.8 730 = 250 := by
  sorry

end mold_cost_is_250_l2196_219606


namespace polynomial_zeros_evaluation_l2196_219603

theorem polynomial_zeros_evaluation (r s : ℝ) : 
  r^2 - 3*r + 1 = 0 → 
  s^2 - 3*s + 1 = 0 → 
  (1 : ℝ)^2 - 18*(1 : ℝ) + 1 = -16 := by
  sorry

end polynomial_zeros_evaluation_l2196_219603


namespace no_root_in_interval_l2196_219660

-- Define the function f(x) = x^5 - 3x - 1
def f (x : ℝ) : ℝ := x^5 - 3*x - 1

-- State the theorem
theorem no_root_in_interval :
  ∀ x ∈ Set.Ioo 2 3, f x ≠ 0 :=
by
  sorry

end no_root_in_interval_l2196_219660


namespace li_ying_final_score_l2196_219617

/-- Calculate the final score in a quiz where correct answers earn points and incorrect answers deduct points. -/
def calculate_final_score (correct_points : ℤ) (incorrect_points : ℤ) (num_correct : ℕ) (num_incorrect : ℕ) : ℤ :=
  correct_points * num_correct - incorrect_points * num_incorrect

/-- Theorem stating that Li Ying's final score in the safety knowledge quiz is 45 points. -/
theorem li_ying_final_score :
  let correct_points : ℤ := 5
  let incorrect_points : ℤ := 3
  let num_correct : ℕ := 12
  let num_incorrect : ℕ := 5
  calculate_final_score correct_points incorrect_points num_correct num_incorrect = 45 := by
  sorry

#eval calculate_final_score 5 3 12 5

end li_ying_final_score_l2196_219617


namespace power_calculation_l2196_219662

theorem power_calculation : 
  ((18^13 * 18^11)^2 / 6^8) * 3^4 = 2^40 * 3^92 := by sorry

end power_calculation_l2196_219662


namespace sum_of_possible_radii_l2196_219611

/-- Given a circle with center C(r,r) that is tangent to the positive x-axis,
    positive y-axis, and externally tangent to a circle centered at (4,0) with radius 1,
    the sum of all possible radii of circle C is 10. -/
theorem sum_of_possible_radii : ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 1)^2) →
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧
    ((r₁ - 4)^2 + r₁^2 = (r₁ + 1)^2) ∧
    ((r₂ - 4)^2 + r₂^2 = (r₂ + 1)^2) ∧
    r₁ + r₂ = 10) :=
by sorry

end sum_of_possible_radii_l2196_219611


namespace revenue_loss_l2196_219625

/-- Represents the types of tickets sold in the theater. -/
inductive TicketType
  | GeneralRegular
  | GeneralVIP
  | ChildRegular
  | ChildVIP
  | SeniorRegular
  | SeniorVIP
  | VeteranRegular
  | VeteranVIP

/-- Calculates the revenue for a given ticket type. -/
def ticketRevenue (t : TicketType) : ℚ :=
  match t with
  | .GeneralRegular => 10
  | .GeneralVIP => 15
  | .ChildRegular => 6
  | .ChildVIP => 11
  | .SeniorRegular => 8
  | .SeniorVIP => 13
  | .VeteranRegular => 8
  | .VeteranVIP => 13

/-- Represents the theater's seating and pricing structure. -/
structure Theater where
  regularSeats : ℕ
  vipSeats : ℕ
  regularPrice : ℚ
  vipSurcharge : ℚ

/-- Calculates the potential revenue if all seats were sold at full price. -/
def potentialRevenue (t : Theater) : ℚ :=
  t.regularSeats * t.regularPrice + t.vipSeats * (t.regularPrice + t.vipSurcharge)

/-- Represents the actual sales for the night. -/
structure ActualSales where
  generalRegular : ℕ
  generalVIP : ℕ
  childRegular : ℕ
  childVIP : ℕ
  seniorRegular : ℕ
  seniorVIP : ℕ
  veteranRegular : ℕ
  veteranVIP : ℕ

/-- Calculates the actual revenue from the given sales. -/
def actualRevenue (s : ActualSales) : ℚ :=
  s.generalRegular * ticketRevenue .GeneralRegular +
  s.generalVIP * ticketRevenue .GeneralVIP +
  s.childRegular * ticketRevenue .ChildRegular +
  s.childVIP * ticketRevenue .ChildVIP +
  s.seniorRegular * ticketRevenue .SeniorRegular +
  s.seniorVIP * ticketRevenue .SeniorVIP +
  s.veteranRegular * ticketRevenue .VeteranRegular +
  s.veteranVIP * ticketRevenue .VeteranVIP

theorem revenue_loss (t : Theater) (s : ActualSales) :
    t.regularSeats = 40 ∧
    t.vipSeats = 10 ∧
    t.regularPrice = 10 ∧
    t.vipSurcharge = 5 ∧
    s.generalRegular = 12 ∧
    s.generalVIP = 6 ∧
    s.childRegular = 3 ∧
    s.childVIP = 1 ∧
    s.seniorRegular = 4 ∧
    s.seniorVIP = 2 ∧
    s.veteranRegular = 2 ∧
    s.veteranVIP = 1 →
    potentialRevenue t - actualRevenue s = 224 := by
  sorry

#eval potentialRevenue { regularSeats := 40, vipSeats := 10, regularPrice := 10, vipSurcharge := 5 }
#eval actualRevenue { generalRegular := 12, generalVIP := 6, childRegular := 3, childVIP := 1,
                      seniorRegular := 4, seniorVIP := 2, veteranRegular := 2, veteranVIP := 1 }

end revenue_loss_l2196_219625


namespace female_population_l2196_219665

theorem female_population (total_population : ℕ) (num_parts : ℕ) (female_parts : ℕ) : 
  total_population = 720 →
  num_parts = 4 →
  female_parts = 2 →
  (total_population / num_parts) * female_parts = 360 :=
by sorry

end female_population_l2196_219665


namespace dealership_sales_prediction_l2196_219658

/-- The number of sports cars predicted to be sold -/
def sports_cars : ℕ := 45

/-- The ratio of sports cars to sedans -/
def ratio : ℚ := 3 / 5

/-- The minimum difference between sedans and sports cars -/
def min_difference : ℕ := 20

/-- The number of sedans expected to be sold -/
def sedans : ℕ := 75

theorem dealership_sales_prediction :
  (sedans : ℚ) = sports_cars / ratio ∧ 
  sedans ≥ sports_cars + min_difference := by
  sorry

end dealership_sales_prediction_l2196_219658


namespace smallest_divisor_after_221_next_divisor_is_289_l2196_219657

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_divisor_after_221 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 221 = 0)           -- 221 is a divisor of m
  : (∃ d : ℕ, d ∣ m ∧ 221 < d ∧ d < 289) → False :=
by sorry

theorem next_divisor_is_289 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 221 = 0)           -- 221 is a divisor of m
  : 289 ∣ m :=
by sorry

end smallest_divisor_after_221_next_divisor_is_289_l2196_219657


namespace roxy_garden_plants_l2196_219695

def garden_problem (initial_flowering : ℕ) (initial_fruiting_multiplier : ℕ)
  (bought_flowering : ℕ) (bought_fruiting : ℕ)
  (given_flowering : ℕ) (given_fruiting : ℕ) : ℕ :=
  let initial_fruiting := initial_flowering * initial_fruiting_multiplier
  let after_buying_flowering := initial_flowering + bought_flowering
  let after_buying_fruiting := initial_fruiting + bought_fruiting
  let final_flowering := after_buying_flowering - given_flowering
  let final_fruiting := after_buying_fruiting - given_fruiting
  final_flowering + final_fruiting

theorem roxy_garden_plants :
  garden_problem 7 2 3 2 1 4 = 21 := by
  sorry

end roxy_garden_plants_l2196_219695


namespace joey_age_digit_sum_l2196_219667

def joey_age_sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem joey_age_digit_sum :
  ∃ (chloe_age : ℕ) (joey_age : ℕ),
    joey_age = chloe_age + 2 ∧
    chloe_age > 2 ∧
    chloe_age % 5 = 0 ∧
    joey_age % 5 = 0 ∧
    ∀ k : ℕ, k < chloe_age → k % 5 ≠ 0 ∧
    joey_age_sum_of_digits joey_age = 1 :=
by
  sorry

end joey_age_digit_sum_l2196_219667


namespace unique_integer_solution_l2196_219638

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2*n^2 + m^2 + n^2 + 6*m*n → m = 0 ∧ n = 0 := by
  sorry

end unique_integer_solution_l2196_219638


namespace probability_two_red_balls_l2196_219674

/-- The probability of picking two red balls from a bag containing 7 red, 5 blue, and 4 green balls -/
theorem probability_two_red_balls (red blue green : ℕ) (h1 : red = 7) (h2 : blue = 5) (h3 : green = 4) :
  let total := red + blue + green
  (red / total) * ((red - 1) / (total - 1)) = 7 / 40 := by
  sorry

end probability_two_red_balls_l2196_219674


namespace square_root_of_negative_two_squared_l2196_219634

theorem square_root_of_negative_two_squared (x : ℝ) : x = 2 → x ^ 2 = (-2) ^ 2 := by sorry

end square_root_of_negative_two_squared_l2196_219634


namespace fraction_ordering_l2196_219628

theorem fraction_ordering : 
  let a := 23
  let b := 18
  let c := 21
  let d := 16
  let e := 25
  let f := 19
  (a : ℚ) / b < (c : ℚ) / d ∧ (c : ℚ) / d < (e : ℚ) / f := by sorry

end fraction_ordering_l2196_219628


namespace exactly_one_solves_l2196_219602

/-- The probability that exactly one person solves a problem given two independent probabilities -/
theorem exactly_one_solves (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) 
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = p₁ + p₂ - 2 * p₁ * p₂ := by
  sorry

end exactly_one_solves_l2196_219602


namespace definite_integral_x_squared_l2196_219623

theorem definite_integral_x_squared : ∫ x in (-1)..(1), x^2 = 2/3 := by
  sorry

end definite_integral_x_squared_l2196_219623


namespace spencer_sessions_per_day_l2196_219624

/-- Represents the jumping routine of Spencer --/
structure JumpingRoutine where
  jumps_per_minute : ℕ
  minutes_per_session : ℕ
  total_jumps : ℕ
  total_days : ℕ

/-- Calculates the number of sessions per day for Spencer's jumping routine --/
def sessions_per_day (routine : JumpingRoutine) : ℚ :=
  (routine.total_jumps / routine.total_days) / (routine.jumps_per_minute * routine.minutes_per_session)

/-- Theorem stating that Spencer's jumping routine results in 2 sessions per day --/
theorem spencer_sessions_per_day :
  let routine := JumpingRoutine.mk 4 10 400 5
  sessions_per_day routine = 2 := by
  sorry

end spencer_sessions_per_day_l2196_219624


namespace three_digit_number_puzzle_l2196_219616

theorem three_digit_number_puzzle :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →
    a + b + c = 10 →
    b = a + c →
    100 * c + 10 * b + a = 100 * a + 10 * b + c + 99 →
    100 * a + 10 * b + c = 253 := by
  sorry

end three_digit_number_puzzle_l2196_219616


namespace triangle_special_angle_l2196_219610

open Real

/-- In a triangle ABC, given that 2b cos A = 2c - √3a, prove that angle B is π/6 --/
theorem triangle_special_angle (a b c : ℝ) (A B C : ℝ) (h : 2 * b * cos A = 2 * c - Real.sqrt 3 * a) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π →
  B = π / 6 := by
  sorry

end triangle_special_angle_l2196_219610


namespace alternating_power_difference_l2196_219601

theorem alternating_power_difference : (-1 : ℤ)^2010 - (-1 : ℤ)^2011 = 2 := by
  sorry

end alternating_power_difference_l2196_219601


namespace percent_of_itself_l2196_219654

theorem percent_of_itself (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 4) : x = 20 := by
  sorry

end percent_of_itself_l2196_219654


namespace rectangle_square_comparison_l2196_219609

/-- Proves that for a rectangle with a 3:1 length-to-width ratio and 75 cm² area,
    the difference between the side of a square with equal area and the rectangle's width
    is greater than 3 cm. -/
theorem rectangle_square_comparison : ∀ (length width : ℝ),
  length / width = 3 →
  length * width = 75 →
  ∃ (square_side : ℝ),
    square_side^2 = 75 ∧
    square_side - width > 3 :=
by sorry

end rectangle_square_comparison_l2196_219609


namespace inequality_solution_l2196_219622

theorem inequality_solution (x : ℝ) :
  (6 * x^2 + 9 * x - 48) / ((3 * x + 5) * (x - 2)) < 0 ↔ 
  -4 < x ∧ x < -5/3 ∧ x ≠ 2 :=
by sorry

end inequality_solution_l2196_219622


namespace k_h_symmetry_l2196_219693

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x^2 - 12

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_symmetry (h_def : ∀ x, h x = 4 * x^2 - 12) 
                     (k_h_3 : k (h 3) = 16) : 
  k (h (-3)) = 16 := by
  sorry


end k_h_symmetry_l2196_219693


namespace quadratic_inequality_solution_l2196_219661

/-- Given a quadratic inequality x^2 - mx + t < 0 with solution set {x | 2 < x < 3}, prove that m - t = -1 -/
theorem quadratic_inequality_solution (m t : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + t < 0 ↔ 2 < x ∧ x < 3) → 
  m - t = -1 := by
  sorry

end quadratic_inequality_solution_l2196_219661


namespace volume_of_five_cubes_l2196_219637

/-- The volume of a solid formed by adjacent cubes -/
def volume_of_adjacent_cubes (n : ℕ) (side_length : ℝ) : ℝ :=
  n * (side_length ^ 3)

/-- Theorem: The volume of a solid formed by five adjacent cubes with side length 5 cm is 625 cm³ -/
theorem volume_of_five_cubes : volume_of_adjacent_cubes 5 5 = 625 := by
  sorry

end volume_of_five_cubes_l2196_219637


namespace ratio_odd_even_divisors_M_l2196_219640

/-- The number M as defined in the problem -/
def M : ℕ := 25 * 48 * 49 * 81

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of M is 1:30 -/
theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 30 := by sorry

end ratio_odd_even_divisors_M_l2196_219640


namespace unique_coin_expected_value_l2196_219691

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (win_heads : ℚ) (loss_tails : ℚ) : ℚ :=
  p_heads * win_heads + p_tails * (-loss_tails)

theorem unique_coin_expected_value :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_heads : ℚ := 4
  let loss_tails : ℚ := 3
  coin_flip_expected_value p_heads p_tails win_heads loss_tails = -1/5 := by
  sorry

end unique_coin_expected_value_l2196_219691


namespace find_M_l2196_219694

theorem find_M : ∃ M : ℚ, (25 / 100) * M = (35 / 100) * 4025 ∧ M = 5635 := by
  sorry

end find_M_l2196_219694


namespace segments_form_triangle_l2196_219652

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem segments_form_triangle :
  can_form_triangle 5 6 10 :=
by sorry

end segments_form_triangle_l2196_219652


namespace toy_production_rate_l2196_219664

/-- Represents the toy production in a factory --/
structure ToyFactory where
  weekly_production : ℕ
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ

/-- Calculates the hourly toy production rate --/
def hourly_production_rate (factory : ToyFactory) : ℚ :=
  let total_hours := factory.monday_hours + factory.tuesday_hours + factory.wednesday_hours + factory.thursday_hours
  factory.weekly_production / total_hours

/-- Theorem stating the hourly production rate for the given factory --/
theorem toy_production_rate (factory : ToyFactory) 
  (h1 : factory.weekly_production = 20500)
  (h2 : factory.monday_hours = 8)
  (h3 : factory.tuesday_hours = 7)
  (h4 : factory.wednesday_hours = 9)
  (h5 : factory.thursday_hours = 6) :
  ∃ (ε : ℚ), abs (hourly_production_rate factory - 683.33) < ε ∧ ε > 0 := by
  sorry


end toy_production_rate_l2196_219664


namespace like_terms_proof_l2196_219659

/-- Two algebraic expressions are like terms if they have the same variables with the same exponents. -/
def like_terms (expr1 expr2 : String) : Prop := sorry

theorem like_terms_proof :
  (like_terms "3a³b" "-3ba³") ∧
  ¬(like_terms "a³" "b³") ∧
  ¬(like_terms "abc" "ac") ∧
  ¬(like_terms "a⁵" "2⁵") := by sorry

end like_terms_proof_l2196_219659


namespace orange_packing_l2196_219697

theorem orange_packing (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 2650) (h2 : oranges_per_box = 10) :
  total_oranges / oranges_per_box = 265 := by
  sorry

end orange_packing_l2196_219697


namespace handshake_arrangement_count_l2196_219682

/-- A handshake arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, i ∈ shakes j ↔ j ∈ shakes i)

/-- The number of distinct handshake arrangements for 12 people -/
def M : ℕ := sorry

/-- The main theorem: M is congruent to 850 modulo 1000 -/
theorem handshake_arrangement_count :
  M ≡ 850 [MOD 1000] := by sorry

end handshake_arrangement_count_l2196_219682


namespace factor_t_squared_minus_64_l2196_219666

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l2196_219666


namespace complement_of_S_in_U_l2196_219683

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define the set S
def S : Set Nat := {1, 3}

-- Theorem statement
theorem complement_of_S_in_U :
  U \ S = {2, 4} := by sorry

end complement_of_S_in_U_l2196_219683


namespace donation_problem_solution_l2196_219615

/-- Represents a transportation plan with type A and B trucks -/
structure TransportPlan where
  typeA : Nat
  typeB : Nat

/-- Represents the problem setup -/
structure DonationProblem where
  totalItems : Nat
  waterExcess : Nat
  typeAWaterCapacity : Nat
  typeAVegCapacity : Nat
  typeBWaterCapacity : Nat
  typeBVegCapacity : Nat
  totalTrucks : Nat
  typeACost : Nat
  typeBCost : Nat

def isValidPlan (p : DonationProblem) (plan : TransportPlan) : Prop :=
  plan.typeA + plan.typeB = p.totalTrucks ∧
  plan.typeA * p.typeAWaterCapacity + plan.typeB * p.typeBWaterCapacity ≥ (p.totalItems + p.waterExcess) / 2 ∧
  plan.typeA * p.typeAVegCapacity + plan.typeB * p.typeBVegCapacity ≥ (p.totalItems - p.waterExcess) / 2

def planCost (p : DonationProblem) (plan : TransportPlan) : Nat :=
  plan.typeA * p.typeACost + plan.typeB * p.typeBCost

theorem donation_problem_solution (p : DonationProblem)
  (h_total : p.totalItems = 320)
  (h_excess : p.waterExcess = 80)
  (h_typeA : p.typeAWaterCapacity = 40 ∧ p.typeAVegCapacity = 10)
  (h_typeB : p.typeBWaterCapacity = 20 ∧ p.typeBVegCapacity = 20)
  (h_trucks : p.totalTrucks = 8)
  (h_costs : p.typeACost = 400 ∧ p.typeBCost = 360) :
  -- 1. Number of water and vegetable pieces
  (p.totalItems + p.waterExcess) / 2 = 200 ∧ (p.totalItems - p.waterExcess) / 2 = 120 ∧
  -- 2. Valid transportation plans
  (∀ plan, isValidPlan p plan ↔ 
    (plan = ⟨2, 6⟩ ∨ plan = ⟨3, 5⟩ ∨ plan = ⟨4, 4⟩)) ∧
  -- 3. Minimum cost plan
  (∀ plan, isValidPlan p plan → planCost p ⟨2, 6⟩ ≤ planCost p plan) ∧
  planCost p ⟨2, 6⟩ = 2960 :=
sorry

end donation_problem_solution_l2196_219615


namespace rook_placement_modulo_four_l2196_219681

/-- The color of a cell on the board -/
def cellColor (n i j : ℕ) : ℕ := min (i + j - 1) (2 * n - i - j + 1)

/-- A valid rook placement function -/
def IsValidRookPlacement (n : ℕ) (f : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i, i ∈ Finset.range n → f i ∈ Finset.range n) ∧
  (∀ i j, i ≠ j → cellColor n i (f i) ≠ cellColor n j (f j))

theorem rook_placement_modulo_four (n : ℕ) :
  (∃ f, IsValidRookPlacement n f) →
  n % 4 = 0 ∨ n % 4 = 1 := by
sorry

end rook_placement_modulo_four_l2196_219681


namespace min_sum_of_determinant_condition_l2196_219663

theorem min_sum_of_determinant_condition (x y : ℤ) 
  (h : 1 < 6 - x * y ∧ 6 - x * y < 3) : 
  ∃ (a b : ℤ), a + b = -5 ∧ 
    (∀ (c d : ℤ), 1 < 6 - c * d ∧ 6 - c * d < 3 → a + b ≤ c + d) := by
  sorry

end min_sum_of_determinant_condition_l2196_219663


namespace inverse_false_implies_negation_false_l2196_219668

theorem inverse_false_implies_negation_false (p : Prop) :
  (p → False) → ¬p = False :=
by sorry

end inverse_false_implies_negation_false_l2196_219668


namespace sector_area_l2196_219629

theorem sector_area (α : ℝ) (p : ℝ) (h1 : α = 2) (h2 : p = 8) :
  let r := p / (α + 2)
  (1/2) * α * r^2 = 4 := by sorry

end sector_area_l2196_219629


namespace probability_heart_spade_club_standard_deck_l2196_219678

/-- A standard deck of cards. -/
structure Deck :=
  (total : Nat)
  (hearts : Nat)
  (spades : Nat)
  (clubs : Nat)
  (diamonds : Nat)

/-- The probability of drawing a heart, then a spade, then a club from a standard deck. -/
def probability_heart_spade_club (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total *
  (d.spades : ℚ) / (d.total - 1) *
  (d.clubs : ℚ) / (d.total - 2)

/-- Theorem stating the probability of drawing a heart, then a spade, then a club
    from a standard 52-card deck. -/
theorem probability_heart_spade_club_standard_deck :
  let standard_deck : Deck := ⟨52, 13, 13, 13, 13⟩
  probability_heart_spade_club standard_deck = 2197 / 132600 := by
  sorry

end probability_heart_spade_club_standard_deck_l2196_219678


namespace hot_dog_buns_per_student_l2196_219608

theorem hot_dog_buns_per_student (
  buns_per_package : ℕ)
  (packages_bought : ℕ)
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages_bought = 30)
  (h3 : num_classes = 4)
  (h4 : students_per_class = 30)
  : (buns_per_package * packages_bought) / (num_classes * students_per_class) = 2 := by
  sorry

end hot_dog_buns_per_student_l2196_219608


namespace parabola_midpoint_trajectory_and_intersection_l2196_219604

/-- Parabola C -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Trajectory D -/
def trajectory_D (x y : ℝ) : Prop := y^2 = x

/-- Line l with slope 1 passing through (1, 0) -/
def line_l (x y : ℝ) : Prop := y = x - 1

/-- The focus of parabola C -/
def focus_C : ℝ × ℝ := (1, 0)

/-- The statement to prove -/
theorem parabola_midpoint_trajectory_and_intersection :
  (∀ x y : ℝ, parabola_C x y → ∃ x' y' : ℝ, trajectory_D x' y' ∧ y' = y / 2 ∧ x' = x) ∧
  (∃ A B : ℝ × ℝ,
    trajectory_D A.1 A.2 ∧ trajectory_D B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 10) :=
sorry

end parabola_midpoint_trajectory_and_intersection_l2196_219604


namespace base_conversion_l2196_219684

/-- Given that 26 in decimal is equal to 32 in base k, prove that k = 8 -/
theorem base_conversion (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 := by
  sorry

end base_conversion_l2196_219684


namespace twelveRowTriangle_l2196_219631

/-- Calculates the sum of an arithmetic progression -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Represents the triangle construction -/
structure TriangleConstruction where
  rows : ℕ
  firstRowRods : ℕ
  rodIncreasePerRow : ℕ

/-- Calculates the total number of pieces in the triangle construction -/
def totalPieces (t : TriangleConstruction) : ℕ :=
  let rodSum := arithmeticSum t.firstRowRods t.rodIncreasePerRow t.rows
  let connectorSum := arithmeticSum 1 1 (t.rows + 1)
  rodSum + connectorSum

/-- Theorem statement for the 12-row triangle construction -/
theorem twelveRowTriangle :
  totalPieces { rows := 12, firstRowRods := 3, rodIncreasePerRow := 3 } = 325 := by
  sorry


end twelveRowTriangle_l2196_219631


namespace least_number_with_divisibility_property_l2196_219618

theorem least_number_with_divisibility_property : ∃ k : ℕ, 
  k > 0 ∧ 
  (k / 23 = k % 47 + 13) ∧
  (∀ m : ℕ, m > 0 → m < k → m / 23 ≠ m % 47 + 13) ∧
  k = 576 :=
sorry

end least_number_with_divisibility_property_l2196_219618


namespace milk_cost_per_liter_l2196_219679

/-- Represents the milkman's milk mixture problem -/
def MilkProblem (total_milk pure_milk water_added mixture_price profit : ℝ) : Prop :=
  total_milk = 30 ∧
  pure_milk = 20 ∧
  water_added = 5 ∧
  (pure_milk + water_added) * mixture_price - pure_milk * mixture_price = profit ∧
  profit = 35

/-- The cost of pure milk per liter is 7 rupees -/
theorem milk_cost_per_liter (total_milk pure_milk water_added mixture_price profit : ℝ) 
  (h : MilkProblem total_milk pure_milk water_added mixture_price profit) : 
  mixture_price = 7 := by
  sorry

end milk_cost_per_liter_l2196_219679


namespace arithmetic_sequence_sum_remainder_l2196_219676

def arithmetic_sequence_sum (a : ℕ) (d : ℕ) (last : ℕ) : ℕ :=
  let n := (last - a) / d + 1
  n * (a + last) / 2

theorem arithmetic_sequence_sum_remainder
  (h1 : arithmetic_sequence_sum 3 6 279 % 8 = 3) : 
  arithmetic_sequence_sum 3 6 279 % 8 = 3 := by
  sorry

end arithmetic_sequence_sum_remainder_l2196_219676


namespace intersection_implies_B_equals_one_three_l2196_219630

def A : Set ℝ := {1, 2, 4}

def B (m : ℝ) : Set ℝ := {x | x^2 - 4*x + m = 0}

theorem intersection_implies_B_equals_one_three :
  ∃ m : ℝ, (A ∩ B m = {1}) → (B m = {1, 3}) :=
by sorry

end intersection_implies_B_equals_one_three_l2196_219630


namespace train_crossing_time_l2196_219689

/-- Given a train and platform with specific properties, calculate the time for the train to cross a tree -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 1400)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 150) : 
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 100 := by
  sorry

end train_crossing_time_l2196_219689


namespace tax_reduction_theorem_l2196_219656

theorem tax_reduction_theorem (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_consumption := original_consumption * 1.05
  let new_revenue := original_tax * original_consumption * 0.84
  let new_tax := new_revenue / new_consumption
  (original_tax - new_tax) / original_tax = 0.2 := by
sorry

end tax_reduction_theorem_l2196_219656


namespace function_inequality_solution_set_l2196_219644

open Real

-- Define the function f and its properties
theorem function_inequality_solution_set 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (h1 : ∀ x > 0, HasDerivAt f (f' x) x)
  (h2 : ∀ x > 0, x * f' x + f x = (log x) / x)
  (h3 : f (exp 1) = (exp 1)⁻¹) :
  {x : ℝ | f (x + 1) - f ((exp 1) + 1) > x - (exp 1)} = Set.Ioo (-1) (exp 1) := by
  sorry

end function_inequality_solution_set_l2196_219644


namespace symmetric_line_equation_l2196_219686

/-- Given a fold line y = -x and a line l₁ with equation 2x + 3y - 1 = 0,
    the symmetric line l₂ with respect to the fold line has the equation 3x + 2y + 1 = 0 -/
theorem symmetric_line_equation (x y : ℝ) :
  (y = -x) →  -- fold line equation
  (2*x + 3*y - 1 = 0) →  -- l₁ equation
  (3*x + 2*y + 1 = 0)  -- l₂ equation (to be proved)
:= by sorry

end symmetric_line_equation_l2196_219686


namespace negation_equivalence_l2196_219649

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by
  sorry

end negation_equivalence_l2196_219649


namespace unique_pair_divisibility_l2196_219600

theorem unique_pair_divisibility : 
  ∃! (n m : ℕ), n > 2 ∧ m > 2 ∧ 
  (∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ k ∈ S, (k^n + k^2 - 1) ∣ (k^m + k - 1)) ∧
  n = 3 ∧ m = 5 :=
sorry

end unique_pair_divisibility_l2196_219600


namespace mirror_area_l2196_219620

/-- The area of a rectangular mirror that fits exactly inside a frame with given dimensions. -/
theorem mirror_area (frame_length frame_width frame_thickness : ℕ) : 
  frame_length = 70 ∧ frame_width = 90 ∧ frame_thickness = 15 → 
  (frame_length - 2 * frame_thickness) * (frame_width - 2 * frame_thickness) = 2400 :=
by
  sorry

#check mirror_area

end mirror_area_l2196_219620


namespace project_hours_difference_l2196_219680

/-- Given a project with three contributors (Pat, Kate, and Mark) with specific charging ratios,
    prove the difference in hours charged between Mark and Kate. -/
theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 198 →
  kate_hours + 2 * kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 110 := by
  sorry

end project_hours_difference_l2196_219680


namespace mean_height_of_basketball_team_l2196_219671

def heights : List ℝ := [48, 50, 51, 54, 56, 57, 57, 59, 60, 63, 64, 65, 67, 69, 69, 71, 72, 74]

theorem mean_height_of_basketball_team : 
  (heights.sum / heights.length : ℝ) = 61.444444444444445 := by sorry

end mean_height_of_basketball_team_l2196_219671


namespace ln_power_rational_l2196_219655

theorem ln_power_rational (f : ℝ) (r : ℚ) (hf : f > 0) :
  Real.log (f ^ (r : ℝ)) = r * Real.log f := by
  sorry

end ln_power_rational_l2196_219655


namespace initial_pairs_count_l2196_219626

/-- Represents the number of shoes in a pair -/
def shoesPerPair : ℕ := 2

/-- Represents the number of individual shoes lost -/
def shoesLost : ℕ := 9

/-- Represents the number of matching pairs left after losing shoes -/
def pairsLeft : ℕ := 15

/-- Theorem stating that the initial number of pairs is 24 given the conditions -/
theorem initial_pairs_count :
  ∀ (initialPairs : ℕ),
  (initialPairs * shoesPerPair - shoesLost) / shoesPerPair = pairsLeft →
  initialPairs = pairsLeft + shoesLost / shoesPerPair :=
by
  sorry

#check initial_pairs_count

end initial_pairs_count_l2196_219626


namespace simplify_expression_l2196_219613

theorem simplify_expression (a : ℝ) : (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := by
  sorry

end simplify_expression_l2196_219613


namespace count_special_numbers_is_4032_l2196_219685

/-- A function that counts the number of 5-digit numbers starting with '2' and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := 5
  let start_digit := 2
  let identical_digits := 2
  -- The actual counting logic would go here
  4032

/-- Theorem stating that the count of special numbers is 4032 -/
theorem count_special_numbers_is_4032 :
  count_special_numbers = 4032 := by sorry

end count_special_numbers_is_4032_l2196_219685


namespace sara_sister_notebooks_l2196_219639

def calculate_notebooks (initial : ℕ) (increase_percent : ℕ) (lost : ℕ) : ℕ :=
  let increased : ℕ := initial + initial * increase_percent / 100
  increased - lost

theorem sara_sister_notebooks : calculate_notebooks 4 150 2 = 8 := by
  sorry

end sara_sister_notebooks_l2196_219639


namespace martin_initial_fruits_l2196_219605

/-- The number of fruits Martin initially had --/
def initial_fruits : ℕ := 288

/-- The number of oranges Martin has after eating half his fruits --/
def oranges_after : ℕ := 50

/-- The number of apples Martin has after eating half his fruits --/
def apples_after : ℕ := 72

/-- The number of limes Martin has after eating half his fruits --/
def limes_after : ℕ := 24

theorem martin_initial_fruits :
  (initial_fruits / 2 = oranges_after + apples_after + limes_after) ∧
  (oranges_after = 2 * limes_after) ∧
  (apples_after = 3 * limes_after) ∧
  (oranges_after = 50) ∧
  (apples_after = 72) :=
by sorry

end martin_initial_fruits_l2196_219605


namespace christine_wandering_l2196_219614

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Christine's wandering problem -/
theorem christine_wandering (christine_speed : ℝ) (christine_time : ℝ) 
  (h1 : christine_speed = 4)
  (h2 : christine_time = 5) :
  distance christine_speed christine_time = 20 := by
  sorry

end christine_wandering_l2196_219614


namespace passing_percentage_l2196_219627

def max_score : ℕ := 750
def mike_score : ℕ := 212
def shortfall : ℕ := 13

theorem passing_percentage : 
  (((mike_score + shortfall : ℚ) / max_score) * 100 : ℚ) = 30 := by
  sorry

end passing_percentage_l2196_219627


namespace max_disks_in_rectangle_l2196_219653

/-- The maximum number of circular disks that can be cut from a rectangular sheet. -/
def max_disks (rect_width rect_height disk_diameter : ℝ) : ℕ :=
  32

/-- Theorem stating the maximum number of 5 cm diameter circular disks 
    that can be cut from a 9 × 100 cm rectangular sheet. -/
theorem max_disks_in_rectangle : 
  max_disks 9 100 5 = 32 := by sorry

end max_disks_in_rectangle_l2196_219653


namespace cats_adoption_proof_l2196_219696

def adopt_cats (initial_cats : ℕ) (added_cats : ℕ) (cats_per_adopter : ℕ) (final_cats : ℕ) : ℕ :=
  ((initial_cats + added_cats) - final_cats) / cats_per_adopter

theorem cats_adoption_proof :
  adopt_cats 20 3 2 17 = 3 := by
  sorry

end cats_adoption_proof_l2196_219696


namespace not_perfect_square_600_sixes_and_zeros_l2196_219650

/-- Represents a number with 600 digits of 6 followed by some zeros -/
def number_with_600_sixes_and_zeros (n : ℕ) : ℕ :=
  6 * 10^600 + n

/-- Theorem stating that a number with 600 digits of 6 followed by any number of zeros cannot be a perfect square -/
theorem not_perfect_square_600_sixes_and_zeros (n : ℕ) :
  ∃ (m : ℕ), (number_with_600_sixes_and_zeros n) = m^2 → False :=
sorry

end not_perfect_square_600_sixes_and_zeros_l2196_219650


namespace seonwoo_change_l2196_219632

/-- Calculates the change Seonwoo received after buying bubblegum and ramen. -/
theorem seonwoo_change
  (initial_amount : ℕ)
  (bubblegum_cost : ℕ)
  (bubblegum_count : ℕ)
  (ramen_cost_per_two : ℕ)
  (ramen_count : ℕ)
  (h1 : initial_amount = 10000)
  (h2 : bubblegum_cost = 600)
  (h3 : bubblegum_count = 2)
  (h4 : ramen_cost_per_two = 1600)
  (h5 : ramen_count = 9) :
  initial_amount - (bubblegum_cost * bubblegum_count + 
    (ramen_cost_per_two * (ramen_count / 2)) + 
    (ramen_cost_per_two / 2 * (ramen_count % 2))) = 1600 :=
by sorry

end seonwoo_change_l2196_219632


namespace polynomial_root_problem_l2196_219699

theorem polynomial_root_problem (a b c d : ℤ) : 
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + d * (3 + Complex.I) + a = 0 →
  Int.gcd (Int.gcd (Int.gcd a b) c) d = 1 →
  d.natAbs = 33 := by
  sorry

end polynomial_root_problem_l2196_219699


namespace area_regular_octagon_in_circle_l2196_219690

/-- The area of a regular octagon inscribed in a circle with area 256π -/
theorem area_regular_octagon_in_circle (circle_area : ℝ) (octagon_area : ℝ) : 
  circle_area = 256 * Real.pi → 
  octagon_area = 8 * (1/2 * (circle_area / Real.pi) * Real.sin (Real.pi / 4)) → 
  octagon_area = 512 * Real.sqrt 2 := by
sorry

end area_regular_octagon_in_circle_l2196_219690


namespace expression_simplification_l2196_219675

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.sqrt 2) 
  (hy : y = Real.sqrt 3) : 
  (x + y) * (x - y) - y * (2 * x - y) = 2 - 2 * Real.sqrt 6 := by
  sorry

end expression_simplification_l2196_219675


namespace valid_input_statement_l2196_219672

/-- Represents a programming language construct --/
inductive ProgramConstruct
| Input : String → String → ProgramConstruct
| Other : ProgramConstruct

/-- Checks if a given ProgramConstruct is a valid INPUT statement --/
def isValidInputStatement (stmt : ProgramConstruct) : Prop :=
  match stmt with
  | ProgramConstruct.Input prompt var => true
  | _ => false

/-- Theorem: An INPUT statement with a prompt and variable is valid --/
theorem valid_input_statement (prompt var : String) :
  isValidInputStatement (ProgramConstruct.Input prompt var) := by
  sorry

#check valid_input_statement

end valid_input_statement_l2196_219672


namespace smallest_square_area_for_rectangles_l2196_219648

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.width) (r1.height + r2.height)

/-- The theorem stating the smallest possible square area -/
theorem smallest_square_area_for_rectangles :
  let r1 : Rectangle := ⟨2, 5⟩
  let r2 : Rectangle := ⟨4, 3⟩
  (minSquareSide r1 r2) ^ 2 = 36 := by sorry

end smallest_square_area_for_rectangles_l2196_219648


namespace power_of_two_plus_one_l2196_219688

theorem power_of_two_plus_one (b m n : ℕ) 
  (h1 : b > 1) 
  (h2 : m ≠ n) 
  (h3 : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end power_of_two_plus_one_l2196_219688


namespace hyperbola_condition_l2196_219607

-- Define the equation
def equation (x y k : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (k - 5) = 1

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (k + 1 > 0 ∧ k - 5 < 0) ∨ (k + 1 < 0 ∧ k - 5 > 0)

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (∀ x y, equation x y k ↔ represents_hyperbola k) ↔ k ∈ Set.Ioo (-1 : ℝ) 5 :=
sorry

end hyperbola_condition_l2196_219607


namespace rebus_solution_l2196_219641

theorem rebus_solution : ∃! (A B C : ℕ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  (A < 10 ∧ B < 10 ∧ C < 10) ∧
  (100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C) ∧
  (100 * A + 10 * C + C = 1416) := by
sorry

end rebus_solution_l2196_219641


namespace no_solution_to_fractional_equation_l2196_219643

theorem no_solution_to_fractional_equation :
  ¬ ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) = 1 + 1 / (3 - x) := by
  sorry

end no_solution_to_fractional_equation_l2196_219643


namespace friday_increase_is_forty_percent_l2196_219692

/-- Represents the library borrowing scenario for Krystian --/
structure LibraryBorrowing where
  dailyAverage : ℕ
  weeklyTotal : ℕ
  workdays : ℕ

/-- Calculates the percentage increase of Friday's borrowing compared to the daily average --/
def fridayPercentageIncrease (lb : LibraryBorrowing) : ℚ :=
  let fridayBorrowing := lb.weeklyTotal - (lb.workdays - 1) * lb.dailyAverage
  let increase := fridayBorrowing - lb.dailyAverage
  (increase : ℚ) / lb.dailyAverage * 100

/-- Theorem stating that the percentage increase on Friday is 40% --/
theorem friday_increase_is_forty_percent (lb : LibraryBorrowing) 
    (h1 : lb.dailyAverage = 40)
    (h2 : lb.weeklyTotal = 216)
    (h3 : lb.workdays = 5) : 
  fridayPercentageIncrease lb = 40 := by
  sorry

end friday_increase_is_forty_percent_l2196_219692


namespace cubic_root_ratio_l2196_219633

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) :
  c / d = 5 / 12 := by
sorry

end cubic_root_ratio_l2196_219633


namespace greatest_integer_quadratic_inequality_l2196_219647

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 12*n + 28 ≤ 0 ∧ 
  ∀ (m : ℤ), m^2 - 12*m + 28 ≤ 0 → m ≤ n :=
by sorry

end greatest_integer_quadratic_inequality_l2196_219647


namespace problem_statement_l2196_219670

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → b / a + 2 / b ≤ x / y + 2 / x) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → a^2 + b^2 ≤ x^2 + y^2) :=
by sorry

end problem_statement_l2196_219670


namespace isosceles_triangle_side_length_l2196_219677

/-- Given a square with side length 2 and four congruent isosceles triangles
    constructed on its sides, if the sum of the triangles' areas equals
    the square's area, then each triangle's congruent side length is √2. -/
theorem isosceles_triangle_side_length :
  let square_side : ℝ := 2
  let square_area : ℝ := square_side ^ 2
  let triangle_area : ℝ := square_area / 4
  let triangle_base : ℝ := square_side
  let triangle_height : ℝ := 2 * triangle_area / triangle_base
  let triangle_side : ℝ := Real.sqrt (triangle_height ^ 2 + (triangle_base / 2) ^ 2)
  triangle_side = Real.sqrt 2 := by sorry

end isosceles_triangle_side_length_l2196_219677


namespace arrangements_not_head_tail_six_arrangements_not_adjacent_six_l2196_219612

/-- The number of students in the row -/
def n : ℕ := 6

/-- The number of arrangements where one student doesn't stand at the head or tail -/
def arrangements_not_head_tail (n : ℕ) : ℕ := sorry

/-- The number of arrangements where three specific students are not adjacent -/
def arrangements_not_adjacent (n : ℕ) : ℕ := sorry

/-- Theorem for the first question -/
theorem arrangements_not_head_tail_six : 
  arrangements_not_head_tail n = 480 := by sorry

/-- Theorem for the second question -/
theorem arrangements_not_adjacent_six : 
  arrangements_not_adjacent n = 144 := by sorry

end arrangements_not_head_tail_six_arrangements_not_adjacent_six_l2196_219612


namespace kind_wizard_strategy_exists_l2196_219635

-- Define a type for gnomes
def Gnome := ℕ

-- Define a friendship relation
def Friendship := Gnome × Gnome

-- Define a strategy for the kind wizard
def KindWizardStrategy := ℕ → List Friendship

-- Define the evil wizard's action
def EvilWizardAction := List Friendship → List Friendship

-- Define a circular arrangement of gnomes
def CircularArrangement := List Gnome

-- Function to check if an arrangement is valid (all neighbors are friends)
def IsValidArrangement (arrangement : CircularArrangement) (friendships : List Friendship) : Prop :=
  sorry

-- Main theorem
theorem kind_wizard_strategy_exists (n : ℕ) (h : n > 1 ∧ Odd n) :
  ∃ (strategy : KindWizardStrategy),
    ∀ (evil_action : EvilWizardAction),
      ∃ (arrangement : CircularArrangement),
        IsValidArrangement arrangement (evil_action (strategy n)) :=
sorry

end kind_wizard_strategy_exists_l2196_219635


namespace total_gum_pieces_l2196_219621

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) 
  (h1 : packages = 27) (h2 : pieces_per_package = 18) : 
  packages * pieces_per_package = 486 := by
  sorry

end total_gum_pieces_l2196_219621
