import Mathlib

namespace julia_car_rental_cost_l2528_252882

/-- Calculates the total cost of a car rental given the daily rate, per-mile charge, days rented, and miles driven. -/
def carRentalCost (dailyRate : ℝ) (perMileCharge : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  dailyRate * daysRented + perMileCharge * milesDriven

/-- Proves that Julia's car rental cost is $46.12 given the specific conditions. -/
theorem julia_car_rental_cost :
  let dailyRate : ℝ := 29
  let perMileCharge : ℝ := 0.08
  let daysRented : ℕ := 1
  let milesDriven : ℝ := 214.0
  carRentalCost dailyRate perMileCharge daysRented milesDriven = 46.12 := by
  sorry

end julia_car_rental_cost_l2528_252882


namespace article_profit_percentage_l2528_252829

theorem article_profit_percentage (cost : ℝ) (reduced_sell : ℝ) (new_profit_percent : ℝ) :
  cost = 40 →
  reduced_sell = 8.40 →
  new_profit_percent = 30 →
  let new_cost := cost * 0.80
  let new_sell := new_cost * (1 + new_profit_percent / 100)
  let orig_sell := new_sell + reduced_sell
  let profit := orig_sell - cost
  let profit_percent := (profit / cost) * 100
  profit_percent = 25 := by
  sorry

end article_profit_percentage_l2528_252829


namespace binomial_coefficient_two_l2528_252824

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2528_252824


namespace profit_calculation_l2528_252832

/-- Given that the cost price of 55 articles equals the selling price of n articles,
    and the percent profit is 10.000000000000004%, prove that n equals 50. -/
theorem profit_calculation (C S : ℝ) (n : ℕ) 
    (h1 : 55 * C = n * S)
    (h2 : (S - C) / C * 100 = 10.000000000000004) :
    n = 50 := by
  sorry

end profit_calculation_l2528_252832


namespace common_chord_of_circles_l2528_252872

/-- Given two circles in the xy-plane, prove that their common chord has a specific equation. -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) :=
by sorry

end common_chord_of_circles_l2528_252872


namespace solution_to_system_l2528_252805

theorem solution_to_system :
  ∃ (x y : ℚ), 3 * x - 24 * y = 3 ∧ x - 3 * y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end solution_to_system_l2528_252805


namespace largest_integer_in_range_l2528_252801

theorem largest_integer_in_range : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 3/5 ∧ 
  ∀ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 → y ≤ x :=
by sorry

end largest_integer_in_range_l2528_252801


namespace spectators_count_l2528_252821

/-- The number of wristbands given to each spectator -/
def wristbands_per_person : ℕ := 2

/-- The total number of wristbands distributed -/
def total_wristbands : ℕ := 250

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 125 := by
  sorry

end spectators_count_l2528_252821


namespace pig_weight_problem_l2528_252841

theorem pig_weight_problem (x y : ℝ) (h1 : x - y = 72) (h2 : x + y = 348) : x = 210 := by
  sorry

end pig_weight_problem_l2528_252841


namespace auction_starting_price_l2528_252852

/-- Auction price calculation -/
theorem auction_starting_price
  (final_price : ℕ)
  (price_increase : ℕ)
  (bids_per_person : ℕ)
  (num_bidders : ℕ)
  (h1 : final_price = 65)
  (h2 : price_increase = 5)
  (h3 : bids_per_person = 5)
  (h4 : num_bidders = 2) :
  final_price - (price_increase * bids_per_person * num_bidders) = 15 :=
by sorry

end auction_starting_price_l2528_252852


namespace period_and_trigonometric_function_l2528_252895

theorem period_and_trigonometric_function (ω : ℝ) (α β : ℝ) : 
  ω > 0 →
  (∀ x, 2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x) = 
    Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)) →
  (∀ x, 2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x) = 
    2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x)) →
  Real.sqrt 2 * Real.sin (α - Real.pi / 4 + Real.pi / 4) = Real.sqrt 2 / 3 →
  Real.sqrt 2 * Real.sin (β - Real.pi / 4 + Real.pi / 4) = 2 * Real.sqrt 2 / 3 →
  α > -Real.pi / 2 →
  α < Real.pi / 2 →
  β > -Real.pi / 2 →
  β < Real.pi / 2 →
  Real.cos (α + β) = (2 * Real.sqrt 10 - 2) / 9 := by
sorry


end period_and_trigonometric_function_l2528_252895


namespace files_deleted_l2528_252816

theorem files_deleted (initial_apps : ℕ) (initial_files : ℕ) (final_apps : ℕ) (final_files : ℕ) 
  (h1 : initial_apps = 17)
  (h2 : initial_files = 21)
  (h3 : final_apps = 3)
  (h4 : final_files = 7) :
  initial_files - final_files = 14 := by
  sorry

end files_deleted_l2528_252816


namespace optimal_purchase_plan_l2528_252830

/-- Represents the daily transportation capacity of machine A in tons -/
def machine_A_capacity : ℝ := 90

/-- Represents the daily transportation capacity of machine B in tons -/
def machine_B_capacity : ℝ := 100

/-- Represents the cost of machine A in yuan -/
def machine_A_cost : ℝ := 15000

/-- Represents the cost of machine B in yuan -/
def machine_B_cost : ℝ := 20000

/-- Represents the total number of machines to be purchased -/
def total_machines : ℕ := 30

/-- Represents the minimum daily transportation requirement in tons -/
def min_daily_transportation : ℝ := 2880

/-- Represents the maximum purchase amount in yuan -/
def max_purchase_amount : ℝ := 550000

/-- Represents the optimal number of A machines to purchase -/
def optimal_A_machines : ℕ := 12

/-- Represents the optimal number of B machines to purchase -/
def optimal_B_machines : ℕ := 18

/-- Represents the total purchase amount for the optimal plan in yuan -/
def optimal_purchase_amount : ℝ := 54000

theorem optimal_purchase_plan :
  (machine_B_capacity = machine_A_capacity + 10) ∧
  (450 / machine_A_capacity = 500 / machine_B_capacity) ∧
  (optimal_A_machines + optimal_B_machines = total_machines) ∧
  (optimal_A_machines * machine_A_capacity + optimal_B_machines * machine_B_capacity ≥ min_daily_transportation) ∧
  (optimal_A_machines * machine_A_cost + optimal_B_machines * machine_B_cost = optimal_purchase_amount) ∧
  (optimal_purchase_amount ≤ max_purchase_amount) ∧
  (∀ a b : ℕ, a + b = total_machines →
    a * machine_A_capacity + b * machine_B_capacity ≥ min_daily_transportation →
    a * machine_A_cost + b * machine_B_cost ≤ max_purchase_amount →
    a * machine_A_cost + b * machine_B_cost ≥ optimal_purchase_amount) := by
  sorry


end optimal_purchase_plan_l2528_252830


namespace sum_of_squared_coefficients_l2528_252899

/-- The original polynomial before multiplication by 3 -/
def original_poly (x : ℝ) : ℝ := x^4 + 2*x^3 + 5*x^2 + x + 2

/-- The expanded polynomial after multiplication by 3 -/
def expanded_poly (x : ℝ) : ℝ := 3 * (original_poly x)

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [3, 6, 15, 3, 6]

/-- Theorem: The sum of the squares of the coefficients of the expanded form of 3(x^4 + 2x^3 + 5x^2 + x + 2) is 315 -/
theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 315 := by
  sorry

end sum_of_squared_coefficients_l2528_252899


namespace arithmetic_sequences_ratio_l2528_252860

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio : 
  let seq1_sum := arithmetic_sum 5 3 59
  let seq2_sum := arithmetic_sum 4 4 64
  seq1_sum / seq2_sum = 19 / 17 := by sorry

end arithmetic_sequences_ratio_l2528_252860


namespace odd_function_symmetry_l2528_252851

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem odd_function_symmetry :
  is_odd f →
  is_increasing_on f 3 7 →
  f 4 = 5 →
  is_increasing_on f (-7) (-3) ∧ f (-4) = -5 := by
  sorry

end odd_function_symmetry_l2528_252851


namespace max_k_for_intersecting_circles_l2528_252822

/-- The maximum value of k for which a circle with radius 1 centered on the line y = kx - 2 
    intersects the circle x² + y² - 8x + 15 = 0 -/
theorem max_k_for_intersecting_circles : 
  ∃ (max_k : ℝ), max_k = 4/3 ∧ 
  (∀ k : ℝ, (∃ x y : ℝ, 
    y = k * x - 2 ∧ 
    (∃ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 = 1 ∧ 
      cx^2 + cy^2 - 8*cx + 15 = 0)) → 
    k ≤ max_k) ∧
  (∃ x y : ℝ, 
    y = max_k * x - 2 ∧ 
    (∃ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 = 1 ∧ 
      cx^2 + cy^2 - 8*cx + 15 = 0)) :=
sorry

end max_k_for_intersecting_circles_l2528_252822


namespace problem_solution_l2528_252891

-- Define the expression as a function of a, b, and x
def expression (a b x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

theorem problem_solution :
  (∀ x, expression 7 (-1) x = 2 * x^2 - 4 * x + 2) ∧
  (∀ x, expression 5 (-3) x = -6 * x + 2) ∧
  (∃ a b, ∀ x, expression a b x = 2) :=
by sorry

end problem_solution_l2528_252891


namespace phone_bill_increase_l2528_252855

theorem phone_bill_increase (original_monthly_bill : ℝ) (new_yearly_bill : ℝ) : 
  original_monthly_bill = 50 → 
  new_yearly_bill = 660 → 
  (new_yearly_bill / (12 * original_monthly_bill) - 1) * 100 = 10 := by
  sorry

end phone_bill_increase_l2528_252855


namespace ellipse_k_range_l2528_252896

-- Define the equation of the ellipse
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 4) + y^2 / (10 - k) = 1

-- Define the property of having foci on the x-axis
def foci_on_x_axis (k : ℝ) : Prop :=
  k - 4 > 0 ∧ 10 - k > 0 ∧ k - 4 > 10 - k

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, ellipse_equation x y k) ∧ foci_on_x_axis k ↔ 7 < k ∧ k < 10 :=
sorry

end ellipse_k_range_l2528_252896


namespace female_employees_count_l2528_252873

/-- Given a company with the following properties:
  1. There are 280 female managers.
  2. 2/5 of all employees are managers.
  3. 2/5 of all male employees are managers.
  Prove that the total number of female employees is 700. -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) :
  let female_managers : ℕ := 280
  let total_managers : ℕ := (2 * total_employees) / 5
  let male_managers : ℕ := (2 * male_employees) / 5
  total_managers = female_managers + male_managers →
  total_employees - male_employees = 700 := by
  sorry

end female_employees_count_l2528_252873


namespace trivia_game_total_score_l2528_252839

theorem trivia_game_total_score :
  let team_a : Int := 2
  let team_b : Int := 9
  let team_c : Int := 4
  let team_d : Int := -3
  let team_e : Int := 7
  let team_f : Int := 0
  let team_g : Int := 5
  let team_h : Int := -2
  team_a + team_b + team_c + team_d + team_e + team_f + team_g + team_h = 22 := by
  sorry

end trivia_game_total_score_l2528_252839


namespace fifteenth_prime_l2528_252808

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime : nthPrime 15 = 47 := by sorry

end fifteenth_prime_l2528_252808


namespace cookie_eating_contest_l2528_252862

theorem cookie_eating_contest (first_student second_student : ℚ) : 
  first_student = 5/6 → second_student = 2/3 → first_student - second_student = 1/6 := by
  sorry

end cookie_eating_contest_l2528_252862


namespace suzy_book_count_l2528_252804

/-- Calculates the final number of books Suzy has after three days of transactions. -/
def final_book_count (initial : ℕ) (wed_out : ℕ) (thur_in : ℕ) (thur_out : ℕ) (fri_in : ℕ) : ℕ :=
  initial - wed_out + thur_in - thur_out + fri_in

/-- Theorem stating that given the specific transactions over three days, 
    Suzy ends up with 80 books. -/
theorem suzy_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

end suzy_book_count_l2528_252804


namespace unique_seven_l2528_252803

/-- A function that returns true if the given positive integer n results in
    exactly one term with a rational coefficient in the binomial expansion
    of (√3x + ∛2)^n -/
def has_one_rational_term (n : ℕ+) : Prop :=
  ∃! r : ℕ, r ≤ n ∧ 3 ∣ r ∧ 2 ∣ (n - r)

/-- Theorem stating that 7 is the only positive integer satisfying the condition -/
theorem unique_seven : ∀ n : ℕ+, has_one_rational_term n ↔ n = 7 := by
  sorry

end unique_seven_l2528_252803


namespace ratio_equality_solution_l2528_252878

theorem ratio_equality_solution (x : ℝ) : 
  (4 + 2*x) / (6 + 3*x) = (2 + x) / (3 + 2*x) → x = 0 ∨ x = 4 := by
sorry

end ratio_equality_solution_l2528_252878


namespace pool_length_is_ten_l2528_252897

/-- Proves that the length of a rectangular pool is 10 feet given its width, depth, and volume. -/
theorem pool_length_is_ten (width : ℝ) (depth : ℝ) (volume : ℝ) :
  width = 8 →
  depth = 6 →
  volume = 480 →
  volume = width * depth * (10 : ℝ) :=
by
  sorry

#check pool_length_is_ten

end pool_length_is_ten_l2528_252897


namespace ellipse_properties_l2528_252861

/-- Properties of a specific ellipse -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_c : 2 = Real.sqrt (a^2 - b^2)
  h_slope : (b - 0) / (0 - a) = -Real.sqrt 3 / 3

/-- Theorem about the standard equation and a geometric property of the ellipse -/
theorem ellipse_properties (e : EllipseC) :
  (∃ (x y : ℝ), x^2 / 6 + y^2 / 2 = 1) ∧
  (∃ (F P M N : ℝ × ℝ),
    F.1 = 2 ∧ F.2 = 0 ∧
    P.1 = 3 ∧
    (M.1^2 / 6 + M.2^2 / 2 = 1) ∧
    (N.1^2 / 6 + N.2^2 / 2 = 1) ∧
    (M.2 - N.2) * (P.1 - F.1) = (P.2 - F.2) * (M.1 - N.1) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) / Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) ≤ Real.sqrt 3) :=
by sorry

end ellipse_properties_l2528_252861


namespace sum_35_25_base6_l2528_252871

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a base 6 number to a natural number --/
def to_nat (b : Base6) : Nat := sorry

/-- Converts a natural number to a base 6 number --/
def from_nat (n : Nat) : Base6 := sorry

/-- Adds two base 6 numbers --/
def add_base6 (a b : Base6) : Base6 := from_nat (to_nat a + to_nat b)

theorem sum_35_25_base6 :
  add_base6 (from_nat 35) (from_nat 25) = from_nat 104 := by sorry

end sum_35_25_base6_l2528_252871


namespace sixth_angle_measure_l2528_252820

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The given measures of five angles in the hexagon -/
def given_angles : List ℝ := [134, 108, 122, 99, 87]

/-- Theorem: In a hexagon where five of the interior angles measure 134°, 108°, 122°, 99°, and 87°, 
    the measure of the sixth angle is 170°. -/
theorem sixth_angle_measure :
  let sum_given_angles := given_angles.sum
  hexagon_angle_sum - sum_given_angles = 170 := by sorry

end sixth_angle_measure_l2528_252820


namespace toy_gift_box_discount_l2528_252834

theorem toy_gift_box_discount (cost_price marked_price discount profit_margin : ℝ) : 
  cost_price = 160 →
  marked_price = 240 →
  discount = 20 →
  profit_margin = 20 →
  marked_price * (1 - discount / 100) = cost_price * (1 + profit_margin / 100) :=
by sorry

end toy_gift_box_discount_l2528_252834


namespace helen_laundry_time_l2528_252813

/-- Represents the time spent on each activity for each item type -/
structure ItemTime where
  wash : Nat
  dry : Nat
  fold : Nat
  iron : Nat

/-- Calculates the total time spent on an item -/
def totalItemTime (item : ItemTime) : Nat :=
  item.wash + item.dry + item.fold + item.iron

/-- Represents the time Helen spends on her delicate items -/
structure HelenLaundryTime where
  silkPillowcases : ItemTime
  woolBlankets : ItemTime
  cashmereScarves : ItemTime
  washingInterval : Nat
  leapYear : Nat
  regularYear : Nat
  numRegularYears : Nat

/-- Calculates the total time Helen spends on laundry over the given period -/
def totalLaundryTime (h : HelenLaundryTime) : Nat :=
  let totalTimePerSession := totalItemTime h.silkPillowcases + totalItemTime h.woolBlankets + totalItemTime h.cashmereScarves
  let totalDays := h.leapYear + h.regularYear * h.numRegularYears
  let totalSessions := totalDays / h.washingInterval
  totalTimePerSession * totalSessions

theorem helen_laundry_time : 
  ∀ h : HelenLaundryTime,
    h.silkPillowcases = { wash := 30, dry := 20, fold := 10, iron := 5 } →
    h.woolBlankets = { wash := 45, dry := 30, fold := 15, iron := 20 } →
    h.cashmereScarves = { wash := 15, dry := 10, fold := 5, iron := 10 } →
    h.washingInterval = 28 →
    h.leapYear = 366 →
    h.regularYear = 365 →
    h.numRegularYears = 3 →
    totalLaundryTime h = 11180 := by
  sorry

end helen_laundry_time_l2528_252813


namespace negation_equivalence_l2528_252800

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l2528_252800


namespace campers_rowing_morning_l2528_252850

theorem campers_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total_campers : ℕ) :
  hiking_morning = 4 →
  rowing_afternoon = 26 →
  total_campers = 71 →
  total_campers - (hiking_morning + rowing_afternoon) = 41 :=
by
  sorry

end campers_rowing_morning_l2528_252850


namespace cricket_team_right_handed_players_l2528_252846

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 55)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : total_players - (total_players - throwers) / 3 = 49 :=
by sorry

end cricket_team_right_handed_players_l2528_252846


namespace equation_solution_exists_l2528_252885

theorem equation_solution_exists : ∃ x : ℤ, 
  |x - ((1125 - 500 + 660 - 200) * (3/2) * (3/4) / 45)| ≤ 1/2 := by
  sorry

end equation_solution_exists_l2528_252885


namespace perfect_square_property_l2528_252887

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem perfect_square_property : 
  (is_perfect_square (factorial 101 * 102 * 102)) ∧ 
  (¬ is_perfect_square (factorial 102 * 103 * 103)) ∧
  (¬ is_perfect_square (factorial 103 * 104 * 104)) ∧
  (¬ is_perfect_square (factorial 104 * 105 * 105)) ∧
  (¬ is_perfect_square (factorial 105 * 106 * 106)) :=
by sorry

end perfect_square_property_l2528_252887


namespace cost_of_dozen_pens_l2528_252880

/-- Given the cost of 3 pens and 5 pencils is Rs. 150, and the ratio of the cost of one pen
    to one pencil is 5:1, prove that the cost of one dozen pens is Rs. 450. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℝ) 
  (h1 : 3 * pen_cost + 5 * pencil_cost = 150)
  (h2 : pen_cost = 5 * pencil_cost) : 
  12 * pen_cost = 450 := by
  sorry

end cost_of_dozen_pens_l2528_252880


namespace real_part_of_complex_product_l2528_252847

theorem real_part_of_complex_product : ∃ z : ℂ, z = (1 - Complex.I) * (2 + Complex.I) ∧ z.re = 3 := by
  sorry

end real_part_of_complex_product_l2528_252847


namespace bullet_train_length_l2528_252828

/-- The length of a bullet train passing a man running in the opposite direction -/
theorem bullet_train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 59 →
  man_speed = 7 →
  passing_time = 12 →
  (train_speed + man_speed) * (1000 / 3600) * passing_time = 220 :=
by sorry

end bullet_train_length_l2528_252828


namespace sarah_savings_l2528_252863

/-- Represents Sarah's savings pattern over time -/
def savings_pattern : List (Nat × Nat) :=
  [(4, 5), (4, 10), (4, 20)]

/-- Calculates the total amount saved given a savings pattern -/
def total_saved (pattern : List (Nat × Nat)) : Nat :=
  pattern.foldl (fun acc (weeks, amount) => acc + weeks * amount) 0

/-- Calculates the total number of weeks in a savings pattern -/
def total_weeks (pattern : List (Nat × Nat)) : Nat :=
  pattern.foldl (fun acc (weeks, _) => acc + weeks) 0

/-- Theorem: Sarah saves $140 in 12 weeks -/
theorem sarah_savings : 
  total_saved savings_pattern = 140 ∧ total_weeks savings_pattern = 12 :=
sorry

end sarah_savings_l2528_252863


namespace negative_45_same_terminal_side_as_315_l2528_252884

def has_same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem negative_45_same_terminal_side_as_315 :
  has_same_terminal_side (-45 : ℝ) 315 :=
sorry

end negative_45_same_terminal_side_as_315_l2528_252884


namespace b_22_mod_35_l2528_252837

/-- Concatenates integers from 1 to n --/
def b (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem b_22_mod_35 : b 22 % 35 = 17 := by
  sorry

end b_22_mod_35_l2528_252837


namespace claire_balloons_l2528_252810

/-- The number of balloons Claire has at the end of the fair -/
def final_balloons (initial : ℕ) (floated_away : ℕ) (given_away : ℕ) (grabbed : ℕ) : ℕ :=
  initial - floated_away - given_away + grabbed

/-- Theorem stating that Claire ends up with 40 balloons -/
theorem claire_balloons : final_balloons 50 12 9 11 = 40 := by
  sorry

end claire_balloons_l2528_252810


namespace line_slope_angle_l2528_252854

/-- The slope angle of a line given by parametric equations -/
def slope_angle (x y : ℝ → ℝ) : ℝ := sorry

theorem line_slope_angle :
  let x : ℝ → ℝ := λ t => Real.sin θ + t * Real.sin (15 * π / 180)
  let y : ℝ → ℝ := λ t => Real.cos θ - t * Real.sin (75 * π / 180)
  slope_angle x y = 105 * π / 180 :=
sorry

end line_slope_angle_l2528_252854


namespace perfect_square_existence_l2528_252877

theorem perfect_square_existence : ∃ n : ℕ, 
  (10^199 - 10^100 : ℕ) < n^2 ∧ n^2 < 10^199 := by
sorry

end perfect_square_existence_l2528_252877


namespace minimum_crossing_time_l2528_252886

/-- Represents an individual with their crossing time -/
structure Individual where
  name : String
  time : Nat

/-- Represents a crossing of the bridge -/
inductive Crossing
  | Single : Individual → Crossing
  | Pair : Individual → Individual → Crossing

/-- Calculates the time taken for a single crossing -/
def crossingTime (c : Crossing) : Nat :=
  match c with
  | Crossing.Single i => i.time
  | Crossing.Pair i j => max i.time j.time

/-- The problem statement -/
theorem minimum_crossing_time
  (a b c d : Individual)
  (ha : a.time = 2)
  (hb : b.time = 3)
  (hc : c.time = 8)
  (hd : d.time = 10)
  (crossings : List Crossing)
  (hcross : crossings = [Crossing.Pair a b, Crossing.Single a, Crossing.Pair c d, Crossing.Single b, Crossing.Pair a b]) :
  (crossings.map crossingTime).sum = 21 ∧
  ∀ (otherCrossings : List Crossing),
    (otherCrossings.map crossingTime).sum ≥ 21 :=
by sorry

end minimum_crossing_time_l2528_252886


namespace least_number_of_trees_least_number_of_trees_is_168_l2528_252842

theorem least_number_of_trees : ℕ → Prop :=
  fun n => (n % 4 = 0) ∧ 
           (n % 7 = 0) ∧ 
           (n % 6 = 0) ∧ 
           (n % 4 = 0) ∧ 
           (n ≥ 100) ∧ 
           (∀ m : ℕ, m < n → ¬(least_number_of_trees m))

theorem least_number_of_trees_is_168 : 
  least_number_of_trees 168 := by sorry

end least_number_of_trees_least_number_of_trees_is_168_l2528_252842


namespace min_distance_squared_l2528_252883

theorem min_distance_squared (a b c d : ℝ) 
  (h : (b + 2 * a^2 - 6 * Real.log a)^2 + |2 * c - d + 6| = 0) :
  ∃ (m : ℝ), m = 20 ∧ ∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ m :=
sorry

end min_distance_squared_l2528_252883


namespace unpartnered_students_correct_l2528_252826

/-- Calculates the number of students unable to partner in square dancing --/
def unpartnered_students (class1_males class1_females class2_males class2_females class3_males class3_females : ℕ) : ℕ :=
  let total_males := class1_males + class2_males + class3_males
  let total_females := class1_females + class2_females + class3_females
  Int.natAbs (total_males - total_females)

/-- Theorem stating that the number of unpartnered students is correct --/
theorem unpartnered_students_correct 
  (class1_males class1_females class2_males class2_females class3_males class3_females : ℕ) :
  unpartnered_students class1_males class1_females class2_males class2_females class3_males class3_females =
  Int.natAbs ((class1_males + class2_males + class3_males) - (class1_females + class2_females + class3_females)) :=
by sorry

#eval unpartnered_students 17 13 14 18 15 17  -- Should evaluate to 2

end unpartnered_students_correct_l2528_252826


namespace colin_running_time_l2528_252811

def total_miles : ℕ := 4
def first_mile_time : ℕ := 6
def fourth_mile_time : ℕ := 4
def average_time : ℕ := 5

theorem colin_running_time (second_mile_time third_mile_time : ℕ) 
  (h1 : second_mile_time = third_mile_time) 
  (h2 : first_mile_time + second_mile_time + third_mile_time + fourth_mile_time = total_miles * average_time) : 
  second_mile_time = 5 ∧ third_mile_time = 5 := by
  sorry

#check colin_running_time

end colin_running_time_l2528_252811


namespace cube_volume_ratio_l2528_252814

/-- The ratio of volumes of two cubes, one with sides of 2 meters and another with sides of 100 centimeters. -/
theorem cube_volume_ratio : 
  let cube1_side : ℝ := 2  -- Side length of Cube 1 in meters
  let cube2_side : ℝ := 100 / 100  -- Side length of Cube 2 in meters (100 cm converted to m)
  let cube1_volume := cube1_side ^ 3
  let cube2_volume := cube2_side ^ 3
  cube1_volume / cube2_volume = 8 := by sorry

end cube_volume_ratio_l2528_252814


namespace red_and_green_peaches_count_l2528_252848

/-- Given a basket of peaches, prove that the total number of red and green peaches is 22. -/
theorem red_and_green_peaches_count (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 6)
  (h2 : green_peaches = 16) : 
  red_peaches + green_peaches = 22 := by
sorry

end red_and_green_peaches_count_l2528_252848


namespace product_expansion_sum_l2528_252825

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(9 - x) = a*x^3 + b*x^2 + c*x + d) → 
  9*a + 3*b + c + d = 58 := by
sorry

end product_expansion_sum_l2528_252825


namespace ad_purchase_cost_is_108000_l2528_252817

/-- Represents the dimensions of an ad space -/
structure AdSpace where
  length : ℝ
  width : ℝ

/-- Represents the cost and quantity information for ad purchases -/
structure AdPurchase where
  numCompanies : ℕ
  adSpacesPerCompany : ℕ
  adSpace : AdSpace
  costPerSquareFoot : ℝ

/-- Calculates the total cost of ad purchases for multiple companies -/
def totalAdCost (purchase : AdPurchase) : ℝ :=
  purchase.numCompanies * purchase.adSpacesPerCompany * 
  purchase.adSpace.length * purchase.adSpace.width * 
  purchase.costPerSquareFoot

/-- Theorem stating that the total cost for the given ad purchase scenario is $108,000 -/
theorem ad_purchase_cost_is_108000 : 
  totalAdCost {
    numCompanies := 3,
    adSpacesPerCompany := 10,
    adSpace := { length := 12, width := 5 },
    costPerSquareFoot := 60
  } = 108000 := by
  sorry

end ad_purchase_cost_is_108000_l2528_252817


namespace intersection_point_properties_l2528_252890

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := ((-1 : ℝ), (2 : ℝ))

-- Define the given line for perpendicularity
def perp_line (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Theorem statement
theorem intersection_point_properties :
  l₁ M.1 M.2 ∧ l₂ M.1 M.2 →
  (∀ x y : ℝ, y = -2 * x ↔ ∃ t : ℝ, x = t * M.1 ∧ y = t * M.2) ∧
  (∀ x y : ℝ, x - 2 * y + 5 = 0 ↔ (y - M.2 = (1/2) * (x - M.1) ∧ 
    ∃ a b : ℝ, perp_line a b ∧ (b - M.2) = (-2) * (a - M.1))) :=
by sorry

end intersection_point_properties_l2528_252890


namespace some_mystical_creatures_are_enchanted_beings_l2528_252876

-- Define the types
variable (U : Type) -- Universe of discourse
variable (Dragon : U → Prop)
variable (MysticalCreature : U → Prop)
variable (EnchantedBeing : U → Prop)

-- Define the premises
variable (h1 : ∀ x, Dragon x → MysticalCreature x)
variable (h2 : ∃ x, EnchantedBeing x ∧ Dragon x)

-- Theorem to prove
theorem some_mystical_creatures_are_enchanted_beings :
  ∃ x, MysticalCreature x ∧ EnchantedBeing x :=
sorry

end some_mystical_creatures_are_enchanted_beings_l2528_252876


namespace cyclic_quadrilateral_theorem_l2528_252894

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral :=
  (a : ℝ) -- Length of side a
  (b : ℝ) -- Length of side b
  (c : ℝ) -- Length of diagonal c
  (d : ℝ) -- Length of diagonal d
  (ha : a > 0) -- Side lengths are positive
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)

/-- In any cyclic quadrilateral, the sum of the squares of the sides 
    is equal to the sum of the squares of the diagonals. -/
theorem cyclic_quadrilateral_theorem (q : CyclicQuadrilateral) :
  q.c^2 + q.d^2 = 2 * (q.a^2 + q.b^2) := by
  sorry

end cyclic_quadrilateral_theorem_l2528_252894


namespace NaCl_selectively_precipitates_Ag_other_reagents_do_not_selectively_precipitate_Ag_l2528_252806

/-- Represents the solubility of a compound in water -/
inductive Solubility
  | Soluble
  | SlightlySoluble
  | Insoluble

/-- Represents a metal ion -/
inductive MetalIon
  | Ag
  | Mg
  | Sr

/-- Represents a reagent -/
inductive Reagent
  | NaCl
  | NaOH
  | Na2SO4
  | Na3PO4

/-- Returns the solubility of the compound formed by a metal ion and a reagent -/
def solubility (ion : MetalIon) (reagent : Reagent) : Solubility :=
  match ion, reagent with
  | MetalIon.Ag, Reagent.NaCl => Solubility.Insoluble
  | MetalIon.Mg, Reagent.NaCl => Solubility.Soluble
  | MetalIon.Sr, Reagent.NaCl => Solubility.Soluble
  | MetalIon.Ag, Reagent.NaOH => Solubility.Insoluble
  | MetalIon.Mg, Reagent.NaOH => Solubility.Insoluble
  | MetalIon.Sr, Reagent.NaOH => Solubility.SlightlySoluble
  | MetalIon.Ag, Reagent.Na2SO4 => Solubility.SlightlySoluble
  | MetalIon.Mg, Reagent.Na2SO4 => Solubility.Soluble
  | MetalIon.Sr, Reagent.Na2SO4 => Solubility.SlightlySoluble
  | MetalIon.Ag, Reagent.Na3PO4 => Solubility.Insoluble
  | MetalIon.Mg, Reagent.Na3PO4 => Solubility.Insoluble
  | MetalIon.Sr, Reagent.Na3PO4 => Solubility.Insoluble

/-- Checks if a reagent selectively precipitates Ag+ -/
def selectivelyPrecipitatesAg (reagent : Reagent) : Prop :=
  solubility MetalIon.Ag reagent = Solubility.Insoluble ∧
  solubility MetalIon.Mg reagent = Solubility.Soluble ∧
  solubility MetalIon.Sr reagent = Solubility.Soluble

theorem NaCl_selectively_precipitates_Ag :
  selectivelyPrecipitatesAg Reagent.NaCl :=
by sorry

theorem other_reagents_do_not_selectively_precipitate_Ag :
  ∀ r : Reagent, r ≠ Reagent.NaCl → ¬selectivelyPrecipitatesAg r :=
by sorry

end NaCl_selectively_precipitates_Ag_other_reagents_do_not_selectively_precipitate_Ag_l2528_252806


namespace food_expenditure_increase_l2528_252867

/-- Represents the annual income in thousand yuan -/
def annual_income : ℝ → ℝ := id

/-- Represents the annual food expenditure in thousand yuan -/
def annual_food_expenditure (x : ℝ) : ℝ := 2.5 * x + 3.2

/-- Theorem stating that when annual income increases by 1, 
    annual food expenditure increases by 2.5 -/
theorem food_expenditure_increase (x : ℝ) : 
  annual_food_expenditure (annual_income x + 1) - annual_food_expenditure (annual_income x) = 2.5 := by
  sorry

end food_expenditure_increase_l2528_252867


namespace peter_pictures_l2528_252802

theorem peter_pictures (peter_pictures : ℕ) (quincy_pictures : ℕ) (randy_pictures : ℕ)
  (h1 : quincy_pictures = peter_pictures + 20)
  (h2 : randy_pictures + peter_pictures + quincy_pictures = 41)
  (h3 : randy_pictures = 5) :
  peter_pictures = 8 := by
sorry

end peter_pictures_l2528_252802


namespace no_intersection_points_l2528_252831

theorem no_intersection_points : 
  ¬∃ (x y : ℝ), (9 * x^2 + y^2 = 9) ∧ (x^2 + 16 * y^2 = 16) := by
  sorry

end no_intersection_points_l2528_252831


namespace least_non_special_fraction_l2528_252870

/-- Represents a fraction in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are positive integers -/
def SpecialFraction (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ+), n = (2^a.val - 2^b.val) / (2^c.val - 2^d.val)

/-- The least positive integer that cannot be represented as a SpecialFraction is 11 -/
theorem least_non_special_fraction : (∀ k < 11, SpecialFraction k) ∧ ¬SpecialFraction 11 := by
  sorry

end least_non_special_fraction_l2528_252870


namespace mudits_age_l2528_252853

theorem mudits_age : ∃ (x : ℕ), x + 16 = 3 * (x - 4) ∧ x = 14 := by
  sorry

end mudits_age_l2528_252853


namespace five_digit_divisible_by_nine_l2528_252840

theorem five_digit_divisible_by_nine (B : ℕ) : 
  B < 10 →
  (40000 + 10000 * B + 500 + 20 + B) % 9 = 0 →
  B = 8 :=
by sorry

end five_digit_divisible_by_nine_l2528_252840


namespace complex_subtraction_l2528_252866

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 3 + 4*I) (h₂ : z₂ = 1 + I) : 
  z₁ - z₂ = 2 + 3*I := by
  sorry

end complex_subtraction_l2528_252866


namespace sum_positive_when_difference_exceeds_absolute_value_l2528_252833

theorem sum_positive_when_difference_exceeds_absolute_value
  (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end sum_positive_when_difference_exceeds_absolute_value_l2528_252833


namespace min_value_theorem_l2528_252856

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a + 2 / b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
sorry

end min_value_theorem_l2528_252856


namespace p_on_x_axis_p_on_line_through_a_m_in_third_quadrant_l2528_252849

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (2*m + 5, 3*m + 3)

-- Define point A
def A : ℝ × ℝ := (-5, 1)

-- Define point M as a function of m
def M (m : ℝ) : ℝ × ℝ := (2*m + 7, 3*m + 6)

-- Theorem 1
theorem p_on_x_axis (m : ℝ) : 
  (P m).2 = 0 → m = -1 := by sorry

-- Theorem 2
theorem p_on_line_through_a (m : ℝ) :
  (P m).1 = A.1 → P m = (-5, -12) := by sorry

-- Theorem 3
theorem m_in_third_quadrant (m : ℝ) :
  (M m).1 < 0 ∧ (M m).2 < 0 ∧ |(M m).1| = 7 → M m = (-7, -15) := by sorry

end p_on_x_axis_p_on_line_through_a_m_in_third_quadrant_l2528_252849


namespace proposition_2_l2528_252881

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem statement
theorem proposition_2 
  (m n : Line) (α β γ : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : perpendicular m β) 
  (h4 : parallel m α) : 
  plane_perpendicular α β :=
sorry

end proposition_2_l2528_252881


namespace inscribed_cone_volume_l2528_252827

/-- The volume of an inscribed cone in a larger cone -/
theorem inscribed_cone_volume 
  (H : ℝ) -- Height of the outer cone
  (α : ℝ) -- Angle between slant height and altitude of outer cone
  (h_pos : H > 0) -- Assumption that height is positive
  (α_range : 0 < α ∧ α < π/2) -- Assumption that α is between 0 and π/2
  : ∃ (V : ℝ), 
    -- V represents the volume of the inscribed cone
    -- The inscribed cone's vertex coincides with the center of the base of the outer cone
    -- The slant heights of both cones are mutually perpendicular
    V = (1/12) * π * H^3 * (Real.sin α)^2 * (Real.sin (2*α))^2 :=
by
  sorry

end inscribed_cone_volume_l2528_252827


namespace cubic_function_properties_l2528_252889

noncomputable def f (a b x : ℝ) := x^3 + 3*(a-1)*x^2 - 12*a*x + b

theorem cubic_function_properties (a b : ℝ) :
  let f := f a b
  ∃ (x₁ x₂ M N : ℝ),
    (∀ x, x ≠ x₁ → x ≠ x₂ → f x ≤ f x₁ ∨ f x ≥ f x₂) →
    (∃ m c, ∀ x, m*x - f x - c = 0 → x = 0 ∧ m = 24 ∧ c = 10) →
    (x₁ = 2 ∧ x₂ = 4 ∧ M = f x₁ ∧ N = f x₂ ∧ M = 10 ∧ N = 6) ∧
    (f 1 > f 2 → x₂ - x₁ = 4 → b = 10 →
      (∀ x, x ≤ -2 → f x ≤ f (-2)) ∧
      (∀ x, -2 ≤ x ∧ x ≤ 2 → f 2 ≤ f x) ∧
      (∀ x, 2 ≤ x → f x ≥ f 2) ∧
      M = 26 ∧ N = -6) :=
by sorry

end cubic_function_properties_l2528_252889


namespace lidia_remaining_money_l2528_252892

/-- Calculates the remaining money after buying apps -/
def remaining_money (app_cost : ℕ) (num_apps : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - app_cost * num_apps

/-- Proves that Lidia will have $6 left after buying apps -/
theorem lidia_remaining_money :
  let app_cost : ℕ := 4
  let num_apps : ℕ := 15
  let initial_money : ℕ := 66
  remaining_money app_cost num_apps initial_money = 6 := by
sorry

end lidia_remaining_money_l2528_252892


namespace quadratic_properties_l2528_252893

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_neg1 : a * (-1)^2 + b * (-1) + c = 0
  point_0 : c = -3
  point_1 : a * 1^2 + b * 1 + c = -4
  point_2 : a * 2^2 + b * 2 + c = -3
  point_3 : a * 3^2 + b * 3 + c = 0

/-- The theorem stating properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (∃ x y, f.a * x^2 + f.b * x + f.c = y ∧ ∀ t, f.a * t^2 + f.b * t + f.c ≥ y) ∧
  (f.a * x^2 + f.b * x + f.c = -4 ↔ x = 1) ∧
  (f.a * 5^2 + f.b * 5 + f.c = 12) ∧
  (∀ x > 1, ∀ y > x, f.a * y^2 + f.b * y + f.c > f.a * x^2 + f.b * x + f.c) := by
  sorry

end quadratic_properties_l2528_252893


namespace perpendicular_planes_from_line_l2528_252865

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- m is contained in α -/
def contained_in (m : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- m is perpendicular to β -/
def perpendicular_line_plane (m : Line3D) (β : Plane3D) : Prop :=
  sorry

/-- α is perpendicular to β -/
def perpendicular_planes (α β : Plane3D) : Prop :=
  sorry

/-- If a line m is contained in a plane α and is perpendicular to another plane β, 
    then α is perpendicular to β -/
theorem perpendicular_planes_from_line 
  (m : Line3D) (α β : Plane3D) : 
  contained_in m α → perpendicular_line_plane m β → perpendicular_planes α β :=
by
  sorry

end perpendicular_planes_from_line_l2528_252865


namespace optimal_candy_purchase_l2528_252875

/-- Represents a set of candies with its cost and quantity. -/
structure CandySet where
  cost : ℕ
  quantity : ℕ

/-- The problem setup -/
def candy_problem :=
  let set1 : CandySet := ⟨50, 25⟩
  let set2 : CandySet := ⟨180, 95⟩
  let set3 : CandySet := ⟨150, 80⟩
  let total_budget : ℕ := 2200
  (set1, set2, set3, total_budget)

/-- Calculate the total cost of the purchase -/
def total_cost (x y z : ℕ) : ℕ :=
  let (set1, set2, set3, _) := candy_problem
  x * set1.cost + y * set2.cost + z * set3.cost

/-- Calculate the total number of candies -/
def total_candies (x y z : ℕ) : ℕ :=
  let (set1, set2, set3, _) := candy_problem
  x * set1.quantity + y * set2.quantity + z * set3.quantity

/-- Check if the purchase is within budget -/
def within_budget (x y z : ℕ) : Prop :=
  let (_, _, _, budget) := candy_problem
  total_cost x y z ≤ budget

/-- The main theorem stating that (2, 5, 8) is the optimal solution -/
theorem optimal_candy_purchase :
  within_budget 2 5 8 ∧
  (∀ x y z : ℕ, within_budget x y z → total_candies x y z ≤ total_candies 2 5 8) :=
sorry

end optimal_candy_purchase_l2528_252875


namespace min_value_product_l2528_252888

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 9) :
  x^3 * y^3 * z^2 ≥ 1/27 :=
sorry

end min_value_product_l2528_252888


namespace hyperbola_eccentricity_l2528_252845

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ := sorry

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := sorry

/-- Theorem: If the distance from a focus to an asymptote is 1/4 of the focal distance,
    then the eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_dist : focus_to_asymptote_distance h = (1/4) * focal_distance h) : 
    eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l2528_252845


namespace quartic_to_quadratic_reduction_l2528_252857

/-- Given a quartic equation and a substitution, prove it can be reduced to two quadratic equations -/
theorem quartic_to_quadratic_reduction (a b c : ℝ) (x y : ℝ) :
  (a * x^4 + b * x^3 + c * x^2 + b * x + a = 0) →
  (y = x + 1/x) →
  ∃ (y₁ y₂ : ℝ),
    (a * y^2 + b * y + (c - 2*a) = 0) ∧
    (x^2 - y₁ * x + 1 = 0 ∨ x^2 - y₂ * x + 1 = 0) :=
by sorry

end quartic_to_quadratic_reduction_l2528_252857


namespace longer_worm_length_l2528_252858

/-- Given two worms, where one is 0.1 inch long and the other is 0.7 inches longer,
    prove that the longer worm is 0.8 inches long. -/
theorem longer_worm_length (short_worm long_worm : ℝ) 
  (h1 : short_worm = 0.1)
  (h2 : long_worm = short_worm + 0.7) :
  long_worm = 0.8 := by
  sorry

end longer_worm_length_l2528_252858


namespace geometric_sequence_sixth_term_l2528_252815

/-- A geometric sequence is a sequence where each term after the first is found by multiplying 
    the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℚ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum1 : a 1 + a 3 = 5/2) 
  (h_sum2 : a 2 + a 4 = 5/4) : 
  a 6 = 1/16 := by
sorry

end geometric_sequence_sixth_term_l2528_252815


namespace rulers_produced_l2528_252843

theorem rulers_produced (rulers_per_minute : ℕ) (minutes : ℕ) : 
  rulers_per_minute = 8 → minutes = 15 → rulers_per_minute * minutes = 120 := by
  sorry

end rulers_produced_l2528_252843


namespace p_necessary_not_sufficient_for_q_l2528_252874

theorem p_necessary_not_sufficient_for_q :
  ∀ x : ℝ,
  (∃ y : ℝ, y < 1 ∧ ¬((y + 2) * (y - 1) < 0)) ∧
  (∀ z : ℝ, (z + 2) * (z - 1) < 0 → z < 1) :=
by sorry

end p_necessary_not_sufficient_for_q_l2528_252874


namespace alpha_beta_equivalence_l2528_252868

theorem alpha_beta_equivalence (α β : ℝ) :
  (α + β > 0) ↔ (α + β > Real.cos α - Real.cos β) := by
  sorry

end alpha_beta_equivalence_l2528_252868


namespace fraction_equality_l2528_252898

theorem fraction_equality : (2018 + 2018 + 2018) / (2018 + 2018 + 2018 + 2018) = 3 / 4 := by
  sorry

end fraction_equality_l2528_252898


namespace custom_ops_simplification_and_evaluation_l2528_252812

/-- Custom addition operation for rational numbers -/
def star (a b : ℚ) : ℚ := a + b

/-- Custom subtraction operation for rational numbers -/
def otimes (a b : ℚ) : ℚ := a - b

/-- Theorem stating the simplification and evaluation of the given expression -/
theorem custom_ops_simplification_and_evaluation :
  ∀ a b : ℚ, 
  (star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b) = 6 * a^2 * b - a * b) ∧
  (star (5^2 * 3) (3 * 5 * 3) + otimes (5 * 5^2 * 3) (4 * 5 * 3) = 435) := by sorry

end custom_ops_simplification_and_evaluation_l2528_252812


namespace linear_equation_solution_l2528_252836

theorem linear_equation_solution (a : ℝ) :
  (1 : ℝ) * a + (-2 : ℝ) = (3 : ℝ) → a = (5 : ℝ) := by
  sorry

end linear_equation_solution_l2528_252836


namespace distance_to_stream_is_six_l2528_252835

/-- Represents a trapezoidal forest with a stream -/
structure TrapezidalForest where
  side1 : ℝ  -- Length of the side closest to Wendy's house
  side2 : ℝ  -- Length of the opposite parallel side
  area : ℝ   -- Total area of the forest
  stream_divides_in_half : Bool  -- Whether the stream divides the area in half

/-- The distance from either parallel side to the stream in the trapezoidal forest -/
def distance_to_stream (forest : TrapezidalForest) : ℝ :=
  sorry

/-- Theorem stating that the distance to the stream is 6 miles for the given forest -/
theorem distance_to_stream_is_six (forest : TrapezidalForest) 
  (h1 : forest.side1 = 8)
  (h2 : forest.side2 = 14)
  (h3 : forest.area = 132)
  (h4 : forest.stream_divides_in_half = true) :
  distance_to_stream forest = 6 :=
  sorry

end distance_to_stream_is_six_l2528_252835


namespace profit_increase_march_to_june_l2528_252819

/-- Calculates the total percent increase in profits from March to June given monthly changes -/
theorem profit_increase_march_to_june 
  (march_profit : ℝ) 
  (april_increase : ℝ) 
  (may_decrease : ℝ) 
  (june_increase : ℝ) 
  (h1 : april_increase = 0.4) 
  (h2 : may_decrease = 0.2) 
  (h3 : june_increase = 0.5) : 
  (((1 + june_increase) * (1 - may_decrease) * (1 + april_increase) - 1) * 100 = 68) := by
sorry

end profit_increase_march_to_june_l2528_252819


namespace complex_number_quadrant_l2528_252818

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 + 4*I) / (1 + I) ∧ (z.re > 0 ∧ z.im > 0) := by
  sorry

end complex_number_quadrant_l2528_252818


namespace typhoon_tree_problem_l2528_252823

theorem typhoon_tree_problem (initial_trees : ℕ) 
  (h1 : initial_trees = 13) 
  (dead_trees : ℕ) 
  (surviving_trees : ℕ) 
  (h2 : surviving_trees = dead_trees + 1) 
  (h3 : dead_trees + surviving_trees = initial_trees) : 
  dead_trees = 6 := by
sorry

end typhoon_tree_problem_l2528_252823


namespace quadratic_roots_sum_and_product_l2528_252879

theorem quadratic_roots_sum_and_product :
  let a : ℝ := 2
  let b : ℝ := -10
  let c : ℝ := 12
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots = 5 ∧ product_of_roots = 6 := by sorry

end quadratic_roots_sum_and_product_l2528_252879


namespace min_k_value_l2528_252859

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0)) →
  (∃ k_min : ℝ, k_min = -4 ∧ ∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0) → k ≥ k_min) :=
by sorry

end min_k_value_l2528_252859


namespace cross_section_distance_from_apex_l2528_252869

-- Define the structure of a right pentagonal pyramid
structure RightPentagonalPyramid where
  -- Add any necessary fields

-- Define a cross section of the pyramid
structure CrossSection where
  area : ℝ
  distanceFromApex : ℝ

-- Define the theorem
theorem cross_section_distance_from_apex 
  (pyramid : RightPentagonalPyramid)
  (section1 section2 : CrossSection)
  (h1 : section1.area = 125 * Real.sqrt 3)
  (h2 : section2.area = 500 * Real.sqrt 3)
  (h3 : section2.distanceFromApex - section1.distanceFromApex = 12)
  (h4 : section2.area > section1.area) :
  section2.distanceFromApex = 24 := by
sorry

end cross_section_distance_from_apex_l2528_252869


namespace quadratic_roots_difference_l2528_252864

theorem quadratic_roots_difference (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 :=
by sorry

end quadratic_roots_difference_l2528_252864


namespace watch_loss_percentage_l2528_252809

def watch_problem (selling_price_loss : ℝ) (selling_price_profit : ℝ) (profit_percentage : ℝ) : Prop :=
  let cost_price := selling_price_profit / (1 + profit_percentage / 100)
  let loss := cost_price - selling_price_loss
  let loss_percentage := (loss / cost_price) * 100
  selling_price_loss < cost_price ∧ 
  selling_price_profit > cost_price ∧
  loss_percentage = 5

theorem watch_loss_percentage : 
  watch_problem 1140 1260 5 := by sorry

end watch_loss_percentage_l2528_252809


namespace rabbit_logs_l2528_252838

theorem rabbit_logs (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  pieces - cuts = 6 := by
  sorry

end rabbit_logs_l2528_252838


namespace cos_to_sin_shift_l2528_252844

theorem cos_to_sin_shift (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) := by
  sorry

end cos_to_sin_shift_l2528_252844


namespace min_value_of_f_l2528_252807

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = -2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → f x ≥ min_val :=
by sorry

end min_value_of_f_l2528_252807
