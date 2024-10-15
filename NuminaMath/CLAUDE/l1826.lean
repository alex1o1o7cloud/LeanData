import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l1826_182695

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The equation of a hyperbola -/
def Hyperbola.equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of an asymptote of a hyperbola -/
def Hyperbola.asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_equation 4 3)
  (h_focus : h.a^2 - h.b^2 = 25) :
  h.equation = fun x y => x^2 / 16 - y^2 / 9 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1826_182695


namespace NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l1826_182692

theorem no_polyhedron_with_area_2015 : ¬ ∃ (n k : ℕ), 6 * n - 2 * k = 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l1826_182692


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_l1826_182693

theorem ice_cream_arrangement (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (n! / k!) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_l1826_182693


namespace NUMINAMATH_CALUDE_total_cds_l1826_182639

/-- The number of CDs each person has -/
structure CDCounts where
  dawn : ℕ
  kristine : ℕ
  mark : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (c : CDCounts) : Prop :=
  c.dawn = 10 ∧
  c.kristine = c.dawn + 7 ∧
  c.mark = 2 * c.kristine ∧
  c.alice = c.kristine + c.mark - 5

/-- The theorem to prove -/
theorem total_cds (c : CDCounts) (h : satisfiesConditions c) :
  c.dawn + c.kristine + c.mark + c.alice = 107 := by
  sorry

#check total_cds

end NUMINAMATH_CALUDE_total_cds_l1826_182639


namespace NUMINAMATH_CALUDE_total_difference_is_90q_minus_250_l1826_182650

/-- The total difference in money between Charles and Richard in cents -/
def total_difference (q : ℤ) : ℤ :=
  let charles_quarters := 6 * q + 2
  let charles_dimes := 3 * q - 2
  let richard_quarters := 2 * q + 10
  let richard_dimes := 4 * q + 3
  let quarter_value := 25
  let dime_value := 10
  (charles_quarters - richard_quarters) * quarter_value + 
  (charles_dimes - richard_dimes) * dime_value

theorem total_difference_is_90q_minus_250 (q : ℤ) : 
  total_difference q = 90 * q - 250 := by
  sorry

end NUMINAMATH_CALUDE_total_difference_is_90q_minus_250_l1826_182650


namespace NUMINAMATH_CALUDE_age_problem_l1826_182617

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = a - 3 →
  a + b + c + d = 44 →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1826_182617


namespace NUMINAMATH_CALUDE_complex_equation_unit_circle_l1826_182640

theorem complex_equation_unit_circle (z : ℂ) :
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 →
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_unit_circle_l1826_182640


namespace NUMINAMATH_CALUDE_last_remaining_is_125_l1826_182637

/-- Represents the marking process on a list of numbers -/
def markingProcess (n : ℕ) : ℕ → Prop :=
  sorry

/-- The last remaining number after the marking process -/
def lastRemaining (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the last remaining number is 125 when starting with 150 numbers -/
theorem last_remaining_is_125 : lastRemaining 150 = 125 :=
  sorry

end NUMINAMATH_CALUDE_last_remaining_is_125_l1826_182637


namespace NUMINAMATH_CALUDE_floor_abs_sum_l1826_182611

theorem floor_abs_sum : ⌊|(-3.7 : ℝ)|⌋ + |⌊(-3.7 : ℝ)⌋| = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l1826_182611


namespace NUMINAMATH_CALUDE_variance_of_binomial_distribution_l1826_182622

/-- The number of trials -/
def n : ℕ := 100

/-- The probability of success (drawing a second) -/
def p : ℝ := 0.02

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of the given binomial distribution is 1.96 -/
theorem variance_of_binomial_distribution :
  binomial_variance n p = 1.96 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_binomial_distribution_l1826_182622


namespace NUMINAMATH_CALUDE_equation_has_real_root_l1826_182612

theorem equation_has_real_root :
  ∃ x : ℝ, (Real.sqrt (x + 16) + 4 / Real.sqrt (x + 16) = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l1826_182612


namespace NUMINAMATH_CALUDE_bailey_points_l1826_182649

/-- 
Given four basketball players and their scoring relationships, 
prove that Bailey scored 14 points when the team's total score was 54.
-/
theorem bailey_points (bailey akiko michiko chandra : ℕ) : 
  chandra = 2 * akiko →
  akiko = michiko + 4 →
  michiko = bailey / 2 →
  bailey + akiko + michiko + chandra = 54 →
  bailey = 14 := by
sorry

end NUMINAMATH_CALUDE_bailey_points_l1826_182649


namespace NUMINAMATH_CALUDE_thursday_spending_l1826_182603

def monday_savings : ℕ := 15
def tuesday_savings : ℕ := 28
def wednesday_savings : ℕ := 13

def total_savings : ℕ := monday_savings + tuesday_savings + wednesday_savings

theorem thursday_spending :
  (total_savings : ℚ) / 2 = 28 := by sorry

end NUMINAMATH_CALUDE_thursday_spending_l1826_182603


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1826_182645

def P : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1826_182645


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l1826_182690

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_intersection_y_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 3 ∧ y₁ = 18) 
  (h_point2 : x₂ = -7 ∧ y₂ = -2) : 
  ∃ (y : ℝ), y = 12 ∧ 
  (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l1826_182690


namespace NUMINAMATH_CALUDE_linda_has_34_candies_l1826_182651

/-- The number of candies Linda and Chloe have together -/
def total_candies : ℕ := 62

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 28

/-- The number of candies Linda has -/
def linda_candies : ℕ := total_candies - chloe_candies

theorem linda_has_34_candies : linda_candies = 34 := by
  sorry

end NUMINAMATH_CALUDE_linda_has_34_candies_l1826_182651


namespace NUMINAMATH_CALUDE_cricket_overs_calculation_l1826_182625

theorem cricket_overs_calculation (total_target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : total_target = 262) (h2 : initial_rate = 3.2) 
  (h3 : required_rate = 5.75) (h4 : remaining_overs = 40) : 
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  total_target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_overs_calculation_l1826_182625


namespace NUMINAMATH_CALUDE_sector_angle_l1826_182688

/-- Given a circle with radius 12 meters and a sector with area 50.28571428571428 square meters,
    the angle at the center of the circle is 40 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 50.28571428571428) :
  (area * 360) / (π * r^2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1826_182688


namespace NUMINAMATH_CALUDE_binomial_parameters_determination_l1826_182634

/-- A random variable X following a Binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a Binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a Binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a Binomial distribution X with EX = 8 and DX = 1.6, n = 100 and p = 0.08 -/
theorem binomial_parameters_determination :
  ∀ X : BinomialDistribution, 
    expectedValue X = 8 → 
    variance X = 1.6 → 
    X.n = 100 ∧ X.p = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_binomial_parameters_determination_l1826_182634


namespace NUMINAMATH_CALUDE_smallest_group_size_l1826_182680

theorem smallest_group_size : ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, m > 2 ∧ n % m = 0) ∧ 
  n % 2 = 0 ∧
  (∀ k : ℕ, k > 0 ∧ (∃ l : ℕ, l > 2 ∧ k % l = 0) ∧ k % 2 = 0 → k ≥ n) ∧
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_group_size_l1826_182680


namespace NUMINAMATH_CALUDE_parking_lot_vehicles_l1826_182600

-- Define the initial number of cars and trucks
def initial_cars : ℝ := 14
def initial_trucks : ℝ := 49

-- Define the changes in the parking lot
def cars_left : ℕ := 3
def trucks_arrived : ℕ := 6

-- Define the ratios
def initial_ratio : ℝ := 3.5
def final_ratio : ℝ := 2.3

-- Theorem statement
theorem parking_lot_vehicles :
  -- Initial condition
  initial_cars = initial_ratio * initial_trucks ∧
  -- Final condition after changes
  (initial_cars - cars_left) = final_ratio * (initial_trucks + trucks_arrived) →
  -- Conclusion: Total number of vehicles originally parked
  initial_cars + initial_trucks = 63 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_parking_lot_vehicles_l1826_182600


namespace NUMINAMATH_CALUDE_cookies_sold_l1826_182624

/-- Proves the number of cookies sold given the problem conditions -/
theorem cookies_sold (original_cupcake_price original_cookie_price : ℚ)
  (price_reduction : ℚ) (cupcakes_sold : ℕ) (total_revenue : ℚ)
  (h1 : original_cupcake_price = 3)
  (h2 : original_cookie_price = 2)
  (h3 : price_reduction = 1/2)
  (h4 : cupcakes_sold = 16)
  (h5 : total_revenue = 32) :
  (total_revenue - cupcakes_sold * (original_cupcake_price * price_reduction)) / (original_cookie_price * price_reduction) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_l1826_182624


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1826_182629

/-- Given three rectangles with the following properties:
    Rectangle 1: length = 16 cm, width = 8 cm
    Rectangle 2: length = 1/2 of Rectangle 1's length, width = 1/2 of Rectangle 1's width
    Rectangle 3: length = 1/2 of Rectangle 2's length, width = 1/2 of Rectangle 2's width
    The perimeter of the figure formed by these rectangles is 60 cm. -/
theorem rectangle_perimeter (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) (rect3_length rect3_width : ℝ) :
  rect1_length = 16 ∧ 
  rect1_width = 8 ∧
  rect2_length = rect1_length / 2 ∧
  rect2_width = rect1_width / 2 ∧
  rect3_length = rect2_length / 2 ∧
  rect3_width = rect2_width / 2 →
  2 * (rect1_length + rect1_width + rect2_width + rect3_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1826_182629


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_value_l1826_182678

theorem unique_solution_implies_a_value (a : ℝ) :
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) → (a = 1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_value_l1826_182678


namespace NUMINAMATH_CALUDE_divisor_35_power_l1826_182660

theorem divisor_35_power (k : ℕ) : 35^k ∣ 1575320897 → 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_35_power_l1826_182660


namespace NUMINAMATH_CALUDE_inequality_solution_and_sum_of_roots_l1826_182631

-- Define the inequality
def inequality (m n x : ℝ) : Prop :=
  |x^2 + m*x + n| ≤ |3*x^2 - 6*x - 9|

-- Main theorem
theorem inequality_solution_and_sum_of_roots (m n : ℝ) 
  (h : ∀ x, inequality m n x) : 
  m = -2 ∧ n = -3 ∧ 
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = m - n → 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_inequality_solution_and_sum_of_roots_l1826_182631


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l1826_182668

/-- Represents the count of households in each income category -/
structure Population :=
  (high : ℕ)
  (middle : ℕ)
  (low : ℕ)

/-- Represents the sample sizes for each income category -/
structure Sample :=
  (high : ℕ)
  (middle : ℕ)
  (low : ℕ)

/-- Calculates the total population size -/
def totalPopulation (p : Population) : ℕ :=
  p.high + p.middle + p.low

/-- Checks if the sample sizes are proportional to the population sizes -/
def isProportionalSample (p : Population) (s : Sample) (sampleSize : ℕ) : Prop :=
  s.high * (totalPopulation p) = sampleSize * p.high ∧
  s.middle * (totalPopulation p) = sampleSize * p.middle ∧
  s.low * (totalPopulation p) = sampleSize * p.low

/-- The main theorem stating that the given sample is proportional for the given population -/
theorem stratified_sample_correct 
  (pop : Population) 
  (sample : Sample) : 
  pop.high = 150 → 
  pop.middle = 360 → 
  pop.low = 90 → 
  sample.high = 25 → 
  sample.middle = 60 → 
  sample.low = 15 → 
  isProportionalSample pop sample 100 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_correct_l1826_182668


namespace NUMINAMATH_CALUDE_more_triangles_2003_l1826_182699

/-- A triangle with integer sides -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of triangles with integer sides and perimeter 2000 -/
def Triangles2000 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2000}

/-- The set of triangles with integer sides and perimeter 2003 -/
def Triangles2003 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2003}

/-- Function that maps a triangle with perimeter 2000 to a triangle with perimeter 2003 -/
def f (t : IntTriangle) : IntTriangle :=
  ⟨t.a + 1, t.b + 1, t.c + 1, sorry⟩

theorem more_triangles_2003 :
  ∃ (g : Triangles2000 → Triangles2003), Function.Injective g ∧
  ∃ (t : Triangles2003), t ∉ Set.range g :=
sorry

end NUMINAMATH_CALUDE_more_triangles_2003_l1826_182699


namespace NUMINAMATH_CALUDE_complex_power_110_deg_36_l1826_182610

theorem complex_power_110_deg_36 :
  (Complex.exp (110 * π / 180 * Complex.I)) ^ 36 = -1/2 + Complex.I * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_110_deg_36_l1826_182610


namespace NUMINAMATH_CALUDE_skittles_distribution_l1826_182686

theorem skittles_distribution (num_friends : ℕ) (total_skittles : ℕ) 
  (h1 : num_friends = 5) 
  (h2 : total_skittles = 200) : 
  total_skittles / num_friends = 40 := by
  sorry

end NUMINAMATH_CALUDE_skittles_distribution_l1826_182686


namespace NUMINAMATH_CALUDE_close_interval_is_two_to_three_l1826_182684

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the close function property
def is_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Theorem statement
theorem close_interval_is_two_to_three :
  ∀ a b : ℝ, a ≤ 2 ∧ 3 ≤ b → (is_close f g a b ↔ a = 2 ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_close_interval_is_two_to_three_l1826_182684


namespace NUMINAMATH_CALUDE_acid_dilution_l1826_182666

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 40 ∧ 
  initial_concentration = 0.4 ∧ 
  water_added = 24 ∧ 
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_dilution_l1826_182666


namespace NUMINAMATH_CALUDE_train_crossing_time_l1826_182601

/-- Proves that a train with given length and speed takes the calculated time to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 360 →
  train_speed_kmh = 216 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1826_182601


namespace NUMINAMATH_CALUDE_butcher_purchase_cost_l1826_182606

/-- Calculates the total cost of a butcher's purchase given the weights and prices of various items. -/
theorem butcher_purchase_cost (steak_weight : ℚ) (steak_price : ℚ)
                               (chicken_weight : ℚ) (chicken_price : ℚ)
                               (sausage_weight : ℚ) (sausage_price : ℚ)
                               (pork_weight : ℚ) (pork_price : ℚ)
                               (bacon_weight : ℚ) (bacon_price : ℚ)
                               (salmon_weight : ℚ) (salmon_price : ℚ) :
  steak_weight = 3/2 ∧ steak_price = 15 ∧
  chicken_weight = 3/2 ∧ chicken_price = 8 ∧
  sausage_weight = 2 ∧ sausage_price = 13/2 ∧
  pork_weight = 7/2 ∧ pork_price = 10 ∧
  bacon_weight = 1/2 ∧ bacon_price = 9 ∧
  salmon_weight = 1/4 ∧ salmon_price = 30 →
  steak_weight * steak_price +
  chicken_weight * chicken_price +
  sausage_weight * sausage_price +
  pork_weight * pork_price +
  bacon_weight * bacon_price +
  salmon_weight * salmon_price = 189/2 := by
sorry


end NUMINAMATH_CALUDE_butcher_purchase_cost_l1826_182606


namespace NUMINAMATH_CALUDE_cd_purchase_remaining_money_l1826_182626

theorem cd_purchase_remaining_money (total_money : ℚ) (num_cds : ℕ) (cd_price : ℚ) :
  (total_money / 5 = num_cds / 3 * cd_price) →
  (total_money - num_cds * cd_price) / total_money = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cd_purchase_remaining_money_l1826_182626


namespace NUMINAMATH_CALUDE_work_completion_equality_first_group_size_l1826_182618

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 20

/-- The number of men in the second group -/
def men_second_group : ℕ := 12

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 30

/-- The number of men in the first group -/
def men_first_group : ℕ := 18

theorem work_completion_equality :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

theorem first_group_size :
  men_first_group = (men_second_group * days_second_group) / days_first_group :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_first_group_size_l1826_182618


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l1826_182670

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l1826_182670


namespace NUMINAMATH_CALUDE_proportion_problem_l1826_182659

theorem proportion_problem (x : ℝ) : x / 12 = 9 / 360 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1826_182659


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1826_182667

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) :
  Nat.lcm X Y = 180 →
  (X : ℚ) / (Y : ℚ) = 2 / 5 →
  Nat.gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1826_182667


namespace NUMINAMATH_CALUDE_assign_teachers_count_l1826_182633

/-- The number of ways to assign 6 teachers to 4 grades -/
def assign_teachers : ℕ :=
  let n_teachers : ℕ := 6
  let n_grades : ℕ := 4
  let two_specific_teachers : ℕ := 2
  -- Define the function to calculate the number of ways
  sorry

/-- Theorem stating that the number of ways to assign teachers is 240 -/
theorem assign_teachers_count : assign_teachers = 240 := by
  sorry

end NUMINAMATH_CALUDE_assign_teachers_count_l1826_182633


namespace NUMINAMATH_CALUDE_reciprocal_negative_four_l1826_182623

theorem reciprocal_negative_four (x : ℚ) : x⁻¹ = -4 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_negative_four_l1826_182623


namespace NUMINAMATH_CALUDE_hanoi_moves_correct_l1826_182685

/-- The minimum number of moves required to solve the Tower of Hanoi problem with n disks -/
def hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: The minimum number of moves required to solve the Tower of Hanoi problem with n disks is 2^n - 1 -/
theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_hanoi_moves_correct_l1826_182685


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1826_182616

/-- Given a polynomial p(x) = ax³ + bx² + cx + d -/
def p (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_remainder_theorem (a b c d : ℝ) :
  (∃ q₁ : ℝ → ℝ, ∀ x, p a b c d x = (x - 1) * q₁ x + 1) →
  (∃ q₂ : ℝ → ℝ, ∀ x, p a b c d x = (x - 2) * q₂ x + 3) →
  ∃ q : ℝ → ℝ, ∀ x, p a b c d x = (x - 1) * (x - 2) * q x + (2 * x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1826_182616


namespace NUMINAMATH_CALUDE_jasons_tip_is_two_dollars_l1826_182619

/-- Calculates the tip amount given the check amount, tax rate, and customer payment. -/
def calculate_tip (check_amount : ℝ) (tax_rate : ℝ) (customer_payment : ℝ) : ℝ :=
  let total_with_tax := check_amount * (1 + tax_rate)
  customer_payment - total_with_tax

/-- Proves that given the specific conditions, Jason's tip is $2.00 -/
theorem jasons_tip_is_two_dollars :
  calculate_tip 15 0.2 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jasons_tip_is_two_dollars_l1826_182619


namespace NUMINAMATH_CALUDE_polygon_d_largest_area_l1826_182641

-- Define the structure of a polygon
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

-- Define the area calculation function
def area (p : Polygon) : ℚ :=
  p.unitSquares + p.rightTriangles / 2

-- Define the five polygons
def polygonA : Polygon := ⟨6, 0⟩
def polygonB : Polygon := ⟨3, 4⟩
def polygonC : Polygon := ⟨4, 5⟩
def polygonD : Polygon := ⟨7, 0⟩
def polygonE : Polygon := ⟨2, 6⟩

-- Define the list of all polygons
def allPolygons : List Polygon := [polygonA, polygonB, polygonC, polygonD, polygonE]

-- Theorem: Polygon D has the largest area
theorem polygon_d_largest_area :
  ∀ p ∈ allPolygons, area polygonD ≥ area p :=
sorry

end NUMINAMATH_CALUDE_polygon_d_largest_area_l1826_182641


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1826_182636

theorem inequality_system_solution_set :
  {x : ℝ | (6 - 2*x ≥ 0) ∧ (2*x + 4 > 0)} = {x : ℝ | -2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1826_182636


namespace NUMINAMATH_CALUDE_marys_friends_ages_sum_l1826_182665

theorem marys_friends_ages_sum : 
  ∀ (a b c d : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit positive integers
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct
    ((a * b = 28 ∧ c * d = 45) ∨ (a * c = 28 ∧ b * d = 45) ∨ 
     (a * d = 28 ∧ b * c = 45) ∨ (b * c = 28 ∧ a * d = 45) ∨ 
     (b * d = 28 ∧ a * c = 45) ∨ (c * d = 28 ∧ a * b = 45)) →
    a + b + c + d = 25 := by
  sorry

end NUMINAMATH_CALUDE_marys_friends_ages_sum_l1826_182665


namespace NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l1826_182691

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def total_lamps : ℕ := num_red_lamps + num_blue_lamps

def probability_specific_arrangement : ℚ :=
  1 / 49

theorem specific_lamp_arrangement_probability :
  probability_specific_arrangement = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l1826_182691


namespace NUMINAMATH_CALUDE_solution_sets_l1826_182694

theorem solution_sets (p q : ℝ) : 
  let A := {x : ℝ | 2 * x^2 + x + p = 0}
  let B := {x : ℝ | 2 * x^2 + q * x + 2 = 0}
  (A ∩ B = {1/2}) → 
  (A = {-1, 1/2} ∧ B = {2, 1/2} ∧ A ∪ B = {-1, 2, 1/2}) := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_l1826_182694


namespace NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l1826_182608

/-- Represents a cube formed by unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_area := cube.size ^ 2
  let total_faces := 3 * face_area
  let shared_edges := 3 * (cube.size - 1)
  let corner_cube := 1
  total_faces - shared_edges + corner_cube

/-- Theorem stating that for a 9x9x9 cube, the maximum number of visible unit cubes is 220 -/
theorem max_visible_cubes_9x9x9 :
  max_visible_cubes ⟨9⟩ = 220 := by
  sorry

#eval max_visible_cubes ⟨9⟩

end NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l1826_182608


namespace NUMINAMATH_CALUDE_roses_before_and_after_cutting_l1826_182605

/-- Given the initial conditions of Mary's rose garden, prove the number of roses before and after cutting. -/
theorem roses_before_and_after_cutting 
  (R : ℕ) -- Initial number of roses in the garden
  (B : ℕ) -- Number of roses left in the garden after cutting
  (h1 : R = B + 10) -- Relation between R and B
  (h2 : ∃ C : ℕ, C = 10 ∧ R - C = B) -- Existence of C satisfying the conditions
  : R = B + 10 ∧ R - 10 = B := by
  sorry

end NUMINAMATH_CALUDE_roses_before_and_after_cutting_l1826_182605


namespace NUMINAMATH_CALUDE_interest_groups_participation_l1826_182672

theorem interest_groups_participation (total_students : ℕ) (total_participants : ℕ) 
  (sports_and_literature : ℕ) (sports_and_math : ℕ) (literature_and_math : ℕ) (all_three : ℕ) :
  total_students = 120 →
  total_participants = 135 →
  sports_and_literature = 15 →
  sports_and_math = 10 →
  literature_and_math = 8 →
  all_three = 4 →
  total_students - (total_participants - sports_and_literature - sports_and_math - literature_and_math + all_three) = 14 :=
by sorry

end NUMINAMATH_CALUDE_interest_groups_participation_l1826_182672


namespace NUMINAMATH_CALUDE_article_selling_price_l1826_182664

def cost_price : ℝ := 250
def profit_percentage : ℝ := 0.60

def selling_price : ℝ := cost_price + (profit_percentage * cost_price)

theorem article_selling_price : selling_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_article_selling_price_l1826_182664


namespace NUMINAMATH_CALUDE_notebook_count_l1826_182657

theorem notebook_count (n : ℕ) 
  (h1 : n > 0)
  (h2 : n^2 + 20 = (n + 1)^2 - 9) : 
  n^2 + 20 = 216 :=
by sorry

end NUMINAMATH_CALUDE_notebook_count_l1826_182657


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1826_182648

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1826_182648


namespace NUMINAMATH_CALUDE_f_s_not_multiplicative_other_l1826_182655

/-- r_s(n) is the number of solutions to x_1^2 + x_2^2 + ... + x_s^2 = n in integers x_1, x_2, ..., x_s -/
def r_s (s : ℕ) (n : ℕ) : ℕ := sorry

/-- f_s(n) = (2s)^(-1) * r_s(n) -/
def f_s (s : ℕ) (n : ℕ) : ℚ :=
  (2 * s : ℚ)⁻¹ * (r_s s n : ℚ)

/-- f_s is multiplicative for s = 1, 2, 4, 8 -/
axiom f_s_multiplicative_special (s : ℕ) (m n : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 4 ∨ s = 8) :
  Nat.Coprime m n → f_s s (m * n) = f_s s m * f_s s n

/-- f_s is not multiplicative for any other value of s -/
theorem f_s_not_multiplicative_other (s : ℕ) (h : s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8) :
  ∃ m n : ℕ, Nat.Coprime m n ∧ f_s s (m * n) ≠ f_s s m * f_s s n := by
  sorry

end NUMINAMATH_CALUDE_f_s_not_multiplicative_other_l1826_182655


namespace NUMINAMATH_CALUDE_eraser_difference_l1826_182687

/-- Proves that the difference between Rachel's erasers and one-half of Tanya's red erasers is 5 -/
theorem eraser_difference (tanya_total : ℕ) (tanya_red : ℕ) (rachel : ℕ) 
  (h1 : tanya_total = 20)
  (h2 : tanya_red = tanya_total / 2)
  (h3 : rachel = tanya_red) :
  rachel - tanya_red / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eraser_difference_l1826_182687


namespace NUMINAMATH_CALUDE_divisibility_property_l1826_182644

theorem divisibility_property (m : ℕ+) (x : ℝ) :
  ∃ k : ℝ, (x + 1)^(2 * m.val) - x^(2 * m.val) - 2*x - 1 = k * (x * (x + 1) * (2*x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1826_182644


namespace NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l1826_182675

theorem odd_multiple_of_nine_is_multiple_of_three (S : ℤ) :
  Odd S → (∃ k : ℤ, S = 9 * k) → (∃ m : ℤ, S = 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l1826_182675


namespace NUMINAMATH_CALUDE_fuel_consumption_population_l1826_182654

/-- Represents a car model -/
structure CarModel where
  name : String

/-- Represents a car of a specific model -/
structure Car where
  model : CarModel

/-- Represents fuel consumption measurement -/
structure FuelConsumption where
  amount : ℝ
  distance : ℝ

/-- Represents a survey of fuel consumption -/
structure FuelConsumptionSurvey where
  model : CarModel
  sample_size : ℕ
  measurements : List FuelConsumption

/-- Definition of population for a fuel consumption survey -/
def survey_population (survey : FuelConsumptionSurvey) : Set FuelConsumption :=
  {fc | ∃ (car : Car), car.model = survey.model ∧ fc.distance = 100}

theorem fuel_consumption_population 
  (survey : FuelConsumptionSurvey) 
  (h1 : survey.sample_size = 20) 
  (h2 : ∀ fc ∈ survey.measurements, fc.distance = 100) :
  survey_population survey = 
    {fc | ∃ (car : Car), car.model = survey.model ∧ fc.distance = 100} := by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_population_l1826_182654


namespace NUMINAMATH_CALUDE_distance_washington_to_idaho_l1826_182602

/-- The distance from Washington to Idaho in miles -/
def distance_WI : ℝ := 640

/-- The distance from Idaho to Nevada in miles -/
def distance_IN : ℝ := 550

/-- The speed from Washington to Idaho in miles per hour -/
def speed_WI : ℝ := 80

/-- The speed from Idaho to Nevada in miles per hour -/
def speed_IN : ℝ := 50

/-- The total travel time in hours -/
def total_time : ℝ := 19

/-- Theorem stating that the distance from Washington to Idaho is 640 miles -/
theorem distance_washington_to_idaho : 
  distance_WI = 640 ∧ 
  distance_WI / speed_WI + distance_IN / speed_IN = total_time := by
  sorry


end NUMINAMATH_CALUDE_distance_washington_to_idaho_l1826_182602


namespace NUMINAMATH_CALUDE_sqrt_two_div_sqrt_half_equals_two_l1826_182696

theorem sqrt_two_div_sqrt_half_equals_two : 
  Real.sqrt 2 / Real.sqrt (1/2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_div_sqrt_half_equals_two_l1826_182696


namespace NUMINAMATH_CALUDE_reflection_over_y_eq_neg_x_l1826_182630

/-- Reflects a point (x, y) over the line y = -x -/
def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

/-- The original point -/
def original_point : ℝ × ℝ := (7, -3)

/-- The expected reflected point -/
def expected_reflected_point : ℝ × ℝ := (3, -7)

theorem reflection_over_y_eq_neg_x :
  reflect_over_y_eq_neg_x original_point = expected_reflected_point := by
  sorry

end NUMINAMATH_CALUDE_reflection_over_y_eq_neg_x_l1826_182630


namespace NUMINAMATH_CALUDE_cube_equal_angle_planes_l1826_182681

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Calculates the angle between a plane and a line -/
def angle_plane_line (p : Plane) (l : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Checks if a plane passes through a given point -/
def plane_through_point (p : Plane) (point : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Theorem: There are exactly 4 planes through vertex A of a cube such that 
    the angles between each plane and the lines AB, AD, and AA₁ are all equal -/
theorem cube_equal_angle_planes (c : Cube) : 
  ∃! (planes : Finset Plane), 
    planes.card = 4 ∧ 
    ∀ p ∈ planes, 
      plane_through_point p (c.vertices 0) ∧
      ∃ θ : ℝ, 
        angle_plane_line p (c.vertices 0, c.vertices 1) = θ ∧
        angle_plane_line p (c.vertices 0, c.vertices 3) = θ ∧
        angle_plane_line p (c.vertices 0, c.vertices 4) = θ :=
  sorry

end NUMINAMATH_CALUDE_cube_equal_angle_planes_l1826_182681


namespace NUMINAMATH_CALUDE_bedroom_set_final_price_l1826_182642

/-- Calculates the final price of a bedroom set after discounts and gift card application --/
def final_price (initial_price gift_card first_discount second_discount : ℚ) : ℚ :=
  let price_after_first_discount := initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount - gift_card

/-- Theorem: The final price of the bedroom set is $1330 --/
theorem bedroom_set_final_price :
  final_price 2000 200 0.15 0.10 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_set_final_price_l1826_182642


namespace NUMINAMATH_CALUDE_golu_travel_distance_l1826_182658

theorem golu_travel_distance (x : ℝ) :
  x > 0 ∧ x^2 + 6^2 = 10^2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_golu_travel_distance_l1826_182658


namespace NUMINAMATH_CALUDE_gingerbread_problem_l1826_182673

theorem gingerbread_problem (total : ℕ) (red_hats blue_boots both : ℕ) : 
  red_hats = 6 →
  blue_boots = 9 →
  2 * red_hats = total →
  both = red_hats + blue_boots - total →
  both = 3 := by
sorry

end NUMINAMATH_CALUDE_gingerbread_problem_l1826_182673


namespace NUMINAMATH_CALUDE_prime_quadratic_equation_solution_l1826_182689

theorem prime_quadratic_equation_solution (a b Q R : ℕ) : 
  Nat.Prime a → 
  Nat.Prime b → 
  a ≠ b → 
  a^2 - a*Q + R = 0 → 
  b^2 - b*Q + R = 0 → 
  R = 6 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_equation_solution_l1826_182689


namespace NUMINAMATH_CALUDE_nathalie_cake_fraction_l1826_182604

theorem nathalie_cake_fraction (cake_weight : ℝ) (num_parts : ℕ) 
  (pierre_amount : ℝ) (nathalie_fraction : ℝ) : 
  cake_weight = 400 →
  num_parts = 8 →
  pierre_amount = 100 →
  pierre_amount = 2 * (nathalie_fraction * cake_weight) →
  nathalie_fraction = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_nathalie_cake_fraction_l1826_182604


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_ten_l1826_182669

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of a tetrahedron PQRS with given edge lengths is 10 -/
theorem tetrahedron_volume_is_ten :
  let t : Tetrahedron := {
    pq := 3,
    pr := 5,
    ps := 6,
    qr := 4,
    qs := Real.sqrt 26,
    rs := 5
  }
  tetrahedronVolume t = 10 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_is_ten_l1826_182669


namespace NUMINAMATH_CALUDE_largest_common_term_l1826_182652

def is_in_first_sequence (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k + 3

def is_in_second_sequence (n : ℕ) : Prop := ∃ m : ℕ, n = 7 * m + 5

theorem largest_common_term :
  (∃ n : ℕ, n < 1000 ∧ is_in_first_sequence n ∧ is_in_second_sequence n) ∧
  (∀ n : ℕ, n < 1000 ∧ is_in_first_sequence n ∧ is_in_second_sequence n → n ≤ 989) ∧
  (is_in_first_sequence 989 ∧ is_in_second_sequence 989) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l1826_182652


namespace NUMINAMATH_CALUDE_jerica_louis_age_ratio_l1826_182607

theorem jerica_louis_age_ratio :
  ∀ (jerica_age louis_age matilda_age : ℕ),
    louis_age = 14 →
    matilda_age = 35 →
    matilda_age = jerica_age + 7 →
    ∃ k : ℕ, jerica_age = k * louis_age →
    jerica_age / louis_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerica_louis_age_ratio_l1826_182607


namespace NUMINAMATH_CALUDE_translation_sum_l1826_182656

/-- Given two points P and Q in a 2D plane, where P is translated m units left
    and n units up to obtain Q, prove that m + n = 4. -/
theorem translation_sum (P Q : ℝ × ℝ) (m n : ℝ) : 
  P = (-1, -3) → Q = (-2, 0) → Q.1 = P.1 - m → Q.2 = P.2 + n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_translation_sum_l1826_182656


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1826_182609

/-- Given a train crossing a bridge, calculate the length of the bridge -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1826_182609


namespace NUMINAMATH_CALUDE_count_valid_plans_l1826_182632

/-- Represents a teacher --/
inductive Teacher : Type
  | A | B | C | D | E

/-- Represents a remote area --/
inductive Area : Type
  | One | Two | Three

/-- A dispatch plan assigns teachers to areas --/
def DispatchPlan := Teacher → Area

/-- Checks if a dispatch plan is valid according to the given conditions --/
def isValidPlan (plan : DispatchPlan) : Prop :=
  (∀ a : Area, ∃ t : Teacher, plan t = a) ∧  -- Each area has at least 1 person
  (plan Teacher.A ≠ plan Teacher.B) ∧        -- A and B are not in the same area
  (plan Teacher.A = plan Teacher.C)          -- A and C are in the same area

/-- The number of valid dispatch plans --/
def numValidPlans : ℕ := sorry

/-- Theorem stating that the number of valid dispatch plans is 30 --/
theorem count_valid_plans : numValidPlans = 30 := by sorry

end NUMINAMATH_CALUDE_count_valid_plans_l1826_182632


namespace NUMINAMATH_CALUDE_toy_poodle_height_is_14_l1826_182647

def standard_poodle_height : ℕ := 28

def height_difference_standard_miniature : ℕ := 8

def height_difference_miniature_toy : ℕ := 6

def toy_poodle_height : ℕ := standard_poodle_height - height_difference_standard_miniature - height_difference_miniature_toy

theorem toy_poodle_height_is_14 : toy_poodle_height = 14 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_is_14_l1826_182647


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1826_182653

theorem inequality_solution_set (x : ℝ) :
  (5 * x - 2 ≤ 3 * (1 + x)) ↔ (x ≤ 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1826_182653


namespace NUMINAMATH_CALUDE_integral_polynomial_l1826_182682

variables (a b c p : ℝ) (x : ℝ)

theorem integral_polynomial (a b c p : ℝ) (x : ℝ) :
  deriv (fun x => (a/4) * x^4 + (b/3) * x^3 + (c/2) * x^2 + p * x) x
  = a * x^3 + b * x^2 + c * x + p :=
by sorry

end NUMINAMATH_CALUDE_integral_polynomial_l1826_182682


namespace NUMINAMATH_CALUDE_identity_mapping_implies_sum_l1826_182620

theorem identity_mapping_implies_sum (a b : ℝ) : 
  let M : Set ℝ := {-1, b/a, 1}
  let N : Set ℝ := {a, b, b-a}
  (∀ x ∈ M, x ∈ N) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_identity_mapping_implies_sum_l1826_182620


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1826_182671

/-- The coefficient of x^r in the expansion of (1 + ax)^n -/
def binomialCoefficient (n : ℕ) (a : ℝ) (r : ℕ) : ℝ :=
  a^r * (n.choose r)

theorem binomial_expansion_coefficient (n : ℕ) :
  binomialCoefficient n 3 2 = 54 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1826_182671


namespace NUMINAMATH_CALUDE_solve_for_c_l1826_182638

theorem solve_for_c (x y c : ℝ) (h1 : x / (2 * y) = 5 / 2) (h2 : (7 * x + 4 * y) / c = 13) :
  c = 3 * y := by
sorry

end NUMINAMATH_CALUDE_solve_for_c_l1826_182638


namespace NUMINAMATH_CALUDE_perfectSquareFactorsOf360_l1826_182646

def perfectSquareFactors (n : ℕ) : ℕ := sorry

theorem perfectSquareFactorsOf360 : perfectSquareFactors 360 = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfectSquareFactorsOf360_l1826_182646


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1826_182698

/-- Given two circles R and S, if the diameter of R is 80% of the diameter of S,
    then the area of R is 64% of the area of S. -/
theorem circle_area_ratio (R S : Real) (hdiameter : R = 0.8 * S) :
  (π * (R / 2)^2) / (π * (S / 2)^2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1826_182698


namespace NUMINAMATH_CALUDE_mathematics_collections_l1826_182627

def word : String := "MATHEMATICS"

def num_vowels : Nat := 4
def num_consonants : Nat := 7
def num_ts : Nat := 2

def vowels_fall_off : Nat := 3
def consonants_fall_off : Nat := 4

def distinct_collections : Nat := 220

theorem mathematics_collections :
  (word.length = num_vowels + num_consonants) →
  (num_vowels = 4) →
  (num_consonants = 7) →
  (num_ts = 2) →
  (vowels_fall_off = 3) →
  (consonants_fall_off = 4) →
  distinct_collections = 220 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_collections_l1826_182627


namespace NUMINAMATH_CALUDE_greatest_divisor_of_exponential_sum_l1826_182697

theorem greatest_divisor_of_exponential_sum :
  ∃ (x : ℕ), x > 0 ∧
  (∀ (y : ℕ), y > 0 → (7^y + 12*y - 1) % x = 0) ∧
  (∀ (z : ℕ), z > x → ∃ (w : ℕ), w > 0 ∧ (7^w + 12*w - 1) % z ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_exponential_sum_l1826_182697


namespace NUMINAMATH_CALUDE_unique_divisible_by_seven_l1826_182661

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 110000 ∧ n % 100 = 1 ∧ (n / 100) % 10 ≠ 0

theorem unique_divisible_by_seven :
  ∃! n : ℕ, is_valid_number n ∧ n % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_seven_l1826_182661


namespace NUMINAMATH_CALUDE_typing_contest_orders_l1826_182643

/-- The number of different possible orders for a given number of participants to finish a contest without ties. -/
def numberOfOrders (n : ℕ) : ℕ := Nat.factorial n

/-- The number of participants in the typing contest. -/
def numberOfParticipants : ℕ := 4

theorem typing_contest_orders :
  numberOfOrders numberOfParticipants = 24 := by
  sorry

end NUMINAMATH_CALUDE_typing_contest_orders_l1826_182643


namespace NUMINAMATH_CALUDE_function_properties_l1826_182676

noncomputable section

variable (I : Set ℝ)
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem function_properties
  (h1 : ∀ x ∈ I, 0 < f' x ∧ f' x < 2)
  (h2 : ∀ x ∈ I, f' x ≠ 1)
  (h3 : ∃ c₁ ∈ I, f c₁ = c₁)
  (h4 : ∃ c₂ ∈ I, f c₂ = 2 * c₂)
  (h5 : ∀ a b, a ∈ I → b ∈ I → a ≤ b → ∃ x ∈ Set.Ioo a b, f b - f a = (b - a) * f' x) :
  (∀ x ∈ I, f x = x → x = Classical.choose h3) ∧
  (∀ x > Classical.choose h4, f x < 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1826_182676


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1826_182662

theorem max_sum_of_squares (x y z : ℕ+) 
  (h1 : x.val * y.val * z.val = (14 - x.val) * (14 - y.val) * (14 - z.val))
  (h2 : x.val + y.val + z.val < 28) :
  x.val^2 + y.val^2 + z.val^2 ≤ 219 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1826_182662


namespace NUMINAMATH_CALUDE_ratio_and_equation_solution_l1826_182679

/-- Given that x, y, and z are in the ratio 1:4:5, y = 15a - 5, and y = 60, prove that a = 13/3 -/
theorem ratio_and_equation_solution (x y z a : ℚ) 
  (h_ratio : x / y = 1 / 4 ∧ y / z = 4 / 5)
  (h_eq : y = 15 * a - 5)
  (h_y : y = 60) : 
  a = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_and_equation_solution_l1826_182679


namespace NUMINAMATH_CALUDE_parallelogram_area_l1826_182677

/-- The area of a parallelogram with base 15 and height 5 is 75 square feet. -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 15 ∧ height = 5 → area = base * height → area = 75

/-- Proof of the parallelogram area theorem -/
lemma prove_parallelogram_area : parallelogram_area 15 5 75 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1826_182677


namespace NUMINAMATH_CALUDE_sum_reciprocals_equal_negative_two_l1826_182683

theorem sum_reciprocals_equal_negative_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y + x * y = 0) : y / x + x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equal_negative_two_l1826_182683


namespace NUMINAMATH_CALUDE_carson_gardening_time_l1826_182615

/-- The total time Carson spends gardening is 108 minutes -/
theorem carson_gardening_time :
  let lines_to_mow : ℕ := 40
  let time_per_line : ℕ := 2
  let flower_rows : ℕ := 8
  let flowers_per_row : ℕ := 7
  let time_per_flower : ℚ := 1/2
  lines_to_mow * time_per_line + flower_rows * flowers_per_row * time_per_flower = 108 := by
  sorry

end NUMINAMATH_CALUDE_carson_gardening_time_l1826_182615


namespace NUMINAMATH_CALUDE_sum_remainder_by_six_l1826_182663

theorem sum_remainder_by_six : (284917 + 517084) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_six_l1826_182663


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1826_182674

/-- The first term of the geometric series -/
def a₁ : ℚ := 4 / 7

/-- The second term of the geometric series -/
def a₂ : ℚ := -16 / 21

/-- The third term of the geometric series -/
def a₃ : ℚ := -64 / 63

/-- The common ratio of the geometric series -/
def r : ℚ := -4 / 3

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1826_182674


namespace NUMINAMATH_CALUDE_percentage_difference_l1826_182614

theorem percentage_difference (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.2 * A) :
  (C - A) / C * 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1826_182614


namespace NUMINAMATH_CALUDE_intersection_and_lines_l1826_182621

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- Define point A
def A : ℝ × ℝ := (-1, -2)

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 1)

-- Define the condition for a
def a_condition (a : ℝ) : Prop := a ≠ -2 ∧ a ≠ -1 ∧ a ≠ 8/3

-- Define the equations of line l
def l_eq₁ (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0
def l_eq₂ (x y : ℝ) : Prop := x + 2 = 0

theorem intersection_and_lines :
  -- 1. P is the intersection point of l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  -- 2. Condition for a to form a triangle
  (∀ a : ℝ, (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ (a*x + 2*y - 6 = 0)) → a_condition a) ∧
  -- 3. Equations of line l passing through P with distance 1 from A
  (∀ x y : ℝ, (l_eq₁ x y ∨ l_eq₂ x y) ↔
    ((x - P.1)^2 + (y - P.2)^2 = 0 ∧
     ((x - A.1)^2 + (y - A.2)^2 - 1)^2 = 
     ((x - P.1)*(A.2 - P.2) - (y - P.2)*(A.1 - P.1))^2 / ((x - P.1)^2 + (y - P.2)^2)))
  := by sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l1826_182621


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1826_182613

/-- The function f(x) -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

/-- Theorem stating that the derivative of f(x) at x = 1 is 3 -/
theorem derivative_f_at_one : 
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1826_182613


namespace NUMINAMATH_CALUDE_lucas_pet_capacity_l1826_182628

/-- The number of pets Lucas can accommodate given his pet bed situation -/
def pets_accommodated (initial_beds : ℕ) (additional_beds : ℕ) (beds_per_pet : ℕ) : ℕ :=
  (initial_beds + additional_beds) / beds_per_pet

theorem lucas_pet_capacity : pets_accommodated 12 8 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lucas_pet_capacity_l1826_182628


namespace NUMINAMATH_CALUDE_smallest_multiple_l1826_182635

theorem smallest_multiple (n : ℕ) : n = 663 ↔ 
  n > 0 ∧ 
  n % 17 = 0 ∧ 
  (n - 6) % 73 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 17 = 0 → (m - 6) % 73 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1826_182635
