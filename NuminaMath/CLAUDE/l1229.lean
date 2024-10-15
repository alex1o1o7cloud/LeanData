import Mathlib

namespace NUMINAMATH_CALUDE_line_parametrization_l1229_122942

/-- The slope of the line -/
def m : ℚ := 3/4

/-- The y-intercept of the line -/
def b : ℚ := 3

/-- The x-coordinate of the point on the line when t = 0 -/
def x₀ : ℚ := -9

/-- The y-coordinate of the direction vector -/
def v : ℚ := -7

/-- The equation of the line -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- The parametric equations of the line -/
def param_eq (x y s l t : ℚ) : Prop :=
  x = x₀ + t * l ∧ y = s + t * v

theorem line_parametrization (s l : ℚ) : 
  (∀ t, line_eq (x₀ + t * l) (s + t * v)) ↔ s = -15/4 ∧ l = -28/3 :=
sorry

end NUMINAMATH_CALUDE_line_parametrization_l1229_122942


namespace NUMINAMATH_CALUDE_greatest_k_value_l1229_122935

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1229_122935


namespace NUMINAMATH_CALUDE_team_score_l1229_122986

def basketball_game (tobee jay sean remy alex : ℕ) : Prop :=
  tobee = 4 ∧
  jay = 2 * tobee + 6 ∧
  sean = jay / 2 ∧
  remy = tobee + jay - 3 ∧
  alex = sean + remy + 4

theorem team_score (tobee jay sean remy alex : ℕ) :
  basketball_game tobee jay sean remy alex →
  tobee + jay + sean + remy + alex = 66 := by
  sorry

end NUMINAMATH_CALUDE_team_score_l1229_122986


namespace NUMINAMATH_CALUDE_length_BI_approx_l1229_122966

/-- Triangle ABC with given side lengths --/
structure Triangle where
  ab : ℝ
  ac : ℝ
  bc : ℝ

/-- The incenter of a triangle --/
def Incenter (t : Triangle) : Point := sorry

/-- The distance between two points --/
def distance (p q : Point) : ℝ := sorry

/-- The given triangle --/
def triangle_ABC : Triangle := { ab := 31, ac := 29, bc := 30 }

/-- Theorem: The length of BI in the given triangle is approximately 17.22 --/
theorem length_BI_approx (ε : ℝ) (h : ε > 0) : 
  ∃ (B I : Point), I = Incenter triangle_ABC ∧ 
    |distance B I - 17.22| < ε := by sorry

end NUMINAMATH_CALUDE_length_BI_approx_l1229_122966


namespace NUMINAMATH_CALUDE_stratified_sampling_third_major_l1229_122970

/-- Given a college with three majors and stratified sampling, prove the number of students
    to be drawn from the third major. -/
theorem stratified_sampling_third_major
  (total_students : ℕ)
  (major_a_students : ℕ)
  (major_b_students : ℕ)
  (total_sample : ℕ)
  (h1 : total_students = 1200)
  (h2 : major_a_students = 380)
  (h3 : major_b_students = 420)
  (h4 : total_sample = 120) :
  (total_students - (major_a_students + major_b_students)) * total_sample / total_students = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_major_l1229_122970


namespace NUMINAMATH_CALUDE_valid_m_set_l1229_122960

def is_valid_m (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ 
    ∃ k : ℕ, m * n = k * k ∧
    ∃ p : ℕ, Nat.Prime p ∧ m - n = p

theorem valid_m_set :
  {m : ℕ | 1000 ≤ m ∧ m ≤ 2021 ∧ is_valid_m m} =
  {1156, 1296, 1369, 1600, 1764} :=
by sorry

end NUMINAMATH_CALUDE_valid_m_set_l1229_122960


namespace NUMINAMATH_CALUDE_jerry_piercing_earnings_l1229_122982

theorem jerry_piercing_earnings :
  let nose_price : ℚ := 20
  let ear_price : ℚ := nose_price * (1 + 1/2)
  let nose_count : ℕ := 6
  let ear_count : ℕ := 9
  nose_price * nose_count + ear_price * ear_count = 390 := by
  sorry

end NUMINAMATH_CALUDE_jerry_piercing_earnings_l1229_122982


namespace NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l1229_122993

theorem infinite_solutions_abs_value_equation (a : ℝ) :
  (∀ x : ℝ, |x - 2| = a * x - 2) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l1229_122993


namespace NUMINAMATH_CALUDE_total_fish_count_l1229_122976

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  morning_catch + afternoon_catch - thrown_back + dad_catch

/-- Theorem stating the total number of fish caught by Brendan and his dad -/
theorem total_fish_count :
  total_fish 8 3 5 13 = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l1229_122976


namespace NUMINAMATH_CALUDE_equation_solution_l1229_122947

theorem equation_solution : ∃ (z : ℂ), 
  (z - 4)^6 + (z - 6)^6 = 32 ∧ 
  (z = 5 + Complex.I * Real.sqrt 3 ∨ z = 5 - Complex.I * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1229_122947


namespace NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l1229_122921

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Define a function to check if a solid can have a triangular cross-section
def canHaveTriangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_no_triangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveTriangularCrossSection solid) ↔ solid = GeometricSolid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l1229_122921


namespace NUMINAMATH_CALUDE_bijection_probability_l1229_122929

/-- The probability of establishing a bijection from a subset of a 4-element set to a 5-element set -/
theorem bijection_probability (A : Finset α) (B : Finset β) 
  (hA : Finset.card A = 4) (hB : Finset.card B = 5) : 
  (Finset.card (Finset.powersetCard 4 B) / (Finset.card B ^ Finset.card A) : ℚ) = 24/125 :=
sorry

end NUMINAMATH_CALUDE_bijection_probability_l1229_122929


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_195_l1229_122914

theorem cos_75_cos_15_minus_sin_75_sin_195 : 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (75 * π / 180) * Real.sin (195 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_195_l1229_122914


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1229_122999

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1229_122999


namespace NUMINAMATH_CALUDE_b_investment_l1229_122938

/-- Represents the investment and profit share of a person in the business. -/
structure Participant where
  investment : ℝ
  profitShare : ℝ

/-- Proves that given the conditions of the problem, b's investment is 10000. -/
theorem b_investment (a b c : Participant)
  (h1 : b.profitShare = 3500)
  (h2 : c.profitShare - a.profitShare = 1399.9999999999998)
  (h3 : a.investment = 8000)
  (h4 : c.investment = 12000)
  (h5 : a.profitShare / a.investment = b.profitShare / b.investment)
  (h6 : c.profitShare / c.investment = b.profitShare / b.investment) :
  b.investment = 10000 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_l1229_122938


namespace NUMINAMATH_CALUDE_jasmine_purchase_cost_l1229_122900

/-- Calculate the total cost for Jasmine's purchase of coffee beans and milk --/
theorem jasmine_purchase_cost :
  let coffee_pounds : ℕ := 4
  let milk_gallons : ℕ := 2
  let coffee_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let discount_rate : ℚ := 1/10
  let tax_rate : ℚ := 2/25

  let total_before_discount : ℚ := coffee_pounds * coffee_price_per_pound + milk_gallons * milk_price_per_gallon
  let discount : ℚ := discount_rate * total_before_discount
  let discounted_price : ℚ := total_before_discount - discount
  let taxes : ℚ := tax_rate * discounted_price
  let final_amount : ℚ := discounted_price + taxes

  final_amount = 1652/100 := by sorry

end NUMINAMATH_CALUDE_jasmine_purchase_cost_l1229_122900


namespace NUMINAMATH_CALUDE_farmer_brown_sheep_l1229_122933

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The total number of legs among all animals Farmer Brown fed -/
def total_legs : ℕ := 34

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of legs a sheep has -/
def sheep_legs : ℕ := 4

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := (total_legs - num_chickens * chicken_legs) / sheep_legs

theorem farmer_brown_sheep : num_sheep = 5 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_sheep_l1229_122933


namespace NUMINAMATH_CALUDE_typing_difference_is_1200_l1229_122952

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 20

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 40

/-- The difference in words typed per hour between Isaiah and Micah -/
def typing_difference : ℕ := isaiah_speed * minutes_per_hour - micah_speed * minutes_per_hour

theorem typing_difference_is_1200 : typing_difference = 1200 := by
  sorry

end NUMINAMATH_CALUDE_typing_difference_is_1200_l1229_122952


namespace NUMINAMATH_CALUDE_max_black_balls_proof_l1229_122920

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of ways to select 2 red balls and 1 black ball -/
def selection_ways : ℕ := 30

/-- Calculates the number of ways to select 2 red balls and 1 black ball -/
def calc_selection_ways (red : ℕ) : ℕ :=
  Nat.choose red 2 * Nat.choose (total_balls - red) 1

/-- Checks if a given number of red balls satisfies the selection condition -/
def satisfies_condition (red : ℕ) : Prop :=
  calc_selection_ways red = selection_ways

/-- The maximum number of black balls -/
def max_black_balls : ℕ := 3

theorem max_black_balls_proof :
  ∃ (red : ℕ), satisfies_condition red ∧
  ∀ (x : ℕ), satisfies_condition x → (total_balls - x ≤ max_black_balls) :=
sorry

end NUMINAMATH_CALUDE_max_black_balls_proof_l1229_122920


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1229_122979

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 22

theorem total_cost_calculation :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 941.6) :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1229_122979


namespace NUMINAMATH_CALUDE_max_cookies_without_ingredients_l1229_122910

theorem max_cookies_without_ingredients (total_cookies : ℕ) 
  (peanut_cookies : ℕ) (choc_cookies : ℕ) (almond_cookies : ℕ) (raisin_cookies : ℕ) : 
  total_cookies = 60 →
  peanut_cookies ≥ 20 →
  choc_cookies ≥ 15 →
  almond_cookies ≥ 12 →
  raisin_cookies ≥ 7 →
  ∃ (plain_cookies : ℕ), plain_cookies ≤ 6 ∧ 
    plain_cookies + peanut_cookies + choc_cookies + almond_cookies + raisin_cookies ≥ total_cookies := by
  sorry

end NUMINAMATH_CALUDE_max_cookies_without_ingredients_l1229_122910


namespace NUMINAMATH_CALUDE_shirt_markup_price_l1229_122911

/-- Given a wholesale price of a shirt, prove that the initial price after 80% markup is $27 -/
theorem shirt_markup_price (P : ℝ) 
  (h1 : 1.80 * P = 1.80 * P) -- Initial price after 80% markup
  (h2 : 2.00 * P = 2.00 * P) -- Price for 100% markup
  (h3 : 2.00 * P - 1.80 * P = 3) -- Difference between 100% and 80% markup is $3
  : 1.80 * P = 27 := by
  sorry

end NUMINAMATH_CALUDE_shirt_markup_price_l1229_122911


namespace NUMINAMATH_CALUDE_pizza_slices_pizza_has_eight_slices_l1229_122932

theorem pizza_slices : ℕ → Prop :=
  fun total_slices =>
    let remaining_after_friend := total_slices - 2
    let james_slices := remaining_after_friend / 2
    james_slices = 3 → total_slices = 8

/-- The pizza has 8 slices. -/
theorem pizza_has_eight_slices : ∃ (n : ℕ), pizza_slices n :=
  sorry

end NUMINAMATH_CALUDE_pizza_slices_pizza_has_eight_slices_l1229_122932


namespace NUMINAMATH_CALUDE_john_annual_cost_l1229_122915

def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def replacements_per_year : ℕ := 2

def annual_cost : ℝ :=
  replacements_per_year * (epipen_cost * (1 - insurance_coverage))

theorem john_annual_cost : annual_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_cost_l1229_122915


namespace NUMINAMATH_CALUDE_bernoulli_prob_zero_success_l1229_122959

/-- The number of Bernoulli trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is (5/7)^7 -/
theorem bernoulli_prob_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_prob_zero_success_l1229_122959


namespace NUMINAMATH_CALUDE_sum_of_30th_set_l1229_122931

/-- Defines the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + (n * (n - 1)) / 2

/-- Defines the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Defines the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

theorem sum_of_30th_set : S 30 = 13515 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_30th_set_l1229_122931


namespace NUMINAMATH_CALUDE_number_equation_solution_l1229_122908

theorem number_equation_solution : 
  ∃ x : ℝ, (42 - 3 * x = 12) ∧ (x = 10) := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1229_122908


namespace NUMINAMATH_CALUDE_take_home_pay_l1229_122977

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem take_home_pay :
  annual_salary * (1 - tax_rate - healthcare_rate) - union_dues = 27200 := by
  sorry

end NUMINAMATH_CALUDE_take_home_pay_l1229_122977


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l1229_122975

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℕ), 5 ≤ n ∧ n ≤ 12 ∧ ∃ (m : ℕ), 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l1229_122975


namespace NUMINAMATH_CALUDE_inequality_and_optimality_l1229_122962

theorem inequality_and_optimality :
  (∀ (x y : ℝ), x > 0 → y > 0 → (x + y)^5 ≥ 12 * x * y * (x^3 + y^3)) ∧
  (∀ (K : ℝ), K > 12 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y)^5 < K * x * y * (x^3 + y^3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_optimality_l1229_122962


namespace NUMINAMATH_CALUDE_ball_probability_l1229_122913

theorem ball_probability (m n : ℕ) : 
  (∃ (total : ℕ), total = m + 8 + n ∧ 
   (8 : ℚ) / total = (m + n : ℚ) / total) → 
  m + n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1229_122913


namespace NUMINAMATH_CALUDE_regular_polygon_with_30_degree_exterior_angle_has_12_sides_l1229_122909

/-- A regular polygon with an exterior angle of 30° has 12 sides. -/
theorem regular_polygon_with_30_degree_exterior_angle_has_12_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n ≥ 3 →
    exterior_angle = 30 * (π / 180) →
    (360 : ℝ) * (π / 180) = n * exterior_angle →
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_30_degree_exterior_angle_has_12_sides_l1229_122909


namespace NUMINAMATH_CALUDE_roses_cut_equality_l1229_122924

/-- Represents the number of roses in various states --/
structure RoseCount where
  initial : ℕ
  thrown : ℕ
  given : ℕ
  final : ℕ

/-- Calculates the number of roses cut --/
def rosesCut (r : RoseCount) : ℕ :=
  r.final - r.initial + r.thrown + r.given

/-- Theorem stating that the number of roses cut is equal to the sum of
    the difference between final and initial roses, roses thrown away, and roses given away --/
theorem roses_cut_equality (r : RoseCount) :
  rosesCut r = r.final - r.initial + r.thrown + r.given :=
by sorry

end NUMINAMATH_CALUDE_roses_cut_equality_l1229_122924


namespace NUMINAMATH_CALUDE_hike_time_calculation_l1229_122978

theorem hike_time_calculation (distance : ℝ) (pace_up : ℝ) (pace_down : ℝ) 
  (h1 : distance = 12)
  (h2 : pace_up = 4)
  (h3 : pace_down = 6) :
  distance / pace_up + distance / pace_down = 5 := by
  sorry

end NUMINAMATH_CALUDE_hike_time_calculation_l1229_122978


namespace NUMINAMATH_CALUDE_sum_of_207_instances_of_33_difference_25_instances_of_112_from_3000_difference_product_and_sum_of_12_and_13_l1229_122950

-- Question 1
theorem sum_of_207_instances_of_33 : (Finset.range 207).sum (λ _ => 33) = 6831 := by sorry

-- Question 2
theorem difference_25_instances_of_112_from_3000 : 3000 - 25 * 112 = 200 := by sorry

-- Question 3
theorem difference_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by sorry

end NUMINAMATH_CALUDE_sum_of_207_instances_of_33_difference_25_instances_of_112_from_3000_difference_product_and_sum_of_12_and_13_l1229_122950


namespace NUMINAMATH_CALUDE_fraction_equality_l1229_122907

theorem fraction_equality (a b c : ℝ) (h : a^2 = b*c) :
  (a + b) / (a - b) = (c + a) / (c - a) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1229_122907


namespace NUMINAMATH_CALUDE_same_duration_trips_l1229_122974

/-- Proves that two trips with given distances and speed ratio have the same duration -/
theorem same_duration_trips (distance1 : ℝ) (distance2 : ℝ) (speed_ratio : ℝ) 
  (h1 : distance1 = 90) 
  (h2 : distance2 = 360) 
  (h3 : speed_ratio = 4) : 
  (distance1 / 1) = (distance2 / speed_ratio) := by
  sorry

#check same_duration_trips

end NUMINAMATH_CALUDE_same_duration_trips_l1229_122974


namespace NUMINAMATH_CALUDE_taxi_arrangement_count_l1229_122971

/-- The number of ways to divide 6 people into two taxis, where each taxi can carry up to 4 people -/
def taxi_arrangements : ℕ := 50

/-- The number of people to be divided -/
def num_people : ℕ := 6

/-- The maximum capacity of each taxi -/
def max_capacity : ℕ := 4

/-- Theorem stating that the number of ways to divide 6 people into two taxis, 
    where each taxi can carry up to 4 people, is equal to 50 -/
theorem taxi_arrangement_count : 
  taxi_arrangements = 50 ∧ 
  num_people = 6 ∧ 
  max_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_taxi_arrangement_count_l1229_122971


namespace NUMINAMATH_CALUDE_age_sum_proof_l1229_122991

/-- Tom's age in years -/
def tom_age : ℕ := 9

/-- Tom's sister's age in years -/
def sister_age : ℕ := tom_age / 2 + 1

/-- The sum of Tom's and his sister's ages -/
def sum_ages : ℕ := tom_age + sister_age

theorem age_sum_proof : sum_ages = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l1229_122991


namespace NUMINAMATH_CALUDE_andys_basketball_team_size_l1229_122905

/-- The number of cookies Andy had initially -/
def initial_cookies : ℕ := 72

/-- The number of cookies Andy ate -/
def cookies_eaten : ℕ := 3

/-- The number of cookies Andy gave to his little brother -/
def cookies_given : ℕ := 5

/-- Calculate the remaining cookies after Andy ate some and gave some to his brother -/
def remaining_cookies : ℕ := initial_cookies - (cookies_eaten + cookies_given)

/-- Function to calculate the sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := n * n

theorem andys_basketball_team_size :
  ∃ (team_size : ℕ), team_size > 0 ∧ sum_odd_numbers team_size = remaining_cookies :=
sorry

end NUMINAMATH_CALUDE_andys_basketball_team_size_l1229_122905


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l1229_122972

theorem no_solution_implies_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, ¬(x > 1 ∧ x < a - 1)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l1229_122972


namespace NUMINAMATH_CALUDE_intersection_when_a_is_2_subset_range_l1229_122922

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 3*(a+1)*x + 2*(3*a+1) < 0}
def B (a : ℝ) : Set ℝ := {x | (x-2*a) / (x-(a^2+1)) < 0}

-- Theorem for part (1)
theorem intersection_when_a_is_2 : A 2 ∩ B 2 = Set.Ioo 4 5 := by sorry

-- Theorem for part (2)
theorem subset_range : {a : ℝ | B a ⊆ A a} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_2_subset_range_l1229_122922


namespace NUMINAMATH_CALUDE_probability_two_girls_chosen_l1229_122944

def total_members : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5

theorem probability_two_girls_chosen (total_members num_boys num_girls : ℕ) 
  (h1 : total_members = 12)
  (h2 : num_boys = 7)
  (h3 : num_girls = 5)
  (h4 : total_members = num_boys + num_girls) :
  (Nat.choose num_girls 2 : ℚ) / (Nat.choose total_members 2) = 5 / 33 := by
sorry

end NUMINAMATH_CALUDE_probability_two_girls_chosen_l1229_122944


namespace NUMINAMATH_CALUDE_fib_75_mod_9_l1229_122953

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_75_mod_9 : fib 74 % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fib_75_mod_9_l1229_122953


namespace NUMINAMATH_CALUDE_expression_evaluation_l1229_122916

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  ((2*a + 3*b) * (2*a - 3*b) - (2*a - b)^2 - 3*a*b) / (-b) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1229_122916


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1229_122928

/-- Two lines in the plane, parameterized by a real number a -/
structure Lines (a : ℝ) where
  l₁ : ℝ → ℝ → Prop := λ x y => x + 2*a*y - 1 = 0
  l₂ : ℝ → ℝ → Prop := λ x y => (a + 1)*x - a*y = 0

/-- The condition for two lines to be parallel -/
def parallel (a : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) / (2*a) = k * ((a + 1) / (-a))

theorem parallel_lines_a_value (a : ℝ) :
  parallel a → a = -3/2 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1229_122928


namespace NUMINAMATH_CALUDE_star_six_three_l1229_122941

-- Define the binary operation *
def star (x y : ℝ) : ℝ := 4*x + 5*y - x*y

-- Theorem statement
theorem star_six_three : star 6 3 = 21 := by sorry

end NUMINAMATH_CALUDE_star_six_three_l1229_122941


namespace NUMINAMATH_CALUDE_card_value_decrease_l1229_122988

/-- Represents the percent decrease in the first year -/
def first_year_decrease : ℝ := sorry

/-- Represents the percent decrease in the second year -/
def second_year_decrease : ℝ := 10

/-- Represents the total percent decrease over two years -/
def total_decrease : ℝ := 55

theorem card_value_decrease :
  (1 - first_year_decrease / 100) * (1 - second_year_decrease / 100) = 1 - total_decrease / 100 ∧
  first_year_decrease = 50 := by sorry

end NUMINAMATH_CALUDE_card_value_decrease_l1229_122988


namespace NUMINAMATH_CALUDE_prime_between_squares_l1229_122964

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ a : ℕ, p = a^2 + 5 ∧ p = (a+1)^2 - 8 :=
by
  sorry

end NUMINAMATH_CALUDE_prime_between_squares_l1229_122964


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l1229_122957

/-- The time (in hours after 8:00 AM) when Cassie and Brian meet -/
def meeting_time : ℝ := 2.68333333

/-- The total distance of the route in miles -/
def total_distance : ℝ := 75

/-- Cassie's speed in miles per hour -/
def cassie_speed : ℝ := 15

/-- Brian's speed in miles per hour -/
def brian_speed : ℝ := 18

/-- The time difference between Cassie and Brian's departure in hours -/
def time_difference : ℝ := 0.75

theorem cyclists_meet_time :
  cassie_speed * meeting_time + brian_speed * (meeting_time - time_difference) = total_distance :=
sorry

end NUMINAMATH_CALUDE_cyclists_meet_time_l1229_122957


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l1229_122965

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ a b : ℚ, y = a * (b ^ (1/2 : ℝ))) → (∃ c d : ℚ, x = c * (d ^ (1/2 : ℝ))) → False

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (26 ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (8 ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical ((1/3 : ℝ) ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (2 / (6 ^ (1/2 : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l1229_122965


namespace NUMINAMATH_CALUDE_parabola_translation_l1229_122925

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := 2 * (x - 2)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = translated_parabola x ↔ y - 3 = original_parabola (x - 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l1229_122925


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l1229_122992

def f (x : ℝ) : ℝ := -x + 1

theorem linear_function_decreasing (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l1229_122992


namespace NUMINAMATH_CALUDE_square_difference_l1229_122995

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1229_122995


namespace NUMINAMATH_CALUDE_contrapositive_divisibility_l1229_122967

theorem contrapositive_divisibility (n : ℤ) : 
  (∀ m : ℤ, m % 6 = 0 → m % 2 = 0) ↔ 
  (∀ k : ℤ, k % 2 ≠ 0 → k % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_divisibility_l1229_122967


namespace NUMINAMATH_CALUDE_stratified_sampling_arrangements_l1229_122946

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4

theorem stratified_sampling_arrangements :
  (Nat.choose total_balls black_balls) = number_of_arrangements :=
by sorry

#check stratified_sampling_arrangements

end NUMINAMATH_CALUDE_stratified_sampling_arrangements_l1229_122946


namespace NUMINAMATH_CALUDE_harvest_rent_proof_l1229_122984

/-- The total rent paid during the harvest season. -/
def total_rent (weekly_rent : ℕ) (weeks : ℕ) : ℕ :=
  weekly_rent * weeks

/-- Proof that the total rent paid during the harvest season is $527,292. -/
theorem harvest_rent_proof :
  total_rent 388 1359 = 527292 := by
  sorry

end NUMINAMATH_CALUDE_harvest_rent_proof_l1229_122984


namespace NUMINAMATH_CALUDE_exponent_of_5_in_30_factorial_is_7_l1229_122918

/-- The exponent of 5 in the prime factorization of 30! -/
def exponent_of_5_in_30_factorial : ℕ :=
  (30 / 5) + (30 / 25)

/-- Theorem stating that the exponent of 5 in the prime factorization of 30! is 7 -/
theorem exponent_of_5_in_30_factorial_is_7 :
  exponent_of_5_in_30_factorial = 7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_5_in_30_factorial_is_7_l1229_122918


namespace NUMINAMATH_CALUDE_apples_bought_by_junhyeok_and_jihyun_l1229_122917

/-- The number of apple boxes Junhyeok bought -/
def junhyeok_boxes : ℕ := 7

/-- The number of apples in each of Junhyeok's boxes -/
def junhyeok_apples_per_box : ℕ := 16

/-- The number of apple boxes Jihyun bought -/
def jihyun_boxes : ℕ := 6

/-- The number of apples in each of Jihyun's boxes -/
def jihyun_apples_per_box : ℕ := 25

/-- The total number of apples bought by Junhyeok and Jihyun -/
def total_apples : ℕ := junhyeok_boxes * junhyeok_apples_per_box + jihyun_boxes * jihyun_apples_per_box

theorem apples_bought_by_junhyeok_and_jihyun : total_apples = 262 := by
  sorry

end NUMINAMATH_CALUDE_apples_bought_by_junhyeok_and_jihyun_l1229_122917


namespace NUMINAMATH_CALUDE_sixth_term_value_l1229_122961

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = (a (n-1) + a (n+1)) / 3

theorem sixth_term_value (a : ℕ → ℕ) :
  sequence_rule a →
  a 2 = 7 →
  a 3 = 20 →
  a 6 = 364 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l1229_122961


namespace NUMINAMATH_CALUDE_one_eighth_of_two_to_36_l1229_122906

theorem one_eighth_of_two_to_36 (y : ℤ) :
  (1 / 8 : ℚ) * (2 : ℚ)^36 = (2 : ℚ)^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_two_to_36_l1229_122906


namespace NUMINAMATH_CALUDE_f_derivative_l1229_122968

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- State the theorem
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l1229_122968


namespace NUMINAMATH_CALUDE_intersection_and_system_solution_l1229_122940

theorem intersection_and_system_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), y = -x + 4 ∧ y = 2*x + m ∧ x = 3 ∧ y = n) →
  (∀ (x y : ℝ), x + y - 4 = 0 ∧ 2*x - y + m = 0 ↔ x = 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_system_solution_l1229_122940


namespace NUMINAMATH_CALUDE_debate_team_group_size_l1229_122997

/-- Proves that the number of students in each group is 9,
    given the number of boys, girls, and total groups. -/
theorem debate_team_group_size
  (boys : ℕ)
  (girls : ℕ)
  (total_groups : ℕ)
  (h1 : boys = 26)
  (h2 : girls = 46)
  (h3 : total_groups = 8) :
  (boys + girls) / total_groups = 9 :=
by sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l1229_122997


namespace NUMINAMATH_CALUDE_four_inch_cube_worth_l1229_122948

/-- The worth of a cube of gold in dollars -/
def worth (side_length : ℝ) : ℝ :=
  300 * side_length^3

/-- Theorem: The worth of a 4-inch cube of gold is $19200 -/
theorem four_inch_cube_worth : worth 4 = 19200 := by
  sorry

end NUMINAMATH_CALUDE_four_inch_cube_worth_l1229_122948


namespace NUMINAMATH_CALUDE_circumcircle_intersection_l1229_122989

-- Define a point in a plane
structure Point : Type :=
  (x : ℝ) (y : ℝ)

-- Define a circle
structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define a function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define a function to create a circumcircle of a triangle
def circumcircle (a b c : Point) : Circle :=
  sorry

-- Define the main theorem
theorem circumcircle_intersection
  (A₁ A₂ B₁ B₂ C₁ C₂ : Point)
  (h : ∃ (P : Point),
    pointOnCircle P (circumcircle A₁ B₁ C₁) ∧
    pointOnCircle P (circumcircle A₁ B₂ C₂) ∧
    pointOnCircle P (circumcircle A₂ B₁ C₂) ∧
    pointOnCircle P (circumcircle A₂ B₂ C₁)) :
  ∃ (Q : Point),
    pointOnCircle Q (circumcircle A₂ B₂ C₂) ∧
    pointOnCircle Q (circumcircle A₂ B₁ C₁) ∧
    pointOnCircle Q (circumcircle A₁ B₂ C₁) ∧
    pointOnCircle Q (circumcircle A₁ B₁ C₂) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_intersection_l1229_122989


namespace NUMINAMATH_CALUDE_point_upper_right_of_line_l1229_122998

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the concept of a point being to the upper right of a line
def upper_right_of_line (x y : ℝ) : Prop := x + y - 3 > 0

-- Theorem statement
theorem point_upper_right_of_line (a : ℝ) :
  upper_right_of_line (-1) a → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_point_upper_right_of_line_l1229_122998


namespace NUMINAMATH_CALUDE_square_difference_l1229_122903

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : 
  (x - y)^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1229_122903


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l1229_122901

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 - x + 1)) →
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l1229_122901


namespace NUMINAMATH_CALUDE_inequality_proof_l1229_122954

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 1 / 10)
  (hb : b = Real.sin 1 / (9 + Real.cos 1))
  (hc : c = (Real.exp (1 / 10)) - 1) :
  b < a ∧ a < c :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1229_122954


namespace NUMINAMATH_CALUDE_extremal_points_imply_a_gt_one_and_sum_gt_two_l1229_122934

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - (1/2) * x^2 - a * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := exp x - x - a

theorem extremal_points_imply_a_gt_one_and_sum_gt_two
  (a : ℝ)
  (x₁ x₂ : ℝ)
  (h₁ : f' a x₁ = 0)
  (h₂ : f' a x₂ = 0)
  (h₃ : x₁ ≠ x₂)
  : a > 1 ∧ f a x₁ + f a x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_extremal_points_imply_a_gt_one_and_sum_gt_two_l1229_122934


namespace NUMINAMATH_CALUDE_decagon_cuts_to_two_regular_polygons_l1229_122969

/-- A regular polygon with n sides -/
structure RegularPolygon where
  sides : Nat
  isRegular : sides ≥ 3

/-- A decagon is a regular polygon with 10 sides -/
def Decagon : RegularPolygon where
  sides := 10
  isRegular := by norm_num

/-- Represent a cut of a polygon along its diagonals -/
structure DiagonalCut (p : RegularPolygon) where
  pieces : List RegularPolygon
  sum_sides : (pieces.map RegularPolygon.sides).sum = p.sides

/-- Theorem: A regular decagon can be cut into two regular polygons -/
theorem decagon_cuts_to_two_regular_polygons : 
  ∃ (cut : DiagonalCut Decagon), cut.pieces.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_cuts_to_two_regular_polygons_l1229_122969


namespace NUMINAMATH_CALUDE_bobbit_worm_predation_l1229_122980

/-- Calculates the number of fish remaining in an aquarium after a Bobbit worm's predation --/
theorem bobbit_worm_predation 
  (initial_fish : ℕ) 
  (daily_eaten : ℕ) 
  (days_before_adding : ℕ) 
  (added_fish : ℕ) 
  (days_after_adding : ℕ) :
  initial_fish = 60 →
  daily_eaten = 2 →
  days_before_adding = 14 →
  added_fish = 8 →
  days_after_adding = 7 →
  initial_fish + added_fish - (daily_eaten * (days_before_adding + days_after_adding)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_bobbit_worm_predation_l1229_122980


namespace NUMINAMATH_CALUDE_coefficient_c_nonzero_l1229_122912

/-- A polynomial of degree 4 with four distinct roots, one of which is 0 -/
structure QuarticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  has_four_distinct_roots : ∃ (p q r : ℝ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧
    ∀ x, x^4 + a*x^3 + b*x^2 + c*x + d = x*(x-p)*(x-q)*(x-r)
  zero_is_root : d = 0

theorem coefficient_c_nonzero (Q : QuarticPolynomial) : Q.c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_c_nonzero_l1229_122912


namespace NUMINAMATH_CALUDE_parallelogram_height_l1229_122990

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) (h1 : area = 33.3) (h2 : base = 9) 
    (h3 : area = base * height) : height = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1229_122990


namespace NUMINAMATH_CALUDE_ratio_equality_l1229_122919

theorem ratio_equality (x y z m n k a b c : ℝ) 
  (h : x / (m * (n * b + k * c - m * a)) = 
       y / (n * (k * c + m * a - n * b)) ∧
       y / (n * (k * c + m * a - n * b)) = 
       z / (k * (m * a + n * b - k * c))) :
  m / (x * (b * y + c * z - a * x)) = 
  n / (y * (c * z + a * x - b * y)) ∧
  n / (y * (c * z + a * x - b * y)) = 
  k / (z * (a * x + b * y - c * z)) :=
by sorry

end NUMINAMATH_CALUDE_ratio_equality_l1229_122919


namespace NUMINAMATH_CALUDE_shiela_paintings_distribution_l1229_122926

/-- Given Shiela has 18 paintings and each relative gets 9 paintings, 
    prove that she is giving paintings to 2 relatives. -/
theorem shiela_paintings_distribution (total_paintings : ℕ) (paintings_per_relative : ℕ) 
  (h1 : total_paintings = 18) 
  (h2 : paintings_per_relative = 9) : 
  total_paintings / paintings_per_relative = 2 := by
  sorry

end NUMINAMATH_CALUDE_shiela_paintings_distribution_l1229_122926


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1229_122949

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1229_122949


namespace NUMINAMATH_CALUDE_average_trees_planted_l1229_122902

def tree_data : List ℕ := [10, 8, 9, 9]

theorem average_trees_planted : 
  (List.sum tree_data) / (List.length tree_data : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_planted_l1229_122902


namespace NUMINAMATH_CALUDE_knitting_time_theorem_l1229_122956

/-- Represents the time in hours to knit each item -/
structure KnittingTime where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  mittens : ℝ
  socks : ℝ

/-- Calculates the total time to knit multiple sets of clothes -/
def totalKnittingTime (time : KnittingTime) (sets : ℕ) : ℝ :=
  (time.hat + time.scarf + time.sweater + time.mittens + time.socks) * sets

/-- Theorem: The total time to knit 3 sets of clothes with given knitting times is 48 hours -/
theorem knitting_time_theorem (time : KnittingTime) 
  (h_hat : time.hat = 2)
  (h_scarf : time.scarf = 3)
  (h_sweater : time.sweater = 6)
  (h_mittens : time.mittens = 2)
  (h_socks : time.socks = 3) :
  totalKnittingTime time 3 = 48 := by
  sorry

#check knitting_time_theorem

end NUMINAMATH_CALUDE_knitting_time_theorem_l1229_122956


namespace NUMINAMATH_CALUDE_bicycle_final_price_l1229_122981

/-- The final selling price of a bicycle given initial cost and profit margins -/
theorem bicycle_final_price (a_cost : ℝ) (a_profit_percent : ℝ) (b_profit_percent : ℝ) : 
  a_cost = 112.5 → 
  a_profit_percent = 60 → 
  b_profit_percent = 25 → 
  a_cost * (1 + a_profit_percent / 100) * (1 + b_profit_percent / 100) = 225 := by
sorry

end NUMINAMATH_CALUDE_bicycle_final_price_l1229_122981


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1229_122994

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  t : ℝ → Point3D

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (l : Line3D) (plane : Plane) : Prop :=
  ∀ t, pointOnPlane (l.t t) plane

/-- The specific point given in the problem -/
def givenPoint : Point3D :=
  { x := 1, y := -3, z := 6 }

/-- The specific line given in the problem -/
def givenLine : Line3D :=
  { t := λ t => { x := 4*t + 2, y := -t - 1, z := 2*t + 3 } }

/-- The plane we need to prove -/
def resultPlane : Plane :=
  { A := 1, B := -18, C := -7, D := -13 }

theorem plane_equation_proof :
  (pointOnPlane givenPoint resultPlane) ∧
  (lineInPlane givenLine resultPlane) ∧
  (resultPlane.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
           (Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l1229_122994


namespace NUMINAMATH_CALUDE_square_root_equation_l1229_122951

theorem square_root_equation (x : ℝ) : Real.sqrt (5 * x - 1) = 3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1229_122951


namespace NUMINAMATH_CALUDE_gcd_228_1995_l1229_122927

theorem gcd_228_1995 : Nat.gcd 228 1995 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l1229_122927


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l1229_122937

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Theorem stating the total cost of fencing for a specific rectangular plot -/
theorem fencing_cost_calculation :
  let length : ℝ := 62
  let breadth : ℝ := 38
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
sorry

end NUMINAMATH_CALUDE_fencing_cost_calculation_l1229_122937


namespace NUMINAMATH_CALUDE_equation_solution_l1229_122996

theorem equation_solution (x : ℝ) : 
  2 - 1 / (3 - x) = 1 / (2 + x) → 
  x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1229_122996


namespace NUMINAMATH_CALUDE_negation_equivalence_l1229_122923

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1229_122923


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1229_122955

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ → ℝ × ℝ := λ m ↦ (2 + m, 3 - m)
  let c : ℝ → ℝ × ℝ := λ m ↦ (3 * m, 1)
  ∀ m : ℝ, (∃ k : ℝ, a = k • (c m - b m)) → m = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1229_122955


namespace NUMINAMATH_CALUDE_weight_of_steel_ingot_l1229_122985

/-- Given a weight vest and purchase conditions, prove the weight of each steel ingot. -/
theorem weight_of_steel_ingot
  (original_weight : ℝ)
  (weight_increase_percent : ℝ)
  (ingot_cost : ℝ)
  (discount_percent : ℝ)
  (final_cost : ℝ)
  (h1 : original_weight = 60)
  (h2 : weight_increase_percent = 0.60)
  (h3 : ingot_cost = 5)
  (h4 : discount_percent = 0.20)
  (h5 : final_cost = 72)
  : ∃ (num_ingots : ℕ), 
    num_ingots > 10 ∧ 
    (2 : ℝ) = (original_weight * weight_increase_percent) / num_ingots :=
by sorry

end NUMINAMATH_CALUDE_weight_of_steel_ingot_l1229_122985


namespace NUMINAMATH_CALUDE_lamp_cost_l1229_122904

/-- Proves the cost of the lamp given Daria's furniture purchase scenario -/
theorem lamp_cost (savings : ℕ) (couch_cost table_cost remaining_debt : ℕ) : 
  savings = 500 → 
  couch_cost = 750 → 
  table_cost = 100 → 
  remaining_debt = 400 → 
  ∃ (lamp_cost : ℕ), 
    lamp_cost = remaining_debt - (couch_cost + table_cost - savings) ∧ 
    lamp_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_lamp_cost_l1229_122904


namespace NUMINAMATH_CALUDE_planes_parallel_from_intersecting_lines_l1229_122987

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_in : Line → Plane → Prop)  -- A line lies in a plane
variable (parallel : Line → Plane → Prop)  -- A line is parallel to a plane
variable (intersect_at : Line → Line → Point → Prop)  -- Two lines intersect at a point
variable (plane_parallel : Plane → Plane → Prop)  -- Two planes are parallel

-- State the theorem
theorem planes_parallel_from_intersecting_lines 
  (l m : Line) (α β : Plane) (P : Point) :
  l ≠ m →  -- l and m are distinct lines
  α ≠ β →  -- α and β are different planes
  lies_in l α →
  lies_in m α →
  intersect_at l m P →
  parallel l β →
  parallel m β →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_intersecting_lines_l1229_122987


namespace NUMINAMATH_CALUDE_clock_angle_at_7pm_l1229_122943

/-- The number of hours on a clock face. -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle. -/
def full_circle_degrees : ℕ := 360

/-- The time in hours (7 p.m. is represented as 19). -/
def time : ℕ := 19

/-- The angle between hour marks on a clock face. -/
def angle_per_hour : ℚ := full_circle_degrees / clock_hours

/-- The number of hour marks between the hour hand and 12 o'clock at the given time. -/
def hour_hand_position : ℕ := time % clock_hours

/-- The angle between the hour and minute hands at the given time. -/
def clock_angle : ℚ := angle_per_hour * hour_hand_position

/-- The smaller angle between the hour and minute hands. -/
def smaller_angle : ℚ := min clock_angle (full_circle_degrees - clock_angle)

/-- 
Theorem: The measure of the smaller angle formed by the hour and minute hands 
of a clock at 7 p.m. is 150°.
-/
theorem clock_angle_at_7pm : smaller_angle = 150 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_7pm_l1229_122943


namespace NUMINAMATH_CALUDE_stating_sum_of_sides_approx_11_2_l1229_122958

/-- Represents a right triangle with angles 40°, 50°, and 90° -/
structure RightTriangle40_50_90 where
  /-- The side opposite to the 50° angle -/
  side_a : ℝ
  /-- The side opposite to the 40° angle -/
  side_b : ℝ
  /-- The hypotenuse -/
  side_c : ℝ
  /-- Constraint that side_a is 8 units long -/
  side_a_eq_8 : side_a = 8

/-- 
Theorem stating that the sum of the two sides (opposite to 40° and 90°) 
in a 40-50-90 right triangle with hypotenuse of 8 units 
is approximately 11.2 units
-/
theorem sum_of_sides_approx_11_2 (t : RightTriangle40_50_90) :
  ∃ ε > 0, abs (t.side_b + t.side_c - 11.2) < ε := by
  sorry


end NUMINAMATH_CALUDE_stating_sum_of_sides_approx_11_2_l1229_122958


namespace NUMINAMATH_CALUDE_sin_cos_sum_special_angles_l1229_122963

theorem sin_cos_sum_special_angles :
  Real.sin (36 * π / 180) * Real.cos (24 * π / 180) + 
  Real.cos (36 * π / 180) * Real.sin (156 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_special_angles_l1229_122963


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1229_122983

-- Define the properties of the function f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_incr : increasing_on_positive f)
  (h_zero : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = Set.Ioo (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1229_122983


namespace NUMINAMATH_CALUDE_log_49_x_equals_half_log_7_x_l1229_122936

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_49_x_equals_half_log_7_x (x : ℝ) (h : log 7 (x + 6) = 2) :
  log 49 x = (log 7 x) / 2 := by sorry

end NUMINAMATH_CALUDE_log_49_x_equals_half_log_7_x_l1229_122936


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l1229_122930

theorem sum_of_distinct_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72 →
  p + q + r + s + t = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l1229_122930


namespace NUMINAMATH_CALUDE_product_in_N_l1229_122939

-- Define set M
def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}

-- Define set N
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

-- Theorem statement
theorem product_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  sorry

end NUMINAMATH_CALUDE_product_in_N_l1229_122939


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1229_122945

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A regular 9-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1229_122945


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1229_122973

theorem product_remainder_mod_five : (1236 * 7483 * 53) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1229_122973
