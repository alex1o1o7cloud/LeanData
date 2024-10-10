import Mathlib

namespace points_collinear_l3258_325800

/-- Given vectors a and b in a vector space, and points A, B, C, D such that
    AB = a + 2b, BC = -5a + 6b, and CD = 7a - 2b, prove that A, B, and D are collinear. -/
theorem points_collinear 
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (a b : V) (A B C D : V) 
  (hAB : B - A = a + 2 • b)
  (hBC : C - B = -5 • a + 6 • b)
  (hCD : D - C = 7 • a - 2 • b) :
  ∃ (t : ℝ), D - A = t • (B - A) :=
sorry

end points_collinear_l3258_325800


namespace solutions_rearrangements_l3258_325884

def word := "SOLUTIONS"

def vowels := ['O', 'I', 'U', 'O']
def consonants := ['S', 'L', 'T', 'N', 'S', 'S']

def vowel_arrangements := Nat.factorial 4 / Nat.factorial 2
def consonant_arrangements := Nat.factorial 6 / Nat.factorial 3

theorem solutions_rearrangements : 
  vowel_arrangements * consonant_arrangements = 1440 := by
  sorry

end solutions_rearrangements_l3258_325884


namespace exponent_addition_l3258_325851

theorem exponent_addition (a : ℝ) : a^3 + a^3 = 2 * a^3 := by
  sorry

end exponent_addition_l3258_325851


namespace fraction_multiplication_equality_l3258_325876

theorem fraction_multiplication_equality : 
  (11/12 - 7/6 + 3/4 - 13/24) * (-48) = 2 := by sorry

end fraction_multiplication_equality_l3258_325876


namespace solve_equation_l3258_325823

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end solve_equation_l3258_325823


namespace product_evaluation_l3258_325878

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l3258_325878


namespace complex_equation_roots_l3258_325832

theorem complex_equation_roots : 
  let z₁ : ℂ := 4 - 0.5 * I
  let z₂ : ℂ := -2 + 0.5 * I
  (z₁^2 - 2*z₁ = 7 - 3*I) ∧ (z₂^2 - 2*z₂ = 7 - 3*I) := by
  sorry

end complex_equation_roots_l3258_325832


namespace polygon_sides_from_angle_sum_l3258_325836

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end polygon_sides_from_angle_sum_l3258_325836


namespace swim_time_ratio_l3258_325861

/-- Proves that the ratio of time taken to swim upstream to time taken to swim downstream is 2:1 -/
theorem swim_time_ratio (swim_speed : ℝ) (stream_speed : ℝ) 
  (h1 : swim_speed = 1.5) (h2 : stream_speed = 0.5) : 
  (swim_speed - stream_speed) / (swim_speed + stream_speed) = 1 / 2 := by
  sorry

end swim_time_ratio_l3258_325861


namespace max_sum_of_entries_l3258_325815

def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_of_entries (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

def is_valid_partition (l1 l2 : List ℕ) : Prop :=
  l1.length = 4 ∧ l2.length = 4 ∧ (l1 ++ l2).toFinset = numbers.toFinset

theorem max_sum_of_entries :
  ∃ (top left : List ℕ), 
    is_valid_partition top left ∧ 
    sum_of_entries top left = 1440 ∧
    ∀ (t l : List ℕ), is_valid_partition t l → sum_of_entries t l ≤ 1440 := by
  sorry

end max_sum_of_entries_l3258_325815


namespace cost_price_calculation_l3258_325879

/-- Proves that the cost price of an article is 540 given the specified conditions -/
theorem cost_price_calculation (marked_up_price : ℝ → ℝ) (discounted_price : ℝ → ℝ) :
  (∀ x, marked_up_price x = x * 1.15) →
  (∀ x, discounted_price x = x * (1 - 0.2608695652173913)) →
  discounted_price (marked_up_price 540) = 459 :=
by sorry

end cost_price_calculation_l3258_325879


namespace angle_value_l3258_325891

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  is_geometric_sequence a →
  (∀ x : ℝ, x^2 - 2*x*Real.sin α - Real.sqrt 3*Real.sin α = 0 ↔ (x = a 1 ∨ x = a 8)) →
  (a 1 + a 8)^2 = 2*a 3*a 6 + 6 →
  0 < α ∧ α < Real.pi/2 →
  α = Real.pi/3 := by sorry

end angle_value_l3258_325891


namespace sqrt_3_times_sqrt_12_l3258_325853

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l3258_325853


namespace n2o_molecular_weight_l3258_325844

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O -/
def n_nitrogen : ℕ := 2

/-- The number of oxygen atoms in N2O -/
def n_oxygen : ℕ := 1

/-- The number of moles of N2O -/
def n_moles : ℝ := 8

theorem n2o_molecular_weight :
  n_moles * (n_nitrogen * nitrogen_weight + n_oxygen * oxygen_weight) = 352.16 := by
  sorry

end n2o_molecular_weight_l3258_325844


namespace system_solution_l3258_325856

theorem system_solution : 
  ∃! (s : Set (ℝ × ℝ)), s = {(2, 4), (4, 2)} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ 
    ((x / y + y / x) * (x + y) = 15 ∧
     (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85) :=
by sorry

end system_solution_l3258_325856


namespace vessel_weight_percentage_l3258_325854

theorem vessel_weight_percentage (E P : ℝ) 
  (h1 : (1/2) * (E + P) = E + 0.42857142857142855 * P) : 
  (E / (E + P)) * 100 = 12.5 := by
  sorry

end vessel_weight_percentage_l3258_325854


namespace orange_apple_difference_l3258_325882

/-- The number of apples Leif has -/
def num_apples : ℕ := 14

/-- The number of dozens of oranges Leif has -/
def dozens_oranges : ℕ := 2

/-- The number of oranges in a dozen -/
def oranges_per_dozen : ℕ := 12

/-- The total number of oranges Leif has -/
def num_oranges : ℕ := dozens_oranges * oranges_per_dozen

theorem orange_apple_difference :
  num_oranges - num_apples = 10 := by sorry

end orange_apple_difference_l3258_325882


namespace system_equations_proof_l3258_325894

theorem system_equations_proof (x y a : ℝ) : 
  (3 * x + y = 2 + 3 * a) →
  (x + 3 * y = 2 + a) →
  (x + y < 0) →
  (a < -1) ∧ (|1 - a| + |a + 1/2| = 1/2 - 2 * a) := by
sorry

end system_equations_proof_l3258_325894


namespace smoothie_size_l3258_325860

-- Define the constants from the problem
def packet_size : ℝ := 3
def water_per_packet : ℝ := 15
def total_smoothies : ℝ := 150
def total_packets : ℝ := 180

-- Define the theorem
theorem smoothie_size :
  let packets_per_smoothie := total_packets / total_smoothies
  let mix_per_smoothie := packets_per_smoothie * packet_size
  let water_per_smoothie := packets_per_smoothie * water_per_packet
  mix_per_smoothie + water_per_smoothie = 21.6 := by
sorry

end smoothie_size_l3258_325860


namespace cone_height_from_sector_l3258_325817

/-- Given a sector paper with radius 13 cm and area 65π cm², prove that when formed into a cone, the height of the cone is 12 cm. -/
theorem cone_height_from_sector (r : ℝ) (h : ℝ) :
  r = 13 →
  r * r * π / 2 = 65 * π →
  h = 12 :=
by sorry

end cone_height_from_sector_l3258_325817


namespace linear_mapping_midpoint_distance_l3258_325852

/-- Linear mapping from a segment of length 10 to a segment of length 5 -/
def LinearMapping (x y : ℝ) : Prop :=
  x / 10 = y / 5

/-- Theorem: In the given linear mapping, when x = 3, x + y = 4.5 -/
theorem linear_mapping_midpoint_distance (x y : ℝ) :
  LinearMapping x y → x = 3 → x + y = 4.5 := by
  sorry

end linear_mapping_midpoint_distance_l3258_325852


namespace budget_this_year_l3258_325867

def cost_supply1 : ℕ := 13
def cost_supply2 : ℕ := 24
def remaining_last_year : ℕ := 6
def remaining_after_purchase : ℕ := 19

theorem budget_this_year :
  (cost_supply1 + cost_supply2 + remaining_after_purchase) - remaining_last_year = 50 := by
  sorry

end budget_this_year_l3258_325867


namespace count_valid_three_digit_numbers_l3258_325814

/-- The count of three-digit numbers with specific exclusions -/
def valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let numbers_with_two_same_nonadjacent_digits := 81
  let numbers_with_increasing_digits := 28
  total_three_digit_numbers - (numbers_with_two_same_nonadjacent_digits + numbers_with_increasing_digits)

/-- Theorem stating the count of valid three-digit numbers -/
theorem count_valid_three_digit_numbers :
  valid_three_digit_numbers = 791 := by
  sorry

end count_valid_three_digit_numbers_l3258_325814


namespace ladder_distance_l3258_325896

theorem ladder_distance (ladder_length : ℝ) (elevation_angle : ℝ) (distance_to_wall : ℝ) :
  ladder_length = 9.2 →
  elevation_angle = 60 * π / 180 →
  distance_to_wall = ladder_length * Real.cos elevation_angle →
  distance_to_wall = 4.6 := by
  sorry

end ladder_distance_l3258_325896


namespace three_year_deposit_optimal_l3258_325821

/-- Represents the deposit options available --/
inductive DepositOption
  | OneYearRepeated
  | OneYearThenTwoYear
  | TwoYearThenOneYear
  | ThreeYear

/-- Calculates the final amount for a given deposit option --/
def calculateFinalAmount (option : DepositOption) (initialDeposit : ℝ) : ℝ :=
  match option with
  | .OneYearRepeated => initialDeposit * (1 + 0.0414 * 0.8)^3
  | .OneYearThenTwoYear => initialDeposit * (1 + 0.0414 * 0.8) * (1 + 0.0468 * 0.8 * 2)
  | .TwoYearThenOneYear => initialDeposit * (1 + 0.0468 * 0.8 * 2) * (1 + 0.0414 * 0.8)
  | .ThreeYear => initialDeposit * (1 + 0.0540 * 3 * 0.8)

/-- Theorem stating that the three-year fixed deposit option yields the highest return --/
theorem three_year_deposit_optimal (initialDeposit : ℝ) (h : initialDeposit > 0) :
  ∀ option : DepositOption, calculateFinalAmount .ThreeYear initialDeposit ≥ calculateFinalAmount option initialDeposit :=
by sorry

end three_year_deposit_optimal_l3258_325821


namespace max_candies_drawn_exists_ten_candies_drawn_l3258_325809

/-- Represents the number of candies of each color --/
structure CandyCount where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the state of candies before and after drawing --/
structure CandyState where
  initial : CandyCount
  drawn : ℕ
  final : CandyCount

/-- Checks if the candy state satisfies all conditions --/
def satisfiesConditions (state : CandyState) : Prop :=
  state.initial.yellow * 3 = state.initial.red * 5 ∧
  state.final.yellow = 2 ∧
  state.final.red = 2 ∧
  state.final.blue ≥ 5 ∧
  state.drawn = state.initial.yellow + state.initial.red + state.initial.blue -
                (state.final.yellow + state.final.red + state.final.blue)

/-- Theorem stating that the maximum number of candies Petya can draw is 10 --/
theorem max_candies_drawn (state : CandyState) :
  satisfiesConditions state → state.drawn ≤ 10 :=
by
  sorry

/-- Theorem stating that it's possible to draw exactly 10 candies while satisfying all conditions --/
theorem exists_ten_candies_drawn :
  ∃ state : CandyState, satisfiesConditions state ∧ state.drawn = 10 :=
by
  sorry

end max_candies_drawn_exists_ten_candies_drawn_l3258_325809


namespace scallop_dinner_cost_l3258_325816

/-- Calculates the cost of scallops for a dinner party. -/
def scallop_cost (people : ℕ) (scallops_per_person : ℕ) (scallops_per_pound : ℕ) (cost_per_pound : ℚ) : ℚ :=
  (people * scallops_per_person : ℚ) / scallops_per_pound * cost_per_pound

/-- Proves that the cost of scallops for 8 people, given 2 scallops per person, 
    is $48.00, when 8 scallops weigh one pound and cost $24.00 per pound. -/
theorem scallop_dinner_cost : 
  scallop_cost 8 2 8 24 = 48 := by
  sorry

end scallop_dinner_cost_l3258_325816


namespace pilot_course_cost_difference_pilot_course_cost_difference_holds_l3258_325875

/-- The cost difference between flight and ground school portions of a private pilot course -/
theorem pilot_course_cost_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_cost flight_cost ground_cost difference =>
    total_cost = 1275 ∧
    flight_cost = 950 ∧
    ground_cost = 325 ∧
    total_cost = flight_cost + ground_cost ∧
    flight_cost > ground_cost ∧
    difference = flight_cost - ground_cost ∧
    difference = 625

/-- The theorem holds for the given costs -/
theorem pilot_course_cost_difference_holds :
  ∃ (total_cost flight_cost ground_cost difference : ℕ),
    pilot_course_cost_difference total_cost flight_cost ground_cost difference :=
by
  sorry

end pilot_course_cost_difference_pilot_course_cost_difference_holds_l3258_325875


namespace max_altitude_triangle_ABC_l3258_325889

/-- Given a triangle ABC with the specified conditions, the maximum altitude on side BC is √3 + 1 -/
theorem max_altitude_triangle_ABC (A B C : Real) (h1 : 3 * (Real.sin B ^ 2 + Real.sin C ^ 2 - Real.sin A ^ 2) = 2 * Real.sqrt 3 * Real.sin B * Real.sin C) 
  (h2 : (1 / 2) * Real.sin A * (Real.sin B / Real.sin A) * (Real.sin C / Real.sin A) = Real.sqrt 6 + Real.sqrt 2) :
  ∃ (h : Real), h ≤ Real.sqrt 3 + 1 ∧ 
    ∀ (h' : Real), (∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      3 * (Real.sin (b / a) ^ 2 + Real.sin (c / a) ^ 2 - Real.sin 1 ^ 2) = 2 * Real.sqrt 3 * Real.sin (b / a) * Real.sin (c / a) ∧
      (1 / 2) * a * b * Real.sin (c / a) = Real.sqrt 6 + Real.sqrt 2 ∧
      h' = (2 * (Real.sqrt 6 + Real.sqrt 2)) / c) → 
    h' ≤ h :=
by sorry

end max_altitude_triangle_ABC_l3258_325889


namespace binary_equals_21_l3258_325801

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of the number in question -/
def binary_number : List Bool := [true, false, true, false, true]

/-- Theorem stating that the given binary number equals 21 in decimal -/
theorem binary_equals_21 : binary_to_decimal binary_number = 21 := by
  sorry

end binary_equals_21_l3258_325801


namespace root_minus_one_implies_k_equals_minus_two_l3258_325847

theorem root_minus_one_implies_k_equals_minus_two (k : ℝ) :
  ((-1 : ℝ)^2 - k*(-1) + 1 = 0) → k = -2 := by
  sorry

end root_minus_one_implies_k_equals_minus_two_l3258_325847


namespace shared_savings_theorem_l3258_325865

/-- Calculates the monthly savings per person for a shared down payment -/
def monthly_savings_per_person (down_payment : ℕ) (years : ℕ) : ℕ :=
  down_payment / (years * 12) / 2

/-- Theorem: Two people saving equally for a $108,000 down payment over 3 years each save $1,500 per month -/
theorem shared_savings_theorem :
  monthly_savings_per_person 108000 3 = 1500 := by
  sorry

end shared_savings_theorem_l3258_325865


namespace angle_sum_pi_half_l3258_325887

theorem angle_sum_pi_half (θ₁ θ₂ : Real) (h_acute₁ : 0 < θ₁ ∧ θ₁ < π/2) (h_acute₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h_eq : (Real.sin θ₁)^2020 / (Real.cos θ₂)^2018 + (Real.cos θ₁)^2020 / (Real.sin θ₂)^2018 = 1) :
  θ₁ + θ₂ = π/2 := by
sorry

end angle_sum_pi_half_l3258_325887


namespace third_number_problem_l3258_325807

theorem third_number_problem (x : ℝ) : 
  (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3 → x = 53 := by
  sorry

end third_number_problem_l3258_325807


namespace sale_price_is_twenty_l3258_325881

/-- The sale price of one bottle of detergent, given the number of loads per bottle and the cost per load when buying two bottles. -/
def sale_price (loads_per_bottle : ℕ) (cost_per_load : ℚ) : ℚ :=
  loads_per_bottle * cost_per_load

/-- Theorem stating that the sale price of one bottle of detergent is $20.00 -/
theorem sale_price_is_twenty :
  sale_price 80 (25 / 100) = 20 := by
  sorry

end sale_price_is_twenty_l3258_325881


namespace circle_equation_l3258_325872

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure CircleOnXAxis where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt2 : radius = Real.sqrt 2
  point_on_circle : (center.1 + 2)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x+1)^2 + y^2 = 2 or (x+3)^2 + y^2 = 2 -/
theorem circle_equation (c : CircleOnXAxis) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∨
  (∀ x y : ℝ, (x + 3)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
sorry

end circle_equation_l3258_325872


namespace function_positivity_implies_m_range_l3258_325838

/-- Given two functions f and g defined on real numbers, 
    prove that if at least one of f(x) or g(x) is positive for all real x,
    then the parameter m is in the open interval (0, 8) -/
theorem function_positivity_implies_m_range 
  (f g : ℝ → ℝ) 
  (m : ℝ) 
  (hf : f = fun x ↦ 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (hg : g = fun x ↦ m * x) 
  (h : ∀ x : ℝ, 0 < f x ∨ 0 < g x) : 
  0 < m ∧ m < 8 := by
  sorry

end function_positivity_implies_m_range_l3258_325838


namespace cubic_function_properties_l3258_325859

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f (-1/2) (-2) c x₁ = 0 ∧ f (-1/2) (-2) c x₂ = 0 ∧ f (-1/2) (-2) c x₃ = 0) →
    -22/27 < c ∧ c < 3/2) :=
by sorry

end cubic_function_properties_l3258_325859


namespace tips_fraction_of_income_l3258_325830

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the total income of a waitress -/
def totalIncome (w : WaitressIncome) : ℚ :=
  w.salary + w.tips

/-- Theorem stating that if a waitress's tips are 3/4 of her salary, 
    then 3/7 of her total income comes from tips -/
theorem tips_fraction_of_income 
  (w : WaitressIncome) 
  (h : w.tips = 3/4 * w.salary) : 
  w.tips / totalIncome w = 3/7 := by
  sorry

end tips_fraction_of_income_l3258_325830


namespace keyboard_mouse_cost_ratio_l3258_325813

/-- Given a mouse cost and total expenditure, proves the ratio of keyboard to mouse cost -/
theorem keyboard_mouse_cost_ratio 
  (mouse_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : mouse_cost = 16) 
  (h2 : total_cost = 64) 
  (h3 : ∃ n : ℝ, total_cost = mouse_cost + n * mouse_cost) :
  ∃ n : ℝ, n = 3 ∧ total_cost = mouse_cost + n * mouse_cost :=
sorry

end keyboard_mouse_cost_ratio_l3258_325813


namespace line_passes_through_fixed_point_l3258_325819

/-- The line ax+by-2=0 passes through the point (4,2) for all a and b that satisfy 2a+b=1 -/
theorem line_passes_through_fixed_point (a b : ℝ) (h : 2*a + b = 1) :
  a*4 + b*2 - 2 = 0 := by sorry

end line_passes_through_fixed_point_l3258_325819


namespace initial_number_problem_l3258_325834

theorem initial_number_problem : 
  let x : ℚ := 10
  ((x + 14) * 14 - 24) / 24 = 13 := by
  sorry

end initial_number_problem_l3258_325834


namespace factor_expression_l3258_325841

theorem factor_expression (y : ℝ) : 3 * y^3 - 75 * y = 3 * y * (y + 5) * (y - 5) := by
  sorry

end factor_expression_l3258_325841


namespace money_duration_l3258_325835

def mowing_earnings : ℕ := 5
def weed_eating_earnings : ℕ := 58
def weekly_spending : ℕ := 7

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end money_duration_l3258_325835


namespace probability_at_least_one_history_or_geography_l3258_325802

def total_outcomes : ℕ := Nat.choose 5 2

def favorable_outcomes : ℕ := Nat.choose 2 1 * Nat.choose 3 1 + Nat.choose 2 2

theorem probability_at_least_one_history_or_geography :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 10 := by
  sorry

end probability_at_least_one_history_or_geography_l3258_325802


namespace largest_divisor_of_n_squared_div_72_l3258_325886

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  12 = Nat.gcd n 72 ∧ ∀ m : ℕ, m ∣ n → m ≤ 12 :=
by sorry

end largest_divisor_of_n_squared_div_72_l3258_325886


namespace point_movement_on_number_line_l3258_325825

theorem point_movement_on_number_line :
  ∀ (a b c : ℝ),
    b = a - 3 →
    c = b + 5 →
    c = 1 →
    a = -1 := by
  sorry

end point_movement_on_number_line_l3258_325825


namespace blue_spotted_fish_ratio_l3258_325895

theorem blue_spotted_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) 
  (h1 : total_fish = 60) 
  (h2 : blue_spotted_fish = 10) : 
  (blue_spotted_fish : ℚ) / ((1 / 3 : ℚ) * total_fish) = 1 / 2 := by
  sorry

end blue_spotted_fish_ratio_l3258_325895


namespace share_distribution_l3258_325848

theorem share_distribution (a b c d : ℝ) : 
  a + b + c + d = 1200 →
  a = (3/5) * (b + c + d) →
  b = (2/3) * (a + c + d) →
  c = (4/7) * (a + b + d) →
  a = 247.5 := by
sorry

end share_distribution_l3258_325848


namespace sum_of_first_2015_digits_l3258_325804

/-- The repeating decimal 0.0142857 -/
def repeatingDecimal : ℚ := 1 / 7

/-- The length of the repeating part of the decimal -/
def repeatLength : ℕ := 6

/-- The sum of digits in one complete cycle of the repeating part -/
def cycleSum : ℕ := 27

/-- The number of complete cycles in the first 2015 digits -/
def completeCycles : ℕ := 2015 / repeatLength

/-- The number of remaining digits after complete cycles -/
def remainingDigits : ℕ := 2015 % repeatLength

/-- The sum of the remaining digits -/
def remainingSum : ℕ := 20

theorem sum_of_first_2015_digits : 
  (cycleSum * completeCycles + remainingSum : ℕ) = 9065 :=
sorry

end sum_of_first_2015_digits_l3258_325804


namespace vanessa_score_is_40_5_l3258_325843

/-- Calculates Vanessa's score in a basketball game. -/
def vanessaScore (totalTeamScore : ℝ) (numPlayers : ℕ) (otherPlayersAverage : ℝ) : ℝ :=
  totalTeamScore - (otherPlayersAverage * (numPlayers - 1 : ℝ))

/-- Proves that Vanessa's score is 40.5 points given the conditions of the game. -/
theorem vanessa_score_is_40_5 :
  vanessaScore 72 8 4.5 = 40.5 := by
  sorry

end vanessa_score_is_40_5_l3258_325843


namespace complex_modulus_problem_l3258_325846

theorem complex_modulus_problem (z : ℂ) (h : Complex.I * z = (1 - 2 * Complex.I)^2) : 
  Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l3258_325846


namespace bernoulli_inequalities_l3258_325870

theorem bernoulli_inequalities (α : ℝ) (n : ℕ) :
  (α > 0 ∧ n > 1 → (1 + α)^n > 1 + n * α) ∧
  (0 < α ∧ α ≤ 1 / n → (1 + α)^n < 1 + n * α + n^2 * α^2) := by
  sorry

end bernoulli_inequalities_l3258_325870


namespace total_dreams_calculation_l3258_325883

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The number of dreams per day in the current year -/
def dreamsPerDay : ℕ := 4

/-- The number of dreams in the current year -/
def dreamsThisYear : ℕ := dreamsPerDay * daysInYear

/-- The number of dreams in the previous year -/
def dreamsLastYear : ℕ := 2 * dreamsThisYear

/-- The total number of dreams over two years -/
def totalDreams : ℕ := dreamsLastYear + dreamsThisYear

theorem total_dreams_calculation :
  totalDreams = 4380 := by sorry

end total_dreams_calculation_l3258_325883


namespace quadratic_function_range_l3258_325824

/-- Given a quadratic function f(x) = ax^2 - c, prove that if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20. -/
theorem quadratic_function_range (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - c
  (-4 ≤ f 1 ∧ f 1 ≤ -1) → (-1 ≤ f 2 ∧ f 2 ≤ 5) → (-1 ≤ f 3 ∧ f 3 ≤ 20) := by
  sorry

end quadratic_function_range_l3258_325824


namespace total_card_units_traded_l3258_325828

/-- Represents the types of trading cards -/
inductive CardType
| A
| B
| C

/-- Represents a trading round -/
structure TradingRound where
  padmaInitial : CardType → ℕ
  robertInitial : CardType → ℕ
  padmaTrades : CardType → ℕ
  robertTrades : CardType → ℕ
  ratios : CardType → CardType → ℚ

/-- Calculates the total card units traded in a round -/
def cardUnitsTradedInRound (round : TradingRound) : ℚ :=
  sorry

/-- The three trading rounds -/
def round1 : TradingRound := {
  padmaInitial := λ | CardType.A => 50 | CardType.B => 45 | CardType.C => 30,
  robertInitial := λ _ => 0,  -- Not specified in the problem
  padmaTrades := λ | CardType.A => 5 | CardType.B => 12 | CardType.C => 0,
  robertTrades := λ | CardType.C => 20 | _ => 0,
  ratios := λ | CardType.A, CardType.C => 2 | CardType.B, CardType.C => 3/2 | _, _ => 1
}

def round2 : TradingRound := {
  padmaInitial := λ _ => 0,  -- Not relevant for this round
  robertInitial := λ | CardType.A => 60 | CardType.B => 50 | CardType.C => 40,
  robertTrades := λ | CardType.A => 10 | CardType.B => 3 | CardType.C => 15,
  padmaTrades := λ | CardType.A => 8 | CardType.B => 18 | CardType.C => 0,
  ratios := λ | CardType.A, CardType.B => 3/2 | CardType.B, CardType.C => 2 | CardType.C, CardType.A => 1 | _, _ => 1
}

def round3 : TradingRound := {
  padmaInitial := λ _ => 0,  -- Not relevant for this round
  robertInitial := λ _ => 0,  -- Not relevant for this round
  padmaTrades := λ | CardType.B => 15 | CardType.C => 10 | CardType.A => 0,
  robertTrades := λ | CardType.A => 12 | _ => 0,
  ratios := λ | CardType.A, CardType.B => 5/4 | CardType.C, CardType.A => 6/5 | _, _ => 1
}

/-- The main theorem stating the total card units traded -/
theorem total_card_units_traded :
  cardUnitsTradedInRound round1 + cardUnitsTradedInRound round2 + cardUnitsTradedInRound round3 = 94.75 := by
  sorry

end total_card_units_traded_l3258_325828


namespace arithmetic_sequence_sum_of_cubes_l3258_325829

/-- Represents an arithmetic sequence with n+1 terms, first term y, and common difference 4 -/
def arithmetic_sequence (y : ℤ) (n : ℕ) : List ℤ :=
  List.range (n + 1) |>.map (fun i => y + 4 * i)

/-- The sum of cubes of all terms in the sequence -/
def sum_of_cubes (seq : List ℤ) : ℤ :=
  seq.map (fun x => x^3) |>.sum

theorem arithmetic_sequence_sum_of_cubes (y : ℤ) (n : ℕ) :
  n > 6 →
  sum_of_cubes (arithmetic_sequence y n) = -5832 →
  n = 11 := by
  sorry

end arithmetic_sequence_sum_of_cubes_l3258_325829


namespace principal_calculation_l3258_325845

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal is 44625 -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 1 / 100  -- 1% converted to decimal
  let time : ℕ := 9
  calculate_principal simple_interest rate time = 44625 := by
  sorry

end principal_calculation_l3258_325845


namespace arithmetic_sequence_property_l3258_325833

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - a 8 = -8 := by
sorry

end arithmetic_sequence_property_l3258_325833


namespace horner_third_step_value_l3258_325808

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 7*x^2 + 6*x - 3

def horner_step (n : ℕ) (x : ℝ) (coeffs : List ℝ) : ℝ :=
  match n, coeffs with
  | 0, _ => 0
  | n+1, a::rest => a + x * horner_step n x rest
  | _, _ => 0

theorem horner_third_step_value :
  let coeffs := [1, -2, 3, -7, 6, -3]
  let x := 2
  horner_step 3 x coeffs = -1 := by sorry

end horner_third_step_value_l3258_325808


namespace mary_gave_one_blue_crayon_l3258_325897

/-- Given that Mary initially has 5 green crayons and 8 blue crayons,
    gives 3 green crayons to Becky, and has 9 crayons left afterwards,
    prove that Mary gave 1 blue crayon to Becky. -/
theorem mary_gave_one_blue_crayon 
  (initial_green : Nat) 
  (initial_blue : Nat)
  (green_given : Nat)
  (total_left : Nat)
  (h1 : initial_green = 5)
  (h2 : initial_blue = 8)
  (h3 : green_given = 3)
  (h4 : total_left = 9)
  (h5 : total_left = initial_green + initial_blue - green_given - blue_given)
  : blue_given = 1 := by
  sorry

#check mary_gave_one_blue_crayon

end mary_gave_one_blue_crayon_l3258_325897


namespace number_of_clerks_l3258_325880

/-- Proves that the number of clerks is 170 given the salary information -/
theorem number_of_clerks (total_avg : ℚ) (officer_avg : ℚ) (clerk_avg : ℚ) (num_officers : ℕ) :
  total_avg = 90 →
  officer_avg = 600 →
  clerk_avg = 84 →
  num_officers = 2 →
  ∃ (num_clerks : ℕ), 
    (num_officers * officer_avg + num_clerks * clerk_avg) / (num_officers + num_clerks) = total_avg ∧
    num_clerks = 170 := by
  sorry


end number_of_clerks_l3258_325880


namespace fixed_points_of_quadratic_l3258_325873

/-- A quadratic function of the form f(x) = mx^2 - 2mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 3

/-- Theorem stating that (0, 3) and (2, 3) are fixed points of f for all non-zero m -/
theorem fixed_points_of_quadratic (m : ℝ) (h : m ≠ 0) :
  (f m 0 = 3) ∧ (f m 2 = 3) := by
  sorry

#check fixed_points_of_quadratic

end fixed_points_of_quadratic_l3258_325873


namespace min_value_and_t_value_l3258_325888

theorem min_value_and_t_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 2/b = 2) :
  (∃ (min : ℝ), min = 4 ∧ ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 2 → 2*x + y ≥ min) ∧
  (∃ (t : ℝ), t = 6 ∧ 4^a = t ∧ 3^b = t) :=
by sorry

end min_value_and_t_value_l3258_325888


namespace tangent_circles_t_value_l3258_325871

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y t : ℝ) : Prop := (x - t)^2 + y^2 = 1

-- Define the condition of external tangency
def externally_tangent (t : ℝ) : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y t

-- Theorem statement
theorem tangent_circles_t_value :
  ∀ t : ℝ, externally_tangent t → (t = 3 ∨ t = -3) :=
by sorry

end tangent_circles_t_value_l3258_325871


namespace pyramid_volume_l3258_325855

theorem pyramid_volume (total_surface_area : ℝ) (triangular_face_ratio : ℝ) :
  total_surface_area = 600 →
  triangular_face_ratio = 2 →
  ∃ (volume : ℝ),
    volume = (1/3) * (total_surface_area / (4 * triangular_face_ratio + 1)) * 
             (Real.sqrt ((4 * triangular_face_ratio + 1) * 
             (4 * triangular_face_ratio - 1) / (triangular_face_ratio^2))) *
             Real.sqrt (total_surface_area / (4 * triangular_face_ratio + 1)) :=
by sorry

end pyramid_volume_l3258_325855


namespace inequality_proof_l3258_325899

theorem inequality_proof (a b c : ℝ) (ha : a = 31/32) (hb : b = Real.cos (1/4)) (hc : c = 4 * Real.sin (1/4)) : c > b ∧ b > a := by
  sorry

end inequality_proof_l3258_325899


namespace parallel_vectors_fraction_l3258_325818

theorem parallel_vectors_fraction (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, 3/2)
  let b : ℝ × ℝ := (Real.cos x, -1)
  (a.1 * b.2 = a.2 * b.1) →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4/3 := by
sorry

end parallel_vectors_fraction_l3258_325818


namespace arctan_sum_in_triangle_l3258_325805

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (angleC : ℝ)
  (pos_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (pos_angleC : angleC > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b)

-- State the theorem
theorem arctan_sum_in_triangle (t : Triangle) : 
  Real.arctan (t.a / (t.b + t.c - t.a)) + Real.arctan (t.b / (t.a + t.c - t.b)) = π / 4 := by
  sorry

end arctan_sum_in_triangle_l3258_325805


namespace football_game_attendance_l3258_325877

theorem football_game_attendance (S : ℕ) 
  (hMonday : ℕ → ℕ := λ x => x - 20)
  (hWednesday : ℕ → ℕ := λ x => x + 50)
  (hFriday : ℕ → ℕ := λ x => x * 2 - 20)
  (hExpected : ℕ := 350)
  (hActual : ℕ := hExpected + 40)
  (hTotal : ℕ → ℕ := λ x => x + hMonday x + hWednesday (hMonday x) + hFriday x) :
  hTotal S = hActual → S = 80 := by
sorry

end football_game_attendance_l3258_325877


namespace grace_walk_distance_l3258_325826

/-- The number of blocks Grace walked south -/
def blocks_south : ℕ := 4

/-- The number of blocks Grace walked west -/
def blocks_west : ℕ := 8

/-- The length of one block in miles -/
def block_length : ℚ := 1 / 4

/-- The total distance Grace walked in miles -/
def total_distance : ℚ := (blocks_south + blocks_west : ℚ) * block_length

theorem grace_walk_distance :
  total_distance = 3 := by sorry

end grace_walk_distance_l3258_325826


namespace square_sum_equals_thirty_l3258_325868

theorem square_sum_equals_thirty (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b = 7) : 
  a^2 + b^2 = 30 := by sorry

end square_sum_equals_thirty_l3258_325868


namespace sector_perimeter_l3258_325866

/-- Given a sector with area 2 and central angle 4 radians, its perimeter is 6. -/
theorem sector_perimeter (A : ℝ) (θ : ℝ) (r : ℝ) (P : ℝ) : 
  A = 2 → θ = 4 → A = (1/2) * r^2 * θ → P = r * θ + 2 * r → P = 6 := by
  sorry

end sector_perimeter_l3258_325866


namespace exists_x_sqrt_x_squared_neq_x_l3258_325858

theorem exists_x_sqrt_x_squared_neq_x : ∃ x : ℝ, Real.sqrt (x^2) ≠ x := by
  sorry

end exists_x_sqrt_x_squared_neq_x_l3258_325858


namespace arithmetic_sequence_sum_l3258_325874

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) :=
by
  sorry

end arithmetic_sequence_sum_l3258_325874


namespace solution_replacement_fraction_l3258_325820

theorem solution_replacement_fraction (Q : ℝ) (h : Q > 0) :
  let initial_conc : ℝ := 0.70
  let replacement_conc : ℝ := 0.25
  let new_conc : ℝ := 0.35
  let x : ℝ := (new_conc * Q - initial_conc * Q) / (replacement_conc * Q - initial_conc * Q)
  x = 7 / 9 := by
sorry

end solution_replacement_fraction_l3258_325820


namespace sum_of_products_nonzero_l3258_325840

/-- A 25x25 matrix with entries either 1 or -1 -/
def SignMatrix := Matrix (Fin 25) (Fin 25) Int

/-- Predicate to check if a matrix is a valid SignMatrix -/
def isValidSignMatrix (M : SignMatrix) : Prop :=
  ∀ i j, M i j = 1 ∨ M i j = -1

/-- Product of elements in a row -/
def rowProduct (M : SignMatrix) (i : Fin 25) : Int :=
  (List.range 25).foldl (fun acc j => acc * M i j) 1

/-- Product of elements in a column -/
def colProduct (M : SignMatrix) (j : Fin 25) : Int :=
  (List.range 25).foldl (fun acc i => acc * M i j) 1

/-- Sum of all row and column products -/
def sumOfProducts (M : SignMatrix) : Int :=
  (List.range 25).foldl (fun acc i => acc + rowProduct M i) 0 +
  (List.range 25).foldl (fun acc j => acc + colProduct M j) 0

theorem sum_of_products_nonzero (M : SignMatrix) (h : isValidSignMatrix M) :
  sumOfProducts M ≠ 0 := by
  sorry


end sum_of_products_nonzero_l3258_325840


namespace extreme_value_implies_a_equals_negative_one_l3258_325822

/-- Given a function f(x) = ax^2 + x^2 that reaches an extreme value at x = -2,
    prove that a = -1 --/
theorem extreme_value_implies_a_equals_negative_one (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + x^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-2 - ε) (-2 + ε), f x ≤ f (-2) ∨ f x ≥ f (-2)) →
  a = -1 :=
by sorry

end extreme_value_implies_a_equals_negative_one_l3258_325822


namespace prove_A_equals_five_l3258_325862

/-- Given that 14A and B73 are three-digit numbers, 14A + B73 = 418, and A and B are single digits, prove that A = 5 -/
theorem prove_A_equals_five (A B : ℕ) : 
  (100 ≤ 14 * A) ∧ (14 * A < 1000) ∧  -- 14A is a three-digit number
  (100 ≤ B * 100 + 73) ∧ (B * 100 + 73 < 1000) ∧  -- B73 is a three-digit number
  (14 * A + B * 100 + 73 = 418) ∧  -- 14A + B73 = 418
  (A < 10) ∧ (B < 10) →  -- A and B are single digits
  A = 5 := by sorry

end prove_A_equals_five_l3258_325862


namespace inequality_equivalence_l3258_325806

theorem inequality_equivalence (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ -4 ≤ x ∧ x < -3/2 :=
by sorry

end inequality_equivalence_l3258_325806


namespace smallest_four_digit_multiple_of_17_l3258_325803

theorem smallest_four_digit_multiple_of_17 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → 1003 ≤ n :=
by sorry

end smallest_four_digit_multiple_of_17_l3258_325803


namespace smallest_N_bound_l3258_325839

theorem smallest_N_bound (x : ℝ) (h : |x - 2| < 0.01) : 
  |x^2 - 4| < 0.0401 ∧ 
  ∀ ε > 0, ∃ y : ℝ, |y - 2| < 0.01 ∧ |y^2 - 4| ≥ 0.0401 - ε :=
sorry

end smallest_N_bound_l3258_325839


namespace reflection_of_point_A_l3258_325893

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point over the origin -/
def reflectOverOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

theorem reflection_of_point_A :
  let A : Point3D := { x := 2, y := 3, z := 4 }
  reflectOverOrigin A = { x := -2, y := -3, z := -4 } := by
  sorry

#check reflection_of_point_A

end reflection_of_point_A_l3258_325893


namespace max_distance_and_total_travel_l3258_325898

/-- Represents a car in the problem -/
structure Car where
  fuelCapacity : ℕ
  fuelEfficiency : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  car : Car
  numCars : ℕ

/-- Defines the problem parameters -/
def problem : ProblemSetup :=
  { car := { fuelCapacity := 24, fuelEfficiency := 60 },
    numCars := 2 }

/-- Theorem stating the maximum distance and total distance traveled -/
theorem max_distance_and_total_travel (p : ProblemSetup)
  (h1 : p.numCars = 2)
  (h2 : p.car.fuelCapacity = 24)
  (h3 : p.car.fuelEfficiency = 60) :
  ∃ (maxDistance totalDistance : ℕ),
    maxDistance = 360 ∧
    totalDistance = 2160 ∧
    maxDistance ≤ (p.car.fuelCapacity * p.car.fuelEfficiency) / 2 ∧
    totalDistance = maxDistance * 2 * 3 := by
  sorry

#check max_distance_and_total_travel

end max_distance_and_total_travel_l3258_325898


namespace opposite_of_negative_fraction_l3258_325811

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end opposite_of_negative_fraction_l3258_325811


namespace infinite_series_sum_l3258_325810

/-- Given positive real numbers c and d where c > d, the sum of the infinite series
    1/(cd) + 1/(c(3c-d)) + 1/((3c-d)(5c-2d)) + 1/((5c-2d)(7c-3d)) + ...
    is equal to 1/((c-d)d). -/
theorem infinite_series_sum (c d : ℝ) (hc : c > 0) (hd : d > 0) (h : c > d) :
  let series := fun n : ℕ => 1 / ((2 * n - 1) * c - (n - 1) * d) / ((2 * n + 1) * c - n * d)
  ∑' n, series n = 1 / ((c - d) * d) := by
  sorry

end infinite_series_sum_l3258_325810


namespace variance_sum_random_nonrandom_l3258_325837

/-- A random function -/
def RandomFunction (α : Type*) := α → ℝ

/-- A non-random function -/
def NonRandomFunction (α : Type*) := α → ℝ

/-- Variance of a random function -/
noncomputable def variance (X : RandomFunction ℝ) (t : ℝ) : ℝ := sorry

/-- The sum of a random function and a non-random function -/
def sumFunction (X : RandomFunction ℝ) (φ : NonRandomFunction ℝ) : RandomFunction ℝ :=
  fun t => X t + φ t

/-- Theorem: The variance of the sum of a random function and a non-random function
    is equal to the variance of the random function -/
theorem variance_sum_random_nonrandom
  (X : RandomFunction ℝ) (φ : NonRandomFunction ℝ) (t : ℝ) :
  variance (sumFunction X φ) t = variance X t := by sorry

end variance_sum_random_nonrandom_l3258_325837


namespace bathroom_tiles_count_l3258_325849

-- Define the bathroom dimensions in feet
def bathroom_length : ℝ := 10
def bathroom_width : ℝ := 6

-- Define the tile side length in inches
def tile_side : ℝ := 6

-- Define the conversion factor from feet to inches
def inches_per_foot : ℝ := 12

theorem bathroom_tiles_count :
  (bathroom_length * inches_per_foot) * (bathroom_width * inches_per_foot) / (tile_side * tile_side) = 240 := by
  sorry

end bathroom_tiles_count_l3258_325849


namespace polynomial_uniqueness_l3258_325842

-- Define the polynomial Q
def Q (a b c d : ℝ) (x : ℝ) : ℝ := a + b * x + c * x^2 + d * x^3

-- State the theorem
theorem polynomial_uniqueness (a b c d : ℝ) :
  Q a b c d (-1) = 2 →
  Q a b c d = (fun x => x^3 - x^2 + x + 2) := by
  sorry

end polynomial_uniqueness_l3258_325842


namespace units_digit_product_l3258_325869

theorem units_digit_product (n : ℕ) : n = 3^401 * 7^402 * 23^403 → n % 10 = 9 := by
  sorry

end units_digit_product_l3258_325869


namespace solution_difference_l3258_325850

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem solution_difference (x y : ℝ) :
  (floor x : ℝ) + frac y = 3.7 →
  frac x + (floor y : ℝ) = 8.2 →
  |x - y| = 5.5 := by sorry

end solution_difference_l3258_325850


namespace comics_reassembly_l3258_325885

theorem comics_reassembly (pages_per_comic : ℕ) (torn_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 25 →
  torn_pages = 150 →
  untorn_comics = 5 →
  (torn_pages / pages_per_comic + untorn_comics : ℕ) = 11 := by
  sorry

end comics_reassembly_l3258_325885


namespace divides_power_plus_one_l3258_325857

theorem divides_power_plus_one (n : ℕ) : (3 ^ (n + 1)) ∣ (2 ^ (3 ^ n) + 1) := by
  sorry

end divides_power_plus_one_l3258_325857


namespace sin_sum_specific_angles_l3258_325892

theorem sin_sum_specific_angles (α β : Real) : 
  0 < α ∧ α < Real.pi → 
  0 < β ∧ β < Real.pi → 
  Real.cos α = -1/2 → 
  Real.sin β = Real.sqrt 3 / 2 → 
  Real.sin (α + β) = -3/4 := by
sorry

end sin_sum_specific_angles_l3258_325892


namespace network_coloring_l3258_325890

/-- A node in the network --/
structure Node where
  lines : Finset (Fin 10)

/-- A network of lines on a plane --/
structure Network where
  nodes : Finset Node
  adjacent : Node → Node → Prop

/-- A coloring of the network --/
def Coloring (n : Network) := Node → Fin 15

/-- A valid coloring of the network --/
def ValidColoring (n : Network) (c : Coloring n) : Prop :=
  ∀ (node1 node2 : Node), n.adjacent node1 node2 → c node1 ≠ c node2

/-- The main theorem: any network can be colored with at most 15 colors --/
theorem network_coloring (n : Network) : ∃ (c : Coloring n), ValidColoring n c := by
  sorry

end network_coloring_l3258_325890


namespace composition_value_l3258_325827

theorem composition_value (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 3)
  (h_comp : ∀ x, f (g x) = 15*x + d) : 
  d = 18 := by
sorry

end composition_value_l3258_325827


namespace three_valid_rental_plans_l3258_325864

/-- Represents a rental plan for vehicles --/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid for the given number of people --/
def isValidPlan (plan : RentalPlan) (totalPeople : ℕ) : Prop :=
  plan.typeA * 6 + plan.typeB * 4 = totalPeople

/-- Theorem stating that there are at least three different valid rental plans --/
theorem three_valid_rental_plans :
  ∃ (plan1 plan2 plan3 : RentalPlan),
    isValidPlan plan1 38 ∧
    isValidPlan plan2 38 ∧
    isValidPlan plan3 38 ∧
    plan1 ≠ plan2 ∧
    plan1 ≠ plan3 ∧
    plan2 ≠ plan3 := by
  sorry

end three_valid_rental_plans_l3258_325864


namespace jon_website_earnings_l3258_325863

/-- Calculates Jon's earnings from his website in a 30-day month -/
theorem jon_website_earnings : 
  let pay_per_visit : ℚ := 0.1
  let visits_per_hour : ℕ := 50
  let hours_per_day : ℕ := 24
  let days_in_month : ℕ := 30
  (pay_per_visit * visits_per_hour * hours_per_day * days_in_month : ℚ) = 3600 := by
  sorry

end jon_website_earnings_l3258_325863


namespace category_d_cost_after_discount_l3258_325812

/-- Represents the cost and discount information for a category of items --/
structure Category where
  percentage : Real
  discount_rate : Real

/-- Calculates the cost of items in a category after applying the discount --/
def cost_after_discount (total_cost : Real) (category : Category) : Real :=
  let cost_before_discount := total_cost * category.percentage
  cost_before_discount * (1 - category.discount_rate)

/-- Theorem stating that the cost of category D items after discount is 562.5 --/
theorem category_d_cost_after_discount (total_cost : Real) (category_d : Category) :
  total_cost = 2500 →
  category_d.percentage = 0.25 →
  category_d.discount_rate = 0.10 →
  cost_after_discount total_cost category_d = 562.5 := by
  sorry

#check category_d_cost_after_discount

end category_d_cost_after_discount_l3258_325812


namespace shorter_worm_length_l3258_325831

theorem shorter_worm_length (worm1_length worm2_length : Real) :
  worm1_length = 0.8 →
  worm2_length = worm1_length + 0.7 →
  min worm1_length worm2_length = 0.8 := by
sorry

end shorter_worm_length_l3258_325831
