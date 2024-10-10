import Mathlib

namespace parallel_intersection_false_l1339_133926

-- Define the types for planes and lines
variable (α β : Plane) (m n : Line)

-- Define the parallel and intersection relations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem parallel_intersection_false :
  ¬(∀ α β m n,
    (parallel m α ∧ intersection α β = n) → parallel_lines m n) :=
sorry

end parallel_intersection_false_l1339_133926


namespace expression_simplification_and_evaluation_l1339_133919

theorem expression_simplification_and_evaluation (m : ℤ) 
  (h1 : -2 ≤ m ∧ m ≤ 2) 
  (h2 : m ≠ -2 ∧ m ≠ 0 ∧ m ≠ 1 ∧ m ≠ 2) :
  (m / (m - 2) - 4 / (m^2 - 2*m)) / ((m + 2) / (m^2 - m)) = 
  ((m - 4) * (m + 1) * (m - 1)) / (m * (m - 2) * (m + 2)) ∧
  ((m - 4) * (m + 1) * (m - 1)) / (m * (m - 2) * (m + 2)) = 0 :=
by sorry

end expression_simplification_and_evaluation_l1339_133919


namespace mark_cereal_boxes_l1339_133913

def soup_cost : ℕ := 6 * 2
def bread_cost : ℕ := 2 * 5
def milk_cost : ℕ := 2 * 4
def cereal_cost : ℕ := 3
def total_payment : ℕ := 4 * 10

def cereal_boxes : ℕ := (total_payment - (soup_cost + bread_cost + milk_cost)) / cereal_cost

theorem mark_cereal_boxes : cereal_boxes = 3 := by
  sorry

end mark_cereal_boxes_l1339_133913


namespace max_b_for_inequality_solution_l1339_133915

theorem max_b_for_inequality_solution (b : ℝ) : 
  (∃ x : ℝ, b * (b ^ (1/2)) * (x^2 - 10*x + 25) + (b ^ (1/2)) / (x^2 - 10*x + 25) ≤ 
    (1/5) * (b ^ (3/4)) * |Real.sin (π * x / 10)|) 
  → b ≤ (1/10000) :=
sorry

end max_b_for_inequality_solution_l1339_133915


namespace monthly_spending_is_99_l1339_133918

def original_price : ℚ := 50
def price_increase_percent : ℚ := 10
def discount_percent : ℚ := 10
def monthly_purchase : ℚ := 2

def calculate_monthly_spending (original_price price_increase_percent discount_percent monthly_purchase : ℚ) : ℚ :=
  let new_price := original_price * (1 + price_increase_percent / 100)
  let discounted_price := new_price * (1 - discount_percent / 100)
  discounted_price * monthly_purchase

theorem monthly_spending_is_99 :
  calculate_monthly_spending original_price price_increase_percent discount_percent monthly_purchase = 99 := by
  sorry

end monthly_spending_is_99_l1339_133918


namespace pigeonhole_trees_leaves_l1339_133954

theorem pigeonhole_trees_leaves (n : ℕ) (L : ℕ → ℕ) 
  (h1 : ∀ i, i < n → L i < n) : 
  ∃ i j, i < n ∧ j < n ∧ i ≠ j ∧ L i = L j :=
by sorry

end pigeonhole_trees_leaves_l1339_133954


namespace two_digit_number_divisible_by_55_l1339_133958

theorem two_digit_number_divisible_by_55 (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → 
  (10 * a + b) % 55 = 0 → 
  (∀ (x y : ℕ), x ≤ 9 → y ≤ 9 → (10 * x + y) % 55 = 0 → x * y ≤ b * a) →
  b * a ≤ 15 →
  10 * a + b = 55 := by
sorry

end two_digit_number_divisible_by_55_l1339_133958


namespace multiples_of_ten_l1339_133990

theorem multiples_of_ten (n : ℕ) : 
  100 + (n - 1) * 10 = 10000 ↔ n = 991 :=
by sorry

#check multiples_of_ten

end multiples_of_ten_l1339_133990


namespace difference_of_squares_l1339_133981

theorem difference_of_squares (a : ℝ) : a^2 - 1 = (a + 1) * (a - 1) := by
  sorry

end difference_of_squares_l1339_133981


namespace restaurant_menu_fraction_l1339_133987

theorem restaurant_menu_fraction (total_vegan : ℕ) (total_menu : ℕ) (vegan_with_nuts : ℕ) :
  total_vegan = 6 →
  total_vegan = total_menu / 3 →
  vegan_with_nuts = 1 →
  (total_vegan - vegan_with_nuts : ℚ) / total_menu = 5 / 18 :=
by sorry

end restaurant_menu_fraction_l1339_133987


namespace reciprocal_of_negative_two_l1339_133934

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end reciprocal_of_negative_two_l1339_133934


namespace no_mem_is_veen_l1339_133969

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem En Veen : Set U) -- Subsets of U

-- State the theorem
theorem no_mem_is_veen 
  (h1 : Mem ⊆ En) -- All Mems are Ens
  (h2 : En ∩ Veen = ∅) -- No Ens are Veens
  : Mem ∩ Veen = ∅ := -- No Mem is a Veen
by
  sorry

end no_mem_is_veen_l1339_133969


namespace speed_ratio_l1339_133975

-- Define the speeds of A and B
def speed_A : ℝ := sorry
def speed_B : ℝ := sorry

-- Define the initial position of B
def initial_B : ℝ := -800

-- Define the equidistant condition after 1 minute
def equidistant_1 : Prop :=
  speed_A = |initial_B + speed_B|

-- Define the equidistant condition after 7 minutes
def equidistant_7 : Prop :=
  7 * speed_A = |initial_B + 7 * speed_B|

-- Theorem stating the ratio of speeds
theorem speed_ratio :
  equidistant_1 → equidistant_7 → speed_A / speed_B = 3 / 2 := by sorry

end speed_ratio_l1339_133975


namespace water_leak_proof_l1339_133977

/-- A linear function representing the total water amount over time -/
def water_function (k b : ℝ) (t : ℝ) : ℝ := k * t + b

theorem water_leak_proof (k b : ℝ) :
  water_function k b 1 = 7 →
  water_function k b 2 = 12 →
  (k = 5 ∧ b = 2) ∧
  water_function k b 20 = 102 ∧
  ((water_function k b 1440 * 30) / 1500 : ℝ) = 144 :=
by sorry


end water_leak_proof_l1339_133977


namespace min_four_dollar_frisbees_l1339_133944

/-- Given the conditions of frisbee sales, proves the minimum number of $4 frisbees sold -/
theorem min_four_dollar_frisbees 
  (total_frisbees : ℕ) 
  (price_low price_high : ℕ) 
  (total_receipts : ℕ) 
  (h_total : total_frisbees = 60)
  (h_price_low : price_low = 3)
  (h_price_high : price_high = 4)
  (h_receipts : total_receipts = 200) :
  ∃ (x y : ℕ), 
    x + y = total_frisbees ∧ 
    price_low * x + price_high * y = total_receipts ∧
    y ≥ 20 :=
sorry

end min_four_dollar_frisbees_l1339_133944


namespace fuel_mixture_problem_l1339_133921

/-- Proves the volume of fuel A in a partially filled tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 214 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (volume_a : ℝ), 
    volume_a + (tank_capacity - volume_a) = tank_capacity ∧
    ethanol_a * volume_a + ethanol_b * (tank_capacity - volume_a) = total_ethanol ∧
    volume_a = 106 := by
  sorry

end fuel_mixture_problem_l1339_133921


namespace trajectory_equation_of_M_l1339_133988

theorem trajectory_equation_of_M (x y : ℝ) (h : y ≠ 0) :
  let P : ℝ × ℝ := (x, 3/2 * y)
  (P.1^2 + P.2^2 = 1) →
  (x^2 + (9 * y^2) / 4 = 1) :=
by sorry

end trajectory_equation_of_M_l1339_133988


namespace remainder_theorem_l1339_133936

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - 7*x^2 + x - 30

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), 
    polynomial x = (divisor x) * (q x) + 234 := by
  sorry

end remainder_theorem_l1339_133936


namespace simple_interest_calculation_l1339_133982

/-- Calculate simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proof that the simple interest for the given conditions is 4016.25 -/
theorem simple_interest_calculation :
  let principal : ℚ := 44625
  let rate : ℚ := 1
  let time : ℚ := 9
  simpleInterest principal rate time = 4016.25 := by
  sorry

#eval simpleInterest 44625 1 9

end simple_interest_calculation_l1339_133982


namespace special_function_monotonicity_l1339_133931

/-- A function f: ℝ → ℝ satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (1 - x) = f x) ∧
  (∀ x, (x - 1/2) * (deriv^[2] f x) > 0)

/-- Theorem stating the monotonicity property of the special function -/
theorem special_function_monotonicity 
  (f : ℝ → ℝ) (hf : SpecialFunction f) (x₁ x₂ : ℝ) 
  (h_order : x₁ < x₂) (h_sum : x₁ + x₂ > 1) : 
  f x₁ < f x₂ := by
  sorry

end special_function_monotonicity_l1339_133931


namespace complex_number_properties_l1339_133949

open Complex

theorem complex_number_properties (z : ℂ) (h : I * (z + 1) = -2 + 2*I) : 
  z.im = 2 ∧ abs (z / (1 - 2*I)) ^ 2015 = 1 := by sorry

end complex_number_properties_l1339_133949


namespace similar_triangles_leg_sum_l1339_133903

theorem similar_triangles_leg_sum 
  (A₁ A₂ : ℝ) 
  (h_areas : A₁ = 12 ∧ A₂ = 192) 
  (h_similar : ∃ (k : ℝ), k > 0 ∧ A₂ = k^2 * A₁) 
  (a b : ℝ) 
  (h_right : a^2 + b^2 = 10^2) 
  (h_leg_ratio : a = 2*b) 
  (h_area_small : A₁ = 1/2 * a * b) : 
  ∃ (c d : ℝ), c^2 + d^2 = (4*10)^2 ∧ A₂ = 1/2 * c * d ∧ c + d = 24 * Real.sqrt 3 := by
sorry


end similar_triangles_leg_sum_l1339_133903


namespace quiz_smallest_n_l1339_133964

/-- The smallest possible value of n in the quiz problem -/
theorem quiz_smallest_n : ∃ (n : ℤ), n = 89 ∧ 
  ∀ (m : ℕ+) (n' : ℤ),
  (m : ℤ) * (n' + 2) - m * (m + 1) = 2009 →
  n ≤ n' :=
by sorry

end quiz_smallest_n_l1339_133964


namespace sara_peaches_l1339_133930

theorem sara_peaches (initial_peaches additional_peaches : ℝ) 
  (h1 : initial_peaches = 61.0) 
  (h2 : additional_peaches = 24.0) : 
  initial_peaches + additional_peaches = 85.0 := by
  sorry

end sara_peaches_l1339_133930


namespace probability_two_red_balls_l1339_133998

/-- The probability of selecting two red balls from a bag with given ball counts. -/
theorem probability_two_red_balls (red blue green : ℕ) (h : red = 5 ∧ blue = 6 ∧ green = 2) :
  let total := red + blue + green
  let choose_two (n : ℕ) := n * (n - 1) / 2
  (choose_two red : ℚ) / (choose_two total) = 5 / 39 := by
  sorry

end probability_two_red_balls_l1339_133998


namespace cubic_root_sum_fourth_power_l1339_133970

theorem cubic_root_sum_fourth_power (p q r : ℝ) : 
  (p^3 - p^2 + 2*p - 3 = 0) → 
  (q^3 - q^2 + 2*q - 3 = 0) → 
  (r^3 - r^2 + 2*r - 3 = 0) → 
  p^4 + q^4 + r^4 = 13 := by
sorry

end cubic_root_sum_fourth_power_l1339_133970


namespace special_triangle_angle_exists_l1339_133939

/-- A triangle with a circumcircle where one altitude is tangent to the circumcircle -/
structure SpecialTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The circumcircle of the triangle -/
  circumcircle : Set (ℝ × ℝ)
  /-- An altitude of the triangle -/
  altitude : Set (ℝ × ℝ)
  /-- The altitude is tangent to the circumcircle -/
  is_tangent : altitude ∩ circumcircle ≠ ∅

/-- The angles of a triangle -/
def angles (t : SpecialTriangle) : Set ℝ := sorry

/-- Theorem: In a SpecialTriangle, there exists an angle greater than 90° and less than 135° -/
theorem special_triangle_angle_exists (t : SpecialTriangle) :
  ∃ θ ∈ angles t, 90 < θ ∧ θ < 135 := by sorry

end special_triangle_angle_exists_l1339_133939


namespace justin_bought_two_striped_jerseys_l1339_133976

/-- The number of striped jerseys Justin bought -/
def num_striped_jerseys : ℕ := 2

/-- The cost of each long-sleeved jersey -/
def long_sleeve_cost : ℕ := 15

/-- The number of long-sleeved jerseys Justin bought -/
def num_long_sleeve : ℕ := 4

/-- The cost of each striped jersey before discount -/
def striped_cost : ℕ := 10

/-- The discount applied to each striped jersey after the first one -/
def striped_discount : ℕ := 2

/-- The total amount Justin spent -/
def total_spent : ℕ := 80

/-- Theorem stating that Justin bought 2 striped jerseys given the conditions -/
theorem justin_bought_two_striped_jerseys :
  num_long_sleeve * long_sleeve_cost +
  striped_cost +
  (num_striped_jerseys - 1) * (striped_cost - striped_discount) =
  total_spent :=
sorry

end justin_bought_two_striped_jerseys_l1339_133976


namespace integer_triplet_sum_product_l1339_133927

theorem integer_triplet_sum_product (a b c : ℤ) : 
  a < 4 ∧ b < 4 ∧ c < 4 ∧ 
  a < b ∧ b < c ∧ 
  a + b + c = a * b * c →
  ((a, b, c) = (1, 2, 3) ∨ 
   (a, b, c) = (-3, -2, -1) ∨ 
   (a, b, c) = (-1, 0, 1) ∨ 
   (a, b, c) = (-2, 0, 2) ∨ 
   (a, b, c) = (-3, 0, 3)) :=
by sorry

end integer_triplet_sum_product_l1339_133927


namespace new_ratio_after_addition_l1339_133993

theorem new_ratio_after_addition (a b c : ℤ) : 
  (3 * a = b) → 
  (b = 15) → 
  (c = a + 10) → 
  (c = b) := by sorry

end new_ratio_after_addition_l1339_133993


namespace fraction_equivalence_l1339_133925

theorem fraction_equivalence : ∃ n : ℚ, (4 + n) / (7 + n) = 2 / 3 :=
by
  use 2
  sorry

#check fraction_equivalence

end fraction_equivalence_l1339_133925


namespace project_cost_increase_l1339_133909

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

theorem project_cost_increase :
  let initial_total_cost := initial_lumber_cost + initial_nails_cost + initial_fabric_cost
  let new_lumber_cost := initial_lumber_cost * (1 + lumber_inflation_rate)
  let new_nails_cost := initial_nails_cost * (1 + nails_inflation_rate)
  let new_fabric_cost := initial_fabric_cost * (1 + fabric_inflation_rate)
  let new_total_cost := new_lumber_cost + new_nails_cost + new_fabric_cost
  new_total_cost - initial_total_cost = 97 := by
sorry

end project_cost_increase_l1339_133909


namespace exists_divisible_sum_squares_l1339_133917

/-- For any polynomial P with real coefficients and positive integer n,
    there exists a polynomial Q with real coefficients such that
    (1+x^2)^n divides P(x)^2 + Q(x)^2. -/
theorem exists_divisible_sum_squares (P : Polynomial ℝ) (n : ℕ) (hn : n > 0) :
  ∃ Q : Polynomial ℝ, (1 + X^2)^n ∣ P^2 + Q^2 := by
  sorry

#check exists_divisible_sum_squares

end exists_divisible_sum_squares_l1339_133917


namespace building_dimension_difference_l1339_133912

-- Define the building structure
structure Building where
  floor1_length : ℝ
  floor1_width : ℝ
  floor2_length : ℝ
  floor2_width : ℝ

-- Define the conditions
def building_conditions (b : Building) : Prop :=
  b.floor1_width = (1/2) * b.floor1_length ∧
  b.floor1_length * b.floor1_width = 578 ∧
  b.floor2_width = (1/3) * b.floor2_length ∧
  b.floor2_length * b.floor2_width = 450

-- Define the combined length and width
def combined_length (b : Building) : ℝ := b.floor1_length + b.floor2_length
def combined_width (b : Building) : ℝ := b.floor1_width + b.floor2_width

-- Theorem statement
theorem building_dimension_difference (b : Building) 
  (h : building_conditions b) : 
  ∃ ε > 0, |combined_length b - combined_width b - 41.494| < ε :=
sorry

end building_dimension_difference_l1339_133912


namespace unique_functional_equation_solution_l1339_133932

theorem unique_functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) = x * f x + y * f y) →
  (∀ x : ℝ, f x = 0) := by
  sorry

end unique_functional_equation_solution_l1339_133932


namespace investment_profit_l1339_133908

/-- The daily price increase rate of the shares -/
def daily_increase : ℝ := 1.1

/-- The amount spent on shares each day in rubles -/
def daily_investment : ℝ := 1000

/-- The number of days the businessman buys shares -/
def investment_days : ℕ := 3

/-- The number of days until the shares are sold -/
def total_days : ℕ := 4

/-- Calculate the total profit from the share investment -/
def calculate_profit : ℝ :=
  let total_investment := daily_investment * investment_days
  let total_value := daily_investment * (daily_increase^3 + daily_increase^2 + daily_increase)
  total_value - total_investment

theorem investment_profit :
  calculate_profit = 641 := by sorry

end investment_profit_l1339_133908


namespace no_14_consecutive_integers_exists_21_consecutive_integers_l1339_133907

-- Define the set of primes for part 1
def primes1 : Set ℕ := {2, 3, 5, 7, 11}

-- Define the set of primes for part 2
def primes2 : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to check if a number is divisible by any prime in a given set
def divisibleByAnyPrime (n : ℕ) (primes : Set ℕ) : Prop :=
  ∃ p ∈ primes, n % p = 0

-- Part 1: No set of 14 consecutive integers satisfies the condition
theorem no_14_consecutive_integers : 
  ¬∃ n : ℕ, ∀ k ∈ Finset.range 14, divisibleByAnyPrime (n + k) primes1 := by
sorry

-- Part 2: There exists a set of 21 consecutive integers that satisfies the condition
theorem exists_21_consecutive_integers : 
  ∃ n : ℕ, ∀ k ∈ Finset.range 21, divisibleByAnyPrime (n + k) primes2 := by
sorry

end no_14_consecutive_integers_exists_21_consecutive_integers_l1339_133907


namespace arrangement_count_l1339_133985

theorem arrangement_count : ℕ := by
  -- Define the total number of people
  let total_people : ℕ := 7

  -- Define the number of boys and girls
  let num_boys : ℕ := 5
  let num_girls : ℕ := 2

  -- Define that boy A must be in the middle
  let boy_A_position : ℕ := (total_people + 1) / 2

  -- Define that the girls must be adjacent
  let girls_adjacent : Prop := true

  -- The number of ways to arrange them
  let arrangement_ways : ℕ := 192

  -- Proof goes here
  sorry

end arrangement_count_l1339_133985


namespace zach_cookies_l1339_133961

/-- The number of cookies Zach baked over three days --/
def total_cookies (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating the total number of cookies Zach had after three days --/
theorem zach_cookies : ∃ (monday tuesday wednesday : ℕ),
  monday = 32 ∧
  tuesday = monday / 2 ∧
  wednesday = tuesday * 3 - 4 ∧
  total_cookies monday tuesday wednesday = 92 := by
  sorry

end zach_cookies_l1339_133961


namespace distance_between_signs_l1339_133960

theorem distance_between_signs (total_distance : ℕ) (distance_to_first_sign : ℕ) (distance_after_second_sign : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_to_first_sign - distance_after_second_sign = 375 := by
  sorry

end distance_between_signs_l1339_133960


namespace point_on_x_axis_l1339_133991

/-- Given a point Q(a-1, a+2) that lies on the x-axis, prove that its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ Q : ℝ × ℝ, Q.1 = a - 1 ∧ Q.2 = a + 2 ∧ Q.2 = 0) → 
  (∃ Q : ℝ × ℝ, Q = (-3, 0)) :=
by sorry

end point_on_x_axis_l1339_133991


namespace class_average_problem_l1339_133940

theorem class_average_problem (n1 n2 : ℕ) (avg2 avg_all : ℝ) (h1 : n1 = 30) (h2 : n2 = 50) 
  (h3 : avg2 = 60) (h4 : avg_all = 56.25) : 
  (n1 + n2 : ℝ) * avg_all = n1 * ((n1 + n2 : ℝ) * avg_all - n2 * avg2) / n1 + n2 * avg2 := by
  sorry

#check class_average_problem

end class_average_problem_l1339_133940


namespace circle_equation_l1339_133900

/-- Given a circle with radius 5 and a line l: x + 2y - 3 = 0 tangent to the circle at point P(1,1),
    prove that the equations of the circle are:
    (x-1-√5)² + (y-1-2√5)² = 25 and (x-1+√5)² + (y-1+2√5)² = 25 -/
theorem circle_equation (x y : ℝ) :
  let r : ℝ := 5
  let l : ℝ → ℝ → ℝ := fun x y ↦ x + 2*y - 3
  let P : ℝ × ℝ := (1, 1)
  (∃ (center : ℝ × ℝ), (center.1 - P.1)^2 + (center.2 - P.2)^2 = r^2 ∧
    l P.1 P.2 = 0 ∧
    (∀ (t : ℝ), t ≠ 0 → l (P.1 + t) (P.2 + t * ((center.2 - P.2) / (center.1 - P.1))) ≠ 0)) →
  ((x - (1 - Real.sqrt 5))^2 + (y - (1 - 2 * Real.sqrt 5))^2 = 25) ∨
  ((x - (1 + Real.sqrt 5))^2 + (y - (1 + 2 * Real.sqrt 5))^2 = 25) :=
by sorry

end circle_equation_l1339_133900


namespace product_of_smallest_primes_smallestOneDigitPrimes_are_prime_smallestTwoDigitPrime_is_prime_l1339_133963

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define the two smallest one-digit primes
def smallestOneDigitPrimes : Fin 2 → ℕ
| 0 => 2
| 1 => 3

-- Define the smallest two-digit prime
def smallestTwoDigitPrime : ℕ := 11

-- Theorem statement
theorem product_of_smallest_primes : 
  (smallestOneDigitPrimes 0) * (smallestOneDigitPrimes 1) * smallestTwoDigitPrime = 66 :=
by
  sorry

-- Prove that the defined numbers are indeed prime
theorem smallestOneDigitPrimes_are_prime :
  ∀ i : Fin 2, isPrime (smallestOneDigitPrimes i) :=
by
  sorry

theorem smallestTwoDigitPrime_is_prime :
  isPrime smallestTwoDigitPrime :=
by
  sorry

end product_of_smallest_primes_smallestOneDigitPrimes_are_prime_smallestTwoDigitPrime_is_prime_l1339_133963


namespace ariel_current_age_l1339_133997

/-- Calculates Ariel's current age based on given information -/
theorem ariel_current_age :
  let birth_year : ℕ := 1992
  let fencing_start_year : ℕ := 2006
  let years_fencing : ℕ := 16
  let current_year : ℕ := fencing_start_year + years_fencing
  let current_age : ℕ := current_year - birth_year
  current_age = 30 := by sorry

end ariel_current_age_l1339_133997


namespace smallest_ef_minus_de_l1339_133992

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.de + t.fd > t.ef ∧ t.ef + t.fd > t.de

/-- Theorem: The smallest possible value of EF - DE is 1 for a triangle DEF 
    with integer side lengths, perimeter 2010, and DE < EF ≤ FD -/
theorem smallest_ef_minus_de :
  ∀ t : Triangle,
    t.de + t.ef + t.fd = 2010 →
    t.de < t.ef →
    t.ef ≤ t.fd →
    isValidTriangle t →
    (∀ t' : Triangle,
      t'.de + t'.ef + t'.fd = 2010 →
      t'.de < t'.ef →
      t'.ef ≤ t'.fd →
      isValidTriangle t' →
      t'.ef - t'.de ≥ t.ef - t.de) →
    t.ef - t.de = 1 := by
  sorry

end smallest_ef_minus_de_l1339_133992


namespace block_run_difference_l1339_133950

/-- The difference in distance run around a square block between outer and inner paths -/
theorem block_run_difference (block_side : ℝ) (street_width : ℝ) : 
  block_side = 500 → street_width = 30 → 
  (4 * (block_side + street_width / 2) * π / 2) = 1030 * π := by sorry

end block_run_difference_l1339_133950


namespace jessica_pie_count_l1339_133941

theorem jessica_pie_count (apples_per_serving : ℝ) (num_guests : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ)
  (h1 : apples_per_serving = 1.5)
  (h2 : num_guests = 12)
  (h3 : servings_per_pie = 8)
  (h4 : apples_per_guest = 3) :
  (num_guests * apples_per_guest / apples_per_serving) / servings_per_pie = 3 := by
  sorry

end jessica_pie_count_l1339_133941


namespace ellipse_min_sum_l1339_133989

theorem ellipse_min_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (3 * Real.sqrt 3)^2 / m^2 + 1 / n^2 = 1 → m + n ≥ 8 := by sorry

end ellipse_min_sum_l1339_133989


namespace pizza_buffet_theorem_l1339_133946

theorem pizza_buffet_theorem (A B C : ℕ+) :
  (A : ℚ) = 1.8 * B ∧
  (B : ℚ) = C / 8 ∧
  A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 8 →
  A = 2 ∧ B = 1 ∧ C = 8 := by
sorry

end pizza_buffet_theorem_l1339_133946


namespace triangle_angle_C_l1339_133968

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  (a = Real.sqrt 6) →
  (b = 2) →
  (B = π / 4) → -- 45° in radians
  (Real.tan A * Real.tan C > 1) →
  -- Conclusion
  C = 5 * π / 12 -- 75° in radians
:= by sorry

end triangle_angle_C_l1339_133968


namespace final_balloon_count_l1339_133962

def total_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  (brooke_initial + brooke_added) + ((tracy_initial + tracy_added) / 2)

theorem final_balloon_count :
  total_balloons 12 8 6 24 = 35 := by
  sorry

end final_balloon_count_l1339_133962


namespace power_six_times_three_six_l1339_133978

theorem power_six_times_three_six : 6^6 * 3^6 = 34012224 := by
  sorry

end power_six_times_three_six_l1339_133978


namespace min_value_expression_l1339_133959

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 16) :
  x^2 + 8*x*y + 16*y^2 + 4*z^2 ≥ 48 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 16 ∧ x₀^2 + 8*x₀*y₀ + 16*y₀^2 + 4*z₀^2 = 48 := by
  sorry

end min_value_expression_l1339_133959


namespace total_peaches_l1339_133920

theorem total_peaches (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h_red : red = 7) (h_yellow : yellow = 15) (h_green : green = 8) :
  red + yellow + green = 30 := by
  sorry

end total_peaches_l1339_133920


namespace f_composition_value_l1339_133951

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) else Real.tan x

theorem f_composition_value : f (f (3 * Real.pi / 4)) = 0 := by
  sorry

end f_composition_value_l1339_133951


namespace game_result_l1339_133916

-- Define the point function
def g (n : Nat) : Nat :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

-- Define Allie's rolls
def allie_rolls : List Nat := [3, 6, 5, 2, 4]

-- Define Betty's rolls
def betty_rolls : List Nat := [2, 6, 1, 4]

-- Calculate total points for a list of rolls
def total_points (rolls : List Nat) : Nat :=
  rolls.map g |>.sum

-- Theorem statement
theorem game_result :
  (total_points allie_rolls) * (total_points betty_rolls) = 308 := by
  sorry

end game_result_l1339_133916


namespace total_food_consumption_l1339_133983

/-- Calculates the total daily food consumption for two armies with different rations -/
theorem total_food_consumption
  (food_per_soldier_side1 : ℕ)
  (food_difference : ℕ)
  (soldiers_side1 : ℕ)
  (soldier_difference : ℕ)
  (h1 : food_per_soldier_side1 = 10)
  (h2 : food_difference = 2)
  (h3 : soldiers_side1 = 4000)
  (h4 : soldier_difference = 500) :
  let food_per_soldier_side2 := food_per_soldier_side1 - food_difference
  let soldiers_side2 := soldiers_side1 - soldier_difference
  soldiers_side1 * food_per_soldier_side1 + soldiers_side2 * food_per_soldier_side2 = 68000 := by
sorry


end total_food_consumption_l1339_133983


namespace beetle_speed_l1339_133947

/-- Beetle's speed in km/h given ant's speed and relative distance -/
theorem beetle_speed (ant_distance : ℝ) (time_minutes : ℝ) (beetle_relative_distance : ℝ) :
  ant_distance = 1000 →
  time_minutes = 30 →
  beetle_relative_distance = 0.9 →
  (ant_distance * beetle_relative_distance / time_minutes) * (60 / 1000) = 1.8 := by
  sorry

end beetle_speed_l1339_133947


namespace salt_solution_dilution_l1339_133905

theorem salt_solution_dilution (initial_volume : ℝ) (initial_salt_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 64 ∧ 
  initial_salt_percentage = 0.1 ∧ 
  added_water = 16 →
  let salt_amount := initial_volume * initial_salt_percentage
  let new_volume := initial_volume + added_water
  let final_salt_percentage := salt_amount / new_volume
  final_salt_percentage = 0.08 := by
sorry

end salt_solution_dilution_l1339_133905


namespace ben_picking_peas_l1339_133910

/-- Given that Ben can pick 56 sugar snap peas in 7 minutes, 
    prove that it takes 9 minutes to pick 72 sugar snap peas. -/
theorem ben_picking_peas (rate : ℚ) (h1 : rate = 56 / 7) : 72 / rate = 9 := by
  sorry

end ben_picking_peas_l1339_133910


namespace problem_statement_l1339_133937

theorem problem_statement (a b c : ℝ) : 
  a < b →
  (∀ x : ℝ, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -1 ∨ |x - 10| ≤ 2)) →
  a + 2 * b + 3 * c = 29 := by
  sorry

end problem_statement_l1339_133937


namespace inhabitable_earth_surface_fraction_l1339_133972

theorem inhabitable_earth_surface_fraction :
  let total_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  (land_fraction * inhabitable_land_fraction : ℚ) = 2 / 9 := by
  sorry

end inhabitable_earth_surface_fraction_l1339_133972


namespace fraction_problem_l1339_133953

theorem fraction_problem (n : ℝ) (h : (1/3) * (1/4) * n = 15) : 
  ∃ f : ℝ, f * n = 54 ∧ f = 3/10 := by
sorry

end fraction_problem_l1339_133953


namespace carries_payment_is_correct_l1339_133965

/-- Calculates Carrie's payment for clothes given the quantities and prices of items, and her mom's contribution ratio --/
def carries_payment (shirt_qty : ℕ) (shirt_price : ℚ) 
                    (pants_qty : ℕ) (pants_price : ℚ)
                    (jacket_qty : ℕ) (jacket_price : ℚ)
                    (skirt_qty : ℕ) (skirt_price : ℚ)
                    (shoes_qty : ℕ) (shoes_price : ℚ)
                    (mom_ratio : ℚ) : ℚ :=
  let total_cost := shirt_qty * shirt_price + 
                    pants_qty * pants_price + 
                    jacket_qty * jacket_price + 
                    skirt_qty * skirt_price + 
                    shoes_qty * shoes_price
  total_cost - (mom_ratio * total_cost)

/-- Theorem: Carrie's payment for clothes is $228.67 --/
theorem carries_payment_is_correct : 
  carries_payment 8 12 4 25 4 75 3 30 2 50 (2/3) = 228.67 := by
  sorry

end carries_payment_is_correct_l1339_133965


namespace quadratic_equation_roots_l1339_133971

theorem quadratic_equation_roots (p : ℝ) (x₁ x₂ : ℝ) : 
  p > 0 → 
  x₁^2 + p*x₁ + 1 = 0 → 
  x₂^2 + p*x₂ + 1 = 0 → 
  |x₁^2 - x₂^2| = p → 
  p = 5 := by
sorry

end quadratic_equation_roots_l1339_133971


namespace daltons_uncle_gift_l1339_133911

/-- The amount of money Dalton's uncle gave him -/
def uncles_gift (jump_rope_cost board_game_cost ball_cost savings needed_more : ℕ) : ℕ :=
  (jump_rope_cost + board_game_cost + ball_cost) - savings - needed_more

/-- Proof that Dalton's uncle gave him $13 -/
theorem daltons_uncle_gift :
  uncles_gift 7 12 4 6 4 = 13 := by
  sorry

end daltons_uncle_gift_l1339_133911


namespace females_watch_count_l1339_133933

/-- The number of people who watch WXLT -/
def total_watch : ℕ := 160

/-- The number of males who watch WXLT -/
def males_watch : ℕ := 85

/-- The number of females who don't watch WXLT -/
def females_dont_watch : ℕ := 120

/-- The total number of people who don't watch WXLT -/
def total_dont_watch : ℕ := 180

/-- The number of females who watch WXLT -/
def females_watch : ℕ := total_watch - males_watch

theorem females_watch_count : females_watch = 75 := by
  sorry

end females_watch_count_l1339_133933


namespace sqrt_50_between_consecutive_integers_product_l1339_133995

theorem sqrt_50_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1) ∧ n * (n + 1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_product_l1339_133995


namespace oil_leak_calculation_l1339_133979

/-- The amount of oil leaked before repairs, in liters -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked during repairs, in liters -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked, in liters -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem oil_leak_calculation :
  total_oil_leaked = 11687 :=
by sorry

end oil_leak_calculation_l1339_133979


namespace student_calculation_error_l1339_133942

theorem student_calculation_error (N : ℚ) : 
  (N / (4/5)) = ((4/5) * N + 45) → N = 100 := by
  sorry

end student_calculation_error_l1339_133942


namespace greatest_sum_consecutive_integers_l1339_133967

theorem greatest_sum_consecutive_integers (n : ℤ) : 
  (n * (n + 1) < 360) → (∀ m : ℤ, m > n → m * (m + 1) ≥ 360) → n + (n + 1) = 37 := by
  sorry

end greatest_sum_consecutive_integers_l1339_133967


namespace regular_polygon_properties_l1339_133928

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  diagonals : ℕ

/-- Calculate the number of diagonals in a polygon with n sides -/
def diagonalsCount (n : ℕ) : ℕ :=
  n * (n - 3) / 2

/-- Calculate the perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ :=
  p.sides * p.sideLength

theorem regular_polygon_properties :
  ∃ (p : RegularPolygon),
    p.diagonals = 15 ∧
    p.sideLength = 6 ∧
    p.sides = 7 ∧
    perimeter p = 42 :=
by sorry

end regular_polygon_properties_l1339_133928


namespace odd_function_zero_at_origin_l1339_133973

-- Define the function f on the interval [-1, 1]
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x

-- State the theorem
theorem odd_function_zero_at_origin (h : isOdd f) : f 0 = 0 := by
  sorry

end odd_function_zero_at_origin_l1339_133973


namespace age_difference_l1339_133924

/-- Given three people A, B, and C, where the total age of A and B is more than
    the total age of B and C, and C is 13 years younger than A, prove that the
    difference between (A + B) and (B + C) is 13 years. -/
theorem age_difference (A B C : ℕ) 
  (h1 : A + B > B + C) 
  (h2 : C = A - 13) : 
  (A + B) - (B + C) = 13 := by
  sorry

end age_difference_l1339_133924


namespace inequality_solution_l1339_133999

theorem inequality_solution (x : ℝ) :
  3 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9 * x + 1 →
  x > (5 + Real.sqrt 29) / 2 ∧ x < 11 :=
by sorry

end inequality_solution_l1339_133999


namespace shopping_cost_difference_l1339_133923

theorem shopping_cost_difference (shirt wallet food : ℝ) 
  (shirt_cost : shirt = wallet / 3)
  (wallet_more_expensive : wallet > food)
  (food_cost : food = 30)
  (total_spent : shirt + wallet + food = 150) :
  wallet - food = 60 := by
  sorry

end shopping_cost_difference_l1339_133923


namespace polygon_exists_l1339_133952

-- Define the number of matches and their length
def num_matches : ℕ := 12
def match_length : ℝ := 2

-- Define the target area
def target_area : ℝ := 16

-- Define a polygon as a list of points
def Polygon := List (ℝ × ℝ)

-- Function to calculate the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Function to calculate the area of a polygon
def area (p : Polygon) : ℝ := sorry

-- Theorem stating the existence of a polygon satisfying the conditions
theorem polygon_exists : 
  ∃ (p : Polygon), 
    perimeter p = num_matches * match_length ∧ 
    area p = target_area :=
sorry

end polygon_exists_l1339_133952


namespace icosikaipentagon_diagonals_from_vertex_l1339_133996

/-- The number of diagonals from a single vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- An icosikaipentagon is a polygon with 25 sides -/
def icosikaipentagon_sides : ℕ := 25

theorem icosikaipentagon_diagonals_from_vertex : 
  diagonals_from_vertex icosikaipentagon_sides = 22 := by
  sorry

end icosikaipentagon_diagonals_from_vertex_l1339_133996


namespace smallest_interesting_number_l1339_133902

theorem smallest_interesting_number :
  ∃ (n : ℕ), n = 1800 ∧
  (∀ m : ℕ, m < n →
    ¬(∃ k : ℕ, 2 * m = k ^ 2) ∨
    ¬(∃ l : ℕ, 15 * m = l ^ 3)) ∧
  (∃ k : ℕ, 2 * n = k ^ 2) ∧
  (∃ l : ℕ, 15 * n = l ^ 3) :=
by sorry

end smallest_interesting_number_l1339_133902


namespace pi_approximation_accuracy_l1339_133956

-- Define the approximation of π
def pi_approx : ℚ := 3.14

-- Define the true value of π (we'll use a rational approximation for simplicity)
def pi_true : ℚ := 355 / 113

-- Define the accuracy of the approximation
def accuracy : ℚ := 0.01

-- Theorem statement
theorem pi_approximation_accuracy :
  |pi_approx - pi_true| < accuracy :=
sorry

end pi_approximation_accuracy_l1339_133956


namespace new_person_weight_l1339_133980

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 4 →
  replaced_weight = 70 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 110 := by
  sorry

end new_person_weight_l1339_133980


namespace tracy_popped_half_balloons_l1339_133904

theorem tracy_popped_half_balloons 
  (brooke_balloons : ℕ) 
  (tracy_initial_balloons : ℕ) 
  (total_balloons_after_popping : ℕ) 
  (tracy_popped_fraction : ℚ) :
  brooke_balloons = 20 →
  tracy_initial_balloons = 30 →
  total_balloons_after_popping = 35 →
  brooke_balloons + tracy_initial_balloons * (1 - tracy_popped_fraction) = total_balloons_after_popping →
  tracy_popped_fraction = 1/2 := by
sorry

end tracy_popped_half_balloons_l1339_133904


namespace probability_of_selecting_two_specific_elements_l1339_133948

theorem probability_of_selecting_two_specific_elements 
  (total_elements : Nat) 
  (elements_to_select : Nat) 
  (h1 : total_elements = 6) 
  (h2 : elements_to_select = 2) :
  (1 : ℚ) / (Nat.choose total_elements elements_to_select) = 1 / 15 := by
  sorry

end probability_of_selecting_two_specific_elements_l1339_133948


namespace two_discounts_price_l1339_133957

/-- The final price of a product after two consecutive 10% discounts -/
def final_price (a : ℝ) : ℝ := a * (1 - 0.1)^2

/-- Theorem stating that the final price after two 10% discounts is correct -/
theorem two_discounts_price (a : ℝ) :
  final_price a = a * (1 - 0.1)^2 := by
  sorry

end two_discounts_price_l1339_133957


namespace train_journey_l1339_133935

theorem train_journey
  (average_speed : ℝ)
  (first_distance : ℝ)
  (first_time : ℝ)
  (second_time : ℝ)
  (h1 : average_speed = 70)
  (h2 : first_distance = 225)
  (h3 : first_time = 3.5)
  (h4 : second_time = 5)
  : (average_speed * (first_time + second_time) - first_distance) = 370 := by
  sorry

end train_journey_l1339_133935


namespace doodads_produced_l1339_133929

/-- Represents the production rate of gizmos per worker per hour -/
def gizmo_rate (workers : ℕ) (hours : ℕ) (gizmos : ℕ) : ℚ :=
  (gizmos : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Represents the production rate of doodads per worker per hour -/
def doodad_rate (workers : ℕ) (hours : ℕ) (doodads : ℕ) : ℚ :=
  (doodads : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Theorem stating the number of doodads produced by 40 workers in 4 hours -/
theorem doodads_produced
  (h1 : gizmo_rate 80 2 160 = gizmo_rate 70 3 210)
  (h2 : doodad_rate 80 2 240 = doodad_rate 70 3 420)
  (h3 : gizmo_rate 40 4 160 = gizmo_rate 80 2 160) :
  (doodad_rate 80 2 240 * (40 : ℚ) * 4) = 320 := by
  sorry

end doodads_produced_l1339_133929


namespace largest_valid_number_l1339_133914

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → (∃! p, 10^p * d ≤ n ∧ n < 10^(p+1) * d)) ∧
  n % 11 = 0

theorem largest_valid_number : 
  (∀ n : ℕ, is_valid_number n → n ≤ 987652413) ∧ 
  is_valid_number 987652413 := by sorry

end largest_valid_number_l1339_133914


namespace probability_non_adjacent_rational_terms_l1339_133938

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def non_adjacent_arrangements (irrational_terms rational_terms : ℕ) : ℕ :=
  Nat.factorial irrational_terms * Nat.choose (irrational_terms + 1) rational_terms

theorem probability_non_adjacent_rational_terms 
  (total_terms : ℕ) 
  (irrational_terms : ℕ) 
  (rational_terms : ℕ) 
  (h1 : total_terms = irrational_terms + rational_terms) 
  (h2 : irrational_terms = 6) 
  (h3 : rational_terms = 3) :
  (non_adjacent_arrangements irrational_terms rational_terms : ℚ) / 
  (total_arrangements total_terms : ℚ) = 5 / 24 := by
  sorry

end probability_non_adjacent_rational_terms_l1339_133938


namespace rectangle_triangle_same_area_altitude_l1339_133994

theorem rectangle_triangle_same_area_altitude (h : ℝ) (w : ℝ) : 
  h > 0 →  -- Altitude is positive
  w > 0 →  -- Width is positive
  12 * h = 12 * w →  -- Areas are equal (12h for triangle, 12w for rectangle)
  w = h :=  -- Width of rectangle equals shared altitude
by
  sorry

end rectangle_triangle_same_area_altitude_l1339_133994


namespace arithmetic_sequence_property_l1339_133974

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = 16)
  (h_a3 : a 3 = 1) :
  a 9 = 15 := by
sorry

end arithmetic_sequence_property_l1339_133974


namespace integer_root_iff_a_value_l1339_133984

def polynomial (a x : ℤ) : ℤ := x^3 + 3*x^2 + a*x - 7

theorem integer_root_iff_a_value (a : ℤ) : 
  (∃ x : ℤ, polynomial a x = 0) ↔ (a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3) := by sorry

end integer_root_iff_a_value_l1339_133984


namespace count_3033_arrangements_l1339_133986

/-- The set of digits in the number 3033 -/
def digits : Finset Nat := {0, 3}

/-- A function that counts the number of four-digit numbers that can be formed from the given digits -/
def countFourDigitNumbers (d : Finset Nat) : Nat :=
  (d.filter (· ≠ 0)).card * d.card * d.card * d.card

/-- Theorem stating that the number of different four-digit numbers formed from 3033 is 1 -/
theorem count_3033_arrangements : countFourDigitNumbers digits = 1 := by
  sorry

end count_3033_arrangements_l1339_133986


namespace square_of_recurring_third_l1339_133943

/-- The repeating decimal 0.333... --/
def recurring_third : ℚ := 1/3

/-- Theorem: The square of 0.333... is equal to 1/9 --/
theorem square_of_recurring_third : recurring_third ^ 2 = 1/9 := by
  sorry

end square_of_recurring_third_l1339_133943


namespace min_correct_answers_for_target_score_l1339_133966

/-- AMC 12 scoring system and problem parameters -/
structure AMC12Params where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Rat
  incorrect_penalty : Rat
  unanswered_points : Rat
  target_score : Rat

/-- Calculate the score based on the number of correct answers -/
def calculate_score (params : AMC12Params) (correct : Nat) : Rat :=
  let incorrect := params.attempted_problems - correct
  let unanswered := params.total_problems - params.attempted_problems
  correct * params.correct_points + 
  incorrect * (-params.incorrect_penalty) + 
  unanswered * params.unanswered_points

/-- The main theorem to prove -/
theorem min_correct_answers_for_target_score 
  (params : AMC12Params)
  (h_total : params.total_problems = 25)
  (h_attempted : params.attempted_problems = 15)
  (h_correct_points : params.correct_points = 7.5)
  (h_incorrect_penalty : params.incorrect_penalty = 2)
  (h_unanswered_points : params.unanswered_points = 2)
  (h_target_score : params.target_score = 120) :
  (∀ k < 14, calculate_score params k < params.target_score) ∧ 
  calculate_score params 14 ≥ params.target_score := by
  sorry

end min_correct_answers_for_target_score_l1339_133966


namespace even_polynomial_iff_composition_l1339_133922

open Polynomial

theorem even_polynomial_iff_composition (P : Polynomial ℝ) :
  (∀ x, P.eval (-x) = P.eval x) ↔ 
  ∃ Q : Polynomial ℝ, P = Q.comp (X ^ 2) :=
sorry

end even_polynomial_iff_composition_l1339_133922


namespace daughter_age_in_three_years_l1339_133955

/-- Given a mother's current age and the fact that she was twice her daughter's age 5 years ago,
    this function calculates the daughter's age in 3 years. -/
def daughters_future_age (mothers_current_age : ℕ) : ℕ :=
  let mothers_past_age := mothers_current_age - 5
  let daughters_past_age := mothers_past_age / 2
  let daughters_current_age := daughters_past_age + 5
  daughters_current_age + 3

/-- Theorem stating that given the problem conditions, the daughter will be 26 years old in 3 years. -/
theorem daughter_age_in_three_years :
  daughters_future_age 41 = 26 := by
  sorry

end daughter_age_in_three_years_l1339_133955


namespace quadratic_inequality_l1339_133901

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_inequality (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x < 0 ∨ x > 2) →
  (a = -2 ∧ b = 0) ∧
  (∀ m : ℝ,
    (m = 0 → ∀ x, ¬(f a b x < m^2 - 1)) ∧
    (m > 0 → ∀ x, f a b x < m^2 - 1 ↔ 1 - m < x ∧ x < 1 + m) ∧
    (m < 0 → ∀ x, f a b x < m^2 - 1 ↔ 1 + m < x ∧ x < 1 - m)) :=
by sorry

end quadratic_inequality_l1339_133901


namespace exists_k_for_prime_divisor_inequality_l1339_133945

/-- The largest prime divisor of a positive integer greater than 1 -/
def largest_prime_divisor (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For any odd prime q, there exists a positive integer k such that
    the largest prime divisor of (q^(2^k) - 1) is less than q, and
    q is less than the largest prime divisor of (q^(2^k) + 1) -/
theorem exists_k_for_prime_divisor_inequality (q : ℕ) (hq : q.Prime) (hq_odd : q % 2 = 1) :
  ∃ k : ℕ, k > 0 ∧
    largest_prime_divisor (q^(2^k) - 1) < q ∧
    q < largest_prime_divisor (q^(2^k) + 1) :=
  sorry

end exists_k_for_prime_divisor_inequality_l1339_133945


namespace inequality_solution_set_l1339_133906

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 - 2 ≥ 2*x - a*x}
  (a = 0 → S = {x : ℝ | x ≤ -1}) ∧
  (a > 0 → S = {x : ℝ | x ≥ 2/a ∨ x ≤ -1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | 2/a ≤ x ∧ x ≤ -1}) ∧
  (a = -2 → S = {x : ℝ | x = -1}) ∧
  (a < -2 → S = {x : ℝ | -1 ≤ x ∧ x ≤ 2/a}) :=
by sorry

end inequality_solution_set_l1339_133906
