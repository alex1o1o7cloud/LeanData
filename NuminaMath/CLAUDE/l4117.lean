import Mathlib

namespace NUMINAMATH_CALUDE_tangent_circles_bound_l4117_411735

/-- The maximum number of pairs of tangent circles for n circles -/
def l (n : ℕ) : ℕ :=
  match n with
  | 3 => 3
  | 4 => 5
  | 5 => 7
  | 7 => 12
  | 8 => 14
  | 9 => 16
  | 10 => 19
  | _ => 3 * n - 11

/-- Theorem: For n ≥ 9, the maximum number of pairs of tangent circles is at most 3n - 11 -/
theorem tangent_circles_bound (n : ℕ) (h : n ≥ 9) : l n ≤ 3 * n - 11 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_bound_l4117_411735


namespace NUMINAMATH_CALUDE_probability_green_given_no_red_l4117_411708

/-- The set of all possible colors for memories -/
inductive Color
| Red
| Green
| Blue
| Yellow
| Purple

/-- A memory coloring is a set of at most two distinct colors -/
def MemoryColoring := Finset Color

/-- The set of all valid memory colorings -/
def AllColorings : Finset MemoryColoring :=
  sorry

/-- The set of memory colorings without red -/
def ColoringsWithoutRed : Finset MemoryColoring :=
  sorry

/-- The set of memory colorings that are at least partly green and have no red -/
def GreenColoringsWithoutRed : Finset MemoryColoring :=
  sorry

/-- The probability of a memory being at least partly green given that it has no red -/
theorem probability_green_given_no_red :
  (Finset.card GreenColoringsWithoutRed) / (Finset.card ColoringsWithoutRed) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_green_given_no_red_l4117_411708


namespace NUMINAMATH_CALUDE_triangle_nabla_equality_l4117_411717

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := 3 * a + 2 * b

-- Define the nabla operation
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Theorem to prove
theorem triangle_nabla_equality : triangle 2 (nabla 3 4) = 42 := by
  sorry

end NUMINAMATH_CALUDE_triangle_nabla_equality_l4117_411717


namespace NUMINAMATH_CALUDE_susie_earnings_l4117_411796

def slice_price : ℕ := 3
def whole_pizza_price : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

theorem susie_earnings : 
  slice_price * slices_sold + whole_pizza_price * whole_pizzas_sold = 117 := by
  sorry

end NUMINAMATH_CALUDE_susie_earnings_l4117_411796


namespace NUMINAMATH_CALUDE_village_population_l4117_411772

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.1) * (1 - 0.25) * (1 - 0.12) * (1 - 0.15) = 4136 → 
  P = 8192 := by
sorry

end NUMINAMATH_CALUDE_village_population_l4117_411772


namespace NUMINAMATH_CALUDE_complex_number_location_l4117_411741

theorem complex_number_location : ∃ (z : ℂ), 
  z = (1 : ℂ) / (2 + Complex.I) + Complex.I ^ 2018 ∧ 
  z.re < 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l4117_411741


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_min_sum_squared_distances_achievable_l4117_411711

/-- The minimum sum of squared distances from a point on a circle to two fixed points -/
theorem min_sum_squared_distances (x y : ℝ) :
  (x - 3)^2 + (y - 4)^2 = 4 →
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 ≥ 26 := by
  sorry

/-- The minimum sum of squared distances is achievable -/
theorem min_sum_squared_distances_achievable :
  ∃ x y : ℝ, (x - 3)^2 + (y - 4)^2 = 4 ∧
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_min_sum_squared_distances_achievable_l4117_411711


namespace NUMINAMATH_CALUDE_emmy_and_gerry_apples_l4117_411776

/-- The number of apples Emmy and Gerry can buy together -/
def total_apples (apple_price : ℕ) (emmy_money : ℕ) (gerry_money : ℕ) : ℕ :=
  (emmy_money + gerry_money) / apple_price

/-- Theorem: Emmy and Gerry can buy 150 apples altogether -/
theorem emmy_and_gerry_apples :
  total_apples 2 200 100 = 150 := by
  sorry

#eval total_apples 2 200 100

end NUMINAMATH_CALUDE_emmy_and_gerry_apples_l4117_411776


namespace NUMINAMATH_CALUDE_one_plus_three_squared_l4117_411751

theorem one_plus_three_squared : 1 + 3^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_one_plus_three_squared_l4117_411751


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l4117_411770

theorem cost_increase_percentage (cost selling_price : ℝ) (increase_factor : ℝ) : 
  cost > 0 →
  selling_price = cost * 2.6 →
  (selling_price - cost * (1 + increase_factor)) / selling_price = 0.5692307692307692 →
  increase_factor = 0.12 := by
sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l4117_411770


namespace NUMINAMATH_CALUDE_roger_has_two_more_candies_l4117_411756

/-- The number of candy bags Sandra has -/
def sandra_bags : ℕ := 2

/-- The number of candy pieces in each of Sandra's bags -/
def sandra_pieces_per_bag : ℕ := 6

/-- The number of candy bags Roger has -/
def roger_bags : ℕ := 2

/-- The number of candy pieces in Roger's first bag -/
def roger_bag1_pieces : ℕ := 11

/-- The number of candy pieces in Roger's second bag -/
def roger_bag2_pieces : ℕ := 3

/-- Theorem stating that Roger has 2 more pieces of candy than Sandra -/
theorem roger_has_two_more_candies : 
  (roger_bag1_pieces + roger_bag2_pieces) - (sandra_bags * sandra_pieces_per_bag) = 2 := by
  sorry

end NUMINAMATH_CALUDE_roger_has_two_more_candies_l4117_411756


namespace NUMINAMATH_CALUDE_zebra_chase_time_l4117_411719

/-- The time (in hours) it takes for the zebra to catch up with the tiger -/
def catchup_time : ℝ := 6

/-- The speed of the zebra in km/h -/
def zebra_speed : ℝ := 55

/-- The speed of the tiger in km/h -/
def tiger_speed : ℝ := 30

/-- The time (in hours) after which the zebra starts chasing the tiger -/
def chase_start_time : ℝ := 5

theorem zebra_chase_time :
  chase_start_time * tiger_speed + catchup_time * tiger_speed = catchup_time * zebra_speed :=
sorry

end NUMINAMATH_CALUDE_zebra_chase_time_l4117_411719


namespace NUMINAMATH_CALUDE_remainder_three_power_twentyfour_mod_seven_l4117_411742

theorem remainder_three_power_twentyfour_mod_seven :
  3^24 % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_three_power_twentyfour_mod_seven_l4117_411742


namespace NUMINAMATH_CALUDE_total_gold_stars_l4117_411773

def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

theorem total_gold_stars : 
  monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_stars_l4117_411773


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l4117_411767

theorem unique_solution_for_prime_equation :
  ∀ a b : ℕ,
  Prime a →
  b > 0 →
  9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) →
  a = 251 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l4117_411767


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4117_411760

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 9*x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4117_411760


namespace NUMINAMATH_CALUDE_different_result_l4117_411777

theorem different_result : 
  (-2 - (-3) ≠ 2 - 3) ∧ 
  (-2 - (-3) ≠ -3 + 2) ∧ 
  (-2 - (-3) ≠ -3 - (-2)) ∧ 
  (2 - 3 = -3 + 2) ∧ 
  (2 - 3 = -3 - (-2)) := by
  sorry

end NUMINAMATH_CALUDE_different_result_l4117_411777


namespace NUMINAMATH_CALUDE_gas_usage_difference_l4117_411704

theorem gas_usage_difference (felicity_gas adhira_gas : ℕ) : 
  felicity_gas = 23 →
  felicity_gas + adhira_gas = 30 →
  4 * adhira_gas - felicity_gas = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_gas_usage_difference_l4117_411704


namespace NUMINAMATH_CALUDE_solution_characterization_l4117_411779

def is_solution (a b : ℕ+) : Prop :=
  (a.val ^ 2 * b.val ^ 2 + 208 : ℕ) = 4 * (Nat.lcm a.val b.val + Nat.gcd a.val b.val) ^ 2

theorem solution_characterization :
  ∀ a b : ℕ+, is_solution a b ↔ 
    ((a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) ∨ (a = 2 ∧ b = 12) ∨ (a = 12 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l4117_411779


namespace NUMINAMATH_CALUDE_baker_cakes_l4117_411757

theorem baker_cakes (initial_cakes : ℕ) 
  (bought_cakes : ℕ := 103)
  (sold_cakes : ℕ := 86)
  (final_cakes : ℕ := 190)
  (h : initial_cakes + bought_cakes - sold_cakes = final_cakes) :
  initial_cakes = 173 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l4117_411757


namespace NUMINAMATH_CALUDE_line_slope_at_minimum_l4117_411797

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) passing through (-1, 2),
    the slope is 2 when 2/a + 1/b is minimized. -/
theorem line_slope_at_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2*b = 2) →
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → 2/x + 1/y ≥ 2/a + 1/b) →
  b/a = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_at_minimum_l4117_411797


namespace NUMINAMATH_CALUDE_substitution_result_l4117_411793

theorem substitution_result (x y : ℝ) :
  (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) →
  (4 * x + 10 * x - 5 = 7) :=
by sorry

end NUMINAMATH_CALUDE_substitution_result_l4117_411793


namespace NUMINAMATH_CALUDE_denominator_one_root_l4117_411795

theorem denominator_one_root (k : ℝ) : 
  (∃! x : ℝ, -2 * x^2 + 8 * x + k = 0) ↔ k = -8 := by sorry

end NUMINAMATH_CALUDE_denominator_one_root_l4117_411795


namespace NUMINAMATH_CALUDE_cylinder_from_constant_radius_l4117_411783

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSet c

theorem cylinder_from_constant_radius (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSet c) := by
  sorry

#check cylinder_from_constant_radius

end NUMINAMATH_CALUDE_cylinder_from_constant_radius_l4117_411783


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisible_by_six_l4117_411739

theorem consecutive_integer_product_divisible_by_six (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

#check consecutive_integer_product_divisible_by_six

end NUMINAMATH_CALUDE_consecutive_integer_product_divisible_by_six_l4117_411739


namespace NUMINAMATH_CALUDE_smallest_square_sum_of_consecutive_integers_l4117_411727

theorem smallest_square_sum_of_consecutive_integers :
  ∃ n : ℕ, 
    (n > 0) ∧ 
    (10 * (2 * n + 19) = 250) ∧ 
    (∀ m : ℕ, m > 0 → m < n → ¬∃ k : ℕ, 10 * (2 * m + 19) = k * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_sum_of_consecutive_integers_l4117_411727


namespace NUMINAMATH_CALUDE_ticket_price_calculation_l4117_411774

def commission_rate : ℝ := 0.12
def desired_net_amount : ℝ := 22

theorem ticket_price_calculation :
  ∃ (price : ℝ), price * (1 - commission_rate) = desired_net_amount ∧ price = 25 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_calculation_l4117_411774


namespace NUMINAMATH_CALUDE_distance_C_D_l4117_411782

/-- An ellipse with equation 16(x-2)^2 + 4y^2 = 64 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 16 * (p.1 - 2)^2 + 4 * p.2^2 = 64}

/-- The center of the ellipse -/
def center : ℝ × ℝ := (2, 0)

/-- The semi-major axis length -/
def a : ℝ := 4

/-- The semi-minor axis length -/
def b : ℝ := 2

/-- An endpoint of the major axis -/
def C : ℝ × ℝ := (center.1, center.2 + a)

/-- An endpoint of the minor axis -/
def D : ℝ × ℝ := (center.1 + b, center.2)

/-- The theorem stating the distance between C and D -/
theorem distance_C_D : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_C_D_l4117_411782


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l4117_411771

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x a : ℝ) :
  x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - (3 + a)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l4117_411771


namespace NUMINAMATH_CALUDE_sector_area_l4117_411775

theorem sector_area (circumference : Real) (central_angle : Real) :
  circumference = 8 * π / 9 + 4 →
  central_angle = 80 * π / 180 →
  (1 / 2) * (circumference - 2 * (circumference / (2 * π + central_angle))) ^ 2 * central_angle / (2 * π) = 8 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l4117_411775


namespace NUMINAMATH_CALUDE_amusement_park_ticket_price_l4117_411765

/-- Given the following conditions for an amusement park admission:
  * The total cost for admission tickets is $720
  * The price of an adult ticket is $15
  * There are 15 children in the group
  * There are 25 more adults than children
  Prove that the price of a child ticket is $8 -/
theorem amusement_park_ticket_price 
  (total_cost : ℕ) 
  (adult_price : ℕ) 
  (num_children : ℕ) 
  (adult_child_diff : ℕ) 
  (h1 : total_cost = 720)
  (h2 : adult_price = 15)
  (h3 : num_children = 15)
  (h4 : adult_child_diff = 25) :
  ∃ (child_price : ℕ), 
    child_price = 8 ∧ 
    total_cost = adult_price * (num_children + adult_child_diff) + child_price * num_children :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_ticket_price_l4117_411765


namespace NUMINAMATH_CALUDE_profit_scenario_theorem_l4117_411725

/-- Represents the profit scenarios for Bill's product sales -/
structure ProfitScenarios where
  original_purchase_price : ℝ
  original_profit_rate : ℝ
  second_purchase_discount : ℝ
  second_profit_rate : ℝ
  second_additional_profit : ℝ
  third_purchase_discount : ℝ
  third_profit_rate : ℝ
  third_additional_profit : ℝ

/-- Calculates the selling prices for each scenario given the profit conditions -/
def calculate_selling_prices (s : ProfitScenarios) : ℝ × ℝ × ℝ :=
  let original_selling_price := s.original_purchase_price * (1 + s.original_profit_rate)
  let second_selling_price := original_selling_price + s.second_additional_profit
  let third_selling_price := original_selling_price + s.third_additional_profit
  (original_selling_price, second_selling_price, third_selling_price)

/-- Theorem stating that given the profit conditions, the selling prices are as calculated -/
theorem profit_scenario_theorem (s : ProfitScenarios) 
  (h1 : s.original_profit_rate = 0.1)
  (h2 : s.second_purchase_discount = 0.1)
  (h3 : s.second_profit_rate = 0.3)
  (h4 : s.second_additional_profit = 35)
  (h5 : s.third_purchase_discount = 0.15)
  (h6 : s.third_profit_rate = 0.5)
  (h7 : s.third_additional_profit = 70) :
  calculate_selling_prices s = (550, 585, 620) := by
  sorry

end NUMINAMATH_CALUDE_profit_scenario_theorem_l4117_411725


namespace NUMINAMATH_CALUDE_count_solutions_eq_288_l4117_411714

/-- The count of positive integers N less than 500 for which x^⌊x⌋ = N has a solution -/
def count_solutions : ℕ :=
  let floor_0_count := 1  -- N = 1 for ⌊x⌋ = 0
  let floor_1_count := 0  -- Already counted in floor_0_count
  let floor_2_count := 5  -- N = 4, 5, ..., 8
  let floor_3_count := 38 -- N = 27, 28, ..., 64
  let floor_4_count := 244 -- N = 256, 257, ..., 499
  floor_0_count + floor_1_count + floor_2_count + floor_3_count + floor_4_count

/-- The main theorem stating that the count of solutions is 288 -/
theorem count_solutions_eq_288 : count_solutions = 288 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_eq_288_l4117_411714


namespace NUMINAMATH_CALUDE_room_area_l4117_411785

/-- The area of a rectangular room with length 5 feet and width 2 feet is 10 square feet. -/
theorem room_area : 
  let length : ℝ := 5
  let width : ℝ := 2
  length * width = 10 := by sorry

end NUMINAMATH_CALUDE_room_area_l4117_411785


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l4117_411743

/-- Represents the population sizes for each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the sample sizes for each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Checks if the sample is proportional to the population -/
def isProportionalSample (pop : Population) (sam : Sample) (totalSample : ℕ) : Prop :=
  sam.elderly * (pop.elderly + pop.middleAged + pop.young) = pop.elderly * totalSample ∧
  sam.middleAged * (pop.elderly + pop.middleAged + pop.young) = pop.middleAged * totalSample ∧
  sam.young * (pop.elderly + pop.middleAged + pop.young) = pop.young * totalSample

theorem stratified_sampling_theorem (pop : Population) (sam : Sample) :
  pop.elderly = 27 →
  pop.middleAged = 54 →
  pop.young = 81 →
  sam.elderly + sam.middleAged + sam.young = 36 →
  isProportionalSample pop sam 36 →
  sam.elderly = 6 ∧ sam.middleAged = 12 ∧ sam.young = 18 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l4117_411743


namespace NUMINAMATH_CALUDE_fourth_quadrant_m_range_l4117_411791

theorem fourth_quadrant_m_range (m : ℝ) :
  let z : ℂ := (1 + m * Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) → -1 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_quadrant_m_range_l4117_411791


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l4117_411705

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c ≤ 3) : 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l4117_411705


namespace NUMINAMATH_CALUDE_puppy_kibble_percentage_proof_l4117_411752

/-- The percentage of vets recommending Puppy Kibble -/
def puppy_kibble_percentage : ℝ := 20

/-- The percentage of vets recommending Yummy Dog Kibble -/
def yummy_kibble_percentage : ℝ := 30

/-- The total number of vets in the state -/
def total_vets : ℕ := 1000

/-- The difference in number of vets recommending Yummy Dog Kibble vs Puppy Kibble -/
def vet_difference : ℕ := 100

theorem puppy_kibble_percentage_proof :
  puppy_kibble_percentage = 20 ∧
  yummy_kibble_percentage = 30 ∧
  total_vets = 1000 ∧
  vet_difference = 100 →
  puppy_kibble_percentage * (total_vets : ℝ) / 100 + vet_difference = 
  yummy_kibble_percentage * (total_vets : ℝ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_puppy_kibble_percentage_proof_l4117_411752


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l4117_411712

theorem min_value_cubic_function (x : ℝ) (h : x > 0) :
  x^3 + 9*x + 81/x^4 ≥ 21 ∧ ∃ y > 0, y^3 + 9*y + 81/y^4 = 21 :=
sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l4117_411712


namespace NUMINAMATH_CALUDE_eleventh_term_is_110_div_7_l4117_411749

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of the first six terms is 30
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 30
  -- Seventh term is 10
  seventh_term : a + 6*d = 10

/-- The eleventh term of the specific arithmetic sequence is 110/7 -/
theorem eleventh_term_is_110_div_7 (seq : ArithmeticSequence) :
  seq.a + 10*seq.d = 110/7 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_110_div_7_l4117_411749


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l4117_411713

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 450 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 450 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 24 ∣ n^2 ∧ 450 ∣ n^3 ∧ ∀ m : ℕ, (m > 0 ∧ 24 ∣ m^2 ∧ 450 ∣ m^3) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l4117_411713


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l4117_411732

def bill_with_discount (bill : Float) (discount_rate : Float) : Float :=
  bill * (1 - discount_rate / 100)

def total_bill (bob_bill kate_bill john_bill sarah_bill : Float)
               (bob_discount kate_discount john_discount sarah_discount : Float) : Float :=
  bill_with_discount bob_bill bob_discount +
  bill_with_discount kate_bill kate_discount +
  bill_with_discount john_bill john_discount +
  bill_with_discount sarah_bill sarah_discount

theorem restaurant_bill_theorem :
  total_bill 35.50 29.75 43.20 27.35 5.75 2.35 3.95 9.45 = 128.76945 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l4117_411732


namespace NUMINAMATH_CALUDE_ellipse_foci_l4117_411761

/-- Represents an ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The foci of an ellipse -/
structure Foci where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- Theorem: The foci of the ellipse x²/1 + y²/10 = 1 are (0, -3) and (0, 3) -/
theorem ellipse_foci (e : Ellipse) (h₁ : e.a = 1) (h₂ : e.b = 10) :
  ∃ f : Foci, f.x₁ = 0 ∧ f.y₁ = -3 ∧ f.x₂ = 0 ∧ f.y₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_l4117_411761


namespace NUMINAMATH_CALUDE_point_on_line_l4117_411721

/-- Prove that for a point P(2, m) lying on the line 3x + y = 2, the value of m is -4. -/
theorem point_on_line (m : ℝ) : (3 * 2 + m = 2) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l4117_411721


namespace NUMINAMATH_CALUDE_juan_saw_three_bicycles_l4117_411754

/-- The number of bicycles Juan saw -/
def num_bicycles : ℕ := 3

/-- The number of cars Juan saw -/
def num_cars : ℕ := 15

/-- The number of pickup trucks Juan saw -/
def num_pickup_trucks : ℕ := 8

/-- The number of tricycles Juan saw -/
def num_tricycles : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 101

/-- The number of tires on a car -/
def tires_per_car : ℕ := 4

/-- The number of tires on a pickup truck -/
def tires_per_pickup : ℕ := 4

/-- The number of tires on a tricycle -/
def tires_per_tricycle : ℕ := 3

/-- The number of tires on a bicycle -/
def tires_per_bicycle : ℕ := 2

theorem juan_saw_three_bicycles :
  num_bicycles * tires_per_bicycle + 
  num_cars * tires_per_car + 
  num_pickup_trucks * tires_per_pickup + 
  num_tricycles * tires_per_tricycle = total_tires :=
by sorry

end NUMINAMATH_CALUDE_juan_saw_three_bicycles_l4117_411754


namespace NUMINAMATH_CALUDE_triangle_count_is_twenty_l4117_411709

/-- Represents a point on the 3x3 grid -/
structure GridPoint where
  x : Fin 3
  y : Fin 3

/-- Represents a triangle on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The set of all possible triangles on the 3x3 grid -/
def allGridTriangles : Set GridTriangle := sorry

/-- Counts the number of triangles in the 3x3 grid -/
def countTriangles : ℕ := sorry

/-- Theorem stating that the number of triangles in the 3x3 grid is 20 -/
theorem triangle_count_is_twenty : countTriangles = 20 := by sorry

end NUMINAMATH_CALUDE_triangle_count_is_twenty_l4117_411709


namespace NUMINAMATH_CALUDE_calculator_probability_l4117_411790

/-- Represents a 7-segment calculator display --/
def SegmentDisplay := Fin 7 → Bool

/-- The probability of a segment being illuminated --/
def segmentProbability : ℚ := 1/2

/-- The total number of possible displays --/
def totalDisplays : ℕ := 2^7

/-- The number of valid digit displays (0-9) --/
def validDigitDisplays : ℕ := 10

/-- The probability of displaying a valid digit --/
def validDigitProbability : ℚ := validDigitDisplays / totalDisplays

theorem calculator_probability (a b : ℕ) (h : validDigitProbability = a / b) :
  9 * a + 2 * b = 173 := by
  sorry

end NUMINAMATH_CALUDE_calculator_probability_l4117_411790


namespace NUMINAMATH_CALUDE_trig_identity_l4117_411755

theorem trig_identity (x : ℝ) : 
  (Real.sin x ^ 6 + Real.cos x ^ 6 - 1) ^ 3 + 27 * Real.sin x ^ 6 * Real.cos x ^ 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4117_411755


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l4117_411780

/-- Given three terms of a geometric progression in the form (15 + x), (45 + x), and (135 + x),
    prove that x = 0 is the unique solution. -/
theorem geometric_progression_solution (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (45 + x) = (15 + x) * r ∧ (135 + x) = (45 + x) * r) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l4117_411780


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l4117_411769

theorem expression_is_perfect_square (x y z : ℤ) (A : ℤ) :
  A = x * y + y * z + z * x →
  A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1) →
  ∃ k : ℤ, (-1) * A = k^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l4117_411769


namespace NUMINAMATH_CALUDE_parallel_planes_from_common_perpendicular_l4117_411759

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_common_perpendicular 
  (m : Line) (α β : Plane) (h_diff : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_common_perpendicular_l4117_411759


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l4117_411799

theorem P_greater_than_Q : ∀ x : ℝ, (x^2 + 2) > (2*x) := by
  sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l4117_411799


namespace NUMINAMATH_CALUDE_worker_r_earnings_l4117_411758

/-- Given the daily earnings of three workers p, q, and r, prove that r earns 50 per day. -/
theorem worker_r_earnings
  (p q r : ℚ)  -- Daily earnings of workers p, q, and r
  (h1 : 9 * (p + q + r) = 1800)  -- p, q, and r together earn 1800 in 9 days
  (h2 : 5 * (p + r) = 600)  -- p and r can earn 600 in 5 days
  (h3 : 7 * (q + r) = 910)  -- q and r can earn 910 in 7 days
  : r = 50 := by
  sorry


end NUMINAMATH_CALUDE_worker_r_earnings_l4117_411758


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4117_411700

theorem inequality_solution_set (a b : ℝ) (h : a ≠ b) :
  {x : ℝ | a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4117_411700


namespace NUMINAMATH_CALUDE_summer_degrees_l4117_411728

/-- Given two people where one has five more degrees than the other, 
    and their combined degrees total 295, prove that the person with 
    more degrees has 150 degrees. -/
theorem summer_degrees (s j : ℕ) 
    (h1 : s = j + 5)
    (h2 : s + j = 295) : 
  s = 150 := by
  sorry

end NUMINAMATH_CALUDE_summer_degrees_l4117_411728


namespace NUMINAMATH_CALUDE_event_C_is_certain_l4117_411778

-- Define an enumeration for the events
inductive Event
  | A -- It will rain after thunder
  | B -- Tomorrow will be sunny
  | C -- 1 hour equals 60 minutes
  | D -- There will be a rainbow after the rain

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.C => True
  | _ => False

-- Theorem stating that Event C is certain
theorem event_C_is_certain : isCertain Event.C := by
  sorry

end NUMINAMATH_CALUDE_event_C_is_certain_l4117_411778


namespace NUMINAMATH_CALUDE_pythagorean_proof_l4117_411720

theorem pythagorean_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  b^2 = 13 * (b - a)^2 → a / b = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_proof_l4117_411720


namespace NUMINAMATH_CALUDE_triangle_distance_set_l4117_411733

theorem triangle_distance_set (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (hk : k^2 > 2*a^2/3 + 2*b^2/3) :
  let S := {P : ℝ × ℝ | P.1^2 + P.2^2 + (P.1 - a)^2 + P.2^2 + P.1^2 + (P.2 - b)^2 < k^2}
  let C := {P : ℝ × ℝ | (P.1 - a/3)^2 + (P.2 - b/3)^2 < (k^2 - 2*a^2/3 - 2*b^2/3) / 3}
  S = C := by sorry

end NUMINAMATH_CALUDE_triangle_distance_set_l4117_411733


namespace NUMINAMATH_CALUDE_boys_at_reunion_l4117_411750

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There are 7 boys at the reunion -/
theorem boys_at_reunion : ∃ n : ℕ, n > 0 ∧ handshakes n = 21 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_at_reunion_l4117_411750


namespace NUMINAMATH_CALUDE_group_frequency_number_l4117_411745

-- Define the sample capacity
def sample_capacity : ℕ := 100

-- Define the frequency of the group
def group_frequency : ℚ := 3/10

-- Define the frequency number calculation
def frequency_number (capacity : ℕ) (frequency : ℚ) : ℚ := capacity * frequency

-- Theorem statement
theorem group_frequency_number :
  frequency_number sample_capacity group_frequency = 30 := by sorry

end NUMINAMATH_CALUDE_group_frequency_number_l4117_411745


namespace NUMINAMATH_CALUDE_units_digit_characteristic_l4117_411703

/-- Given a positive even integer p, if the units digit of p^3 minus the units digit of p^2 is 0
    and the units digit of p + 1 is 7, then the units digit of p is 6. -/
theorem units_digit_characteristic (p : ℕ) : 
  p > 0 → 
  Even p → 
  (p^3 % 10 - p^2 % 10) % 10 = 0 → 
  (p + 1) % 10 = 7 → 
  p % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_characteristic_l4117_411703


namespace NUMINAMATH_CALUDE_hostel_problem_solution_l4117_411730

/-- Represents the hostel problem with given initial conditions -/
structure HostelProblem where
  initial_students : ℕ
  budget_decrease : ℕ
  expenditure_increase : ℕ
  new_total_expenditure : ℕ

/-- Calculates the number of new students given a HostelProblem -/
def new_students (problem : HostelProblem) : ℕ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that for the given problem, 35 new students joined -/
theorem hostel_problem_solution :
  let problem : HostelProblem := {
    initial_students := 100,
    budget_decrease := 10,
    expenditure_increase := 400,
    new_total_expenditure := 5400
  }
  new_students problem = 35 := by
  sorry

end NUMINAMATH_CALUDE_hostel_problem_solution_l4117_411730


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l4117_411736

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l4117_411736


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4117_411731

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 3) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4117_411731


namespace NUMINAMATH_CALUDE_unknown_number_solution_l4117_411738

theorem unknown_number_solution (x : ℝ) : 
  4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ↔ x = 77.31 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l4117_411738


namespace NUMINAMATH_CALUDE_f_derivative_l4117_411729

noncomputable def f (x : ℝ) : ℝ := Real.log (5 * x + Real.sqrt (25 * x^2 + 1)) - Real.sqrt (25 * x^2 + 1) * Real.arctan (5 * x)

theorem f_derivative (x : ℝ) : 
  deriv f x = -(25 * x * Real.arctan (5 * x)) / Real.sqrt (25 * x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l4117_411729


namespace NUMINAMATH_CALUDE_larger_number_proof_l4117_411718

theorem larger_number_proof (a b : ℕ) : 
  (Nat.gcd a b = 25) →
  (Nat.lcm a b = 4550) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  (max a b = 350) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4117_411718


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l4117_411722

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (5, -2)
  are_parallel a b → m = -10 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l4117_411722


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4117_411710

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4117_411710


namespace NUMINAMATH_CALUDE_ceiling_minus_half_integer_l4117_411702

theorem ceiling_minus_half_integer (n : ℤ) : 
  let x : ℝ := n + 1/2
  ⌈x⌉ - x = 1/2 := by sorry

end NUMINAMATH_CALUDE_ceiling_minus_half_integer_l4117_411702


namespace NUMINAMATH_CALUDE_horatio_sonnets_l4117_411747

/-- Proves that Horatio wrote 12 sonnets in total -/
theorem horatio_sonnets (lines_per_sonnet : ℕ) (read_sonnets : ℕ) (unread_lines : ℕ) : 
  lines_per_sonnet = 14 → read_sonnets = 7 → unread_lines = 70 →
  read_sonnets + (unread_lines / lines_per_sonnet) = 12 := by
  sorry

end NUMINAMATH_CALUDE_horatio_sonnets_l4117_411747


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4117_411781

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (5 + n) = 8 → n = 59 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4117_411781


namespace NUMINAMATH_CALUDE_total_weight_of_diamonds_and_jades_l4117_411723

/-- Given that 5 diamonds weigh 100 g and a jade is 10 g heavier than a diamond,
    prove that the total weight of 4 diamonds and 2 jades is 140 g. -/
theorem total_weight_of_diamonds_and_jades :
  let diamond_weight : ℚ := 100 / 5
  let jade_weight : ℚ := diamond_weight + 10
  4 * diamond_weight + 2 * jade_weight = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_diamonds_and_jades_l4117_411723


namespace NUMINAMATH_CALUDE_exists_valid_triangle_l4117_411789

-- Define the necessary structures
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the given elements
variable (p q : Line)
variable (C : Point)
variable (c : ℝ)

-- Define a right triangle
structure RightTriangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (hypotenuse_length : ℝ)

-- Define the conditions for the desired triangle
def is_valid_triangle (t : RightTriangle) : Prop :=
  -- Right angle at C
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0 ∧
  -- Vertex A on line p
  t.A.y = p.slope * t.A.x + p.intercept ∧
  -- Hypotenuse parallel to line q
  (t.A.y - t.C.y) / (t.A.x - t.C.x) = q.slope ∧
  -- Hypotenuse length is c
  t.hypotenuse_length = c ∧
  -- C is the given point
  t.C = C

-- Theorem statement
theorem exists_valid_triangle :
  ∃ (t : RightTriangle), is_valid_triangle p q C c t :=
sorry

end NUMINAMATH_CALUDE_exists_valid_triangle_l4117_411789


namespace NUMINAMATH_CALUDE_sin_150_cos_30_l4117_411746

theorem sin_150_cos_30 : Real.sin (150 * π / 180) * Real.cos (30 * π / 180) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_cos_30_l4117_411746


namespace NUMINAMATH_CALUDE_new_students_count_l4117_411726

theorem new_students_count : ∃! n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 := by
  sorry

end NUMINAMATH_CALUDE_new_students_count_l4117_411726


namespace NUMINAMATH_CALUDE_susie_large_rooms_l4117_411701

/-- Represents the number of rooms of each size in Susie's house. -/
structure RoomCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the time needed to vacuum each type of room. -/
structure VacuumTimes where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total vacuuming time for all rooms. -/
def totalVacuumTime (counts : RoomCounts) (times : VacuumTimes) : Nat :=
  counts.small * times.small + counts.medium * times.medium + counts.large * times.large

/-- The theorem stating that given the conditions, Susie has 2 large rooms. -/
theorem susie_large_rooms : 
  ∀ (counts : RoomCounts) (times : VacuumTimes),
    counts.small = 4 →
    counts.medium = 3 →
    times.small = 15 →
    times.medium = 25 →
    times.large = 35 →
    totalVacuumTime counts times = 225 →
    counts.large = 2 := by
  sorry

end NUMINAMATH_CALUDE_susie_large_rooms_l4117_411701


namespace NUMINAMATH_CALUDE_circle_properties_l4117_411764

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define point P
def P : ℝ × ℝ := (4, 5)

-- Theorem statement
theorem circle_properties :
  -- P is on circle C
  C P.1 P.2 ∧
  -- Distance PQ
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 10 ∧
  -- Slope of PQ
  (P.2 - Q.2) / (P.1 - Q.1) = 1/3 ∧
  -- Maximum and minimum distances from Q to any point on C
  (∀ M : ℝ × ℝ, C M.1 M.2 → 
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ 6 * Real.sqrt 2 ∧
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≥ 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l4117_411764


namespace NUMINAMATH_CALUDE_lucas_overall_accuracy_l4117_411788

theorem lucas_overall_accuracy 
  (emily_individual_accuracy : Real) 
  (emily_overall_accuracy : Real)
  (lucas_individual_accuracy : Real)
  (h1 : emily_individual_accuracy = 0.7)
  (h2 : emily_overall_accuracy = 0.82)
  (h3 : lucas_individual_accuracy = 0.85) :
  lucas_individual_accuracy * 0.5 + (emily_overall_accuracy - emily_individual_accuracy * 0.5) = 0.895 := by
  sorry

end NUMINAMATH_CALUDE_lucas_overall_accuracy_l4117_411788


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l4117_411792

def solutions (x : ℂ) : Prop := x^8 = -256

def positive_real_part (z : ℂ) : Prop := z.re > 0

theorem product_of_positive_real_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, solutions z ∧ positive_real_part z) ∧ 
    (∀ z, solutions z ∧ positive_real_part z → z ∈ S) ∧
    S.prod id = 8 :=
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l4117_411792


namespace NUMINAMATH_CALUDE_all_inhabitants_can_reach_palace_l4117_411798

-- Define the kingdom as a square
def kingdom_side_length : ℝ := 2

-- Define the speed of inhabitants
def inhabitant_speed : ℝ := 3

-- Define the available time
def available_time : ℝ := 7

-- Theorem statement
theorem all_inhabitants_can_reach_palace :
  ∀ (x y : ℝ), 
    0 ≤ x ∧ x ≤ kingdom_side_length ∧
    0 ≤ y ∧ y ≤ kingdom_side_length →
    ∃ (t : ℝ), 
      0 ≤ t ∧ t ≤ available_time ∧
      t * inhabitant_speed ≥ Real.sqrt ((x - kingdom_side_length/2)^2 + (y - kingdom_side_length/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_all_inhabitants_can_reach_palace_l4117_411798


namespace NUMINAMATH_CALUDE_all_routes_have_eight_stations_l4117_411768

/-- Represents a bus route in the city -/
structure BusRoute where
  stations : Set Nat
  station_count : Nat

/-- Represents the city's bus network -/
structure BusNetwork where
  routes : Finset BusRoute
  route_count : Nat

/-- Conditions for the bus network -/
def valid_network (n : BusNetwork) : Prop :=
  -- There are 57 bus routes
  n.route_count = 57 ∧
  -- Any two routes share exactly one station
  ∀ r1 r2 : BusRoute, r1 ∈ n.routes ∧ r2 ∈ n.routes ∧ r1 ≠ r2 →
    ∃! s : Nat, s ∈ r1.stations ∧ s ∈ r2.stations ∧
  -- Each route has at least 3 stations
  ∀ r : BusRoute, r ∈ n.routes → r.station_count ≥ 3 ∧
  -- From any station, it's possible to reach any other station without changing buses
  ∀ s1 s2 : Nat, ∃ r : BusRoute, r ∈ n.routes ∧ s1 ∈ r.stations ∧ s2 ∈ r.stations

/-- The main theorem to prove -/
theorem all_routes_have_eight_stations (n : BusNetwork) (h : valid_network n) :
  ∀ r : BusRoute, r ∈ n.routes → r.station_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_all_routes_have_eight_stations_l4117_411768


namespace NUMINAMATH_CALUDE_exactly_two_successes_in_four_trials_l4117_411737

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability mass function -/
def binomial_pmf (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- The theorem to be proved -/
theorem exactly_two_successes_in_four_trials : 
  binomial_pmf n k p = 0.3456 := by sorry

end NUMINAMATH_CALUDE_exactly_two_successes_in_four_trials_l4117_411737


namespace NUMINAMATH_CALUDE_divisible_by_seven_l4117_411748

def repeated_digit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

theorem divisible_by_seven : ∃ k : Nat,
  (repeated_digit 8 50 * 10 + 5) * 10^50 + repeated_digit 9 50 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l4117_411748


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l4117_411762

-- Define the vectors
def a (m : ℝ) : ℝ × ℝ := (m, 2)
def b (n : ℝ) : ℝ × ℝ := (-1, n)

-- Define the theorem
theorem vector_magnitude_problem (m n : ℝ) : 
  n > 0 ∧ 
  (a m) • (b n) = 0 ∧ 
  m^2 + n^2 = 5 → 
  ‖2 • (a m) + (b n)‖ = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l4117_411762


namespace NUMINAMATH_CALUDE_point_coordinates_l4117_411707

/-- A point in the second quadrant with a specific distance from the x-axis -/
def SecondQuadrantPoint (m : ℝ) : Prop :=
  m - 3 < 0 ∧ m + 2 > 0 ∧ |m + 2| = 4

/-- The theorem stating that a point with the given properties has coordinates (-1, 4) -/
theorem point_coordinates (m : ℝ) (h : SecondQuadrantPoint m) : 
  (m - 3 = -1) ∧ (m + 2 = 4) :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_l4117_411707


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l4117_411787

/-- A hyperbola with vertex and center at (1, 0) and eccentricity 2 -/
structure Hyperbola where
  vertex : ℝ × ℝ
  center : ℝ × ℝ
  eccentricity : ℝ
  vertex_eq_center : vertex = center
  vertex_x : vertex.1 = 1
  vertex_y : vertex.2 = 0
  eccentricity_val : eccentricity = 2

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 - y^2/3 = 1

/-- Theorem stating that the given hyperbola has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 - y^2/3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l4117_411787


namespace NUMINAMATH_CALUDE_f_derivative_and_tangent_lines_l4117_411784

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x^2 + 1) + 1

theorem f_derivative_and_tangent_lines :
  (∃ f' : ℝ → ℝ, ∀ x, deriv f x = f' x ∧ f' x = 3 * x^2 - 2 * x + 1) ∧
  (∃ t₁ t₂ : ℝ → ℝ,
    (∀ x, t₁ x = x) ∧
    (∀ x, t₂ x = 2 * x - 1) ∧
    (t₁ 1 = f 1 ∧ t₂ 1 = f 1) ∧
    (∃ x₀, deriv f x₀ = deriv t₁ x₀ ∧ f x₀ = t₁ x₀) ∧
    (∃ x₁, deriv f x₁ = deriv t₂ x₁ ∧ f x₁ = t₂ x₁)) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_and_tangent_lines_l4117_411784


namespace NUMINAMATH_CALUDE_pizza_sector_chord_length_squared_l4117_411706

theorem pizza_sector_chord_length_squared (r : ℝ) (h : r = 8) :
  let chord_length_squared := 2 * r^2
  chord_length_squared = 128 := by sorry

end NUMINAMATH_CALUDE_pizza_sector_chord_length_squared_l4117_411706


namespace NUMINAMATH_CALUDE_angle_C_is_45_degrees_l4117_411744

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ -- Area of the triangle

-- Define the vectors p and q
def p (t : Triangle) : ℝ × ℝ := (4, t.a^2 + t.b^2 - t.c^2)
def q (t : Triangle) : ℝ × ℝ := (1, t.S)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem angle_C_is_45_degrees (t : Triangle) 
  (h : parallel (p t) (q t)) : t.C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_45_degrees_l4117_411744


namespace NUMINAMATH_CALUDE_soccer_balls_per_basket_l4117_411763

theorem soccer_balls_per_basket
  (num_baskets : ℕ)
  (tennis_balls_per_basket : ℕ)
  (total_balls_removed : ℕ)
  (balls_remaining : ℕ)
  (h1 : num_baskets = 5)
  (h2 : tennis_balls_per_basket = 15)
  (h3 : total_balls_removed = 44)
  (h4 : balls_remaining = 56) :
  (num_baskets * tennis_balls_per_basket + num_baskets * 5 = balls_remaining + total_balls_removed) := by
sorry

end NUMINAMATH_CALUDE_soccer_balls_per_basket_l4117_411763


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4117_411715

def M : Set Int := {1, 2, 3, 4}
def N : Set Int := {-2, 2}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4117_411715


namespace NUMINAMATH_CALUDE_max_teams_double_round_robin_l4117_411724

/-- A schedule for a double round robin tournament. -/
def Schedule (n : ℕ) := Fin n → Fin 4 → List (Fin n)

/-- Predicate to check if a schedule is valid according to the tournament rules. -/
def is_valid_schedule (n : ℕ) (s : Schedule n) : Prop :=
  -- Each team plays with every other team twice
  (∀ i j : Fin n, i ≠ j → (∃ w : Fin 4, i ∈ s j w) ∧ (∃ w : Fin 4, j ∈ s i w)) ∧
  -- If a team has a home game in a week, it cannot have any away games that week
  (∀ i : Fin n, ∀ w : Fin 4, (s i w).length > 0 → ∀ j : Fin n, i ∉ s j w)

/-- The maximum number of teams that can complete the tournament in 4 weeks is 6. -/
theorem max_teams_double_round_robin : 
  (∃ s : Schedule 6, is_valid_schedule 6 s) ∧ 
  (∀ s : Schedule 7, ¬ is_valid_schedule 7 s) :=
sorry

end NUMINAMATH_CALUDE_max_teams_double_round_robin_l4117_411724


namespace NUMINAMATH_CALUDE_students_in_all_classes_l4117_411734

/-- Proves that 8 students are registered for all 3 classes given the problem conditions -/
theorem students_in_all_classes (total_students : ℕ) (history_students : ℕ) (math_students : ℕ) 
  (english_students : ℕ) (two_classes_students : ℕ) : ℕ :=
by
  sorry

#check students_in_all_classes 68 19 14 26 7

end NUMINAMATH_CALUDE_students_in_all_classes_l4117_411734


namespace NUMINAMATH_CALUDE_triangle_properties_l4117_411794

-- Define the points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the properties of triangle ABC
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

def is_perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define the possible coordinates of point C
def C1 : ℝ × ℝ := (3, -1)
def C2 : ℝ × ℝ := (-3, 7)

-- Define the equations of the median lines
def median_eq1 (x y : ℝ) : Prop := 7 * x - y + 3 = 0
def median_eq2 (x y : ℝ) : Prop := x + 7 * y - 21 = 0

-- Theorem statement
theorem triangle_properties :
  (is_isosceles A B C1 ∧ is_perpendicular A B C1 ∧
   median_eq1 ((A.1 + C1.1) / 2) ((A.2 + C1.2) / 2)) ∨
  (is_isosceles A B C2 ∧ is_perpendicular A B C2 ∧
   median_eq2 ((A.1 + C2.1) / 2) ((A.2 + C2.2) / 2)) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4117_411794


namespace NUMINAMATH_CALUDE_new_machine_rate_proof_l4117_411753

/-- The rate of the old machine in bolts per hour -/
def old_machine_rate : ℝ := 100

/-- The time both machines work together in minutes -/
def work_time : ℝ := 84

/-- The total number of bolts produced by both machines -/
def total_bolts : ℝ := 350

/-- The rate of the new machine in bolts per hour -/
def new_machine_rate : ℝ := 150

theorem new_machine_rate_proof :
  (old_machine_rate * work_time / 60 + new_machine_rate * work_time / 60) = total_bolts :=
by sorry

end NUMINAMATH_CALUDE_new_machine_rate_proof_l4117_411753


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l4117_411716

theorem meal_cost_calculation (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  ∃ (meal_cost : ℚ), meal_cost * (adults + children) = total_bill ∧ meal_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l4117_411716


namespace NUMINAMATH_CALUDE_tens_place_of_first_ten_digit_number_l4117_411740

/-- Represents the sequence of grouped numbers -/
def groupedSequence : List (List Nat) := sorry

/-- The number of digits in the nth group -/
def groupDigits (n : Nat) : Nat := n

/-- The sum of digits in the first n groups -/
def sumDigitsUpTo (n : Nat) : Nat := sorry

/-- The first ten-digit number in the sequence -/
def firstTenDigitNumber : Nat := sorry

/-- Theorem: The tens place digit of the first ten-digit number is 2 -/
theorem tens_place_of_first_ten_digit_number :
  (firstTenDigitNumber / 1000000000) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_tens_place_of_first_ten_digit_number_l4117_411740


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_div_2_l4117_411766

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sin x

theorem derivative_f_at_pi_div_2 :
  deriv f (π / 2) = π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_div_2_l4117_411766


namespace NUMINAMATH_CALUDE_students_at_start_l4117_411786

theorem students_at_start (initial_students final_students left_students new_students : ℕ) :
  final_students = 43 →
  left_students = 3 →
  new_students = 42 →
  initial_students + new_students - left_students = final_students →
  initial_students = 4 := by
sorry

end NUMINAMATH_CALUDE_students_at_start_l4117_411786
