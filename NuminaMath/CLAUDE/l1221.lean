import Mathlib

namespace NUMINAMATH_CALUDE_monotonically_increasing_intervals_minimum_value_in_interval_f_properties_l1221_122185

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  (∀ x < 0, f' x > 0) ∧ (∀ x > 1, f' x > 0) :=
sorry

-- Theorem for minimum value in the interval [-1, 2]
theorem minimum_value_in_interval :
  ∀ x ∈ Set.Icc (-1) 2, f x ≥ f (-1) :=
sorry

-- Main theorem combining both parts
theorem f_properties :
  (∀ x < 0, f' x > 0) ∧ (∀ x > 1, f' x > 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ f (-1)) :=
sorry

end NUMINAMATH_CALUDE_monotonically_increasing_intervals_minimum_value_in_interval_f_properties_l1221_122185


namespace NUMINAMATH_CALUDE_abs_fraction_sum_not_one_l1221_122101

theorem abs_fraction_sum_not_one (a b : ℝ) (h : a * b ≠ 0) :
  |a| / a + |b| / b ≠ 1 := by sorry

end NUMINAMATH_CALUDE_abs_fraction_sum_not_one_l1221_122101


namespace NUMINAMATH_CALUDE_system_solution_l1221_122107

theorem system_solution : ∃ (x y z : ℝ), 
  (x + 2*y + 3*z = 3) ∧ 
  (3*x + y + 2*z = 7) ∧ 
  (2*x + 3*y + z = 2) ∧
  (x = 2) ∧ (y = -1) ∧ (z = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1221_122107


namespace NUMINAMATH_CALUDE_function_properties_l1221_122173

def f (a : ℝ) (x : ℝ) : ℝ := a * x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

theorem function_properties (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x, g a (-x) = g a x) ∧
  (∀ x, f a x + g a x = x^2 + a*x + a) ∧
  ((∀ x ∈ Set.Icc 1 2, f a x ≥ 1) ∨ (∃ x ∈ Set.Icc (-1) 2, g a x ≤ -1)) →
  (a ≥ 1 ∨ a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1221_122173


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1221_122197

/-- The area of a triangle given its three altitudes --/
def triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) : ℝ := sorry

/-- A triangle with altitudes 36.4, 39, and 42 has an area of 3549/4 --/
theorem triangle_area_theorem :
  triangle_area_from_altitudes 36.4 39 42 = 3549 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1221_122197


namespace NUMINAMATH_CALUDE_desktop_computers_sold_l1221_122151

theorem desktop_computers_sold (total : ℕ) (laptops : ℕ) (netbooks : ℕ) (desktops : ℕ)
  (h1 : total = 72)
  (h2 : laptops = total / 2)
  (h3 : netbooks = total / 3)
  (h4 : desktops = total - laptops - netbooks) :
  desktops = 12 := by
  sorry

end NUMINAMATH_CALUDE_desktop_computers_sold_l1221_122151


namespace NUMINAMATH_CALUDE_basement_bulbs_l1221_122195

def light_bulbs_problem (bedroom bathroom kitchen basement garage : ℕ) : Prop :=
  bedroom = 2 ∧
  bathroom = 1 ∧
  kitchen = 1 ∧
  garage = basement / 2 ∧
  bedroom + bathroom + kitchen + basement + garage = 12

theorem basement_bulbs :
  ∃ (bedroom bathroom kitchen basement garage : ℕ),
    light_bulbs_problem bedroom bathroom kitchen basement garage ∧
    basement = 5 := by
  sorry

end NUMINAMATH_CALUDE_basement_bulbs_l1221_122195


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l1221_122190

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (r s : ℤ), x^2 + b*x + 2016 = (x + r) * (x + s) → b ≥ 90) ∧
  (∃ (r s : ℤ), x^2 + 90*x + 2016 = (x + r) * (x + s)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l1221_122190


namespace NUMINAMATH_CALUDE_square_roots_theorem_l1221_122135

theorem square_roots_theorem (a : ℝ) :
  (∃ x : ℝ, x^2 = (a + 3)^2 ∧ x^2 = (2*a - 9)^2) →
  (a + 3)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l1221_122135


namespace NUMINAMATH_CALUDE_girl_scout_cookies_l1221_122124

theorem girl_scout_cookies (total_goal : ℕ) (boxes_left : ℕ) (first_customer : ℕ) : 
  total_goal = 150 →
  boxes_left = 75 →
  first_customer = 5 →
  let second_customer := 4 * first_customer
  let third_customer := second_customer / 2
  let fourth_customer := 3 * third_customer
  let sold_to_first_four := first_customer + second_customer + third_customer + fourth_customer
  total_goal - boxes_left - sold_to_first_four = 10 := by
sorry


end NUMINAMATH_CALUDE_girl_scout_cookies_l1221_122124


namespace NUMINAMATH_CALUDE_hexagram_arrangement_count_l1221_122109

/-- A regular hexagram has 12 symmetries (6 rotations and 6 reflections) -/
def hexagram_symmetries : ℕ := 12

/-- The number of ways to arrange 12 distinct objects -/
def total_arrangements : ℕ := Nat.factorial 12

/-- The number of distinct arrangements of 12 different objects on a regular hexagram,
    considering rotations and reflections as equivalent -/
def distinct_hexagram_arrangements : ℕ := total_arrangements / hexagram_symmetries

theorem hexagram_arrangement_count :
  distinct_hexagram_arrangements = Nat.factorial 11 :=
sorry

end NUMINAMATH_CALUDE_hexagram_arrangement_count_l1221_122109


namespace NUMINAMATH_CALUDE_total_pencils_l1221_122123

theorem total_pencils (drawer : Real) (desk_initial : Real) (pencil_case : Real) (dan_added : Real)
  (h1 : drawer = 43.5)
  (h2 : desk_initial = 19.25)
  (h3 : pencil_case = 8.75)
  (h4 : dan_added = 16) :
  drawer + desk_initial + pencil_case + dan_added = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1221_122123


namespace NUMINAMATH_CALUDE_fraction_equality_l1221_122199

-- Define the @ operation
def at_op (a b : ℕ) : ℕ := a * b + b^2

-- Define the # operation
def hash_op (a b : ℕ) : ℕ := a + b + a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 5 3 : ℚ) / (hash_op 5 3) = 24 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1221_122199


namespace NUMINAMATH_CALUDE_solution_replacement_concentration_l1221_122192

/-- Calculates the new concentration of a solution after partial replacement -/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (1 - replaced_fraction) * initial_conc + replaced_fraction * replacement_conc

/-- Theorem stating that replacing 7/9 of a 70% solution with a 25% solution results in a 35% solution -/
theorem solution_replacement_concentration :
  new_concentration 0.7 0.25 (7/9) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_concentration_l1221_122192


namespace NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2019_l1221_122116

/-- A number composed only of 0 and 1 in decimal form -/
def BinaryDecimal (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of numbers composed only of 0 and 1 in decimal form that are divisible by 2019 -/
def BinaryDecimalDivisibleBy2019 : Set ℕ :=
  {n : ℕ | BinaryDecimal n ∧ 2019 ∣ n}

/-- The set of numbers composed only of 0 and 1 in decimal form that are divisible by 2019 is infinite -/
theorem infinite_binary_decimal_divisible_by_2019 :
    Set.Infinite BinaryDecimalDivisibleBy2019 :=
  sorry

end NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2019_l1221_122116


namespace NUMINAMATH_CALUDE_right_prism_circumscribed_sphere_radius_l1221_122146

/-- A right prism with a square base -/
structure RightPrism where
  baseEdgeLength : ℝ
  sideEdgeLength : ℝ

/-- The sphere that circumscribes the right prism -/
structure CircumscribedSphere (p : RightPrism) where
  radius : ℝ
  contains_vertices : Prop  -- This represents the condition that all vertices lie on the sphere

/-- Theorem stating that for a right prism with base edge length 1 and side edge length 2,
    if all its vertices lie on a sphere, then the radius of that sphere is √6/2 -/
theorem right_prism_circumscribed_sphere_radius 
  (p : RightPrism) 
  (s : CircumscribedSphere p) 
  (h1 : p.baseEdgeLength = 1) 
  (h2 : p.sideEdgeLength = 2) 
  (h3 : s.contains_vertices) : 
  s.radius = Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_right_prism_circumscribed_sphere_radius_l1221_122146


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1221_122111

theorem sum_of_fractions : (1 : ℚ) / 2 + 2 / 3 + 3 / 5 = 53 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1221_122111


namespace NUMINAMATH_CALUDE_optimal_solution_is_valid_and_unique_l1221_122139

/-- Represents the solution for the tourist attraction problem -/
structure TouristAttractionSolution where
  small_car_cost : ℕ
  large_car_cost : ℕ
  small_car_trips : ℕ
  large_car_trips : ℕ

/-- Checks if a solution is valid for the tourist attraction problem -/
def is_valid_solution (s : TouristAttractionSolution) : Prop :=
  -- Total number of employees is 70
  4 * s.small_car_trips + 11 * s.large_car_trips = 70 ∧
  -- Small car cost is 5 more than large car cost
  s.small_car_cost = s.large_car_cost + 5 ∧
  -- Revenue difference between large and small car when fully loaded
  11 * s.large_car_cost - 4 * s.small_car_cost = 50 ∧
  -- Total cost does not exceed 5000
  70 * 60 + 4 * s.small_car_trips * s.small_car_cost + 
  11 * s.large_car_trips * s.large_car_cost ≤ 5000

/-- The optimal solution for the tourist attraction problem -/
def optimal_solution : TouristAttractionSolution :=
  { small_car_cost := 15
  , large_car_cost := 10
  , small_car_trips := 1
  , large_car_trips := 6 }

/-- Theorem stating that the optimal solution is valid and unique -/
theorem optimal_solution_is_valid_and_unique :
  is_valid_solution optimal_solution ∧
  ∀ s : TouristAttractionSolution, 
    is_valid_solution s → s = optimal_solution :=
sorry


end NUMINAMATH_CALUDE_optimal_solution_is_valid_and_unique_l1221_122139


namespace NUMINAMATH_CALUDE_sine_plus_abs_sine_integral_l1221_122179

open Set
open MeasureTheory
open Real

theorem sine_plus_abs_sine_integral : 
  ∫ x in (-π/2)..(π/2), (sin x + |sin x|) = 2 := by sorry

end NUMINAMATH_CALUDE_sine_plus_abs_sine_integral_l1221_122179


namespace NUMINAMATH_CALUDE_power_equation_solution_l1221_122143

theorem power_equation_solution :
  (∃ x : ℤ, (10 : ℝ)^655 * (10 : ℝ)^x = 1000) ∧
  (∀ x : ℤ, (10 : ℝ)^655 * (10 : ℝ)^x = 1000 → x = -652) :=
by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1221_122143


namespace NUMINAMATH_CALUDE_inequality_proof_l1221_122127

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_min : min (a * b) (min (b * c) (c * a)) ≥ 1) :
  (((a^2 + 1) * (b^2 + 1) * (c^2 + 1))^(1/3) : ℝ) ≤ ((a + b + c) / 3)^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1221_122127


namespace NUMINAMATH_CALUDE_bianca_extra_flowers_l1221_122115

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem stating that Bianca picked 7 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 39 49 81 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bianca_extra_flowers_l1221_122115


namespace NUMINAMATH_CALUDE_rectangle_width_l1221_122110

theorem rectangle_width (w : ℝ) (l : ℝ) (P : ℝ) : 
  l = 2 * w + 6 →  -- length is 6 more than twice the width
  P = 2 * l + 2 * w →  -- perimeter formula
  P = 120 →  -- given perimeter
  w = 18 :=  -- width to prove
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1221_122110


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1221_122167

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 480) 
  (h2 : height = 15) : 
  area / height = 32 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1221_122167


namespace NUMINAMATH_CALUDE_alice_burger_expense_l1221_122159

/-- The amount Alice spent on burgers in June -/
def aliceSpentOnBurgers (daysInJune : ℕ) (burgersPerDay : ℕ) (costPerBurger : ℕ) : ℕ :=
  daysInJune * burgersPerDay * costPerBurger

/-- Proof that Alice spent $1560 on burgers in June -/
theorem alice_burger_expense :
  aliceSpentOnBurgers 30 4 13 = 1560 := by
  sorry

end NUMINAMATH_CALUDE_alice_burger_expense_l1221_122159


namespace NUMINAMATH_CALUDE_mountaineering_teams_l1221_122184

/-- Represents the number of teams that can be formed in a mountaineering competition. -/
def max_teams (total_students : ℕ) (advanced_climbers : ℕ) (intermediate_climbers : ℕ) (beginner_climbers : ℕ)
  (advanced_points : ℕ) (intermediate_points : ℕ) (beginner_points : ℕ)
  (team_advanced : ℕ) (team_intermediate : ℕ) (team_beginner : ℕ)
  (max_team_points : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of teams that can be formed under the given constraints. -/
theorem mountaineering_teams :
  max_teams 172 45 70 57 80 50 30 5 8 5 1000 = 8 :=
by sorry

end NUMINAMATH_CALUDE_mountaineering_teams_l1221_122184


namespace NUMINAMATH_CALUDE_joan_rock_collection_l1221_122177

theorem joan_rock_collection (minerals_today minerals_yesterday gemstones : ℕ) : 
  gemstones = minerals_yesterday / 2 →
  minerals_today = minerals_yesterday + 6 →
  minerals_today = 48 →
  gemstones = 21 := by
sorry

end NUMINAMATH_CALUDE_joan_rock_collection_l1221_122177


namespace NUMINAMATH_CALUDE_min_overlap_cells_l1221_122120

/-- Given positive integers m and n where m < n, in an n × n board filled with integers from 1 to n^2, 
    if the m largest numbers in each row are colored red and the m largest numbers in each column are colored blue, 
    then the minimum number of cells that are both red and blue is m^2. -/
theorem min_overlap_cells (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) : 
  (∀ (board : Fin n → Fin n → ℕ), 
    (∀ i j, board i j ∈ Finset.range (n^2 + 1)) →
    (∀ i, ∃ (red : Finset (Fin n)), red.card = m ∧ ∀ j ∈ red, ∀ k ∉ red, board i j ≥ board i k) →
    (∀ j, ∃ (blue : Finset (Fin n)), blue.card = m ∧ ∀ i ∈ blue, ∀ k ∉ blue, board i j ≥ board k j) →
    ∃ (overlap : Finset (Fin n × Fin n)), 
      overlap.card = m^2 ∧ 
      (∀ (i j), (i, j) ∈ overlap ↔ (∃ (red blue : Finset (Fin n)), 
        red.card = m ∧ blue.card = m ∧
        (∀ k ∉ red, board i j ≥ board i k) ∧
        (∀ k ∉ blue, board i j ≥ board k j) ∧
        i ∈ red ∧ j ∈ blue))) :=
by sorry

end NUMINAMATH_CALUDE_min_overlap_cells_l1221_122120


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1221_122145

theorem arithmetic_expression_evaluation :
  65 + (126 / 14) + (35 * 11) - 250 - (500 / 5)^2 = -9791 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1221_122145


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1221_122181

/-- Calculates the compound interest earned given the initial principal, interest rate, 
    compounding frequency, time, and final amount -/
def compound_interest_earned (principal : ℝ) (rate : ℝ) (n : ℝ) (time : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - principal

/-- Theorem stating that for an investment with 8% annual interest rate compounded annually 
    for 2 years, resulting in a total of 19828.80, the interest earned is 2828.80 -/
theorem compound_interest_problem :
  ∃ (principal : ℝ),
    principal > 0 ∧
    (principal * (1 + 0.08)^2 = 19828.80) ∧
    (compound_interest_earned principal 0.08 1 2 19828.80 = 2828.80) := by
  sorry


end NUMINAMATH_CALUDE_compound_interest_problem_l1221_122181


namespace NUMINAMATH_CALUDE_odd_prime_condition_l1221_122168

theorem odd_prime_condition (p : ℕ) : 
  (Prime p ∧ Odd p) →
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 → Prime (1 + k * (p - 1))) →
  p = 3 ∨ p = 7 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_condition_l1221_122168


namespace NUMINAMATH_CALUDE_power_of_product_squared_l1221_122155

theorem power_of_product_squared (a : ℝ) : (3 * a^2)^2 = 9 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l1221_122155


namespace NUMINAMATH_CALUDE_tangent_line_product_l1221_122182

/-- A cubic function with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem tangent_line_product (a b : ℝ) (h1 : a ≠ 0) :
  f_derivative a 2 = 0 ∧ f a b 2 = 8 → a * b = 128 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_product_l1221_122182


namespace NUMINAMATH_CALUDE_inequalities_proof_l1221_122130

theorem inequalities_proof :
  (Real.log (Real.sqrt 2) < Real.sqrt 2 / 2) ∧
  (2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1221_122130


namespace NUMINAMATH_CALUDE_sport_water_amount_l1221_122198

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  cornSyrup : ℚ
  water : ℚ

/-- Represents the amount of ingredients in ounces -/
structure DrinkAmount where
  flavoring : ℚ
  cornSyrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standardRatio : DrinkRatio :=
  { flavoring := 1, cornSyrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sportRatio (standard : DrinkRatio) : DrinkRatio :=
  { flavoring := standard.flavoring,
    cornSyrup := standard.cornSyrup / 3,
    water := standard.water * 2 }

/-- Theorem stating the amount of water in the sport formulation -/
theorem sport_water_amount 
  (standard : DrinkRatio)
  (sport : DrinkRatio)
  (sportAmount : DrinkAmount)
  (h1 : sport = sportRatio standard)
  (h2 : sportAmount.cornSyrup = 2) :
  sportAmount.water = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_sport_water_amount_l1221_122198


namespace NUMINAMATH_CALUDE_magnitude_comparison_l1221_122126

theorem magnitude_comparison (a b c : ℝ) 
  (ha : a > 0) 
  (hbc : b * c > a^2) 
  (heq : a^2 - 2*a*b + c^2 = 0) : 
  b > c ∧ c > a :=
by sorry

end NUMINAMATH_CALUDE_magnitude_comparison_l1221_122126


namespace NUMINAMATH_CALUDE_right_triangle_and_inverse_l1221_122102

theorem right_triangle_and_inverse (a b c : Nat) (m : Nat) : 
  a = 48 → b = 55 → c = 73 → m = 4273 →
  a * a + b * b = c * c →
  (∃ (x : Nat), x * 480 ≡ 1 [MOD m]) →
  (∃ (y : Nat), y * 480 ≡ 1643 [MOD m]) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_inverse_l1221_122102


namespace NUMINAMATH_CALUDE_investment_B_is_72000_l1221_122149

/-- Represents the investment and profit distribution in a partnership. -/
structure Partnership where
  investA : ℕ
  investC : ℕ
  profitC : ℕ
  totalProfit : ℕ

/-- Calculates the investment of partner B given the partnership details. -/
def calculateInvestmentB (p : Partnership) : ℕ :=
  p.totalProfit * p.investC / p.profitC - p.investA - p.investC

/-- Theorem stating that given the specified partnership conditions, B's investment is 72000. -/
theorem investment_B_is_72000 (p : Partnership) 
  (h1 : p.investA = 27000)
  (h2 : p.investC = 81000)
  (h3 : p.profitC = 36000)
  (h4 : p.totalProfit = 80000) :
  calculateInvestmentB p = 72000 := by
  sorry

#eval calculateInvestmentB ⟨27000, 81000, 36000, 80000⟩

end NUMINAMATH_CALUDE_investment_B_is_72000_l1221_122149


namespace NUMINAMATH_CALUDE_type_B_completion_time_l1221_122105

/-- The time (in hours) it takes for a type R machine to complete the job -/
def time_R : ℝ := 5

/-- The time (in hours) it takes for 2 type R machines and 3 type B machines working together to complete the job -/
def time_combined : ℝ := 1.2068965517241381

/-- The time (in hours) it takes for a type B machine to complete the job -/
def time_B : ℝ := 7

theorem type_B_completion_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_B - (3 * time_combined) / (1 / time_combined - 2 / time_R)| < ε :=
sorry

end NUMINAMATH_CALUDE_type_B_completion_time_l1221_122105


namespace NUMINAMATH_CALUDE_chris_donuts_l1221_122161

/-- The number of donuts Chris initially bought -/
def initial_donuts : ℕ := 30

/-- The percentage of donuts Chris ate while driving -/
def eaten_percentage : ℚ := 1/10

/-- The number of donuts Chris grabbed for his afternoon snack -/
def afternoon_snack : ℕ := 4

/-- The number of donuts left for Chris's co-workers -/
def remaining_donuts : ℕ := 23

theorem chris_donuts :
  (initial_donuts : ℚ) * (1 - eaten_percentage) - afternoon_snack = remaining_donuts :=
sorry

end NUMINAMATH_CALUDE_chris_donuts_l1221_122161


namespace NUMINAMATH_CALUDE_four_team_tournament_handshakes_l1221_122136

/-- The number of handshakes in a tournament with teams of two -/
def tournament_handshakes (num_teams : ℕ) : ℕ :=
  let total_people := num_teams * 2
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a tournament with 4 teams of 2 people each, where each person
    shakes hands with everyone except their partner and themselves,
    the total number of handshakes is 24. -/
theorem four_team_tournament_handshakes :
  tournament_handshakes 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_team_tournament_handshakes_l1221_122136


namespace NUMINAMATH_CALUDE_abs_difference_given_sum_and_product_l1221_122147

theorem abs_difference_given_sum_and_product (a b : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a + b = 6) : 
  |a - b| = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_given_sum_and_product_l1221_122147


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1221_122165

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^4 + y^2 = 3*y + 3 := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1221_122165


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1221_122187

/-- Given a line L1 with equation x - 2y + 3 = 0, prove that the line L2 with equation 2x + y - 1 = 0
    passes through the point (-1, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ × ℝ → Prop := fun (x, y) ↦ x - 2*y + 3 = 0
  let L2 : ℝ × ℝ → Prop := fun (x, y) ↦ 2*x + y - 1 = 0
  let point : ℝ × ℝ := (-1, 3)
  (L2 point) ∧ 
  (∀ (p q : ℝ × ℝ), L1 p ∧ L1 q ∧ p ≠ q → 
    let v1 := (p.1 - q.1, p.2 - q.2)
    let v2 := (1, 2)
    v1.1 * v2.1 + v1.2 * v2.2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1221_122187


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l1221_122176

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = 2^p - 1 ∧ Nat.Prime n

theorem largest_mersenne_prime_under_1000 :
  ∀ n : ℕ, is_mersenne_prime n → n < 1000 → n ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l1221_122176


namespace NUMINAMATH_CALUDE_triangle_properties_l1221_122134

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Given conditions
  b * (1 + Real.cos C) = c * (2 - Real.cos B) →
  C = π / 3 →
  1/2 * a * b * Real.sin C = 4 * Real.sqrt 3 →
  -- Conclusions to prove
  (a + b = 2 * c ∧ c = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1221_122134


namespace NUMINAMATH_CALUDE_marble_count_l1221_122129

/-- The number of marbles each person has --/
structure Marbles where
  ed : ℕ
  doug : ℕ
  charlie : ℕ

/-- The initial state of marbles before Ed lost some --/
def initial_marbles : Marbles → Marbles
| ⟨ed, doug, charlie⟩ => ⟨ed + 20, doug, charlie⟩

theorem marble_count (m : Marbles) :
  (initial_marbles m).ed = (initial_marbles m).doug + 12 →
  m.ed = 17 →
  m.charlie = 4 * m.doug →
  m.doug = 25 ∧ m.charlie = 100 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l1221_122129


namespace NUMINAMATH_CALUDE_last_four_digits_pow_5_2017_l1221_122154

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The cycle length of the last four digits of powers of 5 -/
def cycleLengthPowersOf5 : ℕ := 4

theorem last_four_digits_pow_5_2017 :
  lastFourDigits (5^2017) = lastFourDigits (5^5) :=
sorry

end NUMINAMATH_CALUDE_last_four_digits_pow_5_2017_l1221_122154


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1221_122119

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (non_intersecting : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane
  (m n : Line) (α β : Plane)
  (different_lines : m ≠ n)
  (non_intersecting_planes : non_intersecting α β)
  (m_parallel_n : parallel m n)
  (n_perp_β : perpendicular n β) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1221_122119


namespace NUMINAMATH_CALUDE_product_of_solutions_l1221_122186

theorem product_of_solutions (x : ℝ) : 
  (12 = 2 * x^2 + 4 * x) → 
  (∃ x₁ x₂ : ℝ, (12 = 2 * x₁^2 + 4 * x₁) ∧ (12 = 2 * x₂^2 + 4 * x₂) ∧ (x₁ * x₂ = -6)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1221_122186


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1221_122174

theorem greatest_of_three_consecutive_integers (x : ℤ) :
  (x + (x + 1) + (x + 2) = 33) → (max x (max (x + 1) (x + 2)) = 12) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1221_122174


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1221_122153

theorem impossible_coin_probabilities :
  ¬ ∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1221_122153


namespace NUMINAMATH_CALUDE_l_shaped_area_l1221_122193

/-- The area of an L-shaped region formed by subtracting three non-overlapping squares
    from a larger square -/
theorem l_shaped_area (side_length : ℝ) (small_square1 : ℝ) (small_square2 : ℝ) (small_square3 : ℝ)
    (h1 : side_length = 6)
    (h2 : small_square1 = 2)
    (h3 : small_square2 = 4)
    (h4 : small_square3 = 2)
    (h5 : small_square1 + small_square2 + small_square3 ≤ side_length) :
    side_length^2 - (small_square1^2 + small_square2^2 + small_square3^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l1221_122193


namespace NUMINAMATH_CALUDE_least_possible_beta_l1221_122142

-- Define a structure for the right triangle
structure RightTriangle where
  alpha : ℕ
  beta : ℕ
  is_right_triangle : alpha + beta = 100
  alpha_prime : Nat.Prime alpha
  beta_prime : Nat.Prime beta
  alpha_odd : Odd alpha
  beta_odd : Odd beta
  alpha_greater : alpha > beta

-- Define the theorem
theorem least_possible_beta (t : RightTriangle) : 
  ∃ (min_beta : ℕ), min_beta = 3 ∧ 
  ∀ (valid_triangle : RightTriangle), valid_triangle.beta ≥ min_beta :=
sorry

end NUMINAMATH_CALUDE_least_possible_beta_l1221_122142


namespace NUMINAMATH_CALUDE_int_poly5_root_count_l1221_122163

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPoly5 where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The number of integer roots (counting multiplicity) of an IntPoly5 -/
def numIntRoots (p : IntPoly5) : ℕ := sorry

/-- The theorem stating the possible values for the number of integer roots -/
theorem int_poly5_root_count (p : IntPoly5) : 
  numIntRoots p ∈ ({0, 1, 2, 3, 4, 5} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_int_poly5_root_count_l1221_122163


namespace NUMINAMATH_CALUDE_trent_total_distance_l1221_122128

/-- Represents the distance Trent traveled throughout his day -/
def trent_travel (block_length : ℕ) (walk_blocks : ℕ) (bus_blocks : ℕ) (bike_blocks : ℕ) : ℕ :=
  2 * (walk_blocks + bus_blocks + bike_blocks) * block_length

/-- Theorem stating the total distance Trent traveled -/
theorem trent_total_distance :
  trent_travel 50 4 7 5 = 1600 := by sorry

end NUMINAMATH_CALUDE_trent_total_distance_l1221_122128


namespace NUMINAMATH_CALUDE_socks_total_is_112_25_l1221_122178

/-- The total number of socks George and Maria have after receiving additional socks -/
def total_socks (george_initial : ℝ) (maria_initial : ℝ) 
                (george_bought : ℝ) (george_from_dad : ℝ) 
                (maria_from_mom : ℝ) (maria_from_aunt : ℝ) : ℝ :=
  (george_initial + george_bought + george_from_dad) + 
  (maria_initial + maria_from_mom + maria_from_aunt)

/-- Theorem stating that the total number of socks is 112.25 -/
theorem socks_total_is_112_25 : 
  total_socks 28.5 24.75 36.25 4.5 15.5 2.75 = 112.25 := by
  sorry

end NUMINAMATH_CALUDE_socks_total_is_112_25_l1221_122178


namespace NUMINAMATH_CALUDE_circle_equation_l1221_122103

/-- Given a circle with center (2, -1) and a chord of length 2√2 intercepted by the line x - y - 1 = 0,
    prove that the equation of the circle is (x-2)² + (y+1)² = 4 -/
theorem circle_equation (x y : ℝ) : 
  let center := (2, -1)
  let line := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  let chord_length := 2 * Real.sqrt 2
  true → (x - 2)^2 + (y + 1)^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_l1221_122103


namespace NUMINAMATH_CALUDE_range_of_r_l1221_122125

noncomputable def r (x : ℝ) : ℝ := 1 / (1 - x)^3

theorem range_of_r :
  Set.range r = {y : ℝ | y < 0 ∨ y > 0} :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l1221_122125


namespace NUMINAMATH_CALUDE_bologna_sandwiches_l1221_122121

/-- Given a ratio of cheese, bologna, and peanut butter sandwiches as 1:7:8 and a total of 80 sandwiches,
    prove that the number of bologna sandwiches is 35. -/
theorem bologna_sandwiches (total : ℕ) (cheese : ℕ) (bologna : ℕ) (peanut_butter : ℕ)
  (h_total : total = 80)
  (h_ratio : cheese + bologna + peanut_butter = 16)
  (h_cheese : cheese = 1)
  (h_bologna : bologna = 7)
  (h_peanut_butter : peanut_butter = 8) :
  (total / (cheese + bologna + peanut_butter)) * bologna = 35 :=
by sorry

end NUMINAMATH_CALUDE_bologna_sandwiches_l1221_122121


namespace NUMINAMATH_CALUDE_songs_added_l1221_122131

theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 7 → final = 28 → final - (initial - deleted) = 24 :=
by sorry

end NUMINAMATH_CALUDE_songs_added_l1221_122131


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l1221_122114

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  /-- The number of small cubes along each edge of the large cube -/
  edge_count : ℕ
  /-- All faces of the large cube are painted -/
  all_faces_painted : Bool
  /-- The number of small cubes with three faces colored -/
  three_face_colored_count : ℕ

/-- Theorem: If a cube is cut so that 8 small cubes have three faces colored, 
    then the total number of small cubes is 8 -/
theorem cut_cube_theorem (c : CutCube) 
  (h1 : c.all_faces_painted = true) 
  (h2 : c.three_face_colored_count = 8) : 
  c.edge_count ^ 3 = 8 := by
  sorry

#check cut_cube_theorem

end NUMINAMATH_CALUDE_cut_cube_theorem_l1221_122114


namespace NUMINAMATH_CALUDE_granola_bar_distribution_l1221_122169

/-- Given that Monroe made x granola bars, she and her husband ate 2/3 of them,
    and the rest were divided equally among y children, with each child receiving z granola bars,
    prove that z = x / (3 * y) -/
theorem granola_bar_distribution (x y z : ℚ) (hx : x > 0) (hy : y > 0) : 
  (2 / 3 * x + y * z = x) → z = x / (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_granola_bar_distribution_l1221_122169


namespace NUMINAMATH_CALUDE_non_working_games_l1221_122132

theorem non_working_games (total : ℕ) (working : ℕ) (h1 : total = 30) (h2 : working = 17) :
  total - working = 13 := by
  sorry

end NUMINAMATH_CALUDE_non_working_games_l1221_122132


namespace NUMINAMATH_CALUDE_paint_canvas_cost_ratio_l1221_122100

theorem paint_canvas_cost_ratio 
  (canvas_original : ℝ) 
  (paint_original : ℝ) 
  (canvas_decrease : ℝ) 
  (paint_decrease : ℝ) 
  (total_decrease : ℝ)
  (h1 : canvas_decrease = 0.4)
  (h2 : paint_decrease = 0.6)
  (h3 : total_decrease = 0.5599999999999999)
  (h4 : canvas_original > 0)
  (h5 : paint_original > 0)
  (h6 : (1 - paint_decrease) * paint_original + (1 - canvas_decrease) * canvas_original 
      = (1 - total_decrease) * (paint_original + canvas_original)) :
  paint_original / canvas_original = 4 := by
sorry

end NUMINAMATH_CALUDE_paint_canvas_cost_ratio_l1221_122100


namespace NUMINAMATH_CALUDE_remainder_sum_l1221_122117

theorem remainder_sum (n : ℤ) (h : n % 12 = 5) : (n % 3 + n % 4 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1221_122117


namespace NUMINAMATH_CALUDE_min_fish_in_aquarium_l1221_122175

/-- Represents the number of fish of each known color in the aquarium -/
structure AquariumFish where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The conditions of the aquarium as described in the problem -/
def aquarium_conditions (fish : AquariumFish) : Prop :=
  fish.yellow = 12 ∧
  fish.blue = fish.yellow / 2 ∧
  fish.green = fish.yellow * 2

/-- The theorem stating the minimum number of fish in the aquarium -/
theorem min_fish_in_aquarium (fish : AquariumFish) 
  (h : aquarium_conditions fish) : 
  fish.yellow + fish.blue + fish.green = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_fish_in_aquarium_l1221_122175


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l1221_122112

theorem pigeonhole_on_permutation_sums (n : ℕ) : 
  ∀ (p : Fin (2*n) → Fin (2*n)), 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ 
  (p i + i.val + 1) % (2*n) = (p j + j.val + 1) % (2*n) := by
sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l1221_122112


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1221_122180

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 4 * Real.sqrt 2 * x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1221_122180


namespace NUMINAMATH_CALUDE_triangle_theorem_l1221_122122

noncomputable section

variables {a b c : ℝ} {A B C : ℝ}

def triangle_area (a b c : ℝ) : ℝ := (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

theorem triangle_theorem 
  (h1 : b^2 + c^2 - a^2 = a*c*(Real.cos C) + c^2*(Real.cos A))
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : a*Real.sin B = b*Real.sin A)
  (h7 : b*Real.sin C = c*Real.sin B)
  (h8 : triangle_area a b c = 25*(Real.sqrt 3)/4)
  (h9 : a = 5) :
  A = π/3 ∧ Real.sin B + Real.sin C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1221_122122


namespace NUMINAMATH_CALUDE_worker_room_arrangement_l1221_122156

/-- The number of rooms and workers -/
def n : ℕ := 5

/-- The number of unchosen rooms -/
def k : ℕ := 2

/-- Represents whether each room choice is equally likely -/
def equal_probability : Prop := sorry

/-- Represents the condition that unchosen rooms are not adjacent -/
def non_adjacent_unchosen : Prop := sorry

/-- The number of ways to arrange workers in rooms with given conditions -/
def arrangement_count : ℕ := sorry

theorem worker_room_arrangement :
  arrangement_count = 900 :=
sorry

end NUMINAMATH_CALUDE_worker_room_arrangement_l1221_122156


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_l1221_122148

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def only_zero_and_one (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting only of 0s and 1s that is divisible by 15 -/
def smallest_T : ℕ := sorry

theorem smallest_divisible_by_15 :
  smallest_T > 0 ∧
  only_zero_and_one smallest_T ∧
  smallest_T % 15 = 0 ∧
  smallest_T / 15 = 74 ∧
  ∀ n : ℕ, n > 0 → only_zero_and_one n → n % 15 = 0 → n ≥ smallest_T :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_l1221_122148


namespace NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l1221_122138

theorem ellipse_max_y_coordinate :
  ∀ x y : ℝ, x^2/25 + (y-3)^2/9 = 1 → y ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l1221_122138


namespace NUMINAMATH_CALUDE_fraction_transformation_l1221_122118

theorem fraction_transformation (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →
  (2 + 3 : ℚ) / (d + 3) = 1 / 3 →
  d = 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1221_122118


namespace NUMINAMATH_CALUDE_complementary_angle_of_37_38_l1221_122191

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- The sum of two angles in degrees and minutes -/
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let carryDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + carryDegrees
  , minutes := totalMinutes % 60 }

/-- Checks if two angles are complementary -/
def are_complementary (a b : Angle) : Prop :=
  Angle.add a b = ⟨90, 0⟩

/-- The main theorem statement -/
theorem complementary_angle_of_37_38 :
  let angle : Angle := ⟨37, 38⟩
  let complement : Angle := ⟨52, 22⟩
  are_complementary angle complement :=
by sorry

end NUMINAMATH_CALUDE_complementary_angle_of_37_38_l1221_122191


namespace NUMINAMATH_CALUDE_acute_triangle_from_sides_l1221_122194

theorem acute_triangle_from_sides (a b c : ℝ) (ha : a = 5) (hb : b = 6) (hc : c = 7) :
  a + b > c ∧ b + c > a ∧ c + a > b ∧ a^2 + b^2 > c^2 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_from_sides_l1221_122194


namespace NUMINAMATH_CALUDE_impossibleToRemoveAllPieces_l1221_122158

/-- Represents the color of a cell or piece -/
inductive Color
| Black
| White

/-- Represents a move on the board -/
structure Move where
  piece1 : Nat × Nat
  piece2 : Nat × Nat
  newPos1 : Nat × Nat
  newPos2 : Nat × Nat

/-- Represents the state of the board -/
structure BoardState where
  pieces : List (Nat × Nat)

/-- Returns the color of a cell given its coordinates -/
def cellColor (pos : Nat × Nat) : Color :=
  if (pos.1 + pos.2) % 2 == 0 then Color.Black else Color.White

/-- Checks if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Nat × Nat) : Bool :=
  (pos1.1 == pos2.1 && (pos1.2 + 1 == pos2.2 || pos1.2 == pos2.2 + 1)) ||
  (pos1.2 == pos2.2 && (pos1.1 + 1 == pos2.1 || pos1.1 == pos2.1 + 1))

/-- Checks if a move is valid -/
def isValidMove (m : Move) : Bool :=
  isAdjacent m.piece1 m.newPos1 && isAdjacent m.piece2 m.newPos2

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (m : Move) : BoardState :=
  sorry

/-- Theorem: It is impossible to remove all pieces from the board -/
theorem impossibleToRemoveAllPieces :
  ∀ (moves : List Move),
    let initialState : BoardState := { pieces := List.range 506 }
    let finalState := moves.foldl applyMove initialState
    finalState.pieces.length > 0 := by
  sorry

end NUMINAMATH_CALUDE_impossibleToRemoveAllPieces_l1221_122158


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1221_122140

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + sequence_a n

theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ sequence_a n ↔ 2^k ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1221_122140


namespace NUMINAMATH_CALUDE_water_consumption_equation_l1221_122160

theorem water_consumption_equation (x : ℝ) (h : x > 0) : 
  (80 / x) - (80 * (1 - 0.2) / x) = 5 ↔ 
  (80 / x) - (80 / (x / (1 - 0.2))) = 5 :=
sorry

end NUMINAMATH_CALUDE_water_consumption_equation_l1221_122160


namespace NUMINAMATH_CALUDE_square_difference_eq_85_solutions_l1221_122157

theorem square_difference_eq_85_solutions : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 85) (Finset.product (Finset.range 1000) (Finset.range 1000))).card :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_eq_85_solutions_l1221_122157


namespace NUMINAMATH_CALUDE_regular_decagon_area_l1221_122150

theorem regular_decagon_area (s : ℝ) (h : s = Real.sqrt 2) :
  let area := 5 * s^2 * (Real.sqrt (5 + 2 * Real.sqrt 5)) / 4
  area = 3.5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_area_l1221_122150


namespace NUMINAMATH_CALUDE_lesser_number_l1221_122196

theorem lesser_number (x y : ℤ) (sum_eq : x + y = 58) (diff_eq : x - y = 6) : 
  min x y = 26 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_l1221_122196


namespace NUMINAMATH_CALUDE_six_grade_sequences_l1221_122183

/-- Represents the number of ways to assign n grades under the given conditions -/
def gradeSequences (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeSequences (n + 1) + 2 * gradeSequences n

/-- The number of ways to assign 6 grades under the given conditions is 448 -/
theorem six_grade_sequences : gradeSequences 6 = 448 := by
  sorry

/-- Helper lemma: The recurrence relation holds for all n ≥ 2 -/
lemma recurrence_relation (n : ℕ) (h : n ≥ 2) :
  gradeSequences n = 2 * gradeSequences (n - 1) + 2 * gradeSequences (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_six_grade_sequences_l1221_122183


namespace NUMINAMATH_CALUDE_complex_distance_theorem_l1221_122170

theorem complex_distance_theorem (z z₁ z₂ : ℂ) :
  z₁ ≠ z₂ →
  z₁^2 = -2 - 2 * Complex.I * Real.sqrt 3 →
  z₂^2 = -2 - 2 * Complex.I * Real.sqrt 3 →
  Complex.abs (z - z₁) = 4 →
  Complex.abs (z - z₂) = 4 →
  Complex.abs z = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_theorem_l1221_122170


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l1221_122137

theorem at_least_one_real_root (m : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5*x + m ≠ 0 ∧ 2*x^2 + x + 6 - m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l1221_122137


namespace NUMINAMATH_CALUDE_similar_triangles_theorem_l1221_122162

/-- Two similar right-angled triangles -/
structure SimilarRightTriangles where
  /-- First triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  m_c : ℝ
  /-- Second triangle -/
  a' : ℝ
  b' : ℝ
  c' : ℝ
  m_c' : ℝ
  /-- c and c' are hypotenuses -/
  hyp_c : c^2 = a^2 + b^2
  hyp_c' : c'^2 = a'^2 + b'^2
  /-- Triangles are similar -/
  similar : ∃ (k : ℝ), k > 0 ∧ a = k * a' ∧ b = k * b' ∧ c = k * c' ∧ m_c = k * m_c'

/-- Theorem about similar right-angled triangles -/
theorem similar_triangles_theorem (t : SimilarRightTriangles) :
  (t.a * t.a' + t.b * t.b' = t.c * t.c') ∧
  (1 / (t.a * t.a') + 1 / (t.b * t.b') = 1 / (t.m_c * t.m_c')) := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_theorem_l1221_122162


namespace NUMINAMATH_CALUDE_orange_bin_count_l1221_122141

theorem orange_bin_count (initial : ℕ) (removed : ℕ) (added : ℕ) :
  initial = 40 →
  removed = 37 →
  added = 7 →
  initial - removed + added = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_count_l1221_122141


namespace NUMINAMATH_CALUDE_product_abcd_l1221_122152

/-- Given positive real numbers a, b, c, and d satisfying the specified conditions,
    prove that their product equals 14400. -/
theorem product_abcd (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 762)
  (h_sum_ab_cd : a * b + c * d = 260)
  (h_sum_ac_bd : a * c + b * d = 365)
  (h_sum_ad_bc : a * d + b * c = 244) :
  a * b * c * d = 14400 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l1221_122152


namespace NUMINAMATH_CALUDE_mountain_climb_speed_l1221_122164

theorem mountain_climb_speed 
  (total_time : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 14) 
  (h2 : speed_difference = 0.5) 
  (h3 : time_difference = 2) 
  (h4 : total_distance = 52) : 
  ∃ (v : ℝ), v > 0 ∧ 
    v * (total_time / 2 + time_difference) + 
    (v + speed_difference) * (total_time / 2 - time_difference) = total_distance ∧
    v + speed_difference = 4 := by
  sorry

#check mountain_climb_speed

end NUMINAMATH_CALUDE_mountain_climb_speed_l1221_122164


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1221_122113

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (3 - 1 - a - b - c)) →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 5 = 51 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1221_122113


namespace NUMINAMATH_CALUDE_small_panda_bamboo_consumption_l1221_122133

/-- The amount of bamboo eaten by small pandas each day -/
def small_panda_bamboo : ℝ := 100

/-- The number of small panda bears -/
def num_small_pandas : ℕ := 4

/-- The number of bigger panda bears -/
def num_big_pandas : ℕ := 5

/-- The amount of bamboo eaten by each bigger panda bear per day -/
def big_panda_bamboo : ℝ := 40

/-- The total amount of bamboo eaten by all pandas in a week -/
def total_weekly_bamboo : ℝ := 2100

theorem small_panda_bamboo_consumption :
  small_panda_bamboo * num_small_pandas +
  big_panda_bamboo * num_big_pandas =
  total_weekly_bamboo / 7 :=
by sorry

end NUMINAMATH_CALUDE_small_panda_bamboo_consumption_l1221_122133


namespace NUMINAMATH_CALUDE_gcd_840_1764_l1221_122108

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l1221_122108


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1221_122188

/-- The value of 'a' for a parabola y^2 = ax (a > 0) with a point P(3/2, y₀) on the parabola,
    where the distance from P to the focus is 2. -/
theorem parabola_focus_distance (a : ℝ) (y₀ : ℝ) : 
  a > 0 ∧ y₀^2 = a * (3/2) ∧ ((3/2 - (-a/4))^2 + y₀^2)^(1/2) = 2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1221_122188


namespace NUMINAMATH_CALUDE_smallest_palindrome_base2_and_base5_l1221_122171

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Checks if a natural number has exactly k digits in the given base. -/
def has_k_digits (n : ℕ) (k : ℕ) (base : ℕ) : Prop := sorry

theorem smallest_palindrome_base2_and_base5 :
  ∀ n : ℕ,
  (has_k_digits n 5 2 ∧ is_palindrome n 2 ∧ is_palindrome (to_base n 5).length 5) →
  n ≥ 27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_palindrome_base2_and_base5_l1221_122171


namespace NUMINAMATH_CALUDE_prob_ace_then_diamond_standard_deck_l1221_122144

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (diamonds : Nat)
  (ace_of_diamonds : Nat)

/-- Probability of drawing an Ace first and a diamond second from a standard deck -/
def prob_ace_then_diamond (d : Deck) : ℚ :=
  let prob_ace_of_diamonds := d.ace_of_diamonds / d.cards
  let prob_other_ace := (d.aces - d.ace_of_diamonds) / d.cards
  let prob_diamond_after_ace_of_diamonds := (d.diamonds - 1) / (d.cards - 1)
  let prob_diamond_after_other_ace := d.diamonds / (d.cards - 1)
  prob_ace_of_diamonds * prob_diamond_after_ace_of_diamonds +
  prob_other_ace * prob_diamond_after_other_ace

theorem prob_ace_then_diamond_standard_deck :
  prob_ace_then_diamond { cards := 52, aces := 4, diamonds := 13, ace_of_diamonds := 1 } = 119 / 3571 :=
sorry

end NUMINAMATH_CALUDE_prob_ace_then_diamond_standard_deck_l1221_122144


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1221_122106

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (7, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 2 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1221_122106


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_example_l1221_122166

/-- Sum of an arithmetic series -/
def arithmetic_series_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℚ :=
  let n : ℚ := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

/-- Theorem: The sum of the arithmetic series with first term -35, last term 1, and common difference 2 is -323 -/
theorem arithmetic_series_sum_example : 
  arithmetic_series_sum (-35) 1 2 = -323 := by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_example_l1221_122166


namespace NUMINAMATH_CALUDE_square_difference_l1221_122172

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) :
  a^2 - b^2 = 40 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1221_122172


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1221_122189

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Length of the smaller segment of the lateral side -/
  smaller_segment : ℝ
  /-- Length of the larger segment of the lateral side -/
  larger_segment : ℝ

/-- Calculate the area of the isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 156 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoidWithInscribedCircle := ⟨4, 9⟩
  area t = 156 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1221_122189


namespace NUMINAMATH_CALUDE_birds_on_fence_l1221_122104

theorem birds_on_fence (initial_birds joining_birds : ℕ) : 
  initial_birds = 2 → joining_birds = 4 → initial_birds + joining_birds = 6 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1221_122104
