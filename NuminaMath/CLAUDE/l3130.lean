import Mathlib

namespace missile_interception_time_l3130_313071

/-- The time for a missile to intercept a circling plane -/
theorem missile_interception_time
  (r : ℝ)             -- radius of the plane's circular path
  (v : ℝ)             -- speed of both the plane and the missile
  (h : r = 10)        -- given radius is 10 km
  (k : v = 1000)      -- given speed is 1000 km/h
  : ∃ t : ℝ,          -- there exists a time t such that
    t = 18 * Real.pi ∧ -- t equals 18π
    t * (5 / 18) = (2 * Real.pi * r) / 4 / v -- t converted to hours equals quarter circumference divided by speed
    :=
by sorry

end missile_interception_time_l3130_313071


namespace inequality_chain_l3130_313039

theorem inequality_chain (a b d m : ℝ) 
  (h1 : a > b) (h2 : b > d) (h3 : d ≥ m) : a > m := by
  sorry

end inequality_chain_l3130_313039


namespace ron_four_times_maurice_age_l3130_313046

/-- The number of years in the future when Ron will be four times as old as Maurice -/
def years_until_four_times_age : ℕ → ℕ → ℕ 
| ron_age, maurice_age => 
  let x : ℕ := (ron_age - 4 * maurice_age) / 3
  x

theorem ron_four_times_maurice_age (ron_current_age maurice_current_age : ℕ) 
  (h1 : ron_current_age = 43)
  (h2 : maurice_current_age = 7) : 
  years_until_four_times_age ron_current_age maurice_current_age = 5 := by
  sorry

end ron_four_times_maurice_age_l3130_313046


namespace largest_possible_z_value_l3130_313030

open Complex

theorem largest_possible_z_value (a b c d z w : ℂ) 
  (h1 : abs a = abs b)
  (h2 : abs b = abs c)
  (h3 : abs c = abs d)
  (h4 : abs a > 0)
  (h5 : a * z^3 + b * w * z^2 + c * z + d = 0)
  (h6 : abs w = 1/2) :
  abs z ≤ 1 ∧ ∃ a b c d z w : ℂ, 
    abs a = abs b ∧ 
    abs b = abs c ∧ 
    abs c = abs d ∧ 
    abs a > 0 ∧
    a * z^3 + b * w * z^2 + c * z + d = 0 ∧
    abs w = 1/2 ∧
    abs z = 1 := by
  sorry

end largest_possible_z_value_l3130_313030


namespace reciprocal_relationship_l3130_313014

theorem reciprocal_relationship (a b : ℝ) : (a + b)^2 - (a - b)^2 = 4 → a * b = 1 := by
  sorry

end reciprocal_relationship_l3130_313014


namespace arithmetic_sequence_common_difference_l3130_313078

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n : ℕ, a n = a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_prod : a 1 * a 3 = 8) 
  (h_second : a 2 = 3) :
  ∃ d : ℝ, (d = 1 ∨ d = -1) ∧ 
    ∀ n : ℕ, a n = a 1 + (n - 1 : ℝ) * d :=
sorry

end arithmetic_sequence_common_difference_l3130_313078


namespace road_travel_cost_l3130_313028

/-- Calculate the cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sqm : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 40)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 3) :
  (((lawn_length * road_width + lawn_width * road_width) - road_width * road_width) : ℚ) * cost_per_sqm = 3300 :=
by sorry

end road_travel_cost_l3130_313028


namespace minimum_value_implies_ratio_l3130_313076

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem minimum_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≥ 10) ∧  -- f(x) has a minimum value of 10
  (f a b 1 = 10) ∧  -- The minimum occurs at x = 1
  (f_derivative a b 1 = 0)  -- The derivative is zero at x = 1
  → b / a = -1 / 2 := by sorry

end minimum_value_implies_ratio_l3130_313076


namespace min_value_inequality_l3130_313027

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9 / 4 := by
  sorry

end min_value_inequality_l3130_313027


namespace inequality_proof_l3130_313072

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x / (y + z) + y / (x + z) + z / (x + y) ≤ x * Real.sqrt x / 2 + y * Real.sqrt y / 2 + z * Real.sqrt z / 2 := by
  sorry

end inequality_proof_l3130_313072


namespace discounted_biographies_count_l3130_313040

theorem discounted_biographies_count (biography_price mystery_price total_savings mystery_count total_discount_rate mystery_discount_rate : ℝ) 
  (h1 : biography_price = 20)
  (h2 : mystery_price = 12)
  (h3 : total_savings = 19)
  (h4 : mystery_count = 3)
  (h5 : total_discount_rate = 0.43)
  (h6 : mystery_discount_rate = 0.375) :
  ∃ (biography_count : ℕ), 
    biography_count = 5 ∧ 
    biography_count * (biography_price * (total_discount_rate - mystery_discount_rate)) + 
    mystery_count * (mystery_price * mystery_discount_rate) = total_savings :=
by sorry

end discounted_biographies_count_l3130_313040


namespace triangle_perimeter_l3130_313021

theorem triangle_perimeter (a b c : ℕ) : 
  a = 7 → b = 2 → Odd c → a + b + c = 16 := by sorry

end triangle_perimeter_l3130_313021


namespace dragon_castle_theorem_l3130_313025

/-- Represents the configuration of a dragon tethered to a cylindrical castle -/
structure DragonCastle where
  castle_radius : ℝ
  chain_length : ℝ
  chain_height : ℝ
  dragon_distance : ℝ

/-- Calculates the length of the chain touching the castle -/
def chain_on_castle (dc : DragonCastle) : ℝ :=
  sorry

/-- Theorem stating the properties of the dragon-castle configuration -/
theorem dragon_castle_theorem (dc : DragonCastle) 
  (h1 : dc.castle_radius = 10)
  (h2 : dc.chain_length = 30)
  (h3 : dc.chain_height = 6)
  (h4 : dc.dragon_distance = 6) :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.Prime c ∧
    chain_on_castle dc = (a - Real.sqrt b) / c ∧
    a = 90 ∧ b = 1440 ∧ c = 3 ∧
    a + b + c = 1533 :=
by sorry

end dragon_castle_theorem_l3130_313025


namespace expand_product_l3130_313057

theorem expand_product (x : ℝ) : (x + 5) * (x + 7) = x^2 + 12*x + 35 := by
  sorry

end expand_product_l3130_313057


namespace sphere_volume_l3130_313065

theorem sphere_volume (r : ℝ) (h : r > 0) :
  (∃ (d : ℝ), d > 0 ∧ d < r ∧
    4 = (r^2 - d^2).sqrt ∧
    d = 3) →
  (4 / 3 * Real.pi * r^3 = 500 * Real.pi / 3) :=
by sorry

end sphere_volume_l3130_313065


namespace school_total_is_125_l3130_313067

/-- Represents the number of students in a school with specific age distribution. -/
structure School where
  /-- The number of students who are 8 years old -/
  eight_years : ℕ
  /-- The proportion of students below 8 years old -/
  below_eight_percent : ℚ
  /-- The ratio of students above 8 years old to students who are 8 years old -/
  above_eight_ratio : ℚ

/-- Calculates the total number of students in the school -/
def total_students (s : School) : ℕ :=
  sorry

/-- Theorem stating that for a school with given age distribution, 
    the total number of students is 125 -/
theorem school_total_is_125 (s : School) 
  (h1 : s.eight_years = 60)
  (h2 : s.below_eight_percent = 1/5)
  (h3 : s.above_eight_ratio = 2/3) : 
  total_students s = 125 := by
  sorry

end school_total_is_125_l3130_313067


namespace artist_june_pictures_l3130_313024

/-- The number of pictures painted in June -/
def june_pictures : ℕ := sorry

/-- The number of pictures painted in July -/
def july_pictures : ℕ := june_pictures + 2

/-- The number of pictures painted in August -/
def august_pictures : ℕ := 9

/-- The total number of pictures painted over the three months -/
def total_pictures : ℕ := 13

theorem artist_june_pictures :
  june_pictures = 1 ∧
  june_pictures + july_pictures + august_pictures = total_pictures :=
sorry

end artist_june_pictures_l3130_313024


namespace max_value_sine_function_l3130_313049

theorem max_value_sine_function (x : ℝ) (h : x ∈ Set.Icc 0 (π/4)) :
  (∃ (max_y : ℝ), max_y = Real.sqrt 3 ∧
    (∀ y : ℝ, y = Real.sqrt 3 * Real.sin (2*x + π/4) → y ≤ max_y) ∧
    max_y = Real.sqrt 3 * Real.sin (2*(π/8) + π/4)) :=
by sorry

end max_value_sine_function_l3130_313049


namespace min_fountains_correct_l3130_313079

/-- Represents a water fountain on a grid -/
structure Fountain where
  row : Nat
  col : Nat

/-- Checks if a fountain can spray a given square -/
def can_spray (f : Fountain) (row col : Nat) : Bool :=
  (f.row = row && (f.col = col - 1 || f.col = col + 1)) ||
  (f.col = col && (f.row = row - 1 || f.row = row + 1 || f.row = row - 2))

/-- Calculates the minimum number of fountains required for a given grid size -/
def min_fountains (m n : Nat) : Nat :=
  if m = 4 then
    2 * ((n + 2) / 3)
  else if m = 3 then
    3 * ((n + 2) / 3)
  else
    0  -- undefined for other cases

theorem min_fountains_correct (m n : Nat) :
  (m = 4 || m = 3) →
  ∃ (fountains : List Fountain),
    (fountains.length = min_fountains m n) ∧
    (∀ row col, row < m ∧ col < n →
      ∃ f ∈ fountains, can_spray f row col) :=
by sorry

#eval min_fountains 4 10  -- Expected: 8
#eval min_fountains 3 10  -- Expected: 12

end min_fountains_correct_l3130_313079


namespace smallest_solution_abs_equation_l3130_313056

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y ∧
  x = -2 := by
sorry

end smallest_solution_abs_equation_l3130_313056


namespace inverse_proportion_problem_l3130_313038

/-- Two real numbers are inversely proportional -/
def InverseProportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverseProportion x₁ y₁)
  (h2 : InverseProportion x₂ y₂)
  (h3 : x₁ = 40)
  (h4 : y₁ = 5)
  (h5 : y₂ = 10) :
  x₂ = 20 := by
  sorry

end inverse_proportion_problem_l3130_313038


namespace water_volume_calculation_l3130_313054

/-- The volume of water in a container can be calculated by multiplying the number of small hemisphere containers required to hold the water by the volume of each small hemisphere container. -/
theorem water_volume_calculation (num_containers : ℕ) (hemisphere_volume : ℝ) (total_volume : ℝ) : 
  num_containers = 2735 →
  hemisphere_volume = 4 →
  total_volume = num_containers * hemisphere_volume →
  total_volume = 10940 := by
sorry

end water_volume_calculation_l3130_313054


namespace ratio_problem_l3130_313050

theorem ratio_problem (x y : ℚ) (h : (3*x - 2*y) / (2*x + y) = 5/4) : x / y = 13/2 := by
  sorry

end ratio_problem_l3130_313050


namespace a_is_perfect_square_l3130_313087

-- Define the sequence c_n
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c n - 4 * c (n + 1) + 2008

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n ≥ 2 then
    5 * (c (n + 1) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501
  else
    0  -- Define a value for n < 2, though it's not used in the theorem

-- Theorem statement
theorem a_is_perfect_square (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, a n = k^2 := by
  sorry

end a_is_perfect_square_l3130_313087


namespace probability_at_least_one_karnataka_l3130_313070

theorem probability_at_least_one_karnataka (total_students : ℕ) 
  (maharashtra_students : ℕ) (karnataka_students : ℕ) (goa_students : ℕ) 
  (students_to_select : ℕ) : 
  total_students = 10 →
  maharashtra_students = 4 →
  karnataka_students = 3 →
  goa_students = 3 →
  students_to_select = 4 →
  (1 : ℚ) - (Nat.choose (total_students - karnataka_students) students_to_select : ℚ) / 
    (Nat.choose total_students students_to_select : ℚ) = 5 / 6 :=
by sorry

end probability_at_least_one_karnataka_l3130_313070


namespace total_balls_purchased_l3130_313060

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- The total number of golf balls purchased -/
def total_balls : ℕ := dan_dozens * balls_per_dozen + gus_dozens * balls_per_dozen + chris_balls

theorem total_balls_purchased :
  total_balls = 132 := by sorry

end total_balls_purchased_l3130_313060


namespace quadratic_inequality_and_minimum_l3130_313077

theorem quadratic_inequality_and_minimum (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → 
  (a > 3 ∧ ∃ m : ℝ, m = 7 ∧ ∀ x : ℝ, x > 3 → x + 9 / (x - 1) ≥ m) :=
by sorry

end quadratic_inequality_and_minimum_l3130_313077


namespace sum_of_repeating_decimals_l3130_313012

/-- The repeating decimal 0.overline{6} --/
def repeating_six : ℚ := 2/3

/-- The repeating decimal 0.overline{3} --/
def repeating_three : ℚ := 1/3

/-- The sum of 0.overline{6} and 0.overline{3} is equal to 1 --/
theorem sum_of_repeating_decimals : repeating_six + repeating_three = 1 := by
  sorry

end sum_of_repeating_decimals_l3130_313012


namespace carey_chairs_moved_l3130_313031

/-- Proves that Carey moved 28 chairs given the total chairs, Pat's chairs, and remaining chairs. -/
theorem carey_chairs_moved (total : ℕ) (pat_moved : ℕ) (remaining : ℕ) 
  (h1 : total = 74)
  (h2 : pat_moved = 29)
  (h3 : remaining = 17) :
  total - pat_moved - remaining = 28 := by
  sorry

#check carey_chairs_moved

end carey_chairs_moved_l3130_313031


namespace simple_interest_calculation_l3130_313018

/-- Given a principal amount with 5% interest rate for 2 years, 
    if the compound interest is 51.25, then the simple interest is 50 -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 50 := by
  sorry

end simple_interest_calculation_l3130_313018


namespace cubic_equation_roots_l3130_313016

theorem cubic_equation_roots (a b : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (1 - 2 * Complex.I : ℂ) ^ 3 + a * (1 - 2 * Complex.I : ℂ) ^ 2 - (1 - 2 * Complex.I : ℂ) + b = 0 →
  a = 1 ∧ b = 15 := by
sorry

end cubic_equation_roots_l3130_313016


namespace fishing_competition_l3130_313023

theorem fishing_competition (n : ℕ) : 
  (∃ (m : ℕ), n * m + 11 * (m + 10) = n^2 + 5*n + 22) → n = 11 :=
by sorry

end fishing_competition_l3130_313023


namespace binomial_expansion_coefficient_l3130_313053

theorem binomial_expansion_coefficient (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - m * x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  a₃ = 40 →
  m = -1 := by
sorry

end binomial_expansion_coefficient_l3130_313053


namespace exponential_equation_solution_l3130_313009

theorem exponential_equation_solution :
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧
  (∀ y : ℝ, (2 : ℝ) ^ (y^2 - 5*y - 6) = (4 : ℝ) ^ (y - 5) ↔ y = y₁ ∨ y = y₂) ∧
  y₁ + y₂ = 5 :=
sorry

end exponential_equation_solution_l3130_313009


namespace owls_on_fence_l3130_313063

theorem owls_on_fence (initial_owls joining_owls : ℕ) : 
  initial_owls = 3 → joining_owls = 2 → initial_owls + joining_owls = 5 :=
by sorry

end owls_on_fence_l3130_313063


namespace uniform_production_theorem_l3130_313084

def device_A_rate : ℚ := 1 / 90
def device_B_rate : ℚ := 1 / 60
def simultaneous_work_days : ℕ := 30
def remaining_days : ℕ := 13

theorem uniform_production_theorem :
  (∃ x : ℚ, x * (device_A_rate + device_B_rate) = 1 ∧ x = 36) ∧
  (∃ y : ℚ, (simultaneous_work_days + y) * device_A_rate + simultaneous_work_days * device_B_rate = 1 ∧ y > remaining_days) :=
by sorry

end uniform_production_theorem_l3130_313084


namespace subtract_abs_from_local_value_l3130_313068

def local_value (n : ℕ) (d : ℕ) : ℕ :=
  let digits := n.digits 10
  let index := digits.findIndex (· = d)
  10 ^ (digits.length - index - 1) * d

def absolute_value (n : ℤ) : ℕ := n.natAbs

theorem subtract_abs_from_local_value :
  local_value 564823 4 - absolute_value 4 = 39996 := by
  sorry

end subtract_abs_from_local_value_l3130_313068


namespace line_circle_intersection_l3130_313097

/-- If a line mx + ny = 0 intersects the circle (x+3)² + (y+1)² = 1 with a chord length of 2, then m/n = -1/3 -/
theorem line_circle_intersection (m n : ℝ) (h : m ≠ 0 ∧ n ≠ 0) :
  (∀ x y : ℝ, m * x + n * y = 0 →
    ((x + 3)^2 + (y + 1)^2 = 1 →
      ∃ x₁ y₁ x₂ y₂ : ℝ,
        m * x₁ + n * y₁ = 0 ∧
        (x₁ + 3)^2 + (y₁ + 1)^2 = 1 ∧
        m * x₂ + n * y₂ = 0 ∧
        (x₂ + 3)^2 + (y₂ + 1)^2 = 1 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  m / n = -1/3 :=
by sorry

end line_circle_intersection_l3130_313097


namespace donut_distribution_proof_l3130_313032

/-- The number of ways to distribute donuts satisfying the given conditions -/
def donut_combinations : ℕ := 126

/-- The number of donut types -/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased -/
def total_donuts : ℕ := 10

/-- The number of remaining donuts after selecting one of each type -/
def remaining_donuts : ℕ := total_donuts - num_types

/-- Binomial coefficient calculation -/
def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range k).foldl (fun m i => m * (n - i) / (i + 1)) 1

theorem donut_distribution_proof :
  binom (remaining_donuts + num_types - 1) (num_types - 1) = donut_combinations :=
by sorry

end donut_distribution_proof_l3130_313032


namespace problem_solution_l3130_313089

theorem problem_solution (x y : ℚ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 7 / 8) :
  y - x = 100 / 3 := by
  sorry

end problem_solution_l3130_313089


namespace dvd_cost_l3130_313091

/-- The cost of each DVD given the total number of movies, trade-in value per VHS, and total replacement cost. -/
theorem dvd_cost (total_movies : ℕ) (vhs_trade_value : ℚ) (total_replacement_cost : ℚ) :
  total_movies = 100 →
  vhs_trade_value = 2 →
  total_replacement_cost = 800 →
  (total_replacement_cost - (total_movies : ℚ) * vhs_trade_value) / total_movies = 6 := by
  sorry

end dvd_cost_l3130_313091


namespace arithmetic_sequence_formula_l3130_313002

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 has the general formula 2n - 3 for its nth term. -/
theorem arithmetic_sequence_formula (a : ℝ) : 
  (∃ (seq : ℕ → ℝ), 
    seq 1 = a - 1 ∧ 
    seq 2 = a + 1 ∧ 
    seq 3 = 2*a + 3 ∧ 
    ∀ n m : ℕ, seq (n + 1) - seq n = seq (m + 1) - seq m) → 
  (∃ (seq : ℕ → ℝ), 
    seq 1 = a - 1 ∧ 
    seq 2 = a + 1 ∧ 
    seq 3 = 2*a + 3 ∧ 
    ∀ n : ℕ, seq n = 2*n - 3) :=
by sorry

end arithmetic_sequence_formula_l3130_313002


namespace zahs_to_bahs_conversion_l3130_313085

/-- Conversion rates between different currencies -/
structure CurrencyRates where
  bah_to_rah : ℚ
  rah_to_yah : ℚ
  yah_to_zah : ℚ

/-- Given conversion rates, calculate the number of bahs equivalent to a given number of zahs -/
def zahs_to_bahs (rates : CurrencyRates) (zahs : ℚ) : ℚ :=
  zahs / rates.yah_to_zah / rates.rah_to_yah / rates.bah_to_rah

/-- Theorem stating the equivalence between 1500 zahs and 400/3 bahs -/
theorem zahs_to_bahs_conversion (rates : CurrencyRates) 
  (h1 : rates.bah_to_rah = 3)
  (h2 : rates.rah_to_yah = 3/2)
  (h3 : rates.yah_to_zah = 5/2) : 
  zahs_to_bahs rates 1500 = 400/3 := by
  sorry

#eval zahs_to_bahs ⟨3, 3/2, 5/2⟩ 1500

end zahs_to_bahs_conversion_l3130_313085


namespace odd_product_probability_l3130_313075

theorem odd_product_probability (n : ℕ) (hn : n = 1000) :
  let odd_count := (n + 1) / 2
  let total_count := n
  let p := (odd_count / total_count) * ((odd_count - 1) / (total_count - 1)) * ((odd_count - 2) / (total_count - 2))
  p < 1 / 8 := by
sorry


end odd_product_probability_l3130_313075


namespace unique_four_digit_numbers_l3130_313010

theorem unique_four_digit_numbers : ∃! (x y : ℕ), 
  (1000 ≤ x ∧ x < 10000) ∧ 
  (1000 ≤ y ∧ y < 10000) ∧ 
  y > x ∧ 
  (∃ (a n : ℕ), 1 ≤ a ∧ a < 10 ∧ y = a * 10^n) ∧
  (x / 1000 + (x / 100) % 10 = y - x) ∧
  (y - x = 5 * (y / 1000)) ∧
  x = 1990 ∧ 
  y = 2000 := by
sorry

end unique_four_digit_numbers_l3130_313010


namespace cubic_factorization_l3130_313013

theorem cubic_factorization :
  ∀ x : ℝ, 343 * x^3 + 125 = (7 * x + 5) * (49 * x^2 - 35 * x + 25) := by
  sorry

end cubic_factorization_l3130_313013


namespace max_diagonal_bd_l3130_313005

/-- Represents the side lengths of a quadrilateral --/
structure QuadrilateralSides where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ

/-- Checks if the given side lengths form a valid cyclic quadrilateral --/
def is_valid_cyclic_quadrilateral (sides : QuadrilateralSides) : Prop :=
  sides.AB < 10 ∧ sides.BC < 10 ∧ sides.CD < 10 ∧ sides.DA < 10 ∧
  sides.AB ≠ sides.BC ∧ sides.AB ≠ sides.CD ∧ sides.AB ≠ sides.DA ∧
  sides.BC ≠ sides.CD ∧ sides.BC ≠ sides.DA ∧ sides.CD ≠ sides.DA ∧
  sides.BC + sides.CD = sides.AB + sides.DA

/-- Calculates the square of the diagonal BD --/
def diagonal_bd_squared (sides : QuadrilateralSides) : ℚ :=
  (sides.AB^2 + sides.BC^2 + sides.CD^2 + sides.DA^2) / 2

theorem max_diagonal_bd (sides : QuadrilateralSides) :
  is_valid_cyclic_quadrilateral sides →
  diagonal_bd_squared sides ≤ 191/2 :=
sorry

end max_diagonal_bd_l3130_313005


namespace h_of_neg_one_eq_three_l3130_313037

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_neg_one_eq_three : h (-1) = 3 := by sorry

end h_of_neg_one_eq_three_l3130_313037


namespace oliver_spending_l3130_313042

theorem oliver_spending (initial_amount spent_amount received_amount final_amount : ℕ) :
  initial_amount = 33 →
  received_amount = 32 →
  final_amount = 61 →
  final_amount = initial_amount - spent_amount + received_amount →
  spent_amount = 4 := by
sorry

end oliver_spending_l3130_313042


namespace triangle_max_area_l3130_313093

theorem triangle_max_area (A B C : Real) (a b c : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * c = 6 →
  Real.sin B + 2 * Real.sin C * Real.cos A = 0 →
  (∀ S : Real, S = (1/2) * a * c * Real.sin B → S ≤ 3/2) ∧
  (∃ S : Real, S = (1/2) * a * c * Real.sin B ∧ S = 3/2) :=
by sorry

end triangle_max_area_l3130_313093


namespace cylinder_surface_area_l3130_313043

/-- The surface area of a cylinder with a square cross-section of side length 2 is 6π. -/
theorem cylinder_surface_area (π : ℝ) (h : π = Real.pi) : 
  let side_length : ℝ := 2
  let radius : ℝ := side_length / 2
  let height : ℝ := side_length
  let lateral_area : ℝ := 2 * π * radius * height
  let base_area : ℝ := 2 * π * radius^2
  lateral_area + base_area = 6 * π :=
by
  sorry

end cylinder_surface_area_l3130_313043


namespace fractional_equation_solution_l3130_313080

theorem fractional_equation_solution :
  ∃ x : ℝ, (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 :=
by
  sorry

end fractional_equation_solution_l3130_313080


namespace sqrt_3_minus_3_power_0_minus_2_power_neg_1_l3130_313069

theorem sqrt_3_minus_3_power_0_minus_2_power_neg_1 :
  (Real.sqrt 3 - 3) ^ 0 - 2 ^ (-1 : ℤ) = 1/2 := by sorry

end sqrt_3_minus_3_power_0_minus_2_power_neg_1_l3130_313069


namespace mary_chewing_gums_l3130_313073

theorem mary_chewing_gums (total sam sue : ℕ) (h1 : total = 30) (h2 : sam = 10) (h3 : sue = 15) :
  total - (sam + sue) = 5 := by
  sorry

end mary_chewing_gums_l3130_313073


namespace matrix_equation_l3130_313096

-- Define the matrices
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 12, 5]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -28/7, 35/7]

-- State the theorem
theorem matrix_equation : N * A = B := by sorry

end matrix_equation_l3130_313096


namespace binomial_max_term_max_term_sqrt_seven_l3130_313059

theorem binomial_max_term (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
sorry

theorem max_term_sqrt_seven :
  let n := 205
  let x := Real.sqrt 7
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j ∧
    k = 149 :=
sorry

end binomial_max_term_max_term_sqrt_seven_l3130_313059


namespace parabola_x_axis_intersection_l3130_313004

theorem parabola_x_axis_intersection :
  let f (x : ℝ) := x^2 - 2*x - 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end parabola_x_axis_intersection_l3130_313004


namespace cubic_root_sum_cubes_l3130_313048

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  (a^3 + b^3 + c^3 = 2) := by
  sorry

end cubic_root_sum_cubes_l3130_313048


namespace diagonal_increase_l3130_313092

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the number of diagonals
    in a convex polygon with n sides and n+1 sides -/
theorem diagonal_increase (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n - 1 := by sorry

end diagonal_increase_l3130_313092


namespace min_value_of_exponential_sum_l3130_313086

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y - 1 = 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2^x + 8^y ≥ z → z ≥ m :=
sorry

end min_value_of_exponential_sum_l3130_313086


namespace complex_equation_implies_unit_magnitude_l3130_313008

theorem complex_equation_implies_unit_magnitude (z : ℂ) : 
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 → Complex.abs z = 1 := by
  sorry

end complex_equation_implies_unit_magnitude_l3130_313008


namespace tom_apple_count_l3130_313000

/-- The number of apples each person has -/
structure AppleCount where
  phillip : ℕ
  ben : ℕ
  tom : ℕ

/-- The conditions of the problem -/
def problem_conditions (ac : AppleCount) : Prop :=
  ac.phillip = 40 ∧
  ac.ben = ac.phillip + 8 ∧
  ac.tom = (3 * ac.ben) / 8

/-- The theorem stating that Tom has 18 apples given the problem conditions -/
theorem tom_apple_count (ac : AppleCount) (h : problem_conditions ac) : ac.tom = 18 := by
  sorry

#check tom_apple_count

end tom_apple_count_l3130_313000


namespace square_root_value_l3130_313094

theorem square_root_value (x : ℝ) (h : x = 5) : Real.sqrt (x - 3) = Real.sqrt 2 := by
  sorry

end square_root_value_l3130_313094


namespace general_admission_tickets_l3130_313082

theorem general_admission_tickets (student_price general_price total_tickets total_money : ℕ) 
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_tickets = 525)
  (h4 : total_money = 2876) :
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_tickets * student_price + general_tickets * general_price = total_money ∧
    general_tickets = 388 := by
  sorry

end general_admission_tickets_l3130_313082


namespace walking_speed_l3130_313017

theorem walking_speed (x : ℝ) : 
  let tom_speed := x^2 - 14*x - 48
  let jerry_distance := x^2 - 5*x - 84
  let jerry_time := x + 8
  let jerry_speed := jerry_distance / jerry_time
  x ≠ -8 → tom_speed = jerry_speed → tom_speed = 6 :=
by sorry

end walking_speed_l3130_313017


namespace certain_number_equals_sixteen_l3130_313006

theorem certain_number_equals_sixteen : ∃ x : ℝ, x^5 = 4^10 ∧ x = 16 := by
  sorry

end certain_number_equals_sixteen_l3130_313006


namespace supplementary_angles_equal_l3130_313041

/-- Two angles that are supplementary to the same angle are equal. -/
theorem supplementary_angles_equal (α β γ : Real) (h1 : α + γ = 180) (h2 : β + γ = 180) : α = β := by
  sorry

end supplementary_angles_equal_l3130_313041


namespace ladder_problem_l3130_313081

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : ℝ, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
sorry

end ladder_problem_l3130_313081


namespace pool_radius_l3130_313022

/-- Proves that a circular pool with a surrounding concrete wall has a radius of 20 feet
    given specific conditions on the wall width and area ratio. -/
theorem pool_radius (r : ℝ) : 
  r > 0 → -- The radius is positive
  (π * ((r + 4)^2 - r^2) = (11/25) * π * r^2) → -- Area ratio condition
  r = 20 := by
  sorry

end pool_radius_l3130_313022


namespace hyperbola_eccentricity_l3130_313047

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∀ x y, (x - c)^2 + y^2 = 4 * a^2) →  -- Circle equation
  (c^2 = a^2 * (1 + b^2 / a^2)) →  -- Semi-latus rectum condition
  (∃ x y, (b * x + a * y = 0) ∧ (x - c)^2 + y^2 = 4 * a^2 ∧ 
    ∃ x' y', (b * x' + a * y' = 0) ∧ (x' - c)^2 + y'^2 = 4 * a^2 ∧
    (x - x')^2 + (y - y')^2 = 4 * b^2) →  -- Asymptote intercepted by circle
  (c^2 / a^2 - 1)^(1/2) = Real.sqrt 3 :=  -- Eccentricity equals sqrt(3)
by sorry

end hyperbola_eccentricity_l3130_313047


namespace kevin_wins_l3130_313026

/-- Represents a player in the chess game -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kevin : Player

/-- Represents the game results for each player -/
structure GameResults :=
  (wins : Player → ℕ)
  (losses : Player → ℕ)

/-- The theorem to prove -/
theorem kevin_wins (results : GameResults) : 
  results.wins Player.Peter = 4 →
  results.losses Player.Peter = 2 →
  results.wins Player.Emma = 3 →
  results.losses Player.Emma = 3 →
  results.losses Player.Kevin = 3 →
  results.wins Player.Kevin = 1 := by
  sorry


end kevin_wins_l3130_313026


namespace largest_divisor_of_n4_minus_n2_l3130_313007

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  ∃ (d : ℕ), d > 0 ∧ (∀ (m : ℤ), (m^4 - m^2) % d = 0) ∧ 
  (∀ (k : ℕ), k > d → ∃ (l : ℤ), (l^4 - l^2) % k ≠ 0) ∧ d = 6 :=
sorry

end largest_divisor_of_n4_minus_n2_l3130_313007


namespace tractor_financing_term_l3130_313095

/-- Calculates the financing term in years given the monthly payment and total financed amount. -/
def financing_term_years (monthly_payment : ℚ) (total_amount : ℚ) : ℚ :=
  (total_amount / monthly_payment) / 12

/-- Theorem stating that the financing term for the given conditions is 5 years. -/
theorem tractor_financing_term :
  let monthly_payment : ℚ := 150
  let total_amount : ℚ := 9000
  financing_term_years monthly_payment total_amount = 5 := by
  sorry

end tractor_financing_term_l3130_313095


namespace tank_filling_l3130_313090

theorem tank_filling (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2 / 5 →
  ∃ (new_buckets : ℕ), 
    (↑new_buckets : ℚ) > (↑original_buckets / capacity_ratio) ∧ 
    (↑new_buckets : ℚ) ≤ (↑original_buckets / capacity_ratio + 1) ∧
    new_buckets = 63 :=
by
  sorry

end tank_filling_l3130_313090


namespace cookie_radius_l3130_313066

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 8

/-- The cookie is a circle -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ x y, cookie_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

/-- The radius of the cookie is √13 -/
theorem cookie_radius :
  ∃ center, is_circle center (Real.sqrt 13) :=
sorry

end cookie_radius_l3130_313066


namespace spider_plant_production_l3130_313083

/-- Represents the number of baby plants produced by a spider plant over time -/
def babyPlants (plantsPerProduction : ℕ) (productionsPerYear : ℕ) (years : ℕ) : ℕ :=
  plantsPerProduction * productionsPerYear * years

/-- Theorem: A spider plant producing 2 baby plants 2 times a year will produce 16 baby plants after 4 years -/
theorem spider_plant_production :
  babyPlants 2 2 4 = 16 := by
  sorry

end spider_plant_production_l3130_313083


namespace num_adoption_ways_l3130_313020

/-- The number of parrots available for adoption -/
def num_parrots : ℕ := 20

/-- The number of snakes available for adoption -/
def num_snakes : ℕ := 10

/-- The number of rabbits available for adoption -/
def num_rabbits : ℕ := 12

/-- The set of possible animal types -/
inductive AnimalType
| Parrot
| Snake
| Rabbit

/-- A function representing Emily's constraint -/
def emily_constraint (a : AnimalType) : Prop :=
  a = AnimalType.Parrot ∨ a = AnimalType.Rabbit

/-- A function representing John's constraint (can adopt any animal) -/
def john_constraint (a : AnimalType) : Prop := True

/-- A function representing Susan's constraint -/
def susan_constraint (a : AnimalType) : Prop :=
  a = AnimalType.Snake

/-- The theorem stating the number of ways to adopt animals -/
theorem num_adoption_ways :
  (num_parrots * num_snakes * num_rabbits) +
  (num_rabbits * num_snakes * num_parrots) = 4800 := by
  sorry

end num_adoption_ways_l3130_313020


namespace sphere_volume_l3130_313051

theorem sphere_volume (r : ℝ) (h1 : r > 0) (h2 : π = (r ^ 2 - 1 ^ 2)) : 
  (4 / 3 : ℝ) * π * r ^ 3 = (8 * Real.sqrt 2 / 3 : ℝ) * π := by
  sorry

end sphere_volume_l3130_313051


namespace sum_first_5_even_numbers_is_30_l3130_313088

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.map (fun i => 2 * i) (List.range n)

theorem sum_first_5_even_numbers_is_30 :
  List.sum (first_n_even_numbers 5) = 30 :=
by
  sorry

#check sum_first_5_even_numbers_is_30

end sum_first_5_even_numbers_is_30_l3130_313088


namespace percentage_of_x_minus_y_l3130_313036

theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (30 / 100) * (x + y) →
  y = (25 / 100) * x →
  P = 50 := by
  sorry

end percentage_of_x_minus_y_l3130_313036


namespace cheese_pizzas_sold_l3130_313019

/-- The number of cheese pizzas sold by a pizza store on Friday -/
def cheese_pizzas (pepperoni bacon total : ℕ) : ℕ :=
  total - (pepperoni + bacon)

/-- Theorem stating the number of cheese pizzas sold -/
theorem cheese_pizzas_sold :
  cheese_pizzas 2 6 14 = 6 := by
  sorry

end cheese_pizzas_sold_l3130_313019


namespace average_carnations_example_l3130_313098

/-- The average number of carnations in three bouquets -/
def average_carnations (b1 b2 b3 : ℕ) : ℚ :=
  (b1 + b2 + b3 : ℚ) / 3

/-- Theorem: The average number of carnations in three bouquets containing 9, 14, and 13 carnations respectively is 12 -/
theorem average_carnations_example : average_carnations 9 14 13 = 12 := by
  sorry

end average_carnations_example_l3130_313098


namespace language_group_selection_ways_l3130_313055

/-- Represents a group of people who know languages -/
structure LanguageGroup where
  total : Nat
  english : Nat
  japanese : Nat
  both : Nat

/-- The number of ways to select one person who knows English and another who knows Japanese -/
def selectWays (group : LanguageGroup) : Nat :=
  (group.english - group.both) * group.japanese + group.both * (group.japanese - 1)

/-- Theorem stating the number of ways to select people in the given scenario -/
theorem language_group_selection_ways :
  ∃ (group : LanguageGroup),
    group.total = 9 ∧
    group.english = 7 ∧
    group.japanese = 3 ∧
    group.total = (group.english - group.both) + (group.japanese - group.both) + group.both ∧
    selectWays group = 20 := by
  sorry

end language_group_selection_ways_l3130_313055


namespace new_salary_after_raise_l3130_313045

def original_salary : ℝ := 500
def raise_percentage : ℝ := 6

theorem new_salary_after_raise :
  original_salary * (1 + raise_percentage / 100) = 530 := by
  sorry

end new_salary_after_raise_l3130_313045


namespace remainder_division_l3130_313001

theorem remainder_division (y k : ℤ) (h : y = 264 * k + 42) : y ≡ 20 [ZMOD 22] := by
  sorry

end remainder_division_l3130_313001


namespace soccer_camp_afternoon_attendance_l3130_313099

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : total_kids % 2 = 0)
  (soccer_kids : ℕ) 
  (h3 : soccer_kids = total_kids / 2)
  (morning_ratio : ℚ)
  (h4 : morning_ratio = 1 / 4)
  (morning_kids : ℕ)
  (h5 : morning_kids = ⌊soccer_kids * morning_ratio⌋)
  (afternoon_kids : ℕ)
  (h6 : afternoon_kids = soccer_kids - morning_kids) :
  afternoon_kids = 750 := by
sorry

end soccer_camp_afternoon_attendance_l3130_313099


namespace coefficient_a6_l3130_313003

theorem coefficient_a6 (x a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  x^2 + x^7 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 →
  a₆ = -7 := by
  sorry

end coefficient_a6_l3130_313003


namespace seventeen_above_zero_l3130_313044

/-- Represents temperature in degrees Celsius -/
structure Temperature where
  value : ℝ
  unit : String
  is_celsius : unit = "°C"

/-- The zero point of the Celsius scale -/
def celsius_zero : Temperature := ⟨10, "°C", rfl⟩

/-- The temperature to be compared -/
def temp_to_compare : Temperature := ⟨17, "°C", rfl⟩

/-- Theorem stating that 17°C represents a temperature above zero degrees Celsius -/
theorem seventeen_above_zero :
  temp_to_compare.value > celsius_zero.value → 
  ∃ (t : ℝ), t > 0 ∧ temp_to_compare.value = t :=
by sorry

end seventeen_above_zero_l3130_313044


namespace quadratic_equation_roots_l3130_313074

theorem quadratic_equation_roots (b c : ℝ) : 
  (∀ x, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) → 
  b = -1 ∧ c = -2 := by
sorry

end quadratic_equation_roots_l3130_313074


namespace square_area_problem_l3130_313011

theorem square_area_problem (small_square_area : ℝ) (triangle_area : ℝ) :
  small_square_area = 16 →
  triangle_area = 1 →
  ∃ (large_square_area : ℝ),
    large_square_area = 18 ∧
    ∃ (small_side large_side triangle_side : ℝ),
      small_side ^ 2 = small_square_area ∧
      triangle_side ^ 2 = 2 ∧
      large_side ^ 2 = large_square_area ∧
      large_side ^ 2 = small_side ^ 2 + triangle_side ^ 2 := by
  sorry


end square_area_problem_l3130_313011


namespace exists_valid_grid_l3130_313064

/-- Represents a 3x3 grid with numbers -/
structure Grid :=
  (top_left top_right bottom_left bottom_right : ℕ)

/-- The sum of numbers along each side of the grid is 13 -/
def valid_sum (g : Grid) : Prop :=
  g.top_left + 4 + g.top_right = 13 ∧
  g.top_right + 2 + g.bottom_right = 13 ∧
  g.bottom_right + 1 + g.bottom_left = 13 ∧
  g.bottom_left + 3 + g.top_left = 13

/-- There exists a valid grid arrangement -/
theorem exists_valid_grid : ∃ (g : Grid), valid_sum g :=
sorry

end exists_valid_grid_l3130_313064


namespace division_result_and_thousandths_digit_l3130_313061

theorem division_result_and_thousandths_digit : 
  let result : ℚ := 57 / 5000
  (result = 0.0114) ∧ 
  (⌊result * 1000⌋ % 10 = 4) := by
  sorry

end division_result_and_thousandths_digit_l3130_313061


namespace prime_divides_binomial_l3130_313035

theorem prime_divides_binomial (n k : ℕ) (h_prime : Nat.Prime n) (h_k_pos : 0 < k) (h_k_lt_n : k < n) :
  n ∣ Nat.choose n k := by
  sorry

end prime_divides_binomial_l3130_313035


namespace cos_arcsin_half_l3130_313058

theorem cos_arcsin_half : Real.cos (Real.arcsin (1/2)) = Real.sqrt 3 / 2 := by
  sorry

end cos_arcsin_half_l3130_313058


namespace smallest_value_of_y_l3130_313062

theorem smallest_value_of_y (x : ℝ) : 
  (17 - x) * (19 - x) * (19 + x) * (17 + x) ≥ -1296 ∧ 
  ∃ x : ℝ, (17 - x) * (19 - x) * (19 + x) * (17 + x) = -1296 :=
by sorry

end smallest_value_of_y_l3130_313062


namespace arithmetic_sequence_fifth_term_l3130_313029

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end arithmetic_sequence_fifth_term_l3130_313029


namespace abc_inequality_l3130_313033

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eq_a : a - 2 = Real.log (a / 2))
  (eq_b : b - 3 = Real.log (b / 3))
  (eq_c : c - 3 = Real.log (c / 2)) :
  c < b ∧ b < a := by
  sorry

end abc_inequality_l3130_313033


namespace expression_value_l3130_313052

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -2) : 
  -a - b^2 + a*b + a^2 = 0 := by sorry

end expression_value_l3130_313052


namespace smallest_sum_in_S_l3130_313015

def S : Set ℚ := {2, 0, -1, -3}

theorem smallest_sum_in_S : 
  ∃ (x y : ℚ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ 
  (∀ (a b : ℚ), a ∈ S → b ∈ S → a ≠ b → x + y ≤ a + b) ∧
  x + y = -4 :=
sorry

end smallest_sum_in_S_l3130_313015


namespace kanul_cash_theorem_l3130_313034

/-- Represents the total amount of cash Kanul had -/
def T : ℝ := sorry

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 5000

/-- The amount spent on machinery -/
def machinery : ℝ := 200

/-- The percentage of total cash spent -/
def percentage_spent : ℝ := 0.30

theorem kanul_cash_theorem : 
  T = (raw_materials + machinery) / (1 - percentage_spent) :=
by sorry

end kanul_cash_theorem_l3130_313034
