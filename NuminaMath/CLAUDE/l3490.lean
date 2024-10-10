import Mathlib

namespace wage_increase_result_l3490_349016

/-- Calculates the new wage after a percentage increase -/
def new_wage (original_wage : ℝ) (percent_increase : ℝ) : ℝ :=
  original_wage * (1 + percent_increase)

/-- Theorem stating that a 50% increase on a $28 wage results in $42 -/
theorem wage_increase_result :
  new_wage 28 0.5 = 42 := by
  sorry

end wage_increase_result_l3490_349016


namespace alcohol_water_ratio_in_combined_mixture_l3490_349029

/-- Given two containers A and B with alcohol mixtures, this theorem proves
    the ratio of pure alcohol to water in the combined mixture. -/
theorem alcohol_water_ratio_in_combined_mixture
  (v₁ v₂ m₁ n₁ m₂ n₂ : ℝ)
  (hv₁ : v₁ > 0)
  (hv₂ : v₂ > 0)
  (hm₁ : m₁ > 0)
  (hn₁ : n₁ > 0)
  (hm₂ : m₂ > 0)
  (hn₂ : n₂ > 0) :
  let pure_alcohol_A := v₁ * m₁ / (m₁ + n₁)
  let water_A := v₁ * n₁ / (m₁ + n₁)
  let pure_alcohol_B := v₂ * m₂ / (m₂ + n₂)
  let water_B := v₂ * n₂ / (m₂ + n₂)
  let total_pure_alcohol := pure_alcohol_A + pure_alcohol_B
  let total_water := water_A + water_B
  (total_pure_alcohol / total_water) = 
    (v₁*m₁*m₂ + v₁*m₁*n₂ + v₂*m₁*m₂ + v₂*m₂*n₁) / 
    (v₁*m₂*n₁ + v₁*n₁*n₂ + v₂*m₁*n₂ + v₂*n₁*n₂) :=
by sorry

end alcohol_water_ratio_in_combined_mixture_l3490_349029


namespace angle_approximation_l3490_349013

/-- Regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  center : ℝ × ℝ
  radius : ℝ
  vertices : Fin n → ℝ × ℝ

/-- Construct points B, C, D, E as described in the problem -/
def constructPoints (p : RegularPolygon 19) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Length of chord DE -/
def chordLength (p : RegularPolygon 19) : ℝ := sorry

/-- Angle formed by radii after 19 sequential measurements -/
def angleAfterMeasurements (p : RegularPolygon 19) : ℝ := sorry

/-- Main theorem: The angle formed after 19 measurements is approximately 4°57' -/
theorem angle_approximation (p : RegularPolygon 19) : 
  ∃ ε > 0, abs (angleAfterMeasurements p - (4 + 57/60) * π / 180) < ε :=
sorry

end angle_approximation_l3490_349013


namespace rectangle_area_diagonal_l3490_349061

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end rectangle_area_diagonal_l3490_349061


namespace three_dice_probability_l3490_349063

theorem three_dice_probability : 
  let dice := 6
  let prob_first := (3 : ℚ) / dice  -- Probability of rolling less than 4 on first die
  let prob_second := (3 : ℚ) / dice -- Probability of rolling an even number on second die
  let prob_third := (2 : ℚ) / dice  -- Probability of rolling greater than 4 on third die
  prob_first * prob_second * prob_third = 1 / 12 := by
sorry

end three_dice_probability_l3490_349063


namespace two_digit_numbers_product_gcd_l3490_349003

theorem two_digit_numbers_product_gcd (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 1728 ∧ 
  Nat.gcd a b = 12 →
  (a = 36 ∧ b = 48) ∨ (a = 48 ∧ b = 36) := by
sorry

end two_digit_numbers_product_gcd_l3490_349003


namespace repair_time_for_14_people_l3490_349050

/-- Represents the time needed for a given number of people to repair the dam -/
structure RepairTime where
  people : ℕ
  minutes : ℕ

/-- The dam repair scenario -/
structure DamRepair where
  repair1 : RepairTime
  repair2 : RepairTime

/-- Calculates the time needed for a given number of people to repair the dam -/
def calculateRepairTime (d : DamRepair) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 14 people need 30 minutes to repair the dam -/
theorem repair_time_for_14_people (d : DamRepair) 
  (h1 : d.repair1 = ⟨10, 45⟩) 
  (h2 : d.repair2 = ⟨20, 20⟩) : 
  calculateRepairTime d 14 = 30 :=
sorry

end repair_time_for_14_people_l3490_349050


namespace rectangular_prism_diagonals_l3490_349034

/-- A rectangular prism with different dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of face diagonals in a rectangular prism -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 ∧ space_diagonals prism = 4 := by
  sorry

end rectangular_prism_diagonals_l3490_349034


namespace inequality_proof_l3490_349039

def M := {x : ℝ | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end inequality_proof_l3490_349039


namespace omar_egg_rolls_l3490_349012

theorem omar_egg_rolls (karen_rolls : ℕ) (total_rolls : ℕ) (omar_rolls : ℕ) : 
  karen_rolls = 229 → total_rolls = 448 → omar_rolls = total_rolls - karen_rolls → omar_rolls = 219 := by
  sorry

end omar_egg_rolls_l3490_349012


namespace min_voters_for_tall_giraffe_win_l3490_349060

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters needed to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The giraffe beauty contest voting structure -/
def giraffe_contest : VotingStructure :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

/-- Theorem stating the minimum number of voters needed for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win :
  min_voters_to_win giraffe_contest = 30 := by
  sorry

#eval min_voters_to_win giraffe_contest

end min_voters_for_tall_giraffe_win_l3490_349060


namespace min_distance_between_curves_l3490_349083

/-- The minimum distance between two points P and Q, where P lies on the curve y = x^2 - ln(x) 
    and Q lies on the line y = x - 2, and both P and Q have the same y-coordinate, is 2. -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → 
    let y₁ := x₁^2 - Real.log x₁
    let y₂ := x₂ - 2
    y₁ = y₂ → |x₂ - x₁| ≥ min_dist) ∧
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ 
    let y₁ := x₁^2 - Real.log x₁
    let y₂ := x₂ - 2
    y₁ = y₂ ∧ |x₂ - x₁| = min_dist) ∧
  min_dist = 2 := by
  sorry

end min_distance_between_curves_l3490_349083


namespace first_day_exceeding_200_chocolates_l3490_349091

def chocolate_count (n : ℕ) : ℕ := 3 * 3^(n - 1)

theorem first_day_exceeding_200_chocolates :
  (∃ n : ℕ, n > 0 ∧ chocolate_count n > 200) ∧
  (∀ m : ℕ, m > 0 ∧ m < 6 → chocolate_count m ≤ 200) ∧
  chocolate_count 6 > 200 :=
sorry

end first_day_exceeding_200_chocolates_l3490_349091


namespace f_derivative_at_zero_l3490_349049

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 2 * x^2 + x^2 * Real.cos (1 / x) else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by
  sorry

end f_derivative_at_zero_l3490_349049


namespace last_two_digits_square_l3490_349073

theorem last_two_digits_square (n : ℕ) : 
  (n % 100 = n^2 % 100) ↔ (n % 100 = 0 ∨ n % 100 = 1 ∨ n % 100 = 25 ∨ n % 100 = 76) := by
  sorry

end last_two_digits_square_l3490_349073


namespace unique_solution_m_value_l3490_349032

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := 16 * x^2 + m * x + 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := m^2 - 4 * 16 * 4

-- Theorem statement
theorem unique_solution_m_value :
  ∃! m : ℝ, m > 0 ∧ (∃! x : ℝ, quadratic_equation m x = 0) :=
by sorry

end unique_solution_m_value_l3490_349032


namespace roots_sum_powers_l3490_349021

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → b^2 - 5*b + 6 = 0 → a^5 + a^4*b + b^5 = -16674 := by
  sorry

end roots_sum_powers_l3490_349021


namespace product_is_rational_l3490_349048

def primes : List Nat := [3, 5, 7, 11, 13, 17]

def product : ℚ :=
  primes.foldl (fun acc p => acc * (1 - 1 / (p * p : ℚ))) 1

theorem product_is_rational : ∃ (a b : ℕ), product = a / b :=
  sorry

end product_is_rational_l3490_349048


namespace stella_profit_is_25_l3490_349004

/-- Represents the profit Stella makes from her antique shop sales -/
def stellas_profit (num_dolls num_clocks num_glasses : ℕ) 
                   (price_doll price_clock price_glass : ℚ) 
                   (cost : ℚ) : ℚ :=
  num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass - cost

/-- Theorem stating that Stella's profit is $25 given the specified conditions -/
theorem stella_profit_is_25 : 
  stellas_profit 3 2 5 5 15 4 40 = 25 := by
  sorry

end stella_profit_is_25_l3490_349004


namespace problem_solution_l3490_349046

noncomputable section

def f (x : ℝ) := 3 - 2 * Real.log x / Real.log 2
def g (x : ℝ) := Real.log x / Real.log 2
def h (x : ℝ) := (f x + 1) * g x
def M (x : ℝ) := max (g x) (f x)

theorem problem_solution :
  (∀ x ∈ Set.Icc 1 8, h x ∈ Set.Icc (-6) 2) ∧
  (∀ x > 0, M x ≤ 1) ∧
  (∃ x > 0, M x = 1) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc 1 8, f (x^2) * f (Real.sqrt x) ≥ k * g x) → k ≤ -3) :=
sorry

end

end problem_solution_l3490_349046


namespace max_value_of_x_plus_reciprocal_l3490_349028

theorem max_value_of_x_plus_reciprocal (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end max_value_of_x_plus_reciprocal_l3490_349028


namespace geometric_sequence_first_term_l3490_349087

/-- A geometric sequence with fifth term 243 and sixth term 729 has first term 3 -/
theorem geometric_sequence_first_term : ∀ (a : ℝ) (r : ℝ),
  a * r^4 = 243 →
  a * r^5 = 729 →
  a = 3 := by
sorry

end geometric_sequence_first_term_l3490_349087


namespace candy_bar_difference_l3490_349076

theorem candy_bar_difference : 
  ∀ (bob_candy : ℕ),
  let fred_candy : ℕ := 12
  let total_candy : ℕ := fred_candy + bob_candy
  let jacqueline_candy : ℕ := 10 * total_candy
  120 = (40 : ℕ) * jacqueline_candy / 100 →
  bob_candy - fred_candy = 6 := by
sorry

end candy_bar_difference_l3490_349076


namespace percentage_of_juniors_l3490_349010

theorem percentage_of_juniors (total_students : ℕ) (juniors_in_sports : ℕ) 
  (sports_percentage : ℚ) (h1 : total_students = 500) 
  (h2 : juniors_in_sports = 140) (h3 : sports_percentage = 70 / 100) :
  (juniors_in_sports / sports_percentage) / total_students = 40 / 100 := by
  sorry

end percentage_of_juniors_l3490_349010


namespace condition_implication_l3490_349017

theorem condition_implication (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
sorry

end condition_implication_l3490_349017


namespace car_p_distance_l3490_349097

/-- The distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_p_distance (v : ℝ) : 
  let car_m_speed := v
  let car_m_time := 3
  let car_n_speed := 3 * v
  let car_n_time := 2
  let car_p_speed := 2 * v
  let car_p_time := 1.5
  distance car_p_speed car_p_time = 3 * v :=
by sorry

end car_p_distance_l3490_349097


namespace sum_recurring_thirds_equals_one_l3490_349095

-- Define the recurring decimal 0.333...
def recurring_third : ℚ := 1 / 3

-- Define the recurring decimal 0.666...
def recurring_two_thirds : ℚ := 2 / 3

-- Theorem: The sum of 0.333... and 0.666... is equal to 1
theorem sum_recurring_thirds_equals_one : 
  recurring_third + recurring_two_thirds = 1 := by
  sorry

end sum_recurring_thirds_equals_one_l3490_349095


namespace jackie_phil_same_heads_l3490_349019

/-- The probability of getting heads for a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- The probability of getting heads for the biased coin -/
def biased_coin_prob : ℚ := 4/7

/-- The probability of getting k heads when flipping the three coins -/
def prob_k_heads (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - fair_coin_prob)^2 * (1 - biased_coin_prob)
  | 1 => 2 * fair_coin_prob * (1 - fair_coin_prob) * (1 - biased_coin_prob) + 
         (1 - fair_coin_prob)^2 * biased_coin_prob
  | 2 => fair_coin_prob^2 * (1 - biased_coin_prob) + 
         2 * fair_coin_prob * (1 - fair_coin_prob) * biased_coin_prob
  | 3 => fair_coin_prob^2 * biased_coin_prob
  | _ => 0

/-- The probability that Jackie and Phil get the same number of heads -/
def prob_same_heads : ℚ :=
  (prob_k_heads 0)^2 + (prob_k_heads 1)^2 + (prob_k_heads 2)^2 + (prob_k_heads 3)^2

theorem jackie_phil_same_heads : prob_same_heads = 123/392 := by
  sorry

end jackie_phil_same_heads_l3490_349019


namespace bianca_tulips_l3490_349023

/-- The number of tulips Bianca picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Bianca picked -/
def total_flowers : ℕ := sorry

/-- The number of roses Bianca picked -/
def roses : ℕ := 49

/-- The number of flowers Bianca used -/
def used_flowers : ℕ := 81

/-- The number of extra flowers -/
def extra_flowers : ℕ := 7

theorem bianca_tulips : 
  tulips = 39 ∧ 
  total_flowers = tulips + roses ∧ 
  total_flowers = used_flowers + extra_flowers :=
sorry

end bianca_tulips_l3490_349023


namespace dog_grouping_combinations_l3490_349085

def total_dogs : ℕ := 15
def group_1_size : ℕ := 6
def group_2_size : ℕ := 5
def group_3_size : ℕ := 4

theorem dog_grouping_combinations :
  (total_dogs = group_1_size + group_2_size + group_3_size) →
  (Nat.choose (total_dogs - 2) (group_1_size - 1) * Nat.choose (total_dogs - group_1_size - 1) group_2_size = 72072) := by
  sorry

end dog_grouping_combinations_l3490_349085


namespace parameterization_validity_l3490_349062

def is_valid_parameterization (x₀ y₀ dx dy : ℝ) : Prop :=
  y₀ = -3 * x₀ + 5 ∧ dy / dx = -3

theorem parameterization_validity 
  (x₀ y₀ dx dy : ℝ) (dx_nonzero : dx ≠ 0) :
  is_valid_parameterization x₀ y₀ dx dy ↔
  (∀ t : ℝ, -3 * (x₀ + t * dx) + 5 = y₀ + t * dy) :=
sorry

end parameterization_validity_l3490_349062


namespace truck_speed_calculation_l3490_349005

/-- The average speed of Truck X in miles per hour -/
def truck_x_speed : ℝ := 57

/-- The average speed of Truck Y in miles per hour -/
def truck_y_speed : ℝ := 63

/-- The initial distance Truck X is ahead of Truck Y in miles -/
def initial_distance : ℝ := 14

/-- The final distance Truck Y is ahead of Truck X in miles -/
def final_distance : ℝ := 4

/-- The time it takes for Truck Y to overtake Truck X in hours -/
def overtake_time : ℝ := 3

theorem truck_speed_calculation :
  truck_x_speed = (truck_y_speed * overtake_time - initial_distance - final_distance) / overtake_time :=
by
  sorry

#check truck_speed_calculation

end truck_speed_calculation_l3490_349005


namespace circle_surrounding_circles_radius_l3490_349025

theorem circle_surrounding_circles_radius (r : ℝ) : 
  r > 0 →  -- r is positive
  (2 + r)^2 = 2 * (2 * r)^2 →  -- Pythagorean theorem for centers
  r = (4 * Real.sqrt 2 + 2) / 7 := by
sorry

end circle_surrounding_circles_radius_l3490_349025


namespace ninth_term_value_l3490_349084

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem ninth_term_value (seq : ArithmeticSequence) 
    (h6 : seq.S 6 = 3) 
    (h11 : seq.S 11 = 18) : 
  seq.a 9 = 3 := by
  sorry

end ninth_term_value_l3490_349084


namespace ducks_park_solution_l3490_349035

def ducks_park_problem (initial_ducks : ℕ) (geese_arrive : ℕ) (ducks_arrive : ℕ) (geese_leave : ℕ) : Prop :=
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let final_ducks : ℕ := initial_ducks + ducks_arrive
  let final_geese : ℕ := initial_geese - geese_leave
  final_geese - final_ducks = 1

theorem ducks_park_solution :
  ducks_park_problem 25 4 4 10 := by
  sorry

end ducks_park_solution_l3490_349035


namespace distance_ratio_theorem_l3490_349066

theorem distance_ratio_theorem (x : ℝ) (h1 : x^2 + (-9)^2 = 18^2) :
  |(-9)| / 18 = 1 / 2 := by sorry

end distance_ratio_theorem_l3490_349066


namespace arithmetic_sequence_ratio_l3490_349007

/-- An arithmetic sequence with first four terms a, x, b, and 2x -/
structure ArithmeticSequence (α : Type) [LinearOrderedField α] where
  a : α
  x : α
  b : α
  arithmetic_property : x - a = 2 * x - b

theorem arithmetic_sequence_ratio 
  {α : Type} [LinearOrderedField α] (seq : ArithmeticSequence α) :
  seq.a / seq.b = 1 / 3 := by
  sorry

end arithmetic_sequence_ratio_l3490_349007


namespace rectangle_area_minus_hole_l3490_349027

def large_rect_length (x : ℝ) : ℝ := x^2 + 7
def large_rect_width (x : ℝ) : ℝ := x^2 + 5
def hole_rect_length (x : ℝ) : ℝ := 2*x^2 - 3
def hole_rect_width (x : ℝ) : ℝ := x^2 - 2

theorem rectangle_area_minus_hole (x : ℝ) :
  large_rect_length x * large_rect_width x - hole_rect_length x * hole_rect_width x
  = -x^4 + 19*x^2 + 29 :=
by sorry

end rectangle_area_minus_hole_l3490_349027


namespace shirt_cost_l3490_349057

theorem shirt_cost (num_shirts : ℕ) (num_jeans : ℕ) (total_earnings : ℕ) :
  num_shirts = 20 →
  num_jeans = 10 →
  total_earnings = 400 →
  ∃ (shirt_cost : ℕ),
    shirt_cost * num_shirts + (2 * shirt_cost) * num_jeans = total_earnings ∧
    shirt_cost = 10 :=
by sorry

end shirt_cost_l3490_349057


namespace new_supervisor_salary_l3490_349067

/-- Proves that the salary of the new supervisor is $510 given the conditions of the problem -/
theorem new_supervisor_salary
  (initial_people : ℕ)
  (initial_average_salary : ℚ)
  (old_supervisor_salary : ℚ)
  (new_average_salary : ℚ)
  (h_initial_people : initial_people = 9)
  (h_initial_average : initial_average_salary = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_average : new_average_salary = 390)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 510 ∧
    (initial_people - 1) * (initial_average_salary * initial_people - old_supervisor_salary) / (initial_people - 1) +
    new_supervisor_salary = new_average_salary * initial_people :=
sorry

end new_supervisor_salary_l3490_349067


namespace integral_x_minus_reciprocal_x_l3490_349008

theorem integral_x_minus_reciprocal_x (f : ℝ → ℝ) (hf : ∀ x ∈ Set.Icc 1 2, HasDerivAt f (x - 1/x) x) :
  ∫ x in Set.Icc 1 2, (x - 1/x) = 1 - Real.log 2 := by
sorry

end integral_x_minus_reciprocal_x_l3490_349008


namespace intersection_equality_l3490_349056

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem intersection_equality (m : ℝ) : 
  A m ∩ B m = B m → m = 3 ∨ m = 0 := by
  sorry

end intersection_equality_l3490_349056


namespace symmetric_circle_l3490_349071

/-- Given a circle C1 and a line of symmetry, this theorem proves the equation of the symmetric circle C2. -/
theorem symmetric_circle (x y : ℝ) : 
  (∃ C1 : ℝ × ℝ → Prop, C1 = λ (x, y) ↦ (x - 3)^2 + (y + 1)^2 = 1) →
  (∃ L : ℝ × ℝ → Prop, L = λ (x, y) ↦ 2*x - y - 2 = 0) →
  (∃ C2 : ℝ × ℝ → Prop, C2 = λ (x, y) ↦ (x + 1)^2 + (y - 1)^2 = 1) :=
by sorry

end symmetric_circle_l3490_349071


namespace max_value_fraction_max_value_achievable_l3490_349014

theorem max_value_fraction (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable : 
  ∃ x y : ℝ, (2*x + 3*y + 4) / Real.sqrt (x^2 + y^2 + 2) = Real.sqrt 29 :=
by sorry

end max_value_fraction_max_value_achievable_l3490_349014


namespace cafeteria_pie_problem_l3490_349070

/-- Given a cafeteria with initial apples, some handed out, and a number of pies made,
    calculate the number of apples used per pie. -/
def apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (pies_made : ℕ) : ℕ :=
  (initial_apples - handed_out) / pies_made

/-- Theorem: In the specific case of 50 initial apples, 5 handed out, and 9 pies made,
    the number of apples per pie is 5. -/
theorem cafeteria_pie_problem :
  apples_per_pie 50 5 9 = 5 := by
  sorry

end cafeteria_pie_problem_l3490_349070


namespace min_distance_A_to_E_l3490_349009

/-- Given five points A, B, C, D, and E with specified distances between them,
    prove that the minimum possible distance between A and E is 2 units. -/
theorem min_distance_A_to_E (A B C D E : ℝ) : 
  (∃ (AB BC CD DE : ℝ), 
    AB = 12 ∧ 
    BC = 5 ∧ 
    CD = 3 ∧ 
    DE = 2 ∧ 
    (∀ (AE : ℝ), AE ≥ 2)) → 
  (∃ (AE : ℝ), AE = 2) :=
by sorry

end min_distance_A_to_E_l3490_349009


namespace square_area_ratio_l3490_349015

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  (2 * s)^2 / s^2 = 4 := by
sorry

end square_area_ratio_l3490_349015


namespace problem_solution_l3490_349037

theorem problem_solution (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 16) : y = 64 := by
  sorry

end problem_solution_l3490_349037


namespace coefficient_x_squared_in_expansion_l3490_349036

/-- The coefficient of x^2 in the expansion of (1+2x)^5 is 40 -/
theorem coefficient_x_squared_in_expansion : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * 2^k * if k = 2 then 1 else 0) = 40 := by
  sorry

end coefficient_x_squared_in_expansion_l3490_349036


namespace min_polyline_distance_circle_line_l3490_349038

/-- Polyline distance between two points -/
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- A point is on the unit circle -/
def on_unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- A point is on the given line -/
def on_line (x y : ℝ) : Prop :=
  2*x + y - 2*Real.sqrt 5 = 0

/-- The minimum polyline distance between the circle and the line -/
theorem min_polyline_distance_circle_line :
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), on_unit_circle x₁ y₁ → on_line x₂ y₂ →
      polyline_distance x₁ y₁ x₂ y₂ ≥ min_dist) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), on_unit_circle x₁ y₁ ∧ on_line x₂ y₂ ∧
      polyline_distance x₁ y₁ x₂ y₂ = min_dist) ∧
    min_dist = Real.sqrt 5 / 2 :=
sorry

end min_polyline_distance_circle_line_l3490_349038


namespace consecutive_odd_sum_l3490_349024

theorem consecutive_odd_sum (n : ℤ) : 
  (∃ k : ℤ, n = 2 * k + 1) →  -- n is odd
  (n + 2 = 9) →              -- middle number is 9
  (n + (n + 2) + (n + 4) - n = 20) := by
  sorry

end consecutive_odd_sum_l3490_349024


namespace palindrome_power_sum_l3490_349020

/-- A function to check if a natural number is a palindrome in decimal representation -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The main theorem stating the condition for 2^n + 2^m + 1 to be a palindrome -/
theorem palindrome_power_sum (m n : ℕ) : 
  isPalindrome (2^n + 2^m + 1) ↔ m ≤ 9 ∨ n ≤ 9 := by sorry

end palindrome_power_sum_l3490_349020


namespace bakery_storage_l3490_349055

theorem bakery_storage (sugar flour baking_soda : ℝ) 
  (h1 : sugar / flour = 5 / 4)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 3000 := by
sorry

end bakery_storage_l3490_349055


namespace complex_equation_solution_l3490_349001

theorem complex_equation_solution (i : ℂ) (h_i : i^2 = -1) :
  ∃ z : ℂ, (2 + i) * z = 2 - i ∧ z = 3/5 - 4/5 * i :=
sorry

end complex_equation_solution_l3490_349001


namespace product_of_roots_roots_product_of_equation_l3490_349030

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  f r₁ = 0 ∧ f r₂ = 0 → r₁ * r₂ = c / a :=
by sorry

theorem roots_product_of_equation :
  let f : ℝ → ℝ := λ x => x^2 + 14*x + 52
  let r₁ := (-14 + Real.sqrt (14^2 - 4*1*52)) / (2*1)
  let r₂ := (-14 - Real.sqrt (14^2 - 4*1*52)) / (2*1)
  f r₁ = 0 ∧ f r₂ = 0 → r₁ * r₂ = 48 :=
by sorry

end product_of_roots_roots_product_of_equation_l3490_349030


namespace infinite_congruent_sum_digits_l3490_349000

/-- Sum of digits function -/
def S (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the existence of infinitely many n such that S(n) ≡ n (mod p) for any prime p -/
theorem infinite_congruent_sum_digits (p : ℕ) (hp : Nat.Prime p) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ (k : ℕ), S (f k) ≡ f k [MOD p] :=
sorry

end infinite_congruent_sum_digits_l3490_349000


namespace integer_expression_l3490_349098

theorem integer_expression (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℕ, k * m = 3) ↔ 
  ∃ z : ℤ, z = (((n + 1)^2 - 3*k) / k^2) * (n.factorial / (k.factorial * (n - k).factorial)) :=
sorry

end integer_expression_l3490_349098


namespace lucas_age_l3490_349080

theorem lucas_age (noah_age mia_age lucas_age : ℕ) : 
  noah_age = 12 →
  mia_age = noah_age + 5 →
  lucas_age = mia_age - 6 →
  lucas_age = 11 :=
by
  sorry

end lucas_age_l3490_349080


namespace largest_prime_with_square_conditions_l3490_349043

theorem largest_prime_with_square_conditions : 
  ∀ p : ℕ, 
    p.Prime → 
    (∃ x : ℕ, (p + 1) / 2 = x^2) → 
    (∃ y : ℕ, (p^2 + 1) / 2 = y^2) → 
    p ≤ 7 := by
  sorry

end largest_prime_with_square_conditions_l3490_349043


namespace intersection_count_l3490_349094

/-- The complementary curve C₂ -/
def complementary_curve (x y : ℝ) : Prop := 1 / x^2 - 1 / y^2 = 1

/-- The hyperbola C₁ -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The line MN passing through (m,0) and (0,n) -/
def line_mn (m n x y : ℝ) : Prop := y = -n/m * x + n

theorem intersection_count (m n : ℝ) :
  complementary_curve m n →
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ line_mn m n p.1 p.2 :=
sorry

end intersection_count_l3490_349094


namespace white_wins_iff_n_gt_3_l3490_349051

/-- Represents the outcome of the game -/
inductive GameOutcome
  | WhiteWins
  | BlackWins

/-- Represents the game state -/
structure GameState where
  board_size : Nat
  white_position : Nat
  black_position : Nat

/-- Determines the winner of the game given the initial state -/
def determine_winner (initial_state : GameState) : GameOutcome :=
  if initial_state.board_size > 3 then GameOutcome.WhiteWins
  else GameOutcome.BlackWins

/-- Theorem stating the winning condition for the game -/
theorem white_wins_iff_n_gt_3 (n : Nat) (h : n > 2) :
  determine_winner {board_size := n, white_position := 1, black_position := n} = GameOutcome.WhiteWins ↔ n > 3 := by
  sorry


end white_wins_iff_n_gt_3_l3490_349051


namespace exam_score_calculation_l3490_349096

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 60 → 
  correct_answers = 34 → 
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 110 := by
  sorry

end exam_score_calculation_l3490_349096


namespace alpha_beta_ratio_l3490_349011

-- Define the angles
variable (α β x y : ℝ)

-- Define the angle relationships
axiom angle_relation_1 : y = x + β
axiom angle_relation_2 : 2 * y = 2 * x + α

-- Theorem to prove
theorem alpha_beta_ratio : α / β = 2 := by
  sorry

end alpha_beta_ratio_l3490_349011


namespace x_percent_of_x_squared_is_nine_l3490_349022

theorem x_percent_of_x_squared_is_nine (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x^2 = 9) :
  ∃ (y : ℝ), abs (x - y) < 0.01 ∧ y^3 = 900 ∧ 
  ∀ (z : ℤ), abs (x - ↑z) ≥ abs (x - 10) :=
sorry

end x_percent_of_x_squared_is_nine_l3490_349022


namespace carries_remaining_money_l3490_349086

/-- The amount of money Carrie has left after shopping -/
def money_left (initial_amount sweater_price tshirt_price shoes_price jeans_original_price jeans_discount : ℚ) : ℚ :=
  initial_amount - (sweater_price + tshirt_price + shoes_price + (jeans_original_price * (1 - jeans_discount)))

/-- Proof that Carrie has $27.50 left after shopping -/
theorem carries_remaining_money :
  money_left 91 24 6 11 30 (25/100) = 27.5 := by
  sorry

end carries_remaining_money_l3490_349086


namespace age_difference_of_parents_l3490_349041

theorem age_difference_of_parents (albert_age brother_age father_age mother_age : ℕ) :
  father_age = albert_age + 48 →
  mother_age = brother_age + 46 →
  brother_age = albert_age - 2 →
  father_age - mother_age = 4 := by
sorry

end age_difference_of_parents_l3490_349041


namespace ellipse_axis_endpoints_distance_l3490_349075

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 25 * (x + 3)^2 + 4 * y^2 = 100

-- Define the center of the ellipse
def center : ℝ × ℝ := (-3, 0)

-- Define the semi-major and semi-minor axis lengths
def semi_major_axis : ℝ := 5
def semi_minor_axis : ℝ := 2

-- Define the endpoints of the major and minor axes
def major_axis_endpoint : ℝ × ℝ := (-3, 5)
def minor_axis_endpoint : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem ellipse_axis_endpoints_distance :
  let C := major_axis_endpoint
  let D := minor_axis_endpoint
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = Real.sqrt 29 :=
by sorry

end ellipse_axis_endpoints_distance_l3490_349075


namespace ben_savings_proof_l3490_349006

/-- Represents the number of days that have elapsed -/
def days : ℕ := 7

/-- Ben's daily starting amount in cents -/
def daily_start : ℕ := 5000

/-- Ben's daily spending in cents -/
def daily_spend : ℕ := 1500

/-- Ben's dad's additional contribution in cents -/
def dad_contribution : ℕ := 1000

/-- Ben's final amount in cents -/
def final_amount : ℕ := 50000

theorem ben_savings_proof :
  2 * (days * (daily_start - daily_spend)) + dad_contribution = final_amount := by
  sorry

end ben_savings_proof_l3490_349006


namespace sin_675_degrees_l3490_349090

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_675_degrees_l3490_349090


namespace abs_x_minus_y_equals_four_l3490_349033

theorem abs_x_minus_y_equals_four (x y : ℝ) 
  (h1 : x^3 + y^3 = 26) 
  (h2 : x*y*(x+y) = -6) : 
  |x - y| = 4 := by
sorry

end abs_x_minus_y_equals_four_l3490_349033


namespace berts_spending_l3490_349072

theorem berts_spending (initial_amount : ℚ) : 
  initial_amount = 44 →
  let hardware_spent := (1 / 4) * initial_amount
  let after_hardware := initial_amount - hardware_spent
  let after_drycleaner := after_hardware - 9
  let grocery_spent := (1 / 2) * after_drycleaner
  let final_amount := after_drycleaner - grocery_spent
  final_amount = 12 := by sorry

end berts_spending_l3490_349072


namespace solve_for_m_l3490_349042

theorem solve_for_m : ∀ m : ℝ, (∃ x : ℝ, x = 3 ∧ 3 * m - 2 * x = 6) → m = 4 := by
  sorry

end solve_for_m_l3490_349042


namespace monotonic_increasing_iff_a_in_range_l3490_349079

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem monotonic_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a ∈ Set.Icc (3/2) 3) := by
  sorry

end monotonic_increasing_iff_a_in_range_l3490_349079


namespace min_value_theorem_l3490_349026

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 4) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ (z : ℝ), (2/x + 1/y) ≥ z → z ≤ min_val :=
by sorry

end min_value_theorem_l3490_349026


namespace chandra_pairings_l3490_349078

/-- Represents the number of valid pairings between bowls and glasses -/
def valid_pairings (num_bowls num_glasses num_unmatched : ℕ) : ℕ :=
  (num_bowls - num_unmatched) * num_glasses + num_unmatched * num_glasses

/-- Theorem: Given 5 bowls and 4 glasses, where one bowl doesn't have a matching glass,
    the total number of valid pairings is 20 -/
theorem chandra_pairings :
  valid_pairings 5 4 1 = 20 := by
  sorry

end chandra_pairings_l3490_349078


namespace inequality_proof_l3490_349082

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) := by
  sorry

end inequality_proof_l3490_349082


namespace solve_exponential_equation_l3490_349059

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^x * (3 : ℝ)^x * (3 : ℝ)^x * (3 : ℝ)^x = (81 : ℝ)^3 ∧ x = 3 := by
  sorry

end solve_exponential_equation_l3490_349059


namespace election_votes_theorem_l3490_349064

theorem election_votes_theorem (total_votes : ℕ) : 
  (total_votes : ℚ) * (60 / 100) - (total_votes : ℚ) * (40 / 100) = 280 → 
  total_votes = 1400 := by
sorry

end election_votes_theorem_l3490_349064


namespace rower_downstream_speed_l3490_349054

/-- Calculates the downstream speed of a rower given their upstream speed and still water speed. -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that a rower with an upstream speed of 12 kmph and a still water speed of 25 kmph
    will have a downstream speed of 38 kmph. -/
theorem rower_downstream_speed :
  downstream_speed 12 25 = 38 := by
  sorry

end rower_downstream_speed_l3490_349054


namespace cost_per_bag_of_chips_l3490_349093

/-- Given three friends buying chips, prove the cost per bag --/
theorem cost_per_bag_of_chips (num_friends : ℕ) (num_bags : ℕ) (payment_per_friend : ℚ) : 
  num_friends = 3 → num_bags = 5 → payment_per_friend = 5 →
  (num_friends * payment_per_friend) / num_bags = 3 := by
sorry

end cost_per_bag_of_chips_l3490_349093


namespace polynomial_evaluation_l3490_349031

theorem polynomial_evaluation (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = (1 - 3*x) * (1 + x)^5) →
  (∀ x, f x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + (1/3)*a₁ + (1/3^2)*a₂ + (1/3^3)*a₃ + (1/3^4)*a₄ + (1/3^5)*a₅ + (1/3^6)*a₆ = 0 := by
sorry


end polynomial_evaluation_l3490_349031


namespace retailer_profit_percentage_l3490_349047

/-- Represents the problem of calculating the profit percentage for a retailer --/
theorem retailer_profit_percentage 
  (monthly_sales : ℕ)
  (profit_per_item : ℚ)
  (discount_rate : ℚ)
  (break_even_sales : ℚ)
  (h1 : monthly_sales = 100)
  (h2 : profit_per_item = 30)
  (h3 : discount_rate = 0.05)
  (h4 : break_even_sales = 156.86274509803923)
  : ∃ (item_price : ℚ), 
    profit_per_item / item_price = 0.16 :=
by sorry

end retailer_profit_percentage_l3490_349047


namespace unique_solution_l3490_349077

theorem unique_solution : ∀ x y : ℕ+, 
  (x : ℝ) ^ (y : ℝ) - 1 = (y : ℝ) ^ (x : ℝ) → 
  2 * (x : ℝ) ^ (y : ℝ) = (y : ℝ) ^ (x : ℝ) + 5 → 
  x = 2 ∧ y = 2 := by
  sorry

end unique_solution_l3490_349077


namespace track_event_races_l3490_349089

/-- The number of races needed to determine a champion in a track event -/
def races_needed (total_athletes : ℕ) (lanes_per_race : ℕ) : ℕ :=
  let first_round := total_athletes / lanes_per_race
  let second_round := first_round / lanes_per_race
  let final_round := 1
  first_round + second_round + final_round

/-- Theorem stating that 43 races are needed for 216 athletes with 6 lanes per race -/
theorem track_event_races : races_needed 216 6 = 43 := by
  sorry

#eval races_needed 216 6

end track_event_races_l3490_349089


namespace melissa_games_l3490_349053

theorem melissa_games (total_points : ℕ) (points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) :
  total_points / points_per_game = 3 :=
by
  sorry

end melissa_games_l3490_349053


namespace slope_through_origin_and_point_l3490_349044

/-- The slope of a line passing through (0, 0) and (5, 1) is 1/5 -/
theorem slope_through_origin_and_point :
  let x1 : ℝ := 0
  let y1 : ℝ := 0
  let x2 : ℝ := 5
  let y2 : ℝ := 1
  let slope : ℝ := (y2 - y1) / (x2 - x1)
  slope = 1 / 5 := by sorry

end slope_through_origin_and_point_l3490_349044


namespace notebook_distribution_l3490_349074

theorem notebook_distribution (total_notebooks : ℕ) 
  (h1 : total_notebooks = 512) : 
  ∃ (num_children : ℕ), 
    (num_children > 0) ∧ 
    (total_notebooks = num_children * (num_children / 8)) ∧
    (total_notebooks = (num_children / 2) * 16) := by
  sorry

end notebook_distribution_l3490_349074


namespace min_roots_symmetric_function_l3490_349068

/-- A function with specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f (2 + x)) ∧
  (∀ x, f (7 - x) = f (7 + x)) ∧
  f 0 = 0

/-- The set of roots of f in the interval [-1000, 1000] -/
def RootSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc (-1000) 1000 ∧ f x = 0}

/-- The theorem stating the minimum number of roots -/
theorem min_roots_symmetric_function (f : ℝ → ℝ) (h : SymmetricFunction f) :
    401 ≤ (RootSet f).ncard := by
  sorry

end min_roots_symmetric_function_l3490_349068


namespace family_size_family_size_proof_l3490_349065

theorem family_size : ℕ → Prop :=
  fun n =>
    ∀ (b : ℕ),
      -- Peter has b brothers and 3b sisters
      (3 * b = n - b - 1) →
      -- Louise has b + 1 brothers and 3b - 1 sisters
      (3 * b - 1 = 2 * (b + 1)) →
      n = 13

-- The proof is omitted
theorem family_size_proof : family_size 13 := by sorry

end family_size_family_size_proof_l3490_349065


namespace triangle_angle_equality_l3490_349092

/-- In a triangle ABC, if sin(A)/a = cos(B)/b, then B = 45° --/
theorem triangle_angle_equality (A B : ℝ) (a b : ℝ) :
  (0 < a) → (0 < b) → (0 < A) → (A < π) → (0 < B) → (B < π) →
  (Real.sin A / a = Real.cos B / b) →
  B = π/4 := by
sorry

end triangle_angle_equality_l3490_349092


namespace ice_cream_sundaes_l3490_349058

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end ice_cream_sundaes_l3490_349058


namespace factorable_quadratic_b_eq_42_l3490_349018

/-- A quadratic expression that can be factored into two linear binomials with integer coefficients -/
structure FactorableQuadratic where
  b : ℤ
  factored : ∃ (d e f g : ℤ), 28 * x^2 + b * x + 14 = (d * x + e) * (f * x + g)

/-- Theorem stating that for a FactorableQuadratic, b must equal 42 -/
theorem factorable_quadratic_b_eq_42 (q : FactorableQuadratic) : q.b = 42 := by
  sorry

end factorable_quadratic_b_eq_42_l3490_349018


namespace line_AC_equation_circumcircle_equation_l3490_349002

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 4)

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the symmetry condition
def symmetric_about_l (p₁ p₂ : ℝ × ℝ) : Prop :=
  let m := (p₁.1 + p₂.1) / 2
  let n := (p₁.2 + p₂.2) / 2
  l m n

-- Define point C
def C : ℝ × ℝ := (-1, 3)

-- Theorem for the equation of line AC
theorem line_AC_equation (x y : ℝ) : x + y - 2 = 0 ↔ 
  (∃ t : ℝ, x = A.1 + t * (C.1 - A.1) ∧ y = A.2 + t * (C.2 - A.2)) :=
sorry

-- Theorem for the equation of the circumcircle
theorem circumcircle_equation (x y : ℝ) : 
  x^2 + y^2 - 3/2*x + 11/2*y - 17 = 0 ↔ 
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
  (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2 :=
sorry

end line_AC_equation_circumcircle_equation_l3490_349002


namespace linear_decreasing_negative_slope_l3490_349040

/-- A linear function f(x) = kx + b that is monotonically decreasing on ℝ has a negative slope k. -/
theorem linear_decreasing_negative_slope (k b : ℝ) : 
  (∀ x y, x < y → (k * x + b) > (k * y + b)) → k < 0 := by
  sorry

end linear_decreasing_negative_slope_l3490_349040


namespace abigail_spending_l3490_349088

theorem abigail_spending (initial_amount : ℝ) : 
  let food_expense := 0.6 * initial_amount
  let remainder_after_food := initial_amount - food_expense
  let phone_bill := 0.25 * remainder_after_food
  let remainder_after_phone := remainder_after_food - phone_bill
  let entertainment_expense := 20
  let final_amount := remainder_after_phone - entertainment_expense
  (final_amount = 40) → (initial_amount = 200) :=
by
  sorry

end abigail_spending_l3490_349088


namespace a_values_l3490_349081

def P : Set ℝ := {x | x^2 = 1}
def Q (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (a : ℝ) : Q a ⊆ P → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end a_values_l3490_349081


namespace sum_of_squares_of_roots_l3490_349069

theorem sum_of_squares_of_roots (a b c d : ℝ) : 
  (3 * a^4 - 6 * a^3 + 11 * a^2 + 15 * a - 7 = 0) →
  (3 * b^4 - 6 * b^3 + 11 * b^2 + 15 * b - 7 = 0) →
  (3 * c^4 - 6 * c^3 + 11 * c^2 + 15 * c - 7 = 0) →
  (3 * d^4 - 6 * d^3 + 11 * d^2 + 15 * d - 7 = 0) →
  a^2 + b^2 + c^2 + d^2 = -10/3 := by
sorry

end sum_of_squares_of_roots_l3490_349069


namespace sin_cos_identity_l3490_349045

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.cos (80 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l3490_349045


namespace arithmetic_square_root_of_25_l3490_349052

theorem arithmetic_square_root_of_25 : Real.sqrt 25 = 5 := by
  sorry

end arithmetic_square_root_of_25_l3490_349052


namespace circle_symmetry_l3490_349099

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 1 = 0

-- Define symmetry with respect to origin
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-2)^2 + y^2 = 5

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ, original_circle x y ↔ symmetric_circle (symmetric_point x y).1 (symmetric_point x y).2 :=
sorry

end circle_symmetry_l3490_349099
