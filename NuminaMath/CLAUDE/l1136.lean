import Mathlib

namespace divisibility_theorem_l1136_113697

theorem divisibility_theorem (a b : ℕ+) (h : (7^2009 : ℕ) ∣ (a^2 + b^2)) :
  (7^2010 : ℕ) ∣ (a * b) := by
  sorry

end divisibility_theorem_l1136_113697


namespace ellipse_major_axis_length_l1136_113645

/-- Given an ellipse with equation x²/m + y²/4 = 1 and focal length 4,
    prove that the length of its major axis is 4√2. -/
theorem ellipse_major_axis_length (m : ℝ) :
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c = 4 ∧ c^2 = m - 4) →     -- Focal length is 4
  (∃ a : ℝ, a = 2 * Real.sqrt 2 ∧ 2 * a = 4 * Real.sqrt 2) := -- Major axis length is 4√2
by sorry


end ellipse_major_axis_length_l1136_113645


namespace three_dogs_walking_time_l1136_113656

def base_charge : ℕ := 20
def per_minute_charge : ℕ := 1
def total_earnings : ℕ := 171
def one_dog_time : ℕ := 10
def two_dogs_time : ℕ := 7

def calculate_charge (dogs : ℕ) (minutes : ℕ) : ℕ :=
  dogs * (base_charge + per_minute_charge * minutes)

theorem three_dogs_walking_time :
  ∃ (x : ℕ), 
    calculate_charge 1 one_dog_time + 
    calculate_charge 2 two_dogs_time + 
    calculate_charge 3 x = total_earnings ∧ 
    x = 9 := by sorry

end three_dogs_walking_time_l1136_113656


namespace octagon_diagonals_l1136_113689

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Proof that an octagon has 20 diagonals -/
theorem octagon_diagonals : num_diagonals 8 = 20 := by
  sorry

end octagon_diagonals_l1136_113689


namespace gcd_one_powers_of_two_l1136_113694

def sequence_a : ℕ → ℕ
  | 0 => 3
  | (n + 1) => sequence_a n + n * (sequence_a n - 1)

theorem gcd_one_powers_of_two (m : ℕ) :
  (∀ n, Nat.gcd m (sequence_a n) = 1) ↔ ∃ t : ℕ, m = 2^t :=
sorry

end gcd_one_powers_of_two_l1136_113694


namespace max_value_implies_a_equals_one_l1136_113622

/-- The function f(x) = -x^2 + 2ax - a - a^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x - a - a^2

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ -2) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = -2) →
  a = 1 :=
sorry

end max_value_implies_a_equals_one_l1136_113622


namespace point_in_first_quadrant_implies_m_gt_one_m_gt_one_implies_point_in_first_quadrant_l1136_113693

/-- A point P(x, y) is in the first quadrant if and only if x > 0 and y > 0 -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The x-coordinate of point P is m - 1 -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P is m + 2 -/
def y_coord (m : ℝ) : ℝ := m + 2

/-- Theorem: If point P(m-1, m+2) is in the first quadrant, then m > 1 -/
theorem point_in_first_quadrant_implies_m_gt_one (m : ℝ) :
  in_first_quadrant (x_coord m) (y_coord m) → m > 1 :=
by sorry

/-- Theorem: If m > 1, then point P(m-1, m+2) is in the first quadrant -/
theorem m_gt_one_implies_point_in_first_quadrant (m : ℝ) :
  m > 1 → in_first_quadrant (x_coord m) (y_coord m) :=
by sorry

end point_in_first_quadrant_implies_m_gt_one_m_gt_one_implies_point_in_first_quadrant_l1136_113693


namespace unique_m_value_l1136_113624

theorem unique_m_value : ∃! m : ℝ, (abs m = 1) ∧ (m - 1 ≠ 0) := by sorry

end unique_m_value_l1136_113624


namespace absolute_value_inequality_l1136_113668

theorem absolute_value_inequality (a : ℝ) : |a| ≠ -|-a| := by
  sorry

end absolute_value_inequality_l1136_113668


namespace combustion_reaction_l1136_113677

/-- Represents the balanced chemical equation for the combustion of methane with chlorine and oxygen -/
structure BalancedEquation where
  ch4 : ℕ
  cl2 : ℕ
  o2 : ℕ
  co2 : ℕ
  hcl : ℕ
  h2o : ℕ
  balanced : ch4 = 1 ∧ cl2 = 4 ∧ o2 = 4 ∧ co2 = 1 ∧ hcl = 4 ∧ h2o = 2

/-- Represents the given quantities and products in the reaction -/
structure ReactionQuantities where
  ch4_given : ℕ
  cl2_given : ℕ
  co2_produced : ℕ
  hcl_produced : ℕ

/-- Theorem stating the required amount of O2 and produced amount of H2O -/
theorem combustion_reaction 
  (eq : BalancedEquation) 
  (quant : ReactionQuantities) 
  (h_ch4 : quant.ch4_given = 24) 
  (h_cl2 : quant.cl2_given = 48) 
  (h_co2 : quant.co2_produced = 24) 
  (h_hcl : quant.hcl_produced = 48) :
  ∃ (o2_required h2o_produced : ℕ), 
    o2_required = 96 ∧ 
    h2o_produced = 48 :=
  sorry

end combustion_reaction_l1136_113677


namespace guaranteed_payoff_probability_l1136_113655

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The score in the game is the sum of two die rolls -/
def score (roll1 roll2 : Die) : Nat := roll1.val + roll2.val + 2

/-- The maximum possible score in the game -/
def max_score : Nat := 12

/-- The number of players in the game -/
def num_players : Nat := 22

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : Die) : Rat := 1 / 6

theorem guaranteed_payoff_probability :
  let guaranteed_score := max_score
  let prob_guaranteed_score := (prob_single_roll ⟨5, by norm_num⟩) * (prob_single_roll ⟨5, by norm_num⟩)
  (∀ s, s < guaranteed_score → ∃ (rolls : Fin num_players → Die × Die), 
    ∃ i, score (rolls i).1 (rolls i).2 ≥ s) ∧ 
  prob_guaranteed_score = 1 / 36 := by
  sorry

end guaranteed_payoff_probability_l1136_113655


namespace difference_is_one_over_1650_l1136_113690

/-- The repeating decimal 0.060606... -/
def repeating_decimal : ℚ := 2 / 33

/-- The terminating decimal 0.06 -/
def terminating_decimal : ℚ := 6 / 100

/-- The difference between the repeating decimal and the terminating decimal -/
def difference : ℚ := repeating_decimal - terminating_decimal

theorem difference_is_one_over_1650 : difference = 1 / 1650 := by
  sorry

end difference_is_one_over_1650_l1136_113690


namespace polynomial_division_l1136_113623

theorem polynomial_division (x : ℝ) :
  5*x^4 - 9*x^3 + 3*x^2 + 7*x - 6 = (x - 1)*(5*x^3 - 4*x^2 + 7*x + 7) := by sorry

end polynomial_division_l1136_113623


namespace park_journey_distance_sum_l1136_113658

/-- Represents the speed and start time of a traveler -/
structure Traveler where
  speed : ℚ
  startTime : ℚ

/-- The problem setup -/
def ParkJourney (d : ℚ) (patrick tanya jose : Traveler) : Prop :=
  patrick.speed > 0 ∧
  patrick.startTime = 0 ∧
  tanya.speed = patrick.speed + 2 ∧
  tanya.startTime = patrick.startTime + 1 ∧
  jose.speed = tanya.speed + 7 ∧
  jose.startTime = tanya.startTime + 1 ∧
  d / patrick.speed = (d / tanya.speed) + 1 ∧
  d / patrick.speed = (d / jose.speed) + 2

theorem park_journey_distance_sum :
  ∀ (d : ℚ) (patrick tanya jose : Traveler),
  ParkJourney d patrick tanya jose →
  ∃ (m n : ℕ), m.Coprime n ∧ d = m / n ∧ m + n = 277 := by
  sorry

end park_journey_distance_sum_l1136_113658


namespace empty_set_subset_of_all_l1136_113611

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end empty_set_subset_of_all_l1136_113611


namespace work_completion_time_l1136_113692

/-- Given two workers a and b, where a is thrice as fast as b, proves that if a can complete
    a work alone in 40 days, then a and b together can complete the work in 30 days. -/
theorem work_completion_time
  (rate_a rate_b : ℝ)  -- Rates at which workers a and b work
  (h1 : rate_a = 3 * rate_b)  -- a is thrice as fast as b
  (h2 : rate_a * 40 = 1)  -- a alone completes the work in 40 days
  : (rate_a + rate_b) * 30 = 1 :=  -- a and b together complete the work in 30 days
by
  sorry


end work_completion_time_l1136_113692


namespace expansion_coefficient_l1136_113621

/-- The coefficient of the x^(3/2) term in the expansion of (√x - a/√x)^5 -/
def coefficient (a : ℝ) : ℝ := -5 * a

theorem expansion_coefficient (a : ℝ) :
  coefficient a = 30 → a = -6 := by sorry

end expansion_coefficient_l1136_113621


namespace min_value_of_2x_plus_y_l1136_113687

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 3) :
  2*x + y ≥ 8/3 ∧ (2*x + y = 8/3 ↔ x = 2/3 ∧ y = 4/3) :=
sorry

end min_value_of_2x_plus_y_l1136_113687


namespace axis_of_symmetry_translated_sine_l1136_113614

theorem axis_of_symmetry_translated_sine (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  ∃ (x : ℝ), x = k * π / 2 + π / 12 ∧
    ∀ (y : ℝ), f (x - y) = f (x + y) := by
  sorry

end axis_of_symmetry_translated_sine_l1136_113614


namespace expansion_coefficient_implies_a_value_l1136_113685

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (x - a/x)^9
def coeff_x3 (a : ℝ) : ℝ := -binomial 9 3 * a^3

-- Theorem statement
theorem expansion_coefficient_implies_a_value (a : ℝ) : 
  coeff_x3 a = -84 → a = 1 := by sorry

end expansion_coefficient_implies_a_value_l1136_113685


namespace video_game_points_l1136_113630

/-- 
Given a video game where:
- Each enemy defeated gives 9 points
- There are 11 enemies total in a level
- You destroy all but 3 enemies

Prove that the number of points earned is 72.
-/
theorem video_game_points : 
  (∀ (points_per_enemy : ℕ) (total_enemies : ℕ) (enemies_left : ℕ),
    points_per_enemy = 9 → 
    total_enemies = 11 → 
    enemies_left = 3 → 
    (total_enemies - enemies_left) * points_per_enemy = 72) :=
by sorry

end video_game_points_l1136_113630


namespace no_rational_solution_for_odd_coeff_quadratic_l1136_113610

theorem no_rational_solution_for_odd_coeff_quadratic 
  (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end no_rational_solution_for_odd_coeff_quadratic_l1136_113610


namespace max_value_sqrt_x_1_minus_9x_l1136_113603

theorem max_value_sqrt_x_1_minus_9x (x : ℝ) (h1 : 0 < x) (h2 : x < 1/9) :
  ∃ (max_val : ℝ), max_val = 1/6 ∧ ∀ y, 0 < y ∧ y < 1/9 → Real.sqrt (y * (1 - 9*y)) ≤ max_val :=
by sorry

end max_value_sqrt_x_1_minus_9x_l1136_113603


namespace herd_size_l1136_113627

theorem herd_size (first_son_fraction : ℚ) (second_son_fraction : ℚ) (third_son_fraction : ℚ) 
  (village_cows : ℕ) (fourth_son_cows : ℕ) :
  first_son_fraction = 1/3 →
  second_son_fraction = 1/6 →
  third_son_fraction = 3/10 →
  village_cows = 10 →
  fourth_son_cows = 9 →
  ∃ (total_cows : ℕ), 
    total_cows = 95 ∧
    (first_son_fraction + second_son_fraction + third_son_fraction) * total_cows + 
    village_cows + fourth_son_cows = total_cows :=
by sorry

end herd_size_l1136_113627


namespace base_of_exponent_l1136_113642

theorem base_of_exponent (x : ℕ) (h : x = 14) :
  (∀ y : ℕ, y > x → ¬(3^y ∣ 9^7)) ∧ (3^x ∣ 9^7) →
  ∃ b : ℕ, b^7 = 9^7 ∧ b = 9 :=
by sorry

end base_of_exponent_l1136_113642


namespace unique_zero_implies_t_bound_l1136_113613

/-- A cubic function parameterized by t -/
def f (t : ℝ) (x : ℝ) : ℝ := -2 * x^3 + 2 * t * x^2 + 1

/-- The derivative of f with respect to x -/
def f_deriv (t : ℝ) (x : ℝ) : ℝ := -6 * x^2 + 4 * t * x

/-- Theorem stating that if f has a unique zero, then t > -3/2 -/
theorem unique_zero_implies_t_bound (t : ℝ) :
  (∃! x, f t x = 0) → t > -3/2 := by
  sorry

end unique_zero_implies_t_bound_l1136_113613


namespace pumpkin_count_l1136_113688

/-- The number of pumpkins grown by Sandy -/
def sandy_pumpkins : ℕ := 51

/-- The number of pumpkins grown by Mike -/
def mike_pumpkins : ℕ := 23

/-- The total number of pumpkins grown by Sandy and Mike -/
def total_pumpkins : ℕ := sandy_pumpkins + mike_pumpkins

theorem pumpkin_count : total_pumpkins = 74 := by
  sorry

end pumpkin_count_l1136_113688


namespace square_difference_of_solutions_l1136_113696

theorem square_difference_of_solutions (α β : ℝ) : 
  α^2 = 2*α + 1 → β^2 = 2*β + 1 → α ≠ β → (α - β)^2 = 8 := by
  sorry

end square_difference_of_solutions_l1136_113696


namespace smallest_geometric_distinct_digits_l1136_113641

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem smallest_geometric_distinct_digits : 
  (∀ n : ℕ, is_three_digit n → 
    digits_are_distinct n → 
    is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) → 
    124 ≤ n) ∧ 
  is_three_digit 124 ∧ 
  digits_are_distinct 124 ∧ 
  is_geometric_sequence (124 / 100) ((124 / 10) % 10) (124 % 10) :=
sorry

end smallest_geometric_distinct_digits_l1136_113641


namespace principal_amount_proof_l1136_113683

/-- Prove that for a given principal amount, interest rate, and time period,
    if the difference between compound and simple interest is 20,
    then the principal amount is 8000. -/
theorem principal_amount_proof (P : ℝ) :
  let r : ℝ := 0.05  -- 5% annual interest rate
  let t : ℝ := 2     -- 2 years time period
  let compound_interest := P * (1 + r) ^ t - P
  let simple_interest := P * r * t
  compound_interest - simple_interest = 20 →
  P = 8000 := by
sorry

end principal_amount_proof_l1136_113683


namespace wall_completion_time_proof_l1136_113695

/-- Represents the wall dimensions -/
structure WallDimensions where
  thickness : ℝ
  length : ℝ
  height : ℝ

/-- Represents the working conditions -/
structure WorkingConditions where
  normal_pace : ℝ
  break_duration : ℝ
  break_count : ℕ
  faster_rate : ℝ
  faster_duration : ℝ
  min_work_between_breaks : ℝ

/-- Calculates the shortest possible time to complete the wall -/
def shortest_completion_time (wall : WallDimensions) (conditions : WorkingConditions) : ℝ :=
  sorry

theorem wall_completion_time_proof (wall : WallDimensions) (conditions : WorkingConditions) :
  wall.thickness = 0.25 ∧
  wall.length = 50 ∧
  wall.height = 2 ∧
  conditions.normal_pace = 26 ∧
  conditions.break_duration = 0.5 ∧
  conditions.break_count = 6 ∧
  conditions.faster_rate = 1.25 ∧
  conditions.faster_duration = 1 ∧
  conditions.min_work_between_breaks = 0.75 →
  shortest_completion_time wall conditions = 27.25 :=
by sorry

end wall_completion_time_proof_l1136_113695


namespace inverse_function_b_value_l1136_113660

/-- Given a function f and its inverse, prove the value of b -/
theorem inverse_function_b_value 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 1 / (2 * x + b)) 
  (h2 : ∀ x, f⁻¹ x = (2 - 3 * x) / (5 * x)) 
  : b = 11 / 5 := by
  sorry

end inverse_function_b_value_l1136_113660


namespace polynomial_simplification_and_division_l1136_113678

theorem polynomial_simplification_and_division (x : ℝ) (h : x ≠ -1) :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) =
  x^2 + 4 * x - 15 + 25 / (x + 1) := by
  sorry

end polynomial_simplification_and_division_l1136_113678


namespace baking_soda_cost_is_one_l1136_113629

/-- Represents the cost of supplies for a science project. -/
structure SupplyCost where
  students : ℕ
  bowCost : ℕ
  vinegarCost : ℕ
  totalCost : ℕ

/-- Calculates the cost of each box of baking soda. -/
def bakingSodaCost (s : SupplyCost) : ℕ :=
  (s.totalCost - (s.students * (s.bowCost + s.vinegarCost))) / s.students

/-- Theorem stating that the cost of each box of baking soda is $1. -/
theorem baking_soda_cost_is_one (s : SupplyCost)
  (h1 : s.students = 23)
  (h2 : s.bowCost = 5)
  (h3 : s.vinegarCost = 2)
  (h4 : s.totalCost = 184) :
  bakingSodaCost s = 1 := by
  sorry

end baking_soda_cost_is_one_l1136_113629


namespace dave_spent_43_tickets_l1136_113626

/-- The number of tickets Dave started with -/
def initial_tickets : ℕ := 98

/-- The number of tickets Dave had left after buying the stuffed tiger -/
def remaining_tickets : ℕ := 55

/-- The number of tickets Dave spent on the stuffed tiger -/
def spent_tickets : ℕ := initial_tickets - remaining_tickets

theorem dave_spent_43_tickets : spent_tickets = 43 := by
  sorry

end dave_spent_43_tickets_l1136_113626


namespace imaginary_part_of_complex_expression_l1136_113643

theorem imaginary_part_of_complex_expression : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I) * Complex.I) = -1 := by
  sorry

end imaginary_part_of_complex_expression_l1136_113643


namespace distinct_arrangements_of_six_l1136_113680

theorem distinct_arrangements_of_six (n : ℕ) (h : n = 6) : 
  Nat.factorial n = 720 := by
  sorry

end distinct_arrangements_of_six_l1136_113680


namespace ginger_water_bottle_capacity_l1136_113649

/-- Proves that Ginger's water bottle holds 2 cups given the problem conditions -/
theorem ginger_water_bottle_capacity 
  (hours_worked : ℕ) 
  (bottles_for_plants : ℕ) 
  (total_cups_used : ℕ) 
  (h1 : hours_worked = 8)
  (h2 : bottles_for_plants = 5)
  (h3 : total_cups_used = 26) :
  (total_cups_used : ℚ) / (hours_worked + bottles_for_plants : ℚ) = 2 := by
  sorry

#check ginger_water_bottle_capacity

end ginger_water_bottle_capacity_l1136_113649


namespace clock_setting_time_l1136_113676

/-- Represents a 24-hour clock time -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a clock time, wrapping around if necessary -/
def addMinutes (t : ClockTime) (m : ℤ) : ClockTime :=
  sorry

/-- Subtracts minutes from a clock time, wrapping around if necessary -/
def subtractMinutes (t : ClockTime) (m : ℕ) : ClockTime :=
  sorry

theorem clock_setting_time 
  (initial_time : ClockTime)
  (elapsed_hours : ℕ)
  (gain_rate : ℕ)
  (loss_rate : ℕ)
  (h : elapsed_hours = 20)
  (hgain : gain_rate = 1)
  (hloss : loss_rate = 2)
  (hfinal_diff : addMinutes initial_time (elapsed_hours * gain_rate) = 
                 addMinutes (subtractMinutes initial_time (elapsed_hours * loss_rate)) 60)
  (hfinal_time : addMinutes initial_time (elapsed_hours * gain_rate) = 
                 { hours := 12, minutes := 0, valid := sorry }) :
  initial_time = { hours := 15, minutes := 40, valid := sorry } :=
sorry

end clock_setting_time_l1136_113676


namespace stratified_sampling_correct_proportions_l1136_113686

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Represents the sample sizes for each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total sample size -/
def sampleSize (s : Sample) : ℕ :=
  s.elderly + s.middleAged + s.young

/-- Checks if the sample is proportional to the population -/
def isProportionalSample (p : Population) (s : Sample) : Prop :=
  s.elderly * totalPopulation p = p.elderly * sampleSize s ∧
  s.middleAged * totalPopulation p = p.middleAged * sampleSize s ∧
  s.young * totalPopulation p = p.young * sampleSize s

theorem stratified_sampling_correct_proportions 
  (p : Population) 
  (s : Sample) :
  p.elderly = 28 → 
  p.middleAged = 56 → 
  p.young = 84 → 
  sampleSize s = 36 → 
  isProportionalSample p s → 
  s.elderly = 6 ∧ s.middleAged = 12 ∧ s.young = 18 :=
sorry

end stratified_sampling_correct_proportions_l1136_113686


namespace circle_intersection_l1136_113601

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*(x - y) - 18 = 0

theorem circle_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    (circle1 x1 y1 ∧ circle2 x1 y1) ∧
    (circle1 x2 y2 ∧ circle2 x2 y2) ∧
    (x1 = 3 ∧ y1 = 3) ∧
    (x2 = -3 ∧ y2 = 5) :=
  sorry

end circle_intersection_l1136_113601


namespace min_distance_MN_l1136_113616

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  -- Equation of the hyperbola: x²/4 - y² = 1
  equation : ℝ → ℝ → Prop
  -- One asymptote has equation x - 2y = 0
  asymptote : ℝ → ℝ → Prop
  -- The hyperbola passes through (2√2, 1)
  passes_through : Prop

/-- Represents a point on the hyperbola -/
structure PointOnHyperbola (C : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : C.equation x y

/-- Represents the vertices of the hyperbola -/
structure HyperbolaVertices (C : Hyperbola) where
  A₁ : ℝ × ℝ  -- Left vertex
  A₂ : ℝ × ℝ  -- Right vertex

/-- Function to calculate |MN| given a point P on the hyperbola -/
def distance_MN (C : Hyperbola) (V : HyperbolaVertices C) (P : PointOnHyperbola C) : ℝ :=
  sorry  -- Definition of |MN| calculation

/-- The main theorem to prove -/
theorem min_distance_MN (C : Hyperbola) (V : HyperbolaVertices C) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 3 ∧
  ∀ (P : PointOnHyperbola C), distance_MN C V P ≥ min_dist :=
sorry

end min_distance_MN_l1136_113616


namespace somu_father_age_ratio_l1136_113669

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The condition that Somu's age 8 years ago was one-fifth of his father's age 8 years ago -/
def age_relation (ages : Ages) : Prop :=
  ages.somu - 8 = (ages.father - 8) / 5

/-- The theorem stating the ratio of Somu's age to his father's age -/
theorem somu_father_age_ratio :
  ∀ (ages : Ages),
  ages.somu = 16 →
  age_relation ages →
  (ages.somu : ℚ) / (ages.father : ℚ) = 1 / 3 := by
  sorry


end somu_father_age_ratio_l1136_113669


namespace lindas_furniture_spending_l1136_113673

/-- Given Linda's original savings and the cost of a TV, 
    prove the fraction of her savings spent on furniture. -/
theorem lindas_furniture_spending 
  (original_savings : ℚ) 
  (tv_cost : ℚ) 
  (h1 : original_savings = 600)
  (h2 : tv_cost = 150) : 
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end lindas_furniture_spending_l1136_113673


namespace evaluate_expression_l1136_113674

theorem evaluate_expression : -(16 / 2 * 8 - 72 + 4^2) = -8 := by sorry

end evaluate_expression_l1136_113674


namespace line_plane_relationships_l1136_113666

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships between lines and planes
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry
def parallel_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry
def perpendicular_line_line (l1 : Line) (l2 : Line) : Prop := sorry
def parallel_line_line (l1 : Line) (l2 : Line) : Prop := sorry

-- Define the theorem
theorem line_plane_relationships 
  (m n : Line) (α β : Plane) 
  (hm : m ≠ n) (hα : α ≠ β) : 
  (perpendicular_line_plane m α ∧ 
   perpendicular_line_plane n β ∧ 
   perpendicular_plane_plane α β → 
   perpendicular_line_line m n) ∧
  (¬ (parallel_line_plane m α ∧ 
      perpendicular_line_plane n β ∧ 
      perpendicular_plane_plane α β → 
      parallel_line_line m n)) ∧
  (perpendicular_line_plane m α ∧ 
   parallel_line_plane n β ∧ 
   parallel_plane_plane α β → 
   perpendicular_line_line m n) ∧
  (perpendicular_line_plane m α ∧ 
   perpendicular_line_plane n β ∧ 
   parallel_plane_plane α β → 
   parallel_line_line m n) := by
  sorry


end line_plane_relationships_l1136_113666


namespace find_x_l1136_113650

theorem find_x : ∃ x : ℚ, (1/2 * x) = (1/3 * x + 110) ∧ x = 660 := by sorry

end find_x_l1136_113650


namespace product_evaluation_l1136_113651

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l1136_113651


namespace equation_describes_two_lines_l1136_113675

/-- The set of points satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of x-axis and y-axis -/
def TwoLines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_describes_two_lines : S = TwoLines := by
  sorry

end equation_describes_two_lines_l1136_113675


namespace polynomial_simplification_l1136_113665

theorem polynomial_simplification (x : ℝ) :
  5 - 7*x - 13*x^2 + 10 + 15*x - 25*x^2 - 20 + 21*x + 33*x^2 - 15*x^3 =
  -15*x^3 - 5*x^2 + 29*x - 5 := by
  sorry

end polynomial_simplification_l1136_113665


namespace quadratic_roots_sum_inverse_l1136_113664

theorem quadratic_roots_sum_inverse (p q : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + p*x1 + q = 0 ∧ x2^2 + p*x2 + q = 0)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ x3^2 + q*x3 + p = 0 ∧ x4^2 + q*x4 + p = 0) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x3 ≠ x4 ∧
    x1^2 + p*x1 + q = 0 ∧ x2^2 + p*x2 + q = 0 ∧
    x3^2 + q*x3 + p = 0 ∧ x4^2 + q*x4 + p = 0 ∧
    1/(x1*x3) + 1/(x1*x4) + 1/(x2*x3) + 1/(x2*x4) = 1 := by
  sorry

end quadratic_roots_sum_inverse_l1136_113664


namespace y_intercept_of_line_l1136_113663

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (m a : ℝ) : ℝ := a

/-- A line in slope-intercept form is defined by y = mx + b, where m is the slope and b is the y-intercept. -/
def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem y_intercept_of_line :
  y_intercept 2 1 = 1 :=
sorry

end y_intercept_of_line_l1136_113663


namespace arrangement_exists_l1136_113682

theorem arrangement_exists : ∃ (p : Fin 100 → Fin 100), Function.Bijective p ∧ 
  ∀ i : Fin 99, 
    (((p (i + 1)).val = (p i).val + 2) ∨ ((p (i + 1)).val = (p i).val - 2)) ∨
    ((p (i + 1)).val = 2 * (p i).val) ∨ ((p i).val = 2 * (p (i + 1)).val) := by
  sorry

end arrangement_exists_l1136_113682


namespace janet_pill_intake_l1136_113618

/-- Represents Janet's pill intake schedule for a month --/
structure PillSchedule where
  multivitamins_per_day : ℕ
  calcium_first_two_weeks : ℕ
  calcium_last_two_weeks : ℕ
  weeks_in_month : ℕ

/-- Calculates the total number of pills Janet takes in a month --/
def total_pills (schedule : PillSchedule) : ℕ :=
  let days_per_period := schedule.weeks_in_month / 2 * 7
  let pills_first_two_weeks := (schedule.multivitamins_per_day + schedule.calcium_first_two_weeks) * days_per_period
  let pills_last_two_weeks := (schedule.multivitamins_per_day + schedule.calcium_last_two_weeks) * days_per_period
  pills_first_two_weeks + pills_last_two_weeks

/-- Theorem stating that Janet's total pill intake for the month is 112 --/
theorem janet_pill_intake :
  ∃ (schedule : PillSchedule),
    schedule.multivitamins_per_day = 2 ∧
    schedule.calcium_first_two_weeks = 3 ∧
    schedule.calcium_last_two_weeks = 1 ∧
    schedule.weeks_in_month = 4 ∧
    total_pills schedule = 112 := by
  sorry

end janet_pill_intake_l1136_113618


namespace color_film_fraction_l1136_113608

/-- Given a film festival selection process, this theorem proves the fraction of selected films that are in color. -/
theorem color_film_fraction (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) : 
  let total_bw : ℝ := 20 * x
  let total_color : ℝ := 4 * y
  let selected_bw : ℝ := (y / x) * total_bw / 100
  let selected_color : ℝ := total_color
  (selected_color) / (selected_bw + selected_color) = 20 / (x + 20) := by
  sorry

end color_film_fraction_l1136_113608


namespace total_green_peaches_is_fourteen_l1136_113647

/-- The number of baskets containing peaches. -/
def num_baskets : ℕ := 7

/-- The number of green peaches in each basket. -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of green peaches in all baskets. -/
def total_green_peaches : ℕ := num_baskets * green_peaches_per_basket

theorem total_green_peaches_is_fourteen : total_green_peaches = 14 := by
  sorry

end total_green_peaches_is_fourteen_l1136_113647


namespace angle_difference_equality_l1136_113633

/-- Represents a triangle with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

/-- Represents the bisection of angle C into C1 and C2 -/
structure BisectedC (t : Triangle) where
  C1 : ℝ
  C2 : ℝ
  sum_C : C1 + C2 = t.C
  positive : 0 < C1 ∧ 0 < C2

theorem angle_difference_equality (t : Triangle) (bc : BisectedC t) 
    (h_A_B : t.A = t.B - 15) 
    (h_C2_adjacent : True) -- This is just a placeholder for the condition that C2 is adjacent to the side opposite B
    : bc.C1 - bc.C2 = 15 := by
  sorry

end angle_difference_equality_l1136_113633


namespace chocolate_cookies_sold_l1136_113605

/-- Proves that the number of chocolate cookies sold is 220 --/
theorem chocolate_cookies_sold (price_chocolate : ℕ) (price_vanilla : ℕ) (vanilla_sold : ℕ) (total_revenue : ℕ) :
  price_chocolate = 1 →
  price_vanilla = 2 →
  vanilla_sold = 70 →
  total_revenue = 360 →
  total_revenue = price_chocolate * (total_revenue - price_vanilla * vanilla_sold) + price_vanilla * vanilla_sold →
  total_revenue - price_vanilla * vanilla_sold = 220 :=
by sorry

end chocolate_cookies_sold_l1136_113605


namespace pairball_playing_time_l1136_113691

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) : 
  total_time = 90 → num_children = 5 → (total_time * 2) / num_children = 36 := by
  sorry

end pairball_playing_time_l1136_113691


namespace dinitrogen_trioxide_weight_l1136_113670

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in Dinitrogen trioxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in Dinitrogen trioxide -/
def O_count : ℕ := 3

/-- The molecular weight of Dinitrogen trioxide in g/mol -/
def molecular_weight_N2O3 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

/-- Theorem stating that the molecular weight of Dinitrogen trioxide is 76.02 g/mol -/
theorem dinitrogen_trioxide_weight : molecular_weight_N2O3 = 76.02 := by
  sorry

end dinitrogen_trioxide_weight_l1136_113670


namespace product_from_lcm_gcd_l1136_113671

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 72) 
  (h_gcd : Nat.gcd x y = 8) : 
  x * y = 576 := by
  sorry

end product_from_lcm_gcd_l1136_113671


namespace pure_imaginary_m_l1136_113684

/-- A complex number z is pure imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of m. -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)

theorem pure_imaginary_m : ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end pure_imaginary_m_l1136_113684


namespace interest_rates_calculation_l1136_113661

/-- Represents the interest calculation for a loan -/
structure Loan where
  principal : ℕ  -- Principal amount in rupees
  time : ℕ       -- Time in years
  interest : ℕ   -- Total interest received in rupees

/-- Calculates the annual interest rate given a loan -/
def calculate_rate (l : Loan) : ℚ :=
  (l.interest : ℚ) * 100 / (l.principal * l.time)

theorem interest_rates_calculation 
  (loan_b : Loan) 
  (loan_c : Loan) 
  (loan_d : Loan) 
  (loan_e : Loan) 
  (h1 : loan_b.principal = 5000 ∧ loan_b.time = 2)
  (h2 : loan_c.principal = 3000 ∧ loan_c.time = 4)
  (h3 : loan_d.principal = 7000 ∧ loan_d.time = 3 ∧ loan_d.interest = 2940)
  (h4 : loan_e.principal = 4500 ∧ loan_e.time = 5 ∧ loan_e.interest = 3375)
  (h5 : loan_b.interest + loan_c.interest = 1980)
  (h6 : calculate_rate loan_b = calculate_rate loan_c) :
  calculate_rate loan_d = 14 ∧ calculate_rate loan_e = 15 :=
sorry

end interest_rates_calculation_l1136_113661


namespace power_fraction_equality_l1136_113602

theorem power_fraction_equality : (2^8 : ℚ) / (8^2 : ℚ) = 4 := by
  sorry

end power_fraction_equality_l1136_113602


namespace harry_pet_feeding_cost_l1136_113634

/-- Represents the annual cost of feeding Harry's pets -/
def annual_feeding_cost (num_geckos num_iguanas num_snakes : ℕ) 
  (gecko_meals_per_month iguana_meals_per_month : ℕ) 
  (snake_meals_per_year : ℕ)
  (gecko_meal_cost iguana_meal_cost snake_meal_cost : ℕ) : ℕ :=
  (num_geckos * gecko_meals_per_month * 12 * gecko_meal_cost) +
  (num_iguanas * iguana_meals_per_month * 12 * iguana_meal_cost) +
  (num_snakes * snake_meals_per_year * snake_meal_cost)

/-- Theorem stating the annual cost of feeding Harry's pets -/
theorem harry_pet_feeding_cost :
  annual_feeding_cost 3 2 4 2 3 6 8 12 20 = 1920 := by
  sorry

#eval annual_feeding_cost 3 2 4 2 3 6 8 12 20

end harry_pet_feeding_cost_l1136_113634


namespace tan_product_thirty_degrees_l1136_113644

theorem tan_product_thirty_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end tan_product_thirty_degrees_l1136_113644


namespace correlation_index_approaching_one_improves_fitting_l1136_113619

/-- The correlation index in regression analysis -/
def correlation_index : ℝ → ℝ := sorry

/-- The fitting effect of a regression model -/
def fitting_effect : ℝ → ℝ := sorry

/-- As the correlation index approaches 1, the fitting effect improves -/
theorem correlation_index_approaching_one_improves_fitting :
  ∀ ε > 0, ∃ δ > 0, ∀ r : ℝ,
    1 - δ < correlation_index r → 
    fitting_effect r > fitting_effect 0 + ε :=
sorry

end correlation_index_approaching_one_improves_fitting_l1136_113619


namespace goose_eggs_count_l1136_113615

theorem goose_eggs_count (total_eggs : ℕ) : 
  (1 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 * total_eggs = 120 →
  total_eggs = 1200 := by
  sorry

end goose_eggs_count_l1136_113615


namespace sports_equipment_purchase_l1136_113679

/-- Represents the purchase of sports equipment --/
structure Equipment where
  price_a : ℕ  -- price of type A equipment
  price_b : ℕ  -- price of type B equipment
  quantity_a : ℕ  -- quantity of type A equipment purchased
  quantity_b : ℕ  -- quantity of type B equipment purchased

/-- The main theorem about the sports equipment purchase --/
theorem sports_equipment_purchase 
  (e : Equipment) 
  (h1 : e.price_b = e.price_a + 10)  -- price difference condition
  (h2 : e.quantity_a * e.price_a = 300)  -- total cost of A
  (h3 : e.quantity_b * e.price_b = 360)  -- total cost of B
  (h4 : e.quantity_a = e.quantity_b)  -- equal quantities purchased
  : 
  (e.price_a = 50 ∧ e.price_b = 60) ∧  -- correct prices
  (∀ m n : ℕ, 
    50 * m + 60 * n = 1000 ↔ 
    ((m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15))) -- possible scenarios
  := by sorry


end sports_equipment_purchase_l1136_113679


namespace max_value_implies_a_l1136_113604

/-- The function f(x) = 2x^3 - 3x^2 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem max_value_implies_a (a : ℝ) :
  (∃ (max : ℝ), max = 6 ∧ ∀ (x : ℝ), f a x ≤ max) →
  a = 6 :=
by sorry

end max_value_implies_a_l1136_113604


namespace min_value_x_plus_2y_l1136_113657

theorem min_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x + 8*y + 1 = 0) :
  ∃ (m : ℝ), m = -2*Real.sqrt 2 - 1 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 - 2*a + 8*b + 1 = 0 → m ≤ a + 2*b :=
sorry

end min_value_x_plus_2y_l1136_113657


namespace sin_double_angle_problem_l1136_113607

theorem sin_double_angle_problem (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (π / 2 - α) = 3 / 5) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end sin_double_angle_problem_l1136_113607


namespace odd_decreasing_properties_l1136_113652

/-- An odd and decreasing function on ℝ -/
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x ≤ y → f y ≤ f x)

theorem odd_decreasing_properties
  (f : ℝ → ℝ) (hf : odd_decreasing_function f) (m n : ℝ) (h : m + n ≥ 0) :
  (f m * f (-m) ≤ 0) ∧ (f m + f n ≤ f (-m) + f (-n)) :=
by sorry

end odd_decreasing_properties_l1136_113652


namespace arithmetic_geometric_mean_problem_l1136_113625

theorem arithmetic_geometric_mean_problem (a b : ℝ) 
  (h1 : (a + b) / 2 = 24) 
  (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 110) : 
  a^2 + b^2 = 1424 := by
  sorry

end arithmetic_geometric_mean_problem_l1136_113625


namespace percentage_women_non_union_l1136_113667

/-- Represents the percentage of employees in a company who are men -/
def percent_men : ℝ := 0.56

/-- Represents the percentage of employees in a company who are unionized -/
def percent_unionized : ℝ := 0.60

/-- Represents the percentage of non-union employees who are women -/
def percent_women_non_union : ℝ := 0.65

/-- Theorem stating that the percentage of women among non-union employees is 65% -/
theorem percentage_women_non_union :
  percent_women_non_union = 0.65 := by sorry

end percentage_women_non_union_l1136_113667


namespace function_sum_derivative_difference_l1136_113681

/-- Given a function f(x) = a*sin(3x) + b*x^3 + 4 where a and b are real numbers,
    prove that f(2014) + f(-2014) + f'(2015) - f'(-2015) = 8 -/
theorem function_sum_derivative_difference (a b : ℝ) : 
  let f (x : ℝ) := a * Real.sin (3 * x) + b * x^3 + 4
  let f' (x : ℝ) := 3 * a * Real.cos (3 * x) + 3 * b * x^2
  f 2014 + f (-2014) + f' 2015 - f' (-2015) = 8 := by
sorry

end function_sum_derivative_difference_l1136_113681


namespace sum_of_triangle_ops_l1136_113699

/-- Operation on three numbers as defined in the problem -/
def triangle_op (a b c : ℕ) : ℕ := a * b - c

/-- Theorem stating the sum of results from two specific triangles -/
theorem sum_of_triangle_ops : 
  triangle_op 4 2 3 + triangle_op 3 5 1 = 19 := by sorry

end sum_of_triangle_ops_l1136_113699


namespace trigonometric_equation_solution_l1136_113648

theorem trigonometric_equation_solution (x : ℝ) :
  (4 * Real.sin x ^ 4 + Real.cos (4 * x) = 1 + 12 * Real.cos x ^ 4) ↔
  (∃ k : ℤ, x = π / 3 * (3 * ↑k + 1) ∨ x = π / 3 * (3 * ↑k - 1)) :=
by sorry

end trigonometric_equation_solution_l1136_113648


namespace intersection_sum_zero_l1136_113635

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 7 = (y + 2)^2

-- Define the intersection points
def intersection_points : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum_zero :
  intersection_points →
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry

end intersection_sum_zero_l1136_113635


namespace complex_magnitude_l1136_113638

theorem complex_magnitude (z : ℂ) (h : (3 + Complex.I) / z = 1 - Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1136_113638


namespace count_linear_inequalities_one_variable_l1136_113620

-- Define a structure for an expression
structure Expression where
  is_linear_inequality : Bool
  has_one_variable : Bool

-- Define the six expressions
def expressions : List Expression := [
  { is_linear_inequality := true,  has_one_variable := true  }, -- ①
  { is_linear_inequality := false, has_one_variable := true  }, -- ②
  { is_linear_inequality := false, has_one_variable := true  }, -- ③
  { is_linear_inequality := true,  has_one_variable := true  }, -- ④
  { is_linear_inequality := true,  has_one_variable := true  }, -- ⑤
  { is_linear_inequality := true,  has_one_variable := false }  -- ⑥
]

-- Theorem statement
theorem count_linear_inequalities_one_variable :
  (expressions.filter (fun e => e.is_linear_inequality && e.has_one_variable)).length = 3 := by
  sorry

end count_linear_inequalities_one_variable_l1136_113620


namespace roots_of_polynomial_l1136_113653

def p (x : ℝ) : ℝ := 4*x^4 - 5*x^3 - 30*x^2 + 40*x + 24

theorem roots_of_polynomial :
  {x : ℝ | p x = 0} = {3, -1, -2, 1} :=
sorry

end roots_of_polynomial_l1136_113653


namespace total_people_in_line_l1136_113639

/-- The number of people in a ticket line -/
def ticket_line (people_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  people_in_front + 1 + (position_from_back - 1)

/-- Theorem: There are 11 people in the ticket line -/
theorem total_people_in_line :
  ticket_line 6 5 = 11 := by
  sorry

end total_people_in_line_l1136_113639


namespace martha_journey_distance_l1136_113636

theorem martha_journey_distance :
  -- Initial conditions
  ∀ (initial_speed : ℝ) (initial_distance : ℝ) (speed_increase : ℝ) (late_time : ℝ),
  initial_speed = 45 →
  initial_distance = 45 →
  speed_increase = 10 →
  late_time = 0.75 →
  -- The actual journey time
  ∃ (t : ℝ),
  -- The total distance
  ∃ (d : ℝ),
  -- Equation for the journey if continued at initial speed
  d = initial_speed * (t + late_time) ∧
  -- Equation for the actual journey with increased speed
  d - initial_distance = (initial_speed + speed_increase) * (t - 1) →
  -- The distance to the meeting place
  d = 230.625 :=
by sorry

end martha_journey_distance_l1136_113636


namespace stock_price_change_l1136_113617

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let day1_price := initial_price * (1 - 0.25)
  let day2_price := day1_price * (1 + 0.35)
  (day2_price - initial_price) / initial_price = 0.0125 := by
  sorry

end stock_price_change_l1136_113617


namespace divisors_of_500_l1136_113654

theorem divisors_of_500 : Finset.card (Nat.divisors 500) = 12 := by
  sorry

end divisors_of_500_l1136_113654


namespace cubic_minus_linear_factorization_l1136_113637

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l1136_113637


namespace jerry_birthday_games_l1136_113600

/-- The number of games Jerry received for his birthday -/
def games_received (initial_games final_games : ℕ) : ℕ :=
  final_games - initial_games

/-- Proof that Jerry received 2 games for his birthday -/
theorem jerry_birthday_games :
  games_received 7 9 = 2 := by
  sorry

end jerry_birthday_games_l1136_113600


namespace polynomial_remainder_l1136_113631

theorem polynomial_remainder (x : ℝ) : 
  (x^15 + 3) % (x + 2) = -32765 := by sorry

end polynomial_remainder_l1136_113631


namespace sum_of_radii_is_14_l1136_113646

/-- The sum of all possible radii of a circle that is tangent to both positive x and y-axes
    and externally tangent to another circle centered at (5,0) with radius 2 is equal to 14. -/
theorem sum_of_radii_is_14 : ∃ r₁ r₂ : ℝ,
  (∀ x y : ℝ, x^2 + y^2 = r₁^2 ∧ x ≥ 0 ∧ y ≥ 0 → (x = r₁ ∧ y = 0) ∨ (x = 0 ∧ y = r₁)) ∧
  (∀ x y : ℝ, x^2 + y^2 = r₂^2 ∧ x ≥ 0 ∧ y ≥ 0 → (x = r₂ ∧ y = 0) ∨ (x = 0 ∧ y = r₂)) ∧
  ((r₁ - 5)^2 + r₁^2 = (r₁ + 2)^2) ∧
  ((r₂ - 5)^2 + r₂^2 = (r₂ + 2)^2) ∧
  r₁ + r₂ = 14 := by
  sorry

end sum_of_radii_is_14_l1136_113646


namespace shifted_line_through_origin_l1136_113659

/-- A line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shift a line horizontally -/
def shift_line (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + l.slope * d }

/-- Check if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem shifted_line_through_origin (b : ℝ) :
  let original_line := Line.mk 2 b
  let shifted_line := shift_line original_line 2
  passes_through shifted_line 0 0 → b = 4 := by
  sorry

end shifted_line_through_origin_l1136_113659


namespace smallest_four_digit_multiple_of_18_l1136_113606

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 
  n = 1008 ∧ 
  n % 18 = 0 ∧ 
  1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m : ℕ, m % 18 = 0 → 1000 ≤ m → m < 10000 → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l1136_113606


namespace cost_price_calculation_l1136_113662

/-- Calculates the cost price of an article given the final sale price, sales tax rate, and profit rate. -/
theorem cost_price_calculation (final_price : ℝ) (sales_tax_rate : ℝ) (profit_rate : ℝ) :
  final_price = 616 →
  sales_tax_rate = 0.1 →
  profit_rate = 0.16 →
  ∃ (cost_price : ℝ),
    cost_price > 0 ∧
    (cost_price * (1 + profit_rate) * (1 + sales_tax_rate) = final_price) ∧
    (abs (cost_price - 482.76) < 0.01) :=
by sorry

end cost_price_calculation_l1136_113662


namespace no_perfect_square_3n_plus_2_17n_l1136_113628

theorem no_perfect_square_3n_plus_2_17n :
  ∀ n : ℕ, ¬∃ m : ℕ, 3^n + 2 * 17^n = m^2 := by
  sorry

end no_perfect_square_3n_plus_2_17n_l1136_113628


namespace six_digit_divisible_by_72_l1136_113609

theorem six_digit_divisible_by_72 (A B : ℕ) : 
  A < 10 →
  B < 10 →
  (A * 100000 + 44610 + B) % 72 = 0 →
  A + B = 12 := by
sorry

end six_digit_divisible_by_72_l1136_113609


namespace complement_M_inter_N_eq_singleton_three_l1136_113672

def M : Set ℤ := {x | x < 3}
def N : Set ℤ := {x | x < 4}
def U : Set ℤ := Set.univ

theorem complement_M_inter_N_eq_singleton_three :
  (U \ M) ∩ N = {3} := by sorry

end complement_M_inter_N_eq_singleton_three_l1136_113672


namespace certified_mail_delivery_l1136_113612

/-- The total number of pieces of certified mail delivered by Johann and his friends -/
def total_mail (friends_mail : ℕ) (johann_mail : ℕ) : ℕ :=
  2 * friends_mail + johann_mail

/-- Theorem stating the total number of pieces of certified mail to be delivered -/
theorem certified_mail_delivery :
  let friends_mail := 41
  let johann_mail := 98
  total_mail friends_mail johann_mail = 180 := by
sorry

end certified_mail_delivery_l1136_113612


namespace root_implies_m_values_l1136_113640

theorem root_implies_m_values (m : ℝ) : 
  ((m + 2) * 1^2 - 2 * 1 + m^2 - 2*m - 6 = 0) → (m = -2 ∨ m = 3) := by
  sorry

end root_implies_m_values_l1136_113640


namespace closest_to_fraction_l1136_113632

def fraction : ℚ := 805 / 0.410

def options : List ℚ := [0.4, 4, 40, 400, 4000]

theorem closest_to_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |fraction - x| ≤ |fraction - y| :=
by sorry

end closest_to_fraction_l1136_113632


namespace find_b_l1136_113698

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the closed interval [2, 2b]
def interval (b : ℝ) : Set ℝ := Set.Icc 2 (2*b)

-- Theorem statement
theorem find_b : 
  ∃ (b : ℝ), b > 1 ∧ 
  (∀ x ∈ interval b, f x ∈ interval b) ∧
  (∀ y ∈ interval b, ∃ x ∈ interval b, f x = y) ∧
  b = 2 := by
  sorry

end find_b_l1136_113698
