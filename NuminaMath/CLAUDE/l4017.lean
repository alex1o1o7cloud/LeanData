import Mathlib

namespace triangle_tangent_product_l4017_401703

theorem triangle_tangent_product (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →  -- Sum of angles in a triangle
  (a > 0) → (b > 0) → (c > 0) →  -- Positive side lengths
  (a / (2 * Real.sin (A / 2)) = b / (2 * Real.sin (B / 2))) →  -- Sine law
  (b / (2 * Real.sin (B / 2)) = c / (2 * Real.sin (C / 2))) →  -- Sine law
  (a + c = 2 * b) →  -- Given condition
  (Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3) :=
by sorry

end triangle_tangent_product_l4017_401703


namespace integral_p_equals_one_l4017_401700

noncomputable def p (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else α * Real.exp (-α * x)

theorem integral_p_equals_one (α : ℝ) (h : α > 0) :
  ∫ (x : ℝ), p α x = 1 := by sorry

end integral_p_equals_one_l4017_401700


namespace range_of_positive_integers_in_list_l4017_401795

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List Int) : List Int :=
  l.filter (λ x => x > 0)

def range (l : List Int) : Int :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list (k : List Int) :
  k = consecutive_integers (-4) 10 →
  range (positive_integers k) = 4 := by
  sorry

end range_of_positive_integers_in_list_l4017_401795


namespace triangle_roots_condition_l4017_401760

/-- Given a cubic polynomial x^3 - ux^2 + vx - w with roots a, b, and c forming a triangle, 
    prove that uv > 2w -/
theorem triangle_roots_condition (u v w a b c : ℝ) : 
  (∀ x, x^3 - u*x^2 + v*x - w = (x - a)*(x - b)*(x - c)) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  uv > 2*w :=
sorry

end triangle_roots_condition_l4017_401760


namespace small_cubes_count_l4017_401727

/-- Given a cube with edge length 9 cm cut into smaller cubes with edge length 3 cm,
    the number of small cubes obtained is 27. -/
theorem small_cubes_count (large_edge : ℕ) (small_edge : ℕ) : 
  large_edge = 9 → small_edge = 3 → (large_edge / small_edge) ^ 3 = 27 := by
  sorry

end small_cubes_count_l4017_401727


namespace raffle_ticket_cost_l4017_401764

theorem raffle_ticket_cost (x : ℚ) : 
  (25 * x + 30 + 20 = 100) → x = 2 := by
  sorry

end raffle_ticket_cost_l4017_401764


namespace finite_sequence_k_value_l4017_401711

/-- A finite sequence with k terms satisfying the given conditions -/
def FiniteSequence (k : ℕ) (a : ℕ → ℝ) : Prop :=
  (∀ n ∈ Finset.range (k - 2), a (n + 2) = a n - (n + 1) / a (n + 1)) ∧
  a 1 = 24 ∧
  a 2 = 51 ∧
  a k = 0

/-- The theorem stating that k must be 50 for the given conditions -/
theorem finite_sequence_k_value :
  ∀ k : ℕ, ∀ a : ℕ → ℝ, FiniteSequence k a → k = 50 :=
by
  sorry

end finite_sequence_k_value_l4017_401711


namespace min_value_theorem_l4017_401706

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (z : ℝ), (y/x) + (3/y) ≥ z → z ≥ m :=
by sorry

end min_value_theorem_l4017_401706


namespace equation_solution_l4017_401761

theorem equation_solution : 
  ∃ x : ℝ, (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + x + (0.9 : ℝ)^2 = 0.2999999999999999 ∧ 
  x = -1.73175 := by
  sorry

end equation_solution_l4017_401761


namespace longest_segment_in_cylinder_l4017_401724

def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 12

theorem longest_segment_in_cylinder :
  let diameter := 2 * cylinder_radius
  let longest_segment := Real.sqrt (cylinder_height ^ 2 + diameter ^ 2)
  longest_segment = Real.sqrt 244 := by
  sorry

end longest_segment_in_cylinder_l4017_401724


namespace mary_warmth_duration_l4017_401704

/-- The number of sticks of wood produced by chopping up furniture -/
def sticksFromFurniture (chairs tables stools : ℕ) : ℕ :=
  6 * chairs + 9 * tables + 2 * stools

/-- The number of hours Mary can keep warm given a certain amount of wood -/
def hoursWarm (totalSticks burningRate : ℕ) : ℕ :=
  totalSticks / burningRate

/-- Theorem: Mary can keep warm for 34 hours with the wood from 18 chairs, 6 tables, and 4 stools -/
theorem mary_warmth_duration :
  let totalSticks := sticksFromFurniture 18 6 4
  let burningRate := 5
  hoursWarm totalSticks burningRate = 34 := by
  sorry

end mary_warmth_duration_l4017_401704


namespace min_value_a_plus_b_l4017_401773

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 2 * x + y → x + y ≥ 2 * Real.sqrt 2 + 3 :=
by sorry

end min_value_a_plus_b_l4017_401773


namespace retailer_profit_percentage_l4017_401762

/-- Calculates the overall profit percentage for a retailer selling three items -/
theorem retailer_profit_percentage
  (radio_cost watch_cost phone_cost : ℚ)
  (radio_overhead watch_overhead phone_overhead : ℚ)
  (radio_sp watch_sp phone_sp : ℚ)
  (h_radio_cost : radio_cost = 225)
  (h_watch_cost : watch_cost = 425)
  (h_phone_cost : phone_cost = 650)
  (h_radio_overhead : radio_overhead = 15)
  (h_watch_overhead : watch_overhead = 20)
  (h_phone_overhead : phone_overhead = 30)
  (h_radio_sp : radio_sp = 300)
  (h_watch_sp : watch_sp = 525)
  (h_phone_sp : phone_sp = 800) :
  let total_cp := radio_cost + watch_cost + phone_cost + radio_overhead + watch_overhead + phone_overhead
  let total_sp := radio_sp + watch_sp + phone_sp
  let profit := total_sp - total_cp
  let profit_percentage := (profit / total_cp) * 100
  ∃ ε > 0, |profit_percentage - 19.05| < ε :=
by sorry

end retailer_profit_percentage_l4017_401762


namespace product_of_integers_with_lcm_and_gcd_l4017_401780

theorem product_of_integers_with_lcm_and_gcd (a b : ℕ+) : 
  Nat.lcm a b = 72 → 
  Nat.gcd a b = 8 → 
  (a = 4 * Nat.gcd a b ∨ b = 4 * Nat.gcd a b) → 
  a * b = 576 := by
sorry

end product_of_integers_with_lcm_and_gcd_l4017_401780


namespace compare_sqrt_expressions_l4017_401794

theorem compare_sqrt_expressions : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := by
  sorry

end compare_sqrt_expressions_l4017_401794


namespace projectile_speed_calculation_l4017_401730

/-- 
Given two projectiles launched simultaneously 1455 km apart, with one traveling at 500 km/h,
prove that the speed of the other projectile is 470 km/h if they meet after 90 minutes.
-/
theorem projectile_speed_calculation (distance : ℝ) (time : ℝ) (speed2 : ℝ) (speed1 : ℝ) : 
  distance = 1455 → 
  time = 1.5 → 
  speed2 = 500 → 
  speed1 = 470 → 
  distance = (speed1 + speed2) * time :=
by sorry

end projectile_speed_calculation_l4017_401730


namespace jerry_logs_count_l4017_401721

/-- The number of logs produced by a pine tree -/
def logsPerPine : ℕ := 80

/-- The number of logs produced by a maple tree -/
def logsPerMaple : ℕ := 60

/-- The number of logs produced by a walnut tree -/
def logsPerWalnut : ℕ := 100

/-- The number of pine trees Jerry cuts -/
def pineTreesCut : ℕ := 8

/-- The number of maple trees Jerry cuts -/
def mapleTreesCut : ℕ := 3

/-- The number of walnut trees Jerry cuts -/
def walnutTreesCut : ℕ := 4

/-- The total number of logs Jerry gets -/
def totalLogs : ℕ := logsPerPine * pineTreesCut + logsPerMaple * mapleTreesCut + logsPerWalnut * walnutTreesCut

theorem jerry_logs_count : totalLogs = 1220 := by sorry

end jerry_logs_count_l4017_401721


namespace isosceles_triangle_perimeter_l4017_401742

/-- An isosceles triangle with sides of length 8 and 3 has a perimeter of 19 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 ∧ b = 8 ∧ c = 3 →
  a + b > c ∧ b + c > a ∧ a + c > b →
  a + b + c = 19 := by
  sorry

end isosceles_triangle_perimeter_l4017_401742


namespace meat_distribution_l4017_401705

/-- Proves the correct distribution of meat between two pots -/
theorem meat_distribution (pot1 pot2 total_meat : ℕ) 
  (h1 : pot1 = 645)
  (h2 : pot2 = 237)
  (h3 : total_meat = 1000) :
  ∃ (meat1 meat2 : ℕ),
    meat1 + meat2 = total_meat ∧
    pot1 + meat1 = pot2 + meat2 ∧
    meat1 = 296 ∧
    meat2 = 704 := by
  sorry

end meat_distribution_l4017_401705


namespace fraction_problem_l4017_401738

theorem fraction_problem (x y : ℚ) (h1 : x + y = 3/4) (h2 : x * y = 1/8) : 
  min x y = 1/4 := by
  sorry

end fraction_problem_l4017_401738


namespace range_of_f_l4017_401759

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -18 ≤ y ∧ y ≤ 2} :=
sorry

end range_of_f_l4017_401759


namespace parabola_directrix_l4017_401767

/-- Given a parabola with equation y = ax² and directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (2 : ℝ) = -1 / (4 * a) →    -- Directrix equation (in standard form)
  a = -1/8 := by
sorry

end parabola_directrix_l4017_401767


namespace factorization_xy_squared_minus_x_l4017_401778

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y + 1) * (y - 1) := by
  sorry

end factorization_xy_squared_minus_x_l4017_401778


namespace vector_difference_magnitude_l4017_401791

/-- Given vectors a and b in R², prove that |a - b| = 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end vector_difference_magnitude_l4017_401791


namespace final_water_level_l4017_401790

/-- The final water level in a system of two connected cylindrical vessels -/
theorem final_water_level 
  (h : ℝ) -- Initial height of both liquids
  (ρ_water : ℝ) -- Density of water
  (ρ_oil : ℝ) -- Density of oil
  (h_pos : h > 0)
  (ρ_water_pos : ρ_water > 0)
  (ρ_oil_pos : ρ_oil > 0)
  (h_val : h = 40)
  (ρ_water_val : ρ_water = 1000)
  (ρ_oil_val : ρ_oil = 700) :
  ∃ (h_water : ℝ), h_water = 280 / 17 ∧ 
    ρ_water * h_water = ρ_oil * (h - h_water) ∧
    h_water > 0 ∧ h_water < h :=
by
  sorry


end final_water_level_l4017_401790


namespace correct_sampling_methods_l4017_401740

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a population with subgroups -/
structure Population where
  total : Nat
  subgroups : List Nat
  sample_size : Nat

/-- Represents a simple population without subgroups -/
structure SimplePopulation where
  total : Nat
  sample_size : Nat

def student_population : Population :=
  { total := 1200
  , subgroups := [400, 600, 200]
  , sample_size := 120 }

def parent_population : SimplePopulation :=
  { total := 10
  , sample_size := 3 }

/-- Determines the best sampling method for a given population -/
def best_sampling_method (pop : Population) : SamplingMethod :=
  sorry

/-- Determines the best sampling method for a simple population -/
def best_simple_sampling_method (pop : SimplePopulation) : SamplingMethod :=
  sorry

theorem correct_sampling_methods :
  (best_sampling_method student_population = SamplingMethod.Stratified) ∧
  (best_simple_sampling_method parent_population = SamplingMethod.SimpleRandom) :=
sorry

end correct_sampling_methods_l4017_401740


namespace greatest_three_digit_multiple_of_17_l4017_401787

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≤ 999 ∧ n ≥ 100 ∧ 17 ∣ n → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l4017_401787


namespace max_sides_diagonal_polygon_13gon_l4017_401745

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a convex n-gon -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon formed by diagonals of a larger polygon -/
structure DiagonalPolygon (n : ℕ) where
  sides : ℕ
  sides_le : sides ≤ n

/-- Theorem: In a convex 13-gon with all diagonals drawn, 
    the maximum number of sides of any polygon formed by these diagonals is 13 -/
theorem max_sides_diagonal_polygon_13gon :
  ∀ (p : ConvexPolygon 13) (d : DiagonalPolygon 13),
    d.sides ≤ 13 ∧ ∃ (d' : DiagonalPolygon 13), d'.sides = 13 :=
sorry

end max_sides_diagonal_polygon_13gon_l4017_401745


namespace smallest_product_l4017_401701

def digits : List Nat := [7, 8, 9, 10]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 63990 :=
by sorry

end smallest_product_l4017_401701


namespace area_between_concentric_circles_l4017_401757

theorem area_between_concentric_circles (R : ℝ) (chord_length : ℝ) : 
  R = 13 → 
  chord_length = 24 → 
  (π * R^2) - (π * (R^2 - (chord_length/2)^2)) = 144 * π :=
by
  sorry

end area_between_concentric_circles_l4017_401757


namespace visited_neither_country_l4017_401763

theorem visited_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 250 →
  iceland = 125 →
  norway = 95 →
  both = 80 →
  total - ((iceland + norway) - both) = 110 :=
by sorry

end visited_neither_country_l4017_401763


namespace new_students_calculation_l4017_401786

/-- Proves that the number of new students is equal to the final number minus
    the difference between the initial number and the number who left. -/
theorem new_students_calculation
  (initial_students : ℕ)
  (students_left : ℕ)
  (final_students : ℕ)
  (h1 : initial_students = 8)
  (h2 : students_left = 5)
  (h3 : final_students = 11) :
  final_students - (initial_students - students_left) = 8 :=
by sorry

end new_students_calculation_l4017_401786


namespace game_theorists_board_size_l4017_401755

/-- Represents the voting process for the game theorists' leadership board. -/
def BoardVotingProcess (initial_members : ℕ) : Prop :=
  ∃ (final_members : ℕ),
    -- The final number of members is less than or equal to the initial number
    final_members ≤ initial_members ∧
    -- The final number of members is of the form 2^n - 1
    ∃ (n : ℕ), final_members = 2^n - 1 ∧
    -- There is no larger number of the form 2^m - 1 that's less than or equal to the initial number
    ∀ (m : ℕ), 2^m - 1 ≤ initial_members → m ≤ n

/-- The theorem stating the result of the voting process for 2020 initial members. -/
theorem game_theorists_board_size :
  BoardVotingProcess 2020 → ∃ (final_members : ℕ), final_members = 1023 :=
by
  sorry


end game_theorists_board_size_l4017_401755


namespace rain_probability_l4017_401782

theorem rain_probability (M T N : ℝ) 
  (hM : M = 0.6)  -- 60% of counties received rain on Monday
  (hT : T = 0.55) -- 55% of counties received rain on Tuesday
  (hN : N = 0.25) -- 25% of counties received no rain on either day
  : M + T - N - 1 = 0.4 := by
  sorry

end rain_probability_l4017_401782


namespace pure_imaginary_solutions_of_polynomial_l4017_401749

theorem pure_imaginary_solutions_of_polynomial :
  let p (x : ℂ) := x^4 - 4*x^3 + 10*x^2 - 40*x - 100
  ∀ x : ℂ, (∃ a : ℝ, x = Complex.I * a) ∧ p x = 0 ↔ x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end pure_imaginary_solutions_of_polynomial_l4017_401749


namespace discount_difference_l4017_401736

def bill_amount : ℝ := 12000
def single_discount : ℝ := 0.42
def first_successive_discount : ℝ := 0.35
def second_successive_discount : ℝ := 0.05

def single_discounted_amount : ℝ := bill_amount * (1 - single_discount)
def successive_discounted_amount : ℝ := bill_amount * (1 - first_successive_discount) * (1 - second_successive_discount)

theorem discount_difference :
  successive_discounted_amount - single_discounted_amount = 450 := by
  sorry

end discount_difference_l4017_401736


namespace rectangle_area_proof_l4017_401752

theorem rectangle_area_proof : ∃ (x y : ℚ), 
  (x - (7/2)) * (y + (3/2)) = x * y ∧ 
  (x + (7/2)) * (y - (5/2)) = x * y ∧ 
  x * y = 20/7 := by
  sorry

end rectangle_area_proof_l4017_401752


namespace arc_length_from_central_angle_l4017_401781

/-- Given a circle with circumference 80 feet and an arc subtended by a central angle of 120°,
    the length of this arc is 80/3 feet. -/
theorem arc_length_from_central_angle (circle : Real) (arc : Real) :
  (circle = 80) →  -- circumference of the circle is 80 feet
  (arc = 120 / 360 * circle) →  -- arc is subtended by a 120° angle
  (arc = 80 / 3) :=  -- length of the arc is 80/3 feet
by sorry

end arc_length_from_central_angle_l4017_401781


namespace stream_speed_l4017_401743

/-- The speed of a stream given boat travel times and distances -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 84) (h2 : upstream_distance = 48) 
  (h3 : time = 2) : ∃ s : ℝ, s = 9 ∧ 
  ∃ b : ℝ, downstream_distance = (b + s) * time ∧ 
           upstream_distance = (b - s) * time :=
by sorry

end stream_speed_l4017_401743


namespace wash_time_is_three_hours_l4017_401746

/-- The number of hours required to wash all clothes given the number of items and washing machine capacity -/
def wash_time (shirts pants sweaters jeans : ℕ) (max_items_per_cycle : ℕ) (minutes_per_cycle : ℕ) : ℚ :=
  let total_items := shirts + pants + sweaters + jeans
  let num_cycles := (total_items + max_items_per_cycle - 1) / max_items_per_cycle
  (num_cycles * minutes_per_cycle : ℚ) / 60

/-- Theorem stating that it takes 3 hours to wash all the clothes under given conditions -/
theorem wash_time_is_three_hours :
  wash_time 18 12 17 13 15 45 = 3 := by
  sorry

end wash_time_is_three_hours_l4017_401746


namespace power_of_2016_expression_evaluation_l4017_401719

-- Part 1
theorem power_of_2016 (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^(m+4*n) = 324) : 
  2016^n = 2016 := by sorry

-- Part 2
theorem expression_evaluation (a : ℝ) (h : a = 5) : 
  (a+2)*(a-2) + a*(1-a) = 1 := by sorry

end power_of_2016_expression_evaluation_l4017_401719


namespace shooting_competition_probabilities_l4017_401753

-- Define the probabilities for A and B hitting different rings
def prob_A_8 : ℝ := 0.6
def prob_A_9 : ℝ := 0.3
def prob_A_10 : ℝ := 0.1
def prob_B_8 : ℝ := 0.4
def prob_B_9 : ℝ := 0.4
def prob_B_10 : ℝ := 0.2

-- Define the probability that A hits more rings than B in a single round
def prob_A_beats_B : ℝ := prob_A_9 * prob_B_8 + prob_A_10 * prob_B_8 + prob_A_10 * prob_B_9

-- Define the probability that A hits more rings than B in at least two out of three independent rounds
def prob_A_beats_B_twice_or_more : ℝ :=
  3 * prob_A_beats_B^2 * (1 - prob_A_beats_B) + prob_A_beats_B^3

theorem shooting_competition_probabilities :
  prob_A_beats_B = 0.2 ∧ prob_A_beats_B_twice_or_more = 0.104 := by
  sorry

end shooting_competition_probabilities_l4017_401753


namespace tangent_sum_simplification_l4017_401799

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (40 * π / 180)) / Real.cos (10 * π / 180) =
  (1/2 + Real.cos (20 * π / 180)^2) / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180)) := by
  sorry

end tangent_sum_simplification_l4017_401799


namespace sheila_hourly_wage_l4017_401788

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_weekly_hours schedule)

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 8
  , tue_thu_hours := 6
  , weekly_earnings := 504 }

/-- Theorem stating that Sheila's hourly wage is $14 --/
theorem sheila_hourly_wage :
  hourly_wage sheila_schedule = 14 := by
  sorry


end sheila_hourly_wage_l4017_401788


namespace maria_receives_no_funds_main_result_l4017_401731

/-- Represents the deposit insurance system in rubles -/
def deposit_insurance_threshold : ℕ := 1600000

/-- Represents Maria's deposit amount in rubles -/
def maria_deposit : ℕ := 0  -- We don't know the exact amount, so we use 0 as a placeholder

/-- Theorem stating that Maria will not receive any funds -/
theorem maria_receives_no_funds (h : maria_deposit < deposit_insurance_threshold) :
  maria_deposit = 0 := by
  sorry

/-- Main theorem combining the conditions and the result -/
theorem main_result : 
  maria_deposit < deposit_insurance_threshold → maria_deposit = 0 := by
  sorry

end maria_receives_no_funds_main_result_l4017_401731


namespace even_increasing_ordering_l4017_401758

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f is increasing on (0, +∞) if x < y implies f(x) < f(y) for all x, y > 0 -/
def IncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem even_increasing_ordering (f : ℝ → ℝ) 
  (h_even : IsEven f) (h_incr : IncreasingOnPositive f) :
  f 3 < f (-Real.pi) ∧ f (-Real.pi) < f (-4) := by
  sorry

end even_increasing_ordering_l4017_401758


namespace email_problem_l4017_401766

theorem email_problem (x : ℚ) : 
  x + x/2 + x/4 + x/8 = 30 → x = 16 := by
  sorry

end email_problem_l4017_401766


namespace pr_cr_relation_l4017_401768

theorem pr_cr_relation (p c : ℝ) :
  (6 * p * 4 = 360) → (p = 15 ∧ 6 * c * 4 = 24 * c) := by
  sorry

end pr_cr_relation_l4017_401768


namespace department_store_discount_rate_l4017_401754

/-- Represents the discount rate calculation for a department store purchase --/
theorem department_store_discount_rate : 
  -- Define the prices of items
  let shoe_price : ℚ := 74
  let sock_price : ℚ := 2
  let bag_price : ℚ := 42
  let sock_quantity : ℕ := 2
  
  -- Calculate total price before discount
  let total_before_discount : ℚ := shoe_price + sock_price * sock_quantity + bag_price
  
  -- Define the threshold for discount application
  let discount_threshold : ℚ := 100
  
  -- Define the amount paid by Jaco
  let amount_paid : ℚ := 118
  
  -- Calculate the discount amount
  let discount_amount : ℚ := total_before_discount - amount_paid
  
  -- Calculate the amount subject to discount
  let amount_subject_to_discount : ℚ := total_before_discount - discount_threshold
  
  -- Calculate the discount rate
  let discount_rate : ℚ := discount_amount / amount_subject_to_discount * 100
  
  discount_rate = 10 := by sorry

end department_store_discount_rate_l4017_401754


namespace problem_statement_l4017_401718

theorem problem_statement : 
  (-1)^2023 + (8 : ℝ)^(1/3) - 2 * (1/4 : ℝ)^(1/2) + |Real.sqrt 3 - 2| = 2 - Real.sqrt 3 := by
  sorry

end problem_statement_l4017_401718


namespace stratified_sampling_major_c_l4017_401707

/-- Represents the number of students to be sampled from a major -/
def sampleSize (totalStudents : ℕ) (sampleTotal : ℕ) (majorStudents : ℕ) : ℕ :=
  (sampleTotal * majorStudents) / totalStudents

/-- Proves that the number of students to be drawn from major C is 40 -/
theorem stratified_sampling_major_c :
  let totalStudents : ℕ := 1200
  let sampleTotal : ℕ := 120
  let majorAStudents : ℕ := 380
  let majorBStudents : ℕ := 420
  let majorCStudents : ℕ := totalStudents - majorAStudents - majorBStudents
  sampleSize totalStudents sampleTotal majorCStudents = 40 := by
  sorry

end stratified_sampling_major_c_l4017_401707


namespace sphere_radius_and_area_l4017_401776

/-- A sphere with a chord creating a hollow on its surface -/
structure SphereWithHollow where
  radius : ℝ
  hollowDiameter : ℝ
  hollowDepth : ℝ

/-- The theorem about the sphere's radius and surface area given the hollow dimensions -/
theorem sphere_radius_and_area (s : SphereWithHollow) 
  (h1 : s.hollowDiameter = 12)
  (h2 : s.hollowDepth = 2) :
  s.radius = 10 ∧ 4 * Real.pi * s.radius^2 = 400 * Real.pi := by
  sorry

#check sphere_radius_and_area

end sphere_radius_and_area_l4017_401776


namespace existence_of_integers_l4017_401735

theorem existence_of_integers (m : ℕ) (hm : m > 0) :
  ∃ (a b : ℤ),
    (abs a ≤ m) ∧
    (abs b ≤ m) ∧
    (0 < a + b * Real.sqrt 2) ∧
    (a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2)) := by
  sorry

end existence_of_integers_l4017_401735


namespace stratified_sample_sum_l4017_401723

/-- Represents the number of varieties in a category -/
structure Category where
  varieties : ℕ

/-- Represents the total population of varieties -/
def total_population (categories : List Category) : ℕ :=
  categories.map (·.varieties) |> List.sum

/-- Calculates the number of items in a stratified sample for a given category -/
def stratified_sample_size (category : Category) (total_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (category.varieties * sample_size) / total_pop

/-- Theorem: The sum of vegetable oils and fruits/vegetables in a stratified sample is 6 -/
theorem stratified_sample_sum (vegetable_oils fruits_vegetables : Category)
    (h1 : vegetable_oils.varieties = 10)
    (h2 : fruits_vegetables.varieties = 20)
    (h3 : total_population [vegetable_oils, fruits_vegetables] = 30)
    (h4 : total_population [Category.mk 40, vegetable_oils, Category.mk 30, fruits_vegetables] = 100) :
    stratified_sample_size vegetable_oils 100 20 + stratified_sample_size fruits_vegetables 100 20 = 6 := by
  sorry


end stratified_sample_sum_l4017_401723


namespace binomial_8_3_l4017_401779

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end binomial_8_3_l4017_401779


namespace mean_temperature_is_85_point_6_l4017_401726

def temperatures : List ℝ := [85, 84, 85, 83, 82, 84, 86, 88, 90, 89]

theorem mean_temperature_is_85_point_6 :
  (List.sum temperatures) / (List.length temperatures) = 85.6 := by
  sorry

end mean_temperature_is_85_point_6_l4017_401726


namespace amount_lent_to_C_l4017_401713

/-- The amount of money A lent to B in rupees -/
def amount_B : ℝ := 5000

/-- The duration of B's loan in years -/
def duration_B : ℝ := 2

/-- The duration of C's loan in years -/
def duration_C : ℝ := 4

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.07000000000000001

/-- The total interest received from both B and C in rupees -/
def total_interest : ℝ := 1540

/-- The amount of money A lent to C in rupees -/
def amount_C : ℝ := 3000

/-- Theorem stating that given the conditions, A lent 3000 rupees to C -/
theorem amount_lent_to_C : 
  amount_B * interest_rate * duration_B + 
  amount_C * interest_rate * duration_C = total_interest :=
by sorry

end amount_lent_to_C_l4017_401713


namespace units_digit_sum_of_powers_l4017_401792

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to calculate the units digit of a^n
def powerUnitsDigit (a n : ℕ) : ℕ :=
  unitsDigit ((unitsDigit a)^n)

theorem units_digit_sum_of_powers : 
  unitsDigit ((35 : ℕ)^87 + (93 : ℕ)^53) = 8 := by sorry

end units_digit_sum_of_powers_l4017_401792


namespace b_is_ten_l4017_401797

/-- The base of the number system that satisfies the given equation -/
def b : ℕ := sorry

/-- The equation that b must satisfy -/
axiom eq_condition : (3 * b + 5)^2 = 1 * b^3 + 2 * b^2 + 2 * b + 5

/-- Proof that b is the only positive integer solution -/
theorem b_is_ten : b = 10 := by sorry

end b_is_ten_l4017_401797


namespace distance_AB_when_parallel_coordinates_C_when_perpendicular_l4017_401750

-- Define the points A, B, C in the Cartesian coordinate system
def A (a : ℝ) : ℝ × ℝ := (-2, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 4)
def C (b : ℝ) : ℝ × ℝ := (b - 2, b)

-- Define the condition that AB is parallel to x-axis
def AB_parallel_x (a : ℝ) : Prop := (A a).2 = (B a).2

-- Define the condition that CD is perpendicular to x-axis
def CD_perpendicular_x (b : ℝ) : Prop := (C b).1 = b - 2

-- Define the condition that CD = 1
def CD_length_1 (b : ℝ) : Prop := (C b).2 - 0 = 1 ∨ (C b).2 - 0 = -1

-- Theorem for part 1
theorem distance_AB_when_parallel (a : ℝ) :
  AB_parallel_x a → (B a).1 - (A a).1 = 4 :=
sorry

-- Theorem for part 2
theorem coordinates_C_when_perpendicular (b : ℝ) :
  CD_perpendicular_x b ∧ CD_length_1 b →
  C b = (-1, 1) ∨ C b = (-3, -1) :=
sorry

end distance_AB_when_parallel_coordinates_C_when_perpendicular_l4017_401750


namespace sum_and_divide_theorem_l4017_401712

theorem sum_and_divide_theorem (n a : ℕ) (ha : a > 1) :
  let sum := (n * (n + 1)) / 2 - (n / a) * ((n / a) * a + a) / 2
  sum / (a * (a - 1) / 2) = (n / a)^2 := by
  sorry

end sum_and_divide_theorem_l4017_401712


namespace correct_delivery_probability_l4017_401716

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- Probability of exactly k out of n packages being delivered to their correct houses -/
def probability_correct_delivery (n k : ℕ) : ℚ :=
  (n.choose k * (k.factorial : ℚ) * ((n - k).factorial : ℚ)) / (n.factorial : ℚ)

/-- Theorem stating the probability of exactly 3 out of 5 packages being delivered correctly -/
theorem correct_delivery_probability :
  probability_correct_delivery n k = 1 / 6 := by
  sorry


end correct_delivery_probability_l4017_401716


namespace trout_division_l4017_401796

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 18 → num_people = 2 → trout_per_person = total_trout / num_people → trout_per_person = 9 := by
  sorry

end trout_division_l4017_401796


namespace no_integer_solution_l4017_401717

theorem no_integer_solution : ¬ ∃ (m n : ℤ), m^3 = 4*n + 2 := by
  sorry

end no_integer_solution_l4017_401717


namespace library_books_checkout_l4017_401784

theorem library_books_checkout (total : ℕ) (ratio_nf : ℕ) (ratio_f : ℕ) (h1 : total = 52) (h2 : ratio_nf = 7) (h3 : ratio_f = 6) : 
  (total * ratio_f) / (ratio_nf + ratio_f) = 24 := by
  sorry

end library_books_checkout_l4017_401784


namespace inverse_x_equals_three_l4017_401789

theorem inverse_x_equals_three (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + 1/27 = x*y) : 1/x = 3 := by
  sorry

end inverse_x_equals_three_l4017_401789


namespace rational_numbers_include_integers_and_fractions_l4017_401734

theorem rational_numbers_include_integers_and_fractions : 
  (∀ n : ℤ, ∃ q : ℚ, (n : ℚ) = q) ∧ 
  (∀ a b : ℤ, b ≠ 0 → ∃ q : ℚ, (a : ℚ) / (b : ℚ) = q) :=
sorry

end rational_numbers_include_integers_and_fractions_l4017_401734


namespace a_minus_b_value_l4017_401715

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem a_minus_b_value (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  a - b = 29/4 :=
by sorry

end a_minus_b_value_l4017_401715


namespace distributive_law_example_l4017_401748

theorem distributive_law_example :
  (7 + 125) * 8 = 7 * 8 + 125 * 8 := by sorry

end distributive_law_example_l4017_401748


namespace smallest_base_sum_l4017_401756

theorem smallest_base_sum : ∃ (c d : ℕ), 
  c ≠ d ∧ 
  c > 9 ∧ 
  d > 9 ∧ 
  8 * c + 9 = 9 * d + 8 ∧ 
  c + d = 19 ∧ 
  (∀ (c' d' : ℕ), c' ≠ d' → c' > 9 → d' > 9 → 8 * c' + 9 = 9 * d' + 8 → c' + d' ≥ 19) :=
by sorry

end smallest_base_sum_l4017_401756


namespace representation_of_1917_l4017_401765

theorem representation_of_1917 : ∃ (a b c : ℤ), 1917 = a^2 - b^2 + c^2 := by
  sorry

end representation_of_1917_l4017_401765


namespace managers_salary_l4017_401722

/-- Given an organization with employees and their salaries, this theorem proves
    the salary of an additional member that would increase the average by a specific amount. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 20 →
  avg_salary = 1700 →
  avg_increase = 100 →
  (num_employees * avg_salary + 3800) / (num_employees + 1) = avg_salary + avg_increase := by
  sorry

#check managers_salary

end managers_salary_l4017_401722


namespace square_in_S_l4017_401725

def S : Set ℕ := {n | ∃ a b c d e f : ℕ, 
  (n - 1 = a^2 + b^2) ∧ 
  (n = c^2 + d^2) ∧ 
  (n + 1 = e^2 + f^2) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0)}

theorem square_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S := by
  sorry

end square_in_S_l4017_401725


namespace power_difference_seven_l4017_401744

theorem power_difference_seven (n k : ℕ) : 2^n - 5^k = 7 ↔ n = 5 ∧ k = 2 := by
  sorry

end power_difference_seven_l4017_401744


namespace quadratic_roots_distance_l4017_401709

theorem quadratic_roots_distance (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (x₁ - x₂)^2 = 1) →
  p = 3 ∧ q = 2 := by
sorry

end quadratic_roots_distance_l4017_401709


namespace student_marks_l4017_401747

theorem student_marks (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 20 →
  M + P = 20 :=
by sorry

end student_marks_l4017_401747


namespace financial_audit_equation_l4017_401702

theorem financial_audit_equation (p v : ℂ) : 
  (7 * p - v = 23000) → (v = 50 + 250 * Complex.I) → 
  (p = 3292.857 + 35.714 * Complex.I) := by
sorry

end financial_audit_equation_l4017_401702


namespace volume_ratio_in_partitioned_cube_l4017_401733

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  edgeLength : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- Calculates the volume of the part of the cube on one side of a plane -/
noncomputable def volumePartition (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem: The ratio of volumes in a cube partitioned by a specific plane -/
theorem volume_ratio_in_partitioned_cube (c : Cube) (e f : Point3D) : 
  let p := Plane.mk 1 1 1 0  -- Placeholder plane, actual coefficients would depend on B, E, F
  volumePartition c p / cubeVolume c = 25 / 72 := by sorry

end volume_ratio_in_partitioned_cube_l4017_401733


namespace other_rectangle_perimeter_l4017_401710

/-- Represents the perimeter of a rectangle --/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Represents the side length of the original square --/
def square_side : ℝ := 5

/-- Represents the perimeter of the first rectangle --/
def first_rectangle_perimeter : ℝ := 16

theorem other_rectangle_perimeter :
  ∀ (l w : ℝ),
  l + w = square_side →
  rectangle_perimeter l w = first_rectangle_perimeter →
  rectangle_perimeter square_side (square_side - w) = 14 :=
by sorry

end other_rectangle_perimeter_l4017_401710


namespace pants_to_shirts_ratio_l4017_401770

/-- Proves that the ratio of pants to shirts is 1/2 given the problem conditions --/
theorem pants_to_shirts_ratio :
  ∀ (num_pants : ℕ),
  (10 * 6 + num_pants * 8 = 100) →
  (num_pants : ℚ) / 10 = 1 / 2 := by
  sorry

end pants_to_shirts_ratio_l4017_401770


namespace tan_theta_minus_pi_over_four_l4017_401774

theorem tan_theta_minus_pi_over_four (θ : ℝ) :
  (Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)).re = 0 →
  Real.tan (θ - π/4) = -7 := by
  sorry

end tan_theta_minus_pi_over_four_l4017_401774


namespace joshua_crates_count_l4017_401714

def bottles_per_crate : ℕ := 12
def total_bottles : ℕ := 130
def unpacked_bottles : ℕ := 10

theorem joshua_crates_count :
  (total_bottles - unpacked_bottles) / bottles_per_crate = 10 := by
  sorry

end joshua_crates_count_l4017_401714


namespace boat_upstream_distance_l4017_401785

/-- Proves that a boat traveling 11 km downstream in one hour with a still water speed of 8 km/h
    will travel 5 km upstream in one hour. -/
theorem boat_upstream_distance
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (h1 : boat_speed = 8)
  (h2 : downstream_distance = 11) :
  boat_speed - (downstream_distance - boat_speed) = 5 :=
by
  sorry


end boat_upstream_distance_l4017_401785


namespace geometric_sequence_problem_l4017_401775

theorem geometric_sequence_problem :
  ∀ (a b c d : ℝ),
    (∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
    a + d = 20 →  -- sum of extreme terms
    b + c = 34 →  -- sum of middle terms
    a^2 + b^2 + c^2 + d^2 = 1300 →  -- sum of squares
    ((a = 16 ∧ b = 8 ∧ c = 4 ∧ d = 2) ∨ (a = 4 ∧ b = 8 ∧ c = 16 ∧ d = 32)) :=
by sorry

end geometric_sequence_problem_l4017_401775


namespace m_eq_one_sufficient_not_necessary_l4017_401777

-- Define the lines l1 and l2 as functions of x and y
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + y + 3 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (3 * m - 2) * x + m * y + 2 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, l1 m x y ↔ l2 m x y

-- Theorem statement
theorem m_eq_one_sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → parallel m) ∧ 
  (∃ m : ℝ, m ≠ 1 ∧ parallel m) :=
sorry

end m_eq_one_sufficient_not_necessary_l4017_401777


namespace age_problem_l4017_401783

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 3 * d →
  a + b + c + d = 87 →
  b = 30 := by
sorry

end age_problem_l4017_401783


namespace cost_per_song_l4017_401741

/-- Calculates the cost per song given monthly music purchase, average song length, and annual expenditure -/
theorem cost_per_song 
  (monthly_hours : ℝ) 
  (song_length_minutes : ℝ) 
  (annual_cost : ℝ) 
  (h1 : monthly_hours = 20)
  (h2 : song_length_minutes = 3)
  (h3 : annual_cost = 2400) : 
  annual_cost / (monthly_hours * 12 * 60 / song_length_minutes) = 0.50 := by
  sorry

end cost_per_song_l4017_401741


namespace arithmetic_mean_of_30_and_18_l4017_401772

theorem arithmetic_mean_of_30_and_18 : (30 + 18) / 2 = 24 := by sorry

end arithmetic_mean_of_30_and_18_l4017_401772


namespace sqrt_meaningful_iff_leq_one_l4017_401720

-- Define the property of being a meaningful square root
def is_meaningful_sqrt (x : ℝ) : Prop := 1 - x ≥ 0

-- State the theorem
theorem sqrt_meaningful_iff_leq_one :
  ∀ x : ℝ, is_meaningful_sqrt x ↔ x ≤ 1 :=
by sorry

end sqrt_meaningful_iff_leq_one_l4017_401720


namespace product_of_cosines_l4017_401729

theorem product_of_cosines : 
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) * 
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 16 := by
  sorry

end product_of_cosines_l4017_401729


namespace max_expected_value_l4017_401769

/-- The probability of winning when there are n red balls and 5 white balls -/
def probability (n : ℕ) : ℚ :=
  (10 * n) / ((n + 5) * (n + 4))

/-- The expected value of the game when there are n red balls -/
def expected_value (n : ℕ) : ℚ :=
  2 * probability n - 1

/-- Theorem stating that the expected value is maximized when n is 4 or 5 -/
theorem max_expected_value :
  ∀ n : ℕ, n > 0 → (expected_value n ≤ expected_value 4 ∧ expected_value n ≤ expected_value 5) :=
by sorry

end max_expected_value_l4017_401769


namespace binomial_congruence_l4017_401751

theorem binomial_congruence (p m n : ℕ) (hp : Prime p) (h_mn : m ≥ n) :
  (Nat.choose (p * m) (p * n)) ≡ (Nat.choose m n) [MOD p] := by
  sorry

end binomial_congruence_l4017_401751


namespace max_distance_C_D_l4017_401798

open Complex

/-- The set of solutions to z^4 - 16 = 0 -/
def C : Set ℂ := {z : ℂ | z^4 - 16 = 0}

/-- The set of solutions to z^4 - 16z^3 + 48z^2 - 64z + 64 = 0 -/
def D : Set ℂ := {z : ℂ | z^4 - 16*z^3 + 48*z^2 - 64*z + 64 = 0}

/-- The maximum distance between any point in C and any point in D is 2 -/
theorem max_distance_C_D : 
  ∃ (c : C) (d : D), ∀ (c' : C) (d' : D), abs (c - d) ≥ abs (c' - d') ∧ abs (c - d) = 2 := by
  sorry

end max_distance_C_D_l4017_401798


namespace imaginary_part_of_complex_fraction_l4017_401739

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + Complex.I) / (1 - Complex.I)) = 1 :=
by sorry

end imaginary_part_of_complex_fraction_l4017_401739


namespace arithmetic_sequence_property_l4017_401771

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₈ = 12, then a₅ = 6. -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 2 + a 8 = 12) : a 5 = 6 := by
  sorry

end arithmetic_sequence_property_l4017_401771


namespace ellipse_properties_l4017_401708

/-- Represents an ellipse with semi-major axis a and semi-minor axis 2 -/
structure Ellipse where
  a : ℝ
  h : a > 2

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / 4 = 1

/-- The eccentricity of the ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The x-coordinate of the right focus -/
def rightFocusX (e : Ellipse) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and point P -/
theorem ellipse_properties (e : Ellipse) (P : PointOnEllipse e) 
  (h_dist : ∃ (F₁ F₂ : ℝ × ℝ), Real.sqrt ((P.x - F₁.1)^2 + (P.y - F₁.2)^2) + 
                                Real.sqrt ((P.x - F₂.1)^2 + (P.y - F₂.2)^2) = 6)
  (h_perp : ∃ (F₂_x : ℝ), P.x = F₂_x) :
  eccentricity e = Real.sqrt 5 / 3 ∧ 
  rightFocusX e = Real.sqrt 5 := by sorry

end ellipse_properties_l4017_401708


namespace log_sum_equals_two_l4017_401728

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end log_sum_equals_two_l4017_401728


namespace jonathan_first_name_length_l4017_401793

/-- The number of letters in Jonathan's first name -/
def jonathan_first_name : ℕ := by sorry

/-- The number of letters in Jonathan's surname -/
def jonathan_surname : ℕ := 10

/-- The number of letters in Jonathan's sister's first name -/
def sister_first_name : ℕ := 5

/-- The number of letters in Jonathan's sister's surname -/
def sister_surname : ℕ := 10

/-- The total number of letters in both their names -/
def total_letters : ℕ := 33

theorem jonathan_first_name_length :
  jonathan_first_name = 8 :=
by
  have h1 : jonathan_first_name + jonathan_surname + sister_first_name + sister_surname = total_letters := by sorry
  sorry

end jonathan_first_name_length_l4017_401793


namespace textbook_cost_l4017_401737

/-- Given a textbook sold by a bookstore, prove that the cost to the bookstore
    is $44 when the selling price is $55 and the profit is $11. -/
theorem textbook_cost (selling_price profit : ℕ) (h1 : selling_price = 55) (h2 : profit = 11) :
  selling_price - profit = 44 := by
  sorry

end textbook_cost_l4017_401737


namespace quadratic_roots_difference_squared_l4017_401732

theorem quadratic_roots_difference_squared :
  ∀ a b : ℝ, (2 * a^2 - 8 * a + 6 = 0) → (2 * b^2 - 8 * b + 6 = 0) → (a - b)^2 = 4 := by
sorry

end quadratic_roots_difference_squared_l4017_401732
