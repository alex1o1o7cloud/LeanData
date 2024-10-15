import Mathlib

namespace NUMINAMATH_GPT_number_of_boys_l609_60937

theorem number_of_boys (n : ℕ) (h1 : (n * 182 - 60) / n = 180): n = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l609_60937


namespace NUMINAMATH_GPT_kiyana_gives_half_l609_60988

theorem kiyana_gives_half (total_grapes : ℕ) (h : total_grapes = 24) : 
  (total_grapes / 2) = 12 :=
by
  sorry

end NUMINAMATH_GPT_kiyana_gives_half_l609_60988


namespace NUMINAMATH_GPT_required_additional_coins_l609_60960

-- Summing up to the first 15 natural numbers
def sum_first_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Given: Alex has 15 friends and 90 coins
def number_of_friends := 15
def initial_coins := 90

-- The total number of coins required
def total_coins_required := sum_first_natural_numbers number_of_friends

-- Calculate the additional coins needed
theorem required_additional_coins : total_coins_required - initial_coins = 30 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_required_additional_coins_l609_60960


namespace NUMINAMATH_GPT_distance_between_intersections_l609_60935

-- Given conditions
def line_eq (x : ℝ) : ℝ := 5
def quad_eq (x : ℝ) : ℝ := 5 * x^2 + 2 * x - 2

-- The proof statement
theorem distance_between_intersections : 
  ∃ (C D : ℝ), line_eq C = quad_eq C ∧ line_eq D = quad_eq D ∧ abs (C - D) = 2.4 :=
by
  -- We will later fill in the proof here
  sorry

end NUMINAMATH_GPT_distance_between_intersections_l609_60935


namespace NUMINAMATH_GPT_perimeter_pentagon_ABCD_l609_60951

noncomputable def AB : ℝ := 2
noncomputable def BC : ℝ := Real.sqrt 8
noncomputable def CD : ℝ := Real.sqrt 18
noncomputable def DE : ℝ := Real.sqrt 32
noncomputable def AE : ℝ := Real.sqrt 62

theorem perimeter_pentagon_ABCD : 
  AB + BC + CD + DE + AE = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  -- Note: The proof has been skipped as per instruction.
  sorry

end NUMINAMATH_GPT_perimeter_pentagon_ABCD_l609_60951


namespace NUMINAMATH_GPT_floor_sum_min_value_l609_60959

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end NUMINAMATH_GPT_floor_sum_min_value_l609_60959


namespace NUMINAMATH_GPT_solve_for_x_l609_60989

theorem solve_for_x (x : ℝ) :
  (x + 3)^3 = -64 → x = -7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l609_60989


namespace NUMINAMATH_GPT_weight_of_each_soda_crate_l609_60950

-- Definitions based on conditions
def bridge_weight_limit := 20000
def empty_truck_weight := 12000
def number_of_soda_crates := 20
def dryer_weight := 3000
def number_of_dryers := 3
def fully_loaded_truck_weight := 24000
def soda_weight := 1000
def produce_weight := 2 * soda_weight
def total_cargo_weight := fully_loaded_truck_weight - empty_truck_weight

-- Lean statement to prove the weight of each soda crate
theorem weight_of_each_soda_crate :
  number_of_soda_crates * ((total_cargo_weight - (number_of_dryers * dryer_weight)) / 3) / number_of_soda_crates = 50 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_each_soda_crate_l609_60950


namespace NUMINAMATH_GPT_savings_after_expense_increase_l609_60903

-- Define constants and initial conditions
def salary : ℝ := 7272.727272727273
def savings_rate : ℝ := 0.10
def expense_increase_rate : ℝ := 0.05

-- Define initial savings, expenses, and new expenses
def initial_savings : ℝ := savings_rate * salary
def initial_expenses : ℝ := salary - initial_savings
def new_expenses : ℝ := initial_expenses * (1 + expense_increase_rate)
def new_savings : ℝ := salary - new_expenses

-- The theorem statement
theorem savings_after_expense_increase : new_savings = 400 := by
  sorry

end NUMINAMATH_GPT_savings_after_expense_increase_l609_60903


namespace NUMINAMATH_GPT_solve_for_x_l609_60961

theorem solve_for_x (x : ℕ) 
  (h : 225 + 2 * 15 * 4 + 16 = x) : x = 361 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l609_60961


namespace NUMINAMATH_GPT_triangle_inradius_l609_60949

theorem triangle_inradius (A s r : ℝ) (h₁ : A = 3 * s) (h₂ : A = r * s) (h₃ : s ≠ 0) : r = 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_triangle_inradius_l609_60949


namespace NUMINAMATH_GPT_football_kick_distance_l609_60984

theorem football_kick_distance (a : ℕ) (avg : ℕ) (x : ℕ)
  (h1 : a = 43)
  (h2 : avg = 37)
  (h3 : 3 * avg = a + 2 * x) :
  x = 34 :=
by
  sorry

end NUMINAMATH_GPT_football_kick_distance_l609_60984


namespace NUMINAMATH_GPT_carbonated_water_percentage_is_correct_l609_60969

-- Given percentages of lemonade and carbonated water in two solutions
def first_solution : Rat := 0.20 -- Lemonade percentage in the first solution
def second_solution : Rat := 0.45 -- Lemonade percentage in the second solution

-- Calculate percentages of carbonated water
def first_solution_carbonated_water := 1 - first_solution
def second_solution_carbonated_water := 1 - second_solution

-- Assume the mixture is 100 units, with equal parts from both solutions
def volume_mixture : Rat := 100
def volume_first_solution : Rat := volume_mixture * 0.50
def volume_second_solution : Rat := volume_mixture * 0.50

-- Calculate total carbonated water in the mixture
def carbonated_water_in_mixture :=
  (volume_first_solution * first_solution_carbonated_water) +
  (volume_second_solution * second_solution_carbonated_water)

-- Calculate the percentage of carbonated water in the mixture
def percentage_carbonated_water_in_mixture : Rat :=
  (carbonated_water_in_mixture / volume_mixture) * 100

-- Prove the percentage of carbonated water in the mixture is 67.5%
theorem carbonated_water_percentage_is_correct :
  percentage_carbonated_water_in_mixture = 67.5 := by
  sorry

end NUMINAMATH_GPT_carbonated_water_percentage_is_correct_l609_60969


namespace NUMINAMATH_GPT_quadratic_function_analysis_l609_60923

theorem quadratic_function_analysis (a b c : ℝ) :
  (a - b + c = -1) →
  (c = 2) →
  (4 * a + 2 * b + c = 2) →
  (16 * a + 4 * b + c = -6) →
  (¬ ∃ x > 3, a * x^2 + b * x + c = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_quadratic_function_analysis_l609_60923


namespace NUMINAMATH_GPT_a10_b10_l609_60918

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a10_b10 : a^10 + b^10 = 123 :=
by
  sorry

end NUMINAMATH_GPT_a10_b10_l609_60918


namespace NUMINAMATH_GPT_digits_exceed_10_power_15_l609_60996

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem digits_exceed_10_power_15 (x : ℝ) 
  (h : log3 (log2 (log2 x)) = 3) : log10 x > 10^15 := 
sorry

end NUMINAMATH_GPT_digits_exceed_10_power_15_l609_60996


namespace NUMINAMATH_GPT_relationship_of_products_l609_60911

theorem relationship_of_products
  {a1 a2 b1 b2 : ℝ}
  (h1 : a1 < a2)
  (h2 : b1 < b2) :
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 :=
sorry

end NUMINAMATH_GPT_relationship_of_products_l609_60911


namespace NUMINAMATH_GPT_length_of_train_l609_60968

theorem length_of_train 
  (L V : ℝ) 
  (h1 : L = V * 8) 
  (h2 : L + 279 = V * 20) : 
  L = 186 :=
by
  -- solve using the given conditions
  sorry

end NUMINAMATH_GPT_length_of_train_l609_60968


namespace NUMINAMATH_GPT_sum_of_ages_l609_60998

theorem sum_of_ages (M S G : ℕ)
  (h1 : M = 2 * S)
  (h2 : S = 2 * G)
  (h3 : G = 20) :
  M + S + G = 140 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l609_60998


namespace NUMINAMATH_GPT_smallest_n_inequality_l609_60962

theorem smallest_n_inequality :
  ∃ (n : ℕ), ∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧ n = 4 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_smallest_n_inequality_l609_60962


namespace NUMINAMATH_GPT_parallel_line_eq_perpendicular_line_eq_l609_60979

-- Define the conditions: A line passing through (1, -4) and the given line equation 2x + 3y + 5 = 0
def passes_through (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Define the theorem statements for parallel and perpendicular lines
theorem parallel_line_eq (m : ℝ) :
  passes_through 1 (-4) 2 3 m → m = 10 := 
sorry

theorem perpendicular_line_eq (n : ℝ) :
  passes_through 1 (-4) 3 (-2) (-n) → n = 11 :=
sorry

end NUMINAMATH_GPT_parallel_line_eq_perpendicular_line_eq_l609_60979


namespace NUMINAMATH_GPT_inscribed_circle_radius_l609_60991

noncomputable def calculate_r (a b c : ℝ) : ℝ :=
  let term1 := 1 / a
  let term2 := 1 / b
  let term3 := 1 / c
  let term4 := 3 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))
  1 / (term1 + term2 + term3 + term4)

theorem inscribed_circle_radius :
  calculate_r 6 10 15 = 30 / (10 * Real.sqrt 26 + 3) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l609_60991


namespace NUMINAMATH_GPT_no_solution_for_system_l609_60952

theorem no_solution_for_system (x y z : ℝ) 
  (h1 : |x| < |y - z|) 
  (h2 : |y| < |z - x|) 
  (h3 : |z| < |x - y|) : 
  false :=
sorry

end NUMINAMATH_GPT_no_solution_for_system_l609_60952


namespace NUMINAMATH_GPT_circle_center_radius_l609_60992

-- Define the necessary parameters and let Lean solve the equivalent proof problem
theorem circle_center_radius:
  (∃ a b r : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y = 1 ↔ (x + 4)^2 + (y - 1)^2 = 18) 
  ∧ a = -4 
  ∧ b = 1 
  ∧ r = 3 * Real.sqrt 2
  ∧ a + b + r = -3 + 3 * Real.sqrt 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_center_radius_l609_60992


namespace NUMINAMATH_GPT_fraction_to_decimal_l609_60922

theorem fraction_to_decimal : (53 : ℚ) / (4 * 5^7) = 1325 / 10^7 := sorry

end NUMINAMATH_GPT_fraction_to_decimal_l609_60922


namespace NUMINAMATH_GPT_corrected_mean_l609_60921

theorem corrected_mean (mean : ℝ) (n : ℕ) (wrong_ob : ℝ) (correct_ob : ℝ) 
(h1 : mean = 36) (h2 : n = 50) (h3 : wrong_ob = 23) (h4 : correct_ob = 34) : 
(mean * n + (correct_ob - wrong_ob)) / n = 36.22 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l609_60921


namespace NUMINAMATH_GPT_sum_of_fractions_l609_60913

theorem sum_of_fractions : 
  (1/12 + 2/12 + 3/12 + 4/12 + 5/12 + 6/12 + 7/12 + 8/12 + 9/12 + 65/12 + 3/4) = 119 / 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l609_60913


namespace NUMINAMATH_GPT_part1_part2_l609_60931

variable (x y : ℤ) (A B : ℤ)

def A_def : ℤ := 3 * x^2 - 5 * x * y - 2 * y^2
def B_def : ℤ := x^2 - 3 * y

theorem part1 : A_def x y - 2 * B_def x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by
  sorry

theorem part2 : A_def 2 (-1) - 2 * B_def 2 (-1) = 6 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l609_60931


namespace NUMINAMATH_GPT_sum_of_gcd_and_lcm_l609_60948

-- Definitions of gcd and lcm for the conditions
def gcd_of_42_and_56 : ℕ := Nat.gcd 42 56
def lcm_of_24_and_18 : ℕ := Nat.lcm 24 18

-- Lean statement that the sum of the gcd and lcm is 86
theorem sum_of_gcd_and_lcm : gcd_of_42_and_56 + lcm_of_24_and_18 = 86 := by
  sorry

end NUMINAMATH_GPT_sum_of_gcd_and_lcm_l609_60948


namespace NUMINAMATH_GPT_canoe_kayak_rental_l609_60938

theorem canoe_kayak_rental:
  ∀ (C K : ℕ), 
    12 * C + 18 * K = 504 → 
    C = (3 * K) / 2 → 
    C - K = 7 :=
  by
    intro C K
    intros h1 h2
    sorry

end NUMINAMATH_GPT_canoe_kayak_rental_l609_60938


namespace NUMINAMATH_GPT_sample_size_six_l609_60967

-- Definitions for the conditions
def num_senior_teachers : ℕ := 18
def num_first_level_teachers : ℕ := 12
def num_top_level_teachers : ℕ := 6
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_top_level_teachers

-- The proof problem statement
theorem sample_size_six (n : ℕ) (h1 : n > 0) : 
  (∀ m : ℕ, m * n = total_teachers → 
             ((n + 1) * m - 1 = 35) → False) → n = 6 :=
sorry

end NUMINAMATH_GPT_sample_size_six_l609_60967


namespace NUMINAMATH_GPT_tickets_per_candy_l609_60939

theorem tickets_per_candy (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) (candies_bought : ℕ)
    (h1 : tickets_whack_a_mole = 26) (h2 : tickets_skee_ball = 19) (h3 : candies_bought = 5) :
    (tickets_whack_a_mole + tickets_skee_ball) / candies_bought = 9 := by
  sorry

end NUMINAMATH_GPT_tickets_per_candy_l609_60939


namespace NUMINAMATH_GPT_min_students_solved_both_l609_60981

/-- A simple mathematical proof problem to find the minimum number of students who solved both problems correctly --/
theorem min_students_solved_both (total_students first_problem second_problem : ℕ)
  (h₀ : total_students = 30)
  (h₁ : first_problem = 21)
  (h₂ : second_problem = 18) :
  ∃ (both_solved : ℕ), both_solved = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_students_solved_both_l609_60981


namespace NUMINAMATH_GPT_correct_operation_l609_60994

variable (N : ℚ) -- Original number (assumed rational for simplicity)
variable (x : ℚ) -- Unknown multiplier

theorem correct_operation (h : (N / 10) = (5 / 100) * (N * x)) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l609_60994


namespace NUMINAMATH_GPT_tennis_balls_per_can_is_three_l609_60977

-- Definition of the number of games in each round
def games_in_round (round: Nat) : Nat :=
  match round with
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => 0

-- Definition of the average number of cans used per game
def cans_per_game : Nat := 5

-- Total number of games in the tournament
def total_games : Nat :=
  games_in_round 1 + games_in_round 2 + games_in_round 3 + games_in_round 4

-- Total number of cans used
def total_cans : Nat :=
  total_games * cans_per_game

-- Total number of tennis balls used
def total_tennis_balls : Nat := 225

-- Number of tennis balls per can
def tennis_balls_per_can : Nat :=
  total_tennis_balls / total_cans

-- Theorem to prove
theorem tennis_balls_per_can_is_three :
  tennis_balls_per_can = 3 :=
by
  -- No proof required, using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_tennis_balls_per_can_is_three_l609_60977


namespace NUMINAMATH_GPT_determine_divisors_l609_60915

theorem determine_divisors (n : ℕ) (h_pos : n > 0) (d : ℕ) (h_div : d ∣ 3 * n^2) (h_exists : ∃ k : ℤ, n^2 + d = k^2) : d = 3 * n^2 := 
sorry

end NUMINAMATH_GPT_determine_divisors_l609_60915


namespace NUMINAMATH_GPT_area_of_triangle_l609_60930

noncomputable def segment_length_AB : ℝ := 10
noncomputable def point_AP : ℝ := 2
noncomputable def point_PB : ℝ := segment_length_AB - point_AP -- PB = AB - AP 
noncomputable def radius_omega1 : ℝ := point_AP / 2 -- radius of ω1
noncomputable def radius_omega2 : ℝ := point_PB / 2 -- radius of ω2
noncomputable def distance_centers : ℝ := 5 -- given directly
noncomputable def length_XY : ℝ := 4 -- given directly
noncomputable def altitude_PZ : ℝ := 8 / 5 -- given directly
noncomputable def area_triangle_XPY : ℝ := (1 / 2) * length_XY * altitude_PZ

theorem area_of_triangle : area_triangle_XPY = 16 / 5 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l609_60930


namespace NUMINAMATH_GPT_frog_eggs_ratio_l609_60942

theorem frog_eggs_ratio
    (first_day : ℕ)
    (second_day : ℕ)
    (third_day : ℕ)
    (total_eggs : ℕ)
    (h1 : first_day = 50)
    (h2 : second_day = first_day * 2)
    (h3 : third_day = second_day + 20)
    (h4 : total_eggs = 810) :
    (total_eggs - (first_day + second_day + third_day)) / (first_day + second_day + third_day) = 2 :=
by
    sorry

end NUMINAMATH_GPT_frog_eggs_ratio_l609_60942


namespace NUMINAMATH_GPT_gcd_consecutive_triplets_l609_60965

theorem gcd_consecutive_triplets : ∀ i : ℕ, 1 ≤ i → gcd (i * (i + 1) * (i + 2)) 6 = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_consecutive_triplets_l609_60965


namespace NUMINAMATH_GPT_lattice_points_on_hyperbola_l609_60997

-- The hyperbola equation
def hyperbola_eq (x y : ℤ) : Prop :=
  x^2 - y^2 = 1800^2

-- The final number of lattice points lying on the hyperbola
theorem lattice_points_on_hyperbola : 
  ∃ (n : ℕ), n = 250 ∧ (∃ (x y : ℤ), hyperbola_eq x y) :=
sorry

end NUMINAMATH_GPT_lattice_points_on_hyperbola_l609_60997


namespace NUMINAMATH_GPT_percentage_error_in_area_l609_60993

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := s * (1 + 0.03)
  let A := s * s
  let A' := s' * s'
  ((A' - A) / A) * 100 = 6.09 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l609_60993


namespace NUMINAMATH_GPT_ellipses_same_eccentricity_l609_60987

theorem ellipses_same_eccentricity 
  (a b : ℝ) (k : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : k > 0)
  (e1_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / (a^2)) + (y^2 / (b^2)) = 1)
  (e2_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = k ↔ (x^2 / (ka^2)) + (y^2 / (kb^2)) = 1) :
  1 - (b^2 / a^2) = 1 - (b^2 / (ka^2)) :=
by
  sorry

end NUMINAMATH_GPT_ellipses_same_eccentricity_l609_60987


namespace NUMINAMATH_GPT_exists_k_undecided_l609_60904

def tournament (n : ℕ) : Type :=
  { T : Fin n → Fin n → Prop // ∀ i j, T i j = ¬T j i }

def k_undecided (n k : ℕ) (T : tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k → ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → T.1 p a

theorem exists_k_undecided (k : ℕ) (hk : 0 < k) : ∃ (n : ℕ), n > k ∧ ∃ (T : tournament n), k_undecided n k T :=
by
  sorry

end NUMINAMATH_GPT_exists_k_undecided_l609_60904


namespace NUMINAMATH_GPT_max_profit_l609_60975

noncomputable def maximum_profit : ℤ := 
  21000

theorem max_profit (x y : ℕ) 
  (h1 : 4 * x + 8 * y ≤ 8000)
  (h2 : 2 * x + y ≤ 1300)
  (h3 : 15 * x + 20 * y ≤ maximum_profit) : 
  15 * x + 20 * y = maximum_profit := 
sorry

end NUMINAMATH_GPT_max_profit_l609_60975


namespace NUMINAMATH_GPT_degrees_multiplication_proof_l609_60976

/-- Convert a measurement given in degrees and minutes to purely degrees. -/
def degrees (d : Int) (m : Int) : ℚ := d + m / 60

/-- Given conditions: -/
def lhs : ℚ := degrees 21 17
def rhs : ℚ := degrees 106 25

/-- The theorem to prove the mathematical problem. -/
theorem degrees_multiplication_proof : lhs * 5 = rhs := sorry

end NUMINAMATH_GPT_degrees_multiplication_proof_l609_60976


namespace NUMINAMATH_GPT_count_ball_box_arrangements_l609_60905

theorem count_ball_box_arrangements :
  ∃ (arrangements : ℕ), arrangements = 20 ∧
  (∃ f : Fin 5 → Fin 5,
    (∃! i1, f i1 = i1) ∧ (∃! i2, f i2 = i2) ∧
    ∀ i, ∃! j, f i = j) :=
sorry

end NUMINAMATH_GPT_count_ball_box_arrangements_l609_60905


namespace NUMINAMATH_GPT_smallest_possible_product_l609_60900

theorem smallest_possible_product : 
  ∃ (x : ℕ) (y : ℕ), (x = 56 ∧ y = 78 ∨ x = 57 ∧ y = 68) ∧ x * y = 3876 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_product_l609_60900


namespace NUMINAMATH_GPT_maximum_area_right_triangle_in_rectangle_l609_60963

theorem maximum_area_right_triangle_in_rectangle :
  ∃ (area : ℕ), 
  (∀ (a b : ℕ), a = 12 ∧ b = 5 → area = 1 / 2 * a * b) :=
by
  use 30
  sorry

end NUMINAMATH_GPT_maximum_area_right_triangle_in_rectangle_l609_60963


namespace NUMINAMATH_GPT_student_thought_six_is_seven_l609_60941

theorem student_thought_six_is_seven
  (n : ℕ → ℕ)
  (h1 : (n 1 + n 3) / 2 = 2)
  (h2 : (n 2 + n 4) / 2 = 3)
  (h3 : (n 3 + n 5) / 2 = 4)
  (h4 : (n 4 + n 6) / 2 = 5)
  (h5 : (n 5 + n 7) / 2 = 6)
  (h6 : (n 6 + n 8) / 2 = 7)
  (h7 : (n 7 + n 9) / 2 = 8)
  (h8 : (n 8 + n 10) / 2 = 9)
  (h9 : (n 9 + n 1) / 2 = 10)
  (h10 : (n 10 + n 2) / 2 = 1) : 
  n 6 = 7 := 
  sorry

end NUMINAMATH_GPT_student_thought_six_is_seven_l609_60941


namespace NUMINAMATH_GPT_solution_set_of_inequality_l609_60990

theorem solution_set_of_inequality :
  {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {-1 / 3} :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_solution_set_of_inequality_l609_60990


namespace NUMINAMATH_GPT_student_in_16th_group_has_number_244_l609_60982

theorem student_in_16th_group_has_number_244 :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 800 ∧ ((k - 36) % 16 = 0) ∧ (n = 3 + (k - 36) / 16)) →
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ 800 ∧ ((m - 244) % 16 = 0) ∧ (16 = 3 + (m - 36) / 16) :=
by
  sorry

end NUMINAMATH_GPT_student_in_16th_group_has_number_244_l609_60982


namespace NUMINAMATH_GPT_average_weight_of_abc_l609_60955

theorem average_weight_of_abc 
  (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 46)
  (h3 : B = 37) :
  (A + B + C) / 3 = 45 := 
by
  sorry

end NUMINAMATH_GPT_average_weight_of_abc_l609_60955


namespace NUMINAMATH_GPT_product_of_roots_quadratic_eq_l609_60958

theorem product_of_roots_quadratic_eq : 
  ∀ (x1 x2 : ℝ), 
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 → (x = x1 ∨ x = x2)) → 
  x1 * x2 = -3 :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_product_of_roots_quadratic_eq_l609_60958


namespace NUMINAMATH_GPT_max_heaps_of_stones_l609_60999

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end NUMINAMATH_GPT_max_heaps_of_stones_l609_60999


namespace NUMINAMATH_GPT_balanced_polygons_characterization_l609_60972

def convex_polygon (n : ℕ) (vertices : Fin n → Point) : Prop := 
  -- Definition of convex_polygon should go here
  sorry

def is_balanced (n : ℕ) (vertices : Fin n → Point) (M : Point) : Prop := 
  -- Definition of is_balanced should go here
  sorry

theorem balanced_polygons_characterization :
  ∀ (n : ℕ) (vertices : Fin n → Point) (M : Point),
  convex_polygon n vertices →
  is_balanced n vertices M →
  n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end NUMINAMATH_GPT_balanced_polygons_characterization_l609_60972


namespace NUMINAMATH_GPT_ashley_cocktail_calories_l609_60966

theorem ashley_cocktail_calories:
  let mango_grams := 150
  let honey_grams := 200
  let water_grams := 300
  let vodka_grams := 100

  let mango_cal_per_100g := 60
  let honey_cal_per_100g := 640
  let vodka_cal_per_100g := 70
  let water_cal_per_100g := 0

  let total_cocktail_grams := mango_grams + honey_grams + water_grams + vodka_grams
  let total_cocktail_calories := (mango_grams * mango_cal_per_100g / 100) +
                                 (honey_grams * honey_cal_per_100g / 100) +
                                 (vodka_grams * vodka_cal_per_100g / 100) +
                                 (water_grams * water_cal_per_100g / 100)
  let caloric_density := total_cocktail_calories / total_cocktail_grams
  let result := 300 * caloric_density
  result = 576 := by
  sorry

end NUMINAMATH_GPT_ashley_cocktail_calories_l609_60966


namespace NUMINAMATH_GPT_initial_paintings_l609_60929

theorem initial_paintings (x : ℕ) (h : x - 3 = 95) : x = 98 :=
sorry

end NUMINAMATH_GPT_initial_paintings_l609_60929


namespace NUMINAMATH_GPT_smallest_positive_integer_l609_60974

theorem smallest_positive_integer {x : ℕ} (h1 : x % 6 = 3) (h2 : x % 8 = 5) : x = 21 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l609_60974


namespace NUMINAMATH_GPT_jane_waiting_time_l609_60947

-- Given conditions as constants for readability
def base_coat_drying_time := 2
def first_color_coat_drying_time := 3
def second_color_coat_drying_time := 3
def top_coat_drying_time := 5

-- Total drying time calculation
def total_drying_time := base_coat_drying_time 
                       + first_color_coat_drying_time 
                       + second_color_coat_drying_time 
                       + top_coat_drying_time

-- The theorem to prove
theorem jane_waiting_time : total_drying_time = 13 := 
by
  sorry

end NUMINAMATH_GPT_jane_waiting_time_l609_60947


namespace NUMINAMATH_GPT_multiply_increase_by_196_l609_60927

theorem multiply_increase_by_196 (x : ℕ) (h : 14 * x = 14 + 196) : x = 15 :=
sorry

end NUMINAMATH_GPT_multiply_increase_by_196_l609_60927


namespace NUMINAMATH_GPT_problem_l609_60933

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end NUMINAMATH_GPT_problem_l609_60933


namespace NUMINAMATH_GPT_newspaper_spending_over_8_weeks_l609_60919

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end NUMINAMATH_GPT_newspaper_spending_over_8_weeks_l609_60919


namespace NUMINAMATH_GPT_sum_lent_borrowed_l609_60945

-- Define the given conditions and the sum lent
def sum_lent (P r t : ℝ) (I : ℝ) : Prop :=
  I = P * r * t / 100 ∧ I = P - 1540

-- Define the main theorem to be proven
theorem sum_lent_borrowed : 
  ∃ P : ℝ, sum_lent P 8 10 ((4 * P) / 5) ∧ P = 7700 :=
by
  sorry

end NUMINAMATH_GPT_sum_lent_borrowed_l609_60945


namespace NUMINAMATH_GPT_complete_square_monomials_l609_60912

theorem complete_square_monomials (x : ℝ) :
  ∃ (m : ℝ), (m = 4 * x ^ 4 ∨ m = 4 * x ∨ m = -4 * x ∨ m = -1 ∨ m = -4 * x ^ 2) ∧
              (∃ (a b : ℝ), (4 * x ^ 2 + 1 + m = a ^ 2 + b ^ 2)) :=
sorry

-- Note: The exact formulation of the problem might vary based on the definition
-- of perfect squares and corresponding polynomials in the Lean environment.

end NUMINAMATH_GPT_complete_square_monomials_l609_60912


namespace NUMINAMATH_GPT_cosine_in_third_quadrant_l609_60925

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end NUMINAMATH_GPT_cosine_in_third_quadrant_l609_60925


namespace NUMINAMATH_GPT_smaug_copper_coins_l609_60956

def copper_value_of_silver (silver_coins silver_to_copper : ℕ) : ℕ :=
  silver_coins * silver_to_copper

def copper_value_of_gold (gold_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  gold_coins * gold_to_silver * silver_to_copper

def total_copper_value (gold_coins silver_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  copper_value_of_gold gold_coins gold_to_silver silver_to_copper +
  copper_value_of_silver silver_coins silver_to_copper

def actual_copper_coins (total_value gold_value silver_value : ℕ) : ℕ :=
  total_value - (gold_value + silver_value)

theorem smaug_copper_coins :
  let gold_coins := 100
  let silver_coins := 60
  let silver_to_copper := 8
  let gold_to_silver := 3
  let total_copper_value := 2913
  let gold_value := copper_value_of_gold gold_coins gold_to_silver silver_to_copper
  let silver_value := copper_value_of_silver silver_coins silver_to_copper
  actual_copper_coins total_copper_value gold_value silver_value = 33 :=
by
  sorry

end NUMINAMATH_GPT_smaug_copper_coins_l609_60956


namespace NUMINAMATH_GPT_chess_tournament_games_l609_60985

-- Define the problem
def total_chess_games (n_players games_per_player : ℕ) : ℕ :=
  (n_players * games_per_player) / 2

-- Conditions: 
-- 1. There are 6 chess amateurs.
-- 2. Each amateur plays exactly 4 games.

theorem chess_tournament_games :
  total_chess_games 6 4 = 10 :=
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l609_60985


namespace NUMINAMATH_GPT_problem_1_problem_2_l609_60917

section Problem1

variable (x a : ℝ)

-- Proposition p
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 1
theorem problem_1 : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by { sorry }

end Problem1

section Problem2

variable (a : ℝ)

-- Proposition p with a as a variable
def p_a (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q with x as a variable
def q_x (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 2
theorem problem_2 : (∀ (x : ℝ), ¬p_a a x → ¬q_x x) → (1 < a ∧ a ≤ 2) :=
by { sorry }

end Problem2

end NUMINAMATH_GPT_problem_1_problem_2_l609_60917


namespace NUMINAMATH_GPT_inverse_of_2_is_46_l609_60995

-- Given the function f(x) = 5x^3 + 6
def f (x : ℝ) : ℝ := 5 * x^3 + 6

-- Prove the statement
theorem inverse_of_2_is_46 : (∃ y, f y = x) ∧ f (2 : ℝ) = 46 → x = 46 :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_2_is_46_l609_60995


namespace NUMINAMATH_GPT_find_irrational_satisfying_conditions_l609_60932

-- Define a real number x which is irrational
def is_irrational (x : ℝ) : Prop := ¬∃ (q : ℚ), (x : ℝ) = q

-- Define that x satisfies the given conditions
def rational_conditions (x : ℝ) : Prop :=
  (∃ (r1 : ℚ), x^3 - 17 * x = r1) ∧ (∃ (r2 : ℚ), x^2 + 4 * x = r2)

-- The main theorem statement
theorem find_irrational_satisfying_conditions (x : ℝ) 
  (hx_irr : is_irrational x) 
  (hx_cond : rational_conditions x) : x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_find_irrational_satisfying_conditions_l609_60932


namespace NUMINAMATH_GPT_area_under_abs_sin_l609_60978

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem area_under_abs_sin : 
  ∫ x in -Real.pi..Real.pi, f x = 4 :=
by
  sorry

end NUMINAMATH_GPT_area_under_abs_sin_l609_60978


namespace NUMINAMATH_GPT_even_function_expression_l609_60946

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x then 2*x + 1 else -2*x + 1

theorem even_function_expression (x : ℝ) (hx : x < 0) :
  f x = -2*x + 1 :=
by sorry

end NUMINAMATH_GPT_even_function_expression_l609_60946


namespace NUMINAMATH_GPT_hotdog_cost_l609_60926

theorem hotdog_cost
  (h s : ℕ) -- Make sure to assume that the cost in cents is a natural number 
  (h1 : 3 * h + 2 * s = 360)
  (h2 : 2 * h + 3 * s = 390) :
  h = 60 :=

sorry

end NUMINAMATH_GPT_hotdog_cost_l609_60926


namespace NUMINAMATH_GPT_maximum_value_of_f_intervals_of_monotonic_increase_l609_60914

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let a1 := a x
  let b1 := b x
  a1.1 * (a1.1 + b1.1) + a1.2 * (a1.2 + b1.2)

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = 3 / 2 + Real.sqrt 2 / 2 := sorry

theorem intervals_of_monotonic_increase :
  ∃ I1 I2 : Set ℝ, 
  I1 = Set.Icc 0 (Real.pi / 8) ∧ 
  I2 = Set.Icc (5 * Real.pi / 8) Real.pi ∧ 
  (∀ x ∈ I1, ∀ y ∈ I2, x ≤ y ∧ f x ≤ f y) ∧
  (∀ x y, x ∈ I1 → y ∈ I1 → x < y → f x < f y) ∧
  (∀ x y, x ∈ I2 → y ∈ I2 → x < y → f x < f y) := sorry

end NUMINAMATH_GPT_maximum_value_of_f_intervals_of_monotonic_increase_l609_60914


namespace NUMINAMATH_GPT_boat_speed_proof_l609_60970

noncomputable def speed_in_still_water : ℝ := sorry -- Defined but proof skipped

def stream_speed : ℝ := 4
def distance_downstream : ℝ := 32
def distance_upstream : ℝ := 16

theorem boat_speed_proof (v : ℝ) :
  (distance_downstream / (v + stream_speed) = distance_upstream / (v - stream_speed)) →
  v = 12 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_proof_l609_60970


namespace NUMINAMATH_GPT_find_analytical_expression_and_a_l609_60906

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

theorem find_analytical_expression_and_a :
  (A > 0) → (ω > 0) → (0 < φ ∧ φ < π / 2) →
  (∀ x, ∃ k : ℤ, f (x + k * π / 2) = f (x)) →
  (∃ A, ∀ x, A * sin (ω * x + φ) ≤ 2) →
  ((∀ x, f (x - π / 6) = -f (-x + π / 6)) ∨ f 0 = sqrt 3 ∨ (∃ x, 2 * x + φ = k * π + π / 2)) →
  (∀ x, f x = 2 * sin (2 * x + π / 3)) ∧
  (∀ (A : ℝ), (0 < A ∧ A < π) → (f A = sqrt 3) →
  (c = 3 ∧ S = 3 * sqrt 3) →
  (a ^ 2 = ((4 * sqrt 3) ^ 2 + 3 ^ 2 - 2 * (4 * sqrt 3) * 3 * cos (π / 6))) → a = sqrt 21) :=
  sorry

end NUMINAMATH_GPT_find_analytical_expression_and_a_l609_60906


namespace NUMINAMATH_GPT_f_strictly_decreasing_intervals_f_max_min_on_interval_l609_60934

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 6 * x^2 - 9 * x + 3

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := -3 * x^2 - 12 * x - 9

-- Statement for part (I)
theorem f_strictly_decreasing_intervals :
  (∀ x : ℝ, x < -3 → f_deriv x < 0) ∧ (∀ x : ℝ, x > -1 → f_deriv x < 0) := by
  sorry

-- Statement for part (II)
theorem f_max_min_on_interval :
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≥ -47) :=
  sorry

end NUMINAMATH_GPT_f_strictly_decreasing_intervals_f_max_min_on_interval_l609_60934


namespace NUMINAMATH_GPT_integral_problem1_integral_problem2_integral_problem3_l609_60920

open Real

noncomputable def integral1 := ∫ x in (0 : ℝ)..1, x * exp (-x) = 1 - 2 / exp 1
noncomputable def integral2 := ∫ x in (1 : ℝ)..2, x * log x / log 2 = 2 - 3 / (4 * log 2)
noncomputable def integral3 := ∫ x in (1 : ℝ)..Real.exp 1, (log x) ^ 2 = exp 1 - 2

theorem integral_problem1 : integral1 := sorry
theorem integral_problem2 : integral2 := sorry
theorem integral_problem3 : integral3 := sorry

end NUMINAMATH_GPT_integral_problem1_integral_problem2_integral_problem3_l609_60920


namespace NUMINAMATH_GPT_powers_of_2_not_powers_of_4_l609_60944

theorem powers_of_2_not_powers_of_4 (n : ℕ) (h1 : n < 500000) (h2 : ∃ k : ℕ, n = 2^k) (h3 : ∀ m : ℕ, n ≠ 4^m) : n = 9 := 
by
  sorry

end NUMINAMATH_GPT_powers_of_2_not_powers_of_4_l609_60944


namespace NUMINAMATH_GPT_sample_size_is_80_l609_60986

-- Define the given conditions
variables (x : ℕ) (numA numB numC n : ℕ)

-- Conditions in Lean
def ratio_condition (x numA numB numC : ℕ) : Prop :=
  numA = 2 * x ∧ numB = 3 * x ∧ numC = 5 * x

def sample_condition (numA : ℕ) : Prop :=
  numA = 16

-- Definition of the proof problem
theorem sample_size_is_80 (x : ℕ) (numA numB numC n : ℕ)
  (h_ratio : ratio_condition x numA numB numC)
  (h_sample : sample_condition numA) : 
  n = 80 :=
by
-- The proof is omitted, just state the theorem
sorry

end NUMINAMATH_GPT_sample_size_is_80_l609_60986


namespace NUMINAMATH_GPT_smallest_N_l609_60909

theorem smallest_N (N : ℕ) : (N * 3 ≥ 75) ∧ (N * 2 < 75) → N = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_N_l609_60909


namespace NUMINAMATH_GPT_simplified_expr_eval_l609_60957

theorem simplified_expr_eval
  (x : ℚ) (y : ℚ) (h_x : x = -1/2) (h_y : y = 1) :
  (5*x^2 - 10*y^2) = -35/4 := 
by
  subst h_x
  subst h_y
  sorry

end NUMINAMATH_GPT_simplified_expr_eval_l609_60957


namespace NUMINAMATH_GPT_evaluate_x_squared_minus_y_squared_l609_60971

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_x_squared_minus_y_squared_l609_60971


namespace NUMINAMATH_GPT_parabola_vertex_l609_60940

noncomputable def is_vertex (x y : ℝ) : Prop :=
  y^2 + 8 * y + 4 * x + 5 = 0 ∧ (∀ y₀, y₀^2 + 8 * y₀ + 4 * x + 5 ≥ 0)

theorem parabola_vertex : is_vertex (11 / 4) (-4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l609_60940


namespace NUMINAMATH_GPT_binary_addition_l609_60907

theorem binary_addition (M : ℕ) (hM : M = 0b101110) :
  let M_plus_five := M + 5 
  let M_plus_five_binary := 0b110011
  let M_plus_five_predecessor := 0b110010
  M_plus_five = M_plus_five_binary ∧ M_plus_five - 1 = M_plus_five_predecessor :=
by
  sorry

end NUMINAMATH_GPT_binary_addition_l609_60907


namespace NUMINAMATH_GPT_fraction_division_l609_60924

theorem fraction_division :
  (5 / 4) / (8 / 15) = 75 / 32 :=
sorry

end NUMINAMATH_GPT_fraction_division_l609_60924


namespace NUMINAMATH_GPT_algebraic_expression_perfect_square_l609_60980

theorem algebraic_expression_perfect_square (a : ℤ) :
  (∃ b : ℤ, ∀ x : ℤ, x^2 + (a - 1) * x + 16 = (x + b)^2) →
  (a = 9 ∨ a = -7) :=
sorry

end NUMINAMATH_GPT_algebraic_expression_perfect_square_l609_60980


namespace NUMINAMATH_GPT_initial_salt_percentage_l609_60902

theorem initial_salt_percentage (initial_mass : ℝ) (added_salt_mass : ℝ) (final_solution_percentage : ℝ) (final_mass : ℝ) 
  (h1 : initial_mass = 100) 
  (h2 : added_salt_mass = 38.46153846153846) 
  (h3 : final_solution_percentage = 0.35) 
  (h4 : final_mass = 138.46153846153846) : 
  ((10 / 100) * 100) = 10 := 
sorry

end NUMINAMATH_GPT_initial_salt_percentage_l609_60902


namespace NUMINAMATH_GPT_river_flow_rate_l609_60916

theorem river_flow_rate
  (h : ℝ) (h_eq : h = 3)
  (w : ℝ) (w_eq : w = 36)
  (V : ℝ) (V_eq : V = 3600)
  (conversion_factor : ℝ) (conversion_factor_eq : conversion_factor = 3.6) :
  (60 / (w * h)) * conversion_factor = 2 := by
  sorry

end NUMINAMATH_GPT_river_flow_rate_l609_60916


namespace NUMINAMATH_GPT_widget_production_difference_l609_60910

variable (w t : ℕ)
variable (h_wt : w = 2 * t)

theorem widget_production_difference (w t : ℕ)
    (h_wt : w = 2 * t) :
  (w * t) - ((w + 5) * (t - 3)) = t + 15 :=
by 
  sorry

end NUMINAMATH_GPT_widget_production_difference_l609_60910


namespace NUMINAMATH_GPT_total_time_spent_l609_60936

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end NUMINAMATH_GPT_total_time_spent_l609_60936


namespace NUMINAMATH_GPT_abs_eq_of_sq_eq_l609_60983

theorem abs_eq_of_sq_eq (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  intro h
  sorry

end NUMINAMATH_GPT_abs_eq_of_sq_eq_l609_60983


namespace NUMINAMATH_GPT_problem_I_l609_60954

theorem problem_I (x m : ℝ) (h1 : |x - m| < 1) (h2 : (1/3 : ℝ) < x ∧ x < (1/2 : ℝ)) : (-1/2 : ℝ) ≤ m ∧ m ≤ (4/3 : ℝ) :=
sorry

end NUMINAMATH_GPT_problem_I_l609_60954


namespace NUMINAMATH_GPT_no_integer_solutions_to_system_l609_60901

theorem no_integer_solutions_to_system :
  ¬ ∃ (x y z : ℤ),
    x^2 - 2 * x * y + y^2 - z^2 = 17 ∧
    -x^2 + 3 * y * z + 3 * z^2 = 27 ∧
    x^2 - x * y + 5 * z^2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_to_system_l609_60901


namespace NUMINAMATH_GPT_gribblean_words_count_l609_60928

universe u

-- Define the Gribblean alphabet size
def alphabet_size : Nat := 3

-- Words of length 1 to 4
def words_of_length (n : Nat) : Nat :=
  alphabet_size ^ n

-- All possible words count
def total_words : Nat :=
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3) + (words_of_length 4)

-- Theorem statement
theorem gribblean_words_count : total_words = 120 :=
by
  sorry

end NUMINAMATH_GPT_gribblean_words_count_l609_60928


namespace NUMINAMATH_GPT_problem_one_problem_two_l609_60908

variables (a₁ a₂ a₃ : ℤ) (n : ℕ)
def arith_sequence : Prop :=
  a₁ + a₂ + a₃ = 21 ∧ a₁ * a₂ * a₃ = 231

theorem problem_one (h : arith_sequence a₁ a₂ a₃) : a₂ = 7 :=
sorry

theorem problem_two (h : arith_sequence a₁ a₂ a₃) :
  (∃ d : ℤ, (d = -4 ∨ d = 4) ∧ (a_n = a₁ + (n - 1) * d ∨ a_n = a₃ + (n - 1) * d)) :=
sorry

end NUMINAMATH_GPT_problem_one_problem_two_l609_60908


namespace NUMINAMATH_GPT_greatest_five_consecutive_odd_integers_l609_60943

theorem greatest_five_consecutive_odd_integers (A B C D E : ℤ) (x : ℤ) 
  (h1 : B = x + 2) 
  (h2 : C = x + 4)
  (h3 : D = x + 6)
  (h4 : E = x + 8)
  (h5 : A + B + C + D + E = 148) :
  E = 33 :=
by {
  sorry -- proof not required
}

end NUMINAMATH_GPT_greatest_five_consecutive_odd_integers_l609_60943


namespace NUMINAMATH_GPT_simplified_evaluation_eq_half_l609_60953

theorem simplified_evaluation_eq_half :
  ∃ x y : ℝ, (|x - 2| + (y + 1)^2 = 0) → 
             (3 * x - 2 * (x^2 - (1/2) * y^2) + (x - (1/2) * y^2) = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_simplified_evaluation_eq_half_l609_60953


namespace NUMINAMATH_GPT_john_sales_percentage_l609_60964

noncomputable def percentage_buyers (houses_visited_per_day : ℕ) (work_days_per_week : ℕ) (weekly_sales : ℝ) (low_price : ℝ) (high_price : ℝ) : ℝ :=
  let total_houses_per_week := houses_visited_per_day * work_days_per_week
  let average_sale_per_customer := (low_price + high_price) / 2
  let total_customers := weekly_sales / average_sale_per_customer
  (total_customers / total_houses_per_week) * 100

theorem john_sales_percentage :
  percentage_buyers 50 5 5000 50 150 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_john_sales_percentage_l609_60964


namespace NUMINAMATH_GPT_domain_of_f_l609_60973

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x + 1 > 0 ∧ x + 1 ≠ 1} = {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0} :=
by 
  sorry

end NUMINAMATH_GPT_domain_of_f_l609_60973
