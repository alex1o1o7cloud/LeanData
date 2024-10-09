import Mathlib

namespace problem_statement_l496_49601

-- Define proposition p
def prop_p : Prop := ∃ x : ℝ, Real.exp x ≥ x + 1

-- Define proposition q
def prop_q : Prop := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- The final statement we want to prove
theorem problem_statement : (prop_p ∧ ¬prop_q) :=
by
  sorry

end problem_statement_l496_49601


namespace john_total_cost_l496_49614

theorem john_total_cost :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 4
  let base_video_card_cost := 300
  let upgraded_video_card_cost := 2.5 * base_video_card_cost
  let video_card_discount := 0.12 * upgraded_video_card_cost
  let upgraded_video_card_final_cost := upgraded_video_card_cost - video_card_discount
  let foreign_monitor_cost_local := 200
  let exchange_rate := 1.25
  let foreign_monitor_cost_usd := foreign_monitor_cost_local / exchange_rate
  let peripherals_sales_tax := 0.05 * peripherals_cost
  let subtotal := computer_cost + peripherals_cost + upgraded_video_card_final_cost + peripherals_sales_tax
  let store_loyalty_discount := 0.07 * (computer_cost + peripherals_cost + upgraded_video_card_final_cost)
  let final_cost := subtotal - store_loyalty_discount + foreign_monitor_cost_usd
  final_cost = 2536.30 := sorry

end john_total_cost_l496_49614


namespace bounded_regions_l496_49670

noncomputable def regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => regions n + n + 1

theorem bounded_regions (n : ℕ) :
  (regions n = n * (n + 1) / 2 + 1) := by
  sorry

end bounded_regions_l496_49670


namespace exterior_angle_octagon_degree_l496_49638

-- Conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def number_of_sides_octagon : ℕ := 8

-- Question and correct answer
theorem exterior_angle_octagon_degree :
  (sum_of_exterior_angles 8) / number_of_sides_octagon = 45 :=
by
  sorry

end exterior_angle_octagon_degree_l496_49638


namespace math_problem_l496_49644

noncomputable def log_8 := Real.log 8
noncomputable def log_27 := Real.log 27
noncomputable def expr := (9 : ℝ) ^ (log_8 / log_27) + (2 : ℝ) ^ (log_27 / log_8)

theorem math_problem : expr = 7 := by
  sorry

end math_problem_l496_49644


namespace probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l496_49610

noncomputable def defect_rate_first_lathe : ℝ := 0.06
noncomputable def defect_rate_second_lathe : ℝ := 0.05
noncomputable def defect_rate_third_lathe : ℝ := 0.05
noncomputable def proportion_first_lathe : ℝ := 0.25
noncomputable def proportion_second_lathe : ℝ := 0.30
noncomputable def proportion_third_lathe : ℝ := 0.45

theorem probability_defective_first_lathe :
  defect_rate_first_lathe * proportion_first_lathe = 0.015 :=
by sorry

theorem overall_probability_defective :
  defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe = 0.0525 :=
by sorry

theorem conditional_probability_second_lathe :
  (defect_rate_second_lathe * proportion_second_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 2 / 7 :=
by sorry

theorem conditional_probability_third_lathe :
  (defect_rate_third_lathe * proportion_third_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 3 / 7 :=
by sorry

end probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l496_49610


namespace stops_away_pinedale_mall_from_yahya_house_l496_49692

-- Definitions based on problem conditions
def bus_speed_kmh : ℕ := 60
def stop_interval_minutes : ℕ := 5
def distance_to_mall_km : ℕ := 40

-- Definition of how many stops away is Pinedale mall from Yahya's house
def stops_to_mall : ℕ := distance_to_mall_km / (bus_speed_kmh / 60 * stop_interval_minutes)

-- Lean statement to prove the given conditions lead to the correct number of stops
theorem stops_away_pinedale_mall_from_yahya_house :
  stops_to_mall = 8 :=
by 
  -- This is a placeholder for the proof. 
  -- Actual proof steps would convert units and calculate as described in the problem.
  sorry

end stops_away_pinedale_mall_from_yahya_house_l496_49692


namespace min_ab_l496_49600

theorem min_ab (a b : ℝ) (h_cond1 : a > 0) (h_cond2 : b > 0)
  (h_eq : a * b = a + b + 3) : a * b = 9 :=
sorry

end min_ab_l496_49600


namespace nathan_tokens_used_is_18_l496_49622

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end nathan_tokens_used_is_18_l496_49622


namespace min_value_x_plus_y_l496_49655

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 2) : x + y = 8 :=
sorry

end min_value_x_plus_y_l496_49655


namespace distance_le_radius_l496_49626

variable (L : Line) (O : Circle)
variable (d r : ℝ)

-- Condition: Line L intersects with circle O
def intersects (L : Line) (O : Circle) : Prop := sorry -- Sketch: define what it means for a line to intersect a circle

axiom intersection_condition : intersects L O

-- Problem: Prove that if a line L intersects a circle O, then the distance d from the center of the circle to the line is less than or equal to the radius r of the circle.
theorem distance_le_radius (L : Line) (O : Circle) (d r : ℝ) :
  intersects L O → d ≤ r := by
  sorry

end distance_le_radius_l496_49626


namespace expand_polynomial_l496_49650

theorem expand_polynomial (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x ^ 2 - 36 := 
by {
  sorry
}

end expand_polynomial_l496_49650


namespace hari_digs_well_alone_in_48_days_l496_49607

theorem hari_digs_well_alone_in_48_days :
  (1 / 16 + 1 / 24 + 1 / (Hari_days)) = 1 / 8 → Hari_days = 48 :=
by
  intro h
  sorry

end hari_digs_well_alone_in_48_days_l496_49607


namespace find_4a_3b_l496_49648

noncomputable def g (x : ℝ) : ℝ := 4 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := g x + 2

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_4a_3b (a b : ℝ) (h_inv : ∀ x : ℝ, f (f_inv x) a b = x) : 4 * a + 3 * b = 4 :=
by
  -- Proof skipped for now
  sorry

end find_4a_3b_l496_49648


namespace KimSweaterTotal_l496_49634

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l496_49634


namespace solution_set_l496_49683

noncomputable def satisfies_equations (x y : ℝ) : Prop :=
  (x^2 + 3 * x * y = 12) ∧ (x * y = 16 + y^2 - x * y - x^2)

theorem solution_set :
  {p : ℝ × ℝ | satisfies_equations p.1 p.2} = {(4, 1), (-4, -1), (-4, 1), (4, -1)} :=
by sorry

end solution_set_l496_49683


namespace students_like_basketball_or_cricket_or_both_l496_49603

theorem students_like_basketball_or_cricket_or_both {A B C : ℕ} (hA : A = 12) (hB : B = 8) (hC : C = 3) :
    A + B - C = 17 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l496_49603


namespace fred_balloon_count_l496_49642

def sally_balloons : ℕ := 6

def fred_balloons (sally_balloons : ℕ) := 3 * sally_balloons

theorem fred_balloon_count : fred_balloons sally_balloons = 18 := by
  sorry

end fred_balloon_count_l496_49642


namespace factor_polynomial_l496_49690

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l496_49690


namespace trigonometric_identity_l496_49689

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by 
  sorry

end trigonometric_identity_l496_49689


namespace rahul_share_of_payment_l496_49691

-- Definitions
def rahulWorkDays : ℕ := 3
def rajeshWorkDays : ℕ := 2
def totalPayment : ℤ := 355

-- Theorem statement
theorem rahul_share_of_payment :
  let rahulWorkRate := 1 / (rahulWorkDays : ℝ)
  let rajeshWorkRate := 1 / (rajeshWorkDays : ℝ)
  let combinedWorkRate := rahulWorkRate + rajeshWorkRate
  let rahulShareRatio := rahulWorkRate / combinedWorkRate
  let rahulShare := (totalPayment : ℝ) * rahulShareRatio
  rahulShare = 142 :=
by
  sorry

end rahul_share_of_payment_l496_49691


namespace find_a_20_l496_49660

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Definitions: The sequence is geometric: a_n = a_1 * r^(n-1)
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 1 * r^(n-1)

-- Conditions in the problem: a_10 and a_30 satisfy the quadratic equation
def satisfies_quadratic_roots (a10 a30 : ℝ) : Prop :=
  a10 + a30 = 11 ∧ a10 * a30 = 16

-- Question: Find a_20
theorem find_a_20 (h1 : is_geometric_sequence a r)
                  (h2 : satisfies_quadratic_roots (a 10) (a 30)) :
  a 20 = 4 :=
sorry

end find_a_20_l496_49660


namespace xy_plus_y_square_l496_49669

theorem xy_plus_y_square {x y : ℝ} (h1 : x * y = 16) (h2 : x + y = 8) : x^2 + y^2 = 32 :=
sorry

end xy_plus_y_square_l496_49669


namespace range_of_a_l496_49685

-- Definitions for propositions
def p (a : ℝ) : Prop :=
  (1 - 4 * (a^2 - 6 * a) > 0) ∧ (a^2 - 6 * a < 0)

def q (a : ℝ) : Prop :=
  (a - 3)^2 - 4 ≥ 0

-- Proof statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (a ≤ 0 ∨ 1 < a ∧ a < 5 ∨ a ≥ 6) :=
by 
  sorry

end range_of_a_l496_49685


namespace Jessica_biking_speed_l496_49678

theorem Jessica_biking_speed
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance total_time : ℝ)
  (h1 : swim_distance = 0.5)
  (h2 : swim_speed = 1)
  (h3 : run_distance = 5)
  (h4 : run_speed = 5)
  (h5 : bike_distance = 20)
  (h6 : total_time = 4) :
  bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed)) = 8 :=
by
  -- Proof omitted
  sorry

end Jessica_biking_speed_l496_49678


namespace solve_absolute_value_equation_l496_49677

theorem solve_absolute_value_equation (x : ℝ) : x^2 - 3 * |x| - 4 = 0 ↔ x = 4 ∨ x = -4 :=
by
  sorry

end solve_absolute_value_equation_l496_49677


namespace m_value_l496_49609

theorem m_value (A : Set ℝ) (B : Set ℝ) (m : ℝ) 
                (hA : A = {0, 1, 2}) 
                (hB : B = {1, m}) 
                (h_subset : B ⊆ A) : 
                m = 0 ∨ m = 2 :=
by
  sorry

end m_value_l496_49609


namespace cricketer_hits_two_sixes_l496_49604

-- Definitions of the given conditions
def total_runs : ℕ := 132
def boundaries_count : ℕ := 12
def running_percent : ℚ := 54.54545454545454 / 100

-- Function to calculate runs made by running
def runs_by_running (total: ℕ) (percent: ℚ) : ℚ :=
  percent * total

-- Function to calculate runs made from boundaries
def runs_from_boundaries (count: ℕ) : ℕ :=
  count * 4

-- Function to calculate runs made from sixes
def runs_from_sixes (total: ℕ) (boundaries_runs: ℕ) (running_runs: ℚ) : ℚ :=
  total - boundaries_runs - running_runs

-- Function to calculate number of sixes hit
def number_of_sixes (sixes_runs: ℚ) : ℚ :=
  sixes_runs / 6

-- The proof statement for the cricketer hitting 2 sixes
theorem cricketer_hits_two_sixes:
  number_of_sixes (runs_from_sixes total_runs (runs_from_boundaries boundaries_count) (runs_by_running total_runs running_percent)) = 2 := by
  sorry

end cricketer_hits_two_sixes_l496_49604


namespace west_movement_is_negative_seven_l496_49616

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end west_movement_is_negative_seven_l496_49616


namespace loss_percentage_initial_selling_l496_49687

theorem loss_percentage_initial_selling (CP SP' : ℝ) 
  (hCP : CP = 1250) 
  (hSP' : SP' = CP * 1.15) 
  (h_diff : SP' - 500 = 937.5) : 
  (CP - 937.5) / CP * 100 = 25 := 
by 
  sorry

end loss_percentage_initial_selling_l496_49687


namespace diff_of_squares_l496_49671

variable {x y : ℝ}

theorem diff_of_squares : (x + y) * (x - y) = x^2 - y^2 := 
sorry

end diff_of_squares_l496_49671


namespace compute_five_fold_application_l496_49618

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then -x^2 else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -16 :=
by
  sorry

end compute_five_fold_application_l496_49618


namespace leak_rate_l496_49615

-- Definitions based on conditions
def initialWater : ℕ := 10   -- 10 cups
def finalWater : ℕ := 2      -- 2 cups
def firstThreeMilesWater : ℕ := 3 * 1    -- 1 cup per mile for first 3 miles
def lastMileWater : ℕ := 3               -- 3 cups during the last mile
def hikeDuration : ℕ := 2    -- 2 hours

-- Proving the leak rate
theorem leak_rate (drunkWater : ℕ) (leakedWater : ℕ) (leakRate : ℕ) :
  drunkWater = firstThreeMilesWater + lastMileWater ∧ 
  (initialWater - finalWater) = (drunkWater + leakedWater) ∧
  hikeDuration = 2 ∧ 
  leakRate = leakedWater / hikeDuration → leakRate = 1 :=
by
  intros h
  sorry

end leak_rate_l496_49615


namespace perimeter_of_rectangle_l496_49686

theorem perimeter_of_rectangle (L W : ℝ) (h1 : L / W = 5 / 2) (h2 : L * W = 4000) : 2 * L + 2 * W = 280 :=
sorry

end perimeter_of_rectangle_l496_49686


namespace find_s_for_g_l496_49696

def g (x : ℝ) (s : ℝ) : ℝ := 3*x^4 - 2*x^3 + 2*x^2 + x + s

theorem find_s_for_g (s : ℝ) : g (-1) s = 0 ↔ s = -6 :=
by
  sorry

end find_s_for_g_l496_49696


namespace points_and_conditions_proof_l496_49619

noncomputable def points_and_conditions (x y : ℝ) : Prop := 
|x - 3| + |y + 5| = 0

noncomputable def min_AM_BM (m : ℝ) : Prop :=
|3 - m| + |-5 - m| = 7 / 4 * |8|

noncomputable def min_PA_PB (p : ℝ) : Prop :=
|p - 3| + |p + 5| = 8

noncomputable def min_PD_PO (p : ℝ) : Prop :=
|p + 1| - |p| = -1

noncomputable def range_of_a (a : ℝ) : Prop :=
a ∈ Set.Icc (-5) (-1)

theorem points_and_conditions_proof (x y : ℝ) (m p a : ℝ) :
  points_and_conditions x y → 
  x = 3 ∧ y = -5 ∧ 
  ((m = -8 ∨ m = 6) → min_AM_BM m) ∧ 
  (min_PA_PB p) ∧ 
  (min_PD_PO p) ∧ 
  (range_of_a a) :=
by 
  sorry

end points_and_conditions_proof_l496_49619


namespace find_cos_value_l496_49658

theorem find_cos_value (α : Real) 
  (h : Real.cos (Real.pi / 8 - α) = 1 / 6) : 
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end find_cos_value_l496_49658


namespace cannot_move_reach_goal_l496_49694

structure Point :=
(x : ℤ)
(y : ℤ)

def area (p1 p2 p3 : Point) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

noncomputable def isTriangleAreaPreserved (initPos finalPos : Point) (helper1Init helper1Final helper2Init helper2Final : Point) : Prop :=
  area initPos helper1Init helper2Init = area finalPos helper1Final helper2Final

theorem cannot_move_reach_goal :
  ¬ ∃ (r₀ r₁ : Point) (a₀ a₁ : Point) (s₀ s₁ : Point),
    r₀ = ⟨0, 0⟩ ∧ r₁ = ⟨2, 2⟩ ∧
    a₀ = ⟨0, 1⟩ ∧ a₁ = ⟨0, 1⟩ ∧
    s₀ = ⟨1, 0⟩ ∧ s₁ = ⟨1, 0⟩ ∧
    isTriangleAreaPreserved r₀ r₁ a₀ a₁ s₀ s₁ :=
by sorry

end cannot_move_reach_goal_l496_49694


namespace gear_angular_speed_proportion_l496_49698

theorem gear_angular_speed_proportion :
  ∀ (ω_A ω_B ω_C ω_D k: ℝ),
    30 * ω_A = k →
    45 * ω_B = k →
    50 * ω_C = k →
    60 * ω_D = k →
    ω_A / ω_B = 1 ∧
    ω_B / ω_C = 45 / 50 ∧
    ω_C / ω_D = 50 / 60 ∧
    ω_A / ω_D = 10 / 7.5 :=
  by
    -- proof goes here
    sorry

end gear_angular_speed_proportion_l496_49698


namespace cost_equivalence_at_325_l496_49636

def cost_plan1 (x : ℕ) : ℝ := 65 + 0.40 * x
def cost_plan2 (x : ℕ) : ℝ := 0.60 * x

theorem cost_equivalence_at_325 : cost_plan1 325 = cost_plan2 325 :=
by sorry

end cost_equivalence_at_325_l496_49636


namespace temperature_fifth_day_l496_49659

variable (T1 T2 T3 T4 T5 : ℝ)

-- Conditions
def condition1 : T1 + T2 + T3 + T4 = 4 * 58 := by sorry
def condition2 : T2 + T3 + T4 + T5 = 4 * 59 := by sorry
def condition3 : T5 = (8 / 7) * T1 := by sorry

-- The statement we need to prove
theorem temperature_fifth_day : T5 = 32 := by
  -- Using the provided conditions
  sorry

end temperature_fifth_day_l496_49659


namespace competition_score_l496_49681

theorem competition_score (x : ℕ) (h : x ≥ 15) : 10 * x - 5 * (20 - x) > 120 := by
  sorry

end competition_score_l496_49681


namespace total_outfits_l496_49667

-- Define the number of shirts, pants, ties (including no-tie option), and shoes as given in the conditions.
def num_shirts : ℕ := 5
def num_pants : ℕ := 4
def num_ties : ℕ := 6 -- 5 ties + 1 no-tie option
def num_shoes : ℕ := 2

-- Proof statement: The total number of different outfits is 240.
theorem total_outfits : num_shirts * num_pants * num_ties * num_shoes = 240 :=
by
  sorry

end total_outfits_l496_49667


namespace henry_added_water_l496_49631

theorem henry_added_water (initial_fraction full_capacity final_fraction : ℝ) (h_initial_fraction : initial_fraction = 3/4) (h_full_capacity : full_capacity = 56) (h_final_fraction : final_fraction = 7/8) :
  final_fraction * full_capacity - initial_fraction * full_capacity = 7 :=
by
  sorry

end henry_added_water_l496_49631


namespace scientific_notation_14nm_l496_49684

theorem scientific_notation_14nm :
  0.000000014 = 1.4 * 10^(-8) := 
by 
  sorry

end scientific_notation_14nm_l496_49684


namespace gcd_f_x_x_l496_49623

theorem gcd_f_x_x (x : ℕ) (h : ∃ k : ℕ, x = 35622 * k) :
  Nat.gcd ((3 * x + 4) * (5 * x + 6) * (11 * x + 9) * (x + 7)) x = 378 :=
by
  sorry

end gcd_f_x_x_l496_49623


namespace max_value_expression_l496_49630

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by
  exact sorry

end max_value_expression_l496_49630


namespace find_c_l496_49640

theorem find_c (c q : ℤ) (h : ∃ (a b : ℤ), (3*x^3 + c*x + 9 = (x^2 + q*x + 1) * (a*x + b))) : c = -24 :=
sorry

end find_c_l496_49640


namespace reflection_matrix_determine_l496_49625

theorem reflection_matrix_determine (a b : ℚ)
  (h1 : (a^2 - (3/4) * b) = 1)
  (h2 : (-(3/4) * b + (1/16)) = 1)
  (h3 : (a * b + (1/4) * b) = 0)
  (h4 : (-(3/4) * a - (3/16)) = 0) :
  (a, b) = (1/4, -5/4) := 
sorry

end reflection_matrix_determine_l496_49625


namespace find_b_l496_49668

noncomputable def Q (x : ℝ) (a b c : ℝ) := 3 * x ^ 3 + a * x ^ 2 + b * x + c

theorem find_b (a b c : ℝ) (h₀ : c = 6) 
  (h₁ : ∃ (r₁ r₂ r₃ : ℝ), Q r₁ a b c = 0 ∧ Q r₂ a b c = 0 ∧ Q r₃ a b c = 0 ∧ (r₁ + r₂ + r₃) / 3 = -(c / 3) ∧ r₁ * r₂ * r₃ = -(c / 3))
  (h₂ : 3 + a + b + c = -(c / 3)): 
  b = -29 :=
sorry

end find_b_l496_49668


namespace distinct_real_numbers_sum_l496_49605

theorem distinct_real_numbers_sum:
  ∀ (p q r s : ℝ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    (r + s = 12 * p) →
    (r * s = -13 * q) →
    (p + q = 12 * r) →
    (p * q = -13 * s) →
    p + q + r + s = 2028 :=
by
  intros p q r s h_distinct h1 h2 h3 h4
  sorry

end distinct_real_numbers_sum_l496_49605


namespace find_three_tuple_solutions_l496_49697

open Real

theorem find_three_tuple_solutions :
  (x y z : ℝ) → (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z)
  → (3 * x^2 + 2 * y^2 + z^2 = 240)
  → (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) :=
by
  intro x y z
  intro h1 h2
  sorry

end find_three_tuple_solutions_l496_49697


namespace cassie_has_8_parrots_l496_49695

-- Define the conditions
def num_dogs : ℕ := 4
def nails_per_foot : ℕ := 4
def feet_per_dog : ℕ := 4
def nails_per_dog := nails_per_foot * feet_per_dog

def nails_total_dogs : ℕ := num_dogs * nails_per_dog

def claws_per_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def normal_claws_per_parrot := claws_per_leg * legs_per_parrot

def extra_toe_parrot_claws : ℕ := normal_claws_per_parrot + 1

def total_nails : ℕ := 113

-- Establishing the proof problem
theorem cassie_has_8_parrots : 
  ∃ (P : ℕ), (6 * (P - 1) + 7 = 49) ∧ P = 8 := by
  sorry

end cassie_has_8_parrots_l496_49695


namespace lowest_possible_price_l496_49663

theorem lowest_possible_price
  (manufacturer_suggested_price : ℝ := 45)
  (regular_discount_percentage : ℝ := 0.30)
  (sale_discount_percentage : ℝ := 0.20)
  (regular_discounted_price : ℝ := manufacturer_suggested_price * (1 - regular_discount_percentage))
  (final_price : ℝ := regular_discounted_price * (1 - sale_discount_percentage)) :
  final_price = 25.20 :=
by sorry

end lowest_possible_price_l496_49663


namespace fish_in_pond_l496_49632

-- Conditions
variable (N : ℕ)
variable (h₁ : 80 * 80 = 2 * N)

-- Theorem to prove 
theorem fish_in_pond (h₁ : 80 * 80 = 2 * N) : N = 3200 := 
by 
  sorry

end fish_in_pond_l496_49632


namespace problem1_problem2_l496_49688

-- Problem 1
theorem problem1 : ∀ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 → x = 8 :=
by
  intro x
  intro h
  sorry

-- Problem 2
theorem problem2 : ∀ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 → x = 1 :=
by
  intro x
  intro h
  sorry

end problem1_problem2_l496_49688


namespace book_width_l496_49635

noncomputable def phi_conjugate : ℝ := (Real.sqrt 5 - 1) / 2

theorem book_width {w l : ℝ} (h_ratio : w / l = phi_conjugate) (h_length : l = 14) :
  w = 7 * Real.sqrt 5 - 7 :=
by
  sorry

end book_width_l496_49635


namespace smallest_10_digit_number_with_sum_81_l496_49664

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

theorem smallest_10_digit_number_with_sum_81 {n : Nat} :
  n ≥ 1000000000 ∧ n < 10000000000 ∧ sum_of_digits n ≥ 81 → 
  n = 1899999999 :=
sorry

end smallest_10_digit_number_with_sum_81_l496_49664


namespace block_measure_is_40_l496_49679

def jony_walks (start_time : String) (start_block end_block stop_block : ℕ) (stop_time : String) (speed : ℕ) : ℕ :=
  let total_time := 40 -- walking time in minutes
  let total_distance := speed * total_time -- total distance walked in meters
  let blocks_forward := end_block - start_block -- blocks walked forward
  let blocks_backward := end_block - stop_block -- blocks walked backward
  let total_blocks := blocks_forward + blocks_backward -- total blocks walked
  total_distance / total_blocks

theorem block_measure_is_40 :
  jony_walks "07:00" 10 90 70 "07:40" 100 = 40 := by
  sorry

end block_measure_is_40_l496_49679


namespace solution_set_of_inequality_l496_49617

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x ^ 2 - x + 2 < 0} = {x : ℝ | x < -(2 / 3)} ∪ {x | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l496_49617


namespace division_result_l496_49661

theorem division_result : (0.284973 / 29 = 0.009827) := 
by sorry

end division_result_l496_49661


namespace mariela_cards_received_l496_49676

theorem mariela_cards_received (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403) (h2 : cards_at_home = 287) : 
  cards_in_hospital + cards_at_home = 690 := 
by 
  sorry

end mariela_cards_received_l496_49676


namespace combined_afternoon_burning_rate_l496_49627

theorem combined_afternoon_burning_rate 
  (morning_period_hours : ℕ)
  (afternoon_period_hours : ℕ)
  (rate_A_morning : ℕ)
  (rate_B_morning : ℕ)
  (total_morning_burn : ℕ)
  (initial_wood : ℕ)
  (remaining_wood : ℕ) :
  morning_period_hours = 4 →
  afternoon_period_hours = 4 →
  rate_A_morning = 2 →
  rate_B_morning = 1 →
  total_morning_burn = 12 →
  initial_wood = 50 →
  remaining_wood = 6 →
  ((initial_wood - remaining_wood - total_morning_burn) / afternoon_period_hours) = 8 := 
by
  intros
  -- We would continue with a proof here
  sorry

end combined_afternoon_burning_rate_l496_49627


namespace infinite_sum_computation_l496_49646

theorem infinite_sum_computation : 
  ∑' n : ℕ, (3 * (n + 1) + 2) / (n * (n + 1) * (n + 3)) = 10 / 3 :=
by sorry

end infinite_sum_computation_l496_49646


namespace poly_sum_correct_l496_49641

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end poly_sum_correct_l496_49641


namespace square_diff_l496_49643

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l496_49643


namespace kristin_annual_income_l496_49693

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end kristin_annual_income_l496_49693


namespace tom_initial_balloons_l496_49675

noncomputable def initial_balloons (x : ℕ) : ℕ :=
  if h₁ : x % 2 = 1 ∧ (x / 3) + 10 = 45 then x else 0

theorem tom_initial_balloons : initial_balloons 105 = 105 :=
by {
  -- Given x is an odd number and the equation (x / 3) + 10 = 45 holds, prove x = 105.
  -- These conditions follow from the problem statement directly.
  -- Proof is skipped.
  sorry
}

end tom_initial_balloons_l496_49675


namespace find_second_number_l496_49662

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 280 ∧
  a = 2 * b ∧
  c = 2 / 3 * a ∧
  d = b + c

theorem find_second_number (a b c d : ℚ) (h : problem a b c d) : b = 52.5 :=
by
  -- Proof will go here.
  sorry

end find_second_number_l496_49662


namespace percentage_good_oranges_tree_A_l496_49673

theorem percentage_good_oranges_tree_A
  (total_trees : ℕ)
  (trees_A : ℕ)
  (trees_B : ℕ)
  (total_good_oranges : ℕ)
  (oranges_A_per_month : ℕ) 
  (oranges_B_per_month : ℕ)
  (good_oranges_B_ratio : ℚ)
  (good_oranges_total_B : ℕ) 
  (good_oranges_total_A : ℕ)
  (good_oranges_total : ℕ)
  (x : ℚ) 
  (total_trees_eq : total_trees = 10)
  (tree_percentage_eq : trees_A = total_trees / 2 ∧ trees_B = total_trees / 2)
  (oranges_A_per_month_eq : oranges_A_per_month = 10)
  (oranges_B_per_month_eq : oranges_B_per_month = 15)
  (good_oranges_B_ratio_eq : good_oranges_B_ratio = 1/3)
  (good_oranges_total_eq : total_good_oranges = 55)
  (good_oranges_total_B_eq : good_oranges_total_B = trees_B * oranges_B_per_month * good_oranges_B_ratio)
  (good_oranges_total_A_eq : good_oranges_total_A = total_good_oranges - good_oranges_total_B):
  trees_A * oranges_A_per_month * x = good_oranges_total_A → 
  x = 0.6 := by
  sorry

end percentage_good_oranges_tree_A_l496_49673


namespace inequality_preservation_l496_49639

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l496_49639


namespace stratified_sampling_first_grade_selection_l496_49620

theorem stratified_sampling_first_grade_selection
  (total_students : ℕ)
  (students_grade1 : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 2000)
  (h_grade1 : students_grade1 = 400)
  (h_sample : sample_size = 200) :
  sample_size * students_grade1 / total_students = 40 := by
  sorry

end stratified_sampling_first_grade_selection_l496_49620


namespace find_A_l496_49682

-- Given a three-digit number AB2 such that AB2 - 41 = 591
def valid_number (A B : ℕ) : Prop :=
  (A * 100) + (B * 10) + 2 - 41 = 591

-- We aim to prove that A = 6 given B = 2
theorem find_A (A : ℕ) (B : ℕ) (hB : B = 2) : A = 6 :=
  by
  have h : valid_number A B := by sorry
  sorry

end find_A_l496_49682


namespace sum_of_three_numbers_l496_49647

theorem sum_of_three_numbers (x y z : ℝ) (h₁ : x + y = 29) (h₂ : y + z = 46) (h₃ : z + x = 53) : x + y + z = 64 :=
by
  sorry

end sum_of_three_numbers_l496_49647


namespace other_pencil_length_l496_49680

-- Definitions based on the conditions identified in a)
def pencil1_length : Nat := 12
def total_length : Nat := 24

-- Problem: Prove that the length of the other pencil (pencil2) is 12 cubes.
theorem other_pencil_length : total_length - pencil1_length = 12 := by 
  sorry

end other_pencil_length_l496_49680


namespace glasses_per_pitcher_l496_49672

theorem glasses_per_pitcher (t p g : ℕ) (ht : t = 54) (hp : p = 9) : g = t / p := by
  rw [ht, hp]
  norm_num
  sorry

end glasses_per_pitcher_l496_49672


namespace common_chord_length_l496_49637

noncomputable def dist_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / Real.sqrt (a^2 + b^2)

theorem common_chord_length
  (x y : ℝ)
  (h1 : (x-2)^2 + (y-1)^2 = 10)
  (h2 : (x+6)^2 + (y+3)^2 = 50) :
  (dist_to_line (2, 1) 2 1 0 = Real.sqrt 5) →
  2 * Real.sqrt 5 = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_length_l496_49637


namespace a2_eq_1_l496_49611

-- Define the geometric sequence and the conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a1_eq_2 : a 1 = 2
axiom condition1 : geometric_sequence a q
axiom condition2 : 16 * a 3 * a 5 = 8 * a 4 - 1

-- Prove that a_2 = 1
theorem a2_eq_1 : a 2 = 1 :=
by
  -- This is where the proof would go
  sorry

end a2_eq_1_l496_49611


namespace Fred_hourly_rate_l496_49653

-- Define the conditions
def hours_worked : ℝ := 8
def total_earned : ℝ := 100

-- Assert the proof goal
theorem Fred_hourly_rate : total_earned / hours_worked = 12.5 :=
by
  sorry

end Fred_hourly_rate_l496_49653


namespace tom_sleep_increase_l496_49649

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end tom_sleep_increase_l496_49649


namespace trip_time_40mph_l496_49602

noncomputable def trip_time_80mph : ℝ := 6.75
noncomputable def speed_80mph : ℝ := 80
noncomputable def speed_40mph : ℝ := 40

noncomputable def distance : ℝ := speed_80mph * trip_time_80mph

theorem trip_time_40mph : distance / speed_40mph = 13.50 :=
by
  sorry

end trip_time_40mph_l496_49602


namespace plaster_cost_correct_l496_49666

def length : ℝ := 25
def width : ℝ := 12
def depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.30

def area_longer_walls : ℝ := 2 * (length * depth)
def area_shorter_walls : ℝ := 2 * (width * depth)
def area_bottom : ℝ := length * width
def total_area : ℝ := area_longer_walls + area_shorter_walls + area_bottom

def calculated_cost : ℝ := total_area * cost_per_sq_meter
def correct_cost : ℝ := 223.2

theorem plaster_cost_correct : calculated_cost = correct_cost := by
  sorry

end plaster_cost_correct_l496_49666


namespace gary_asparagus_l496_49645

/-- Formalization of the problem -/
theorem gary_asparagus (A : ℝ) (ha : 700 * 0.50 = 350) (hg : 40 * 2.50 = 100) (hw : 630 = 3 * A + 350 + 100) : A = 60 :=
by
  sorry

end gary_asparagus_l496_49645


namespace lilith_additional_fund_l496_49629

theorem lilith_additional_fund
  (num_water_bottles : ℕ)
  (original_price : ℝ)
  (reduced_price : ℝ)
  (expected_difference : ℝ)
  (h1 : num_water_bottles = 5 * 12)
  (h2 : original_price = 2)
  (h3 : reduced_price = 1.85)
  (h4 : expected_difference = 9) :
  (num_water_bottles * original_price) - (num_water_bottles * reduced_price) = expected_difference :=
by
  sorry

end lilith_additional_fund_l496_49629


namespace le_condition_l496_49606

-- Given positive numbers a, b, c
variables {a b c : ℝ}
-- Assume positive values for the numbers
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
-- Given condition a² + b² - ab = c²
axiom condition : a^2 + b^2 - a*b = c^2

-- We need to prove (a - c)(b - c) ≤ 0
theorem le_condition : (a - c) * (b - c) ≤ 0 :=
sorry

end le_condition_l496_49606


namespace algebraic_expression_value_l496_49652

theorem algebraic_expression_value (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end algebraic_expression_value_l496_49652


namespace gcd_105_88_l496_49674

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end gcd_105_88_l496_49674


namespace yellow_paint_amount_l496_49624

theorem yellow_paint_amount (b y : ℕ) (h_ratio : y * 7 = 3 * b) (h_blue_amount : b = 21) : y = 9 :=
by
  sorry

end yellow_paint_amount_l496_49624


namespace find_quadruples_l496_49657

theorem find_quadruples (a b p n : ℕ) (h_prime : Prime p) (h_eq : a^3 + b^3 = p^n) :
  ∃ k : ℕ, (a, b, p, n) = (2^k, 2^k, 2, 3*k + 1) ∨ 
           (a, b, p, n) = (3^k, 2 * 3^k, 3, 3*k + 2) ∨ 
           (a, b, p, n) = (2 * 3^k, 3^k, 3, 3*k + 2) :=
sorry

end find_quadruples_l496_49657


namespace total_amount_after_refunds_and_discounts_l496_49665

-- Definitions
def individual_bookings : ℤ := 12000
def group_bookings_before_discount : ℤ := 16000
def discount_rate : ℕ := 10
def refund_individual_1 : ℤ := 500
def count_refund_individual_1 : ℕ := 3
def refund_individual_2 : ℤ := 300
def count_refund_individual_2 : ℕ := 2
def total_refund_group : ℤ := 800

-- Calculation proofs
theorem total_amount_after_refunds_and_discounts : 
(individual_bookings + (group_bookings_before_discount - (discount_rate * group_bookings_before_discount / 100))) - 
((count_refund_individual_1 * refund_individual_1) + (count_refund_individual_2 * refund_individual_2) + total_refund_group) = 23500 := by
    sorry

end total_amount_after_refunds_and_discounts_l496_49665


namespace arithmetic_sequence_S6_by_S4_l496_49633

-- Define the arithmetic sequence and the sum function
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def S1 : ℕ := 1
def r (S2 S4 : ℕ) : Prop := S4 / S2 = 4

-- Proof statement
theorem arithmetic_sequence_S6_by_S4 :
  ∀ (a d : ℕ), 
  (sum_arithmetic_sequence a d 1 = S1) → (r (sum_arithmetic_sequence a d 2) (sum_arithmetic_sequence a d 4)) → 
  (sum_arithmetic_sequence a d 6 / sum_arithmetic_sequence a d 4 = 9 / 4) := 
by
  sorry

end arithmetic_sequence_S6_by_S4_l496_49633


namespace minimum_radius_of_third_sphere_l496_49608

noncomputable def cone_height : ℝ := 4
noncomputable def cone_base_radius : ℝ := 3

noncomputable def radius_identical_spheres : ℝ := 4 / 3  -- derived from the conditions

theorem minimum_radius_of_third_sphere
    (h r1 r2 : ℝ) -- heights and radii one and two
    (R1 R2 Rb : ℝ) -- radii of the common base
    (cond_h : h = 4)
    (cond_Rb : Rb = 3)
    (cond_radii_eq : r1 = r2) 
  : r2 = 27 / 35 :=
by
  sorry

end minimum_radius_of_third_sphere_l496_49608


namespace math_proof_problem_l496_49699

noncomputable def find_value (a b c : ℝ) : ℝ :=
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c))

theorem math_proof_problem (a b c : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a * b + a * c + b * c ≠ 0) :
  find_value a b c = 3 :=
by 
  -- sorry is used as we are only asked to provide the theorem statement in Lean.
  sorry

end math_proof_problem_l496_49699


namespace Balaganov_made_a_mistake_l496_49654

variable (n1 n2 n3 : ℕ) (x : ℝ)
variable (average : ℝ)

def total_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ := 27 * n1 + 35 * n2 + x * n3

def number_of_employees (n1 n2 n3 : ℕ) : ℕ := n1 + n2 + n3

noncomputable def calculated_average_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ :=
 total_salary n1 n2 x n3 / number_of_employees n1 n2 n3

theorem Balaganov_made_a_mistake (h₀ : n1 > n2) 
  (h₁ : calculated_average_salary n1 n2 x n3 = average) 
  (h₂ : 31 < average) : false :=
sorry

end Balaganov_made_a_mistake_l496_49654


namespace sequence_geometric_sequence_general_term_l496_49621

theorem sequence_geometric (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∃ r : ℕ, (a 1 + 1) = 3 ∧ (∀ n, (a (n + 1) + 1) = r * (a n + 1)) := by
  sorry

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 3 * 2^(n-1) - 1 := by
  sorry

end sequence_geometric_sequence_general_term_l496_49621


namespace parabola_translation_l496_49656

theorem parabola_translation :
  ∀(x y : ℝ), y = - (1 / 3) * (x - 5) ^ 2 + 3 →
  ∃(x' y' : ℝ), y' = -(1/3) * x'^2 + 6 := by
  sorry

end parabola_translation_l496_49656


namespace number_of_subcommittees_l496_49651

theorem number_of_subcommittees :
  ∃ (k : ℕ), ∀ (num_people num_sub_subcommittees subcommittee_size : ℕ), 
  num_people = 360 → 
  num_sub_subcommittees = 3 → 
  subcommittee_size = 6 → 
  k = (num_people * num_sub_subcommittees) / subcommittee_size :=
sorry

end number_of_subcommittees_l496_49651


namespace intersection_correct_l496_49613

noncomputable def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def intersection_M_N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_correct : M ∩ N = intersection_M_N :=
by
  sorry

end intersection_correct_l496_49613


namespace heads_not_consecutive_probability_l496_49628

theorem heads_not_consecutive_probability :
  (∃ n m : ℕ, n = 2^4 ∧ m = 1 + Nat.choose 4 1 + Nat.choose 3 2 ∧ (m / n : ℚ) = 1 / 2) :=
by
  use 16     -- n
  use 8      -- m
  sorry

end heads_not_consecutive_probability_l496_49628


namespace cody_money_final_l496_49612

theorem cody_money_final (initial_money : ℕ) (birthday_money : ℕ) (money_spent : ℕ) (final_money : ℕ) 
  (h1 : initial_money = 45) (h2 : birthday_money = 9) (h3 : money_spent = 19) :
  final_money = initial_money + birthday_money - money_spent :=
by {
  sorry  -- The proof is not required here.
}

end cody_money_final_l496_49612
