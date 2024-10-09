import Mathlib

namespace general_term_a_general_term_b_sum_first_n_terms_l1368_136852

def a : Nat → Nat
| 0     => 1
| (n+1) => 2 * a n

def b (n : Nat) : Int :=
  3 * (n + 1) - 2

def S (n : Nat) : Int :=
  2^n - (3 * n^2) / 2 + n / 2 - 1

-- We state the theorems with the conditions included.

theorem general_term_a (n : Nat) : a n = 2^(n - 1) := by
  sorry

theorem general_term_b (n : Nat) : b n = 3 * (n + 1) - 2 := by
  sorry

theorem sum_first_n_terms (n : Nat) : 
  (Finset.range n).sum (λ i => a i - b i) = 2^n - (3 * n^2) / 2 + n / 2 - 1 := by
  sorry

end general_term_a_general_term_b_sum_first_n_terms_l1368_136852


namespace original_smallest_element_l1368_136804

theorem original_smallest_element (x : ℤ) 
  (h1 : x < -1) 
  (h2 : x + 14 + 0 + 6 + 9 = 2 * (2 + 3 + 0 + 6 + 9)) : 
  x = -4 :=
by sorry

end original_smallest_element_l1368_136804


namespace cost_effectiveness_l1368_136815

-- Define general parameters and conditions given in the problem
def a : ℕ := 70 -- We use 70 since it must be greater than 50

-- Define the scenarios
def cost_scenario1 (a: ℕ) : ℕ := 4500 + 27 * a
def cost_scenario2 (a: ℕ) : ℕ := 4400 + 30 * a

-- The theorem to be proven
theorem cost_effectiveness (h : a > 50) : cost_scenario1 a < cost_scenario2 a :=
  by
  -- First, let's replace a with 70 (this step is unnecessary in the proof since a = 70 is fixed)
  let a := 70
  -- Now, prove the inequality
  sorry

end cost_effectiveness_l1368_136815


namespace trigonometric_identity_l1368_136851

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π / 4 + α) = 1 / 2) : 
  (Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α)) * Real.cos (7 * π / 4 - α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l1368_136851


namespace SplitWinnings_l1368_136809

noncomputable def IstvanInitialContribution : ℕ := 5000 * 20
noncomputable def IstvanSecondPeriodContribution : ℕ := (5000 + 4000) * 30
noncomputable def IstvanThirdPeriodContribution : ℕ := (5000 + 4000 - 2500) * 40
noncomputable def IstvanTotalContribution : ℕ := IstvanInitialContribution + IstvanSecondPeriodContribution + IstvanThirdPeriodContribution

noncomputable def KalmanContribution : ℕ := 4000 * 70
noncomputable def LaszloContribution : ℕ := 2500 * 40
noncomputable def MiklosContributionAdjustment : ℕ := 2000 * 90

noncomputable def IstvanExpectedShare : ℕ := IstvanTotalContribution * 12 / 100
noncomputable def KalmanExpectedShare : ℕ := KalmanContribution * 12 / 100
noncomputable def LaszloExpectedShare : ℕ := LaszloContribution * 12 / 100
noncomputable def MiklosExpectedShare : ℕ := MiklosContributionAdjustment * 12 / 100

noncomputable def IstvanActualShare : ℕ := IstvanExpectedShare * 7 / 8
noncomputable def KalmanActualShare : ℕ := (KalmanExpectedShare - MiklosExpectedShare) * 7 / 8
noncomputable def LaszloActualShare : ℕ := LaszloExpectedShare * 7 / 8
noncomputable def MiklosActualShare : ℕ := MiklosExpectedShare * 7 / 8

theorem SplitWinnings :
  IstvanActualShare = 54600 ∧ KalmanActualShare = 7800 ∧ LaszloActualShare = 10500 ∧ MiklosActualShare = 18900 :=
by
  sorry

end SplitWinnings_l1368_136809


namespace cost_of_one_jacket_l1368_136835

theorem cost_of_one_jacket
  (S J : ℝ)
  (h1 : 10 * S + 20 * J = 800)
  (h2 : 5 * S + 15 * J = 550) : J = 30 :=
sorry

end cost_of_one_jacket_l1368_136835


namespace remainder_sum_first_150_div_11300_l1368_136867

theorem remainder_sum_first_150_div_11300 :
  let n := 150
  let S := n * (n + 1) / 2
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l1368_136867


namespace same_terminal_side_of_minus_80_l1368_136823

theorem same_terminal_side_of_minus_80 :
  ∃ k : ℤ, 1 * 360 - 80 = 280 := 
  sorry

end same_terminal_side_of_minus_80_l1368_136823


namespace line_equation_through_point_with_intercepts_conditions_l1368_136870

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end line_equation_through_point_with_intercepts_conditions_l1368_136870


namespace count_adjacent_pairs_sum_multiple_of_three_l1368_136893

def adjacent_digit_sum_multiple_of_three (n : ℕ) : ℕ :=
  -- A function to count the number of pairs with a sum multiple of 3
  sorry

-- Define the sequence from 100 to 999 as digits concatenation
def digit_sequence : List ℕ := List.join (List.map (fun x => x.digits 10) (List.range' 100 900))

theorem count_adjacent_pairs_sum_multiple_of_three :
  adjacent_digit_sum_multiple_of_three digit_sequence.length = 897 :=
sorry

end count_adjacent_pairs_sum_multiple_of_three_l1368_136893


namespace total_toys_l1368_136886

theorem total_toys (bill_toys hana_toys hash_toys: ℕ) 
  (hb: bill_toys = 60)
  (hh: hana_toys = (5 * bill_toys) / 6)
  (hs: hash_toys = (hana_toys / 2) + 9) :
  (bill_toys + hana_toys + hash_toys) = 144 :=
by
  sorry

end total_toys_l1368_136886


namespace quadrilateral_area_l1368_136894

theorem quadrilateral_area 
  (d : ℝ) (h₁ h₂ : ℝ) 
  (hd : d = 22) 
  (hh₁ : h₁ = 9) 
  (hh₂ : h₂ = 6) : 
  (1/2 * d * h₁ + 1/2 * d * h₂ = 165) :=
by
  sorry

end quadrilateral_area_l1368_136894


namespace negation_of_existence_l1368_136889

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_existence_l1368_136889


namespace arithmetic_sequence_term_count_l1368_136827

theorem arithmetic_sequence_term_count (a d l n : ℕ) (h1 : a = 11) (h2 : d = 4) (h3 : l = 107) :
  l = a + (n - 1) * d → n = 25 := by
  sorry

end arithmetic_sequence_term_count_l1368_136827


namespace find_abc_value_l1368_136881

noncomputable def given_conditions (a b c : ℝ) : Prop :=
  (a * b / (a + b) = 2) ∧ (b * c / (b + c) = 5) ∧ (c * a / (c + a) = 9)

theorem find_abc_value (a b c : ℝ) (h : given_conditions a b c) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 :=
sorry

end find_abc_value_l1368_136881


namespace profit_calculation_l1368_136860

variable (price : ℕ) (cost : ℕ) (exchange_rate : ℕ) (profit_per_bottle : ℚ)

-- Conditions
def conditions := price = 2 ∧ cost = 1 ∧ exchange_rate = 5

-- Profit per bottle is 0.66 yuan considering the exchange policy
theorem profit_calculation (h : conditions price cost exchange_rate) : profit_per_bottle = 0.66 := sorry

end profit_calculation_l1368_136860


namespace envelope_of_family_of_lines_l1368_136896

theorem envelope_of_family_of_lines (a α : ℝ) (hα : α > 0) :
    ∀ (x y : ℝ), (∃ α > 0,
    (x = a * α / 2 ∧ y = a / (2 * α))) ↔ (x * y = a^2 / 4) := by
  sorry

end envelope_of_family_of_lines_l1368_136896


namespace determine_y_l1368_136888

variable {R : Type} [LinearOrderedField R]
variables {x y : R}

theorem determine_y (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x + 9 * y = 6) : y = -4 / 15 :=
by
  sorry

end determine_y_l1368_136888


namespace no_common_root_l1368_136862

theorem no_common_root 
  (a b : ℚ) 
  (α : ℂ) 
  (h1 : α^5 = α + 1) 
  (h2 : α^2 = -a * α - b) : 
  False :=
sorry

end no_common_root_l1368_136862


namespace simplify_expression_l1368_136802

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l1368_136802


namespace angle_C_in_parallelogram_l1368_136832

theorem angle_C_in_parallelogram (ABCD : Type)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A = angle_C)
  (h2 : angle_B = angle_D)
  (h3 : angle_A + angle_B = 180)
  (h4 : angle_A / angle_B = 3) :
  angle_C = 135 :=
  sorry

end angle_C_in_parallelogram_l1368_136832


namespace eq_solution_l1368_136868

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end eq_solution_l1368_136868


namespace direction_vector_correct_l1368_136854

open Real

def line_eq (x y : ℝ) : Prop := x - 3 * y + 1 = 0

noncomputable def direction_vector : ℝ × ℝ := (3, 1)

theorem direction_vector_correct (x y : ℝ) (h : line_eq x y) : 
    ∃ k : ℝ, direction_vector = (k * (1 : ℝ), k * (1 / 3)) :=
by
  use 3
  sorry

end direction_vector_correct_l1368_136854


namespace time_brushing_each_cat_l1368_136890

theorem time_brushing_each_cat :
  ∀ (t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats : ℕ),
  t_total_free_time = 3 * 60 →
  t_vacuum = 45 →
  t_dust = 60 →
  t_mop = 30 →
  t_cats = 3 →
  t_free_left_after_cleaning = 30 →
  ((t_total_free_time - t_free_left_after_cleaning) - (t_vacuum + t_dust + t_mop)) / t_cats = 5
 := by
  intros t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats
  intros h_total_free_time h_vacuum h_dust h_mop h_cats h_free_left
  sorry

end time_brushing_each_cat_l1368_136890


namespace fraction_computation_l1368_136830

theorem fraction_computation : (2 / 3) * (3 / 4 * 40) = 20 := 
by
  -- The proof will go here, for now we use sorry to skip the proof.
  sorry

end fraction_computation_l1368_136830


namespace solve_for_x_l1368_136847

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solve_for_x (x : ℝ) 
  (h1 : infinite_power_tower x = 4) : 
  x = Real.sqrt 2 := 
sorry

end solve_for_x_l1368_136847


namespace problem_value_expression_l1368_136891

theorem problem_value_expression 
  (x y : ℝ)
  (h₁ : x + y = 4)
  (h₂ : x * y = -2) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := 
sorry

end problem_value_expression_l1368_136891


namespace rectangle_sides_l1368_136814

theorem rectangle_sides (S d : ℝ) (a b : ℝ) : 
  a = Real.sqrt (S + d^2 / 4) + d / 2 ∧ 
  b = Real.sqrt (S + d^2 / 4) - d / 2 →
  S = a * b ∧ d = a - b :=
by
  -- definitions and conditions will be used here in the proofs
  sorry

end rectangle_sides_l1368_136814


namespace second_share_interest_rate_is_11_l1368_136873

noncomputable def calculate_interest_rate 
    (total_investment : ℝ)
    (amount_in_second_share : ℝ)
    (interest_rate_first : ℝ)
    (total_interest : ℝ) : ℝ := 
  let A := total_investment - amount_in_second_share
  let interest_first := (interest_rate_first / 100) * A
  let interest_second := total_interest - interest_first
  (100 * interest_second) / amount_in_second_share

theorem second_share_interest_rate_is_11 :
  calculate_interest_rate 100000 12499.999999999998 9 9250 = 11 := 
by
  sorry

end second_share_interest_rate_is_11_l1368_136873


namespace total_time_to_clean_and_complete_l1368_136882

def time_to_complete_assignment : Nat := 10
def num_remaining_keys : Nat := 14
def time_per_key : Nat := 3

theorem total_time_to_clean_and_complete :
  time_to_complete_assignment + num_remaining_keys * time_per_key = 52 :=
by
  sorry

end total_time_to_clean_and_complete_l1368_136882


namespace question_equals_answer_l1368_136831

theorem question_equals_answer (x y : ℝ) (h : abs (x - 6) + (y + 4)^2 = 0) : x + y = 2 :=
sorry

end question_equals_answer_l1368_136831


namespace cube_volume_l1368_136836

variables (x s : ℝ)
theorem cube_volume (h : 6 * s^2 = 6 * x^2) : s^3 = x^3 :=
by sorry

end cube_volume_l1368_136836


namespace annual_cost_l1368_136897

def monday_miles : ℕ := 50
def wednesday_miles : ℕ := 50
def friday_miles : ℕ := 50
def sunday_miles : ℕ := 50

def tuesday_miles : ℕ := 100
def thursday_miles : ℕ := 100
def saturday_miles : ℕ := 100

def cost_per_mile : ℝ := 0.1
def weekly_fee : ℝ := 100
def weeks_in_year : ℕ := 52

noncomputable def total_weekly_miles : ℕ := 
  (monday_miles + wednesday_miles + friday_miles + sunday_miles) * 1 +
  (tuesday_miles + thursday_miles + saturday_miles) * 1

noncomputable def weekly_mileage_cost : ℝ := total_weekly_miles * cost_per_mile

noncomputable def weekly_total_cost : ℝ := weekly_fee + weekly_mileage_cost

noncomputable def annual_total_cost : ℝ := weekly_total_cost * weeks_in_year

theorem annual_cost (monday_miles wednesday_miles friday_miles sunday_miles
                     tuesday_miles thursday_miles saturday_miles : ℕ)
                     (cost_per_mile weekly_fee : ℝ) 
                     (weeks_in_year : ℕ) :
  monday_miles = 50 → wednesday_miles = 50 → friday_miles = 50 → sunday_miles = 50 →
  tuesday_miles = 100 → thursday_miles = 100 → saturday_miles = 100 →
  cost_per_mile = 0.1 → weekly_fee = 100 → weeks_in_year = 52 →
  annual_total_cost = 7800 :=
by
  intros
  sorry

end annual_cost_l1368_136897


namespace age_difference_l1368_136857

theorem age_difference (P M Mo : ℕ) (h1 : P = (3 * M) / 5) (h2 : Mo = (4 * M) / 3) (h3 : P + M + Mo = 88) : Mo - P = 22 := 
by sorry

end age_difference_l1368_136857


namespace banana_distribution_correct_l1368_136887

noncomputable def proof_problem : Prop :=
  let bananas := 40
  let marbles := 4
  let boys := 18
  let girls := 12
  let total_friends := 30
  let bananas_for_boys := (3/8 : ℝ) * bananas
  let bananas_for_girls := (1/4 : ℝ) * bananas
  let bananas_left := bananas - (bananas_for_boys + bananas_for_girls)
  let bananas_per_marble := bananas_left / marbles
  bananas_for_boys = 15 ∧ bananas_for_girls = 10 ∧ bananas_per_marble = 3.75

theorem banana_distribution_correct : proof_problem :=
by
  -- Proof is omitted
  sorry

end banana_distribution_correct_l1368_136887


namespace equilateral_triangle_side_length_l1368_136842

theorem equilateral_triangle_side_length (total_length : ℕ) (h1 : total_length = 78) : (total_length / 3) = 26 :=
by
  sorry

end equilateral_triangle_side_length_l1368_136842


namespace calculation_result_l1368_136824

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end calculation_result_l1368_136824


namespace integer_solutions_l1368_136839

theorem integer_solutions (n : ℤ) : ∃ m : ℤ, n^2 + 15 = m^2 ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 :=
by
  sorry

end integer_solutions_l1368_136839


namespace total_distance_traveled_by_children_l1368_136878

theorem total_distance_traveled_by_children :
  let ap := 50
  let dist_1_vertex_skip := (50 : ℝ) * Real.sqrt 2
  let dist_2_vertices_skip := (50 : ℝ) * Real.sqrt (2 + 2 * Real.sqrt 2)
  let dist_diameter := (2 : ℝ) * 50
  let single_child_distance := 2 * dist_1_vertex_skip + 2 * dist_2_vertices_skip + dist_diameter
  8 * single_child_distance = 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + 2 * Real.sqrt 2) + 800 :=
sorry

end total_distance_traveled_by_children_l1368_136878


namespace total_fireworks_l1368_136876

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end total_fireworks_l1368_136876


namespace decagon_area_l1368_136849

theorem decagon_area 
    (perimeter_square : ℝ) 
    (side_division : ℕ) 
    (side_length : ℝ) 
    (triangle_area : ℝ) 
    (total_triangle_area : ℝ) 
    (square_area : ℝ)
    (decagon_area : ℝ) :
    perimeter_square = 150 →
    side_division = 5 →
    side_length = perimeter_square / 4 →
    triangle_area = 1 / 2 * (side_length / side_division) * (side_length / side_division) →
    total_triangle_area = 8 * triangle_area →
    square_area = side_length * side_length →
    decagon_area = square_area - total_triangle_area →
    decagon_area = 1181.25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end decagon_area_l1368_136849


namespace sector_area_l1368_136880

theorem sector_area (θ r : ℝ) (hθ : θ = 2) (hr : r = 1) :
  (1 / 2) * r^2 * θ = 1 :=
by
  -- Conditions are instantiated
  rw [hθ, hr]
  -- Simplification is left to the proof
  sorry

end sector_area_l1368_136880


namespace find_other_diagonal_l1368_136840

theorem find_other_diagonal (A : ℝ) (d1 : ℝ) (hA : A = 80) (hd1 : d1 = 16) :
  ∃ d2 : ℝ, 2 * A / d1 = d2 :=
by
  use 10
  -- Rest of the proof goes here
  sorry

end find_other_diagonal_l1368_136840


namespace train_passing_time_l1368_136879

-- conditions
def train_length := 490 -- in meters
def train_speed_kmh := 63 -- in kilometers per hour
def conversion_factor := 1000 / 3600 -- to convert km/hr to m/s

-- conversion
def train_speed_ms := train_speed_kmh * conversion_factor -- speed in meters per second

-- expected correct answer
def expected_time := 28 -- in seconds

-- Theorem statement
theorem train_passing_time :
  train_length / train_speed_ms = expected_time :=
by
  sorry

end train_passing_time_l1368_136879


namespace negation_proof_l1368_136845

theorem negation_proof : ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by
  sorry

end negation_proof_l1368_136845


namespace trigonometric_identity_l1368_136833

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) + Real.sin (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α) = -1 :=
by
  sorry

end trigonometric_identity_l1368_136833


namespace polygon_sides_l1368_136892

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l1368_136892


namespace find_C_l1368_136864

theorem find_C (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 :=
sorry

end find_C_l1368_136864


namespace correct_solutions_l1368_136872

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ (x y : ℝ), f (x * y) = f x * f y - 2 * x * y

theorem correct_solutions :
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) := sorry

end correct_solutions_l1368_136872


namespace marbles_each_friend_gets_l1368_136875

-- Definitions for the conditions
def total_marbles : ℕ := 100
def marbles_kept : ℕ := 20
def number_of_friends : ℕ := 5

-- The math proof problem
theorem marbles_each_friend_gets :
  (total_marbles - marbles_kept) / number_of_friends = 16 :=
by
  -- We include the proof steps within a by notation but stop at sorry for automated completion skipping proof steps
  sorry

end marbles_each_friend_gets_l1368_136875


namespace temperature_at_midnight_l1368_136884

-- Define temperature in the morning
def T_morning := -2 -- in degrees Celsius

-- Temperature change at noon
def delta_noon := 12 -- in degrees Celsius

-- Temperature change at midnight
def delta_midnight := -8 -- in degrees Celsius

-- Function to compute temperature
def compute_temperature (T : ℤ) (delta1 : ℤ) (delta2 : ℤ) : ℤ :=
  T + delta1 + delta2

-- The proposition to prove
theorem temperature_at_midnight :
  compute_temperature T_morning delta_noon delta_midnight = 2 :=
by
  sorry

end temperature_at_midnight_l1368_136884


namespace negative_integer_solution_l1368_136818

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l1368_136818


namespace opposite_of_neg_2022_l1368_136861

theorem opposite_of_neg_2022 : -(-2022) = 2022 :=
by
  sorry

end opposite_of_neg_2022_l1368_136861


namespace kanul_total_amount_l1368_136825

-- Definitions based on the conditions
def raw_materials_cost : ℝ := 35000
def machinery_cost : ℝ := 40000
def marketing_cost : ℝ := 15000
def total_spent : ℝ := raw_materials_cost + machinery_cost + marketing_cost
def spending_percentage : ℝ := 0.25

-- The statement we want to prove
theorem kanul_total_amount (T : ℝ) (h : total_spent = spending_percentage * T) : T = 360000 :=
by
  sorry

end kanul_total_amount_l1368_136825


namespace evaluate_expression_l1368_136838

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l1368_136838


namespace ratio_MN_l1368_136871

variables (Q P R M N : ℝ)

def satisfies_conditions (Q P R M N : ℝ) : Prop :=
  M = 0.40 * Q ∧
  Q = 0.25 * P ∧
  R = 0.60 * P ∧
  N = 0.50 * R

theorem ratio_MN (Q P R M N : ℝ) (h : satisfies_conditions Q P R M N) : M / N = 1 / 3 :=
by {
  sorry
}

end ratio_MN_l1368_136871


namespace shirts_total_cost_l1368_136895

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l1368_136895


namespace sqrt_domain_l1368_136822

theorem sqrt_domain (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
sorry

end sqrt_domain_l1368_136822


namespace value_of_expression_l1368_136811

theorem value_of_expression : 
  ∀ (a x y : ℤ), 
  (x = a + 5) → 
  (a = 20) → 
  (y = 25) → 
  (x - y) * (x + y) = 0 :=
by
  intros a x y h1 h2 h3
  -- proof goes here
  sorry

end value_of_expression_l1368_136811


namespace fred_gave_sandy_balloons_l1368_136816

theorem fred_gave_sandy_balloons :
  ∀ (original_balloons given_balloons final_balloons : ℕ),
    original_balloons = 709 →
    final_balloons = 488 →
    given_balloons = original_balloons - final_balloons →
    given_balloons = 221 := by
  sorry

end fred_gave_sandy_balloons_l1368_136816


namespace isosceles_triangle_solution_l1368_136813

noncomputable def isosceles_triangle_sides (x y : ℝ) : Prop :=
(x + 1/2 * y = 6 ∧ 1/2 * x + y = 12) ∨ (x + 1/2 * y = 12 ∧ 1/2 * x + y = 6)

theorem isosceles_triangle_solution :
  ∃ (x y : ℝ), isosceles_triangle_sides x y ∧ x = 8 ∧ y = 2 :=
sorry

end isosceles_triangle_solution_l1368_136813


namespace events_are_mutually_exclusive_but_not_opposite_l1368_136800

-- Definitions based on the conditions:
structure BallBoxConfig where
  ball1 : Fin 4 → ℕ     -- Function representing the placement of ball number 1 into one of the 4 boxes
  h_distinct : ∀ i j, i ≠ j → ball1 i ≠ ball1 j

def event_A (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 1
def event_B (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 2

-- The proof problem:
theorem events_are_mutually_exclusive_but_not_opposite (cfg : BallBoxConfig) :
  (event_A cfg ∨ event_B cfg) ∧ ¬ (event_A cfg ∧ event_B cfg) :=
sorry

end events_are_mutually_exclusive_but_not_opposite_l1368_136800


namespace determine_h_l1368_136877

-- Define the initial quadratic expression
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the form we want to prove
def completed_square_form (x h k : ℝ) : ℝ := 3 * (x - h)^2 + k

-- The proof problem translated to Lean 4
theorem determine_h : ∃ k : ℝ, ∀ x : ℝ, quadratic x = completed_square_form x (-4 / 3) k :=
by
  exists (29 / 3)
  intro x
  sorry

end determine_h_l1368_136877


namespace gain_percent_is_80_l1368_136846

noncomputable def cost_price : ℝ := 600
noncomputable def selling_price : ℝ := 1080
noncomputable def gain : ℝ := selling_price - cost_price
noncomputable def gain_percent : ℝ := (gain / cost_price) * 100

theorem gain_percent_is_80 :
  gain_percent = 80 := by
  sorry

end gain_percent_is_80_l1368_136846


namespace determine_right_triangle_l1368_136820

theorem determine_right_triangle (a b c : ℕ) :
  (∀ c b, (c + b) * (c - b) = a^2 → c^2 = a^2 + b^2) ∧
  (∀ A B C, A + B = C → C = 90) ∧
  (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 → a^2 + b^2 ≠ c^2) ∧
  (a = 5 ∧ b = 12 ∧ c = 13 → a^2 + b^2 = c^2) → 
  ( ∃ x y z : ℕ, x = a ∧ y = b ∧ z = c ∧ x^2 + y^2 ≠ z^2 )
:= by
  sorry

end determine_right_triangle_l1368_136820


namespace equation_has_exactly_one_real_solution_l1368_136841

-- Definitions for the problem setup
def equation (k : ℝ) (x : ℝ) : Prop := (3 * x + 8) * (x - 6) = -54 + k * x

-- The property that we need to prove
theorem equation_has_exactly_one_real_solution (k : ℝ) :
  (∀ x : ℝ, equation k x → ∃! x : ℝ, equation k x) ↔ k = 6 * Real.sqrt 2 - 10 ∨ k = -6 * Real.sqrt 2 - 10 := 
sorry

end equation_has_exactly_one_real_solution_l1368_136841


namespace courier_problem_l1368_136808

variable (x : ℝ) -- Let x represent the specified time in minutes
variable (d : ℝ) -- Let d represent the total distance traveled in km

theorem courier_problem
  (h1 : 1.2 * (x - 10) = d)
  (h2 : 0.8 * (x + 5) = d) :
  x = 40 ∧ d = 36 :=
by
  -- This theorem statement encapsulates the conditions and the answer.
  sorry

end courier_problem_l1368_136808


namespace geometric_sequence_product_l1368_136819

/-- Given a geometric sequence with positive terms where a_3 = 3 and a_6 = 1/9,
    prove that a_4 * a_5 = 1/3. -/
theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
    (h_geometric : ∀ n, a (n + 1) = a n * q) (ha3 : a 3 = 3) (ha6 : a 6 = 1 / 9) :
  a 4 * a 5 = 1 / 3 := 
by
  sorry

end geometric_sequence_product_l1368_136819


namespace required_run_rate_per_batsman_l1368_136828

variable (initial_run_rate : ℝ) (overs_played : ℕ) (remaining_overs : ℕ)
variable (remaining_wickets : ℕ) (total_target : ℕ) 

theorem required_run_rate_per_batsman 
  (h_initial_run_rate : initial_run_rate = 3.4)
  (h_overs_played : overs_played = 10)
  (h_remaining_overs  : remaining_overs = 40)
  (h_remaining_wickets : remaining_wickets = 7)
  (h_total_target : total_target = 282) :
  (total_target - initial_run_rate * overs_played) / remaining_overs = 6.2 :=
by
  sorry

end required_run_rate_per_batsman_l1368_136828


namespace ironman_age_l1368_136834

theorem ironman_age (T C P I : ℕ) (h1 : T = 13 * C) (h2 : C = 7 * P) (h3 : I = P + 32) (h4 : T = 1456) : I = 48 := 
by
  sorry

end ironman_age_l1368_136834


namespace solve_for_y_l1368_136848

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l1368_136848


namespace floor_equation_l1368_136810

theorem floor_equation (n : ℤ) (h : ⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋^2 = 5) : n = 11 :=
sorry

end floor_equation_l1368_136810


namespace initial_budget_calculation_l1368_136855

variable (flaskCost testTubeCost safetyGearCost totalExpenses remainingAmount initialBudget : ℕ)

theorem initial_budget_calculation (h1 : flaskCost = 150)
                               (h2 : testTubeCost = 2 * flaskCost / 3)
                               (h3 : safetyGearCost = testTubeCost / 2)
                               (h4 : totalExpenses = flaskCost + testTubeCost + safetyGearCost)
                               (h5 : remainingAmount = 25)
                               (h6 : initialBudget = totalExpenses + remainingAmount) :
                               initialBudget = 325 := by
  sorry

end initial_budget_calculation_l1368_136855


namespace pyramid_volume_correct_l1368_136805

noncomputable def volume_of_pyramid (l α β : ℝ) (Hα : α = π/8) (Hβ : β = π/4) :=
  (1 / 3) * (l^3 / 24) * Real.sqrt (Real.sqrt 2 + 1)

theorem pyramid_volume_correct :
  ∀ (l : ℝ), l = 6 → volume_of_pyramid l (π/8) (π/4) (rfl) (rfl) = 9 * Real.sqrt (Real.sqrt 2 + 1) :=
by
  intros l hl
  rw [hl]
  norm_num
  sorry

end pyramid_volume_correct_l1368_136805


namespace ratio_of_ages_l1368_136812

variables (X Y : ℕ)

theorem ratio_of_ages (h1 : X - 6 = 24) (h2 : X + Y = 36) : X / Y = 2 :=
by 
  have h3 : X = 30 - 6 := by sorry
  have h4 : X = 24 := by sorry
  have h5 : X + Y = 36 := by sorry
  have h6 : Y = 12 := by sorry
  have h7 : X / Y = 2 := by sorry
  exact h7

end ratio_of_ages_l1368_136812


namespace notebook_cost_l1368_136885

theorem notebook_cost (s n c : ℕ) (h1 : s > 25)
                                 (h2 : n % 2 = 1)
                                 (h3 : n > 1)
                                 (h4 : c > n)
                                 (h5 : s * n * c = 2739) :
  c = 7 :=
sorry

end notebook_cost_l1368_136885


namespace set_union_inter_eq_l1368_136803

open Set

-- Conditions: Definitions of sets M, N, and P
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3, 4, 5}

-- Claim: The result of (M ∩ N) ∪ P equals {1, 2, 3, 4, 5}
theorem set_union_inter_eq :
  (M ∩ N ∪ P) = {1, 2, 3, 4, 5} := 
by
  sorry

end set_union_inter_eq_l1368_136803


namespace intersection_A_B_l1368_136807

-- Defining sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l1368_136807


namespace expression_value_l1368_136821

theorem expression_value : ((40 + 15) ^ 2 - 15 ^ 2) = 2800 := 
by
  sorry

end expression_value_l1368_136821


namespace percent_of_srp_bob_paid_l1368_136899

theorem percent_of_srp_bob_paid (SRP MP PriceBobPaid : ℝ) 
  (h1 : MP = 0.60 * SRP)
  (h2 : PriceBobPaid = 0.60 * MP) :
  (PriceBobPaid / SRP) * 100 = 36 := by
  sorry

end percent_of_srp_bob_paid_l1368_136899


namespace areaOfTangencyTriangle_l1368_136856

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaABC (a b c : ℝ) : ℝ :=
  let p := semiPerimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

noncomputable def excircleRadius (a b c : ℝ) : ℝ :=
  let S := areaABC a b c
  let p := semiPerimeter a b c
  S / (p - a)

theorem areaOfTangencyTriangle (a b c R : ℝ) :
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  (S * (ra / (2 * R))) = (S ^ 2 / (2 * R * (p - a))) :=
by
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  sorry

end areaOfTangencyTriangle_l1368_136856


namespace product_divisible_by_4_l1368_136866

theorem product_divisible_by_4 (a b c d : ℤ) 
    (h : a^2 + b^2 + c^2 = d^2) : 4 ∣ (a * b * c) :=
sorry

end product_divisible_by_4_l1368_136866


namespace trader_profit_percentage_l1368_136863

-- Definitions for the conditions
def original_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.80 * P
def selling_price (P : ℝ) : ℝ := 0.80 * P * 1.45

-- Theorem statement including the problem's question and the correct answer
theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) : 
  (selling_price P - original_price P) / original_price P * 100 = 16 :=
by
  sorry

end trader_profit_percentage_l1368_136863


namespace not_necessarily_divisor_of_44_l1368_136865

theorem not_necessarily_divisor_of_44 {k : ℤ} (h1 : ∃ k, n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) :
  ¬(44 ∣ n) :=
sorry

end not_necessarily_divisor_of_44_l1368_136865


namespace july_16_2010_is_wednesday_l1368_136853

-- Define necessary concepts for the problem

def is_tuesday (d : ℕ) : Prop := (d % 7 = 2)
def day_after_n_days (d n : ℕ) : ℕ := (d + n) % 7

-- The statement we want to prove
theorem july_16_2010_is_wednesday (h : is_tuesday 1) : day_after_n_days 1 15 = 3 := 
sorry

end july_16_2010_is_wednesday_l1368_136853


namespace solve_inequality_l1368_136858

open Set

-- Define a predicate for the inequality solution sets
def inequality_solution_set (k : ℝ) : Set ℝ :=
  if h : k = 0 then {x | x < 1}
  else if h : 0 < k ∧ k < 2 then {x | x < 1 ∨ x > 2 / k}
  else if h : k = 2 then {x | True} \ {1}
  else if h : k > 2 then {x | x < 2 / k ∨ x > 1}
  else {x | 2 / k < x ∧ x < 1}

-- The statement of the proof
theorem solve_inequality (k : ℝ) :
  ∀ x : ℝ, k * x^2 - (k + 2) * x + 2 < 0 ↔ x ∈ inequality_solution_set k :=
by
  sorry

end solve_inequality_l1368_136858


namespace vertex_of_parabola_l1368_136843

theorem vertex_of_parabola (x y : ℝ) : (y^2 - 4 * y + 3 * x + 7 = 0) → (x, y) = (-1, 2) :=
by
  sorry

end vertex_of_parabola_l1368_136843


namespace ratio_of_triangle_side_to_rectangle_width_l1368_136869

theorem ratio_of_triangle_side_to_rectangle_width
  (t w : ℕ)
  (ht : 3 * t = 24)
  (hw : 6 * w = 24) :
  t / w = 2 := by
  sorry

end ratio_of_triangle_side_to_rectangle_width_l1368_136869


namespace MapleLeafHigh_points_l1368_136837

def MapleLeafHigh (x y : ℕ) : Prop :=
  (1/3 * x + 3/8 * x + 18 + y = x) ∧ (10 ≤ y) ∧ (y ≤ 30)

theorem MapleLeafHigh_points : ∃ y, MapleLeafHigh 104 y ∧ y = 21 := 
by
  use 21
  sorry

end MapleLeafHigh_points_l1368_136837


namespace inequality_always_holds_true_l1368_136801

theorem inequality_always_holds_true (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) :
  (a / (c^2 + 1)) > (b / (c^2 + 1)) :=
by
  sorry

end inequality_always_holds_true_l1368_136801


namespace yogurt_cost_l1368_136826

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l1368_136826


namespace words_per_page_is_106_l1368_136883

noncomputable def book_pages := 154
noncomputable def max_words_per_page := 120
noncomputable def total_words_mod := 221
noncomputable def mod_val := 217

def number_of_words_per_page (p : ℕ) : Prop :=
  (book_pages * p ≡ total_words_mod [MOD mod_val]) ∧ (p ≤ max_words_per_page)

theorem words_per_page_is_106 : number_of_words_per_page 106 :=
by
  sorry

end words_per_page_is_106_l1368_136883


namespace inequality_holds_for_positive_x_l1368_136850

theorem inequality_holds_for_positive_x (x : ℝ) (h : x > 0) : 
  x^8 - x^5 - 1/x + 1/(x^4) ≥ 0 := 
sorry

end inequality_holds_for_positive_x_l1368_136850


namespace arithmetic_expression_eval_l1368_136874

theorem arithmetic_expression_eval : 8 / 4 - 3 - 9 + 3 * 9 = 17 :=
by
  sorry

end arithmetic_expression_eval_l1368_136874


namespace plastic_skulls_number_l1368_136859

-- Define the conditions
def num_broomsticks : ℕ := 4
def num_spiderwebs : ℕ := 12
def num_pumpkins := 2 * num_spiderwebs
def num_cauldron : ℕ := 1
def budget_left_to_buy : ℕ := 20
def num_left_to_put_up : ℕ := 10
def total_decorations : ℕ := 83

-- The number of plastic skulls calculation as a function
def num_other_decorations : ℕ :=
  num_broomsticks + num_spiderwebs + num_pumpkins + num_cauldron + budget_left_to_buy + num_left_to_put_up

def num_plastic_skulls := total_decorations - num_other_decorations

-- The theorem to be proved
theorem plastic_skulls_number : num_plastic_skulls = 12 := by
  sorry

end plastic_skulls_number_l1368_136859


namespace range_of_m_for_inequality_l1368_136817

theorem range_of_m_for_inequality (m : Real) : 
  (∀ (x : Real), 1 < x ∧ x < 2 → x^2 + m * x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end range_of_m_for_inequality_l1368_136817


namespace analogous_to_tetrahedron_is_triangle_l1368_136844

-- Define the objects as types
inductive Object
| Quadrilateral
| Pyramid
| Triangle
| Prism
| Tetrahedron

-- Define the analogous relationship
def analogous (a b : Object) : Prop :=
  (a = Object.Tetrahedron ∧ b = Object.Triangle)
  ∨ (b = Object.Tetrahedron ∧ a = Object.Triangle)

-- The main statement to prove
theorem analogous_to_tetrahedron_is_triangle :
  ∃ (x : Object), analogous Object.Tetrahedron x ∧ x = Object.Triangle :=
by
  sorry

end analogous_to_tetrahedron_is_triangle_l1368_136844


namespace laps_needed_to_reach_total_distance_l1368_136806

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end laps_needed_to_reach_total_distance_l1368_136806


namespace valid_marble_arrangements_eq_48_l1368_136898

def ZaraMarbleArrangements (n : ℕ) : ℕ := sorry

theorem valid_marble_arrangements_eq_48 : ZaraMarbleArrangements 5 = 48 := sorry

end valid_marble_arrangements_eq_48_l1368_136898


namespace g_value_at_50_l1368_136829

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, 0 < x → 0 < y → x * g y + y * g x = g (x * y)) :
  g 50 = 0 :=
sorry

end g_value_at_50_l1368_136829
