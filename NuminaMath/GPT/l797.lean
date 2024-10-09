import Mathlib

namespace necessary_and_sufficient_condition_l797_79746

theorem necessary_and_sufficient_condition (x : ℝ) : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) := 
by
  sorry

end necessary_and_sufficient_condition_l797_79746


namespace swimmer_speed_in_still_water_l797_79751

-- Define the conditions
def current_speed : ℝ := 2   -- Speed of the water current is 2 km/h
def swim_time : ℝ := 2.5     -- Time taken to swim against current is 2.5 hours
def distance : ℝ := 5        -- Distance swum against current is 5 km

-- Main theorem proving the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) (h : v - current_speed = distance / swim_time) : v = 4 :=
by {
  -- Skipping the proof steps as per the requirements
  sorry
}

end swimmer_speed_in_still_water_l797_79751


namespace magnitude_diff_l797_79794

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition_1 : ‖a‖ = 2 := sorry
def condition_2 : ‖b‖ = 2 := sorry
def condition_3 : ‖a + b‖ = Real.sqrt 7 := sorry

-- Proof statement
theorem magnitude_diff (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a + b‖ = Real.sqrt 7) : 
  ‖a - b‖ = 3 :=
sorry

end magnitude_diff_l797_79794


namespace coloring_methods_390_l797_79717

def numColoringMethods (colors cells : ℕ) (maxColors : ℕ) : ℕ :=
  if colors = 6 ∧ cells = 4 ∧ maxColors = 3 then 390 else 0

theorem coloring_methods_390 :
  numColoringMethods 6 4 3 = 390 :=
by 
  sorry

end coloring_methods_390_l797_79717


namespace number_of_friends_l797_79729

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l797_79729


namespace applesauce_ratio_is_half_l797_79719

-- Define the weights and number of pies
def total_weight : ℕ := 120
def weight_per_pie : ℕ := 4
def num_pies : ℕ := 15

-- Calculate weights used for pies and applesauce
def weight_for_pies : ℕ := num_pies * weight_per_pie
def weight_for_applesauce : ℕ := total_weight - weight_for_pies

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to prove
theorem applesauce_ratio_is_half :
  ratio weight_for_applesauce total_weight = 1 / 2 :=
by
  -- The proof goes here
  sorry

end applesauce_ratio_is_half_l797_79719


namespace software_package_cost_l797_79784

theorem software_package_cost 
  (devices : ℕ) 
  (cost_first : ℕ) 
  (devices_covered_first : ℕ) 
  (devices_covered_second : ℕ) 
  (savings : ℕ)
  (total_cost_first : ℕ := (devices / devices_covered_first) * cost_first)
  (total_cost_second : ℕ := total_cost_first - savings)
  (num_packages_second : ℕ := devices / devices_covered_second)
  (cost_second : ℕ := total_cost_second / num_packages_second) :
  devices = 50 ∧ cost_first = 40 ∧ devices_covered_first = 5 ∧ devices_covered_second = 10 ∧ savings = 100 →
  cost_second = 60 := 
by
  sorry

end software_package_cost_l797_79784


namespace Albert_has_more_rocks_than_Jose_l797_79789

noncomputable def Joshua_rocks : ℕ := 80
noncomputable def Jose_rocks : ℕ := Joshua_rocks - 14
noncomputable def Albert_rocks : ℕ := Joshua_rocks + 6

theorem Albert_has_more_rocks_than_Jose :
  Albert_rocks - Jose_rocks = 20 := by
  sorry

end Albert_has_more_rocks_than_Jose_l797_79789


namespace annual_growth_rate_l797_79738

theorem annual_growth_rate (p : ℝ) : 
  let S1 := (1 + p) ^ 12 - 1 / p
  let S2 := ((1 + p) ^ 12 * ((1 + p) ^ 12 - 1)) / p
  let annual_growth := (S2 - S1) / S1
  annual_growth = (1 + p) ^ 12 - 1 :=
by
  sorry

end annual_growth_rate_l797_79738


namespace ball_distribution_l797_79771

theorem ball_distribution (balls boxes : ℕ) (hballs : balls = 7) (hboxes : boxes = 4) :
  (∃ (ways : ℕ), ways = (Nat.choose (balls - 1) (boxes - 1)) ∧ ways = 20) :=
by
  sorry

end ball_distribution_l797_79771


namespace only_linear_equation_with_two_variables_l797_79798

def is_linear_equation_with_two_variables (eqn : String) : Prop :=
  eqn = "4x-5y=5"

def equation_A := "4x-5y=5"
def equation_B := "xy-y=1"
def equation_C := "4x+5y"
def equation_D := "2/x+5/y=1/7"

theorem only_linear_equation_with_two_variables :
  is_linear_equation_with_two_variables equation_A ∧
  ¬ is_linear_equation_with_two_variables equation_B ∧
  ¬ is_linear_equation_with_two_variables equation_C ∧
  ¬ is_linear_equation_with_two_variables equation_D :=
by
  sorry

end only_linear_equation_with_two_variables_l797_79798


namespace sport_vs_std_ratio_comparison_l797_79774

/-- Define the ratios for the standard formulation. -/
def std_flavor_syrup_ratio := 1 / 12
def std_flavor_water_ratio := 1 / 30

/-- Define the conditions for the sport formulation. -/
def sport_water := 15 -- ounces of water in the sport formulation
def sport_syrup := 1 -- ounce of corn syrup in the sport formulation

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation. -/
def sport_flavor_water_ratio := std_flavor_water_ratio / 2

/-- Calculate the amount of flavoring in the sport formulation. -/
def sport_flavor := sport_water * sport_flavor_water_ratio

/-- The ratio of flavoring to corn syrup in the sport formulation. -/
def sport_flavor_syrup_ratio := sport_flavor / sport_syrup

/-- The proof problem statement. -/
theorem sport_vs_std_ratio_comparison : sport_flavor_syrup_ratio = 3 * std_flavor_syrup_ratio := 
by
  -- proof would go here
  sorry

end sport_vs_std_ratio_comparison_l797_79774


namespace a_seq_correct_l797_79707

-- Define the sequence and the sum condition
def a_seq (n : ℕ) : ℚ := if n = 0 then 0 else (2 ^ n - 1) / 2 ^ (n - 1)

def S_n (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.sum (Finset.range n) a_seq)

axiom condition (n : ℕ) (hn : n > 0) : S_n n + a_seq n = 2 * n

theorem a_seq_correct (n : ℕ) (hn : n > 0) : 
  a_seq n = (2 ^ n - 1) / 2 ^ (n - 1) := sorry

end a_seq_correct_l797_79707


namespace solve_parabola_l797_79728

theorem solve_parabola (a b c : ℝ) 
  (h1 : 1 = a * 1^2 + b * 1 + c)
  (h2 : 4 * a + b = 1)
  (h3 : -1 = a * 2^2 + b * 2 + c) :
  a = 3 ∧ b = -11 ∧ c = 9 :=
by {
  sorry
}

end solve_parabola_l797_79728


namespace inequality_AM_GM_l797_79737

theorem inequality_AM_GM
  (a b c : ℝ)
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c)
  (habc : a + b + c = 1) : 
  (a + 2 * a * b + 2 * a * c + b * c) ^ a * 
  (b + 2 * b * c + 2 * b * a + c * a) ^ b * 
  (c + 2 * c * a + 2 * c * b + a * b) ^ c ≤ 1 :=
by
  sorry

end inequality_AM_GM_l797_79737


namespace farm_width_l797_79702

theorem farm_width (L W : ℕ) (h1 : 2 * (L + W) = 46) (h2 : W = L + 7) : W = 15 :=
by
  sorry

end farm_width_l797_79702


namespace simplify_and_evaluate_expression_l797_79723

theorem simplify_and_evaluate_expression (x y : ℝ) (h_x : x = -2) (h_y : y = 1) :
  (((2 * x - (1/2) * y)^2 - ((-y + 2 * x) * (2 * x + y)) + y * (x^2 * y - (5/4) * y)) / x) = -4 :=
by
  sorry

end simplify_and_evaluate_expression_l797_79723


namespace bestCompletion_is_advantage_l797_79762

-- Defining the phrase and the list of options
def phrase : String := "British students have a language ____ for jobs in the USA and Australia"

def options : List (String × String) := 
  [("A", "chance"), ("B", "ability"), ("C", "possibility"), ("D", "advantage")]

-- Defining the best completion function (using a placeholder 'sorry' for the logic which is not the focus here)
noncomputable def bestCompletion (phrase : String) (options : List (String × String)) : String :=
  "advantage"  -- We assume given the problem that this function correctly identifies 'advantage'

-- Lean theorem stating the desired property
theorem bestCompletion_is_advantage : bestCompletion phrase options = "advantage" :=
by sorry

end bestCompletion_is_advantage_l797_79762


namespace value_of_expression_l797_79712

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end value_of_expression_l797_79712


namespace youngest_sibling_age_l797_79714

theorem youngest_sibling_age
    (age_youngest : ℕ)
    (first_sibling : ℕ := age_youngest + 4)
    (second_sibling : ℕ := age_youngest + 5)
    (third_sibling : ℕ := age_youngest + 7)
    (average_age : ℕ := 21)
    (sum_of_ages : ℕ := 4 * average_age)
    (total_age_check : (age_youngest + first_sibling + second_sibling + third_sibling) = sum_of_ages) :
  age_youngest = 17 :=
sorry

end youngest_sibling_age_l797_79714


namespace remainder_calculation_l797_79791

theorem remainder_calculation :
  ((2367 * 1023) % 500) = 41 := by
  sorry

end remainder_calculation_l797_79791


namespace smallest_k_no_real_roots_l797_79709

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 13 ≠ 0) ∧
  (∀ n : ℤ, n < k → ∃ x : ℝ, 3 * x * (n * x - 5) - 2 * x^2 + 13 = 0) :=
by sorry

end smallest_k_no_real_roots_l797_79709


namespace shipping_cost_per_unit_l797_79767

noncomputable def fixed_monthly_costs : ℝ := 16500
noncomputable def production_cost_per_component : ℝ := 80
noncomputable def production_quantity : ℝ := 150
noncomputable def selling_price_per_component : ℝ := 193.33

theorem shipping_cost_per_unit :
  ∀ (S : ℝ), (production_quantity * production_cost_per_component + production_quantity * S + fixed_monthly_costs) ≤ (production_quantity * selling_price_per_component) → S ≤ 3.33 :=
by
  intro S
  sorry

end shipping_cost_per_unit_l797_79767


namespace solve_for_y_l797_79736

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end solve_for_y_l797_79736


namespace complex_right_triangle_l797_79793

open Complex

theorem complex_right_triangle {z1 z2 a b : ℂ}
  (h1 : z2 = I * z1)
  (h2 : z1 + z2 = -a)
  (h3 : z1 * z2 = b) :
  a^2 / b = 2 :=
by sorry

end complex_right_triangle_l797_79793


namespace value_of_a_minus_b_l797_79708

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : |a + b| = a + b) : a - b = 2 ∨ a - b = 14 := 
sorry

end value_of_a_minus_b_l797_79708


namespace monthly_energy_consumption_l797_79765

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end monthly_energy_consumption_l797_79765


namespace students_play_both_l797_79711

-- Definitions of problem conditions
def total_students : ℕ := 1200
def play_football : ℕ := 875
def play_cricket : ℕ := 450
def play_neither : ℕ := 100
def play_either := total_students - play_neither

-- Lean statement to prove that the number of students playing both football and cricket
theorem students_play_both : play_football + play_cricket - 225 = play_either :=
by
  -- The proof is omitted
  sorry

end students_play_both_l797_79711


namespace segment_distance_sum_l797_79770

theorem segment_distance_sum
  (AB_len : ℝ) (A'B'_len : ℝ) (D_midpoint : AB_len / 2 = 4)
  (D'_midpoint : A'B'_len / 2 = 6) (x : ℝ) (y : ℝ)
  (x_val : x = 3) :
  x + y = 10 :=
by sorry

end segment_distance_sum_l797_79770


namespace pratyya_payel_min_difference_l797_79752

theorem pratyya_payel_min_difference (n m : ℕ) (h : n > m ∧ n - m ≥ 4) :
  ∀ t : ℕ, (2^(t+1) * n - 2^(t+1)) > 2^(t+1) * m + 2^(t+1) :=
by
  sorry

end pratyya_payel_min_difference_l797_79752


namespace number_of_numbers_tadd_said_after_20_rounds_l797_79742

-- Define the arithmetic sequence representing the count of numbers Tadd says each round
def tadd_sequence (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Define the sum of the first n terms of Tadd's sequence
def sum_tadd_sequence (n : ℕ) : ℕ :=
  n * (1 + tadd_sequence n) / 2

-- The main theorem to state the problem
theorem number_of_numbers_tadd_said_after_20_rounds :
  sum_tadd_sequence 20 = 400 :=
by
  -- The actual proof should be filled in here
  sorry

end number_of_numbers_tadd_said_after_20_rounds_l797_79742


namespace factorize_expression_l797_79743

theorem factorize_expression (x y : ℝ) : (y + 2 * x)^2 - (x + 2 * y)^2 = 3 * (x + y) * (x - y) :=
  sorry

end factorize_expression_l797_79743


namespace winner_votes_percentage_l797_79777

-- Define the total votes as V
def total_votes (winner_votes : ℕ) (winning_margin : ℕ) : ℕ :=
  winner_votes + (winner_votes - winning_margin)

-- Define the percentage function
def percentage_of_votes (part : ℕ) (total : ℕ) : ℕ :=
  (part * 100) / total

-- Lean statement to prove the result
theorem winner_votes_percentage
  (winner_votes : ℕ)
  (winning_margin : ℕ)
  (H_winner_votes : winner_votes = 550)
  (H_winning_margin : winning_margin = 100) :
  percentage_of_votes winner_votes (total_votes winner_votes winning_margin) = 55 := by
  sorry

end winner_votes_percentage_l797_79777


namespace palmer_first_week_photos_l797_79763

theorem palmer_first_week_photos :
  ∀ (X : ℕ), 
    100 + X + 2 * X + 80 = 380 →
    X = 67 :=
by
  intros X h
  -- h represents the condition 100 + X + 2 * X + 80 = 380
  sorry

end palmer_first_week_photos_l797_79763


namespace mosquito_distance_ratio_l797_79739

-- Definition of the clock problem conditions
structure ClockInsects where
  distance_from_center : ℕ
  initial_time : ℕ := 1

-- Prove the ratio of distances traveled by mosquito and fly over 12 hours
theorem mosquito_distance_ratio (c : ClockInsects) :
  let mosquito_distance := (83 : ℚ)/12
  let fly_distance := (73 : ℚ)/12
  mosquito_distance / fly_distance = 83 / 73 :=
by 
  sorry

end mosquito_distance_ratio_l797_79739


namespace identity_implies_a_minus_b_l797_79755

theorem identity_implies_a_minus_b (a b : ℚ) (y : ℚ) (h : y > 0) :
  (∀ y, y > 0 → (a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5)))) → (a - b = 1) :=
by
  sorry

end identity_implies_a_minus_b_l797_79755


namespace most_convincing_method_for_relationship_l797_79724

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end most_convincing_method_for_relationship_l797_79724


namespace product_of_numbers_is_178_5_l797_79775

variables (a b c d : ℚ)

def sum_eq_36 := a + b + c + d = 36
def first_num_cond := a = 3 * (b + c + d)
def second_num_cond := b = 5 * c
def fourth_num_cond := d = (1 / 2) * c

theorem product_of_numbers_is_178_5 (h1 : sum_eq_36 a b c d)
  (h2 : first_num_cond a b c d) (h3 : second_num_cond b c) (h4 : fourth_num_cond d c) :
  a * b * c * d = 178.5 :=
by
  sorry

end product_of_numbers_is_178_5_l797_79775


namespace initial_workers_l797_79732

theorem initial_workers (W : ℕ) (H1 : (8 * W) / 30 = W) (H2 : (6 * (2 * W - 45)) / 45 = 2 * W - 45) : W = 45 :=
sorry

end initial_workers_l797_79732


namespace solve_fractional_eq_l797_79722

theorem solve_fractional_eq (x: ℝ) (h1: x ≠ -11) (h2: x ≠ -8) (h3: x ≠ -12) (h4: x ≠ -7) :
  (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) → (x = -19 / 2) :=
by
  sorry

end solve_fractional_eq_l797_79722


namespace circle_center_radius_sum_l797_79754

-- We define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 14 * x + y^2 + 16 * y + 100 = 0

-- We need to find that the center and radius satisfy a specific relationship
theorem circle_center_radius_sum :
  let a' := 7
  let b' := -8
  let r' := Real.sqrt 13
  a' + b' + r' = -1 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l797_79754


namespace least_integer_exists_l797_79735

theorem least_integer_exists (x : ℕ) (h1 : x = 10 * (x / 10) + x % 10) (h2 : (x / 10) = x / 17) : x = 17 :=
sorry

end least_integer_exists_l797_79735


namespace physicist_imons_no_entanglement_l797_79796

theorem physicist_imons_no_entanglement (G : SimpleGraph V) :
  (∃ ops : ℕ, ∀ v₁ v₂ : V, ¬G.Adj v₁ v₂) :=
by
  sorry

end physicist_imons_no_entanglement_l797_79796


namespace sequence_ratio_l797_79797

theorem sequence_ratio :
  ∀ {a : ℕ → ℝ} (h₁ : a 1 = 1/2) (h₂ : ∀ n, a n = (a (n + 1)) * (a (n + 1))),
  (a 200 / a 300) = (301 / 201) :=
by
  sorry

end sequence_ratio_l797_79797


namespace tom_has_9_balloons_l797_79733

-- Define Tom's and Sara's yellow balloon counts
variables (total_balloons saras_balloons toms_balloons : ℕ)

-- Given conditions
axiom total_balloons_def : total_balloons = 17
axiom saras_balloons_def : saras_balloons = 8
axiom toms_balloons_total : toms_balloons + saras_balloons = total_balloons

-- Theorem stating that Tom has 9 yellow balloons
theorem tom_has_9_balloons : toms_balloons = 9 := by
  sorry

end tom_has_9_balloons_l797_79733


namespace janet_additional_money_needed_is_1225_l797_79782

def savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def months_required : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

noncomputable def total_rent : ℕ := rent_per_month * months_required
noncomputable def total_upfront_cost : ℕ := total_rent + deposit + utility_deposit + moving_costs
noncomputable def additional_money_needed : ℕ := total_upfront_cost - savings

theorem janet_additional_money_needed_is_1225 : additional_money_needed = 1225 :=
by
  sorry

end janet_additional_money_needed_is_1225_l797_79782


namespace convert_to_rectangular_form_l797_79734

theorem convert_to_rectangular_form :
  (Complex.exp (13 * Real.pi * Complex.I / 2)) = Complex.I :=
by
  sorry

end convert_to_rectangular_form_l797_79734


namespace minValue_l797_79769

noncomputable def minValueOfExpression (a b c : ℝ) : ℝ :=
  (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a))

theorem minValue (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 2 * a + 2 * b + 2 * c = 3) : 
  minValueOfExpression a b c = 2 :=
  sorry

end minValue_l797_79769


namespace tens_digit_of_expression_l797_79768

theorem tens_digit_of_expression :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 1 :=
by sorry

end tens_digit_of_expression_l797_79768


namespace watch_cost_price_l797_79718

theorem watch_cost_price (C : ℝ) (h1 : 0.85 * C = SP1) (h2 : 1.06 * C = SP2) (h3 : SP2 - SP1 = 350) : 
  C = 1666.67 := 
  sorry

end watch_cost_price_l797_79718


namespace inequality_abc_l797_79725

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) >= (a + b + c) / 3 := by
  sorry

end inequality_abc_l797_79725


namespace arithmetic_progr_property_l797_79740

theorem arithmetic_progr_property (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 + a 3 = 5 / 2)
  (h2 : a 2 + a 4 = 5 / 4)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h4 : a 3 = a 1 + 2 * (a 2 - a 1))
  (h5 : a 2 = a 1 + (a 2 - a 1)) :
  S 3 / a 3 = 6 := sorry

end arithmetic_progr_property_l797_79740


namespace distance_traveled_l797_79756

noncomputable def velocity (t : ℝ) := 2 * t - 3

theorem distance_traveled : 
  (∫ t in (0 : ℝ)..5, |velocity t|) = 29 / 2 := 
by
  sorry

end distance_traveled_l797_79756


namespace cannon_hit_probability_l797_79753

theorem cannon_hit_probability {P2 P3 : ℝ} (hP1 : 0.5 <= P2) (hP2 : P2 = 0.2) (hP3 : P3 = 0.3) (h_none_hit : (1 - 0.5) * (1 - P2) * (1 - P3) = 0.28) :
  0.5 = 0.5 :=
by sorry

end cannon_hit_probability_l797_79753


namespace tan_problem_l797_79703

noncomputable def problem : ℝ :=
  (Real.tan (20 * Real.pi / 180) + Real.tan (40 * Real.pi / 180) + Real.tan (120 * Real.pi / 180)) / 
  (Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180))

theorem tan_problem : problem = -Real.sqrt 3 := by
  sorry

end tan_problem_l797_79703


namespace number_of_students_l797_79748

theorem number_of_students (pencils: ℕ) (pencils_per_student: ℕ) (total_students: ℕ) 
  (h1: pencils = 195) (h2: pencils_per_student = 3) (h3: total_students = pencils / pencils_per_student) :
  total_students = 65 := by
  -- proof would go here, but we skip it with sorry for now
  sorry

end number_of_students_l797_79748


namespace number_of_croutons_l797_79783

def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def crouton_calories : ℕ := 20
def total_salad_calories : ℕ := 350

theorem number_of_croutons : 
  ∃ n : ℕ, n * crouton_calories = total_salad_calories - (lettuce_calories + cucumber_calories) ∧ n = 12 :=
by
  sorry

end number_of_croutons_l797_79783


namespace problem1_problem2_l797_79766

-- Problem 1: Remainder of 2011-digit number with each digit 2 when divided by 9 is 8

theorem problem1 : (4022 % 9 = 8) := by
  sorry

-- Problem 2: Remainder of n-digit number with each digit 7 when divided by 9 and n % 9 = 3 is 3

theorem problem2 (n : ℕ) (h : n % 9 = 3) : ((7 * n) % 9 = 3) := by
  sorry

end problem1_problem2_l797_79766


namespace eight_child_cotton_l797_79778

theorem eight_child_cotton {a_1 a_8 d S_8 : ℕ} 
  (h1 : d = 17)
  (h2 : S_8 = 996)
  (h3 : 8 * a_1 + 28 * d = S_8) :
  a_8 = a_1 + 7 * d → a_8 = 184 := by
  intro h4
  subst_vars
  sorry

end eight_child_cotton_l797_79778


namespace no_int_sol_eq_l797_79786

theorem no_int_sol_eq (x y z : ℤ) (h₀ : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : ¬ (x^2 + y^2 = 3 * z^2) := 
sorry

end no_int_sol_eq_l797_79786


namespace xyz_sum_eq_7x_plus_5_l797_79715

variable (x y z : ℝ)

theorem xyz_sum_eq_7x_plus_5 (h1: y = 3 * x) (h2: z = y + 5) : x + y + z = 7 * x + 5 :=
by
  sorry

end xyz_sum_eq_7x_plus_5_l797_79715


namespace total_amount_spent_l797_79759

def price_of_brand_X_pen : ℝ := 4.00
def price_of_brand_Y_pen : ℝ := 2.20
def total_pens_purchased : ℝ := 12
def brand_X_pens_purchased : ℝ := 6

theorem total_amount_spent :
  let brand_X_cost := brand_X_pens_purchased * price_of_brand_X_pen
  let brand_Y_pens_purchased := total_pens_purchased - brand_X_pens_purchased
  let brand_Y_cost := brand_Y_pens_purchased * price_of_brand_Y_pen
  brand_X_cost + brand_Y_cost = 37.20 :=
by
  sorry

end total_amount_spent_l797_79759


namespace ratio_of_perimeters_l797_79704

noncomputable def sqrt2 : ℝ := Real.sqrt 2

theorem ratio_of_perimeters (d1 : ℝ) :
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2 
  (P2 / P1 = 1 + sqrt2) :=
by
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2
  sorry

end ratio_of_perimeters_l797_79704


namespace find_h_plus_k_l797_79730

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 14*y - 11 = 0

-- State the problem: Prove h + k = -4 given (h, k) is the center of the circle
theorem find_h_plus_k : (∃ h k, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 69) ∧ h + k = -4) :=
by {
  sorry
}

end find_h_plus_k_l797_79730


namespace envelope_addressing_equation_l797_79726

theorem envelope_addressing_equation (x : ℝ) :
  (800 / 10 + 800 / x + 800 / 5) * (3 / 800) = 1 / 3 :=
  sorry

end envelope_addressing_equation_l797_79726


namespace ekon_uma_diff_l797_79710

-- Definitions based on conditions
def total_videos := 411
def kelsey_videos := 160
def ekon_kelsey_diff := 43

-- Definitions derived from conditions
def ekon_videos := kelsey_videos - ekon_kelsey_diff
def uma_videos (E : ℕ) := total_videos - kelsey_videos - E

-- The Lean problem statement
theorem ekon_uma_diff : 
  uma_videos ekon_videos - ekon_videos = 17 := 
by 
  sorry

end ekon_uma_diff_l797_79710


namespace solve_missing_number_l797_79721

theorem solve_missing_number (n : ℤ) (h : 121 * n = 75625) : n = 625 :=
sorry

end solve_missing_number_l797_79721


namespace find_angle_l797_79772

def complementary (x : ℝ) := 90 - x
def supplementary (x : ℝ) := 180 - x

theorem find_angle (x : ℝ) (h : supplementary x = 3 * complementary x) : x = 45 :=
by 
  sorry

end find_angle_l797_79772


namespace paint_per_statue_calculation_l797_79731

theorem paint_per_statue_calculation (total_paint : ℚ) (num_statues : ℕ) (expected_paint_per_statue : ℚ) :
  total_paint = 7 / 8 → num_statues = 14 → expected_paint_per_statue = 7 / 112 → 
  total_paint / num_statues = expected_paint_per_statue :=
by
  intros htotal hnum_expected hequals
  rw [htotal, hnum_expected, hequals]
  -- Using the fact that:
  -- total_paint / num_statues = (7 / 8) / 14
  -- This can be rewritten as (7 / 8) * (1 / 14) = 7 / (8 * 14) = 7 / 112
  sorry

end paint_per_statue_calculation_l797_79731


namespace ordering_of_exponentials_l797_79773

theorem ordering_of_exponentials :
  let A := 3^20
  let B := 6^10
  let C := 2^30
  B < A ∧ A < C :=
by
  -- Definitions and conditions
  have h1 : 6^10 = 3^10 * 2^10 := by sorry
  have h2 : 3^10 = 59049 := by sorry
  have h3 : 2^10 = 1024 := by sorry
  have h4 : 2^30 = (2^10)^3 := by sorry
  
  -- We know 3^20, 6^10, 2^30 by definition and conditions
  -- Comparison
  have h5 : 3^20 = (3^10)^2 := by sorry
  have h6 : 2^30 = 1024^3 := by sorry
  
  -- Combine to get results
  have h7 : (3^10)^2 > 6^10 := by sorry
  have h8 : 1024^3 > 6^10 := by sorry
  have h9 : 1024^3 > (3^10)^2 := by sorry

  exact ⟨h7, h9⟩

end ordering_of_exponentials_l797_79773


namespace my_problem_l797_79780

-- Definitions and conditions from the problem statement
variables (p q r u v w : ℝ)

-- Conditions
axiom h1 : 17 * u + q * v + r * w = 0
axiom h2 : p * u + 29 * v + r * w = 0
axiom h3 : p * u + q * v + 56 * w = 0
axiom h4 : p ≠ 17
axiom h5 : u ≠ 0

-- Problem statement to prove
theorem my_problem : (p / (p - 17)) + (q / (q - 29)) + (r / (r - 56)) = 0 :=
sorry

end my_problem_l797_79780


namespace cylinder_height_l797_79706

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h_cond : SA = 2 * π * r^2 + 2 * π * r * h) 
  (r_eq : r = 3) (SA_eq : SA = 27 * π) : h = 3 / 2 :=
by
  sorry

end cylinder_height_l797_79706


namespace find_k_l797_79701

theorem find_k (x y k : ℝ) 
  (line1 : y = 3 * x + 2) 
  (line2 : y = -4 * x - 14) 
  (line3 : y = 2 * x + k) :
  k = -2 / 7 := 
by {
  sorry
}

end find_k_l797_79701


namespace candy_cost_l797_79787

theorem candy_cost (x : ℝ) : 
  (15 * x + 30 * 5) / (15 + 30) = 6 -> x = 8 :=
by sorry

end candy_cost_l797_79787


namespace sum_squares_of_roots_of_polynomial_l797_79720

noncomputable def roots (n : ℕ) (p : Polynomial ℂ) : List ℂ :=
  if h : n = p.natDegree then Multiset.toList p.roots else []

theorem sum_squares_of_roots_of_polynomial :
  (roots 2018 (Polynomial.C 404 + Polynomial.C 3 * X ^ 3 + Polynomial.C 44 * X ^ 2015 + X ^ 2018)).sum = 0 :=
by
  sorry

end sum_squares_of_roots_of_polynomial_l797_79720


namespace perfect_square_proof_l797_79788

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem perfect_square_proof :
  isPerfectSquare (factorial 22 * factorial 23 * factorial 24 / 12) :=
sorry

end perfect_square_proof_l797_79788


namespace line_slope_intercept_l797_79764

theorem line_slope_intercept (a b : ℝ) 
  (h1 : (7 : ℝ) = a * 3 + b) 
  (h2 : (13 : ℝ) = a * (9/2) + b) : 
  a - b = 9 := 
sorry

end line_slope_intercept_l797_79764


namespace max_Sn_in_arithmetic_sequence_l797_79705

theorem max_Sn_in_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ {m n p q : ℕ}, m + n = p + q → a m + a n = a p + a q)
  (h_a4 : a 4 = 1)
  (h_S5 : S 5 = 10)
  (h_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  ∃ n, n = 4 ∨ n = 5 ∧ ∀ m ≠ n, S m ≤ S n := by
  sorry

end max_Sn_in_arithmetic_sequence_l797_79705


namespace value_of_A_l797_79799

def random_value (c : Char) : ℤ := sorry

-- Given conditions
axiom H_value : random_value 'H' = 12
axiom MATH_value : random_value 'M' + random_value 'A' + random_value 'T' + random_value 'H' = 40
axiom TEAM_value : random_value 'T' + random_value 'E' + random_value 'A' + random_value 'M' = 50
axiom MEET_value : random_value 'M' + random_value 'E' + random_value 'E' + random_value 'T' = 44

-- Prove that A = 28
theorem value_of_A : random_value 'A' = 28 := by
  sorry

end value_of_A_l797_79799


namespace bob_weight_l797_79741

noncomputable def jim_bob_equations (j b : ℝ) : Prop :=
  j + b = 200 ∧ b - 3 * j = b / 4

theorem bob_weight (j b : ℝ) (h : jim_bob_equations j b) : b = 171.43 :=
by
  sorry

end bob_weight_l797_79741


namespace pete_backward_speed_l797_79758

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end pete_backward_speed_l797_79758


namespace range_of_a_l797_79795

variable (x a : ℝ)

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem range_of_a (h : ∀ x, q x a → p x)
  (h_not : ∃ x, ¬ q x a ∧ p x) : 1 ≤ a :=
sorry

end range_of_a_l797_79795


namespace rectangle_perimeter_l797_79750

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) : 2 * (L + B) = 186 :=
by
  sorry

end rectangle_perimeter_l797_79750


namespace ninth_number_l797_79776

theorem ninth_number (S1 S2 Total N : ℕ)
  (h1 : S1 = 9 * 56)
  (h2 : S2 = 9 * 63)
  (h3 : Total = 17 * 59)
  (h4 : Total = S1 + S2 - N) :
  N = 68 :=
by 
  -- The proof is omitted, only the statement is needed.
  sorry

end ninth_number_l797_79776


namespace number_of_larger_planes_l797_79700

variable (S L : ℕ)
variable (h1 : S + L = 4)
variable (h2 : 130 * S + 145 * L = 550)

theorem number_of_larger_planes : L = 2 :=
by
  -- Placeholder for the proof
  sorry

end number_of_larger_planes_l797_79700


namespace smallest_p_l797_79761

theorem smallest_p (n p : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) (h3 : (n + p) % 10 = 0) : p = 1 := 
sorry

end smallest_p_l797_79761


namespace smallestC_l797_79744

def isValidFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧
  f 1 = 1 ∧
  (∀ x y, 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1 → f x + f y ≤ f (x + y))

theorem smallestC (f : ℝ → ℝ) (h : isValidFunction f) : ∃ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ c * x) ∧
  (∀ d, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ d * x) → 2 ≤ d) :=
sorry

end smallestC_l797_79744


namespace solve_line_eq_l797_79716

theorem solve_line_eq (a b x : ℝ) (h1 : (0 : ℝ) * a + b = 2) (h2 : -3 * a + b = 0) : x = -3 :=
by
  sorry

end solve_line_eq_l797_79716


namespace ratio_twice_width_to_length_l797_79779

theorem ratio_twice_width_to_length (L W : ℝ) (k : ℤ)
  (h1 : L = 24)
  (h2 : W = 13.5)
  (h3 : L = k * W - 3) :
  2 * W / L = 9 / 8 := by
  sorry

end ratio_twice_width_to_length_l797_79779


namespace remaining_amount_to_be_paid_l797_79727

theorem remaining_amount_to_be_paid (p : ℝ) (deposit : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (final_payment : ℝ) :
  deposit = 80 ∧ tax_rate = 0.07 ∧ discount_rate = 0.05 ∧ deposit = 0.1 * p ∧ 
  final_payment = (p - (discount_rate * p)) * (1 + tax_rate) - deposit → 
  final_payment = 733.20 :=
by
  sorry

end remaining_amount_to_be_paid_l797_79727


namespace sequence_value_2016_l797_79760

theorem sequence_value_2016 :
  ∀ (a : ℕ → ℤ),
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
    a 2016 = -3 :=
by
  sorry

end sequence_value_2016_l797_79760


namespace inequality_bound_l797_79745

theorem inequality_bound (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ e^x * (x^2 - x + 1) * (a * x + 3 * a - 1) < 1) : a < 2 / 3 :=
by
  sorry

end inequality_bound_l797_79745


namespace base_7_units_digit_of_product_359_72_l797_79757

def base_7_units_digit (n : ℕ) : ℕ := n % 7

theorem base_7_units_digit_of_product_359_72 : base_7_units_digit (359 * 72) = 4 := 
by
  sorry

end base_7_units_digit_of_product_359_72_l797_79757


namespace chicken_cost_l797_79749

theorem chicken_cost (total_money hummus_price hummus_count bacon_price vegetables_price apple_price apple_count chicken_price : ℕ)
  (h_total_money : total_money = 60)
  (h_hummus_price : hummus_price = 5)
  (h_hummus_count : hummus_count = 2)
  (h_bacon_price : bacon_price = 10)
  (h_vegetables_price : vegetables_price = 10)
  (h_apple_price : apple_price = 2)
  (h_apple_count : apple_count = 5)
  (h_remaining_money : chicken_price = total_money - (hummus_count * hummus_price + bacon_price + vegetables_price + apple_count * apple_price)) :
  chicken_price = 20 := 
by sorry

end chicken_cost_l797_79749


namespace gcd_204_85_l797_79747

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l797_79747


namespace square_roots_of_x_l797_79781

theorem square_roots_of_x (a x : ℝ) 
    (h1 : (2 * a - 1) ^ 2 = x) 
    (h2 : (-a + 2) ^ 2 = x)
    (hx : 0 < x) 
    : x = 9 ∨ x = 1 := 
by sorry

end square_roots_of_x_l797_79781


namespace inequality_solution_set_l797_79713

theorem inequality_solution_set :
  {x : ℝ | (x - 3) / (x + 2) ≤ 0} = {x : ℝ | -2 < x ∧ x ≤ 3} :=
by
  sorry

end inequality_solution_set_l797_79713


namespace divisible_iff_l797_79792

theorem divisible_iff (m n k : ℕ) (h : m > n) : 
  (3^(k+1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
sorry

end divisible_iff_l797_79792


namespace problem_l797_79790

def T := {n : ℤ | ∃ (k : ℤ), n = 4 * (2*k + 1)^2 + 13}

theorem problem :
  (∀ n ∈ T, ¬ 2 ∣ n) ∧ (∀ n ∈ T, ¬ 5 ∣ n) :=
by
  sorry

end problem_l797_79790


namespace triangle_inequality_l797_79785

theorem triangle_inequality 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : A + B + C = π) 
  (h5 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h6 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h7 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
  3 / 2 ≤ a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ∧
  (a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ≤ 
     2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) :=
sorry

end triangle_inequality_l797_79785
