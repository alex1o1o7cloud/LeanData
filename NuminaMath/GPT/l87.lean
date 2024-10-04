import Mathlib

namespace factorize_expression_l87_87967

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 :=
by
  sorry

end factorize_expression_l87_87967


namespace find_expression_value_l87_87201

theorem find_expression_value (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := 
by 
  sorry

end find_expression_value_l87_87201


namespace symmetric_point_xOz_l87_87886

def symmetric_point (p : (ℝ × ℝ × ℝ)) (plane : ℝ → Prop) : (ℝ × ℝ × ℝ) :=
match p with
| (x, y, z) => (x, -y, z)

theorem symmetric_point_xOz (x y z : ℝ) : symmetric_point (-1, 2, 1) (λ y, y = 0) = (-1, -2, 1) :=
by
  sorry

end symmetric_point_xOz_l87_87886


namespace largest_integer_x_l87_87450

theorem largest_integer_x (x : ℤ) (h : 3 - 5 * x > 22) : x ≤ -4 :=
by
  sorry

end largest_integer_x_l87_87450


namespace solve_fractional_equation_l87_87758

theorem solve_fractional_equation (x : ℚ) (h: x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 2) ↔ (x = 7 / 6) := 
by
  sorry

end solve_fractional_equation_l87_87758


namespace jack_correct_percentage_l87_87575

theorem jack_correct_percentage (y : ℝ) (h : y ≠ 0) :
  ((8 * y - (2 * y - 3)) / (8 * y)) * 100 = 75 + (75 / (2 * y)) :=
by
  sorry

end jack_correct_percentage_l87_87575


namespace solve_y_l87_87792

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end solve_y_l87_87792


namespace heads_at_least_twice_in_5_tosses_l87_87474

noncomputable def probability_at_least_two_heads (n : ℕ) (p : ℚ) : ℚ :=
1 - (n : ℚ) * p^(n : ℕ)

theorem heads_at_least_twice_in_5_tosses :
  probability_at_least_two_heads 5 (1/2) = 13/16 :=
by
  sorry

end heads_at_least_twice_in_5_tosses_l87_87474


namespace exactly_one_divisible_by_4_l87_87596

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end exactly_one_divisible_by_4_l87_87596


namespace checkered_rectangles_unique_gray_cells_l87_87066

noncomputable def num_checkered_rectangles (num_gray_cells : ℕ) (num_blue_cells : ℕ) (rects_per_blue_cell : ℕ)
    (num_red_cells : ℕ) (rects_per_red_cell : ℕ) : ℕ :=
    (num_blue_cells * rects_per_blue_cell) + (num_red_cells * rects_per_red_cell)

theorem checkered_rectangles_unique_gray_cells : num_checkered_rectangles 40 36 4 4 8 = 176 := 
sorry

end checkered_rectangles_unique_gray_cells_l87_87066


namespace no_integer_solutions_3a2_eq_b2_plus_1_l87_87524

theorem no_integer_solutions_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
by
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l87_87524


namespace problem1_problem2_problem3_problem4_l87_87250

open scoped BigOperators

section problem

variables (M F : ℕ) -- Number of male and female athletes
variables (cM cF : ℕ) -- Designating the male and female captains

noncomputable def waysToSelectTeam_3M_2F (M F : ℕ) : ℕ :=
  Nat.choose M 3 * Nat.choose F 2

theorem problem1 (hM : M = 6) (hF : F = 4) : waysToSelectTeam_3M_2F M F = 120 :=
by
  rw [hM, hF]
  norm_num

noncomputable def waysToSelectTeam_with_at_least_1F (M F : ℕ) : ℕ :=
  Nat.choose 10 5 - Nat.choose M 5

theorem problem2 (H : M = 6) : waysToSelectTeam_with_at_least_1F M F = 246 :=
by
  rw [H]
  norm_num

noncomputable def waysToSelectTeam_with_at_least_1C (M F : ℕ) (cM cF : ℕ) : ℕ :=
  Nat.choose 8 4 + Nat.choose 8 4 + Nat.choose 8 3

theorem problem3 (hM : M = 6) (hF : F = 4) (hcM : cM = 1) (hcF : cF = 1) : waysToSelectTeam_with_at_least_1C M F cM cF = 196 :=
by
  rw [hM, hF, hcM, hcF]
  norm_num

noncomputable def waysToSelectTeam_with_1C_and_1F (M F : ℕ) (cM cF : ℕ) : ℕ :=
  Nat.choose 9 4 + (Nat.choose 8 4 - Nat.choose 5 4)

theorem problem4 (hM : M = 6) (hF : F = 4) (hcM : cM = 1) (hcF : cF = 1) : waysToSelectTeam_with_1C_and_1F M F cM cF = 191 :=
by
  rw [hM, hF, hcM, hcF]
  norm_num

end problem

end problem1_problem2_problem3_problem4_l87_87250


namespace lcm_12_18_is_36_l87_87528

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l87_87528


namespace problem1_seating_arrangement_problem2_standing_arrangement_problem3_spots_distribution_l87_87479

theorem problem1_seating_arrangement :
  let n_seats := 8
  let n_people := 3
  seating_ways (n_seats : ℕ) (n_people : ℕ) : ℕ :=
sorry

theorem problem2_standing_arrangement :
  let n_people := 5
  arrangement_ways (n_people : ℕ) (A_right_of_B : Prop) : ℕ :=
sorry

theorem problem3_spots_distribution :
  let n_spots := 10
  let n_schools := 7
  let min_spot_per_school := 1
  distribution_ways (n_spots : ℕ) (n_schools : ℕ) (min_spot_per_school : ℕ) : ℕ :=
sorry

end problem1_seating_arrangement_problem2_standing_arrangement_problem3_spots_distribution_l87_87479


namespace no_solution_for_A_to_make_47A8_div_by_5_l87_87142

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem no_solution_for_A_to_make_47A8_div_by_5 (A : ℕ) :
  ¬ (divisible_by_5 (47 * 1000 + A * 100 + 8)) :=
by
  sorry

end no_solution_for_A_to_make_47A8_div_by_5_l87_87142


namespace percentage_of_knives_l87_87170

def initial_knives : Nat := 6
def initial_forks : Nat := 12
def initial_spoons : Nat := 3 * initial_knives
def traded_knives : Nat := 10
def traded_spoons : Nat := 6

theorem percentage_of_knives :
  100 * (initial_knives + traded_knives) / (initial_knives + initial_forks + initial_spoons - traded_spoons + traded_knives) = 40 := by
  sorry

end percentage_of_knives_l87_87170


namespace value_of_gg_neg1_l87_87995

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_gg_neg1 : g (g (-1)) = 199 := by
  sorry

end value_of_gg_neg1_l87_87995


namespace max_value_of_fraction_l87_87122

theorem max_value_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 1) (h_discriminant : a^2 = 4 * (b - 1)) :
  a = 2 → b = 2 → (3 * a + 2 * b) / (a + b) = 5 / 2 :=
by
  intro ha_eq
  intro hb_eq
  sorry

end max_value_of_fraction_l87_87122


namespace class_2_3_tree_count_total_tree_count_l87_87352

-- Definitions based on the given conditions
def class_2_5_trees := 142
def class_2_3_trees := class_2_5_trees - 18

-- Statements to be proved
theorem class_2_3_tree_count :
  class_2_3_trees = 124 :=
sorry

theorem total_tree_count :
  class_2_5_trees + class_2_3_trees = 266 :=
sorry

end class_2_3_tree_count_total_tree_count_l87_87352


namespace cannot_cut_square_into_7_rectangles_l87_87750

theorem cannot_cut_square_into_7_rectangles (a : ℝ) :
  ¬ ∃ (x : ℝ), 7 * 2 * x ^ 2 = a ^ 2 ∧ 
    ∀ (i : ℕ), 0 ≤ i → i < 7 → (∃ (rect : ℝ × ℝ), rect.1 = x ∧ rect.2 = 2 * x ) :=
by
  sorry

end cannot_cut_square_into_7_rectangles_l87_87750


namespace dice_probability_l87_87153

-- Definitions for the problem setup
def num_20_sided_dice : ℕ := 6
def num_one_digit : ℕ := 9
def num_two_digit : ℕ := 11
def p_one_digit : ℚ := 9 / 20
def p_two_digit : ℚ := 11 / 20

-- Binomial coefficient for choosing 3 out of 6 dice
noncomputable def binom_6_3 : ℕ := combinatorics.binom 6 3

-- Calculation for the probability
noncomputable def probability : ℚ := binom_6_3 * (p_one_digit^3) * (p_two_digit^3)

-- The theorem to prove
theorem dice_probability:
  probability = 969969 / 32000000 := 
sorry

end dice_probability_l87_87153


namespace oranges_per_child_l87_87520

theorem oranges_per_child (children oranges : ℕ) (h1 : children = 4) (h2 : oranges = 12) : oranges / children = 3 := by
  sorry

end oranges_per_child_l87_87520


namespace total_scoops_needed_l87_87904

def cups_of_flour : ℕ := 4
def cups_of_sugar : ℕ := 3
def cups_of_milk : ℕ := 2

def flour_scoop_size : ℚ := 1 / 4
def sugar_scoop_size : ℚ := 1 / 3
def milk_scoop_size : ℚ := 1 / 2

theorem total_scoops_needed : 
  (cups_of_flour / flour_scoop_size) + (cups_of_sugar / sugar_scoop_size) + (cups_of_milk / milk_scoop_size) = 29 := 
by {
  sorry
}

end total_scoops_needed_l87_87904


namespace units_digit_47_power_47_l87_87456

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l87_87456


namespace common_pasture_area_l87_87477

variable (Area_Ivanov Area_Petrov Area_Sidorov Area_Vasilev Area_Ermolaev : ℝ)
variable (Common_Pasture : ℝ)

theorem common_pasture_area :
  Area_Ivanov = 24 ∧
  Area_Petrov = 28 ∧
  Area_Sidorov = 10 ∧
  Area_Vasilev = 20 ∧
  Area_Ermolaev = 30 →
  Common_Pasture = 17.5 :=
sorry

end common_pasture_area_l87_87477


namespace investment_in_business_l87_87804

theorem investment_in_business (Q : ℕ) (P : ℕ) 
  (h1 : Q = 65000) 
  (h2 : 4 * Q = 5 * P) : 
  P = 52000 :=
by
  rw [h1] at h2
  linarith

end investment_in_business_l87_87804


namespace find_x_l87_87642

theorem find_x (x : ℤ) (h : 4 * x - 23 = 33) : x = 14 := 
by 
  sorry

end find_x_l87_87642


namespace one_fourth_of_six_point_three_as_fraction_l87_87043

noncomputable def one_fourth_of_six_point_three_is_simplified : ℚ :=
  6.3 / 4

theorem one_fourth_of_six_point_three_as_fraction :
  one_fourth_of_six_point_three_is_simplified = 63 / 40 :=
by
  sorry

end one_fourth_of_six_point_three_as_fraction_l87_87043


namespace find_base_l87_87775

-- Definitions based on the conditions of the problem
def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) := ∃ m : ℕ, m * m * m = n
def is_perfect_fourth (n : ℕ) := ∃ m : ℕ, m * m * m * m = n

-- Define the number A in terms of base a
def A (a : ℕ) : ℕ := 4 * a * a + 4 * a + 1

-- Problem statement: find a base a > 4 such that A is both a perfect cube and a perfect fourth power
theorem find_base (a : ℕ)
  (ha : a > 4)
  (h_square : is_perfect_square (A a)) :
  is_perfect_cube (A a) ∧ is_perfect_fourth (A a) :=
sorry

end find_base_l87_87775


namespace total_ways_is_13_l87_87137

-- Define the problem conditions
def num_bus_services : ℕ := 8
def num_train_services : ℕ := 3
def num_ferry_services : ℕ := 2

-- Define the total number of ways a person can travel from A to B
def total_ways : ℕ := num_bus_services + num_train_services + num_ferry_services

-- State the theorem that the total number of ways is 13
theorem total_ways_is_13 : total_ways = 13 :=
by
  -- Add a sorry placeholder for the proof
  sorry

end total_ways_is_13_l87_87137


namespace necessary_condition_lg_l87_87857

theorem necessary_condition_lg (x : ℝ) : ¬(x > -1) → ¬(10^1 > x + 1) := by {
    sorry
}

end necessary_condition_lg_l87_87857


namespace value_of_M_l87_87133

theorem value_of_M (x y z M : ℚ) : 
  (x + y + z = 48) ∧ (x - 5 = M) ∧ (y + 9 = M) ∧ (z / 5 = M) → M = 52 / 7 :=
by
  sorry

end value_of_M_l87_87133


namespace solve_for_x_l87_87611

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l87_87611


namespace solve_inequality_l87_87912

theorem solve_inequality (a : ℝ) : 
  (a = 0 → {x : ℝ | x ≥ -1} = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
  (a ≠ 0 → 
    ((a > 0 → { x : ℝ | -1 ≤ x ∧ x ≤ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (-2 < a ∧ a < 0 → { x : ℝ | x ≤ 2 / a } ∪ { x : ℝ | -1 ≤ x }  = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a < -2 → { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a = -2 → { x : ℝ | True } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 })
)) :=
sorry

end solve_inequality_l87_87912


namespace Morse_code_number_of_distinct_symbols_l87_87207

def count_sequences (n : ℕ) : ℕ :=
  2 ^ n

theorem Morse_code_number_of_distinct_symbols :
  (count_sequences 1) + (count_sequences 2) + (count_sequences 3) + (count_sequences 4) + (count_sequences 5) = 62 :=
by
  simp [count_sequences]
  norm_num
  sorry

end Morse_code_number_of_distinct_symbols_l87_87207


namespace Jessie_final_weight_l87_87398

variable (initial_weight : ℝ) (loss_first_week : ℝ) (loss_rate_second_week : ℝ)
variable (loss_second_week : ℝ) (total_loss : ℝ) (final_weight : ℝ)

def Jessie_weight_loss_problem : Prop :=
  initial_weight = 92 ∧
  loss_first_week = 5 ∧
  loss_rate_second_week = 1.3 ∧
  loss_second_week = loss_rate_second_week * loss_first_week ∧
  total_loss = loss_first_week + loss_second_week ∧
  final_weight = initial_weight - total_loss ∧
  final_weight = 80.5

theorem Jessie_final_weight : Jessie_weight_loss_problem initial_weight loss_first_week loss_rate_second_week loss_second_week total_loss final_weight :=
by
  sorry

end Jessie_final_weight_l87_87398


namespace ratio_female_to_male_l87_87945

variable (m f : ℕ)

-- Average ages given in the conditions
def avg_female_age : ℕ := 35
def avg_male_age : ℕ := 45
def avg_total_age : ℕ := 40

-- Total ages based on number of members
def total_female_age (f : ℕ) : ℕ := avg_female_age * f
def total_male_age (m : ℕ) : ℕ := avg_male_age * m
def total_age (f m : ℕ) : ℕ := total_female_age f + total_male_age m

-- Equation based on average age of all members
def avg_age_eq (f m : ℕ) : Prop :=
  total_age f m / (f + m) = avg_total_age

theorem ratio_female_to_male : avg_age_eq f m → f = m :=
by
  sorry

end ratio_female_to_male_l87_87945


namespace range_of_function_l87_87145

theorem range_of_function : 
  (∀ x, (Real.pi / 4) ≤ x ∧ x ≤ (Real.pi / 2) → 
   1 ≤ (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ∧ 
    (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ≤ 3 / 2) :=
sorry

end range_of_function_l87_87145


namespace find_digit_l87_87766

theorem find_digit (A : ℕ) (hA : A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  9 ∣ (3 * 1000 + A * 110 + 1) ↔ A = 7 :=
by
  sorry

end find_digit_l87_87766


namespace theta_in_fourth_quadrant_l87_87381

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (∃ k : ℤ, θ = 2 * π * k + 7 * π / 4 ∨ θ = 2 * π * k + π / 4) ∧ θ = 2 * π * k + 7 * π / 4 :=
sorry

end theta_in_fourth_quadrant_l87_87381


namespace both_firms_participate_l87_87213

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end both_firms_participate_l87_87213


namespace lcm_12_18_l87_87530

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l87_87530


namespace total_cards_1750_l87_87650

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end total_cards_1750_l87_87650


namespace function_graph_second_quadrant_l87_87571

theorem function_graph_second_quadrant (b : ℝ) (h : ∀ x, 2 ^ x + b - 1 ≥ 0): b ≤ 0 :=
sorry

end function_graph_second_quadrant_l87_87571


namespace fourth_sphere_radius_l87_87360

theorem fourth_sphere_radius (R r : ℝ) (h1 : R > 0)
  (h2 : ∀ (a b c d : ℝ × ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    dist a b = 2*R ∧ dist b c = 2*R ∧ dist c d = 2*R ∧ dist d a = R + r ∧
    dist a c = R + r ∧ dist b d = R + r) :
  r = 4*R/3 :=
  sorry

end fourth_sphere_radius_l87_87360


namespace ratio_of_height_to_width_l87_87128

-- Define variables
variable (W H L V : ℕ)
variable (x : ℝ)

-- Given conditions
def condition_1 := W = 3
def condition_2 := H = x * W
def condition_3 := L = 7 * H
def condition_4 := V = 6804

-- Prove that the ratio of height to width is 6√3
theorem ratio_of_height_to_width : (W = 3 ∧ H = x * W ∧ L = 7 * H ∧ V = 6804 ∧ V = W * H * L) → x = 6 * Real.sqrt 3 :=
by
  sorry

end ratio_of_height_to_width_l87_87128


namespace infinite_geometric_series_sum_l87_87949

theorem infinite_geometric_series_sum : 
  ∑' n : ℕ, (1 / 3) ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l87_87949


namespace range_of_m_l87_87856

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0)
  (h_equation : (2 / x) + (1 / y) = 1 / 3)
  (h_inequality : x + 2 * y > m^2 - 2 * m) : 
  -4 < m ∧ m < 6 := 
sorry

end range_of_m_l87_87856


namespace president_vice_secretary_choice_l87_87880

theorem president_vice_secretary_choice (n : ℕ) (h : n = 6) :
  (∀ a b c : fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (n * (n - 1) * (n - 2) = 120) := 
sorry

end president_vice_secretary_choice_l87_87880


namespace greatest_product_sum_2000_eq_1000000_l87_87291

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87291


namespace ratio_of_games_played_to_losses_l87_87800

-- Definitions based on the conditions
def total_games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := total_games_played - games_won

-- The proof problem
theorem ratio_of_games_played_to_losses : (total_games_played / Nat.gcd total_games_played games_lost) = 2 ∧ (games_lost / Nat.gcd total_games_played games_lost) = 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l87_87800


namespace yogurt_combinations_l87_87825

theorem yogurt_combinations (flavors toppings : ℕ) (h_flavors : flavors = 5) (h_toppings : toppings = 7) :
  (flavors * Nat.choose toppings 3) = 175 := by
  sorry

end yogurt_combinations_l87_87825


namespace scientific_notation_36000_l87_87751

theorem scientific_notation_36000 : 36000 = 3.6 * (10^4) := 
by 
  -- Skipping the proof by adding sorry
  sorry

end scientific_notation_36000_l87_87751


namespace coloring_integers_l87_87427

theorem coloring_integers 
  (color : ℤ → ℕ) 
  (x y : ℤ) 
  (hx : x % 2 = 1) 
  (hy : y % 2 = 1) 
  (h_neq : |x| ≠ |y|) 
  (h_color_range : ∀ n : ℤ, color n < 4) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = x ∨ a - b = y ∨ a - b = x + y ∨ a - b = x - y) :=
sorry

end coloring_integers_l87_87427


namespace find_new_songs_l87_87799

-- Definitions for the conditions
def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

-- The number of new songs added
def new_songs_added : ℕ := 20

-- Statement of the proof problem
theorem find_new_songs (n d f x : ℕ) (h1 : n = initial_songs) (h2 : d = deleted_songs) (h3 : f = final_songs) : f = n - d + x → x = new_songs_added :=
by
  intros h4
  sorry

end find_new_songs_l87_87799


namespace max_sum_integers_differ_by_60_l87_87263

theorem max_sum_integers_differ_by_60 (b : ℕ) (c : ℕ) (h_diff : 0 < b) (h_sqrt : (Nat.sqrt b : ℝ) + (Nat.sqrt (b + 60) : ℝ) = (Nat.sqrt c : ℝ)) (h_not_square : ¬ ∃ (k : ℕ), k * k = c) :
  ∃ (b : ℕ), b + (b + 60) = 156 := 
sorry

end max_sum_integers_differ_by_60_l87_87263


namespace negation_of_p_equiv_h_l87_87062

variable (p : ∀ x : ℝ, Real.sin x ≤ 1)
variable (h : ∃ x : ℝ, Real.sin x ≥ 1)

theorem negation_of_p_equiv_h : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x ≥ 1) :=
by
  sorry

end negation_of_p_equiv_h_l87_87062


namespace length_DE_l87_87430

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end length_DE_l87_87430


namespace cylinder_volume_from_cone_l87_87024

/-- Given the volume of a cone, prove the volume of a cylinder with the same base and height. -/
theorem cylinder_volume_from_cone (V_cone : ℝ) (h : V_cone = 3.6) : 
  ∃ V_cylinder : ℝ, V_cylinder = 0.0108 :=
by
  have V_cylinder := 3 * V_cone
  have V_cylinder_meters := V_cylinder / 1000
  use V_cylinder_meters
  sorry

end cylinder_volume_from_cone_l87_87024


namespace greatest_product_sum_2000_l87_87278

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87278


namespace min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l87_87978

variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b)

theorem min_ab : 16 ≤ a * b :=
sorry

theorem min_a_b : 9 ≤ a + b :=
sorry

theorem max_two_a_one_b : 2 > (2 / a + 1 / b) :=
sorry

theorem min_one_a_sq_four_b_sq : 1 / 5 ≤ (1 / a^2 + 4 / b^2) :=
sorry

end min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l87_87978


namespace radius_range_of_sector_l87_87441

theorem radius_range_of_sector (a : ℝ) (h : a > 0) :
  ∃ (R : ℝ), (a / (2 * (1 + π)) < R ∧ R < a / 2) :=
sorry

end radius_range_of_sector_l87_87441


namespace arithmetic_sequence_solution_l87_87391

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence : Prop :=
  ∀ n, a (n+1) = a n + d

-- The given condition a_3 + a_5 = 12 - a_7
def condition : Prop :=
  a 3 + a 5 = 12 - a 7

-- The proof statement
theorem arithmetic_sequence_solution 
  (h_arith : is_arithmetic_sequence a d) 
  (h_cond : condition a): a 1 + a 9 = 8 :=
sorry

end arithmetic_sequence_solution_l87_87391


namespace sector_area_l87_87872

noncomputable def radius_of_sector (l α : ℝ) : ℝ := l / α

noncomputable def area_of_sector (r l : ℝ) : ℝ := (1 / 2) * r * l

theorem sector_area {α l S : ℝ} (hα : α = 2) (hl : l = 3 * Real.pi) (hS : S = 9 * Real.pi ^ 2 / 4) :
  area_of_sector (radius_of_sector l α) l = S := 
by 
  rw [hα, hl, hS]
  rw [radius_of_sector, area_of_sector]
  sorry

end sector_area_l87_87872


namespace inequality_k_ge_2_l87_87582

theorem inequality_k_ge_2 {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℤ) (h_k : k ≥ 2) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a)) ≥ 3 / 2 :=
by
  sorry

end inequality_k_ge_2_l87_87582


namespace terry_lunch_combo_l87_87151

theorem terry_lunch_combo :
  let lettuce_options : ℕ := 2
  let tomato_options : ℕ := 3
  let olive_options : ℕ := 4
  let soup_options : ℕ := 2
  (lettuce_options * tomato_options * olive_options * soup_options = 48) := 
by
  sorry

end terry_lunch_combo_l87_87151


namespace after_2_pow_2009_days_is_monday_l87_87141

-- Define the current day as Thursday
def today := "Thursday"

-- Define the modulo operation for calculating days of the week
def day_of_week_after (days : ℕ) : ℕ :=
  days % 7

-- Define the exponent in question
def exponent := 2009

-- Since today is Thursday, which we can represent as 4 (considering Sunday as 0, Monday as 1, ..., Saturday as 6)
def today_as_num := 4

-- Calculate the day after 2^2009 days
def future_day := (today_as_num + day_of_week_after (2 ^ exponent)) % 7

-- Prove that the future_day is 1 (Monday)
theorem after_2_pow_2009_days_is_monday : future_day = 1 := by
  sorry

end after_2_pow_2009_days_is_monday_l87_87141


namespace percentage_decrease_of_y_compared_to_z_l87_87878

theorem percentage_decrease_of_y_compared_to_z (x y z : ℝ)
  (h1 : x = 1.20 * y)
  (h2 : x = 0.60 * z) :
  (y = 0.50 * z) → (1 - (y / z)) * 100 = 50 :=
by
  sorry

end percentage_decrease_of_y_compared_to_z_l87_87878


namespace mutually_exclusive_but_not_opposite_l87_87518

-- Define the cards and the people
inductive Card
| Red
| Black
| Blue
| White

inductive Person
| A
| B
| C
| D

-- Define the events
def eventA_gets_red (distribution : Person → Card) : Prop :=
distribution Person.A = Card.Red

def eventB_gets_red (distribution : Person → Card) : Prop :=
distribution Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Prop) : Prop :=
P → ¬ Q

-- Statement of the problem
theorem mutually_exclusive_but_not_opposite :
  ∀ (distribution : Person → Card), 
    mutually_exclusive (eventA_gets_red distribution) (eventB_gets_red distribution) ∧ 
    ¬ (eventA_gets_red distribution ↔ eventB_gets_red distribution) :=
by sorry

end mutually_exclusive_but_not_opposite_l87_87518


namespace union_condition_intersection_condition_l87_87409

def setA : Set ℝ := {x | x^2 - 5 * x + 6 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 3}

theorem union_condition (a : ℝ) : setA ∪ setB a = setB a ↔ a < 2 := sorry

theorem intersection_condition (a : ℝ) : setA ∩ setB a = setB a ↔ a ≥ 2 := sorry

end union_condition_intersection_condition_l87_87409


namespace distance_PF_l87_87187

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the point P on the parabola with x-coordinate 4
def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

-- Prove the distance |PF| for given conditions
theorem distance_PF
  (hP : ∃ y : ℝ, parabola 4 y)
  (hF : focus = (2, 0)) :
  ∃ y : ℝ, y^2 = 8 * 4 ∧ abs (4 - 2) + abs y = 6 := 
by
  sorry

end distance_PF_l87_87187


namespace angle_terminal_side_equiv_l87_87620

-- Define the function to check angle equivalence
def angle_equiv (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + k * 360

-- Theorem statement
theorem angle_terminal_side_equiv : angle_equiv 330 (-30) :=
  sorry

end angle_terminal_side_equiv_l87_87620


namespace prove_Φ_eq_8_l87_87681

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end prove_Φ_eq_8_l87_87681


namespace sequence_S_n_a_n_l87_87861

noncomputable def sequence_S (n : ℕ) : ℝ := -1 / (n : ℝ)

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then -1 else 1 / ((n : ℝ) * (n - 1))

theorem sequence_S_n_a_n (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = -1 →
  (∀ n, (a (n + 1)) / (S (n + 1)) = S n) →
  S n = sequence_S n ∧ a n = sequence_a n :=
by
  intros h1 h2
  sorry

end sequence_S_n_a_n_l87_87861


namespace C0E_hex_to_dec_l87_87358

theorem C0E_hex_to_dec : 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  result = 3086 :=
by 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  sorry

end C0E_hex_to_dec_l87_87358


namespace round_robin_cycles_l87_87664

-- Define the conditions
def teams : ℕ := 28
def wins_per_team : ℕ := 13
def losses_per_team : ℕ := 13
def total_teams_games := teams * (teams - 1) / 2
def sets_of_three_teams := (teams * (teams - 1) * (teams - 2)) / 6

-- Define the problem statement
theorem round_robin_cycles :
  -- We need to show that the number of sets of three teams {A, B, C} where A beats B, B beats C, and C beats A is 1092
  (sets_of_three_teams - (teams * (wins_per_team * (wins_per_team - 1)) / 2)) = 1092 :=
by
  sorry

end round_robin_cycles_l87_87664


namespace consecutive_even_integers_sum_l87_87772

theorem consecutive_even_integers_sum :
  ∀ (y : Int), (y = 2 * (y + 2)) → y + (y + 2) = -6 :=
by
  intro y
  intro h
  sorry

end consecutive_even_integers_sum_l87_87772


namespace value_of_k_l87_87619

theorem value_of_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (hk : k ≠ 0) : k = 8 :=
sorry

end value_of_k_l87_87619


namespace sin_cos_quotient_l87_87186

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_prime (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem sin_cos_quotient 
  (x : ℝ)
  (h : f_prime x = 3 * f x) 
  : (Real.sin x ^ 2 - 3) / (Real.cos x ^ 2 + 1) = -14 / 9 := 
by 
  sorry

end sin_cos_quotient_l87_87186


namespace cos_45_minus_cos_90_eq_sqrt2_over_2_l87_87602

theorem cos_45_minus_cos_90_eq_sqrt2_over_2 :
  (Real.cos (45 * Real.pi / 180) - Real.cos (90 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  have h1 : Real.cos (90 * Real.pi / 180) = 0 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  sorry

end cos_45_minus_cos_90_eq_sqrt2_over_2_l87_87602


namespace sophie_total_spending_l87_87423

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end sophie_total_spending_l87_87423


namespace radius_of_circumscribed_sphere_eq_a_l87_87246

-- Assume a to be a real number representing the side length of the base and height of the hexagonal pyramid
variables (a : ℝ)

-- Representing the base as a regular hexagon and the pyramid as having equal side length and height
def regular_hexagonal_pyramid (a : ℝ) : Type := {b : ℝ // b = a}

-- The radius of the circumscribed sphere to a given regular hexagonal pyramid
def radius_of_circumscribed_sphere (a : ℝ) : ℝ := a

-- Theorem stating that the radius of the sphere circumscribed around a regular hexagonal pyramid 
-- with side length and height both equal to a is a
theorem radius_of_circumscribed_sphere_eq_a (a : ℝ) :
  radius_of_circumscribed_sphere a = a :=
by {
  sorry
}

end radius_of_circumscribed_sphere_eq_a_l87_87246


namespace brad_siblings_product_l87_87388

theorem brad_siblings_product (S B : ℕ) (hS : S = 5) (hB : B = 7) : S * B = 35 :=
by
  have : S = 5 := hS
  have : B = 7 := hB
  sorry

end brad_siblings_product_l87_87388


namespace max_product_of_sum_2000_l87_87304

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87304


namespace dan_remaining_money_l87_87953

noncomputable def calculate_remaining_money (initial_amount : ℕ) : ℕ :=
  let candy_bars_qty := 5
  let candy_bar_price := 125
  let candy_bars_discount := 10
  let gum_qty := 3
  let gum_price := 80
  let soda_qty := 4
  let soda_price := 240
  let chips_qty := 2
  let chip_price := 350
  let chips_discount := 15
  let low_tax := 7
  let high_tax := 12

  let total_candy_bars_cost := candy_bars_qty * candy_bar_price
  let discounted_candy_bars_cost := total_candy_bars_cost * (100 - candy_bars_discount) / 100

  let total_gum_cost := gum_qty * gum_price

  let total_soda_cost := soda_qty * soda_price

  let total_chips_cost := chips_qty * chip_price
  let discounted_chips_cost := total_chips_cost * (100 - chips_discount) / 100

  let candy_bars_tax := discounted_candy_bars_cost * low_tax / 100
  let gum_tax := total_gum_cost * low_tax / 100

  let soda_tax := total_soda_cost * high_tax / 100
  let chips_tax := discounted_chips_cost * high_tax / 100

  let total_candy_bars_with_tax := discounted_candy_bars_cost + candy_bars_tax
  let total_gum_with_tax := total_gum_cost + gum_tax
  let total_soda_with_tax := total_soda_cost + soda_tax
  let total_chips_with_tax := discounted_chips_cost + chips_tax

  let total_cost := total_candy_bars_with_tax + total_gum_with_tax + total_soda_with_tax + total_chips_with_tax

  initial_amount - total_cost

theorem dan_remaining_money : 
  calculate_remaining_money 10000 = 7399 :=
sorry

end dan_remaining_money_l87_87953


namespace estimated_probability_is_2_div_9_l87_87155

def groups : List (List ℕ) :=
  [[3, 4, 3], [4, 3, 2], [3, 4, 1], [3, 4, 2], [2, 3, 4], [1, 4, 2], [2, 4, 3], [3, 3, 1], [1, 1, 2],
   [3, 4, 2], [2, 4, 1], [2, 4, 4], [4, 3, 1], [2, 3, 3], [2, 1, 4], [3, 4, 4], [1, 4, 2], [1, 3, 4]]

def count_desired_groups (gs : List (List ℕ)) : Nat :=
  gs.foldl (fun acc g =>
    if g.contains 1 ∧ g.contains 2 ∧ g.length ≥ 3 then acc + 1 else acc) 0

theorem estimated_probability_is_2_div_9 :
  (count_desired_groups groups) = 4 →
  4 / 18 = 2 / 9 :=
by
  intro h
  sorry

end estimated_probability_is_2_div_9_l87_87155


namespace parabola_midpoint_AB_square_length_l87_87099

noncomputable def parabola_y (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 4

theorem parabola_midpoint_AB_square_length :
  let A := (7/6, parabola_y (7/6))
  let B := (5/6, parabola_y (5/6))
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  midpoint = (1, 2) →
  let AB_squared := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2
  AB_squared = sorry :=
begin
  sorry
end

end parabola_midpoint_AB_square_length_l87_87099


namespace find_x_l87_87452

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |> List.sum

def k := (10^45 - 999999999999999999999999999999999999999999994 : ℕ)

theorem find_x :
  digit_sum k = 397 := 
sorry

end find_x_l87_87452


namespace mirror_area_proof_l87_87412

-- Definitions of conditions
def outer_width := 100
def outer_height := 70
def frame_width := 15
def mirror_width := outer_width - 2 * frame_width -- 100 - 2 * 15 = 70
def mirror_height := outer_height - 2 * frame_width -- 70 - 2 * 15 = 40

-- Statement of the proof problem
theorem mirror_area_proof : 
  (mirror_width * mirror_height) = 2800 := 
by
  sorry

end mirror_area_proof_l87_87412


namespace lcm_12_18_l87_87536

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l87_87536


namespace troy_needs_more_money_l87_87783

theorem troy_needs_more_money (initial_savings : ℕ) (sold_computer : ℕ) (new_computer_cost : ℕ) :
  initial_savings = 50 → sold_computer = 20 → new_computer_cost = 80 → 
  new_computer_cost - (initial_savings + sold_computer) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end troy_needs_more_money_l87_87783


namespace total_profit_amount_l87_87341

-- Definitions representing the conditions:
def ratio_condition (P_X P_Y : ℝ) : Prop :=
  P_X / P_Y = (1 / 2) / (1 / 3)

def difference_condition (P_X P_Y : ℝ) : Prop :=
  P_X - P_Y = 160

-- The proof problem statement:
theorem total_profit_amount (P_X P_Y : ℝ) (h1 : ratio_condition P_X P_Y) (h2 : difference_condition P_X P_Y) :
  P_X + P_Y = 800 := by
  sorry

end total_profit_amount_l87_87341


namespace probability_wait_at_least_10_seconds_l87_87148

theorem probability_wait_at_least_10_seconds
  (red_duration : ℚ)
  (green_duration : ℚ)
  (yellow_duration : ℚ)
  (wait_time : ℚ)
  (encountered_red : Bool)
  (red_duration_eq : red_duration = 30)
  (green_duration_eq : green_duration = 30)
  (yellow_duration_eq : yellow_duration = 5)
  (wait_time_eq : wait_time = 10)
  (initial_red : encountered_red = true) :
  ((red_duration - wait_time) / red_duration) = 2 / 3 :=
by
  -- Placeholder proof
  sorry

end probability_wait_at_least_10_seconds_l87_87148


namespace units_digit_47_pow_47_l87_87462

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l87_87462


namespace ratio_a_div_8_to_b_div_7_l87_87565

theorem ratio_a_div_8_to_b_div_7 (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 8) / (b / 7) = 1 :=
sorry

end ratio_a_div_8_to_b_div_7_l87_87565


namespace min_value_m_plus_2n_exists_min_value_l87_87067

variable (n : ℝ) -- Declare n as a real number.

-- Define m in terms of n
def m (n : ℝ) : ℝ := n^2

-- State and prove that the minimum value of m + 2n is -1
theorem min_value_m_plus_2n : (m n + 2 * n) ≥ -1 :=
by sorry

-- Show there exists an n such that m + 2n = -1
theorem exists_min_value : ∃ n : ℝ, m n + 2 * n = -1 :=
by sorry

end min_value_m_plus_2n_exists_min_value_l87_87067


namespace line_parallel_condition_l87_87328

theorem line_parallel_condition (a : ℝ) : (a = 2) ↔ (∀ x y : ℝ, (ax + 2 * y = 0 → x + y ≠ 1)) :=
by
  sorry

end line_parallel_condition_l87_87328


namespace Mark_average_speed_l87_87096

theorem Mark_average_speed 
  (start_time : ℝ) (end_time : ℝ) (distance : ℝ)
  (h1 : start_time = 8.5) (h2 : end_time = 14.75) (h3 : distance = 210) :
  distance / (end_time - start_time) = 33.6 :=
by 
  sorry

end Mark_average_speed_l87_87096


namespace ali_ate_half_to_percent_l87_87036

theorem ali_ate_half_to_percent : (1 / 2 : ℚ) * 100 = 50 := by
  sorry

end ali_ate_half_to_percent_l87_87036


namespace simplify_expression_l87_87421

variable (x : ℝ)

theorem simplify_expression :
  (2 * x ^ 6 + 3 * x ^ 5 + x ^ 4 + x ^ 3 + 5) - (x ^ 6 + 4 * x ^ 5 + 2 * x ^ 4 - x ^ 3 + 7) = 
  x ^ 6 - x ^ 5 - x ^ 4 + 2 * x ^ 3 - 2 := by
  sorry

end simplify_expression_l87_87421


namespace relationship_abc_l87_87979

noncomputable def a : ℝ := (0.7 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (-0.6 : ℝ)
noncomputable def c : ℝ := (0.6 : ℝ) ^ (0.7 : ℝ)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Proof will go here
  sorry

end relationship_abc_l87_87979


namespace max_product_two_integers_sum_2000_l87_87280

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87280


namespace average_annual_growth_rate_l87_87072

-- Definitions of the provided conditions
def initial_amount : ℝ := 200
def final_amount : ℝ := 338
def periods : ℝ := 2

-- Statement of the goal
theorem average_annual_growth_rate :
  (final_amount / initial_amount)^(1 / periods) - 1 = 0.3 := 
sorry

end average_annual_growth_rate_l87_87072


namespace solve_for_c_l87_87721

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem solve_for_c (a b c d : ℝ) 
    (h : ∀ x : ℝ, quadratic_function a b c x ≥ d) : c = d + b^2 / (4 * a) :=
by
  sorry

end solve_for_c_l87_87721


namespace relationship_between_fractions_l87_87583

variable (a a' b b' : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a' > 0)
variable (h₃ : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2)

theorem relationship_between_fractions
  (a : ℝ) (a' : ℝ) (b : ℝ) (b' : ℝ)
  (h1 : a > 0) (h2 : a' > 0)
  (h3 : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2) :
  (b^2) / (a^2) > (b'^2) / (a'^2) :=
by sorry

end relationship_between_fractions_l87_87583


namespace Rebecca_worked_56_l87_87139

-- Define the conditions
variables (x : ℕ)
def Toby_hours := 2 * x - 10
def Rebecca_hours := Toby_hours - 8
def Total_hours := x + Toby_hours + Rebecca_hours

-- Theorem stating that under the given conditions, Rebecca worked 56 hours
theorem Rebecca_worked_56 
  (h : Total_hours = 157) 
  (hx : x = 37) : Rebecca_hours = 56 :=
by sorry

end Rebecca_worked_56_l87_87139


namespace prime_pair_solution_l87_87044

-- Steps a) and b) are incorporated into this Lean statement
theorem prime_pair_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q ∣ 3^p + 3^q ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) :=
sorry

end prime_pair_solution_l87_87044


namespace problem_l87_87669

-- Definition of triangular number
def is_triangular (n k : ℕ) := n = k * (k + 1) / 2

-- Definition of choosing 2 marbles
def choose_2 (n m : ℕ) := n = m * (m - 1) / 2

-- Definition of Cathy's condition
def cathy_condition (n s : ℕ) := s * s < 2 * n ∧ 2 * n - s * s = 20

theorem problem (n k m s : ℕ) :
  is_triangular n k →
  choose_2 n m →
  cathy_condition n s →
  n = 210 :=
by
  sorry

end problem_l87_87669


namespace cos_C_in_triangle_l87_87393

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end cos_C_in_triangle_l87_87393


namespace people_in_room_eq_33_l87_87255

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end people_in_room_eq_33_l87_87255


namespace max_product_two_integers_sum_2000_l87_87283

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87283


namespace radius_of_third_circle_l87_87011

open Real

theorem radius_of_third_circle (r : ℝ) :
  let r_large := 40
  let r_small := 25
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  let region_area := area_large - area_small
  let half_region_area := region_area / 2
  let third_circle_area := π * r^2
  (third_circle_area = half_region_area) -> r = 15 * sqrt 13 :=
by
  sorry

end radius_of_third_circle_l87_87011


namespace common_point_of_function_and_inverse_l87_87672

-- Define the points P, Q, M, and N
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (1, 2)
def M : ℝ × ℝ := (2, 3)
def N : ℝ × ℝ := (0.5, 0.25)

-- Define a predicate to check if a point lies on the line y = x
def lies_on_y_eq_x (point : ℝ × ℝ) : Prop := point.1 = point.2

-- The main theorem statement
theorem common_point_of_function_and_inverse (a : ℝ) : 
  lies_on_y_eq_x P ∧ ¬ lies_on_y_eq_x Q ∧ ¬ lies_on_y_eq_x M ∧ ¬ lies_on_y_eq_x N :=
by
  -- We write 'sorry' here to skip the proof
  sorry

end common_point_of_function_and_inverse_l87_87672


namespace geometric_sequence_second_term_l87_87627

theorem geometric_sequence_second_term (a_1 q a_3 a_4 : ℝ) (h3 : a_1 * q^2 = 12) (h4 : a_1 * q^3 = 18) : a_1 * q = 8 :=
by
  sorry

end geometric_sequence_second_term_l87_87627


namespace infinite_fractions_2_over_odd_l87_87640

theorem infinite_fractions_2_over_odd (a b : ℕ) (n : ℕ) : 
  (a = 2 → 2 * b + 1 ≠ 0) ∧ ((b = 2 * n + 1) → (2 + 2) / (2 * (2 * n + 1)) = 2 / (2 * n + 1)) ∧ (a / b = 2 / (2 * n + 1)) :=
by
  sorry

end infinite_fractions_2_over_odd_l87_87640


namespace value_of_expression_l87_87981

open Real

theorem value_of_expression (m n r t : ℝ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := 
by
  sorry

end value_of_expression_l87_87981


namespace solution_set_of_inequality_l87_87625

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 3 * x - 2 > 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l87_87625


namespace minimum_workers_needed_l87_87040

-- Definitions
def job_completion_time : ℕ := 45
def days_worked : ℕ := 9
def portion_job_done : ℚ := 1 / 5
def team_size : ℕ := 10
def job_remaining : ℚ := (1 - portion_job_done)
def days_remaining : ℕ := job_completion_time - days_worked
def daily_completion_rate_by_team : ℚ := portion_job_done / days_worked
def daily_completion_rate_per_person : ℚ := daily_completion_rate_by_team / team_size
def required_daily_rate : ℚ := job_remaining / days_remaining

-- Statement to be proven
theorem minimum_workers_needed :
  (required_daily_rate / daily_completion_rate_per_person) = 10 :=
sorry

end minimum_workers_needed_l87_87040


namespace intersection_A_B_l87_87940

namespace SetTheory

open Set

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end SetTheory

end intersection_A_B_l87_87940


namespace problem_inequality_l87_87089

open Real

theorem problem_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_prod : x * y * z = 1) :
    1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x :=
sorry

end problem_inequality_l87_87089


namespace probability_not_sit_at_ends_l87_87097

theorem probability_not_sit_at_ends (h1: ∀ M J: ℕ, M ≠ J → M ≠ 1 ∧ M ≠ 8 ∧ J ≠ 1 ∧ J ≠ 8) : 
  (∃ p: ℚ, p = (3 / 7)) :=
by 
  sorry

end probability_not_sit_at_ends_l87_87097


namespace find_abcd_abs_eq_one_l87_87093

noncomputable def non_zero_real (r : ℝ) := r ≠ 0

theorem find_abcd_abs_eq_one
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : a^2 + (1/b) = b^2 + (1/c) ∧ b^2 + (1/c) = c^2 + (1/d) ∧ c^2 + (1/d) = d^2 + (1/a)) :
  |a * b * c * d| = 1 :=
sorry

end find_abcd_abs_eq_one_l87_87093


namespace reusable_bag_trips_correct_lowest_carbon_solution_l87_87411

open Real

-- Conditions definitions
def canvas_CO2 := 600 -- in pounds
def polyester_CO2 := 250 -- in pounds
def recycled_plastic_CO2 := 150 -- in pounds
def CO2_per_plastic_bag := 4 / 16 -- 4 ounces per bag, converted to pounds
def bags_per_trip := 8

-- Total CO2 per trip using plastic bags
def CO2_per_trip := CO2_per_plastic_bag * bags_per_trip

-- Proof of correct number of trips
theorem reusable_bag_trips_correct :
  canvas_CO2 / CO2_per_trip = 300 ∧
  polyester_CO2 / CO2_per_trip = 125 ∧
  recycled_plastic_CO2 / CO2_per_trip = 75 :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

-- Proof that recycled plastic bag is the lowest-carbon solution
theorem lowest_carbon_solution :
  min (canvas_CO2 / CO2_per_trip) (min (polyester_CO2 / CO2_per_trip) (recycled_plastic_CO2 / CO2_per_trip)) = recycled_plastic_CO2 / CO2_per_trip :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

end reusable_bag_trips_correct_lowest_carbon_solution_l87_87411


namespace jose_tabs_remaining_l87_87890

def initial_tabs : Nat := 400
def step1_tabs_closed (n : Nat) : Nat := n / 4
def step2_tabs_closed (n : Nat) : Nat := 2 * n / 5
def step3_tabs_closed (n : Nat) : Nat := n / 2

theorem jose_tabs_remaining :
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  after_step3 = 90 :=
by
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  have h : after_step3 = 90 := sorry
  exact h

end jose_tabs_remaining_l87_87890


namespace number_of_even_red_faces_cubes_l87_87346

def painted_cubes_even_faces : Prop :=
  let block_length := 4
  let block_width := 4
  let block_height := 1
  let edge_cubes_count := 8  -- The count of edge cubes excluding corners
  edge_cubes_count = 8

theorem number_of_even_red_faces_cubes : painted_cubes_even_faces := by
  sorry

end number_of_even_red_faces_cubes_l87_87346


namespace sophie_total_spending_l87_87424

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end sophie_total_spending_l87_87424


namespace area_reflected_arcs_l87_87032

theorem area_reflected_arcs (s : ℝ) (h : s = 2) : 
  ∃ A, A = 2 * Real.pi * Real.sqrt 2 - 8 :=
by
  -- constants
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := Real.pi * r^2 / 8
  let triangle_area := 1 -- Equilateral triangle properties
  let reflected_arc_area := sector_area - triangle_area
  let total_area := 8 * reflected_arc_area
  use total_area
  sorry

end area_reflected_arcs_l87_87032


namespace parallel_lines_l87_87991

theorem parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, a * x + 2 * y - 1 = k * (2 * x + a * y + 2)) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end parallel_lines_l87_87991


namespace percentage_pine_cones_on_roof_l87_87829

theorem percentage_pine_cones_on_roof 
  (num_trees : Nat) 
  (pine_cones_per_tree : Nat) 
  (pine_cone_weight_oz : Nat) 
  (total_pine_cone_weight_on_roof_oz : Nat) 
  : num_trees = 8 ∧ pine_cones_per_tree = 200 ∧ pine_cone_weight_oz = 4 ∧ total_pine_cone_weight_on_roof_oz = 1920 →
    (total_pine_cone_weight_on_roof_oz / pine_cone_weight_oz) / (num_trees * pine_cones_per_tree) * 100 = 30 := 
by
  sorry

end percentage_pine_cones_on_roof_l87_87829


namespace inequality_proof_l87_87548

theorem inequality_proof
  (n : ℕ) (hn : n ≥ 3) (x y z : ℝ) (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (hxyz_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 :=
by sorry

end inequality_proof_l87_87548


namespace Alyssa_weekly_allowance_l87_87037

theorem Alyssa_weekly_allowance
  (A : ℝ)
  (h1 : A / 2 + 8 = 12) :
  A = 8 := 
sorry

end Alyssa_weekly_allowance_l87_87037


namespace exists_ordering_no_arithmetic_progression_l87_87048

theorem exists_ordering_no_arithmetic_progression (m : ℕ) (hm : 0 < m) :
  ∃ (a : Fin (2^m) → ℕ), (∀ i j k : Fin (2^m), i < j → j < k → a j - a i ≠ a k - a j) := sorry

end exists_ordering_no_arithmetic_progression_l87_87048


namespace baker_total_cost_is_correct_l87_87484

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end baker_total_cost_is_correct_l87_87484


namespace election_winner_votes_l87_87326

theorem election_winner_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 360) :
  0.62 * V = 930 :=
by {
  sorry
}

end election_winner_votes_l87_87326


namespace intersection_is_23_l87_87871

open Set

def setA : Set ℤ := {1, 2, 3, 4}
def setB : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem intersection_is_23 : setA ∩ setB = {2, 3} := 
by 
  sorry

end intersection_is_23_l87_87871


namespace shaded_area_approx_l87_87511

-- Define the dimensions of the rectangle and the two circles as given in the conditions
def length : ℝ := 16
def width : ℝ := 8
def radius_large : ℝ := 4
def radius_small : ℝ := 2

-- Define the area of the rectangle
def area_rectangle : ℝ := length * width

-- Define the area of the larger circle
def area_large_circle : ℝ := Real.pi * (radius_large ^ 2)

-- Define the area of the smaller circle
def area_small_circle : ℝ := Real.pi * (radius_small ^ 2)

-- Define the total area subtracted due to the two circles
def area_total_subtracted : ℝ := area_large_circle + area_small_circle

-- Define the shaded area remaining in the rectangle
def area_shaded : ℝ := area_rectangle - area_total_subtracted

-- Lean statement to prove the total shaded area is approximately 65.2 square feet
theorem shaded_area_approx : area_shaded ≈ 65.2 := by
  -- apply the actual computation here and show that area_shaded is approximately 65.2
  sorry

end shaded_area_approx_l87_87511


namespace boat_speed_in_still_water_l87_87321

variable (B S : ℝ)

def downstream_speed := 10
def upstream_speed := 4

theorem boat_speed_in_still_water :
  B + S = downstream_speed → 
  B - S = upstream_speed → 
  B = 7 :=
by
  intros h₁ h₂
  -- We would insert the proof steps here
  sorry

end boat_speed_in_still_water_l87_87321


namespace book_price_increase_percentage_l87_87646

theorem book_price_increase_percentage :
  let P_original := 300
  let P_new := 480
  (P_new - P_original : ℝ) / P_original * 100 = 60 :=
by
  sorry

end book_price_increase_percentage_l87_87646


namespace largest_possible_value_of_b_l87_87895

theorem largest_possible_value_of_b (b : ℚ) (h : (3 * b + 4) * (b - 2) = 9 * b) : b ≤ 4 :=
sorry

end largest_possible_value_of_b_l87_87895


namespace max_non_intersecting_diagonals_l87_87014

theorem max_non_intersecting_diagonals (n : ℕ) (h : n ≥ 3) :
  ∃ k, k ≤ n - 3 ∧ (∀ m, m > k → ¬(m ≤ n - 3)) :=
by
  sorry

end max_non_intersecting_diagonals_l87_87014


namespace min_value_y_l87_87366

noncomputable def y (x : ℝ) := x^4 - 4*x + 3

theorem min_value_y : ∃ x ∈ Set.Icc (-2 : ℝ) 3, y x = 0 ∧ ∀ x' ∈ Set.Icc (-2 : ℝ) 3, y x' ≥ 0 :=
by
  sorry

end min_value_y_l87_87366


namespace function_domain_l87_87244

theorem function_domain (x : ℝ) :
  (x + 5 ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≥ -5) ∧ (x ≠ -2) :=
by
  sorry

end function_domain_l87_87244


namespace total_recovery_time_l87_87480

theorem total_recovery_time 
  (lions: ℕ := 3) (rhinos: ℕ := 2) (time_per_animal: ℕ := 2) :
  (lions + rhinos) * time_per_animal = 10 := by
  sorry

end total_recovery_time_l87_87480


namespace car_speed_without_red_light_l87_87832

theorem car_speed_without_red_light (v : ℝ) :
  (∃ k : ℕ+, v = 10 / k) ↔ 
  ∀ (dist : ℝ) (green_duration red_duration total_cycle : ℝ),
    dist = 1500 ∧ green_duration = 90 ∧ red_duration = 60 ∧ total_cycle = 150 →
    v * total_cycle = dist / (green_duration + red_duration) := 
by
  sorry

end car_speed_without_red_light_l87_87832


namespace find_t_value_l87_87195

theorem find_t_value (t : ℝ) (a b : ℝ × ℝ) (h₁ : a = (t, 1)) (h₂ : b = (1, 2)) 
  (h₃ : (a.1 + b.1)^2 + (a.2 + b.2)^2 = a.1^2 + a.2^2 + b.1^2 + b.2^2) : 
  t = -2 :=
by 
  sorry

end find_t_value_l87_87195


namespace actual_length_correct_l87_87753

-- Definitions based on the conditions
def blueprint_scale : ℝ := 20
def measured_length_cm : ℝ := 16

-- Statement of the proof problem
theorem actual_length_correct :
  measured_length_cm * blueprint_scale = 320 := 
sorry

end actual_length_correct_l87_87753


namespace units_digit_47_pow_47_l87_87454

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l87_87454


namespace discount_on_pickles_l87_87738

theorem discount_on_pickles :
  ∀ (meat_weight : ℝ) (meat_price_per_pound : ℝ) (bun_price : ℝ) (lettuce_price : ℝ)
    (tomato_weight : ℝ) (tomato_price_per_pound : ℝ) (pickles_price : ℝ) (total_paid : ℝ) (change : ℝ),
  meat_weight = 2 ∧
  meat_price_per_pound = 3.50 ∧
  bun_price = 1.50 ∧
  lettuce_price = 1.00 ∧
  tomato_weight = 1.5 ∧
  tomato_price_per_pound = 2.00 ∧
  pickles_price = 2.50 ∧
  total_paid = 20.00 ∧
  change = 6 →
  pickles_price - (total_paid - change - (meat_weight * meat_price_per_pound + tomato_weight * tomato_price_per_pound + bun_price + lettuce_price)) = 1 := 
by
  -- Begin the proof here (not required for this task)
  sorry

end discount_on_pickles_l87_87738


namespace fraction_of_men_collected_dues_l87_87162

theorem fraction_of_men_collected_dues
  (M W : ℕ)
  (x : ℚ)
  (h1 : 45 * x * M + 5 * W = 17760)
  (h2 : M + W = 3552)
  (h3 : 1 / 12 * W = W / 12) :
  x = 1 / 9 :=
by
  -- Proof steps would go here
  sorry

end fraction_of_men_collected_dues_l87_87162


namespace banks_should_offer_benefits_to_seniors_l87_87600

-- Definitions based on conditions
def better_credit_reliability (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_better_credit_reliability p

def stable_pension_income (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_stable_income p

def indirect_financial_benefits (pensioners : Type) : Prop :=
  ∀ (p : pensioners), receives_financial_benefit p

def propensity_to_save (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_saving_habits p

def preference_long_term_deposits (pensioners : Type) : Prop :=
  ∀ (p : pensioners), prefers_long_term_deposits p

-- Main theorem statement
theorem banks_should_offer_benefits_to_seniors
  (P : Type)
  (h1 : better_credit_reliability P)
  (h2 : stable_pension_income P)
  (h3 : indirect_financial_benefits P)
  (h4 : propensity_to_save P)
  (h5 : preference_long_term_deposits P) :
  ∃ benefits : Type, benefits.make_sense :=
sorry

end banks_should_offer_benefits_to_seniors_l87_87600


namespace smallest_whole_number_l87_87016

theorem smallest_whole_number (a : ℕ) : 
  (a % 4 = 1) ∧ (a % 3 = 1) ∧ (a % 5 = 2) → a = 37 :=
by
  intros
  sorry

end smallest_whole_number_l87_87016


namespace production_value_n_l87_87387

theorem production_value_n :
  -- Definitions based on conditions:
  (∀ a b : ℝ,
    (120 * a + 120 * b) / 60 = 6 ∧
    (100 * a + 100 * b) / 30 = 30) →
  (∃ n : ℝ, 80 * 3 * (a + b) = 480 * a + n * b) →
  n = 120 :=
by
  sorry

end production_value_n_l87_87387


namespace base_circumference_of_cone_l87_87937

theorem base_circumference_of_cone (r : ℝ) (theta : ℝ) (C : ℝ) 
  (h_radius : r = 6)
  (h_theta : theta = 180)
  (h_C : C = 2 * Real.pi * r) :
  (theta / 360) * C = 6 * Real.pi :=
by
  sorry

end base_circumference_of_cone_l87_87937


namespace choose_4_out_of_10_l87_87077

theorem choose_4_out_of_10 : Nat.choose 10 4 = 210 := by
  sorry

end choose_4_out_of_10_l87_87077


namespace trig_identity_problem_l87_87368

theorem trig_identity_problem {α : ℝ} (h : Real.tan α = 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trig_identity_problem_l87_87368


namespace largest_whole_number_solution_for_inequality_l87_87787

theorem largest_whole_number_solution_for_inequality :
  ∀ (x : ℕ), ((1 : ℝ) / 4 + (x : ℝ) / 5 < 2) → x ≤ 23 :=
by sorry

end largest_whole_number_solution_for_inequality_l87_87787


namespace compute_f_g_2_l87_87566

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem compute_f_g_2 : f (g 2) = -19 := 
by {
  sorry
}

end compute_f_g_2_l87_87566


namespace arrangement_of_accommodation_l87_87515

open Nat

noncomputable def num_arrangements_accommodation : ℕ :=
  (factorial 13) / ((factorial 2) * (factorial 2) * (factorial 2) * (factorial 2))

theorem arrangement_of_accommodation : num_arrangements_accommodation = 389188800 := by
  sorry

end arrangement_of_accommodation_l87_87515


namespace cauliflower_area_l87_87659

theorem cauliflower_area
  (s : ℕ) (a : ℕ) 
  (H1 : s * s / a = 40401)
  (H2 : s * s / a = 40000) :
  a = 1 :=
sorry

end cauliflower_area_l87_87659


namespace n_divisible_by_100_l87_87568

theorem n_divisible_by_100 
    (n : ℕ) 
    (h_pos : 0 < n) 
    (h_div : 100 ∣ n^3) : 
    100 ∣ n := 
sorry

end n_divisible_by_100_l87_87568


namespace cylinder_height_relationship_l87_87012

variables (π r₁ r₂ h₁ h₂ : ℝ)

theorem cylinder_height_relationship
  (h_volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius_rel : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
by {
  sorry -- proof not required as per instructions
}

end cylinder_height_relationship_l87_87012


namespace car_average_speed_l87_87020

noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 24
  let t3 := (D / 3) / 30
  let total_time := t1 + t2 + t3
  D / total_time

theorem car_average_speed :
  average_speed D = 34.2857 := by
  sorry

end car_average_speed_l87_87020


namespace problem_1_problem_2_l87_87713

open Set

-- First problem: when a = 2
theorem problem_1:
  ∀ (x : ℝ), 2 * x^2 - x - 1 > 0 ↔ (x < -(1 / 2) ∨ x > 1) :=
by
  sorry

-- Second problem: when a > -1
theorem problem_2 (a : ℝ) (h : a > -1) :
  ∀ (x : ℝ), 
    (if a = 0 then x - 1 > 0 else if a > 0 then  a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (x < -1 / a ∨ x > 1) 
    else a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (1 < x ∧ x < -1 / a)) :=
by
  sorry

end problem_1_problem_2_l87_87713


namespace degree_g_is_six_l87_87240

theorem degree_g_is_six 
  (f g : Polynomial ℂ) 
  (h : Polynomial ℂ) 
  (h_def : h = f.comp g + Polynomial.X * g) 
  (deg_h : h.degree = 7) 
  (deg_f : f.degree = 3) 
  : g.degree = 6 := 
sorry

end degree_g_is_six_l87_87240


namespace biscuits_initial_l87_87102

theorem biscuits_initial (F M A L B : ℕ) 
  (father_gave : F = 13) 
  (mother_gave : M = 15) 
  (brother_ate : A = 20) 
  (left_with : L = 40) 
  (remaining : B + F + M - A = L) :
  B = 32 := 
by 
  subst father_gave
  subst mother_gave
  subst brother_ate
  subst left_with
  simp at remaining
  linarith

end biscuits_initial_l87_87102


namespace area_quadrilateral_is_60_l87_87418

-- Definitions of the lengths of the quadrilateral sides and the ratio condition
def AB : ℝ := 8
def BC : ℝ := 5
def CD : ℝ := 17
def DA : ℝ := 10

-- Function representing the area of the quadrilateral ABCD
def area_ABCD (AB BC CD DA : ℝ) (ratio: ℝ) : ℝ :=
  -- Here we define the function to calculate the area, incorporating the given ratio
  sorry

-- The theorem to show that the area of quadrilateral ABCD is 60
theorem area_quadrilateral_is_60 : 
  area_ABCD AB BC CD DA (1/2) = 60 :=
by
  sorry

end area_quadrilateral_is_60_l87_87418


namespace inequality_solution_l87_87844

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ (-1 < x)) :=
by sorry

end inequality_solution_l87_87844


namespace second_divisor_is_340_l87_87788

theorem second_divisor_is_340 
  (n : ℕ)
  (h1 : n = 349)
  (h2 : n % 13 = 11)
  (h3 : n % D = 9) : D = 340 :=
by
  sorry

end second_divisor_is_340_l87_87788


namespace trick_deck_cost_l87_87697

theorem trick_deck_cost :
  ∀ (x : ℝ), 3 * x + 2 * x = 35 → x = 7 :=
by
  sorry

end trick_deck_cost_l87_87697


namespace triangle_angle_C_triangle_max_area_l87_87218

noncomputable def cos (θ : Real) : Real := sorry
noncomputable def sin (θ : Real) : Real := sorry

theorem triangle_angle_C (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) : C = (2 * Real.pi) / 3 :=
sorry

theorem triangle_max_area (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) (hc : c = 6)
  (hC : C = (2 * Real.pi) / 3) : 
  ∃ (S : Real), S = 3 * Real.sqrt 3 := 
sorry

end triangle_angle_C_triangle_max_area_l87_87218


namespace P_inequality_l87_87087

def P (n : ℕ) (x : ℝ) : ℝ := (Finset.range (n + 1)).sum (λ k => x^k)

theorem P_inequality (x : ℝ) (hx : 0 < x) :
  P 20 x * P 21 (x^2) ≤ P 20 (x^2) * P 22 x :=
by
  sorry

end P_inequality_l87_87087


namespace banks_policies_for_seniors_justified_l87_87599

-- Defining conditions
def better_credit_repayment_reliability : Prop := sorry
def stable_pension_income : Prop := sorry
def indirect_younger_relative_contributions : Prop := sorry
def pensioners_inclination_to_save : Prop := sorry
def regular_monthly_income : Prop := sorry
def preference_for_long_term_deposits : Prop := sorry

-- Lean theorem statement using the conditions
theorem banks_policies_for_seniors_justified :
  better_credit_repayment_reliability →
  stable_pension_income →
  indirect_younger_relative_contributions →
  pensioners_inclination_to_save →
  regular_monthly_income →
  preference_for_long_term_deposits →
  (banks_should_offer_higher_deposit_and_lower_loan_rates_to_seniors : Prop) :=
by
  -- Insert proof here that given all the conditions the conclusion follows
  sorry -- proof not required, so skipping

end banks_policies_for_seniors_justified_l87_87599


namespace total_profit_l87_87668

theorem total_profit (A B C : ℕ) (A_invest B_invest C_invest A_share : ℕ) (total_invest total_profit : ℕ)
  (h1 : A_invest = 6300)
  (h2 : B_invest = 4200)
  (h3 : C_invest = 10500)
  (h4 : A_share = 3630)
  (h5 : total_invest = A_invest + B_invest + C_invest)
  (h6 : total_profit * A_share = A_invest * total_invest) :
  total_profit = 12100 :=
by
  sorry

end total_profit_l87_87668


namespace charging_time_l87_87160

theorem charging_time (S T L : ℕ → ℕ) 
  (HS : ∀ t, S t = 15 * t) 
  (HT : ∀ t, T t = 8 * t) 
  (HL : ∀ t, L t = 5 * t)
  (smartphone_capacity tablet_capacity laptop_capacity : ℕ)
  (smartphone_percentage tablet_percentage laptop_percentage : ℕ)
  (h_smartphone : smartphone_capacity = 4500)
  (h_tablet : tablet_capacity = 10000)
  (h_laptop : laptop_capacity = 20000)
  (p_smartphone : smartphone_percentage = 75)
  (p_tablet : tablet_percentage = 25)
  (p_laptop : laptop_percentage = 50)
  (required_charge_s required_charge_t required_charge_l : ℕ)
  (h_rq_s : required_charge_s = smartphone_capacity * smartphone_percentage / 100)
  (h_rq_t : required_charge_t = tablet_capacity * tablet_percentage / 100)
  (h_rq_l : required_charge_l = laptop_capacity * laptop_percentage / 100)
  (time_s time_t time_l : ℕ)
  (h_time_s : time_s = required_charge_s / 15)
  (h_time_t : time_t = required_charge_t / 8)
  (h_time_l : time_l = required_charge_l / 5) : 
  max time_s (max time_t time_l) = 2000 := 
by 
  -- This theorem states that the maximum time taken for charging is 2000 minutes
  sorry

end charging_time_l87_87160


namespace lcm_12_18_l87_87539

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l87_87539


namespace toothpicks_needed_l87_87198

-- Defining the number of rows in the large equilateral triangle.
def rows : ℕ := 10

-- Formula to compute the total number of smaller equilateral triangles.
def total_small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

-- Number of small triangles in this specific case.
def num_small_triangles : ℕ := total_small_triangles rows

-- Total toothpicks without sharing sides.
def total_sides_no_sharing (n : ℕ) : ℕ := 3 * num_small_triangles

-- Adjust for shared toothpicks internally.
def shared_toothpicks (n : ℕ) : ℕ := (total_sides_no_sharing n - 3 * rows) / 2 + 3 * rows

-- Total boundary toothpicks.
def boundary_toothpicks (n : ℕ) : ℕ := 3 * rows

-- Final total number of toothpicks required.
def total_toothpicks (n : ℕ) : ℕ := shared_toothpicks n + boundary_toothpicks n

-- The theorem to be proved
theorem toothpicks_needed : total_toothpicks rows = 98 :=
by
  -- You can complete the proof.
  sorry

end toothpicks_needed_l87_87198


namespace arithmetic_expression_evaluation_l87_87785

theorem arithmetic_expression_evaluation :
  4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end arithmetic_expression_evaluation_l87_87785


namespace circle_eq_center_tangent_l87_87914

theorem circle_eq_center_tangent (x y : ℝ) : 
  let center := (5, 4)
  let radius := 4
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 :=
by
  sorry

end circle_eq_center_tangent_l87_87914


namespace total_cards_l87_87651

theorem total_cards (H F B : ℕ) (hH : H = 200) (hF : F = 4 * H) (hB : B = F - 50) : H + F + B = 1750 := 
by 
  sorry

end total_cards_l87_87651


namespace line_passes_through_fixed_point_l87_87702

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2 * q - 1 = 0) :
  p * (1/2) + 3 * (-1/6) + q = 0 :=
by
  -- placeholders for the actual proof steps
  sorry

end line_passes_through_fixed_point_l87_87702


namespace min_value_x_plus_y_l87_87703

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 :=
by
  sorry

end min_value_x_plus_y_l87_87703


namespace Henry_age_ratio_l87_87941

theorem Henry_age_ratio (A S H : ℕ)
  (hA : A = 15)
  (hS : S = 3 * A)
  (h_sum : A + S + H = 240) :
  H / S = 4 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end Henry_age_ratio_l87_87941


namespace andrey_travel_distance_l87_87676

theorem andrey_travel_distance:
  ∃ s t: ℝ, 
    (s = 60 * (t + 4/3) + 20  ∧ s = 90 * (t - 1/3) + 60) ∧ s = 180 :=
by
  sorry

end andrey_travel_distance_l87_87676


namespace both_firms_participate_social_optimality_l87_87216

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end both_firms_participate_social_optimality_l87_87216


namespace lcm_12_18_l87_87538

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l87_87538


namespace factorize_f_l87_87966

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x

theorem factorize_f (x : ℝ) : f(x) = x * (x - 2)^2 := by
  sorry

end factorize_f_l87_87966


namespace simplify_expression_l87_87563

theorem simplify_expression (x y z : ℝ) (h1 : x ≠ 2) (h2 : y ≠ 3) (h3 : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by 
sorry

end simplify_expression_l87_87563


namespace find_m_if_polynomial_is_perfect_square_l87_87382

theorem find_m_if_polynomial_is_perfect_square (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = x^2 + m * x + 4) → (m = 4 ∨ m = -4) :=
sorry

end find_m_if_polynomial_is_perfect_square_l87_87382


namespace max_product_of_sum_2000_l87_87301

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87301


namespace max_product_of_two_integers_sum_2000_l87_87312

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87312


namespace troy_needs_more_money_to_buy_computer_l87_87778

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l87_87778


namespace rowing_distance_l87_87660
-- Lean 4 Statement

theorem rowing_distance (v_m v_t D : ℝ) 
  (h1 : D = v_m + v_t)
  (h2 : 30 = 10 * (v_m - v_t))
  (h3 : 30 = 6 * (v_m + v_t)) :
  D = 5 :=
by sorry

end rowing_distance_l87_87660


namespace solve_for_x_l87_87612

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l87_87612


namespace mask_usage_duration_l87_87943

-- Define given conditions
def TotalMasks : ℕ := 75
def FamilyMembers : ℕ := 7
def MaskChangeInterval : ℕ := 2

-- Define the goal statement, which is to prove that the family will take 21 days to use all masks
theorem mask_usage_duration 
  (M : ℕ := 75)  -- total masks
  (N : ℕ := 7)   -- family members
  (d : ℕ := 2)   -- mask change interval
  : (M / N) * d + 1 = 21 :=
sorry

end mask_usage_duration_l87_87943


namespace find_k_l87_87560

-- Define vectors a, b, and c
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 2)

-- Define the dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for perpendicular vectors
def perpendicular_condition (k : ℝ) : Prop :=
  dot_product (a.1 - k, -1) b = 0

-- State the theorem
theorem find_k : ∃ k : ℝ, perpendicular_condition k ∧ k = 0 := by
  sorry

end find_k_l87_87560


namespace dog_ate_cost_6_l87_87402

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end dog_ate_cost_6_l87_87402


namespace multiply_then_divide_eq_multiply_l87_87643

theorem multiply_then_divide_eq_multiply (x : ℚ) :
  (x * (2 / 5)) / (3 / 7) = x * (14 / 15) :=
by
  sorry

end multiply_then_divide_eq_multiply_l87_87643


namespace geometric_sequence_exists_l87_87956

theorem geometric_sequence_exists 
  (a r : ℚ)
  (h1 : a = 3)
  (h2 : a * r = 8 / 9)
  (h3 : a * r^2 = 32 / 81) : 
  r = 8 / 27 :=
by
  sorry

end geometric_sequence_exists_l87_87956


namespace final_position_D_l87_87591

open Function

-- Define the original points of the parallelogram
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (9, 4)
def D : ℝ × ℝ := (7, 0)

-- Define the reflection across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the translation by (0, 1)
def translate_up (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)
def translate_down (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

-- Define the reflection across y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combine the transformations to get the final reflection across y = x - 1
def reflect_across_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_down (reflect_y_eq_x (translate_up p))

-- Prove that the final position of D after the two transformations is (1, -8)
theorem final_position_D'' : reflect_across_y_eq_x_minus_1 (reflect_y_axis D) = (1, -8) :=
  sorry

end final_position_D_l87_87591


namespace percentage_increase_l87_87475

theorem percentage_increase (original new : ℝ) (h₁ : original = 50) (h₂ : new = 80) :
  ((new - original) / original) * 100 = 60 :=
by
  sorry

end percentage_increase_l87_87475


namespace total_amount_sold_l87_87163

theorem total_amount_sold (metres_sold : ℕ) (loss_per_metre cost_price_per_metre : ℕ) 
  (h1 : metres_sold = 600) (h2 : loss_per_metre = 5) (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre - loss_per_metre) * metres_sold = 18000 :=
by
  sorry

end total_amount_sold_l87_87163


namespace domain_of_g_cauchy_schwarz_inequality_l87_87701

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Question 1: Prove the domain of g(x) = log(f(x) - 2) is {x | 0.5 < x < 2.5}
theorem domain_of_g : {x : ℝ | 0.5 < x ∧ x < 2.5} = {x : ℝ | 0.5 < x ∧ x < 2.5} :=
by
  sorry

-- Minimum value of f(x)
def m : ℝ := 1

-- Question 2: Prove a^2 + b^2 + c^2 ≥ 1/3 given a + b + c = m
theorem cauchy_schwarz_inequality (a b c : ℝ) (h : a + b + c = m) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end domain_of_g_cauchy_schwarz_inequality_l87_87701


namespace smallest_c1_in_arithmetic_sequence_l87_87248

theorem smallest_c1_in_arithmetic_sequence (S3 S7 : ℕ) (S3_natural : S3 > 0) (S7_natural : S7 > 0)
    (c1_geq_one_third : ∀ d : ℚ, (c1 : ℚ) = (7*S3 - S7) / 14 → c1 ≥ 1/3) : 
    ∃ c1 : ℚ, c1 = 5/14 ∧ c1 ≥ 1/3 := 
by 
  sorry

end smallest_c1_in_arithmetic_sequence_l87_87248


namespace max_product_of_two_integers_sum_2000_l87_87299

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87299


namespace geoff_election_l87_87323

theorem geoff_election (Votes: ℝ) (Percent: ℝ) (ExtraVotes: ℝ) (x: ℝ) 
  (h1 : Votes = 6000) 
  (h2 : Percent = 1) 
  (h3 : ExtraVotes = 3000) 
  (h4 : ReceivedVotes = (Percent / 100) * Votes) 
  (h5 : TotalVotesNeeded = ReceivedVotes + ExtraVotes) 
  (h6 : x = (TotalVotesNeeded / Votes) * 100) :
  x = 51 := 
  by 
    sorry

end geoff_election_l87_87323


namespace line_through_point_parallel_l87_87123

theorem line_through_point_parallel 
    (x y : ℝ)
    (h0 : (x = -1) ∧ (y = 3))
    (h1 : ∃ c : ℝ, (∀ x y : ℝ, x - 2 * y + c = 0 ↔ x - 2 * y + 3 = 0)) :
     ∃ c : ℝ, ∀ x y : ℝ, (x = -1) ∧ (y = 3) → (∃ (a b : ℝ), a - 2 * b + c = 0) :=
by
  sorry

end line_through_point_parallel_l87_87123


namespace ceil_sqrt_196_eq_14_l87_87964

theorem ceil_sqrt_196_eq_14 : ⌈Real.sqrt 196⌉ = 14 := 
by 
  sorry

end ceil_sqrt_196_eq_14_l87_87964


namespace food_insufficiency_l87_87230

-- Given conditions
def number_of_dogs : ℕ := 5
def food_per_meal : ℚ := 3 / 4
def meals_per_day : ℕ := 3
def initial_food : ℚ := 45
def days_in_two_weeks : ℕ := 14

-- Definitions derived from conditions
def daily_food_per_dog : ℚ := food_per_meal * meals_per_day
def daily_food_for_all_dogs : ℚ := daily_food_per_dog * number_of_dogs
def total_food_in_two_weeks : ℚ := daily_food_for_all_dogs * days_in_two_weeks

-- Proof statement: proving the food consumed exceeds the initial amount
theorem food_insufficiency : total_food_in_two_weeks > initial_food :=
by {
  sorry
}

end food_insufficiency_l87_87230


namespace minimum_weight_of_grass_seed_l87_87486

-- Definitions of cost and weights
def price_5_pound_bag : ℝ := 13.85
def price_10_pound_bag : ℝ := 20.43
def price_25_pound_bag : ℝ := 32.20
def max_weight : ℝ := 80
def min_cost : ℝ := 98.68

-- Lean proposition to prove the minimum weight given the conditions
theorem minimum_weight_of_grass_seed (w : ℝ) :
  w = 75 ↔ (w ≤ max_weight ∧
            ∃ (n5 n10 n25 : ℕ), 
              w = 5 * n5 + 10 * n10 + 25 * n25 ∧
              min_cost ≤ n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ∧
              n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ≤ min_cost) := 
by
  sorry

end minimum_weight_of_grass_seed_l87_87486


namespace arithmetic_sequence_problem_l87_87022

-- Define the arithmetic sequence and given properties
variable {a : ℕ → ℝ} -- an arithmetic sequence such that for all n, a_{n+1} - a_{n} is constant
variable (d : ℝ) (a1 : ℝ) -- common difference 'd' and first term 'a1'

-- Express the terms using the common difference 'd' and first term 'a1'
def a_n (n : ℕ) : ℝ := a1 + (n-1) * d

-- Given condition
axiom given_condition : a_n 3 + a_n 8 = 10

-- Proof goal
theorem arithmetic_sequence_problem : 3 * a_n 5 + a_n 7 = 20 :=
by
  -- Define the sequence in terms of common difference and the first term
  let a_n := fun n => a1 + (n-1) * d
  -- Simplify using the given condition
  sorry

end arithmetic_sequence_problem_l87_87022


namespace mixed_oil_rate_l87_87998

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end mixed_oil_rate_l87_87998


namespace polynomial_value_l87_87383

variables {R : Type*} [CommRing R] {x : R}

theorem polynomial_value (h : 2 * x^2 - x = 1) : 
  4 * x^4 - 4 * x^3 + 3 * x^2 - x - 1 = 1 :=
by 
  sorry

end polynomial_value_l87_87383


namespace monotonicity_of_f_on_interval_l87_87354

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x - 2

theorem monotonicity_of_f_on_interval (a b : ℝ) (h1 : a = -3) (h2 : b = 0) :
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 → f x1 a b ≥ f x2 a b := 
by
  sorry

end monotonicity_of_f_on_interval_l87_87354


namespace troy_needs_more_money_to_buy_computer_l87_87779

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l87_87779


namespace max_product_l87_87265

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87265


namespace find_smallest_angle_l87_87849

theorem find_smallest_angle (x : ℝ) (h1 : Real.tan (2 * x) + Real.tan (3 * x) = 1) :
  x = 9 * Real.pi / 180 :=
by
  sorry

end find_smallest_angle_l87_87849


namespace number_of_people_in_room_l87_87257

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end number_of_people_in_room_l87_87257


namespace ratio_tin_copper_in_b_l87_87332

variable (L_a T_a T_b C_b : ℝ)

-- Conditions
axiom h1 : 170 + 250 = 420
axiom h2 : L_a / T_a = 1 / 3
axiom h3 : T_a + T_b = 221.25
axiom h4 : T_a + L_a = 170
axiom h5 : T_b + C_b = 250

-- Target
theorem ratio_tin_copper_in_b (h1 : 170 + 250 = 420) (h2 : L_a / T_a = 1 / 3)
  (h3 : T_a + T_b = 221.25) (h4 : T_a + L_a = 170) (h5 : T_b + C_b = 250) :
  T_b / C_b = 3 / 5 := by
  sorry

end ratio_tin_copper_in_b_l87_87332


namespace average_not_1380_l87_87786

-- Define the set of numbers
def numbers := [1200, 1400, 1510, 1520, 1530, 1200]

-- Define the claimed average
def claimed_avg := 1380

-- The sum of the numbers
def sumNumbers := numbers.sum

-- The number of items in the set
def countNumbers := numbers.length

-- The correct average calculation
def correct_avg : ℚ := sumNumbers / countNumbers

-- The proof problem: proving that the correct average is not equal to the claimed average
theorem average_not_1380 : correct_avg ≠ claimed_avg := by
  sorry

end average_not_1380_l87_87786


namespace min_solution_of_x_abs_x_eq_3x_plus_4_l87_87543

theorem min_solution_of_x_abs_x_eq_3x_plus_4 : 
  ∃ x : ℝ, (x * |x| = 3 * x + 4) ∧ ∀ y : ℝ, (y * |y| = 3 * y + 4) → x ≤ y :=
sorry

end min_solution_of_x_abs_x_eq_3x_plus_4_l87_87543


namespace algebra_problem_l87_87050

theorem algebra_problem 
  (a : ℝ) 
  (h : a^3 + 3 * a^2 + 3 * a + 2 = 0) :
  (a + 1) ^ 2008 + (a + 1) ^ 2009 + (a + 1) ^ 2010 = 1 :=
by 
  sorry

end algebra_problem_l87_87050


namespace largest_n_divisible_l87_87636

theorem largest_n_divisible (n : ℕ) : (n^3 + 150) % (n + 15) = 0 → n ≤ 2385 := by
  sorry

end largest_n_divisible_l87_87636


namespace obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l87_87907

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 3

theorem obtain_1_after_3_operations:
  (operation (operation (operation 1)) = 1) ∨ 
  (operation (operation (operation 8)) = 1) := by
  sorry

theorem obtain_1_after_4_operations:
  (operation (operation (operation (operation 1))) = 1) ∨ 
  (operation (operation (operation (operation 5))) = 1) ∨ 
  (operation (operation (operation (operation 16))) = 1) := by
  sorry

theorem obtain_1_after_5_operations:
  (operation (operation (operation (operation (operation 4)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 10)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 13)))) = 1) := by
  sorry

end obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l87_87907


namespace part_one_binomial_coefficient_part_two_binomial_coefficient_l87_87059

theorem part_one_binomial_coefficient (n : ℕ) :
    (Nat.choose n 4 + Nat.choose n 6 = 2 * Nat.choose n 5) →
    (n = 14) →
    (Nat.choose 14 7 * (1 / 2)^7 * 2^7 = 3432) :=
by
  intros h_arith_seq n_eq
  sorry

theorem part_two_binomial_coefficient :
    (Nat.choose 12 0 + Nat.choose 12 1 + Nat.choose 12 2 = 79) →
    ((1 / 2)^12 * Nat.choose 12 10 * 4^10 = 16896) :=
by
  intros h_sum
  sorry

end part_one_binomial_coefficient_part_two_binomial_coefficient_l87_87059


namespace monomial_sum_l87_87696

variable {x y : ℝ}

theorem monomial_sum (a : ℝ) (h : -2 * x^2 * y^3 + 5 * x^(a-1) * y^3 = c * x^k * y^3) : a = 3 :=
  by
  sorry

end monomial_sum_l87_87696


namespace log_ordering_l87_87742

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := Real.log 10 / Real.log 5

theorem log_ordering : a > b ∧ b > c :=
by {
  sorry
}

end log_ordering_l87_87742


namespace fraction_problem_l87_87950

-- Define the fractions involved in the problem
def frac1 := 18 / 45
def frac2 := 3 / 8
def frac3 := 1 / 9

-- Define the expected result
def expected_result := 49 / 360

-- The proof statement
theorem fraction_problem : frac1 - frac2 + frac3 = expected_result := by
  sorry

end fraction_problem_l87_87950


namespace workers_number_l87_87473

theorem workers_number (W A : ℕ) (h1 : W * 25 = A) (h2 : (W + 10) * 15 = A) : W = 15 :=
by
  sorry

end workers_number_l87_87473


namespace black_haired_girls_count_l87_87586

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l87_87586


namespace reciprocal_of_neg_2023_l87_87003

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87003


namespace parallel_vectors_x_val_l87_87716

open Real

theorem parallel_vectors_x_val (x : ℝ) :
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (x, 1/2)
  a.1 * b.2 = a.2 * b.1 →
  x = 3/8 := 
by
  intro h
  -- Use this line if you need to skip the proof
  sorry

end parallel_vectors_x_val_l87_87716


namespace heather_counts_209_l87_87962

def alice_numbers (n : ℕ) : ℕ := 5 * n - 2
def general_skip_numbers (m : ℕ) : ℕ := 3 * m - 1
def heather_number := 209

theorem heather_counts_209 :
  (∀ n, alice_numbers n > 0 ∧ alice_numbers n ≤ 500 → ¬heather_number = alice_numbers n) ∧
  (∀ m, general_skip_numbers m > 0 ∧ general_skip_numbers m ≤ 500 → ¬heather_number = general_skip_numbers m) ∧
  (1 ≤ heather_number ∧ heather_number ≤ 500) :=
by
  sorry

end heather_counts_209_l87_87962


namespace art_piece_future_value_multiple_l87_87630

theorem art_piece_future_value_multiple (original_price increase_in_value future_value multiple : ℕ)
  (h1 : original_price = 4000)
  (h2 : increase_in_value = 8000)
  (h3 : future_value = original_price + increase_in_value)
  (h4 : multiple = future_value / original_price) :
  multiple = 3 := 
sorry

end art_piece_future_value_multiple_l87_87630


namespace gardener_trees_problem_l87_87933

theorem gardener_trees_problem 
  (maple_trees : ℕ) (oak_trees : ℕ) (birch_trees : ℕ) 
  (total_trees : ℕ) (valid_positions : ℕ) 
  (total_arrangements : ℕ) (probability_numerator : ℕ) (probability_denominator : ℕ) 
  (reduced_numerator : ℕ) (reduced_denominator : ℕ) (m_plus_n : ℕ) :
  (maple_trees = 5) ∧ 
  (oak_trees = 3) ∧ 
  (birch_trees = 7) ∧ 
  (total_trees = 15) ∧ 
  (valid_positions = 8) ∧ 
  (total_arrangements = 120120) ∧ 
  (probability_numerator = 40) ∧ 
  (probability_denominator = total_arrangements) ∧ 
  (reduced_numerator = 1) ∧ 
  (reduced_denominator = 3003) ∧ 
  (m_plus_n = reduced_numerator + reduced_denominator) → 
  m_plus_n = 3004 := 
by
  intros _
  sorry

end gardener_trees_problem_l87_87933


namespace arithmetic_sequence_count_l87_87177

noncomputable def count_arithmetic_triplets : ℕ := 17

theorem arithmetic_sequence_count :
  ∃ S : Finset (Finset ℕ), 
    (∀ s ∈ S, s.card = 3 ∧ (∃ d, ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, (x ≠ y ∧ y ≠ z ∧ x ≠ z) → ((x = y + d ∨ x = z + d ∨ y = z + d) ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9))) ∧ 
    S.card = count_arithmetic_triplets :=
by
  -- placeholder for proof
  sorry

end arithmetic_sequence_count_l87_87177


namespace part1_part2_l87_87729

-- Define the coordinates of point P as functions of n
def pointP (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3 * n)

-- Condition 1: Point P is in the fourth quadrant
def inFourthQuadrant (n : ℝ) : Prop :=
  let point := pointP n
  point.1 > 0 ∧ point.2 < 0

-- Condition 2: Distance from P to the x-axis is 1 greater than the distance to the y-axis
def distancesCondition (n : ℝ) : Prop :=
  abs (2 - 3 * n) + 1 = abs (n + 3)

-- Definition of point Q
def pointQ (n : ℝ) : ℝ × ℝ := (n, -4)

-- Condition 3: PQ is parallel to the x-axis
def pqParallelX (n : ℝ) : Prop :=
  (pointP n).2 = (pointQ n).2

-- Theorems to prove the coordinates of point P and the length of PQ
theorem part1 (n : ℝ) (h1 : inFourthQuadrant n) (h2 : distancesCondition n) : pointP n = (6, -7) :=
sorry

theorem part2 (n : ℝ) (h1 : pqParallelX n) : abs ((pointP n).1 - (pointQ n).1) = 3 :=
sorry

end part1_part2_l87_87729


namespace find_multiple_l87_87362

theorem find_multiple (x k : ℕ) (hx : x > 0) (h_eq : x + 17 = k * (1/x)) (h_x : x = 3) : k = 60 :=
by
  sorry

end find_multiple_l87_87362


namespace number_of_trees_l87_87447

theorem number_of_trees (n : ℕ) (diff : ℕ) (count1 : ℕ) (count2 : ℕ) (timur1 : ℕ) (alexander1 : ℕ) (timur2 : ℕ) (alexander2 : ℕ) : 
  diff = alexander1 - timur1 ∧
  count1 = timur2 + (alexander2 - timur1) ∧
  n = count1 + diff →
  n = 118 :=
by
  sorry

end number_of_trees_l87_87447


namespace largest_alternating_geometric_four_digit_number_l87_87143

theorem largest_alternating_geometric_four_digit_number :
  ∃ (a b c d : ℕ), 
  (9 = 2 * b) ∧ (b = 2 * c) ∧ (a = 3) ∧ (9 * d = b * c) ∧ 
  (a > b) ∧ (b < c) ∧ (c > d) ∧ (1000 * a + 100 * b + 10 * c + d = 9632) := sorry

end largest_alternating_geometric_four_digit_number_l87_87143


namespace problem_bounds_l87_87415

theorem problem_bounds :
  ∀ (A_0 B_0 C_0 A_1 B_1 C_1 A_2 B_2 C_2 A_3 B_3 C_3 : Point),
    (A_0B_0 + B_0C_0 + C_0A_0 = 1) →
    (A_1B_1 = A_0B_0) →
    (B_1C_1 = B_0C_0) →
    (A_2 = A_1 ∧ B_2 = B_1 ∧ C_2 = C_1 ∨
     A_2 = A_1 ∧ B_2 = C_1 ∧ C_2 = B_1 ∨
     A_2 = B_1 ∧ B_2 = A_1 ∧ C_2 = C_1 ∨
     A_2 = B_1 ∧ B_2 = C_1 ∧ C_2 = A_1 ∨
     A_2 = C_1 ∧ B_2 = A_1 ∧ C_2 = B_1 ∨
     A_2 = C_1 ∧ B_2 = B_1 ∧ C_2 = A_1) →
    (A_3B_3 = A_2B_2) →
    (B_3C_3 = B_2C_2) →
    (A_3B_3 + B_3C_3 + C_3A_3) ≥ 1 / 3 ∧ 
    (A_3B_3 + B_3C_3 + C_3A_3) ≤ 3 :=
by
  -- Proof goes here
  sorry

end problem_bounds_l87_87415


namespace smallest_value_of_x_l87_87042

theorem smallest_value_of_x (x : ℝ) (h : |x - 3| = 8) : x = -5 :=
sorry

end smallest_value_of_x_l87_87042


namespace no_integer_roots_of_polynomial_l87_87176

theorem no_integer_roots_of_polynomial :
  ¬ ∃ x : ℤ, x^3 - 4 * x^2 - 14 * x + 28 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l87_87176


namespace find_pencils_l87_87798

theorem find_pencils :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (6 ∣ n) ∧ (9 ∣ n) ∧ n % 7 = 1 ∧ n = 36 :=
by
  sorry

end find_pencils_l87_87798


namespace greatest_product_sum_2000_l87_87276

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87276


namespace prove_a_lt_zero_l87_87984

variable (a b c : ℝ)

-- Define the quadratic function
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions:
-- The polynomial has roots at -2 and 3
def has_roots : Prop := 
  a ≠ 0 ∧ (a * (-2)^2 + b * (-2) + c = 0) ∧ (a * 3^2 + b * 3 + c = 0)

-- f(-b/(2*a)) > 0
def vertex_positive : Prop := 
  f a b c (-b / (2 * a)) > 0

-- Target: Prove a < 0
theorem prove_a_lt_zero 
  (h_roots : has_roots a b c)
  (h_vertex : vertex_positive a b c) : a < 0 := 
sorry

end prove_a_lt_zero_l87_87984


namespace monday_has_greatest_temp_range_l87_87249

-- Define the temperatures
def high_temp (day : String) : Int :=
  if day = "Monday" then 6 else
  if day = "Tuesday" then 3 else
  if day = "Wednesday" then 4 else
  if day = "Thursday" then 4 else
  if day = "Friday" then 8 else 0

def low_temp (day : String) : Int :=
  if day = "Monday" then -4 else
  if day = "Tuesday" then -6 else
  if day = "Wednesday" then -2 else
  if day = "Thursday" then -5 else
  if day = "Friday" then 0 else 0

-- Define the temperature range for a given day
def temp_range (day : String) : Int :=
  high_temp day - low_temp day

-- Statement to prove: Monday has the greatest temperature range
theorem monday_has_greatest_temp_range : 
  temp_range "Monday" > temp_range "Tuesday" ∧
  temp_range "Monday" > temp_range "Wednesday" ∧
  temp_range "Monday" > temp_range "Thursday" ∧
  temp_range "Monday" > temp_range "Friday" := 
sorry

end monday_has_greatest_temp_range_l87_87249


namespace proof_problem_l87_87795

noncomputable def cos75_squared : ℝ :=
  cos 75 * cos 75

noncomputable def optionA : Prop :=
  cos75_squared = (2 - Real.sqrt 3) / 4

noncomputable def optionB : Prop :=
  (1 + tan 105) / (1 - tan 105) ≠ Real.sqrt 3 / 3

noncomputable def optionC : Prop :=
  tan 1 + tan 44 + tan 1 * tan 44 = 1

noncomputable def optionD : Prop :=
  sin 70 * ((Real.sqrt 3) / tan 40 - 1) ≠ 2

theorem proof_problem : optionA ∧ optionC ∧ optionB ∧ optionD := by
  sorry

end proof_problem_l87_87795


namespace find_range_of_k_l87_87865

noncomputable def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k * x

theorem find_range_of_k :
  (∀ x : ℝ, 0 < x → 0 ≤ f x k) → (-1 ≤ k) :=
by
  sorry

end find_range_of_k_l87_87865


namespace career_preference_representation_l87_87916

noncomputable def male_to_female_ratio : ℕ × ℕ := (2, 3)
noncomputable def total_students := male_to_female_ratio.1 + male_to_female_ratio.2
noncomputable def students_prefer_career := 2
noncomputable def full_circle_degrees := 360

theorem career_preference_representation :
  (students_prefer_career / total_students : ℚ) * full_circle_degrees = 144 := by
  sorry

end career_preference_representation_l87_87916


namespace morse_code_symbols_l87_87208

def morse_code_symbols_count : ℕ :=
  let count n := 2^n
  (count 1) + (count 2) + (count 3) + (count 4) + (count 5)

theorem morse_code_symbols :
  morse_code_symbols_count = 62 :=
by
  unfold morse_code_symbols_count
  simp
  sorry

end morse_code_symbols_l87_87208


namespace completing_the_square_l87_87467

theorem completing_the_square (x : ℝ) : x^2 + 2 * x - 5 = 0 → (x + 1)^2 = 6 := by
  intro h
  -- Starting from h and following the steps outlined to complete the square.
  sorry

end completing_the_square_l87_87467


namespace solve_equation_l87_87644

theorem solve_equation (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
    x^2 * (Real.log 27 / Real.log x) * (Real.log x / Real.log 9) = x + 4 → x = 2 :=
by
  sorry

end solve_equation_l87_87644


namespace greatest_product_sum_2000_eq_1000000_l87_87292

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87292


namespace find_g_x2_minus_2_l87_87898

def g : ℝ → ℝ := sorry -- Define g as some real-valued polynomial function.

theorem find_g_x2_minus_2 (x : ℝ) 
(h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1) : 
  g (x^2 - 2) = x^4 - 3 * x^2 - 7 := 
by sorry

end find_g_x2_minus_2_l87_87898


namespace determine_gallons_l87_87777

def current_amount : ℝ := 7.75
def desired_total : ℝ := 14.75
def needed_to_add (x : ℝ) : Prop := desired_total = current_amount + x

theorem determine_gallons : needed_to_add 7 :=
by
  sorry

end determine_gallons_l87_87777


namespace find_abc_values_l87_87958

-- Define the problem conditions as lean definitions
def represents_circle (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x - b * y + c = 0

def circle_center_and_radius_condition (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 3^2

-- Lean 4 statement for the proof problem
theorem find_abc_values (a b c : ℝ) :
  (∀ x y : ℝ, represents_circle a b c x y ↔ circle_center_and_radius_condition x y) →
  a = -2 ∧ b = 6 ∧ c = 4 :=
by
  intro h
  sorry

end find_abc_values_l87_87958


namespace S15_eq_l87_87859

-- Definitions in terms of the geometric sequence and given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions given in the problem
axiom geom_seq (n : ℕ) : S n = (a 0) * (1 - (a 1) ^ n) / (1 - (a 1))
axiom S5_eq : S 5 = 10
axiom S10_eq : S 10 = 50

-- The problem statement to prove
theorem S15_eq : S 15 = 210 :=
by sorry

end S15_eq_l87_87859


namespace cos_arithmetic_sequence_l87_87863

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (h_seq : arithmetic_sequence a) (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
  Real.cos (a 3 + a 7) = -1 / 2 :=
sorry

end cos_arithmetic_sequence_l87_87863


namespace nine_point_five_minutes_in_seconds_l87_87868

-- Define the number of seconds in one minute
def seconds_per_minute : ℝ := 60

-- Define the function to convert minutes to seconds
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * seconds_per_minute

-- Define the theorem to prove
theorem nine_point_five_minutes_in_seconds : minutes_to_seconds 9.5 = 570 :=
by
  sorry

end nine_point_five_minutes_in_seconds_l87_87868


namespace solve_digits_A_B_l87_87839

theorem solve_digits_A_B :
    ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 
    (A * (10 * A + B) = 100 * B + 10 * A + A) ∧ A = 8 ∧ B = 6 :=
by
  sorry

end solve_digits_A_B_l87_87839


namespace lcm_12_18_l87_87535

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l87_87535


namespace slices_left_for_lunch_tomorrow_l87_87519

-- Definitions according to conditions
def initial_slices : ℕ := 12
def slices_eaten_for_lunch := initial_slices / 2
def remaining_slices_after_lunch := initial_slices - slices_eaten_for_lunch
def slices_eaten_for_dinner := 1 / 3 * remaining_slices_after_lunch
def remaining_slices_after_dinner := remaining_slices_after_lunch - slices_eaten_for_dinner
def slices_shared_with_friend := 1 / 4 * remaining_slices_after_dinner
def remaining_slices_after_sharing := remaining_slices_after_dinner - slices_shared_with_friend
def slices_eaten_by_sibling := if (1 / 5 * remaining_slices_after_sharing < 1) then 0 else 1 / 5 * remaining_slices_after_sharing
def remaining_slices_after_sibling := remaining_slices_after_sharing - slices_eaten_by_sibling

-- Lean statement of the proof problem
theorem slices_left_for_lunch_tomorrow : remaining_slices_after_sibling = 3 := by
  sorry

end slices_left_for_lunch_tomorrow_l87_87519


namespace percentage_donated_l87_87818

def income : ℝ := 1200000
def children_percentage : ℝ := 0.20
def wife_percentage : ℝ := 0.30
def remaining : ℝ := income - (children_percentage * 3 * income + wife_percentage * income)
def left_amount : ℝ := 60000
def donated : ℝ := remaining - left_amount

theorem percentage_donated : (donated / remaining) * 100 = 50 := by
  sorry

end percentage_donated_l87_87818


namespace max_working_groups_l87_87445

theorem max_working_groups (teachers groups : ℕ) (memberships_per_teacher group_size : ℕ) 
  (h_teachers : teachers = 36) (h_memberships_per_teacher : memberships_per_teacher = 2)
  (h_group_size : group_size = 4) 
  (h_max_memberships : teachers * memberships_per_teacher = 72) :
  groups ≤ 18 :=
by
  sorry

end max_working_groups_l87_87445


namespace letters_containing_only_dot_l87_87071

theorem letters_containing_only_dot (DS S_only : ℕ) (total : ℕ) (h1 : DS = 20) (h2 : S_only = 36) (h3 : total = 60) :
  total - (DS + S_only) = 4 :=
by
  sorry

end letters_containing_only_dot_l87_87071


namespace total_cookies_l87_87319

theorem total_cookies
  (num_bags : ℕ)
  (cookies_per_bag : ℕ)
  (h_num_bags : num_bags = 286)
  (h_cookies_per_bag : cookies_per_bag = 452) :
  num_bags * cookies_per_bag = 129272 :=
by
  sorry

end total_cookies_l87_87319


namespace dixie_cup_ounces_l87_87316

def gallons_to_ounces (gallons : ℕ) : ℕ := gallons * 128

def initial_water_gallons (gallons : ℕ) : ℕ := gallons_to_ounces gallons

def total_chairs (rows chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

theorem dixie_cup_ounces (initial_gallons rows chairs_per_row water_left : ℕ) 
  (h1 : initial_gallons = 3) 
  (h2 : rows = 5) 
  (h3 : chairs_per_row = 10) 
  (h4 : water_left = 84) 
  (h5 : 128 = 128) : 
  (initial_water_gallons initial_gallons - water_left) / total_chairs rows chairs_per_row = 6 :=
by 
  sorry

end dixie_cup_ounces_l87_87316


namespace supplement_complement_diff_l87_87763

theorem supplement_complement_diff (α : ℝ) : (180 - α) - (90 - α) = 90 := 
by
  sorry

end supplement_complement_diff_l87_87763


namespace abs_neg_2023_eq_2023_l87_87111

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l87_87111


namespace missing_number_is_correct_l87_87510

theorem missing_number_is_correct (mean : ℝ) (observed_numbers : List ℝ) (total_obs : ℕ) (x : ℝ) :
  mean = 14.2 →
  observed_numbers = [8, 13, 21, 7, 23] →
  total_obs = 6 →
  (mean * total_obs = x + observed_numbers.sum) →
  x = 13.2 :=
by
  intros h_mean h_obs h_total h_sum
  sorry

end missing_number_is_correct_l87_87510


namespace total_tiles_144_l87_87231

-- Define the dimensions of the dining room
def diningRoomLength : ℕ := 15
def diningRoomWidth : ℕ := 20

-- Define the border width using 1x1 tiles
def borderWidth : ℕ := 2

-- Area of each 3x3 tile
def tileArea : ℕ := 9

-- Calculate the dimensions of the inner area after the border
def innerAreaLength : ℕ := diningRoomLength - 2 * borderWidth
def innerAreaWidth : ℕ := diningRoomWidth - 2 * borderWidth

-- Calculate the area of the inner region
def innerArea : ℕ := innerAreaLength * innerAreaWidth

-- Calculate the number of 3x3 tiles
def numThreeByThreeTiles : ℕ := (innerArea + tileArea - 1) / tileArea -- rounded up division

-- Calculate the number of 1x1 tiles for the border
def numOneByOneTiles : ℕ :=
  2 * (innerAreaLength + innerAreaWidth + 4 * borderWidth)

-- Total number of tiles
def totalTiles : ℕ := numOneByOneTiles + numThreeByThreeTiles

-- Prove that the total number of tiles is 144
theorem total_tiles_144 : totalTiles = 144 := by
  sorry

end total_tiles_144_l87_87231


namespace geometric_sum_of_ratios_l87_87744

theorem geometric_sum_of_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ) 
  (ha2 : a2 = k * p) (ha3 : a3 = k * p^2) 
  (hb2 : b2 = k * r) (hb3 : b3 = k * r^2) 
  (h : a3 - b3 = 5 * (a2 - b2)) :
  p + r = 5 :=
by {
  sorry
}

end geometric_sum_of_ratios_l87_87744


namespace solve_for_diamond_l87_87380

theorem solve_for_diamond (d : ℕ) (h : d * 5 + 3 = d * 6 + 2) : d = 1 :=
by
  sorry

end solve_for_diamond_l87_87380


namespace Carol_optimal_choice_l87_87166

noncomputable def Alice_choices := Set.Icc 0 (1 : ℝ)
noncomputable def Bob_choices := Set.Icc (1 / 3) (3 / 4 : ℝ)

theorem Carol_optimal_choice : 
  ∀ (c : ℝ), c ∈ Set.Icc 0 1 → 
  (∃! c, c = 7 / 12) := 
sorry

end Carol_optimal_choice_l87_87166


namespace geom_seq_div_a5_a7_l87_87577

variable {a : ℕ → ℝ}

-- Given sequence is geometric and positive
def is_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Positive geometric sequence with decreasing terms
def is_positive_decreasing_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  is_geom_sequence a r ∧ ∀ n, a (n + 1) < a n ∧ a n > 0

-- Conditions
variables (r : ℝ) (hp : is_positive_decreasing_geom_sequence a r)
           (h2 : a 2 * a 8 = 6) (h3 : a 4 + a 6 = 5)

-- Goal
theorem geom_seq_div_a5_a7 : a 5 / a 7 = 3 / 2 :=
by
  sorry

end geom_seq_div_a5_a7_l87_87577


namespace solve_for_x_l87_87610

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l87_87610


namespace latest_start_time_l87_87903

-- Define the weights of the turkeys
def turkey_weights : List ℕ := [16, 18, 20, 22]

-- Define the roasting time per pound
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time in 24-hour format
def dinner_time : ℕ := 18 * 60 -- 18:00 in minutes

-- Calculate the total roasting time
def total_roasting_time (weights : List ℕ) (time_per_pound : ℕ) : ℕ :=
  weights.foldr (λ weight acc => weight * time_per_pound + acc) 0

-- Calculate the latest start time
def latest_roasting_start_time (total_time : ℕ) (dinner_time : ℕ) : ℕ :=
  let start_time := dinner_time - total_time
  if start_time < 0 then start_time + 24 * 60 else start_time

-- Convert minutes to hours:minutes format
def time_in_hours_minutes (time : ℕ) : String :=
  let hours := time / 60
  let minutes := time % 60
  toString hours ++ ":" ++ toString minutes

theorem latest_start_time : 
  time_in_hours_minutes (latest_roasting_start_time (total_roasting_time turkey_weights roasting_time_per_pound) dinner_time) = "23:00" := by
  sorry

end latest_start_time_l87_87903


namespace largest_n_in_base10_l87_87618

-- Definitions corresponding to the problem conditions
def n_eq_base8_expr (A B C : ℕ) : ℕ := 64 * A + 8 * B + C
def n_eq_base12_expr (A B C : ℕ) : ℕ := 144 * C + 12 * B + A

-- Problem statement translated into Lean
theorem largest_n_in_base10 (n A B C : ℕ) (h1 : n = n_eq_base8_expr A B C) 
    (h2 : n = n_eq_base12_expr A B C) (hA : A < 8) (hB : B < 8) (hC : C < 12) (h_pos: n > 0) : 
    n ≤ 509 :=
sorry

end largest_n_in_base10_l87_87618


namespace find_a12_l87_87053

namespace ArithmeticSequence

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a n = a 0 + n * d

theorem find_a12 {a : ℕ → α} (h1 : a 4 = 1) (h2 : a 7 + a 9 = 16) :
  a 12 = 15 := 
sorry

end ArithmeticSequence

end find_a12_l87_87053


namespace union_complement_l87_87893

-- Definitions of the sets
def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 4}

-- Statement of the proof problem
theorem union_complement : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end union_complement_l87_87893


namespace hens_count_l87_87149

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 :=
by
  sorry

end hens_count_l87_87149


namespace reciprocal_of_neg_2023_l87_87000

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l87_87000


namespace binary_multiplication_l87_87947

theorem binary_multiplication :
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  a * b = product :=
by 
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  sorry

end binary_multiplication_l87_87947


namespace find_a_l87_87233

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (x - a)^2 + y^2 = (x^2 + (y-1)^2)) ∧ (¬ ∃ x y : ℝ, y = x + 1) → a = 1 :=
by
  sorry

end find_a_l87_87233


namespace min_value_fraction_l87_87369

variable (a b : ℝ)

theorem min_value_fraction (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 2 * b = 1) : 
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_l87_87369


namespace increasing_iff_a_le_0_l87_87743

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - a * x + 1

theorem increasing_iff_a_le_0 : (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
by
  sorry

end increasing_iff_a_le_0_l87_87743


namespace sum_of_a_and_b_l87_87329

def otimes (x y : ℝ) : ℝ := x * (1 - y)

variable (a b : ℝ)

theorem sum_of_a_and_b :
  ({ x : ℝ | (x - a) * (1 - (x - b)) > 0 } = { x : ℝ | 2 < x ∧ x < 3 }) →
  a + b = 4 :=
by
  intro h
  have h_eq : ∀ x, (x - a) * ((1 : ℝ) - (x - b)) = (x - a) * (x - (b + 1)) := sorry
  have h_ineq : ∀ x, (x - a) * (x - (b + 1)) > 0 ↔ 2 < x ∧ x < 3 := sorry
  have h_set_eq : { x | (x - a) * ((1 : ℝ) - (x - b)) > 0 } = { x | 2 < x ∧ x < 3 } := sorry
  have h_roots_2_3 : (2 - a) * (2 - (b + 1)) = 0 ∧ (3 - a) * (3 - (b + 1)) = 0 := sorry
  have h_2_eq : 2 - a = 0 ∨ 2 - (b + 1) = 0 := sorry
  have h_3_eq : 3 - a = 0 ∨ 3 - (b + 1) = 0 := sorry
  have h_a_2 : a = 2 ∨ b + 1 = 2 := sorry
  have h_b_2 : b = 2 - 1 := sorry
  have h_a_3 : a = 3 ∨ b + 1 = 3 := sorry
  have h_b_3 : b = 3 - 1 := sorry
  sorry

end sum_of_a_and_b_l87_87329


namespace factor_expression_l87_87125

theorem factor_expression (a b : ℕ) (h_factor : (x - a) * (x - b) = x^2 - 18 * x + 72) (h_nonneg : 0 ≤ a ∧ 0 ≤ b) (h_order : a > b) : 4 * b - a = 27 := by
  sorry

end factor_expression_l87_87125


namespace tom_dimes_now_l87_87448

-- Define the initial number of dimes and the number of dimes given by dad
def initial_dimes : ℕ := 15
def dimes_given_by_dad : ℕ := 33

-- Define the final count of dimes Tom has now
def final_dimes (initial_dimes dimes_given_by_dad : ℕ) : ℕ :=
  initial_dimes + dimes_given_by_dad

-- The main theorem to prove "how many dimes Tom has now"
theorem tom_dimes_now : initial_dimes + dimes_given_by_dad = 48 :=
by
  -- The proof can be skipped using sorry
  sorry

end tom_dimes_now_l87_87448


namespace max_product_two_integers_sum_2000_l87_87285

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87285


namespace equal_center_intersection_l87_87404

/-- Given a finite group G, and subgroups H1 and H2, if for any representation of G on
a finite-dimensional complex vector space V, the dimensions of the H1-invariant
and H2-invariant subspaces are equal, then Z(G) ∩ H1 = Z(G) ∩ H2, where Z(G) is the center of G. -/
theorem equal_center_intersection {G : Type*} [Group G] [Fintype G] [FiniteGroup G]
  (H1 H2 : Subgroup G)
  (h : ∀ (V : Type*) [AddCommGroup V] [Module ℂ V] [FiniteDimensional ℂ V],
    ∀ (ρ : Representation ℂ G V),
      FiniteDimensional.finrank ℂ (ρ.fixedPoints H1) =
      FiniteDimensional.finrank ℂ (ρ.fixedPoints H2)) :
  (Subgroup.center G) ⊓ H1 = (Subgroup.center G) ⊓ H2 :=
by sorry

end equal_center_intersection_l87_87404


namespace cone_volume_l87_87158

theorem cone_volume (V_f : ℝ) (A1 A2 : ℝ) (V : ℝ)
  (h1 : V_f = 78)
  (h2 : A1 = 9 * A2) :
  V = 81 :=
sorry

end cone_volume_l87_87158


namespace range_of_a_for_intersections_l87_87376

theorem range_of_a_for_intersections (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (x₁^3 - 3 * x₁ = a) ∧ (x₂^3 - 3 * x₂ = a) ∧ (x₃^3 - 3 * x₃ = a)) ↔ 
  (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_for_intersections_l87_87376


namespace abs_neg_2023_l87_87108

-- Define the absolute value function following the provided condition
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- State the theorem we need to prove
theorem abs_neg_2023 : abs (-2023) = 2023 :=
by
  -- Placeholder for proof
  sorry

end abs_neg_2023_l87_87108


namespace total_amount_invested_l87_87385

theorem total_amount_invested (x y total : ℝ) (h1 : 0.10 * x - 0.08 * y = 83) (h2 : y = 650) : total = 2000 :=
sorry

end total_amount_invested_l87_87385


namespace finance_to_manufacturing_ratio_l87_87229

theorem finance_to_manufacturing_ratio : 
    let finance_angle := 72
    let manufacturing_angle := 108
    (finance_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 2 ∧ 
    (manufacturing_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 3 := 
by 
    sorry

end finance_to_manufacturing_ratio_l87_87229


namespace original_price_l87_87815

theorem original_price (sale_price gain_percent : ℕ) (h_sale : sale_price = 130) (h_gain : gain_percent = 30) : 
    ∃ P : ℕ, (P * (1 + gain_percent / 100)) = sale_price := 
by
  use 100
  rw [h_sale, h_gain]
  norm_num
  sorry

end original_price_l87_87815


namespace min_value_expression_l87_87090

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 := by
  sorry

end min_value_expression_l87_87090


namespace clay_capacity_second_box_l87_87333

-- Define the dimensions and clay capacity of the first box
def height1 : ℕ := 4
def width1 : ℕ := 2
def length1 : ℕ := 3
def clay1 : ℕ := 24

-- Define the dimensions of the second box
def height2 : ℕ := 3 * height1
def width2 : ℕ := 2 * width1
def length2 : ℕ := length1

-- The volume relation
def volume_relation (height width length clay: ℕ) : ℕ :=
  height * width * length * clay

theorem clay_capacity_second_box (height1 width1 length1 clay1 : ℕ) (height2 width2 length2 : ℕ) :
  height1 = 4 →
  width1 = 2 →
  length1 = 3 →
  clay1 = 24 →
  height2 = 3 * height1 →
  width2 = 2 * width1 →
  length2 = length1 →
  volume_relation height2 width2 length2 1 = 6 * volume_relation height1 width1 length1 1 →
  volume_relation height2 width2 length2 clay1 / volume_relation height1 width1 length1 1 = 144 :=
by
  intros h1 w1 l1 c1 h2 w2 l2 vol_rel
  sorry

end clay_capacity_second_box_l87_87333


namespace troy_needs_additional_money_l87_87781

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l87_87781


namespace rafael_weekly_earnings_l87_87236

def weekly_hours (m t r : ℕ) : ℕ := m + t + r
def total_earnings (hours wage : ℕ) : ℕ := hours * wage

theorem rafael_weekly_earnings : 
  ∀ (m t r wage : ℕ), m = 10 → t = 8 → r = 20 → wage = 20 → total_earnings (weekly_hours m t r) wage = 760 :=
by
  intros m t r wage hm ht hr hw
  rw [hm, ht, hr, hw]
  simp only [weekly_hours, total_earnings]
  sorry -- Proof needs to be completed

end rafael_weekly_earnings_l87_87236


namespace abs_neg_2023_l87_87109

-- Define the absolute value function following the provided condition
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- State the theorem we need to prove
theorem abs_neg_2023 : abs (-2023) = 2023 :=
by
  -- Placeholder for proof
  sorry

end abs_neg_2023_l87_87109


namespace perpendicular_line_sum_l87_87057

theorem perpendicular_line_sum (a b c : ℝ) (h1 : a + 4 * c - 2 = 0) (h2 : 2 - 5 * c + b = 0) 
  (perpendicular : (a / -4) * (2 / 5) = -1) : a + b + c = -4 := 
sorry

end perpendicular_line_sum_l87_87057


namespace largest_square_side_length_largest_rectangle_dimensions_l87_87052

variable (a b : ℝ)

-- Part a
theorem largest_square_side_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ s : ℝ, s = (a * b) / (a + b) :=
sorry

-- Part b
theorem largest_rectangle_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, (x = a / 2 ∧ y = b / 2) :=
sorry

end largest_square_side_length_largest_rectangle_dimensions_l87_87052


namespace find_f2_l87_87060

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
sorry

end find_f2_l87_87060


namespace abs_neg_number_l87_87117

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end abs_neg_number_l87_87117


namespace missing_number_eq_6_l87_87330

theorem missing_number_eq_6 (x : ℕ) : (x! - 4!) / 5! = 58 / 10 → x = 6 := by
  have four_factorial : 4! = 24 := by norm_num
  have five_factorial : 5! = 120 := by norm_num
  rw [four_factorial, five_factorial]
  sorry

end missing_number_eq_6_l87_87330


namespace percentage_sophia_ate_l87_87508

theorem percentage_sophia_ate : 
  ∀ (caden zoe noah sophia : ℝ),
    caden = 20 / 100 →
    zoe = caden + (0.5 * caden) →
    noah = zoe + (0.5 * zoe) →
    caden + zoe + noah + sophia = 1 →
    sophia = 5 / 100 :=
by
  intros
  sorry

end percentage_sophia_ate_l87_87508


namespace prove_x_ge_neg_one_sixth_l87_87055

variable (x y : ℝ)

theorem prove_x_ge_neg_one_sixth (h : x^4 * y^2 + y^4 + 2 * x^3 * y + 6 * x^2 * y + x^2 + 8 ≤ 0) :
  x ≥ -1 / 6 :=
sorry

end prove_x_ge_neg_one_sixth_l87_87055


namespace max_product_two_integers_sum_2000_l87_87282

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87282


namespace lcm_12_18_l87_87531

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l87_87531


namespace greatest_product_sum_2000_eq_1000000_l87_87289

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87289


namespace arithmetic_sequence_nth_term_l87_87695

theorem arithmetic_sequence_nth_term (S : ℕ → ℕ) (h : ∀ n, S n = 5 * n + 4 * n^2) (r : ℕ) : 
  S r - S (r - 1) = 8 * r + 1 := 
by
  sorry

end arithmetic_sequence_nth_term_l87_87695


namespace amy_l87_87497

theorem amy's_speed (a b : ℝ) (s : ℝ) 
  (h1 : ∀ (major minor : ℝ), major = 2 * minor) 
  (h2 : ∀ (w : ℝ), w = 4) 
  (h3 : ∀ (t_diff : ℝ), t_diff = 48) 
  (h4 : 2 * a + 2 * Real.pi * Real.sqrt ((4 * b^2 + b^2) / 2) - (2 * a + 2 * Real.pi * Real.sqrt (((2 * b + 8)^2 + (b + 4)^2) / 2)) = 48 * s) :
  s = Real.pi / 2 := sorry

end amy_l87_87497


namespace find_certain_number_l87_87869

theorem find_certain_number (x : ℝ) (h : ((x^4) * 3.456789)^10 = 10^20) : x = 10 :=
sorry

end find_certain_number_l87_87869


namespace angle_BPC_measure_l87_87079

noncomputable def angle_BPC (AB BE BA sq_len PQ_len : ℝ) 
(BAE ABE ABC ACME : ℝ) : ℝ :=
  let α := BAE
  let β := (180 - (2 * ABE + α)) / 2
  let γ := ABC - β
  180 - γ - ACM


theorem angle_BPC_measure :
  -- Conditions
  let AB := 6
  let BE := 6
  let BA := 6
  let sq_len := 6
  let PQ_len := x
  let BAE := 45
  let ABE := 67.5
  let ABC := 90
  let ACM := 45
  let BP := [(AB + √2 * AB div 2 )/ 2 ]
  
  AB = AB -> BE = 6 -> BE = AB -> ABC = 90 -> 
  -- Conclusion
  angle_BPC AB BE BAE sq_len PQ_len BAE ABE ABC ACM = 112.5 :=
by
  sorry

end angle_BPC_measure_l87_87079


namespace line_slope_intercept_through_points_l87_87572

theorem line_slope_intercept_through_points (a b : ℝ) :
  (∀ x y : ℝ, (x, y) = (3, 7) ∨ (x, y) = (7, 19) → y = a * x + b) →
  a - b = 5 :=
by
  sorry

end line_slope_intercept_through_points_l87_87572


namespace probability_two_chinese_knights_attack_l87_87414

-- Define the chess board and knight attack rules
def china_knight (i j : Nat) (board : Matrix Nat Nat Bool) : Bool :=
  -- Implement the attack condition for a Chinese knight
  sorry

-- Number of favourable outcomes
noncomputable def favorable_outcomes : ℚ := (79 : ℚ) / 256
  
-- Probabilistic model of the board
def board_model : ProbabilityModel (Matrix Nat Nat Bool) := sorry

-- Theorem asserting the correct probability under specified conditions
theorem probability_two_chinese_knights_attack :
  Probability board_model (λ board, ∃ i j, china_knight i j board) = favorable_outcomes :=
sorry

end probability_two_chinese_knights_attack_l87_87414


namespace transportation_degrees_correct_l87_87655

-- Define the percentages for the different categories.
def salaries_percent := 0.60
def research_development_percent := 0.09
def utilities_percent := 0.05
def equipment_percent := 0.04
def supplies_percent := 0.02

-- Define the total percentage of non-transportation categories.
def non_transportation_percent := 
  salaries_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent

-- Define the full circle in degrees.
def full_circle_degrees := 360.0

-- Total percentage which must sum to 1 (i.e., 100%).
def total_budget_percent := 1.0

-- Calculate the percentage for transportation.
def transportation_percent := total_budget_percent - non_transportation_percent

-- Define the result for degrees allocated to transportation.
def transportation_degrees := transportation_percent * full_circle_degrees

-- Prove that the transportation degrees are 72.
theorem transportation_degrees_correct : transportation_degrees = 72.0 :=
by
  unfold transportation_degrees transportation_percent non_transportation_percent
  sorry

end transportation_degrees_correct_l87_87655


namespace ad_plus_bc_eq_pm_one_l87_87021

theorem ad_plus_bc_eq_pm_one
  (a b c d : ℤ)
  (h1 : ∃ n : ℤ, n = ad + bc ∧ n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d) :
  ad + bc = 1 ∨ ad + bc = -1 := 
sorry

end ad_plus_bc_eq_pm_one_l87_87021


namespace parabola_relative_positions_l87_87355

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 3
def parabola3 (x : ℝ) : ℝ := x^2 + 2*x + 3

noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

theorem parabola_relative_positions :
  vertex_x 1 (-1) 3 < vertex_x 1 1 3 ∧ vertex_x 1 1 3 < vertex_x 1 2 3 :=
by {
  sorry
}

end parabola_relative_positions_l87_87355


namespace binary_multiplication_l87_87948

theorem binary_multiplication : 
  (nat.bin_to_num [1, 1, 0, 1] * nat.bin_to_num [1, 1, 1] = nat.bin_to_num [1, 1, 0, 0, 1, 1, 1]) :=
by
  sorry

end binary_multiplication_l87_87948


namespace average_speed_of_rocket_l87_87936

def distance_soared (speed_soaring : ℕ) (time_soaring : ℕ) : ℕ :=
  speed_soaring * time_soaring

def distance_plummeted : ℕ := 600

def total_distance (distance_soared : ℕ) (distance_plummeted : ℕ) : ℕ :=
  distance_soared + distance_plummeted

def total_time (time_soaring : ℕ) (time_plummeting : ℕ) : ℕ :=
  time_soaring + time_plummeting

def average_speed (total_distance : ℕ) (total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_rocket :
  let speed_soaring := 150
  let time_soaring := 12
  let time_plummeting := 3
  distance_soared speed_soaring time_soaring +
  distance_plummeted = 2400
  →
  total_time time_soaring time_plummeting = 15
  →
  average_speed (distance_soared speed_soaring time_soaring + distance_plummeted)
                (total_time time_soaring time_plummeting) = 160 :=
by
  sorry

end average_speed_of_rocket_l87_87936


namespace intersection_equal_l87_87989

noncomputable def M := { y : ℝ | ∃ x : ℝ, y = Real.log (x + 1) / Real.log (1 / 2) ∧ x ≥ 3 }
noncomputable def N := { x : ℝ | x^2 + 2 * x - 3 ≤ 0 }

theorem intersection_equal : M ∩ N = {a : ℝ | -3 ≤ a ∧ a ≤ -2} :=
by
  sorry

end intersection_equal_l87_87989


namespace females_who_chose_malt_l87_87803

-- Definitions
def total_cheerleaders : ℕ := 26
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_who_chose_malt : ℕ := 6

-- Main statement
theorem females_who_chose_malt (C M F : ℕ) (hM : M = 2 * C) (h_total : C + M = total_cheerleaders) (h_males_malt : males_who_chose_malt = total_males) : F = 10 :=
sorry

end females_who_chose_malt_l87_87803


namespace units_digit_47_pow_47_l87_87461

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l87_87461


namespace apple_juice_cost_l87_87784

noncomputable def cost_of_apple_juice (cost_per_orange_juice : ℝ) (total_bottles : ℕ) (total_cost : ℝ) (orange_juice_bottles : ℕ) : ℝ :=
  (total_cost - cost_per_orange_juice * orange_juice_bottles) / (total_bottles - orange_juice_bottles)

theorem apple_juice_cost :
  let cost_per_orange_juice := 0.7
  let total_bottles := 70
  let total_cost := 46.2
  let orange_juice_bottles := 42
  cost_of_apple_juice cost_per_orange_juice total_bottles total_cost orange_juice_bottles = 0.6 := by
    sorry

end apple_juice_cost_l87_87784


namespace solve_for_x_l87_87605

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l87_87605


namespace reciprocal_of_neg_2023_l87_87004

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87004


namespace find_g2_l87_87621

-- Define the conditions of the problem
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x

-- Prove the desired value of g(2)
theorem find_g2 (g : ℝ → ℝ) (h : satisfies_condition g) : g 2 = -19 / 6 :=
by
  sorry

end find_g2_l87_87621


namespace exactly_one_divisible_by_4_l87_87595

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end exactly_one_divisible_by_4_l87_87595


namespace find_tan_half_sum_of_angles_l87_87897

theorem find_tan_half_sum_of_angles (x y : ℝ) 
  (h₁ : Real.cos x + Real.cos y = 1)
  (h₂ : Real.sin x + Real.sin y = 1 / 2) : 
  Real.tan ((x + y) / 2) = 1 / 2 := 
by 
  sorry

end find_tan_half_sum_of_angles_l87_87897


namespace find_coefficient_of_x_in_expansion_l87_87121

noncomputable def coefficient_of_x_in_expansion (x : ℤ) : ℤ :=
  (1 / 2 * x - 1) * (2 * x - 1 / x) ^ 6

theorem find_coefficient_of_x_in_expansion :
  coefficient_of_x_in_expansion x = -80 :=
by {
  sorry
}

end find_coefficient_of_x_in_expansion_l87_87121


namespace packages_per_box_l87_87399

theorem packages_per_box (P : ℕ) 
  (h1 : 100 * 25 = 2500) 
  (h2 : 2 * P * 250 = 2500) : 
  P = 5 := 
sorry

end packages_per_box_l87_87399


namespace intersection_of_A_and_B_l87_87558

-- Define sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x < 1}

-- Statement of the proof problem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} := by
  sorry -- The proof goes here

end intersection_of_A_and_B_l87_87558


namespace solve_for_y_l87_87567

theorem solve_for_y (x y : ℝ) (h₁ : x - y = 16) (h₂ : x + y = 4) : y = -6 := 
by 
  sorry

end solve_for_y_l87_87567


namespace width_of_metallic_sheet_is_36_l87_87816

-- Given conditions
def length_of_metallic_sheet : ℕ := 48
def side_length_of_cutoff_square : ℕ := 8
def volume_of_box : ℕ := 5120

-- Proof statement
theorem width_of_metallic_sheet_is_36 :
  ∀ (w : ℕ), w - 2 * side_length_of_cutoff_square = 36 - 16 →  length_of_metallic_sheet - 2* side_length_of_cutoff_square = 32  →  5120 = 256 * (w - 16)  := sorry

end width_of_metallic_sheet_is_36_l87_87816


namespace measure_of_angle_B_scalene_triangle_l87_87260

theorem measure_of_angle_B_scalene_triangle (A B C : ℝ) (hA_gt_0 : A > 0) (hB_gt_0 : B > 0) (hC_gt_0 : C > 0) 
(h_angles_sum : A + B + C = 180) (hB_eq_2A : B = 2 * A) (hC_eq_3A : C = 3 * A) : B = 60 :=
by
  sorry

end measure_of_angle_B_scalene_triangle_l87_87260


namespace point_on_graph_l87_87769

theorem point_on_graph (x y : ℝ) (h : y = 3 * x + 1) : (x, y) = (2, 7) :=
sorry

end point_on_graph_l87_87769


namespace city_a_location_l87_87078

theorem city_a_location (ϕ A_latitude : ℝ) (m : ℝ) (h_eq_height : true)
  (h_shadows_3x : true) 
  (h_angle: true) (h_southern : A_latitude < 0) 
  (h_rad_lat : ϕ = abs A_latitude):

  ϕ = 45 ∨ ϕ = 7.14 :=
by 
  sorry

end city_a_location_l87_87078


namespace arrangement_count_l87_87073

open Nat

/-- Define the four subjects. -/
inductive Subject
| Chinese
| Mathematics
| PE
| English

/-- Prove there are 12 valid arrangements of the subjects given PE is neither first nor last. -/
theorem arrangement_count : let subjects := Finset.univ : Finset Subject,
                               positions := Finset.range 4 in
                             ∑ pe_pos in positions\{0, 3},
                             (∏ remaining_slots in Finset.erase positions pe_pos, remaining_slots ) = 12 :=
sorry

end arrangement_count_l87_87073


namespace geometric_series_S_n_div_a_n_l87_87708

-- Define the conditions and the properties of the geometric sequence
variables (a_3 a_5 a_4 a_6 S_n a_n : ℝ) (n : ℕ)
variable (q : ℝ) -- common ratio of the geometric sequence

-- Conditions given in the problem
axiom h1 : a_3 + a_5 = 5 / 4
axiom h2 : a_4 + a_6 = 5 / 8

-- The value we want to prove
theorem geometric_series_S_n_div_a_n : 
  (a_3 + a_5) * q = 5 / 8 → 
  q = 1 / 2 → 
  S_n = a_n * (2^n - 1) :=
by
  intros h1 h2
  sorry

end geometric_series_S_n_div_a_n_l87_87708


namespace calculate_f_at_2_l87_87350

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_at_2 :
  (∀ x : ℝ, 25 * f (x / 1580) + (3 - Real.sqrt 34) * f (1580 / x) = 2017 * x) →
  f 2 = 265572 :=
by
  intro h
  sorry

end calculate_f_at_2_l87_87350


namespace radius_of_circle_with_tangent_parabolas_l87_87686

theorem radius_of_circle_with_tangent_parabolas (r : ℝ) : 
  (∀ x : ℝ, (x^2 + r = x → ∃ x0 : ℝ, x^2 + r = x0)) → r = 1 / 4 :=
by
  sorry

end radius_of_circle_with_tangent_parabolas_l87_87686


namespace candyStoreSpending_l87_87197

-- Definitions based on conditions provided
def weeklyAllowance : ℚ := 345 / 100   -- John's weekly allowance is $3.45
def arcadeFraction : ℚ := 3 / 5        -- John spent 3/5 of his allowance at the arcade
def toyStoreFraction : ℚ := 1 / 3      -- John spent 1/3 of his remaining allowance at the toy store

-- Main theorem to prove
theorem candyStoreSpending :
  let arcadeSpending := arcadeFraction * weeklyAllowance
  let remainingAfterArcade := weeklyAllowance - arcadeSpending
  let toyStoreSpending := toyStoreFraction * remainingAfterArcade
  let remainingAfterToyStore := remainingAfterArcade - toyStoreSpending
  remainingAfterToyStore = 92 / 100 := 
by
  sorry

end candyStoreSpending_l87_87197


namespace find_pairs_of_positive_integers_l87_87680

theorem find_pairs_of_positive_integers (x y : ℕ) (h : x > 0 ∧ y > 0) (h_eq : x + y + x * y = 2006) :
  (x, y) = (2, 668) ∨ (x, y) = (668, 2) ∨ (x, y) = (8, 222) ∨ (x, y) = (222, 8) :=
sorry

end find_pairs_of_positive_integers_l87_87680


namespace plane_fuel_consumption_rate_l87_87821

/-- A plane has 6.3333 gallons of fuel left and can continue to fly for 0.6667 hours.
    Prove that the rate of fuel consumption per hour is approximately 9.5 gallons per hour. -/
theorem plane_fuel_consumption_rate :
  let fuel_left := 6.3333
  let time_left_to_fly := 0.6667
  let rate_of_fuel_consumption_per_hour := fuel_left / time_left_to_fly
  abs (rate_of_fuel_consumption_per_hour - 9.5) < 0.01 :=
by {
  let fuel_left := 6.3333
  let time_left_to_fly := 0.6667
  let rate_of_fuel_consumption_per_hour := fuel_left / time_left_to_fly
  show abs (rate_of_fuel_consumption_per_hour - 9.5) < 0.01,
  apply sorry
}

end plane_fuel_consumption_rate_l87_87821


namespace inequality_holds_for_m_l87_87694

theorem inequality_holds_for_m (m : ℝ) :
  (-2 : ℝ) ≤ m ∧ m ≤ (3 : ℝ) ↔ ∀ x : ℝ, x < -1 →
    (m - m^2) * (4 : ℝ)^x + (2 : ℝ)^x + 1 > 0 :=
by sorry

end inequality_holds_for_m_l87_87694


namespace total_time_eight_runners_l87_87074

theorem total_time_eight_runners :
  (let t₁ := 8 -- time for the first five runners
       t₂ := t₁ + 2 -- time for the remaining three runners
       n₁ := 5 -- number of first runners
       n₂ := 3 -- number of remaining runners
   in n₁ * t₁ + n₂ * t₂ = 70) :=
by
  sorry

end total_time_eight_runners_l87_87074


namespace proof_g_2_l87_87091

def g (x : ℝ) : ℝ := 3 * x ^ 8 - 4 * x ^ 4 + 2 * x ^ 2 - 6

theorem proof_g_2 :
  g (-2) = 10 → g (2) = 1402 := by
  sorry

end proof_g_2_l87_87091


namespace fraction_meaningful_iff_x_ne_pm1_l87_87544

theorem fraction_meaningful_iff_x_ne_pm1 (x : ℝ) : (x^2 - 1 ≠ 0) ↔ (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end fraction_meaningful_iff_x_ne_pm1_l87_87544


namespace Toms_walking_speed_l87_87631

theorem Toms_walking_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (run_distance : ℝ)
  (run_speed : ℝ)
  (walk_distance : ℝ)
  (walk_time : ℝ)
  (walk_speed : ℝ)
  (h1 : total_distance = 1800)
  (h2 : total_time ≤ 20)
  (h3 : run_distance = 600)
  (h4 : run_speed = 210)
  (h5 : total_distance = run_distance + walk_distance)
  (h6 : total_time = walk_time + run_distance / run_speed)
  (h7 : walk_speed = walk_distance / walk_time) :
  walk_speed ≤ 70 := sorry

end Toms_walking_speed_l87_87631


namespace solve_for_x_l87_87614

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l87_87614


namespace units_digit_of_expression_l87_87046

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_expression : units_digit (7 * 18 * 1978 - 7^4) = 7 := by
  sorry

end units_digit_of_expression_l87_87046


namespace quadratic_equivalence_statement_l87_87867

noncomputable def quadratic_in_cos (a b c x : ℝ) : Prop := 
  a * (Real.cos x)^2 + b * Real.cos x + c = 0

noncomputable def transform_to_cos2x (a b c : ℝ) : Prop := 
  (4*a^2) * (Real.cos (2*a))^2 + (2*a^2 + 4*a*c - 2*b^2) * Real.cos (2*a) + a^2 + 4*a*c - 2*b^2 + 4*c^2 = 0

theorem quadratic_equivalence_statement (a b c : ℝ) (h : quadratic_in_cos 4 2 (-1) a) :
  transform_to_cos2x 16 12 (-4) :=
sorry

end quadratic_equivalence_statement_l87_87867


namespace element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l87_87628

-- defining element, nuclide, and valence based on protons, neutrons, and electrons
def Element (protons : ℕ) := protons
def Nuclide (protons neutrons : ℕ) := (protons, neutrons)
def ChemicalProperties (outermostElectrons : ℕ) := outermostElectrons
def HighestPositiveValence (mainGroupNum : ℕ) := mainGroupNum

-- The proof problems as Lean theorems
theorem element_type_determined_by_protons (protons : ℕ) :
  Element protons = protons := sorry

theorem nuclide_type_determined_by_protons_neutrons (protons neutrons : ℕ) :
  Nuclide protons neutrons = (protons, neutrons) := sorry

theorem chemical_properties_determined_by_outermost_electrons (outermostElectrons : ℕ) :
  ChemicalProperties outermostElectrons = outermostElectrons := sorry
  
theorem highest_positive_valence_determined_by_main_group_num (mainGroupNum : ℕ) :
  HighestPositiveValence mainGroupNum = mainGroupNum := sorry

end element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l87_87628


namespace rectangle_area_is_432_l87_87930

-- Definition of conditions and problem in Lean 4
noncomputable def circle_radius : ℝ := 6
noncomputable def rectangle_ratio_length_width : ℝ := 3 / 1
noncomputable def calculate_rectangle_area (radius : ℝ) (ratio : ℝ) : ℝ :=
  let diameter := 2 * radius
  let width := diameter
  let length := ratio * width
  length * width

-- Lean statement to prove the area
theorem rectangle_area_is_432 : calculate_rectangle_area circle_radius rectangle_ratio_length_width = 432 := by
  sorry

end rectangle_area_is_432_l87_87930


namespace intersecting_lines_l87_87018

theorem intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ (x = 0 ∨ y = 0) := by
  sorry

end intersecting_lines_l87_87018


namespace cos_C_in_triangle_l87_87394

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end cos_C_in_triangle_l87_87394


namespace problem1_problem2_l87_87039

-- Problem 1
theorem problem1 : 2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2 :=
by sorry

-- Problem 2
theorem problem2 : (Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6 :=
by sorry

end problem1_problem2_l87_87039


namespace max_value_of_function_for_x_lt_0_l87_87840

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem max_value_of_function_for_x_lt_0 :
  ∀ x : ℝ, x < 0 → f x ≤ -4 ∧ (∃ y : ℝ, f y = -4 ∧ y < 0) := sorry

end max_value_of_function_for_x_lt_0_l87_87840


namespace difference_in_peaches_l87_87167

-- Define the number of peaches Audrey has
def audrey_peaches : ℕ := 26

-- Define the number of peaches Paul has
def paul_peaches : ℕ := 48

-- Define the expected difference
def expected_difference : ℕ := 22

-- The theorem stating the problem
theorem difference_in_peaches : (paul_peaches - audrey_peaches = expected_difference) :=
by
  sorry

end difference_in_peaches_l87_87167


namespace supplierB_stats_l87_87810

noncomputable def supplierB_data : List ℝ :=
  [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def mode (l : List ℝ) : ℝ :=
  l.foldl (λ m x, if l.count x > l.count m then x else m) l.head!

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x, (x - μ) ^ 2)).sum / l.length

theorem supplierB_stats :
  mean supplierB_data = 75 ∧
  mode supplierB_data = 75 ∧
  variance supplierB_data = 6 :=
by
  sorry

end supplierB_stats_l87_87810


namespace solve_y_l87_87791

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end solve_y_l87_87791


namespace acute_triangle_l87_87247

theorem acute_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
                       (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a)
                       (h7 : a^3 + b^3 = c^3) :
                       c^2 < a^2 + b^2 :=
by {
  sorry
}

end acute_triangle_l87_87247


namespace capital_contribution_A_l87_87648

theorem capital_contribution_A (P C : ℚ) (x : ℚ) : 
  (B_profit_share : ℚ) (B_months : ℕ) (A_months : ℕ) 
  (profit_ratio : ℚ) (capital_ratio : ℚ)
  (B_profit_share = 2 / 3) 
  (A_months = 15) 
  (B_months = 10) 
  (profit_ratio = 1 / 2) 
  (capital_ratio = (15 * x) / (10 * (1 - x))) 
  (profit_ratio = capital_ratio) : 
  x = 1 / 4 := 
sorry

end capital_contribution_A_l87_87648


namespace exists_four_distinct_indices_l87_87746

theorem exists_four_distinct_indices
  (a : Fin 5 → ℝ)
  (h : ∀ i, 0 < a i) :
  ∃ i j k l : (Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < 1 / 2 :=
by
  sorry

end exists_four_distinct_indices_l87_87746


namespace katie_new_games_l87_87737

theorem katie_new_games (K : ℕ) (h : K + 8 = 92) : K = 84 :=
by
  sorry

end katie_new_games_l87_87737


namespace system_of_equations_has_no_solution_l87_87516

theorem system_of_equations_has_no_solution
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y + z = 10)
  (h2 : 6 * x - 8 * y + 2 * z = 16)
  (h3 : x + y - z = 3) :
  false :=
by 
  sorry

end system_of_equations_has_no_solution_l87_87516


namespace cos_C_in_triangle_l87_87395

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end cos_C_in_triangle_l87_87395


namespace horner_multiplications_additions_l87_87351

-- Define the polynomial
def f (x : ℤ) : ℤ := x^7 + 2 * x^5 + 3 * x^4 + 4 * x^3 + 5 * x^2 + 6 * x + 7

-- Define the number of multiplications and additions required by Horner's method
def horner_method_mults (n : ℕ) : ℕ := n
def horner_method_adds (n : ℕ) : ℕ := n - 1

-- Define the value of x
def x : ℤ := 3

-- Define the degree of the polynomial
def degree_of_polynomial : ℕ := 7

-- Define the statements for the proof
theorem horner_multiplications_additions :
  horner_method_mults degree_of_polynomial = 7 ∧
  horner_method_adds degree_of_polynomial = 6 :=
by
  sorry

end horner_multiplications_additions_l87_87351


namespace smallest_natural_number_l87_87178

open Nat

theorem smallest_natural_number (n : ℕ) :
  (n + 1) % 4 = 0 ∧ (n + 1) % 6 = 0 ∧ (n + 1) % 10 = 0 ∧ (n + 1) % 12 = 0 →
  n = 59 :=
by
  sorry

end smallest_natural_number_l87_87178


namespace remove_least_candies_l87_87963

theorem remove_least_candies (total_candies : ℕ) (friends : ℕ) (candies_remaining : ℕ) : total_candies = 34 ∧ friends = 5 ∧ candies_remaining = 4 → (total_candies % friends = candies_remaining) :=
by
  intros h
  sorry

end remove_least_candies_l87_87963


namespace borrowed_amount_correct_l87_87471

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end borrowed_amount_correct_l87_87471


namespace post_height_l87_87822

-- Conditions
def spiral_path (circuit_per_rise rise_distance : ℝ) := ∀ (total_distance circ_circumference height : ℝ), 
  circuit_per_rise = total_distance / circ_circumference ∧ 
  height = circuit_per_rise * rise_distance

-- Given conditions
def cylinder_post : Prop := 
  ∀ (total_distance circ_circumference rise_distance : ℝ), 
    spiral_path (total_distance / circ_circumference) rise_distance ∧ 
    circ_circumference = 3 ∧ 
    rise_distance = 4 ∧ 
    total_distance = 12

-- Proof problem: Post height
theorem post_height : cylinder_post → ∃ height : ℝ, height = 16 := 
by sorry

end post_height_l87_87822


namespace increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l87_87173

-- Define the function z = x * y
def z (x y : ℝ) : ℝ := x * y

-- Initial point M0
def M0 : ℝ × ℝ := (1, 2)

-- Points to which we move
def M1 : ℝ × ℝ := (1.1, 2)
def M2 : ℝ × ℝ := (1, 1.9)
def M3 : ℝ × ℝ := (1.1, 2.2)

-- Proofs for the increments
theorem increment_M0_to_M1 : z M1.1 M1.2 - z M0.1 M0.2 = 0.2 :=
by sorry

theorem increment_M0_to_M2 : z M2.1 M2.2 - z M0.1 M0.2 = -0.1 :=
by sorry

theorem increment_M0_to_M3 : z M3.1 M3.2 - z M0.1 M0.2 = 0.42 :=
by sorry

end increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l87_87173


namespace abs_neg_number_l87_87118

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end abs_neg_number_l87_87118


namespace total_precious_stones_is_305_l87_87870

theorem total_precious_stones_is_305 :
  let agate := 25
  let olivine := agate + 5
  let sapphire := 2 * olivine
  let diamond := olivine + 11
  let amethyst := sapphire + diamond
  let ruby := diamond + 7
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 :=
by
  sorry

end total_precious_stones_is_305_l87_87870


namespace problem_statement_l87_87597

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end problem_statement_l87_87597


namespace total_bill_for_group_is_129_l87_87944

theorem total_bill_for_group_is_129 :
  let num_adults := 6
  let num_teenagers := 3
  let num_children := 1
  let cost_adult_meal := 9
  let cost_teenager_meal := 7
  let cost_child_meal := 5
  let cost_soda := 2.50
  let num_sodas := 10
  let cost_dessert := 4
  let num_desserts := 3
  let cost_appetizer := 6
  let num_appetizers := 2
  let total_bill := 
    (num_adults * cost_adult_meal) +
    (num_teenagers * cost_teenager_meal) +
    (num_children * cost_child_meal) +
    (num_sodas * cost_soda) +
    (num_desserts * cost_dessert) +
    (num_appetizers * cost_appetizer)
  total_bill = 129 := by
sorry

end total_bill_for_group_is_129_l87_87944


namespace inequality_proof_l87_87739

section
variable {a b x y : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hab : a + b = 1) :
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
  sorry
end

end inequality_proof_l87_87739


namespace unique_solution_values_l87_87049

theorem unique_solution_values (x y a : ℝ) :
  (∀ x y a, x^2 + y^2 + 2 * x ≤ 1 ∧ x - y + a = 0) → (a = -1 ∨ a = 3) :=
by
  intro h
  sorry

end unique_solution_values_l87_87049


namespace original_group_size_l87_87653

theorem original_group_size (M : ℕ) (R : ℕ) :
  (M * R * 40 = (M - 5) * R * 50) → M = 25 :=
by
  sorry

end original_group_size_l87_87653


namespace max_product_two_integers_sum_2000_l87_87281

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87281


namespace tetrahedron_cd_length_l87_87624

theorem tetrahedron_cd_length (a b c d : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] [MetricSpace d] :
  let ab := 53
  let edge_lengths := [17, 23, 29, 39, 46, 53]
  ∃ cd, cd = 17 :=
by
  sorry

end tetrahedron_cd_length_l87_87624


namespace find_s2_length_l87_87027

variables (s r : ℝ)

def condition1 : Prop := 2 * r + s = 2420
def condition2 : Prop := 2 * r + 3 * s = 4040

theorem find_s2_length (h1 : condition1 s r) (h2 : condition2 s r) : s = 810 :=
sorry

end find_s2_length_l87_87027


namespace find_q_l87_87719

theorem find_q {q : ℕ} (h : 27^8 = 9^q) : q = 12 := by
  sorry

end find_q_l87_87719


namespace baker_total_cost_is_correct_l87_87485

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end baker_total_cost_is_correct_l87_87485


namespace point_on_line_l87_87768

theorem point_on_line : ∀ (x y : ℝ), (x = 2 ∧ y = 7) → (y = 3 * x + 1) := 
by 
  intros x y h
  cases h with hx hy
  rw [hx, hy]
  sorry

end point_on_line_l87_87768


namespace black_haired_girls_count_l87_87587

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l87_87587


namespace inequality_problem_l87_87373

open Real

theorem inequality_problem {a b c d : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_ac : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3 := 
by 
  sorry

end inequality_problem_l87_87373


namespace admission_price_for_children_l87_87827

theorem admission_price_for_children 
  (admission_price_adult : ℕ)
  (total_persons : ℕ)
  (total_amount_dollars : ℕ)
  (children_attended : ℕ)
  (admission_price_children : ℕ)
  (h1 : admission_price_adult = 60)
  (h2 : total_persons = 280)
  (h3 : total_amount_dollars = 140)
  (h4 : children_attended = 80)
  (h5 : (total_persons - children_attended) * admission_price_adult + children_attended * admission_price_children = total_amount_dollars * 100)
  : admission_price_children = 25 := 
by 
  sorry

end admission_price_for_children_l87_87827


namespace greatest_product_sum_2000_l87_87277

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87277


namespace max_product_of_two_integers_sum_2000_l87_87309

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87309


namespace find_percentage_l87_87331

noncomputable def percentage_solve (x : ℝ) : Prop :=
  0.15 * 40 = (x / 100) * 16 + 2

theorem find_percentage (x : ℝ) (h : percentage_solve x) : x = 25 :=
by
  sorry

end find_percentage_l87_87331


namespace repetend_of_5_over_13_l87_87365

theorem repetend_of_5_over_13 : (∃ r : ℕ, r = 384615) :=
by
  let d := 13
  let n := 5
  let r := 384615
  -- Definitions to use:
  -- d is denominator 13
  -- n is numerator 5
  -- r is the repetend 384615
  sorry

end repetend_of_5_over_13_l87_87365


namespace find_a_l87_87671

theorem find_a
  (r1 r2 r3 : ℕ)
  (hr1 : r1 > 2) (hr2 : r2 > 2) (hr3 : r3 > 2)
  (a b c : ℤ)
  (hr : (Polynomial.X - Polynomial.C (r1 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r2 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r3 : ℤ)) = 
         Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
  (h : a + b + c + 1 = -2009) :
  a = -58 := sorry

end find_a_l87_87671


namespace quadratic_factorization_b_value_l87_87915

theorem quadratic_factorization_b_value (b : ℤ) (c d e f : ℤ) (h1 : 24 * c + 24 * d = 240) :
  (24 * (c * e) + b + 24) = 0 →
  (c * e = 24) →
  (c * f + d * e = b) →
  (d * f = 24) →
  (c + d = 10) →
  b = 52 :=
by
  intros
  sorry

end quadratic_factorization_b_value_l87_87915


namespace base15_mod_9_l87_87315

noncomputable def base15_to_decimal : ℕ :=
  2 * 15^3 + 6 * 15^2 + 4 * 15^1 + 3 * 15^0

theorem base15_mod_9 (n : ℕ) (h : n = base15_to_decimal) : n % 9 = 0 :=
sorry

end base15_mod_9_l87_87315


namespace degree_of_f_plus_cg_l87_87357

noncomputable def c : ℚ := -5 / 9

def f : Polynomial ℚ := 1 - 12 * Polynomial.X + 3 * Polynomial.X^2 - 4 * Polynomial.X^3 + 5 * Polynomial.X^4

def g : Polynomial ℚ := 3 - 2 * Polynomial.X - 6 * Polynomial.X^3 + 9 * Polynomial.X^4

theorem degree_of_f_plus_cg (h : c = -5 / 9) : (f + c * g).degree = 3 := sorry

end degree_of_f_plus_cg_l87_87357


namespace fraction_surface_area_red_l87_87687

theorem fraction_surface_area_red :
  ∀ (num_unit_cubes : ℕ) (side_length_large_cube : ℕ) (total_surface_area_painted : ℕ) (total_surface_area_unit_cubes : ℕ),
    num_unit_cubes = 8 →
    side_length_large_cube = 2 →
    total_surface_area_painted = 6 * (side_length_large_cube ^ 2) →
    total_surface_area_unit_cubes = num_unit_cubes * 6 →
    (total_surface_area_painted : ℝ) / total_surface_area_unit_cubes = 1 / 2 :=
by
  intros num_unit_cubes side_length_large_cube total_surface_area_painted total_surface_area_unit_cubes
  sorry

end fraction_surface_area_red_l87_87687


namespace abs_neg_number_l87_87116

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end abs_neg_number_l87_87116


namespace find_b8_l87_87434

noncomputable section

def increasing_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

axiom b_seq : ℕ → ℕ

axiom seq_inc : increasing_sequence b_seq

axiom b7_eq : b_seq 7 = 198

theorem find_b8 : b_seq 8 = 321 := by
  sorry

end find_b8_l87_87434


namespace cost_price_percentage_l87_87069

-- Define the condition that the profit percent is 11.11111111111111%
def profit_percent (CP SP: ℝ) : Prop :=
  ((SP - CP) / CP) * 100 = 11.11111111111111

-- Prove that under this condition, the cost price (CP) is 90% of the selling price (SP).
theorem cost_price_percentage (CP SP : ℝ) (h: profit_percent CP SP) : (CP / SP) * 100 = 90 :=
sorry

end cost_price_percentage_l87_87069


namespace total_price_for_pizza_l87_87677

-- Definitions based on conditions
def num_friends : ℕ := 5
def amount_per_person : ℕ := 8

-- The claim to be proven
theorem total_price_for_pizza : num_friends * amount_per_person = 40 := by
  -- Since the proof detail is not required, we use 'sorry' to skip the proof.
  sorry

end total_price_for_pizza_l87_87677


namespace intersection_l87_87377

namespace Proof

def A := {x : ℝ | 0 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | 3 * x^2 + x - 8 ≤ 0}

theorem intersection (x : ℝ) : x ∈ A ∩ B ↔ 0 ≤ x ∧ x ≤ (4:ℝ)/3 := 
by 
  sorry  -- proof placeholder

end Proof

end intersection_l87_87377


namespace tan_17pi_over_4_l87_87843

theorem tan_17pi_over_4 : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_17pi_over_4_l87_87843


namespace find_length_of_wood_l87_87819

-- Definitions based on given conditions
def Area := 24  -- square feet
def Width := 6  -- feet

-- The mathematical proof problem turned into Lean 4 statement
theorem find_length_of_wood (h : Area = 24) (hw : Width = 6) : (Length : ℕ) ∈ {l | l = Area / Width ∧ l = 4} :=
by {
  sorry
}

end find_length_of_wood_l87_87819


namespace find_value_of_y_l87_87790

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end find_value_of_y_l87_87790


namespace find_ab_solutions_l87_87364

theorem find_ab_solutions (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h1 : (a + 1) ∣ (a ^ 3 * b - 1))
  (h2 : (b - 1) ∣ (b ^ 3 * a + 1)) : 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) :=
sorry

end find_ab_solutions_l87_87364


namespace hexagon_angle_sum_l87_87732

theorem hexagon_angle_sum 
  (mA mB mC x y : ℝ)
  (hA : mA = 34)
  (hB : mB = 80)
  (hC : mC = 30)
  (hx' : x = 36 - y) : x + y = 36 :=
by
  sorry

end hexagon_angle_sum_l87_87732


namespace area_of_shaded_quadrilateral_l87_87824

-- The problem setup
variables 
  (triangle : Type) [Nonempty triangle]
  (area : triangle → ℝ)
  (EFA FAB FBD CEDF : triangle)
  (h_EFA : area EFA = 5)
  (h_FAB : area FAB = 9)
  (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF)

-- The goal to prove
theorem area_of_shaded_quadrilateral (EFA FAB FBD CEDF : triangle) 
  (h_EFA : area EFA = 5) (h_FAB : area FAB = 9) (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF) : 
  area CEDF = 45 :=
by
  sorry

end area_of_shaded_quadrilateral_l87_87824


namespace problem1_l87_87929

theorem problem1 (x : ℝ) (n : ℕ) (h : x^n = 2) : (3 * x^n)^2 - 4 * (x^2)^n = 20 :=
by
  sorry

end problem1_l87_87929


namespace trigonometric_identity_l87_87982

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (2 * Real.sin θ - 4 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l87_87982


namespace dog_biscuit_cost_l87_87504

open Real

theorem dog_biscuit_cost :
  (∀ (x : ℝ),
    (4 * x + 2) * 7 = 21 →
    x = 1 / 4) :=
by
  intro x h
  sorry

end dog_biscuit_cost_l87_87504


namespace Bobby_paycheck_final_amount_l87_87946

theorem Bobby_paycheck_final_amount :
  let salary := 450
  let federal_tax := (1 / 3 : ℚ) * salary
  let state_tax := 0.08 * salary
  let health_insurance := 50
  let life_insurance := 20
  let city_fee := 10
  let total_deductions := federal_tax + state_tax + health_insurance + life_insurance + city_fee
  salary - total_deductions = 184 :=
by
  -- We put sorry here to skip the proof step
  sorry

end Bobby_paycheck_final_amount_l87_87946


namespace circle_condition_l87_87570

theorem circle_condition (m : ℝ): (∃ x y : ℝ, (x^2 + y^2 - 2*x - 4*y + m = 0)) ↔ (m < 5) :=
by
  sorry

end circle_condition_l87_87570


namespace systematic_sampling_works_l87_87698

def missiles : List ℕ := List.range' 1 60 

-- Define the systematic sampling function
def systematic_sampling (start interval n : ℕ) : List ℕ :=
  List.range' 0 n |>.map (λ i => start + i * interval)

-- Stating the proof problem.
theorem systematic_sampling_works :
  systematic_sampling 5 12 5 = [5, 17, 29, 41, 53] :=
sorry

end systematic_sampling_works_l87_87698


namespace max_product_l87_87270

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87270


namespace chef_pillsbury_flour_l87_87836

theorem chef_pillsbury_flour (x : ℕ) (h : 7 / 2 = 28 / x) : x = 8 := sorry

end chef_pillsbury_flour_l87_87836


namespace stickers_per_friend_l87_87065

variable (d: ℕ) (h_d : d > 0)

theorem stickers_per_friend (h : 72 % d = 0) : 72 / d = 72 / d := by
  sorry

end stickers_per_friend_l87_87065


namespace find_abcd_l87_87440

theorem find_abcd {abcd abcde M : ℕ} :
  (M > 0) ∧ (∃ e, M % 100000 = e ∧ M^2 % 100000 = e) ∧ (M // 10000 > 0) ∧ (M // 10000 < 10) →
  (abcd = M // 10) →
  abcd = 9687 :=
by
  sorry

end find_abcd_l87_87440


namespace xiaofang_time_l87_87495

-- Definitions
def overlap_time (t : ℕ) : Prop :=
  t - t / 12 = 40

def opposite_time (t : ℕ) : Prop :=
  t - t / 12 = 40

-- Theorem statement
theorem xiaofang_time :
  ∃ (x y : ℕ), 
    480 + x = 8 * 60 + 43 ∧
    840 + y = 2 * 60 + 43 ∧
    overlap_time x ∧
    opposite_time y ∧
    (y + 840 - (x + 480)) = 6 * 60 :=
by
  sorry

end xiaofang_time_l87_87495


namespace abs_neg_2023_l87_87110

-- Define the absolute value function following the provided condition
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- State the theorem we need to prove
theorem abs_neg_2023 : abs (-2023) = 2023 :=
by
  -- Placeholder for proof
  sorry

end abs_neg_2023_l87_87110


namespace number_of_people_in_room_l87_87256

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end number_of_people_in_room_l87_87256


namespace Yoque_borrowed_150_l87_87469

noncomputable def Yoque_borrowed_amount (X : ℝ) : Prop :=
  1.10 * X = 11 * 15

theorem Yoque_borrowed_150 (X : ℝ) : Yoque_borrowed_amount X → X = 150 :=
by
  -- proof will be filled in
  sorry

end Yoque_borrowed_150_l87_87469


namespace magnitude_of_difference_between_roots_l87_87862

variable (α β m : ℝ)

theorem magnitude_of_difference_between_roots
    (hαβ_root : ∀ x, x^2 - 2 * m * x + m^2 - 4 = 0 → (x = α ∨ x = β)) :
    |α - β| = 4 := by
  sorry

end magnitude_of_difference_between_roots_l87_87862


namespace Yoongi_class_students_l87_87574

theorem Yoongi_class_students (Total_a Total_b Total_ab : ℕ)
  (h1 : Total_a = 18)
  (h2 : Total_b = 24)
  (h3 : Total_ab = 7)
  (h4 : Total_a + Total_b - Total_ab = 35) : 
  Total_a + Total_b - Total_ab = 35 :=
sorry

end Yoongi_class_students_l87_87574


namespace distance_between_chords_l87_87131

theorem distance_between_chords (R : ℝ) (AB CD : ℝ) (d : ℝ) : 
  R = 25 → AB = 14 → CD = 40 → (d = 39 ∨ d = 9) :=
by intros; sorry

end distance_between_chords_l87_87131


namespace find_f2_l87_87855

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := 
by
  sorry

end find_f2_l87_87855


namespace green_blue_tile_difference_is_15_l87_87219

def initial_blue_tiles : Nat := 13
def initial_green_tiles : Nat := 6
def second_blue_tiles : Nat := 2 * initial_blue_tiles
def second_green_tiles : Nat := 2 * initial_green_tiles
def border_green_tiles : Nat := 36
def total_blue_tiles : Nat := initial_blue_tiles + second_blue_tiles
def total_green_tiles : Nat := initial_green_tiles + second_green_tiles + border_green_tiles
def tile_difference : Nat := total_green_tiles - total_blue_tiles

theorem green_blue_tile_difference_is_15 : tile_difference = 15 := by
  sorry

end green_blue_tile_difference_is_15_l87_87219


namespace watermelon_percentage_l87_87814

theorem watermelon_percentage (total_drink : ℕ)
  (orange_percentage : ℕ)
  (grape_juice : ℕ)
  (watermelon_amount : ℕ)
  (W : ℕ) :
  total_drink = 300 →
  orange_percentage = 25 →
  grape_juice = 105 →
  watermelon_amount = total_drink - (orange_percentage * total_drink) / 100 - grape_juice →
  W = (watermelon_amount * 100) / total_drink →
  W = 40 :=
sorry

end watermelon_percentage_l87_87814


namespace lcm_12_18_l87_87537

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l87_87537


namespace book_sale_revenue_l87_87931

noncomputable def total_amount_received (price_per_book : ℝ) (B : ℕ) (sold_fraction : ℝ) :=
  sold_fraction * B * price_per_book

theorem book_sale_revenue (B : ℕ) (price_per_book : ℝ) (unsold_books : ℕ) (sold_fraction : ℝ) :
  (1 / 3 : ℝ) * B = unsold_books →
  price_per_book = 3.50 →
  unsold_books = 36 →
  sold_fraction = 2 / 3 →
  total_amount_received price_per_book B sold_fraction = 252 :=
by
  intros h1 h2 h3 h4
  sorry

end book_sale_revenue_l87_87931


namespace largest_consecutive_odd_numbers_l87_87761

theorem largest_consecutive_odd_numbers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) : 
  x + 6 = 27 :=
  sorry

end largest_consecutive_odd_numbers_l87_87761


namespace prob2_l87_87224

-- Given conditions
def X : ℝ → ℝ := sorry -- Define the random variable X
def mu : ℝ := 500
def sigma : ℝ := 60
def dist_X : ProbabilityDistribution := normal_distribution mu sigma

-- Given probability condition
axiom prob1 : dist_X.cdf 440 = 0.16

-- Goal: Prove the required probability
theorem prob2 : dist_X.prob (set.Ici 560) = 0.16 := 
sorry

end prob2_l87_87224


namespace possible_values_of_expression_l87_87707

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ v : ℝ, v = (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|) ∧ 
            (v = 5 ∨ v = 1 ∨ v = -3 ∨ v = -5)) :=
by
  sorry

end possible_values_of_expression_l87_87707


namespace subcommittees_with_at_least_one_teacher_l87_87443

theorem subcommittees_with_at_least_one_teacher
  (total_members teachers : ℕ)
  (total_members_eq : total_members = 12)
  (teachers_eq : teachers = 5)
  (subcommittee_size : ℕ)
  (subcommittee_size_eq : subcommittee_size = 5) :
  ∃ (n : ℕ), n = 771 :=
by
  sorry

end subcommittees_with_at_least_one_teacher_l87_87443


namespace max_product_of_two_integers_sum_2000_l87_87296

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87296


namespace sum_of_x_intercepts_l87_87593

theorem sum_of_x_intercepts (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (5 : ℤ) * (3 : ℤ) = (a : ℤ) * (b : ℤ)) : 
  ((-5 : ℤ) / (a : ℤ)) + ((-5 : ℤ) / (3 : ℤ)) + ((-1 : ℤ) / (1 : ℤ)) + ((-1 : ℤ) / (15 : ℤ)) = -8 := 
by 
  sorry

end sum_of_x_intercepts_l87_87593


namespace total_cost_is_80_l87_87483

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end total_cost_is_80_l87_87483


namespace solve_for_x_l87_87616

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l87_87616


namespace digit_A_of_3AA1_divisible_by_9_l87_87765

theorem digit_A_of_3AA1_divisible_by_9 (A : ℕ) (h : (3 + A + A + 1) % 9 = 0) : A = 7 :=
sorry

end digit_A_of_3AA1_divisible_by_9_l87_87765


namespace negation_q_sufficient_not_necessary_negation_p_l87_87745

theorem negation_q_sufficient_not_necessary_negation_p :
  (∃ x : ℝ, (∃ p : 16 - x^2 < 0, (x ∈ [-4, 4]))) →
  (∃ x : ℝ, (∃ q : x^2 + x - 6 > 0, (x ∈ [-3, 2]))) :=
sorry

end negation_q_sufficient_not_necessary_negation_p_l87_87745


namespace sequence_formula_l87_87733

theorem sequence_formula (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h : ∀ n, S_n n = 3 + 2 * a_n n) :
  ∀ n, a_n n = -3 * 2^(n - 1) :=
by
  sorry

end sequence_formula_l87_87733


namespace sandy_nickels_remaining_l87_87755

def original_nickels : ℕ := 31
def nickels_borrowed : ℕ := 20

theorem sandy_nickels_remaining : (original_nickels - nickels_borrowed) = 11 :=
by
  sorry

end sandy_nickels_remaining_l87_87755


namespace find_x_l87_87925

theorem find_x
  (x : ℤ)
  (h : 3 * x + 3 * 15 + 3 * 18 + 11 = 152) :
  x = 14 :=
by
  sorry

end find_x_l87_87925


namespace trigonometric_identities_l87_87794

open Real

theorem trigonometric_identities :
  (cos 75 * cos 75 = (2 - sqrt 3) / 4) ∧
  ((1 + tan 105) / (1 - tan 105) ≠ sqrt 3 / 3) ∧
  (tan 1 + tan 44 + tan 1 * tan 44 = 1) ∧
  (sin 70 * (sqrt 3 / tan 40 - 1) ≠ 2) :=
by
  sorry

end trigonometric_identities_l87_87794


namespace trees_chopped_in_first_half_l87_87663

theorem trees_chopped_in_first_half (x : ℕ) (h1 : ∀ t, t = x + 300) (h2 : 3 * t = 1500) : x = 200 :=
by
  sorry

end trees_chopped_in_first_half_l87_87663


namespace mark_sold_8_boxes_less_l87_87410

theorem mark_sold_8_boxes_less (T M A x : ℕ) (hT : T = 9) 
    (hM : M = T - x) (hA : A = T - 2) 
    (hM_ge_1 : 1 ≤ M) (hA_ge_1 : 1 ≤ A) 
    (h_sum_lt_T : M + A < T) : x = 8 := 
by
  sorry

end mark_sold_8_boxes_less_l87_87410


namespace complement_intersection_l87_87063

def set_P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def set_Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_intersection (P Q : Set ℝ) (hP : P = set_P) (hQ : Q = set_Q) :
  (Pᶜ ∩ Q) = {x | 1 < x ∧ x < 2} :=
by
  sorry

end complement_intersection_l87_87063


namespace john_started_5_days_ago_l87_87889

noncomputable def daily_wage (x : ℕ) : Prop := 250 + 10 * x = 750

theorem john_started_5_days_ago :
  ∃ x : ℕ, daily_wage x ∧ 250 / x = 5 :=
by
  sorry

end john_started_5_days_ago_l87_87889


namespace segments_have_common_point_l87_87776

-- Define the predicate that checks if two segments intersect
def segments_intersect (seg1 seg2 : ℝ × ℝ) : Prop :=
  let (a1, b1) := seg1
  let (a2, b2) := seg2
  max a1 a2 ≤ min b1 b2

-- Define the main theorem
theorem segments_have_common_point (segments : Fin 2019 → ℝ × ℝ)
  (h_intersect : ∀ (i j : Fin 2019), i ≠ j → segments_intersect (segments i) (segments j)) :
  ∃ p : ℝ, ∀ i : Fin 2019, (segments i).1 ≤ p ∧ p ≤ (segments i).2 :=
by
  sorry

end segments_have_common_point_l87_87776


namespace john_sells_percentage_of_newspapers_l87_87580

theorem john_sells_percentage_of_newspapers
    (n_newspapers : ℕ)
    (selling_price : ℝ)
    (cost_price_discount : ℝ)
    (profit : ℝ)
    (sold_percentage : ℝ)
    (h1 : n_newspapers = 500)
    (h2 : selling_price = 2)
    (h3 : cost_price_discount = 0.75)
    (h4 : profit = 550)
    (h5 : sold_percentage = 80) : 
    ( ∃ (sold_n : ℕ), 
      sold_n / n_newspapers * 100 = sold_percentage ∧
      sold_n * selling_price = 
        n_newspapers * selling_price * (1 - cost_price_discount) + profit) :=
by
  sorry

end john_sells_percentage_of_newspapers_l87_87580


namespace largest_triangle_perimeter_l87_87345

theorem largest_triangle_perimeter :
  ∀ (x : ℕ), 1 < x ∧ x < 15 → (7 + 8 + x = 29) :=
by
  intro x
  intro h
  sorry

end largest_triangle_perimeter_l87_87345


namespace no_rational_solutions_l87_87959

theorem no_rational_solutions (a b c d : ℚ) (n : ℕ) :
  ¬ ((a + b * (Real.sqrt 2))^(2 * n) + (c + d * (Real.sqrt 2))^(2 * n) = 5 + 4 * (Real.sqrt 2)) :=
sorry

end no_rational_solutions_l87_87959


namespace lcm_of_12_and_15_l87_87242
-- Import the entire Mathlib library

-- Define the given conditions
def HCF (a b : ℕ) : ℕ := gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / (gcd a b)

-- Given the values
def a := 12
def b := 15
def hcf := 3

-- State the proof problem
theorem lcm_of_12_and_15 : LCM a b = 60 :=
by
  -- Proof goes here (skipped)
  sorry

end lcm_of_12_and_15_l87_87242


namespace john_drinks_2_cups_per_day_l87_87085

noncomputable def fluid_ounces_in_gallon : ℕ := 128

noncomputable def half_gallon_in_fluid_ounces : ℕ := 64

noncomputable def standard_cup_size : ℕ := 8

noncomputable def cups_in_half_gallon : ℕ :=
  half_gallon_in_fluid_ounces / standard_cup_size

noncomputable def days_to_consume_half_gallon : ℕ := 4

noncomputable def cups_per_day : ℕ :=
  cups_in_half_gallon / days_to_consume_half_gallon

theorem john_drinks_2_cups_per_day :
  cups_per_day = 2 :=
by
  -- The proof is left as an exercise, but the statement should be correct.
  sorry

end john_drinks_2_cups_per_day_l87_87085


namespace smallest_n_between_76_and_100_l87_87917

theorem smallest_n_between_76_and_100 :
  ∃ (n : ℕ), (n > 1) ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 5 = 1) ∧ (76 < n) ∧ (n < 100) :=
sorry

end smallest_n_between_76_and_100_l87_87917


namespace part_one_part_two_l87_87054

-- Define the set M and sum of subsets S_n
def M (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}
def S_n (n : ℕ) [Fact (3 ≤ n)] : ℕ :=
  ∑ t in (Finset.powersetLen 3 (Finset.range (n + 1))), (∑ x in t, x)

-- Part (Ⅰ): Prove S_n = C_{n-1}^2 * (n(n+1)/2)
theorem part_one (n : ℕ) [Fact (3 ≤ n)] :
  S_n n = Nat.choose (n - 1) 2 * (n * (n + 1) / 2) :=
by
  sorry

-- Part (Ⅱ): Prove S_3 + S_4 + ... + S_n = 6C_{n+2}^5
theorem part_two (n : ℕ) [Fact (3 ≤ n)] :
  (∑ k in Finset.range' 3 (n - 2), S_n k) = 6 * Nat.choose ((n + 2) - 1) 5 :=
by
  sorry

end part_one_part_two_l87_87054


namespace totalMoney_l87_87579

noncomputable def joannaMoney : ℕ := 8
noncomputable def brotherMoney : ℕ := 3 * joannaMoney
noncomputable def sisterMoney : ℕ := joannaMoney / 2

theorem totalMoney : joannaMoney + brotherMoney + sisterMoney = 36 := by
  sorry

end totalMoney_l87_87579


namespace choose_officers_from_six_l87_87882

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l87_87882


namespace MorseCodeDistinctSymbols_l87_87210

theorem MorseCodeDistinctSymbols:
  (1.sequence (λ _, bool).length = {1, 2, 3, 4, 5}).card = 62 :=
by
  sorry

end MorseCodeDistinctSymbols_l87_87210


namespace initial_men_count_l87_87759

theorem initial_men_count (M : ℕ) (P : ℝ) 
  (h1 : P = M * 12) 
  (h2 : P = (M + 300) * 9.662337662337663) :
  M = 1240 :=
sorry

end initial_men_count_l87_87759


namespace sum_of_cubes_of_integers_l87_87773

theorem sum_of_cubes_of_integers (n: ℕ) (h1: (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 8830) : 
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 52264 :=
by
  sorry

end sum_of_cubes_of_integers_l87_87773


namespace delegates_without_badges_l87_87502

theorem delegates_without_badges :
  ∀ (total_delegates : ℕ)
    (preprinted_frac hand_written_frac break_frac : ℚ),
  total_delegates = 100 →
  preprinted_frac = 1/5 →
  break_frac = 3/7 →
  hand_written_frac = 2/9 →
  let preprinted_delegates := (preprinted_frac * total_delegates).natAbs in
  let remaining_after_preprinted := total_delegates - preprinted_delegates in
  let break_delegates := (break_frac * remaining_after_preprinted).natAbs in
  let remaining_after_break := remaining_after_preprinted - break_delegates in
  let handwritten_delegates := (hand_written_frac * remaining_after_break).natAbs in
  let non_badge_delegates := remaining_after_break - handwritten_delegates in
  non_badge_delegates = 36 :=
by
  intros
  sorry

end delegates_without_badges_l87_87502


namespace quadratic_inequality_solution_l87_87239

theorem quadratic_inequality_solution : ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l87_87239


namespace borrowed_amount_correct_l87_87472

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end borrowed_amount_correct_l87_87472


namespace inequalities_sufficient_but_not_necessary_l87_87008

theorem inequalities_sufficient_but_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d) → (a + c > b + d) ∧ ¬((a + c > b + d) → (a > b ∧ c > d)) :=
by
  sorry

end inequalities_sufficient_but_not_necessary_l87_87008


namespace units_digit_47_pow_47_l87_87460

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l87_87460


namespace probability_of_sequence_HTHT_l87_87924

noncomputable def prob_sequence_HTHT : ℚ :=
  let p := 1 / 2
  (p * p * p * p)

theorem probability_of_sequence_HTHT :
  prob_sequence_HTHT = 1 / 16 := 
by
  sorry

end probability_of_sequence_HTHT_l87_87924


namespace cyclic_sum_inequality_l87_87594

theorem cyclic_sum_inequality (n : ℕ) (a : Fin n.succ -> ℕ) (h : ∀ i, a i > 0) : 
  (Finset.univ.sum fun i => a i / a ((i + 1) % n)) ≥ n :=
by
  sorry

end cyclic_sum_inequality_l87_87594


namespace original_inhabitants_7200_l87_87828

noncomputable def original_inhabitants (X : ℝ) : Prop :=
  let initial_decrease := 0.9 * X
  let final_decrease := 0.75 * initial_decrease
  final_decrease = 4860

theorem original_inhabitants_7200 : ∃ X : ℝ, original_inhabitants X ∧ X = 7200 := by
  use 7200
  unfold original_inhabitants
  simp
  sorry

end original_inhabitants_7200_l87_87828


namespace geom_mean_between_2_and_8_l87_87847

theorem geom_mean_between_2_and_8 (b : ℝ) (h : b^2 = 16) : b = 4 ∨ b = -4 :=
by
  sorry

end geom_mean_between_2_and_8_l87_87847


namespace initial_liquid_A_amount_l87_87801

noncomputable def initial_amount_of_A (x : ℚ) : ℚ :=
  3 * x

theorem initial_liquid_A_amount {x : ℚ} (h : (3 * x - 3) / (2 * x + 3) = 3 / 5) : initial_amount_of_A (8 / 3) = 8 := by
  sorry

end initial_liquid_A_amount_l87_87801


namespace Morse_code_distinct_symbols_l87_87206

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l87_87206


namespace sum_of_roots_of_quadratic_l87_87200

theorem sum_of_roots_of_quadratic (x1 x2 : ℝ) (h : x1 * x2 + -(x1 + x2) * 6 + 5 = 0) : x1 + x2 = 6 :=
by
-- Vieta's formulas for the sum of the roots of a quadratic equation state that x1 + x2 = -b / a.
sorry

end sum_of_roots_of_quadratic_l87_87200


namespace find_S9_l87_87704

variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Condition: an arithmetic sequence with the sum of first n terms S_n.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a_3 + a_4 + a_5 + a_6 + a_7 = 20.
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 20

-- The sum of the first n terms.
def sum_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (a 1 + a n)

-- Prove that S_9 = 36.
theorem find_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic_sequence : arithmetic_sequence a) 
  (h_given_condition : given_condition a)
  (h_sum_terms : sum_terms S a) : 
  S 9 = 36 :=
sorry

end find_S9_l87_87704


namespace probability_calculation_l87_87084

-- Define the initial conditions of Jar A
def jarA_initial_green : ℕ := 6
def jarA_initial_red : ℕ := 3
def jarA_initial_blue : ℕ := 9

-- Define the total number of buttons initially in Jar A
def jarA_initial_total : ℕ := jarA_initial_green + jarA_initial_red + jarA_initial_blue

-- Define the transfer conditions
variable (x : ℕ) -- number of green buttons moved
variable (y : ℕ) -- number of blue buttons moved
def transfer_condition : Prop := y = 2 * x

-- Define the condition of half of the buttons remaining in Jar A
def half_buttons_in_jarA : ℕ := jarA_initial_total / 2

-- Define the number of buttons moved
def buttons_moved : ℕ := x + y
def half_condition : Prop := buttons_moved = half_buttons_in_jarA

-- After the transfer, define the remaining buttons in Jar A
def jarA_remaining_green : ℕ := jarA_initial_green - x
def jarA_remaining_blue : ℕ := jarA_initial_blue - y
def jarA_remaining_red : ℕ := jarA_initial_red

-- The remaining total in Jar A
def jarA_remaining_total : ℕ :=
  jarA_remaining_green + jarA_remaining_red + jarA_remaining_blue

-- Probabilities calculations
def prob_blue_jarA : ℚ := jarA_remaining_blue / jarA_remaining_total
def prob_green_jarB : ℚ := x / buttons_moved

-- Define the statement to be proven
theorem probability_calculation (h_transfer : transfer_condition x y)
                                (h_half : half_condition x y) :
  prob_blue_jarA x y * prob_green_jarB x y = 1 / 9 :=
begin
  -- Required hypothesis and skipped proof
  sorry
end

end probability_calculation_l87_87084


namespace percentage_markup_l87_87438

theorem percentage_markup (selling_price cost_price : ℚ)
  (h_selling_price : selling_price = 8325)
  (h_cost_price : cost_price = 7239.13) :
  ((selling_price - cost_price) / cost_price) * 100 = 15 := 
sorry

end percentage_markup_l87_87438


namespace isabella_stops_l87_87685

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_stops (P : ℕ → ℚ) (h : ∀ n, P n = 1 / (n * (n + 1))) : 
  ∃ n : ℕ, n = 55 ∧ P n < 1 / 3000 :=
by {
  sorry
}

end isabella_stops_l87_87685


namespace speed_of_freight_train_l87_87026

-- Definitions based on the conditions
def distance := 390  -- The towns are 390 km apart
def express_speed := 80  -- The express train travels at 80 km per hr
def travel_time := 3  -- They pass one another 3 hr later

-- The freight train travels 30 km per hr slower than the express train
def freight_speed := express_speed - 30

-- The statement that we aim to prove:
theorem speed_of_freight_train : freight_speed = 50 := 
by 
  sorry

end speed_of_freight_train_l87_87026


namespace ella_savings_l87_87688

theorem ella_savings
  (initial_cost_per_lamp : ℝ)
  (num_lamps : ℕ)
  (discount_rate : ℝ)
  (additional_discount : ℝ)
  (initial_total_cost : ℝ := num_lamps * initial_cost_per_lamp)
  (discounted_lamp_cost : ℝ := initial_cost_per_lamp - (initial_cost_per_lamp * discount_rate))
  (total_cost_with_discount : ℝ := num_lamps * discounted_lamp_cost)
  (total_cost_after_additional_discount : ℝ := total_cost_with_discount - additional_discount) :
  initial_cost_per_lamp = 15 →
  num_lamps = 3 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  initial_total_cost - total_cost_after_additional_discount = 16.25 :=
by
  intros
  sorry

end ella_savings_l87_87688


namespace contradiction_to_at_least_one_not_greater_than_60_l87_87926

-- Define a condition for the interior angles of a triangle being > 60
def all_angles_greater_than_60 (α β γ : ℝ) : Prop :=
  α > 60 ∧ β > 60 ∧ γ > 60

-- Define the negation of the proposition "At least one of the interior angles is not greater than 60"
def at_least_one_not_greater_than_60 (α β γ : ℝ) : Prop :=
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The mathematically equivalent proof problem
theorem contradiction_to_at_least_one_not_greater_than_60 (α β γ : ℝ) :
  ¬ at_least_one_not_greater_than_60 α β γ ↔ all_angles_greater_than_60 α β γ := by
  sorry

end contradiction_to_at_least_one_not_greater_than_60_l87_87926


namespace find_four_digit_number_abcd_exists_l87_87439

theorem find_four_digit_number_abcd_exists (M : ℕ) (H1 : M > 0) (H2 : M % 10 ≠ 0) 
    (H3 : M % 100000 = M^2 % 100000) : ∃ abcd : ℕ, abcd = 2502 :=
by
  -- Proof is omitted
  sorry

end find_four_digit_number_abcd_exists_l87_87439


namespace simplify_expression_l87_87756

variable (a b c d x y z : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * a^2 * y^3 + c^2 * z^3) + dz * (a^2 * x^3 + 3 * c^2 * y^3 + b^2 * z^3)) / (cx + dz) =
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
sorry

end simplify_expression_l87_87756


namespace numPermutationsOfDigits_l87_87259

def numUniqueDigitPermutations : Nat :=
  Nat.factorial 4

theorem numPermutationsOfDigits : numUniqueDigitPermutations = 24 := 
by
  -- proof goes here
  sorry

end numPermutationsOfDigits_l87_87259


namespace red_not_equal_blue_l87_87013

theorem red_not_equal_blue (total_cubes : ℕ) (red_cubes : ℕ) (blue_cubes : ℕ) (edge_length : ℕ)
  (total_surface_squares : ℕ) (max_red_squares : ℕ) :
  total_cubes = 27 →
  red_cubes = 9 →
  blue_cubes = 18 →
  edge_length = 3 →
  total_surface_squares = 6 * edge_length^2 →
  max_red_squares = 26 →
  ¬ (total_surface_squares = 2 * max_red_squares) :=
by
  intros
  sorry

end red_not_equal_blue_l87_87013


namespace find_divisor_l87_87232

theorem find_divisor : ∃ D : ℕ, 14698 = (D * 89) + 14 ∧ D = 165 :=
by
  use 165
  sorry

end find_divisor_l87_87232


namespace project_selection_l87_87157

theorem project_selection :
  let num_key_projects := 4
  let num_general_projects := 6
  let ways_select_key := Nat.choose 4 2
  let ways_select_general := Nat.choose 6 2
  let total_ways := ways_select_key * ways_select_general
  let ways_excluding_A := Nat.choose 3 2
  let ways_excluding_B := Nat.choose 5 2
  let ways_excluding_A_and_B := ways_excluding_A * ways_excluding_B
  total_ways - ways_excluding_A_and_B = 60
:= by
  sorry

end project_selection_l87_87157


namespace proportion_of_solution_x_in_mixture_l87_87476

-- Definitions for the conditions in given problem
def solution_x_contains_perc_a : ℚ := 0.20
def solution_y_contains_perc_a : ℚ := 0.30
def solution_z_contains_perc_a : ℚ := 0.40

def solution_y_to_z_ratio : ℚ := 3 / 2
def final_mixture_perc_a : ℚ := 0.25

-- Proving the proportion of solution x in the mixture equals 9/14
theorem proportion_of_solution_x_in_mixture
  (x y z : ℚ) (k : ℚ) (hx : x = 9 * k) (hy : y = 3 * k) (hz : z = 2 * k) :
  solution_x_contains_perc_a * x + solution_y_contains_perc_a * y + solution_z_contains_perc_a * z
  = final_mixture_perc_a * (x + y + z) →
  x / (x + y + z) = 9 / 14 :=
by
  intros h
  -- leaving the proof as a placeholder
  sorry

end proportion_of_solution_x_in_mixture_l87_87476


namespace abs_square_implication_l87_87318

theorem abs_square_implication (a b : ℝ) (h : abs a > abs b) : a^2 > b^2 :=
by sorry

end abs_square_implication_l87_87318


namespace swimmer_speed_proof_l87_87033

-- Definition of the conditions
def current_speed : ℝ := 2
def swimming_time : ℝ := 1.5
def swimming_distance : ℝ := 3

-- Prove: Swimmer's speed in still water
def swimmer_speed_in_still_water : ℝ := 4

-- Statement: Given the conditions, the swimmer's speed in still water equals 4 km/h
theorem swimmer_speed_proof :
  (swimming_distance = (swimmer_speed_in_still_water - current_speed) * swimming_time) →
  swimmer_speed_in_still_water = 4 :=
by
  intro h
  sorry

end swimmer_speed_proof_l87_87033


namespace max_product_of_two_integers_sum_2000_l87_87294

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87294


namespace units_digit_47_pow_47_l87_87463

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l87_87463


namespace both_firms_participate_l87_87214

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end both_firms_participate_l87_87214


namespace cakes_count_l87_87905

theorem cakes_count (x y : ℕ) 
  (price_fruit price_chocolate total_cost : ℝ) 
  (avg_price : ℝ) 
  (H1 : price_fruit = 4.8)
  (H2 : price_chocolate = 6.6)
  (H3 : total_cost = 167.4)
  (H4 : avg_price = 6.2)
  (H5 : price_fruit * x + price_chocolate * y = total_cost)
  (H6 : total_cost / (x + y) = avg_price) : 
  x = 6 ∧ y = 21 := 
by
  sorry

end cakes_count_l87_87905


namespace mo_tea_cups_l87_87590

theorem mo_tea_cups (n t : ℤ) 
  (h1 : 2 * n + 5 * t = 26) 
  (h2 : 5 * t = 2 * n + 14) :
  t = 4 :=
sorry

end mo_tea_cups_l87_87590


namespace tables_needed_l87_87656

-- Conditions
def n_invited : ℕ := 18
def n_no_show : ℕ := 12
def capacity_per_table : ℕ := 3

-- Calculation of attendees
def n_attendees : ℕ := n_invited - n_no_show

-- Proof for the number of tables needed
theorem tables_needed : (n_attendees / capacity_per_table) = 2 := by
  -- Sorry will be here to show it's incomplete
  sorry

end tables_needed_l87_87656


namespace reciprocal_lcm_of_24_and_208_l87_87442

theorem reciprocal_lcm_of_24_and_208 :
  (1 / (Nat.lcm 24 208)) = (1 / 312) :=
by
  sorry

end reciprocal_lcm_of_24_and_208_l87_87442


namespace tree_height_increase_fraction_l87_87146

theorem tree_height_increase_fraction :
  ∀ (initial_height annual_increase : ℝ) (additional_years₄ additional_years₆ : ℕ),
    initial_height = 4 →
    annual_increase = 0.4 →
    additional_years₄ = 4 →
    additional_years₆ = 6 →
    ((initial_height + annual_increase * additional_years₆) - (initial_height + annual_increase * additional_years₄)) / (initial_height + annual_increase * additional_years₄) = 1 / 7 :=
by
  sorry

end tree_height_increase_fraction_l87_87146


namespace max_product_of_two_integers_sum_2000_l87_87297

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87297


namespace no_square_ends_in_4444_l87_87514

theorem no_square_ends_in_4444:
  ∀ (a k : ℕ), (a ^ 2 = 1000 * k + 444) → (∃ b m n : ℕ, (b = 500 * n + 38) ∨ (b = 500 * n - 38) → (a = 2 * b) →
  (a ^ 2 ≠ 1000 * m + 4444)) :=
by
  sorry

end no_square_ends_in_4444_l87_87514


namespace isosceles_triangle_angles_l87_87499

theorem isosceles_triangle_angles (α β γ : ℝ) (h_iso : α = β ∨ α = γ ∨ β = γ) (h_angle : α + β + γ = 180) (h_40 : α = 40 ∨ β = 40 ∨ γ = 40) :
  (α = 70 ∧ β = 70 ∧ γ = 40) ∨ (α = 40 ∧ β = 100 ∧ γ = 40) ∨ (α = 40 ∧ β = 40 ∧ γ = 100) :=
by
  sorry

end isosceles_triangle_angles_l87_87499


namespace people_in_room_eq_33_l87_87254

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end people_in_room_eq_33_l87_87254


namespace total_race_time_l87_87075

theorem total_race_time 
  (num_runners : ℕ) 
  (first_five_time : ℕ) 
  (additional_time : ℕ) 
  (total_runners : ℕ) 
  (num_first_five : ℕ)
  (num_last_three : ℕ) 
  (total_expected_time : ℕ) 
  (h1 : num_runners = 8) 
  (h2 : first_five_time = 8) 
  (h3 : additional_time = 2) 
  (h4 : num_first_five = 5)
  (h5 : num_last_three = num_runners - num_first_five)
  (h6 : total_runners = num_first_five + num_last_three)
  (h7 : 5 * first_five_time + 3 * (first_five_time + additional_time) = total_expected_time)
  : total_expected_time = 70 := 
by
  sorry

end total_race_time_l87_87075


namespace proof_numbers_exist_l87_87841

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end proof_numbers_exist_l87_87841


namespace scooter_gain_percent_l87_87237

theorem scooter_gain_percent 
  (purchase_price : ℕ) 
  (repair_costs : ℕ) 
  (selling_price : ℕ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by
  sorry

end scooter_gain_percent_l87_87237


namespace Frank_initial_savings_l87_87853

theorem Frank_initial_savings 
  (cost_per_toy : Nat)
  (number_of_toys : Nat)
  (allowance : Nat)
  (total_cost : Nat)
  (initial_savings : Nat)
  (h1 : cost_per_toy = 8)
  (h2 : number_of_tys = 5)
  (h3 : allowance = 37)
  (h4 : total_cost = number_of_toys * cost_per_toy)
  (h5 : initial_savings + allowance = total_cost)
  : initial_savings = 3 := 
by
  sorry

end Frank_initial_savings_l87_87853


namespace range_of_a_for_common_points_l87_87709

theorem range_of_a_for_common_points (a : ℝ) : (∃ x : ℝ, x > 0 ∧ ax^2 = Real.exp x) ↔ a ≥ Real.exp 2 / 4 :=
sorry

end range_of_a_for_common_points_l87_87709


namespace reduce_to_one_piece_l87_87623

-- Definitions representing the conditions:
def plane_divided_into_unit_triangles : Prop := sorry
def initial_configuration (n : ℕ) : Prop := sorry
def possible_moves : Prop := sorry

-- Main theorem statement:
theorem reduce_to_one_piece (n : ℕ) 
  (H1 : plane_divided_into_unit_triangles) 
  (H2 : initial_configuration n) 
  (H3 : possible_moves) : 
  ∃ k : ℕ, k * 3 = n :=
sorry

end reduce_to_one_piece_l87_87623


namespace largest_n_divides_1005_fact_l87_87041

theorem largest_n_divides_1005_fact (n : ℕ) : (∃ n, 10^n ∣ (Nat.factorial 1005)) ↔ n = 250 :=
by
  sorry

end largest_n_divides_1005_fact_l87_87041


namespace order_exponents_l87_87906

theorem order_exponents :
  (2:ℝ) ^ 300 < (3:ℝ) ^ 200 ∧ (3:ℝ) ^ 200 < (10:ℝ) ^ 100 :=
by
  sorry

end order_exponents_l87_87906


namespace repeating_decimals_sum_as_fraction_l87_87691

theorem repeating_decimals_sum_as_fraction :
  (0.3333...).to_rat + (0.020202...).to_rat + (0.00030003...).to_rat = 3538 / 9999 := by
sorry

end repeating_decimals_sum_as_fraction_l87_87691


namespace symmetric_point_in_xOz_l87_87888

def symmetric_point (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, -P.2, P.3)

theorem symmetric_point_in_xOz (P : ℝ × ℝ × ℝ) : 
  symmetric_point P = (P.1, -P.2, P.3) :=
by
  sorry

example : symmetric_point (-1, 2, 1) = (-1, -2, 1) :=
by
  rw symmetric_point_in_xOz
  rw symmetric_point
  sorry

end symmetric_point_in_xOz_l87_87888


namespace new_oranges_added_l87_87823
-- Import the necessary library

-- Define the constants and conditions
def initial_oranges : ℕ := 5
def thrown_away : ℕ := 2
def total_oranges_now : ℕ := 31

-- Define new_oranges as the variable we want to prove
def new_oranges (x : ℕ) : Prop := x = 28

-- The theorem to prove how many new oranges were added
theorem new_oranges_added :
  ∃ (x : ℕ), new_oranges x ∧ total_oranges_now = initial_oranges - thrown_away + x :=
by
  sorry

end new_oranges_added_l87_87823


namespace valid_parameterizations_l87_87356

noncomputable def line_equation (x y : ℝ) : Prop := y = (5/3) * x + 1

def parametrize_A (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (3 + t * 3, 6 + t * 5) ∧ line_equation x y

def parametrize_D (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (-1 + t * 3, -2/3 + t * 5) ∧ line_equation x y

theorem valid_parameterizations : parametrize_A t ∧ parametrize_D t :=
by
  -- Proof steps are skipped
  sorry

end valid_parameterizations_l87_87356


namespace repeating_decimal_equals_fraction_l87_87846

theorem repeating_decimal_equals_fraction : 
  let a := 58 / 100
  let r := 1 / 100
  let S := a / (1 - r)
  S = (58 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_equals_fraction_l87_87846


namespace total_sample_needed_l87_87335

-- Given constants
def elementary_students : ℕ := 270
def junior_high_students : ℕ := 360
def senior_high_students : ℕ := 300
def junior_high_sample : ℕ := 12

-- Calculate the total number of students in the school
def total_students : ℕ := elementary_students + junior_high_students + senior_high_students

-- Define the sampling ratio based on junior high section
def sampling_ratio : ℚ := junior_high_sample / junior_high_students

-- Apply the sampling ratio to the total number of students to get the total sample size
def total_sample : ℚ := sampling_ratio * total_students

-- Prove that the total number of students that need to be sampled is 31
theorem total_sample_needed : total_sample = 31 := sorry

end total_sample_needed_l87_87335


namespace average_sale_six_months_l87_87339

theorem average_sale_six_months :
  let sale1 := 2500
  let sale2 := 6500
  let sale3 := 9855
  let sale4 := 7230
  let sale5 := 7000
  let sale6 := 11915
  let total_sales := sale1 + sale2 + sale3 + sale4 + sale5 + sale6
  let num_months := 6
  (total_sales / num_months) = 7500 :=
by
  sorry

end average_sale_six_months_l87_87339


namespace goldfish_in_each_pond_l87_87138

variable (x : ℕ)
variable (l1 h1 l2 h2 : ℕ)

-- Conditions
def cond1 : Prop := l1 + h1 = x ∧ l2 + h2 = x
def cond2 : Prop := 4 * l1 = 3 * h1
def cond3 : Prop := 3 * l2 = 5 * h2
def cond4 : Prop := l2 = l1 + 33

theorem goldfish_in_each_pond : cond1 x l1 h1 l2 h2 ∧ cond2 l1 h1 ∧ cond3 l2 h2 ∧ cond4 l1 l2 → 
  x = 168 := 
by 
  sorry

end goldfish_in_each_pond_l87_87138


namespace smallest_N_l87_87665

-- Definitions for the problem conditions
def is_rectangular_block (a b c : ℕ) (N : ℕ) : Prop :=
  N = a * b * c ∧ 143 = (a - 1) * (b - 1) * (c - 1)

-- Theorem to prove the smallest possible value of N
theorem smallest_N : ∃ a b c : ℕ, is_rectangular_block a b c 336 :=
by
  sorry

end smallest_N_l87_87665


namespace min_value_fraction_l87_87405

theorem min_value_fraction (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_sum : a + 3 * b = 1) :
  (∀ x y : ℝ, (0 < x) → (0 < y) → x + 3 * y = 1 → 16 ≤ 1 / x + 3 / y) :=
sorry

end min_value_fraction_l87_87405


namespace area_of_triangle_l87_87034

open EuclideanGeometry

-- Define the points
def pointA : Point := (3, -3)
def pointB : Point := (8, 4)
def pointC : Point := (3, 4)

theorem area_of_triangle :
  area_triangle pointA pointB pointC = 17.5 :=
  by
  -- Proof steps will go here
  sorry

end area_of_triangle_l87_87034


namespace semicircle_radius_l87_87325

noncomputable def radius_of_semicircle (P : ℝ) (h : P = 144) : ℝ :=
  144 / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (h : P = 144) : radius_of_semicircle P h = 144 / (Real.pi + 2) :=
  by sorry

end semicircle_radius_l87_87325


namespace basketball_holes_l87_87228

theorem basketball_holes (soccer_balls total_basketballs soccer_balls_with_hole balls_without_holes basketballs_without_holes: ℕ) 
  (h1: soccer_balls = 40) 
  (h2: total_basketballs = 15)
  (h3: soccer_balls_with_hole = 30) 
  (h4: balls_without_holes = 18) 
  (h5: basketballs_without_holes = 8) 
  : (total_basketballs - basketballs_without_holes = 7) := 
by
  sorry

end basketball_holes_l87_87228


namespace calculate_expression_l87_87678

theorem calculate_expression :
  150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 :=
by
  sorry

end calculate_expression_l87_87678


namespace count_routes_from_P_to_Q_l87_87951

variable (P Q R S T : Type)
variable (roadPQ roadPS roadPT roadQR roadQS roadRS roadST : Prop)

theorem count_routes_from_P_to_Q :
  ∃ (routes : ℕ), routes = 16 :=
by
  sorry

end count_routes_from_P_to_Q_l87_87951


namespace cost_of_math_book_l87_87921

-- The definitions based on the conditions from the problem
def total_books : ℕ := 90
def math_books : ℕ := 54
def history_books := total_books - math_books -- 36
def cost_history_book : ℝ := 5
def total_cost : ℝ := 396

-- The theorem we want to prove: the cost of each math book
theorem cost_of_math_book (M : ℝ) : (math_books * M + history_books * cost_history_book = total_cost) → M = 4 := 
by 
  sorry

end cost_of_math_book_l87_87921


namespace city_population_distribution_l87_87657

theorem city_population_distribution :
  (20 + 35) = 55 :=
by
  sorry

end city_population_distribution_l87_87657


namespace mixed_oil_rate_l87_87997

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end mixed_oil_rate_l87_87997


namespace infinite_solutions_d_eq_5_l87_87955

theorem infinite_solutions_d_eq_5 :
  ∃ (d : ℝ), d = 5 ∧ ∀ (y : ℝ), 3 * (5 + d * y) = 15 * y + 15 :=
by
  sorry

end infinite_solutions_d_eq_5_l87_87955


namespace common_ratio_of_geometric_series_l87_87972

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end common_ratio_of_geometric_series_l87_87972


namespace unique_arrangement_l87_87182

def valid_grid (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  (∀ i : Fin 4, (∃ j1 j2 j3 : Fin 4,
    j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3 ∧
    arrangement i j1 = 'A' ∧
    arrangement i j2 = 'B' ∧
    arrangement i j3 = 'C')) ∧
  (∀ j : Fin 4, (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 j = 'A' ∧
    arrangement i2 j = 'B' ∧
    arrangement i3 j = 'C')) ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 i1 = 'A' ∧
    arrangement i2 i2 = 'B' ∧
    arrangement i3 i3 = 'C') ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 (Fin.mk (3 - i1.val) sorry) = 'A' ∧
    arrangement i2 (Fin.mk (3 - i2.val) sorry) = 'B' ∧
    arrangement i3 (Fin.mk (3 - i3.val) sorry) = 'C')

def fixed_upper_left (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  arrangement 0 0 = 'A'

theorem unique_arrangement : ∃! arrangement : Matrix (Fin 4) (Fin 4) Char,
  valid_grid arrangement ∧ fixed_upper_left arrangement :=
sorry

end unique_arrangement_l87_87182


namespace equation_solution_l87_87731

theorem equation_solution : ∃ x : ℝ, (3 / 20) + (3 / x) = (8 / x) + (1 / 15) ∧ x = 60 :=
by
  use 60
  -- skip the proof
  sorry

end equation_solution_l87_87731


namespace max_product_l87_87269

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87269


namespace calvin_total_insects_l87_87225

def R : ℕ := 15
def S : ℕ := 2 * R - 8
def C : ℕ := 11 -- rounded from (1/2) * R + 3
def P : ℕ := 3 * S + 7
def B : ℕ := 4 * C - 2
def E : ℕ := 3 * (R + S + C + P + B)
def total_insects : ℕ := R + S + C + P + B + E

theorem calvin_total_insects : total_insects = 652 :=
by
  -- service the proof here.
  sorry

end calvin_total_insects_l87_87225


namespace Sophie_l87_87426

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end Sophie_l87_87426


namespace find_a_20_l87_87902

-- Definitions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ (a₀ d : ℤ), ∀ n, a n = a₀ + n * d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 0 + a (n - 1)) / 2

-- Conditions and question
theorem find_a_20 (a S : ℕ → ℤ) (a₀ d : ℤ) :
  arithmetic_seq a ∧ sum_first_n a S ∧ 
  S 6 = 8 * (S 3) ∧ a 3 - a 5 = 8 → a 20 = -74 :=
by
  sorry

end find_a_20_l87_87902


namespace arrange_in_order_l87_87806

noncomputable def x1 : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def x2 : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def x3 : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))
noncomputable def x4 : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))

theorem arrange_in_order : 
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 := 
by 
  sorry

end arrange_in_order_l87_87806


namespace desired_yearly_income_l87_87487

theorem desired_yearly_income (total_investment : ℝ) 
  (investment1 : ℝ) (rate1 : ℝ) 
  (investment2 : ℝ) (rate2 : ℝ) 
  (rate_remainder : ℝ) 
  (h_total : total_investment = 10000) 
  (h_invest1 : investment1 = 4000)
  (h_rate1 : rate1 = 0.05) 
  (h_invest2 : investment2 = 3500)
  (h_rate2 : rate2 = 0.04)
  (h_rate_remainder : rate_remainder = 0.064)
  : (rate1 * investment1 + rate2 * investment2 + rate_remainder * (total_investment - (investment1 + investment2))) = 500 := 
by
  sorry

end desired_yearly_income_l87_87487


namespace geo_seq_sum_eq_l87_87983

variable {a : ℕ → ℝ}

-- Conditions
def is_geo_seq (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r
def positive_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > 0
def specific_eq (a : ℕ → ℝ) : Prop := a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 25

theorem geo_seq_sum_eq (a : ℕ → ℝ) (h_geo : is_geo_seq a) (h_pos : positive_seq a) (h_eq : specific_eq a) : 
  a 2 + a 4 = 5 :=
by
  sorry

end geo_seq_sum_eq_l87_87983


namespace original_money_l87_87662
noncomputable def original_amount (x : ℝ) :=
  let after_first_loss := (2/3) * x
  let after_first_win := after_first_loss + 10
  let after_second_loss := after_first_win - (1/3) * after_first_win
  let after_second_win := after_second_loss + 20
  after_second_win

theorem original_money (x : ℝ) (h : original_amount x = x) : x = 48 :=
by {
  sorry
}

end original_money_l87_87662


namespace train_speed_in_kmph_l87_87344

noncomputable def motorbike_speed : ℝ := 64
noncomputable def overtaking_time : ℝ := 40
noncomputable def train_length_meters : ℝ := 400.032

theorem train_speed_in_kmph :
  let train_length_km := train_length_meters / 1000
  let overtaking_time_hours := overtaking_time / 3600
  let relative_speed := train_length_km / overtaking_time_hours
  let train_speed := motorbike_speed + relative_speed
  train_speed = 100.00288 := by
  sorry

end train_speed_in_kmph_l87_87344


namespace troy_needs_additional_money_l87_87780

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l87_87780


namespace evaluate_expression_l87_87361

def cyclical_i (z : ℂ) : Prop := z^4 = 1

theorem evaluate_expression (i : ℂ) (h : cyclical_i i) : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end evaluate_expression_l87_87361


namespace woman_work_time_l87_87927

theorem woman_work_time :
  ∀ (M W B : ℝ), (M = 1/6) → (B = 1/12) → (M + W + B = 1/3) → (W = 1/12) → (1 / W = 12) :=
by
  intros M W B hM hB h_combined hW
  sorry

end woman_work_time_l87_87927


namespace interest_rate_calculation_l87_87639

theorem interest_rate_calculation
  (P : ℕ) 
  (I : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (principal : P = 9200) 
  (time : T = 3) 
  (interest_diff : P - 5888 = I) 
  (interest_formula : I = P * R * T / 100) 
  : R = 12 :=
sorry

end interest_rate_calculation_l87_87639


namespace greatest_product_sum_2000_eq_1000000_l87_87287

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87287


namespace min_ab_sum_l87_87894

theorem min_ab_sum (a b : ℤ) (h : a * b = 72) : a + b >= -17 :=
by
  sorry

end min_ab_sum_l87_87894


namespace part_I_part_II_l87_87712

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (x a : ℝ) : ℝ := (f x a) + (g x)

theorem part_I (a : ℝ) :
  (∀ x > 0, f x a ≥ g x) → a ≤ 0.5 :=
by
  sorry

theorem part_II (a x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (hx1_lt_half : x1 < 0.5) :
  (h x1 a = 2 * x1^2 + Real.log x1) →
  (h x2 a = 2 * x2^2 + Real.log x2) →
  (x1 * x2 = 0.5) →
  h x1 a - h x2 a > (3 / 4) - Real.log 2 :=
by
  sorry

end part_I_part_II_l87_87712


namespace total_rent_correct_recoup_investment_period_maximize_average_return_l87_87942

noncomputable def initialInvestment := 720000
noncomputable def firstYearRent := 54000
noncomputable def annualRentIncrease := 4000
noncomputable def maxRentalPeriod := 40

-- Conditions on the rental period
variable (x : ℝ) (hx : 0 < x ∧ x ≤ 40)

-- Function for total rent after x years
noncomputable def total_rent (x : ℝ) := 0.2 * x^2 + 5.2 * x

-- Condition for investment recoup period
noncomputable def recoupInvestmentTime := ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment

-- Function for transfer price
noncomputable def transfer_price (x : ℝ) := -0.3 * x^2 + 10.56 * x + 57.6

-- Function for average return on investment
noncomputable def annual_avg_return (x : ℝ) := (transfer_price x + total_rent x - initialInvestment) / x

-- Statement of theorems
theorem total_rent_correct (x : ℝ) (hx : 0 < x ∧ x ≤ 40) :
  total_rent x = 0.2 * x^2 + 5.2 * x := sorry

theorem recoup_investment_period :
  ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment := sorry

theorem maximize_average_return :
  ∃ x : ℝ, x = 12 ∧ (∀ y : ℝ, annual_avg_return x ≥ annual_avg_return y) := sorry

end total_rent_correct_recoup_investment_period_maximize_average_return_l87_87942


namespace polynomial_remainder_l87_87542

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 + 2*x + 3

-- Define the divisor q(x)
def q (x : ℝ) : ℝ := x + 2

-- The theorem asserting the remainder when p(x) is divided by q(x)
theorem polynomial_remainder : (p (-2)) = -9 :=
by
  sorry

end polynomial_remainder_l87_87542


namespace samantha_tenth_finger_l87_87760

def g (x : ℕ) : ℕ :=
  match x with
  | 2 => 2
  | _ => 0  -- Assume a simple piecewise definition for the sake of the example.

theorem samantha_tenth_finger : g (2) = 2 :=
by  sorry

end samantha_tenth_finger_l87_87760


namespace graphs_intersect_exactly_eight_times_l87_87952

theorem graphs_intersect_exactly_eight_times (A : ℝ) (hA : 0 < A) :
  ∃ (count : ℕ), count = 8 ∧ ∀ x y : ℝ, y = A * x ^ 4 → y ^ 2 + 5 = x ^ 2 + 6 * y :=
sorry

end graphs_intersect_exactly_eight_times_l87_87952


namespace investment_period_l87_87262

theorem investment_period (P : ℝ) (r1 r2 : ℝ) (diff : ℝ) (t : ℝ) :
  P = 900 ∧ r1 = 0.04 ∧ r2 = 0.045 ∧ (P * r2 * t) - (P * r1 * t) = 31.50 → t = 7 :=
by
  sorry

end investment_period_l87_87262


namespace perpendicular_line_through_point_l87_87124

open Real

def line (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) :
  (line 2 1 (-5) x y) → (x = 3) ∧ (y = 0) → (line 1 (-2) 3 x y) := by
sorry

end perpendicular_line_through_point_l87_87124


namespace find_selling_price_l87_87413

variable (SP CP : ℝ)

def original_selling_price (SP CP : ℝ) : Prop :=
  0.9 * SP = CP + 0.08 * CP

theorem find_selling_price (h1 : CP = 17500)
  (h2 : original_selling_price SP CP) : SP = 21000 :=
by
  sorry

end find_selling_price_l87_87413


namespace square_side_length_exists_l87_87129

-- Define the dimensions of the tile
structure Tile where
  width : Nat
  length : Nat

-- Define the specific tile used in the problem
def given_tile : Tile :=
  { width := 16, length := 24 }

-- Define the condition of forming a square using 6 tiles
def forms_square_with_6_tiles (tile : Tile) (side_length : Nat) : Prop :=
  (2 * tile.length = side_length) ∧ (3 * tile.width = side_length)

-- Problem statement requiring proof
theorem square_side_length_exists : forms_square_with_6_tiles given_tile 48 :=
  sorry

end square_side_length_exists_l87_87129


namespace reciprocal_of_neg_2023_l87_87007

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87007


namespace total_goals_by_other_players_l87_87165

theorem total_goals_by_other_players (total_players goals_season games_played : ℕ)
  (third_players_goals avg_goals_per_third : ℕ)
  (h1 : total_players = 24)
  (h2 : goals_season = 150)
  (h3 : games_played = 15)
  (h4 : third_players_goals = total_players / 3)
  (h5 : avg_goals_per_third = 1)
  : (goals_season - (third_players_goals * avg_goals_per_third * games_played)) = 30 :=
by
  sorry

end total_goals_by_other_players_l87_87165


namespace original_price_l87_87334

theorem original_price (x : ℝ) (h1 : 0.75 * x + 12 = x - 12) (h2 : 0.90 * x - 42 = x - 12) : x = 360 :=
by
  sorry

end original_price_l87_87334


namespace max_product_of_two_integers_sum_2000_l87_87298

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87298


namespace perpendicular_lines_slope_l87_87375

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 1 - a ∧ (a - 2) * x + 3 * y + 2 = 0) → a = 1 / 2 := 
by 
  sorry

end perpendicular_lines_slope_l87_87375


namespace rabbit_jumps_before_dog_catches_l87_87338

/-- Prove that the number of additional jumps the rabbit can make before the dog catches up is 700,
    given the initial conditions:
      1. The rabbit has a 50-jump head start.
      2. The dog makes 5 jumps in the time the rabbit makes 6 jumps.
      3. The distance covered by 7 jumps of the dog equals the distance covered by 9 jumps of the rabbit. -/
theorem rabbit_jumps_before_dog_catches (h_head_start : ℕ) (h_time_ratio : ℚ) (h_distance_ratio : ℚ) : 
    h_head_start = 50 → h_time_ratio = 5/6 → h_distance_ratio = 7/9 → 
    ∃ (rabbit_additional_jumps : ℕ), rabbit_additional_jumps = 700 :=
by
  intro h_head_start_intro h_time_ratio_intro h_distance_ratio_intro
  have rabbit_additional_jumps := 700
  use rabbit_additional_jumps
  sorry

end rabbit_jumps_before_dog_catches_l87_87338


namespace frac_calc_l87_87835

theorem frac_calc : (2 / 9) * (5 / 11) + 1 / 3 = 43 / 99 :=
by sorry

end frac_calc_l87_87835


namespace bathroom_area_l87_87909

-- Definitions based on conditions
def totalHouseArea : ℝ := 1110
def numBedrooms : ℕ := 4
def bedroomArea : ℝ := 11 * 11
def kitchenArea : ℝ := 265
def numBathrooms : ℕ := 2

-- Mathematically equivalent proof problem
theorem bathroom_area :
  let livingArea := kitchenArea  -- living area is equal to kitchen area
  let totalRoomArea := numBedrooms * bedroomArea + kitchenArea + livingArea
  let remainingArea := totalHouseArea - totalRoomArea
  let bathroomArea := remainingArea / numBathrooms
  bathroomArea = 48 :=
by
  repeat { sorry }

end bathroom_area_l87_87909


namespace distance_to_nearest_river_l87_87670

theorem distance_to_nearest_river (d : ℝ) (h₁ : ¬ (d ≤ 12)) (h₂ : ¬ (d ≥ 15)) (h₃ : ¬ (d ≥ 10)) :
  12 < d ∧ d < 15 :=
by 
  sorry

end distance_to_nearest_river_l87_87670


namespace min_value_expression_l87_87202

theorem min_value_expression (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) : (x * y + x^2) ≥ 4 :=
sorry

end min_value_expression_l87_87202


namespace find_third_side_l87_87161

def vol_of_cube (side : ℝ) : ℝ := side ^ 3

def vol_of_box (length width height : ℝ) : ℝ := length * width * height

theorem find_third_side (n : ℝ) (vol_cube : ℝ) (num_cubes : ℝ) (l w : ℝ) (vol_box : ℝ) :
  num_cubes = 24 →
  vol_cube = 27 →
  l = 8 →
  w = 12 →
  vol_box = num_cubes * vol_cube →
  vol_box = vol_of_box l w n →
  n = 6.75 :=
by
  intros hcubes hc_vol hl hw hvbox1 hvbox2
  -- The proof goes here
  sorry

end find_third_side_l87_87161


namespace abs_neg_product_eq_product_l87_87994

variable (a b : ℝ)

theorem abs_neg_product_eq_product (h1 : a < 0) (h2 : 0 < b) : |-a * b| = a * b := by
  sorry

end abs_neg_product_eq_product_l87_87994


namespace remainder_when_690_div_170_l87_87770

theorem remainder_when_690_div_170 :
  ∃ r : ℕ, ∃ k l : ℕ, 
    gcd (690 - r) (875 - 25) = 170 ∧
    r = 690 % 170 ∧
    l = 875 / 170 ∧
    r = 10 :=
by 
  sorry

end remainder_when_690_div_170_l87_87770


namespace matrix_system_solution_range_l87_87714

theorem matrix_system_solution_range (m : ℝ) :
  (∃ x y: ℝ, 
    (m * x + y = m + 1) ∧ 
    (x + m * y = 2 * m)) ↔ m ≠ -1 :=
by
  sorry

end matrix_system_solution_range_l87_87714


namespace joyce_initial_eggs_l87_87086

theorem joyce_initial_eggs :
  ∃ E : ℕ, (E + 6 = 14) ∧ E = 8 :=
sorry

end joyce_initial_eggs_l87_87086


namespace area_triangle_QDA_l87_87838

-- Define the points
def Q : ℝ × ℝ := (0, 15)
def A (q : ℝ) : ℝ × ℝ := (q, 15)
def D (p : ℝ) : ℝ × ℝ := (0, p)

-- Define the conditions
variable (q : ℝ) (p : ℝ)
variable (hq : q > 0) (hp : p < 15)

-- Theorem stating the area of the triangle QDA in terms of q and p
theorem area_triangle_QDA : 
  1 / 2 * q * (15 - p) = 1 / 2 * q * (15 - p) :=
by sorry

end area_triangle_QDA_l87_87838


namespace metallic_sheet_width_l87_87817

theorem metallic_sheet_width (length : ℝ) (side : ℝ) (volume : ℝ) (width : ℝ) :
  length = 48 → side = 8 → volume = 5120 → volume = (length - 2 * side) * (width - 2 * side) * side → width = 36 :=
by
  intro h_length h_side h_volume h_eq
  have h1 : length - 2 * side = 32 := by sorry
  have h2 : side = 8 := h_side
  have h3 : h_volume = (32) * (width - 16) * 8 := by sorry
  have h4 : width - 16 = 20 := by sorry
  show width = 36 from by sorry

end metallic_sheet_width_l87_87817


namespace cost_of_parts_l87_87735

theorem cost_of_parts (C : ℝ) 
  (h1 : ∀ n ∈ List.range 60, (1.4 * C * n) = (1.4 * C * 60))
  (h2 : 5000 + 3000 = 8000)
  (h3 : 60 * C * 1.4 - (60 * C + 8000) = 11200) : 
  C = 800 := by
  sorry

end cost_of_parts_l87_87735


namespace greatest_product_sum_2000_eq_1000000_l87_87290

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87290


namespace greatest_distance_is_correct_l87_87752

-- Define the coordinates of the post.
def post_coordinate : ℝ × ℝ := (6, -2)

-- Define the length of the rope.
def rope_length : ℝ := 12

-- Define the origin.
def origin : ℝ × ℝ := (0, 0)

-- Define the formula to calculate the Euclidean distance between two points in ℝ².
noncomputable def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ := by
  sorry

-- Define the distance from the origin to the post.
noncomputable def distance_origin_to_post : ℝ := euclidean_distance origin post_coordinate

-- Define the greatest distance the dog can be from the origin.
noncomputable def greatest_distance_from_origin : ℝ := distance_origin_to_post + rope_length

-- Prove that the greatest distance the dog can be from the origin is 12 + 2 * sqrt 10.
theorem greatest_distance_is_correct : greatest_distance_from_origin = 12 + 2 * Real.sqrt 10 := by
  sorry

end greatest_distance_is_correct_l87_87752


namespace largest_prime_factor_5985_l87_87637

theorem largest_prime_factor_5985 : ∃ p, Nat.Prime p ∧ p ∣ 5985 ∧ ∀ q, Nat.Prime q ∧ q ∣ 5985 → q ≤ p :=
sorry

end largest_prime_factor_5985_l87_87637


namespace reciprocal_of_neg_2023_l87_87001

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87001


namespace wheel_diameter_calculation_l87_87667

def total_distance : ℝ := 1056
def revolutions : ℝ := 8.007279344858963
def correct_diameter : ℝ := 41.975

theorem wheel_diameter_calculation 
  (h1 : revolutions ≠ 0) : 
  ((total_distance / revolutions) / Real.pi) ≈ correct_diameter :=
by 
  sorry

end wheel_diameter_calculation_l87_87667


namespace no_such_nat_n_l87_87220

theorem no_such_nat_n :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → (10 * (10 * a + n) + b) % (10 * a + b) = 0 :=
by
  sorry

end no_such_nat_n_l87_87220


namespace gcd_48_180_l87_87923

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  have f1 : 48 = 2^4 * 3 := by norm_num
  have f2 : 180 = 2^2 * 3^2 * 5 := by norm_num
  sorry

end gcd_48_180_l87_87923


namespace M_inter_N_eq_l87_87064

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem M_inter_N_eq : {x | -2 < x ∧ x < 2} = M ∩ N := by
  sorry

end M_inter_N_eq_l87_87064


namespace prove_true_statement_l87_87796

-- Definitions based on conditions in the problem
def A_statement := ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0

-- Equivalent proof problem in Lean 4
theorem prove_true_statement : A_statement :=
by
  sorry

end prove_true_statement_l87_87796


namespace no_three_in_range_l87_87546

theorem no_three_in_range (c : ℝ) : c > 4 → ¬ (∃ x : ℝ, x^2 + 2 * x + c = 3) :=
by
  sorry

end no_three_in_range_l87_87546


namespace derivative_at_pi_over_4_l87_87556

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 :
  deriv f (π / 4) = (Real.sqrt 2 / 2) + (Real.sqrt 2 * π / 8) :=
by
  -- Since the focus is only on the statement, the proof is not required.
  sorry

end derivative_at_pi_over_4_l87_87556


namespace president_vice_secretary_choice_l87_87881

theorem president_vice_secretary_choice (n : ℕ) (h : n = 6) :
  (∀ a b c : fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (n * (n - 1) * (n - 2) = 120) := 
sorry

end president_vice_secretary_choice_l87_87881


namespace min_packs_120_cans_l87_87757

theorem min_packs_120_cans (p8 p16 p32 : ℕ) (total_cans packs_needed : ℕ) :
  total_cans = 120 →
  p8 * 8 + p16 * 16 + p32 * 32 = total_cans →
  packs_needed = p8 + p16 + p32 →
  (∀ (q8 q16 q32 : ℕ), q8 * 8 + q16 * 16 + q32 * 32 = total_cans → q8 + q16 + q32 ≥ packs_needed) →
  packs_needed = 5 :=
by {
  sorry
}

end min_packs_120_cans_l87_87757


namespace percent_increase_fifth_triangle_l87_87675

noncomputable def initial_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.2
noncomputable def num_triangles : ℕ := 5

noncomputable def side_length (n : ℕ) : ℝ :=
  initial_side_length * growth_factor ^ (n - 1)

noncomputable def perimeter_length (n : ℕ) : ℝ :=
  3 * side_length n

noncomputable def percent_increase (n : ℕ) : ℝ :=
  ((perimeter_length n / perimeter_length 1) - 1) * 100

theorem percent_increase_fifth_triangle :
  percent_increase 5 = 107.4 :=
by
  sorry

end percent_increase_fifth_triangle_l87_87675


namespace problem_lean_l87_87876

noncomputable def a : ℕ+ → ℝ := sorry

theorem problem_lean :
  a 11 = 1 / 52 ∧ (∀ n : ℕ+, 1 / a (n + 1) - 1 / a n = 5) → a 1 = 1 / 2 :=
by
  sorry

end problem_lean_l87_87876


namespace max_product_of_two_integers_sum_2000_l87_87311

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87311


namespace relationship_y1_y2_y3_l87_87988

-- Define the function y = 3(x + 1)^2 - 8
def quadratic_fn (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define points A, B, and C on the graph of the quadratic function
def y1 := quadratic_fn 1
def y2 := quadratic_fn 2
def y3 := quadratic_fn (-2)

-- The goal is to prove the relationship y2 > y1 > y3
theorem relationship_y1_y2_y3 :
  y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_y1_y2_y3_l87_87988


namespace addition_in_sets_l87_87585

theorem addition_in_sets (a b : ℤ) (hA : ∃ k : ℤ, a = 2 * k) (hB : ∃ k : ℤ, b = 2 * k + 1) : ∃ k : ℤ, a + b = 2 * k + 1 :=
by
  sorry

end addition_in_sets_l87_87585


namespace cannot_determine_total_inhabitants_without_additional_info_l87_87727

variable (T : ℝ) (M F : ℝ)

axiom inhabitants_are_males_females : M + F = 1
axiom twenty_percent_of_males_are_literate : M * 0.20 * T = 0.20 * M * T
axiom twenty_five_percent_of_all_literates : 0.25 = 0.25 * T / T
axiom thirty_two_five_percent_of_females_are_literate : F = 1 - M ∧ F * 0.325 * T = 0.325 * (1 - M) * T

theorem cannot_determine_total_inhabitants_without_additional_info :
  ∃ (T : ℝ), True ↔ False := by
  sorry

end cannot_determine_total_inhabitants_without_additional_info_l87_87727


namespace tank_capacity_l87_87934

theorem tank_capacity (C : ℝ) (rate_leak : ℝ) (rate_inlet : ℝ) (combined_rate_empty : ℝ) :
  rate_leak = C / 3 ∧ rate_inlet = 6 * 60 ∧ combined_rate_empty = C / 12 →
  C = 864 :=
by
  intros h
  sorry

end tank_capacity_l87_87934


namespace value_of_expression_l87_87466

theorem value_of_expression :
  (43 + 15)^2 - (43^2 + 15^2) = 2 * 43 * 15 :=
by
  sorry

end value_of_expression_l87_87466


namespace compute_fraction_l87_87353

theorem compute_fraction :
  ( (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400) ) /
  ( (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400) ) = 799 := 
by
  sorry

end compute_fraction_l87_87353


namespace area_region_sum_l87_87771

theorem area_region_sum (r1 r2 : ℝ) (angle : ℝ) (a b c : ℕ) : 
  r1 = 6 → r2 = 3 → angle = 30 → (54 * Real.sqrt 3 + (9 : ℝ) * Real.pi - (9 : ℝ) * Real.pi = a * Real.sqrt b + c * Real.pi) → a + b + c = 10 :=
by
  intros
  -- We fill this with the actual proof steps later
  sorry

end area_region_sum_l87_87771


namespace find_y_l87_87977

theorem find_y (x y : ℤ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end find_y_l87_87977


namespace number_of_terms_in_expanded_polynomial_l87_87957

theorem number_of_terms_in_expanded_polynomial : 
  ∀ (a : Fin 4 → Type) (b : Fin 2 → Type) (c : Fin 3 → Type), 
  (4 * 2 * 3 = 24) := 
by
  intros a b c
  sorry

end number_of_terms_in_expanded_polynomial_l87_87957


namespace tens_digit_N_to_20_l87_87584

theorem tens_digit_N_to_20 (N : ℕ) (h1 : Even N) (h2 : ¬(∃ k : ℕ, N = 10 * k)) : 
  ((N ^ 20) / 10) % 10 = 7 := 
by 
  sorry

end tens_digit_N_to_20_l87_87584


namespace problem_1_problem_2_l87_87864

variable {c : ℝ}

def p (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → c ^ x₁ > c ^ x₂

def q (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, (1 / 2) < x₁ ∧ x₁ < x₂ → (x₁ ^ 2 - 2 * c * x₁ + 1) < (x₂ ^ 2 - 2 * c * x₂ + 1)

theorem problem_1 (hc : 0 < c) (hcn1 : c ≠ 1) (hp : p c) (hnq_false : ¬ ¬ q c) : 0 < c ∧ c ≤ 1 / 2 :=
by
  sorry

theorem problem_2 (hc : 0 < c) (hcn1 : c ≠ 1) (hpq_false : ¬ (p c ∧ q c)) (hp_or_q : p c ∨ q c) : 1 / 2 < c ∧ c < 1 :=
by
  sorry

end problem_1_problem_2_l87_87864


namespace lowest_possible_price_l87_87030

def typeADiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 15 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 20 / 100
  discountedPrice - additionalDiscount

def typeBDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 25 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 15 / 100
  discountedPrice - additionalDiscount

def typeCDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 30 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 10 / 100
  discountedPrice - additionalDiscount

def finalPrice (discountedPrice : ℕ) : ℕ :=
  let tax := discountedPrice * 7 / 100
  discountedPrice + tax

theorem lowest_possible_price : 
  min (finalPrice (typeADiscountedPrice 4500)) 
      (min (finalPrice (typeBDiscountedPrice 5500)) 
           (finalPrice (typeCDiscountedPrice 5000))) = 3274 :=
by {
  sorry
}

end lowest_possible_price_l87_87030


namespace intersection_M_N_l87_87715

-- Define the universal set U, and subsets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- Prove that the intersection of M and N is as stated
theorem intersection_M_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 1} :=
by
  -- This is where the proof would go
  sorry

end intersection_M_N_l87_87715


namespace problem_solution_l87_87488

def cylinder_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def probability_inside_sphere_in_cylinder : ℝ :=
  let cyl_vol := cylinder_volume 2 4
  let sph_vol := sphere_volume 2
  sph_vol / cyl_vol

theorem problem_solution :
  probability_inside_sphere_in_cylinder = 2 / 3 := by
  sorry

end problem_solution_l87_87488


namespace total_number_of_bricks_l87_87340

/-- Given bricks of volume 80 unit cubes and 42 unit cubes,
 and a box of volume 1540 unit cubes,
 prove the total number of bricks that can fill the box exactly is 24. -/
theorem total_number_of_bricks (x y : ℕ) (vol_a vol_b total_vol : ℕ)
  (vol_a_def : vol_a = 80)
  (vol_b_def : vol_b = 42)
  (total_vol_def : total_vol = 1540)
  (volume_filled : x * vol_a + y * vol_b = total_vol) :
  x + y = 24 :=
  sorry

end total_number_of_bricks_l87_87340


namespace probability_either_A1_or_B1_not_both_is_half_l87_87251

-- Definitions of the students
inductive Student
| A : ℕ → Student
| B : ℕ → Student
| C : ℕ → Student

-- Excellent grades students
def math_students := [Student.A 1, Student.A 2, Student.A 3]
def physics_students := [Student.B 1, Student.B 2]
def chemistry_students := [Student.C 1, Student.C 2]

-- Total number of ways to select one student from each category
def total_ways : ℕ := 3 * 2 * 2

-- Number of ways either A_1 or B_1 is selected but not both
def special_ways : ℕ := 1 * 1 * 2 + 2 * 1 * 2

-- Probability calculation
def probability := (special_ways : ℚ) / total_ways

-- Theorem to be proven
theorem probability_either_A1_or_B1_not_both_is_half :
  probability = 1 / 2 := by
  sorry

end probability_either_A1_or_B1_not_both_is_half_l87_87251


namespace Ursula_hot_dogs_l87_87920

theorem Ursula_hot_dogs 
  (H : ℕ) 
  (cost_hot_dog : ℚ := 1.50) 
  (cost_salad : ℚ := 2.50) 
  (num_salads : ℕ := 3) 
  (total_money : ℚ := 20) 
  (change : ℚ := 5) :
  (cost_hot_dog * H + cost_salad * num_salads = total_money - change) → H = 5 :=
by
  sorry

end Ursula_hot_dogs_l87_87920


namespace find_common_ratio_l87_87971

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end find_common_ratio_l87_87971


namespace gcd_polynomial_l87_87551

theorem gcd_polynomial (b : ℤ) (h : 2142 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 :=
sorry

end gcd_polynomial_l87_87551


namespace repeating_decimal_to_fraction_l87_87513

theorem repeating_decimal_to_fraction :
  let x := 0.431431431 + 0.000431431431 + 0.000000431431431
  let y := 0.4 + x
  y = 427 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l87_87513


namespace problem_f_increasing_l87_87710

theorem problem_f_increasing (a : ℝ) 
  (h1 : ∀ x, 2 ≤ x → 0 < x^2 - a * x + 3 * a) 
  (h2 : ∀ x, 2 ≤ x → 0 ≤ 2 * x - a) : 
  -4 < a ∧ a ≤ 4 := by
  sorry

end problem_f_increasing_l87_87710


namespace sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l87_87181

-- Problem 1: Define the sum of the first n odd numbers and prove it equals n^2 when n = 5.
theorem sum_first_five_odds_equals_25 : (1 + 3 + 5 + 7 + 9 = 5^2) := 
sorry

-- Problem 2: Prove that if the smallest number in the decomposition of m^3 is 21, then m = 5.
theorem smallest_in_cube_decomposition_eq_21 : 
  (∃ m : ℕ, m > 0 ∧ 21 = 2 * m - 1 ∧ m = 5) := 
sorry

end sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l87_87181


namespace determine_digit_phi_l87_87683

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end determine_digit_phi_l87_87683


namespace determine_a_l87_87406

theorem determine_a (a : ℝ) (h : ∃ r : ℝ, (a / (1+1*I : ℂ) + (1+1*I : ℂ) / 2).im = 0) : a = 1 :=
sorry

end determine_a_l87_87406


namespace sum_of_odd_powers_l87_87188

variable (x y z a : ℝ) (k : ℕ)

theorem sum_of_odd_powers (h1 : x + y + z = a) (h2 : x^3 + y^3 + z^3 = a^3) (hk : k % 2 = 1) : 
  x^k + y^k + z^k = a^k :=
sorry

end sum_of_odd_powers_l87_87188


namespace order_of_abc_l87_87199

noncomputable def a := Real.log 1.2
noncomputable def b := (11 / 10) - (10 / 11)
noncomputable def c := 1 / (5 * Real.exp 0.1)

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l87_87199


namespace calculate_expression_l87_87314

theorem calculate_expression : 
  let a := 0.82
  let b := 0.1
  a^3 - b^3 / (a^2 + 0.082 + b^2) = 0.7201 := sorry

end calculate_expression_l87_87314


namespace geometric_sequence_a6_l87_87371

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h1 : a 4 * a 8 = 9) 
  (h2 : a 4 + a 8 = 8) 
  (geom_seq : ∀ n m, a (n + m) = a n * a m): 
  a 6 = 3 :=
by
  -- skipped proof
  sorry

end geometric_sequence_a6_l87_87371


namespace part_I_part_II_part_III_l87_87986

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Part (I)
theorem part_I (a : ℝ) (h_a : a = 1) : 
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 0 ∧ f x a * g x = 1 := sorry

-- Part (II)
theorem part_II (a : ℝ) (h_a : a = -1) (k : ℝ) :
  (∃ x : ℝ, f x a = k * g x ∧ ∀ y : ℝ, y ≠ x → f y a ≠ k * g y) ↔ 
  (k > 3 * Real.exp (-2) ∨ (0 < k ∧ k < 1 * Real.exp (-1))) := sorry

-- Part (III)
theorem part_III (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), (x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂) →
  abs (f x₁ a - f x₂ a) < abs (g x₁ - g x₂)) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) := sorry

end part_I_part_II_part_III_l87_87986


namespace complex_magnitude_l87_87547

open Complex

theorem complex_magnitude {x y : ℝ} (h : (1 + Complex.I) * x = 1 + y * Complex.I) : abs (x + y * Complex.I) = Real.sqrt 2 :=
sorry

end complex_magnitude_l87_87547


namespace find_k_l87_87088

theorem find_k (k : ℝ) (h1 : k > 1) 
(h2 : ∑' n : ℕ, (7 * (n + 1) - 3) / k^(n + 1) = 2) : 
  k = 2 + 3 * Real.sqrt 2 / 2 := 
sorry

end find_k_l87_87088


namespace polynomial_real_root_l87_87101

variable {A B C D E : ℝ}

theorem polynomial_real_root
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
by
  sorry

end polynomial_real_root_l87_87101


namespace sum_of_two_numbers_l87_87130

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  x + y = (16 * Real.sqrt 3) / 3 := 
sorry

end sum_of_two_numbers_l87_87130


namespace total_cards_l87_87652

theorem total_cards (H F B : ℕ) (hH : H = 200) (hF : F = 4 * H) (hB : B = F - 50) : H + F + B = 1750 := 
by 
  sorry

end total_cards_l87_87652


namespace bus_stops_12_minutes_per_hour_l87_87690

noncomputable def stopping_time (speed_excluding_stoppages : ℝ) (speed_including_stoppages : ℝ) : ℝ :=
  let distance_lost_per_hour := speed_excluding_stoppages - speed_including_stoppages
  let speed_per_minute := speed_excluding_stoppages / 60
  distance_lost_per_hour / speed_per_minute

theorem bus_stops_12_minutes_per_hour :
  stopping_time 50 40 = 12 :=
by
  sorry

end bus_stops_12_minutes_per_hour_l87_87690


namespace find_number_A_l87_87774

theorem find_number_A (A B : ℝ) (h₁ : A + B = 14.85) (h₂ : B = 10 * A) : A = 1.35 :=
sorry

end find_number_A_l87_87774


namespace black_haired_girls_count_l87_87588

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l87_87588


namespace largest_rhombus_diagonal_in_circle_l87_87874

theorem largest_rhombus_diagonal_in_circle (r : ℝ) (h : r = 10) : (2 * r = 20) :=
by
  sorry

end largest_rhombus_diagonal_in_circle_l87_87874


namespace solve_for_x_l87_87607

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l87_87607


namespace least_prime_in_sum_even_set_of_7_distinct_primes_l87_87408

noncomputable def is_prime (n : ℕ) : Prop := sorry -- Assume an implementation of prime numbers

theorem least_prime_in_sum_even_set_of_7_distinct_primes {q : Finset ℕ} 
  (hq_distinct : q.card = 7) 
  (hq_primes : ∀ n ∈ q, is_prime n) 
  (hq_sum_even : q.sum id % 2 = 0) :
  ∃ m ∈ q, m = 2 :=
by
  sorry

end least_prime_in_sum_even_set_of_7_distinct_primes_l87_87408


namespace union_of_sets_l87_87189

theorem union_of_sets (A B : Set ℤ) (hA : A = {-1, 3}) (hB : B = {2, 3}) : A ∪ B = {-1, 2, 3} := 
by
  sorry

end union_of_sets_l87_87189


namespace max_edges_dodecahedron_no_shared_vertices_l87_87638

noncomputable def dodecahedron := sorry -- Placeholder for actual dodecahedron graph definition

theorem max_edges_dodecahedron_no_shared_vertices (G : SimpleGraph) :
  G = dodecahedron ->
  ∀ (E : Finset G.Edge), (∀ (e1 e2 : G.Edge), e1 ≠ e2 ∧ (e1 ∩ e2).Nonempty → False) →
  E.card ≤ 10 ∧ (∃ (E' : Finset G.Edge), E'.card = 10 ∧ ∀ (e1 e2 : G.Edge), e1 ≠ e2 ∧ (e1 ∩ e2).Nonempty → False) :=
sorry

end max_edges_dodecahedron_no_shared_vertices_l87_87638


namespace price_of_each_sundae_l87_87808

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℝ)
  (price_per_ice_cream_bar : ℝ)
  (total_cost_for_sundaes : ℝ) :
  num_ice_cream_bars = 225 →
  num_sundaes = 125 →
  total_price = 200 →
  price_per_ice_cream_bar = 0.60 →
  total_cost_for_sundaes = total_price - (num_ice_cream_bars * price_per_ice_cream_bar) →
  (total_cost_for_sundaes / num_sundaes) = 0.52 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_each_sundae_l87_87808


namespace point_not_in_second_quadrant_l87_87192

-- Define the point P and the condition
def point_is_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def point (m : ℝ) : ℝ × ℝ :=
  (m + 1, m)

-- The main theorem stating that P cannot be in the second quadrant
theorem point_not_in_second_quadrant (m : ℝ) : ¬ point_is_in_second_quadrant (point m) :=
by
  sorry

end point_not_in_second_quadrant_l87_87192


namespace find_ellipse_eq_find_circle_eq_l87_87056

-- For the given problem conditions

def eccentricity := (c a : ℝ) : ℝ := c / a

/-- 
Existence of ellipse with given properties.
- The equation of ellipse C passing through the point (1, 3/2)
- Ellipse Conditions: 
  center of symmetry at origin O, 
  foci on x-axis, 
  eccentricity 1/2 
  C passes through (1, 3/2)
-/
def ellipse_eq (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (b^2 = a^2 - (a/2)^2) ∧ (x^2 / a^2 + (y^2 / b^2) = 1)

/--
Finding the equation of the ellipse
-/
theorem find_ellipse_eq : 
  ellipse_eq 1 (3/2) → 
  (∃ a b : ℝ, a = 2 ∧ b^2 = 3 ∧ (∃ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)) :=
begin
  -- proof outline
  sorry
end

/--
Finding the equation of circle tangent to a line
-/
def circle_at_origin (x y : ℝ) : Prop :=
  x^2 + y^2 = 1 / 2

/--
The circle that is tangent to the line passing through the left focus
-/
theorem find_circle_eq 
  (area_triangle : ℝ := 6 * real.sqrt 2 / 7)
  (l : ℝ → ℝ)
  (h_l : ∃ k : ℝ, l = λ x, k * (x + 1))
  (intersect_ellipse : ∃ A B : ℝ × ℝ, A = (-1, - 3 / 2) ∧ B = (-1, 3 / 2)):
  ∃ r : ℝ, r^2 = 1 / 2 :=
begin
  -- proof outline
  sorry
end

end find_ellipse_eq_find_circle_eq_l87_87056


namespace minimum_phi_l87_87105

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 4))

theorem minimum_phi (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = (3/8) * Real.pi - (k * Real.pi / 2)) → φ = (3/8) * Real.pi :=
by
  sorry

end minimum_phi_l87_87105


namespace find_radius_l87_87342

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

theorem find_radius (C1 : ℝ × ℝ × ℝ) (r1 : ℝ) (C2 : ℝ × ℝ × ℝ) (r : ℝ) :
  C1 = (3, 5, 0) →
  r1 = 2 →
  C2 = (0, 5, -8) →
  (sphere ((3, 5, -8) : ℝ × ℝ × ℝ) (2 * Real.sqrt 17)) →
  r = Real.sqrt 59 :=
by
  intros h1 h2 h3 h4
  sorry

end find_radius_l87_87342


namespace greatest_product_sum_2000_l87_87274

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87274


namespace necessary_but_not_sufficient_condition_l87_87896

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((0 < x ∧ x < 5) → (|x - 2| < 3)) ∧ ¬ ((|x - 2| < 3) → (0 < x ∧ x < 5)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l87_87896


namespace psychology_majors_percentage_in_liberal_arts_l87_87501

theorem psychology_majors_percentage_in_liberal_arts 
  (total_students : ℕ) 
  (percent_freshmen : ℝ) 
  (percent_freshmen_liberal_arts : ℝ) 
  (percent_freshmen_psych_majors_liberal_arts : ℝ) 
  (h1: percent_freshmen = 0.40) 
  (h2: percent_freshmen_liberal_arts = 0.50)
  (h3: percent_freshmen_psych_majors_liberal_arts = 0.10) :
  ((percent_freshmen_psych_majors_liberal_arts / (percent_freshmen * percent_freshmen_liberal_arts)) * 100 = 50) :=
by
  sorry

end psychology_majors_percentage_in_liberal_arts_l87_87501


namespace simplify_expression_l87_87465

theorem simplify_expression (a b : ℤ) (h_a : a = 43) (h_b : b = 15) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1290 := 
by
  -- We state the goal that needs to be proven:
  have h_simplified : (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2 * a * b := sorry
  -- Subsequently substituting values for a and b:
  rw [h_a, h_b] at h_simplified
  assumption

end simplify_expression_l87_87465


namespace greatest_product_sum_2000_l87_87272

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87272


namespace common_ratio_of_geometric_series_l87_87973

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end common_ratio_of_geometric_series_l87_87973


namespace line_eq_l87_87468

variables {x x1 x2 y y1 y2 : ℝ}

theorem line_eq (h : x2 ≠ x1 ∧ y2 ≠ y1) : 
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) :=
sorry

end line_eq_l87_87468


namespace exists_real_root_iff_l87_87569

theorem exists_real_root_iff {m : ℝ} :
  (∃x : ℝ, 25 - abs (x + 1) - 4 * 5 - abs (x + 1) - m = 0) ↔ (-3 < m ∧ m < 0) :=
by
  sorry

end exists_real_root_iff_l87_87569


namespace inequality_solution_l87_87975

theorem inequality_solution (x : ℝ) (h : x ≠ 1) : (x + 1) * (x + 3) / (x - 1)^2 ≤ 0 ↔ (-3 ≤ x ∧ x ≤ -1) :=
by
  sorry

end inequality_solution_l87_87975


namespace john_has_dollars_left_l87_87581

-- Definitions based on the conditions
def john_savings_octal : ℕ := 5273
def rental_car_cost_decimal : ℕ := 1500

-- Define the function to convert octal to decimal
def octal_to_decimal (n : ℕ) : ℕ := -- Conversion logic
sorry

-- Statements for the conversion and subtraction
def john_savings_decimal : ℕ := octal_to_decimal john_savings_octal
def amount_left_for_gas_and_accommodations : ℕ :=
  john_savings_decimal - rental_car_cost_decimal

-- Theorem statement equivalent to the correct answer
theorem john_has_dollars_left :
  amount_left_for_gas_and_accommodations = 1247 :=
by sorry

end john_has_dollars_left_l87_87581


namespace problem1_solution_problem2_solution_l87_87987

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

-- Problem 1
theorem problem1_solution :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} :=
sorry

-- Problem 2
theorem problem2_solution :
  ∀ (a : ℝ), (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9 / 4 :=
sorry

end problem1_solution_problem2_solution_l87_87987


namespace exists_nat_m_inequality_for_large_n_l87_87081

section sequence_problem

-- Define the sequence
noncomputable def a (n : ℕ) : ℚ :=
if n = 7 then 16 / 3 else
if n < 7 then 0 else -- hands off values before a7 that are not needed
3 * a (n - 1) / (7 - a (n - 1) + 4)

-- Define the properties to be proven
theorem exists_nat_m {m : ℕ} :
  (∀ n, n > m → a n < 2) ∧ (∀ n, n ≤ m → a n > 2) :=
sorry

theorem inequality_for_large_n (n : ℕ) (hn : n ≥ 10) :
  (a (n - 1) + a n + 1) / 2 < a n :=
sorry

end sequence_problem

end exists_nat_m_inequality_for_large_n_l87_87081


namespace deposit_amount_correct_l87_87831

noncomputable def deposit_amount (initial_amount : ℝ) : ℝ :=
  let first_step := 0.30 * initial_amount
  let second_step := 0.25 * first_step
  0.20 * second_step

theorem deposit_amount_correct :
  deposit_amount 50000 = 750 :=
by
  sorry

end deposit_amount_correct_l87_87831


namespace first_expression_second_expression_l87_87190

open Real 

theorem first_expression (a x y : ℝ) : 
    (-2 * a)^6 * (-3 * a^3) + [2 * a^2]^3 / (1 / (1 / ((-2)^2 * 3^2 * (x * y)^3))) = 
    192 * a^9 + 288 * a^6 * (x * y)^3 := 
sorry

theorem second_expression : 
    abs (-1/8) + π^3 + (- (1/2)^3 - (1/3)^2) = 
    π^3 - 1 / 72 := 
sorry

end first_expression_second_expression_l87_87190


namespace remainder_equality_l87_87094

variables (A B D : ℕ) (S S' s s' : ℕ)

theorem remainder_equality 
  (h1 : A > B) 
  (h2 : (A + 3) % D = S) 
  (h3 : (B - 2) % D = S') 
  (h4 : ((A + 3) * (B - 2)) % D = s) 
  (h5 : (S * S') % D = s') : 
  s = s' := 
sorry

end remainder_equality_l87_87094


namespace max_product_l87_87271

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87271


namespace estimate_points_in_interval_l87_87726

-- Define the conditions
def total_data_points : ℕ := 1000
def frequency_interval : ℝ := 0.16
def interval_estimation : ℝ := total_data_points * frequency_interval

-- Lean theorem statement
theorem estimate_points_in_interval : interval_estimation = 160 :=
by
  sorry

end estimate_points_in_interval_l87_87726


namespace lcm_12_18_l87_87540

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l87_87540


namespace lcm_12_18_l87_87541

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l87_87541


namespace tan_alpha_value_trigonometric_expression_value_l87_87184

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  Real.tan α = 2 :=
sorry

theorem trigonometric_expression_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  (4 * Real.sin (π - α) + 2 * Real.cos (2 * π - α)) / (Real.sin (π / 2 - α) + Real.sin (-α)) = -10 := 
sorry

end tan_alpha_value_trigonometric_expression_value_l87_87184


namespace ellipse_with_foci_on_x_axis_l87_87873

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end ellipse_with_foci_on_x_axis_l87_87873


namespace part1_part2_part3_l87_87884

variable (a b c d S A B C D : ℝ)

-- The given conditions
def cond1 : Prop := a + c = b + d
def cond2 : Prop := A + C = B + D
def cond3 : Prop := S^2 = a * b * c * d

-- The statements to prove
theorem part1 (h1 : cond1 a b c d) (h2 : cond2 A B C D) : cond3 a b c d S := sorry
theorem part2 (h1 : cond1 a b c d) (h3 : cond3 a b c d S) : cond2 A B C D := sorry
theorem part3 (h2 : cond2 A B C D) : cond3 a b c d S := sorry

end part1_part2_part3_l87_87884


namespace power_product_l87_87226

theorem power_product (m n : ℕ) (hm : 2 < m) (hn : 0 < n) : 
  (2^m - 1) * (2^n + 1) > 0 :=
by 
  sorry

end power_product_l87_87226


namespace length_of_DE_l87_87429

variable (A B C X Y Z D E : Type)
variable [LinearOrderedField ℝ]

def base_length_ABC : ℝ := 15
def triangle_area_ratio : ℝ := 0.25

theorem length_of_DE (h1 : DE // BC ∥ BC) 
                    (h2 : triangle_area_ratio * (base_length_ABC ^ 2) = DE ^ 2)
                    : DE = 7.5 :=
sorry

end length_of_DE_l87_87429


namespace sum_of_fifth_powers_52070424_l87_87851

noncomputable def sum_of_fifth_powers (n : ℤ) : ℤ :=
  (n-1)^5 + n^5 + (n+1)^5

theorem sum_of_fifth_powers_52070424 :
  ∃ (n : ℤ), (n-1)^2 + n^2 + (n+1)^2 = 2450 ∧ sum_of_fifth_powers n = 52070424 :=
by
  sorry

end sum_of_fifth_powers_52070424_l87_87851


namespace final_price_is_correct_l87_87437

/-- 
  The original price of a suit is $200.
-/
def original_price : ℝ := 200

/-- 
  The price increased by 25%, therefore the increase is 25% of the original price.
-/
def increase : ℝ := 0.25 * original_price

/-- 
  The new price after the price increase.
-/
def increased_price : ℝ := original_price + increase

/-- 
  After the increase, a 25% off coupon is applied.
-/
def discount : ℝ := 0.25 * increased_price

/-- 
  The final price consumers pay for the suit.
-/
def final_price : ℝ := increased_price - discount

/-- 
  Prove that the consumers paid $187.50 for the suit.
-/
theorem final_price_is_correct : final_price = 187.50 :=
by sorry

end final_price_is_correct_l87_87437


namespace actors_per_group_l87_87879

theorem actors_per_group (actors_per_hour : ℕ) (show_time_per_actor : ℕ) (total_show_time : ℕ)
  (h1 : show_time_per_actor = 15) (h2 : actors_per_hour = 20) (h3 : total_show_time = 60) :
  actors_per_hour * show_time_per_actor / total_show_time = 5 :=
by sorry

end actors_per_group_l87_87879


namespace square_measurement_error_l87_87349

theorem square_measurement_error (S S' : ℝ) (error_percentage : ℝ)
  (area_error_percentage : ℝ) (h1 : area_error_percentage = 2.01) :
  error_percentage = 1 :=
by
  sorry

end square_measurement_error_l87_87349


namespace FourConsecIntsSum34Unique_l87_87993

theorem FourConsecIntsSum34Unique :
  ∃! (a b c d : ℕ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (a + b + c + d = 34) ∧ (d = a + 3) :=
by
  -- The proof will be placed here
  sorry

end FourConsecIntsSum34Unique_l87_87993


namespace jam_cost_l87_87522

theorem jam_cost (N B J H : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 6 * J + 2 * H) = 342) :
  6 * N * J = 270 := 
sorry

end jam_cost_l87_87522


namespace gray_region_area_l87_87576

theorem gray_region_area (r R : ℝ) (hR : R = 3 * r) (h_diff : R - r = 3) :
  π * (R^2 - r^2) = 18 * π :=
by
  -- The proof goes here
  sorry

end gray_region_area_l87_87576


namespace units_digit_47_power_47_l87_87457

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l87_87457


namespace inequality_system_no_solution_l87_87070

theorem inequality_system_no_solution (k x : ℝ) (h₁ : 1 < x ∧ x ≤ 2) (h₂ : x > k) : k ≥ 2 :=
sorry

end inequality_system_no_solution_l87_87070


namespace probability_not_greater_than_two_l87_87252

theorem probability_not_greater_than_two : 
  let cards := [1, 2, 3, 4]
  let favorable_cards := [1, 2]
  let total_scenarios := cards.length
  let favorable_scenarios := favorable_cards.length
  let prob := favorable_scenarios / total_scenarios
  prob = 1 / 2 :=
by
  sorry

end probability_not_greater_than_two_l87_87252


namespace symmetric_axis_of_quadratic_l87_87134

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := (x - 3) * (x + 5)

-- Prove that the symmetric axis of the quadratic function is the line x = -1
theorem symmetric_axis_of_quadratic : ∀ (x : ℝ), quadratic_function x = (x - 3) * (x + 5) → x = -1 :=
by
  intro x h
  sorry

end symmetric_axis_of_quadratic_l87_87134


namespace lcm_12_18_l87_87532

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l87_87532


namespace amount_dog_ate_cost_l87_87403

-- Define the costs of each ingredient
def cost_flour : Real := 4
def cost_sugar : Real := 2
def cost_butter : Real := 2.5
def cost_eggs : Real := 0.5

-- Define the number of slices
def number_of_slices := 6

-- Define the number of slices eaten by Laura's mother
def slices_eaten_by_mother := 2

-- Calculate the total cost of the ingredients
def total_cost := cost_flour + cost_sugar + cost_butter + cost_eggs

-- Calculate the cost per slice
def cost_per_slice := total_cost / number_of_slices

-- Calculate the number of slices eaten by Kevin
def slices_eaten_by_kevin := number_of_slices - slices_eaten_by_mother

-- Define the total cost of slices eaten by Kevin
def cost_eaten_by_kevin := slices_eaten_by_kevin * cost_per_slice

-- The main statement to prove
theorem amount_dog_ate_cost :
  cost_eaten_by_kevin = 6 := by
    sorry

end amount_dog_ate_cost_l87_87403


namespace relay_race_length_correct_l87_87446

def relay_race_length (num_members distance_per_member : ℕ) : ℕ := num_members * distance_per_member

theorem relay_race_length_correct :
  relay_race_length 5 30 = 150 :=
by
  -- The proof would go here
  sorry

end relay_race_length_correct_l87_87446


namespace prob_heart_king_l87_87919

theorem prob_heart_king :
    let total_cards := 52
    let probability_heart := 13 / 52
    let probability_king := 4 / 51 in
    (1 / 52 * 3 / 51 + 12 / 52 * 4 / 51) = 1 / 52 :=
by sorry

end prob_heart_king_l87_87919


namespace greatest_odd_integer_l87_87635

theorem greatest_odd_integer (x : ℕ) (h_odd : x % 2 = 1) (h_pos : x > 0) (h_ineq : x^2 < 50) : x = 7 :=
by sorry

end greatest_odd_integer_l87_87635


namespace solve_equation_l87_87604

theorem solve_equation (x : ℚ) (h1 : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 := by
  sorry

end solve_equation_l87_87604


namespace solve_percentage_chromium_first_alloy_l87_87390

noncomputable def percentage_chromium_first_alloy (x : ℝ) : Prop :=
  let w1 := 15 -- weight of the first alloy
  let c2 := 10 -- percentage of chromium in the second alloy
  let w2 := 35 -- weight of the second alloy
  let w_total := 50 -- total weight of the new alloy formed by mixing
  let c_new := 10.6 -- percentage of chromium in the new alloy
  -- chromium percentage equation
  ((x / 100) * w1 + (c2 / 100) * w2) = (c_new / 100) * w_total

theorem solve_percentage_chromium_first_alloy : percentage_chromium_first_alloy 12 :=
  sorry -- proof goes here

end solve_percentage_chromium_first_alloy_l87_87390


namespace monotonic_increasing_intervals_max_min_values_l87_87061

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x - Real.pi / 3)

theorem monotonic_increasing_intervals (k : ℤ) :
  ∃ (a b : ℝ), a = k * Real.pi - Real.pi / 12 ∧ b = k * Real.pi + 5 * Real.pi / 12 ∧
    ∀ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂ :=
sorry

theorem max_min_values : ∃ (xmin xmax : ℝ) (fmin fmax : ℝ),
  xmin = 0 ∧ fmin = f 0 ∧ fmin = - Real.sqrt 3 / 2 ∧
  xmax = 5 * Real.pi / 12 ∧ fmax = f (5 * Real.pi / 12) ∧ fmax = 1 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 →
    fmin ≤ f x ∧ f x ≤ fmax :=
sorry

end monotonic_increasing_intervals_max_min_values_l87_87061


namespace amy_spent_32_l87_87673

theorem amy_spent_32 (x: ℝ) (h1: 0.15 * x + 1.6 * x + x = 55) : 1.6 * x = 32 :=
by
  sorry

end amy_spent_32_l87_87673


namespace minimum_x_condition_l87_87647

theorem minimum_x_condition (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (h : x - 2 * y = (x + 16 * y) / (2 * x * y)) : 
  x ≥ 4 :=
sorry

end minimum_x_condition_l87_87647


namespace average_expenditure_week_l87_87913

theorem average_expenditure_week (avg_3_days: ℝ) (avg_4_days: ℝ) (total_days: ℝ):
  avg_3_days = 350 → avg_4_days = 420 → total_days = 7 → 
  ((3 * avg_3_days + 4 * avg_4_days) / total_days = 390) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_expenditure_week_l87_87913


namespace solve_for_x_l87_87606

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l87_87606


namespace trapezium_area_l87_87693

theorem trapezium_area (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 10) : 
  (1 / 2) * (a + b) * h = 190 := 
by
  -- We provide the conditions:
  rw [h_a, h_b, h_h]
  -- The proof steps will be skipped using 'sorry'
  sorry

end trapezium_area_l87_87693


namespace negation_statement_l87_87245

variable (x y : ℝ)

theorem negation_statement :
  ¬ (x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) :=
by
  sorry

end negation_statement_l87_87245


namespace gcd_1729_1314_l87_87634

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 :=
by
  sorry

end gcd_1729_1314_l87_87634


namespace abs_eq_inequality_l87_87723

theorem abs_eq_inequality (m : ℝ) (h : |m - 9| = 9 - m) : m ≤ 9 :=
sorry

end abs_eq_inequality_l87_87723


namespace sum_of_numbers_l87_87203

theorem sum_of_numbers (avg : ℝ) (num : ℕ) (h1 : avg = 5.2) (h2 : num = 8) : 
  (avg * num = 41.6) :=
by
  sorry

end sum_of_numbers_l87_87203


namespace calculate_price_l87_87223

-- Define variables for prices
def sugar_price_in_terms_of_salt (T : ℝ) : ℝ := 2 * T
def rice_price_in_terms_of_salt (T : ℝ) : ℝ := 3 * T
def apple_price : ℝ := 1.50
def pepper_price : ℝ := 1.25

-- Define pricing conditions
def condition_1 (T : ℝ) : Prop :=
  5 * (sugar_price_in_terms_of_salt T) + 3 * T + 2 * (rice_price_in_terms_of_salt T) + 3 * apple_price + 4 * pepper_price = 35

def condition_2 (T : ℝ) : Prop :=
  4 * (sugar_price_in_terms_of_salt T) + 2 * T + 1 * (rice_price_in_terms_of_salt T) + 2 * apple_price + 3 * pepper_price = 24

-- Define final price calculation with discounts
def total_price (T : ℝ) : ℝ :=
  8 * (sugar_price_in_terms_of_salt T) * 0.9 +
  5 * T +
  (rice_price_in_terms_of_salt T + 3 * (rice_price_in_terms_of_salt T - 0.5)) +
  -- adding two free apples to the count
  5 * apple_price +
  6 * pepper_price

-- Main theorem to prove
theorem calculate_price (T : ℝ) (h1 : condition_1 T) (h2 : condition_2 T) :
  total_price T = 55.64 :=
sorry -- proof omitted

end calculate_price_l87_87223


namespace relationship_ab_c_l87_87185
open Real

noncomputable def a : ℝ := (1 / 3) ^ (log 3 / log (1 / 3))
noncomputable def b : ℝ := (1 / 3) ^ (log 4 / log (1 / 3))
noncomputable def c : ℝ := 3 ^ log 3

theorem relationship_ab_c : c > b ∧ b > a := by
  sorry

end relationship_ab_c_l87_87185


namespace factory_produces_correct_number_of_candies_l87_87120

-- Definitions of the given conditions
def candies_per_hour : ℕ := 50
def hours_per_day : ℕ := 10
def days_to_complete_order : ℕ := 8

-- The theorem we want to prove
theorem factory_produces_correct_number_of_candies :
  days_to_complete_order * hours_per_day * candies_per_hour = 4000 :=
by 
  sorry

end factory_produces_correct_number_of_candies_l87_87120


namespace eight_bees_have_48_legs_l87_87152

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end eight_bees_have_48_legs_l87_87152


namespace min_washes_l87_87641

theorem min_washes (x : ℕ) :
  (1 / 4)^x ≤ 1 / 100 → x ≥ 4 :=
by sorry

end min_washes_l87_87641


namespace Yoque_borrowed_150_l87_87470

noncomputable def Yoque_borrowed_amount (X : ℝ) : Prop :=
  1.10 * X = 11 * 15

theorem Yoque_borrowed_150 (X : ℝ) : Yoque_borrowed_amount X → X = 150 :=
by
  -- proof will be filled in
  sorry

end Yoque_borrowed_150_l87_87470


namespace ray_climbs_l87_87601

theorem ray_climbs (n : ℕ) (h1 : n % 3 = 1) (h2 : n % 5 = 3) (h3 : n % 7 = 1) (h4 : n > 15) : n = 73 :=
sorry

end ray_climbs_l87_87601


namespace equation_of_circle_l87_87812

theorem equation_of_circle :
  ∃ (a : ℝ), a < 0 ∧ (∀ (x y : ℝ), (x + 2 * y = 0) → (x + 5)^2 + y^2 = 5) :=
by
  sorry

end equation_of_circle_l87_87812


namespace crayon_division_l87_87854

theorem crayon_division (total_crayons : ℕ) (crayons_each : ℕ) (Fred Benny Jason : ℕ) 
  (h_total : total_crayons = 24) (h_each : crayons_each = 8) 
  (h_division : Fred = crayons_each ∧ Benny = crayons_each ∧ Jason = crayons_each) : 
  Fred + Benny + Jason = total_crayons :=
by
  sorry

end crayon_division_l87_87854


namespace lcm_12_18_l87_87534

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l87_87534


namespace sum_of_products_nonpos_l87_87900

theorem sum_of_products_nonpos (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 :=
sorry

end sum_of_products_nonpos_l87_87900


namespace total_games_won_l87_87507

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end total_games_won_l87_87507


namespace john_sales_percentage_l87_87222

noncomputable def percentage_buyers (houses_visited_per_day : ℕ) (work_days_per_week : ℕ) (weekly_sales : ℝ) (low_price : ℝ) (high_price : ℝ) : ℝ :=
  let total_houses_per_week := houses_visited_per_day * work_days_per_week
  let average_sale_per_customer := (low_price + high_price) / 2
  let total_customers := weekly_sales / average_sale_per_customer
  (total_customers / total_houses_per_week) * 100

theorem john_sales_percentage :
  percentage_buyers 50 5 5000 50 150 = 20 := 
by 
  sorry

end john_sales_percentage_l87_87222


namespace derivative_of_y_is_correct_l87_87432

noncomputable def y (x : ℝ) := x^2 * Real.sin x

theorem derivative_of_y_is_correct : (deriv y x = 2 * x * Real.sin x + x^2 * Real.cos x) :=
by
  sorry

end derivative_of_y_is_correct_l87_87432


namespace greatest_product_sum_2000_l87_87273

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87273


namespace sum_a_b_is_nine_l87_87031

theorem sum_a_b_is_nine (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (b + 2 - a)^2 + (a - b)^2 + (b + 2 + a)^2 + (a + b)^2 = 324) 
    (h4 : ∃ a' b', a' = a ∧ b' = b ∧ (b + 2 - a) * 1 = -(b + 2 - a)) : 
  a + b = 9 :=
sorry

end sum_a_b_is_nine_l87_87031


namespace fuel_consumption_rate_l87_87820

theorem fuel_consumption_rate (fuel_left time_left r: ℝ) 
    (h_fuel: fuel_left = 6.3333) 
    (h_time: time_left = 0.6667) 
    (h_rate: r = fuel_left / time_left) : r = 9.5 := 
by
    sorry

end fuel_consumption_rate_l87_87820


namespace expression_for_3_diamond_2_l87_87068

variable {a b : ℝ}

def diamond (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem expression_for_3_diamond_2 (a : ℝ) :
  3 * diamond a 2 = 12 * a - 18 :=
by
  sorry

end expression_for_3_diamond_2_l87_87068


namespace rafael_earnings_l87_87235

theorem rafael_earnings 
  (hours_monday : ℕ) 
  (hours_tuesday : ℕ) 
  (hours_left : ℕ) 
  (rate_per_hour : ℕ) 
  (h_monday : hours_monday = 10) 
  (h_tuesday : hours_tuesday = 8) 
  (h_left : hours_left = 20) 
  (h_rate : rate_per_hour = 20) : 
  (hours_monday + hours_tuesday + hours_left) * rate_per_hour = 760 := 
by
  sorry

end rafael_earnings_l87_87235


namespace no_rational_positive_and_negative_l87_87082

-- Definitions of conditions
def is_positive (a : ℚ) : Prop := a > 0
def is_negative (a : ℚ) : Prop := a < 0

-- The mathematically equivalent proof problem in Lean 4 statement
theorem no_rational_positive_and_negative :
  ¬ ∃ a : ℚ, is_positive a ∧ is_negative a := by
  sorry

end no_rational_positive_and_negative_l87_87082


namespace sufficient_but_not_necessary_condition_l87_87444

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 / 8) → (∀ x : ℝ, x > 0 → 2 * x + a / x ≥ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l87_87444


namespace domain_of_f_log2x_is_0_4_l87_87555

def f : ℝ → ℝ := sorry

-- Given condition: domain of y = f(2x) is (-1, 1)
def dom_f_2x (x : ℝ) : Prop := -1 < 2 * x ∧ 2 * x < 1

-- Conclusion: domain of y = f(log_2 x) is (0, 4)
def dom_f_log2x (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem domain_of_f_log2x_is_0_4 (x : ℝ) :
  (dom_f_2x x) → (dom_f_log2x x) :=
by
  sorry

end domain_of_f_log2x_is_0_4_l87_87555


namespace lcm_12_18_is_36_l87_87529

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l87_87529


namespace probability_of_three_given_sum_seven_l87_87793

theorem probability_of_three_given_sum_seven : 
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7) 
    ∧ (dice1 = 3 ∨ dice2 = 3)) →
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7)) →
  ∃ (p : ℚ), p = 1/3 :=
by 
  sorry

end probability_of_three_given_sum_seven_l87_87793


namespace min_value_of_m_cauchy_schwarz_inequality_l87_87706

theorem min_value_of_m (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m = a + 1 / ((a - b) * b)) : 
  ∃ t, t = 3 ∧ ∀ a b : ℝ, a > b → b > 0 → m = a + 1 / ((a - b) * b) → m ≥ t :=
sorry

theorem cauchy_schwarz_inequality (x y z : ℝ) :
  (x^2 + 4 * y^2 + z^2 = 3) → |x + 2 * y + z| ≤ 3 :=
sorry

end min_value_of_m_cauchy_schwarz_inequality_l87_87706


namespace balance_scale_cereal_l87_87629

def scales_are_balanced (left_pan : ℕ) (right_pan : ℕ) : Prop :=
  left_pan = right_pan

theorem balance_scale_cereal (inaccurate_scales : ℕ → ℕ → Prop)
  (cereal : ℕ)
  (correct_weight : ℕ) :
  (∀ left_pan right_pan, inaccurate_scales left_pan right_pan → left_pan = right_pan) →
  (cereal / 2 = 1) →
  true :=
  sorry

end balance_scale_cereal_l87_87629


namespace mineral_sample_ages_l87_87029

/--
We have a mineral sample with digits {2, 2, 3, 3, 5, 9}.
Given the condition that the age must start with an odd number,
we need to prove that the total number of possible ages is 120.
-/
theorem mineral_sample_ages : 
  ∀ (l : List ℕ), l = [2, 2, 3, 3, 5, 9] → 
  (l.filter odd).length > 0 →
  ∃ n : ℕ, n = 120 :=
by
  intros l h_digits h_odd
  sorry

end mineral_sample_ages_l87_87029


namespace anayet_speed_is_61_l87_87496

-- Define the problem conditions
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_time : ℝ := 2
def total_distance : ℝ := 369
def remaining_distance : ℝ := 121

-- Calculate derived values
def amoli_distance : ℝ := amoli_speed * amoli_time
def covered_distance : ℝ := total_distance - remaining_distance
def anayet_distance : ℝ := covered_distance - amoli_distance

-- Define the theorem to prove Anayet's speed
theorem anayet_speed_is_61 : anayet_distance / anayet_time = 61 :=
by
  -- sorry is a placeholder for the proof
  sorry

end anayet_speed_is_61_l87_87496


namespace probability_S6_between_2_and_4_l87_87147

noncomputable def a_n (n : ℕ) (roll : ℕ) : ℤ :=
  if roll % 2 = 1 then 1 else -1

noncomputable def S_n (n : ℕ) (rolls : ℕ → ℕ) : ℤ :=
  (Finset.range n).sum (λ i, a_n i rolls i)

theorem probability_S6_between_2_and_4 :
  let p := PMF.ofMultiset ![1/6, 1/6, 1/6, 1/6, 1/6, 1/6] in
  P ((λ rolls : ℕ → ℕ, 2 ≤ S_n 6 rolls ∧ S_n 6 rolls ≤ 4)) = 21 / 64 :=
sorry

end probability_S6_between_2_and_4_l87_87147


namespace cos_C_in_triangle_l87_87396

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end cos_C_in_triangle_l87_87396


namespace find_original_wage_l87_87150

-- Defining the conditions
variables (W : ℝ) (W_new : ℝ) (h : W_new = 35) (h2 : W_new = 1.40 * W)

-- Statement that needs to be proved
theorem find_original_wage : W = 25 :=
by
  -- proof omitted
  sorry

end find_original_wage_l87_87150


namespace lcm_12_18_is_36_l87_87526

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l87_87526


namespace solve_for_x_l87_87613

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l87_87613


namespace shaded_l_shaped_area_l87_87842

def square (side : ℕ) : ℕ := side * side
def rectangle (length width : ℕ) : ℕ := length * width

theorem shaded_l_shaped_area :
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  areaABCD - total_area_small_shapes = 20 :=
by
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  have h : areaABCD - total_area_small_shapes = 20 := sorry
  exact h

end shaded_l_shaped_area_l87_87842


namespace number_of_triangles_l87_87980

theorem number_of_triangles (x : ℕ) (h₁ : 2 + x > 6) (h₂ : 8 > x) : ∃! t, t = 3 :=
by {
  sorry
}

end number_of_triangles_l87_87980


namespace find_triplet_l87_87525

def ordered_triplet : Prop :=
  ∃ (x y z : ℚ), 
  7 * x + 3 * y = z - 10 ∧ 
  2 * x - 4 * y = 3 * z + 20 ∧ 
  x = 0 ∧ 
  y = -50 / 13 ∧ 
  z = -20 / 13

theorem find_triplet : ordered_triplet := 
  sorry

end find_triplet_l87_87525


namespace choose_officers_from_six_l87_87883

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l87_87883


namespace sample_size_drawn_l87_87183

theorem sample_size_drawn (sample_size : ℕ) (probability : ℚ) (N : ℚ) 
  (h1 : sample_size = 30) 
  (h2 : probability = 0.25) 
  (h3 : probability = sample_size / N) : 
  N = 120 := by
  sorry

end sample_size_drawn_l87_87183


namespace length_DE_l87_87431

-- Definitions for conditions in the problem
variables (AB : ℝ) (DE : ℝ) (areaABC : ℝ)

-- Condition: AB is 15 cm
axiom length_AB : AB = 15

-- Condition: The area of triangle projected below the base is 25% of the area of triangle ABC
axiom area_ratio_condition : (1 / 4) * areaABC = (1 / 2)^2 * areaABC

-- The problem statement translated to Lean proof
theorem length_DE : DE = 7.5 :=
by
  -- Definitions and conditions
  have h1 : AB = 15 := length_AB
  have h2 : (1 / 2)^2 = 1 / 4 := by ring
  calc
    DE = (0.5) * AB :  sorry  -- proportional relationship since triangles are similar
    ... = 0.5 * 15   :  by rw [h1]
    ... = 7.5       :  by norm_num

end length_DE_l87_87431


namespace max_product_of_two_integers_sum_2000_l87_87310

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87310


namespace calculate_weight_5_moles_Al2O3_l87_87015

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def molecular_weight_Al2O3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_O)
def moles_Al2O3 : ℝ := 5
def weight_5_moles_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem calculate_weight_5_moles_Al2O3 :
  weight_5_moles_Al2O3 = 509.8 :=
by sorry

end calculate_weight_5_moles_Al2O3_l87_87015


namespace Miss_Stevie_payment_l87_87578

theorem Miss_Stevie_payment:
  let painting_hours := 8
  let painting_rate := 15
  let painting_earnings := painting_hours * painting_rate
  let mowing_hours := 6
  let mowing_rate := 10
  let mowing_earnings := mowing_hours * mowing_rate
  let plumbing_hours := 4
  let plumbing_rate := 18
  let plumbing_earnings := plumbing_hours * plumbing_rate
  let total_earnings := painting_earnings + mowing_earnings + plumbing_earnings
  let discount := 0.10 * total_earnings
  let amount_paid := total_earnings - discount
  amount_paid = 226.80 :=
by
  sorry

end Miss_Stevie_payment_l87_87578


namespace determine_f_value_l87_87553

noncomputable def f (t : ℝ) : ℝ := t^2 + 2

theorem determine_f_value : f 3 = 11 := by
  sorry

end determine_f_value_l87_87553


namespace linear_valid_arrangements_circular_valid_arrangements_l87_87217

def word := "EFFERVESCES"
def multiplicities := [("E", 4), ("F", 2), ("S", 2), ("R", 1), ("V", 1), ("C", 1)]

-- Number of valid linear arrangements
def linear_arrangements_no_adj_e : ℕ := 88200

-- Number of valid circular arrangements
def circular_arrangements_no_adj_e : ℕ := 6300

theorem linear_valid_arrangements : 
  ∃ n, n = linear_arrangements_no_adj_e := 
  by
    sorry 

theorem circular_valid_arrangements :
  ∃ n, n = circular_arrangements_no_adj_e :=
  by
    sorry

end linear_valid_arrangements_circular_valid_arrangements_l87_87217


namespace greatest_product_sum_2000_eq_1000000_l87_87286

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87286


namespace david_still_has_less_than_750_l87_87954

theorem david_still_has_less_than_750 (S R : ℝ) 
  (h1 : S + R = 1500)
  (h2 : R < S) : 
  R < 750 :=
by 
  sorry

end david_still_has_less_than_750_l87_87954


namespace sum_of_ages_l87_87400

variables (K T1 T2 : ℕ)

theorem sum_of_ages (h1 : K * T1 * T2 = 72) (h2 : T1 = T2) (h3 : T1 < K) : K + T1 + T2 = 14 :=
sorry

end sum_of_ages_l87_87400


namespace pies_sold_by_mcgee_l87_87603

/--
If Smith's Bakery sold 70 pies, and they sold 6 more than four times the number of pies that Mcgee's Bakery sold,
prove that Mcgee's Bakery sold 16 pies.
-/
theorem pies_sold_by_mcgee (x : ℕ) (h1 : 4 * x + 6 = 70) : x = 16 :=
by
  sorry

end pies_sold_by_mcgee_l87_87603


namespace triangle_incircle_ratio_l87_87493

theorem triangle_incircle_ratio (r p k : ℝ) (h1 : k = r * (p / 2)) : 
  p / k = 2 / r :=
by
  sorry

end triangle_incircle_ratio_l87_87493


namespace cube_and_difference_of_squares_l87_87976

theorem cube_and_difference_of_squares (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 :=
by {
  sorry
}

end cube_and_difference_of_squares_l87_87976


namespace exists_integer_solution_l87_87797

theorem exists_integer_solution (x : ℤ) (h : x - 1 < 0) : ∃ y : ℤ, y < 1 :=
by
  sorry

end exists_integer_solution_l87_87797


namespace nickel_ate_4_chocolates_l87_87754

theorem nickel_ate_4_chocolates (R N : ℕ) (h1 : R = 13) (h2 : R = N + 9) : N = 4 :=
by
  sorry

end nickel_ate_4_chocolates_l87_87754


namespace max_a_l87_87370

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem max_a (a : ℝ) :
  (∀ x ∈ set.Icc (1 / 2 : ℝ) (2 : ℝ), f x ≥ a) ↔ a ≤ 0 :=
by
  sorry

end max_a_l87_87370


namespace sharpening_cost_l87_87968

theorem sharpening_cost
  (trees_chopped : ℕ)
  (trees_per_sharpening : ℕ)
  (total_cost : ℕ)
  (min_trees_chopped : trees_chopped ≥ 91)
  (trees_per_sharpening_eq : trees_per_sharpening = 13)
  (total_cost_eq : total_cost = 35) :
  total_cost / (trees_chopped / trees_per_sharpening) = 5 := by
  sorry

end sharpening_cost_l87_87968


namespace both_firms_participate_social_optimality_l87_87215

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end both_firms_participate_social_optimality_l87_87215


namespace diagonal_in_parallelogram_l87_87127

-- Define the conditions of the problem
variable (A B C D M : Point)
variable (parallelogram : Parallelogram A B C D)
variable (height_bisects_side : Midpoint M A D)
variable (height_length : Distance B M = 2)
variable (acute_angle_30 : Angle A B D = 30)

-- Define the theorem based on the conditions
theorem diagonal_in_parallelogram (h1 : parallelogram) (h2 : height_bisects_side)
  (h3 : height_length) (h4 : acute_angle_30) : 
  ∃ (BD_length : ℝ) (angle1 angle2 : ℝ), BD_length = 4 ∧ angle1 = 30 ∧ angle2 = 120 := 
sorry

end diagonal_in_parallelogram_l87_87127


namespace student_score_l87_87490

theorem student_score
    (total_questions : ℕ)
    (correct_responses : ℕ)
    (grading_method : ℕ → ℕ → ℕ)
    (h1 : total_questions = 100)
    (h2 : correct_responses = 92)
    (h3 : grading_method = λ correct incorrect => correct - 2 * incorrect) :
  grading_method correct_responses (total_questions - correct_responses) = 76 :=
by
  -- proof would be here, but is skipped
  sorry

end student_score_l87_87490


namespace number_that_multiplies_b_l87_87724

theorem number_that_multiplies_b (a b x : ℝ) (h0 : 4 * a = x * b) (h1 : a * b ≠ 0) (h2 : (a / 5) / (b / 4) = 1) : x = 5 :=
by
  sorry

end number_that_multiplies_b_l87_87724


namespace abs_neg_2023_l87_87107

-- Define the absolute value function following the provided condition
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- State the theorem we need to prove
theorem abs_neg_2023 : abs (-2023) = 2023 :=
by
  -- Placeholder for proof
  sorry

end abs_neg_2023_l87_87107


namespace extremum_value_l87_87552

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem extremum_value (a b : ℝ) (h1 : (3 - 6 * a + b = 0)) (h2 : (-1 + 3 * a - b + a^2 = 0)) :
  a - b = -7 :=
by
  sorry

end extremum_value_l87_87552


namespace problem_1_problem_2_l87_87227

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- 1. Prove that A ∩ B = {x | -2 < x ≤ 2}
theorem problem_1 : A ∩ B = {x | -2 < x ∧ x ≤ 2} :=
by
  sorry

-- 2. Prove that (complement U A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}
theorem problem_2 : (U \ A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3} :=
by
  sorry

end problem_1_problem_2_l87_87227


namespace intersection_is_A_l87_87193

-- Define the set M based on the given condition
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define the set N based on the given condition
def N : Set ℝ := {x | ∃ y, y = 3 * x^2 + 1}

-- Define the set A as the intersection of M and N
def A : Set ℝ := {x | x > 1}

-- Prove that the intersection of M and N is equal to the set A
theorem intersection_is_A : (M ∩ N = A) :=
by {
  sorry
}

end intersection_is_A_l87_87193


namespace remaining_tabs_after_closures_l87_87891

theorem remaining_tabs_after_closures (initial_tabs : ℕ) (first_fraction : ℚ) (second_fraction : ℚ) (third_fraction : ℚ) 
  (initial_eq : initial_tabs = 400) :
  (initial_tabs - initial_tabs * first_fraction - (initial_tabs - initial_tabs * first_fraction) * second_fraction - 
      ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction) = 90 :=
by
  have h1 : initial_tabs * first_fraction = 100 := by rw [initial_eq]; norm_num
  have h2 : initial_tabs - initial_tabs * first_fraction = 300 := by rw [initial_eq, h1]; norm_num
  have h3 : (initial_tabs - initial_tabs * first_fraction) * second_fraction = 120 := by rw [h2]; norm_num
  have h4 : (initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction = 180 := by { rw [h2, h3]; norm_num }
  have h5 : ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction = 90 := by rw [h4]; norm_num
  have h6 : ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction - ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction) = 90 := by rw [h4, h5]; norm_num
  exact h6


end remaining_tabs_after_closures_l87_87891


namespace find_value_of_y_l87_87789

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end find_value_of_y_l87_87789


namespace total_cards_1750_l87_87649

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end total_cards_1750_l87_87649


namespace find_common_ratio_l87_87970

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end find_common_ratio_l87_87970


namespace find_solution_l87_87363

def satisfies_conditions (x y z : ℝ) :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solution (x y z : ℝ) :
  satisfies_conditions x y z →
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
by
  sorry

end find_solution_l87_87363


namespace right_triangle_30_60_90_l87_87076

theorem right_triangle_30_60_90 (a b : ℝ) (h : a = 15) :
  (b = 30) ∧ (b = 15 * Real.sqrt 3) :=
by
  sorry

end right_triangle_30_60_90_l87_87076


namespace two_packs_remainder_l87_87965

theorem two_packs_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 :=
by {
  sorry
}

end two_packs_remainder_l87_87965


namespace expression_zero_iff_x_eq_three_l87_87359

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) → ((x^2 - 6 * x + 9 = 0) ↔ (x = 3)) :=
by
  sorry

end expression_zero_iff_x_eq_three_l87_87359


namespace triangle_inequality_of_three_l87_87100

theorem triangle_inequality_of_three (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := 
sorry

end triangle_inequality_of_three_l87_87100


namespace rebecca_hours_worked_l87_87140

theorem rebecca_hours_worked (x : ℕ)
  (total_hours : Thomas Toby Rebecca : ℕ)
  (thomas_worked : Thomas = x)
  (toby_worked : Toby = 2 * x - 10)
  (rebecca_worked : Rebecca = 2 * x - 18)
  (total_hours_worked : Thomas + Toby + Rebecca = 157) :
  Rebecca = 56 := 
sorry

end rebecca_hours_worked_l87_87140


namespace total_bouncy_balls_l87_87749

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def balls_per_pack := 10

theorem total_bouncy_balls:
  (red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack) = 160 :=
by 
  sorry

end total_bouncy_balls_l87_87749


namespace ratio_of_areas_l87_87261

-- Definitions based on the conditions given
variables (A B M N P Q O : Type) 
variables (AB BM BP : ℝ)

-- Assumptions
axiom hAB : AB = 6
axiom hBM : BM = 9
axiom hBP : BP = 5

-- Theorem statement
theorem ratio_of_areas (hMN : M ≠ N) (hPQ : P ≠ Q) :
  (1 / 121 : ℝ) = sorry :=
by sorry

end ratio_of_areas_l87_87261


namespace solution_concentration_l87_87023

theorem solution_concentration (C : ℝ) :
  (0.16 + 0.01 * C * 2 = 0.36) ↔ (C = 10) :=
by
  sorry

end solution_concentration_l87_87023


namespace total_cost_is_80_l87_87482

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end total_cost_is_80_l87_87482


namespace largest_divisor_three_consecutive_l87_87740

theorem largest_divisor_three_consecutive (u v w : ℤ) (h1 : u + 1 = v) (h2 : v + 1 = w) (h3 : ∃ n : ℤ, (u = 5 * n) ∨ (v = 5 * n) ∨ (w = 5 * n)) : 
  ∀ d ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c}, 
  15 ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c} :=
sorry

end largest_divisor_three_consecutive_l87_87740


namespace max_product_two_integers_sum_2000_l87_87284

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87284


namespace sarahs_score_l87_87419

theorem sarahs_score (g s : ℕ) (h1 : s = g + 60) (h2 : s + g = 260) : s = 160 :=
sorry

end sarahs_score_l87_87419


namespace reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l87_87932

-- Definition: reversing a deck of n cards in k operations
def can_reverse_deck (n k : ℕ) : Prop := sorry -- Placeholder definition

-- Proof Part (a)
theorem reverse_9_in_5_operations :
  can_reverse_deck 9 5 :=
sorry

-- Proof Part (b)
theorem reverse_52_in_27_operations :
  can_reverse_deck 52 27 :=
sorry

-- Proof Part (c)
theorem not_reverse_52_in_17_operations :
  ¬can_reverse_deck 52 17 :=
sorry

-- Proof Part (d)
theorem not_reverse_52_in_26_operations :
  ¬can_reverse_deck 52 26 :=
sorry

end reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l87_87932


namespace max_product_of_sum_2000_l87_87306

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87306


namespace junior_score_is_90_l87_87212

theorem junior_score_is_90 {n : ℕ} (hn : n > 0)
    (j : ℕ := n / 5) (s : ℕ := 4 * n / 5)
    (overall_avg : ℝ := 86)
    (senior_avg : ℝ := 85)
    (junior_score : ℝ)
    (h1 : 20 * j = n)
    (h2 : 80 * s = n * 4)
    (h3 : overall_avg * n = 86 * n)
    (h4 : senior_avg * s = 85 * s)
    (h5 : j * junior_score = overall_avg * n - senior_avg * s) :
    junior_score = 90 :=
by
  sorry

end junior_score_is_90_l87_87212


namespace total_spokes_in_garage_l87_87833

def bicycle1_front_spokes : ℕ := 16
def bicycle1_back_spokes : ℕ := 18
def bicycle2_front_spokes : ℕ := 20
def bicycle2_back_spokes : ℕ := 22
def bicycle3_front_spokes : ℕ := 24
def bicycle3_back_spokes : ℕ := 26
def bicycle4_front_spokes : ℕ := 28
def bicycle4_back_spokes : ℕ := 30
def tricycle_front_spokes : ℕ := 32
def tricycle_middle_spokes : ℕ := 34
def tricycle_back_spokes : ℕ := 36

theorem total_spokes_in_garage :
  bicycle1_front_spokes + bicycle1_back_spokes +
  bicycle2_front_spokes + bicycle2_back_spokes +
  bicycle3_front_spokes + bicycle3_back_spokes +
  bicycle4_front_spokes + bicycle4_back_spokes +
  tricycle_front_spokes + tricycle_middle_spokes + tricycle_back_spokes = 286 :=
by
  sorry

end total_spokes_in_garage_l87_87833


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l87_87416

section ProbabilityQuiz

variable (total_questions : ℕ) (mc_questions : ℕ) (tf_questions : ℕ)

def prob_A_mc_and_B_tf (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  (mc_questions * tf_questions : ℚ) / (total_questions * (total_questions - 1))

def prob_at_least_one_mc (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  1 - ((tf_questions * (tf_questions - 1) : ℚ) / (total_questions * (total_questions - 1)))

theorem probability_A_mc_and_B_tf :
  prob_A_mc_and_B_tf 10 6 4 = 4 / 15 := by
  sorry

theorem probability_at_least_one_mc :
  prob_at_least_one_mc 10 6 4 = 13 / 15 := by
  sorry

end ProbabilityQuiz

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l87_87416


namespace smallest_prime_factor_of_1917_l87_87144

theorem smallest_prime_factor_of_1917 : ∃ p : ℕ, Prime p ∧ (p ∣ 1917) ∧ (∀ q : ℕ, Prime q ∧ (q ∣ 1917) → q ≥ p) :=
by
  sorry

end smallest_prime_factor_of_1917_l87_87144


namespace hostel_cost_l87_87802

def first_week_rate : ℝ := 18
def additional_week_rate : ℝ := 12
def first_week_days : ℕ := 7
def total_days : ℕ := 23

theorem hostel_cost :
  (first_week_days * first_week_rate + 
  (total_days - first_week_days) / first_week_days * first_week_days * additional_week_rate + 
  (total_days - first_week_days) % first_week_days * additional_week_rate) = 318 := 
by
  sorry

end hostel_cost_l87_87802


namespace fabric_sales_fraction_l87_87813

def total_sales := 36
def stationery_sales := 15
def jewelry_sales := total_sales / 4
def fabric_sales := total_sales - jewelry_sales - stationery_sales

theorem fabric_sales_fraction:
  (fabric_sales : ℝ) / total_sales = 1 / 3 :=
by
  sorry

end fabric_sales_fraction_l87_87813


namespace base_six_four_digit_odd_final_l87_87545

theorem base_six_four_digit_odd_final :
  ∃ b : ℕ, (b^4 > 285 ∧ 285 ≥ b^3 ∧ (285 % b) % 2 = 1) :=
by 
  use 6
  sorry

end base_six_four_digit_odd_final_l87_87545


namespace length_of_PW_l87_87080

-- Given variables
variables (CD WX DP PX : ℝ) (CW : ℝ)

-- Condition 1: CD is parallel to WX
axiom h1 : true -- Parallelism is given as part of the problem

-- Condition 2: CW = 60 units
axiom h2 : CW = 60

-- Condition 3: DP = 18 units
axiom h3 : DP = 18

-- Condition 4: PX = 36 units
axiom h4 : PX = 36

-- Question/Answer: Prove that the length of PW = 40 units
theorem length_of_PW (PW CP : ℝ) (h5 : CP = PW / 2) (h6 : CW = CP + PW) : PW = 40 :=
by sorry

end length_of_PW_l87_87080


namespace rate_of_dividend_is_12_l87_87661

-- Defining the conditions
def total_investment : ℝ := 4455
def price_per_share : ℝ := 8.25
def annual_income : ℝ := 648
def face_value_per_share : ℝ := 10

-- Expected rate of dividend
def expected_rate_of_dividend : ℝ := 12

-- The proof problem statement: Prove that the rate of dividend is 12% given the conditions.
theorem rate_of_dividend_is_12 :
  ∃ (r : ℝ), r = 12 ∧ annual_income = 
    (total_investment / price_per_share) * (r / 100) * face_value_per_share :=
by 
  use 12
  sorry

end rate_of_dividend_is_12_l87_87661


namespace reciprocal_of_neg_2023_l87_87002

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87002


namespace evaluate_expression_l87_87689

theorem evaluate_expression :
  8^(-1/3 : ℝ) + (49^(-1/2 : ℝ))^(1/2 : ℝ) = (Real.sqrt 7 + 2) / (2 * Real.sqrt 7) := by
  sorry

end evaluate_expression_l87_87689


namespace units_digit_47_pow_47_l87_87455

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l87_87455


namespace avg_age_family_now_l87_87324

namespace average_age_family

-- Define initial conditions
def avg_age_husband_wife_marriage := 23
def years_since_marriage := 5
def age_child := 1
def number_of_family_members := 3

-- Prove that the average age of the family now is 19 years
theorem avg_age_family_now :
  (2 * avg_age_husband_wife_marriage + 2 * years_since_marriage + age_child) / number_of_family_members = 19 := by
  sorry

end average_age_family

end avg_age_family_now_l87_87324


namespace find_amplitude_l87_87505

noncomputable def amplitude_of_cosine (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  a

theorem find_amplitude (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_max : amplitude_of_cosine a b h_a h_b = 3) :
  a = 3 :=
sorry

end find_amplitude_l87_87505


namespace Morse_code_sequences_l87_87209

theorem Morse_code_sequences : 
  let symbols (n : ℕ) := 2^n in
  symbols 1 + symbols 2 + symbols 3 + symbols 4 + symbols 5 = 62 :=
by
  sorry

end Morse_code_sequences_l87_87209


namespace negation_of_P_is_non_P_l87_87550

open Real

/-- Proposition P: For any x in the real numbers, sin(x) <= 1 -/
def P : Prop := ∀ x : ℝ, sin x ≤ 1

/-- Negation of P: There exists x in the real numbers such that sin(x) >= 1 -/
def non_P : Prop := ∃ x : ℝ, sin x ≥ 1

theorem negation_of_P_is_non_P : ¬P ↔ non_P :=
by 
  sorry

end negation_of_P_is_non_P_l87_87550


namespace probability_of_sunglasses_given_caps_l87_87098

theorem probability_of_sunglasses_given_caps
  (s c sc : ℕ) 
  (h₀ : s = 60) 
  (h₁ : c = 40)
  (h₂ : sc = 20)
  (h₃ : sc = 1 / 3 * s) : 
  (sc / c) = 1 / 2 :=
by
  sorry

end probability_of_sunglasses_given_caps_l87_87098


namespace ratio_geometric_sequence_of_arithmetic_l87_87860

variable {d : ℤ}
variable {a : ℕ → ℤ}

-- definition of an arithmetic sequence with common difference d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- definition of a geometric sequence for a_5, a_9, a_{15}
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 9 * a 9 = a 5 * a 15

theorem ratio_geometric_sequence_of_arithmetic
  (h_arith : arithmetic_sequence a d) (h_nonzero : d ≠ 0) (h_geom : geometric_sequence a) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end ratio_geometric_sequence_of_arithmetic_l87_87860


namespace cell_survival_after_6_hours_l87_87654

def cell_sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 2) ∧ (∀ n, a (n + 1) = 2 * a n - 1)

theorem cell_survival_after_6_hours :
  ∃ a : ℕ → ℕ, cell_sequence a ∧ a 6 = 65 :=
by
  sorry

end cell_survival_after_6_hours_l87_87654


namespace problem_statement_l87_87598

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end problem_statement_l87_87598


namespace train_departure_time_l87_87492

theorem train_departure_time 
(distance speed : ℕ) (arrival_time_chicago difference : ℕ) (arrival_time_new_york departure_time : ℕ) 
(h_dist : distance = 480) 
(h_speed : speed = 60)
(h_arrival_chicago : arrival_time_chicago = 17) 
(h_difference : difference = 1)
(h_arrival_new_york : arrival_time_new_york = arrival_time_chicago + difference) : 
  departure_time = arrival_time_new_york - distance / speed :=
by
  sorry

end train_departure_time_l87_87492


namespace age_proof_l87_87253

noncomputable def father_age_current := 33
noncomputable def xiaolin_age_current := 3

def father_age (X : ℕ) := 11 * X
def future_father_age (F : ℕ) := F + 7
def future_xiaolin_age (X : ℕ) := X + 7

theorem age_proof (F X : ℕ) (h1 : F = father_age X) 
  (h2 : future_father_age F = 4 * future_xiaolin_age X) : 
  F = father_age_current ∧ X = xiaolin_age_current :=
by 
  sorry

end age_proof_l87_87253


namespace solve_for_x_l87_87608

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l87_87608


namespace men_joined_l87_87433

-- Definitions for initial conditions
def initial_men : ℕ := 10
def initial_days : ℕ := 50
def extended_days : ℕ := 25

-- Theorem stating the number of men who joined the camp
theorem men_joined (x : ℕ) 
    (initial_food : initial_men * initial_days = (initial_men + x) * extended_days) : 
    x = 10 := 
sorry

end men_joined_l87_87433


namespace mixed_oil_rate_l87_87999

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end mixed_oil_rate_l87_87999


namespace width_of_field_l87_87645

noncomputable def field_width : ℝ := 60

theorem width_of_field (L W : ℝ) (hL : L = (7/5) * W) (hP : 288 = 2 * L + 2 * W) : W = field_width :=
by
  sorry

end width_of_field_l87_87645


namespace max_product_of_sum_2000_l87_87303

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87303


namespace pure_imaginary_value_l87_87554

theorem pure_imaginary_value (a : ℝ) 
  (h1 : (a^2 - 3 * a + 2) = 0) 
  (h2 : (a - 2) ≠ 0) : a = 1 := sorry

end pure_imaginary_value_l87_87554


namespace abs_neg_2023_eq_2023_l87_87114

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l87_87114


namespace simplify_expression_l87_87911

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := 
by 
sorry

end simplify_expression_l87_87911


namespace find_a_plus_b_l87_87700

variables {a b : ℝ}

theorem find_a_plus_b (h1 : a - b = -3) (h2 : a * b = 2) : a + b = Real.sqrt 17 ∨ a + b = -Real.sqrt 17 := by
  -- Proof can be filled in here
  sorry

end find_a_plus_b_l87_87700


namespace max_product_of_sum_2000_l87_87305

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87305


namespace probability_other_side_red_l87_87156

def card_black_black := 4
def card_black_red := 2
def card_red_red := 2

def total_cards := card_black_black + card_black_red + card_red_red

-- Calculate the total number of red faces
def total_red_faces := (card_red_red * 2) + card_black_red

-- Number of red faces that have the other side also red
def red_faces_with_other_red := card_red_red * 2

-- Target probability to prove
theorem probability_other_side_red (h : total_cards = 8) : 
  (red_faces_with_other_red / total_red_faces) = 2 / 3 := 
  sorry

end probability_other_side_red_l87_87156


namespace troy_needs_more_money_l87_87782

theorem troy_needs_more_money (initial_savings : ℕ) (sold_computer : ℕ) (new_computer_cost : ℕ) :
  initial_savings = 50 → sold_computer = 20 → new_computer_cost = 80 → 
  new_computer_cost - (initial_savings + sold_computer) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end troy_needs_more_money_l87_87782


namespace negation_of_proposition_l87_87435

theorem negation_of_proposition (x : ℝ) : 
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := 
sorry

end negation_of_proposition_l87_87435


namespace solve_angle_CBO_l87_87119

theorem solve_angle_CBO 
  (BAO CAO : ℝ) (CBO ABO : ℝ) (ACO BCO : ℝ) (AOC : ℝ) 
  (h1 : BAO = CAO) 
  (h2 : CBO = ABO) 
  (h3 : ACO = BCO) 
  (h4 : AOC = 110) 
  : CBO = 20 :=
by
  sorry

end solve_angle_CBO_l87_87119


namespace number_of_sandwiches_l87_87420

-- Define the constants and assumptions

def soda_cost : ℤ := 1
def number_of_sodas : ℤ := 3
def cost_of_sodas : ℤ := number_of_sodas * soda_cost

def number_of_soups : ℤ := 2
def soup_cost : ℤ := cost_of_sodas
def cost_of_soups : ℤ := number_of_soups * soup_cost

def sandwich_cost : ℤ := 3 * soup_cost
def total_cost : ℤ := 18

-- The mathematical statement we want to prove
theorem number_of_sandwiches :
  ∃ n : ℤ, (n * sandwich_cost + cost_of_sodas + cost_of_soups = total_cost) ∧ n = 1 :=
by
  sorry

end number_of_sandwiches_l87_87420


namespace correct_statements_about_C_l87_87103

-- Conditions: Curve C is defined by the equation x^4 + y^2 = 1
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Prove the properties of curve C
theorem correct_statements_about_C :
  (-- 1. Symmetric about the x-axis
    (∀ x y : ℝ, C x y → C x (-y)) ∧
    -- 2. Symmetric about the y-axis
    (∀ x y : ℝ, C x y → C (-x) y) ∧
    -- 3. Symmetric about the origin
    (∀ x y : ℝ, C x y → C (-x) (-y)) ∧
    -- 6. A closed figure with an area greater than π
    (∃ (area : ℝ), area > π)) := sorry

end correct_statements_about_C_l87_87103


namespace least_common_multiple_increments_l87_87451

theorem least_common_multiple_increments :
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  Nat.lcm (Nat.lcm (Nat.lcm a' b') c') d' = 8645 :=
by
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  sorry

end least_common_multiple_increments_l87_87451


namespace max_value_expression_l87_87747

theorem max_value_expression (x1 x2 x3 : ℝ) (h1 : x1 + x2 + x3 = 1) (h2 : 0 < x1) (h3 : 0 < x2) (h4 : 0 < x3) :
    x1 * x2^2 * x3 + x1 * x2 * x3^2 ≤ 27 / 1024 :=
sorry

end max_value_expression_l87_87747


namespace new_cost_after_decrease_l87_87347

theorem new_cost_after_decrease (C new_C : ℝ) (hC : C = 1100) (h_decrease : new_C = 0.76 * C) : new_C = 836 :=
-- To be proved based on the given conditions
sorry

end new_cost_after_decrease_l87_87347


namespace reciprocal_of_neg_2023_l87_87005

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87005


namespace probability_div_int_l87_87106

theorem probability_div_int
    (r : ℤ) (k : ℤ)
    (hr : -5 < r ∧ r < 10)
    (hk : 1 < k ∧ k < 8)
    (hk_prime : Nat.Prime (Int.natAbs k)) :
    ∃ p q : ℕ, (p = 3 ∧ q = 14) ∧ p / q = 3 / 14 := 
by {
  sorry
}

end probability_div_int_l87_87106


namespace range_of_k_l87_87748

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- State the theorem
theorem range_of_k (k : ℝ) : (M ∩ N k).Nonempty ↔ k ∈ Set.Ici (-1) :=
by
  sorry

end range_of_k_l87_87748


namespace reciprocal_of_neg_2023_l87_87006

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l87_87006


namespace solve_x_l87_87617

theorem solve_x (x : ℝ) (h : (4 * x + 3) / (3 * x ^ 2 + 4 * x - 4) = 3 * x / (3 * x - 2)) :
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_x_l87_87617


namespace remaining_water_at_end_of_hike_l87_87561

-- Define conditions
def initial_water : ℝ := 9
def hike_length : ℝ := 7
def hike_duration : ℝ := 2
def leak_rate : ℝ := 1
def drink_rate_6_miles : ℝ := 0.6666666666666666
def drink_last_mile : ℝ := 2

-- Define the question and answer
def remaining_water (initial: ℝ) (duration: ℝ) (leak: ℝ) (drink6: ℝ) (drink_last: ℝ) : ℝ :=
  initial - ((drink6 * 6) + drink_last + (leak * duration))

-- Theorem stating the proof problem 
theorem remaining_water_at_end_of_hike :
  remaining_water initial_water hike_duration leak_rate drink_rate_6_miles drink_last_mile = 1 :=
by
  sorry

end remaining_water_at_end_of_hike_l87_87561


namespace max_product_l87_87268

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87268


namespace no_roots_ge_two_l87_87807

theorem no_roots_ge_two (x : ℝ) (h : x ≥ 2) : 4 * x^3 - 5 * x^2 - 6 * x + 3 ≠ 0 := by
  sorry

end no_roots_ge_two_l87_87807


namespace books_sold_l87_87622

theorem books_sold {total_books sold_fraction left_fraction : ℕ} (h_total : total_books = 9900)
    (h_fraction : left_fraction = 4/6) (h_sold : sold_fraction = 1 - left_fraction) : 
  (sold_fraction * total_books) = 3300 := 
  by 
  sorry

end books_sold_l87_87622


namespace coachClass_seats_count_l87_87674

-- Defining the conditions as given in a)
variables (F : ℕ) -- Number of first-class seats
variables (totalSeats : ℕ := 567) -- Total number of seats is given as 567
variables (businessClassSeats : ℕ := 3 * F) -- Business class seats defined in terms of F
variables (coachClassSeats : ℕ := 7 * F + 5) -- Coach class seats defined in terms of F
variables (firstClassSeats : ℕ := F) -- The variable itself

-- The statement to prove
theorem coachClass_seats_count : 
  F + businessClassSeats + coachClassSeats = totalSeats →
  coachClassSeats = 362 :=
by
  sorry -- The proof would go here

end coachClass_seats_count_l87_87674


namespace janelle_initial_green_marbles_l87_87083

def initial_green_marbles (blue_bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let blue_marbles := blue_bags * marbles_per_bag
  let remaining_blue_marbles := blue_marbles - gift_blue
  let remaining_green_marbles := remaining_marbles - remaining_blue_marbles
  remaining_green_marbles + gift_green

theorem janelle_initial_green_marbles :
  initial_green_marbles 6 10 6 8 72 = 26 :=
by
  rfl

end janelle_initial_green_marbles_l87_87083


namespace calen_pencils_loss_l87_87509

theorem calen_pencils_loss
  (P_Candy : ℕ)
  (P_Caleb : ℕ)
  (P_Calen_original : ℕ)
  (P_Calen_after_loss : ℕ)
  (h1 : P_Candy = 9)
  (h2 : P_Caleb = 2 * P_Candy - 3)
  (h3 : P_Calen_original = P_Caleb + 5)
  (h4 : P_Calen_after_loss = 10) :
  P_Calen_original - P_Calen_after_loss = 10 := 
sorry

end calen_pencils_loss_l87_87509


namespace student_allowance_l87_87666

def spend_on_clothes (A : ℚ) := (4 / 7) * A
def spend_on_games (A : ℚ) := (4 / 7) * (3 / 5) * A
def spend_on_books (A : ℚ) := (4 / 7) * (3 / 5) * (5 / 9) * A
def spend_on_charity (A : ℚ) := (4 / 7) * (3 / 5) * (5 / 9) * (1 / 2) * A
def remaining_after_candy (A : ℚ) := (2 / 21) * A - 3.75

theorem student_allowance :
  ∃ A : ℚ, remaining_after_candy A = 0 → A = 39.375 :=
begin
  sorry
end

end student_allowance_l87_87666


namespace no_real_m_perpendicular_l87_87992

theorem no_real_m_perpendicular (m : ℝ) : 
  ¬ ∃ m, ((m - 2) * m = -3) := 
sorry

end no_real_m_perpendicular_l87_87992


namespace mans_rate_in_still_water_l87_87928

theorem mans_rate_in_still_water : 
  ∀ (V_m V_s : ℝ), 
  V_m + V_s = 16 → 
  V_m - V_s = 4 → 
  V_m = 10 :=
by
  intros V_m V_s h1 h2
  sorry

end mans_rate_in_still_water_l87_87928


namespace max_product_l87_87266

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87266


namespace max_product_of_sum_2000_l87_87300

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87300


namespace Alyssa_weekly_allowance_l87_87038

theorem Alyssa_weekly_allowance : ∃ A : ℝ, (A / 2) + 8 = 12 ∧ A = 8 :=
by
  use 8
  split
  · sorry
  · sorry

end Alyssa_weekly_allowance_l87_87038


namespace larger_of_two_numbers_l87_87126

theorem larger_of_two_numbers (A B : ℕ) (hcf lcm : ℕ) (h1 : hcf = 23)
                              (h2 : lcm = hcf * 14 * 15) 
                              (h3 : lcm = A * B) (h4 : A = 23 * 14) 
                              (h5 : B = 23 * 15) : max A B = 345 :=
    sorry

end larger_of_two_numbers_l87_87126


namespace max_product_of_two_integers_sum_2000_l87_87313

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87313


namespace problem_statement_l87_87179

def Omega (n : ℕ) : ℕ := 
  -- Number of prime factors of n, counting multiplicity
  sorry

def f1 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 1 (mod 4)
  sorry

def f3 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 3 (mod 4)
  sorry

theorem problem_statement : 
  f3 (6 ^ 2020) - f1 (6 ^ 2020) = (1 / 10 : ℚ) * (6 ^ 2021 - 3 ^ 2021 - 2 ^ 2021 - 1) :=
sorry

end problem_statement_l87_87179


namespace sum_of_coeffs_l87_87517

theorem sum_of_coeffs (A B C D : ℤ) (h₁ : A = 1) (h₂ : B = -1) (h₃ : C = -12) (h₄ : D = 3) :
  A + B + C + D = -9 := 
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end sum_of_coeffs_l87_87517


namespace grid_values_equal_l87_87503

theorem grid_values_equal (f : ℤ × ℤ → ℕ) (h : ∀ (i j : ℤ), 
  f (i, j) = 1 / 4 * (f (i + 1, j) + f (i - 1, j) + f (i, j + 1) + f (i, j - 1))) :
  ∀ (i j i' j' : ℤ), f (i, j) = f (i', j') :=
by
  sorry

end grid_values_equal_l87_87503


namespace chairperson_and_committee_ways_l87_87728

-- Definitions based on conditions
def total_people : ℕ := 10
def ways_to_choose_chairperson : ℕ := total_people
def ways_to_choose_committee (remaining_people : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose remaining_people committee_size

-- The resulting theorem
theorem chairperson_and_committee_ways :
  ways_to_choose_chairperson * ways_to_choose_committee (total_people - 1) 3 = 840 :=
by
  sorry

end chairperson_and_committee_ways_l87_87728


namespace abs_neg_2023_eq_2023_l87_87113

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l87_87113


namespace range_of_a_l87_87990

-- Definition for set A
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = -|x| - 2 }

-- Definition for set B
def B (a : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - a)^2 + y^2 = a^2 }

-- Statement of the problem in Lean
theorem range_of_a (a : ℝ) : (∀ p, p ∈ A → p ∉ B a) → -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 := by
  sorry

end range_of_a_l87_87990


namespace soccer_camp_ratio_l87_87009

theorem soccer_camp_ratio :
  let total_kids := 2000
  let half_total := total_kids / 2
  let afternoon_camp := 750
  let morning_camp := half_total - afternoon_camp
  half_total ≠ 0 → 
  (morning_camp / half_total) = 1 / 4 := by
  sorry

end soccer_camp_ratio_l87_87009


namespace second_range_is_18_l87_87481

variable (range1 range2 range3 : ℕ)

theorem second_range_is_18
  (h1 : range1 = 30)
  (h2 : range2 = 18)
  (h3 : range3 = 32) :
  range2 = 18 := by
  sorry

end second_range_is_18_l87_87481


namespace convert_base_10_to_base_8_l87_87171

theorem convert_base_10_to_base_8 (n : ℕ) (n_eq : n = 3275) : 
  n = 3275 → ∃ (a b c d : ℕ), (a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 = 6323) :=
by 
  sorry

end convert_base_10_to_base_8_l87_87171


namespace unique_digit_sum_l87_87885

theorem unique_digit_sum (A B C D : ℕ) (h1 : A + B + C + D = 20) (h2 : B + A + 1 = 11) (uniq : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D)) : D = 8 :=
sorry

end unique_digit_sum_l87_87885


namespace total_guppies_l87_87348

noncomputable def initial_guppies : Nat := 7
noncomputable def baby_guppies_first_set : Nat := 3 * 12
noncomputable def baby_guppies_additional : Nat := 9

theorem total_guppies : initial_guppies + baby_guppies_first_set + baby_guppies_additional = 52 :=
by
  sorry

end total_guppies_l87_87348


namespace trains_clear_time_l87_87327

-- Definitions based on conditions
def length_train1 : ℕ := 160
def length_train2 : ℕ := 280
def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

-- Conversion factor from km/h to m/s
def kmph_to_mps (s : ℕ) : ℕ := s * 1000 / 3600

-- Computation of relative speed in m/s
def relative_speed_mps : ℕ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Total distance to be covered for the trains to clear each other
def total_distance : ℕ := length_train1 + length_train2

-- Time taken for the trains to clear each other
def time_to_clear_each_other : ℕ := total_distance / relative_speed_mps

-- Theorem stating that time taken is 22 seconds
theorem trains_clear_time : time_to_clear_each_other = 22 := by
  sorry

end trains_clear_time_l87_87327


namespace walking_time_l87_87498

theorem walking_time (intervals_time : ℕ) (poles_12_time : ℕ) (speed_constant : Prop) : 
  intervals_time = 2 → poles_12_time = 22 → speed_constant → 39 * intervals_time = 78 :=
by
  sorry

end walking_time_l87_87498


namespace collinear_c1_c2_l87_87805

def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (8, 3, -1)
def b : vec3 := (4, 1, 3)

def c1 : vec3 := (2 * 8 - 4, 2 * 3 - 1, 2 * (-1) - 3) -- (12, 5, -5)
def c2 : vec3 := (2 * 4 - 4 * 8, 2 * 1 - 4 * 3, 2 * 3 - 4 * (-1)) -- (-24, -10, 10)

theorem collinear_c1_c2 : ∃ γ : ℝ, c1 = (γ * -24, γ * -10, γ * 10) :=
  sorry

end collinear_c1_c2_l87_87805


namespace percentage_change_difference_l87_87500

-- Define the initial and final percentages of students
def initial_liked_percentage : ℝ := 0.4
def initial_disliked_percentage : ℝ := 0.6
def final_liked_percentage : ℝ := 0.8
def final_disliked_percentage : ℝ := 0.2

-- Define the problem statement
theorem percentage_change_difference :
  (final_liked_percentage - initial_liked_percentage) + 
  (initial_disliked_percentage - final_disliked_percentage) = 0.6 :=
sorry

end percentage_change_difference_l87_87500


namespace total_owed_proof_l87_87892

-- Define initial conditions
def initial_owed : ℕ := 20
def borrowed : ℕ := 8

-- Define the total amount owed
def total_owed : ℕ := initial_owed + borrowed

-- Prove the statement
theorem total_owed_proof : total_owed = 28 := 
by 
  -- Proof is omitted with sorry
  sorry

end total_owed_proof_l87_87892


namespace simplify_equation_l87_87422

open Fraction

theorem simplify_equation (x : ℚ) : (1 - (x + 3) / 3 = x / 2) → (6 - 2 * x - 6 = 3 * x) :=
by
  intro h
  sorry

end simplify_equation_l87_87422


namespace slower_speed_percentage_l87_87028

noncomputable def usual_speed_time : ℕ := 16  -- usual time in minutes
noncomputable def additional_time : ℕ := 24  -- additional time in minutes

theorem slower_speed_percentage (S S_slow : ℝ) (D : ℝ) 
  (h1 : D = S * usual_speed_time) 
  (h2 : D = S_slow * (usual_speed_time + additional_time)) : 
  (S_slow / S) * 100 = 40 :=
by 
  sorry

end slower_speed_percentage_l87_87028


namespace isosceles_triangle_length_l87_87234

theorem isosceles_triangle_length (a : ℝ) (h_graph_A : ∃ y, (a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2})
  (h_graph_B : ∃ y, (-a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2}) 
  (h_isosceles : ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
    dist (a, -a^2) O = dist (-a, -a^2) O ∧ dist (a, -a^2) (-a, -a^2) = dist (-a, -a^2) O) :
  dist (a, -a^2) (0, 0) = 2 * Real.sqrt 3 := sorry

end isosceles_triangle_length_l87_87234


namespace sum_of_areas_of_circles_l87_87494

-- Definitions of the conditions given in the problem
def triangle_side1 : ℝ := 6
def triangle_side2 : ℝ := 8
def triangle_side3 : ℝ := 10

-- Definitions of the radii r, s, t
variables (r s t : ℝ)

-- Conditions derived from the problem
axiom rs_eq : r + s = triangle_side1
axiom rt_eq : r + t = triangle_side2
axiom st_eq : s + t = triangle_side3

-- Main theorem to prove
theorem sum_of_areas_of_circles : (π * r^2) + (π * s^2) + (π * t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_circles_l87_87494


namespace correct_calculation_l87_87017

theorem correct_calculation (a b m : ℤ) : 
  (¬((a^3)^2 = a^5)) ∧ ((-2 * m^3)^2 = 4 * m^6) ∧ (¬(a^6 / a^2 = a^3)) ∧ (¬((a + b)^2 = a^2 + b^2)) := 
by
  sorry

end correct_calculation_l87_87017


namespace sector_central_angle_l87_87132

theorem sector_central_angle 
  (R : ℝ) (P : ℝ) (θ : ℝ) (π : ℝ) (L : ℝ)
  (h1 : P = 83) 
  (h2 : R = 14)
  (h3 : P = 2 * R + L)
  (h4 : L = θ * R)
  (degree_conversion : θ * (180 / π) = 225) : 
  θ * (180 / π) = 225 :=
by sorry

end sector_central_angle_l87_87132


namespace find_second_bank_account_balance_l87_87104

theorem find_second_bank_account_balance : 
  (exists (X : ℝ),  
    let raw_material_cost := 100
    let machinery_cost := 125
    let raw_material_tax := 0.05 * raw_material_cost
    let discounted_machinery_cost := machinery_cost - (0.1 * machinery_cost)
    let machinery_tax := 0.08 * discounted_machinery_cost
    let total_raw_material_cost := raw_material_cost + raw_material_tax
    let total_machinery_cost := discounted_machinery_cost + machinery_tax
    let total_spent := total_raw_material_cost + total_machinery_cost
    let total_cash := 900 + X
    let spent_proportion := 0.2 * total_cash
    total_spent = spent_proportion → X = 232.50) :=
by {
  sorry
}

end find_second_bank_account_balance_l87_87104


namespace evaluate_g_of_neg_one_l87_87172

def g (x : ℤ) : ℤ :=
  x^2 - 2*x + 1

theorem evaluate_g_of_neg_one :
  g (g (g (g (g (g (-1 : ℤ)))))) = 15738504 := by
  sorry

end evaluate_g_of_neg_one_l87_87172


namespace initial_volume_of_mixture_l87_87159

variable (V : ℝ)
variable (H1 : 0.2 * V + 12 = 0.25 * (V + 12))

theorem initial_volume_of_mixture (H : 0.2 * V + 12 = 0.25 * (V + 12)) : V = 180 := by
  sorry

end initial_volume_of_mixture_l87_87159


namespace abs_neg_number_l87_87115

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end abs_neg_number_l87_87115


namespace line_through_P_midpoint_l87_87935

noncomputable section

open Classical

variables (l l1 l2 : ℝ → ℝ → Prop) (P A B : ℝ × ℝ)

def line1 (x y : ℝ) := 2 * x - y - 2 = 0
def line2 (x y : ℝ) := x + y + 3 = 0

theorem line_through_P_midpoint (P A B : ℝ × ℝ)
  (hP : P = (3, 0))
  (hl1 : ∀ x y, line1 x y → l x y)
  (hl2 : ∀ x y, line2 x y → l x y)
  (hmid : (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2)) :
  ∃ k : ℝ, ∀ x y, (y = k * (x - 3)) ↔ (8 * x - y - 24 = 0) :=
by
  sorry

end line_through_P_midpoint_l87_87935


namespace algorithm_output_l87_87058

noncomputable def algorithm (x : ℝ) : ℝ :=
if x < 0 then x + 1 else -x^2

theorem algorithm_output :
  algorithm (-2) = -1 ∧ algorithm 3 = -9 :=
by
  -- proof omitted using sorry
  sorry

end algorithm_output_l87_87058


namespace tiffany_bags_found_day_after_next_day_l87_87258

noncomputable def tiffany_start : Nat := 10
noncomputable def tiffany_next_day : Nat := 3
noncomputable def tiffany_total : Nat := 20
noncomputable def tiffany_day_after_next_day : Nat := 20 - (tiffany_start + tiffany_next_day)

theorem tiffany_bags_found_day_after_next_day : tiffany_day_after_next_day = 7 := by
  sorry

end tiffany_bags_found_day_after_next_day_l87_87258


namespace fraction_sum_to_decimal_l87_87169

theorem fraction_sum_to_decimal :
  (3 / 20 : ℝ) + (5 / 200 : ℝ) + (7 / 2000 : ℝ) = 0.1785 :=
by 
  sorry

end fraction_sum_to_decimal_l87_87169


namespace problem_result_l87_87241

noncomputable def max_value (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) : ℝ :=
  2 * x^2 + x * y + y^2

theorem problem (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) :
  max_value x y hx = (75 + 60 * Real.sqrt 2) / 7 :=
sorry

theorem result : 75 + 60 + 2 + 7 = 144 :=
by norm_num

end problem_result_l87_87241


namespace hockey_cards_count_l87_87154

-- Define integer variables for the number of hockey, football and baseball cards
variables (H F B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := F = 4 * H
def condition2 : Prop := B = F - 50
def condition3 : Prop := H > 0
def condition4 : Prop := H + F + B = 1750

-- The theorem to prove
theorem hockey_cards_count 
  (h1 : condition1 H F)
  (h2 : condition2 F B)
  (h3 : condition3 H)
  (h4 : condition4 H F B) : 
  H = 200 := by
sorry

end hockey_cards_count_l87_87154


namespace quadratic_real_roots_prob_classical_correct_quadratic_real_roots_prob_geometric_correct_l87_87837

noncomputable def quadratic_real_roots_prob_classical (A : ℕ → ℕ → Prop) : ℚ :=
let events := {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)} in
have favorable_events := {(a, b) ∈ events | A a b},
(favorable_events.card / events.card : ℚ)

theorem quadratic_real_roots_prob_classical_correct :
  quadratic_real_roots_prob_classical (λ a b => a ≥ b) = 3/4 :=
sorry

noncomputable def quadratic_real_roots_prob_geometric (A : ℝ → ℝ → Prop) : ℚ :=
let volume_total := 6 in  -- Area of [0,3] x [0,2] region
let volume_event := 5 in  -- Area of region satisfying the condition a ≥ b
(volume_event / volume_total : ℚ)

theorem quadratic_real_roots_prob_geometric_correct :
  quadratic_real_roots_prob_geometric (λ a b => a ≥ b) = 2/3 :=
sorry

end quadratic_real_roots_prob_classical_correct_quadratic_real_roots_prob_geometric_correct_l87_87837


namespace range_of_m_l87_87848

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (m+1)*x^2 - m*x + m - 1 ≥ 0) ↔ m ≥ (2*Real.sqrt 3)/3 := by
  sorry

end range_of_m_l87_87848


namespace shot_put_surface_area_l87_87489

noncomputable def radius (d : ℝ) : ℝ := d / 2

noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem shot_put_surface_area :
  surface_area (radius 5) = 25 * Real.pi :=
by
  sorry

end shot_put_surface_area_l87_87489


namespace max_product_of_two_integers_sum_2000_l87_87308

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87308


namespace max_product_of_two_integers_sum_2000_l87_87293

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87293


namespace determine_digit_phi_l87_87684

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end determine_digit_phi_l87_87684


namespace part_a_part_b_l87_87092

theorem part_a (p : ℕ) (hp : Nat.Prime p) (a b : ℤ) (h : a ≡ b [ZMOD p]) : a ^ p ≡ b ^ p [ZMOD p^2] :=
  sorry

theorem part_b (p : ℕ) (hp : Nat.Prime p) : 
  Nat.card { n | n ∈ Finset.range (p^2) ∧ ∃ x, x ^ p ≡ n [ZMOD p^2] } = p :=
  sorry

end part_a_part_b_l87_87092


namespace total_games_won_l87_87506

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end total_games_won_l87_87506


namespace max_sqrt_sum_l87_87858

theorem max_sqrt_sum (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hxy : x + y = 8) :
  abs (Real.sqrt (x - 1 / y) + Real.sqrt (y - 1 / x)) ≤ Real.sqrt 15 :=
sorry

end max_sqrt_sum_l87_87858


namespace cylindrical_to_rectangular_coords_l87_87512

/--
Cylindrical coordinates (r, θ, z)
Rectangular coordinates (x, y, z)
-/
theorem cylindrical_to_rectangular_coords (r θ z : ℝ) (hx : x = r * Real.cos θ)
    (hy : y = r * Real.sin θ) (hz : z = z) :
    (r, θ, z) = (5, Real.pi / 4, 2) → (x, y, z) = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  sorry

end cylindrical_to_rectangular_coords_l87_87512


namespace no_four_digit_with_five_units_divisible_by_ten_l87_87718

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def units_place_is_five (n : ℕ) : Prop :=
  n % 10 = 5

def divisible_by_ten (n : ℕ) : Prop :=
  n % 10 = 0

theorem no_four_digit_with_five_units_divisible_by_ten : ∀ n : ℕ, 
  is_four_digit n → units_place_is_five n → ¬ divisible_by_ten n :=
by
  intro n h1 h2
  rw [units_place_is_five] at h2
  rw [divisible_by_ten, h2]
  sorry

end no_four_digit_with_five_units_divisible_by_ten_l87_87718


namespace black_haired_girls_count_l87_87589

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l87_87589


namespace max_value_of_a_l87_87384

open Real

theorem max_value_of_a (a : ℝ) :
  (∃ (l : linear_map ℝ ℝ ℝ), 
    ∃ (n m : ℝ), n > 0 ∧ m > 0 ∧ 
    (∀ x, l x = 2 * n * x - n^2) ∧ 
    (∀ x, l x = (a / m) * x + a * (log m - 1))
  ) → a ≤ 2 * exp 1 :=
begin
  sorry
end

end max_value_of_a_l87_87384


namespace compute_g_ggg2_l87_87901

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 5 then 2 * n + 2
  else 4 * n - 3

theorem compute_g_ggg2 : g (g (g 2)) = 65 :=
by
  sorry

end compute_g_ggg2_l87_87901


namespace junior_score_calculation_l87_87211

variable {total_students : ℕ}
variable {junior_score senior_average : ℕ}
variable {junior_ratio senior_ratio : ℚ}
variable {class_average total_average : ℚ}

-- Hypotheses from the conditions
theorem junior_score_calculation (h1 : junior_ratio = 0.2)
                               (h2 : senior_ratio = 0.8)
                               (h3 : class_average = 82)
                               (h4 : senior_average = 80)
                               (h5 : total_students = 10)
                               (h6 : total_average * total_students = total_students * class_average)
                               (h7 : total_average = (junior_ratio * junior_score + senior_ratio * senior_average))
                               : junior_score = 90 :=
sorry

end junior_score_calculation_l87_87211


namespace sin_pi_over_six_l87_87047

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 :=
sorry

end sin_pi_over_six_l87_87047


namespace max_product_l87_87267

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l87_87267


namespace limestone_amount_l87_87809

theorem limestone_amount (L S : ℝ) (h1 : L + S = 100) (h2 : 3 * L + 5 * S = 425) : L = 37.5 :=
by
  -- Proof will go here
  sorry

end limestone_amount_l87_87809


namespace point_in_fourth_quadrant_l87_87592

def point : ℝ × ℝ := (1, -2)

def is_fourth_quadrant (p: ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l87_87592


namespace Agnes_age_now_l87_87035

variable (A : ℕ) (J : ℕ := 6)

theorem Agnes_age_now :
  (2 * (J + 13) = A + 13) → A = 25 :=
by
  intro h
  sorry

end Agnes_age_now_l87_87035


namespace length_of_train_is_110_l87_87939

-- Define the speeds and time as constants
def speed_train_kmh := 90
def speed_man_kmh := 9
def time_pass_seconds := 4

-- Define the conversion factor from km/h to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh : ℚ) * (5 / 18)

-- Calculate relative speed in m/s
def relative_speed_mps : ℚ := kmh_to_mps (speed_train_kmh + speed_man_kmh)

-- Define the length of the train in meters
def length_of_train : ℚ := relative_speed_mps * time_pass_seconds

-- The theorem to prove: The length of the train is 110 meters
theorem length_of_train_is_110 : length_of_train = 110 := 
by sorry

end length_of_train_is_110_l87_87939


namespace beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l87_87180

def beautiful_association_number (x y a t : ℚ) : Prop :=
  |x - a| + |y - a| = t

theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 8 :=
by sorry

theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 4 ↔ x = 6 ∨ x = 0 :=
by sorry

theorem beautiful_association_number_part3 (x0 x1 x2 x3 x4 : ℚ) :
  beautiful_association_number x0 x1 1 1 ∧ 
  beautiful_association_number x1 x2 2 1 ∧ 
  beautiful_association_number x2 x3 3 1 ∧ 
  beautiful_association_number x3 x4 4 1 →
  x1 + x2 + x3 + x4 = 10 :=
by sorry

end beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l87_87180


namespace problem_l87_87767

def f (x : ℝ) : ℝ := sorry  -- f is a function from ℝ to ℝ

theorem problem (h : ∀ x : ℝ, 3 * f x + f (2 - x) = 4 * x^2 + 1) : f 5 = 133 / 4 := 
by 
  sorry -- the proof is omitted

end problem_l87_87767


namespace division_minutes_per_day_l87_87238

-- Define the conditions
def total_hours : ℕ := 5
def minutes_multiplication_per_day : ℕ := 10
def days_total : ℕ := 10

-- Convert hours to minutes
def total_minutes : ℕ := total_hours * 60

-- Total minutes spent on multiplication
def total_minutes_multiplication : ℕ := minutes_multiplication_per_day * days_total

-- Total minutes spent on division
def total_minutes_division : ℕ := total_minutes - total_minutes_multiplication

-- Minutes spent on division per day
def minutes_division_per_day : ℕ := total_minutes_division / days_total

-- The theorem to prove
theorem division_minutes_per_day : minutes_division_per_day = 20 := by
  sorry

end division_minutes_per_day_l87_87238


namespace correct_choices_l87_87386

/-- Definition of balls and events as per given conditions. --/
def num_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2

/-- Definition of events A and B. --/
def A : Prop := one_ball_drawn_is_red (first_draw)
def B : Prop := one_ball_drawn_is_red (second_draw)
def not_A : Prop := one_ball_drawn_is_white (first_draw)

/-- Probability of event A --/
axiom prob_A : ℙ(A) = 2/3

/-- Probability of event B given not A --/
axiom prob_notA_B : ℙ(B | not_A) = 4/5

/-- Theorem proving correct conclusions --/
theorem correct_choices :
  (ℙ(A) = 2/3) ∧ (ℙ(B | not_A) = 4/5) := by
  exact ⟨prob_A, prob_notA_B⟩

end correct_choices_l87_87386


namespace train_stops_15_min_per_hour_l87_87317

/-
Without stoppages, a train travels a certain distance with an average speed of 80 km/h,
and with stoppages, it covers the same distance with an average speed of 60 km/h.
Prove that the train stops for 15 minutes per hour.
-/
theorem train_stops_15_min_per_hour (D : ℝ) (h1 : 0 < D) :
  let T_no_stop := D / 80
  let T_stop := D / 60
  let T_lost := T_stop - T_no_stop
  let mins_per_hour := T_lost * 60
  mins_per_hour = 15 := by
  sorry

end train_stops_15_min_per_hour_l87_87317


namespace find_circle_equation_l87_87045

-- Define the conditions and problem
def circle_standard_equation (p1 p2 : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (xc, yc) := center
  (x2 - xc)^2 + (y2 - yc)^2 = radius^2

-- Define the conditions as given in the problem
def point_on_circle : Prop := circle_standard_equation (2, 0) (2, 2) (2, 2) 2

-- The main theorem to prove that the standard equation of the circle holds
theorem find_circle_equation : 
  point_on_circle →
  ∃ h k r, h = 2 ∧ k = 2 ∧ r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end find_circle_equation_l87_87045


namespace perpendicular_line_slope_l87_87875

theorem perpendicular_line_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x - 2 * y + 5 = 0 → x = 2 * y - 5)
  (h2 : ∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = - (2 / m) * x + 6 / m)
  (h3 : (1 / 2 : ℝ) * - (2 / m) = -1) : m = 1 :=
sorry

end perpendicular_line_slope_l87_87875


namespace g_increasing_on_interval_l87_87711

noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/5) * x + 13 * Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin ((1/5) * (x - 10 * Real.pi / 3) + 13 * Real.pi / 6)

theorem g_increasing_on_interval : ∀ x y : ℝ, (π ≤ x ∧ x < y ∧ y ≤ 2 * π) → g x < g y :=
by
  intro x y h
  -- Mathematical steps to prove this
  sorry

end g_increasing_on_interval_l87_87711


namespace TreyHasSevenTimesAsManyTurtles_l87_87010

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)

-- Conditions
def KristenHas12 : Kristen_turtles = 12 := sorry
def KrisHasQuarterOfKristen : Kris_turtles = Kristen_turtles / 4 := sorry
def TreyHas9MoreThanKristen : Trey_turtles = Kristen_turtles + 9 := sorry

-- Question: Prove that Trey has 7 times as many turtles as Kris
theorem TreyHasSevenTimesAsManyTurtles :
  Kristen_turtles = 12 → 
  Kris_turtles = Kristen_turtles / 4 → 
  Trey_turtles = Kristen_turtles + 9 → 
  Trey_turtles = 7 * Kris_turtles := sorry

end TreyHasSevenTimesAsManyTurtles_l87_87010


namespace negation_of_forall_log_gt_one_l87_87557

noncomputable def negation_of_p : Prop :=
∃ x : ℝ, Real.log x ≤ 1

theorem negation_of_forall_log_gt_one :
  (¬ (∀ x : ℝ, Real.log x > 1)) ↔ negation_of_p :=
by
  sorry

end negation_of_forall_log_gt_one_l87_87557


namespace positive_reals_power_equality_l87_87725

open Real

theorem positive_reals_power_equality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : a < 1) : a = b := 
  by
  sorry

end positive_reals_power_equality_l87_87725


namespace avg_of_first_5_numbers_equal_99_l87_87428

def avg_of_first_5 (S1 : ℕ) : ℕ := S1 / 5

theorem avg_of_first_5_numbers_equal_99
  (avg_9 : ℕ := 104) (avg_last_5 : ℕ := 100) (fifth_num : ℕ := 59)
  (sum_9 := 9 * avg_9) (sum_last_5 := 5 * avg_last_5) :
  avg_of_first_5 (sum_9 - sum_last_5 + fifth_num) = 99 :=
by
  sorry

end avg_of_first_5_numbers_equal_99_l87_87428


namespace infinite_primes_solutions_l87_87908

theorem infinite_primes_solutions :
  ∀ (P : Finset ℕ), (∀ p ∈ P, Prime p) →
  ∃ q, Prime q ∧ q ∉ P ∧ ∃ x y : ℤ, x^2 + x + 1 = q * y :=
by sorry

end infinite_primes_solutions_l87_87908


namespace opposite_face_of_orange_is_blue_l87_87025

structure CubeOrientation :=
  (top : String)
  (front : String)
  (right : String)

def first_view : CubeOrientation := { top := "B", front := "Y", right := "S" }
def second_view : CubeOrientation := { top := "B", front := "V", right := "S" }
def third_view : CubeOrientation := { top := "B", front := "K", right := "S" }

theorem opposite_face_of_orange_is_blue
  (colors : List String)
  (c1 : CubeOrientation)
  (c2 : CubeOrientation)
  (c3 : CubeOrientation)
  (no_orange_in_views : "O" ∉ colors.erase c1.top ∧ "O" ∉ colors.erase c1.front ∧ "O" ∉ colors.erase c1.right ∧
                         "O" ∉ colors.erase c2.top ∧ "O" ∉ colors.erase c2.front ∧ "O" ∉ colors.erase c2.right ∧
                         "O" ∉ colors.erase c3.top ∧ "O" ∉ colors.erase c3.front ∧ "O" ∉ colors.erase c3.right) :
  (c1.top = "B" → c2.top = "B" → c3.top = "B" → c1.right = "S" → c2.right = "S" → c3.right = "S" → 
  ∃ opposite_color, opposite_color = "B") :=
sorry

end opposite_face_of_orange_is_blue_l87_87025


namespace subtract_base8_l87_87168

theorem subtract_base8 (a b : ℕ) (h₁ : a = 0o2101) (h₂ : b = 0o1245) :
  a - b = 0o634 := sorry

end subtract_base8_l87_87168


namespace max_product_of_sum_2000_l87_87302

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l87_87302


namespace expression_one_expression_two_l87_87834

-- Define the expressions to be proved.
theorem expression_one : (3.6 - 0.8) * (1.8 + 2.05) = 10.78 :=
by sorry

theorem expression_two : (34.28 / 2) - (16.2 / 4) = 13.09 :=
by sorry

end expression_one_expression_two_l87_87834


namespace math_problem_l87_87699

variables {a b : ℝ}
open Real

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (a - 1) * (b - 1) = 1 ∧ 
  (∀ b : ℝ, (a = 2 * b → a + 4 * b = 9)) ∧ 
  (∀ b : ℝ, (b = 3 → (1 / a^2 + 2 / b^2) = 2 / 3)) :=
by
  sorry

end math_problem_l87_87699


namespace sin_odd_monotonically_increasing_l87_87019

theorem sin_odd_monotonically_increasing :
  (∀ x, sin (-x) = -sin x) ∧ (∀ x ∈ Ioo (0 : ℝ) 1, (deriv sin x > 0)) := by
  sorry

end sin_odd_monotonically_increasing_l87_87019


namespace no_valid_abc_l87_87491

theorem no_valid_abc : 
  ∀ (a b c : ℕ), (100 * a + 10 * b + c) % 15 = 0 → (10 * b + c) % 4 = 0 → a > b → b > c → false :=
by
  intros a b c habc_mod15 hbc_mod4 h_ab_gt h_bc_gt
  sorry

end no_valid_abc_l87_87491


namespace factor_of_change_l87_87762

-- Given conditions
def avg_marks_before : ℕ := 45
def avg_marks_after : ℕ := 90
def num_students : ℕ := 30

-- Prove the factor F by which marks are changed
theorem factor_of_change : ∃ F : ℕ, avg_marks_before * F = avg_marks_after := 
by
  use 2
  have h1 : 30 * avg_marks_before = 30 * 45 := rfl
  have h2 : 30 * avg_marks_after = 30 * 90 := rfl
  sorry

end factor_of_change_l87_87762


namespace find_a_l87_87741

variable (a : ℕ) (N : ℕ)
variable (h1 : Nat.gcd (2 * a + 1) (2 * a + 2) = 1) 
variable (h2 : Nat.gcd (2 * a + 1) (2 * a + 3) = 1)
variable (h3 : Nat.gcd (2 * a + 2) (2 * a + 3) = 2)
variable (hN : N = Nat.lcm (2 * a + 1) (Nat.lcm (2 * a + 2) (2 * a + 3)))
variable (hDiv : (2 * a + 4) ∣ N)

theorem find_a (h_pos : a > 0) : a = 1 :=
by
  -- Lean proof code will go here
  sorry

end find_a_l87_87741


namespace unique_representation_l87_87562

theorem unique_representation (n : ℕ) (h_pos : 0 < n) : 
  ∃! (a b : ℚ), a = 1 / n ∧ b = 1 / (n + 1) ∧ (a + b = (2 * n + 1) / (n * (n + 1))) :=
by
  sorry

end unique_representation_l87_87562


namespace perfect_square_trinomial_l87_87722

theorem perfect_square_trinomial (y : ℝ) (m : ℝ) : 
  (∃ b : ℝ, y^2 - m*y + 9 = (y + b)^2) → (m = 6 ∨ m = -6) :=
by
  intro h
  sorry

end perfect_square_trinomial_l87_87722


namespace equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l87_87264

-- Define a transformation predicate for words
inductive transform : List Char -> List Char -> Prop
| xy_to_yyx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 'y'] ++ l2) (l1 ++ ['y', 'y', 'x'] ++ l2)
| yyx_to_xy : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 'y', 'x'] ++ l2) (l1 ++ ['x', 'y'] ++ l2)
| xt_to_ttx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 't'] ++ l2) (l1 ++ ['t', 't', 'x'] ++ l2)
| ttx_to_xt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 't', 'x'] ++ l2) (l1 ++ ['x', 't'] ++ l2)
| yt_to_ty : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 't'] ++ l2) (l1 ++ ['t', 'y'] ++ l2)
| ty_to_yt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 'y'] ++ l2) (l1 ++ ['y', 't'] ++ l2)

-- Reflexive and transitive closure of transform
inductive transforms : List Char -> List Char -> Prop
| base : ∀ l, transforms l l
| step : ∀ l m n, transform l m → transforms m n → transforms l n

-- Definitions for the words and their information
def word1 := ['x', 'x', 'y', 'y']
def word2 := ['x', 'y', 'y', 'y', 'y', 'x']
def word3 := ['x', 'y', 't', 'x']
def word4 := ['t', 'x', 'y', 't']
def word5 := ['x', 'y']
def word6 := ['x', 't']

-- Proof statements
theorem equivalent_xy_xxyy : transforms word1 word2 :=
by sorry

theorem not_equivalent_xyty_txy : ¬ transforms word3 word4 :=
by sorry

theorem not_equivalent_xy_xt : ¬ transforms word5 word6 :=
by sorry

end equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l87_87264


namespace problem_true_propositions_l87_87830

-- Definitions
def is_square (q : ℕ) : Prop := q = 4
def is_trapezoid (q : ℕ) : Prop := q ≠ 4
def is_parallelogram (q : ℕ) : Prop := q = 2

-- Propositions
def prop_negation (p : Prop) : Prop := ¬ p
def prop_contrapositive (p q : Prop) : Prop := ¬ q → ¬ p
def prop_inverse (p q : Prop) : Prop := p → q

-- True propositions
theorem problem_true_propositions (a b c : ℕ) (h1 : ¬ (is_square 4)) (h2 : ¬ (is_parallelogram 3)) (h3 : ¬ (a * c^2 > b * c^2 → a > b)) : 
    (prop_negation (is_square 4) ∧ prop_contrapositive (is_trapezoid 3) (is_parallelogram 3)) ∧ ¬ prop_inverse (a * c^2 > b * c^2) (a > b) := 
by
    sorry

end problem_true_propositions_l87_87830


namespace total_pushups_l87_87320

theorem total_pushups (zachary_pushups : ℕ) (david_more_pushups : ℕ) 
  (h1 : zachary_pushups = 44) (h2 : david_more_pushups = 58) : 
  zachary_pushups + (zachary_pushups + david_more_pushups) = 146 :=
by
  sorry

end total_pushups_l87_87320


namespace john_needs_to_add_empty_cans_l87_87221

theorem john_needs_to_add_empty_cans :
  ∀ (num_full_cans : ℕ) (weight_per_full_can total_weight weight_per_empty_can required_weight : ℕ),
  num_full_cans = 6 →
  weight_per_full_can = 14 →
  total_weight = 88 →
  weight_per_empty_can = 2 →
  required_weight = total_weight - (num_full_cans * weight_per_full_can) →
  required_weight / weight_per_empty_can = 2 :=
by
  intros
  sorry

end john_needs_to_add_empty_cans_l87_87221


namespace range_of_a_l87_87191

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x = 1 → ¬ ((x + 1) / (x + a) < 2))) ↔ -1 ≤ a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l87_87191


namespace greatest_product_sum_2000_eq_1000000_l87_87288

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l87_87288


namespace line_segment_AB_length_l87_87374

noncomputable def length_AB (xA yA xB yB : ℝ) : ℝ :=
  Real.sqrt ((xA - xB)^2 + (yA - yB)^2)

theorem line_segment_AB_length :
  ∀ (xA yA xB yB : ℝ),
    (xA - yA = 0) →
    (xB + yB = 0) →
    (∃ k : ℝ, yA = k * (xA + 1) ∧ yB = k * (xB + 1)) →
    (-1 ≤ xA ∧ xA ≤ 0) →
    (xA + xB = 2 * k ∧ yA + yB = 2 * k) →
    length_AB xA yA xB yB = (4/3) * Real.sqrt 5 :=
by
  intros xA yA xB yB h1 h2 h3 h4 h5
  sorry

end line_segment_AB_length_l87_87374


namespace evaluate_expression_l87_87523

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end evaluate_expression_l87_87523


namespace weights_of_first_two_cats_l87_87397

noncomputable def cats_weight_proof (W : ℝ) : Prop :=
  (∀ (w1 w2 : ℝ), w1 = W ∧ w2 = W ∧ (w1 + w2 + 14.7 + 9.3) / 4 = 12) → (W = 12)

theorem weights_of_first_two_cats (W : ℝ) :
  cats_weight_proof W :=
by
  sorry

end weights_of_first_two_cats_l87_87397


namespace max_product_of_two_integers_sum_2000_l87_87307

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l87_87307


namespace total_expenditure_is_108_l87_87910

-- Define the costs of items and quantities purchased by Robert and Teddy
def cost_pizza := 10   -- cost of one box of pizza
def cost_soft_drink := 2  -- cost of one can of soft drink
def cost_hamburger := 3   -- cost of one hamburger

def qty_pizza_robert := 5     -- quantity of pizza boxes by Robert
def qty_soft_drink_robert := 10 -- quantity of soft drinks by Robert

def qty_hamburger_teddy := 6  -- quantity of hamburgers by Teddy
def qty_soft_drink_teddy := 10 -- quantity of soft drinks by Teddy

-- Calculate total expenditure for Robert and Teddy
def total_cost_robert := (qty_pizza_robert * cost_pizza) + (qty_soft_drink_robert * cost_soft_drink)
def total_cost_teddy := (qty_hamburger_teddy * cost_hamburger) + (qty_soft_drink_teddy * cost_soft_drink)

-- Total expenditure in all
def total_expenditure := total_cost_robert + total_cost_teddy

-- We formulate the theorem to prove that the total expenditure is $108
theorem total_expenditure_is_108 : total_expenditure = 108 :=
by 
  -- Placeholder proof
  sorry

end total_expenditure_is_108_l87_87910


namespace gym_membership_count_l87_87658

theorem gym_membership_count :
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  number_of_members = 300 :=
by
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  sorry

end gym_membership_count_l87_87658


namespace vector_c_equals_combination_l87_87877

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)
def vector_c : ℝ × ℝ := (-2, 4)

theorem vector_c_equals_combination : vector_c = (vector_a.1 - 3 * vector_b.1, vector_a.2 - 3 * vector_b.2) :=
sorry

end vector_c_equals_combination_l87_87877


namespace polynomial_evaluation_l87_87899

theorem polynomial_evaluation 
  (g : ℝ → ℝ) (h : ∀ x, g(x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x, g(x^2 - 2) = x^4 - 3 * x^2 - 3 := 
by
  intro x
  have h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1 := h x
  sorry

end polynomial_evaluation_l87_87899


namespace cake_stand_cost_calculation_l87_87692

-- Define the constants given in the problem
def flour_cost : ℕ := 5
def money_given : ℕ := 43
def change_received : ℕ := 10

-- Define the cost of the cake stand based on the problem's conditions
def cake_stand_cost : ℕ := (money_given - change_received) - flour_cost

-- The theorem we want to prove
theorem cake_stand_cost_calculation : cake_stand_cost = 28 :=
by
  sorry

end cake_stand_cost_calculation_l87_87692


namespace math_proof_problem_l87_87372

noncomputable def question_1 (x y : ℝ) (a b : ℝ) : Prop :=
  (x = -1) ∧ (y = (real.sqrt 2) / 2) ∧
  (a^2 = 2) ∧ (b^2 = 1) ∧
  ((x^2 / (a^2)) + (y^2 / (b^2)) = 1)

noncomputable def question_2 (λ S : ℝ) : Prop :=
  (λ >= 2 / 3) ∧ (λ <= 3 / 4) ∧
  (S >= (real.sqrt 6) / 4) ∧ (S <= 2 / 3)

theorem math_proof_problem : 
  ∀ x y a b λ S, 
  question_1 x y a b → question_2 λ S → 
  ((x^2 / 2 + y^2 = 1) ∧ (S >= (real.sqrt 6) / 4) ∧ (S <= 2 / 3)) :=
by sorry

end math_proof_problem_l87_87372


namespace tomatoes_grew_in_absence_l87_87389

def initial_tomatoes : ℕ := 36
def multiplier : ℕ := 100
def total_tomatoes_after_vacation : ℕ := initial_tomatoes * multiplier

theorem tomatoes_grew_in_absence : 
  total_tomatoes_after_vacation - initial_tomatoes = 3564 :=
by
  -- skipped proof with 'sorry'
  sorry

end tomatoes_grew_in_absence_l87_87389


namespace original_price_of_shirt_l87_87736

theorem original_price_of_shirt (P : ℝ) (h : 0.5625 * P = 18) : P = 32 := 
by 
sorry

end original_price_of_shirt_l87_87736


namespace units_digit_47_pow_47_l87_87453

theorem units_digit_47_pow_47 :
  let cycle := [7, 9, 3, 1] in
  cycle.nth (47 % 4) = some 3 :=
by
  let cycle := [7, 9, 3, 1]
  have h : 47 % 4 = 3 := by norm_num
  rw h
  simp
  exact trivial

end units_digit_47_pow_47_l87_87453


namespace r_exceeds_s_l87_87559

theorem r_exceeds_s (x y : ℚ) (h1 : x + 2 * y = 16 / 3) (h2 : 5 * x + 3 * y = 26) :
  x - y = 106 / 21 :=
sorry

end r_exceeds_s_l87_87559


namespace settle_debt_using_coins_l87_87633

theorem settle_debt_using_coins :
  ∃ n m : ℕ, 49 * n - 99 * m = 1 :=
sorry

end settle_debt_using_coins_l87_87633


namespace relationship_xy_qz_l87_87720

theorem relationship_xy_qz
  (a c b d : ℝ)
  (x y q z : ℝ)
  (h1 : a^(2 * x) = c^(2 * q) ∧ c^(2 * q) = b^2)
  (h2 : c^(3 * y) = a^(3 * z) ∧ a^(3 * z) = d^2) :
  x * y = q * z :=
by
  sorry

end relationship_xy_qz_l87_87720


namespace inequality_holds_l87_87417

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := sorry

end inequality_holds_l87_87417


namespace min_value_f_l87_87051

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

theorem min_value_f (a b : ℝ) (h : ∀ x : ℝ, 0 < x → f a b x ≤ 5) : 
  ∀ x : ℝ, x < 0 → f a b x ≥ -1 :=
by
  -- Since this is a statement-only problem, we leave the proof to be filled in
  sorry

end min_value_f_l87_87051


namespace minimize_PA2_PB2_PC2_l87_87243

def point : Type := ℝ × ℝ

noncomputable def distance_sq (P Q : point) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

noncomputable def PA_sq (P : point) : ℝ := distance_sq P (5, 0)
noncomputable def PB_sq (P : point) : ℝ := distance_sq P (0, 5)
noncomputable def PC_sq (P : point) : ℝ := distance_sq P (-4, -3)

noncomputable def circumcircle (P : point) : Prop := 
  P.1^2 + P.2^2 = 25

noncomputable def objective_function (P : point) : ℝ := 
  PA_sq P + PB_sq P + PC_sq P

theorem minimize_PA2_PB2_PC2 : ∃ P : point, circumcircle P ∧ 
  (∀ Q : point, circumcircle Q → objective_function P ≤ objective_function Q) :=
sorry

end minimize_PA2_PB2_PC2_l87_87243


namespace acme_vowel_soup_sequences_l87_87826

-- Define the vowels and their frequencies
def vowels : List (Char × ℕ) := [('A', 6), ('E', 6), ('I', 6), ('O', 4), ('U', 4)]

-- Noncomputable definition to calculate the total number of sequences
noncomputable def number_of_sequences : ℕ :=
  let single_vowel_choices := 6 + 6 + 6 + 4 + 4
  single_vowel_choices^5

-- Theorem stating the number of five-letter sequences
theorem acme_vowel_soup_sequences : number_of_sequences = 11881376 := by
  sorry

end acme_vowel_soup_sequences_l87_87826


namespace units_digit_47_pow_47_l87_87464

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l87_87464


namespace part_I_part_II_l87_87521

theorem part_I (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_adbc: a * d = b * c) (h_ineq1: a + d > b + c): |a - d| > |b - c| :=
sorry

theorem part_II (a b c d t: ℝ) 
(h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(h_eq: t * (Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2)) = Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)):
t >= Real.sqrt 2 :=
sorry

end part_I_part_II_l87_87521


namespace max_product_of_two_integers_sum_2000_l87_87295

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l87_87295


namespace lean_proof_problem_l87_87407

section

variable {R : Type*} [AddCommGroup R]

def is_odd_function (f : ℝ → R) : Prop :=
  ∀ x, f (-x) = -f x

theorem lean_proof_problem (f: ℝ → ℝ) (h_odd: is_odd_function f)
    (h_cond: f 3 + f (-2) = 2) : f 2 - f 3 = -2 :=
by
  sorry

end

end lean_proof_problem_l87_87407


namespace card_draw_sequential_same_suit_l87_87938

theorem card_draw_sequential_same_suit : 
  let hearts := 13
  let diamonds := 13
  let total_suits := hearts + diamonds
  ∃ ways : ℕ, ways = total_suits * (hearts - 1) :=
by
  sorry

end card_draw_sequential_same_suit_l87_87938


namespace distance_between_foci_of_hyperbola_l87_87974

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), x^2 - 6 * x - 4 * y^2 - 8 * y = 27 → (4 * Real.sqrt 10) = 4 * Real.sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l87_87974


namespace symmetric_point_xOz_l87_87887

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end symmetric_point_xOz_l87_87887


namespace initial_HNO3_percentage_is_correct_l87_87811

def initial_percentage_of_HNO3 (P : ℚ) : Prop :=
  let initial_volume := 60
  let added_volume := 18
  let final_volume := 78
  let final_percentage := 50
  (P / 100) * initial_volume + added_volume = (final_percentage / 100) * final_volume

theorem initial_HNO3_percentage_is_correct :
  initial_percentage_of_HNO3 35 :=
by
  sorry

end initial_HNO3_percentage_is_correct_l87_87811


namespace paintings_on_Sep27_l87_87960

-- Definitions for the problem conditions
def total_days := 6
def paintings_per_2_days := (6 : ℕ)
def paintings_per_3_days := (8 : ℕ)
def paintings_P22_to_P26 := 30

-- Function to calculate paintings over a given period
def paintings_in_days (days : ℕ) (frequency : ℕ) : ℕ := days / frequency

-- Function to calculate total paintings from the given artists
def total_paintings (d : ℕ) (p2 : ℕ) (p3 : ℕ) : ℕ :=
  p2 * paintings_in_days d 2 + p3 * paintings_in_days d 3

-- Calculate total paintings in 6 days
def total_paintings_in_6_days := total_paintings total_days paintings_per_2_days paintings_per_3_days

-- Proof problem: Show the number of paintings on the last day (September 27)
theorem paintings_on_Sep27 : total_paintings_in_6_days - paintings_P22_to_P26 = 4 :=
by
  sorry

end paintings_on_Sep27_l87_87960


namespace value_of_a3_l87_87392

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem value_of_a3 (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 + a 4 = 20) :
  a 2 = 4 :=
sorry

end value_of_a3_l87_87392


namespace sum_of_coefficients_l87_87679

def P (x : ℝ) : ℝ :=
  -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : P 1 = 48 := by
  sorry

end sum_of_coefficients_l87_87679


namespace inequality_sin_values_l87_87174

theorem inequality_sin_values :
  let a := Real.sin (-5)
  let b := Real.sin 3
  let c := Real.sin 5
  a > b ∧ b > c :=
by
  sorry

end inequality_sin_values_l87_87174


namespace soccer_team_goals_l87_87164

theorem soccer_team_goals (total_players total_goals games_played : ℕ)
(one_third_players_goals : ℕ) :
  total_players = 24 →
  total_goals = 150 →
  games_played = 15 →
  one_third_players_goals = (total_players / 3) * games_played →
  (total_goals - one_third_players_goals) = 30 :=
by
  intros h1 h2 h3 h4
  rw h1 at h4
  rw h3 at h4
  sorry

end soccer_team_goals_l87_87164


namespace identify_counterfeit_coins_l87_87367

theorem identify_counterfeit_coins (m : ℕ) (coins : Finset ℕ)
  (h₁ : coins.card = 4^m)
  (h₂ : ∃ (G C : Finset ℕ), G.card = C.card ∧ G.card = 2^(2 * m) ∧ ∀ c ∈ C, c < ∀ g ∈ G, g) :
  ∃ weigh_method : list (Finset ℕ × Finset ℕ), 
    weigh_method.length ≤ 3^m ∧ 
    ∀ (w : Finset ℕ × Finset ℕ) in weigh_method,
    (∀ g₁ g₂ ∈ coins, g₁ ∈ w.1 → g₂ ∈ w.2),
    (∀ g₃ g₄ ∈ coins, g₃ = g₄ → g₃ ∈ w.1 → g₄ ∈ w.2) :=
sorry

end identify_counterfeit_coins_l87_87367


namespace lcm_12_18_is_36_l87_87527

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l87_87527


namespace radius_of_smaller_circle_l87_87337

theorem radius_of_smaller_circle (A1 : ℝ) (r1 r2 : ℝ) (h1 : π * r2^2 = 4 * A1)
    (h2 : r2 = 4) : r1 = 2 :=
by
  sorry

end radius_of_smaller_circle_l87_87337


namespace wendy_walked_l87_87449

theorem wendy_walked (x : ℝ) (h1 : 19.83 = x + 10.67) : x = 9.16 :=
sorry

end wendy_walked_l87_87449


namespace abs_neg_2023_eq_2023_l87_87112

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l87_87112


namespace problem1_problem2_l87_87194

-- Definitions and conditions
def A (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }
def B : Set ℝ := { x | x < -6 ∨ x > 1 }

-- (Ⅰ) Problem statement: Prove that if A ∩ B = ∅, then -6 ≤ m ≤ 0.
theorem problem1 (m : ℝ) : A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 := 
by
  sorry

-- (Ⅱ) Problem statement: Prove that if A ⊆ B, then m < -7 or m > 1.
theorem problem2 (m : ℝ) : A m ⊆ B ↔ m < -7 ∨ m > 1 := 
by
  sorry

end problem1_problem2_l87_87194


namespace sufficient_but_not_necessary_condition_l87_87764

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 4 * x < 0) ∧ ¬ (∀ x : ℝ, x^2 - 4 * x < 0 → 0 < x ∧ x < 5) :=
sorry

end sufficient_but_not_necessary_condition_l87_87764


namespace grandpa_max_pieces_l87_87378

theorem grandpa_max_pieces (m n : ℕ) (h : (m - 3) * (n - 3) = 9) : m * n = 112 :=
sorry

end grandpa_max_pieces_l87_87378


namespace units_digit_47_pow_47_l87_87459

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l87_87459


namespace father_current_age_l87_87322

variable (M F : ℕ)

/-- The man's current age is (2 / 5) of the age of his father. -/
axiom man_age : M = (2 / 5) * F

/-- After 12 years, the man's age will be (1 / 2) of his father's age. -/
axiom age_relation_in_12_years : (M + 12) = (1 / 2) * (F + 12)

/-- Prove that the father's current age, F, is 60. -/
theorem father_current_age : F = 60 :=
by
  sorry

end father_current_age_l87_87322


namespace find_x_l87_87850

theorem find_x (x : ℝ) : 9999 * x = 724787425 ↔ x = 72487.5 := 
sorry

end find_x_l87_87850


namespace money_bounds_l87_87852

   theorem money_bounds (a b : ℝ) (h₁ : 4 * a + 2 * b > 110) (h₂ : 2 * a + 3 * b = 105) : a > 15 ∧ b < 25 :=
   by
     sorry
   
end money_bounds_l87_87852


namespace solve_for_x_l87_87609

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l87_87609


namespace range_of_a_l87_87204

variable {x a : ℝ}

theorem range_of_a (h1 : 2 * x - a < 0)
                   (h2 : 1 - 2 * x ≥ 7)
                   (h3 : ∀ x, x ≤ -3) : ∀ a, a > -6 :=
by
  sorry

end range_of_a_l87_87204


namespace max_product_two_integers_sum_2000_l87_87279

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l87_87279


namespace find_y_interval_l87_87564

theorem find_y_interval (y : ℝ) (h : y^2 - 8 * y + 12 < 0) : 2 < y ∧ y < 6 :=
sorry

end find_y_interval_l87_87564


namespace blue_balls_unchanged_l87_87205

def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5
def added_yellow_balls : ℕ := 4

theorem blue_balls_unchanged :
  initial_blue_balls = 2 := by
  sorry

end blue_balls_unchanged_l87_87205


namespace no_integer_solutions_l87_87969

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 17 * y^3 = 50 := 
by 
  sorry

end no_integer_solutions_l87_87969


namespace frac_3125_over_1024_gt_e_l87_87478

theorem frac_3125_over_1024_gt_e : (3125 : ℝ) / 1024 > Real.exp 1 := sorry

end frac_3125_over_1024_gt_e_l87_87478


namespace Sophie_l87_87425

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end Sophie_l87_87425


namespace lcm_12_18_l87_87533

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l87_87533


namespace solve_for_x_l87_87615

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l87_87615


namespace variation_of_powers_l87_87996

theorem variation_of_powers (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end variation_of_powers_l87_87996


namespace brothers_travel_distance_l87_87436

theorem brothers_travel_distance
  (x : ℝ)
  (hb_x : (120 : ℝ) / (x : ℝ) - 4 = (120 : ℝ) / (x + 40))
  (total_time : 2 = 2) :
  x = 20 ∧ (x + 40) = 60 :=
by
  -- we need to prove the distances
  sorry

end brothers_travel_distance_l87_87436


namespace prove_Φ_eq_8_l87_87682

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end prove_Φ_eq_8_l87_87682


namespace fraction_is_three_fourths_l87_87626

-- Define the number
def n : ℝ := 8.0

-- Define the fraction
variable (x : ℝ)

-- The main statement to be proved
theorem fraction_is_three_fourths
(h : x * n + 2 = 8) : x = 3 / 4 :=
sorry

end fraction_is_three_fourths_l87_87626


namespace grace_wait_time_l87_87196

variable (hose1_rate : ℕ) (hose2_rate : ℕ) (pool_capacity : ℕ) (time_after_second_hose : ℕ)
variable (h : ℕ)

theorem grace_wait_time 
  (h1 : hose1_rate = 50)
  (h2 : hose2_rate = 70)
  (h3 : pool_capacity = 390)
  (h4 : time_after_second_hose = 2) : 
  50 * h + (50 + 70) * 2 = 390 → h = 3 :=
by
  sorry

end grace_wait_time_l87_87196


namespace ratio_of_larger_to_smaller_l87_87918

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
a / b

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a * b = 50) (h3 : a > b) :
  ratio_of_numbers a b = 4 / 3 :=
sorry

end ratio_of_larger_to_smaller_l87_87918


namespace intersection_result_l87_87705

noncomputable def A : Set ℝ := { x | x^2 - 5*x - 6 < 0 }
noncomputable def B : Set ℝ := { x | 2022^x > Real.sqrt 2022 }
noncomputable def intersection : Set ℝ := { x | A x ∧ B x }

theorem intersection_result : intersection = Set.Ioo (1/2 : ℝ) 6 := by
  sorry

end intersection_result_l87_87705


namespace units_digit_47_power_47_l87_87458

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l87_87458


namespace longer_subsegment_of_YZ_l87_87734

/-- In triangle XYZ with sides in the ratio 3:4:5, and side YZ being 12 cm.
    The angle bisector XW divides side YZ into segments YW and ZW.
    Prove that the length of ZW is 48/7 cm. --/
theorem longer_subsegment_of_YZ (YZ : ℝ) (hYZ : YZ = 12)
    (XY XZ : ℝ) (hRatio : XY / XZ = 3 / 4) : 
    ∃ ZW : ℝ, ZW = 48 / 7 :=
by
  -- We would provide proof here
  sorry

end longer_subsegment_of_YZ_l87_87734


namespace sphere_intersection_radius_l87_87343

theorem sphere_intersection_radius (r : ℝ) :
  (let center := (3 : ℝ, 5, -8),
       xy_center := (3, 5, 0),
       yz_center := (0, 5, -8),
       xy_radius := 2,
       sphere_radius := real.sqrt (2^2 + (xy_center.3 - center.3)^2) in
   (real.sqrt (r^2 + (yz_center.1 - center.1)^2) = sphere_radius)) :=
by
  let center := (3 : ℝ, 5, -8),
      xy_center := (3, 5, 0),
      yz_center := (0, 5, -8),
      xy_radius := 2,
      sphere_radius := real.sqrt (2^2 + (xy_center.3 - center.3)^2)
  use (real.sqrt 59)
  sorry

end sphere_intersection_radius_l87_87343


namespace line_slope_y_intercept_l87_87573

theorem line_slope_y_intercept :
  (∃ (a b : ℝ), (∀ (x y : ℝ), (x = 3 → y = 7 → y = a * x + b) ∧ (x = 7 → y = 19 → y = a * x + b)) ∧ (a - b = 5)) :=
begin
  sorry
end

end line_slope_y_intercept_l87_87573


namespace systematic_sampling_method_l87_87336

def num_rows : ℕ := 50
def num_seats_per_row : ℕ := 30

def is_systematic_sampling (select_interval : ℕ) : Prop :=
  ∀ n, select_interval = n * num_seats_per_row + 8

theorem systematic_sampling_method :
  is_systematic_sampling 30 :=
by
  sorry

end systematic_sampling_method_l87_87336


namespace max_area_of_triangle_l87_87632

theorem max_area_of_triangle (AB BC AC : ℝ) (ratio : BC / AC = 3 / 5) (hAB : AB = 10) :
  ∃ A : ℝ, (A ≤ 260.52) :=
sorry

end max_area_of_triangle_l87_87632


namespace dog_ate_cost_is_six_l87_87401

-- Definitions for the costs
def flour_cost : ℝ := 4
def sugar_cost : ℝ := 2
def butter_cost : ℝ := 2.5
def eggs_cost : ℝ := 0.5

-- Total cost calculation
def total_cost := flour_cost + sugar_cost + butter_cost + eggs_cost

-- Initial slices and remaining slices
def initial_slices : ℕ := 6
def eaten_slices : ℕ := 2
def remaining_slices := initial_slices - eaten_slices

-- The cost calculation of the amount the dog ate
def dog_ate_cost := (remaining_slices / initial_slices) * total_cost

-- Proof statement
theorem dog_ate_cost_is_six : dog_ate_cost = 6 :=
by
  sorry

end dog_ate_cost_is_six_l87_87401


namespace hyeoncheol_initial_money_l87_87379

theorem hyeoncheol_initial_money
  (X : ℕ)
  (h1 : X / 2 / 2 = 1250) :
  X = 5000 :=
sorry

end hyeoncheol_initial_money_l87_87379


namespace determine_a_range_l87_87866

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem determine_a_range (e : ℝ) (he : e = Real.exp 1) :
  ∃ a_range : Set ℝ, a_range = Set.Icc 1 (e + 1 / e) :=
by 
  sorry

end determine_a_range_l87_87866


namespace number_of_blue_lights_l87_87136

-- Conditions
def total_colored_lights : Nat := 95
def red_lights : Nat := 26
def yellow_lights : Nat := 37
def blue_lights : Nat := total_colored_lights - (red_lights + yellow_lights)

-- Statement we need to prove
theorem number_of_blue_lights : blue_lights = 32 := by
  sorry

end number_of_blue_lights_l87_87136


namespace expand_expression_l87_87175

variable (x y : ℝ)

theorem expand_expression :
  12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 :=
by
  sorry

end expand_expression_l87_87175


namespace geometric_sequence_problem_l87_87095

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * (Real.log x)
  else (Real.log x) / x

theorem geometric_sequence_problem
  (a : ℕ → ℝ) 
  (r : ℝ)
  (h1 : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h2 : a 3 * a 4 * a 5 = 1)
  (h3 : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1) :
  a 1 = Real.exp 2 :=
sorry

end geometric_sequence_problem_l87_87095


namespace normal_dist_prob_l87_87985

noncomputable theory
open ProbabilityTheory MeasureTheory

variables {Ω : Type*} [MeasureSpace Ω]

-- Defining the random variable ξ with normal distribution N(3, 16)
def xi : Ω → ℝ := sorry
axiom xi_normal : Normal ℝ 3 4 xi

-- Statement of the proof problem
theorem normal_dist_prob : Pξ < 3) = 0.5 :=
by
  sorry

end normal_dist_prob_l87_87985


namespace range_of_f_on_interval_l87_87135

-- Definition of the function
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Definition of the interval
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The main statement
theorem range_of_f_on_interval : 
  ∀ y, (∃ x, domain x ∧ f x = y) ↔ (1 ≤ y ∧ y ≤ 10) :=
by
  sorry

end range_of_f_on_interval_l87_87135


namespace plane_through_points_l87_87845

noncomputable def plane_equation : ℝ × ℝ × ℝ × ℝ := (2, 3, -4, 9)

theorem plane_through_points :
  ∃ (A B C D : ℝ), 
    -- Plane equation in the form Ax + By + Cz + D = 0
    plane_equation = (A, B, C, D) ∧
    -- Points (2, -3, 1), (6, -3, 3), (4, -5, 2) lie on the plane
    (A * 2 + B * -3 + C * 1 + D = 0) ∧
    (A * 6 + B * -3 + C * 3 + D = 0) ∧
    (A * 4 + B * -5 + C * 2 + D = 0) ∧
    -- Additional constraints: A > 0 and gcd(|A|, |B|, |C|, |D|) = 1
    (A > 0) ∧ (Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1) :=
begin
  use [2, 3, -4, 9],
  split, exact rfl,
  split, norm_num,
  split, norm_num,
  split, norm_num,
  split,
  exact zero_lt_two,
  norm_num,
end

end plane_through_points_l87_87845


namespace sum_first_8_terms_arithmetic_sequence_l87_87730

theorem sum_first_8_terms_arithmetic_sequence (a : ℕ → ℝ) (h : a 4 + a 5 = 12) :
    (8 * (a 1 + a 8)) / 2 = 48 :=
by
  sorry

end sum_first_8_terms_arithmetic_sequence_l87_87730


namespace perpendicular_vectors_l87_87717

theorem perpendicular_vectors (x y : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (2 + x, 1 - y)) 
  (hperp : (a.1 * b.1 + a.2 * b.2 = 0)) : 2 * y - x = 4 :=
sorry

end perpendicular_vectors_l87_87717


namespace greatest_product_sum_2000_l87_87275

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l87_87275


namespace calculate_hardcover_volumes_l87_87961

theorem calculate_hardcover_volumes (h p : ℕ) 
  (h_total_volumes : h + p = 12)
  (h_cost_equation : 27 * h + 16 * p = 284)
  (h_p_relation : p = 12 - h) : h = 8 :=
by
  sorry

end calculate_hardcover_volumes_l87_87961


namespace incircle_contact_point_l87_87549

-- Definitions for the problem setup
variables (F₁ F₂ : Point) -- Foci of the hyperbola
variables (M N : Point)   -- Vertices of the hyperbola
variables (P : Point)     -- Point on the hyperbola

-- Hypotheses
hypothesis H_p_on_hyperbola : IsOnHyperbola P F₁ F₂ M N
hypothesis H_incircle : HasIncircle (Triangle.mk P F₁ F₂)

-- The theorem statement
theorem incircle_contact_point (G : Point) :
  IncircleContactPoint (Triangle.mk P F₁ F₂) F₁ F₂ G →
  G = M ∨ G = N := 
sorry 

end incircle_contact_point_l87_87549


namespace polyhedron_edges_vertices_l87_87922

theorem polyhedron_edges_vertices (F : ℕ) (triangular_faces : Prop) (hF : F = 20) : ∃ S A : ℕ, S = 12 ∧ A = 30 :=
by
  -- stating the problem conditions and desired conclusion
  sorry

end polyhedron_edges_vertices_l87_87922
