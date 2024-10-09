import Mathlib

namespace problem_equiv_conditions_l1099_109960

theorem problem_equiv_conditions (n : ℕ) :
  (∀ a : ℕ, n ∣ a^n - a) ↔ (∀ p : ℕ, p ∣ n → Prime p → ¬ p^2 ∣ n ∧ (p - 1) ∣ (n - 1)) :=
sorry

end problem_equiv_conditions_l1099_109960


namespace rebecca_income_l1099_109961

variable (R : ℝ) -- Rebecca's current yearly income (denoted as R)
variable (increase : ℝ := 7000) -- The increase in Rebecca's income
variable (jimmy_income : ℝ := 18000) -- Jimmy's yearly income
variable (combined_income : ℝ := (R + increase) + jimmy_income) -- Combined income after increase
variable (new_income_ratio : ℝ := 0.55) -- Proportion of total income that is Rebecca's new income

theorem rebecca_income : (R + increase) = new_income_ratio * combined_income → R = 15000 :=
by
  sorry

end rebecca_income_l1099_109961


namespace max_value_of_f_l1099_109969

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, ∃ k : ℤ, f x = 3 ∧ x = k * Real.pi :=
by
  -- The proof is omitted
  sorry

end max_value_of_f_l1099_109969


namespace original_calculation_l1099_109987

theorem original_calculation
  (x : ℝ)
  (h : ((x * 3) + 14) * 2 = 946) :
  ((x / 3) + 14) * 2 = 130 :=
sorry

end original_calculation_l1099_109987


namespace tile_difference_is_11_l1099_109948

-- Define the initial number of blue and green tiles
def initial_blue_tiles : ℕ := 13
def initial_green_tiles : ℕ := 6

-- Define the number of additional green tiles added as border
def additional_green_tiles : ℕ := 18

-- Define the total number of green tiles in the new figure
def total_green_tiles : ℕ := initial_green_tiles + additional_green_tiles

-- Define the total number of blue tiles in the new figure (remains the same)
def total_blue_tiles : ℕ := initial_blue_tiles

-- Define the difference between the total number of green tiles and blue tiles
def tile_difference : ℕ := total_green_tiles - total_blue_tiles

-- The theorem stating that the difference between the total number of green tiles 
-- and the total number of blue tiles in the new figure is 11
theorem tile_difference_is_11 : tile_difference = 11 := by
  sorry

end tile_difference_is_11_l1099_109948


namespace quadrant_of_P_l1099_109929

theorem quadrant_of_P (m n : ℝ) (h1 : m * n > 0) (h2 : m + n < 0) : (m < 0 ∧ n < 0) :=
by
  sorry

end quadrant_of_P_l1099_109929


namespace value_of_ab_plus_bc_plus_ca_l1099_109938

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end value_of_ab_plus_bc_plus_ca_l1099_109938


namespace gcd_bn_bn1_l1099_109993

def b (n : ℕ) : ℤ := (7^n - 1) / 6
def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 1))

theorem gcd_bn_bn1 (n : ℕ) : e n = 1 := by
  sorry

end gcd_bn_bn1_l1099_109993


namespace tim_kittens_l1099_109941

theorem tim_kittens (K : ℕ) (h1 : (3 / 5 : ℚ) * (2 / 3 : ℚ) * K = 12) : K = 30 :=
sorry

end tim_kittens_l1099_109941


namespace find_total_coins_l1099_109970

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l1099_109970


namespace estimate_blue_balls_l1099_109977

theorem estimate_blue_balls (total_balls : ℕ) (prob_yellow : ℚ)
  (h_total : total_balls = 80)
  (h_prob_yellow : prob_yellow = 0.25) :
  total_balls * (1 - prob_yellow) = 60 :=
by
  -- proof
  sorry

end estimate_blue_balls_l1099_109977


namespace speed_goods_train_l1099_109995

def length_train : ℝ := 50
def length_platform : ℝ := 250
def time_crossing : ℝ := 15

/-- The speed of the goods train in km/hr given the length of the train, the length of the platform, and the time to cross the platform. -/
theorem speed_goods_train :
  (length_train + length_platform) / time_crossing * 3.6 = 72 :=
by
  sorry

end speed_goods_train_l1099_109995


namespace work_completion_days_l1099_109940

theorem work_completion_days (A B : Type) (A_work_rate B_work_rate : ℝ) :
  (1 / 16 : ℝ) = (1 / 20) + A_work_rate → B_work_rate = (1 / 80) := by
  sorry

end work_completion_days_l1099_109940


namespace problem_bounds_l1099_109903

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

end problem_bounds_l1099_109903


namespace service_cleaning_fee_percentage_is_correct_l1099_109911

noncomputable def daily_rate : ℝ := 125
noncomputable def pet_fee : ℝ := 100
noncomputable def duration : ℕ := 14
noncomputable def security_deposit_percentage : ℝ := 0.5
noncomputable def security_deposit : ℝ := 1110

noncomputable def total_expected_cost : ℝ := (daily_rate * duration) + pet_fee
noncomputable def entire_bill : ℝ := security_deposit / security_deposit_percentage
noncomputable def service_cleaning_fee : ℝ := entire_bill - total_expected_cost

theorem service_cleaning_fee_percentage_is_correct : 
  (service_cleaning_fee / entire_bill) * 100 = 16.67 :=
by 
  sorry

end service_cleaning_fee_percentage_is_correct_l1099_109911


namespace original_class_strength_l1099_109921

theorem original_class_strength 
  (x : ℕ) 
  (h1 : ∀ a_avg n, a_avg = 40 → n = x)
  (h2 : ∀ b_avg m, b_avg = 32 → m = 12)
  (h3 : ∀ new_avg, new_avg = 36 → ((x * 40 + 12 * 32) = ((x + 12) * 36))) : 
  x = 12 :=
by 
  sorry

end original_class_strength_l1099_109921


namespace regions_bounded_by_blue_lines_l1099_109907

theorem regions_bounded_by_blue_lines (n : ℕ) : 
  (2 * n^2 + 3 * n + 2) -(n - 1) * (2 * n + 1) ≥ 4 * n + 2 :=
by
  sorry

end regions_bounded_by_blue_lines_l1099_109907


namespace exist_ints_a_b_for_any_n_l1099_109953

theorem exist_ints_a_b_for_any_n (n : ℤ) : ∃ a b : ℤ, n = Int.floor (a * Real.sqrt 2) + Int.floor (b * Real.sqrt 3) := by
  sorry

end exist_ints_a_b_for_any_n_l1099_109953


namespace total_pages_read_l1099_109949

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end total_pages_read_l1099_109949


namespace work_completion_time_l1099_109945

theorem work_completion_time 
  (M W : ℝ) 
  (h1 : (10 * M + 15 * W) * 6 = 1) 
  (h2 : M * 100 = 1) 
  : W * 225 = 1 := 
by
  sorry

end work_completion_time_l1099_109945


namespace bread_left_in_pond_l1099_109967

theorem bread_left_in_pond (total_bread : ℕ) 
                           (half_bread_duck : ℕ)
                           (second_duck_bread : ℕ)
                           (third_duck_bread : ℕ)
                           (total_bread_thrown : total_bread = 100)
                           (half_duck_eats : half_bread_duck = total_bread / 2)
                           (second_duck_eats : second_duck_bread = 13)
                           (third_duck_eats : third_duck_bread = 7) :
                           total_bread - (half_bread_duck + second_duck_bread + third_duck_bread) = 30 :=
    by
    sorry

end bread_left_in_pond_l1099_109967


namespace hexagon_coloring_unique_l1099_109994

-- Define the coloring of the hexagon using enumeration
inductive Color
  | green
  | blue
  | orange

-- Assume we have a function that represents the coloring of the hexagons
-- with the constraints given in the problem
def is_valid_coloring (coloring : ℕ → ℕ → Color) : Prop :=
  ∀ x y : ℕ, -- For all hexagons
  (coloring x y = Color.green ∧ x = 0 ∧ y = 0) ∨ -- The labeled hexagon G is green
  (coloring x y ≠ coloring (x + 1) y ∧ -- No two hexagons with a common side have the same color
   coloring x y ≠ coloring (x - 1) y ∧ 
   coloring x y ≠ coloring x (y + 1) ∧
   coloring x y ≠ coloring x (y - 1))

-- The problem is to prove there are exactly 2 valid colorings of the hexagon grid
theorem hexagon_coloring_unique :
  ∃ (count : ℕ), count = 2 ∧
  ∀ coloring : (ℕ → ℕ → Color), is_valid_coloring coloring → count = 2 :=
by
  sorry

end hexagon_coloring_unique_l1099_109994


namespace minimum_toothpicks_removal_l1099_109965

theorem minimum_toothpicks_removal
    (num_toothpicks : ℕ) 
    (num_triangles : ℕ) 
    (h1 : num_toothpicks = 40) 
    (h2 : num_triangles > 35) :
    ∃ (min_removal : ℕ), min_removal = 15 
    := 
    sorry

end minimum_toothpicks_removal_l1099_109965


namespace specific_natural_numbers_expr_l1099_109984

theorem specific_natural_numbers_expr (a b c : ℕ) 
  (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1) : 
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ (n = (a + b) / c + (b + c) / a + (c + a) / b) :=
by sorry

end specific_natural_numbers_expr_l1099_109984


namespace eval_nested_fractions_l1099_109906

theorem eval_nested_fractions : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end eval_nested_fractions_l1099_109906


namespace find_k_all_reals_l1099_109912

theorem find_k_all_reals (a b c : ℝ) : 
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
sorry

end find_k_all_reals_l1099_109912


namespace minimum_value_expression_l1099_109931

theorem minimum_value_expression :
  ∀ (r s t : ℝ), (1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4) →
  (r - 1) ^ 2 + (s / r - 1) ^ 2 + (t / s - 1) ^ 2 + (4 / t - 1) ^ 2 = 4 * (Real.sqrt 2 - 1) ^ 2 := 
sorry

end minimum_value_expression_l1099_109931


namespace child_ticket_cost_l1099_109971

/-- Defining the conditions and proving the cost of a child's ticket --/
theorem child_ticket_cost:
  (∀ c: ℕ, 
      -- Revenue from Monday
      (7 * c + 5 * 4) + 
      -- Revenue from Tuesday
      (4 * c + 2 * 4) = 
      -- Total revenue for both days
      61 
    ) → 
    -- Proving c
    (c = 3) :=
by
  sorry

end child_ticket_cost_l1099_109971


namespace probability_of_female_selection_probability_of_male_host_selection_l1099_109950

/-!
In a competition, there are eight contestants consisting of five females and three males.
If three contestants are chosen randomly to progress to the next round, what is the 
probability that all selected contestants are female? Additionally, from those who 
do not proceed, one is selected as a host. What is the probability that this host is male?
-/

noncomputable def number_of_ways_select_3_from_8 : ℕ := Nat.choose 8 3

noncomputable def number_of_ways_select_3_females_from_5 : ℕ := Nat.choose 5 3

noncomputable def probability_all_3_females : ℚ := number_of_ways_select_3_females_from_5 / number_of_ways_select_3_from_8

noncomputable def number_of_remaining_contestants : ℕ := 8 - 3

noncomputable def number_of_males_remaining : ℕ := 3 - 1

noncomputable def number_of_ways_select_1_male_from_2 : ℕ := Nat.choose 2 1

noncomputable def number_of_ways_select_1_from_5 : ℕ := Nat.choose 5 1

noncomputable def probability_host_is_male : ℚ := number_of_ways_select_1_male_from_2 / number_of_ways_select_1_from_5

theorem probability_of_female_selection : probability_all_3_females = 5 / 28 := by
  sorry

theorem probability_of_male_host_selection : probability_host_is_male = 2 / 5 := by
  sorry

end probability_of_female_selection_probability_of_male_host_selection_l1099_109950


namespace number_of_teams_l1099_109974

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end number_of_teams_l1099_109974


namespace find_x_l1099_109936

-- Define the vectors and the condition of them being parallel
def vector_a : (ℝ × ℝ) := (3, 1)
def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -1)
def parallel (a b : (ℝ × ℝ)) := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The theorem to prove
theorem find_x (x : ℝ) (h : parallel (3, 1) (x, -1)) : x = -3 :=
by
  sorry

end find_x_l1099_109936


namespace ken_kept_pencils_l1099_109908

def ken_total_pencils := 50
def pencils_given_to_manny := 10
def pencils_given_to_nilo := pencils_given_to_manny + 10
def pencils_given_away := pencils_given_to_manny + pencils_given_to_nilo

theorem ken_kept_pencils : ken_total_pencils - pencils_given_away = 20 := by
  sorry

end ken_kept_pencils_l1099_109908


namespace reformulate_and_find_product_l1099_109997

theorem reformulate_and_find_product (a b x y : ℝ)
  (h : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 2)) :
  ∃ m' n' p' : ℤ, (a^m' * x - a^n') * (a^p' * y - a^3) = a^5 * b^5 ∧ m' * n' * p' = 48 :=
by
  sorry

end reformulate_and_find_product_l1099_109997


namespace num_people_got_on_bus_l1099_109973

-- Definitions based on the conditions
def initialNum : ℕ := 4
def currentNum : ℕ := 17
def peopleGotOn (initial : ℕ) (current : ℕ) : ℕ := current - initial

-- Theorem statement
theorem num_people_got_on_bus : peopleGotOn initialNum currentNum = 13 := 
by {
  sorry -- Placeholder for the proof
}

end num_people_got_on_bus_l1099_109973


namespace rotation_90_deg_l1099_109990

theorem rotation_90_deg (z : ℂ) (r : ℂ → ℂ) (h : ∀ (x y : ℝ), r (x + y*I) = -y + x*I) :
  r (8 - 5*I) = 5 + 8*I :=
by sorry

end rotation_90_deg_l1099_109990


namespace find_sphere_volume_l1099_109910

noncomputable def sphere_volume (d: ℝ) (V: ℝ) : Prop := d = 3 * (16 / 9) * V

theorem find_sphere_volume :
  sphere_volume (2 / 3) (1 / 6) :=
by
  sorry

end find_sphere_volume_l1099_109910


namespace find_second_expression_l1099_109999

theorem find_second_expression (a : ℕ) (x : ℕ) 
  (h1 : (2 * a + 16 + x) / 2 = 74) (h2 : a = 28) : x = 76 := 
by
  sorry

end find_second_expression_l1099_109999


namespace profit_percent_calculation_l1099_109915

variable (SP : ℝ) (CP : ℝ) (Profit : ℝ) (ProfitPercent : ℝ)
variable (h1 : CP = 0.75 * SP)
variable (h2 : Profit = SP - CP)
variable (h3 : ProfitPercent = (Profit / CP) * 100)

theorem profit_percent_calculation : ProfitPercent = 33.33 := 
sorry

end profit_percent_calculation_l1099_109915


namespace expression_value_l1099_109920

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 :=
by
  rw [hx, hy, hz]
  -- After substituting the values, we will need to simplify the expression to reach 20.
  sorry

end expression_value_l1099_109920


namespace heavy_rain_duration_l1099_109959

-- Define the conditions as variables and constants
def initial_volume := 100 -- Initial volume in liters
def final_volume := 280   -- Final volume in liters
def flow_rate := 2        -- Flow rate in liters per minute

-- Define the duration query as a theorem to be proved
theorem heavy_rain_duration : 
  (final_volume - initial_volume) / flow_rate = 90 := 
by
  sorry

end heavy_rain_duration_l1099_109959


namespace number_of_people_in_village_l1099_109980

variable (P : ℕ) -- Define the total number of people in the village

def people_not_working : ℕ := 50
def people_with_families : ℕ := 25
def people_singing_in_shower : ℕ := 75
def max_people_overlap : ℕ := 50

theorem number_of_people_in_village :
  P - people_not_working + P - people_with_families + P - people_singing_in_shower - max_people_overlap = P → 
  P = 100 :=
by
  sorry

end number_of_people_in_village_l1099_109980


namespace volume_and_area_of_pyramid_l1099_109964

-- Define the base of the pyramid.
def rect (EF FG : ℕ) : Prop := EF = 10 ∧ FG = 6

-- Define the perpendicular relationships and height of the pyramid.
def pyramid (EF FG PE : ℕ) : Prop := 
  rect EF FG ∧
  PE = 10 ∧ 
  (PE > 0) -- Given conditions include perpendicular properties, implying height is positive.

-- Problem translation: Prove the volume and area calculations.
theorem volume_and_area_of_pyramid (EF FG PE : ℕ) 
  (h1 : rect EF FG) 
  (h2 : PE = 10) : 
  (1 / 3 * EF * FG * PE = 200 ∧ 1 / 2 * EF * FG = 30) := 
by
  sorry

end volume_and_area_of_pyramid_l1099_109964


namespace ferry_speed_difference_l1099_109981

theorem ferry_speed_difference :
  let V_p := 6
  let Time_P := 3
  let Distance_P := V_p * Time_P
  let Distance_Q := 2 * Distance_P
  let Time_Q := Time_P + 1
  let V_q := Distance_Q / Time_Q
  V_q - V_p = 3 := by
  sorry

end ferry_speed_difference_l1099_109981


namespace solve_for_question_mark_l1099_109935

def cube_root (x : ℝ) := x^(1/3)
def square_root (x : ℝ) := x^(1/2)

theorem solve_for_question_mark : 
  cube_root (5568 / 87) + square_root (72 * 2) = square_root 256 := by
  sorry

end solve_for_question_mark_l1099_109935


namespace all_iterated_quadratic_eq_have_integer_roots_l1099_109916

noncomputable def initial_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 + x2 = -p ∧ x1 * x2 = q

noncomputable def iterated_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∀ i : ℕ, i ≤ 9 → ∃ x1 x2 : ℤ, x1 + x2 = -(p + i) ∧ x1 * x2 = (q + i)

theorem all_iterated_quadratic_eq_have_integer_roots :
  ∃ p q : ℤ, initial_quadratic_eq_has_integer_roots p q ∧ iterated_quadratic_eq_has_integer_roots p q :=
sorry

end all_iterated_quadratic_eq_have_integer_roots_l1099_109916


namespace newborn_members_approximation_l1099_109952

-- Defining the conditions
def survival_prob_first_month : ℚ := 7/8
def survival_prob_second_month : ℚ := 7/8
def survival_prob_third_month : ℚ := 7/8
def survival_prob_three_months : ℚ := (7/8) ^ 3
def expected_survivors : ℚ := 133.984375

-- Statement to prove that the number of newborn members, N, approximates to 200
theorem newborn_members_approximation (N : ℚ) : 
  N * survival_prob_three_months = expected_survivors → 
  N = 200 :=
by
  sorry

end newborn_members_approximation_l1099_109952


namespace total_rocks_needed_l1099_109943

def rocks_already_has : ℕ := 64
def rocks_needed : ℕ := 61

theorem total_rocks_needed : rocks_already_has + rocks_needed = 125 :=
by
  sorry

end total_rocks_needed_l1099_109943


namespace platform_length_l1099_109979

variable (L : ℝ) -- The length of the platform
variable (train_length : ℝ := 300) -- The length of the train
variable (time_pole : ℝ := 26) -- Time to cross the signal pole
variable (time_platform : ℝ := 39) -- Time to cross the platform

theorem platform_length :
  (train_length / time_pole) = (train_length + L) / time_platform → L = 150 := sorry

end platform_length_l1099_109979


namespace inscribed_sphere_surface_area_l1099_109902

theorem inscribed_sphere_surface_area (V S : ℝ) (hV : V = 2) (hS : S = 3) : 4 * Real.pi * (3 * V / S)^2 = 16 * Real.pi := by
  sorry

end inscribed_sphere_surface_area_l1099_109902


namespace prove_central_angle_of_sector_l1099_109937

noncomputable def central_angle_of_sector (R α : ℝ) : Prop :=
  (2 * R + R * α = 8) ∧ (1 / 2 * α * R^2 = 4)

theorem prove_central_angle_of_sector :
  ∃ α R : ℝ, central_angle_of_sector R α ∧ α = 2 :=
sorry

end prove_central_angle_of_sector_l1099_109937


namespace polar_to_cartesian_l1099_109927

theorem polar_to_cartesian (ρ : ℝ) (θ : ℝ) (hx : ρ = 3) (hy : θ = π / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.cos (π / 6), 3 * Real.sin (π / 6)) := by
  sorry

end polar_to_cartesian_l1099_109927


namespace halfway_fraction_l1099_109991

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l1099_109991


namespace range_of_m_l1099_109928

variable {R : Type} [LinearOrderedField R]

def discriminant (a b c : R) : R := b^2 - 4 * a * c

theorem range_of_m (m : R) :
  (discriminant (1:R) m (m + 3) > 0) ↔ (m < -2 ∨ m > 6) :=
by
  sorry

end range_of_m_l1099_109928


namespace fraction_multiplication_l1099_109939

-- Given fractions a and b
def a := (1 : ℚ) / 4
def b := (1 : ℚ) / 8

-- The first product result
def result1 := a * b

-- The final product result when multiplied by 4
def result2 := result1 * 4

-- The theorem to prove
theorem fraction_multiplication : result2 = (1 : ℚ) / 8 := by
  sorry

end fraction_multiplication_l1099_109939


namespace max_values_of_x_max_area_abc_l1099_109978

noncomputable def m (x : ℝ) : ℝ × ℝ := ⟨2 * Real.sin x, Real.sin x - Real.cos x⟩
noncomputable def n (x : ℝ) : ℝ × ℝ := ⟨Real.sqrt 3 * Real.cos x, Real.sin x + Real.cos x⟩
noncomputable def f (x : ℝ) : ℝ := Prod.fst (m x) * Prod.fst (n x) + Prod.snd (m x) * Prod.snd (n x)

theorem max_values_of_x
  (k : ℤ) : ∃ x, x = k * Real.pi + Real.pi / 3 ∧ f x = 2 * Real.sin (2 * x - π / 6) :=
sorry

noncomputable def C : ℝ := Real.pi / 3
noncomputable def area_abc (a b c : ℝ) : ℝ := 1 / 2 * a * b * Real.sin C

theorem max_area_abc (a b : ℝ) (h₁ : c = Real.sqrt 3) (h₂ : f C = 2) :
  area_abc a b c ≤ 3 * Real.sqrt 3 / 4 :=
sorry

end max_values_of_x_max_area_abc_l1099_109978


namespace distinct_positive_integers_factors_PQ_RS_l1099_109947

theorem distinct_positive_integers_factors_PQ_RS (P Q R S : ℕ) (hP : P > 0) (hQ : Q > 0) (hR : R > 0) (hS : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDistinctPQ : P ≠ Q) (hDistinctRS : R ≠ S) (hPQR_S : P + Q = R - S) :
  P = 4 :=
by
  sorry

end distinct_positive_integers_factors_PQ_RS_l1099_109947


namespace unique_N_l1099_109900

-- Given conditions and question in the problem
variable (N : Matrix (Fin 2) (Fin 2) ℝ)

-- Problem statement: prove that the matrix defined below is the only matrix satisfying the given condition
theorem unique_N 
  (h : ∀ (w : Fin 2 → ℝ), N.mulVec w = -7 • w) 
  : N = ![![-7, 0], ![0, -7]] := 
sorry

end unique_N_l1099_109900


namespace opposite_numbers_l1099_109975

theorem opposite_numbers (a b : ℤ) (h1 : -5^2 = a) (h2 : (-5)^2 = b) : a = -b :=
by sorry

end opposite_numbers_l1099_109975


namespace maci_pays_total_cost_l1099_109954

def cost_blue_pen : ℝ := 0.10
def num_blue_pens : ℕ := 10
def num_red_pens : ℕ := 15
def cost_red_pen : ℝ := 2 * cost_blue_pen

def total_cost : ℝ := num_blue_pens * cost_blue_pen + num_red_pens * cost_red_pen

theorem maci_pays_total_cost : total_cost = 4 := by
  -- Proof goes here
  sorry

end maci_pays_total_cost_l1099_109954


namespace sin_add_pi_over_2_l1099_109982

theorem sin_add_pi_over_2 (θ : ℝ) (h : Real.cos θ = -3 / 5) : Real.sin (θ + π / 2) = -3 / 5 :=
sorry

end sin_add_pi_over_2_l1099_109982


namespace smallest_integer_value_l1099_109926

theorem smallest_integer_value (x : ℤ) (h : 3 * |x| + 8 < 29) : x = -6 :=
sorry

end smallest_integer_value_l1099_109926


namespace total_pencils_is_220_l1099_109918

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end total_pencils_is_220_l1099_109918


namespace no_integer_roots_quadratic_l1099_109917

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end no_integer_roots_quadratic_l1099_109917


namespace square_binomial_l1099_109925

theorem square_binomial (x : ℝ) : (-x - 1) ^ 2 = x^2 + 2 * x + 1 :=
by
  sorry

end square_binomial_l1099_109925


namespace total_coffee_needed_l1099_109934

-- Conditions as definitions
def weak_coffee_amount_per_cup : ℕ := 1
def strong_coffee_amount_per_cup : ℕ := 2 * weak_coffee_amount_per_cup
def cups_of_weak_coffee : ℕ := 12
def cups_of_strong_coffee : ℕ := 12

-- Prove that the total amount of coffee needed equals 36 tablespoons
theorem total_coffee_needed : (weak_coffee_amount_per_cup * cups_of_weak_coffee) + (strong_coffee_amount_per_cup * cups_of_strong_coffee) = 36 :=
by
  sorry

end total_coffee_needed_l1099_109934


namespace root_implies_value_l1099_109951

theorem root_implies_value (b c : ℝ) (h : 2 * b - c = 4) : 4 * b - 2 * c + 1 = 9 :=
by
  sorry

end root_implies_value_l1099_109951


namespace goods_train_length_l1099_109996

theorem goods_train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) 
    (h_speed : speed_kmph = 72) (h_platform : platform_length_m = 250) (h_time : time_s = 24) : 
    ∃ train_length_m : ℕ, train_length_m = 230 := 
by 
  sorry

end goods_train_length_l1099_109996


namespace total_savings_correct_l1099_109905

theorem total_savings_correct :
  let price_chlorine := 10
  let discount1_chlorine := 0.20
  let discount2_chlorine := 0.10
  let price_soap := 16
  let discount1_soap := 0.25
  let discount2_soap := 0.05
  let price_wipes := 8
  let bogo_discount_wipes := 0.50
  let quantity_chlorine := 4
  let quantity_soap := 6
  let quantity_wipes := 8
  let final_chlorine_price := (price_chlorine * (1 - discount1_chlorine)) * (1 - discount2_chlorine)
  let final_soap_price := (price_soap * (1 - discount1_soap)) * (1 - discount2_soap)
  let final_wipes_price_per_two := price_wipes + price_wipes * bogo_discount_wipes
  let final_wipes_price := final_wipes_price_per_two / 2
  let total_original_price := quantity_chlorine * price_chlorine + quantity_soap * price_soap + quantity_wipes * price_wipes
  let total_final_price := quantity_chlorine * final_chlorine_price + quantity_soap * final_soap_price + quantity_wipes * final_wipes_price
  let total_savings := total_original_price - total_final_price
  total_savings = 55.80 :=
by sorry

end total_savings_correct_l1099_109905


namespace unique_function_eq_id_l1099_109989

theorem unique_function_eq_id (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f x = x^2 * f (1 / x)) →
  (∀ x y : ℝ, f (x + y) = f x + f y) →
  (f 1 = 1) →
  (∀ x : ℝ, f x = x) :=
by
  intro h1 h2 h3
  sorry

end unique_function_eq_id_l1099_109989


namespace base10_to_base4_156_eq_2130_l1099_109966

def base10ToBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem base10_to_base4_156_eq_2130 :
  base10ToBase4 156 = [2, 1, 3, 0] := sorry

end base10_to_base4_156_eq_2130_l1099_109966


namespace net_effect_sale_value_l1099_109933

variable (P Q : ℝ) -- New price and quantity sold

theorem net_effect_sale_value (P Q : ℝ) :
  let new_sale_value := (0.75 * P) * (1.75 * Q)
  let original_sale_value := P * Q
  new_sale_value - original_sale_value = 0.3125 * (P * Q) := 
by
  sorry

end net_effect_sale_value_l1099_109933


namespace distinct_exponentiation_values_l1099_109955

theorem distinct_exponentiation_values : 
  ∃ (standard other1 other2 other3 : ℕ), 
    standard ≠ other1 ∧ 
    standard ≠ other2 ∧ 
    standard ≠ other3 ∧ 
    other1 ≠ other2 ∧ 
    other1 ≠ other3 ∧ 
    other2 ≠ other3 := 
sorry

end distinct_exponentiation_values_l1099_109955


namespace cleaning_time_is_correct_l1099_109985

-- Define the given conditions
def vacuuming_minutes_per_day : ℕ := 30
def vacuuming_days_per_week : ℕ := 3
def dusting_minutes_per_day : ℕ := 20
def dusting_days_per_week : ℕ := 2

-- Define the total cleaning time per week
def total_cleaning_time_per_week : ℕ :=
  (vacuuming_minutes_per_day * vacuuming_days_per_week) + (dusting_minutes_per_day * dusting_days_per_week)

-- State the theorem we want to prove
theorem cleaning_time_is_correct : total_cleaning_time_per_week = 130 := by
  sorry

end cleaning_time_is_correct_l1099_109985


namespace problem_part_1_problem_part_2_l1099_109922

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2

theorem problem_part_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  (vector_a x).1 * vector_b.2 = (vector_a x).2 * vector_b.1 → 
  x = 5 * Real.pi / 6 :=
by
  sorry

theorem problem_part_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) :
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≤ f t) → x = 0 ∧ f 0 = 3 ∧ 
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≥ f t) → x = 5 * Real.pi / 6 ∧ f (5 * Real.pi / 6) = -2 * Real.sqrt 3 :=
by
  sorry

end problem_part_1_problem_part_2_l1099_109922


namespace factory_minimize_salary_l1099_109913

theorem factory_minimize_salary :
  ∃ x : ℕ, ∃ W : ℕ,
    x + (120 - x) = 120 ∧
    800 * x + 1000 * (120 - x) = W ∧
    120 - x ≥ 3 * x ∧
    x = 30 ∧
    W = 114000 :=
  sorry

end factory_minimize_salary_l1099_109913


namespace remainder_t4_mod7_l1099_109968

def T : ℕ → ℕ
| 0 => 0 -- Not used
| 1 => 6
| n+1 => 6 ^ (T n)

theorem remainder_t4_mod7 : (T 4 % 7) = 6 := by
  sorry

end remainder_t4_mod7_l1099_109968


namespace each_person_ate_2_cakes_l1099_109930

def initial_cakes : ℕ := 8
def number_of_friends : ℕ := 4

theorem each_person_ate_2_cakes (h_initial_cakes : initial_cakes = 8)
  (h_number_of_friends : number_of_friends = 4) :
  initial_cakes / number_of_friends = 2 :=
by sorry

end each_person_ate_2_cakes_l1099_109930


namespace peanuts_total_correct_l1099_109963

def initial_peanuts : ℕ := 4
def added_peanuts : ℕ := 6
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_total_correct : total_peanuts = 10 := by
  sorry

end peanuts_total_correct_l1099_109963


namespace a_ge_zero_of_set_nonempty_l1099_109919

theorem a_ge_zero_of_set_nonempty {a : ℝ} (h : ∃ x : ℝ, x^2 = a) : a ≥ 0 :=
sorry

end a_ge_zero_of_set_nonempty_l1099_109919


namespace min_value_inequality_l1099_109998

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_inequality_l1099_109998


namespace third_player_game_count_l1099_109958

theorem third_player_game_count (fp_games : ℕ) (sp_games : ℕ) (tp_games : ℕ) (total_games : ℕ) 
  (h1 : fp_games = 10) (h2 : sp_games = 21) (h3 : total_games = sp_games) 
  (h4 : total_games = fp_games + tp_games + 1): tp_games = 11 := 
  sorry

end third_player_game_count_l1099_109958


namespace negative_cube_root_l1099_109924

theorem negative_cube_root (a : ℝ) : ∃ x : ℝ, x ^ 3 = -a^2 - 1 ∧ x < 0 :=
by
  sorry

end negative_cube_root_l1099_109924


namespace vertical_asymptotes_count_l1099_109992

theorem vertical_asymptotes_count : 
  let f (x : ℝ) := (x - 2) / (x^2 + 4*x - 5) 
  ∃! c : ℕ, c = 2 :=
by
  sorry

end vertical_asymptotes_count_l1099_109992


namespace find_a_l1099_109914

theorem find_a (a : ℝ) (i : ℂ) (hi : i = Complex.I) (z : ℂ) (hz : z = a + i) (h : z^2 + z = 1 - 3 * Complex.I) :
  a = -2 :=
by {
  sorry
}

end find_a_l1099_109914


namespace greatest_integer_a_exists_l1099_109932

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end greatest_integer_a_exists_l1099_109932


namespace smallest_four_digit_number_l1099_109976

theorem smallest_four_digit_number :
  ∃ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ n : ℕ, 21 * m = n^2) ∧ m = 1029 :=
by sorry

end smallest_four_digit_number_l1099_109976


namespace krystiana_monthly_earnings_l1099_109957

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l1099_109957


namespace nina_total_amount_l1099_109909

theorem nina_total_amount:
  ∃ (x y z w : ℕ), 
  x + y + z + w = 27 ∧
  y = 2 * z ∧
  z = 2 * x ∧
  7 < w ∧ w < 20 ∧
  10 * x + 5 * y + 2 * z + 3 * w = 107 :=
by 
  sorry

end nina_total_amount_l1099_109909


namespace charlie_steps_proof_l1099_109972

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end charlie_steps_proof_l1099_109972


namespace largest_sum_l1099_109956

theorem largest_sum :
  max (max (max (max (1/4 + 1/9) (1/4 + 1/10)) (1/4 + 1/11)) (1/4 + 1/12)) (1/4 + 1/13) = 13/36 := 
sorry

end largest_sum_l1099_109956


namespace range_of_m_l1099_109904

theorem range_of_m (m : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) →
  m < -1 / 2 :=
by
  sorry

end range_of_m_l1099_109904


namespace books_shelves_l1099_109962

def initial_books : ℝ := 40.0
def additional_books : ℝ := 20.0
def books_per_shelf : ℝ := 4.0

theorem books_shelves :
  (initial_books + additional_books) / books_per_shelf = 15 :=
by 
  sorry

end books_shelves_l1099_109962


namespace parabola_inequality_l1099_109942

theorem parabola_inequality (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * (-1 - 1)^2 + c)
  (h3 : y2 = a * (4 - 1)^2 + c) :
  y1 > y2 :=
sorry

end parabola_inequality_l1099_109942


namespace minimize_F_l1099_109901

theorem minimize_F : ∃ x1 x2 x3 x4 x5 : ℝ, 
  (-2 * x1 + x2 + x3 = 2) ∧ 
  (x1 - 2 * x2 + x4 = 2) ∧ 
  (x1 + x2 + x5 = 5) ∧ 
  (x1 ≥ 0) ∧ 
  (x2 ≥ 0) ∧ 
  (x2 - x1 = -3) :=
by {
  sorry
}

end minimize_F_l1099_109901


namespace sum_of_four_circles_l1099_109983

open Real

theorem sum_of_four_circles:
  ∀ (s c : ℝ), 
  (2 * s + 3 * c = 26) → 
  (3 * s + 2 * c = 23) → 
  (4 * c = 128 / 5) :=
by
  intros s c h1 h2
  sorry

end sum_of_four_circles_l1099_109983


namespace circle_area_from_circumference_l1099_109944

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end circle_area_from_circumference_l1099_109944


namespace cylindrical_to_cartesian_l1099_109986

theorem cylindrical_to_cartesian :
  ∀ (r θ z : ℝ), r = 2 → θ = π / 3 → z = 2 → 
  (r * Real.cos θ, r * Real.sin θ, z) = (1, Real.sqrt 3, 2) :=
by
  intros r θ z hr hθ hz
  sorry

end cylindrical_to_cartesian_l1099_109986


namespace converse_negation_contrapositive_l1099_109988

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 3 * x + 2 ≠ 0
def Q (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2

theorem converse (h : Q x) : P x := by
  sorry

theorem negation (h : ¬ P x) : ¬ Q x := by
  sorry

theorem contrapositive (h : ¬ Q x) : ¬ P x := by
  sorry

end converse_negation_contrapositive_l1099_109988


namespace tina_mother_took_out_coins_l1099_109946

theorem tina_mother_took_out_coins :
  let first_hour := 20
  let next_two_hours := 30 * 2
  let fourth_hour := 40
  let total_coins := first_hour + next_two_hours + fourth_hour
  let coins_left_after_fifth_hour := 100
  let coins_taken_out := total_coins - coins_left_after_fifth_hour
  coins_taken_out = 20 :=
by
  sorry

end tina_mother_took_out_coins_l1099_109946


namespace sum_of_coefficients_of_y_terms_l1099_109923

theorem sum_of_coefficients_of_y_terms: 
  let p := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 3)
  ∃ (a b c: ℝ), p = (10 * x^2 + a * x * y + 19 * x + b * y^2 + c * y + 6) ∧ a + b + c = 65 :=
by
  sorry

end sum_of_coefficients_of_y_terms_l1099_109923
