import Mathlib

namespace NUMINAMATH_GPT_travel_time_reduction_l2266_226606

theorem travel_time_reduction
  (original_speed : ℝ)
  (new_speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (new_time : ℝ)
  (h1 : original_speed = 80)
  (h2 : new_speed = 50)
  (h3 : time = 3)
  (h4 : distance = original_speed * time)
  (h5 : new_time = distance / new_speed) :
  new_time = 4.8 := 
sorry

end NUMINAMATH_GPT_travel_time_reduction_l2266_226606


namespace NUMINAMATH_GPT_group_population_l2266_226673

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end NUMINAMATH_GPT_group_population_l2266_226673


namespace NUMINAMATH_GPT_equivalent_single_discount_l2266_226612

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.30
noncomputable def second_discount : ℝ := 0.15
noncomputable def third_discount : ℝ := 0.10

theorem equivalent_single_discount :
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  (1 - (final_price / original_price)) = 0.4645 :=
by
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l2266_226612


namespace NUMINAMATH_GPT_solve_system_part1_solve_system_part3_l2266_226604

noncomputable def solution_part1 : Prop :=
  ∃ (x y : ℝ), (x + y = 2) ∧ (5 * x - 2 * (x + y) = 6) ∧ (x = 2) ∧ (y = 0)

-- Part (1) Statement
theorem solve_system_part1 : solution_part1 := sorry

noncomputable def solution_part3 : Prop :=
  ∃ (a b c : ℝ), (a + b = 3) ∧ (5 * a + 3 * c = 1) ∧ (a + b + c = 0) ∧ (a = 2) ∧ (b = 1) ∧ (c = -3)

-- Part (3) Statement
theorem solve_system_part3 : solution_part3 := sorry

end NUMINAMATH_GPT_solve_system_part1_solve_system_part3_l2266_226604


namespace NUMINAMATH_GPT_time_difference_180_div_vc_l2266_226623

open Real

theorem time_difference_180_div_vc
  (V_A V_B V_C : ℝ)
  (h_ratio : V_A / V_C = 5 ∧ V_B / V_C = 4)
  (start_A start_B start_C : ℝ)
  (h_start_A : start_A = 100)
  (h_start_B : start_B = 80)
  (h_start_C : start_C = 0)
  (race_distance : ℝ)
  (h_race_distance : race_distance = 1200) :
  (race_distance - start_A) / V_A - race_distance / V_C = 180 / V_C := 
sorry

end NUMINAMATH_GPT_time_difference_180_div_vc_l2266_226623


namespace NUMINAMATH_GPT_unique_solution_nat_triplet_l2266_226601

theorem unique_solution_nat_triplet (x y l : ℕ) (h : x^3 + y^3 - 53 = 7^l) : (x, y, l) = (3, 3, 0) :=
sorry

end NUMINAMATH_GPT_unique_solution_nat_triplet_l2266_226601


namespace NUMINAMATH_GPT_profit_percentage_correct_l2266_226681

noncomputable def overall_profit_percentage : ℚ :=
  let cost_radio := 225
  let overhead_radio := 15
  let price_radio := 300
  let cost_watch := 425
  let overhead_watch := 20
  let price_watch := 525
  let cost_mobile := 650
  let overhead_mobile := 30
  let price_mobile := 800
  
  let total_cost_price := (cost_radio + overhead_radio) + (cost_watch + overhead_watch) + (cost_mobile + overhead_mobile)
  let total_selling_price := price_radio + price_watch + price_mobile
  let total_profit := total_selling_price - total_cost_price
  (total_profit * 100 : ℚ) / total_cost_price
  
theorem profit_percentage_correct :
  overall_profit_percentage = 19.05 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_correct_l2266_226681


namespace NUMINAMATH_GPT_carwash_num_cars_l2266_226617

variable (C : ℕ)

theorem carwash_num_cars 
    (h1 : 5 * 7 + 5 * 6 + C * 5 = 100)
    : C = 7 := 
by
    sorry

end NUMINAMATH_GPT_carwash_num_cars_l2266_226617


namespace NUMINAMATH_GPT_mod_37_5_l2266_226653

theorem mod_37_5 : 37 % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_mod_37_5_l2266_226653


namespace NUMINAMATH_GPT_reusable_bag_trips_correct_lowest_carbon_solution_l2266_226638

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

end NUMINAMATH_GPT_reusable_bag_trips_correct_lowest_carbon_solution_l2266_226638


namespace NUMINAMATH_GPT_wicket_keeper_age_l2266_226613

/-- The cricket team consists of 11 members with an average age of 22 years.
    One member is 25 years old, and the wicket keeper is W years old.
    Excluding the 25-year-old and the wicket keeper, the average age of the remaining players is 21 years.
    Prove that the wicket keeper is 6 years older than the average age of the team. -/
theorem wicket_keeper_age (W : ℕ) (team_avg_age : ℕ := 22) (total_team_members : ℕ := 11) 
                          (other_member_age : ℕ := 25) (remaining_avg_age : ℕ := 21) :
    W = 28 → W - team_avg_age = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_wicket_keeper_age_l2266_226613


namespace NUMINAMATH_GPT_number_of_speedster_convertibles_l2266_226671

def proof_problem (T : ℕ) :=
  let Speedsters := 2 * T / 3
  let NonSpeedsters := 50
  let TotalInventory := NonSpeedsters * 3
  let SpeedsterConvertibles := 4 * Speedsters / 5
  (Speedsters = 2 * TotalInventory / 3) ∧ (SpeedsterConvertibles = 4 * Speedsters / 5)

theorem number_of_speedster_convertibles : proof_problem 150 → ∃ (x : ℕ), x = 80 :=
by
  -- Provide the definition of Speedsters, NonSpeedsters, TotalInventory, and SpeedsterConvertibles
  sorry

end NUMINAMATH_GPT_number_of_speedster_convertibles_l2266_226671


namespace NUMINAMATH_GPT_sarah_correct_answer_percentage_l2266_226650

theorem sarah_correct_answer_percentage
  (q1 q2 q3 : ℕ)   -- Number of questions in the first, second, and third tests.
  (p1 p2 p3 : ℕ → ℝ)   -- Percentages of questions Sarah got right in the first, second, and third tests.
  (m : ℕ)   -- Number of calculation mistakes:
  (h_q1 : q1 = 30) (h_q2 : q2 = 20) (h_q3 : q3 = 50)
  (h_p1 : p1 q1 = 0.85) (h_p2 : p2 q2 = 0.75) (h_p3 : p3 q3 = 0.90)
  (h_m : m = 3) :
  ∃ pct_correct : ℝ, pct_correct = 83 :=
by
  sorry

end NUMINAMATH_GPT_sarah_correct_answer_percentage_l2266_226650


namespace NUMINAMATH_GPT_people_distribution_l2266_226699

theorem people_distribution
  (total_mentions : ℕ)
  (mentions_house : ℕ)
  (mentions_fountain : ℕ)
  (mentions_bench : ℕ)
  (mentions_tree : ℕ)
  (each_person_mentions : ℕ)
  (total_people : ℕ)
  (facing_house : ℕ)
  (facing_fountain : ℕ)
  (facing_bench : ℕ)
  (facing_tree : ℕ)
  (h_total_mentions : total_mentions = 27)
  (h_mentions_house : mentions_house = 5)
  (h_mentions_fountain : mentions_fountain = 6)
  (h_mentions_bench : mentions_bench = 7)
  (h_mentions_tree : mentions_tree = 9)
  (h_each_person_mentions : each_person_mentions = 3)
  (h_total_people : total_people = 9)
  (h_facing_house : facing_house = 5)
  (h_facing_fountain : facing_fountain = 4)
  (h_facing_bench : facing_bench = 2)
  (h_facing_tree : facing_tree = 9) :
  total_mentions / each_person_mentions = total_people ∧ 
  facing_house = mentions_house ∧
  facing_fountain = total_people - mentions_house ∧
  facing_bench = total_people - mentions_bench ∧
  facing_tree = total_people - mentions_tree :=
by
  sorry

end NUMINAMATH_GPT_people_distribution_l2266_226699


namespace NUMINAMATH_GPT_parallel_vectors_l2266_226663

noncomputable def vector_a : (ℤ × ℤ) := (1, 3)
noncomputable def vector_b (m : ℤ) : (ℤ × ℤ) := (-2, m)

theorem parallel_vectors (m : ℤ) (h : vector_a = (1, 3) ∧ vector_b m = (-2, m))
  (hp: ∃ k : ℤ, ∀ (a1 a2 b1 b2 : ℤ), (a1, a2) = vector_a ∧ (b1, b2) = (1 + k * (-2), 3 + k * m)):
  m = -6 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l2266_226663


namespace NUMINAMATH_GPT_parabola_distance_l2266_226605

theorem parabola_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (h_distance_focus : (P.1 - 1)^2 + P.2^2 = 9) : 
  Real.sqrt (P.1^2 + P.2^2) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_distance_l2266_226605


namespace NUMINAMATH_GPT_volume_of_rectangular_solid_l2266_226642

theorem volume_of_rectangular_solid : 
  let l := 100 -- length in cm
  let w := 20  -- width in cm
  let h := 50  -- height in cm
  let V := l * w * h
  V = 100000 :=
by
  rfl

end NUMINAMATH_GPT_volume_of_rectangular_solid_l2266_226642


namespace NUMINAMATH_GPT_find_d_values_l2266_226696

open Set

theorem find_d_values :
  ∀ {f : ℝ → ℝ}, ContinuousOn f (Icc 0 1) → (f 0 = f 1) →
  ∃ (d : ℝ), d ∈ Ioo 0 1 ∧ (∀ x₀, x₀ ∈ Icc 0 (1 - d) → (f x₀ = f (x₀ + d))) ↔
  ∃ k : ℕ, d = 1 / k :=
by
  sorry

end NUMINAMATH_GPT_find_d_values_l2266_226696


namespace NUMINAMATH_GPT_joan_paid_230_l2266_226651

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 := 
by 
  sorry

end NUMINAMATH_GPT_joan_paid_230_l2266_226651


namespace NUMINAMATH_GPT_dividend_percentage_paid_by_company_l2266_226690

-- Define the parameters
def faceValue : ℝ := 50
def investmentReturnPercentage : ℝ := 25
def investmentPerShare : ℝ := 37

-- Define the theorem
theorem dividend_percentage_paid_by_company :
  (investmentReturnPercentage / 100 * investmentPerShare / faceValue * 100) = 18.5 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_dividend_percentage_paid_by_company_l2266_226690


namespace NUMINAMATH_GPT_expected_groups_l2266_226667

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end NUMINAMATH_GPT_expected_groups_l2266_226667


namespace NUMINAMATH_GPT_henry_apple_weeks_l2266_226689

theorem henry_apple_weeks (apples_per_box : ℕ) (boxes : ℕ) (people : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) :
  apples_per_box = 14 → boxes = 3 → people = 2 → apples_per_day = 1 → days_per_week = 7 →
  (apples_per_box * boxes) / (people * apples_per_day * days_per_week) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_henry_apple_weeks_l2266_226689


namespace NUMINAMATH_GPT_trigonometric_expression_eval_l2266_226602

-- Conditions
variable (α : Real) (h1 : ∃ x : Real, 3 * x^2 - x - 2 = 0 ∧ x = Real.cos α) (h2 : α > π ∧ α < 3 * π / 2)

-- Question and expected answer
theorem trigonometric_expression_eval :
  (Real.sin (-α + 3 * π / 2) * Real.cos (3 * π / 2 + α) * Real.tan (π - α)^2) /
  (Real.cos (π / 2 + α) * Real.sin (π / 2 - α)) = 5 / 4 := sorry

end NUMINAMATH_GPT_trigonometric_expression_eval_l2266_226602


namespace NUMINAMATH_GPT_complement_union_A_B_in_U_l2266_226694

open Set Nat

def U : Set ℕ := { x | x < 6 ∧ x > 0 }
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_A_B_in_U : (U \ (A ∪ B)) = {2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_in_U_l2266_226694


namespace NUMINAMATH_GPT_length_of_box_l2266_226652

theorem length_of_box (rate : ℕ) (width : ℕ) (depth : ℕ) (time : ℕ) (volume : ℕ) (length : ℕ) :
  rate = 4 →
  width = 6 →
  depth = 2 →
  time = 21 →
  volume = rate * time →
  length = volume / (width * depth) →
  length = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_length_of_box_l2266_226652


namespace NUMINAMATH_GPT_final_expression_simplified_l2266_226686

variable (b : ℝ)

theorem final_expression_simplified :
  ((3 * b + 6 - 5 * b) / 3) = (-2 / 3) * b + 2 := by
  sorry

end NUMINAMATH_GPT_final_expression_simplified_l2266_226686


namespace NUMINAMATH_GPT_surface_area_l2266_226678

theorem surface_area (r : ℝ) (π : ℝ) (V : ℝ) (S : ℝ) 
  (h1 : V = 48 * π) 
  (h2 : V = (4 / 3) * π * r^3) : 
  S = 4 * π * r^2 :=
  sorry

end NUMINAMATH_GPT_surface_area_l2266_226678


namespace NUMINAMATH_GPT_cross_section_area_correct_l2266_226634

noncomputable def cross_section_area (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 33) / 8

theorem cross_section_area_correct
  (AB CC1 : ℝ)
  (h1 : AB = a)
  (h2 : CC1 = 2 * a) :
  cross_section_area a = (3 * a^2 * Real.sqrt 33) / 8 :=
by
  sorry

end NUMINAMATH_GPT_cross_section_area_correct_l2266_226634


namespace NUMINAMATH_GPT_tensor_A_B_eq_l2266_226609

-- Define sets A and B
def A : Set ℕ := {0, 2}
def B : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

-- Define set operation ⊗
def tensor (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

-- Prove that A ⊗ B = {0, 2, 4}
theorem tensor_A_B_eq : tensor A B = {0, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_tensor_A_B_eq_l2266_226609


namespace NUMINAMATH_GPT_savings_of_person_l2266_226610

-- Definitions as given in the problem
def income := 18000
def ratio_income_expenditure := 5 / 4

-- Implied definitions based on the conditions and problem context
noncomputable def expenditure := income * (4/5)
noncomputable def savings := income - expenditure

-- Theorem statement
theorem savings_of_person : savings = 3600 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_savings_of_person_l2266_226610


namespace NUMINAMATH_GPT_toothpicks_in_12th_stage_l2266_226692

def toothpicks_in_stage (n : ℕ) : ℕ :=
  3 * n

theorem toothpicks_in_12th_stage : toothpicks_in_stage 12 = 36 :=
by
  -- Proof steps would go here, including simplification and calculations, but are omitted with 'sorry'.
  sorry

end NUMINAMATH_GPT_toothpicks_in_12th_stage_l2266_226692


namespace NUMINAMATH_GPT_fixed_point_coordinates_l2266_226695

theorem fixed_point_coordinates (a b x y : ℝ) 
  (h1 : a + 2 * b = 1) 
  (h2 : (a * x + 3 * y + b) = 0) :
  x = 1 / 2 ∧ y = -1 / 6 := by
  sorry

end NUMINAMATH_GPT_fixed_point_coordinates_l2266_226695


namespace NUMINAMATH_GPT_eval_expression_l2266_226675

theorem eval_expression : (503 * 503 - 502 * 504) = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l2266_226675


namespace NUMINAMATH_GPT_problem_l2266_226679

noncomputable def h (p x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15

noncomputable def k (q r x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

theorem problem
  (p q r : ℝ)
  (h_has_distinct_roots: ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ h p a = 0 ∧ h p b = 0 ∧ h p c = 0)
  (h_roots_are_k_roots: ∀ x, h p x = 0 → k q r x = 0) :
  k q r 1 = -3322.25 :=
sorry

end NUMINAMATH_GPT_problem_l2266_226679


namespace NUMINAMATH_GPT_rectangle_area_diagonal_ratio_l2266_226600

theorem rectangle_area_diagonal_ratio (d : ℝ) (x : ℝ) (h_ratio : 5 * x ≥ 0 ∧ 2 * x ≥ 0)
  (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_diagonal_ratio_l2266_226600


namespace NUMINAMATH_GPT_worker_original_daily_wage_l2266_226647

-- Given Conditions
def increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]
def new_total_weekly_salary : ℝ := 1457

-- Define the sum of the weekly increases
def total_increase : ℝ := (1 + increases.get! 0) + (1 + increases.get! 1) + (1 + increases.get! 2) + (1 + increases.get! 3) + (1 + increases.get! 4)

-- Main Theorem
theorem worker_original_daily_wage : ∀ (W : ℝ), total_increase * W = new_total_weekly_salary → W = 242.83 :=
by
  intro W h
  sorry

end NUMINAMATH_GPT_worker_original_daily_wage_l2266_226647


namespace NUMINAMATH_GPT_unique_suwy_product_l2266_226629

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then Char.toNat c - Char.toNat 'A' + 1 else 0

def product_of_chars (l : List Char) : Nat :=
  l.foldr (λ c acc => letter_value c * acc) 1

theorem unique_suwy_product :
  ∀ (l : List Char), l.length = 4 → product_of_chars l = 19 * 21 * 23 * 25 → l = ['S', 'U', 'W', 'Y'] := 
by
  intro l hlen hproduct
  sorry

end NUMINAMATH_GPT_unique_suwy_product_l2266_226629


namespace NUMINAMATH_GPT_investment_doubles_in_9_years_l2266_226649

noncomputable def years_to_double (initial_amount : ℕ) (interest_rate : ℕ) : ℕ :=
  72 / interest_rate

theorem investment_doubles_in_9_years :
  ∀ (initial_amount : ℕ) (interest_rate : ℕ) (investment_period_val : ℕ) (expected_value : ℕ),
  initial_amount = 8000 ∧ interest_rate = 8 ∧ investment_period_val = 18 ∧ expected_value = 32000 →
  years_to_double initial_amount interest_rate = 9 :=
by
  intros initial_amount interest_rate investment_period_val expected_value h
  sorry

end NUMINAMATH_GPT_investment_doubles_in_9_years_l2266_226649


namespace NUMINAMATH_GPT_cylinder_height_l2266_226687

theorem cylinder_height
  (r : ℝ) (SA : ℝ) (h : ℝ)
  (h_radius : r = 3)
  (h_surface_area_given : SA = 30 * Real.pi)
  (h_surface_area_formula : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  h = 2 :=
by
  -- Proof can be written here
  sorry

end NUMINAMATH_GPT_cylinder_height_l2266_226687


namespace NUMINAMATH_GPT_equal_chords_divide_equally_l2266_226662

theorem equal_chords_divide_equally 
  {A B C D M : ℝ} 
  (in_circle : ∃ (O : ℝ), (dist O A = dist O B) ∧ (dist O C = dist O D) ∧ (dist O M < dist O A))
  (chords_equal : dist A B = dist C D)
  (intersection_M : dist A M + dist M B = dist C M + dist M D ∧ dist A M = dist C M ∧ dist B M = dist D M) :
  dist A M = dist M B ∧ dist C M = dist M D := 
sorry

end NUMINAMATH_GPT_equal_chords_divide_equally_l2266_226662


namespace NUMINAMATH_GPT_find_numbers_l2266_226665

theorem find_numbers : ∃ x y : ℕ, x + y = 2016 ∧ (∃ d : ℕ, d < 10 ∧ (x = 10 * y + d) ∧ x = 1833 ∧ y = 183) :=
by 
  sorry

end NUMINAMATH_GPT_find_numbers_l2266_226665


namespace NUMINAMATH_GPT_absolute_value_inequality_l2266_226635

variable (a b c d : ℝ)

theorem absolute_value_inequality (h₁ : a + b + c + d > 0) (h₂ : a > c) (h₃ : b > d) : 
  |a + b| > |c + d| := sorry

end NUMINAMATH_GPT_absolute_value_inequality_l2266_226635


namespace NUMINAMATH_GPT_max_even_integers_with_odd_product_l2266_226641

theorem max_even_integers_with_odd_product (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) (h_odd_product : (a * b * c * d * e * f) % 2 = 1) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) := 
sorry

end NUMINAMATH_GPT_max_even_integers_with_odd_product_l2266_226641


namespace NUMINAMATH_GPT_odd_function_value_sum_l2266_226661

theorem odd_function_value_sum
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fneg1 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end NUMINAMATH_GPT_odd_function_value_sum_l2266_226661


namespace NUMINAMATH_GPT_probability_donation_to_A_l2266_226645

-- Define population proportions
def prob_O : ℝ := 0.50
def prob_A : ℝ := 0.15
def prob_B : ℝ := 0.30
def prob_AB : ℝ := 0.05

-- Define blood type compatibility predicate
def can_donate_to_A (blood_type : ℝ) : Prop := 
  blood_type = prob_O ∨ blood_type = prob_A

-- Theorem statement
theorem probability_donation_to_A : 
  prob_O + prob_A = 0.65 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_probability_donation_to_A_l2266_226645


namespace NUMINAMATH_GPT_parallel_vectors_imply_x_value_l2266_226620

theorem parallel_vectors_imply_x_value (x : ℝ) : 
    let a := (1, 2)
    let b := (-1, x)
    (1 / -1:ℝ) = (2 / x) → x = -2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_vectors_imply_x_value_l2266_226620


namespace NUMINAMATH_GPT_annie_initial_money_l2266_226668

def cost_of_hamburgers (n : Nat) : Nat := n * 4
def cost_of_milkshakes (m : Nat) : Nat := m * 5
def total_cost (n m : Nat) : Nat := cost_of_hamburgers n + cost_of_milkshakes m
def initial_money (n m left : Nat) : Nat := total_cost n m + left

theorem annie_initial_money : initial_money 8 6 70 = 132 := by
  sorry

end NUMINAMATH_GPT_annie_initial_money_l2266_226668


namespace NUMINAMATH_GPT_mr_green_potato_yield_l2266_226683

theorem mr_green_potato_yield :
  let steps_to_feet := 2.5
  let length_steps := 18
  let width_steps := 25
  let yield_per_sqft := 0.75
  let length_feet := length_steps * steps_to_feet
  let width_feet := width_steps * steps_to_feet
  let area_sqft := length_feet * width_feet
  let expected_yield := area_sqft * yield_per_sqft
  expected_yield = 2109.375 := by sorry

end NUMINAMATH_GPT_mr_green_potato_yield_l2266_226683


namespace NUMINAMATH_GPT_PropA_neither_sufficient_nor_necessary_for_PropB_l2266_226614

variable (a b : ℤ)

-- Proposition A
def PropA : Prop := a + b ≠ 4

-- Proposition B
def PropB : Prop := a ≠ 1 ∧ b ≠ 3

-- The required statement
theorem PropA_neither_sufficient_nor_necessary_for_PropB : ¬(PropA a b → PropB a b) ∧ ¬(PropB a b → PropA a b) :=
by
  sorry

end NUMINAMATH_GPT_PropA_neither_sufficient_nor_necessary_for_PropB_l2266_226614


namespace NUMINAMATH_GPT_rope_segments_after_folding_l2266_226622

theorem rope_segments_after_folding (n : ℕ) (h : n = 6) : 2^n + 1 = 65 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_rope_segments_after_folding_l2266_226622


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2266_226624

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2266_226624


namespace NUMINAMATH_GPT_Chris_had_before_birthday_l2266_226632

-- Define the given amounts
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Define the total birthday money received
def birthday_money : ℕ := grandmother_money + aunt_uncle_money + parents_money

-- Define the amount of money Chris had before his birthday
def money_before_birthday (total_now birthday_money : ℕ) : ℕ := total_now - birthday_money

-- Proposition to prove
theorem Chris_had_before_birthday : money_before_birthday total_money_now birthday_money = 159 := by
  sorry

end NUMINAMATH_GPT_Chris_had_before_birthday_l2266_226632


namespace NUMINAMATH_GPT_solve_system_of_equations_l2266_226656

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 7) (h2 : 2 * x - y = 2) :
  x = 3 ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2266_226656


namespace NUMINAMATH_GPT_average_number_of_fish_is_75_l2266_226648

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end NUMINAMATH_GPT_average_number_of_fish_is_75_l2266_226648


namespace NUMINAMATH_GPT_abs_sum_fraction_le_sum_abs_fraction_l2266_226616

variable (a b : ℝ)

theorem abs_sum_fraction_le_sum_abs_fraction (a b : ℝ) :
  (|a + b| / (1 + |a + b|)) ≤ (|a| / (1 + |a|)) + (|b| / (1 + |b|)) :=
sorry

end NUMINAMATH_GPT_abs_sum_fraction_le_sum_abs_fraction_l2266_226616


namespace NUMINAMATH_GPT_parallel_condition_l2266_226618

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4 * x)

-- Define the condition for parallelism for two-dimensional vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Define the theorem to prove
theorem parallel_condition (x : ℝ) :
  parallel (vector_a x) (vector_b x) ↔ |x| = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_condition_l2266_226618


namespace NUMINAMATH_GPT_journey_time_l2266_226672

noncomputable def journey_time_proof : Prop :=
  ∃ t1 t2 t3 : ℝ,
    25 * t1 - 25 * t2 + 25 * t3 = 100 ∧
    5 * t1 + 5 * t2 + 25 * t3 = 100 ∧
    25 * t1 + 5 * t2 + 5 * t3 = 100 ∧
    t1 + t2 + t3 = 8

theorem journey_time : journey_time_proof := by sorry

end NUMINAMATH_GPT_journey_time_l2266_226672


namespace NUMINAMATH_GPT_ayse_guarantee_win_l2266_226659

def can_ayse_win (m n k : ℕ) : Prop :=
  -- Function defining the winning strategy for Ayşe
  sorry -- The exact strategy definition would be here

theorem ayse_guarantee_win :
  ((can_ayse_win 1 2012 2014) ∧ 
   (can_ayse_win 2011 2011 2012) ∧ 
   (can_ayse_win 2011 2012 2013) ∧ 
   (can_ayse_win 2011 2012 2014) ∧ 
   (can_ayse_win 2011 2013 2013)) = true :=
sorry -- Proof goes here

end NUMINAMATH_GPT_ayse_guarantee_win_l2266_226659


namespace NUMINAMATH_GPT_income_is_10000_l2266_226676

-- Define the necessary variables: income, expenditure, and savings
variables (income expenditure : ℕ) (x : ℕ)

-- Define the conditions given in the problem
def ratio_condition : Prop := income = 10 * x ∧ expenditure = 7 * x
def savings_condition : Prop := income - expenditure = 3000

-- State the theorem that needs to be proved
theorem income_is_10000 (h_ratio : ratio_condition income expenditure x) (h_savings : savings_condition income expenditure) : income = 10000 :=
sorry

end NUMINAMATH_GPT_income_is_10000_l2266_226676


namespace NUMINAMATH_GPT_quadratic_form_sum_const_l2266_226677

theorem quadratic_form_sum_const (a b c x : ℝ) (h : 4 * x^2 - 28 * x - 48 = a * (x + b)^2 + c) : 
  a + b + c = -96.5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_form_sum_const_l2266_226677


namespace NUMINAMATH_GPT_kickers_goals_in_first_period_l2266_226666

theorem kickers_goals_in_first_period (K : ℕ) 
  (h1 : ∀ n : ℕ, n = K) 
  (h2 : ∀ n : ℕ, n = 2 * K) 
  (h3 : ∀ n : ℕ, n = K / 2) 
  (h4 : ∀ n : ℕ, n = 4 * K) 
  (h5 : K + 2 * K + (K / 2) + 4 * K = 15) : 
  K = 2 := 
by
  sorry

end NUMINAMATH_GPT_kickers_goals_in_first_period_l2266_226666


namespace NUMINAMATH_GPT_P_positive_l2266_226639

variable (P : ℕ → ℝ)

axiom P_cond_0 : P 0 > 0
axiom P_cond_1 : P 1 > P 0
axiom P_cond_2 : P 2 > 2 * P 1 - P 0
axiom P_cond_3 : P 3 > 3 * P 2 - 3 * P 1 + P 0
axiom P_cond_n : ∀ n, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n

theorem P_positive (n : ℕ) (h : n > 0) : P n > 0 := by
  sorry

end NUMINAMATH_GPT_P_positive_l2266_226639


namespace NUMINAMATH_GPT_water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l2266_226670

-- Define the tiered water pricing function
def tiered_water_cost (m : ℕ) : ℝ :=
  if m ≤ 20 then
    1.6 * m
  else if m ≤ 30 then
    1.6 * 20 + 2.4 * (m - 20)
  else
    1.6 * 20 + 2.4 * 10 + 4.8 * (m - 30)

-- Problem 1
theorem water_cost_10_tons : tiered_water_cost 10 = 16 := 
sorry

-- Problem 2
theorem water_cost_27_tons : tiered_water_cost 27 = 48.8 := 
sorry

-- Problem 3
theorem water_cost_between_20_30 (m : ℕ) (h : 20 < m ∧ m < 30) : tiered_water_cost m = 2.4 * m - 16 := 
sorry

-- Problem 4
theorem water_cost_above_30 (m : ℕ) (h : m > 30) : tiered_water_cost m = 4.8 * m - 88 := 
sorry

end NUMINAMATH_GPT_water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l2266_226670


namespace NUMINAMATH_GPT_find_unique_pair_l2266_226697

theorem find_unique_pair (x y : ℝ) :
  (∀ (u v : ℝ), (u * x + v * y = u) ∧ (u * y + v * x = v)) ↔ (x = 1 ∧ y = 0) :=
by
  -- This is to ignore the proof part
  sorry

end NUMINAMATH_GPT_find_unique_pair_l2266_226697


namespace NUMINAMATH_GPT_true_discount_double_time_l2266_226680

theorem true_discount_double_time (PV FV1 FV2 I1 I2 TD1 TD2 : ℕ) 
  (h1 : FV1 = 110)
  (h2 : TD1 = 10)
  (h3 : FV1 - TD1 = PV)
  (h4 : I1 = FV1 - PV)
  (h5 : FV2 = PV + 2 * I1)
  (h6 : TD2 = FV2 - PV) :
  TD2 = 20 := by
  sorry

end NUMINAMATH_GPT_true_discount_double_time_l2266_226680


namespace NUMINAMATH_GPT_least_number_divisible_by_11_l2266_226657

theorem least_number_divisible_by_11 (n : ℕ) (k : ℕ) (h₁ : n = 2520 * k + 1) (h₂ : 11 ∣ n) : n = 12601 :=
sorry

end NUMINAMATH_GPT_least_number_divisible_by_11_l2266_226657


namespace NUMINAMATH_GPT_remainder_101_pow_37_mod_100_l2266_226669

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_101_pow_37_mod_100_l2266_226669


namespace NUMINAMATH_GPT_find_y_l2266_226646

-- Define the points and slope conditions
def point_R : ℝ × ℝ := (-3, 4)
def x2 : ℝ := 5

-- Define the y coordinate and its corresponding condition
def y_condition (y : ℝ) : Prop := (y - 4) / (5 - (-3)) = 1 / 2

-- The main theorem stating the conditions and conclusion
theorem find_y (y : ℝ) (h : y_condition y) : y = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2266_226646


namespace NUMINAMATH_GPT_number_of_bags_needed_l2266_226637

def cost_corn_seeds : ℕ := 50
def cost_fertilizers_pesticides : ℕ := 35
def cost_labor : ℕ := 15
def profit_percentage : ℝ := 0.10
def price_per_bag : ℝ := 11

theorem number_of_bags_needed (total_cost : ℕ) (total_revenue : ℝ) (num_bags : ℝ) :
  total_cost = cost_corn_seeds + cost_fertilizers_pesticides + cost_labor →
  total_revenue = ↑total_cost + (↑total_cost * profit_percentage) →
  num_bags = total_revenue / price_per_bag →
  num_bags = 10 := 
by
  sorry

end NUMINAMATH_GPT_number_of_bags_needed_l2266_226637


namespace NUMINAMATH_GPT_bench_allocation_l2266_226693

theorem bench_allocation (M : ℕ) : (∃ M, M > 0 ∧ 5 * M = 13 * M) → M = 5 :=
by
  sorry

end NUMINAMATH_GPT_bench_allocation_l2266_226693


namespace NUMINAMATH_GPT_units_digit_n_l2266_226608

theorem units_digit_n (m n : ℕ) (hm : m % 10 = 9) (h : m * n = 18^5) : n % 10 = 2 :=
sorry

end NUMINAMATH_GPT_units_digit_n_l2266_226608


namespace NUMINAMATH_GPT_christine_commission_rate_l2266_226658

theorem christine_commission_rate (C : ℝ) (H1 : 24000 ≠ 0) (H2 : 0.4 * (C / 100 * 24000) = 1152) :
  C = 12 :=
by
  sorry

end NUMINAMATH_GPT_christine_commission_rate_l2266_226658


namespace NUMINAMATH_GPT_fraction_of_married_women_l2266_226691

theorem fraction_of_married_women (total_employees : ℕ) 
  (women_fraction : ℝ) (married_fraction : ℝ) (single_men_fraction : ℝ)
  (hwf : women_fraction = 0.64) (hmf : married_fraction = 0.60) 
  (hsf : single_men_fraction = 2/3) : 
  ∃ (married_women_fraction : ℝ), married_women_fraction = 3/4 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_married_women_l2266_226691


namespace NUMINAMATH_GPT_child_ticket_price_correct_l2266_226655

-- Definitions based on conditions
def total_collected := 104
def price_adult := 6
def total_tickets := 21
def children_tickets := 11

-- Derived conditions
def adult_tickets := total_tickets - children_tickets
def total_revenue_child (C : ℕ) := children_tickets * C
def total_revenue_adult := adult_tickets * price_adult

-- Main statement to prove
theorem child_ticket_price_correct (C : ℕ) 
  (h1 : total_revenue_child C + total_revenue_adult = total_collected) : 
  C = 4 :=
by
  sorry

end NUMINAMATH_GPT_child_ticket_price_correct_l2266_226655


namespace NUMINAMATH_GPT_taxi_fare_distance_l2266_226685

variable (x : ℝ)

theorem taxi_fare_distance (h1 : 0 ≤ x - 2) (h2 : 3 + 1.2 * (x - 2) = 9) : x = 7 := by
  sorry

end NUMINAMATH_GPT_taxi_fare_distance_l2266_226685


namespace NUMINAMATH_GPT_cos_squared_diff_tan_l2266_226631

theorem cos_squared_diff_tan (α : ℝ) (h : Real.tan α = 3) :
  Real.cos (α + π/4) ^ 2 - Real.cos (α - π/4) ^ 2 = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_squared_diff_tan_l2266_226631


namespace NUMINAMATH_GPT_flowers_on_porch_l2266_226684

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end NUMINAMATH_GPT_flowers_on_porch_l2266_226684


namespace NUMINAMATH_GPT_age_of_participant_who_left_l2266_226660

theorem age_of_participant_who_left
  (avg_age_first_room : ℕ)
  (num_people_first_room : ℕ)
  (avg_age_second_room : ℕ)
  (num_people_second_room : ℕ)
  (increase_in_avg_age : ℕ)
  (total_num_people : ℕ)
  (final_avg_age : ℕ)
  (initial_avg_age : ℕ)
  (sum_ages : ℕ)
  (person_left : ℕ) :
  avg_age_first_room = 20 ∧ 
  num_people_first_room = 8 ∧
  avg_age_second_room = 45 ∧
  num_people_second_room = 12 ∧
  increase_in_avg_age = 1 ∧
  total_num_people = num_people_first_room + num_people_second_room ∧
  final_avg_age = initial_avg_age + increase_in_avg_age ∧
  initial_avg_age = (sum_ages) / total_num_people ∧
  sum_ages = (avg_age_first_room * num_people_first_room + avg_age_second_room * num_people_second_room) ∧
  19 * final_avg_age = sum_ages - person_left
  → person_left = 16 :=
by sorry

end NUMINAMATH_GPT_age_of_participant_who_left_l2266_226660


namespace NUMINAMATH_GPT_original_price_of_trouser_l2266_226643

theorem original_price_of_trouser (sale_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 50) (h2 : discount_rate = 0.50) (h3 : sale_price = (1 - discount_rate) * original_price) : 
  original_price = 100 :=
sorry

end NUMINAMATH_GPT_original_price_of_trouser_l2266_226643


namespace NUMINAMATH_GPT_max_balls_in_cube_l2266_226621

theorem max_balls_in_cube 
  (radius : ℝ) (side_length : ℝ) 
  (ball_volume : ℝ := (4 / 3) * Real.pi * (radius^3)) 
  (cube_volume : ℝ := side_length^3) 
  (max_balls : ℝ := cube_volume / ball_volume) :
  radius = 3 ∧ side_length = 8 → Int.floor max_balls = 4 := 
by
  intro h
  rw [h.left, h.right]
  -- further proof would use numerical evaluation
  sorry

end NUMINAMATH_GPT_max_balls_in_cube_l2266_226621


namespace NUMINAMATH_GPT_car_distance_after_y_begins_l2266_226603

theorem car_distance_after_y_begins (v_x v_y : ℝ) (t_y_start t_x_after_y : ℝ) (d_x_before_y : ℝ) :
  v_x = 35 → v_y = 50 → t_y_start = 1.2 → d_x_before_y = v_x * t_y_start → t_x_after_y = 2.8 →
  (d_x_before_y + v_x * t_x_after_y = 98) :=
by
  intros h_vx h_vy h_ty_start h_dxbefore h_txafter
  simp [h_vx, h_vy, h_ty_start, h_dxbefore, h_txafter]
  sorry

end NUMINAMATH_GPT_car_distance_after_y_begins_l2266_226603


namespace NUMINAMATH_GPT_bob_and_jim_total_skips_l2266_226698

-- Definitions based on conditions
def bob_skips_per_rock : Nat := 12
def jim_skips_per_rock : Nat := 15
def rocks_skipped_by_each : Nat := 10

-- Total skips calculation based on the given conditions
def bob_total_skips : Nat := bob_skips_per_rock * rocks_skipped_by_each
def jim_total_skips : Nat := jim_skips_per_rock * rocks_skipped_by_each
def total_skips : Nat := bob_total_skips + jim_total_skips

-- Theorem statement
theorem bob_and_jim_total_skips : total_skips = 270 := by
  sorry

end NUMINAMATH_GPT_bob_and_jim_total_skips_l2266_226698


namespace NUMINAMATH_GPT_mimi_spent_on_clothes_l2266_226654

theorem mimi_spent_on_clothes :
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  total_spent - total_sneakers_cost = 2600 :=
by
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  show total_spent - total_sneakers_cost = 2600
  sorry

end NUMINAMATH_GPT_mimi_spent_on_clothes_l2266_226654


namespace NUMINAMATH_GPT_total_sample_variance_l2266_226682

/-- In a survey of the heights (in cm) of high school students at Shuren High School:

 - 20 boys were selected with an average height of 174 cm and a variance of 12.
 - 30 girls were selected with an average height of 164 cm and a variance of 30.

We need to prove that the variance of the total sample is 46.8. -/
theorem total_sample_variance :
  let boys_count := 20
  let girls_count := 30
  let boys_avg := 174
  let girls_avg := 164
  let boys_var := 12
  let girls_var := 30
  let total_count := boys_count + girls_count
  let overall_avg := (boys_avg * boys_count + girls_avg * girls_count) / total_count
  let total_var := 
    (boys_count * (boys_var + (boys_avg - overall_avg)^2) / total_count)
    + (girls_count * (girls_var + (girls_avg - overall_avg)^2) / total_count)
  total_var = 46.8 := by
    sorry

end NUMINAMATH_GPT_total_sample_variance_l2266_226682


namespace NUMINAMATH_GPT_inequality_proof_l2266_226636

variable (a b c : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)

theorem inequality_proof :
  (2 * a + b + c)^2 / (2 * a^2 + (b + c)^2) +
  (a + 2 * b + c)^2 / (2 * b^2 + (c + a)^2) +
  (a + b + 2 * c)^2 / (2 * c^2 + (a + b)^2) ≤ 8 := sorry

end NUMINAMATH_GPT_inequality_proof_l2266_226636


namespace NUMINAMATH_GPT_dice_multiple_3_prob_l2266_226674

-- Define the probability calculations for the problem
noncomputable def single_roll_multiple_3_prob: ℝ := 1 / 3
noncomputable def single_roll_not_multiple_3_prob: ℝ := 1 - single_roll_multiple_3_prob
noncomputable def eight_rolls_not_multiple_3_prob: ℝ := (single_roll_not_multiple_3_prob) ^ 8
noncomputable def at_least_one_roll_multiple_3_prob: ℝ := 1 - eight_rolls_not_multiple_3_prob

-- The lean theorem statement
theorem dice_multiple_3_prob : 
  at_least_one_roll_multiple_3_prob = 6305 / 6561 := by 
sorry

end NUMINAMATH_GPT_dice_multiple_3_prob_l2266_226674


namespace NUMINAMATH_GPT_last_week_profit_min_selling_price_red_beauty_l2266_226625

theorem last_week_profit (x kgs_of_red_beauty x_green : ℕ) 
  (purchase_cost_red_beauty_per_kg selling_cost_red_beauty_per_kg 
  purchase_cost_xiangshan_green_per_kg selling_cost_xiangshan_green_per_kg
  total_weight total_cost all_fruits_profit : ℕ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  selling_cost_red_beauty_per_kg = 35 ->
  purchase_cost_xiangshan_green_per_kg = 5 ->
  selling_cost_xiangshan_green_per_kg = 10 ->
  total_weight = 300 ->
  total_cost = 3000 ->
  x * purchase_cost_red_beauty_per_kg + (total_weight - x) * purchase_cost_xiangshan_green_per_kg = total_cost ->
  all_fruits_profit = x * (selling_cost_red_beauty_per_kg - purchase_cost_red_beauty_per_kg) +
  (total_weight - x) * (selling_cost_xiangshan_green_per_kg - purchase_cost_xiangshan_green_per_kg) -> 
  all_fruits_profit = 2500 := sorry

theorem min_selling_price_red_beauty (last_week_profit : ℕ) (x kgs_of_red_beauty x_green damaged_ratio : ℝ) 
  (purchase_cost_red_beauty_per_kg profit_last_week selling_cost_xiangshan_per_kg 
  total_weight total_cost : ℝ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  profit_last_week = 2500 ->
  damaged_ratio = 0.1 ->
  x = 100 ->
  (profit_last_week = 
    x * (35 - purchase_cost_red_beauty_per_kg) + (total_weight - x) * (10 - 5)) ->
  90 * (purchase_cost_red_beauty_per_kg + (last_week_profit - 15 * (total_weight - x) / 90)) ≥ 1500 ->
  profit_last_week / (90 * (90 * (purchase_cost_red_beauty_per_kg + (2500 - 15 * (300 - x) / 90)))) >=
  (36.7 - 20 / purchase_cost_red_beauty_per_kg) :=
  sorry

end NUMINAMATH_GPT_last_week_profit_min_selling_price_red_beauty_l2266_226625


namespace NUMINAMATH_GPT_solve_system_eq_l2266_226628

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_eq_l2266_226628


namespace NUMINAMATH_GPT_molecular_weight_is_44_02_l2266_226607

-- Definition of atomic weights and the number of atoms
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def count_N : ℕ := 2
def count_O : ℕ := 1

-- The compound's molecular weight calculation
def molecular_weight : ℝ := (count_N * atomic_weight_N) + (count_O * atomic_weight_O)

-- The proof statement that the molecular weight of the compound is approximately 44.02 amu
theorem molecular_weight_is_44_02 : molecular_weight = 44.02 := 
by
  sorry

#eval molecular_weight  -- Should output 44.02 (not part of the theorem, just for checking)

end NUMINAMATH_GPT_molecular_weight_is_44_02_l2266_226607


namespace NUMINAMATH_GPT_max_M_value_l2266_226688

noncomputable def M (x y z w : ℝ) : ℝ :=
  x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z

theorem max_M_value (x y z w : ℝ) (h : x + y + z + w = 1) :
  (M x y z w) ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_max_M_value_l2266_226688


namespace NUMINAMATH_GPT_caochong_weighing_equation_l2266_226615

-- Definitions for porter weight, stone weight, and the counts in the respective steps
def porter_weight : ℝ := 120
def stone_weight (x : ℝ) : ℝ := x
def first_step_weight (x : ℝ) : ℝ := 20 * stone_weight x + 3 * porter_weight
def second_step_weight (x : ℝ) : ℝ := (20 + 1) * stone_weight x + 1 * porter_weight

-- Theorem stating the equality condition ensuring the same water level
theorem caochong_weighing_equation (x : ℝ) :
  first_step_weight x = second_step_weight x :=
by
  sorry

end NUMINAMATH_GPT_caochong_weighing_equation_l2266_226615


namespace NUMINAMATH_GPT_geom_seq_min_value_l2266_226627

open Real

/-- 
Theorem: For a geometric sequence {a_n} where a_n > 0 and a_7 = √2/2, 
the minimum value of 1/a_3 + 2/a_11 is 4.
-/
theorem geom_seq_min_value (a : ℕ → ℝ) (a_pos : ∀ n, 0 < a n) (h7 : a 7 = (sqrt 2) / 2) :
  (1 / (a 3) + 2 / (a 11) >= 4) :=
sorry

end NUMINAMATH_GPT_geom_seq_min_value_l2266_226627


namespace NUMINAMATH_GPT_original_number_is_80_l2266_226640

variable (e : ℝ)

def increased_value := 1.125 * e
def decreased_value := 0.75 * e
def difference_condition := increased_value e - decreased_value e = 30

theorem original_number_is_80 (h : difference_condition e) : e = 80 :=
sorry

end NUMINAMATH_GPT_original_number_is_80_l2266_226640


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2266_226644

theorem geometric_sequence_sum (a1 r : ℝ) (S : ℕ → ℝ) :
  S 2 = 3 → S 4 = 15 →
  (∀ n, S n = a1 * (1 - r^n) / (1 - r)) → S 6 = 63 :=
by
  intros hS2 hS4 hSn
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2266_226644


namespace NUMINAMATH_GPT_exactly_one_correct_proposition_l2266_226633

variables (l1 l2 : Line) (alpha : Plane)

-- Definitions for the conditions
def perpendicular_lines (l1 l2 : Line) : Prop := -- definition of perpendicular lines
sorry

def perpendicular_to_plane (l : Line) (alpha : Plane) : Prop := -- definition of line perpendicular to plane
sorry

def line_in_plane (l : Line) (alpha : Plane) : Prop := -- definition of line in a plane
sorry

-- Problem statement
theorem exactly_one_correct_proposition 
  (h1 : perpendicular_lines l1 l2) 
  (h2 : perpendicular_to_plane l1 alpha) 
  (h3 : line_in_plane l2 alpha) : 
  (¬(perpendicular_lines l1 l2 ∧ perpendicular_to_plane l1 alpha → line_in_plane l2 alpha) ∧
   ¬(perpendicular_lines l1 l2 ∧ line_in_plane l2 alpha → perpendicular_to_plane l1 alpha) ∧
   (perpendicular_to_plane l1 alpha ∧ line_in_plane l2 alpha → perpendicular_lines l1 l2)) :=
sorry

end NUMINAMATH_GPT_exactly_one_correct_proposition_l2266_226633


namespace NUMINAMATH_GPT_extremum_is_not_unique_l2266_226619

-- Define the extremum conditionally in terms of unique extremum within an interval for a function
def isExtremum {α : Type*} [Preorder α] (f : α → ℝ) (x : α) :=
  ∀ y, f y ≤ f x ∨ f x ≤ f y

theorem extremum_is_not_unique (α : Type*) [Preorder α] (f : α → ℝ) :
  ¬ ∀ x, isExtremum f x → (∀ y, isExtremum f y → x = y) :=
by
  sorry

end NUMINAMATH_GPT_extremum_is_not_unique_l2266_226619


namespace NUMINAMATH_GPT_mrs_lee_earnings_percentage_l2266_226626

noncomputable def percentage_earnings_june (T : ℝ) : ℝ :=
  let L := 0.5 * T
  let L_June := 1.2 * L
  let total_income_june := T
  (L_June / total_income_june) * 100

theorem mrs_lee_earnings_percentage (T : ℝ) (hT : T ≠ 0) : percentage_earnings_june T = 60 :=
by
  sorry

end NUMINAMATH_GPT_mrs_lee_earnings_percentage_l2266_226626


namespace NUMINAMATH_GPT_total_books_on_shelves_l2266_226664

theorem total_books_on_shelves (shelves books_per_shelf : ℕ) (h_shelves : shelves = 350) (h_books_per_shelf : books_per_shelf = 25) :
  shelves * books_per_shelf = 8750 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_books_on_shelves_l2266_226664


namespace NUMINAMATH_GPT_xyz_square_sum_l2266_226611

theorem xyz_square_sum {x y z a b c d : ℝ} (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x + y + z = d) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  x^2 + y^2 + z^2 = d^2 - 2 * (a + b + c) :=
sorry

end NUMINAMATH_GPT_xyz_square_sum_l2266_226611


namespace NUMINAMATH_GPT_volume_of_wedge_l2266_226630

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_wedge_l2266_226630
