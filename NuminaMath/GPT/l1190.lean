import Mathlib

namespace joann_lollipop_wednesday_l1190_119069

variable (a : ℕ) (d : ℕ) (n : ℕ)

def joann_lollipop_count (a d n : ℕ) : ℕ :=
  a + d * n

theorem joann_lollipop_wednesday :
  let a := 4
  let d := 3
  let total_days := 7
  let target_total := 133
  ∀ (monday tuesday wednesday thursday friday saturday sunday : ℕ),
    monday = a ∧
    tuesday = a + d ∧
    wednesday = a + 2 * d ∧
    thursday = a + 3 * d ∧
    friday = a + 4 * d ∧
    saturday = a + 5 * d ∧
    sunday = a + 6 * d ∧
    (monday + tuesday + wednesday + thursday + friday + saturday + sunday = target_total) →
    wednesday = 10 :=
by
  sorry

end joann_lollipop_wednesday_l1190_119069


namespace Jake_has_more_peaches_than_Jill_l1190_119062

variables (Jake Steven Jill : ℕ)
variable (h1 : Jake = Steven - 5)
variable (h2 : Steven = Jill + 18)
variable (h3 : Jill = 87)

theorem Jake_has_more_peaches_than_Jill (Jake Steven Jill : ℕ) (h1 : Jake = Steven - 5) (h2 : Steven = Jill + 18) (h3 : Jill = 87) :
  Jake - Jill = 13 :=
by
  sorry

end Jake_has_more_peaches_than_Jill_l1190_119062


namespace line_passes_through_fixed_point_l1190_119092

theorem line_passes_through_fixed_point 
  (m : ℝ) : ∃ x y : ℝ, y = m * x + (2 * m + 1) ∧ (x, y) = (-2, 1) :=
by
  use (-2), (1)
  sorry

end line_passes_through_fixed_point_l1190_119092


namespace jebb_take_home_pay_l1190_119027

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l1190_119027


namespace soccer_tournament_probability_l1190_119097

noncomputable def prob_teamA_more_points : ℚ :=
  (163 : ℚ) / 256

theorem soccer_tournament_probability :
  m + n = 419 ∧ prob_teamA_more_points = 163 / 256 := sorry

end soccer_tournament_probability_l1190_119097


namespace students_surveyed_l1190_119094

theorem students_surveyed (S : ℕ)
  (h1 : (2/3 : ℝ) * 6 + (1/3 : ℝ) * 4 = 16/3)
  (h2 : S * (16/3 : ℝ) = 320) :
  S = 60 :=
sorry

end students_surveyed_l1190_119094


namespace hypotenuse_length_l1190_119035

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l1190_119035


namespace class_gpa_l1190_119023

theorem class_gpa (n : ℕ) (h_n : n = 60)
  (n1 : ℕ) (h_n1 : n1 = 20) (gpa1 : ℕ) (h_gpa1 : gpa1 = 15)
  (n2 : ℕ) (h_n2 : n2 = 15) (gpa2 : ℕ) (h_gpa2 : gpa2 = 17)
  (n3 : ℕ) (h_n3 : n3 = 25) (gpa3 : ℕ) (h_gpa3 : gpa3 = 19) :
  (20 * 15 + 15 * 17 + 25 * 19 : ℕ) / 60 = 1717 / 100 := 
sorry

end class_gpa_l1190_119023


namespace Kolya_is_correct_Valya_is_incorrect_l1190_119022

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l1190_119022


namespace monotonically_increasing_range_of_a_l1190_119084

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end monotonically_increasing_range_of_a_l1190_119084


namespace rice_weight_per_container_in_grams_l1190_119090

-- Define the initial problem conditions
def total_weight_pounds : ℚ := 35 / 6
def number_of_containers : ℕ := 5
def pound_to_grams : ℚ := 453.592

-- Define the expected answer
def expected_answer : ℚ := 529.1907

-- The statement to prove
theorem rice_weight_per_container_in_grams :
  (total_weight_pounds / number_of_containers) * pound_to_grams = expected_answer :=
by
  sorry

end rice_weight_per_container_in_grams_l1190_119090


namespace average_probable_weight_l1190_119056

theorem average_probable_weight (weight : ℝ) (h1 : 61 < weight) (h2 : weight ≤ 64) : 
  (61 + 64) / 2 = 62.5 := 
by
  sorry

end average_probable_weight_l1190_119056


namespace certain_number_l1190_119088

theorem certain_number (x : ℝ) : 
  0.55 * x = (4/5 : ℝ) * 25 + 2 → 
  x = 40 :=
by
  sorry

end certain_number_l1190_119088


namespace jerry_total_logs_l1190_119021

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end jerry_total_logs_l1190_119021


namespace restaurant_bill_split_l1190_119020

def original_bill : ℝ := 514.16
def tip_rate : ℝ := 0.18
def number_of_people : ℕ := 9
def final_amount_per_person : ℝ := 67.41

theorem restaurant_bill_split :
  final_amount_per_person = (1 + tip_rate) * original_bill / number_of_people :=
by
  sorry

end restaurant_bill_split_l1190_119020


namespace steve_average_speed_l1190_119045

/-
Problem Statement:
Prove that the average speed of Steve's travel for the entire journey is 55 mph given the following conditions:
1. Steve's first part of journey: 5 hours at 40 mph.
2. Steve's second part of journey: 3 hours at 80 mph.
-/

theorem steve_average_speed :
  let time1 := 5 -- hours
  let speed1 := 40 -- mph
  let time2 := 3 -- hours
  let speed2 := 80 -- mph
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 55 := by
  sorry

end steve_average_speed_l1190_119045


namespace set_union_l1190_119082

theorem set_union :
  let M := {x | x^2 + 2 * x - 3 = 0}
  let N := {-1, 2, 3}
  M ∪ N = {-1, 1, 2, -3, 3} :=
by
  sorry

end set_union_l1190_119082


namespace total_driving_routes_l1190_119012

def num_starting_points : ℕ := 4
def num_destinations : ℕ := 3

theorem total_driving_routes (h1 : ¬(num_starting_points = 0)) (h2 : ¬(num_destinations = 0)) : 
  num_starting_points * num_destinations = 12 :=
by
  sorry

end total_driving_routes_l1190_119012


namespace triangle_type_l1190_119078

-- Definitions given in the problem
def is_not_equal (a : ℝ) (b : ℝ) : Prop := a ≠ b
def log_eq (b x : ℝ) : Prop := Real.log x = Real.log 4 / Real.log b + Real.log (4 * x - 4) / Real.log b

-- Main theorem stating the type of triangle ABC
theorem triangle_type (a b c A B C : ℝ) (h_b_ne_1 : is_not_equal b 1) (h_C_over_A_root : log_eq b (C / A)) (h_sin_B_over_sin_A_root : log_eq b (Real.sin B / Real.sin A)) : (B = 90) ∧ (A ≠ C) :=
by
  sorry

end triangle_type_l1190_119078


namespace set_characteristics_l1190_119091

-- Define the characteristics of elements in a set
def characteristic_definiteness := true
def characteristic_distinctness := true
def characteristic_unorderedness := true
def characteristic_reality := false -- We aim to prove this

-- The problem statement in Lean
theorem set_characteristics :
  ¬ characteristic_reality :=
by
  -- Here would be the proof, but we add sorry as indicated.
  sorry

end set_characteristics_l1190_119091


namespace pure_imaginary_z1_over_z2_l1190_119064

theorem pure_imaginary_z1_over_z2 (b : Real) : 
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  (Complex.re ((z1 / z2) : Complex)) = 0 → b = -3 / 2 :=
by
  intros
  -- Conditions
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  -- Assuming that the real part of (z1 / z2) is zero
  have h : Complex.re (z1 / z2) = 0 := ‹_›
  -- Require to prove that b = -3 / 2
  sorry

end pure_imaginary_z1_over_z2_l1190_119064


namespace daniel_candy_removal_l1190_119016

theorem daniel_candy_removal (n k : ℕ) (h1 : n = 24) (h2 : k = 4) : ∃ m : ℕ, n % k = 0 → m = 0 :=
by
  sorry

end daniel_candy_removal_l1190_119016


namespace max_wooden_pencils_l1190_119067

theorem max_wooden_pencils (m w : ℕ) (p : ℕ) (h1 : m + w = 72) (h2 : m = w + p) (hp : Nat.Prime p) : w = 35 :=
by
  sorry

end max_wooden_pencils_l1190_119067


namespace remainder_when_four_times_n_minus_9_l1190_119007

theorem remainder_when_four_times_n_minus_9
  (n : ℤ) (h : n % 5 = 3) : (4 * n - 9) % 5 = 3 := 
by 
  sorry

end remainder_when_four_times_n_minus_9_l1190_119007


namespace find_a1_l1190_119036

theorem find_a1 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (1 - a n)) (h2 : a 8 = 2)
: a 1 = 1 / 2 :=
sorry

end find_a1_l1190_119036


namespace positive_integers_solution_l1190_119008

open Nat

theorem positive_integers_solution (a b m n : ℕ) (r : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h_gcd : Nat.gcd m n = 1) :
  (a^2 + b^2)^m = (a * b)^n ↔ a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 :=
sorry

end positive_integers_solution_l1190_119008


namespace total_rainbow_nerds_l1190_119018

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end total_rainbow_nerds_l1190_119018


namespace total_cost_is_correct_l1190_119032

def num_children : ℕ := 5
def daring_children : ℕ := 3
def ferris_wheel_cost_per_child : ℕ := 5
def merry_go_round_cost_per_child : ℕ := 3
def ice_cream_cones_per_child : ℕ := 2
def ice_cream_cost_per_cone : ℕ := 8

def total_spent_on_ferris_wheel : ℕ := daring_children * ferris_wheel_cost_per_child
def total_spent_on_merry_go_round : ℕ := num_children * merry_go_round_cost_per_child
def total_spent_on_ice_cream : ℕ := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone

def total_spent : ℕ := total_spent_on_ferris_wheel + total_spent_on_merry_go_round + total_spent_on_ice_cream

theorem total_cost_is_correct : total_spent = 110 := by
  sorry

end total_cost_is_correct_l1190_119032


namespace solve_system_of_equations_l1190_119014

theorem solve_system_of_equations (x y : ℝ) (h1 : 2 * x + 3 * y = 7) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
    -- The proof is not required, so we put a sorry here.
    sorry

end solve_system_of_equations_l1190_119014


namespace perpendicular_chords_cosine_bound_l1190_119033

theorem perpendicular_chords_cosine_bound 
  (a b : ℝ) 
  (h_ab : a > b) 
  (h_b0 : b > 0) 
  (θ1 θ2 : ℝ) 
  (x y : ℝ → ℝ) 
  (h_ellipse : ∀ t, x t = a * Real.cos t ∧ y t = b * Real.sin t) 
  (h_theta1 : ∃ t1, (x t1 = a * Real.cos θ1 ∧ y t1 = b * Real.sin θ1)) 
  (h_theta2 : ∃ t2, (x t2 = a * Real.cos θ2 ∧ y t2 = b * Real.sin θ2)) 
  (h_perpendicular: θ1 = θ2 + π / 2 ∨ θ1 = θ2 - π / 2) :
  0 ≤ |Real.cos (θ1 - θ2)| ∧ |Real.cos (θ1 - θ2)| ≤ (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2) :=
sorry

end perpendicular_chords_cosine_bound_l1190_119033


namespace sequence_general_formula_l1190_119044

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  ∀ n, a n = n^2 - n + 1 :=
by sorry

end sequence_general_formula_l1190_119044


namespace ellipse_standard_equation_parabola_standard_equation_l1190_119099

theorem ellipse_standard_equation (x y : ℝ) (a b : ℝ) (h₁ : a > b ∧ b > 0)
  (h₂ : 2 * a = Real.sqrt ((3 + 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2) 
      + Real.sqrt ((3 - 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2))
  (h₃ : b^2 = a^2 - 4) 
  : (x^2 / 36 + y^2 / 32 = 1) :=
by sorry

theorem parabola_standard_equation (y : ℝ) (p : ℝ) (h₁ : p > 0)
  (h₂ : -p / 2 = -1 / 2) 
  : (y^2 = 2 * p * 1) :=
by sorry

end ellipse_standard_equation_parabola_standard_equation_l1190_119099


namespace red_lettuce_cost_l1190_119048

-- Define the known conditions
def cost_per_pound : Nat := 2
def total_pounds : Nat := 7
def cost_green_lettuce : Nat := 8

-- Define the total cost calculation
def total_cost : Nat := total_pounds * cost_per_pound
def cost_red_lettuce : Nat := total_cost - cost_green_lettuce

-- Statement to prove: cost_red_lettuce = 6
theorem red_lettuce_cost :
  cost_red_lettuce = 6 :=
by
  sorry

end red_lettuce_cost_l1190_119048


namespace find_initial_principal_amount_l1190_119087

noncomputable def compound_interest (initial_principal : ℝ) : ℝ :=
  let year1 := initial_principal * 1.09
  let year2 := (year1 + 500) * 1.10
  let year3 := (year2 - 300) * 1.08
  let year4 := year3 * 1.08
  let year5 := year4 * 1.09
  year5

theorem find_initial_principal_amount :
  ∃ (P : ℝ), (|compound_interest P - 1120| < 0.01) :=
sorry

end find_initial_principal_amount_l1190_119087


namespace initial_candies_l1190_119065

-- Define initial variables and conditions
variable (x : ℕ)
variable (remaining_candies_after_first_day : ℕ)
variable (remaining_candies_after_second_day : ℕ)

-- Conditions as per given problem
def condition1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
def condition2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
def final_condition : remaining_candies_after_second_day = 10 := sorry

-- Goal: Prove that initially, Liam had 52 candies
theorem initial_candies : x = 52 := by
  have h1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
  have h2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
  have h3 : remaining_candies_after_second_day = 10 := sorry
    
  -- Combine conditions to solve for x
  sorry

end initial_candies_l1190_119065


namespace possible_values_of_f_zero_l1190_119031

theorem possible_values_of_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x * f y) :
  f 0 = 0 ∨ f 0 = 1 :=
by
  sorry

end possible_values_of_f_zero_l1190_119031


namespace range_of_a_l1190_119000

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → a < x ∧ x < 5) → a ≤ 1 := 
sorry

end range_of_a_l1190_119000


namespace sufficient_not_necessary_l1190_119093

-- Definitions based on the conditions
def f1 (x y : ℝ) : Prop := x^2 + y^2 = 0
def f2 (x y : ℝ) : Prop := x * y = 0

-- The theorem we need to prove
theorem sufficient_not_necessary (x y : ℝ) : f1 x y → f2 x y ∧ ¬ (f2 x y → f1 x y) := 
by sorry

end sufficient_not_necessary_l1190_119093


namespace lengths_of_triangle_sides_l1190_119060

open Real

noncomputable def triangle_side_lengths (a b c : ℝ) (A B C : ℝ) :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ A + B + C = π ∧ A = 60 * π / 180 ∧
  10 * sqrt 3 = 0.5 * a * b * sin A ∧
  a + b = 13 ∧
  c = sqrt (a^2 + b^2 - 2 * a * b * cos A)

theorem lengths_of_triangle_sides
  (a b c : ℝ) (A B C : ℝ)
  (h : triangle_side_lengths a b c A B C) :
  (a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7) :=
sorry

end lengths_of_triangle_sides_l1190_119060


namespace codys_grandmother_age_l1190_119072

theorem codys_grandmother_age
  (cody_age : ℕ)
  (grandmother_multiplier : ℕ)
  (h_cody_age : cody_age = 14)
  (h_grandmother_multiplier : grandmother_multiplier = 6) :
  (cody_age * grandmother_multiplier = 84) :=
by
  sorry

end codys_grandmother_age_l1190_119072


namespace minimum_phi_l1190_119073

noncomputable def initial_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * x + ϕ)

noncomputable def translated_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * (x - (Real.pi / 6)) + ϕ)

theorem minimum_phi (ϕ : ℝ) :
  (∃ k : ℤ, ϕ = k * Real.pi + 7 * Real.pi / 6) →
  (∃ ϕ_min : ℝ, (ϕ_min = ϕ ∧ ϕ_min = Real.pi / 6)) :=
by
  sorry

end minimum_phi_l1190_119073


namespace find_root_equation_l1190_119063

theorem find_root_equation : ∃ x : ℤ, x - (5 / (x - 4)) = 2 - (5 / (x - 4)) ∧ x = 2 :=
by
  sorry

end find_root_equation_l1190_119063


namespace maximize_quadratic_expression_l1190_119061

theorem maximize_quadratic_expression :
  ∃ x : ℝ, (∀ y : ℝ, -2 * y^2 - 8 * y + 10 ≤ -2 * x^2 - 8 * x + 10) ∧ x = -2 :=
by
  sorry

end maximize_quadratic_expression_l1190_119061


namespace tod_north_distance_l1190_119028

-- Given conditions as variables
def speed : ℕ := 25  -- speed in miles per hour
def time : ℕ := 6    -- time in hours
def west_distance : ℕ := 95  -- distance to the west in miles

-- Prove the distance to the north given conditions
theorem tod_north_distance : time * speed - west_distance = 55 := by
  sorry

end tod_north_distance_l1190_119028


namespace sasha_work_fraction_l1190_119049

theorem sasha_work_fraction :
  let sasha_first := 1 / 3
  let sasha_second := 1 / 5
  let sasha_third := 1 / 15
  let total_sasha_contribution := sasha_first + sasha_second + sasha_third
  let fraction_per_car := total_sasha_contribution / 3
  fraction_per_car = 1 / 5 :=
by
  sorry

end sasha_work_fraction_l1190_119049


namespace intersection_subset_proper_l1190_119013

-- Definitions of P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- The problem statement to prove
theorem intersection_subset_proper : P ∩ Q ⊂ P := by
  sorry

end intersection_subset_proper_l1190_119013


namespace tammy_earnings_after_3_weeks_l1190_119025

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end tammy_earnings_after_3_weeks_l1190_119025


namespace overall_percentage_gain_is_0_98_l1190_119005

noncomputable def original_price : ℝ := 100
noncomputable def increased_price := original_price * 1.32
noncomputable def after_first_discount := increased_price * 0.90
noncomputable def final_price := after_first_discount * 0.85
noncomputable def overall_gain := final_price - original_price
noncomputable def overall_percentage_gain := (overall_gain / original_price) * 100

theorem overall_percentage_gain_is_0_98 :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_is_0_98_l1190_119005


namespace sufficient_condition_for_gt_l1190_119083

theorem sufficient_condition_for_gt (a : ℝ) : (∀ x : ℝ, x > a → x > 1) → (∃ x : ℝ, x > 1 ∧ x ≤ a) → a > 1 :=
by
  sorry

end sufficient_condition_for_gt_l1190_119083


namespace rectangle_ratio_l1190_119055

theorem rectangle_ratio (s : ℝ) (w h : ℝ) (h_cond : h = 3 * s) (w_cond : w = 2 * s) :
  h / w = 3 / 2 :=
by
  sorry

end rectangle_ratio_l1190_119055


namespace max_value_m_l1190_119086

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end max_value_m_l1190_119086


namespace solution_xyz_uniqueness_l1190_119038

theorem solution_xyz_uniqueness (x y z : ℝ) :
  x + y + z = 3 ∧ x^2 + y^2 + z^2 = 3 ∧ x^3 + y^3 + z^3 = 3 → x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end solution_xyz_uniqueness_l1190_119038


namespace solve_equations_l1190_119096

theorem solve_equations :
  (∀ x : ℝ, x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2) ∧ 
  (∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ↔ x = 1/2) :=
by sorry

end solve_equations_l1190_119096


namespace graph_passes_through_quadrants_l1190_119004

theorem graph_passes_through_quadrants (k : ℝ) (h : k < 0) :
  ∀ (x y : ℝ), (y = k * x - k) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_passes_through_quadrants_l1190_119004


namespace min_workers_for_profit_l1190_119071

theorem min_workers_for_profit
    (maintenance_fees : ℝ)
    (worker_hourly_wage : ℝ)
    (widgets_per_hour : ℝ)
    (widget_price : ℝ)
    (work_hours : ℝ)
    (n : ℕ)
    (h_maintenance : maintenance_fees = 470)
    (h_wage : worker_hourly_wage = 10)
    (h_production : widgets_per_hour = 6)
    (h_price : widget_price = 3.5)
    (h_hours : work_hours = 8) :
  470 + 80 * n < 168 * n → n ≥ 6 := 
by
  sorry

end min_workers_for_profit_l1190_119071


namespace pencil_sharpening_time_l1190_119009

theorem pencil_sharpening_time (t : ℕ) :
  let hand_crank_rate := 45
  let electric_rate := 20
  let sharpened_by_hand := (60 * t) / hand_crank_rate
  let sharpened_by_electric := (60 * t) / electric_rate
  (sharpened_by_electric = sharpened_by_hand + 10) → 
  t = 6 :=
by
  intros hand_crank_rate electric_rate sharpened_by_hand sharpened_by_electric h
  sorry

end pencil_sharpening_time_l1190_119009


namespace find_first_group_men_l1190_119077

variable (M : ℕ)

def first_group_men := M
def days_for_first_group := 20
def men_in_second_group := 12
def days_for_second_group := 30

theorem find_first_group_men (h1 : first_group_men * days_for_first_group = men_in_second_group * days_for_second_group) :
  first_group_men = 18 :=
by {
  sorry
}

end find_first_group_men_l1190_119077


namespace correctly_subtracted_value_l1190_119030

theorem correctly_subtracted_value (x : ℤ) (h1 : 122 = x - 64) : 
  x - 46 = 140 :=
by
  -- Proof goes here
  sorry

end correctly_subtracted_value_l1190_119030


namespace sum_of_numbers_l1190_119029

theorem sum_of_numbers :
  2.12 + 0.004 + 0.345 = 2.469 :=
sorry

end sum_of_numbers_l1190_119029


namespace mars_moon_cost_share_l1190_119080

theorem mars_moon_cost_share :
  let total_cost := 40 * 10^9 -- total cost in dollars
  let num_people := 200 * 10^6 -- number of people sharing the cost
  (total_cost / num_people) = 200 := by
  sorry

end mars_moon_cost_share_l1190_119080


namespace shaded_area_eq_63_l1190_119003

noncomputable def rect1_width : ℕ := 4
noncomputable def rect1_height : ℕ := 12
noncomputable def rect2_width : ℕ := 5
noncomputable def rect2_height : ℕ := 7
noncomputable def overlap_width : ℕ := 4
noncomputable def overlap_height : ℕ := 5

theorem shaded_area_eq_63 :
  (rect1_width * rect1_height) + (rect2_width * rect2_height) - (overlap_width * overlap_height) = 63 := by
  sorry

end shaded_area_eq_63_l1190_119003


namespace current_failing_rate_l1190_119051

def failing_student_rate := 28

def is_failing_student_rate (V : Prop) (n : ℕ) (rate : ℕ) : Prop :=
  (V ∧ rate = 24 ∧ n = 25) ∨ (¬V ∧ rate = 25 ∧ n - 1 = 24)

theorem current_failing_rate (V : Prop) (n : ℕ) (rate : ℕ) :
  is_failing_student_rate V n rate → rate = failing_student_rate :=
by
  sorry

end current_failing_rate_l1190_119051


namespace average_marks_of_all_students_l1190_119074

theorem average_marks_of_all_students (n₁ n₂ a₁ a₂ : ℕ) (h₁ : n₁ = 30) (h₂ : a₁ = 40) (h₃ : n₂ = 50) (h₄ : a₂ = 80) :
  ((n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 65) :=
by
  sorry

end average_marks_of_all_students_l1190_119074


namespace trigonometric_identity_l1190_119019

open Real

noncomputable def sin_alpha (x y : ℝ) : ℝ :=
  y / sqrt (x^2 + y^2)

noncomputable def tan_alpha (x y : ℝ) : ℝ :=
  y / x

theorem trigonometric_identity (x y : ℝ) (h_x : x = 3/5) (h_y : y = -4/5) :
  sin_alpha x y * tan_alpha x y = 16/15 :=
by {
  -- math proof to be provided here
  sorry
}

end trigonometric_identity_l1190_119019


namespace odd_n_cube_plus_one_not_square_l1190_119079

theorem odd_n_cube_plus_one_not_square (n : ℤ) (h : n % 2 = 1) : ¬ ∃ (x : ℤ), x^2 = n^3 + 1 :=
by
  sorry

end odd_n_cube_plus_one_not_square_l1190_119079


namespace solve_equation_l1190_119017

theorem solve_equation (x : ℝ) : (x - 1) * (x + 3) = 5 ↔ x = 2 ∨ x = -4 := by
  sorry

end solve_equation_l1190_119017


namespace rectangular_box_in_sphere_radius_l1190_119011

theorem rectangular_box_in_sphere_radius (a b c s : ℝ) 
  (h1 : a + b + c = 40) 
  (h2 : 2 * a * b + 2 * b * c + 2 * a * c = 608) 
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) : 
  s = 16 * Real.sqrt 2 :=
by
  sorry

end rectangular_box_in_sphere_radius_l1190_119011


namespace main_l1190_119081

-- Definition for part (a)
def part_a : Prop :=
  ∀ (a b : ℕ), a = 300 ∧ b = 200 → 3^b > 2^a

-- Definition for part (b)
def part_b : Prop :=
  ∀ (c d : ℕ), c = 40 ∧ d = 28 → 3^d > 2^c

-- Definition for part (c)
def part_c : Prop :=
  ∀ (e f : ℕ), e = 44 ∧ f = 53 → 4^f > 5^e

-- Main conjecture proving all parts
theorem main : part_a ∧ part_b ∧ part_c :=
by
  sorry

end main_l1190_119081


namespace determine_avery_height_l1190_119037

-- Define Meghan's height
def meghan_height : ℕ := 188

-- Define range of players' heights
def height_range : ℕ := 33

-- Define the predicate to determine Avery's height
def avery_height : ℕ := meghan_height - height_range

-- The theorem we need to prove
theorem determine_avery_height : avery_height = 155 := by
  sorry

end determine_avery_height_l1190_119037


namespace radius_of_circle_area_of_sector_l1190_119053

theorem radius_of_circle (L : ℝ) (θ : ℝ) (hL : L = 50) (hθ : θ = 200) : 
  ∃ r : ℝ, r = 45 / Real.pi := 
by
  sorry

theorem area_of_sector (L : ℝ) (r : ℝ) (hL : L = 50) (hr : r = 45 / Real.pi) : 
  ∃ S : ℝ, S = 1125 / Real.pi := 
by
  sorry

end radius_of_circle_area_of_sector_l1190_119053


namespace prob_no_infection_correct_prob_one_infection_correct_l1190_119058

-- Probability that no chicken is infected
def prob_no_infection (p_not_infected : ℚ) (n : ℕ) : ℚ := p_not_infected^n

-- Given
def p_not_infected : ℚ := 4 / 5
def n : ℕ := 5

-- Expected answer for no chicken infected
def expected_prob_no_infection : ℚ := 1024 / 3125

-- Lean statement
theorem prob_no_infection_correct : 
  prob_no_infection p_not_infected n = expected_prob_no_infection := by
  sorry

-- Probability that exactly one chicken is infected
def prob_one_infection (p_infected : ℚ) (p_not_infected : ℚ) (n : ℕ) : ℚ := 
  (n * p_not_infected^(n-1) * p_infected)

-- Given
def p_infected : ℚ := 1 / 5

-- Expected answer for exactly one chicken infected
def expected_prob_one_infection : ℚ := 256 / 625

-- Lean statement
theorem prob_one_infection_correct : 
  prob_one_infection p_infected p_not_infected n = expected_prob_one_infection := by
  sorry

end prob_no_infection_correct_prob_one_infection_correct_l1190_119058


namespace platform_length_l1190_119024

theorem platform_length (train_length : ℕ) (time_cross_platform : ℕ) (time_cross_pole : ℕ) (train_speed : ℕ) (L : ℕ)
  (h1 : train_length = 500) 
  (h2 : time_cross_platform = 65) 
  (h3 : time_cross_pole = 25) 
  (h4 : train_speed = train_length / time_cross_pole)
  (h5 : train_speed = (train_length + L) / time_cross_platform) :
  L = 800 := 
sorry

end platform_length_l1190_119024


namespace units_digit_of_x_l1190_119043

theorem units_digit_of_x 
  (a x : ℕ) 
  (h1 : a * x = 14^8) 
  (h2 : a % 10 = 9) : 
  x % 10 = 4 := 
by 
  sorry

end units_digit_of_x_l1190_119043


namespace ten_integers_disjoint_subsets_same_sum_l1190_119054

theorem ten_integers_disjoint_subsets_same_sum (S : Finset ℕ) (h : S.card = 10) (h_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ≠ B ∧ A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end ten_integers_disjoint_subsets_same_sum_l1190_119054


namespace gcd_79625_51575_l1190_119040

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 :=
by
  sorry

end gcd_79625_51575_l1190_119040


namespace sum_of_numbers_le_1_1_l1190_119026

theorem sum_of_numbers_le_1_1 :
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  filtered.sum = 1.4 :=
by
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  have : filtered = [0.9, 0.5] := sorry
  have : filtered.sum = 1.4 := sorry
  exact this

end sum_of_numbers_le_1_1_l1190_119026


namespace function_passes_through_fixed_point_l1190_119002

noncomputable def passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : Prop :=
  ∃ y : ℝ, y = a^(1-1) + 1 ∧ y = 2

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : passes_through_fixed_point a h :=
by
  sorry

end function_passes_through_fixed_point_l1190_119002


namespace scientific_notation_correct_l1190_119052

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end scientific_notation_correct_l1190_119052


namespace num_of_nickels_is_two_l1190_119050

theorem num_of_nickels_is_two (d n : ℕ) 
    (h1 : 10 * d + 5 * n = 70) 
    (h2 : d + n = 8) : 
    n = 2 := 
by 
    sorry

end num_of_nickels_is_two_l1190_119050


namespace transportable_load_l1190_119001

theorem transportable_load 
  (mass_of_load : ℝ) 
  (num_boxes : ℕ) 
  (box_capacity : ℝ) 
  (num_trucks : ℕ) 
  (truck_capacity : ℝ) 
  (h1 : mass_of_load = 13.5) 
  (h2 : box_capacity = 0.35) 
  (h3 : truck_capacity = 1.5) 
  (h4 : num_trucks = 11)
  (boxes_condition : ∀ (n : ℕ), n * box_capacity ≥ mass_of_load) :
  mass_of_load ≤ num_trucks * truck_capacity :=
by
  sorry

end transportable_load_l1190_119001


namespace unique_solution_l1190_119039

def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (hx : 0 < x), 
    ∃! (y : ℝ) (hy : 0 < y), 
      x * f y + y * f x ≤ 2

theorem unique_solution : ∀ (f : ℝ → ℝ), 
  is_solution f ↔ (∀ x, 0 < x → f x = 1 / x) :=
by
  intros
  sorry

end unique_solution_l1190_119039


namespace percentage_of_stock_l1190_119047

-- Definitions based on conditions
def income := 500  -- I
def investment := 1500  -- Inv
def price := 90  -- Price

-- Initiate the Lean 4 statement for the proof
theorem percentage_of_stock (P : ℝ) (h : income = (investment * P) / price) : P = 30 :=
by
  sorry

end percentage_of_stock_l1190_119047


namespace trigonometric_identity_l1190_119042

theorem trigonometric_identity : 
  (Real.sin (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (138 * Real.pi / 180) * Real.cos (72 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l1190_119042


namespace quadratic_trinomial_int_l1190_119066

theorem quadratic_trinomial_int (a b c x : ℤ) (h : y = (x - a) * (x - 6) + 1) :
  ∃ (b c : ℤ), (x + b) * (x + c) = (x - 8) * (x - 6) + 1 :=
by
  sorry

end quadratic_trinomial_int_l1190_119066


namespace arithmetic_sequence_sum_l1190_119034

theorem arithmetic_sequence_sum
  (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (H1 : a1 = -2017)
  (H2 : (S 2013 : ℤ) / 2013 - (S 2011 : ℤ) / 2011 = 2)
  (H3 : ∀ n : ℕ, S n = n * a1 + (n * (n - 1) / 2) * d) :
  S 2017 = -2017 :=
by
  sorry

end arithmetic_sequence_sum_l1190_119034


namespace heartsuit_value_l1190_119068

def heartsuit (x y : ℝ) := 4 * x + 6 * y

theorem heartsuit_value : heartsuit 3 4 = 36 := by
  sorry

end heartsuit_value_l1190_119068


namespace sail_time_difference_l1190_119046

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end sail_time_difference_l1190_119046


namespace max_non_managers_l1190_119015

-- Definitions of the problem conditions
variable (m n : ℕ)
variable (h : m = 8)
variable (hratio : (7:ℚ) / 24 < m / n)

-- The theorem we need to prove
theorem max_non_managers (m n : ℕ) (h : m = 8) (hratio : ((7:ℚ) / 24 < m / n)) :
  n ≤ 27 := 
sorry

end max_non_managers_l1190_119015


namespace income_expenditure_ratio_l1190_119075

theorem income_expenditure_ratio (I E S : ℝ) (h1 : I = 20000) (h2 : S = 4000) (h3 : S = I - E) :
    I / E = 5 / 4 :=
sorry

end income_expenditure_ratio_l1190_119075


namespace carvings_per_shelf_l1190_119098

def total_wood_carvings := 56
def num_shelves := 7

theorem carvings_per_shelf : total_wood_carvings / num_shelves = 8 := by
  sorry

end carvings_per_shelf_l1190_119098


namespace x_minus_y_value_l1190_119006

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) : x - y = 1 ∨ x - y = 5 := by
  sorry

end x_minus_y_value_l1190_119006


namespace find_adult_buffet_price_l1190_119059

variable {A : ℝ} -- Let A be the price for the adult buffet
variable (children_cost : ℝ := 45) -- Total cost for the children's buffet
variable (senior_discount : ℝ := 0.9) -- Discount for senior citizens
variable (total_cost : ℝ := 159) -- Total amount spent by Mr. Smith
variable (num_adults : ℕ := 2) -- Number of adults (Mr. Smith and his wife)
variable (num_seniors : ℕ := 2) -- Number of senior citizens

theorem find_adult_buffet_price (h1 : children_cost = 45)
    (h2 : total_cost = 159)
    (h3 : ∀ x, num_adults * x + num_seniors * (senior_discount * x) + children_cost = total_cost)
    : A = 30 :=
by
  sorry

end find_adult_buffet_price_l1190_119059


namespace book_prices_purchasing_plans_l1190_119089

theorem book_prices (x y : ℕ) (h1 : 20 * x + 40 * y = 1600) (h2 : 20 * x = 30 * y + 200) : x = 40 ∧ y = 20 :=
by
  sorry

theorem purchasing_plans (m : ℕ) (h3 : 2 * m + 20 ≥ 70) (h4 : 40 * m + 20 * (m + 20) ≤ 2000) :
  (m = 25 ∧ m + 20 = 45) ∨ (m = 26 ∧ m + 20 = 46) :=
by
  -- proof steps
  sorry

end book_prices_purchasing_plans_l1190_119089


namespace boss_salary_percentage_increase_l1190_119070

theorem boss_salary_percentage_increase (W B : ℝ) (h : W = 0.2 * B) : ((B / W - 1) * 100) = 400 := by
sorry

end boss_salary_percentage_increase_l1190_119070


namespace average_goals_l1190_119095

def num_goals_3 := 3
def num_players_3 := 2
def num_goals_4 := 4
def num_players_4 := 3
def num_goals_5 := 5
def num_players_5 := 1
def num_goals_6 := 6
def num_players_6 := 1

def total_goals := (num_goals_3 * num_players_3) + (num_goals_4 * num_players_4) + (num_goals_5 * num_players_5) + (num_goals_6 * num_players_6)
def total_players := num_players_3 + num_players_4 + num_players_5 + num_players_6

theorem average_goals :
  (total_goals / total_players : ℚ) = 29 / 7 :=
sorry

end average_goals_l1190_119095


namespace gain_percentage_is_15_l1190_119076

-- Initial conditions
def CP_A : ℤ := 100
def CP_B : ℤ := 200
def CP_C : ℤ := 300
def SP_A : ℤ := 110
def SP_B : ℤ := 250
def SP_C : ℤ := 330

-- Definitions for total values
def Total_CP : ℤ := CP_A + CP_B + CP_C
def Total_SP : ℤ := SP_A + SP_B + SP_C
def Overall_gain : ℤ := Total_SP - Total_CP
def Gain_percentage : ℚ := (Overall_gain * 100) / Total_CP

-- Theorem to prove the overall gain percentage
theorem gain_percentage_is_15 :
  Gain_percentage = 15 := 
by
  -- Proof placeholder
  sorry

end gain_percentage_is_15_l1190_119076


namespace union_is_correct_l1190_119010

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_is_correct : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_is_correct_l1190_119010


namespace total_games_played_l1190_119041

-- Define the conditions as Lean 4 definitions
def games_won : Nat := 12
def games_lost : Nat := 4

-- Prove the total number of games played is 16
theorem total_games_played : games_won + games_lost = 16 := 
by
  -- Place a proof placeholder
  sorry

end total_games_played_l1190_119041


namespace bryan_total_after_discount_l1190_119085

theorem bryan_total_after_discount 
  (n : ℕ) (p : ℝ) (d : ℝ) (h_n : n = 8) (h_p : p = 1785) (h_d : d = 0.12) :
  (n * p - (n * p * d) = 12566.4) :=
by
  sorry

end bryan_total_after_discount_l1190_119085


namespace line_slope_intercept_l1190_119057

theorem line_slope_intercept (a b: ℝ) (h₁: ∀ x y, (x, y) = (2, 3) ∨ (x, y) = (10, 19) → y = a * x + b)
  (h₂: (a * 6 + b) = 11) : a - b = 3 :=
by
  sorry

end line_slope_intercept_l1190_119057
