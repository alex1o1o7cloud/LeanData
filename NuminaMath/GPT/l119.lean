import Mathlib

namespace square_side_length_l119_11956

variables (s : ℝ) (π : ℝ)
  
theorem square_side_length (h : 4 * s = π * s^2 / 2) : s = 8 / π :=
by sorry

end square_side_length_l119_11956


namespace points_on_line_l119_11982

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l119_11982


namespace yan_distance_ratio_l119_11944

-- Define conditions
variable (x z w: ℝ)  -- x: distance from Yan to his home, z: distance from Yan to the school, w: Yan's walking speed
variable (h1: z / w = x / w + (x + z) / (5 * w))  -- Both choices require the same amount of time

-- The ratio of Yan's distance from his home to his distance from the school is 2/3
theorem yan_distance_ratio :
    x / z = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l119_11944


namespace correct_option_l119_11954

theorem correct_option : (∃ x, x = -3 ∧ x^3 = -27) :=
by {
  -- Given conditions
  let x := -3
  use x
  constructor
  . rfl
  . norm_num
}

end correct_option_l119_11954


namespace vertex_parabola_shape_l119_11999

theorem vertex_parabola_shape
  (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (P : ℝ → ℝ → Prop), 
  (∀ t : ℝ, ∃ (x y : ℝ), P x y ∧ (x = (-t / (2 * a))) ∧ (y = -a * (x^2) + d)) ∧
  (∀ x y : ℝ, P x y ↔ (y = -a * (x^2) + d)) :=
by
  sorry

end vertex_parabola_shape_l119_11999


namespace solve_for_A_l119_11902

variable (a b : ℝ) 

theorem solve_for_A (A : ℝ) (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : 
  A = 60 * a * b := by
  sorry

end solve_for_A_l119_11902


namespace sum_of_final_two_numbers_l119_11909

theorem sum_of_final_two_numbers (a b S : ℝ) (h : a + b = S) : 
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_final_two_numbers_l119_11909


namespace percentage_of_allowance_spent_l119_11938

noncomputable def amount_spent : ℝ := 14
noncomputable def amount_left : ℝ := 26
noncomputable def total_allowance : ℝ := amount_spent + amount_left

theorem percentage_of_allowance_spent :
  ((amount_spent / total_allowance) * 100) = 35 := 
by 
  sorry

end percentage_of_allowance_spent_l119_11938


namespace minimum_point_translation_l119_11997

noncomputable def f (x : ℝ) : ℝ := |x| - 2

theorem minimum_point_translation :
  let minPoint := (0, f 0)
  let newMinPoint := (minPoint.1 + 4, minPoint.2 + 5)
  newMinPoint = (4, 3) :=
by
  sorry

end minimum_point_translation_l119_11997


namespace kim_candy_bars_saved_l119_11980

theorem kim_candy_bars_saved
  (n : ℕ)
  (c : ℕ)
  (w : ℕ)
  (total_bought : ℕ := n * c)
  (total_eaten : ℕ := n / w)
  (candy_bars_saved : ℕ := total_bought - total_eaten) :
  candy_bars_saved = 28 :=
by
  sorry

end kim_candy_bars_saved_l119_11980


namespace find_velocity_of_current_l119_11978

-- Define the conditions given in the problem
def rowing_speed_in_still_water : ℤ := 10
def distance_to_place : ℤ := 48
def total_travel_time : ℤ := 10

-- Define the primary goal, which is to find the velocity of the current given the conditions
theorem find_velocity_of_current (v : ℤ) 
  (h1 : rowing_speed_in_still_water = 10)
  (h2 : distance_to_place = 48)
  (h3 : total_travel_time = 10) 
  (h4 : rowing_speed_in_still_water * 2 + v * 0 = 
   rowing_speed_in_still_water - v) :
  v = 2 := 
sorry

end find_velocity_of_current_l119_11978


namespace total_food_in_10_days_l119_11923

theorem total_food_in_10_days :
  (let ella_food_per_day := 20
   let days := 10
   let dog_food_ratio := 4
   let ella_total_food := ella_food_per_day * days
   let dog_total_food := dog_food_ratio * ella_total_food
   ella_total_food + dog_total_food = 1000) :=
by
  sorry

end total_food_in_10_days_l119_11923


namespace min_value_expression_l119_11952

theorem min_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ x : ℝ, x = (1 / (a - 1)) + (1 / (2 * b)) ∧ x ≥ (3 / 2 + Real.sqrt 2)) :=
sorry

end min_value_expression_l119_11952


namespace calculate_fraction_l119_11918

def x : ℚ := 2 / 3
def y : ℚ := 8 / 10

theorem calculate_fraction :
  (6 * x + 10 * y) / (60 * x * y) = 3 / 8 := by
  sorry

end calculate_fraction_l119_11918


namespace option_D_correct_l119_11977

theorem option_D_correct (a : ℝ) :
  3 * a ^ 2 - a ≠ 2 * a ∧
  a - (1 - 2 * a) ≠ a - 1 ∧
  -5 * (1 - a ^ 2) ≠ -5 - 5 * a ^ 2 ∧
  a ^ 3 + 7 * a ^ 3 - 5 * a ^ 3 = 3 * a ^ 3 :=
by
  sorry

end option_D_correct_l119_11977


namespace spring_summer_work_hours_l119_11986

def john_works_spring_summer : Prop :=
  ∀ (work_hours_winter_week : ℕ) (weeks_winter : ℕ) (earnings_winter : ℕ)
    (weeks_spring_summer : ℕ) (earnings_spring_summer : ℕ) (hourly_rate : ℕ),
    work_hours_winter_week = 40 →
    weeks_winter = 8 →
    earnings_winter = 3200 →
    weeks_spring_summer = 24 →
    earnings_spring_summer = 4800 →
    hourly_rate = earnings_winter / (work_hours_winter_week * weeks_winter) →
    (earnings_spring_summer / hourly_rate) / weeks_spring_summer = 20

theorem spring_summer_work_hours : john_works_spring_summer :=
  sorry

end spring_summer_work_hours_l119_11986


namespace oranges_in_each_box_l119_11971

theorem oranges_in_each_box (O B : ℕ) (h1 : O = 24) (h2 : B = 3) :
  O / B = 8 :=
by
  sorry

end oranges_in_each_box_l119_11971


namespace volume_of_prism_l119_11942

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 20) (h3 : c * a = 12) (h4 : a + b + c = 11) :
  a * b * c = 12 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l119_11942


namespace bruce_money_left_l119_11996

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l119_11996


namespace range_of_square_of_difference_of_roots_l119_11964

theorem range_of_square_of_difference_of_roots (a : ℝ) (h : (a - 1) * (a - 2) < 0) :
  ∃ (S : Set ℝ), S = { x | 0 < x ∧ x ≤ 1 } ∧ ∀ (x1 x2 : ℝ),
  x1 + x2 = 2 * a ∧ x1 * x2 = 2 * a^2 - 3 * a + 2 → (x1 - x2)^2 ∈ S :=
sorry

end range_of_square_of_difference_of_roots_l119_11964


namespace intersection_when_a_eq_4_range_for_A_subset_B_l119_11908

-- Define the conditions
def setA : Set ℝ := { x | (1 - x) / (x - 7) > 0 }
def setB (a : ℝ) : Set ℝ := { x | x^2 - 2 * x - a^2 - 2 * a < 0 }

-- First proof goal: When a = 4, find A ∩ B
theorem intersection_when_a_eq_4 :
  setA ∩ (setB 4) = { x : ℝ | 1 < x ∧ x < 6 } :=
sorry

-- Second proof goal: Find the range for a such that A ⊆ B
theorem range_for_A_subset_B :
  { a : ℝ | setA ⊆ setB a } = { a : ℝ | a ≤ -7 ∨ a ≥ 5 } :=
sorry

end intersection_when_a_eq_4_range_for_A_subset_B_l119_11908


namespace hoodies_ownership_l119_11976

-- Step a): Defining conditions
variables (Fiona_casey_hoodies_total: ℕ) (Casey_difference: ℕ) (Alex_hoodies: ℕ)

-- Functions representing the constraints
def hoodies_owned_by_Fiona (F : ℕ) : Prop :=
  (F + (F + 2) + 3 = 15)

-- Step c): Prove the correct number of hoodies owned by each
theorem hoodies_ownership (F : ℕ) (H1 : hoodies_owned_by_Fiona F) : 
  F = 5 ∧ (F + 2 = 7) ∧ (3 = 3) :=
by {
  -- Skipping proof details
  sorry
}

end hoodies_ownership_l119_11976


namespace find_four_digit_number_l119_11900

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end find_four_digit_number_l119_11900


namespace total_dots_on_left_faces_l119_11940

-- Define the number of dots on the faces A, B, C, and D
def d_A : ℕ := 3
def d_B : ℕ := 5
def d_C : ℕ := 6
def d_D : ℕ := 5

-- The statement we need to prove
theorem total_dots_on_left_faces : d_A + d_B + d_C + d_D = 19 := by
  sorry

end total_dots_on_left_faces_l119_11940


namespace complement_intersection_l119_11947

-- Definitions to set the universal set and other sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

-- Complement of M with respect to U
def CU_M : Set ℕ := U \ M

-- Intersection of (CU_M) and N
def intersection_CU_M_N : Set ℕ := CU_M ∩ N

-- The proof problem statement
theorem complement_intersection :
  intersection_CU_M_N = {2, 5} :=
sorry

end complement_intersection_l119_11947


namespace scientist_born_on_saturday_l119_11984

noncomputable def day_of_week := List String

noncomputable def calculate_day := 
  let days_in_regular_years := 113
  let days_in_leap_years := 2 * 37
  let total_days_back := days_in_regular_years + days_in_leap_years
  total_days_back % 7

theorem scientist_born_on_saturday :
  let anniversary_day := 4  -- 0=Sunday, 1=Monday, ..., 4=Thursday
  calculate_day = 5 → 
  let birth_day := (anniversary_day + 7 - calculate_day) % 7 
  birth_day = 6 := sorry

end scientist_born_on_saturday_l119_11984


namespace min_bottles_needed_l119_11929

theorem min_bottles_needed (fluid_ounces_needed : ℝ) (bottle_size_ml : ℝ) (conversion_factor : ℝ) :
  fluid_ounces_needed = 60 ∧ bottle_size_ml = 250 ∧ conversion_factor = 33.8 →
  ∃ (n : ℕ), n = 8 ∧ (fluid_ounces_needed / conversion_factor * 1000 / bottle_size_ml) <= ↑n :=
by
  sorry

end min_bottles_needed_l119_11929


namespace distance_from_Beijing_to_Lanzhou_l119_11907

-- Conditions
def distance_Beijing_Lanzhou_Lhasa : ℕ := 3985
def distance_Lanzhou_Lhasa : ℕ := 2054

-- Define the distance from Beijing to Lanzhou
def distance_Beijing_Lanzhou : ℕ := distance_Beijing_Lanzhou_Lhasa - distance_Lanzhou_Lhasa

-- Proof statement that given conditions imply the correct answer
theorem distance_from_Beijing_to_Lanzhou :
  distance_Beijing_Lanzhou = 1931 :=
by
  -- conditions and definitions are already given
  sorry

end distance_from_Beijing_to_Lanzhou_l119_11907


namespace seq1_general_formula_seq2_general_formula_l119_11958

-- Sequence (1): Initial condition and recurrence relation
def seq1 (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + (2 * n - 1)

-- Proving the general formula for sequence (1)
theorem seq1_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq1 a) :
  a n = (n - 1) ^ 2 :=
sorry

-- Sequence (2): Initial condition and recurrence relation
def seq2 (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n

-- Proving the general formula for sequence (2)
theorem seq2_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq2 a) :
  a n = 3 ^ n :=
sorry

end seq1_general_formula_seq2_general_formula_l119_11958


namespace desks_in_classroom_l119_11990

theorem desks_in_classroom (d c : ℕ) (h1 : c = 4 * d) (h2 : 4 * c + 6 * d = 728) : d = 33 :=
by
  -- The proof is omitted, this placeholder is to indicate that it is required to complete the proof.
  sorry

end desks_in_classroom_l119_11990


namespace cage_chicken_problem_l119_11926

theorem cage_chicken_problem :
  (∃ x : ℕ, 6 ≤ x ∧ x ≤ 10 ∧ (4 * x + 1 = 5 * (x - 1))) ∧
  (∀ x : ℕ, 6 ≤ x ∧ x ≤ 10 → (4 * x + 1 ≥ 25 ∧ 4 * x + 1 ≤ 41)) :=
by
  sorry

end cage_chicken_problem_l119_11926


namespace erdos_problem_l119_11924

variable (X : Type) [Infinite X] (𝓗 : Set (Set X))
variable (h1 : ∀ (A : Set X) (hA : A.Finite), ∃ (H1 H2 : Set X) (hH1 : H1 ∈ 𝓗) (hH2 : H2 ∈ 𝓗), H1 ∩ H2 = ∅ ∧ H1 ∪ H2 = A)

theorem erdos_problem (k : ℕ) (hk : k > 0) : 
  ∃ (A : Set X) (ways : Finset (Set X × Set X)), A.Finite ∧ (∀ (p : Set X × Set X), p ∈ ways → p.1 ∈ 𝓗 ∧ p.2 ∈ 𝓗 ∧ p.1 ∩ p.2 = ∅ ∧ p.1 ∪ p.2 = A) ∧ ways.card ≥ k :=
by
  sorry

end erdos_problem_l119_11924


namespace candy_problem_l119_11939

-- Define conditions and the statement
theorem candy_problem (K : ℕ) (h1 : 49 = K + 3 * K + 8 + 6 + 10 + 5) : K = 5 :=
sorry

end candy_problem_l119_11939


namespace total_shells_is_correct_l119_11912

def morning_shells : Nat := 292
def afternoon_shells : Nat := 324
def total_shells : Nat := morning_shells + afternoon_shells

theorem total_shells_is_correct : total_shells = 616 :=
by
  sorry

end total_shells_is_correct_l119_11912


namespace sum_of_products_of_roots_l119_11927

noncomputable def poly : Polynomial ℝ := 5 * Polynomial.X^3 - 10 * Polynomial.X^2 + 17 * Polynomial.X - 7

theorem sum_of_products_of_roots :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ poly.eval p = 0 ∧ poly.eval q = 0 ∧ poly.eval r = 0) →
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ ((p * q + p * r + q * r) = 17 / 5)) :=
by
  sorry

end sum_of_products_of_roots_l119_11927


namespace alfred_bill_days_l119_11914

-- Definitions based on conditions
def combined_work_rate := 1 / 24
def alfred_to_bill_ratio := 2 / 3

-- Theorem statement
theorem alfred_bill_days (A B : ℝ) (ha : A = alfred_to_bill_ratio * B) (hcombined : A + B = combined_work_rate) : 
  A = 1 / 60 ∧ B = 1 / 40 :=
by
  sorry

end alfred_bill_days_l119_11914


namespace polynomial_irreducible_over_Z_iff_Q_l119_11950

theorem polynomial_irreducible_over_Z_iff_Q (f : Polynomial ℤ) :
  Irreducible f ↔ Irreducible (f.map (Int.castRingHom ℚ)) :=
sorry

end polynomial_irreducible_over_Z_iff_Q_l119_11950


namespace collinear_points_in_cube_l119_11974

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end collinear_points_in_cube_l119_11974


namespace quadratic_factor_conditions_l119_11921

theorem quadratic_factor_conditions (b : ℤ) :
  (∃ m n p q : ℤ, m * p = 15 ∧ n * q = 75 ∧ mq + np = b) → ∃ (c : ℤ), b = c :=
sorry

end quadratic_factor_conditions_l119_11921


namespace bus_passenger_count_l119_11920

-- Definition of the function f representing the number of passengers per trip
def passengers (n : ℕ) : ℕ :=
  120 - 2 * n

-- The total number of trips is 18 (from 9 AM to 5:30 PM inclusive)
def total_trips : ℕ := 18

-- Sum of passengers over all trips
def total_passengers : ℕ :=
  List.sum (List.map passengers (List.range total_trips))

-- Problem statement
theorem bus_passenger_count :
  total_passengers = 1854 :=
sorry

end bus_passenger_count_l119_11920


namespace proposition_C_correct_l119_11917

theorem proposition_C_correct (a b c : ℝ) (h : a * c ^ 2 > b * c ^ 2) : a > b :=
sorry

end proposition_C_correct_l119_11917


namespace anna_candy_division_l119_11985

theorem anna_candy_division : 
  ∀ (total_candies friends : ℕ), 
  total_candies = 30 → 
  friends = 4 → 
  ∃ (candies_to_remove : ℕ), 
  candies_to_remove = 2 ∧ 
  (total_candies - candies_to_remove) % friends = 0 := 
by
  sorry

end anna_candy_division_l119_11985


namespace total_time_of_flight_l119_11992

variables {V_0 g t t_1 H : ℝ}  -- Define variables

-- Define conditions
def initial_condition (V_0 g t_1 H : ℝ) : Prop :=
H = (1/2) * g * t_1^2

def return_condition (V_0 g t : ℝ) : Prop :=
t = 2 * (V_0 / g)

theorem total_time_of_flight
  (V_0 g : ℝ)
  (h1 : initial_condition V_0 g (V_0 / g) (1/2 * g * (V_0 / g)^2))
  : return_condition V_0 g (2 * V_0 / g) :=
by
  sorry

end total_time_of_flight_l119_11992


namespace production_relationship_l119_11910

noncomputable def production_function (a : ℕ) (p : ℝ) (x : ℕ) : ℝ := a * (1 + p / 100)^x

theorem production_relationship (a : ℕ) (p : ℝ) (m : ℕ) (x : ℕ) (hx : 0 ≤ x ∧ x ≤ m) :
  production_function a p x = a * (1 + p / 100)^x := by
  sorry

end production_relationship_l119_11910


namespace f_periodic_l119_11963

noncomputable def f : ℝ → ℝ := sorry

variable (a : ℝ) (h_a : 0 < a)
variable (h_cond : ∀ x : ℝ, f (x + a) = 1 / 2 + sqrt (f x - (f x)^2))

theorem f_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_periodic_l119_11963


namespace bread_cost_l119_11953

theorem bread_cost {packs_meat packs_cheese sandwiches : ℕ} 
  (cost_meat cost_cheese cost_sandwich coupon_meat coupon_cheese total_cost : ℝ) 
  (h_meat_cost : cost_meat = 5.00) 
  (h_cheese_cost : cost_cheese = 4.00)
  (h_coupon_meat : coupon_meat = 1.00)
  (h_coupon_cheese : coupon_cheese = 1.00)
  (h_cost_sandwich : cost_sandwich = 2.00)
  (h_packs_meat : packs_meat = 2)
  (h_packs_cheese : packs_cheese = 2)
  (h_sandwiches : sandwiches = 10)
  (h_total_revenue : total_cost = sandwiches * cost_sandwich) :
  ∃ (bread_cost : ℝ), bread_cost = total_cost - ((packs_meat * cost_meat - coupon_meat) + (packs_cheese * cost_cheese - coupon_cheese)) :=
sorry

end bread_cost_l119_11953


namespace cost_price_is_50_l119_11955

-- Define the conditions
def selling_price : ℝ := 80
def profit_rate : ℝ := 0.6

-- The cost price should be proven to be 50
def cost_price (C : ℝ) : Prop :=
  selling_price = C + (C * profit_rate)

theorem cost_price_is_50 : ∃ C : ℝ, cost_price C ∧ C = 50 := by
  sorry

end cost_price_is_50_l119_11955


namespace corn_plants_multiple_of_nine_l119_11995

theorem corn_plants_multiple_of_nine 
  (num_sunflowers : ℕ) (num_tomatoes : ℕ) (num_corn : ℕ) (max_plants_per_row : ℕ)
  (h1 : num_sunflowers = 45) (h2 : num_tomatoes = 63) (h3 : max_plants_per_row = 9)
  : ∃ k : ℕ, num_corn = 9 * k :=
by
  sorry

end corn_plants_multiple_of_nine_l119_11995


namespace new_class_mean_l119_11998

theorem new_class_mean 
  (n1 n2 : ℕ) 
  (mean1 mean2 : ℝ)
  (students_total : ℕ)
  (total_score1 total_score2 : ℝ)
  (h1 : n1 = 45)
  (h2 : n2 = 5)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : students_total = 50)
  (h6 : total_score1 = n1 * mean1)
  (h7 : total_score2 = n2 * mean2) :
  (total_score1 + total_score2) / students_total = 81 :=
by
  sorry

end new_class_mean_l119_11998


namespace find_a6_l119_11970

variable {a : ℕ → ℝ} -- Sequence a is indexed by natural numbers and the terms are real numbers.

-- Conditions
def a_is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)
def a1_eq_4 (a : ℕ → ℝ) := a 1 = 4
def a3_eq_a2_mul_a4 (a : ℕ → ℝ) := a 3 = a 2 * a 4

theorem find_a6 (a : ℕ → ℝ) 
  (h1 : a_is_geom_seq a)
  (h2 : a1_eq_4 a)
  (h3 : a3_eq_a2_mul_a4 a) : 
  a 6 = 1 / 8 ∨ a 6 = - (1 / 8) := 
by 
  sorry

end find_a6_l119_11970


namespace solution_set_of_inequality_l119_11946

theorem solution_set_of_inequality (x : ℝ) : (1 / |x - 1| ≥ 1) ↔ (0 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l119_11946


namespace Q1_no_such_a_b_Q2_no_such_a_b_c_l119_11935

theorem Q1_no_such_a_b :
  ∀ (a b : ℕ), (0 < a) ∧ (0 < b) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b) := sorry

theorem Q2_no_such_a_b_c :
  ∀ (a b c : ℕ), (0 < a) ∧ (0 < b) ∧ (0 < c) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b + c) := sorry

end Q1_no_such_a_b_Q2_no_such_a_b_c_l119_11935


namespace cost_per_bag_l119_11930

theorem cost_per_bag (total_friends: ℕ) (amount_paid_per_friend: ℕ) (total_bags: ℕ) 
  (h1 : total_friends = 3) (h2 : amount_paid_per_friend = 5) (h3 : total_bags = 5) 
  : total_friends * amount_paid_per_friend / total_bags = 3 := by
  sorry

end cost_per_bag_l119_11930


namespace four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l119_11965

theorem four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime
  (N : ℕ) (hN : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬ ∃ k : ℕ, k^2 = 4 * n * (N - n) + 1) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l119_11965


namespace quadratic_complete_square_l119_11945

theorem quadratic_complete_square :
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + 800 * x + 500 = (x + d)^2 + e) ∧
    (e / d = -398.75) :=
by
  use 400
  use -159500
  sorry

end quadratic_complete_square_l119_11945


namespace bonus_received_l119_11957

-- Definitions based on the conditions
def total_sales (S : ℝ) : Prop :=
  S > 10000

def commission (S : ℝ) : ℝ :=
  0.09 * S

def excess_amount (S : ℝ) : ℝ :=
  S - 10000

def additional_commission (S : ℝ) : ℝ :=
  0.03 * (S - 10000)

def total_commission (S : ℝ) : ℝ :=
  commission S + additional_commission S

-- Given the conditions
axiom total_sales_commission : ∀ S : ℝ, total_sales S → total_commission S = 1380

-- The goal is to prove the bonus
theorem bonus_received (S : ℝ) (h : total_sales S) : additional_commission S = 120 := 
by 
  sorry

end bonus_received_l119_11957


namespace determine_a_l119_11916
open Set

-- Given Condition Definitions
def U : Set ℕ := {1, 3, 5, 7}
def M (a : ℤ) : Set ℕ := {1, Int.natAbs (a - 5)} -- using ℤ for a and natAbs to get |a - 5|

-- Problem statement
theorem determine_a (a : ℤ) (hM_subset_U : M a ⊆ U) (h_complement : U \ M a = {5, 7}) : a = 2 ∨ a = 8 :=
by sorry

end determine_a_l119_11916


namespace find_largest_t_l119_11906

theorem find_largest_t (t : ℝ) : 
  (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t = 7 * t - 2 → t ≤ 1 := 
by 
  intro h
  sorry

end find_largest_t_l119_11906


namespace part_I_part_II_l119_11931

noncomputable def f (x : ℝ) := |x - 2| - |2 * x + 1|

theorem part_I :
  { x : ℝ | f x ≤ 0 } = { x : ℝ | x ≤ -3 ∨ x ≥ (1 : ℝ) / 3 } :=
by
  sorry

theorem part_II :
  ∀ x : ℝ, f x - 2 * m^2 ≤ 4 * m :=
by
  sorry

end part_I_part_II_l119_11931


namespace kevin_trip_distance_l119_11905

theorem kevin_trip_distance :
  let D := 600
  (∃ T : ℕ, D = 50 * T ∧ D = 75 * (T - 4)) := 
sorry

end kevin_trip_distance_l119_11905


namespace main_theorem_l119_11972

noncomputable def problem_statement : Prop :=
  ∀ x : ℂ, (x ≠ -2) →
  ((15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48) ↔
  (x = 12 + 2 * Real.sqrt 38 ∨ x = 12 - 2 * Real.sqrt 38 ∨
  x = -1/2 + Complex.I * Real.sqrt 95 / 2 ∨
  x = -1/2 - Complex.I * Real.sqrt 95 / 2)

-- Provide the main statement without the proof
theorem main_theorem : problem_statement := sorry

end main_theorem_l119_11972


namespace bell_rings_before_geography_l119_11903

def number_of_bell_rings : Nat :=
  let assembly_start := 1
  let assembly_end := 1
  let maths_start := 1
  let maths_end := 1
  let history_start := 1
  let history_end := 1
  let quiz_start := 1
  let quiz_end := 1
  let geography_start := 1
  assembly_start + assembly_end + maths_start + maths_end + 
  history_start + history_end + quiz_start + quiz_end + 
  geography_start

theorem bell_rings_before_geography : number_of_bell_rings = 9 := 
by
  -- Proof omitted
  sorry

end bell_rings_before_geography_l119_11903


namespace largest_integer_is_190_l119_11988

theorem largest_integer_is_190 (A B C D : ℤ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D) 
  (h4 : (A + B + C + D) / 4 = 76) 
  (h5 : A = 37) 
  (h6 : B = 38) 
  (h7 : C = 39) : 
  D = 190 := 
sorry

end largest_integer_is_190_l119_11988


namespace sum_of_k_values_l119_11959

-- Conditions
def P (x : ℝ) : ℝ := x^2 - 4 * x + 3
def Q (x k : ℝ) : ℝ := x^2 - 6 * x + k

-- Statement of the mathematical problem
theorem sum_of_k_values (k1 k2 : ℝ) (h1 : P 1 = 0) (h2 : P 3 = 0) 
  (h3 : Q 1 k1 = 0) (h4 : Q 3 k2 = 0) : k1 + k2 = 14 := 
by
  -- Here we would proceed with the proof steps corresponding to the solution
  sorry

end sum_of_k_values_l119_11959


namespace range_of_a_quadratic_root_conditions_l119_11936

theorem range_of_a_quadratic_root_conditions (a : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ < 2 ∧ (ax^2 - 2*(a+1)*x + a-1 = 0)) ↔ (0 < a ∧ a < 5)) :=
by
  sorry

end range_of_a_quadratic_root_conditions_l119_11936


namespace num_quarters_left_l119_11975

-- Define initial amounts and costs
def initial_amount : ℝ := 40
def pizza_cost : ℝ := 2.75
def soda_cost : ℝ := 1.50
def jeans_cost : ℝ := 11.50
def quarter_value : ℝ := 0.25

-- Define the total amount spent
def total_spent : ℝ := pizza_cost + soda_cost + jeans_cost

-- Define the remaining amount
def remaining_amount : ℝ := initial_amount - total_spent

-- Prove the number of quarters left
theorem num_quarters_left : remaining_amount / quarter_value = 97 :=
by
  sorry

end num_quarters_left_l119_11975


namespace team_A_win_probability_l119_11933

theorem team_A_win_probability :
  let win_prob := (1 / 3 : ℝ)
  let team_A_lead := 2
  let total_sets := 5
  let require_wins := 3
  let remaining_sets := total_sets - team_A_lead
  let prob_team_B_win_remaining := (1 - win_prob) ^ remaining_sets
  let prob_team_A_win := 1 - prob_team_B_win_remaining
  prob_team_A_win = 19 / 27 := by
    sorry

end team_A_win_probability_l119_11933


namespace find_equation_of_tangent_line_l119_11993

def is_tangent_at_point (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) := 
  ∃ x y, (x - 1)^2 + (y + 2)^2 = 1 ∧ l x₀ y₀ ∧ l x y

def equation_of_line (l : ℝ → ℝ → Prop) := 
  ∀ x y, l x y ↔ (x = 2 ∨ 12 * x - 5 * y - 9 = 0)

theorem find_equation_of_tangent_line : 
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y, l x y ↔ (x - 1)^2 + (y + 2)^2 ≠ 1 ∧ (x, y) = (2,3))
  → is_tangent_at_point l 2 3
  → equation_of_line l := 
sorry

end find_equation_of_tangent_line_l119_11993


namespace not_on_graph_ln_l119_11968

theorem not_on_graph_ln {a b : ℝ} (h : b = Real.log a) : ¬ (1 + b = Real.log (a + Real.exp 1)) :=
by
  sorry

end not_on_graph_ln_l119_11968


namespace combinations_medical_team_l119_11928

noncomputable def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_medical_team : 
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  numWaysMale * numWaysFemale = 75 :=
by
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  show numWaysMale * numWaysFemale = 75 
  sorry

end combinations_medical_team_l119_11928


namespace calculate_volume_from_measurements_l119_11960

variables (r h : ℝ) (P : ℝ × ℝ)

noncomputable def volume_truncated_cylinder (area_base : ℝ) (height_segment : ℝ) : ℝ :=
  area_base * height_segment

theorem calculate_volume_from_measurements
    (radius : ℝ) (height : ℝ)
    (area_base : ℝ := π * radius^2)
    (P : ℝ × ℝ)  -- intersection point on the axis
    (height_segment : ℝ) : 
    volume_truncated_cylinder area_base height_segment = area_base * height_segment :=
by
  -- The proof would involve demonstrating the relationship mathematically
  sorry

end calculate_volume_from_measurements_l119_11960


namespace domain_of_f_l119_11941

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f : 
  {x : ℝ | x^2 - 5*x + 6 > 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l119_11941


namespace nested_geometric_sum_l119_11981

theorem nested_geometric_sum :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))))))) = 1398100 :=
by
  sorry

end nested_geometric_sum_l119_11981


namespace smallest_of_five_consecutive_even_sum_500_l119_11922

theorem smallest_of_five_consecutive_even_sum_500 : 
  ∃ (n : Int), (n - 4, n - 2, n, n + 2, n + 4).1 = 96 ∧ 
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4) = 500) :=
by
  sorry

end smallest_of_five_consecutive_even_sum_500_l119_11922


namespace ball_probability_l119_11919

theorem ball_probability:
  let total_balls := 120
  let red_balls := 12
  let purple_balls := 18
  let yellow_balls := 15
  let desired_probability := 33 / 1190
  let probability_red := red_balls / total_balls
  let probability_purple_or_yellow := (purple_balls + yellow_balls) / (total_balls - 1)
  (probability_red * probability_purple_or_yellow = desired_probability) :=
sorry

end ball_probability_l119_11919


namespace problem_equivalent_proof_l119_11951

def sequence_row1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence_row2 (n : ℕ) : ℤ := sequence_row1 n - 1
def sequence_row3 (n : ℕ) : ℤ := (-2)^n - sequence_row2 n

theorem problem_equivalent_proof :
  let a := sequence_row1 7
  let b := sequence_row2 7
  let c := sequence_row3 7
  a - b + c = -254 :=
by
  sorry

end problem_equivalent_proof_l119_11951


namespace max_sum_of_digits_of_S_l119_11934

def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def distinctDigits (n : ℕ) : Prop :=
  let digits := (n.digits 10).toFinset
  digits.card = (n.digits 10).length

def digitsRange (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 9

theorem max_sum_of_digits_of_S : ∃ a b S, 
  isThreeDigit a ∧ 
  isThreeDigit b ∧ 
  distinctDigits a ∧ 
  distinctDigits b ∧ 
  digitsRange a ∧ 
  digitsRange b ∧ 
  isThreeDigit S ∧ 
  S = a + b ∧ 
  (S.digits 10).sum = 12 :=
sorry

end max_sum_of_digits_of_S_l119_11934


namespace basketball_probability_l119_11901

-- Define the probabilities of A and B making a shot
def prob_A : ℝ := 0.4
def prob_B : ℝ := 0.6

-- Define the probability that both miss their shots in one round
def prob_miss_one_round : ℝ := (1 - prob_A) * (1 - prob_B)

-- Define the probability that A takes k shots to make a basket
noncomputable def P_xi (k : ℕ) : ℝ := (prob_miss_one_round)^(k-1) * prob_A

-- State the theorem
theorem basketball_probability (k : ℕ) : 
  P_xi k = 0.24^(k-1) * 0.4 :=
by
  unfold P_xi
  unfold prob_miss_one_round
  sorry

end basketball_probability_l119_11901


namespace yearly_exports_calculation_l119_11962

variable (Y : Type) 
variable (fruit_exports_total yearly_exports : ℝ)
variable (orange_exports : ℝ := 4.25 * 10^6)
variable (fruit_exports_percent : ℝ := 0.20)
variable (orange_exports_fraction : ℝ := 1/6)

-- The main statement to prove
theorem yearly_exports_calculation
  (h1 : yearly_exports * fruit_exports_percent = fruit_exports_total)
  (h2 : fruit_exports_total * orange_exports_fraction = orange_exports) :
  yearly_exports = 127.5 * 10^6 :=
by
  -- Proof (omitted)
  sorry

end yearly_exports_calculation_l119_11962


namespace richard_more_pins_than_patrick_l119_11989

theorem richard_more_pins_than_patrick :
  ∀ (R P R2 P2 : ℕ), 
    P = 70 → 
    R > P →
    P2 = 2 * R →
    R2 = P2 - 3 → 
    (R + R2) = (P + P2) + 12 → 
    R = 70 + 15 := 
by 
  intros R P R2 P2 hP hRp hP2 hR2 hTotal
  sorry

end richard_more_pins_than_patrick_l119_11989


namespace find_f_2000_l119_11911

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : ∀ (x y : ℝ), f (x + y) = f (x * y)
axiom f_property2 : f (-1/2) = -1/2

theorem find_f_2000 : f 2000 = -1/2 := 
sorry

end find_f_2000_l119_11911


namespace paint_grid_l119_11904

theorem paint_grid (paint : Fin 3 × Fin 3 → Bool) (no_adjacent : ∀ i j, (paint (i, j) = true) → (paint (i+1, j) = false) ∧ (paint (i-1, j) = false) ∧ (paint (i, j+1) = false) ∧ (paint (i, j-1) = false)) : 
  ∃! (count : ℕ), count = 8 :=
sorry

end paint_grid_l119_11904


namespace quadratic_max_value_4_at_2_l119_11925

theorem quadratic_max_value_4_at_2 (a b c : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, x ≠ 2 → (a * 2^2 + b * 2 + c) = 4)
  (h2 : a * 0^2 + b * 0 + c = -20)
  (h3 : a * 5^2 + b * 5 + c = m) :
  m = -50 :=
sorry

end quadratic_max_value_4_at_2_l119_11925


namespace find_initial_number_of_girls_l119_11948

theorem find_initial_number_of_girls (b g : ℕ) : 
  (b = 3 * (g - 12)) ∧ (4 * (b - 36) = g - 12) → g = 25 :=
by
  intros h
  sorry

end find_initial_number_of_girls_l119_11948


namespace geometric_sequence_a5_l119_11915

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 2 * a 8 = 4) : a 5 = 2 :=
sorry

end geometric_sequence_a5_l119_11915


namespace sum_of_remainders_l119_11967

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end sum_of_remainders_l119_11967


namespace trapezoid_LM_sqrt2_l119_11969

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l119_11969


namespace Peter_drew_more_l119_11949

theorem Peter_drew_more :
  ∃ (P : ℕ), 5 + P + (P + 20) = 41 ∧ (P - 5 = 3) :=
sorry

end Peter_drew_more_l119_11949


namespace work_days_l119_11932

/-- A needs 20 days to complete the work alone. B needs 10 days to complete the work alone.
    The total work must be completed in 12 days. We need to find how many days B must work 
    before A continues, such that the total work equals the full task. -/
theorem work_days (x : ℝ) (h0 : 0 ≤ x ∧ x ≤ 12) (h1 : 1 / 10 * x + 1 / 20 * (12 - x) = 1) : x = 8 := by
  sorry

end work_days_l119_11932


namespace solve_fraction_l119_11943

theorem solve_fraction (x : ℝ) (h1 : x + 2 = 0) (h2 : 2 * x - 4 ≠ 0) : x = -2 := 
by 
  sorry

end solve_fraction_l119_11943


namespace equation_has_no_solution_l119_11973

theorem equation_has_no_solution (k : ℝ) : ¬ (∃ x : ℝ , (x ≠ 3 ∧ x ≠ 4) ∧ (x - 1) / (x - 3) = (x - k) / (x - 4)) ↔ k = 2 :=
by
  sorry

end equation_has_no_solution_l119_11973


namespace problem_solution_l119_11983

theorem problem_solution:
  2019 ^ Real.log (Real.log 2019) - Real.log 2019 ^ Real.log 2019 = 0 :=
by
  sorry

end problem_solution_l119_11983


namespace interest_rate_compound_interest_l119_11937

theorem interest_rate_compound_interest :
  ∀ (P A : ℝ) (t n : ℕ), 
  P = 156.25 → A = 169 → t = 2 → n = 1 → 
  (∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r * 100 = 4) :=
by
  intros P A t n hP hA ht hn
  use 0.04
  rw [hP, hA, ht, hn]
  sorry

end interest_rate_compound_interest_l119_11937


namespace probability_neither_snow_nor_rain_in_5_days_l119_11913

def probability_no_snow (p_snow : ℚ) : ℚ := 1 - p_snow
def probability_no_rain (p_rain : ℚ) : ℚ := 1 - p_rain
def probability_no_snow_and_no_rain (p_no_snow p_no_rain : ℚ) : ℚ := p_no_snow * p_no_rain
def probability_no_snow_and_no_rain_5_days (p : ℚ) : ℚ := p ^ 5

theorem probability_neither_snow_nor_rain_in_5_days
    (p_snow : ℚ) (p_rain : ℚ)
    (h1 : p_snow = 2/3) (h2 : p_rain = 1/2) :
    probability_no_snow_and_no_rain_5_days (probability_no_snow_and_no_rain (probability_no_snow p_snow) (probability_no_rain p_rain)) = 1/7776 := by
  sorry

end probability_neither_snow_nor_rain_in_5_days_l119_11913


namespace equation_of_plane_l119_11979

-- Definitions based on conditions
def line_equation (A B C x y : ℝ) : Prop :=
  A * x + B * y + C = 0

def A_B_nonzero (A B : ℝ) : Prop :=
  A ^ 2 + B ^ 2 ≠ 0

-- Statement for the problem
noncomputable def plane_equation (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem equation_of_plane (A B C D : ℝ) :
  (A ^ 2 + B ^ 2 + C ^ 2 ≠ 0) → (∀ x y z : ℝ, plane_equation A B C D x y z) :=
by
  sorry

end equation_of_plane_l119_11979


namespace road_renovation_l119_11961

theorem road_renovation (x : ℕ) (h : 200 / (x + 20) = 150 / x) : 
  x = 60 ∧ (x + 20) = 80 :=
by {
  sorry
}

end road_renovation_l119_11961


namespace inequality_solution_l119_11966

theorem inequality_solution (x : ℝ) :
  (-4 ≤ x ∧ x < -3 / 2) ↔ (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) :=
by
  sorry

end inequality_solution_l119_11966


namespace problem8x_eq_5_200timesreciprocal_l119_11991

theorem problem8x_eq_5_200timesreciprocal (x : ℚ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := 
by 
  sorry

end problem8x_eq_5_200timesreciprocal_l119_11991


namespace correct_calculation_result_l119_11987

theorem correct_calculation_result (x : ℝ) (h : x / 12 = 8) : 12 * x = 1152 :=
sorry

end correct_calculation_result_l119_11987


namespace branches_number_l119_11994

-- Conditions (converted into Lean definitions)
def total_leaves : ℕ := 12690
def twigs_per_branch : ℕ := 90
def leaves_per_twig_percentage_4 : ℝ := 0.3
def leaves_per_twig_percentage_5 : ℝ := 0.7
def leaves_per_twig_4 : ℕ := 4
def leaves_per_twig_5 : ℕ := 5

-- The goal
theorem branches_number (B : ℕ) 
  (h1 : twigs_per_branch = 90) 
  (h2 : leaves_per_twig_percentage_4 = 0.3) 
  (h3 : leaves_per_twig_percentage_5 = 0.7) 
  (h4 : leaves_per_twig_4 = 4) 
  (h5 : leaves_per_twig_5 = 5) 
  (h6 : total_leaves = 12690) :
  B = 30 := 
sorry

end branches_number_l119_11994
