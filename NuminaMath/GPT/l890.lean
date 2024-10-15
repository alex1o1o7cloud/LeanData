import Mathlib

namespace NUMINAMATH_GPT_matrix_vector_computation_l890_89016

-- Setup vectors and their corresponding matrix multiplication results
variables {R : Type*} [Field R]
variables {M : Matrix (Fin 2) (Fin 2) R} {u z : Fin 2 → R}

-- Conditions given in (a)
def condition1 : M.mulVec u = ![3, -4] :=
  sorry

def condition2 : M.mulVec z = ![-1, 6] :=
  sorry

-- Statement equivalent to the proof problem given in (c)
theorem matrix_vector_computation :
  M.mulVec (3 • u - 2 • z) = ![11, -24] :=
by
  -- Use the conditions to prove the theorem
  sorry

end NUMINAMATH_GPT_matrix_vector_computation_l890_89016


namespace NUMINAMATH_GPT_max_score_top_three_teams_l890_89051

theorem max_score_top_three_teams : 
  ∀ (teams : Finset String) (points : String → ℕ), 
    teams.card = 6 →
    (∀ team, team ∈ teams → (points team = 0 ∨ points team = 1 ∨ points team = 3)) →
    ∃ top_teams : Finset String, top_teams.card = 3 ∧ 
    (∀ team, team ∈ top_teams → points team = 24) := 
by sorry

end NUMINAMATH_GPT_max_score_top_three_teams_l890_89051


namespace NUMINAMATH_GPT_total_time_spent_l890_89077

-- Definition of the problem conditions
def warm_up_time : ℕ := 10
def additional_puzzles : ℕ := 2
def multiplier : ℕ := 3

-- Statement to prove the total time spent solving puzzles
theorem total_time_spent : warm_up_time + (additional_puzzles * (multiplier * warm_up_time)) = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_time_spent_l890_89077


namespace NUMINAMATH_GPT_value_of_y_l890_89039

theorem value_of_y (y m : ℕ) (h1 : ((1 ^ m) / (y ^ m)) * (1 ^ 16 / 4 ^ 16) = 1 / (2 * 10 ^ 31)) (h2 : m = 31) : 
  y = 5 := 
sorry

end NUMINAMATH_GPT_value_of_y_l890_89039


namespace NUMINAMATH_GPT_andy_incorrect_l890_89072

theorem andy_incorrect (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 8) : a = 14 :=
by
  sorry

end NUMINAMATH_GPT_andy_incorrect_l890_89072


namespace NUMINAMATH_GPT_max_volume_prism_l890_89070

theorem max_volume_prism (V : ℝ) (h l w : ℝ) 
  (h_eq_2h : l = 2 * h ∧ w = 2 * h) 
  (surface_area_eq : l * h + w * h + l * w = 36) : 
  V = 27 * Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_max_volume_prism_l890_89070


namespace NUMINAMATH_GPT_find_fraction_l890_89064

def f (x : ℤ) : ℤ := 3 * x + 4
def g (x : ℤ) : ℤ := 4 * x - 3

theorem find_fraction :
  (f (g (f 2)):ℚ) / (g (f (g 2)):ℚ) = 115 / 73 := by
  sorry

end NUMINAMATH_GPT_find_fraction_l890_89064


namespace NUMINAMATH_GPT_problem_statement_l890_89004

theorem problem_statement (x y z : ℝ) (h : (x - z)^2 - 4 * (x - y) * (y - z) = 0) : x + z - 2 * y = 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l890_89004


namespace NUMINAMATH_GPT_hour_hand_degrees_noon_to_2_30_l890_89052

def degrees_moved (hours: ℕ) : ℝ := (hours * 30)

theorem hour_hand_degrees_noon_to_2_30 :
  degrees_moved 2 + degrees_moved 1 / 2 = 75 :=
sorry

end NUMINAMATH_GPT_hour_hand_degrees_noon_to_2_30_l890_89052


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_min_value_l890_89091

-- For Problem (1)
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem problem1_solution_set (x : ℝ) (h : f x 1 1 ≤ 4) : 
  -2 ≤ x ∧ x ≤ 2 :=
sorry

-- For Problem (2)
theorem problem2_min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∀ x : ℝ, f x a b ≥ 2) : 
  (1 / a) + (2 / b) = 3 :=
sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_min_value_l890_89091


namespace NUMINAMATH_GPT_problem_1_l890_89022

noncomputable def derivative_y (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) : ℝ :=
  (2 * a) / (3 * (1 - y^2))

theorem problem_1 (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) :
  derivative_y a x y h = (2 * a) / (3 * (1 - y^2)) :=
sorry

end NUMINAMATH_GPT_problem_1_l890_89022


namespace NUMINAMATH_GPT_ratio_of_areas_l890_89000

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C M : V}

-- Define the collinearity condition point M in the triangle plane with respect to vectors AB and AC
def point_condition (A B C M : V) : Prop :=
  5 • (M - A) = (B - A) + 3 • (C - A)

-- Define an area ratio function
def area_ratio_triangles (A B C M : V) [AddCommGroup V] [Module ℝ V] : ℝ :=
  sorry  -- Implementation of area ratio comparison, abstracted out for the given problem statement

-- The theorem to prove
theorem ratio_of_areas (hM : point_condition A B C M) : area_ratio_triangles A B C M = 3 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_l890_89000


namespace NUMINAMATH_GPT_total_chickens_l890_89083

theorem total_chickens (ducks geese : ℕ) (hens roosters chickens: ℕ) :
  ducks = 45 → geese = 28 →
  hens = ducks - 13 → roosters = geese + 9 →
  chickens = hens + roosters →
  chickens = 69 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_chickens_l890_89083


namespace NUMINAMATH_GPT_minimum_experiments_fractional_method_l890_89038

/--
A pharmaceutical company needs to optimize the cultivation temperature for a certain medicinal liquid through bioassay.
The experimental range is set from 29℃ to 63℃, with an accuracy requirement of ±1℃.
Prove that the minimum number of experiments required to ensure the best cultivation temperature is found using the fractional method is 7.
-/
theorem minimum_experiments_fractional_method
  (range_start : ℕ)
  (range_end : ℕ)
  (accuracy : ℕ)
  (fractional_method : ∀ (range_start range_end accuracy: ℕ), ℕ) :
  range_start = 29 → range_end = 63 → accuracy = 1 → fractional_method range_start range_end accuracy = 7 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_minimum_experiments_fractional_method_l890_89038


namespace NUMINAMATH_GPT_cos_675_eq_sqrt2_div_2_l890_89035

theorem cos_675_eq_sqrt2_div_2 : Real.cos (675 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_cos_675_eq_sqrt2_div_2_l890_89035


namespace NUMINAMATH_GPT_new_students_admitted_l890_89021

-- Definitions of the conditions
def original_students := 35
def increase_in_expenses := 42
def decrease_in_average_expense := 1
def original_expenditure := 420

-- Main statement: proving the number of new students admitted
theorem new_students_admitted : ∃ x : ℕ, 
  (original_expenditure + increase_in_expenses = 11 * (original_students + x)) ∧ 
  (x = 7) := 
sorry

end NUMINAMATH_GPT_new_students_admitted_l890_89021


namespace NUMINAMATH_GPT_zoo_gorilla_percentage_l890_89017

theorem zoo_gorilla_percentage :
  ∀ (visitors_per_hour : ℕ) (open_hours : ℕ) (gorilla_visitors : ℕ) (total_visitors : ℕ)
    (percentage : ℕ),
  visitors_per_hour = 50 → open_hours = 8 → gorilla_visitors = 320 →
  total_visitors = visitors_per_hour * open_hours →
  percentage = (gorilla_visitors * 100) / total_visitors →
  percentage = 80 :=
by
  intros visitors_per_hour open_hours gorilla_visitors total_visitors percentage
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3, h4] at h5
  exact h5

end NUMINAMATH_GPT_zoo_gorilla_percentage_l890_89017


namespace NUMINAMATH_GPT_sector_central_angle_l890_89030

noncomputable def sector_angle (R L : ℝ) : ℝ := L / R

theorem sector_central_angle :
  ∃ R L : ℝ, 
    (R > 0) ∧ 
    (L > 0) ∧ 
    (1 / 2 * L * R = 5) ∧ 
    (2 * R + L = 9) ∧ 
    (sector_angle R L = 8 / 5 ∨ sector_angle R L = 5 / 2) :=
sorry

end NUMINAMATH_GPT_sector_central_angle_l890_89030


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l890_89037

theorem radius_of_inscribed_circle (r1 r2 : ℝ) (AC BC AB : ℝ) 
  (h1 : AC = 2 * r1)
  (h2 : BC = 2 * r2)
  (h3 : AB = 2 * Real.sqrt (r1^2 + r2^2)) : 
  (r1 + r2 - Real.sqrt (r1^2 + r2^2)) = ((2 * r1 + 2 * r2 - 2 * Real.sqrt (r1^2 + r2^2)) / 2) := 
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l890_89037


namespace NUMINAMATH_GPT_total_treat_value_is_339100_l890_89003

def hotel_cost (cost_per_night : ℕ) (nights : ℕ) (discount : ℕ) : ℕ :=
  let total_cost := cost_per_night * nights
  total_cost - (total_cost * discount / 100)

def car_cost (base_price : ℕ) (tax : ℕ) : ℕ :=
  base_price + (base_price * tax / 100)

def house_cost (car_base_price : ℕ) (multiplier : ℕ) (property_tax : ℕ) : ℕ :=
  let house_value := car_base_price * multiplier
  house_value + (house_value * property_tax / 100)

def yacht_cost (hotel_value : ℕ) (car_value : ℕ) (multiplier : ℕ) (discount : ℕ) : ℕ :=
  let combined_value := hotel_value + car_value
  let yacht_value := combined_value * multiplier
  yacht_value - (yacht_value * discount / 100)

def gold_coins_cost (yacht_value : ℕ) (multiplier : ℕ) (tax : ℕ) : ℕ :=
  let gold_value := yacht_value * multiplier
  gold_value + (gold_value * tax / 100)

theorem total_treat_value_is_339100 :
  let hotel_value := hotel_cost 4000 2 5
  let car_value := car_cost 30000 10
  let house_value := house_cost 30000 4 2
  let yacht_value := yacht_cost 8000 30000 2 7
  let gold_coins_value := gold_coins_cost 76000 3 3
  hotel_value + car_value + house_value + yacht_value + gold_coins_value = 339100 :=
by sorry

end NUMINAMATH_GPT_total_treat_value_is_339100_l890_89003


namespace NUMINAMATH_GPT_combinations_x_eq_2_or_8_l890_89087

theorem combinations_x_eq_2_or_8 (x : ℕ) (h_pos : 0 < x) (h_comb : Nat.choose 10 x = Nat.choose 10 2) : x = 2 ∨ x = 8 :=
sorry

end NUMINAMATH_GPT_combinations_x_eq_2_or_8_l890_89087


namespace NUMINAMATH_GPT_total_heartbeats_during_race_l890_89059

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_total_heartbeats_during_race_l890_89059


namespace NUMINAMATH_GPT_grandmother_total_payment_l890_89061

theorem grandmother_total_payment
  (senior_discount : Real := 0.30)
  (children_discount : Real := 0.40)
  (num_seniors : Nat := 2)
  (num_children : Nat := 2)
  (num_regular : Nat := 2)
  (senior_ticket_price : Real := 7.50)
  (regular_ticket_price : Real := senior_ticket_price / (1 - senior_discount))
  (children_ticket_price : Real := regular_ticket_price * (1 - children_discount))
  : (num_seniors * senior_ticket_price + num_regular * regular_ticket_price + num_children * children_ticket_price) = 49.27 := 
by
  sorry

end NUMINAMATH_GPT_grandmother_total_payment_l890_89061


namespace NUMINAMATH_GPT_charity_donation_ratio_l890_89075

theorem charity_donation_ratio :
  let total_winnings := 114
  let hot_dog_cost := 2
  let remaining_amount := 55
  let donation_amount := 114 - (remaining_amount + hot_dog_cost)
  donation_amount = 55 :=
by
  sorry

end NUMINAMATH_GPT_charity_donation_ratio_l890_89075


namespace NUMINAMATH_GPT_total_people_selected_l890_89024

-- Define the number of residents in each age group
def residents_21_to_35 : Nat := 840
def residents_36_to_50 : Nat := 700
def residents_51_to_65 : Nat := 560

-- Define the number of people selected from the 36 to 50 age group
def selected_36_to_50 : Nat := 100

-- Define the total number of residents
def total_residents : Nat := residents_21_to_35 + residents_36_to_50 + residents_51_to_65

-- Theorem: Prove that the total number of people selected in this survey is 300
theorem total_people_selected : (100 : ℕ) / (700 : ℕ) * (residents_21_to_35 + residents_36_to_50 + residents_51_to_65) = 300 :=
  by 
    sorry

end NUMINAMATH_GPT_total_people_selected_l890_89024


namespace NUMINAMATH_GPT_find_PA_values_l890_89080

theorem find_PA_values :
  ∃ P A : ℕ, 10 ≤ P * 10 + A ∧ P * 10 + A < 100 ∧
            (P * 10 + A) ^ 2 / 1000 = P ∧ (P * 10 + A) ^ 2 % 10 = A ∧
            ((P = 9 ∧ A = 5) ∨ (P = 9 ∧ A = 6)) := by
  sorry

end NUMINAMATH_GPT_find_PA_values_l890_89080


namespace NUMINAMATH_GPT_factor_polynomial_l890_89047

theorem factor_polynomial (x : ℝ) :
  (x^3 - 12 * x + 16) = (x + 4) * ((x - 2)^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l890_89047


namespace NUMINAMATH_GPT_cookies_count_l890_89034

theorem cookies_count :
  ∀ (Tom Lucy Millie Mike Frank : ℕ), 
  (Tom = 16) →
  (Lucy = Nat.sqrt Tom) →
  (Millie = 2 * Lucy) →
  (Mike = 3 * Millie) →
  (Frank = Mike / 2 - 3) →
  Frank = 9 :=
by
  intros Tom Lucy Millie Mike Frank hTom hLucy hMillie hMike hFrank
  have h1 : Tom = 16 := hTom
  have h2 : Lucy = Nat.sqrt Tom := hLucy
  have h3 : Millie = 2 * Lucy := hMillie
  have h4 : Mike = 3 * Millie := hMike
  have h5 : Frank = Mike / 2 - 3 := hFrank
  sorry

end NUMINAMATH_GPT_cookies_count_l890_89034


namespace NUMINAMATH_GPT_zinc_weight_in_mixture_l890_89010

theorem zinc_weight_in_mixture (total_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) (total_parts : ℝ) (fraction_zinc : ℝ) (weight_zinc : ℝ) :
  zinc_ratio = 9 ∧ copper_ratio = 11 ∧ total_weight = 70 ∧ total_parts = zinc_ratio + copper_ratio ∧
  fraction_zinc = zinc_ratio / total_parts ∧ weight_zinc = fraction_zinc * total_weight →
  weight_zinc = 31.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_zinc_weight_in_mixture_l890_89010


namespace NUMINAMATH_GPT_chessboard_overlap_area_l890_89089

theorem chessboard_overlap_area :
  let n := 8
  let cell_area := 1
  let side_length := 8
  let overlap_area := 32 * (Real.sqrt 2 - 1)
  (∃ black_overlap_area : ℝ, black_overlap_area = overlap_area) :=
by
  sorry

end NUMINAMATH_GPT_chessboard_overlap_area_l890_89089


namespace NUMINAMATH_GPT_river_depth_is_correct_l890_89033

noncomputable def depth_of_river (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_min : ℝ) : ℝ :=
  let flow_rate_mpm := (flow_rate_kmph * 1000) / 60
  let cross_sectional_area := volume_per_min / flow_rate_mpm
  cross_sectional_area / width

theorem river_depth_is_correct :
  depth_of_river 65 6 26000 = 4 :=
by
  -- Steps to compute depth (converted from solution)
  sorry

end NUMINAMATH_GPT_river_depth_is_correct_l890_89033


namespace NUMINAMATH_GPT_find_salary_l890_89015

theorem find_salary (S : ℤ) (food house_rent clothes left : ℤ) 
  (h_food : food = S / 5) 
  (h_house_rent : house_rent = S / 10) 
  (h_clothes : clothes = 3 * S / 5) 
  (h_left : left = 18000) 
  (h_spent : food + house_rent + clothes + left = S) : 
  S = 180000 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_salary_l890_89015


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l890_89094

-- Definitions
variable (a b c : ℝ)  -- semi-major axis, semi-minor axis, and distance from center to a focus
variable (h_c_eq_b : c = b)  -- given condition focal length equals length of minor axis
variable (h_a_eq_sqrt_sum : a = Real.sqrt (c^2 + b^2))  -- relationship in ellipse

-- Question: Prove the eccentricity of the ellipse e = √2 / 2
theorem eccentricity_of_ellipse : (c = b) → (a = Real.sqrt (c^2 + b^2)) → (c / a = Real.sqrt 2 / 2) :=
by
  intros h_c_eq_b h_a_eq_sqrt_sum
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l890_89094


namespace NUMINAMATH_GPT_smallest_blocks_required_l890_89031

theorem smallest_blocks_required (L H : ℕ) (block_height block_long block_short : ℕ) 
  (vert_joins_staggered : Prop) (consistent_end_finish : Prop) : 
  L = 120 → H = 10 → block_height = 1 → block_long = 3 → block_short = 1 → 
  (vert_joins_staggered) → (consistent_end_finish) → 
  ∃ n, n = 415 :=
by
  sorry

end NUMINAMATH_GPT_smallest_blocks_required_l890_89031


namespace NUMINAMATH_GPT_isosceles_trapezoid_side_length_is_five_l890_89029

noncomputable def isosceles_trapezoid_side_length (b1 b2 area : ℝ) : ℝ :=
  let h := 2 * area / (b1 + b2)
  let base_diff_half := (b2 - b1) / 2
  Real.sqrt (h^2 + base_diff_half^2)
  
theorem isosceles_trapezoid_side_length_is_five :
  isosceles_trapezoid_side_length 6 12 36 = 5 := by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_side_length_is_five_l890_89029


namespace NUMINAMATH_GPT_part_one_part_two_l890_89088

-- First part: Prove that \( (1)(-1)^{2017}+(\frac{1}{2})^{-2}+(3.14-\pi)^{0} = 4\)
theorem part_one : (1 * (-1:ℤ)^2017 + (1/2)^(-2:ℤ) + (3.14 - Real.pi)^0 : ℝ) = 4 := 
  sorry

-- Second part: Prove that \( ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 \)
theorem part_two (x : ℝ) : ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 := 
  sorry

end NUMINAMATH_GPT_part_one_part_two_l890_89088


namespace NUMINAMATH_GPT_ratio_of_u_to_v_l890_89014

theorem ratio_of_u_to_v (b : ℚ) (hb : b ≠ 0) (u v : ℚ)
  (hu : u = -b / 8) (hv : v = -b / 12) :
  u / v = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_u_to_v_l890_89014


namespace NUMINAMATH_GPT_inequality_min_value_l890_89081

theorem inequality_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, (x + 2 * y) * (2 / x + 1 / y) ≥ m ∧ m ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_min_value_l890_89081


namespace NUMINAMATH_GPT_sum_of_digits_least_N_l890_89002

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 * N : ℚ) / 5⌉) / (N + 1)

theorem sum_of_digits_least_N (k : ℕ) (h_k : k = 2) (h1 : ∀ N, P N k < 8 / 10 ) :
  ∃ N : ℕ, (N % 10) + (N / 10) = 1 ∧ (P N k < 8 / 10) ∧ (∀ M : ℕ, M < N → P M k ≥ 8 / 10) := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_least_N_l890_89002


namespace NUMINAMATH_GPT_magician_earnings_l890_89011

theorem magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (decks_remaining : ℕ) (money_earned : ℕ) : 
    price_per_deck = 7 →
    initial_decks = 16 →
    decks_remaining = 8 →
    money_earned = (initial_decks - decks_remaining) * price_per_deck →
    money_earned = 56 :=
by
  intros hp hi hd he
  rw [hp, hi, hd] at he
  exact he

end NUMINAMATH_GPT_magician_earnings_l890_89011


namespace NUMINAMATH_GPT_eval_complex_fraction_expr_l890_89063

def complex_fraction_expr : ℚ :=
  2 + (3 / (4 + (5 / (6 + (7 / 8)))))

theorem eval_complex_fraction_expr : complex_fraction_expr = 137 / 52 :=
by
  -- we skip the actual proof but ensure it can build successfully.
  sorry

end NUMINAMATH_GPT_eval_complex_fraction_expr_l890_89063


namespace NUMINAMATH_GPT_paulina_convertibles_l890_89056

-- Definitions for conditions
def total_cars : ℕ := 125
def percentage_regular_cars : ℚ := 64 / 100
def percentage_trucks : ℚ := 8 / 100
def percentage_convertibles : ℚ := 1 - (percentage_regular_cars + percentage_trucks)

-- Theorem to prove the number of convertibles
theorem paulina_convertibles : (percentage_convertibles * total_cars) = 35 := by
  sorry

end NUMINAMATH_GPT_paulina_convertibles_l890_89056


namespace NUMINAMATH_GPT_diameter_of_inscribed_circle_l890_89055

theorem diameter_of_inscribed_circle (a b c r : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_radius : r = (a + b - c) / 2) : 
  2 * r = a + b - c :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_inscribed_circle_l890_89055


namespace NUMINAMATH_GPT_probability_53_sundays_in_leap_year_l890_89099

-- Define the conditions
def num_days_in_leap_year : ℕ := 366
def num_weeks_in_leap_year : ℕ := 52
def extra_days_in_leap_year : ℕ := 2
def num_combinations : ℕ := 7
def num_sunday_combinations : ℕ := 2

-- Define the problem statement
theorem probability_53_sundays_in_leap_year (hdays : num_days_in_leap_year = 52 * 7 + extra_days_in_leap_year) :
  (num_sunday_combinations / num_combinations : ℚ) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_53_sundays_in_leap_year_l890_89099


namespace NUMINAMATH_GPT_thomas_percentage_l890_89045

/-- 
Prove that if Emmanuel gets 100 jelly beans out of a total of 200 jelly beans, and 
Barry and Emmanuel share the remainder in a 4:5 ratio, then Thomas takes 10% 
of the jelly beans.
-/
theorem thomas_percentage (total_jelly_beans : ℕ) (emmanuel_jelly_beans : ℕ)
  (barry_ratio : ℕ) (emmanuel_ratio : ℕ) (thomas_percentage : ℕ) :
  total_jelly_beans = 200 → emmanuel_jelly_beans = 100 → barry_ratio = 4 → emmanuel_ratio = 5 →
  thomas_percentage = 10 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_thomas_percentage_l890_89045


namespace NUMINAMATH_GPT_percentage_increase_in_yield_after_every_harvest_is_20_l890_89079

theorem percentage_increase_in_yield_after_every_harvest_is_20
  (P : ℝ)
  (h1 : ∀ n : ℕ, n = 1 → 20 * n = 20)
  (h2 : 20 + 20 * (1 + P / 100) = 44) :
  P = 20 := 
sorry

end NUMINAMATH_GPT_percentage_increase_in_yield_after_every_harvest_is_20_l890_89079


namespace NUMINAMATH_GPT_area_of_enclosing_square_is_100_l890_89009

noncomputable def radius : ℝ := 5

noncomputable def diameter_of_circle (r : ℝ) : ℝ := 2 * r

noncomputable def side_length_of_square (d : ℝ) : ℝ := d

noncomputable def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_enclosing_square_is_100 :
  area_of_square (side_length_of_square (diameter_of_circle radius)) = 100 :=
by
  sorry

end NUMINAMATH_GPT_area_of_enclosing_square_is_100_l890_89009


namespace NUMINAMATH_GPT_trapezoid_segment_ratio_l890_89020

theorem trapezoid_segment_ratio (s l : ℝ) (h₁ : 3 * s + l = 1) (h₂ : 2 * l + 6 * s = 2) :
  l = 2 * s :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_segment_ratio_l890_89020


namespace NUMINAMATH_GPT_find_x_for_y_equals_six_l890_89019

variable (x y k : ℚ)

-- Conditions
def varies_inversely_as_square := x = k / y^2
def initial_condition := (y = 3 ∧ x = 1)

-- Problem Statement
theorem find_x_for_y_equals_six (h₁ : varies_inversely_as_square x y k) (h₂ : initial_condition x y) :
  ∃ k, (k = 9 ∧ x = k / 6^2 ∧ x = 1 / 4) :=
sorry

end NUMINAMATH_GPT_find_x_for_y_equals_six_l890_89019


namespace NUMINAMATH_GPT_line_equation_perpendicular_l890_89023

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem line_equation_perpendicular (c : ℝ) :
  (∃ k : ℝ, x - 2 * y + k = 0) ∧ is_perpendicular 2 1 1 (-2) → x - 2 * y - 3 = 0 := by
  sorry

end NUMINAMATH_GPT_line_equation_perpendicular_l890_89023


namespace NUMINAMATH_GPT_ratio_of_b_plus_e_over_c_plus_f_l890_89049

theorem ratio_of_b_plus_e_over_c_plus_f 
  (a b c d e f : ℝ)
  (h1 : a + b = 2 * a + c)
  (h2 : a - 2 * b = 4 * c)
  (h3 : a + b + c = 21)
  (h4 : d + e = 3 * d + f)
  (h5 : d - 2 * e = 5 * f)
  (h6 : d + e + f = 32) :
  (b + e) / (c + f) = -3.99 :=
sorry

end NUMINAMATH_GPT_ratio_of_b_plus_e_over_c_plus_f_l890_89049


namespace NUMINAMATH_GPT_calculate_brick_height_cm_l890_89085

noncomputable def wall_length_cm : ℕ := 1000  -- 10 m converted to cm
noncomputable def wall_width_cm : ℕ := 800   -- 8 m converted to cm
noncomputable def wall_height_cm : ℕ := 2450 -- 24.5 m converted to cm

noncomputable def wall_volume_cm3 : ℕ := wall_length_cm * wall_width_cm * wall_height_cm

noncomputable def brick_length_cm : ℕ := 20
noncomputable def brick_width_cm : ℕ := 10
noncomputable def number_of_bricks : ℕ := 12250

noncomputable def brick_area_cm2 : ℕ := brick_length_cm * brick_width_cm

theorem calculate_brick_height_cm (h : ℕ) : brick_area_cm2 * h * number_of_bricks = wall_volume_cm3 → 
  h = wall_volume_cm3 / (brick_area_cm2 * number_of_bricks) := by
  sorry

end NUMINAMATH_GPT_calculate_brick_height_cm_l890_89085


namespace NUMINAMATH_GPT_area_of_trapezoid_RSQT_l890_89048
-- Import the required library

-- Declare the geometrical setup and given areas
variables (PQ PR : ℝ)
variable (PQR_area : ℝ)
variable (small_triangle_area : ℝ)
variable (num_small_triangles : ℕ)
variable (inner_triangle_area : ℝ)
variable (trapezoid_RSQT_area : ℝ)

-- Define the conditions from part a)
def isosceles_triangle : Prop := PQ = PR
def triangle_PQR_area_given : Prop := PQR_area = 75
def small_triangle_area_given : Prop := small_triangle_area = 3
def num_small_triangles_given : Prop := num_small_triangles = 9
def inner_triangle_area_given : Prop := inner_triangle_area = 5 * small_triangle_area

-- Define the target statement (question == answer)
theorem area_of_trapezoid_RSQT :
  isosceles_triangle PQ PR ∧
  triangle_PQR_area_given PQR_area ∧
  small_triangle_area_given small_triangle_area ∧
  num_small_triangles_given num_small_triangles ∧
  inner_triangle_area_given small_triangle_area inner_triangle_area → 
  trapezoid_RSQT_area = 60 :=
sorry

end NUMINAMATH_GPT_area_of_trapezoid_RSQT_l890_89048


namespace NUMINAMATH_GPT_find_remainder_l890_89086

theorem find_remainder (P Q R D D' Q' R' C : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  (P % (D * D')) = (D * R' + R + C) :=
sorry

end NUMINAMATH_GPT_find_remainder_l890_89086


namespace NUMINAMATH_GPT_equal_playing_time_for_each_player_l890_89069

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end NUMINAMATH_GPT_equal_playing_time_for_each_player_l890_89069


namespace NUMINAMATH_GPT_fraction_power_computation_l890_89084

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end NUMINAMATH_GPT_fraction_power_computation_l890_89084


namespace NUMINAMATH_GPT_cookies_per_batch_l890_89054

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end NUMINAMATH_GPT_cookies_per_batch_l890_89054


namespace NUMINAMATH_GPT_possible_values_of_b_l890_89082

theorem possible_values_of_b (r s : ℝ) (t t' : ℝ)
  (hp : ∀ x, x^3 + a * x + b = 0 → (x = r ∨ x = s ∨ x = t))
  (hq : ∀ x, x^3 + a * x + b + 240 = 0 → (x = r + 4 ∨ x = s - 3 ∨ x = t'))
  (h_sum_p : r + s + t = 0)
  (h_sum_q : (r + 4) + (s - 3) + t' = 0)
  (ha_p : a = r * s + r * t + s * t)
  (ha_q : a = (r + 4) * (s - 3) + (r + 4) * (t' - 1) + (s - 3) * (t' - 1))
  (ht'_def : t' = t - 1)
  : b = -330 ∨ b = 90 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_b_l890_89082


namespace NUMINAMATH_GPT_banks_investments_count_l890_89001

-- Conditions
def revenue_per_investment_banks := 500
def revenue_per_investment_elizabeth := 900
def number_of_investments_elizabeth := 5
def extra_revenue_elizabeth := 500

-- Total revenue calculations
def total_revenue_elizabeth := number_of_investments_elizabeth * revenue_per_investment_elizabeth
def total_revenue_banks := total_revenue_elizabeth - extra_revenue_elizabeth

-- Number of investments for Mr. Banks
def number_of_investments_banks := total_revenue_banks / revenue_per_investment_banks

theorem banks_investments_count : number_of_investments_banks = 8 := by
  sorry

end NUMINAMATH_GPT_banks_investments_count_l890_89001


namespace NUMINAMATH_GPT_office_needs_24_pencils_l890_89026

noncomputable def number_of_pencils (total_cost : ℝ) (cost_per_pencil : ℝ) (cost_per_folder : ℝ) (number_of_folders : ℕ) : ℝ :=
  (total_cost - (number_of_folders * cost_per_folder)) / cost_per_pencil

theorem office_needs_24_pencils :
  number_of_pencils 30 0.5 0.9 20 = 24 :=
by
  sorry

end NUMINAMATH_GPT_office_needs_24_pencils_l890_89026


namespace NUMINAMATH_GPT_niu_fraction_property_l890_89027

open Nat

-- Given mn <= 2009, where m, n are positive integers and (n/m) is in lowest terms
-- Prove that for adjacent terms in the sequence, m_k n_{k+1} - m_{k+1} n_k = 1.

noncomputable def is_numerator_denom_pair_in_seq (m n : ℕ) : Bool :=
  m > 0 ∧ n > 0 ∧ m * n ≤ 2009

noncomputable def are_sorted_adjacent_in_seq (m_k n_k m_k1 n_k1 : ℕ) : Bool :=
  m_k * n_k1 - m_k1 * n_k = 1

theorem niu_fraction_property :
  ∀ (m_k n_k m_k1 n_k1 : ℕ),
  is_numerator_denom_pair_in_seq m_k n_k →
  is_numerator_denom_pair_in_seq m_k1 n_k1 →
  m_k < m_k1 →
  are_sorted_adjacent_in_seq m_k n_k m_k1 n_k1
:=
sorry

end NUMINAMATH_GPT_niu_fraction_property_l890_89027


namespace NUMINAMATH_GPT_angle_same_terminal_side_l890_89093

theorem angle_same_terminal_side (θ : ℝ) (α : ℝ) 
  (hθ : θ = -950) 
  (hα_range : 0 ≤ α ∧ α ≤ 180) 
  (h_terminal_side : ∃ k : ℤ, θ = α + k * 360) : 
  α = 130 := by
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l890_89093


namespace NUMINAMATH_GPT_total_players_l890_89096

def kabaddi (K : ℕ) (Kho_only : ℕ) (Both : ℕ) : ℕ :=
  K - Both + Kho_only + Both

theorem total_players (K : ℕ) (Kho_only : ℕ) (Both : ℕ)
  (hK : K = 10)
  (hKho_only : Kho_only = 35)
  (hBoth : Both = 5) :
  kabaddi K Kho_only Both = 45 :=
by
  rw [hK, hKho_only, hBoth]
  unfold kabaddi
  norm_num

end NUMINAMATH_GPT_total_players_l890_89096


namespace NUMINAMATH_GPT_OQ_value_l890_89042

variables {X Y Z N O Q R : Type}
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables [MetricSpace N] [MetricSpace O] [MetricSpace Q] [MetricSpace R]
variables (XY YZ XN NY ZO XO OZ YN XR OQ RQ : ℝ)
variables (triangle_XYZ : Triangle X Y Z)
variables (X_equal_midpoint_XY : XY = 540)
variables (Y_equal_midpoint_YZ : YZ = 360)
variables (XN_equal_NY : XN = NY)
variables (ZO_is_angle_bisector : is_angle_bisector Z O X Y)
variables (intersection_YN_ZO : Q = intersection YN ZO)
variables (N_midpoint_RQ : is_midpoint N R Q)
variables (XR_value : XR = 216)

theorem OQ_value : OQ = 216 := sorry

end NUMINAMATH_GPT_OQ_value_l890_89042


namespace NUMINAMATH_GPT_Q_neither_necessary_nor_sufficient_l890_89068

-- Define the propositions P and Q
def PropositionP (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  ∀ x : ℝ, (a1*x^2 + b1*x + c1 > 0) ↔ (a2*x^2 + b2*x + c2 > 0)

def PropositionQ (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 / a2 = b1 / b2) ∧ (b1 / b2 = c1 / c2)

-- The final statement to prove that Q is neither necessary nor sufficient for P
theorem Q_neither_necessary_nor_sufficient (a1 b1 c1 a2 b2 c2 : ℝ) :
  ¬ ((PropositionQ a1 b1 c1 a2 b2 c2) ↔ (PropositionP a1 b1 c1 a2 b2 c2)) := sorry

end NUMINAMATH_GPT_Q_neither_necessary_nor_sufficient_l890_89068


namespace NUMINAMATH_GPT_satisfy_eq_pairs_l890_89074

theorem satisfy_eq_pairs (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ (x = 4 ∧ (y = 1 ∨ y = -3) ∨ x = -4 ∧ (y = 1 ∨ y = -3)) :=
by
  sorry

end NUMINAMATH_GPT_satisfy_eq_pairs_l890_89074


namespace NUMINAMATH_GPT_combined_rate_is_29_l890_89097

def combined_rate_of_mpg (miles_ray : ℕ) (mpg_ray : ℕ) (miles_tom : ℕ) (mpg_tom : ℕ) (miles_jerry : ℕ) (mpg_jerry : ℕ) : ℕ :=
  let gallons_ray := miles_ray / mpg_ray
  let gallons_tom := miles_tom / mpg_tom
  let gallons_jerry := miles_jerry / mpg_jerry
  let total_gallons := gallons_ray + gallons_tom + gallons_jerry
  let total_miles := miles_ray + miles_tom + miles_jerry
  total_miles / total_gallons

theorem combined_rate_is_29 :
  combined_rate_of_mpg 60 50 60 20 60 30 = 29 :=
by
  sorry

end NUMINAMATH_GPT_combined_rate_is_29_l890_89097


namespace NUMINAMATH_GPT_jen_problem_correct_answer_l890_89050

-- Definitions based on the conditions
def sum_178_269 : ℤ := 178 + 269
def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n - (n % 100) + 100 else n - (n % 100)

-- Prove the statement
theorem jen_problem_correct_answer :
  round_to_nearest_hundred sum_178_269 = 400 :=
by
  have h1 : sum_178_269 = 447 := rfl
  have h2 : round_to_nearest_hundred 447 = 400 := by sorry
  exact h2

end NUMINAMATH_GPT_jen_problem_correct_answer_l890_89050


namespace NUMINAMATH_GPT_beam_count_represents_number_of_beams_l890_89018

def price := 6210
def transport_cost_per_beam := 3
def beam_condition (x : ℕ) : Prop := 
  transport_cost_per_beam * x * (x - 1) = price

theorem beam_count_represents_number_of_beams (x : ℕ) :
  beam_condition x → (∃ n : ℕ, x = n) := 
sorry

end NUMINAMATH_GPT_beam_count_represents_number_of_beams_l890_89018


namespace NUMINAMATH_GPT_solve_fractional_equation_l890_89025

theorem solve_fractional_equation
  (x : ℝ)
  (h1 : x ≠ 0)
  (h2 : x ≠ 2)
  (h_eq : 2 / x - 1 / (x - 2) = 0) : 
  x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l890_89025


namespace NUMINAMATH_GPT_both_buyers_correct_l890_89098

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers who purchase muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the number of buyers who purchase neither cake mix nor muffin mix
def neither_buyers : ℕ := 29

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- The assertion to be proved
theorem both_buyers_correct :
  neither_buyers = total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_buyers) :=
sorry

end NUMINAMATH_GPT_both_buyers_correct_l890_89098


namespace NUMINAMATH_GPT_conference_total_duration_is_715_l890_89090

structure ConferenceSession where
  hours : ℕ
  minutes : ℕ

def totalDuration (s1 s2 : ConferenceSession): ℕ :=
  (s1.hours * 60 + s1.minutes) + (s2.hours * 60 + s2.minutes)

def session1 : ConferenceSession := { hours := 8, minutes := 15 }
def session2 : ConferenceSession := { hours := 3, minutes := 40 }

theorem conference_total_duration_is_715 :
  totalDuration session1 session2 = 715 := 
sorry

end NUMINAMATH_GPT_conference_total_duration_is_715_l890_89090


namespace NUMINAMATH_GPT_Helga_articles_written_this_week_l890_89013

def articles_per_30_minutes : ℕ := 5
def work_hours_per_day : ℕ := 4
def work_days_per_week : ℕ := 5
def extra_hours_thursday : ℕ := 2
def extra_hours_friday : ℕ := 3

def articles_per_hour : ℕ := articles_per_30_minutes * 2
def regular_daily_articles : ℕ := articles_per_hour * work_hours_per_day
def regular_weekly_articles : ℕ := regular_daily_articles * work_days_per_week
def extra_thursday_articles : ℕ := articles_per_hour * extra_hours_thursday
def extra_friday_articles : ℕ := articles_per_hour * extra_hours_friday
def extra_weekly_articles : ℕ := extra_thursday_articles + extra_friday_articles
def total_weekly_articles : ℕ := regular_weekly_articles + extra_weekly_articles

theorem Helga_articles_written_this_week : total_weekly_articles = 250 := by
  sorry

end NUMINAMATH_GPT_Helga_articles_written_this_week_l890_89013


namespace NUMINAMATH_GPT_necess_suff_cond_odd_function_l890_89065

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) := Real.sin (ω * x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def P (ω ϕ : ℝ) : Prop := f ω ϕ 0 = 0
def Q (ω ϕ : ℝ) : Prop := is_odd (f ω ϕ)

theorem necess_suff_cond_odd_function (ω ϕ : ℝ) : P ω ϕ ↔ Q ω ϕ := by
  sorry

end NUMINAMATH_GPT_necess_suff_cond_odd_function_l890_89065


namespace NUMINAMATH_GPT_total_steps_l890_89043

theorem total_steps (steps_per_floor : ℕ) (n : ℕ) (m : ℕ) (h : steps_per_floor = 20) (hm : m = 11) (hn : n = 1) : 
  steps_per_floor * (m - n) = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_steps_l890_89043


namespace NUMINAMATH_GPT_hockeyPlayers_count_l890_89078

def numPlayers := 50
def cricketPlayers := 12
def footballPlayers := 11
def softballPlayers := 10

theorem hockeyPlayers_count : 
  let hockeyPlayers := numPlayers - (cricketPlayers + footballPlayers + softballPlayers)
  hockeyPlayers = 17 :=
by
  sorry

end NUMINAMATH_GPT_hockeyPlayers_count_l890_89078


namespace NUMINAMATH_GPT_question_1_question_2_l890_89053

open Real

theorem question_1 (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ab < m / 2 → m > 2 := sorry

theorem question_2 (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) (h4 : 9 / a + 1 / b ≥ |x - 1| + |x + 2|) :
  -9/2 ≤ x ∧ x ≤ 7/2 := sorry

end NUMINAMATH_GPT_question_1_question_2_l890_89053


namespace NUMINAMATH_GPT_smallest_sum_divisible_by_5_l890_89041

-- Definition of a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of four consecutive primes greater than 5
def four_consecutive_primes_greater_than_five (a b c d : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ a > 5 ∧ b > 5 ∧ c > 5 ∧ d > 5 ∧ 
  b = a + 4 ∧ c = b + 6 ∧ d = c + 2

-- The statement to prove
theorem smallest_sum_divisible_by_5 :
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) % 5 = 0 ∧
   ∀ x y z w : ℕ, four_consecutive_primes_greater_than_five x y z w → (x + y + z + w) % 5 = 0 → a + b + c + d ≤ x + y + z + w) →
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) = 60) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_divisible_by_5_l890_89041


namespace NUMINAMATH_GPT_speed_of_water_l890_89066

variable (v : ℝ)
variable (swimming_speed_still_water : ℝ := 10)
variable (time_against_current : ℝ := 8)
variable (distance_against_current : ℝ := 16)

theorem speed_of_water :
  distance_against_current = (swimming_speed_still_water - v) * time_against_current ↔ v = 8 := by
  sorry

end NUMINAMATH_GPT_speed_of_water_l890_89066


namespace NUMINAMATH_GPT_range_of_a_l890_89095

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (x^2 + 1) * Real.exp x ≥ a * x^2

theorem range_of_a (a : ℝ) : satisfies_inequality a ↔ a ≤ 2 * Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l890_89095


namespace NUMINAMATH_GPT_driving_time_is_correct_l890_89032

-- Define conditions
def flight_departure : ℕ := 20 * 60 -- 8:00 pm in minutes since 0:00
def checkin_time : ℕ := flight_departure - 2 * 60 -- 2 hours early
def latest_leave_time : ℕ := 17 * 60 -- 5:00 pm in minutes since 0:00
def additional_time : ℕ := 15 -- 15 minutes to park and make their way to the terminal

-- Define question
def driving_time : ℕ := checkin_time - additional_time - latest_leave_time

-- Prove the expected answer
theorem driving_time_is_correct : driving_time = 45 :=
by
  -- omitting the proof
  sorry

end NUMINAMATH_GPT_driving_time_is_correct_l890_89032


namespace NUMINAMATH_GPT_binomial_coefficient_and_factorial_l890_89007

open Nat

/--
  Given:
    - The binomial coefficient definition: Nat.choose n k = n! / (k! * (n - k)!)
    - The factorial definition: Nat.factorial n = n * (n - 1) * ... * 1
  Prove:
    Nat.choose 60 3 * Nat.factorial 10 = 124467072000
-/
theorem binomial_coefficient_and_factorial :
  Nat.choose 60 3 * Nat.factorial 10 = 124467072000 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_and_factorial_l890_89007


namespace NUMINAMATH_GPT_compute_4_star_3_l890_89006

def custom_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem compute_4_star_3 : custom_op 4 3 = 13 :=
by
  sorry

end NUMINAMATH_GPT_compute_4_star_3_l890_89006


namespace NUMINAMATH_GPT_subset_proper_l890_89067

def M : Set ℝ := {x | x^2 - x ≤ 0}

def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem subset_proper : N ⊂ M := by
  sorry

end NUMINAMATH_GPT_subset_proper_l890_89067


namespace NUMINAMATH_GPT_custom_op_example_l890_89062

def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

theorem custom_op_example : custom_op (-4) 6 = 68 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_example_l890_89062


namespace NUMINAMATH_GPT_complement_set_M_l890_89005

-- Definitions of sets based on given conditions
def universal_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def set_M : Set ℝ := {x | x^2 - x ≤ 0}

-- The proof statement that we need to prove
theorem complement_set_M :
  {x | 1 < x ∧ x ≤ 2} = universal_set \ set_M := by
  sorry

end NUMINAMATH_GPT_complement_set_M_l890_89005


namespace NUMINAMATH_GPT_vector_parallel_solution_l890_89046

theorem vector_parallel_solution 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (x, -9)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  x = -6 :=
by
  sorry

end NUMINAMATH_GPT_vector_parallel_solution_l890_89046


namespace NUMINAMATH_GPT_k_value_opposite_solutions_l890_89008

theorem k_value_opposite_solutions (k x1 x2 : ℝ) 
  (h1 : 3 * (2 * x1 - 1) = 1 - 2 * x1)
  (h2 : 8 - k = 2 * (x2 + 1))
  (opposite : x2 = -x1) :
  k = 7 :=
by sorry

end NUMINAMATH_GPT_k_value_opposite_solutions_l890_89008


namespace NUMINAMATH_GPT_swap_original_x_y_l890_89071

variables (x y z : ℕ)

theorem swap_original_x_y (x_original y_original : ℕ) 
  (step1 : z = x_original)
  (step2 : x = y_original)
  (step3 : y = z) :
  x = y_original ∧ y = x_original :=
sorry

end NUMINAMATH_GPT_swap_original_x_y_l890_89071


namespace NUMINAMATH_GPT_theta_half_quadrant_l890_89036

open Real

theorem theta_half_quadrant (θ : ℝ) (k : ℤ) 
  (h1 : 2 * k * π + 3 * π / 2 ≤ θ ∧ θ ≤ 2 * k * π + 2 * π) 
  (h2 : |cos (θ / 2)| = -cos (θ / 2)) : 
  k * π + 3 * π / 4 ≤ θ / 2 ∧ θ / 2 ≤ k * π + π ∧ cos (θ / 2) < 0 := 
sorry

end NUMINAMATH_GPT_theta_half_quadrant_l890_89036


namespace NUMINAMATH_GPT_neighbor_to_johnson_yield_ratio_l890_89040

-- Definitions
def johnsons_yield (months : ℕ) : ℕ := 80 * (months / 2)
def neighbors_yield_per_hectare (x : ℕ) (months : ℕ) : ℕ := 80 * x * (months / 2)
def total_neighor_yield (x : ℕ) (months : ℕ) : ℕ := 2 * neighbors_yield_per_hectare x months

-- Theorem statement
theorem neighbor_to_johnson_yield_ratio
  (x : ℕ)
  (h1 : johnsons_yield 6 = 240)
  (h2 : total_neighor_yield x 6 = 480 * x)
  (h3 : johnsons_yield 6 + total_neighor_yield x 6 = 1200)
  : x = 2 := by
sorry

end NUMINAMATH_GPT_neighbor_to_johnson_yield_ratio_l890_89040


namespace NUMINAMATH_GPT_problem1_problem2_l890_89012

-- Problem (I)
theorem problem1 (a b : ℝ) (h : a ≥ b ∧ b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := sorry

-- Problem (II)
theorem problem2 (a b c x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 10) 
  (h2 : x^2 + y^2 + z^2 = 40) 
  (h3 : a * x + b * y + c * z = 20) : 
  (a + b + c) / (x + y + z) = 1 / 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l890_89012


namespace NUMINAMATH_GPT_starting_number_of_range_l890_89044

theorem starting_number_of_range (N : ℕ) : ∃ (start : ℕ), 
  (∀ n, n ≥ start ∧ n ≤ 200 → ∃ k, 8 * k = n) ∧ -- All numbers between start and 200 inclusive are multiples of 8
  (∃ k, k = (200 / 8) ∧ 25 - k = 13.5) ∧ -- There are 13.5 multiples of 8 in the range
  start = 84 := 
sorry

end NUMINAMATH_GPT_starting_number_of_range_l890_89044


namespace NUMINAMATH_GPT_required_hours_for_fifth_week_l890_89057

def typical_hours_needed (week1 week2 week3 week4 week5 add_hours total_weeks target_avg : ℕ) : ℕ :=
  if (week1 + week2 + week3 + week4 + week5 + add_hours) / total_weeks = target_avg then 
    week5 
  else 
    0

theorem required_hours_for_fifth_week :
  typical_hours_needed 10 14 11 9 x 1 5 12 = 15 :=
by
  sorry

end NUMINAMATH_GPT_required_hours_for_fifth_week_l890_89057


namespace NUMINAMATH_GPT_uncle_bradley_money_l890_89092

-- Definitions of the variables and conditions
variables (F H M : ℝ)
variables (h1 : F + H = 13)
variables (h2 : 50 * F = (3 / 10) * M)
variables (h3 : 100 * H = (7 / 10) * M)

-- The theorem statement
theorem uncle_bradley_money : M = 1300 :=
by
  sorry

end NUMINAMATH_GPT_uncle_bradley_money_l890_89092


namespace NUMINAMATH_GPT_sum_of_possible_M_l890_89058

theorem sum_of_possible_M (M : ℝ) (h : M * (M - 8) = -8) : M = 4 ∨ M = 4 := 
by sorry

end NUMINAMATH_GPT_sum_of_possible_M_l890_89058


namespace NUMINAMATH_GPT_ratio_of_cats_l890_89028

-- Definitions from conditions
def total_animals_anthony := 12
def fraction_cats_anthony := 2 / 3
def extra_dogs_leonel := 7
def total_animals_both := 27

-- Calculate number of cats and dogs Anthony has
def cats_anthony := fraction_cats_anthony * total_animals_anthony
def dogs_anthony := total_animals_anthony - cats_anthony

-- Calculate number of dogs Leonel has
def dogs_leonel := dogs_anthony + extra_dogs_leonel

-- Calculate number of cats Leonel has
def cats_leonel := total_animals_both - (cats_anthony + dogs_anthony + dogs_leonel)

-- Prove the desired ratio
theorem ratio_of_cats : (cats_leonel / cats_anthony) = (1 / 2) := by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_ratio_of_cats_l890_89028


namespace NUMINAMATH_GPT_utility_bills_total_l890_89060

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end NUMINAMATH_GPT_utility_bills_total_l890_89060


namespace NUMINAMATH_GPT_prob_three_heads_is_one_eighth_l890_89073

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end NUMINAMATH_GPT_prob_three_heads_is_one_eighth_l890_89073


namespace NUMINAMATH_GPT_lcm_of_three_numbers_l890_89076

theorem lcm_of_three_numbers (x : ℕ) :
  (Nat.gcd (3 * x) (Nat.gcd (4 * x) (5 * x)) = 40) →
  Nat.lcm (3 * x) (Nat.lcm (4 * x) (5 * x)) = 2400 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_three_numbers_l890_89076
