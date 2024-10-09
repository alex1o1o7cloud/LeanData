import Mathlib

namespace pythagorean_theorem_mod_3_l1710_171076

theorem pythagorean_theorem_mod_3 {x y z : ℕ} (h : x^2 + y^2 = z^2) : x % 3 = 0 ∨ y % 3 = 0 ∨ z % 3 = 0 :=
by 
  sorry

end pythagorean_theorem_mod_3_l1710_171076


namespace mary_total_nickels_l1710_171038

-- Define the initial number of nickels Mary had
def mary_initial_nickels : ℕ := 7

-- Define the number of nickels her dad gave her
def mary_received_nickels : ℕ := 5

-- The goal is to prove the total number of nickels Mary has now is 12
theorem mary_total_nickels : mary_initial_nickels + mary_received_nickels = 12 :=
by
  sorry

end mary_total_nickels_l1710_171038


namespace largest_y_l1710_171014

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

theorem largest_y (x y : ℕ) (hx : x ≥ y) (hy : y ≥ 3) 
  (h : (interior_angle x * 28) = (interior_angle y * 29)) :
  y = 57 :=
by
  sorry

end largest_y_l1710_171014


namespace integer_solutions_of_inequality_system_l1710_171094

theorem integer_solutions_of_inequality_system :
  { x : ℤ | (3 * x - 2) / 3 ≥ 1 ∧ 3 * x + 5 > 4 * x - 2 } = {2, 3, 4, 5, 6} :=
by {
  sorry
}

end integer_solutions_of_inequality_system_l1710_171094


namespace percentage_divisible_by_7_l1710_171065

-- Define the total integers and the condition for being divisible by 7
def total_ints := 140
def divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

-- Calculate the number of integers between 1 and 140 that are divisible by 7
def count_divisible_by_7 : ℕ := Nat.succ (140 / 7)

-- The theorem to prove
theorem percentage_divisible_by_7 : (count_divisible_by_7 / total_ints : ℚ) * 100 = 14.29 := by
  sorry

end percentage_divisible_by_7_l1710_171065


namespace rubber_band_problem_l1710_171044

noncomputable def a : ℤ := 4
noncomputable def b : ℤ := 12
noncomputable def c : ℤ := 3
noncomputable def band_length := a * Real.pi + b * Real.sqrt c

theorem rubber_band_problem (r1 r2 d : ℝ) (h1 : r1 = 3) (h2 : r2 = 9) (h3 : d = 12) :
  let a := 4
  let b := 12
  let c := 3
  let band_length := a * Real.pi + b * Real.sqrt c
  a + b + c = 19 :=
by
  sorry

end rubber_band_problem_l1710_171044


namespace ratio_of_liquid_p_to_q_initial_l1710_171017

noncomputable def initial_ratio_of_p_to_q : ℚ :=
  let p := 20
  let q := 15
  p / q

theorem ratio_of_liquid_p_to_q_initial
  (p q : ℚ)
  (h1 : p + q = 35)
  (h2 : p / (q + 13) = 5 / 7) :
  p / q = 4 / 3 := by
  sorry

end ratio_of_liquid_p_to_q_initial_l1710_171017


namespace num_paths_from_E_to_G_pass_through_F_l1710_171011

-- Definitions for the positions on the grid.
def E := (0, 4)
def G := (5, 0)
def F := (3, 3)

-- Function to calculate the number of combinations.
def binom (n k: ℕ) : ℕ := Nat.choose n k

-- The mathematical statement to be proven.
theorem num_paths_from_E_to_G_pass_through_F :
  (binom 4 1) * (binom 5 2) = 40 :=
by
  -- Placeholder for the proof.
  sorry

end num_paths_from_E_to_G_pass_through_F_l1710_171011


namespace length_of_major_axis_l1710_171003

def ellipse_length_major_axis (a b : ℝ) : ℝ := 2 * a

theorem length_of_major_axis : ellipse_length_major_axis 4 1 = 8 :=
by
  unfold ellipse_length_major_axis
  norm_num

end length_of_major_axis_l1710_171003


namespace total_units_l1710_171083

theorem total_units (A B C: ℕ) (hA: A = 2 + 4 + 6 + 8 + 10 + 12) (hB: B = A) (hC: C = 3 + 5 + 7 + 9) : 
  A + B + C = 108 := 
sorry

end total_units_l1710_171083


namespace choir_members_l1710_171027

theorem choir_members (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 11 = 6) (h3 : 200 ≤ n ∧ n ≤ 300) :
  n = 220 :=
sorry

end choir_members_l1710_171027


namespace value_of_m_l1710_171037

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end value_of_m_l1710_171037


namespace distance_to_city_center_l1710_171019

theorem distance_to_city_center 
  (D : ℕ) 
  (H1 : D = 200 + 200 + D) 
  (H_total : 900 = 200 + 200 + D) : 
  D = 500 :=
by { sorry }

end distance_to_city_center_l1710_171019


namespace initial_people_in_elevator_l1710_171062

theorem initial_people_in_elevator (W n : ℕ) (avg_initial_weight avg_new_weight new_person_weight : ℚ)
  (h1 : avg_initial_weight = 152)
  (h2 : avg_new_weight = 151)
  (h3 : new_person_weight = 145)
  (h4 : W = n * avg_initial_weight)
  (h5 : W + new_person_weight = (n + 1) * avg_new_weight) :
  n = 6 :=
by
  sorry

end initial_people_in_elevator_l1710_171062


namespace distance_BC_l1710_171015

theorem distance_BC (AB AC CD DA: ℝ) (hAB: AB = 50) (hAC: AC = 40) (hCD: CD = 25) (hDA: DA = 35):
  BC = 10 ∨ BC = 90 :=
by
  sorry

end distance_BC_l1710_171015


namespace find_k_l1710_171035

theorem find_k (k : ℕ) (h_pos : k > 0) (h_coef : 15 * k^4 < 120) : k = 1 :=
sorry

end find_k_l1710_171035


namespace average_age_of_population_l1710_171061

theorem average_age_of_population
  (k : ℕ)
  (ratio_women_men : 7 * (k : ℕ) = 7 * (k : ℕ) + 5 * (k : ℕ) - 5 * (k : ℕ))
  (avg_age_women : ℝ := 38)
  (avg_age_men : ℝ := 36)
  : ( (7 * k * avg_age_women) + (5 * k * avg_age_men) ) / (12 * k) = 37 + (1 / 6) :=
by
  sorry

end average_age_of_population_l1710_171061


namespace john_makes_money_l1710_171013

-- Definitions of the conditions
def num_cars := 5
def time_first_3_cars := 3 * 40 -- 3 cars each take 40 minutes
def time_remaining_car := 40 * 3 / 2 -- Each remaining car takes 50% longer
def time_remaining_cars := 2 * time_remaining_car -- 2 remaining cars
def total_time_min := time_first_3_cars + time_remaining_cars
def total_time_hr := total_time_min / 60 -- Convert total time from minutes to hours
def rate_per_hour := 20

-- Theorem statement
theorem john_makes_money : total_time_hr * rate_per_hour = 80 := by
  sorry

end john_makes_money_l1710_171013


namespace origami_papers_total_l1710_171032

-- Define the conditions as Lean definitions
def num_cousins : ℕ := 6
def papers_per_cousin : ℕ := 8

-- Define the total number of origami papers that Haley has to give away
def total_papers : ℕ := num_cousins * papers_per_cousin

-- Statement of the proof
theorem origami_papers_total : total_papers = 48 :=
by
  -- Skipping the proof for now
  sorry

end origami_papers_total_l1710_171032


namespace necessary_and_sufficient_condition_l1710_171021

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x - 4 * a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
sorry

end necessary_and_sufficient_condition_l1710_171021


namespace floor_length_is_twelve_l1710_171007

-- Definitions based on the conditions
def floor_width := 10
def strip_width := 3
def rug_area := 24

-- Problem statement
theorem floor_length_is_twelve (L : ℕ) 
  (h1 : rug_area = (L - 2 * strip_width) * (floor_width - 2 * strip_width)) :
  L = 12 := 
sorry

end floor_length_is_twelve_l1710_171007


namespace bucket_B_more_than_C_l1710_171034

-- Define the number of pieces of fruit in bucket B as a constant
def B := 12

-- Define the number of pieces of fruit in bucket C as a constant
def C := 9

-- Define the number of pieces of fruit in bucket A based on B
def A := B + 4

-- Define the total number of pieces of fruit in all three buckets
def total_fruit := A + B + C

-- Prove that bucket B has 3 more pieces of fruit than bucket C
theorem bucket_B_more_than_C : B - C = 3 := by
  -- sorry is used to skip the proof
  sorry

end bucket_B_more_than_C_l1710_171034


namespace rhombus_area_l1710_171047

-- Define the lengths of the diagonals
def d1 : ℝ := 6
def d2 : ℝ := 8

-- Problem statement: The area of the rhombus
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : (1 / 2) * d1 * d2 = 24 := by
  -- The proof is not required, so we use sorry.
  sorry

end rhombus_area_l1710_171047


namespace product_of_solutions_l1710_171069

theorem product_of_solutions :
  ∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) →
  (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x = x1 ∨ x = x2) → x1 * x2 = 0) :=
by
  sorry

end product_of_solutions_l1710_171069


namespace hiker_total_distance_l1710_171085

def hiker_distance (day1_hours day1_speed day2_speed : ℕ) : ℕ :=
  let day2_hours := day1_hours - 1
  let day3_hours := day1_hours
  (day1_hours * day1_speed) + (day2_hours * day2_speed) + (day3_hours * day2_speed)

theorem hiker_total_distance :
  hiker_distance 6 3 4 = 62 := 
by 
  sorry

end hiker_total_distance_l1710_171085


namespace James_trout_pounds_l1710_171079

def pounds_trout (T : ℝ) : Prop :=
  let salmon := 1.5 * T
  let tuna := 2 * T
  T + salmon + tuna = 1100

theorem James_trout_pounds :
  ∃ T : ℝ, pounds_trout T ∧ T = 244 :=
sorry

end James_trout_pounds_l1710_171079


namespace common_real_root_pair_l1710_171060

theorem common_real_root_pair (n : ℕ) (hn : n > 1) :
  ∃ x : ℝ, (∃ a b : ℤ, ((x^n + (a : ℝ) * x = 2008) ∧ (x^n + (b : ℝ) * x = 2009))) ↔
    ((a = 2007 ∧ b = 2008) ∨
     (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end common_real_root_pair_l1710_171060


namespace graph_of_transformed_function_l1710_171008

theorem graph_of_transformed_function
  (f : ℝ → ℝ)
  (hf : f⁻¹ 1 = 0) :
  f (1 - 1) = 1 :=
by
  sorry

end graph_of_transformed_function_l1710_171008


namespace proof_inequality_l1710_171026

noncomputable def proof_problem (x : ℝ) (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) : Prop :=
  let a := Real.log x
  let b := (1 / 2) ^ (Real.log x)
  let c := Real.exp (Real.log x)
  b > c ∧ c > a

theorem proof_inequality {x : ℝ} (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) :
  proof_problem x Hx :=
sorry

end proof_inequality_l1710_171026


namespace sequence_difference_l1710_171041

theorem sequence_difference : 
  (∃ (a : ℕ → ℤ) (S : ℕ → ℤ), 
    (∀ n : ℕ, S n = n^2 + 2 * n) ∧ 
    (∀ n : ℕ, n > 0 → a n = S n - S (n - 1) ) ∧ 
    (a 4 - a 2 = 4)) :=
by
  sorry

end sequence_difference_l1710_171041


namespace candy_seller_initial_candies_l1710_171054

-- Given conditions
def num_clowns : ℕ := 4
def num_children : ℕ := 30
def candies_per_person : ℕ := 20
def candies_left : ℕ := 20

-- Question: What was the initial number of candies?
def total_people : ℕ := num_clowns + num_children
def total_candies_given_out : ℕ := total_people * candies_per_person
def initial_candies : ℕ := total_candies_given_out + candies_left

theorem candy_seller_initial_candies : initial_candies = 700 :=
by
  sorry

end candy_seller_initial_candies_l1710_171054


namespace relationship_among_neg_a_square_neg_a_cube_l1710_171029

theorem relationship_among_neg_a_square_neg_a_cube (a : ℝ) (h : -1 < a ∧ a < 0) : (-a > a^2 ∧ a^2 > -a^3) :=
by
  sorry

end relationship_among_neg_a_square_neg_a_cube_l1710_171029


namespace wire_cut_perimeter_equal_l1710_171084

theorem wire_cut_perimeter_equal (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 4 * (a / 4) = 8 * (b / 8)) :
  a / b = 1 :=
sorry

end wire_cut_perimeter_equal_l1710_171084


namespace sum_of_arithmetic_sequence_l1710_171056

theorem sum_of_arithmetic_sequence
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (hS : ∀ n : ℕ, S n = n * a n)
    (h_condition : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
    S 19 = -38 :=
sorry

end sum_of_arithmetic_sequence_l1710_171056


namespace compound_interest_second_year_l1710_171071

variables {P r CI_2 CI_3 : ℝ}

-- Given conditions as definitions in Lean
def interest_rate : ℝ := 0.05
def year_3_interest : ℝ := 1260
def relation_between_CI2_and_CI3 (CI_2 CI_3 : ℝ) : Prop :=
  CI_3 = CI_2 * (1 + interest_rate)

-- The theorem to prove
theorem compound_interest_second_year :
  relation_between_CI2_and_CI3 CI_2 year_3_interest ∧
  r = interest_rate →
  CI_2 = 1200 := 
sorry

end compound_interest_second_year_l1710_171071


namespace women_in_club_l1710_171082

theorem women_in_club (total_members : ℕ) (men : ℕ) (total_members_eq : total_members = 52) (men_eq : men = 37) :
  ∃ women : ℕ, women = 15 :=
by
  sorry

end women_in_club_l1710_171082


namespace inequality_solution_minimum_value_l1710_171006

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem inequality_solution :
  {x : ℝ | f x > 7} = {x | x > 4 ∨ x < -3} :=
by
  sorry

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : ∀ x, f x ≥ m + n) :
  m + n = 3 →
  (m^2 + n^2 ≥ 9 / 2 ∧ (m = 3 / 2 ∧ n = 3 / 2)) :=
by
  sorry

end inequality_solution_minimum_value_l1710_171006


namespace base7_sub_base5_to_base10_l1710_171018

def base7to10 (n : Nat) : Nat :=
  match n with
  | 52403 => 5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def base5to10 (n : Nat) : Nat :=
  match n with
  | 20345 => 2 * 5^4 + 0 * 5^3 + 3 * 5^2 + 4 * 5^1 + 5 * 5^0
  | _ => 0

theorem base7_sub_base5_to_base10 :
  base7to10 52403 - base5to10 20345 = 11540 :=
by
  sorry

end base7_sub_base5_to_base10_l1710_171018


namespace scientific_notation_of_448000_l1710_171012

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l1710_171012


namespace problem_statement_l1710_171063

variable {a b c d : ℝ}

theorem problem_statement (h : a * d - b * c = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := 
sorry

end problem_statement_l1710_171063


namespace john_father_age_difference_l1710_171089

theorem john_father_age_difference (J F X : ℕ) (h1 : J + F = 77) (h2 : J = 15) (h3 : F = 2 * J + X) : X = 32 :=
by
  -- Adding the "sory" to skip the proof
  sorry

end john_father_age_difference_l1710_171089


namespace probability_king_of_diamonds_top_two_l1710_171095

-- Definitions based on the conditions
def total_cards : ℕ := 54
def king_of_diamonds : ℕ := 1
def jokers : ℕ := 2

-- The main theorem statement proving the probability
theorem probability_king_of_diamonds_top_two :
  let prob := (king_of_diamonds / total_cards) + ((total_cards - 1) / total_cards * king_of_diamonds / (total_cards - 1))
  prob = 1 / 27 :=
by
  sorry

end probability_king_of_diamonds_top_two_l1710_171095


namespace perfect_square_m_value_l1710_171058

theorem perfect_square_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ x : ℝ, (x^2 + (m : ℝ)*x + 1 : ℝ) = (x + (a : ℝ))^2) → m = 2 ∨ m = -2 :=
by
  sorry

end perfect_square_m_value_l1710_171058


namespace problem_statement_l1710_171042

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * (x - 1))

theorem problem_statement :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x1 x2 : ℝ, x1 + x2 = π / 2 → g x1 = g x2) :=
by 
  sorry

end problem_statement_l1710_171042


namespace mn_necessary_not_sufficient_l1710_171087

variable (m n : ℝ)

def is_ellipse (m n : ℝ) : Prop := 
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem mn_necessary_not_sufficient : (mn > 0) → (is_ellipse m n) ↔ false := 
by sorry

end mn_necessary_not_sufficient_l1710_171087


namespace sqrt_sum_ge_two_l1710_171074

theorem sqrt_sum_ge_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 := 
by
  sorry

end sqrt_sum_ge_two_l1710_171074


namespace count_five_digit_multiples_of_five_l1710_171043

theorem count_five_digit_multiples_of_five : 
  ∃ (n : ℕ), n = 18000 ∧ (∀ x, 10000 ≤ x ∧ x ≤ 99999 ∧ x % 5 = 0 ↔ ∃ k, 10000 ≤ 5 * k ∧ 5 * k ≤ 99999) :=
by
  sorry

end count_five_digit_multiples_of_five_l1710_171043


namespace grading_combinations_l1710_171030

/-- There are 12 students in the class. -/
def num_students : ℕ := 12

/-- There are 4 possible grades (A, B, C, and D). -/
def num_grades : ℕ := 4

/-- The total number of ways to assign grades. -/
theorem grading_combinations : (num_grades ^ num_students) = 16777216 := 
by
  sorry

end grading_combinations_l1710_171030


namespace profit_percentage_l1710_171049

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 625) : 
  ((SP - CP) / CP) * 100 = 25 := 
by 
  sorry

end profit_percentage_l1710_171049


namespace last_digit_B_l1710_171064

theorem last_digit_B 
  (B : ℕ) 
  (h : ∀ n : ℕ, n % 10 = (B - 287)^2 % 10 → n % 10 = 4) :
  (B = 5 ∨ B = 9) :=
sorry

end last_digit_B_l1710_171064


namespace range_of_a_l1710_171070

noncomputable def has_root_in_R (f : ℝ → ℝ) : Prop :=
∃ x : ℝ, f x = 0

theorem range_of_a (a : ℝ) (h : has_root_in_R (λ x => 4 * x + a * 2^x + a + 1)) : a ≤ 0 :=
sorry

end range_of_a_l1710_171070


namespace quintuplets_babies_l1710_171093

theorem quintuplets_babies (t r q : ℕ) (h1 : r = 6 * q)
  (h2 : t = 2 * r)
  (h3 : 2 * t + 3 * r + 5 * q = 1500) :
  5 * q = 160 :=
by
  sorry

end quintuplets_babies_l1710_171093


namespace ice_forms_inner_surface_in_winter_l1710_171022

-- Definitions based on conditions
variable (humid_air_inside : Prop) 
variable (heat_transfer_inner_surface : Prop) 
variable (heat_transfer_outer_surface : Prop) 
variable (temp_inner_surface_below_freezing : Prop) 
variable (condensation_inner_surface_below_freezing : Prop)
variable (ice_formation_inner_surface : Prop)
variable (cold_dry_air_outside : Prop)
variable (no_significant_condensation_outside : Prop)

-- Proof of the theorem
theorem ice_forms_inner_surface_in_winter :
  humid_air_inside ∧
  heat_transfer_inner_surface ∧
  heat_transfer_outer_surface ∧
  (¬sufficient_heating → temp_inner_surface_below_freezing) ∧
  (condensation_inner_surface_below_freezing ↔ (temp_inner_surface_below_freezing ∧ humid_air_inside)) ∧
  (ice_formation_inner_surface ↔ (condensation_inner_surface_below_freezing ∧ temp_inner_surface_below_freezing)) ∧
  (cold_dry_air_outside → ¬ice_formation_outer_surface)
  → ice_formation_inner_surface :=
sorry

end ice_forms_inner_surface_in_winter_l1710_171022


namespace sqrt_three_squared_l1710_171055

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end sqrt_three_squared_l1710_171055


namespace domain_of_log_function_l1710_171000

theorem domain_of_log_function (x : ℝ) :
  (5 - x > 0) ∧ (x - 2 > 0) ∧ (x - 2 ≠ 1) ↔ (2 < x ∧ x < 3) ∨ (3 < x ∧ x < 5) :=
by
  sorry

end domain_of_log_function_l1710_171000


namespace john_speed_above_limit_l1710_171090

theorem john_speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) 
  (h1 : distance = 150) (h2 : time = 2) (h3 : speed_limit = 60) : 
  (distance / time) - speed_limit = 15 :=
by
  -- steps to show the proof
  sorry

end john_speed_above_limit_l1710_171090


namespace similar_triangles_x_value_l1710_171020

theorem similar_triangles_x_value
  (x : ℝ)
  (h_similar : ∀ (AB BC DE EF : ℝ), AB / BC = DE / EF)
  (h_AB : AB = x)
  (h_BC : BC = 33)
  (h_DE : DE = 96)
  (h_EF : EF = 24) :
  x = 132 :=
by
  -- Proof steps will be here
  sorry

end similar_triangles_x_value_l1710_171020


namespace exp_increasing_a_lt_zero_l1710_171002

theorem exp_increasing_a_lt_zero (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (1 - a) ^ x1 < (1 - a) ^ x2) : a < 0 := 
sorry

end exp_increasing_a_lt_zero_l1710_171002


namespace prism_lateral_edges_correct_cone_axial_section_equilateral_l1710_171091

/-- Defining the lateral edges of a prism and its properties --/
structure Prism (r : ℝ) :=
(lateral_edges_equal : ∀ (e1 e2 : ℝ), e1 = r ∧ e2 = r)

/-- Defining the axial section of a cone with properties of base radius and generatrix length --/
structure Cone (r : ℝ) :=
(base_radius : ℝ := r)
(generatrix_length : ℝ := 2 * r)
(is_equilateral : base_radius * 2 = generatrix_length)

theorem prism_lateral_edges_correct (r : ℝ) (P : Prism r) : 
 ∃ e, e = r ∧ ∀ e', e' = r :=
by {
  sorry
}

theorem cone_axial_section_equilateral (r : ℝ) (C : Cone r) : 
 base_radius * 2 = generatrix_length :=
by {
  sorry
}

end prism_lateral_edges_correct_cone_axial_section_equilateral_l1710_171091


namespace number_of_pages_in_book_l1710_171009

-- Define the conditions using variables and hypotheses
variables (P : ℝ) (h1 : 0.30 * P = 150)

-- State the theorem to be proved
theorem number_of_pages_in_book : P = 500 :=
by
  -- Proof would go here, but we use sorry to skip it
  sorry

end number_of_pages_in_book_l1710_171009


namespace pictures_hung_in_new_galleries_l1710_171033

noncomputable def total_pencils_used : ℕ := 218
noncomputable def pencils_per_picture : ℕ := 5
noncomputable def pencils_per_exhibition : ℕ := 3

noncomputable def pictures_initial : ℕ := 9
noncomputable def galleries_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]
noncomputable def total_exhibitions : ℕ := 1 + galleries_requests.length

theorem pictures_hung_in_new_galleries :
  let total_pencils_for_signing := total_exhibitions * pencils_per_exhibition
  let total_pencils_for_drawing := total_pencils_used - total_pencils_for_signing
  let total_pictures_drawn := total_pencils_for_drawing / pencils_per_picture
  let pictures_in_new_galleries := total_pictures_drawn - pictures_initial
  pictures_in_new_galleries = 29 :=
by
  sorry

end pictures_hung_in_new_galleries_l1710_171033


namespace proportional_parts_middle_l1710_171066

theorem proportional_parts_middle (x : ℚ) (hx : x + (1/2) * x + (1/4) * x = 120) : (1/2) * x = 240 / 7 :=
by
  sorry

end proportional_parts_middle_l1710_171066


namespace no_real_solution_ineq_l1710_171096

theorem no_real_solution_ineq (x : ℝ) (h : x ≠ 5) : ¬ (x^3 - 125) / (x - 5) < 0 := 
by
  sorry

end no_real_solution_ineq_l1710_171096


namespace girls_without_notebooks_l1710_171024

noncomputable def girls_in_class : Nat := 20
noncomputable def students_with_notebooks : Nat := 25
noncomputable def boys_with_notebooks : Nat := 16

theorem girls_without_notebooks : 
  (girls_in_class - (students_with_notebooks - boys_with_notebooks)) = 11 := by
  sorry

end girls_without_notebooks_l1710_171024


namespace probability_correct_l1710_171048

noncomputable def probability_B1_eq_5_given_WB : ℚ :=
  let P_B1_eq_5 : ℚ := 1 / 8
  let P_WB : ℚ := 1 / 5
  let P_WB_given_B1_eq_5 : ℚ := 1 / 16 + 369 / 2048
  (P_B1_eq_5 * P_WB_given_B1_eq_5) / P_WB

theorem probability_correct :
  probability_B1_eq_5_given_WB = 115 / 1024 :=
by
  sorry

end probability_correct_l1710_171048


namespace intersection_closure_M_and_N_l1710_171081

noncomputable def set_M : Set ℝ :=
  { x | 2 / x < 1 }

noncomputable def closure_M : Set ℝ :=
  Set.Icc 0 2

noncomputable def set_N : Set ℝ :=
  { y | ∃ x, y = Real.sqrt (x - 1) }

theorem intersection_closure_M_and_N :
  (closure_M ∩ set_N) = Set.Icc 0 2 :=
by
  sorry

end intersection_closure_M_and_N_l1710_171081


namespace tan_alpha_eq_4_over_3_expression_value_eq_4_l1710_171078

-- Conditions
variable (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi / 2)) (h_sin : Real.sin α = 4 / 5)

-- Prove: tan α = 4 / 3
theorem tan_alpha_eq_4_over_3 : Real.tan α = 4 / 3 :=
by
  sorry

-- Prove: the value of the given expression is 4
theorem expression_value_eq_4 : 
  (Real.sin (α + Real.pi) - 2 * Real.cos ((Real.pi / 2) + α)) / 
  (- Real.sin (-α) + Real.cos (Real.pi + α)) = 4 :=
by
  sorry

end tan_alpha_eq_4_over_3_expression_value_eq_4_l1710_171078


namespace remainder_of_x50_div_x_plus_1_cubed_l1710_171052

theorem remainder_of_x50_div_x_plus_1_cubed (x : ℚ) : 
  (x ^ 50) % ((x + 1) ^ 3) = 1225 * x ^ 2 + 2450 * x + 1176 :=
by sorry

end remainder_of_x50_div_x_plus_1_cubed_l1710_171052


namespace transform_polynomial_l1710_171031

variables {x y : ℝ}

theorem transform_polynomial (h : y = x - 1 / x) :
  (x^6 + x^5 - 5 * x^4 + 2 * x^3 - 5 * x^2 + x + 1 = 0) ↔ (x^2 * (y^2 + y - 3) = 0) :=
sorry

end transform_polynomial_l1710_171031


namespace ordered_quadruple_ellipse_l1710_171057

noncomputable def ellipse_quadruple := 
  let f₁ : (ℝ × ℝ) := (1, 1)
  let f₂ : (ℝ × ℝ) := (1, 7)
  let p : (ℝ × ℝ) := (12, -1)
  let a := (5 / 2) * (Real.sqrt 5 + Real.sqrt 37)
  let b := (1 / 2) * Real.sqrt (1014 + 50 * Real.sqrt 185)
  let h := 1
  let k := 4
  (a, b, h, k)

theorem ordered_quadruple_ellipse :
  let e : (ℝ × ℝ × ℝ × ℝ) := θse_quadruple
  e = ((5 / 2 * (Real.sqrt 5 + Real.sqrt 37)), (1 / 2 * Real.sqrt (1014 + 50 * Real.sqrt 185)), 1, 4) :=
by
  sorry

end ordered_quadruple_ellipse_l1710_171057


namespace quadrilateral_perimeter_l1710_171077

noncomputable def EG (FH : ℝ) : ℝ := Real.sqrt ((FH + 5) ^ 2 + FH ^ 2)

theorem quadrilateral_perimeter 
  (EF FH GH : ℝ) 
  (h1 : EF = 12)
  (h2 : FH = 7)
  (h3 : GH = FH) :
  EF + FH + GH + EG FH = 26 + Real.sqrt 193 :=
by
  rw [h1, h2, h3]
  sorry

end quadrilateral_perimeter_l1710_171077


namespace roots_difference_squared_l1710_171023

theorem roots_difference_squared
  {Φ ϕ : ℝ}
  (hΦ : Φ^2 - Φ - 2 = 0)
  (hϕ : ϕ^2 - ϕ - 2 = 0)
  (h_diff : Φ ≠ ϕ) :
  (Φ - ϕ)^2 = 9 :=
by sorry

end roots_difference_squared_l1710_171023


namespace initial_percentage_acid_l1710_171059

theorem initial_percentage_acid (P : ℝ) (h1 : 27 * P / 100 = 18 * 60 / 100) : P = 40 :=
sorry

end initial_percentage_acid_l1710_171059


namespace time_after_10000_seconds_l1710_171050

def time_add_seconds (h m s : Nat) (t : Nat) : (Nat × Nat × Nat) :=
  let total_seconds := h * 3600 + m * 60 + s + t
  let hours := (total_seconds / 3600) % 24
  let minutes := (total_seconds % 3600) / 60
  let seconds := (total_seconds % 3600) % 60
  (hours, minutes, seconds)

theorem time_after_10000_seconds :
  time_add_seconds 5 45 0 10000 = (8, 31, 40) :=
by
  sorry

end time_after_10000_seconds_l1710_171050


namespace lowest_total_points_l1710_171075

-- Five girls and their respective positions
inductive Girl where
  | Fiona
  | Gertrude
  | Hannah
  | India
  | Janice
  deriving DecidableEq, Repr, Inhabited

open Girl

-- Initial position mapping
def initial_position : Girl → Nat
  | Fiona => 1
  | Gertrude => 2
  | Hannah => 3
  | India => 4
  | Janice => 5

-- Final position mapping
def final_position : Girl → Nat
  | Fiona => 3
  | Gertrude => 2
  | Hannah => 5
  | India => 1
  | Janice => 4

-- Define a function to calculate points for given initial and final positions
def points_awarded (g : Girl) : Nat :=
  initial_position g - final_position g

-- Define a function to calculate the total number of points
def total_points : Nat :=
  points_awarded Fiona + points_awarded Gertrude + points_awarded Hannah + points_awarded India + points_awarded Janice

theorem lowest_total_points : total_points = 5 :=
by
  -- Placeholder to skip the proof steps
  sorry

end lowest_total_points_l1710_171075


namespace matchstick_triangles_l1710_171092

theorem matchstick_triangles (perimeter : ℕ) (h_perimeter : perimeter = 30) : 
  ∃ n : ℕ, n = 17 ∧ 
  (∀ a b c : ℕ, a + b + c = perimeter → a > 0 → b > 0 → c > 0 → 
                a + b > c ∧ a + c > b ∧ b + c > a → 
                a ≤ b ∧ b ≤ c → n = 17) := 
sorry

end matchstick_triangles_l1710_171092


namespace prob_divisible_by_5_of_digits_ending_in_7_l1710_171097

theorem prob_divisible_by_5_of_digits_ending_in_7 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ N % 10 = 7) → (0 : ℚ) = 0 :=
by
  intro N
  sorry

end prob_divisible_by_5_of_digits_ending_in_7_l1710_171097


namespace outer_boundary_diameter_l1710_171045

theorem outer_boundary_diameter (statue_width garden_width path_width fountain_diameter : ℝ) 
  (h_statue : statue_width = 2) 
  (h_garden : garden_width = 10) 
  (h_path : path_width = 8) 
  (h_fountain : fountain_diameter = 12) : 
  2 * ((fountain_diameter / 2 + statue_width) + garden_width + path_width) = 52 :=
by
  sorry

end outer_boundary_diameter_l1710_171045


namespace increasing_m_range_l1710_171040

noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x - 2 * m * x

theorem increasing_m_range (m : ℝ) : 
  (∀ x > 0, (2 * x + 1 / x - 2 * m ≥ 0)) → m ≤ Real.sqrt 2 :=
by
  intros h
  -- Proof steps would go here
  sorry

end increasing_m_range_l1710_171040


namespace swimming_pool_water_remaining_l1710_171039

theorem swimming_pool_water_remaining :
  let initial_water := 500 -- initial water in gallons
  let evaporation_rate := 1.5 -- water loss due to evaporation in gallons/day
  let leak_rate := 0.8 -- water loss due to leak in gallons/day
  let total_days := 20 -- total number of days

  let total_daily_loss := evaporation_rate + leak_rate -- total daily loss in gallons/day
  let total_loss := total_daily_loss * total_days -- total loss over the period in gallons
  let remaining_water := initial_water - total_loss -- remaining water after 20 days in gallons

  remaining_water = 454 :=
by
  sorry

end swimming_pool_water_remaining_l1710_171039


namespace carson_total_seed_fertilizer_l1710_171005

-- Definitions based on the conditions
variable (F S : ℝ)
variable (h_seed : S = 45)
variable (h_relation : S = 3 * F)

-- Theorem stating the total amount of seed and fertilizer used
theorem carson_total_seed_fertilizer : S + F = 60 := by
  -- Use the given conditions to relate and calculate the total
  sorry

end carson_total_seed_fertilizer_l1710_171005


namespace factorization_problem_l1710_171086

theorem factorization_problem (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 1 := 
by
  sorry

end factorization_problem_l1710_171086


namespace susan_initial_amount_l1710_171053

theorem susan_initial_amount :
  ∃ S: ℝ, (S - (1/5 * S + 1/4 * S + 120) = 1200) → S = 2400 :=
by
  sorry

end susan_initial_amount_l1710_171053


namespace best_fitting_model_is_model1_l1710_171001

noncomputable def model1_R2 : ℝ := 0.98
noncomputable def model2_R2 : ℝ := 0.80
noncomputable def model3_R2 : ℝ := 0.54
noncomputable def model4_R2 : ℝ := 0.35

theorem best_fitting_model_is_model1 :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by
  sorry

end best_fitting_model_is_model1_l1710_171001


namespace range_of_a_l1710_171072

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def strictly_increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (m n : ℝ) (h_even : is_even_function f)
  (h_strict : strictly_increasing_on_nonnegative f)
  (h_m : m = 1/2) (h_f : ∀ x, m ≤ x ∧ x ≤ n → f (a * x + 1) ≤ f 2) :
  a ≤ 2 :=
sorry

end range_of_a_l1710_171072


namespace add_base7_l1710_171068

-- Define the two numbers in base 7 to be added.
def number1 : ℕ := 2 * 7 + 5
def number2 : ℕ := 5 * 7 + 4

-- Define the expected result in base 7.
def expected_sum : ℕ := 1 * 7^2 + 1 * 7 + 2

theorem add_base7 :
  let sum : ℕ := number1 + number2
  sum = expected_sum := sorry

end add_base7_l1710_171068


namespace tangent_line_equation_parallel_to_given_line_l1710_171016

theorem tangent_line_equation_parallel_to_given_line :
  ∃ (x y : ℝ),  y = x^3 - 3 * x^2
    ∧ (3 * x^2 - 6 * x = -3)
    ∧ (y = -2)
    ∧ (3 * x + y - 1 = 0) :=
sorry

end tangent_line_equation_parallel_to_given_line_l1710_171016


namespace number_of_valid_M_l1710_171067

def base_4_representation (M : ℕ) :=
  let c_3 := (M / 256) % 4
  let c_2 := (M / 64) % 4
  let c_1 := (M / 16) % 4
  let c_0 := M % 4
  (256 * c_3) + (64 * c_2) + (16 * c_1) + (4 * c_0)

def base_7_representation (M : ℕ) :=
  let d_3 := (M / 343) % 7
  let d_2 := (M / 49) % 7
  let d_1 := (M / 7) % 7
  let d_0 := M % 7
  (343 * d_3) + (49 * d_2) + (7 * d_1) + d_0

def valid_M (M T : ℕ) :=
  1000 ≤ M ∧ M < 10000 ∧ 
  T = base_4_representation M + base_7_representation M ∧ 
  (T % 100) = ((3 * M) % 100)

theorem number_of_valid_M : 
  ∃ n : ℕ, n = 81 ∧ ∀ M T, valid_M M T → n = (81 : ℕ) :=
sorry

end number_of_valid_M_l1710_171067


namespace x_value_unique_l1710_171046

theorem x_value_unique (x : ℝ) (h : ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) :
  x = 3 / 2 :=
sorry

end x_value_unique_l1710_171046


namespace find_int_less_than_neg3_l1710_171088

theorem find_int_less_than_neg3 : 
  ∃ x ∈ ({-4, -2, 0, 3} : Set Int), x < -3 ∧ x = -4 := 
by
  -- formal proof goes here
  sorry

end find_int_less_than_neg3_l1710_171088


namespace find_lawn_length_l1710_171010

theorem find_lawn_length
  (width_lawn : ℕ)
  (road_width : ℕ)
  (cost_total : ℕ)
  (cost_per_sqm : ℕ)
  (total_area_roads : ℕ)
  (area_roads_length : ℕ)
  (area_roads_breadth : ℕ)
  (length_lawn : ℕ) :
  width_lawn = 60 →
  road_width = 10 →
  cost_total = 3600 →
  cost_per_sqm = 3 →
  total_area_roads = cost_total / cost_per_sqm →
  area_roads_length = road_width * length_lawn →
  area_roads_breadth = road_width * (width_lawn - road_width) →
  total_area_roads = area_roads_length + area_roads_breadth →
  length_lawn = 70 :=
by
  intros h_width_lawn h_road_width h_cost_total h_cost_per_sqm h_total_area_roads h_area_roads_length h_area_roads_breadth h_total_area_roads_eq
  sorry

end find_lawn_length_l1710_171010


namespace spaceship_journey_time_l1710_171099

theorem spaceship_journey_time
  (initial_travel_1 : ℕ)
  (first_break : ℕ)
  (initial_travel_2 : ℕ)
  (second_break : ℕ)
  (travel_per_segment : ℕ)
  (break_per_segment : ℕ)
  (total_break_time : ℕ)
  (remaining_break_time : ℕ)
  (num_segments : ℕ)
  (total_travel_time : ℕ)
  (total_time : ℕ) :
  initial_travel_1 = 10 →
  first_break = 3 →
  initial_travel_2 = 10 →
  second_break = 1 →
  travel_per_segment = 11 →
  break_per_segment = 1 →
  total_break_time = 8 →
  remaining_break_time = total_break_time - (first_break + second_break) →
  num_segments = remaining_break_time / break_per_segment →
  total_travel_time = initial_travel_1 + initial_travel_2 + (num_segments * travel_per_segment) →
  total_time = total_travel_time + total_break_time →
  total_time = 72 :=
by
  intros
  sorry

end spaceship_journey_time_l1710_171099


namespace proof_problem_l1710_171098

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end proof_problem_l1710_171098


namespace smallest_yellow_marbles_l1710_171028

def total_marbles (n : ℕ) := n

def blue_marbles (n : ℕ) := n / 3

def red_marbles (n : ℕ) := n / 4

def green_marbles := 6

def yellow_marbles (n : ℕ) := n - (blue_marbles n + red_marbles n + green_marbles)

theorem smallest_yellow_marbles (n : ℕ) (hn : n % 12 = 0) (blue : blue_marbles n = n / 3)
  (red : red_marbles n = n / 4) (green : green_marbles = 6) :
  yellow_marbles n = 4 ↔ n = 24 :=
by sorry

end smallest_yellow_marbles_l1710_171028


namespace monthly_income_ratio_l1710_171051

noncomputable def A_annual_income : ℝ := 571200
noncomputable def C_monthly_income : ℝ := 17000
noncomputable def B_monthly_income : ℝ := C_monthly_income * 1.12
noncomputable def A_monthly_income : ℝ := A_annual_income / 12

theorem monthly_income_ratio :
  (A_monthly_income / B_monthly_income) = 2.5 :=
by
  sorry

end monthly_income_ratio_l1710_171051


namespace largest_digit_divisible_by_9_l1710_171073

theorem largest_digit_divisible_by_9 : ∀ (B : ℕ), B < 10 → (∃ n : ℕ, 9 * n = 5 + B + 4 + 8 + 6 + 1) → B = 9 := by
  sorry

end largest_digit_divisible_by_9_l1710_171073


namespace remainder_8_pow_2023_div_5_l1710_171036

-- Definition for modulo operation
def mod_five (a : Nat) : Nat := a % 5

-- Key theorem to prove
theorem remainder_8_pow_2023_div_5 : mod_five (8 ^ 2023) = 2 :=
by
  sorry -- This is where the proof would go, but it's not required per the instructions

end remainder_8_pow_2023_div_5_l1710_171036


namespace minimum_games_pasha_wins_l1710_171080

noncomputable def pasha_initial_money : Nat := 9 -- Pasha has a single-digit amount
noncomputable def igor_initial_money : Nat := 1000 -- Igor has a four-digit amount
noncomputable def pasha_final_money : Nat := 100 -- Pasha has a three-digit amount
noncomputable def igor_final_money : Nat := 99 -- Igor has a two-digit amount

theorem minimum_games_pasha_wins :
  ∃ (games_won_by_pasha : Nat), 
    (games_won_by_pasha >= 7) ∧
    (games_won_by_pasha <= 7) := sorry

end minimum_games_pasha_wins_l1710_171080


namespace solution_set_l1710_171004

/-- Definition: integer solutions (a, b, c) with c ≤ 94 that satisfy the equation -/
def int_solutions (a b c : ℤ) : Prop :=
  c ≤ 94 ∧ (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c

/-- Proposition: The integer solutions (a, b, c) that satisfy the equation are exactly these -/
theorem solution_set :
  { (a, b, c) : ℤ × ℤ × ℤ  | int_solutions a b c } =
  { (3, 7, 41), (4, 6, 44), (5, 5, 45), (6, 4, 44), (7, 3, 41) } :=
by
  sorry

end solution_set_l1710_171004


namespace quadratic_other_root_l1710_171025

theorem quadratic_other_root (k : ℝ) (h : ∀ x, x^2 - k*x - 4 = 0 → x = 2 ∨ x = -2) :
  ∀ x, x^2 - k*x - 4 = 0 → x = -2 :=
by
  sorry

end quadratic_other_root_l1710_171025
