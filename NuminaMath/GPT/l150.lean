import Mathlib

namespace files_to_organize_in_afternoon_l150_15058

-- Defining the given conditions.
def initial_files : ℕ := 60
def files_organized_in_the_morning : ℕ := initial_files / 2
def missing_files_in_the_afternoon : ℕ := 15

-- The theorem to prove:
theorem files_to_organize_in_afternoon : 
  files_organized_in_the_morning + missing_files_in_the_afternoon = initial_files / 2 →
  ∃ afternoon_files : ℕ, 
    afternoon_files = (initial_files - files_organized_in_the_morning) - missing_files_in_the_afternoon :=
by
  -- Proof will go here, skipping with sorry for now.
  sorry

end files_to_organize_in_afternoon_l150_15058


namespace find_polynomials_g_l150_15010

-- Define functions f and proof target is g
def f (x : ℝ) : ℝ := x ^ 2

-- g is defined as an unknown polynomial with some constraints
variable (g : ℝ → ℝ)

-- The proof problem stating that if f(g(x)) = 9x^2 + 12x + 4, 
-- then g(x) = 3x + 2 or g(x) = -3x - 2
theorem find_polynomials_g (h : ∀ x : ℝ, f (g x) = 9 * x ^ 2 + 12 * x + 4) :
  (∀ x : ℝ, g x = 3 * x + 2) ∨ (∀ x : ℝ, g x = -3 * x - 2) := 
by
  sorry

end find_polynomials_g_l150_15010


namespace find_S_l150_15022

theorem find_S :
  (1/4 : ℝ) * (1/6 : ℝ) * S = (1/5 : ℝ) * (1/8 : ℝ) * 160 → S = 96 :=
by
  intro h
  -- Proof is omitted
  sorry 

end find_S_l150_15022


namespace smallest_number_of_eggs_l150_15081

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l150_15081


namespace sum_of_non_domain_elements_l150_15090

theorem sum_of_non_domain_elements :
    let f (x : ℝ) : ℝ := 1 / (1 + 1 / (1 + 1 / (1 + 1 / x)))
    let is_not_in_domain (x : ℝ) := x = 0 ∨ x = -1 ∨ x = -1/2 ∨ x = -2/3
    (0 : ℝ) + (-1) + (-1/2) + (-2/3) = -19/6 :=
by 
  sorry

end sum_of_non_domain_elements_l150_15090


namespace sum_of_first_3m_terms_l150_15029

variable {a : ℕ → ℝ}   -- The arithmetic sequence
variable {S : ℕ → ℝ}   -- The sum of the first n terms of the sequence

def arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  S m = 30 ∧ S (2 * m) = 100 ∧ S (3 * m) = 170

theorem sum_of_first_3m_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence_sum a S m :=
by
  sorry

end sum_of_first_3m_terms_l150_15029


namespace degree_of_d_l150_15062

noncomputable def f : Polynomial ℝ := sorry
noncomputable def d : Polynomial ℝ := sorry
noncomputable def q : Polynomial ℝ := sorry
noncomputable def r : Polynomial ℝ := 5 * Polynomial.X^2 + 3 * Polynomial.X - 8

axiom deg_f : f.degree = 15
axiom deg_q : q.degree = 7
axiom deg_r : r.degree = 2
axiom poly_div : f = d * q + r

theorem degree_of_d : d.degree = 8 :=
by
  sorry

end degree_of_d_l150_15062


namespace Todd_ate_5_cupcakes_l150_15031

theorem Todd_ate_5_cupcakes (original_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) (remaining_cupcakes : ℕ) :
  original_cupcakes = 50 ∧ packages = 9 ∧ cupcakes_per_package = 5 ∧ remaining_cupcakes = packages * cupcakes_per_package →
  original_cupcakes - remaining_cupcakes = 5 :=
by
  sorry

end Todd_ate_5_cupcakes_l150_15031


namespace absolute_value_expression_l150_15071

theorem absolute_value_expression {x : ℤ} (h : x = 2024) :
  abs (abs (abs x - x) - abs x) = 0 :=
by
  sorry

end absolute_value_expression_l150_15071


namespace line_eqn_with_given_conditions_l150_15055

theorem line_eqn_with_given_conditions : 
  ∃(m c : ℝ), (∀ x y : ℝ, y = m*x + c → x + y - 3 = 0) ↔ 
  ∀ x y, x + y = 3 :=
sorry

end line_eqn_with_given_conditions_l150_15055


namespace evaluate_expression_l150_15057

theorem evaluate_expression :
  let a := (1 : ℚ) / 5
  let b := (1 : ℚ) / 3
  let c := (3 : ℚ) / 7
  let d := (1 : ℚ) / 4
  (a + b) / (c - d) = 224 / 75 := by
sorry

end evaluate_expression_l150_15057


namespace cos_half_angle_quadrant_l150_15083

theorem cos_half_angle_quadrant 
  (α : ℝ) 
  (h1 : 25 * Real.sin α ^ 2 + Real.sin α - 24 = 0) 
  (h2 : π / 2 < α ∧ α < π) 
  : Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 :=
by
  sorry

end cos_half_angle_quadrant_l150_15083


namespace quadratic_inequality_solution_set_l150_15007

theorem quadratic_inequality_solution_set {x : ℝ} : 
  x^2 < x + 6 ↔ (-2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_set_l150_15007


namespace min_cubes_are_three_l150_15066

/-- 
  A toy construction set consists of cubes, each with one button on one side and socket holes on the other five sides.
  Prove that the minimum number of such cubes required to build a structure where all buttons are hidden, and only the sockets are visible is 3.
--/

def min_cubes_to_hide_buttons (num_cubes : ℕ) : Prop :=
  num_cubes = 3

theorem min_cubes_are_three : ∃ (n : ℕ), (∀ (num_buttons : ℕ), min_cubes_to_hide_buttons num_buttons) :=
by
  use 3
  sorry

end min_cubes_are_three_l150_15066


namespace smallest_integer_ratio_l150_15069

theorem smallest_integer_ratio (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (h_sum : x + y = 120) (h_even : x % 2 = 0) : ∃ (k : ℕ), k = x / y ∧ k = 1 :=
by
  sorry

end smallest_integer_ratio_l150_15069


namespace amount_paid_is_correct_l150_15030

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l150_15030


namespace inequality_holds_for_all_reals_l150_15040

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l150_15040


namespace flower_bed_width_l150_15045

theorem flower_bed_width (length area : ℝ) (h_length : length = 4) (h_area : area = 143.2) :
  area / length = 35.8 :=
by
  sorry

end flower_bed_width_l150_15045


namespace three_integers_same_parity_l150_15005

theorem three_integers_same_parity (a b c : ℤ) : 
  (∃ i j, i ≠ j ∧ (i = a ∨ i = b ∨ i = c) ∧ (j = a ∨ j = b ∨ j = c) ∧ (i % 2 = j % 2)) :=
by
  sorry

end three_integers_same_parity_l150_15005


namespace max_groups_eq_one_l150_15032

-- Defining the conditions 
def eggs : ℕ := 16
def marbles : ℕ := 3
def rubber_bands : ℕ := 5

-- The theorem statement
theorem max_groups_eq_one
  (h1 : eggs = 16)
  (h2 : marbles = 3)
  (h3 : rubber_bands = 5) :
  ∀ g : ℕ, (g ≤ eggs ∧ g ≤ marbles ∧ g ≤ rubber_bands) →
  (eggs % g = 0) ∧ (marbles % g = 0) ∧ (rubber_bands % g = 0) →
  g = 1 :=
by
  sorry

end max_groups_eq_one_l150_15032


namespace fraction_product_simplified_l150_15096

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l150_15096


namespace sum_zero_implies_product_terms_nonpositive_l150_15059

theorem sum_zero_implies_product_terms_nonpositive (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 := 
by 
  sorry

end sum_zero_implies_product_terms_nonpositive_l150_15059


namespace total_boys_l150_15082

theorem total_boys (T F : ℕ) 
  (avg_all : 37 * T = 39 * 110 + 15 * F) 
  (total_eq : T = 110 + F) : 
  T = 120 := 
sorry

end total_boys_l150_15082


namespace round_robin_points_change_l150_15052

theorem round_robin_points_change (n : ℕ) (athletes : Finset ℕ) (tournament1_scores tournament2_scores : ℕ → ℚ) :
  Finset.card athletes = 2 * n →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) ≥ n) →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) = n) :=
by
  sorry

end round_robin_points_change_l150_15052


namespace nonnegative_interval_l150_15074

theorem nonnegative_interval (x : ℝ) : 
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3) ≥ 0 ↔ (x ≥ 0 ∧ x < 3) :=
by sorry

end nonnegative_interval_l150_15074


namespace lucy_initial_balance_l150_15002

theorem lucy_initial_balance (final_balance deposit withdrawal : Int) 
  (h_final : final_balance = 76)
  (h_deposit : deposit = 15)
  (h_withdrawal : withdrawal = 4) :
  let initial_balance := final_balance + withdrawal - deposit
  initial_balance = 65 := 
by
  sorry

end lucy_initial_balance_l150_15002


namespace sandy_total_money_l150_15050

-- Definitions based on conditions
def X_initial (X : ℝ) : Prop := 
  X - 0.30 * X = 210

def watch_cost : ℝ := 50

-- Question translated into a proof goal
theorem sandy_total_money (X : ℝ) (h : X_initial X) : 
  X + watch_cost = 350 := by
  sorry

end sandy_total_money_l150_15050


namespace average_age_is_25_l150_15067

theorem average_age_is_25 (A B C : ℝ) (h_avg_ac : (A + C) / 2 = 29) (h_b : B = 17) :
  (A + B + C) / 3 = 25 := 
  by
    sorry

end average_age_is_25_l150_15067


namespace area_of_triangle_AEC_l150_15088

theorem area_of_triangle_AEC (BE EC : ℝ) (h_ratio : BE / EC = 3 / 2) (area_abe : ℝ) (h_area_abe : area_abe = 27) : 
  ∃ area_aec, area_aec = 18 :=
by
  sorry

end area_of_triangle_AEC_l150_15088


namespace max_area_rectangle_l150_15016

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end max_area_rectangle_l150_15016


namespace intersection_A_B_l150_15009

-- Definitions based on conditions
variable (U : Set Int) (A B : Set Int)

#check Set

-- Given conditions
def U_def : Set Int := {-1, 3, 5, 7, 9}
def compl_U_A : Set Int := {-1, 9}
def B_def : Set Int := {3, 7, 9}

-- A is defined as the set difference of U and the complement of A in U
def A_def : Set Int := { x | x ∈ U_def ∧ ¬ (x ∈ compl_U_A) }

-- Theorem stating the intersection of A and B equals {3, 7}
theorem intersection_A_B : A_def ∩ B_def = {3, 7} :=
by
  -- Here would be the proof block, but we add 'sorry' to indicate it is unfinished.
  sorry

end intersection_A_B_l150_15009


namespace sqrt_product_eq_l150_15061

theorem sqrt_product_eq :
  (Int.sqrt (2 ^ 2 * 3 ^ 4) : ℤ) = 18 :=
sorry

end sqrt_product_eq_l150_15061


namespace smallest_n_l150_15077

theorem smallest_n (n : ℕ) : 
  (n % 6 = 2) ∧ (n % 7 = 3) ∧ (n % 8 = 4) → n = 8 :=
  by sorry

end smallest_n_l150_15077


namespace least_possible_value_of_c_l150_15054

theorem least_possible_value_of_c (a b c : ℕ) 
  (h1 : a + b + c = 60) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : b = a + 13) : c = 45 :=
sorry

end least_possible_value_of_c_l150_15054


namespace largest_possible_p_l150_15026

theorem largest_possible_p (m n p : ℕ) (h1 : m > 2) (h2 : n > 2) (h3 : p > 2) (h4 : gcd m n = 1) (h5 : gcd n p = 1) (h6 : gcd m p = 1)
  (h7 : (1/m : ℚ) + (1/n : ℚ) + (1/p : ℚ) = 1/2) : p ≤ 42 :=
by sorry

end largest_possible_p_l150_15026


namespace trig_identity_l150_15046

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3 / 4 :=
by
  sorry

end trig_identity_l150_15046


namespace distance_between_points_l150_15034

theorem distance_between_points (a b c d m k : ℝ) 
  (h1 : b = 2 * m * a + k) (h2 : d = -m * c + k) : 
  (Real.sqrt ((c - a)^2 + (d - b)^2)) = Real.sqrt ((1 + m^2) * (c - a)^2) := 
by {
  sorry
}

end distance_between_points_l150_15034


namespace find_matrix_M_l150_15027

theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) (h : M^3 - 3 • M^2 + 4 • M = ![![6, 12], ![3, 6]]) :
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_M_l150_15027


namespace elaine_rent_percentage_l150_15038

theorem elaine_rent_percentage (E : ℝ) (hE : E > 0) :
  let rent_last_year := 0.20 * E
  let earnings_this_year := 1.25 * E
  let rent_this_year := 0.30 * earnings_this_year
  (rent_this_year / rent_last_year) * 100 = 187.5 :=
by
  sorry

end elaine_rent_percentage_l150_15038


namespace calculate_F_5_f_6_l150_15035

def f (a : ℤ) : ℤ := a + 3

def F (a b : ℤ) : ℤ := b^3 - 2 * a

theorem calculate_F_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end calculate_F_5_f_6_l150_15035


namespace volume_of_prism_l150_15043

theorem volume_of_prism 
  (a b c : ℝ) 
  (h₁ : a * b = 51) 
  (h₂ : b * c = 52) 
  (h₃ : a * c = 53) 
  : (a * b * c) = 374 :=
by sorry

end volume_of_prism_l150_15043


namespace angelina_speed_l150_15073

theorem angelina_speed (v : ℝ) (h₁ : ∀ t : ℝ, t = 100 / v) (h₂ : ∀ t : ℝ, t = 180 / (2 * v)) 
  (h₃ : ∀ d t : ℝ, 100 / v - 40 = 180 / (2 * v)) : 
  2 * v = 1 / 2 :=
by
  sorry

end angelina_speed_l150_15073


namespace greatest_sum_consecutive_integers_l150_15020

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_l150_15020


namespace charlie_metal_storage_l150_15072

theorem charlie_metal_storage (total_needed : ℕ) (amount_to_buy : ℕ) (storage : ℕ) 
    (h1 : total_needed = 635) 
    (h2 : amount_to_buy = 359) 
    (h3 : total_needed = storage + amount_to_buy) : 
    storage = 276 := 
sorry

end charlie_metal_storage_l150_15072


namespace problem_l150_15006

theorem problem (a b : ℕ)
  (ha : a = 2) 
  (hb : b = 121) 
  (h_minPrime : ∀ n, n < a → ¬ (∀ d, d ∣ n → d = 1 ∨ d = n))
  (h_threeDivisors : ∀ n, n < 150 → ∀ d, d ∣ n → d = 1 ∨ d = n → n = 121) :
  a + b = 123 := by
  sorry

end problem_l150_15006


namespace watermelon_and_banana_weight_l150_15094

variables (w b : ℕ)
variables (h1 : 2 * w + b = 8100)
variables (h2 : 2 * w + 3 * b = 8300)

theorem watermelon_and_banana_weight (Hw : w = 4000) (Hb : b = 100) :
  2 * w + b = 8100 ∧ 2 * w + 3 * b = 8300 :=
by
  sorry

end watermelon_and_banana_weight_l150_15094


namespace new_tax_rate_l150_15093

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end new_tax_rate_l150_15093


namespace trip_distance_l150_15098

theorem trip_distance (D : ℝ) (t1 t2 : ℝ) :
  (30 / 60 = t1) →
  (70 / 35 = t2) →
  (t1 + t2 = 2.5) →
  (40 = D / (t1 + t2)) →
  D = 100 :=
by
  intros h1 h2 h3 h4
  sorry

end trip_distance_l150_15098


namespace solve_equation_l150_15086

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end solve_equation_l150_15086


namespace quadratic_function_condition_l150_15041

theorem quadratic_function_condition (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
  sorry

end quadratic_function_condition_l150_15041


namespace lines_intersect_at_common_point_iff_l150_15018

theorem lines_intersect_at_common_point_iff (a b : ℝ) :
  (∃ x y : ℝ, a * x + 2 * b * y + 3 * (a + b + 1) = 0 ∧ 
               b * x + 2 * (a + b + 1) * y + 3 * a = 0 ∧ 
               (a + b + 1) * x + 2 * a * y + 3 * b = 0) ↔ 
  a + b = -1/2 :=
by
  sorry

end lines_intersect_at_common_point_iff_l150_15018


namespace river_width_l150_15097

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : depth = 5 → flow_rate_kmph = 2 → volume_per_minute = 5833.333333333333 → 
  (volume_per_minute / ((flow_rate_kmph * 1000 / 60) * depth) = 35) :=
by 
  intros h_depth h_flow_rate h_volume
  sorry

end river_width_l150_15097


namespace complex_coordinates_l150_15079

theorem complex_coordinates (i : ℂ) (z : ℂ) (h : i^2 = -1) (h_z : z = (1 + 2 * i^3) / (2 + i)) :
  z = -i := 
by {
  sorry
}

end complex_coordinates_l150_15079


namespace fraction_sum_equals_decimal_l150_15095

theorem fraction_sum_equals_decimal : 
  (3 / 30 + 9 / 300 + 27 / 3000 = 0.139) :=
by sorry

end fraction_sum_equals_decimal_l150_15095


namespace mo_hot_chocolate_l150_15001

noncomputable def cups_of_hot_chocolate (total_drinks: ℕ) (extra_tea: ℕ) (non_rainy_days: ℕ) (tea_per_day: ℕ) : ℕ :=
  let tea_drinks := non_rainy_days * tea_per_day 
  let chocolate_drinks := total_drinks - tea_drinks 
  (extra_tea - chocolate_drinks)

theorem mo_hot_chocolate :
  cups_of_hot_chocolate 36 14 5 5 = 11 :=
by
  sorry

end mo_hot_chocolate_l150_15001


namespace car_distance_l150_15076

theorem car_distance (time_am_18 : ℕ) (time_car_48 : ℕ) (h : time_am_18 = time_car_48) : 
  let distance_am_18 := 18
  let distance_car_48 := 48
  let total_distance_am := 675
  let distance_ratio := (distance_am_18 : ℝ) / (distance_car_48 : ℝ)
  let distance_car := (total_distance_am : ℝ) * (distance_car_48 : ℝ) / (distance_am_18 : ℝ)
  distance_car = 1800 :=
by
  sorry

end car_distance_l150_15076


namespace abs_diff_31st_term_l150_15056

-- Define the sequences C and D
def C (n : ℕ) : ℤ := 40 + 20 * (n - 1)
def D (n : ℕ) : ℤ := 40 - 20 * (n - 1)

-- Question: What is the absolute value of the difference between the 31st term of C and D?
theorem abs_diff_31st_term : |C 31 - D 31| = 1200 := by
  sorry

end abs_diff_31st_term_l150_15056


namespace exists_common_element_l150_15064

variable (S : Fin 2011 → Set ℤ)
variable (h1 : ∀ i, (S i).Nonempty)
variable (h2 : ∀ i j, (S i ∩ S j).Nonempty)

theorem exists_common_element :
  ∃ a : ℤ, ∀ i, a ∈ S i :=
by {
  sorry
}

end exists_common_element_l150_15064


namespace smallest_h_divisible_by_primes_l150_15091

theorem smallest_h_divisible_by_primes :
  ∃ h k : ℕ, (∀ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p > 8 ∧ q > 11 ∧ r > 24 → (h + k) % (p * q * r) = 0 ∧ h = 1) :=
by
  sorry

end smallest_h_divisible_by_primes_l150_15091


namespace problem_solution_l150_15023

-- Given non-zero numbers x and y such that x = 1 / y,
-- prove that (2x - 1/x) * (y - 1/y) = -2x^2 + y^2.
theorem problem_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 :=
by
  sorry

end problem_solution_l150_15023


namespace choir_girls_count_l150_15084

noncomputable def number_of_girls_in_choir (o b t c b_boys : ℕ) : ℕ :=
  c - b_boys

theorem choir_girls_count (o b t b_boys : ℕ) (h1 : o = 20) (h2 : b = 2 * o) (h3 : t = 88)
  (h4 : b_boys = 12) : number_of_girls_in_choir o b t (t - (o + b)) b_boys = 16 :=
by
  sorry

end choir_girls_count_l150_15084


namespace area_excluding_hole_correct_l150_15089

def large_rectangle_area (x: ℝ) : ℝ :=
  4 * (x + 7) * (x + 5)

def hole_area (x: ℝ) : ℝ :=
  9 * (2 * x - 3) * (x - 2)

def area_excluding_hole (x: ℝ) : ℝ :=
  large_rectangle_area x - hole_area x

theorem area_excluding_hole_correct (x: ℝ) :
  area_excluding_hole x = -14 * x^2 + 111 * x + 86 :=
by
  -- The proof is omitted
  sorry

end area_excluding_hole_correct_l150_15089


namespace inheritance_amount_l150_15037

-- Definitions based on conditions given
def inheritance (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_federal := x - federal_tax
  let state_tax := 0.15 * remaining_after_federal
  let total_tax := federal_tax + state_tax
  total_tax = 15000

-- The statement to be proven
theorem inheritance_amount (x : ℝ) (hx : inheritance x) : x = 41379 :=
by
  -- Proof goes here
  sorry

end inheritance_amount_l150_15037


namespace cheese_cost_l150_15087

theorem cheese_cost (bread_cost cheese_cost total_paid total_change coin_change nickels_value : ℝ) 
                    (quarter dime nickels_count : ℕ)
                    (h1 : bread_cost = 4.20)
                    (h2 : total_paid = 7.00)
                    (h3 : quarter = 1)
                    (h4 : dime = 1)
                    (h5 : nickels_count = 8)
                    (h6 : coin_change = (quarter * 0.25) + (dime * 0.10) + (nickels_count * 0.05))
                    (h7 : total_change = total_paid - bread_cost)
                    (h8 : cheese_cost = total_change - coin_change) :
                    cheese_cost = 2.05 :=
by {
    sorry
}

end cheese_cost_l150_15087


namespace minimum_b_l150_15068

theorem minimum_b (k a b : ℝ) (h1 : 1 < k) (h2 : k < a) (h3 : a < b)
  (h4 : ¬(k + a > b)) (h5 : ¬(1/a + 1/b > 1/k)) :
  2 * k ≤ b :=
by
  sorry

end minimum_b_l150_15068


namespace quadratic_b_value_l150_15024

theorem quadratic_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b * x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 :=
by 
  sorry

end quadratic_b_value_l150_15024


namespace two_n_plus_m_value_l150_15051

theorem two_n_plus_m_value (n m : ℤ) :
  3 * n - m < 5 ∧ n + m > 26 ∧ 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
sorry

end two_n_plus_m_value_l150_15051


namespace problem_statement_l150_15042

def f (x : ℝ) : ℝ := x^2 - 3 * x + 6

def g (x : ℝ) : ℝ := x + 4

theorem problem_statement : f (g 3) - g (f 3) = 24 := by
  sorry

end problem_statement_l150_15042


namespace group_d_forms_triangle_l150_15036

-- Definitions for the stick lengths in each group
def group_a := (1, 2, 6)
def group_b := (2, 2, 4)
def group_c := (1, 2, 3)
def group_d := (2, 3, 4)

-- Statement to prove that Group D can form a triangle
theorem group_d_forms_triangle (a b c : ℕ) : a = 2 → b = 3 → c = 4 → a + b > c ∧ a + c > b ∧ b + c > a := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  apply And.intro
  sorry
  apply And.intro
  sorry
  sorry

end group_d_forms_triangle_l150_15036


namespace midpoint_on_hyperbola_l150_15099

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l150_15099


namespace neither_happy_nor_sad_boys_is_5_l150_15004

-- Define the total number of children
def total_children := 60

-- Define the number of happy children
def happy_children := 30

-- Define the number of sad children
def sad_children := 10

-- Define the number of neither happy nor sad children
def neither_happy_nor_sad_children := 20

-- Define the number of boys
def boys := 17

-- Define the number of girls
def girls := 43

-- Define the number of happy boys
def happy_boys := 6

-- Define the number of sad girls
def sad_girls := 4

-- Define the number of neither happy nor sad boys
def neither_happy_nor_sad_boys := boys - (happy_boys + (sad_children - sad_girls))

theorem neither_happy_nor_sad_boys_is_5 :
  neither_happy_nor_sad_boys = 5 :=
by
  -- This skips the proof
  sorry

end neither_happy_nor_sad_boys_is_5_l150_15004


namespace problem1_problem2_l150_15063

section ProofProblems

-- Definitions for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Prove that n! = binom(n, k) * k! * (n-k)!
theorem problem1 (n k : ℕ) : n.factorial = binom n k * k.factorial * (n - k).factorial :=
by sorry

-- Problem 2: Prove that binom(n, k) = binom(n-1, k) + binom(n-1, k-1)
theorem problem2 (n k : ℕ) : binom n k = binom (n-1) k + binom (n-1) (k-1) :=
by sorry

end ProofProblems

end problem1_problem2_l150_15063


namespace solve_equation_1_solve_equation_2_l150_15008

theorem solve_equation_1 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 + 5 * x + 3 = 0 ↔ (x = -1 ∨ x = -3/2) :=
by sorry

end solve_equation_1_solve_equation_2_l150_15008


namespace hunter_time_comparison_l150_15070

-- Definitions for time spent in swamp, forest, and highway
variables {a b c : ℝ}

-- Given conditions
-- 1. Total time equation
#check a + b + c = 4

-- 2. Total distance equation
#check 2 * a + 4 * b + 6 * c = 17

-- Prove that the hunter spent more time on the highway than in the swamp
theorem hunter_time_comparison (h1 : a + b + c = 4) (h2 : 2 * a + 4 * b + 6 * c = 17) : c > a :=
by sorry

end hunter_time_comparison_l150_15070


namespace number_of_classes_l150_15013

theorem number_of_classes
  (p : ℕ) (s : ℕ) (t : ℕ) (c : ℕ)
  (hp : p = 2) (hs : s = 30) (ht : t = 360) :
  c = t / (p * s) :=
by
  simp [hp, hs, ht]
  sorry

end number_of_classes_l150_15013


namespace increasing_sequence_nec_but_not_suf_l150_15080

theorem increasing_sequence_nec_but_not_suf (a : ℕ → ℝ) :
  (∀ n, abs (a (n + 1)) > a n) → (∀ n, a (n + 1) > a n) ↔ 
  ∃ (n : ℕ), ¬ (abs (a (n + 1)) > a n) ∧ (a (n + 1) > a n) :=
sorry

end increasing_sequence_nec_but_not_suf_l150_15080


namespace Miriam_gave_brother_60_marbles_l150_15014

def Miriam_current_marbles : ℕ := 30
def Miriam_initial_marbles : ℕ := 300
def brother_marbles (B : ℕ) : Prop := B = 60
def sister_marbles (B : ℕ) : ℕ := 2 * B
def friend_marbles : ℕ := 90
def total_given_away_marbles (B : ℕ) : ℕ := B + sister_marbles B + friend_marbles

theorem Miriam_gave_brother_60_marbles (B : ℕ) 
    (h1 : Miriam_current_marbles = 30) 
    (h2 : Miriam_initial_marbles = 300)
    (h3 : total_given_away_marbles B = Miriam_initial_marbles - Miriam_current_marbles) : 
    brother_marbles B :=
by 
    sorry

end Miriam_gave_brother_60_marbles_l150_15014


namespace stopped_clock_more_accurate_l150_15033

theorem stopped_clock_more_accurate (slow_correct_time_frequency : ℕ)
  (stopped_correct_time_frequency : ℕ)
  (h1 : slow_correct_time_frequency = 720)
  (h2 : stopped_correct_time_frequency = 2) :
  stopped_correct_time_frequency > slow_correct_time_frequency / 720 :=
by
  sorry

end stopped_clock_more_accurate_l150_15033


namespace adults_not_wearing_blue_is_10_l150_15012

section JohnsonFamilyReunion

-- Define the number of children
def children : ℕ := 45

-- Define the ratio between adults and children
def adults : ℕ := children / 3

-- Define the ratio of adults who wore blue
def adults_wearing_blue : ℕ := adults / 3

-- Define the number of adults who did not wear blue
def adults_not_wearing_blue : ℕ := adults - adults_wearing_blue

-- Theorem stating the number of adults who did not wear blue
theorem adults_not_wearing_blue_is_10 : adults_not_wearing_blue = 10 :=
by
  -- This is a placeholder for the actual proof
  sorry

end JohnsonFamilyReunion

end adults_not_wearing_blue_is_10_l150_15012


namespace number_of_ways_to_select_starting_lineup_l150_15092

noncomputable def choose (n k : ℕ) : ℕ := 
if h : k ≤ n then Nat.choose n k else 0

theorem number_of_ways_to_select_starting_lineup (n k : ℕ) (h : n = 12) (h1 : k = 5) : 
  12 * choose 11 4 = 3960 := 
by sorry

end number_of_ways_to_select_starting_lineup_l150_15092


namespace max_triangle_perimeter_l150_15060

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end max_triangle_perimeter_l150_15060


namespace yarn_cut_parts_l150_15085

-- Define the given conditions
def total_length : ℕ := 10
def crocheted_parts : ℕ := 3
def crocheted_length : ℕ := 6

-- The main problem statement
theorem yarn_cut_parts (total_length crocheted_parts crocheted_length : ℕ) (h1 : total_length = 10) (h2 : crocheted_parts = 3) (h3 : crocheted_length = 6) :
  (total_length / (crocheted_length / crocheted_parts)) = 5 :=
by
  sorry

end yarn_cut_parts_l150_15085


namespace scientific_notation_of_0_00000012_l150_15075

theorem scientific_notation_of_0_00000012 :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_0_00000012_l150_15075


namespace sum_of_positive_numbers_is_360_l150_15048

variable (x y : ℝ)
variable (h1 : x * y = 50 * (x + y))
variable (h2 : x * y = 75 * (x - y))

theorem sum_of_positive_numbers_is_360 (hx : 0 < x) (hy : 0 < y) : x + y = 360 :=
by sorry

end sum_of_positive_numbers_is_360_l150_15048


namespace exists_a_perfect_power_l150_15011

def is_perfect_power (n : ℕ) : Prop :=
  ∃ b k : ℕ, b > 0 ∧ k ≥ 2 ∧ n = b^k

theorem exists_a_perfect_power :
  ∃ a > 0, ∀ n, 2015 ≤ n ∧ n ≤ 2558 → is_perfect_power (n * a) :=
sorry

end exists_a_perfect_power_l150_15011


namespace cups_of_flour_required_l150_15021

/-- Define the number of cups of sugar and salt required by the recipe. --/
def sugar := 14
def salt := 7
/-- Define the number of cups of flour already added. --/
def flour_added := 2
/-- Define the additional requirement of flour being 3 more cups than salt. --/
def additional_flour_requirement := 3

/-- Main theorem to prove the total amount of flour the recipe calls for. --/
theorem cups_of_flour_required : total_flour = 10 :=
by
  sorry

end cups_of_flour_required_l150_15021


namespace pairs_characterization_l150_15078

noncomputable def valid_pairs (A : ℝ) : Set (ℕ × ℕ) :=
  { p | ∃ x : ℝ, x > 0 ∧ (1 + x) ^ p.1 = (1 + A * x) ^ p.2 }

theorem pairs_characterization (A : ℝ) (hA : A > 1) :
  valid_pairs A = { p | p.2 < p.1 ∧ p.1 < A * p.2 } :=
by
  sorry

end pairs_characterization_l150_15078


namespace part1_part2_l150_15019

-- Define the conditions
def P_condition (a x : ℝ) : Prop := 1 - a / x < 0
def Q_condition (x : ℝ) : Prop := abs (x + 2) < 3

-- First part: Given a = 3, prove the solution set P
theorem part1 (x : ℝ) : P_condition 3 x ↔ 0 < x ∧ x < 3 := by 
  sorry

-- Second part: Prove the range of values for the positive number a
theorem part2 (a : ℝ) (ha : 0 < a) : 
  (∀ x, (P_condition a x → Q_condition x)) → 0 < a ∧ a ≤ 1 := by 
  sorry

end part1_part2_l150_15019


namespace solve_for_x_l150_15003

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 = -2 * x + 10) : x = 3 := 
sorry

end solve_for_x_l150_15003


namespace green_pens_l150_15065

theorem green_pens (blue_pens green_pens : ℕ) (ratio_blue_to_green : blue_pens / green_pens = 4 / 3) (total_blue : blue_pens = 16) : green_pens = 12 :=
by sorry

end green_pens_l150_15065


namespace probability_third_attempt_success_l150_15039

noncomputable def P_xi_eq_3 : ℚ :=
  (4 / 5) * (3 / 4) * (1 / 3)

theorem probability_third_attempt_success :
  P_xi_eq_3 = 1 / 5 := by
  sorry

end probability_third_attempt_success_l150_15039


namespace seating_arrangement_correct_l150_15049

noncomputable def seating_arrangements_around_table : Nat :=
  7

def B_G_next_to_C (A B C D E F G : Prop) (d : Nat) : Prop :=
  d = 48

theorem seating_arrangement_correct : ∃ d, d = 48 := sorry

end seating_arrangement_correct_l150_15049


namespace calculate_y_l150_15047

theorem calculate_y (x : ℤ) (y : ℤ) (h1 : x = 121) (h2 : 2 * x - y = 102) : y = 140 :=
by
  -- Placeholder proof
  sorry

end calculate_y_l150_15047


namespace find_constant_l150_15044

theorem find_constant (c : ℝ) (f : ℝ → ℝ)
  (h : f x = c * x^3 + 19 * x^2 - 4 * c * x + 20)
  (hx : f (-7) = 0) :
  c = 3 :=
sorry

end find_constant_l150_15044


namespace three_consecutive_multiples_sum_l150_15028

theorem three_consecutive_multiples_sum (h1 : Int) (h2 : h1 % 3 = 0) (h3 : Int) (h4 : h3 = h1 - 3) (h5 : Int) (h6 : h5 = h1 - 6) (h7: h1 = 27) : h1 + h3 + h5 = 72 := 
by 
  -- let numbers be n, n-3, n-6 and n = 27
  -- so n + n-3 + n-6 = 27 + 24 + 21 = 72
  sorry

end three_consecutive_multiples_sum_l150_15028


namespace cost_of_one_shirt_l150_15025

theorem cost_of_one_shirt
  (cost_J : ℕ)  -- The cost of one pair of jeans
  (cost_S : ℕ)  -- The cost of one shirt
  (h1 : 3 * cost_J + 2 * cost_S = 69)
  (h2 : 2 * cost_J + 3 * cost_S = 81) :
  cost_S = 21 :=
by
  sorry

end cost_of_one_shirt_l150_15025


namespace total_movies_in_series_l150_15015

def book_count := 4
def total_books_read := 19
def movies_watched := 7
def movies_to_watch := 10

theorem total_movies_in_series : movies_watched + movies_to_watch = 17 := by
  sorry

end total_movies_in_series_l150_15015


namespace simplify_polynomial_l150_15017

theorem simplify_polynomial :
  (3 * x ^ 4 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 10) + (7 * x ^ 5 - 3 * x ^ 4 + x ^ 3 - 7 * x ^ 2 + 2 * x - 2)
  = 7 * x ^ 5 - x ^ 3 - 2 * x ^ 2 - 6 * x + 8 :=
by sorry

end simplify_polynomial_l150_15017


namespace billy_total_tickets_l150_15053

theorem billy_total_tickets :
  let ferris_wheel_rides := 7
  let bumper_car_rides := 3
  let roller_coaster_rides := 4
  let teacups_rides := 5
  let ferris_wheel_cost := 5
  let bumper_car_cost := 6
  let roller_coaster_cost := 8
  let teacups_cost := 4
  let total_ferris_wheel := ferris_wheel_rides * ferris_wheel_cost
  let total_bumper_cars := bumper_car_rides * bumper_car_cost
  let total_roller_coaster := roller_coaster_rides * roller_coaster_cost
  let total_teacups := teacups_rides * teacups_cost
  let total_tickets := total_ferris_wheel + total_bumper_cars + total_roller_coaster + total_teacups
  total_tickets = 105 := 
sorry

end billy_total_tickets_l150_15053


namespace cricket_team_players_l150_15000

-- Define conditions 
def non_throwers (T P : ℕ) : ℕ := P - T
def left_handers (N : ℕ) : ℕ := N / 3
def right_handers_non_thrower (N : ℕ) : ℕ := 2 * N / 3
def total_right_handers (T R : ℕ) : Prop := R = T + right_handers_non_thrower (non_throwers T R)

-- Assume conditions are given
variables (P N R T : ℕ)
axiom hT : T = 37
axiom hR : R = 49
axiom hNonThrower : N = non_throwers T P
axiom hRightHanders : right_handers_non_thrower N = R - T

-- Prove the total number of players is 55
theorem cricket_team_players : P = 55 :=
by
  sorry

end cricket_team_players_l150_15000
