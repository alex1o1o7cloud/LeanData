import Mathlib

namespace distance_between_X_and_Y_l2359_235929

theorem distance_between_X_and_Y 
  (b_walked_distance : ℕ) 
  (time_difference : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_rate : ℕ) 
  (time_bob_walked : ℕ) 
  (distance_when_met : ℕ) 
  (bob_walked_8_miles : b_walked_distance = 8) 
  (one_hour_time_difference : time_difference = 1) 
  (yolanda_3_mph : yolanda_rate = 3) 
  (bob_4_mph : bob_rate = 4) 
  (time_bob_2_hours : time_bob_walked = b_walked_distance / bob_rate)
  : 
  distance_when_met = yolanda_rate * (time_bob_walked + time_difference) + bob_rate * time_bob_walked :=
by
  sorry  -- proof steps

end distance_between_X_and_Y_l2359_235929


namespace original_people_complete_work_in_four_days_l2359_235921

noncomputable def original_people_work_days (P D : ℕ) :=
  (2 * P) * 2 = (1 / 2) * (P * D)

theorem original_people_complete_work_in_four_days (P D : ℕ) (h : original_people_work_days P D) : D = 4 :=
by
  sorry

end original_people_complete_work_in_four_days_l2359_235921


namespace problem_solution_l2359_235977

noncomputable def solve_problem : Prop :=
  ∃ (d : ℝ), 
    (∃ int_part : ℤ, 
        (3 * int_part^2 - 12 * int_part + 9 = 0 ∧ ⌊d⌋ = int_part) ∧
        ∀ frac_part : ℝ,
            (4 * frac_part^3 - 8 * frac_part^2 + 3 * frac_part - 0.5 = 0 ∧ frac_part = d - ⌊d⌋) )
    ∧ (d = 1.375 ∨ d = 3.375)

theorem problem_solution : solve_problem :=
by sorry

end problem_solution_l2359_235977


namespace john_payment_l2359_235928

def camera_value : ℝ := 5000
def weekly_rental_percentage : ℝ := 0.10
def rental_period : ℕ := 4
def friend_contribution_percentage : ℝ := 0.40

theorem john_payment :
  let weekly_rental_fee := camera_value * weekly_rental_percentage
  let total_rental_fee := weekly_rental_fee * rental_period
  let friend_contribution := total_rental_fee * friend_contribution_percentage
  let john_payment := total_rental_fee - friend_contribution
  john_payment = 1200 :=
by
  sorry

end john_payment_l2359_235928


namespace union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l2359_235922

open Set

variables (U : Set ℝ) (A B : Set ℝ) (a : ℝ)

def A_def : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }
def B_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def comp_U_A : Set ℝ := { x | x < 1 ∨ x > 4 }

theorem union_A_B_at_a_3 (h : a = 3) :
  A_def ∪ B_def 3 = { x | 1 ≤ x ∧ x ≤ 5 } :=
sorry

theorem inter_B_compl_A_at_a_3 (h : a = 3) :
  B_def 3 ∩ comp_U_A = { x | 4 < x ∧ x ≤ 5 } :=
sorry

theorem B_subset_A_imp_a_range (h : B_def a ⊆ A_def) :
  1 ≤ a ∧ a ≤ 2 :=
sorry

end union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l2359_235922


namespace sum_abc_l2359_235905

variable {a b c : ℝ}
variables (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
variables (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))

theorem sum_abc (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))
   (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) :
   a + b + c = 1128 / 35 := 
sorry

end sum_abc_l2359_235905


namespace largest_of_seven_consecutive_integers_l2359_235971

theorem largest_of_seven_consecutive_integers (n : ℕ) (h : n > 0) (h_sum : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) = 2222) : (n + 6) = 320 :=
by sorry

end largest_of_seven_consecutive_integers_l2359_235971


namespace log_inequality_l2359_235978

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem log_inequality : a > c ∧ c > b :=
by {
  -- Proof goes here
  sorry
}

end log_inequality_l2359_235978


namespace fourth_term_of_geometric_sequence_is_320_l2359_235912

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end fourth_term_of_geometric_sequence_is_320_l2359_235912


namespace percent_of_number_l2359_235967

theorem percent_of_number (x : ℝ) (h : 18 = 0.75 * x) : x = 24 := by
  sorry

end percent_of_number_l2359_235967


namespace max_profit_at_boundary_l2359_235954

noncomputable def profit (x : ℝ) : ℝ :=
  -50 * (x - 55) ^ 2 + 11250

def within_bounds (x : ℝ) : Prop :=
  40 ≤ x ∧ x ≤ 52

theorem max_profit_at_boundary :
  within_bounds 52 ∧ 
  (∀ x : ℝ, within_bounds x → profit x ≤ profit 52) :=
by
  sorry

end max_profit_at_boundary_l2359_235954


namespace solve_for_q_l2359_235919

theorem solve_for_q (k r q : ℕ) (h1 : 4 / 5 = k / 90) (h2 : 4 / 5 = (k + r) / 105) (h3 : 4 / 5 = (q - r) / 150) : q = 132 := 
  sorry

end solve_for_q_l2359_235919


namespace main_theorem_l2359_235995

-- Define the sets M and N
def M : Set ℝ := { x | 0 < x ∧ x < 10 }
def N : Set ℝ := { x | x < -4/3 ∨ x > 3 }

-- Define the complement of N in ℝ
def comp_N : Set ℝ := { x | ¬ (x < -4/3 ∨ x > 3) }

-- The main theorem to be proved
theorem main_theorem : M ∩ comp_N = { x | 0 < x ∧ x ≤ 3 } := 
by
  sorry

end main_theorem_l2359_235995


namespace verify_drawn_numbers_when_x_is_24_possible_values_of_x_l2359_235935

-- Population size and group division
def population_size := 1000
def number_of_groups := 10
def group_size := population_size / number_of_groups

-- Systematic sampling function
def systematic_sample (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 1000

-- Prove the drawn 10 numbers when x = 24
theorem verify_drawn_numbers_when_x_is_24 :
  (∃ drawn_numbers, drawn_numbers = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921]) :=
  sorry

-- Prove possible values of x given last two digits equal to 87
theorem possible_values_of_x (k : ℕ) (h : k < number_of_groups) :
  (∃ x_values, x_values = [87, 54, 21, 88, 55, 22, 89, 56, 23, 90]) :=
  sorry

end verify_drawn_numbers_when_x_is_24_possible_values_of_x_l2359_235935


namespace album_pages_l2359_235994

variable (x y : ℕ)

theorem album_pages :
  (20 * x < y) ∧
  (23 * x > y) ∧
  (21 * x + y = 500) →
  x = 12 := by
  sorry

end album_pages_l2359_235994


namespace larger_volume_of_rotated_rectangle_l2359_235938

-- Definitions based on the conditions
def length : ℝ := 4
def width : ℝ := 3

-- Problem statement: Proving the volume of the larger geometric solid
theorem larger_volume_of_rotated_rectangle :
  max (Real.pi * (width ^ 2) * length) (Real.pi * (length ^ 2) * width) = 48 * Real.pi :=
by
  sorry

end larger_volume_of_rotated_rectangle_l2359_235938


namespace canoe_trip_shorter_l2359_235961

def lake_diameter : ℝ := 2
def pi_value : ℝ := 3.14

theorem canoe_trip_shorter : (2 * pi_value * (lake_diameter / 2) - lake_diameter) = 4.28 :=
by
  sorry

end canoe_trip_shorter_l2359_235961


namespace remainder_7n_mod_4_l2359_235996

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l2359_235996


namespace angle_DGO_is_50_degrees_l2359_235975

theorem angle_DGO_is_50_degrees
  (triangle_DOG : Type)
  (D G O : triangle_DOG)
  (angle_DOG : ℝ)
  (angle_DGO : ℝ)
  (angle_OGD : ℝ)
  (bisect : Prop) :

  angle_DGO = 50 := 
by
  -- Conditions
  have h1 : angle_DGO = angle_DOG := sorry
  have h2 : angle_DOG = 40 := sorry
  have h3 : bisect := sorry
  -- Goal
  sorry

end angle_DGO_is_50_degrees_l2359_235975


namespace num_ordered_pairs_l2359_235948

theorem num_ordered_pairs : ∃! n : ℕ, n = 4 ∧ 
  ∃ (x y : ℤ), y = (x - 90)^2 - 4907 ∧ 
  (∃ m : ℕ, y = m^2) := 
sorry

end num_ordered_pairs_l2359_235948


namespace problem_ns_k_divisibility_l2359_235908

theorem problem_ns_k_divisibility (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
  (∃ (a b : ℕ), (a = 1 ∨ a = 5) ∧ (b = 1 ∨ b = 5) ∧ a = n ∧ b = k) ↔ 
  n * k ∣ (2^(2^n) + 1) * (2^(2^k) + 1) := 
sorry

end problem_ns_k_divisibility_l2359_235908


namespace part1_part2_l2359_235911

-- Define what a double root equation is
def is_double_root_eq (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ * a + x₁ * b + c = 0 ∧ x₂ = 2 * x₁ ∧ x₂ * x₂ * a + x₂ * b + c = 0

-- Statement for part 1: proving x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_eq 1 (-3) 2 :=
sorry

-- Statement for part 2: finding correct values of a and b for ax^2 + bx - 6 = 0 to be a double root equation with one root 2
theorem part2 : (∃ a b : ℝ, is_double_root_eq a b (-6) ∧ (a = -3 ∧ b = 9) ∨ (a = -3/4 ∧ b = 9/2)) :=
sorry

end part1_part2_l2359_235911


namespace motorcyclists_speeds_l2359_235931

theorem motorcyclists_speeds 
  (distance_AB : ℝ) (distance1 : ℝ) (distance2 : ℝ) (time_diff : ℝ) 
  (x y : ℝ) 
  (h1 : distance_AB = 600) 
  (h2 : distance1 = 250) 
  (h3 : distance2 = 200) 
  (h4 : time_diff = 3)
  (h5 : distance1 / x = distance2 / y)
  (h6 : distance_AB / x + time_diff = distance_AB / y) : 
  x = 50 ∧ y = 40 := 
sorry

end motorcyclists_speeds_l2359_235931


namespace largest_integer_n_l2359_235958

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l2359_235958


namespace exponential_function_passes_through_fixed_point_l2359_235920

theorem exponential_function_passes_through_fixed_point {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  (a^(2 - 2) + 3) = 4 :=
by
  sorry

end exponential_function_passes_through_fixed_point_l2359_235920


namespace smallest_and_largest_x_l2359_235999

theorem smallest_and_largest_x (x : ℝ) :
  (|5 * x - 4| = 29) → ((x = -5) ∨ (x = 6.6)) :=
by
  sorry

end smallest_and_largest_x_l2359_235999


namespace distribute_balls_into_boxes_l2359_235925

theorem distribute_balls_into_boxes : (Nat.choose (5 + 4 - 1) (4 - 1)) = 56 := by
  sorry

end distribute_balls_into_boxes_l2359_235925


namespace passing_marks_l2359_235930

theorem passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.45 * T = P + 30) : 
  P = 240 := 
by
  sorry

end passing_marks_l2359_235930


namespace real_complex_number_l2359_235956

theorem real_complex_number (x : ℝ) (hx1 : x^2 - 3 * x - 3 > 0) (hx2 : x - 3 = 1) : x = 4 :=
by
  sorry

end real_complex_number_l2359_235956


namespace length_BC_fraction_AD_l2359_235973

-- Given
variables {A B C D : Type*} [AddCommGroup D] [Module ℝ D]
variables (A B C D : D)
variables (AB BD AC CD AD BC : ℝ)

-- Conditions
def segment_AD := A + D
def segment_BD := B + D
def segment_AB := A + B
def segment_CD := C + D
def segment_AC := A + C
def relation_AB_BD : AB = 3 * BD := sorry
def relation_AC_CD : AC = 5 * CD := sorry

-- Proof
theorem length_BC_fraction_AD :
  BC = (1/12) * AD :=
sorry

end length_BC_fraction_AD_l2359_235973


namespace product_of_local_and_absolute_value_l2359_235936

def localValue (n : ℕ) (digit : ℕ) : ℕ :=
  match n with
  | 564823 =>
    match digit with
    | 4 => 4000
    | _ => 0 -- only defining for digit 4 as per problem
  | _ => 0 -- only case for 564823 is considered

def absoluteValue (x : ℤ) : ℤ := if x < 0 then -x else x

theorem product_of_local_and_absolute_value:
  localValue 564823 4 * absoluteValue 4 = 16000 :=
by
  sorry

end product_of_local_and_absolute_value_l2359_235936


namespace find_ratio_l2359_235986

variables (a b c d : ℝ)

def condition1 : Prop := a / b = 5
def condition2 : Prop := b / c = 1 / 4
def condition3 : Prop := c^2 / d = 16

theorem find_ratio (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c d) :
  d / a = 1 / 25 :=
sorry

end find_ratio_l2359_235986


namespace vines_painted_l2359_235962

-- Definitions based on the conditions in the problem statement
def time_per_lily : ℕ := 5
def time_per_rose : ℕ := 7
def time_per_orchid : ℕ := 3
def time_per_vine : ℕ := 2
def total_time_spent : ℕ := 213
def lilies_painted : ℕ := 17
def roses_painted : ℕ := 10
def orchids_painted : ℕ := 6

-- The theorem to prove the number of vines painted
theorem vines_painted (vines_painted : ℕ) : 
  213 = (17 * 5) + (10 * 7) + (6 * 3) + (vines_painted * 2) → 
  vines_painted = 20 :=
by
  intros h
  sorry

end vines_painted_l2359_235962


namespace find_range_a_l2359_235900

def bounded_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 2 → a * (4 ^ x) + 2 ^ x + 1 ≥ 0

theorem find_range_a :
  ∃ (a : ℝ), bounded_a a ↔ a ≥ -5 / 16 :=
sorry

end find_range_a_l2359_235900


namespace total_cats_l2359_235957

theorem total_cats (a b c d : ℝ) (ht : a = 15.5) (hs : b = 11.6) (hg : c = 24.2) (hr : d = 18.3) :
  a + b + c + d = 69.6 :=
by
  sorry

end total_cats_l2359_235957


namespace smallest_n_divisible_by_23_l2359_235932

theorem smallest_n_divisible_by_23 :
  ∃ n : ℕ, (n^3 + 12 * n^2 + 15 * n + 180) % 23 = 0 ∧
            ∀ m : ℕ, (m^3 + 12 * m^2 + 15 * m + 180) % 23 = 0 → n ≤ m :=
sorry

end smallest_n_divisible_by_23_l2359_235932


namespace squares_difference_l2359_235964

theorem squares_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) : a^2 - b^2 = 15 :=
by
  sorry

end squares_difference_l2359_235964


namespace multiply_same_exponents_l2359_235909

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end multiply_same_exponents_l2359_235909


namespace min_surface_area_of_sphere_l2359_235953

theorem min_surface_area_of_sphere (a b c : ℝ) (volume : ℝ) (height : ℝ) 
  (h_volume : a * b * c = volume) (h_height : c = height) 
  (volume_val : volume = 12) (height_val : height = 4) : 
  ∃ r : ℝ, 4 * π * r^2 = 22 * π := 
by
  sorry

end min_surface_area_of_sphere_l2359_235953


namespace sum_of_consecutive_even_numbers_l2359_235982

theorem sum_of_consecutive_even_numbers (x : ℤ) (h : (x + 2)^2 - x^2 = 84) : x + (x + 2) = 42 :=
by 
  sorry

end sum_of_consecutive_even_numbers_l2359_235982


namespace correct_division_result_l2359_235924

theorem correct_division_result (x : ℝ) (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by
  sorry

end correct_division_result_l2359_235924


namespace proposition_induction_l2359_235944

theorem proposition_induction {P : ℕ → Prop} (h : ∀ n, P n → P (n + 1)) (hn : ¬ P 7) : ¬ P 6 :=
by
  sorry

end proposition_induction_l2359_235944


namespace find_certain_number_l2359_235969

theorem find_certain_number (x : ℤ) (h : ((x / 4) + 25) * 3 = 150) : x = 100 :=
by
  sorry

end find_certain_number_l2359_235969


namespace triangle_inequality_l2359_235963

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2)
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a < b + c) (h6 : b < a + c) (h7 : c < a + b) :
  a^2 + b^2 + c^2 + 2 * a * b * c < 2 := 
sorry

end triangle_inequality_l2359_235963


namespace p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l2359_235976

def p (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 1) + (y^2) / (m - 4) = 1
def q (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 2) + (y^2) / (4 - m) = 1

theorem p_hyperbola_implies_m_range (m : ℝ) (x y : ℝ) :
  p m x y → 1 < m ∧ m < 4 :=
sorry

theorem p_necessary_not_sufficient_for_q (m : ℝ) (x y : ℝ) :
  (1 < m ∧ m < 4) ∧ p m x y →
  (q m x y → (2 < m ∧ m < 3) ∨ (3 < m ∧ m < 4)) :=
sorry

end p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l2359_235976


namespace BANANA_arrangements_l2359_235910

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l2359_235910


namespace additional_money_earned_l2359_235933

-- Define the conditions as variables
def price_duck : ℕ := 10
def price_chicken : ℕ := 8
def num_chickens_sold : ℕ := 5
def num_ducks_sold : ℕ := 2
def half (x : ℕ) : ℕ := x / 2
def double (x : ℕ) : ℕ := 2 * x

-- Define the calculations based on the conditions
def earnings_chickens : ℕ := num_chickens_sold * price_chicken 
def earnings_ducks : ℕ := num_ducks_sold * price_duck 
def total_earnings : ℕ := earnings_chickens + earnings_ducks 
def cost_wheelbarrow : ℕ := half total_earnings
def selling_price_wheelbarrow : ℕ := double cost_wheelbarrow
def additional_earnings : ℕ := selling_price_wheelbarrow - cost_wheelbarrow

-- The theorem to prove the correct additional earnings
theorem additional_money_earned : additional_earnings = 30 := by
  sorry

end additional_money_earned_l2359_235933


namespace gary_initial_money_l2359_235960

/-- The initial amount of money Gary had, given that he spent $55 and has $18 left. -/
theorem gary_initial_money (amount_spent : ℤ) (amount_left : ℤ) (initial_amount : ℤ) 
  (h1 : amount_spent = 55) 
  (h2 : amount_left = 18) 
  : initial_amount = amount_spent + amount_left :=
by
  sorry

end gary_initial_money_l2359_235960


namespace melanie_books_bought_l2359_235901

def books_before_yard_sale : ℝ := 41.0
def books_after_yard_sale : ℝ := 128
def books_bought : ℝ := books_after_yard_sale - books_before_yard_sale

theorem melanie_books_bought : books_bought = 87 := by
  sorry

end melanie_books_bought_l2359_235901


namespace compound_interest_rate_l2359_235974

theorem compound_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 10000)
  (hA : A = 12155.06)
  (hn : n = 4)
  (ht : t = 1)
  (h_eq : A = P * (1 + r / n) ^ (n * t)):
  r = 0.2 :=
by
  sorry

end compound_interest_rate_l2359_235974


namespace subtraction_result_l2359_235968

theorem subtraction_result :
  5.3567 - 2.1456 - 1.0211 = 2.1900 := 
sorry

end subtraction_result_l2359_235968


namespace find_Q_plus_R_l2359_235926

-- P, Q, R must be digits in base 8 (distinct and non-zero)
def is_valid_digit (d : Nat) : Prop :=
  d > 0 ∧ d < 8

def digits_distinct (P Q R : Nat) : Prop :=
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define the base 8 number from its digits
def base8_number (P Q R : Nat) : Nat :=
  8^2 * P + 8 * Q + R

-- Define the given condition
def condition (P Q R : Nat) : Prop :=
  is_valid_digit P ∧ is_valid_digit Q ∧ is_valid_digit R ∧ digits_distinct P Q R ∧ 
  (base8_number P Q R + base8_number Q R P + base8_number R P Q = 8^3 * P + 8^2 * P + 8 * P + 8)

-- The result: Q + R in base 8 is 10_8 which is 8 + 2 (in decimal is 10)
theorem find_Q_plus_R (P Q R : Nat) (h : condition P Q R) : Q + R = 8 + 2 :=
sorry

end find_Q_plus_R_l2359_235926


namespace train_speed_approx_900072_kmph_l2359_235915

noncomputable def speed_of_train (train_length platform_length time_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / time_seconds
  speed_m_s * 3.6

theorem train_speed_approx_900072_kmph :
  abs (speed_of_train 225 400.05 25 - 90.0072) < 0.001 :=
by
  sorry

end train_speed_approx_900072_kmph_l2359_235915


namespace arithmetic_sequence_seventh_term_l2359_235902

theorem arithmetic_sequence_seventh_term (a d : ℚ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 29 / 3 := 
sorry

end arithmetic_sequence_seventh_term_l2359_235902


namespace sum_of_first_2009_terms_l2359_235985

variable (a : ℕ → ℝ) (d : ℝ)

-- conditions: arithmetic sequence and specific sum condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_condition (a : ℕ → ℝ) : Prop :=
  a 1004 + a 1005 + a 1006 = 3

-- sum of the first 2009 terms
noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n.succ * (a 0 + a n.succ) / 2)

-- proof problem
theorem sum_of_first_2009_terms (h1 : is_arithmetic_sequence a d) (h2 : sum_condition a) :
  sum_first_n_terms a 2008 = 2009 :=
sorry

end sum_of_first_2009_terms_l2359_235985


namespace terminal_side_in_quadrant_l2359_235966

theorem terminal_side_in_quadrant (α : ℝ) (h : α = -5) : 
  ∃ (q : ℕ), q = 4 ∧ 270 ≤ (α + 360) % 360 ∧ (α + 360) % 360 < 360 := by 
  sorry

end terminal_side_in_quadrant_l2359_235966


namespace cos_pi_div_3_l2359_235904

theorem cos_pi_div_3 : Real.cos (π / 3) = 1 / 2 := 
by
  sorry

end cos_pi_div_3_l2359_235904


namespace simplify_expression_1_simplify_expression_2_l2359_235934

-- Define the algebraic simplification problem for the first expression
theorem simplify_expression_1 (x y : ℝ) : 5 * x - 3 * (2 * x - 3 * y) + x = 9 * y :=
by
  sorry

-- Define the algebraic simplification problem for the second expression
theorem simplify_expression_2 (a : ℝ) : 3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l2359_235934


namespace trajectory_equation_of_P_l2359_235916

variable {x y : ℝ}
variable (A B P : ℝ × ℝ)

def in_line_through (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let k := (p.2 - a.2) / (p.1 - a.1)
  (b.2 - a.2) / (b.1 - a.1) = k

theorem trajectory_equation_of_P
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : in_line_through A B P)
  (slope_product : (P.2 / (P.1 + 1)) * (P.2 / (P.1 - 1)) = -1) :
  P.1 ^ 2 + P.2 ^ 2 = 1 ∧ P.1 ≠ 1 ∧ P.1 ≠ -1 := 
sorry

end trajectory_equation_of_P_l2359_235916


namespace value_of_a_l2359_235927

theorem value_of_a (x y z a : ℤ) (k : ℤ) 
  (h1 : x = 4 * k) (h2 : y = 6 * k) (h3 : z = 10 * k) 
  (hy_eq : y^2 = 40 * a - 20) 
  (ha_int : ∃ m : ℤ, a = m) : a = 1 := 
  sorry

end value_of_a_l2359_235927


namespace additional_workers_needed_l2359_235913

theorem additional_workers_needed :
  let initial_workers := 4
  let initial_parts := 108
  let initial_hours := 3
  let target_parts := 504
  let target_hours := 8
  (target_parts / target_hours) / (initial_parts / (initial_hours * initial_workers)) - initial_workers = 3 := by
  sorry

end additional_workers_needed_l2359_235913


namespace brocard_inequalities_l2359_235959

theorem brocard_inequalities (α β γ φ: ℝ) (h1: φ > 0) (h2: φ < π / 6)
  (h3: α > 0) (h4: β > 0) (h5: γ > 0) (h6: α + β + γ = π) : 
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := 
by 
  sorry

end brocard_inequalities_l2359_235959


namespace perfect_squares_digits_l2359_235970

theorem perfect_squares_digits 
  (a b : ℕ) 
  (ha : ∃ m : ℕ, a = m * m) 
  (hb : ∃ n : ℕ, b = n * n) 
  (a_units_digit_1 : a % 10 = 1) 
  (b_units_digit_6 : b % 10 = 6) 
  (a_tens_digit : ∃ x : ℕ, (a / 10) % 10 = x) 
  (b_tens_digit : ∃ y : ℕ, (b / 10) % 10 = y) : 
  ∃ x y : ℕ, (x % 2 = 0) ∧ (y % 2 = 1) := 
sorry

end perfect_squares_digits_l2359_235970


namespace garden_perimeter_l2359_235988

noncomputable def find_perimeter (l w : ℕ) : ℕ := 2 * l + 2 * w

theorem garden_perimeter :
  ∀ (l w : ℕ),
  (l = 3 * w + 2) →
  (l = 38) →
  find_perimeter l w = 100 :=
by
  intros l w H1 H2
  sorry

end garden_perimeter_l2359_235988


namespace intersection_sets_l2359_235903

def universal_set : Set ℝ := Set.univ
def set_A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def set_B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_sets (x : ℝ) : 
  (x ∈ set_A ∩ set_B) ↔ (-2 < x ∧ x < 4) :=
by sorry

end intersection_sets_l2359_235903


namespace solve_system_of_equations_l2359_235972

theorem solve_system_of_equations (x y z : ℝ) : 
  (y * z = 3 * y + 2 * z - 8) ∧
  (z * x = 4 * z + 3 * x - 8) ∧
  (x * y = 2 * x + y - 1) ↔ 
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5 / 2 ∧ z = -1)) :=
by
  sorry

end solve_system_of_equations_l2359_235972


namespace rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l2359_235993

variable (x : ℚ)

-- Polynomial 1
def polynomial1 := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_polynomial1 :
  (polynomial1 (-1) = 0) ∧
  (polynomial1 2 = 0) ∧
  (polynomial1 (-2) = 0) ∧
  (polynomial1 4 = 0) :=
sorry

-- Polynomial 2
def polynomial2 := 8*x^3 - 20*x^2 - 2*x + 5

theorem rational_roots_polynomial2 :
  (polynomial2 (1/2) = 0) ∧
  (polynomial2 (-1/2) = 0) ∧
  (polynomial2 (5/2) = 0) :=
sorry

-- Polynomial 3
def polynomial3 := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem rational_roots_polynomial3 :
  (polynomial3 (-1/2) = 0) ∧
  (polynomial3 (1/2) = 0) ∧
  (polynomial3 1 = 0) ∧
  (polynomial3 3 = 0) :=
sorry

end rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l2359_235993


namespace wage_difference_l2359_235949

-- Definitions of the problem
variables (P Q h : ℝ)
axiom total_pay : P * h = 480
axiom wage_relation : P = 1.5 * Q
axiom time_relation : Q * (h + 10) = 480

-- Theorem to prove the hourly wage difference
theorem wage_difference : P - Q = 8 :=
by
  sorry

end wage_difference_l2359_235949


namespace trigonometric_expression_l2359_235947

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.tan α = 3) : 
  (Real.sin α + 3 * Real.cos α) / (Real.cos α - 3 * Real.sin α) = -3/4 := 
by
  sorry

end trigonometric_expression_l2359_235947


namespace necessary_but_not_sufficient_l2359_235979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  ((a ≥ 4 ∨ a ≤ 0) ↔ (∃ x : ℝ, f a x = 0)) ∧ ¬((a ≥ 4 ∨ a ≤ 0) → (∃ x : ℝ, f a x = 0)) :=
sorry

end necessary_but_not_sufficient_l2359_235979


namespace cleaning_time_l2359_235980

noncomputable def combined_cleaning_time (sawyer_time nick_time sarah_time : ℕ) : ℚ :=
  let rate_sawyer := 1 / sawyer_time
  let rate_nick := 1 / nick_time
  let rate_sarah := 1 / sarah_time
  1 / (rate_sawyer + rate_nick + rate_sarah)

theorem cleaning_time : combined_cleaning_time 6 9 4 = 36 / 19 := by
  have h1 : 1 / 6 = 1 / 6 := rfl
  have h2 : 1 / 9 = 1 / 9 := rfl
  have h3 : 1 / 4 = 1 / 4 := rfl
  rw [combined_cleaning_time, h1, h2, h3]
  norm_num
  sorry

end cleaning_time_l2359_235980


namespace square_garden_perimeter_l2359_235990

theorem square_garden_perimeter (A : ℝ) (s : ℝ) (N : ℝ) 
  (h1 : A = 9)
  (h2 : s^2 = A)
  (h3 : N = 4 * s) 
  : N = 12 := 
by
  sorry

end square_garden_perimeter_l2359_235990


namespace sum_of_distinct_prime_factors_of_462_l2359_235907

-- Given a number n, define its prime factors.
def prime_factors (n : ℕ) : List ℕ :=
  if h : n = 462 then [2, 3, 7, 11] else []

-- Defines the sum of a list of natural numbers.
def sum_list (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

-- The main theorem statement.
theorem sum_of_distinct_prime_factors_of_462 : sum_list (prime_factors 462) = 23 :=
by
  sorry

end sum_of_distinct_prime_factors_of_462_l2359_235907


namespace bus_stops_bound_l2359_235923

-- Definitions based on conditions
variables (n x : ℕ)

-- Condition 1: Any bus stop is serviced by at most 3 bus lines
def at_most_three_bus_lines (bus_stops : ℕ) : Prop :=
  ∀ (stop : ℕ), stop < bus_stops → stop ≤ 3

-- Condition 2: Any bus line has at least two stops
def at_least_two_stops (bus_lines : ℕ) : Prop :=
  ∀ (line : ℕ), line < bus_lines → line ≥ 2

-- Condition 3: For any two specific bus lines, there is a third line such that passengers can transfer
def transfer_line_exists (bus_lines : ℕ) : Prop :=
  ∀ (line1 line2 : ℕ), line1 < bus_lines ∧ line2 < bus_lines →
  ∃ (line3 : ℕ), line3 < bus_lines

-- Theorem statement: The number of bus stops is at least 5/6 (n-5)
theorem bus_stops_bound (h1 : at_most_three_bus_lines x) (h2 : at_least_two_stops n)
  (h3 : transfer_line_exists n) : x ≥ (5 * (n - 5)) / 6 :=
sorry

end bus_stops_bound_l2359_235923


namespace boy_scouts_signed_slips_l2359_235939

-- Definitions for the problem conditions have only been used; solution steps are excluded.

theorem boy_scouts_signed_slips (total_scouts : ℕ) (signed_slips : ℕ) (boy_scouts : ℕ) (girl_scouts : ℕ)
  (boy_scouts_signed : ℕ) (girl_scouts_signed : ℕ)
  (h1 : signed_slips = 4 * total_scouts / 5)  -- 80% of the scouts arrived with signed permission slips
  (h2 : boy_scouts = 2 * total_scouts / 5)  -- 40% of the scouts were boy scouts
  (h3 : girl_scouts = total_scouts - boy_scouts)  -- Rest are girl scouts
  (h4 : girl_scouts_signed = 8333 * girl_scouts / 10000)  -- 83.33% of girl scouts with permission slips
  (h5 : signed_slips = boy_scouts_signed + girl_scouts_signed)  -- Total signed slips by both boy and girl scouts
  : (boy_scouts_signed * 100 / boy_scouts = 75) :=    -- 75% of boy scouts with permission slips
by
  -- Proof to be filled in.
  sorry

end boy_scouts_signed_slips_l2359_235939


namespace amelia_money_left_l2359_235918

theorem amelia_money_left :
  let first_course := 15
  let second_course := first_course + 5
  let dessert := 0.25 * second_course
  let total_first_three_courses := first_course + second_course + dessert
  let drink := 0.20 * total_first_three_courses
  let pre_tip_total := total_first_three_courses + drink
  let tip := 0.15 * pre_tip_total
  let total_bill := pre_tip_total + tip
  let initial_money := 60
  let money_left := initial_money - total_bill
  money_left = 4.8 :=
by
  sorry

end amelia_money_left_l2359_235918


namespace intersection_P_Q_l2359_235942

def P : Set ℝ := { x | x^2 - 9 < 0 }
def Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem intersection_P_Q : (P ∩ (coe '' Q)) = { x : ℝ | x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 } :=
by sorry

end intersection_P_Q_l2359_235942


namespace min_value_is_2_sqrt_2_l2359_235992

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + b^2 / (a - b)

theorem min_value_is_2_sqrt_2 (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : a * b = 1) : 
  min_value a b = 2 * Real.sqrt 2 := 
sorry

end min_value_is_2_sqrt_2_l2359_235992


namespace num_of_arithmetic_sequences_l2359_235941

-- Define the set of digits {1, 2, ..., 15}
def digits := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define an arithmetic sequence condition 
def is_arithmetic_sequence (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

-- Define the count of valid sequences with a specific difference
def count_arithmetic_sequences_with_difference (d : ℕ) : ℕ :=
  if d = 1 then 13
  else if d = 5 then 6
  else 0

-- Define the total count of valid sequences
def total_arithmetic_sequences : ℕ :=
  count_arithmetic_sequences_with_difference 1 +
  count_arithmetic_sequences_with_difference 5

-- The final statement to prove
theorem num_of_arithmetic_sequences : total_arithmetic_sequences = 19 := 
  sorry

end num_of_arithmetic_sequences_l2359_235941


namespace units_digit_of_n_l2359_235937

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 11^4) (h2 : m % 10 = 9) : n % 10 = 9 := 
sorry

end units_digit_of_n_l2359_235937


namespace translate_line_up_l2359_235950

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := -2 * x

-- Define the transformed line equation as a function
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Prove that translating the original line upward by 1 unit gives the translated line
theorem translate_line_up (x : ℝ) :
  original_line x + 1 = translated_line x :=
by
  unfold original_line translated_line
  simp

end translate_line_up_l2359_235950


namespace fastest_slowest_difference_l2359_235997

-- Given conditions
def length_A : ℕ := 8
def length_B : ℕ := 10
def length_C : ℕ := 6
def section_length : ℕ := 2

def sections_A : ℕ := 24
def sections_B : ℕ := 25
def sections_C : ℕ := 27

-- Calculate number of cuts required
def cuts_per_segment_A := length_A / section_length - 1
def cuts_per_segment_B := length_B / section_length - 1
def cuts_per_segment_C := length_C / section_length - 1

-- Calculate total number of cuts
def total_cuts_A := cuts_per_segment_A * (sections_A / (length_A / section_length))
def total_cuts_B := cuts_per_segment_B * (sections_B / (length_B / section_length))
def total_cuts_C := cuts_per_segment_C * (sections_C / (length_C / section_length))

-- Finding min and max cuts
def max_cuts := max total_cuts_A (max total_cuts_B total_cuts_C)
def min_cuts := min total_cuts_A (min total_cuts_B total_cuts_C)

-- Prove that the difference between max cuts and min cuts is 2
theorem fastest_slowest_difference :
  max_cuts - min_cuts = 2 := by
  sorry

end fastest_slowest_difference_l2359_235997


namespace squirrel_acorns_l2359_235955

theorem squirrel_acorns (S A : ℤ) 
  (h1 : A = 4 * S + 3) 
  (h2 : A = 5 * S - 6) : 
  A = 39 :=
by sorry

end squirrel_acorns_l2359_235955


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l2359_235917

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l2359_235917


namespace expression_value_l2359_235951

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x + y + z = 0) (h2 : xy + xz + yz ≠ 0) :
  (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)^2) = 3 / (x^2 + xy + y^2)^2 :=
by
  sorry

end expression_value_l2359_235951


namespace probability_of_C_l2359_235945

def region_prob_A := (1 : ℚ) / 4
def region_prob_B := (1 : ℚ) / 3
def region_prob_D := (1 : ℚ) / 6

theorem probability_of_C :
  (region_prob_A + region_prob_B + region_prob_D + (1 : ℚ) / 4) = 1 :=
by
  sorry

end probability_of_C_l2359_235945


namespace quadratic_function_monotonicity_l2359_235991

theorem quadratic_function_monotonicity
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, x ≤ y ∧ y ≤ -1 → a * x^2 + b * x + 3 ≤ a * y^2 + b * y + 3)
  (h2 : ∀ x y : ℝ, -1 ≤ x ∧ x ≤ y → a * x^2 + b * x + 3 ≥ a * y^2 + b * y + 3) :
  b = 2 * a ∧ a < 0 :=
sorry

end quadratic_function_monotonicity_l2359_235991


namespace negate_proposition_l2359_235989

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) := by
  sorry

end negate_proposition_l2359_235989


namespace arithmetic_sequence_sum_l2359_235940

theorem arithmetic_sequence_sum {a b : ℤ} (h : ∀ n : ℕ, 3 + n * 6 = if n = 2 then a else if n = 3 then b else 33) : a + b = 48 := by
  sorry

end arithmetic_sequence_sum_l2359_235940


namespace range_of_k_l2359_235914

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4 ∧ y = k * x + 3) ∧ 
  (∃ M N : ℝ × ℝ, ((M.1 - N.1)^2 + (M.2 - N.2)^2)^(1/2) ≥ 2) →
  (k ≤ 0) :=
by
  sorry

end range_of_k_l2359_235914


namespace atomic_weight_chlorine_l2359_235906

-- Define the given conditions and constants
def molecular_weight_compound : ℝ := 53
def atomic_weight_nitrogen : ℝ := 14.01
def atomic_weight_hydrogen : ℝ := 1.01
def number_of_hydrogen_atoms : ℝ := 4
def number_of_nitrogen_atoms : ℝ := 1

-- Define the total weight of nitrogen and hydrogen in the compound
def total_weight_nh : ℝ := (number_of_nitrogen_atoms * atomic_weight_nitrogen) + (number_of_hydrogen_atoms * atomic_weight_hydrogen)

-- Define the statement to be proved: the atomic weight of chlorine
theorem atomic_weight_chlorine : (molecular_weight_compound - total_weight_nh) = 34.95 := by
  sorry

end atomic_weight_chlorine_l2359_235906


namespace intersection_M_N_l2359_235981

def M : Set ℝ := { x | x^2 + x - 6 < 0 }
def N : Set ℝ := { x | |x - 1| ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } :=
sorry

end intersection_M_N_l2359_235981


namespace correct_operation_l2359_235952

theorem correct_operation (a : ℝ) : 
    (a ^ 2 + a ^ 4 ≠ a ^ 6) ∧ 
    (a ^ 2 * a ^ 3 ≠ a ^ 6) ∧ 
    (a ^ 3 / a ^ 2 = a) ∧ 
    ((a ^ 2) ^ 3 ≠ a ^ 5) :=
by
  sorry

end correct_operation_l2359_235952


namespace conditions_not_sufficient_nor_necessary_l2359_235984

theorem conditions_not_sufficient_nor_necessary (a : ℝ) (b : ℝ) :
  (a ≠ 5) ∧ (b ≠ -5) ↔ ¬((a ≠ 5) ∨ (b ≠ -5)) ∧ (a + b ≠ 0) := 
sorry

end conditions_not_sufficient_nor_necessary_l2359_235984


namespace smallest_sum_of_xy_l2359_235946

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l2359_235946


namespace mul_mod_remainder_l2359_235943

theorem mul_mod_remainder (a b m : ℕ)
  (h₁ : a ≡ 8 [MOD 9])
  (h₂ : b ≡ 1 [MOD 9]) :
  (a * b) % 9 = 8 := 
  sorry

def main : IO Unit :=
  IO.println "The theorem statement has been defined."

end mul_mod_remainder_l2359_235943


namespace number_of_solid_shapes_is_three_l2359_235987

-- Define the geometric shapes and their dimensionality
inductive GeomShape
| square : GeomShape
| cuboid : GeomShape
| circle : GeomShape
| sphere : GeomShape
| cone : GeomShape

def isSolid (shape : GeomShape) : Bool :=
  match shape with
  | GeomShape.square => false
  | GeomShape.cuboid => true
  | GeomShape.circle => false
  | GeomShape.sphere => true
  | GeomShape.cone => true

-- Formal statement of the problem
theorem number_of_solid_shapes_is_three :
  (List.filter isSolid [GeomShape.square, GeomShape.cuboid, GeomShape.circle, GeomShape.sphere, GeomShape.cone]).length = 3 :=
by
  -- proof omitted
  sorry

end number_of_solid_shapes_is_three_l2359_235987


namespace multiplication_of_powers_same_base_l2359_235983

theorem multiplication_of_powers_same_base (x : ℝ) : x^3 * x^2 = x^5 :=
by
-- proof steps go here
sorry

end multiplication_of_powers_same_base_l2359_235983


namespace find_c_l2359_235998

noncomputable def condition1 (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

noncomputable def condition2 (c : ℝ) : Prop :=
  6 * 15 * c = 1

theorem find_c (c : ℝ) (h1 : condition1 6 15 c) (h2 : condition2 c) : c = 11 := 
by
  sorry

end find_c_l2359_235998


namespace students_who_saw_l2359_235965

variable (B G : ℕ)

theorem students_who_saw (h : B + G = 33) : (2 * G / 3) + (2 * B / 3) = 22 :=
by
  sorry

end students_who_saw_l2359_235965
