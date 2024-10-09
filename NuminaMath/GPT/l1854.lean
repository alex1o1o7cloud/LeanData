import Mathlib

namespace amy_biking_miles_l1854_185470

theorem amy_biking_miles (x : ℕ) (h1 : ∀ y : ℕ, y = 2 * x - 3) (h2 : ∀ y : ℕ, x + y = 33) : x = 12 :=
by
  sorry

end amy_biking_miles_l1854_185470


namespace subgroups_of_integers_l1854_185441

theorem subgroups_of_integers (G : AddSubgroup ℤ) : ∃ (d : ℤ), G = AddSubgroup.zmultiples d := 
sorry

end subgroups_of_integers_l1854_185441


namespace batsman_average_after_17th_inning_l1854_185428

theorem batsman_average_after_17th_inning (A : ℝ) (h1 : 16 * A + 200 = 17 * (A + 10)) : 
  A + 10 = 40 := 
by
  sorry

end batsman_average_after_17th_inning_l1854_185428


namespace total_number_of_coins_l1854_185430

theorem total_number_of_coins {N B : ℕ} 
    (h1 : B - 2 = Nat.floor (N / 9))
    (h2 : N - 6 * (B - 3) = 3) 
    : N = 45 :=
by
  sorry

end total_number_of_coins_l1854_185430


namespace solve_for_x_l1854_185415

theorem solve_for_x 
  (a b : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P = (3, -1) ∧ (P.2 = 3 + b) ∧ (P.2 = a * 3 + 2)) :
  (a - 1) * 3 = b - 2 :=
by sorry

end solve_for_x_l1854_185415


namespace part_a_l1854_185483

theorem part_a (α : ℝ) (n : ℕ) (hα : α > 0) (hn : n > 1) : (1 + α)^n > 1 + n * α :=
sorry

end part_a_l1854_185483


namespace win_game_A_win_game_C_l1854_185462

-- Define the probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probability of winning Game A
def prob_win_game_A : ℚ := (prob_heads ^ 3) + (prob_tails ^ 3)

-- Define the probability of winning Game C
def prob_win_game_C : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)

-- State the theorem for Game A
theorem win_game_A : prob_win_game_A = 7 / 16 :=
by 
  -- Lean will check this proof
  sorry

-- State the theorem for Game C
theorem win_game_C : prob_win_game_C = 41 / 128 :=
by 
  -- Lean will check this proof
  sorry

end win_game_A_win_game_C_l1854_185462


namespace convert_base_10_to_base_7_l1854_185489

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 784) : 
  ∃ a b c d : ℕ, n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by
  sorry

end convert_base_10_to_base_7_l1854_185489


namespace quad_eq_complete_square_l1854_185445

theorem quad_eq_complete_square (p q : ℝ) 
  (h : ∀ x : ℝ, (4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16)) : q / p = -4 := by
  sorry

end quad_eq_complete_square_l1854_185445


namespace giyoon_chocolates_l1854_185493

theorem giyoon_chocolates (C X : ℕ) (h1 : C = 8 * X) (h2 : C = 6 * (X + 1) + 4) : C = 40 :=
by sorry

end giyoon_chocolates_l1854_185493


namespace functional_equation_solution_l1854_185481

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f (m + n)) = f m + f n) ↔
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a ∨ f n = 0) :=
sorry

end functional_equation_solution_l1854_185481


namespace roots_difference_l1854_185465

theorem roots_difference (a b c : ℝ) (h_eq : a = 1) (h_b : b = -11) (h_c : c = 24) :
    let r1 := (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    let r2 := (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    r1 - r2 = 5 := 
by
  sorry

end roots_difference_l1854_185465


namespace fraction_equiv_l1854_185424

theorem fraction_equiv (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 :=
by
  sorry

end fraction_equiv_l1854_185424


namespace number_of_zeros_at_end_l1854_185477

def N (n : Nat) := 10^(n+1) + 1

theorem number_of_zeros_at_end (n : Nat) (h : n = 2017) : 
  (N n)^(n + 1) - 1 ≡ 0 [MOD 10^(n + 1)] :=
sorry

end number_of_zeros_at_end_l1854_185477


namespace sum_of_interior_angles_l1854_185455

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l1854_185455


namespace correct_operation_B_l1854_185482

theorem correct_operation_B (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 :=
sorry

end correct_operation_B_l1854_185482


namespace run_to_grocery_store_time_l1854_185456

theorem run_to_grocery_store_time
  (running_time: ℝ)
  (grocery_distance: ℝ)
  (friend_distance: ℝ)
  (half_way : friend_distance = grocery_distance / 2)
  (constant_pace : running_time / grocery_distance = (25 : ℝ) / 3)
  : (friend_distance * (25 / 3)) + (friend_distance * (25 / 3)) = 25 :=
by
  -- Given proofs for the conditions can be filled here
  sorry

end run_to_grocery_store_time_l1854_185456


namespace viewers_difference_l1854_185408

theorem viewers_difference :
  let second_game := 80
  let first_game := second_game - 20
  let third_game := second_game + 15
  let fourth_game := third_game + (third_game / 10)
  let total_last_week := 350
  let total_this_week := first_game + second_game + third_game + fourth_game
  total_this_week - total_last_week = -10 := 
by
  sorry

end viewers_difference_l1854_185408


namespace arc_length_of_sector_l1854_185485

theorem arc_length_of_sector : 
  ∀ (r : ℝ) (theta: ℝ), r = 1 ∧ theta = 30 * (Real.pi / 180) → (theta * r = Real.pi / 6) :=
by
  sorry

end arc_length_of_sector_l1854_185485


namespace completing_square_result_l1854_185425

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l1854_185425


namespace volume_of_alcohol_correct_l1854_185417

noncomputable def radius := 3 / 2 -- radius of the tank
noncomputable def total_height := 9 -- total height of the tank
noncomputable def full_solution_height := total_height / 3 -- height of the liquid when the tank is one-third full
noncomputable def volume := Real.pi * radius^2 * full_solution_height -- volume of liquid in the tank
noncomputable def alcohol_ratio := 1 / 6 -- ratio of alcohol to the total solution
noncomputable def volume_of_alcohol := volume * alcohol_ratio -- volume of alcohol in the tank

theorem volume_of_alcohol_correct : volume_of_alcohol = (9 / 8) * Real.pi :=
by
  -- Proof would go here
  sorry

end volume_of_alcohol_correct_l1854_185417


namespace bugs_eat_flowers_l1854_185459

-- Define the problem conditions
def number_of_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Define the expected outcome
def total_flowers_eaten : ℕ := 6

-- Prove that total flowers eaten is equal to the product of the number of bugs and flowers per bug
theorem bugs_eat_flowers : number_of_bugs * flowers_per_bug = total_flowers_eaten :=
by
  sorry

end bugs_eat_flowers_l1854_185459


namespace arcade_fraction_spent_l1854_185452

noncomputable def weekly_allowance : ℚ := 2.25 
def y (x : ℚ) : ℚ := 1 - x
def remainding_after_toy (x : ℚ) : ℚ := y x - (1/3) * y x

theorem arcade_fraction_spent : 
  ∃ x : ℚ, remainding_after_toy x = 0.60 ∧ x = 3/5 :=
by
  sorry

end arcade_fraction_spent_l1854_185452


namespace cucumbers_count_l1854_185451

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end cucumbers_count_l1854_185451


namespace single_elimination_tournament_games_23_teams_l1854_185480

noncomputable def single_elimination_tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games_23_teams :
  single_elimination_tournament_games 23 = 22 :=
by
  -- Proof has been intentionally omitted
  sorry

end single_elimination_tournament_games_23_teams_l1854_185480


namespace deduce_pi_from_cylinder_volume_l1854_185427

theorem deduce_pi_from_cylinder_volume 
  (C h V : ℝ) 
  (Circumference : C = 20) 
  (Height : h = 11)
  (VolumeFormula : V = (1 / 12) * C^2 * h) : 
  pi = 3 :=
by 
  -- Carry out the proof
  sorry

end deduce_pi_from_cylinder_volume_l1854_185427


namespace miranda_saves_half_of_salary_l1854_185463

noncomputable def hourly_wage := 10
noncomputable def daily_hours := 10
noncomputable def weekly_days := 5
noncomputable def weekly_salary := hourly_wage * daily_hours * weekly_days

noncomputable def robby_saving_fraction := 2 / 5
noncomputable def jaylen_saving_fraction := 3 / 5
noncomputable def total_savings := 3000
noncomputable def weeks := 4

noncomputable def robby_weekly_savings := robby_saving_fraction * weekly_salary
noncomputable def jaylen_weekly_savings := jaylen_saving_fraction * weekly_salary
noncomputable def robby_total_savings := robby_weekly_savings * weeks
noncomputable def jaylen_total_savings := jaylen_weekly_savings * weeks
noncomputable def combined_savings_rj := robby_total_savings + jaylen_total_savings
noncomputable def miranda_total_savings := total_savings - combined_savings_rj
noncomputable def miranda_weekly_savings := miranda_total_savings / weeks

noncomputable def miranda_saving_fraction := miranda_weekly_savings / weekly_salary

theorem miranda_saves_half_of_salary:
  miranda_saving_fraction = 1 / 2 := 
by sorry

end miranda_saves_half_of_salary_l1854_185463


namespace quadratic_root_signs_l1854_185494

-- Variables representation
variables {x m : ℝ}

-- Given: The quadratic equation with one positive root and one negative root
theorem quadratic_root_signs (h : ∃ a b : ℝ, 2*a*2*b + (m+1)*(a + b) + m = 0 ∧ a > 0 ∧ b < 0) : 
  m < 0 := 
sorry

end quadratic_root_signs_l1854_185494


namespace nick_total_quarters_l1854_185404

theorem nick_total_quarters (Q : ℕ)
  (h1 : 2 / 5 * Q = state_quarters)
  (h2 : 1 / 2 * state_quarters = PA_quarters)
  (h3 : PA_quarters = 7) :
  Q = 35 := by
  sorry

end nick_total_quarters_l1854_185404


namespace exists_valid_circle_group_l1854_185431

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end exists_valid_circle_group_l1854_185431


namespace craig_distance_ridden_farther_l1854_185469

/-- Given that Craig rode the bus for 3.83 miles and walked for 0.17 miles,
    prove that the distance he rode farther than he walked is 3.66 miles. -/
theorem craig_distance_ridden_farther :
  let distance_bus := 3.83
  let distance_walked := 0.17
  distance_bus - distance_walked = 3.66 :=
by
  let distance_bus := 3.83
  let distance_walked := 0.17
  show distance_bus - distance_walked = 3.66
  sorry

end craig_distance_ridden_farther_l1854_185469


namespace train_platform_length_equal_l1854_185438

theorem train_platform_length_equal 
  (v : ℝ) (t : ℝ) (L_train : ℝ)
  (h1 : v = 144 * (1000 / 3600))
  (h2 : t = 60)
  (h3 : L_train = 1200) :
  L_train = 2400 - L_train := 
sorry

end train_platform_length_equal_l1854_185438


namespace simplify_and_evaluate_expression_l1854_185486

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -2) :
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l1854_185486


namespace betty_age_l1854_185421

def ages (A M B : ℕ) : Prop :=
  A = 2 * M ∧ A = 4 * B ∧ M = A - 22

theorem betty_age (A M B : ℕ) : ages A M B → B = 11 :=
by
  sorry

end betty_age_l1854_185421


namespace taxi_ride_cost_l1854_185412

-- Lean statement
theorem taxi_ride_cost (base_fare : ℝ) (rate1 : ℝ) (rate1_miles : ℝ) (rate2 : ℝ) (total_miles : ℝ) 
  (h_base_fare : base_fare = 2.00)
  (h_rate1 : rate1 = 0.30)
  (h_rate1_miles : rate1_miles = 3)
  (h_rate2 : rate2 = 0.40)
  (h_total_miles : total_miles = 8) :
  let rate1_cost := rate1 * rate1_miles
  let rate2_cost := rate2 * (total_miles - rate1_miles)
  base_fare + rate1_cost + rate2_cost = 4.90 := by
  sorry

end taxi_ride_cost_l1854_185412


namespace count_distinct_four_digit_numbers_ending_in_25_l1854_185429

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end count_distinct_four_digit_numbers_ending_in_25_l1854_185429


namespace number_of_slices_left_l1854_185426

-- Conditions
def total_slices : ℕ := 8
def slices_given_to_joe_and_darcy : ℕ := total_slices / 2
def slices_given_to_carl : ℕ := total_slices / 4

-- Question: How many slices were left?
def slices_left : ℕ := total_slices - (slices_given_to_joe_and_darcy + slices_given_to_carl)

-- Proof statement to demonstrate that slices_left == 2
theorem number_of_slices_left : slices_left = 2 := by
  sorry

end number_of_slices_left_l1854_185426


namespace average_rainfall_correct_l1854_185414

-- Definitions based on given conditions
def total_rainfall : ℚ := 420 -- inches
def days_in_august : ℕ := 31
def hours_in_a_day : ℕ := 24

-- Defining total hours in August
def total_hours_in_august : ℕ := days_in_august * hours_in_a_day

-- The average rainfall in inches per hour
def average_rainfall_per_hour : ℚ := total_rainfall / total_hours_in_august

-- The statement to prove
theorem average_rainfall_correct :
  average_rainfall_per_hour = 420 / 744 :=
by
  sorry

end average_rainfall_correct_l1854_185414


namespace total_weekly_messages_l1854_185484

theorem total_weekly_messages (n r1 r2 r3 r4 r5 m1 m2 m3 m4 m5 : ℕ) 
(p1 p2 p3 p4 : ℕ) (h1 : n = 200) (h2 : r1 = 15) (h3 : r2 = 25) (h4 : r3 = 10) 
(h5 : r4 = 20) (h6 : r5 = 5) (h7 : m1 = 40) (h8 : m2 = 60) (h9 : m3 = 50) 
(h10 : m4 = 30) (h11 : m5 = 20) (h12 : p1 = 15) (h13 : p2 = 25) (h14 : p3 = 40) 
(h15 : p4 = 10) : 
  let total_members_removed := r1 + r2 + r3 + r4 + r5
  let remaining_members := n - total_members_removed
  let daily_messages :=
        (25 * remaining_members / 100 * p1) +
        (50 * remaining_members / 100 * p2) +
        (20 * remaining_members / 100 * p3) +
        (5 * remaining_members / 100 * p4)
  let weekly_messages := daily_messages * 7
  weekly_messages = 21663 :=
by
  sorry

end total_weekly_messages_l1854_185484


namespace problem_l1854_185433

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f x = -f (-x)
axiom periodic_g : ∀ x, g x = g (x + 2)
axiom f_at_neg1 : f (-1) = 3
axiom g_at_1 : g 1 = 3
axiom g_function : ∀ n : ℕ, g (2 * n * f 1) = n * f (f 1 + g (-1)) + 2

theorem problem : g (-6) + f 0 = 2 :=
by sorry

end problem_l1854_185433


namespace inequality_proof_l1854_185490

theorem inequality_proof (a b c : ℝ) (ha : a = 2 / 21) (hb : b = Real.log 1.1) (hc : c = 21 / 220) : a < b ∧ b < c :=
by
  sorry

end inequality_proof_l1854_185490


namespace unanswered_questions_equal_nine_l1854_185457

theorem unanswered_questions_equal_nine
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : 
  z = 9 := by
  sorry

end unanswered_questions_equal_nine_l1854_185457


namespace brenda_age_l1854_185492

theorem brenda_age (A B J : ℝ)
  (h1 : A = 4 * B)
  (h2 : J = B + 8)
  (h3 : A = J + 2) :
  B = 10 / 3 :=
by
  sorry

end brenda_age_l1854_185492


namespace arithmetic_sequence_S7_eq_28_l1854_185409

/--
Given the arithmetic sequence \( \{a_n\} \) and the sum of its first \( n \) terms is \( S_n \),
if \( a_3 + a_4 + a_5 = 12 \), then prove \( S_7 = 28 \).
-/
theorem arithmetic_sequence_S7_eq_28
  (a : ℕ → ℤ) -- Sequence a_n
  (S : ℕ → ℤ) -- Sum sequence S_n
  (h1 : a 3 + a 4 + a 5 = 12)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum formula
  : S 7 = 28 :=
sorry

end arithmetic_sequence_S7_eq_28_l1854_185409


namespace tom_trout_count_l1854_185478

theorem tom_trout_count (M T : ℕ) (hM : M = 8) (hT : T = 2 * M) : T = 16 :=
by
  -- proof goes here
  sorry

end tom_trout_count_l1854_185478


namespace michael_total_payment_correct_l1854_185467

variable (original_suit_price : ℕ := 430)
variable (suit_discount : ℕ := 100)
variable (suit_tax_rate : ℚ := 0.05)

variable (original_shoes_price : ℕ := 190)
variable (shoes_discount : ℕ := 30)
variable (shoes_tax_rate : ℚ := 0.07)

variable (original_dress_shirt_price : ℕ := 80)
variable (original_tie_price : ℕ := 50)
variable (combined_discount_rate : ℚ := 0.20)
variable (dress_shirt_tax_rate : ℚ := 0.06)
variable (tie_tax_rate : ℚ := 0.04)

def calculate_total_amount_paid : ℚ :=
  let discounted_suit_price := original_suit_price - suit_discount
  let suit_tax := discounted_suit_price * suit_tax_rate
  let discounted_shoes_price := original_shoes_price - shoes_discount
  let shoes_tax := discounted_shoes_price * shoes_tax_rate
  let combined_original_price := original_dress_shirt_price + original_tie_price
  let combined_discount := combined_discount_rate * combined_original_price
  let discounted_combined_price := combined_original_price - combined_discount
  let discounted_dress_shirt_price := (original_dress_shirt_price / combined_original_price) * discounted_combined_price
  let discounted_tie_price := (original_tie_price / combined_original_price) * discounted_combined_price
  let dress_shirt_tax := discounted_dress_shirt_price * dress_shirt_tax_rate
  let tie_tax := discounted_tie_price * tie_tax_rate
  discounted_suit_price + suit_tax + discounted_shoes_price + shoes_tax + discounted_dress_shirt_price + dress_shirt_tax + discounted_tie_price + tie_tax

theorem michael_total_payment_correct : calculate_total_amount_paid = 627.14 := by
  sorry

end michael_total_payment_correct_l1854_185467


namespace total_shingles_needed_l1854_185401

-- Defining the dimensions of the house and the porch
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_length : ℝ := 6
def porch_width : ℝ := 4.5

-- The goal is to prove that the total area of the shingles needed is 232 square feet
theorem total_shingles_needed :
  (house_length * house_width) + (porch_length * porch_width) = 232 := by
  sorry

end total_shingles_needed_l1854_185401


namespace Xiaoming_age_l1854_185407

theorem Xiaoming_age (x : ℕ) (h1 : x = x) (h2 : x + 18 = 2 * (x + 6)) : x = 6 :=
sorry

end Xiaoming_age_l1854_185407


namespace bike_tire_fixing_charge_l1854_185450

theorem bike_tire_fixing_charge (total_profit rent_profit retail_profit: ℝ) (cost_per_tire_parts charge_per_complex_parts charge_per_complex: ℝ) (complex_repairs tire_repairs: ℕ) (charge_per_tire: ℝ) :
  total_profit  = 3000 → rent_profit = 4000 → retail_profit = 2000 →
  cost_per_tire_parts = 5 → charge_per_complex_parts = 50 → charge_per_complex = 300 →
  complex_repairs = 2 → tire_repairs = 300 →
  total_profit = (tire_repairs * charge_per_tire + complex_repairs * charge_per_complex + retail_profit - tire_repairs * cost_per_tire_parts - complex_repairs * charge_per_complex_parts - rent_profit) →
  charge_per_tire = 20 :=
by 
  sorry

end bike_tire_fixing_charge_l1854_185450


namespace ratio_sequences_l1854_185437

-- Define positive integers n and k, with k >= n and k - n even.
variables {n k : ℕ} (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0)

-- Define the sets S_N and S_M
def S_N (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_N
def S_M (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_M

-- Main theorem: N / M = 2^(k - n)
theorem ratio_sequences (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0) :
  (S_N n k : ℝ) / (S_M n k : ℝ) = 2^(k - n) := sorry

end ratio_sequences_l1854_185437


namespace KaydenceAge_l1854_185466

-- Definitions for ages of family members based on the problem conditions
def fatherAge := 60
def motherAge := fatherAge - 2 
def brotherAge := fatherAge / 2 
def sisterAge := 40
def totalFamilyAge := 200

-- Lean statement to prove the age of Kaydence
theorem KaydenceAge : 
  fatherAge + motherAge + brotherAge + sisterAge + Kaydence = totalFamilyAge → 
  Kaydence = 12 := 
by
  sorry

end KaydenceAge_l1854_185466


namespace sum_powers_div_5_iff_l1854_185419

theorem sum_powers_div_5_iff (n : ℕ) (h : n > 0) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_div_5_iff_l1854_185419


namespace alpha_beta_sum_l1854_185476

variable (α β : ℝ)

theorem alpha_beta_sum (h : ∀ x, (x - α) / (x + β) = (x^2 - 64 * x + 992) / (x^2 + 56 * x - 3168)) :
  α + β = 82 :=
sorry

end alpha_beta_sum_l1854_185476


namespace quadratic_transformation_concept_l1854_185453

theorem quadratic_transformation_concept :
  ∀ x : ℝ, (x-3)^2 - 4*(x-3) = 0 ↔ (x = 3 ∨ x = 7) :=
by
  intro x
  sorry

end quadratic_transformation_concept_l1854_185453


namespace domain_of_f_l1854_185496

-- Define the function y = sqrt(x-1) + sqrt(x*(3-x))
noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + Real.sqrt (x * (3 - x))

-- Proposition about the domain of the function
theorem domain_of_f (x : ℝ) : (∃ y : ℝ, y = f x) ↔ 1 ≤ x ∧ x ≤ 3 :=
by
  sorry

end domain_of_f_l1854_185496


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l1854_185495

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l1854_185495


namespace green_pairs_count_l1854_185473

theorem green_pairs_count 
  (blue_students : ℕ)
  (green_students : ℕ)
  (total_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ) 
  (mixed_pairs_students : ℕ) 
  (green_green_pairs : ℕ) 
  (count_blue : blue_students = 65)
  (count_green : green_students = 67)
  (count_total_students : total_students = 132)
  (count_total_pairs : total_pairs = 66)
  (count_blue_blue_pairs : blue_blue_pairs = 29)
  (count_mixed_blue_students : mixed_pairs_students = 7)
  (count_green_green_pairs : green_green_pairs = 30) :
  green_green_pairs = 30 :=
sorry

end green_pairs_count_l1854_185473


namespace find_a7_l1854_185406

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l1854_185406


namespace equal_division_of_balls_l1854_185420

def total_balls : ℕ := 10
def num_boxes : ℕ := 5
def balls_per_box : ℕ := total_balls / num_boxes

theorem equal_division_of_balls :
  balls_per_box = 2 :=
by
  sorry

end equal_division_of_balls_l1854_185420


namespace quadratic_vertex_on_x_axis_l1854_185488

theorem quadratic_vertex_on_x_axis (k : ℝ) :
  (∃ x : ℝ, (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end quadratic_vertex_on_x_axis_l1854_185488


namespace integer_solutions_2x2_2xy_9x_y_eq_2_l1854_185443

theorem integer_solutions_2x2_2xy_9x_y_eq_2 : ∀ (x y : ℤ), 2 * x^2 - 2 * x * y + 9 * x + y = 2 → (x, y) = (1, 9) ∨ (x, y) = (2, 8) ∨ (x, y) = (0, 2) ∨ (x, y) = (-1, 3) := 
by 
  intros x y h
  sorry

end integer_solutions_2x2_2xy_9x_y_eq_2_l1854_185443


namespace min_value_when_a_equals_1_range_of_a_for_f_geq_a_l1854_185403

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem min_value_when_a_equals_1 : 
  ∃ x, f x 1 = 1 :=
by
  sorry

theorem range_of_a_for_f_geq_a (a : ℝ) :
  (∀ x, x ≥ -1 → f x a ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end min_value_when_a_equals_1_range_of_a_for_f_geq_a_l1854_185403


namespace min_colors_needed_is_3_l1854_185461

noncomputable def min_colors_needed (S : Finset (Fin 7)) : Nat :=
  -- function to determine the minimum number of colors needed
  if ∀ (f : Finset (Fin 7) → Fin 3), ∀ (A B : Finset (Fin 7)), A.card = 3 ∧ B.card = 3 →
    A ∩ B = ∅ → f A ≠ f B then
    3
  else
    sorry

theorem min_colors_needed_is_3 :
  ∀ S : Finset (Fin 7), min_colors_needed S = 3 :=
by
  sorry

end min_colors_needed_is_3_l1854_185461


namespace third_person_fraction_removed_l1854_185405

-- Define the number of teeth for each person and the fractions that are removed
def total_teeth := 32
def total_removed := 40

def first_person_removed := (1 / 4) * total_teeth
def second_person_removed := (3 / 8) * total_teeth
def fourth_person_removed := 4

-- Define the total teeth removed by the first, second, and fourth persons
def known_removed := first_person_removed + second_person_removed + fourth_person_removed

-- Define the total teeth removed by the third person
def third_person_removed := total_removed - known_removed

-- Prove that the third person had 1/2 of his teeth removed
theorem third_person_fraction_removed :
  third_person_removed / total_teeth = 1 / 2 :=
by
  sorry

end third_person_fraction_removed_l1854_185405


namespace polynomial_factor_l1854_185400

def factorization_condition (p q : ℤ) : Prop :=
  ∃ r s : ℤ, 
    p = 4 * r ∧ 
    q = -3 * r + 4 * s ∧ 
    40 = 2 * r - 3 * s + 16 ∧ 
    -20 = s - 12

theorem polynomial_factor (p q : ℤ) (hpq : factorization_condition p q) : (p, q) = (0, -32) :=
by sorry

end polynomial_factor_l1854_185400


namespace symmetry_center_of_f_l1854_185446

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos (2 * x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f (2 * x + π / 6) = Real.sin (2 * (-π / 12) + π / 6) :=
sorry

end symmetry_center_of_f_l1854_185446


namespace quotient_base_6_l1854_185471

noncomputable def base_6_to_base_10 (n : ℕ) : ℕ := 
  match n with
  | 2314 => 2 * 6^3 + 3 * 6^2 + 1 * 6^1 + 4
  | 14 => 1 * 6^1 + 4
  | _ => 0

noncomputable def base_10_to_base_6 (n : ℕ) : ℕ := 
  match n with
  | 55 => 1 * 6^2 + 3 * 6^1 + 5
  | _ => 0

theorem quotient_base_6 :
  base_10_to_base_6 ((base_6_to_base_10 2314) / (base_6_to_base_10 14)) = 135 :=
by
  sorry

end quotient_base_6_l1854_185471


namespace total_people_can_ride_l1854_185423

theorem total_people_can_ride (num_people_per_teacup : Nat) (num_teacups : Nat) (h1 : num_people_per_teacup = 9) (h2 : num_teacups = 7) : num_people_per_teacup * num_teacups = 63 := by
  sorry

end total_people_can_ride_l1854_185423


namespace h_h_neg1_l1854_185454

def h (x: ℝ) : ℝ := 3 * x^2 - x + 1

theorem h_h_neg1 : h (h (-1)) = 71 := by
  sorry

end h_h_neg1_l1854_185454


namespace words_written_first_two_hours_l1854_185416

def essay_total_words : ℕ := 1200
def words_per_hour_first_two_hours (W : ℕ) : ℕ := 2 * W
def words_per_hour_next_two_hours : ℕ := 2 * 200

theorem words_written_first_two_hours (W : ℕ) (h : words_per_hour_first_two_hours W + words_per_hour_next_two_hours = essay_total_words) : W = 400 := 
by 
  sorry

end words_written_first_two_hours_l1854_185416


namespace winning_candidate_percentage_l1854_185472

theorem winning_candidate_percentage
  (votes1 votes2 votes3 : ℕ)
  (h1 : votes1 = 3000)
  (h2 : votes2 = 5000)
  (h3 : votes3 = 20000) :
  ((votes3 : ℝ) / (votes1 + votes2 + votes3) * 100) = 71.43 := by
  sorry

end winning_candidate_percentage_l1854_185472


namespace total_divisors_7350_l1854_185491

def primeFactorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 7350 then [(2, 1), (3, 1), (5, 2), (7, 2)] else []

def totalDivisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc (p : ℕ × ℕ) => acc * (p.snd + 1)) 1

theorem total_divisors_7350 : totalDivisors (primeFactorization 7350) = 36 :=
by
  sorry

end total_divisors_7350_l1854_185491


namespace maximum_fraction_sum_l1854_185422

noncomputable def max_fraction_sum (n : ℕ) (a b c d : ℕ) : ℝ :=
  1 - (1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1)))

theorem maximum_fraction_sum (n a b c d : ℕ) (h₀ : n > 1) (h₁ : a + c ≤ n) (h₂ : (a : ℚ) / b + (c : ℚ) / d < 1) :
  ∃ m : ℝ, m = max_fraction_sum n a b c d := by
  sorry

end maximum_fraction_sum_l1854_185422


namespace power_multiplication_equals_result_l1854_185410

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end power_multiplication_equals_result_l1854_185410


namespace intersection_points_count_l1854_185444

noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x ^ 2 - 4 * x + 4

theorem intersection_points_count : ∃! x y : ℝ, 0 < x ∧ f x = g x ∧ y ≠ x ∧ f y = g y :=
sorry

end intersection_points_count_l1854_185444


namespace sector_central_angle_l1854_185464

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : 1/2 * l * r = 1) : l / r = 2 := 
by
  sorry

end sector_central_angle_l1854_185464


namespace part1_geometric_sequence_part2_sum_of_terms_l1854_185474

/- Part 1 -/
theorem part1_geometric_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h₀ : a 1 = 3) 
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2 * a n) 
  (h₂ : ∀ n, 2 ^ b n = a n + 1) :
  ∃ r, ∀ n, b (n + 1) = r * b n ∧ r = 2 :=
by 
  use 2 
  sorry

/- Part 2 -/
theorem part2_sum_of_terms (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₀ : ∀ n, b n = 2 ^ n)
  (h₁ : ∀ n, c n = n / b n + 1) :
  ∀ n, T n = n + 2 - (n + 2) / 2 ^ n :=
by
  sorry

end part1_geometric_sequence_part2_sum_of_terms_l1854_185474


namespace ab_root_inequality_l1854_185439

theorem ab_root_inequality (a b : ℝ) (h1: ∀ x : ℝ, (x + a) * (x + b) = -9) (h2: a < 0) (h3: b < 0) :
  a + b < -6 :=
sorry

end ab_root_inequality_l1854_185439


namespace divisor_count_of_45_l1854_185436

theorem divisor_count_of_45 : 
  ∃ (n : ℤ), n = 12 ∧ ∀ d : ℤ, d ∣ 45 → (d > 0 ∨ d < 0) := sorry

end divisor_count_of_45_l1854_185436


namespace lateral_surface_of_prism_is_parallelogram_l1854_185487

-- Definitions based on conditions
def is_right_prism (P : Type) : Prop := sorry
def is_oblique_prism (P : Type) : Prop := sorry
def is_rectangle (S : Type) : Prop := sorry
def is_parallelogram (S : Type) : Prop := sorry
def lateral_surface (P : Type) : Type := sorry

-- Condition 1: The lateral surface of a right prism is a rectangle
axiom right_prism_surface_is_rectangle (P : Type) (h : is_right_prism P) : is_rectangle (lateral_surface P)

-- Condition 2: The lateral surface of an oblique prism can either be a rectangle or a parallelogram
axiom oblique_prism_surface_is_rectangle_or_parallelogram (P : Type) (h : is_oblique_prism P) :
  is_rectangle (lateral_surface P) ∨ is_parallelogram (lateral_surface P)

-- Lean 4 statement for the proof problem
theorem lateral_surface_of_prism_is_parallelogram (P : Type) (p : is_right_prism P ∨ is_oblique_prism P) :
  is_parallelogram (lateral_surface P) :=
by
  sorry

end lateral_surface_of_prism_is_parallelogram_l1854_185487


namespace solve_system_infinite_solutions_l1854_185475

theorem solve_system_infinite_solutions (m : ℝ) (h1 : ∀ x y : ℝ, x + m * y = 2) (h2 : ∀ x y : ℝ, m * x + 16 * y = 8) :
  m = 4 :=
sorry

end solve_system_infinite_solutions_l1854_185475


namespace six_digit_pair_divisibility_l1854_185432

theorem six_digit_pair_divisibility (a b : ℕ) (ha : 100000 ≤ a ∧ a < 1000000) (hb : 100000 ≤ b ∧ b < 1000000) :
  ((1000000 * a + b) % (a * b) = 0) ↔ (a = 166667 ∧ b = 333334) ∨ (a = 500001 ∧ b = 500001) :=
by sorry

end six_digit_pair_divisibility_l1854_185432


namespace Mr_Spacek_birds_l1854_185442

theorem Mr_Spacek_birds :
  ∃ N : ℕ, 50 < N ∧ N < 100 ∧ N % 9 = 0 ∧ N % 4 = 0 ∧ N = 72 :=
by
  sorry

end Mr_Spacek_birds_l1854_185442


namespace fraction_less_than_thirty_percent_l1854_185447

theorem fraction_less_than_thirty_percent (x : ℚ) (hx : x * 180 = 36) (hx_lt : x < 0.3) : x = 1 / 5 := 
by
  sorry

end fraction_less_than_thirty_percent_l1854_185447


namespace water_bottles_needed_l1854_185499

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l1854_185499


namespace sheila_monthly_savings_l1854_185413

-- Define the conditions and the question in Lean
def initial_savings : ℕ := 3000
def family_contribution : ℕ := 7000
def years : ℕ := 4
def final_amount : ℕ := 23248

-- Function to calculate the monthly saving given the conditions
def monthly_savings (initial_savings family_contribution years final_amount : ℕ) : ℕ :=
  (final_amount - (initial_savings + family_contribution)) / (years * 12)

-- The theorem we need to prove in Lean
theorem sheila_monthly_savings :
  monthly_savings initial_savings family_contribution years final_amount = 276 :=
by
  sorry

end sheila_monthly_savings_l1854_185413


namespace circumradius_relationship_l1854_185434

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end circumradius_relationship_l1854_185434


namespace sum_primes_less_than_20_l1854_185498

theorem sum_primes_less_than_20 : (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := sorry

end sum_primes_less_than_20_l1854_185498


namespace cube_of_number_l1854_185460

theorem cube_of_number (n : ℕ) (h1 : 40000 < n^3) (h2 : n^3 < 50000) (h3 : (n^3 % 10) = 6) : n = 36 := by
  sorry

end cube_of_number_l1854_185460


namespace max_possible_value_l1854_185440

theorem max_possible_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∀ (z : ℝ), (z = (x + y + 1) / x) → z ≤ -0.2 :=
by sorry

end max_possible_value_l1854_185440


namespace third_shiny_penny_prob_l1854_185458

open Nat

def num_shiny : Nat := 4
def num_dull : Nat := 5
def total_pennies : Nat := num_shiny + num_dull

theorem third_shiny_penny_prob :
  let a := 5
  let b := 9
  a + b = 14 := 
by
  sorry

end third_shiny_penny_prob_l1854_185458


namespace trig_identity_l1854_185418

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) : 
  2 * Real.sin x + 3 * Real.cos x = 1 ∨ 2 * Real.sin x + 3 * Real.cos x = 3 :=
sorry

end trig_identity_l1854_185418


namespace inequality_proof_l1854_185479

theorem inequality_proof (n : ℕ) (h : n > 1) : 
  1 / (2 * n * Real.exp 1) < 1 / Real.exp 1 - (1 - 1 / n) ^ n ∧ 
  1 / Real.exp 1 - (1 - 1 / n) ^ n < 1 / (n * Real.exp 1) := 
by
  sorry

end inequality_proof_l1854_185479


namespace sufficient_not_necessary_condition_l1854_185402
open Real

theorem sufficient_not_necessary_condition (m : ℝ) :
  ((m = 0) → ∃ x y : ℝ, (m + 1) * x + (1 - m) * y - 1 = 0 ∧ (m - 1) * x + (2 * m + 1) * y + 4 = 0 ∧ 
  ((m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0 ∨ (m = 1 ∨ m = 0))) :=
by sorry

end sufficient_not_necessary_condition_l1854_185402


namespace initial_water_percentage_l1854_185497

noncomputable def S : ℝ := 4.0
noncomputable def V_initial : ℝ := 440
noncomputable def V_final : ℝ := 460
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8
noncomputable def kola_percentage : ℝ := 8.0 / 100.0
noncomputable def final_sugar_percentage : ℝ := 4.521739130434784 / 100.0

theorem initial_water_percentage : 
  ∀ (W S : ℝ),
  V_initial * (S / 100) + sugar_added = final_sugar_percentage * V_final →
  (W + 8.0 + S) = 100.0 →
  W = 88.0
:=
by
  intros W S h1 h2
  sorry

end initial_water_percentage_l1854_185497


namespace measure_angle_P_l1854_185468

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end measure_angle_P_l1854_185468


namespace charge_per_trousers_l1854_185411

-- Definitions
def pairs_of_trousers : ℕ := 10
def shirts : ℕ := 10
def bill : ℕ := 140
def charge_per_shirt : ℕ := 5

-- Theorem statement
theorem charge_per_trousers :
  ∃ (T : ℕ), (pairs_of_trousers * T + shirts * charge_per_shirt = bill) ∧ (T = 9) :=
by 
  sorry

end charge_per_trousers_l1854_185411


namespace Maria_green_towels_l1854_185435

-- Definitions
variable (G : ℕ) -- number of green towels

-- Conditions
def initial_towels := G + 21
def final_towels := initial_towels - 34

-- Theorem statement
theorem Maria_green_towels : final_towels = 22 → G = 35 :=
by
  sorry

end Maria_green_towels_l1854_185435


namespace find_x_l1854_185449

-- We are given points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 2)

-- Vector a is (2x + 3, x^2 - 4)
def vec_a (x : ℝ) : ℝ × ℝ := (2 * x + 3, x^2 - 4)

-- Vector AB is calculated as
def vec_AB : ℝ × ℝ := (3 - 1, 2 - 2)

-- Define the condition that vec_a and vec_AB form 0° angle
def forms_zero_angle (u v : ℝ × ℝ) : Prop := (u.1 * v.2 - u.2 * v.1) = 0 ∧ (u.1 = v.1 ∧ v.2 = 0)

-- The proof statement
theorem find_x (x : ℝ) (h₁ : forms_zero_angle (vec_a x) vec_AB) : x = 2 :=
by
  sorry

end find_x_l1854_185449


namespace medium_stores_count_l1854_185448

-- Define the total number of stores
def total_stores : ℕ := 300

-- Define the number of medium stores
def medium_stores : ℕ := 75

-- Define the sample size
def sample_size : ℕ := 20

-- Define the expected number of medium stores in the sample
def expected_medium_stores : ℕ := 5

-- The theorem statement claiming that the number of medium stores in the sample is 5
theorem medium_stores_count : 
  (sample_size * medium_stores) / total_stores = expected_medium_stores :=
by
  -- Proof omitted
  sorry

end medium_stores_count_l1854_185448
