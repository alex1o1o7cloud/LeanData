import Mathlib

namespace triangle_perimeter_l326_32655

variable (r A p : ℝ)

-- Define the conditions from the problem
def inradius (r : ℝ) := r = 3
def area (A : ℝ) := A = 30
def perimeter (A r p : ℝ) := A = r * (p / 2)

-- The theorem stating the problem
theorem triangle_perimeter (h1 : inradius r) (h2 : area A) (h3 : perimeter A r p) : p = 20 := 
by
  -- Proof is provided by the user, so we skip it with sorry
  sorry

end triangle_perimeter_l326_32655


namespace domain_range_of_g_l326_32602

variable (f : ℝ → ℝ)
variable (dom_f : Set.Icc 1 3)
variable (rng_f : Set.Icc 0 1)
variable (g : ℝ → ℝ)
variable (g_eq : ∀ x, g x = 2 - f (x - 1))

theorem domain_range_of_g :
  (Set.Icc 2 4) = { x | ∃ y, x = y ∧ g y = (g y) } ∧ Set.Icc 1 2 = { z | ∃ w, z = g w} :=
  sorry

end domain_range_of_g_l326_32602


namespace max_a4_l326_32656

variable (a1 d : ℝ)

theorem max_a4 (h1 : 2 * a1 + 6 * d ≥ 10) (h2 : 2.5 * a1 + 10 * d ≤ 15) :
  ∃ max_a4, max_a4 = 4 ∧ a1 + 3 * d ≤ max_a4 :=
by
  sorry

end max_a4_l326_32656


namespace opposite_of_x_is_positive_l326_32628

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l326_32628


namespace abs_neg_two_l326_32652

def absolute_value (x : Int) : Int :=
  if x >= 0 then x else -x

theorem abs_neg_two : absolute_value (-2) = 2 := 
by 
  sorry

end abs_neg_two_l326_32652


namespace quadratic_root_range_quadratic_product_of_roots_l326_32633

-- Problem (1): Prove the range of m.
theorem quadratic_root_range (m : ℝ) :
  (∀ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 → x1 ≠ x2) ↔ m < 1 := 
sorry

-- Problem (2): Prove the existence of m such that x1 * x2 = 0.
theorem quadratic_product_of_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 ∧ x1 * x2 = 0) ↔ m = -1 := 
sorry

end quadratic_root_range_quadratic_product_of_roots_l326_32633


namespace B_investment_l326_32636

theorem B_investment (A : ℝ) (t_B : ℝ) (profit_ratio : ℝ) (B_investment_result : ℝ) : 
  A = 27000 → t_B = 4.5 → profit_ratio = 2 → B_investment_result = 36000 :=
by
  intro hA htB hpR
  sorry

end B_investment_l326_32636


namespace solution_part1_solution_part2_l326_32647

variable (f : ℝ → ℝ) (a x m : ℝ)

def problem_statement :=
  (∀ x : ℝ, f x = abs (x - a)) ∧
  (∀ x : ℝ, f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5)

theorem solution_part1 (x : ℝ) (h : problem_statement f a) : a = 2 :=
by
  sorry

theorem solution_part2 (x : ℝ) (h : problem_statement f a) :
  (∀ x : ℝ, f x + f (x + 5) ≥ m) → m ≤ 5 :=
by
  sorry

end solution_part1_solution_part2_l326_32647


namespace find_X_l326_32606

def spadesuit (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

theorem find_X (X : ℝ) (h : spadesuit X 5 = 23) : X = 7.75 :=
by sorry

end find_X_l326_32606


namespace value_of_y_square_plus_inverse_square_l326_32693

variable {y : ℝ}
variable (h : 35 = y^4 + 1 / y^4)

theorem value_of_y_square_plus_inverse_square (h : 35 = y^4 + 1 / y^4) : y^2 + 1 / y^2 = Real.sqrt 37 := 
sorry

end value_of_y_square_plus_inverse_square_l326_32693


namespace contradiction_proof_l326_32699

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
sorry

end contradiction_proof_l326_32699


namespace community_cleaning_children_l326_32670

theorem community_cleaning_children (total_members adult_men_ratio adult_women_ratio : ℕ) 
(h_total : total_members = 2000)
(h_men_ratio : adult_men_ratio = 30) 
(h_women_ratio : adult_women_ratio = 2) :
  (total_members - (adult_men_ratio * total_members / 100 + 
  adult_women_ratio * (adult_men_ratio * total_members / 100))) = 200 :=
by
  sorry

end community_cleaning_children_l326_32670


namespace patrick_savings_l326_32672

theorem patrick_savings :
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  saved_money - lent_money = 25 := by
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  sorry

end patrick_savings_l326_32672


namespace tan_diff_eq_sqrt_three_l326_32611

open Real

theorem tan_diff_eq_sqrt_three (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : cos α * cos β = 1 / 6) (h5 : sin α * sin β = 1 / 3) : 
  tan (β - α) = sqrt 3 := by
  sorry

end tan_diff_eq_sqrt_three_l326_32611


namespace jill_spent_more_l326_32659

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end jill_spent_more_l326_32659


namespace phantom_needs_more_money_l326_32616

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end phantom_needs_more_money_l326_32616


namespace solve_for_z_l326_32635

variable (x y z : ℝ)

theorem solve_for_z (h : 1 / x - 1 / y = 1 / z) : z = x * y / (y - x) := 
sorry

end solve_for_z_l326_32635


namespace solve_for_a_l326_32608

theorem solve_for_a (x a : ℝ) (hx_pos : 0 < x) (hx_sqrt1 : x = (a+1)^2) (hx_sqrt2 : x = (a-3)^2) : a = 1 :=
by
  sorry

end solve_for_a_l326_32608


namespace bullet_train_speed_l326_32617

theorem bullet_train_speed 
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (speed_train2 : ℝ)
  (time_cross : ℝ)
  (combined_length : ℝ)
  (time_cross_hours : ℝ)
  (relative_speed : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 270 → 
  length_train2 = 230.04 →
  speed_train2 = 80 →
  time_cross = 9 →
  combined_length = (length_train1 + length_train2) / 1000 →
  time_cross_hours = time_cross / 3600 →
  relative_speed = combined_length / time_cross_hours →
  relative_speed = speed_train1 + speed_train2 →
  speed_train1 = 120.016 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bullet_train_speed_l326_32617


namespace sum_equals_1584_l326_32658

-- Let's define the function that computes the sum, according to the pattern
def sumPattern : ℕ → ℝ
  | 0 => 0
  | k + 1 => if (k + 1) % 3 = 0 then - (k + 1) + sumPattern k
             else (k + 1) + sumPattern k

-- This function defines the problem setting and the final expected result
theorem sum_equals_1584 : sumPattern 99 = 1584 := by
  sorry

end sum_equals_1584_l326_32658


namespace truth_probability_l326_32600

variables (P_A P_B P_AB : ℝ)

theorem truth_probability (h1 : P_B = 0.60) (h2 : P_AB = 0.48) : P_A = 0.80 :=
by
  have h3 : P_AB = P_A * P_B := sorry  -- Placeholder for the rule: P(A and B) = P(A) * P(B)
  rw [h2, h1] at h3
  sorry

end truth_probability_l326_32600


namespace percentage_difference_l326_32669

theorem percentage_difference :
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  a1 - a2 = 1.484 := 
by
  -- Definitions
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  -- Proof body (skipped for this task)
  sorry

end percentage_difference_l326_32669


namespace evaluate_expression_l326_32646

theorem evaluate_expression :
  let a := 2020
  let b := 2016
  (2^a + 2^b) / (2^a - 2^b) = 17 / 15 :=
by
  sorry

end evaluate_expression_l326_32646


namespace second_machine_time_l326_32686

theorem second_machine_time (x : ℝ) : 
  (600 / 10) + (1000 / x) = 1000 / 4 ↔ 
  1 / 10 + 1 / x = 1 / 4 :=
by
  sorry

end second_machine_time_l326_32686


namespace find_integer_solutions_l326_32653

theorem find_integer_solutions (k : ℕ) (hk : k > 1) : 
  ∃ x y : ℤ, y^k = x^2 + x ↔ (k = 2 ∧ (x = 0 ∨ x = -1)) ∨ (k > 2 ∧ y^k ≠ x^2 + x) :=
by
  sorry

end find_integer_solutions_l326_32653


namespace pluto_orbit_scientific_notation_l326_32641

theorem pluto_orbit_scientific_notation : 5900000000 = 5.9 * 10^9 := by
  sorry

end pluto_orbit_scientific_notation_l326_32641


namespace number_of_dogs_l326_32680

theorem number_of_dogs (h1 : 24 = 2 * 2 + 4 * n) : n = 5 :=
by
  sorry

end number_of_dogs_l326_32680


namespace negation_of_proposition_l326_32679

theorem negation_of_proposition :
  ¬(∃ x₀ : ℝ, 0 < x₀ ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by
  sorry

end negation_of_proposition_l326_32679


namespace gcd_35_91_840_l326_32687

theorem gcd_35_91_840 : Nat.gcd (Nat.gcd 35 91) 840 = 7 :=
by
  sorry

end gcd_35_91_840_l326_32687


namespace problem_l326_32678

theorem problem 
  (a : ℝ) 
  (h_a : ∀ x : ℝ, |x + 1| - |2 - x| ≤ a ∧ a ≤ |x + 1| + |2 - x|)
  {m n : ℝ} 
  (h_mn : m > n) 
  (h_n : n > 0)
  (h: a = 3) 
  : 2 * m + 1 / (m^2 - 2 * m * n + n^2) ≥ 2 * n + a :=
by
  sorry

end problem_l326_32678


namespace initial_apps_l326_32695

-- Define the initial condition stating the number of files Dave had initially
def files_initial : ℕ := 21

-- Define the condition after deletion
def apps_after_deletion : ℕ := 3
def files_after_deletion : ℕ := 7

-- Define the number of files deleted
def files_deleted : ℕ := 14

-- Prove that the initial number of apps Dave had was 3
theorem initial_apps (a : ℕ) (h1 : files_initial = 21) 
(h2 : files_after_deletion = 7) 
(h3 : files_deleted = 14) 
(h4 : a - 3 = 0) : a = 3 :=
by sorry

end initial_apps_l326_32695


namespace probability_of_drawing_white_ball_is_zero_l326_32689

theorem probability_of_drawing_white_ball_is_zero
  (red_balls blue_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : blue_balls = 5)
  (white_balls : ℕ)
  (h3 : white_balls = 0) : 
  (0 / (red_balls + blue_balls + white_balls) = 0) :=
sorry

end probability_of_drawing_white_ball_is_zero_l326_32689


namespace one_third_of_1206_is_100_5_percent_of_400_l326_32654

theorem one_third_of_1206_is_100_5_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l326_32654


namespace find_costs_of_A_and_B_find_price_reduction_l326_32648

-- Definitions for part 1
def cost_of_type_A_and_B (x y : ℕ) : Prop :=
  (5 * x + 3 * y = 450) ∧ (10 * x + 8 * y = 1000)

-- Part 1: Prove that x and y satisfy the cost conditions
theorem find_costs_of_A_and_B (x y : ℕ) (hx : 5 * x + 3 * y = 450) (hy : 10 * x + 8 * y = 1000) : 
  x = 60 ∧ y = 50 :=
sorry

-- Definitions for part 2
def daily_profit_condition (m : ℕ) : Prop :=
  (100 + 20 * m > 200) ∧ ((80 - m) * (100 + 20 * m) + 7000 = 10000)

-- Part 2: Prove that the price reduction m meets the profit condition
theorem find_price_reduction (m : ℕ) (hm : 100 + 20 * m > 200) (hp : (80 - m) * (100 + 20 * m) + 7000 = 10000) : 
  m = 10 :=
sorry

end find_costs_of_A_and_B_find_price_reduction_l326_32648


namespace total_parallelepipeds_l326_32619

theorem total_parallelepipeds (m n k : ℕ) : 
  ∃ (num : ℕ), num == (m * n * k * (m + 1) * (n + 1) * (k + 1)) / 8 :=
  sorry

end total_parallelepipeds_l326_32619


namespace minimum_loaves_arithmetic_sequence_l326_32682

theorem minimum_loaves_arithmetic_sequence :
  ∃ a d : ℚ, 
    (5 * a = 100) ∧ (3 * a + 3 * d = 7 * (2 * a - 3 * d)) ∧ (a - 2 * d = 5/3) :=
sorry

end minimum_loaves_arithmetic_sequence_l326_32682


namespace interest_difference_l326_32663

theorem interest_difference (P R T : ℝ) (SI : ℝ) (Diff : ℝ) :
  P = 250 ∧ R = 4 ∧ T = 8 ∧ SI = (P * R * T) / 100 ∧ Diff = P - SI → Diff = 170 :=
by sorry

end interest_difference_l326_32663


namespace product_of_special_triplet_l326_32677

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_triangular (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1) / 2

def three_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1

theorem product_of_special_triplet :
  ∃ a b c : ℕ, a < b ∧ b < c ∧ c < 20 ∧ three_consecutive a b c ∧
   is_prime a ∧ is_even b ∧ is_triangular c ∧ a * b * c = 2730 :=
sorry

end product_of_special_triplet_l326_32677


namespace jelly_beans_in_jar_X_l326_32650

theorem jelly_beans_in_jar_X : 
  ∀ (X Y : ℕ), (X + Y = 1200) → (X = 3 * Y - 400) → X = 800 :=
by
  sorry

end jelly_beans_in_jar_X_l326_32650


namespace Isabel_earning_l326_32620

-- Define the number of bead necklaces sold
def bead_necklaces : ℕ := 3

-- Define the number of gem stone necklaces sold
def gemstone_necklaces : ℕ := 3

-- Define the cost of each necklace
def cost_per_necklace : ℕ := 6

-- Calculate the total number of necklaces sold
def total_necklaces : ℕ := bead_necklaces + gemstone_necklaces

-- Calculate the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings is 36 dollars
theorem Isabel_earning : total_earnings = 36 := by
  sorry

end Isabel_earning_l326_32620


namespace boat_speed_24_l326_32607

def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  let speed_downstream := x + 3
  let time := 1 / 4 -- 15 minutes in hours
  let distance := 6.75
  let equation := distance = speed_downstream * time
  equation ∧ x = 24

theorem boat_speed_24 (x : ℝ) (rate_of_current : ℝ) (time_minutes : ℝ) (distance_traveled : ℝ) 
  (h1 : rate_of_current = 3) (h2 : time_minutes = 15) (h3 : distance_traveled = 6.75) : speed_of_boat_in_still_water 24 := 
by
  -- Convert time in minutes to hours
  have time_in_hours : ℝ := time_minutes / 60
  -- Effective downstream speed
  have effective_speed := 24 + rate_of_current
  -- The equation to be satisfied
  have equation := distance_traveled = effective_speed * time_in_hours
  -- Simplify and solve
  sorry

end boat_speed_24_l326_32607


namespace base_k_132_eq_30_l326_32683

theorem base_k_132_eq_30 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_132_eq_30_l326_32683


namespace model_y_completion_time_l326_32618

theorem model_y_completion_time :
  ∀ (T : ℝ), (∃ k ≥ 0, k = 20) →
  (∀ (task_completed_x_per_minute : ℝ), task_completed_x_per_minute = 1 / 60) →
  (∀ (task_completed_y_per_minute : ℝ), task_completed_y_per_minute = 1 / T) →
  (20 * (1 / 60) + 20 * (1 / T) = 1) →
  T = 30 :=
by
  sorry

end model_y_completion_time_l326_32618


namespace lorry_weight_l326_32668

theorem lorry_weight : 
  let empty_lorry_weight := 500
  let apples_weight := 10 * 55
  let oranges_weight := 5 * 45
  let watermelons_weight := 3 * 125
  let firewood_weight := 2 * 75
  let loaded_items_weight := apples_weight + oranges_weight + watermelons_weight + firewood_weight
  let total_weight := empty_lorry_weight + loaded_items_weight
  total_weight = 1800 :=
by 
  sorry

end lorry_weight_l326_32668


namespace remainder_when_divided_by_15_l326_32660

theorem remainder_when_divided_by_15 (c d : ℤ) (h1 : c % 60 = 47) (h2 : d % 45 = 14) : (c + d) % 15 = 1 :=
  sorry

end remainder_when_divided_by_15_l326_32660


namespace sum_of_fourth_powers_of_consecutive_integers_l326_32675

-- Definitions based on conditions
def consecutive_squares_sum (x : ℤ) : Prop :=
  (x - 1)^2 + x^2 + (x + 1)^2 = 12246

-- Statement of the problem
theorem sum_of_fourth_powers_of_consecutive_integers (x : ℤ)
  (h : consecutive_squares_sum x) : 
  (x - 1)^4 + x^4 + (x + 1)^4 = 50380802 :=
sorry

end sum_of_fourth_powers_of_consecutive_integers_l326_32675


namespace total_canoes_built_by_april_l326_32661

theorem total_canoes_built_by_april
  (initial : ℕ)
  (production_increase : ℕ → ℕ) 
  (total_canoes : ℕ) :
  initial = 5 →
  (∀ n, production_increase n = 3 * n) →
  total_canoes = initial + production_increase initial + production_increase (production_increase initial) + production_increase (production_increase (production_increase initial)) →
  total_canoes = 200 :=
by
  intros h_initial h_production h_total
  sorry

end total_canoes_built_by_april_l326_32661


namespace jaymee_older_than_twice_shara_l326_32626

-- Given conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 22

-- Theorem to prove how many years older Jaymee is than twice Shara's age
theorem jaymee_older_than_twice_shara : jaymee_age - 2 * shara_age = 2 := by
  sorry

end jaymee_older_than_twice_shara_l326_32626


namespace tan_of_sin_in_interval_l326_32651

theorem tan_of_sin_in_interval (α : ℝ) (h1 : Real.sin α = 4 / 5) (h2 : 0 < α ∧ α < Real.pi) :
  Real.tan α = 4 / 3 ∨ Real.tan α = -4 / 3 :=
  sorry

end tan_of_sin_in_interval_l326_32651


namespace ring_toss_total_earnings_l326_32623

noncomputable def daily_earnings : ℕ := 144
noncomputable def number_of_days : ℕ := 22
noncomputable def total_earnings : ℕ := daily_earnings * number_of_days

theorem ring_toss_total_earnings :
  total_earnings = 3168 := by
  sorry

end ring_toss_total_earnings_l326_32623


namespace correct_calculation_l326_32613

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := 
by sorry

end correct_calculation_l326_32613


namespace solve_for_x_l326_32601

theorem solve_for_x (x : ℝ) (h : 5 / (4 + 1 / x) = 1) : x = 1 :=
by
  sorry

end solve_for_x_l326_32601


namespace reciprocal_neg_one_div_2022_l326_32657

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end reciprocal_neg_one_div_2022_l326_32657


namespace number_of_refuels_needed_l326_32690

noncomputable def fuelTankCapacity : ℕ := 50
noncomputable def distanceShanghaiHarbin : ℕ := 2560
noncomputable def fuelConsumptionRate : ℕ := 8
noncomputable def safetyFuel : ℕ := 6

theorem number_of_refuels_needed
  (fuelTankCapacity : ℕ)
  (distanceShanghaiHarbin : ℕ)
  (fuelConsumptionRate : ℕ)
  (safetyFuel : ℕ) :
  (fuelTankCapacity = 50) →
  (distanceShanghaiHarbin = 2560) →
  (fuelConsumptionRate = 8) →
  (safetyFuel = 6) →
  ∃ n : ℕ, n = 4 := by
  sorry

end number_of_refuels_needed_l326_32690


namespace total_pokemon_cards_l326_32637

def initial_cards : Nat := 27
def received_cards : Nat := 41
def lost_cards : Nat := 20

theorem total_pokemon_cards : initial_cards + received_cards - lost_cards = 48 := by
  sorry

end total_pokemon_cards_l326_32637


namespace inequality_solution_set_l326_32696

theorem inequality_solution_set :
  {x : ℝ | (x / (x ^ 2 - 8 * x + 15) ≥ 2) ∧ (x ^ 2 - 8 * x + 15 ≠ 0)} =
  {x : ℝ | (5 / 2 ≤ x ∧ x < 3) ∨ (5 < x ∧ x ≤ 6)} :=
by
  -- The proof is omitted
  sorry

end inequality_solution_set_l326_32696


namespace board_numbers_l326_32697

theorem board_numbers (a b c : ℕ) (h1 : a = 3) (h2 : b = 9) (h3 : c = 15)
    (op : ∀ x y z : ℕ, (x = y + z - t) → true)  -- simplifying the operation representation
    (min_number : ∃ x, x = 2013) : ∃ n m, n = 2019 ∧ m = 2025 := 
sorry

end board_numbers_l326_32697


namespace vec_eq_l326_32685

def a : ℝ × ℝ := (-1, 0)
def b : ℝ × ℝ := (0, 2)

theorem vec_eq : (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) = (-2, -6) := by
  sorry

end vec_eq_l326_32685


namespace product_of_primes_l326_32622

theorem product_of_primes :
  let p1 := 11
  let p2 := 13
  let p3 := 997
  p1 * p2 * p3 = 142571 :=
by
  sorry

end product_of_primes_l326_32622


namespace a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l326_32629

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l326_32629


namespace black_pens_count_l326_32640

variable (T B : ℕ)
variable (h1 : (3/10:ℚ) * T = 12)
variable (h2 : (1/5:ℚ) * T = B)

theorem black_pens_count (h1 : (3/10:ℚ) * T = 12) (h2 : (1/5:ℚ) * T = B) : B = 8 := by
  sorry

end black_pens_count_l326_32640


namespace second_set_parallel_lines_l326_32644

theorem second_set_parallel_lines (n : ℕ) (h1 : 5 * (n - 1) = 420) : n = 85 :=
by sorry

end second_set_parallel_lines_l326_32644


namespace factorize_difference_of_squares_l326_32627

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) :=
sorry

end factorize_difference_of_squares_l326_32627


namespace min_value_f_max_value_bac_l326_32615

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| - |x - 1|

theorem min_value_f : ∃ k : ℝ, (∀ x : ℝ, f x ≥ k) ∧ k = -2 := 
by
  sorry

theorem max_value_bac (a b c : ℝ) 
  (h1 : a^2 + c^2 + b^2 / 2 = 2) : 
  ∃ m : ℝ, (∀ a b c : ℝ, a^2 + c^2 + b^2 / 2 = 2 → b * (a + c) ≤ m) ∧ m = 2 := 
by
  sorry

end min_value_f_max_value_bac_l326_32615


namespace total_cost_of_books_l326_32691

theorem total_cost_of_books
  (C1 : ℝ)
  (C2 : ℝ)
  (H1 : C1 = 285.8333333333333)
  (H2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2327.5 :=
by
  sorry

end total_cost_of_books_l326_32691


namespace R_l326_32639

variable (a d n : ℕ)

def arith_sum (k : ℕ) : ℕ :=
  k * (a + (k - 1) * d / 2)

def s1 := arith_sum n
def s2 := arith_sum (3 * n)
def s3 := arith_sum (5 * n)
def s4 := arith_sum (7 * n)

def R' := s4 - s3 - s2

theorem R'_depends_on_d_n : 
  R' = 2 * d * n^2 := 
by 
  sorry

end R_l326_32639


namespace anand_present_age_l326_32621

theorem anand_present_age (A B : ℕ) 
  (h1 : B = A + 10)
  (h2 : A - 10 = (B - 10) / 3) :
  A = 15 :=
sorry

end anand_present_age_l326_32621


namespace young_li_age_l326_32630

theorem young_li_age (x : ℝ) (old_li_age : ℝ) 
  (h1 : old_li_age = 2.5 * x)  
  (h2 : old_li_age + 10 = 2 * (x + 10)) : 
  x = 20 := 
by
  sorry

end young_li_age_l326_32630


namespace compute_sum_of_squares_l326_32645

noncomputable def polynomial_roots (p q r : ℂ) : Prop := 
  (p^3 - 15 * p^2 + 22 * p - 8 = 0) ∧ 
  (q^3 - 15 * q^2 + 22 * q - 8 = 0) ∧ 
  (r^3 - 15 * r^2 + 22 * r - 8 = 0) 

theorem compute_sum_of_squares (p q r : ℂ) (h : polynomial_roots p q r) :
  (p + q) ^ 2 + (q + r) ^ 2 + (r + p) ^ 2 = 406 := 
sorry

end compute_sum_of_squares_l326_32645


namespace current_value_l326_32625

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l326_32625


namespace find_abcde_l326_32631

theorem find_abcde (N : ℕ) (a b c d e f : ℕ) (h : a ≠ 0) 
(h1 : N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f)
(h2 : (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) :
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437 :=
by sorry

end find_abcde_l326_32631


namespace ellipse_eccentricity_equilateral_triangle_l326_32605

theorem ellipse_eccentricity_equilateral_triangle
  (c a : ℝ) (h : c / a = 1 / 2) : eccentricity = 1 / 2 :=
by
  -- Proof goes here, we add sorry to skip proof content
  sorry

end ellipse_eccentricity_equilateral_triangle_l326_32605


namespace player_A_prize_received_event_A_not_low_probability_l326_32634

-- Condition Definitions
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3
def a : ℚ := 243

-- Part 1: Player A's Prize
theorem player_A_prize_received :
  (a * (p * p + 3 * p * (1 - p) * p + 3 * (1 - p) * p * p + (1 - p) * (1 - p) * p * p)) = 216 := sorry

-- Part 2: Probability of Event A with Low Probability Conditions
def low_probability_event (prob : ℚ) : Prop := prob < 0.05

-- Probability that player B wins the entire prize
def event_A_probability (p : ℚ) : ℚ :=
  (1 - p) ^ 3 + 3 * p * (1 - p) ^ 3

theorem event_A_not_low_probability (p : ℚ) (hp : p ≥ 3 / 4) :
  ¬ low_probability_event (event_A_probability p) := sorry

end player_A_prize_received_event_A_not_low_probability_l326_32634


namespace smallest_class_size_l326_32692

theorem smallest_class_size (n : ℕ) (h : 5 * n + 2 > 40) : 5 * n + 2 ≥ 42 :=
by
  sorry

end smallest_class_size_l326_32692


namespace soda_cost_l326_32667

theorem soda_cost (b s : ℕ) 
  (h₁ : 3 * b + 2 * s = 450) 
  (h₂ : 2 * b + 3 * s = 480) : 
  s = 108 := 
by
  sorry

end soda_cost_l326_32667


namespace imaginary_power_sum_zero_l326_32665

theorem imaginary_power_sum_zero (i : ℂ) (n : ℤ) (h : i^2 = -1) :
  i^(2*n - 3) + i^(2*n - 1) + i^(2*n + 1) + i^(2*n + 3) = 0 :=
by {
  sorry
}

end imaginary_power_sum_zero_l326_32665


namespace number_of_ways_to_partition_22_as_triangle_pieces_l326_32681

theorem number_of_ways_to_partition_22_as_triangle_pieces : 
  (∃ (a b c : ℕ), a + b + c = 22 ∧ a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃! (count : ℕ), count = 10 :=
by sorry

end number_of_ways_to_partition_22_as_triangle_pieces_l326_32681


namespace min_bottles_needed_l326_32676

theorem min_bottles_needed (bottle_size : ℕ) (min_ounces : ℕ) (n : ℕ) 
  (h1 : bottle_size = 15) 
  (h2 : min_ounces = 195) 
  (h3 : 15 * n >= 195) : n = 13 :=
sorry

end min_bottles_needed_l326_32676


namespace profit_at_original_price_l326_32673

theorem profit_at_original_price (x : ℝ) (h : 0.8 * x = 1.2) : x - 1 = 0.5 :=
by
  sorry

end profit_at_original_price_l326_32673


namespace find_angle_C_l326_32603

-- Definitions based on conditions
variables (α β γ : ℝ) -- Angles of the triangle

-- Condition: Angles between the altitude and the angle bisector at vertices A and B are equal
-- This implies α = β
def angles_equal (α β : ℝ) : Prop :=
  α = β

-- Condition: Sum of the angles in a triangle is 180 degrees
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Condition: Angle at vertex C is greater than angles at vertices A and B
def c_greater_than_a_and_b (α γ : ℝ) : Prop :=
  γ > α

-- The proof problem: Prove γ = 120 degrees given the conditions
theorem find_angle_C (α β γ : ℝ) (h1 : angles_equal α β) (h2 : angles_sum_to_180 α β γ) (h3 : c_greater_than_a_and_b α γ) : γ = 120 :=
by
  sorry

end find_angle_C_l326_32603


namespace pool_filled_in_48_minutes_with_both_valves_open_l326_32643

def rate_first_valve_fills_pool_in_2_hours (V1 : ℚ) : Prop :=
  V1 * 120 = 12000

def rate_second_valve_50_more_than_first (V1 V2 : ℚ) : Prop :=
  V2 = V1 + 50

def pool_capacity : ℚ := 12000

def combined_rate (V1 V2 combinedRate : ℚ) : Prop :=
  combinedRate = V1 + V2

def time_to_fill_pool_with_both_valves_open (combinedRate time : ℚ) : Prop :=
  time = pool_capacity / combinedRate

theorem pool_filled_in_48_minutes_with_both_valves_open
  (V1 V2 combinedRate time : ℚ) :
  rate_first_valve_fills_pool_in_2_hours V1 →
  rate_second_valve_50_more_than_first V1 V2 →
  combined_rate V1 V2 combinedRate →
  time_to_fill_pool_with_both_valves_open combinedRate time →
  time = 48 :=
by
  intros
  sorry

end pool_filled_in_48_minutes_with_both_valves_open_l326_32643


namespace geometric_sequence_common_ratio_l326_32671

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 2 * a 0 + a 1 = a 2)
  : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l326_32671


namespace Carol_max_chance_l326_32642

-- Definitions of the conditions
def Alice_random_choice (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def Bob_random_choice (b : ℝ) : Prop := 0.4 ≤ b ∧ b ≤ 0.6
def Carol_wins (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Statement that Carol maximizes her chances by picking 0.5
theorem Carol_max_chance : ∃ c : ℝ, (∀ a b : ℝ, Alice_random_choice a → Bob_random_choice b → Carol_wins a b c) ∧ c = 0.5 := 
sorry

end Carol_max_chance_l326_32642


namespace seedling_costs_and_purchase_l326_32698

variable (cost_A cost_B : ℕ)
variable (m n : ℕ)

-- Conditions
def conditions : Prop :=
  (cost_A = cost_B + 5) ∧ 
  (400 / cost_A = 300 / cost_B)

-- Prove costs and purchase for minimal costs
theorem seedling_costs_and_purchase (cost_A cost_B : ℕ) (m n : ℕ)
  (h1 : conditions cost_A cost_B)
  (h2 : m + n = 150)
  (h3 : m ≥ n / 2)
  : cost_A = 20 ∧ cost_B = 15 ∧ 5 * 50 + 2250 = 2500 
  := by
  sorry

end seedling_costs_and_purchase_l326_32698


namespace least_positive_multiple_of_17_gt_500_l326_32624

theorem least_positive_multiple_of_17_gt_500 : ∃ n: ℕ, n > 500 ∧ n % 17 = 0 ∧ n = 510 := by
  sorry

end least_positive_multiple_of_17_gt_500_l326_32624


namespace max_value_frac_l326_32609

theorem max_value_frac (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  ∃ z, z = (x + y) / x ∧ z ≤ 2 / 3 := by
  sorry

end max_value_frac_l326_32609


namespace smallest_c_plus_d_l326_32694

theorem smallest_c_plus_d :
  ∃ (c d : ℕ), (8 * c + 3 = 3 * d + 8) ∧ c + d = 27 :=
by
  sorry

end smallest_c_plus_d_l326_32694


namespace solve_for_y_l326_32674

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l326_32674


namespace problem_1_problem_2_l326_32638

-- (1) Conditions and proof statement
theorem problem_1 (x y m : ℝ) (P : ℝ × ℝ) (k : ℝ) :
  (x, y) = (1, 2) → m = 1 →
  ((x - 1)^2 + (y - 2)^2 = 4) →
  P = (3, -1) →
  (l : ℝ → ℝ → Prop) →
  (∀ x y, l x y ↔ x = 3 ∨ (5 * x + 12 * y - 3 = 0)) →
  l 3 (-1) →
  l (x + k * (3 - x)) (y-1) := sorry

-- (2) Conditions and proof statement
theorem problem_2 (x y m : ℝ) (line : ℝ → ℝ) :
  (x - 1)^2 + (y - 2)^2 = 5 - m →
  m < 5 →
  (2 * (5 - m - 20) ^ (1/2) = 2 * (5) ^ (1/2)) →
  m = -20 := sorry

end problem_1_problem_2_l326_32638


namespace product_fraction_simplification_l326_32688

theorem product_fraction_simplification : 
  (1^4 - 1) / (1^4 + 1) * (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) *
  (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) *
  (7^4 - 1) / (7^4 + 1) = 50 := 
  sorry

end product_fraction_simplification_l326_32688


namespace called_back_students_l326_32612

/-- Given the number of girls, boys, and students who didn't make the cut,
    this theorem proves the number of students who got called back. -/
theorem called_back_students (girls boys didnt_make_the_cut : ℕ)
    (h_girls : girls = 39)
    (h_boys : boys = 4)
    (h_didnt_make_the_cut : didnt_make_the_cut = 17) :
    girls + boys - didnt_make_the_cut = 26 := by
  sorry

end called_back_students_l326_32612


namespace find_ratio_l326_32604

noncomputable def ratio_CN_AN (BM MC BK AB CN AN : ℝ) (h1 : BM / MC = 4 / 5) (h2 : BK / AB = 1 / 5) : Prop :=
  CN / AN = 5 / 24

theorem find_ratio (BM MC BK AB CN AN : ℝ) (h1 : BM / MC = 4 / 5) (h2 : BK / AB = 1 / 5) (h3 : BM + MC = BC) (h4 : BK = BK) (h5 : BK + AB = 6 * BK) : 
  ratio_CN_AN BM MC BK AB CN AN h1 h2 :=
by
  sorry

end find_ratio_l326_32604


namespace largest_k_for_positive_root_l326_32632

theorem largest_k_for_positive_root : ∃ k : ℤ, k = 1 ∧ ∀ k' : ℤ, (k' > 1) → ¬ (∃ x > 0, 3 * x * (2 * k' * x - 5) - 2 * x^2 + 8 = 0) :=
by
  sorry

end largest_k_for_positive_root_l326_32632


namespace quadratic_complete_square_l326_32684

theorem quadratic_complete_square : 
  ∃ d e : ℝ, ((x^2 - 16*x + 15) = ((x + d)^2 + e)) ∧ (d + e = -57) := by
  sorry

end quadratic_complete_square_l326_32684


namespace arithmetic_sequence_sum_l326_32610

variable (S : ℕ → ℕ) -- Define a function S that gives the sum of the first n terms.
variable (n : ℕ)     -- Define a natural number n.

-- Conditions based on the problem statement
axiom h1 : S n = 3
axiom h2 : S (2 * n) = 10

-- The theorem we need to prove
theorem arithmetic_sequence_sum : S (3 * n) = 21 :=
by
  sorry

end arithmetic_sequence_sum_l326_32610


namespace expression_simplifies_to_zero_l326_32662

theorem expression_simplifies_to_zero (x y : ℝ) (h : x = 2024) :
    5 * (x ^ 3 - 3 * x ^ 2 * y - 2 * x * y ^ 2) -
    3 * (x ^ 3 - 5 * x ^ 2 * y + 2 * y ^ 3) +
    2 * (-x ^ 3 + 5 * x * y ^ 2 + 3 * y ^ 3) = 0 :=
by {
    sorry
}

end expression_simplifies_to_zero_l326_32662


namespace least_tiles_required_l326_32614

def floor_length : ℕ := 5000
def floor_breadth : ℕ := 1125
def gcd_floor : ℕ := Nat.gcd floor_length floor_breadth
def tile_area : ℕ := gcd_floor ^ 2
def floor_area : ℕ := floor_length * floor_breadth
def tiles_count : ℕ := floor_area / tile_area

theorem least_tiles_required : tiles_count = 360 :=
by
  sorry

end least_tiles_required_l326_32614


namespace chickens_and_rabbits_l326_32666

theorem chickens_and_rabbits (c r : ℕ) (h1 : c + r = 15) (h2 : 2 * c + 4 * r = 40) : c = 10 ∧ r = 5 :=
sorry

end chickens_and_rabbits_l326_32666


namespace solution_set_a_neg5_solution_set_general_l326_32664

theorem solution_set_a_neg5 (x : ℝ) : (-5 * x^2 + 3 * x + 2 > 0) ↔ (-2/5 < x ∧ x < 1) := 
sorry

theorem solution_set_general (a x : ℝ) : 
  (ax^2 + (a + 3) * x + 3 > 0) ↔
  ((0 < a ∧ a < 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 3 ∧ x ≠ -1) ∨ 
   (a > 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 0 ∧ x > -1) ∨ 
   (a < 0 ∧ -1 < x ∧ x < -3/a)) := 
sorry

end solution_set_a_neg5_solution_set_general_l326_32664


namespace cost_of_scissor_l326_32649

noncomputable def scissor_cost (initial_money: ℕ) (scissors: ℕ) (eraser_count: ℕ) (eraser_cost: ℕ) (remaining_money: ℕ) :=
  (initial_money - remaining_money - (eraser_count * eraser_cost)) / scissors

theorem cost_of_scissor : scissor_cost 100 8 10 4 20 = 5 := 
by 
  sorry 

end cost_of_scissor_l326_32649
