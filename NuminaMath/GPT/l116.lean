import Mathlib

namespace find_k_l116_11652

noncomputable def is_perfect_square (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ a : ℝ, x^2 + 2*(k-1)*x + 64 = (x + a)^2

theorem find_k (k : ℝ) : is_perfect_square k ↔ (k = 9 ∨ k = -7) :=
sorry

end find_k_l116_11652


namespace complementary_angles_difference_l116_11639

-- Given that the measures of two complementary angles are in the ratio 4:1,
-- we want to prove that the positive difference between the measures of the two angles is 54 degrees.

theorem complementary_angles_difference (x : ℝ) (h_complementary : 4 * x + x = 90) : 
  abs (4 * x - x) = 54 :=
by
  sorry

end complementary_angles_difference_l116_11639


namespace exponent_of_5_in_30_fact_l116_11646

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l116_11646


namespace betty_blue_beads_l116_11669

theorem betty_blue_beads (r b : ℕ) (h1 : r = 30) (h2 : 3 * b = 2 * r) : b = 20 :=
by
  sorry

end betty_blue_beads_l116_11669


namespace cow_value_increase_l116_11689

theorem cow_value_increase :
  let starting_weight : ℝ := 732
  let increase_factor : ℝ := 1.35
  let price_per_pound : ℝ := 2.75
  let new_weight := starting_weight * increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  let increase_in_value := value_at_new_weight - value_at_starting_weight
  increase_in_value = 704.55 :=
by
  sorry

end cow_value_increase_l116_11689


namespace bookmarks_per_day_l116_11653

theorem bookmarks_per_day (pages_now : ℕ) (pages_end_march : ℕ) (days_in_march : ℕ) (pages_added : ℕ) (pages_per_day : ℕ)
  (h1 : pages_now = 400)
  (h2 : pages_end_march = 1330)
  (h3 : days_in_march = 31)
  (h4 : pages_added = pages_end_march - pages_now)
  (h5 : pages_per_day = pages_added / days_in_march) :
  pages_per_day = 30 := sorry

end bookmarks_per_day_l116_11653


namespace find_total_original_cost_l116_11699

noncomputable def original_total_cost (x y z : ℝ) : ℝ :=
x + y + z

theorem find_total_original_cost (x y z : ℝ)
  (h1 : x * 1.30 = 351)
  (h2 : y * 1.25 = 275)
  (h3 : z * 1.20 = 96) :
  original_total_cost x y z = 570 :=
sorry

end find_total_original_cost_l116_11699


namespace det_A_is_half_l116_11660

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
![![Real.cos (20 * Real.pi / 180), Real.sin (40 * Real.pi / 180)], ![Real.sin (20 * Real.pi / 180), Real.cos (40 * Real.pi / 180)]]

theorem det_A_is_half : A.det = 1 / 2 := by
  sorry

end det_A_is_half_l116_11660


namespace integral_identity_proof_l116_11674

noncomputable def integral_identity : Prop :=
  ∫ x in (0 : Real)..(Real.pi / 2), (Real.cos (Real.cos x))^2 + (Real.sin (Real.sin x))^2 = Real.pi / 2

theorem integral_identity_proof : integral_identity :=
sorry

end integral_identity_proof_l116_11674


namespace solution_set_of_inequality_l116_11665

theorem solution_set_of_inequality (x : ℝ) :
  (|x| - 2) * (x - 1) ≥ 0 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l116_11665


namespace contains_all_integers_l116_11684

def is_closed_under_divisors (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, b ∣ a → a ∈ A → b ∈ A

def contains_product_plus_one (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, 1 < a → a < b → a ∈ A → b ∈ A → (1 + a * b) ∈ A

theorem contains_all_integers
  (A : Set ℕ)
  (h1 : is_closed_under_divisors A)
  (h2 : contains_product_plus_one A)
  (h3 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ 1 < a ∧ 1 < b ∧ 1 < c) :
  ∀ n : ℕ, n > 0 → n ∈ A := 
  by 
    sorry

end contains_all_integers_l116_11684


namespace trim_length_l116_11673

theorem trim_length {π : ℝ} (r : ℝ)
  (π_approx : π = 22 / 7)
  (area : π * r^2 = 616) :
  2 * π * r + 5 = 93 :=
by
  sorry

end trim_length_l116_11673


namespace find_e_l116_11624

variable (p j t e : ℝ)

def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - e / 100)

theorem find_e (h1 : condition1 p j)
               (h2 : condition2 j t)
               (h3 : condition3 t e p) : e = 6.25 :=
by sorry

end find_e_l116_11624


namespace exists_six_digit_no_identical_six_endings_l116_11626

theorem exists_six_digit_no_identical_six_endings :
  ∃ (A : ℕ), (100000 ≤ A ∧ A < 1000000) ∧ ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 500000) → 
  (∀ d, d ≠ 0 → d < 10 → (k * A) % 1000000 ≠ d * 111111) :=
by
  sorry

end exists_six_digit_no_identical_six_endings_l116_11626


namespace max_height_reached_l116_11675

def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

theorem max_height_reached : ∃ (t : ℝ), h t = 41.25 :=
by
  use 1.25
  sorry

end max_height_reached_l116_11675


namespace sum_of_numbers_odd_probability_l116_11688

namespace ProbabilityProblem

/-- 
  Given a biased die where the probability of rolling an even number is 
  twice the probability of rolling an odd number, and rolling the die three times,
  the probability that the sum of the numbers rolled is odd is 13/27.
-/
theorem sum_of_numbers_odd_probability :
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let prob_all_odd := (p_odd) ^ 3
  let prob_one_odd_two_even := 3 * (p_odd) * (p_even) ^ 2
  prob_all_odd + prob_one_odd_two_even = 13 / 27 :=
by
  sorry

end sum_of_numbers_odd_probability_l116_11688


namespace find_g1_l116_11668

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x : ℝ) (hx : x ≠ 1 / 2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x + 1

theorem find_g1 : g 1 = 39 / 11 :=
by
  sorry

end find_g1_l116_11668


namespace quadratic_one_real_root_positive_n_l116_11636

theorem quadratic_one_real_root_positive_n (n : ℝ) (h : (n ≠ 0)) :
  (∃ x : ℝ, (x^2 - 6*n*x - 9*n) = 0) ∧
  (∀ x y : ℝ, (x^2 - 6*n*x - 9*n) = 0 → (y^2 - 6*n*y - 9*n) = 0 → x = y) ↔
  n = 0 := by
  sorry

end quadratic_one_real_root_positive_n_l116_11636


namespace same_points_among_teams_l116_11623

theorem same_points_among_teams :
  ∀ (n : Nat), n = 28 → 
  ∀ (G D N : Nat), G = 378 → D >= 284 → N <= 94 →
  (∃ (team_scores : Fin n → Int), ∀ (i j : Fin n), i ≠ j → team_scores i = team_scores j) := by
sorry

end same_points_among_teams_l116_11623


namespace find_larger_number_l116_11633

variable {x y : ℕ} 

theorem find_larger_number (h_ratio : 4 * x = 3 * y) (h_sum : x + y + 100 = 500) : y = 1600 / 7 := by 
  sorry

end find_larger_number_l116_11633


namespace percentage_not_sophomores_l116_11612

variable (Total : ℕ) (Juniors Senior : ℕ) (Freshmen Sophomores : ℕ)

-- Conditions
axiom total_students : Total = 800
axiom percent_juniors : (22 / 100) * Total = Juniors
axiom number_seniors : Senior = 160
axiom freshmen_sophomores_relation : Freshmen = Sophomores + 64
axiom total_composition : Freshmen + Sophomores + Juniors + Senior = Total

-- Proof Objective
theorem percentage_not_sophomores :
  (Total - Sophomores) / Total * 100 = 75 :=
by
  -- proof omitted
  sorry

end percentage_not_sophomores_l116_11612


namespace cows_dogs_ratio_l116_11625

theorem cows_dogs_ratio (C D : ℕ) (hC : C = 184) (hC_remain : 3 / 4 * C = 138)
  (hD_remain : 1 / 4 * D + 138 = 161) : C / D = 2 :=
sorry

end cows_dogs_ratio_l116_11625


namespace remainder_poly_div_l116_11606

theorem remainder_poly_div 
    (x : ℤ) 
    (h1 : (x^2 + x + 1) ∣ (x^3 - 1)) 
    (h2 : x^5 - 1 = (x^3 - 1) * (x^2 + x + 1) - x * (x^2 + x + 1) + 1) : 
  ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 :=
by
  sorry

end remainder_poly_div_l116_11606


namespace triple_hash_100_l116_11615

def hash (N : ℝ) : ℝ :=
  0.5 * N + N

theorem triple_hash_100 : hash (hash (hash 100)) = 337.5 :=
by
  sorry

end triple_hash_100_l116_11615


namespace sqrt_a_squared_b_l116_11630

variable {a b : ℝ}

theorem sqrt_a_squared_b (h: a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end sqrt_a_squared_b_l116_11630


namespace find_f8_l116_11672

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x * f y
axiom initial_condition : f 2 = 4

theorem find_f8 : f 8 = 256 := by
  sorry

end find_f8_l116_11672


namespace find_minimal_sum_l116_11690

theorem find_minimal_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * (x + 1)) ∣ (y * (y + 1)) →
  ¬(x ∣ y ∨ x ∣ (y + 1)) →
  ¬((x + 1) ∣ y ∨ (x + 1) ∣ (y + 1)) →
  x = 14 ∧ y = 35 ∧ x^2 + y^2 = 1421 :=
sorry

end find_minimal_sum_l116_11690


namespace find_value_of_a_l116_11670

theorem find_value_of_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53 ^ 2017 + a) % 13 = 0) : a = 12 := 
by 
  sorry

end find_value_of_a_l116_11670


namespace rotated_line_equation_l116_11687

-- Define the original equation of the line
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the rotated line equation we want to prove
def rotated_line (x y : ℝ) : Prop := -x + 2 * y + 4 = 0

-- Proof problem statement in Lean 4
theorem rotated_line_equation :
  ∀ (x y : ℝ), original_line x y → rotated_line x y :=
by
  sorry

end rotated_line_equation_l116_11687


namespace solve_system_l116_11618

theorem solve_system (x y : ℝ) (h1 : 2 * x - y = 0) (h2 : x + 2 * y = 1) : 
  x = 1 / 5 ∧ y = 2 / 5 :=
by
  sorry

end solve_system_l116_11618


namespace common_speed_is_10_l116_11676

noncomputable def speed_jack (x : ℝ) : ℝ := x^2 - 11 * x - 22
noncomputable def speed_jill (x : ℝ) : ℝ := 
  if x = -6 then 0 else (x^2 - 4 * x - 12) / (x + 6)

theorem common_speed_is_10 (x : ℝ) (h : speed_jack x = speed_jill x) (hx : x = 16) : 
  speed_jack x = 10 :=
by
  sorry

end common_speed_is_10_l116_11676


namespace polynomial_representation_l116_11609

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end polynomial_representation_l116_11609


namespace marble_problem_l116_11631

-- Define the initial number of marbles
def initial_marbles : Prop :=
  ∃ (x y : ℕ), (y - 4 = 2 * (x + 4)) ∧ (y + 2 = 11 * (x - 2)) ∧ (y = 20) ∧ (x = 4)

-- The main theorem to prove the initial number of marbles
theorem marble_problem (x y : ℕ) (cond1 : y - 4 = 2 * (x + 4)) (cond2 : y + 2 = 11 * (x - 2)) :
  y = 20 ∧ x = 4 :=
sorry

end marble_problem_l116_11631


namespace D_72_is_22_l116_11659

def D (n : ℕ) : ℕ :=
   -- function definition for D that satisfies the problem's conditions
   sorry

theorem D_72_is_22 : D 72 = 22 :=
by sorry

end D_72_is_22_l116_11659


namespace relationship_among_numbers_l116_11645

theorem relationship_among_numbers :
  let a := 0.7 ^ 2.1
  let b := 0.7 ^ 2.5
  let c := 2.1 ^ 0.7
  b < a ∧ a < c := by
  sorry

end relationship_among_numbers_l116_11645


namespace angles_same_terminal_side_l116_11656

def angle_equiv (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angles_same_terminal_side : angle_equiv (-390 : ℝ) (330 : ℝ) :=
sorry

end angles_same_terminal_side_l116_11656


namespace roots_product_of_quadratic_equation_l116_11622

variables (a b : ℝ)

-- Given that a and b are roots of the quadratic equation x^2 - ax + b = 0
-- and given conditions that a + b = 5 and ab = 6,
-- prove that a * b = 6.
theorem roots_product_of_quadratic_equation 
  (h₁ : a + b = 5) 
  (h₂ : a * b = 6) : 
  a * b = 6 := 
by 
 sorry

end roots_product_of_quadratic_equation_l116_11622


namespace internet_plan_comparison_l116_11693

theorem internet_plan_comparison (d : ℕ) :
    3000 + 200 * d > 5000 → d > 10 :=
by
  intro h
  -- Proof will be written here
  sorry

end internet_plan_comparison_l116_11693


namespace find_common_ratio_l116_11666

variable (a_n : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

theorem find_common_ratio (h1 : a_n 1 = 2) (h2 : a_n 4 = 16) (h_geom : ∀ n, a_n n = a_n (n - 1) * q)
  : q = 2 := by
  sorry

end find_common_ratio_l116_11666


namespace number_of_ways_to_score_l116_11617

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end number_of_ways_to_score_l116_11617


namespace find_m_of_slope_is_12_l116_11677

theorem find_m_of_slope_is_12 (m : ℝ) :
  let A := (-m, 6)
  let B := (1, 3 * m)
  let slope := (3 * m - 6) / (1 + m)
  slope = 12 → m = -2 :=
by
  sorry

end find_m_of_slope_is_12_l116_11677


namespace math_problem_l116_11608

variables (x y z : ℝ)

theorem math_problem
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( (x^2 / (x + y) >= (3 * x - y) / 4) ) ∧ 
  ( (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) >= (x * y + y * z + z * x) / 2 ) :=
by sorry

end math_problem_l116_11608


namespace geometric_progression_product_l116_11621

theorem geometric_progression_product (n : ℕ) (S R : ℝ) (hS : S > 0) (hR : R > 0)
  (h_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ S = a * (q^n - 1) / (q - 1))
  (h_reciprocal_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ R = (1 - q^n) / (a * q^(n-1) * (q - 1))) :
  ∃ P : ℝ, P = (S / R)^(n / 2) := sorry

end geometric_progression_product_l116_11621


namespace dietitian_lunch_fraction_l116_11695

theorem dietitian_lunch_fraction
  (total_calories : ℕ)
  (recommended_calories : ℕ)
  (extra_calories : ℕ)
  (h1 : total_calories = 40)
  (h2 : recommended_calories = 25)
  (h3 : extra_calories = 5)
  : (recommended_calories + extra_calories) / total_calories = 3 / 4 :=
by
  sorry

end dietitian_lunch_fraction_l116_11695


namespace intersection_points_form_rectangle_l116_11644

theorem intersection_points_form_rectangle
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 + y^2 = 34) :
  ∃ (a b u v : ℝ), (a * b = 8) ∧ (a^2 + b^2 = 34) ∧ 
  (u * v = 8) ∧ (u^2 + v^2 = 34) ∧
  ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧ 
  ((u = -x ∧ v = -y) ∨ (u = -y ∧ v = -x)) ∧
  ((a = u ∧ b = v) ∨ (a = v ∧ b = u)) ∧ 
  ((x = -u ∧ y = -v) ∨ (x = -v ∧ y = -u)) ∧
  (
    (a, b) ≠ (u, v) ∧ (a, b) ≠ (-u, -v) ∧ 
    (a, b) ≠ (v, u) ∧ (a, b) ≠ (-v, -u) ∧
    (u, v) ≠ (-a, -b) ∧ (u, v) ≠ (b, a) ∧ 
    (u, v) ≠ (-b, -a)
  ) :=
by sorry

end intersection_points_form_rectangle_l116_11644


namespace corrected_mean_l116_11600

theorem corrected_mean (mean_initial : ℝ) (num_obs : ℕ) (obs_incorrect : ℝ) (obs_correct : ℝ) :
  mean_initial = 36 → num_obs = 50 → obs_incorrect = 23 → obs_correct = 30 →
  (mean_initial * ↑num_obs + (obs_correct - obs_incorrect)) / ↑num_obs = 36.14 :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_l116_11600


namespace sum_of_first_five_terms_l116_11682

noncomputable def S₅ (a : ℕ → ℝ) := (a 1 + a 5) / 2 * 5

theorem sum_of_first_five_terms (a : ℕ → ℝ) (a_2 a_4 : ℝ)
  (h1 : a 2 = 4)
  (h2 : a 4 = 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S₅ a = 15 :=
sorry

end sum_of_first_five_terms_l116_11682


namespace number_of_tests_initially_l116_11647

theorem number_of_tests_initially (n : ℕ) (h1 : (90 * n) / n = 90)
  (h2 : ((90 * n) - 75) / (n - 1) = 95) : n = 4 :=
sorry

end number_of_tests_initially_l116_11647


namespace fraction_of_sum_l116_11640

theorem fraction_of_sum (l : List ℝ) (n : ℝ) (h_len : l.length = 21) (h_mem : n ∈ l)
  (h_n_avg : n = 4 * (l.erase n).sum / 20) :
  n / l.sum = 1 / 6 := by
  sorry

end fraction_of_sum_l116_11640


namespace dragons_total_games_l116_11634

noncomputable def numberOfGames (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) : ℕ :=
y + 12

theorem dragons_total_games (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) :
  numberOfGames y x h1 h2 = 90 := 
sorry

end dragons_total_games_l116_11634


namespace license_plate_increase_l116_11694

theorem license_plate_increase :
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  new_plates / old_plates = (900 / 17576) * 100 :=
by
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  have h : new_plates / old_plates = (900 / 17576) * 100 := sorry
  exact h

end license_plate_increase_l116_11694


namespace necessary_and_sufficient_condition_l116_11663

theorem necessary_and_sufficient_condition (x : ℝ) :
  x > 0 ↔ x + 1/x ≥ 2 :=
by sorry

end necessary_and_sufficient_condition_l116_11663


namespace oprod_eval_l116_11649

def oprod (a b : ℕ) : ℕ :=
  (a * 2 + b) / 2

theorem oprod_eval : oprod (oprod 4 6) 8 = 11 :=
by
  -- Definitions given in conditions
  let r := (4 * 2 + 6) / 2
  have h1 : oprod 4 6 = r := by rfl
  let s := (r * 2 + 8) / 2
  have h2 : oprod r 8 = s := by rfl
  exact (show s = 11 from sorry)

end oprod_eval_l116_11649


namespace equation_has_one_integral_root_l116_11642

theorem equation_has_one_integral_root:
  ∃ x : ℤ, (x - 9 / (x + 4 : ℝ) = 2 - 9 / (x + 4 : ℝ)) ∧ ∀ y : ℤ, 
  (y - 9 / (y + 4 : ℝ) = 2 - 9 / (y + 4 : ℝ)) → y = x := 
by
  sorry

end equation_has_one_integral_root_l116_11642


namespace credit_card_more_beneficial_l116_11681

def gift_cost : ℝ := 8000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.0075
def debit_card_interest_rate : ℝ := 0.005

def credit_card_total_income : ℝ := gift_cost * (credit_card_cashback_rate + debit_card_interest_rate)
def debit_card_total_income : ℝ := gift_cost * debit_card_cashback_rate

theorem credit_card_more_beneficial :
  credit_card_total_income > debit_card_total_income :=
by
  sorry

end credit_card_more_beneficial_l116_11681


namespace fraction_to_decimal_l116_11671

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l116_11671


namespace matt_needs_38_plates_l116_11664

def plates_needed (days_with_only_matt_and_son days_with_parents plates_per_day plates_per_person_with_parents : ℕ) : ℕ :=
  (days_with_only_matt_and_son * plates_per_day) + (days_with_parents * 4 * plates_per_person_with_parents)

theorem matt_needs_38_plates :
  plates_needed 3 4 2 2 = 38 :=
by
  sorry

end matt_needs_38_plates_l116_11664


namespace original_proposition_true_converse_false_l116_11657

-- Lean 4 statement for the equivalent proof problem
theorem original_proposition_true_converse_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬((a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_false_l116_11657


namespace number_of_students_l116_11601

theorem number_of_students 
    (N : ℕ) 
    (h_percentage_5 : 28 * N % 100 = 0)
    (h_percentage_4 : 35 * N % 100 = 0)
    (h_percentage_3 : 25 * N % 100 = 0)
    (h_percentage_2 : 12 * N % 100 = 0)
    (h_class_limit : N ≤ 4 * 30) 
    (h_num_classes : 4 * 30 < 120)
    : N = 100 := 
by 
  sorry

end number_of_students_l116_11601


namespace tangent_line_equation_at_point_l116_11607

theorem tangent_line_equation_at_point 
  (x y : ℝ) (h_curve : y = x^3 - 2 * x) (h_point : (x, y) = (1, -1)) : 
  (x - y - 2 = 0) := 
sorry

end tangent_line_equation_at_point_l116_11607


namespace division_result_l116_11661

theorem division_result (x : ℕ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end division_result_l116_11661


namespace find_sum_of_pqr_l116_11611

theorem find_sum_of_pqr (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) 
(h4 : q = p * (4 - p)) (h5 : r = q * (4 - q)) (h6 : p = r * (4 - r)) : 
p + q + r = 6 :=
sorry

end find_sum_of_pqr_l116_11611


namespace unusual_numbers_exist_l116_11613

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end unusual_numbers_exist_l116_11613


namespace necessary_but_not_sufficient_l116_11680

open Set

namespace Mathlib

noncomputable def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) :=
by
  sorry

end Mathlib

end necessary_but_not_sufficient_l116_11680


namespace toll_constant_l116_11620

theorem toll_constant (t : ℝ) (x : ℝ) (constant : ℝ) : 
  (t = 1.50 + 0.50 * (x - constant)) → 
  (x = 18 / 2) → 
  (t = 5) → 
  constant = 2 :=
by
  intros h1 h2 h3
  sorry

end toll_constant_l116_11620


namespace triangle_ab_value_l116_11696

theorem triangle_ab_value (a b c : ℝ) (A B C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by
  sorry

end triangle_ab_value_l116_11696


namespace urea_moles_produced_l116_11679

-- Define the reaction
def chemical_reaction (CO2 NH3 Urea Water : ℕ) :=
  CO2 = 1 ∧ NH3 = 2 ∧ Urea = 1 ∧ Water = 1

-- Given initial moles of reactants
def initial_moles (CO2 NH3 : ℕ) :=
  CO2 = 1 ∧ NH3 = 2

-- The main theorem to prove
theorem urea_moles_produced (CO2 NH3 Urea Water : ℕ) :
  initial_moles CO2 NH3 → chemical_reaction CO2 NH3 Urea Water → Urea = 1 :=
by
  intro H1 H2
  rcases H1 with ⟨HCO2, HNH3⟩
  rcases H2 with ⟨HCO2', HNH3', HUrea, _⟩
  sorry

end urea_moles_produced_l116_11679


namespace geometric_sequence_expression_l116_11637

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 2 = 1)
(h2 : a 3 * a 5 = 2 * a 7) : a n = 1 / 2 ^ (n - 2) :=
sorry

end geometric_sequence_expression_l116_11637


namespace stratified_sampling_major_C_l116_11643

theorem stratified_sampling_major_C
  (students_A : ℕ) (students_B : ℕ) (students_C : ℕ) (students_D : ℕ)
  (total_students : ℕ) (sample_size : ℕ)
  (hA : students_A = 150) (hB : students_B = 150) (hC : students_C = 400) (hD : students_D = 300)
  (hTotal : total_students = students_A + students_B + students_C + students_D)
  (hSample : sample_size = 40)
  : students_C * (sample_size / total_students) = 16 :=
by
  sorry

end stratified_sampling_major_C_l116_11643


namespace fraction_ratio_x_div_y_l116_11686

theorem fraction_ratio_x_div_y (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : y / (x + z) = (x - y) / z) 
(h5 : y / (x + z) = x / (y + 2 * z)) :
  x / y = 2 / 3 := 
  sorry

end fraction_ratio_x_div_y_l116_11686


namespace log_a1_plus_log_a9_l116_11648

variable {a : ℕ → ℝ}
variable {log : ℝ → ℝ}

-- Assume the provided conditions
axiom is_geometric_sequence : ∀ n, a (n + 1) / a n = a 1 / a 0
axiom a3a5a7_eq_one : a 3 * a 5 * a 7 = 1
axiom log_mul : ∀ x y, log (x * y) = log x + log y
axiom log_one_eq_zero : log 1 = 0

theorem log_a1_plus_log_a9 : log (a 1) + log (a 9) = 0 := 
by {
    sorry
}

end log_a1_plus_log_a9_l116_11648


namespace intersection_eq_l116_11605

open Set

variable {α : Type*}

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_eq : M ∩ N = {2, 3} := by
  apply Set.ext
  intro x
  simp [M, N]
  sorry

end intersection_eq_l116_11605


namespace polynomial_horner_v4_value_l116_11678

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define Horner's Rule step by step for x = 2
def horner_eval (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  let v4 := v3 * x + 240
  v4

-- Prove that the value of v4 when x = 2 is 80
theorem polynomial_horner_v4_value : horner_eval 2 = 80 := by
  sorry

end polynomial_horner_v4_value_l116_11678


namespace smallest_positive_integer_ends_in_7_and_divisible_by_5_l116_11658

theorem smallest_positive_integer_ends_in_7_and_divisible_by_5 : 
  ∃ n : ℤ, n > 0 ∧ n % 10 = 7 ∧ n % 5 = 0 ∧ n = 37 := 
by 
  sorry

end smallest_positive_integer_ends_in_7_and_divisible_by_5_l116_11658


namespace divisor_proof_l116_11614

def original_number : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

def remainder : ℕ := 36

theorem divisor_proof (D : ℕ) (Q : ℕ) (h : original_number = D * Q + remainder) : original_number % D = remainder :=
by 
  sorry

end divisor_proof_l116_11614


namespace gcd_calculation_l116_11650

theorem gcd_calculation : 
  Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := 
by
  sorry

end gcd_calculation_l116_11650


namespace brazil_medal_fraction_closest_l116_11632

theorem brazil_medal_fraction_closest :
  let frac_win : ℚ := 23 / 150
  let frac_1_6 : ℚ := 1 / 6
  let frac_1_7 : ℚ := 1 / 7
  let frac_1_8 : ℚ := 1 / 8
  let frac_1_9 : ℚ := 1 / 9
  let frac_1_10 : ℚ := 1 / 10
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_6) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_8) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_9) ∧
  abs (frac_win - frac_1_7) < abs (frac_win - frac_1_10) :=
by
  sorry

end brazil_medal_fraction_closest_l116_11632


namespace inequality_proof_l116_11638

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l116_11638


namespace fraction_product_l116_11662

theorem fraction_product :
  (7 / 4) * (8 / 14) * (16 / 24) * (32 / 48) * (28 / 7) * (15 / 9) *
  (50 / 25) * (21 / 35) = 32 / 3 :=
by
  sorry

end fraction_product_l116_11662


namespace bamboo_break_height_l116_11610

-- Conditions provided in the problem
def original_height : ℝ := 20  -- 20 chi
def distance_tip_to_root : ℝ := 6  -- 6 chi

-- Function to check if the height of the break satisfies the equation
def equationHolds (x : ℝ) : Prop :=
  (original_height - x) ^ 2 - x ^ 2 = distance_tip_to_root ^ 2

-- Main statement to prove the height of the break is 9.1 chi
theorem bamboo_break_height : equationHolds 9.1 :=
by
  sorry

end bamboo_break_height_l116_11610


namespace units_digit_of_fraction_l116_11628

theorem units_digit_of_fraction :
  ((30 * 31 * 32 * 33 * 34) / 400) % 10 = 4 :=
by
  sorry

end units_digit_of_fraction_l116_11628


namespace age_difference_proof_l116_11602

def AlexAge : ℝ := 16.9996700066
def AlexFatherAge (A : ℝ) (F : ℝ) : Prop := F = 2 * A + 4.9996700066
def FatherAgeSixYearsAgo (A : ℝ) (F : ℝ) : Prop := A - 6 = 1 / 3 * (F - 6)

theorem age_difference_proof :
  ∃ (A F : ℝ), A = 16.9996700066 ∧
  (AlexFatherAge A F) ∧
  (FatherAgeSixYearsAgo A F) :=
by
  sorry

end age_difference_proof_l116_11602


namespace sqrt_200_eq_10_sqrt_2_l116_11683

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l116_11683


namespace sangeun_initial_money_l116_11619

theorem sangeun_initial_money :
  ∃ (X : ℝ), 
  ((X / 2 - 2000) / 2 - 2000 = 0) ∧ 
  X = 12000 :=
by sorry

end sangeun_initial_money_l116_11619


namespace total_paint_remaining_l116_11697

-- Definitions based on the conditions
def paint_per_statue : ℚ := 1 / 16
def statues_to_paint : ℕ := 14

-- Theorem statement to prove the answer
theorem total_paint_remaining : (statues_to_paint : ℚ) * paint_per_statue = 7 / 8 := 
by sorry

end total_paint_remaining_l116_11697


namespace brooke_earns_144_dollars_l116_11685

-- Definitions based on the identified conditions
def price_of_milk_per_gallon : ℝ := 3
def production_cost_per_gallon_of_butter : ℝ := 0.5
def sticks_of_butter_per_gallon : ℝ := 2
def price_of_butter_per_stick : ℝ := 1.5
def number_of_cows : ℕ := 12
def milk_per_cow : ℝ := 4
def number_of_customers : ℕ := 6
def min_milk_per_customer : ℝ := 4
def max_milk_per_customer : ℝ := 8

-- Auxiliary calculations
def total_milk_produced : ℝ := number_of_cows * milk_per_cow
def min_total_customer_demand : ℝ := number_of_customers * min_milk_per_customer
def max_total_customer_demand : ℝ := number_of_customers * max_milk_per_customer

-- Problem statement
theorem brooke_earns_144_dollars :
  (0 <= total_milk_produced) ∧
  (min_total_customer_demand <= max_total_customer_demand) ∧
  (total_milk_produced = max_total_customer_demand) →
  (total_milk_produced * price_of_milk_per_gallon = 144) :=
by
  -- Sorry is added here since the proof is not required
  sorry

end brooke_earns_144_dollars_l116_11685


namespace game_ends_in_draw_for_all_n_l116_11651

noncomputable def andrey_representation_count (n : ℕ) : ℕ := 
  -- The function to count Andrey's representation should be defined here
  sorry

noncomputable def petya_representation_count (n : ℕ) : ℕ := 
  -- The function to count Petya's representation should be defined here
  sorry

theorem game_ends_in_draw_for_all_n (n : ℕ) (h : 0 < n) : 
  andrey_representation_count n = petya_representation_count n :=
  sorry

end game_ends_in_draw_for_all_n_l116_11651


namespace largest_non_expressible_number_l116_11603

theorem largest_non_expressible_number :
  ∀ (x y z : ℕ), 15 * x + 18 * y + 20 * z ≠ 97 :=
by sorry

end largest_non_expressible_number_l116_11603


namespace sample_processing_l116_11616

-- Define sample data
def standard: ℕ := 220
def samples: List ℕ := [230, 226, 218, 223, 214, 225, 205, 212]

-- Calculate deviations
def deviations (samples: List ℕ) (standard: ℕ) : List ℤ :=
  samples.map (λ x => x - standard)

-- Total dosage of samples
def total_dosage (samples: List ℕ): ℕ :=
  samples.sum

-- Total cost to process to standard dosage
def total_cost (deviations: List ℤ) (cost_per_ml_adjustment: ℤ) : ℤ :=
  cost_per_ml_adjustment * (deviations.map Int.natAbs).sum

-- Theorem statement
theorem sample_processing :
  let deviation_vals := deviations samples standard;
  let total_dosage_val := total_dosage samples;
  let total_cost_val := total_cost deviation_vals 10;
  deviation_vals = [10, 6, -2, 3, -6, 5, -15, -8] ∧
  total_dosage_val = 1753 ∧
  total_cost_val = 550 :=
by
  sorry

end sample_processing_l116_11616


namespace solve_linear_system_l116_11641

variable {x y : ℚ}

theorem solve_linear_system (h1 : 4 * x - 3 * y = -17) (h2 : 5 * x + 6 * y = -4) :
  (x, y) = (-(74 / 13 : ℚ), -(25 / 13 : ℚ)) :=
by
  sorry

end solve_linear_system_l116_11641


namespace minimum_common_ratio_l116_11604

theorem minimum_common_ratio (a : ℕ) (n : ℕ) (q : ℝ) (h_pos : ∀ i, i < n → 0 < a * q^i) (h_geom : ∀ i j, i < j → a * q^i < a * q^j) (h_q : 1 < q ∧ q < 2) : q = 6 / 5 :=
by
  sorry

end minimum_common_ratio_l116_11604


namespace greatest_root_f_l116_11635

noncomputable def f (x : ℝ) : ℝ := 21 * x ^ 4 - 20 * x ^ 2 + 3

theorem greatest_root_f :
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
sorry

end greatest_root_f_l116_11635


namespace fraction_calculation_l116_11654

theorem fraction_calculation :
  ((1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25) := 
by 
  sorry

end fraction_calculation_l116_11654


namespace proof_inequality_l116_11691

theorem proof_inequality (x : ℝ) : (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5 ∨ -9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end proof_inequality_l116_11691


namespace length_of_longer_leg_of_smallest_triangle_l116_11692

theorem length_of_longer_leg_of_smallest_triangle 
  (hypotenuse_largest : ℝ) 
  (h1 : hypotenuse_largest = 10)
  (h45 : ∀ hyp, (hyp / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2) = hypotenuse_largest / 4) :
  (hypotenuse_largest / 4) = 5 / 2 := by
  sorry

end length_of_longer_leg_of_smallest_triangle_l116_11692


namespace composite_evaluation_at_two_l116_11655

-- Define that P(x) is a polynomial with coefficients in {0, 1}
def is_binary_coefficient_polynomial (P : Polynomial ℤ) : Prop :=
  ∀ (n : ℕ), P.coeff n = 0 ∨ P.coeff n = 1

-- Define that P(x) can be factored into two nonconstant polynomials with integer coefficients
def is_reducible_to_nonconstant_polynomials (P : Polynomial ℤ) : Prop :=
  ∃ (f g : Polynomial ℤ), f.degree > 0 ∧ g.degree > 0 ∧ P = f * g

theorem composite_evaluation_at_two {P : Polynomial ℤ}
  (h1 : is_binary_coefficient_polynomial P)
  (h2 : is_reducible_to_nonconstant_polynomials P) :
  ∃ (m n : ℤ), m > 1 ∧ n > 1 ∧ P.eval 2 = m * n := sorry

end composite_evaluation_at_two_l116_11655


namespace find_line_eq_l116_11667

-- Definitions for the conditions
def passes_through_M (l : ℝ × ℝ) : Prop :=
  l = (1, 2)

def segment_intercepted_length (l : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ,
    ∀ p : ℝ × ℝ, l p → ((4 * p.1 + 3 * p.2 + 1 = 0 ∨ 4 * p.1 + 3 * p.2 + 6 = 0) ∧ (A = p ∨ B = p)) ∧
    dist A B = Real.sqrt 2

-- Predicates for the lines to be proven
def line_eq1 (p : ℝ × ℝ) : Prop :=
  p.1 + 7 * p.2 = 15

def line_eq2 (p : ℝ × ℝ) : Prop :=
  7 * p.1 - p.2 = 5

-- The proof problem statement
theorem find_line_eq (l : ℝ × ℝ → Prop) :
  passes_through_M (1, 2) →
  segment_intercepted_length l →
  (∀ p, l p → line_eq1 p) ∨ (∀ p, l p → line_eq2 p) :=
by
  sorry

end find_line_eq_l116_11667


namespace eiffel_tower_scale_l116_11698

theorem eiffel_tower_scale (height_model : ℝ) (height_actual : ℝ) (h_model : height_model = 30) (h_actual : height_actual = 984) : 
  height_actual / height_model = 32.8 := by
  sorry

end eiffel_tower_scale_l116_11698


namespace correct_statements_l116_11627

def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonic_increasing_on_neg1_0 : ∀ ⦃x y : ℝ⦄, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y
axiom functional_eqn (x : ℝ) : f (1 - x) + f (1 + x) = 0

theorem correct_statements :
  (∀ x, f (1 - x) = -f (1 + x)) ∧ f 2 ≤ f x :=
by
  sorry

end correct_statements_l116_11627


namespace length_of_each_part_l116_11629

-- Conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def parts_count : ℕ := 4

-- Question
theorem length_of_each_part : total_length_in_inches / parts_count = 20 :=
by
  -- leave the proof as a sorry
  sorry

end length_of_each_part_l116_11629
