import Mathlib

namespace initial_oranges_l1925_192545

theorem initial_oranges (O : ℕ) (h1 : O + 6 - 3 = 6) : O = 3 :=
by
  sorry

end initial_oranges_l1925_192545


namespace solution_z_sq_eq_neg_4_l1925_192544

theorem solution_z_sq_eq_neg_4 (x y : ℝ) (i : ℂ) (z : ℂ) (h : z = x + y * i) (hi : i^2 = -1) : 
  z^2 = -4 ↔ z = 2 * i ∨ z = -2 * i := 
by
  sorry

end solution_z_sq_eq_neg_4_l1925_192544


namespace number_of_friends_l1925_192568

-- Conditions/Definitions
def total_cost : ℤ := 13500
def cost_per_person : ℤ := 900

-- Prove that Dawson is going with 14 friends.
theorem number_of_friends (h1 : total_cost = 13500) (h2 : cost_per_person = 900) :
  (total_cost / cost_per_person) - 1 = 14 :=
by
  sorry

end number_of_friends_l1925_192568


namespace solve_for_x_l1925_192573

theorem solve_for_x :
  (∀ y : ℝ, 10 * x * y - 15 * y + 4 * x - 6 = 0) ↔ x = 3 / 2 :=
by
  sorry

end solve_for_x_l1925_192573


namespace maximum_n_l1925_192502

variable (x y z : ℝ)

theorem maximum_n (h1 : x + y + z = 12) (h2 : x * y + y * z + z * x = 30) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
by
  sorry

end maximum_n_l1925_192502


namespace tangent_line_at_P_range_of_a_l1925_192549

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := a * (x - 1/x) - Real.log x

-- Problem (Ⅰ): Tangent line equation at P(1, f(1)) for a = 1
theorem tangent_line_at_P (x : ℝ) (h : x = 1) : (∃ y : ℝ, f x 1 = y ∧ x - y - 1 = 0) := sorry

-- Problem (Ⅱ): Range of a for f(x) ≥ 0 ∀ x ≥ 1
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x ≥ 1 → f x a ≥ 0) : a ≥ 1/2 := sorry

end tangent_line_at_P_range_of_a_l1925_192549


namespace range_of_x_l1925_192575

noncomputable 
def proposition_p (x : ℝ) : Prop := 6 - 3 * x ≥ 0

noncomputable 
def proposition_q (x : ℝ) : Prop := 1 / (x + 1) < 0

theorem range_of_x (x : ℝ) : proposition_p x ∧ ¬proposition_q x → x ∈ Set.Icc (-1 : ℝ) (2 : ℝ) := by
  sorry

end range_of_x_l1925_192575


namespace taehyung_collected_most_points_l1925_192571

def largest_collector : Prop :=
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  taehyung_points > yoongi_points ∧ 
  taehyung_points > jungkook_points ∧ 
  taehyung_points > yuna_points ∧ 
  taehyung_points > yoojung_points

theorem taehyung_collected_most_points : largest_collector :=
by
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  sorry

end taehyung_collected_most_points_l1925_192571


namespace find_real_numbers_l1925_192564

theorem find_real_numbers :
  ∀ (x y z : ℝ), x^2 - y*z = |y - z| + 1 ∧ y^2 - z*x = |z - x| + 1 ∧ z^2 - x*y = |x - y| + 1 ↔
  (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
  (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
  (x = -5/3 ∧ y = 4/3 ∧ z = 4/3) ∨
  (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
  (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
  (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) :=
by
  sorry

end find_real_numbers_l1925_192564


namespace remainder_55_57_div_8_l1925_192535

def remainder (a b n : ℕ) := (a * b) % n

theorem remainder_55_57_div_8 : remainder 55 57 8 = 7 := by
  -- proof omitted
  sorry

end remainder_55_57_div_8_l1925_192535


namespace most_stable_performance_l1925_192522

theorem most_stable_performance :
  ∀ (σ2_A σ2_B σ2_C σ2_D : ℝ), 
  σ2_A = 0.56 → 
  σ2_B = 0.78 → 
  σ2_C = 0.42 → 
  σ2_D = 0.63 → 
  σ2_C ≤ σ2_A ∧ σ2_C ≤ σ2_B ∧ σ2_C ≤ σ2_D :=
by
  intros σ2_A σ2_B σ2_C σ2_D hA hB hC hD
  sorry

end most_stable_performance_l1925_192522


namespace number_of_family_members_l1925_192504

noncomputable def total_money : ℝ :=
  123 * 0.01 + 85 * 0.05 + 35 * 0.10 + 26 * 0.25

noncomputable def leftover_money : ℝ := 0.48

noncomputable def double_scoop_cost : ℝ := 3.0

noncomputable def amount_spent : ℝ := total_money - leftover_money

noncomputable def number_of_double_scoops : ℝ := amount_spent / double_scoop_cost

theorem number_of_family_members :
  number_of_double_scoops = 5 := by
  sorry

end number_of_family_members_l1925_192504


namespace exists_positive_x_for_inequality_l1925_192580

-- Define the problem conditions and the final proof goal.
theorem exists_positive_x_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Ico (-9/4 : ℝ) (2 : ℝ) :=
by
  sorry

end exists_positive_x_for_inequality_l1925_192580


namespace minimum_x_plus_y_l1925_192532

variable (x y : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1)

theorem minimum_x_plus_y (hx : 0 < x) (hy : 0 < y) (h : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1) : x + y ≥ 9 / 4 :=
sorry

end minimum_x_plus_y_l1925_192532


namespace cupcake_cost_l1925_192569

def initialMoney : ℝ := 20
def moneyFromMother : ℝ := 2 * initialMoney
def totalMoney : ℝ := initialMoney + moneyFromMother
def costPerBoxOfCookies : ℝ := 3
def numberOfBoxesOfCookies : ℝ := 5
def costOfCookies : ℝ := costPerBoxOfCookies * numberOfBoxesOfCookies
def moneyAfterCookies : ℝ := totalMoney - costOfCookies
def moneyLeftAfterCupcakes : ℝ := 30
def numberOfCupcakes : ℝ := 10

noncomputable def costPerCupcake : ℝ := 
  (moneyAfterCookies - moneyLeftAfterCupcakes) / numberOfCupcakes

theorem cupcake_cost :
  costPerCupcake = 1.50 :=
by 
  sorry

end cupcake_cost_l1925_192569


namespace polynomial_sum_l1925_192507

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l1925_192507


namespace tangent_line_solution_l1925_192518

variables (x y : ℝ)

noncomputable def circle_equation (m : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + m * y = 0

def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

theorem tangent_line_solution (m : ℝ) :
  point_on_circle m →
  m = 2 →
  tangent_line_equation 1 1 :=
by
  sorry

end tangent_line_solution_l1925_192518


namespace temperature_celsius_range_l1925_192570

theorem temperature_celsius_range (C : ℝ) :
  (∀ C : ℝ, let F_approx := 2 * C + 30;
             let F_exact := (9 / 5) * C + 32;
             abs ((2 * C + 30 - ((9 / 5) * C + 32)) / ((9 / 5) * C + 32)) ≤ 0.05) →
  (40 / 29) ≤ C ∧ C ≤ (360 / 11) :=
by
  intros h
  sorry

end temperature_celsius_range_l1925_192570


namespace cookies_baked_on_monday_is_32_l1925_192551

-- Definitions for the problem.
variable (X : ℕ)

-- Conditions.
def cookies_baked_on_monday := X
def cookies_baked_on_tuesday := X / 2
def cookies_baked_on_wednesday := 3 * (X / 2) - 4

-- Total cookies at the end of three days.
def total_cookies := cookies_baked_on_monday X + cookies_baked_on_tuesday X + cookies_baked_on_wednesday X

-- Theorem statement to prove the number of cookies baked on Monday.
theorem cookies_baked_on_monday_is_32 : total_cookies X = 92 → cookies_baked_on_monday X = 32 :=
by
  -- We would add the proof steps here.
  sorry

end cookies_baked_on_monday_is_32_l1925_192551


namespace eeshas_usual_time_l1925_192537

/-- Eesha's usual time to reach her office from home is 60 minutes,
given that she started 30 minutes late and reached her office
50 minutes late while driving 25% slower than her usual speed. -/
theorem eeshas_usual_time (T T' : ℝ) (h1 : T' = T / 0.75) (h2 : T' = T + 20) : T = 60 := by
  sorry

end eeshas_usual_time_l1925_192537


namespace remaining_amount_after_purchase_l1925_192586

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end remaining_amount_after_purchase_l1925_192586


namespace solve_for_z_l1925_192519

theorem solve_for_z (z : ℂ) (i : ℂ) (h : i^2 = -1) : 3 + 2 * i * z = 5 - 3 * i * z → z = - (2 * i) / 5 :=
by
  intro h_equation
  -- Proof steps will be provided here.
  sorry

end solve_for_z_l1925_192519


namespace value_of_expression_l1925_192553

theorem value_of_expression : (85 + 32 / 113) * 113 = 9635 :=
by
  sorry

end value_of_expression_l1925_192553


namespace problem_l1925_192581

variable (x y : ℝ)

theorem problem
  (h : (3 * x + 1) ^ 2 + |y - 3| = 0) :
  (x + 2 * y) * (x - 2 * y) + (x + 2 * y) ^ 2 - x * (2 * x + 3 * y) = -1 :=
sorry

end problem_l1925_192581


namespace remainder_43_pow_43_plus_43_mod_44_l1925_192540

theorem remainder_43_pow_43_plus_43_mod_44 : (43^43 + 43) % 44 = 42 :=
by 
    sorry

end remainder_43_pow_43_plus_43_mod_44_l1925_192540


namespace angle_triple_of_supplement_l1925_192563

theorem angle_triple_of_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_of_supplement_l1925_192563


namespace sqrt_fraction_equiv_l1925_192516

-- Define the fractions
def frac1 : ℚ := 25 / 36
def frac2 : ℚ := 16 / 9

-- Define the expression under the square root
def sum_frac : ℚ := frac1 + (frac2 * 36 / 36)

-- State the problem
theorem sqrt_fraction_equiv : (Real.sqrt sum_frac) = Real.sqrt 89 / 6 :=
by
  -- Steps and proof are omitted; we use sorry to indicate the proof is skipped
  sorry

end sqrt_fraction_equiv_l1925_192516


namespace smallest_palindrome_in_base3_and_base5_l1925_192578

def is_palindrome_base (b n : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_palindrome_in_base3_and_base5 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome_base 3 n ∧ is_palindrome_base 5 n ∧ n = 20 :=
by
  sorry

end smallest_palindrome_in_base3_and_base5_l1925_192578


namespace find_a2023_l1925_192558

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∃ a1 : ℤ, ∀ n : ℕ, a n = a1 + n * d

theorem find_a2023 (a : ℕ → ℤ) (h_arith : arithmetic_sequence a)
  (h_cond1 : a 2 + a 7 = a 8 + 1)
  (h_cond2 : (a 4)^2 = a 2 * a 8) :
  a 2023 = 2023 := 
sorry

end find_a2023_l1925_192558


namespace actual_length_of_road_l1925_192554

-- Define the conditions
def scale_factor : ℝ := 2500000
def length_on_map : ℝ := 6
def cm_to_km : ℝ := 100000

-- State the theorem
theorem actual_length_of_road : (length_on_map * scale_factor) / cm_to_km = 150 := by
  sorry

end actual_length_of_road_l1925_192554


namespace april_roses_l1925_192579

theorem april_roses (price_per_rose earnings roses_left : ℤ) 
  (h1 : price_per_rose = 4)
  (h2 : earnings = 36)
  (h3 : roses_left = 4) :
  4 + (earnings / price_per_rose) = 13 :=
by
  sorry

end april_roses_l1925_192579


namespace longest_diagonal_length_l1925_192546

-- Defining conditions
variable (d1 d2 : ℝ)
variable (x : ℝ)
variable (area : ℝ)
variable (h_area : area = 144)
variable (h_ratio : d1 = 4 * x)
variable (h_ratio' : d2 = 3 * x)
variable (h_area_eq : area = 1 / 2 * d1 * d2)

-- The Lean statement, asserting the length of the longest diagonal is 8 * sqrt(6)
theorem longest_diagonal_length (x : ℝ) (h_area : 1 / 2 * (4 * x) * (3 * x) = 144) :
  4 * x = 8 * Real.sqrt 6 := by
sorry

end longest_diagonal_length_l1925_192546


namespace cows_eat_husk_l1925_192557

theorem cows_eat_husk :
  ∀ (cows : ℕ) (days : ℕ) (husk_per_cow : ℕ),
    cows = 45 →
    days = 45 →
    husk_per_cow = 1 →
    (cows * husk_per_cow = 45) :=
by
  intros cows days husk_per_cow h_cows h_days h_husk_per_cow
  sorry

end cows_eat_husk_l1925_192557


namespace chloe_total_score_l1925_192547

theorem chloe_total_score :
  let first_level_treasure_points := 9
  let first_level_bonus_points := 15
  let first_level_treasures := 6
  let second_level_treasure_points := 11
  let second_level_bonus_points := 20
  let second_level_treasures := 3

  let first_level_score := first_level_treasures * first_level_treasure_points + first_level_bonus_points
  let second_level_score := second_level_treasures * second_level_treasure_points + second_level_bonus_points

  first_level_score + second_level_score = 122 :=
by
  sorry

end chloe_total_score_l1925_192547


namespace chickens_count_l1925_192559

theorem chickens_count (rabbits frogs : ℕ) (h_rabbits : rabbits = 49) (h_frogs : frogs = 37) :
  ∃ (C : ℕ), frogs + C = rabbits + 9 ∧ C = 21 :=
by
  sorry

end chickens_count_l1925_192559


namespace trapezium_area_l1925_192593

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l1925_192593


namespace wendy_boxes_l1925_192503

theorem wendy_boxes (x : ℕ) (w_brother : ℕ) (total : ℕ) (candy_per_box : ℕ) 
    (h_w_brother : w_brother = 6) 
    (h_candy_per_box : candy_per_box = 3) 
    (h_total : total = 12) 
    (h_equation : 3 * x + w_brother = total) : 
    x = 2 :=
by
  -- Proof would go here
  sorry

end wendy_boxes_l1925_192503


namespace hotdog_eating_ratio_l1925_192574

variable (rate_first rate_second rate_third total_hotdogs time_minutes : ℕ)
variable (rate_ratio : ℕ)

def rate_first_eq : rate_first = 10 := by sorry
def rate_second_eq : rate_second = 3 * rate_first := by sorry
def total_hotdogs_eq : total_hotdogs = 300 := by sorry
def time_minutes_eq : time_minutes = 5 := by sorry
def rate_third_eq : rate_third = total_hotdogs / time_minutes := by sorry

theorem hotdog_eating_ratio :
  rate_ratio = rate_third / rate_second :=
  by sorry

end hotdog_eating_ratio_l1925_192574


namespace accurate_bottle_weight_l1925_192539

-- Define the options as constants
def OptionA : ℕ := 500 -- milligrams
def OptionB : ℕ := 500 * 1000 -- grams
def OptionC : ℕ := 500 * 1000 * 1000 -- kilograms
def OptionD : ℕ := 500 * 1000 * 1000 * 1000 -- tons

-- Define a threshold range for the weight of a standard bottle of mineral water in grams
def typicalBottleWeightMin : ℕ := 400 -- for example
def typicalBottleWeightMax : ℕ := 600 -- for example

-- Translate the question and conditions into a proof statement
theorem accurate_bottle_weight : OptionB = 500 * 1000 :=
by
  -- Normally, we would add the necessary steps here to prove the statement
  sorry

end accurate_bottle_weight_l1925_192539


namespace find_A_l1925_192582

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) 
(h_div9 : (A + 1 + 5 + B + 9 + 4) % 9 = 0) 
(h_div11 : (A + 5 + 9 - (1 + B + 4)) % 11 = 0) : A = 5 :=
by sorry

end find_A_l1925_192582


namespace maximum_value_of_sum_l1925_192596

variables (x y : ℝ)

def s : ℝ := x + y

theorem maximum_value_of_sum (h : s ≤ 9) : s = 9 :=
sorry

end maximum_value_of_sum_l1925_192596


namespace sum_m_n_is_192_l1925_192529

def smallest_prime : ℕ := 2

def largest_four_divisors_under_200 : ℕ :=
  -- we assume this as 190 based on the provided problem's solution
  190

theorem sum_m_n_is_192 :
  smallest_prime = 2 →
  largest_four_divisors_under_200 = 190 →
  smallest_prime + largest_four_divisors_under_200 = 192 :=
by
  intros h1 h2
  sorry

end sum_m_n_is_192_l1925_192529


namespace rhombus_side_length_l1925_192508

-- Definitions
def is_rhombus_perimeter (P s : ℝ) : Prop := P = 4 * s

-- Theorem to prove
theorem rhombus_side_length (P : ℝ) (hP : P = 4) : ∃ s : ℝ, is_rhombus_perimeter P s ∧ s = 1 :=
by
  sorry

end rhombus_side_length_l1925_192508


namespace carlson_fraction_l1925_192588

-- Define variables
variables (n m k p T : ℝ)

theorem carlson_fraction (h1 : k = 0.6 * n)
                         (h2 : p = 2.5 * m)
                         (h3 : T = n * m + k * p) :
                         k * p / T = 3 / 5 := by
  -- Omitted proof
  sorry

end carlson_fraction_l1925_192588


namespace sum_of_solutions_eqn_l1925_192513

theorem sum_of_solutions_eqn : 
  (∀ x : ℝ, -48 * x^2 + 100 * x + 200 = 0 → False) → 
  (-100 / -48) = (25 / 12) :=
by
  intros
  sorry

end sum_of_solutions_eqn_l1925_192513


namespace no_solution_when_k_equals_7_l1925_192542

noncomputable def no_solution_eq (k x : ℝ) : Prop :=
  (x - 3) / (x - 4) = (x - k) / (x - 8)
  
theorem no_solution_when_k_equals_7 :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ¬ no_solution_eq 7 x :=
by
  sorry

end no_solution_when_k_equals_7_l1925_192542


namespace joan_balloons_l1925_192577

def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2
def total_balloons : ℕ := 16

theorem joan_balloons : sally_balloons + jessica_balloons = 7 ∧ total_balloons = 16 → total_balloons - (sally_balloons + jessica_balloons) = 9 :=
by
  sorry

end joan_balloons_l1925_192577


namespace larger_square_side_length_l1925_192599

theorem larger_square_side_length (s1 s2 : ℝ) (h1 : s1 = 5) (h2 : s2 = s1 * 3) (a1 a2 : ℝ) (h3 : a1 = s1^2) (h4 : a2 = s2^2) : s2 = 15 := 
by
  sorry

end larger_square_side_length_l1925_192599


namespace rikki_poetry_sales_l1925_192515

theorem rikki_poetry_sales :
  let words_per_5min := 25
  let total_minutes := 2 * 60
  let intervals := total_minutes / 5
  let total_words := words_per_5min * intervals
  let total_earnings := 6
  let price_per_word := total_earnings / total_words
  price_per_word = 0.01 :=
by
  sorry

end rikki_poetry_sales_l1925_192515


namespace exists_ij_aij_gt_ij_l1925_192525

theorem exists_ij_aij_gt_ij (a : ℕ → ℕ → ℕ) 
  (h_a_positive : ∀ i j, 0 < a i j)
  (h_a_distribution : ∀ k, (∃ S : Finset (ℕ × ℕ), S.card = 8 ∧ ∀ ij : ℕ × ℕ, ij ∈ S ↔ a ij.1 ij.2 = k)) :
  ∃ i j, a i j > i * j :=
by
  sorry

end exists_ij_aij_gt_ij_l1925_192525


namespace jessica_deposited_fraction_l1925_192552

-- Definitions based on conditions
def original_balance (B : ℝ) : Prop :=
  B * (3 / 5) = B - 200

def final_balance (B : ℝ) (F : ℝ) : Prop :=
  ((3 / 5) * B) + (F * ((3 / 5) * B)) = 360

-- Theorem statement proving that the fraction deposited is 1/5
theorem jessica_deposited_fraction (B : ℝ) (F : ℝ) (h1 : original_balance B) (h2 : final_balance B F) : F = 1 / 5 :=
  sorry

end jessica_deposited_fraction_l1925_192552


namespace product_of_decimal_numbers_l1925_192523

theorem product_of_decimal_numbers 
  (h : 213 * 16 = 3408) : 
  1.6 * 21.3 = 34.08 :=
by
  sorry

end product_of_decimal_numbers_l1925_192523


namespace ratio_ab_l1925_192501

theorem ratio_ab (a b : ℚ) (h : b / a = 5 / 13) : (a - b) / (a + b) = 4 / 9 :=
by
  sorry

end ratio_ab_l1925_192501


namespace smart_charging_piles_growth_l1925_192511

noncomputable def a : ℕ := 301
noncomputable def b : ℕ := 500
variable (x : ℝ) -- Monthly average growth rate

theorem smart_charging_piles_growth :
  a * (1 + x) ^ 2 = b :=
by
  -- Proof should go here
  sorry

end smart_charging_piles_growth_l1925_192511


namespace time_relationship_l1925_192595

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end time_relationship_l1925_192595


namespace bouquets_sold_on_Monday_l1925_192521

theorem bouquets_sold_on_Monday
  (tuesday_three_times_monday : ∀ (x : ℕ), bouquets_sold_Tuesday = 3 * x)
  (wednesday_third_of_tuesday : ∀ (bouquets_sold_Tuesday : ℕ), bouquets_sold_Wednesday = bouquets_sold_Tuesday / 3)
  (total_bouquets : bouquets_sold_Monday + bouquets_sold_Tuesday + bouquets_sold_Wednesday = 60)
  : bouquets_sold_Monday = 12 := 
sorry

end bouquets_sold_on_Monday_l1925_192521


namespace range_of_F_l1925_192550

theorem range_of_F (A B C : ℝ) (h1 : 0 < A) (h2 : A ≤ B) (h3 : B ≤ C) (h4 : C < π / 2) :
  1 + (Real.sqrt 2) / 2 < (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) ∧
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 :=
  sorry

end range_of_F_l1925_192550


namespace reciprocal_of_abs_neg_two_l1925_192510

theorem reciprocal_of_abs_neg_two : 1 / |(-2: ℤ)| = (1 / 2: ℚ) := by
  sorry

end reciprocal_of_abs_neg_two_l1925_192510


namespace paws_on_ground_are_correct_l1925_192530

-- Problem statement
def num_paws_on_ground (total_dogs : ℕ) (half_on_all_fours : ℕ) (paws_on_all_fours : ℕ) (half_on_two_legs : ℕ) (paws_on_two_legs : ℕ) : ℕ :=
  half_on_all_fours * paws_on_all_fours + half_on_two_legs * paws_on_two_legs

theorem paws_on_ground_are_correct :
  let total_dogs := 12
  let half_on_all_fours := 6
  let half_on_two_legs := 6
  let paws_on_all_fours := 4
  let paws_on_two_legs := 2
  num_paws_on_ground total_dogs half_on_all_fours paws_on_all_fours half_on_two_legs paws_on_two_legs = 36 :=
by sorry

end paws_on_ground_are_correct_l1925_192530


namespace sandy_correct_sums_l1925_192583

variables (x y : ℕ)

theorem sandy_correct_sums :
  (x + y = 30) →
  (3 * x - 2 * y = 50) →
  x = 22 :=
by
  intro h1 h2
  -- Proof will be filled in here
  sorry

end sandy_correct_sums_l1925_192583


namespace series_sum_proof_l1925_192592

noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, if n % 3 = 0 then 1 / (27 ^ (n / 3)) * (5 / 9) else 0

theorem series_sum_proof : infinite_series_sum = 15 / 26 :=
  sorry

end series_sum_proof_l1925_192592


namespace sum_of_odd_powers_divisible_by_six_l1925_192520

theorem sum_of_odd_powers_divisible_by_six (a1 a2 a3 a4 : ℤ)
    (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) :
    ∀ k : ℕ, k % 2 = 1 → 6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
by
  intros k hk
  sorry

end sum_of_odd_powers_divisible_by_six_l1925_192520


namespace total_pencils_owned_l1925_192505

def SetA_pencils := 10
def SetB_pencils := 20
def SetC_pencils := 30

def friends_SetA_Buys := 3
def friends_SetB_Buys := 2
def friends_SetC_Buys := 2

def Chloe_SetA_Buys := 1
def Chloe_SetB_Buys := 1
def Chloe_SetC_Buys := 1

def total_friends_pencils := friends_SetA_Buys * SetA_pencils + friends_SetB_Buys * SetB_pencils + friends_SetC_Buys * SetC_pencils
def total_Chloe_pencils := Chloe_SetA_Buys * SetA_pencils + Chloe_SetB_Buys * SetB_pencils + Chloe_SetC_Buys * SetC_pencils
def total_pencils := total_friends_pencils + total_Chloe_pencils

theorem total_pencils_owned : total_pencils = 190 :=
by
  sorry

end total_pencils_owned_l1925_192505


namespace first_divisor_is_13_l1925_192561

theorem first_divisor_is_13 (x : ℤ) (h : (377 / x) / 29 * (1/4 : ℚ) / 2 = (1/8 : ℚ)) : x = 13 := by
  sorry

end first_divisor_is_13_l1925_192561


namespace avg_weight_l1925_192591

theorem avg_weight (A B C : ℝ)
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by sorry

end avg_weight_l1925_192591


namespace prove_a_eq_b_l1925_192500

theorem prove_a_eq_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_eq : a^b = b^a) (h_a_lt_1 : a < 1) : a = b :=
by
  sorry

end prove_a_eq_b_l1925_192500


namespace sum_three_numbers_l1925_192598

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_three_numbers_l1925_192598


namespace max_area_quadrilateral_cdfg_l1925_192534

theorem max_area_quadrilateral_cdfg (s : ℝ) (x : ℝ)
  (h1 : s = 1) (h2 : x > 0) (h3 : x < s) (h4 : AE = x) (h5 : AF = x) : 
  ∃ x, x > 0 ∧ x < 1 ∧ (1 - x) * x ≤ 5 / 8 :=
sorry

end max_area_quadrilateral_cdfg_l1925_192534


namespace general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l1925_192562

noncomputable def a_seq (n : ℕ) : ℕ :=
  if h : n > 0 then n else 1

noncomputable def b_seq (n : ℕ) : ℚ :=
  if h : n > 0 then n * (n - 1) / 4 else 0

noncomputable def c_seq (n : ℕ) : ℚ :=
  a_seq n ^ 2 - 4 * b_seq n

theorem general_formula_for_sequences (n : ℕ) (h : n > 0) :
  a_seq n = n ∧ b_seq n = (n * (n - 1)) / 4 :=
sorry

theorem c_seq_is_arithmetic (n : ℕ) (h : n > 0) : 
  ∀ m : ℕ, (h2 : m > 0) -> c_seq (m+1) - c_seq m = 1 :=
sorry

theorem fn_integer_roots (n : ℕ) : 
  ∃ k : ℤ, n = k ^ 2 ∧ k ≠ 0 :=
sorry

end general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l1925_192562


namespace fifth_graders_more_than_seventh_l1925_192538

theorem fifth_graders_more_than_seventh (price_per_pencil : ℕ) (price_per_pencil_pos : price_per_pencil > 0)
    (total_cents_7th : ℕ) (total_cents_7th_val : total_cents_7th = 201)
    (total_cents_5th : ℕ) (total_cents_5th_val : total_cents_5th = 243)
    (pencil_price_div_7th : total_cents_7th % price_per_pencil = 0)
    (pencil_price_div_5th : total_cents_5th % price_per_pencil = 0) :
    (total_cents_5th / price_per_pencil - total_cents_7th / price_per_pencil = 14) := 
by
    sorry

end fifth_graders_more_than_seventh_l1925_192538


namespace jellybeans_initial_amount_l1925_192543

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l1925_192543


namespace inequality_solution_m_range_l1925_192548

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a = 1 → f x + a - 1 > 0 ↔ x ≠ 2) ∧
  (a > 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ True) ∧
  (a < 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ x < a + 1 ∨ x > 3 - a) :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 5 :=
by
  sorry

end inequality_solution_m_range_l1925_192548


namespace evaluate_expression_at_x_eq_2_l1925_192565

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l1925_192565


namespace value_of_a4_l1925_192597

theorem value_of_a4 (a : ℕ → ℕ) (r : ℕ) (h1 : ∀ n, a (n+1) = r * a n) (h2 : a 4 / a 2 - a 3 = 0) (h3 : r = 2) :
  a 4 = 8 :=
sorry

end value_of_a4_l1925_192597


namespace lucas_change_l1925_192526

-- Define the given conditions as constants in Lean
def num_bananas : ℕ := 5
def cost_per_banana : ℝ := 0.70
def num_oranges : ℕ := 2
def cost_per_orange : ℝ := 0.80
def amount_paid : ℝ := 10.00

-- Define a noncomputable constant to represent the change received
noncomputable def change_received : ℝ := 
  amount_paid - (num_bananas * cost_per_banana + num_oranges * cost_per_orange)

-- State the theorem to be proved
theorem lucas_change : change_received = 4.90 := 
by 
  -- Dummy proof since the actual proof is not required
  sorry

end lucas_change_l1925_192526


namespace arithmetic_seq_contains_geometric_seq_l1925_192514

theorem arithmetic_seq_contains_geometric_seq (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (ns : ℕ → ℕ) (k : ℝ), k ≠ 1 ∧ (∀ n, a + b * (ns (n + 1)) = k * (a + b * (ns n)))) ↔ (∃ (q : ℚ), a = q * b) :=
sorry

end arithmetic_seq_contains_geometric_seq_l1925_192514


namespace find_integer_pairs_l1925_192524

theorem find_integer_pairs :
  ∀ (a b : ℕ), 0 < a → 0 < b → a * b + 2 = a^3 + 2 * b →
  (a = 1 ∧ b = 1) ∨ (a = 3 ∧ b = 25) ∨ (a = 4 ∧ b = 31) ∨ (a = 5 ∧ b = 41) ∨ (a = 8 ∧ b = 85) :=
by
  intros a b ha hb hab_eq
  -- Proof goes here
  sorry

end find_integer_pairs_l1925_192524


namespace inequality_x_y_l1925_192555

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end inequality_x_y_l1925_192555


namespace movie_length_l1925_192585

theorem movie_length (paused_midway : ∃ t : ℝ, t = t ∧ t / 2 = 30) : 
  ∃ total_length : ℝ, total_length = 60 :=
by {
  sorry
}

end movie_length_l1925_192585


namespace ryan_flyers_l1925_192587

theorem ryan_flyers (total_flyers : ℕ) (alyssa_flyers : ℕ) (scott_flyers : ℕ) (belinda_percentage : ℚ) (belinda_flyers : ℕ) (ryan_flyers : ℕ)
  (htotal : total_flyers = 200)
  (halyssa : alyssa_flyers = 67)
  (hscott : scott_flyers = 51)
  (hbelinda_percentage : belinda_percentage = 0.20)
  (hbelinda : belinda_flyers = belinda_percentage * total_flyers)
  (hryan : ryan_flyers = total_flyers - (alyssa_flyers + scott_flyers + belinda_flyers)) :
  ryan_flyers = 42 := by
    sorry

end ryan_flyers_l1925_192587


namespace expenditure_fraction_l1925_192531

variable (B : ℝ)
def cost_of_book (x y : ℝ) (B : ℝ) := x = 0.30 * (B - 2 * y)
def cost_of_coffee (x y : ℝ) (B : ℝ) := y = 0.10 * (B - x)

theorem expenditure_fraction (x y : ℝ) (B : ℝ) 
  (hx : cost_of_book x y B) 
  (hy : cost_of_coffee x y B) : 
  (x + y) / B = 31 / 94 :=
sorry

end expenditure_fraction_l1925_192531


namespace sum_of_three_numbers_l1925_192589

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) 
 (h_median : b = 10) 
 (h_mean_least : (a + b + c) / 3 = a + 8)
 (h_mean_greatest : (a + b + c) / 3 = c - 20) : 
 a + b + c = 66 :=
by 
  sorry

end sum_of_three_numbers_l1925_192589


namespace max_sides_of_polygon_in_1950_gon_l1925_192509

theorem max_sides_of_polygon_in_1950_gon (n : ℕ) (h : n = 1950) :
  ∃ (m : ℕ), (m ≤ 1949) ∧ (∀ k, k > m → k ≤ 1949) :=
sorry

end max_sides_of_polygon_in_1950_gon_l1925_192509


namespace tangent_line_with_smallest_slope_l1925_192527

-- Define the given curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the given curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the equation of the tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := 3 * x - y = 11

-- Prove that the equation of the tangent line with the smallest slope on the curve is 3x - y - 11 = 0
theorem tangent_line_with_smallest_slope :
  ∃ x y : ℝ, curve x = y ∧ curve_derivative x = 3 ∧ tangent_line x y :=
by
  sorry

end tangent_line_with_smallest_slope_l1925_192527


namespace total_point_value_of_test_l1925_192506

theorem total_point_value_of_test (total_questions : ℕ) (five_point_questions : ℕ) 
  (ten_point_questions : ℕ) (points_5 : ℕ) (points_10 : ℕ) 
  (h1 : total_questions = 30) (h2 : five_point_questions = 20) 
  (h3 : ten_point_questions = total_questions - five_point_questions) 
  (h4 : points_5 = 5) (h5 : points_10 = 10) : 
  five_point_questions * points_5 + ten_point_questions * points_10 = 200 :=
by
  sorry

end total_point_value_of_test_l1925_192506


namespace percentage_of_girls_l1925_192541

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 400) (h2 : B = 80) :
  (G * 100) / (B + G) = 80 :=
by sorry

end percentage_of_girls_l1925_192541


namespace proof_statements_BCD_l1925_192512

variable (a b : ℝ)

theorem proof_statements_BCD (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) :=
by
  sorry

end proof_statements_BCD_l1925_192512


namespace roof_ratio_l1925_192566

theorem roof_ratio (L W : ℝ) 
  (h1 : L * W = 784) 
  (h2 : L - W = 42) : 
  L / W = 4 := by 
  sorry

end roof_ratio_l1925_192566


namespace power_mean_inequality_l1925_192556

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end power_mean_inequality_l1925_192556


namespace g_at_10_l1925_192590

noncomputable def g (n : ℕ) : ℝ := sorry

axiom g_definition : g 2 = 4
axiom g_recursive : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (3 * g (2 * m) + g (2 * n)) / 4

theorem g_at_10 : g 10 = 64 := sorry

end g_at_10_l1925_192590


namespace b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l1925_192528

variable (a b : ℕ)

-- Conditions
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k
def is_multiple_of_10 (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k

-- Given conditions in the problem
axiom h_a : is_multiple_of_5 a
axiom h_b : is_multiple_of_10 b

-- Statements to be proved
theorem b_is_multiple_of_5 : is_multiple_of_5 b :=
sorry

theorem a_plus_b_is_multiple_of_5 : is_multiple_of_5 (a + b) :=
sorry

end b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l1925_192528


namespace four_digit_3_or_6_l1925_192584

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l1925_192584


namespace find_a2_l1925_192567

def arithmetic_sequence (a : ℕ → ℚ) := 
  (a 1 = 1) ∧ ∀ n, a (n + 2) - a n = 3

theorem find_a2 (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 2 = 5 / 2 := 
by
  -- Conditions
  have a1 : a 1 = 1 := h.1
  have h_diff : ∀ n, a (n + 2) - a n = 3 := h.2
  -- Proof steps can be written here
  sorry

end find_a2_l1925_192567


namespace Alice_wins_no_matter_what_Bob_does_l1925_192533

theorem Alice_wins_no_matter_what_Bob_does (a b c : ℝ) :
  (∀ d : ℝ, (b + d) ^ 2 - 4 * (a + d) * (c + d) ≤ 0) → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intro h
  sorry

end Alice_wins_no_matter_what_Bob_does_l1925_192533


namespace sequence_an_properties_l1925_192576

theorem sequence_an_properties
(S : ℕ → ℝ) (a : ℕ → ℝ)
(h_mean : ∀ n, 2 * a n = S n + 2) :
a 1 = 2 ∧ a 2 = 4 ∧ ∀ n, a n = 2 ^ n :=
by
  sorry

end sequence_an_properties_l1925_192576


namespace ratio_of_packets_to_tent_stakes_l1925_192594

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end ratio_of_packets_to_tent_stakes_l1925_192594


namespace distance_from_center_to_chord_l1925_192560

theorem distance_from_center_to_chord (a b : ℝ) : 
  ∃ d : ℝ, d = (1/4) * |a - b| := 
sorry

end distance_from_center_to_chord_l1925_192560


namespace range_of_a_l1925_192517

def decreasing_range (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ 4 → y ≤ 4 → x < y → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)

theorem range_of_a (a : ℝ) : decreasing_range a ↔ a ≤ -3 := 
  sorry

end range_of_a_l1925_192517


namespace max_cities_visited_l1925_192536

theorem max_cities_visited (n k : ℕ) : ∃ t, t = n - k :=
by
  sorry

end max_cities_visited_l1925_192536


namespace Christine_picked_10_pounds_l1925_192572

-- Variable declarations for the quantities involved
variable (C : ℝ) -- Pounds of strawberries Christine picked
variable (pieStrawberries : ℝ := 3) -- Pounds of strawberries per pie
variable (pies : ℝ := 10) -- Number of pies
variable (totalStrawberries : ℝ := 30) -- Total pounds of strawberries for pies

-- The condition that Rachel picked twice as many strawberries as Christine
variable (R : ℝ := 2 * C)

-- The condition for the total pounds of strawberries picked by Christine and Rachel
axiom strawberries_eq : C + R = totalStrawberries

-- The goal is to prove that Christine picked 10 pounds of strawberries
theorem Christine_picked_10_pounds : C = 10 := by
  sorry

end Christine_picked_10_pounds_l1925_192572
