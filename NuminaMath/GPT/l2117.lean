import Mathlib

namespace shaded_hexagons_are_balanced_l2117_211768

-- Definitions and conditions from the problem
def is_balanced (a b c : ℕ) : Prop :=
  (a = b ∧ b = c) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

def hexagon_grid_balanced (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ),
  (i % 2 = 0 ∧ grid (i, j) = grid (i, j + 1) ∧ grid (i, j + 1) = grid (i + 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i, j + 1) ∧ grid (i, j + 1) ≠ grid (i + 1, j + 1) ∧ grid (i, j) ≠ grid (i + 1, j + 1))
  ∨ (i % 2 ≠ 0 ∧ grid (i, j) = grid (i - 1, j) ∧ grid (i - 1, j) = grid (i - 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i - 1, j) ∧ grid (i - 1, j) ≠ grid (i - 1, j + 1) ∧ grid (i, j) ≠ grid (i - 1, j + 1))

theorem shaded_hexagons_are_balanced (grid : ℕ × ℕ → ℕ) (h_balanced : hexagon_grid_balanced grid) :
  is_balanced (grid (1, 1)) (grid (1, 10)) (grid (10, 10)) :=
sorry

end shaded_hexagons_are_balanced_l2117_211768


namespace negation_of_exists_equiv_forall_neg_l2117_211737

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l2117_211737


namespace general_term_arithmetic_sequence_l2117_211743

theorem general_term_arithmetic_sequence (a_n : ℕ → ℚ) (d : ℚ) (h_seq : ∀ n, a_n n = a_n 0 + n * d)
  (h_geometric : (a_n 2)^2 = a_n 1 * a_n 6)
  (h_condition : 2 * a_n 0 + a_n 1 = 1)
  (h_d_nonzero : d ≠ 0) :
  ∀ n, a_n n = (5/3) - n := 
by
  sorry

end general_term_arithmetic_sequence_l2117_211743


namespace minji_combinations_l2117_211788

theorem minji_combinations : (3 * 5) = 15 :=
by sorry

end minji_combinations_l2117_211788


namespace hannah_remaining_money_l2117_211701

-- Define the conditions of the problem
def initial_amount : Nat := 120
def rides_cost : Nat := initial_amount * 40 / 100
def games_cost : Nat := initial_amount * 15 / 100
def remaining_after_rides_games : Nat := initial_amount - rides_cost - games_cost

def dessert_cost : Nat := 8
def cotton_candy_cost : Nat := 5
def hotdog_cost : Nat := 6
def keychain_cost : Nat := 7
def poster_cost : Nat := 10
def additional_attraction_cost : Nat := 15
def total_food_souvenirs_cost : Nat := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + additional_attraction_cost

def final_remaining_amount : Nat := remaining_after_rides_games - total_food_souvenirs_cost

-- Formulate the theorem to prove
theorem hannah_remaining_money : final_remaining_amount = 3 := by
  sorry

end hannah_remaining_money_l2117_211701


namespace sum_of_coefficients_l2117_211727

theorem sum_of_coefficients (a : ℕ → ℝ) :
  (∀ x : ℝ, (2 - x) ^ 10 = a 0 + a 1 * x + a 2 * x ^ 2 + a 3 * x ^ 3 + a 4 * x ^ 4 + a 5 * x ^ 5 + a 6 * x ^ 6 + a 7 * x ^ 7 + a 8 * x ^ 8 + a 9 * x ^ 9 + a 10 * x ^ 10) →
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 1 →
  a 0 = 1024 →
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -1023 :=  
by
  intro h1 h2 h3
  sorry

end sum_of_coefficients_l2117_211727


namespace square_b_perimeter_l2117_211725

/-- Square A has an area of 121 square centimeters. Square B has a certain perimeter.
  If square B is placed within square A and a random point is chosen within square A,
  the probability that the point is not within square B is 0.8677685950413223.
  Prove the perimeter of square B is 16 centimeters. -/
theorem square_b_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) 
  (h1 : area_A = 121)
  (h2 : prob = 0.8677685950413223)
  (h3 : ∃ (a b : ℝ), area_A = a * a ∧ a * a - b * b = prob * area_A) :
  perimeter_B = 16 :=
sorry

end square_b_perimeter_l2117_211725


namespace solve_for_a_l2117_211750

theorem solve_for_a (a : ℝ) (h : |2 * a + 1| = 3 * |a| - 2) : a = -1 ∨ a = 3 :=
by
  sorry

end solve_for_a_l2117_211750


namespace length_of_RT_in_trapezoid_l2117_211785

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end length_of_RT_in_trapezoid_l2117_211785


namespace solve_inequality_l2117_211731

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∨ a = 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ False)) ∧
  (0 < a ∧ a < 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ a^2 < x ∧ x < a)) ∧
  (a < 0 ∨ a > 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ a < x ∧ x < a^2)) :=
  by
    sorry

end solve_inequality_l2117_211731


namespace Emily_candies_l2117_211720

theorem Emily_candies (jennifer_candies emily_candies bob_candies : ℕ) 
    (h1: jennifer_candies = 2 * emily_candies)
    (h2: jennifer_candies = 3 * bob_candies)
    (h3: bob_candies = 4) : emily_candies = 6 :=
by
  -- Proof to be provided
  sorry

end Emily_candies_l2117_211720


namespace sum_of_squares_l2117_211755

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 5) (h2 : ab + bc + ac = 5) : a^2 + b^2 + c^2 = 15 :=
by sorry

end sum_of_squares_l2117_211755


namespace supermarkets_in_us_l2117_211744

noncomputable def number_of_supermarkets_in_canada : ℕ := 35
noncomputable def number_of_supermarkets_total : ℕ := 84
noncomputable def diff_us_canada : ℕ := 14
noncomputable def number_of_supermarkets_in_us : ℕ := number_of_supermarkets_in_canada + diff_us_canada

theorem supermarkets_in_us : number_of_supermarkets_in_us = 49 := by
  sorry

end supermarkets_in_us_l2117_211744


namespace increasing_function_on_interval_l2117_211782

noncomputable def f_A (x : ℝ) : ℝ := 3 - x
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3 * x
noncomputable def f_C (x : ℝ) : ℝ := - (1 / (x + 1))
noncomputable def f_D (x : ℝ) : ℝ := -|x|

theorem increasing_function_on_interval (h0 : ∀ x : ℝ, x > 0):
  (∀ x y : ℝ, 0 < x -> x < y -> f_C x < f_C y) ∧ 
  (∀ (g : ℝ → ℝ), (g ≠ f_C) → (∀ x y : ℝ, 0 < x -> x < y -> g x ≥ g y)) :=
by sorry

end increasing_function_on_interval_l2117_211782


namespace max_two_digit_times_max_one_digit_is_three_digit_l2117_211706

def max_two_digit : ℕ := 99
def max_one_digit : ℕ := 9
def product := max_two_digit * max_one_digit

theorem max_two_digit_times_max_one_digit_is_three_digit :
  100 ≤ product ∧ product < 1000 :=
by
  -- Prove that the product is a three-digit number
  sorry

end max_two_digit_times_max_one_digit_is_three_digit_l2117_211706


namespace min_fraction_in_domain_l2117_211779

theorem min_fraction_in_domain :
  ∃ x y : ℝ, (1/4 ≤ x ∧ x ≤ 2/3) ∧ (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ x' y' : ℝ, (1/4 ≤ x' ∧ x' ≤ 2/3) ∧ (1/5 ≤ y' ∧ y' ≤ 1/2) → 
      (xy / (x^2 + y^2) ≤ x'y' / (x'^2 + y'^2))) ∧ 
      xy / (x^2 + y^2) = 2/5 :=
sorry

end min_fraction_in_domain_l2117_211779


namespace fraction_of_milk_in_cup1_l2117_211791

def initial_tea_cup1 : ℚ := 6
def initial_milk_cup2 : ℚ := 6

def tea_transferred_step2 : ℚ := initial_tea_cup1 / 3
def tea_cup1_after_step2 : ℚ := initial_tea_cup1 - tea_transferred_step2
def total_cup2_after_step2 : ℚ := initial_milk_cup2 + tea_transferred_step2

def mixture_transfer_step3 : ℚ := total_cup2_after_step2 / 2
def tea_ratio_cup2 : ℚ := tea_transferred_step2 / total_cup2_after_step2
def milk_ratio_cup2 : ℚ := initial_milk_cup2 / total_cup2_after_step2
def tea_transferred_step3 : ℚ := mixture_transfer_step3 * tea_ratio_cup2
def milk_transferred_step3 : ℚ := mixture_transfer_step3 * milk_ratio_cup2

def tea_cup1_after_step3 : ℚ := tea_cup1_after_step2 + tea_transferred_step3
def milk_cup1_after_step3 : ℚ := milk_transferred_step3

def mixture_transfer_step4 : ℚ := (tea_cup1_after_step3 + milk_cup1_after_step3) / 4
def tea_ratio_cup1_step4 : ℚ := tea_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)
def milk_ratio_cup1_step4 : ℚ := milk_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)

def tea_transferred_step4 : ℚ := mixture_transfer_step4 * tea_ratio_cup1_step4
def milk_transferred_step4 : ℚ := mixture_transfer_step4 * milk_ratio_cup1_step4

def final_tea_cup1 : ℚ := tea_cup1_after_step3 - tea_transferred_step4
def final_milk_cup1 : ℚ := milk_cup1_after_step3 - milk_transferred_step4
def final_total_liquid_cup1 : ℚ := final_tea_cup1 + final_milk_cup1

theorem fraction_of_milk_in_cup1 : final_milk_cup1 / final_total_liquid_cup1 = 3/8 := by
  sorry

end fraction_of_milk_in_cup1_l2117_211791


namespace distinct_solution_condition_l2117_211714

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l2117_211714


namespace alyssa_earnings_l2117_211763

theorem alyssa_earnings
    (weekly_allowance: ℤ)
    (spent_on_movies_fraction: ℤ)
    (amount_ended_with: ℤ)
    (h1: weekly_allowance = 8)
    (h2: spent_on_movies_fraction = 1 / 2)
    (h3: amount_ended_with = 12)
    : ∃ money_earned_from_car_wash: ℤ, money_earned_from_car_wash = 8 :=
by
  sorry

end alyssa_earnings_l2117_211763


namespace solution_set_l2117_211724

theorem solution_set {x : ℝ} :
  abs ((7 - x) / 4) < 3 ∧ 0 ≤ x ↔ 0 ≤ x ∧ x < 19 :=
by
  sorry

end solution_set_l2117_211724


namespace saltwater_concentration_l2117_211747

theorem saltwater_concentration (salt_mass water_mass : ℝ) (h₁ : salt_mass = 8) (h₂ : water_mass = 32) : 
  salt_mass / (salt_mass + water_mass) * 100 = 20 := 
by
  sorry

end saltwater_concentration_l2117_211747


namespace remainder_when_dividing_928927_by_6_l2117_211729

theorem remainder_when_dividing_928927_by_6 :
  928927 % 6 = 1 :=
by
  sorry

end remainder_when_dividing_928927_by_6_l2117_211729


namespace find_T_l2117_211758

variables (h K T : ℝ)
variables (h_val : 4 * h * 7 + 2 = 58)
variables (K_val : K = 9)

theorem find_T : T = 74 :=
by
  sorry

end find_T_l2117_211758


namespace hyeongjun_older_sister_age_l2117_211757

-- Define the ages of Hyeongjun and his older sister
variables (H S : ℕ)

-- Conditions
def age_gap := S = H + 2
def sum_of_ages := H + S = 26

-- Theorem stating that the older sister's age is 14
theorem hyeongjun_older_sister_age (H S : ℕ) (h1 : age_gap H S) (h2 : sum_of_ages H S) : S = 14 := 
by 
  sorry

end hyeongjun_older_sister_age_l2117_211757


namespace inequality_abc_l2117_211792

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / (b ^ (1/2 : ℝ)) + b / (a ^ (1/2 : ℝ)) ≥ a ^ (1/2 : ℝ) + b ^ (1/2 : ℝ) :=
by { sorry }

end inequality_abc_l2117_211792


namespace krishan_money_l2117_211715

variable {R G K : ℕ}

theorem krishan_money 
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (hR : R = 588)
  : K = 3468 :=
by
  sorry

end krishan_money_l2117_211715


namespace interest_rate_proof_l2117_211767

-- Define the given values
def P : ℝ := 1500
def t : ℝ := 2.4
def A : ℝ := 1680

-- Define the interest rate per annum to be proven
def r : ℝ := 0.05

-- Prove that the calculated interest rate matches the given interest rate per annum
theorem interest_rate_proof 
  (principal : ℝ := P) 
  (time_period : ℝ := t) 
  (amount : ℝ := A) 
  (interest_rate : ℝ := r) :
  (interest_rate = ((amount / principal - 1) / time_period)) :=
by
  sorry

end interest_rate_proof_l2117_211767


namespace cos_double_angle_identity_l2117_211751

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 := 
sorry

end cos_double_angle_identity_l2117_211751


namespace intersection_equals_l2117_211702

def A : Set ℝ := {x | x < 1}

def B : Set ℝ := {x | x^2 + x ≤ 6}

theorem intersection_equals : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equals_l2117_211702


namespace discriminant_of_trinomial_l2117_211789

theorem discriminant_of_trinomial (x1 x2 : ℝ) (h : x2 - x1 = 2) : (x2 - x1)^2 = 4 :=
by
  sorry

end discriminant_of_trinomial_l2117_211789


namespace behavior_of_g_l2117_211712

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 4 * x ^ 2 + 5

theorem behavior_of_g :
  (∀ x, (∃ M, x ≥ M → g x < 0)) ∧ (∀ x, (∃ N, x ≤ N → g x > 0)) :=
by
  sorry

end behavior_of_g_l2117_211712


namespace find_a_l2117_211713

theorem find_a (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_a_l2117_211713


namespace candy_bag_division_l2117_211734

theorem candy_bag_division (total_candy bags_candy : ℕ) (h1 : total_candy = 42) (h2 : bags_candy = 21) : 
  total_candy / bags_candy = 2 := 
by
  sorry

end candy_bag_division_l2117_211734


namespace pieces_missing_l2117_211769

def total_pieces : ℕ := 32
def pieces_present : ℕ := 24

theorem pieces_missing : total_pieces - pieces_present = 8 := by
sorry

end pieces_missing_l2117_211769


namespace evaluate_f_at_7_l2117_211707

theorem evaluate_f_at_7 :
  (∃ f : ℕ → ℕ, (∀ x, f (2 * x + 1) = x ^ 2 - 2 * x) ∧ f 7 = 3) :=
by 
  sorry

end evaluate_f_at_7_l2117_211707


namespace intersection_points_in_plane_l2117_211733

-- Define the cones with parallel axes and equal angles
def cone1 (a1 b1 c1 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a1)^2 + (y - b1)^2 = k^2 * (z - c1)^2

def cone2 (a2 b2 c2 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a2)^2 + (y - b2)^2 = k^2 * (z - c2)^2

-- Given conditions
variable (a1 b1 c1 a2 b2 c2 k : ℝ)

-- The theorem to be proven
theorem intersection_points_in_plane (x y z : ℝ) 
  (h1 : cone1 a1 b1 c1 k x y z) (h2 : cone2 a2 b2 c2 k x y z) : 
  ∃ (A B C D : ℝ), A * x + B * y + C * z + D = 0 :=
by
  sorry

end intersection_points_in_plane_l2117_211733


namespace problem_intersection_l2117_211748

noncomputable def A (x : ℝ) : Prop := 1 < x ∧ x < 4
noncomputable def B (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem problem_intersection : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end problem_intersection_l2117_211748


namespace inequality_constant_l2117_211773

noncomputable def smallest_possible_real_constant : ℝ :=
  1.0625

theorem inequality_constant (C : ℝ) : 
  (∀ x y z : ℝ, (x + y + z = -1) → 
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1| ) ↔ C ≥ smallest_possible_real_constant :=
sorry

end inequality_constant_l2117_211773


namespace domain_of_sqrt_ln_eq_l2117_211780

noncomputable def domain_of_function : Set ℝ :=
  {x | 2 * x + 1 >= 0 ∧ 3 - 4 * x > 0}

theorem domain_of_sqrt_ln_eq :
  domain_of_function = Set.Icc (-1 / 2) (3 / 4) \ {3 / 4} :=
by
  sorry

end domain_of_sqrt_ln_eq_l2117_211780


namespace fraction_of_friends_l2117_211740

variable (x y : ℕ) -- number of first-grade students and sixth-grade students

-- Conditions from the problem
def condition1 : Prop := ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a * x = b * y ∧ 1 / 3 = a / (a + b)
def condition2 : Prop := ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c * y = d * x ∧ 2 / 5 = c / (c + d)

-- Theorem statement to prove that the fraction of students who are friends is 4/11
theorem fraction_of_friends (h1 : condition1 x y) (h2 : condition2 x y) :
  (1 / 3 : ℚ) * y + (2 / 5 : ℚ) * x / (x + y) = 4 / 11 :=
sorry

end fraction_of_friends_l2117_211740


namespace area_of_triangle_F1PF2P_l2117_211765

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4
noncomputable def c : ℝ := 3
noncomputable def PF1 : ℝ := sorry 
noncomputable def PF2 : ℝ := sorry

-- Given conditions
def ellipse_eq_holds (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Given point P is on the ellipse
def P_on_ellipse (x y : ℝ) : Prop := ellipse_eq_holds x y

-- Given angle F1PF2
def angle_F1PF2_eq_60 : Prop := sorry

-- Proving the area of △F₁PF₂
theorem area_of_triangle_F1PF2P : S = (16 * Real.sqrt 3) / 3 :=
by sorry

end area_of_triangle_F1PF2P_l2117_211765


namespace abc_sum_l2117_211711

theorem abc_sum (f : ℝ → ℝ) (a b c : ℝ) :
  f (x - 2) = 2 * x^2 - 5 * x + 3 → f x = a * x^2 + b * x + c → a + b + c = 6 :=
by
  intros h₁ h₂
  sorry

end abc_sum_l2117_211711


namespace condition_C_for_D_condition_A_for_B_l2117_211771

theorem condition_C_for_D (C D : Prop) (h : C → D) : C → D :=
by
  exact h

theorem condition_A_for_B (A B D : Prop) (hA_to_D : A → D) (hD_to_B : D → B) : A → B :=
by
  intro hA
  apply hD_to_B
  apply hA_to_D
  exact hA

end condition_C_for_D_condition_A_for_B_l2117_211771


namespace part1_part2_l2117_211710

-- Part (1)
theorem part1 (x : ℝ) (m : ℝ) (h : x = 2) : 
  (x / (x - 3) + m / (3 - x) = 3) → m = 5 :=
sorry

-- Part (2)
theorem part2 (x : ℝ) (m : ℝ) :
  (x / (x - 3) + m / (3 - x) = 3) → (x > 0) → (m < 9) ∧ (m ≠ 3) :=
sorry

end part1_part2_l2117_211710


namespace sum_of_final_numbers_l2117_211784

variable {x y T : ℝ}

theorem sum_of_final_numbers (h : x + y = T) : 3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by 
  -- The place for the proof steps, which will later be filled
  sorry

end sum_of_final_numbers_l2117_211784


namespace pieces_of_candy_l2117_211774

def total_items : ℝ := 3554
def secret_eggs : ℝ := 145.0

theorem pieces_of_candy : (total_items - secret_eggs) = 3409 :=
by 
  sorry

end pieces_of_candy_l2117_211774


namespace neg_prop_l2117_211772

theorem neg_prop : ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 4) ↔ ∃ x : ℝ, x^2 - 2 * x + 4 > 4 := 
by 
  sorry

end neg_prop_l2117_211772


namespace inequality_must_hold_l2117_211796

theorem inequality_must_hold (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_must_hold_l2117_211796


namespace find_m_l2117_211794

def A (m : ℝ) : Set ℝ := {x | x^2 - m * x + m^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {2, -4}

theorem find_m (m : ℝ) : (A m ∩ B).Nonempty ∧ (A m ∩ C) = ∅ → m = -2 := by
  sorry

end find_m_l2117_211794


namespace triple_composition_f_3_l2117_211700

def f (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_f_3 : f (f (f 3)) = 107 :=
by
  sorry

end triple_composition_f_3_l2117_211700


namespace Alice_and_Dave_weight_l2117_211781

variable (a b c d : ℕ)

-- Conditions
variable (h1 : a + b = 230)
variable (h2 : b + c = 220)
variable (h3 : c + d = 250)

-- Proof statement
theorem Alice_and_Dave_weight :
  a + d = 260 :=
sorry

end Alice_and_Dave_weight_l2117_211781


namespace eccentricity_of_ellipse_l2117_211764

theorem eccentricity_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b > 0) :
  (∀ x y : ℝ, (y = -2 * x + 1 → ∃ x₁ y₁ x₂ y₂ : ℝ, (y₁ = -2 * x₁ + 1 ∧ y₂ = -2 * x₂ + 1) ∧ 
    (x₁ / a * x₁ / a + y₁ / b * y₁ / b = 1) ∧ (x₂ / a * x₂ / a + y₂ / b * y₂ / b = 1) ∧ 
    ((x₁ + x₂) / 2 = 4 * (y₁ + y₂) / 2)) → (x / a)^2 + (y / b)^2 = 1) →
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = (Real.sqrt 2) / 2 :=
sorry

end eccentricity_of_ellipse_l2117_211764


namespace p_x_range_l2117_211754

variable (x : ℝ)

def inequality_condition := x^2 - 5*x + 6 < 0
def polynomial_function := x^2 + 5*x + 6

theorem p_x_range (x_ineq : inequality_condition x) : 
  20 < polynomial_function x ∧ polynomial_function x < 30 :=
sorry

end p_x_range_l2117_211754


namespace find_a2015_l2117_211718

def seq (a : ℕ → ℕ) :=
  (a 1 = 1) ∧
  (a 2 = 4) ∧
  (a 3 = 9) ∧
  (∀ n, 4 ≤ n → a n = a (n-1) + a (n-2) - a (n-3))

theorem find_a2015 (a : ℕ → ℕ) (h_seq : seq a) : a 2015 = 8057 :=
sorry

end find_a2015_l2117_211718


namespace range_of_c_l2117_211777

-- Definitions of p and q based on conditions
def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (c > 1 / 2)

-- The theorem states the required condition on c
theorem range_of_c (c : ℝ) (h : c > 0) :
  ¬(p c ∧ q c) ∧ (p c ∨ q c) ↔ (0 < c ∧ c ≤ 1 / 2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l2117_211777


namespace cookies_in_jar_l2117_211752

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l2117_211752


namespace problem_incorrect_statement_D_l2117_211775

theorem problem_incorrect_statement_D :
  (∀ x y, x = -y → x + y = 0) ∧
  (∃ x : ℕ, x^2 + 2 * x = 0) ∧
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (¬ (∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ (x + y > 2))) :=
by sorry

end problem_incorrect_statement_D_l2117_211775


namespace arcsin_arccos_add_eq_pi6_l2117_211759

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def arccos (x : Real) : Real := sorry

theorem arcsin_arccos_add_eq_pi6 (x : Real) (hx_range : -1 ≤ x ∧ x ≤ 1)
    (h3x_range : -1 ≤ 3 * x ∧ 3 * x ≤ 1) 
    (h : arcsin x + arccos (3 * x) = Real.pi / 6) :
    x = Real.sqrt (3 / 124) := 
  sorry

end arcsin_arccos_add_eq_pi6_l2117_211759


namespace percentage_men_science_majors_l2117_211798

theorem percentage_men_science_majors (total_students : ℕ) (women_science_majors_ratio : ℚ) (nonscience_majors_ratio : ℚ) (men_class_ratio : ℚ) :
  women_science_majors_ratio = 0.2 → 
  nonscience_majors_ratio = 0.6 → 
  men_class_ratio = 0.4 → 
  ∃ men_science_majors_percent : ℚ, men_science_majors_percent = 0.7 :=
by
  intros h_women_science_majors h_nonscience_majors h_men_class
  sorry

end percentage_men_science_majors_l2117_211798


namespace linear_term_zero_implies_sum_zero_l2117_211709

-- Define the condition that the product does not have a linear term
def no_linear_term (x a b : ℝ) : Prop :=
  (x + a) * (x + b) = x^2 + (a + b) * x + a * b

-- Given the condition, we need to prove that a + b = 0
theorem linear_term_zero_implies_sum_zero {a b : ℝ} (h : ∀ x : ℝ, no_linear_term x a b) : a + b = 0 :=
by 
  sorry

end linear_term_zero_implies_sum_zero_l2117_211709


namespace taxi_ride_cost_l2117_211704

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end taxi_ride_cost_l2117_211704


namespace tangent_same_at_origin_l2117_211745

noncomputable def f (x : ℝ) := Real.exp (3 * x) - 1
noncomputable def g (x : ℝ) := 3 * Real.exp x - 3

theorem tangent_same_at_origin :
  (deriv f 0 = deriv g 0) ∧ (f 0 = g 0) :=
by
  sorry

end tangent_same_at_origin_l2117_211745


namespace product_mod_10_l2117_211719

theorem product_mod_10 (a b c : ℕ) (ha : a % 10 = 4) (hb : b % 10 = 5) (hc : c % 10 = 5) :
  (a * b * c) % 10 = 0 :=
sorry

end product_mod_10_l2117_211719


namespace number_of_students_at_table_l2117_211717

theorem number_of_students_at_table :
  ∃ (n : ℕ), n ∣ 119 ∧ (n = 7 ∨ n = 17) :=
sorry

end number_of_students_at_table_l2117_211717


namespace square_of_1008_l2117_211735

theorem square_of_1008 : 1008^2 = 1016064 := 
by sorry

end square_of_1008_l2117_211735


namespace asymptote_slope_of_hyperbola_l2117_211746

theorem asymptote_slope_of_hyperbola :
  ∀ (x y : ℝ), (x ≠ 0) ∧ (y/x = 3/4 ∨ y/x = -3/4) ↔ (x^2 / 144 - y^2 / 81 = 1) := 
by
  sorry

end asymptote_slope_of_hyperbola_l2117_211746


namespace sequence_all_perfect_squares_l2117_211753

theorem sequence_all_perfect_squares (n : ℕ) : 
  ∃ k : ℕ, (∃ m : ℕ, 2 * 10^n + 1 = 3 * m) ∧ (x_n = (m^2 / 9)) :=
by
  sorry

end sequence_all_perfect_squares_l2117_211753


namespace inequality_solution_l2117_211730

theorem inequality_solution (y : ℝ) : 
  (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ (7 ≤ y ∧ y ≤ 11 ∨ -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end inequality_solution_l2117_211730


namespace circle_equation_l2117_211721

theorem circle_equation (x y : ℝ) : (x^2 = 16 * y) → (y = 4) → (x, -4) = (x, 4) → x^2 + (y-4)^2 = 64 :=
by
  sorry

end circle_equation_l2117_211721


namespace geometric_sequence_a2_l2117_211786

theorem geometric_sequence_a2 (a1 a2 a3 : ℝ) (h1 : 1 * (1/a1) = a1)
  (h2 : a1 * (1/a2) = a2) (h3 : a2 * (1/a3) = a3) (h4 : a3 * (1/4) = 4)
  (h5 : a2 > 0) : a2 = 2 := sorry

end geometric_sequence_a2_l2117_211786


namespace max_val_a_l2117_211728

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3 * x + 2)

theorem max_val_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 1, f a x ≥ 0) : a ≤ 1 := sorry

end max_val_a_l2117_211728


namespace maria_spent_60_dollars_l2117_211749

theorem maria_spent_60_dollars :
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  true
    → total_cost = 60 := 
by 
  intros
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  sorry

end maria_spent_60_dollars_l2117_211749


namespace math_problem_proof_l2117_211799

theorem math_problem_proof (n : ℕ) 
  (h1 : n / 37 = 2) 
  (h2 : n % 37 = 26) :
  48 - n / 4 = 23 := by
  sorry

end math_problem_proof_l2117_211799


namespace driver_net_hourly_rate_l2117_211723

theorem driver_net_hourly_rate
  (hours : ℝ) (speed : ℝ) (efficiency : ℝ) (cost_per_gallon : ℝ) (compensation_rate : ℝ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : efficiency = 25)
  (h4 : cost_per_gallon = 2.50)
  (h5 : compensation_rate = 0.60)
  :
  ((compensation_rate * (speed * hours) - (cost_per_gallon * (speed * hours / efficiency))) / hours) = 25 :=
sorry

end driver_net_hourly_rate_l2117_211723


namespace seashells_count_l2117_211722

theorem seashells_count {s : ℕ} (h : s + 6 = 25) : s = 19 :=
by
  sorry

end seashells_count_l2117_211722


namespace sample_systematic_draw_first_group_l2117_211787

theorem sample_systematic_draw_first_group :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 8 →
  (x + 15 * 8 = 126) →
  x = 6 :=
by
  intros x h1 h2
  sorry

end sample_systematic_draw_first_group_l2117_211787


namespace fraction_transformed_l2117_211739

variables (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab_pos : a * b > 0)

noncomputable def frac_orig := (a + 2 * b) / (2 * a * b)
noncomputable def frac_new := (3 * a + 2 * 3 * b) / (2 * 3 * a * 3 * b)

theorem fraction_transformed :
  frac_new a b = (1 / 3) * frac_orig a b :=
sorry

end fraction_transformed_l2117_211739


namespace problem_intersection_l2117_211778

open Set

variable {x : ℝ}

def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def C : Set ℝ := {x | (5 / 2) ≤ x ∧ x < 3}

theorem problem_intersection : A ∩ B = C := by
  sorry

end problem_intersection_l2117_211778


namespace stewarts_theorem_l2117_211756

theorem stewarts_theorem
  (A B C D : ℝ)
  (AB AC AD : ℝ)
  (BD CD BC : ℝ)
  (hD_on_BC : BD + CD = BC) :
  AB^2 * CD + AC^2 * BD - AD^2 * BC = BD * CD * BC := 
sorry

end stewarts_theorem_l2117_211756


namespace systematic_sampling_second_group_l2117_211776

theorem systematic_sampling_second_group
    (N : ℕ) (n : ℕ) (k : ℕ := N / n)
    (number_from_16th_group : ℕ)
    (number_from_1st_group : ℕ := number_from_16th_group - 15 * k)
    (number_from_2nd_group : ℕ := number_from_1st_group + k) :
    N = 160 → n = 20 → number_from_16th_group = 123 → number_from_2nd_group = 11 :=
by
  sorry

end systematic_sampling_second_group_l2117_211776


namespace random_event_proof_l2117_211703

def statement_A := "Strong youth leads to a strong country"
def statement_B := "Scooping the moon in the water"
def statement_C := "Waiting by the stump for a hare"
def statement_D := "Green waters and lush mountains are mountains of gold and silver"

def is_random_event (statement : String) : Prop :=
statement = statement_C

theorem random_event_proof : is_random_event statement_C :=
by
  -- Based on the analysis in the problem, Statement C is determined to be random.
  sorry

end random_event_proof_l2117_211703


namespace remainder_when_7n_divided_by_5_l2117_211726

theorem remainder_when_7n_divided_by_5 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
  sorry

end remainder_when_7n_divided_by_5_l2117_211726


namespace minimum_value_of_f_l2117_211736

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem minimum_value_of_f : ∃ y, (∀ x, f x ≥ y) ∧ y = 3 := 
by
  sorry

end minimum_value_of_f_l2117_211736


namespace sin_cos_values_trigonometric_expression_value_l2117_211708

-- Define the conditions
variables (α : ℝ)
def point_on_terminal_side (x y : ℝ) (r : ℝ) : Prop :=
  (x = 3) ∧ (y = 4) ∧ (r = 5)

-- Define the problem statements
theorem sin_cos_values (x y r : ℝ) (h: point_on_terminal_side x y r) : 
  (Real.sin α = 4 / 5) ∧ (Real.cos α = 3 / 5) :=
sorry

theorem trigonometric_expression_value (h1: Real.sin α = 4 / 5) (h2: Real.cos α = 3 / 5) :
  (2 * Real.cos (π / 2 - α) - Real.cos (π + α)) / (2 * Real.sin (π - α)) = 11 / 8 :=
sorry

end sin_cos_values_trigonometric_expression_value_l2117_211708


namespace problem1_problem2_l2117_211766

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : 2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3 :=
by sorry

end problem1_problem2_l2117_211766


namespace polygon_sides_from_angle_sum_l2117_211795

-- Let's define the problem
theorem polygon_sides_from_angle_sum : 
  ∀ (n : ℕ), (n - 2) * 180 = 900 → n = 7 :=
by
  intros n h
  sorry

end polygon_sides_from_angle_sum_l2117_211795


namespace triangle_area_l2117_211738

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Define the property of being a right triangle via the Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of a right triangle given base and height
def area_right_triangle (a b : ℕ) : ℕ := (a * b) / 2

-- The main theorem, stating that the area of the triangle with sides 9, 12, 15 is 54
theorem triangle_area : is_right_triangle a b c → area_right_triangle a b = 54 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l2117_211738


namespace find_ab_for_equation_l2117_211732

theorem find_ab_for_equation (a b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (∃ x, x = 12 - x1 - x2) ∧ (a * x1^2 - 24 * x1 + b) / (x1^2 - 1) = x1
  ∧ (a * x2^2 - 24 * x2 + b) / (x2^2 - 1) = x2) ∧ (a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819) := sorry

end find_ab_for_equation_l2117_211732


namespace inequality_sin_values_l2117_211716

theorem inequality_sin_values :
  let a := Real.sin (-5)
  let b := Real.sin 3
  let c := Real.sin 5
  a > b ∧ b > c :=
by
  sorry

end inequality_sin_values_l2117_211716


namespace hypotenuse_length_l2117_211741

theorem hypotenuse_length (a b c : ℝ) 
  (h_right_angled : c^2 = a^2 + b^2) 
  (h_sum_squares : a^2 + b^2 + c^2 = 980) : 
  c = 70 :=
by
  sorry

end hypotenuse_length_l2117_211741


namespace calculate_product_l2117_211797

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l2117_211797


namespace paintings_left_correct_l2117_211742

def initial_paintings := 98
def paintings_gotten_rid_of := 3

theorem paintings_left_correct :
  initial_paintings - paintings_gotten_rid_of = 95 :=
by
  sorry

end paintings_left_correct_l2117_211742


namespace nails_needed_for_house_wall_l2117_211770

theorem nails_needed_for_house_wall :
  let large_planks : Nat := 13
  let nails_per_large_plank : Nat := 17
  let additional_nails : Nat := 8
  large_planks * nails_per_large_plank + additional_nails = 229 := by
  sorry

end nails_needed_for_house_wall_l2117_211770


namespace find_a_b_and_water_usage_l2117_211760

noncomputable def water_usage_april (a : ℝ) :=
  (15 * (a + 0.8) = 45)

noncomputable def water_usage_may (a b : ℝ) :=
  (17 * (a + 0.8) + 8 * (b + 0.8) = 91)

noncomputable def water_usage_june (a b x : ℝ) :=
  (17 * (a + 0.8) + 13 * (b + 0.8) + (x - 30) * 6.8 = 150)

theorem find_a_b_and_water_usage :
  ∃ (a b x : ℝ), water_usage_april a ∧ water_usage_may a b ∧ water_usage_june a b x ∧ a = 2.2 ∧ b = 4.2 ∧ x = 35 :=
by {
  sorry
}

end find_a_b_and_water_usage_l2117_211760


namespace circle_x_intercept_l2117_211705

theorem circle_x_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3) (k1 : y1 = 2) (h2 : x2 = 11) (k2 : y2 = 8) :
  ∃ x : ℝ, (x ≠ 3) ∧ ((x - 7) ^ 2 + (0 - 5) ^ 2 = 25) ∧ (x = 7) :=
by
  sorry

end circle_x_intercept_l2117_211705


namespace smallest_number_of_beads_l2117_211793

theorem smallest_number_of_beads (M : ℕ) (h1 : ∃ d : ℕ, M = 5 * d + 2) (h2 : ∃ e : ℕ, M = 7 * e + 2) (h3 : ∃ f : ℕ, M = 9 * f + 2) (h4 : M > 1) : M = 317 := sorry

end smallest_number_of_beads_l2117_211793


namespace seventh_term_arithmetic_sequence_l2117_211762

theorem seventh_term_arithmetic_sequence (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 20)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 28 / 3 := 
sorry

end seventh_term_arithmetic_sequence_l2117_211762


namespace noah_holidays_l2117_211761

theorem noah_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_total : ℕ) 
  (h1 : holidays_per_month = 3) (h2 : months_in_year = 12) (h3 : holidays_total = holidays_per_month * months_in_year) : 
  holidays_total = 36 := 
by
  sorry

end noah_holidays_l2117_211761


namespace tangency_condition_intersection_condition_l2117_211783

-- Definitions of the circle and line for the given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 8 * y + 12 = 0
def line_eq (a x y : ℝ) : Prop := a * x + y + 2 * a = 0

/-- Theorem for the tangency condition -/
theorem tangency_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = 2) →
  a = 3 / 4 :=
by
  sorry

/-- Theorem for the intersection condition -/
theorem intersection_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = Real.sqrt 2) →
  (a = 1 ∨ a = 7) →
  (∀ x y : ℝ,
    (line_eq 1 x y ∧ line_eq 7 x y ↔ 
    (7 * x + y + 14 = 0 ∨ x + y + 2 = 0))) :=
by
  sorry

end tangency_condition_intersection_condition_l2117_211783


namespace find_value_l2117_211790

variable {a b c : ℝ}

def ellipse_eqn (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

theorem find_value 
  (h1 : a^2 + b^2 - 3*c^2 = 0)
  (h2 : a^2 = b^2 + c^2) :
  (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := 
  sorry

end find_value_l2117_211790
