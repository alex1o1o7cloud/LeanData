import Mathlib

namespace find_max_a_l398_39862

def f (a x : ℝ) := a * x^3 - x

theorem find_max_a (a : ℝ) (h : ∃ t : ℝ, |f a (t + 2) - f a t| ≤ 2 / 3) :
  a ≤ 4 / 3 :=
sorry

end find_max_a_l398_39862


namespace store_credit_percentage_l398_39885

theorem store_credit_percentage (SN NES cash_given change_back game_value : ℕ) (P : ℚ)
  (hSN : SN = 150)
  (hNES : NES = 160)
  (hcash_given : cash_given = 80)
  (hchange_back : change_back = 10)
  (hgame_value : game_value = 30)
  (hP_def : NES = P * SN + (cash_given - change_back) + game_value) :
  P = 0.4 :=
  sorry

end store_credit_percentage_l398_39885


namespace trigonometric_identity_l398_39819

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l398_39819


namespace joint_purchases_popular_l398_39870

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end joint_purchases_popular_l398_39870


namespace parabola_b_value_l398_39834

variable (a b c p : ℝ)
variable (h1 : p ≠ 0)
variable (h2 : ∀ x, y = a*x^2 + b*x + c)
variable (h3 : vertex' y = (p, -p))
variable (h4 : y-intercept' y = (0, p))

theorem parabola_b_value : b = -4 :=
sorry

end parabola_b_value_l398_39834


namespace average_marks_l398_39843

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end average_marks_l398_39843


namespace area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l398_39890

-- (a) Proving the area of triangle ABC
theorem area_triangle_ABC (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 3) (h_right : true) : 
  (1 / 2) * AB * BC = 3 := sorry

-- (b) Proving the area of figure DEFGH
theorem area_figure_DEFGH (DH HG : ℝ) (hDH : DH = 5) (hHG : HG = 5) (triangle_area : ℝ) (hEPF : triangle_area = 3) : 
  DH * HG - triangle_area = 22 := sorry

-- (c) Proving the area of triangle JKL 
theorem area_triangle_JKL (side_area : ℝ) (h_side : side_area = 25) 
  (area_JSK : ℝ) (h_JSK : area_JSK = 3) 
  (area_LQJ : ℝ) (h_LQJ : area_LQJ = 15/2) 
  (area_LRK : ℝ) (h_LRK : area_LRK = 5) : 
  side_area - area_JSK - area_LQJ - area_LRK = 19/2 := sorry

end area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l398_39890


namespace inequality_proof_l398_39838

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * b * (b + 1) * (c + 1))) + 
  (1 / (b * c * (c + 1) * (a + 1))) + 
  (1 / (c * a * (a + 1) * (b + 1))) ≥ 
  (3 / (1 + a * b * c)^2) :=
sorry

end inequality_proof_l398_39838


namespace number_of_groups_is_correct_l398_39867

-- Defining the conditions
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6
def total_players : Nat := new_players + returning_players

-- Theorem to prove the number of groups
theorem number_of_groups_is_correct : total_players / players_per_group = 9 := by
  sorry

end number_of_groups_is_correct_l398_39867


namespace star_3_2_l398_39814

-- Definition of the operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- The proof problem
theorem star_3_2 : star 3 2 = 22 :=
by
  sorry

end star_3_2_l398_39814


namespace numbers_with_special_remainder_property_l398_39863

theorem numbers_with_special_remainder_property (n : ℕ) :
  (∀ q : ℕ, q > 0 → n % (q ^ 2) < (q ^ 2) / 2) ↔ (n = 1 ∨ n = 4) := 
by
  sorry

end numbers_with_special_remainder_property_l398_39863


namespace F_at_2_eq_minus_22_l398_39837

variable (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x^5 + c * x^3 + d * x

def F (x : ℝ) : ℝ := f a b c d x - 6

theorem F_at_2_eq_minus_22 (h : F a b c d (-2) = 10) : F a b c d 2 = -22 :=
by
  sorry

end F_at_2_eq_minus_22_l398_39837


namespace find_d_l398_39809

theorem find_d (d : ℤ) (h : ∀ x : ℤ, 8 * x^3 + 23 * x^2 + d * x + 45 = 0 → 2 * x + 5 = 0) : 
  d = 163 := 
sorry

end find_d_l398_39809


namespace unique_number_l398_39864

theorem unique_number (a : ℕ) (h1 : 1 < a) 
  (h2 : ∀ p : ℕ, Prime p → p ∣ a^6 - 1 → p ∣ a^3 - 1 ∨ p ∣ a^2 - 1) : a = 2 :=
by
  sorry

end unique_number_l398_39864


namespace sum_of_proper_divisors_30_is_42_l398_39871

def is_proper_divisor (n d : ℕ) : Prop := d ∣ n ∧ d ≠ n

-- The set of proper divisors of 30.
def proper_divisors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15}

-- The sum of all proper divisors of 30.
def sum_proper_divisors_30 : ℕ := proper_divisors_30.sum id

theorem sum_of_proper_divisors_30_is_42 : sum_proper_divisors_30 = 42 := 
by
  -- Proof can be filled in here
  sorry

end sum_of_proper_divisors_30_is_42_l398_39871


namespace find_ab_l398_39857
-- Import the necessary Lean libraries 

-- Define the statement for the proof problem
theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : ab = 9 :=
by {
    sorry
}

end find_ab_l398_39857


namespace initial_units_of_phones_l398_39879

theorem initial_units_of_phones
  (X : ℕ) 
  (h1 : 5 = 5) 
  (h2 : X - 5 = 3 + 5 + 7) : 
  X = 20 := 
by
  sorry

end initial_units_of_phones_l398_39879


namespace inequality_arith_geo_mean_l398_39805

theorem inequality_arith_geo_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
by
  sorry

end inequality_arith_geo_mean_l398_39805


namespace hyperbola_standard_equation_l398_39849

theorem hyperbola_standard_equation (a b : ℝ) :
  (∃ (P Q : ℝ × ℝ), P = (-3, 2 * Real.sqrt 7) ∧ Q = (-6 * Real.sqrt 2, -7) ∧
    (∀ x y b, y^2 / b^2 - x^2 / a^2 = 1 ∧ (2 * Real.sqrt 7)^2 / b^2 - (-3)^2 / a^2 = 1
    ∧ (-7)^2 / b^2 - (-6 * Real.sqrt 2)^2 / a^2 = 1)) →
  b^2 = 25 → a^2 = 75 →
  (∀ x y, y^2 / (25:ℝ) - x^2 / (75:ℝ) = 1) :=
sorry

end hyperbola_standard_equation_l398_39849


namespace weight_of_new_person_l398_39887

theorem weight_of_new_person 
  (average_weight_first_20 : ℕ → ℕ → ℕ)
  (new_average_weight : ℕ → ℕ → ℕ) 
  (total_weight_21 : ℕ): 
  (average_weight_first_20 1200 20 = 60) → 
  (new_average_weight (1200 + total_weight_21) 21 = 55) → 
  total_weight_21 = 55 := 
by 
  intros 
  sorry

end weight_of_new_person_l398_39887


namespace polynomial_remainder_l398_39812

theorem polynomial_remainder (a b : ℤ) :
  (∀ x : ℤ, 3 * x ^ 6 - 2 * x ^ 4 + 5 * x ^ 2 - 9 = (x + 1) * (x + 2) * (q : ℤ) + a * x + b) →
  (a = -174 ∧ b = -177) :=
by sorry

end polynomial_remainder_l398_39812


namespace distribute_teachers_l398_39869

theorem distribute_teachers :
  let math_teachers := 3
  let lang_teachers := 3 
  let schools := 2
  let teachers_each_school := 3
  let distribution_plans := 
    (math_teachers.choose 2) * (lang_teachers.choose 1) + 
    (math_teachers.choose 1) * (lang_teachers.choose 2)
  distribution_plans = 18 := 
by
  sorry

end distribute_teachers_l398_39869


namespace friend_decks_l398_39826

noncomputable def cost_per_deck : ℕ := 8
noncomputable def victor_decks : ℕ := 6
noncomputable def total_amount_spent : ℕ := 64

theorem friend_decks :
  ∃ x : ℕ, (victor_decks * cost_per_deck) + (x * cost_per_deck) = total_amount_spent ∧ x = 2 :=
by
  sorry

end friend_decks_l398_39826


namespace find_pre_tax_remuneration_l398_39823

def pre_tax_remuneration (x : ℝ) : Prop :=
  let taxable_amount := if x <= 4000 then x - 800 else x * 0.8
  let tax_due := taxable_amount * 0.2
  let final_tax := tax_due * 0.7
  final_tax = 280

theorem find_pre_tax_remuneration : ∃ x : ℝ, pre_tax_remuneration x ∧ x = 2800 := by
  sorry

end find_pre_tax_remuneration_l398_39823


namespace andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l398_39816

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that André wins the book is 1/4. -/
theorem andre_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let probability := (black_balls : ℚ) / total_balls
  probability = 1 / 4 := 
by 
  sorry

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that Dalva wins the book is 1/4. -/
theorem dalva_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let andre_white := (3 / 4 : ℚ)
  let bianca_white := (2 / 3 : ℚ)
  let carlos_white := (1 / 2 : ℚ)
  let probability := andre_white * bianca_white * carlos_white * (black_balls / (total_balls - 3))
  probability = 1 / 4 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that André wins the book is 5/14. -/
theorem andre_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_first_black := (black_balls : ℚ) / total_balls
  let andre_fifth_black := (((6 / 8 : ℚ) * (5 / 7 : ℚ) * (4 / 6 : ℚ) * (3 / 5 : ℚ)) * black_balls / (total_balls - 4))
  let probability := andre_first_black + andre_fifth_black
  probability = 5 / 14 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that Dalva wins the book is 1/7. -/
theorem dalva_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_white := (6 / 8 : ℚ)
  let bianca_white := (5 / 7 : ℚ)
  let carlos_white := (4 / 6 : ℚ)
  let dalva_black := (black_balls / (total_balls - 3))
  let probability := andre_white * bianca_white * carlos_white * dalva_black
  probability = 1 / 7 := 
by 
  sorry

end andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l398_39816


namespace shopper_saves_more_l398_39801

-- Definitions and conditions
def cover_price : ℝ := 30
def percent_discount : ℝ := 0.25
def dollar_discount : ℝ := 5
def first_discounted_price : ℝ := cover_price * (1 - percent_discount)
def second_discounted_price : ℝ := first_discounted_price - dollar_discount
def first_dollar_discounted_price : ℝ := cover_price - dollar_discount
def second_percent_discounted_price : ℝ := first_dollar_discounted_price * (1 - percent_discount)

def additional_savings : ℝ := second_percent_discounted_price - second_discounted_price

-- Theorem stating the shopper saves 125 cents more with 25% first
theorem shopper_saves_more : additional_savings = 1.25 := by
  sorry

end shopper_saves_more_l398_39801


namespace rabbits_initially_bought_l398_39883

theorem rabbits_initially_bought (R : ℕ) (h : ∃ (k : ℕ), R + 6 = 17 * k) : R = 28 :=
sorry

end rabbits_initially_bought_l398_39883


namespace problem_HMMT_before_HMT_l398_39889
noncomputable def probability_of_sequence (seq: List Char) : ℚ := sorry
def probability_H : ℚ := 1 / 3
def probability_M : ℚ := 1 / 3
def probability_T : ℚ := 1 / 3

theorem problem_HMMT_before_HMT : probability_of_sequence ['H', 'M', 'M', 'T'] = 1 / 4 :=
sorry

end problem_HMMT_before_HMT_l398_39889


namespace total_monthly_cost_l398_39861

theorem total_monthly_cost (volume_per_box : ℕ := 1800) 
                          (total_volume : ℕ := 1080000)
                          (cost_per_box_per_month : ℝ := 0.8) 
                          (expected_cost : ℝ := 480) : 
                          (total_volume / volume_per_box) * cost_per_box_per_month = expected_cost :=
by
  sorry

end total_monthly_cost_l398_39861


namespace V_product_is_V_form_l398_39854

noncomputable def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3 * a * b * c

theorem V_product_is_V_form (a b c x y z : ℝ) :
  V a b c * V x y z = V (a * x + b * y + c * z) (b * x + c * y + a * z) (c * x + a * y + b * z) := by
  sorry

end V_product_is_V_form_l398_39854


namespace find_m_if_perpendicular_l398_39872

theorem find_m_if_perpendicular 
  (m : ℝ)
  (h : ∀ m (slope1 : ℝ) (slope2 : ℝ), 
    (slope1 = -m) → 
    (slope2 = (-1) / (3 - 2 * m)) → 
    slope1 * slope2 = -1)
  : m = 3 := 
by
  sorry

end find_m_if_perpendicular_l398_39872


namespace pool_depths_l398_39836

theorem pool_depths (J S Su : ℝ) 
  (h1 : J = 15) 
  (h2 : J = 2 * S + 5) 
  (h3 : Su = J + S - 3) : 
  S = 5 ∧ Su = 17 := 
by 
  -- proof steps go here
  sorry

end pool_depths_l398_39836


namespace monthly_rent_of_shop_l398_39888

theorem monthly_rent_of_shop
  (length width : ℕ)
  (annual_rent_per_sq_ft : ℕ)
  (length_def : length = 18)
  (width_def : width = 22)
  (annual_rent_per_sq_ft_def : annual_rent_per_sq_ft = 68) :
  (18 * 22 * 68) / 12 = 2244 := 
by
  sorry

end monthly_rent_of_shop_l398_39888


namespace modified_full_house_probability_l398_39841

def total_choices : ℕ := Nat.choose 52 6

def ways_rank1 : ℕ := 13
def ways_3_cards : ℕ := Nat.choose 4 3
def ways_rank2 : ℕ := 12
def ways_2_cards : ℕ := Nat.choose 4 2
def ways_additional_card : ℕ := 11 * 4

def ways_modified_full_house : ℕ := ways_rank1 * ways_3_cards * ways_rank2 * ways_2_cards * ways_additional_card

def probability_modified_full_house : ℚ := ways_modified_full_house / total_choices

theorem modified_full_house_probability : probability_modified_full_house = 24 / 2977 := 
by sorry

end modified_full_house_probability_l398_39841


namespace min_value_of_expression_l398_39806

theorem min_value_of_expression :
  ∀ (x y : ℝ), ∃ a b : ℝ, x = 5 ∧ y = -3 ∧ (x^2 + y^2 - 10*x + 6*y + 25) = -9 := 
by
  sorry

end min_value_of_expression_l398_39806


namespace find_first_number_in_list_l398_39875

theorem find_first_number_in_list
  (x : ℕ)
  (h1 : x < 10)
  (h2 : ∃ n : ℕ, 2012 = x + 9 * n)
  : x = 5 :=
by
  sorry

end find_first_number_in_list_l398_39875


namespace range_of_p_l398_39832

noncomputable def a_n (p : ℝ) (n : ℕ) : ℝ := -2 * n + p
noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n - 7)

noncomputable def c_n (p : ℝ) (n : ℕ) : ℝ :=
if a_n p n <= b_n n then a_n p n else b_n n

theorem range_of_p (p : ℝ) :
  (∀ n : ℕ, n ≠ 10 → c_n p 10 > c_n p n) ↔ 24 < p ∧ p < 30 :=
sorry

end range_of_p_l398_39832


namespace tall_cupboard_glasses_l398_39804

-- Define the number of glasses held by the tall cupboard (T)
variable (T : ℕ)

-- Condition: Wide cupboard holds twice as many glasses as the tall cupboard
def wide_cupboard_holds_twice_as_many (T : ℕ) : Prop :=
  ∃ W : ℕ, W = 2 * T

-- Condition: Narrow cupboard holds 15 glasses initially, 5 glasses per shelf, one shelf broken
def narrow_cupboard_holds_after_break : Prop :=
  ∃ N : ℕ, N = 10

-- Final statement to prove: Number of glasses in the tall cupboard is 5
theorem tall_cupboard_glasses (T : ℕ) (h1 : wide_cupboard_holds_twice_as_many T) (h2 : narrow_cupboard_holds_after_break) : T = 5 :=
sorry

end tall_cupboard_glasses_l398_39804


namespace find_first_part_l398_39855

variable (x y : ℕ)

theorem find_first_part (h₁ : x + y = 24) (h₂ : 7 * x + 5 * y = 146) : x = 13 :=
by
  -- The proof is omitted
  sorry

end find_first_part_l398_39855


namespace socks_expected_value_l398_39884

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l398_39884


namespace remainder_98_pow_50_mod_100_l398_39803

theorem remainder_98_pow_50_mod_100 :
  (98 : ℤ) ^ 50 % 100 = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l398_39803


namespace square_fits_in_unit_cube_l398_39895

theorem square_fits_in_unit_cube (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
  let PQ := Real.sqrt (2 * (1 - x) ^ 2)
  let PS := Real.sqrt (1 + 2 * x ^ 2)
  (PQ > 1.05 ∧ PS > 1.05) :=
by
  sorry

end square_fits_in_unit_cube_l398_39895


namespace solution_exists_l398_39829

theorem solution_exists :
  ∃ x : ℝ, x = 2 ∧ (-2 * x + 4 = 0) :=
sorry

end solution_exists_l398_39829


namespace find_x_l398_39810

theorem find_x (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 3 * x → x = 3 / 2 := by
  sorry

end find_x_l398_39810


namespace fraction_calls_processed_by_team_B_l398_39827

variable (A B C_A C_B : ℕ)

theorem fraction_calls_processed_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : C_A = (2 / 5) * C_B) :
  (B * C_B) / ((A * C_A) + (B * C_B)) = 8 / 9 := by
  sorry

end fraction_calls_processed_by_team_B_l398_39827


namespace number_of_tangent_lines_l398_39878

def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 + a

def on_line (a x y : ℝ) : Prop := 3 * x + y = a + 1

theorem number_of_tangent_lines (a m : ℝ) (h1 : on_line a m (a + 1 - 3 * m)) :
  ∃ n : ℤ, n = 1 ∨ n = 2 :=
sorry

end number_of_tangent_lines_l398_39878


namespace puppies_per_female_dog_l398_39835

theorem puppies_per_female_dog
  (number_of_dogs : ℕ)
  (percent_female : ℝ)
  (fraction_female_giving_birth : ℝ)
  (remaining_puppies : ℕ)
  (donated_puppies : ℕ)
  (total_puppies : ℕ)
  (number_of_female_dogs : ℕ)
  (number_female_giving_birth : ℕ)
  (puppies_per_dog : ℕ) :
  number_of_dogs = 40 →
  percent_female = 0.60 →
  fraction_female_giving_birth = 0.75 →
  remaining_puppies = 50 →
  donated_puppies = 130 →
  total_puppies = remaining_puppies + donated_puppies →
  number_of_female_dogs = percent_female * number_of_dogs →
  number_female_giving_birth = fraction_female_giving_birth * number_of_female_dogs →
  puppies_per_dog = total_puppies / number_female_giving_birth →
  puppies_per_dog = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end puppies_per_female_dog_l398_39835


namespace difference_face_local_value_8_l398_39833

theorem difference_face_local_value_8 :
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3  -- 0-indexed place for thousands
  let local_value := digit * 10^position
  local_value - face_value = 7992 :=
by
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3
  let local_value := digit * 10^position
  show local_value - face_value = 7992
  sorry

end difference_face_local_value_8_l398_39833


namespace similar_triangles_height_ratio_l398_39898

-- Given condition: two similar triangles have a similarity ratio of 3:5
def similar_triangles (ratio : ℕ) : Prop := ratio = 3 ∧ ratio = 5

-- Goal: What is the ratio of their corresponding heights?
theorem similar_triangles_height_ratio (r : ℕ) (h : similar_triangles r) :
  r = 3 / 5 :=
sorry

end similar_triangles_height_ratio_l398_39898


namespace sugar_amount_l398_39876

noncomputable def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ :=
  a + b / c

theorem sugar_amount (a : ℚ) (h : a = mixed_to_improper 7 3 4) : 1 / 3 * a = 2 + 7 / 12 :=
by
  rw [h]
  simp
  sorry

end sugar_amount_l398_39876


namespace find_m_if_a_b_parallel_l398_39820

theorem find_m_if_a_b_parallel :
  ∃ m : ℝ, (∃ a : ℝ × ℝ, a = (-2, 1)) ∧ (∃ b : ℝ × ℝ, b = (1, m)) ∧ (m * -2 = 1) ∧ (m = -1 / 2) :=
by
  sorry

end find_m_if_a_b_parallel_l398_39820


namespace calculate_expression_l398_39881

theorem calculate_expression :
  -1 ^ 2023 + (Real.pi - 3.14) ^ 0 + |(-2 : ℝ)| = 2 :=
by
  sorry

end calculate_expression_l398_39881


namespace detergent_required_l398_39811

def ounces_of_detergent_per_pound : ℕ := 2
def pounds_of_clothes : ℕ := 9

theorem detergent_required :
  (ounces_of_detergent_per_pound * pounds_of_clothes) = 18 := by
  sorry

end detergent_required_l398_39811


namespace multiplication_difference_is_1242_l398_39868

theorem multiplication_difference_is_1242 (a b c : ℕ) (h1 : a = 138) (h2 : b = 43) (h3 : c = 34) :
  a * b - a * c = 1242 :=
by
  sorry

end multiplication_difference_is_1242_l398_39868


namespace total_games_played_l398_39845

-- Definition of the number of teams
def num_teams : ℕ := 20

-- Definition of the number of games each pair plays
def games_per_pair : ℕ := 10

-- Theorem stating the total number of games played
theorem total_games_played : (num_teams * (num_teams - 1) / 2) * games_per_pair = 1900 :=
by sorry

end total_games_played_l398_39845


namespace negation_equiv_l398_39897

-- Define the proposition that the square of all real numbers is positive
def pos_of_all_squares : Prop := ∀ x : ℝ, x^2 > 0

-- Define the negation of the proposition
def neg_pos_of_all_squares : Prop := ∃ x : ℝ, x^2 ≤ 0

theorem negation_equiv (h : ¬ pos_of_all_squares) : neg_pos_of_all_squares :=
  sorry

end negation_equiv_l398_39897


namespace sock_pairs_count_l398_39856

theorem sock_pairs_count :
  let white_socks := 5
  let brown_socks := 3
  let blue_socks := 4
  let blue_white_pairs := blue_socks * white_socks
  let blue_brown_pairs := blue_socks * brown_socks
  let total_pairs := blue_white_pairs + blue_brown_pairs
  total_pairs = 32 :=
by
  sorry

end sock_pairs_count_l398_39856


namespace area_of_figure_l398_39891
-- Import necessary libraries

-- Define the conditions as functions/constants
def length_left : ℕ := 7
def width_top : ℕ := 6
def height_middle : ℕ := 3
def width_middle : ℕ := 4
def height_right : ℕ := 5
def width_right : ℕ := 5

-- State the problem as a theorem
theorem area_of_figure : 
  (length_left * width_top) + 
  (width_middle * height_middle) + 
  (width_right * height_right) = 79 := 
  by
  sorry

end area_of_figure_l398_39891


namespace john_task_completion_time_l398_39850

/-- John can complete a task alone in 18 days given the conditions. -/
theorem john_task_completion_time :
  ∀ (John Jane taskDays : ℝ), 
    Jane = 12 → 
    taskDays = 10.8 → 
    (10.8 - 6) * (1 / 12) + 10.8 * (1 / John) = 1 → 
    John = 18 :=
by
  intros John Jane taskDays hJane hTaskDays hWorkDone
  sorry

end john_task_completion_time_l398_39850


namespace expression_value_l398_39846

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end expression_value_l398_39846


namespace rectangle_constant_k_l398_39802

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end rectangle_constant_k_l398_39802


namespace stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l398_39892

-- Definitions of fixed points and stable points
def is_fixed_point(f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def is_stable_point(f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x 

-- Problem 1: Stable points of g(x) = 2x - 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem stable_points_of_g : {x : ℝ | is_stable_point g x} = {1} :=
sorry

-- Problem 2: Prove A ⊂ B for any function f
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) : 
  {x : ℝ | is_fixed_point f x} ⊆ {x : ℝ | is_stable_point f x} :=
sorry

-- Problem 3: Range of a for f(x) = ax^2 - 1 when A = B ≠ ∅
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

theorem range_of_a (a : ℝ) (h : ∃ x, is_fixed_point (f a) x ∧ is_stable_point (f a) x):
  - (1/4 : ℝ) ≤ a ∧ a ≤ (3/4 : ℝ) :=
sorry

end stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l398_39892


namespace sqrt_diff_eq_neg_four_sqrt_five_l398_39866

theorem sqrt_diff_eq_neg_four_sqrt_five : 
  (Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5)) = -4 * Real.sqrt 5 := 
sorry

end sqrt_diff_eq_neg_four_sqrt_five_l398_39866


namespace range_of_m_l398_39853

noncomputable def f (a x: ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem range_of_m (a m x₁ x₂: ℝ) (h₁: a ∈ Set.Icc (-3) (0)) (h₂: x₁ ∈ Set.Icc (0) (2)) (h₃: x₂ ∈ Set.Icc (0) (2)) : m ∈ Set.Ici (5) → m - a * m^2 ≥ |f a x₁ - f a x₂| :=
sorry

end range_of_m_l398_39853


namespace max_min_values_of_f_l398_39859

noncomputable def f (x : ℝ) : ℝ :=
  4^x - 2^(x+1) - 3

theorem max_min_values_of_f :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → (∀ y, y = f x → y ≤ 5) ∧ (∃ y, y = f 2 ∧ y = 5) ∧ (∀ y, y = f x → y ≥ -4) ∧ (∃ y, y = f 0 ∧ y = -4) :=
by
  sorry

end max_min_values_of_f_l398_39859


namespace quad_root_l398_39894

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end quad_root_l398_39894


namespace bicycle_total_distance_l398_39839

noncomputable def front_wheel_circumference : ℚ := 4/3
noncomputable def rear_wheel_circumference : ℚ := 3/2
noncomputable def extra_revolutions : ℕ := 25

theorem bicycle_total_distance :
  (front_wheel_circumference * extra_revolutions + (rear_wheel_circumference * 
  ((front_wheel_circumference * extra_revolutions) / (rear_wheel_circumference - front_wheel_circumference))) = 300) := sorry

end bicycle_total_distance_l398_39839


namespace composite_numbers_with_same_main_divisors_are_equal_l398_39858

theorem composite_numbers_with_same_main_divisors_are_equal (a b : ℕ) 
  (h_a_not_prime : ¬ Prime a)
  (h_b_not_prime : ¬ Prime b)
  (h_a_comp : 1 < a ∧ ∃ p, p ∣ a ∧ p ≠ a)
  (h_b_comp : 1 < b ∧ ∃ p, p ∣ b ∧ p ≠ b)
  (main_divisors : {d : ℕ // d ∣ a ∧ d ≠ a} = {d : ℕ // d ∣ b ∧ d ≠ b}) :
  a = b := 
sorry

end composite_numbers_with_same_main_divisors_are_equal_l398_39858


namespace jellybean_count_l398_39821

theorem jellybean_count (x : ℕ) (h : (0.7 : ℝ) ^ 3 * x = 34) : x = 99 :=
sorry

end jellybean_count_l398_39821


namespace tail_wind_distance_l398_39817

-- Definitions based on conditions
def speed_still_air : ℝ := 262.5
def t1 : ℝ := 3
def t2 : ℝ := 4

def effective_speed_tail_wind (w : ℝ) : ℝ := speed_still_air + w
def effective_speed_against_wind (w : ℝ) : ℝ := speed_still_air - w

theorem tail_wind_distance (w : ℝ) (d : ℝ) :
  effective_speed_tail_wind w * t1 = effective_speed_against_wind w * t2 →
  d = t1 * effective_speed_tail_wind w →
  d = 900 :=
by
  sorry

end tail_wind_distance_l398_39817


namespace ratio_of_selling_to_buying_l398_39852

noncomputable def natasha_has_3_times_carla (N C : ℕ) : Prop :=
  N = 3 * C

noncomputable def carla_has_2_times_cosima (C S : ℕ) : Prop :=
  C = 2 * S

noncomputable def total_buying_price (N C S : ℕ) : ℕ :=
  N + C + S

noncomputable def total_selling_price (buying_price profit : ℕ) : ℕ :=
  buying_price + profit

theorem ratio_of_selling_to_buying (N C S buying_price selling_price ratio : ℕ) 
  (h1 : natasha_has_3_times_carla N C)
  (h2 : carla_has_2_times_cosima C S)
  (h3 : N = 60)
  (h4 : buying_price = total_buying_price N C S)
  (h5 : total_selling_price buying_price 36 = selling_price)
  (h6 : 18 * ratio = selling_price * 5): ratio = 7 :=
by
  sorry

end ratio_of_selling_to_buying_l398_39852


namespace range_of_g_l398_39825

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) -
  (Real.arcsin (x / 3))^2 + (Real.pi^2 / 18) * (x^2 + 12 * x + 27)

lemma arccos_arcsin_identity (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.arccos x + Real.arcsin x = Real.pi / 2 := sorry

theorem range_of_g : ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, g x = y ∧ y ∈ Set.Icc (Real.pi^2 / 4) (5 * Real.pi^2 / 2) :=
sorry

end range_of_g_l398_39825


namespace order_of_abc_l398_39822

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem order_of_abc : c < a ∧ a < b :=
by {
  -- The proof would go here
  sorry
}

end order_of_abc_l398_39822


namespace sale_in_third_month_l398_39831

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (avg_sale_months : ℕ) 
  (total_sales : ℕ)
  (h1 : sale1 = 6435) 
  (h2 : sale2 = 6927) 
  (h4 : sale4 = 7230) 
  (h5 : sale5 = 6562) 
  (h6 : sale6 = 7991) 
  (h_avg : avg_sale_months = 7000) 
  (h_total : total_sales = 6 * avg_sale_months) 
  : (total_sales - (sale1 + sale2 + sale4 + sale5 + sale6)) = 6855 :=
by
  have sales_sum := sale1 + sale2 + sale4 + sale5 + sale6
  have required_sales := total_sales - sales_sum
  sorry

end sale_in_third_month_l398_39831


namespace polynomial_factor_l398_39807

theorem polynomial_factor (a b : ℝ) : 
  (∃ c d : ℝ, (5 * c = a) ∧ (5 * d - 3 * c = b) ∧ (2 * c - 3 * d + 25 = 45) ∧ (2 * d - 15 = -18)) 
  → (a = 151.25 ∧ b = -98.25) :=
by
  sorry

end polynomial_factor_l398_39807


namespace perimeter_of_triangle_l398_39840

theorem perimeter_of_triangle (r a : ℝ) (p : ℝ) (h1 : r = 3.5) (h2 : a = 56) :
  p = 32 :=
by
  sorry

end perimeter_of_triangle_l398_39840


namespace white_ball_probability_l398_39893

theorem white_ball_probability (m : ℕ) 
  (initial_black : ℕ := 6) 
  (initial_white : ℕ := 10) 
  (added_white := 14) 
  (probability := 0.8) :
  (10 + added_white) / (16 + added_white) = probability :=
by
  -- no proof required
  sorry

end white_ball_probability_l398_39893


namespace find_a_value_l398_39813

theorem find_a_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq1 : a^b = b^a) (h_eq2 : b = 3 * a) : a = Real.sqrt 3 :=
  sorry

end find_a_value_l398_39813


namespace largest_final_number_l398_39848

-- Define the sequence and conditions
def initial_number := List.replicate 40 [3, 1, 1, 2, 3] |> List.join

-- The transformation rule
def valid_transform (a b : ℕ) : ℕ := if a + b <= 9 then a + b else 0

-- Sum of digits of a number
def sum_digits : List ℕ → ℕ := List.foldr (· + ·) 0

-- Define the final valid number pattern
def valid_final_pattern (n : ℕ) : Prop := n = 77

-- The main theorem statement
theorem largest_final_number (seq : List ℕ) (h_seq : seq = initial_number) :
  valid_final_pattern (sum_digits seq) := sorry

end largest_final_number_l398_39848


namespace rose_tom_profit_difference_l398_39886

def investment_months (amount: ℕ) (months: ℕ) : ℕ :=
  amount * months

def total_investment_months (john_inv: ℕ) (rose_inv: ℕ) (tom_inv: ℕ) : ℕ :=
  john_inv + rose_inv + tom_inv

def profit_share (investment: ℕ) (total_investment: ℕ) (total_profit: ℕ) : ℤ :=
  (investment * total_profit) / total_investment

theorem rose_tom_profit_difference
  (john_inv rs_per_year: ℕ := 18000 * 12)
  (rose_inv rs_per_9_months: ℕ := 12000 * 9)
  (tom_inv rs_per_8_months: ℕ := 9000 * 8)
  (total_profit: ℕ := 4070):
  profit_share rose_inv (total_investment_months john_inv rose_inv tom_inv) total_profit -
  profit_share tom_inv (total_investment_months john_inv rose_inv tom_inv) total_profit = 370 := 
by
  sorry

end rose_tom_profit_difference_l398_39886


namespace general_term_of_sequence_l398_39865

variable (a : ℕ → ℕ)
variable (h1 : ∀ m : ℕ, a (m^2) = a m ^ 2)
variable (h2 : ∀ m k : ℕ, a (m^2 + k^2) = a m * a k)

theorem general_term_of_sequence : ∀ n : ℕ, n > 0 → a n = 1 :=
by
  intros n hn
  sorry

end general_term_of_sequence_l398_39865


namespace toothpick_count_l398_39844

theorem toothpick_count (length width : ℕ) (h_len : length = 20) (h_width : width = 10) : 
  2 * (length * (width + 1) + width * (length + 1)) = 430 :=
by
  sorry

end toothpick_count_l398_39844


namespace trapezium_side_length_l398_39847

variable (length1 length2 height area : ℕ)

theorem trapezium_side_length
  (h1 : length1 = 20)
  (h2 : height = 15)
  (h3 : area = 270)
  (h4 : area = (length1 + length2) * height / 2) :
  length2 = 16 :=
by
  sorry

end trapezium_side_length_l398_39847


namespace jerry_age_is_10_l398_39896

-- Define the ages of Mickey and Jerry
def MickeyAge : ℝ := 20
def mickey_eq_jerry (JerryAge : ℝ) : Prop := MickeyAge = 2.5 * JerryAge - 5

theorem jerry_age_is_10 : ∃ JerryAge : ℝ, mickey_eq_jerry JerryAge ∧ JerryAge = 10 :=
by
  -- By solving the equation MickeyAge = 2.5 * JerryAge - 5,
  -- we can find that Jerry's age must be 10.
  use 10
  sorry

end jerry_age_is_10_l398_39896


namespace towel_decrease_percentage_l398_39882

variable (L B : ℝ)
variable (h1 : 0.70 * L = L - (0.30 * L))
variable (h2 : 0.60 * B = B - (0.40 * B))

theorem towel_decrease_percentage (L B : ℝ) 
  (h1 : 0.70 * L = L - (0.30 * L))
  (h2 : 0.60 * B = B - (0.40 * B)) :
  ((L * B - (0.70 * L) * (0.60 * B)) / (L * B)) * 100 = 58 := 
by
  sorry

end towel_decrease_percentage_l398_39882


namespace decimal_to_binary_45_l398_39808

theorem decimal_to_binary_45 :
  (45 : ℕ) = (0b101101 : ℕ) :=
sorry

end decimal_to_binary_45_l398_39808


namespace store_A_profit_margin_l398_39899

theorem store_A_profit_margin
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > x)
  (h : (y - x) / x + 0.12 = (y - 0.9 * x) / (0.9 * x)) :
  (y - x) / x = 0.08 :=
by {
  sorry
}

end store_A_profit_margin_l398_39899


namespace num_games_last_year_l398_39860

-- Definitions from conditions
def num_games_this_year : ℕ := 14
def total_num_games : ℕ := 43

-- Theorem to prove
theorem num_games_last_year (num_games_last_year : ℕ) : 
  total_num_games - num_games_this_year = num_games_last_year ↔ num_games_last_year = 29 :=
by
  sorry

end num_games_last_year_l398_39860


namespace measure_of_angle_C_l398_39851

variable (A B C : ℕ)

theorem measure_of_angle_C :
  (A = B - 20) →
  (C = A + 40) →
  (A + B + C = 180) →
  C = 80 :=
by
  intros h1 h2 h3
  sorry

end measure_of_angle_C_l398_39851


namespace sum_excluding_multiples_l398_39877

theorem sum_excluding_multiples (S_total S_2 S_3 S_6 : ℕ) 
  (hS_total : S_total = (100 * (1 + 100)) / 2) 
  (hS_2 : S_2 = (50 * (2 + 100)) / 2) 
  (hS_3 : S_3 = (33 * (3 + 99)) / 2) 
  (hS_6 : S_6 = (16 * (6 + 96)) / 2) :
  S_total - S_2 - S_3 + S_6 = 1633 :=
by
  sorry

end sum_excluding_multiples_l398_39877


namespace sum_of_other_two_angles_is_108_l398_39873

theorem sum_of_other_two_angles_is_108 (A B C : Type) (angleA angleB angleC : ℝ) 
  (h_angle_sum : angleA + angleB + angleC = 180) (h_angleB : angleB = 72) :
  angleA + angleC = 108 := 
by
  sorry

end sum_of_other_two_angles_is_108_l398_39873


namespace fraction_difference_in_simplest_form_l398_39830

noncomputable def difference_fraction : ℚ := (5 / 19) - (2 / 23)

theorem fraction_difference_in_simplest_form :
  difference_fraction = 77 / 437 := by sorry

end fraction_difference_in_simplest_form_l398_39830


namespace ice_cream_to_afford_games_l398_39828

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l398_39828


namespace average_weight_20_boys_l398_39815

theorem average_weight_20_boys 
  (A : Real)
  (numBoys₁ numBoys₂ : ℕ)
  (weight₂ : Real)
  (avg_weight_class : Real)
  (h_numBoys₁ : numBoys₁ = 20)
  (h_numBoys₂ : numBoys₂ = 8)
  (h_weight₂ : weight₂ = 45.15)
  (h_avg_weight_class : avg_weight_class = 48.792857142857144)
  (h_total_boys : numBoys₁ + numBoys₂ = 28)
  (h_eq_weight : numBoys₁ * A + numBoys₂ * weight₂ = 28 * avg_weight_class) :
  A = 50.25 :=
  sorry

end average_weight_20_boys_l398_39815


namespace katy_read_books_l398_39874

theorem katy_read_books (juneBooks : ℕ) (julyBooks : ℕ) (augustBooks : ℕ)
  (H1 : juneBooks = 8)
  (H2 : julyBooks = 2 * juneBooks)
  (H3 : augustBooks = julyBooks - 3) :
  juneBooks + julyBooks + augustBooks = 37 := by
  -- Proof goes here
  sorry

end katy_read_books_l398_39874


namespace tangent_line_iff_l398_39824

theorem tangent_line_iff (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 8 * y + 12 = 0 → ax + y + 2 * a = 0) ↔ a = -3 / 4 :=
by
  sorry

end tangent_line_iff_l398_39824


namespace lcm_second_factor_l398_39842

theorem lcm_second_factor (A B : ℕ) (hcf : ℕ) (f1 f2 : ℕ) 
  (h₁ : hcf = 25) 
  (h₂ : A = 350) 
  (h₃ : Nat.gcd A B = hcf) 
  (h₄ : Nat.lcm A B = hcf * f1 * f2) 
  (h₅ : f1 = 13)
  : f2 = 14 := 
sorry

end lcm_second_factor_l398_39842


namespace find_positive_k_l398_39800

noncomputable def cubic_roots (a b k : ℝ) : Prop :=
  (3 * a * a * a + 9 * a * a - 135 * a + k = 0) ∧
  (a * a * b = -45 / 2)

theorem find_positive_k :
  ∃ (a b : ℝ), ∃ (k : ℝ) (pos : k > 0), (cubic_roots a b k) ∧ (k = 525) :=
by
  sorry

end find_positive_k_l398_39800


namespace Lisa_initial_pencils_l398_39818

-- Variables
variable (G_L_initial : ℕ) (L_L_initial : ℕ) (G_L_total : ℕ)

-- Conditions
def G_L_initial_def := G_L_initial = 2
def G_L_total_def := G_L_total = 101
def Lisa_gives_pencils : Prop := G_L_total = G_L_initial + L_L_initial

-- Proof statement
theorem Lisa_initial_pencils (G_L_initial : ℕ) (G_L_total : ℕ)
  (h1 : G_L_initial = 2) (h2 : G_L_total = 101) (h3 : G_L_total = G_L_initial + L_L_initial) :
  L_L_initial = 99 := 
by 
  sorry

end Lisa_initial_pencils_l398_39818


namespace perpendicular_slope_l398_39880

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end perpendicular_slope_l398_39880
