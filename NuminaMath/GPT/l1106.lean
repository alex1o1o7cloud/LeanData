import Mathlib

namespace problem_condition_problem_statement_l1106_110659

noncomputable def a : ℕ → ℕ 
| 0     => 2
| (n+1) => 3 * a n

noncomputable def S : ℕ → ℕ
| 0     => 0
| (n+1) => S n + a n

theorem problem_condition : ∀ n, 3 * a n - 2 * S n = 2 :=
by
  sorry

theorem problem_statement (n : ℕ) (h : ∀ n, 3 * a n - 2 * S n = 2) :
  (S (n+1))^2 - (S n) * (S (n+2)) = 4 * 3^n :=
by
  sorry

end problem_condition_problem_statement_l1106_110659


namespace polynomial_evaluation_l1106_110629

-- Define the polynomial p(x) and the conditions
noncomputable def p (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

-- Given conditions for p(1), p(2), p(3)
variables (a b c d : ℝ)
axiom h₁ : p 1 a b c d = 1993
axiom h₂ : p 2 a b c d = 3986
axiom h₃ : p 3 a b c d = 5979

-- The final proof statement
theorem polynomial_evaluation :
  (1 / 4) * (p 11 a b c d + p (-7) a b c d) = 5233 :=
sorry

end polynomial_evaluation_l1106_110629


namespace sin_log_infinite_zeros_in_01_l1106_110616

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01_l1106_110616


namespace simplify_expression_l1106_110699

theorem simplify_expression (x y : ℝ) : 3 * x + 2 * y + 4 * x + 5 * y + 7 = 7 * x + 7 * y + 7 := 
by sorry

end simplify_expression_l1106_110699


namespace height_of_fourth_person_l1106_110652

theorem height_of_fourth_person 
  (H : ℕ) 
  (h_avg : ((H) + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) :
  (H + 10 = 85) :=
by
  sorry

end height_of_fourth_person_l1106_110652


namespace Bobby_paycheck_final_amount_l1106_110663

theorem Bobby_paycheck_final_amount :
  let salary := 450
  let federal_tax := (1 / 3 : ℚ) * salary
  let state_tax := 0.08 * salary
  let health_insurance := 50
  let life_insurance := 20
  let city_fee := 10
  let total_deductions := federal_tax + state_tax + health_insurance + life_insurance + city_fee
  salary - total_deductions = 184 :=
by
  -- We put sorry here to skip the proof step
  sorry

end Bobby_paycheck_final_amount_l1106_110663


namespace total_members_in_club_l1106_110687

theorem total_members_in_club (females : ℕ) (males : ℕ) (total : ℕ) : 
  (females = 12) ∧ (females = 2 * males) ∧ (total = females + males) → total = 18 := 
by
  sorry

end total_members_in_club_l1106_110687


namespace find_constant_t_l1106_110626

theorem find_constant_t :
  (exists t : ℚ,
  ∀ x : ℚ,
    (5 * x ^ 2 - 6 * x + 7) * (4 * x ^ 2 + t * x + 10) =
      20 * x ^ 4 - 48 * x ^ 3 + 114 * x ^ 2 - 102 * x + 70) :=
sorry

end find_constant_t_l1106_110626


namespace max_integer_value_l1106_110686

theorem max_integer_value (x : ℝ) : ∃ (m : ℤ), m = 53 ∧ ∀ y : ℝ, (1 + 13 / (3 * y^2 + 9 * y + 7) ≤ m) := 
sorry

end max_integer_value_l1106_110686


namespace product_of_sums_of_squares_l1106_110633

theorem product_of_sums_of_squares (a b : ℤ) 
  (h1 : ∃ x1 y1 : ℤ, a = x1^2 + y1^2)
  (h2 : ∃ x2 y2 : ℤ, b = x2^2 + y2^2) : 
  ∃ x y : ℤ, a * b = x^2 + y^2 :=
by
  sorry

end product_of_sums_of_squares_l1106_110633


namespace age_problem_l1106_110651

theorem age_problem (S Sh K : ℕ) 
  (h1 : S / Sh = 4 / 3)
  (h2 : S / K = 4 / 2)
  (h3 : K + 10 = S)
  (h4 : S + 8 = 30) :
  S = 22 ∧ Sh = 17 ∧ K = 10 := 
sorry

end age_problem_l1106_110651


namespace simplify_sum_of_polynomials_l1106_110608

-- Definitions of the given polynomials
def P (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15
def Q (x : ℝ) : ℝ := -5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9

-- Statement to prove that the sum of P and Q equals the simplified polynomial
theorem simplify_sum_of_polynomials (x : ℝ) : 
  P x + Q x = 2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := 
sorry

end simplify_sum_of_polynomials_l1106_110608


namespace max_value_2ab_2bc_root_3_l1106_110681

theorem max_value_2ab_2bc_root_3 (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a^2 + b^2 + c^2 = 3) :
  2 * a * b + 2 * b * c * Real.sqrt 3 ≤ 6 := by
sorry

end max_value_2ab_2bc_root_3_l1106_110681


namespace candy_cost_55_cents_l1106_110605

theorem candy_cost_55_cents
  (paid: ℕ) (change: ℕ) (num_coins: ℕ)
  (coin1 coin2 coin3 coin4: ℕ)
  (h1: paid = 100)
  (h2: num_coins = 4)
  (h3: coin1 = 25)
  (h4: coin2 = 10)
  (h5: coin3 = 10)
  (h6: coin4 = 0)
  (h7: change = coin1 + coin2 + coin3 + coin4) :
  paid - change = 55 :=
by
  -- The proof can be provided here.
  sorry

end candy_cost_55_cents_l1106_110605


namespace number_of_buses_l1106_110653

theorem number_of_buses (vans people_per_van buses people_per_bus extra_people_in_buses : ℝ) 
  (h_vans : vans = 6.0) 
  (h_people_per_van : people_per_van = 6.0) 
  (h_people_per_bus : people_per_bus = 18.0) 
  (h_extra_people_in_buses : extra_people_in_buses = 108.0) 
  (h_eq : people_per_bus * buses = vans * people_per_van + extra_people_in_buses) : 
  buses = 8.0 :=
by
  sorry

end number_of_buses_l1106_110653


namespace cost_price_of_one_toy_l1106_110682

theorem cost_price_of_one_toy (C : ℝ) (h : 21 * C = 21000) : C = 1000 :=
by sorry

end cost_price_of_one_toy_l1106_110682


namespace leak_empties_tank_in_8_hours_l1106_110602

theorem leak_empties_tank_in_8_hours (capacity : ℕ) (inlet_rate_per_minute : ℕ) (time_with_inlet_open : ℕ) (time_without_inlet_open : ℕ) : 
  capacity = 8640 ∧ inlet_rate_per_minute = 6 ∧ time_with_inlet_open = 12 ∧ time_without_inlet_open = 8 := 
by 
  sorry

end leak_empties_tank_in_8_hours_l1106_110602


namespace largest_number_is_870_l1106_110673

-- Define the set of digits {8, 7, 0}
def digits : Set ℕ := {8, 7, 0}

-- Define the largest number that can be made by arranging these digits
def largest_number (s : Set ℕ) : ℕ := 870

-- Statement to prove
theorem largest_number_is_870 : largest_number digits = 870 :=
by
  -- Proof is omitted
  sorry

end largest_number_is_870_l1106_110673


namespace function_properties_l1106_110667

theorem function_properties
  (f : ℝ → ℝ)
  (h1 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : ∀ x, f (x - t) = f (x + t)) 
  (h3_even : ∀ x, f (-x) = f x)
  (h3_decreasing : ∀ x1 x2, x1 < x2 ∧ x2 < 0 → f x1 > f x2)
  (h3_at_neg2 : f (-2) = 0)
  (h4_odd : ∀ x, f (-x) = -f x) : 
  ((∀ x1 x2, x1 < x2 → f x1 > f x2) ∧
   (¬∀ x, (f x > 0) ↔ (-2 < x ∧ x < 2)) ∧
   (∀ x, f (x) * f (|x|) = - f (-x) * f |x|) ∧
   (¬∀ x, f (x) = f (x + 2 * t))) :=
by 
  sorry

end function_properties_l1106_110667


namespace ratio_of_boys_to_girls_l1106_110625

-- Variables for the number of boys, girls, and teachers
variables (B G T : ℕ)

-- Conditions from the problem
def number_of_girls := G = 60
def number_of_teachers := T = (20 * B) / 100
def total_people := B + G + T = 114

-- Proving the ratio of boys to girls is 3:4 given the conditions
theorem ratio_of_boys_to_girls 
  (hG : number_of_girls G)
  (hT : number_of_teachers B T)
  (hTotal : total_people B G T) :
  B / 15 = 3 ∧ G / 15 = 4 :=
by {
  sorry
}

end ratio_of_boys_to_girls_l1106_110625


namespace Kevin_lost_cards_l1106_110674

theorem Kevin_lost_cards (initial_cards final_cards : ℝ) (h1 : initial_cards = 47.0) (h2 : final_cards = 40) :
  initial_cards - final_cards = 7 :=
by
  sorry

end Kevin_lost_cards_l1106_110674


namespace ratio_of_gilled_to_spotted_l1106_110671

theorem ratio_of_gilled_to_spotted (total_mushrooms gilled_mushrooms spotted_mushrooms : ℕ) 
  (h1 : total_mushrooms = 30) 
  (h2 : gilled_mushrooms = 3) 
  (h3 : spotted_mushrooms = total_mushrooms - gilled_mushrooms) :
  gilled_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 1 ∧ 
  spotted_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 9 := 
by
  sorry

end ratio_of_gilled_to_spotted_l1106_110671


namespace second_workshop_production_l1106_110678

theorem second_workshop_production (a b c : ℕ) (h₁ : a + b + c = 3600) (h₂ : a + c = 2 * b) : b * 3 = 3600 := 
by 
  sorry

end second_workshop_production_l1106_110678


namespace expected_sample_size_l1106_110697

noncomputable def highSchoolTotalStudents (f s j : ℕ) : ℕ :=
  f + s + j

noncomputable def expectedSampleSize (total : ℕ) (p : ℝ) : ℝ :=
  total * p

theorem expected_sample_size :
  let f := 400
  let s := 320
  let j := 280
  let p := 0.2
  let total := highSchoolTotalStudents f s j
  expectedSampleSize total p = 200 :=
by
  sorry

end expected_sample_size_l1106_110697


namespace bc_possible_values_l1106_110656

theorem bc_possible_values (a b c : ℝ) 
  (h1 : a + b + c = 100) 
  (h2 : ab + bc + ca = 20) 
  (h3 : (a + b) * (a + c) = 24) : 
  bc = -176 ∨ bc = 224 :=
by
  sorry

end bc_possible_values_l1106_110656


namespace ratio_sum_ineq_l1106_110696

theorem ratio_sum_ineq 
  (a b α β : ℝ) 
  (hαβ : 0 < α ∧ 0 < β) 
  (h_range : α ≤ a ∧ a ≤ β ∧ α ≤ b ∧ b ≤ β) : 
  (b / a + a / b ≤ β / α + α / β) ∧ 
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β ∨ a = β ∧ b = α)) :=
by
  sorry

end ratio_sum_ineq_l1106_110696


namespace binom_8_3_eq_56_and_2_pow_56_l1106_110654

theorem binom_8_3_eq_56_and_2_pow_56 :
  (Nat.choose 8 3 = 56) ∧ (2 ^ (Nat.choose 8 3) = 2 ^ 56) :=
by
  sorry

end binom_8_3_eq_56_and_2_pow_56_l1106_110654


namespace sally_money_l1106_110600

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l1106_110600


namespace alexander_eq_alice_l1106_110683

-- Definitions and conditions
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.07

-- Calculation functions for Alexander and Alice
def alexander_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let taxed_price := price * (1 + tax)
  let discounted_price := taxed_price * (1 - discount)
  discounted_price

def alice_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

-- Proof that the difference between Alexander's and Alice's total is 0
theorem alexander_eq_alice : 
  alexander_total original_price discount_rate sales_tax_rate = 
  alice_total original_price discount_rate sales_tax_rate :=
by
  sorry

end alexander_eq_alice_l1106_110683


namespace volume_parallelepiped_l1106_110668

open Real

theorem volume_parallelepiped :
  ∃ (a h : ℝ), 
    let S_base := (4 : ℝ)
    let AB := a
    let AD := 2 * a
    let lateral_face1 := (6 : ℝ)
    let lateral_face2 := (12 : ℝ)
    (AB * h = lateral_face1) ∧
    (AD * h = lateral_face2) ∧
    (1 / 2 * AD * S_base = AB * (1 / 2 * AD)) ∧ 
    (AB^2 + AD^2 - 2 * AB * AD * (cos (π / 6)) = S_base) ∧
    (a = 2) ∧
    (h = 3) ∧ 
    (S_base * h = 12) :=
sorry

end volume_parallelepiped_l1106_110668


namespace average_income_B_and_C_l1106_110691

variables (A_income B_income C_income : ℝ)

noncomputable def average_monthly_income_B_and_C (A_income : ℝ) :=
  (B_income + C_income) / 2

theorem average_income_B_and_C
  (h1 : (A_income + B_income) / 2 = 5050)
  (h2 : (A_income + C_income) / 2 = 5200)
  (h3 : A_income = 4000) :
  average_monthly_income_B_and_C 4000 = 6250 :=
by
  sorry

end average_income_B_and_C_l1106_110691


namespace last_digit_of_expression_l1106_110627

-- Conditions
def a : ℤ := 25
def b : ℤ := -3

-- Statement to be proved
theorem last_digit_of_expression :
  (a ^ 1999 + b ^ 2002) % 10 = 4 :=
by
  -- proof would go here
  sorry

end last_digit_of_expression_l1106_110627


namespace width_to_length_ratio_l1106_110635

variables {w l P : ℕ}

theorem width_to_length_ratio :
  l = 10 → P = 30 → P = 2 * (l + w) → (w : ℚ) / l = 1 / 2 :=
by
  intro h1 h2 h3
  -- Noncomputable definition for rational division
  -- (ℚ is used for exact rational division)
  sorry

#check width_to_length_ratio

end width_to_length_ratio_l1106_110635


namespace Bert_sandwiches_left_l1106_110606

theorem Bert_sandwiches_left : (Bert:Type) → 
  (sandwiches_made : ℕ) → 
  sandwiches_made = 12 → 
  (sandwiches_eaten_day1 : ℕ) → 
  sandwiches_eaten_day1 = sandwiches_made / 2 → 
  (sandwiches_eaten_day2 : ℕ) → 
  sandwiches_eaten_day2 = sandwiches_eaten_day1 - 2 →
  (sandwiches_left : ℕ) → 
  sandwiches_left = sandwiches_made - (sandwiches_eaten_day1 + sandwiches_eaten_day2) → 
  sandwiches_left = 2 := 
  sorry

end Bert_sandwiches_left_l1106_110606


namespace two_positive_roots_condition_l1106_110615

theorem two_positive_roots_condition (a : ℝ) :
  (1 < a ∧ a ≤ 2) ∨ (a ≥ 10) ↔
  ∃ x1 x2 : ℝ, (1-a) * x1^2 + (a+2) * x1 - 4 = 0 ∧ 
               (1-a) * x2^2 + (a+2) * x2 - 4 = 0 ∧ 
               x1 > 0 ∧ x2 > 0 :=
sorry

end two_positive_roots_condition_l1106_110615


namespace quadratic_y1_gt_y2_l1106_110680

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end quadratic_y1_gt_y2_l1106_110680


namespace choose_officers_from_six_l1106_110601

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l1106_110601


namespace M_inter_N_empty_l1106_110694

def M : Set ℝ := {a : ℝ | (1 / 2 < a ∧ a < 1) ∨ (1 < a)}
def N : Set ℝ := {a : ℝ | 0 < a ∧ a ≤ 1 / 2}

theorem M_inter_N_empty : M ∩ N = ∅ :=
sorry

end M_inter_N_empty_l1106_110694


namespace percent_daisies_l1106_110641

theorem percent_daisies 
    (total_flowers : ℕ)
    (yellow_flowers : ℕ)
    (yellow_tulips : ℕ)
    (blue_flowers : ℕ)
    (blue_daisies : ℕ)
    (h1 : 2 * yellow_tulips = yellow_flowers) 
    (h2 : 3 * blue_daisies = blue_flowers)
    (h3 : 10 * yellow_flowers = 7 * total_flowers) : 
    100 * (yellow_flowers / 2 + blue_daisies) = 45 * total_flowers :=
by
  sorry

end percent_daisies_l1106_110641


namespace circle_area_from_points_l1106_110650

theorem circle_area_from_points (C D : ℝ × ℝ) (hC : C = (2, 3)) (hD : D = (8, 9)) : 
  ∃ A : ℝ, A = 18 * Real.pi :=
by
  sorry

end circle_area_from_points_l1106_110650


namespace age_difference_l1106_110637

variable (A B C D : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 16) : (A + B) - (B + C) = 16 :=
by
  sorry

end age_difference_l1106_110637


namespace scienceStudyTime_l1106_110636

def totalStudyTime : ℕ := 60
def mathStudyTime : ℕ := 35

theorem scienceStudyTime : totalStudyTime - mathStudyTime = 25 :=
by sorry

end scienceStudyTime_l1106_110636


namespace total_chewing_gums_l1106_110660

-- Definitions for the conditions
def mary_gums : Nat := 5
def sam_gums : Nat := 10
def sue_gums : Nat := 15

-- Lean 4 Theorem statement to prove the total chewing gums
theorem total_chewing_gums : mary_gums + sam_gums + sue_gums = 30 := by
  sorry

end total_chewing_gums_l1106_110660


namespace harvest_unripe_oranges_l1106_110619

theorem harvest_unripe_oranges (R T D U: ℕ) (h1: R = 28) (h2: T = 2080) (h3: D = 26)
  (h4: T = D * (R + U)) :
  U = 52 :=
by
  sorry

end harvest_unripe_oranges_l1106_110619


namespace tangency_condition_l1106_110685

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 3)^2 = 4

theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m → x^2 = 9 - 9 * y^2 ∧ x^2 = 4 + m * (y + 3)^2 → ((m - 9) * y^2 + 6 * m * y + (9 * m - 5) = 0 → 36 * m^2 - 4 * (m - 9) * (9 * m - 5) = 0 ) ) → 
  m = 5 / 54 :=
by
  sorry

end tangency_condition_l1106_110685


namespace surface_area_of_figure_l1106_110609

theorem surface_area_of_figure 
  (block_surface_area : ℕ) 
  (loss_per_block : ℕ) 
  (number_of_blocks : ℕ) 
  (effective_surface_area : ℕ)
  (total_surface_area : ℕ) 
  (h_block : block_surface_area = 18) 
  (h_loss : loss_per_block = 2) 
  (h_blocks : number_of_blocks = 4) 
  (h_effective : effective_surface_area = block_surface_area - loss_per_block) 
  (h_total : total_surface_area = number_of_blocks * effective_surface_area) : 
  total_surface_area = 64 :=
by
  sorry

end surface_area_of_figure_l1106_110609


namespace find_m_for_parallel_lines_l1106_110611

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 6 * x + m * y - 1 = 0 ↔ 2 * x - y + 1 = 0) → m = -3 :=
by
  sorry

end find_m_for_parallel_lines_l1106_110611


namespace maize_donation_amount_l1106_110658

-- Definitions and Conditions
def monthly_storage : ℕ := 1
def months_in_year : ℕ := 12
def years : ℕ := 2
def stolen_tonnes : ℕ := 5
def total_tonnes_at_end : ℕ := 27

-- Theorem statement
theorem maize_donation_amount :
  let total_stored := monthly_storage * (months_in_year * years)
  let remaining_after_theft := total_stored - stolen_tonnes
  total_tonnes_at_end - remaining_after_theft = 8 :=
by
  -- This part is just the statement, hence we use sorry to omit the proof.
  sorry

end maize_donation_amount_l1106_110658


namespace determine_a_l1106_110628

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x ^ 2 + a else 2 ^ x

theorem determine_a (a : ℝ) (h1 : a > -1) (h2 : f a (f a (-1)) = 4) : a = 1 :=
sorry

end determine_a_l1106_110628


namespace geometric_N_digit_not_20_l1106_110690

-- Variables and definitions
variables (a b c : ℕ)

-- Given conditions
def geometric_progression (a b c : ℕ) : Prop :=
  ∃ q : ℚ, (b = q * a) ∧ (c = q * b)

def ends_with_20 (N : ℕ) : Prop := N % 100 = 20

-- Prove the main theorem
theorem geometric_N_digit_not_20 (h1 : geometric_progression a b c) (h2 : ends_with_20 (a^3 + b^3 + c^3 - 3 * a * b * c)) :
  False :=
sorry

end geometric_N_digit_not_20_l1106_110690


namespace distributive_property_example_l1106_110630

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = (3/4) * (-36) + (7/12) * (-36) - (5/9) * (-36) :=
by
  sorry

end distributive_property_example_l1106_110630


namespace max_min_M_l1106_110657

noncomputable def M (x y : ℝ) : ℝ :=
  abs (x + y) + abs (y + 1) + abs (2 * y - x - 4)

theorem max_min_M (x y : ℝ) (hx : abs x ≤ 1) (hy : abs y ≤ 1) :
  3 ≤ M x y ∧ M x y ≤ 7 :=
sorry

end max_min_M_l1106_110657


namespace no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l1106_110688
-- Bringing in the entirety of Mathlib

-- Problem (a): There are no non-zero integers that increase by 7 or 9 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_7_or_9 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ (10 * X + d = 7 * n ∨ 10 * X + d = 9 * n)) :=
by sorry

-- Problem (b): There are no non-zero integers that increase by 4 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_4 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ 10 * X + d = 4 * n) :=
by sorry

end no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l1106_110688


namespace like_terms_exponents_product_l1106_110643

theorem like_terms_exponents_product (m n : ℤ) (a b : ℝ) 
  (h1 : 3 * a^m * b^2 = -1 * a^2 * b^(n+3)) : m * n = -2 :=
  sorry

end like_terms_exponents_product_l1106_110643


namespace T_shaped_area_l1106_110670

theorem T_shaped_area (a b c d : ℕ) (side1 side2 side3 large_side : ℕ)
  (h_side1: side1 = 2)
  (h_side2: side2 = 2)
  (h_side3: side3 = 4)
  (h_large_side: large_side = 6)
  (h_area_large_square : a = large_side * large_side)
  (h_area_square1 : b = side1 * side1)
  (h_area_square2 : c = side2 * side2)
  (h_area_square3 : d = side3 * side3) :
  a - (b + c + d) = 12 := by
  sorry

end T_shaped_area_l1106_110670


namespace points_on_hyperbola_l1106_110689

theorem points_on_hyperbola {s : ℝ} :
  let x := Real.exp s - Real.exp (-s)
  let y := 5 * (Real.exp s + Real.exp (-s))
  (y^2 / 100 - x^2 / 4 = 1) :=
by
  sorry

end points_on_hyperbola_l1106_110689


namespace complement_of_A_is_correct_l1106_110655

-- Define the universal set U and the set A.
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U.
def A_complement : Set ℕ := {x ∈ U | x ∉ A}

-- The theorem statement that the complement of A in U is {2, 4}.
theorem complement_of_A_is_correct : A_complement = {2, 4} :=
sorry

end complement_of_A_is_correct_l1106_110655


namespace seashells_after_giving_away_l1106_110613

-- Define the given conditions
def initial_seashells : ℕ := 79
def given_away_seashells : ℕ := 63

-- State the proof problem
theorem seashells_after_giving_away : (initial_seashells - given_away_seashells) = 16 :=
  by 
    sorry

end seashells_after_giving_away_l1106_110613


namespace choose_students_l1106_110648

/-- There are 50 students in the class, including one class president and one vice-president. 
    We want to select 5 students to participate in an activity such that at least one of 
    the class president or vice-president is included. We assert that there are exactly 2 
    distinct methods for making this selection. -/
theorem choose_students (students : Finset ℕ) (class_president vice_president : ℕ) (students_card : students.card = 50)
  (students_ex : class_president ∈ students ∧ vice_president ∈ students) : 
  ∃ valid_methods : Finset (Finset ℕ), valid_methods.card = 2 :=
by
  sorry

end choose_students_l1106_110648


namespace expected_sufferers_l1106_110623

theorem expected_sufferers 
  (fraction_condition : ℚ := 1 / 4)
  (sample_size : ℕ := 400) 
  (expected_number : ℕ := 100) : 
  fraction_condition * sample_size = expected_number := 
by 
  sorry

end expected_sufferers_l1106_110623


namespace solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l1106_110634

-- Problem 1: Solution Set of the Inequality
theorem solution_set_x2_minus_5x_plus_4 : 
  {x : ℝ | x^2 - 5 * x + 4 > 0} = {x : ℝ | x < 1 ∨ x > 4} :=
sorry

-- Problem 2: Range of Values for a
theorem range_of_a_if_x2_plus_ax_plus_4_gt_0 (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 4 > 0) :
  -4 < a ∧ a < 4 :=
sorry

end solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l1106_110634


namespace check_conditions_l1106_110640

noncomputable def f (x a b : ℝ) : ℝ := |x^2 - 2 * a * x + b|

theorem check_conditions (a b : ℝ) :
  ¬ (∀ x : ℝ, f x a b = f (-x) a b) ∧         -- f(x) is not necessarily an even function
  ¬ (∀ x : ℝ, (f 0 a b = f 2 a b → (f x a b = f (2 - x) a b))) ∧ -- No guaranteed symmetry about x=1
  (a^2 - b^2 ≤ 0 → ∀ x : ℝ, x ≥ a → ∀ y : ℝ, y ≥ x → f y a b ≥ f x a b) ∧ -- f(x) is increasing on [a, +∞) if a^2 - b^2 ≤ 0
  ¬ (∀ x : ℝ, f x a b ≤ |a^2 - b|)         -- f(x) does not necessarily have a max value of |a^2 - b|
:= sorry

end check_conditions_l1106_110640


namespace track_width_l1106_110693

theorem track_width (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 10 * π) : r1 - r2 = 5 :=
sorry

end track_width_l1106_110693


namespace factor_poly_l1106_110638

theorem factor_poly (x : ℤ) :
  (x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1)) :=
by
  sorry

end factor_poly_l1106_110638


namespace find_third_number_l1106_110675

-- Definitions and conditions for the problem
def x : ℚ := 1.35
def third_number := 5
def proportion (a b c d : ℚ) := a * d = b * c 

-- Proposition to prove
theorem find_third_number : proportion 0.75 x third_number 9 := 
by
  -- It's advisable to split the proof steps here, but the proof itself is condensed.
  sorry

end find_third_number_l1106_110675


namespace rhombus_difference_l1106_110621

theorem rhombus_difference (n : ℕ) (h : n > 3)
    (m : ℕ := 3 * (n - 1) * n / 2)
    (d : ℕ := 3 * (n - 3) * (n - 2) / 2) :
    m - d = 6 * n - 9 := by {
  -- Proof omitted
  sorry
}

end rhombus_difference_l1106_110621


namespace chocolate_cost_l1106_110698

theorem chocolate_cost (Ccb Cc : ℝ) (h1 : Ccb = 6) (h2 : Ccb = Cc + 3) : Cc = 3 :=
by
  sorry

end chocolate_cost_l1106_110698


namespace solve_equation_l1106_110677

theorem solve_equation (x : ℝ) : (x + 2) * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 1) :=
by sorry

end solve_equation_l1106_110677


namespace elevator_travel_time_l1106_110664

noncomputable def total_time_in_hours (floors : ℕ) (time_first_half : ℕ) (time_next_floors_per_floor : ℕ) (next_floors : ℕ) (time_final_floors_per_floor : ℕ) (final_floors : ℕ) : ℕ :=
  let time_first_part := time_first_half
  let time_next_part := time_next_floors_per_floor * next_floors
  let time_final_part := time_final_floors_per_floor * final_floors
  (time_first_part + time_next_part + time_final_part) / 60

theorem elevator_travel_time :
  total_time_in_hours 20 15 5 5 16 5 = 2 := 
by
  sorry

end elevator_travel_time_l1106_110664


namespace triangle_incenter_equilateral_l1106_110620

theorem triangle_incenter_equilateral (a b c : ℝ) (h : (b + c) / a = (a + c) / b ∧ (a + c) / b = (a + b) / c) : a = b ∧ b = c :=
by
  sorry

end triangle_incenter_equilateral_l1106_110620


namespace vertex_of_parabola_l1106_110676

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l1106_110676


namespace range_of_a_l1106_110695

noncomputable def isNotPurelyImaginary (a : ℝ) : Prop :=
  let re := a^2 - a - 2
  re ≠ 0

theorem range_of_a (a : ℝ) (h : isNotPurelyImaginary a) : a ≠ -1 :=
  sorry

end range_of_a_l1106_110695


namespace sine_product_identity_l1106_110618

open Real

theorem sine_product_identity :
  sin 12 * sin 36 * sin 54 * sin 72 = 1 / 16 := by
  have h1 : sin 72 = cos 18 := by sorry
  have h2 : sin 54 = cos 36 := by sorry
  have h3 : ∀ θ, sin θ * cos θ = 1 / 2 * sin (2 * θ) := by sorry
  have h4 : ∀ θ, cos (2 * θ) = 2 * cos θ ^ 2 - 1 := by sorry
  have h5 : cos 36 = 1 - 2 * (sin 18) ^ 2 := by sorry
  have h6 : ∀ θ, sin (180 - θ) = sin θ := by sorry
  sorry

end sine_product_identity_l1106_110618


namespace find_ab_from_conditions_l1106_110679

theorem find_ab_from_conditions (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := 
by
  sorry

end find_ab_from_conditions_l1106_110679


namespace quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l1106_110649

open Real

-- Mathematical translations of conditions and proofs
theorem quadratic_real_roots_range_of_m (m : ℝ) (h1 : ∃ x : ℝ, x^2 + 2 * x - (m - 2) = 0) :
  m ≥ 1 := by
  sorry

theorem quadratic_root_and_other_m (h1 : (1:ℝ) ^ 2 + 2 * 1 - (m - 2) = 0) :
  m = 3 ∧ ∃ x : ℝ, (x = -3) ∧ (x^2 + 2 * x - 3 = 0) := by
  sorry

end quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l1106_110649


namespace find_a_l1106_110604

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 * Real.exp x

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 1 → (x - a) * (x - a + 2) ≤ 0) → a = 1 :=
by
  intro h
  sorry 

end find_a_l1106_110604


namespace M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l1106_110645

-- Definition of the curve using parametric equations
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t, 2 * t^2 + 1)

-- Questions and proof statements
theorem M1_on_curve_C : ∃ t : ℝ, curve t = (0, 1) :=
by { 
  sorry 
}

theorem M2_not_on_curve_C : ¬ (∃ t : ℝ, curve t = (5, 4)) :=
by { 
  sorry 
}

theorem M3_on_curve_C_a_eq_9 (a : ℝ) : (∃ t : ℝ, curve t = (6, a)) → a = 9 :=
by { 
  sorry 
}

end M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l1106_110645


namespace baron_munchausen_incorrect_l1106_110639

theorem baron_munchausen_incorrect : 
  ∀ (n : ℕ) (ab : ℕ), 10 ≤ n → n ≤ 99 → 0 ≤ ab → ab ≤ 99 
  → ¬ (∃ (m : ℕ), n * 100 + ab = m * m) := 
by
  intros n ab n_lower_bound n_upper_bound ab_lower_bound ab_upper_bound
  sorry

end baron_munchausen_incorrect_l1106_110639


namespace remaining_nap_time_is_three_hours_l1106_110617

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end remaining_nap_time_is_three_hours_l1106_110617


namespace prob_neither_A_nor_B_l1106_110632

theorem prob_neither_A_nor_B
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ)
  (h1 : P_A = 0.25) (h2 : P_B = 0.30) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.60 :=
by
  sorry

end prob_neither_A_nor_B_l1106_110632


namespace books_per_shelf_l1106_110622

theorem books_per_shelf :
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  remaining_books / shelves = 3 :=
by
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  have h1 : remaining_books = 12 := by simp [remaining_books]
  have h2 : remaining_books / shelves = 3 := by norm_num [remaining_books, shelves]
  exact h2

end books_per_shelf_l1106_110622


namespace cost_per_top_l1106_110612
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end cost_per_top_l1106_110612


namespace two_talents_students_l1106_110692

-- Definitions and conditions
def total_students : ℕ := 120
def cannot_sing : ℕ := 50
def cannot_dance : ℕ := 75
def cannot_act : ℕ := 35

-- Definitions based on conditions
def can_sing : ℕ := total_students - cannot_sing
def can_dance : ℕ := total_students - cannot_dance
def can_act : ℕ := total_students - cannot_act

-- The main theorem statement
theorem two_talents_students : can_sing + can_dance + can_act - total_students = 80 :=
by
  -- substituting actual numbers to prove directly
  have h_can_sing : can_sing = 70 := rfl
  have h_can_dance : can_dance = 45 := rfl
  have h_can_act : can_act = 85 := rfl
  sorry

end two_talents_students_l1106_110692


namespace inequality_proof_l1106_110642

open Real

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l1106_110642


namespace common_ratio_of_geometric_sequence_l1106_110666

theorem common_ratio_of_geometric_sequence (a_1 a_2 a_3 a_4 q : ℝ)
  (h1 : a_1 * a_2 * a_3 = 27)
  (h2 : a_2 + a_4 = 30)
  (geometric_sequence : a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3) :
  q = 3 ∨ q = -3 :=
sorry

end common_ratio_of_geometric_sequence_l1106_110666


namespace cos_of_angle_in_third_quadrant_l1106_110644

theorem cos_of_angle_in_third_quadrant (A : ℝ) (hA : π < A ∧ A < 3 * π / 2) (h_sin : Real.sin A = -1 / 3) :
  Real.cos A = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_of_angle_in_third_quadrant_l1106_110644


namespace main_problem_proof_l1106_110661

def main_problem : Prop :=
  (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2

theorem main_problem_proof : main_problem :=
by {
  sorry
}

end main_problem_proof_l1106_110661


namespace prob_rain_both_days_l1106_110646

-- Declare the probabilities involved
def P_Monday : ℝ := 0.40
def P_Tuesday : ℝ := 0.30
def P_Tuesday_given_Monday : ℝ := 0.30

-- Prove the probability of it raining on both days
theorem prob_rain_both_days : P_Monday * P_Tuesday_given_Monday = 0.12 :=
by
  sorry

end prob_rain_both_days_l1106_110646


namespace rectangle_length_l1106_110662

theorem rectangle_length {width length : ℝ} (h1 : (3 : ℝ) * 3 = 9) (h2 : width = 3) (h3 : width * length = 9) : 
  length = 3 :=
by
  sorry

end rectangle_length_l1106_110662


namespace turban_price_l1106_110631

theorem turban_price (T : ℝ) (total_salary : ℝ) (received_salary : ℝ)
  (cond1 : total_salary = 90 + T)
  (cond2 : received_salary = 65 + T)
  (cond3 : received_salary = (3 / 4) * total_salary) :
  T = 10 :=
by
  sorry

end turban_price_l1106_110631


namespace find_k_l1106_110672

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem find_k (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 :=
by
  sorry

end find_k_l1106_110672


namespace min_neg_condition_l1106_110603

theorem min_neg_condition (a : ℝ) (x : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) → a < -7 :=
sorry

end min_neg_condition_l1106_110603


namespace gcd_884_1071_l1106_110669

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end gcd_884_1071_l1106_110669


namespace expression_equals_36_l1106_110665

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l1106_110665


namespace man_walking_time_l1106_110647

section TrainProblem

variables {T W : ℕ}

/-- Each day a man meets his wife at the train station after work,
    and then she drives him home. She always arrives exactly on time to pick him up.
    One day he catches an earlier train and arrives at the station an hour early.
    He immediately begins walking home along the same route the wife drives.
    Eventually, his wife sees him on her way to the station and drives him the rest of the way home.
    When they arrive home, the man notices that they arrived 30 minutes earlier than usual.
    How much time did the man spend walking? -/
theorem man_walking_time : 
    (∃ (T : ℕ), T > 30 ∧ (W = T - 30) ∧ (W + 30 = T)) → W = 30 :=
sorry

end TrainProblem

end man_walking_time_l1106_110647


namespace find_S10_value_l1106_110610

noncomputable def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, 4 * S n = n * (a n + a (n + 1))

theorem find_S10_value (a S : ℕ → ℕ) (h1 : a 4 = 7) (h2 : sequence_sum a S) :
  S 10 = 100 :=
sorry

end find_S10_value_l1106_110610


namespace converse_l1106_110684

theorem converse (x y : ℝ) (h : x + y ≥ 5) : x ≥ 2 ∧ y ≥ 3 := 
sorry

end converse_l1106_110684


namespace ceil_floor_subtraction_l1106_110624

theorem ceil_floor_subtraction :
  ⌈(7:ℝ) / 3⌉ + ⌊- (7:ℝ) / 3⌋ - 3 = -3 := 
by
  sorry   -- Placeholder for the proof

end ceil_floor_subtraction_l1106_110624


namespace problem_1_problem_2_l1106_110614

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem problem_1 (h₁ : ∀ x, x > 0 → x ≠ 1 → f x = x / Real.log x) :
  (∀ x, 1 < x ∧ x < Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) ∧
  (∀ x, x > Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) :=
sorry

theorem problem_2 (h₁ : f x₁ = 1) (h₂ : f x₂ = 1) (h₃ : x₁ ≠ x₂) (h₄ : x₁ > 0) (h₅ : x₂ > 0):
  x₁ + x₂ > 2 * Real.exp 1 :=
sorry

end problem_1_problem_2_l1106_110614


namespace distance_traveled_by_second_hand_l1106_110607

def second_hand_length : ℝ := 8
def time_period_minutes : ℝ := 45
def rotations_per_minute : ℝ := 1

theorem distance_traveled_by_second_hand :
  let circumference := 2 * Real.pi * second_hand_length
  let rotations := time_period_minutes * rotations_per_minute
  let total_distance := rotations * circumference
  total_distance = 720 * Real.pi := by
  sorry

end distance_traveled_by_second_hand_l1106_110607
