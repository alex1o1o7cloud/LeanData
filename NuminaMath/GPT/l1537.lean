import Mathlib

namespace NUMINAMATH_GPT_volunteer_org_percentage_change_l1537_153772

theorem volunteer_org_percentage_change :
  ∀ (X : ℝ), X > 0 → 
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  (X - spring_decrease) / X * 100 = 11.71 :=
by
  intro X hX
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  show (_ - _) / _ * _ = _
  sorry

end NUMINAMATH_GPT_volunteer_org_percentage_change_l1537_153772


namespace NUMINAMATH_GPT_volleyball_team_l1537_153727

theorem volleyball_team :
  let total_combinations := (Nat.choose 15 6)
  let without_triplets := (Nat.choose 12 6)
  total_combinations - without_triplets = 4081 :=
by
  -- Definitions based on the problem conditions
  let team_size := 15
  let starters := 6
  let triplets := 3
  let total_combinations := Nat.choose team_size starters
  let without_triplets := Nat.choose (team_size - triplets) starters
  -- Identify the proof goal
  have h : total_combinations - without_triplets = 4081 := sorry
  exact h

end NUMINAMATH_GPT_volleyball_team_l1537_153727


namespace NUMINAMATH_GPT_johns_number_is_thirteen_l1537_153764

theorem johns_number_is_thirteen (x : ℕ) (h1 : 10 ≤ x) (h2 : x < 100) (h3 : ∃ a b : ℕ, 10 * a + b = 4 * x + 17 ∧ 92 ≤ 10 * b + a ∧ 10 * b + a ≤ 96) : x = 13 :=
sorry

end NUMINAMATH_GPT_johns_number_is_thirteen_l1537_153764


namespace NUMINAMATH_GPT_highland_park_science_fair_l1537_153784

noncomputable def juniors_and_seniors_participants (j s : ℕ) : ℕ :=
  (3 * j) / 4 + s / 2

theorem highland_park_science_fair 
  (j s : ℕ)
  (h1 : (3 * j) / 4 = s / 2)
  (h2 : j + s = 240) :
  juniors_and_seniors_participants j s = 144 := by
  sorry

end NUMINAMATH_GPT_highland_park_science_fair_l1537_153784


namespace NUMINAMATH_GPT_intersection_complement_l1537_153787

def M (x : ℝ) : Prop := x^2 - 2 * x < 0
def N (x : ℝ) : Prop := x < 1

theorem intersection_complement (x : ℝ) :
  (M x ∧ ¬N x) ↔ (1 ≤ x ∧ x < 2) := 
sorry

end NUMINAMATH_GPT_intersection_complement_l1537_153787


namespace NUMINAMATH_GPT_solve_for_A_l1537_153746

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l1537_153746


namespace NUMINAMATH_GPT_min_sum_m_n_l1537_153719

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem min_sum_m_n (m n : ℕ) (h : (binomial m 2) * 2 = binomial (m + n) 2) : m + n = 4 := by
  sorry

end NUMINAMATH_GPT_min_sum_m_n_l1537_153719


namespace NUMINAMATH_GPT_true_propositions_count_l1537_153741

theorem true_propositions_count {a b c : ℝ} (h : a ≤ b) : 
  (if (c^2 ≥ 0 ∧ a * c^2 ≤ b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ a * c^2 > b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a * c^2 ≤ b * c^2) → ¬(a ≤ b)) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a ≤ b) → ¬(a * c^2 ≤ b * c^2)) then 1 else 0) = 2 :=
sorry

end NUMINAMATH_GPT_true_propositions_count_l1537_153741


namespace NUMINAMATH_GPT_algebraic_expression_value_l1537_153749

theorem algebraic_expression_value (p q : ℝ)
  (h : p * 3^3 + q * 3 + 3 = 2005) :
  p * (-3)^3 + q * (-3) + 3 = -1999 :=
by
   sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1537_153749


namespace NUMINAMATH_GPT_solve_for_x_l1537_153792

theorem solve_for_x (x y : ℕ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1537_153792


namespace NUMINAMATH_GPT_convert_polar_to_rectangular_l1537_153702

noncomputable def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  polarToRectangular 8 (7 * Real.pi / 6) = (-4 * Real.sqrt 3, -4) :=
by
  sorry

end NUMINAMATH_GPT_convert_polar_to_rectangular_l1537_153702


namespace NUMINAMATH_GPT_book_cost_l1537_153725

theorem book_cost (b : ℝ) : (11 * b < 15) ∧ (12 * b > 16.20) → b = 1.36 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_book_cost_l1537_153725


namespace NUMINAMATH_GPT_largest_number_is_310_l1537_153752

def largest_number_formed (a b c : ℕ) : ℕ :=
  max (a * 100 + b * 10 + c) (max (a * 100 + c * 10 + b) (max (b * 100 + a * 10 + c) 
  (max (b * 100 + c * 10 + a) (max (c * 100 + a * 10 + b) (c * 100 + b * 10 + a)))))

theorem largest_number_is_310 : largest_number_formed 3 1 0 = 310 :=
by simp [largest_number_formed]; sorry

end NUMINAMATH_GPT_largest_number_is_310_l1537_153752


namespace NUMINAMATH_GPT_rational_linear_independent_sqrt_prime_l1537_153734

theorem rational_linear_independent_sqrt_prime (p : ℕ) (hp : Nat.Prime p) (m n m1 n1 : ℚ) :
  m + n * Real.sqrt p = m1 + n1 * Real.sqrt p → m = m1 ∧ n = n1 :=
sorry

end NUMINAMATH_GPT_rational_linear_independent_sqrt_prime_l1537_153734


namespace NUMINAMATH_GPT_find_third_term_l1537_153701

theorem find_third_term :
  ∃ (a : ℕ → ℝ), a 0 = 5 ∧ a 4 = 2025 ∧ (∀ n, a (n + 1) = a n * r) ∧ a 2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_find_third_term_l1537_153701


namespace NUMINAMATH_GPT_satisfying_integers_l1537_153755

theorem satisfying_integers (a b : ℤ) :
  a^4 + (a + b)^4 + b^4 = x^2 → a = 0 ∧ b = 0 :=
by
  -- Proof is required to be filled in here.
  sorry

end NUMINAMATH_GPT_satisfying_integers_l1537_153755


namespace NUMINAMATH_GPT_find_f2_l1537_153733

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x + 1

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l1537_153733


namespace NUMINAMATH_GPT_max_value_xyz_l1537_153713

theorem max_value_xyz 
  (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 3) : 
  ∃ M, M = 243 ∧ (x + y^4 + z^5) ≤ M := 
  by sorry

end NUMINAMATH_GPT_max_value_xyz_l1537_153713


namespace NUMINAMATH_GPT_caleb_apples_less_than_kayla_l1537_153732

theorem caleb_apples_less_than_kayla :
  ∀ (Kayla Suraya Caleb : ℕ),
  (Kayla = 20) →
  (Suraya = Kayla + 7) →
  (Suraya = Caleb + 12) →
  (Suraya = 27) →
  (Kayla - Caleb = 5) :=
by
  intros Kayla Suraya Caleb hKayla hSuraya1 hSuraya2 hSuraya3
  sorry

end NUMINAMATH_GPT_caleb_apples_less_than_kayla_l1537_153732


namespace NUMINAMATH_GPT_original_selling_price_l1537_153747

theorem original_selling_price (P : ℝ) (S : ℝ) (h1 : S = 1.10 * P) (h2 : 1.17 * P = 1.10 * P + 35) : S = 550 := 
by
  sorry

end NUMINAMATH_GPT_original_selling_price_l1537_153747


namespace NUMINAMATH_GPT_average_daily_visitors_l1537_153720

theorem average_daily_visitors
    (avg_sun : ℕ)
    (avg_other : ℕ)
    (days : ℕ)
    (starts_sun : Bool)
    (H1 : avg_sun = 630)
    (H2 : avg_other = 240)
    (H3 : days = 30)
    (H4 : starts_sun = true) :
    (5 * avg_sun + 25 * avg_other) / days = 305 :=
by
  sorry

end NUMINAMATH_GPT_average_daily_visitors_l1537_153720


namespace NUMINAMATH_GPT_average_age_of_both_teams_l1537_153707

theorem average_age_of_both_teams (n_men : ℕ) (age_men : ℕ) (n_women : ℕ) (age_women : ℕ) :
  n_men = 8 → age_men = 35 → n_women = 6 → age_women = 30 → 
  (8 * 35 + 6 * 30) / (8 + 6) = 32.857 := 
by
  intros h1 h2 h3 h4
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_average_age_of_both_teams_l1537_153707


namespace NUMINAMATH_GPT_anna_reading_time_l1537_153782

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_anna_reading_time_l1537_153782


namespace NUMINAMATH_GPT_value_of_one_house_l1537_153799

theorem value_of_one_house
  (num_brothers : ℕ) (num_houses : ℕ) (payment_each : ℕ) 
  (total_money_paid : ℕ) (num_older : ℕ) (num_younger : ℕ)
  (share_per_younger : ℕ) (total_inheritance : ℕ) (value_of_house : ℕ) :
  num_brothers = 5 →
  num_houses = 3 →
  num_older = 3 →
  num_younger = 2 →
  payment_each = 800 →
  total_money_paid = num_older * payment_each →
  share_per_younger = total_money_paid / num_younger →
  total_inheritance = num_brothers * share_per_younger →
  value_of_house = total_inheritance / num_houses →
  value_of_house = 2000 :=
by {
  -- Provided conditions and statements without proofs
  sorry
}

end NUMINAMATH_GPT_value_of_one_house_l1537_153799


namespace NUMINAMATH_GPT_abs_quotient_eq_sqrt_7_div_2_l1537_153779

theorem abs_quotient_eq_sqrt_7_div_2 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_quotient_eq_sqrt_7_div_2_l1537_153779


namespace NUMINAMATH_GPT_no_intersection_of_asymptotes_l1537_153718

noncomputable def given_function (x : ℝ) : ℝ :=
  (x^2 - 9 * x + 20) / (x^2 - 9 * x + 18)

theorem no_intersection_of_asymptotes : 
  (∀ x, x = 3 → ¬ ∃ y, y = given_function x) ∧ 
  (∀ x, x = 6 → ¬ ∃ y, y = given_function x) ∧ 
  ¬ ∃ x, (x = 3 ∨ x = 6) ∧ given_function x = 1 := 
by
  sorry

end NUMINAMATH_GPT_no_intersection_of_asymptotes_l1537_153718


namespace NUMINAMATH_GPT_zero_points_ordering_l1537_153761

noncomputable def f (x : ℝ) : ℝ := x + 2^x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) : ℝ := x^3 + x - 2

theorem zero_points_ordering :
  ∃ x1 x2 x3 : ℝ,
    f x1 = 0 ∧ x1 < 0 ∧ 
    g x2 = 0 ∧ 0 < x2 ∧ x2 < 1 ∧
    h x3 = 0 ∧ 1 < x3 ∧ x3 < 2 ∧
    x1 < x2 ∧ x2 < x3 := sorry

end NUMINAMATH_GPT_zero_points_ordering_l1537_153761


namespace NUMINAMATH_GPT_total_output_correct_l1537_153781

variable (a : ℝ)

-- Define a function that captures the total output from this year to the fifth year
def totalOutput (a : ℝ) : ℝ :=
  1.1 * a + (1.1 ^ 2) * a + (1.1 ^ 3) * a + (1.1 ^ 4) * a + (1.1 ^ 5) * a

theorem total_output_correct (a : ℝ) : 
  totalOutput a = 11 * (1.1 ^ 5 - 1) * a := by
  sorry

end NUMINAMATH_GPT_total_output_correct_l1537_153781


namespace NUMINAMATH_GPT_value_of_g_at_3_l1537_153724

def g (x : ℝ) := x^2 + 1

theorem value_of_g_at_3 : g 3 = 10 := by
  sorry

end NUMINAMATH_GPT_value_of_g_at_3_l1537_153724


namespace NUMINAMATH_GPT_find_a_if_perpendicular_l1537_153744

theorem find_a_if_perpendicular (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → False) →
  a = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_perpendicular_l1537_153744


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1537_153711

theorem solve_equation_1 (x : ℝ) : 2 * x^2 - x = 0 ↔ x = 0 ∨ x = 1 / 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : (2 * x + 1)^2 - 9 = 0 ↔ x = 1 ∨ x = -2 := 
by sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1537_153711


namespace NUMINAMATH_GPT_sum_from_neg_50_to_75_l1537_153775

def sum_of_integers (a b : ℤ) : ℤ :=
  (b * (b + 1)) / 2 - (a * (a - 1)) / 2

theorem sum_from_neg_50_to_75 : sum_of_integers (-50) 75 = 1575 := by
  sorry

end NUMINAMATH_GPT_sum_from_neg_50_to_75_l1537_153775


namespace NUMINAMATH_GPT_ratio_of_amount_spent_on_movies_to_weekly_allowance_l1537_153736

-- Define weekly allowance
def weekly_allowance : ℕ := 10

-- Define final amount after all transactions
def final_amount : ℕ := 11

-- Define earnings from washing the car
def earnings : ℕ := 6

-- Define amount left before washing the car
def amount_left_before_wash : ℕ := final_amount - earnings

-- Define amount spent on movies
def amount_spent_on_movies : ℕ := weekly_allowance - amount_left_before_wash

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Prove the required ratio
theorem ratio_of_amount_spent_on_movies_to_weekly_allowance :
  ratio amount_spent_on_movies weekly_allowance = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_amount_spent_on_movies_to_weekly_allowance_l1537_153736


namespace NUMINAMATH_GPT_distance_between_trees_l1537_153770

theorem distance_between_trees (L : ℝ) (n : ℕ) (hL : L = 375) (hn : n = 26) : 
  (L / (n - 1) = 15) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1537_153770


namespace NUMINAMATH_GPT_num_congruent_mod_7_count_mod_7_eq_22_l1537_153794

theorem num_congruent_mod_7 (n : ℕ) :
  (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) → ∃ k, 0 ≤ k ∧ k ≤ 21 ∧ n = 7 * k + 1 :=
sorry

theorem count_mod_7_eq_22 : 
  (∃ n_set : Finset ℕ, 
    (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) ∧ 
    Finset.card n_set = 22) :=
sorry

end NUMINAMATH_GPT_num_congruent_mod_7_count_mod_7_eq_22_l1537_153794


namespace NUMINAMATH_GPT_math_ineq_problem_l1537_153791

variable (a b c : ℝ)

theorem math_ineq_problem
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : a + b + c ≤ 1)
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_math_ineq_problem_l1537_153791


namespace NUMINAMATH_GPT_find_f2_l1537_153748

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_f2_l1537_153748


namespace NUMINAMATH_GPT_probability_same_color_opposite_feet_l1537_153739

/-- Define the initial conditions: number of pairs of each color. -/
def num_black_pairs : ℕ := 8
def num_brown_pairs : ℕ := 4
def num_gray_pairs : ℕ := 3
def num_red_pairs : ℕ := 1

/-- The total number of shoes. -/
def total_shoes : ℕ := 2 * (num_black_pairs + num_brown_pairs + num_gray_pairs + num_red_pairs)

theorem probability_same_color_opposite_feet :
  ((num_black_pairs * (num_black_pairs - 1)) + 
   (num_brown_pairs * (num_brown_pairs - 1)) + 
   (num_gray_pairs * (num_gray_pairs - 1)) + 
   (num_red_pairs * (num_red_pairs - 1))) * 2 / (total_shoes * (total_shoes - 1)) = 45 / 248 :=
by sorry

end NUMINAMATH_GPT_probability_same_color_opposite_feet_l1537_153739


namespace NUMINAMATH_GPT_number_of_cows_l1537_153716

variable (D C : Nat)

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 30) : C = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l1537_153716


namespace NUMINAMATH_GPT_tan_alpha_eq_two_imp_inv_sin_double_angle_l1537_153763

theorem tan_alpha_eq_two_imp_inv_sin_double_angle (α : ℝ) (h : Real.tan α = 2) : 
  (1 / Real.sin (2 * α)) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_two_imp_inv_sin_double_angle_l1537_153763


namespace NUMINAMATH_GPT_power_mod_result_l1537_153715

theorem power_mod_result :
  9^1002 % 50 = 1 := by
  sorry

end NUMINAMATH_GPT_power_mod_result_l1537_153715


namespace NUMINAMATH_GPT_prove_solution_l1537_153737

noncomputable def problem_statement : Prop := ∀ x : ℝ, (16 : ℝ)^(2 * x - 3) = (4 : ℝ)^(3 - x) → x = 9 / 5

theorem prove_solution : problem_statement :=
by
  intro x h
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_prove_solution_l1537_153737


namespace NUMINAMATH_GPT_smallest_y_value_l1537_153705

theorem smallest_y_value (y : ℝ) : 3 * y ^ 2 + 33 * y - 105 = y * (y + 16) → y = -21 / 2 ∨ y = 5 := sorry

end NUMINAMATH_GPT_smallest_y_value_l1537_153705


namespace NUMINAMATH_GPT_intersection_points_l1537_153708

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, 3*x^2 - 4*x + 2) ∧ p = (x, x^3 - 2*x^2 + 5*x - 1))} =
  {(1, 1), (3, 17)} :=
  sorry

end NUMINAMATH_GPT_intersection_points_l1537_153708


namespace NUMINAMATH_GPT_jade_living_expenses_l1537_153710

-- Definitions from the conditions
variable (income : ℝ) (insurance_fraction : ℝ) (savings : ℝ) (P : ℝ)

-- Constants from the given problem
noncomputable def jadeIncome : income = 1600 := by sorry
noncomputable def jadeInsuranceFraction : insurance_fraction = 1 / 5 := by sorry
noncomputable def jadeSavings : savings = 80 := by sorry

-- The proof problem statement
theorem jade_living_expenses :
    (P * 1600 + (1 / 5) * 1600 + 80 = 1600) → P = 3 / 4 := by
    intros h
    sorry

end NUMINAMATH_GPT_jade_living_expenses_l1537_153710


namespace NUMINAMATH_GPT_fisherman_daily_earnings_l1537_153773

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end NUMINAMATH_GPT_fisherman_daily_earnings_l1537_153773


namespace NUMINAMATH_GPT_fraction_of_milk_in_second_cup_l1537_153762

noncomputable def ratio_mixture (V: ℝ) (x: ℝ) :=
  ((2 / 5 * V + (1 - x) * V) / (3 / 5 * V + x * V))

theorem fraction_of_milk_in_second_cup
  (V: ℝ) 
  (hV: V > 0)
  (hx: ratio_mixture V x = 3 / 7) :
  x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_milk_in_second_cup_l1537_153762


namespace NUMINAMATH_GPT_range_of_a_l1537_153790

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, 4 ≤ x → (if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a) ≥ 4) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1537_153790


namespace NUMINAMATH_GPT_complex_square_eq_l1537_153756

variables {a b : ℝ} {i : ℂ}

theorem complex_square_eq :
  a + i = 2 - b * i → (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end NUMINAMATH_GPT_complex_square_eq_l1537_153756


namespace NUMINAMATH_GPT_maximum_sum_of_squares_l1537_153771

theorem maximum_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 20 :=
sorry

end NUMINAMATH_GPT_maximum_sum_of_squares_l1537_153771


namespace NUMINAMATH_GPT_no_rearrangement_of_power_of_two_l1537_153754

theorem no_rearrangement_of_power_of_two (k n : ℕ) (hk : k > 3) (hn : n > k) : 
  ∀ m : ℕ, 
    (m.toDigits = (2^k).toDigits → m ≠ 2^n) :=
by
  sorry

end NUMINAMATH_GPT_no_rearrangement_of_power_of_two_l1537_153754


namespace NUMINAMATH_GPT_rectangle_perimeter_l1537_153757

variable (L W : ℝ) 

theorem rectangle_perimeter (h1 : L > 4) (h2 : W > 4) (h3 : (L * W) - ((L - 4) * (W - 4)) = 168) : 
  2 * (L + W) = 92 := 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1537_153757


namespace NUMINAMATH_GPT_no_politics_reporters_l1537_153777

theorem no_politics_reporters (X Y Both XDontY YDontX International PercentageTotal : ℝ) 
  (hX : X = 0.35)
  (hY : Y = 0.25)
  (hBoth : Both = 0.20)
  (hXDontY : XDontY = 0.30)
  (hInternational : International = 0.15)
  (hPercentageTotal : PercentageTotal = 1.0) :
  PercentageTotal - ((X + Y - Both) - XDontY + International) = 0.75 :=
by sorry

end NUMINAMATH_GPT_no_politics_reporters_l1537_153777


namespace NUMINAMATH_GPT_rolls_to_neighbor_l1537_153738

theorem rolls_to_neighbor (total_needed rolls_to_grandmother rolls_to_uncle rolls_needed : ℕ) (h1 : total_needed = 45) (h2 : rolls_to_grandmother = 1) (h3 : rolls_to_uncle = 10) (h4 : rolls_needed = 28) :
  total_needed - rolls_needed - (rolls_to_grandmother + rolls_to_uncle) = 6 := by
  sorry

end NUMINAMATH_GPT_rolls_to_neighbor_l1537_153738


namespace NUMINAMATH_GPT_doubled_team_completes_half_in_three_days_l1537_153780

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end NUMINAMATH_GPT_doubled_team_completes_half_in_three_days_l1537_153780


namespace NUMINAMATH_GPT_part_a_gray_black_area_difference_l1537_153768

theorem part_a_gray_black_area_difference :
    ∀ (a b : ℕ), 
        a = 4 → 
        b = 3 →
        a^2 - b^2 = 7 :=
by
  intros a b h_a h_b
  sorry

end NUMINAMATH_GPT_part_a_gray_black_area_difference_l1537_153768


namespace NUMINAMATH_GPT_ratio_horizontal_to_checkered_l1537_153731

/--
In a cafeteria, 7 people are wearing checkered shirts, while the rest are wearing vertical stripes
and horizontal stripes. There are 40 people in total, and 5 of them are wearing vertical stripes.
What is the ratio of the number of people wearing horizontal stripes to the number of people wearing
checkered shirts?
-/
theorem ratio_horizontal_to_checkered
  (total_people : ℕ)
  (checkered_people : ℕ)
  (vertical_people : ℕ)
  (horizontal_people : ℕ)
  (ratio : ℕ)
  (h_total : total_people = 40)
  (h_checkered : checkered_people = 7)
  (h_vertical : vertical_people = 5)
  (h_horizontal : horizontal_people = total_people - checkered_people - vertical_people)
  (h_ratio : ratio = horizontal_people / checkered_people) :
  ratio = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_horizontal_to_checkered_l1537_153731


namespace NUMINAMATH_GPT_total_profit_is_28000_l1537_153714

noncomputable def investment_A (investment_B : ℝ) : ℝ := 3 * investment_B
noncomputable def period_A (period_B : ℝ) : ℝ := 2 * period_B
noncomputable def profit_B : ℝ := 4000
noncomputable def total_profit (investment_B period_B : ℝ) : ℝ :=
  let x := investment_B * period_B
  let a_share := 6 * x
  profit_B + a_share

theorem total_profit_is_28000 (investment_B period_B : ℝ) : 
  total_profit investment_B period_B = 28000 :=
by
  have h1 : profit_B = 4000 := rfl
  have h2 : investment_A investment_B = 3 * investment_B := rfl
  have h3 : period_A period_B = 2 * period_B := rfl
  simp [total_profit, h1, h2, h3]
  have x_def : investment_B * period_B = 4000 := by sorry
  simp [x_def]
  sorry

end NUMINAMATH_GPT_total_profit_is_28000_l1537_153714


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1537_153795

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b * x / Real.log x) - (a * x)
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ :=
  (b * (Real.log x - 1) / (Real.log x)^2) - a

theorem problem_part1 (a b : ℝ) :
  (f' (Real.exp 2) a b = -(3/4)) ∧ (f (Real.exp 2) a b = -(1/2) * (Real.exp 2)) →
  a = 1 ∧ b = 1 :=
sorry

theorem problem_part2 (a : ℝ) :
  (∃ x1 x2, x1 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ x2 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f x1 a 1 ≤ f' x2 a 1 + a) →
  a ≥ (1/2 - 1/(4 * Real.exp 2)) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1537_153795


namespace NUMINAMATH_GPT_nonnegative_integer_pairs_solution_l1537_153797

open Int

theorem nonnegative_integer_pairs_solution (x y : ℕ) : 
  3 * x ^ 2 + 2 * 9 ^ y = x * (4 ^ (y + 1) - 1) ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) :=
by 
  sorry

end NUMINAMATH_GPT_nonnegative_integer_pairs_solution_l1537_153797


namespace NUMINAMATH_GPT_calories_consumed_Jean_l1537_153767

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end NUMINAMATH_GPT_calories_consumed_Jean_l1537_153767


namespace NUMINAMATH_GPT_range_of_a_l1537_153783

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1537_153783


namespace NUMINAMATH_GPT_probability_of_multiple_of_42_is_zero_l1537_153796

-- Given conditions
def factors_200 : Set ℕ := {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200}
def multiple_of_42 (n : ℕ) : Prop := n % 42 = 0

-- Problem statement: the probability of selecting a multiple of 42 from the factors of 200 is 0.
theorem probability_of_multiple_of_42_is_zero : 
  ∀ (n : ℕ), n ∈ factors_200 → ¬ multiple_of_42 n := 
by
  sorry

end NUMINAMATH_GPT_probability_of_multiple_of_42_is_zero_l1537_153796


namespace NUMINAMATH_GPT_dealership_vans_expected_l1537_153798

theorem dealership_vans_expected (trucks vans : ℕ) (h_ratio : 3 * vans = 5 * trucks) (h_trucks : trucks = 45) : vans = 75 :=
by
  sorry

end NUMINAMATH_GPT_dealership_vans_expected_l1537_153798


namespace NUMINAMATH_GPT_john_score_l1537_153776

theorem john_score (s1 s2 s3 s4 s5 s6 : ℕ) (h1 : s1 = 85) (h2 : s2 = 88) (h3 : s3 = 90) (h4 : s4 = 92) (h5 : s5 = 83) (h6 : s6 = 102) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 90 :=
by
  sorry

end NUMINAMATH_GPT_john_score_l1537_153776


namespace NUMINAMATH_GPT_a_six_between_three_and_four_l1537_153726

theorem a_six_between_three_and_four (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := 
sorry

end NUMINAMATH_GPT_a_six_between_three_and_four_l1537_153726


namespace NUMINAMATH_GPT_find_x3_y3_l1537_153729

theorem find_x3_y3 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := 
by 
  sorry

end NUMINAMATH_GPT_find_x3_y3_l1537_153729


namespace NUMINAMATH_GPT_probability_same_group_l1537_153750

noncomputable def calcProbability : ℚ := 
  let totalOutcomes := 18 * 17
  let favorableCase1 := 6 * 5
  let favorableCase2 := 4 * 3
  let totalFavorableOutcomes := favorableCase1 + favorableCase2
  totalFavorableOutcomes / totalOutcomes

theorem probability_same_group (cards : Finset ℕ) (draws : Finset ℕ) (number1 number2 : ℕ) (condition_cardinality : cards.card = 20) 
  (condition_draws : draws.card = 4) (condition_numbers : number1 = 5 ∧ number2 = 14 ∧ number1 ∈ cards ∧ number2 ∈ cards) 
  : calcProbability = 7 / 51 :=
sorry

end NUMINAMATH_GPT_probability_same_group_l1537_153750


namespace NUMINAMATH_GPT_symmetric_scanning_codes_count_l1537_153721

theorem symmetric_scanning_codes_count :
  let grid_size := 5
  let total_squares := grid_size * grid_size
  let symmetry_classes := 5 -- Derived from classification in the solution
  let possible_combinations := 2 ^ symmetry_classes
  let invalid_combinations := 2 -- All black or all white grid
  total_squares = 25 
  ∧ (possible_combinations - invalid_combinations) = 30 :=
by sorry

end NUMINAMATH_GPT_symmetric_scanning_codes_count_l1537_153721


namespace NUMINAMATH_GPT_cuboid_edge_lengths_l1537_153785

theorem cuboid_edge_lengths (a b c : ℕ) (S V : ℕ) :
  (S = 2 * (a * b + b * c + c * a)) ∧ (V = a * b * c) ∧ (V = S) ∧ 
  (∃ d : ℕ, d = Int.sqrt (a^2 + b^2 + c^2)) →
  (∃ a b c : ℕ, a = 4 ∧ b = 8 ∧ c = 8) :=
by
  sorry

end NUMINAMATH_GPT_cuboid_edge_lengths_l1537_153785


namespace NUMINAMATH_GPT_total_slices_sold_l1537_153728

theorem total_slices_sold (sold_yesterday served_today : ℕ) (h1 : sold_yesterday = 5) (h2 : served_today = 2) :
  sold_yesterday + served_today = 7 :=
by
  -- Proof skipped
  exact sorry

end NUMINAMATH_GPT_total_slices_sold_l1537_153728


namespace NUMINAMATH_GPT_bottles_from_B_l1537_153753

-- Definitions for the bottles from each shop and the total number of bottles Don can buy
def bottles_from_A : Nat := 150
def bottles_from_C : Nat := 220
def total_bottles : Nat := 550

-- Lean statement to prove that the number of bottles Don buys from Shop B is 180
theorem bottles_from_B :
  total_bottles - (bottles_from_A + bottles_from_C) = 180 := 
by
  sorry

end NUMINAMATH_GPT_bottles_from_B_l1537_153753


namespace NUMINAMATH_GPT_candy_cost_l1537_153789

theorem candy_cost
  (C : ℝ) -- cost per pound of the first candy
  (w1 : ℝ := 30) -- weight of the first candy
  (c2 : ℝ := 5) -- cost per pound of the second candy
  (w2 : ℝ := 60) -- weight of the second candy
  (w_mix : ℝ := 90) -- total weight of the mixture
  (c_mix : ℝ := 6) -- desired cost per pound of the mixture
  (h1 : w1 * C + w2 * c2 = w_mix * c_mix) -- cost equation for the mixture
  : C = 8 :=
by
  sorry

end NUMINAMATH_GPT_candy_cost_l1537_153789


namespace NUMINAMATH_GPT_population_approx_10000_2090_l1537_153706

def population (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / 20)

theorem population_approx_10000_2090 :
  ∃ y, y = 2090 ∧ population 500 (2090 - 2010) = 500 * 2 ^ (80 / 20) :=
by
  sorry

end NUMINAMATH_GPT_population_approx_10000_2090_l1537_153706


namespace NUMINAMATH_GPT_anagrams_without_three_consecutive_identical_l1537_153751

theorem anagrams_without_three_consecutive_identical :
  let total_anagrams := 100800
  let anagrams_with_three_A := 6720
  let anagrams_with_three_B := 6720
  let anagrams_with_three_A_and_B := 720
  let valid_anagrams := total_anagrams - anagrams_with_three_A - anagrams_with_three_B + anagrams_with_three_A_and_B
  valid_anagrams = 88080 := by
  sorry

end NUMINAMATH_GPT_anagrams_without_three_consecutive_identical_l1537_153751


namespace NUMINAMATH_GPT_relay_team_average_time_l1537_153700

theorem relay_team_average_time :
  let d1 := 200
  let t1 := 38
  let d2 := 300
  let t2 := 56
  let d3 := 250
  let t3 := 47
  let d4 := 400
  let t4 := 80
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  let average_time_per_meter := total_time / total_distance
  average_time_per_meter = 0.1922 := by
  sorry

end NUMINAMATH_GPT_relay_team_average_time_l1537_153700


namespace NUMINAMATH_GPT_area_of_306090_triangle_l1537_153743

-- Conditions
def is_306090_triangle (a b c : ℝ) : Prop :=
  a / b = 1 / Real.sqrt 3 ∧ a / c = 1 / 2

-- Given values
def hypotenuse : ℝ := 6

-- To prove
theorem area_of_306090_triangle :
  ∃ (a b c : ℝ), is_306090_triangle a b c ∧ c = hypotenuse ∧ (1 / 2) * a * b = (9 * Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_306090_triangle_l1537_153743


namespace NUMINAMATH_GPT_EG_perpendicular_to_AC_l1537_153778

noncomputable def rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 < B.1 ∧ A.2 = B.2 ∧ B.1 < C.1 ∧ B.2 < C.2 ∧ C.1 = D.1 ∧ C.2 > D.2 ∧ D.1 > A.1 ∧ D.2 = A.2

theorem EG_perpendicular_to_AC
  {A B C D E F G: ℝ × ℝ}
  (h1: rectangle A B C D)
  (h2: E = (B.1, C.2) ∨ E = (C.1, B.2)) -- Assuming E lies on BC or BA
  (h3: F = (B.1, A.2) ∨ F = (A.1, B.2)) -- Assuming F lies on BA or BC
  (h4: G = (C.1, D.2) ∨ G = (D.1, C.2)) -- Assuming G lies on CD
  (h5: (F.1, G.2) = (A.1, C.2)) -- Line through F parallel to AC meets CD at G
: ∃ (H : ℝ × ℝ → ℝ × ℝ → ℝ), H E G = 0 := sorry

end NUMINAMATH_GPT_EG_perpendicular_to_AC_l1537_153778


namespace NUMINAMATH_GPT_range_of_m_l1537_153774

def p (m : ℝ) : Prop :=
  let Δ := m^2 - 4
  Δ > 0 ∧ -m < 0

def q (m : ℝ) : Prop :=
  let Δ := 16*(m-2)^2 - 16
  Δ < 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ ((1 < m ∧ m ≤ 2) ∨ 3 ≤ m) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_m_l1537_153774


namespace NUMINAMATH_GPT_patternD_cannot_form_pyramid_l1537_153766

-- Define the patterns
inductive Pattern
| A
| B
| C
| D

-- Define the condition for folding into a pyramid with a square base
def canFormPyramidWithSquareBase (p : Pattern) : Prop :=
  p = Pattern.A ∨ p = Pattern.B ∨ p = Pattern.C

-- Goal: Prove that Pattern D cannot be folded into a pyramid with a square base
theorem patternD_cannot_form_pyramid : ¬ canFormPyramidWithSquareBase Pattern.D :=
by
  -- Need to provide the proof here
  sorry

end NUMINAMATH_GPT_patternD_cannot_form_pyramid_l1537_153766


namespace NUMINAMATH_GPT_cello_viola_pairs_l1537_153793

theorem cello_viola_pairs (cellos violas : Nat) (p_same_tree : ℚ) (P : Nat)
  (h_cellos : cellos = 800)
  (h_violas : violas = 600)
  (h_p_same_tree : p_same_tree = 0.00020833333333333335)
  (h_equation : P * ((1 : ℚ) / cellos * (1 : ℚ) / violas) = p_same_tree) :
  P = 100 := 
by
  sorry

end NUMINAMATH_GPT_cello_viola_pairs_l1537_153793


namespace NUMINAMATH_GPT_polynomial_coef_sum_l1537_153742

theorem polynomial_coef_sum :
  ∃ (a b c d : ℝ), (∀ x : ℝ, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 14) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coef_sum_l1537_153742


namespace NUMINAMATH_GPT_original_number_l1537_153717

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 37.66666666666667) : 
  x + y = 32.7 := 
sorry

end NUMINAMATH_GPT_original_number_l1537_153717


namespace NUMINAMATH_GPT_max_flow_increase_proof_l1537_153712

noncomputable def max_flow_increase : ℕ :=
  sorry

theorem max_flow_increase_proof
  (initial_pipes_AB: ℕ) (initial_pipes_BC: ℕ) (flow_increase_per_pipes_swap: ℕ)
  (swap_increase: initial_pipes_AB = 10)
  (swap_increase_2: initial_pipes_BC = 10)
  (flow_increment: flow_increase_per_pipes_swap = 30) : 
  max_flow_increase = 150 :=
  sorry

end NUMINAMATH_GPT_max_flow_increase_proof_l1537_153712


namespace NUMINAMATH_GPT_angle_movement_condition_l1537_153758

noncomputable def angle_can_reach_bottom_right (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) : Prop :=
  (m % 2 = 1) ∧ (n % 2 = 1)

theorem angle_movement_condition (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) :
  angle_can_reach_bottom_right m n h1 h2 ↔ (m % 2 = 1 ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_GPT_angle_movement_condition_l1537_153758


namespace NUMINAMATH_GPT_count_negative_numbers_l1537_153703

def evaluate (e : String) : Int :=
  match e with
  | "-3^2" => -9
  | "(-3)^2" => 9
  | "-(-3)" => 3
  | "-|-3|" => -3
  | _ => 0

def isNegative (n : Int) : Bool := n < 0

def countNegatives (es : List String) : Int :=
  es.map evaluate |>.filter isNegative |>.length

theorem count_negative_numbers :
  countNegatives ["-3^2", "(-3)^2", "-(-3)", "-|-3|"] = 2 :=
by
  sorry

end NUMINAMATH_GPT_count_negative_numbers_l1537_153703


namespace NUMINAMATH_GPT_total_bowling_balls_l1537_153786

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end NUMINAMATH_GPT_total_bowling_balls_l1537_153786


namespace NUMINAMATH_GPT_ratio_of_sums_l1537_153704

variable {α : Type*} [LinearOrderedField α] 

variable (a : ℕ → α) (S : ℕ → α)
variable (a1 d : α)

def isArithmeticSequence (a : ℕ → α) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + n * d

def sumArithmeticSequence (a : α) (d : α) (n : ℕ) : α :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ratio_of_sums (h_arith : isArithmeticSequence a) (h_S : ∀ n, S n = sumArithmeticSequence a1 d n)
  (h_a5_5a3 : a 5 = 5 * a 3) : S 9 / S 5 = 9 := by sorry

end NUMINAMATH_GPT_ratio_of_sums_l1537_153704


namespace NUMINAMATH_GPT_range_of_a_l1537_153722

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - 2 * a

theorem range_of_a (a : ℝ) :
  (∃ (x₀ : ℝ), x₀ ≤ a ∧ f x₀ a ≥ 0) ↔ (a ∈ Set.Icc (-1 : ℝ) 0 ∪ Set.Ici 2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1537_153722


namespace NUMINAMATH_GPT_combined_work_time_l1537_153740

theorem combined_work_time (A B C D : ℕ) (hA : A = 10) (hB : B = 15) (hC : C = 20) (hD : D = 30) :
  1 / (1 / A + 1 / B + 1 / C + 1 / D) = 4 := by
  -- Replace the following "sorry" with your proof.
  sorry

end NUMINAMATH_GPT_combined_work_time_l1537_153740


namespace NUMINAMATH_GPT_cyclist_total_distance_l1537_153765

-- Definitions for velocities and times
def v1 : ℝ := 2  -- velocity in the first minute (m/s)
def v2 : ℝ := 4  -- velocity in the second minute (m/s)
def t : ℝ := 60  -- time interval in seconds (1 minute)

-- Total distance covered in two minutes
def total_distance : ℝ := v1 * t + v2 * t

-- The proof statement
theorem cyclist_total_distance : total_distance = 360 := by
  sorry

end NUMINAMATH_GPT_cyclist_total_distance_l1537_153765


namespace NUMINAMATH_GPT_non_negative_combined_quadratic_l1537_153730

theorem non_negative_combined_quadratic (a b c A B C : ℝ) (h1 : a ≥ 0) (h2 : b^2 ≤ a * c) (h3 : A ≥ 0) (h4 : B^2 ≤ A * C) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_non_negative_combined_quadratic_l1537_153730


namespace NUMINAMATH_GPT_max_horizontal_segment_length_l1537_153760

theorem max_horizontal_segment_length (y : ℝ → ℝ) (h : ∀ x, y x = x^3 - x) :
  ∃ a, (∀ x₁, y x₁ = y (x₁ + a)) ∧ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_horizontal_segment_length_l1537_153760


namespace NUMINAMATH_GPT_articles_profit_l1537_153745

variable {C S : ℝ}

theorem articles_profit (h1 : 20 * C = x * S) (h2 : S = 1.25 * C) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_articles_profit_l1537_153745


namespace NUMINAMATH_GPT_sqrt3_minus1_plus_inv3_pow_minus2_l1537_153788

theorem sqrt3_minus1_plus_inv3_pow_minus2 :
  (Real.sqrt 3 - 1) + (1 / (1/3) ^ 2) = Real.sqrt 3 + 8 :=
by
  sorry

end NUMINAMATH_GPT_sqrt3_minus1_plus_inv3_pow_minus2_l1537_153788


namespace NUMINAMATH_GPT_longest_playing_time_l1537_153709

theorem longest_playing_time (total_playtime : ℕ) (n : ℕ) (k : ℕ) (standard_time : ℚ) (long_time : ℚ) :
  total_playtime = 120 ∧ n = 6 ∧ k = 2 ∧ long_time = k * standard_time →
  5 * standard_time + long_time = 240 →
  long_time = 68 :=
by
  sorry

end NUMINAMATH_GPT_longest_playing_time_l1537_153709


namespace NUMINAMATH_GPT_mike_chocolate_squares_l1537_153759

theorem mike_chocolate_squares (M : ℕ) (h1 : 65 = 3 * M + 5) : M = 20 :=
by {
  -- proof of the theorem (not included as per instructions)
  sorry
}

end NUMINAMATH_GPT_mike_chocolate_squares_l1537_153759


namespace NUMINAMATH_GPT_correct_statement_l1537_153723

variable (P Q : Prop)
variable (hP : P)
variable (hQ : Q)

theorem correct_statement :
  (P ∧ Q) :=
by
  exact ⟨hP, hQ⟩

end NUMINAMATH_GPT_correct_statement_l1537_153723


namespace NUMINAMATH_GPT_cities_drawn_from_group_b_l1537_153769

def group_b_cities : ℕ := 8
def selection_probability : ℝ := 0.25

theorem cities_drawn_from_group_b : 
  group_b_cities * selection_probability = 2 :=
by
  sorry

end NUMINAMATH_GPT_cities_drawn_from_group_b_l1537_153769


namespace NUMINAMATH_GPT_infinite_series_sum_l1537_153735

theorem infinite_series_sum :
  (∑' n : ℕ, if n = 0 then 0 else (3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)))) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1537_153735
