import Mathlib

namespace old_car_fuel_consumption_l344_344500

theorem old_car_fuel_consumption : 
  ∀ d : ℝ, (d > 0) → 
  (∀ C_old C_new : ℝ, 
   (C_new = C_old - 2) → 
   (C_old = (100 / d)) → 
   (C_new = (100 / (d + 4.4)))
  ) → 
  abs((100 / d) - 7.82) < 0.01 :=
by intro d hd hcons;
   sorry

end old_car_fuel_consumption_l344_344500


namespace calculation_eq_minus_one_l344_344917

noncomputable def calculation : ℝ :=
  (-1)^(53 : ℤ) + 3^((2^3 + 5^2 - 7^2) : ℤ)

theorem calculation_eq_minus_one : calculation = -1 := 
by 
  sorry

end calculation_eq_minus_one_l344_344917


namespace sequence_bounded_l344_344431

noncomputable def x : ℕ → ℕ
| 0 => 10^2007 + 1
| (n + 1) => (11 * x n) % 10^(Nat.digits 10 (11 * x n) - 1)

theorem sequence_bounded : ∃ M, ∀ n, x n < M :=
by
  sorry

end sequence_bounded_l344_344431


namespace fraction_addition_l344_344549

theorem fraction_addition : 
  (2 : ℚ) / 5 + (3 : ℚ) / 8 + 1 = 71 / 40 :=
by
  sorry

end fraction_addition_l344_344549


namespace sum_of_six_least_solutions_l344_344350

def tau (n : ℕ) : ℕ := sorry -- This should be defined based on divisor count, a placeholder for now.

def satisfies_condition (n : ℕ) : Prop := tau(n) + tau(n + 1) = 8

theorem sum_of_six_least_solutions : ∃ (ns : Fin₆ (ℕ)), 
  (∀ n ∈ ns, satisfies_condition n) ∧ 
  ns.to_list.sum = 800 := 
sorry

end sum_of_six_least_solutions_l344_344350


namespace max_drumming_bunnies_l344_344497

structure Bunny where
  drum : ℕ
  drumsticks : ℕ

def can_drum (b1 b2 : Bunny) : Bool :=
  b1.drum > b2.drum ∧ b1.drumsticks > b2.drumsticks

theorem max_drumming_bunnies (bunnies : List Bunny) (h_size : bunnies.length = 7) : 
  ∃ (maxDrumming : ℕ), maxDrumming = 6 := 
by
  have h_drumming_limits : ∃ n, n ≤ 6 := 
    sorry -- Placeholder for the reasoning step
  use 6
  apply Eq.refl

-- Sorry is used to bypass the detailed proof, and placeholder comments indicate the steps needed for proof reasoning.

end max_drumming_bunnies_l344_344497


namespace student_B_speed_l344_344515

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l344_344515


namespace P_sufficient_but_not_necessary_for_Q_l344_344616

variable (x : ℝ)

def P := x ≥ 0
def Q := 2 * x + 1 / (2 * x + 1) ≥ 1

theorem P_sufficient_but_not_necessary_for_Q : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l344_344616


namespace count_rectangles_in_grid_l344_344280

theorem count_rectangles_in_grid :
  let horizontal_strip := 1,
      vertical_strip := 1,
      horizontal_rects := 1 + 2 + 3 + 4 + 5,
      vertical_rects := 1 + 2 + 3 + 4,
      double_counted := 1
  in horizontal_rects + vertical_rects - double_counted = 24 :=
by
  -- Definitions based on conditions
  let horizontal_strip := 1
  let vertical_strip := 1
  let horizontal_rects := 1 + 2 + 3 + 4 + 5
  let vertical_rects := 1 + 2 + 3 + 4
  let double_counted := 1

  -- Assertion of equality based on problem solution
  have h : horizontal_rects + vertical_rects - double_counted = 24 :=
    calc
      horizontal_rects + vertical_rects - double_counted
      = (1 + 2 + 3 + 4 + 5) + (1 + 2 + 3 + 4) - 1 : by rfl
      = 15 + 10 - 1 : by rfl
      = 24 : by rfl

  exact h

end count_rectangles_in_grid_l344_344280


namespace exists_such_h_l344_344946

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end exists_such_h_l344_344946


namespace grocery_store_more_expensive_per_can_l344_344876

theorem grocery_store_more_expensive_per_can :
  ∀ (bulk_case_price : ℝ) (bulk_cans_per_case : ℕ)
    (grocery_case_price : ℝ) (grocery_cans_per_case : ℕ),
  bulk_case_price = 12.00 →
  bulk_cans_per_case = 48 →
  grocery_case_price = 6.00 →
  grocery_cans_per_case = 12 →
  (grocery_case_price / grocery_cans_per_case - bulk_case_price / bulk_cans_per_case) * 100 = 25 :=
by
  intros _ _ _ _ h1 h2 h3 h4
  sorry

end grocery_store_more_expensive_per_can_l344_344876


namespace calculation_eq_minus_one_l344_344916

noncomputable def calculation : ℝ :=
  (-1)^(53 : ℤ) + 3^((2^3 + 5^2 - 7^2) : ℤ)

theorem calculation_eq_minus_one : calculation = -1 := 
by 
  sorry

end calculation_eq_minus_one_l344_344916


namespace Jeremy_songs_l344_344336

theorem Jeremy_songs (songs_yesterday : ℕ) (songs_difference : ℕ) (songs_today : ℕ) (total_songs : ℕ) :
  songs_yesterday = 9 ∧ songs_difference = 5 ∧ songs_today = songs_yesterday + songs_difference ∧ 
  total_songs = songs_yesterday + songs_today → total_songs = 23 :=
by
  intros h
  sorry

end Jeremy_songs_l344_344336


namespace student_B_speed_l344_344514

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l344_344514


namespace proof_problem_l344_344063

noncomputable def statement_1 (a : ℕ → ℕ) : Prop := ∀ n, a n = n
def statement_2 : Prop := true -- It involves analogical reasoning which is correct
def statement_3 : Prop := false -- Comparing a planar triangle with a parallelepiped is incorrect
def statement_4 (m : ℕ) : Prop := (m % 3 = 0 → m % 9 = 0) = false

theorem proof_problem : (statement_2 ∧ statement_4 3) ∧ (¬ statement_1 (λ n, n) ∧ ¬ statement_3) := 
by {
  -- Placeholder for the proof to ensure the Lean statement builds successfully.
  sorry
}

end proof_problem_l344_344063


namespace teacher_age_l344_344470

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_total : ℕ) (num_total : ℕ) (h1 : avg_age_students = 21) (h2 : num_students = 20) (h3 : avg_age_total = 22) (h4 : num_total = 21) :
  let total_age_students := avg_age_students * num_students
  let total_age_class := avg_age_total * num_total
  let teacher_age := total_age_class - total_age_students
  teacher_age = 42 :=
by
  sorry

end teacher_age_l344_344470


namespace profit_when_x_is_6_max_profit_l344_344882

noncomputable def design_fee : ℝ := 20000 / 10000
noncomputable def production_cost_per_100 : ℝ := 10000 / 10000

noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 14.7 - 9 / (x - 3)

noncomputable def cost_of_x_sets (x : ℝ) : ℝ :=
  design_fee + x * production_cost_per_100

noncomputable def profit (x : ℝ) : ℝ :=
  P x - cost_of_x_sets x

theorem profit_when_x_is_6 :
  profit 6 = 3.7 := sorry

theorem max_profit :
  ∀ x : ℝ, profit x ≤ 3.7 := sorry

end profit_when_x_is_6_max_profit_l344_344882


namespace amusement_park_ticket_cost_l344_344880

theorem amusement_park_ticket_cost
  (children_ages : ℕ × ℕ)
  (discount : ℕ)
  (total_given : ℕ)
  (change_received : ℕ)
  (children_discount_threshold : ℕ)
  (regular_ticket_cost : ℕ) :
  (fst children_ages < children_discount_threshold) →
  (snd children_ages < children_discount_threshold) →
  discount = 5 →
  total_given = 500 →
  change_received = 74 →
  children_discount_threshold = 12 →
  let total_cost := total_given - change_received
  in total_cost = 2 * regular_ticket_cost + 2 * (regular_ticket_cost - discount) →
     regular_ticket_cost = 109 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end amusement_park_ticket_cost_l344_344880


namespace factorize_x9_minus_512_l344_344153

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l344_344153


namespace relationship_between_abc_l344_344221

noncomputable def a : Real := Real.sqrt 1.2
noncomputable def b : Real := Real.exp 0.1
noncomputable def c : Real := 1 + Real.log 1.1

theorem relationship_between_abc : b > a ∧ a > c :=
by {
  -- a = sqrt(1.2)
  -- b = exp(0.1)
  -- c = 1 + log(1.1)
  -- We need to prove: b > a > c
  sorry
}

end relationship_between_abc_l344_344221


namespace coffee_ratio_l344_344380

theorem coffee_ratio (initial_amount : ℕ) (drank_on_way : ℕ) (remaining_after_way : ℕ)
(drinks_in_office : ℕ) (drinks_when_cold : ℕ) (remaining_after_cold : ℕ)
(h_initial : initial_amount = 12)
(h_drank_on_way : drank_on_way = initial_amount / 4)
(h_remaining_after_way : remaining_after_way = initial_amount - drank_on_way)
(h_drinks_when_cold : drinks_when_cold = 1)
(h_remaining_after_cold : remaining_after_cold = 2)
(h_remaining_before_cold : remaining_after_cold + drinks_when_cold = 3)
(h_drinks_in_office : drinks_in_office = remaining_after_way - 3) :
(drinks_in_office : ℕ) (remaining_after_way : ℕ) := 2 / 3 :=
by
  -- Proof steps are omitted
  sorry

end coffee_ratio_l344_344380


namespace exercise_l344_344693

theorem exercise (YX XZ : ℝ) (hYX : YX = 45) (hXZ : XZ = 60)
  (XYZ_right : ∃ X Y Z : ℝ × ℝ, Y = (0, 0) ∧ Z = (75, 0) ∧
    X = (27, 36) ∧ right_triangle X Y Z ∧ 
    YX ^ 2 + XZ ^ 2 = (75 : ℝ) ^ 2) :
  let W : (ℝ × ℝ) := (27, 0) in
  dist W (75, 0) = 48 := by
  sorry

end exercise_l344_344693


namespace rectangle_angle_XPY_l344_344313

noncomputable def measure_angle_XPY (W X Y Z P : ℝ) (h_WZ : W = 8) (h_XY : Y = 4)
  (h_ratio : (sin (angle W P Y) / sin (angle X P Y)) = 2) : Prop := 
  (angle X P Y) = 45

-- statement of the theorem
theorem rectangle_angle_XPY (W X Y Z P : ℝ) 
  (h_WZ : W = 8) (h_XY : Y = 4)
  (h_ratio : (sin (angle W P Y) / sin (angle X P Y)) = 2) : 
  measure_angle_XPY W X Y Z P h_WZ h_XY h_ratio :=
sorry

end rectangle_angle_XPY_l344_344313


namespace simplify_fraction_l344_344396

theorem simplify_fraction :
  ( (3 * 5 * 7 : ℚ) / (9 * 11 * 13) ) * ( (7 * 9 * 11 * 15) / (3 * 5 * 14) ) = 15 / 26 :=
by
  sorry

end simplify_fraction_l344_344396


namespace apple_production_total_l344_344126

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l344_344126


namespace fraction_to_decimal_l344_344955

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l344_344955


namespace quadratic_two_distinct_real_roots_l344_344762

theorem quadratic_two_distinct_real_roots
  (a1 a2 a3 a4 : ℝ)
  (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - (a1 + a2 + a3 + a4) * x1 + (a1 * a3 + a2 * a4) = 0)
  ∧ (x2^2 - (a1 + a2 + a3 + a4) * x2 + (a1 * a3 + a2 * a4) = 0) :=
by 
  sorry

end quadratic_two_distinct_real_roots_l344_344762


namespace tan_theta_half_l344_344608

theorem tan_theta_half (θ : ℝ) (a b : ℝ × ℝ) 
  (h₀ : a = (Real.sin θ, 1)) 
  (h₁ : b = (-2, Real.cos θ)) 
  (h₂ : a.1 * b.1 + a.2 * b.2 = 0) : Real.tan θ = 1 / 2 :=
sorry

end tan_theta_half_l344_344608


namespace correct_number_of_stationery_sales_l344_344489

/-- Define the given conditions -/
def fabric_percentage : ℝ := 27.5
def jewelry_percentage : ℝ := 18.25
def knitting_percentage : ℝ := 12.5
def home_decor_percentage : ℝ := 7.75
def total_sales : ℝ := 315

/-- Calculate the percentage of sales in the stationery section -/
def stationery_percentage : ℝ := 100 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage)

/-- Calculate the number of sales in the stationery section -/
def sales_in_stationery : ℝ := (stationery_percentage / 100) * total_sales

/-- The number of sales in the stationery section rounded to the nearest whole number -/
def sales_in_stationery_rounded : ℕ := sales_in_stationery.to_nat

/-- The theorem to prove the correct answer is 107 -/
theorem correct_number_of_stationery_sales : sales_in_stationery_rounded = 107 :=
  by
    sorry

end correct_number_of_stationery_sales_l344_344489


namespace square_side_length_l344_344789

theorem square_side_length :
  ∀ (s : ℝ), (∃ w l : ℝ, w = 6 ∧ l = 24 ∧ s^2 = w * l) → s = 12 := by 
  sorry

end square_side_length_l344_344789


namespace light_glow_duration_l344_344421

-- Define the conditions
def total_time_seconds : ℕ := 4969
def glow_times : ℚ := 292.29411764705884

-- Prove the equivalent statement
theorem light_glow_duration :
  (total_time_seconds / glow_times) = 17 := by
  sorry

end light_glow_duration_l344_344421


namespace trapezium_side_length_l344_344195

theorem trapezium_side_length :
  ∃ (x : ℝ), 
    (let area := 342 in
     let side1 := 24 in
     let height := 18 in
     (1 / 2) * (side1 + x) * height = area) := 
begin
  use 14,
  sorry
end

end trapezium_side_length_l344_344195


namespace fraction_to_decimal_l344_344967

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l344_344967


namespace math_proof_problem_l344_344328

variable (n : ℕ) (a b : ℕ → ℕ) (S T : ℕ → ℕ)

-- Conditions
def condition_a1 : Prop := a 1 = 1
def condition_b1 : Prop := b 1 = 4
def condition_sum : Prop := ∀ n ∈ ℕ, n * S (n+1) - (n+3) * S n = 0
def condition_geom_mean : Prop := ∀ n ∈ ℕ, 2 * a (n+1) = ((b n) * (b (n+1))) / 4

-- Proof goals
def goal_a2 : Prop := a 2 = 3
def goal_b2 : Prop := b 2 = 9
def goal_general_terms_a : Prop := ∀ n ∈ ℕ, a n = n * (n+1) / 2
def goal_general_terms_b : Prop := ∀ n ∈ ℕ, b n = (n+1)^2
def goal_t : Prop := ∀ n ≥ 3, |T n| < 2 * n^2

-- Main theorem statement
theorem math_proof_problem :
  condition_a1 ∧ condition_b1 ∧ condition_sum ∧ condition_geom_mean →
  (goal_a2 ∧ goal_b2) ∧
  (goal_general_terms_a ∧ goal_general_terms_b) ∧
  goal_t :=
 by sorry

end math_proof_problem_l344_344328


namespace basketball_cricket_students_l344_344681

theorem basketball_cricket_students {A B : Finset ℕ} (hA : A.card = 7) (hB : B.card = 8) (hAB : (A ∩ B).card = 3) :
  (A ∪ B).card = 12 :=
by
  sorry

end basketball_cricket_students_l344_344681


namespace students_not_next_to_each_other_l344_344311

-- Let's define the problem in Lean.
theorem students_not_next_to_each_other : 
  let total_arrangements := (5.factorial : ℕ)
  let blocked_arrangements := (4.factorial : ℕ) * (2.factorial : ℕ)
  total_arrangements - blocked_arrangements = 72 :=
by
  sorry

end students_not_next_to_each_other_l344_344311


namespace simplify_sqrt_300_l344_344776

theorem simplify_sqrt_300 :
  sqrt 300 = 10 * sqrt 3 :=
by
  -- Proof would go here
  sorry

end simplify_sqrt_300_l344_344776


namespace find_third_number_l344_344790

theorem find_third_number (x : ℝ) : 
  let avg1 : ℝ := (10 + 70 + 28) / 3 in
  let avg2 : ℝ := avg1 + 4 in
  avg1 = 36 → avg2 = 40 → 
  (20 + 40 + x) / 3 = 40 → x = 60 :=
by
  intros avg1 avg2 h1 h2 h3
  simp at h1 h2 h3
  sorry

end find_third_number_l344_344790


namespace soy_sauce_bottle_size_l344_344398

theorem soy_sauce_bottle_size 
  (ounces_per_cup : ℕ)
  (cups_recipe1 : ℕ)
  (cups_recipe2 : ℕ)
  (cups_recipe3 : ℕ)
  (number_of_bottles : ℕ)
  (total_ounces_needed : ℕ)
  (ounces_per_bottle : ℕ) :
  ounces_per_cup = 8 →
  cups_recipe1 = 2 →
  cups_recipe2 = 1 →
  cups_recipe3 = 3 →
  number_of_bottles = 3 →
  total_ounces_needed = (cups_recipe1 + cups_recipe2 + cups_recipe3) * ounces_per_cup →
  ounces_per_bottle = total_ounces_needed / number_of_bottles →
  ounces_per_bottle = 16 :=
by
  sorry

end soy_sauce_bottle_size_l344_344398


namespace polynomial_solution_l344_344555

noncomputable def P : ℝ → ℝ
  := sorry

theorem polynomial_solution (n : ℕ) (hn : n ≥ 1) (P : ℕ → ℤ)
  (hP : ∀ x : ℝ, P x = ∏ i in finset.range n, (x - P i)) :
  ∀ x : ℝ, P x = x := 
begin
  sorry
end

end polynomial_solution_l344_344555


namespace parallel_lines_implies_m_neg1_l344_344638

theorem parallel_lines_implies_m_neg1 (m : ℝ) :
  (∀ (x y : ℝ), x + m * y + 6 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + 3 * y + 2 * m = 0) ∧
  ∀ (l₁ l₂ : ℝ), l₁ = -(1 / m) ∧ l₂ = -((m - 2) / 3) ∧ l₁ = l₂ → m = -1 :=
by
  sorry

end parallel_lines_implies_m_neg1_l344_344638


namespace total_time_for_12000_dolls_l344_344342

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end total_time_for_12000_dolls_l344_344342


namespace each_organization_receives_l344_344891

-- Definition of conditions
def total_amount_raised : ℝ := 2500
def donation_rate : ℝ := 0.80
def num_organizations : ℝ := 8

-- Calculation based on conditions
def amount_donated : ℝ := donation_rate * total_amount_raised
def amount_per_organization : ℝ := amount_donated / num_organizations

-- Theorem statement
theorem each_organization_receives :
  amount_per_organization = 250 := by
  sorry

end each_organization_receives_l344_344891


namespace part_a_part_b_part_c_l344_344236

open Int

variable (a : ℤ)
variable (h : 2 < a)

-- Part (a)
theorem part_a : ∃ n : ℕ, ∃ p : ℕ, Prime p ∧ p ∣ a - 1 ∧ n = p * p ∧ a^n ≡ 1 [MOD n] ∧ n ≠ 1 ∧ ¬ Prime n :=
by sorry

-- Part (b)
theorem part_b (p : ℕ) (hp : p ≠ 1) (hpa : a^p ≡ 1 [MOD p]) : Prime p :=
by sorry

-- Part (c)
theorem part_c : ¬ ∃ n : ℕ, n ≠ 1 ∧ 2^n ≡ 1 [MOD n] :=
by sorry

end part_a_part_b_part_c_l344_344236


namespace negation_of_proposition_l344_344263

theorem negation_of_proposition (a : ℝ) :
  (¬ (∀ x : ℝ, (x - a) ^ 2 + 2 > 0)) ↔ (∃ x : ℝ, (x - a) ^ 2 + 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l344_344263


namespace find_m_l344_344362

theorem find_m (x1 x2 m : ℝ)
  (h1 : ∀ x, x^2 - 4 * x + m = 0 → x = x1 ∨ x = x2)
  (h2 : x1 + x2 - x1 * x2 = 1) :
  m = 3 :=
sorry

end find_m_l344_344362


namespace cell_phone_usage_planned_l344_344149

theorem cell_phone_usage_planned 
    (total_spending : ℝ) (food_spending : ℝ) (rent_spending : ℝ)
    (video_streaming_spending : ℝ) (savings_percentage : ℝ) (savings_amount : ℝ) :
    total_spending = 1980 →
    food_spending = 100 * 4 →
    rent_spending = 1500 →
    video_streaming_spending = 30 →
    savings_percentage = 0.10 →
    savings_amount = 198 →
    total_spending - (food_spending + rent_spending + video_streaming_spending) = 50 :=
by
  intros h_total_spending h_food_spending h_rent_spending h_video_streaming_spending h_savings_percentage h_savings_amount
  rw [h_total_spending, h_food_spending, h_rent_spending, h_video_streaming_spending]
  calc
    1980 - (400 + 1500 + 30) = 1980 - 1930 := by rfl
    ... = 50 := by rfl
sorry

end cell_phone_usage_planned_l344_344149


namespace theater_total_seats_l344_344685

theorem theater_total_seats :
  ∃ (n : ℕ), (n > 0) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ n → 15 + (k-1) * 2 ≤ 53) ∧
  let S := n * (15 + 53) / 2 in S = 680 :=
by
  sorry

end theater_total_seats_l344_344685


namespace sum_of_first_and_third_is_68_l344_344036

theorem sum_of_first_and_third_is_68
  (A B C : ℕ)
  (h1 : A + B + C = 98)
  (h2 : A * 3 = B * 2)  -- implying A / B = 2 / 3
  (h3 : B * 8 = C * 5)  -- implying B / C = 5 / 8
  (h4 : B = 30) :
  A + C = 68 :=
sorry

end sum_of_first_and_third_is_68_l344_344036


namespace angle_ABC_lt_60_l344_344317

variable {A B C H M : Point}
variable (ABC_is_acute : ∀ (a b c : ℕ), (a + b + c = 180) → (a < 90) → (b < 90) → (c < 90) → True)
variable (altitude_AH_is_longest : ∀ x y z : Point, (between A H H) → ((distance A H) > (distance x y)))
variable (AH_eq_BM : distance A H = distance B M)

theorem angle_ABC_lt_60 (ABC_is_acute : ∀ (a b c : ℕ), (a + b + c = 180) → (a < 90) → (b < 90) → (c < 90) → True)
    (altitude_AH_is_longest : ∀ x y z : Point, (between A H H) → ((distance A H) > (distance x y)))
    (AH_eq_BM : distance A H = distance B M) :
    measureAngle B A C < 60 := sorry

end angle_ABC_lt_60_l344_344317


namespace product_of_real_parts_of_solutions_l344_344412

theorem product_of_real_parts_of_solutions (x y : ℂ) (hx : x^3 + x^2 + 3 * x = 2 + 2 * complex.I)
    (hy : y^3 + y^2 + 3 * y = 2 + 2 * complex.I)
    (hxy : x ≠ y) :
    ((complex.re x) * (complex.re y) = 1 - real.sqrt 2) :=
sorry

end product_of_real_parts_of_solutions_l344_344412


namespace bridge_length_l344_344081

-- Definitions based on conditions
def Lt : ℕ := 148
def Skm : ℕ := 45
def T : ℕ := 30

-- Conversion from km/h to m/s
def conversion_factor : ℕ := 1000 / 3600
def Sm : ℝ := Skm * conversion_factor

-- Calculation of distance traveled in 30 seconds
def distance : ℝ := Sm * T

-- The length of the bridge
def L_bridge : ℝ := distance - Lt

theorem bridge_length : L_bridge = 227 := sorry

end bridge_length_l344_344081


namespace right_triangle_pythagoras_l344_344688

variable (ABC : Triangle)
variable (a b c : ℝ)
variable (angleA : ABC.angle = 90)

theorem right_triangle_pythagoras {ABC : Triangle} (h : ABC.is_right_triangle angleA) : b^2 + c^2 = a^2 :=
by
  sorry

end right_triangle_pythagoras_l344_344688


namespace fraction_to_decimal_l344_344980

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344980


namespace number_of_valid_sequences_of_10_is_0_l344_344782

-- Define the vertices A, B, C, and D
def A := (1, 1)
def B := (-1, 1)
def C := (-1, -1)
def D := (1, -1)

-- Define the transformations
def L (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)  -- 90 degree counterclockwise rotation
def R (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)  -- 90 degree clockwise rotation

-- Define the identity transformation
def I (p : ℝ × ℝ) : ℝ × ℝ := p

-- Define the sequence of transformations returning to the original position
def sequence_of_10_transformations_returns_to_origin (T : list (ℝ × ℝ -> ℝ × ℝ)) (n : ℕ) : Prop :=
  (list.foldl (λ acc t, t acc) (1, 1) T = I (1, 1)) ∧ 
  (list.foldl (λ acc t, t acc) (-1, 1) T = I (-1, 1)) ∧ 
  (list.foldl (λ acc t, t acc) (-1, -1) T = I (-1, -1)) ∧ 
  (list.foldl (λ acc t, t acc) (1, -1) T = I (1, -1))

-- The main statement proving the number of such sequences is 0
theorem number_of_valid_sequences_of_10_is_0 : 
  ∀ T : list (ℝ × ℝ -> ℝ × ℝ), T.length = 10 → (∀ t ∈ T, t = L ∨ t = R) → ¬ sequence_of_10_transformations_returns_to_origin T 10 :=
  sorry

end number_of_valid_sequences_of_10_is_0_l344_344782


namespace part_a_impossible_part_b_possible_l344_344440

/-- Given 10 initial numbers 1, 2, ..., 10,
there exists no sequence of operations (replacing any two numbers a and b
with a + 2b and b + 2a) such that, eventually, all numbers become the same. -/
theorem part_a_impossible : ¬ ∃ f : ℕ → ℕ → ℕ, ∀ a b : ℕ,
  (a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) → f a b ∈ {∀ x, x = x} :=
sorry

/-- Given any 10 different numbers,
there exists a sequence of operations (replacing any two numbers a and b
with a + 2b and b + 2a) such that, eventually, all numbers become the same. -/
theorem part_b_possible : ∀ (nums : set ℕ), nums.card = 10 → (∀ x y, x ≠ y → x ∈ nums → y ∈ nums) →
  ∃ g : ℕ → ℕ → ℕ, ∀ a b : ℕ,
  (a ∈ nums ∧ b ∈ nums) → g a b ∈ {∀ x, x = x} :=
sorry

end part_a_impossible_part_b_possible_l344_344440


namespace g_five_eq_one_l344_344802

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one 
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : ∀ x : ℝ, g x = g (-x)) : 
  g 5 = 1 :=
sorry

end g_five_eq_one_l344_344802


namespace problem_Ashwin_Sah_l344_344165

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end problem_Ashwin_Sah_l344_344165


namespace sum_of_six_least_solutions_l344_344352

def tau (n : ℕ) : ℕ := sorry -- This should be defined based on divisor count, a placeholder for now.

def satisfies_condition (n : ℕ) : Prop := tau(n) + tau(n + 1) = 8

theorem sum_of_six_least_solutions : ∃ (ns : Fin₆ (ℕ)), 
  (∀ n ∈ ns, satisfies_condition n) ∧ 
  ns.to_list.sum = 800 := 
sorry

end sum_of_six_least_solutions_l344_344352


namespace symmetry_about_x_eq_1_l344_344933

def g (x : ℝ) : ℝ := abs (floor (2 * x)) - abs (floor (2 - x))

theorem symmetry_about_x_eq_1 : ∀ x : ℝ, g x = g (1 - x) := by
  intros
  sorry

end symmetry_about_x_eq_1_l344_344933


namespace bananas_count_l344_344177

/-- Elias bought some bananas and ate 1 of them. 
    After eating, he has 11 bananas left.
    Prove that Elias originally bought 12 bananas. -/
theorem bananas_count (x : ℕ) (h1 : x - 1 = 11) : x = 12 := by
  sorry

end bananas_count_l344_344177


namespace circumsphere_radius_of_pyramid_l344_344806

theorem circumsphere_radius_of_pyramid (a α : ℝ) (h1 : 0 < a) (h2 : 0 < α ∧ α < π) : 
  ∃ R, 
    R = a * cos (α / 2) / (2 * sqrt (sin (π / 3 + α / 2) * sin (π / 3 - α / 2))) :=
by sorry

end circumsphere_radius_of_pyramid_l344_344806


namespace restore_triangle_l344_344754

noncomputable def triangle_coordinates
  (M N P : ℝ × ℝ) 
  (H_M : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (H_N : N = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (H_P : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let A := (2 * N.1 - C.1, 2 * N.2 - C.2)
  let B := (2 * P.1 - A.1, 2 * P.2 - A.2)
  let C := (2 * M.1 - B.1, 2 * M.2 - B.2)
  (A, B, C)

theorem restore_triangle
  (M N P : ℝ × ℝ) 
  (H_M : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (H_N : N = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (H_P : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ A B C : ℝ × ℝ, 
  A = (2 * N.1 - C.1, 2 * N.2 - C.2) ∧
  B = (2 * P.1 - A.1, 2 * P.2 - A.2) ∧
  C = (2 * M.1 - B.1, 2 * M.2 - B.2) :=
  sorry

end restore_triangle_l344_344754


namespace acetone_C_mass_percentage_l344_344459

noncomputable def mass_percentage_C_in_acetone : ℝ :=
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + (1 * atomic_mass_O)
  let total_mass_C := 3 * atomic_mass_C
  (total_mass_C / molar_mass_acetone) * 100

theorem acetone_C_mass_percentage :
  abs (mass_percentage_C_in_acetone - 62.01) < 0.01 := by
  sorry

end acetone_C_mass_percentage_l344_344459


namespace value_of_some_number_l344_344663

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l344_344663


namespace integral_relation_l344_344238

theorem integral_relation:
  (∫ x in 1..5, (1/x)) / 5 < (∫ x in 1..2, (1/x)) / 2 ∧ (∫ x in 1..2, (1/x)) / 2 < (∫ x in 1..3, (1/x)) / 3 := 
  sorry

end integral_relation_l344_344238


namespace reading_time_difference_l344_344065

theorem reading_time_difference 
  (xanthia_reading_speed : ℕ) 
  (molly_reading_speed : ℕ) 
  (book_pages : ℕ) 
  (time_conversion_factor : ℕ)
  (hx : xanthia_reading_speed = 150)
  (hm : molly_reading_speed = 75)
  (hp : book_pages = 300)
  (ht : time_conversion_factor = 60) :
  ((book_pages / molly_reading_speed - book_pages / xanthia_reading_speed) * time_conversion_factor = 120) := 
by
  sorry

end reading_time_difference_l344_344065


namespace rectangles_in_grid_l344_344284

theorem rectangles_in_grid : 
  let horizontal_strip := 1 * 5 in
  let vertical_strip := 1 * 4 in
  ∃ (rectangles : ℕ), rectangles = 24 := 
  by
  sorry

end rectangles_in_grid_l344_344284


namespace frankie_total_pets_l344_344214

def total_pets (C snakes parrots tortoises dogs hamsters fish four_legged_pets) : Prop :=
  snakes = 2 * C ∧
  parrots = C - 1 ∧
  tortoises = C ∧
  dogs = 2 ∧
  hamsters = 3 ∧
  fish = 5 ∧
  four_legged_pets = 14 ∧
  (C + tortoises + dogs = four_legged_pets) ∧
  (C + snakes + parrots + tortoises + dogs + hamsters + fish = 39)

theorem frankie_total_pets : ∃ C snakes parrots tortoises dogs hamsters fish four_legged_pets, total_pets C snakes parrots tortoises dogs hamsters fish four_legged_pets :=
by { sorry }

end frankie_total_pets_l344_344214


namespace cups_of_baking_mix_planned_l344_344109

-- Definitions
def butter_per_cup := 2 -- 2 ounces of butter per 1 cup of baking mix
def coconut_oil_per_butter := 2 -- 2 ounces of coconut oil can substitute 2 ounces of butter
def butter_remaining := 4 -- Chef had 4 ounces of butter
def coconut_oil_used := 8 -- Chef used 8 ounces of coconut oil

-- Statement to be proven
theorem cups_of_baking_mix_planned : 
  (butter_remaining / butter_per_cup) + (coconut_oil_used / coconut_oil_per_butter) = 6 := 
by 
  sorry

end cups_of_baking_mix_planned_l344_344109


namespace AB_equal_AF_l344_344620

variables {A B C D H F G : Type} [AffineSpace H] [MetricSpace H]

def is_parallelogram (A B C D : H) : Prop :=
-- Define the condition for quadrilateral ABCD to be a parallelogram
sorry

def is_concyclic (H A C D : H) : Prop :=
-- Define the condition for points H, A, C, D to be concyclic
sorry

def intersection (A B C D : H) (F : H) : Prop :=
-- Define the condition for F to be the intersection point of AD and BG
sorry

def equal_lengths (F G H : H) : Prop :=
-- Define the condition for HF = HD = HG
sorry

theorem AB_equal_AF
  (h_parallelogram : is_parallelogram A B C D)
  (h_concyclic : is_concyclic H A C D)
  (h_intersection : intersection A D B G F)
  (h_equal_lengths : equal_lengths F H G) :
  dist A B = dist A F :=
by sorry

end AB_equal_AF_l344_344620


namespace product_remainder_l344_344583

theorem product_remainder (n : ℕ) (hn : n = 20) : 
  (List.prod (List.map (λ i, 10 * i + 4) (List.range n))) % 5 = 1 := by
  sorry

end product_remainder_l344_344583


namespace infinite_sqrt_evaluation_l344_344183

noncomputable def infinite_sqrt : ℝ := 
  sqrt (15 + infinite_sqrt)

theorem infinite_sqrt_evaluation : infinite_sqrt = (1 + sqrt 61) / 2 := by
  sorry

end infinite_sqrt_evaluation_l344_344183


namespace stitches_per_flower_l344_344148

-- Define all conditions as Lean assumptions

-- Carolyn's sewing speed
constant sew_rate : ℕ := 4

-- Stitches required for each pattern
constant unicorn_stitches : ℕ := 180
constant godzilla_stitches : ℕ := 800

-- Embroidery details
constant TotalMinutes : ℕ := 1085
constant TotalUnicorns : ℕ := 3
constant TotalFlowers : ℕ := 50

-- Stitches for the required patterns
constant TotalTimeSpentSewing : ℕ
constant TotalStitches : ℕ := sew_rate * TotalMinutes
constant TotalUnicornStitches : ℕ := TotalUnicorns * unicorn_stitches
constant TotalGodzillaStitches : ℕ := godzilla_stitches

-- Prove the number of stitches per flower
theorem stitches_per_flower : ∃ x : ℕ, x = (TotalStitches - (TotalUnicornStitches + TotalGodzillaStitches)) / TotalFlowers :=
by
  existsi 60
  sorry

end stitches_per_flower_l344_344148


namespace compute_summation_l344_344925

theorem compute_summation :
  (1 / 2 ^ 1990) * ∑ n in Finset.range 996, (-3 : ℝ) ^ n * Nat.choose 1990 (2 * n) = -1 / 2 := 
sorry

end compute_summation_l344_344925


namespace ratio_doubled_to_original_l344_344106

theorem ratio_doubled_to_original (x : ℝ) (h : 3 * (2 * x + 9) = 69) : (2 * x) / x = 2 :=
by
  -- We skip the proof here.
  sorry

end ratio_doubled_to_original_l344_344106


namespace functional_relationship_l344_344222

variable (x y k1 k2 : ℝ)

axiom h1 : y = k1 * x + k2 / (x - 2)
axiom h2 : (y = -1) ↔ (x = 1)
axiom h3 : (y = 5) ↔ (x = 3)

theorem functional_relationship :
  (∀ x y, y = k1 * x + k2 / (x - 2) ∧
    ((x = 1) → y = -1) ∧
    ((x = 3) → y = 5) → y = x + 2 / (x - 2)) :=
by
  sorry

end functional_relationship_l344_344222


namespace highest_score_is_151_l344_344004

-- Definitions for the problem conditions
def total_runs : ℕ := 2704
def total_runs_excluding_HL : ℕ := 2552

variables (H L : ℕ) 

-- Problem conditions as hypotheses
axiom h1 : H - L = 150
axiom h2 : H + L = 152
axiom h3 : 2704 = 2552 + H + L

-- Proof statement
theorem highest_score_is_151 (H L : ℕ) (h1 : H - L = 150) (h2 : H + L = 152) (h3 : 2704 = 2552 + H + L) : H = 151 :=
by sorry

end highest_score_is_151_l344_344004


namespace fraction_to_decimal_l344_344960

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344960


namespace cone_lateral_surface_area_l344_344249

noncomputable def lateral_surface_area_of_cone (r l : ℝ) : ℝ := π * r * l

theorem cone_lateral_surface_area (h l : ℝ) (h_gt_zero : h > 0) (l_eq_two : l = 2) (h_eq_sqrt3 : h = sqrt 3) :
  ∃ r, (r = 1) ∧ (lateral_surface_area_of_cone r l = 2 * π) :=
by sorry

end cone_lateral_surface_area_l344_344249


namespace Aubriella_pouring_time_l344_344542

theorem Aubriella_pouring_time (total_capacity : ℕ) (rate : ℕ) (to_be_poured : ℕ) (already_poured : ℕ) :
  total_capacity = 50 ∧ rate = 20 ∧ to_be_poured = 32 ∧ already_poured = 18 → ((already_poured * rate) / 60 = 6) :=
by
  intro h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end Aubriella_pouring_time_l344_344542


namespace reciprocal_sum_bound_l344_344225

noncomputable def a_seq (a : ℝ) : ℕ → ℝ
| 0     := 1
| 1     := a
| (n+2) := ( (a_seq (n+1))^2 / (a_seq n)^2 - 2 ) * (a_seq (n+1))

theorem reciprocal_sum_bound (a : ℝ) (k : ℕ) (h : a > 2) :
  (∑ i in Finset.range (k + 1), (1 / a_seq a i)) < 1 / 2 * (2 + a - Real.sqrt (a^2 - 4)) :=
by
  sorry

end reciprocal_sum_bound_l344_344225


namespace find_some_number_l344_344657

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l344_344657


namespace new_plan_cost_correct_l344_344738

-- Define the conditions
def old_plan_cost := 150
def increase_rate := 0.3

-- Define the increased amount
def increase_amount := increase_rate * old_plan_cost

-- Define the cost of the new plan
def new_plan_cost := old_plan_cost + increase_amount

-- Prove the main statement
theorem new_plan_cost_correct : new_plan_cost = 195 :=
by
  sorry

end new_plan_cost_correct_l344_344738


namespace line_O1O2_bisects_AP_l344_344691

variable {A D P E B C O1 O2 : Type} -- Declaring variables

-- Assuming the relevant points and properties
axiom convex_quadrilateral_ADPE (A D P E : Type) : convex (quad A D P E)
axiom angle_ADP_eq_AEP (A D P E : Type) : angle A D P = angle A E P
axiom angle_DPB_eq_EPC (A D P E B C : Type) : angle D P B = angle E P C
axiom circumcenter_O1 (A D E : Type) : is_circumcenter O1 (triangle A D E)
axiom circumcenter_O2 (A B C : Type) : is_circumcenter O2 (triangle A B C)
axiom no_intersection_circumcircles
  (A D E B C : Type) : ¬(intersects (circumcircle (triangle A D E)) (circumcircle (triangle A B C)))

-- The proof statement
theorem line_O1O2_bisects_AP
  (A D P E B C O1 O2 : Type)
  [convex_quadrilateral_ADPE A D P E]
  [angle_ADP_eq_AEP A D P E]
  [angle_DPB_eq_EPC A D P E B C]
  [circumcenter_O1 A D E]
  [circumcenter_O2 A B C]
  [no_intersection_circumcircles A D E B C] :
  bisects (line O1 O2) (segment A P) :=
  sorry

end line_O1O2_bisects_AP_l344_344691


namespace factor_expression_l344_344569

theorem factor_expression (x : ℝ) :
  80 * x ^ 5 - 250 * x ^ 9 = -10 * x ^ 5 * (25 * x ^ 4 - 8) :=
by
  sorry

end factor_expression_l344_344569


namespace find_wsquared_l344_344651

theorem find_wsquared : 
  (2 * w + 10) ^ 2 = (5 * w + 15) * (w + 6) →
  w ^ 2 = (90 + 10 * Real.sqrt 65) / 4 := 
by 
  intro h₀
  sorry

end find_wsquared_l344_344651


namespace sequence_sum_l344_344604

def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else a_n (n / 2)

def S_n (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a_n (i + 1))

theorem sequence_sum :
  S_n (2 ^ 2016 - 1) = (4 ^ 2016 - 1) / 3 :=
by 
  sorry

end sequence_sum_l344_344604


namespace speed_of_man_in_still_water_l344_344466

def upstream_speed := 34 -- in kmph
def downstream_speed := 48 -- in kmph

def speed_in_still_water := (upstream_speed + downstream_speed) / 2

theorem speed_of_man_in_still_water :
  speed_in_still_water = 41 := by
  sorry

end speed_of_man_in_still_water_l344_344466


namespace simplify_sqrt_300_l344_344770

theorem simplify_sqrt_300 :
  ∃ (x : ℝ), sqrt 300 = x * sqrt 3 ∧ x = 10 :=
by
  use 10
  split
  sorry
  rfl

end simplify_sqrt_300_l344_344770


namespace bacteria_doubling_l344_344332

theorem bacteria_doubling (d: ℕ) (n: ℕ) (H1: n = 24) (H2: ∀ t ≤ 24, bacteria_amount t = (1:ℚ) / 64 * 2^(24 - t)) (H3: bacteria_amount 24 = 1 / 64):
  bacteria_amount 30 = 1 :=
by
  -- Proof goes here
  sorry

end bacteria_doubling_l344_344332


namespace binomials_product_l344_344920

noncomputable def poly1 (x y : ℝ) : ℝ := 2 * x^2 + 3 * y - 4
noncomputable def poly2 (y : ℝ) : ℝ := y + 6

theorem binomials_product (x y : ℝ) :
  (poly1 x y) * (poly2 y) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 :=
by sorry

end binomials_product_l344_344920


namespace school_club_profit_l344_344895

def total_cost (num_bars : ℕ) (bulk_discount : ℕ → Nat → ℚ) : ℚ :=
  bulk_discount 8 3 * num_bars / 8

def selling_price_per_bar : ℚ := 2 / 3

def total_revenue (num_bars : ℕ) (price_per_bar : ℚ) : ℚ :=
  price_per_bar * num_bars

def profit (revenue : ℚ) (cost : ℚ) : ℚ :=
  revenue - cost

theorem school_club_profit (num_bars : ℕ) (bulk_discount : ℕ → ℕ → ℚ) 
  (price_per_bar : ℚ) (discount_price_per_bar : ℚ) :
  (discount_price_per_bar = bulk_discount 8 3 / 8) →
  (price_per_bar = selling_price_per_bar) →
  (num_bars = 1200) →
  profit (total_revenue num_bars price_per_bar) (total_cost num_bars bulk_discount) = 350 := 
by
  intros h_discounted_price_per_bar h_price_per_bar h_num_bars
  rw [h_discounted_price_per_bar, h_price_per_bar, h_num_bars]
  -- The actual proof steps would go here.
  -- Skip proof with sorry for now.
  sorry

end school_club_profit_l344_344895


namespace pythagorean_triplet_l344_344463

theorem pythagorean_triplet (k : ℕ) :
  let a := k
  let b := 2 * k - 2
  let c := 2 * k - 1
  (a * b) ^ 2 + c ^ 2 = (2 * k ^ 2 - 2 * k + 1) ^ 2 :=
by
  sorry

end pythagorean_triplet_l344_344463


namespace unique_peg_arrangement_l344_344233

theorem unique_peg_arrangement (triangular_board : Type) [fintype triangular_board]
  (pegs : triangular_board → option ℕ)
  (is_triangular : ∀ (i j : ℕ), i + j < 7) 
  (yellow_pegs : fin 6 → option ℕ)
  (red_pegs : fin 5 → option ℕ)
  (green_pegs : fin 4 → option ℕ)
  (blue_pegs : fin 3 → option ℕ)
  (orange_pegs : fin 2 → option ℕ)
  (purple_peg : fin 1 → option ℕ) :
  (∀ (i j : fin 6), 
     if ((yellow_pegs i).is_some) then
       (pegs (i, j) = yellow_pegs i) ∧
       (∀ k ≠ i, pegs (k, j) ≠ yellow_pegs i) ∧ 
       (∀ l ≠ j, pegs (i, l) ≠ yellow_pegs i))
  ∧
  (∀ (i j : fin 5), 
     if ((red_pegs i).is_some) then
       (pegs (i, j) = red_pegs i) ∧
       (∀ k ≠ i, pegs (k, j) ≠ red_pegs i) ∧ 
       (∀ l ≠ j, pegs (i, l) ≠ red_pegs i))
  ∧
  (∀ (i j : fin 4), 
     if ((green_pegs i).is_some) then
       (pegs (i, j) = green_pegs i) ∧
       (∀ k ≠ i, pegs (k, j) ≠ green_pegs i) ∧ 
       (∀ l ≠ j, pegs (i, l) ≠ green_pegs i))
  ∧
  (∀ (i j : fin 3), 
     if ((blue_pegs i).is_some) then
       (pegs (i, j) = blue_pegs i) ∧
       (∀ k ≠ i, pegs (k, j) ≠ blue_pegs i) ∧ 
       (∀ l ≠ j, pegs (i, l) ≠ blue_pegs i))
  ∧
  (∀ (i j : fin 2), 
     if ((orange_pegs i).is_some) then
       (pegs (i, j) = orange_pegs i) ∧
       (∀ k ≠ i, pegs (k, j) ≠ orange_pegs i) ∧ 
       (∀ l ≠ j, pegs (i, l) ≠ orange_pegs i))
  ∧
  (purple_peg (fintype.of 1), option.is_some purple_peg → 
     pegs (0, 0) = purple_peg) 
  ∧
  ∃! arrangement : (triangular_board → option ℕ),
    (arrangement = pegs) :=
sorry


end unique_peg_arrangement_l344_344233


namespace real_roots_sum_of_polynomials_l344_344345

noncomputable def P1 (A : ℝ) (b c : ℝ) : ℝ → ℝ := λ x, A * (x - b) * (x - c)
noncomputable def P2 (B : ℝ) (a c : ℝ) : ℝ → ℝ := λ x, B * (x - c) * (x - a)
noncomputable def P3 (C : ℝ) (a b : ℝ) : ℝ → ℝ := λ x, C * (x - a) * (x - b)

theorem real_roots_sum_of_polynomials
  (A B C a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (∃ x, P1 A b c x = 0 ∧ P2 B a c x = 0) →
  (∃ x, P2 B a c x = 0 ∧ P3 C a b x = 0) →
  (∃ x, P3 C a b x = 0 ∧ P1 A b c x = 0) →
  (∃ x, (P1 A b c x + P2 B a c x + P3 C a b x) = 0) :=
by
  sorry

end real_roots_sum_of_polynomials_l344_344345


namespace captain_total_coins_3850_l344_344884

-- Define the conditions and the final amount of coins the captain receives.
def initial_coins (x : ℕ) : Prop := x ≥ 120
def captain_initial_take : ℕ := 120
def captain_bonus : ℕ := 45
def pirate_take (remaining : ℕ) (k : ℕ) : ℕ := remaining * k / 15
def all_pirates_take (x : ℕ) : ℕ :=
  let remaining := x - captain_initial_take in
  (remaining * 14 / 15 * 13 / 15 * ... * 2 / 15) -- implied remaining calculation for each pirate.

-- The total coins the captain receives
def captain_received_coins (x : ℕ) : ℕ :=
  captain_initial_take + captain_bonus + all_pirates_take x

-- The theorem we need to prove:
theorem captain_total_coins_3850 (x : ℕ) (h : initial_coins x) :
  captain_received_coins x = 3850 :=
sorry

end captain_total_coins_3850_l344_344884


namespace first_tap_time_l344_344486

noncomputable def fill_rate (T : ℝ) : ℝ := 1 / T
def empty_rate : ℝ := 1 / 8
def net_fill_rate_both : ℝ := 1 / 4.8

theorem first_tap_time (T : ℝ) (h : fill_rate T - empty_rate = net_fill_rate_both) : T = 3 := 
sorry

end first_tap_time_l344_344486


namespace angle_APD_eq_angle_XYZ_l344_344716

variables {A B C D P X Y Z : Type}
variables [add_comm_group A] [vector_space ℝ A]
variables [add_comm_group B] [vector_space ℝ B]
variables [add_comm_group C] [vector_space ℝ C]
variables [add_comm_group D] [vector_space ℝ D]

def is_convex_quadrilateral (ABCD : Type) : Prop :=
  ∃ (A B C D : Type), convex ABCD

def ratio_is_two (A B : Type) (X : Type) : Prop :=
  ∃ (AX XB : ℝ), AX / XB = 2

def tangent_circumcircle (XY : Type) (C Y Z : Type) : Prop :=
  ∃ (circumCircle : Type), tangent XY circumCircle

theorem angle_APD_eq_angle_XYZ (A B C D P X Y Z : Type)
  (H1 : is_convex_quadrilateral ABCD)
  (H2 : P = intersect (diagonal AC) (diagonal BD))
  (H3 : ratio_is_two A B X)
  (H4 : ratio_is_two B C Y)
  (H5 : ratio_is_two C D Z)
  (H6 : tangent_circumcircle XY C Y Z)
  (H7 : tangent_circumcircle YZ B X Y):
  angle APD = angle XYZ :=
sorry

end angle_APD_eq_angle_XYZ_l344_344716


namespace men_entered_room_l344_344709

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l344_344709


namespace number_of_red_socks_l344_344736

-- Definitions:
def red_sock_pairs (R : ℕ) := R
def red_sock_cost (R : ℕ) := 3 * R
def blue_socks_pairs : ℕ := 6
def blue_sock_cost : ℕ := 5
def total_amount_spent := 42

-- Proof Statement
theorem number_of_red_socks (R : ℕ) (h : red_sock_cost R + blue_socks_pairs * blue_sock_cost = total_amount_spent) : 
  red_sock_pairs R = 4 :=
by 
  sorry

end number_of_red_socks_l344_344736


namespace exists_function_239_times_l344_344389

def f (x : ℝ) : ℝ := x / (1 + x / 239)

theorem exists_function_239_times (x : ℝ) (hx : 0 ≤ x) :
  (f^[239] x) = x / (x + 1) :=
sorry

end exists_function_239_times_l344_344389


namespace inner_circle_radius_l344_344825

theorem inner_circle_radius (r : ℝ) (h1 : 0 < r)
  (h2 : let outer_initial := 6 in let outer_new := 6 * 1.5 in
       let inner_new := r * 0.75 in
       let a_initial := π * (outer_initial ^ 2 - r ^ 2) in
       let a_final := π * (outer_new ^ 2 - inner_new ^ 2) in
       a_final = 3.6 * a_initial) :
  r = 4 := by
  sorry

end inner_circle_radius_l344_344825


namespace problem1_problem2_l344_344632

def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

theorem problem1 (x : ℝ) : f x (-1) ≤ 2 ↔ -1 / 2 ≤ x ∧ x ≤ 1 / 2 :=
by sorry

theorem problem2 (a : ℝ) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 1, f x a ≤ |2 * x + 1|) → (0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l344_344632


namespace find_cos_alpha_l344_344290

variable (α β : ℝ)

-- Conditions
def acute_angles (α β : ℝ) : Prop := 0 < α ∧ α < (Real.pi / 2) ∧ 0 < β ∧ β < (Real.pi / 2)
def cos_alpha_beta : Prop := Real.cos (α + β) = 12 / 13
def cos_2alpha_beta : Prop := Real.cos (2 * α + β) = 3 / 5

-- Main theorem
theorem find_cos_alpha (h1 : acute_angles α β) (h2 : cos_alpha_beta α β) (h3 : cos_2alpha_beta α β) : 
  Real.cos α = 56 / 65 :=
sorry

end find_cos_alpha_l344_344290


namespace two_chords_intersect_probability_l344_344564

noncomputable def intersecting_chords_probability (n : ℕ) : ℚ :=
  let favorable_cases := 48
  let total_permutations := n!
  (favorable_cases / total_permutations) ^ 2

theorem two_chords_intersect_probability :
  intersecting_chords_probability 8 = 1 / 705600 :=
by {
  unfold intersecting_chords_probability,
  norm_num,
  sorry
}

end two_chords_intersect_probability_l344_344564


namespace geometric_shapes_sum_l344_344042

noncomputable def Δ : ℝ := sorry
noncomputable def ∘ : ℝ := sorry
noncomputable def ⧈ : ℝ := sorry

-- Assuming conditions from the problem
axiom h1 : 2 * Δ + 2 * ∘ + ⧈ = 27
axiom h2 : 2 * ∘ + Δ + ⧈ = 26
axiom h3 : 2 * ⧈ + Δ + ∘ = 23

-- The goal is to prove this statement
theorem geometric_shapes_sum : 2 * Δ + 3 * ∘ + ⧈ = 45.5 :=
by
  sorry

end geometric_shapes_sum_l344_344042


namespace fraction_to_decimal_l344_344969

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l344_344969


namespace geom_seq_formula_arith_sum_formula_l344_344323

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 * 3^(n-1)

noncomputable def b_n (n : ℕ) : ℕ :=
if n = 0 then 0 else 2 + (n - 1) * -6

theorem geom_seq_formula (n : ℕ) (a2_eq : a_n 2 = 6) (a2_a3_eq : a_n 2 + a_n 3 = 24) :
  a_n n = 2 * 3^(n-1) :=
sorry

theorem arith_sum_formula (n : ℕ) (b1_eq : b_n 1 = 2) (b3_eq : b_n 3 = -10) :
  ∑ i in Finset.range n, b_n (i + 1) = -3 * n^2 + 5 * n :=
sorry

end geom_seq_formula_arith_sum_formula_l344_344323


namespace total_grid_rectangles_l344_344277

-- Define the horizontal and vertical rectangle counting functions
def count_horizontal_rects : ℕ :=
  (1 + 2 + 3 + 4 + 5)

def count_vertical_rects : ℕ :=
  (1 + 2 + 3 + 4)

-- Subtract the overcounted intersection and calculate the total
def total_rectangles (horizontal : ℕ) (vertical : ℕ) (overcounted : ℕ) : ℕ :=
  horizontal + vertical - overcounted

-- Main statement
theorem total_grid_rectangles : count_horizontal_rects + count_vertical_rects - 1 = 24 :=
by
  simp [count_horizontal_rects, count_vertical_rects]
  norm_num
  sorry

end total_grid_rectangles_l344_344277


namespace total_shaded_area_is_sixteen_l344_344894

-- Given conditions
def initial_leg_length : ℝ := 8
def initial_area (leg_length : ℝ) : ℝ :=
  0.5 * leg_length * leg_length

def shaded_area_factor : ℝ := 1 / 3

-- Define the series sum for the shaded areas
noncomputable def shaded_area_total (initial_area : ℝ) (factor : ℝ) : ℝ :=
  initial_area * (factor / (1 - factor))

-- Main theorem: Prove the total shaded area equals 16 cm²
theorem total_shaded_area_is_sixteen :
  let area := initial_area initial_leg_length in
  shaded_area_total area shaded_area_factor = 16 :=
by
  let area := initial_area initial_leg_length
  let total := shaded_area_total area shaded_area_factor
  sorry

end total_shaded_area_is_sixteen_l344_344894


namespace numbering_tube_contacts_l344_344111

theorem numbering_tube_contacts {n : ℕ} (hn : n = 7) :
  ∃ (f g : ℕ → ℕ), (∀ k : ℕ, f k = k % n) ∧ (∀ k : ℕ, g k = (n - k) % n) ∧ 
  (∀ m : ℕ, ∃ k : ℕ, f (k + m) % n = g k % n) :=
by
  sorry

end numbering_tube_contacts_l344_344111


namespace max_min_values_l344_344999

noncomputable def function_y (x : ℝ) : ℝ := 1 - Math.cos x

theorem max_min_values : 
  (∀ k : ℤ, function_y (2 * k * Real.pi + Real.pi) = 2) ∧ 
  (∀ k : ℤ, function_y (2 * k * Real.pi) = 0) :=
by
  sorry

end max_min_values_l344_344999


namespace notebooks_remaining_l344_344854

noncomputable def notebooks_left (total : ℕ) (yeonju_fraction minji_fraction : ℚ) : ℕ :=
  let yeonju := yeonju_fraction * total
  let minji := minji_fraction * total
  total - (yeonju + minji)

theorem notebooks_remaining : 
  notebooks_left 28 (1/4) (3/7) = 9 :=
by
  -- The actual proof would go here
  sorry

end notebooks_remaining_l344_344854


namespace number_of_regions_l344_344038

theorem number_of_regions (m n : ℕ) (h : ∀ (p : ℝ × ℝ), (∃ (l1 l2 : ℕ), l1 ≠ l2 ∧ l1 ≤ m + n ∧ l2 ≤ m + n ∧ (p ∈ (lines.(l1)) ∧ p ∈ (lines.(l2)))) → (l1 ≤ m ∧ l2 ≤ m ∨ (l1 > m ∧ l2 > m))) :
  let non_parallel_regions := 1 + (n * (n + 1)) / 2
  let parallel_contribution := m * (n + 1)
  in regions m n = non_parallel_regions + parallel_contribution := 
sorry

end number_of_regions_l344_344038


namespace no_solution_inequality_l344_344648

theorem no_solution_inequality (a b x : ℝ) (h : |a - b| > 2) : ¬(|x - a| + |x - b| ≤ 2) :=
sorry

end no_solution_inequality_l344_344648


namespace fraction_to_decimal_l344_344975

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344975


namespace evaluate_expression_l344_344190

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l344_344190


namespace bicycle_speed_B_l344_344510

theorem bicycle_speed_B 
  (distance : ℝ := 12)
  (ratio : ℝ := 1.2)
  (time_diff : ℝ := 1 / 6) : 
  ∃ (B_speed : ℝ), B_speed = 12 :=
by
  let A_speed := ratio * B_speed
  have eqn : distance / B_speed - time_diff = distance / A_speed := sorry
  exact ⟨12, sorry⟩

end bicycle_speed_B_l344_344510


namespace roberto_outfits_l344_344765

-- Define the conditions
def trousers := 5
def shirts := 8
def jackets := 4

-- Define the total number of outfits
def total_outfits : ℕ := trousers * shirts * jackets

-- The theorem stating the actual problem and answer
theorem roberto_outfits : total_outfits = 160 :=
by
  -- skip the proof for now
  sorry

end roberto_outfits_l344_344765


namespace second_customer_headphones_l344_344100

theorem second_customer_headphones
  (H : ℕ)
  (M : ℕ)
  (x : ℕ)
  (H_eq : H = 30)
  (eq1 : 5 * M + 8 * H = 840)
  (eq2 : 3 * M + x * H = 480) :
  x = 4 :=
by
  sorry

end second_customer_headphones_l344_344100


namespace fraction_to_decimal_l344_344970

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344970


namespace max_height_of_particle_l344_344890

theorem max_height_of_particle :
  let v₀ := 180 : ℝ
  let θ := Real.pi / 4 -- 45 degrees in radians
  let g := 32 : ℝ
  let v₀y := v₀ * Real.sin θ
  let t := v₀y / g
  let H := v₀y * t - 0.5 * g * t^2
  H = 253.125 := by trivial

end max_height_of_particle_l344_344890


namespace functional_equation_solution_l344_344570

/-- For all functions f: ℝ → ℝ, that satisfy the given functional equation -/
def functional_equation (f: ℝ → ℝ) : Prop :=
  ∀ x y: ℝ, f (x + y * f (x + y)) = y ^ 2 + f (x * f (y + 1))

/-- The solution to the functional equation is f(x) = x -/
theorem functional_equation_solution :
  ∀ f: ℝ → ℝ, functional_equation f → (∀ x: ℝ, f x = x) :=
by
  intros f h x
  sorry

end functional_equation_solution_l344_344570


namespace side_length_equilateral_l344_344399

-- Define our parameters
variables {A B C Q : Point} (t : ℝ)

-- Define the conditions
def is_equilateral (A B C : Point) (t : ℝ) := dist A B = t ∧ dist B C = t ∧ dist C A = t
def distances_to_point (A B C Q : Point) := dist A Q = 2 ∧ dist B Q = 2 * Real.sqrt 2 ∧ dist C Q = 3

-- The theorem to prove
theorem side_length_equilateral (A B C Q : Point) (t : ℝ) 
  (h_eq : is_equilateral A B C t)
  (h_dist : distances_to_point A B C Q) :
  t = Real.sqrt 15 :=
sorry

end side_length_equilateral_l344_344399


namespace apple_tree_total_apples_l344_344133

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l344_344133


namespace car_average_speed_is_60_l344_344091

def average_speed_problem (s : ℝ) : Prop :=
  let v := 60 in
  let time_first_half := (s / 2) / (v + 20) in
  let time_second_half := (s / 2) / (0.8 * v) in
  let total_time := time_first_half + time_second_half in
  v = s / total_time

theorem car_average_speed_is_60 (s : ℝ) : average_speed_problem s := 
  by 
    let v := 60
    let time_first_half := (s / 2) / (v + 20)
    let time_second_half := (s / 2) / (0.8 * v)
    let total_time := time_first_half + time_second_half
    have h1 : v = s / total_time,
    sorry
    exact h1

end car_average_speed_is_60_l344_344091


namespace jelly_bean_ratio_l344_344826

theorem jelly_bean_ratio
  (initial_jelly_beans : ℕ)
  (num_people : ℕ)
  (remaining_jelly_beans : ℕ)
  (amount_taken_by_each_of_last_four : ℕ)
  (total_taken_by_last_four : ℕ)
  (total_jelly_beans_taken : ℕ)
  (X : ℕ)
  (ratio : ℕ)
  (h0 : initial_jelly_beans = 8000)
  (h1 : num_people = 10)
  (h2 : remaining_jelly_beans = 1600)
  (h3 : amount_taken_by_each_of_last_four = 400)
  (h4 : total_taken_by_last_four = 4 * amount_taken_by_each_of_last_four)
  (h5 : total_jelly_beans_taken = initial_jelly_beans - remaining_jelly_beans)
  (h6 : X = total_jelly_beans_taken - total_taken_by_last_four)
  (h7 : ratio = X / total_taken_by_last_four)
  : ratio = 3 :=
by sorry

end jelly_bean_ratio_l344_344826


namespace triangle_ABC_AB_length_l344_344331

theorem triangle_ABC_AB_length :
  ∀ (A B C : Type) [is_triangle A B C], ∀ (angle_A angle_B : ℝ) (side_AC side_BC : ℝ),
  angle_A = 2 * angle_B →
  side_AC = 4 →
  side_BC = 6 →
  (AB_length A B C = 4) ∨ (AB_length A B C = 2 * Real.sqrt 13) :=
by
  intros A B C is_triangle angle_A angle_B side_AC side_BC h1 h2 h3
  sorry

end triangle_ABC_AB_length_l344_344331


namespace fraction_to_decimal_l344_344964

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l344_344964


namespace range_of_c_extreme_values_l344_344671

noncomputable def f (c x : ℝ) : ℝ := x^3 - 2 * c * x^2 + x

theorem range_of_c_extreme_values 
  (c : ℝ) 
  (h : ∃ a b : ℝ, a ≠ b ∧ (3 * a^2 - 4 * c * a + 1 = 0) ∧ (3 * b^2 - 4 * c * b + 1 = 0)) :
  c < - (Real.sqrt 3 / 2) ∨ c > (Real.sqrt 3 / 2) :=
by sorry

end range_of_c_extreme_values_l344_344671


namespace solve_for_a_b_l344_344642

-- Definition of the given condition
def given_condition (a b : ℚ) : Prop :=
  (1 + real.sqrt 2)^(5 : ℕ) = a + ∑ i in finset.range (6), (↑(nat.choose 5 i) * (↑(1)^i : ℚ) * ((real.sqrt 2)^(5 - i) : ℚ))

-- The theorem we want to prove
theorem solve_for_a_b (a b : ℚ) (h : given_condition a b) : a - b = 12 :=
by {
  sorry
}

end solve_for_a_b_l344_344642


namespace points_in_circle_max_distance_l344_344779

open Real

theorem points_in_circle_max_distance {b : ℝ} :
  (∀ (points : fin 6 → ℝ × ℝ), (∀ i, ∥points i∥ ≤ 0.5) → 
  (∃ (i j : fin 6), i ≠ j ∧ dist (points i) (points j) ≤ b)) ↔ b = 0.5 :=
begin
  sorry
end

end points_in_circle_max_distance_l344_344779


namespace monotonic_increasing_interval_of_f_l344_344424

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - x - 1)

theorem monotonic_increasing_interval_of_f :
  ∀ x : ℝ, x ≤ 1/2 → monotone_increasing (f x) :=
sorry

end monotonic_increasing_interval_of_f_l344_344424


namespace adults_not_wearing_blue_is_10_l344_344142

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

end adults_not_wearing_blue_is_10_l344_344142


namespace total_cost_supplies_l344_344899

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end total_cost_supplies_l344_344899


namespace supplies_total_cost_l344_344901

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end supplies_total_cost_l344_344901


namespace student_B_speed_l344_344508

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l344_344508


namespace tetrahedron_inequality_l344_344387

theorem tetrahedron_inequality (V A B C D M : Type) 
    [MetricSpace V] [VectorSpace ℝ V] 
    (S_A S_B S_C S_D : ℝ) (d_A d_B d_C d_D : ℝ) 
    (R_A R_B R_C R_D : ℝ) :
    (S_A * R_A + S_B * R_B + S_C * R_C + S_D * R_D) 
    ≥ 3 * (S_A * d_A + S_B * d_B + S_C * d_C + S_D * d_D) :=
sorry

end tetrahedron_inequality_l344_344387


namespace biking_time_l344_344714

/-- Problem Statement:
June and Julia live 1.5 miles apart. It takes June 6 minutes to ride her bike directly to Julia's house.
From Julia's house, Bernard's house is another 4 miles away.
Prove that at the same biking rate, it will take June 22 minutes to ride from her own house to Bernard's house
after stopping at Julia's.
-/
theorem biking_time (dJJ: ℝ) (tJJ: ℝ) (dJB: ℝ) (v: ℝ) :  dJJ = 1.5  ∧ tJJ = 6 ∧ dJB = 4 ∧ v = dJJ / tJJ → 
  (dJJ + dJB) / v = 22 :=
by
  intros h
  rcases h with ⟨h_dJJ, h_tJJ, h_dJB, h_v⟩
  rw [h_dJJ, h_tJJ, h_dJB, h_v]
  norm_num
  sorry

end biking_time_l344_344714


namespace isosceles_angle_SUM_l344_344831

def Triangle (A B C : Type) : Prop := 
∀ x y z : Type, ∃ a1 a2 a3 : Type, a1 = A ∧ a2 = B ∧ a3 = C

noncomputable def isosceles (T : Triangle) (x y : Type) : Prop :=
∀ a b c : Type, T a b c → (a = x ∨ b = y ∨ c = x ∨ a = y ∨ b = x ∨ c = y)

noncomputable def angle (T : Triangle) (A B C : Type) : ℝ → Prop := 
∀ x : ℝ, T A B C → (∃ y z : ℝ, x + y + z = 180)

theorem isosceles_angle_SUM {A B C D : Type}:
  ∀ (T1 T2 : Triangle), 
  isosceles T1 A C ∧
  isosceles T2 A D ∧
  angle T1 B C 30 ∧ 
  angle T2 A D 150 → 
  angle T1 A B 60 :=
by
  intros T1 T2 h
  sorry

end isosceles_angle_SUM_l344_344831


namespace mask_production_ratio_l344_344879

-- Define the conditions as constants
constant P_March : ℕ := 3000
constant P_July : ℕ := 48000

-- State the problem in Lean 4
theorem mask_production_ratio (x : ℝ) (h1 : P_March * x^4 = P_July) : x = 2 :=
by 
  sorry

end mask_production_ratio_l344_344879


namespace exists_x_in_zero_one_l344_344872

theorem exists_x_in_zero_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  ∃ x ∈ Icc (0 : ℝ) (1 : ℝ), (4 / Real.pi) * (f 1 - f 0) = (1 + x^2) * (f' x) := by
  sorry

end exists_x_in_zero_one_l344_344872


namespace nancy_mow_yard_alone_time_l344_344750

theorem nancy_mow_yard_alone_time :
  ∃ N : ℝ, (Peter_time = 4 ∧ together_time = 1.71428571429) ∧ 
           (1/N + 1/4 = 1/1.71428571429) ∧
           N ≈ 3 :=
begin
  let Peter_time : ℝ := 4,
  let together_time : ℝ := 1.71428571429,
  use 3,
  split,
  {
    split,
    exact Peter_time,
    exact together_time,
  },
  split,
  {
    simp,
    sorry,  -- skip the proof
  },
  {
    simp,
    sorry,  -- skip the proof
  }
end

end nancy_mow_yard_alone_time_l344_344750


namespace fraction_to_decimal_l344_344977

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344977


namespace men_entered_l344_344702

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l344_344702


namespace oreo_shop_ways_l344_344533

theorem oreo_shop_ways (α β : ℕ) (products total_ways : ℕ) :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  (α + β = products) ∧ (products = 4) ∧ (total_ways = 2143) ∧ 
  (α ≤ 2 * total_flavors) ∧ (β ≤ 4 * oreo_flavors) →
  total_ways = 2143 :=
by sorry


end oreo_shop_ways_l344_344533


namespace evaluate_expression_l344_344568

theorem evaluate_expression :
  (2:ℝ) ^ ((0:ℝ) ^ (Real.sin (Real.pi / 2)) ^ 2) + ((3:ℝ) ^ 0) ^ 1 ^ 4 = 2 := by
  -- Given conditions
  have h1 : Real.sin (Real.pi / 2) = 1 := by sorry
  have h2 : (3:ℝ) ^ 0 = 1 := by sorry
  have h3 : (0:ℝ) ^ 1 = 0 := by sorry
  -- Proof omitted
  sorry

end evaluate_expression_l344_344568


namespace common_ratio_geometric_series_l344_344196

theorem common_ratio_geometric_series
  (a₁ a₂ a₃ : ℚ)
  (h₁ : a₁ = 7 / 8)
  (h₂ : a₂ = -14 / 27)
  (h₃ : a₃ = 56 / 81) :
  (a₂ / a₁ = a₃ / a₂) ∧ (a₂ / a₁ = -2 / 3) :=
by
  -- The proof will follow here
  sorry

end common_ratio_geometric_series_l344_344196


namespace birthday_paradox_l344_344641

/-- The pigeonhole principle applied to birthdays:
    Given 365 days in a year, at least 366 people are required to ensure at least two have the same birthday. -/
theorem birthday_paradox (days_in_year : ℕ) (pigeons : ℕ) (h : days_in_year = 365) (h_pigeons : pigeons = 366) :
  ∃ (p : ℕ), p = pigeons ∧ (∀ (birthdays : Fin days_in_year → ℕ → ℕ), ∃ i j, i ≠ j ∧ birthdays days_in_year i = birthdays days_in_year j) :=
begin
  -- This is where the proof would go; we currently state the conditions and conclusion.
  sorry
end

end birthday_paradox_l344_344641


namespace solve_abs_eq_2x_plus_1_l344_344223

theorem solve_abs_eq_2x_plus_1 (x : ℝ) (h : |x| = 2 * x + 1) : x = -1 / 3 :=
by 
  sorry

end solve_abs_eq_2x_plus_1_l344_344223


namespace arrangement_count_eq_220_l344_344043

theorem arrangement_count_eq_220 :
  let total_positions := 12 in
  let benches := 3 in
  let combinations := Nat.choose total_positions benches in
  combinations = 220 := 
by
  let total_positions := 12
  let benches := 3
  have combinations := Nat.choose total_positions benches
  have h_eq: combinations = 220 := sorry
  exact h_eq

end arrangement_count_eq_220_l344_344043


namespace maximum_bunnies_drum_l344_344499

-- Define the conditions as provided in the problem
def drumsticks := ℕ -- Natural number type for simplicity
def drum := ℕ -- Natural number type for simplicity

structure Bunny :=
(drum_size : drum)
(stick_length : drumsticks)

def max_drumming_bunnies (bunnies : List Bunny) : ℕ := 
  -- Actual implementation to find the maximum number of drumming bunnies
  sorry

theorem maximum_bunnies_drum (bunnies : List Bunny) (h_size : bunnies.length = 7) : max_drumming_bunnies bunnies = 6 :=
by
  -- Proof of the theorem
  sorry

end maximum_bunnies_drum_l344_344499


namespace integer_solutions_of_inequality_l344_344588

theorem integer_solutions_of_inequality : 
  let x : ℤ → Prop := λ x, 0 < x ∧ x < 9
  { var : ℤ // x var }.card = 8 :=
sorry

end integer_solutions_of_inequality_l344_344588


namespace files_remaining_correct_l344_344124

-- Definitions for the original number of files
def music_files_original : ℕ := 4
def video_files_original : ℕ := 21
def document_files_original : ℕ := 12
def photo_files_original : ℕ := 30
def app_files_original : ℕ := 7

-- Definitions for the number of deleted files
def video_files_deleted : ℕ := 15
def document_files_deleted : ℕ := 10
def photo_files_deleted : ℕ := 18
def app_files_deleted : ℕ := 3

-- Definitions for the remaining number of files
def music_files_remaining : ℕ := music_files_original
def video_files_remaining : ℕ := video_files_original - video_files_deleted
def document_files_remaining : ℕ := document_files_original - document_files_deleted
def photo_files_remaining : ℕ := photo_files_original - photo_files_deleted
def app_files_remaining : ℕ := app_files_original - app_files_deleted

-- The proof problem statement
theorem files_remaining_correct : 
  music_files_remaining + video_files_remaining + document_files_remaining + photo_files_remaining + app_files_remaining = 28 :=
by
  rw [music_files_remaining, video_files_remaining, document_files_remaining, photo_files_remaining, app_files_remaining]
  exact rfl


end files_remaining_correct_l344_344124


namespace min_omega_value_l344_344609

noncomputable def A := (Real.pi / 6, Real.sqrt 3 / 2)
noncomputable def B := (Real.pi / 4, 1)
noncomputable def C := (Real.pi / 2, 0)

def f (ω x : ℝ) := Real.sin (ω * x)

theorem min_omega_value : 
  ∃ ω > 0, 
    (f ω (A.1) = A.2 ∧ f ω (C.1) = C.2 ∧ f ω (B.1) ≠ B.2) 
  ∨ (f ω (A.1) = A.2 ∧ f ω (B.1) = B.2 ∧ f ω (C.1) ≠ C.2) 
  ∨ (f ω (B.1) = B.2 ∧ f ω (C.1) = C.2 ∧ f ω (A.1) ≠ A.2) 
  ∧ ω = 4 := 
sorry

end min_omega_value_l344_344609


namespace men_entered_l344_344707

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l344_344707


namespace AdultsNotWearingBlue_l344_344139

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end AdultsNotWearingBlue_l344_344139


namespace det_of_A_squared_minus_3A_l344_344635

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 4], ![3, 2]]

theorem det_of_A_squared_minus_3A : det (A * A - 3 • A) = 140 := by
  sorry

end det_of_A_squared_minus_3A_l344_344635


namespace remainder_when_product_divided_by_5_l344_344581

theorem remainder_when_product_divided_by_5 :
  (∏ i in Finset.range 20, (10 * i + 4)) % 5 = 1 :=
by
  sorry

end remainder_when_product_divided_by_5_l344_344581


namespace faster_train_overtakes_slower_l344_344786

-- Definitions for conditions
def speed_denali : ℝ := 50  -- mph
def speed_glacier : ℝ := 70  -- mph
def length_train : ℝ := 1/6  -- miles

-- Relative speed in miles per second
def rel_speed : ℝ := (speed_glacier - speed_denali) / 3600  -- converting to miles per second

-- Combined length of both trains
def total_length : ℝ := 2 * length_train

-- Expected time to overtake in seconds
def expected_time : ℝ := 60

-- Lean theorem statement
theorem faster_train_overtakes_slower :
  (total_length / rel_speed) = expected_time :=
sorry

end faster_train_overtakes_slower_l344_344786


namespace sequence_monotonic_b_gt_neg3_l344_344231

theorem sequence_monotonic_b_gt_neg3 (b : ℝ) :
  (∀ n : ℕ, n > 0 → (n+1)^2 + b*(n+1) > n^2 + b*n) ↔ b > -3 :=
by sorry

end sequence_monotonic_b_gt_neg3_l344_344231


namespace fraction_to_decimal_l344_344952

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l344_344952


namespace find_sum_of_six_least_positive_integers_l344_344357

def tau (n : ℕ) : ℕ := (List.range (n + 1)).count_dvd n

theorem find_sum_of_six_least_positive_integers :
  ∃ (a b c d e f : ℕ), 
  (tau a + tau (a + 1) = 8) ∧
  (tau b + tau (b + 1) = 8) ∧
  (tau c + tau (c + 1) = 8) ∧
  (tau d + tau (d + 1) = 8) ∧
  (tau e + tau (e + 1) = 8) ∧
  (tau f + tau (f + 1) = 8) ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
  a + b + c + d + e + f = 1017 :=
by
  sorry

end find_sum_of_six_least_positive_integers_l344_344357


namespace car_speed_l344_344069

theorem car_speed (v : ℝ) (h₁ : (1/75 * 3600) + 12 = 1/v * 3600) : v = 60 := 
by 
  sorry

end car_speed_l344_344069


namespace problem_statement_l344_344187

-- Defining the problem and its conditions
def x : ℝ := sqrt (15 + sqrt (15 + sqrt (15 + sqrt (15 + ...))))

-- The statement to be proved
theorem problem_statement (hx : x = sqrt (15 + x)) : x = (1 + sqrt 61) / 2 :=
sorry

end problem_statement_l344_344187


namespace distribution_methods_l344_344096

theorem distribution_methods (total_employees translators programmers: ℕ) 
  (total_employees = 8) 
  (translators = 2) 
  (programmers = 3) 
  (∀ dA dB: set ℕ, dA ∪ dB = set.range 8 ∧ dA ∩ dB = ∅ ∧ ∀ tA ∈ dA ∩ (set.range translators) ∧ ∀ tB ∈ dB ∩ (set.range translators), tA ≠ tB ∧ ∀ pA ∈ dA ∩ (set.range programmers) ∧ ∀ pB ∈ dB ∩ (set.range programmers), pA ≠ pB):
  ∃ n, n = 36 :=
by
  sorry

end distribution_methods_l344_344096


namespace count_four_digit_numbers_with_digit_sum_5_count_four_digit_numbers_with_digit_sum_6_count_four_digit_numbers_with_digit_sum_7_l344_344173

open Nat List

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def digit_sum (n : ℕ) : ℕ := (to_digits 10 n).sum

theorem count_four_digit_numbers_with_digit_sum_5 : 
  (finset.filter (λ n, digit_sum n = 5) (finset.filter is_four_digit (finset.range 10000))).card = 35 :=
sorry

theorem count_four_digit_numbers_with_digit_sum_6 : 
  (finset.filter (λ n, digit_sum n = 6) (finset.filter is_four_digit (finset.range 10000))).card = 56 :=
sorry

theorem count_four_digit_numbers_with_digit_sum_7 : 
  (finset.filter (λ n, digit_sum n = 7) (finset.filter is_four_digit (finset.range 10000))).card = 84 :=
sorry

end count_four_digit_numbers_with_digit_sum_5_count_four_digit_numbers_with_digit_sum_6_count_four_digit_numbers_with_digit_sum_7_l344_344173


namespace no_tiling_after_removal_l344_344551

theorem no_tiling_after_removal :
  ∀ (chessboard : list (list (option bool))) (n : ℕ),
  let rows := 8 in let cols := 8 in 
  let topLeftRemoved := (chessboard[0] = chessboard[0].update_nth 0 none) in
  let bottomRightRemoved := (chessboard[rows - 1] = chessboard[rows - 1].update_nth (cols - 1) none) in
  let updatedChessboard := (chessboard.update_nth 0 (chessboard[0].update_nth 0 none)).update_nth (rows - 1) 
                             (chessboard[rows - 1].update_nth (cols - 1) none) in
  ∀ (domino : ℕ → option bool × option bool),
  ∀ (remainingSquares : ℕ),
  rows = 8 ∧ cols = 8 ∧
  remainingSquares = 62 ∧
  (∀ i j, domino (i * cols + j) = some (chessboard[i][j] , chessboard[i][j + 1])
           ∨ domino (i * cols + j) = some (chessboard[i][j] , chessboard[i + 1][j])) → false :=
begin
  sorry
end

end no_tiling_after_removal_l344_344551


namespace determine_y_l344_344597
noncomputable theory

variables {a b y : ℝ}
variables (r : ℝ)
variables (h : (2 * a) ^ (2 * b) = a ^ (2 * b) * y ^ (2 * b))
variables (hp : a > 0) (bp : b > 0)

theorem determine_y : y = 2 :=
by
  sorry

end determine_y_l344_344597


namespace triangle_area_proof_side_a_proof_l344_344700

variables {A B C : RealAngle}
variables {a b c : ℝ} -- sides of the triangle
variables {area : ℝ} -- area of the triangle

-- Given conditions
-- 1. \( \cos \frac{A}{2} = \frac{2\sqrt{5}}{5} \)
def cos_half_A_eq : RealAngle := arccos (2*sqrt 5 / 5)

-- 2. \( \overrightarrow{AB} \cdot \overrightarrow{AC} = 15 \)
def dot_product_eq : ℝ := 15

-- 3. \( \tan B = 2 \)
def tan_B_eq : ℝ := 2

-- Proof goals

-- Goal 1: Prove that the area of \( \triangle ABC \) is 10
theorem triangle_area_proof 
  (h1 : cos (cos_half_A_eq * 2) = (2 * (2 * sqrt 5 / 5)^2 - 1))
  (h2 : b * c * (3/5) = dot_product_eq)
  (h3 : area = 1/2 * b * c * sqrt(1 - (3/5)^2)) :
  area = 10 :=
by sorry

-- Goal 2: Prove that side \( a \) is \( 2\sqrt{5} \)
theorem side_a_proof 
  (h1 : tan B = tan_B_eq)
  (h2 : A + B + C = pi)
  (h3 : b = 5) (h4 : c = 5)
  (h5 : a^2 = b^2 + c^2 - 2 * b * c * (3/5)) :
  a = 2 * sqrt 5 :=
by sorry

end triangle_area_proof_side_a_proof_l344_344700


namespace exponent_of_first_term_l344_344694

theorem exponent_of_first_term (x s m : ℕ) (h_eq : (2 ^ x) * (25 ^ s) = 5 * (10 ^ m)) (h_m : m = 16) : x = 16 :=
begin
  sorry
end

end exponent_of_first_term_l344_344694


namespace max_expression_value_l344_344457

-- Define the expression
def expression (a b c : ℕ) : ℝ :=
  1 / (a + 2010 / (b + 1 / c))

-- State the main theorem
theorem max_expression_value :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  (expression a b c = 1 / 203) :=
sorry

end max_expression_value_l344_344457


namespace length_of_train_is_125_l344_344077

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_sec : ℝ := 5
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def length_train : ℝ := speed_mps * time_sec

theorem length_of_train_is_125 :
  length_train = 125 := 
by
  sorry

end length_of_train_is_125_l344_344077


namespace num_rows_in_display_l344_344102

theorem num_rows_in_display (n : ℕ) (h : ∑ i in finset.range n, (2 * i + 3) = 169) : n = 12 :=
by sorry

end num_rows_in_display_l344_344102


namespace area_triangle_MAB_eq_l344_344316

-- Define the parametric equations for curve C1.
def parametric_eqns_C1 (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 2 + 2 * Real.sin θ)

-- Define the polar equation for curve C2.
def polar_eqn_C2 (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Define the polar equation transformation for curve C1.
def polar_eqn_C1 (θ : ℝ) : ℝ :=
  2 + 2 * Real.sin θ

-- Given a ray θ = π/3 and point M(2,0), and their intersection with curve C1 and C2,
-- prove the area of triangle MAB is 3 - √3.
theorem area_triangle_MAB_eq (θ : ℝ) (hθ : θ = Real.pi / 3) (hM : (2, 0) = (2, 0)) :
  let A := (2 * Real.cos θ, 2 + 2 * Real.sin θ),
      B := (polar_eqn_C2 θ * Real.cos θ, polar_eqn_C2 θ * Real.sin θ),
      M := (2, 0),
      d := 2 * Real.sin (Real.pi / 3),
      ab := (2 * Real.sqrt 3 - 2 * Real.cos (Real.pi / 3)),
      area := 1/2 * ab * d
  in area = 3 - Real.sqrt 3 :=
sorry

end area_triangle_MAB_eq_l344_344316


namespace sum_first_six_terms_arithmetic_sequence_l344_344411

theorem sum_first_six_terms_arithmetic_sequence (a_8 a_9 a_10 : ℤ) (d a_1 : ℤ) :
  a_8 = a_1 + 7 * d → 
  a_9 = a_1 + 8 * d → 
  a_{10} = a_1 + 9 * d → 
  d = a_9 - a_8 → 
  d = a_{10} - a_9 → 
  S_6 = 3 * (2 * a_1 + 5 * d) →
  S_6 = -24 :=
begin
  intros h8 h9 h10 h_d8 h_d9 h_sum,
  sorry
end

end sum_first_six_terms_arithmetic_sequence_l344_344411


namespace find_first_reduction_percentage_l344_344813

variable (P : ℝ) -- The original price of the jacket
variable (x : ℝ) -- The first reduction percentage
variable reduction_during_sale : ∀ x P, P * (1 - x / 100) * 0.75 * 1.7778 = P -- The given condition to restore price

theorem find_first_reduction_percentage
  (h : reduction_during_sale x P)
  : x = 25 :=
  sorry

end find_first_reduction_percentage_l344_344813


namespace square_area_with_circles_l344_344929

theorem square_area_with_circles (r : ℝ) (h_r : r = 10) :
  let d := 2 * r in
  let side_length := 2 * d in
  (side_length ^ 2) = 1600 :=
by
  sorry

end square_area_with_circles_l344_344929


namespace find_min_k_l344_344734

theorem find_min_k :
  ∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 ≤ 8 * (w^6 + x^6 + y^6 + z^6) :=
begin
  intros w x y z,
  sorry
end

end find_min_k_l344_344734


namespace new_plan_cost_correct_l344_344741

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end new_plan_cost_correct_l344_344741


namespace problem_statement_l344_344243

theorem problem_statement (a : ℕ → ℝ) (b : ℕ → ℝ) (B : ℝ) (n : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ n → a i ≥ 0) →
  (∀ i, 1 ≤ i ∧ i < n → a i ≥ a (i + 1)) →
  B = max (abs (b 1)) (max (abs (b 1 + b 2)) (abs (finset.univ.sum (λ i, b i)))) →
  abs (finset.univ.sum (λ i, a i * b i)) ≤ B * a 1 :=
by sorry

end problem_statement_l344_344243


namespace std_dev_represents_magnitude_of_fluctuation_l344_344314

-- Define the standard deviation of a sample
def std_dev (s : List ℝ) : ℝ :=
  let mean := s.sum / s.length
  ((s.map (λ x => (x - mean)^2)).sum / s.length.toFloat) ** 0.5

-- Define the conditions
axiom low_std_dev {s : List ℝ} : std_dev s < 1 → ∀ x ∈ s, (x - s.sum / s.length) < 1
axiom high_std_dev {s : List ℝ} : std_dev s > 1 → ∃ x y ∈ s, (x - s.sum / s.length) > 1 ∧ (y - s.sum / s.length) < -1

-- Define the magnitude of fluctuation concept
def magnitude_of_fluctuation (s : List ℝ) : ℝ := std_dev s

-- Prove the standard deviation of a sample reflects the magnitude of fluctuation
theorem std_dev_represents_magnitude_of_fluctuation (s : List ℝ) : 
  std_dev s = magnitude_of_fluctuation s :=
by
  sorry

end std_dev_represents_magnitude_of_fluctuation_l344_344314


namespace sum_of_infinite_series_eq_four_l344_344896

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 ∨ n = 2 then 1 else (1/3) * sequence (n - 1) + (1/4) * sequence (n - 2)

noncomputable def sum_of_sequence : ℝ :=
  ∑' n, sequence n

theorem sum_of_infinite_series_eq_four : sum_of_sequence = 4 := 
  sorry

end sum_of_infinite_series_eq_four_l344_344896


namespace area_triangle_BXC_l344_344830

/-- Given a trapezoid ABCD with bases AB = 15 units and CD = 35 units, and diagonals AC and BD intersecting at X, if the area of the trapezoid is 375 square units, then the area of triangle BXC is 78.75 square units. -/
theorem area_triangle_BXC (AB CD : ℝ) (area_trapezoid : ℝ) (X : Point) :
  AB = 15 →
  CD = 35 →
  area_trapezoid = 375 →
  ∃ area_BXC, area_BXC = 78.75 :=
by
  intros h1 h2 h3
  use 78.75
  sorry

end area_triangle_BXC_l344_344830


namespace lcm_of_28_and_24_is_168_l344_344383

/-- Racing car A completes the track in 28 seconds.
    Racing car B completes the track in 24 seconds.
    Both cars start at the same time.
    We want to prove that the time after which both cars will be side by side again
    (least common multiple of their lap times) is 168 seconds. -/
theorem lcm_of_28_and_24_is_168 :
  Nat.lcm 28 24 = 168 :=
sorry

end lcm_of_28_and_24_is_168_l344_344383


namespace intersection_points_on_hyperbola_l344_344211

theorem intersection_points_on_hyperbola (p x y : ℝ) :
  (2*p*x - 3*y - 4*p = 0) ∧ (4*x - 3*p*y - 6 = 0) → 
  (∃ a b : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1) :=
by
  intros h
  sorry

end intersection_points_on_hyperbola_l344_344211


namespace sum_of_tens_place_of_hump_numbers_l344_344329

def is_hump_number (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
    (n >= 100) ∧ (n < 1000) ∧ (digits.nth 1 < digits.nth 0) ∧ (digits.nth 1 < digits.nth 2)

def possible_digits := {1, 2, 3, 4, 5}

noncomputable def hump_numbers : List ℕ :=
  (possible_digits.toFinset.powerset.filter (λ s, s.card = 3)).val.flat_map (λ s, 
    let [a, b, c] := s.val.toList in
    [ a * 100 + b * 10 + c, a * 100 + c * 10 + b, b * 100 + a * 10 + c,
      b * 100 + c * 10 + a, c * 100 + a * 10 + b, c * 100 + b * 10 + a ].filter is_hump_number)

def sum_of_tens_place (ns : List ℕ) : ℕ :=
  ns.map (λ n, n / 10 % 10).sum

theorem sum_of_tens_place_of_hump_numbers :
  sum_of_tens_place hump_numbers = 30 := sorry

end sum_of_tens_place_of_hump_numbers_l344_344329


namespace unerased_number_in_range_l344_344824

theorem unerased_number_in_range 
  (n : ℕ) 
  (h1 : 8034 ≤ n) 
  (h2 : n ≤ 8038) 
  (total_sum : ℕ := ∑ i in finset.range 21, 1999 + i)
  (remaining : ℕ := total_sum - 5 * n) :
  1999 ≤ remaining ∧ remaining ≤ 2019 :=
begin
  sorry
end

end unerased_number_in_range_l344_344824


namespace initial_number_of_small_bottles_l344_344897

def number_of_big_bottles := 10000
def small_bottle_sold_rate := 0.12
def big_bottle_sold_rate := 0.15
def total_remaining_bottles := 13780

theorem initial_number_of_small_bottles (S : ℕ) 
    (h1 : S * (1 - small_bottle_sold_rate) + number_of_big_bottles * (1 - big_bottle_sold_rate) = total_remaining_bottles) :
    S = 6000 :=
by {
  rw [small_bottle_sold_rate, big_bottle_sold_rate, number_of_big_bottles, total_remaining_bottles] at h1,
  sorry
}

end initial_number_of_small_bottles_l344_344897


namespace number_of_math_students_l344_344540

-- Definitions for the problem conditions
variables (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
variable (total_students_eq : total_students = 100)
variable (both_classes_eq : both_classes = 10)
variable (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))

-- Theorem statement
theorem number_of_math_students (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
  (total_students_eq : total_students = 100)
  (both_classes_eq : both_classes = 10)
  (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))
  (total_students_eq : total_students = physics_class + math_class - both_classes) :
  math_class = 88 :=
sorry

end number_of_math_students_l344_344540


namespace expression_equals_500_l344_344062

theorem expression_equals_500 :
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  C = 500 :=
by
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  sorry

end expression_equals_500_l344_344062


namespace fraction_to_decimal_l344_344954

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l344_344954


namespace round_to_nearest_tenth_l344_344766

theorem round_to_nearest_tenth : 
  let x := 3967149.6587234 in
  Real.round_to_tenth x = 3967149.7 :=
by
  sorry

end round_to_nearest_tenth_l344_344766


namespace schur_theorem_variant_l344_344400

theorem schur_theorem_variant
  (M : Finset ℕ)
  (t : ℕ)
  (M_i : Fin t → Finset ℕ)
  (m_i : Fin t → ℕ)
  (h_partition : ∀ i : Fin t, (M_i i) ⊆ M)
  (h_cardinality : ∀ i : Fin t, (M_i i).card = m_i i)
  (h_order : ∀ i : Fin (t-1), m_i i ≥ m_i ⟨i+1, by simp [i.is_lt]⟩)
  (h_n_gt_te : M.card > nat.factorial t * Real.exp 1) :
  ∃ z : Fin t, ∃ x_i x_j x_k ∈ M_i z, x_i - x_j = x_k := sorry

end schur_theorem_variant_l344_344400


namespace find_C_coordinates_l344_344758

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 11, y := 9 }
def B : Point := { x := 2, y := -3 }
def D : Point := { x := -1, y := 3 }

-- Define the isosceles property
def is_isosceles (A B C : Point) : Prop :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2) = Real.sqrt ((A.x - C.x) ^ 2 + (A.y - C.y) ^ 2)

-- Define the midpoint property
def is_midpoint (D B C : Point) : Prop :=
  D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

theorem find_C_coordinates (C : Point)
  (h_iso : is_isosceles A B C)
  (h_mid : is_midpoint D B C) :
  C = { x := -4, y := 9 } := 
  sorry

end find_C_coordinates_l344_344758


namespace basketball_shooting_probability_l344_344898

theorem basketball_shooting_probability (n : ℕ) : (1/2 : ℝ) ^ n < 0.1 → n ≥ 4 :=
begin
  sorry
end

end basketball_shooting_probability_l344_344898


namespace sum_of_smallest_n_l344_344353

def tau (n : ℕ) : ℕ := n.divisors.count

theorem sum_of_smallest_n (h : ∑ (k : ℕ) in (Finset.filter 
  (λ n, tau n + tau (n + 1) = 8) (Finset.range 1000)).sort (≤) (Finset.range 6) = 73) : 
  ℕ := sorry

end sum_of_smallest_n_l344_344353


namespace find_positive_integer_pairs_l344_344571

theorem find_positive_integer_pairs :
  ∀ (m n : ℕ), m > 0 ∧ n > 0 → ∃ k : ℕ, (2^n - 13^m = k^3) ↔ (m = 2 ∧ n = 9) :=
by
  sorry

end find_positive_integer_pairs_l344_344571


namespace rectangles_in_grid_l344_344283

theorem rectangles_in_grid : 
  let horizontal_strip := 1 * 5 in
  let vertical_strip := 1 * 4 in
  ∃ (rectangles : ℕ), rectangles = 24 := 
  by
  sorry

end rectangles_in_grid_l344_344283


namespace length_NK_l344_344302

-- Definition of the conditions
variables {DE EF FD NK : ℝ}
variable {N : Point} -- Assuming a Point type is defined somewhere
variable {K : Foot} -- Assuming a Foot type is defined somewhere
variable {D E F : Point}

-- Given conditions translated into assumptions
axiom H1 : DE = 15
axiom H2 : EF = 18
axiom H3 : FD = 21
axiom H4 : midpoint N D E -- Assuming midpoint takes 3 arguments, the point and the endpoints
axiom H5 : altitude_foot K D EF -- Assuming altitude_foot takes 3 arguments, the foot, the point, and the line

-- Problem statement in Lean: Prove that the length of NK is 19.36
theorem length_NK (H6 : distance N K = 19.36) : 
  NK = 19.36 :=
by sorry

end length_NK_l344_344302


namespace no_reverse_order_l344_344870

theorem no_reverse_order (n : ℕ) :
  ¬ ∃ f : fin n → fin n, (∀ i j : fin n, (i.val + 2 = j.val ∨ j.val + 2 = i.val) → (f i).val = j.val) ∧
    ∀ i : fin n, f i = ⟨n - i.val - 1, sorry⟩ :=
sorry

end no_reverse_order_l344_344870


namespace sqrt_of_300_l344_344771

theorem sqrt_of_300 : sqrt 300 = 10 * sqrt 3 := by
  have h300 : 300 = 2 ^ 2 * 3 * 5 ^ 2 := by norm_num
  rw [h300, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_pow, Real.sqrt_pow]
  all_goals { norm_num, apply_instance }
  sorry

end sqrt_of_300_l344_344771


namespace sum_of_solutions_l344_344202

theorem sum_of_solutions (x : ℝ) : 
  (∑ x in {3 | (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2) ∧ (∀ x : ℝ, -12 * x / (x^2 - 4) = 3 * x / (x + 2) - 9 / (x - 2))}, x) = 3 := 
sorry

end sum_of_solutions_l344_344202


namespace ticket_cost_is_correct_l344_344464

-- Conditions
def total_amount_raised : ℕ := 620
def number_of_tickets_sold : ℕ := 155

-- Definition of cost per ticket
def cost_per_ticket : ℕ := total_amount_raised / number_of_tickets_sold

-- The theorem to be proven
theorem ticket_cost_is_correct : cost_per_ticket = 4 :=
by
  sorry

end ticket_cost_is_correct_l344_344464


namespace some_number_value_correct_l344_344658

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l344_344658


namespace pentagon_AE_length_l344_344728

/-- Let \(ABCDE\) be a convex pentagon such that \(\angle ABC = \angle ACD = \angle ADE = 90^\circ\) and 
\(AB = BC = CD = DE = 1\). Prove that the length \(AE\) is \(2\). -/
theorem pentagon_AE_length :
  ∃ (A B C D E : ℝ × ℝ),
    A = (0, 0) ∧
    B = (1, 0) ∧
    C = (1, 1) ∧
    D = (0, 1) ∧
    E = (0, 2) ∧
    ∠ABC = 90 ∧
    ∠ACD = 90 ∧
    ∠ADE = 90 ∧
    dist A B = 1 ∧
    dist B C = 1 ∧
    dist C D = 1 ∧
    dist D E = 1 ∧
    dist A E = 2 :=
begin
  sorry
end

end pentagon_AE_length_l344_344728


namespace polygon_area_correct_l344_344682

-- Define the conditions from the problem
def edge_length : ℝ := 30
def AP : ℝ := 10
def PB : ℝ := 20
def BQ : ℝ := 20
def CR : ℝ := 5

-- Define the points P, Q, and R based on given conditions
def P : ℝ × ℝ × ℝ := (10, 0, 0)
def Q : ℝ × ℝ × ℝ := (30, 20, 0)
def R : ℝ × ℝ × ℝ := (30, 30, 5)

-- Define the equation of the plane passing through points P, Q, and R
def plane_eq (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := P
  let (x2, y2, z2) := Q
  let (x3, y3, z3) := R
  let a := (y1 - y2) * (z1 - z3) - (z1 - z2) * (y1 - y3)
  let b := (z1 - z2) * (x1 - x3) - (x1 - x2) * (z1 - z3)
  let c := (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
  let d := - (a * x1 + b * y1 + c * z1)
  (a, b, c, d)

-- Assert the properties we're proving
theorem polygon_area_correct :
  let plane := plane_eq P Q R in
  let intersections := sorry in -- Compute intersections (details omitted)
  let area := sorry in -- Calculate the area (details omitted)
  area = 450 := 
sorry

end polygon_area_correct_l344_344682


namespace simplify_sqrt_300_l344_344775

theorem simplify_sqrt_300 :
  sqrt 300 = 10 * sqrt 3 :=
by
  -- Proof would go here
  sorry

end simplify_sqrt_300_l344_344775


namespace fraction_to_decimal_l344_344953

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l344_344953


namespace fraction_to_decimal_l344_344957

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l344_344957


namespace erica_fish_catch_l344_344566

theorem erica_fish_catch (X : ℕ) : 
  (20 * X) + (20 * (2 * X)) = 4800 → X = 80 :=
by
  intro h
  calc
    20 * X + 20 * (2 * X) = 4800 : h
    60 * X = 4800             : by ring
    X = 80                    : by exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end erica_fish_catch_l344_344566


namespace centroid_of_triangle_is_intersection_of_medians_l344_344044

open EuclideanGeometry

theorem centroid_of_triangle_is_intersection_of_medians (ABC : Triangle) :
  ∃ G : Point, is_centroid G ABC ∧ G ∈ line_medians_intersection ABC :=
sorry

def is_centroid (G : Point) (ABC : Triangle) : Prop :=
  is_balanced_at G ABC

def is_balanced_at (G : Point) (ABC : Triangle) : Prop :=
  -- Definition of balancing at point G for triangle ABC;

def line_medians_intersection (ABC : Triangle) : Set Point :=
  {G | ∃ (M : Median), G ∈ intersection_Median_lines ABC}

def intersection_Median_lines (ABC : Triangle) : Set Point :=
  -- Definition of the set of points where medians intersect in triangle ABC

end centroid_of_triangle_is_intersection_of_medians_l344_344044


namespace cost_of_dozen_pens_l344_344860

-- Define the costs and conditions as given in the problem.
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x

-- The given conditions transformed into Lean definitions.
def condition1 (x : ℝ) : Prop := 3 * cost_of_pen x + 5 * cost_of_pencil x = 100
def condition2 (x : ℝ) : Prop := cost_of_pen x / cost_of_pencil x = 5

-- Prove that the cost of one dozen pens is Rs. 300.
theorem cost_of_dozen_pens : ∃ x : ℝ, condition1 x ∧ condition2 x ∧ 12 * cost_of_pen x = 300 := by
  sorry

end cost_of_dozen_pens_l344_344860


namespace intersection_points_C1_C2_in_polar_l344_344262

def equation_C1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 25

def equation_C2 (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

def polar_coordinates (x y : ℝ) : (ℝ × ℝ) := 
  let ρ := real.sqrt (x^2 + y^2)
  let θ := real.atan2 y x
  (ρ, θ)

theorem intersection_points_C1_C2_in_polar :
  ∃ (ρ1 θ1 ρ2 θ2 : ℝ), (ρ1 > 0 ∧ θ1 ≥ 0 ∧ θ1 < 2*real.pi) ∧ (ρ2 > 0 ∧ θ2 ≥ 0 ∧ θ2 < 2*real.pi) ∧ 
  (polar_coordinates 0 2 = (ρ1, θ1)) ∧ (polar_coordinates 1 1 = (ρ2, θ2)) ∧ 
  (equation_C1 0 2) ∧ (equation_C1 1 1) ∧
  (equation_C2 0 2) ∧ (equation_C2 1 1) :=
by
  use [2, real.pi / 2, real.sqrt 2, real.pi / 4]
  split; try { split; try { linarith } }
  split; try { split; try { linarith } }
  { simp [polar_coordinates, real.sqrt, real.atan2, equation_C1, equation_C2] }
  sorry

end intersection_points_C1_C2_in_polar_l344_344262


namespace triangle_area_PQR_l344_344840

def point := (ℝ × ℝ)

def P : point := (2, 3)
def Q : point := (7, 3)
def R : point := (4, 10)

noncomputable def triangle_area (A B C : point) : ℝ :=
  (1/2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_PQR : triangle_area P Q R = 17.5 :=
  sorry

end triangle_area_PQR_l344_344840


namespace plane_through_Ox_and_point_plane_parallel_Oz_and_points_l344_344199

-- Definitions for first plane problem
def plane1_through_Ox_axis (y z : ℝ) : Prop := 3 * y + 2 * z = 0

-- Definitions for second plane problem
def plane2_parallel_Oz (x y : ℝ) : Prop := x + 3 * y - 1 = 0

theorem plane_through_Ox_and_point : plane1_through_Ox_axis 2 (-3) := 
by {
  -- Hint: Prove that substituting y = 2 and z = -3 in the equation results in LHS equals RHS.
  -- proof
  sorry 
}

theorem plane_parallel_Oz_and_points : 
  plane2_parallel_Oz 1 0 ∧ plane2_parallel_Oz (-2) 1 :=
by {
  -- Hint: Prove that substituting the points (1, 0) and (-2, 1) in the equation results in LHS equals RHS.
  -- proof
  sorry
}

end plane_through_Ox_and_point_plane_parallel_Oz_and_points_l344_344199


namespace city_growth_rate_order_l344_344442

theorem city_growth_rate_order 
  (Dover Eden Fairview : Type) 
  (highest lowest : Type)
  (h1 : Dover = highest → ¬(Eden = highest) ∧ (Fairview = lowest))
  (h2 : ¬(Dover = highest) ∧ Eden = highest ∧ Fairview = lowest → Eden = highest ∧ Dover = lowest ∧ Fairview = highest)
  (h3 : ¬(Fairview = lowest) → ¬(Eden = highest) ∧ ¬(Dover = highest)) : 
  Eden = highest ∧ Dover = lowest ∧ Fairview = highest ∧ Eden ≠ lowest :=
by
  sorry

end city_growth_rate_order_l344_344442


namespace find_some_number_l344_344656

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l344_344656


namespace intersecting_ellipse_line_l344_344606

theorem intersecting_ellipse_line (m : ℝ) :
  (∃ (x y : ℝ), (x + 2 * y = 2) ∧ (x^2 / 3 + y^2 / m = 1)) ↔ m ∈ set.Ioo (1/12) 3 ∪ set.Ioi 3 :=
begin
  sorry
end

end intersecting_ellipse_line_l344_344606


namespace mark_new_phone_plan_cost_l344_344744

theorem mark_new_phone_plan_cost (old_cost : ℕ) (h_old_cost : old_cost = 150) : 
  let new_cost := old_cost + (0.3 * old_cost) in 
  new_cost = 195 :=
by 
  sorry

end mark_new_phone_plan_cost_l344_344744


namespace company_women_percentage_l344_344105

theorem company_women_percentage
  (initial_workers_A : ℕ := 90)
  (initial_men_A : ℕ := 60)
  (initial_women_A : ℕ := 30)
  (initial_workers_B : ℕ := 150)
  (initial_men_B : ℕ := 60)
  (initial_women_B : ℕ := 90)
  (new_hires_A : ℕ := 5)
  (new_men_hires_A : ℕ := 3)
  (new_women_hires_A : ℕ := 2)
  (new_hires_B : ℕ := 8)
  (new_men_hires_B : ℕ := 4)
  (new_women_hires_B : ℕ := 4)
  (transfer_A_to_B : ℕ := 10)
  (transfer_men_A_to_B : ℕ := 6)
  (transfer_women_A_to_B : ℕ := 4)
  (men_attrition_rate_A : ℚ := 0.1)
  (women_attrition_rate_B : ℚ := 0.05) :
  let final_men_A := initial_men_A + new_men_hires_A - transfer_men_A_to_B - (initial_men_A + new_men_hires_A - transfer_men_A_to_B) * men_attrition_rate_A
  let final_women_A := initial_women_A + new_women_hires_A - transfer_women_A_to_B
  let final_men_B := initial_men_B + new_men_hires_B + transfer_men_A_to_B
  let final_women_B := initial_women_B + new_women_hires_B + transfer_women_A_to_B - (initial_women_B + new_women_hires_B + transfer_women_A_to_B) * women_attrition_rate_B
  let total_final_workers := final_men_A + final_women_A + final_men_B + final_women_B
  let total_final_women := final_women_A + final_women_B
  (total_final_women / total_final_workers) = 0.5 :=
begin
  sorry
end

end company_women_percentage_l344_344105


namespace verify_three_smallest_balanced_numbers_verify_num_balanced_below_2014_l344_344110

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 1000 in
  let d2 := (n % 1000) / 100 in
  let d3 := (n % 100) / 10 in
  let d4 := n % 10 in
  d1 * 4 == d1 + d2 + d3 + d4 ∨
  d2 * 4 == d1 + d2 + d3 + d4 ∨
  d3 * 4 == d1 + d2 + d3 + d4 ∨
  d4 * 4 == d1 + d2 + d3 + d4

def three_smallest_balanced_numbers : set ℕ :=
  {1003, 1012, 1021}

def num_balanced_below_2014 : ℕ :=
  90

theorem verify_three_smallest_balanced_numbers :
  {n : ℕ | is_balanced n} ∩ {n | n < 2014} = three_smallest_balanced_numbers :=
sorry

theorem verify_num_balanced_below_2014 :
  {n : ℕ | is_balanced n ∧ n < 2014}.card = num_balanced_below_2014 :=
sorry

end verify_three_smallest_balanced_numbers_verify_num_balanced_below_2014_l344_344110


namespace problem_incorrect_statement_l344_344266

def A : set ℕ := {x | x < 5}

theorem problem_incorrect_statement : ¬(5 ∈ A) :=
by {
  sorry
}

end problem_incorrect_statement_l344_344266


namespace remainder_27_pow_482_div_13_l344_344056

theorem remainder_27_pow_482_div_13 :
  27^482 % 13 = 1 :=
sorry

end remainder_27_pow_482_div_13_l344_344056


namespace units_digit_7_pow_5_l344_344060

theorem units_digit_7_pow_5 : (7^5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l344_344060


namespace ring_inner_circumference_l344_344048

theorem ring_inner_circumference (C2 : ℝ) (W : ℝ) (hC2 : C2 = 528 / 7) (hW : W = 4.001609997739084) : ∃ C1 : ℝ, C1 ≈ 211.6194331982906 :=
by
  -- Calculation details skipped. Only the statement is provided.
  sorry

end ring_inner_circumference_l344_344048


namespace who_cleaned_classroom_l344_344530

def xiao_hong : Prop := Xiao_Qiang_did_it
def xiao_qiang : Prop := ¬Qiang_did_it
def xiao_hua : Prop := ¬Hua_did_it

axiom xiao_statements : (xiao_hong ∧ ¬xiao_qiang ∧ ¬xiao_hua) ∨ (¬xiao_hong ∧ xiao_qiang ∧ ¬xiao_hua) ∨ (¬xiao_hong ∧ ¬xiao_qiang ∧ xiao_hua)

theorem who_cleaned_classroom : Xiao_Hua_cleaned_the_classroom :=
sorry

end who_cleaned_classroom_l344_344530


namespace smallest_positive_phi_symmetric_y_axis_l344_344416

theorem smallest_positive_phi_symmetric_y_axis :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = sin (2 * x) + cos (2 * x)) →
  (∃ (ϕ : ℝ), ϕ > 0 ∧ (∀ x, f x = x - ϕ) → (∀ x, f x = f (-x)) → ϕ = (3 * π) / 8) :=
by
  intro f h₁
  use (3 * π) / 8
  sorry

end smallest_positive_phi_symmetric_y_axis_l344_344416


namespace min_distance_PQ_l344_344324

theorem min_distance_PQ :
  let P_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = 4
  let Q_circle (ρ θ : ℝ) := ρ^2 = 4 * ρ * Real.cos θ - 3
  ∃ (P Q : ℝ × ℝ), 
    (∃ ρP θP, P = (ρP * Real.cos θP, ρP * Real.sin θP) ∧ P_line ρP θP) ∧
    (∃ ρQ θQ, Q = (ρQ * Real.cos θQ, ρQ * Real.sin θQ) ∧ Q_circle ρQ θQ) ∧
    ∀ R S : ℝ × ℝ, 
      (∃ ρR θR, R = (ρR * Real.cos θR, ρR * Real.sin θR) ∧ P_line ρR θR) →
      (∃ ρS θS, S = (ρS * Real.cos θS, ρS * Real.sin θS) ∧ Q_circle ρS θS) →
      dist P Q ≤ dist R S :=
  sorry

end min_distance_PQ_l344_344324


namespace pairwise_distances_inequality_l344_344753

noncomputable def sum_pairwise_distances (points : List ℝ) : ℝ :=
  points.combinations 2 |>.sum (λ pair, (pair.head - pair.tail.head).abs)

theorem pairwise_distances_inequality {n : ℕ} (blue_points red_points : Fin n → ℝ) :
  sum_pairwise_distances (List.ofFn blue_points) + sum_pairwise_distances (List.ofFn red_points) ≤
  sum_pairwise_distances (List.ofFn (λ i => blue_points i)) + 
  sum_pairwise_distances (List.ofFn (λ i => red_points i))) := sorry

end pairwise_distances_inequality_l344_344753


namespace sum_of_first_ten_nicely_odd_numbers_is_775_l344_344167

def is_nicely_odd (n : ℕ) : Prop :=
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ (Odd p ∧ Odd q) ∧ n = p * q)
  ∨ (∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ n = p ^ 3)

theorem sum_of_first_ten_nicely_odd_numbers_is_775 :
  let nicely_odd_nums := [15, 27, 21, 35, 125, 33, 77, 343, 55, 39]
  ∃ (nums : List ℕ), List.length nums = 10 ∧
  (∀ n ∈ nums, is_nicely_odd n) ∧ List.sum nums = 775 := by
  sorry

end sum_of_first_ten_nicely_odd_numbers_is_775_l344_344167


namespace simplify_and_evaluate_expression_l344_344395

theorem simplify_and_evaluate_expression :
  (2 * (-1/2) + 3 * 1)^2 - (2 * (-1/2) + 1) * (2 * (-1/2) - 1) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l344_344395


namespace parabola_properties_l344_344435

-- Define the quadratic function for the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- The proof problem in Lean 4 statement
theorem parabola_properties :
  (vertex of parabola = (5/4, -1/8)) ∧
  (number of intersection points with coordinate axes = 3) :=
by
  sorry

end parabola_properties_l344_344435


namespace staircase_problem_l344_344936

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def staircase_sums (n a b : ℕ) : ℕ :=
  if 3 * n + 5 * a - 2 * b = 190 ∧
    (a = 0 ∨ a = 1) ∧
    (b ∈ [0, 1, 2, 3, 4]) ∧
    (2 * b - 5 * a) % 3 = 2
  then n
  else 0

def possible_steps_sum : ℕ :=
  staircase_sums 64 0 1 + staircase_sums 66 0 4 + staircase_sums 63 1 2

theorem staircase_problem : sum_of_digits possible_steps_sum = 13 := by
  sorry

end staircase_problem_l344_344936


namespace cost_of_gas_l344_344391

variable (start_odom end_odom : ℕ) (fuel_efficiency : ℕ) (gas_price : ℚ)

theorem cost_of_gas (h_start: start_odom = 85412) (h_end: end_odom = 85443) (h_efficiency: fuel_efficiency = 25) (h_price: gas_price = 3.95) 
: ((end_odom - start_odom : ℚ) / fuel_efficiency * gas_price).round(2) = 4.90 :=
sorry

end cost_of_gas_l344_344391


namespace sixth_number_is_34_l344_344003

theorem sixth_number_is_34
  (A : Fin 11 → ℤ)
  (h_avg_11 : (∑ i in Finset.univ, A i) = 22 * 11)
  (h_avg_first_6 : (Finset.range 6).sum A = 19 * 6)
  (h_avg_last_6 : (Finset.Ico 5 11).sum A = 27 * 6) :
  A 5 = 34 :=
by
  sorry

end sixth_number_is_34_l344_344003


namespace hyperbola_eccentricity_l344_344698

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), ((x^2 / a^2) - (y^2 / 5) = 1) ∧ (y = (√5 / 2) * x)) → 
  (a = 2) ∧ (let b := √5 in (e : ℝ) = (√ (a^2 + b^2)) / a) → e = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l344_344698


namespace factorize_x9_minus_512_l344_344155

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l344_344155


namespace student_B_speed_l344_344506

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l344_344506


namespace sin_270_eq_neg_one_l344_344158

theorem sin_270_eq_neg_one : 
  let Q := (0, -1) in
  ∃ (θ : ℝ), θ = 270 * Real.pi / 180 ∧ ∃ (Q : ℝ × ℝ), 
    Q = ⟨Real.cos θ, Real.sin θ⟩ ∧ Real.sin θ = -1 :=
by 
  sorry

end sin_270_eq_neg_one_l344_344158


namespace circumcircle_radius_l344_344304

/-- In triangle ABC:
- angle A is 30 degrees,
- side AB is 2,
- the area of triangle ABC is sqrt(3),
Prove that the radius of the circumcircle of triangle ABC is 2. --/
theorem circumcircle_radius (A B C : Type) [metric_space A] 
{a b c : ℝ}
(angle_A : ∡ A = 30)
(side_AB : a = 2)
(area_ABC : 1/2 * b * a * (sin 30) = sqrt 3) :
radius_circumcircle A B C = 2 :=
sorry

end circumcircle_radius_l344_344304


namespace sum_of_powers_modulo_seven_l344_344033

theorem sum_of_powers_modulo_seven :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7) = 1 := by
  sorry

end sum_of_powers_modulo_seven_l344_344033


namespace train_speed_calculation_l344_344521

theorem train_speed_calculation
  (time_crossing_pole : ℝ)
  (length_of_train : ℝ)
  (time_crossing_pole_eq : time_crossing_pole = 5)
  (length_of_train_eq : length_of_train = 125.01) :
  (length_of_train / 1000) / (time_crossing_pole / 3600) = 90.0072 :=
by
  rw [length_of_train_eq, time_crossing_pole_eq]
  norm_num
  sorry

end train_speed_calculation_l344_344521


namespace collinear_condition_l344_344349

-- Definitions of vectors a and b, and that they are non-collinear
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (a b : V)
variables (a_non_collinear_b : a ≠ 0 ∧ b ≠ 0 ∧ ¬ collinear ℝ (![a, b]))

-- Definitions of vectors AB and BC
def vecAB := a - 2 • b
def vecBC (k : ℝ) := 3 • a + k • b

-- Theorem stating the collinearity condition
theorem collinear_condition (k : ℝ) : 
  collinear ℝ (![vecAB a b, vecBC a b k]) ↔ k = -6 :=
sorry

end collinear_condition_l344_344349


namespace calculate_area_shaded_region_l344_344994

open Real

def line (p1 p2 : ℝ × ℝ) : ℝ → ℝ :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1)
  λ x => m * (x - p1.1) + p1.2

def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem calculate_area_shaded_region :
  let l1 := line (0, 5) (6, 3) 
  let l2 := line (2, 6) (9, 1)
  let intersection_x := 6.375  -- The x-coordinate of the intersection point
  let intersection_y := 2.875  -- The y-coordinate of the intersection point
  area_under_curve (λ x => l2 x - l1 x) 0 intersection_x = 3.1875 :=
begin
  sorry
end

end calculate_area_shaded_region_l344_344994


namespace exists_h_not_divisible_l344_344948

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end exists_h_not_divisible_l344_344948


namespace volume_ratio_of_frustum_l344_344883

theorem volume_ratio_of_frustum
  (h_s h : ℝ)
  (A_s A : ℝ)
  (V_s V : ℝ)
  (ratio_lateral_area : ℝ)
  (ratio_height : ℝ)
  (ratio_base_area : ℝ)
  (H_lateral_area: ratio_lateral_area = 9 / 16)
  (H_height: ratio_height = 3 / 5)
  (H_base_area: ratio_base_area = 9 / 25)
  (H_volume_small: V_s = 1 / 3 * h_s * A_s)
  (H_volume_total: V = 1 / 3 * h * A - 1 / 3 * h_s * A_s) :
  V_s / V = 27 / 98 :=
by
  sorry

end volume_ratio_of_frustum_l344_344883


namespace area_of_circle_l344_344934

theorem area_of_circle (r θ : ℝ) (h : r = 4 * Real.cos θ - 3 * Real.sin θ) :
  ∃ π : ℝ, π * (5/2)^2 = 25 * π / 4 :=
by 
  sorry

end area_of_circle_l344_344934


namespace year_with_max_decrease_l344_344026

structure YearlySales where
  year : ℕ
  sales : ℝ

def sales_data : List YearlySales :=
  [{year := 1994, sales := 1.5}, {year := 1995, sales := 1.8}, {year := 1996, sales := 2.25}, 
   {year := 1997, sales := 2.7}, {year := 1998, sales := 3.15}, {year := 1999, sales := 3.6},
   {year := 2000, sales := 4.2}, {year := 2001, sales := 5.2}, {year := 2002, sales := 4.75},
   {year := 2003, sales := 3.25}]

def largest_decrease_year (sales_data : List YearlySales) : ℕ :=
  let decreases := sales_data.zip sales_data.tail |>.map (fun ((s1, s2)) => (s1.year, s2.sales - s1.sales))
  decreases.foldl (fun acc (y, d) => if d < acc.2 then (y, d) else acc) (0, 0.0) |>.fst

theorem year_with_max_decrease
  (sales_data : List YearlySales)
  (h : sales_data = sales_data) :
  largest_decrease_year sales_data = 2002 :=
by
  sorry

end year_with_max_decrease_l344_344026


namespace smallest_odd_factors_gt_100_l344_344559

theorem smallest_odd_factors_gt_100 : ∃ n : ℕ, n > 100 ∧ (∀ d : ℕ, d ∣ n → (∃ m : ℕ, n = m * m)) ∧ (∀ m : ℕ, m > 100 ∧ (∀ d : ℕ, d ∣ m → (∃ k : ℕ, m = k * k)) → n ≤ m) :=
by
  sorry

end smallest_odd_factors_gt_100_l344_344559


namespace cosA_sinA_plus_cosC_sinC_eq_l344_344301

variables {A B C a b c : Real}

-- Conditions
hypothesis (triangle_ABC : Triangle A B C)      -- Triangle condition
hypothesis (sides_opposite : (a = sideOppositeTo A ∧ b = sideOppositeTo B ∧ c = sideOppositeTo C)) -- Sides opposite
hypothesis (geometric_progression : b * b = a * c) -- Geometric progression
hypothesis (cos_B : cos B = 12 / 13)          -- Cosine of B

-- Proof statement
theorem cosA_sinA_plus_cosC_sinC_eq : 
  (cos A / sin A) + (cos C / sin C) = 13 / 5 :=
sorry

end cosA_sinA_plus_cosC_sinC_eq_l344_344301


namespace transform_function_l344_344256

-- Conditions
def f (ω : ℝ) (x : ℝ) := sin(ω * x) * cos(ω * x) + (sqrt 3 * cos(2 * ω * x)) / 2

/-- Prove the transformation of the function f to g -/
theorem transform_function (x : ℝ) (hx : x = x) :
  f (π / 4) x = sin(π / 2 * (x - 1 / 6) + π / 3) →
  (λ x, sin((π / 2) * x + 3 * π / 6)) = λ x, sin((π / 4) * x + π / 4) :=
sorry

end transform_function_l344_344256


namespace number_of_tables_l344_344306

/-- Problem Statement
  In a hall used for a conference, each table is surrounded by 8 stools and 4 chairs. Each stool has 3 legs,
  each chair has 4 legs, and each table has 4 legs. If the total number of legs for all tables, stools, and chairs is 704,
  the number of tables in the hall is 16. -/
theorem number_of_tables (legs_per_stool legs_per_chair legs_per_table total_legs t : ℕ) 
  (Hstools : ∀ tables, stools = 8 * tables)
  (Hchairs : ∀ tables, chairs = 4 * tables)
  (Hlegs : 3 * stools + 4 * chairs + 4 * t = total_legs)
  (Hleg_values : legs_per_stool = 3 ∧ legs_per_chair = 4 ∧ legs_per_table = 4)
  (Htotal_legs : total_legs = 704) :
  t = 16 := by
  sorry

end number_of_tables_l344_344306


namespace enclosed_area_correct_l344_344794

noncomputable def enclosed_area_closed_curve : ℝ :=
  let r := 5 / 4 in  -- From the length of arcs
  let s := 3 in      -- Side length of the octagon
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2 in
  let sector_area := (5 * Real.pi * r^2) / 6 in
  octagon_area + 12 * sector_area / 2 - 12 * sector_area / 4 -- Approximate enclosed area

theorem enclosed_area_correct :
  enclosed_area_closed_curve = 54 + 54 * Real.sqrt 2 + Real.pi :=
by
  sorry

end enclosed_area_correct_l344_344794


namespace Joan_books_gathered_up_l344_344337

variable (books_sold : ℕ) (books_left : ℕ)

-- Conditions given in the problem
def Joan_books_sold : books_sold = 26 := by sorry
def Joan_books_left : books_left = 7 := by sorry

-- Statement to prove
theorem Joan_books_gathered_up : books_sold + books_left = 33 := by
  rw [Joan_books_sold, Joan_books_left]
  exact add_comm _ _

end Joan_books_gathered_up_l344_344337


namespace james_not_train_days_l344_344712

-- Definitions based on conditions
def hours_per_day_train := 2 * 4
def total_hours_per_year := 2080
def weeks_per_year := 52
def days_per_week := 7

-- Lean statement to prove the number of days per week James does not train
theorem james_not_train_days : 
  let hours_per_week := total_hours_per_year / weeks_per_year in
  let days_train_per_week := hours_per_week / hours_per_day_train in
  (days_per_week - days_train_per_week) = 2 :=
by
  let hours_per_week := total_hours_per_year / weeks_per_year
  let days_train_per_week := hours_per_week / hours_per_day_train
  have h : (days_per_week - days_train_per_week) = 2 := sorry
  exact h

end james_not_train_days_l344_344712


namespace four_consecutive_none_multiple_of_5_l344_344822

theorem four_consecutive_none_multiple_of_5 (n : ℤ) :
  (∃ k : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 5 * k) →
  ¬ (∃ m : ℤ, (n = 5 * m) ∨ (n + 1 = 5 * m) ∨ (n + 2 = 5 * m) ∨ (n + 3 = 5 * m)) :=
by sorry

end four_consecutive_none_multiple_of_5_l344_344822


namespace find_number_l344_344476

variable (x : ℝ)

theorem find_number (hx : 5100 - (102 / x) = 5095) : x = 20.4 := 
by
  sorry

end find_number_l344_344476


namespace sphere_radius_l344_344869

-- Definitions from the problem conditions
variables (S A B C D S' O: Type) [EuclideanGeometry S A B C D S']

-- Conditions
def is_regular_octahedron := ∀ (e : Real), edge_length S A B C D S' e → e = 1
def tangent_to_face (O: Type) (f: Type) := tangent_to O f
def tangent_to_extended_planes (O: Type) (planes: List Type) := ∀ p ∈ planes, tangent_to O p

-- Proof problem statement
theorem sphere_radius 
  (h1 : is_regular_octahedron S A B C D S')
  (h2 : tangent_to_face O S(A B))
  (h3 : tangent_to_extended_planes O [S(B C), S(A D), S’(A B)]) :
  ∃ r : Real, r = 1 / 4 * sqrt (2 / 3) :=
begin
  sorry
end

end sphere_radius_l344_344869


namespace secant_slope_2_pow_x_l344_344818

theorem secant_slope_2_pow_x : 
  let f := (λ x : ℝ, 2 ^ x) in
  f 0 = 1 ∧ f 1 = 2 → (f 1 - f 0) / (1 - 0) = 1 := 
by
  intro h
  rw [and.elim_left h, and.elim_right h]
  have h_slope : (2 - 1) / (1 - 0) = 1 := by
    calc
      (2 - 1) / (1 - 0) = 1 / 1 := by sorry
                           _     = 1 := by sorry
  exact h_slope

end secant_slope_2_pow_x_l344_344818


namespace ekon_uma_diff_l344_344446

-- Definitions based on conditions
def total_videos := 411
def kelsey_videos := 160
def ekon_kelsey_diff := 43

-- Definitions derived from conditions
def ekon_videos := kelsey_videos - ekon_kelsey_diff
def uma_videos (E : ℕ) := total_videos - kelsey_videos - E

-- The Lean problem statement
theorem ekon_uma_diff : 
  uma_videos ekon_videos - ekon_videos = 17 := 
by 
  sorry

end ekon_uma_diff_l344_344446


namespace no_99_percent_confidence_distribution_expectation_variance_l344_344778

open ProbabilityTheory MeasureTheory

-- Data from the conditions
def a : ℕ := 40
def b : ℕ := 10
def c : ℕ := 30
def d : ℕ := 20
def n : ℕ := 100

-- Definitions required for the part (1)
def k_square : ℝ := (n * ((a * d - b * c) ^ 2) : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement for part (1)
theorem no_99_percent_confidence : k_square < 6.635 :=
sorry

-- Definitions required for part (2)
noncomputable def p : ℝ := 2 / 5

def X : Type := Fin₃

def P_X (k : ℕ) : ℝ :=
  if k = 0 then (3.choose 0) * ((1 - p) ^ 3)
  else if k = 1 then (3.choose 1) * (p) * ((1 - p) ^ 2)
  else if k = 2 then (3.choose 2) * (p ^ 2) * ((1 - p))
  else if k = 3 then (3.choose 3) * (p ^ 3)
  else 0

-- Discrete table for distribution of X
def distribution_X : List (ℕ × ℝ) :=
  [(0, P_X 0), (1, P_X 1), (2, P_X 2), (3, P_X 3)]

-- Expected value
def E_X : ℝ := 3 * p

-- Variance
def var_X : ℝ := 3 * p * (1 - p)

theorem distribution_expectation_variance :
  distribution_X = [(0, 27/125), (1, 54/125), (2, 36/125), (3, 8/125)] ∧
  E_X = 6/5 ∧
  var_X = 18/25 :=
sorry

end no_99_percent_confidence_distribution_expectation_variance_l344_344778


namespace solid_could_be_rectangular_prism_or_cylinder_l344_344670

-- Definitions for the conditions
def is_rectangular_prism (solid : Type) : Prop := sorry
def is_cylinder (solid : Type) : Prop := sorry
def front_view_is_rectangle (solid : Type) : Prop := sorry
def side_view_is_rectangle (solid : Type) : Prop := sorry

-- Main statement
theorem solid_could_be_rectangular_prism_or_cylinder
  {solid : Type}
  (h1 : front_view_is_rectangle solid)
  (h2 : side_view_is_rectangle solid) :
  is_rectangular_prism solid ∨ is_cylinder solid :=
sorry

end solid_could_be_rectangular_prism_or_cylinder_l344_344670


namespace sum_of_smallest_n_l344_344354

def tau (n : ℕ) : ℕ := n.divisors.count

theorem sum_of_smallest_n (h : ∑ (k : ℕ) in (Finset.filter 
  (λ n, tau n + tau (n + 1) = 8) (Finset.range 1000)).sort (≤) (Finset.range 6) = 73) : 
  ℕ := sorry

end sum_of_smallest_n_l344_344354


namespace constant_term_correct_l344_344615

noncomputable def constant_term_expansion : ℝ :=
  let n := ∫ x in 0..(Real.pi / 2), 10 * Real.sin x
  if (n = 10) then 210 else 0

theorem constant_term_correct :
  (∫ x in 0..(Real.pi / 2), 10 * Real.sin x = 10) →
  constant_term_expansion = 210 :=
by
  intros h
  simp [constant_term_expansion, h]
  sorry

end constant_term_correct_l344_344615


namespace roots_sum_of_squares_l344_344815

theorem roots_sum_of_squares :
  let x₁ x₂ : ℝ :=
    by
      have h : 2 * x^2 - 3 * x - 1 = 0
      sorry
  let sum : ℝ := 3 / 2
  let product : ℝ := -1 / 2
  (x₁ + x₂ = sum) ∧ (x₁ * x₂ = product) → x₁^2 + x₂^2 = 13 / 4 :=
by
  assume x₁ x₂ : ℝ
  have h_eq : 2 * x₁² - 3 * x₁ - 1 = 0 := sorry
  have h_sum : x₁ + x₂ = 3/2 := sorry
  have h_product : x₁ * x₂ = -1/2 := sorry
  calc
    x₁^2 + x₂^2
    = (x₁ + x₂)^2 - 2 * (x₁ * x₂) : by sorry
    ... = (3/2)^2 - 2 * (-1/2) : by rw [h_sum, h_product]
    ... = 13/4 : by norm_num

end roots_sum_of_squares_l344_344815


namespace rod_length_proportional_l344_344650

theorem rod_length_proportional (L : ℝ) :
  (∀ (w : ℝ) (l : ℝ), (l / w) = (11.25 / 42.75)) → ((26.6 / 42.75) * 11.25 = L) → L = 7 :=
by
  intro prop weight
  have calculation : L = (11.25 * 26.6) / 42.75 := 
    by rw [← weight, prop, mul_comm 26.6 11.25, ←mul_assoc, div_mul_cancel, one_mul]
  exact calculation
  sorry

end rod_length_proportional_l344_344650


namespace some_number_value_correct_l344_344661

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l344_344661


namespace measure_of_angle_ABC_l344_344210

variables {A B C D : Type} [geometry A B C D]

-- Given conditions
def angle_BAC : real := 60
def angle_CAD : real := 60
def sum_AB_AD_eq_AC (AB AD AC : real) : Prop := AB + AD = AC
def angle_ACD : real := 23

-- Main statement
theorem measure_of_angle_ABC (AB AD AC : real) 
  (h1 : ∠ BAC = angle_BAC) 
  (h2 : ∠ CAD = angle_CAD) 
  (h3 : sum_AB_AD_eq_AC AB AD AC) 
  (h4 : ∠ ACD = angle_ACD) : 
  ∠ ABC = 83 :=
by
  sorry

end measure_of_angle_ABC_l344_344210


namespace total_cost_function_l344_344627

-- Define the given conditions
def dist_AB : ℝ := 120  -- km
def speed_min : ℝ := 50  -- km/h
def speed_max : ℝ := 100  -- km/h
def fuel_price : ℝ := 6  -- yuan/L
def fuel_cons_rate (x : ℝ) : ℝ := 3 + x^2 / 360  -- L/h
def driver_wage : ℝ := 42  -- yuan/h

-- Define the cost function
def cost_function (x : ℝ) : ℝ :=
  let travel_time := dist_AB / x
  let fuel_cost := travel_time * (fuel_cons_rate x) * fuel_price
  let wage_cost := travel_time * driver_wage
  fuel_cost + wage_cost  -- total cost

-- The theorem to prove
theorem total_cost_function (x : ℝ) (hx : speed_min ≤ x ∧ x ≤ speed_max) :
  cost_function x = 2 * x + 7200 / x :=
by
  sorry

end total_cost_function_l344_344627


namespace transport_cases_l344_344040

theorem transport_cases (bus_ways : ℕ) (subway_ways : ℕ) (h_bus : bus_ways = 3) (h_subway : subway_ways = 4) : bus_ways + subway_ways = 7 :=
by
  rw [h_bus, h_subway]
  sorry

end transport_cases_l344_344040


namespace sum_lent_l344_344070

theorem sum_lent (P : ℝ) (r t : ℝ) (I : ℝ) (h1 : r = 6) (h2 : t = 6) (h3 : I = P - 672) (h4 : I = P * r * t / 100) :
  P = 1050 := by
  sorry

end sum_lent_l344_344070


namespace non_empty_set_A_l344_344230

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {x | x ^ 2 = a}

-- Theorem statement
theorem non_empty_set_A (a : ℝ) (h : (A a).Nonempty) : 0 ≤ a :=
by
  sorry

end non_empty_set_A_l344_344230


namespace positive_real_solution_count_l344_344174

noncomputable def f (x : ℝ) : ℝ := x^4 + 5 * x^3 + 28 * x^2 + 145 * x - 1897

theorem positive_real_solution_count : ∀ x : ℝ, (x^10 + 5 * x^9 + 28 * x^8 + 145 * x^7 - 1897 * x^6 = 0) → (∃! x > 0, f x = 0) :=
by
  sorry

end positive_real_solution_count_l344_344174


namespace largest_whole_number_satisfies_inequality_l344_344843

theorem largest_whole_number_satisfies_inequality :
  ∃ x : ℤ, (∀ y : ℤ, (1 / 4 + y / 5 < 1 → y ≤ x)) ∧ (1 / 4 + x / 5 < 1) :=
by
  have h₀ : ∀ x, 1 / 4 + x / 5 < 1 ↔ 4 * (1 / 4 + x / 5) < 4 * 1 := by sorry
  have h₁ : ∀ x, 4 * (1 / 4 + x / 5) < 4 := by sorry
  have h₂ : ∀ x, 1 + x < 4 := by sorry
  have h₃ : ∀ x, x < 3 := by sorry
  exact ⟨3, λ y hy, by linarith, by norm_num⟩

end largest_whole_number_satisfies_inequality_l344_344843


namespace similarity_FEG_ABC_l344_344867

variables {A B C E F M G : Type} 
variables (triangle : Triangle A B C) 
          (point_E : E ∈ segment B C) 
          (point_F : F ∈ segment C A) 
          (midpoint_M : midpoint M E F) 
          (intersection_G : intersection (line C M) (line A B) = G)
          (cond1 : (CE / CB + CF / CA = 1))
          (cond2 : ∠ C E F = ∠ C A B)

theorem similarity_FEG_ABC 
    (triangle : Triangle A B C) 
    (point_E : E ∈ segment B C) 
    (point_F : F ∈ segment C A) 
    (midpoint_M : midpoint M E F) 
    (intersection_G : intersection (line C M) (line A B) = G)
    (cond1 : (CE / CB + CF / CA = 1))
    (cond2 : ∠ C E F = ∠ C A B) 
    : Similar_Triangle (Triangle F E G) (Triangle A B C) :=
begin
  sorry
end

end similarity_FEG_ABC_l344_344867


namespace probability_sum_is_five_l344_344767

theorem probability_sum_is_five :
  let S := {1, 2, 3, 4}
  let pairs := [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
  let favorable_pairs := [(1, 4), (2, 3)]
  let probability := (favorable_pairs.length : ℚ) / (pairs.length : ℚ)
  in probability = 1 / 3 := 
sorry

end probability_sum_is_five_l344_344767


namespace minimum_value_f_condition_f_geq_zero_l344_344631

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem minimum_value_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ f (Real.log a) a) ∧ f (Real.log a) a = a - a * Real.log a - 1 :=
by 
  sorry

theorem condition_f_geq_zero (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ a = 1 :=
by 
  sorry

end minimum_value_f_condition_f_geq_zero_l344_344631


namespace correct_equation_l344_344679

/-- Definitions and conditions used in the problem -/
def jan_revenue := 250
def feb_revenue (x : ℝ) := jan_revenue * (1 + x)
def mar_revenue (x : ℝ) := jan_revenue * (1 + x)^2
def first_quarter_target := 900

/-- Proof problem statement -/
theorem correct_equation (x : ℝ) : 
  jan_revenue + feb_revenue x + mar_revenue x = first_quarter_target := 
by
  sorry

end correct_equation_l344_344679


namespace longer_side_length_l344_344485

-- Definitions and conditions
def circle_radius : ℝ := 6
def circle_area : ℝ := Real.pi * circle_radius ^ 2
def rectangle_area : ℝ := 4 * circle_area
def shorter_side_length : ℝ := 2 * circle_radius

-- The length of the longer side of the rectangle
theorem longer_side_length :
  ∃ L : ℝ, rectangle_area = shorter_side_length * L ∧ L = 12 * Real.pi :=
  sorry

end longer_side_length_l344_344485


namespace math_proof_problem_l344_344179

noncomputable def problem_statement : Nat := 
  ⌈Real.sqrt (12 / 5)⌉ +
  ⌈(12 / 5)⌉ +
  ⌈(12 / 5) ^ 2⌉ +
  ⌈(12 / 5) ^ 3⌉

theorem math_proof_problem : problem_statement = 25 := 
by
  sorry

end math_proof_problem_l344_344179


namespace triangle_ratio_max_dot_product_l344_344269

theorem triangle_ratio_max_dot_product
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : C = π / 3) 
  (h2 : c = 2)
  (h3 : A + B + C = π)
  (h4 : ¬ (a = 0 ∨ b = 0 ∨ c = 0)) :
  (∃ A', maxima (A', A = π / 12) 
   (λ t, a * b * (2 * cos(C - A') + cos(C - 2 * A'))) ∧
    (a = 2 * sin(A) / (sin(C - A')))
  → (b / a = 2 + sqrt 3)) :=
sorry  -- Proof not provided, only the statement as requested.

end triangle_ratio_max_dot_product_l344_344269


namespace nicky_cristina_race_l344_344079

theorem nicky_cristina_race :
  ∀ (c_speed n_speed head_start : ℕ),
    c_speed = 5 →
    n_speed = 3 →
    head_start = 12 →
    let catch_up_time := (c_speed * head_start) / (c_speed - n_speed) in
    let total_time := head_start + catch_up_time in
    total_time = 42 :=
by
  intros c_speed n_speed head_start c_speed_def n_speed_def head_start_def
  let catch_up_time := (c_speed * head_start) / (c_speed - n_speed)
  let total_time := head_start + catch_up_time
  simp [c_speed_def, n_speed_def, head_start_def] at *
  sorry

end nicky_cristina_race_l344_344079


namespace probability_X_eq_2_l344_344264

theorem probability_X_eq_2 (n p : ℚ) (X : ℕ → ℚ) 
  (h1 : n * p = 2) 
  (h2 : n * p * (1 - p) = 4 / 3) :
  let P_X_2 := (nat.choose 6 2) * (1 / 3)^2 * (2 / 3)^4 in 
  P_X_2 = 80 / 243 :=
by
  sorry

end probability_X_eq_2_l344_344264


namespace combination_number_formula_l344_344795

theorem combination_number_formula (n r : ℕ) (h₁ : n > r) (h₂ : r ≥ 1) :
  nat.choose n r = (n * nat.choose (n - 1) (r - 1)) / r :=
sorry

end combination_number_formula_l344_344795


namespace parabola_equation_l344_344823

theorem parabola_equation (vertex focus : ℝ × ℝ) 
  (h_vertex : vertex = (0, 0)) 
  (h_focus_line : ∃ x y : ℝ, focus = (x, y) ∧ x - y + 2 = 0) 
  (h_symmetry_axis : ∃ axis : ℝ × ℝ → ℝ, ∀ p : ℝ × ℝ, axis p = 0): 
  ∃ k : ℝ, k > 0 ∧ (∀ x y : ℝ, y^2 = -8*x ∨ x^2 = 8*y) :=
by {
  sorry
}

end parabola_equation_l344_344823


namespace changed_answers_percentage_l344_344888

variables (n : ℕ) (a b c d : ℕ)

theorem changed_answers_percentage (h1 : a + b + c + d = 100)
  (h2 : a + d + c = 50)
  (h3 : a + c = 60)
  (h4 : b + d = 40) :
  10 ≤ c + d ∧ c + d ≤ 90 :=
  by sorry

end changed_answers_percentage_l344_344888


namespace subcommittees_with_at_least_one_teacher_l344_344487

theorem subcommittees_with_at_least_one_teacher (total_members teachers: ℕ) (subcommittee_size: ℕ) 
    (h1: total_members = 12) (h2: teachers = 5) (h3: subcommittee_size = 5) : 
    (Finset.card (Finset.filter (λ s, (Finset.card (s ∩ (Finset.range teachers).val) > 0)) 
    (Finset.filter (λ s, Finset.card s = subcommittee_size) 
      (Finset.powerset (Finset.range total_members)))) = 771) :=
by
  sorry

end subcommittees_with_at_least_one_teacher_l344_344487


namespace incorrect_contrapositive_statement_l344_344064

theorem incorrect_contrapositive_statement
  (x : ℝ) : ¬((x ≠ 1 ∨ x ≠ -1) → x^2 ≠ 1) :=
by
  -- We'll use a proof by counterexample, showing the contrapositive fails.
  -- Specifically, x = 1 or x = -1 would not satisfy the contrapositive's statement.
  intro h
  have h₁ := h 1
  have h₂ := h (-1)
  -- We leave the proof as sorry since explicit steps are not required in the problem.
  sorry

end incorrect_contrapositive_statement_l344_344064


namespace probability_two_vertices_on_same_edge_is_correct_l344_344046

noncomputable def probability_two_vertices_on_same_edge : ℚ :=
  let total_vertices := 20
  let total_ways := (20.choose 3)  -- number of ways to choose 3 vertices out of 20
  let total_edges := 30
  let ways_with_connected_vertices :=
    total_edges * (total_vertices - 2)  -- for each edge, there are 18 remaining vertices
  (ways_with_connected_vertices : ℚ) / total_ways

theorem probability_two_vertices_on_same_edge_is_correct :
  probability_two_vertices_on_same_edge = 9 / 19 :=
  sorry

end probability_two_vertices_on_same_edge_is_correct_l344_344046


namespace sum_values_count_l344_344913

def BagC := {1, 2, 3, 4}
def BagD := {3, 5, 7}

theorem sum_values_count :
  {s | ∃ (c ∈ BagC) (d ∈ BagD), s = c + d}.toFinset.card = 8 :=
by
  sorry

end sum_values_count_l344_344913


namespace find_a_b_l344_344250

noncomputable def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x : ℝ | -3 < x ∧ x < 2 }
noncomputable def sol_set (a b : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + b < 0 }

theorem find_a_b :
  (sol_set (-2) (3 - 6)) = A ∩ B → (-1) + (-2) = -3 :=
by
  intros h1
  sorry

end find_a_b_l344_344250


namespace perimeter_result_l344_344520

-- Define the side length of the square
def side_length : ℕ := 100

-- Define the dimensions of the rectangle
def rectangle_dim1 : ℕ := side_length
def rectangle_dim2 : ℕ := side_length / 2

-- Perimeter calculation based on the arrangement
def perimeter : ℕ :=
  3 * rectangle_dim1 + 4 * rectangle_dim2

-- The statement of the problem
theorem perimeter_result :
  perimeter = 500 :=
by
  sorry

end perimeter_result_l344_344520


namespace coloring_points_formula_l344_344858

def coloring_points : Nat → Nat
| 0       := 0
| 1       := 5
| 2       := 13
| (n + 3) := 2 * coloring_points (n + 2) + 3 * coloring_points (n + 1)

theorem coloring_points_formula (n : Nat) : coloring_points n = (3 ^ n + (-1 : Int) ^ n) / 2 := by
  sorry

end coloring_points_formula_l344_344858


namespace product_remainder_l344_344584

theorem product_remainder (n : ℕ) (hn : n = 20) : 
  (List.prod (List.map (λ i, 10 * i + 4) (List.range n))) % 5 = 1 := by
  sorry

end product_remainder_l344_344584


namespace sin_270_eq_neg_one_l344_344160

theorem sin_270_eq_neg_one : Real.sin (270 * Real.pi / 180) = -1 := 
by
  sorry

end sin_270_eq_neg_one_l344_344160


namespace problem_solution_l344_344675

-- Define the operations star and circ
def star (a b : ℝ) : ℝ := a + a / b
def circ (x y : ℝ) : ℝ := (star x y) * y

-- State the theorem
theorem problem_solution : circ 8 4 = 40 :=
by
  sorry

end problem_solution_l344_344675


namespace possible_to_fill_array_l344_344711

open BigOperators

theorem possible_to_fill_array :
  ∃ (f : (Fin 10) × (Fin 10) → ℕ),
    (∀ i j : Fin 10, 
      (i ≠ 0 → f (i, j) ∣ f (i - 1, j) ∧ f (i, j) ≠ f (i - 1, j))) ∧
    (∀ i : Fin 10, ∃ n : ℕ, ∀ j : Fin 10, f (i, j) = n + j) :=
sorry

end possible_to_fill_array_l344_344711


namespace constant_term_in_binomial_expansion_l344_344796

theorem constant_term_in_binomial_expansion : 
  let T_r_plus_1 (r : ℕ) := (Nat.choose 6 r) * (-1)^r * (x^(3-r : ℤ))
  (-20 : ℤ) =
    T_r_plus_1 3 := 
by 
  sorry

end constant_term_in_binomial_expansion_l344_344796


namespace percentage_difference_between_M_and_J_is_34_74_percent_l344_344378

-- Definitions of incomes and relationships
variables (J T M : ℝ)
variables (h1 : T = 0.80 * J)
variables (h2 : M = 1.60 * T)

-- Definitions of savings and expenses
variables (Msavings : ℝ := 0.15 * M)
variables (Mexpenses : ℝ := 0.25 * M)
variables (Tsavings : ℝ := 0.12 * T)
variables (Texpenses : ℝ := 0.30 * T)
variables (Jsavings : ℝ := 0.18 * J)
variables (Jexpenses : ℝ := 0.20 * J)

-- Total savings and expenses
variables (Mtotal : ℝ := Msavings + Mexpenses)
variables (Jtotal : ℝ := Jsavings + Jexpenses)

-- Prove the percentage difference between Mary's and Juan's total savings and expenses combined
theorem percentage_difference_between_M_and_J_is_34_74_percent :
  M = 1.28 * J → 
  Mtotal = 0.40 * M →
  Jtotal = 0.38 * J →
  ( (Mtotal - Jtotal) / Jtotal ) * 100 = 34.74 :=
by
  sorry

end percentage_difference_between_M_and_J_is_34_74_percent_l344_344378


namespace find_line_equation_l344_344198

open Real

noncomputable def line_through (x1 y1 k : ℝ) : ℝ → ℝ := λ x, k * (x - x1) + y1
noncomputable def line_vertical (x1 : ℝ) : ℝ → ℝ := λ _, x1
def dist_to_line (line : ℝ → ℝ) (x y : ℝ) : ℝ := abs (line x - y) / (1 + line x * line x).sqrt

theorem find_line_equation :
  let A := (2, 3), B := (0, -5), P := (1, 2);
  (∃ k : ℝ, ∃ line : ℝ → ℝ, line P.1 = P.2 ∧
    dist_to_line line A.1 A.2 = dist_to_line line B.1 B.2 ∧
    (line = line_through P.1 P.2 k ∨ line = line_vertical P.1))
:=
begin
  let A := (2, 3),
  let B := (0, -5),
  let P := (1, 2),
  use 4,
  use line_through P.1 P.2 4,
  split,
  { rw line_through,
    simp },
  split,
  { rw [dist_to_line, line_through],
    sorry },
  left,
  refl
end

end find_line_equation_l344_344198


namespace range_of_a_l344_344418

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≤ a^2 - 3a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
  sorry

end range_of_a_l344_344418


namespace positive_pairs_count_l344_344425

theorem positive_pairs_count :
  (∃ n : ℕ, n = 7 ∧ 
    (∀ a b : ℕ, 
      (a > 0) → (b > 0) → 
      (a + b ≤ 100) → 
      ((a + b⁻¹) / (a⁻¹ + b) = 13) → 
      (∃! (a b : ℕ), a = 13 * b ∧ a + b ≤ 100 ∧ a > 0 ∧ b > 0))
  ) := 
sorry

end positive_pairs_count_l344_344425


namespace profit_with_discount_l344_344114

theorem profit_with_discount 
  (CP : ℝ)
  (discount_rate : ℝ)
  (profit_no_discount_perc : ℝ)
  (hCP : CP = 100)
  (h_discount_rate : discount_rate = 0.04)
  (h_profit_no_discount_perc : profit_no_discount_perc = 0.4375) : 
  (profit_with_discount_perc : ℝ)
  (h : profit_with_discount_perc = 0.38) :=
sorry

end profit_with_discount_l344_344114


namespace cristian_cookie_problem_l344_344938

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end cristian_cookie_problem_l344_344938


namespace max_min_distances_l344_344689

-- Definition of the ellipse in parametric form.
def ellipse_parametric (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 * real.cos θ, real.sin θ)

-- Definition of the line in Cartesian form derived from the given polar form.
def line_cartesian (x y : ℝ) := x - sqrt 3 * y - 3 * sqrt 6 = 0

-- Compute the distance from a point on the ellipse to the line.
noncomputable def distance_to_line (θ : ℝ) : ℝ :=
  abs (sqrt 3 * real.cos θ - sqrt 3 * real.sin θ - 3 * sqrt 6) / 2

theorem max_min_distances :
  ∃ (max_dist min_dist : ℝ),
    (max_dist = 2 * sqrt 6) ∧ (min_dist = sqrt 6) ∧
    ∀ θ : ℝ, distance_to_line θ ≤ max_dist ∧ distance_to_line θ ≥ min_dist :=
sorry

end max_min_distances_l344_344689


namespace infinite_sqrt_evaluation_l344_344182

noncomputable def infinite_sqrt : ℝ := 
  sqrt (15 + infinite_sqrt)

theorem infinite_sqrt_evaluation : infinite_sqrt = (1 + sqrt 61) / 2 := by
  sorry

end infinite_sqrt_evaluation_l344_344182


namespace bayes_theorem_for_influenza_l344_344805
open Classical

variable (A C : Prop)

def P : Prop → ℝ := sorry

axiom P_AC : P (A ∩ C) = 0.9 * P C
axiom P_notA_notC : P (¬A ∩ ¬C) = 0.9 * P (¬C)
axiom P_C : P C = 0.005

theorem bayes_theorem_for_influenza : P (C ∩ A) / P A = 9 / 208 := by
  sorry

end bayes_theorem_for_influenza_l344_344805


namespace range_of_a_l344_344169

noncomputable def matrix_det_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem range_of_a : 
  {a : ℝ | matrix_det_2x2 (a^2) 1 3 2 < matrix_det_2x2 a 0 4 1} = {a : ℝ | -1 < a ∧ a < 3/2} :=
by
  sorry

end range_of_a_l344_344169


namespace expected_expenditure_2017_l344_344101

-- Defining the problem conditions
def average (l : List ℝ) : ℝ := (l.sum) / (l.length)

def regression_expenditure (x : List ℝ) (y : List ℝ) (b : ℝ) (x_new : ℝ) : ℝ :=
  let x_bar := average x
  let y_bar := average y
  let a := y_bar - b * x_bar
  b * x_new + a

-- Given conditions in the problem
def x_values := [8.2, 8.6, 10.0, 11.3, 11.9]
def y_values := [6.2, 7.5, 8.0, 8.5, 9.8]
def beta := 0.76
def x_2017 := 15.0 -- Income in 2017 in ten thousand yuan

-- The theorem to prove the question == answer given conditions
theorem expected_expenditure_2017 :
  regression_expenditure x_values y_values beta x_2017 = 11.8 := by sorry

end expected_expenditure_2017_l344_344101


namespace alpha_has_winning_strategy_l344_344049

def Pile := Nat

structure Game :=
  (pile1 : Pile)
  (pile2 : Pile)
  (alpha_turn : Bool)

noncomputable def initial_game : Game := {
  pile1 := 33,
  pile2 := 35,
  alpha_turn := true
}

def can_split (pile : Pile) : Bool :=
  pile > 1

def winning_strategy (game: Game) : Prop :=
  ∃ moves : List Game,
    moves.head = game ∧
    (∀ g ∈ moves.init, can_split g.pile1 ∨ can_split g.pile2) ∧
    (moves.last.alpha_turn = false ∧ ¬ can_split moves.last.pile1 ∧ ¬ can_split moves.last.pile2)

theorem alpha_has_winning_strategy : winning_strategy initial_game :=
  sorry

end alpha_has_winning_strategy_l344_344049


namespace subset_bound_l344_344727

open Finset

variables {α : Type*}

theorem subset_bound (n : ℕ) (S : Finset (Finset (Fin (4 * n)))) (hS : ∀ {s t : Finset (Fin (4 * n))}, s ∈ S → t ∈ S → s ≠ t → (s ∩ t).card ≤ n) (h_card : ∀ s ∈ S, s.card = 2 * n) :
  S.card ≤ 6 ^ ((n + 1) / 2) :=
sorry

end subset_bound_l344_344727


namespace trapezoid_diagonal_segment_length_l344_344084

theorem trapezoid_diagonal_segment_length
  (PQ RS T : Type) [Trapezoid PQ RS T]
  (H1 : PQ = 3 * RS) 
  (H2 : DiagonalsIntersectAt PQ RS T)
  (H3 : length PR = 15) :
  length RT = 15 / 4 :=
by
  sorry

end trapezoid_diagonal_segment_length_l344_344084


namespace machine_present_value_l344_344490

theorem machine_present_value
  (rate_of_decay : ℝ) (n_periods : ℕ) (final_value : ℝ) (initial_value : ℝ)
  (h_decay : rate_of_decay = 0.25)
  (h_periods : n_periods = 2)
  (h_final_value : final_value = 225) :
  initial_value = 400 :=
by
  -- The proof would go here. 
  sorry

end machine_present_value_l344_344490


namespace probability_one_overcomes_another_l344_344403
-- We will import all necessary libraries

-- Define the elements and their relationships
inductive Element
| Metal
| Wood
| Earth
| Water
| Fire

open Element

-- Define a function that asserts one element overcomes another
def overcomes : Element → Element → Prop
| Metal, Wood => true
| Wood, Earth => true
| Earth, Water => true
| Water, Fire => true
| Fire, Metal => true
| _, _ => false

-- The theorem to be proved
theorem probability_one_overcomes_another : 
  (∃ els : List Element, els.length = 2 ∧ (overcomes els.head! els.tail.head! ∨ overcomes els.tail.head! els.head!)) 
  → ((Finset.card (Finset.filter (λ x => x) (Finset.image (uncurry overcomes) (Finset.unorderedPairs (Finset.univ : Finset Element))))) / 
  Finset.card (Finset.unorderedPairs (Finset.univ : Finset Element)) = 1/2) :=
by
  sorry

end probability_one_overcomes_another_l344_344403


namespace football_team_people_count_l344_344030

theorem football_team_people_count (original_count : ℕ) (new_members : ℕ) (total_count : ℕ) 
  (h1 : original_count = 36) (h2 : new_members = 14) : total_count = 50 :=
by
  -- This is where the proof would go. We write 'sorry' because it is not required.
  sorry

end football_team_people_count_l344_344030


namespace correct_statements_l344_344010

theorem correct_statements (S1 S3 S4 S5 : Prop)
  (h1 : S1 = "The smaller the standard deviation, the smaller the fluctuation in sample data.")
  (h3 : S3 = "In regression analysis, the predicted variable is determined by both the explanatory variable and random error.")
  (h4 : S4 = "The coefficient of determination, \(R^2\), is used to characterize the regression effect; the larger the \(R^2\), the better the fit of the regression model.")
  (h5 : S5 = "For the observed value \(k\) of the random variable \(K^2\) for categorical variables \(X\) and \(Y\), the smaller the \(k\), the less confident one can be about the relationship between \(X\) and \(Y\).")
  (C1 : S1)
  (C3 : S3)
  (C4 : S4)
  (C5 : S5) : 
  (S1 ∧ S3 ∧ S4 ∧ S5) :=
by
  exact ⟨C1, C3, C4, C5⟩

end correct_statements_l344_344010


namespace people_left_line_l344_344912

-- Definitions based on the conditions given in the problem
def initial_people := 7
def new_people := 8
def final_people := 11

-- Proof statement
theorem people_left_line (L : ℕ) (h : 7 - L + 8 = 11) : L = 4 :=
by
  -- Adding the proof steps directly skips to the required proof
  sorry

end people_left_line_l344_344912


namespace highest_power_of_2_divides_l344_344578

def a : ℕ := 17
def b : ℕ := 15
def n : ℕ := a^5 - b^5

def highestPowerOf2Divides (k : ℕ) : ℕ :=
  -- Function to find the highest power of 2 that divides k, implementation is omitted
  sorry

theorem highest_power_of_2_divides :
  highestPowerOf2Divides n = 2^5 := by
    sorry

end highest_power_of_2_divides_l344_344578


namespace largest_integer_x_divisible_l344_344579

theorem largest_integer_x_divisible (x : ℤ) : 
  (∃ x : ℤ, (x^2 + 3 * x + 8) % (x - 2) = 0 ∧ x ≤ 1) → x = 1 :=
sorry

end largest_integer_x_divisible_l344_344579


namespace trip_duration_l344_344526

noncomputable def start_time : ℕ := 11 * 60 + 25 -- 11:25 a.m. in minutes
noncomputable def end_time : ℕ := 16 * 60 + 43 + 38 / 60 -- 4:43:38 p.m. in minutes

theorem trip_duration :
  end_time - start_time = 5 * 60 + 18 := 
sorry

end trip_duration_l344_344526


namespace fraction_to_decimal_l344_344966

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l344_344966


namespace total_study_hours_during_semester_l344_344009

-- Definitions of the given conditions
def semester_weeks : ℕ := 15
def weekday_study_hours_per_day : ℕ := 3
def saturday_study_hours : ℕ := 4
def sunday_study_hours : ℕ := 5

-- Theorem statement to prove the total study hours during the semester
theorem total_study_hours_during_semester : 
  (semester_weeks * ((5 * weekday_study_hours_per_day) + saturday_study_hours + sunday_study_hours)) = 360 := by
  -- We are skipping the proof step and adding a placeholder
  sorry

end total_study_hours_during_semester_l344_344009


namespace sequence_remainder_zero_l344_344846

theorem sequence_remainder_zero :
  let a := 3
  let d := 8
  let n := 32
  let aₙ := a + (n - 1) * d
  let Sₙ := n * (a + aₙ) / 2
  aₙ = 251 → Sₙ % 8 = 0 :=
by
  intros
  sorry

end sequence_remainder_zero_l344_344846


namespace weight_of_new_person_l344_344406

theorem weight_of_new_person
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (replaced_weight : ℝ)
  (weight_increase_total : ℝ)
  (W : ℝ)
  (h1 : avg_increase = 4.5)
  (h2 : num_persons = 8)
  (h3 : replaced_weight = 65)
  (h4 : weight_increase_total = 8 * 4.5)
  (h5 : W = replaced_weight + weight_increase_total) :
  W = 101 :=
by
  sorry

end weight_of_new_person_l344_344406


namespace hyperbola_eccentricity_l344_344247

theorem hyperbola_eccentricity :
  ∀ (a b : ℝ), (a = 2 * b) → (a^2 + b^2 = 5 * b^2) → 
               (∃ e : ℝ, e = (sqrt 5) / 2) :=
by
  intros a b h1 h2
  use (sqrt 5) / 2
  sorry

end hyperbola_eccentricity_l344_344247


namespace sum_of_angles_lt_1100_l344_344090

-- Definitions of conditions
def car_speed : ℝ := 16.67  -- Speed of the car in meters per second
def fence_length : ℝ := 100  -- Length of the fence in meters
def measurement_interval : ℝ := 1  -- Measurement interval in seconds
def total_time : ℝ := fence_length / car_speed  -- Total time over which measurements are taken

-- The mathematical statement to prove
theorem sum_of_angles_lt_1100 :
  let n := total_time / measurement_interval in
  ∑ i in finset.range (n.ceil), (
    -- Assuming we have an angle function θ(fence_length, i * car_speed), defining the logic here
    θ := λ (fence_length i : ℝ) : ℝ, sorry,
    θ fence_length i
  ) < 1100 :=
by
  sorry

end sum_of_angles_lt_1100_l344_344090


namespace uniform_pdf_normalization_l344_344203

open MeasureTheory

variables (α β : ℝ) (p : ℝ → ℝ)

-- Define the probability density function for the uniform distribution
def uniform_pdf (c : ℝ) (x : ℝ) : ℝ :=
  if (α ≤ x ∧ x ≤ β) then c else 0

-- The integral condition for any probability density function
def integral_condition (p : ℝ → ℝ) : Prop :=
  ∫ x, p x = 1

-- The uniform_pdf function must satisfy the integral condition
theorem uniform_pdf_normalization (h : α < β) :
  integral_condition (uniform_pdf (1 / (β - α)) α β) :=
by
  sorry

end uniform_pdf_normalization_l344_344203


namespace gcd_gx_x_l344_344617

noncomputable def g (x : ℤ) : ℤ :=
  (3 * x + 5) * (9 * x + 4) * (11 * x + 8) * (x + 11)

theorem gcd_gx_x (x : ℤ) (h : 34914 ∣ x) : Int.gcd (g x) x = 1760 :=
by
  sorry

end gcd_gx_x_l344_344617


namespace matroskin_milk_amount_l344_344863

theorem matroskin_milk_amount :
  ∃ S M x : ℝ, S + M = 10 ∧ (S - x) = (1 / 3) * S ∧ (M + x) = 3 * M ∧ (M + x) = 7.5 := 
sorry

end matroskin_milk_amount_l344_344863


namespace distinct_ints_divisibility_l344_344367

theorem distinct_ints_divisibility
  (x y z : ℤ) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : z ≠ x) : 
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * (y - z) * (z - x) * (x - y) * k := 
by 
  sorry

end distinct_ints_divisibility_l344_344367


namespace pyramid_construction_possible_l344_344686

theorem pyramid_construction_possible
  (A B C E F G : Type)
  (is_acute_angled_triangle : ∀ (a b c : Type), AcuteAngledTriangle a b c)
  (semicircles_constructed_outward_on_sides : ∀ (a b c : Type), SemicirclesOutwardConstructed a b c)
  (altitudes_intersect_semicircles : ∀ (a b c e f g : Type), AltitudesIntersectSemicircles a b c e f g)
  (base : Type)
  (hexagon : Type)
  (hexagon_is_AGEBCF : hexagon = Hexagon AGEBCF)
  (base_is_ABC : base = Triangle ABC) :
  ∃ (model : PyramidModel), ModelConstructionPossible model base hexagon :=
by
  sorry

end pyramid_construction_possible_l344_344686


namespace cube_surface_illumination_l344_344099

noncomputable def cube_edge_length : ℝ := real.sqrt (2 + real.sqrt 2)

noncomputable def beam_radius : ℝ := real.sqrt 2

noncomputable def illuminated_area_on_cube_surface : ℝ :=
  (real.pi * real.sqrt 3) / 2 + 3 * real.sqrt 6

theorem cube_surface_illumination (a : ℝ) (ρ : ℝ) (h_a : a = cube_edge_length) (h_ρ : ρ = beam_radius) :
  ∃ area : ℝ, area = illuminated_area_on_cube_surface :=
  begin
    use illuminated_area_on_cube_surface,
    rw [h_a, h_ρ],
    exact sorry -- This is where the proof would go.
  end

end cube_surface_illumination_l344_344099


namespace after_tax_dividend_amount_l344_344092

noncomputable def expected_earnings_per_share : ℝ := 0.80
noncomputable def actual_earnings_per_share : ℝ := 1.10
noncomputable def additional_dividend_rate : ℝ := 0.04
noncomputable def additional_earnings_increment : ℝ := 0.10
noncomputable def dividend_rate : ℝ := 0.5
noncomputable def tax_rate_1 : ℝ := 0.15
noncomputable def tax_rate_2 : ℝ := 0.20
noncomputable def share_count : ℕ := 300

theorem after_tax_dividend_amount : 
  let additional_earnings := actual_earnings_per_share - expected_earnings_per_share in
  let additional_dividend := (additional_earnings / additional_earnings_increment) * additional_dividend_rate in
  let expected_dividend := expected_earnings_per_share * dividend_rate in
  let total_dividend_before_tax := expected_dividend + additional_dividend in
  let tax_rate := if actual_earnings_per_share > 1.00 then tax_rate_2 else if actual_earnings_per_share > 0.80 then tax_rate_1 else 0 in
  let total_dividend_per_share_after_tax := total_dividend_before_tax * (1 - tax_rate) in
  let total_dividend_amount := share_count * total_dividend_per_share_after_tax in
  total_dividend_amount = 124.80 :=
by
  let additional_earnings := actual_earnings_per_share - expected_earnings_per_share
  let additional_dividend := (additional_earnings / additional_earnings_increment) * additional_dividend_rate
  let expected_dividend := expected_earnings_per_share * dividend_rate
  let total_dividend_before_tax := expected_dividend + additional_dividend
  let tax_rate := if actual_earnings_per_share > 1.00 then tax_rate_2 else if actual_earnings_per_share > 0.80 then tax_rate_1 else 0
  let total_dividend_per_share_after_tax := total_dividend_before_tax * (1 - tax_rate)
  let total_dividend_amount := share_count * total_dividend_per_share_after_tax
  show total_dividend_amount = 124.80
  sorry

end after_tax_dividend_amount_l344_344092


namespace find_g_l344_344360

noncomputable def g (x : ℝ) : ℝ := 2 - 4 * x

theorem find_g :
  g 0 = 2 ∧ (∀ x y : ℝ, g (x * y) = g ((3 * x ^ 2 + y ^ 2) / 4) + 3 * (x - y) ^ 2) → ∀ x : ℝ, g x = 2 - 4 * x :=
by
  sorry

end find_g_l344_344360


namespace cube_roots_of_three_distinct_primes_not_arithmetic_progression_l344_344392

theorem cube_roots_of_three_distinct_primes_not_arithmetic_progression
  (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) :
  ¬ ∃ (a b c d : ℤ), (a = real.cbrt p) ∧ (b = real.cbrt q) ∧ (c = real.cbrt r) ∧ (b = a + d) ∧ (c = a + 2 * d) := 
sorry

end cube_roots_of_three_distinct_primes_not_arithmetic_progression_l344_344392


namespace total_units_l344_344875

/-
In Lean, we need to represent the conditions and show that under these conditions, the total number of units, U, is 300.
-/

variable (U : ℕ)
variable (residences offices restaurants : ℕ)

-- Conditions given in the problem
def condition1 : Prop := residences = U / 2
def condition2 : Prop := offices = U / 4 ∧ restaurants = U / 4
def condition3 : Prop := restaurants = 75

-- The statement we want to prove
theorem total_units (h1 : condition1 U residences)
                    (h2 : condition2 U offices restaurants)
                    (h3 : condition3 U restaurants) :
                    U = 300 := 
by
  sorry

end total_units_l344_344875


namespace total_students_l344_344307

theorem total_students (T : ℕ)
  (h1 : 40 / 100 * T := 0.4 * T)
  (h2 : 10 / 100 * T := 0.1 * T)
  (h3 : 125 = 50 / 100 * T := 0.5 * T) : 
  T = 250 := 
sorry

end total_students_l344_344307


namespace sum_of_six_least_solutions_l344_344351

def tau (n : ℕ) : ℕ := sorry -- This should be defined based on divisor count, a placeholder for now.

def satisfies_condition (n : ℕ) : Prop := tau(n) + tau(n + 1) = 8

theorem sum_of_six_least_solutions : ∃ (ns : Fin₆ (ℕ)), 
  (∀ n ∈ ns, satisfies_condition n) ∧ 
  ns.to_list.sum = 800 := 
sorry

end sum_of_six_least_solutions_l344_344351


namespace students_shorter_than_yoongi_l344_344067

variable (total_students taller_than_yoongi : Nat)

theorem students_shorter_than_yoongi (h₁ : total_students = 20) (h₂ : taller_than_yoongi = 11) : 
    total_students - (taller_than_yoongi + 1) = 8 :=
by
  -- Here would be the proof
  sorry

end students_shorter_than_yoongi_l344_344067


namespace sqrt_of_300_l344_344773

theorem sqrt_of_300 : sqrt 300 = 10 * sqrt 3 := by
  have h300 : 300 = 2 ^ 2 * 3 * 5 ^ 2 := by norm_num
  rw [h300, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_pow, Real.sqrt_pow]
  all_goals { norm_num, apply_instance }
  sorry

end sqrt_of_300_l344_344773


namespace compare_abc_l344_344724

noncomputable def a := log 10 0.2
noncomputable def b := log 3 2
noncomputable def c := 5^(1/3)

theorem compare_abc : a < b ∧ b < c :=
by
  have ha : a = log 10 (0.2) := rfl
  have hb : b = log 3 (2) := rfl
  have hc : c = 5^(1/3) := rfl
  sorry

end compare_abc_l344_344724


namespace chantel_bracelets_at_end_l344_344204

-- Definitions based on conditions
def bracelets_day1 := 4
def days1 := 7
def given_away1 := 8

def bracelets_day2 := 5
def days2 := 10
def given_away2 := 12

-- Computation based on conditions
def total_bracelets := days1 * bracelets_day1 - given_away1 + days2 * bracelets_day2 - given_away2

-- The proof statement
theorem chantel_bracelets_at_end : total_bracelets = 58 := by
  sorry

end chantel_bracelets_at_end_l344_344204


namespace value_of_some_number_l344_344666

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l344_344666


namespace find_a1_l344_344245

variable {n : ℕ}

-- Definitions given in the conditions
variable (a : ℕ → ℝ)
variable {q : ℝ} (h_q_pos : 0 < q)
variable (h1 : a 2 = 1)
variable (h2 : a 2 * a 8 = 2 * (a 4) ^ 2)

-- Definition to state and prove the equivalence question = answer
theorem find_a1 (hq9 : q = real.sqrt 2) : 
  a 0 = real.sqrt 2 / 2 :=
sorry

end find_a1_l344_344245


namespace division_remainder_l344_344055

/-- The remainder when 3572 is divided by 49 is 44. -/
theorem division_remainder :
  3572 % 49 = 44 :=
by
  sorry

end division_remainder_l344_344055


namespace verify_grazing_non_overlap_verify_chain_length_percentage_l344_344213

noncomputable def grazing_non_overlap (R : ℝ) : Prop :=
  let A := 0
  let B := 2 * Real.pi / 3
  let C := 4 * Real.pi / 3
  let ρ := R / 2
  ∀ θ ∈ {A, B, C}, 
    ∀ φ ∈ {A, B, C}, 
    θ ≠ φ → ∥θ - φ∥ ≥ ρ

noncomputable def chain_length_percentage (R : ℝ) : ℝ :=
  let ρ := 0.775 * R
  ρ

theorem verify_grazing_non_overlap (R : ℝ) : grazing_non_overlap R :=
begin
  sorry,
end

theorem verify_chain_length_percentage (R : ℝ) : chain_length_percentage R = 0.775 * R :=
begin
  refl,
end

end verify_grazing_non_overlap_verify_chain_length_percentage_l344_344213


namespace men_entered_l344_344703

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l344_344703


namespace symmetric_lines_intersect_at_circumcircle_l344_344444

-- Define the given triangle and its vertices
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the concept of parallel lines passing through given points
variables (l1 l2 l3 : Type) [Line l1] [Line l2] [Line l3]
[h_parallel_1 : ∀ p : l1, is_parallel p A]
[h_parallel_2 : ∀ p : l2, is_parallel p B]
[h_parallel_3 : ∀ p : l3, is_parallel p C]

-- Define angle bisectors and symmetric reflections
variables (angle_bisector_A : Line A)
variables (angle_bisector_B : Line B)
variables (angle_bisector_C : Line C)
variables (symmetric_l1 : Line)
variables (symmetric_l2 : Line)
variables (symmetric_l3 : Line)

-- Define reflection properties
variables (h_reflect_l1 : is_symmetric l1 angle_bisector_A symmetric_l1)
variables (h_reflect_l2 : is_symmetric l2 angle_bisector_B symmetric_l2)
variables (h_reflect_l3 : is_symmetric l3 angle_bisector_C symmetric_l3)

-- Define the circumcircle of the triangle
variables (O : Type) [Circumcircle O A B C]

-- The main theorem statement
theorem symmetric_lines_intersect_at_circumcircle :
  ∃ (P : O), intersect_point symmetric_l1 symmetric_l2 P ∧ intersect_point symmetric_l2 symmetric_l3 P ∧ intersect_point symmetric_l1 symmetric_l3 P :=
sorry

end symmetric_lines_intersect_at_circumcircle_l344_344444


namespace problem_statement_l344_344239

open Real

-- Define a synchronous property
def synchronous (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  f m = g m ∧ f n = g n

/-- The given problem translated to Lean 4 statement -/
theorem problem_statement :
  ∀ (f g : ℝ → ℝ),
  (∀ m n : ℝ, synchronous f g m n → (m,n) ⊆ Icc 0 1) ∧
  ¬ synchronous (λ x, x^2) (λ x, 2 * x) 1 4 ∧
  (∃ n ∈ Ioo (1/2 : ℝ) 1, synchronous (λ x, exp x - 1) (λ x, sin (π * x)) 0 n) ∧
  (∀ (m n : ℝ), synchronous (λ x, a * log x) (λ x, x ^ 2) m n → a > 2 * exp 1) ∧
  ∃ m n : ℝ, ¬ synchronous (λ x, x + 1) (λ x, log (x + 1)) m n :=
sorry

end problem_statement_l344_344239


namespace desargues_theorem_l344_344019

open Point Line Collinear

variable {A A1 B B1 C C1 : Point}
variable {AA1 BB1 CC1 : Line}
variable {O A2 B2 C2 : Point}
variable {AB A1B1 BC B1C1 AC A1C1 : Line}

-- lines intersections as given conditions
axiom h1 : AA1 = Line_through A A1
axiom h2 : BB1 = Line_through B B1
axiom h3 : CC1 = Line_through C C1
axiom h4 : intersect AA1 BB1 CC1 = O

-- intersection points definitions
def A2 : Point := intersection BC B1C1
def B2 : Point := intersection AC A1C1
def C2 : Point := intersection AB A1B1

-- to prove they are collinear
theorem desargues_theorem (h1 : intersect AA1 BB1 CC1 = O) :
    Collinear A2 B2 C2 := by
  sorry

end desargues_theorem_l344_344019


namespace inequality_solution_range_l344_344261

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| + |x| ≤ a) ↔ a ∈ set.Ici 2 :=
sorry

end inequality_solution_range_l344_344261


namespace find_a_of_expansion_l344_344622

theorem find_a_of_expansion :
  ∃ a : ℝ, 
  let term_x2 := 80 - 10 * a in
  term_x2 = 70 → a = 1 :=
begin
  use 1,
  intro h,
  simp at h,
  sorry
end

end find_a_of_expansion_l344_344622


namespace interval_length_g_decreasing_exists_k_l344_344016

-- Conditions
def f (x a : ℝ) := a * x^2 + (a^2 + 1) * x
def I (a : ℝ) : Set ℝ := { x | f x a > 0 }
def length_I (a : ℝ) : ℝ := -(a^2 + 1) / a

-- Problem I
theorem interval_length (a : ℝ) (h : a < 0) : length_I a = -a - 1/a :=
  sorry

-- Problem II
def g (a : ℝ) : ℝ := -a - 1/a

theorem g_decreasing (a1 a2 : ℝ) (h1 : a1 ∈ Set.Iic (-1)) (h2 : a2 ∈ Set.Iic (-1)) (h3 : a1 < a2) : g a1 > g a2 :=
  sorry

-- Problem III
theorem exists_k (k : ℝ) :
  (∀ x : ℝ, g (k - real.sin x - 3) ≤ g (k^2 - real.sin x^2 - 4)) →
  k ∈ Set.Icc (-1/2) 1 :=
  sorry

end interval_length_g_decreasing_exists_k_l344_344016


namespace area_quadrilateral_SUVQ_l344_344448

theorem area_quadrilateral_SUVQ :
  ∀ (P Q R S T U V : Type)
  (PQ PR : ℝ),
  PQR_area : ℝ,
  midpoint_PQ: S = midpoint P Q,
  midpoint_PR: T = midpoint P R,
  angle_bisector_QPR_ST: U = intersection (angle_bisector (angle P Q R) (angle P R Q)) (segment S T),
  angle_bisector_QPR_QR: V = intersection (angle_bisector (angle P Q R) (angle P R Q)) (segment Q R),
  PQ = 40,
  PR = 20,
  PQR_area = 150,
  area_SUVQ = 93.75 :=
  sorry


end area_quadrilateral_SUVQ_l344_344448


namespace dodecagon_area_constraint_l344_344420

theorem dodecagon_area_constraint 
    (a : ℕ) -- side length of the square
    (N : ℕ) -- a large number with 2017 digits, breaking it down as 2 * (10^2017 - 1) / 9
    (hN : N = (2 * (10^2017 - 1)) / 9) 
    (H : ∃ n : ℕ, (n * n) = 3 * a^2 / 2) :
    False :=
by
    sorry

end dodecagon_area_constraint_l344_344420


namespace sum_of_x_values_l344_344059

theorem sum_of_x_values (x : ℝ) (h : x ≠ 3) (hx : 10 = (x^3 - 3 * x^2 - 12 * x + 6) / (x - 3)) : 
  x = real.sqrt 22 ∨ x = -real.sqrt 22 → 
  (∃ xs : list ℝ, (∀ x ∈ xs, 10 = (x^3 - 3 * x^2 - 12 * x + 6) / (x - 3)) ∧ xs.sum = 0) :=
begin
  sorry
end

end sum_of_x_values_l344_344059


namespace greatest_k_divides_l344_344200

theorem greatest_k_divides (m : ℕ) (hm : m > 0) : ∃ k : ℕ, k = 2 ∧ ∀ m, 3^k ∣ (2^(3^m) + 1) :=
by
  use 2
  intros m
  sorry

end greatest_k_divides_l344_344200


namespace percentage_increase_is_50_l344_344118

-- Defining the conditions
def new_wage : ℝ := 51
def original_wage : ℝ := 34
def increase : ℝ := new_wage - original_wage

-- Proving the required percentage increase is 50%
theorem percentage_increase_is_50 :
  (increase / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l344_344118


namespace fraction_to_decimal_l344_344962

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344962


namespace part1_part2_l344_344376

open Nat

-- Definitions based on conditions
variable (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℚ)

-- Conditions
axiom condition1 : ∀ n : ℕ, S n = 2 * a n + 2 * n - 6
axiom condition2 : T m = 127 / 258

-- Questions to be proved

-- Part 1: The sequence {a_n - 2} is geometric and general term of a_n
theorem part1 (h1 : S 1 = 2 * a 1 - 4) (h2 : ∀ n ≥ 2, S n - S (n - 1) = (2 * a n + 2 * n - 6) - (2 * a (n - 1) + 2 * (n - 1) - 6)) :
    a n - 2 = 2 ^ n := 
sorry

-- Part 2: Find the value of m
theorem part2 (a_n a_n1 : ℕ → ℕ) (m : ℕ) (T_m: ℕ → ℚ) (h3 : T_m m = 127 / 258) :
    m = 7 :=
sorry

end part1_part2_l344_344376


namespace complete_the_square_transforms_l344_344851

theorem complete_the_square_transforms (x : ℝ) :
  (x^2 + 8 * x + 7 = 0) → ((x + 4) ^ 2 = 9) :=
by
  intro h
  have step1 : x^2 + 8 * x = -7 := by sorry
  have step2 : x^2 + 8 * x + 16 = -7 + 16 := by sorry
  have step3 : (x + 4) ^ 2 = 9 := by sorry
  exact step3

end complete_the_square_transforms_l344_344851


namespace cube_opposite_face_l344_344866

theorem cube_opposite_face (A Б В Г Д Е : Prop) (adj_Г_A : Prop) (adj_Г_Б : Prop) (adj_Г_В : Prop) (adj_Г_Д : Prop) :
  (opposite_face Е Г) :=
by
  sorry

end cube_opposite_face_l344_344866


namespace layla_goldfish_count_l344_344344

def goldfish_count (total_food : ℕ) (swordtails_count : ℕ) (swordtails_food : ℕ) (guppies_count : ℕ) (guppies_food : ℕ) (goldfish_food : ℕ) : ℕ :=
  total_food - (swordtails_count * swordtails_food + guppies_count * guppies_food) / goldfish_food

theorem layla_goldfish_count : goldfish_count 12 3 2 8 1 1 = 2 := by
  sorry

end layla_goldfish_count_l344_344344


namespace barrel_500_is_4_l344_344472

-- Define the counting pattern for the barrels
def barrel_sequence (n : ℕ) : ℕ := 
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | 6 => 9
  | 7 => 8
  | 0 => 7
  | _ => 0 -- This should not be hit as n % 8 ∈ {0..7}
  end

-- Statement to prove that the 500th barrel is labeled 4
theorem barrel_500_is_4 : barrel_sequence 500 = 4 := 
  by sorry

end barrel_500_is_4_l344_344472


namespace find_q_l344_344028

-- Define the polynomial Q
def Q (x : ℝ) (p q r : ℝ) := x^3 + p * x^2 + q * x + r

-- Define mean of zeros, product of zeros, and sum of coefficients conditions
def mean_of_zeros (p : ℝ) := -p / 3
def product_of_zeros_two_at_a_time (q : ℝ) := q
def sum_of_coefficients (p q r : ℝ) := 1 + p + q + r

-- The main theorem to prove
theorem find_q (p q r : ℝ) (h1 : mean_of_zeros p = q) 
                         (h2 : q = sum_of_coefficients p q r)
                         (h3 : r = 5) : q = 2 :=
by
  sorry

end find_q_l344_344028


namespace symmetric_z_axis_correct_l344_344602

-- Definition of symmetry with respect to the z-axis
def symmetric_z_axis (M : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-M.1, -M.2, M.3)

-- Given the point M(a, b, c), we need to prove that the symmetric point with respect to the z-axis is (-a, -b, c)
theorem symmetric_z_axis_correct (a b c : ℝ) : 
  symmetric_z_axis (a, b, c) = (-a, -b, c) :=
by
  -- proof would go here
  sorry

end symmetric_z_axis_correct_l344_344602


namespace determine_a_value_l344_344611

noncomputable def a_value : ℝ :=
a

def origin : ℝ × ℝ :=
(0, 0)

def circle (x y : ℝ) : Prop :=
x^2 + y^2 = 4

def line (x y : ℝ) : Prop :=
y = x + a_value

def intersection_pts (A B : ℝ × ℝ) : Prop :=
circle A.1 A.2 ∧ circle B.1 B.2 ∧
line A.1 A.2 ∧ line B.1 B.2

def oa (A : ℝ × ℝ) : (ℝ × ℝ) :=
(A.1, A.2)

def ob (B : ℝ × ℝ) : (ℝ × ℝ) :=
(B.1, B.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

theorem determine_a_value (A B : ℝ × ℝ) (h : intersection_pts A B) :
  dot_product (oa A) (ob B) = -2 → a_value = sqrt 2 ∨ a_value = -sqrt 2 :=
sorry

end determine_a_value_l344_344611


namespace total_arrangements_l344_344445

def num_arrangements (n : ℕ) (k : ℕ) (A_min : ℕ) := (C(n, k) * A(k, k))

theorem total_arrangements : 
  ∀ (students schools : ℕ) (school_A_min : ℕ) (total : ℕ), 
  students = 5 → 
  schools = 3 → 
  school_A_min = 2 →
  total = 80 → 
  (∃ (arrangements : ℕ), arrangements = 
                       num_arrangements 5 3 2 
                       + (C(5, 2) * C(3, 1) * C(2, 2) * A(2, 2))) 
  → arrangements = total :=
by
  sorry

end total_arrangements_l344_344445


namespace bicycle_speed_B_l344_344509

theorem bicycle_speed_B 
  (distance : ℝ := 12)
  (ratio : ℝ := 1.2)
  (time_diff : ℝ := 1 / 6) : 
  ∃ (B_speed : ℝ), B_speed = 12 :=
by
  let A_speed := ratio * B_speed
  have eqn : distance / B_speed - time_diff = distance / A_speed := sorry
  exact ⟨12, sorry⟩

end bicycle_speed_B_l344_344509


namespace part1_part2_l344_344147

theorem part1 : (\sqrt 12 + \sqrt (4 / 3)) * \sqrt 3 = 8 := sorry
  
theorem part2 : \sqrt 48 - \sqrt (54 / 2) + (3 - \sqrt 3) * (3 + \sqrt 3) = \sqrt 3 + 6 := sorry

end part1_part2_l344_344147


namespace distance_origin_to_midpoint_l344_344800

theorem distance_origin_to_midpoint :
  let p1 := (-3 : ℝ, 4 : ℝ)
  let p2 := (5 : ℝ, -6 : ℝ)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  Real.sqrt ((midpoint.1 - 0)^2 + (midpoint.2 - 0)^2) = Real.sqrt 2 :=
by
  sorry -- proof is omitted

end distance_origin_to_midpoint_l344_344800


namespace sum_of_adjacent_products_l344_344436

variables {n : ℕ} {x : Fin n -> ℝ}

theorem sum_of_adjacent_products (h₁ : n ≥ 4) (h₂ : ∀ i, x i ≥ 0) (h₃ : (∑ i, x i) = 1) :
  (∑ i, x i * x ((i + 1) % n)) ≤ 1 / 4 :=
sorry

end sum_of_adjacent_products_l344_344436


namespace count_rectangles_in_grid_l344_344279

theorem count_rectangles_in_grid :
  let horizontal_strip := 1,
      vertical_strip := 1,
      horizontal_rects := 1 + 2 + 3 + 4 + 5,
      vertical_rects := 1 + 2 + 3 + 4,
      double_counted := 1
  in horizontal_rects + vertical_rects - double_counted = 24 :=
by
  -- Definitions based on conditions
  let horizontal_strip := 1
  let vertical_strip := 1
  let horizontal_rects := 1 + 2 + 3 + 4 + 5
  let vertical_rects := 1 + 2 + 3 + 4
  let double_counted := 1

  -- Assertion of equality based on problem solution
  have h : horizontal_rects + vertical_rects - double_counted = 24 :=
    calc
      horizontal_rects + vertical_rects - double_counted
      = (1 + 2 + 3 + 4 + 5) + (1 + 2 + 3 + 4) - 1 : by rfl
      = 15 + 10 - 1 : by rfl
      = 24 : by rfl

  exact h

end count_rectangles_in_grid_l344_344279


namespace rectangle_dimensions_l344_344000

-- Definitions from conditions
def is_rectangle (length width : ℝ) : Prop :=
  3 * width = length ∧ 3 * width^2 = 8 * width

-- The theorem to prove
theorem rectangle_dimensions :
  ∃ (length width : ℝ), is_rectangle length width ∧ width = 8 / 3 ∧ length = 8 := by
  sorry

end rectangle_dimensions_l344_344000


namespace some_number_value_correct_l344_344662

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l344_344662


namespace trig_expression_value_quadratic_roots_l344_344146

theorem trig_expression_value :
  (Real.tan (Real.pi / 6))^2 + 2 * Real.sin (Real.pi / 4) - 2 * Real.cos (Real.pi / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

theorem quadratic_roots :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = (-2 + Real.sqrt 2) / 2 ∨ x = (-2 - Real.sqrt 2) / 2) := by
  sorry

end trig_expression_value_quadratic_roots_l344_344146


namespace rearrange_letters_no_adjacent_repeats_l344_344275

-- Factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Problem conditions
def distinct_permutations (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  factorial (String.length word) / (factorial freq_I * factorial freq_L)

-- No-adjacent-repeated permutations
def no_adjacent_repeats (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  let total_permutations := distinct_permutations word freq_I freq_L
  let i_superletter_permutations := distinct_permutations (String.dropRight word 1) (freq_I - 1) freq_L
  let l_superletter_permutations := distinct_permutations (String.dropRight word 1) freq_I (freq_L - 1)
  let both_superletter_permutations := factorial (String.length word - 2)
  total_permutations - (i_superletter_permutations + l_superletter_permutations - both_superletter_permutations)

-- Given problem definition
def word := "BRILLIANT"
def freq_I := 2
def freq_L := 2

-- Proof problem statement
theorem rearrange_letters_no_adjacent_repeats :
  no_adjacent_repeats word freq_I freq_L = 55440 := by
  sorry

end rearrange_letters_no_adjacent_repeats_l344_344275


namespace angle_in_third_quadrant_l344_344645

variable α : Real

theorem angle_in_third_quadrant
  (h1 : Real.sin α < 0)
  (h2 : Real.tan α > 0)
  : (3 * Real.pi / 2) < α ∧ α < 2 * Real.pi := 
sorry

end angle_in_third_quadrant_l344_344645


namespace fraction_to_decimal_l344_344963

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344963


namespace proof_union_complement_l344_344267

def U := {-1, 0, 1, 2, 3}
def P := {0, 1, 2}
def Q := {-1, 0}

def complement_U_P := U \ P
def union_complement_U_P_Q := complement_U_P ∪ Q

theorem proof_union_complement :
  union_complement_U_P_Q = {-1, 0, 3} := by
  sorry

end proof_union_complement_l344_344267


namespace right_triangle_area_l344_344764

theorem right_triangle_area (a : ℝ) (r : ℝ) (area : ℝ) :
  a = 3 → r = 3 / 8 → area = 21 / 16 :=
by 
  sorry

end right_triangle_area_l344_344764


namespace expected_winnings_l344_344522

-- Define the probabilities
def prob_heads : ℚ := 1/2
def prob_tails : ℚ := 1/3
def prob_edge : ℚ := 1/6

-- Define the winnings
def win_heads : ℚ := 1
def win_tails : ℚ := 3
def lose_edge : ℚ := -5

-- Define the expected value function
def expected_value (p1 p2 p3 : ℚ) (w1 w2 w3 : ℚ) : ℚ :=
  p1 * w1 + p2 * w2 + p3 * w3

-- The expected winnings from flipping this coin
theorem expected_winnings : expected_value prob_heads prob_tails prob_edge win_heads win_tails lose_edge = 2/3 :=
by
  sorry

end expected_winnings_l344_344522


namespace min_value_a_3b_9c_l344_344370

theorem min_value_a_3b_9c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 
  a + 3 * b + 9 * c ≥ 27 := 
sorry

end min_value_a_3b_9c_l344_344370


namespace distribution_of_X_supporting_plan_1_l344_344697

-- Definitions for the conditions
def score_correct : ℕ := 5
def score_partial : ℕ := 2
def score_wrong : ℕ := 0

def strategy_time_A : ℕ := 3
def strategy_time_B : ℕ := 6

def prob_correct_11_A : ℝ := 0.8
def prob_partial_11_B : ℝ := 0.5
def prob_correct_11_B : ℝ := 0.4

def prob_correct_12_A : ℝ := 0.7
def prob_partial_12_B : ℝ := 0.6
def prob_correct_12_B : ℝ := 0.3

def penalty : ℕ := 2

-- Proof Problem (1): Distribution of X
theorem distribution_of_X : 
  (∀ X : ℕ, 
  (X = 0 ∧ P(X) = 0.03) ∨ 
  (X = 2 ∧ P(X) = 0.22) ∨ 
  (X = 4 ∧ P(X) = 0.35) ∨ 
  (X = 5 ∧ P(X) = 0.12) ∨ 
  (X = 7 ∧ P(X) = 0.28)) :=
sorry

-- Proof Problem (2): Supporting Plan 1
theorem supporting_plan_1 :
  (∀ plan : ℕ, 
  (plan = 1 ∧ expected_score_plan_1 > expected_score_plan_2 - penalty) ∨ 
  (plan = 2 ∧ expected_score_plan_2 - penalty < expected_score_plan_1)) :=
sorry

end distribution_of_X_supporting_plan_1_l344_344697


namespace solve_equation_l344_344990

noncomputable def cube_root (x : ℝ) := x^(1 / 3)

theorem solve_equation (x : ℝ) :
  cube_root x = 15 / (8 - cube_root x) →
  x = 27 ∨ x = 125 :=
by
  sorry

end solve_equation_l344_344990


namespace ratio_of_areas_l344_344368

-- mathematical objects and definitions
variables {s : ℝ}

def square_area (side : ℝ) : ℝ := side ^ 2

-- points E, F, G, H given that A, B, C, D are on the sides of square ABCD
def E := (s / 2, s * (sqrt 3 / 2))
def F := (s + s * (sqrt 3 / 2), s / 2)
def G := (s / 2, s + s * (sqrt 3 / 2))
def H := (-s * (sqrt 3 / 2), s / 2)

-- lengths of square sides
def side_length_ABCD := s
def side_length_EFGH := sqrt 8 * s / 2

-- areas of the squares
def area_ABCD := square_area s
def area_EFGH := square_area (sqrt 2 * s)

-- theorem statement
theorem ratio_of_areas : area_EFGH / area_ABCD = 2 :=
by
  sorry

end ratio_of_areas_l344_344368


namespace find_side_ab_length_l344_344300

noncomputable def triangle_abc (A B C M N: Type) : Prop :=
  ∃ G : Type,
    is_centroid_of G A B C ∧
    medians_perpendicular A M B N ∧
    median_length A M = 15 ∧
    median_length B N = 20 ∧
    distance A B = 50 / 3

theorem find_side_ab_length : ∀ {A B C M N : Type},
  triangle_abc A B C M N →
  ∃ AB : ℝ, AB = 50 / 3 :=
sorry

end find_side_ab_length_l344_344300


namespace value_of_f_l344_344168

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ set.Icc (0 : ℝ) (Real.pi / 2) then Real.sin x else sorry

theorem value_of_f :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + Real.pi) = f x)
  ∧ (∀ x : ℝ, x ∈ set.Icc (0 : ℝ) (Real.pi / 2) → f x = Real.sin x)
  → f (5 * Real.pi / 3) = Real.sqrt 3 / 2 :=
by sorry

end value_of_f_l344_344168


namespace fraction_to_decimal_l344_344959

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344959


namespace handshakes_13_people_l344_344684

-- Define the condition
def people_in_room : ℕ := 13

-- Using the combination formula to define the handshakes function
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

-- The theorem to prove the total number of handshakes
theorem handshakes_13_people : handshakes 13 = 78 := by
  sorry

end handshakes_13_people_l344_344684


namespace balloon_height_l344_344587

-- Five points A, B, C, D, and O on a flat field
def north_of (A O : ℝ × ℝ) := A = (O.1, O.2 + 1)
def west_of (B O : ℝ × ℝ) := B = (O.1 - 1, O.2)
def south_of (C O : ℝ × ℝ) := C = (O.1, O.2 - 1)
def east_of (D O : ℝ × ℝ) := D = (O.1 + 1, O.2)

-- Point H is above O
def above (H O : ℝ × ℝ × ℝ) := H = (O.1, O.2, 1)

-- Rope lengths
def rope_length_HC (H C : ℝ × ℝ × ℝ) := (H.1 - C.1)^2 + (H.2 - C.2)^2 + H.3^2 = 170^2
def rope_length_HD (H D : ℝ × ℝ × ℝ) := (H.1 - D.1)^2 + (H.2 - D.2)^2 + H.3^2 = 160^2
def rope_length_HB (H B : ℝ × ℝ × ℝ) := (H.1 - B.1)^2 + (H.2 - B.2)^2 + H.3^2 = 150^2

-- Distance CD
def distance_CD (C D : ℝ × ℝ) := (C.1 - D.1)^2 + (C.2 - D.2)^2 = 180^2

theorem balloon_height :
  ∀ (A B C D O : ℝ × ℝ) (H : ℝ × ℝ × ℝ),
  north_of A O → west_of B O → south_of C O → east_of D O → above H (O.1, O.2, 0) →
  distance_CD C D →
  rope_length_HC H C → rope_length_HD H D → rope_length_HB H B →
  H.3 = 30 * real.sqrt 41 :=
sorry

end balloon_height_l344_344587


namespace coordinates_of_foci_max_distance_l344_344629

noncomputable def ellipse := { p : ℝ × ℝ // (p.1^2)/4 + (p.2^2) = 1 }
noncomputable def circle := { p : ℝ × ℝ // p.1^2 + p.2^2 = 1 }
noncomputable def tangent_point_of_circle (m : ℝ) : (ℝ × ℝ) := (m, 0)
noncomputable def tangent_line (m k : ℝ) (p : ℝ × ℝ) : Prop := p.2 = k * (p.1 - m)
noncomputable def points_A_B (m k : ℝ) (p : ℝ × ℝ) : Prop := p ∈ ellipse ∧ tangent_line(m, k, p)
noncomputable def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem coordinates_of_foci :
  ∀ f : ℝ × ℝ, (f ∈ { x : ℝ × ℝ | x = (-real.sqrt 3, 0) ∨ x = (real.sqrt 3, 0) } ) :=
by sorry

theorem max_distance :
  ∀ m ∈ set.Ici 1 ∪ set.Iic (-1),
  ∀ A B : ℝ × ℝ, points_A_B m (m * (1 + 3/m^2)) A ∧ points_A_B m (m * (1 + 3/m^2)) B → 
  distance A B ≤ 2 :=
by sorry 

end coordinates_of_foci_max_distance_l344_344629


namespace sum_of_smallest_n_l344_344355

def tau (n : ℕ) : ℕ := n.divisors.count

theorem sum_of_smallest_n (h : ∑ (k : ℕ) in (Finset.filter 
  (λ n, tau n + tau (n + 1) = 8) (Finset.range 1000)).sort (≤) (Finset.range 6) = 73) : 
  ℕ := sorry

end sum_of_smallest_n_l344_344355


namespace solve_a_b_powers_l344_344932

theorem solve_a_b_powers :
  ∃ a b : ℂ, (a + b = 1) ∧ 
             (a^2 + b^2 = 3) ∧ 
             (a^3 + b^3 = 4) ∧ 
             (a^4 + b^4 = 7) ∧ 
             (a^5 + b^5 = 11) ∧ 
             (a^10 + b^10 = 93) :=
sorry

end solve_a_b_powers_l344_344932


namespace proof_problem_l344_344162

def satisfies_equation (c : ℝ) (x y : ℝ) : Prop :=
  (sqrt (x^3 * y) = c^c) ∧ (log c (x^(log c y)) + log c (y^(log c x)) = 5 * c^5)

def correct_values (c : ℝ) : Prop :=
  ∀ x y : ℝ, satisfies_equation c x y → (c > 0) ∧ (c <= (2 / 5)^(1/3))

theorem proof_problem : (c > 0) ∧ (c <= (2 / 5)^(1/3)) → ∃ x y : ℝ, satisfies_equation c x y := sorry

end proof_problem_l344_344162


namespace number_with_properties_exists_l344_344072

theorem number_with_properties_exists : ∃ n : ℕ, 
  (n % 2 = 1) ∧
  (n % 3 = 2) ∧
  (n % 4 = 3) ∧
  (n % 5 = 4) ∧
  (n % 6 = 5) ∧
  (n % 7 = 6) ∧
  (n % 8 = 7) ∧
  (n % 9 = 8) ∧
  (n % 10 = 9) ∧
  n = 2519 :=
begin
  sorry
end

end number_with_properties_exists_l344_344072


namespace sphere_surface_area_l344_344893

/-
Given a regular triangular pyramid P-ABC inscribed in a sphere O,
with the center O of the sphere located on the base ABC, and AB = sqrt(3),
prove that the surface area of the sphere is 4π.
-/

-- Assumptions
variables (P A B C O : Point)
variables (sphere_O : Sphere)
variables (is_regular_triangular_pyramid : RegularTriangularPyramid P A B C)
variables (inscribed_in_sphere : InscribedInSphere P A B C sphere_O)
variables (center_O_on_base_ABC : CenterOnBase O sphere_O A B C)
variables (edge_AB : Distance A B = sqrt 3)

-- Theorem
theorem sphere_surface_area : surface_area sphere_O = 4 * π := by
  sorry

end sphere_surface_area_l344_344893


namespace find_m_l344_344015

noncomputable def y := Real.logBase 2

theorem find_m (m : ℝ) : (∃ m : ℝ, (∀ x : ℝ, y (x - m) + 1 = 1 ↔ x = 3)) → m = 2 :=
by
  sorry

end find_m_l344_344015


namespace appropriate_chart_for_temperature_statistics_l344_344051

theorem appropriate_chart_for_temperature_statistics (chart_type : String) (is_line_chart : chart_type = "line chart") : chart_type = "line chart" :=
by
  sorry

end appropriate_chart_for_temperature_statistics_l344_344051


namespace minimum_expression_value_l344_344647

-- Define the mathematical function we're interested in
def expression (x y : ℝ) : ℝ :=
  Real.sqrt (4 + y^2) + Real.sqrt ((x - 2) ^ 2 + (y - 2) ^ 2) + Real.sqrt ((x - 4) ^ 2 + 1)

-- State that the minimum of this expression is 5
theorem minimum_expression_value : ∀ (x y : ℝ), expression x y ≥ 5 ∧ (∃ (x y : ℝ), expression x y = 5) :=
by
  sorry

end minimum_expression_value_l344_344647


namespace smallest_n_series_zero_l344_344359

def pi_div_2010 := Real.pi / 2010

def series (n : ℕ) : ℝ :=
2 * (∑ k in Finset.range n, (if k = 0 then Real.cos ((k+1)^2 * pi_div_2010) * Real.sin(k+1 * pi_div_2010) else 
               1.5 * Real.cos ((k+1)^2 * pi_div_2010) * Real.sin((k+1) * pi_div_2010)))

theorem smallest_n_series_zero (n : ℕ) (h : n > 0) :
  series n = 0 ↔ n = 67 :=
sorry

end smallest_n_series_zero_l344_344359


namespace trisected_triangle_segments_l344_344906

variables (A B C P Q X Y Z : Type) [AffineSpace ℝ A]
  (AC_line : Line ℝ A)
  (BC_line : Line ℝ A)
  (trisect_points : P, Q on BC_line)
  (trisect_condition : dist B P = dist P Q ∧ dist P Q = dist Q C)
  (par_line : ∀ (X Y Z : A), parallel AC_line (Line.of_points ℝ [X, Y, Z]))
  (intersections : (X ∈ line_through A B) ∧ 
                   (Y ∈ line_through A (segment_points P AC_line)) ∧ 
                   (Z ∈ line_through A (segment_points Q AC_line)))
  (trisect_section : segment_split BC_line [B, P, Q, C] [1, 1, 1])

theorem trisected_triangle_segments : dist Y Z = 3 * dist X Y := 
sorry

end trisected_triangle_segments_l344_344906


namespace compare_neg_rats_l344_344548

theorem compare_neg_rats : (-3/8 : ℚ) > (-4/9 : ℚ) :=
by sorry

end compare_neg_rats_l344_344548


namespace reciprocal_sum_hcf_lcm_l344_344861

variables (m n : ℕ)

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem reciprocal_sum_hcf_lcm (h₁ : HCF m n = 6) (h₂ : LCM m n = 210) (h₃ : m + n = 60) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 21 :=
by
  -- The proof will be inserted here.
  sorry

end reciprocal_sum_hcf_lcm_l344_344861


namespace milk_per_day_l344_344287

variable (total_milk : ℕ) (days : ℕ)

theorem milk_per_day (h1 : total_milk = 2804) (h2 : days = 10) :
  total_milk / days = 280.4 := sorry

end milk_per_day_l344_344287


namespace mean_less_than_median_of_q_l344_344857

def q := [1, 7, 18, 20, 29, 33]
def mean (l : List ℝ) : ℝ := l.sum / l.length
def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (≤)
  if l.length % 2 = 1
  then sorted.nth_le (l.length / 2) (by linarith)
  else (sorted.nth_le (l.length / 2) (by linarith) + sorted.nth_le (l.length / 2 - 1) (by linarith)) / 2

theorem mean_less_than_median_of_q :
  (median q - mean q = 1) := sorry

end mean_less_than_median_of_q_l344_344857


namespace add_pure_chocolate_to_achieve_percentage_l344_344640

/--
Given:
    Initial amount of chocolate topping: 620 ounces.
    Initial chocolate percentage: 10%.
    Desired total weight of the final mixture: 1000 ounces.
    Desired chocolate percentage in the final mixture: 70%.
Prove:
    The amount of pure chocolate to be added to achieve the desired mixture is 638 ounces.
-/
theorem add_pure_chocolate_to_achieve_percentage :
  ∃ x : ℝ,
    0.10 * 620 + x = 0.70 * 1000 ∧
    x = 638 :=
by
  sorry

end add_pure_chocolate_to_achieve_percentage_l344_344640


namespace exists_such_h_l344_344945

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end exists_such_h_l344_344945


namespace value_of_p_l344_344812

noncomputable def p_value_condition (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) : Prop :=
  (9 * p^8 * q = 36 * p^7 * q^2)

theorem value_of_p (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : p_value_condition p q h1 h2 h3) :
  p = 4 / 5 :=
by
  sorry

end value_of_p_l344_344812


namespace nina_sits_in_first_car_l344_344713

/- Definitions of the problem conditions -/
def nina_in_first_car : Prop := Nina sits in first car
def jenna_in_front_of_leah : Prop := ∀ (posJ posL : Nat), posJ + 1 = posL
def oscar_behind_jenna : Prop := ∀ (posJ posO: Nat), posJ < posO
def tyson_not_adjacent_to_leah : Prop := ∀ (posT posL: Nat), (abs (posT - posL) > 1)

/- The main theorem stating the proof problem -/
theorem nina_sits_in_first_car (H1 : nina_in_first_car)
                                (H2 : jenna_in_front_of_leah)
                                (H3 : oscar_behind_jenna)
                                (H4 : tyson_not_adjacent_to_leah) : nina_in_first_car :=
by sorry

end nina_sits_in_first_car_l344_344713


namespace proof_problem_l344_344346

open Set Function

-- Definitions of the conditions
variable {n : ℕ} (hn : n % 2 = 1) (S : Set ℕ) (f : ℕ × ℕ → ℕ)

-- The conditions of the problem
def condition1 : Prop := ∀ {r s}, r ∈ S → s ∈ S → f (r, s) = f (s, r)
def condition2 : Prop := ∀ r ∈ S, Inter (Image (λ s, f (r, s)) S) S = S

-- The goal statement
theorem proof_problem (hn : n % 2 = 1) (S : Finset ℕ) (f : ℕ × ℕ → ℕ)
  (hS : S = Finset.range (n + 1)) (h1 : ∀ {r s}, r ∈ S → s ∈ S → f (r, s) = f (s, r))
  (h2 : ∀ r ∈ S, (Finset.image (λ s, f (r, s)) S).toSet = S) :
  Finset.image (λ r, f (r, r)) S.toSet = S := by sorry

end proof_problem_l344_344346


namespace compare_two_sqrt_three_with_three_l344_344924

theorem compare_two_sqrt_three_with_three : 2 * Real.sqrt 3 > 3 :=
sorry

end compare_two_sqrt_three_with_three_l344_344924


namespace sub_one_by_repeating_decimal_l344_344985

theorem sub_one_by_repeating_decimal : (1 - (0.\overline{9})) = 0 := 
by
  -- We assume that 0.\overline{9} is mathematically equal to 1
  have h : 0.\overline{9} = 1, by sorry,
  rw h,
  exact sub_self 1

end sub_one_by_repeating_decimal_l344_344985


namespace apple_tree_total_production_l344_344128

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l344_344128


namespace concyclic_points_and_midpoint_center_l344_344474

variables {B C I_B I_C N : Type} 

-- Definition stating that internal and external bisectors intersect at 90 degrees
def bisectors_intersect_90 (angle : Type) (int_bis ext_bis : angle -> angle) : Prop :=
  ∀ (a : angle), int_bis a ≠ ext_bis a ∧ int_bis a ⊥ ext_bis a

-- Definition that I_BI_C is the diameter of the circle passing through B, C, I_B, I_C
def diameter_thru_points (I_B I_C B C : Type) [circle I_B I_C B C] : Prop :=
  ∀ (I_BI_C_circle : circle I_B I_C B C), diameter I_B I_C I_BI_C_circle

-- Definition stating that N lies on both the external bisector and the perpendicular bisector of BC
def lies_on_bisectors (N B C : Type) (ext_bis : B -> C -> Type) (perp_bis : B -> C -> Type) : Prop :=
  ∀ (b : B) (c : C), ext_bis b c = perp_bis b c ∧ N ∈ ext_bis b c ∧ N ∈ perp_bis b c

variables (angle : Type)
variables (int_bis ext_bis perp_bis : angle -> angle)
variables [circle I_B I_C B C]
variables (a : angle)

-- Given conditions
axiom h1 : bisectors_intersect_90 angle int_bis ext_bis
axiom h2 : diameter_thru_points I_B I_C B C
axiom h3 : lies_on_bisectors N B C ext_bis perp_bis

-- Prove the points are concyclic and N is the center of their circle
theorem concyclic_points_and_midpoint_center : 
  (concyclic B C I_B I_C) ∧ (center_of_circle B C I_B I_C N) :=
sorry

end concyclic_points_and_midpoint_center_l344_344474


namespace largest_rectangle_area_l344_344022

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end largest_rectangle_area_l344_344022


namespace graph_of_equation_is_two_lines_l344_344852

theorem graph_of_equation_is_two_lines : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 ↔ (x = 0 ∨ y = 0) := 
by
  sorry

end graph_of_equation_is_two_lines_l344_344852


namespace apple_tree_total_apples_l344_344131

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l344_344131


namespace water_volume_into_sea_l344_344856

def flow_rate_kmph : ℝ := 1  -- flow rate in kilometers per hour
def flow_rate_mpm : ℝ := 1000 / 60  -- converting to meters per minute, flow_rate_kmph * 1000 / 60
def river_depth : ℝ := 3  -- depth in meters
def river_width : ℝ := 55  -- width in meters

def cross_sectional_area (depth : ℝ) (width : ℝ) : ℝ :=
  depth * width

def volume_per_minute (area : ℝ) (flow_rate : ℝ) : ℝ :=
  area * flow_rate

theorem water_volume_into_sea : 
  volume_per_minute (cross_sectional_area river_depth river_width) flow_rate_mpm = 2750.55 :=
by
  sorry

end water_volume_into_sea_l344_344856


namespace six_digit_permutations_l344_344397

-- Represents a function to check if two numbers are permutations of each other.
def is_permutation (a b : ℕ) : Prop :=
  (a.to_digits.to_list.nodup ∧ b.to_digits.to_list.nodup ∧ list.diff a.to_digits.to_list b.to_digits.to_list).empty

noncomputable def N : ℕ := 142857

-- Lean 4 statement
theorem six_digit_permutations :
  (N.digits.length = 6) ∧
  (N.digits.nodup) ∧ 
  (is_permutation N (2 * N)) ∧
  (is_permutation N (3 * N)) ∧
  (is_permutation N (4 * N)) ∧
  (is_permutation N (5 * N)) ∧
  (is_permutation N (6 * N)) :=
by
  -- Placeholder for the proof
  sorry

end six_digit_permutations_l344_344397


namespace find_greatest_three_digit_number_l344_344052

def is_one_more_than_multiple_of_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k + 1

def is_three_more_than_multiple_of_5 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 5 * k + 3

theorem find_greatest_three_digit_number : ∃ n : ℕ,
  n < 1000 ∧
  n ≥ 100 ∧
  is_one_more_than_multiple_of_9 n ∧
  is_three_more_than_multiple_of_5 n ∧
  ∀ m : ℕ,
    (m < 1000 ∧ m ≥ 100 ∧ is_one_more_than_multiple_of_9 m ∧ is_three_more_than_multiple_of_5 m) → m ≤ n :=
begin
  use 973,
  split,
  { exact nat.lt_of_sub_one_lt 1000 1, },
  split,
  { linarith, },
  split,
  { unfold is_one_more_than_multiple_of_9,
    use 108,
    exact rfl, },
  split,
  { unfold is_three_more_than_multiple_of_5,
    use 194,
    exact rfl, },
  { intros m hm,
    cases hm with hm1 hm,
    cases hm with hm2 hm,
    cases hm with hm3 hm4,
    unfold is_one_more_than_multiple_of_9 at hm3,
    unfold is_three_more_than_multiple_of_5 at hm4,
    rcases hm3 with ⟨k1, hk1⟩,
    rcases hm4 with ⟨k2, hk2⟩,
    suffices : ∃ k : ℕ, m = 45 * k + 28,
    { rcases this with ⟨k, hk⟩,
      rw hk,
      have hk' : 45 * k + 28 ≤ 965 + 28 := by {
        apply add_le_add,
        norm_num,
        exact (nat.div_eq_of_eq_mul 45 21)),
      },
    },
  },
  sorry
end

end find_greatest_three_digit_number_l344_344052


namespace irreducible_rational_fraction_root_factored_Q_l344_344763

noncomputable def irreducible_rational_fraction_root (p q : ℤ) (P : polynomial ℤ) : Prop :=
  P.eval (p / q) = 0 ∧ gcd p q = 1 ∧ ∀ Q : polynomial ℤ, ¬P = Q * polynomial.X - polynomial.C p * polynomial.C q

theorem irreducible_rational_fraction_root_factored_Q
  (p q : ℤ) (P : polynomial ℤ) (hP1 : gcd p q = 1) (hP2 : P.eval (p / q) = 0) :
  ∃ Q : polynomial ℤ, P = (polynomial.C q * polynomial.X - polynomial.C p) * Q :=
sorry

end irreducible_rational_fraction_root_factored_Q_l344_344763


namespace sum_of_special_xs_l344_344207

-- Define the conditions and construct the proof statement
theorem sum_of_special_xs :
  (∑ k in Finset.range 9, (k + 1) + (1 / ((k + 1) * ((k + 1) + 1)))) = 459 / 10 :=
by
  sorry

end sum_of_special_xs_l344_344207


namespace fraction_expression_simplifies_to_313_l344_344163

theorem fraction_expression_simplifies_to_313 :
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324) /
  (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324) = 313 :=
by
  sorry

end fraction_expression_simplifies_to_313_l344_344163


namespace math_problem_l344_344156

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 1 16, x - complex.exp (2 * complex.pi * complex.I * k / 17)
noncomputable def Q (x : ℂ) : ℂ := ∏ j in finset.range 1 13, x - complex.exp (2 * complex.pi * complex.I * j / 13)

theorem math_problem :
  ∏ k in finset.range 1 16, 
    ∏ j in finset.range 1 13, 
      (complex.exp (2 * complex.pi * complex.I * j / 13) - complex.exp (2 * complex.pi * complex.I * k / 17)) = 1 :=
by
  sorry

end math_problem_l344_344156


namespace expand_product_l344_344193

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end expand_product_l344_344193


namespace find_max_min_of_y_l344_344494

def g (t : ℝ) : ℝ := 80 - 2 * t
def f (t : ℝ) : ℝ := 20 - (1 / 2) * |t - 10|
def y (t : ℝ) : ℝ := g(t) * f(t)

theorem find_max_min_of_y :
  (∀ (t : ℝ), 0 ≤ t ∧ t ≤ 20 → y t ≤ 1225) ∧ 
  (∃ (t₁ t₂ : ℝ), 0 ≤ t₁ ∧ t₁ ≤ 20 ∧ y t₁ = 1225 ∧ 
                   0 ≤ t₂ ∧ t₂ ≤ 20 ∧ y t₂ = 600) :=
by
  sorry

end find_max_min_of_y_l344_344494


namespace rate_of_barbed_wire_l344_344404

def side_length (area : ℕ) : ℕ := Nat.sqrt area

def perimeter (side : ℕ) : ℕ := 4 * side

def adjust_perimeter (perimeter gate_count gate_width : ℕ) : ℕ :=
  perimeter - gate_count * gate_width

def rate_per_meter (total_cost length : ℕ) : ℕ :=
  total_cost / length

theorem rate_of_barbed_wire
  (A : ℕ) 
  (C : ℕ) 
  (gate_count : ℕ) 
  (gate_width : ℕ) 
  (expected_rate : ℕ) :
  side_length A = 56 →
  perimeter (side_length A) = 224 →
  adjust_perimeter (perimeter (side_length A)) gate_count gate_width = 222 →
  rate_per_meter C (adjust_perimeter (perimeter (side_length A)) gate_count gate_width) = expected_rate →
  expected_rate = 6
:= by
  intro h1 h2 h3 h4
  rw [side_length, perimeter, adjust_perimeter, rate_per_meter]
  sorry

end rate_of_barbed_wire_l344_344404


namespace apple_production_total_l344_344127

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l344_344127


namespace num_lineups_example_l344_344755

def num_lineups (team_size : ℕ) (center_choices : ℕ) (other_positions : ℕ → ℕ) : ℕ :=
  center_choices * other_positions 1 * other_positions 2 * other_positions 3

theorem num_lineups_example :
  num_lineups 12 2 (λ i, match i with
                         | 1 => 11  -- point guard choices after choosing center
                         | 2 => 10  -- shooting guard choices after choosing center and point guard
                         | 3 => 9   -- small forward choices after choosing center, point guard, and shooting guard
                         | _ => 0   -- no other position
                         end) = 1980 :=
by
  sorry

end num_lineups_example_l344_344755


namespace smallest_positive_multiple_of_19_l344_344849

theorem smallest_positive_multiple_of_19 :
  ∃ (a : ℕ), 19 * a ≡ 3 [MOD 97] ∧ 19 * a = 494 := 
by
  existsi 26
  split
  · -- Prove 19 * 26 ≡ 3 [MOD 97]
    sorry
  · -- Prove 19 * 26 = 494
    sorry

end smallest_positive_multiple_of_19_l344_344849


namespace solution_l344_344643

-- Define the given condition as an axiom
axiom given_condition (x : ℤ) : 5 * x + 9 ≡ 3 [MOD 19]

-- Statement to prove
theorem solution (x : ℤ) (h : given_condition x) : 3 * x + 15 ≡ 0 [MOD 19] :=
by
  sorry

end solution_l344_344643


namespace number_of_bits_ABCDEF_16_l344_344556

theorem number_of_bits_ABCDEF_16 : 
  let n := 11240375
  in ∃ k : ℕ, k = 24 ∧ (2^k > n ∧ 2^(k - 1) ≤ n) :=
by
  sorry

end number_of_bits_ABCDEF_16_l344_344556


namespace sum_of_all_three_digit_numbers_l344_344838

theorem sum_of_all_three_digit_numbers : 
  let digits := [1, 2, 3],
      numbers := { n | ∃ a b c, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ n = 100 * a + 10 * b + c } in
  let sum := ∑ n in numbers, n in
  sum = 5994 :=
by
  sorry

end sum_of_all_three_digit_numbers_l344_344838


namespace maximum_bunnies_drum_l344_344498

-- Define the conditions as provided in the problem
def drumsticks := ℕ -- Natural number type for simplicity
def drum := ℕ -- Natural number type for simplicity

structure Bunny :=
(drum_size : drum)
(stick_length : drumsticks)

def max_drumming_bunnies (bunnies : List Bunny) : ℕ := 
  -- Actual implementation to find the maximum number of drumming bunnies
  sorry

theorem maximum_bunnies_drum (bunnies : List Bunny) (h_size : bunnies.length = 7) : max_drumming_bunnies bunnies = 6 :=
by
  -- Proof of the theorem
  sorry

end maximum_bunnies_drum_l344_344498


namespace part_a_part_b_l344_344699

-- Definitions based on conditions
def is_perpendicular {α : Type*} [Field α] (p1 p2 p3 : Point α) : Prop :=
  (p2.slope p1) * (p2.slope p3) = -1

noncomputable def length (p1 p2 : Point ℝ) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

def is_midpoint (p1 p2 p3 : Point ℝ) : Prop :=
  p3 = Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

def right_triangle (A B C : Point ℝ) : Prop :=
  (A.y = B.y + 3) ∧ (B.x = A.x) ∧ (C.y = B.y) ∧ (C.x = B.x + 4)

def BM_eq_MD (B M D : Point ℝ) : Prop :=
  length B M = length M D

def AB_eq_BD (A B D : Point ℝ) : Prop :=
  length A B = length B D

def quadrilateral_area (A B D C : Point ℝ) : ℝ :=
  triangle_area A B D + triangle_area A B C

noncomputable theory

-- Given points
def A : Point ℝ := Point.mk 0 3
def B : Point ℝ := Point.mk 0 0
def C : Point ℝ := Point.mk 4 0
def M : Point ℝ := Point.mk 2 (3 / 2)
def D : Point ℝ := some (Point.mk xD yD such_that (BM_eq_MD B M D ∧ AB_eq_BD A B D ∧ D ≠ A))

-- Proof problems
theorem part_a : right_triangle A B C → is_midpoint A C M → BM_eq_MD B M D → AB_eq_BD A B D → is_perpendicular B M D :=
sorry

theorem part_b : right_triangle A B C → is_midpoint A C M → BM_eq_MD B M D → AB_eq_BD A B D → quadrilateral_area A B D C = 10.32 :=
sorry

end part_a_part_b_l344_344699


namespace remainder_sum_1_to_15_div_11_l344_344847

theorem remainder_sum_1_to_15_div_11 : 
  let S := (15 * (15 + 1)) / 2 
  in S % 11 = 10 :=
by
  sorry

end remainder_sum_1_to_15_div_11_l344_344847


namespace minimum_distance_l344_344293

noncomputable def point_on_plane (P : ℝ × ℝ) : Prop := true

noncomputable def point_on_curve (Q : ℝ × ℝ) : Prop := (Q.1 ^ 2 + (Q.2 + 2) ^ 2 = 1)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem minimum_distance (P Q : ℝ × ℝ) (hP : point_on_plane P) (hQ : point_on_curve Q) :
  ∃ Q' : ℝ × ℝ, point_on_curve Q' ∧ distance P Q' = 1 := sorry

end minimum_distance_l344_344293


namespace quadrilateral_area_l344_344260

noncomputable def hyperbola_area : ℝ :=
  let a : ℝ := 4 in
  let y0 : ℝ := (3 / 2) in
  2 * (1 / 2) * (4 * real.sqrt 5) * (3 / 2)

theorem quadrilateral_area (a : ℝ) (ha : a > 0) (ecc : (real.sqrt (a ^ 2 + 4)) / a = real.sqrt 5 / 2)
  (y0 : ℝ) (hy0 : y0 = 3 / 2 ∨ y0 = -3 / 2) :
  hyperbola_area = 6 * real.sqrt 5 :=
by
  sorry

end quadrilateral_area_l344_344260


namespace problem_statement_l344_344188

-- Defining the problem and its conditions
def x : ℝ := sqrt (15 + sqrt (15 + sqrt (15 + sqrt (15 + ...))))

-- The statement to be proved
theorem problem_statement (hx : x = sqrt (15 + x)) : x = (1 + sqrt 61) / 2 :=
sorry

end problem_statement_l344_344188


namespace isosceles_triangle_base_length_l344_344811

def is_isosceles (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_base_length
  (x y : ℝ)
  (h1 : 2 * x + 2 * y = 16)
  (h2 : 4^2 + y^2 = x^2)
  (h3 : is_isosceles x x (2 * y) ) :
  2 * y = 6 := 
by
  sorry

end isosceles_triangle_base_length_l344_344811


namespace part1_part2_l344_344733

variables {α : ℝ} (a : ℝ × ℝ) (b : ℝ × ℝ)
def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)

def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem part1 (h1 : 0 ≤ α ∧ α < 2 * Real.pi) (h2 : a ≠ b) :
  is_perpendicular (vector_a α + vector_b) (vector_a α - vector_b) :=
sorry

theorem part2 (h : ∥vector_a α * Real.sqrt 3 + vector_b∥ = ∥vector_a α - vector_b * Real.sqrt 3∥) :
  α = Real.pi / 6 ∨ α = 7 * Real.pi / 6 :=
sorry

end part1_part2_l344_344733


namespace triangle_area_division_l344_344868

theorem triangle_area_division (T T_1 T_2 T_3 : ℝ) 
  (hT1_pos : 0 < T_1) (hT2_pos : 0 < T_2) (hT3_pos : 0 < T_3) (hT : T = T_1 + T_2 + T_3) :
  T = (Real.sqrt T_1 + Real.sqrt T_2 + Real.sqrt T_3) ^ 2 :=
sorry

end triangle_area_division_l344_344868


namespace problem1_problem2_l344_344372

noncomputable def f (x a : ℝ) := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x (-1) ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 :=
sorry

theorem problem2 (a : ℝ) : (∀ x, f x a ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end problem1_problem2_l344_344372


namespace boy_real_name_is_kolya_l344_344483

variable (days_answers : Fin 6 → String)
variable (lies_on : Fin 6 → Bool)
variable (truth_days : List (Fin 6))

-- Define the conditions
def condition_truth_days : List (Fin 6) := [0, 1] -- Suppose Thursday is 0, Friday is 1.
def condition_lies_on (d : Fin 6) : Bool := d = 2 -- Suppose Tuesday is 2.

-- The sequence of answers
def condition_days_answers : Fin 6 → String := 
  fun d => match d with
    | 0 => "Kolya"
    | 1 => "Petya"
    | 2 => "Kolya"
    | 3 => "Petya"
    | 4 => "Vasya"
    | 5 => "Petya"
    | _ => "Unknown"

-- The proof problem statement
theorem boy_real_name_is_kolya : 
  ∀ (d : Fin 6), 
  (d ∈ condition_truth_days → condition_days_answers d = "Kolya") ∧
  (condition_lies_on d → condition_days_answers d ≠ "Vasya") ∧ 
  (¬(d ∈ condition_truth_days ∨ condition_lies_on d) → True) →
  "Kolya" = "Kolya" :=
by
  sorry

end boy_real_name_is_kolya_l344_344483


namespace truck_speeds_and_length_l344_344452

variables (s t : ℝ) (v1 v2 : ℝ)
variable h1 : 0 < s

-- Conditions stating trucks meet at distances stated in the problem given time t
def meeting_condition_1 (t : ℝ) := s - 6.4 / t = v1
def meeting_condition_2 (t : ℝ) := 6.4 / t = v2

-- Condition for modified starting times and meeting conditions
def meeting_condition_mod_1 (t : ℝ) := 6.4 / (t - 1/12) = v1
def meeting_condition_mod_2 (t : ℝ) := (s - 6.4) / (t + 1/8) = v2

-- Combining the results variable to prove
theorem truck_speeds_and_length (s t v1 v2 : ℝ) (h_s : s = 16) (h_t : t = 1/4) (h_v1 : v1 = 38.4) (h_v2 : v2 = 25.6) :
  meeting_condition_1 t ∧ meeting_condition_2 t ∧ meeting_condition_mod_1 t ∧ meeting_condition_mod_2 t := by
  sorry

end truck_speeds_and_length_l344_344452


namespace cone_minimum_volume_l344_344113

theorem cone_minimum_volume (a b : ℝ) (d : ℝ) (m n p : ℕ)
  (h1 : a = 3) (h2 : b = 4) (h3 : d = 3)
  (h_volume : m = 25 ∧ n = 11 ∧ p = 24 ∧ gcd m p = 1 ∧ squarefree n ∧ 
              volume = (m * Real.pi * Real.sqrt n) / p)
  : m + n + p = 60 := 
sorry

end cone_minimum_volume_l344_344113


namespace polar_to_rectangular_equivalent_l344_344027

-- Definitions for polar coordinates and conversion to rectangular coordinates
def polarToRectangular (ρ θ : ℝ) : ℝ := ρ * Real.cos θ

-- Given condition: ρ * cos θ = 2
def polarEquation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Theorem: equivalent rectangular coordinate equation
theorem polar_to_rectangular_equivalent (ρ θ : ℝ) (h : polarEquation ρ θ) : polarToRectangular ρ θ = 2 := by
  -- Directly using the given condition which matches the conversion rule
  exact h

end polar_to_rectangular_equivalent_l344_344027


namespace pascal_tenth_row_sum_l344_344931

theorem pascal_tenth_row_sum : ∑ k in Finset.range (11), Nat.choose 10 k = 1024 :=
  sorry

end pascal_tenth_row_sum_l344_344931


namespace original_number_not_800_l344_344107

theorem original_number_not_800 (x : ℕ) (h : 10 * x = x + 720) : x ≠ 800 :=
by {
  sorry
}

end original_number_not_800_l344_344107


namespace problem1_problem2_l344_344422

variable (A B : ℝ × ℝ) (x1 x2 : ℝ)
variable (y1 y2 : ℝ)
variable (m : ℝ)

def parabola : Prop := 
  (y1^2 = 4 * x1) ∧ (y2^2 = 4 * x2)

def line_through_E : Prop := 
  (x1, y1).fst = m * (x1, y1).snd - 1 ∧ (x2, y2).fst = m * (x2, y2).snd - 1

def ab_midpoint_x : Prop := 
  (x1 + x2) / 2 = 3

def midpoint_condition : Prop := 
  (x1 + x2 = 6)

def product_condition : Prop := 
  (y1 + y2 = 4 * m) ∧ (y1 * y2 = 8)

def delta_condition : Prop := 
  (16 * m^2 - 16 > 0)

noncomputable def af_bf_sum : Prop := 
  |x1 + 1| + |x2 + 1| = 8

noncomputable def af_bf_product : Prop := 
  |x1 + 1| * |x2 + 1| ∈ set.Ioi (4 : ℝ)

theorem problem1 (h_parabola : parabola)
  (h_mid_x : ab_midpoint_x) 
  (h_midpoint_condition : midpoint_condition) : 
  af_bf_sum := 
sorry

theorem problem2 (h_line : line_through_E)
  (h_prod_condition : product_condition) 
  (h_delta_condition : delta_condition) 
  (h_m_gt_1 : m^2 > 1) : 
  af_bf_product := 
sorry

end problem1_problem2_l344_344422


namespace convex_quadrilaterals_count_l344_344927

/-- 
Given 12 distinct points on the circumference of a circle,
prove that the number of convex quadrilaterals such that two specific points
A and B are not both vertices is 450.
-/
theorem convex_quadrilaterals_count :
  let n := 12 in
  let binom := λ (n k : ℕ), nat.choose n k in
  binom n 4 - binom 10 2 = 450 :=
by
  let n := 12
  let binom := λ (n k : ℕ), nat.choose n k
  calc
    binom n 4 - binom 10 2 = 495 - 45 : by rw [nat.choose, nat.choose]
                        ... = 450 : by norm_num

end convex_quadrilaterals_count_l344_344927


namespace fraction_to_decimal_l344_344958

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344958


namespace circle_tangent_distance_l344_344296

theorem circle_tangent_distance (r1 r2 d: ℝ) (h1: d = r2 - r1 ∨ d = r1 + r2) (h_radii: r1 = 1) (h_radius2: r2 = 7) :
  d = 6 ∨ d = 8 :=
by
  subst h_radii
  subst h_radius2
  finish

end circle_tangent_distance_l344_344296


namespace boundary_shadow_of_sphere_l344_344820

noncomputable def sphere_shadow_boundary (x : ℝ) : ℝ :=
  x^2 / 4 - 1

theorem boundary_shadow_of_sphere :
  ∀ x, ∃ f : ℝ → ℝ, f x = sphere_shadow_boundary x ∧
                     ∃ P : ℝ × ℝ × ℝ, P = (0, -1, 2) ∧
                     ∃ O : ℝ × ℝ × ℝ, O = (0, 0, 1) ∧
                     ∃ r : ℝ, r = 1 ∧
                     (∀ X : ℝ × ℝ × ℝ, ∃ y : ℝ, X = (x, y, 0) ∧
                     tangent P X O r ∧
                     f x = y) :=
by
  sorry


end boundary_shadow_of_sphere_l344_344820


namespace calculation_result_l344_344145

theorem calculation_result : 
  (16 = 2^4) → 
  (8 = 2^3) → 
  (4 = 2^2) → 
  (16^6 * 8^3 / 4^10 = 8192) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end calculation_result_l344_344145


namespace sqrt_eq_cubrt_l344_344297

theorem sqrt_eq_cubrt (x : ℝ) (h : Real.sqrt x = x^(1/3)) : x = 0 ∨ x = 1 :=
by
  sorry

end sqrt_eq_cubrt_l344_344297


namespace amy_deleted_23_files_l344_344123

def files_initial (music_files : Nat) (video_files : Nat) : Nat := music_files + video_files

def files_remaining (initial_files : Nat) (deleted_files : Nat) : Nat := initial_files - deleted_files

theorem amy_deleted_23_files (music_files : Nat) (video_files : Nat) (remaining_files : Nat) 
  (h1 : music_files = 4) (h2 : video_files = 21) (h3 : remaining_files = 2) : 
  (files_initial music_files video_files) - remaining_files = 23 :=
by
  rw [h1, h2, h3]
  simp [files_initial, files_remaining]
  sorry

end amy_deleted_23_files_l344_344123


namespace solve_for_d_l344_344832

noncomputable def calc_area (a b c : ℝ) : ℝ := 
    let s := (a + b + c) / 2
    sqrt (s * (s - a) * (s - b) * (s - c))

theorem solve_for_d
  (c d : ℝ)
  (h_perimeter : 2 * c + d = 22)
  (h_area : d * sqrt (c^2 - (d / 2)^2) = 10 * sqrt 11) :
  d = 7 :=
by
  sorry

end solve_for_d_l344_344832


namespace fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l344_344836

-- Conditions
def square_side : ℕ := 1
def area_per_square : ℕ := square_side * square_side
def area_of_stair (n : ℕ) : ℕ := (n * (n + 1)) / 2
def perimeter_of_stair (n : ℕ) : ℕ := 4 * n

-- Part (a)
theorem fifth_stair_area_and_perimeter :
  area_of_stair 5 = 15 ∧ perimeter_of_stair 5 = 20 := by
  sorry

-- Part (b)
theorem stair_for_area_78 :
  ∃ n, area_of_stair n = 78 ∧ n = 12 := by
  sorry

-- Part (c)
theorem stair_for_perimeter_100 :
  ∃ n, perimeter_of_stair n = 100 ∧ n = 25 := by
  sorry

end fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l344_344836


namespace fixed_points_a1_bm2_range_of_a_minimum_b_l344_344554

def f (a b x : ℝ) : ℝ := a * x ^ 2 + (b + 1) * x + b - 1

def is_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = x₀

theorem fixed_points_a1_bm2 : 
  ∃ x, is_fixed_point (f 1 (-2)) x ∧ (x = -1 ∨ x = 3) := 
sorry

theorem range_of_a (a b : ℝ) (h : ∀ b, ∃ x₁ x₂, is_fixed_point (f a b) x₁ ∧ 
    is_fixed_point (f a b) x₂ ∧ x₁ ≠ x₂) : 
  0 < a ∧ a < 1 := 
sorry

theorem minimum_b (a : ℝ) (h : 0 < a ∧ a < 1) :
  let g (x : ℝ) := -x + a / (5 * a^2 - 4 * a + 1)
  ∃ C : ℝ, C ∈ g ∧ ∃ b, (∃ x₁ x₂, is_fixed_point (f a b) x₁ ∧ 
    is_fixed_point (f a b) x₂ ∧ x₁ ≠ x₂) ∧ 
    (x₁ + x₂)/2 = C ∧ b = -a^2 / ((1/a - 2)^2 + 1) := 
begin
  have : a = 1 / 2, sorry,
  let b := -1,
  use (-1),
  split,
  exact sorry,
  use b,
  split,
  exact sorry,
  split,
  exact sorry,
  exact sorry
end

end fixed_points_a1_bm2_range_of_a_minimum_b_l344_344554


namespace value_of_some_number_l344_344667

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l344_344667


namespace quadratic_inequality_solution_range_l344_344676

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + m * x + 2 > 0) ↔ m > -3 := 
sorry

end quadratic_inequality_solution_range_l344_344676


namespace projections_parallel_or_concurrent_l344_344427

variables (P A B C A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : Point)

-- Assumptions
-- P is a point such that:
-- - A₂, B₂, and C₂ are the projections of P onto CA, AB, and BC
-- - A₃, B₃, and C₃ are the projections of P onto AA₁, BB₁, and CC₁
-- Prove that:
-- - A₂A₃, B₂B₃, and C₂C₃ are concurrent or parallel
-- - These lines are parallel if ∠ C₃C₂B + ∠ B + ∠ A₃A₂B = 360°

theorem projections_parallel_or_concurrent 
    (h_proj_CA : projection P CA A₂)
    (h_proj_AB : projection P AB B₂)
    (h_proj_BC : projection P BC C₂)
    (h_proj_AA₁ : projection P AA₁ A₃)
    (h_proj_BB₁ : projection P BB₁ B₃)
    (h_proj_CC₁ : projection P CC₁ C₃) :
   (concurrent A₂ A₃ B₂ B₃ C₂ C₃ ∨ parallel A₂ A₃ B₂ B₃ C₂ C₃) ∧ 
   (∀ h_parallel : parallel A₂ A₃ B₂ B₃ C₂ C₃, 
    ∡(C₃, C₂, B) + ∡(B) + ∡(A₃, A₂, B) = 360) :=
sorry

end projections_parallel_or_concurrent_l344_344427


namespace some_number_value_correct_l344_344660

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l344_344660


namespace total_games_single_elimination_l344_344116

theorem total_games_single_elimination (teams : ℕ) (h_teams : teams = 24)
  (preliminary_matches : ℕ) (h_preliminary_matches : preliminary_matches = 8)
  (preliminary_teams : ℕ) (h_preliminary_teams : preliminary_teams = 16)
  (idle_teams : ℕ) (h_idle_teams : idle_teams = 8)
  (main_draw_teams : ℕ) (h_main_draw_teams : main_draw_teams = 16) :
  (games : ℕ) -> games = 23 :=
by
  sorry

end total_games_single_elimination_l344_344116


namespace total_dots_of_assembled_dice_l344_344041

def opposite_faces_sum (x : ℕ) : Prop := x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6

theorem total_dots_of_assembled_dice (D : Type) [fintype D] [decidable_eq D] 
    (dice : fin 7 → (fin 6 → fin 6))
    (opposite_sum_7 : ∀ i, opposite_faces_sum (dice i 0 + dice i 5) ∧ 
                             opposite_faces_sum (dice i 1 + dice i 4) ∧
                             opposite_faces_sum (dice i 2 + dice i 3))
    (gluing_faces_same_dots : ∀ i j, (dice i 0 = dice j 0) ∨ 
                                      (dice i 1 = dice j 1) ∨ 
                                      (dice i 2 = dice j 2) ∨ 
                                      (dice i 3 = dice j 3) ∨ 
                                      (dice i 4 = dice j 4) ∨ 
                                      (dice i 5 = dice j 5))
    (visible_faces_non_erased : ∀ i, (dice i 0 + dice i 1 + dice i 2 + dice i 3 + dice i 4 + dice i 5) ∈ (set.of_fintype (@finset.univ {n // n ≠ dice i 0})) → 54)
    (total_visible_faces : fin 9)
    : (sum (list.map (@sum int (list.of_fn (λ i, list.of_fn (dice i)))) (@list.of_univs _ dice)).univ = 75) :=
by
  sorry

end total_dots_of_assembled_dice_l344_344041


namespace similarity_transformation_result_l344_344690

-- Define the original coordinates of point A and the similarity ratio
def A : ℝ × ℝ := (2, 2)
def ratio : ℝ := 2

-- Define the similarity transformation that scales coordinates, optionally considering reflection
def similarity_transform (p : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  (r * p.1, r * p.2)

-- Use Lean to state the theorem based on the given conditions and expected answer
theorem similarity_transformation_result :
  similarity_transform A ratio = (4, 4) ∨ similarity_transform A (-ratio) = (-4, -4) :=
by
  sorry

end similarity_transformation_result_l344_344690


namespace total_time_for_12000_dolls_l344_344343

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end total_time_for_12000_dolls_l344_344343


namespace choose_two_fruits_l344_344286

theorem choose_two_fruits :
  let n := 5
  let k := 2
  Nat.choose n k = 10 := 
by 
  let n := 5
  let k := 2
  sorry

end choose_two_fruits_l344_344286


namespace max_wx_plus_xy_plus_yz_l344_344361

theorem max_wx_plus_xy_plus_yz (w x y z : ℝ) (h1 : w ≥ 0) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) (h_sum : w + x + y + z = 200) : wx + xy + yz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_l344_344361


namespace find_constant_d_l344_344995

/-- Prove the divisor remainder property -/
theorem find_constant_d (d : ℚ) :
  let P := λ x : ℚ, 2 * x^3 + d * x^2 - 17 * x + 53,
      D := λ x : ℚ, 2 * x + 7,
      R := 10 in
  ∀ x : ℚ, P x % D x = R → d = 45.14 :=
sorry

end find_constant_d_l344_344995


namespace standard_equation_of_ellipse_range_of_k_l344_344235

-- Define the given conditions and problem statement in Lean.

variables {a b c : ℝ} (k : ℝ)
axiom ellipse_eq : ∀ {x y : ℝ}, (x^2)/(a^2) + (y^2)/(b^2) = 1
axiom eccentricity : c/a = √3/2
axiom ellipse_condition : a > b ∧ b > 0
axiom point_B : ellipse_eq 0 1

theorem standard_equation_of_ellipse :
    a = 2 → b = 1 → c = √3 → (∀ {x y : ℝ}, (x^2)/4 + y^2 = 1) :=
sorry

theorem range_of_k :
    ( ∀ {x1 y1 x2 y2 : ℝ}, y1 = k * (x1 + 2) → y2 = k * (x2 + 2) → 
        (x1^2)/4 + y1^2 = 1 → (x2^2)/4 + y2^2 = 1 →
        (-2 * x2 - y2 + 1 < 0) ) 
    → k ∈ Ioo (-3 / 10 : ℝ) (1 / 2 : ℝ) :=
sorry

end standard_equation_of_ellipse_range_of_k_l344_344235


namespace seq_is_arithmetic_l344_344298

-- Define the sequence sum S_n and the sequence a_n
noncomputable def S (a : ℕ) (n : ℕ) : ℕ := a * n^2 + n
noncomputable def a_n (a : ℕ) (n : ℕ) : ℕ := S a n - S a (n - 1)

-- Define the property of being an arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → (a_n (n + 1) : ℤ) - (a_n n : ℤ) = d

-- The theorem to be proven
theorem seq_is_arithmetic (a : ℕ) (h : 0 < a) : is_arithmetic_seq (a_n a) :=
by
  sorry

end seq_is_arithmetic_l344_344298


namespace b_12_is_90_l344_344723

-- Definition of the sequence
def b : ℕ → ℕ
| 1 := 2
| (k + 2) := b (k + 1) + b 1 + (k + 1) * 1

theorem b_12_is_90 : b 12 = 90 := by
  sorry

end b_12_is_90_l344_344723


namespace line_slope_intercept_l344_344817

theorem line_slope_intercept :
  ∃ k b, (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → y = k * x + b) ∧ k = 2/3 ∧ b = 2 :=
by
  sorry

end line_slope_intercept_l344_344817


namespace sum_of_capacities_l344_344455

-- Define the capacity of a string as the number of times "10" occurs in the string
def capacity (s : String) : ℕ :=
  s.each_stride 2 |>.filter (λ substr => substr = "10").length

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  (nat.choose n k)

-- Define the sum of capacities of all arrangements of 50 "0"s and 50 "1"s
theorem sum_of_capacities : ∑ s in finset.univ.image (λ p : finset (fin 100), ...) (capacity s) = 99 * binom 98 49 :=
sorry

end sum_of_capacities_l344_344455


namespace problem1_solution_problem2_solution_problem3_solution_problem4_solution_l344_344545

noncomputable def problem1 : Int :=
  (-7) - (+5) + (-4) - (-10)

theorem problem1_solution : problem1 = -6 := by
  sorry

noncomputable def problem2 : Int :=
  -1 + 5 / (-1 / 4) * (-4)

theorem problem2_solution : problem2 = 79 := by
  sorry

noncomputable def problem3 : Int :=
  -1^4 - (1 / 7) * (2 - (-3)^2)

theorem problem3_solution : problem3 = 0 := by
  sorry

noncomputable def problem4 : Int :=
  -36 * (5 / 6 - 4 / 9 + 2 / 3)

theorem problem4_solution : problem4 = -38 := by
  sorry

end problem1_solution_problem2_solution_problem3_solution_problem4_solution_l344_344545


namespace new_sales_tax_rate_l344_344430

-- Define the given constants
def original_sales_tax_rate : ℚ := 7 / 200
def market_price : ℚ := 10800
def savings : ℚ := 18

-- Define the original and new tax amounts
def original_tax : ℚ := market_price * original_sales_tax_rate
def new_tax : ℚ := original_tax - savings

-- Define the goal: 
theorem new_sales_tax_rate :
  let x := new_tax * 100 / market_price in
  x = 3 + 1 / 3 :=
by
  sorry

end new_sales_tax_rate_l344_344430


namespace circumcenter_on_AD_l344_344757

open EuclideanGeometry -- Assume we are in the context of Euclidean geometry

theorem circumcenter_on_AD 
  (A B C D : Point) 
  (angle_ABC angle_ACB : Angle)
  (acute_angle_triangle_ABC : is_acute A B C)
  (outside_D : is_outside A B C D)
  (angle_condition_1 : ∠ ABC + ∠ ABD = 180)
  (angle_condition_2 : ∠ ACB + ∠ ACD = 180) :
  circumcenter A B C ∈ segment A D :=
sorry -- Proof to be filled in

end circumcenter_on_AD_l344_344757


namespace fraction_to_decimal_l344_344972

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344972


namespace coplanar_condition_l344_344720

/-- Define vectors for points P, Q, R, S and the origin O -/
variables {V : Type*} [inner_product_space ℝ V]
variables (P Q R S O : V)

/-- Given condition -/
def vec_condition (P Q R S O : V) (k : ℝ) :=
  4 • (P - O) - 3 • (Q - O) + 6 • (R - O) + k • (S - O) = (0 : V)

/-- Prove that k = -7 ensures P, Q, R, S are coplanar -/
theorem coplanar_condition (P Q R S O : V) : 
  vec_condition P Q R S O (-7) → coplanar ℝ ![P, Q, R, S] :=
begin
  sorry
end

end coplanar_condition_l344_344720


namespace total_households_surveyed_l344_344495

-- Define the conditions
variables (households : ℕ)
variables (neither_B_W : ℕ) (only_W : ℕ) (both_B_W : ℕ)
variable (ratio_B_only_to_both : ℕ)
variables (only_B : ℕ)

-- Specify the given conditions
def condition_1 := neither_B_W = 80
def condition_2 := only_W = 60
def condition_3 := both_B_W = 40
def condition_4 := ratio_B_only_to_both = 3
def condition_5 := only_B = 3 * both_B_W

-- The proof statement to complete later
theorem total_households_surveyed
  (h1 : condition_1)
  (h2 : condition_2)
  (h3 : condition_3)
  (h4 : condition_4)
  (h5 : condition_5) :
  households = neither_B_W + only_W + only_B + both_B_W :=
begin
  sorry,
end

#eval total_households_surveyed 300 sorry sorry sorry sorry sorry

end total_households_surveyed_l344_344495


namespace last_disc_is_blue_l344_344479

def initial_state := (R B Y : ℕ)
def draw_same_color (s : initial_state) : initial_state := s
def draw_different_color (s : initial_state) : initial_state :=
  match s with
  | (R, B, Y) => ((R - 1), (B - 1), (Y + 1))

def process (s : initial_state) : initial_state :=
  if s.1 = 0 ∨ s.2 = 0 ∨ s.3 = 0 then s
  else if s.1 % 2 = s.2 % 2 then draw_same_color s
  else if s.2 % 2 = s.3 % 2 then draw_same_color s
  else draw_different_color s

noncomputable def reduce_to_one (s : initial_state) : ℕ × ℕ × ℕ :=
  if s.1 + s.2 = 0 then (0, 0, 1)
  else if s.2 + s.3 = 0 then (0, 1, 0)
  else if s.1 + s.3 = 0 then (1, 0, 0)
  else sorry  -- representing the process of reduction to a single disc or homogeneous color state

theorem last_disc_is_blue (R B Y : ℕ) (hR : R = 7) (hB : B = 8) (hY : Y = 9) :
  reduce_to_one (R, B, Y) = (0, 1, 0) :=
sorry

end last_disc_is_blue_l344_344479


namespace nested_sqrt_solution_l344_344184

theorem nested_sqrt_solution : ∃ x : ℝ, x = sqrt (15 + x) ∧ x = (1 + sqrt 61) / 2 :=
by
  sorry

end nested_sqrt_solution_l344_344184


namespace find_special_number_l344_344194

theorem find_special_number :
  ∃ (n : ℤ), 
  (0 < n) ∧ 
  (digits n 10).length = 1000 ∧ 
  (∀ d ∈ digits n 10, d ≠ 0) ∧ 
  (let m := (sum (map (λ p, prod p.fst p.snd) (pairs (digits n 10)))) in 
    m ∣ n) ∧ 
  n = ((10 ^ 1000 - 1) / 9 + 3 * 10 ^ 423) :=
by
  sorry

end find_special_number_l344_344194


namespace parabola_equation_exists_line_m_equation_exists_l344_344501

noncomputable def problem_1 : Prop :=
  ∃ (p : ℝ), p > 0 ∧ (∀ (x y : ℝ), x^2 = 2 * p * y → y = x^2 / (2 * p)) ∧ 
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 = 2 * p * y1 → x2^2 = 2 * p * y2 → 
    (y1 + y2 = 8 - p) ∧ ((y1 + y2) / 2 = 3) → p = 2)

noncomputable def problem_2 : Prop :=
  ∃ (k : ℝ), (k^2 = 1 / 4) ∧ (∀ (x : ℝ), (x^2 - 4 * k * x - 24 = 0) → 
    (∃ (x1 x2 : ℝ), x1 + x2 = 4 * k ∧ x1 * x2 = -24)) ∧
  (∀ (x1 x2 : ℝ), x1^2 = 4 * (k * x1 + 6) ∧ x2^2 = 4 * (k * x2 + 6) → 
    ∀ (x3 x4 : ℝ), (x1 * x2) ^ 2 - 4 * ((x1 + x2) ^ 2 - 2 * x1 * x2) + 16 + 16 * x1 * x2 = 0 → 
    (k = 1 / 2 ∨ k = -1 / 2))

theorem parabola_equation_exists : problem_1 :=
by {
  sorry
}

theorem line_m_equation_exists : problem_2 :=
by {
  sorry
}

end parabola_equation_exists_line_m_equation_exists_l344_344501


namespace triangle_angle_y_l344_344844

theorem triangle_angle_y (y : ℝ) (h : y + 3 * y + 45 = 180) : y = 33.75 :=
by
  have h1 : 4 * y + 45 = 180 := by sorry
  have h2 : 4 * y = 135 := by sorry
  have h3 : y = 33.75 := by sorry
  exact h3

end triangle_angle_y_l344_344844


namespace problem_l344_344601

variable (R : Type*) [LinearOrderedField R]
variable {f : R → R}

-- Conditions
axiom axiom1 : ∀ x : R, f (1 + x) = f (1 - x)
axiom axiom2 : ∀ x : R, f (1 - x) = 1 - f x
axiom axiom3 : ∀ x1 x2 : R, x1 < x2 → x1 ∈ Icc (0 : R) 1 → x2 ∈ Icc (0 : R) 1 → f x1 ≤ f x2
axiom axiom4 : f 0 = 0
axiom axiom5 : ∀ x : R, f (x / 3) = (1 / 2) * f x

-- Theorem
theorem problem : f (-5 / 3) + f (1 / 8) = 3 / 4 :=
by sorry

end problem_l344_344601


namespace location_of_z_in_quadrant_l344_344669

-- Define the complex number z
def z : ℂ := 3 - I

-- Define the coordinates corresponding to z in the complex plane
def point : ℝ × ℝ := (z.re, z.im)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Statement of the theorem
theorem location_of_z_in_quadrant (h : z = 3 - I) : in_fourth_quadrant point :=
by
  -- Proof would go here
  sorry

end location_of_z_in_quadrant_l344_344669


namespace determine_d_iff_l344_344561

theorem determine_d_iff (x : ℝ) : 
  (x ∈ Set.Ioo (-5/2) 3) ↔ (x * (2 * x + 3) < 15) :=
by
  sorry

end determine_d_iff_l344_344561


namespace mark_new_phone_plan_cost_l344_344743

theorem mark_new_phone_plan_cost (old_cost : ℕ) (h_old_cost : old_cost = 150) : 
  let new_cost := old_cost + (0.3 * old_cost) in 
  new_cost = 195 :=
by 
  sorry

end mark_new_phone_plan_cost_l344_344743


namespace maximal_product_of_geometric_sequence_l344_344305

-- Define the geometric sequence
def geometric_sequence (a₁ : ℤ) (q : ℚ) (n : ℕ) : ℚ :=
  a₁ * (q^(n - 1))

-- Define the product of the first n terms of the sequence
def product_of_sequence (a₁ : ℤ) (q : ℚ) (n : ℕ) : ℚ :=
  (List.iota n).map (λ k => geometric_sequence a₁ q (k + 1)).prod

theorem maximal_product_of_geometric_sequence :
  ∀ (n : ℕ), n > 0 → n = 12 → 
    product_of_sequence 1536 (-1/2) 12 = 
      max (λ m => product_of_sequence 1536 (-1/2) m) {m : ℕ // m > 0} :=
by
  -- Proof omitted
  sorry

end maximal_product_of_geometric_sequence_l344_344305


namespace samantha_total_expenditure_l344_344138

def cost_per_wash : ℝ := 4
def num_washes : ℕ := 3
def special_soap_cost : ℝ := 2.50
def num_dryers : ℕ := 4
def drying_time_per_dryer : ℕ := 45
def cost_per_10_mins_drying : ℝ := 0.25
def membership_fee : ℝ := 10

theorem samantha_total_expenditure :
  let total_washing_cost := num_washes * cost_per_wash
  let total_soap_cost := special_soap_cost
  let num_drying_intervals := (drying_time_per_dryer / 10).ceil
  let total_drying_cost := num_dryers * num_drying_intervals * cost_per_10_mins_drying
  total_washing_cost + total_soap_cost + total_drying_cost + membership_fee = 29.50 := by
  sorry

end samantha_total_expenditure_l344_344138


namespace evaluate_expression_l344_344208

noncomputable def greatest_integer (x : Real) : Int := ⌊x⌋

theorem evaluate_expression (y : Real) (h : y = 7.2) :
  greatest_integer 6.5 * greatest_integer (2 / 3)
  + greatest_integer 2 * y
  + greatest_integer 8.4 - 6.0 = 16.4 := by
  simp [greatest_integer, h]
  sorry

end evaluate_expression_l344_344208


namespace probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l344_344783

open BigOperators

/-- Suppose 30 balls are tossed independently and at random into one 
of the 6 bins. Let p be the probability that one bin ends up with 3 
balls, another with 6 balls, another with 5, another with 4, another 
with 2, and the last one with 10 balls. Let q be the probability 
that each bin ends up with 5 balls. Calculate p / q. 
-/
theorem probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5 :
  (Nat.factorial 5 ^ 6 : ℚ) / ((Nat.factorial 3:ℚ) * Nat.factorial 6 * Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 10) = 0.125 := 
sorry

end probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l344_344783


namespace apple_production_total_l344_344125

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l344_344125


namespace math_problem_l344_344919

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end math_problem_l344_344919


namespace total_machine_operation_time_l344_344340

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end total_machine_operation_time_l344_344340


namespace log_a_lt_0_l344_344623

theorem log_a_lt_0 (a : ℝ) (h : 2^a - 1/a = 0) : log 10 a < 0 := 
sorry

end log_a_lt_0_l344_344623


namespace other_root_l344_344246

/-- Given the quadratic equation x^2 - 3x + k = 0 has one root as 1, 
    prove that the other root is 2. -/
theorem other_root (k : ℝ) (h : 1^2 - 3 * 1 + k = 0) : 
  2^2 - 3 * 2 + k = 0 := 
by 
  sorry

end other_root_l344_344246


namespace r_exceeds_s_by_two_l344_344268

theorem r_exceeds_s_by_two (x y r s : ℝ) (h1 : 3 * x + 2 * y = 16) (h2 : 5 * x + 3 * y = 26)
  (hr : r = x) (hs : s = y) : r - s = 2 :=
by
  sorry

end r_exceeds_s_by_two_l344_344268


namespace sum_of_S_p_l344_344205

def arithmetic_prog_sum (p : Nat) : Int :=
  25 * (149 * p - 49)

def total_sum : Int :=
  (List.range 10).sum (λ p => arithmetic_prog_sum (p + 1))

theorem sum_of_S_p : total_sum = 192625 := by
  sorry

end sum_of_S_p_l344_344205


namespace valid_starting_lineups_correct_l344_344756

-- Define the parameters from the problem
def volleyball_team : Finset ℕ := Finset.range 18
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the main computation: total lineups excluding those where all quadruplets are chosen
noncomputable def valid_starting_lineups : ℕ :=
  (volleyball_team.card.choose 7) - ((volleyball_team \ quadruplets).card.choose 3)

-- The theorem states that the number of valid starting lineups is 31460
theorem valid_starting_lineups_correct : valid_starting_lineups = 31460 := by
  sorry

end valid_starting_lineups_correct_l344_344756


namespace intersection_volume_l344_344477

-- Define the hypercube vertices in 4-dimensional space
def hypercube_vertices : List (ℝ × ℝ × ℝ × ℝ) :=
  [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
   (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
   (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
   (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]

-- Define the hyperplane equation as a predicate
def hyperplane (p : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z, w) := p
  x + y + z + w = 2

-- Filter vertices to find those lying on the hyperplane
def hyperplane_vertices : List (ℝ × ℝ × ℝ × ℝ) :=
  hypercube_vertices.filter hyperplane

-- Define the theorem
theorem intersection_volume :
  ∑ (v : ℝ × ℝ × ℝ × ℝ) in hyperplane_vertices.to_finset, 1 / hyperplane_vertices.length = 1 :=
by { sorry }

end intersection_volume_l344_344477


namespace sum_of_specific_divisors_l344_344585

theorem sum_of_specific_divisors :
  let N := 19^88 - 1 in
  ∑ d in {d | d ∣ N ∧ ∃ a b : ℕ, d = 2^a * 3^b}, d = 819 :=
by
  let N := 19^88 - 1
  have h : ∑ d in {d | d ∣ N ∧ ∃ a b : ℕ, d = 2^a * 3^b}, d = 819
  have prime_factors : ∑ a in finset.range 6, 2^a = 63 ∧ ∑ b in finset.range 3, 3^b = 13
  calc
    ∑ d in {d | d ∣ N ∧ ∃ a b : ℕ, d = 2^a * 3^b}, d 
        = ∑ a in finset.range 6, 2^a * ∑ b in finset.range 3, 3^b : by sorry
    ... = 63 * 13 : by sorry
    ... = 819 : by sorry
  have : ∑ d in {d | d ∣ N ∧ ∃ a b : ℕ, d = 2^a * 3^b}, d = 819 from calc_sums h -- Placeholder for actual sums calculation
  exact this 

end sum_of_specific_divisors_l344_344585


namespace lines_not_parallel_if_perpendicular_to_same_line_l344_344462

theorem lines_not_parallel_if_perpendicular_to_same_line 
  (L M N : Type) [linear_ordered_field L] [metric_space M] [normed_group M] [inner_product_space L M] 
  (l m n : M) 
  (hlm : ∃ n, inner_product_space.is_orthogonal l n ∧ inner_product_space.is_orthogonal m n) :
  ¬ inner_product_space.is_parallel l m :=
sorry

end lines_not_parallel_if_perpendicular_to_same_line_l344_344462


namespace cubic_root_equation_solution_l344_344992

theorem cubic_root_equation_solution (x : ℝ) : ∃ y : ℝ, y = real.cbrt x ∧ y = 15 / (8 - y) ↔ x = 27 ∨ x = 125 :=
by
  sorry

end cubic_root_equation_solution_l344_344992


namespace at_least_two_positive_real_solutions_l344_344201

noncomputable def polynomial : ℝ[X] :=
  X^11 + 8 * X^10 + 15 * X^9 - 1729 * X^8 + 1379 * X^7 - 172 * X^6

theorem at_least_two_positive_real_solutions :
  ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b ∧ polynomial.eval a = 0 ∧ polynomial.eval b = 0 :=
begin
  sorry
end

end at_least_two_positive_real_solutions_l344_344201


namespace infinitely_many_good_numbers_seven_does_not_divide_good_number_l344_344808

-- Define what it means for a number to be good
def is_good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = n ∧ (a * b) ∣ (n^2 + n + 1)

-- Part (a): Show that there are infinitely many good numbers
theorem infinitely_many_good_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_good_number (f n) :=
sorry

-- Part (b): Show that if n is a good number, then 7 does not divide n
theorem seven_does_not_divide_good_number (n : ℕ) (h : is_good_number n) : ¬ (7 ∣ n) :=
sorry

end infinitely_many_good_numbers_seven_does_not_divide_good_number_l344_344808


namespace calculate_constants_l344_344347

variables {C D Q : Type}
variables [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ Q]

def is_on_segment (C D Q : Type) [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ Q] (r s : ℝ) : Prop :=
  Q = r • C + s • D ∧ r + s = 1 ∧ r / s = 3 / 5

theorem calculate_constants (C D Q : Type) [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ Q]
  (h : is_on_segment C D Q (5/8) (3/8)) :
  ∃ (s v : ℝ), (s = 5/8) ∧ (v = 3/8) := by
  sorry

end calculate_constants_l344_344347


namespace largest_determinant_is_sqrt_521_l344_344722

def v : ℝ × ℝ × ℝ := (3, 2, -2)
def w : ℝ × ℝ × ℝ := (-1, 4, 5)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.3 - a.3 * b.2.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2.2 - a.2.2 * b.1)
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2.2 * a.2.2 + a.3 * a.3)

def largest_possible_determinant : ℝ :=
  magnitude (cross_product v w)

theorem largest_determinant_is_sqrt_521 :
  largest_possible_determinant = real.sqrt 521 :=
by 
  sorry

end largest_determinant_is_sqrt_521_l344_344722


namespace choose_officials_l344_344312

theorem choose_officials (n : ℕ) (h : n = 8) : 
  ∃ (ways : ℕ), ways = 336 :=
by
  use 336
  sorry

end choose_officials_l344_344312


namespace problem_statement_l344_344717

def a : ℕ → ℕ
| 1       := 1
| (n + 1) := a n * Nat.prime n

def tau (x : ℕ) : ℕ :=
  Divisors.finset x).card

theorem problem_statement :
  (∑ n in Finset.range 2020, ∑ d in (Divisors a (n+1)), tau d) % 91 = 40 :=
by sorry

end problem_statement_l344_344717


namespace technician_roundtrip_completion_l344_344468

theorem technician_roundtrip_completion (D : ℝ) (hD : D > 0) :
  let round_trip := 2 * D in
  let distance_completed := D + 0.4 * D in
  (distance_completed / round_trip) * 100 = 70 := 
by
  sorry

end technician_roundtrip_completion_l344_344468


namespace range_a_for_increasing_f_l344_344012

theorem range_a_for_increasing_f :
  (∀ (x : ℝ), 1 ≤ x → (2 * x - 2 * a) ≥ 0) → a ≤ 1 := by
  intro h
  sorry

end range_a_for_increasing_f_l344_344012


namespace nested_sqrt_solution_l344_344185

theorem nested_sqrt_solution : ∃ x : ℝ, x = sqrt (15 + x) ∧ x = (1 + sqrt 61) / 2 :=
by
  sorry

end nested_sqrt_solution_l344_344185


namespace student_B_speed_l344_344516

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l344_344516


namespace problem_2535_l344_344730

theorem problem_2535 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + b + (a^3 / b^2) + (b^3 / a^2) = 2535 := sorry

end problem_2535_l344_344730


namespace fraction_to_decimal_l344_344976

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344976


namespace total_machine_operation_time_l344_344341

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end total_machine_operation_time_l344_344341


namespace sin_270_eq_neg_one_l344_344159

theorem sin_270_eq_neg_one : 
  let Q := (0, -1) in
  ∃ (θ : ℝ), θ = 270 * Real.pi / 180 ∧ ∃ (Q : ℝ × ℝ), 
    Q = ⟨Real.cos θ, Real.sin θ⟩ ∧ Real.sin θ = -1 :=
by 
  sorry

end sin_270_eq_neg_one_l344_344159


namespace mary_cleaned_homes_l344_344377

theorem mary_cleaned_homes : 
  ∀ (earnings_per_home take_home_pay : ℝ) (tax_rate expense : ℝ), 
  earnings_per_home = 46 ∧ 
  take_home_pay = 276 ∧ 
  tax_rate = 0.10 ∧ 
  expense = 15 → 
  ∃ (homes_cleaned : ℕ), homes_cleaned = 6 :=
by
  intros earnings_per_home take_home_pay tax_rate expense h,
  obtain ⟨hh1, hh2, hh3, hh4⟩ := h,
  sorry  -- proof to be completed

end mary_cleaned_homes_l344_344377


namespace perfect_square_factors_count_l344_344171

theorem perfect_square_factors_count :
  let n := 8640
  let prime_factors := (2, 6) :: (3, 3) :: (5, 1) :: [] 
  ∃ (a b c : ℕ), (0 ≤ a ∧ a ≤ 6) ∧ (0 ≤ b ∧ b ≤ 3) ∧ (0 ≤ c ∧ c ≤ 1) ∧ 
    (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) ∧ 
    (∑(x : ℕ × ℕ) in prime_factors, (x.1 ^ (if x.1 = 2 then a else if x.1 = 3 then b else c)) = 8) :=
sorry

end perfect_square_factors_count_l344_344171


namespace find_circle_equation_l344_344226

noncomputable def dist_point_line (a b c x y : ℝ) : ℝ :=
(abs (a * x + b * y + c)) / (Math.sqrt (a * a + b * b))

noncomputable def circle_equation (h0 : ℝ) (k0 : ℝ) (r : ℝ) : (ℝ × ℝ) → ℝ
| (x, y) := (x - h0) ^ 2 + (y - k0) ^ 2 - r ^ 2

theorem find_circle_equation :
  ∃ h k r : ℝ,
    0 < h ∧ k = 0 ∧
    (circle_equation h k r (0, Math.sqrt 5)) = 0 ∧
    dist_point_line 2 (-1) 0 h k = 4 * Math.sqrt 5 / 5 ∧
    circle_equation 2 0 3 = λ (x, y), (x - 2)^2 + y^2 - 9 := sorry

end find_circle_equation_l344_344226


namespace rowing_speed_downstream_l344_344491

/--
A man can row upstream at 25 kmph and downstream at a certain speed. 
The speed of the man in still water is 30 kmph. 
Prove that the speed of the man rowing downstream is 35 kmph.
-/
theorem rowing_speed_downstream (V_u V_sw V_s V_d : ℝ)
  (h1 : V_u = 25) 
  (h2 : V_sw = 30) 
  (h3 : V_u = V_sw - V_s) 
  (h4 : V_d = V_sw + V_s) :
  V_d = 35 :=
by
  sorry

end rowing_speed_downstream_l344_344491


namespace student_B_speed_l344_344507

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l344_344507


namespace cost_of_first_15_kgs_l344_344535

def cost_33_kg := 333
def cost_36_kg := 366
def kilo_33 := 33
def kilo_36 := 36
def first_limit := 30
def extra_3kg := 3  -- 33 - 30
def extra_6kg := 6  -- 36 - 30

theorem cost_of_first_15_kgs (l q : ℕ) 
  (h1 : first_limit * l + extra_3kg * q = cost_33_kg)
  (h2 : first_limit * l + extra_6kg * q = cost_36_kg) :
  15 * l = 150 :=
by
  sorry

end cost_of_first_15_kgs_l344_344535


namespace range_of_g_l344_344558

noncomputable def g (x : ℝ) := (sin x)^3 + 4 * (sin x)^2 - 3 * sin x + 3 * (cos x)^2 - 9

theorem range_of_g :
  (∀ x : ℝ, sin x ≠ 1 → (g x) / (sin x - 1) ∈ set.Ico 5 9) :=
sorry

end range_of_g_l344_344558


namespace complex_transformations_result_l344_344451

theorem complex_transformations_result : 
  let z : ℂ := -4 - 6 * complex.i
  let rotation : ℂ := (√3 / 2) + (1 / 2) * complex.i
  let dilation : ℂ := 2
  in z * (rotation * dilation) = (-4 * √3 + 6) - (6 * √3 + 4) * complex.i :=
by {
  let z : ℂ := -4 - 6 * complex.i,
  let rotation : ℂ := (√3 / 2) + (1 / 2) * complex.i,
  let dilation : ℂ := 2,
  show z * (rotation * dilation) = (-4 * √3 + 6) - (6 * √3 + 4) * complex.i,
  sorry
}

end complex_transformations_result_l344_344451


namespace storage_methods_l344_344410

-- Definitions for the vertices and edges of the pyramid
structure Pyramid :=
  (P A B C D : Type)
  
-- Edges of the pyramid represented by pairs of vertices
def edges (P A B C D : Type) := [(P, A), (P, B), (P, C), (P, D), (A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]

-- Safe storage condition: No edges sharing a common vertex in the same warehouse
def safe (edge1 edge2 : (Type × Type)) : Prop :=
  edge1.1 ≠ edge2.1 ∧ edge1.1 ≠ edge2.2 ∧ edge1.2 ≠ edge2.1 ∧ edge1.2 ≠ edge2.2

-- The number of different methods to store the chemical products safely
def number_of_safe_storage_methods : Nat :=
  -- We should replace this part by actual calculation or combinatorial methods relevant to the problem
  48

theorem storage_methods (P A B C D : Type) : number_of_safe_storage_methods = 48 :=
  sorry

end storage_methods_l344_344410


namespace fraction_to_decimal_l344_344971

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344971


namespace f_expression_when_x_gt_1_l344_344013

variable (f : ℝ → ℝ)

-- conditions
def f_even : Prop := ∀ x, f (x + 1) = f (-x + 1)
def f_defn_when_x_lt_1 : Prop := ∀ x, x < 1 → f x = x ^ 2 + 1

-- theorem to prove
theorem f_expression_when_x_gt_1 (h_even : f_even f) (h_defn : f_defn_when_x_lt_1 f) : 
  ∀ x, x > 1 → f x = x ^ 2 - 4 * x + 5 := 
by
  sorry

end f_expression_when_x_gt_1_l344_344013


namespace round_trip_percentage_l344_344080

-- Definitions based on the conditions
variable (P : ℝ) -- Total number of passengers
variable (R : ℝ) -- Number of round-trip ticket holders

-- First condition: 20% of passengers held round-trip tickets and took their cars aboard
def condition1 := 0.20 * P = 0.60 * R

-- Second condition: 40% of passengers with round-trip tickets did not take their cars aboard (implies 60% did)
theorem round_trip_percentage (h1 : condition1 P R) : (R / P) * 100 = 33.33 := by
  sorry

end round_trip_percentage_l344_344080


namespace interval_length_l344_344017

theorem interval_length (c : ℝ) (h : ∀ x : ℝ, 3 ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ c → 
                             (3 * (x) + 4 ≤ c ∧ 3 ≤ 3 * x + 4)) :
  (∃ c : ℝ, ((c - 4) / 3) - ((-1) / 3) = 15) → (c - 3 = 45) :=
sorry

end interval_length_l344_344017


namespace bicycle_speed_B_l344_344511

theorem bicycle_speed_B 
  (distance : ℝ := 12)
  (ratio : ℝ := 1.2)
  (time_diff : ℝ := 1 / 6) : 
  ∃ (B_speed : ℝ), B_speed = 12 :=
by
  let A_speed := ratio * B_speed
  have eqn : distance / B_speed - time_diff = distance / A_speed := sorry
  exact ⟨12, sorry⟩

end bicycle_speed_B_l344_344511


namespace distance_between_lines_l344_344799

-- Define the geometric entities and conditions
variables {a : ℝ} 
variables (A B C D K E F : TopologicalSpace.Point ℝ) 

-- Midpoint conditions
def is_midpoint (K : TopologicalSpace.Point ℝ) (A B : TopologicalSpace.Point ℝ) : Prop :=
  dist K A = dist K B ∧ 2 * dist K A = dist A B

-- Ratio conditions
def ratio_segment (E : TopologicalSpace.Point ℝ) (C D : TopologicalSpace.Point ℝ) (r s : ℝ) : Prop :=
  dist E C * s = dist E D * r

-- Center conditions
def is_centroid (F : TopologicalSpace.Point ℝ) (A B C : TopologicalSpace.Point ℝ) : Prop :=
  dist F A = dist F B ∧ dist F B = dist F C ∧ dist F C = dist F A

-- Main proof statement
theorem distance_between_lines (h_mid : is_midpoint K A B) 
                               (h_ratio : ratio_segment E C D 1 2) 
                               (h_centroid : is_centroid F A B C) 
                               : dist_between_lines BC KE = a * sqrt (6 / 81) := 
begin
  sorry
end

end distance_between_lines_l344_344799


namespace men_entered_l344_344704

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l344_344704


namespace inequality_always_true_l344_344419

theorem inequality_always_true (a : ℝ) : (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ 3 ≤ a :=
by
  sorry

end inequality_always_true_l344_344419


namespace find_AM2_LB2_eq_four_l344_344325

variable (AB AD : ℝ)
variable (P : ℝ × ℝ)
variable (u : ℝ)

noncomputable theory

-- Define the rectangle ABCD with given conditions
def is_rectangle (AB AD : ℝ) (AD_sqrt2 : Prop) :=
  AB = 2 ∧ AD_sqrt2

-- Define the ellipse equation 
def is_ellipse (AB : ℝ) (u : ℝ) (P : ℝ × ℝ) :=
  P.1 ^ 2 + (2 * P.2 ^ 2) / (u ^ 2) = 1

-- Point P is not an endpoint of the major axis
def valid_point (P : ℝ × ℝ) :=
  P ≠ (1, 0) ∧ P ≠ (-1, 0)

-- Intersection points of PC and PD with AB
def intersection_points (P : ℝ × ℝ) (u : ℝ) : ℝ × ℝ :=
  let x_M := ((P.1 - 1) * u / ((P.1 - 1) - P.2 * u / √2)) + 1
  let x_L := ((P.1 + 1) * u / ((P.1 + 1) - P.2 * u / √2)) - 1
  (x_M, x_L)

-- Define the target equation to prove
theorem find_AM2_LB2_eq_four 
  (h1 : is_rectangle AB AD (AD < √2)) 
  (h2 : is_ellipse AB u P) 
  (h3 : valid_point P) : 
  let (AM, LB) := intersection_points P u in
  AM ^ 2 + LB ^ 2 = 4 :=
sorry

end find_AM2_LB2_eq_four_l344_344325


namespace principal_amount_borrowed_l344_344467

theorem principal_amount_borrowed (SI R T : ℝ) (h_SI : SI = 2000) (h_R : R = 4) (h_T : T = 10) : 
    ∃ P, SI = (P * R * T) / 100 ∧ P = 5000 :=
by
    sorry

end principal_amount_borrowed_l344_344467


namespace fill_trough_time_l344_344532

theorem fill_trough_time 
  (old_pump_rate : ℝ := 1 / 600) 
  (new_pump_rate : ℝ := 1 / 200) : 
  1 / (old_pump_rate + new_pump_rate) = 150 := 
by 
  sorry

end fill_trough_time_l344_344532


namespace max_sum_distances_l344_344562

variables {A B C : Type} [decidable_eq A] [decidable_eq B] [decidable_eq C]
variables [linear_ordered_field A] {a b c : A}

/-- Representation of the midpoint of a segment -/
noncomputable def midpoint (a b : A) : A := (a + b) / 2

/-- Check if a triangle is acute-angled. -/
def is_acute (A B C: A) : Prop :=
(A < 90) ∧ (B < 90) ∧ (C < 90)

/-- The required line that maximizes the sum of distances from B and C is perpendicular to the median AM -/
theorem max_sum_distances (A B C : A) (M : A) [a < 90] [b < 90] [c < 90]:
  ∀ (L: A), L ∈ A → ¬L ∈ B → 
  (M = midpoint B C) →
  is_acute A B C →
  maximizes_sum (dist B L + dist C L) (L ⊥ A M) :=
by
  sorry

end max_sum_distances_l344_344562


namespace fraction_to_decimal_l344_344973

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344973


namespace problem_I_problem_II_l344_344610

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) :=
  { (x, y) | x^2 = 2*p*y ∧ p > 0 }

noncomputable def line (k p : ℝ) : set (ℝ × ℝ) :=
  { (x, y) | y = k*x + p/2 }

def intersects (A B : ℝ × ℝ) : Prop :=
  -- Define that A and B are points of intersection of parabola and line
  ∃ (p k : ℝ), (A ∈ parabola p) ∧ (B ∈ parabola p) ∧ (A ∈ line k p) ∧ (B ∈ line k p)

def segment_length (A B : ℝ × ℝ) : ℝ :=
  real.dist A B

theorem problem_I (p : ℝ) (A B : ℝ × ℝ) (h : 1 = 1 ∧ segment_length A B = 8 ∧ intersects A B) :
  parabola p = { (x, y) | x^2 = 4*y } :=
sorry

theorem problem_II (p : ℝ) (P F : ℝ × ℝ) (k : ℝ) (l : set (ℝ × ℝ))
  (h₁ : P ∈ l)
  (h₂ : F ∈ parabola p)
  (h₃ : slopes_sum k P F = -3/2) :
  k = -2 ∨ k = 1/2 :=
sorry

-- Definition of sum of slopes
def slopes_sum (k : ℝ) (P F : (ℝ × ℝ)) : ℝ :=
  let kPF := (F.2 - P.2) / (F.1 - P.1) in -- slope of line PF
  kPF + k

end problem_I_problem_II_l344_344610


namespace highland_high_students_highland_high_num_both_clubs_l344_344539

theorem highland_high_students (total_students drama_club science_club either_both both_clubs : ℕ)
  (h1 : total_students = 320)
  (h2 : drama_club = 90)
  (h3 : science_club = 140)
  (h4 : either_both = 200) : 
  both_clubs = drama_club + science_club - either_both :=
by
  sorry

noncomputable def num_both_clubs : ℕ :=
if h : 320 = 320 ∧ 90 = 90 ∧ 140 = 140 ∧ 200 = 200
then 90 + 140 - 200
else 0

theorem highland_high_num_both_clubs : num_both_clubs = 30 :=
by
  sorry

end highland_high_students_highland_high_num_both_clubs_l344_344539


namespace quadratic_no_real_roots_l344_344251

theorem quadratic_no_real_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + b * x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 :=
sorry

end quadratic_no_real_roots_l344_344251


namespace eval_polynomial_at_minus_two_l344_344453

theorem eval_polynomial_at_minus_two :
  let f : ℤ → ℤ := λ x, x^5 + 4 * x^4 + x^2 + 20 * x + 16 in
  f (-2) = -4 := by
  sorry

end eval_polynomial_at_minus_two_l344_344453


namespace men_entered_l344_344706

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l344_344706


namespace exist_infinite_a_l344_344864

theorem exist_infinite_a (n : ℕ) (a : ℕ) (h₁ : ∃ k : ℕ, k > 0 ∧ (n^6 + 3 * a = (n^2 + 3 * k)^3)) : 
  ∃ f : ℕ → ℕ, ∀ m : ℕ, (∃ k : ℕ, k > 0 ∧ f m = 9 * k^3 + 3 * n^2 * k * (n^2 + 3 * k)) :=
by 
  sorry

end exist_infinite_a_l344_344864


namespace total_profit_is_18900_l344_344523

-- Defining the conditions
variable (x : ℕ)  -- A's initial investment
variable (A_share : ℕ := 6300)  -- A's share in rupees

-- Total profit calculation
def total_annual_gain : ℕ :=
  (x * 12) + (2 * x * 6) + (3 * x * 4)

-- The main statement
theorem total_profit_is_18900 (x : ℕ) (A_share : ℕ := 6300) :
  3 * A_share = total_annual_gain x :=
by sorry

end total_profit_is_18900_l344_344523


namespace difference_of_areas_l344_344576

-- Defining the side length of the square
def square_side_length : ℝ := 8

-- Defining the side lengths of the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 5

-- Defining the area functions
def area_of_square (side_length : ℝ) : ℝ := side_length * side_length
def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ := length * width

-- Stating the theorem
theorem difference_of_areas :
  area_of_square square_side_length - area_of_rectangle rectangle_length rectangle_width = 14 :=
by
  sorry

end difference_of_areas_l344_344576


namespace boat_speed_is_correct_l344_344481

noncomputable def downstream_speed (v : ℝ) :=
v = 32.81792717086835

theorem boat_speed_is_correct
  (v : ℝ)
  (upstream_distance : ℝ := 90)
  (downstream_distance : ℝ := 90)
  (upstream_speed : ℝ := v - 3)
  (downstream_time : ℝ := 2.5191640969412834)
  (total_upstream_time : ℝ := 3.0191640969412834) :
  (upstream_distance / upstream_speed = total_upstream_time) →
  (downstream_distance / v = downstream_time) →
  downstream_speed v :=
by
  intro h_upstream h_downstream
  have h1 : v - 3 = upstream_distance / total_upstream_time := by sorry
  have h2 : v = 32.81792717086835 := by sorry
  exact h2.symm

end boat_speed_is_correct_l344_344481


namespace digits_in_product_l344_344943

def power (base : Nat) (exp : Nat) : Nat :=
  base ^ exp

def numDigits (n : Nat) : Nat :=
  n.toString.length

theorem digits_in_product : numDigits (2 ^ 15 * 5 ^ 10 * 3 ^ 2) = 13 := by
  sorry

end digits_in_product_l344_344943


namespace complex_point_line_l344_344227

def complex_point_on_line (z : ℂ) : Prop := 
  let (x, y) := (z.re, z.im)
  in y = -1/2

theorem complex_point_line (z : ℂ) (h: z * (1 + Complex.i)^2 = 1 - Complex.i) :
  complex_point_on_line z := 
by sorry

end complex_point_line_l344_344227


namespace first_12_payments_amount_l344_344093

theorem first_12_payments_amount:
  ∃ (x : ℝ), 
  let total_installments := 52 in
  let first_12_payments := 12 in
  let remaining_payments := total_installments - first_12_payments in
  let increase_amount := 65 in
  let average_payment := 460 in
  12 * x + 40 * (x + increase_amount) = total_installments * average_payment ∧ x = 410 :=
begin
  sorry
end

end first_12_payments_amount_l344_344093


namespace quadrilateral_area_is_63_l344_344590

noncomputable section

namespace QuadrilateralArea

open Real

-- Define the points
def P : ℝ × ℝ := (7, 6)
def Q : ℝ × ℝ := (-5, 1)
def R : ℝ × ℝ := (-2, -3)
def S : ℝ × ℝ := (10, 2)

-- Define the area theorem
theorem quadrilateral_area_is_63 :
  let area_quadrilateral : ℝ :=
    let area_triangle (a b c : ℝ × ℝ) : ℝ :=
      (1 / 2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))
    in
    let area_total : ℝ := 135 - (30 + 30 + 6 + 6)
    in area_total
  in area_quadrilateral = 63 :=
  by
    -- Introduce the points
    let P := (7, 6)
    let Q := (-5, 1)
    let R := (-2, -3)
    let S := (10, 2)
    -- Here you calculate the area_quadrilateral
    sorry

end QuadrilateralArea

end quadrilateral_area_is_63_l344_344590


namespace ice_cream_combinations_l344_344289

theorem ice_cream_combinations :
  ∃ n : ℕ, (n = Nat.choose 8 3 ∧ n = 56) := 
begin
  use Nat.choose 8 3,
  split,
  {refl},
  {norm_num}
end


end ice_cream_combinations_l344_344289


namespace streetlights_turn_off_problem_l344_344382
-- Step 1: Import the required library

-- Step 2: State the problem with definitions and the theorem

def is_valid_turn_off (n : ℕ) (i j : ℕ) : Prop :=
  2 ≤ i ∧ i < j ∧ j ≤ n - 1 ∧ j = i + 1 → False

theorem streetlights_turn_off_problem :
  ∃ (n i j : ℕ), 
  n = 10 ∧ 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ is_valid_turn_off n i j ∧ 
  (nat.choose 7 2 = 21) :=
begin
  sorry
end

end streetlights_turn_off_problem_l344_344382


namespace fx_periodic_even_with_period_pi_l344_344801

def f (x : ℝ) : ℝ := sin (π / 4 + x) * sin (π / 4 - x)

theorem fx_periodic_even_with_period_pi : 
  ∀ x : ℝ, f x = sin (π / 4 + x) * sin (π / 4 - x) → 
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ x : ℝ, f (-x) = f x) :=
by sorry

end fx_periodic_even_with_period_pi_l344_344801


namespace intersect_and_eq_dists_l344_344719

def Parallelogram (A B C D : Type*) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ (A ≠ C) ∧ (B ≠ D)

variables {A B C D M N P Q : Type*}

noncomputable def midpoint (A B : Type*) : Type* := sorry

noncomputable def intersection (L1 L2 : Type*) : Type* := sorry

theorem intersect_and_eq_dists (A B C D : Type*) (h : Parallelogram A B C D)
  (M : Type*) (hM : M = midpoint B C) (N : Type*) (hN : N = midpoint C D)
  (Q : Type*) (hQ : Q = intersection (line_through A N) (line_through B D))
  (P : Type*) (hP : P = intersection (line_through A M) (line_through B D)) :
  dist B P = dist P Q ∧ dist P Q = dist Q D :=
sorry

end intersect_and_eq_dists_l344_344719


namespace max_value_at_a_l344_344255

def f'' (x : ℝ) : ℝ := -x / 2

noncomputable def f (x : ℝ) : ℝ := - (2 * f'' 1) / 3 * sqrt x - x^2

theorem max_value_at_a : ∃ a : ℝ, a = (Real.sqrt (Real.sqrt 4) / 4) ∧ ∀ x : ℝ, f x ≤ f a :=
begin
  use Real.sqrt (Real.sqrt 4) / 4,
  sorry
end

end max_value_at_a_l344_344255


namespace h_properties_l344_344258

def f (x : ℝ) := Real.log2 (1 - x)
def g (x : ℝ) := Real.log2 (1 + x)
def h (x : ℝ) := f x - g x

-- Domain definition
def domain (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Theorem statement
theorem h_properties :
  (∀ x : ℝ, domain x) ∧
  (∀ x : ℝ, domain x → h (-x) = -h x) ∧
  (∀ x1 x2 : ℝ, domain x1 → domain x2 → x1 < x2 → h x1 < h x2) :=
by
  sorry

end h_properties_l344_344258


namespace complement_of_M_in_U_l344_344732

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def M : Set ℕ := {0, 1}

theorem complement_of_M_in_U : (U \ M) = {2, 3, 4, 5} :=
by
  -- The proof is omitted here.
  sorry

end complement_of_M_in_U_l344_344732


namespace cover_square_with_rectangles_l344_344058

theorem cover_square_with_rectangles :
  ∃ n : ℕ, n = 24 ∧
  ∀ (rect_area : ℕ) (square_area : ℕ), rect_area = 2 * 3 → square_area = 12 * 12 → square_area / rect_area = n :=
by
  use 24
  sorry

end cover_square_with_rectangles_l344_344058


namespace solution_set_inequality_l344_344254

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 2 else -1

theorem solution_set_inequality (x : ℝ) :
  { x | x + 2 * x * f (x + 1) > 5 } = { x | x < -5 } ∪ { x | x > 1 } :=
by
  sorry

end solution_set_inequality_l344_344254


namespace variance_of_xi_l344_344814

noncomputable def xi : Type := {0, 1, 2}

def P (v : xi → ℝ) : Prop := 
v (0 : xi) = 1/5

def E (xi : xi → ℝ) : ℝ :=
∑ v in {0, 1, 2}, v * P xi v

def var (xi : xi → ℝ) : ℝ :=
∑ v in {0, 1, 2}, (v^2) * P xi v - (E xi)^2

theorem variance_of_xi (h₀ : P(0) = 1/5) (h₁ : E xi = 1) : var xi = 2/5 :=
by
  sorry

end variance_of_xi_l344_344814


namespace roots_interlaced_l344_344241

variable {α : Type*} [LinearOrderedField α]
variables {f g : α → α}

theorem roots_interlaced
    (x1 x2 x3 x4 : α)
    (h1 : x1 < x2) (h2 : x3 < x4)
    (hfx1 : f x1 = 0) (hfx2 : f x2 = 0)
    (hfx_distinct : x1 ≠ x2)
    (hgx3 : g x3 = 0) (hgx4 : g x4 = 0)
    (hgx_distinct : x3 ≠ x4)
    (hgx1_ne_0 : g x1 ≠ 0) (hgx2_ne_0 : g x2 ≠ 0)
    (hgx1_gx2_lt_0 : g x1 * g x2 < 0) :
    (x1 < x3 ∧ x3 < x2 ∧ x2 < x4) ∨ (x3 < x1 ∧ x1 < x4 ∧ x4 < x2) :=
sorry

end roots_interlaced_l344_344241


namespace area_of_triangle_l344_344574

noncomputable def triangle_area_tangent_curve : ℚ := by 
  let y := λ x : ℝ, x^3
  let dy := λ x : ℝ, 3 * x^2
  let point := (1 : ℝ, 1 : ℝ)
  let slope := dy point.1
  let tangent_line := λ x : ℝ, slope * (x - point.1) + point.2
  let x_line := 2
  let intersection_y := tangent_line x_line
  let intersection_x := (2 : ℚ) / 3
  let base := (x_line : ℚ) - intersection_x
  let height := (intersection_y : ℚ)
  exact (1 / 2) * base * height

theorem area_of_triangle : triangle_area_tangent_curve = 8 / 3 := 
  sorry

end area_of_triangle_l344_344574


namespace trapezoid_area_l344_344839

noncomputable def area_trapezoid (B1 B2 h : ℝ) : ℝ := (1 / 2 * (B1 + B2) * h)

theorem trapezoid_area
    (h1 : ∀ x : ℝ, 3 * x = 10 → x = 10 / 3)
    (h2 : ∀ x : ℝ, 3 * x = 5 → x = 5 / 3)
    (h3 : B1 = 10 / 3)
    (h4 : B2 = 5 / 3)
    (h5 : h = 5)
    : area_trapezoid B1 B2 h = 12.5 := by
  sorry

end trapezoid_area_l344_344839


namespace percentage_reduction_correct_l344_344104

def original_sheets : ℕ := 350
def lines_per_original_sheet : ℕ := 85
def characters_per_original_line : ℕ := 100
def lines_per_retyped_sheet : ℕ := 120
def characters_per_retyped_line : ℕ := 110

noncomputable def percentage_reduction_in_sheets : ℚ :=
  let total_chars_original := (original_sheets * lines_per_original_sheet * characters_per_original_line : ℚ)
  let total_chars_per_retyped_sheet := (lines_per_retyped_sheet * characters_per_retyped_line : ℚ)
  let retyped_sheets := (total_chars_original / total_chars_per_retyped_sheet).ceil
  let reduction := ((original_sheets - retyped_sheets) : ℚ) / original_sheets * 100
  reduction

theorem percentage_reduction_correct : percentage_reduction_in_sheets ≈ 35.43 :=
by
  sorry

end percentage_reduction_correct_l344_344104


namespace Mike_siblings_l344_344415

-- Define the types for EyeColor, HairColor and Sport
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define the Child structure
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define all the children based on the given conditions
def Lily : Child := { name := "Lily", eyeColor := EyeColor.Green, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Mike : Child := { name := "Mike", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Oliver : Child := { name := "Oliver", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Emma : Child := { name := "Emma", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Jacob : Child := { name := "Jacob", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }
def Sophia : Child := { name := "Sophia", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }

-- Siblings relation
def areSiblings (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.favoriteSport = c2.favoriteSport) ∧
  (c1.eyeColor = c3.eyeColor ∨ c1.hairColor = c3.hairColor ∨ c1.favoriteSport = c3.favoriteSport) ∧
  (c2.eyeColor = c3.eyeColor ∨ c2.hairColor = c3.hairColor ∨ c2.favoriteSport = c3.favoriteSport)

-- The proof statement
theorem Mike_siblings : areSiblings Mike Emma Jacob := by
  -- Proof must be provided here
  sorry

end Mike_siblings_l344_344415


namespace knight_moves_least_moves_l344_344458

def f (n : ℕ) : ℕ :=
  2 * (n + 1) / 3

theorem knight_moves_least_moves (n : ℕ) (h : n ≥ 4) :
  f(n) = 2 * (n + 1) / 3 := sorry

end knight_moves_least_moves_l344_344458


namespace exist_divisible_number_l344_344760

theorem exist_divisible_number (d : ℕ) (hd : d > 0) :
  ∃ n : ℕ, (n % d = 0) ∧ ∃ k : ℕ, (k > 0) ∧ (k < 10) ∧ 
  ((∃ m : ℕ, m = n - k*(10^k / 10^k) ∧ m % d = 0) ∨ ∃ m : ℕ, m = n - k * (10^(k - 1)) ∧ m % d = 0) :=
sorry

end exist_divisible_number_l344_344760


namespace area_QRS_l344_344369

noncomputable def triangle_area (p q r : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 2) * (Math.sqrt ((q.1 - p.1) * (r.2 - p.2) - (q.2 - p.2) * (r.1 - p.1))^2)

noncomputable def tetrahedron (P Q R S : ℝ × ℝ × ℝ) :=
  (P.1 = 0 ∧ P.2 = 0 ∧ P.3 = 0) ∧
  (Q.2 = 0 ∧ Q.3 = 0) ∧
  (R.1 = 0 ∧ R.3 = 0) ∧
  (S.1 = 0 ∧ S.2 = 0)

theorem area_QRS 
  (P Q R S : ℝ × ℝ × ℝ)
  (h_tetrahedron : tetrahedron P Q R S)
  (a : ℝ)
  (h_a : triangle_area P Q R = a)
  (b : ℝ)
  (h_b : triangle_area P R S = b)
  (c : ℝ)
  (h_c : triangle_area P Q S = c) :
  triangle_area Q R S = Real.sqrt (a^2 + b^2 + c^2) :=
sorry

end area_QRS_l344_344369


namespace supplies_total_cost_l344_344902

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end supplies_total_cost_l344_344902


namespace triangle_solution_l344_344701

theorem triangle_solution (a b c : ℕ) (k m n : ℕ) (h1: k > 0) (h2: m > 0) (h3: n > m)
  (h4: Nat.coprime m n) :
  (a = k * m * (2 * n ^ 2 - m ^ 2)) ∧ (b = k * n ^ 3) ∧ (c = k * n * (n ^ 2 - m ^ 2)) :=
sorry

end triangle_solution_l344_344701


namespace parabola_shift_l344_344810

theorem parabola_shift (x : ℝ) : 
  let original_y := -2 * x^2 + 1 in
  let shifted_y := -2 * (x - 1)^2 + 3 in
  (∀ x : ℝ, (y = -2 * x^2 + 1) → (y = -2 * (x - 1)^2 + 3)) :=
sorry

end parabola_shift_l344_344810


namespace correct_propositions_l344_344428

theorem correct_propositions :
  (1 : ℕ ∈ {1, 4}) ∧ ¬(2 : ℕ ∈ {1, 4}) ∧ ¬(3 : ℕ ∈ {1, 4}) ∧ (4 : ℕ ∈ {1, 4}) := 
by {
  split,
  -- Proposition 1 is correct (y = sin(2/3 x) is odd)
  {
    unfold membership,
    sorry,  -- proof for proposition 1 correctness
  },
  split,
  -- Proposition 2 is incorrect
  {
    unfold membership,
    sorry,  -- proof for proposition 2 incorrectness
  },
  split,
  -- Proposition 3 is incorrect
  {
    unfold membership,
    sorry,  -- proof for proposition 3 incorrectness
  },
  -- Proposition 4 is correct
  {
    unfold membership,
    sorry,  -- proof for proposition 4 correctness
  }
}

end correct_propositions_l344_344428


namespace area_of_triangle_AMB_l344_344108

theorem area_of_triangle_AMB :
  let vertex_x := 1
      vertex_y := 2
      A_x := -1
      A_y := 0
      parabola_eq := λ x : ℝ, - (1 / 2) * (x - vertex_x) ^ 2 + vertex_y
      B_x := 3
      B_y := 0
      M_x := 0
      M_y := parabola_eq M_x
      area_of_triangle := (1 / 2) * abs (A_x - B_x) * abs M_y
  in area_of_triangle = 3 :=
by
  let vertex_x := 1
  let vertex_y := 2
  let A_x := -1
  let A_y := 0
  let parabola_eq := λ x : ℝ, - (1 / 2) * (x - vertex_x) ^ 2 + vertex_y
  let B_x := 3
  let B_y := 0
  let M_x := 0
  let M_y := parabola_eq M_x
  let area_of_triangle := (1 / 2) * abs (A_x - B_x) * abs M_y
  have : area_of_triangle = 3 := sorry
  exact this

end area_of_triangle_AMB_l344_344108


namespace Jeremy_songs_l344_344335

theorem Jeremy_songs (songs_yesterday : ℕ) (songs_difference : ℕ) (songs_today : ℕ) (total_songs : ℕ) :
  songs_yesterday = 9 ∧ songs_difference = 5 ∧ songs_today = songs_yesterday + songs_difference ∧ 
  total_songs = songs_yesterday + songs_today → total_songs = 23 :=
by
  intros h
  sorry

end Jeremy_songs_l344_344335


namespace tan_identity_tan_product_identity_l344_344074

-- Part (a)
theorem tan_identity (k : ℝ) : 
  (1 + real.tan k) * (1 + real.tan (real.pi / 4 - k)) = 2 := 
sorry

-- Part (b)
theorem tan_product_identity (n : ℕ) :
  (∏ k in (finset.range 45).image (λ i, i + 1), 1 + real.tan (k * real.pi / 180)) = 2^23 :=
sorry

end tan_identity_tan_product_identity_l344_344074


namespace verify_magic_square_l344_344984

-- Define the grid as a 3x3 matrix
def magic_square := Matrix (Fin 3) (Fin 3) ℕ

-- Conditions for the magic square
def is_magic_square (m : magic_square) : Prop :=
  (∀ i : Fin 3, (m i 0) + (m i 1) + (m i 2) = 15) ∧
  (∀ j : Fin 3, (m 0 j) + (m 1 j) + (m 2 j) = 15) ∧
  ((m 0 0) + (m 1 1) + (m 2 2) = 15) ∧
  ((m 0 2) + (m 1 1) + (m 2 0) = 15)

-- Given specific filled numbers in the grid
def given_filled_values (m : magic_square) : Prop :=
  (m 0 1 = 5) ∧
  (m 1 0 = 2) ∧
  (m 2 2 = 8)

-- The complete grid based on the solution
def completed_magic_square : magic_square :=
  ![![4, 9, 2], ![3, 5, 7], ![8, 1, 6]]

-- The main theorem to prove
theorem verify_magic_square : 
  is_magic_square completed_magic_square ∧ 
  given_filled_values completed_magic_square := 
by 
  sorry

end verify_magic_square_l344_344984


namespace symmetric_y_axis_l344_344603

-- Definition of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of point symmetry with respect to the y-axis
def symmetric_about_y_axis (M : Point3D) : Point3D := 
  { x := -M.x, y := M.y, z := -M.z }

-- Theorem statement: proving the symmetry
theorem symmetric_y_axis (M : Point3D) : 
  symmetric_about_y_axis M = { x := -M.x, y := M.y, z := -M.z } := by
  sorry  -- Proof is left out as per instruction.

end symmetric_y_axis_l344_344603


namespace value_of_f_at_2_l344_344624

-- Define f as an odd function and its behavior for x < 0
def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else -f (-x)

theorem value_of_f_at_2 : f 2 = 12 :=
by 
  have h1 : f (-2) = 2 * (-2 : ℝ)^3 + (-2 : ℝ)^2 := if_pos (by linarith)
  have h2 : f (-2) = -12 := by calc 
    f (-2) = 2 * (-2)^3 + (-2)^2 : h1
    ...    = 2 * -8 + 4          : by simp
    ...    = -16 + 4            : by ring
    ...    = -12                : by ring
  show f 2 = 12 from 
    have : f 2 = - f (-2) := 
      if_neg (by linarith)
    calc 
      f 2 = - f (-2) : this
      ...   = - (-12) : by rw h2
      ...   = 12      : by ring

end value_of_f_at_2_l344_344624


namespace probability_of_blue_tile_l344_344482

def is_blue_tile (n : ℕ) : Prop :=
  n % 7 = 3

def total_tiles := 100
def blue_tiles := finset.filter is_blue_tile (finset.range (total_tiles + 1))

theorem probability_of_blue_tile :
  (blue_tiles.card : ℝ) / (total_tiles : ℝ) = 7 / 50 := sorry

end probability_of_blue_tile_l344_344482


namespace a4_value_l344_344031

noncomputable def a_seq : ℕ → ℝ
| 1 := 1
| 2 := 2
| (n+1) := 2 * n / a_seq n

theorem a4_value :
  let λ := 2
  in a_seq 4 = 3 :=
by
  sorry

end a4_value_l344_344031


namespace rearranging_2025_digits_l344_344274

theorem rearranging_2025_digits (h : List.perm ⟨[2, 0, 2, 5]⟩ [2, 0, 2, 5]) :
  ∃ l : List ℕ, (∀ x ∈ l, x ≠ 0) ∧ l.length = 9 ∧
  (∀ x ∈ l, (x = [2, 0, 2, 5].perms.length) ∨ (x = [2, 5, 2, 0].perms.length)) := sorry

end rearranging_2025_digits_l344_344274


namespace cake_percentage_eaten_l344_344212

/-- 
Forest and his friends have prepared a birthday cake for their friend Juelz having 240 cake pieces. 
After singing the birthday song, they ate a certain percentage of the cake's pieces, and later, 
Juelz divided the remaining pieces among his three sisters. Each sister received 32 pieces of cake. 
Prove that they ate 60% of the cake's pieces.
-/

theorem cake_percentage_eaten (initial_pieces : ℕ) (sisters : ℕ) (pieces_per_sister : ℕ) : 
  initial_pieces = 240 →
  sisters = 3 →
  pieces_per_sister = 32 →
  (initial_pieces - sisters * pieces_per_sister) * 100 / initial_pieces = 60 :=
by
  intros h0 h1 h2
  simp [h0, h1, h2]
  sorry

end cake_percentage_eaten_l344_344212


namespace sin_cos_difference_l344_344237

theorem sin_cos_difference (α : ℝ) (h_cond : sin α + cos α = 1/2) (h_dom : 0 < α ∧ α < π) : 
  sin α - cos α = sqrt 7 / 2 :=
sorry

end sin_cos_difference_l344_344237


namespace geometric_prod_l344_344821

theorem geometric_prod (n : ℕ) (b : ℕ → ℝ) (Sn Sn1 : ℝ) 
  (h1 : Sn = ∑ i in finset.range n, b i) 
  (h2 : Sn1 = ∑ i in finset.range n, (1 / b i))
  (h3 : Sn = (1 / 8) * Sn1) :
  b 0 * b (n - 1) = 1 / 8 :=
by
  sorry

end geometric_prod_l344_344821


namespace conductor_surname_is_Wang_l344_344384

-- Define the individuals and their attributes
structure Individual :=
  (surname : String)
  (role : Option String := none)   -- Optional role: driver, conductor, train police officer
  (city : Option String := none)   -- Optional city of residence
  (likes_basketball : Bool := false)

-- Define the passengers and crew members
noncomputable def PassengerZhang : Individual := { surname := "Zhang" }
noncomputable def PassengerWang : Individual := { surname := "Wang" }
noncomputable def PassengerLi : Individual := { surname := "Li", city := some "Beijing" }

noncomputable def Conductor : Individual := { role := some "conductor", city := some "Tianjin" }

-- Define the conditions
axiom cond1 : PassengerLi.city = some "Beijing"
axiom cond2 : Conductor.city = some "Tianjin"
axiom cond3 : PassengerWang.likes_basketball = false
axiom cond4 : ∃ p : Individual, p.surname = Conductor.surname ∧ p.city = some "Shanghai"
axiom cond5 : ∃ p : Individual, p.likes_basketball = true ∧ p.surname = PassengerZhang.surname

-- The theorem to prove
theorem conductor_surname_is_Wang : Conductor.surname = "Wang" :=
sorry

end conductor_surname_is_Wang_l344_344384


namespace max_intersections_perpendicular_lines_l344_344224

theorem max_intersections_perpendicular_lines :
  ∃ (points : Fin 5 → ℝ × ℝ),
    (∀ i j : Fin 5, i ≠ j → let (x1, y1) := points i 
                                 (x2, y2) := points j in
                              let slope := (y2 - y1) / (x2 - x1) in
                                slope ≠ 0 ∧ slope ≠ -(1 / slope)) →
    (max_intersection_points points = 310) :=
by
  -- specific details of this theorem proof will, in practice, require defining the function max_intersection_points
  sorry

-- In practice, detailed definitions for "max_intersecting_points" would need to be provided
-- and intensive geometric reasoning would be involved to prove the theorem above.

end max_intersections_perpendicular_lines_l344_344224


namespace shirts_bought_by_peter_l344_344143

-- Define the constants and assumptions
variables (P S x : ℕ)

-- State the conditions given in the problem
def condition1 : P = 6 :=
by sorry

def condition2 : 2 * S = 20 :=
by sorry

def condition3 : 2 * P + x * S = 62 :=
by sorry

-- State the theorem to be proven
theorem shirts_bought_by_peter : x = 5 :=
by sorry

end shirts_bought_by_peter_l344_344143


namespace inclination_angle_of_transformed_line_l344_344434

theorem inclination_angle_of_transformed_line
  (a b c : ℝ)
  (h_symmetry_axis : ∀ x : ℝ, a * sin (π / 4 + x) - b * cos (π / 4 + x) = a * sin (π / 4 - x) - b * cos (π / 4 - x)) :
  let K := a / b in 
  ∀ x, K = -1 →
  ((K = -1) → arctan K = 3 * π / 4) ∧ K = -1 :=
by
  sorry

end inclination_angle_of_transformed_line_l344_344434


namespace solve_system_of_equations_l344_344781

-- Define the system of equations
def system_of_equations (a b x y : ℝ) : Prop :=
  x^2 = a * x + b * y ∧ y^2 = b * x + a * y

-- Define the solutions
def solutions (a b x y : ℝ) : Prop :=
  (x = 0 ∧ y = 0) ∨
  (x = a + b ∧ y = a + b) ∨
  (x = (a - b - real.sqrt((a - b) * (a + 3 * b))) / 2 ∧
   y = (a - b + real.sqrt((a - b) * (a + 3 * b))) / 2) ∨
  (x = (a - b + real.sqrt((a - b) * (a + 3 * b))) / 2 ∧
   y = (a - b - real.sqrt((a - b) * (a + 3 * b))) / 2)

-- The final statement to be proven
theorem solve_system_of_equations (a b : ℝ) :
  ∀ x y : ℝ, system_of_equations a b x y ↔ solutions a b x y :=
by
  sorry

end solve_system_of_equations_l344_344781


namespace rectangles_in_grid_l344_344282

theorem rectangles_in_grid : 
  let horizontal_strip := 1 * 5 in
  let vertical_strip := 1 * 4 in
  ∃ (rectangles : ℕ), rectangles = 24 := 
  by
  sorry

end rectangles_in_grid_l344_344282


namespace dining_bill_before_tip_l344_344037

theorem dining_bill_before_tip (share_per_person : ℝ) (num_people : ℕ) (tip_percent : ℝ) :
  share_per_person = 19.1125 → num_people = 8 → tip_percent = 0.10 → 
  let B := (share_per_person * num_people) / (1 + tip_percent) 
  in B = 139 :=
by 
  intros
  let B := (share_per_person * num_people) / (1 + tip_percent)
  sorry

end dining_bill_before_tip_l344_344037


namespace least_value_of_b_l344_344784

-- Definitions based on the conditions
def has_exact_factors (n factors : ℕ) : Prop :=
  (finset.range n).filter (λ i, n % (i + 1) = 0).card = factors

def is_divisible_by (n m : ℕ) : Prop :=
  n % m = 0

-- Theorem statement based on the translation of the problem
theorem least_value_of_b (a b : ℕ) :
  a > 0 ∧ b > 0 ∧
  has_exact_factors a 4 ∧
  has_exact_factors b a ∧
  is_divisible_by b (a + 1) →
  b = 42 :=
sorry

end least_value_of_b_l344_344784


namespace parallel_lines_iff_equal_slopes_l344_344271

theorem parallel_lines_iff_equal_slopes (k1 k2 : ℝ) : 
  (∀ x y : ℝ, k1 * x + y + 1 = 0 ∧ k2 * x + y - 1 = 0) ↔ (k1 = k2) : Prop :=
begin
  sorry
end

end parallel_lines_iff_equal_slopes_l344_344271


namespace remainder_when_product_divided_by_5_l344_344582

theorem remainder_when_product_divided_by_5 :
  (∏ i in Finset.range 20, (10 * i + 4)) % 5 = 1 :=
by
  sorry

end remainder_when_product_divided_by_5_l344_344582


namespace mean_of_four_numbers_l344_344034

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 1/2) : (a + b + c + d) / 4 = 1 / 8 :=
by
  -- proof skipped
  sorry

end mean_of_four_numbers_l344_344034


namespace evaluate_expression_l344_344180

theorem evaluate_expression : (∃ (x : Real), 6 < x ∧ x < 7 ∧ x = Real.sqrt 45) → (Int.floor (Real.sqrt 45))^2 + 2*Int.floor (Real.sqrt 45) + 1 = 49 := 
by
  sorry

end evaluate_expression_l344_344180


namespace combined_area_of_removed_triangles_l344_344136

theorem combined_area_of_removed_triangles (x s : ℝ) (h1 : s ∈ ℝ) (h2 : x ∈ ℝ) (h3 : x - 2 * s = 15) : 
  2 * s^2 = 225 :=
by
  sorry

#align combined_area_of_removed_triangles

end combined_area_of_removed_triangles_l344_344136


namespace solve_system_of_equations_l344_344988

noncomputable theory
open Classical Real

theorem solve_system_of_equations (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x - sqrt (x * y) - 4 * sqrt (x / y) + 2 = 0 ∧ 2 * x^2 + x^2 * y^4 = 18 * y^2)
  ↔ ((x = 2 ∧ y = 2) ∨ (x = (sqrt 286)^(1/4) / 4 ∧ y = (sqrt 286)^(1/4))) :=
by sorry

end solve_system_of_equations_l344_344988


namespace min_f_value_min_fraction_value_l344_344257

noncomputable theory

def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Problem (1)
theorem min_f_value : ∃ x₀ : ℝ, f x₀ ≤ 2 :=
sorry

-- Problem (2)
theorem min_fraction_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 3 * a + b = 2) :
  (1 / (2 * a) + 1 / (a + b)) ≥ 2 :=
sorry

end min_f_value_min_fraction_value_l344_344257


namespace eq_neg_one_fifth_l344_344144

theorem eq_neg_one_fifth : 
  ((1 : ℝ) / ((-5) ^ 4) ^ 2 * (-5) ^ 7) = -1 / 5 := by
  sorry

end eq_neg_one_fifth_l344_344144


namespace distance_between_parallel_lines_l344_344797

theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y, x + 2 * y + 3 = 0
  let line2 : ℝ → ℝ → Prop := λ x y, 2 * x + 4 * y + 5 = 0
  ∃ d : ℝ, d = |(-6) - (-5)| / sqrt(4 + 16) ∧ d = sqrt(5) / 10 :=
sorry

end distance_between_parallel_lines_l344_344797


namespace probability_of_bug_visiting_all_vertices_exactly_once_in_three_moves_l344_344088

def tetrahedron (V E : Type) := 
  (is_tetrahedron : ∀ (v : V), ∃! (e : set E), ∃! v' : V, v ≠ v' ∧ (v,v') ∈ e)

def bug_moves (bug : Type) (V : Type) (E : Type) := 
  (initial_vertex : V)
  (can_move : bug → V → list V)
  (adjacent_move : ∀ (b : bug) (v v' : V), (v, v') ∈ E → (v' ∈ can_move b v))

def probability_of_visiting_all_vertices (α : Type) (V : Type) : α := sorry

theorem probability_of_bug_visiting_all_vertices_exactly_once_in_three_moves
  (α : Type) (V : Type) (E : Type)
  (t : tetrahedron V E)
  (b : bug_moves α V E) :
  probability_of_visiting_all_vertices α V = 1/9 := 
sorry

end probability_of_bug_visiting_all_vertices_exactly_once_in_three_moves_l344_344088


namespace probability_foci_on_y_axis_l344_344390

-- Define the interval [1, 5] and the condition for foci on the y-axis
def in_interval (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 5
def foci_on_y_axis (m : ℝ) : Prop := m ^ 2 > 4

-- Define the probability calculation
def probability := (5 - 2) / (5 - 1) -- Equivalent to 3/4

-- The theorem to prove
theorem probability_foci_on_y_axis :
  ∀ m : ℝ, in_interval m → (foci_on_y_axis m → probability = 3 / 4) :=
by
  intros m h_in_interval h_foci
  rw [probability]
  norm_num
  trivial

end probability_foci_on_y_axis_l344_344390


namespace fraction_to_decimal_l344_344965

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l344_344965


namespace moles_of_Be2C_required_l344_344639

def chemical_equation_balanced : Prop :=
  (∀ (x : ℕ), (Be2C(x) + 4 * H2O(x) → 2 * Be(OH)2(x) + CH4(x)))

def reaction_moles (mBeOH2 : ℕ) (mCH4 : ℕ) : ℕ :=
  3 -- 3 moles of Be2C required

theorem moles_of_Be2C_required :
  ∀ (mH2O : ℕ), 
  mH2O = 12 → chemical_equation_balanced → reaction_moles 6 3 = 3 :=
by
  intros mH2O mH2O_eq equation
  simp [reaction_moles, mH2O_eq]
  sorry  

end moles_of_Be2C_required_l344_344639


namespace solution_to_inequalities_l344_344573

theorem solution_to_inequalities (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end solution_to_inequalities_l344_344573


namespace Nadine_pebbles_l344_344749

theorem Nadine_pebbles :
  ∀ (white red blue green x : ℕ),
    white = 20 →
    red = white / 2 →
    blue = red / 3 →
    green = blue + 5 →
    red = (1/5) * x →
    x = 50 :=
by
  intros white red blue green x h_white h_red h_blue h_green h_percentage
  sorry

end Nadine_pebbles_l344_344749


namespace canvas_vs_plastic_l344_344746

theorem canvas_vs_plastic (canvas_co2_pounds : ℕ) 
                          (plastic_co2_ounces : ℕ) 
                          (bags_per_trip : ℕ) 
                          (pounds_to_ounces : ℕ) 
                          (canvas_co2_ounces : ℕ) 
                          (trips_required : ℕ) 
                          (conversion_factor : pounds_to_ounces = 16) 
                          (canvas_co2_ounces := canvas_co2_pounds * pounds_to_ounces)
                          (plastic_co2_per_trip := plastic_co2_ounces * bags_per_trip) :
                          canvas_co2_pounds = 600 →
                          plastic_co2_ounces = 4 →
                          bags_per_trip = 8 →
                          canvas_co2_ounces / plastic_co2_per_trip = 300 :=
begin
  intros,
  sorry
end

end canvas_vs_plastic_l344_344746


namespace trigonometric_identity_correct_l344_344594

noncomputable def trigonometric_identity (a : ℝ) : Prop :=
  cos (a - π / 3) - cos a = 1 / 3 → cos (a + π / 3) = -1 / 3

theorem trigonometric_identity_correct (a : ℝ) : trigonometric_identity a := by
  sorry

end trigonometric_identity_correct_l344_344594


namespace radius_of_circle_l344_344432

noncomputable def radius_circle (rho theta phi: ℝ) : ℝ := 
  let x := rho * Real.sin(phi) * Real.cos(theta)
  let y := rho * Real.sin(phi) * Real.sin(theta)
  let z := rho * Real.cos(phi)
  Real.sqrt (x^2 + y^2)

theorem radius_of_circle :
  ∀ (theta: ℝ),
  radius_circle 2 theta (Real.pi / 4) = Real.sqrt 2 :=
by
  intro theta
  -- proceed with the proof steps
  sorry

end radius_of_circle_l344_344432


namespace equatorial_expression_count_l344_344401

-- Definition of equatorial algebra operations
def natural (x y : ℝ) := x + y
def sharp (x y : ℝ) := max x y
def flat (x y : ℝ) := min x y

-- Definition of complexity criteria
inductive EquatorialExpression : Type
| var_x : EquatorialExpression
| var_y : EquatorialExpression
| var_z : EquatorialExpression
| op_natural (P Q : EquatorialExpression) : EquatorialExpression
| op_sharp (P Q : EquatorialExpression) : EquatorialExpression
| op_flat (P Q : EquatorialExpression) : EquatorialExpression

-- Recursively calculate the complexity of the expressions
def complexity : EquatorialExpression → ℕ
| EquatorialExpression.var_x => 0
| EquatorialExpression.var_y => 0
| EquatorialExpression.var_z => 0
| EquatorialExpression.op_natural P Q => 1 + complexity P + complexity Q
| EquatorialExpression.op_sharp P Q => 1 + complexity P + complexity Q
| EquatorialExpression.op_flat P Q => 1 + complexity P + complexity Q

-- The final proof statement
theorem equatorial_expression_count : (∃ n, n = 419 ∧ 
  ∀ f : {f : ℝ × ℝ × ℝ → ℝ // ∃ e : EquatorialExpression, complexity e ≤ 3}, true) :=
sorry -- Proof is omitted

end equatorial_expression_count_l344_344401


namespace quadratic_equation_unique_solution_l344_344429

theorem quadratic_equation_unique_solution 
  (a c : ℝ) (h1 : ∃ x : ℝ, a * x^2 + 8 * x + c = 0)
  (h2 : a + c = 10)
  (h3 : a < c) :
  (a, c) = (2, 8) := 
sorry

end quadratic_equation_unique_solution_l344_344429


namespace boys_at_least_as_many_as_girls_l344_344330

-- Define the conditions given in the problem
variables {B G : Type} -- Sets of boys and girls
variables (knows : B → set G) (knows_each_other : G → G → Prop)

-- Constraints:
-- 1. Each boy knows only girls who are acquainted with each other.
def boy_condition (b : B) : Prop :=
  ∀ g1 g2 : G, g1 ∈ knows b → g2 ∈ knows b → g1 ≠ g2 → knows_each_other g1 g2

-- 2. Each girl knows more boys than girls.
def girl_condition (g : G) (n_girls_knows : G → set G) (n_boys_knows : G → set B) : Prop :=
  g ∈ n_girls_knows g →
  g ∈ n_boys_knows g →  
  #(n_boys_knows g) > #(n_girls_knows g)

-- The statement to be proved, i.e., there are at least as many boys as there are girls
theorem boys_at_least_as_many_as_girls
  (boys : finset B) (girls : finset G)
  (knows : B → set G) (knows_each_other : G → G → Prop)
  (n_girls_knows : G → set G) (n_boys_knows : G → set B)
  (all_boys_condition : ∀ b ∈ boys, boy_condition knows_each_other knows b)
  (all_girls_condition : ∀ g ∈ girls, girl_condition n_girls_knows n_boys_knows g) :
  boys.card ≥ girls.card :=
by
  sorry

end boys_at_least_as_many_as_girls_l344_344330


namespace max_drumming_bunnies_l344_344496

structure Bunny where
  drum : ℕ
  drumsticks : ℕ

def can_drum (b1 b2 : Bunny) : Bool :=
  b1.drum > b2.drum ∧ b1.drumsticks > b2.drumsticks

theorem max_drumming_bunnies (bunnies : List Bunny) (h_size : bunnies.length = 7) : 
  ∃ (maxDrumming : ℕ), maxDrumming = 6 := 
by
  have h_drumming_limits : ∃ n, n ≤ 6 := 
    sorry -- Placeholder for the reasoning step
  use 6
  apply Eq.refl

-- Sorry is used to bypass the detailed proof, and placeholder comments indicate the steps needed for proof reasoning.

end max_drumming_bunnies_l344_344496


namespace intersection_point_of_lines_l344_344996

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 3 * x + 4 * y - 2 = 0 ∧ 2 * x + y + 2 = 0 := 
by
  sorry

end intersection_point_of_lines_l344_344996


namespace cubic_eq_one_real_root_l344_344614

-- Given a, b, c forming a geometric sequence
variables {a b c : ℝ}

-- Definition of a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Equation ax^3 + bx^2 + cx = 0
def cubic_eq (a b c x : ℝ) : Prop :=
  a * x^3 + b * x^2 + c * x = 0

-- Prove the number of real roots
theorem cubic_eq_one_real_root (h : geometric_sequence a b c) :
  ∃ x : ℝ, cubic_eq a b c x ∧ ¬∃ y ≠ x, cubic_eq a b c y :=
sorry

end cubic_eq_one_real_root_l344_344614


namespace value_of_some_number_l344_344664

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l344_344664


namespace speed_of_faster_train_l344_344050

-- Definitions based on the conditions.
def length_train_1 : ℝ := 180
def length_train_2 : ℝ := 360
def time_to_cross : ℝ := 21.598272138228943
def speed_slow_train_kmph : ℝ := 30
def speed_fast_train_kmph : ℝ := 60

-- The theorem that needs to be proven.
theorem speed_of_faster_train :
  (length_train_1 + length_train_2) / time_to_cross * 3.6 = speed_slow_train_kmph + speed_fast_train_kmph :=
sorry

end speed_of_faster_train_l344_344050


namespace rational_pairs_a_b_l344_344085

theorem rational_pairs_a_b (a b : ℚ) (q : ℕ) (h0 : 0 < a) (h1 : a < b) (h2 : a^a = b^b) (h3 : q ≥ 2) :
  (a, b) = ((q-1)/q)^q, ((q-1)/q)^(q-1) :=
sorry

end rational_pairs_a_b_l344_344085


namespace AdultsNotWearingBlue_l344_344140

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end AdultsNotWearingBlue_l344_344140


namespace construction_not_feasible_l344_344234

-- This is our theorem statement in Lean
theorem construction_not_feasible (A B C A' B' C': Type) (O: Type)
  [triangle ABC] [equilateral_triangle A'B'C'] 
  (acute_angle_ABC : acute_angle ABC)
  (passes_through_O : passes_through O [A, A', B, B', C, C']) :
  ¬ ∃ constr : (∀(T : Type), A'B'C' ⊆ T),
    (∃ acute_angle (T : Type), acute_angle T) → (∀ (T : Type) (constr : ∀(ABC : Type), acute_angle ABC), equilateral_triangle (A'B'C')) :=
sorry

end construction_not_feasible_l344_344234


namespace exist_distinct_integers_sum_squares_eq_sum_cubes_l344_344761

theorem exist_distinct_integers_sum_squares_eq_sum_cubes (k : ℕ) (h : 0 < k) : 
  ∃ (a : Fin k → ℤ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (Finset.univ.sum (λ i, (a i) ^ 2) = Finset.univ.sum (λ i, (a i) ^ 3)) :=
  sorry

end exist_distinct_integers_sum_squares_eq_sum_cubes_l344_344761


namespace scaled_model_height_l344_344687

-- Conditions
def original_height : ℝ := 60
def original_spherical_top_volume : ℝ := 150000
def scaled_spherical_top_volume : ℝ := 0.15
def height_ratio := ((100 : ℝ))⁻¹ -- obtained from cube root of volume ratio

-- Given radius calculation function
noncomputable def radius_calculation (volume : ℝ) : ℝ := cbrt (3 * volume / (4 * π))

-- Calculation of the original sphere radius
noncomputable def original_radius : ℝ := radius_calculation original_spherical_top_volume

-- Calculation of total height in original tower
def total_original_height : ℝ := 3 * original_radius

-- Scaling down the total height
noncomputable def scaled_total_height : ℝ := total_original_height * height_ratio

-- The theorem to prove
theorem scaled_model_height :
  scaled_total_height = 3 * cbrt (450000 / (4 * π)) / 100 :=
by
  sorry

end scaled_model_height_l344_344687


namespace min_value_of_quadratic_form_l344_344618

theorem min_value_of_quadratic_form (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + 2 * y^2 + 3 * z^2 ≥ 1/3 :=
sorry

end min_value_of_quadratic_form_l344_344618


namespace journey_includes_every_road_l344_344117

-- Define the given conditions of the problem:
def town_planner_conditions (N : ℕ) : Prop :=
  ∃ (roundabouts : Finset Roundabout) (roads : Finset Road),
  roundabouts.card = 2 * N ∧
  (∀ r ∈ roundabouts, (roads.filter (λ road, road.connects_roundabout r)).card = 3) ∧
  are_all_roads_two_way roads ∧
  are_all_roundabouts_clockwise roundabouts ∧
  Vlad_takes_first_exit roundabouts roads

-- State the problem:
theorem journey_includes_every_road (N : ℕ) (h : town_planner_conditions N) : N % 2 = 1 :=
sorry

end journey_includes_every_road_l344_344117


namespace find_coefficients_l344_344538

variables {x1 x2 x3 x4 x5 x6 x7 : ℝ}

theorem find_coefficients
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 14)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 30)
  (h4 : 16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 70) :
  25*x1 + 36*x2 + 49*x3 + 64*x4 + 81*x5 + 100*x6 + 121*x7 = 130 :=
sorry

end find_coefficients_l344_344538


namespace hours_per_day_asphalting_l344_344828

def asphalting_hours (H : ℝ) : Prop :=
  let man_hours_1km := 30 * 12 * 8
  let man_hours_2km := man_hours_1km * 2
  let total_man_hours := 20 * 19.2 * H
  man_hours_2km = total_man_hours

theorem hours_per_day_asphalting : asphalting_hours 15 :=
by
  let man_hours_1km := 30 * 12 * 8
  let man_hours_2km := man_hours_1km * 2
  let total_man_hours_for_15 := 20 * 19.2 * 15
  have : man_hours_2km = total_man_hours_for_15 := by
    unfold man_hours_1km man_hours_2km total_man_hours_for_15 at *
    norm_num
  exact this

end hours_per_day_asphalting_l344_344828


namespace black_faces_possible_values_l344_344598

theorem black_faces_possible_values :
  ∃ (n : ℕ), 23 ≤ n ∧ n ≤ 25 ∧
  (∃ (cubes : Fin 8 → Fin 6 → Bool), 
  let faces := (λ (i : Fin 8), (λ (j : Fin 6), cubes i j)) in
  (∃ (count_black_faces : Fin 8 → ℕ), 
  let total_black_faces := (λ (i : Fin 8), (count_black_faces i)) in 
  (∑ i, total_black_faces i = n) ∧ 
  (∑ i, total_black_faces i = 24))) :=
  sorry

end black_faces_possible_values_l344_344598


namespace find_x_l344_344987

theorem find_x (x : ℝ) (h : log 10 (3 * x - 5) = 2) : x = 35 := 
by
  sorry

end find_x_l344_344987


namespace Ernesto_forms_figure_with_d_l344_344567

def figure (segments semicircles : ℕ) : Type := 
{ s : ℕ // (s = segments, s = semicircles) }

def original_figure := figure 4 3

def option_a := figure 3 1
def option_b := figure 0 1
def option_c := figure 0 0
def option_d := figure 4 3
def option_e := figure 0 4

theorem Ernesto_forms_figure_with_d : 
original_figure = option_d :=
by sorry

end Ernesto_forms_figure_with_d_l344_344567


namespace vincent_books_l344_344454

theorem vincent_books (x : ℕ) (h1 : 10 + 3 + x = 13 + x)
                      (h2 : 16 * (13 + x) = 224) : x = 1 :=
by sorry

end vincent_books_l344_344454


namespace x_is_integer_l344_344206

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ a : ℤ, a = x^1960 - x^1919)
  (h2 : ∃ b : ℤ, b = x^2001 - x^1960) :
  ∃ k : ℤ, x = k :=
sorry

end x_is_integer_l344_344206


namespace cost_of_one_dozen_pens_l344_344408

-- Definitions for conditions
variable {x : ℝ} (hx : x > 0)

def cost_pen := 5 * x
def cost_pencil := x

def total_cost := 3 * cost_pen hx + 5 * cost_pencil hx

-- Theorem for the proof problem
theorem cost_of_one_dozen_pens 
  (h1 : total_cost hx = 240)
  (h2 : 5 * x * 12 = 720) :
  12 * cost_pen hx = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l344_344408


namespace arithmetic_mean_of_fractions_l344_344841

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 7
  let b := (5 : ℚ) / 9
  let c := (7 : ℚ) / 11
  (a + b + c) / 3 = 1123 / 2079 :=
by
  let a := (3 : ℚ) / 7
  let b := (5 : ℚ) / 9
  let c := (7 : ℚ) / 11
  have h_a : a = 3 / 7 := rfl
  have h_b : b = 5 / 9 := rfl
  have h_c : c = 7 / 11 := rfl
  have common_denominator_sum : a + b + c = 1123 / 693 :=
    calc a + b + c
        = 3 / 7 + 5 / 9 + 7 / 11 : by rw [h_a, h_b, h_c]
    ... = 297 / 693 + 385 / 693 + 441 / 693 : by sorry -- Conversion to common denominator
    ... = 1123 / 693 : by sorry -- Summation
  have mean_result : (a + b + c) / 3 = (1123 / 693) / 3 := by rw common_denominator_sum
  show (a + b + c) / 3 = 1123 / 2079,
  from calc (a + b + c) / 3
           = (1123 / 693) / 3 : by rw mean_result
       ... = 1123 / 2079 : by sorry -- Actual division step

end arithmetic_mean_of_fractions_l344_344841


namespace white_crows_likelier_remain_l344_344441

-- Let's define our conditions
variables (a b c d : ℕ) -- a, b on birch; c, d on oak
constants (total_birch total_oak : ℕ) (h_birch : a + b = total_birch) (h_oak : c + d = total_oak)
constants (h_b_at_least_a : b ≥ a) (h_d_c : d ≥ c - 1)

-- The event that the number of white crows on the birch tree remains the same after transitions
def prob_A := (b * (d+1) + a * (c+1)) / (total_birch * 51)

-- The event that the number of white crows on the birch tree changes after transitions
def prob_not_A := (b * c + a * d) / (total_birch * 51)

-- The proof statement
theorem white_crows_likelier_remain (h_b_at_least_a : b ≥ a) (h_d_c : d ≥ c - 1) :
  prob_A a b c d total_birch > prob_not_A a b c d total_birch :=
sorry

end white_crows_likelier_remain_l344_344441


namespace apple_tree_total_apples_l344_344132

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l344_344132


namespace student_B_speed_l344_344513

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l344_344513


namespace find_ec_l344_344240

theorem find_ec (angle_A : ℝ) (BC : ℝ) (BD_perp_AC : Prop) (CE_perp_AB : Prop)
  (angle_DBC_2_angle_ECB : Prop) :
  angle_A = 45 ∧ 
  BC = 8 ∧
  BD_perp_AC ∧
  CE_perp_AB ∧
  angle_DBC_2_angle_ECB → 
  ∃ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 2 ∧ a + b + c = 7 :=
sorry

end find_ec_l344_344240


namespace cube_distance_to_plane_l344_344881

/-- A cube with side length 8 is suspended above a plane. The vertex closest to the plane 
  is labeled A and the three vertices adjacent to vertex A are at heights 8, 9, and 10 above the plane.
  Prove that the distance from vertex A to the plane can be expressed as (27 - sqrt 280) / 5. -/
theorem cube_distance_to_plane :
  ∃ p q u : ℕ, p = 27 ∧ q = 280 ∧ u = 5 ∧
  let dist := (p - Real.sqrt q) / u in
  dist = (27 - Real.sqrt 280) / 5 :=
by
  sorry

end cube_distance_to_plane_l344_344881


namespace men_entered_l344_344705

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l344_344705


namespace nth_number_eq_l344_344518

noncomputable def nth_number (n : Nat) : ℚ := n / (n^2 + 1)

theorem nth_number_eq (n : Nat) : nth_number n = n / (n^2 + 1) :=
by
  sorry

end nth_number_eq_l344_344518


namespace packing_heights_difference_l344_344450

-- Definitions based on conditions
def diameter := 8   -- Each pipe has a diameter of 8 cm
def num_pipes := 160 -- Each crate contains 160 pipes

-- Heights of the crates based on the given packing methods
def height_crate_A := 128 -- Calculated height for Crate A

noncomputable def height_crate_B := 8 + 60 * Real.sqrt 3 -- Calculated height for Crate B

-- Positive difference in the total heights of the two packings
noncomputable def delta_height := height_crate_A - height_crate_B

-- The goal to prove
theorem packing_heights_difference :
  delta_height = 120 - 60 * Real.sqrt 3 :=
sorry

end packing_heights_difference_l344_344450


namespace coefficient_of_x3_in_expansion_l344_344695

/-- Given the expansion of (x - 3/x)^5, prove that the coefficient of x^3 is -15. -/
theorem coefficient_of_x3_in_expansion : 
    ∀ (x : ℝ), (coefficients (expand ((x - 3 / x)^5)) 3) = -15 := 
by
  sorry

end coefficient_of_x3_in_expansion_l344_344695


namespace percentage_per_annum_is_correct_l344_344792

-- Define the conditions of the problem
def banker_gain : ℝ := 24
def present_worth : ℝ := 600
def time : ℕ := 2

-- Define the formula for the amount due
def amount_due (r : ℝ) (t : ℕ) (PW : ℝ) : ℝ := PW * (1 + r * t)

-- Define the given conditions translated from the problem
def given_conditions (r : ℝ) : Prop :=
  amount_due r time present_worth = present_worth + banker_gain

-- Lean statement of the problem to be proved
theorem percentage_per_annum_is_correct :
  ∃ r : ℝ, given_conditions r ∧ r = 0.02 :=
by {
  sorry
}

end percentage_per_annum_is_correct_l344_344792


namespace direction_and_distance_fuel_consumption_l344_344829

-- Define the distances traveled
def distances : List ℤ := [+2, -3, +4, -1, -5, +3, -6, +2]

-- Define the fuel consumption per kilometer
variable (a : ℝ)

-- Define a function to calculate the total distance including return to the starting point
noncomputable def total_fuel_consumption (d : List ℤ) (a : ℝ) : ℝ :=
  let total_distance := d.map(λ x => abs x).sum
  let return_distance := abs(d.sum)
  (total_distance + return_distance) * a

-- Statement 1: Prove the direction and distance from the starting point
theorem direction_and_distance : distances.sum = -4 := by
  sorry

-- Statement 2: Prove the total fuel consumption including return to the starting point
theorem fuel_consumption : total_fuel_consumption distances a = 30 * a := by
  sorry

end direction_and_distance_fuel_consumption_l344_344829


namespace original_number_is_45_l344_344904

theorem original_number_is_45 (x y : ℕ) (h1 : x + y = 9) (h2 : 10 * y + x = 10 * x + y + 9) : 10 * x + y = 45 := by
  sorry

end original_number_is_45_l344_344904


namespace sqrt_of_300_l344_344772

theorem sqrt_of_300 : sqrt 300 = 10 * sqrt 3 := by
  have h300 : 300 = 2 ^ 2 * 3 * 5 ^ 2 := by norm_num
  rw [h300, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_pow, Real.sqrt_pow]
  all_goals { norm_num, apply_instance }
  sorry

end sqrt_of_300_l344_344772


namespace fraction_to_decimal_l344_344979

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344979


namespace exists_h_not_divisible_l344_344947

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end exists_h_not_divisible_l344_344947


namespace f_odd_f_increasing_solve_inequality_l344_344630

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.exp x + 1) + 1

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

theorem solve_inequality (x : ℝ) :
  (f (Real.log (Real.log 2 x) ^ 2) + f (Real.log (Real.log (Real.sqrt 2) x) - 3) ≤ 0) ↔ (x ∈ Set.Icc (1/8) 2) := sorry

end f_odd_f_increasing_solve_inequality_l344_344630


namespace variance_of_nums_l344_344950

def nums : List ℕ := [6, 9, 5, 8, 10, 4]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x : ℚ) - m)^2).sum / l.length

theorem variance_of_nums :
  variance nums = 14 / 3 :=
by
  sorry

end variance_of_nums_l344_344950


namespace marek_sequence_sum_l344_344865

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end marek_sequence_sum_l344_344865


namespace fraction_to_decimal_l344_344956

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l344_344956


namespace biased_die_expected_value_is_neg_1_5_l344_344873

noncomputable def biased_die_expected_value : ℚ :=
  let prob_123 := (1 / 6 : ℚ) + (1 / 6) + (1 / 6)
  let prob_456 := (1 / 2 : ℚ)
  let gain := prob_123 * 2
  let loss := prob_456 * -5
  gain + loss

theorem biased_die_expected_value_is_neg_1_5 :
  biased_die_expected_value = - (3 / 2 : ℚ) :=
by
  -- We skip the detailed proof steps here.
  sorry

end biased_die_expected_value_is_neg_1_5_l344_344873


namespace scientific_notation_l344_344982

theorem scientific_notation (n : ℕ) (h : n = 660000) : 
  n = 6.6 * 10^5 := 
  by
    sorry

end scientific_notation_l344_344982


namespace common_chord_length_eq_sqrt_3_l344_344673

theorem common_chord_length_eq_sqrt_3
  (a : ℝ)
  (h_a_pos : a > 0)
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 4 → x^2 + y^2 + 2*a*y - 6 = 0)
  (chord_length : 2) :
  a = Real.sqrt 3 := by
  sorry

end common_chord_length_eq_sqrt_3_l344_344673


namespace avg_weight_class_correct_l344_344039

-- Definitions based on the conditions
def sectionA_students : Nat := 40
def sectionB_students : Nat := 20

def avg_weight_sectionA : Float := 50
def avg_weight_sectionB : Float := 40

-- Compute total weights
def total_weight_sectionA : Float := sectionA_students * avg_weight_sectionA
def total_weight_sectionB : Float := sectionB_students * avg_weight_sectionB

-- Compute total weight of the class
def total_weight_class : Float := total_weight_sectionA + total_weight_sectionB

-- Compute total number of students
def total_students : Nat := sectionA_students + sectionB_students

-- Compute the average weight of the class
def avg_weight_class : Float := total_weight_class / total_students

-- The theorem stating the average weight of the class is 46.67 kg
theorem avg_weight_class_correct : avg_weight_class = 46.67 :=
by
  sorry

end avg_weight_class_correct_l344_344039


namespace f_2023_value_l344_344029

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : 2^n = a + b) : f a + f b = n^2 + 1

theorem f_2023_value : f 2023 = 107 :=
by 
  sorry

end f_2023_value_l344_344029


namespace simplify_sqrt_300_l344_344774

theorem simplify_sqrt_300 :
  sqrt 300 = 10 * sqrt 3 :=
by
  -- Proof would go here
  sorry

end simplify_sqrt_300_l344_344774


namespace trajectory_of_center_l344_344315

theorem trajectory_of_center :
  ∃ (x y : ℝ), (x + 1) ^ 2 + y ^ 2 = 49 / 4 ∧ (x - 1) ^ 2 + y ^ 2 = 1 / 4 ∧ ( ∀ P, (P = (x, y) → (P.1^2) / 4 + (P.2^2) / 3 = 1) ) := sorry

end trajectory_of_center_l344_344315


namespace maintenance_increase_l344_344134

theorem maintenance_increase (I : ℕ) (f : ℝ) (I_new : ℝ) : 
  I = 50 → 
  f = 1.20 → 
  I_new = I * f → 
  I_new = 60 :=
by
  intros hI hf hI_new
  rw [hI, hf, hI_new]
  norm_num

end maintenance_increase_l344_344134


namespace geometric_sequence_sum_ratio_l344_344433

variable {α : Type*} [LinearOrderedField α]

-- Define the sums of the first n terms in a geometric sequence
def sum_geometric_seq (n : ℕ) (a r : α) : α := a * (1 - r^n) / (1 - r)

-- Variables representing the sums of the sequence
variables (S_n S_2n S_3n : α) (n : ℕ)

-- Given conditions
axiom sum_S_2n : S_2n = 3 * S_n

-- Claim to prove
theorem geometric_sequence_sum_ratio (S_n S_2n S_3n : α) (h : S_2n = 3 * S_n) :
  S_3n = 7 * S_n → S_3n / S_2n = 7 / 3 :=
by
  intro h₁,
  have h₂ : S_2n ≠ 0 := by sorry, -- Since S_n is part of a geometric sequence, S_2n is not zero.
  rw [h₁, h] at *,
  field_simp [h₂]
  sorry

end geometric_sequence_sum_ratio_l344_344433


namespace count_different_numerators_l344_344348

theorem count_different_numerators :
  let T := { r : ℚ | (0 < r) ∧ (r < 1) ∧ ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧ r = (a * 10 + b) / 99 }
  (set.image (λ r : ℚ, (numerator r).nat_abs) T).to_finset.card = 60 :=
by {
  let T := { r : ℚ | (0 < r) ∧ (r < 1) ∧ ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧ r = (a * 10 + b) / 99 },
  exact (set.image (λ r : ℚ, (numerator r).nat_abs) T).to_finset.card sorry
}

end count_different_numerators_l344_344348


namespace orange_juice_bottles_l344_344862

theorem orange_juice_bottles (O A : ℕ) 
  (h1 : O + A = 70) 
  (h2 : 0.70 * O + 0.60 * A = 46.20) : 
  O = 42 := 
by 
  sorry

end orange_juice_bottles_l344_344862


namespace fraction_to_decimal_l344_344974

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344974


namespace square_tiles_count_l344_344087

theorem square_tiles_count (t s p : ℕ) (h1 : t + s + p = 30) (h2 : 3 * t + 4 * s + 5 * p = 108) : s = 6 := by
  sorry

end square_tiles_count_l344_344087


namespace new_plan_cost_correct_l344_344739

-- Define the conditions
def old_plan_cost := 150
def increase_rate := 0.3

-- Define the increased amount
def increase_amount := increase_rate * old_plan_cost

-- Define the cost of the new plan
def new_plan_cost := old_plan_cost + increase_amount

-- Prove the main statement
theorem new_plan_cost_correct : new_plan_cost = 195 :=
by
  sorry

end new_plan_cost_correct_l344_344739


namespace solve_equation_l344_344989

noncomputable def cube_root (x : ℝ) := x^(1 / 3)

theorem solve_equation (x : ℝ) :
  cube_root x = 15 / (8 - cube_root x) →
  x = 27 ∨ x = 125 :=
by
  sorry

end solve_equation_l344_344989


namespace claudia_total_earnings_l344_344923

-- Definition of the problem conditions
def class_fee : ℕ := 10
def kids_saturday : ℕ := 20
def kids_sunday : ℕ := kids_saturday / 2

-- Theorem stating that Claudia makes $300.00 for the weekend
theorem claudia_total_earnings : (kids_saturday * class_fee) + (kids_sunday * class_fee) = 300 := 
by
  sorry

end claudia_total_earnings_l344_344923


namespace count_squares_in_3x3_grid_l344_344449

def num_squares_in_grid (grid_points: Finset (Fin 3 × Fin 3)) : ℕ :=
  let small_squares := 4
  let medium_squares := 4
  let large_squares := 1
  let small_diagonal_squares := 2
  small_squares + medium_squares + large_squares + small_diagonal_squares

theorem count_squares_in_3x3_grid :
  num_squares_in_grid (Finset.univ : Finset (Fin 3 × Fin 3)) = 11 :=
by
  simp [num_squares_in_grid]
  sorry

end count_squares_in_3x3_grid_l344_344449


namespace find_initial_breads_l344_344438

def bread_theft_problem : Prop :=
  ∃ B : ℕ , (∀ k : ℕ, k ∈ {1, 2, 3, 4, 5} →
  let L := [3, 3 + 1/2, 7, 7 + 1/2, 15, 15 + 1/2, 31, 31 + 1/2, 63, 63 + 1/2, 127] in
  ( B = L[10] ))

theorem find_initial_breads : bread_theft_problem := by
  sorry

end find_initial_breads_l344_344438


namespace tangent_line_to_parabola_k_value_l344_344586

theorem tangent_line_to_parabola_k_value (k : ℝ) :
  (∀ x y : ℝ, 4 * x - 3 * y + k = 0 → y^2 = 16 * x → (4 * x - 3 * y + k = 0 ∧ y^2 = 16 * x) ∧ (144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_k_value_l344_344586


namespace minimize_distance_school_l344_344443

-- Define the coordinates for the towns X, Y, and Z
def X_coord : ℕ × ℕ := (0, 0)
def Y_coord : ℕ × ℕ := (200, 0)
def Z_coord : ℕ × ℕ := (0, 300)

-- Define the population of the towns
def X_population : ℕ := 100
def Y_population : ℕ := 200
def Z_population : ℕ := 300

theorem minimize_distance_school : ∃ (x y : ℕ), x + y = 300 := by
  -- This should follow from the problem setup and conditions.
  sorry

end minimize_distance_school_l344_344443


namespace remainder_of_b2_minus_3a_div_6_l344_344613

theorem remainder_of_b2_minus_3a_div_6 (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 5) : 
  (b^2 - 3 * a) % 6 = 1 := 
sorry

end remainder_of_b2_minus_3a_div_6_l344_344613


namespace exists_circle_touching_tangents_and_radius_l344_344407

theorem exists_circle_touching_tangents_and_radius 
  (r1 r2 r3 : ℝ) 
  (h1 : r1 > r2) 
  (h2 : r1 > r3) 
  (h_external : ∀ (k1 k2 : ℝ), k1 ≠ k2 → ¬intersects k1 k2) :
  ∃ (r : ℝ), r = r1 * r2 * r3 / (r1 * (r2 + r3) - r2 * r3) :=
by
  sorry

end exists_circle_touching_tangents_and_radius_l344_344407


namespace max_rectangle_area_l344_344024

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end max_rectangle_area_l344_344024


namespace new_plan_cost_correct_l344_344737

-- Define the conditions
def old_plan_cost := 150
def increase_rate := 0.3

-- Define the increased amount
def increase_amount := increase_rate * old_plan_cost

-- Define the cost of the new plan
def new_plan_cost := old_plan_cost + increase_amount

-- Prove the main statement
theorem new_plan_cost_correct : new_plan_cost = 195 :=
by
  sorry

end new_plan_cost_correct_l344_344737


namespace number_of_bits_ABCDEF_16_l344_344557

theorem number_of_bits_ABCDEF_16 : 
  let n := 11240375
  in ∃ k : ℕ, k = 24 ∧ (2^k > n ∧ 2^(k - 1) ≤ n) :=
by
  sorry

end number_of_bits_ABCDEF_16_l344_344557


namespace passengers_first_stop_l344_344005

theorem passengers_first_stop :
  ∃ (X : ℕ), -- There exists a natural number X
    X + 2 + 2 = 11 := -- The final equation considering all stops
begin
  -- Here we start the proof
  use 7, -- We will prove that X = 7 satisfies the equation
  exact rfl, -- And this is trivially true
end

end passengers_first_stop_l344_344005


namespace men_in_second_group_l344_344475

theorem men_in_second_group (M : ℕ) : 
    (18 * 20 = M * 24) → M = 15 :=
by
  intro h
  sorry

end men_in_second_group_l344_344475


namespace jason_seashells_initial_count_l344_344334

variable (initialSeashells : ℕ) (seashellsGivenAway : ℕ)
variable (seashellsNow : ℕ) (initialSeashells := 49)
variable (seashellsGivenAway := 13) (seashellsNow := 36)

theorem jason_seashells_initial_count :
  initialSeashells - seashellsGivenAway = seashellsNow → initialSeashells = 49 := by
  sorry

end jason_seashells_initial_count_l344_344334


namespace perpendicular_bisector_of_AB_l344_344294

theorem perpendicular_bisector_of_AB :
  ∃ A B : ℝ × ℝ, 
    (A ≠ B) ∧
    (A.1^2  + A.2^2 - 2 * A.1 - 5 = 0) ∧ (B.1^2 + B.2^2 - 2 * B.1 - 5 = 0) ∧
    (A.1^2  + A.2^2 + 2 * A.1 - 4 * B.2 - 4 = 0) ∧ (B.1^2 + B.2^2 + 2 * B.1 - 4 * B.2 - 4 = 0) ∧
    (∀ x y : ℝ, 
      (x, y) ∈ line_through_points A B →
      (x, y) ∈ perpendicular_bisector A B →
      x + y - 1 = 0) :=
sorry

end perpendicular_bisector_of_AB_l344_344294


namespace find_some_number_l344_344654

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l344_344654


namespace find_length_MN_l344_344094

variables (A B C D E M N : Point) (a b : ℝ)

-- Assume two circles:
-- 1. First circle with diameter AB.
-- 2. Second circle with center A intersecting first circle at points C and D and diameter at point E.
-- 3. Point M on arc CE of the second circle, without D.
-- 4. Ray BM intersects the first circle at point N.

-- Assume lengths:
-- Let |CN| = a and |DN| = b.

-- Show: |MN| = sqrt(ab)

axiom geometric_setup : diameter AB ∧ centered_at A ∧ (intersects C) ∧ (intersects D) ∧ (intersects E)
axiom point_M_on_CE : M ∉ {C, E} ∧ on_arc_CE M D
axiom ray_BM_intersects : intersects_at BM N
axiom lengths_given : (dist C N = a) ∧ (dist D N = b)
axiom angle_properties : (angle subtended_by (N B) on_circle AB = 90)

theorem find_length_MN : dist M N = Real.sqrt (a * b) :=
by
  sorry

end find_length_MN_l344_344094


namespace mark_new_phone_plan_cost_l344_344745

theorem mark_new_phone_plan_cost (old_cost : ℕ) (h_old_cost : old_cost = 150) : 
  let new_cost := old_cost + (0.3 * old_cost) in 
  new_cost = 195 :=
by 
  sorry

end mark_new_phone_plan_cost_l344_344745


namespace some_number_value_correct_l344_344659

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l344_344659


namespace factorial_division_l344_344460

theorem factorial_division : (13! - 12!) / 10! = 1584 := by
  sorry

end factorial_division_l344_344460


namespace range_of_g_l344_344253

noncomputable def f (t : ℝ) : ℝ := sqrt ((1 - t) / (1 + t))

noncomputable def g (x : ℝ) : ℝ := 
  let f_sin_x := f (sin x)
  let f_cos_x := f (cos x)
  cos x * f_sin_x + sin x * f_cos_x

theorem range_of_g : 
  set.range g = set.Icc (-sqrt 2 - 2) -3 :=
by
  sorry

end range_of_g_l344_344253


namespace three_b_minus_a_eq_neg_five_l344_344291

theorem three_b_minus_a_eq_neg_five (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : 3 * b - a = -5 :=
sorry

end three_b_minus_a_eq_neg_five_l344_344291


namespace johns_hat_total_cost_l344_344338

noncomputable def totalCost : ℕ → ℕ := 
  λ hats, if hats ≥ 50 then
    let odd_days := hats / 2
    let even_days := hats / 2
    let cost := (odd_days * 45) + (even_days * 60)
    cost - (cost / 10)
  else let odd_days := hats / 2
       let even_days := hats / 2
       (odd_days * 45) + (even_days * 60)

theorem johns_hat_total_cost : totalCost 140 = 6615 :=
  by
    sorry

end johns_hat_total_cost_l344_344338


namespace lucy_first_round_cookies_l344_344735

theorem lucy_first_round_cookies (x : ℕ) : 
  (x + 27 = 61) → x = 34 :=
by
  intros h
  sorry

end lucy_first_round_cookies_l344_344735


namespace work_increase_absent_members_l344_344309

theorem work_increase_absent_members {p W : ℝ} (hp : p > 0) (W : ℝ) :
  let absent_fraction := 1 / 7 in
  let present_fraction := 1 - absent_fraction in
  let original_share := W / p in
  let new_share := W / (present_fraction * p) in
  ((new_share - original_share) / original_share) * 100 = 16.67 :=
by
  sorry

end work_increase_absent_members_l344_344309


namespace cubic_has_one_real_root_l344_344426

theorem cubic_has_one_real_root :
  (∃ x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0) ∧ ∀ x y : ℝ, (x^3 - 6*x^2 + 9*x - 10 = 0) ∧ (y^3 - 6*y^2 + 9*y - 10 = 0) → x = y :=
by
  sorry

end cubic_has_one_real_root_l344_344426


namespace tan_theta_l344_344903

-- Define the side lengths of the triangle
def a := 13
def b := 14
def c := 15

-- Define the semi-perimeter of the triangle
def s := (a + b + c) / 2

-- Area of the triangle using Heron's formula
def A := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the conditions for the lines bisecting the perimeter and area
def p := (s - a) -- Placeholder for one line segment part
def q := (s - b) -- Placeholder for the other line segment part

-- Acute angle between the bisecting lines
def θ := 
  Real.sqrt (1 + (3 * Real.sqrt 391 / 210)^2) - (3 * Real.sqrt 391 / 210)

theorem tan_theta (a b c : ℕ) (h₁ : a = 13) (h₂ : b = 14) (h₃ : c = 15) 
  (p := (s - a)) (q := (s - b)) :
  Real.tan θ = (420 * Real.sqrt 391 - 3 * Real.sqrt 391) / 210 := by
  simp [h₁, h₂, h₃]
  sorry

end tan_theta_l344_344903


namespace calculate_value_l344_344365

noncomputable def P (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry
def d : ℝ := sorry

axiom condition_P1 : P 1 = 1993
axiom condition_P2 : P 2 = 3986
axiom condition_P3 : P 3 = 5979

theorem calculate_value : - (1 / 4) * (P 11 + P (-7)) = 5233 :=
by
  sorry

end calculate_value_l344_344365


namespace spherical_to_rectangular_coords_l344_344935

theorem spherical_to_rectangular_coords
  (ρ θ φ : ℝ)
  (hρ : ρ = 6)
  (hθ : θ = 7 * Real.pi / 4)
  (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = -3 * Real.sqrt 6 ∧ y = -3 * Real.sqrt 6 ∧ z = 3 :=
by
  sorry

end spherical_to_rectangular_coords_l344_344935


namespace cylinder_and_sphere_are_bodies_of_revolution_l344_344528

-- Definitions of geometric solids as per the conditions
def is_body_of_revolution (s: Type) : Prop :=
  ∃ (shape: Type) (axis: line), s = rotation_of_2Dshape_around_axis shape axis

def cylinder : Type := sorry
def hexagonal_pyramid : Type := sorry
def cube : Type := sorry
def sphere : Type := sorry
def tetrahedron : Type := sorry

-- Theorem statement to prove that the cylinder and sphere are bodies of revolution
theorem cylinder_and_sphere_are_bodies_of_revolution :
  is_body_of_revolution cylinder ∧ is_body_of_revolution sphere :=
sorry

end cylinder_and_sphere_are_bodies_of_revolution_l344_344528


namespace sum_of_sequence_l344_344375

theorem sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, ∑ i in finset.range n, a i = S n) →
  (∀ n, (∑ i in finset.range n, (a i)^2 / (i+1)^2) = (4 * n) - 4) →
  (∀ n, a n ≥ 0) →
  S 100 = 10098 :=
  sorry

end sum_of_sequence_l344_344375


namespace fraction_expression_l344_344461

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 :=
by
  sorry

end fraction_expression_l344_344461


namespace program_output_is_990_l344_344626

-- Define the initial values and the loop conditions
def initialState : Nat × Nat := (1, 10)

-- Define the loop action
def loopStep (state: Nat × Nat) : Nat × Nat :=
  (state.1 * (state.2 + 1), state.2 - 1)

-- Define the condition under which the loop terminates
def loopCondition (state: Nat × Nat) : Prop :=
  state.2 < 9

-- Define the final state after the loop executes
noncomputable def finalState : Nat × Nat :=
  let rec loop (state: Nat × Nat) : Nat × Nat :=
    if loopCondition(state) then state else loop (loopStep state)
  loop initialState

-- Statement of the proof problem
theorem program_output_is_990 : finalState.1 = 990 :=
by
  sorry

end program_output_is_990_l344_344626


namespace cole_drive_time_l344_344152

-- Definitions of variables
def distance : Type := ℝ
def time : Type := ℝ

-- Given conditions
variable (D : distance) -- distance from home to work
variable (v_work v_home : ℝ) (t_total : time)
variables (v_work := 75) (v_home := 105) (t_total := 2)

-- Time to work and back home equations
noncomputable def t_work := D / v_work
noncomputable def t_home := D / v_home

-- Proof statement
theorem cole_drive_time (h : t_work + t_home = t_total) : 
  t_work * 60 = 70 :=
by
s sorry

end cole_drive_time_l344_344152


namespace distance_vertex_A_face_BCD_unique_l344_344785

def tetrahedron_side_lengths (AB BD BC AC CD AD : ℝ) :=
  AB = 6 ∧ BD = 6 * (Real.sqrt 2) ∧ BC = 10 ∧ AC = 8 ∧ CD = 10 ∧ AD = 6

def distance_from_vertex_to_face (a b c : ℕ) :=
  gcd a c = 1 ∧ b ∉ (set.range (fun n : ℕ => n ^ 2)) ∧ a = 24 ∧ b = 41 ∧ c = 41

theorem distance_vertex_A_face_BCD_unique (a b c : ℕ)
  (h : tetrahedron_side_lengths 6 (6 * (Real.sqrt 2)) 10 8 10 6)
  (d : distance_from_vertex_to_face a b c) :
  100 * a + 10 * b + c = 2851 :=
by
  sorry

end distance_vertex_A_face_BCD_unique_l344_344785


namespace dice_roll_probability_is_correct_l344_344121

/-- Define the probability calculation based on conditions of the problem. --/
def dice_rolls_probability_diff_by_two (successful_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

/-- Given the problem conditions, there are 8 successful outcomes and 36 total outcomes. --/
theorem dice_roll_probability_is_correct :
  dice_rolls_probability_diff_by_two 8 36 = 2 / 9 :=
by
  sorry

end dice_roll_probability_is_correct_l344_344121


namespace constant_term_binomial_expansion_l344_344575

theorem constant_term_binomial_expansion :
  Let T (r : ℕ) := (Nat.choose 6 r) * ((-2)^r) * (x^((6 : ℤ) - (3 * r / 2 : ℤ))),
  ∃ r : ℕ, r = 4 ∧ (x^((6 : ℤ) - (3 * r / 2 : ℤ)) = 1) ∧ T(r) = 240 := sorry

end constant_term_binomial_expansion_l344_344575


namespace marble_problem_l344_344534

theorem marble_problem (a : ℚ) (h1 : ∑ i in {a, 1.5 * a, 3.75 * a, 15 * a}, id i = 90) : a = 72 / 17 :=
sorry

end marble_problem_l344_344534


namespace candy_left_in_each_bag_l344_344591

theorem candy_left_in_each_bag (initial_candy : ℕ) (num_bags : ℕ) (candy_taken_out : ℝ) :
  initial_candy = 80 → num_bags = 4 → candy_taken_out = 2.5 → 
  (initial_candy / num_bags - candy_taken_out = 17.5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end candy_left_in_each_bag_l344_344591


namespace largest_rectangle_area_l344_344023

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end largest_rectangle_area_l344_344023


namespace sum_of_sequence_l344_344605

theorem sum_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ n : ℕ, S n = n^2 * a n)
  (h2 : a 1 = 1) : ∀ n : ℕ, S n = 2 * n / (n + 1) :=
begin
  sorry
end

end sum_of_sequence_l344_344605


namespace intersection_ratios_l344_344807

variable {α : Type*} [Field α] [Module α (EuclideanSpace ℝ (Fin 3))] (A B C D E F G : EuclideanSpace ℝ (Fin 3))

def is_parallel (u v : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ k : ℝ, v = k • u

def parallelogram (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop :=
  A - B = D - C ∧ A - D = B - C

-- Given conditions
variable (l : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3))  -- Line l
variable (h_parallelogram : parallelogram A B C D)
variable (h_intersect_AB_E : E ∈ line_through A B ∧ E ∈ l)
variable (h_intersect_AD_F : F ∈ line_through A D ∧ F ∈ l)
variable (h_intersect_AC_G : G ∈ line_through A C ∧ G ∈ l)

theorem intersection_ratios (h_parallelogram : parallelogram A B C D)
    (h_intersect_AB_E : E ∈ line_through A B ∧ E ∈ l)
    (h_intersect_AD_F : F ∈ line_through A D ∧ F ∈ l)
    (h_intersect_AC_G : G ∈ line_through A C ∧ G ∈ l) :
  dist A B / dist A E + dist A D / dist A F = dist A C / dist A G :=
  sorry

end intersection_ratios_l344_344807


namespace apple_tree_total_production_l344_344129

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l344_344129


namespace find_CQ_minus_DQ_l344_344018

noncomputable def solution : ℝ :=
  let Q := (Real.sqrt 5, 0)
  let C := (λ c : ℝ, ((2 * c ^ 2 - 5) / 3, c))
  let D := (λ d : ℝ, ((2 * d ^ 2 - 5) / 3, d))
  Real.abs ((Real.sqrt 6 / Real.sqrt 5) * (Real.abs (c + d))) / (3 * Real.sqrt 5)

theorem find_CQ_minus_DQ
  (line_eq : ∀ (x y : ℝ), y - x * Real.sqrt 5 + 5 = 0)
  (parabola_eq : ∀ (x y : ℝ), 2 * y ^ 2 = 3 * x + 5)
  (Q : ℝ × ℝ)
  (Q_eq : Q = (Real.sqrt 5, 0))
  (C : ℝ → ℝ × ℝ)
  (D : ℝ → ℝ × ℝ)
  (c d : ℝ)
  (C_eq : C c = ((2 * c ^ 2 - 5) / 3, c))
  (D_eq : D d = ((2 * d ^ 2 - 5) / 3, d))
  (c_neg : c < 0)
  (d_pos : d > 0) :
  |(Real.sqrt 6 / Real.sqrt 5) * |c + d|| / (3 * Real.sqrt 5) = 2 * Real.sqrt 6 / 15 := 
sorry

end find_CQ_minus_DQ_l344_344018


namespace average_side_length_of_squares_l344_344002

theorem average_side_length_of_squares (a b c : ℕ) (h₁ : a = 36) (h₂ : b = 64) (h₃ : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
  sorry

end average_side_length_of_squares_l344_344002


namespace friends_with_john_l344_344339

def total_slices (pizzas slices_per_pizza : Nat) : Nat := pizzas * slices_per_pizza

def total_people (total_slices slices_per_person : Nat) : Nat := total_slices / slices_per_person

def number_of_friends (total_people john : Nat) : Nat := total_people - john

theorem friends_with_john (pizzas slices_per_pizza slices_per_person john friends : Nat) (h_pizzas : pizzas = 3) 
                          (h_slices_per_pizza : slices_per_pizza = 8) (h_slices_per_person : slices_per_person = 4)
                          (h_john : john = 1) (h_friends : friends = 5) :
  number_of_friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) john = friends := by
  sorry

end friends_with_john_l344_344339


namespace tank_capacity_l344_344076

theorem tank_capacity (C : ℝ) 
  (h1 : 10 > 0) 
  (h2 : 16 > (10 : ℝ))
  (h3 : ((C/10) - 480 = (C/16))) : C = 1280 := 
by 
  sorry

end tank_capacity_l344_344076


namespace expenditure_on_digging_well_l344_344997

theorem expenditure_on_digging_well :
  let d := 5.0 -- diameter in meters
  let r := d / 2 -- radius in meters
  let h := 20.0 -- depth in meters
  let cost_per_cubic_meter := 25.0 -- cost in Rs/m³
  let pi := Real.pi -- approximate value of π
  let volume := pi * r^2 * h -- volume of the cylinder
  let expenditure := volume * cost_per_cubic_meter -- total cost
  expenditure ≈ 9817 :=
by
  sorry

end expenditure_on_digging_well_l344_344997


namespace no_such_abc_exists_l344_344175

open Polynomial

theorem no_such_abc_exists (a b c : ℝ) : (¬ ∃ (a b c : ℝ) (hnz_a : a ≠ 0) (hnz_b : b ≠ 0) (hnz_c : c ≠ 0), 
  ∀ n : ℤ, n ≥ 4 → ∃ roots : Fin n → ℤ, ∀ i : Fin n, polynomial.eval (roots i) (∑ i in range (n + 1), monomial i (if i = n then 1 else if i = 2 then a else if i = 1 then b else if i = 0 then c else 1)) = 0)) :=
begin
  sorry
end

end no_such_abc_exists_l344_344175


namespace sum_fraction_inequality_l344_344242

theorem sum_fraction_inequality (n : ℕ) (h1 : n ≥ 2) (x : ℕ → ℝ) 
    (h2 : ∑ k in finset.range (n + 1), |x k| = 1) (h3 : ∑ i in finset.range (n + 1), x i = 0) :
    |∑ k in finset.range (n + 1), x k / (k + 1)| ≤ 1 / 2 - 1 / (2 * (n + 1)) := 
by 
    sorry

end sum_fraction_inequality_l344_344242


namespace probability_sum_six_when_two_fair_dice_rolled_l344_344061

def fair_dice_probability_sum_six : Prop :=
  let n := 36 -- Total number of possible outcomes
  let favorable_outcomes := 5 -- Number of outcomes where sum is 6
  let probability := favorable_outcomes.toRat / n.toRat -- Probability computation
  probability = (5 : ℚ) / 36 -- Correct answer

theorem probability_sum_six_when_two_fair_dice_rolled :
  fair_dice_probability_sum_six :=
by
  sorry

end probability_sum_six_when_two_fair_dice_rolled_l344_344061


namespace min_degree_q_l344_344552

-- Define polynomials p, q, and r
variable {R : Type*} [CommRing R] 
variable {p q r : R[X]}

-- Assumptions
-- 1. Polynomial equation 5 * p(x) + 6 * q(x) = r(x)
-- 2. Degree of p(x) is 10
-- 3. Degree of r(x) is 11
theorem min_degree_q (h1 : 5 • p + 6 • q = r)
                     (h2 : p.degree = 10)
                     (h3 : r.degree = 11) : 
                     q.degree ≥ 11 :=
sorry

end min_degree_q_l344_344552


namespace intersect_ellipse_range_longest_chord_line_l344_344628

-- Definitions of the ellipse and the line
def ellipse (x y : ℝ) : Prop := 4*x^2 + y^2 = 1
def line (x y m : ℝ) : Prop := y = x + m

-- Proof 1: The range of m for the line to intersect the ellipse
theorem intersect_ellipse_range (m : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -real.sqrt 5 / 2 ≤ m ∧ m ≤ real.sqrt 5 / 2 :=
sorry

-- Proof 2: The equation of the line where the longest chord cut by the ellipse is located
theorem longest_chord_line : 
  (∀ x y : ℝ, ellipse x y → line x y 0) :=
sorry

end intersect_ellipse_range_longest_chord_line_l344_344628


namespace OF_constant_length_l344_344386

variable {Point : Type}
variable [MetricSpace Point] [ZeroHomClass Point]

-- Definitions for points
variable (A B M P Q O C D E F : Point)

-- Midpoint definitions
noncomputable def is_midpoint (X Y Z : Point) : Prop :=
  dist Z X = dist Z Y

-- Perpendicularity
def is_perpendicular (X Y Z : Point) : Prop :=
  (dist X Y * dist Y Z = 0)

-- Conditions
variable 
  (hP_mid : is_midpoint A M P)
  (hQ_mid : is_midpoint B M Q)
  (hO_mid : is_midpoint P Q O)
  (hC_right : dist C A * dist C B = 0)
  (hMD_perp : is_perpendicular M D C)
  (hME_perp : is_perpendicular M E C)
  (hF_mid : is_midpoint D E F)

-- The goal to prove
theorem OF_constant_length (A B M P Q O C D E F : Point)
  (hP_mid : is_midpoint A M P)
  (hQ_mid : is_midpoint B M Q)
  (hO_mid : is_midpoint P Q O)
  (hC_right : dist C A * dist C B = 0)
  (hMD_perp : is_perpendicular M D C)
  (hME_perp : is_perpendicular M E C)
  (hF_mid : is_midpoint D E F) :
  ∃ k : ℝ, dist O F = k :=
by
  sorry

end OF_constant_length_l344_344386


namespace domain_log_function_l344_344008

theorem domain_log_function : 
  ∀ x : ℝ, (∃ y : ℝ, y = log 2 (x^2 + 2 * x - 3)) ↔ (x < -3 ∨ x > 1) := 
by 
  sorry

end domain_log_function_l344_344008


namespace ratio_of_areas_is_correct_l344_344115

-- Definitions based on the conditions
def side_length : ℝ := 6
def radius : ℝ := side_length / 2
def area_square : ℝ := side_length ^ 2
def area_semicircle : ℝ := (1 / 2) * Real.pi * radius ^ 2
def total_area_semicircles : ℝ := 4 * area_semicircle
def area_new_figure : ℝ := area_square + total_area_semicircles
def ratio_of_areas : ℝ := area_new_figure / area_square

theorem ratio_of_areas_is_correct : ratio_of_areas = 1 + (Real.pi / 2) := by
  sorry

end ratio_of_areas_is_correct_l344_344115


namespace total_grid_rectangles_l344_344276

-- Define the horizontal and vertical rectangle counting functions
def count_horizontal_rects : ℕ :=
  (1 + 2 + 3 + 4 + 5)

def count_vertical_rects : ℕ :=
  (1 + 2 + 3 + 4)

-- Subtract the overcounted intersection and calculate the total
def total_rectangles (horizontal : ℕ) (vertical : ℕ) (overcounted : ℕ) : ℕ :=
  horizontal + vertical - overcounted

-- Main statement
theorem total_grid_rectangles : count_horizontal_rects + count_vertical_rects - 1 = 24 :=
by
  simp [count_horizontal_rects, count_vertical_rects]
  norm_num
  sorry

end total_grid_rectangles_l344_344276


namespace find_income_l344_344905

-- Definitions of percentages used in calculations
def rent_percentage : ℝ := 0.15
def education_percentage : ℝ := 0.15
def misc_percentage : ℝ := 0.10
def medical_percentage : ℝ := 0.15

-- Remaining amount after all expenses
def final_amount : ℝ := 5548

-- Income calculation function
def calc_income (X : ℝ) : ℝ :=
  let after_rent := X * (1 - rent_percentage)
  let after_education := after_rent * (1 - education_percentage)
  let after_misc := after_education * (1 - misc_percentage)
  let after_medical := after_misc * (1 - medical_percentage)
  after_medical

-- Theorem statement to prove the woman's income
theorem find_income (X : ℝ) (h : calc_income X = final_amount) : X = 10038.46 := by
  sorry

end find_income_l344_344905


namespace distance_calculation_l344_344456

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem distance_calculation :
  distance (1, 2, 2) (2, -2, 1) = 3 * real.sqrt 2 :=
by
  sorry

end distance_calculation_l344_344456


namespace max_real_part_w_sum_l344_344725

-- Definitions for the problem setup
noncomputable def z (j : ℕ) : ℂ := 2 * exp (complex.I * (2 * real.pi * j / 16))

-- Define w_j as being either z_j or i * z_j to maximize the real part of the sum
noncomputable def w (j : ℕ) : ℂ :=
if j < 8 then z j else complex.I * z j

-- Define the sum of w_j from 1 to 16
noncomputable def w_sum : ℂ := ∑ j in finset.range 16, w j

-- The maximum possible value of the real part of this sum
theorem max_real_part_w_sum : (w_sum.re : ℝ) = 16 := by
  sorry

end max_real_part_w_sum_l344_344725


namespace cos_2alpha_solution_l344_344596

theorem cos_2alpha_solution (α : ℝ) (h : sin α + cos α = 2 / 3) :
  cos (2 * α) = ± (2 * sqrt 14 / 9) :=
sorry

end cos_2alpha_solution_l344_344596


namespace household_savings_regression_l344_344892

-- Define the problem conditions in Lean
def n := 10
def sum_x := 80
def sum_y := 20
def sum_xy := 184
def sum_x2 := 720

-- Define the averages
def x_bar := sum_x / n
def y_bar := sum_y / n

-- Define the lxx and lxy as per the solution
def lxx := sum_x2 - n * x_bar^2
def lxy := sum_xy - n * x_bar * y_bar

-- Define the regression coefficients
def b_hat := lxy / lxx
def a_hat := y_bar - b_hat * x_bar

-- State the theorem to be proved
theorem household_savings_regression :
  (∀ (x: ℝ), y = b_hat * x + a_hat) :=
by
  sorry -- skip the proof

end household_savings_regression_l344_344892


namespace difference_of_largest_and_smallest_numbers_l344_344537

theorem difference_of_largest_and_smallest_numbers 
  (a b c d e f g h i : ℕ)
  (h_a : a * b * c = 64)
  (h_d : d * e * f = 35)
  (h_g : g * h * i = 81)
  (sum_hundreds : a + d + g = 24)
  (sum_tens : b + e + h = 12)
  (sum_units : c + f + i = 6) : 
  (let arthur := 100 * a + 10 * b + c,
       bob := 100 * d + 10 * e + f,
       carla := 100 * g + 10 * h + i in 
   max (max arthur bob) carla - min (min arthur bob) carla) = 182 := 
sorry

end difference_of_largest_and_smallest_numbers_l344_344537


namespace range_of_a_l344_344677

theorem range_of_a (a : ℝ) : (forall x : ℝ, (a-3) * x > 1 → x < 1 / (a-3)) → a < 3 :=
by
  sorry

end range_of_a_l344_344677


namespace innings_count_l344_344480

theorem innings_count
  (n : ℕ)
  (avg_all : ℝ) (avg_all = 61)
  (diff_high_low : ℝ) (diff_high_low = 150)
  (avg_excl : ℝ) (avg_excl = 58)
  (high_score : ℝ) (high_score = 202)
  (R : ℝ) (R = avg_all * n)
  (low_score : ℝ) (low_score = high_score - diff_high_low)
  (R' : ℝ) (R' = R - (high_score + low_score))
  (R'_eq : R' = avg_excl * (n - 2))
  : n = 46 :=
sorry

end innings_count_l344_344480


namespace railway_tunnel_construction_days_l344_344098

theorem railway_tunnel_construction_days
  (a b t : ℝ)
  (h1 : a = 1/3)
  (h2 : b = 20/100)
  (h3 : t = 4/5 ∨ t = 0.8)
  (total_days : ℝ)
  (h_total_days : total_days = 185)
  : total_days = 180 := 
sorry

end railway_tunnel_construction_days_l344_344098


namespace area_triangle_bcm_is_correct_l344_344178

noncomputable def area_of_triangle_bcm : ℝ :=
  let a := 3 in -- side length of equilateral triangle ABC
  let b := 1.5 in -- derived lengths of segments AM, MC, and BC
  let c := 60 in -- internal angle in degrees, to be converted to radians in the formula
  0.5 * b * b * Real.sin (Real.pi / 3) -- Real.sin (60 degrees) which is π/3 radians

theorem area_triangle_bcm_is_correct :
  area_of_triangle_bcm = 0.5625 * Real.sqrt 3 :=
by
  sorry

end area_triangle_bcm_is_correct_l344_344178


namespace trajectory_passes_incenter_l344_344619

variables {α : Type*} [inner_product_space ℝ α]

structure AffinePoint (P : Type*) :=
(O A B C P : P)
(λ : ℝ)
(h1 : λ ∈ Ioi (0 : ℝ))
(h2 : ∥A - B∥ ≠ 0)
(h3 : ∥A - C∥ ≠ 0)
(h4 : P = O + λ • ((B - A) / ∥B - A∥ + (C - A) / ∥C - A∥))

-- The theorem statement
theorem trajectory_passes_incenter (points : AffinePoint α) : 
  passes_through_incenter points.O points.A points.B points.C points.P :=
sorry

end trajectory_passes_incenter_l344_344619


namespace total_daisies_l344_344565

theorem total_daisies (n : ℕ) 
  (h1 : ∃ k, k = n / 14) 
  (h2 : ∃ m, m = 2 * (n / 14)) 
  (h3 : ∃ p, p = 2 * (2 * (n / 14))) 
  (toto_painted : 7000) 
  (total_painted : (n / 14) + 2 * (n / 14) + 4 * (n / 14) + 7000 = n) : 
  n = 14000 :=
  sorry

end total_daisies_l344_344565


namespace height_difference_in_ordered_pairs_l344_344683

/-
  There are 15 boys and 15 girls. Each pair of boy and girl has a height difference not exceeding 10 cm.
  We want to show that if we pair these boys and girls again in ascending order of their heights, the height difference in the new pairs also does not exceed 10 cm.
-/

theorem height_difference_in_ordered_pairs (b_height g_height : Fin 15 → ℝ) :
  (∀ i : Fin 15, |b_height i - g_height i| ≤ 10) →
  (let b_height_ordered := (Fin 15).sort_by b_height in
   let g_height_ordered := (Fin 15).sort_by g_height in
   ∀ i : Fin 15, |b_height_ordered i - g_height_ordered i| ≤ 10) :=
by
  sorry

end height_difference_in_ordered_pairs_l344_344683


namespace light2011_is_green_l344_344176

def light_pattern : list string := ["green", "yellow", "yellow", "red", "red", "red"]

def color_of_light (n : ℕ) : string :=
  light_pattern[(n - 1) % 6]

theorem light2011_is_green : color_of_light 2011 = "green" :=
  by sorry

end light2011_is_green_l344_344176


namespace geometric_sequence_tangent_identity_l344_344625

theorem geometric_sequence_tangent_identity 
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n+1) / a n = a 2 / a 1)
  (h_cond : a 2 * a 3 * a 4 = - (a 7)^2)
  (h_neg64 : - (a 7)^2 = -64) :
  let x := a 4 * a 6
  in 1 / cos (x/3 * π) = - sqrt 3 := 
by sorry

end geometric_sequence_tangent_identity_l344_344625


namespace second_factorial_l344_344437

-- Define a function to count trailing zeroes in a factorial
def count_trailing_zeroes (n : ℕ) : ℕ := 
  let rec aux (n : ℕ) (p : ℕ) (count : ℕ) : ℕ :=
    if p > n then count
    else aux n (p * 5) (count + n / p)
  aux n 5 0

-- Define factorial (already exists in Mathlib)
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem
theorem second_factorial (n : ℕ) (k : ℕ) (h₀ : n = factorial 70 + factorial k) (h₁ : count_trailing_zeroes n = 16) : 
  k = 4 := by
  sorry

end second_factorial_l344_344437


namespace path_length_is_five_l344_344504

def Point : Type := ℝ × ℝ

def A : Point := (0, 1)
def B : Point := (3, 3)
def C : Point := (3, 0)

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem path_length_is_five : distance A C + distance C B = 5 :=
by sorry

end path_length_is_five_l344_344504


namespace part1_part2_l344_344086

-- Step 1: Define the first part of the problem.
theorem part1 (k : ℕ) (h : k ≥ 2) : 
  (1 / (2^k) + 1 / (2^k + 1) + 1 / (2^k + 2) + ... + 1 / (2^(k + 1) - 1) < 1) :=
sorry

-- Step 2: Define the second part of the problem.
theorem part2 :
  (∀ (k : ℕ), k ≥ 2 → (1 / (2^k) + 1 / (2^k + 1) + 1 / (2^k + 2) + ... + 1 / (2^(k + 1) - 1) < 1)) →
  (∃ (f : ℕ → ℝ), ∀ (n : ℕ), f n = 1 / n ∧ squares_with_side_lengths_1_over_n_can_be_placed_inside_square_length_3_2) :=
sorry

end part1_part2_l344_344086


namespace complete_work_in_4_days_l344_344469

noncomputable def a_speed (b_speed : ℝ) : ℝ := 3 * b_speed
noncomputable def c_speed (b_speed : ℝ) : ℝ := 1.5 * b_speed
def work_done_per_day (speed : ℝ) : ℝ := speed / 12

theorem complete_work_in_4_days (b_speed : ℝ) (h : b_speed > 0) : 
  let a_speed := a_speed b_speed,
      c_speed := c_speed b_speed,
      c_speed_doubled := 2 * c_speed,
      initial_work_done_per_day := work_done_per_day a_speed + work_done_per_day b_speed + work_done_per_day c_speed,
      combined_work_first_4_days := 4 * initial_work_done_per_day,
      new_c_speed := if 4 >= 4 then c_speed_doubled else c_speed,
      final_work_done_per_day := work_done_per_day a_speed + work_done_per_day b_speed + work_done_per_day new_c_speed in
  if combined_work_first_4_days >= 1 then 4
  else sorry = 4 := sorry

end complete_work_in_4_days_l344_344469


namespace divisor_of_99_l344_344021

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ (nat.digits 10 n).reverse) : k ∣ 99 :=
sorry

end divisor_of_99_l344_344021


namespace positive_number_representation_l344_344529

theorem positive_number_representation (a : ℝ) : 
  (a > 0) ↔ (a ≠ 0 ∧ a > 0 ∧ ¬(a < 0)) :=
by 
  sorry

end positive_number_representation_l344_344529


namespace apple_tree_total_production_l344_344130

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l344_344130


namespace probability_of_avg_greater_than_3_l344_344215

/-- Define the set of natural numbers -/
def S : set ℕ := {1, 2, 3, 4, 5}

/-- Define the condition about selecting 3 distinct numbers. -/
def is_3_distinct_sub_set (s : set ℕ) := s ⊆ S ∧ s.card = 3 ∧ s.pairwise (≠)

/-- Define the average of a set of numbers. -/
def average (s : set ℕ) : ℚ := (s.sum / s.card : ℚ)

/-- Define the condition that the average is greater than 3 -/
def avg_greater_than_3 (s : set ℕ) : Prop := average s > 3

/-- The set of all subsets of S such that are 3 distinct elements. -/
def all_subsets : set (set ℕ) := {s | is_3_distinct_sub_set s}

/-- The set of favorable subsets having average greater than 3. -/
def favorable_subsets : set (set ℕ) := {s | is_3_distinct_sub_set s ∧ avg_greater_than_3 s}

/-- The probability calculation statement. -/
def probability := (favorable_subsets.to_finset.card : ℚ) / (all_subsets.to_finset.card : ℚ)

/-- Statement of the problem as Lean 4 theorem. -/
theorem probability_of_avg_greater_than_3 : probability = 2 / 5 := by
  sorry

end probability_of_avg_greater_than_3_l344_344215


namespace eval_expression_l344_344192

theorem eval_expression : (538 * 538) - (537 * 539) = 1 :=
by
  sorry

end eval_expression_l344_344192


namespace bicycle_speed_B_l344_344512

theorem bicycle_speed_B 
  (distance : ℝ := 12)
  (ratio : ℝ := 1.2)
  (time_diff : ℝ := 1 / 6) : 
  ∃ (B_speed : ℝ), B_speed = 12 :=
by
  let A_speed := ratio * B_speed
  have eqn : distance / B_speed - time_diff = distance / A_speed := sorry
  exact ⟨12, sorry⟩

end bicycle_speed_B_l344_344512


namespace transport_budget_percentage_l344_344484

noncomputable def percentage_spent_on_transportation
  (total_angle : ℕ) (salaries_angle : ℕ) (percent_RD percent_utilities percent_equipment percent_supplies : ℕ) : ℕ :=
  let percent_salaries := salaries_angle * 100 / total_angle in
  100 - (percent_RD + percent_utilities + percent_equipment + percent_supplies + percent_salaries)

theorem transport_budget_percentage :
  percentage_spent_on_transportation 360 234 9 5 4 2 = 15 :=
by
  sorry

end transport_budget_percentage_l344_344484


namespace fibonacci_sum_l344_344402

def FibonacciSequence : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := FibonacciSequence (n+1) + FibonacciSequence n

theorem fibonacci_sum : FibonacciSequence 39 + FibonacciSequence 40 = FibonacciSequence 41 :=
by
  sorry

end fibonacci_sum_l344_344402


namespace find_a_for_symmetric_point_on_circle_l344_344244

theorem find_a_for_symmetric_point_on_circle (a : ℝ) :
  ∀ (P : ℝ × ℝ),
    (P.1 ^ 2 + P.2 ^ 2 + 4 * P.1 + a * P.2 - 5 = 0) →
    let P_sym := ((2 * 1 - P.2 + 1) - P.1, P.2) in
    (P_sym.1 ^ 2 + P_sym.2 ^ 2 + 4 * P_sym.1 + a * P_sym.2 - 5 = 0) →
    a = -10 :=
by
  intros P hP hP_sym
  sorry

end find_a_for_symmetric_point_on_circle_l344_344244


namespace nested_sqrt_solution_l344_344186

theorem nested_sqrt_solution : ∃ x : ℝ, x = sqrt (15 + x) ∧ x = (1 + sqrt 61) / 2 :=
by
  sorry

end nested_sqrt_solution_l344_344186


namespace polynomial_remainder_constant_l344_344986

theorem polynomial_remainder_constant (b : ℚ) :
    ∀ (x : ℚ), 
    let p := 15 * x^4 - 6 * x^3 + b * x - 8 in
    let d := 3 * x^2 - 2 * x + 1 in
    let (q, r) := polynomial.div_mod p d in
    degree r <= 0 → b = 4 / 3 :=
begin
    sorry
end

end polynomial_remainder_constant_l344_344986


namespace correct_values_of_a_l344_344563

def last_digit_to_first (n : ℕ) : ℕ :=
  let s := n.digits 10
  (s.ilast::s.take (s.length - 1)).of_digits 10

def first_digit_to_last (n : ℕ) : ℕ :=
  let s := n.digits 10
  (s.drop 1).of_digits 10 * 10 + s.head

def d (a : ℕ) : ℕ :=
  first_digit_to_last ((last_digit_to_first a) ^ 2)

theorem correct_values_of_a (a : ℕ) : d(a) = a^2 ↔ a = 2 ∨ a = 3 := by
  sorry

end correct_values_of_a_l344_344563


namespace fee_calculation_1_fee_calculation_2_average_fee_calculation_l344_344878

def water_fee (x : ℝ) : ℝ :=
  if x ≤ 20 then 1.9 * x
  else 2.8 * x - 18

theorem fee_calculation_1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 20) : water_fee x = 1.9 * x :=
by {
  rw [water_fee],
  simp [h.left, h.right],
  sorry
}

theorem fee_calculation_2 (x : ℝ) (h : x > 20) : water_fee x = 2.8 * x - 18 :=
by {
  rw [water_fee],
  split_ifs,
  { exfalso, linarith },
  { simp [h] },
  sorry
}

theorem average_fee_calculation (x : ℝ) (h : water_fee x / x = 2.3) : x = 36 :=
by {
  have hx : x > 20,
  { 
    sorry
  },
  have : water_fee x = 2.3 * x,
  { 
    sorry
  },
  rw water_fee at this,
  split_ifs at this,
  { exfalso, linarith [hx] },
  { linarith only [this] },
  sorry
}

end fee_calculation_1_fee_calculation_2_average_fee_calculation_l344_344878


namespace simplify_expression_l344_344394

theorem simplify_expression (x : ℤ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 24 = 45 * x + 42 := 
by 
  -- proof steps
  sorry

end simplify_expression_l344_344394


namespace candy_store_spending_l344_344273

noncomputable def weekly_allowance : ℝ := 4.50
noncomputable def arcade_fraction : ℝ := 3/5
noncomputable def toy_store_fraction : ℝ := 1/3

theorem candy_store_spending : 
  let spent_arcade := arcade_fraction * weekly_allowance in
  let remaining_after_arcade := weekly_allowance - spent_arcade in
  let spent_toy_store := toy_store_fraction * remaining_after_arcade in
  let remaining_after_toy_store := remaining_after_arcade - spent_toy_store in
  remaining_after_toy_store = 1.20 :=
by
  sorry

end candy_store_spending_l344_344273


namespace equal_angles_BCG_BCF_l344_344363

theorem equal_angles_BCG_BCF {A B C D E F G : Type} [LinearOrder (Set A)] [euclidean_space ℝ A]
  (triangle_ABC : triangle A B C) (D_on_AC : D ∈ segment A C)
  (BD_eq_DC : |BD| = |DC|)
  (line_FE_parallel_BD : ∃ (l : line), l ∥ line.of_points B D ∧ E ∈ segment B C ∧ F ∈ line.of_points A B)
  (G_intersection_AE_BD : G ∈ (line.of_points A E) ∩ (line.of_points B D)) :
  angle B C G = angle B C F := 
by
  sorry

end equal_angles_BCG_BCF_l344_344363


namespace probability_red_rosebushes_middle_l344_344465

-- Define the total number of arrangements
def total_arrangements := nat.fact 4 / (nat.fact 2 * nat.fact 2)

-- Define the number of favorable arrangements (where 2 red rosebushes are in the middle)
def favorable_arrangements := 1

-- Define the probability that the 2 rosebushes in the middle of the row will be the red rosebushes
def probability_red_in_middle := (favorable_arrangements : ℚ) / total_arrangements

-- State the theorem for the problem
theorem probability_red_rosebushes_middle :
  probability_red_in_middle = 1 / 6 :=
by sorry

end probability_red_rosebushes_middle_l344_344465


namespace count_distinct_even_numbers_l344_344216

theorem count_distinct_even_numbers : 
  ∃ c, c = 37 ∧ ∀ d1 d2 d3, d1 ≠ d2 → d2 ≠ d3 → d1 ≠ d3 → (d1 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d2 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d3 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (∃ n : ℕ, n / 10 ^ 2 = d1 ∧ (n / 10) % 10 = d2 ∧ n % 10 = d3 ∧ n % 2 = 0) :=
sorry

end count_distinct_even_numbers_l344_344216


namespace f_is_odd_f_is_decreasing_inequality_solution_l344_344137

variable (f : ℝ → ℝ)

-- Conditions
axiom additivity (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom negativity (x : ℝ) (hx : x > 0) : f(x) < 0

-- 1. Prove that f is an odd function
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by 
  sorry

-- 2. Prove that f is a decreasing function
theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x2) < f(x1) :=
by 
  sorry

-- 3. Solve the inequality
theorem inequality_solution (b x : ℝ) (hb : b^2 ≠ 2) :
    (1/2) * f(b * x^2) - f(x) > (1/2) * f(b^2 * x) - f(b) ↔
    (if b = 0 then x ∈ Set.Ioi 0 else
        if b < -Real.sqrt 2 then x > 2 / b ∨ x < b else
        if b > -Real.sqrt 2 ∧ b < 0 then x < 2 / b ∨ x > b else
        if b > 0 ∧ b < Real.sqrt 2 then b < x ∧ x < 2 / b else
        b > Real.sqrt 2 ∧ (2 / b < x ∧ x < b)) :=
by 
  sorry

end f_is_odd_f_is_decreasing_inequality_solution_l344_344137


namespace dividend_calculation_l344_344385

theorem dividend_calculation 
  (D : ℝ) (Q : ℕ) (R : ℕ) 
  (hD : D = 164.98876404494382)
  (hQ : Q = 89)
  (hR : R = 14) :
  ⌈D * Q + R⌉ = 14698 :=
sorry

end dividend_calculation_l344_344385


namespace problem1_problem2_l344_344921

-- Problem 1
theorem problem1 : 23 + (-13) + (-17) + 8 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : - (2^3) - (1 + 0.5) / (1/3) * (-3) = 11/2 :=
by
  sorry

end problem1_problem2_l344_344921


namespace polynomial_property_l344_344364

-- Definitions related to the problem statement
variables {k : ℕ} {a : fin (k+1) → ℤ}
variable {P : ℤ → ℤ}
variable {x1 x2 x3 x4 : ℤ}

-- Assume P(x) is a polynomial with integer coefficients and the given conditions
def is_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a : fin (k+1) → ℤ), ∀ x, P x = ∑ i, a i * x^i

def distinct_integers (x1 x2 x3 x4 : ℤ) : Prop :=
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4

def P_conditions (P : ℤ → ℤ) (x1 x2 x3 x4 : ℤ) : Prop :=
  P x1 = 2 ∧ P x2 = 2 ∧ P x3 = 2 ∧ P x4 = 2

-- The statement we need to prove
theorem polynomial_property (P : ℤ→ ℤ) 
  (h_poly : is_polynomial P) (h_distinct : distinct_integers x1 x2 x3 x4) 
  (h_cond : P_conditions P x1 x2 x3 x4) :
  ∀ x : ℤ, P x ∉ {1, 3, 5, 7, 9} :=
begin
  sorry
end


end polynomial_property_l344_344364


namespace second_player_wins_l344_344834

noncomputable def game_conditions :=
∃ (segments : ℕ → (ℚ × ℚ)),
  (∀ n, segments n).fst < (segments n).snd ∧
  (∀ n, (segments (n + 1)).fst ≥ (segments n).fst ∧ (segments (n + 1)).snd ≤ (segments n).snd)

theorem second_player_wins (segments : ℕ → (ℚ × ℚ)) (h : game_conditions segments) :
  ∀ q : ℚ, q ∉ ⋂ n, set.Icc (segments n).fst (segments n).snd :=
by sorry

end second_player_wins_l344_344834


namespace number_of_cats_l344_344949

def wildlife_refuge (total_animals birds mammals dogs cats : ℕ) : Prop :=
  total_animals = 1200 ∧
  birds = mammals + 145 ∧
  cats = dogs + 75 ∧
  mammals = dogs + cats

theorem number_of_cats (D C B M : ℕ) (h : wildlife_refuge 1200 B M D C) :
  C = 301 :=
begin
  sorry
end

end number_of_cats_l344_344949


namespace car_travel_time_l344_344541

-- Definitions
def speed : ℝ := 50
def miles_per_gallon : ℝ := 30
def tank_capacity : ℝ := 15
def fraction_used : ℝ := 0.5555555555555556

-- Theorem statement
theorem car_travel_time : (fraction_used * tank_capacity * miles_per_gallon / speed) = 5 :=
sorry

end car_travel_time_l344_344541


namespace final_number_greater_than_one_l344_344082

theorem final_number_greater_than_one (n : Nat) (initial_value : Nat) (h : initial_value = 2023) (hn : n = 2023) :
  (∃ x : Nat, x > 1 ∧ (∀ (a b : Nat) (hab : a, b ∈ list.repeat initial_value n) (x = (a + b) / 4), x)) := 
sorry

end final_number_greater_than_one_l344_344082


namespace ratio_of_squares_l344_344164

theorem ratio_of_squares : (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3 / 10 := 
sorry

end ratio_of_squares_l344_344164


namespace lines_distance_is_three_tenths_l344_344798

-- Define the two lines as given in the conditions
def line1 (x y : ℝ) := 4*x + 3*y - 1 = 0
def line2 (x y : ℝ) := 8*x + 6*y - 5 = 0

-- Calculate the distance between the two parallel lines
def distance_between_parallel_lines {A B C₁ C₂ : ℝ} (hb : ∥(A, B)∥ ≠ 0)
  (h₁ : ∀ x y, A*x + B*y + C₁ = 0 ↔ line1 x y)
  (h₂ : ∀ x y, A*x + B*y + C₂ = 0 ↔ line2 x y) : ℝ :=
  abs (C₁ - C₂) / ∥(A, B)∥

-- Prove that the distance is 3/10 for the given lines
theorem lines_distance_is_three_tenths :
  distance_between_parallel_lines (λ x y, line1 x y) (λ x y, line2 x y) = 3 / 10 :=
sorry

end lines_distance_is_three_tenths_l344_344798


namespace person_birth_date_l344_344502

theorem person_birth_date
  (x : ℕ)
  (h1 : 1937 - x = x^2 - x)
  (d m : ℕ)
  (h2 : 44 + m = d^2)
  (h3 : 0 < m ∧ m < 13)
  (h4 : d = 7 ∧ m = 5) :
  (x = 44 ∧ 1937 - (x + x^2) = 1892) ∧  d = 7 ∧ m = 5 :=
by
  sorry

end person_birth_date_l344_344502


namespace general_formula_sum_formula_l344_344721

variable (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ)

-- Given conditions
axiom a1_ne_zero : a 1 ≠ 0
axiom condition : ∀ n, n > 0 → 2 * a n - a 1 = S 1 * S n

-- Proof for general formula of {a_n}
theorem general_formula : ∀ n, n > 0 → a n = 2^(n-1) :=
by
  intro n
  sorry

-- Proof for sum of first n terms of {na_n}
theorem sum_formula : T n = 1 + (n-1)2^n :=
by
  intro n
  sorry

end general_formula_sum_formula_l344_344721


namespace part_a_part_b_l344_344887

def bright (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^3

theorem part_a (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ n in at_top, bright (r + n) ∧ bright (s + n) := 
by sorry

theorem part_b (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ m in at_top, bright (r * m) ∧ bright (s * m) := 
by sorry

end part_a_part_b_l344_344887


namespace adults_not_wearing_blue_is_10_l344_344141

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

end adults_not_wearing_blue_is_10_l344_344141


namespace ratio_volumes_l344_344047

variables (V1 V2 : ℝ)
axiom h1 : (3 / 5) * V1 = (2 / 3) * V2

theorem ratio_volumes : V1 / V2 = 10 / 9 := by
  sorry

end ratio_volumes_l344_344047


namespace domain_of_f_l344_344172

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_f :
  { x : ℝ | 2 * x - 1 > 0 } = { x : ℝ | x > 1 / 2 } :=
by
  sorry

end domain_of_f_l344_344172


namespace simplify_expression_l344_344777

theorem simplify_expression (x : ℕ) (h : x = 100) :
  (x + 1) * (x - 1) + x * (2 - x) + (x - 1) ^ 2 = 10000 := by
  sorry

end simplify_expression_l344_344777


namespace problem1_problem2_l344_344922

-- Problem 1: Prove that sqrt(9) + abs(-sqrt(3)) - cbrt(-8) = 5 + sqrt(3)
theorem problem1 : Real.sqrt 9 + Real.abs (-Real.sqrt 3) - Real.cbrt (-8) = 5 + Real.sqrt 3 := 
by 
  sorry

-- Problem 2: Prove that sqrt(2) * (3 - sqrt(2)) + sqrt(2^2) = 3 * sqrt(2)
theorem problem2 : Real.sqrt 2 * (3 - Real.sqrt 2) + Real.sqrt (2^2) = 3 * Real.sqrt 2 := 
by 
  sorry

end problem1_problem2_l344_344922


namespace correlation_function_even_l344_344388

-- Definitions for the stationary process and correlation function
def is_stationary (X : ℝ → ℝ) : Prop := sorry -- definition of a stationary random process
def correlation_function (X : ℝ → ℝ) (k_X : ℝ → ℝ → ℝ) : Prop := 
  ∀ t₁ t₂ : ℝ, k_X t₁ t₂ = k_X (t₁ - t₂)

-- The theorem statement
theorem correlation_function_even (X : ℝ → ℝ) (k_X : ℝ → ℝ → ℝ) 
  (h_stationary : is_stationary X) 
  (h_corr : correlation_function X k_X) : 
  ∀ τ : ℝ, k_X τ = k_X (-τ) := 
by
  sorry

end correlation_function_even_l344_344388


namespace smallest_integral_k_no_real_roots_l344_344057

theorem smallest_integral_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧ 
           (∀ j : ℤ, j < k → (∃ x : ℝ, 2 * x * (j * x - 4) - x^2 + 6 = 0)) ∧
           k = 2 :=
by sorry

end smallest_integral_k_no_real_roots_l344_344057


namespace fraction_to_decimal_l344_344961

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344961


namespace multiple_of_spending_on_wednesday_l344_344217

-- Definitions based on the conditions
def monday_spending : ℤ := 60
def tuesday_spending : ℤ := 4 * monday_spending
def total_spending : ℤ := 600

-- Problem to prove
theorem multiple_of_spending_on_wednesday (x : ℤ) : 
  monday_spending + tuesday_spending + x * monday_spending = total_spending → 
  x = 5 := by
  sorry

end multiple_of_spending_on_wednesday_l344_344217


namespace simplify_sqrt_300_l344_344768

theorem simplify_sqrt_300 :
  ∃ (x : ℝ), sqrt 300 = x * sqrt 3 ∧ x = 10 :=
by
  use 10
  split
  sorry
  rfl

end simplify_sqrt_300_l344_344768


namespace alicia_remaining_sets_l344_344527

def initial_sets : Nat := 600
def guggenheim_donation : Nat := 51
def met_fraction : Rat := 1 / 3
def louvre_fraction : Rat := 1 / 4
def damaged_sets : Nat := 30
def british_fraction : Rat := 40 / 100
def gallery_fraction : Rat := 1 / 8

theorem alicia_remaining_sets : 
  let after_guggenheim := initial_sets - guggenheim_donation in
  let after_met := after_guggenheim - Nat.floor (met_fraction * after_guggenheim) in
  let after_louvre := after_met - Nat.floor (louvre_fraction * after_met) in
  let after_damage := after_louvre - damaged_sets in
  let after_british := after_damage - Nat.floor (british_fraction * after_damage) in
  let after_gallery := after_british - Nat.floor (gallery_fraction * after_british) in
  after_gallery = 129 :=
by
  sorry

end alicia_remaining_sets_l344_344527


namespace number_of_pairs_summing_gt_100_l344_344593

theorem number_of_pairs_summing_gt_100 :
  (∑ a in Finset.range 100, Finset.filter (λ b, a + b > 100 ∧ a ≠ b) (Finset.range 100).card) = 2500 :=
sorry

end number_of_pairs_summing_gt_100_l344_344593


namespace positive_real_solution_count_l344_344942

noncomputable def P (x : ℝ) : ℝ := x^8 + 3*x^7 + 6*x^6 + 2023*x^5 - 2000*x^4

theorem positive_real_solution_count : 
  (∃! x : ℝ, x > 0 ∧ P x = 0) :=
begin
  sorry
end

end positive_real_solution_count_l344_344942


namespace combined_percentage_error_volume_l344_344909

theorem combined_percentage_error_volume
  (L W H : ℝ)
  (L' : ℝ := L * (1 + 0.02))
  (W' : ℝ := W * (1 - 0.03))
  (H' : ℝ := H * (1 + 0.04)) :
  ( (L' * W' * H') / (L * W * H) - 1) * 100 ≈ 3.0744 :=
by
  sorry

end combined_percentage_error_volume_l344_344909


namespace concentric_circles_area_l344_344007

noncomputable theory
open_locale real

theorem concentric_circles_area (r_small r_big : ℝ) (A B P : ℝ) (hAB : A = 60) (hOP : r_small = 30) (hAP : B = 60) :
  r_big = 30 * real.sqrt 5 ∧ π * (r_big ^ 2 - r_small ^ 2) = 3600 * π :=
by
  sorry

end concentric_circles_area_l344_344007


namespace probability_of_selecting_two_girls_l344_344103

def total_students : ℕ := 5
def boys : ℕ := 2
def girls : ℕ := 3
def selected_students : ℕ := 2

theorem probability_of_selecting_two_girls :
  (Nat.choose girls selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ) = 0.3 := by
  sorry

end probability_of_selecting_two_girls_l344_344103


namespace brianna_money_left_l344_344544

-- Let m be the total amount of Brianna's money
-- Let n be the total number of CDs
-- Let c be the cost of one CD
variable (m n c : ℝ)

-- Given conditions
def one_fourth_money (m : ℝ) : ℝ := 1 / 4 * m
def one_fourth_cds (n : ℝ) (c : ℝ) : ℝ := 1 / 4 * n * c

-- Proof problem
theorem brianna_money_left (h : one_fourth_money m = one_fourth_cds n c) :
  m - n * c = 0 :=
by sorry

end brianna_money_left_l344_344544


namespace find_hypotenuse_l344_344793

theorem find_hypotenuse (a b : ℝ) : 
  let c := sqrt (a^2 + b^2 + a * b * sqrt 2) in 
  ∃ c, true := sorry

end find_hypotenuse_l344_344793


namespace largest_in_given_numbers_l344_344853

noncomputable def A := 5.14322
noncomputable def B := 5.1432222222222222222 -- B = 5.143(bar)2
noncomputable def C := 5.1432323232323232323 -- C = 5.14(bar)32
noncomputable def D := 5.1432432432432432432 -- D = 5.1(bar)432
noncomputable def E := 5.1432143214321432143 -- E = 5.(bar)4321

theorem largest_in_given_numbers : D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_in_given_numbers_l344_344853


namespace isosceles_triangle_cosines_l344_344197

theorem isosceles_triangle_cosines 
  (a b c : ℝ)
  (h_isosceles : a = c)
  (α : ℝ)
  (h_triangle_angles : ∠ABC = α)
  (H : ∠BCA = α)
  (O : Type)
  (B : ℝ)
  (D : ℝ)
  (H_orthocenter_bisect : B / 2 = D) :
  cos (α) = sqrt 3 / 3 ∧ cos (∠ABC) = 1 / 3 :=
sorry

end isosceles_triangle_cosines_l344_344197


namespace inequality_a_b_c_l344_344220

noncomputable def a := Real.log (Real.pi / 3)
noncomputable def b := Real.log (Real.exp 1 / 3)
noncomputable def c := Real.exp (0.5)

theorem inequality_a_b_c : c > a ∧ a > b := by
  sorry

end inequality_a_b_c_l344_344220


namespace find_sum_of_six_least_positive_integers_l344_344356

def tau (n : ℕ) : ℕ := (List.range (n + 1)).count_dvd n

theorem find_sum_of_six_least_positive_integers :
  ∃ (a b c d e f : ℕ), 
  (tau a + tau (a + 1) = 8) ∧
  (tau b + tau (b + 1) = 8) ∧
  (tau c + tau (c + 1) = 8) ∧
  (tau d + tau (d + 1) = 8) ∧
  (tau e + tau (e + 1) = 8) ∧
  (tau f + tau (f + 1) = 8) ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
  a + b + c + d + e + f = 1017 :=
by
  sorry

end find_sum_of_six_least_positive_integers_l344_344356


namespace equal_segments_through_midpoint_l344_344599

theorem equal_segments_through_midpoint 
(circle : Type) [Metric.Space circle] [ProperSpace circle]
(chord_AB : Segment circle)
(midpoint_C : Point circle)
(chord_KL : Segment circle)
(chord_MN : Segment circle)
(point_K point_L point_M point_N point_P point_Q : Point circle)
(intersect_KN_AB : KN.Intersects chord_AB point_P)
(intersect_LM_AB : LM.Intersects chord_AB point_Q)
(midpoint_C_is_midpoint : chord_AB.Midpoint = midpoint_C)
(through_C_KL : chord_KL.PassesThrough midpoint_C)
(through_C_MN : chord_MN.PassesThrough midpoint_C)
(on_same_side_K_M : chord_AB.OnSameSide point_K point_M)
(on_same_side_L_N : chord_AB.OnOppositeSide point_L point_N) :
  dist point_P midpoint_C = dist point_Q midpoint_C := by
  sorry

end equal_segments_through_midpoint_l344_344599


namespace alexander_money_l344_344907

theorem alexander_money (a : ℕ) (n : ℕ) : 
  (∀ k : ℕ, 0 ≤ k ≤ 2012 → 
  ∃ m : ℕ, ∃ x_k : ℕ, 
  x_k = (a / 3 ^ k) / 6 ∧ 
  m = a - x_k ∧ 
  m ∈ ℕ) → 
  a = 4 * 3 ^ 2012 := sorry

end alexander_money_l344_344907


namespace simplify_sqrt_300_l344_344769

theorem simplify_sqrt_300 :
  ∃ (x : ℝ), sqrt 300 = x * sqrt 3 ∧ x = 10 :=
by
  use 10
  split
  sorry
  rfl

end simplify_sqrt_300_l344_344769


namespace P_cap_Q_cardinality_l344_344637

/-- Define the set P as pairs (x, y) with y = 2x^2 + 3x + 1 and x between -2 and 3. -/
def P : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ y = 2 * x^2 + 3 * x + 1 ∧ -2 ≤ x ∧ x ≤ 3}

/-- Define the set Q as pairs (x, y) with x = a is any real number and y is any real number. -/
def Q : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ a y : ℝ, p = (a, y) }

/-- Prove that the number of elements in the set intersection of P and Q is 0 or 1. -/
theorem P_cap_Q_cardinality :
  ∀ a : ℝ, (a ∈ Icc (-2 : ℝ) (3 : ℝ)) → Finset.card (Finset.filter (λ p : ℝ × ℝ, p ∈ P ∧ p ∈ Q) (Finset.univ : Finset (ℝ × ℝ))) ≤ 1 :=
by
  sorry

end P_cap_Q_cardinality_l344_344637


namespace gcd_455_299_eq_13_l344_344417

theorem gcd_455_299_eq_13 : Nat.gcd 455 299 = 13 := by
  sorry

end gcd_455_299_eq_13_l344_344417


namespace decreased_value_l344_344696

noncomputable def original_expression (x y: ℝ) : ℝ :=
  x * y^2

noncomputable def decreased_expression (x y: ℝ) : ℝ :=
  (1 / 2) * x * (1 / 2 * y) ^ 2

theorem decreased_value (x y: ℝ) :
  decreased_expression x y = (1 / 8) * original_expression x y :=
by
  sorry

end decreased_value_l344_344696


namespace min_distance_highest_lowest_points_l344_344014

-- Define the original function
def original_function (x : ℝ) : ℝ := sin (2 * x)

-- Define the translation of the function by π/6 units to the left
def translated_function (x : ℝ) : ℝ :=
  original_function (x + π / 6)

-- Define the final stretched function
def stretched_function (x : ℝ) : ℝ :=
  translated_function (x / 2)

-- Define the function f(x)
def f (x : ℝ) : ℝ := stretched_function x

-- State the theorem
theorem min_distance_highest_lowest_points :
  let max_value := 1
  let min_value := -1
  -- The formula for the distance
  let distance := max_value - min_value
  -- The minimum value of this distance is sqrt(π^2 + 4)
  sqrt (π^2 + 4) = distance := 
  sorry

end min_distance_highest_lowest_points_l344_344014


namespace binomial_19_11_l344_344157

theorem binomial_19_11 :
  (nat.choose 19 11) = 85306 :=
begin
  -- Given conditions
  have h1 : (nat.choose 17 10) = 24310,
  { sorry },
  have h2 : (nat.choose 17  8) = 24310,
  { sorry },
  -- Result from conditions and calculation
  sorry
end

end binomial_19_11_l344_344157


namespace triangle_conditions_l344_344299

variable {α : Type} [Real α]
open Real

theorem triangle_conditions (A B C a b c : α) (h1 : a - b = 1)
    (h2 : 2 * cos ((A + B) / 2) ^ 2 - cos (2 * C) = 1)
    (h3 : 3 * sin B = 2 * sin A)
    (h4 : C = π / 3) :
    c / b = sqrt 7 / 2 :=
sorry

end triangle_conditions_l344_344299


namespace count_rectangles_in_grid_l344_344281

theorem count_rectangles_in_grid :
  let horizontal_strip := 1,
      vertical_strip := 1,
      horizontal_rects := 1 + 2 + 3 + 4 + 5,
      vertical_rects := 1 + 2 + 3 + 4,
      double_counted := 1
  in horizontal_rects + vertical_rects - double_counted = 24 :=
by
  -- Definitions based on conditions
  let horizontal_strip := 1
  let vertical_strip := 1
  let horizontal_rects := 1 + 2 + 3 + 4 + 5
  let vertical_rects := 1 + 2 + 3 + 4
  let double_counted := 1

  -- Assertion of equality based on problem solution
  have h : horizontal_rects + vertical_rects - double_counted = 24 :=
    calc
      horizontal_rects + vertical_rects - double_counted
      = (1 + 2 + 3 + 4 + 5) + (1 + 2 + 3 + 4) - 1 : by rfl
      = 15 + 10 - 1 : by rfl
      = 24 : by rfl

  exact h

end count_rectangles_in_grid_l344_344281


namespace calculation_result_l344_344848

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end calculation_result_l344_344848


namespace shaded_area_fraction_l344_344409

-- Define the side length of the square and points as given conditions
def side_length : ℝ := 1
def midpoint (x y : ℝ) : ℝ := (x + y) / 2

-- Define the vertices of the square WXYZ
def W : ℝ × ℝ := (0, 0)
def X : ℝ × ℝ := (1, 0)
def Y : ℝ × ℝ := (1, 1)
def Z : ℝ × ℝ := (0, 1)

-- Define the midpoints P, Q, R
def P : ℝ × ℝ := midpoint <$> Z <*> W
def Q : ℝ × ℝ := midpoint <$> X <*> Y
def R : ℝ × ℝ := midpoint <$> Y <*> Z

-- Define the midpoint U and intersection point V (assuming V is known)
def U : ℝ × ℝ := midpoint <$> W <*> X
-- V needs to be determined dynamically, for simplicity let's just define it as a variable here
variable V : ℝ × ℝ
  
-- Function to calculate the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2|

theorem shaded_area_fraction :
  triangle_area W X R - triangle_area W X V = (side_length ^ 2) * (3 / 8) := 
by
  sorry

end shaded_area_fraction_l344_344409


namespace sum_of_angles_l344_344612

theorem sum_of_angles (a β : ℝ) (h₁ : tan a = 2) (h₂ : tan β = 3) (h₃ : 0 < a ∧ a < π / 2) (h₄ : 0 < β ∧ β < π / 2) :
  a + β = 3 * π / 4 :=
sorry

end sum_of_angles_l344_344612


namespace area_of_EFGH_is_correct_l344_344310

-- Define the convex quadrilateral EFGH with given side lengths and angle
structure Quadrilateral (EF FG GH HE : ℝ) (angleEGH : ℝ) :=
  (EF_pos : EF > 0)
  (FG_pos : FG > 0)
  (GH_pos : GH > 0)
  (HE_pos : HE > 0)
  (angleEGH_valid : 0 < angleEGH ∧ angleEGH < 180)

-- Instantiate the quadrilateral with the given problem conditions
def EFGH : Quadrilateral 10 5 8 8 120 :=
{ EF_pos := by norm_num,
  FG_pos := by norm_num,
  GH_pos := by norm_num,
  HE_pos := by norm_num,
  angleEGH_valid := by norm_num }

-- Prove that the area of the quadrilateral EFGH is 16√3 + 25
theorem area_of_EFGH_is_correct : 
  area_of_quad EFGH = 16 * Real.sqrt 3 + 25 :=
sorry

end area_of_EFGH_is_correct_l344_344310


namespace trigonometric_identity_l344_344560

noncomputable def trig_expr (θ φ : ℝ) : ℝ :=
  (Real.cos θ)^2 + (Real.cos φ)^2 + Real.cos θ * Real.cos φ

theorem trigonometric_identity :
  trig_expr (75 * Real.pi / 180) (15 * Real.pi / 180) = 5 / 4 :=
  sorry

end trigonometric_identity_l344_344560


namespace sin_angle_HAE_proof_l344_344083

noncomputable def sin_angle_HAE (A B C D E F G H : ℝ × ℝ × ℝ) (AB CD EF GH AE BF CG DH AC EG : ℝ) : ℝ :=
if h : AB = 1 ∧ CD = 1 ∧ EF = 1 ∧ GH = 1 ∧ AE = 2 ∧ BF = 2√2 ∧ CG = 2 ∧ DH = 2√2 ∧ AC = 2 ∧ EG = 2 then
  let HA := 2
  let AE := 2
  let HE := sqrt ((H.1 - E.1)^2 + (H.2 - E.2)^2 + (H.3 - E.3)^2)
  let cos_angle_HAE := (HA^2 + AE^2 - HE^2) / (2 * HA * AE) in
  sqrt (1 - cos_angle_HAE^2)
else 0

theorem sin_angle_HAE_proof (A B C D E F G H : ℝ × ℝ × ℝ) (AB CD EF GH AE BF CG DH AC EG : ℝ) 
  (h : AB = 1 ∧ CD = 1 ∧ EF = 1 ∧ GH = 1 ∧ AE = 2 ∧ BF = 2√2 ∧ CG = 2 ∧ DH = 2√2 ∧ AC = 2 ∧ EG = 2) :
  sin_angle_HAE A B C D E F G H 1 1 1 1 2 (2:ℝ) 2 (2:ℝ) 2 2 = have : sqrt 15 / 8 :=
  sorry

end sin_angle_HAE_proof_l344_344083


namespace men_entered_room_l344_344710

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l344_344710


namespace possible_multisets_l344_344928

theorem possible_multisets (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℤ) :
  (∀ s : ℤ, (b_6 * s ^ 6 + b_5 * s ^ 5 + b_4 * s ^ 4 + b_3 * s ^ 3 + b_2 * s ^ 2 + b_1 * s + b_0 = 0) ↔
             (b_0 * s ^ 6 + b_1 * s ^ 5 + b_2 * s ^ 4 + b_3 * s ^ 3 + b_4 * s ^ 2 + b_5 * s + b_6 = 0)) →
  ∃ T : finset (multiset ℤ), T.card = 7 :=
by
  sorry

end possible_multisets_l344_344928


namespace work_completion_days_l344_344885

theorem work_completion_days :
  ∀ (rate_A rate_B : ℝ),
  rate_A = 2 * rate_B ∧ rate_B = 1 / 24 →
  1 / (rate_A + rate_B) = 8 :=
by
  intros rate_A rate_B
  intro h
  cases h with h1 h2
  -- skipping the proof steps
  sorry

end work_completion_days_l344_344885


namespace value_of_a_l344_344589

noncomputable def a_solution (z : ℂ) : ℝ :=
if H : z = (λ a, a * complex.I / (1 + 2 * complex.I)) a ∧ |z| = real.sqrt 5 ∧ a < 0 
then -5 else 0

theorem value_of_a (a : ℝ) : z = a * complex.I / (1 + 2 * complex.I) → |z| = real.sqrt 5 → a < 0 → a = -5 :=
by sorry

end value_of_a_l344_344589


namespace last_digit_2_pow_2023_l344_344752

-- Define the cycle of last digits for powers of 2
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Define a function to get the last digit of 2^n based on the cycle
def last_digit (n : ℕ) : ℕ :=
  last_digit_cycle[(n % 4 : ℕ)]

-- The theorem that states the last digit of 2^2023 is 8
theorem last_digit_2_pow_2023 : last_digit 2023 = 8 :=
  sorry

end last_digit_2_pow_2023_l344_344752


namespace hundred_as_sum_of_fib_l344_344816

/-- Define the Fibonacci sequence -/
def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

/-- Definition of a positive integer n expressed as a sum of distinct Fibonacci numbers -/
def is_sum_of_distinct_fib (n : ℕ) : Prop :=
  ∃ S : finset ℕ, (∀ m ∈ S, ∃ k, fibonacci k = m) ∧ S.sum id = n

/-- Main theorem that verifies 100 can be expressed as a sum of several distinct Fibonacci numbers -/
theorem hundred_as_sum_of_fib : is_sum_of_distinct_fib 100 := sorry

end hundred_as_sum_of_fib_l344_344816


namespace rectangular_room_area_l344_344519

-- Define the given conditions
def square_area : ℕ := 169
def cut_length : ℕ := 2

-- Define the proof problem
theorem rectangular_room_area : 
  let side_length := Int.sqrt square_area in
  let new_length := (side_length - cut_length) in
  let new_width := side_length in
  new_length * new_width = 143 :=
by
  sorry

end rectangular_room_area_l344_344519


namespace probability_correct_l344_344120

-- Define the set of numbers from 1 to 120
def numbers : set ℕ := { n | 0 < n ∧ n ≤ 120 }

-- Define function that checks if a number is a multiple of 2, 3, 5, or 7
def is_multiple_2_3_5_7 (n : ℕ) : Prop :=
  n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0

-- Count the number of desired multiples
def count_multiples : ℕ := set.count (λ n, is_multiple_2_3_5_7 n) numbers

-- Total number of cards
def total_cards : ℕ := set.count id numbers

-- The probability as a fraction
def probability : ℚ := count_multiples / total_cards

-- Prove the probability equals 33/40
theorem probability_correct : probability = 33 / 40 := by 
  sorry

end probability_correct_l344_344120


namespace find_AM_length_l344_344321

structure Square :=
  (A B C D : Point)
  (AB_len : ℝ)
  (AB_eq_BC : A.distance B = B.distance C)
  (BC_eq_CD : B.distance C = C.distance D)
  (CD_eq_DA : C.distance D = D.distance A)
  (DA_eq_AB : D.distance A = A.distance B)

def area_of_shaded_region (M : Point) (square : Square) : ℝ := sorry

theorem find_AM_length (A B C D M : Point) (square : Square) 
  (AB_len_20 : square.AB_len = 20)
  (shaded_area_40 : area_of_shaded_region M square = 40) :
  A.distance M = 5 :=
by
  sorry

end find_AM_length_l344_344321


namespace merchant_profit_percentage_l344_344071

-- Define the initial conditions
def cost_price : ℝ := 100
def markup_percentage : ℝ := 0.30
def discount_percentage : ℝ := 0.10

-- Define the calculated values from the solution steps
def marked_price : ℝ := cost_price * (1 + markup_percentage)
def discount_amount : ℝ := marked_price * discount_percentage
def selling_price : ℝ := marked_price - discount_amount
def profit : ℝ := selling_price - cost_price
def profit_percentage : ℝ := (profit / cost_price) * 100

-- The theorem that we need to prove
theorem merchant_profit_percentage :
  profit_percentage = 17 :=
by
  sorry

end merchant_profit_percentage_l344_344071


namespace average_income_P_Q_l344_344405

   variable (P Q R : ℝ)

   theorem average_income_P_Q
     (h1 : (Q + R) / 2 = 6250)
     (h2 : (P + R) / 2 = 5200)
     (h3 : P = 4000) :
     (P + Q) / 2 = 5050 := by
   sorry
   
end average_income_P_Q_l344_344405


namespace favorable_probability_correct_l344_344547

open Finset

-- Define the set of positive integers from 1 to 60
def range_set : Finset ℕ := Finset.range 60 \ {0}

-- Define the probability calculation
def favorable_pairs_count : ℕ :=
Finset.card { x : ℕ × ℕ // (x.1 ∈ (range_set ∖ {y | (y % 6 == 5)})) ∧ (x.2 ∈ (range_set ∖ {y | (y % 6 == 5)})) ∧ x.1 ≠ x.2 }

def total_pairs_count : ℕ :=
Finset.card (range_set.product range_set \ {(x, y) | x = y})

def favorable_probability : ℚ := favorable_pairs_count / total_pairs_count

-- The main statement
theorem favorable_probability_correct : favorable_probability = 91 / 295 := by
  sorry

end favorable_probability_correct_l344_344547


namespace problem_l344_344170

def a : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 4 * a (n+1) - a n

def b : ℕ → ℤ
| 0     := 1
| 1     := 2
| (n+2) := 4 * b (n+1) - b n

theorem problem (n : ℕ) : b n ^ 2 = 3 * (a n) ^ 2 + 1 := 
sorry

end problem_l344_344170


namespace part_i_part_ii_l344_344413

variable {b c : ℤ}

theorem part_i (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p ≠ q ∧ 2 * b ^ 2 = p ^ 2 + q ^ 2 :=
sorry

theorem part_ii (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (r s : ℤ), r > 0 ∧ s > 0 ∧ r ≠ s ∧ b ^ 2 = r ^ 2 + s ^ 2 :=
sorry

end part_i_part_ii_l344_344413


namespace non_intersecting_square_exists_l344_344308

theorem non_intersecting_square_exists :
  ∃ (w : ℤ × ℤ), (w.1 >= 0 ∧ w.1 < 8) ∧ (w.2 >= 0 ∧ w.2 < 8) ∧ 
  (∀ (b : (ℤ × ℤ) × (ℤ × ℤ)), b ∈ set_of black_squares → ¬ intersects w b) :=
sorry

end non_intersecting_square_exists_l344_344308


namespace cos_tan_quadrant_l344_344644

theorem cos_tan_quadrant (α : ℝ) 
  (hcos : Real.cos α < 0) 
  (htan : Real.tan α > 0) : 
  (2 * π / 2 < α ∧ α < π) :=
by
  sorry

end cos_tan_quadrant_l344_344644


namespace max_intersections_l344_344371

-- Define the conditions and the question to be proved
variable (P1 P2 : Type)
variable [convex_polygon P1 P2]
variable (n1 n2 : ℕ)
variable (inside_P2 : P1 ⊆ P2)
variable (n1_lt_n2 : n1 < n2)
variable (vertices_do_not_touch : ∀ vertex ∈ vertices(P1), vertex ∉ vertices(P2))

-- Define the expected result as a theorem
theorem max_intersections (n1 : ℕ) (n2 : ℕ) (P1 P2 : Type) [convex_polygon P1 P2] 
  (inside_P2 : P1 ⊆ P2) (n1_lt_n2 : n1 < n2) (vertices_do_not_touch : ∀ vertex ∈ vertices(P1), vertex ∉ vertices(P2)) :
  max_intersections(P1, P2) = 2 * n1 :=
sorry

end max_intersections_l344_344371


namespace samantha_annual_income_l344_344680

-- Define the constants
variables (q : ℝ) 

-- Define the income and tax equations
def incomeTax (Y : ℝ) : ℝ :=
  if Y ≤ 30000 then 0.01 * q * Y
  else 0.01 * q * 30000 + 0.01 * (q + 1) * (Y - 30000)

noncomputable def annualIncome : ℝ :=
  let Y := 60000 in incomeTax q Y

-- The theorem statement
theorem samantha_annual_income (q : ℝ) :
  ∃ Y, incomeTax q Y = 0.01 * (q + 0.5) * Y ∧ Y = 60000 :=
by
  use 60000
  sorry

end samantha_annual_income_l344_344680


namespace two_dice_probability_l344_344288

noncomputable def probability_same_side (p_maroon p_teal p_cyan p_sparkly : ℚ) : ℚ :=
  (p_maroon^2 + p_teal^2 + p_cyan^2 + p_sparkly^2)

theorem two_dice_probability :
  let p_maroon := 3 / 12
  let p_teal := 4 / 12
  let p_cyan := 4 / 12
  let p_sparkly := 1 / 12 in
  probability_same_side p_maroon p_teal p_cyan p_sparkly = (7 / 24) := 
by 
  sorry

end two_dice_probability_l344_344288


namespace smallest_positive_period_pie_l344_344908

theorem smallest_positive_period_pie 
    (hA : ∀ x, |sin x + π| = |sin x|) 
    (hB : ∀ x, tan (2 * x + π / 2) = tan (2 * x)) 
    (hC : ∀ x, cos (x / 2 + 2 * π) = cos (x / 2)) 
    (hD : ∀ x, sin (x + 2 * π) = sin x) : 
    (∃ T > 0, T = π ∧ ∀ x, |sin x + T| = |sin x|) ∧
    ¬ (∃ T > 0, T = π ∧ ∀ x, tan (2 * x + T) = tan (2 * x)) ∧
    ¬ (∃ T > 0, T = π ∧ ∀ x, cos (x / 2 + T) = cos (x / 2)) ∧
    ¬ (∃ T > 0, T = π ∧ ∀ x, sin (x + T) = sin x) := 
by
    sorry

end smallest_positive_period_pie_l344_344908


namespace tangent_line_to_parabola_l344_344414

theorem tangent_line_to_parabola :
  (∀ (x y : ℝ), y = x^2 → x = -1 → y = 1 → 2 * x + y + 1 = 0) :=
by
  intro x y parabola eq_x eq_y
  sorry

end tangent_line_to_parabola_l344_344414


namespace adult_tickets_sold_l344_344095

theorem adult_tickets_sold (A C : ℕ) (h1 : A + C = 85) (h2 : 5 * A + 2 * C = 275) : A = 35 := by
  sorry

end adult_tickets_sold_l344_344095


namespace min_value_at_2_l344_344633

noncomputable def f (x : ℝ) : ℝ := (2 / (x^2)) + Real.log x

theorem min_value_at_2 : (∀ x ∈ Set.Ioi (0 : ℝ), f x ≥ f 2) ∧ (∃ x ∈ Set.Ioi (0 : ℝ), f x = f 2) :=
by
  sorry

end min_value_at_2_l344_344633


namespace new_plan_cost_correct_l344_344740

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end new_plan_cost_correct_l344_344740


namespace probability_two_white_balls_l344_344478

/-- A bag contains 5 balls of the same size, including 2 black balls and 3 white balls. 
    Now, 2 balls are randomly drawn from the bag. The probability that both balls 
    drawn are white is 3/10. -/
theorem probability_two_white_balls :
  let total_balls := 5,
      white_balls := 3,
      draw_count := 2 in
  (binomial white_balls draw_count : ℚ) / (binomial total_balls draw_count : ℚ) = 3 / 10 :=
by
  sorry

end probability_two_white_balls_l344_344478


namespace algebraic_expression_value_zero_l344_344646

theorem algebraic_expression_value_zero (a b : ℝ) (h : a - b = 2) : (a^3 - 2 * a^2 * b + a * b^2 - 4 * a = 0) :=
sorry

end algebraic_expression_value_zero_l344_344646


namespace sequence_an_l344_344232

theorem sequence_an (a b S T : ℕ → ℝ) (n: ℕ) (h1 : ∀ n, 4 * n * S n = (n + 1) ^ 2 * a n) (h2 : a 1 = 1) 
(h3 : ∀ n, b n = n / (a n)) (h4 : ∀ n, T n = ∑ i in range (n + 1), b i) 
(ha : ∀ n, a n = n^3): 
∃ an: ℕ → ℝ, (∀ n, a n = an n) ∧ (∀ n, T n < 7 / 4) := by 
  sorry

end sequence_an_l344_344232


namespace product_with_3_divisors_is_square_l344_344726

theorem product_with_3_divisors_is_square :
  let N := ∏ p in {p^2 | p : ℕ, p.prime ∧ p^2 ≤ 100}, p
  in N = 210^2 := by
  sorry

end product_with_3_divisors_is_square_l344_344726


namespace table_covered_three_layers_l344_344439

theorem table_covered_three_layers (runners_areas_1_2_3 : ℝ) (runners_areas_4_5 : ℝ)
  (percent_table_covered : ℝ) (table_area : ℝ)
  (exactly_two_layers_area : ℝ) (exactly_one_layer_area : ℝ)
  (h1: runners_areas_1_2_3 = 324)
  (h2: runners_areas_4_5 = 216)
  (h3: percent_table_covered = 0.75)
  (h4: table_area = 320)
  (h5: exactly_two_layers_area = 36)
  (h6: exactly_one_layer_area = 48) : 
  let total_covered_area := percent_table_covered * table_area in
  let exactly_three_layers_area := total_covered_area - (exactly_two_layers_area + exactly_one_layer_area) in
  exactly_three_layers_area = 156 :=
by
  sorry

end table_covered_three_layers_l344_344439


namespace max_rectangle_area_l344_344025

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end max_rectangle_area_l344_344025


namespace area_of_side_face_l344_344553

theorem area_of_side_face (l w h : ℝ)
  (h_front_top : w * h = 0.5 * (l * h))
  (h_top_side : l * h = 1.5 * (w * h))
  (h_volume : l * w * h = 3000) :
  w * h = 200 := 
sorry

end area_of_side_face_l344_344553


namespace circumcenter_PQH_on_median_l344_344729

-- Define the acute-angled triangle ABC
variables {A B C P Q H O : Type} [geometry : geometry.angle_props ABC]

-- Given conditions:
axiom acute_angled_triangle : geometry.acute_angle A B C
axiom circumcenter_O : geometry.is_circumcenter O A B C
axiom orthocenter_H : geometry.is_orthocenter H A B C
axiom points_PQ_on_altitudes : geometry.points_on_altitudes P Q A B C O

-- The theorem to show:
theorem circumcenter_PQH_on_median :
  geometry.center_of_circumcircle_on_median PQH ABC :=
sorry

end circumcenter_PQH_on_median_l344_344729


namespace find_M_plus_N_l344_344011

    noncomputable def f (x : ℝ) : ℝ := 
      Real.log (Real.sqrt (x ^ 2 + 1) - x) + (3 * Real.exp x + 1) / (Real.exp x + 1)
    
    theorem find_M_plus_N : 
      let M := Sup (set.image f (set.Icc (-2 : ℝ) 2))
      let N := Inf (set.image f (set.Icc (-2 : ℝ) 2))
      M + N = 4 :=
    by
      sorry
    
end find_M_plus_N_l344_344011


namespace exists_primitive_root_mod_2p_alpha_l344_344366

theorem exists_primitive_root_mod_2p_alpha (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (α : ℕ) :
  ∃ x : ℕ, Nat.PrimitiveRoot x (2 * p^α) := 
sorry

end exists_primitive_root_mod_2p_alpha_l344_344366


namespace number_of_solutions_in_interval_l344_344940

def g (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

theorem number_of_solutions_in_interval :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 4 * g 6 x - 3 * g 8 x = g 2 x}.finite.toFinset.card = 5 :=
by 
  sorry

end number_of_solutions_in_interval_l344_344940


namespace range_of_a_l344_344634

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) ↔ a < 1004 := 
sorry

end range_of_a_l344_344634


namespace intersection_on_line_AA1_l344_344759

open Classical

variables {A A1 O P P1 Q Q1 : Point}
variables {circle : Circle}

-- Define symmetry and similar rays
def symmetric (A A1 O : Point) : Prop := dist O A = dist O A1 ∧ ∃ M, midpoint M A A1 ∧ M = O
def similar_rays (A P A1 P1 : Point) : Prop := direction A P = direction A1 P1

-- Define conditions
axiom A1_symmetric : symmetric A A1 O
axiom A_in_circle : circle.contains A
axiom A1_in_circle : circle.contains A1
axiom similar_rays_AP_A1P1 : similar_rays A P A1 P1
axiom similar_rays_AQ_A1Q1 : similar_rays A Q A1 Q1
axiom P_on_circle : circle.contains P
axiom P1_on_circle : circle.contains P1
axiom Q_on_circle : circle.contains Q
axiom Q1_on_circle : circle.contains Q1

-- Prove the intersection of P1Q and PQ1 lies on the line AA1
theorem intersection_on_line_AA1 : 
  ∃ R : Point, (line_through P1 Q).contains R ∧ (line_through P Q1).contains R ∧ (line_through A A1).contains R :=
sorry

end intersection_on_line_AA1_l344_344759


namespace average_side_length_of_squares_l344_344001

theorem average_side_length_of_squares (a b c : ℕ) (h₁ : a = 36) (h₂ : b = 64) (h₃ : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
  sorry

end average_side_length_of_squares_l344_344001


namespace math_problem_l344_344918

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end math_problem_l344_344918


namespace partA_l344_344073

theorem partA (n : ℕ) : 
  1 < (n + 1 / 2) * Real.log (1 + 1 / n) ∧ (n + 1 / 2) * Real.log (1 + 1 / n) < 1 + 1 / (12 * n * (n + 1)) := 
sorry

end partA_l344_344073


namespace arrangements_with_A_arrangements_with_A_and_B_not_japan_l344_344592

-- Define the necessary elements from the conditions
def volunteers : ℕ := 6
def selected : ℕ := 4
def pavilions : ℕ := 4
def germany : ℕ := 1
def japan : ℕ := 1
def italy : ℕ := 1
def sweden : ℕ := 1
def arrangements_if_A_must_go : ℕ := 240
def arrangements_if_A_and_B_not_japan : ℕ := 240

-- Lean 4 statement for part (1)
theorem arrangements_with_A :
  (person_A_included : True) → arrangements_if_A_must_go = 240 :=
begin
  sorry
end

-- Lean 4 statement for part (2)
theorem arrangements_with_A_and_B_not_japan :
  (person_A_not_in_japan : True) → (person_B_not_in_japan : True) → arrangements_if_A_and_B_not_japan = 240 :=
begin
  sorry
end

end arrangements_with_A_arrangements_with_A_and_B_not_japan_l344_344592


namespace cylinder_volume_l344_344951

open Real

theorem cylinder_volume (d h : ℝ) (π : ℝ) (r : ℝ) : 
  d = 20 → h = 10 → r = d / 2 → π = real.pi → 
  ∃ V : ℝ, V = π * r^2 * h ∧ V = 1000 * π :=
by
  sorry

end cylinder_volume_l344_344951


namespace fraction_to_decimal_l344_344981

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344981


namespace new_plan_cost_correct_l344_344742

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end new_plan_cost_correct_l344_344742


namespace checkerboard_black_squares_l344_344151

/-- The total number of black squares on a 29x29 checkerboard that starts with a black square
    and alternates colors along each row and column is 421. -/
theorem checkerboard_black_squares :
  let n := 29 in
  let is_black (i j : Nat) := (i + j) % 2 = 0 in
  let black_squares := (Finset.range n).sum (λ i => (Finset.range n).count (λ j => is_black i j)) in
  black_squares = 421 :=
by
  -- The proof goes here
  sorry

end checkerboard_black_squares_l344_344151


namespace aiden_nap_is_15_minutes_l344_344524

def aiden_nap_duration_in_minutes (nap_in_hours : ℚ) (minutes_per_hour : ℕ) : ℚ :=
  nap_in_hours * minutes_per_hour

theorem aiden_nap_is_15_minutes :
  aiden_nap_duration_in_minutes (1/4) 60 = 15 := by
  sorry

end aiden_nap_is_15_minutes_l344_344524


namespace bob_salary_april_l344_344914

variable (initial_salary : ℕ) (feb_raise_rate : ℚ) (mar_cut_rate : ℚ) (apr_bonus : ℕ)

def feb_salary := initial_salary * (1 + feb_raise_rate)
def mar_salary := feb_salary * (1 - mar_cut_rate)
def apr_salary := mar_salary + apr_bonus

theorem bob_salary_april :
  initial_salary = 3000 →
  feb_raise_rate = 0.15 →
  mar_cut_rate = 0.10 →
  apr_bonus = 500 →
  apr_salary initial_salary feb_raise_rate mar_cut_rate apr_bonus = 3605 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end bob_salary_april_l344_344914


namespace volume_parallelepiped_l344_344944

variables {V : Type*} [InnerProductSpace ℝ V]

-- Given vectors a, b, and c in the vector space V
variables (a b c : V)

-- Condition: The volume of the parallelepiped determined by a, b, and c is 6
def volume_condition : Prop := abs (inner a (cross_product b c)) = 6

-- The statement to prove:
theorem volume_parallelepiped (h : volume_condition a b c) :
  abs (inner (a + 2 • b) (cross_product (b - 5 • c) (c + 2 • a))) = 54 :=
sorry

end volume_parallelepiped_l344_344944


namespace perimeter_of_new_shape_l344_344112

-- Define the lengths of the sides of the rectangle
def length_of_rectangle : ℝ := 4 / Real.pi
def width_of_rectangle : ℝ := 1 / Real.pi

-- Define the perimeter of the new shape formed by semicircular arcs
theorem perimeter_of_new_shape :
  let long_arc_perimeter := 2 * (Real.pi * length_of_rectangle / 2 / Real.pi)
  let short_arc_perimeter := 2 * (Real.pi * width_of_rectangle / 2 / Real.pi)
  long_arc_perimeter + short_arc_perimeter = 9 := by
  sorry

end perimeter_of_new_shape_l344_344112


namespace bruce_purchased_mangoes_l344_344915

-- Condition definitions
def cost_of_grapes (k_gra kg_cost_gra : ℕ) : ℕ := k_gra * kg_cost_gra
def amount_spent_on_mangoes (total_paid cost_gra : ℕ) : ℕ := total_paid - cost_gra
def quantity_of_mangoes (total_amt_mangoes rate_per_kg_mangoes : ℕ) : ℕ := total_amt_mangoes / rate_per_kg_mangoes

-- Parameters
variable (k_gra rate_per_kg_gra rate_per_kg_mangoes total_paid : ℕ)
variable (kg_gra_total_amt spent_amt_mangoes_qty : ℕ)

-- Given values
axiom A1 : k_gra = 7
axiom A2 : rate_per_kg_gra = 70
axiom A3 : rate_per_kg_mangoes = 55
axiom A4 : total_paid = 985

-- Calculations based on conditions
axiom H1 : cost_of_grapes k_gra rate_per_kg_gra = kg_gra_total_amt
axiom H2 : amount_spent_on_mangoes total_paid kg_gra_total_amt = spent_amt_mangoes_qty
axiom H3 : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9

-- Proof statement to be proven
theorem bruce_purchased_mangoes : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9 := sorry

end bruce_purchased_mangoes_l344_344915


namespace probability_odd_divisor_of_factorial_l344_344809

theorem probability_odd_divisor_of_factorial (n : ℕ) (h : n = 15) :
  let fact : ℕ := Nat.factorial 15,
      total_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1),
      odd_divisors := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) in
  (odd_divisors : ℚ) / total_divisors = (1 : ℚ) / 12 :=
by
  -- condition: 15! prime factorization
  have fact : ℕ := Nat.factorial 15,
  sorry

end probability_odd_divisor_of_factorial_l344_344809


namespace triangle_rows_l344_344874

theorem triangle_rows (h : ∃ n : ℕ, n * (n + 1) = 930) : ∃ n : ℕ, n = 30 :=
by
  cases h with n hn
  use 30
  sorry

end triangle_rows_l344_344874


namespace inequality_solution_l344_344993

def expr(x : ℝ) : ℝ := (x^3 - 1) / (x - 2)^2

theorem inequality_solution :
  { x : ℝ | expr x ≥ 0 } = { x : ℝ | (1 ≤ x ∧ x < 2) ∨ (2 < x) } :=
by
  sorry

end inequality_solution_l344_344993


namespace one_in_M_l344_344374

def M : set ℕ := {0, 1, 2}

theorem one_in_M : 1 ∈ M := by
  sorry

end one_in_M_l344_344374


namespace find_b_minus_a_l344_344471

theorem find_b_minus_a (a b : ℤ) (h1 : a * b = 2 * (a + b) + 11) (h2 : b = 7) : b - a = 2 :=
by sorry

end find_b_minus_a_l344_344471


namespace jade_transactions_l344_344859

-- Definitions for each condition
def transactions_mabel : ℕ := 90
def transactions_anthony : ℕ := transactions_mabel + (transactions_mabel / 10)
def transactions_cal : ℕ := 2 * transactions_anthony / 3
def transactions_jade : ℕ := transactions_cal + 17

-- The theorem stating that Jade handled 83 transactions
theorem jade_transactions : transactions_jade = 83 := by
  sorry

end jade_transactions_l344_344859


namespace find_a_and_b_nonnegative_when_x_nonnegative_range_of_m_l344_344731

section ProofProblem

variables {x a b m : ℝ}

-- Definition of the function
def f (x : ℝ) : ℝ := a * (x + 1)^2 * Real.log (x + 1) + b * x

-- Conditions
def passes_through_point (c d : ℝ) := f (c) = d
def tangent_at_origin := f' 0 = 0

-- Theorem statements
theorem find_a_and_b (ha : a = 1) (hb : b = -1) :
  (tangent_at_origin ∧ passes_through_point (Real.exp 1 - 1) (Real.exp 2 - Real.exp 1 + 1)) :=
sorry

theorem nonnegative_when_x_nonnegative (ha : a = 1) (hb : b = -1) (hx : x ≥ 0) :
  (f x) ≥ x^2 :=
sorry

theorem range_of_m (ha : a = 1) (hb : b = -1) (hx : x ≥ 0) :
  (∀x, f x ≥ m * x^2) → m ≤ 3 / 2 :=
sorry

end ProofProblem

end find_a_and_b_nonnegative_when_x_nonnegative_range_of_m_l344_344731


namespace q_sufficient_but_not_necessary_for_p_l344_344607

variable {x : ℝ}

def p : Prop := |x + 1| > 2
def q : Prop := x^2 - 5x + 6 < 0

theorem q_sufficient_but_not_necessary_for_p : (q → p) ∧ ¬(p → q) := by
  sorry

end q_sufficient_but_not_necessary_for_p_l344_344607


namespace sarah_walked_total_distance_l344_344209

theorem sarah_walked_total_distance :
  ∀ (D : ℝ), 
    (D / 3 + D / 4 = 3.5) →
    2 * D = 12 :=
by
  intro D
  intro h
  have := calc
    2 * D = 12 :=
      sorry
  exact this

end sarah_walked_total_distance_l344_344209


namespace number_of_incorrect_statements_l344_344747

def is_incorrect (statement : Prop) : Prop := ¬statement

def statement_1 : Prop :=
  ∀ (A B C D : ℝ), parallelogram A B C D → side_lengths_fixed A B C D →
  angle_sum_opposite_changes A C

def statement_2 : Prop :=
  ∀ (n : ℕ) (polygon : Polygon n), sum_exterior_angles polygon = 360

def statement_3 : Prop :=
  ∀ (A B C : Point) (triangle : Triangle A B C), rotate_around_vertex A B C → 
  interior_angles_unchanged A B C

def statement_4 : Prop :=
  ∀ (α : Angle) (figure : Figure α), magnify_figure α figure →
  angle_unchanged_before_after_magnification α

def statement_5 : Prop :=
  ∀ (r : ℝ), radius_changes r → (circumference_radius_ratio_const r = 2 * Real.pi)

def statement_6 : Prop :=
  ∀ (r : ℝ), radius_changes r → (circumference_area_ratio_const r)

theorem number_of_incorrect_statements : 
  ( is_incorrect statement_1 ∧
    ¬is_incorrect statement_2 ∧
    ¬is_incorrect statement_3 ∧
    ¬is_incorrect statement_4 ∧
    ¬is_incorrect statement_5 ∧
    is_incorrect statement_6 ) → 2 := sorry

end number_of_incorrect_statements_l344_344747


namespace dante_eggs_l344_344166

theorem dante_eggs (E F : ℝ) (h1 : F = E / 2) (h2 : F + E = 90) : E = 60 :=
by
  sorry

end dante_eggs_l344_344166


namespace integral_eq_e_minus_inv_e_l344_344191

noncomputable def eval_definite_integral : ℝ := ∫ x in -1 .. 1, exp x + 2 * x

theorem integral_eq_e_minus_inv_e : eval_definite_integral = real.exp 1 - real.exp (-1) :=
by
  sorry

end integral_eq_e_minus_inv_e_l344_344191


namespace largest_possible_k_l344_344668

theorem largest_possible_k (k : ℝ) : 
  (∀ (x : ℝ), 
     x ∈ Ioo 0 (π / 2) → 
     (sin x) ^ 3 / (cos x) + (cos x) ^ 3 / (sin x) ≥ k) 
  ↔ k ≤ 1 := 
sorry

end largest_possible_k_l344_344668


namespace perfect_square_d_sequence_l344_344572

theorem perfect_square_d_sequence (d : ℕ) (k : ℕ) (h_k : k ≥ 3) :
  (∃ l : List ℕ, l ~ [i * d | i in List.range k.succ] ∧ 
  ∀ i : ℕ, i < l.length - 1 → ∃ m : ℕ, l.nthLe i sorry + l.nthLe (i + 1) sorry = m ^ 2) ↔ ∃ s : ℕ, d = s^2 :=
sorry

end perfect_square_d_sequence_l344_344572


namespace polygon_sides_l344_344035

-- Definition of the conditions used in the problem
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Statement of the theorem
theorem polygon_sides (n : ℕ) (h : sum_of_interior_angles n = 1080) : n = 8 :=
by
  sorry  -- Proof placeholder

end polygon_sides_l344_344035


namespace natasha_average_speed_l344_344751

theorem natasha_average_speed
  (time_up time_down : ℝ)
  (speed_up distance_up total_distance total_time average_speed : ℝ)
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 3)
  (h4 : distance_up = speed_up * time_up)
  (h5 : total_distance = distance_up + distance_up)
  (h6 : total_time = time_up + time_down)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 4 := by
  sorry

end natasha_average_speed_l344_344751


namespace translated_min_point_l344_344804

theorem translated_min_point : 
  let original_eq (x : ℝ) := abs x - 4,
      translated_eq (x : ℝ) := abs (x - 4) - 4 + 5 in 
  ∃ x y : ℝ, (y = translated_eq x) ∧ ∀ x' y' : ℝ, (y' = translated_eq x') → y ≤ y' :=
begin
  sorry
end

end translated_min_point_l344_344804


namespace round_to_nearest_hundredth_l344_344788

theorem round_to_nearest_hundredth (x : ℝ) (hx : x = 4.509) : Float.round (x * 100) / 100 = 4.51 := by
  sorry

end round_to_nearest_hundredth_l344_344788


namespace modulus_of_2_minus_z_l344_344600

theorem modulus_of_2_minus_z (z : ℂ) (h : z = -1 - complex.i) : complex.abs (2 - z) = real.sqrt 10 := 
by
  rw [h]
  change complex.abs (2 - (-1 - complex.i)) = _
  norm_cast
  rw [sub_neg_eq_add, two_add_eq_add_two, mul_one, inc_le_rfl (inc_le_of_le_zero (inc_le_of_one complex.one_le_one), inc_le_rfl)] 
  sorry

end modulus_of_2_minus_z_l344_344600


namespace test_methods_first_last_test_methods_within_six_l344_344219

open Classical

def perms (n k : ℕ) : ℕ := sorry -- placeholder for permutation function

theorem test_methods_first_last
  (prod_total : ℕ) (defective : ℕ) (first_test : ℕ) (last_test : ℕ) 
  (A4_2 : ℕ) (A5_2 : ℕ) (A6_4 : ℕ) : first_test = 2 → last_test = 8 → 
  perms 4 2 * perms 5 2 * perms 6 4 = A4_2 * A5_2 * A6_4 :=
by
  intro h_first_test h_last_test
  simp [perms]
  sorry

theorem test_methods_within_six
  (prod_total : ℕ) (defective : ℕ) 
  (A4_4 : ℕ) (A4_3_A6_1 : ℕ) (A5_3_A6_2 : ℕ) (A6_6 : ℕ)
  : perms 4 4 + 4 * perms 4 3 * perms 6 1 + 4 * perms 5 3 * perms 6 2 + perms 6 6 
  = A4_4 + 4 * A4_3_A6_1 + 4 * A5_3_A6_2 + A6_6 :=
by
  simp [perms]
  sorry

end test_methods_first_last_test_methods_within_six_l344_344219


namespace volume_pyramid_SABC_l344_344621

noncomputable def diameter := 4
noncomputable def SC := diameter/2
noncomputable def AB := Real.sqrt 3
noncomputable def angleASC := 30
noncomputable def angleBSC := 30

theorem volume_pyramid_SABC (diameter : ℝ) (AB : ℝ) (angleASC : ℝ) (angleBSC : ℝ) :
  diameter = 4 → AB = Real.sqrt 3 → angleASC = 30 → angleBSC = 30 → 
  volume_pyramid S A B C = 2 :=
by
  sorry

end volume_pyramid_SABC_l344_344621


namespace simplify_expr_l344_344393

theorem simplify_expr (x : ℝ) : (3 * x)^5 + (4 * x) * (x^4) = 247 * x^5 :=
by
  sorry

end simplify_expr_l344_344393


namespace value_of_some_number_l344_344665

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l344_344665


namespace population_30_3_million_is_30300000_l344_344303

theorem population_30_3_million_is_30300000 :
  let million := 1000000
  let population_1998 := 30.3 * million
  population_1998 = 30300000 :=
by
  -- Proof goes here
  sorry

end population_30_3_million_is_30300000_l344_344303


namespace find_x_l344_344320

-- Definitions for the angles
def angle1 (x : ℝ) := 3 * x
def angle2 (x : ℝ) := 7 * x
def angle3 (x : ℝ) := 4 * x
def angle4 (x : ℝ) := 2 * x
def angle5 (x : ℝ) := x

-- The condition that the sum of the angles equals 360 degrees
def sum_of_angles (x : ℝ) := angle1 x + angle2 x + angle3 x + angle4 x + angle5 x = 360

-- The statement to prove
theorem find_x (x : ℝ) (hx : sum_of_angles x) : x = 360 / 17 := by
  -- Proof to be written here
  sorry

end find_x_l344_344320


namespace third_quadrant_expressions_l344_344652

variable {α : ℝ} 

theorem third_quadrant_expressions (h1 : sin α < 0) (h2 : cos α < 0) (h3 : tan α > 0) : 
    ¬(tan α - sin α < 0) :=
by 
  intro h
  sorry

end third_quadrant_expressions_l344_344652


namespace other_brick_dimension_l344_344488

theorem other_brick_dimension :
  let courtyard_length_cm := 18 * 100
  let courtyard_breadth_cm := 12 * 100
  let courtyard_area_cm2 := courtyard_length_cm * courtyard_breadth_cm
  let brick_length_cm := 15
  let number_of_bricks := 11077
  (courtyard_area_cm2 = number_of_bricks * brick_length_cm * x) → x = 13 :=
by
  let courtyard_length_cm := 18 * 100
  let courtyard_breadth_cm := 12 * 100
  let courtyard_area_cm2 := courtyard_length_cm * courtyard_breadth_cm
  let brick_length_cm := 15
  let number_of_bricks := 11077
  intro h
  have h_equation : courtyard_area_cm2 = number_of_bricks * brick_length_cm * 13,
  { sorry }
  have h_x : x = 13,
  { sorry }
  exact h_x

end other_brick_dimension_l344_344488


namespace sum_primitive_roots_11_l344_344850

def is_primitive_root_mod (a p : ℕ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n < p → a ^ n % p ≠ 1

def sum_primitive_roots_mod (s : Finset ℕ) (p : ℕ) : ℕ :=
  (s.filter (λ x => (x ≠ 0) ∧ is_primitive_root_mod x p)).sum id

theorem sum_primitive_roots_11 : sum_primitive_roots_mod (Finset.range 11) 11 = 10 :=
by sorry

end sum_primitive_roots_11_l344_344850


namespace vector_dot_product_l344_344595

-- Defining vectors a and b
def a : ℝ × ℝ × ℝ := (-3, 2, 5)
def b : ℝ × ℝ × ℝ := (1, 5, -1)

-- Defining the scalar multiplication and vector addition
def scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def vector_add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

-- Defining the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Proof that a ⋅ (a + 3b) = 44
theorem vector_dot_product : dot_product a (vector_add a (scalar_mult 3 b)) = 44 :=
  sorry

end vector_dot_product_l344_344595


namespace infinite_sqrt_evaluation_l344_344181

noncomputable def infinite_sqrt : ℝ := 
  sqrt (15 + infinite_sqrt)

theorem infinite_sqrt_evaluation : infinite_sqrt = (1 + sqrt 61) / 2 := by
  sorry

end infinite_sqrt_evaluation_l344_344181


namespace painters_time_l344_344333

-- Define the initial conditions
def n1 : ℕ := 3
def d1 : ℕ := 2
def W := n1 * d1
def n2 : ℕ := 2
def d2 := W / n2
def d_r := (3 * d2) / 4

-- Theorem statement
theorem painters_time (h : d_r = 9 / 4) : d_r = 9 / 4 := by
  sorry

end painters_time_l344_344333


namespace cubic_root_equation_solution_l344_344991

theorem cubic_root_equation_solution (x : ℝ) : ∃ y : ℝ, y = real.cbrt x ∧ y = 15 / (8 - y) ↔ x = 27 ∨ x = 125 :=
by
  sorry

end cubic_root_equation_solution_l344_344991


namespace solve_inequality_l344_344819

variable (a b x : ℝ)

noncomputable def inequality_condition (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  x^2 + 2*x < (a / b) + (16 * b / a)

noncomputable def min_value_condition : Prop :=
  x^2 + 2*x < 8

theorem solve_inequality (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (inequality_condition a b a_pos b_pos → min_value_condition) :=
sorry

end solve_inequality_l344_344819


namespace box_volume_l344_344939

structure Box where
  L : ℝ  -- Length
  W : ℝ  -- Width
  H : ℝ  -- Height

def front_face_area (box : Box) : ℝ := box.L * box.H
def top_face_area (box : Box) : ℝ := box.L * box.W
def side_face_area (box : Box) : ℝ := box.H * box.W

noncomputable def volume (box : Box) : ℝ := box.L * box.W * box.H

theorem box_volume (box : Box)
  (h1 : front_face_area box = 0.5 * top_face_area box)
  (h2 : top_face_area box = 1.5 * side_face_area box)
  (h3 : side_face_area box = 72) :
  volume box = 648 := by
  sorry

end box_volume_l344_344939


namespace problem_statement_l344_344252

-- Definitions based on conditions
def A : ℝ := 2
def w : ℝ := 2
def φ : ℝ := π / 6

noncomputable def f (x : ℝ) : ℝ := A * Real.sin(w * x + φ)

-- The proof problem statement
theorem problem_statement :
  (A > 0 ∧ w > 0 ∧ abs φ < π / 2 ∧ 
   f (π / 6) = 2 ∧ f (2 * π / 3) = -2) → 
  (∃ φ, A * Real.sin(w * x + φ) = 2 * Real.sin(2 * x + π / 6)) ∧
  (∀ k : ℤ, ∃ a b : ℝ, f a = f (k * π - π / 3) ∧ 
                      f b = f (k * π + π / 6)) ∧
  (∀ k : ℤ, x = k * π / 2 + π / 6) :=
by
  -- The proof steps
  sorry

end problem_statement_l344_344252


namespace quotient_remainder_base5_l344_344983

theorem quotient_remainder_base5 (n m : ℕ) 
    (hn : n = 3 * 5^3 + 2 * 5^2 + 3 * 5^1 + 2)
    (hm : m = 2 * 5^1 + 1) :
    n / m = 40 ∧ n % m = 2 :=
by
  sorry

end quotient_remainder_base5_l344_344983


namespace tap_B_time_l344_344531

-- Define the capacities and time variables
variable (A_rate B_rate : ℝ) -- rates in percentage per hour
variable (T_A T_B : ℝ) -- time in hours

-- Define the conditions as hypotheses
def conditions : Prop :=
  (4 * (A_rate + B_rate) = 50) ∧ (2 * A_rate = 15)

-- Define the question and the target time
def target_time := 7

-- Define the goal to prove
theorem tap_B_time (h : conditions A_rate B_rate) : T_B = target_time := by
  sorry

end tap_B_time_l344_344531


namespace percentage_per_annum_is_correct_l344_344791

-- Define the conditions of the problem
def banker_gain : ℝ := 24
def present_worth : ℝ := 600
def time : ℕ := 2

-- Define the formula for the amount due
def amount_due (r : ℝ) (t : ℕ) (PW : ℝ) : ℝ := PW * (1 + r * t)

-- Define the given conditions translated from the problem
def given_conditions (r : ℝ) : Prop :=
  amount_due r time present_worth = present_worth + banker_gain

-- Lean statement of the problem to be proved
theorem percentage_per_annum_is_correct :
  ∃ r : ℝ, given_conditions r ∧ r = 0.02 :=
by {
  sorry
}

end percentage_per_annum_is_correct_l344_344791


namespace problem1_problem2_problem3_l344_344871

noncomputable def arrangements_question1 : Nat :=
  let total_students := 7
  let ab_together := 6.factorial * 2.factorial
  ab_together

theorem problem1 : arrangements_question1 = 1440 := by
  sorry

noncomputable def arrangements_question2 : Nat :=
  let total_students := 7
  let abc_not_together := 4.factorial * (5.factorial / 2.factorial)
  abc_not_together

theorem problem2 : arrangements_question2 = 1440 := by
  sorry

noncomputable def arrangements_question3 : Nat :=
  let total_students := 7
  let total_arrangements := total_students.factorial
  let a_at_head_or_b_at_tail := 2 * 6.factorial
  let both_conditions := (6 - 2).factorial
  let result := total_arrangements - a_at_head_or_b_at_tail + both_conditions
  result

theorem problem3 : arrangements_question3 = 3720 := by
  sorry

end problem1_problem2_problem3_l344_344871


namespace calculate_area_l344_344322

def leftmost_rectangle_area (height width : ℕ) : ℕ := height * width
def middle_rectangle_area (height width : ℕ) : ℕ := height * width
def rightmost_rectangle_area (height width : ℕ) : ℕ := height * width

theorem calculate_area : 
  let leftmost_segment_height := 7
  let bottom_width := 6
  let segment_above_3 := 3
  let segment_above_2 := 2
  let rightmost_width := 5
  leftmost_rectangle_area leftmost_segment_height bottom_width + 
  middle_rectangle_area segment_above_3 segment_above_3 + 
  rightmost_rectangle_area segment_above_2 rightmost_width = 
  61 := by
    sorry

end calculate_area_l344_344322


namespace total_paint_area_correct_l344_344119

-- Define dimensions in a structured format
structure Wall :=
  (width : ℝ)
  (height : ℝ)

structure Opening :=
  (width : ℝ)
  (height : ℝ)

def area (shape : {width : ℝ // width > 0} × {height : ℝ // height > 0}) : ℝ :=
  shape.1.val * shape.2.val

def paintable_area (wall : Wall) (openings : List Opening) : ℝ :=
  let wall_area := area ⟨wall.width, wall.height⟩
  let total_opening_area := openings.foldr (λ opening acc, acc + area ⟨opening.width, opening.height⟩) 0
  wall_area - total_opening_area

def total_paintable_area (walls : List (Wall × List Opening)) : ℝ :=
  walls.foldr (λ (wall_openings : Wall × List Opening) acc, acc + paintable_area wall_openings.1 wall_openings.2) 0

-- Define the dimensions for walls, windows, and doors
def wall1 : Wall := ⟨4, 8⟩
def window1 : Opening := ⟨2, 3⟩

def wall2 : Wall := ⟨6, 8⟩
def door1 : Opening := ⟨3, 6.5⟩

def wall3 : Wall := ⟨4, 8⟩
def window2 : Opening := ⟨3, 4⟩

def wall4 : Wall := ⟨6, 8⟩

-- Prove that the total paintable area is 122.5 square feet
theorem total_paint_area_correct :
  total_paintable_area [(wall1, [window1]), (wall2, [door1]), (wall3, [window2]), (wall4, [])] = 122.5 :=
by
  sorry

end total_paint_area_correct_l344_344119


namespace maximize_net_profit_k1_k_range_for_health_risk_l344_344835

-- Definitions based on conditions
def P (k x : ℝ) : ℝ := k * x ^ 3
def Q (x : ℝ) : ℝ := (1/2) * x ^ 2 + 10 * x

-- Define the net profit
def net_profit (k x : ℝ) : ℝ := Q x - P k x

-- The derivative of net_profit
def d_net_profit (k x : ℝ) : ℝ := -3 * k * x ^ 2 + x + 10

-- Problem statements
theorem maximize_net_profit_k1 : (maximize x : ℝ, net_profit 1 x).val = 2 := sorry

theorem k_range_for_health_risk : ∀ k ∈ [11/3, 10], ∀ x > 1, net_profit k x ≤ net_profit k 1 := sorry

end maximize_net_profit_k1_k_range_for_health_risk_l344_344835


namespace card_probability_l344_344827

-- Definitions to capture the problem's conditions in Lean
def total_cards : ℕ := 52
def remaining_after_first : ℕ := total_cards - 1
def remaining_after_second : ℕ := total_cards - 2

def kings : ℕ := 4
def non_heart_kings : ℕ := 3
def non_kings_in_hearts : ℕ := 12
def spades_and_diamonds : ℕ := 26

-- Define probabilities for each step
def prob_first_king : ℚ := non_heart_kings / total_cards
def prob_second_heart : ℚ := non_kings_in_hearts / remaining_after_first
def prob_third_spade_or_diamond : ℚ := spades_and_diamonds / remaining_after_second

-- Calculate total probability
def total_probability : ℚ := prob_first_king * prob_second_heart * prob_third_spade_or_diamond

-- Theorem statement that encapsulates the problem
theorem card_probability : total_probability = 26 / 3675 :=
by sorry

end card_probability_l344_344827


namespace solve_for_x_l344_344780

theorem solve_for_x (x : ℝ) :
  sqrt (9 + sqrt (27 + 3 * x)) + sqrt (3 + sqrt (1 + x)) = 3 + 3 * sqrt 3 →
  x = 10 + 8 * sqrt 3 :=
by
  sorry

end solve_for_x_l344_344780


namespace length_shorter_diagonal_eq_sqrt_29_l344_344248

variables (p q : ℝ)
variables [InnerProductSpace ℝ (EuclideanSpace ℝ (fin 2))]

variables (magnitude_p : ‖p • (1 : EuclideanSpace ℝ (fin 2))‖ = real.sqrt 2)
variables (magnitude_q : ‖q • (1 : EuclideanSpace ℝ (fin 2))‖ = 1)
variables (angle_pq : real.angle (p • (1 : EuclideanSpace ℝ (fin 2))) (q • (1 : EuclideanSpace ℝ (fin 2))) = real.pi / 4)
variables (a : EuclideanSpace ℝ (fin 2)) (b : EuclideanSpace ℝ (fin 2))

noncomputable def vector_a : EuclideanSpace ℝ (fin 2) := 3 • (p • (1 : EuclideanSpace ℝ (fin 2))) + 2 • (q • (1 : EuclideanSpace ℝ (fin 2)))
noncomputable def vector_b : EuclideanSpace ℝ (fin 2) := (p • (1 : EuclideanSpace ℝ (fin 2))) - (q • (1 : EuclideanSpace ℝ (fin 2)))

theorem length_shorter_diagonal_eq_sqrt_29 : 
  (min 
    (real.sqrt ((‖vector_a p q - vector_b p q‖ : ℝ)^2))
    (real.sqrt ((‖vector_a p q + vector_b p q‖ : ℝ)^2))
  ) = real.sqrt 29 :=
sorry

end length_shorter_diagonal_eq_sqrt_29_l344_344248


namespace UTBEL_position_l344_344837

theorem UTBEL_position : 
  let letters := ['B', 'E', 'L', 'T', 'U'] in
  ∃ n : ℕ, n = 117 ∧ (list.insert (list_perm (list.sort letters)) "UTBEL") = n :=
by sorry

end UTBEL_position_l344_344837


namespace option_A_option_B_option_C_option_D_l344_344228

-- Given a complex number z = a + bi where a, b ∈ ℝ
variables {a b : ℝ}
def z : ℂ := complex.of_real a + complex.i * b

-- Prove that z^2 ≥ 0 is a necessary condition for z ∈ ℝ
theorem option_A (a b : ℝ) : (b = 0) → (z^2 ≥ 0) :=
by sorry

-- Prove that if |z| = 1, then the maximum value of |z - z * conjugate z| is 2
theorem option_B (a b : ℝ) (h : complex.abs z = 1) : ∃ z, complex.abs (z - z * complex.conj z) = 2 :=
by sorry

-- If a = 0, b = 1, then prove that ∑_{k=1}^{2023} z^k ≠ 0
theorem option_C : (a = 0) ∧ (b = 1) → (finset.sum (finset.range 2023) (λ k, z^k) ≠ 0) :=
by sorry

-- Prove that if m, n ∈ ℝ, the equation z^2 + m * |z| + n = 0 can have at most 4 solutions in ℝ
theorem option_D (m n : ℝ) : ∃ (count : ℕ), count ≤ 4 ∧ ∀ z : ℂ, (z^2 + m * complex.abs z + n = 0) → z ∈ ℝ :=
by sorry

end option_A_option_B_option_C_option_D_l344_344228


namespace number_of_investment_plans_l344_344097

theorem number_of_investment_plans :
  ∃ (plans : ℕ), plans = 60 ∧
    (∃ cities: Finset String, cities.card = 4 ∧
    "Beijing" ∈ cities ∧ "Shanghai" ∈ cities ∧
    "Hefei" ∈ cities ∧ "Tianzhushan" ∈ cities ∧
    (∃ projects: Finset String, projects.card = 3 ∧
    (∀ city ∈ cities, (projects.filter (λ p, p = city)).card ≤ 2))) :=
begin
  sorry
end

end number_of_investment_plans_l344_344097


namespace binomial_sum_evaluation_l344_344292

theorem binomial_sum_evaluation (m : ℕ) (hm : m > 0) : 
  (∑ k in finset.range (m/2 + 1 : ℕ), (-1:ℤ)^k * (nat.choose (m - k) k : ℤ) * (1:ℝ) / (m - k)) = 
  if 3 ∣ m then 
    (-1:ℤ)^m * (2:ℝ) / m 
  else 
    (-1:ℤ)^(m + 1) * (1:ℝ) / m :=
by sorry

end binomial_sum_evaluation_l344_344292


namespace inverse_of_217_mod_397_l344_344550

theorem inverse_of_217_mod_397 :
  ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ 217 * a % 397 = 1 :=
sorry

end inverse_of_217_mod_397_l344_344550


namespace donuts_left_for_coworkers_l344_344150

theorem donuts_left_for_coworkers :
  ∀ (total_donuts gluten_free regular gluten_free_chocolate gluten_free_plain regular_chocolate regular_plain consumed_gluten_free consumed_regular afternoon_gluten_free_chocolate afternoon_gluten_free_plain afternoon_regular_chocolate afternoon_regular_plain left_gluten_free_chocolate left_gluten_free_plain left_regular_chocolate left_regular_plain),
  total_donuts = 30 →
  gluten_free = 12 →
  regular = 18 →
  gluten_free_chocolate = 6 →
  gluten_free_plain = 6 →
  regular_chocolate = 11 →
  regular_plain = 7 →
  consumed_gluten_free = 1 →
  consumed_regular = 1 →
  afternoon_gluten_free_chocolate = 2 →
  afternoon_gluten_free_plain = 1 →
  afternoon_regular_chocolate = 2 →
  afternoon_regular_plain = 1 →
  left_gluten_free_chocolate = gluten_free_chocolate - consumed_gluten_free * 0.5 - afternoon_gluten_free_chocolate →
  left_gluten_free_plain = gluten_free_plain - consumed_gluten_free * 0.5 - afternoon_gluten_free_plain →
  left_regular_chocolate = regular_chocolate - consumed_regular * 1 - afternoon_regular_chocolate →
  left_regular_plain = regular_plain - consumed_regular * 0 - afternoon_regular_plain →
  left_gluten_free_chocolate + left_gluten_free_plain + left_regular_chocolate + left_regular_plain = 23 :=
by
  intros
  sorry

end donuts_left_for_coworkers_l344_344150


namespace find_t_l344_344692

open Real

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem find_t (t : ℝ) 
  (area_eq_50 : area_of_triangle 3 15 15 0 0 t = 50) :
  t = 325 / 12 ∨ t = 125 / 12 := 
sorry

end find_t_l344_344692


namespace measure_of_angle_C_l344_344678

variables {A B C : Type*} [NormedAddCommGroup A]
variables {a b c : ℝ}
variables {angle_C : ℝ}

/-- In a triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c, respectively. 
    If a^2 + b^2 - sqrt(2)ab = c^2, then the measure of angle C is π/4. -/
theorem measure_of_angle_C 
  (h : a^2 + b^2 - real.sqrt 2 * a * b = c^2) 
  (h₀ : 0 < a) 
  (h₁ : 0 < b)
  (h₂ : 0 < c)
  (h₃ : 0 < angle_C) 
  (h₄ : angle_C < real.pi) : 
  angle_C = real.pi / 4 :=
sorry

end measure_of_angle_C_l344_344678


namespace nancy_first_album_pictures_l344_344379

theorem nancy_first_album_pictures (total_pics : ℕ) (total_albums : ℕ) (pics_per_album : ℕ)
    (h1 : total_pics = 51) (h2 : total_albums = 8) (h3 : pics_per_album = 5) :
    (total_pics - total_albums * pics_per_album = 11) :=
by
    sorry

end nancy_first_album_pictures_l344_344379


namespace equal_areas_N1_N2_of_parallelogram_with_acute_angle_30_l344_344787

theorem equal_areas_N1_N2_of_parallelogram_with_acute_angle_30 {N0 N1 N2 : Type*} 
  (N0_is_parallelogram : Parallelogram N0)
  (acute_angle_N0 : Angle N0 = 30)
  (N1_is_bisected_N0 : FormedByAngleBisectors N1 N0)
  (N2_is_bisected_N1 : FormedByAngleBisectors N2 N1) :
  Area N1 = Area N2 :=
sorry

end equal_areas_N1_N2_of_parallelogram_with_acute_angle_30_l344_344787


namespace percent_difference_l344_344285

theorem percent_difference : 
  let a := 0.60 * 50
  let b := 0.45 * 30
  a - b = 16.5 :=
by
  let a := 0.60 * 50
  let b := 0.45 * 30
  sorry

end percent_difference_l344_344285


namespace arithmetic_sequence_a5_value_l344_344318

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = 2 * a 6 + a 4) : 
  a 5 = -1 / 2 :=
by
  sorry

end arithmetic_sequence_a5_value_l344_344318


namespace polynomial_degree_conditions_l344_344006

theorem polynomial_degree_conditions
  (m n : ℕ) (P : ℤ[X])
  (h1 : P = (polynomial.C (m - 1) * polynomial.X ^ 3 + polynomial.X ^ n - 1))
  (h2 : polynomial.degree P = 2) :
  m = 1 ∧ n = 2 ∧ polynomial.coeff P 0 = -1 := 
sorry

end polynomial_degree_conditions_l344_344006


namespace inequalities_not_equivalent_l344_344536

theorem inequalities_not_equivalent (x : ℝ) (h1 : x ≠ 1) :
  (x + 3 - (1 / (x - 1)) > -x + 2 - (1 / (x - 1))) ↔ (x + 3 > -x + 2) → False :=
by
  sorry

end inequalities_not_equivalent_l344_344536


namespace cone_volume_l344_344517

theorem cone_volume (R : ℝ) (h r : ℝ) (h_circ : 2 * pi * r = pi * R) (h_pyth : R^2 = r^2 + h^2) (h_vol : V = (1/3) * pi * r^2 * h) : 
∃ (V : ℝ), V = (pi * R^3 * real.sqrt 3) / 24 :=
by 
  use (pi * R^3 * real.sqrt 3) / 24
  sorry

end cone_volume_l344_344517


namespace first_investment_amount_is_500_l344_344135

-- Define the conditions as constants
constant yearly_return_first_investment_rate : ℝ := 0.07
constant yearly_return_second_investment_rate : ℝ := 0.19
constant yearly_combined_return_rate : ℝ := 0.16
constant second_investment_amount : ℕ := 1500

-- The problem is to find the first investment amount such that
-- the yearly return conditions hold.
constant first_investment_amount : ℝ

-- The yearly return of the first investment
def yearly_return_first_investment (x : ℝ) : ℝ := yearly_return_first_investment_rate * x

-- The yearly return of the second investment
def yearly_return_second_investment : ℝ := yearly_return_second_investment_rate * second_investment_amount

-- The combined yearly return of the two investments
def combined_yearly_return (x : ℝ) : ℝ := yearly_combined_return_rate * (x + second_investment_amount)

-- Now we need to state the theorem (problem)
theorem first_investment_amount_is_500 :
  ∃ x : ℝ, (yearly_return_first_investment x + yearly_return_second_investment = combined_yearly_return x) ∧ x = 500 :=
sorry

end first_investment_amount_is_500_l344_344135


namespace smallest_prime_after_seven_non_primes_l344_344327

-- Define the property of being non-prime
def non_prime (n : ℕ) : Prop :=
¬Nat.Prime n

-- Statement of the proof problem
theorem smallest_prime_after_seven_non_primes :
  ∃ m : ℕ, (∀ i : ℕ, (m - 7 ≤ i ∧ i < m) → non_prime i) ∧ Nat.Prime m ∧
  (∀ p : ℕ, (∀ i : ℕ, (p - 7 ≤ i ∧ i < p) → non_prime i) → Nat.Prime p → m ≤ p) :=
sorry

end smallest_prime_after_seven_non_primes_l344_344327


namespace find_A_days_l344_344089

-- Definitions based on the given conditions
def B_days := 20
def work_together_days := 6
def remaining_work := 0.3

-- Question: How many days does it take for A to do the work alone?
theorem find_A_days (x : ℕ) (h_cond : work_together_days * (1 / x + 1 / B_days) = 0.7) : x = 15 :=
sorry

end find_A_days_l344_344089


namespace average_M_in_range_is_11_l344_344842

/-- The average of all integer values of M such that 7 < M < 15 is 11. -/
theorem average_M_in_range_is_11 : (let s := {M : ℕ | 7 < M ∧ M < 15} in (s.sum id) / s.card = 11) :=
by
  sorry

end average_M_in_range_is_11_l344_344842


namespace magn_of_vector_diff_l344_344272

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b_squared := b.1^2 + b.2^2
  let scalar := dot_product / norm_b_squared
  (scalar * b.1, scalar * b.2)

theorem magn_of_vector_diff :
  ∀ (a b : ℝ × ℝ),
  a.1 * b.1 + a.2 * b.2 = -2 →
  b = (1, Real.sqrt 3) →
  let c := vector_projection a b in
  Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) = 3 :=
by
  intros a b hab hbc
  sorry

end magn_of_vector_diff_l344_344272


namespace cristian_cookie_problem_l344_344937

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end cristian_cookie_problem_l344_344937


namespace no_permutation_satisfies_condition_l344_344580

/-- A permutation (b₁, b₂, b₃, b₄, b₅, b₆) of (1, 2, 3, 4, 5, 6) that satisfies the condition 
  (b₁ + 1) / 3 * (b₂ + 2) / 3 * (b₃ + 3) / 3 * (b₄ + 4) / 3 * (b₅ + 5) / 3 * (b₆ + 6) / 3 > 5! does not exist. -/
theorem no_permutation_satisfies_condition :
  ¬ ∃ (b: Fin 6 → ℕ), (∀ i, 1 ≤ b i ∧ b i ≤ 6) 
    ∧ Multiset.countp (λ n, 1 ≤ n ∧ n ≤ 6) (Multiset.of_fn b) = 6
    ∧ (∏ i, (b i + (i + 1)) / 3 : ℝ) > real.factorial 5 :=
sorry

end no_permutation_satisfies_condition_l344_344580


namespace probability_at_most_two_heads_l344_344045

open Finset

-- Definitions for the problem
def sample_space : Finset (Finset ℕ) := 
  { {0, 1, 2}, {0, 1}, {0, 2}, {1, 2}, {0}, {1}, {2}, ∅ }

-- Outcome conditions
def at_most_two_heads : Finset (Finset ℕ) :=
  { {0, 1, 2}, {0, 1}, {0, 2}, {1, 2}, {0}, {1}, {2} } -- Excludes ∅

def favorable_outcomes : ℕ := (at_most_two_heads.card : ℕ)
def total_outcomes : ℕ := (sample_space.card)

def probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_at_most_two_heads : probability = 7 / 8 := by
  -- The exact proof to be filled here
  -- For now, we leave a sorry as per the instruction
  sorry

end probability_at_most_two_heads_l344_344045


namespace problem_statement_l344_344189

-- Defining the problem and its conditions
def x : ℝ := sqrt (15 + sqrt (15 + sqrt (15 + sqrt (15 + ...))))

-- The statement to be proved
theorem problem_statement (hx : x = sqrt (15 + x)) : x = (1 + sqrt 61) / 2 :=
sorry

end problem_statement_l344_344189


namespace abs_value_expression_l344_344649

theorem abs_value_expression (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 :=
sorry

end abs_value_expression_l344_344649


namespace dogs_eat_times_per_day_l344_344748

theorem dogs_eat_times_per_day (dogs : ℕ) (food_per_dog_per_meal : ℚ) (total_food : ℚ) 
                                (food_left : ℚ) (days : ℕ) 
                                (dogs_eat_times_per_day : ℚ)
                                (h_dogs : dogs = 3)
                                (h_food_per_dog_per_meal : food_per_dog_per_meal = 1 / 2)
                                (h_total_food : total_food = 30)
                                (h_food_left : food_left = 9)
                                (h_days : days = 7) :
                                dogs_eat_times_per_day = 2 :=
by
  -- Proof goes here
  sorry

end dogs_eat_times_per_day_l344_344748


namespace sequence_sum_l344_344636

theorem sequence_sum :
  (∀ n : ℕ, 0 < n → a (n + 1) = a n + 1) ∧ a 1 = 1 →
  (∑ i in Finset.range 99,  (1 : ℚ) / (a i.succ * a (i.succ + 1))) = 99 / 100 := 
by
  sorry

end sequence_sum_l344_344636


namespace find_some_number_l344_344653

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l344_344653


namespace probability_all_three_blue_l344_344068

theorem probability_all_three_blue :
  let total_jellybeans := 20
  let initial_blue := 10
  let initial_red := 10
  let prob_first_blue := initial_blue / total_jellybeans
  let prob_second_blue := (initial_blue - 1) / (total_jellybeans - 1)
  let prob_third_blue := (initial_blue - 2) / (total_jellybeans - 2)
  prob_first_blue * prob_second_blue * prob_third_blue = 2 / 19 := 
by
  sorry

end probability_all_three_blue_l344_344068


namespace remainder_proof_l344_344845

theorem remainder_proof : 1234567 % 12 = 7 := sorry

end remainder_proof_l344_344845


namespace count_modified_monotonous_numbers_l344_344941

def isModifiedMonotonous (s : List ℕ) : Prop :=
  (s.length = 1 ∨
    (∀ i, i < s.length - 1 → s.get! i < s.get! (i + 1)) ∨
    (∀ i, i < s.length - 1 → s.get! i > s.get! (i + 1)))

theorem count_modified_monotonous_numbers :
  ∃ n, n = 2018 ∧
    (∀ s : List ℕ, isModifiedMonotonous s → s.length.divide n) :=
by
  sorry

end count_modified_monotonous_numbers_l344_344941


namespace sequence_a3_value_l344_344265

theorem sequence_a3_value :
  ∀ (a : ℕ → ℕ),
    (a 1 = 1) →
    (∀ n, a (n + 1) = 2 * a n + 1) →
    a 3 = 7 :=
by {
  intro a,
  intro h1,
  intro h_rec,
  sorry
}

end sequence_a3_value_l344_344265


namespace sin_270_eq_neg_one_l344_344161

theorem sin_270_eq_neg_one : Real.sin (270 * Real.pi / 180) = -1 := 
by
  sorry

end sin_270_eq_neg_one_l344_344161


namespace time_spent_on_type_a_l344_344078

theorem time_spent_on_type_a (num_questions : ℕ) 
                             (exam_duration : ℕ)
                             (type_a_count : ℕ)
                             (time_ratio : ℕ)
                             (type_b_count : ℕ)
                             (x : ℕ)
                             (total_time : ℕ) :
  num_questions = 200 ∧
  exam_duration = 180 ∧
  type_a_count = 20 ∧
  time_ratio = 2 ∧
  type_b_count = 180 ∧
  total_time = 36 →
  time_ratio * x * type_a_count + x * type_b_count = exam_duration →
  total_time = 36 :=
by
  sorry

end time_spent_on_type_a_l344_344078


namespace red_chips_drawn_first_probability_l344_344886

noncomputable def prob_red_chips_drawn_first : ℚ :=
  let total_chips := {chip | chip ∈ {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}}
  in let favorable_arrangements := {arrangement | arrangement.take(3).count(= red_chip) = 3}
  in favorable_arrangements.card / total_chips.card

theorem red_chips_drawn_first_probability :
  prob_red_chips_drawn_first = 1 / 2 :=
sorry

end red_chips_drawn_first_probability_l344_344886


namespace total_cost_supplies_l344_344900

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end total_cost_supplies_l344_344900


namespace student_B_speed_l344_344505

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l344_344505


namespace greatest_root_of_f_one_is_root_of_f_l344_344998

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∀ x : ℝ, f x = 0 → x ≤ 1 :=
sorry

theorem one_is_root_of_f :
  f 1 = 0 :=
sorry

end greatest_root_of_f_one_is_root_of_f_l344_344998


namespace no_ghost_not_multiple_of_p_l344_344718

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sequence_S (p : ℕ) (S : ℕ → ℕ) : Prop :=
  (is_prime p ∧ p % 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i < p → S i = i) ∧
  (∀ n, n ≥ p → (S n > S (n-1) ∧ 
    ∀ (a b c : ℕ), (a < b ∧ b < c ∧ c < n ∧ S a < S b ∧ S b < S c ∧
    S b - S a = S c - S b → false)))

def is_ghost (p : ℕ) (S : ℕ → ℕ) (g : ℕ) : Prop :=
  ∀ n : ℕ, S n ≠ g

theorem no_ghost_not_multiple_of_p (p : ℕ) (S : ℕ → ℕ) :
  (is_prime p ∧ p % 2 = 1) ∧ sequence_S p S → 
  ∀ g : ℕ, is_ghost p S g → p ∣ g :=
by 
  sorry

end no_ghost_not_multiple_of_p_l344_344718


namespace find_sum_of_six_least_positive_integers_l344_344358

def tau (n : ℕ) : ℕ := (List.range (n + 1)).count_dvd n

theorem find_sum_of_six_least_positive_integers :
  ∃ (a b c d e f : ℕ), 
  (tau a + tau (a + 1) = 8) ∧
  (tau b + tau (b + 1) = 8) ∧
  (tau c + tau (c + 1) = 8) ∧
  (tau d + tau (d + 1) = 8) ∧
  (tau e + tau (e + 1) = 8) ∧
  (tau f + tau (f + 1) = 8) ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
  a + b + c + d + e + f = 1017 :=
by
  sorry

end find_sum_of_six_least_positive_integers_l344_344358


namespace yanna_total_cost_proof_l344_344066

-- Defining the costs of each category Yanna purchased
def total_cost_shirts := 10 * 5
def total_cost_sandals := 3 * 3
def total_cost_hats := 5 * 8
def total_cost_bags := 7 * 14
def total_cost_sunglasses := 2 * 12

-- Defining the total cost of all items
def total_cost := total_cost_shirts + total_cost_sandals + total_cost_hats + total_cost_bags + total_cost_sunglasses

-- Defining the amount given by Yanna
def yanna_given := 200

-- Stating the theorem that Yanna needs to provide an additional $21
theorem yanna_total_cost_proof : total_cost = 221 ∧ yanna_given = 200 → yanna_given - total_cost = -21 :=
by {
  sorry
}

end yanna_total_cost_proof_l344_344066


namespace smallest_palindrome_in_base3_and_base5_l344_344926

def is_palindrome_base (b n : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_palindrome_in_base3_and_base5 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome_base 3 n ∧ is_palindrome_base 5 n ∧ n = 20 :=
by
  sorry

end smallest_palindrome_in_base3_and_base5_l344_344926


namespace evolute_of_ellipse_l344_344577

theorem evolute_of_ellipse (a b θ : ℝ) (x y : ℝ) 
  (h1 : (x = a * Real.cos θ ∧ y = b * Real.sin θ)) 
  (h2 : a > 0 ∧ b > 0) 
  (h3 : (x^2 / a^2) + (y^2 / b^2) = 1) : 
  (((a * x) ^ (2 / 3)) + ((b * y) ^ (2 / 3)) = ((a^2 - b^2) ^ (2 / 3))) := 
begin
  sorry
end

end evolute_of_ellipse_l344_344577


namespace solution_set_of_inequality_l344_344032

theorem solution_set_of_inequality : {x : ℝ | -2 < x ∧ x < 1} = {x : ℝ | -x^2 - x + 2 > 0} :=
by
  sorry

end solution_set_of_inequality_l344_344032


namespace final_number_is_88_or_94_l344_344889

-- Initial condition: a number consisting of 98 eights
def initialNumber : ℕ := (list.replicate 98 8).foldl (λ acc, acc * 10 + 8) 0

-- Operation 1: Replace the number with (|a - b|)
def operation1 (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := n % 1000
  abs (a - b)

-- Operation 2: Increase a digit that is not 9 and has neighboring digits > 0 by 1, decrease neighbors by 1
def operation2 (n : ℕ) : ℕ := sorry

-- Define the final condition: the two-digit number remains
def finalTwoDigitNumber : ℕ := sorry

-- Lean theorem: Prove that the remaining two-digit number is either 88 or 94
theorem final_number_is_88_or_94 : ∃ n : ℕ, n / 10 < 10 ∧ n / 10 > 0 ∧ (n = 88 ∨ n = 94) :=
  sorry

end final_number_is_88_or_94_l344_344889


namespace range_of_x_l344_344672

theorem range_of_x (x : ℝ) : 
  (∀ (m : ℝ), |m| ≤ 1 → x^2 - 2 > m * x) ↔ (x < -2 ∨ x > 2) :=
by 
  sorry

end range_of_x_l344_344672


namespace even_function_f3_l344_344295

theorem even_function_f3 (a : ℝ) (h : ∀ x : ℝ, (x + 2) * (x - a) = (-x + 2) * (-x - a)) : (3 + 2) * (3 - a) = 5 := by
  sorry

end even_function_f3_l344_344295


namespace transform_cos_to_sin_shift_l344_344259

theorem transform_cos_to_sin_shift {x : ℝ} :
  (∀ x ∈ Icc (-π / 2) (3 * π / 2), y = cos x) →
  (∀ x ∈ Icc 0 (2 * π), y = sin x) →
  (∀ x ∈ Icc 0 (2 * π), sin x = cos (x - π / 2)) :=
by
  intros h_cos h_sin
  funext x hx
  exact sorry

end transform_cos_to_sin_shift_l344_344259


namespace maximal_area_of_cross_section_l344_344503

theorem maximal_area_of_cross_section :
  ∀ (prism : Type) (s : ℝ),
  (∀ p1 p2 p3 p4 : prism, 
    ((p1.to_point = (4 * real.sqrt 2, 0, 0)) ∧ (p2.to_point = (0, 4 * real.sqrt 2, 0)) ∧ 
    (p3.to_point = (-4 * real.sqrt 2, 0, 0)) ∧ (p4.to_point = (0, -4 * real.sqrt 2, 0))) → 
    ∃ (plane : ℝ × ℝ × ℝ × ℝ), plane = (3, -5, 2, 20) →
    ∃ (area : ℝ), area = 32 * real.sqrt 11) :=
sorry

end maximal_area_of_cross_section_l344_344503


namespace percent_questions_answered_correctly_l344_344381

-- Definition of conditions
variables (y : ℕ)
def total_questions := 7 * y
def questions_missed := 2 * y

-- Theorem statement
theorem percent_questions_answered_correctly (y : ℕ) :
  let correct_questions := total_questions y - questions_missed y in
  let fraction_correct := (correct_questions : ℚ) / (total_questions y) in
  (fraction_correct * 100 = 500 / 7) :=
by
  sorry

end percent_questions_answered_correctly_l344_344381


namespace factorize_x9_minus_512_l344_344154

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l344_344154


namespace total_grid_rectangles_l344_344278

-- Define the horizontal and vertical rectangle counting functions
def count_horizontal_rects : ℕ :=
  (1 + 2 + 3 + 4 + 5)

def count_vertical_rects : ℕ :=
  (1 + 2 + 3 + 4)

-- Subtract the overcounted intersection and calculate the total
def total_rectangles (horizontal : ℕ) (vertical : ℕ) (overcounted : ℕ) : ℕ :=
  horizontal + vertical - overcounted

-- Main statement
theorem total_grid_rectangles : count_horizontal_rects + count_vertical_rects - 1 = 24 :=
by
  simp [count_horizontal_rects, count_vertical_rects]
  norm_num
  sorry

end total_grid_rectangles_l344_344278


namespace fraction_to_decimal_l344_344968

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l344_344968


namespace num_correct_props_is_2_l344_344122

-- Definitions of the conditions stated in the problem
def prop1 : Prop :=
∀ (rect_prism : Type) (has_equal_edges : rect_prism → Prop), 
  ∀ p : rect_prism, has_equal_edges p → (∃ (c : Type) (is_cube : c → Prop), is_cube p)

def prop2 : Prop :=
∀ (parallelepiped : Type) (has_equal_diagonals : parallelepiped → Prop) (is_right_parallelepiped : parallelepiped → Prop),
  ∀ p : parallelepiped, has_equal_diagonals p → is_right_parallelepiped p

def prop3 : Prop :=
∀ (parallelepiped : Type) (has_two_lateral_edges_perpendicular_to_side_of_base : parallelepiped → Prop) (is_right_parallelepiped : parallelepiped → Prop),
  ∀ p : parallelepiped, has_two_lateral_edges_perpendicular_to_side_of_base p → ¬ is_right_parallelepiped p

def prop4 : Prop :=
∀ (parallelepiped : Type) 
   (has_all_faces_rhombi : parallelepiped → Prop) 
   (is_circumcenter_projection : parallelepiped → Prop),
  ∀ p : parallelepiped, has_all_faces_rhombi p → is_circumcenter_projection p

def prop5 : Prop :=
∀ (parallelepiped : Type) 
   (has_all_faces_rectangles : parallelepiped → Prop) 
   (is_incenter_projection : parallelepiped → Prop),
  ∀ p : parallelepiped, has_all_faces_rectangles p → ¬ is_incenter_projection p

-- Statement asserting that the number of correct propositions is 2
theorem num_correct_props_is_2 : 
  (∃ prop1_prop : prop1, true) ∧
  (∃ prop2_prop : prop2, true) ∧ 
  (∃ prop3_prop : not prop3, true) ∧ 
  (∃ prop4_prop : prop4, true) ∧ 
  (∃ prop5_prop : not prop5, true) → 
  (finset.card (finset.filter (λ p, p) 
    (finset.of_list [prop1, prop2, not prop3, prop4, not prop5])) = 2) :=
sorry

end num_correct_props_is_2_l344_344122


namespace measure_angle_DAC_l344_344270

open EuclideanGeometry

theorem measure_angle_DAC (A B C D : Point)
  (h₁: ∠ B = 60)
  (h₂: ∠ C = 75)
  (h₃: IsoscelesRightTriangle B D C) :
  ∠ D A C = 30 :=
sorry

end measure_angle_DAC_l344_344270


namespace smallest_square_side_length_paintings_l344_344525

theorem smallest_square_side_length_paintings (n : ℕ) :
  ∃ n : ℕ, (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2020 → 1 * i ≤ n * n) → n = 1430 :=
by
  sorry

end smallest_square_side_length_paintings_l344_344525


namespace probability_not_blue_marble_l344_344674

-- Define the conditions
def odds_for_blue_marble : ℕ := 5
def odds_for_not_blue_marble : ℕ := 6
def total_outcomes := odds_for_blue_marble + odds_for_not_blue_marble

-- Define the question and statement to be proven
theorem probability_not_blue_marble :
  (odds_for_not_blue_marble : ℚ) / total_outcomes = 6 / 11 :=
by
  -- skipping the proof step as per instruction
  sorry

end probability_not_blue_marble_l344_344674


namespace percentage_loss_l344_344493
noncomputable def sell_price := 50
noncomputable def gain := 0.20
noncomputable def articles_sold := 20
noncomputable def new_sell_price := 90
noncomputable def new_articles := 53.99999325000085

theorem percentage_loss : 
  (sell_price / (1 + gain) / articles_sold * new_articles - new_sell_price) / (sell_price / (1 + gain) / articles_sold * new_articles) * 100 = 20 :=
sorry

end percentage_loss_l344_344493


namespace coin_problem_l344_344326

def alpha := (Math.sqrt 29 - 1) / 2

theorem coin_problem (n : ℕ) (h : alpha > 2) :
  ∃ (k : ℕ) (a b : ℕ → ℕ), (∀ m, a m < 7 ∧ b m < 7) ∧ (n = Σ k : ℕ, a k + b k * (alpha^k)) :=
by
  sorry

end coin_problem_l344_344326


namespace translation_of_sin_function_l344_344447

def original_function (x : ℝ) : ℝ := sin (2 * x)

def translated_function (x : ℝ) : ℝ := sin (2 * (x - π / 3))

theorem translation_of_sin_function :
  ∀ x : ℝ, translated_function x = sin (2 * x - 2 * π / 3) :=
by
  intros
  have h1 : 2 * (x - π / 3) = 2 * x - 2 * π / 3
  {
    ring
  }
  rw [translated_function, h1]
  sorry

end translation_of_sin_function_l344_344447


namespace cylinder_surface_area_l344_344546

theorem cylinder_surface_area (h r : ℝ) (Hh : h = 8) (Hr : r = 5) : 
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 130 * Real.pi :=
by
  rw [Hh, Hr]
  norm_num
  ring
  sorry

end cylinder_surface_area_l344_344546


namespace time_period_proof_l344_344877

noncomputable def equivalent_time_period (P : ℝ) (initial_rate : ℝ) (initial_time_years : ℝ) (initial_time_months : ℕ) (final_rate : ℝ) : ℕ × ℕ × ℕ :=
  let initial_time := initial_time_years + initial_time_months / 12
  let I1 := P * initial_rate * initial_time / 100
  let t2 := I1 * 100 / (P * final_rate)
  let t2_years : ℕ := floor t2
  let remaining_months := (t2 - t2_years) * 12
  let t2_months : ℕ := floor remaining_months
  let remaining_days := (remaining_months - t2_months) * 30
  let t2_days : ℕ := floor remaining_days
  (t2_years, t2_months, t2_days)

theorem time_period_proof :
  equivalent_time_period 1 4 2 5 3.75 = (2, 6, 28) :=
  sorry

end time_period_proof_l344_344877


namespace tetrahedron_CD_length_l344_344930

noncomputable def lengthCD (AB AC BC AD BD: ℝ) : Set ℝ :=
  {sqrt (35) + sqrt (23), 
   sqrt (35) - sqrt (23)}

theorem tetrahedron_CD_length (A B C D : Point) 
  (hAB : dist A B = 2)
  (hAC : dist A C = 5)
  (hBC : dist B C = 5)
  (hAD : dist A D = 6)
  (hBD : dist B D = 6)
  (h_cylinder : inscribed_in_cylinder A B C D)
  (h_parallel : parallel CD (axis_of_cylinder A B C D)) :
  dist C D ∈ lengthCD 2 5 5 6 6 :=
sorry

end tetrahedron_CD_length_l344_344930


namespace gcf_of_26_and_16_is_8_l344_344833

-- Conditions and definitions
def n : ℕ := 26
def m : ℕ := 16
def lcm : ℕ := 52

-- Proof statement
theorem gcf_of_26_and_16_is_8 : Nat.gcd n m = 8 :=
by
  have h_lcm : Nat.lcm n m = lcm := by sorry
  have h_eq : n * m = lcm * Nat.gcd n m := by sorry
  sorry

end gcf_of_26_and_16_is_8_l344_344833


namespace number_of_teams_l344_344911

theorem number_of_teams (n k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  nat.choose n k = 252 :=
by
  sorry

end number_of_teams_l344_344911


namespace man_man_man_man_l344_344492

section BoatSpeedProof

variable (Vc1 Vc2 : ℝ) (Vc3 : ℝ)
variable (Vb Vb_sec1 Vb_sec2 Vb_sec3 : ℝ)

axiom Vc1_def : Vc1 = 2.8
axiom Vc2_def : Vc2 = 3.0
axiom Vc3_def : Vc3 = 4.5
axiom Vb_sec1_def : Vb_sec1 = 15

theorem man's_speed_in_still_water : Vb = 15 - 2.8 := by
  rw [Vc1_def]
  exact rfl

theorem man's_speed_in_first_section : Vb_sec1 = Vb + 2.8 := by
  rw [Vc1_def]
  exact rfl

theorem man's_speed_in_second_section : Vb_sec2 = Vb := by
  exact rfl

theorem man's_speed_in_third_section : Vb_sec3 = Vb - Vc3 := by
  rw [Vc3_def]
  exact rfl

-- Definitions of values are based on previous calculations.
noncomputable def Vb : ℝ := 12.2
noncomputable def Vb_sec1 : ℝ := 15
noncomputable def Vb_sec2 : ℝ := 12.2
noncomputable def Vb_sec3 : ℝ := 7.7

end BoatSpeedProof

end man_man_man_man_l344_344492


namespace divisor_sum_2014_squared_l344_344373

theorem divisor_sum_2014_squared :
  let divisors : List ℕ := (List.range (2014^2).succ).filter (λ n, 2014^2 % n = 0)
  ∑ i in divisors.map (λ d, 1/(d + 2014 : ℚ)), = (27 / 4028 : ℚ) :=
by
  let p1 := 2014
  let fact_p1 := p1 * p1
  have divisors := filter (λ n, fact_p1 % n = 0) (range (fact_p1 + 1))
  have hd_len : length divisors = 27 :=
    sorry
  have sum_eq : ∑ i in map (λ d, 1 / (d + 2014 : ℚ)) divisors = 27 / 4028
    := sorry
  exact sum_eq

end divisor_sum_2014_squared_l344_344373


namespace find_some_number_l344_344655

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l344_344655


namespace problem1_problem2_l344_344473

theorem problem1:
  3^(1 + log 3 2) = 6 :=
sorry

theorem problem2:
  log 3 (∏ i in finset.range(80), (i+1) / (i+2)) = -4 :=
sorry

end problem1_problem2_l344_344473


namespace men_entered_room_l344_344708

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l344_344708


namespace simplify_expression_l344_344855

theorem simplify_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) (h3 : (2 * a) / ((1 + a) * (1 + a)^(1/3)) ≥ 0) :
  sqrt((2 * a) / ((1 + a) * (1 + a)^(1/3))) * ((4 + (8 / a) + (4 / a^2))^(1/3) / (2^(1/2))^(1/3)) = (2 * a^(5/6)) / a := sorry

end simplify_expression_l344_344855


namespace river_length_l344_344715

theorem river_length :
  let still_water_speed := 10 -- Karen's paddling speed on still water in miles per hour
  let current_speed      := 4  -- River's current speed in miles per hour
  let time               := 2  -- Time it takes Karen to paddle up the river in hours
  let effective_speed    := still_water_speed - current_speed -- Karen's effective speed against the current
  effective_speed * time = 12 -- Length of the river in miles
:= by
  sorry

end river_length_l344_344715


namespace billy_initial_lemon_heads_l344_344543

theorem billy_initial_lemon_heads (n f : ℕ) (h_friends : f = 6) (h_eat : n = 12) :
  f * n = 72 := 
by
  -- Proceed by proving the statement using Lean
  sorry

end billy_initial_lemon_heads_l344_344543


namespace quadrilateral_perimeter_l344_344053

theorem quadrilateral_perimeter
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC DC AD : ℝ)
  (h1 : AB = 6) (h2 : DC = 6) (h3 : BC = 7)
  (h_perp1 : AB ⊥ BC) (h_perp2 : DC ⊥ BC) :
  AB + BC + DC + AD = 26 :=
by
  sorry

end quadrilateral_perimeter_l344_344053


namespace min_sum_product_l344_344218

theorem min_sum_product (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 9/n = 1) :
  m * n = 48 :=
sorry

end min_sum_product_l344_344218


namespace arithmetic_series_remainder_l344_344054

-- Define the sequence parameters
def a : ℕ := 2
def l : ℕ := 12
def d : ℕ := 1
def n : ℕ := (l - a) / d + 1

-- Define the sum of the arithmetic series
def S : ℕ := n * (a + l) / 2

-- The final theorem statement
theorem arithmetic_series_remainder : S % 9 = 5 := 
by sorry

end arithmetic_series_remainder_l344_344054


namespace part1_part2_l344_344229

-- Definitions of the line and parabola based on their equations
def line (m : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = p.1 + m
def parabola : ℝ × ℝ → Prop := λ p, p.2^2 = 8 * p.1

-- Proof problem for (1) |AB| = 10 ⇒ m = 7/16
theorem part1 (m : ℝ) (A B : ℝ × ℝ) (hA : line m A) (hB : line m B) (hP : parabola A) (hQ : parabola B)
  (hD : (real.sqrt 2) * (real.abs (A.1 - B.1)) = 10) :
  m = 7 / 16 :=
sorry

-- Proof problem for (2) ⟪OA, OB⟫ = 0 ⇒ m = -8
theorem part2 (m : ℝ) (A B : ℝ × ℝ) (hA : line m A) (hB : line m B) (hP : parabola A) (hQ : parabola B)
  (hD : ⟪A.1 - 0, A.2 - 0⟫ * ⟪B.1 - 0, B.2 - 0⟫ = 0) :
  m = -8 :=
sorry

end part1_part2_l344_344229


namespace arithmetic_sequence_a3_l344_344910

theorem arithmetic_sequence_a3 {a1 d a3 : ℝ} (h1 : a1 + 7 * d = 24) (h2 : 3 * a1 + 7.5 * d = 24) : a3 = 6 :=
begin
  /- Definitions and conditions are directly from the problem -/
  let a₄ := a1 + 3 * d,
  let a₅ := a1 + 4 * d,
  let S₆ := 3 * (2 * a1 + 5 * d),
  /- Condition 1: Given -/
  have h₁ : a₄ + a₅ = 24 := by assumption,
  /- Condition 2: Given -/
  have h₂ : S₆ = 48 := by assumption,
  /- From h1, we can calculate expressions for a1 and d -/
  have eqn_1 : 2 * a1 + 7 * d = 24 := by linarith [h₁],
  have eqn_2 : 6 * a1 + 15 * d = 48 := by linarith [h₂],
  /- Proceed with the necessary calculations -/
  sorry
end

end arithmetic_sequence_a3_l344_344910


namespace fraction_to_decimal_l344_344978

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l344_344978


namespace min_value_of_y_l344_344423

noncomputable theory

open Real

def y (x : ℝ) : ℝ := - (sin x)^2 - 3 * cos x + 3

theorem min_value_of_y :
  ∃ x : ℝ, y x = 0 :=
sorry

end min_value_of_y_l344_344423


namespace numberOfSquaresWithNaturalVertices_l344_344319

def isSquareWithNaturalVertices (x y : ℕ) (center : ℕ × ℕ) : Prop :=
  let (cx, cy) := center
  ∃ k, k > 0 ∧
    ((x = cx + k ∧ y = cy + k) ∨
    (x = cx - k ∧ y = cy + k) ∨
    (x = cx + k ∧ y = cy - k) ∨
    (x = cx - k ∧ y = cy - k))

theorem numberOfSquaresWithNaturalVertices :
  let center := (55, 40) in
  (∃ n, n = 39) →
  (∑ d in Finset.range 40, if d > 0 then (d.dvd 55 ∧ d.dvd 40) else 0) + 39 = 1560 :=
by sorry

end numberOfSquaresWithNaturalVertices_l344_344319


namespace dynaco_shares_l344_344075

theorem dynaco_shares (M D : ℕ) 
  (h1 : M + D = 300)
  (h2 : 36 * M + 44 * D = 12000) : 
  D = 150 :=
sorry

end dynaco_shares_l344_344075


namespace max_sequence_term_at_2_l344_344803

def a_n (n : ℕ) : ℝ := (8 / 3) * (1 / 8)^n - 3 * (1 / 4)^n + (1 / 2)^n

theorem max_sequence_term_at_2 (n : ℕ) : a_n n ≤ a_n 2 :=
  sorry

end max_sequence_term_at_2_l344_344803


namespace magician_draws_two_cards_l344_344020

-- Define the total number of unique cards
def total_cards : ℕ := 15^2

-- Define the number of duplicate cards
def duplicate_cards : ℕ := 15

-- Define the number of ways to choose 2 cards from the duplicate cards
def choose_two_duplicates : ℕ := Nat.choose 15 2

-- Define the number of ways to choose 1 duplicate card and 1 non-duplicate card
def choose_one_duplicate_one_nonduplicate : ℕ := (15 * (total_cards - 15 - 14 - 14))

-- The main theorem to prove
theorem magician_draws_two_cards : choose_two_duplicates + choose_one_duplicate_one_nonduplicate = 2835 := by
  sorry

end magician_draws_two_cards_l344_344020
