import Mathlib

namespace exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l1958_195871

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019 :
  ∃ N : ℕ, (N % 2019 = 0) ∧ ((sum_of_digits N) % 2019 = 0) :=
by 
  sorry

end exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l1958_195871


namespace max_value_expression_l1958_195816

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end max_value_expression_l1958_195816


namespace find_value_of_n_l1958_195846

theorem find_value_of_n (n : ℤ) : 
    n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 :=
by 
  intro h
  sorry

end find_value_of_n_l1958_195846


namespace no_positive_integer_n_satisfies_conditions_l1958_195858

theorem no_positive_integer_n_satisfies_conditions :
  ¬ ∃ (n : ℕ), (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_positive_integer_n_satisfies_conditions_l1958_195858


namespace angle_B_is_pi_over_3_l1958_195830

theorem angle_B_is_pi_over_3
  (A B C a b c : ℝ)
  (h1 : b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2)
  (h2 : 0 < B)
  (h3 : B < Real.pi)
  (h4 : 0 < A)
  (h5 : A < Real.pi)
  (h6 : 0 < C)
  (h7 : C < Real.pi) :
  B = Real.pi / 3 :=
by
  sorry

end angle_B_is_pi_over_3_l1958_195830


namespace problem_1_problem_2_l1958_195837

open Real

theorem problem_1
  (a b m n : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hm : m > 0)
  (hn : n > 0) :
  (m ^ 2 / a + n ^ 2 / b) ≥ ((m + n) ^ 2 / (a + b)) :=
sorry

theorem problem_2
  (x : ℝ)
  (hx1 : 0 < x)
  (hx2 : x < 1 / 2) :
  (2 / x + 9 / (1 - 2 * x)) ≥ 25 ∧ (2 / x + 9 / (1 - 2 * x)) = 25 ↔ x = 1 / 5 :=
sorry

end problem_1_problem_2_l1958_195837


namespace least_positive_t_l1958_195895

theorem least_positive_t
  (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (ht : ∃ t, 0 < t ∧ (∃ r, (Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧ 
                            Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
                            Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))))) :
  t = 6 :=
sorry

end least_positive_t_l1958_195895


namespace value_of_expression_l1958_195845

theorem value_of_expression (x y : ℝ) (h₁ : x * y = 3) (h₂ : x + y = 4) : x ^ 2 + y ^ 2 - 3 * x * y = 1 := 
by
  sorry

end value_of_expression_l1958_195845


namespace min_value_l1958_195852

/-- Given x and y are positive real numbers such that x + 3y = 2,
    the minimum value of (2x + y) / (xy) is 1/2 * (7 + 2 * sqrt 6). -/
theorem min_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = 2) :
  ∃ c : ℝ, c = (1/2) * (7 + 2 * Real.sqrt 6) ∧ ∀ (x y : ℝ), (0 < x) → (0 < y) → (x + 3 * y = 2) → ((2 * x + y) / (x * y)) ≥ c :=
sorry

end min_value_l1958_195852


namespace acquaintances_condition_l1958_195819

theorem acquaintances_condition (n : ℕ) (hn : n > 1) (acquainted : ℕ → ℕ → Prop) :
  (∀ X Y, acquainted X Y → acquainted Y X) ∧
  (∀ X, ¬acquainted X X) →
  (∀ n, n ≠ 2 → n ≠ 4 → ∃ (A B : ℕ), (∃ (C : ℕ), acquainted C A ∧ acquainted C B) ∨ (∃ (D : ℕ), ¬acquainted D A ∧ ¬acquainted D B)) :=
by
  intros
  sorry

end acquaintances_condition_l1958_195819


namespace Henry_age_ratio_l1958_195888

theorem Henry_age_ratio (A S H : ℕ)
  (hA : A = 15)
  (hS : S = 3 * A)
  (h_sum : A + S + H = 240) :
  H / S = 4 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end Henry_age_ratio_l1958_195888


namespace no_solution_exists_l1958_195825

theorem no_solution_exists :
  ¬ ∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) :=
by sorry

end no_solution_exists_l1958_195825


namespace find_a8_l1958_195814

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end find_a8_l1958_195814


namespace jimin_has_most_candy_left_l1958_195856

-- Definitions based on conditions
def fraction_jimin_ate := 1 / 9
def fraction_taehyung_ate := 1 / 3
def fraction_hoseok_ate := 1 / 6

-- The goal to prove
theorem jimin_has_most_candy_left : 
  (1 - fraction_jimin_ate) > (1 - fraction_taehyung_ate) ∧ (1 - fraction_jimin_ate) > (1 - fraction_hoseok_ate) :=
by
  -- The actual proof steps are omitted here.
  sorry

end jimin_has_most_candy_left_l1958_195856


namespace greatest_radius_l1958_195828

theorem greatest_radius (r : ℕ) (h : π * (r : ℝ)^2 < 50 * π) : r = 7 :=
sorry

end greatest_radius_l1958_195828


namespace seq_geq_4_l1958_195875

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem seq_geq_4 (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n ≥ 1 → a n ≥ 4 :=
sorry

end seq_geq_4_l1958_195875


namespace necessary_and_sufficient_condition_l1958_195813

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (∃ x : ℝ, f a x < 0) ↔ |a| > 2 :=
by
  sorry

end necessary_and_sufficient_condition_l1958_195813


namespace tessa_initial_apples_l1958_195820

theorem tessa_initial_apples (x : ℝ) (h : x + 5.0 - 4.0 = 11) : x = 10 :=
by
  sorry

end tessa_initial_apples_l1958_195820


namespace min_x8_x9_x10_eq_618_l1958_195810

theorem min_x8_x9_x10_eq_618 (x : ℕ → ℕ) (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 10 → x i < x j)
  (h2 : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10 = 2023) :
  x 8 + x 9 + x 10 = 618 :=
sorry

end min_x8_x9_x10_eq_618_l1958_195810


namespace part1_part2_l1958_195821

def my_mul (x y : Int) : Int :=
  if x = 0 then abs y
  else if y = 0 then abs x
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then abs x + abs y
  else - (abs x + abs y)

theorem part1 : my_mul (-15) (my_mul 3 0) = -18 := 
  by
  sorry

theorem part2 (a : Int) : 
  my_mul 3 a + a = 
  if a < 0 then 2 * a - 3 
  else if a = 0 then 3
  else 2 * a + 3 :=
  by
  sorry

end part1_part2_l1958_195821


namespace inverse_of_A_cubed_l1958_195891

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -2,  3],
    ![  0,  1]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3) = ![![ -8,  9],
                    ![  0,  1]] :=
by sorry

end inverse_of_A_cubed_l1958_195891


namespace units_digit_17_pow_28_l1958_195826

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end units_digit_17_pow_28_l1958_195826


namespace ratio_of_tshirts_l1958_195803

def spending_on_tshirts (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ) : Prop :=
  Lisa_tshirts = 40 ∧
  Lisa_jeans = Lisa_tshirts / 2 ∧
  Lisa_coats = 2 * Lisa_tshirts ∧
  Carly_jeans = 3 * Lisa_jeans ∧
  Carly_coats = Lisa_coats / 4 ∧
  Lisa_tshirts + Lisa_jeans + Lisa_coats + Carly_tshirts + Carly_jeans + Carly_coats = 230

theorem ratio_of_tshirts 
  (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ)
  (h : spending_on_tshirts Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats)
  : Carly_tshirts / Lisa_tshirts = 1 / 4 := 
sorry

end ratio_of_tshirts_l1958_195803


namespace largest_unachievable_score_l1958_195854

theorem largest_unachievable_score :
  ∀ (x y : ℕ), 3 * x + 7 * y ≠ 11 :=
by
  sorry

end largest_unachievable_score_l1958_195854


namespace units_digit_17_pow_2107_l1958_195863

theorem units_digit_17_pow_2107 : (17 ^ 2107) % 10 = 3 := by
  -- Definitions derived from conditions:
  -- 1. Powers of 17 have the same units digit as the corresponding powers of 7.
  -- 2. Units digits of powers of 7 cycle: 7, 9, 3, 1.
  -- 3. 2107 modulo 4 gives remainder 3.
  sorry

end units_digit_17_pow_2107_l1958_195863


namespace equation_of_circle_l1958_195806

variable (x y : ℝ)

def center_line : ℝ → ℝ := fun x => -4 * x
def tangent_line : ℝ → ℝ := fun x => 1 - x

def P : ℝ × ℝ := (3, -2)
def center_O : ℝ × ℝ := (1, -4)

theorem equation_of_circle :
  (x - 1)^2 + (y + 4)^2 = 8 :=
sorry

end equation_of_circle_l1958_195806


namespace local_min_f_at_2_implies_a_eq_2_l1958_195886

theorem local_min_f_at_2_implies_a_eq_2 (a : ℝ) : 
  (∃ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x * (x - a)^2) ∧ 
     (∀ f' : ℝ → ℝ, 
       (∀ x : ℝ, f' x = 3 * x^2 - 4 * a * x + a^2) ∧ 
       f' 2 = 0 ∧ 
       (∀ f'' : ℝ → ℝ, 
         (∀ x : ℝ, f'' x = 6 * x - 4 * a) ∧ 
         f'' 2 > 0
       )
     )
  ) → a = 2 :=
sorry

end local_min_f_at_2_implies_a_eq_2_l1958_195886


namespace parts_purchased_l1958_195800

noncomputable def price_per_part : ℕ := 80
noncomputable def total_paid_after_discount : ℕ := 439
noncomputable def total_discount : ℕ := 121

theorem parts_purchased : 
  ∃ n : ℕ, price_per_part * n - total_discount = total_paid_after_discount → n = 7 :=
by
  sorry

end parts_purchased_l1958_195800


namespace chalk_boxes_needed_l1958_195865

theorem chalk_boxes_needed (pieces_per_box : ℕ) (total_pieces : ℕ) (pieces_per_box_pos : pieces_per_box > 0) : 
  (total_pieces + pieces_per_box - 1) / pieces_per_box = 194 :=
by 
  let boxes_needed := (total_pieces + pieces_per_box - 1) / pieces_per_box
  have h: boxes_needed = 194 := sorry
  exact h

end chalk_boxes_needed_l1958_195865


namespace total_fencing_cost_is_correct_l1958_195860

-- Define the fencing cost per side
def costPerSide : Nat := 69

-- Define the number of sides for a square
def sidesOfSquare : Nat := 4

-- Define the total cost calculation for fencing the square
def totalCostOfFencing (costPerSide : Nat) (sidesOfSquare : Nat) := costPerSide * sidesOfSquare

-- Prove that for a given cost per side and number of sides, the total cost of fencing the square is 276 dollars
theorem total_fencing_cost_is_correct : totalCostOfFencing 69 4 = 276 :=
by
    -- Proof goes here
    sorry

end total_fencing_cost_is_correct_l1958_195860


namespace find_smaller_number_l1958_195815

-- Define the conditions
def condition1 (x y : ℤ) : Prop := x + y = 30
def condition2 (x y : ℤ) : Prop := x - y = 10

-- Define the theorem to prove the smaller number is 10
theorem find_smaller_number (x y : ℤ) (h1 : condition1 x y) (h2 : condition2 x y) : y = 10 := 
sorry

end find_smaller_number_l1958_195815


namespace jane_earnings_l1958_195864

def earnings_per_bulb : ℝ := 0.50
def tulip_bulbs : ℕ := 20
def iris_bulbs : ℕ := tulip_bulbs / 2
def daffodil_bulbs : ℕ := 30
def crocus_bulbs : ℕ := daffodil_bulbs * 3
def total_earnings : ℝ := (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs) * earnings_per_bulb

theorem jane_earnings : total_earnings = 75.0 := by
  sorry

end jane_earnings_l1958_195864


namespace probability_merlin_dismissed_l1958_195883

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l1958_195883


namespace domain_f_l1958_195832

def domain_of_f (x : ℝ) : Prop :=
  (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x < 4)

theorem domain_f :
  ∀ x, domain_of_f x ↔ (x ≥ 2 ∧ x < 4) ∧ x ≠ 3 :=
by
  sorry

end domain_f_l1958_195832


namespace john_profit_percentage_is_50_l1958_195899

noncomputable def profit_percentage
  (P : ℝ)  -- The sum of money John paid for purchasing 30 pens
  (recovered_amount : ℝ)  -- The amount John recovered when he sold 20 pens
  (condition : recovered_amount = P) -- Condition that John recovered the full amount P when he sold 20 pens
  : ℝ := 
  ((P / 20) - (P / 30)) / (P / 30) * 100

theorem john_profit_percentage_is_50
  (P : ℝ)
  (recovered_amount : ℝ)
  (condition : recovered_amount = P) :
  profit_percentage P recovered_amount condition = 50 := 
  by 
  sorry

end john_profit_percentage_is_50_l1958_195899


namespace Jack_has_18_dimes_l1958_195807

theorem Jack_has_18_dimes :
  ∃ d q : ℕ, (d = q + 3 ∧ 10 * d + 25 * q = 555) ∧ d = 18 :=
by
  sorry

end Jack_has_18_dimes_l1958_195807


namespace determine_c_l1958_195823

theorem determine_c (a c : ℝ) (h : (2 * a - 1) / -3 < - (c + 1) / -4) : c ≠ -1 ∧ (c > 0 ∨ c < 0) :=
by sorry

end determine_c_l1958_195823


namespace calculate_neg4_mul_three_div_two_l1958_195844

theorem calculate_neg4_mul_three_div_two : (-4) * (3 / 2) = -6 := 
by
  sorry

end calculate_neg4_mul_three_div_two_l1958_195844


namespace arithmetic_sequence_sum_l1958_195885

theorem arithmetic_sequence_sum {S : ℕ → ℤ} (m : ℕ) (hm : 0 < m)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l1958_195885


namespace inverse_proportion_range_l1958_195831

theorem inverse_proportion_range (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (y = (m + 5) / x) → ((x > 0 → y < 0) ∧ (x < 0 → y > 0))) →
  m < -5 :=
by
  intros h
  -- Skipping proof with sorry as specified
  sorry

end inverse_proportion_range_l1958_195831


namespace distance_home_gym_l1958_195822

theorem distance_home_gym 
  (v_WangLei v_ElderSister : ℕ)  -- speeds in meters per minute
  (d_meeting : ℕ)                -- distance in meters from the gym to the meeting point
  (t_gym : ℕ)                    -- time in minutes for the older sister to the gym
  (speed_diff : v_ElderSister = v_WangLei + 20)  -- speed difference
  (t_gym_reached : d_meeting / 2 = (25 * (v_WangLei + 20)) - d_meeting): 
  v_WangLei * t_gym = 1500 :=
by
  sorry

end distance_home_gym_l1958_195822


namespace scientific_notation_of_20000_l1958_195836

def number : ℕ := 20000

theorem scientific_notation_of_20000 : number = 2 * 10 ^ 4 :=
by
  sorry

end scientific_notation_of_20000_l1958_195836


namespace pages_per_day_l1958_195890

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (result : ℕ) :
  total_pages = 81 ∧ days = 3 → result = 27 :=
by
  sorry

end pages_per_day_l1958_195890


namespace new_person_age_l1958_195840

theorem new_person_age (T : ℕ) : 
  (T / 10) = ((T - 46 + A) / 10) + 3 → (A = 16) :=
by
  sorry

end new_person_age_l1958_195840


namespace remainder_7531_mod_11_is_5_l1958_195853

theorem remainder_7531_mod_11_is_5 :
  let n := 7531
  let m := 7 + 5 + 3 + 1
  n % 11 = 5 ∧ m % 11 = 5 :=
by
  let n := 7531
  let m := 7 + 5 + 3 + 1
  have h : n % 11 = m % 11 := sorry  -- by property of digits sum mod
  have hm : m % 11 = 5 := sorry      -- calculation
  exact ⟨h, hm⟩

end remainder_7531_mod_11_is_5_l1958_195853


namespace minyoung_money_l1958_195897

theorem minyoung_money (A M : ℕ) (h1 : M = 90 * A) (h2 : M = 60 * A + 270) : M = 810 :=
by 
  sorry

end minyoung_money_l1958_195897


namespace proof_problem_l1958_195802

theorem proof_problem 
  {a b c : ℝ} (h_cond : 1/a + 1/b + 1/c = 1/(a + b + c))
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (n : ℕ) :
  1/a^(2*n+1) + 1/b^(2*n+1) + 1/c^(2*n+1) = 1/(a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) :=
sorry

end proof_problem_l1958_195802


namespace vegetable_planting_methods_l1958_195868

theorem vegetable_planting_methods :
  let vegetables := ["cucumber", "cabbage", "rape", "lentils"]
  let cucumber := "cucumber"
  let other_vegetables := ["cabbage", "rape", "lentils"]
  let choose_2_out_of_3 := Nat.choose 3 2
  let arrangements := Nat.factorial 3
  total_methods = choose_2_out_of_3 * arrangements := by
  let total_methods := 3 * 6
  sorry

end vegetable_planting_methods_l1958_195868


namespace area_of_QCA_l1958_195869

noncomputable def area_of_triangle (x p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) : ℝ :=
  1 / 2 * x * (15 - p)

theorem area_of_QCA (x : ℝ) (p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) :
  area_of_triangle x p hx_pos hp_bounds = 1 / 2 * x * (15 - p) :=
sorry

end area_of_QCA_l1958_195869


namespace derivative_at_0_l1958_195874

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- State the theorem
theorem derivative_at_0 : f' 0 = 4 :=
by {
  -- Inserting sorry to skip the proof
  sorry
}

end derivative_at_0_l1958_195874


namespace find_S7_l1958_195811

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

axiom a1_def : a 1 = 1 / 2
axiom a_next_def : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * S n + 1
axiom S_def : ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem find_S7 : S 7 = 1457 / 2 := by
  sorry

end find_S7_l1958_195811


namespace g_triple_application_l1958_195834

def g (x : ℤ) : ℤ := 7 * x - 3

theorem g_triple_application : g (g (g 3)) = 858 := by
  sorry

end g_triple_application_l1958_195834


namespace cost_of_bananas_and_cantaloupe_l1958_195817

theorem cost_of_bananas_and_cantaloupe (a b c d h : ℚ) 
  (h1: a + b + c + d + h = 30)
  (h2: d = 4 * a)
  (h3: c = 2 * a - b) :
  b + c = 50 / 7 := 
sorry

end cost_of_bananas_and_cantaloupe_l1958_195817


namespace parabola_vertex_shift_l1958_195877

theorem parabola_vertex_shift
  (vertex_initial : ℝ × ℝ)
  (h₀ : vertex_initial = (0, 0))
  (move_left : ℝ)
  (move_up : ℝ)
  (h₁ : move_left = -2)
  (h₂ : move_up = 3):
  (vertex_initial.1 + move_left, vertex_initial.2 + move_up) = (-2, 3) :=
by
  sorry

end parabola_vertex_shift_l1958_195877


namespace div_by_prime_power_l1958_195850

theorem div_by_prime_power (p α x : ℕ) (hp : Nat.Prime p) (hpg : p > 2) (hα : α > 0) (t : ℤ) :
  (∃ k : ℤ, x^2 - 1 = k * p^α) ↔ (∃ t : ℤ, x = t * p^α + 1 ∨ x = t * p^α - 1) :=
sorry

end div_by_prime_power_l1958_195850


namespace sugar_needed_for_partial_recipe_l1958_195851

theorem sugar_needed_for_partial_recipe :
  let initial_sugar := 5 + 3/4
  let part := 3/4
  let needed_sugar := 4 + 5/16
  initial_sugar * part = needed_sugar := 
by 
  sorry

end sugar_needed_for_partial_recipe_l1958_195851


namespace parabola_conditions_l1958_195827

-- Definitions based on conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 4*x - 3 + a

def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def intersects_at_2_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Proof Problem Statement
theorem parabola_conditions (a : ℝ) :
  (passes_through (quadratic_function a) 0 1 → a = 4) ∧
  (intersects_at_2_points (quadratic_function a) → (a = 3 ∨ a = 7)) :=
by
  sorry

end parabola_conditions_l1958_195827


namespace perimeter_is_correct_l1958_195857

def side_length : ℕ := 2
def original_horizontal_segments : ℕ := 16
def original_vertical_segments : ℕ := 10

def horizontal_length : ℕ := original_horizontal_segments * side_length
def vertical_length : ℕ := original_vertical_segments * side_length

def perimeter : ℕ := horizontal_length + vertical_length

theorem perimeter_is_correct : perimeter = 52 :=
by 
  -- Proof goes here.
  sorry

end perimeter_is_correct_l1958_195857


namespace percent_receiving_speeding_tickets_l1958_195873

theorem percent_receiving_speeding_tickets
  (total_motorists : ℕ)
  (percent_exceeding_limit percent_exceeding_limit_without_ticket : ℚ)
  (h_exceeding_limit : percent_exceeding_limit = 0.5)
  (h_exceeding_limit_without_ticket : percent_exceeding_limit_without_ticket = 0.2) :
  let exceeding_limit := percent_exceeding_limit * total_motorists
  let without_tickets := percent_exceeding_limit_without_ticket * exceeding_limit
  let with_tickets := exceeding_limit - without_tickets
  (with_tickets / total_motorists) * 100 = 40 :=
by
  sorry

end percent_receiving_speeding_tickets_l1958_195873


namespace solve_ineqs_l1958_195870

theorem solve_ineqs (a x : ℝ) (h1 : |x - 2 * a| ≤ 3) (h2 : 0 < x + a ∧ x + a ≤ 4) 
  (ha : a = 3) (hx : x = 1) : 
  (|x - 2 * a| ≤ 3) ∧ (0 < x + a ∧ x + a ≤ 4) :=
by
  sorry

end solve_ineqs_l1958_195870


namespace largest_difference_l1958_195849

def U : ℕ := 2 * 1002 ^ 1003
def V : ℕ := 1002 ^ 1003
def W : ℕ := 1001 * 1002 ^ 1002
def X : ℕ := 2 * 1002 ^ 1002
def Y : ℕ := 1002 ^ 1002
def Z : ℕ := 1002 ^ 1001

theorem largest_difference : (U - V) = 1002 ^ 1003 ∧ 
  (V - W) = 1002 ^ 1002 ∧ 
  (W - X) = 999 * 1002 ^ 1002 ∧ 
  (X - Y) = 1002 ^ 1002 ∧ 
  (Y - Z) = 1001 * 1002 ^ 1001 ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 999 * 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1002 ^ 1002) ∧ 
  (1002 ^ 1003 > 1001 * 1002 ^ 1001) :=
by {
  sorry
}

end largest_difference_l1958_195849


namespace expand_polynomials_eq_l1958_195829

-- Define the polynomials P(z) and Q(z)
def P (z : ℝ) : ℝ := 3 * z^3 + 2 * z^2 - 4 * z + 1
def Q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the result polynomial R(z)
def R (z : ℝ) : ℝ := 12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2

-- State the theorem that proves P(z) * Q(z) = R(z)
theorem expand_polynomials_eq :
  ∀ (z : ℝ), (P z) * (Q z) = R z :=
by
  intros z
  sorry

end expand_polynomials_eq_l1958_195829


namespace problem_part_one_problem_part_two_l1958_195808

theorem problem_part_one : 23 - 17 - (-6) + (-16) = -4 :=
by
  sorry

theorem problem_part_two : 0 - 32 / ((-2)^3 - (-4)) = 8 :=
by
  sorry

end problem_part_one_problem_part_two_l1958_195808


namespace isosceles_triangle_perimeter_l1958_195884

theorem isosceles_triangle_perimeter (x y : ℝ) (h : |x - 4| + (y - 8)^2 = 0) :
  4 + 8 + 8 = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l1958_195884


namespace tara_dad_second_year_attendance_l1958_195801

theorem tara_dad_second_year_attendance :
  let games_played_per_year := 20
  let attendance_rate := 0.90
  let first_year_games_attended := attendance_rate * games_played_per_year
  let second_year_games_difference := 4
  first_year_games_attended - second_year_games_difference = 14 :=
by
  -- We skip the proof here
  sorry

end tara_dad_second_year_attendance_l1958_195801


namespace largest_distinct_arithmetic_sequence_number_l1958_195805

theorem largest_distinct_arithmetic_sequence_number :
  ∃ a b c d : ℕ, 
    (100 * a + 10 * b + c = 789) ∧ 
    (b - a = d) ∧ 
    (c - b = d) ∧ 
    (a ≠ b) ∧ 
    (b ≠ c) ∧ 
    (a ≠ c) ∧ 
    (a < 10) ∧ 
    (b < 10) ∧ 
    (c < 10) :=
sorry

end largest_distinct_arithmetic_sequence_number_l1958_195805


namespace employee_discount_percentage_l1958_195861

def wholesale_cost : ℝ := 200
def retail_markup : ℝ := 0.20
def employee_paid_price : ℝ := 228

theorem employee_discount_percentage :
  let retail_price := wholesale_cost * (1 + retail_markup)
  let discount := retail_price - employee_paid_price
  (discount / retail_price) * 100 = 5 := by
  sorry

end employee_discount_percentage_l1958_195861


namespace find_value_of_y_l1958_195876

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end find_value_of_y_l1958_195876


namespace average_temperature_is_95_l1958_195833

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l1958_195833


namespace sequence_term_general_sequence_sum_term_general_l1958_195887

theorem sequence_term_general (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S (n + 1) = 2 * S n + 1) →
  a 1 = 1 →
  (∀ n ≥ 1, a n = 2^(n-1)) :=
  sorry

theorem sequence_sum_term_general (na : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ k, na k = k * 2^(k-1)) →
  (∀ n, T n = (n - 1) * 2^n + 1) :=
  sorry

end sequence_term_general_sequence_sum_term_general_l1958_195887


namespace only_integer_solution_l1958_195838

theorem only_integer_solution (n : ℕ) (h1 : n > 1) (h2 : (2 * n + 1) % n ^ 2 = 0) : n = 3 := 
sorry

end only_integer_solution_l1958_195838


namespace bike_race_difference_l1958_195818

-- Define the conditions
def carlos_miles : ℕ := 70
def dana_miles : ℕ := 50
def time_period : ℕ := 5

-- State the theorem to prove the difference in miles biked
theorem bike_race_difference :
  carlos_miles - dana_miles = 20 := 
sorry

end bike_race_difference_l1958_195818


namespace next_elements_l1958_195867

-- Define the conditions and the question
def next_elements_in_sequence (n : ℕ) : String :=
  match n with
  | 1 => "О"  -- "Один"
  | 2 => "Д"  -- "Два"
  | 3 => "Т"  -- "Три"
  | 4 => "Ч"  -- "Четыре"
  | 5 => "П"  -- "Пять"
  | 6 => "Ш"  -- "Шесть"
  | 7 => "С"  -- "Семь"
  | 8 => "В"  -- "Восемь"
  | _ => "?"

theorem next_elements (n : ℕ) :
  next_elements_in_sequence 7 = "С" ∧ next_elements_in_sequence 8 = "В" := by
  sorry

end next_elements_l1958_195867


namespace maria_must_earn_l1958_195866

-- Define the given conditions
def retail_price : ℕ := 600
def maria_savings : ℕ := 120
def mother_contribution : ℕ := 250

-- Total amount Maria has from savings and her mother's contribution
def total_savings : ℕ := maria_savings + mother_contribution

-- Prove that Maria must earn $230 to be able to buy the bike
theorem maria_must_earn : 600 - total_savings = 230 :=
by sorry

end maria_must_earn_l1958_195866


namespace net_profit_start_year_better_investment_option_l1958_195841

-- Question 1: From which year does the developer start to make a net profit?
def investment_cost : ℕ := 81 -- in 10,000 yuan
def first_year_renovation_cost : ℕ := 1 -- in 10,000 yuan
def renovation_cost_increase : ℕ := 2 -- in 10,000 yuan per year
def annual_rental_income : ℕ := 30 -- in 10,000 yuan per year

theorem net_profit_start_year : ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, ¬ (annual_rental_income * m > investment_cost + m^2) :=
by sorry

-- Question 2: Which option is better: maximizing total profit or average annual profit?
def profit_function (n : ℕ) : ℤ := 30 * n - (81 + n^2)
def average_annual_profit (n : ℕ) : ℤ := (30 * n - (81 + n^2)) / n
def max_total_profit_year : ℕ := 15
def max_total_profit : ℤ := 144 -- in 10,000 yuan
def max_average_profit_year : ℕ := 9
def max_average_profit : ℤ := 12 -- in 10,000 yuan

theorem better_investment_option : (average_annual_profit max_average_profit_year) ≥ (profit_function max_total_profit_year) / max_total_profit_year :=
by sorry

end net_profit_start_year_better_investment_option_l1958_195841


namespace number_of_persons_in_second_group_l1958_195893

-- Definitions based on conditions
def total_man_hours_first_group : ℕ := 42 * 12 * 5

def total_man_hours_second_group (X : ℕ) : ℕ := X * 14 * 6

-- Theorem stating that the number of persons in the second group is 30, given the conditions
theorem number_of_persons_in_second_group (X : ℕ) : 
  total_man_hours_first_group = total_man_hours_second_group X → X = 30 :=
by
  sorry

end number_of_persons_in_second_group_l1958_195893


namespace count_ways_to_complete_20160_l1958_195881

noncomputable def waysToComplete : Nat :=
  let choices_for_last_digit := 5
  let choices_for_first_three_digits := 9^3
  choices_for_last_digit * choices_for_first_three_digits

theorem count_ways_to_complete_20160 (choices : Fin 9 → Fin 9) : waysToComplete = 3645 := by
  sorry

end count_ways_to_complete_20160_l1958_195881


namespace scientific_notation_361000000_l1958_195812

theorem scientific_notation_361000000 :
  361000000 = 3.61 * 10^8 :=
sorry

end scientific_notation_361000000_l1958_195812


namespace arithmetic_sequence_sum_property_l1958_195879

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)  -- sequence terms are real numbers
  (d : ℝ)      -- common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_condition : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 :=
sorry

end arithmetic_sequence_sum_property_l1958_195879


namespace total_bouncy_balls_l1958_195843

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def blue_packs := 6

def red_balls_per_pack := 12
def yellow_balls_per_pack := 10
def green_balls_per_pack := 14
def blue_balls_per_pack := 8

def total_red_balls := red_packs * red_balls_per_pack
def total_yellow_balls := yellow_packs * yellow_balls_per_pack
def total_green_balls := green_packs * green_balls_per_pack
def total_blue_balls := blue_packs * blue_balls_per_pack

def total_balls := total_red_balls + total_yellow_balls + total_green_balls + total_blue_balls

theorem total_bouncy_balls : total_balls = 232 :=
by
  -- calculation proof goes here
  sorry

end total_bouncy_balls_l1958_195843


namespace simplest_common_denominator_l1958_195880

theorem simplest_common_denominator (x a : ℕ) :
  let d1 := 3 * x
  let d2 := 6 * x^2
  lcm d1 d2 = 6 * x^2 := 
by
  let d1 := 3 * x
  let d2 := 6 * x^2
  show lcm d1 d2 = 6 * x^2
  sorry

end simplest_common_denominator_l1958_195880


namespace mod_equiv_l1958_195862

theorem mod_equiv (a b c d e : ℤ) (n : ℤ) (h1 : a = 101)
                                    (h2 : b = 15)
                                    (h3 : c = 7)
                                    (h4 : d = 9)
                                    (h5 : e = 5)
                                    (h6 : n = 17) :
  (a * b - c * d + e) % n = 7 := by
  sorry

end mod_equiv_l1958_195862


namespace ram_work_rate_l1958_195889

-- Definitions as given in the problem
variable (W : ℕ) -- Total work can be represented by some natural number W
variable (R M : ℕ) -- Raja's work rate and Ram's work rate, respectively

-- Given conditions
variable (combined_work_rate : R + M = W / 4)
variable (raja_work_rate : R = W / 12)

-- Theorem to be proven
theorem ram_work_rate (combined_work_rate : R + M = W / 4) (raja_work_rate : R = W / 12) : M = W / 6 := 
  sorry

end ram_work_rate_l1958_195889


namespace number_of_adult_tickets_l1958_195839

-- Let's define our conditions and the theorem to prove.
theorem number_of_adult_tickets (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) : A = 131 :=
by
  sorry

end number_of_adult_tickets_l1958_195839


namespace hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l1958_195898

-- Define a hexagon with legal points and triangles

structure Hexagon :=
  (A B C D E F : ℝ)

-- Legal point occurs when certain conditions on intersection between diagonals hold
def legal_point (h : Hexagon) (x : ℝ) (y : ℝ) : Prop :=
  -- Placeholder, we need to define the exact condition based on problem constraints.
  sorry

-- Function to check if a division is legal based on defined rules
def legal_triangle_division (h : Hexagon) (n : ℕ) : Prop :=
  -- Placeholder, this requires a definition based on how points and triangles are formed
  sorry

-- Prove the specific cases
theorem hexagon_six_legal_triangles (h : Hexagon) : legal_triangle_division h 6 :=
  sorry

theorem hexagon_ten_legal_triangles (h : Hexagon) : legal_triangle_division h 10 :=
  sorry

theorem hexagon_two_thousand_fourteen_legal_triangles (h : Hexagon)  : legal_triangle_division h 2014 :=
  sorry

end hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l1958_195898


namespace iterated_kernels_l1958_195809

noncomputable def K (x t : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < t then 
    x + t 
  else if t < x ∧ x ≤ 1 then 
    x - t 
  else 
    0

noncomputable def K1 (x t : ℝ) : ℝ := K x t

noncomputable def K2 (x t : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0

theorem iterated_kernels (x t : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  K1 x t = K x t ∧
  K2 x t = 
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0 := by
  sorry

end iterated_kernels_l1958_195809


namespace f_eight_l1958_195842

noncomputable def f : ℝ → ℝ := sorry -- Defining the function without implementing it here

axiom f_x_neg {x : ℝ} (hx : x < 0) : f x = Real.log (-x) + x
axiom f_symmetric {x : ℝ} (hx : -Real.exp 1 ≤ x ∧ x ≤ Real.exp 1) : f (-x) = -f x
axiom f_periodic {x : ℝ} (hx : x > 1) : f (x + 2) = f x

theorem f_eight : f 8 = 2 - Real.log 2 := 
by
  sorry

end f_eight_l1958_195842


namespace contrapositive_necessary_condition_l1958_195855

theorem contrapositive_necessary_condition (a b : Prop) (h : a → b) : ¬b → ¬a :=
by
  sorry

end contrapositive_necessary_condition_l1958_195855


namespace girls_with_rulers_l1958_195892

theorem girls_with_rulers 
  (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) 
  (total_girls : ℕ) (student_count : total_students = 50) 
  (ruler_count : students_with_rulers = 28) 
  (boys_with_set_squares_count : boys_with_set_squares = 14) 
  (girl_count : total_girls = 31) 
  : total_girls - (total_students - students_with_rulers - boys_with_set_squares) = 23 := 
by
  sorry

end girls_with_rulers_l1958_195892


namespace ants_harvest_remaining_sugar_l1958_195882

-- Define the initial conditions
def ants_removal_rate : ℕ := 4
def initial_sugar_amount : ℕ := 24
def hours_passed : ℕ := 3

-- Calculate the correct answer
def remaining_sugar (initial : ℕ) (rate : ℕ) (hours : ℕ) : ℕ :=
  initial - (rate * hours)

def additional_hours_needed (remaining_sugar : ℕ) (rate : ℕ) : ℕ :=
  remaining_sugar / rate

-- The specification of the proof problem
theorem ants_harvest_remaining_sugar :
  additional_hours_needed (remaining_sugar initial_sugar_amount ants_removal_rate hours_passed) ants_removal_rate = 3 :=
by
  -- Proof omitted
  sorry

end ants_harvest_remaining_sugar_l1958_195882


namespace work_done_is_halved_l1958_195896

theorem work_done_is_halved
  (A₁₂ A₃₄ : ℝ)
  (isothermal_process : ∀ (p V₁₂ V₃₄ : ℝ), V₁₂ = 2 * V₃₄ → p * V₁₂ = A₁₂ → p * V₃₄ = A₃₄) :
  A₃₄ = (1 / 2) * A₁₂ :=
sorry

end work_done_is_halved_l1958_195896


namespace quadratic_inequality_l1958_195894

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * a * x + a > 0) : 0 < a ∧ a < 1 :=
sorry

end quadratic_inequality_l1958_195894


namespace product_of_number_and_its_digits_sum_l1958_195872

theorem product_of_number_and_its_digits_sum :
  ∃ (n : ℕ), (n = 24 ∧ (n % 10) = ((n / 10) % 10) + 2) ∧ (n * (n % 10 + (n / 10) % 10) = 144) :=
by
  sorry

end product_of_number_and_its_digits_sum_l1958_195872


namespace coefficients_sum_l1958_195835

theorem coefficients_sum:
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (1+x)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  have h0 : a_0 = 1
  sorry -- proof when x=0
  have h1 : a_1 + a_2 + a_3 + a_4 + a_5 = 31
  sorry -- proof when x=1
  exact h1

end coefficients_sum_l1958_195835


namespace part1_solution_part2_solution_l1958_195847

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |x - 1|

theorem part1_solution :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x} :=
by
  sorry

theorem part2_solution (x0 : ℝ) :
  (∃ x0 : ℝ, ∀ t : ℝ, f x0 < |(x0 + t)| + |(t - x0)|) →
  ∀ m : ℝ, (f x0 < |m + t| + |t - m|) ↔ m ≠ 0 ∧ (|m| > 5 / 4) :=
by
  sorry

end part1_solution_part2_solution_l1958_195847


namespace remainder_eq_four_l1958_195848

theorem remainder_eq_four {x : ℤ} (h : x % 61 = 24) : x % 5 = 4 :=
sorry

end remainder_eq_four_l1958_195848


namespace simplify_expression_l1958_195859

variable (x y : ℤ) -- Assume x and y are integers for simplicity

theorem simplify_expression : (5 - 2 * x) - (8 - 6 * x + 3 * y) = -3 + 4 * x - 3 * y := by
  sorry

end simplify_expression_l1958_195859


namespace symmetric_point_coordinates_l1958_195878

theorem symmetric_point_coordinates 
  (k : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : ∀ k, k * (P.1) - P.2 + k - 2 = 0) 
  (P' : ℝ × ℝ) 
  (h2 : P'.1 + P'.2 = 3) 
  (h3 : 2 * P'.1^2 + 2 * P'.2^2 + 4 * P'.1 + 8 * P'.2 + 5 = 0) 
  (hP : P = (-1, -2)): 
  P' = (2, 1) := 
sorry

end symmetric_point_coordinates_l1958_195878


namespace complement_union_eq_ge_two_l1958_195824

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l1958_195824


namespace exists_nested_rectangles_l1958_195804

theorem exists_nested_rectangles (rectangles : ℕ × ℕ → Prop) :
  (∀ n m : ℕ, rectangles (n, m)) → ∃ (n1 m1 n2 m2 : ℕ), n1 ≤ n2 ∧ m1 ≤ m2 ∧ rectangles (n1, m1) ∧ rectangles (n2, m2) :=
by {
  sorry
}

end exists_nested_rectangles_l1958_195804
