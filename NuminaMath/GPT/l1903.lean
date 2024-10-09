import Mathlib

namespace find_total_amount_l1903_190303

-- Definitions according to the conditions
def is_proportion (a b c : ℚ) (p q r : ℚ) : Prop :=
  (a * q = b * p) ∧ (a * r = c * p) ∧ (b * r = c * q)

def total_amount (second_part : ℚ) (prop_total : ℚ) : ℚ :=
  second_part / (1/3) * prop_total

-- Main statement to be proved
theorem find_total_amount (second_part : ℚ) (p1 p2 p3 : ℚ)
  (h : is_proportion p1 p2 p3 (1/2 : ℚ) (1/3 : ℚ) (3/4 : ℚ))
  : second_part = 164.6315789473684 → total_amount second_part (19/12 : ℚ) = 65.16 :=
by
  sorry

end find_total_amount_l1903_190303


namespace h_eq_x_solution_l1903_190301

noncomputable def h (x : ℝ) : ℝ := (3 * ((x + 3) / 5) + 10)

theorem h_eq_x_solution (x : ℝ) (h_cond : ∀ y, h (5 * y - 3) = 3 * y + 10) : h x = x → x = 29.5 :=
by
  sorry

end h_eq_x_solution_l1903_190301


namespace initial_number_of_nurses_l1903_190351

theorem initial_number_of_nurses (N : ℕ) (initial_doctors : ℕ) (remaining_staff : ℕ) 
  (h1 : initial_doctors = 11) 
  (h2 : remaining_staff = 22) 
  (h3 : initial_doctors - 5 + N - 2 = remaining_staff) : N = 18 :=
by
  rw [h1, h2] at h3
  sorry

end initial_number_of_nurses_l1903_190351


namespace supremum_of_function_l1903_190372

theorem supremum_of_function : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
  (∃ M : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ M) ∧
    (∀ K : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ K) → M ≤ K) → M = -9 / 2) := 
sorry

end supremum_of_function_l1903_190372


namespace sum_of_distinct_integers_l1903_190370

noncomputable def distinct_integers (p q r s t : ℤ) : Prop :=
  (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ 
  (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ 
  (r ≠ s) ∧ (r ≠ t) ∧ 
  (s ≠ t)

theorem sum_of_distinct_integers
  (p q r s t : ℤ)
  (h_distinct : distinct_integers p q r s t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120) :
  p + q + r + s + t = 22 :=
  sorry

end sum_of_distinct_integers_l1903_190370


namespace tan_identity_l1903_190398

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l1903_190398


namespace smallest_prime_factor_2379_l1903_190382

-- Define the given number
def n : ℕ := 2379

-- Define the condition that 3 is a prime number.
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define the smallest prime factor
def smallest_prime_factor (n p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q, is_prime q → q ∣ n → p ≤ q)

-- The statement that 3 is the smallest prime factor of 2379
theorem smallest_prime_factor_2379 : smallest_prime_factor n 3 :=
sorry

end smallest_prime_factor_2379_l1903_190382


namespace total_money_given_to_children_l1903_190347

theorem total_money_given_to_children (B : ℕ) (x : ℕ) (total : ℕ) 
  (h1 : B = 300) 
  (h2 : x = B / 3) 
  (h3 : total = (2 * x) + (3 * x) + (4 * x)) : 
  total = 900 := 
by 
  sorry

end total_money_given_to_children_l1903_190347


namespace find_sum_of_numbers_l1903_190346

-- Define the problem using the given conditions
def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

-- The main theorem we want to prove
theorem find_sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) :
  sum_of_three_numbers a b c = 22 :=
by
  sorry

end find_sum_of_numbers_l1903_190346


namespace tan_add_pi_over_six_l1903_190353

theorem tan_add_pi_over_six (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + Real.pi / 6) = 5 + 2 * Real.sqrt 3 :=
sorry

end tan_add_pi_over_six_l1903_190353


namespace percentage_difference_l1903_190338

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.60 * x) (h2 : z = 0.60 * y) :
  abs ((z - x) / z * 100) = 4.17 :=
by
  sorry

end percentage_difference_l1903_190338


namespace regular_eqn_exists_l1903_190325

noncomputable def parametric_eqs (k : ℝ) : ℝ × ℝ :=
  (4 * k / (1 - k^2), 4 * k^2 / (1 - k^2))

theorem regular_eqn_exists (k : ℝ) (x y : ℝ) (h1 : x = 4 * k / (1 - k^2)) 
(h2 : y = 4 * k^2 / (1 - k^2)) : x^2 - y^2 - 4 * y = 0 :=
sorry

end regular_eqn_exists_l1903_190325


namespace student_A_final_score_l1903_190395

theorem student_A_final_score (total_questions : ℕ) (correct_responses : ℕ) 
  (h1 : total_questions = 100) (h2 : correct_responses = 93) : 
  correct_responses - 2 * (total_questions - correct_responses) = 79 :=
by
  rw [h1, h2]
  -- sorry

end student_A_final_score_l1903_190395


namespace original_number_divisible_by_3_l1903_190375

theorem original_number_divisible_by_3:
  ∃ (a b c d e f g h : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h) ∧
  (c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h) ∧
  (d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h) ∧
  (e ≠ f ∧ e ≠ g ∧ e ≠ h) ∧
  (f ≠ g ∧ f ≠ h) ∧
  (g ≠ h) ∧ 
  (a + b + c + b + d + e + f + e + g + d + h) % 3 = 0 :=
sorry

end original_number_divisible_by_3_l1903_190375


namespace parrots_false_statements_l1903_190362

theorem parrots_false_statements (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 140 ∧ 
    (∀ statements : ℕ → Prop, 
      (statements 0 = false) ∧ 
      (∀ i : ℕ, 1 ≤ i → i < n → 
          (statements i = true → 
            (∃ fp : ℕ, fp < i ∧ 7 * (fp + 1) > 10 * i)))) := 
by
  sorry

end parrots_false_statements_l1903_190362


namespace hotel_rooms_l1903_190323

theorem hotel_rooms (h₁ : ∀ R : ℕ, (∃ n : ℕ, n = R * 3) → (∃ m : ℕ, m = 2 * R * 3) → m = 60) : (∃ R : ℕ, R = 10) :=
by
  sorry

end hotel_rooms_l1903_190323


namespace dawn_annual_salary_l1903_190397

variable (M : ℝ)

theorem dawn_annual_salary (h1 : 0.10 * M = 400) : M * 12 = 48000 := by
  sorry

end dawn_annual_salary_l1903_190397


namespace quadratic_solution_range_l1903_190320

theorem quadratic_solution_range {x : ℝ} 
  (h : x^2 - 6 * x + 8 < 0) : 
  25 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 49 :=
sorry

end quadratic_solution_range_l1903_190320


namespace find_m_l1903_190356

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (-1, -1)
noncomputable def a_minus_b : ℝ × ℝ := (2, 3)
noncomputable def m_a_plus_b (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m - 1)

theorem find_m (m : ℝ) : (a_minus_b.1 * (m_a_plus_b m).1 + a_minus_b.2 * (m_a_plus_b m).2) = 0 → m = 5 / 8 := 
by
  sorry

end find_m_l1903_190356


namespace sum_of_acute_angles_l1903_190391

theorem sum_of_acute_angles (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : β > 0 ∧ β < π / 2) (h3: γ > 0 ∧ γ < π / 2) (h4 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) :
  (3 * π / 4) < α + β + γ ∧ α + β + γ < π :=
by
  sorry

end sum_of_acute_angles_l1903_190391


namespace product_of_dice_divisible_by_9_l1903_190333

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end product_of_dice_divisible_by_9_l1903_190333


namespace weight_of_D_l1903_190352

open Int

def weights (A B C D : Int) : Prop :=
  A < B ∧ B < C ∧ C < D ∧ 
  A + B = 45 ∧ A + C = 49 ∧ A + D = 55 ∧ 
  B + C = 54 ∧ B + D = 60 ∧ C + D = 64

theorem weight_of_D {A B C D : Int} (h : weights A B C D) : D = 35 := 
  by
    sorry

end weight_of_D_l1903_190352


namespace function_increasing_iff_m_eq_1_l1903_190357

theorem function_increasing_iff_m_eq_1 (m : ℝ) : 
  (m^2 - 4 * m + 4 = 1) ∧ (m^2 - 6 * m + 8 > 0) ↔ m = 1 :=
by {
  sorry
}

end function_increasing_iff_m_eq_1_l1903_190357


namespace g_x_minus_3_l1903_190309

def g (x : ℝ) : ℝ := x^2

theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6 * x + 9 :=
by
  -- This is where the proof would go
  sorry

end g_x_minus_3_l1903_190309


namespace solve_for_x_l1903_190374

theorem solve_for_x :
  (16^x * 16^x * 16^x * 4^(3 * x) = 64^(4 * x)) → x = 0 := by
  sorry

end solve_for_x_l1903_190374


namespace team_selection_l1903_190314

open Nat

theorem team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_to_choose := 5
  let girls_to_choose := 3
  choose boys boys_to_choose * choose girls girls_to_choose = 55440 :=
by
  sorry

end team_selection_l1903_190314


namespace find_y_l1903_190341

noncomputable def angle_ABC := 75
noncomputable def angle_BAC := 70
noncomputable def angle_CDE := 90
noncomputable def angle_BCA : ℝ := 180 - (angle_ABC + angle_BAC)
noncomputable def y : ℝ := 90 - angle_BCA

theorem find_y : y = 55 :=
by
  have h1: angle_BCA = 180 - (75 + 70) := rfl
  have h2: y = 90 - angle_BCA := rfl
  rw [h1] at h2
  exact h2.trans (by norm_num)

end find_y_l1903_190341


namespace min_value_of_a_sq_plus_b_sq_l1903_190316

theorem min_value_of_a_sq_plus_b_sq {a b t : ℝ} (h : 2 * a + 3 * b = t) :
  ∃ a b : ℝ, (2 * a + 3 * b = t) ∧ (a^2 + b^2 = (13 * t^2) / 169) :=
by
  sorry

end min_value_of_a_sq_plus_b_sq_l1903_190316


namespace group_membership_l1903_190336

theorem group_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 11 = 6) (h3 : 100 ≤ n ∧ n ≤ 200) :
  n = 116 ∨ n = 193 :=
sorry

end group_membership_l1903_190336


namespace annika_current_age_l1903_190365

-- Define the conditions
def hans_age_current : ℕ := 8
def hans_age_in_4_years : ℕ := hans_age_current + 4
def annika_age_in_4_years : ℕ := 3 * hans_age_in_4_years

-- lean statement to prove Annika's current age
theorem annika_current_age (A : ℕ) (hyp : A + 4 = annika_age_in_4_years) : A = 32 :=
by
  -- Skipping the proof
  sorry

end annika_current_age_l1903_190365


namespace least_number_when_increased_by_6_is_divisible_l1903_190306

theorem least_number_when_increased_by_6_is_divisible :
  ∃ n : ℕ, 
    (n + 6) % 24 = 0 ∧ 
    (n + 6) % 32 = 0 ∧ 
    (n + 6) % 36 = 0 ∧ 
    (n + 6) % 54 = 0 ∧ 
    n = 858 :=
by
  sorry

end least_number_when_increased_by_6_is_divisible_l1903_190306


namespace find_circle_center_l1903_190355

def circle_center (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 16 = 0

theorem find_circle_center (x y : ℝ) :
  circle_center x y ↔ (x, y) = (3, 4) :=
by
  sorry

end find_circle_center_l1903_190355


namespace find_larger_number_l1903_190366

theorem find_larger_number (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 :=
sorry

end find_larger_number_l1903_190366


namespace negative_y_implies_negative_y_is_positive_l1903_190390

theorem negative_y_implies_negative_y_is_positive (y : ℝ) (h : y < 0) : -y > 0 :=
sorry

end negative_y_implies_negative_y_is_positive_l1903_190390


namespace max_shapes_in_8x14_grid_l1903_190308

def unit_squares := 3
def grid_8x14 := 8 * 14
def grid_points (m n : ℕ) := (m + 1) * (n + 1)
def shapes_grid_points := 8
def max_shapes (total_points shape_points : ℕ) := total_points / shape_points

theorem max_shapes_in_8x14_grid 
  (m n : ℕ) (shape_points : ℕ) 
  (h1 : m = 8) (h2 : n = 14)
  (h3 : shape_points = 8) :
  max_shapes (grid_points m n) shape_points = 16 := by
  sorry

end max_shapes_in_8x14_grid_l1903_190308


namespace choose_5_from_12_l1903_190339

theorem choose_5_from_12 : Nat.choose 12 5 = 792 := by
  sorry

end choose_5_from_12_l1903_190339


namespace win_lottery_amount_l1903_190358

theorem win_lottery_amount (W : ℝ) (cond1 : W * 0.20 + 5 = 35) : W = 50 := by
  sorry

end win_lottery_amount_l1903_190358


namespace problem_l1903_190330

-- Definitions and conditions
variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of first n terms

-- Condition: a_n ≠ 0 for all n ∈ ℕ^*
axiom h1 : ∀ n : ℕ, n > 0 → a n ≠ 0

-- Condition: a_n * a_{n+1} = S_n
axiom h2 : ∀ n : ℕ, n > 0 → a n * a (n + 1) = S n

-- Given: S_1 = a_1
axiom h3 : S 1 = a 1

-- Given: S_2 = a_1 + a_2
axiom h4 : S 2 = a 1 + a 2

-- Prove: a_3 - a_1 = 1
theorem problem : a 3 - a 1 = 1 := by
  sorry

end problem_l1903_190330


namespace bill_has_six_times_more_nuts_l1903_190364

-- Definitions for the conditions
def sue_has_nuts : ℕ := 48
def harry_has_nuts (sueNuts : ℕ) : ℕ := 2 * sueNuts
def combined_nuts (harryNuts : ℕ) (billNuts : ℕ) : ℕ := harryNuts + billNuts
def bill_has_nuts (totalNuts : ℕ) (harryNuts : ℕ) : ℕ := totalNuts - harryNuts

-- Statement to prove
theorem bill_has_six_times_more_nuts :
  ∀ sueNuts billNuts harryNuts totalNuts,
    sueNuts = sue_has_nuts →
    harryNuts = harry_has_nuts sueNuts →
    totalNuts = 672 →
    combined_nuts harryNuts billNuts = totalNuts →
    billNuts = bill_has_nuts totalNuts harryNuts →
    billNuts = 6 * harryNuts :=
by
  intros sueNuts billNuts harryNuts totalNuts hsueNuts hharryNuts htotalNuts hcombinedNuts hbillNuts
  sorry

end bill_has_six_times_more_nuts_l1903_190364


namespace max_value_f_l1903_190373

def f (a x y : ℝ) : ℝ := a * x + y

theorem max_value_f (a : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : |x| + |y| ≤ 1) :
    f a x y ≤ 1 :=
by
  sorry

end max_value_f_l1903_190373


namespace negation_equiv_l1903_190381

noncomputable def negate_existential : Prop :=
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0

noncomputable def universal_negation : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0

theorem negation_equiv : negate_existential = universal_negation :=
by
  -- Proof to be filled in
  sorry

end negation_equiv_l1903_190381


namespace determine_k_values_l1903_190376

theorem determine_k_values (k : ℝ) :
  (∃ a b : ℝ, 3 * a ^ 2 + 6 * a + k = 0 ∧ 3 * b ^ 2 + 6 * b + k = 0 ∧ |a - b| = 1 / 2 * (a ^ 2 + b ^ 2)) → (k = 0 ∨ k = 12) :=
by
  sorry

end determine_k_values_l1903_190376


namespace f_2016_eq_neg1_l1903_190378

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_property : ∀ x y : ℝ, f x * f y = f (x + y) + f (x - y)

theorem f_2016_eq_neg1 : f 2016 = -1 := 
by 
  sorry

end f_2016_eq_neg1_l1903_190378


namespace smallest_lcm_value_l1903_190350

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end smallest_lcm_value_l1903_190350


namespace range_of_m_l1903_190319

theorem range_of_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ 2 * x + m - 3 = 0) : m < 3 :=
sorry

end range_of_m_l1903_190319


namespace percentage_female_on_duty_l1903_190310

-- Definitions as per conditions in the problem:
def total_on_duty : ℕ := 240
def female_on_duty := total_on_duty / 2 -- Half of those on duty are female
def total_female_officers : ℕ := 300
def percentage_of_something (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Statement of the problem to prove
theorem percentage_female_on_duty : percentage_of_something female_on_duty total_female_officers = 40 :=
by
  sorry

end percentage_female_on_duty_l1903_190310


namespace eight_digit_number_divisibility_l1903_190315

theorem eight_digit_number_divisibility (a b c d : ℕ) (Z : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
(h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (hZ : Z = 1001 * (1000 * a + 100 * b + 10 * c + d)) : 
  10001 ∣ Z := 
  by sorry

end eight_digit_number_divisibility_l1903_190315


namespace rectangle_area_difference_l1903_190380

theorem rectangle_area_difference :
  let area (l w : ℝ) := l * w
  let combined_area (l w : ℝ) := 2 * area l w
  combined_area 11 19 - combined_area 9.5 11 = 209 :=
by
  sorry

end rectangle_area_difference_l1903_190380


namespace parallelepiped_eq_l1903_190326

-- Definitions of the variables and conditions
variables (a b c u v w : ℝ)

-- Prove the identity given the conditions:
theorem parallelepiped_eq :
  u * v * w = a * v * w + b * u * w + c * u * v :=
sorry

end parallelepiped_eq_l1903_190326


namespace polynomial_coeff_divisible_by_5_l1903_190360

theorem polynomial_coeff_divisible_by_5
  (a b c : ℤ)
  (h : ∀ k : ℤ, (a * k^2 + b * k + c) % 5 = 0) :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 :=
by
  sorry

end polynomial_coeff_divisible_by_5_l1903_190360


namespace A_20_equals_17711_l1903_190313

def A : ℕ → ℕ
| 0     => 1  -- by definition, an alternating sequence on an empty set, counting empty sequence
| 1     => 2  -- base case
| 2     => 3  -- base case
| (n+3) => A (n+2) + A (n+1)

theorem A_20_equals_17711 : A 20 = 17711 := 
sorry

end A_20_equals_17711_l1903_190313


namespace p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l1903_190399

variable (x : ℝ)

def p := |x| = x
def q := x^2 + x ≥ 0

theorem p_implies_q : p x → q x :=
by sorry

theorem q_not_implies_p : q x → ¬p x :=
by sorry

theorem p_sufficient_but_not_necessary : (p x → q x) ∧ ¬(q x → p x) :=
by sorry

end p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l1903_190399


namespace susan_correct_guess_probability_l1903_190394

theorem susan_correct_guess_probability :
  (1 - (5/6)^6) = 31031/46656 := 
sorry

end susan_correct_guess_probability_l1903_190394


namespace eq_value_l1903_190307

theorem eq_value (x y : ℕ) (h1 : x - y = 9) (h2 : x = 9) : 3 ^ x * 4 ^ y = 19683 := by
  sorry

end eq_value_l1903_190307


namespace max_k_for_3_pow_11_as_sum_of_consec_integers_l1903_190389

theorem max_k_for_3_pow_11_as_sum_of_consec_integers :
  ∃ k n : ℕ, (3^11 = k * (2 * n + k + 1) / 2) ∧ (k = 486) :=
by
  sorry

end max_k_for_3_pow_11_as_sum_of_consec_integers_l1903_190389


namespace correct_fill_l1903_190385

/- Define the conditions and the statement in Lean 4 -/
def sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

/- The condition is that the phrase without the number has 21 characters -/
def initial_length : ℕ := 21

/- Define the term "тридцать две" as the correct number to fill the blank -/
def correct_number := "тридцать две"

/- The target phrase with the correct number filled in -/
def target_sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ " ++ correct_number ++ " БУКВЫ"

/- Prove that the correct number fills the blank correctly -/
theorem correct_fill :
  (String.length target_sentence = 38) :=
by
  /- Convert everything to string length and verify -/
  sorry

end correct_fill_l1903_190385


namespace compare_sums_l1903_190318

theorem compare_sums (a b c : ℝ) (h : a > b ∧ b > c) : a^2 * b + b^2 * c + c^2 * a > a * b^2 + b * c^2 + c * a^2 := by
  sorry

end compare_sums_l1903_190318


namespace total_money_shared_l1903_190367

theorem total_money_shared (rA rB rC : ℕ) (pA : ℕ) (total : ℕ) 
  (h_ratio : rA = 1 ∧ rB = 2 ∧ rC = 7) 
  (h_A_money : pA = 20) 
  (h_total : total = pA * rA + pA * rB + pA * rC) : 
  total = 200 := by 
  sorry

end total_money_shared_l1903_190367


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1903_190327

theorem solve_eq1 : ∀ (x : ℝ), x^2 - 5 * x = 0 ↔ x = 0 ∨ x = 5 :=
by sorry

theorem solve_eq2 : ∀ (x : ℝ), (2 * x + 1)^2 = 4 ↔ x = -3 / 2 ∨ x = 1 / 2 :=
by sorry

theorem solve_eq3 : ∀ (x : ℝ), x * (x - 1) + 3 * (x - 1) = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem solve_eq4 : ∀ (x : ℝ), x^2 - 2 * x - 8 = 0 ↔ x = -2 ∨ x = 4 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1903_190327


namespace find_intersection_find_range_of_a_l1903_190377

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | x < -2 ∨ (3 < x ∧ x < 4) }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 5 }

-- Proof Problem 1: Prove the intersection A ∩ B
theorem find_intersection : (A ∩ B) = { x : ℝ | 3 < x ∧ x ≤ 5 } := by
  sorry

-- Define the set C and the condition B ∩ C = B
def C (a : ℝ) : Set ℝ := { x : ℝ | x ≥ a }
def condition (a : ℝ) : Prop := B ∩ C a = B

-- Proof Problem 2: Find the range of a
theorem find_range_of_a : ∀ a : ℝ, condition a → a ≤ -3 := by
  sorry

end find_intersection_find_range_of_a_l1903_190377


namespace age_problem_l1903_190359

theorem age_problem
  (D M : ℕ)
  (h1 : M = D + 45)
  (h2 : M - 5 = 6 * (D - 5)) :
  D = 14 ∧ M = 59 := by
  sorry

end age_problem_l1903_190359


namespace star_result_l1903_190396

-- Define the operation star
def star (m n p q : ℚ) := (m * p) * (n / q)

-- Given values
def a := (5 : ℚ) / 9
def b := (10 : ℚ) / 6

-- Condition to check
theorem star_result : star 5 9 10 6 = 75 := by
  sorry

end star_result_l1903_190396


namespace length_of_AB_l1903_190324

theorem length_of_AB :
  ∃ (a b c d e : ℝ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
  (b - a = 5) ∧ -- AB = 5
  ((c - b) = 2 * (d - c)) ∧ -- bc = 2 * cd
  (d - e) = 4 ∧ -- de = 4
  (c - a) = 11 ∧ -- ac = 11
  (e - a) = 18 := -- ae = 18
by 
  sorry

end length_of_AB_l1903_190324


namespace find_angle_A_l1903_190344

theorem find_angle_A 
  (a b : ℝ) (A B : ℝ) 
  (h1 : b = 2 * a)
  (h2 : B = A + 60) : 
  A = 30 :=
  sorry

end find_angle_A_l1903_190344


namespace volume_Q4_l1903_190343

noncomputable def tetrahedron_sequence (n : ℕ) : ℝ :=
  -- Define the sequence recursively
  match n with
  | 0       => 1
  | (n + 1) => tetrahedron_sequence n + (4^n * (1 / 27)^(n + 1))

theorem volume_Q4 : tetrahedron_sequence 4 = 1.173832 :=
by
  sorry

end volume_Q4_l1903_190343


namespace product_of_midpoint_coordinates_l1903_190371

def x1 := 10
def y1 := -3
def x2 := 4
def y2 := 7

def midpoint_x := (x1 + x2) / 2
def midpoint_y := (y1 + y2) / 2

theorem product_of_midpoint_coordinates : 
  midpoint_x * midpoint_y = 14 :=
by
  sorry

end product_of_midpoint_coordinates_l1903_190371


namespace solve_for_x_l1903_190321

-- Define the given equation as a predicate
def equation (x: ℚ) : Prop := (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the problem in a Lean theorem
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -2 / 11 :=
by
  existsi -2 / 11
  constructor
  repeat { sorry }

end solve_for_x_l1903_190321


namespace intersection_S_T_l1903_190349

def S : Set ℝ := {x | x > -2}

def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_S_T : S ∩ T = {x | -2 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_S_T_l1903_190349


namespace find_8th_result_l1903_190363

theorem find_8th_result 
  (S_17 : ℕ := 17 * 24) 
  (S_7 : ℕ := 7 * 18) 
  (S_5_1 : ℕ := 5 * 23) 
  (S_5_2 : ℕ := 5 * 32) : 
  S_17 - S_7 - S_5_1 - S_5_2 = 7 := 
by
  sorry

end find_8th_result_l1903_190363


namespace polygon_sides_l1903_190386

theorem polygon_sides (n : ℕ) (h : n ≥ 3) (sum_angles : (n - 2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end polygon_sides_l1903_190386


namespace cost_price_computer_table_l1903_190369

variable (CP SP : ℝ)

theorem cost_price_computer_table (h1 : SP = 2 * CP) (h2 : SP = 1000) : CP = 500 := by
  sorry

end cost_price_computer_table_l1903_190369


namespace right_triangle_conditions_l1903_190387

theorem right_triangle_conditions (x y z h α β : ℝ) : 
  x - y = α → 
  z - h = β → 
  x^2 + y^2 = z^2 → 
  x * y = h * z → 
  β > α :=
by 
sorry

end right_triangle_conditions_l1903_190387


namespace increasing_function_range_l1903_190354

theorem increasing_function_range (k : ℝ) :
  (∀ x y : ℝ, x < y → (k + 2) * x + 1 < (k + 2) * y + 1) ↔ k > -2 :=
by
  sorry

end increasing_function_range_l1903_190354


namespace a1_plus_a9_l1903_190312

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a1_plus_a9 : (a 1) + (a 9) = 19 := by
  sorry

end a1_plus_a9_l1903_190312


namespace fencing_required_l1903_190304

theorem fencing_required (L W : ℝ) (h1 : L = 40) (h2 : L * W = 680) : 2 * W + L = 74 :=
by
  sorry

end fencing_required_l1903_190304


namespace factorization_identity_l1903_190331

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l1903_190331


namespace total_area_of_figure_l1903_190393

theorem total_area_of_figure :
  let h := 7
  let w1 := 6
  let h1 := 2
  let h2 := 3
  let h3 := 1
  let w2 := 5
  let a1 := h * w1
  let a2 := (h - h1) * (11 - 7)
  let a3 := (h - h1 - h2) * (11 - 7)
  let a4 := (15 - 11) * h3
  a1 + a2 + a3 + a4 = 74 :=
by
  sorry

end total_area_of_figure_l1903_190393


namespace min_distance_between_M_and_N_l1903_190384

noncomputable def f (x : ℝ) := Real.sin x + (1 / 6) * x^3
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_between_M_and_N :
  ∃ (x1 x2 : ℝ), x1 ≥ 0 ∧ x2 ≥ 0 ∧ f x1 = g x2 ∧ (x2 - x1 = 1) :=
sorry

end min_distance_between_M_and_N_l1903_190384


namespace f_divisible_by_27_l1903_190383

theorem f_divisible_by_27 (n : ℕ) : 27 ∣ (2^(2*n - 1) - 9 * n^2 + 21 * n - 14) :=
sorry

end f_divisible_by_27_l1903_190383


namespace polynomial_coeff_sum_l1903_190335

theorem polynomial_coeff_sum :
  let p := ((Polynomial.C 1 + Polynomial.X)^3 * (Polynomial.C 2 + Polynomial.X)^2)
  let a0 := p.coeff 0
  let a2 := p.coeff 2
  let a4 := p.coeff 4
  a4 + a2 + a0 = 36 := by 
  sorry

end polynomial_coeff_sum_l1903_190335


namespace smallest_circle_radius_eq_l1903_190361

open Real

-- Declaring the problem's conditions
def largestCircleRadius : ℝ := 10
def smallestCirclesCount : ℕ := 6
def congruentSmallerCirclesFitWithinLargerCircle (r : ℝ) : Prop :=
  3 * (2 * r) = 2 * largestCircleRadius

-- Stating the theorem to prove
theorem smallest_circle_radius_eq :
  ∃ r : ℝ, congruentSmallerCirclesFitWithinLargerCircle r ∧ r = 10 / 3 :=
by
  sorry

end smallest_circle_radius_eq_l1903_190361


namespace initial_amount_l1903_190368

theorem initial_amount (bread_price : ℝ) (bread_qty : ℝ) (pb_price : ℝ) (leftover : ℝ) :
  bread_price = 2.25 → bread_qty = 3 → pb_price = 2 → leftover = 5.25 →
  bread_qty * bread_price + pb_price + leftover = 14 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num


end initial_amount_l1903_190368


namespace algebraic_expression_value_l1903_190311

theorem algebraic_expression_value (x : ℝ) (h : (x^2 - x)^2 - 4 * (x^2 - x) - 12 = 0) : x^2 - x + 1 = 7 :=
sorry

end algebraic_expression_value_l1903_190311


namespace minimum_value_an_eq_neg28_at_n_eq_3_l1903_190322

noncomputable def seq_an (n : ℕ) : ℝ :=
  if n > 0 then (5 / 2) * n^2 - (13 / 2) * n
  else 0

noncomputable def delta_seq_an (n : ℕ) : ℝ := seq_an (n + 1) - seq_an n

noncomputable def delta2_seq_an (n : ℕ) : ℝ := delta_seq_an (n + 1) - delta_seq_an n

theorem minimum_value_an_eq_neg28_at_n_eq_3 : 
  ∃ (n : ℕ), n = 3 ∧ seq_an n = -28 :=
by
  sorry

end minimum_value_an_eq_neg28_at_n_eq_3_l1903_190322


namespace delta_delta_delta_l1903_190345

-- Define the function Δ
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Mathematical statement to be proved
theorem delta_delta_delta (x : ℝ) : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end delta_delta_delta_l1903_190345


namespace percent_increase_twice_eq_44_percent_l1903_190334

variable (P : ℝ) (x : ℝ)

theorem percent_increase_twice_eq_44_percent (h : P * (1 + x)^2 = P * 1.44) : x = 0.2 :=
by sorry

end percent_increase_twice_eq_44_percent_l1903_190334


namespace find_x_l1903_190392

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end find_x_l1903_190392


namespace product_plus_one_is_square_l1903_190328

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end product_plus_one_is_square_l1903_190328


namespace variance_of_white_balls_l1903_190342

section
variable (n : ℕ := 7) 
variable (p : ℚ := 3/7)

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls : binomial_variance n p = 12/7 :=
by
  sorry
end

end variance_of_white_balls_l1903_190342


namespace reduced_price_is_25_l1903_190337

def original_price (P : ℝ) (X : ℝ) (R : ℝ) : Prop :=
  R = 0.85 * P ∧ 
  500 = X * P ∧ 
  500 = (X + 3) * R

theorem reduced_price_is_25 (P X R : ℝ) (h : original_price P X R) :
  R = 25 :=
by
  sorry

end reduced_price_is_25_l1903_190337


namespace like_terms_solutions_l1903_190329

theorem like_terms_solutions (x y : ℤ) (h1 : 5 = 4 * x + 1) (h2 : 3 * y = 6) :
  x = 1 ∧ y = 2 := 
by 
  -- proof goes here
  sorry

end like_terms_solutions_l1903_190329


namespace find_line_equation_l1903_190302

-- Define point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the conditions
def passes_through_point_A (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq (-3) 4 = 0

def intercept_condition (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ line_eq (2 * a) 0 = 0 ∧ line_eq 0 a = 0

-- Define the equations of the line
def line1 (x y : ℝ) : ℝ := 3 * y + 4 * x
def line2 (x y : ℝ) : ℝ := 2 * x - y - 5

-- Statement of the problem
theorem find_line_equation : 
  (passes_through_point_A line1 ∧ intercept_condition line1) ∨
  (passes_through_point_A line2 ∧ intercept_condition line2) :=
sorry

end find_line_equation_l1903_190302


namespace prisha_other_number_l1903_190305

def prisha_numbers (a b : ℤ) : Prop :=
  3 * a + 2 * b = 105 ∧ (a = 15 ∨ b = 15)

theorem prisha_other_number (a b : ℤ) (h : prisha_numbers a b) : b = 30 :=
sorry

end prisha_other_number_l1903_190305


namespace Kenny_played_basketball_for_10_hours_l1903_190379

theorem Kenny_played_basketball_for_10_hours
  (played_basketball ran practiced_trumpet : ℕ)
  (H1 : practiced_trumpet = 40)
  (H2 : ran = 2 * played_basketball)
  (H3 : practiced_trumpet = 2 * ran) :
  played_basketball = 10 :=
by
  sorry

end Kenny_played_basketball_for_10_hours_l1903_190379


namespace expand_and_simplify_l1903_190340

theorem expand_and_simplify (x : ℝ) : 
  -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 :=
sorry

end expand_and_simplify_l1903_190340


namespace tetrahedron_paintings_l1903_190332

theorem tetrahedron_paintings (n : ℕ) (h : n ≥ 4) : 
  let term1 := (n - 1) * (n - 2) * (n - 3) / 12
  let term2 := (n - 1) * (n - 2) / 3
  let term3 := n - 1
  let term4 := 1
  2 * (term1 + term2 + term3) + n = 
  n * (term1 + term2 + term3 + term4) := by
{
  sorry
}

end tetrahedron_paintings_l1903_190332


namespace find_a1_l1903_190317

theorem find_a1 (S : ℕ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) :
  (∀ n : ℕ, S n = a1 * (2^n - 1)) → a 4 = 24 → 
  a 4 = S 4 - S 3 → 
  a1 = 3 :=
by
  sorry

end find_a1_l1903_190317


namespace candle_height_l1903_190388

variable (h d a b x : ℝ)

theorem candle_height (h d a b : ℝ) : x = h * (1 + d / (a + b)) :=
by
  sorry

end candle_height_l1903_190388


namespace average_salary_l1903_190348

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 14000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

theorem average_salary : (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8200 := 
  by 
    sorry

end average_salary_l1903_190348


namespace time_taken_by_A_l1903_190300

theorem time_taken_by_A (v_A v_B D t_A t_B : ℚ) (h1 : v_A / v_B = 3 / 4) 
  (h2 : t_A = t_B + 30) (h3 : t_A = D / v_A) (h4 : t_B = D / v_B) 
  : t_A = 120 := 
by 
  sorry

end time_taken_by_A_l1903_190300
