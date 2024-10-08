import Mathlib

namespace zero_function_solution_l70_70914

theorem zero_function_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = f (x^3) + 3 * x^2 * f (x) * f (y) + 3 * (f (x) * f (y))^2 + y^6 * f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_solution_l70_70914


namespace zero_a_if_square_every_n_l70_70789

theorem zero_a_if_square_every_n (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 := 
sorry

end zero_a_if_square_every_n_l70_70789


namespace largest_reciprocal_l70_70796

theorem largest_reciprocal :
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  (1 / b) > (1 / a) ∧ (1 / b) > (1 / c) ∧ (1 / b) > (1 / d) ∧ (1 / b) > (1 / e) :=
by
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  sorry

end largest_reciprocal_l70_70796


namespace school_competition_students_l70_70357

theorem school_competition_students (n : ℤ)
  (h1 : 100 < n) 
  (h2 : n < 200) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2) 
  (h5 : n % 6 = 2) :
  n = 122 ∨ n = 182 :=
sorry

end school_competition_students_l70_70357


namespace expected_value_of_winnings_l70_70507

/-- A fair 6-sided die is rolled. If the roll is even, then you win the amount of dollars 
equal to the square of the number you roll. If the roll is odd, you win nothing. 
Prove that the expected value of your winnings is 28/3 dollars. -/
theorem expected_value_of_winnings : 
  (1 / 6) * (2^2 + 4^2 + 6^2) = 28 / 3 := by
sorry

end expected_value_of_winnings_l70_70507


namespace abs_sum_le_abs_one_plus_mul_l70_70588

theorem abs_sum_le_abs_one_plus_mul {x y : ℝ} (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  |x + y| ≤ |1 + x * y| :=
sorry

end abs_sum_le_abs_one_plus_mul_l70_70588


namespace find_Luisa_books_l70_70513

structure Books where
  Maddie : ℕ
  Amy : ℕ
  Amy_and_Luisa : ℕ
  Luisa : ℕ

theorem find_Luisa_books (L M A : ℕ) (hM : M = 15) (hA : A = 6) (hAL : L + A = M + 9) : L = 18 := by
  sorry

end find_Luisa_books_l70_70513


namespace work_completion_time_l70_70863

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 6
def work_rate_C : ℚ := 1 / 6

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the total work to be done (1 represents the whole job)
def total_work : ℚ := 1

-- Calculate the number of days to complete the work together
def days_to_complete_work : ℚ := total_work / combined_work_rate

theorem work_completion_time :
  work_rate_A = 1 / 6 ∧
  work_rate_B = 1 / 6 ∧
  work_rate_C = 1 / 6 →
  combined_work_rate = (work_rate_A + work_rate_B + work_rate_C) →
  days_to_complete_work = 2 :=
by
  intros
  sorry

end work_completion_time_l70_70863


namespace simplify_identity_l70_70463

theorem simplify_identity :
  ∀ θ : ℝ, θ = 160 → (1 / (Real.sqrt (1 + Real.tan (θ : ℝ) ^ 2))) = -Real.cos (θ : ℝ) :=
by
  intro θ h
  rw [h]
  sorry  

end simplify_identity_l70_70463


namespace chocolate_oranges_initial_l70_70387

theorem chocolate_oranges_initial (p_c p_o G n_c x : ℕ) 
  (h_candy_bar_price : p_c = 5) 
  (h_orange_price : p_o = 10) 
  (h_goal : G = 1000) 
  (h_candy_bars_sold : n_c = 160) 
  (h_equation : G = p_o * x + p_c * n_c) : 
  x = 20 := 
by
  sorry

end chocolate_oranges_initial_l70_70387


namespace common_ratio_of_geom_seq_l70_70100

-- Define the conditions: geometric sequence and the given equation
def is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_geom_seq
  (a : ℕ → ℝ)
  (h_geom : is_geom_seq a)
  (h_eq : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, is_geom_seq a ∧ q = 3 := 
sorry

end common_ratio_of_geom_seq_l70_70100


namespace measure_angle_A_l70_70800

theorem measure_angle_A (a b c : ℝ) (A B C : ℝ)
  (h1 : ∀ (Δ : Type), Δ → Δ → Δ)
  (h2 : a / Real.cos A = b / (2 * Real.cos B) ∧ 
        a / Real.cos A = c / (3 * Real.cos C))
  (h3 : A + B + C = Real.pi) : 
  A = Real.pi / 4 :=
sorry

end measure_angle_A_l70_70800


namespace cylinder_volume_l70_70781

-- Define the volume of the cone
def V_cone : ℝ := 18.84

-- Define the volume of the cylinder
def V_cylinder : ℝ := 3 * V_cone

-- Prove that the volume of the cylinder is 56.52 cubic meters
theorem cylinder_volume :
  V_cylinder = 56.52 := 
by 
  -- the proof will go here
  sorry

end cylinder_volume_l70_70781


namespace Alexis_mangoes_l70_70949

-- Define the variables for the number of mangoes each person has.
variable (A D Ash : ℕ)

-- Conditions given in the problem.
axiom h1 : A = 4 * (D + Ash)
axiom h2 : A + D + Ash = 75

-- The proof goal.
theorem Alexis_mangoes : A = 60 :=
sorry

end Alexis_mangoes_l70_70949


namespace geometric_sequence_find_a_n_l70_70560

variable {n m p : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom h1 : ∀ n, 2 * S (n + 1) - 3 * S n = 2 * a 1
axiom h2 : a 1 ≠ 0
axiom h3 : ∀ n, S (n + 1) = S n + a (n + 1)

-- Part (1)
theorem geometric_sequence : ∃ r, ∀ n, a (n + 1) = r * a n :=
sorry

-- Part (2)
axiom p_geq_3 : 3 ≤ p
axiom a1_pos : 0 < a 1
axiom a_p_pos : 0 < a p
axiom constraint1 : a 1 ≥ m ^ (p - 1)
axiom constraint2 : a p ≤ (m + 1) ^ (p - 1)

theorem find_a_n : ∀ n, a n = 2 ^ (p - 1) * (3 / 2) ^ (n - 1) :=
sorry

end geometric_sequence_find_a_n_l70_70560


namespace average_age_of_4_students_l70_70021

theorem average_age_of_4_students :
  let total_age_15 := 15 * 15
  let age_15th := 25
  let total_age_9 := 16 * 9
  (total_age_15 - total_age_9 - age_15th) / 4 = 14 :=
by
  sorry

end average_age_of_4_students_l70_70021


namespace inequality_true_l70_70306

variable (a b : ℝ)

theorem inequality_true (h : a > b ∧ b > 0) : (b^2 / a) < (a^2 / b) := by
  sorry

end inequality_true_l70_70306


namespace gumball_problem_l70_70419

theorem gumball_problem:
  ∀ (total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs: ℕ),
    total_gumballs = 45 →
    given_to_Todd = 4 →
    given_to_Alisha = 2 * given_to_Todd →
    remaining_gumballs = 6 →
    given_to_Todd + given_to_Alisha + given_to_Bobby + remaining_gumballs = total_gumballs →
    given_to_Bobby = 45 - 18 →
    4 * given_to_Alisha - given_to_Bobby = 5 :=
by
  intros total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs ht hTodd hAlisha hRemaining hSum hBobby
  rw [ht, hTodd] at *
  rw [hAlisha, hRemaining] at *
  sorry

end gumball_problem_l70_70419


namespace intersection_volume_is_zero_l70_70237

-- Definitions of the regions
def region1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 2
def region2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1

-- Main theorem stating the volume of their intersection
theorem intersection_volume_is_zero : 
  ∀ (x y z : ℝ), region1 x y z ∧ region2 x y z → (x = 0 ∧ y = 0 ∧ z = 2) := 
sorry

end intersection_volume_is_zero_l70_70237


namespace usual_time_to_school_l70_70923

variables (R T : ℝ)

theorem usual_time_to_school (h₁ : T > 0) (h₂ : R > 0) (h₃ : R / T = (5 / 4 * R) / (T - 4)) :
  T = 20 :=
by
  sorry

end usual_time_to_school_l70_70923


namespace min_value_of_f_l70_70092

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (2 * x / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 1 + 2 * Real.sqrt 2) := 
sorry

end min_value_of_f_l70_70092


namespace evaluate_expression_l70_70957

def numerator : ℤ :=
  (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1)

def denominator : ℤ :=
  (2 - 3) + (4 - 5) + (6 - 7) + (8 - 9) + (10 - 11) + 12

theorem evaluate_expression : numerator / denominator = 6 / 7 := by
  sorry

end evaluate_expression_l70_70957


namespace neznaika_is_wrong_l70_70813

theorem neznaika_is_wrong (avg_december avg_january : ℝ)
  (h_avg_dec : avg_december = 10)
  (h_avg_jan : avg_january = 5) : 
  ∃ (dec_days jan_days : ℕ), 
    (avg_december = (dec_days * 10 + (31 - dec_days) * 0) / 31) ∧
    (avg_january = (jan_days * 10 + (31 - jan_days) * 0) / 31) ∧
    jan_days > dec_days :=
by 
  sorry

end neznaika_is_wrong_l70_70813


namespace even_number_of_divisors_less_than_100_l70_70012

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l70_70012


namespace find_expression_value_l70_70047

theorem find_expression_value (x : ℝ) : 
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  have h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 := sorry
  exact h

end find_expression_value_l70_70047


namespace find_x_l70_70218

def op (a b : ℤ) : ℤ := -2 * a + b

theorem find_x (x : ℤ) (h : op x (-5) = 3) : x = -4 :=
by
  sorry

end find_x_l70_70218


namespace sequence_inequality_l70_70225

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (m n : ℕ)
  (h1 : a 1 = 21/16)
  (h2 : ∀ n ≥ 2, 2 * a n - 3 * a (n - 1) = 3 / 2^(n + 1))
  (h3 : m ≥ 2)
  (h4 : n ≤ m) :
  (a n + 3 / 2^(n + 3))^(1 / m) * (m - (2 / 3)^(n * (m - 1) / m)) < (m^2 - 1) / (m - n + 1) :=
sorry

end sequence_inequality_l70_70225


namespace total_oranges_for_philip_l70_70644

-- Define the initial conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def combined_oranges : ℕ := betty_oranges + bill_oranges
def frank_oranges : ℕ := 3 * combined_oranges
def seeds_planted : ℕ := 4 * frank_oranges
def successful_trees : ℕ := (3 / 4) * seeds_planted

-- The ratio of trees with different quantities of oranges
def ratio_parts : ℕ := 2 + 3 + 5
def trees_with_8_oranges : ℕ := (2 * successful_trees) / ratio_parts
def trees_with_10_oranges : ℕ := (3 * successful_trees) / ratio_parts
def trees_with_14_oranges : ℕ := (5 * successful_trees) / ratio_parts

-- Calculate the total number of oranges
def total_oranges : ℕ :=
  (trees_with_8_oranges * 8) +
  (trees_with_10_oranges * 10) +
  (trees_with_14_oranges * 14)

-- Statement to prove
theorem total_oranges_for_philip : total_oranges = 2798 :=
by
  sorry

end total_oranges_for_philip_l70_70644


namespace Q_subset_P_l70_70525

-- Definitions
def P : Set ℝ := {x : ℝ | x ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Statement to prove
theorem Q_subset_P : Q ⊆ P :=
sorry

end Q_subset_P_l70_70525


namespace train_speed_l70_70590

theorem train_speed 
  (t1 : ℝ) (t2 : ℝ) (L : ℝ) (v : ℝ) 
  (h1 : t1 = 12) 
  (h2 : t2 = 44) 
  (h3 : L = v * 12)
  (h4 : L + 320 = v * 44) : 
  (v * 3.6 = 36) :=
by
  sorry

end train_speed_l70_70590


namespace circumcircle_incircle_inequality_l70_70847

theorem circumcircle_incircle_inequality
  (a b : ℝ)
  (h_a : a = 16)
  (h_b : b = 11)
  (R r : ℝ)
  (triangle_inequality : ∀ c : ℝ, 5 < c ∧ c < 27) :
  R ≥ 2.2 * r := sorry

end circumcircle_incircle_inequality_l70_70847


namespace tom_dollars_more_than_jerry_l70_70270

theorem tom_dollars_more_than_jerry (total_slices : ℕ)
  (jerry_slices : ℕ)
  (tom_slices : ℕ)
  (plain_cost : ℕ)
  (pineapple_additional_cost : ℕ)
  (total_cost : ℕ)
  (cost_per_slice : ℚ)
  (cost_jerry : ℚ)
  (cost_tom : ℚ)
  (jerry_ate_plain : jerry_slices = 5)
  (tom_ate_pineapple : tom_slices = 5)
  (total_slices_10 : total_slices = 10)
  (plain_cost_10 : plain_cost = 10)
  (pineapple_additional_cost_3 : pineapple_additional_cost = 3)
  (total_cost_13 : total_cost = plain_cost + pineapple_additional_cost)
  (cost_per_slice_calc : cost_per_slice = total_cost / total_slices)
  (cost_jerry_calc : cost_jerry = cost_per_slice * jerry_slices)
  (cost_tom_calc : cost_tom = cost_per_slice * tom_slices) :
  cost_tom - cost_jerry = 0 := by
  sorry

end tom_dollars_more_than_jerry_l70_70270


namespace final_price_is_correct_l70_70137

/-- 
  The original price of a suit is $200.
-/
def original_price : ℝ := 200

/-- 
  The price increased by 25%, therefore the increase is 25% of the original price.
-/
def increase : ℝ := 0.25 * original_price

/-- 
  The new price after the price increase.
-/
def increased_price : ℝ := original_price + increase

/-- 
  After the increase, a 25% off coupon is applied.
-/
def discount : ℝ := 0.25 * increased_price

/-- 
  The final price consumers pay for the suit.
-/
def final_price : ℝ := increased_price - discount

/-- 
  Prove that the consumers paid $187.50 for the suit.
-/
theorem final_price_is_correct : final_price = 187.50 :=
by sorry

end final_price_is_correct_l70_70137


namespace calculate_expression_l70_70956

theorem calculate_expression : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := 
by
  sorry

end calculate_expression_l70_70956


namespace negation_proposition_l70_70981

theorem negation_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 ≥ 0) := 
by
  sorry

end negation_proposition_l70_70981


namespace carrie_is_left_with_50_l70_70972

-- Definitions for the conditions given in the problem
def amount_given : ℕ := 91
def cost_of_sweater : ℕ := 24
def cost_of_tshirt : ℕ := 6
def cost_of_shoes : ℕ := 11

-- Definition of the total amount spent
def total_spent : ℕ := cost_of_sweater + cost_of_tshirt + cost_of_shoes

-- Definition of the amount left
def amount_left : ℕ := amount_given - total_spent

-- The theorem we want to prove
theorem carrie_is_left_with_50 : amount_left = 50 :=
by
  have h1 : amount_given = 91 := rfl
  have h2 : total_spent = 41 := rfl
  have h3 : amount_left = 50 := rfl
  exact rfl

end carrie_is_left_with_50_l70_70972


namespace polynomial_roots_sum_l70_70510

noncomputable def roots (p : Polynomial ℚ) : Set ℚ := {r | p.eval r = 0}

theorem polynomial_roots_sum :
  ∀ a b c : ℚ, (a ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (b ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (c ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c = 8) →
  (a * b + a * c + b * c = 7) →
  (a * b * c = -3) →
  (a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) = 17 / 2) := by
    intros a b c ha hb hc hab habc hac sum_nums sum_prods prod_roots
    sorry

#check polynomial_roots_sum

end polynomial_roots_sum_l70_70510


namespace quadratic_function_m_value_l70_70795

theorem quadratic_function_m_value
  (m : ℝ)
  (h1 : m^2 - 7 = 2)
  (h2 : 3 - m ≠ 0) :
  m = -3 := by
  sorry

end quadratic_function_m_value_l70_70795


namespace harry_terry_difference_l70_70439

theorem harry_terry_difference :
  let H := 12 - (3 + 6)
  let T := 12 - 3 + 6 * 2
  H - T = -18 :=
by
  sorry

end harry_terry_difference_l70_70439


namespace dogs_in_kennel_l70_70198

variable (C D : ℕ)

-- definition of the ratio condition 
def ratio_condition : Prop :=
  C * 4 = 3 * D

-- definition of the difference condition
def difference_condition : Prop :=
  C = D - 8

theorem dogs_in_kennel (h1 : ratio_condition C D) (h2 : difference_condition C D) : D = 32 :=
by 
  -- proof steps go here
  sorry

end dogs_in_kennel_l70_70198


namespace jack_second_half_time_l70_70248

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l70_70248


namespace compute_2a_minus_b_l70_70860

noncomputable def conditions (a b : ℝ) : Prop :=
  a^3 - 12 * a^2 + 47 * a - 60 = 0 ∧
  -b^3 + 12 * b^2 - 47 * b + 180 = 0

theorem compute_2a_minus_b (a b : ℝ) (h : conditions a b) : 2 * a - b = 2 := 
  sorry

end compute_2a_minus_b_l70_70860


namespace determine_mass_l70_70673

noncomputable def mass_of_water 
  (P : ℝ) (t1 t2 : ℝ) (deltaT : ℝ) (cw : ℝ) : ℝ :=
  P * t1 / ((cw * deltaT) + ((cw * deltaT) / t2) * t1)

theorem determine_mass (P : ℝ) (t1 : ℝ) (deltaT : ℝ) (t2 : ℝ) (cw : ℝ) :
  P = 1000 → t1 = 120 → deltaT = 2 → t2 = 60 → cw = 4200 →
  mass_of_water P t1 deltaT t2 cw = 4.76 :=
by
  intros hP ht1 hdeltaT ht2 hcw
  sorry

end determine_mass_l70_70673


namespace B_joined_amount_l70_70192

theorem B_joined_amount (T : ℝ)
  (A_investment : ℝ := 45000)
  (B_time : ℝ := 2)
  (profit_ratio : ℝ := 2 / 1)
  (investment_ratio_rule : (A_investment * T) / (B_investment_amount * B_time) = profit_ratio) :
  B_investment_amount = 22500 :=
by
  sorry

end B_joined_amount_l70_70192


namespace age_difference_l70_70839

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end age_difference_l70_70839


namespace sum_of_products_of_roots_l70_70046

theorem sum_of_products_of_roots :
  ∀ (p q r : ℝ), (4 * p^3 - 6 * p^2 + 17 * p - 10 = 0) ∧ 
                 (4 * q^3 - 6 * q^2 + 17 * q - 10 = 0) ∧ 
                 (4 * r^3 - 6 * r^2 + 17 * r - 10 = 0) →
                 (p * q + q * r + r * p = 17 / 4) :=
by
  sorry

end sum_of_products_of_roots_l70_70046


namespace probability_divisible_by_256_l70_70503

theorem probability_divisible_by_256 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1000) :
  ((n * (n + 1) * (n + 2)) % 256 = 0) → (∃ p : ℚ, p = 0.006 ∧ (∃ k : ℕ, k ≤ 1000 ∧ (n = k))) :=
sorry

end probability_divisible_by_256_l70_70503


namespace rebecca_eggs_l70_70759

/-- Rebecca wants to split a collection of eggs into 4 groups. Each group will have 2 eggs. -/
def number_of_groups : Nat := 4

def eggs_per_group : Nat := 2

theorem rebecca_eggs : (number_of_groups * eggs_per_group) = 8 := by
  sorry

end rebecca_eggs_l70_70759


namespace abs_case_inequality_solution_l70_70716

theorem abs_case_inequality_solution (x : ℝ) :
  (|x + 1| + |x - 4| ≥ 7) ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 5) :=
by
  sorry

end abs_case_inequality_solution_l70_70716


namespace intersection_complement_l70_70484

open Set

-- Define sets A and B as provided in the conditions
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the theorem to prove the question is equal to the answer given the conditions
theorem intersection_complement : (A ∩ compl B) = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l70_70484


namespace xy_sum_value_l70_70451

theorem xy_sum_value (x y : ℝ) (h1 : x + Real.cos y = 1010) (h2 : x + 1010 * Real.sin y = 1009) (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (Real.pi / 2)) :
  x + y = 1010 + (Real.pi / 2) := 
by
  sorry

end xy_sum_value_l70_70451


namespace find_k_l70_70165

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 1/a + 2/b = 1) : k = 18 := by
  sorry

end find_k_l70_70165


namespace find_remainder_l70_70546

def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 1

theorem find_remainder : p 2 = 41 :=
by sorry

end find_remainder_l70_70546


namespace union_condition_intersection_condition_l70_70633

def setA : Set ℝ := {x | x^2 - 5 * x + 6 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 3}

theorem union_condition (a : ℝ) : setA ∪ setB a = setB a ↔ a < 2 := sorry

theorem intersection_condition (a : ℝ) : setA ∩ setB a = setB a ↔ a ≥ 2 := sorry

end union_condition_intersection_condition_l70_70633


namespace group_is_abelian_l70_70258

variable {G : Type} [Group G]
variable (e : G)
variable (h : ∀ x : G, x * x = e)

theorem group_is_abelian (a b : G) : a * b = b * a :=
sorry

end group_is_abelian_l70_70258


namespace coordinates_of_P_l70_70395

def point (x y : ℝ) := (x, y)

def A : (ℝ × ℝ) := point 1 1
def B : (ℝ × ℝ) := point 4 0

def vector_sub (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2)

def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem coordinates_of_P
  (P : ℝ × ℝ)
  (hP : vector_sub P A = scalar_mult 3 (vector_sub B P)) :
  P = (11 / 2, -1 / 2) :=
by
  sorry

end coordinates_of_P_l70_70395


namespace values_for_a_l70_70226

def has_two (A : Set ℤ) : Prop :=
  2 ∈ A

def candidate_values (a : ℤ) : Set ℤ :=
  {-2, 2 * a, a * a - a}

theorem values_for_a (a : ℤ) :
  has_two (candidate_values a) ↔ a = 1 ∨ a = 2 :=
by
  sorry

end values_for_a_l70_70226


namespace total_distance_l70_70172

theorem total_distance (x y : ℝ) (h1 : x * y = 18) :
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  D_total = y * x + y - x + 32 :=
by
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  sorry

end total_distance_l70_70172


namespace equivalent_single_discount_l70_70299

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.25
noncomputable def coupon_discount : ℝ := 0.10
noncomputable def final_price : ℝ := 33.75

theorem equivalent_single_discount :
  (1 - final_price / original_price) * 100 = 32.5 :=
by
  sorry

end equivalent_single_discount_l70_70299


namespace polynomial_simplify_l70_70648

theorem polynomial_simplify (x : ℝ) :
  (2*x^5 + 3*x^3 - 5*x^2 + 8*x - 6) + (-6*x^5 + x^3 + 4*x^2 - 8*x + 7) = -4*x^5 + 4*x^3 - x^2 + 1 :=
  sorry

end polynomial_simplify_l70_70648


namespace cube_root_simplification_l70_70496

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l70_70496


namespace student_exchanges_l70_70075

theorem student_exchanges (x : ℕ) : x * (x - 1) = 72 :=
sorry

end student_exchanges_l70_70075


namespace quadratic_inequality_solution_l70_70253

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 36 * x + 318 ≤ 0 ↔ 18 - Real.sqrt 6 ≤ x ∧ x ≤ 18 + Real.sqrt 6 := by
  sorry

end quadratic_inequality_solution_l70_70253


namespace boat_travel_time_l70_70653

noncomputable def total_travel_time (stream_speed boat_speed distance_AB : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance_BC := distance_AB / 2
  (distance_AB / downstream_speed) + (distance_BC / upstream_speed)

theorem boat_travel_time :
  total_travel_time 4 14 180 = 19 :=
by
  sorry

end boat_travel_time_l70_70653


namespace evaluate_expression_l70_70615

theorem evaluate_expression (a b : ℕ) :
  a = 3 ^ 1006 →
  b = 7 ^ 1007 →
  (a + b)^2 - (a - b)^2 = 42 * 10^x :=
by
  intro h1 h2
  sorry

end evaluate_expression_l70_70615


namespace sphere_surface_area_l70_70766

theorem sphere_surface_area (a b c : ℝ)
  (h1 : a * b * c = Real.sqrt 6)
  (h2 : a * b = Real.sqrt 2)
  (h3 : b * c = Real.sqrt 3) :
  4 * Real.pi * (Real.sqrt (a^2 + b^2 + c^2) / 2) ^ 2 = 6 * Real.pi :=
sorry

end sphere_surface_area_l70_70766


namespace number_of_third_year_students_to_sample_l70_70578

theorem number_of_third_year_students_to_sample
    (total_students : ℕ)
    (first_year_students : ℕ)
    (second_year_students : ℕ)
    (third_year_students : ℕ)
    (total_to_sample : ℕ)
    (h_total : total_students = 1200)
    (h_first : first_year_students = 480)
    (h_second : second_year_students = 420)
    (h_third : third_year_students = 300)
    (h_sample : total_to_sample = 100) :
    third_year_students * total_to_sample / total_students = 25 :=
by
  sorry

end number_of_third_year_students_to_sample_l70_70578


namespace astronaut_days_on_orbius_l70_70351

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end astronaut_days_on_orbius_l70_70351


namespace sum_of_bases_l70_70996

theorem sum_of_bases (S₁ S₂ G₁ G₂ : ℚ)
  (h₁ : G₁ = 4 * S₁ / (S₁^2 - 1) + 8 / (S₁^2 - 1))
  (h₂ : G₂ = 8 * S₁ / (S₁^2 - 1) + 4 / (S₁^2 - 1))
  (h₃ : G₁ = 3 * S₂ / (S₂^2 - 1) + 6 / (S₂^2 - 1))
  (h₄ : G₂ = 6 * S₂ / (S₂^2 - 1) + 3 / (S₂^2 - 1)) :
  S₁ + S₂ = 23 :=
by
  sorry

end sum_of_bases_l70_70996


namespace exists_rectangle_with_diagonal_zeros_and_ones_l70_70837

-- Define the problem parameters
def n := 2012
def table := Matrix (Fin n) (Fin n) (Fin 2)

-- Conditions
def row_contains_zero_and_one (m : table) (r : Fin n) : Prop :=
  ∃ c1 c2 : Fin n, m r c1 = 0 ∧ m r c2 = 1

def col_contains_zero_and_one (m : table) (c : Fin n) : Prop :=
  ∃ r1 r2 : Fin n, m r1 c = 0 ∧ m r2 c = 1

-- Problem statement
theorem exists_rectangle_with_diagonal_zeros_and_ones
  (m : table)
  (h_rows : ∀ r : Fin n, row_contains_zero_and_one m r)
  (h_cols : ∀ c : Fin n, col_contains_zero_and_one m c) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n),
    m r1 c1 = 0 ∧ m r2 c2 = 0 ∧ m r1 c2 = 1 ∧ m r2 c1 = 1 :=
sorry

end exists_rectangle_with_diagonal_zeros_and_ones_l70_70837


namespace total_recess_correct_l70_70571

-- Definitions based on the conditions
def base_recess : Int := 20
def recess_for_A (n : Int) : Int := n * 2
def recess_for_B (n : Int) : Int := n * 1
def recess_for_C (n : Int) : Int := n * 0
def recess_for_D (n : Int) : Int := -n * 1

def total_recess (a b c d : Int) : Int :=
  base_recess + recess_for_A a + recess_for_B b + recess_for_C c + recess_for_D d

-- The proof statement originally there would use these inputs
theorem total_recess_correct : total_recess 10 12 14 5 = 47 := by
  sorry

end total_recess_correct_l70_70571


namespace angle_in_second_quadrant_l70_70624

-- Definitions of conditions
def sin2_pos : Prop := Real.sin 2 > 0
def cos2_neg : Prop := Real.cos 2 < 0

-- Statement of the problem
theorem angle_in_second_quadrant (h1 : sin2_pos) (h2 : cos2_neg) : 
    (∃ α, 0 < α ∧ α < π ∧ P = (Real.sin α, Real.cos α)) :=
by
  sorry

end angle_in_second_quadrant_l70_70624


namespace share_of_a_l70_70848

variables {a b c d : ℝ}
variables {total : ℝ}

-- Conditions
def condition1 (a b c d : ℝ) := a = (3/5) * (b + c + d)
def condition2 (a b c d : ℝ) := b = (2/3) * (a + c + d)
def condition3 (a b c d : ℝ) := c = (4/7) * (a + b + d)
def total_distributed (a b c d : ℝ) := a + b + c + d = 1200

-- Theorem to prove
theorem share_of_a (a b c d : ℝ) (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 a b c d) (h4 : total_distributed a b c d) : 
  a = 247.5 :=
sorry

end share_of_a_l70_70848


namespace calculation_correct_l70_70662

theorem calculation_correct :
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 :=
by
  sorry

end calculation_correct_l70_70662


namespace intersection_eq_l70_70289

namespace Proof

universe u

-- Define the natural number set M
def M : Set ℕ := { x | x > 0 ∧ x < 6 }

-- Define the set N based on the condition |x-1| ≤ 2
def N : Set ℝ := { x | abs (x - 1) ≤ 2 }

-- Define the complement of N with respect to the real numbers
def ComplementN : Set ℝ := { x | x < -1 ∨ x > 3 }

-- Define the intersection of M and the complement of N
def IntersectMCompN : Set ℕ := { x | x ∈ M ∧ (x : ℝ) ∈ ComplementN }

-- Provide the theorem to be proved
theorem intersection_eq : IntersectMCompN = { 4, 5 } :=
by
  sorry

end Proof

end intersection_eq_l70_70289


namespace tables_made_this_month_l70_70840

theorem tables_made_this_month (T : ℕ) 
  (h1: ∀ t, t = T → t - 3 < t) 
  (h2 : T + (T - 3) = 17) :
  T = 10 := by
  sorry

end tables_made_this_month_l70_70840


namespace books_from_first_shop_l70_70999

theorem books_from_first_shop (x : ℕ) (h : (2080 : ℚ) / (x + 50) = 18.08695652173913) : x = 65 :=
by
  -- proof steps
  sorry

end books_from_first_shop_l70_70999


namespace right_triangle_shorter_leg_l70_70032

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l70_70032


namespace running_time_difference_l70_70277

theorem running_time_difference :
  ∀ (distance speed usual_speed : ℝ), 
  distance = 30 →
  usual_speed = 10 →
  speed = (distance / (usual_speed / 2)) - (distance / (usual_speed * 1.5)) →
  speed = 4 :=
by
  intros distance speed usual_speed hd hu hs
  sorry

end running_time_difference_l70_70277


namespace inequality_a2_b2_c2_geq_abc_l70_70210

theorem inequality_a2_b2_c2_geq_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_cond: a + b + c ≥ a * b * c) :
  a^2 + b^2 + c^2 ≥ a * b * c := 
sorry

end inequality_a2_b2_c2_geq_abc_l70_70210


namespace cone_rolls_path_l70_70338

theorem cone_rolls_path (r h m n : ℝ) (rotations : ℕ) 
  (h_rotations : rotations = 20)
  (h_ratio : h / r = 3 * Real.sqrt 133)
  (h_m : m = 3)
  (h_n : n = 133) : 
  m + n = 136 := 
by sorry

end cone_rolls_path_l70_70338


namespace paper_clips_collected_l70_70504

theorem paper_clips_collected (boxes paper_clips_per_box total_paper_clips : ℕ) 
  (h1 : boxes = 9) 
  (h2 : paper_clips_per_box = 9) 
  (h3 : total_paper_clips = boxes * paper_clips_per_box) : 
  total_paper_clips = 81 :=
by {
  sorry
}

end paper_clips_collected_l70_70504


namespace min_links_for_weights_l70_70060

def min_links_to_break (n : ℕ) : ℕ :=
  if n = 60 then 3 else sorry

theorem min_links_for_weights (n : ℕ) (h1 : n = 60) :
  min_links_to_break n = 3 :=
by
  rw [h1]
  trivial

end min_links_for_weights_l70_70060


namespace no_real_roots_quadratic_l70_70099

theorem no_real_roots_quadratic (k : ℝ) : 
  ∀ (x : ℝ), k * x^2 - 2 * x + 1 / 2 ≠ 0 → k > 2 :=
by 
  intro x h
  have h1 : (-2)^2 - 4 * k * (1/2) < 0 := sorry
  have h2 : 4 - 2 * k < 0 := sorry
  have h3 : 2 < k := sorry
  exact h3

end no_real_roots_quadratic_l70_70099


namespace skyscraper_anniversary_l70_70152

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end skyscraper_anniversary_l70_70152


namespace one_div_abs_z_eq_sqrt_two_l70_70126

open Complex

theorem one_div_abs_z_eq_sqrt_two (z : ℂ) (h : z = i / (1 - i)) : 1 / Complex.abs z = Real.sqrt 2 :=
by
  sorry

end one_div_abs_z_eq_sqrt_two_l70_70126


namespace student_correct_answers_l70_70135

-- Definitions based on the conditions
def total_questions : ℕ := 100
def score (correct incorrect : ℕ) : ℕ := correct - 2 * incorrect
def studentScore : ℕ := 73

-- Main theorem to prove
theorem student_correct_answers (C I : ℕ) (h1 : C + I = total_questions) (h2 : score C I = studentScore) : C = 91 :=
by
  sorry

end student_correct_answers_l70_70135


namespace triangle_side_length_l70_70617

theorem triangle_side_length {A B C : Type*} 
  (a b : ℝ) (S : ℝ) (ha : a = 4) (hb : b = 5) (hS : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end triangle_side_length_l70_70617


namespace inequality_squares_l70_70910

theorem inequality_squares (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end inequality_squares_l70_70910


namespace smallest_repeating_block_digits_l70_70433

theorem smallest_repeating_block_digits (n : ℕ) (d : ℕ) (hd_pos : d > 0) (hd_coprime : Nat.gcd n d = 1)
  (h_fraction : (n : ℚ) / d = 8 / 11) : n = 2 :=
by
  -- proof will go here
  sorry

end smallest_repeating_block_digits_l70_70433


namespace ratio_greater_than_two_ninths_l70_70881

-- Define the conditions
def M : ℕ := 8
def N : ℕ := 36

-- State the theorem
theorem ratio_greater_than_two_ninths : (M : ℚ) / (N : ℚ) > 2 / 9 := 
by {
    -- skipping the proof with sorry
    sorry
}

end ratio_greater_than_two_ninths_l70_70881


namespace largest_positive_real_root_bound_l70_70354

theorem largest_positive_real_root_bound (b0 b1 b2 : ℝ)
  (h_b0 : abs b0 ≤ 1) (h_b1 : abs b1 ≤ 1) (h_b2 : abs b2 ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b2 * r^2 + b1 * r + b0 = 0 ∧ 1.5 < r ∧ r < 2 := 
sorry

end largest_positive_real_root_bound_l70_70354


namespace rahim_average_price_l70_70231

/-- 
Rahim bought 40 books for Rs. 600 from one shop and 20 books for Rs. 240 from another.
What is the average price he paid per book?
-/
def books1 : ℕ := 40
def cost1 : ℕ := 600
def books2 : ℕ := 20
def cost2 : ℕ := 240
def totalBooks : ℕ := books1 + books2
def totalCost : ℕ := cost1 + cost2
def averagePricePerBook : ℕ := totalCost / totalBooks

theorem rahim_average_price :
  averagePricePerBook = 14 :=
by
  sorry

end rahim_average_price_l70_70231


namespace min_ab_value_l70_70516

variable (a b : ℝ)

theorem min_ab_value (h1 : a > -1) (h2 : b > -2) (h3 : (a+1) * (b+2) = 16) : a + b ≥ 5 :=
by
  sorry

end min_ab_value_l70_70516


namespace calc1_calc2_l70_70485

theorem calc1 : (-2) * (-1/8) = 1/4 :=
by
  sorry

theorem calc2 : (-5) / (6/5) = -25/6 :=
by
  sorry

end calc1_calc2_l70_70485


namespace value_of_x_l70_70017

theorem value_of_x (m n : ℝ) (z x : ℝ) (hz : z ≠ 0) (hx : x = m * (n / z) ^ 3) (hconst : 5 * (16 ^ 3) = m * (n ^ 3)) (hz_const : z = 64) : x = 5 / 64 :=
by
  -- proof omitted
  sorry

end value_of_x_l70_70017


namespace log9_6_eq_mn_over_2_l70_70215

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log9_6_eq_mn_over_2
  (m n : ℝ)
  (h1 : log_base 7 4 = m)
  (h2 : log_base 4 6 = n) : 
  log_base 9 6 = (m * n) / 2 := by
  sorry

end log9_6_eq_mn_over_2_l70_70215


namespace arithmetic_sequence_properties_l70_70901

theorem arithmetic_sequence_properties
    (a_1 : ℕ)
    (d : ℕ)
    (sequence : Fin 240 → ℕ)
    (h1 : ∀ n, sequence n = a_1 + n * d)
    (h2 : sequence 0 = a_1)
    (h3 : 1 ≤ a_1 ∧ a_1 ≤ 9)
    (h4 : ∃ n₁, sequence n₁ = 100)
    (h5 : ∃ n₂, sequence n₂ = 3103) :
  (a_1 = 9 ∧ d = 13) ∨ (a_1 = 1 ∧ d = 33) ∨ (a_1 = 9 ∧ d = 91) :=
sorry

end arithmetic_sequence_properties_l70_70901


namespace ratio_of_eggs_l70_70702

/-- Megan initially had 24 eggs (12 from the store and 12 from her neighbor). She used 6 eggs in total (2 for an omelet and 4 for baking). She set aside 9 eggs for three meals (3 eggs per meal). Finally, Megan divided the remaining 9 eggs by giving 9 to her aunt and keeping 9 for herself. The ratio of the eggs she gave to her aunt to the eggs she kept is 1:1. -/
theorem ratio_of_eggs
  (eggs_bought : ℕ)
  (eggs_from_neighbor : ℕ)
  (eggs_omelet : ℕ)
  (eggs_baking : ℕ)
  (meals : ℕ)
  (eggs_per_meal : ℕ)
  (aunt_got : ℕ)
  (kept_for_meals : ℕ)
  (initial_eggs := eggs_bought + eggs_from_neighbor)
  (used_eggs := eggs_omelet + eggs_baking)
  (remaining_eggs := initial_eggs - used_eggs)
  (assigned_eggs := meals * eggs_per_meal)
  (final_eggs := remaining_eggs - assigned_eggs)
  (ratio : ℚ := aunt_got / kept_for_meals) :
  eggs_bought = 12 ∧
  eggs_from_neighbor = 12 ∧
  eggs_omelet = 2 ∧
  eggs_baking = 4 ∧
  meals = 3 ∧
  eggs_per_meal = 3 ∧
  aunt_got = 9 ∧
  kept_for_meals = assigned_eggs →
  ratio = 1 := by
  sorry

end ratio_of_eggs_l70_70702


namespace total_pears_l70_70359

theorem total_pears (S P C : ℕ) (hS : S = 20) (hP : P = (S - S / 2)) (hC : C = (P + P / 5)) : S + P + C = 42 :=
by
  -- We state the theorem with the given conditions and the goal of proving S + P + C = 42.
  sorry

end total_pears_l70_70359


namespace shaded_region_area_l70_70467

noncomputable def shaded_area (π_approx : ℝ := 3.14) (r : ℝ := 1) : ℝ :=
  let square_area := (r / Real.sqrt 2) ^ 2
  let quarter_circle_area := (π_approx * r ^ 2) / 4
  quarter_circle_area - square_area

theorem shaded_region_area :
  shaded_area = 0.285 :=
by
  sorry

end shaded_region_area_l70_70467


namespace neg_ex_iff_forall_geq_0_l70_70027

theorem neg_ex_iff_forall_geq_0 :
  ¬(∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≥ 0 :=
by
  sorry

end neg_ex_iff_forall_geq_0_l70_70027


namespace c_share_of_profit_l70_70890

theorem c_share_of_profit
  (a_investment : ℝ)
  (b_investment : ℝ)
  (c_investment : ℝ)
  (total_profit : ℝ)
  (ha : a_investment = 30000)
  (hb : b_investment = 45000)
  (hc : c_investment = 50000)
  (hp : total_profit = 90000) :
  (c_investment / (a_investment + b_investment + c_investment)) * total_profit = 36000 := 
by
  sorry

end c_share_of_profit_l70_70890


namespace roots_range_l70_70085

theorem roots_range (b : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + b = 0 → 0 < x) ↔ 0 < b ∧ b ≤ 1 :=
sorry

end roots_range_l70_70085


namespace cube_side_length_l70_70421

-- Given conditions for the problem
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- Theorem statement
theorem cube_side_length (h : surface_area a = 864) : a = 12 :=
by
  sorry

end cube_side_length_l70_70421


namespace not_divisible_by_44_l70_70040

theorem not_divisible_by_44 (k : ℤ) (n : ℤ) (h1 : n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) : ¬ (44 ∣ n) :=
sorry

end not_divisible_by_44_l70_70040


namespace pull_ups_of_fourth_student_l70_70462

theorem pull_ups_of_fourth_student 
  (avg_pullups : ℕ) 
  (num_students : ℕ) 
  (pullups_first : ℕ) 
  (pullups_second : ℕ) 
  (pullups_third : ℕ) 
  (pullups_fifth : ℕ) 
  (H_avg : avg_pullups = 10) 
  (H_students : num_students = 5) 
  (H_first : pullups_first = 9) 
  (H_second : pullups_second = 12) 
  (H_third : pullups_third = 9) 
  (H_fifth : pullups_fifth = 8) : 
  ∃ (pullups_fourth : ℕ), pullups_fourth = 12 := by
  sorry

end pull_ups_of_fourth_student_l70_70462


namespace tan_alpha_not_unique_l70_70970

theorem tan_alpha_not_unique (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi) (h3 : (Real.sin α)^2 + Real.cos (2 * α) = 1) :
  ¬(∃ t : ℝ, Real.tan α = t) :=
by
  sorry

end tan_alpha_not_unique_l70_70970


namespace ship_total_distance_l70_70665

variables {v_r : ℝ} {t_total : ℝ} {a d : ℝ}

-- Given conditions
def conditions (v_r t_total a d : ℝ) :=
  v_r = 2 ∧ t_total = 3.2 ∧
  (∃ v : ℝ, ∀ t : ℝ, t = a/(v + v_r) + (a + d)/v + (a + 2*d)/(v - v_r)) 

-- The main statement to prove
theorem ship_total_distance (d_total : ℝ) :
  conditions 2 3.2 a d → d_total = 102 :=
by
  sorry

end ship_total_distance_l70_70665


namespace min_a_squared_plus_b_squared_l70_70558

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 := 
sorry

end min_a_squared_plus_b_squared_l70_70558


namespace correct_average_of_20_numbers_l70_70784

theorem correct_average_of_20_numbers 
  (incorrect_avg : ℕ) 
  (n : ℕ) 
  (incorrectly_read : ℕ) 
  (correction : ℕ) 
  (a b c d e f g h i j : ℤ) 
  (sum_a_b_c_d_e : ℤ)
  (sum_f_g_h_i_j : ℤ)
  (incorrect_sum : ℤ)
  (correction_sum : ℤ) 
  (corrected_sum : ℤ)
  (correct_avg : ℤ) : 
  incorrect_avg = 35 ∧ 
  n = 20 ∧ 
  incorrectly_read = 5 ∧ 
  correction = 136 ∧ 
  a = 90 ∧ b = 73 ∧ c = 85 ∧ d = -45 ∧ e = 64 ∧ 
  f = 45 ∧ g = 36 ∧ h = 42 ∧ i = -27 ∧ j = 35 ∧ 
  sum_a_b_c_d_e = a + b + c + d + e ∧
  sum_f_g_h_i_j = f + g + h + i + j ∧
  incorrect_sum = incorrect_avg * n ∧ 
  correction_sum = sum_a_b_c_d_e - sum_f_g_h_i_j ∧ 
  corrected_sum = incorrect_sum + correction_sum → correct_avg = corrected_sum / n := 
  by sorry

end correct_average_of_20_numbers_l70_70784


namespace next_term_geometric_sequence_l70_70158

theorem next_term_geometric_sequence (x : ℝ) (r : ℝ) (a₀ a₃ next_term : ℝ)
    (h1 : a₀ = 2)
    (h2 : r = 3 * x)
    (h3 : a₃ = 54 * x^3)
    (h4 : next_term = a₃ * r) :
    next_term = 162 * x^4 := by
  sorry

end next_term_geometric_sequence_l70_70158


namespace mary_flour_indeterminate_l70_70151

theorem mary_flour_indeterminate 
  (sugar : ℕ) (flour : ℕ) (salt : ℕ) (needed_sugar_more : ℕ) 
  (h_sugar : sugar = 11) (h_flour : flour = 6)
  (h_salt : salt = 9) (h_condition : needed_sugar_more = 2) :
  ∃ (current_flour : ℕ), current_flour ≠ current_flour :=
by
  sorry

end mary_flour_indeterminate_l70_70151


namespace calculate_expression_l70_70937

theorem calculate_expression : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end calculate_expression_l70_70937


namespace binomial_product_l70_70610

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l70_70610


namespace systematic_sampling_method_l70_70608

theorem systematic_sampling_method (k : ℕ) (n : ℕ) 
  (invoice_stubs : ℕ → ℕ) : 
  (k > 0) → 
  (n > 0) → 
  (invoice_stubs 15 = k) → 
  (∀ i : ℕ, invoice_stubs (15 + i * 50) = k + i * 50)
  → (sampling_method = "systematic") :=
by 
  intro h1 h2 h3 h4
  sorry

end systematic_sampling_method_l70_70608


namespace cheburashkas_erased_l70_70105

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l70_70105


namespace milan_billed_minutes_l70_70549

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l70_70549


namespace solve_for_n_l70_70950

theorem solve_for_n : 
  (∃ n : ℤ, (1 / (n + 2) + 2 / (n + 2) + (n + 1) / (n + 2) = 3)) ↔ n = -1 :=
sorry

end solve_for_n_l70_70950


namespace opposite_of_4_l70_70144

theorem opposite_of_4 : ∃ x, 4 + x = 0 ∧ x = -4 :=
by sorry

end opposite_of_4_l70_70144


namespace sqrt_product_simplifies_l70_70011

theorem sqrt_product_simplifies (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := 
by
  sorry

end sqrt_product_simplifies_l70_70011


namespace hexagon_coloring_l70_70531

-- Definitions based on conditions
variable (A B C D E F : ℕ)
variable (color : ℕ → ℕ)
variable (v1 v2 : ℕ)

-- The question is about the number of different colorings
theorem hexagon_coloring (h_distinct : ∀ (x y : ℕ), x ≠ y → color x ≠ color y) 
    (h_colors : ∀ (x : ℕ), x ∈ [A, B, C, D, E, F] → 0 < color x ∧ color x < 5) :
    4 * 3 * 3 * 3 * 3 * 3 = 972 :=
by
  sorry

end hexagon_coloring_l70_70531


namespace tetrahedron_painting_l70_70219

theorem tetrahedron_painting (unique_coloring_per_face : ∀ f : Fin 4, ∃ c : Fin 4, True)
  (rotation_identity : ∀ f g : Fin 4, (f = g → unique_coloring_per_face f = unique_coloring_per_face g))
  : (number_of_distinct_paintings : ℕ) = 2 :=
sorry

end tetrahedron_painting_l70_70219


namespace shaina_chocolate_l70_70260

-- Define the conditions
def total_chocolate : ℚ := 48 / 5
def number_of_piles : ℚ := 4

-- Define the assertion to prove
theorem shaina_chocolate : (total_chocolate / number_of_piles) = (12 / 5) := 
by 
  sorry

end shaina_chocolate_l70_70260


namespace find_x_average_l70_70235

theorem find_x_average :
  ∃ x : ℝ, (x + 8 + (7 * x - 3) + (3 * x + 10) + (-x + 6)) / 4 = 5 * x - 4 ∧ x = 3.7 :=
  by
  use 3.7
  sorry

end find_x_average_l70_70235


namespace central_cell_value_l70_70675

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l70_70675


namespace number_of_b_objects_l70_70324

theorem number_of_b_objects
  (total_objects : ℕ) 
  (a_objects : ℕ) 
  (b_objects : ℕ) 
  (h1 : total_objects = 35) 
  (h2 : a_objects = 17) 
  (h3 : total_objects = a_objects + b_objects) :
  b_objects = 18 :=
by
  sorry

end number_of_b_objects_l70_70324


namespace parrots_left_l70_70173

theorem parrots_left 
  (c : Nat)   -- The initial number of crows
  (x : Nat)   -- The number of parrots and crows that flew away
  (h1 : 7 + c = 13)          -- Initial total number of birds
  (h2 : c - x = 1)           -- Number of crows left
  : 7 - x = 2 :=             -- Number of parrots left
by
  sorry

end parrots_left_l70_70173


namespace evaluate_expression_l70_70059

theorem evaluate_expression : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := 
by
  sorry

end evaluate_expression_l70_70059


namespace solve_for_x_l70_70410

theorem solve_for_x (x : ℝ) (h : 1 / 3 + 1 / x = 2 / 3) : x = 3 :=
sorry

end solve_for_x_l70_70410


namespace max_red_tiles_l70_70392

theorem max_red_tiles (n : ℕ) (color : ℕ → ℕ → color) :
    (∀ i j, color i j ≠ color (i + 1) j ∧ color i j ≠ color i (j + 1) ∧ color i j ≠ color (i + 1) (j + 1) 
           ∧ color i j ≠ color (i - 1) j ∧ color i j ≠ color i (j - 1) ∧ color i j ≠ color (i - 1) (j - 1)) 
    → ∃ m ≤ 2500, ∀ i j, (color i j = red ↔ i * n + j < m) :=
sorry

end max_red_tiles_l70_70392


namespace quadratic_function_points_relationship_l70_70362

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end quadratic_function_points_relationship_l70_70362


namespace professors_women_tenured_or_both_l70_70622

variable (professors : ℝ) -- Total number of professors as percentage
variable (women tenured men_tenured tenured_women : ℝ) -- Given percentages

-- Conditions
variables (hw : women = 0.69 * professors) 
          (ht : tenured = 0.7 * professors)
          (hm_t : men_tenured = 0.52 * (1 - women) * professors)
          (htw : tenured_women = tenured - men_tenured)
          
-- The statement to prove
theorem professors_women_tenured_or_both :
  women + tenured - tenured_women = 0.8512 * professors :=
by
  sorry

end professors_women_tenured_or_both_l70_70622


namespace carol_allowance_problem_l70_70557

open Real

theorem carol_allowance_problem (w : ℝ) 
  (fixed_allowance : ℝ := 20) 
  (extra_earnings_per_week : ℝ := 22.5) 
  (total_money : ℝ := 425) :
  fixed_allowance * w + extra_earnings_per_week * w = total_money → w = 10 :=
by
  intro h
  -- Proof skipped
  sorry

end carol_allowance_problem_l70_70557


namespace unique_lottery_ticket_number_l70_70570

noncomputable def five_digit_sum_to_age (ticket : ℕ) (neighbor_age : ℕ) := 
  (ticket >= 10000 ∧ ticket <= 99999) ∧ 
  (neighbor_age = 5 * ((ticket / 10000) + (ticket % 10000 / 1000) + 
                        (ticket % 1000 / 100) + (ticket % 100 / 10) + 
                        (ticket % 10)))

theorem unique_lottery_ticket_number {ticket : ℕ} {neighbor_age : ℕ} 
    (h : five_digit_sum_to_age ticket neighbor_age) 
    (unique_solution : ∀ ticket1 ticket2, 
                        five_digit_sum_to_age ticket1 neighbor_age → 
                        five_digit_sum_to_age ticket2 neighbor_age → 
                        ticket1 = ticket2) : 
  ticket = 99999 :=
  sorry

end unique_lottery_ticket_number_l70_70570


namespace point_on_line_l70_70656

theorem point_on_line (m : ℝ) : (2 = m - 1) → (m = 3) :=
by sorry

end point_on_line_l70_70656


namespace symmetric_point_x_axis_l70_70828

variable (P : (ℝ × ℝ)) (x : ℝ) (y : ℝ)

-- Given P is a point (x, y)
def symmetric_about_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

-- Special case for the point (-2, 3)
theorem symmetric_point_x_axis : 
  symmetric_about_x_axis (-2, 3) = (-2, -3) :=
by 
  sorry

end symmetric_point_x_axis_l70_70828


namespace math_problem_proof_l70_70493

noncomputable def question_to_equivalent_proof_problem : Prop :=
  ∃ (p q r : ℤ), 
    (p + q + r = 0) ∧ 
    (p * q + q * r + r * p = -2023) ∧ 
    (|p| + |q| + |r| = 84)

theorem math_problem_proof : question_to_equivalent_proof_problem := 
  by 
    -- proof goes here
    sorry

end math_problem_proof_l70_70493


namespace max_sum_when_product_is_399_l70_70792

theorem max_sum_when_product_is_399 :
  ∃ (X Y Z : ℕ), X * Y * Z = 399 ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X ∧ X + Y + Z = 29 :=
by
  sorry

end max_sum_when_product_is_399_l70_70792


namespace mr_jones_loss_l70_70528

theorem mr_jones_loss :
  ∃ (C_1 C_2 : ℝ), 
    (1.2 = 1.2 * C_1 / 1.2) ∧ 
    (1.2 = 0.8 * C_2) ∧ 
    ((C_1 + C_2) - (2 * 1.2)) = -0.1 :=
by
  sorry

end mr_jones_loss_l70_70528


namespace sequence_sum_of_geometric_progressions_l70_70539

theorem sequence_sum_of_geometric_progressions
  (u1 v1 q p : ℝ)
  (h1 : u1 + v1 = 0)
  (h2 : u1 * q + v1 * p = 0) :
  u1 * q^2 + v1 * p^2 = 0 :=
by sorry

end sequence_sum_of_geometric_progressions_l70_70539


namespace average_death_rate_l70_70835

def birth_rate := 4 -- people every 2 seconds
def net_increase_per_day := 43200 -- people

def seconds_per_day := 86400 -- 24 * 60 * 60

def net_increase_per_second := net_increase_per_day / seconds_per_day -- people per second

def death_rate := (birth_rate / 2) - net_increase_per_second -- people per second

theorem average_death_rate :
  death_rate * 2 = 3 := by
  -- proof is omitted
  sorry

end average_death_rate_l70_70835


namespace work_done_together_in_six_days_l70_70274

theorem work_done_together_in_six_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
  1 / (A + B) = 6 :=
by
  sorry

end work_done_together_in_six_days_l70_70274


namespace andy_max_cookies_l70_70013

-- Definitions for the problem conditions
def total_cookies := 36
def bella_eats (andy_cookies : ℕ) := 2 * andy_cookies
def charlie_eats (andy_cookies : ℕ) := andy_cookies
def consumed_cookies (andy_cookies : ℕ) := andy_cookies + bella_eats andy_cookies + charlie_eats andy_cookies

-- The statement to prove
theorem andy_max_cookies : ∃ (a : ℕ), consumed_cookies a = total_cookies ∧ a = 9 :=
by
  sorry

end andy_max_cookies_l70_70013


namespace total_cost_3m3_topsoil_l70_70113

def topsoil_cost (V C : ℕ) : ℕ :=
  V * C

theorem total_cost_3m3_topsoil : topsoil_cost 3 12 = 36 :=
by
  unfold topsoil_cost
  exact rfl

end total_cost_3m3_topsoil_l70_70113


namespace range_f_g_f_eq_g_implies_A_l70_70028

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (x : ℝ) : ℝ := 4 * x + 1

theorem range_f_g :
  (range f ∩ Icc 1 17 = Icc 1 17) ∧ (range g ∩ Icc 1 17 = Icc 1 17) :=
sorry

theorem f_eq_g_implies_A :
  ∀ A ⊆ Icc 0 4, (∀ x ∈ A, f x = g x) → A = {0} ∨ A = {4} ∨ A = {0, 4} :=
sorry

end range_f_g_f_eq_g_implies_A_l70_70028


namespace garden_table_ratio_l70_70582

theorem garden_table_ratio (x y : ℝ) (h₁ : x + y = 750) (h₂ : y = 250) : x / y = 2 :=
by
  -- Proof omitted
  sorry

end garden_table_ratio_l70_70582


namespace fraction_equality_l70_70371

theorem fraction_equality (x : ℝ) :
  (4 + 2 * x) / (7 + 3 * x) = (2 + 3 * x) / (4 + 5 * x) ↔ x = -1 ∨ x = -2 := by
  sorry

end fraction_equality_l70_70371


namespace feet_per_inch_of_model_l70_70587

def height_of_statue := 75 -- in feet
def height_of_model := 5 -- in inches

theorem feet_per_inch_of_model : (height_of_statue / height_of_model) = 15 :=
by
  sorry

end feet_per_inch_of_model_l70_70587


namespace x_in_M_sufficient_condition_for_x_in_N_l70_70477

def M := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0}
def N := {y : ℝ | ∃ x : ℝ, y = Real.sqrt ((1 - x) / x)}

theorem x_in_M_sufficient_condition_for_x_in_N :
  (∀ x, x ∈ M → x ∈ N) ∧ ¬ (∀ x, x ∈ N → x ∈ M) :=
by sorry

end x_in_M_sufficient_condition_for_x_in_N_l70_70477


namespace statues_created_first_year_l70_70651

-- Definition of the initial conditions and the variable representing the number of statues created in the first year.
variables (S : ℕ)

-- Condition 1: In the second year, statues are quadrupled.
def second_year_statues : ℕ := 4 * S

-- Condition 2: In the third year, 12 statues are added, and 3 statues are broken.
def third_year_statues : ℕ := second_year_statues S + 12 - 3

-- Condition 3: In the fourth year, twice as many new statues are added as had been broken the previous year (2 * 3).
def fourth_year_added_statues : ℕ := 2 * 3
def fourth_year_statues : ℕ := third_year_statues S + fourth_year_added_statues

-- Condition 4: Total number of statues at the end of four years is 31.
def total_statues : ℕ := fourth_year_statues S

theorem statues_created_first_year : total_statues S = 31 → S = 4 :=
by {
  sorry
}

end statues_created_first_year_l70_70651


namespace find_range_a_l70_70093

noncomputable def f (a x : ℝ) : ℝ := abs (2 * x * a + abs (x - 1))

theorem find_range_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 5) ↔ a ≥ 6 :=
by
  sorry

end find_range_a_l70_70093


namespace find_natural_number_pairs_l70_70579

theorem find_natural_number_pairs (a b q : ℕ) : 
  (a ∣ b^2 ∧ b ∣ a^2 ∧ (a + 1) ∣ (b^2 + 1)) ↔ 
  ((a = q^2 ∧ b = q) ∨ 
   (a = q^2 ∧ b = q^3) ∨ 
   (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by
  sorry

end find_natural_number_pairs_l70_70579


namespace arithmetic_sequence_general_formula_inequality_satisfaction_l70_70891

namespace Problem

-- Definitions for the sequences and the sum of terms
def a (n : ℕ) : ℕ := sorry -- define based on conditions
def S (n : ℕ) : ℕ := sorry -- sum of first n terms of {a_n}
def b (n : ℕ) : ℕ := 2 * (S (n + 1) - S n) * S n - n * (S (n + 1) + S n)

-- Part 1: Prove the general formula for the arithmetic sequence
theorem arithmetic_sequence_general_formula :
  (∀ n : ℕ, b n = 0) → (∀ n : ℕ, a n = 0 ∨ a n = n) :=
sorry

-- Part 2: Conditions for geometric sequences and inequality
def a_2n_minus_1 (n : ℕ) : ℕ := 2 ^ n
def a_2n (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)
def b_2n (n : ℕ) : ℕ := sorry -- compute based on conditions
def b_2n_minus_1 (n : ℕ) : ℕ := sorry -- compute based on conditions

def b_condition (n : ℕ) : Prop := b_2n n < b_2n_minus_1 n

-- Prove the set of all positive integers n that satisfy the inequality
theorem inequality_satisfaction :
  { n : ℕ | b_condition n } = {1, 2, 3, 4, 5, 6} :=
sorry

end Problem

end arithmetic_sequence_general_formula_inequality_satisfaction_l70_70891


namespace friends_with_john_l70_70221

def total_slices (pizzas slices_per_pizza : Nat) : Nat := pizzas * slices_per_pizza

def total_people (total_slices slices_per_person : Nat) : Nat := total_slices / slices_per_person

def number_of_friends (total_people john : Nat) : Nat := total_people - john

theorem friends_with_john (pizzas slices_per_pizza slices_per_person john friends : Nat) (h_pizzas : pizzas = 3) 
                          (h_slices_per_pizza : slices_per_pizza = 8) (h_slices_per_person : slices_per_person = 4)
                          (h_john : john = 1) (h_friends : friends = 5) :
  number_of_friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) john = friends := by
  sorry

end friends_with_john_l70_70221


namespace slope_CD_l70_70014

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

theorem slope_CD :
  ∀ C D : ℝ × ℝ, circle1 C.1 C.2 → circle2 D.1 D.2 → 
  (C ≠ D → (D.2 - C.2) / (D.1 - C.1) = 5 / 2) := 
by
  -- proof to be completed
  sorry

end slope_CD_l70_70014


namespace probability_G_is_one_fourth_l70_70732

-- Definitions and conditions
variables (p_E p_F p_G p_H : ℚ)
axiom probability_E : p_E = 1/3
axiom probability_F : p_F = 1/6
axiom prob_G_eq_H : p_G = p_H
axiom total_prob_sum : p_E + p_F + p_G + p_G = 1

-- Theorem statement
theorem probability_G_is_one_fourth : p_G = 1/4 :=
by 
  -- Lean proof omitted, only the statement required
  sorry

end probability_G_is_one_fourth_l70_70732


namespace solve_for_s_l70_70257

theorem solve_for_s : ∃ (s t : ℚ), (8 * s + 7 * t = 160) ∧ (s = t - 3) ∧ (s = 139 / 15) := by
  sorry

end solve_for_s_l70_70257


namespace polyhedron_edges_vertices_l70_70599

theorem polyhedron_edges_vertices (F : ℕ) (triangular_faces : Prop) (hF : F = 20) : ∃ S A : ℕ, S = 12 ∧ A = 30 :=
by
  -- stating the problem conditions and desired conclusion
  sorry

end polyhedron_edges_vertices_l70_70599


namespace tan_theta_of_obtuse_angle_l70_70924

noncomputable def theta_expression (θ : Real) : Complex :=
  Complex.mk (3 * Real.sin θ) (Real.cos θ)

theorem tan_theta_of_obtuse_angle {θ : Real} (h_modulus : Complex.abs (theta_expression θ) = Real.sqrt 5) 
  (h_obtuse : π / 2 < θ ∧ θ < π) : Real.tan θ = -1 := 
  sorry

end tan_theta_of_obtuse_angle_l70_70924


namespace river_flow_speed_eq_l70_70993

-- Definitions of the given conditions
def ship_speed : ℝ := 30
def distance_downstream : ℝ := 144
def distance_upstream : ℝ := 96

-- Lean 4 statement to prove the condition
theorem river_flow_speed_eq (v : ℝ) :
  (distance_downstream / (ship_speed + v) = distance_upstream / (ship_speed - v)) :=
by { sorry }

end river_flow_speed_eq_l70_70993


namespace doubled_radius_and_arc_length_invariant_l70_70929

theorem doubled_radius_and_arc_length_invariant (r l : ℝ) : (l / r) = (2 * l / (2 * r)) :=
by
  sorry

end doubled_radius_and_arc_length_invariant_l70_70929


namespace clock_angle_at_8_20_is_130_degrees_l70_70710

/--
A clock has 12 hours, and each hour represents 30 degrees.
The minute hand moves 6 degrees per minute.
The hour hand moves 0.5 degrees per minute from its current hour position.
Prove that the smaller angle between the hour and minute hands at 8:20 p.m. is 130 degrees.
-/
theorem clock_angle_at_8_20_is_130_degrees
    (hours_per_clock : ℝ := 12)
    (degrees_per_hour : ℝ := 360 / hours_per_clock)
    (minutes_per_hour : ℝ := 60)
    (degrees_per_minute : ℝ := 360 / minutes_per_hour)
    (hour_slider_per_minute : ℝ := degrees_per_hour / minutes_per_hour)
    (minute_hand_at_20 : ℝ := 20 * degrees_per_minute)
    (hour_hand_at_8: ℝ := 8 * degrees_per_hour)
    (hour_hand_move_in_20_minutes : ℝ := 20 * hour_slider_per_minute)
    (hour_hand_at_8_20 : ℝ := hour_hand_at_8 + hour_hand_move_in_20_minutes) :
  |hour_hand_at_8_20 - minute_hand_at_20| = 130 :=
by
  sorry

end clock_angle_at_8_20_is_130_degrees_l70_70710


namespace hyperbola_equation_sum_of_slopes_l70_70661

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 3

theorem hyperbola_equation :
  ∀ (a b : ℝ) (H1 : a > 0) (H2 : b > 0) (H3 : (2^2) = a^2 + b^2)
    (H4 : ∀ (x₀ y₀ : ℝ), (x₀ ≠ -a) ∧ (x₀ ≠ a) → (y₀^2 = (b^2 / a^2) * (x₀^2 - a^2)) ∧ ((y₀ / (x₀ + a) * y₀ / (x₀ - a)) = 3)),
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 - y^2 / 3 = 1)) :=
by
  intros a b H1 H2 H3 H4 x y Hxy
  sorry

theorem sum_of_slopes (m n : ℝ) (H1 : m < 1) :
  ∀ (k1 k2 : ℝ) (H2 : A ≠ B) (H3 : ((k1 ≠ k2) ∧ (1 + k1^2) / (3 - k1^2) = (1 + k2^2) / (3 - k2^2))),
  k1 + k2 = 0 :=
by
  intros k1 k2 H2 H3
  exact sorry

end hyperbola_equation_sum_of_slopes_l70_70661


namespace find_value_of_expression_l70_70572

-- Given conditions
variable (a : ℝ)
variable (h_root : a^2 + 2 * a - 2 = 0)

-- Mathematically equivalent proof problem
theorem find_value_of_expression : 3 * a^2 + 6 * a + 2023 = 2029 :=
by
  sorry

end find_value_of_expression_l70_70572


namespace number_of_bird_cages_l70_70767

-- Definitions for the problem conditions
def birds_per_cage : ℕ := 2 + 7
def total_birds : ℕ := 72

-- The theorem to prove the number of bird cages is 8
theorem number_of_bird_cages : total_birds / birds_per_cage = 8 := by
  sorry

end number_of_bird_cages_l70_70767


namespace smallest_addition_to_make_multiple_of_5_l70_70629

theorem smallest_addition_to_make_multiple_of_5 : ∃ k : ℕ, k > 0 ∧ (729 + k) % 5 = 0 ∧ k = 1 := sorry

end smallest_addition_to_make_multiple_of_5_l70_70629


namespace albert_brother_younger_l70_70597

variables (A B Y F M : ℕ)
variables (h1 : F = 48)
variables (h2 : M = 46)
variables (h3 : F - M = 4)
variables (h4 : Y = A - B)

theorem albert_brother_younger (h_cond : (F - M = 4) ∧ (F = 48) ∧ (M = 46) ∧ (Y = A - B)) : Y = 2 :=
by
  rcases h_cond with ⟨h_diff, h_father, h_mother, h_ages⟩
  -- Assuming that each step provided has correct assertive logic.
  sorry

end albert_brother_younger_l70_70597


namespace find_angle_A_l70_70885

theorem find_angle_A (a b : ℝ) (B A : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = Real.pi / 4) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l70_70885


namespace range_of_a_l70_70064

theorem range_of_a (a : ℝ) (h : a > 0) (h1 : ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) : a ≥ 2 := 
sorry

end range_of_a_l70_70064


namespace sequence_formula_l70_70292

noncomputable def a : ℕ → ℕ
| 0       => 2
| (n + 1) => a n ^ 2 - n * a n + 1

theorem sequence_formula (n : ℕ) : a n = n + 2 :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end sequence_formula_l70_70292


namespace greatest_possible_value_l70_70631

theorem greatest_possible_value (x y : ℝ) (h1 : -4 ≤ x) (h2 : x ≤ -2) (h3 : 2 ≤ y) (h4 : y ≤ 4) : 
  ∃ z: ℝ, z = (x + y) / x ∧ (∀ z', z' = (x' + y') / x' ∧ -4 ≤ x' ∧ x' ≤ -2 ∧ 2 ≤ y' ∧ y' ≤ 4 → z' ≤ z) ∧ z = 0 :=
by
  sorry

end greatest_possible_value_l70_70631


namespace cricket_count_l70_70327

theorem cricket_count (x : ℕ) (h : x + 11 = 18) : x = 7 :=
by sorry

end cricket_count_l70_70327


namespace ship_length_is_correct_l70_70039

-- Define the variables
variables (L E S C : ℝ)

-- Define the given conditions
def condition1 (L E S C : ℝ) : Prop := 320 * E = L + 320 * (S - C)
def condition2 (L E S C : ℝ) : Prop := 80 * E = L - 80 * (S + C)

-- Mathematical statement to be proven
theorem ship_length_is_correct
  (L E S C : ℝ)
  (h1 : condition1 L E S C)
  (h2 : condition2 L E S C) :
  L = 26 * E + (2 / 3) * E :=
sorry

end ship_length_is_correct_l70_70039


namespace common_ratio_eq_l70_70630

variables {x y z r : ℝ}

theorem common_ratio_eq (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hgp : x * (y - z) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (x - y) ≠ 0 ∧ 
          (y * (z - x)) / (x * (y - z)) = r ∧ (z * (x - y)) / (y * (z - x)) = r) :
  r^2 + r + 1 = 0 :=
sorry

end common_ratio_eq_l70_70630


namespace find_rate_l70_70261

def plan1_cost (minutes : ℕ) : ℝ :=
  if minutes <= 500 then 50 else 50 + (minutes - 500) * 0.35

def plan2_cost (minutes : ℕ) (x : ℝ) : ℝ :=
  if minutes <= 1000 then 75 else 75 + (minutes - 1000) * x

theorem find_rate (x : ℝ) :
  plan1_cost 2500 = plan2_cost 2500 x → x = 0.45 := by
  sorry

end find_rate_l70_70261


namespace volume_of_inscribed_cubes_l70_70836

noncomputable def tetrahedron_cube_volume (a m : ℝ) : ℝ × ℝ :=
  let V1 := (a * m / (a + m))^3
  let V2 := (a * m / (a + (Real.sqrt 2) * m))^3
  (V1, V2)

theorem volume_of_inscribed_cubes (a m : ℝ) (ha : 0 < a) (hm : 0 < m) :
  tetrahedron_cube_volume a m = 
  ( (a * m / (a + m))^3, 
    (a * m / (a + (Real.sqrt 2) * m))^3 ) :=
  by
    sorry

end volume_of_inscribed_cubes_l70_70836


namespace sin_product_eq_one_sixteenth_l70_70886

theorem sin_product_eq_one_sixteenth : 
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * 
  (Real.sin (78 * Real.pi / 180)) = 
  1 / 16 := 
sorry

end sin_product_eq_one_sixteenth_l70_70886


namespace toll_for_18_wheel_truck_l70_70544

noncomputable def toll (x : ℕ) : ℝ :=
  2.50 + 0.50 * (x - 2)

theorem toll_for_18_wheel_truck :
  let num_wheels := 18
  let wheels_on_front_axle := 2
  let wheels_per_other_axle := 4
  let num_other_axles := (num_wheels - wheels_on_front_axle) / wheels_per_other_axle
  let total_num_axles := num_other_axles + 1
  toll total_num_axles = 4.00 :=
by
  sorry

end toll_for_18_wheel_truck_l70_70544


namespace tan_shift_symmetric_l70_70584

theorem tan_shift_symmetric :
  let f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)
  let g (x : ℝ) := f (x + Real.pi / 6)
  g (Real.pi / 4) = 0 ∧ ∀ x, g (Real.pi / 2 - x) = -g (Real.pi / 2 + x) :=
by
  sorry

end tan_shift_symmetric_l70_70584


namespace geometric_seq_a4_l70_70947

variable {a : ℕ → ℝ}

theorem geometric_seq_a4 (h : ∀ n, a (n + 2) / a n = a 2 / a 0)
  (root_condition1 : a 2 * a 6 = 64)
  (root_condition2 : a 2 + a 6 = 34) :
  a 4 = 8 :=
by
  sorry

end geometric_seq_a4_l70_70947


namespace meeting_time_and_location_l70_70757

/-- Define the initial conditions -/
def start_time : ℕ := 8 -- 8:00 AM
def city_distance : ℕ := 12 -- 12 kilometers
def pedestrian_speed : ℚ := 6 -- 6 km/h
def cyclist_speed : ℚ := 18 -- 18 km/h

/-- Define the conditions for meeting time and location -/
theorem meeting_time_and_location :
  ∃ (meet_time : ℕ) (meet_distance : ℚ),
    meet_time = 9 * 60 + 15 ∧   -- 9:15 AM in minutes
    meet_distance = 4.5 :=      -- 4.5 kilometers
sorry

end meeting_time_and_location_l70_70757


namespace anna_age_when_married_l70_70944

-- Define constants for the conditions
def j_married : ℕ := 22
def m : ℕ := 30
def combined_age_today : ℕ := 5 * j_married
def j_current : ℕ := j_married + m

-- Define Anna's current age based on the combined age today and Josh's current age
def a_current : ℕ := combined_age_today - j_current

-- Define Anna's age when married
def a_married : ℕ := a_current - m

-- Statement of the theorem to be proved
theorem anna_age_when_married : a_married = 28 :=
by
  sorry

end anna_age_when_married_l70_70944


namespace circle_equation_l70_70082

theorem circle_equation (a : ℝ) (h : a = 1) :
  (∀ (C : ℝ × ℝ), C = (a, a) →
  (∀ (r : ℝ), r = dist C (1, 0) →
  r = 1 → ((x - a) ^ 2 + (y - a) ^ 2 = r ^ 2))) :=
by
  sorry

end circle_equation_l70_70082


namespace problem1_inequality_problem2_inequality_l70_70655

theorem problem1_inequality (x : ℝ) (h1 : 2 * x + 10 ≤ 5 * x + 1) (h2 : 3 * (x - 1) > 9) : x > 4 := sorry

theorem problem2_inequality (x : ℝ) (h1 : 3 * (x + 2) ≥ 2 * x + 5) (h2 : 2 * x - (3 * x + 1) / 2 < 1) : -1 ≤ x ∧ x < 3 := sorry

end problem1_inequality_problem2_inequality_l70_70655


namespace solve_card_trade_problem_l70_70667

def card_trade_problem : Prop :=
  ∃ V : ℕ, 
  (75 - V + 10 + 88 - 8 + V = 75 + 88 - 8 + 10 ∧ V + 15 = 35)

theorem solve_card_trade_problem : card_trade_problem :=
  sorry

end solve_card_trade_problem_l70_70667


namespace Lily_points_l70_70312

variable (x y z : ℕ) -- points for inner ring (x), middle ring (y), and outer ring (z)

-- Tom's score
axiom Tom_score : 3 * x + y + 2 * z = 46

-- John's score
axiom John_score : x + 3 * y + 2 * z = 34

-- Lily's score
def Lily_score : ℕ := 40

theorem Lily_points : ∀ (x y z : ℕ), 3 * x + y + 2 * z = 46 → x + 3 * y + 2 * z = 34 → Lily_score = 40 := by
  intros x y z Tom_score John_score
  sorry

end Lily_points_l70_70312


namespace metal_rods_per_sheet_l70_70428

theorem metal_rods_per_sheet :
  (∀ (metal_rod_for_sheets metal_rod_for_beams total_metal_rod num_sheet_per_panel num_panel num_rod_per_beam),
    (num_rod_per_beam = 4) →
    (total_metal_rod = 380) →
    (metal_rod_for_beams = num_panel * (2 * num_rod_per_beam)) →
    (metal_rod_for_sheets = total_metal_rod - metal_rod_for_beams) →
    (num_sheet_per_panel = 3) →
    (num_panel = 10) →
    (metal_rod_per_sheet = metal_rod_for_sheets / (num_panel * num_sheet_per_panel)) →
    metal_rod_per_sheet = 10
  ) := sorry

end metal_rods_per_sheet_l70_70428


namespace average_of_first_15_even_numbers_is_16_l70_70307

-- Define the sum of the first 15 even numbers
def sum_first_15_even_numbers : ℕ :=
  2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30

-- Define the average of the first 15 even numbers
def average_of_first_15_even_numbers : ℕ :=
  sum_first_15_even_numbers / 15

-- Prove that the average is equal to 16
theorem average_of_first_15_even_numbers_is_16 : average_of_first_15_even_numbers = 16 :=
by
  -- Sorry placeholder for the proof
  sorry

end average_of_first_15_even_numbers_is_16_l70_70307


namespace new_paint_intensity_l70_70115

variable (V : ℝ)  -- V is the volume of the original 50% intensity red paint.
variable (I₁ I₂ : ℝ)  -- I₁ is the intensity of the original paint, I₂ is the intensity of the replaced paint.
variable (f : ℝ)  -- f is the fraction of the original paint being replaced.

-- Assume given conditions
axiom intensity_original : I₁ = 0.5
axiom intensity_new : I₂ = 0.25
axiom fraction_replaced : f = 0.8

-- Prove that the new intensity is 30%
theorem new_paint_intensity :
  (f * I₂ + (1 - f) * I₁) = 0.3 := 
by 
  -- This is the main theorem we want to prove
  sorry

end new_paint_intensity_l70_70115


namespace price_of_other_pieces_l70_70747

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end price_of_other_pieces_l70_70747


namespace probability_female_wears_glasses_l70_70459

def prob_female_wears_glasses (total_females : ℕ) (females_no_glasses : ℕ) : ℚ :=
  let females_with_glasses := total_females - females_no_glasses
  females_with_glasses / total_females

theorem probability_female_wears_glasses :
  prob_female_wears_glasses 18 8 = 5 / 9 := by
  sorry  -- Proof is skipped

end probability_female_wears_glasses_l70_70459


namespace running_problem_l70_70136

variables (x y : ℝ)

theorem running_problem :
  (5 * x = 5 * y + 10) ∧ (4 * x = 4 * y + 2 * y) :=
by
  sorry

end running_problem_l70_70136


namespace smallest_positive_multiple_of_23_mod_89_is_805_l70_70194

theorem smallest_positive_multiple_of_23_mod_89_is_805 : 
  ∃ a : ℕ, 23 * a ≡ 4 [MOD 89] ∧ 23 * a = 805 := 
by
  sorry

end smallest_positive_multiple_of_23_mod_89_is_805_l70_70194


namespace third_quadrant_point_m_l70_70674

theorem third_quadrant_point_m (m : ℤ) (h1 : 2 - m < 0) (h2 : m - 4 < 0) : m = 3 :=
by
  sorry

end third_quadrant_point_m_l70_70674


namespace break_even_machines_l70_70035

def cost_parts : ℤ := 3600
def cost_patent : ℤ := 4500
def selling_price : ℤ := 180

def total_costs : ℤ := cost_parts + cost_patent

def machines_to_break_even : ℤ := total_costs / selling_price

theorem break_even_machines :
  machines_to_break_even = 45 := by
  sorry

end break_even_machines_l70_70035


namespace functional_equation_solution_l70_70821

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (1 + a * x)

theorem functional_equation_solution (a : ℝ) (x y : ℝ)
  (ha : 0 < a) (hx : 0 < x) (hy : 0 < y) :
  f a x * f a (y * f a x) = f a (x + y) :=
sorry

end functional_equation_solution_l70_70821


namespace absolute_difference_of_integers_l70_70050

theorem absolute_difference_of_integers (x y : ℤ) (h1 : (x + y) / 2 = 15) (h2 : Int.sqrt (x * y) + 6 = 15) : |x - y| = 24 :=
  sorry

end absolute_difference_of_integers_l70_70050


namespace binom_60_3_eq_34220_l70_70907

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l70_70907


namespace determine_m_if_root_exists_l70_70438

def fractional_equation_has_root (x m : ℝ) : Prop :=
  (3 / (x - 4) + (x + m) / (4 - x) = 1)

theorem determine_m_if_root_exists (x : ℝ) (h : fractional_equation_has_root x m) : m = -1 :=
sorry

end determine_m_if_root_exists_l70_70438


namespace coordinates_of_point_l70_70161

noncomputable def point_on_x_axis (x : ℝ) :=
  (x, 0)

theorem coordinates_of_point (x : ℝ) (hx : abs x = 3) :
  point_on_x_axis x = (3, 0) ∨ point_on_x_axis x = (-3, 0) :=
  sorry

end coordinates_of_point_l70_70161


namespace number_of_subsets_including_1_and_10_l70_70286

def A : Set ℕ := {a : ℕ | ∃ x y z : ℕ, a = 2^x * 3^y * 5^z}
def B : Set ℕ := {b : ℕ | b ∈ A ∧ 1 ≤ b ∧ b ≤ 10}

theorem number_of_subsets_including_1_and_10 :
  ∃ (s : Finset (Finset ℕ)), (∀ x ∈ s, 1 ∈ x ∧ 10 ∈ x) ∧ s.card = 128 := by
  sorry

end number_of_subsets_including_1_and_10_l70_70286


namespace problem_statement_l70_70118

/-- Let x, y, z be nonzero real numbers such that x + y + z = 0.
    Prove that ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → x + y + z = 0 → (x^3 + y^3 + z^3) / (x * y * z) = 3. -/
theorem problem_statement (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by 
  sorry

end problem_statement_l70_70118


namespace total_distance_of_the_race_l70_70443

-- Define the given conditions
def A_beats_B_by_56_meters_or_7_seconds : Prop :=
  ∃ D : ℕ, ∀ S_B S_A : ℕ, S_B = 8 ∧ S_A = D / 8 ∧ D = S_B * (8 + 7)

-- Define the question and correct answer
theorem total_distance_of_the_race : A_beats_B_by_56_meters_or_7_seconds → ∃ D : ℕ, D = 120 :=
by
  sorry

end total_distance_of_the_race_l70_70443


namespace sarah_must_solve_at_least_16_l70_70568

theorem sarah_must_solve_at_least_16
  (total_problems : ℕ)
  (problems_attempted : ℕ)
  (problems_unanswered : ℕ)
  (points_per_correct : ℕ)
  (points_per_unanswered : ℕ)
  (target_score : ℕ)
  (h1 : total_problems = 30)
  (h2 : points_per_correct = 7)
  (h3 : points_per_unanswered = 2)
  (h4 : problems_unanswered = 5)
  (h5 : problems_attempted = 25)
  (h6 : target_score = 120) :
  ∃ (correct_solved : ℕ), correct_solved ≥ 16 ∧ correct_solved ≤ problems_attempted ∧
    (correct_solved * points_per_correct) + (problems_unanswered * points_per_unanswered) ≥ target_score :=
by {
  sorry
}

end sarah_must_solve_at_least_16_l70_70568


namespace find_a_values_l70_70442

noncomputable def system_has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((|y - 4| + |x + 12| - 3) * (x^2 + y^2 - 12) = 0) ∧ 
    ((x + 5)^2 + (y - 4)^2 = a)

theorem find_a_values : system_has_exactly_three_solutions 16 ∨ 
                        system_has_exactly_three_solutions (41 + 4 * Real.sqrt 123) :=
  by sorry

end find_a_values_l70_70442


namespace question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l70_70936

theorem question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1
    (a b c d : ℤ)
    (h1 : a + b = 11)
    (h2 : b + c = 9)
    (h3 : c + d = 3)
    : a + d = -1 :=
by
  sorry

end question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l70_70936


namespace max_divisions_circle_and_lines_l70_70239

theorem max_divisions_circle_and_lines (n : ℕ) (h₁ : n = 5) : 
  let R_lines := n * (n + 1) / 2 + 1 -- Maximum regions formed by n lines
  let R_circle_lines := 2 * n       -- Additional regions formed by a circle intersecting n lines
  R_lines + R_circle_lines = 26 := by
  sorry

end max_divisions_circle_and_lines_l70_70239


namespace ryan_spends_7_hours_on_english_l70_70042

variable (C : ℕ)
variable (E : ℕ)

def hours_spent_on_english (C : ℕ) : ℕ := C + 2

theorem ryan_spends_7_hours_on_english :
  C = 5 → E = hours_spent_on_english C → E = 7 :=
by
  intro hC hE
  rw [hC] at hE
  exact hE

end ryan_spends_7_hours_on_english_l70_70042


namespace trig_identity_l70_70953

-- Proving the equality (we state the problem here)
theorem trig_identity :
  Real.sin (40 * Real.pi / 180) * (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) = -8 / 3 :=
by
  sorry

end trig_identity_l70_70953


namespace factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l70_70222

-- Factorization of 4a^2 - 9 as (2a + 3)(2a - 3)
theorem factorize_4a2_minus_9 (a : ℝ) : 4 * a^2 - 9 = (2 * a + 3) * (2 * a - 3) :=
by 
  sorry

-- Factorization of 2x^2 y - 8xy + 8y as 2y(x-2)^2
theorem factorize_2x2y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2) ^ 2 :=
by 
  sorry

end factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l70_70222


namespace nonnegative_fraction_iff_interval_l70_70339

theorem nonnegative_fraction_iff_interval (x : ℝ) : 
  0 ≤ x ∧ x < 3 ↔ 0 ≤ (x^2 - 12 * x^3 + 36 * x^4) / (9 - x^3) := by
  sorry

end nonnegative_fraction_iff_interval_l70_70339


namespace some_number_proof_l70_70935

def g (n : ℕ) : ℕ :=
  if n < 3 then 1 else 
  if n % 2 = 0 then g (n - 1) else 
    g (n - 2) * n

theorem some_number_proof : g 106 - g 103 = 105 :=
by sorry

end some_number_proof_l70_70935


namespace triangle_formation_and_acuteness_l70_70649

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end triangle_formation_and_acuteness_l70_70649


namespace cookies_on_first_plate_l70_70415

theorem cookies_on_first_plate :
  ∃ a1 a2 a3 a4 a5 a6 : ℤ, 
  a2 = 7 ∧ 
  a3 = 10 ∧
  a4 = 14 ∧
  a5 = 19 ∧
  a6 = 25 ∧
  a2 = a1 + 2 ∧ 
  a3 = a2 + 3 ∧ 
  a4 = a3 + 4 ∧ 
  a5 = a4 + 5 ∧ 
  a6 = a5 + 6 ∧ 
  a1 = 5 :=
sorry

end cookies_on_first_plate_l70_70415


namespace cost_per_quart_l70_70906

theorem cost_per_quart (paint_cost : ℝ) (coverage : ℝ) (cost_to_paint_cube : ℝ) (cube_edge : ℝ) 
    (h_coverage : coverage = 1200) (h_cost_to_paint_cube : cost_to_paint_cube = 1.60) 
    (h_cube_edge : cube_edge = 10) : paint_cost = 3.20 := by 
  sorry

end cost_per_quart_l70_70906


namespace product_area_perimeter_square_EFGH_l70_70404

theorem product_area_perimeter_square_EFGH:
  let E := (5, 5)
  let F := (5, 1)
  let G := (1, 1)
  let H := (1, 5)
  let side_length := 4
  let area := side_length * side_length
  let perimeter := 4 * side_length
  area * perimeter = 256 :=
by
  sorry

end product_area_perimeter_square_EFGH_l70_70404


namespace arithmetic_sum_property_l70_70563

variable {a : ℕ → ℤ} -- declare the sequence as a sequence of integers

-- Define the condition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

-- Given condition: sum of specific terms in the sequence equals 400
def sum_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 400

-- The goal: if the sum_condition holds, then a_2 + a_8 = 160
theorem arithmetic_sum_property
  (h_sum : sum_condition a)
  (h_arith : arithmetic_sequence a) :
  a 2 + a 8 = 160 := by
  sorry

end arithmetic_sum_property_l70_70563


namespace original_number_l70_70141

theorem original_number (x : ℕ) (h : x / 3 = 42) : x = 126 :=
sorry

end original_number_l70_70141


namespace Lisa_pay_per_hour_is_15_l70_70905

-- Given conditions:
def Greta_hours : ℕ := 40
def Greta_pay_per_hour : ℕ := 12
def Lisa_hours : ℕ := 32

-- Define Greta's earnings based on the given conditions:
def Greta_earnings : ℕ := Greta_hours * Greta_pay_per_hour

-- The main statement to prove:
theorem Lisa_pay_per_hour_is_15 (h1 : Greta_earnings = Greta_hours * Greta_pay_per_hour) 
                                (h2 : Greta_earnings = Lisa_hours * L) :
  L = 15 :=
by sorry

end Lisa_pay_per_hour_is_15_l70_70905


namespace probability_black_white_ball_l70_70520

theorem probability_black_white_ball :
  let total_balls := 5
  let black_balls := 3
  let white_balls := 2
  let favorable_outcomes := (Nat.choose 3 1) * (Nat.choose 2 1)
  let total_outcomes := Nat.choose 5 2
  (favorable_outcomes / total_outcomes) = (3 / 5) := 
by
  sorry

end probability_black_white_ball_l70_70520


namespace decimal_equivalent_of_one_tenth_squared_l70_70054

theorem decimal_equivalent_of_one_tenth_squared : 
  (1 / 10 : ℝ)^2 = 0.01 := by
  sorry

end decimal_equivalent_of_one_tenth_squared_l70_70054


namespace ratio_d_a_l70_70829

theorem ratio_d_a (a b c d : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 5) : 
  d / a = 1 / 30 := 
by 
  sorry

end ratio_d_a_l70_70829


namespace find_ordered_pair_l70_70942

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 7 * x - 30 * y = 3) 
  (h2 : 3 * y - x = 5) : 
  x = -53 / 3 ∧ y = -38 / 9 :=
sorry

end find_ordered_pair_l70_70942


namespace digit_is_9_if_divisible_by_11_l70_70447

theorem digit_is_9_if_divisible_by_11 (d : ℕ) : 
  (678000 + 9000 + 800 + 90 + d) % 11 = 0 -> d = 9 := by
  sorry

end digit_is_9_if_divisible_by_11_l70_70447


namespace quadrilateral_area_correct_l70_70186

open Real
open Function
open Classical

noncomputable def quadrilateral_area : ℝ :=
  let A := (0, 0)
  let B := (2, 3)
  let C := (5, 0)
  let D := (3, -2)
  let vector_cross_product (u v : ℝ × ℝ) : ℝ := u.1 * v.2 - u.2 * v.1
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 0.5 * abs (vector_cross_product (p2 - p1) (p3 - p1))
  area_triangle A B D + area_triangle B C D

theorem quadrilateral_area_correct : quadrilateral_area = 17 / 2 :=
  sorry

end quadrilateral_area_correct_l70_70186


namespace tom_teaching_years_l70_70461

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l70_70461


namespace petStoreHasSixParrots_l70_70501

def petStoreParrotsProof : Prop :=
  let cages := 6.0
  let parakeets := 2.0
  let birds_per_cage := 1.333333333
  let total_birds := cages * birds_per_cage
  let number_of_parrots := total_birds - parakeets
  number_of_parrots = 6.0

theorem petStoreHasSixParrots : petStoreParrotsProof := by
  sorry

end petStoreHasSixParrots_l70_70501


namespace distance_between_trees_l70_70287

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (num_spaces : ℕ) (distance : ℕ)
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_spaces = num_trees - 1)
  (h4 : distance = yard_length / num_spaces) :
  distance = 18 :=
by
  sorry

end distance_between_trees_l70_70287


namespace sum_items_l70_70445

theorem sum_items (A B : ℕ) (h1 : A = 585) (h2 : A = B + 249) : A + B = 921 :=
by
  -- Proof step skipped
  sorry

end sum_items_l70_70445


namespace parabola_axis_of_symmetry_l70_70543

theorem parabola_axis_of_symmetry : 
  ∀ (x : ℝ), x = -1 → (∃ y : ℝ, y = -x^2 - 2*x - 3) :=
by
  sorry

end parabola_axis_of_symmetry_l70_70543


namespace tens_digit_of_3_pow_2023_l70_70476

theorem tens_digit_of_3_pow_2023 : (3 ^ 2023 % 100) / 10 = 2 := 
sorry

end tens_digit_of_3_pow_2023_l70_70476


namespace solve_for_k_l70_70469

theorem solve_for_k (x k : ℝ) (h : x = -3) (h_eq : k * (x + 4) - 2 * k - x = 5) : k = -2 :=
by sorry

end solve_for_k_l70_70469


namespace find_y_l70_70636

theorem find_y (y : ℤ) (h : (15 + 24 + y) / 3 = 23) : y = 30 :=
by
  sorry

end find_y_l70_70636


namespace water_needed_l70_70107

theorem water_needed (nutrient_concentrate : ℝ) (distilled_water : ℝ) (total_volume : ℝ) 
    (h1 : nutrient_concentrate = 0.08) (h2 : distilled_water = 0.04) (h3 : total_volume = 1) :
    total_volume * (distilled_water / (nutrient_concentrate + distilled_water)) = 0.333 :=
by
  sorry

end water_needed_l70_70107


namespace continuity_at_2_l70_70220

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem continuity_at_2 (b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) → b = 9 :=
by
  sorry  

end continuity_at_2_l70_70220


namespace part1_and_part2_l70_70178

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l70_70178


namespace percentage_of_blue_flowers_l70_70191

theorem percentage_of_blue_flowers 
  (total_flowers : Nat)
  (red_flowers : Nat)
  (white_flowers : Nat)
  (total_flowers_eq : total_flowers = 10)
  (red_flowers_eq : red_flowers = 4)
  (white_flowers_eq : white_flowers = 2)
  :
  ( (total_flowers - (red_flowers + white_flowers)) * 100 ) / total_flowers = 40 :=
by
  sorry

end percentage_of_blue_flowers_l70_70191


namespace simple_interest_years_l70_70618

theorem simple_interest_years (P : ℝ) (difference : ℝ) (N : ℝ) : 
  P = 2300 → difference = 69 → (23 * N = 69) → N = 3 :=
by
  intros hP hdifference heq
  sorry

end simple_interest_years_l70_70618


namespace max_value_of_a_l70_70714

theorem max_value_of_a {a : ℝ} (h : ∀ x ≥ 1, -3 * x^2 + a ≤ 0) : a ≤ 3 :=
sorry

end max_value_of_a_l70_70714


namespace michaels_brother_money_end_l70_70698

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l70_70698


namespace k_is_perfect_square_l70_70982

theorem k_is_perfect_square (m n : ℤ) (hm : m > 0) (hn : n > 0) (k := ((m + n) ^ 2) / (4 * m * (m - n) ^ 2 + 4)) :
  ∃ (a : ℤ), k = a ^ 2 := by
  sorry

end k_is_perfect_square_l70_70982


namespace goods_train_speed_l70_70877

theorem goods_train_speed (man_train_speed_kmh : Float) 
    (goods_train_length_m : Float) 
    (passing_time_s : Float) 
    (kmh_to_ms : Float := 1000 / 3600) : 
    man_train_speed_kmh = 50 → 
    goods_train_length_m = 280 → 
    passing_time_s = 9 → 
    Float.round ((goods_train_length_m / passing_time_s + man_train_speed_kmh * kmh_to_ms) * 3600 / 1000) = 61.99
:= by
  sorry

end goods_train_speed_l70_70877


namespace division_remainder_3012_97_l70_70743

theorem division_remainder_3012_97 : 3012 % 97 = 5 := 
by 
  sorry

end division_remainder_3012_97_l70_70743


namespace longest_interval_green_l70_70975

-- Definitions for the conditions
def light_cycle_duration : ℕ := 180 -- total cycle duration in seconds
def green_duration : ℕ := 90 -- green light duration in seconds
def red_delay : ℕ := 10 -- red light delay between consecutive lights in seconds
def num_lights : ℕ := 8 -- number of lights

-- Theorem statement to be proved
theorem longest_interval_green (h1 : ∀ i : ℕ, i < num_lights → 
  ∃ t : ℕ, t < light_cycle_duration ∧ (∀ k : ℕ, i + k < num_lights → t + k * red_delay < light_cycle_duration ∧ t + k * red_delay + green_duration <= light_cycle_duration)):
  ∃ interval : ℕ, interval = 20 :=
sorry

end longest_interval_green_l70_70975


namespace inverse_of_g_l70_70455

noncomputable def u (x : ℝ) : ℝ := sorry
noncomputable def v (x : ℝ) : ℝ := sorry
noncomputable def w (x : ℝ) : ℝ := sorry

noncomputable def u_inv (x : ℝ) : ℝ := sorry
noncomputable def v_inv (x : ℝ) : ℝ := sorry
noncomputable def w_inv (x : ℝ) : ℝ := sorry

lemma u_inverse : ∀ x, u_inv (u x) = x ∧ u (u_inv x) = x := sorry
lemma v_inverse : ∀ x, v_inv (v x) = x ∧ v (v_inv x) = x := sorry
lemma w_inverse : ∀ x, w_inv (w x) = x ∧ w (w_inv x) = x := sorry

noncomputable def g (x : ℝ) : ℝ := v (u (w x))

noncomputable def g_inv (x : ℝ) : ℝ := w_inv (u_inv (v_inv x))

theorem inverse_of_g :
  ∀ x : ℝ, g_inv (g x) = x ∧ g (g_inv x) = x :=
by
  intro x
  -- proof omitted
  sorry

end inverse_of_g_l70_70455


namespace triangle_right_angled_l70_70918

-- Define the variables and the condition of the problem
variables {a b c : ℝ}

-- Given condition of the problem
def triangle_condition (a b c : ℝ) : Prop :=
  2 * (a ^ 8 + b ^ 8 + c ^ 8) = (a ^ 4 + b ^ 4 + c ^ 4) ^ 2

-- The theorem to prove the triangle is right-angled
theorem triangle_right_angled (h : triangle_condition a b c) : a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 :=
sorry

end triangle_right_angled_l70_70918


namespace regular_ticket_cost_l70_70211

theorem regular_ticket_cost
    (adults : ℕ) (children : ℕ) (cash_given : ℕ) (change_received : ℕ) (adult_cost : ℕ) (child_cost : ℕ) :
    adults = 2 →
    children = 3 →
    cash_given = 40 →
    change_received = 1 →
    child_cost = adult_cost - 2 →
    2 * adult_cost + 3 * child_cost = cash_given - change_received →
    adult_cost = 9 :=
by
  intros h_adults h_children h_cash_given h_change_received h_child_cost h_sum
  sorry

end regular_ticket_cost_l70_70211


namespace factorial_equation_solution_l70_70411

theorem factorial_equation_solution (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → (a, b, c) = (3, 3, 4) :=
by
  sorry

end factorial_equation_solution_l70_70411


namespace find_k_l70_70240

-- Define point type and distances
structure Point :=
(x : ℝ)
(y : ℝ)

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition: H is the orthocenter of triangle ABC
variable (A B C H Q : Point)
variable (H_is_orthocenter : ∀ P : Point, dist P H = dist P A + dist P B - dist A B)

-- Prove the given equation
theorem find_k :
  dist Q A + dist Q B + dist Q C = 3 * dist Q H + dist H A + dist H B + dist H C :=
sorry

end find_k_l70_70240


namespace total_spent_in_may_l70_70980

-- Conditions as definitions
def cost_per_weekday : ℕ := (2 * 15) + (2 * 18)
def cost_per_weekend_day : ℕ := (3 * 12) + (2 * 20)
def weekdays_in_may : ℕ := 22
def weekend_days_in_may : ℕ := 9

-- The statement to prove
theorem total_spent_in_may :
  cost_per_weekday * weekdays_in_may + cost_per_weekend_day * weekend_days_in_may = 2136 :=
by
  sorry

end total_spent_in_may_l70_70980


namespace f_1984_and_f_1985_l70_70423

namespace Proof

variable {N M : Type} [AddMonoid M] [Zero M] (f : ℕ → M)

-- Conditions
axiom f_10 : f 10 = 0
axiom f_last_digit_3 {n : ℕ} : (n % 10 = 3) → f n = 0
axiom f_mn (m n : ℕ) : f (m * n) = f m + f n

-- Prove f(1984) = 0 and f(1985) = 0
theorem f_1984_and_f_1985 : f 1984 = 0 ∧ f 1985 = 0 :=
by
  sorry

end Proof

end f_1984_and_f_1985_l70_70423


namespace find_width_of_first_tract_l70_70708

-- Definitions based on given conditions
noncomputable def area_first_tract (W : ℝ) : ℝ := 300 * W
def area_second_tract : ℝ := 250 * 630
def combined_area : ℝ := 307500

-- The theorem we need to prove: width of the first tract is 500 meters
theorem find_width_of_first_tract (W : ℝ) (h : area_first_tract W + area_second_tract = combined_area) : W = 500 :=
by
  sorry

end find_width_of_first_tract_l70_70708


namespace last_four_digits_5_pow_2011_l70_70096

theorem last_four_digits_5_pow_2011 : 
  (5^2011 % 10000) = 8125 :=
by
  -- Definitions based on conditions in the problem
  have h5 : 5^5 % 10000 = 3125 := sorry
  have h6 : 5^6 % 10000 = 5625 := sorry
  have h7 : 5^7 % 10000 = 8125 := sorry
  
  -- Prove using periodicity and modular arithmetic
  sorry

end last_four_digits_5_pow_2011_l70_70096


namespace common_card_cost_l70_70939

def totalDeckCost (rareCost uncommonCost commonCost numRares numUncommons numCommons : ℝ) : ℝ :=
  (numRares * rareCost) + (numUncommons * uncommonCost) + (numCommons * commonCost)

theorem common_card_cost (numRares numUncommons numCommons : ℝ) (rareCost uncommonCost totalCost : ℝ) : 
  numRares = 19 → numUncommons = 11 → numCommons = 30 → 
  rareCost = 1 → uncommonCost = 0.5 → totalCost = 32 → 
  commonCost = 0.25 :=
by 
  intros 
  sorry

end common_card_cost_l70_70939


namespace sum_of_solutions_l70_70803

theorem sum_of_solutions (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ S, S = 9 ∧ ∀ x1 x2, x1 + x2 = S) := by
  intros h
  sorry

end sum_of_solutions_l70_70803


namespace find_range_of_a_l70_70858

variable {f : ℝ → ℝ}
noncomputable def domain_f : Set ℝ := {x | 7 ≤ x ∧ x < 15}
noncomputable def domain_f_2x_plus_1 : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}
noncomputable def A_or_B_eq_r (a : ℝ) : Prop := domain_f_2x_plus_1 ∪ B a = Set.univ

theorem find_range_of_a (a : ℝ) : 
  A_or_B_eq_r a → 3 ≤ a ∧ a < 6 := 
sorry

end find_range_of_a_l70_70858


namespace charles_total_earnings_l70_70751

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end charles_total_earnings_l70_70751


namespace stratified_sampling_correct_l70_70162

def num_students := 500
def num_male_students := 500
def num_female_students := 400
def ratio_male_female := num_male_students / num_female_students

def selected_male_students := 25
def selected_female_students := (selected_male_students * num_female_students) / num_male_students

theorem stratified_sampling_correct :
  selected_female_students = 20 :=
by
  sorry

end stratified_sampling_correct_l70_70162


namespace mrs_hilt_found_nickels_l70_70134

theorem mrs_hilt_found_nickels : 
  ∀ (total cents quarter cents dime cents nickel cents : ℕ), 
    total = 45 → 
    quarter = 25 → 
    dime = 10 → 
    nickel = 5 → 
    ((total - (quarter + dime)) / nickel) = 2 := 
by
  intros total quarter dime nickel h_total h_quarter h_dime h_nickel
  sorry

end mrs_hilt_found_nickels_l70_70134


namespace range_of_a_l70_70465

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l70_70465


namespace imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l70_70887

theorem imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half :
  (Complex.exp (-Complex.I * Real.pi / 6)).im = -1/2 := by
sorry

end imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l70_70887


namespace cement_amount_l70_70680

theorem cement_amount
  (originally_had : ℕ)
  (bought : ℕ)
  (total : ℕ)
  (son_brought : ℕ)
  (h1 : originally_had = 98)
  (h2 : bought = 215)
  (h3 : total = 450)
  (h4 : originally_had + bought + son_brought = total) :
  son_brought = 137 :=
by
  sorry

end cement_amount_l70_70680


namespace retail_price_machine_l70_70000

theorem retail_price_machine (P : ℝ) :
  let wholesale_price := 99
  let discount_rate := 0.10
  let profit_rate := 0.20
  let selling_price := wholesale_price + (profit_rate * wholesale_price)
  0.90 * P = selling_price → P = 132 :=

by
  intro wholesale_price discount_rate profit_rate selling_price h
  sorry -- Proof will be handled here

end retail_price_machine_l70_70000


namespace Harry_bought_five_packets_of_chili_pepper_l70_70478

noncomputable def price_pumpkin : ℚ := 2.50
noncomputable def price_tomato : ℚ := 1.50
noncomputable def price_chili_pepper : ℚ := 0.90
noncomputable def packets_pumpkin : ℕ := 3
noncomputable def packets_tomato : ℕ := 4
noncomputable def total_spent : ℚ := 18
noncomputable def packets_chili_pepper (p : ℕ) := price_pumpkin * packets_pumpkin + price_tomato * packets_tomato + price_chili_pepper * p = total_spent

theorem Harry_bought_five_packets_of_chili_pepper :
  ∃ p : ℕ, packets_chili_pepper p ∧ p = 5 :=
by 
  sorry

end Harry_bought_five_packets_of_chili_pepper_l70_70478


namespace sum_of_solutions_eq_0_l70_70678

-- Define the conditions
def y : ℝ := 6
def main_eq (x : ℝ) : Prop := x^2 + y^2 = 145

-- State the theorem
theorem sum_of_solutions_eq_0 : 
  let x1 := Real.sqrt 109
  let x2 := -Real.sqrt 109
  x1 + x2 = 0 :=
by {
  sorry
}

end sum_of_solutions_eq_0_l70_70678


namespace ratio_of_logs_l70_70696

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem ratio_of_logs (a b: ℝ) (h1 : log_base 8 a = log_base 18 b) 
    (h2 : log_base 18 b = log_base 32 (a + b)) 
    (hpos : 0 < a ∧ 0 < b) :
    b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) :=
by 
    sorry

end ratio_of_logs_l70_70696


namespace john_spent_15_dollars_on_soap_l70_70703

-- Define the number of soap bars John bought
def num_bars : ℕ := 20

-- Define the weight of each bar of soap in pounds
def weight_per_bar : ℝ := 1.5

-- Define the cost per pound of soap in dollars
def cost_per_pound : ℝ := 0.5

-- Total weight of the soap in pounds
def total_weight : ℝ := num_bars * weight_per_bar

-- Total cost of the soap in dollars
def total_cost : ℝ := total_weight * cost_per_pound

-- Statement to prove
theorem john_spent_15_dollars_on_soap : total_cost = 15 :=
by sorry

end john_spent_15_dollars_on_soap_l70_70703


namespace average_score_of_class_l70_70794

theorem average_score_of_class (n : ℕ) (k : ℕ) (jimin_score : ℕ) (jungkook_score : ℕ) (avg_others : ℕ) 
  (total_students : n = 40) (excluding_students : k = 38) 
  (avg_excluding_others : avg_others = 79) 
  (jimin : jimin_score = 98) 
  (jungkook : jungkook_score = 100) : 
  (98 + 100 + (38 * 79)) / 40 = 80 :=
sorry

end average_score_of_class_l70_70794


namespace polar_circle_l70_70777

def is_circle (ρ θ : ℝ) : Prop :=
  ρ = Real.cos (Real.pi / 4 - θ)

theorem polar_circle : 
  ∀ ρ θ : ℝ, is_circle ρ θ ↔ ∃ (x y : ℝ), (x - 1/(2 * Real.sqrt 2))^2 + (y - 1/(2 * Real.sqrt 2))^2 = (1/(2 * Real.sqrt 2))^2 :=
by
  intro ρ θ
  sorry

end polar_circle_l70_70777


namespace sum_odd_integers_correct_l70_70756

def sum_odd_integers_from_13_to_41 : ℕ := 
  let a := 13
  let l := 41
  let n := 15
  n * (a + l) / 2

theorem sum_odd_integers_correct : sum_odd_integers_from_13_to_41 = 405 :=
  by sorry

end sum_odd_integers_correct_l70_70756


namespace calculate_expression_l70_70819

theorem calculate_expression : (1100 * 1100) / ((260 * 260) - (240 * 240)) = 121 := by
  sorry

end calculate_expression_l70_70819


namespace endpoint_sum_l70_70965

theorem endpoint_sum
  (x y : ℤ)
  (H_midpoint_x : (x + 15) / 2 = 10)
  (H_midpoint_y : (y - 8) / 2 = -3) :
  x + y = 7 :=
sorry

end endpoint_sum_l70_70965


namespace sqrt8_sub_sqrt2_eq_sqrt2_l70_70262

theorem sqrt8_sub_sqrt2_eq_sqrt2 : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt8_sub_sqrt2_eq_sqrt2_l70_70262


namespace find_initial_length_of_cloth_l70_70489

noncomputable def initial_length_of_cloth : ℝ :=
  let work_rate_of_8_men := 36 / 0.75
  work_rate_of_8_men

theorem find_initial_length_of_cloth (L : ℝ) (h1 : (4:ℝ) * 2 = L / ((4:ℝ) / (L / 8)))
    (h2 : (8:ℝ) / L = 36 / 0.75) : L = 48 :=
by
  sorry

end find_initial_length_of_cloth_l70_70489


namespace sin_C_value_l70_70336

noncomputable def triangle_sine_proof (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : Real := by
  -- Utilizing the Law of Sines and given conditions to find sin C
  sorry

theorem sin_C_value (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : triangle_sine_proof A B C a b c hB hb = Real.sqrt 3 / 6 := by
  sorry

end sin_C_value_l70_70336


namespace connie_earbuds_tickets_l70_70285

theorem connie_earbuds_tickets (total_tickets : ℕ) (koala_fraction : ℕ) (bracelet_tickets : ℕ) (earbud_tickets : ℕ) :
  total_tickets = 50 →
  koala_fraction = 2 →
  bracelet_tickets = 15 →
  (total_tickets / koala_fraction) + bracelet_tickets + earbud_tickets = total_tickets →
  earbud_tickets = 10 :=
by
  intros h_total h_koala h_bracelets h_sum
  sorry

end connie_earbuds_tickets_l70_70285


namespace mean_of_set_median_is_128_l70_70233

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l70_70233


namespace xy_in_N_l70_70706

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  -- hint: use any knowledge and axioms from Mathlib to aid your proof
  sorry

end xy_in_N_l70_70706


namespace circle_condition_l70_70812

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  -- Define constants and equation representation
  let d : ℝ := -2
  let e : ℝ := -4
  let f : ℝ := m
  -- Use the condition for the circle equation
  have h : d^2 + e^2 - 4*f > 0 := sorry
  -- Prove the inequality
  sorry

end circle_condition_l70_70812


namespace units_digit_specified_expression_l70_70195

theorem units_digit_specified_expression :
  let numerator := (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11)
  let denominator := 8000
  let product := numerator * 20
  (∃ d, product / denominator = d ∧ (d % 10 = 6)) :=
by
  sorry

end units_digit_specified_expression_l70_70195


namespace painters_completing_rooms_l70_70472

theorem painters_completing_rooms (three_painters_three_rooms_three_hours : 3 * 3 * 3 ≥ 3 * 3) :
  9 * 3 * 9 ≥ 9 * 27 :=
by 
  sorry

end painters_completing_rooms_l70_70472


namespace total_texts_sent_l70_70044

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end total_texts_sent_l70_70044


namespace smallest_trees_in_three_types_l70_70176

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l70_70176


namespace gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l70_70077

def a : ℕ := 2^1025 - 1
def b : ℕ := 2^1056 - 1
def answer : ℕ := 2147483647

theorem gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1 :
  Int.gcd a b = answer := by
  sorry

end gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l70_70077


namespace evening_temperature_l70_70482

-- Define the given conditions
def t_noon : ℤ := 1
def d : ℤ := 3

-- The main theorem stating that the evening temperature is -2℃
theorem evening_temperature : t_noon - d = -2 := by
  sorry

end evening_temperature_l70_70482


namespace range_of_fraction_l70_70820

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∀ z, z = x / y → (1 / 6 ≤ z ∧ z ≤ 4 / 3) :=
sorry

end range_of_fraction_l70_70820


namespace cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l70_70613

theorem cube_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^3 + x₂^3 = 18 :=
sorry

theorem ratio_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  (x₂ / x₁) + (x₁ / x₂) = 7 :=
sorry

end cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l70_70613


namespace derivative_of_f_domain_of_f_range_of_f_l70_70074

open Real

noncomputable def f (x : ℝ) := 1 / (x + sqrt (1 + 2 * x^2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = - ((sqrt (1 + 2 * x^2) + 2 * x) / (sqrt (1 + 2 * x^2) * (x + sqrt (1 + 2 * x^2))^2)) :=
by
  sorry

theorem domain_of_f : ∀ x : ℝ, f x ≠ 0 :=
by
  sorry

theorem range_of_f : 
  ∀ y : ℝ, 0 < y ∧ y ≤ sqrt 2 → ∃ x : ℝ, f x = y :=
by
  sorry

end derivative_of_f_domain_of_f_range_of_f_l70_70074


namespace rhombus_area_3cm_45deg_l70_70583

noncomputable def rhombusArea (a : ℝ) (theta : ℝ) : ℝ :=
  a * (a * Real.sin theta)

theorem rhombus_area_3cm_45deg :
  rhombusArea 3 (Real.pi / 4) = 9 * Real.sqrt 2 / 2 := 
by
  sorry

end rhombus_area_3cm_45deg_l70_70583


namespace roots_quadratic_relation_l70_70593

theorem roots_quadratic_relation (a b c d A B : ℝ)
  (h1 : a^2 + A * a + 1 = 0)
  (h2 : b^2 + A * b + 1 = 0)
  (h3 : c^2 + B * c + 1 = 0)
  (h4 : d^2 + B * d + 1 = 0) :
  (a - c) * (b - c) * (a + d) * (b + d) = B^2 - A^2 :=
sorry

end roots_quadratic_relation_l70_70593


namespace sum_of_all_different_possible_areas_of_cool_rectangles_l70_70844

-- Define the concept of a cool rectangle
def is_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 2 * (2 * a + 2 * b)

-- Define the function to calculate the area of a rectangle
def area (a b : ℕ) : ℕ := a * b

-- Define the set of pairs (a, b) that satisfy the cool rectangle condition
def cool_rectangle_pairs : List (ℕ × ℕ) :=
  [(5, 20), (6, 12), (8, 8)]

-- Calculate the sum of all different possible areas of cool rectangles
def sum_of_cool_rectangle_areas : ℕ :=
  List.sum (cool_rectangle_pairs.map (λ p => area p.fst p.snd))

-- Theorem statement
theorem sum_of_all_different_possible_areas_of_cool_rectangles :
  sum_of_cool_rectangle_areas = 236 :=
by
  -- This is where the proof would go based on the given solution.
  sorry

end sum_of_all_different_possible_areas_of_cool_rectangles_l70_70844


namespace nat_divides_2_pow_n_minus_1_l70_70110

theorem nat_divides_2_pow_n_minus_1 (n : ℕ) (hn : 0 < n) : n ∣ 2^n - 1 ↔ n = 1 :=
  sorry

end nat_divides_2_pow_n_minus_1_l70_70110


namespace nancy_carrots_l70_70895

def carrots_total 
  (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

theorem nancy_carrots : 
  carrots_total 12 2 21 = 31 :=
by
  -- Add the proof here
  sorry

end nancy_carrots_l70_70895


namespace arithmetic_sequence_geometric_property_l70_70389

theorem arithmetic_sequence_geometric_property (a : ℕ → ℤ) (d : ℤ) (h_d : d = 2)
  (h_a3 : a 3 = a 1 + 4) (h_a4 : a 4 = a 1 + 6)
  (geo_seq : (a 1 + 4) * (a 1 + 4) = a 1 * (a 1 + 6)) :
  a 2 = -6 := sorry

end arithmetic_sequence_geometric_property_l70_70389


namespace third_speed_is_9_kmph_l70_70790

/-- Problem Statement: Given the total travel time, total distance, and two speeds, 
    prove that the third speed is 9 km/hr when distances are equal. -/
theorem third_speed_is_9_kmph (t : ℕ) (d_total : ℕ) (v1 v2 : ℕ) (d1 d2 d3 : ℕ) 
(h_t : t = 11)
(h_d_total : d_total = 900)
(h_v1 : v1 = 3)
(h_v2 : v2 = 6)
(h_d_eq : d1 = 300 ∧ d2 = 300 ∧ d3 = 300)
(h_sum_t : d1 / (v1 * 1000 / 60) + d2 / (v2 * 1000 / 60) + d3 / (v3 * 1000 / 60) = t) 
: (v3 = 9) :=
by 
  sorry

end third_speed_is_9_kmph_l70_70790


namespace length_AC_l70_70505

variable {A B C : Type} [Field A] [Field B] [Field C]

-- Definitions for the problem conditions
noncomputable def length_AB : ℝ := 3
noncomputable def angle_A : ℝ := Real.pi * 120 / 180
noncomputable def area_ABC : ℝ := (15 * Real.sqrt 3) / 4

-- The theorem statement
theorem length_AC (b : ℝ) (h1 : b = length_AB) (h2 : angle_A = Real.pi * 120 / 180) (h3 : area_ABC = (15 * Real.sqrt 3) / 4) : b = 5 :=
sorry

end length_AC_l70_70505


namespace tan_diff_angle_neg7_l70_70711

-- Define the main constants based on the conditions given
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -3/5
axiom alpha_in_fourth_quadrant : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2

-- Define the statement that needs to be proven based on the question and the correct answer
theorem tan_diff_angle_neg7 : 
  Real.tan (α - Real.pi / 4) = -7 :=
sorry

end tan_diff_angle_neg7_l70_70711


namespace geometric_sequence_ratio_l70_70414

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n+1) = q * a n)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 :=
sorry

end geometric_sequence_ratio_l70_70414


namespace heat_production_example_l70_70548

noncomputable def heat_produced_by_current (R : ℝ) (I : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
∫ (t : ℝ) in t1..t2, (I t)^2 * R

theorem heat_production_example :
  heat_produced_by_current 40 (λ t => 5 + 4 * t) 0 10 = 303750 :=
by
  sorry

end heat_production_example_l70_70548


namespace growth_rate_equation_l70_70031

variable (a x : ℝ)

-- Condition: The number of visitors in March is three times that of January
def visitors_in_march := 3 * a

-- Condition: The average growth rate of visitors in February and March is x
def growth_rate := x

-- Statement to prove
theorem growth_rate_equation 
  (h : (1 + x)^2 = 3) : true :=
by sorry

end growth_rate_equation_l70_70031


namespace range_of_a_l70_70524

-- Define an odd function f on ℝ such that f(x) = x^2 for x >= 0
noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 else -(x^2)

-- Prove the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc a (a + 2) → f (x - a) ≥ f (3 * x + 1)) →
  a ≤ -5 := sorry

end range_of_a_l70_70524


namespace fixed_point_for_any_k_l70_70199

-- Define the function f representing our quadratic equation
def f (k : ℝ) (x : ℝ) : ℝ :=
  8 * x^2 + 3 * k * x - 5 * k
  
-- The statement representing our proof problem
theorem fixed_point_for_any_k :
  ∀ (a b : ℝ), (∀ (k : ℝ), f k a = b) → (a, b) = (5, 200) :=
by
  sorry

end fixed_point_for_any_k_l70_70199


namespace cucumbers_count_l70_70147

theorem cucumbers_count (c : ℕ) (n : ℕ) (additional : ℕ) (initial_cucumbers : ℕ) (total_cucumbers : ℕ) :
  c = 4 → n = 10 → additional = 2 → initial_cucumbers = n - c → total_cucumbers = initial_cucumbers + additional → total_cucumbers = 8 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  simp at h4
  rw [h4, h3] at h5
  simp at h5
  exact h5

end cucumbers_count_l70_70147


namespace problem_l70_70090

noncomputable def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def quadratic_roots (a₃ a₁₀ : ℝ) : Prop :=
a₃^2 - 3 * a₃ - 5 = 0 ∧ a₁₀^2 - 3 * a₁₀ - 5 = 0

theorem problem (a : ℕ → ℝ) (h1 : is_arithmetic_seq a)
  (h2 : quadratic_roots (a 3) (a 10)) :
  a 5 + a 8 = 3 :=
sorry

end problem_l70_70090


namespace book_contains_300_pages_l70_70429

-- The given conditions
def total_digits : ℕ := 792
def digits_per_page_1_to_9 : ℕ := 9 * 1
def digits_per_page_10_to_99 : ℕ := 90 * 2
def remaining_digits : ℕ := total_digits - digits_per_page_1_to_9 - digits_per_page_10_to_99
def pages_with_3_digits : ℕ := remaining_digits / 3

-- The total number of pages
def total_pages : ℕ := 99 + pages_with_3_digits

theorem book_contains_300_pages : total_pages = 300 := by
  sorry

end book_contains_300_pages_l70_70429


namespace biography_increase_l70_70871

theorem biography_increase (B N : ℝ) (hN : N = 0.35 * (B + N) - 0.20 * B):
  (N / (0.20 * B) * 100) = 115.38 :=
by
  sorry

end biography_increase_l70_70871


namespace blue_balls_taken_out_l70_70638

theorem blue_balls_taken_out :
  ∃ x : ℕ, (0 ≤ x ∧ x ≤ 7) ∧ (7 - x) / (15 - x) = 1 / 3 ∧ x = 3 :=
sorry

end blue_balls_taken_out_l70_70638


namespace repeating_decimal_to_fraction_l70_70033

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l70_70033


namespace smallest_b_l70_70168

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x - 2 * a

theorem smallest_b (a : ℝ) (b : ℝ) (x : ℝ) : (1 < a ∧ a < 4) → (0 < x) → (f a b x > 0) → b ≥ 11 :=
by
  -- placeholder for the proof
  sorry

end smallest_b_l70_70168


namespace cookies_number_l70_70893

-- Define all conditions in the problem
def number_of_chips_per_cookie := 7
def number_of_cookies_per_dozen := 12
def number_of_uneaten_chips := 168

-- Define D as the number of dozens of cookies
variable (D : ℕ)

-- Prove the Lean theorem
theorem cookies_number (h : 7 * 6 * D = 168) : D = 4 :=
by
  sorry

end cookies_number_l70_70893


namespace product_of_points_l70_70726

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def Chris_rolls : List ℕ := [5, 2, 1, 6]
def Dana_rolls : List ℕ := [6, 2, 3, 3]

def Chris_points : ℕ := (Chris_rolls.map f).sum
def Dana_points : ℕ := (Dana_rolls.map f).sum

theorem product_of_points : Chris_points * Dana_points = 297 := by
  sorry

end product_of_points_l70_70726


namespace transaction_mistake_in_cents_l70_70234

theorem transaction_mistake_in_cents
  (x y : ℕ)
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (error_cents : 100 * y + x - (100 * x + y) = 5616) :
  y = x + 56 :=
by {
  sorry
}

end transaction_mistake_in_cents_l70_70234


namespace find_valid_triples_l70_70699

-- Define the theorem to prove the conditions and results
theorem find_valid_triples :
  ∀ (a b c : ℕ), 
    (2^a + 2^b + 1) % (2^c - 1) = 0 ↔ (a = 0 ∧ b = 0 ∧ c = 2) ∨ 
                                      (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                      (a = 2 ∧ b = 1 ∧ c = 3) := 
sorry  -- Proof omitted

end find_valid_triples_l70_70699


namespace dice_sum_prob_l70_70619

theorem dice_sum_prob :
  (3 / 6) * (3 / 6) * (2 / 5) * (1 / 6) * 2 = 13 / 216 :=
by sorry

end dice_sum_prob_l70_70619


namespace median_avg_scores_compare_teacher_avg_scores_l70_70724

-- Definitions of conditions
def class1_students (a : ℕ) := a
def class2_students (b : ℕ) := b
def class3_students (c : ℕ) := c
def class4_students (c : ℕ) := c

def avg_score_1 := 68
def avg_score_2 := 78
def avg_score_3 := 74
def avg_score_4 := 72

-- Part 1: Prove the median of the average scores.
theorem median_avg_scores : 
  let scores := [68, 72, 74, 78]
  ∃ m, m = 73 :=
by 
  sorry

-- Part 2: Prove that the average scores for Teacher Wang and Teacher Li are not necessarily the same.
theorem compare_teacher_avg_scores (a b c : ℕ) (h_ab : a ≠ 0 ∧ b ≠ 0) : 
  let wang_avg := (68 * a + 78 * b) / (a + b)
  let li_avg := 73
  wang_avg ≠ li_avg :=
by
  sorry

end median_avg_scores_compare_teacher_avg_scores_l70_70724


namespace intersection_of_sets_l70_70133

def setA := { x : ℝ | x / (x - 1) < 0 }
def setB := { x : ℝ | 0 < x ∧ x < 3 }
def setIntersect := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_of_sets :
  ∀ x : ℝ, x ∈ setA ∧ x ∈ setB ↔ x ∈ setIntersect := 
by
  sorry

end intersection_of_sets_l70_70133


namespace problem_solution_l70_70537

theorem problem_solution (x : ℝ) (h : x ≠ 5) : (x ≥ 8) ↔ ((x + 1) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l70_70537


namespace part1_part2_l70_70272

-- Conditions: Definitions of A and B
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem statements
theorem part1 (a b : ℝ) :  2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := sorry

theorem part2 (a : ℝ) (h : ∀ a, 2 * A a 2 - B a 2 = - 4 * a * 2 + 6 * 2 + 8 * a) : 2 = 2 := sorry

end part1_part2_l70_70272


namespace no_positive_solution_l70_70912

theorem no_positive_solution (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) :
  ¬ (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) :=
sorry

end no_positive_solution_l70_70912


namespace equivalent_statements_l70_70704

variable (P Q R : Prop)

theorem equivalent_statements :
  ((¬ P ∧ ¬ Q) → R) ↔ (P ∨ Q ∨ R) :=
sorry

end equivalent_statements_l70_70704


namespace even_square_minus_self_l70_70595

theorem even_square_minus_self (a : ℤ) : 2 ∣ (a^2 - a) :=
sorry

end even_square_minus_self_l70_70595


namespace max_value_proof_l70_70372

noncomputable def maximum_value (x y z : ℝ) : ℝ := 
  (2/x) + (1/y) - (2/z) + 2

theorem max_value_proof {x y z : ℝ} 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0):
  maximum_value x y z ≤ 3 :=
sorry

end max_value_proof_l70_70372


namespace gcd_of_gx_and_x_l70_70388

theorem gcd_of_gx_and_x (x : ℤ) (hx : x % 11739 = 0) :
  Int.gcd ((3 * x + 4) * (5 * x + 3) * (11 * x + 5) * (x + 11)) x = 3 :=
sorry

end gcd_of_gx_and_x_l70_70388


namespace commute_proof_l70_70832

noncomputable def commute_problem : Prop :=
  let d : ℝ := 1.5 -- distance in miles
  let v_w : ℝ := 3 -- walking speed in miles per hour
  let v_t : ℝ := 20 -- train speed in miles per hour
  let walking_minutes : ℝ := (d / v_w) * 60 -- walking time in minutes
  let train_minutes : ℝ := (d / v_t) * 60 -- train time in minutes
  ∃ x : ℝ, walking_minutes = train_minutes + x + 25 ∧ x = 0.5

theorem commute_proof : commute_problem :=
  sorry

end commute_proof_l70_70832


namespace convex_polygon_angles_eq_nine_l70_70515

theorem convex_polygon_angles_eq_nine (n : ℕ) (a : ℕ → ℝ) (d : ℝ)
  (h1 : a (n - 1) = 180)
  (h2 : ∀ k, a (k + 1) - a k = d)
  (h3 : d = 10) :
  n = 9 :=
by
  sorry

end convex_polygon_angles_eq_nine_l70_70515


namespace problem_statement_l70_70621

-- Define the odd function and the conditions given
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Main theorem statement
theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (3 - x))
  (h_f1 : f 1 = -2) :
  2012 * f 2012 - 2013 * f 2013 = -4026 := 
sorry

end problem_statement_l70_70621


namespace solve_absolute_value_equation_l70_70026

theorem solve_absolute_value_equation :
  {x : ℝ | 3 * x^2 + 3 * x + 6 = abs (-20 + 5 * x)} = {1.21, -3.87} :=
by
  sorry

end solve_absolute_value_equation_l70_70026


namespace range_of_a_l70_70498

noncomputable def interval1 (a : ℝ) : Prop := -2 < a ∧ a <= 1 / 2
noncomputable def interval2 (a : ℝ) : Prop := a >= 2

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * a| > 1

theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, p a ∨ q a) (h2 : ¬ (∀ x : ℝ, p a ∧ q a)) : 
  interval1 a ∨ interval2 a :=
sorry

end range_of_a_l70_70498


namespace total_medals_1996_l70_70577

variable (g s b : Nat)

theorem total_medals_1996 (h_g : g = 16) (h_s : s = 22) (h_b : b = 12) :
  g + s + b = 50 :=
by
  sorry

end total_medals_1996_l70_70577


namespace entrance_exit_plans_l70_70098

-- Definitions as per the conditions in the problem
def south_gates : Nat := 4
def north_gates : Nat := 3
def west_gates : Nat := 2

-- Conditions translated into Lean definitions
def ways_to_enter := south_gates + north_gates
def ways_to_exit := west_gates + north_gates

-- The theorem to be proved: the number of entrance and exit plans
theorem entrance_exit_plans : ways_to_enter * ways_to_exit = 35 := by
  sorry

end entrance_exit_plans_l70_70098


namespace fg_of_2_eq_225_l70_70626

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem fg_of_2_eq_225 : f (g 2) = 225 := by
  sorry

end fg_of_2_eq_225_l70_70626


namespace sin_75_equals_sqrt_1_plus_sin_2_equals_l70_70437

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)
noncomputable def sqrt_1_plus_sin_2 : ℝ := Real.sqrt (1 + Real.sin 2)

theorem sin_75_equals :
  sin_75 = (Real.sqrt 2 + Real.sqrt 6) / 4 := 
sorry

theorem sqrt_1_plus_sin_2_equals :
  sqrt_1_plus_sin_2 = Real.sin 1 + Real.cos 1 := 
sorry

end sin_75_equals_sqrt_1_plus_sin_2_equals_l70_70437


namespace Sanji_received_86_coins_l70_70692

noncomputable def total_coins := 280

def Jack_coins (x : ℕ) := x
def Jimmy_coins (x : ℕ) := x + 11
def Tom_coins (x : ℕ) := x - 15
def Sanji_coins (x : ℕ) := x + 20

theorem Sanji_received_86_coins (x : ℕ) (hx : Jack_coins x + Jimmy_coins x + Tom_coins x + Sanji_coins x = total_coins) : Sanji_coins x = 86 :=
sorry

end Sanji_received_86_coins_l70_70692


namespace total_towels_l70_70754

theorem total_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) : packs * towels_per_pack = 27 := by
  sorry

end total_towels_l70_70754


namespace jill_total_tax_percentage_l70_70592

theorem jill_total_tax_percentage (total_spent : ℝ) 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ)
  (tax_clothing_rate : ℝ) (tax_food_rate : ℝ) (tax_other_rate : ℝ)
  (h_clothing : spent_clothing = 0.45 * total_spent)
  (h_food : spent_food = 0.45 * total_spent)
  (h_other : spent_other = 0.10 * total_spent)
  (h_tax_clothing : tax_clothing_rate = 0.05)
  (h_tax_food : tax_food_rate = 0.0)
  (h_tax_other : tax_other_rate = 0.10) :
  ((spent_clothing * tax_clothing_rate + spent_food * tax_food_rate + spent_other * tax_other_rate) / total_spent) * 100 = 3.25 :=
by
  sorry

end jill_total_tax_percentage_l70_70592


namespace larger_number_is_33_l70_70811

theorem larger_number_is_33 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : max x y = 33 :=
sorry

end larger_number_is_33_l70_70811


namespace peyton_juice_boxes_needed_l70_70479

def juice_boxes_needed
  (john_juice_per_day : ℕ)
  (samantha_juice_per_day : ℕ)
  (heather_juice_mon_wed : ℕ)
  (heather_juice_tue_thu : ℕ)
  (heather_juice_fri : ℕ)
  (john_weeks : ℕ)
  (samantha_weeks : ℕ)
  (heather_weeks : ℕ)
  : ℕ :=
  let john_juice_per_week := john_juice_per_day * 5
  let samantha_juice_per_week := samantha_juice_per_day * 5
  let heather_juice_per_week := heather_juice_mon_wed * 2 + heather_juice_tue_thu * 2 + heather_juice_fri
  let john_total_juice := john_juice_per_week * john_weeks
  let samantha_total_juice := samantha_juice_per_week * samantha_weeks
  let heather_total_juice := heather_juice_per_week * heather_weeks
  john_total_juice + samantha_total_juice + heather_total_juice

theorem peyton_juice_boxes_needed :
  juice_boxes_needed 2 1 3 2 1 25 20 25 = 625 :=
by
  sorry

end peyton_juice_boxes_needed_l70_70479


namespace exists_polynomials_Q_R_l70_70117

noncomputable def polynomial_with_integer_coeff (P : Polynomial ℤ) : Prop :=
  true

theorem exists_polynomials_Q_R (P : Polynomial ℤ) (hP : polynomial_with_integer_coeff P) :
  ∃ (Q R : Polynomial ℤ), 
    (∃ g : Polynomial ℤ, P * Q = Polynomial.comp g (Polynomial.X ^ 2)) ∧ 
    (∃ h : Polynomial ℤ, P * R = Polynomial.comp h (Polynomial.X ^ 3)) :=
by
  sorry

end exists_polynomials_Q_R_l70_70117


namespace multiply_63_57_l70_70365

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l70_70365


namespace find_contaminated_constant_l70_70025

theorem find_contaminated_constant (contaminated_constant : ℝ) (x : ℝ) 
  (h1 : 2 * (x - 3) - contaminated_constant = x + 1) 
  (h2 : x = 9) : contaminated_constant = 2 :=
  sorry

end find_contaminated_constant_l70_70025


namespace no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l70_70427

theorem no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant :
    ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
                     (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) →
                     (a^2 + b^2 + c^2 + d^2 = 100) → False := by
  sorry

end no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l70_70427


namespace evaluate_expression_l70_70460

theorem evaluate_expression :
  ((3^1 - 2 + 7^3 + 1 : ℚ)⁻¹ * 6) = (2 / 115) := by
  sorry

end evaluate_expression_l70_70460


namespace find_sale_month4_l70_70625

-- Define sales for each month
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month5 : ℕ := 4500
def sale_month6 : ℕ := 1200
def avg_sale_per_month : ℕ := 5600

-- Define the total number of months
def num_months : ℕ := 6

-- Define the expression for total sales required
def total_sales_required : ℕ := avg_sale_per_month * num_months

-- Define the expression for total known sales
def total_known_sales : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6

-- State and prove the theorem:
theorem find_sale_month4 : sale_month1 = 5400 → sale_month2 = 9000 → sale_month3 = 6300 → 
                            sale_month5 = 4500 → sale_month6 = 1200 → avg_sale_per_month = 5600 →
                            num_months = 6 → (total_sales_required - total_known_sales = 8200) := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end find_sale_month4_l70_70625


namespace triangle_inequality_l70_70491

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c R : ℝ) : ℝ := a * b * c / (4 * R)
noncomputable def inradius_area (a b c r : ℝ) : ℝ := semiperimeter a b c * r

theorem triangle_inequality (a b c R r : ℝ) (h₁ : a ≤ 1) (h₂ : b ≤ 1) (h₃ : c ≤ 1)
  (h₄ : area a b c R = semiperimeter a b c * r) : 
  semiperimeter a b c * (1 - 2 * R * r) ≥ 1 :=
by 
  -- Proof goes here
  sorry

end triangle_inequality_l70_70491


namespace price_of_when_you_rescind_cd_l70_70776

variable (W : ℕ) -- Defining W as a natural number since prices can't be negative

theorem price_of_when_you_rescind_cd
  (price_life_journey : ℕ := 100)
  (price_day_life : ℕ := 50)
  (num_cds_each : ℕ := 3)
  (total_spent : ℕ := 705) :
  3 * price_life_journey + 3 * price_day_life + 3 * W = total_spent → 
  W = 85 :=
by
  intros h
  sorry

end price_of_when_you_rescind_cd_l70_70776


namespace power_function_solution_l70_70212

theorem power_function_solution (m : ℤ)
  (h1 : ∃ (f : ℝ → ℝ), ∀ x : ℝ, f x = x^(-m^2 + 2 * m + 3) ∧ ∀ x, f x = f (-x))
  (h2 : ∀ x : ℝ, x > 0 → (x^(-m^2 + 2 * m + 3)) < x^(-m^2 + 2 * m + 3 + x)) :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^4 :=
by
  sorry

end power_function_solution_l70_70212


namespace phi_range_l70_70818

noncomputable def f (ω φ x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + φ) + 1

theorem phi_range (ω φ : ℝ) 
  (h₀ : ω > 0)
  (h₁ : |φ| ≤ Real.pi / 2)
  (h₂ : ∃ x₁ x₂, x₁ ≠ x₂ ∧ f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₂ - x₁| = Real.pi / 3)
  (h₃ : ∀ x, x ∈ Set.Ioo (-Real.pi / 8) (Real.pi / 3) → f ω φ x > 1) :
  φ ∈ Set.Icc (Real.pi / 4) (Real.pi / 3) :=
sorry

end phi_range_l70_70818


namespace sunflower_seeds_more_than_half_on_day_three_l70_70322

-- Define the initial state and parameters
def initial_sunflower_seeds : ℚ := 0.4
def initial_other_seeds : ℚ := 0.6
def daily_added_sunflower_seeds : ℚ := 0.2
def daily_added_other_seeds : ℚ := 0.3
def daily_sunflower_eaten_factor : ℚ := 0.7
def daily_other_eaten_factor : ℚ := 0.4

-- Define the recurrence relations for sunflower seeds and total seeds
def sunflower_seeds (n : ℕ) : ℚ :=
  match n with
  | 0     => initial_sunflower_seeds
  | (n+1) => daily_sunflower_eaten_factor * sunflower_seeds n + daily_added_sunflower_seeds

def total_seeds (n : ℕ) : ℚ := 1 + (n : ℚ) * 0.5

-- Define the main theorem stating that on Tuesday (Day 3), sunflower seeds are more than half
theorem sunflower_seeds_more_than_half_on_day_three : sunflower_seeds 2 / total_seeds 2 > 0.5 :=
by
  -- Formal proof will go here
  sorry

end sunflower_seeds_more_than_half_on_day_three_l70_70322


namespace prob_both_A_B_prob_exactly_one_l70_70958

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end prob_both_A_B_prob_exactly_one_l70_70958


namespace solve_for_x_l70_70174

theorem solve_for_x :
  ∃ x : ℝ, 40 + (5 * x) / (180 / 3) = 41 ∧ x = 12 :=
by
  sorry

end solve_for_x_l70_70174


namespace min_value_l70_70009

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) : 
  ∃ m, m = (1 / x + 1 / y) ∧ m = 9 :=
by
  sorry

end min_value_l70_70009


namespace surface_area_of_cube_l70_70922

theorem surface_area_of_cube (edge : ℝ) (h : edge = 5) : 6 * (edge * edge) = 150 := by
  have h_square : edge * edge = 25 := by
    rw [h]
    norm_num
  rw [h_square]
  norm_num

end surface_area_of_cube_l70_70922


namespace roots_of_quadratic_l70_70084

theorem roots_of_quadratic (x : ℝ) : (5 * x^2 = 4 * x) → (x = 0 ∨ x = 4 / 5) :=
by
  sorry

end roots_of_quadratic_l70_70084


namespace num_people_second_hour_l70_70945

theorem num_people_second_hour 
  (n1_in n2_in n1_left n2_left : ℕ) 
  (rem_hour1 rem_hour2 : ℕ)
  (h1 : n1_in = 94)
  (h2 : n1_left = 27)
  (h3 : n2_left = 9)
  (h4 : rem_hour2 = 76)
  (h5 : rem_hour1 = n1_in - n1_left)
  (h6 : rem_hour2 = rem_hour1 + n2_in - n2_left) :
  n2_in = 18 := 
  by 
  sorry

end num_people_second_hour_l70_70945


namespace least_faces_combined_l70_70616

noncomputable def num_faces_dice_combined : ℕ :=
  let a := 11
  let b := 7
  a + b

/-- Given the conditions on the dice setups for sums of 8, 11, and 15,
the least number of faces on the two dice combined is 18. -/
theorem least_faces_combined (a b : ℕ) (h1 : 6 < a) (h2 : 6 < b)
  (h_sum_8 : ∃ (p : ℕ), p = 7)  -- 7 ways to roll a sum of 8
  (h_sum_11 : ∃ (q : ℕ), q = 14)  -- half probability means 14 ways to roll a sum of 11
  (h_sum_15 : ∃ (r : ℕ), r = 2) : a + b = 18 :=
by
  sorry

end least_faces_combined_l70_70616


namespace inscribed_square_side_length_l70_70989

theorem inscribed_square_side_length (AC BC : ℝ) (h₀ : AC = 6) (h₁ : BC = 8) :
  ∃ x : ℝ, x = 24 / 7 :=
by
  sorry

end inscribed_square_side_length_l70_70989


namespace pentagon_diagl_sum_pentagon_diagonal_391_l70_70782

noncomputable def diagonal_sum (AB CD BC DE AE : ℕ) 
  (AC : ℚ) (BD : ℚ) (CE : ℚ) (AD : ℚ) (BE : ℚ) : ℚ :=
  3 * AC + AD + BE

theorem pentagon_diagl_sum (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  diagonal_sum AB CD BC DE AE AC BD CE AD BE = 385 / 6 := sorry

theorem pentagon_diagonal_391 (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  ∃ m n : ℕ, 
    m.gcd n = 1 ∧
    m / n = 385 / 6 ∧
    m + n = 391 := sorry

end pentagon_diagl_sum_pentagon_diagonal_391_l70_70782


namespace setB_is_empty_l70_70063

noncomputable def setB := {x : ℝ | x^2 + 1 = 0}

theorem setB_is_empty : setB = ∅ :=
by
  sorry

end setB_is_empty_l70_70063


namespace range_of_a_l70_70866

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end range_of_a_l70_70866


namespace rental_cost_per_day_l70_70001

theorem rental_cost_per_day (p m c : ℝ) (d : ℝ) (hc : c = 0.08) (hm : m = 214.0) (hp : p = 46.12) (h_total : p = d + m * c) : d = 29.00 := 
by
  sorry

end rental_cost_per_day_l70_70001


namespace coffee_mix_price_per_pound_l70_70865

-- Definitions based on conditions
def total_weight : ℝ := 100
def columbian_price_per_pound : ℝ := 8.75
def brazilian_price_per_pound : ℝ := 3.75
def columbian_weight : ℝ := 52
def brazilian_weight : ℝ := total_weight - columbian_weight

-- Goal to prove
theorem coffee_mix_price_per_pound :
  (columbian_weight * columbian_price_per_pound + brazilian_weight * brazilian_price_per_pound) / total_weight = 6.35 :=
by
  sorry

end coffee_mix_price_per_pound_l70_70865


namespace series_solution_eq_l70_70030

theorem series_solution_eq (x : ℝ) 
  (h : (∃ a : ℕ → ℝ, (∀ n, a n = 1 + 6 * n) ∧ (∑' n, a n * x^n = 100))) :
  x = 23/25 ∨ x = 1/50 :=
sorry

end series_solution_eq_l70_70030


namespace percentage_reduction_l70_70660

theorem percentage_reduction (y x z p q : ℝ) (hy : y ≠ 0) (h1 : x = y - 10) (h2 : z = y - 20) :
  p = 1000 / y ∧ q = 2000 / y := by
  sorry

end percentage_reduction_l70_70660


namespace closing_price_l70_70153

theorem closing_price (opening_price : ℝ) (percent_increase : ℝ) (closing_price : ℝ) 
  (h₀ : opening_price = 6) (h₁ : percent_increase = 0.3333) : closing_price = 8 :=
by
  sorry

end closing_price_l70_70153


namespace added_water_is_18_l70_70146

def capacity : ℕ := 40

def initial_full_percent : ℚ := 0.30

def final_full_fraction : ℚ := 3/4

def initial_water (capacity : ℕ) (initial_full_percent : ℚ) : ℚ :=
  initial_full_percent * capacity

def final_water (capacity : ℕ) (final_full_fraction : ℚ) : ℚ :=
  final_full_fraction * capacity

def water_added (initial_water : ℚ) (final_water : ℚ) : ℚ :=
  final_water - initial_water

theorem added_water_is_18 :
  water_added (initial_water capacity initial_full_percent) (final_water capacity final_full_fraction) = 18 := by
  sorry

end added_water_is_18_l70_70146


namespace tom_saves_80_dollars_l70_70869

def normal_doctor_cost : ℝ := 200
def discount_percentage : ℝ := 0.7
def discount_clinic_cost_per_visit : ℝ := normal_doctor_cost * (1 - discount_percentage)
def number_of_visits : ℝ := 2
def total_discount_clinic_cost : ℝ := discount_clinic_cost_per_visit * number_of_visits
def savings : ℝ := normal_doctor_cost - total_discount_clinic_cost

theorem tom_saves_80_dollars : savings = 80 := by
  sorry

end tom_saves_80_dollars_l70_70869


namespace number_of_pigs_l70_70492

variable (cows pigs : Nat)

theorem number_of_pigs (h1 : 2 * (7 + pigs) = 32) : pigs = 9 := by
  sorry

end number_of_pigs_l70_70492


namespace arithmetic_sequence_sum_S15_l70_70681

theorem arithmetic_sequence_sum_S15 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hs5 : S 5 = 10) (hs10 : S 10 = 30) 
  (has : ∀ n, S n = n * (2 * a 1 + (n - 1) * a 2) / 2) : 
  S 15 = 60 := 
sorry

end arithmetic_sequence_sum_S15_l70_70681


namespace minimum_x2_y2_z2_l70_70758

theorem minimum_x2_y2_z2 :
  ∀ x y z : ℝ, (x^3 + y^3 + z^3 - 3 * x * y * z = 1) → (∃ a b c : ℝ, a = x ∨ a = y ∨ a = z ∧ b = x ∨ b = y ∨ b = z ∧ c = x ∨ c = y ∨ a ≤ b ∨ a ≤ c ∧ b ≤ c) → (x^2 + y^2 + z^2 ≥ 1) :=
by
  sorry

end minimum_x2_y2_z2_l70_70758


namespace fractions_addition_l70_70894

theorem fractions_addition :
  (1 / 3) * (3 / 4) * (1 / 5) + (1 / 6) = 13 / 60 :=
by 
  sorry

end fractions_addition_l70_70894


namespace find_n_l70_70666

-- Definitions based on the given conditions
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- The mathematically equivalent proof problem statement:
theorem find_n (n : ℕ) (p : ℝ) (h1 : binomial_expectation n p = 6) (h2 : binomial_variance n p = 3) : n = 12 :=
sorry

end find_n_l70_70666


namespace symmetric_points_y_axis_l70_70859

theorem symmetric_points_y_axis (a b : ℝ) (h1 : a - b = -3) (h2 : 2 * a + b = 2) :
  a = -1 / 3 ∧ b = 8 / 3 :=
by
  sorry

end symmetric_points_y_axis_l70_70859


namespace find_p_l70_70897

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (h1 : p + q = r + 2) (h2 : 1 < p) (h3 : p < q) :
  p = 2 := 
sorry

end find_p_l70_70897


namespace consecutive_weights_sum_to_63_l70_70412

theorem consecutive_weights_sum_to_63 : ∃ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 63 :=
by
  sorry

end consecutive_weights_sum_to_63_l70_70412


namespace largest_angle_in_ratio_3_4_5_l70_70898

theorem largest_angle_in_ratio_3_4_5 : ∃ (A B C : ℝ), (A / 3 = B / 4 ∧ B / 4 = C / 5) ∧ (A + B + C = 180) ∧ (C = 75) :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l70_70898


namespace leo_amount_after_settling_debts_l70_70376

theorem leo_amount_after_settling_debts (total_amount : ℝ) (ryan_share : ℝ) (ryan_owes_leo : ℝ) (leo_owes_ryan : ℝ) 
  (h1 : total_amount = 48) 
  (h2 : ryan_share = (2 / 3) * total_amount) 
  (h3 : ryan_owes_leo = 10) 
  (h4 : leo_owes_ryan = 7) : 
  (total_amount - ryan_share) + (ryan_owes_leo - leo_owes_ryan) = 19 :=
by
  sorry

end leo_amount_after_settling_debts_l70_70376


namespace largest_plot_area_l70_70247

def plotA_area : Real := 10
def plotB_area : Real := 10 + 1
def plotC_area : Real := 9 + 1.5
def plotD_area : Real := 12
def plotE_area : Real := 11 + 1

theorem largest_plot_area :
  max (max (max (max plotA_area plotB_area) plotC_area) plotD_area) plotE_area = 12 ∧ 
  (plotD_area = 12 ∧ plotE_area = 12) := by sorry

end largest_plot_area_l70_70247


namespace function_characterization_l70_70932
noncomputable def f : ℕ → ℕ := sorry

theorem function_characterization (h : ∀ m n : ℕ, m^2 + f n ∣ m * f m + n) : 
  ∀ n : ℕ, f n = n :=
by
  intro n
  sorry

end function_characterization_l70_70932


namespace unpainted_unit_cubes_l70_70874

theorem unpainted_unit_cubes (total_cubes painted_faces edge_overlaps corner_overlaps : ℕ) :
  total_cubes = 6 * 6 * 6 ∧
  painted_faces = 6 * (2 * 6) ∧
  edge_overlaps = 12 * 3 / 2 ∧
  corner_overlaps = 8 ∧
  total_cubes - (painted_faces - edge_overlaps - corner_overlaps) = 170 :=
by
  sorry

end unpainted_unit_cubes_l70_70874


namespace find_x_l70_70138

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l70_70138


namespace table_tennis_possible_outcomes_l70_70386

-- Two people are playing a table tennis match. The first to win 3 games wins the match.
-- The match continues until a winner is determined.
-- Considering all possible outcomes (different numbers of wins and losses for each player are considered different outcomes),
-- prove that there are a total of 30 possible outcomes.

theorem table_tennis_possible_outcomes : 
  ∃ total_outcomes : ℕ, total_outcomes = 30 := 
by
  -- We need to prove that the total number of possible outcomes is 30
  sorry

end table_tennis_possible_outcomes_l70_70386


namespace product_of_three_greater_than_product_of_two_or_four_l70_70188

theorem product_of_three_greater_than_product_of_two_or_four
  (nums : Fin 10 → ℝ)
  (h_positive : ∀ i, 0 < nums i)
  (h_distinct : Function.Injective nums) :
  ∃ (a b c : Fin 10),
    (∃ (d e : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (c ≠ d) ∧ (c ≠ e) ∧ nums a * nums b * nums c > nums d * nums e) ∨
    (∃ (d e f g : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ nums a * nums b * nums c > nums d * nums e * nums f * nums g) :=
sorry

end product_of_three_greater_than_product_of_two_or_four_l70_70188


namespace quadratic_function_properties_l70_70841

-- Definitions based on given conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def pointCondition (a b c : ℝ) : Prop := quadraticFunction a b c (-2) = 0
def inequalityCondition (a b c : ℝ) : Prop := ∀ x : ℝ, 2 * x ≤ quadraticFunction a b c x ∧ quadraticFunction a b c x ≤ (1 / 2) * x^2 + 2
def strengthenCondition (f : ℝ → ℝ) (t : ℝ) : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f (x + t) < f (x / 3)

-- Our primary statement to prove
theorem quadratic_function_properties :
  ∃ a b c, pointCondition a b c ∧ inequalityCondition a b c ∧
           (a = 1 / 4 ∧ b = 1 ∧ c = 1) ∧
           (∀ t, (-8 / 3 < t ∧ t < -2 / 3) ↔ strengthenCondition (quadraticFunction (1 / 4) 1 1) t) :=
by sorry 

end quadratic_function_properties_l70_70841


namespace intercept_sum_l70_70506

theorem intercept_sum (x y : ℝ) (h : y - 3 = -3 * (x + 2)) :
  (∃ (x_int : ℝ), y = 0 ∧ x_int = -1) ∧ (∃ (y_int : ℝ), x = 0 ∧ y_int = -3) →
  (-1 + (-3) = -4) := by
  sorry

end intercept_sum_l70_70506


namespace min_x_plus_y_l70_70402

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
by
  sorry

end min_x_plus_y_l70_70402


namespace johns_horses_l70_70825

theorem johns_horses 
  (feeding_per_day : ℕ := 2) 
  (food_per_feeding : ℝ := 20) 
  (bag_weight : ℝ := 1000) 
  (num_bags : ℕ := 60) 
  (days : ℕ := 60)
  (total_food : ℝ := num_bags * bag_weight) 
  (daily_food_consumption : ℝ := total_food / days) 
  (food_per_horse_per_day : ℝ := food_per_feeding * feeding_per_day) :
  ∀ H : ℝ, (daily_food_consumption / food_per_horse_per_day = H) → H = 25 := 
by
  intros H hH
  sorry

end johns_horses_l70_70825


namespace line_through_fixed_point_and_parabola_l70_70930

theorem line_through_fixed_point_and_parabola :
  (∀ (a : ℝ), ∃ (P : ℝ × ℝ), 
    (a - 1) * P.1 - P.2 + 2 * a + 1 = 0 ∧ 
    (∀ (x y : ℝ), (y^2 = - ((9:ℝ) / 2) * x ∧ x = -2 ∧ y = 3) ∨ (x^2 = (4:ℝ) / 3 * y ∧ x = -2 ∧ y = 3))) :=
by
  sorry

end line_through_fixed_point_and_parabola_l70_70930


namespace compute_expression_l70_70512

theorem compute_expression :
  25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := 
sorry

end compute_expression_l70_70512


namespace charges_needed_to_vacuum_house_l70_70010

-- Conditions definitions
def battery_last_minutes : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def number_of_bedrooms : ℕ := 3
def number_of_kitchens : ℕ := 1
def number_of_living_rooms : ℕ := 1

-- Question (proof problem statement)
theorem charges_needed_to_vacuum_house :
  ((number_of_bedrooms + number_of_kitchens + number_of_living_rooms) * vacuum_time_per_room) / battery_last_minutes = 2 :=
by
  sorry

end charges_needed_to_vacuum_house_l70_70010


namespace original_cost_l70_70189

theorem original_cost (P : ℝ) (h : 0.85 * 0.76 * P = 988) : P = 1529.41 := by
  sorry

end original_cost_l70_70189


namespace sum_simplest_form_probability_eq_7068_l70_70301

/-- A jar has 15 red candies and 20 blue candies. Terry picks three candies at random,
    then Mary picks three of the remaining candies at random.
    Given that the probability that they get the same color combination (all reds or all blues, irrespective of order),
    find this probability in the simplest form. The sum of the numerator and denominator in simplest form is: 7068. -/
noncomputable def problem_statement : Nat :=
  let total_candies := 15 + 20;
  let terry_red_prob := (15 * 14 * 13) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_red_prob := (12 * 11 * 10) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_red := terry_red_prob * mary_red_prob;

  let terry_blue_prob := (20 * 19 * 18) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_blue_prob := (17 * 16 * 15) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_blue := terry_blue_prob * mary_blue_prob;

  let total_probability := both_red + both_blue;
  let simplest := 243 / 6825; -- This should be simplified form
  243 + 6825 -- Sum of numerator and denominator

theorem sum_simplest_form_probability_eq_7068 : problem_statement = 7068 :=
by sorry

end sum_simplest_form_probability_eq_7068_l70_70301


namespace sara_total_spent_l70_70573

-- Definitions based on the conditions
def ticket_price : ℝ := 10.62
def discount_rate : ℝ := 0.10
def rented_movie : ℝ := 1.59
def bought_movie : ℝ := 13.95
def snacks : ℝ := 7.50
def sales_tax_rate : ℝ := 0.05

-- Problem statement
theorem sara_total_spent : 
  let total_tickets := 2 * ticket_price
  let discount := total_tickets * discount_rate
  let discounted_tickets := total_tickets - discount
  let subtotal := discounted_tickets + rented_movie + bought_movie
  let sales_tax := subtotal * sales_tax_rate
  let total_with_tax := subtotal + sales_tax
  let total_amount := total_with_tax + snacks
  total_amount = 43.89 :=
by
  sorry

end sara_total_spent_l70_70573


namespace largest_of_sums_l70_70797

noncomputable def a1 := (1 / 4 : ℚ) + (1 / 5 : ℚ)
noncomputable def a2 := (1 / 4 : ℚ) + (1 / 6 : ℚ)
noncomputable def a3 := (1 / 4 : ℚ) + (1 / 3 : ℚ)
noncomputable def a4 := (1 / 4 : ℚ) + (1 / 8 : ℚ)
noncomputable def a5 := (1 / 4 : ℚ) + (1 / 7 : ℚ)

theorem largest_of_sums :
  max a1 (max a2 (max a3 (max a4 a5))) = 7 / 12 :=
by sorry

end largest_of_sums_l70_70797


namespace no_solutions_abs_eq_quadratic_l70_70379

theorem no_solutions_abs_eq_quadratic (x : ℝ) : ¬ (|x - 3| = x^2 + 2 * x + 4) := 
by
  sorry

end no_solutions_abs_eq_quadratic_l70_70379


namespace smallest_five_digit_multiple_of_53_l70_70783

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l70_70783


namespace correct_exponent_operation_l70_70806

theorem correct_exponent_operation (a : ℝ) : a^4 / a^3 = a := 
by
  sorry

end correct_exponent_operation_l70_70806


namespace alloy_price_per_kg_l70_70081

theorem alloy_price_per_kg (cost_A cost_B ratio_A_B total_cost total_weight price_per_kg : ℤ)
  (hA : cost_A = 68) 
  (hB : cost_B = 96) 
  (hRatio : ratio_A_B = 3) 
  (hTotalCost : total_cost = 3 * cost_A + cost_B) 
  (hTotalWeight : total_weight = 3 + 1)
  (hPricePerKg : price_per_kg = total_cost / total_weight) : 
  price_per_kg = 75 := 
by
  sorry

end alloy_price_per_kg_l70_70081


namespace quadratic_equation_solution_l70_70078

theorem quadratic_equation_solution :
  ∃ x1 x2 : ℝ, (x1 = (-1 + Real.sqrt 13) / 2 ∧ x2 = (-1 - Real.sqrt 13) / 2 
  ∧ (∀ x : ℝ, x^2 + x - 3 = 0 → x = x1 ∨ x = x2)) :=
sorry

end quadratic_equation_solution_l70_70078


namespace area_of_circumcircle_l70_70350

-- Define the problem:
theorem area_of_circumcircle 
  (a b c : ℝ) 
  (A B C : Real) 
  (h_cosC : Real.cos C = (2 * Real.sqrt 2) / 3) 
  (h_bcosA_acoB : b * Real.cos A + a * Real.cos B = 2)
  (h_sides : c = 2):
  let sinC := Real.sqrt (1 - (2 * Real.sqrt 2 / 3)^2)
  let R := c / (2 * sinC)
  let area := Real.pi * R^2
  area = 9 * Real.pi / 5 :=
by 
  sorry

end area_of_circumcircle_l70_70350


namespace pizza_slices_with_both_toppings_l70_70920

theorem pizza_slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices n : ℕ) 
    (h1 : total_slices = 14) 
    (h2 : pepperoni_slices = 8) 
    (h3 : mushroom_slices = 12) 
    (h4 : ∀ s, s = pepperoni_slices + mushroom_slices - n ∧ s = total_slices := by sorry) :
    n = 6 :=
sorry

end pizza_slices_with_both_toppings_l70_70920


namespace intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l70_70518

noncomputable def f (x : ℝ) := x * Real.log (-x)
noncomputable def g (x a : ℝ) := x * f (a * x) - Real.exp (x - 2)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < -1 / Real.exp 1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 / Real.exp 1 < x ∧ x < 0 → deriv f x < 0) ∧
  f (-1 / Real.exp 1) = 1 / Real.exp 1 :=
sorry

theorem number_of_zeros_of_g (a : ℝ) :
  (a > 0 ∨ a = -1 / Real.exp 1 → ∃! x : ℝ, g x a = 0) ∧
  (a < 0 ∧ a ≠ -1 / Real.exp 1 → ∀ x : ℝ, g x a ≠ 0) :=
sorry

end intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l70_70518


namespace ms_cole_total_students_l70_70303

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end ms_cole_total_students_l70_70303


namespace find_loss_percentage_l70_70576

theorem find_loss_percentage (W : ℝ) (profit_percentage : ℝ) (remaining_percentage : ℝ)
  (overall_loss : ℝ) (stock_worth : ℝ) (L : ℝ) :
  W = 12499.99 →
  profit_percentage = 0.20 →
  remaining_percentage = 0.80 →
  overall_loss = -500 →
  0.04 * W - (L / 100) * (remaining_percentage * W) = overall_loss →
  L = 10 :=
by
  intro hW hprofit_percentage hremaining_percentage hoverall_loss heq
  -- We'll provide the proof here
  sorry

end find_loss_percentage_l70_70576


namespace gyeonghun_climbing_l70_70167

variable (t_up t_down d_up d_down : ℝ)
variable (h1 : t_up + t_down = 4) 
variable (h2 : d_down = d_up + 2)
variable (h3 : t_up = d_up / 3)
variable (h4 : t_down = d_down / 4)

theorem gyeonghun_climbing (h1 : t_up + t_down = 4) (h2 : d_down = d_up + 2) (h3 : t_up = d_up / 3) (h4 : t_down = d_down / 4) :
  t_up = 2 :=
by
  sorry

end gyeonghun_climbing_l70_70167


namespace original_length_of_ribbon_l70_70005

theorem original_length_of_ribbon (n : ℕ) (cm_per_piece : ℝ) (remaining_meters : ℝ) 
  (pieces_cm_to_m : cm_per_piece / 100 = 0.15) (remaining_ribbon : remaining_meters = 36) 
  (pieces_cut : n = 100) : n * (cm_per_piece / 100) + remaining_meters = 51 := 
by 
  sorry

end original_length_of_ribbon_l70_70005


namespace probability_of_black_given_not_white_l70_70659

variable (total_balls white_balls black_balls red_balls : ℕ)
variable (ball_is_not_white : Prop)

theorem probability_of_black_given_not_white 
  (h1 : total_balls = 10)
  (h2 : white_balls = 5)
  (h3 : black_balls = 3)
  (h4 : red_balls = 2)
  (h5 : ball_is_not_white) :
  (3 : ℚ) / 5 = (black_balls : ℚ) / (total_balls - white_balls) :=
by
  simp only [h1, h2, h3, h4]
  sorry

end probability_of_black_given_not_white_l70_70659


namespace quad_root_magnitude_l70_70641

theorem quad_root_magnitude (m : ℝ) :
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → m = 2 ∨ m = -2 :=
by
  sorry

end quad_root_magnitude_l70_70641


namespace integer_roots_l70_70236

noncomputable def is_quadratic_root (p q x : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem integer_roots (p q x1 x2 : ℝ)
  (hq1 : is_quadratic_root p q x1)
  (hq2 : is_quadratic_root p q x2)
  (hd : x1 ≠ x2)
  (hx : |x1 - x2| = 1)
  (hpq : |p - q| = 1) :
  (∃ (p_int q_int x1_int x2_int : ℤ), 
      p = p_int ∧ q = q_int ∧ x1 = x1_int ∧ x2 = x2_int) :=
sorry

end integer_roots_l70_70236


namespace carnations_third_bouquet_l70_70051

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end carnations_third_bouquet_l70_70051


namespace small_bottle_sold_percentage_l70_70521

-- Definitions for initial conditions
def small_bottles_initial : ℕ := 6000
def large_bottles_initial : ℕ := 15000
def large_bottle_sold_percentage : ℝ := 0.14
def total_remaining_bottles : ℕ := 18180

-- The statement we need to prove
theorem small_bottle_sold_percentage :
  ∃ k : ℝ, (0 ≤ k ∧ k ≤ 100) ∧
  (small_bottles_initial - (k / 100) * small_bottles_initial + 
   large_bottles_initial - large_bottle_sold_percentage * large_bottles_initial = total_remaining_bottles) ∧
  (k = 12) :=
sorry

end small_bottle_sold_percentage_l70_70521


namespace range_of_t_l70_70737

noncomputable def condition (t : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 5 / 2 ∧ (t * x^2 + 2 * x - 2 > 0)

theorem range_of_t (t : ℝ) : ¬¬ condition t → t > - 1 / 2 :=
by
  intros h
  -- The actual proof should be here
  sorry

end range_of_t_l70_70737


namespace ratio_of_areas_of_concentric_circles_eq_9_over_4_l70_70545

theorem ratio_of_areas_of_concentric_circles_eq_9_over_4
  (C1 C2 : ℝ)
  (h1 : ∃ Q : ℝ, true) -- Existence of point Q
  (h2 : (30 / 360) * C1 = (45 / 360) * C2) -- Arcs formed by 30-degree and 45-degree angles are equal in length
  : (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 9 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_eq_9_over_4_l70_70545


namespace range_of_a_minus_b_l70_70586

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) : -3 < a - b ∧ a - b < 0 :=
by
  sorry

end range_of_a_minus_b_l70_70586


namespace find_symmetric_L_like_shape_l70_70480

-- Define the L-like shape and its mirror image
def L_like_shape : Type := sorry  -- Placeholder for the actual geometry definition
def mirrored_L_like_shape : Type := sorry  -- Placeholder for the actual mirrored shape

-- Condition: The vertical symmetry function
def symmetric_about_vertical_line (shape1 shape2 : Type) : Prop :=
   sorry  -- Define what it means for shape1 to be symmetric to shape2

-- Given conditions (A to E as L-like shape variations)
def option_A : Type := sorry  -- An inverted L-like shape
def option_B : Type := sorry  -- An upside-down T-like shape
def option_C : Type := mirrored_L_like_shape  -- A mirrored L-like shape
def option_D : Type := sorry  -- A rotated L-like shape by 180 degrees
def option_E : Type := L_like_shape  -- An unchanged L-like shape

-- The theorem statement
theorem find_symmetric_L_like_shape :
  symmetric_about_vertical_line L_like_shape option_C :=
  sorry

end find_symmetric_L_like_shape_l70_70480


namespace evaluate_expression_is_41_l70_70330

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end evaluate_expression_is_41_l70_70330


namespace cloth_coloring_problem_l70_70440

theorem cloth_coloring_problem (lengthOfCloth : ℕ) 
  (women_can_color_100m_in_1_day : 5 * 1 = 100) 
  (women_can_color_in_3_days : 6 * 3 = lengthOfCloth) : lengthOfCloth = 360 := 
sorry

end cloth_coloring_problem_l70_70440


namespace parallelogram_angle_H_l70_70383

theorem parallelogram_angle_H (F H : ℝ) (h1 : F = 125) (h2 : F + H = 180) : H = 55 :=
by
  have h3 : H = 180 - F := by linarith
  rw [h1] at h3
  rw [h3]
  norm_num

end parallelogram_angle_H_l70_70383


namespace combination_lock_code_l70_70889

theorem combination_lock_code :
  ∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (x + y + x * y = 10 * x + y) →
  10 * x + y = 19 ∨ 10 * x + y = 29 ∨ 10 * x + y = 39 ∨ 10 * x + y = 49 ∨
  10 * x + y = 59 ∨ 10 * x + y = 69 ∨ 10 * x + y = 79 ∨ 10 * x + y = 89 ∨
  10 * x + y = 99 :=
by
  sorry

end combination_lock_code_l70_70889


namespace shaded_area_difference_l70_70695

theorem shaded_area_difference (A1 A3 A4 : ℚ) (h1 : 4 = 2 * 2) (h2 : A1 + 5 * A1 + 7 * A1 = 6) (h3 : p + q = 49) : 
  ∃ p q : ℕ, p + q = 49 ∧ p = 36 ∧ q = 13 :=
by {
  sorry
}

end shaded_area_difference_l70_70695


namespace final_price_lower_than_budget_l70_70318

theorem final_price_lower_than_budget :
  let budget := 1500
  let T := 750 -- budget equally split for TV
  let S := 750 -- budget equally split for Sound System
  let TV_price_with_discount := (T - 150) * 0.80
  let SoundSystem_price_with_discount := S * 0.85
  let combined_price_before_tax := TV_price_with_discount + SoundSystem_price_with_discount
  let final_price_with_tax := combined_price_before_tax * 1.08
  budget - final_price_with_tax = 293.10 :=
by
  sorry

end final_price_lower_than_budget_l70_70318


namespace subset_condition_l70_70668

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_condition_l70_70668


namespace max_4x3_y3_l70_70213

theorem max_4x3_y3 (x y : ℝ) (h1 : x ≤ 2) (h2 : y ≤ 3) (h3 : x + y = 3) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : 
  4 * x^3 + y^3 ≤ 33 :=
sorry

end max_4x3_y3_l70_70213


namespace tangent_circle_given_r_l70_70963

theorem tangent_circle_given_r (r : ℝ) (h_pos : 0 < r)
    (h_tangent : ∀ x y : ℝ, (2 * x + y = r) → (x^2 + y^2 = 2 * r))
  : r = 10 :=
sorry

end tangent_circle_given_r_l70_70963


namespace non_congruent_triangles_proof_l70_70499

noncomputable def non_congruent_triangles_count : ℕ :=
  let points := [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
  9

theorem non_congruent_triangles_proof :
  non_congruent_triangles_count = 9 :=
sorry

end non_congruent_triangles_proof_l70_70499


namespace chocolate_distribution_l70_70913

theorem chocolate_distribution
  (total_chocolate : ℚ)
  (num_piles : ℕ)
  (piles_given_to_shaina : ℕ)
  (weight_each_pile : ℚ)
  (weight_of_shaina_piles : ℚ)
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_given_to_shaina = 2)
  (h4 : weight_each_pile = total_chocolate / num_piles)
  (h5 : weight_of_shaina_piles = piles_given_to_shaina * weight_each_pile) :
  weight_of_shaina_piles = 24 / 7 := by
  sorry

end chocolate_distribution_l70_70913


namespace find_urn_yellow_balls_l70_70097

theorem find_urn_yellow_balls :
  ∃ (M : ℝ), 
    (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
    M = 111 := 
sorry

end find_urn_yellow_balls_l70_70097


namespace fraction_comparison_l70_70401

theorem fraction_comparison : 
  (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) :=
sorry

end fraction_comparison_l70_70401


namespace polynomial_real_root_l70_70620

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a * x^3 - x^2 + a^2 * x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end polynomial_real_root_l70_70620


namespace log3_infinite_nested_l70_70594

theorem log3_infinite_nested (x : ℝ) (h : x = Real.logb 3 (64 + x)) : x = 4 :=
by
  sorry

end log3_infinite_nested_l70_70594


namespace technician_round_trip_l70_70824

theorem technician_round_trip (D : ℝ) (hD : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  (traveled_distance / round_trip * 100) = 65 := by
  -- Definitions based on the given conditions
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  
  -- Placeholder for the proof to satisfy Lean syntax.
  sorry

end technician_round_trip_l70_70824


namespace integral_solutions_l70_70606

/-- 
  Prove that the integral solutions to the equation 
  (m^2 - n^2)^2 = 1 + 16n are exactly (m, n) = (±1, 0), (±4, 3), (±4, 5). 
--/
theorem integral_solutions (m n : ℤ) :
  (m^2 - n^2)^2 = 1 + 16 * n ↔ (m = 1 ∧ n = 0) ∨ (m = -1 ∧ n = 0) ∨
                        (m = 4 ∧ n = 3) ∨ (m = -4 ∧ n = 3) ∨
                        (m = 4 ∧ n = 5) ∨ (m = -4 ∧ n = 5) :=
by
  sorry

end integral_solutions_l70_70606


namespace bc_sum_l70_70785

theorem bc_sum (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 10) : B + C = 310 := by
  sorry

end bc_sum_l70_70785


namespace problem_statement_l70_70108

noncomputable def k_value (k : ℝ) : Prop :=
  (∀ (x y : ℝ), x + y = k → x^2 + y^2 = 4) ∧ (∀ (A B : ℝ × ℝ), (∃ (x y : ℝ), A = (x, y) ∧ x^2 + y^2 = 4) ∧ (∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4) ∧ 
  (∃ (xa ya xb yb : ℝ), A = (xa, ya) ∧ B = (xb, yb) ∧ |(xa - xb, ya - yb)| = |(xa, ya)| + |(xb, yb)|)) → k = 2

theorem problem_statement (k : ℝ) (h : k > 0) : k_value k :=
  sorry

end problem_statement_l70_70108


namespace fifteenth_triangular_number_is_120_l70_70018

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem fifteenth_triangular_number_is_120 : triangular_number 15 = 120 := by
  sorry

end fifteenth_triangular_number_is_120_l70_70018


namespace age_of_new_person_l70_70627

theorem age_of_new_person (T A : ℕ) (h1 : (T / 10 : ℤ) - 3 = (T - 40 + A) / 10) : A = 10 := 
sorry

end age_of_new_person_l70_70627


namespace tiffany_optimal_area_l70_70163

def optimal_area (A : ℕ) : Prop :=
  ∃ l w : ℕ, l + w = 160 ∧ l ≥ 85 ∧ w ≥ 45 ∧ A = l * w

theorem tiffany_optimal_area : optimal_area 6375 :=
  sorry

end tiffany_optimal_area_l70_70163


namespace meeting_point_l70_70049

theorem meeting_point (n : ℕ) (petya_start vasya_start petya_end vasya_end meeting_lamp : ℕ) : 
  n = 100 → petya_start = 1 → vasya_start = 100 → petya_end = 22 → vasya_end = 88 → meeting_lamp = 64 :=
by
  intros h_n h_p_start h_v_start h_p_end h_v_end
  sorry

end meeting_point_l70_70049


namespace common_ratio_is_two_l70_70992

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end common_ratio_is_two_l70_70992


namespace area_of_square_l70_70642

theorem area_of_square (r s L B: ℕ) (h1 : r = s) (h2 : L = 5 * r) (h3 : B = 11) (h4 : 220 = L * B) : s^2 = 16 := by
  sorry

end area_of_square_l70_70642


namespace intersection_point_of_lines_l70_70611

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), (2 * y = 3 * x - 6) ∧ (x + 5 * y = 10) ∧ (x = 50 / 17) ∧ (y = 24 / 17) :=
by
  sorry

end intersection_point_of_lines_l70_70611


namespace infinite_perfect_squares_in_ap_l70_70283

open Nat

def is_arithmetic_progression (a d : ℕ) (an : ℕ → ℕ) : Prop :=
  ∀ n, an n = a + n * d

def is_perfect_square (x : ℕ) : Prop :=
  ∃ m, m * m = x

theorem infinite_perfect_squares_in_ap (a d : ℕ) (an : ℕ → ℕ) (m : ℕ)
  (h_arith_prog : is_arithmetic_progression a d an)
  (h_initial_square : a = m * m) :
  ∃ (f : ℕ → ℕ), ∀ n, is_perfect_square (an (f n)) :=
sorry

end infinite_perfect_squares_in_ap_l70_70283


namespace statement_A_statement_C_statement_D_l70_70771

theorem statement_A (x : ℝ) :
  (¬ (∀ x ≥ 3, 2 * x - 10 ≥ 0)) ↔ (∃ x0 ≥ 3, 2 * x0 - 10 < 0) := 
sorry

theorem statement_C {a b c : ℝ} (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

theorem statement_D {a b m : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (a / b) > ((a + m) / (b + m)) := 
sorry

end statement_A_statement_C_statement_D_l70_70771


namespace pencil_length_l70_70267

theorem pencil_length (L : ℝ) (h1 : (1 / 8) * L + (1 / 2) * (7 / 8) * L + (7 / 2) = L) : L = 16 :=
by
  sorry

end pencil_length_l70_70267


namespace wifes_raise_l70_70740

variable (D W : ℝ)
variable (h1 : 0.08 * D = 800)
variable (h2 : 1.08 * D - 1.08 * W = 540)

theorem wifes_raise : 0.08 * W = 760 :=
by
  sorry

end wifes_raise_l70_70740


namespace find_triangle_sides_l70_70311

noncomputable def side_lengths (k c d : ℕ) : Prop :=
  let p1 := 26
  let p2 := 32
  let p3 := 30
  (2 * k = 6) ∧ (2 * k + 6 * c = p3) ∧ (2 * c + 2 * d = p1)

theorem find_triangle_sides (k c d : ℕ) (h1 : side_lengths k c d) : k = 3 ∧ c = 4 ∧ d = 5 := 
  sorry

end find_triangle_sides_l70_70311


namespace x_squared_eq_r_floor_x_has_2_or_3_solutions_l70_70300

theorem x_squared_eq_r_floor_x_has_2_or_3_solutions (r : ℝ) (hr : r > 2) : 
  ∃! (s : Finset ℝ), s.card = 2 ∨ s.card = 3 ∧ ∀ x ∈ s, x^2 = r * (⌊x⌋) :=
by
  sorry

end x_squared_eq_r_floor_x_has_2_or_3_solutions_l70_70300


namespace time_to_cover_length_l70_70418

/-- Constants -/
def speed_escalator : ℝ := 10
def length_escalator : ℝ := 112
def speed_person : ℝ := 4

/-- Proof problem -/
theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person)) = 8 := by
  sorry

end time_to_cover_length_l70_70418


namespace louise_winning_strategy_2023x2023_l70_70446

theorem louise_winning_strategy_2023x2023 :
  ∀ (n : ℕ), (n % 2 = 1) → (n = 2023) →
  ∃ (strategy : ℕ × ℕ → Prop),
    (∀ turn : ℕ, ∃ (i j : ℕ), i < n ∧ j < n ∧ strategy (i, j)) ∧
    (∃ i j : ℕ, strategy (i, j) ∧ (i = 0 ∧ j = 0)) :=
by
  sorry

end louise_winning_strategy_2023x2023_l70_70446


namespace probability_of_exactly_three_blue_marbles_l70_70284

-- Define the conditions
def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def total_selections : ℕ := 6
def blue_selections : ℕ := 3
def blue_probability : ℚ := 8 / 15
def red_probability : ℚ := 7 / 15
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability formula calculation
def binomial_probability : ℚ :=
  binomial_coefficient total_selections blue_selections * (blue_probability ^ blue_selections) * (red_probability ^ (total_selections - blue_selections))

-- The hypothesis (conditions) and conclusion (the solution)
theorem probability_of_exactly_three_blue_marbles :
  binomial_probability = (3512320 / 11390625) :=
by sorry

end probability_of_exactly_three_blue_marbles_l70_70284


namespace smallest_n_for_Tn_gt_2006_over_2016_l70_70727

-- Definitions from the given problem
def Sn (n : ℕ) : ℚ := n^2 / (n + 1)
def an (n : ℕ) : ℚ := if n = 1 then 1 / 2 else Sn n - Sn (n - 1)
def bn (n : ℕ) : ℚ := an n / (n^2 + n - 1)

-- Definition of Tn sum
def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ k => bn (k + 1))

-- The main statement
theorem smallest_n_for_Tn_gt_2006_over_2016 : ∃ n : ℕ, Tn n > 2006 / 2016 := by
  sorry

end smallest_n_for_Tn_gt_2006_over_2016_l70_70727


namespace optionA_optionB_optionC_optionD_l70_70867

-- Statement for option A
theorem optionA : (∀ x : ℝ, x ≠ 3 → x^2 - 4 * x + 3 ≠ 0) ↔ (x^2 - 4 * x + 3 = 0 → x = 3) := sorry

-- Statement for option B
theorem optionB : (¬ (∀ x : ℝ, x^2 - x + 2 > 0) ↔ ∃ x0 : ℝ, x0^2 - x0 + 2 ≤ 0) := sorry

-- Statement for option C
theorem optionC (p q : Prop) : p ∧ q → p ∧ q := sorry

-- Statement for option D
theorem optionD (x : ℝ) : (x > -1 → x^2 + 4 * x + 3 > 0) ∧ ¬ (∀ x : ℝ, x^2 + 4 * x + 3 > 0 → x > -1) := sorry

end optionA_optionB_optionC_optionD_l70_70867


namespace calories_in_300g_l70_70183

/-
Define the conditions of the problem.
-/

def lemon_juice_grams := 150
def sugar_grams := 200
def lime_juice_grams := 50
def water_grams := 500

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 390
def lime_juice_calories_per_100g := 20
def water_calories := 0

/-
Define the total weight of the beverage.
-/
def total_weight := lemon_juice_grams + sugar_grams + lime_juice_grams + water_grams

/-
Define the total calories of the beverage.
-/
def total_calories := 
  (lemon_juice_calories_per_100g * lemon_juice_grams / 100) + 
  (sugar_calories_per_100g * sugar_grams / 100) + 
  (lime_juice_calories_per_100g * lime_juice_grams / 100) + 
  water_calories

/-
Prove the number of calories in 300 grams of the beverage.
-/
theorem calories_in_300g : (total_calories / total_weight) * 300 = 278 := by
  sorry

end calories_in_300g_l70_70183


namespace probability_of_woman_lawyer_is_54_percent_l70_70181

variable (total_members : ℕ) (women_percentage lawyers_percentage : ℕ)
variable (H_total_members_pos : total_members > 0) 
variable (H_women_percentage : women_percentage = 90)
variable (H_lawyers_percentage : lawyers_percentage = 60)

def probability_woman_lawyer : ℕ :=
  (women_percentage * lawyers_percentage * total_members) / (100 * 100)

theorem probability_of_woman_lawyer_is_54_percent (H_total_members_pos : total_members > 0)
  (H_women_percentage : women_percentage = 90)
  (H_lawyers_percentage : lawyers_percentage = 60) :
  probability_woman_lawyer total_members women_percentage lawyers_percentage = 54 :=
by
  sorry

end probability_of_woman_lawyer_is_54_percent_l70_70181


namespace right_triangle_other_acute_angle_l70_70282

theorem right_triangle_other_acute_angle (A B C : ℝ) (r : A + B + C = 180) (h : A = 90) (a : B = 30) :
  C = 60 :=
sorry

end right_triangle_other_acute_angle_l70_70282


namespace planes_parallel_l70_70817

-- Given definitions and conditions
variables {Line Plane : Type}
variables (a b : Line) (α β γ : Plane)

-- Conditions from the problem
axiom perp_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_plane_plane (plane1 plane2 : Plane) : Prop

-- Conditions
variable (h1 : parallel_plane_plane γ α)
variable (h2 : parallel_plane_plane γ β)

-- Proof statement
theorem planes_parallel (h1 : parallel_plane_plane γ α) (h2 : parallel_plane_plane γ β) : parallel_plane_plane α β := sorry

end planes_parallel_l70_70817


namespace coplanar_lines_l70_70196

def vector3 := ℝ × ℝ × ℝ

def vec1 : vector3 := (2, -1, 3)
def vec2 (k : ℝ) : vector3 := (3 * k, 1, 2)
def pointVec : vector3 := (-3, 2, -3)

def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem coplanar_lines (k : ℝ) : det3x3 2 (-1) 3 (3 * k) 1 2 (-3) 2 (-3) = 0 → k = -29 / 9 :=
  sorry

end coplanar_lines_l70_70196


namespace sin_A_value_of_triangle_l70_70904

theorem sin_A_value_of_triangle 
  (a b : ℝ) (A B C : ℝ) (h_triangle : a = 2) (h_b : b = 3) (h_tanB : Real.tan B = 3) :
  Real.sin A = Real.sqrt 10 / 5 :=
sorry

end sin_A_value_of_triangle_l70_70904


namespace calculate_value_of_A_plus_C_l70_70217

theorem calculate_value_of_A_plus_C (A B C : ℕ) (hA : A = 238) (hAB : A = B + 143) (hBC : C = B + 304) : A + C = 637 :=
by
  sorry

end calculate_value_of_A_plus_C_l70_70217


namespace roots_polynomial_sum_l70_70540

theorem roots_polynomial_sum (p q r s : ℂ)
  (h_roots : (p, q, r, s) ∈ { (p, q, r, s) | (Polynomial.eval p (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval q (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval r (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval s (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) })
  (h_sum_two_at_a_time : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (h_product : p*q*r*s = 6) :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 := by
  sorry

end roots_polynomial_sum_l70_70540


namespace marathon_y_distance_l70_70065

theorem marathon_y_distance (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (total_yards : ℕ) (y : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : yards_per_marathon = 312) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 8) 
  (H5 : total_yards = num_marathons * yards_per_marathon) 
  (H6 : total_yards % yards_per_mile = y) 
  (H7 : 0 ≤ y) 
  (H8 : y < yards_per_mile) : 
  y = 736 :=
by 
  sorry

end marathon_y_distance_l70_70065


namespace geometric_sequence_sum_l70_70325

theorem geometric_sequence_sum {a : ℕ → ℝ} (h : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) 
  (h_cond : (1 / (a 2 * a 4)) + (2 / (a 4 * a 4)) + (1 / (a 4 * a 6)) = 81) :
  (1 / a 3) + (1 / a 5) = 9 :=
sorry

end geometric_sequence_sum_l70_70325


namespace bus_patrons_correct_l70_70094

-- Definitions corresponding to conditions
def number_of_golf_carts : ℕ := 13
def patrons_per_cart : ℕ := 3
def car_patrons : ℕ := 12

-- Multiply to get total patrons transported by golf carts
def total_patrons := number_of_golf_carts * patrons_per_cart

-- Calculate bus patrons
def bus_patrons := total_patrons - car_patrons

-- The statement to prove
theorem bus_patrons_correct : bus_patrons = 27 :=
by
  sorry

end bus_patrons_correct_l70_70094


namespace xiaochun_age_l70_70297

theorem xiaochun_age
  (x y : ℕ)
  (h1 : x = y - 18)
  (h2 : 2 * (x + 3) = y + 3) :
  x = 15 :=
sorry

end xiaochun_age_l70_70297


namespace triangle_median_inequality_l70_70720

variable (a b c m_a m_b m_c D : ℝ)

-- Assuming the conditions are required to make the proof valid
axiom median_formula_m_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2
axiom median_formula_m_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2
axiom median_formula_m_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2

theorem triangle_median_inequality : 
  a^2 + b^2 <= m_c * 6 * D ∧ b^2 + c^2 <= m_a * 6 * D ∧ c^2 + a^2 <= m_b * 6 * D → 
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b <= 6 * D := 
by
  sorry

end triangle_median_inequality_l70_70720


namespace problem_l70_70541

theorem problem (a b c : ℤ) (h1 : 0 < c) (h2 : c < 90) (h3 : Real.sqrt (9 - 8 * Real.sin (50 * Real.pi / 180)) = a + b * Real.sin (c * Real.pi / 180)) : 
  (a + b) / c = 1 / 2 :=
by
  sorry

end problem_l70_70541


namespace largest_sphere_radius_in_prism_l70_70145

noncomputable def largestInscribedSphereRadius (m : ℝ) : ℝ :=
  (Real.sqrt 6 - Real.sqrt 2) / 4 * m

theorem largest_sphere_radius_in_prism (m : ℝ) (h : 0 < m) :
  ∃ r, r = largestInscribedSphereRadius m ∧ r < m/2 :=
sorry

end largest_sphere_radius_in_prism_l70_70145


namespace min_value_at_3_l70_70536

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l70_70536


namespace combination_property_problem_solution_l70_70707

open Nat

def combination (n k : ℕ) : ℕ :=
  if h : k ≤ n then (factorial n) / (factorial k * factorial (n - k)) else 0

theorem combination_property (n k : ℕ) (h₀ : 1 ≤ k) (h₁ : k ≤ n) :
  combination n k + combination n (k - 1) = combination (n + 1) k := sorry

theorem problem_solution :
  (combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + combination 7 2 + 
   combination 8 2 + combination 9 2 + combination 10 2 + combination 11 2 + combination 12 2 + 
   combination 13 2 + combination 14 2 + combination 15 2 + combination 16 2 + combination 17 2 + 
   combination 18 2 + combination 19 2) = 1139 := sorry

end combination_property_problem_solution_l70_70707


namespace range_of_x_l70_70111

variable (x y : ℝ)

theorem range_of_x (h1 : 2 * x - y = 4) (h2 : -2 < y ∧ y ≤ 3) :
  1 < x ∧ x ≤ 7 / 2 :=
  sorry

end range_of_x_l70_70111


namespace negation_proposition_l70_70246

theorem negation_proposition (l : ℝ) (h : l = 1) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) = (∀ x : ℝ, x + l < 0) := by 
  sorry

end negation_proposition_l70_70246


namespace vector_addition_correct_l70_70663

def a : ℝ × ℝ := (-1, 6)
def b : ℝ × ℝ := (3, -2)
def c : ℝ × ℝ := (2, 4)

theorem vector_addition_correct : a + b = c := by
  sorry

end vector_addition_correct_l70_70663


namespace find_a_l70_70735

theorem find_a (a b c : ℚ)
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) : a = 10 / 11 :=
by
  sorry

end find_a_l70_70735


namespace racing_cars_lcm_l70_70733

theorem racing_cars_lcm :
  let a := 28
  let b := 24
  let c := 32
  Nat.lcm a (Nat.lcm b c) = 672 :=
by
  sorry

end racing_cars_lcm_l70_70733


namespace num_pos_int_x_l70_70197

theorem num_pos_int_x (x : ℕ) : 
  (30 < x^2 + 5 * x + 10) ∧ (x^2 + 5 * x + 10 < 60) ↔ x = 3 ∨ x = 4 ∨ x = 5 := 
sorry

end num_pos_int_x_l70_70197


namespace range_of_m_l70_70399

open Set

-- Definitions and conditions
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0
def neg_p (x : ℝ) : Prop := ¬ p x
def neg_q (x m : ℝ) : Prop := ¬ q x m

-- Theorem statement
theorem range_of_m (x m : ℝ) (h₁ : ¬ p x → ¬ q x m) (h₂ : m > 0) : m ≥ 9 :=
  sorry

end range_of_m_l70_70399


namespace mathematician_daily_questions_l70_70079

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end mathematician_daily_questions_l70_70079


namespace estimate_undetected_typos_l70_70207

variables (a b c : ℕ)
-- a, b, c ≥ 0 are non-negative integers representing discovered errors by proofreader A, B, and common errors respectively.

theorem estimate_undetected_typos (h : c ≤ a ∧ c ≤ b) :
  ∃ n : ℕ, n = a * b / c - a - b + c :=
sorry

end estimate_undetected_typos_l70_70207


namespace winnie_the_pooh_wins_l70_70755

variable (cones : ℕ)

def can_guarantee_win (initial_cones : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ), 
    (strategy initial_cones = 4 ∨ strategy initial_cones = 1) ∧ 
    ∀ n, (strategy n = 1 → (n = 2012 - 4 ∨ n = 2007 - 1 ∨ n = 2005 - 1)) ∧
         (strategy n = 4 → n = 2012)

theorem winnie_the_pooh_wins : can_guarantee_win 2012 :=
sorry

end winnie_the_pooh_wins_l70_70755


namespace benny_apples_l70_70127

theorem benny_apples (benny dan : ℕ) (total : ℕ) (H1 : dan = 9) (H2 : total = 11) (H3 : benny + dan = total) : benny = 2 :=
by
  sorry

end benny_apples_l70_70127


namespace total_children_on_playground_l70_70214

theorem total_children_on_playground
  (boys : ℕ) (girls : ℕ)
  (h_boys : boys = 44) (h_girls : girls = 53) :
  boys + girls = 97 :=
by 
  -- Proof omitted
  sorry

end total_children_on_playground_l70_70214


namespace third_competitor_eats_l70_70690

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end third_competitor_eats_l70_70690


namespace equivalent_forms_l70_70750

-- Given line equation
def given_line_eq (x y : ℝ) : Prop :=
  (3 * x - 2) / 4 - (2 * y - 1) / 2 = 1

-- General form of the line
def general_form (x y : ℝ) : Prop :=
  3 * x - 8 * y - 2 = 0

-- Slope-intercept form of the line
def slope_intercept_form (x y : ℝ) : Prop := 
  y = (3 / 8) * x - 1 / 4

-- Intercept form of the line
def intercept_form (x y : ℝ) : Prop :=
  x / (2 / 3) + y / (-1 / 4) = 1

-- Normal form of the line
def normal_form (x y : ℝ) : Prop :=
  3 / Real.sqrt 73 * x - 8 / Real.sqrt 73 * y - 2 / Real.sqrt 73 = 0

-- Proof problem: Prove that the given line equation is equivalent to the derived forms
theorem equivalent_forms (x y : ℝ) :
  given_line_eq x y ↔ (general_form x y ∧ slope_intercept_form x y ∧ intercept_form x y ∧ normal_form x y) :=
sorry

end equivalent_forms_l70_70750


namespace necklace_cost_l70_70369

def bead_necklaces := 3
def gemstone_necklaces := 3
def total_necklaces := bead_necklaces + gemstone_necklaces
def total_earnings := 36

theorem necklace_cost :
  (total_earnings / total_necklaces) = 6 :=
by
  -- Proof goes here
  sorry

end necklace_cost_l70_70369


namespace find_x_plus_z_l70_70722

theorem find_x_plus_z :
  ∃ (x y z : ℝ), 
  (x + y + z = 0) ∧
  (2016 * x + 2017 * y + 2018 * z = 0) ∧
  (2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) ∧
  (x + z = 4036) :=
sorry

end find_x_plus_z_l70_70722


namespace complex_number_solution_l70_70249

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end complex_number_solution_l70_70249


namespace strictly_increasing_and_symmetric_l70_70458

open Real

noncomputable def f1 (x : ℝ) : ℝ := x^((1 : ℝ)/2)
noncomputable def f2 (x : ℝ) : ℝ := x^((1 : ℝ)/3)
noncomputable def f3 (x : ℝ) : ℝ := x^((2 : ℝ)/3)
noncomputable def f4 (x : ℝ) : ℝ := x^(-(1 : ℝ)/3)

theorem strictly_increasing_and_symmetric : 
  ∀ f : ℝ → ℝ,
  (f = f2) ↔ 
  ((∀ x : ℝ, 0 < x → f x = x^((1 : ℝ)/3) ∧ f (-x) = -(f x)) ∧ 
   (∀ x y : ℝ, 0 < x ∧ 0 < y → (x < y → f x < f y))) :=
sorry

end strictly_increasing_and_symmetric_l70_70458


namespace find_x_l70_70335

theorem find_x (x : ℕ) : (x > 20) ∧ (x < 120) ∧ (∃ y : ℕ, x = y^2) ∧ (x % 3 = 0) ↔ (x = 36) ∨ (x = 81) :=
by
  sorry

end find_x_l70_70335


namespace sum_of_roots_l70_70361

theorem sum_of_roots (p q : ℝ) (h_eq : 2 * p + 3 * q = 6) (h_roots : ∀ x : ℝ, x ^ 2 - p * x + q = 0) : p = 2 := by
sorry

end sum_of_roots_l70_70361


namespace find_total_stock_worth_l70_70833

noncomputable def total_stock_worth (X : ℝ) : Prop :=
  let profit := 0.10 * (0.20 * X)
  let loss := 0.05 * (0.80 * X)
  loss - profit = 450

theorem find_total_stock_worth (X : ℝ) (h : total_stock_worth X) : X = 22500 :=
by
  sorry

end find_total_stock_worth_l70_70833


namespace abs_eq_5_iff_l70_70787

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l70_70787


namespace smallest_n_for_Qn_l70_70008

theorem smallest_n_for_Qn (n : ℕ) : 
  (∃ n : ℕ, 1 / (n * (2 * n + 1)) < 1 / 2023 ∧ ∀ m < n, 1 / (m * (2 * m + 1)) ≥ 1 / 2023) ↔ n = 32 := by
sorry

end smallest_n_for_Qn_l70_70008


namespace fraction_equals_i_l70_70964

theorem fraction_equals_i (m n : ℝ) (i : ℂ) (h : i * i = -1) (h_cond : m * (1 + i) = (11 + n * i)) :
  (m + n * i) / (m - n * i) = i :=
sorry

end fraction_equals_i_l70_70964


namespace min_value_of_function_l70_70802

theorem min_value_of_function : 
  ∃ (c : ℝ), (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2) ≥ c) ∧
             (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2 = c) → c = 1) := 
sorry

end min_value_of_function_l70_70802


namespace number_of_solutions_l70_70976

theorem number_of_solutions :
  (∃(x y : ℤ), x^4 + y^2 = 6 * y - 8) ∧ ∃!(x y : ℤ), x^4 + y^2 = 6 * y - 8 := 
sorry

end number_of_solutions_l70_70976


namespace exchange_yen_for_yuan_l70_70843

-- Define the condition: 100 Japanese yen could be exchanged for 7.2 yuan
def exchange_rate : ℝ := 7.2
def yen_per_100_yuan : ℝ := 100

-- Define the amount in yuan we want to exchange
def yuan_amount : ℝ := 720

-- The mathematical assertion (proof problem)
theorem exchange_yen_for_yuan : 
  (yuan_amount / exchange_rate) * yen_per_100_yuan = 10000 :=
by
  sorry

end exchange_yen_for_yuan_l70_70843


namespace light_flash_fraction_l70_70522

def light_flash_fraction_of_hour (n : ℕ) (t : ℕ) (flashes : ℕ) := 
  (n * t) / (60 * 60)

theorem light_flash_fraction (n : ℕ) (t : ℕ) (flashes : ℕ) (h1 : t = 12) (h2 : flashes = 300) : 
  light_flash_fraction_of_hour n t flashes = 1 := 
by
  sorry

end light_flash_fraction_l70_70522


namespace distinct_integers_are_squares_l70_70321

theorem distinct_integers_are_squares
  (n : ℕ) 
  (h_n : n = 2000) 
  (x : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → x i ≠ x j)
  (h_product_square : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ∃ (m : ℕ), x i * x j * x k = m^2) :
  ∀ i : Fin n, ∃ (m : ℕ), x i = m^2 := 
sorry

end distinct_integers_are_squares_l70_70321


namespace num_solutions_abs_x_plus_abs_y_lt_100_l70_70738

theorem num_solutions_abs_x_plus_abs_y_lt_100 :
  (∃ n : ℕ, n = 338350 ∧ ∀ (x y : ℤ), (|x| + |y| < 100) → True) :=
sorry

end num_solutions_abs_x_plus_abs_y_lt_100_l70_70738


namespace center_of_hyperbola_l70_70129

-- Define the given equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  ((3 * y - 6)^2 / 8^2) - ((4 * x - 5)^2 / 3^2) = 1

-- Prove that the center of the hyperbola is (5 / 4, 2)
theorem center_of_hyperbola :
  (∃ h k : ℝ, h = 5 / 4 ∧ k = 2 ∧ ∀ x y : ℝ, hyperbola_eq x y ↔ ((y - k)^2 / (8 / 3)^2 - (x - h)^2 / (3 / 4)^2 = 1)) :=
sorry

end center_of_hyperbola_l70_70129


namespace carrots_picked_by_Carol_l70_70185

theorem carrots_picked_by_Carol (total_carrots mom_carrots : ℕ) (h1 : total_carrots = 38 + 7) (h2 : mom_carrots = 16) :
  total_carrots - mom_carrots = 29 :=
by {
  sorry
}

end carrots_picked_by_Carol_l70_70185


namespace Nell_initial_cards_l70_70978

theorem Nell_initial_cards 
  (cards_given : ℕ)
  (cards_left : ℕ)
  (cards_given_eq : cards_given = 301)
  (cards_left_eq : cards_left = 154) :
  cards_given + cards_left = 455 := by
sorry

end Nell_initial_cards_l70_70978


namespace bobbit_worm_days_l70_70396

variable (initial_fish : ℕ)
variable (fish_added : ℕ)
variable (fish_eaten_per_day : ℕ)
variable (week_days : ℕ)
variable (final_fish : ℕ)
variable (d : ℕ)

theorem bobbit_worm_days (h1 : initial_fish = 60)
                         (h2 : fish_added = 8)
                         (h3 : fish_eaten_per_day = 2)
                         (h4 : week_days = 7)
                         (h5 : final_fish = 26) :
  60 - 2 * d + 8 - 2 * week_days = 26 → d = 14 :=
by {
  sorry
}

end bobbit_worm_days_l70_70396


namespace sin_3x_sin_x_solutions_l70_70441

open Real

theorem sin_3x_sin_x_solutions :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (3 * x) = sin x) ∧ s.card = 7 := 
by sorry

end sin_3x_sin_x_solutions_l70_70441


namespace total_cats_handled_last_year_l70_70058

theorem total_cats_handled_last_year (num_adult_cats : ℕ) (two_thirds_female : ℕ) (seventy_five_percent_litters : ℕ) 
                                     (kittens_per_litter : ℕ) (adopted_returned : ℕ) :
  num_adult_cats = 120 →
  two_thirds_female = (2 * num_adult_cats) / 3 →
  seventy_five_percent_litters = (3 * two_thirds_female) / 4 →
  kittens_per_litter = 3 →
  adopted_returned = 15 →
  num_adult_cats + seventy_five_percent_litters * kittens_per_litter + adopted_returned = 315 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cats_handled_last_year_l70_70058


namespace necessarily_negative_l70_70407

theorem necessarily_negative
  (a b c : ℝ)
  (ha : -2 < a ∧ a < -1)
  (hb : 0 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 0) :
  b + c < 0 :=
sorry

end necessarily_negative_l70_70407


namespace school_children_count_l70_70851

-- Define the conditions
variable (A P C B G : ℕ)
variable (A_eq : A = 160)
variable (kids_absent : ∀ (present kids absent children : ℕ), present = kids - absent → absent = 160)
variable (bananas_received : ∀ (two_per child kids : ℕ), (2 * kids) + (2 * 160) = 2 * 6400 + (4 * (6400 / 160)))
variable (boys_girls : B = 3 * G)

-- State the theorem
theorem school_children_count (C : ℕ) (A P B G : ℕ) 
  (A_eq : A = 160)
  (kids_absent : P = C - A)
  (bananas_received : (2 * P) + (2 * A) = 2 * P + (4 * (P / A)))
  (boys_girls : B = 3 * G)
  (total_bananas : 2 * P + 4 * (P / A) = 12960) :
  C = 6560 := 
sorry

end school_children_count_l70_70851


namespace total_poles_needed_l70_70799

theorem total_poles_needed (longer_side_poles : ℕ) (shorter_side_poles : ℕ) (internal_fence_poles : ℕ) :
  longer_side_poles = 35 → 
  shorter_side_poles = 27 → 
  internal_fence_poles = (shorter_side_poles - 1) → 
  ((longer_side_poles * 2) + (shorter_side_poles * 2) - 4 + internal_fence_poles) = 146 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_poles_needed_l70_70799


namespace min_value_of_expression_l70_70555

noncomputable def f (x : ℝ) : ℝ :=
  2 / x + 9 / (1 - 2 * x)

theorem min_value_of_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 2) : ∃ m, f x = m ∧ m = 25 :=
by
  sorry

end min_value_of_expression_l70_70555


namespace math_problem_l70_70538

theorem math_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^b + 3 = b^a) (h4 : 3 * a^b = b^a + 13) : 
  (a = 2) ∧ (b = 3) :=
sorry

end math_problem_l70_70538


namespace bill_miles_sunday_l70_70502

variables (B : ℕ)
def miles_ran_Bill_Saturday := B
def miles_ran_Bill_Sunday := B + 4
def miles_ran_Julia_Sunday := 2 * (B + 4)
def total_miles_ran := miles_ran_Bill_Saturday + miles_ran_Bill_Sunday + miles_ran_Julia_Sunday

theorem bill_miles_sunday (h1 : total_miles_ran B = 32) : 
  miles_ran_Bill_Sunday B = 9 := 
by sorry

end bill_miles_sunday_l70_70502


namespace johns_father_fraction_l70_70179

theorem johns_father_fraction (total_money : ℝ) (given_to_mother_fraction remaining_after_father : ℝ) :
  total_money = 200 →
  given_to_mother_fraction = 3 / 8 →
  remaining_after_father = 65 →
  ((total_money - given_to_mother_fraction * total_money) - remaining_after_father) / total_money
  = 3 / 10 :=
by
  intros h1 h2 h3
  sorry

end johns_father_fraction_l70_70179


namespace amount_paid_l70_70364

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l70_70364


namespace sawyer_total_octopus_legs_l70_70849

-- Formalization of the problem conditions
def num_octopuses : Nat := 5
def legs_per_octopus : Nat := 8

-- Formalization of the question and answer
def total_legs : Nat := num_octopuses * legs_per_octopus

-- The proof statement
theorem sawyer_total_octopus_legs : total_legs = 40 :=
by
  sorry

end sawyer_total_octopus_legs_l70_70849


namespace chooseOneFromEachCategory_chooseTwoDifferentTypes_l70_70919

-- Define the number of different paintings in each category
def traditionalChinesePaintings : ℕ := 5
def oilPaintings : ℕ := 2
def watercolorPaintings : ℕ := 7

-- Part (1): Prove that the number of ways to choose one painting from each category is 70
theorem chooseOneFromEachCategory : traditionalChinesePaintings * oilPaintings * watercolorPaintings = 70 := by
  sorry

-- Part (2): Prove that the number of ways to choose two paintings of different types is 59
theorem chooseTwoDifferentTypes :
  (traditionalChinesePaintings * oilPaintings) + 
  (traditionalChinesePaintings * watercolorPaintings) + 
  (oilPaintings * watercolorPaintings) = 59 := by
  sorry

end chooseOneFromEachCategory_chooseTwoDifferentTypes_l70_70919


namespace xy_product_given_conditions_l70_70052

variable (x y : ℝ)

theorem xy_product_given_conditions (hx : x - y = 5) (hx3 : x^3 - y^3 = 35) : x * y = -6 :=
by
  sorry

end xy_product_given_conditions_l70_70052


namespace rectangular_prism_volume_increase_l70_70909

theorem rectangular_prism_volume_increase (L B H : ℝ) :
  let V_original := L * B * H
  let L_new := L * 1.07
  let B_new := B * 1.18
  let H_new := H * 1.25
  let V_new := L_new * B_new * H_new
  let increase_in_volume := (V_new - V_original) / V_original * 100
  increase_in_volume = 56.415 :=
by
  sorry

end rectangular_prism_volume_increase_l70_70909


namespace min_value_a_2b_l70_70345

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 :=
sorry

end min_value_a_2b_l70_70345


namespace trains_meet_at_noon_l70_70290

noncomputable def meeting_time_of_trains : Prop :=
  let distance_between_stations := 200
  let speed_of_train_A := 20
  let starting_time_A := 7
  let speed_of_train_B := 25
  let starting_time_B := 8
  let initial_distance_covered_by_A := speed_of_train_A * (starting_time_B - starting_time_A)
  let remaining_distance := distance_between_stations - initial_distance_covered_by_A
  let relative_speed := speed_of_train_A + speed_of_train_B
  let time_to_meet_after_B_starts := remaining_distance / relative_speed
  let meeting_time := starting_time_B + time_to_meet_after_B_starts
  meeting_time = 12

theorem trains_meet_at_noon : meeting_time_of_trains :=
by
  sorry

end trains_meet_at_noon_l70_70290


namespace total_study_hours_during_semester_l70_70884

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

end total_study_hours_during_semester_l70_70884


namespace contrapositive_l70_70872

variable (P Q : Prop)

theorem contrapositive (h : P → Q) : ¬Q → ¬P :=
sorry

end contrapositive_l70_70872


namespace no_spiky_two_digit_numbers_l70_70056

def is_spiky (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧
             10 ≤ n ∧ n < 100 ∧
             n = 10 * a + b ∧
             n = a + b^3 - 2 * a

theorem no_spiky_two_digit_numbers : ∀ n, 10 ≤ n ∧ n < 100 → ¬ is_spiky n :=
by
  intro n h
  sorry

end no_spiky_two_digit_numbers_l70_70056


namespace percent_singles_l70_70635

theorem percent_singles :
  ∀ (total_hits home_runs triples doubles : ℕ),
  total_hits = 50 →
  home_runs = 2 →
  triples = 4 →
  doubles = 10 →
  (total_hits - (home_runs + triples + doubles)) * 100 / total_hits = 68 :=
by
  sorry

end percent_singles_l70_70635


namespace prob_all_fail_prob_at_least_one_pass_l70_70266

def prob_pass := 1 / 2
def prob_fail := 1 - prob_pass

def indep (A B C : Prop) : Prop := true -- Usually we prove independence in a detailed manner, but let's assume it's given as true.

theorem prob_all_fail (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : (prob_fail * prob_fail * prob_fail) = 1 / 8 :=
by
  sorry

theorem prob_at_least_one_pass (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : 1 - (prob_fail * prob_fail * prob_fail) = 7 / 8 :=
by
  sorry

end prob_all_fail_prob_at_least_one_pass_l70_70266


namespace triangle_area_l70_70230

theorem triangle_area : 
  ∀ (x y: ℝ), (x / 5 + y / 2 = 1) → (x = 5) ∨ (y = 2) → ∃ A : ℝ, A = 5 :=
by
  intros x y h1 h2
  -- Definitions based on the problem conditions
  have hx : x = 5 := sorry
  have hy : y = 2 := sorry
  have base := 5
  have height := 2
  have area := 1 / 2 * base * height
  use area
  sorry

end triangle_area_l70_70230


namespace find_f2_l70_70900

-- Definitions based on the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_g_def : ∀ x, g x = f x + 9)
variable (h_g_val : g (-2) = 3)

-- Prove the required goal
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l70_70900


namespace divisor_of_difference_is_62_l70_70457

-- The problem conditions as definitions
def x : Int := 859622
def y : Int := 859560
def difference : Int := x - y

-- The proof statement
theorem divisor_of_difference_is_62 (d : Int) (h₁ : d ∣ y) (h₂ : d ∣ difference) : d = 62 := by
  sorry

end divisor_of_difference_is_62_l70_70457


namespace abs_neg_eight_plus_three_pow_zero_eq_nine_l70_70048

theorem abs_neg_eight_plus_three_pow_zero_eq_nine :
  |-8| + 3^0 = 9 :=
by
  sorry

end abs_neg_eight_plus_three_pow_zero_eq_nine_l70_70048


namespace width_of_house_l70_70464

theorem width_of_house (L P_L P_W A_total : ℝ) (hL : L = 20.5) (hPL : P_L = 6) (hPW : P_W = 4.5) (hAtotal : A_total = 232) :
  ∃ W : ℝ, W = 10 :=
by
  have area_porch : ℝ := P_L * P_W
  have area_house := A_total - area_porch
  use area_house / L
  sorry

end width_of_house_l70_70464


namespace roots_cubic_sum_cubes_l70_70398

theorem roots_cubic_sum_cubes (a b c : ℝ) 
    (h1 : 6 * a^3 - 803 * a + 1606 = 0)
    (h2 : 6 * b^3 - 803 * b + 1606 = 0)
    (h3 : 6 * c^3 - 803 * c + 1606 = 0) :
    (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := 
by
  sorry

end roots_cubic_sum_cubes_l70_70398


namespace problem_solution_l70_70879

theorem problem_solution (x : ℝ) : (∃ (x : ℝ), 5 < x ∧ x ≤ 6) ↔ (∃ (x : ℝ), (x - 3) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l70_70879


namespace toaster_popularity_l70_70729

theorem toaster_popularity
  (c₁ c₂ : ℤ) (p₁ p₂ k : ℤ)
  (h₀ : p₁ * c₁ = k)
  (h₁ : p₁ = 12)
  (h₂ : c₁ = 500)
  (h₃ : c₂ = 750)
  (h₄ : k = p₁ * c₁) :
  p₂ * c₂ = k → p₂ = 8 :=
by
  sorry

end toaster_popularity_l70_70729


namespace total_unique_plants_l70_70067

noncomputable def bed_A : ℕ := 600
noncomputable def bed_B : ℕ := 550
noncomputable def bed_C : ℕ := 400
noncomputable def bed_D : ℕ := 300

noncomputable def intersection_A_B : ℕ := 75
noncomputable def intersection_A_C : ℕ := 125
noncomputable def intersection_B_D : ℕ := 50
noncomputable def intersection_A_B_C : ℕ := 25

theorem total_unique_plants : 
  bed_A + bed_B + bed_C + bed_D - intersection_A_B - intersection_A_C - intersection_B_D + intersection_A_B_C = 1625 := 
by
  sorry

end total_unique_plants_l70_70067


namespace volume_of_intersecting_octahedra_l70_70103

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l70_70103


namespace floor_x_floor_x_eq_44_iff_l70_70899

theorem floor_x_floor_x_eq_44_iff (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 44) ↔ (7.333 ≤ x ∧ x < 7.5) :=
by
  sorry

end floor_x_floor_x_eq_44_iff_l70_70899


namespace bus_stop_time_l70_70971

theorem bus_stop_time (speed_excl_stops speed_incl_stops : ℝ) (h1 : speed_excl_stops = 50) (h2 : speed_incl_stops = 45) : (60 * ((speed_excl_stops - speed_incl_stops) / speed_excl_stops)) = 6 := 
by
  sorry

end bus_stop_time_l70_70971


namespace largest_number_is_27_l70_70554

-- Define the condition as a predicate
def three_consecutive_multiples_sum_to (k : ℕ) (sum : ℕ) : Prop :=
  ∃ n : ℕ, (3 * n) + (3 * n + 3) + (3 * n + 6) = sum

-- Define the proof statement
theorem largest_number_is_27 : three_consecutive_multiples_sum_to 3 72 → 3 * 7 + 6 = 27 :=
by
  intro h
  cases' h with n h_eq
  sorry

end largest_number_is_27_l70_70554


namespace camel_height_in_feet_correct_l70_70509

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end camel_height_in_feet_correct_l70_70509


namespace train_length_l70_70581

-- Definitions and conditions based on the problem
def time : ℝ := 28.997680185585153
def bridge_length : ℝ := 150
def train_speed : ℝ := 10

-- The theorem to prove
theorem train_length : (train_speed * time) - bridge_length = 139.97680185585153 :=
by
  sorry

end train_length_l70_70581


namespace maximum_value_of_f_in_interval_l70_70654

noncomputable def f (x : ℝ) := (Real.sin x)^2 + (Real.sqrt 3) * Real.cos x - (3 / 4)

theorem maximum_value_of_f_in_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 := 
  sorry

end maximum_value_of_f_in_interval_l70_70654


namespace unpacked_books_30_l70_70494

theorem unpacked_books_30 :
  let total_books := 1485 * 42
  let books_per_box := 45
  total_books % books_per_box = 30 :=
by
  let total_books := 1485 * 42
  let books_per_box := 45
  have h : total_books % books_per_box = 30 := sorry
  exact h

end unpacked_books_30_l70_70494


namespace asymptote_of_hyperbola_l70_70652

theorem asymptote_of_hyperbola (x y : ℝ) :
  (x^2 - (y^2 / 4) = 1) → (y = 2 * x ∨ y = -2 * x) := sorry

end asymptote_of_hyperbola_l70_70652


namespace weight_of_berries_l70_70255

theorem weight_of_berries (total_weight : ℝ) (melon_weight : ℝ) : total_weight = 0.63 → melon_weight = 0.25 → total_weight - melon_weight = 0.38 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end weight_of_berries_l70_70255


namespace dealership_sales_l70_70205

theorem dealership_sales (sports_cars : ℕ) (sedans : ℕ) (trucks : ℕ) 
  (h1 : sports_cars = 36)
  (h2 : (3 : ℤ) * sedans = 5 * sports_cars)
  (h3 : (3 : ℤ) * trucks = 4 * sports_cars) :
  sedans = 60 ∧ trucks = 48 := 
sorry

end dealership_sales_l70_70205


namespace minimum_value_expression_l70_70367

theorem minimum_value_expression (x y z : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2) ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2)) :=
sorry

end minimum_value_expression_l70_70367


namespace cross_country_hours_l70_70473

-- Definitions based on the conditions from part a)
def total_hours_required : ℕ := 1500
def hours_day_flying : ℕ := 50
def hours_night_flying : ℕ := 9
def goal_months : ℕ := 6
def hours_per_month : ℕ := 220

-- Problem statement: prove she has already completed 1261 hours of cross-country flying
theorem cross_country_hours : 
  (goal_months * hours_per_month) - (hours_day_flying + hours_night_flying) = 1261 := 
by
  -- Proof omitted (using the solution steps)
  sorry

end cross_country_hours_l70_70473


namespace cost_of_baking_soda_l70_70328

-- Definitions of the condition
def students : ℕ := 23
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def total_cost_of_supplies : ℕ := 184

-- Main statement to prove
theorem cost_of_baking_soda : 
  (∀ (students : ℕ) (cost_of_bow : ℕ) (cost_of_vinegar : ℕ) (total_cost_of_supplies : ℕ),
    total_cost_of_supplies = students * (cost_of_bow + cost_of_vinegar) + students) → 
  total_cost_of_supplies = 23 * (5 + 2) + 23 → 
  184 = 23 * (5 + 2 + 1) :=
by
  sorry

end cost_of_baking_soda_l70_70328


namespace correct_assignment_l70_70417

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l70_70417


namespace find_height_of_cylinder_l70_70765

theorem find_height_of_cylinder (h r : ℝ) (π : ℝ) (SA : ℝ) (r_val : r = 3) (SA_val : SA = 36 * π) 
  (SA_formula : SA = 2 * π * r^2 + 2 * π * r * h) : h = 3 := 
by
  sorry

end find_height_of_cylinder_l70_70765


namespace solve_quadratic_inequality_l70_70600

theorem solve_quadratic_inequality (a x : ℝ) :
  (x ^ 2 - (2 + a) * x + 2 * a < 0) ↔ 
  ((a < 2 ∧ a < x ∧ x < 2) ∨ (a = 2 ∧ false) ∨ 
   (a > 2 ∧ 2 < x ∧ x < a)) :=
by sorry

end solve_quadratic_inequality_l70_70600


namespace original_savings_eq_920_l70_70875

variable (S : ℝ) -- Define S as a real number representing Linda's savings
variable (h1 : S * (1 / 4) = 230) -- Given condition

theorem original_savings_eq_920 :
  S = 920 :=
by
  sorry

end original_savings_eq_920_l70_70875


namespace A_wins_when_n_is_9_l70_70119

-- Definition of the game conditions and the strategy
def game (n : ℕ) (A_first : Bool) :=
  ∃ strategy : ℕ → ℕ,
    ∀ taken balls_left : ℕ,
      balls_left - taken > 0 →
      taken ≥ 1 → taken ≤ 3 →
      if A_first then
        (balls_left - taken = 0 → strategy (balls_left - taken) = 1) ∧
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)
      else
        (balls_left - taken = 0 → strategy (balls_left - taken) = 0) ∨
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)

-- Prove that for n = 9 A has a winning strategy
theorem A_wins_when_n_is_9 : game 9 true :=
sorry

end A_wins_when_n_is_9_l70_70119


namespace calculation_is_correct_l70_70295

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end calculation_is_correct_l70_70295


namespace quadratic_has_real_roots_iff_l70_70990

theorem quadratic_has_real_roots_iff (k : ℝ) : (∃ x : ℝ, x^2 + 2*x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end quadratic_has_real_roots_iff_l70_70990


namespace Juan_has_498_marbles_l70_70375

def ConnieMarbles : Nat := 323
def JuanMoreMarbles : Nat := 175
def JuanMarbles : Nat := ConnieMarbles + JuanMoreMarbles

theorem Juan_has_498_marbles : JuanMarbles = 498 := by
  sorry

end Juan_has_498_marbles_l70_70375


namespace derivative_of_f_l70_70201

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + (1 / 3) * ((3 * x - 1) / (3 * x^2 - 2 * x + 1))

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
by intros; sorry

end derivative_of_f_l70_70201


namespace polynomial_root_sum_l70_70969

theorem polynomial_root_sum : 
  ∀ (r1 r2 r3 r4 : ℝ), 
  (r1^4 - r1 - 504 = 0) ∧ 
  (r2^4 - r2 - 504 = 0) ∧ 
  (r3^4 - r3 - 504 = 0) ∧ 
  (r4^4 - r4 - 504 = 0) → 
  r1^4 + r2^4 + r3^4 + r4^4 = 2016 := by
sorry

end polynomial_root_sum_l70_70969


namespace triangle_area_40_l70_70022

noncomputable def area_of_triangle (base height : ℕ) : ℕ :=
  base * height / 2

theorem triangle_area_40
  (a : ℕ) (P B Q : (ℕ × ℕ)) (PB_side : (P.1 = 0 ∧ P.2 = 0) ∧ (B.1 = 10 ∧ B.2 = 0))
  (Q_vert_aboveP : Q.1 = 0 ∧ Q.2 = 8)
  (PQ_perp_PB : P.1 = Q.1)
  (PQ_length : (Q.snd - P.snd) = 8) :
  area_of_triangle 10 8 = 40 := by
  sorry

end triangle_area_40_l70_70022


namespace fraction_division_l70_70672

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l70_70672


namespace product_with_a_equals_3_l70_70921

theorem product_with_a_equals_3 (a : ℤ) (h : a = 3) : 
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 :=
by
  sorry

end product_with_a_equals_3_l70_70921


namespace prime_divisor_condition_l70_70139

theorem prime_divisor_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : q ∣ 2^p - 1) : p ∣ q - 1 :=
  sorry

end prime_divisor_condition_l70_70139


namespace circle_radius_value_l70_70997

theorem circle_radius_value (k : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → k = 16 :=
by
  sorry

end circle_radius_value_l70_70997


namespace solve_equation_l70_70121

theorem solve_equation (x : ℚ) (h1 : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 := by
  sorry

end solve_equation_l70_70121


namespace least_whole_number_for_ratio_l70_70315

theorem least_whole_number_for_ratio :
  ∃ x : ℕ, (6 - x) * 21 < (7 - x) * 16 ∧ x = 3 :=
by
  sorry

end least_whole_number_for_ratio_l70_70315


namespace find_length_QR_l70_70007

-- Conditions
variables {D E F Q R : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace Q] [MetricSpace R]
variables {DE EF DF QR : ℝ} (tangent : Q = E ∧ R = D)
variables (t₁ : de = 5) (t₂ : ef = 12) (t₃ : df = 13)

-- Problem: Prove that QR = 5 given the conditions.
theorem find_length_QR : QR = 5 :=
sorry

end find_length_QR_l70_70007


namespace most_likely_outcome_l70_70072

-- Define the probabilities for each outcome
def P_all_boys := (1/2)^6
def P_all_girls := (1/2)^6
def P_3_girls_3_boys := (Nat.choose 6 3) * (1/2)^6
def P_4_one_2_other := 2 * (Nat.choose 6 2) * (1/2)^6

-- Terms with values of each probability
lemma outcome_A : P_all_boys = 1 / 64 := by sorry
lemma outcome_B : P_all_girls = 1 / 64 := by sorry
lemma outcome_C : P_3_girls_3_boys = 20 / 64 := by sorry
lemma outcome_D : P_4_one_2_other = 30 / 64 := by sorry

-- Prove the main statement
theorem most_likely_outcome :
  P_4_one_2_other > P_all_boys ∧ P_4_one_2_other > P_all_girls ∧ P_4_one_2_other > P_3_girls_3_boys :=
by
  rw [outcome_A, outcome_B, outcome_C, outcome_D]
  sorry

end most_likely_outcome_l70_70072


namespace next_divisor_of_4_digit_even_number_l70_70434

theorem next_divisor_of_4_digit_even_number (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000)
  (h2 : n % 2 = 0) (hDiv : n % 221 = 0) :
  ∃ d, d > 221 ∧ d < n ∧ d % 13 = 0 ∧ d % 17 = 0 ∧ d = 442 :=
by
  use 442
  sorry

end next_divisor_of_4_digit_even_number_l70_70434


namespace sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l70_70041

noncomputable def sum_of_consecutive_triplets (a : Fin 12 → ℕ) (i : Fin 12) : ℕ :=
a i + a ((i + 1) % 12) + a ((i + 2) % 12)

theorem sum_of_consecutive_at_least_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i ≥ 20 :=
by
  sorry

theorem sum_of_consecutive_greater_than_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i > 20 :=
by
  sorry

end sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l70_70041


namespace tv_price_change_l70_70968

theorem tv_price_change (P : ℝ) :
  let decrease := 0.20
  let increase := 0.45
  let new_price := P * (1 - decrease)
  let final_price := new_price * (1 + increase)
  final_price - P = 0.16 * P := 
by
  sorry

end tv_price_change_l70_70968


namespace village_current_population_l70_70650

theorem village_current_population (initial_population : ℕ) (ten_percent_die : ℕ)
  (twenty_percent_leave : ℕ) : 
  initial_population = 4399 →
  ten_percent_die = initial_population / 10 →
  twenty_percent_leave = (initial_population - ten_percent_die) / 5 →
  (initial_population - ten_percent_die) - twenty_percent_leave = 3167 :=
sorry

end village_current_population_l70_70650


namespace multiplication_in_A_l70_70556

def A : Set ℤ :=
  {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem multiplication_in_A (x1 x2 : ℤ) (h1 : x1 ∈ A) (h2 : x2 ∈ A) :
  x1 * x2 ∈ A :=
sorry

end multiplication_in_A_l70_70556


namespace largest_whole_number_for_inequality_l70_70591

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end largest_whole_number_for_inequality_l70_70591


namespace probability_gpa_at_least_3_is_2_over_9_l70_70701

def gpa_points (grade : ℕ) : ℕ :=
  match grade with
  | 4 => 4 -- A
  | 3 => 3 -- B
  | 2 => 2 -- C
  | 1 => 1 -- D
  | _ => 0 -- otherwise

def probability_of_GPA_at_least_3 : ℚ :=
  let points_physics := gpa_points 4
  let points_chemistry := gpa_points 4
  let points_biology := gpa_points 3
  let total_known_points := points_physics + points_chemistry + points_biology
  let required_points := 18 - total_known_points -- 18 points needed in total for a GPA of at least 3.0
  -- Probabilities in Mathematics:
  let prob_math_A := 1 / 9
  let prob_math_B := 4 / 9
  let prob_math_C :=  4 / 9
  -- Probabilities in Sociology:
  let prob_soc_A := 1 / 3
  let prob_soc_B := 1 / 3
  let prob_soc_C := 1 / 3
  -- Calculate the total probability of achieving at least 7 points from Mathematics and Sociology
  let prob_case_1 := prob_math_A * prob_soc_A -- Both A in Mathematics and Sociology
  let prob_case_2 := prob_math_A * prob_soc_B -- A in Mathematics and B in Sociology
  let prob_case_3 := prob_math_B * prob_soc_A -- B in Mathematics and A in Sociology
  prob_case_1 + prob_case_2 + prob_case_3 -- Total Probability

theorem probability_gpa_at_least_3_is_2_over_9 : probability_of_GPA_at_least_3 = 2 / 9 :=
by sorry

end probability_gpa_at_least_3_is_2_over_9_l70_70701


namespace sin_cos_eq_one_sol_set_l70_70842

-- Define the interval
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := Real.sin x + Real.cos x = 1

-- Theorem statement: prove that the solution set is {0, π/2}
theorem sin_cos_eq_one_sol_set :
  ∀ (x : ℝ), in_interval x → satisfies_eq x ↔ x = 0 ∨ x = Real.pi / 2 := by
  sorry

end sin_cos_eq_one_sol_set_l70_70842


namespace smallest_n_19n_congruent_1453_mod_8_l70_70810

theorem smallest_n_19n_congruent_1453_mod_8 : 
  ∃ (n : ℕ), 19 * n % 8 = 1453 % 8 ∧ ∀ (m : ℕ), (19 * m % 8 = 1453 % 8 → n ≤ m) := 
sorry

end smallest_n_19n_congruent_1453_mod_8_l70_70810


namespace inequality_solution_set_l70_70838

def f (x : ℝ) : ℝ := x^3

theorem inequality_solution_set (x : ℝ) :
  (f (2 * x) + f (x - 1) < 0) ↔ (x < (1 / 3)) := 
sorry

end inequality_solution_set_l70_70838


namespace f_value_at_3_l70_70449

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_at_3 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 4 * x^2) : f 3 = -4 / 3 :=
by
  sorry

end f_value_at_3_l70_70449


namespace least_possible_value_of_smallest_integer_l70_70962

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℤ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + C + D) / 4 = 76 ∧ D = 90 →
  A = 37 :=
by
  sorry

end least_possible_value_of_smallest_integer_l70_70962


namespace quadratic_sum_l70_70892

theorem quadratic_sum (a b c : ℝ) (h : ∀ x : ℝ, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) :
  a + b + c = -88 := by
  sorry

end quadratic_sum_l70_70892


namespace number_of_pots_of_rosemary_l70_70773

-- Definitions based on the conditions
def total_leaves_basil (pots_basil : ℕ) (leaves_per_basil : ℕ) : ℕ := pots_basil * leaves_per_basil
def total_leaves_rosemary (pots_rosemary : ℕ) (leaves_per_rosemary : ℕ) : ℕ := pots_rosemary * leaves_per_rosemary
def total_leaves_thyme (pots_thyme : ℕ) (leaves_per_thyme : ℕ) : ℕ := pots_thyme * leaves_per_thyme

-- The given problem conditions
def pots_basil : ℕ := 3
def leaves_per_basil : ℕ := 4
def leaves_per_rosemary : ℕ := 18
def pots_thyme : ℕ := 6
def leaves_per_thyme : ℕ := 30
def total_leaves : ℕ := 354

-- Proving the number of pots of rosemary
theorem number_of_pots_of_rosemary : 
  ∃ (pots_rosemary : ℕ), 
  total_leaves_basil pots_basil leaves_per_basil + 
  total_leaves_rosemary pots_rosemary leaves_per_rosemary + 
  total_leaves_thyme pots_thyme leaves_per_thyme = 
  total_leaves ∧ pots_rosemary = 9 :=
by
  sorry  -- proof is omitted

end number_of_pots_of_rosemary_l70_70773


namespace initial_numbers_unique_l70_70353

theorem initial_numbers_unique 
  (A B C A' B' C' : ℕ) 
  (h1: 1 ≤ A ∧ A ≤ 50) 
  (h2: 1 ≤ B ∧ B ≤ 50) 
  (h3: 1 ≤ C ∧ C ≤ 50) 
  (final_ana : 104 = 2 * A + B + C)
  (final_beto : 123 = A + 2 * B + C)
  (final_caio : 137 = A + B + 2 * C) : 
  A = 13 ∧ B = 32 ∧ C = 46 :=
sorry

end initial_numbers_unique_l70_70353


namespace geometric_sequence_product_l70_70122

-- Define the geometric sequence sum and the initial conditions
variables {S : ℕ → ℚ} {a : ℕ → ℚ}
variables (q : ℚ) (h1 : a 1 = -1/2)
variables (h2 : S 6 / S 3 = 7 / 8)

-- The main proof problem statement
theorem geometric_sequence_product (h_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 2 * a 4 = 1 / 64 :=
sorry

end geometric_sequence_product_l70_70122


namespace triangle_internal_angle_A_l70_70114

theorem triangle_internal_angle_A {B C A : ℝ} (hB : Real.tan B = -2) (hC : Real.tan C = 1 / 3) (h_sum: A = π - B - C) : A = π / 4 :=
by
  sorry

end triangle_internal_angle_A_l70_70114


namespace percent_exceed_not_ticketed_l70_70400

-- Defining the given conditions
def total_motorists : ℕ := 100
def percent_exceed_limit : ℕ := 50
def percent_with_tickets : ℕ := 40

-- Calculate the number of motorists exceeding the limit and receiving tickets
def motorists_exceed_limit := total_motorists * percent_exceed_limit / 100
def motorists_with_tickets := total_motorists * percent_with_tickets / 100

-- Theorem: Percentage of motorists exceeding the limit but not receiving tickets
theorem percent_exceed_not_ticketed : 
  (motorists_exceed_limit - motorists_with_tickets) * 100 / motorists_exceed_limit = 20 := 
by
  sorry

end percent_exceed_not_ticketed_l70_70400


namespace polygon_diagonals_l70_70087

-- Lean statement of the problem

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 2018) : n = 2021 :=
  by sorry

end polygon_diagonals_l70_70087


namespace find_a_l70_70243

theorem find_a (a : ℝ) (h : (a - 1) ≠ 0) :
  (∃ x : ℝ, ((a - 1) * x^2 + x + a^2 - 1 = 0) ∧ x = 0) → a = -1 :=
by
  sorry

end find_a_l70_70243


namespace Elberta_has_21_dollars_l70_70124

theorem Elberta_has_21_dollars
  (Granny_Smith : ℕ)
  (Anjou : ℕ)
  (Elberta : ℕ)
  (h1 : Granny_Smith = 72)
  (h2 : Anjou = Granny_Smith / 4)
  (h3 : Elberta = Anjou + 3) :
  Elberta = 21 := 
  by {
    sorry
  }

end Elberta_has_21_dollars_l70_70124


namespace ribbon_used_l70_70083

def total_ribbon : ℕ := 84
def leftover_ribbon : ℕ := 38
def used_ribbon : ℕ := 46

theorem ribbon_used : total_ribbon - leftover_ribbon = used_ribbon := sorry

end ribbon_used_l70_70083


namespace arctan_tan_sub_eq_l70_70228

noncomputable def arctan_tan_sub (a b : ℝ) : ℝ := Real.arctan (Real.tan a - 3 * Real.tan b)

theorem arctan_tan_sub_eq (a b : ℝ) (ha : a = 75) (hb : b = 15) :
  arctan_tan_sub a b = 75 :=
by
  sorry

end arctan_tan_sub_eq_l70_70228


namespace sector_area_l70_70242

theorem sector_area (radius : ℝ) (central_angle : ℝ) (h1 : radius = 3) (h2 : central_angle = 2 * Real.pi / 3) : 
    (1 / 2) * radius^2 * central_angle = 6 * Real.pi :=
by
  rw [h1, h2]
  sorry

end sector_area_l70_70242


namespace volume_of_dug_out_earth_l70_70717

theorem volume_of_dug_out_earth
  (diameter depth : ℝ)
  (h_diameter : diameter = 2) 
  (h_depth : depth = 14) 
  : abs ((π * (1 / 2 * diameter / 2) ^ 2 * depth) - 44) < 0.1 :=
by
  -- Provide a placeholder for the proof
  sorry

end volume_of_dug_out_earth_l70_70717


namespace solve_for_x_l70_70988

theorem solve_for_x (x : ℝ) (h : Real.exp (Real.log 7) = 9 * x + 2) : x = 5 / 9 :=
by {
    -- Proof needs to be filled here
    sorry
}

end solve_for_x_l70_70988


namespace ending_number_of_sequence_divisible_by_11_l70_70713

theorem ending_number_of_sequence_divisible_by_11 : 
  ∃ (n : ℕ), 19 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → n = 19 + 11 * k) ∧ n = 77 :=
by
  sorry

end ending_number_of_sequence_divisible_by_11_l70_70713


namespace highest_y_coordinate_l70_70985

-- Define the conditions
def ellipse_condition (x y : ℝ) : Prop :=
  (x^2 / 25) + ((y - 3)^2 / 9) = 1

-- The theorem to prove
theorem highest_y_coordinate : ∃ x : ℝ, ∀ y : ℝ, ellipse_condition x y → y ≤ 6 :=
sorry

end highest_y_coordinate_l70_70985


namespace regions_formula_l70_70095

-- Define the number of regions R(n) created by n lines
def regions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

-- Theorem statement: for n lines, no two parallel, no three concurrent, the regions are defined by the formula
theorem regions_formula (n : ℕ) : regions n = 1 + (n * (n + 1)) / 2 := 
by sorry

end regions_formula_l70_70095


namespace remainder_when_divided_by_r_minus_1_l70_70068

def f (r : Int) : Int := r^14 - r + 5

theorem remainder_when_divided_by_r_minus_1 : f 1 = 5 := by
  sorry

end remainder_when_divided_by_r_minus_1_l70_70068


namespace find_integer_l70_70326

theorem find_integer (n : ℤ) (h : 5 * (n - 2) = 85) : n = 19 :=
sorry

end find_integer_l70_70326


namespace no_possible_blue_socks_l70_70700

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l70_70700


namespace find_k_l70_70676

variables (l w : ℝ) (p A k : ℝ)

def rectangle_conditions : Prop :=
  (l / w = 5 / 2) ∧ (p = 2 * (l + w))

theorem find_k (h : rectangle_conditions l w p) :
  A = (5 / 98) * p^2 :=
sorry

end find_k_l70_70676


namespace max_minus_min_on_interval_l70_70637

def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_minus_min_on_interval (a : ℝ) :
  let M := max (f 0 a) (f 3 a)
  let N := f 1 a
  M - N = 20 :=
by
  sorry

end max_minus_min_on_interval_l70_70637


namespace unattainable_y_value_l70_70809

theorem unattainable_y_value :
  ∀ (y x : ℝ), (y = (1 - x) / (2 * x^2 + 3 * x + 4)) → (∀ x, 2 * x^2 + 3 * x + 4 ≠ 0) → y ≠ 0 :=
by
  intros y x h1 h2
  -- Proof to be provided
  sorry

end unattainable_y_value_l70_70809


namespace positive_difference_even_odd_sum_l70_70448

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l70_70448


namespace acres_used_for_corn_l70_70780

noncomputable def total_acres : ℝ := 1634
noncomputable def beans_ratio : ℝ := 4.5
noncomputable def wheat_ratio : ℝ := 2.3
noncomputable def corn_ratio : ℝ := 3.8
noncomputable def barley_ratio : ℝ := 3.4

noncomputable def total_parts : ℝ := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
noncomputable def acres_per_part : ℝ := total_acres / total_parts
noncomputable def corn_acres : ℝ := corn_ratio * acres_per_part

theorem acres_used_for_corn :
  corn_acres = 443.51 := by
  sorry

end acres_used_for_corn_l70_70780


namespace sin_alpha_sub_beta_cos_beta_l70_70382

variables (α β : ℝ)
variables (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
variables (h1 : Real.sin α = 3 / 5)
variables (h2 : Real.tan (α - β) = -1 / 3)

theorem sin_alpha_sub_beta : Real.sin (α - β) = - Real.sqrt 10 / 10 :=
by
  sorry

theorem cos_beta : Real.cos β = 9 * Real.sqrt 10 / 50 :=
by
  sorry

end sin_alpha_sub_beta_cos_beta_l70_70382


namespace bucket_problem_l70_70265

variable (A B C : ℝ)

theorem bucket_problem :
  (A - 6 = (1 / 3) * (B + 6)) →
  (B - 6 = (1 / 2) * (A + 6)) →
  (C - 8 = (1 / 2) * (A + 8)) →
  A = 13.2 :=
by
  sorry

end bucket_problem_l70_70265


namespace expression_evaluation_l70_70561

theorem expression_evaluation (m n : ℤ) (h : m * n = m + 3) : 2 * m * n + 3 * m - 5 * m * n - 10 = -19 := 
by 
  sorry

end expression_evaluation_l70_70561


namespace coconuts_total_l70_70024

theorem coconuts_total (B_trips : Nat) (Ba_coconuts_per_trip : Nat) (Br_coconuts_per_trip : Nat) (combined_trips : Nat) (B_totals : B_trips = 12) (Ba_coconuts : Ba_coconuts_per_trip = 4) (Br_coconuts : Br_coconuts_per_trip = 8) : combined_trips * (Ba_coconuts_per_trip + Br_coconuts_per_trip) = 144 := 
by
  simp [B_totals, Ba_coconuts, Br_coconuts]
  sorry

end coconuts_total_l70_70024


namespace total_length_of_scale_l70_70003

theorem total_length_of_scale 
  (n : ℕ) (len_per_part : ℕ) 
  (h_n : n = 5) 
  (h_len_per_part : len_per_part = 25) :
  n * len_per_part = 125 :=
by
  sorry

end total_length_of_scale_l70_70003


namespace find_some_number_l70_70435

noncomputable def some_number : ℝ := 1000
def expr_approx (a b c d : ℝ) := (a * b) / c = d

theorem find_some_number :
  expr_approx 3.241 14 some_number 0.045374000000000005 :=
by sorry

end find_some_number_l70_70435


namespace find_range_of_x_l70_70816

-- Conditions
variable (f : ℝ → ℝ)
variable (even_f : ∀ x : ℝ, f x = f (-x))
variable (mono_incr_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Equivalent proof statement
theorem find_range_of_x (x : ℝ) :
  f (Real.log (abs (x + 1)) / Real.log (1 / 2)) < f (-1) ↔ x ∈ Set.Ioo (-3 : ℝ) (-3 / 2) ∪ Set.Ioo (-1 / 2) 1 := by
  sorry

end find_range_of_x_l70_70816


namespace total_weight_full_bucket_l70_70349

theorem total_weight_full_bucket (x y c d : ℝ) 
(h1 : x + 3/4 * y = c) 
(h2 : x + 1/3 * y = d) :
x + y = (8 * c - 3 * d) / 5 :=
sorry

end total_weight_full_bucket_l70_70349


namespace problem_1_problem_2_l70_70101

-- Define propositions
def prop_p (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (4 - m) + y^2 / m = 1)

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

def prop_s (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

-- Problems
theorem problem_1 (m : ℝ) (h : prop_s m) : m < 0 ∨ m ≥ 1 := 
  sorry

theorem problem_2 {m : ℝ} (h1 : prop_p m ∨ prop_q m) (h2 : ¬ prop_q m) : 1 ≤ m ∧ m < 2 :=
  sorry

end problem_1_problem_2_l70_70101


namespace train_length_l70_70868

theorem train_length (speed_faster speed_slower : ℝ) (time_sec : ℝ) (length_each_train : ℝ) :
  speed_faster = 47 ∧ speed_slower = 36 ∧ time_sec = 36 ∧ 
  (length_each_train = 55 ↔ 2 * length_each_train = ((speed_faster - speed_slower) * (1000/3600) * time_sec)) :=
by {
  -- We declare the speeds in km/hr and convert the relative speed to m/s for calculation.
  sorry
}

end train_length_l70_70868


namespace josh_initial_marbles_l70_70882

def marbles_initial (lost : ℕ) (left : ℕ) : ℕ := lost + left

theorem josh_initial_marbles :
  marbles_initial 5 4 = 9 :=
by sorry

end josh_initial_marbles_l70_70882


namespace operation_is_commutative_and_associative_l70_70854

variables {S : Type} (op : S → S → S)

-- defining the properties given in the conditions
def idempotent (op : S → S → S) : Prop :=
  ∀ (a : S), op a a = a

def medial (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op (op b c) a

-- defining commutative and associative properties
def commutative (op : S → S → S) : Prop :=
  ∀ (a b : S), op a b = op b a

def associative (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op a (op b c)

-- statement of the theorem to prove
theorem operation_is_commutative_and_associative 
  (idemp : idempotent op) 
  (med : medial op) : commutative op ∧ associative op :=
sorry

end operation_is_commutative_and_associative_l70_70854


namespace problem_inequality_l70_70752

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end problem_inequality_l70_70752


namespace restaurant_customers_prediction_l70_70358

def total_customers_saturday (breakfast_customers_friday lunch_customers_friday dinner_customers_friday : ℝ) : ℝ :=
  let breakfast_customers_saturday := 2 * breakfast_customers_friday
  let lunch_customers_saturday := lunch_customers_friday + 0.25 * lunch_customers_friday
  let dinner_customers_saturday := dinner_customers_friday - 0.15 * dinner_customers_friday
  breakfast_customers_saturday + lunch_customers_saturday + dinner_customers_saturday

theorem restaurant_customers_prediction :
  let breakfast_customers_friday := 73
  let lunch_customers_friday := 127
  let dinner_customers_friday := 87
  total_customers_saturday breakfast_customers_friday lunch_customers_friday dinner_customers_friday = 379 := 
by
  sorry

end restaurant_customers_prediction_l70_70358


namespace pencils_distributed_per_container_l70_70263

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end pencils_distributed_per_container_l70_70263


namespace intersection_M_N_l70_70827

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l70_70827


namespace no_real_solutions_l70_70775

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (2 - x^2) / x

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 → (f x + 2 * f (1 / x) = 3 * x)) →
  (∀ x : ℝ, f x = f (-x) → false) :=
by
  intro h1 h2
  sorry

end no_real_solutions_l70_70775


namespace problem_proof_l70_70712

noncomputable def arithmetic_sequences (a b : ℕ → ℤ) (S T : ℕ → ℤ) :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2 ∧
         T n = (n * (2 * b 0 + (n - 1) * (b 1 - b 0))) / 2

theorem problem_proof 
  (a b : ℕ → ℤ) 
  (S T : ℕ → ℤ)
  (h_seq : arithmetic_sequences a b S T)
  (h_relation : ∀ n, S n / T n = (7 * n : ℤ) / (n + 3)) :
  (a 5) / (b 5) = 21 / 4 :=
by 
  sorry

end problem_proof_l70_70712


namespace condition_necessary_but_not_sufficient_l70_70682

variable (m : ℝ)

/-- The problem statement and proof condition -/
theorem condition_necessary_but_not_sufficient :
  (∀ x : ℝ, |x - 2| + |x + 2| > m) → (∀ x : ℝ, x^2 + m * x + 4 > 0) :=
by {
  sorry
}

end condition_necessary_but_not_sufficient_l70_70682


namespace no_solution_fractional_eq_l70_70609

theorem no_solution_fractional_eq :
  ¬∃ x : ℝ, (1 - x) / (x - 2) = 1 / (2 - x) + 1 :=
by
  -- The proof is intentionally omitted.
  sorry

end no_solution_fractional_eq_l70_70609


namespace dalton_movies_l70_70814

variable (D : ℕ) -- Dalton's movies
variable (Hunter : ℕ := 12) -- Hunter's movies
variable (Alex : ℕ := 15) -- Alex's movies
variable (Together : ℕ := 2) -- Movies watched together
variable (TotalDifferentMovies : ℕ := 30) -- Total different movies

theorem dalton_movies (h : D + Hunter + Alex - Together * 3 = TotalDifferentMovies) : D = 9 := by
  sorry

end dalton_movies_l70_70814


namespace inequality_solution_l70_70946

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x - 3)^2 ≥ 0} = (Set.Iic (-1) ∪ Set.Ici 1) :=
by
  sorry

end inequality_solution_l70_70946


namespace find_a_from_function_property_l70_70251

theorem find_a_from_function_property {a : ℝ} (h : ∀ (x : ℝ), (0 ≤ x → x ≤ 1 → ax ≤ 3) ∧ (0 ≤ x → x ≤ 1 → ax ≥ 3)) :
  a = 3 :=
sorry

end find_a_from_function_property_l70_70251


namespace dot_product_expression_max_value_of_dot_product_l70_70256

variable (x : ℝ)
variable (k : ℤ)
variable (a : ℝ × ℝ := (Real.cos x, -1 + Real.sin x))
variable (b : ℝ × ℝ := (2 * Real.cos x, Real.sin x))

theorem dot_product_expression :
  (a.1 * b.1 + a.2 * b.2) = 2 - 3 * (Real.sin x)^2 - (Real.sin x) := 
sorry

theorem max_value_of_dot_product :
  ∃ (x : ℝ), 2 - 3 * (Real.sin x)^2 - (Real.sin x) = 9 / 4 ∧ 
  (Real.sin x = -1/2 ∧ 
  (x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 11 * Real.pi / 6 + 2 * k * Real.pi)) := 
sorry

end dot_product_expression_max_value_of_dot_product_l70_70256


namespace members_playing_both_badminton_and_tennis_l70_70688

-- Definitions based on conditions
def N : ℕ := 35  -- Total number of members in the sports club
def B : ℕ := 15  -- Number of people who play badminton
def T : ℕ := 18  -- Number of people who play tennis
def Neither : ℕ := 5  -- Number of people who do not play either sport

-- The theorem based on the inclusion-exclusion principle
theorem members_playing_both_badminton_and_tennis :
  (B + T - (N - Neither) = 3) :=
by
  sorry

end members_playing_both_badminton_and_tennis_l70_70688


namespace problem_l70_70966

variable (R S : Prop)

theorem problem (h1 : R → S) :
  ((¬S → ¬R) ∧ (¬R ∨ S)) :=
by
  sorry

end problem_l70_70966


namespace prime_factors_of_expression_l70_70180

theorem prime_factors_of_expression
  (p : ℕ) (prime_p : Nat.Prime p) :
  (∀ x y : ℕ, 0 < x → 0 < y → p ∣ ((x + y)^19 - x^19 - y^19)) ↔ (p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) :=
by
  sorry

end prime_factors_of_expression_l70_70180


namespace identify_genuine_coins_l70_70377

section IdentifyGenuineCoins

variables (coins : Fin 25 → ℝ) 
          (is_genuine : Fin 25 → Prop) 
          (is_counterfeit : Fin 25 → Prop)

-- Conditions
axiom coin_total : ∀ i, is_genuine i ∨ is_counterfeit i
axiom genuine_count : ∃ s : Finset (Fin 25), s.card = 22 ∧ ∀ i ∈ s, is_genuine i
axiom counterfeit_count : ∃ t : Finset (Fin 25), t.card = 3 ∧ ∀ i ∈ t, is_counterfeit i
axiom genuine_weight : ∃ w : ℝ, ∀ i, is_genuine i → coins i = w
axiom counterfeit_weight : ∃ c : ℝ, ∀ i, is_counterfeit i → coins i = c
axiom counterfeit_lighter : ∀ (w c : ℝ), (∃ i, is_genuine i → coins i = w) ∧ (∃ j, is_counterfeit j → coins j = c) → c < w

-- Theorem: Identifying 6 genuine coins using two weighings
theorem identify_genuine_coins : ∃ s : Finset (Fin 25), s.card = 6 ∧ ∀ i ∈ s, is_genuine i :=
sorry

end IdentifyGenuineCoins

end identify_genuine_coins_l70_70377


namespace solution_l70_70607

noncomputable def problem : Prop := 
  - (Real.sin (133 * Real.pi / 180)) * (Real.cos (197 * Real.pi / 180)) -
  (Real.cos (47 * Real.pi / 180)) * (Real.cos (73 * Real.pi / 180)) = 1 / 2

theorem solution : problem :=
by
  sorry

end solution_l70_70607


namespace transformation_correct_l70_70340

noncomputable def original_function (x : ℝ) : ℝ := 2^x
noncomputable def transformed_function (x : ℝ) : ℝ := 2^x - 1
noncomputable def log_function (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1

theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = log_function (original_function x) :=
by
  intros x
  rw [transformed_function, log_function, original_function]
  sorry

end transformation_correct_l70_70340


namespace sum_of_coeffs_eq_59049_l70_70368

-- Definition of the polynomial
def poly (x y z : ℕ) : ℕ :=
  (2 * x - 3 * y + 4 * z) ^ 10

-- Conjecture: The sum of the numerical coefficients in poly when x, y, and z are set to 1 is 59049
theorem sum_of_coeffs_eq_59049 : poly 1 1 1 = 59049 := by
  sorry

end sum_of_coeffs_eq_59049_l70_70368


namespace find_number_l70_70954

theorem find_number (x : ℝ) :
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 :=
by
  intro h
  sorry

end find_number_l70_70954


namespace solve_expression_l70_70456

theorem solve_expression : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 :=
by
  sorry

end solve_expression_l70_70456


namespace solution_x_alcohol_percentage_l70_70762

theorem solution_x_alcohol_percentage (P : ℝ) :
  let y_percentage := 0.30
  let mixture_percentage := 0.25
  let y_volume := 600
  let x_volume := 200
  let mixture_volume := y_volume + x_volume
  let y_alcohol_content := y_volume * y_percentage
  let mixture_alcohol_content := mixture_volume * mixture_percentage
  P * x_volume + y_alcohol_content = mixture_alcohol_content →
  P = 0.10 :=
by
  intros
  sorry

end solution_x_alcohol_percentage_l70_70762


namespace correct_calculation_l70_70076

theorem correct_calculation (x y a b : ℝ) :
  (3*x + 3*y ≠ 6*x*y) ∧
  (x + x ≠ x^2) ∧
  (-9*y^2 + 16*y^2 ≠ 7) ∧
  (9*a^2*b - 9*a^2*b = 0) :=
by
  sorry

end correct_calculation_l70_70076


namespace four_op_two_l70_70332

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end four_op_two_l70_70332


namespace ratio_of_trout_l70_70721

-- Definition of the conditions
def trout_caught_by_Sara : Nat := 5
def trout_caught_by_Melanie : Nat := 10

-- Theorem stating the main claim to be proved
theorem ratio_of_trout : trout_caught_by_Melanie / trout_caught_by_Sara = 2 := by
  sorry

end ratio_of_trout_l70_70721


namespace problem_part1_problem_part2_l70_70278

noncomputable def f (x : ℝ) (m : ℝ) := Real.sqrt (|x + 2| + |x - 4| - m)

theorem problem_part1 (m : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 := 
by
  sorry

theorem problem_part2 (a b : ℕ) (n : ℝ) (h1 : (0 < a) ∧ (0 < b)) (h2 : n = 6) 
  (h3 : (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = n) : 
  ∃ (value : ℝ), 4 * a + 7 * b = 3 / 2 := 
by
  sorry

end problem_part1_problem_part2_l70_70278


namespace find_m_value_l70_70987

variable (m : ℝ)

theorem find_m_value (h1 : m^2 - 3 * m = 4)
                     (h2 : m^2 = 5 * m + 6) : m = -1 :=
sorry

end find_m_value_l70_70987


namespace simplify_expression_l70_70471

theorem simplify_expression (y : ℝ) : 
  3 * y - 5 * y ^ 2 + 12 - (7 - 3 * y + 5 * y ^ 2) = -10 * y ^ 2 + 6 * y + 5 :=
by 
  sorry

end simplify_expression_l70_70471


namespace equivalence_of_negation_l70_70296

-- Define the statement for the negation
def negation_stmt := ¬ ∃ x0 : ℝ, x0 ≤ 0 ∧ x0^2 ≥ 0

-- Define the equivalent statement after negation
def equivalent_stmt := ∀ x : ℝ, x ≤ 0 → x^2 < 0

-- The theorem stating that the negation_stmt is equivalent to equivalent_stmt
theorem equivalence_of_negation : negation_stmt ↔ equivalent_stmt := 
sorry

end equivalence_of_negation_l70_70296


namespace scale_length_discrepancy_l70_70155

theorem scale_length_discrepancy
  (scale_length_feet : ℝ)
  (parts : ℕ)
  (part_length_inches : ℝ)
  (ft_to_inch : ℝ := 12)
  (total_length_inches : ℝ := parts * part_length_inches)
  (scale_length_inches : ℝ := scale_length_feet * ft_to_inch) :
  scale_length_feet = 7 → 
  parts = 4 → 
  part_length_inches = 24 →
  total_length_inches - scale_length_inches = 12 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end scale_length_discrepancy_l70_70155


namespace exposed_circular_segment_sum_l70_70845

theorem exposed_circular_segment_sum (r h : ℕ) (angle : ℕ) (a b c : ℕ) :
    r = 8 ∧ h = 10 ∧ angle = 90 ∧ a = 16 ∧ b = 0 ∧ c = 0 → a + b + c = 16 :=
by
  intros
  sorry

end exposed_circular_segment_sum_l70_70845


namespace mean_of_reciprocals_of_first_four_primes_l70_70534

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l70_70534


namespace problem_equivalent_proof_l70_70380

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem_equivalent_proof : ((sqrt 3 - 2) ^ 0 - Real.logb 2 (sqrt 2)) = 1 / 2 :=
by
  sorry

end problem_equivalent_proof_l70_70380


namespace find_first_number_in_second_set_l70_70603

theorem find_first_number_in_second_set: 
  ∃ x: ℕ, (20 + 40 + 60) / 3 = (x + 80 + 15) / 3 + 5 ∧ x = 10 :=
by
  sorry

end find_first_number_in_second_set_l70_70603


namespace average_a_b_l70_70741

theorem average_a_b (a b : ℝ) (h : (4 + 6 + 8 + a + b) / 5 = 20) : (a + b) / 2 = 41 :=
by
  sorry

end average_a_b_l70_70741


namespace beach_relaxing_people_l70_70016

def row1_original := 24
def row1_got_up := 3

def row2_original := 20
def row2_got_up := 5

def row3_original := 18

def total_left_relaxing (r1o r1u r2o r2u r3o : Nat) : Nat :=
  r1o + r2o + r3o - (r1u + r2u)

theorem beach_relaxing_people : total_left_relaxing row1_original row1_got_up row2_original row2_got_up row3_original = 54 :=
by
  sorry

end beach_relaxing_people_l70_70016


namespace binary_remainder_div_8_l70_70370

theorem binary_remainder_div_8 (n : ℕ) (h : n = 0b101100110011) : n % 8 = 3 :=
by sorry

end binary_remainder_div_8_l70_70370


namespace max_k_consecutive_sum_l70_70791

theorem max_k_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, k * (2 * n + k - 1) = 2^2 * 3^8 ∧ ∀ k' > k, ¬ ∃ n', n' > 0 ∧ k' * (2 * n' + k' - 1) = 2^2 * 3^8 := sorry

end max_k_consecutive_sum_l70_70791


namespace total_weight_lifted_l70_70566

-- Given definitions from the conditions
def weight_left_hand : ℕ := 10
def weight_right_hand : ℕ := 10

-- The proof problem statement
theorem total_weight_lifted : weight_left_hand + weight_right_hand = 20 := 
by 
  -- Proof goes here
  sorry

end total_weight_lifted_l70_70566


namespace distinct_sum_product_problem_l70_70420

theorem distinct_sum_product_problem (S : ℤ) (hS : S ≥ 100) :
  ∃ a b c P : ℤ, a > b ∧ b > c ∧ a + b + c = S ∧ a * b * c = P ∧ 
    ¬(∀ x y z : ℤ, x > y ∧ y > z ∧ x + y + z = S → a = x ∧ b = y ∧ c = z) := 
sorry

end distinct_sum_product_problem_l70_70420


namespace height_of_oil_truck_tank_l70_70768

/-- 
Given that a stationary oil tank is a right circular cylinder 
with a radius of 100 feet and its oil level dropped by 0.025 feet,
proving that if this oil is transferred to a right circular 
cylindrical oil truck's tank with a radius of 5 feet, then the 
height of the oil in the truck's tank will be 10 feet. 
-/
theorem height_of_oil_truck_tank
    (radius_stationary : ℝ) (height_drop_stationary : ℝ) (radius_truck : ℝ) 
    (height_truck : ℝ) (π : ℝ)
    (h1 : radius_stationary = 100)
    (h2 : height_drop_stationary = 0.025)
    (h3 : radius_truck = 5)
    (pi_approx : π = 3.14159265) :
    height_truck = 10 :=
by
    sorry

end height_of_oil_truck_tank_l70_70768


namespace height_difference_zero_l70_70634

-- Define the problem statement and conditions
theorem height_difference_zero (a b : ℝ) (h1 : ∀ x, y = 2 * x^2)
  (h2 : b - a^2 = 1 / 4) : 
  ( b - 2 * a^2) = 0 :=
by
  sorry

end height_difference_zero_l70_70634


namespace smallest_n_for_triangle_area_l70_70436

theorem smallest_n_for_triangle_area :
  ∃ n : ℕ, 10 * n^4 - 8 * n^3 - 52 * n^2 + 32 * n - 24 > 10000 ∧ ∀ m : ℕ, 
  (m < n → ¬ (10 * m^4 - 8 * m^3 - 52 * m^2 + 32 * m - 24 > 10000)) :=
sorry

end smallest_n_for_triangle_area_l70_70436


namespace fourth_number_pascal_row_l70_70671

theorem fourth_number_pascal_row : (Nat.choose 12 3) = 220 := sorry

end fourth_number_pascal_row_l70_70671


namespace vehicle_speeds_l70_70575

theorem vehicle_speeds (d t: ℕ) (b_speed c_speed : ℕ) (h1 : d = 80) (h2 : c_speed = 3 * b_speed) (h3 : t = 3) (arrival_difference : ℕ) (h4 : arrival_difference = 1 / 3):
  b_speed = 20 ∧ c_speed = 60 :=
by
  sorry

end vehicle_speeds_l70_70575


namespace tan_of_obtuse_angle_l70_70559

theorem tan_of_obtuse_angle (α : ℝ) (h_cos : Real.cos α = -1/2) (h_obtuse : π/2 < α ∧ α < π) :
  Real.tan α = -Real.sqrt 3 :=
sorry

end tan_of_obtuse_angle_l70_70559


namespace sum_of_divisors_2000_l70_70830

theorem sum_of_divisors_2000 (n : ℕ) (h : n < 2000) :
  ∃ (s : Finset ℕ), (s ⊆ {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000}) ∧ s.sum id = n :=
by
  -- Proof goes here
  sorry

end sum_of_divisors_2000_l70_70830


namespace triangle_is_isosceles_right_l70_70677

theorem triangle_is_isosceles_right
  (a b c : ℝ)
  (A B C : ℕ)
  (h1 : c = a * Real.cos B)
  (h2 : b = a * Real.sin C) :
  C = 90 ∧ B = 90 ∧ A = 90 :=
sorry

end triangle_is_isosceles_right_l70_70677


namespace percent_greater_than_fraction_l70_70550

theorem percent_greater_than_fraction : 
  (0.80 * 40) - (4/5) * 20 = 16 :=
by
  sorry

end percent_greater_than_fraction_l70_70550


namespace area_of_rhombus_with_diagonals_6_and_8_l70_70276

theorem area_of_rhombus_with_diagonals_6_and_8 : 
  ∀ (d1 d2 : ℕ), d1 = 6 → d2 = 8 → (1 / 2 : ℝ) * d1 * d2 = 24 :=
by
  intros d1 d2 h1 h2
  sorry

end area_of_rhombus_with_diagonals_6_and_8_l70_70276


namespace mult_closest_l70_70807

theorem mult_closest :
  0.0004 * 9000000 = 3600 := sorry

end mult_closest_l70_70807


namespace mini_marshmallows_count_l70_70298

theorem mini_marshmallows_count (total_marshmallows large_marshmallows : ℕ) (h1 : total_marshmallows = 18) (h2 : large_marshmallows = 8) :
  total_marshmallows - large_marshmallows = 10 :=
by 
  sorry

end mini_marshmallows_count_l70_70298


namespace positive_root_condition_negative_root_condition_zero_root_condition_l70_70045

-- Positive root condition
theorem positive_root_condition {a b : ℝ} (h : a * b < 0) : ∃ x : ℝ, a * x + b = 0 ∧ x > 0 :=
by
  sorry

-- Negative root condition
theorem negative_root_condition {a b : ℝ} (h : a * b > 0) : ∃ x : ℝ, a * x + b = 0 ∧ x < 0 :=
by
  sorry

-- Root equal to zero condition
theorem zero_root_condition {a b : ℝ} (h₁ : b = 0) (h₂ : a ≠ 0) : ∃ x : ℝ, a * x + b = 0 ∧ x = 0 :=
by
  sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l70_70045


namespace expression_value_l70_70422

theorem expression_value (x : ℝ) (hx1 : x ≠ -1) (hx2 : x ≠ 2) :
  (2 * x ^ 2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := 
by
  sorry

end expression_value_l70_70422


namespace roots_of_quadratic_range_k_l70_70646

theorem roots_of_quadratic_range_k :
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ 
    x1 ≠ x2 ∧ 
    (x1 ≠ 1 ∧ x2 ≠ 1) ∧
    ∀ k : ℝ, x1 ^ 2 + (k - 3) * x1 + k ^ 2 = 0 ∧ x2 ^ 2 + (k - 3) * x2 + k ^ 2 = 0) ↔
  ((k : ℝ) < 1 ∧ k > -2) :=
sorry

end roots_of_quadratic_range_k_l70_70646


namespace tissue_magnification_l70_70309

theorem tissue_magnification
  (diameter_magnified : ℝ)
  (diameter_actual : ℝ)
  (h1 : diameter_magnified = 5)
  (h2 : diameter_actual = 0.005) :
  diameter_magnified / diameter_actual = 1000 :=
by
  -- proof goes here
  sorry

end tissue_magnification_l70_70309


namespace shelby_gold_stars_today_l70_70288

-- Define the number of gold stars Shelby earned yesterday
def gold_stars_yesterday := 4

-- Define the total number of gold stars Shelby earned
def total_gold_stars := 7

-- Define the number of gold stars Shelby earned today
def gold_stars_today := total_gold_stars - gold_stars_yesterday

-- The theorem to prove
theorem shelby_gold_stars_today : gold_stars_today = 3 :=
by 
  -- The proof will go here.
  sorry

end shelby_gold_stars_today_l70_70288


namespace jackie_break_duration_l70_70685

noncomputable def push_ups_no_breaks : ℕ := 30

noncomputable def push_ups_with_breaks : ℕ := 22

noncomputable def total_breaks : ℕ := 2

theorem jackie_break_duration :
  (5 * 6 - push_ups_with_breaks) * (10 / 5) / total_breaks = 8 := by
-- Given that
-- 1) Jackie does 5 push-ups in 10 seconds
-- 2) Jackie takes 2 breaks in one minute and performs 22 push-ups
-- We need to prove the duration of each break
sorry

end jackie_break_duration_l70_70685


namespace remainder_of_sum_l70_70669

theorem remainder_of_sum (a b c : ℕ) (h₁ : a * b * c % 7 = 1) (h₂ : 2 * c % 7 = 5) (h₃ : 3 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_of_sum_l70_70669


namespace unique_nonzero_b_l70_70745

variable (a b m n : ℝ)
variable (h_ne : m ≠ n)
variable (h_m_nonzero : m ≠ 0)
variable (h_n_nonzero : n ≠ 0)

theorem unique_nonzero_b (h : (a * m + b * n + m)^2 - (a * m + b * n + n)^2 = (m - n)^2) : 
  a = 0 ∧ b = -1 :=
sorry

end unique_nonzero_b_l70_70745


namespace inverse_proposition_equivalence_l70_70786

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end inverse_proposition_equivalence_l70_70786


namespace solution_set_of_inequality_system_l70_70378

theorem solution_set_of_inequality_system (x : ℝ) : 
  (x + 5 < 4) ∧ (3 * x + 1 ≥ 2 * (2 * x - 1)) ↔ (x < -1) :=
  by
  sorry

end solution_set_of_inequality_system_l70_70378


namespace initial_professors_l70_70086

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end initial_professors_l70_70086


namespace normal_line_eq_l70_70408

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem normal_line_eq (x_0 : ℝ) (h : x_0 = 1) :
  ∃ y_0 : ℝ, y_0 = f x_0 ∧ 
  ∀ x y : ℝ, y = -(x - 1) + y_0 ↔ f 1 = 0 ∧ y = -x + 1 :=
by
  sorry

end normal_line_eq_l70_70408


namespace solve_real_triples_l70_70470

theorem solve_real_triples (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b) ∧
   b * (c^2 + a) = a * (a + b * c) ∧
   c * (a^2 + b) = b * (b + c * a)) ↔ 
  (∃ (x : ℝ), (a = x) ∧ (b = x) ∧ (c = x)) ∨ 
  (b = 0 ∧ c = 0) :=
sorry

end solve_real_triples_l70_70470


namespace women_attended_l70_70896

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end women_attended_l70_70896


namespace prob_first_two_same_color_expected_value_eta_l70_70731

-- Definitions and conditions
def num_white : ℕ := 4
def num_black : ℕ := 3
def total_pieces : ℕ := num_white + num_black

-- Probability of drawing two pieces of the same color
def prob_same_color : ℚ :=
  (4/7 * 3/6) + (3/7 * 2/6)

-- Expected value of the number of white pieces drawn in the first four draws
def E_eta : ℚ :=
  1 * (4 / 35) + 2 * (18 / 35) + 3 * (12 / 35) + 4 * (1 / 35)

-- Proof statements
theorem prob_first_two_same_color : prob_same_color = 3 / 7 :=
  by sorry

theorem expected_value_eta : E_eta = 16 / 7 :=
  by sorry

end prob_first_two_same_color_expected_value_eta_l70_70731


namespace james_prom_total_cost_l70_70405

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end james_prom_total_cost_l70_70405


namespace necessarily_positive_l70_70348

-- Conditions
variables {x y z : ℝ}

-- Statement to prove
theorem necessarily_positive (h1 : 0 < x) (h2 : x < 1) (h3 : -2 < y) (h4 : y < 0) (h5 : 2 < z) (h6 : z < 3) :
  0 < y + 2 * z :=
sorry

end necessarily_positive_l70_70348


namespace number_of_students_l70_70384

/--
Statement: Several students are seated around a circular table. 
Each person takes one piece from a bag containing 120 pieces of candy 
before passing it to the next. Chris starts with the bag, takes one piece 
and also ends up with the last piece. Prove that the number of students
at the table could be 7 or 17.
-/
theorem number_of_students (n : Nat) (h : 120 > 0) :
  (∃ k, 119 = k * n ∧ n ≥ 1) → (n = 7 ∨ n = 17) :=
by
  sorry

end number_of_students_l70_70384


namespace certain_number_eq_40_l70_70157

theorem certain_number_eq_40 (x : ℝ) 
    (h : (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5) : x = 40 := 
by
  sorry

end certain_number_eq_40_l70_70157


namespace diagram_is_knowledge_structure_l70_70598

inductive DiagramType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

axiom given_diagram : DiagramType
axiom diagram_is_one_of_them : 
  given_diagram = DiagramType.ProgramFlowchart ∨ 
  given_diagram = DiagramType.ProcessFlowchart ∨ 
  given_diagram = DiagramType.KnowledgeStructureDiagram ∨ 
  given_diagram = DiagramType.OrganizationalStructureDiagram

theorem diagram_is_knowledge_structure :
  given_diagram = DiagramType.KnowledgeStructureDiagram :=
sorry

end diagram_is_knowledge_structure_l70_70598


namespace original_triangle_area_quadrupled_l70_70793

theorem original_triangle_area_quadrupled {A : ℝ} (h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64)) : A = 4 :=
by
  have h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64) := by
    intro a ha
    sorry
  sorry

end original_triangle_area_quadrupled_l70_70793


namespace gas_station_constant_l70_70746

structure GasStationData where
  amount : ℝ
  unit_price : ℝ
  price_per_yuan_per_liter : ℝ

theorem gas_station_constant (data : GasStationData) (h1 : data.amount = 116.64) (h2 : data.unit_price = 18) (h3 : data.price_per_yuan_per_liter = 6.48) : data.unit_price = 18 :=
sorry

end gas_station_constant_l70_70746


namespace value_of_x_l70_70275

theorem value_of_x (x : ℝ) : abs (4 * x - 8) ≤ 0 ↔ x = 2 :=
by {
  sorry
}

end value_of_x_l70_70275


namespace Harriet_siblings_product_l70_70527

variable (Harry_sisters : Nat)
variable (Harry_brothers : Nat)
variable (Harriet_sisters : Nat)
variable (Harriet_brothers : Nat)

theorem Harriet_siblings_product:
  Harry_sisters = 4 -> 
  Harry_brothers = 6 ->
  Harriet_sisters = Harry_sisters -> 
  Harriet_brothers = Harry_brothers ->
  Harriet_sisters * Harriet_brothers = 24 :=
by
  intro hs hb hhs hhb
  rw [hhs, hhb]
  sorry

end Harriet_siblings_product_l70_70527


namespace friends_reach_destinations_l70_70941

noncomputable def travel_times (d : ℕ) := 
  let walking_speed := 6
  let cycling_speed := 18
  let meet_time := d / (walking_speed + cycling_speed)
  let remaining_time := d / cycling_speed
  let total_time_A := meet_time + (d - cycling_speed * meet_time) / walking_speed
  let total_time_B := (cycling_speed * meet_time) / walking_speed + (d - cycling_speed * meet_time) / walking_speed
  let total_time_C := remaining_time + meet_time
  (total_time_A, total_time_B, total_time_C)

theorem friends_reach_destinations (d : ℕ) (d_eq_24 : d = 24) : 
  let (total_time_A, total_time_B, total_time_C) := travel_times d
  total_time_A ≤ 160 / 60 ∧ total_time_B ≤ 160 / 60 ∧ total_time_C ≤ 160 / 60 :=
by 
  sorry

end friends_reach_destinations_l70_70941


namespace gcd_5670_9800_l70_70342

-- Define the two given numbers
def a := 5670
def b := 9800

-- State that the GCD of a and b is 70
theorem gcd_5670_9800 : Int.gcd a b = 70 := by
  sorry

end gcd_5670_9800_l70_70342


namespace inequality_problem_l70_70951

theorem inequality_problem
  (a b c d : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h_sum : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1 / 5 :=
by
  sorry

end inequality_problem_l70_70951


namespace yellow_balls_count_l70_70643

theorem yellow_balls_count (x y z : ℕ) 
  (h1 : x + y + z = 68)
  (h2 : y = 2 * x)
  (h3 : 3 * z = 4 * y) : y = 24 :=
by {
  sorry
}

end yellow_balls_count_l70_70643


namespace value_of_M_l70_70454

theorem value_of_M (x y z M : ℝ) (h1 : x + y + z = 90)
    (h2 : x - 5 = M)
    (h3 : y + 5 = M)
    (h4 : 5 * z = M) :
    M = 450 / 11 :=
by
    sorry

end value_of_M_l70_70454


namespace find_k_l70_70023

noncomputable def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - (1 / x) + 2

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → 
  k = - 134 / 3 :=
by
  sorry

end find_k_l70_70023


namespace exists_n_prime_factors_m_exp_n_plus_n_exp_m_l70_70986

theorem exists_n_prime_factors_m_exp_n_plus_n_exp_m (m k : ℕ) (hm : m > 0) (hm_odd : m % 2 = 1) (hk : k > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ primes : Finset ℕ, primes.card ≥ k ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ m ^ n + n ^ m) := 
sorry

end exists_n_prime_factors_m_exp_n_plus_n_exp_m_l70_70986


namespace triangle_base_length_l70_70694

theorem triangle_base_length (x : ℝ) :
  (∃ s : ℝ, 4 * s = 64 ∧ s * s = 256) ∧ (32 * x / 2 = 256) → x = 16 := by
  sorry

end triangle_base_length_l70_70694


namespace time_to_pass_trolley_l70_70562

/--
Conditions:
- Length of the train = 110 m
- Speed of the train = 60 km/hr
- Speed of the trolley = 12 km/hr

Prove that the time it takes for the train to pass the trolley completely is 5.5 seconds.
-/
theorem time_to_pass_trolley :
  ∀ (train_length : ℝ) (train_speed_kmh : ℝ) (trolley_speed_kmh : ℝ),
    train_length = 110 →
    train_speed_kmh = 60 →
    trolley_speed_kmh = 12 →
  train_length / ((train_speed_kmh + trolley_speed_kmh) * (1000 / 3600)) = 5.5 :=
by
  intros
  sorry

end time_to_pass_trolley_l70_70562


namespace length_of_bridge_l70_70728

noncomputable def speed_in_m_per_s (v_kmh : ℕ) : ℝ :=
  v_kmh * (1000 / 3600)

noncomputable def total_distance (v : ℝ) (t : ℝ) : ℝ :=
  v * t

theorem length_of_bridge (L_train : ℝ) (v_train_kmh : ℕ) (t : ℝ) (L_bridge : ℝ) :
  L_train = 288 →
  v_train_kmh = 29 →
  t = 48.29 →
  L_bridge = total_distance (speed_in_m_per_s v_train_kmh) t - L_train →
  L_bridge = 100.89 := by
  sorry

end length_of_bridge_l70_70728


namespace correct_expression_l70_70334

theorem correct_expression (a b : ℝ) : (a^2 * b)^3 = (a^6 * b^3) := 
by
sorry

end correct_expression_l70_70334


namespace smallest_angle_in_triangle_l70_70259

open Real

theorem smallest_angle_in_triangle
  (a b c : ℝ)
  (h : a = (b + c) / 3)
  (triangle_inequality_1 : a + b > c)
  (triangle_inequality_2 : a + c > b)
  (triangle_inequality_3 : b + c > a) :
  ∃ A B C α β γ : ℝ, -- A, B, C are the angles opposite to sides a, b, c respectively
  0 < α ∧ α < β ∧ α < γ :=
sorry

end smallest_angle_in_triangle_l70_70259


namespace max_min_values_in_region_l70_70020

-- Define the function
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D (x y : ℝ) : Prop := (0 ≤ x) ∧ (x - 2 * y ≤ 0) ∧ (x + y - 6 ≤ 0)

-- Define the proof problem
theorem max_min_values_in_region :
  (∀ (x y : ℝ), D x y → z x y ≥ 0) ∧
  (∀ (x y : ℝ), D x y → z x y ≤ 32) :=
by 
  sorry -- Proof omitted

end max_min_values_in_region_l70_70020


namespace find_a_l70_70927

theorem find_a (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) 
  (h_max : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^(2*x) + 2 * a^x - 1 ≤ 7) 
  (h_eq : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 7) : 
  a = 2 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l70_70927


namespace find_angle_C_find_side_a_l70_70171

namespace TriangleProof

-- Declare the conditions and the proof promises
variables {A B C : ℝ} {a b c S : ℝ}

-- First part: Prove angle C
theorem find_angle_C (h1 : c^2 = a^2 + b^2 - a * b) : C = 60 :=
sorry

-- Second part: Prove the value of a
theorem find_side_a (h2 : b = 2) (h3 : S = (3 * Real.sqrt 3) / 2) : a = 3 :=
sorry

end TriangleProof

end find_angle_C_find_side_a_l70_70171


namespace nobel_prize_laureates_at_workshop_l70_70772

theorem nobel_prize_laureates_at_workshop :
  ∃ (T W W_and_N N_no_W X N : ℕ), 
    T = 50 ∧ 
    W = 31 ∧ 
    W_and_N = 16 ∧ 
    (N_no_W = X + 3) ∧ 
    (T - W = 19) ∧ 
    (N_no_W + X = 19) ∧ 
    (N = W_and_N + N_no_W) ∧ 
    N = 27 :=
by
  sorry

end nobel_prize_laureates_at_workshop_l70_70772


namespace number_of_steaks_needed_l70_70329

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end number_of_steaks_needed_l70_70329


namespace compare_f_minus1_f_1_l70_70403

variable (f : ℝ → ℝ)

-- Given conditions
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x : ℝ, f x = x^2 + 2 * x * (f 2 - 2 * x))

-- Goal statement
theorem compare_f_minus1_f_1 : f (-1) > f 1 :=
by sorry

end compare_f_minus1_f_1_l70_70403


namespace geometric_seq_term_positive_l70_70604

theorem geometric_seq_term_positive :
  ∃ (b : ℝ), 81 * (b / 81) = b ∧ b * (b / 81) = (8 / 27) ∧ b > 0 ∧ b = 2 * Real.sqrt 6 :=
by 
  use 2 * Real.sqrt 6
  sorry

end geometric_seq_term_positive_l70_70604


namespace quoted_value_of_stock_l70_70323

theorem quoted_value_of_stock (F P : ℝ) (h1 : F > 0) (h2 : P = 1.25 * F) : 
  (0.10 * F) / P = 0.08 := 
sorry

end quoted_value_of_stock_l70_70323


namespace complex_arithmetic_1_complex_arithmetic_2_l70_70057

-- Proof Problem 1
theorem complex_arithmetic_1 : 
  (1 : ℂ) * (-2 - 4 * I) - (7 - 5 * I) + (1 + 7 * I) = -8 + 8 * I := 
sorry

-- Proof Problem 2
theorem complex_arithmetic_2 : 
  (1 + I) * (2 + I) + (5 + I) / (1 - I) + (1 - I) ^ 2 = 3 + 4 * I := 
sorry

end complex_arithmetic_1_complex_arithmetic_2_l70_70057


namespace intersection_line_exists_unique_l70_70574

universe u

noncomputable section

structure Point (α : Type u) :=
(x y z : α)

structure Line (α : Type u) :=
(dir point : Point α)

variables {α : Type u} [Field α]

-- Define skew lines conditions
def skew_lines (l1 l2 : Line α) : Prop :=
¬ ∃ p : Point α, ∃ t1 t2 : α, 
  l1.point = p ∧ l1.dir ≠ (Point.mk 0 0 0) ∧ l2.point = p ∧ l2.dir ≠ (Point.mk 0 0 0) ∧
  l1.dir.x * t1 = l2.dir.x * t2 ∧
  l1.dir.y * t1 = l2.dir.y * t2 ∧
  l1.dir.z * t1 = l2.dir.z * t2

-- Define a point not on the lines
def point_not_on_lines (p : Point α) (l1 l2 : Line α) : Prop :=
  (∀ t1 : α, p ≠ Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1))
  ∧
  (∀ t2 : α, p ≠ Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2))

-- Main theorem: existence and typical uniqueness of the intersection line
theorem intersection_line_exists_unique {l1 l2 : Line α} {O : Point α}
  (h_skew : skew_lines l1 l2) (h_point_not_on_lines : point_not_on_lines O l1 l2) :
  ∃! l : Line α, l.point = O ∧ (
    ∃ t1 : α, ∃ t2 : α,
    Point.mk (O.x + l.dir.x * t1) (O.y + l.dir.y * t1) (O.z + l.dir.z * t1) = 
    Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1) ∧
    Point.mk (O.x + l.dir.x * t2) (O.y + l.dir.x * t2) (O.z + l.dir.z * t2) = 
    Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2)
  ) :=
by
  sorry

end intersection_line_exists_unique_l70_70574


namespace sin_150_eq_half_l70_70547

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end sin_150_eq_half_l70_70547


namespace rectangular_plot_area_l70_70319

theorem rectangular_plot_area (P : ℝ) (L W : ℝ) (h1 : P = 24) (h2 : L = 2 * W) :
    A = 32 := by
  sorry

end rectangular_plot_area_l70_70319


namespace remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l70_70029

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def sum_first_seven_primes : ℕ := first_seven_primes.sum

def eighth_prime : ℕ := 19

theorem remainder_when_multiplied_by_three_and_divided_by_eighth_prime :
  ((sum_first_seven_primes * 3) % eighth_prime = 3) :=
by
  sorry

end remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l70_70029


namespace smallest_d_for_inequality_l70_70264

open Real

theorem smallest_d_for_inequality :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + 1 * |x^2 - y^2| ≥ exp ((x + y) / 2)) ∧
  (∀ d > 0, (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + d * |x^2 - y^2| ≥ exp ((x + y) / 2)) → d ≥ 1) :=
by
  sorry

end smallest_d_for_inequality_l70_70264


namespace water_difference_l70_70564

variables (S H : ℝ)

theorem water_difference 
  (h_diff_after : S - 0.43 - (H + 0.43) = 0.88)
  (h_seungmin_more : S > H) :
  S - H = 1.74 :=
by
  sorry

end water_difference_l70_70564


namespace find_other_endpoint_of_diameter_l70_70880

theorem find_other_endpoint_of_diameter 
    (center endpoint : ℝ × ℝ) 
    (h_center : center = (5, -2)) 
    (h_endpoint : endpoint = (2, 3))
    : (center.1 + (center.1 - endpoint.1), center.2 + (center.2 - endpoint.2)) = (8, -7) := 
by
  sorry

end find_other_endpoint_of_diameter_l70_70880


namespace minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l70_70061

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions based on the problem statements
axiom a1_neg : a 1 < 0
axiom S2015_neg : S 2015 < 0
axiom S2016_pos : S 2016 > 0

-- Defining n value where S_n reaches its minimum
def n_min := 1008

theorem minimum_S_n_at_1008 : S n_min = S 1008 := sorry

-- Additional theorems to satisfy the provided conditions
theorem a1008_neg : a 1008 < 0 := sorry
theorem a1009_pos : a 1009 > 0 := sorry
theorem common_difference_pos : ∀ n : ℕ, a (n + 1) - a n > 0 := sorry

end minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l70_70061


namespace train_average_speed_l70_70933

open Real -- Assuming all required real number operations 

noncomputable def average_speed (distances : List ℝ) (times : List ℝ) : ℝ := 
  let total_distance := distances.sum
  let total_time := times.sum
  total_distance / total_time

theorem train_average_speed :
  average_speed [125, 270] [2.5, 3] = 71.82 := 
by 
  -- Details of the actual proof steps are omitted
  sorry

end train_average_speed_l70_70933


namespace smallest_vertical_distance_between_graphs_l70_70943

noncomputable def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem smallest_vertical_distance_between_graphs :
  ∃ (d : ℝ), (∀ (x : ℝ), |f x - g x| ≥ d) ∧ (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), |f x - g x| < d + ε) ∧ d = 3 / 4 :=
by
  sorry

end smallest_vertical_distance_between_graphs_l70_70943


namespace indeterminate_4wheelers_l70_70373

-- Define conditions and the main theorem to state that the number of 4-wheelers cannot be uniquely determined.
theorem indeterminate_4wheelers (x y : ℕ) (h : 2 * x + 4 * y = 58) : ∃ k : ℤ, y = ((29 : ℤ) - k - x) / 2 :=
by
  sorry

end indeterminate_4wheelers_l70_70373


namespace simplify_polynomial_l70_70991

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  x * (4 * x^2 - 2) - 5 * (x^2 - 3 * x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 :=
by
  sorry

end simplify_polynomial_l70_70991


namespace jake_present_weight_l70_70873

theorem jake_present_weight (J S B : ℝ) (h1 : J - 20 = 2 * S) (h2 : B = 0.5 * J) (h3 : J + S + B = 330) :
  J = 170 :=
by sorry

end jake_present_weight_l70_70873


namespace estimate_total_fish_l70_70614

theorem estimate_total_fish (m n k : ℕ) (hk : k ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0):
  ∃ x : ℕ, x = (m * n) / k :=
by
  sorry

end estimate_total_fish_l70_70614


namespace heather_distance_l70_70640

-- Definitions based on conditions
def distance_from_car_to_entrance (x : ℝ) : ℝ := x
def distance_from_entrance_to_rides (x : ℝ) : ℝ := x
def distance_from_rides_to_car : ℝ := 0.08333333333333333
def total_distance_walked : ℝ := 0.75

-- Lean statement to prove
theorem heather_distance (x : ℝ) (h : distance_from_car_to_entrance x + distance_from_entrance_to_rides x + distance_from_rides_to_car = total_distance_walked) :
  x = 0.33333333333333335 :=
by
  sorry

end heather_distance_l70_70640


namespace unique_wxyz_solution_l70_70612

theorem unique_wxyz_solution (w x y z : ℕ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : w.factorial = x.factorial + y.factorial + z.factorial) : (w, x, y, z) = (3, 2, 2, 2) :=
by
  sorry

end unique_wxyz_solution_l70_70612


namespace sequence_starting_point_l70_70632

theorem sequence_starting_point
  (n : ℕ) 
  (k : ℕ) 
  (h₁ : n * 9 ≤ 100000)
  (h₂ : k = 11110)
  (h₃ : 9 * (n + k - 1) = 99999) : 
  9 * n = 88890 :=
by 
  sorry

end sequence_starting_point_l70_70632


namespace sqrt_expression_l70_70159

theorem sqrt_expression (y : ℝ) (hy : y < 0) : 
  Real.sqrt (y / (1 - ((y - 2) / y))) = -y / Real.sqrt 2 := 
sorry

end sqrt_expression_l70_70159


namespace rectangle_area_increase_l70_70293

theorem rectangle_area_increase (x y : ℕ) 
  (hxy : x * y = 180) 
  (hperimeter : 2 * x + 2 * y = 54) : 
  (x + 6) * (y + 6) = 378 :=
by sorry

end rectangle_area_increase_l70_70293


namespace sum_of_coefficients_l70_70511

theorem sum_of_coefficients (a b c d : ℤ)
  (h1 : a + c = 2)
  (h2 : a * c + b + d = -3)
  (h3 : a * d + b * c = 7)
  (h4 : b * d = -6) :
  a + b + c + d = 7 :=
sorry

end sum_of_coefficients_l70_70511


namespace smallest_solution_floor_eq_l70_70073

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l70_70073


namespace friend_saves_per_week_l70_70385

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end friend_saves_per_week_l70_70385


namespace circumradius_of_right_triangle_l70_70273

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) : 
  ∃ R : ℝ, R = 8.5 :=
by
  sorry

end circumradius_of_right_triangle_l70_70273


namespace range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l70_70697

-- Definition of the sets A and B
def A (a : ℝ) (x : ℝ) : Prop := a - 1 < x ∧ x < 2 * a + 1
def B (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Proving range of a for A ∩ B = ∅
theorem range_a_A_intersect_B_empty (a : ℝ) :
  (¬ ∃ x : ℝ, A a x ∧ B x) ↔ (a ≤ -2 ∨ a ≥ 2 ∨ (-2 < a ∧ a ≤ -1/2)) := sorry

-- Proving range of a for A ∪ B = B
theorem range_a_A_union_B_eq_B (a : ℝ) :
  (∀ x : ℝ, A a x ∨ B x → B x) ↔ (a ≤ -2) := sorry

end range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l70_70697


namespace area_of_triangle_is_11_25_l70_70080

noncomputable def area_of_triangle : ℝ :=
  let A := (1 / 2, 2)
  let B := (8, 2)
  let C := (2, 5)
  let base := (B.1 - A.1 : ℝ)
  let height := (C.2 - A.2 : ℝ)
  0.5 * base * height

theorem area_of_triangle_is_11_25 :
  area_of_triangle = 11.25 := sorry

end area_of_triangle_is_11_25_l70_70080


namespace grasshopper_jump_distance_l70_70861

-- Definitions based on conditions
def frog_jump : ℤ := 39
def higher_jump_distance : ℤ := 22
def grasshopper_jump : ℤ := frog_jump - higher_jump_distance

-- The statement we need to prove
theorem grasshopper_jump_distance :
  grasshopper_jump = 17 :=
by
  -- Here, proof would be provided but we skip with sorry
  sorry

end grasshopper_jump_distance_l70_70861


namespace wake_up_time_l70_70736

-- Definition of the conversion ratio from normal minutes to metric minutes
def conversion_ratio := 36 / 25

-- Definition of normal minutes in a full day
def normal_minutes_in_day := 24 * 60

-- Definition of metric minutes in a full day
def metric_minutes_in_day := 10 * 100

-- Definition to convert normal time (6:36 AM) to normal minutes
def normal_minutes_from_midnight (h m : ℕ) := h * 60 + m

-- Converting normal minutes to metric minutes using the conversion ratio
def metric_minutes (normal_mins : ℕ) := (normal_mins / 36) * 25

-- Definition of the final metric time 2:75
def metric_time := (2 * 100 + 75)

-- Proving the final answer is 275
theorem wake_up_time : 100 * 2 + 10 * 7 + 5 = 275 := by
  sorry

end wake_up_time_l70_70736


namespace max_value_of_expression_l70_70500

theorem max_value_of_expression (a b c : ℝ) (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) 
    (h_sum: a + b + c = 3) :
    (ab / (a + b) + ac / (a + c) + bc / (b + c) ≤ 3 / 2) :=
by
  sorry

end max_value_of_expression_l70_70500


namespace min_value_of_x_plus_y_l70_70406

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l70_70406


namespace problem_l70_70934

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + Real.sin x) / (x ^ 2 + 1)

noncomputable def f' (x : ℝ) : ℝ := ((2 + Real.cos x) * (x ^ 2 + 1) - (2 * x + Real.sin x) * (2 * x)) / (x ^ 2 + 1) ^ 2

theorem problem : f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end problem_l70_70934


namespace student_correct_answers_l70_70535

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 70) : C = 90 :=
sorry

end student_correct_answers_l70_70535


namespace can_adjust_to_357_l70_70055

structure Ratio (L O V : ℕ) :=
(lemon : ℕ)
(oil : ℕ)
(vinegar : ℕ)

def MixA : Ratio 1 2 3 := ⟨1, 2, 3⟩
def MixB : Ratio 3 4 5 := ⟨3, 4, 5⟩
def TargetC : Ratio 3 5 7 := ⟨3, 5, 7⟩

theorem can_adjust_to_357 (x y : ℕ) (hA : x * MixA.lemon + y * MixB.lemon = 3 * (x + y))
    (hO : x * MixA.oil + y * MixB.oil = 5 * (x + y))
    (hV : x * MixA.vinegar + y * MixB.vinegar = 7 * (x + y)) :
    (∃ a b : ℕ, x = 3 * a ∧ y = 2 * b) :=
sorry

end can_adjust_to_357_l70_70055


namespace solve_for_x_l70_70209

theorem solve_for_x : ∀ (x : ℝ), 
  (x + 2 * x + 3 * x + 4 * x = 5) → (x = 1 / 2) :=
by 
  intros x H
  sorry

end solve_for_x_l70_70209


namespace domain_of_v_l70_70352

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (Real.cos x)

theorem domain_of_v :
  (∀ x : ℝ, (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < x ∧ x < 2 * n * Real.pi + Real.pi / 2) ↔ 
    ∀ x : ℝ, ∀ x_in_domain : ℝ, (0 < Real.cos x ∧ 1 / Real.sqrt (Real.cos x) = x_in_domain)) :=
sorry

end domain_of_v_l70_70352


namespace original_salary_l70_70852

def final_salary_after_changes (S : ℝ) : ℝ :=
  let increased_10 := S * 1.10
  let promoted_8 := increased_10 * 1.08
  let deducted_5 := promoted_8 * 0.95
  let decreased_7 := deducted_5 * 0.93
  decreased_7

theorem original_salary (S : ℝ) (h : final_salary_after_changes S = 6270) : S = 5587.68 :=
by
  -- Proof to be completed here
  sorry

end original_salary_l70_70852


namespace landscape_length_l70_70739

theorem landscape_length (b l : ℕ) (playground_area : ℕ) (total_area : ℕ) 
  (h1 : l = 4 * b) (h2 : playground_area = 1200) (h3 : total_area = 3 * playground_area) (h4 : total_area = l * b) :
  l = 120 := 
by 
  sorry

end landscape_length_l70_70739


namespace square_root_and_quadratic_solution_l70_70686

theorem square_root_and_quadratic_solution
  (a b : ℤ)
  (h1 : 2 * a + b = 0)
  (h2 : 3 * b + 12 = 0) :
  (2 * a - 3 * b = 16) ∧ (a * x^2 + 4 * b - 2 = 0 → x^2 = 9) :=
by {
  -- Placeholder for proof
  sorry
}

end square_root_and_quadratic_solution_l70_70686


namespace cost_per_use_l70_70281

def cost : ℕ := 30
def uses_in_a_week : ℕ := 3
def weeks : ℕ := 2
def total_uses : ℕ := uses_in_a_week * weeks

theorem cost_per_use : cost / total_uses = 5 := by
  sorry

end cost_per_use_l70_70281


namespace ellipse_problem_part1_ellipse_problem_part2_l70_70769

-- Statement of the problem
theorem ellipse_problem_part1 :
  ∃ k : ℝ, (∀ x y : ℝ, (x^2 / 2) + y^2 = 1 → (
    (∃ t > 0, x = t * y + 1) → k = (Real.sqrt 2) / 2)) :=
sorry

theorem ellipse_problem_part2 :
  ∃ S_max : ℝ, ∀ (t : ℝ), (t > 0 → (S_max = (4 * (t^2 + 1)^2) / ((t^2 + 2) * (2 * t^2 + 1)))) → t^2 = 1 → S_max = 16 / 9 :=
sorry

end ellipse_problem_part1_ellipse_problem_part2_l70_70769


namespace flowchart_correct_option_l70_70302

-- Definitions based on conditions
def typical_flowchart (start_points end_points : ℕ) : Prop :=
  start_points = 1 ∧ end_points ≥ 1

-- Theorem to prove
theorem flowchart_correct_option :
  ∃ (start_points end_points : ℕ), typical_flowchart start_points end_points ∧ "Option C" = "Option C" :=
by {
  sorry -- This part skips the proof itself,
}

end flowchart_correct_option_l70_70302


namespace cos_product_equals_one_over_128_l70_70623

theorem cos_product_equals_one_over_128 :
  (Real.cos (Real.pi / 15)) *
  (Real.cos (2 * Real.pi / 15)) *
  (Real.cos (3 * Real.pi / 15)) *
  (Real.cos (4 * Real.pi / 15)) *
  (Real.cos (5 * Real.pi / 15)) *
  (Real.cos (6 * Real.pi / 15)) *
  (Real.cos (7 * Real.pi / 15))
  = 1 / 128 := 
sorry

end cos_product_equals_one_over_128_l70_70623


namespace necessary_but_not_sufficient_condition_l70_70034

-- Definitions of conditions
def condition_p (x : ℝ) := (x - 1) * (x + 2) ≤ 0
def condition_q (x : ℝ) := abs (x + 1) ≤ 1

-- The theorem statement
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (∀ x, condition_q x → condition_p x) ∧ ¬(∀ x, condition_p x → condition_q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l70_70034


namespace octagon_properties_l70_70450

-- Definitions for a regular octagon inscribed in a circle
def regular_octagon (r : ℝ) := ∀ (a b : ℝ), abs (a - b) = r
def side_length := 5
def inscribed_in_circle (r : ℝ) := ∃ (a b : ℝ), a * a + b * b = r * r

-- Main theorem statement
theorem octagon_properties (r : ℝ) (h : r = side_length) (h1 : regular_octagon r) (h2 : inscribed_in_circle r) :
  let arc_length := (5 * π) / 4
  let area_sector := (25 * π) / 8
  arc_length = (5 * π) / 4 ∧ area_sector = (25 * π) / 8 := by
  sorry

end octagon_properties_l70_70450


namespace find_a_l70_70426

variable (m n a : ℝ)
variable (h1 : m = 2 * n + 5)
variable (h2 : m + a = 2 * (n + 1.5) + 5)

theorem find_a : a = 3 := by
  sorry

end find_a_l70_70426


namespace range_of_f_l70_70394

noncomputable def f (x : Real) : Real :=
  if x ≤ 1 then 2 * x + 1 else Real.log x + 1

theorem range_of_f (x : Real) : f x + f (x + 1) > 1 ↔ (x > -(3 / 4)) :=
  sorry

end range_of_f_l70_70394


namespace arithmetic_contains_geometric_l70_70409

theorem arithmetic_contains_geometric (a b : ℚ) (h : a^2 + b^2 ≠ 0) :
  ∃ (q : ℚ) (c : ℚ) (n₀ : ℕ) (n : ℕ → ℕ), (∀ k : ℕ, n (k+1) = n k + c * q^k) ∧
  ∀ k : ℕ, ∃ r : ℚ, a + b * n k = r * q^k :=
sorry

end arithmetic_contains_geometric_l70_70409


namespace roots_transformation_l70_70715

noncomputable def poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 10

noncomputable def transformed_poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270

theorem roots_transformation (r₁ r₂ r₃ : ℝ) (h : poly_with_roots r₁ r₂ r₃ = 0) :
  transformed_poly_with_roots (3 * r₁) (3 * r₂) (3 * r₃) = Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270 :=
by
  sorry

end roots_transformation_l70_70715


namespace algebraic_expr_pos_int_vals_l70_70241

noncomputable def algebraic_expr_ineq (x : ℕ) : Prop :=
  x > 0 ∧ ((x + 1)/3 - (2*x - 1)/4 ≥ (x - 3)/6)

theorem algebraic_expr_pos_int_vals : {x : ℕ | algebraic_expr_ineq x} = {1, 2, 3} :=
sorry

end algebraic_expr_pos_int_vals_l70_70241


namespace additional_people_needed_l70_70131

-- Define the initial number of people and time they take to mow the lawn 
def initial_people : ℕ := 8
def initial_time : ℕ := 3

-- Define total person-hours required to mow the lawn
def total_person_hours : ℕ := initial_people * initial_time

-- Define the time in which we want to find out how many people can mow the lawn
def desired_time : ℕ := 2

-- Define the number of people needed in desired_time to mow the lawn
def required_people : ℕ := total_person_hours / desired_time

-- Define the additional people required to mow the lawn in desired_time
def additional_people : ℕ := required_people - initial_people

-- Statement to be proved
theorem additional_people_needed : additional_people = 4 := by
  -- Proof to be filled in
  sorry

end additional_people_needed_l70_70131


namespace cory_initial_money_l70_70341

variable (cost_per_pack : ℝ) (packs : ℕ) (additional_needed : ℝ) (total_cost : ℝ) (initial_money : ℝ)

-- Conditions
def cost_per_pack_def : Prop := cost_per_pack = 49
def packs_def : Prop := packs = 2
def additional_needed_def : Prop := additional_needed = 78
def total_cost_def : Prop := total_cost = packs * cost_per_pack
def initial_money_def : Prop := initial_money = total_cost - additional_needed

-- Theorem
theorem cory_initial_money : cost_per_pack = 49 ∧ packs = 2 ∧ additional_needed = 78 → initial_money = 20 := by
  intro h
  have h1 : cost_per_pack = 49 := h.1
  have h2 : packs = 2 := h.2.1
  have h3 : additional_needed = 78 := h.2.2
  -- sorry
  sorry

end cory_initial_money_l70_70341


namespace total_onions_l70_70200

theorem total_onions (sara sally fred amy matthew : Nat) 
  (hs : sara = 40) (hl : sally = 55) 
  (hf : fred = 90) (ha : amy = 25) 
  (hm : matthew = 75) :
  sara + sally + fred + amy + matthew = 285 := 
by
  sorry

end total_onions_l70_70200


namespace ethan_hours_per_day_l70_70244

-- Define the known constants
def hourly_wage : ℝ := 18
def work_days_per_week : ℕ := 5
def total_earnings : ℝ := 3600
def weeks_worked : ℕ := 5

-- Define the main theorem
theorem ethan_hours_per_day :
  (∃ hours_per_day : ℝ, 
    hours_per_day = total_earnings / (weeks_worked * work_days_per_week * hourly_wage)) →
  hours_per_day = 8 :=
by
  sorry

end ethan_hours_per_day_l70_70244


namespace abs_inequality_l70_70822

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end abs_inequality_l70_70822


namespace unique_digit_for_prime_l70_70452

theorem unique_digit_for_prime (B : ℕ) (hB : B < 10) (hprime : Nat.Prime (30420 * 10 + B)) : B = 1 :=
sorry

end unique_digit_for_prime_l70_70452


namespace problem1_problem2_l70_70926

variable (a b : ℝ)

-- Proof problem for Question 1
theorem problem1 : 2 * a * (a^2 - 3 * a - 1) = 2 * a^3 - 6 * a^2 - 2 * a :=
by sorry

-- Proof problem for Question 2
theorem problem2 : (a^2 * b - 2 * a * b^2 + b^3) / b - (a + b)^2 = -4 * a * b :=
by sorry

end problem1_problem2_l70_70926


namespace tan_alpha_cos2alpha_plus_2sin2alpha_l70_70184

theorem tan_alpha_cos2alpha_plus_2sin2alpha (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end tan_alpha_cos2alpha_plus_2sin2alpha_l70_70184


namespace trigonometric_relationship_l70_70109

theorem trigonometric_relationship :
  let a := [10, 9, 8, 7, 6, 4, 3, 2, 1]
  let sum_of_a := a.sum
  let x := Real.sin sum_of_a
  let y := Real.cos sum_of_a
  let z := Real.tan sum_of_a
  sum_of_a = 50 →
  z < x ∧ x < y :=
by
  sorry

end trigonometric_relationship_l70_70109


namespace intersection_equal_l70_70317

-- Define the sets M and N based on given conditions
def M : Set ℝ := {x : ℝ | x^2 - 3 * x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Define the intersection of M and N
def intersection : Set ℝ := {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)}

-- The statement to be proved
theorem intersection_equal : M ∩ N = intersection :=
by 
  sorry -- Skipping the proof

end intersection_equal_l70_70317


namespace dice_probability_exactly_four_twos_l70_70481

theorem dice_probability_exactly_four_twos :
  let probability := (Nat.choose 8 4 : ℚ) * (1 / 8)^4 * (7 / 8)^4 
  probability = 168070 / 16777216 :=
by
  sorry

end dice_probability_exactly_four_twos_l70_70481


namespace fraction_of_visitors_l70_70036

variable (V E U : ℕ)
variable (H1 : E = U)
variable (H2 : 600 - E - 150 = 450)

theorem fraction_of_visitors (H3 : 600 = E + 150 + 450) : (450 : ℚ) / 600 = (3 : ℚ) / 4 :=
by
  apply sorry

end fraction_of_visitors_l70_70036


namespace find_x_such_that_custom_op_neg3_eq_one_l70_70006

def custom_op (x y : Int) : Int := x * y - 2 * (x + y)

theorem find_x_such_that_custom_op_neg3_eq_one :
  ∃ x : Int, custom_op x (-3) = 1 ∧ x = 1 :=
by
  use 1
  sorry

end find_x_such_that_custom_op_neg3_eq_one_l70_70006


namespace degrees_for_salaries_l70_70855

def transportation_percent : ℕ := 15
def research_development_percent : ℕ := 9
def utilities_percent : ℕ := 5
def equipment_percent : ℕ := 4
def supplies_percent : ℕ := 2
def total_percent : ℕ := 100
def total_degrees : ℕ := 360

theorem degrees_for_salaries :
  total_degrees * (total_percent - (transportation_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent)) / total_percent = 234 := 
by
  sorry

end degrees_for_salaries_l70_70855


namespace max_popsicles_with_10_dollars_l70_70391

def price (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 3 then 2
  else if n = 5 then 3
  else if n = 7 then 4
  else 0

theorem max_popsicles_with_10_dollars : ∀ (a b c d : ℕ),
  a * price 1 + b * price 3 + c * price 5 + d * price 7 = 10 →
  a + 3 * b + 5 * c + 7 * d ≤ 17 :=
sorry

end max_popsicles_with_10_dollars_l70_70391


namespace remainder_of_polynomial_l70_70227

   def polynomial_division_remainder (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

   theorem remainder_of_polynomial : polynomial_division_remainder 1 = 4 :=
   by
     -- This placeholder indicates that the proof is omitted.
     sorry
   
end remainder_of_polynomial_l70_70227


namespace least_number_of_pairs_l70_70305

theorem least_number_of_pairs :
  let students := 100
  let messages_per_student := 50
  ∃ (pairs_of_students : ℕ), pairs_of_students = 50 := sorry

end least_number_of_pairs_l70_70305


namespace solve_system_of_equations_l70_70994

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 6.751 * x + 3.249 * y = 26.751) 
  (h2 : 3.249 * x + 6.751 * y = 23.249) : 
  x = 3 ∧ y = 2 := 
sorry

end solve_system_of_equations_l70_70994


namespace abes_total_budget_l70_70684

theorem abes_total_budget
    (B : ℝ)
    (h1 : B = (1/3) * B + (1/4) * B + 1250) :
    B = 3000 :=
sorry

end abes_total_budget_l70_70684


namespace find_k_of_quadratic_eq_ratio_3_to_1_l70_70856

theorem find_k_of_quadratic_eq_ratio_3_to_1 (k : ℝ) :
  (∃ (x : ℝ), x ≠ 0 ∧ (x^2 + 8 * x + k = 0) ∧
              (∃ (r : ℝ), x = 3 * r ∧ 3 * r + r = -8)) → k = 12 :=
by {
  sorry
}

end find_k_of_quadratic_eq_ratio_3_to_1_l70_70856


namespace probability_hare_killed_l70_70719

theorem probability_hare_killed (P_hit_1 P_hit_2 P_hit_3 : ℝ)
  (h1 : P_hit_1 = 3 / 5) (h2 : P_hit_2 = 3 / 10) (h3 : P_hit_3 = 1 / 10) :
  (1 - ((1 - P_hit_1) * (1 - P_hit_2) * (1 - P_hit_3))) = 0.748 :=
by
  sorry

end probability_hare_killed_l70_70719


namespace remainder_when_divided_by_6_l70_70333

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l70_70333


namespace ratio_of_supply_to_demand_l70_70960

def supply : ℕ := 1800000
def demand : ℕ := 2400000

theorem ratio_of_supply_to_demand : (supply / demand : ℚ) = 3 / 4 := by
  sorry

end ratio_of_supply_to_demand_l70_70960


namespace fraction_weevils_25_percent_l70_70764

-- Define the probabilities
def prob_good_milk : ℝ := 0.8
def prob_good_egg : ℝ := 0.4
def prob_all_good : ℝ := 0.24

-- The problem definition and statement
def fraction_weevils (F : ℝ) : Prop :=
  0.32 * (1 - F) = 0.24

theorem fraction_weevils_25_percent : fraction_weevils 0.25 :=
by sorry

end fraction_weevils_25_percent_l70_70764


namespace value_range_neg_x_squared_l70_70658

theorem value_range_neg_x_squared:
  (∀ y, (-9 ≤ y ∧ y ≤ 0) ↔ ∃ x, (-3 ≤ x ∧ x ≤ 1) ∧ y = -x^2) :=
by
  sorry

end value_range_neg_x_squared_l70_70658


namespace wrapping_paper_area_correct_l70_70533

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end wrapping_paper_area_correct_l70_70533


namespace scientific_notation_of_0_0000023_l70_70483

theorem scientific_notation_of_0_0000023 : 
  0.0000023 = 2.3 * 10 ^ (-6) :=
by
  sorry

end scientific_notation_of_0_0000023_l70_70483


namespace contradiction_method_l70_70104

theorem contradiction_method (x y : ℝ) (h : x + y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end contradiction_method_l70_70104


namespace value_of_x_plus_y_div_y_l70_70495

variable (w x y : ℝ)
variable (hx : w / x = 1 / 6)
variable (hy : w / y = 1 / 5)

theorem value_of_x_plus_y_div_y : (x + y) / y = 11 / 5 :=
by
  sorry

end value_of_x_plus_y_div_y_l70_70495


namespace solve_inequality_l70_70580

theorem solve_inequality (x : ℝ) : x > 13 ↔ x^3 - 16 * x^2 + 73 * x > 84 :=
by
  sorry

end solve_inequality_l70_70580


namespace distinct_parenthesizations_of_3_3_3_3_l70_70977

theorem distinct_parenthesizations_of_3_3_3_3 : 
  ∃ (v1 v2 v3 v4 v5 : ℕ), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5 ∧ 
    v1 = 3 ^ (3 ^ (3 ^ 3)) ∧ 
    v2 = 3 ^ ((3 ^ 3) ^ 3) ∧ 
    v3 = (3 ^ 3) ^ (3 ^ 3) ∧ 
    v4 = ((3 ^ 3) ^ 3) ^ 3 ∧ 
    v5 = 3 ^ (27 ^ 27) :=
  sorry

end distinct_parenthesizations_of_3_3_3_3_l70_70977


namespace average_speed_l70_70166

-- Define the problem conditions and provide the proof statement
theorem average_speed (D : ℝ) (hD0 : D > 0) : 
  let speed_1 := 80
  let speed_2 := 24
  let speed_3 := 60
  let time_1 := (D / 3) / speed_1
  let time_2 := (D / 3) / speed_2
  let time_3 := (D / 3) / speed_3
  let total_time := time_1 + time_2 + time_3
  let average_speed := D / total_time
  average_speed = 720 / 17 := 
by
  sorry

end average_speed_l70_70166


namespace journey_time_equality_l70_70834

variables {v : ℝ} (h : v > 0)

theorem journey_time_equality (v : ℝ) (hv : v > 0) :
  let t1 := 80 / v
  let t2 := 160 / (2 * v)
  t1 = t2 :=
by
  sorry

end journey_time_equality_l70_70834


namespace Cody_reads_books_in_7_weeks_l70_70804

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end Cody_reads_books_in_7_weeks_l70_70804


namespace solve_inequality_system_l70_70870

theorem solve_inequality_system : 
  (∀ x : ℝ, (1 / 3 * x - 1 ≤ 1 / 2 * x + 1) ∧ ((3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3)) → (2 ≤ x ∧ x < 4)) := 
by
  intro x h
  sorry

end solve_inequality_system_l70_70870


namespace triangle_side_c_l70_70585

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the respective angles

-- Conditions given
variable (h1 : Real.tan A = 2 * Real.tan B)
variable (h2 : a^2 - b^2 = (1 / 3) * c)

-- The proof problem
theorem triangle_side_c (h1 : Real.tan A = 2 * Real.tan B) (h2 : a^2 - b^2 = (1 / 3) * c) : c = 1 :=
by sorry

end triangle_side_c_l70_70585


namespace ball_cost_l70_70826

theorem ball_cost (B C : ℝ) (h1 : 7 * B + 6 * C = 3800) (h2 : 3 * B + 5 * C = 1750) (hb : B = 500) : C = 50 :=
by
  sorry

end ball_cost_l70_70826


namespace scissors_total_l70_70425

theorem scissors_total (original_scissors : ℕ) (added_scissors : ℕ) (total_scissors : ℕ) 
  (h1 : original_scissors = 39)
  (h2 : added_scissors = 13)
  (h3 : total_scissors = original_scissors + added_scissors) : total_scissors = 52 :=
by
  rw [h1, h2] at h3
  exact h3

end scissors_total_l70_70425


namespace sum_of_cubes_eq_96_over_7_l70_70753

-- Define the conditions from the problem
variables (a r : ℝ)
axiom condition_sum : a / (1 - r) = 2
axiom condition_sum_squares : a^2 / (1 - r^2) = 6

-- Define the correct answer that we expect to prove
theorem sum_of_cubes_eq_96_over_7 :
  a^3 / (1 - r^3) = 96 / 7 :=
sorry

end sum_of_cubes_eq_96_over_7_l70_70753


namespace range_of_m_l70_70355

open Set

noncomputable def M (m : ℝ) : Set ℝ := {x | x ≤ m}
noncomputable def N : Set ℝ := {y | y ≥ 1}

theorem range_of_m (m : ℝ) : M m ∩ N = ∅ → m < 1 := by
  intros h
  sorry

end range_of_m_l70_70355


namespace number_of_sheep_l70_70565

theorem number_of_sheep (S H : ℕ)
  (h1 : S / H = 4 / 7)
  (h2 : H * 230 = 12880) :
  S = 32 :=
by
  sorry

end number_of_sheep_l70_70565


namespace two_digit_number_ratio_l70_70206

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b
def swapped_two_digit_number (a b : ℕ) : ℕ := 10 * b + a

theorem two_digit_number_ratio (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) (h_ratio : 6 * two_digit_number a b = 5 * swapped_two_digit_number a b) : 
  two_digit_number a b = 45 :=
by
  sorry

end two_digit_number_ratio_l70_70206


namespace problem_l70_70984

theorem problem (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : f 1 = f 3) 
  (h2 : f 1 > f 4) 
  (hf : ∀ x, f x = a * x ^ 2 + b * x + c) :
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l70_70984


namespace five_b_value_l70_70424

theorem five_b_value (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 4) 
  (h2 : a = b - 3) : 
  5 * b = 65 / 7 := 
by
  sorry

end five_b_value_l70_70424


namespace plane_stops_at_20_seconds_l70_70308

/-- The analytical expression of the function of the distance s the plane travels during taxiing 
after landing with respect to the time t is given by s = -1.5t^2 + 60t. 

Prove that the plane stops after taxiing for 20 seconds. -/

noncomputable def plane_distance (t : ℝ) : ℝ :=
  -1.5 * t^2 + 60 * t

theorem plane_stops_at_20_seconds :
  ∃ t : ℝ, t = 20 ∧ plane_distance t = plane_distance (20 : ℝ) :=
by
  sorry

end plane_stops_at_20_seconds_l70_70308


namespace cat_food_insufficient_l70_70998

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l70_70998


namespace negation_of_symmetry_about_y_eq_x_l70_70120

theorem negation_of_symmetry_about_y_eq_x :
  ¬ (∀ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x) ↔ ∃ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x :=
by sorry

end negation_of_symmetry_about_y_eq_x_l70_70120


namespace find_n_l70_70468

theorem find_n (n x y k : ℕ) (h_coprime : Nat.gcd x y = 1) (h_eq : 3^n = x^k + y^k) : n = 2 :=
sorry

end find_n_l70_70468


namespace exists_100_integers_with_distinct_pairwise_sums_l70_70238

-- Define number of integers and the constraint limit
def num_integers : ℕ := 100
def max_value : ℕ := 25000

-- Define the predicate for all pairwise sums being different
def pairwise_different_sums (as : Fin num_integers → ℕ) : Prop :=
  ∀ i j k l : Fin num_integers, i ≠ j ∧ k ≠ l → as i + as j ≠ as k + as l

-- Main theorem statement
theorem exists_100_integers_with_distinct_pairwise_sums :
  ∃ as : Fin num_integers → ℕ, (∀ i : Fin num_integers, as i > 0 ∧ as i ≤ max_value) ∧ pairwise_different_sums as :=
sorry

end exists_100_integers_with_distinct_pairwise_sums_l70_70238


namespace find_k_l70_70770

-- Definitions based on given conditions
def ellipse_equation (x y : ℝ) (k : ℝ) : Prop :=
  5 * x^2 + k * y^2 = 5

def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = 2

-- Statement of the problem
theorem find_k (k : ℝ) :
  (∀ x y, ellipse_equation x y k) →
  is_focus 0 2 →
  k = 1 :=
sorry

end find_k_l70_70770


namespace smallest_base10_integer_l70_70917

theorem smallest_base10_integer :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (2 * a + 2 = 3 * b + 3) ∧ (2 * a + 2 = 18) :=
by
  existsi 8 -- assign specific solutions to a
  existsi 5 -- assign specific solutions to b
  exact sorry -- follows from the validations done above

end smallest_base10_integer_l70_70917


namespace sequence_a_10_l70_70815

theorem sequence_a_10 (a : ℕ → ℤ) 
  (H1 : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q)
  (H2 : a 2 = -6) : 
  a 10 = -30 :=
sorry

end sequence_a_10_l70_70815


namespace quadratic_discriminant_l70_70983

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l70_70983


namespace audit_sampling_is_systematic_l70_70878

def is_systematic_sampling (population_size : Nat) (step : Nat) (initial_index : Nat) : Prop :=
  ∃ (k : Nat), ∀ (n : Nat), n ≠ 0 → initial_index + step * (n - 1) ≤ population_size

theorem audit_sampling_is_systematic :
  ∀ (population_size : Nat) (random_index : Nat),
  population_size = 50 * 50 →  -- This represents the total number of invoices (50% of a larger population segment)
  random_index < 50 →         -- Randomly selected index from the first 50 invoices
  is_systematic_sampling population_size 50 random_index := 
by
  intros
  sorry

end audit_sampling_is_systematic_l70_70878


namespace initial_amount_of_water_l70_70125

theorem initial_amount_of_water 
  (W : ℚ) 
  (h1 : W - (7/15) * W - (5/8) * (W - (7/15) * W) - (2/3) * (W - (7/15) * W - (5/8) * (W - (7/15) * W)) = 2.6) 
  : W = 39 := 
sorry

end initial_amount_of_water_l70_70125


namespace dorothy_annual_earnings_correct_l70_70670

-- Define the conditions
def dorothyEarnings (X : ℝ) : Prop :=
  X - 0.18 * X = 49200

-- Define the amount Dorothy earns a year
def dorothyAnnualEarnings : ℝ := 60000

-- State the theorem
theorem dorothy_annual_earnings_correct : dorothyEarnings dorothyAnnualEarnings :=
by
-- The proof will be inserted here
sorry

end dorothy_annual_earnings_correct_l70_70670


namespace no_square_with_odd_last_two_digits_l70_70356

def last_two_digits_odd (n : ℤ) : Prop :=
  (n % 10) % 2 = 1 ∧ ((n / 10) % 10) % 2 = 1

theorem no_square_with_odd_last_two_digits (n : ℤ) (k : ℤ) :
  (k^2 = n) → last_two_digits_odd n → False :=
by
  -- A placeholder for the proof
  sorry

end no_square_with_odd_last_two_digits_l70_70356


namespace dogs_food_consumption_l70_70268

theorem dogs_food_consumption :
  (let cups_per_meal_momo_fifi := 1.5
   let meals_per_day := 3
   let cups_per_meal_gigi := 2
   let cups_to_pounds := 3
   let daily_food_momo_fifi := cups_per_meal_momo_fifi * meals_per_day * 2
   let daily_food_gigi := cups_per_meal_gigi * meals_per_day
   daily_food_momo_fifi + daily_food_gigi) / cups_to_pounds = 5 :=
by
  sorry

end dogs_food_consumption_l70_70268


namespace washing_machines_removed_per_box_l70_70089

theorem washing_machines_removed_per_box 
  (crates : ℕ) (boxes_per_crate : ℕ) (washing_machines_per_box : ℕ) 
  (total_removed : ℕ) (total_crates : ℕ) (total_boxes_per_crate : ℕ) 
  (total_washing_machines_per_box : ℕ) 
  (h1 : crates = total_crates) (h2 : boxes_per_crate = total_boxes_per_crate) 
  (h3 : washing_machines_per_box = total_washing_machines_per_box) 
  (h4 : total_removed = 60) (h5 : total_crates = 10) 
  (h6 : total_boxes_per_crate = 6) 
  (h7 : total_washing_machines_per_box = 4):
  total_removed / (total_crates * total_boxes_per_crate) = 1 :=
by
  sorry

end washing_machines_removed_per_box_l70_70089


namespace sum_of_consecutive_even_integers_l70_70280

theorem sum_of_consecutive_even_integers (a : ℤ) (h : a + (a + 6) = 136) :
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by
  sorry

end sum_of_consecutive_even_integers_l70_70280


namespace cistern_filling_time_with_leak_l70_70169

theorem cistern_filling_time_with_leak (T : ℝ) (h1 : 1 / T - 1 / 4 = 1 / (T + 2)) : T = 4 :=
by
  sorry

end cistern_filling_time_with_leak_l70_70169


namespace find_value_of_c_l70_70687

theorem find_value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 8 > 0 ↔ x < -2 ∨ x > 4)) → c = 2 :=
by
  sorry

end find_value_of_c_l70_70687


namespace moles_of_water_l70_70902

-- Definitions related to the reaction conditions.
def HCl : Type := sorry
def NaHCO3 : Type := sorry
def NaCl : Type := sorry
def H2O : Type := sorry
def CO2 : Type := sorry

def reaction (h : HCl) (n : NaHCO3) : Nat := sorry -- Represents the balanced reaction

-- Given conditions in Lean.
axiom one_mole_HCl : HCl
axiom one_mole_NaHCO3 : NaHCO3
axiom balanced_equation : reaction one_mole_HCl one_mole_NaHCO3 = 1 -- 1 mole of water is produced

-- The theorem to prove.
theorem moles_of_water : reaction one_mole_HCl one_mole_NaHCO3 = 1 :=
by
  -- The proof would go here
  sorry

end moles_of_water_l70_70902


namespace matchstick_triangle_sides_l70_70552

theorem matchstick_triangle_sides (a b c : ℕ) :
  a + b + c = 100 ∧ max a (max b c) = 3 * min a (min b c) ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c) →
  (a = 15 ∧ b = 40 ∧ c = 45 ∨ a = 16 ∧ b = 36 ∧ c = 48) :=
by
  sorry

end matchstick_triangle_sides_l70_70552


namespace Johnson_Carter_Tie_August_l70_70938

structure MonthlyHomeRuns where
  March : Nat
  April : Nat
  May : Nat
  June : Nat
  July : Nat
  August : Nat
  September : Nat

def Johnson_runs : MonthlyHomeRuns := { March:= 2, April:= 11, May:= 15, June:= 9, July:= 7, August:= 9, September:= 0 }
def Carter_runs : MonthlyHomeRuns := { March:= 1, April:= 9, May:= 8, June:= 19, July:= 6, August:= 10, September:= 0 }

noncomputable def cumulative_runs (runs: MonthlyHomeRuns) (month: String) : Nat :=
  match month with
  | "March" => runs.March
  | "April" => runs.March + runs.April
  | "May" => runs.March + runs.April + runs.May
  | "June" => runs.March + runs.April + runs.May + runs.June
  | "July" => runs.March + runs.April + runs.May + runs.June + runs.July
  | "August" => runs.March + runs.April + runs.May + runs.June + runs.July + runs.August
  | _ => 0

theorem Johnson_Carter_Tie_August :
  cumulative_runs Johnson_runs "August" = cumulative_runs Carter_runs "August" := 
  by
  sorry

end Johnson_Carter_Tie_August_l70_70938


namespace sequence_bounded_l70_70071

open Classical

noncomputable def bounded_sequence (a : ℕ → ℝ) (M : ℝ) :=
  ∀ n : ℕ, n > 0 → a n < M

theorem sequence_bounded {a : ℕ → ℝ} (h0 : 0 ≤ a 1 ∧ a 1 ≤ 2)
  (h : ∀ n : ℕ, n > 0 → a (n + 1) = a n + (a n)^2 / n^3) :
  ∃ M : ℝ, 0 < M ∧ bounded_sequence a M :=
by
  sorry

end sequence_bounded_l70_70071


namespace equal_focal_distances_l70_70487

theorem equal_focal_distances (k : ℝ) (h₁ : k ≠ 0) (h₂ : 16 - k ≠ 0) 
  (h_hyperbola : ∀ x y, (x^2) / (16 - k) - (y^2) / k = 1)
  (h_ellipse : ∀ x y, 9 * x^2 + 25 * y^2 = 225) :
  0 < k ∧ k < 16 :=
sorry

end equal_focal_distances_l70_70487


namespace gcd_digit_bound_l70_70529

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l70_70529


namespace import_rate_for_rest_of_1997_l70_70381

theorem import_rate_for_rest_of_1997
    (import_1996: ℝ)
    (import_first_two_months_1997: ℝ)
    (excess_imports_1997: ℝ)
    (import_rate_first_two_months: ℝ)
    (expected_total_imports_1997: ℝ)
    (remaining_imports_1997: ℝ)
    (R: ℝ):
    excess_imports_1997 = 720e6 →
    expected_total_imports_1997 = import_1996 + excess_imports_1997 →
    remaining_imports_1997 = expected_total_imports_1997 - import_first_two_months_1997 →
    10 * R = remaining_imports_1997 →
    R = 180e6 :=
by
    intros h_import1996 h_import_first_two_months h_excess_imports h_import_rate_first_two_months 
           h_expected_total_imports h_remaining_imports h_equation
    sorry

end import_rate_for_rest_of_1997_l70_70381


namespace rectangle_area_l70_70709

theorem rectangle_area (x : ℝ) (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l^2 + w^2 = x^2) :
    l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l70_70709


namespace total_writing_instruments_l70_70182

theorem total_writing_instruments 
 (bags : ℕ) (compartments_per_bag : ℕ) (empty_compartments : ℕ) (one_compartment : ℕ) (remaining_compartments : ℕ) 
 (writing_instruments_per_compartment : ℕ) (writing_instruments_in_one : ℕ) : 
 bags = 16 → 
 compartments_per_bag = 6 → 
 empty_compartments = 5 → 
 one_compartment = 1 → 
 remaining_compartments = 90 →
 writing_instruments_per_compartment = 8 → 
 writing_instruments_in_one = 6 → 
 (remaining_compartments * writing_instruments_per_compartment + one_compartment * writing_instruments_in_one) = 726 := 
  by
   sorry

end total_writing_instruments_l70_70182


namespace total_money_spent_on_clothing_l70_70862

theorem total_money_spent_on_clothing (cost_shirt cost_jacket : ℝ)
  (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) :
  cost_shirt + cost_jacket = 25.31 :=
sorry

end total_money_spent_on_clothing_l70_70862


namespace symmetric_line_eq_l70_70393

theorem symmetric_line_eq (x y : ℝ) :  
  (x - 2 * y + 3 = 0) → (x + 2 * y + 3 = 0) :=
sorry

end symmetric_line_eq_l70_70393


namespace functional_equation_solution_l70_70331

theorem functional_equation_solution {
  f : ℝ → ℝ
} (h : ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2 * y * f x + (f y)^2) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = x + 1) :=
sorry

end functional_equation_solution_l70_70331


namespace union_of_A_and_B_l70_70043

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l70_70043


namespace perimeter_of_shaded_shape_l70_70925

noncomputable def shaded_perimeter (x : ℝ) : ℝ := 
  let l := 18 - 2 * x
  3 * l

theorem perimeter_of_shaded_shape (x : ℝ) (hx : x > 0) (h_sectors : 2 * x + (18 - 2 * x) = 18) : 
  shaded_perimeter x = 54 := 
by
  rw [shaded_perimeter]
  rw [← h_sectors]
  simp
  sorry

end perimeter_of_shaded_shape_l70_70925


namespace problem1_problem2_l70_70216

noncomputable def op (a b : ℝ) := 2 * a - (3 / 2) * (a + b)

theorem problem1 (x : ℝ) (h : op x 4 = 0) : x = 12 :=
by sorry

theorem problem2 (x m : ℝ) (h : op x m = op (-2) (x + 4)) (hnn : x ≥ 0) : m ≥ 14 / 3 :=
by sorry

end problem1_problem2_l70_70216


namespace mia_12th_roll_last_is_approximately_027_l70_70245

noncomputable def mia_probability_last_roll_on_12th : ℚ :=
  (5/6) ^ 10 * (1/6)

theorem mia_12th_roll_last_is_approximately_027 : 
  abs (mia_probability_last_roll_on_12th - 0.027) < 0.001 :=
sorry

end mia_12th_roll_last_is_approximately_027_l70_70245


namespace find_p_l70_70523

theorem find_p (p q : ℚ) (h1 : 5 * p + 3 * q = 10) (h2 : 3 * p + 5 * q = 20) : 
  p = -5 / 8 :=
by
  sorry

end find_p_l70_70523


namespace base8_minus_base7_base10_eq_l70_70475

-- Definitions of the two numbers in their respective bases
def n1_base8 : ℕ := 305
def n2_base7 : ℕ := 165

-- Conversion of these numbers to base 10
def n1_base10 : ℕ := 3 * 8^2 + 0 * 8^1 + 5 * 8^0
def n2_base10 : ℕ := 1 * 7^2 + 6 * 7^1 + 5 * 7^0

-- Statement of the theorem to be proven
theorem base8_minus_base7_base10_eq :
  (n1_base10 - n2_base10 = 101) :=
  by
    -- The proof would go here
    sorry

end base8_minus_base7_base10_eq_l70_70475


namespace negation_of_no_honors_students_attend_school_l70_70693

-- Definitions (conditions and question)
def honors_student (x : Type) : Prop := sorry -- The condition defining an honors student
def attends_school (x : Type) : Prop := sorry -- The condition defining a student attending the school

-- The theorem statement
theorem negation_of_no_honors_students_attend_school :
  (¬ ∃ x : Type, honors_student x ∧ attends_school x) ↔ (∃ x : Type, honors_student x ∧ attends_school x) :=
sorry

end negation_of_no_honors_students_attend_school_l70_70693


namespace allison_upload_ratio_l70_70271

theorem allison_upload_ratio :
  ∃ (x y : ℕ), (x + y = 30) ∧ (10 * x + 20 * y = 450) ∧ (x / 30 = 1 / 2) :=
by
  sorry

end allison_upload_ratio_l70_70271


namespace charity_meaning_l70_70876

theorem charity_meaning (noun_charity : String) (h : noun_charity = "charity") : 
  (noun_charity = "charity" → "charity" = "charitable organization") :=
by
  sorry

end charity_meaning_l70_70876


namespace gcd_101_power_l70_70760

theorem gcd_101_power (a b : ℕ) (h1 : a = 101^6 + 1) (h2 : b = 3 * 101^6 + 101^3 + 1) (h_prime : Nat.Prime 101) : Nat.gcd a b = 1 :=
by
  -- proof goes here
  sorry

end gcd_101_power_l70_70760


namespace price_of_each_pizza_l70_70397

variable (P : ℝ)

theorem price_of_each_pizza (h1 : 4 * P + 5 = 45) : P = 10 := by
  sorry

end price_of_each_pizza_l70_70397


namespace weight_of_new_person_l70_70805

theorem weight_of_new_person (A : ℤ) (avg_weight_dec : ℤ) (n : ℤ) (new_avg : ℤ)
  (h1 : A = 102)
  (h2 : avg_weight_dec = 2)
  (h3 : n = 30) 
  (h4 : new_avg = A - avg_weight_dec) : 
  (31 * new_avg) - (30 * A) = 40 := 
by 
  sorry

end weight_of_new_person_l70_70805


namespace z_rate_per_rupee_of_x_l70_70130

-- Given conditions as definitions in Lean 4
def x_share := 1 -- x gets Rs. 1 for this proof
def y_rate_per_rupee_of_x := 0.45
def y_share := 27
def total_amount := 105

-- The statement to prove
theorem z_rate_per_rupee_of_x :
  (105 - (1 * 60) - 27) / 60 = 0.30 :=
by
  sorry

end z_rate_per_rupee_of_x_l70_70130


namespace fraction_cookies_blue_or_green_l70_70530

theorem fraction_cookies_blue_or_green (C : ℕ) (h1 : 1/C = 1/4) (h2 : 0.5555555555555556 = 5/9) :
  (1/4 + (5/9) * (3/4)) = (2/3) :=
by sorry

end fraction_cookies_blue_or_green_l70_70530


namespace water_cost_function_solve_for_x_and_payments_l70_70532

def water_usage_A (x : ℕ) : ℕ := 5 * x
def water_usage_B (x : ℕ) : ℕ := 3 * x

def water_payment_A (x : ℕ) : ℕ :=
  if water_usage_A x <= 15 then 
    water_usage_A x * 2 
  else 
    15 * 2 + (water_usage_A x - 15) * 3

def water_payment_B (x : ℕ) : ℕ :=
  if water_usage_B x <= 15 then 
    water_usage_B x * 2 
  else 
    15 * 2 + (water_usage_B x - 15) * 3

def total_payment (x : ℕ) : ℕ := water_payment_A x + water_payment_B x

theorem water_cost_function (x : ℕ) : total_payment x =
  if 0 < x ∧ x ≤ 3 then 16 * x
  else if 3 < x ∧ x ≤ 5 then 21 * x - 15
  else if 5 < x then 24 * x - 30
  else 0 := sorry

theorem solve_for_x_and_payments (y : ℕ) : y = 114 → ∃ x, total_payment x = y ∧
  water_usage_A x = 30 ∧ water_payment_A x = 75 ∧
  water_usage_B x = 18 ∧ water_payment_B x = 39 := sorry

end water_cost_function_solve_for_x_and_payments_l70_70532


namespace painting_cost_3x_l70_70294

-- Define the dimensions of the original room and the painting cost
variables (L B H : ℝ)
def cost_of_painting (area : ℝ) : ℝ := 350

-- Create a definition for the calculation of area
def paint_area (L B H : ℝ) : ℝ := 2 * (L * H + B * H)

-- Define the new dimensions
def new_dimensions (L B H : ℝ) : ℝ × ℝ × ℝ := (3 * L, 3 * B, 3 * H)

-- Create a definition for the calculation of the new area
def new_paint_area (L B H : ℝ) : ℝ := 18 * (paint_area L B H)

-- Calculate the new cost
def new_cost (L B H : ℝ) : ℝ := 18 * cost_of_painting (paint_area L B H)

-- The theorem to be proved
theorem painting_cost_3x (L B H : ℝ) : new_cost L B H = 6300 :=
by 
  simp [new_cost, cost_of_painting, paint_area]
  sorry

end painting_cost_3x_l70_70294


namespace length_of_AB_l70_70413

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def slope_of_line : ℝ := Real.tan (Real.pi / 6)

-- Equation of the line in point-slope form
noncomputable def line_eq (x : ℝ) : ℝ :=
  (slope_of_line * x) + 1

-- Intersection points of the line with the parabola y = (1/4)x^2
noncomputable def parabola_eq (x : ℝ) : ℝ :=
  (1/4) * x ^ 2

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, 
    (A.2 = parabola_eq A.1) ∧
    (B.2 = parabola_eq B.1) ∧ 
    (A.2 = line_eq A.1) ∧
    (B.2 = line_eq B.1) ∧
    ((((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ^ (1 / 2)) = 16 / 3) :=
by
  sorry

end length_of_AB_l70_70413


namespace area_of_shaded_quadrilateral_l70_70019

-- The problem setup
variables 
  (triangle : Type) [Nonempty triangle]
  (area : triangle → ℝ)
  (EFA FAB FBD CEDF : triangle)
  (h_EFA : area EFA = 5)
  (h_FAB : area FAB = 9)
  (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF)

-- The goal to prove
theorem area_of_shaded_quadrilateral (EFA FAB FBD CEDF : triangle) 
  (h_EFA : area EFA = 5) (h_FAB : area FAB = 9) (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF) : 
  area CEDF = 45 :=
by
  sorry

end area_of_shaded_quadrilateral_l70_70019


namespace find_vertex_A_l70_70903

variables (B C: ℝ × ℝ × ℝ)

-- Defining midpoints conditions
def midpoint_BC : ℝ × ℝ × ℝ := (1, 5, -1)
def midpoint_AC : ℝ × ℝ × ℝ := (0, 4, -2)
def midpoint_AB : ℝ × ℝ × ℝ := (2, 3, 4)

-- The coordinates of point A we need to prove
def target_A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Lean statement proving the coordinates of A
theorem find_vertex_A (A B C : ℝ × ℝ × ℝ)
  (hBC : midpoint_BC = (1, 5, -1))
  (hAC : midpoint_AC = (0, 4, -2))
  (hAB : midpoint_AB = (2, 3, 4)) :
  A = (1, 2, 3) := 
sorry

end find_vertex_A_l70_70903


namespace find_b_from_ellipse_l70_70416

-- Definitions used in conditions
variables {F₁ F₂ : ℝ → ℝ} -- foci
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Conditions
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse a b P.1 P.2
def perpendicular_vectors (P : ℝ × ℝ) : Prop := true -- Simplified, use correct condition in detailed proof
def area_of_triangle (P : ℝ × ℝ) (F₁ F₂ : ℝ → ℝ) : ℝ := 9

-- The target statement
theorem find_b_from_ellipse (P : ℝ × ℝ) (condition1 : point_on_ellipse a b P)
  (condition2 : perpendicular_vectors P) 
  (condition3 : area_of_triangle P F₁ F₂ = 9) : 
  b = 3 := 
sorry

end find_b_from_ellipse_l70_70416


namespace determine_a_l70_70164

theorem determine_a (a : ℚ) (x : ℚ) : 
  (∃ r s : ℚ, (r*x + s)^2 = a*x^2 + 18*x + 16) → 
  a = 81/16 := 
sorry

end determine_a_l70_70164


namespace find_m_l70_70142

theorem find_m (n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 :=
by
  sorry

end find_m_l70_70142


namespace intersection_M_N_eq_neg2_l70_70888

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l70_70888


namespace gcd_1234_2047_l70_70596

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 :=
by sorry

end gcd_1234_2047_l70_70596


namespace greatest_number_of_dimes_l70_70798

theorem greatest_number_of_dimes (total_value : ℝ) (num_dimes : ℕ) (num_nickels : ℕ) 
  (h_same_num : num_dimes = num_nickels) (h_total_value : total_value = 4.80) 
  (h_value_calculation : 0.10 * num_dimes + 0.05 * num_nickels = total_value) :
  num_dimes = 32 :=
by
  sorry

end greatest_number_of_dimes_l70_70798


namespace range_of_x_l70_70444

open Real

theorem range_of_x (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * sqrt 2 :=
sorry

end range_of_x_l70_70444


namespace pow_addition_l70_70601

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end pow_addition_l70_70601


namespace Calvin_insect_count_l70_70112

theorem Calvin_insect_count:
  ∀ (roaches scorpions crickets caterpillars : ℕ), 
    roaches = 12 →
    scorpions = 3 →
    crickets = roaches / 2 →
    caterpillars = scorpions * 2 →
    roaches + scorpions + crickets + caterpillars = 27 := 
by
  intros roaches scorpions crickets caterpillars h_roaches h_scorpions h_crickets h_caterpillars
  rw [h_roaches, h_scorpions, h_crickets, h_caterpillars]
  norm_num
  sorry

end Calvin_insect_count_l70_70112


namespace ellipse_value_l70_70647

noncomputable def a_c_ratio (a c : ℝ) : ℝ :=
  (a + c) / (a - c)

theorem ellipse_value (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2) 
  (h2 : a^2 + b^2 - 3 * c^2 = 0) :
  a_c_ratio a c = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_value_l70_70647


namespace sum_of_coordinates_reflection_l70_70123

theorem sum_of_coordinates_reflection (y : ℝ) :
  let A := (3, y)
  let B := (3, -y)
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  let A := (3, y)
  let B := (3, -y)
  sorry

end sum_of_coordinates_reflection_l70_70123


namespace ratio_of_counters_l70_70474

theorem ratio_of_counters (C_K M_K C_total M_ratio : ℕ)
  (h1 : C_K = 40)
  (h2 : M_K = 50)
  (h3 : M_ratio = 4 * M_K)
  (h4 : C_total = C_K + M_ratio)
  (h5 : C_total = 320) :
  C_K ≠ 0 → (320 - M_ratio) / C_K = 3 :=
by
  sorry

end ratio_of_counters_l70_70474


namespace student_A_more_stable_performance_l70_70763

theorem student_A_more_stable_performance
    (mean : ℝ)
    (n : ℕ)
    (variance_A variance_B : ℝ)
    (h1 : mean = 1.6)
    (h2 : n = 10)
    (h3 : variance_A = 1.4)
    (h4 : variance_B = 2.5) :
    variance_A < variance_B :=
by {
    -- The proof is omitted as we are only writing the statement here.
    sorry
}

end student_A_more_stable_performance_l70_70763


namespace solid_is_cylinder_l70_70432

def solid_views (v1 v2 v3 : String) : Prop := 
  -- This definition makes a placeholder for the views of the solid.
  sorry

def is_cylinder (s : String) : Prop := 
  s = "Cylinder"

theorem solid_is_cylinder (v1 v2 v3 : String) (h : solid_views v1 v2 v3) :
  ∃ s : String, is_cylinder s :=
sorry

end solid_is_cylinder_l70_70432


namespace diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l70_70864

noncomputable def diagonals_in_regular_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

noncomputable def exterior_angle (n : ℕ) : ℝ :=
  360.0 / n

theorem diagonals_of_60_sided_polygon :
  diagonals_in_regular_polygon 60 = 1710 :=
by
  sorry

theorem exterior_angle_of_60_sided_polygon :
  exterior_angle 60 = 6.0 :=
by
  sorry

end diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l70_70864


namespace smallest_positive_integer_g_l70_70605

theorem smallest_positive_integer_g (g : ℕ) (h_pos : g > 0) (h_square : ∃ k : ℕ, 3150 * g = k^2) : g = 14 := 
  sorry

end smallest_positive_integer_g_l70_70605


namespace period_in_years_proof_l70_70639

-- Definitions
def marbles (P : ℕ) : ℕ := P

def remaining_marbles (M : ℕ) : ℕ := (M / 4)

def doubled_remaining_marbles (M : ℕ) : ℕ := 2 * (M / 4)

def age_in_five_years (current_age : ℕ) : ℕ := current_age + 5

-- Given Conditions
variables (P : ℕ) (current_age : ℕ) (H1 : marbles P = P) (H2 : current_age = 45)

-- Final Proof Goal
theorem period_in_years_proof (H3 : doubled_remaining_marbles P = age_in_five_years current_age) : P = 100 :=
sorry

end period_in_years_proof_l70_70639


namespace third_median_length_is_9_l70_70718

noncomputable def length_of_third_median_of_triangle (m₁ m₂ m₃ area : ℝ) : Prop :=
  ∃ median : ℝ, median = m₃

theorem third_median_length_is_9 :
  length_of_third_median_of_triangle 5 7 9 (6 * Real.sqrt 10) :=
by
  sorry

end third_median_length_is_9_l70_70718


namespace michael_points_scored_l70_70931

theorem michael_points_scored (team_points : ℕ) (other_players : ℕ) (average_points : ℕ) (michael_points : ℕ) :
  team_points = 72 → other_players = 8 → average_points = 9 → 
  michael_points = team_points - other_players * average_points → michael_points = 36 :=
by
  intro h_team_points h_other_players h_average_points h_calculation
  -- skip the actual proof for now
  sorry

end michael_points_scored_l70_70931


namespace find_slope_of_line_l70_70015

theorem find_slope_of_line
  (k : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (3, 0))
  (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ x^2 - y^2 / 3 = 1)
  (A B : ℝ × ℝ)
  (hA : C A.1 A.2)
  (hB : C B.1 B.2)
  (line : ℝ → ℝ → Prop)
  (hline : ∀ x y, line x y ↔ y = k * (x - 3))
  (hintersectA : line A.1 A.2)
  (hintersectB : line B.1 B.2)
  (F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hfoci_sum : ∀ z : ℝ × ℝ, |z.1 - F.1| + |z.2 - F.2| = 16) :
  k = 3 ∨ k = -3 :=
by
  sorry

end find_slope_of_line_l70_70015


namespace right_triangle_sides_l70_70038

theorem right_triangle_sides (a b c : ℝ) (h : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c = 60 → h = 12 → a^2 + b^2 = c^2 → a * b = 12 * c → 
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end right_triangle_sides_l70_70038


namespace anne_carries_total_weight_l70_70313

-- Definitions for the conditions
def weight_female_cat : ℕ := 2
def weight_male_cat : ℕ := 2 * weight_female_cat

-- Problem statement
theorem anne_carries_total_weight : weight_female_cat + weight_male_cat = 6 :=
by
  sorry

end anne_carries_total_weight_l70_70313


namespace probabilityOfWearingSunglassesGivenCap_l70_70150

-- Define the conditions as Lean constants
def peopleWearingSunglasses : ℕ := 80
def peopleWearingCaps : ℕ := 60
def probabilityOfWearingCapGivenSunglasses : ℚ := 3 / 8
def peopleWearingBoth : ℕ := (3 / 8) * 80

-- Prove the desired probability
theorem probabilityOfWearingSunglassesGivenCap : (peopleWearingBoth / peopleWearingCaps = 1 / 2) :=
by
  -- sorry is used here to skip the proof
  sorry

end probabilityOfWearingSunglassesGivenCap_l70_70150


namespace determine_xyz_l70_70691

theorem determine_xyz (x y z : ℂ) (h1 : x * y + 3 * y = -9) (h2 : y * z + 3 * z = -9) (h3 : z * x + 3 * x = -9) : 
  x * y * z = 27 := 
by
  sorry

end determine_xyz_l70_70691


namespace tetrahedron_probability_correct_l70_70514

noncomputable def tetrahedron_probability : ℚ :=
  let total_arrangements := 16
  let suitable_arrangements := 2
  suitable_arrangements / total_arrangements

theorem tetrahedron_probability_correct : tetrahedron_probability = 1 / 8 :=
by
  sorry

end tetrahedron_probability_correct_l70_70514


namespace Sue_chewing_gums_count_l70_70314

theorem Sue_chewing_gums_count (S : ℕ) 
  (hMary : 5 = 5) 
  (hSam : 10 = 10) 
  (hTotal : 5 + 10 + S = 30) : S = 15 := 
by {
  sorry
}

end Sue_chewing_gums_count_l70_70314


namespace absolute_value_bound_l70_70551

theorem absolute_value_bound (x : ℝ) (hx : |x| ≤ 2) : |3 * x - x^3| ≤ 2 := 
by
  sorry

end absolute_value_bound_l70_70551


namespace area_of_rectangle_l70_70846

def length_fence (x : ℝ) : ℝ := 2 * x + 2 * x

theorem area_of_rectangle (x : ℝ) (h : length_fence x = 150) : x * 2 * x = 2812.5 :=
by
  sorry

end area_of_rectangle_l70_70846


namespace percentage_increase_l70_70232

def old_price : ℝ := 300
def new_price : ℝ := 330

theorem percentage_increase : ((new_price - old_price) / old_price) * 100 = 10 := by
  sorry

end percentage_increase_l70_70232


namespace advertising_department_size_l70_70734

-- Define the conditions provided in the problem.
def total_employees : Nat := 1000
def sample_size : Nat := 80
def advertising_sample_size : Nat := 4

-- Define the main theorem to prove the given problem.
theorem advertising_department_size :
  ∃ n : Nat, (advertising_sample_size : ℚ) / n = (sample_size : ℚ) / total_employees ∧ n = 50 :=
by
  sorry

end advertising_department_size_l70_70734


namespace max_deflection_angle_l70_70664

variable (M m : ℝ)
variable (h : M > m)

theorem max_deflection_angle :
  ∃ α : ℝ, α = Real.arcsin (m / M) := by
  sorry

end max_deflection_angle_l70_70664


namespace monotonically_increasing_range_l70_70269

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l70_70269


namespace garden_width_is_14_l70_70948

theorem garden_width_is_14 (w : ℝ) (h1 : ∃ (l : ℝ), l = 3 * w ∧ l * w = 588) : w = 14 :=
sorry

end garden_width_is_14_l70_70948


namespace bobby_initial_candy_count_l70_70204

theorem bobby_initial_candy_count (C : ℕ) (h : C + 4 + 14 = 51) : C = 33 :=
by
  sorry

end bobby_initial_candy_count_l70_70204


namespace sum_of_coeffs_eq_negative_21_l70_70070

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end sum_of_coeffs_eq_negative_21_l70_70070


namespace remainder_problem_l70_70940

theorem remainder_problem (x y : ℤ) (k m : ℤ) 
  (hx : x = 126 * k + 11) 
  (hy : y = 126 * m + 25) :
  (x + y + 23) % 63 = 59 := 
by
  sorry

end remainder_problem_l70_70940


namespace area_of_FDBG_l70_70466

noncomputable def area_quadrilateral (AB AC : ℝ) (area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let sin_A := (2 * area_ABC) / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sin_A
  let BC := (2 * area_ABC) / (AC * sin_A)
  let GC := BC / 3
  let area_AGC := (1 / 2) * AC * GC * sin_A
  area_ABC - (area_ADE + area_AGC)

theorem area_of_FDBG (AB AC : ℝ) (area_ABC : ℝ)
  (h1 : AB = 30)
  (h2 : AC = 15) 
  (h3 : area_ABC = 90) :
  area_quadrilateral AB AC area_ABC = 37.5 :=
by
  intros
  sorry

end area_of_FDBG_l70_70466


namespace percentage_increase_in_rent_l70_70088

theorem percentage_increase_in_rent
  (avg_rent_per_person_before : ℝ)
  (num_friends : ℕ)
  (friend_original_rent : ℝ)
  (avg_rent_per_person_after : ℝ)
  (total_rent_before : ℝ := num_friends * avg_rent_per_person_before)
  (total_rent_after : ℝ := num_friends * avg_rent_per_person_after)
  (rent_increase : ℝ := total_rent_after - total_rent_before)
  (percentage_increase : ℝ := (rent_increase / friend_original_rent) * 100)
  (h1 : avg_rent_per_person_before = 800)
  (h2 : num_friends = 4)
  (h3 : friend_original_rent = 1400)
  (h4 : avg_rent_per_person_after = 870) :
  percentage_increase = 20 :=
by
  sorry

end percentage_increase_in_rent_l70_70088


namespace tan_product_30_60_l70_70955

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * Real.pi / 180)) * (1 + Real.tan (60 * Real.pi / 180)) = 2 + (4 * Real.sqrt 3) / 3 := 
  sorry

end tan_product_30_60_l70_70955


namespace ellipse_condition_l70_70304

theorem ellipse_condition (m n : ℝ) :
  (mn > 0) → (¬ (∃ x y : ℝ, (m = 1) ∧ (n = 1) ∧ (x^2)/m + (y^2)/n = 1 ∧ (x, y) ≠ (0,0))) :=
sorry

end ellipse_condition_l70_70304


namespace ideal_type_circle_D_l70_70916

-- Define the line equation
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the distance condition for circles
def ideal_type_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), 
    line_l P.1 P.2 ∧ line_l Q.1 Q.2 ∧
    dist P (0, 0) = radius ∧
    dist Q (0, 0) = radius ∧
    dist (P, Q) = 1

-- Definition of given circles A, B, C, D
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 1
def circle_D (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 16

-- Define circle centers and radii for A, B, C, D
def center_A : ℝ × ℝ := (0, 0)
def radius_A : ℝ := 1
def center_B : ℝ × ℝ := (0, 0)
def radius_B : ℝ := 4
def center_C : ℝ × ℝ := (4, 4)
def radius_C : ℝ := 1
def center_D : ℝ × ℝ := (4, 4)
def radius_D : ℝ := 4

-- Problem Statement: Prove that option D is the "ideal type" circle
theorem ideal_type_circle_D : 
  ideal_type_circle center_D radius_D :=
sorry

end ideal_type_circle_D_l70_70916


namespace find_angle_A_l70_70801

theorem find_angle_A 
  (a b c A B C : ℝ)
  (h₀ : a = Real.sqrt 2)
  (h₁ : b = 2)
  (h₂ : Real.sin B - Real.cos B = Real.sqrt 2)
  (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  : A = Real.pi / 6 := 
  sorry

end find_angle_A_l70_70801


namespace sequence_general_term_l70_70193

theorem sequence_general_term (a : ℕ → ℤ) (n : ℕ) 
  (h₀ : a 0 = 1) 
  (h_rec : ∀ n, a (n + 1) = 2 * a n + n) :
  a n = 2^(n + 1) - n - 1 :=
by sorry

end sequence_general_term_l70_70193


namespace kiwis_to_add_for_25_percent_oranges_l70_70679

theorem kiwis_to_add_for_25_percent_oranges :
  let oranges := 24
  let kiwis := 30
  let apples := 15
  let bananas := 20
  let total_fruits := oranges + kiwis + apples + bananas
  let target_total_fruits := (oranges : ℝ) / 0.25
  let fruits_to_add := target_total_fruits - (total_fruits : ℝ)
  fruits_to_add = 7 := by
  sorry

end kiwis_to_add_for_25_percent_oranges_l70_70679


namespace hiring_manager_acceptance_l70_70430

theorem hiring_manager_acceptance {k : ℤ} 
  (avg_age : ℤ) (std_dev : ℤ) (num_accepted_ages : ℤ) 
  (h_avg : avg_age = 20) (h_std_dev : std_dev = 8)
  (h_num_accepted : num_accepted_ages = 17) : 
  (20 + k * 8 - (20 - k * 8) + 1) = 17 → k = 1 :=
by
  intros
  sorry

end hiring_manager_acceptance_l70_70430


namespace fraction_of_grid_covered_l70_70508

open Real EuclideanGeometry

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem fraction_of_grid_covered :
  let A := (2, 2)
  let B := (6, 2)
  let C := (4, 5)
  let grid_area := 7 * 7
  let triangle_area := area_of_triangle A B C
  triangle_area / grid_area = 6 / 49 := by
  sorry

end fraction_of_grid_covered_l70_70508


namespace simplified_evaluated_expression_l70_70774

noncomputable def a : ℚ := 1 / 3
noncomputable def b : ℚ := 1 / 2
noncomputable def c : ℚ := 1

def expression (a b c : ℚ) : ℚ := a^2 + 2 * b - c

theorem simplified_evaluated_expression :
  expression a b c = 1 / 9 :=
by
  sorry

end simplified_evaluated_expression_l70_70774


namespace xy_product_l70_70346

-- Define the proof problem with the conditions and required statement
theorem xy_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy_distinct : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 := 
  sorry

end xy_product_l70_70346


namespace maze_paths_unique_l70_70979

-- Define the conditions and branching points
def maze_structure (x : ℕ) (b : ℕ) : Prop :=
  x > 0 ∧ b > 0 ∧
  -- This represents the structure and unfolding paths at each point
  ∀ (i : ℕ), i < x → ∃ j < b, True

-- Define a function to count the number of unique paths given the number of branching points
noncomputable def count_paths (x : ℕ) (b : ℕ) : ℕ :=
  x * (2 ^ b)

-- State the main theorem
theorem maze_paths_unique : ∃ x b, maze_structure x b ∧ count_paths x b = 16 :=
by
  -- The proof contents are skipped for now
  sorry

end maze_paths_unique_l70_70979


namespace range_of_a_for_empty_solution_set_l70_70542

theorem range_of_a_for_empty_solution_set :
  {a : ℝ | ∀ x : ℝ, (a^2 - 9) * x^2 + (a + 3) * x - 1 < 0} = 
  {a : ℝ | -3 ≤ a ∧ a < 9 / 5} :=
sorry

end range_of_a_for_empty_solution_set_l70_70542


namespace sum_of_powers_l70_70850

theorem sum_of_powers (a b : ℝ) (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 72 := 
by
  sorry

end sum_of_powers_l70_70850


namespace smallest_n_cube_ends_with_2016_l70_70431

theorem smallest_n_cube_ends_with_2016 : ∃ n : ℕ, (n^3 % 10000 = 2016) ∧ (∀ m : ℕ, (m^3 % 10000 = 2016) → n ≤ m) :=
sorry

end smallest_n_cube_ends_with_2016_l70_70431


namespace snickers_bars_needed_l70_70106

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end snickers_bars_needed_l70_70106


namespace men_absent_l70_70488

theorem men_absent (x : ℕ) :
  let original_men := 42
  let original_days := 17
  let remaining_days := 21 
  let total_work := original_men * original_days
  let remaining_men_work := (original_men - x) * remaining_days 
  total_work = remaining_men_work →
  x = 8 :=
by
  intros
  let total_work := 42 * 17
  let remaining_men_work := (42 - x) * 21
  have h : total_work = remaining_men_work := ‹total_work = remaining_men_work›
  sorry

end men_absent_l70_70488


namespace pufferfish_count_l70_70310

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end pufferfish_count_l70_70310


namespace perpendicular_line_equation_l70_70390

theorem perpendicular_line_equation (x y : ℝ) (h : 2 * x + y + 3 = 0) (hx : ∃ c : ℝ, x - 2 * y + c = 0) :
  (c = 7 ↔ ∀ p : ℝ × ℝ, p = (-1, 3) → (p.1 - 2 * p.2 + 7 = 0)) :=
sorry

end perpendicular_line_equation_l70_70390


namespace christian_sue_need_more_money_l70_70569

-- Definition of initial amounts
def christian_initial := 5
def sue_initial := 7

-- Definition of earnings from activities
def christian_per_yard := 5
def christian_yards := 4
def sue_per_dog := 2
def sue_dogs := 6

-- Definition of perfume cost
def perfume_cost := 50

-- Theorem statement for the math problem
theorem christian_sue_need_more_money :
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  total_money < perfume_cost → perfume_cost - total_money = 6 :=
by 
  intros
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  sorry

end christian_sue_need_more_money_l70_70569


namespace sqrt_of_4_l70_70175

theorem sqrt_of_4 :
  ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) :=
sorry

end sqrt_of_4_l70_70175


namespace total_subjects_l70_70374

theorem total_subjects (subjects_monica subjects_marius subjects_millie : ℕ)
  (h1 : subjects_monica = 10)
  (h2 : subjects_marius = subjects_monica + 4)
  (h3 : subjects_millie = subjects_marius + 3) :
  subjects_monica + subjects_marius + subjects_millie = 41 :=
by
  sorry

end total_subjects_l70_70374


namespace searchlight_reflector_distance_l70_70519

noncomputable def parabola_vertex_distance : Rat :=
  let diameter := 60 -- in cm
  let depth := 40 -- in cm
  let x := 40 -- x-coordinate of the point
  let y := 30 -- y-coordinate of the point
  let p := (y^2) / (2 * x)
  p / 2

theorem searchlight_reflector_distance : parabola_vertex_distance = 45 / 8 := by
  sorry

end searchlight_reflector_distance_l70_70519


namespace find_dividing_line_l70_70224

/--
A line passing through point P(1,1) divides the circular region \{(x, y) \mid x^2 + y^2 \leq 4\} into two parts,
making the difference in area between these two parts the largest. Prove that the equation of this line is x + y - 2 = 0.
-/
theorem find_dividing_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∃ (A B C : ℝ), A * 1 + B * 1 + C = 0 ∧
                 (∀ x y, x^2 + y^2 ≤ 4 → A * x + B * y + C = 0 → (x + y - 2) = 0) :=
sorry

end find_dividing_line_l70_70224


namespace angles_equal_l70_70928

theorem angles_equal (A B C : ℝ) (h1 : A + B = 180) (h2 : B + C = 180) : A = C := sorry

end angles_equal_l70_70928


namespace total_number_of_students_l70_70602

theorem total_number_of_students
  (ratio_girls_to_boys : ℕ) (ratio_boys_to_girls : ℕ)
  (num_girls : ℕ)
  (ratio_condition : ratio_girls_to_boys = 5 ∧ ratio_boys_to_girls = 8)
  (num_girls_condition : num_girls = 160)
  : (num_girls * (ratio_girls_to_boys + ratio_boys_to_girls) / ratio_girls_to_boys = 416) :=
by
  sorry

end total_number_of_students_l70_70602


namespace possible_quadrilateral_areas_l70_70995

-- Define the problem set up
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  side_length : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

-- Defines the division points on each side of the square
def division_points (A B C D : Point) : List Point :=
  [
    -- Points on AB
    { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
    -- Points on BC
    { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
    -- Points on CD
    { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
    -- Points on DA
    { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
  ]

-- Possible areas calculation using the Shoelace Theorem
def quadrilateral_areas : List ℝ :=
  [6, 7, 7.5, 8, 8.5, 9, 10]

-- Math proof problem in Lean, we need to prove that the quadrilateral areas match the given values
theorem possible_quadrilateral_areas (ABCD : Square) (pts : List Point) :
    (division_points ABCD.A ABCD.B ABCD.C ABCD.D) = [
      { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
      { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
      { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
      { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
    ] → 
    (∃ areas, areas ⊆ quadrilateral_areas) := by
  sorry

end possible_quadrilateral_areas_l70_70995


namespace find_m_l70_70250

-- Define the operation a * b
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

theorem find_m (m : ℝ) (h : star 3 m = 17) : m = 14 :=
by
  -- Placeholder for the proof
  sorry

end find_m_l70_70250


namespace operation_1_2010_l70_70116

def operation (m n : ℕ) : ℕ := sorry

axiom operation_initial : operation 1 1 = 2
axiom operation_step (m n : ℕ) : operation m (n + 1) = operation m n + 3

theorem operation_1_2010 : operation 1 2010 = 6029 := sorry

end operation_1_2010_l70_70116


namespace smallest_value_of_y1_y2_y3_sum_l70_70037

noncomputable def y_problem := 
  ∃ (y1 y2 y3 : ℝ), 
  (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)

theorem smallest_value_of_y1_y2_y3_sum :
  (∃ (y1 y2 y3 : ℝ), 0 < y1 ∧ 0 < y2 ∧ 0 < y3 ∧ (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)) :=
by 
  sorry

end smallest_value_of_y1_y2_y3_sum_l70_70037


namespace infinite_geometric_series_sum_l70_70004

-- First term of the geometric series
def a : ℚ := 5/3

-- Common ratio of the geometric series
def r : ℚ := -1/4

-- The sum of the infinite geometric series
def S : ℚ := a / (1 - r)

-- Prove that the sum of the series is equal to 4/3
theorem infinite_geometric_series_sum : S = 4/3 := by
  sorry

end infinite_geometric_series_sum_l70_70004


namespace football_field_area_l70_70553

-- Define the conditions
def fertilizer_spread : ℕ := 1200
def area_partial : ℕ := 3600
def fertilizer_partial : ℕ := 400

-- Define the expected result
def area_total : ℕ := 10800

-- Theorem to prove
theorem football_field_area :
  (fertilizer_spread / (fertilizer_partial / area_partial)) = area_total :=
by sorry

end football_field_area_l70_70553


namespace opposite_of_neg_three_l70_70091

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l70_70091


namespace linear_regression_equation_l70_70344

-- Given conditions
variables (x y : ℝ)
variable (corr_pos : x ≠ 0 → y / x > 0)
noncomputable def x_mean : ℝ := 2.4
noncomputable def y_mean : ℝ := 3.2

-- Regression line equation
theorem linear_regression_equation :
  (y = 0.5 * x + 2) ∧ (∀ x' y', (x' = x_mean ∧ y' = y_mean) → (y' = 0.5 * x' + 2)) :=
by
  sorry

end linear_regression_equation_l70_70344


namespace range_of_expression_l70_70160

theorem range_of_expression (a : ℝ) : (∃ a : ℝ, a + 1 ≥ 0 ∧ a - 2 ≠ 0) → (a ≥ -1 ∧ a ≠ 2) := 
by sorry

end range_of_expression_l70_70160


namespace find_a_l70_70915

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a - 2, a^2 + 4*a, 10}) (h : -3 ∈ A) : a = -3 := 
by
  -- placeholder proof
  sorry

end find_a_l70_70915


namespace log_12_eq_2a_plus_b_l70_70911

variable (lg : ℝ → ℝ)
variable (lg_2_eq_a : lg 2 = a)
variable (lg_3_eq_b : lg 3 = b)

theorem log_12_eq_2a_plus_b : lg 12 = 2 * a + b :=
by
  sorry

end log_12_eq_2a_plus_b_l70_70911


namespace find_x_l70_70961

def infinite_sqrt (d : ℝ) : ℝ := sorry -- A placeholder since infinite nesting is non-trivial

def bowtie (c d : ℝ) : ℝ := c - infinite_sqrt d

theorem find_x (x : ℝ) (h : bowtie 7 x = 3) : x = 20 :=
sorry

end find_x_l70_70961


namespace odd_integer_solution_l70_70883

theorem odd_integer_solution
  (y : ℤ) (hy_odd : y % 2 = 1)
  (h : ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1) :
  y = 1 :=
sorry

end odd_integer_solution_l70_70883


namespace solution_set_leq_2_l70_70683

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end solution_set_leq_2_l70_70683


namespace ratio_of_points_l70_70589

def Noa_points : ℕ := 30
def total_points : ℕ := 90

theorem ratio_of_points (Phillip_points : ℕ) (h1 : Phillip_points = 2 * Noa_points) (h2 : Noa_points + Phillip_points = total_points) : Phillip_points / Noa_points = 2 := 
by
  intros
  sorry

end ratio_of_points_l70_70589


namespace tickets_sold_l70_70177

def advanced_purchase_tickets := ℕ
def door_purchase_tickets := ℕ

variable (A D : ℕ)

theorem tickets_sold :
  (A + D = 140) →
  (8 * A + 14 * D = 1720) →
  A = 40 :=
by
  intros h1 h2
  sorry

end tickets_sold_l70_70177


namespace decomposition_x_pqr_l70_70967

-- Definitions of vectors x, p, q, r
def x : ℝ := sorry
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- The linear combination we want to prove
theorem decomposition_x_pqr : 
  (x = -1 • p + 4 • q + 3 • r) :=
sorry

end decomposition_x_pqr_l70_70967


namespace boxes_with_neither_l70_70128

def total_boxes : ℕ := 15
def boxes_with_stickers : ℕ := 9
def boxes_with_stamps : ℕ := 5
def boxes_with_both : ℕ := 3

theorem boxes_with_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_stamps : ℕ)
  (boxes_with_both : ℕ) :
  total_boxes - ((boxes_with_stickers + boxes_with_stamps) - boxes_with_both) = 4 :=
by
  sorry

end boxes_with_neither_l70_70128


namespace simplify_expression_l70_70202

variable (b c : ℝ)

theorem simplify_expression :
  3 * b * (3 * b ^ 3 + 2 * b) - 2 * b ^ 2 + c * (3 * b ^ 2 - c) = 9 * b ^ 4 + 4 * b ^ 2 + 3 * b ^ 2 * c - c ^ 2 :=
by
  sorry

end simplify_expression_l70_70202


namespace calc_product_eq_243_l70_70066

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l70_70066


namespace sum_of_variables_l70_70705

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_of_variables (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧ log 3 (log 4 (log 2 y)) = 0 ∧ log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 :=
by
  sorry

end sum_of_variables_l70_70705


namespace determine_c_l70_70490

noncomputable def c_floor : ℤ := -3
noncomputable def c_frac : ℝ := (25 - Real.sqrt 481) / 8

theorem determine_c : c_floor + c_frac = -2.72 := by
  have h1 : 3 * (c_floor : ℝ)^2 + 19 * (c_floor : ℝ) - 63 = 0 := by
    sorry
  have h2 : 4 * c_frac^2 - 25 * c_frac + 9 = 0 := by
    sorry
  sorry

end determine_c_l70_70490


namespace b_minus_a_l70_70908

theorem b_minus_a :
  ∃ (a b : ℝ), (2 + 4 = -a) ∧ (2 * 4 = b) ∧ (b - a = 14) :=
by
  use (-6 : ℝ)
  use (8 : ℝ)
  simp
  sorry

end b_minus_a_l70_70908


namespace proof_problem_l70_70363

variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be increasing on (-∞, 0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y < 0 → f x < f y

-- Define what it means for a function to be decreasing on (0, +∞)
def is_decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f y < f x

theorem proof_problem 
  (h_even : is_even_function f) 
  (h_inc_neg : is_increasing_on_neg f) : 
  (∀ x : ℝ, f (-x) - f x = 0) ∧ (is_decreasing_on_pos f) :=
by
  sorry

end proof_problem_l70_70363


namespace bird_cages_count_l70_70156

-- Definitions based on the conditions provided
def num_parrots_per_cage : ℕ := 2
def num_parakeets_per_cage : ℕ := 7
def total_birds_per_cage : ℕ := num_parrots_per_cage + num_parakeets_per_cage
def total_birds_in_store : ℕ := 54
def num_bird_cages : ℕ := total_birds_in_store / total_birds_per_cage

-- The proof we need to derive
theorem bird_cages_count : num_bird_cages = 6 := by
  sorry

end bird_cages_count_l70_70156


namespace solutions_to_h_eq_1_l70_70628

noncomputable def h (x : ℝ) : ℝ :=
if x ≤ 0 then 5 * x + 10 else 3 * x - 5

theorem solutions_to_h_eq_1 : {x : ℝ | h x = 1} = {-9/5, 2} :=
by
  sorry

end solutions_to_h_eq_1_l70_70628


namespace A_inter_B_empty_l70_70689

def Z_plus := { n : ℤ // 0 < n }

def A : Set ℤ := { x | ∃ n : Z_plus, x = 2 * (n.1) - 1 }
def B : Set ℤ := { y | ∃ x ∈ A, y = 3 * x - 1 }

theorem A_inter_B_empty : A ∩ B = ∅ :=
by {
  sorry
}

end A_inter_B_empty_l70_70689


namespace problem1_problem2_l70_70143

-- Definition for the first problem
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- First Lean 4 statement for 2^n + 3 = x^2
theorem problem1 (n : ℕ) (h : isPerfectSquare (2^n + 3)) : n = 0 :=
sorry

-- Second Lean 4 statement for 2^n + 1 = x^2
theorem problem2 (n : ℕ) (h : isPerfectSquare (2^n + 1)) : n = 3 :=
sorry

end problem1_problem2_l70_70143


namespace minimum_cuts_for_48_rectangles_l70_70778

theorem minimum_cuts_for_48_rectangles : 
  ∃ n : ℕ, n = 6 ∧ (∀ m < 6, 2 ^ m < 48) ∧ 2 ^ n ≥ 48 :=
by
  sorry

end minimum_cuts_for_48_rectangles_l70_70778


namespace circle_intersection_zero_l70_70170

theorem circle_intersection_zero :
  (∀ θ : ℝ, ∀ r1 : ℝ, r1 = 3 * Real.cos θ → ∀ r2 : ℝ, r2 = 6 * Real.sin (2 * θ) → False) :=
by 
  sorry

end circle_intersection_zero_l70_70170


namespace luis_finish_fourth_task_l70_70316

-- Define the starting and finishing times
def start_time : ℕ := 540  -- 9:00 AM is 540 minutes from midnight
def finish_third_task : ℕ := 750  -- 12:30 PM is 750 minutes from midnight
def duration_one_task : ℕ := (750 - 540) / 3  -- Time for one task

-- Define the problem statement
theorem luis_finish_fourth_task :
  start_time = 540 →
  finish_third_task = 750 →
  3 * duration_one_task = finish_third_task - start_time →
  finish_third_task + duration_one_task = 820 :=
by
  -- You can place the proof for the theorem here
  sorry

end luis_finish_fourth_task_l70_70316


namespace michael_choices_l70_70788

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem michael_choices : combination 10 4 = 210 := by
  sorry

end michael_choices_l70_70788


namespace brown_loss_percentage_is_10_l70_70208

-- Define the initial conditions
def initialHousePrice : ℝ := 100000
def profitPercentage : ℝ := 0.10
def sellingPriceBrown : ℝ := 99000

-- Compute the price Mr. Brown bought the house
def priceBrownBought := initialHousePrice * (1 + profitPercentage)

-- Define the loss percentage as a goal to prove
theorem brown_loss_percentage_is_10 :
  ((priceBrownBought - sellingPriceBrown) / priceBrownBought) * 100 = 10 := by
  sorry

end brown_loss_percentage_is_10_l70_70208


namespace distinct_patterns_4x4_three_squares_l70_70366

noncomputable def count_distinct_patterns : ℕ :=
  sorry

theorem distinct_patterns_4x4_three_squares :
  count_distinct_patterns = 12 :=
by sorry

end distinct_patterns_4x4_three_squares_l70_70366


namespace intersection_point_of_lines_l70_70497

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end intersection_point_of_lines_l70_70497


namespace cost_per_set_l70_70831

variable {C : ℝ} -- Define the variable cost per set.

theorem cost_per_set
  (initial_outlay : ℝ := 10000) -- Initial outlay for manufacturing.
  (revenue_per_set : ℝ := 50) -- Revenue per set sold.
  (sets_sold : ℝ := 500) -- Sets produced and sold.
  (profit : ℝ := 5000) -- Profit from selling 500 sets.

  (h_profit_eq : profit = (revenue_per_set * sets_sold) - (initial_outlay + C * sets_sold)) :
  C = 20 :=
by
  -- Proof to be filled in later.
  sorry

end cost_per_set_l70_70831


namespace find_e_l70_70291

theorem find_e (d e f : ℝ) (h1 : f = 5)
  (h2 : -d / 3 = -f)
  (h3 : -f = 1 + d + e + f) :
  e = -26 := 
by
  sorry

end find_e_l70_70291


namespace pens_in_shop_l70_70748

theorem pens_in_shop (P Pe E : ℕ) (h_ratio : 14 * Pe = 4 * P) (h_ratio2 : 14 * E = 14 * 3 + 11) (h_P : P = 140) (h_E : E = 30) : Pe = 40 :=
sorry

end pens_in_shop_l70_70748


namespace scrabble_middle_letter_value_l70_70102

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end scrabble_middle_letter_value_l70_70102


namespace probability_suitable_joint_given_physique_l70_70279

noncomputable def total_children : ℕ := 20
noncomputable def suitable_physique : ℕ := 4
noncomputable def suitable_joint_structure : ℕ := 5
noncomputable def both_physique_and_joint : ℕ := 2

noncomputable def P (n m : ℕ) : ℚ := n / m

theorem probability_suitable_joint_given_physique :
  P both_physique_and_joint total_children / P suitable_physique total_children = 1 / 2 :=
by
  sorry

end probability_suitable_joint_given_physique_l70_70279


namespace clock_chime_time_l70_70723

theorem clock_chime_time (t_5oclock : ℕ) (n_5chimes : ℕ) (t_10oclock : ℕ) (n_10chimes : ℕ)
  (h1: t_5oclock = 8) (h2: n_5chimes = 5) (h3: n_10chimes = 10) : 
  t_10oclock = 18 :=
by
  sorry

end clock_chime_time_l70_70723


namespace mixed_number_calculation_l70_70343

theorem mixed_number_calculation :
  47 * (2 + 2/3 - (3 + 1/4)) / (3 + 1/2 + (2 + 1/5)) = -4 - 25/38 :=
by
  sorry

end mixed_number_calculation_l70_70343


namespace rectangular_field_area_l70_70517

theorem rectangular_field_area (a c : ℝ) (h_a : a = 13) (h_c : c = 17) :
  ∃ b : ℝ, (b = 2 * Real.sqrt 30) ∧ (a * b = 26 * Real.sqrt 30) :=
by
  sorry

end rectangular_field_area_l70_70517


namespace insects_in_lab_l70_70252

theorem insects_in_lab (total_legs number_of_legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : number_of_legs_per_insect = 6) : (total_legs / number_of_legs_per_insect) = 6 :=
by
  sorry

end insects_in_lab_l70_70252


namespace problem_statement_l70_70808

theorem problem_statement : (515 % 1000) = 515 :=
by
  sorry

end problem_statement_l70_70808


namespace longest_piece_length_l70_70486

-- Define the lengths of the ropes
def rope1 : ℕ := 45
def rope2 : ℕ := 75
def rope3 : ℕ := 90

-- Define the greatest common divisor we need to prove
def gcd_of_ropes : ℕ := Nat.gcd rope1 (Nat.gcd rope2 rope3)

-- Goal theorem stating the problem
theorem longest_piece_length : gcd_of_ropes = 15 := by
  sorry

end longest_piece_length_l70_70486


namespace find_number_of_students_l70_70857

-- Parameters
variable (n : ℕ) (C : ℕ)
def first_and_last_picked_by_sam (n : ℕ) (C : ℕ) : Prop := 
  C + 1 = 2 * n

-- Conditions: number of candies is 120, the bag completes 2 full rounds at the table.
theorem find_number_of_students
  (C : ℕ) (h_C: C = 120) (h_rounds: 2 * n = C):
  n = 60 :=
by
  sorry

end find_number_of_students_l70_70857


namespace largest_number_is_l70_70526

-- Define the conditions stated in the problem
def sum_of_three_numbers_is_100 (a b c : ℝ) : Prop :=
  a + b + c = 100

def two_larger_numbers_differ_by_8 (b c : ℝ) : Prop :=
  c - b = 8

def two_smaller_numbers_differ_by_5 (a b : ℝ) : Prop :=
  b - a = 5

-- Define the hypothesis
def problem_conditions (a b c : ℝ) : Prop :=
  sum_of_three_numbers_is_100 a b c ∧
  two_larger_numbers_differ_by_8 b c ∧
  two_smaller_numbers_differ_by_5 a b

-- Define the proof problem
theorem largest_number_is (a b c : ℝ) (h : problem_conditions a b c) : 
  c = 121 / 3 :=
sorry

end largest_number_is_l70_70526


namespace nat_numbers_square_minus_one_power_of_prime_l70_70974

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ ∃ (k : ℕ), x = p ^ k

theorem nat_numbers_square_minus_one_power_of_prime (n : ℕ) (hn : 1 ≤ n) :
  is_power_of_prime (n ^ 2 - 1) ↔ (n = 2 ∨ n = 3) := by
  sorry

end nat_numbers_square_minus_one_power_of_prime_l70_70974


namespace sin_45_eq_sqrt2_div_2_l70_70347

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l70_70347


namespace expression_evaluation_l70_70744

theorem expression_evaluation : 
  (2^10 * 3^3) / (6 * 2^5) = 144 :=
by 
  sorry

end expression_evaluation_l70_70744


namespace meaningful_sqrt_l70_70190

theorem meaningful_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x = 6 :=
sorry

end meaningful_sqrt_l70_70190


namespace odd_factor_form_l70_70132

theorem odd_factor_form (n : ℕ) (x y : ℕ) (h_n : n > 0) (h_gcd : Nat.gcd x y = 1) :
  ∀ p, p ∣ (x ^ (2 ^ n) + y ^ (2 ^ n)) ∧ Odd p → ∃ k > 0, p = 2^(n+1) * k + 1 := 
by
  sorry

end odd_factor_form_l70_70132


namespace lcm_36_225_l70_70069

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end lcm_36_225_l70_70069


namespace Walter_allocates_for_school_l70_70853

open Nat

def Walter_works_5_days_a_week := 5
def Walter_earns_per_hour := 5
def Walter_works_per_day := 4
def Proportion_for_school := 3/4

theorem Walter_allocates_for_school :
  let daily_earnings := Walter_works_per_day * Walter_earns_per_hour
  let weekly_earnings := daily_earnings * Walter_works_5_days_a_week
  let school_allocation := weekly_earnings * Proportion_for_school
  school_allocation = 75 := by
  sorry

end Walter_allocates_for_school_l70_70853


namespace exists_four_scientists_l70_70002

theorem exists_four_scientists {n : ℕ} (h1 : n = 50)
  (knows : Fin n → Finset (Fin n))
  (h2 : ∀ x, (knows x).card ≥ 25) :
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  a ≠ c ∧ b ≠ d ∧
  a ∈ knows b ∧ b ∈ knows c ∧ c ∈ knows d ∧ d ∈ knows a :=
by
  sorry

end exists_four_scientists_l70_70002


namespace possible_values_of_n_l70_70742

theorem possible_values_of_n (n : ℕ) (h1 : 0 < n)
  (h2 : 12 * n^3 = n^4 + 11 * n^2) :
  n = 1 ∨ n = 11 :=
sorry

end possible_values_of_n_l70_70742


namespace total_silk_dyed_correct_l70_70749

-- Define the conditions
def green_silk_yards : ℕ := 61921
def pink_silk_yards : ℕ := 49500
def total_silk_yards : ℕ := green_silk_yards + pink_silk_yards

-- State the theorem to be proved
theorem total_silk_dyed_correct : total_silk_yards = 111421 := by
  sorry

end total_silk_dyed_correct_l70_70749


namespace andrea_living_room_area_l70_70823

/-- Given that 60% of Andrea's living room floor is covered by a carpet 
     which has dimensions 4 feet by 9 feet, prove that the area of 
     Andrea's living room floor is 60 square feet. -/
theorem andrea_living_room_area :
  ∃ A, (0.60 * A = 4 * 9) ∧ A = 60 :=
by
  sorry

end andrea_living_room_area_l70_70823


namespace cylindrical_to_rectangular_multiplied_l70_70567

theorem cylindrical_to_rectangular_multiplied :
  let r := 7
  let θ := Real.pi / 4
  let z := -3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (2 * x, 2 * y, 2 * z) = (7 * Real.sqrt 2, 7 * Real.sqrt 2, -6) := 
by
  sorry

end cylindrical_to_rectangular_multiplied_l70_70567


namespace base_of_right_angled_triangle_l70_70320

theorem base_of_right_angled_triangle 
  (height : ℕ) (area : ℕ) (hypotenuse : ℕ) (b : ℕ) 
  (h_height : height = 8)
  (h_area : area = 24)
  (h_hypotenuse : hypotenuse = 10) 
  (h_area_eq : area = (1 / 2 : ℕ) * b * height)
  (h_pythagorean : hypotenuse^2 = height^2 + b^2) : 
  b = 6 := 
sorry

end base_of_right_angled_triangle_l70_70320


namespace volume_of_cuboctahedron_l70_70154

def points (i j : ℕ) (A : ℕ → ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := A 0
  let (xi, yi, zi) := A i
  let (xj, yj, zj) := A j
  (xi - xj, yi - yj, zi - zj)

def is_cuboctahedron (points_set : Set (ℝ × ℝ × ℝ)) : Prop :=
  -- Insert specific conditions that define a cuboctahedron
  sorry

theorem volume_of_cuboctahedron : 
  let A := fun 
    | 0 => (0, 0, 0)
    | 1 => (1, 0, 0)
    | 2 => (0, 1, 0)
    | 3 => (0, 0, 1)
    | _ => (0, 0, 0)
  let P_ij := 
    {p | ∃ i j : ℕ, i ≠ j ∧ p = points i j A}
  ∃ v : ℝ, is_cuboctahedron P_ij ∧ v = 10 / 3 :=
sorry

end volume_of_cuboctahedron_l70_70154


namespace victor_percentage_of_marks_l70_70952

theorem victor_percentage_of_marks (marks_obtained : ℝ) (maximum_marks : ℝ) (h1 : marks_obtained = 285) (h2 : maximum_marks = 300) : 
  (marks_obtained / maximum_marks) * 100 = 95 :=
by
  sorry

end victor_percentage_of_marks_l70_70952


namespace trigonometric_identity_l70_70187

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) + Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 * Real.tan (10 * Real.pi / 180) :=
by
  sorry

end trigonometric_identity_l70_70187


namespace maximum_triangle_area_le_8_l70_70779

def lengths : List ℝ := [2, 3, 4, 5, 6]

-- Function to determine if three lengths can form a valid triangle
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a 

-- Heron's formula to compute the area of a triangle given its sides
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement to prove that the maximum possible area with given stick lengths is less than or equal to 8 cm²
theorem maximum_triangle_area_le_8 :
  ∃ (a b c : ℝ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ 
  is_valid_triangle a b c ∧ heron_area a b c ≤ 8 :=
sorry

end maximum_triangle_area_le_8_l70_70779


namespace floor_of_neg_sqrt_frac_l70_70148

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l70_70148


namespace shaded_fraction_eighth_triangle_l70_70973

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2
def square_number (n : Nat) : Nat := n * n

theorem shaded_fraction_eighth_triangle :
  let shaded_triangles := triangular_number 7
  let total_triangles := square_number 8
  shaded_triangles / total_triangles = 7 / 16 := 
by
  sorry

end shaded_fraction_eighth_triangle_l70_70973


namespace rem_fraction_l70_70223

theorem rem_fraction : 
  let rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋;
  rem (5/7) (-3/4) = -1/28 := 
by
  sorry

end rem_fraction_l70_70223


namespace plot_length_l70_70149

variable (b length : ℝ)

theorem plot_length (h1 : length = b + 10)
  (fence_N_cost : ℝ := 26.50 * (b + 10))
  (fence_E_cost : ℝ := 32 * b)
  (fence_S_cost : ℝ := 22 * (b + 10))
  (fence_W_cost : ℝ := 30 * b)
  (total_cost : ℝ := fence_N_cost + fence_E_cost + fence_S_cost + fence_W_cost)
  (h2 : 1.05 * total_cost = 7500) :
  length = 70.25 := by
  sorry

end plot_length_l70_70149


namespace average_age_after_person_leaves_l70_70337

theorem average_age_after_person_leaves 
  (initial_people : ℕ) 
  (initial_average_age : ℕ) 
  (person_leaving_age : ℕ) 
  (remaining_people : ℕ) 
  (new_average_age : ℝ)
  (h1 : initial_people = 7) 
  (h2 : initial_average_age = 32) 
  (h3 : person_leaving_age = 22) 
  (h4 : remaining_people = 6) :
  new_average_age = 34 := 
by 
  sorry

end average_age_after_person_leaves_l70_70337


namespace boys_total_count_l70_70062

theorem boys_total_count 
  (avg_age_all: ℤ) (avg_age_first6: ℤ) (avg_age_last6: ℤ)
  (total_first6: ℤ) (total_last6: ℤ) (total_age_all: ℤ) :
  avg_age_all = 50 →
  avg_age_first6 = 49 →
  avg_age_last6 = 52 →
  total_first6 = 6 * avg_age_first6 →
  total_last6 = 6 * avg_age_last6 →
  total_age_all = total_first6 + total_last6 →
  total_age_all = avg_age_all * 13 :=
by
  intros h_avg_all h_avg_first6 h_avg_last6 h_total_first6 h_total_last6 h_total_age_all
  rw [h_avg_all, h_avg_first6, h_avg_last6] at *
  -- Proof steps skipped
  sorry

end boys_total_count_l70_70062


namespace find_third_root_l70_70645

-- Define the polynomial
def poly (a b x : ℚ) : ℚ := a * x^3 + 2 * (a + b) * x^2 + (b - 2 * a) * x + (10 - a)

-- Define the roots condition
def is_root (a b x : ℚ) : Prop := poly a b x = 0

-- Given conditions and required proof
theorem find_third_root (a b : ℚ) (ha : a = 350 / 13) (hb : b = -1180 / 13) :
  is_root a b (-1) ∧ is_root a b 4 → 
  ∃ r : ℚ, is_root a b r ∧ r ≠ -1 ∧ r ≠ 4 ∧ r = 61 / 35 :=
by sorry

end find_third_root_l70_70645


namespace range_of_r_l70_70657

theorem range_of_r (r : ℝ) (h_r : r > 0) :
  let M := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}
  let N := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}
  (∀ p, p ∈ N → p ∈ M) → 0 < r ∧ r ≤ 2 - Real.sqrt 2 :=
by
  sorry

end range_of_r_l70_70657


namespace system_of_equations_solutions_l70_70229

theorem system_of_equations_solutions :
  ∃ (sol : Finset (ℝ × ℝ)), sol.card = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ sol ↔ (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1)) :=
by
  sorry

end system_of_equations_solutions_l70_70229


namespace meal_total_cost_l70_70725

theorem meal_total_cost (x : ℝ) (h_initial: x/5 - 15 = x/8) : x = 200 :=
by sorry

end meal_total_cost_l70_70725


namespace sin_cos_product_l70_70140

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l70_70140


namespace find_a_l70_70203

theorem find_a (a : ℤ) (A : Set ℤ) (B : Set ℤ) :
  A = {-2, 3 * a - 1, a^2 - 3} ∧
  B = {a - 2, a - 1, a + 1} ∧
  A ∩ B = {-2} → a = -3 :=
by
  intro H
  sorry

end find_a_l70_70203


namespace derivative_at_one_is_four_l70_70254

-- Define the function y = x^2 + 2x + 1
def f (x : ℝ) := x^2 + 2*x + 1

-- State the theorem: The derivative of f at x = 1 is 4
theorem derivative_at_one_is_four : (deriv f 1) = 4 :=
by
  -- The proof is omitted here.
  sorry

end derivative_at_one_is_four_l70_70254


namespace tetrahedron_inscribed_sphere_radius_l70_70053

theorem tetrahedron_inscribed_sphere_radius (a : ℝ) (r : ℝ) (a_pos : 0 < a) :
  (r = a * (Real.sqrt 6 + 1) / 8) ∨ 
  (r = a * (Real.sqrt 6 - 1) / 8) :=
sorry

end tetrahedron_inscribed_sphere_radius_l70_70053


namespace period_of_time_l70_70761

-- We define the annual expense and total amount spent as constants
def annual_expense : ℝ := 2
def total_amount_spent : ℝ := 20

-- Theorem to prove the period of time (in years)
theorem period_of_time : total_amount_spent / annual_expense = 10 :=
by 
  -- Placeholder proof
  sorry

end period_of_time_l70_70761


namespace can_all_mushrooms_become_good_l70_70730

def is_bad (w : Nat) : Prop := w ≥ 10
def is_good (w : Nat) : Prop := w < 10

def mushrooms_initially_bad := 90
def mushrooms_initially_good := 10

def total_mushrooms := mushrooms_initially_bad + mushrooms_initially_good
def total_worms_initial := mushrooms_initially_bad * 10

theorem can_all_mushrooms_become_good :
  ∃ worms_distribution : Fin total_mushrooms → Nat,
  (∀ i : Fin total_mushrooms, is_good (worms_distribution i)) :=
sorry

end can_all_mushrooms_become_good_l70_70730


namespace functional_equation_solution_l70_70453

theorem functional_equation_solution {f : ℚ → ℚ} :
  (∀ x y z t : ℚ, x < y ∧ y < z ∧ z < t ∧ (y - x) = (z - y) ∧ (z - y) = (t - z) →
    f x + f t = f y + f z) → 
  ∃ c b : ℚ, ∀ q : ℚ, f q = c * q + b := 
by
  sorry

end functional_equation_solution_l70_70453


namespace real_solutions_of_equation_l70_70959

theorem real_solutions_of_equation (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 12) ↔ (x = 13 ∨ x = -5) :=
by
  sorry

end real_solutions_of_equation_l70_70959


namespace sequence_and_sum_problems_l70_70360

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n-1) * d) / 2

def geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := b * r^(n-1)

noncomputable def sum_geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := 
(if r = 1 then b * n
 else b * (r^n - 1) / (r - 1))

theorem sequence_and_sum_problems :
  (∀ n : ℕ, arithmetic_sequence 19 (-2) n = 21 - 2 * n) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 19 (-2) n = 20 * n - n^2) ∧
  (∀ n : ℕ, ∃ a_n : ℤ, (geometric_sequence 1 3 n + (a_n - geometric_sequence 1 3 n) = 21 - 2 * n + 3^(n-1)) ∧
    sum_geometric_sequence 1 3 n = (sum_arithmetic_sequence 19 (-2) n + (3^n - 1) / 2))
:= by
  sorry

end sequence_and_sum_problems_l70_70360
