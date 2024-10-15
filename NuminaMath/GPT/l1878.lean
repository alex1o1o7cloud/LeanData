import Mathlib

namespace NUMINAMATH_GPT_ratio_of_pats_stick_not_covered_to_sarah_stick_l1878_187811

-- Defining the given conditions
def pat_stick_length : ℕ := 30
def dirt_covered : ℕ := 7
def jane_stick_length : ℕ := 22
def two_feet : ℕ := 24

-- Computing Sarah's stick length from Jane's stick length and additional two feet
def sarah_stick_length : ℕ := jane_stick_length + two_feet

-- Computing the portion of Pat's stick not covered in dirt
def portion_not_covered_in_dirt : ℕ := pat_stick_length - dirt_covered

-- The statement we need to prove
theorem ratio_of_pats_stick_not_covered_to_sarah_stick : 
  (portion_not_covered_in_dirt : ℚ) / (sarah_stick_length : ℚ) = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_ratio_of_pats_stick_not_covered_to_sarah_stick_l1878_187811


namespace NUMINAMATH_GPT_speed_of_first_boy_proof_l1878_187882

noncomputable def speed_of_first_boy := 5.9

theorem speed_of_first_boy_proof :
  ∀ (x : ℝ) (t : ℝ) (d : ℝ),
    (d = x * t) → (d = (x - 5.6) * 35) →
    d = 10.5 →
    t = 35 →
    x = 5.9 := 
by
  intros x t d h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_speed_of_first_boy_proof_l1878_187882


namespace NUMINAMATH_GPT_problem_inequality_l1878_187849

variables (a b c : ℝ)
open Real

theorem problem_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l1878_187849


namespace NUMINAMATH_GPT_pony_jeans_discount_rate_l1878_187886

noncomputable def fox_price : ℝ := 15
noncomputable def pony_price : ℝ := 18

-- Define the conditions
def total_savings (F P : ℝ) : Prop :=
  3 * (F / 100 * fox_price) + 2 * (P / 100 * pony_price) = 9

def discount_sum (F P : ℝ) : Prop :=
  F + P = 22

-- Main statement to be proven
theorem pony_jeans_discount_rate (F P : ℝ) (h1 : total_savings F P) (h2 : discount_sum F P) : P = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pony_jeans_discount_rate_l1878_187886


namespace NUMINAMATH_GPT_estimate_value_l1878_187864

theorem estimate_value : 1 < (3 - Real.sqrt 3) ∧ (3 - Real.sqrt 3) < 2 :=
by
  have h₁ : Real.sqrt 18 = 3 * Real.sqrt 2 :=
    by sorry
  have h₂ : Real.sqrt 6 = Real.sqrt 3 * Real.sqrt 2 :=
    by sorry
  have h₃ : (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 = (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 :=
    by sorry
  have h₄ : (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 = 3 - Real.sqrt 3 :=
    by sorry
  have h₅ : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 :=
    by sorry
  sorry

end NUMINAMATH_GPT_estimate_value_l1878_187864


namespace NUMINAMATH_GPT_broken_glass_pieces_l1878_187877

theorem broken_glass_pieces (x : ℕ) 
    (total_pieces : ℕ := 100) 
    (safe_fee : ℕ := 3) 
    (compensation : ℕ := 5) 
    (total_fee : ℕ := 260) 
    (h : safe_fee * (total_pieces - x) - compensation * x = total_fee) : x = 5 := by
  sorry

end NUMINAMATH_GPT_broken_glass_pieces_l1878_187877


namespace NUMINAMATH_GPT_product_of_p_and_q_l1878_187833

theorem product_of_p_and_q (p q : ℝ) (hpq_sum : p + q = 10) (hpq_cube_sum : p^3 + q^3 = 370) : p * q = 21 :=
by
  sorry

end NUMINAMATH_GPT_product_of_p_and_q_l1878_187833


namespace NUMINAMATH_GPT_union_complement_l1878_187834

open Set

-- Definitions for the universal set U and subsets A, B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}

-- Definition for the complement of A with respect to U
def CuA : Set ℕ := U \ A

-- Proof statement
theorem union_complement (U_def : U = {0, 1, 2, 3, 4})
                         (A_def : A = {0, 3, 4})
                         (B_def : B = {1, 3}) :
  (CuA ∪ B) = {1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_complement_l1878_187834


namespace NUMINAMATH_GPT_circle_equation_l1878_187856

def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem circle_equation : ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ∧ (y = parabola x) ∧ (x = -1 ∨ x = 3 ∨ (x = 0 ∧ y = -3)) :=
by { sorry }

end NUMINAMATH_GPT_circle_equation_l1878_187856


namespace NUMINAMATH_GPT_exists_divisible_triangle_l1878_187860

theorem exists_divisible_triangle (p : ℕ) (n : ℕ) (m : ℕ) (points : Fin m → ℤ × ℤ) 
  (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_pos : 0 < n) (hm_eight : m = 8) 
  (on_circle : ∀ k : Fin m, (points k).fst ^ 2 + (points k).snd ^ 2 = (p ^ n) ^ 2) :
  ∃ (i j k : Fin m), (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ (∃ d : ℕ, (points i).fst - (points j).fst = p ^ d ∧ 
  (points i).snd - (points j).snd = p ^ d ∧ d ≥ n + 1) :=
sorry

end NUMINAMATH_GPT_exists_divisible_triangle_l1878_187860


namespace NUMINAMATH_GPT_triangle_angle_relation_l1878_187892

theorem triangle_angle_relation 
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : b = (a + c) / Real.sqrt 2)
  (h2 : β = (α + γ) / 2)
  (h3 : c > a)
  : γ = α + 90 :=
sorry

end NUMINAMATH_GPT_triangle_angle_relation_l1878_187892


namespace NUMINAMATH_GPT_possible_m_values_l1878_187848

theorem possible_m_values (m : ℝ) :
  let A := {x : ℝ | mx - 1 = 0}
  let B := {2, 3}
  (A ⊆ B) → (m = 0 ∨ m = 1 / 2 ∨ m = 1 / 3) :=
by
  intro A B h
  sorry

end NUMINAMATH_GPT_possible_m_values_l1878_187848


namespace NUMINAMATH_GPT_solution_l1878_187831

-- Given conditions in the problem
def F (x : ℤ) : ℤ := sorry -- Placeholder for the polynomial with integer coefficients
variables (a : ℕ → ℤ) (m : ℕ)

-- Given that: ∀ n, ∃ k, F(n) is divisible by a(k) for some k in {1, 2, ..., m}
axiom forall_n_exists_k : ∀ n : ℤ, ∃ k : ℕ, k < m ∧ a k ∣ F n

-- Desired conclusion: ∃ k, ∀ n, F(n) is divisible by a(k)
theorem solution : ∃ k : ℕ, k < m ∧ (∀ n : ℤ, a k ∣ F n) :=
sorry

end NUMINAMATH_GPT_solution_l1878_187831


namespace NUMINAMATH_GPT_sum_of_digits_B_l1878_187866

/- 
  Let A be the natural number formed by concatenating integers from 1 to 100.
  Let B be the smallest possible natural number formed by removing 100 digits from A.
  We need to prove that the sum of the digits of B equals 486.
-/
def A : ℕ := sorry -- construct the natural number 1234567891011121314...99100

def sum_of_digits (n : ℕ) : ℕ := sorry -- function to calculate the sum of digits of a natural number

def B : ℕ := sorry -- construct the smallest possible number B by removing 100 digits from A

theorem sum_of_digits_B : sum_of_digits B = 486 := sorry

end NUMINAMATH_GPT_sum_of_digits_B_l1878_187866


namespace NUMINAMATH_GPT_inequality_solution_l1878_187862

theorem inequality_solution (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1878_187862


namespace NUMINAMATH_GPT_work_completion_l1878_187897

/-- 
  Let A, B, and C have work rates where:
  1. A completes the work in 4 days (work rate: 1/4 per day)
  2. C completes the work in 12 days (work rate: 1/12 per day)
  3. Together with B, they complete the work in 2 days (combined work rate: 1/2 per day)
  Prove that B alone can complete the work in 6 days.
--/
theorem work_completion (A B C : ℝ) (x : ℝ)
  (hA : A = 1/4)
  (hC : C = 1/12)
  (h_combined : A + 1/x + C = 1/2) :
  x = 6 := sorry

end NUMINAMATH_GPT_work_completion_l1878_187897


namespace NUMINAMATH_GPT_speed_of_second_train_l1878_187808

def speed_of_first_train := 40 -- speed of the first train in kmph
def distance_from_mumbai := 120 -- distance from Mumbai where the trains meet in km
def head_start_time := 1 -- head start time in hours for the first train
def total_remaining_distance := distance_from_mumbai - speed_of_first_train * head_start_time -- remaining distance for the first train to travel in km after head start
def time_to_meet_first_train := total_remaining_distance / speed_of_first_train -- time in hours for the first train to reach the meeting point after head start
def second_train_meeting_time := time_to_meet_first_train -- the second train takes the same time to meet the first train
def distance_covered_by_second_train := distance_from_mumbai -- same meeting point distance for second train from Mumbai

theorem speed_of_second_train : 
  ∃ v : ℝ, v = distance_covered_by_second_train / second_train_meeting_time ∧ v = 60 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_l1878_187808


namespace NUMINAMATH_GPT_articles_count_l1878_187888

noncomputable def cost_price_per_article : ℝ := 1
noncomputable def selling_price_per_article (x : ℝ) : ℝ := x / 16
noncomputable def profit : ℝ := 0.50

theorem articles_count (x : ℝ) (h1 : cost_price_per_article * x = selling_price_per_article x * 16)
                       (h2 : selling_price_per_article 16 = cost_price_per_article * (1 + profit)) :
  x = 24 :=
by
  sorry

end NUMINAMATH_GPT_articles_count_l1878_187888


namespace NUMINAMATH_GPT_linear_equation_in_options_l1878_187881

def is_linear_equation_with_one_variable (eqn : String) : Prop :=
  eqn = "3 - 2x = 5"

theorem linear_equation_in_options :
  is_linear_equation_with_one_variable "3 - 2x = 5" :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_in_options_l1878_187881


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1878_187872

theorem hyperbola_eccentricity (m : ℤ) (h1 : -2 < m) (h2 : m < 2) : 
  let a := m
  let b := (4 - m^2).sqrt 
  let c := (a^2 + b^2).sqrt
  let e := c / a
  e = 2 := by
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1878_187872


namespace NUMINAMATH_GPT_cousin_reading_time_l1878_187838

theorem cousin_reading_time (my_time_hours : ℕ) (speed_ratio : ℕ) (my_time_minutes := my_time_hours * 60) :
  (my_time_hours = 3) ∧ (speed_ratio = 5) → 
  (my_time_minutes / speed_ratio = 36) :=
by
  sorry

end NUMINAMATH_GPT_cousin_reading_time_l1878_187838


namespace NUMINAMATH_GPT_find_point_Q_l1878_187841

theorem find_point_Q {a b c : ℝ} 
  (h1 : ∀ x y z : ℝ, (x + 1)^2 + (y - 3)^2 + (z + 2)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) 
  (h2 : ∀ x y z: ℝ, 8 * x - 6 * y + 12 * z = 34) : 
  (a = 3) ∧ (b = -6) ∧ (c = 8) :=
by
  sorry

end NUMINAMATH_GPT_find_point_Q_l1878_187841


namespace NUMINAMATH_GPT_simplify_expression_l1878_187814

-- Define the given condition as a hypothesis
theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 :=
by
  sorry -- Proof will be provided here.

end NUMINAMATH_GPT_simplify_expression_l1878_187814


namespace NUMINAMATH_GPT_hydrocarbon_tree_configurations_l1878_187879

theorem hydrocarbon_tree_configurations (n : ℕ) 
  (h1 : 3 * n + 2 > 0) -- Total vertices count must be positive
  (h2 : 2 * n + 2 > 0) -- Leaves count must be positive
  (h3 : n > 0) -- Internal nodes count must be positive
  : (n:ℕ) ^ (n-2) = n ^ (n-2) :=
sorry

end NUMINAMATH_GPT_hydrocarbon_tree_configurations_l1878_187879


namespace NUMINAMATH_GPT_factorization_of_expression_l1878_187809

theorem factorization_of_expression (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) :=
by 
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l1878_187809


namespace NUMINAMATH_GPT_smallest_a_plus_b_l1878_187845

theorem smallest_a_plus_b 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : 2^10 * 3^5 = a^b) : a + b = 248833 :=
sorry

end NUMINAMATH_GPT_smallest_a_plus_b_l1878_187845


namespace NUMINAMATH_GPT_fraction_reduction_by_11_l1878_187839

theorem fraction_reduction_by_11 (k : ℕ) :
  (k^2 - 5 * k + 8) % 11 = 0 → 
  (k^2 + 6 * k + 19) % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_fraction_reduction_by_11_l1878_187839


namespace NUMINAMATH_GPT_infinite_powers_of_two_in_sequence_l1878_187868

theorem infinite_powers_of_two_in_sequence :
  ∃ᶠ n in at_top, ∃ k : ℕ, ∃ a : ℕ, (a = ⌊n * Real.sqrt 2⌋ ∧ a = 2^k) :=
sorry

end NUMINAMATH_GPT_infinite_powers_of_two_in_sequence_l1878_187868


namespace NUMINAMATH_GPT_min_fraction_value_l1878_187891

theorem min_fraction_value
    (a x y : ℕ)
    (h1 : a > 100)
    (h2 : x > 100)
    (h3 : y > 100)
    (h4 : y^2 - 1 = a^2 * (x^2 - 1))
    : a / x ≥ 2 := 
sorry

end NUMINAMATH_GPT_min_fraction_value_l1878_187891


namespace NUMINAMATH_GPT_translate_function_l1878_187803

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1

theorem translate_function :
  ∀ x : ℝ, f (x) = 2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_translate_function_l1878_187803


namespace NUMINAMATH_GPT_x_power_12_l1878_187885

theorem x_power_12 (x : ℝ) (h : x + 1 / x = 2) : x^12 = 1 :=
by sorry

end NUMINAMATH_GPT_x_power_12_l1878_187885


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l1878_187806

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l1878_187806


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l1878_187815

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l1878_187815


namespace NUMINAMATH_GPT_quotient_is_zero_l1878_187824

def square_mod_16 (n : ℕ) : ℕ :=
  (n * n) % 16

def distinct_remainders_in_range : List ℕ :=
  List.eraseDup $
    List.map square_mod_16 (List.range' 1 15)

def sum_of_distinct_remainders : ℕ :=
  distinct_remainders_in_range.sum

theorem quotient_is_zero :
  (sum_of_distinct_remainders / 16) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quotient_is_zero_l1878_187824


namespace NUMINAMATH_GPT_find_y_intercept_l1878_187822

theorem find_y_intercept (m b x y : ℝ) (h1 : m = 2) (h2 : (x, y) = (239, 480)) (line_eq : y = m * x + b) : b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_intercept_l1878_187822


namespace NUMINAMATH_GPT_molecular_weight_of_compound_is_correct_l1878_187836

noncomputable def molecular_weight (nC nH nN nO : ℕ) (wC wH wN wO : ℝ) :=
  nC * wC + nH * wH + nN * wN + nO * wO

theorem molecular_weight_of_compound_is_correct :
  molecular_weight 8 18 2 4 12.01 1.008 14.01 16.00 = 206.244 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_is_correct_l1878_187836


namespace NUMINAMATH_GPT_percentage_below_50000_l1878_187829

-- Define all the conditions
def cities_between_50000_and_100000 := 35 -- percentage
def cities_below_20000 := 45 -- percentage
def cities_between_20000_and_50000 := 10 -- percentage
def cities_above_100000 := 10 -- percentage

-- The proof statement
theorem percentage_below_50000 : 
    cities_below_20000 + cities_between_20000_and_50000 = 55 :=
by
    unfold cities_below_20000 cities_between_20000_and_50000
    sorry

end NUMINAMATH_GPT_percentage_below_50000_l1878_187829


namespace NUMINAMATH_GPT_ratio_of_x_y_l1878_187846

theorem ratio_of_x_y (x y : ℚ) (h : (2 * x - y) / (x + y) = 2 / 3) : x / y = 5 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_y_l1878_187846


namespace NUMINAMATH_GPT_bread_needed_for_sandwiches_l1878_187850

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end NUMINAMATH_GPT_bread_needed_for_sandwiches_l1878_187850


namespace NUMINAMATH_GPT_product_xyz_equals_zero_l1878_187878

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end NUMINAMATH_GPT_product_xyz_equals_zero_l1878_187878


namespace NUMINAMATH_GPT_number_of_packages_l1878_187800

-- Given conditions
def totalMarkers : ℕ := 40
def markersPerPackage : ℕ := 5

-- Theorem: Calculate the number of packages
theorem number_of_packages (totalMarkers: ℕ) (markersPerPackage: ℕ) : totalMarkers / markersPerPackage = 8 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_packages_l1878_187800


namespace NUMINAMATH_GPT_pirates_on_schooner_l1878_187861

def pirate_problem (N : ℝ) : Prop :=
  let total_pirates       := N
  let non_participants    := 10
  let participants        := total_pirates - non_participants
  let lost_arm            := 0.54 * participants
  let lost_arm_and_leg    := 0.34 * participants
  let lost_leg            := (2 / 3) * total_pirates
  -- The number of pirates who lost only a leg can be calculated.
  let lost_only_leg       := lost_leg - lost_arm_and_leg
  -- The equation that needs to be satisfied
  lost_leg = lost_arm_and_leg + lost_only_leg

theorem pirates_on_schooner : ∃ N : ℝ, N > 10 ∧ pirate_problem N :=
sorry

end NUMINAMATH_GPT_pirates_on_schooner_l1878_187861


namespace NUMINAMATH_GPT_negation_equiv_l1878_187857

variable (f : ℝ → ℝ)

theorem negation_equiv :
  ¬ (∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
sorry

end NUMINAMATH_GPT_negation_equiv_l1878_187857


namespace NUMINAMATH_GPT_combination_coins_l1878_187867

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end NUMINAMATH_GPT_combination_coins_l1878_187867


namespace NUMINAMATH_GPT_total_votes_proof_l1878_187810

noncomputable def total_votes (A : ℝ) (T : ℝ) := 0.40 * T = A
noncomputable def votes_in_favor (A : ℝ) := A + 68
noncomputable def total_votes_calc (T : ℝ) (Favor : ℝ) (A : ℝ) := T = Favor + A

theorem total_votes_proof (A T : ℝ) (Favor : ℝ) 
  (hA : total_votes A T) 
  (hFavor : votes_in_favor A = Favor) 
  (hT : total_votes_calc T Favor A) : 
  T = 340 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_proof_l1878_187810


namespace NUMINAMATH_GPT_paul_completion_time_l1878_187889

theorem paul_completion_time :
  let george_rate := 1 / 15
  let remaining_work := 2 / 5
  let combined_rate (P : ℚ) := george_rate + P
  let P_work := 4 * combined_rate P = remaining_work
  let paul_rate := 13 / 90
  let total_work := 1
  let time_paul_alone := total_work / paul_rate
  P_work → time_paul_alone = (90 / 13) := by
  intros
  -- all necessary definitions and conditions are used
  sorry

end NUMINAMATH_GPT_paul_completion_time_l1878_187889


namespace NUMINAMATH_GPT_alpha_quadrant_l1878_187813

variable {α : ℝ}

theorem alpha_quadrant
  (sin_alpha_neg : Real.sin α < 0)
  (tan_alpha_pos : Real.tan α > 0) :
  ∃ k : ℤ, k = 1 ∧ π < α - 2 * π * k ∧ α - 2 * π * k < 3 * π :=
by
  sorry

end NUMINAMATH_GPT_alpha_quadrant_l1878_187813


namespace NUMINAMATH_GPT_largest_possible_percent_error_l1878_187827

theorem largest_possible_percent_error
  (d : ℝ) (error_percent : ℝ) (actual_area : ℝ)
  (h_d : d = 30) (h_error_percent : error_percent = 0.1)
  (h_actual_area : actual_area = 225 * Real.pi) :
  ∃ max_error_percent : ℝ,
    (max_error_percent = 21) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_percent_error_l1878_187827


namespace NUMINAMATH_GPT_total_distance_run_l1878_187893

def track_meters : ℕ := 9
def laps_already_run : ℕ := 6
def laps_to_run : ℕ := 5

theorem total_distance_run :
  (laps_already_run * track_meters) + (laps_to_run * track_meters) = 99 := by
  sorry

end NUMINAMATH_GPT_total_distance_run_l1878_187893


namespace NUMINAMATH_GPT_vertical_asymptote_once_l1878_187859

theorem vertical_asymptote_once (c : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + c) / (x^2 - x - 12) = (x^2 + 2*x + c) / ((x - 4) * (x + 3))) → 
  (c = -24 ∨ c = -3) :=
by 
  sorry

end NUMINAMATH_GPT_vertical_asymptote_once_l1878_187859


namespace NUMINAMATH_GPT_product_sin_eq_one_eighth_l1878_187802

theorem product_sin_eq_one_eighth (h1 : Real.sin (3 * Real.pi / 8) = Real.cos (Real.pi / 8))
                                  (h2 : Real.sin (Real.pi / 8) = Real.cos (3 * Real.pi / 8)) :
  ((1 - Real.sin (Real.pi / 8)) * (1 - Real.sin (3 * Real.pi / 8)) * 
   (1 + Real.sin (Real.pi / 8)) * (1 + Real.sin (3 * Real.pi / 8)) = 1 / 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_product_sin_eq_one_eighth_l1878_187802


namespace NUMINAMATH_GPT_sum_q_p_values_l1878_187853

def p (x : ℤ) : ℤ := x^2 - 4
def q (x : ℤ) : ℤ := -x

def q_p_composed (x : ℤ) : ℤ := q (p x)

theorem sum_q_p_values :
  q_p_composed (-3) + q_p_composed (-2) + q_p_composed (-1) + q_p_composed 0 + 
  q_p_composed 1 + q_p_composed 2 + q_p_composed 3 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_q_p_values_l1878_187853


namespace NUMINAMATH_GPT_arithmetic_sequence_term_count_l1878_187854

theorem arithmetic_sequence_term_count (a1 d an : ℤ) (h₀ : a1 = -6) (h₁ : d = 5) (h₂ : an = 59) :
  ∃ n : ℤ, an = a1 + (n - 1) * d ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_count_l1878_187854


namespace NUMINAMATH_GPT_circles_chord_length_l1878_187855

theorem circles_chord_length (r1 r2 r3 : ℕ) (m n p : ℕ) (h1 : r1 = 4) (h2 : r2 = 10) (h3 : r3 = 14)
(h4 : gcd m p = 1) (h5 : ¬ (∃ (k : ℕ), k^2 ∣ n)) : m + n + p = 19 :=
by
  sorry

end NUMINAMATH_GPT_circles_chord_length_l1878_187855


namespace NUMINAMATH_GPT_thabo_total_books_l1878_187801

noncomputable def total_books (H PNF PF : ℕ) : ℕ := H + PNF + PF

theorem thabo_total_books :
  ∀ (H PNF PF : ℕ),
    H = 30 →
    PNF = H + 20 →
    PF = 2 * PNF →
    total_books H PNF PF = 180 :=
by
  intros H PNF PF hH hPNF hPF
  sorry

end NUMINAMATH_GPT_thabo_total_books_l1878_187801


namespace NUMINAMATH_GPT_sum_difference_l1878_187863

def even_sum (n : ℕ) : ℕ :=
  n * (n + 1)

def odd_sum (n : ℕ) : ℕ :=
  n^2

theorem sum_difference : even_sum 100 - odd_sum 100 = 100 := by
  sorry

end NUMINAMATH_GPT_sum_difference_l1878_187863


namespace NUMINAMATH_GPT_average_income_eq_58_l1878_187821

def income_day1 : ℕ := 45
def income_day2 : ℕ := 50
def income_day3 : ℕ := 60
def income_day4 : ℕ := 65
def income_day5 : ℕ := 70
def number_of_days : ℕ := 5

theorem average_income_eq_58 :
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / number_of_days = 58 := by
  sorry

end NUMINAMATH_GPT_average_income_eq_58_l1878_187821


namespace NUMINAMATH_GPT_library_visits_l1878_187823

theorem library_visits
  (william_visits_per_week : ℕ := 2)
  (jason_visits_per_week : ℕ := 4 * william_visits_per_week)
  (emma_visits_per_week : ℕ := 3 * jason_visits_per_week)
  (zoe_visits_per_week : ℕ := william_visits_per_week / 2)
  (chloe_visits_per_week : ℕ := emma_visits_per_week / 3)
  (jason_total_visits : ℕ := jason_visits_per_week * 8)
  (emma_total_visits : ℕ := emma_visits_per_week * 8)
  (zoe_total_visits : ℕ := zoe_visits_per_week * 8)
  (chloe_total_visits : ℕ := chloe_visits_per_week * 8)
  (total_visits : ℕ := jason_total_visits + emma_total_visits + zoe_total_visits + chloe_total_visits) :
  total_visits = 328 := by
  sorry

end NUMINAMATH_GPT_library_visits_l1878_187823


namespace NUMINAMATH_GPT_species_below_threshold_in_year_2019_l1878_187828

-- Definitions based on conditions in the problem.
def initial_species (N : ℝ) : ℝ := N
def yearly_decay_rate : ℝ := 0.70
def threshold : ℝ := 0.05

-- The problem statement to prove.
theorem species_below_threshold_in_year_2019 (N : ℝ) (hN : N > 0):
  ∃ k : ℕ, k ≥ 9 ∧ yearly_decay_rate ^ k * initial_species N < threshold * initial_species N :=
sorry

end NUMINAMATH_GPT_species_below_threshold_in_year_2019_l1878_187828


namespace NUMINAMATH_GPT_ratio_of_a_plus_b_to_b_plus_c_l1878_187837

variable (a b c : ℝ)

theorem ratio_of_a_plus_b_to_b_plus_c (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_plus_b_to_b_plus_c_l1878_187837


namespace NUMINAMATH_GPT_vector_c_correct_l1878_187875

theorem vector_c_correct (a b c : ℤ × ℤ) (h_a : a = (1, -3)) (h_b : b = (-2, 4))
    (h_condition : 4 • a + (3 • b - 2 • a) + c = (0, 0)) :
    c = (4, -6) :=
by 
  -- The proof steps go here, but we'll skip them with 'sorry' for now.
  sorry

end NUMINAMATH_GPT_vector_c_correct_l1878_187875


namespace NUMINAMATH_GPT_sum_of_volumes_of_two_cubes_l1878_187835

-- Definitions for edge length and volume formula
def edge_length : ℕ := 5

def volume (s : ℕ) : ℕ := s ^ 3

-- Statement to prove the sum of volumes of two cubes with edge length 5 cm
theorem sum_of_volumes_of_two_cubes : volume edge_length + volume edge_length = 250 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_volumes_of_two_cubes_l1878_187835


namespace NUMINAMATH_GPT_toys_produced_on_sunday_l1878_187816

-- Given conditions
def factory_production (day: ℕ) : ℕ :=
  2500 + 25 * day

theorem toys_produced_on_sunday : factory_production 6 = 2650 :=
by {
  -- The proof steps are omitted as they are not required.
  sorry
}

end NUMINAMATH_GPT_toys_produced_on_sunday_l1878_187816


namespace NUMINAMATH_GPT_local_maximum_at_neg2_l1878_187887

noncomputable def y (x : ℝ) : ℝ :=
  (1/3) * x^3 - 4 * x + 4

theorem local_maximum_at_neg2 :
  ∃ x : ℝ, x = -2 ∧ 
           y x = 28/3 ∧
           (∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 2) < δ → y z < y (-2)) := by
  sorry

end NUMINAMATH_GPT_local_maximum_at_neg2_l1878_187887


namespace NUMINAMATH_GPT_no_integer_solutions_19x2_minus_76y2_eq_1976_l1878_187805

theorem no_integer_solutions_19x2_minus_76y2_eq_1976 :
  ∀ x y : ℤ, 19 * x^2 - 76 * y^2 ≠ 1976 :=
by sorry

end NUMINAMATH_GPT_no_integer_solutions_19x2_minus_76y2_eq_1976_l1878_187805


namespace NUMINAMATH_GPT_find_numbers_l1878_187825

theorem find_numbers (p q x : ℝ) (h : (p ≠ 1)) :
  ((p * x) ^ 2 - x ^ 2) / (p * x + x) = q ↔ x = q / (p - 1) ∧ p * x = (p * q) / (p - 1) := 
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1878_187825


namespace NUMINAMATH_GPT_dice_product_probability_is_one_l1878_187873

def dice_probability_product_is_one : Prop :=
  ∀ (a b c d e : ℕ), (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 → 
    (a * b * c * d * e) = 1) ∧
  ∃ (p : ℚ), p = (1/6)^5 ∧ p = 1/7776

theorem dice_product_probability_is_one (a b c d e : ℕ) :
  dice_probability_product_is_one :=
by
  sorry

end NUMINAMATH_GPT_dice_product_probability_is_one_l1878_187873


namespace NUMINAMATH_GPT_radius_of_circle_l1878_187852

theorem radius_of_circle (d : ℝ) (h : d = 22) : (d / 2) = 11 := by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1878_187852


namespace NUMINAMATH_GPT_range_of_a_l1878_187832

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 4 * a > x^2 - x^3) → a > 1 / 27 :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_range_of_a_l1878_187832


namespace NUMINAMATH_GPT_determine_position_correct_l1878_187826

def determine_position (option : String) : Prop :=
  option = "East longitude 120°, North latitude 30°"

theorem determine_position_correct :
  determine_position "East longitude 120°, North latitude 30°" :=
by
  sorry

end NUMINAMATH_GPT_determine_position_correct_l1878_187826


namespace NUMINAMATH_GPT_trigonometric_expression_l1878_187844

noncomputable def cosθ (θ : ℝ) := 1 / Real.sqrt 10
noncomputable def sinθ (θ : ℝ) := 3 / Real.sqrt 10
noncomputable def tanθ (θ : ℝ) := 3

theorem trigonometric_expression (θ : ℝ) (h : tanθ θ = 3) :
  (1 + cosθ θ) / sinθ θ + sinθ θ / (1 - cosθ θ) = (10 * Real.sqrt 10 + 10) / 9 := 
  sorry

end NUMINAMATH_GPT_trigonometric_expression_l1878_187844


namespace NUMINAMATH_GPT_f_2000_equals_1499001_l1878_187880

noncomputable def f (x : ℕ) : ℝ → ℝ := sorry

axiom f_initial : f 0 = 1

axiom f_recursive (x : ℕ) : f (x + 4) = f x + 3 * x + 4

theorem f_2000_equals_1499001 : f 2000 = 1499001 :=
by sorry

end NUMINAMATH_GPT_f_2000_equals_1499001_l1878_187880


namespace NUMINAMATH_GPT_aaron_brothers_l1878_187894

theorem aaron_brothers (A : ℕ) (h1 : 6 = 2 * A - 2) : A = 4 :=
by
  sorry

end NUMINAMATH_GPT_aaron_brothers_l1878_187894


namespace NUMINAMATH_GPT_division_of_decimals_l1878_187883

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end NUMINAMATH_GPT_division_of_decimals_l1878_187883


namespace NUMINAMATH_GPT_chessboard_cover_l1878_187804

open Nat

/-- 
  For an m × n chessboard, after removing any one small square, it can always be completely covered
  with L-shaped tiles if and only if 3 divides (mn - 1) and min(m,n) is not equal to 1, 2, 5 or m=n=2.
-/
theorem chessboard_cover (m n : ℕ) :
  (∃ k : ℕ, 3 * k = m * n - 1) ∧ (min m n ≠ 1 ∧ min m n ≠ 2 ∧ min m n ≠ 5 ∨ m = 2 ∧ n = 2) :=
sorry

end NUMINAMATH_GPT_chessboard_cover_l1878_187804


namespace NUMINAMATH_GPT_correct_option_is_B_l1878_187842

def natural_growth_rate (birth_rate death_rate : ℕ) : ℕ :=
  birth_rate - death_rate

def option_correct (birth_rate death_rate : ℕ) :=
  (∃ br dr, natural_growth_rate br dr = br - dr)

theorem correct_option_is_B (birth_rate death_rate : ℕ) :
  option_correct birth_rate death_rate :=
by 
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l1878_187842


namespace NUMINAMATH_GPT_roots_polynomial_sum_l1878_187830

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end NUMINAMATH_GPT_roots_polynomial_sum_l1878_187830


namespace NUMINAMATH_GPT_james_monthly_earnings_l1878_187899

theorem james_monthly_earnings :
  let initial_subscribers := 150
  let gifted_subscribers := 50
  let rate_per_subscriber := 9
  let total_subscribers := initial_subscribers + gifted_subscribers
  let total_earnings := total_subscribers * rate_per_subscriber
  total_earnings = 1800 := by
  sorry

end NUMINAMATH_GPT_james_monthly_earnings_l1878_187899


namespace NUMINAMATH_GPT_find_constants_l1878_187890

def f (x : ℝ) (a : ℝ) : ℝ := 2 * x ^ 3 + a * x
def g (x : ℝ) (b c : ℝ) : ℝ := b * x ^ 2 + c
def f' (x : ℝ) (a : ℝ) : ℝ := 6 * x ^ 2 + a
def g' (x : ℝ) (b : ℝ) : ℝ := 2 * b * x

theorem find_constants (a b c : ℝ) :
  f 2 a = 0 ∧ g 2 b c = 0 ∧ f' 2 a = g' 2 b →
  a = -8 ∧ b = 4 ∧ c = -16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_constants_l1878_187890


namespace NUMINAMATH_GPT_P_lt_Q_l1878_187898

variable {x : ℝ}

def P (x : ℝ) : ℝ := (x - 2) * (x - 4)
def Q (x : ℝ) : ℝ := (x - 3) ^ 2

theorem P_lt_Q : P x < Q x := by
  sorry

end NUMINAMATH_GPT_P_lt_Q_l1878_187898


namespace NUMINAMATH_GPT_cos_value_l1878_187807

theorem cos_value {α : ℝ} (h : Real.sin (π / 6 + α) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_cos_value_l1878_187807


namespace NUMINAMATH_GPT_perfect_square_expression_l1878_187851

theorem perfect_square_expression (x y : ℝ) (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y, f x = f y → 4 * x^2 - (k - 1) * x * y + 9 * y^2 = (f x) ^ 2) ↔ (k = 13 ∨ k = -11) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l1878_187851


namespace NUMINAMATH_GPT_value_of_a_l1878_187847

theorem value_of_a (a : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 1, 2}) 
  (hB : B = {a + 1, a ^ 2 + 3}) 
  (h_inter : A ∩ B = {2}) : 
  a = 1 := 
by sorry

end NUMINAMATH_GPT_value_of_a_l1878_187847


namespace NUMINAMATH_GPT_large_A_exists_l1878_187865

noncomputable def F_n (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem large_A_exists : ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  ∀ a : ℕ, a ≤ 53590 → 
  F_n n6 (F_n n5 (F_n n4 (F_n n3 (F_n n2 (F_n n1 a))))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_large_A_exists_l1878_187865


namespace NUMINAMATH_GPT_lawnmower_blades_l1878_187871

theorem lawnmower_blades (B : ℤ) (h : 8 * B + 7 = 39) : B = 4 :=
by 
  sorry

end NUMINAMATH_GPT_lawnmower_blades_l1878_187871


namespace NUMINAMATH_GPT_circle_intersection_probability_l1878_187840

noncomputable def probability_circles_intersect : ℝ :=
  1

theorem circle_intersection_probability :
  ∀ (A_X B_X : ℝ), (0 ≤ A_X) → (A_X ≤ 2) → (0 ≤ B_X) → (B_X ≤ 2) →
  (∃ y, y ≥ 1 ∧ y ≤ 2) →
  ∃ p : ℝ, p = probability_circles_intersect ∧
  p = 1 :=
by
  sorry

end NUMINAMATH_GPT_circle_intersection_probability_l1878_187840


namespace NUMINAMATH_GPT_find_positive_int_sol_l1878_187884

theorem find_positive_int_sol (a b c d n : ℕ) (h1 : n > 1) (h2 : a ≤ b) (h3 : b ≤ c) :
  ((n^a + n^b + n^c = n^d) ↔ 
  ((a = b ∧ b = c - 1 ∧ c = d - 1 ∧ n = 2) ∨ 
  (a = b ∧ b = c ∧ c = d - 1 ∧ n = 3))) :=
  sorry

end NUMINAMATH_GPT_find_positive_int_sol_l1878_187884


namespace NUMINAMATH_GPT_find_k_l1878_187896

theorem find_k (k : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, 3 * x - k * y + c = 0) ∧ (∀ x y : ℝ, k * x + y + 1 = 0 → 3 * k + (-k) = 0) → k = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1878_187896


namespace NUMINAMATH_GPT_fifteenth_term_ratio_l1878_187895

noncomputable def U (n : ℕ) (c f : ℚ) := n * (2 * c + (n - 1) * f) / 2
noncomputable def V (n : ℕ) (g h : ℚ) := n * (2 * g + (n - 1) * h) / 2

theorem fifteenth_term_ratio (c f g h : ℚ)
  (h1 : ∀ n : ℕ, (n > 0) → (U n c f) / (V n g h) = (5 * (n * n) + 3 * n + 2) / (3 * (n * n) + 2 * n + 30)) :
  (c + 14 * f) / (g + 14 * h) = 125 / 99 :=
by
  sorry

end NUMINAMATH_GPT_fifteenth_term_ratio_l1878_187895


namespace NUMINAMATH_GPT_total_athletes_l1878_187874

theorem total_athletes (g : ℕ) (p : ℕ)
  (h₁ : g = 7)
  (h₂ : p = 5)
  (h₃ : 3 * (g + p - 1) = 33) : 
  3 * (g + p - 1) = 33 :=
sorry

end NUMINAMATH_GPT_total_athletes_l1878_187874


namespace NUMINAMATH_GPT_pow_2023_eq_one_or_neg_one_l1878_187870

theorem pow_2023_eq_one_or_neg_one (x : ℂ) (h : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) : 
  x^2023 = 1 ∨ x^2023 = -1 := 
by 
{
  sorry
}

end NUMINAMATH_GPT_pow_2023_eq_one_or_neg_one_l1878_187870


namespace NUMINAMATH_GPT_lcm_18_24_l1878_187812

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_GPT_lcm_18_24_l1878_187812


namespace NUMINAMATH_GPT_median_circumradius_altitude_inequality_l1878_187869

variable (h R m_a m_b m_c : ℝ)

-- Define the condition for the lengths of the medians and other related parameters
-- m_a, m_b, m_c are medians, R is the circumradius, h is the greatest altitude

theorem median_circumradius_altitude_inequality :
  m_a + m_b + m_c ≤ 3 * R + h :=
sorry

end NUMINAMATH_GPT_median_circumradius_altitude_inequality_l1878_187869


namespace NUMINAMATH_GPT_kevin_food_expenditure_l1878_187858

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end NUMINAMATH_GPT_kevin_food_expenditure_l1878_187858


namespace NUMINAMATH_GPT_jonathan_tax_per_hour_l1878_187876

-- Given conditions
def wage : ℝ := 25          -- wage in dollars per hour
def tax_rate : ℝ := 0.024    -- tax rate in decimal

-- Prove statement
theorem jonathan_tax_per_hour :
  (wage * 100) * tax_rate = 60 :=
sorry

end NUMINAMATH_GPT_jonathan_tax_per_hour_l1878_187876


namespace NUMINAMATH_GPT_symmetric_point_R_l1878_187843

variable (a b : ℝ) 

def symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def symmetry_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_R :
  let M := (a, b)
  let N := symmetry_x M
  let P := symmetry_y N
  let Q := symmetry_x P
  let R := symmetry_y Q
  R = (a, b) := by
  unfold symmetry_x symmetry_y
  sorry

end NUMINAMATH_GPT_symmetric_point_R_l1878_187843


namespace NUMINAMATH_GPT_total_oranges_l1878_187817

theorem total_oranges (a b c : ℕ) 
  (h₁ : a = 22) 
  (h₂ : b = a + 17) 
  (h₃ : c = b - 11) : 
  a + b + c = 89 := 
by
  sorry

end NUMINAMATH_GPT_total_oranges_l1878_187817


namespace NUMINAMATH_GPT_nancy_small_gardens_l1878_187819

theorem nancy_small_gardens (total_seeds big_garden_seeds small_garden_seed_count : ℕ) 
    (h1 : total_seeds = 52) 
    (h2 : big_garden_seeds = 28) 
    (h3 : small_garden_seed_count = 4) : 
    (total_seeds - big_garden_seeds) / small_garden_seed_count = 6 := by 
    sorry

end NUMINAMATH_GPT_nancy_small_gardens_l1878_187819


namespace NUMINAMATH_GPT_coins_after_10_hours_l1878_187820

def numberOfCoinsRemaining : Nat :=
  let hour1_coins := 20
  let hour2_coins := hour1_coins + 30
  let hour3_coins := hour2_coins + 30
  let hour4_coins := hour3_coins + 40
  let hour5_coins := hour4_coins - (hour4_coins * 20 / 100)
  let hour6_coins := hour5_coins + 50
  let hour7_coins := hour6_coins + 60
  let hour8_coins := hour7_coins - (hour7_coins / 5)
  let hour9_coins := hour8_coins + 70
  let hour10_coins := hour9_coins - (hour9_coins * 15 / 100)
  hour10_coins

theorem coins_after_10_hours : numberOfCoinsRemaining = 200 := by
  sorry

end NUMINAMATH_GPT_coins_after_10_hours_l1878_187820


namespace NUMINAMATH_GPT_k_value_if_perfect_square_l1878_187818

theorem k_value_if_perfect_square (a k : ℝ) (h : ∃ b : ℝ, a^2 + 2*k*a + 1 = (a + b)^2) : k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_GPT_k_value_if_perfect_square_l1878_187818
