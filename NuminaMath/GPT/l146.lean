import Mathlib

namespace total_distance_proof_l146_146665

-- Define the conditions
def amoli_speed : ℕ := 42      -- Amoli's speed in miles per hour
def amoli_time : ℕ := 3        -- Amoli's driving time in hours
def anayet_speed : ℕ := 61     -- Anayet's speed in miles per hour
def anayet_time : ℕ := 2       -- Anayet's driving time in hours
def remaining_distance : ℕ := 121  -- Remaining distance to be traveled in miles

-- Total distance calculation
def total_distance : ℕ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

-- The theorem to prove
theorem total_distance_proof : total_distance = 369 :=
by
  -- Proof goes here
  sorry

end total_distance_proof_l146_146665


namespace geometric_sequence_sum_l146_146673

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom a1 : (a 1) = 1
axiom a2 : ∀ (n : ℕ), n ≥ 2 → 2 * a (n + 1) + 2 * a (n - 1) = 5 * a n
axiom increasing : ∀ (n m : ℕ), n < m → a n < a m

-- Target
theorem geometric_sequence_sum : S 5 = 31 := by
  sorry

end geometric_sequence_sum_l146_146673


namespace find_tangent_equal_l146_146080

theorem find_tangent_equal (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180)) : n = 75 :=
sorry

end find_tangent_equal_l146_146080


namespace price_reduction_proof_l146_146466

theorem price_reduction_proof (x : ℝ) : 256 * (1 - x) ^ 2 = 196 :=
sorry

end price_reduction_proof_l146_146466


namespace lara_total_space_larger_by_1500_square_feet_l146_146174

theorem lara_total_space_larger_by_1500_square_feet :
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  total_area - area_square = 1500 :=
by
  -- Definitions
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  
  -- Calculation
  have h_area_rect : area_rect = 1500 := by
    norm_num [area_rect, length_rect, width_rect]

  have h_area_square : area_square = 2500 := by
    norm_num [area_square, side_square]

  have h_total_area : total_area = 4000 := by
    norm_num [total_area, h_area_rect, h_area_square]

  -- Final comparison
  have h_difference : total_area - area_square = 1500 := by
    norm_num [total_area, area_square, h_area_square]

  exact h_difference

end lara_total_space_larger_by_1500_square_feet_l146_146174


namespace total_items_in_quiz_l146_146016

theorem total_items_in_quiz (score_percent : ℝ) (mistakes : ℕ) (total_items : ℕ) 
  (h1 : score_percent = 80) 
  (h2 : mistakes = 5) :
  total_items = 25 :=
sorry

end total_items_in_quiz_l146_146016


namespace min_t_of_BE_CF_l146_146033

theorem min_t_of_BE_CF (A B C E F: ℝ)
  (hE_midpoint_AC : ∃ D, D = (A + C) / 2 ∧ E = D)
  (hF_midpoint_AB : ∃ D, D = (A + B) / 2 ∧ F = D)
  (h_AB_AC_ratio : B - A = 2 / 3 * (C - A)) :
  ∃ t : ℝ, t = 7 / 8 ∧ ∀ (BE CF : ℝ), BE = dist B E ∧ CF = dist C F → BE / CF < t := by
  sorry

end min_t_of_BE_CF_l146_146033


namespace book_sale_total_amount_l146_146371

noncomputable def total_amount_received (total_books price_per_book : ℕ → ℝ) : ℝ :=
  price_per_book 80

theorem book_sale_total_amount (B : ℕ)
  (h1 : (1/3 : ℚ) * B = 40)
  (h2 : ∀ (n : ℕ), price_per_book n = 3.50) :
  total_amount_received B price_per_book = 280 := 
by
  sorry

end book_sale_total_amount_l146_146371


namespace lcm_12_21_30_l146_146996

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end lcm_12_21_30_l146_146996


namespace circle_equation_l146_146752

open Real

variable {x y : ℝ}

theorem circle_equation (a : ℝ) (h_a_positive : a > 0) 
    (h_tangent : abs (3 * a + 4) / sqrt (3^2 + 4^2) = 2) :
    (∀ x y : ℝ, (x - a)^2 + y^2 = 4) := sorry

end circle_equation_l146_146752


namespace complex_modulus_inequality_l146_146682

theorem complex_modulus_inequality (z : ℂ) : (‖z‖ ^ 2 + 2 * ‖z - 1‖) ≥ 1 :=
by
  sorry

end complex_modulus_inequality_l146_146682


namespace martha_black_butterflies_l146_146561

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies : ℕ)
  (h1 : total_butterflies = 11)
  (h2 : blue_butterflies = 4)
  (h3 : blue_butterflies = 2 * yellow_butterflies) :
  ∃ black_butterflies : ℕ, black_butterflies = total_butterflies - blue_butterflies - yellow_butterflies :=
sorry

end martha_black_butterflies_l146_146561


namespace cost_per_piece_l146_146836

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end cost_per_piece_l146_146836


namespace total_animals_after_addition_l146_146736

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end total_animals_after_addition_l146_146736


namespace joan_half_dollars_spent_on_wednesday_l146_146914

variable (x : ℝ)
variable (h1 : x * 0.5 + 14 * 0.5 = 9)

theorem joan_half_dollars_spent_on_wednesday :
  x = 4 :=
by
  -- The proof is not required, hence using sorry
  sorry

end joan_half_dollars_spent_on_wednesday_l146_146914


namespace total_white_balls_l146_146075

theorem total_white_balls : ∃ W R B : ℕ,
  W + R = 300 ∧ B = 100 ∧
  ∃ (bw1 bw2 rw3 rw W3 : ℕ),
  bw1 = 27 ∧
  rw3 + rw = 42 ∧
  W3 = rw ∧
  B = bw1 + W3 + rw3 + bw2 ∧
  W = bw1 + 2 * bw2 + 3 * W3 ∧
  R = 3 * rw3 + rw ∧
  W = 158 :=
by
  sorry

end total_white_balls_l146_146075


namespace concentric_circles_ratio_l146_146917

theorem concentric_circles_ratio (d1 d2 d3 : ℝ) (h1 : d1 = 2) (h2 : d2 = 4) (h3 : d3 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let r3 := d3 / 2
  let A_red := π * r1 ^ 2
  let A_middle := π * r2 ^ 2
  let A_large := π * r3 ^ 2
  let A_blue := A_middle - A_red
  let A_green := A_large - A_middle
  (A_green / A_blue) = 5 / 3 := 
by
  sorry

end concentric_circles_ratio_l146_146917


namespace father_l146_146671

-- Conditions definitions
def man's_current_age (F : ℕ) : ℕ := (2 / 5) * F
def man_after_5_years (M F : ℕ) : Prop := M + 5 = (1 / 2) * (F + 5)

-- Main statement to prove
theorem father's_age (F : ℕ) (h₁ : man's_current_age F = (2 / 5) * F)
  (h₂ : ∀ M, man_after_5_years M F → M = (2 / 5) * F + 5): F = 25 :=
sorry

end father_l146_146671


namespace tournament_games_l146_146412

theorem tournament_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 5) : 
  (n * (n - 1) / 2) * k = 2175 := by
  sorry

end tournament_games_l146_146412


namespace proof_equivalent_problem_l146_146842

variables (a b c : ℝ)
-- Conditions
axiom h1 : a < b
axiom h2 : b < 0
axiom h3 : c > 0

theorem proof_equivalent_problem :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end proof_equivalent_problem_l146_146842


namespace find_n_l146_146641

theorem find_n : ∃ n : ℕ, 50^4 + 43^4 + 36^4 + 6^4 = n^4 := by
  sorry

end find_n_l146_146641


namespace man_l146_146256

theorem man's_speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h_current_speed : current_speed = 5) (h_against_current_speed : against_current_speed = 12) 
  (h_v : v - current_speed = against_current_speed) : 
  v + current_speed = 22 := 
by
  sorry

end man_l146_146256


namespace vector_subtraction_proof_l146_146654

theorem vector_subtraction_proof (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
    3 • b - a = (-3, -5) := by
  sorry

end vector_subtraction_proof_l146_146654


namespace single_point_graph_d_l146_146442

theorem single_point_graph_d (d : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + d = 0 ↔ x = -1 ∧ y = 6) → d = 39 :=
by 
  sorry

end single_point_graph_d_l146_146442


namespace gcd_90_150_l146_146130

theorem gcd_90_150 : Int.gcd 90 150 = 30 := 
by sorry

end gcd_90_150_l146_146130


namespace arithmetic_seq_term_six_l146_146202

theorem arithmetic_seq_term_six {a : ℕ → ℝ} (a1 : ℝ) (S3 : ℝ) (h1 : a1 = 2) (h2 : S3 = 12) :
  a 6 = 12 :=
sorry

end arithmetic_seq_term_six_l146_146202


namespace largest_class_is_28_l146_146902

-- definition and conditions
def largest_class_students (x : ℕ) : Prop :=
  let total_students := x + (x - 2) + (x - 4) + (x - 6) + (x - 8)
  total_students = 120

-- statement to prove
theorem largest_class_is_28 : ∃ x : ℕ, largest_class_students x ∧ x = 28 :=
by
  sorry

end largest_class_is_28_l146_146902


namespace min_buses_needed_l146_146575

theorem min_buses_needed (n : ℕ) : 325 / 45 ≤ n ∧ n < 325 / 45 + 1 ↔ n = 8 :=
by
  sorry

end min_buses_needed_l146_146575


namespace min_value_expression_l146_146543

theorem min_value_expression :
  (∀ y : ℝ, abs y ≤ 1 → ∃ x : ℝ, 2 * x + y = 1 ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1 → 
    (∃ y : ℝ, 2 * x + y = 1 ∧ abs y ≤ 1 ∧ (2 * x ^ 2 + 16 * x + 3 * y ^ 2) = 3))) :=
sorry

end min_value_expression_l146_146543


namespace phase_shift_correct_l146_146037

-- Given the function y = 3 * sin (x - π / 5)
-- We need to prove that the phase shift is π / 5.

theorem phase_shift_correct :
  ∀ x : ℝ, 3 * Real.sin (x - Real.pi / 5) = 3 * Real.sin (x - C) →
  C = Real.pi / 5 :=
by
  sorry

end phase_shift_correct_l146_146037


namespace find_normal_monthly_charge_l146_146967

-- Define the conditions
def normal_monthly_charge (x : ℕ) : Prop :=
  let first_month_charge := x / 3
  let fourth_month_charge := x + 15
  let other_months_charge := 4 * x
  (first_month_charge + fourth_month_charge + other_months_charge = 175)

-- The statement to prove
theorem find_normal_monthly_charge : ∃ x : ℕ, normal_monthly_charge x ∧ x = 30 := by
  sorry

end find_normal_monthly_charge_l146_146967


namespace range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l146_146084

variable (m : ℝ)

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Problem 1: Prove the range of m if B ⊆ A is (-∞, 3]
theorem range_m_if_B_subset_A : (set_B m ⊆ set_A) ↔ m ≤ 3 := sorry

-- Problem 2: Prove the range of m if A ∩ B = ∅ is m < 2 or m > 4
theorem range_m_if_A_inter_B_empty : (set_A ∩ set_B m = ∅) ↔ m < 2 ∨ m > 4 := sorry

end range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l146_146084


namespace flour_per_new_base_is_one_fifth_l146_146134

def total_flour : ℚ := 40 * (1 / 8)

def flour_per_new_base (p : ℚ) (total_flour : ℚ) : ℚ := total_flour / p

theorem flour_per_new_base_is_one_fifth :
  flour_per_new_base 25 total_flour = 1 / 5 :=
by
  sorry

end flour_per_new_base_is_one_fifth_l146_146134


namespace least_multiple_of_25_gt_390_l146_146741

theorem least_multiple_of_25_gt_390 : ∃ n : ℕ, n * 25 > 390 ∧ (∀ m : ℕ, m * 25 > 390 → m * 25 ≥ n * 25) ∧ n * 25 = 400 :=
by
  sorry

end least_multiple_of_25_gt_390_l146_146741


namespace Fk_same_implies_eq_l146_146030

def Q (n: ℕ) : ℕ :=
  -- Implementation of the square part of n
  sorry

def N (n: ℕ) : ℕ :=
  -- Implementation of the non-square part of n
  sorry

def Fk (k: ℕ) (n: ℕ) : ℕ :=
  -- Implementation of Fk function calculating the smallest positive integer bigger than kn such that Fk(n) * n is a perfect square
  sorry

theorem Fk_same_implies_eq (k: ℕ) (n m: ℕ) (hk: 0 < k) : Fk k n = Fk k m → n = m :=
  sorry

end Fk_same_implies_eq_l146_146030


namespace natural_number_property_l146_146635

theorem natural_number_property (N k : ℕ) (hk : k > 0)
    (h1 : 10^(k-1) ≤ N) (h2 : N < 10^k) (h3 : N * 10^(k-1) ≤ N^2) (h4 : N^2 ≤ N * 10^k) :
    N = 10^(k-1) := 
sorry

end natural_number_property_l146_146635


namespace domain_of_f_symmetry_of_f_l146_146065

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 * x^2 - x^4)) / (abs (x - 2) - 2)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

theorem symmetry_of_f :
  ∀ x : ℝ, f (x + 1) + 1 = f (-(x + 1)) + 1 :=
by
  sorry

end domain_of_f_symmetry_of_f_l146_146065


namespace negation_exists_implies_forall_l146_146768

theorem negation_exists_implies_forall : 
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by
  sorry

end negation_exists_implies_forall_l146_146768


namespace greatest_possible_sum_l146_146211

theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 100) : x + y ≤ 14 :=
sorry

end greatest_possible_sum_l146_146211


namespace find_unknown_rate_l146_146729

theorem find_unknown_rate :
  ∃ x : ℝ, (300 + 750 + 2 * x) / 10 = 170 ↔ x = 325 :=
by
    sorry

end find_unknown_rate_l146_146729


namespace inverse_prop_function_through_point_l146_146889

theorem inverse_prop_function_through_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = k / x) → (f 1 = 2) → (f (-1) = -2) :=
by
  intros f h_inv_prop h_f1
  sorry

end inverse_prop_function_through_point_l146_146889


namespace solve_inequality_l146_146051

theorem solve_inequality : 
  {x : ℝ | (x^3 - x^2 - 6 * x) / (x^2 - 3 * x + 2) > 0} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x)} :=
sorry

end solve_inequality_l146_146051


namespace problem_r_of_3_eq_88_l146_146843

def q (x : ℤ) : ℤ := 2 * x - 5
def r (x : ℤ) : ℤ := x^3 + 2 * x^2 - x - 4

theorem problem_r_of_3_eq_88 : r 3 = 88 :=
by
  sorry

end problem_r_of_3_eq_88_l146_146843


namespace least_positive_integer_division_conditions_l146_146467

theorem least_positive_integer_division_conditions :
  ∃ M : ℤ, M > 0 ∧
  M % 11 = 10 ∧
  M % 12 = 11 ∧
  M % 13 = 12 ∧
  M % 14 = 13 ∧
  M = 30029 := 
by
  sorry

end least_positive_integer_division_conditions_l146_146467


namespace prime_between_30_and_40_has_remainder_7_l146_146714

theorem prime_between_30_and_40_has_remainder_7 (p : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_interval : 30 < p ∧ p < 40) 
  (h_mod : p % 9 = 7) : 
  p = 34 := 
sorry

end prime_between_30_and_40_has_remainder_7_l146_146714


namespace simplify_fraction_expression_l146_146661

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a^3 - b^3 = a - b)

theorem simplify_fraction_expression : (a / b) + (b / a) + (1 / (a * b)) = 2 := by
  sorry

end simplify_fraction_expression_l146_146661


namespace solution_for_a_l146_146510

theorem solution_for_a (x : ℝ) (a : ℝ) (h : 2 * x - a = 0) (hx : x = 1) : a = 2 := by
  rw [hx] at h
  linarith


end solution_for_a_l146_146510


namespace problem_l146_146108

theorem problem (a : ℝ) (h : a^2 - 2 * a - 2 = 0) :
  (1 - 1 / (a + 1)) / (a^3 / (a^2 + 2 * a + 1)) = 1 / 2 :=
by
  sorry

end problem_l146_146108


namespace ratio_problem_l146_146420

theorem ratio_problem (m n p q : ℚ) 
  (h1 : m / n = 12) 
  (h2 : p / n = 4) 
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 :=
by
  sorry

end ratio_problem_l146_146420


namespace harkamal_total_payment_l146_146022

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_of_grapes : ℕ := grapes_quantity * grapes_rate
def cost_of_mangoes : ℕ := mangoes_quantity * mangoes_rate

def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

theorem harkamal_total_payment : total_amount_paid = 1195 := by
  sorry

end harkamal_total_payment_l146_146022


namespace proportion_sets_l146_146300

-- Define unit lengths for clarity
def length (n : ℕ) := n 

-- Define the sets of line segments
def setA := (length 4, length 5, length 6, length 7)
def setB := (length 3, length 4, length 5, length 8)
def setC := (length 5, length 15, length 3, length 9)
def setD := (length 8, length 4, length 1, length 3)

-- Define a condition for a set to form a proportion
def is_proportional (a b c d : ℕ) : Prop :=
  a * d = b * c

-- Main theorem: setC forms a proportion while others don't
theorem proportion_sets : is_proportional 5 15 3 9 ∧ 
                         ¬ is_proportional 4 5 6 7 ∧ 
                         ¬ is_proportional 3 4 5 8 ∧ 
                         ¬ is_proportional 8 4 1 3 := by
  sorry

end proportion_sets_l146_146300


namespace question1_is_random_event_question2_probability_xiuShui_l146_146529

-- Definitions for projects
inductive Project
| A | B | C | D

-- Definition for the problem context and probability computation
def xiuShuiProjects : List Project := [Project.A, Project.B]
def allProjects : List Project := [Project.A, Project.B, Project.C, Project.D]

-- Question 1
def isRandomEvent (event : Project) : Prop :=
  event = Project.C ∧ event ∈ allProjects

theorem question1_is_random_event : isRandomEvent Project.C := by
sorry

-- Question 2: Probability both visit Xiu Shui projects is 1/4
def favorable_outcomes : List (Project × Project) :=
  [(Project.A, Project.A), (Project.A, Project.B), (Project.B, Project.A), (Project.B, Project.B)]

def total_outcomes : List (Project × Project) :=
  List.product allProjects allProjects

def probability (fav : ℕ) (total : ℕ) : ℚ := fav / total

theorem question2_probability_xiuShui : probability favorable_outcomes.length total_outcomes.length = 1 / 4 := by
sorry

end question1_is_random_event_question2_probability_xiuShui_l146_146529


namespace mosquito_drops_per_feed_l146_146039

-- Defining the constants and conditions.
def drops_per_liter : ℕ := 5000
def liters_to_die : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

-- The assertion we want to prove.
theorem mosquito_drops_per_feed :
  (drops_per_liter * liters_to_die) / mosquitoes_to_kill = 20 :=
by
  sorry

end mosquito_drops_per_feed_l146_146039


namespace ellipse_equation_correct_l146_146705

noncomputable def ellipse_equation_proof : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), (x - 2 * y + 4 = 0) ∧ (∃ (f : ℝ × ℝ), f = (-4, 0)) ∧ (∃ (v : ℝ × ℝ), v = (0, 2)) → 
    (x^2 / (a^2) + y^2 / (b^2) = 1 → x^2 / 20 + y^2 / 4 = 1))

theorem ellipse_equation_correct : ellipse_equation_proof :=
  sorry

end ellipse_equation_correct_l146_146705


namespace power_sum_l146_146923

theorem power_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := 
by
  sorry

end power_sum_l146_146923


namespace quadratic_one_solution_l146_146043

theorem quadratic_one_solution (b d : ℝ) (h1 : b + d = 35) (h2 : b < d) (h3 : (24 : ℝ)^2 - 4 * b * d = 0) :
  (b, d) = (35 - Real.sqrt 649 / 2, 35 + Real.sqrt 649 / 2) := 
sorry

end quadratic_one_solution_l146_146043


namespace ashley_cocktail_calories_l146_146325

theorem ashley_cocktail_calories:
  let mango_grams := 150
  let honey_grams := 200
  let water_grams := 300
  let vodka_grams := 100

  let mango_cal_per_100g := 60
  let honey_cal_per_100g := 640
  let vodka_cal_per_100g := 70
  let water_cal_per_100g := 0

  let total_cocktail_grams := mango_grams + honey_grams + water_grams + vodka_grams
  let total_cocktail_calories := (mango_grams * mango_cal_per_100g / 100) +
                                 (honey_grams * honey_cal_per_100g / 100) +
                                 (vodka_grams * vodka_cal_per_100g / 100) +
                                 (water_grams * water_cal_per_100g / 100)
  let caloric_density := total_cocktail_calories / total_cocktail_grams
  let result := 300 * caloric_density
  result = 576 := by
  sorry

end ashley_cocktail_calories_l146_146325


namespace seq_nth_term_2009_l146_146910

theorem seq_nth_term_2009 (n x : ℤ) (h : 2 * x - 3 = 5 ∧ 5 * x - 11 = 9 ∧ 3 * x + 1 = 13) :
  n = 502 ↔ 2009 = (2 * x - 3) + (n - 1) * ((5 * x - 11) - (2 * x - 3)) :=
sorry

end seq_nth_term_2009_l146_146910


namespace probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l146_146255

noncomputable def P_n (n : ℕ) : ℚ :=
  if n = 3 then 1 / 4
  else if n = 4 then 3 / 4
  else 0

theorem probability_center_in_convex_hull_3_points :
  P_n 3 = 1 / 4 :=
by
  sorry

theorem probability_center_in_convex_hull_4_points :
  P_n 4 = 3 / 4 :=
by
  sorry

end probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l146_146255


namespace geom_progression_contra_l146_146125

theorem geom_progression_contra (q : ℝ) (p n : ℕ) (hp : p > 0) (hn : n > 0) :
  (11 = 10 * q^p) → (12 = 10 * q^n) → False :=
by
  -- proof steps should follow here
  sorry

end geom_progression_contra_l146_146125


namespace z_is_200_percent_of_x_l146_146884

theorem z_is_200_percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : y = 0.75 * x) :
  z = 2 * x :=
sorry

end z_is_200_percent_of_x_l146_146884


namespace exists_ten_positive_integers_l146_146674

theorem exists_ten_positive_integers :
  ∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → ¬ (a i ∣ a j))
  ∧ (∀ i j, (a i)^2 ∣ a j) :=
sorry

end exists_ten_positive_integers_l146_146674


namespace tin_to_copper_ratio_l146_146169

theorem tin_to_copper_ratio (L_A T_A T_B C_B : ℝ) 
  (h_total_mass_A : L_A + T_A = 90)
  (h_ratio_A : L_A / T_A = 3 / 4)
  (h_total_mass_B : T_B + C_B = 140)
  (h_total_tin : T_A + T_B = 91.42857142857143) :
  T_B / C_B = 2 / 5 :=
sorry

end tin_to_copper_ratio_l146_146169


namespace soaking_time_l146_146655

theorem soaking_time (time_per_grass_stain : ℕ) (time_per_marinara_stain : ℕ) 
    (number_of_grass_stains : ℕ) (number_of_marinara_stains : ℕ) : 
    time_per_grass_stain = 4 ∧ time_per_marinara_stain = 7 ∧ 
    number_of_grass_stains = 3 ∧ number_of_marinara_stains = 1 →
    (time_per_grass_stain * number_of_grass_stains + time_per_marinara_stain * number_of_marinara_stains) = 19 :=
by
  sorry

end soaking_time_l146_146655


namespace total_students_l146_146066

theorem total_students (S F G B N : ℕ) 
  (hF : F = 41) 
  (hG : G = 22) 
  (hB : B = 9) 
  (hN : N = 24) 
  (h_total : S = (F + G - B) + N) : 
  S = 78 :=
by
  sorry

end total_students_l146_146066


namespace tennis_balls_per_can_is_three_l146_146346

-- Definition of the number of games in each round
def games_in_round (round: Nat) : Nat :=
  match round with
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => 0

-- Definition of the average number of cans used per game
def cans_per_game : Nat := 5

-- Total number of games in the tournament
def total_games : Nat :=
  games_in_round 1 + games_in_round 2 + games_in_round 3 + games_in_round 4

-- Total number of cans used
def total_cans : Nat :=
  total_games * cans_per_game

-- Total number of tennis balls used
def total_tennis_balls : Nat := 225

-- Number of tennis balls per can
def tennis_balls_per_can : Nat :=
  total_tennis_balls / total_cans

-- Theorem to prove
theorem tennis_balls_per_can_is_three :
  tennis_balls_per_can = 3 :=
by
  -- No proof required, using sorry to skip the proof
  sorry

end tennis_balls_per_can_is_three_l146_146346


namespace faye_candies_final_count_l146_146869

def initialCandies : ℕ := 47
def candiesEaten : ℕ := 25
def candiesReceived : ℕ := 40

theorem faye_candies_final_count : (initialCandies - candiesEaten + candiesReceived) = 62 :=
by
  sorry

end faye_candies_final_count_l146_146869


namespace equation_has_solution_iff_l146_146132

open Real

theorem equation_has_solution_iff (a : ℝ) : 
  (∃ x : ℝ, (1/3)^|x| + a - 1 = 0) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end equation_has_solution_iff_l146_146132


namespace range_of_f_l146_146068

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x ^ 2 else Real.cos x

theorem range_of_f : Set.range f = Set.Ici (-1) := 
by
  sorry

end range_of_f_l146_146068


namespace fraction_division_l146_146223

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l146_146223


namespace cyclist_speed_l146_146760

variable (circumference : ℝ) (v₂ : ℝ) (t : ℝ)

theorem cyclist_speed (h₀ : circumference = 180) (h₁ : v₂ = 8) (h₂ : t = 12)
  (h₃ : (7 * t + v₂ * t) = circumference) : 7 = 7 :=
by
  -- From given conditions, we derived that v₁ should be 7
  sorry

end cyclist_speed_l146_146760


namespace quadratic_solution_l146_146395

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end quadratic_solution_l146_146395


namespace train_speed_l146_146102

def distance := 11.67 -- distance in km
def time := 10.0 / 60.0 -- time in hours (10 minutes is 10/60 hours)

theorem train_speed : (distance / time) = 70.02 := by
  sorry

end train_speed_l146_146102


namespace reduced_price_per_kg_of_oil_l146_146878

theorem reduced_price_per_kg_of_oil
  (P : ℝ)
  (h : (1000 / (0.75 * P) - 1000 / P = 5)) :
  0.75 * (1000 / 15) = 50 := 
sorry

end reduced_price_per_kg_of_oil_l146_146878


namespace sin_B_value_cos_A_minus_cos_C_value_l146_146596

variables {A B C : ℝ} {a b c : ℝ}

theorem sin_B_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) : Real.sin B = Real.sqrt 7 / 4 := 
sorry

theorem cos_A_minus_cos_C_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) (h₂ : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := 
sorry

end sin_B_value_cos_A_minus_cos_C_value_l146_146596


namespace substitution_modulo_l146_146854

-- Definitions based on conditions
def total_players := 15
def starting_lineup := 10
def substitutes := 5
def max_substitutions := 2

-- Define the number of substitutions ways for the cases 0, 1, and 2 substitutions
def a_0 := 1
def a_1 := starting_lineup * substitutes
def a_2 := starting_lineup * substitutes * (starting_lineup - 1) * (substitutes - 1)

-- Summing the total number of substitution scenarios
def total_substitution_scenarios := a_0 + a_1 + a_2

-- Theorem statement to verify the result modulo 500
theorem substitution_modulo : total_substitution_scenarios % 500 = 351 := by
  sorry

end substitution_modulo_l146_146854


namespace sin_three_pi_div_two_l146_146078

theorem sin_three_pi_div_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_div_two_l146_146078


namespace intersection_A_B_at_3_range_of_a_l146_146832

open Set

-- Definitions from the condition
def A (x : ℝ) : Prop := abs x ≥ 2
def B (x a : ℝ) : Prop := (x - 2 * a) * (x + 3) < 0

-- Part (Ⅰ)
theorem intersection_A_B_at_3 :
  let a := 3
  let A := {x : ℝ | abs x ≥ 2}
  let B := {x : ℝ | (x - 6) * (x + 3) < 0}
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (2 ≤ x ∧ x < 6)} :=
by
  sorry

-- Part (Ⅱ)
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, A x ∨ B x a) → a ≥ 1 :=
by
  sorry

end intersection_A_B_at_3_range_of_a_l146_146832


namespace probability_of_MATHEMATICS_letter_l146_146494

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end probability_of_MATHEMATICS_letter_l146_146494


namespace minimizes_G_at_7_over_12_l146_146512

def F (p q : ℝ) : ℝ :=
  -2 * p * q + 3 * p * (1 - q) + 3 * (1 - p) * q - 4 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (3 * p - 4) (3 - 5 * p)

theorem minimizes_G_at_7_over_12 :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → (∀ p, G p ≥ G (7 / 12)) ↔ p = 7 / 12 :=
by
  sorry

end minimizes_G_at_7_over_12_l146_146512


namespace hexagon_diagonals_l146_146136

-- Define a hexagon as having 6 vertices
def hexagon_vertices : ℕ := 6

-- From one vertex of a hexagon, there are (6 - 1) vertices it can potentially connect to
def potential_connections (vertices : ℕ) : ℕ := vertices - 1

-- Remove the two adjacent vertices to count diagonals
def diagonals_from_vertex (connections : ℕ) : ℕ := connections - 2

theorem hexagon_diagonals : diagonals_from_vertex (potential_connections hexagon_vertices) = 3 := by
  -- The proof is intentionally left as a sorry placeholder.
  sorry

end hexagon_diagonals_l146_146136


namespace jenny_boxes_sold_l146_146503

/--
Jenny sold some boxes of Trefoils. Each box has 8.0 packs. She sold 192 packs in total.
Prove that Jenny sold 24 boxes.
-/
theorem jenny_boxes_sold (packs_per_box : Real) (total_packs_sold : Real) (num_boxes_sold : Real) 
  (h1 : packs_per_box = 8.0) (h2 : total_packs_sold = 192) : num_boxes_sold = 24 :=
by
  have h3 : num_boxes_sold = total_packs_sold / packs_per_box :=
    by sorry
  sorry

end jenny_boxes_sold_l146_146503


namespace trigonometric_identity_l146_146723

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = -2 :=
by 
  sorry

end trigonometric_identity_l146_146723


namespace sum_of_ages_l146_146338

theorem sum_of_ages (M S G : ℕ)
  (h1 : M = 2 * S)
  (h2 : S = 2 * G)
  (h3 : G = 20) :
  M + S + G = 140 :=
sorry

end sum_of_ages_l146_146338


namespace all_statements_true_l146_146074

theorem all_statements_true (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2 < (a + b)^2) ∧ 
  (ab > 0) ∧ 
  (a > b) ∧ 
  (a > 0) ∧
  (b > 0) :=
by
  sorry

end all_statements_true_l146_146074


namespace smaller_number_l146_146773

theorem smaller_number {a b : ℕ} (h_ratio : b = 5 * a / 2) (h_lcm : Nat.lcm a b = 160) : a = 64 := 
by
  sorry

end smaller_number_l146_146773


namespace band_weight_correct_l146_146698

universe u

structure InstrumentGroup where
  count : ℕ
  weight_per_instrument : ℕ

def total_weight (ig : InstrumentGroup) : ℕ :=
  ig.count * ig.weight_per_instrument

def total_band_weight : ℕ :=
  (total_weight ⟨6, 5⟩) + (total_weight ⟨9, 5⟩) +
  (total_weight ⟨8, 10⟩) + (total_weight ⟨3, 20⟩) + (total_weight ⟨2, 15⟩)

theorem band_weight_correct : total_band_weight = 245 := by
  rfl

end band_weight_correct_l146_146698


namespace share_expenses_l146_146257

theorem share_expenses (h l : ℕ) : 
  let henry_paid := 120
  let linda_paid := 150
  let jack_paid := 210
  let total_paid := henry_paid + linda_paid + jack_paid
  let each_should_pay := total_paid / 3
  let henry_owes := each_should_pay - henry_paid
  let linda_owes := each_should_pay - linda_paid
  (h = henry_owes) → 
  (l = linda_owes) → 
  h - l = 30 := by
  sorry

end share_expenses_l146_146257


namespace nathan_banana_payment_l146_146672

theorem nathan_banana_payment
  (bunches_8 : ℕ)
  (cost_per_bunch_8 : ℝ)
  (bunches_7 : ℕ)
  (cost_per_bunch_7 : ℝ)
  (discount : ℝ)
  (total_payment : ℝ) :
  bunches_8 = 6 →
  cost_per_bunch_8 = 2.5 →
  bunches_7 = 5 →
  cost_per_bunch_7 = 2.2 →
  discount = 0.10 →
  total_payment = 6 * 2.5 + 5 * 2.2 - 0.10 * (6 * 2.5 + 5 * 2.2) →
  total_payment = 23.40 :=
by
  intros
  sorry

end nathan_banana_payment_l146_146672


namespace wheel_speed_l146_146875

def original_circumference_in_miles := 10 / 5280
def time_factor := 3600
def new_time_factor := 3600 - (1/3)

theorem wheel_speed
  (r : ℝ) 
  (original_speed : r * time_factor = original_circumference_in_miles * 3600)
  (new_speed : (r + 5) * (time_factor - 1/10800) = original_circumference_in_miles * 3600) :
  r = 10 :=
sorry

end wheel_speed_l146_146875


namespace conditionD_necessary_not_sufficient_l146_146017

variable (a b : ℝ)

-- Define each of the conditions as separate variables
def conditionA : Prop := |a| < |b|
def conditionB : Prop := 2 * a < 2 * b
def conditionC : Prop := a < b - 1
def conditionD : Prop := a < b + 1

-- Prove that condition D is necessary but not sufficient for a < b
theorem conditionD_necessary_not_sufficient : conditionD a b → (¬ conditionA a b ∨ ¬ conditionB a b ∨ ¬ conditionC a b) ∧ ¬(conditionD a b ↔ a < b) :=
by sorry

end conditionD_necessary_not_sufficient_l146_146017


namespace a_value_l146_146096

-- Definition of the operation
def star (x y : ℝ) : ℝ := x + y - x * y

-- Main theorem to prove
theorem a_value :
  let a := star 1 (star 0 1)
  a = 1 :=
by
  sorry

end a_value_l146_146096


namespace total_payment_is_correct_l146_146731

def daily_rental_cost : ℝ := 30
def per_mile_cost : ℝ := 0.25
def one_time_service_charge : ℝ := 15
def rent_duration : ℝ := 4
def distance_driven : ℝ := 500

theorem total_payment_is_correct :
  (daily_rental_cost * rent_duration + per_mile_cost * distance_driven + one_time_service_charge) = 260 := 
by
  sorry

end total_payment_is_correct_l146_146731


namespace equation_of_line_l146_146427

theorem equation_of_line :
  ∃ m : ℝ, ∀ x y : ℝ, (y = m * x - m ∧ (m = 2 ∧ x = 1 ∧ y = 0)) ∧ 
  ∀ x : ℝ, ¬(4 * x^2 - (m * x - m)^2 - 8 * x = 12) → m = 2 → y = 2 * x - 2 :=
by sorry

end equation_of_line_l146_146427


namespace find_fraction_l146_146659

variable (F N : ℚ)

-- Defining the conditions
def condition1 : Prop := (1 / 3) * F * N = 18
def condition2 : Prop := (3 / 10) * N = 64.8

-- Proof statement
theorem find_fraction (h1 : condition1 F N) (h2 : condition2 N) : F = 1 / 4 := by 
  sorry

end find_fraction_l146_146659


namespace solve_purchase_price_problem_l146_146925

def purchase_price_problem : Prop :=
  ∃ P : ℝ, (0.10 * P + 12 = 35) ∧ (P = 230)

theorem solve_purchase_price_problem : purchase_price_problem :=
  by
    sorry

end solve_purchase_price_problem_l146_146925


namespace sums_of_squares_divisibility_l146_146167

theorem sums_of_squares_divisibility :
  (∀ n : ℤ, (3 * n^2 + 2) % 3 ≠ 0) ∧ (∃ n : ℤ, (3 * n^2 + 2) % 11 = 0) := 
by
  sorry

end sums_of_squares_divisibility_l146_146167


namespace difference_of_integers_l146_146964

theorem difference_of_integers :
  ∀ (x y : ℤ), (x = 32) → (y = 5*x + 2) → (y - x = 130) :=
by
  intros x y hx hy
  sorry

end difference_of_integers_l146_146964


namespace probability_of_neither_event_l146_146781

theorem probability_of_neither_event (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.25) (h2 : P_B = 0.40) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.50 :=
by
  rw [h1, h2, h3]
  sorry

end probability_of_neither_event_l146_146781


namespace john_tax_rate_l146_146082

theorem john_tax_rate { P: Real → Real → Real → Real → Prop }:
  ∀ (cNikes cBoots totalPaid taxRate: ℝ), 
  cNikes = 150 →
  cBoots = 120 →
  totalPaid = 297 →
  taxRate = ((totalPaid - (cNikes + cBoots)) / (cNikes + cBoots)) * 100 →
  taxRate = 10 :=
by
  intros cNikes cBoots totalPaid taxRate HcNikes HcBoots HtotalPaid HtaxRate
  sorry

end john_tax_rate_l146_146082


namespace C_eq_D_iff_n_eq_3_l146_146535

noncomputable def C (n : ℕ) : ℝ :=
  1000 * (1 - (1 / 3^n)) / (1 - 1 / 3)

noncomputable def D (n : ℕ) : ℝ :=
  2700 * (1 - (1 / (-3)^n)) / (1 + 1 / 3)

theorem C_eq_D_iff_n_eq_3 (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 3 :=
by
  unfold C D
  sorry

end C_eq_D_iff_n_eq_3_l146_146535


namespace minimum_order_amount_to_get_discount_l146_146548

theorem minimum_order_amount_to_get_discount 
  (cost_quiche : ℝ) (cost_croissant : ℝ) (cost_biscuit : ℝ) (n_quiches : ℝ) (n_croissants : ℝ) (n_biscuits : ℝ)
  (discount_percent : ℝ) (total_with_discount : ℝ) (min_order_amount : ℝ) :
  cost_quiche = 15.0 → cost_croissant = 3.0 → cost_biscuit = 2.0 →
  n_quiches = 2 → n_croissants = 6 → n_biscuits = 6 →
  discount_percent = 0.10 → total_with_discount = 54.0 →
  (n_quiches * cost_quiche + n_croissants * cost_croissant + n_biscuits * cost_biscuit) * (1 - discount_percent) = total_with_discount →
  min_order_amount = 60.0 :=
by
  sorry

end minimum_order_amount_to_get_discount_l146_146548


namespace correct_articles_l146_146900

-- Define the given conditions
def specific_experience : Prop := true
def countable_noun : Prop := true

-- Problem statement: given the conditions, choose the correct articles to fill in the blanks
theorem correct_articles (h1 : specific_experience) (h2 : countable_noun) : 
  "the; a" = "the; a" :=
by
  sorry

end correct_articles_l146_146900


namespace no_infinite_sequence_of_positive_integers_l146_146105

theorem no_infinite_sequence_of_positive_integers (a : ℕ → ℕ) (H : ∀ n, a n > 0) :
  ¬(∀ n, (a (n+1))^2 ≥ 2 * (a n) * (a (n+2))) :=
sorry

end no_infinite_sequence_of_positive_integers_l146_146105


namespace least_number_to_add_l146_146128

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) (k : ℕ) (l : ℕ) (h₁ : n = 1077) (h₂ : d = 23) (h₃ : n % d = r) (h₄ : d - r = k) (h₅ : r = 19) (h₆ : k = l) : l = 4 :=
by
  sorry

end least_number_to_add_l146_146128


namespace speed_in_still_water_l146_146252

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 25) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 45 :=
by
  sorry

end speed_in_still_water_l146_146252


namespace john_works_30_hours_per_week_l146_146765

/-- Conditions --/
def hours_per_week_fiona : ℕ := 40
def hours_per_week_jeremy : ℕ := 25
def hourly_wage : ℕ := 20
def monthly_total_payment : ℕ := 7600
def weeks_in_month : ℕ := 4

/-- Derived Definitions --/
def monthly_hours_fiona_jeremy : ℕ :=
  (hours_per_week_fiona + hours_per_week_jeremy) * weeks_in_month

def monthly_payment_fiona_jeremy : ℕ :=
  hourly_wage * monthly_hours_fiona_jeremy

def monthly_payment_john : ℕ :=
  monthly_total_payment - monthly_payment_fiona_jeremy

def hours_per_month_john : ℕ :=
  monthly_payment_john / hourly_wage

def hours_per_week_john : ℕ :=
  hours_per_month_john / weeks_in_month

/-- Theorem stating that John works 30 hours per week --/
theorem john_works_30_hours_per_week :
  hours_per_week_john = 30 := by
  sorry

end john_works_30_hours_per_week_l146_146765


namespace min_y_squared_isosceles_trapezoid_l146_146350

theorem min_y_squared_isosceles_trapezoid:
  ∀ (EF GH y : ℝ) (circle_center : ℝ)
    (isosceles_trapezoid : Prop)
    (tangent_EH : Prop)
    (tangent_FG : Prop),
  isosceles_trapezoid ∧ EF = 72 ∧ GH = 45 ∧ EH = y ∧ FG = y ∧
  (∃ (circle : ℝ), circle_center = (EF / 2) ∧ tangent_EH ∧ tangent_FG)
  → y^2 = 486 :=
by sorry

end min_y_squared_isosceles_trapezoid_l146_146350


namespace min_value_a2b3c_l146_146710

theorem min_value_a2b3c {m : ℝ} (hm : m > 0)
  (hineq : ∀ x : ℝ, |x + 1| + |2 * x - 1| ≥ m)
  {a b c : ℝ} (habc : a^2 + 2 * b^2 + 3 * c^2 = m) :
  a + 2 * b + 3 * c ≥ -3 :=
sorry

end min_value_a2b3c_l146_146710


namespace correct_operation_l146_146861

theorem correct_operation : -5 * 3 = -15 :=
by sorry

end correct_operation_l146_146861


namespace rainy_days_l146_146366

theorem rainy_days (n R NR : ℕ): (n * R + 3 * NR = 20) ∧ (3 * NR = n * R + 10) ∧ (R + NR = 7) → R = 2 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end rainy_days_l146_146366


namespace calculate_difference_l146_146576

def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

theorem calculate_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) :=
by
  sorry

end calculate_difference_l146_146576


namespace expected_points_A_correct_prob_A_B_same_points_correct_l146_146281

-- Conditions
def game_is_independent := true

def prob_A_B_win := 2/5
def prob_A_B_draw := 1/5

def prob_A_C_win := 1/3
def prob_A_C_draw := 1/3

def prob_B_C_win := 1/2
def prob_B_C_draw := 1/6

noncomputable def prob_A_B_lose := 1 - prob_A_B_win - prob_A_B_draw
noncomputable def prob_A_C_lose := 1 - prob_A_C_win - prob_A_C_draw
noncomputable def prob_B_C_lose := 1 - prob_B_C_win - prob_B_C_draw

noncomputable def expected_points_A : ℚ := 0 * (prob_A_B_lose * prob_A_C_lose)        /- P(ξ=0) = 2/15 -/
                                       + 1 * ((prob_A_B_draw * prob_A_C_lose) +
                                              (prob_A_B_lose * prob_A_C_draw))        /- P(ξ=1) = 1/5 -/
                                       + 2 * (prob_A_B_draw * prob_A_C_draw)         /- P(ξ=2) = 1/15 -/
                                       + 3 * ((prob_A_B_win * prob_A_C_lose) + 
                                              (prob_A_B_win * prob_A_C_draw) + 
                                              (prob_A_C_win * prob_A_B_lose))        /- P(ξ=3) = 4/15 -/
                                       + 4 * ((prob_A_B_draw * prob_A_C_win) +
                                              (prob_A_B_win * prob_A_C_win))         /- P(ξ=4) = 1/5 -/
                                       + 6 * (prob_A_B_win * prob_A_C_win)           /- P(ξ=6) = 2/15 -/

theorem expected_points_A_correct : expected_points_A = 41 / 15 :=
by
  sorry

noncomputable def prob_A_B_same_points: ℚ := ((prob_A_B_draw * prob_A_C_lose) * prob_B_C_lose)  /- both 1 point -/
                                            + ((prob_A_B_draw * prob_A_C_draw) * prob_B_C_draw)/- both 2 points -/
                                            + ((prob_A_B_win * prob_B_C_win) * prob_A_C_lose)  /- both 3 points -/
                                            + ((prob_A_B_win * prob_A_C_lose) * prob_B_C_win)  /- both 3 points -/
                                            + ((prob_A_B_draw * prob_A_C_win) * prob_B_C_win)  /- both 4 points -/

theorem prob_A_B_same_points_correct : prob_A_B_same_points = 8 / 45 :=
by
  sorry

end expected_points_A_correct_prob_A_B_same_points_correct_l146_146281


namespace men_wages_l146_146949

def men := 5
def women := 5
def boys := 7
def total_wages := 90
def wage_man := 7.5

theorem men_wages (men women boys : ℕ) (total_wages wage_man : ℝ)
  (h1 : 5 = women) (h2 : women = boys) (h3 : 5 * wage_man + 1 * wage_man + 7 * wage_man = total_wages) :
  5 * wage_man = 37.5 :=
  sorry

end men_wages_l146_146949


namespace area_under_abs_sin_l146_146313

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem area_under_abs_sin : 
  ∫ x in -Real.pi..Real.pi, f x = 4 :=
by
  sorry

end area_under_abs_sin_l146_146313


namespace problem_l146_146966

def g (x : ℝ) (d e f : ℝ) := d * x^2 + e * x + f

theorem problem (d e f : ℝ) (h_vertex : ∀ x : ℝ, g d e f (x + 2) = -1 * (x + 2)^2 + 5) :
  d + e + 3 * f = 14 := 
sorry

end problem_l146_146966


namespace paint_cost_per_liter_l146_146603

def cost_brush : ℕ := 20
def cost_canvas : ℕ := 3 * cost_brush
def min_liters : ℕ := 5
def total_earning : ℕ := 200
def total_profit : ℕ := 80
def total_cost : ℕ := total_earning - total_profit

theorem paint_cost_per_liter :
  (total_cost = cost_brush + cost_canvas + (5 * 8)) :=
by
  sorry

end paint_cost_per_liter_l146_146603


namespace length_of_train_l146_146308

theorem length_of_train 
  (L V : ℝ) 
  (h1 : L = V * 8) 
  (h2 : L + 279 = V * 20) : 
  L = 186 :=
by
  -- solve using the given conditions
  sorry

end length_of_train_l146_146308


namespace mike_scored_212_l146_146721

variable {M : ℕ}

def passing_marks (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

def mike_marks (passing_marks shortfall : ℕ) : ℕ := passing_marks - shortfall

theorem mike_scored_212 (max_marks : ℕ) (shortfall : ℕ)
  (h1 : max_marks = 790)
  (h2 : shortfall = 25)
  (h3 : M = mike_marks (passing_marks max_marks) shortfall) : 
  M = 212 := 
by 
  sorry

end mike_scored_212_l146_146721


namespace list_price_of_article_l146_146903

theorem list_price_of_article
  (P : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (final_price : ℝ)
  (h1 : discount1 = 0.10)
  (h2 : discount2 = 0.01999999999999997)
  (h3 : final_price = 61.74) :
  P = 70 :=
by
  sorry

end list_price_of_article_l146_146903


namespace mohit_discount_l146_146146

variable (SP : ℝ) -- Selling price
variable (CP : ℝ) -- Cost price
variable (discount_percentage : ℝ) -- Discount percentage

-- Conditions
axiom h1 : SP = 21000
axiom h2 : CP = 17500
axiom h3 : discount_percentage = ( (SP - (CP + 0.08 * CP)) / SP) * 100

-- Theorem to prove
theorem mohit_discount : discount_percentage = 10 :=
  sorry

end mohit_discount_l146_146146


namespace base8_to_base10_l146_146454

theorem base8_to_base10 (n : ℕ) : n = 4 * 8^3 + 3 * 8^2 + 7 * 8^1 + 2 * 8^0 → n = 2298 :=
by 
  sorry

end base8_to_base10_l146_146454


namespace largest_two_numbers_l146_146582

def a : Real := 2^(1/2)
def b : Real := 3^(1/3)
def c : Real := 8^(1/8)
def d : Real := 9^(1/9)

theorem largest_two_numbers : 
  (max (max (max a b) c) d = b) ∧ 
  (max (max a c) d = a) := 
sorry

end largest_two_numbers_l146_146582


namespace remainder_of_122_div_20_l146_146627

theorem remainder_of_122_div_20 :
  (∃ (q r : ℕ), 122 = 20 * q + r ∧ r < 20 ∧ q = 6) →
  r = 2 :=
by
  sorry

end remainder_of_122_div_20_l146_146627


namespace chocolates_sold_l146_146500

theorem chocolates_sold (C S : ℝ) (n : ℕ) (h1 : 165 * C = n * S) (h2 : ((S - C) / C) * 100 = 10) : n = 150 :=
by
  sorry

end chocolates_sold_l146_146500


namespace father_l146_146156

theorem father's_age (M F : ℕ) 
  (h1 : M = (2 / 5 : ℝ) * F)
  (h2 : M + 14 = (1 / 2 : ℝ) * (F + 14)) : 
  F = 70 := 
  sorry

end father_l146_146156


namespace total_people_who_eat_vegetarian_l146_146525

def people_who_eat_only_vegetarian := 16
def people_who_eat_both_vegetarian_and_non_vegetarian := 12

-- We want to prove that the total number of people who eat vegetarian is 28
theorem total_people_who_eat_vegetarian : 
  people_who_eat_only_vegetarian + people_who_eat_both_vegetarian_and_non_vegetarian = 28 :=
by 
  sorry

end total_people_who_eat_vegetarian_l146_146525


namespace ann_older_than_susan_l146_146490

variables (A S : ℕ)

theorem ann_older_than_susan (h1 : S = 11) (h2 : A + S = 27) : A - S = 5 := by
  -- Proof is skipped
  sorry

end ann_older_than_susan_l146_146490


namespace numbers_not_perfect_squares_or_cubes_l146_146868

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l146_146868


namespace simplify_evaluate_expression_l146_146791

noncomputable def a : ℝ := 2 * Real.cos (60 * Real.pi / 180) + 1

theorem simplify_evaluate_expression : (a - (a^2) / (a + 1)) / ((a^2) / ((a^2) - 1)) = 1 / 2 :=
by sorry

end simplify_evaluate_expression_l146_146791


namespace Lizette_average_above_94_l146_146954

noncomputable def Lizette_new_weighted_average
  (score3: ℝ) (avg3: ℝ) (weight3: ℝ) (score_new1 score_new2: ℝ) (weight_new: ℝ) :=
  let total_points3 := avg3 * 3
  let total_weight3 := 3 * weight3
  let total_points := total_points3 + score_new1 + score_new2
  let total_weight := total_weight3 + 2 * weight_new
  total_points / total_weight

theorem Lizette_average_above_94:
  ∀ (score3 avg3 weight3 score_new1 score_new2 weight_new: ℝ),
  score3 = 92 →
  avg3 = 94 →
  weight3 = 0.15 →
  score_new1 > 94 →
  score_new2 > 94 →
  weight_new = 0.20 →
  Lizette_new_weighted_average score3 avg3 weight3 score_new1 score_new2 weight_new > 94 :=
by
  intros score3 avg3 weight3 score_new1 score_new2 weight_new h1 h2 h3 h4 h5 h6
  sorry

end Lizette_average_above_94_l146_146954


namespace area_of_ADC_l146_146364

theorem area_of_ADC
  (BD DC : ℝ)
  (h_ratio : BD / DC = 2 / 3)
  (area_ABD : ℝ)
  (h_area_ABD : area_ABD = 30) :
  ∃ area_ADC, area_ADC = 45 :=
by {
  sorry
}

end area_of_ADC_l146_146364


namespace skew_lines_sufficient_not_necessary_l146_146866

-- Definitions for the conditions
def skew_lines (l1 l2 : Type) : Prop := sorry -- Definition of skew lines
def do_not_intersect (l1 l2 : Type) : Prop := sorry -- Definition of not intersecting

-- The main theorem statement
theorem skew_lines_sufficient_not_necessary (l1 l2 : Type) :
  (skew_lines l1 l2) → (do_not_intersect l1 l2) ∧ ¬ (do_not_intersect l1 l2 → skew_lines l1 l2) :=
by
  sorry

end skew_lines_sufficient_not_necessary_l146_146866


namespace geometric_seq_common_ratio_l146_146549

theorem geometric_seq_common_ratio 
  (a : ℕ → ℝ) -- a_n is the sequence
  (S : ℕ → ℝ) -- S_n is the partial sum of the sequence
  (h1 : a 3 = 2 * S 2 + 1) -- condition a_3 = 2S_2 + 1
  (h2 : a 4 = 2 * S 3 + 1) -- condition a_4 = 2S_3 + 1
  (h3 : S 2 = a 1 / (1 / q) * (1 - q^3) / (1 - q)) -- sum of first 2 terms
  (h4 : S 3 = a 1 / (1 / q) * (1 - q^4) / (1 - q)) -- sum of first 3 terms
  : q = 3 := -- conclusion
by sorry

end geometric_seq_common_ratio_l146_146549


namespace find_n_150_l146_146888

def special_sum (k n : ℕ) : ℕ := (n * (2 * k + n - 1)) / 2

theorem find_n_150 : ∃ n : ℕ, special_sum 3 n = 150 ∧ n = 15 :=
by
  sorry

end find_n_150_l146_146888


namespace pencil_length_total_l146_146047

theorem pencil_length_total :
  (1.5 + 0.5 + 2 + 1.25 + 0.75 + 1.8 + 2.5 = 10.3) :=
by
  sorry

end pencil_length_total_l146_146047


namespace power_of_four_l146_146118

-- Definition of the conditions
def prime_factors (x: ℕ): ℕ := 2 * x + 5 + 2

-- The statement we need to prove given the conditions
theorem power_of_four (x: ℕ) (h: prime_factors x = 33) : x = 13 :=
by
  -- Proof goes here
  sorry

end power_of_four_l146_146118


namespace max_gcd_15n_plus_4_8n_plus_1_l146_146199

theorem max_gcd_15n_plus_4_8n_plus_1 (n : ℕ) (h : n > 0) : 
  ∃ g, g = gcd (15 * n + 4) (8 * n + 1) ∧ g ≤ 17 :=
sorry

end max_gcd_15n_plus_4_8n_plus_1_l146_146199


namespace f_one_zero_x_range_l146_146538

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
-- f is defined for x > 0
variable (f : ℝ → ℝ)
variables (h_domain : ∀ x, x > 0 → ∃ y, f x = y)
variables (h1 : f 2 = 1)
variables (h2 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
variables (h3 : ∀ x y, x > y → f x > f y)

-- Question 1
theorem f_one_zero (hf1 : f 1 = 0) : True := 
  by trivial
  
-- Question 2
theorem x_range (x: ℝ) (hx: f 3 + f (4 - 8 * x) > 2) : x ≤ 1/3 := sorry

end f_one_zero_x_range_l146_146538


namespace last_digits_nn_periodic_l146_146692

theorem last_digits_nn_periodic (n : ℕ) : 
  ∃ p > 0, ∀ k, (n + k * p)^(n + k * p) % 10 = n^n % 10 := 
sorry

end last_digits_nn_periodic_l146_146692


namespace no_solution_if_and_only_if_l146_146021

theorem no_solution_if_and_only_if (n : ℝ) : 
  ¬ ∃ (x y z : ℝ), 
    (n * x + y = 1) ∧ 
    (n * y + z = 1) ∧ 
    (x + n * z = 1) ↔ n = -1 :=
by
  sorry

end no_solution_if_and_only_if_l146_146021


namespace student_in_16th_group_has_number_244_l146_146301

theorem student_in_16th_group_has_number_244 :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 800 ∧ ((k - 36) % 16 = 0) ∧ (n = 3 + (k - 36) / 16)) →
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ 800 ∧ ((m - 244) % 16 = 0) ∧ (16 = 3 + (m - 36) / 16) :=
by
  sorry

end student_in_16th_group_has_number_244_l146_146301


namespace power_of_two_plus_one_is_power_of_integer_l146_146598

theorem power_of_two_plus_one_is_power_of_integer (n : ℕ) (hn : 0 < n) (a k : ℕ) (ha : 2^n + 1 = a^k) (hk : 1 < k) : n = 3 :=
by
  sorry

end power_of_two_plus_one_is_power_of_integer_l146_146598


namespace funnel_paper_area_l146_146432

theorem funnel_paper_area
  (slant_height : ℝ)
  (base_circumference : ℝ)
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi):
  (1 / 2) * base_circumference * slant_height = 18 * Real.pi :=
by
  sorry

end funnel_paper_area_l146_146432


namespace sum_of_squares_l146_146695

def satisfies_conditions (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h : satisfies_conditions x y z) :
  ∀ (x y z : ℕ), x + y + z = 24 ∧ Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  x^2 + y^2 + z^2 = 216 :=
sorry

end sum_of_squares_l146_146695


namespace quotient_equivalence_l146_146478

variable (N H J : ℝ)

theorem quotient_equivalence
  (h1 : N / H = 1.2)
  (h2 : H / J = 5 / 6) :
  N / J = 1 := by
  sorry

end quotient_equivalence_l146_146478


namespace tan_sum_simplification_l146_146502

theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (Real.pi / 4)) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  sorry

end tan_sum_simplification_l146_146502


namespace min_value_x_plus_y_l146_146241

theorem min_value_x_plus_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
sorry

end min_value_x_plus_y_l146_146241


namespace emma_time_l146_146941

theorem emma_time (E : ℝ) (h1 : 2 * E + E = 60) : E = 20 :=
sorry

end emma_time_l146_146941


namespace prove_math_problem_l146_146936

noncomputable def math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : Prop :=
  (x + y = 1) ∧ (x^5 + y^5 = 11)

theorem prove_math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : math_problem x y h1 h2 h3 :=
  sorry

end prove_math_problem_l146_146936


namespace intersection_point_exists_l146_146201

theorem intersection_point_exists :
  ∃ (x y z t : ℝ), (x = 1 - 2 * t) ∧ (y = 2 + t) ∧ (z = -1 - t) ∧
                   (x - 2 * y + 5 * z + 17 = 0) ∧ 
                   (x = -1) ∧ (y = 3) ∧ (z = -2) :=
by
  sorry

end intersection_point_exists_l146_146201


namespace sum_fourth_powers_l146_146896

theorem sum_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 25 / 6 :=
by sorry

end sum_fourth_powers_l146_146896


namespace find_functions_l146_146730

variable (f : ℝ → ℝ)

def isFunctionPositiveReal := ∀ x : ℝ, x > 0 → f x > 0

axiom functional_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) : f (x ^ y) = f x ^ f y

theorem find_functions (hf : isFunctionPositiveReal f) :
  (∀ x : ℝ, x > 0 → f x = 1) ∨ (∀ x : ℝ, x > 0 → f x = x) := sorry

end find_functions_l146_146730


namespace statement_is_true_l146_146826

theorem statement_is_true (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h : ∀ x : ℝ, |x + 2| < b → |(3 * x + 2) + 4| < a) : b ≤ a / 3 :=
by
  sorry

end statement_is_true_l146_146826


namespace skylar_current_age_l146_146273

theorem skylar_current_age (started_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) (h1 : started_age = 17) (h2 : annual_donation = 8000) (h3 : total_donation = 440000) : 
  (started_age + total_donation / annual_donation = 72) :=
by
  sorry

end skylar_current_age_l146_146273


namespace original_profit_percentage_l146_146595

theorem original_profit_percentage (C S : ℝ) 
  (h1 : S - 1.12 * C = 0.5333333333333333 * S) : 
  ((S - C) / C) * 100 = 140 :=
sorry

end original_profit_percentage_l146_146595


namespace track_extension_needed_l146_146192

noncomputable def additional_track_length (r : ℝ) (g1 g2 : ℝ) : ℝ :=
  let l1 := r / g1
  let l2 := r / g2
  l2 - l1

theorem track_extension_needed :
  additional_track_length 800 0.04 0.015 = 33333 :=
by
  sorry

end track_extension_needed_l146_146192


namespace variance_scaled_l146_146536

theorem variance_scaled (s1 : ℝ) (c : ℝ) (h1 : s1 = 3) (h2 : c = 3) :
  s1 * (c^2) = 27 :=
by
  rw [h1, h2]
  norm_num

end variance_scaled_l146_146536


namespace gcd_gx_x_l146_146753

theorem gcd_gx_x (x : ℕ) (h : 2520 ∣ x) : 
  Nat.gcd ((4*x + 5) * (5*x + 2) * (11*x + 8) * (3*x + 7)) x = 280 := 
sorry

end gcd_gx_x_l146_146753


namespace black_region_area_is_correct_l146_146848

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end black_region_area_is_correct_l146_146848


namespace positive_integer_solutions_l146_146121

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end positive_integer_solutions_l146_146121


namespace least_number_of_people_l146_146399

-- Conditions
def first_caterer_cost (x : ℕ) : ℕ := 120 + 18 * x
def second_caterer_cost (x : ℕ) : ℕ := 250 + 15 * x

-- Proof Statement
theorem least_number_of_people (x : ℕ) (h : x ≥ 44) : first_caterer_cost x > second_caterer_cost x :=
by sorry

end least_number_of_people_l146_146399


namespace find_y_l146_146870

theorem find_y (x y : ℕ) (h1 : x > 0 ∧ y > 0) (h2 : x % y = 9) (h3 : (x:ℝ) / (y:ℝ) = 96.45) : y = 20 :=
by
  sorry

end find_y_l146_146870


namespace fixed_monthly_costs_l146_146270

theorem fixed_monthly_costs
  (cost_per_component : ℕ) (shipping_cost : ℕ) 
  (num_components : ℕ) (selling_price : ℚ)
  (F : ℚ) :
  cost_per_component = 80 →
  shipping_cost = 6 →
  num_components = 150 →
  selling_price = 196.67 →
  F = (num_components * selling_price) - (num_components * (cost_per_component + shipping_cost)) →
  F = 16600.5 :=
by
  intros
  sorry

end fixed_monthly_costs_l146_146270


namespace part1_part2_l146_146491

noncomputable def quadratic_eq (m x : ℝ) : Prop := m * x^2 - 2 * x + 1 = 0

theorem part1 (m : ℝ) : 
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 ≠ x2) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

theorem part2 (m : ℝ) (x1 x2 : ℝ) : 
  (quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 * x2 - x1 - x2 = 1/2) ↔ (m = -2) :=
by sorry

end part1_part2_l146_146491


namespace solution_set_of_inequality_l146_146296

theorem solution_set_of_inequality :
  {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {-1 / 3} :=
by {
  sorry -- Proof goes here
}

end solution_set_of_inequality_l146_146296


namespace value_of_a_minus_b_l146_146052

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the invertible function f

theorem value_of_a_minus_b (a b : ℝ) (hf_inv : Function.Injective f)
  (hfa : f a = b) (hfb : f b = 6) (ha1 : f 3 = 1) (hb1 : f 1 = 6) : a - b = 2 :=
sorry

end value_of_a_minus_b_l146_146052


namespace avg_stoppage_time_is_20_minutes_l146_146994

noncomputable def avg_stoppage_time : Real :=
let train1 := (60, 40) -- without stoppages, with stoppages (in kmph)
let train2 := (75, 50) -- without stoppages, with stoppages (in kmph)
let train3 := (90, 60) -- without stoppages, with stoppages (in kmph)
let time1 := (train1.1 - train1.2 : Real) / train1.1
let time2 := (train2.1 - train2.2 : Real) / train2.1
let time3 := (train3.1 - train3.2 : Real) / train3.1
let total_time := time1 + time2 + time3
(total_time / 3) * 60 -- convert hours to minutes

theorem avg_stoppage_time_is_20_minutes :
  avg_stoppage_time = 20 :=
sorry

end avg_stoppage_time_is_20_minutes_l146_146994


namespace sampling_is_systematic_l146_146977

-- Define the total seats in each row and the total number of rows
def total_seats_per_row : ℕ := 25
def total_rows : ℕ := 30

-- Define a function to identify if the sampling is systematic
def is_systematic_sampling (sample_count : ℕ) (n : ℕ) (interval : ℕ) : Prop :=
  interval = total_seats_per_row ∧ sample_count = total_rows

-- Define the count and interval for the problem
def sample_count : ℕ := 30
def sampling_interval : ℕ := 25

-- Theorem statement: Given the conditions, it is systematic sampling
theorem sampling_is_systematic :
  is_systematic_sampling sample_count total_rows sampling_interval = true :=
sorry

end sampling_is_systematic_l146_146977


namespace largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l146_146858

theorem largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19 : 
  ∃ p : ℕ, Prime p ∧ p = 19 ∧ ∀ q : ℕ, Prime q → q ∣ (18^3 + 15^4 - 3^7) → q ≤ 19 :=
sorry

end largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l146_146858


namespace no_integer_roots_l146_146349

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 11 * x + 20 ≠ 0 := 
by
  sorry

end no_integer_roots_l146_146349


namespace sum_first_n_terms_arithmetic_sequence_l146_146024

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 2 + a 4 = 10) ∧ (∀ n : ℕ, a (n + 1) - a n = 2) → 
  (∀ n : ℕ, S n = n^2) := by
  intro h
  sorry

end sum_first_n_terms_arithmetic_sequence_l146_146024


namespace Carlos_candy_share_l146_146690

theorem Carlos_candy_share (total_candy : ℚ) (num_piles : ℕ) (piles_for_Carlos : ℕ)
  (h_total_candy : total_candy = 75 / 7)
  (h_num_piles : num_piles = 5)
  (h_piles_for_Carlos : piles_for_Carlos = 2) :
  (piles_for_Carlos * (total_candy / num_piles) = 30 / 7) :=
by
  sorry

end Carlos_candy_share_l146_146690


namespace value_of_k_l146_146943

theorem value_of_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = 2 * a * b) : k = 3 * Real.sqrt 2 :=
by
  sorry

end value_of_k_l146_146943


namespace min_value_inequality_l146_146766

theorem min_value_inequality (y1 y2 y3 : ℝ) (h_pos : 0 < y1 ∧ 0 < y2 ∧ 0 < y3) (h_sum : 2 * y1 + 3 * y2 + 4 * y3 = 120) :
  y1^2 + 4 * y2^2 + 9 * y3^2 ≥ 14400 / 29 :=
sorry

end min_value_inequality_l146_146766


namespace time_for_P_to_finish_job_alone_l146_146106

variable (T : ℝ)

theorem time_for_P_to_finish_job_alone (h1 : 0 < T) (h2 : 3 * (1 / T + 1 / 20) + 0.4 * (1 / T) = 1) : T = 4 :=
by
  sorry

end time_for_P_to_finish_job_alone_l146_146106


namespace tan_addition_formula_l146_146899

theorem tan_addition_formula (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_addition_formula_l146_146899


namespace find_x_value_l146_146394

theorem find_x_value (x : ℝ) (h : 150 + 90 + x + 90 = 360) : x = 30 := by
  sorry

end find_x_value_l146_146394


namespace range_of_m_l146_146509

theorem range_of_m (m : ℝ) :
  (¬(∀ x y : ℝ, x^2 / (25 - m) + y^2 / (m - 7) = 1 → 25 - m > 0 ∧ m - 7 > 0 ∧ 25 - m > m - 7) ∨ 
   ¬(∀ x y : ℝ, y^2 / 5 - x^2 / m = 1 → 1 < (5 + m) / 5 ∧ (5 + m) / 5 < 4)) 
  → 7 < m ∧ m < 15 :=
by
  sorry

end range_of_m_l146_146509


namespace no_such_set_exists_l146_146709

theorem no_such_set_exists :
  ¬ ∃ (A : Finset ℕ), A.card = 11 ∧
  (∀ (s : Finset ℕ), s ⊆ A → s.card = 6 → ¬ 6 ∣ s.sum id) :=
sorry

end no_such_set_exists_l146_146709


namespace equation_of_parallel_line_l146_146505

theorem equation_of_parallel_line (A : ℝ × ℝ) (c : ℝ) : 
  A = (-1, 0) → (∀ x y, 2 * x - y + 1 = 0 → 2 * x - y + c = 0) → 
  2 * (-1) - 0 + c = 0 → c = 2 :=
by
  intros A_coord parallel_line point_on_line
  sorry

end equation_of_parallel_line_l146_146505


namespace solution_set_of_xf_l146_146287

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem solution_set_of_xf (f : ℝ → ℝ) (hf_odd : is_odd_function f) (hf_one : f 1 = 0)
    (h_derivative : ∀ x > 0, (x * (deriv f x) - f x) / (x^2) > 0) :
    {x : ℝ | x * f x > 0} = {x : ℝ | x < -1 ∨ x > 1} :=
by
  sorry

end solution_set_of_xf_l146_146287


namespace fish_ranking_l146_146426

def ranks (P V K T : ℕ) : Prop :=
  P < K ∧ K < T ∧ T < V

theorem fish_ranking (P V K T : ℕ) (h1 : K < T) (h2 : P + V = K + T) (h3 : P + T < V + K) : ranks P V K T :=
by
  sorry

end fish_ranking_l146_146426


namespace range_of_m_l146_146140

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end range_of_m_l146_146140


namespace total_selling_price_of_cloth_l146_146436

theorem total_selling_price_of_cloth
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (total_meters : ℕ)
  (total_selling_price : ℕ) :
  profit_per_meter = 7 →
  cost_price_per_meter = 118 →
  total_meters = 80 →
  total_selling_price = (cost_price_per_meter + profit_per_meter) * total_meters →
  total_selling_price = 10000 :=
by
  intros h_profit h_cost h_total h_selling_price
  rw [h_profit, h_cost, h_total] at h_selling_price
  exact h_selling_price

end total_selling_price_of_cloth_l146_146436


namespace ratio_of_areas_l146_146238

theorem ratio_of_areas (AB CD AH BG CF DG S_ABCD S_KLMN : ℕ)
  (h1 : AB = 15)
  (h2 : CD = 19)
  (h3 : DG = 17)
  (condition1 : S_ABCD = 17 * (AH + BG))
  (midpoints_AH_CF : AH = BG)
  (midpoints_CF_CD : CF = CD/2)
  (condition2 : (∃ h₁ h₂ : ℕ, S_KLMN = h₁ * AH + h₂ * CF / 2))
  (h_case1 : (S_KLMN = (AH + BG + CD)))
  (h_case2 : (S_KLMN = (AB + (CD - DG)))) :
  (S_ABCD / S_KLMN = 2 / 3 ∨ S_ABCD / S_KLMN = 2) :=
  sorry

end ratio_of_areas_l146_146238


namespace find_a_l146_146992

noncomputable def curve (x a : ℝ) : ℝ := 1/x + (Real.log x)/a
noncomputable def curve_derivative (x a : ℝ) : ℝ := 
  (-1/(x^2)) + (1/(a * x))

theorem find_a (a : ℝ) : 
  (curve_derivative 1 a = 3/2) ∧ ((∃ l : ℝ, curve 1 a = l) → ∃ m : ℝ, m * (-2/3) = -1)  → a = 2/5 :=
by
  sorry

end find_a_l146_146992


namespace diagonals_of_square_equal_proof_l146_146359

-- Let us define the conditions
def square (s : Type) : Prop := True -- Placeholder for the actual definition of square
def parallelogram (p : Type) : Prop := True -- Placeholder for the actual definition of parallelogram
def diagonals_equal (q : Type) : Prop := True -- Placeholder for the property that diagonals are equal

-- Given conditions
axiom square_is_parallelogram {s : Type} (h1 : square s) : parallelogram s
axiom diagonals_of_parallelogram_equal {p : Type} (h2 : parallelogram p) : diagonals_equal p
axiom diagonals_of_square_equal {s : Type} (h3 : square s) : diagonals_equal s

-- Proof statement
theorem diagonals_of_square_equal_proof (s : Type) (h1 : square s) : diagonals_equal s :=
by
  apply diagonals_of_square_equal h1

end diagonals_of_square_equal_proof_l146_146359


namespace mod_equivalence_l146_146389

theorem mod_equivalence (x y m : ℤ) (h1 : x ≡ 25 [ZMOD 60]) (h2 : y ≡ 98 [ZMOD 60]) (h3 : m = 167) :
  x - y ≡ m [ZMOD 60] :=
sorry

end mod_equivalence_l146_146389


namespace sum_of_reciprocals_negative_l146_146088

theorem sum_of_reciprocals_negative {a b c : ℝ} (h₁ : a + b + c = 0) (h₂ : a * b * c > 0) :
  1/a + 1/b + 1/c < 0 :=
sorry

end sum_of_reciprocals_negative_l146_146088


namespace decreasing_condition_l146_146812

noncomputable def f (a x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem decreasing_condition (a : ℝ) :
  (∀ x > 1, (Real.log x - 1) / (Real.log x)^2 + a ≤ 0) → a ≤ -1/4 := by
  sorry

end decreasing_condition_l146_146812


namespace lines_parallel_l146_146471

theorem lines_parallel (a : ℝ) 
  (h₁ : (∀ x y : ℝ, ax + (a + 2) * y + 2 = 0)) 
  (h₂ : (∀ x y : ℝ, x + a * y + 1 = 0)) 
  : a = -1 :=
sorry

end lines_parallel_l146_146471


namespace undefined_expression_l146_146279

theorem undefined_expression (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_expression_l146_146279


namespace digits_exceed_10_power_15_l146_146297

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem digits_exceed_10_power_15 (x : ℝ) 
  (h : log3 (log2 (log2 x)) = 3) : log10 x > 10^15 := 
sorry

end digits_exceed_10_power_15_l146_146297


namespace percentage_error_in_area_l146_146329

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := s * (1 + 0.03)
  let A := s * s
  let A' := s' * s'
  ((A' - A) / A) * 100 = 6.09 :=
by
  sorry

end percentage_error_in_area_l146_146329


namespace roots_poly_sum_l146_146361

noncomputable def Q (z : ℂ) (a b c : ℝ) : ℂ := z^3 + (a:ℂ)*z^2 + (b:ℂ)*z + (c:ℂ)

theorem roots_poly_sum (a b c : ℝ) (u : ℂ)
  (h1 : u.im = 0) -- Assuming u is a real number
  (h2 : Q (u + 5 * Complex.I) a b c = 0)
  (h3 : Q (u + 15 * Complex.I) a b c = 0)
  (h4 : Q (2 * u - 6) a b c = 0) :
  a + b + c = -196 := by
  sorry

end roots_poly_sum_l146_146361


namespace no_solution_fraction_eq_l146_146990

theorem no_solution_fraction_eq (m : ℝ) : 
  ¬(∃ x : ℝ, x ≠ -1 ∧ 3 * x / (x + 1) = m / (x + 1) + 2) ↔ m = -3 :=
by
  sorry

end no_solution_fraction_eq_l146_146990


namespace smallest_natural_with_properties_l146_146025

theorem smallest_natural_with_properties :
  ∃ n : ℕ, (∃ N : ℕ, n = 10 * N + 6) ∧ 4 * (10 * N + 6) = 6 * 10^(5 : ℕ) + N ∧ n = 153846 := sorry

end smallest_natural_with_properties_l146_146025


namespace inscribed_circle_radius_in_quadrilateral_pyramid_l146_146147

theorem inscribed_circle_radius_in_quadrilateral_pyramid
  (a : ℝ) (α : ℝ)
  (h_pos : 0 < a) (h_α : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r = a * Real.sqrt 2 / (1 + 2 * Real.cos α + Real.sqrt (4 * Real.cos α ^ 2 + 1)) :=
by
  sorry

end inscribed_circle_radius_in_quadrilateral_pyramid_l146_146147


namespace tan_five_pi_over_four_l146_146933

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l146_146933


namespace age_of_15th_student_l146_146715

theorem age_of_15th_student (avg_age_15 : ℕ) (avg_age_6 : ℕ) (avg_age_8 : ℕ) (num_students_15 : ℕ) (num_students_6 : ℕ) (num_students_8 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_6 : avg_age_6 = 14) 
  (h_avg_8 : avg_age_8 = 16) 
  (h_num_15 : num_students_15 = 15) 
  (h_num_6 : num_students_6 = 6) 
  (h_num_8 : num_students_8 = 8) : 
  ∃ age_15th_student : ℕ, age_15th_student = 13 := 
by
  sorry


end age_of_15th_student_l146_146715


namespace find_solutions_l146_146224

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Int.gcd (Int.gcd a b) c = 1 ∧
  (a + b + c) ∣ (a^12 + b^12 + c^12) ∧
  (a + b + c) ∣ (a^23 + b^23 + c^23) ∧
  (a + b + c) ∣ (a^11004 + b^11004 + c^11004)

theorem find_solutions :
  (is_solution 1 1 1) ∧ (is_solution 1 1 4) ∧ 
  (∀ a b c : ℕ, is_solution a b c → 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4)) := 
sorry

end find_solutions_l146_146224


namespace additional_teddies_per_bunny_l146_146462

theorem additional_teddies_per_bunny (teddies bunnies koala total_mascots: ℕ) 
  (h1 : teddies = 5) 
  (h2 : bunnies = 3 * teddies) 
  (h3 : koala = 1) 
  (h4 : total_mascots = 51): 
  (total_mascots - (teddies + bunnies + koala)) / bunnies = 2 := 
by 
  sorry

end additional_teddies_per_bunny_l146_146462


namespace find_initial_children_l146_146226

variables (x y : ℕ)

-- Defining the conditions 
def initial_children_on_bus (x : ℕ) : Prop :=
  ∃ y : ℕ, x - 68 + y = 12 ∧ 68 - y = 24 + y

-- Theorem statement
theorem find_initial_children : initial_children_on_bus x → x = 58 :=
by
  -- Skipping the proof for now
  sorry

end find_initial_children_l146_146226


namespace total_cost_sandwiches_and_sodas_l146_146831

theorem total_cost_sandwiches_and_sodas :
  let price_sandwich : Real := 2.49
  let price_soda : Real := 1.87
  let quantity_sandwich : ℕ := 2
  let quantity_soda : ℕ := 4
  (quantity_sandwich * price_sandwich + quantity_soda * price_soda) = 12.46 := 
by
  sorry

end total_cost_sandwiches_and_sodas_l146_146831


namespace sufficient_but_not_necessary_not_necessary_l146_146196

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end sufficient_but_not_necessary_not_necessary_l146_146196


namespace dot_product_correct_l146_146143

theorem dot_product_correct:
  let a : ℝ × ℝ := (5, -7)
  let b : ℝ × ℝ := (-6, -4)
  (a.1 * b.1) + (a.2 * b.2) = -2 := by
sorry

end dot_product_correct_l146_146143


namespace no_real_roots_of_quad_eq_l146_146911

theorem no_real_roots_of_quad_eq (k : ℝ) : ¬(k ≠ 0 ∧ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0) :=
by
  sorry

end no_real_roots_of_quad_eq_l146_146911


namespace mixed_number_arithmetic_l146_146263

theorem mixed_number_arithmetic :
  26 * (2 + 4 / 7 - (3 + 1 / 3)) + (3 + 1 / 5 + (2 + 3 / 7)) = -14 - 223 / 735 :=
by
  sorry

end mixed_number_arithmetic_l146_146263


namespace calculate_expression_l146_146312

theorem calculate_expression : 
  let a := 0.82
  let b := 0.1
  a^3 - b^3 / (a^2 + 0.082 + b^2) = 0.7201 := sorry

end calculate_expression_l146_146312


namespace cricket_throwers_l146_146115

theorem cricket_throwers (T L R : ℕ) 
  (h1 : T + L + R = 55)
  (h2 : T + R = 49) 
  (h3 : L = (1/3) * (L + R))
  (h4 : R = (2/3) * (L + R)) :
  T = 37 :=
by sorry

end cricket_throwers_l146_146115


namespace length_of_AB_l146_146089

theorem length_of_AB (x1 y1 x2 y2 : ℝ) 
  (h_parabola_A : y1^2 = 8 * x1) 
  (h_focus_line_A : y1 = 2 * (x1 - 2)) 
  (h_parabola_B : y2^2 = 8 * x2) 
  (h_focus_line_B : y2 = 2 * (x2 - 2)) 
  (h_sum_x : x1 + x2 = 6) : 
  |x1 - x2| = 10 :=
sorry

end length_of_AB_l146_146089


namespace solve_complex_eq_l146_146277

open Complex

theorem solve_complex_eq (z : ℂ) (h : (3 - 4 * I) * z = 5) : z = (3 / 5) + (4 / 5) * I :=
by
  sorry

end solve_complex_eq_l146_146277


namespace inverse_function_l146_146319

variable (x : ℝ)

def f (x : ℝ) : ℝ := (x^(1 / 3)) + 1
def g (x : ℝ) : ℝ := (x - 1)^3

theorem inverse_function :
  ∀ x, f (g x) = x ∧ g (f x) = x :=
by
  -- Proof goes here
  sorry

end inverse_function_l146_146319


namespace weight_of_a_is_75_l146_146589

theorem weight_of_a_is_75 (a b c d e : ℕ) 
  (h1 : (a + b + c) / 3 = 84) 
  (h2 : (a + b + c + d) / 4 = 80) 
  (h3 : e = d + 3) 
  (h4 : (b + c + d + e) / 4 = 79) : 
  a = 75 :=
by
  -- Proof omitted
  sorry

end weight_of_a_is_75_l146_146589


namespace hands_coincide_again_l146_146593

-- Define the angular speeds of minute and hour hands
def speed_minute_hand : ℝ := 6
def speed_hour_hand : ℝ := 0.5

-- Define the initial condition: coincidence at midnight
def initial_time : ℝ := 0

-- Define the function that calculates the angle of the minute hand at time t
def angle_minute_hand (t : ℝ) : ℝ := speed_minute_hand * t

-- Define the function that calculates the angle of the hour hand at time t
def angle_hour_hand (t : ℝ) : ℝ := speed_hour_hand * t

-- Define the time at which the hands coincide again after midnight
noncomputable def coincidence_time : ℝ := 720 / 11

-- The proof problem statement: The hands coincide again at coincidence_time minutes
theorem hands_coincide_again : 
  angle_minute_hand coincidence_time = angle_hour_hand coincidence_time + 360 :=
sorry

end hands_coincide_again_l146_146593


namespace solve_for_x_l146_146295

theorem solve_for_x (x : ℝ) :
  (x + 3)^3 = -64 → x = -7 :=
by
  intro h
  sorry

end solve_for_x_l146_146295


namespace sally_seashells_l146_146049

theorem sally_seashells (T S: ℕ) (hT : T = 37) (h_total : T + S = 50) : S = 13 := by
  -- Skip the proof
  sorry

end sally_seashells_l146_146049


namespace gcf_of_lcm_9_21_and_10_22_eq_one_l146_146375

theorem gcf_of_lcm_9_21_and_10_22_eq_one :
  Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 :=
sorry

end gcf_of_lcm_9_21_and_10_22_eq_one_l146_146375


namespace temperature_at_midnight_l146_146968

theorem temperature_at_midnight :
  ∀ (morning_temp noon_rise midnight_drop midnight_temp : ℤ),
    morning_temp = -3 →
    noon_rise = 6 →
    midnight_drop = -7 →
    midnight_temp = morning_temp + noon_rise + midnight_drop →
    midnight_temp = -4 :=
by
  intros
  sorry

end temperature_at_midnight_l146_146968


namespace sequence_general_term_l146_146417

theorem sequence_general_term (n : ℕ) : 
  (∃ (f : ℕ → ℕ), (∀ k, f k = k^2) ∧ (∀ m, f m = m^2)) :=
by
  -- Given the sequence 1, 4, 9, 16, 25, ...
  sorry

end sequence_general_term_l146_146417


namespace necessary_sufficient_condition_l146_146178

theorem necessary_sufficient_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℚ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
sorry

end necessary_sufficient_condition_l146_146178


namespace weight_loss_percentage_l146_146757

theorem weight_loss_percentage {W : ℝ} (hW : 0 < W) :
  (((W - ((1 - 0.13 + 0.02 * (1 - 0.13)) * W)) / W) * 100) = 11.26 :=
by
  sorry

end weight_loss_percentage_l146_146757


namespace paintings_in_four_weeks_l146_146119

theorem paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → num_weeks = 4 → 
  (hours_per_week / hours_per_painting) * num_weeks = 40 :=
by
  -- Sorry is used since we are not providing the proof
  sorry

end paintings_in_four_weeks_l146_146119


namespace cosine_identity_l146_146969

theorem cosine_identity (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (π / 2 + α) = -1 / 3 := by
  sorry

end cosine_identity_l146_146969


namespace inequality_of_function_l146_146594

theorem inequality_of_function (x : ℝ) : 
  (1 / 2 : ℝ) ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 / 2 : ℝ) :=
sorry

end inequality_of_function_l146_146594


namespace find_X_l146_146036

theorem find_X : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 ∧ X = 1016.67 := by
  sorry

end find_X_l146_146036


namespace christina_has_three_snakes_l146_146567

def snake_lengths : List ℕ := [24, 16, 10]

def total_length : ℕ := 50

theorem christina_has_three_snakes
  (lengths : List ℕ)
  (total : ℕ)
  (h_lengths : lengths = snake_lengths)
  (h_total : total = total_length)
  : lengths.length = 3 :=
by
  sorry

end christina_has_three_snakes_l146_146567


namespace factorization_x6_minus_5x4_plus_8x2_minus_4_l146_146523

theorem factorization_x6_minus_5x4_plus_8x2_minus_4 (x : ℝ) :
  x^6 - 5 * x^4 + 8 * x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 :=
sorry

end factorization_x6_minus_5x4_plus_8x2_minus_4_l146_146523


namespace largest_even_not_sum_of_two_composite_odds_l146_146560

-- Definitions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ k, k > 1 ∧ k < n ∧ n % k = 0

-- Theorem statement
theorem largest_even_not_sum_of_two_composite_odds :
  ∀ n : ℕ, is_even n → n > 0 → (¬ (∃ a b : ℕ, is_odd a ∧ is_odd b ∧ is_composite a ∧ is_composite b ∧ n = a + b)) ↔ n = 38 := 
by
  sorry

end largest_even_not_sum_of_two_composite_odds_l146_146560


namespace solve_students_in_fifth_grade_class_l146_146098

noncomputable def number_of_students_in_each_fifth_grade_class 
    (third_grade_classes : ℕ) 
    (third_grade_students_per_class : ℕ)
    (fourth_grade_classes : ℕ) 
    (fourth_grade_students_per_class : ℕ) 
    (fifth_grade_classes : ℕ)
    (total_lunch_cost : ℝ)
    (hamburger_cost : ℝ)
    (carrot_cost : ℝ)
    (cookie_cost : ℝ) : ℝ :=
  
  let total_students_third := third_grade_classes * third_grade_students_per_class
  let total_students_fourth := fourth_grade_classes * fourth_grade_students_per_class
  let lunch_cost_per_student := hamburger_cost + carrot_cost + cookie_cost
  let total_students := total_students_third + total_students_fourth
  let total_cost_third_fourth := total_students * lunch_cost_per_student
  let total_cost_fifth := total_lunch_cost - total_cost_third_fourth
  let fifth_grade_students := total_cost_fifth / lunch_cost_per_student
  let students_per_fifth_class := fifth_grade_students / fifth_grade_classes
  students_per_fifth_class

theorem solve_students_in_fifth_grade_class : 
    number_of_students_in_each_fifth_grade_class 5 30 4 28 4 1036 2.10 0.50 0.20 = 27 := 
by 
  sorry

end solve_students_in_fifth_grade_class_l146_146098


namespace calculate_T1_T2_l146_146691

def triangle (a b c : ℤ) : ℤ := a + b - 2 * c

def T1 := triangle 3 4 5
def T2 := triangle 6 8 2

theorem calculate_T1_T2 : 2 * T1 + 3 * T2 = 24 :=
  by
    sorry

end calculate_T1_T2_l146_146691


namespace joe_lists_count_l146_146438

theorem joe_lists_count : ∃ (n : ℕ), n = 15 * 14 := sorry

end joe_lists_count_l146_146438


namespace problem_1_problem_2_problem_3_l146_146243

-- Problem 1
theorem problem_1 (x : ℝ) (h : 4.8 - 3 * x = 1.8) : x = 1 :=
by { sorry }

-- Problem 2
theorem problem_2 (x : ℝ) (h : (1 / 8) / (1 / 5) = x / 24) : x = 15 :=
by { sorry }

-- Problem 3
theorem problem_3 (x : ℝ) (h : 7.5 * x + 6.5 * x = 2.8) : x = 0.2 :=
by { sorry }

end problem_1_problem_2_problem_3_l146_146243


namespace arithmetic_sequence_general_formula_l146_146271

noncomputable def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_formula {a : ℕ → ℤ} (h_seq : arithmetic_seq a) 
  (h_a1 : a 1 = 6) (h_a3a5 : a 3 + a 5 = 0) : 
  ∀ n, a n = 8 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l146_146271


namespace ratio_b_to_c_l146_146477

theorem ratio_b_to_c (x a b c : ℤ) 
    (h1 : x = 100 * a + 10 * b + c)
    (h2 : a > 0)
    (h3 : 999 - x = 241) : (b : ℚ) / c = 5 / 8 :=
by
  sorry

end ratio_b_to_c_l146_146477


namespace trip_duration_l146_146995

noncomputable def start_time : ℕ := 11 * 60 + 25 -- 11:25 a.m. in minutes
noncomputable def end_time : ℕ := 16 * 60 + 43 + 38 / 60 -- 4:43:38 p.m. in minutes

theorem trip_duration :
  end_time - start_time = 5 * 60 + 18 := 
sorry

end trip_duration_l146_146995


namespace geometric_prog_common_ratio_one_l146_146091

variable {x y z : ℝ}
variable {r : ℝ}

theorem geometric_prog_common_ratio_one
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (hgeom : ∃ a : ℝ, a = x * (y - z) ∧ a * r = y * (z - x) ∧ a * r^2 = z * (x - y))
  (hprod : (x * (y - z)) * (y * (z - x)) * (z * (x - y)) * r^3 = (y * (z - x))^2) : 
  r = 1 := sorry

end geometric_prog_common_ratio_one_l146_146091


namespace intersection_eq_l146_146865

-- Definitions of sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

-- The theorem statement
theorem intersection_eq : A ∩ B = {1} :=
by
  unfold A B
  sorry

end intersection_eq_l146_146865


namespace cyclists_meet_at_starting_point_l146_146980

-- Define the conditions: speeds of cyclists and the circumference of the circle
def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def circumference : ℝ := 300

-- Define the total speed by summing individual speeds
def relative_speed : ℝ := speed_cyclist1 + speed_cyclist2

-- Define the time required to meet at the starting point
def meeting_time : ℝ := 20

-- The theorem statement which states that given the conditions, the cyclists will meet after 20 seconds
theorem cyclists_meet_at_starting_point :
  meeting_time = circumference / relative_speed :=
sorry

end cyclists_meet_at_starting_point_l146_146980


namespace amount_of_sugar_l146_146632

-- Let ratio_sugar_flour be the ratio of sugar to flour.
def ratio_sugar_flour : ℕ := 10

-- Let flour be the amount of flour used in ounces.
def flour : ℕ := 5

-- Let sugar be the amount of sugar used in ounces.
def sugar (ratio_sugar_flour : ℕ) (flour : ℕ) : ℕ := ratio_sugar_flour * flour

-- The proof goal: given the conditions, prove that the amount of sugar used is 50 ounces.
theorem amount_of_sugar (h_ratio : ratio_sugar_flour = 10) (h_flour : flour = 5) : sugar ratio_sugar_flour flour = 50 :=
by
  -- Proof omitted.
  sorry
 
end amount_of_sugar_l146_146632


namespace day_of_week_150th_day_of_year_N_minus_1_l146_146041

/-- Given that the 250th day of year N is a Friday and year N is a leap year,
    prove that the 150th day of year N-1 is a Friday. -/
theorem day_of_week_150th_day_of_year_N_minus_1
  (N : ℕ) 
  (H1 : (250 % 7 = 5) → true)  -- Condition that 250th day is five days after Sunday (Friday).
  (H2 : 366 % 7 = 2)           -- Condition that year N is a leap year with 366 days.
  (H3 : (N - 1) % 7 = (N - 1) % 7) -- Used for year transition check.
  : 150 % 7 = 5 := sorry       -- Proving that the 150th of year N-1 is Friday.

end day_of_week_150th_day_of_year_N_minus_1_l146_146041


namespace sum_of_fractions_irreducible_l146_146770

noncomputable def is_irreducible (num denom : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ num ∧ d ∣ denom → d = 1

theorem sum_of_fractions_irreducible (a b : ℕ) (h_coprime : Nat.gcd a b = 1) :
  is_irreducible (2 * a + b) (a * (a + b)) :=
by
  sorry

end sum_of_fractions_irreducible_l146_146770


namespace find_triangle_angles_l146_146107

theorem find_triangle_angles 
  (α β γ : ℝ)
  (a b : ℝ)
  (h1 : γ = 2 * α)
  (h2 : b = 2 * a)
  (h3 : α + β + γ = 180) :
  α = 30 ∧ β = 90 ∧ γ = 60 := 
by 
  sorry

end find_triangle_angles_l146_146107


namespace cost_formula_correct_l146_146747

def total_cost (P : ℕ) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem cost_formula_correct (P : ℕ) : 
  total_cost P = (if P ≤ 2 then 15 else 15 + 5 * (P - 2)) :=
by 
  exact rfl

end cost_formula_correct_l146_146747


namespace coconut_grove_yield_l146_146247

theorem coconut_grove_yield (x Y : ℕ) (h1 : x = 10)
  (h2 : (x + 2) * 30 + x * Y + (x - 2) * 180 = 3 * x * 100) : Y = 120 :=
by
  -- Proof to be provided
  sorry

end coconut_grove_yield_l146_146247


namespace modified_determinant_l146_146185

def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem modified_determinant (x y z w : ℝ)
  (h : determinant_2x2 x y z w = 6) :
  determinant_2x2 x (5 * x + 4 * y) z (5 * z + 4 * w) = 24 := by
  sorry

end modified_determinant_l146_146185


namespace compute_expression_l146_146403

theorem compute_expression :
  120 * 2400 - 20 * 2400 - 100 * 2400 = 0 :=
sorry

end compute_expression_l146_146403


namespace ratio_of_B_to_C_l146_146158

variables (A B C : ℕ)

-- Conditions from the problem
axiom h1 : A = B + 2
axiom h2 : A + B + C = 12
axiom h3 : B = 4

-- Goal: Prove that the ratio of B's age to C's age is 2
theorem ratio_of_B_to_C : B / C = 2 :=
by {
  sorry
}

end ratio_of_B_to_C_l146_146158


namespace compute_expression_l146_146485

theorem compute_expression : 7 * (1 / 21) * 42 = 14 :=
by
  sorry

end compute_expression_l146_146485


namespace sequence_either_increases_or_decreases_l146_146571

theorem sequence_either_increases_or_decreases {x : ℕ → ℝ} (x1_pos : 0 < x 1) (x1_ne_one : x 1 ≠ 1) 
    (recurrence : ∀ n : ℕ, x (n + 1) = x n * (x n ^ 2 + 3) / (3 * x n ^ 2 + 1)) :
    (∀ n : ℕ, x n < x (n + 1)) ∨ (∀ n : ℕ, x n > x (n + 1)) :=
sorry

end sequence_either_increases_or_decreases_l146_146571


namespace max_n_l146_146689

noncomputable def seq_a (n : ℕ) : ℤ := 3 * n - 1

noncomputable def seq_b (n : ℕ) : ℤ := 2 * n - 3

noncomputable def sum_T (n : ℕ) : ℤ := n * (3 * n + 1) / 2

noncomputable def sum_S (n : ℕ) : ℤ := n^2 - 2 * n

theorem max_n (n : ℕ) :
  ∃ n_max : ℕ, T_n < 20 * seq_b n ∧ (∀ m : ℕ, m > n_max → T_n ≥ 20 * seq_b n) :=
  sorry

end max_n_l146_146689


namespace distance_l1_l2_l146_146570

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

theorem distance_l1_l2 :
  distance_between_parallel_lines 3 4 (-3) 2 = 1 :=
by
  -- Add the conditions needed to assert the theorem
  let l1 := (3, 4, -3) -- definition of line l1
  let l2 := (3, 4, 2)  -- definition of line l2
  -- Calculate the distance using the given formula
  let d := distance_between_parallel_lines 3 4 (-3) 2
  -- Assert the result
  show d = 1
  sorry

end distance_l1_l2_l146_146570


namespace shorter_piece_length_l146_146618

/-- A 69-inch board is cut into 2 pieces. One piece is 2 times the length of the other.
    Prove that the length of the shorter piece is 23 inches. -/
theorem shorter_piece_length (x : ℝ) :
  let shorter := x
  let longer := 2 * x
  (shorter + longer = 69) → shorter = 23 :=
by
  intro h
  sorry

end shorter_piece_length_l146_146618


namespace probability_of_6_heads_in_10_flips_l146_146486

theorem probability_of_6_heads_in_10_flips :
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := Nat.choose 10 6
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 210 / 1024 :=
by
  sorry

end probability_of_6_heads_in_10_flips_l146_146486


namespace perimeter_region_l146_146841

theorem perimeter_region (rectangle_height : ℕ) (height_eq_sixteen : rectangle_height = 16) (rect_area_eq : 12 * rectangle_height = 192) (total_area_eq : 12 * rectangle_height - 60 = 132):
  (rectangle_height + 12 + 4 + 6 + 10 * 2) = 54 :=
by
  have h1 : 12 * 16 = 192 := by sorry
  exact sorry


end perimeter_region_l146_146841


namespace inverse_proportional_example_l146_146204

variable (x y : ℝ)

def inverse_proportional (x y : ℝ) := y = 8 / (x - 1)

theorem inverse_proportional_example
  (h1 : y = 4)
  (h2 : x = 3) :
  inverse_proportional x y :=
by
  sorry

end inverse_proportional_example_l146_146204


namespace find_gross_salary_l146_146810

open Real

noncomputable def bill_take_home_salary : ℝ := 40000
noncomputable def property_tax : ℝ := 2000
noncomputable def sales_tax : ℝ := 3000
noncomputable def income_tax_rate : ℝ := 0.10

theorem find_gross_salary (gross_salary : ℝ) :
  bill_take_home_salary = gross_salary - (income_tax_rate * gross_salary + property_tax + sales_tax) →
  gross_salary = 50000 :=
by
  sorry

end find_gross_salary_l146_146810


namespace quadratic_roots_sum_square_l146_146422

theorem quadratic_roots_sum_square (u v : ℝ) 
  (h1 : u^2 - 5*u + 3 = 0) (h2 : v^2 - 5*v + 3 = 0) 
  (h3 : u ≠ v) : u^2 + v^2 + u*v = 22 := 
by
  sorry

end quadratic_roots_sum_square_l146_146422


namespace vector_parallel_eq_l146_146663

theorem vector_parallel_eq (k : ℝ) (a b : ℝ × ℝ) 
  (h_a : a = (k, 2)) (h_b : b = (1, 1)) (h_parallel : (∃ c : ℝ, a = (c * 1, c * 1))) : k = 2 := by
  sorry

end vector_parallel_eq_l146_146663


namespace Steven_more_than_Jill_l146_146032

variable (Jill Jake Steven : ℕ)

def Jill_peaches : Jill = 87 := by sorry
def Jake_peaches_more : Jake = Jill + 13 := by sorry
def Steven_peaches_more : Steven = Jake + 5 := by sorry

theorem Steven_more_than_Jill : Steven - Jill = 18 := by
  -- Proof steps to be filled
  sorry

end Steven_more_than_Jill_l146_146032


namespace solution_set_l146_146055

-- Define the conditions
variable (f : ℝ → ℝ)
variable (odd_func : ∀ x : ℝ, f (-x) = -f x)
variable (increasing_pos : ∀ a b : ℝ, 0 < a → 0 < b → a < b → f a < f b)
variable (f_neg3_zero : f (-3) = 0)

-- State the theorem
theorem solution_set (x : ℝ) : x * f x < 0 ↔ (-3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3) :=
sorry

end solution_set_l146_146055


namespace quadratic_matches_sin_values_l146_146336

noncomputable def quadratic_function (x : ℝ) : ℝ := - (4 / (Real.pi ^ 2)) * (x ^ 2) + (4 / Real.pi) * x

theorem quadratic_matches_sin_values :
  (quadratic_function 0 = Real.sin 0) ∧
  (quadratic_function (Real.pi / 2) = Real.sin (Real.pi / 2)) ∧
  (quadratic_function Real.pi = Real.sin Real.pi) :=
by
  sorry

end quadratic_matches_sin_values_l146_146336


namespace sum_of_numbers_l146_146198

theorem sum_of_numbers : 
  5678 + 6785 + 7856 + 8567 = 28886 := 
by 
  sorry

end sum_of_numbers_l146_146198


namespace correct_option_l146_146828

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end correct_option_l146_146828


namespace recurrence_sequence_a5_l146_146404

theorem recurrence_sequence_a5 :
  ∃ a : ℕ → ℚ, (a 1 = 5 ∧ (∀ n, a (n + 1) = 1 + 1 / a n) ∧ a 5 = 28 / 17) :=
  sorry

end recurrence_sequence_a5_l146_146404


namespace jason_total_hours_l146_146921

variables (hours_after_school hours_total : ℕ)

def earnings_after_school := 4 * hours_after_school
def earnings_saturday := 6 * 8
def total_earnings := earnings_after_school + earnings_saturday

theorem jason_total_hours :
  4 * hours_after_school + earnings_saturday = 88 →
  hours_total = hours_after_school + 8 →
  total_earnings = 88 →
  hours_total = 18 :=
by
  intros h1 h2 h3
  sorry

end jason_total_hours_l146_146921


namespace vasya_wins_game_l146_146062

/- Define the conditions of the problem -/

def grid_size : Nat := 9
def total_matchsticks : Nat := 2 * grid_size * (grid_size + 1)

/-- Given a game on a 9x9 matchstick grid with Petya going first, 
    Prove that Vasya can always win by ensuring that no whole 1x1 
    squares remain in the end. -/
theorem vasya_wins_game : 
  ∃ strategy_for_vasya : Nat → Nat → Prop, -- Define a strategy for Vasya
  ∀ (matchsticks_left : Nat),
  matchsticks_left % 2 = 1 →     -- Petya makes a move and the remaining matchsticks are odd
  strategy_for_vasya matchsticks_left total_matchsticks :=
sorry

end vasya_wins_game_l146_146062


namespace triangle_perimeter_l146_146789

-- Definitions for the conditions
def inscribed_circle_of_triangle_tangent_at (radius : ℝ) (DP : ℝ) (PE : ℝ) : Prop :=
  radius = 27 ∧ DP = 29 ∧ PE = 33

-- Perimeter calculation theorem
theorem triangle_perimeter (r DP PE : ℝ) (h : inscribed_circle_of_triangle_tangent_at r DP PE) : 
  ∃ perimeter : ℝ, perimeter = 774 :=
by
  sorry

end triangle_perimeter_l146_146789


namespace nth_equation_l146_146591

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := by
  sorry

end nth_equation_l146_146591


namespace distribute_coins_l146_146005

theorem distribute_coins (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 - y^2 = 16 * (x - y)) :
  x = 8 ∧ y = 8 :=
by {
  sorry
}

end distribute_coins_l146_146005


namespace tom_ate_one_pound_of_carrots_l146_146625

noncomputable def calories_from_carrots (C : ℝ) : ℝ := 51 * C
noncomputable def calories_from_broccoli (C : ℝ) : ℝ := (51 / 3) * (2 * C)
noncomputable def total_calories (C : ℝ) : ℝ :=
  calories_from_carrots C + calories_from_broccoli C

theorem tom_ate_one_pound_of_carrots :
  ∃ C : ℝ, total_calories C = 85 ∧ C = 1 :=
by
  use 1
  simp [total_calories, calories_from_carrots, calories_from_broccoli]
  sorry

end tom_ate_one_pound_of_carrots_l146_146625


namespace blue_face_probability_l146_146060

def sides : ℕ := 12
def green_faces : ℕ := 5
def blue_faces : ℕ := 4
def red_faces : ℕ := 3

theorem blue_face_probability : 
  (blue_faces : ℚ) / sides = 1 / 3 :=
by
  sorry

end blue_face_probability_l146_146060


namespace find_other_number_l146_146862

-- Definitions for the given conditions
def A : ℕ := 500
def LCM : ℕ := 3000
def HCF : ℕ := 100

-- Theorem statement: If A = 500, LCM(A, B) = 3000, and HCF(A, B) = 100, then B = 600.
theorem find_other_number (B : ℕ) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 :=
by
  sorry

end find_other_number_l146_146862


namespace counties_under_50k_perc_l146_146935

def percentage (s: String) : ℝ := match s with
  | "20k_to_49k" => 45
  | "less_than_20k" => 30
  | _ => 0

theorem counties_under_50k_perc : percentage "20k_to_49k" + percentage "less_than_20k" = 75 := by
  sorry

end counties_under_50k_perc_l146_146935


namespace find_an_l146_146248

def sequence_sum (k : ℝ) (n : ℕ) : ℝ :=
  k * n ^ 2 + n

def term_of_sequence (k : ℝ) (n : ℕ) (S_n : ℝ) (S_nm1 : ℝ) : ℝ :=
  S_n - S_nm1

theorem find_an (k : ℝ) (n : ℕ) (h₁ : n > 0) :
  term_of_sequence k n (sequence_sum k n) (sequence_sum k (n - 1)) = 2 * k * n - k + 1 :=
by
  sorry

end find_an_l146_146248


namespace product_of_two_numbers_eq_a_mul_100_a_l146_146807

def product_of_two_numbers (a : ℝ) (b : ℝ) : ℝ := a * b

theorem product_of_two_numbers_eq_a_mul_100_a (a : ℝ) (b : ℝ) (h : a + b = 100) :
    product_of_two_numbers a b = a * (100 - a) :=
by
  sorry

end product_of_two_numbers_eq_a_mul_100_a_l146_146807


namespace proof_a_in_S_l146_146353

def S : Set ℤ := {n : ℤ | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem proof_a_in_S (a : ℤ) (h1 : 3 * a ∈ S) : a ∈ S :=
sorry

end proof_a_in_S_l146_146353


namespace pedro_squares_correct_l146_146124

def squares_jesus : ℕ := 60
def squares_linden : ℕ := 75
def squares_pedro (s_jesus s_linden : ℕ) : ℕ := (s_jesus + s_linden) + 65

theorem pedro_squares_correct :
  squares_pedro squares_jesus squares_linden = 200 :=
by
  sorry

end pedro_squares_correct_l146_146124


namespace problem1_problem2_problem3_problem4_l146_146817

variable (f : ℝ → ℝ)
variables (H1 : f (-1) = 2) 
          (H2 : ∀ x, x < 0 → f x > 1)
          (H3 : ∀ x y, f (x + y) = f x * f y)

-- (1) Prove f(0) = 1
theorem problem1 : f 0 = 1 := sorry

-- (2) Prove f(-4) = 16
theorem problem2 : f (-4) = 16 := sorry

-- (3) Prove f(x) is strictly decreasing
theorem problem3 : ∀ x y, x < y → f x > f y := sorry

-- (4) Solve f(-4x^2)f(10x) ≥ 1/16
theorem problem4 : { x : ℝ | f (-4 * x ^ 2) * f (10 * x) ≥ 1 / 16 } = { x | x ≤ 1 / 2 ∨ 2 ≤ x } := sorry

end problem1_problem2_problem3_problem4_l146_146817


namespace older_brother_stamps_l146_146687

variable (y o : ℕ)

def condition1 : Prop := o = 2 * y + 1
def condition2 : Prop := o + y = 25

theorem older_brother_stamps (h1 : condition1 y o) (h2 : condition2 y o) : o = 17 :=
by
  sorry

end older_brother_stamps_l146_146687


namespace sin_cos_of_theta_l146_146984

open Real

theorem sin_cos_of_theta (θ : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4))
  (hxθ : ∃ r, r > 0 ∧ P = (r * cos θ, r * sin θ)) :
  sin θ + cos θ = 1 / 5 := 
by
  sorry

end sin_cos_of_theta_l146_146984


namespace product_zero_probability_l146_146756

noncomputable def probability_product_is_zero : ℚ :=
  let S := [-3, -1, 0, 0, 2, 5]
  let total_ways := 15 -- Calculated as 6 choose 2 taking into account repetition
  let favorable_ways := 8 -- Calculated as (2 choose 1) * (4 choose 1)
  favorable_ways / total_ways

theorem product_zero_probability : probability_product_is_zero = 8 / 15 := by
  sorry

end product_zero_probability_l146_146756


namespace remainder_when_divided_by_22_l146_146970

theorem remainder_when_divided_by_22 (n : ℤ) (h : (2 * n) % 11 = 2) : n % 22 = 1 :=
by
  sorry

end remainder_when_divided_by_22_l146_146970


namespace mila_needs_48_hours_to_earn_as_much_as_agnes_l146_146816

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end mila_needs_48_hours_to_earn_as_much_as_agnes_l146_146816


namespace salt_concentration_solution_l146_146496

theorem salt_concentration_solution
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 2 * x + 3 * y = 35)
  (h3 : 3 * y + 2 * z = 45) :
  x = 10 ∧ y = 5 ∧ z = 15 := by
  sorry

end salt_concentration_solution_l146_146496


namespace polynomial_degree_add_sub_l146_146127

noncomputable def degree (p : Polynomial ℂ) : ℕ := 
p.natDegree

variable (M N : Polynomial ℂ)

def is_fifth_degree (M : Polynomial ℂ) : Prop :=
degree M = 5

def is_third_degree (N : Polynomial ℂ) : Prop :=
degree N = 3

theorem polynomial_degree_add_sub (hM : is_fifth_degree M) (hN : is_third_degree N) :
  degree (M + N) = 5 ∧ degree (M - N) = 5 :=
by sorry

end polynomial_degree_add_sub_l146_146127


namespace rectangle_length_15_l146_146323

theorem rectangle_length_15
  (w l : ℝ)
  (h_ratio : 5 * w = 2 * l + 2 * w)
  (h_area : l * w = 150) :
  l = 15 :=
sorry

end rectangle_length_15_l146_146323


namespace quadratic_trinomial_value_at_6_l146_146314

theorem quadratic_trinomial_value_at_6 {p q : ℝ} 
  (h1 : ∃ r1 r2, r1 = q ∧ r2 = 1 + p + q ∧ r1 + r2 = -p ∧ r1 * r2 = q) : 
  (6^2 + p * 6 + q) = 31 :=
by
  sorry

end quadratic_trinomial_value_at_6_l146_146314


namespace power_sum_result_l146_146750

theorem power_sum_result : (64 ^ (-1/3 : ℝ)) + (81 ^ (-1/4 : ℝ)) = (7 / 12 : ℝ) :=
by
  have h64 : (64 : ℝ) = 2 ^ 6 := by norm_num
  have h81 : (81 : ℝ) = 3 ^ 4 := by norm_num
  sorry

end power_sum_result_l146_146750


namespace price_equation_l146_146830

variable (x : ℝ)

def first_discount (x : ℝ) : ℝ := x - 5

def second_discount (price_after_first_discount : ℝ) : ℝ := 0.8 * price_after_first_discount

theorem price_equation
  (hx : second_discount (first_discount x) = 60) :
  0.8 * (x - 5) = 60 := by
  sorry

end price_equation_l146_146830


namespace union_of_sets_l146_146871

-- Defining the sets A and B
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

-- The theorem we want to prove
theorem union_of_sets : A ∪ B = {1, 2, 3, 6} := by
  sorry

end union_of_sets_l146_146871


namespace bob_total_investment_l146_146626

variable (x : ℝ) -- the amount invested at 14%

noncomputable def total_investment_amount : ℝ :=
  let interest18 := 7000 * 0.18
  let interest14 := x * 0.14
  let total_interest := 3360
  let total_investment := 7000 + x
  total_investment

theorem bob_total_investment (h : 7000 * 0.18 + x * 0.14 = 3360) :
  total_investment_amount x = 22000 := by
  sorry

end bob_total_investment_l146_146626


namespace area_of_shaded_region_l146_146971

/-- A 4-inch by 4-inch square adjoins a 10-inch by 10-inch square. 
The bottom right corner of the smaller square touches the midpoint of the left side of the larger square. 
Prove that the area of the shaded region is 92/7 square inches. -/
theorem area_of_shaded_region : 
  let small_square_side := 4
  let large_square_side := 10 
  let midpoint := large_square_side / 2
  let height_from_midpoint := midpoint - small_square_side / 2
  let dg := (height_from_midpoint * small_square_side) / ((midpoint + height_from_midpoint))
  (small_square_side * small_square_side) - ((1/2) * dg * small_square_side) = 92 / 7 :=
by
  sorry

end area_of_shaded_region_l146_146971


namespace amount_paid_for_peaches_l146_146291

def total_spent := 23.86
def cherries_spent := 11.54
def peaches_spent := 12.32

theorem amount_paid_for_peaches :
  total_spent - cherries_spent = peaches_spent :=
sorry

end amount_paid_for_peaches_l146_146291


namespace fourth_watercraft_is_submarine_l146_146615

-- Define the conditions as Lean definitions
def same_direction_speed (w1 w2 w3 w4 : Type) : Prop :=
  -- All watercraft are moving in the same direction at the same speed
  true

def separation (w1 w2 w3 w4 : Type) (d : ℝ) : Prop :=
  -- Each pair of watercraft is separated by distance d
  true

def cargo_ship (w : Type) : Prop := true
def fishing_boat (w : Type) : Prop := true
def passenger_vessel (w : Type) : Prop := true

-- Define that the fourth watercraft is unique
def unique_watercraft (w : Type) : Prop := true

-- Proof statement that the fourth watercraft is a submarine
theorem fourth_watercraft_is_submarine 
  (w1 w2 w3 w4 : Type)
  (h1 : same_direction_speed w1 w2 w3 w4)
  (h2 : separation w1 w2 w3 w4 100)
  (h3 : cargo_ship w1)
  (h4 : fishing_boat w2)
  (h5 : passenger_vessel w3) :
  unique_watercraft w4 := 
sorry

end fourth_watercraft_is_submarine_l146_146615


namespace regular_polygon_sides_l146_146531

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l146_146531


namespace anna_cupcakes_remaining_l146_146357

theorem anna_cupcakes_remaining :
  let total_cupcakes := 60
  let cupcakes_given_away := (4 / 5 : ℝ) * total_cupcakes
  let cupcakes_after_giving := total_cupcakes - cupcakes_given_away
  let cupcakes_eaten := 3
  let cupcakes_left := cupcakes_after_giving - cupcakes_eaten
  cupcakes_left = 9 :=
by
  sorry

end anna_cupcakes_remaining_l146_146357


namespace angle_B_eq_pi_over_3_range_of_area_l146_146398

-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- And given vectors m and n represented as stated and are collinear
-- Prove that angle B is π/3
theorem angle_B_eq_pi_over_3
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (a b c A B C : ℝ)
  (m := (2 * Real.sin (A + C), - Real.sqrt 3))
  (n := (Real.cos (2 * B), 2 * Real.cos (B / 2) ^ 2 - 1))
  (collinear_m_n : m.1 * n.2 = m.2 * n.1) :
  B = Real.pi / 3 :=
sorry

-- Given side b = 1, find the range of the area S of triangle ABC
theorem range_of_area
  (a c A B C : ℝ)
  (triangle_area : ℝ)
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (hB : B = Real.pi / 3)
  (hb : b = 1)
  (cosine_theorem : 1 = a^2 + c^2 - a*c)
  (area_formula : triangle_area = (1/2) * a * c * Real.sin B) :
  0 < triangle_area ∧ triangle_area ≤ (Real.sqrt 3) / 4 :=
sorry

end angle_B_eq_pi_over_3_range_of_area_l146_146398


namespace sin_minus_cos_value_l146_146456

theorem sin_minus_cos_value
  (α : ℝ)
  (h1 : Real.tan α = (Real.sqrt 3) / 3)
  (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α - Real.cos α = -1/2 + Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_value_l146_146456


namespace brick_length_l146_146481

theorem brick_length (x : ℝ) (brick_width : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (number_of_bricks : ℕ)
  (h_brick : brick_width = 11.25) (h_brick_height : brick_height = 6)
  (h_wall : wall_length = 800) (h_wall_width : wall_width = 600) 
  (h_wall_height : wall_height = 22.5) (h_bricks_number : number_of_bricks = 1280)
  (h_eq : (wall_length * wall_width * wall_height) = (x * brick_width * brick_height) * number_of_bricks) : 
  x = 125 := by
  sorry

end brick_length_l146_146481


namespace sarah_mean_score_l146_146608

noncomputable def john_mean_score : ℝ := 86
noncomputable def john_num_tests : ℝ := 4
noncomputable def test_scores : List ℝ := [78, 80, 85, 87, 90, 95, 100]
noncomputable def total_sum : ℝ := test_scores.sum
noncomputable def sarah_num_tests : ℝ := 3

theorem sarah_mean_score :
  let john_total_score := john_mean_score * john_num_tests
  let sarah_total_score := total_sum - john_total_score
  let sarah_mean_score := sarah_total_score / sarah_num_tests
  sarah_mean_score = 90.3 :=
by
  sorry

end sarah_mean_score_l146_146608


namespace circle_representation_l146_146231

theorem circle_representation (a : ℝ): 
  (∃ (x y : ℝ), (x^2 + y^2 + 2*x + a = 0) ∧ (∃ D E F, D = 2 ∧ E = 0 ∧ F = -a ∧ (D^2 + E^2 - 4*F > 0))) ↔ (a > -1) :=
by 
  sorry

end circle_representation_l146_146231


namespace circle_center_radius_l146_146328

-- Define the necessary parameters and let Lean solve the equivalent proof problem
theorem circle_center_radius:
  (∃ a b r : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y = 1 ↔ (x + 4)^2 + (y - 1)^2 = 18) 
  ∧ a = -4 
  ∧ b = 1 
  ∧ r = 3 * Real.sqrt 2
  ∧ a + b + r = -3 + 3 * Real.sqrt 2) :=
by {
  sorry
}

end circle_center_radius_l146_146328


namespace sally_cards_l146_146726

theorem sally_cards (initial_cards dan_cards bought_cards : ℕ) (h1 : initial_cards = 27) (h2 : dan_cards = 41) (h3 : bought_cards = 20) :
  initial_cards + dan_cards + bought_cards = 88 := by
  sorry

end sally_cards_l146_146726


namespace find_n_l146_146181

-- Define the operation €
def operation (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem find_n (n : ℕ) (h : operation 8 (operation 4 n) = 640) : n = 5 :=
  by
  sorry

end find_n_l146_146181


namespace average_visitors_on_sundays_l146_146611

theorem average_visitors_on_sundays 
  (avg_other_days : ℕ) (avg_per_day : ℕ) (days_in_month : ℕ) (sundays : ℕ) (S : ℕ)
  (h_avg_other_days : avg_other_days = 240)
  (h_avg_per_day : avg_per_day = 310)
  (h_days_in_month : days_in_month = 30)
  (h_sundays : sundays = 5) :
  (sundays * S + (days_in_month - sundays) * avg_other_days = avg_per_day * days_in_month) → 
  S = 660 :=
by
  intros h
  rw [h_avg_other_days, h_avg_per_day, h_days_in_month, h_sundays] at h
  sorry

end average_visitors_on_sundays_l146_146611


namespace distance_between_trains_l146_146788

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end distance_between_trains_l146_146788


namespace equation_solution_l146_146388

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + (2 / 5) = 0 ↔ 
  a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5 :=
by sorry

end equation_solution_l146_146388


namespace eval_expression_l146_146293

theorem eval_expression : abs (-6) - (-4) + (-7) = 3 :=
by
  sorry

end eval_expression_l146_146293


namespace determine_value_of_m_l146_146290

noncomputable def conics_same_foci (m : ℝ) : Prop :=
  let c1 := Real.sqrt (4 - m^2)
  let c2 := Real.sqrt (m + 2)
  (∀ (x y : ℝ),
    (x^2 / 4 + y^2 / m^2 = 1) → (x^2 / m - y^2 / 2 = 1) → c1 = c2) → 
  m = 1

theorem determine_value_of_m : ∃ (m : ℝ), conics_same_foci m :=
sorry

end determine_value_of_m_l146_146290


namespace quadratic_completion_l146_146235

theorem quadratic_completion (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 26 * x + 81 = (x + b)^2 + c) → b + c = -101 :=
by 
  intro h
  sorry

end quadratic_completion_l146_146235


namespace lcm_of_ratio_and_hcf_l146_146356

theorem lcm_of_ratio_and_hcf (a b : ℕ) (x : ℕ) (h_ratio : a = 3 * x ∧ b = 4 * x) (h_hcf : Nat.gcd a b = 4) : Nat.lcm a b = 48 :=
by
  sorry

end lcm_of_ratio_and_hcf_l146_146356


namespace percent_of_whole_l146_146172

theorem percent_of_whole (Part Whole : ℝ) (Percent : ℝ) (hPart : Part = 160) (hWhole : Whole = 50) :
  Percent = (Part / Whole) * 100 → Percent = 320 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_whole_l146_146172


namespace max_students_exam_l146_146495

/--
An exam contains 4 multiple-choice questions, each with three options (A, B, C). Several students take the exam.
For any group of 3 students, there is at least one question where their answers are all different.
Each student answers all questions. Prove that the maximum number of students who can take the exam is 9.
-/
theorem max_students_exam (n : ℕ) (A B C : ℕ → ℕ → ℕ) (q : ℕ) :
  (∀ (s1 s2 s3 : ℕ), ∃ (q : ℕ), (1 ≤ q ∧ q ≤ 4) ∧ (A s1 q ≠ A s2 q ∧ A s1 q ≠ A s3 q ∧ A s2 q ≠ A s3 q)) →
  q = 4 ∧ (∀ s, 1 ≤ s → s ≤ n) → n ≤ 9 :=
by
  sorry

end max_students_exam_l146_146495


namespace algebraic_expression_perfect_square_l146_146344

theorem algebraic_expression_perfect_square (a : ℤ) :
  (∃ b : ℤ, ∀ x : ℤ, x^2 + (a - 1) * x + 16 = (x + b)^2) →
  (a = 9 ∨ a = -7) :=
sorry

end algebraic_expression_perfect_square_l146_146344


namespace determine_k_l146_146565

noncomputable def k_value (k : ℤ) : Prop :=
  let m := (-2 - 2) / (3 - 1)
  let b := 2 - m * 1
  let y := m * 4 + b
  let point := (4, k / 3)
  point.2 = y

theorem determine_k :
  ∃ k : ℤ, k_value k ∧ k = -12 :=
by
  use -12
  sorry

end determine_k_l146_146565


namespace find_square_digit_l146_146387

-- Define the known sum of the digits 4, 7, 6, and 9
def sum_known_digits := 4 + 7 + 6 + 9

-- Define the condition that the number 47,69square must be divisible by 6
def is_multiple_of_6 (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∧ (sum_known_digits + d) % 3 = 0

-- Theorem statement that verifies both the conditions and finds possible values of square
theorem find_square_digit (d : ℕ) (h : is_multiple_of_6 d) : d = 4 ∨ d = 8 :=
by sorry

end find_square_digit_l146_146387


namespace function_neither_even_nor_odd_l146_146083

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 - x

theorem function_neither_even_nor_odd : ¬is_even_function f ∧ ¬is_odd_function f := by
  sorry

end function_neither_even_nor_odd_l146_146083


namespace maximum_volume_l146_146099

noncomputable def volume (x : ℝ) : ℝ :=
  (48 - 2*x)^2 * x

theorem maximum_volume :
  (∀ x : ℝ, (0 < x) ∧ (x < 24) → volume x ≤ volume 8) ∧ (volume 8 = 8192) :=
by
  sorry

end maximum_volume_l146_146099


namespace football_kick_distance_l146_146332

theorem football_kick_distance (a : ℕ) (avg : ℕ) (x : ℕ)
  (h1 : a = 43)
  (h2 : avg = 37)
  (h3 : 3 * avg = a + 2 * x) :
  x = 34 :=
by
  sorry

end football_kick_distance_l146_146332


namespace meaningful_expression_l146_146093

theorem meaningful_expression (x : ℝ) : (1 / (x - 2) ≠ 0) ↔ (x ≠ 2) :=
by
  sorry

end meaningful_expression_l146_146093


namespace solve_wire_cut_problem_l146_146703

def wire_cut_problem : Prop :=
  ∃ x y : ℝ, x + y = 35 ∧ y = (2/5) * x ∧ x = 25

theorem solve_wire_cut_problem : wire_cut_problem := by
  sorry

end solve_wire_cut_problem_l146_146703


namespace Linda_journey_length_l146_146416

theorem Linda_journey_length : 
  (∃ x : ℝ, x = 30 + x * 1/4 + x * 1/7) → x = 840 / 17 :=
by
  sorry

end Linda_journey_length_l146_146416


namespace orthogonal_vectors_l146_146882

theorem orthogonal_vectors (x : ℝ) :
  (3 * x - 4 * 6 = 0) → x = 8 :=
by
  intro h
  sorry

end orthogonal_vectors_l146_146882


namespace find_x_set_eq_l146_146237

noncomputable def f : ℝ → ℝ :=
sorry -- The actual definition of f according to its properties is omitted

lemma odd_function (x : ℝ) : f (-x) = -f x :=
sorry

lemma periodic_function (x : ℝ) : f (x + 2) = -f x :=
sorry

lemma f_definition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 1 / 2 * x :=
sorry

theorem find_x_set_eq (x : ℝ) : (f x = -1 / 2) ↔ (∃ k : ℤ, x = 4 * k - 1) :=
sorry

end find_x_set_eq_l146_146237


namespace englishman_land_earnings_l146_146170

noncomputable def acres_to_square_yards (acres : ℝ) : ℝ := acres * 4840
noncomputable def square_yards_to_square_meters (sq_yards : ℝ) : ℝ := sq_yards * (0.9144 ^ 2)
noncomputable def square_meters_to_hectares (sq_meters : ℝ) : ℝ := sq_meters / 10000
noncomputable def cost_of_land (hectares : ℝ) (price_per_hectare : ℝ) : ℝ := hectares * price_per_hectare

theorem englishman_land_earnings
  (acres_owned : ℝ)
  (price_per_hectare : ℝ)
  (acre_to_yard : ℝ)
  (yard_to_meter : ℝ)
  (hectare_to_meter : ℝ)
  (h1 : acres_owned = 2)
  (h2 : price_per_hectare = 500000)
  (h3 : acre_to_yard = 4840)
  (h4 : yard_to_meter = 0.9144)
  (h5 : hectare_to_meter = 10000)
  : cost_of_land (square_meters_to_hectares (square_yards_to_square_meters (acres_to_square_yards acres_owned))) price_per_hectare = 404685.6 := sorry

end englishman_land_earnings_l146_146170


namespace not_prime_for_some_n_l146_146670

theorem not_prime_for_some_n (a : ℕ) (h : 1 < a) : ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := 
sorry

end not_prime_for_some_n_l146_146670


namespace sample_size_six_l146_146341

-- Definitions for the conditions
def num_senior_teachers : ℕ := 18
def num_first_level_teachers : ℕ := 12
def num_top_level_teachers : ℕ := 6
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_top_level_teachers

-- The proof problem statement
theorem sample_size_six (n : ℕ) (h1 : n > 0) : 
  (∀ m : ℕ, m * n = total_teachers → 
             ((n + 1) * m - 1 = 35) → False) → n = 6 :=
sorry

end sample_size_six_l146_146341


namespace admission_price_for_adults_l146_146097

-- Constants and assumptions
def children_ticket_price : ℕ := 25
def total_persons : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80

-- Definitions based on the conditions
def adults_attended : ℕ := total_persons - children_attended
def total_amount_from_children : ℕ := children_attended * children_ticket_price
def total_amount_from_adults (A : ℕ) : ℕ := total_collected_cents - total_amount_from_children
def adult_ticket_price := (total_collected_cents - total_amount_from_children) / adults_attended

-- Theorem statement to be proved
theorem admission_price_for_adults : adult_ticket_price = 60 := by
  sorry

end admission_price_for_adults_l146_146097


namespace yellow_jelly_bean_probability_l146_146110

theorem yellow_jelly_bean_probability :
  ∀ (p_red p_orange p_green p_total p_yellow : ℝ),
    p_red = 0.15 →
    p_orange = 0.35 →
    p_green = 0.25 →
    p_total = 1 →
    p_red + p_orange + p_green + p_yellow = p_total →
    p_yellow = 0.25 :=
by
  intros p_red p_orange p_green p_total p_yellow h_red h_orange h_green h_total h_sum
  sorry

end yellow_jelly_bean_probability_l146_146110


namespace base8_difference_divisible_by_7_l146_146219

theorem base8_difference_divisible_by_7 (A B : ℕ) (h₁ : A < 8) (h₂ : B < 8) (h₃ : A ≠ B) : 
  ∃ k : ℕ, k * 7 = (if 8 * A + B > 8 * B + A then 8 * A + B - (8 * B + A) else 8 * B + A - (8 * A + B)) :=
by
  sorry

end base8_difference_divisible_by_7_l146_146219


namespace cross_product_correct_l146_146103

def v : ℝ × ℝ × ℝ := (-3, 4, 5)
def w : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1,
 a.2.2 * b.1 - a.1 * b.2.2,
 a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct : cross_product v w = (21, 22, -5) :=
by
  sorry

end cross_product_correct_l146_146103


namespace average_speed_is_75_l146_146940

-- Define the conditions
def speed_first_hour : ℕ := 90
def speed_second_hour : ℕ := 60
def total_time : ℕ := 2

-- Define the average speed and prove it is equal to the given answer
theorem average_speed_is_75 : 
  (speed_first_hour + speed_second_hour) / total_time = 75 := 
by 
  -- We will skip the proof for now
  sorry

end average_speed_is_75_l146_146940


namespace tan_theta_eq_neg_two_l146_146469

theorem tan_theta_eq_neg_two (f : ℝ → ℝ) (θ : ℝ) 
  (h₁ : ∀ x, f x = Real.sin (2 * x + θ)) 
  (h₂ : ∀ x, f x + 2 * Real.cos (2 * x + θ) = -(f (-x) + 2 * Real.cos (2 * (-x) + θ))) :
  Real.tan θ = -2 :=
by
  sorry

end tan_theta_eq_neg_two_l146_146469


namespace smallest_b_for_perfect_square_l146_146054

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), 3 * b + 4 = n * n ∧ b = 7 := by
  sorry

end smallest_b_for_perfect_square_l146_146054


namespace range_of_f_l146_146909

noncomputable def f (x : ℝ) := Real.arcsin (x ^ 2 - x)

theorem range_of_f :
  Set.range f = Set.Icc (-Real.arcsin (1/4)) (Real.pi / 2) :=
sorry

end range_of_f_l146_146909


namespace exists_positive_integer_k_l146_146962

theorem exists_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → ¬ Nat.Prime (2^n * k + 1) ∧ 2^n * k + 1 > 1 :=
by
  sorry

end exists_positive_integer_k_l146_146962


namespace sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l146_146497

theorem sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms :
    let a := 63
    let b := 25
    a + b = 88 := by
  sorry

end sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l146_146497


namespace ordered_pair_and_sum_of_squares_l146_146880

theorem ordered_pair_and_sum_of_squares :
  ∃ x y : ℚ, 
    6 * x - 48 * y = 2 ∧ 
    3 * y - x = 4 ∧ 
    x ^ 2 + y ^ 2 = 442 / 25 :=
by
  sorry

end ordered_pair_and_sum_of_squares_l146_146880


namespace part1_part2_l146_146559

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l146_146559


namespace necessary_but_not_sufficient_l146_146811

-- Definitions from conditions
def abs_gt_2 (x : ℝ) : Prop := |x| > 2
def x_lt_neg_2 (x : ℝ) : Prop := x < -2

-- Statement to prove
theorem necessary_but_not_sufficient : 
  ∀ x : ℝ, (abs_gt_2 x → x_lt_neg_2 x) ∧ (¬(x_lt_neg_2 x → abs_gt_2 x)) := 
by 
  sorry

end necessary_but_not_sufficient_l146_146811


namespace probability_function_meaningful_l146_146458

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_meaningful (x : ℝ) : Prop := 1 - x^2 > 0

def measure_interval (a b : ℝ) : ℝ := b - a

theorem probability_function_meaningful:
  let interval_a := -2
  let interval_b := 1
  let meaningful_a := -1
  let meaningful_b := 1
  let total_interval := measure_interval interval_a interval_b
  let meaningful_interval := measure_interval meaningful_a meaningful_b
  let P := meaningful_interval / total_interval
  (P = (2/3)) :=
by
  sorry

end probability_function_meaningful_l146_146458


namespace big_sale_commission_l146_146669

theorem big_sale_commission (avg_increase : ℝ) (new_avg : ℝ) (num_sales : ℕ) 
  (prev_avg := new_avg - avg_increase)
  (total_prev := prev_avg * (num_sales - 1))
  (total_new := new_avg * num_sales)
  (C := total_new - total_prev) :
  avg_increase = 150 → new_avg = 250 → num_sales = 6 → C = 1000 :=
by
  intros 
  sorry

end big_sale_commission_l146_146669


namespace days_from_friday_l146_146722

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l146_146722


namespace calculate_yield_l146_146112

-- Define the conditions
def x := 6
def x_pos := 3
def x_tot := 3 * x
def nuts_x_pos := x + x_pos
def nuts_x := x
def nuts_x_neg := x - x_pos
def yield_x_pos := 60
def yield_x := 120
def avg_yield := 100

-- Calculate yields
def nuts_x_pos_yield : ℕ := nuts_x_pos * yield_x_pos
def nuts_x_yield : ℕ := nuts_x * yield_x
noncomputable def total_yield (yield_x_neg : ℕ) : ℕ :=
  nuts_x_pos_yield + nuts_x_yield + nuts_x_neg * yield_x_neg

-- Equation combining all
lemma yield_per_tree : (total_yield Y) / x_tot = avg_yield := sorry

-- Prove Y = 180
theorem calculate_yield : (x = 6 → ((nuts_x_neg * 180 = 540) ∧ rate = 180)) := sorry

end calculate_yield_l146_146112


namespace factorize_expression_l146_146194

variable (m n : ℤ)

theorem factorize_expression : 2 * m * n^2 - 12 * m * n + 18 * m = 2 * m * (n - 3)^2 := by
  sorry

end factorize_expression_l146_146194


namespace find_q_l146_146787

theorem find_q (q: ℕ) (h: 81^10 = 3^q) : q = 40 :=
by
  sorry

end find_q_l146_146787


namespace find_valid_pairs_l146_146092

-- Decalred the main definition for the problem.
def valid_pairs (x y : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99) ∧ ((x + y)^2 = 100 * x + y)

-- Stating the theorem without the proof.
theorem find_valid_pairs :
  valid_pairs 20 25 ∧ valid_pairs 30 25 :=
sorry

end find_valid_pairs_l146_146092


namespace shortest_chord_length_l146_146897

/-- The shortest chord passing through point D given the conditions provided. -/
theorem shortest_chord_length
  (O : Point) (D : Point) (r : ℝ) (OD : ℝ)
  (h_or : r = 5) (h_od : OD = 3) :
  ∃ (AB : ℝ), AB = 8 := 
  sorry

end shortest_chord_length_l146_146897


namespace number_of_solutions_in_positive_integers_l146_146800

theorem number_of_solutions_in_positive_integers (x y : ℕ) (h1 : 3 * x + 4 * y = 806) : 
  ∃ n : ℕ, n = 67 := 
sorry

end number_of_solutions_in_positive_integers_l146_146800


namespace simplify_expr_l146_146186

open Real

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : 
  sqrt (1 + ( (x^6 - 2) / (3 * x^3) )^2) = sqrt (x^12 + 5 * x^6 + 4) / (3 * x^3) :=
by
  sorry

end simplify_expr_l146_146186


namespace mean_profit_first_15_days_l146_146597

-- Definitions and conditions
def mean_daily_profit_entire_month : ℝ := 350
def total_days_in_month : ℕ := 30
def mean_daily_profit_last_15_days : ℝ := 445

-- Proof statement
theorem mean_profit_first_15_days : 
  (mean_daily_profit_entire_month * (total_days_in_month : ℝ) 
   - mean_daily_profit_last_15_days * 15) / 15 = 255 :=
by
  sorry

end mean_profit_first_15_days_l146_146597


namespace number_of_workers_is_25_l146_146131

noncomputable def original_workers (W : ℕ) :=
  W * 35 = (W + 10) * 25

theorem number_of_workers_is_25 : ∃ W, original_workers W ∧ W = 25 :=
by
  use 25
  unfold original_workers
  sorry

end number_of_workers_is_25_l146_146131


namespace range_of_m_l146_146360

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = -3) (h3 : x + y > 0) : m > 2 :=
by
  sorry

end range_of_m_l146_146360


namespace tan_ratio_l146_146440

open Real

theorem tan_ratio (x y : ℝ) (h1 : sin x / cos y + sin y / cos x = 2) (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 2 :=
sorry

end tan_ratio_l146_146440


namespace range_of_f_l146_146678

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_f 
  (x : ℝ) : f (x - 1) + f (x + 1) > 0 ↔ x ∈ Set.Ioi 0 :=
by
  sorry

end range_of_f_l146_146678


namespace maximum_daily_sales_l146_146991

def price (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then t + 20
else if (25 ≤ t ∧ t ≤ 30) then -t + 100
else 0

def sales_volume (t : ℕ) : ℝ :=
if (0 < t ∧ t ≤ 30) then -t + 40
else 0

def daily_sales (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then (t + 20) * (-t + 40)
else if (25 ≤ t ∧ t ≤ 30) then (-t + 100) * (-t + 40)
else 0

theorem maximum_daily_sales : ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_sales t = 1125 :=
sorry

end maximum_daily_sales_l146_146991


namespace inequality_proof_l146_146981

theorem inequality_proof (a b c d : ℝ) (h : a > 0) (h : b > 0) (h : c > 0) (h : d > 0)
  (h₁ : (a * b) / (c * d) = (a + b) / (c + d)) : (a + b) * (c + d) ≥ (a + c) * (b + d) :=
sorry

end inequality_proof_l146_146981


namespace triangle_cosine_l146_146718

theorem triangle_cosine {A : ℝ} (h : 0 < A ∧ A < π / 2) (tan_A : Real.tan A = -2) :
  Real.cos A = - (Real.sqrt 5) / 5 :=
sorry

end triangle_cosine_l146_146718


namespace total_students_in_classes_l146_146400

theorem total_students_in_classes (t1 t2 x y: ℕ) (h1 : t1 = 273) (h2 : t2 = 273) (h3 : (x - 1) * 7 = t1) (h4 : (y - 1) * 13 = t2) : x + y = 62 :=
by
  sorry

end total_students_in_classes_l146_146400


namespace cars_pass_same_order_l146_146163

theorem cars_pass_same_order (num_cars : ℕ) (num_points : ℕ)
    (cities_speeds speeds_outside_cities : Fin num_cars → ℝ) :
    num_cars = 10 → num_points = 2011 → 
    ∃ (p1 p2 : Fin num_points), p1 ≠ p2 ∧ (∀ i j : Fin num_cars, (i < j) → 
    (cities_speeds i) / (cities_speeds i + speeds_outside_cities i) = 
    (cities_speeds j) / (cities_speeds j + speeds_outside_cities j) → p1 = p2 ) :=
by
  sorry

end cars_pass_same_order_l146_146163


namespace sum_of_coefficients_l146_146850

theorem sum_of_coefficients (a : ℕ → ℤ) (x : ℂ) :
  (2*x - 1)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + 
  a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10 →
  a 0 = 1 →
  a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 20 :=
sorry

end sum_of_coefficients_l146_146850


namespace no_unique_solution_for_c_l146_146734

theorem no_unique_solution_for_c (k : ℕ) (hk : k = 9) (c : ℕ) :
  (∀ x y : ℕ, 9 * x + c * y = 30 → 3 * x + 4 * y = 12) → c = 12 :=
by
  sorry

end no_unique_solution_for_c_l146_146734


namespace all_faces_rhombuses_l146_146639

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end all_faces_rhombuses_l146_146639


namespace wilfred_carrots_total_l146_146434

-- Define the number of carrots Wilfred eats each day
def tuesday_carrots := 4
def wednesday_carrots := 6
def thursday_carrots := 5

-- Define the total number of carrots eaten from Tuesday to Thursday
def total_carrots := tuesday_carrots + wednesday_carrots + thursday_carrots

-- The theorem to prove that the total number of carrots is 15
theorem wilfred_carrots_total : total_carrots = 15 := by
  sorry

end wilfred_carrots_total_l146_146434


namespace no_snow_five_days_l146_146180

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l146_146180


namespace total_price_eq_2500_l146_146764

theorem total_price_eq_2500 (C P : ℕ)
  (hC : C = 2000)
  (hE : C + 500 + P = 6 * P)
  : C + P = 2500 := 
by
  sorry

end total_price_eq_2500_l146_146764


namespace find_an_from_sums_l146_146365

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end find_an_from_sums_l146_146365


namespace least_number_of_cans_l146_146513

theorem least_number_of_cans (maaza : ℕ) (pepsi : ℕ) (sprite : ℕ) (gcd_val : ℕ) (total_cans : ℕ)
  (h1 : maaza = 50) (h2 : pepsi = 144) (h3 : sprite = 368) (h_gcd : gcd maaza (gcd pepsi sprite) = gcd_val)
  (h_total_cans : total_cans = maaza / gcd_val + pepsi / gcd_val + sprite / gcd_val) :
  total_cans = 281 :=
sorry

end least_number_of_cans_l146_146513


namespace populations_equal_after_years_l146_146236

-- Defining the initial population and rates of change
def initial_population_X : ℕ := 76000
def rate_of_decrease_X : ℕ := 1200
def initial_population_Y : ℕ := 42000
def rate_of_increase_Y : ℕ := 800

-- Define the number of years for which we need to find the populations to be equal
def years (n : ℕ) : Prop :=
  (initial_population_X - rate_of_decrease_X * n) = (initial_population_Y + rate_of_increase_Y * n)

-- Theorem stating that the populations will be equal at n = 17
theorem populations_equal_after_years {n : ℕ} (h : n = 17) : years n :=
by
  sorry

end populations_equal_after_years_l146_146236


namespace arc_length_of_sector_l146_146476

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 3) :
  l = r * θ := by
  sorry

end arc_length_of_sector_l146_146476


namespace discount_threshold_l146_146468

-- Definitions based on given conditions
def photocopy_cost : ℝ := 0.02
def discount_percentage : ℝ := 0.25
def copies_needed_each : ℕ := 80
def total_savings : ℝ := 0.40 * 2 -- total savings for both Steve and Dennison

-- Minimum number of photocopies required to get the discount
def min_copies_for_discount : ℕ := 160

-- Lean statement to prove the minimum number of photocopies required for the discount
theorem discount_threshold :
  ∀ (x : ℕ),
  photocopy_cost * (x : ℝ) - (photocopy_cost * (1 - discount_percentage) * (x : ℝ)) * 2 = total_savings → 
  min_copies_for_discount = 160 :=
by sorry

end discount_threshold_l146_146468


namespace lattice_points_on_hyperbola_l146_146298

-- The hyperbola equation
def hyperbola_eq (x y : ℤ) : Prop :=
  x^2 - y^2 = 1800^2

-- The final number of lattice points lying on the hyperbola
theorem lattice_points_on_hyperbola : 
  ∃ (n : ℕ), n = 250 ∧ (∃ (x y : ℤ), hyperbola_eq x y) :=
sorry

end lattice_points_on_hyperbola_l146_146298


namespace probJackAndJillChosen_l146_146151

-- Define the probabilities of each worker being chosen
def probJack : ℝ := 0.20
def probJill : ℝ := 0.15

-- Define the probability that Jack and Jill are both chosen
def probJackAndJill : ℝ := probJack * probJill

-- Theorem stating the probability that Jack and Jill are both chosen
theorem probJackAndJillChosen : probJackAndJill = 0.03 := 
by
  -- Replace this sorry with the complete proof
  sorry

end probJackAndJillChosen_l146_146151


namespace set_aside_bars_each_day_l146_146086

-- Definitions for the conditions
def total_bars : Int := 20
def bars_traded : Int := 3
def bars_per_sister : Int := 5
def number_of_sisters : Int := 2
def days_in_week : Int := 7

-- Our goal is to prove that Greg set aside 1 bar per day
theorem set_aside_bars_each_day
  (h1 : 20 - 3 = 17)
  (h2 : 5 * 2 = 10)
  (h3 : 17 - 10 = 7)
  (h4 : 7 / 7 = 1) :
  (total_bars - bars_traded - (bars_per_sister * number_of_sisters)) / days_in_week = 1 := by
  sorry

end set_aside_bars_each_day_l146_146086


namespace arithmetic_sequence_formula_sum_Tn_formula_l146_146285

variable {a : ℕ → ℤ} -- The sequence a_n
variable {S : ℕ → ℤ} -- The sum S_n
variable {a₃ : ℤ} (h₁ : a₃ = 20)
variable {S₃ S₄ : ℤ} (h₂ : 2 * S₃ = S₄ + 8)

/- The general formula for the arithmetic sequence a_n -/
theorem arithmetic_sequence_formula (d : ℤ) (a₁ : ℤ)
  (h₃ : (a₃ = a₁ + 2 * d))
  (h₄ : (S₃ = 3 * a₁ + 3 * d))
  (h₅ : (S₄ = 4 * a₁ + 6 * d)) :
  ∀ n : ℕ, a n = 8 * n - 4 :=
by
  sorry

variable {b : ℕ → ℚ} -- Define b_n
variable {T : ℕ → ℚ} -- Define T_n
variable {S_general : ℕ → ℚ} (h₆ : ∀ n, S n = 4 * n ^ 2)
variable {b_general : ℚ → ℚ} (h₇ : ∀ n, b n = 1 / (S n - 1))
variable {T_general : ℕ → ℚ} -- Define T_n

/- The formula for T_n given b_n -/
theorem sum_Tn_formula :
  ∀ n : ℕ, T n = n / (2 * n + 1) :=
by
  sorry

end arithmetic_sequence_formula_sum_Tn_formula_l146_146285


namespace greatest_integer_solution_l146_146264

theorem greatest_integer_solution :
  ∃ n : ℤ, (n^2 - 17 * n + 72 ≤ 0) ∧ (∀ m : ℤ, (m^2 - 17 * m + 72 ≤ 0) → m ≤ n) ∧ n = 9 :=
sorry

end greatest_integer_solution_l146_146264


namespace sum_possible_values_l146_146958

theorem sum_possible_values (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := 
by
  sorry

end sum_possible_values_l146_146958


namespace sum_even_odd_probability_l146_146847

theorem sum_even_odd_probability :
  (∀ (a b : ℕ), ∃ (P_even P_odd : ℚ),
    P_even = 1/2 ∧ P_odd = 1/2 ∧
    (a % 2 = 0 ∧ b % 2 = 0 ↔ (a + b) % 2 = 0) ∧
    (a % 2 = 1 ∧ b % 2 = 1 ↔ (a + b) % 2 = 0) ∧
    ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0) ↔ (a + b) % 2 = 1)) :=
sorry

end sum_even_odd_probability_l146_146847


namespace compound_interest_second_year_l146_146681

theorem compound_interest_second_year
  (P : ℝ) (r : ℝ) (CI_3 : ℝ) (CI_2 : ℝ) 
  (h1 : r = 0.08) 
  (h2 : CI_3 = 1512)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1400 :=
by
  rw [h1, h2] at h3
  sorry

end compound_interest_second_year_l146_146681


namespace vectors_parallel_implies_fraction_l146_146558

theorem vectors_parallel_implies_fraction (α : ℝ) :
  let a := (Real.sin α, 3)
  let b := (Real.cos α, 1)
  (a.1 / b.1 = 3) → (Real.sin (2 * α) / (Real.cos α) ^ 2 = 6) :=
by
  sorry

end vectors_parallel_implies_fraction_l146_146558


namespace chess_tournament_games_l146_146307

-- Define the problem
def total_chess_games (n_players games_per_player : ℕ) : ℕ :=
  (n_players * games_per_player) / 2

-- Conditions: 
-- 1. There are 6 chess amateurs.
-- 2. Each amateur plays exactly 4 games.

theorem chess_tournament_games :
  total_chess_games 6 4 = 10 :=
  sorry

end chess_tournament_games_l146_146307


namespace increasing_exponential_is_necessary_condition_l146_146844

variable {a : ℝ}

theorem increasing_exponential_is_necessary_condition (h : ∀ x y : ℝ, x < y → a ^ x < a ^ y) :
    (a > 1) ∧ (¬ (a > 2 → a > 1)) :=
by
  sorry

end increasing_exponential_is_necessary_condition_l146_146844


namespace line_equation_l146_146369

-- Given a point and a direction vector
def point : ℝ × ℝ := (3, 4)
def direction_vector : ℝ × ℝ := (-2, 1)

-- Equation of the line passing through the given point with the given direction vector
theorem line_equation (x y : ℝ) : 
  (x = 3 ∧ y = 4) → ∃a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -11 ∧ a*x + b*y + c = 0 :=
by
  sorry

end line_equation_l146_146369


namespace isabel_ds_games_left_l146_146825

-- Define the initial number of DS games Isabel had
def initial_ds_games : ℕ := 90

-- Define the number of DS games Isabel gave to her friend
def ds_games_given : ℕ := 87

-- Define a function to calculate the remaining DS games
def remaining_ds_games (initial : ℕ) (given : ℕ) : ℕ := initial - given

-- Statement of the theorem we need to prove
theorem isabel_ds_games_left : remaining_ds_games initial_ds_games ds_games_given = 3 := by
  sorry

end isabel_ds_games_left_l146_146825


namespace hagrid_divisible_by_three_l146_146166

def distinct_digits (n : ℕ) : Prop :=
  n < 10

theorem hagrid_divisible_by_three (H A G R I D : ℕ) (H_dist A_dist G_dist R_dist I_dist D_dist : distinct_digits H ∧ distinct_digits A ∧ distinct_digits G ∧ distinct_digits R ∧ distinct_digits I ∧ distinct_digits D)
  (distinct_letters: H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ R ≠ I ∧ R ≠ D ∧ I ≠ D) :
  3 ∣ (H * 100000 + A * 10000 + G * 1000 + R * 100 + I * 10 + D) * H * A * G * R * I * D :=
sorry

end hagrid_divisible_by_three_l146_146166


namespace sum_lent_l146_146003

theorem sum_lent (P : ℝ) (R : ℝ := 4) (T : ℝ := 8) (I : ℝ) (H1 : I = P - 204) (H2 : I = (P * R * T) / 100) : 
  P = 300 :=
by 
  sorry

end sum_lent_l146_146003


namespace farm_own_more_horses_than_cows_after_transaction_l146_146070

theorem farm_own_more_horses_than_cows_after_transaction :
  ∀ (x : Nat), 
    3 * (3 * x - 15) = 5 * (x + 15) →
    75 - 45 = 30 :=
by
  intro x h
  -- This is a placeholder for the proof steps which we skip.
  sorry

end farm_own_more_horses_than_cows_after_transaction_l146_146070


namespace range_of_a_l146_146621

variables (a : ℝ) (x : ℝ) (x0 : ℝ)

def proposition_P (a : ℝ) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def proposition_Q (a : ℝ) : Prop :=
  ∃ x0, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_P a ∧ proposition_Q a) → a ∈ {a : ℝ | a ≤ -2} ∪ {a : ℝ | a = 1} :=
by {
  sorry -- Proof goes here.
}

end range_of_a_l146_146621


namespace determine_uv_l146_146464

theorem determine_uv :
  ∃ u v : ℝ, (u = 5 / 17) ∧ (v = -31 / 17) ∧
    ((⟨3, -2⟩ : ℝ × ℝ) + u • ⟨5, 8⟩ = (⟨-1, 4⟩ : ℝ × ℝ) + v • ⟨-3, 2⟩) :=
by
  sorry

end determine_uv_l146_146464


namespace carbonated_water_percentage_is_correct_l146_146309

-- Given percentages of lemonade and carbonated water in two solutions
def first_solution : Rat := 0.20 -- Lemonade percentage in the first solution
def second_solution : Rat := 0.45 -- Lemonade percentage in the second solution

-- Calculate percentages of carbonated water
def first_solution_carbonated_water := 1 - first_solution
def second_solution_carbonated_water := 1 - second_solution

-- Assume the mixture is 100 units, with equal parts from both solutions
def volume_mixture : Rat := 100
def volume_first_solution : Rat := volume_mixture * 0.50
def volume_second_solution : Rat := volume_mixture * 0.50

-- Calculate total carbonated water in the mixture
def carbonated_water_in_mixture :=
  (volume_first_solution * first_solution_carbonated_water) +
  (volume_second_solution * second_solution_carbonated_water)

-- Calculate the percentage of carbonated water in the mixture
def percentage_carbonated_water_in_mixture : Rat :=
  (carbonated_water_in_mixture / volume_mixture) * 100

-- Prove the percentage of carbonated water in the mixture is 67.5%
theorem carbonated_water_percentage_is_correct :
  percentage_carbonated_water_in_mixture = 67.5 := by
  sorry

end carbonated_water_percentage_is_correct_l146_146309


namespace correct_operation_l146_146326

variable (N : ℚ) -- Original number (assumed rational for simplicity)
variable (x : ℚ) -- Unknown multiplier

theorem correct_operation (h : (N / 10) = (5 / 100) * (N * x)) : x = 2 :=
by
  sorry

end correct_operation_l146_146326


namespace chromosomal_variations_l146_146998

-- Define the conditions
def condition1 := "Plants grown from anther culture in vitro."
def condition2 := "Addition or deletion of DNA base pairs on chromosomes."
def condition3 := "Free combination of non-homologous chromosomes."
def condition4 := "Crossing over between non-sister chromatids in a tetrad."
def condition5 := "Cells of a patient with Down syndrome have three copies of chromosome 21."

-- Define a concept of belonging to chromosomal variations
def belongs_to_chromosomal_variations (condition: String) : Prop :=
  condition = condition1 ∨ condition = condition5

-- State the theorem
theorem chromosomal_variations :
  belongs_to_chromosomal_variations condition1 ∧ 
  belongs_to_chromosomal_variations condition5 ∧ 
  ¬ (belongs_to_chromosomal_variations condition2 ∨ 
     belongs_to_chromosomal_variations condition3 ∨ 
     belongs_to_chromosomal_variations condition4) :=
by
  sorry

end chromosomal_variations_l146_146998


namespace problem_statement_l146_146590

-- Given conditions
variable (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k)

-- Hypothesis configuration for inductive proof and goal statement
theorem problem_statement : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end problem_statement_l146_146590


namespace quadratic_real_roots_l146_146423

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ∧ (∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) → k ≥ -1 :=
by
  sorry

end quadratic_real_roots_l146_146423


namespace parallel_line_eq_perpendicular_line_eq_l146_146343

-- Define the conditions: A line passing through (1, -4) and the given line equation 2x + 3y + 5 = 0
def passes_through (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Define the theorem statements for parallel and perpendicular lines
theorem parallel_line_eq (m : ℝ) :
  passes_through 1 (-4) 2 3 m → m = 10 := 
sorry

theorem perpendicular_line_eq (n : ℝ) :
  passes_through 1 (-4) 3 (-2) (-n) → n = 11 :=
sorry

end parallel_line_eq_perpendicular_line_eq_l146_146343


namespace number_of_truthful_people_l146_146819

-- Definitions from conditions
def people := Fin 100
def tells_truth (p : people) : Prop := sorry -- Placeholder definition.

-- Conditions
axiom c1 : ∃ p : people, ¬ tells_truth p
axiom c2 : ∀ p1 p2 : people, p1 ≠ p2 → (tells_truth p1 ∨ tells_truth p2)

-- Goal
theorem number_of_truthful_people : 
  ∃ S : Finset people, S.card = 99 ∧ (∀ p ∈ S, tells_truth p) :=
sorry

end number_of_truthful_people_l146_146819


namespace consecutive_integers_sum_l146_146537

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l146_146537


namespace imaginary_part_z_l146_146580

theorem imaginary_part_z : 
  ∀ (z : ℂ), z = (5 - I) / (1 - I) → z.im = 2 := 
by
  sorry

end imaginary_part_z_l146_146580


namespace min_value_x_plus_4y_l146_146457

theorem min_value_x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_cond : (1 / x) + (1 / (2 * y)) = 1) : x + 4 * y = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_x_plus_4y_l146_146457


namespace smallest_n_for_n_cubed_ends_in_888_l146_146581

/-- Proof Problem: Prove that 192 is the smallest positive integer \( n \) such that the last three digits of \( n^3 \) are 888. -/
theorem smallest_n_for_n_cubed_ends_in_888 : ∃ n : ℕ, n > 0 ∧ (n^3 % 1000 = 888) ∧ ∀ m : ℕ, 0 < m ∧ (m^3 % 1000 = 888) → n ≤ m :=
by
  sorry

end smallest_n_for_n_cubed_ends_in_888_l146_146581


namespace cos_alpha_plus_5pi_over_4_eq_16_over_65_l146_146402

theorem cos_alpha_plus_5pi_over_4_eq_16_over_65
  (α β : ℝ)
  (hα : -π / 4 < α ∧ α < 0)
  (hβ : π / 2 < β ∧ β < π)
  (hcos_sum : Real.cos (α + β) = -4/5)
  (hcos_diff : Real.cos (β - π / 4) = 5/13) :
  Real.cos (α + 5 * π / 4) = 16/65 :=
by
  sorry

end cos_alpha_plus_5pi_over_4_eq_16_over_65_l146_146402


namespace probability_of_one_radio_operator_per_group_l146_146539

def total_ways_to_assign_soldiers_to_groups : ℕ := 27720
def ways_to_assign_radio_operators_to_groups : ℕ := 7560

theorem probability_of_one_radio_operator_per_group :
  (ways_to_assign_radio_operators_to_groups : ℚ) / (total_ways_to_assign_soldiers_to_groups : ℚ) = 3 / 11 := 
sorry

end probability_of_one_radio_operator_per_group_l146_146539


namespace relationship_among_g_a_0_f_b_l146_146014

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem relationship_among_g_a_0_f_b (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  -- Function properties are non-trivial and are omitted.
  sorry

end relationship_among_g_a_0_f_b_l146_146014


namespace arithmetic_sequence_sum_l146_146890

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_sum 
  {a1 d : ℕ} (h_pos_d : d > 0) 
  (h_sum : a1 + (a1 + d) + (a1 + 2 * d) = 15) 
  (h_prod : a1 * (a1 + d) * (a1 + 2 * d) = 80) 
  : a_n a1 d 11 + a_n a1 d 12 + a_n a1 d 13 = 105 :=
sorry

end arithmetic_sequence_sum_l146_146890


namespace vector_identity_l146_146946

namespace VectorAddition

variable {V : Type*} [AddCommGroup V]

theorem vector_identity
  (AD DC AB BC : V)
  (h1 : AD + DC = AC)
  (h2 : AC - AB = BC) :
  AD + DC - AB = BC :=
by
  sorry

end VectorAddition

end vector_identity_l146_146946


namespace james_carrot_sticks_l146_146498

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end james_carrot_sticks_l146_146498


namespace curve_intersects_at_point_2_3_l146_146838

open Real

theorem curve_intersects_at_point_2_3 :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
                 (t₁^2 - 4 = t₂^2 - 4) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = t₂^3 - 6 * t₂ + 3) ∧ 
                 (t₁^2 - 4 = 2) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = 3) :=
by
  sorry

end curve_intersects_at_point_2_3_l146_146838


namespace arithmetic_sequence_value_l146_146873

theorem arithmetic_sequence_value 
    (a1 : ℤ) (a2 a3 a4 : ℤ) (a1_a4 : a1 = 18) 
    (b1 b2 b3 : ℤ) 
    (b1_b3 : b3 - b2 = 6 ∧ b2 - b1 = 6 ∧ b2 = 15 ∧ b3 = 21)
    (b1_a3 : a3 = b1 - 6 ∧ a4 = a1 + (a3 - 18) / 3) 
    (c1 c2 c3 c4 : ℝ) 
    (c1_b3 : c1 = a4) 
    (c2 : c2 = -14) 
    (c4 : ∃ m, c4 = b1 - m * (6 :ℝ) + - 0.5) 
    (n : ℝ) : 
    n = -12.5 := by 
  sorry

end arithmetic_sequence_value_l146_146873


namespace solve_for_x_l146_146122

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l146_146122


namespace range_of_a_l146_146275

theorem range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (x-a) / (2 - (x + 1 - a)) > 0)
  ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l146_146275


namespace gibbs_inequality_l146_146578

noncomputable section

open BigOperators

variable {r : ℕ} (p q : Fin r → ℝ)

/-- (p_i) is a probability distribution -/
def isProbabilityDistribution (p : Fin r → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ (∑ i, p i = 1)

/-- -\sum_{i=1}^{r} p_i \ln p_i \leqslant -\sum_{i=1}^{r} p_i \ln q_i for probability distributions p and q -/
theorem gibbs_inequality
  (hp : isProbabilityDistribution p)
  (hq : isProbabilityDistribution q) :
  -∑ i, p i * Real.log (p i) ≤ -∑ i, p i * Real.log (q i) := 
by
  sorry

end gibbs_inequality_l146_146578


namespace total_seats_l146_146155

theorem total_seats (F : ℕ) 
  (h1 : 305 = 4 * F + 2) 
  (h2 : 310 = 4 * F + 2) : 
  310 + F = 387 :=
by
  sorry

end total_seats_l146_146155


namespace existence_of_x2_with_sum_ge_2_l146_146012

variables (a b c x1 x2 : ℝ) (h_root1 : a * x1^2 + b * x1 + c = 0) (h_x1_pos : x1 > 0)

theorem existence_of_x2_with_sum_ge_2 :
  ∃ x2, (c * x2^2 + b * x2 + a = 0) ∧ (x1 + x2 ≥ 2) :=
sorry

end existence_of_x2_with_sum_ge_2_l146_146012


namespace simplify_expression_1_simplify_expression_2_l146_146664

-- Problem 1
theorem simplify_expression_1 (a b : ℤ) : a + 2 * b + 3 * a - 2 * b = 4 * a :=
by
  sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℤ) (h_m : m = 2) (h_n : n = 1) :
  (2 * m ^ 2 - 3 * m * n + 8) - (5 * m * n - 4 * m ^ 2 + 8) = 8 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l146_146664


namespace problem_1_problem_2_l146_146999

theorem problem_1 :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
by
  sorry

theorem problem_2 :
  (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) = (3^32 - 1) / 2 :=
by
  sorry

end problem_1_problem_2_l146_146999


namespace three_digit_numbers_with_distinct_digits_avg_condition_l146_146879

theorem three_digit_numbers_with_distinct_digits_avg_condition : 
  ∃ (S : Finset (Fin 1000)), 
  (∀ n ∈ S, (n / 100 ≠ (n / 10 % 10) ∧ (n / 100 ≠ n % 10) ∧ (n / 10 % 10 ≠ n % 10))) ∧
  (∀ n ∈ S, ((n / 100 + n % 10) / 2 = n / 10 % 10)) ∧
  (∀ n ∈ S, abs ((n / 100) - (n / 10 % 10)) ≤ 5 ∧ abs ((n / 10 % 10) - (n % 10)) ≤ 5) ∧
  S.card = 120 :=
sorry

end three_digit_numbers_with_distinct_digits_avg_condition_l146_146879


namespace pyramid_total_surface_area_l146_146780

theorem pyramid_total_surface_area :
  ∀ (s h : ℝ), s = 8 → h = 10 →
  6 * (1/2 * s * (Real.sqrt (h^2 - (s/2)^2))) = 48 * Real.sqrt 21 :=
by
  intros s h s_eq h_eq
  rw [s_eq, h_eq]
  sorry

end pyramid_total_surface_area_l146_146780


namespace gecko_cricket_eating_l146_146776

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end gecko_cricket_eating_l146_146776


namespace triangle_area_is_32_5_l146_146431

-- Define points A, B, and C
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (1, 7)
def C : ℝ × ℝ := (4, -1)

-- Calculate the area directly using the determinant method for the area of a triangle given by coordinates
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (
    A.1 * (B.2 - C.2) +
    B.1 * (C.2 - A.2) +
    C.1 * (A.2 - B.2)
  )

-- Define the statement to be proved
theorem triangle_area_is_32_5 : area_triangle A B C = 32.5 := 
  by
  -- proof to be filled in
  sorry

end triangle_area_is_32_5_l146_146431


namespace cards_problem_l146_146688

theorem cards_problem : 
  ∀ (cards people : ℕ),
  cards = 60 →
  people = 8 →
  ∃ fewer_people : ℕ,
  (∀ p: ℕ, p < people → (p < fewer_people → cards/people < 8)) ∧ 
  fewer_people = 4 := 
by 
  intros cards people h_cards h_people
  use 4
  sorry

end cards_problem_l146_146688


namespace hexagon_inequality_l146_146207

variables {Point : Type} [MetricSpace Point]

-- Definitions of points and distances
variables (A B C D E F G H : Point) 
variables (dist : Point → Point → ℝ)
variables (angle : Point → Point → Point → ℝ)

-- Conditions
variables (hABCDEF : ConvexHexagon A B C D E F)
variables (hAB_BC_CD : dist A B = dist B C ∧ dist B C = dist C D)
variables (hDE_EF_FA : dist D E = dist E F ∧ dist E F = dist F A)
variables (hBCD_60 : angle B C D = 60)
variables (hEFA_60 : angle E F A = 60)
variables (hAGB_120 : angle A G B = 120)
variables (hDHE_120 : angle D H E = 120)

-- Objective statement
theorem hexagon_inequality : 
  dist A G + dist G B + dist G H + dist D H + dist H E ≥ dist C F :=
sorry

end hexagon_inequality_l146_146207


namespace abs_eq_of_sq_eq_l146_146331

theorem abs_eq_of_sq_eq (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  intro h
  sorry

end abs_eq_of_sq_eq_l146_146331


namespace mindy_tax_rate_l146_146638

variables (M : ℝ) -- Mork's income
variables (r : ℝ) -- Mindy's tax rate

-- Conditions
def Mork_tax_rate := 0.45 -- 45% tax rate
def Mindx_income := 4 * M -- Mindy earned 4 times as much as Mork
def combined_tax_rate := 0.21 -- Combined tax rate is 21%

-- Equation derived from the conditions
def combined_tax_rate_eq := (0.45 * M + 4 * M * r) / (M + 4 * M) = 0.21

theorem mindy_tax_rate : combined_tax_rate_eq M r → r = 0.15 :=
by
  intros conditional_eq
  sorry

end mindy_tax_rate_l146_146638


namespace interest_years_calculation_l146_146470

theorem interest_years_calculation 
  (total_sum : ℝ)
  (second_sum : ℝ)
  (interest_rate_first : ℝ)
  (interest_rate_second : ℝ)
  (time_second : ℝ)
  (interest_second : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : total_sum = 2795)
  (h2 : second_sum = 1720)
  (h3 : interest_rate_first = 3)
  (h4 : interest_rate_second = 5)
  (h5 : time_second = 3)
  (h6 : interest_second = (second_sum * interest_rate_second * time_second) / 100)
  (h7 : interest_second = 258)
  (h8 : x = (total_sum - second_sum))
  (h9 : (interest_rate_first * x * y) / 100 = interest_second)
  : y = 8 := sorry

end interest_years_calculation_l146_146470


namespace rectangle_area_l146_146774

theorem rectangle_area
  (s : ℝ)
  (h_square_area : s^2 = 49)
  (rect_width : ℝ := s)
  (rect_length : ℝ := 3 * rect_width)
  (h_rect_width_eq_s : rect_width = s)
  (h_rect_length_eq_3w : rect_length = 3 * rect_width) :
  rect_width * rect_length = 147 :=
by 
  skip
  sorry

end rectangle_area_l146_146774


namespace difference_between_numbers_l146_146532

theorem difference_between_numbers 
  (A B : ℝ)
  (h1 : 0.075 * A = 0.125 * B)
  (h2 : A = 2430 ∨ B = 2430) :
  A - B = 972 :=
by
  sorry

end difference_between_numbers_l146_146532


namespace remainder_n_plus_1008_l146_146048

variable (n : ℕ)

theorem remainder_n_plus_1008 (h1 : n % 4 = 1) (h2 : n % 5 = 3) : (n + 1008) % 4 = 1 := by
  sorry

end remainder_n_plus_1008_l146_146048


namespace rate_of_discount_l146_146135

theorem rate_of_discount (marked_price selling_price : ℝ) (h1 : marked_price = 200) (h2 : selling_price = 120) : 
  ((marked_price - selling_price) / marked_price) * 100 = 40 :=
by
  sorry

end rate_of_discount_l146_146135


namespace fran_speed_l146_146932

-- Definitions for conditions
def joann_speed : ℝ := 15 -- in miles per hour
def joann_time : ℝ := 4 -- in hours
def fran_time : ℝ := 2 -- in hours
def joann_distance : ℝ := joann_speed * joann_time -- distance Joann traveled

-- Proof Goal Statement
theorem fran_speed (hf: fran_time ≠ 0) : (joann_speed * joann_time) / fran_time = 30 :=
by
  -- Sorry placeholder skips the proof steps
  sorry

end fran_speed_l146_146932


namespace smallest_nat_number_l146_146378

theorem smallest_nat_number : ∃ a : ℕ, (a % 3 = 2) ∧ (a % 5 = 4) ∧ (a % 7 = 4) ∧ (∀ b : ℕ, (b % 3 = 2) ∧ (b % 5 = 4) ∧ (b % 7 = 4) → a ≤ b) ∧ a = 74 := 
sorry

end smallest_nat_number_l146_146378


namespace find_AX_l146_146584

theorem find_AX
  (AB AC BC : ℚ)
  (H : AB = 80)
  (H1 : AC = 50)
  (H2 : BC = 30)
  (angle_bisector_theorem_1 : ∀ (AX XC y : ℚ), AX = 8 * y ∧ XC = 3 * y ∧ 11 * y = AC → y = 50 / 11)
  (angle_bisector_theorem_2 : ∀ (BD DC z : ℚ), BD = 8 * z ∧ DC = 5 * z ∧ 13 * z = BC → z = 30 / 13) :
  AX = 400 / 11 := 
sorry

end find_AX_l146_146584


namespace digit_150th_of_17_div_70_is_7_l146_146927

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l146_146927


namespace rectangular_field_area_l146_146658

theorem rectangular_field_area (w l A : ℝ) 
  (h1 : l = 3 * w)
  (h2 : 2 * (w + l) = 80) :
  A = w * l → A = 300 :=
by
  sorry

end rectangular_field_area_l146_146658


namespace no_solution_for_equation_l146_146521

theorem no_solution_for_equation :
  ¬ (∃ x : ℝ, 
    4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 5 * (12 - (4 * (x + 1) - 3 * x)) = 
    18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11)))) :=
by
  sorry

end no_solution_for_equation_l146_146521


namespace find_a_b_find_k_range_l146_146785

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l146_146785


namespace loss_percentage_l146_146101

variable (CP SP : ℕ) -- declare the variables for cost price and selling price

theorem loss_percentage (hCP : CP = 1400) (hSP : SP = 1190) : 
  ((CP - SP) / CP * 100) = 15 := by
sorry

end loss_percentage_l146_146101


namespace probability_chord_length_not_less_than_radius_l146_146706

theorem probability_chord_length_not_less_than_radius
  (R : ℝ) (M N : ℝ) (h_circle : N = 2 * π * R) : 
  (∃ P : ℝ, P = 2 / 3) :=
sorry

end probability_chord_length_not_less_than_radius_l146_146706


namespace minimum_value_l146_146545

open Real

theorem minimum_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : 2 * x + y = 2) :
    ∃ x y, (0 < x) ∧ (0 < y) ∧ (2 * x + y = 2) ∧ (x + sqrt (x^2 + y^2) = 8 / 5) :=
sorry

end minimum_value_l146_146545


namespace prob_of_nine_correct_is_zero_l146_146806

-- Define the necessary components and properties of the problem
def is_correct_placement (letter: ℕ) (envelope: ℕ) : Prop := letter = envelope

def is_random_distribution (letters : Fin 10 → Fin 10) : Prop := true

-- State the theorem formally
theorem prob_of_nine_correct_is_zero (f : Fin 10 → Fin 10) :
  is_random_distribution f →
  (∃ (count : ℕ), count = 9 ∧ (∀ i : Fin 10, is_correct_placement i (f i) ↔ i = count)) → false :=
by
  sorry

end prob_of_nine_correct_is_zero_l146_146806


namespace work_completion_in_days_l146_146410

noncomputable def work_days_needed : ℕ :=
  let A_rate := 1 / 9
  let B_rate := 1 / 18
  let C_rate := 1 / 12
  let D_rate := 1 / 24
  let AB_rate := A_rate + B_rate
  let CD_rate := C_rate + D_rate
  let two_day_work := AB_rate + CD_rate
  let total_cycles := 24 / 7
  let total_days := (if total_cycles % 1 = 0 then total_cycles else total_cycles + 1) * 2
  total_days

theorem work_completion_in_days :
  work_days_needed = 8 :=
by
  sorry

end work_completion_in_days_l146_146410


namespace accommodate_students_l146_146251

-- Define the parameters
def number_of_classrooms := 15
def one_third_classrooms := number_of_classrooms / 3
def desks_per_classroom_30 := 30
def desks_per_classroom_25 := 25

-- Define the number of classrooms for each type
def classrooms_with_30_desks := one_third_classrooms
def classrooms_with_25_desks := number_of_classrooms - classrooms_with_30_desks

-- Calculate total number of students that can be accommodated
def total_students : ℕ := 
  (classrooms_with_30_desks * desks_per_classroom_30) +
  (classrooms_with_25_desks * desks_per_classroom_25)

-- Prove that total number of students that the school can accommodate is 400
theorem accommodate_students : total_students = 400 := sorry

end accommodate_students_l146_146251


namespace rectangle_area_is_1600_l146_146321

theorem rectangle_area_is_1600 (l w : ℕ) 
  (h₁ : l = 4 * w)
  (h₂ : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_is_1600_l146_146321


namespace shelves_used_l146_146208

def coloring_books := 87
def sold_books := 33
def books_per_shelf := 6

theorem shelves_used (h1: coloring_books - sold_books = 54) : 54 / books_per_shelf = 9 :=
by
  sorry

end shelves_used_l146_146208


namespace fraction_problem_l146_146739

noncomputable def zero_point_one_five : ℚ := 5 / 33
noncomputable def two_point_four_zero_three : ℚ := 2401 / 999

theorem fraction_problem :
  (zero_point_one_five / two_point_four_zero_three) = (4995 / 79233) :=
by
  sorry

end fraction_problem_l146_146739


namespace cos_pi_over_2_minus_l146_146015

theorem cos_pi_over_2_minus (A : ℝ) (h : Real.sin A = 1 / 2) : Real.cos (3 * Real.pi / 2 - A) = -1 / 2 :=
  sorry

end cos_pi_over_2_minus_l146_146015


namespace cave_depth_l146_146818

theorem cave_depth 
  (total_depth : ℕ) 
  (remaining_depth : ℕ) 
  (h1 : total_depth = 974) 
  (h2 : remaining_depth = 386) : 
  total_depth - remaining_depth = 588 := 
by 
  sorry

end cave_depth_l146_146818


namespace count_squares_below_graph_l146_146947

theorem count_squares_below_graph (x y: ℕ) (h : 5 * x + 195 * y = 975) :
  ∃ n : ℕ, n = 388 ∧ 
  ∀ a b : ℕ, 0 ≤ a ∧ a ≤ 195 ∧ 0 ≤ b ∧ b ≤ 5 →
    1 * a + 1 * b < 195 * 5 →
    n = 388 := 
sorry

end count_squares_below_graph_l146_146947


namespace black_and_blue_lines_l146_146633

-- Definition of given conditions
def grid_size : ℕ := 50
def total_points : ℕ := grid_size * grid_size
def blue_points : ℕ := 1510
def blue_edge_points : ℕ := 110
def red_segments : ℕ := 947
def corner_points : ℕ := 4

-- Calculations based on conditions
def red_points : ℕ := total_points - blue_points

def edge_points (size : ℕ) : ℕ := (size - 1) * 4
def non_corner_edge_points (edge : ℕ) : ℕ := edge - corner_points

-- Math translation
noncomputable def internal_red_points : ℕ := red_points - corner_points - (edge_points grid_size - blue_edge_points)
noncomputable def connections_from_red_points : ℕ :=
  corner_points * 2 + (non_corner_edge_points (edge_points grid_size) - blue_edge_points) * 3 + internal_red_points * 4

noncomputable def adjusted_red_lines : ℕ := red_segments * 2
noncomputable def black_lines : ℕ := connections_from_red_points - adjusted_red_lines

def total_lines (size : ℕ) : ℕ := (size - 1) * size + (size - 1) * size
noncomputable def blue_lines : ℕ := total_lines grid_size - red_segments - black_lines

-- The theorem to be proven
theorem black_and_blue_lines :
  (black_lines = 1972) ∧ (blue_lines = 1981) :=
by
  sorry

end black_and_blue_lines_l146_146633


namespace petri_dishes_count_l146_146805

def germs_total : ℕ := 5400000
def germs_per_dish : ℕ := 500
def petri_dishes : ℕ := germs_total / germs_per_dish

theorem petri_dishes_count : petri_dishes = 10800 := by
  sorry

end petri_dishes_count_l146_146805


namespace a_can_be_any_sign_l146_146973

theorem a_can_be_any_sign (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b)^2 < (c / d)^2) (hcd : c = -d) : True :=
by
  have := h
  subst hcd
  sorry

end a_can_be_any_sign_l146_146973


namespace sin_cos_ratio_l146_146647

open Real

theorem sin_cos_ratio
  (θ : ℝ)
  (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) :
  sin θ * cos θ = 3 / 10 := 
by
  sorry

end sin_cos_ratio_l146_146647


namespace parabola_focus_distance_l146_146239

theorem parabola_focus_distance (p m : ℝ) (hp : p > 0)
  (P_on_parabola : m^2 = 2 * p)
  (PF_dist : (1 + p / 2) = 3) : p = 4 := 
  sorry

end parabola_focus_distance_l146_146239


namespace time_difference_l146_146528

theorem time_difference (dist1 dist2 : ℕ) (speed : ℕ) (h_dist : dist1 = 600) (h_dist2 : dist2 = 550) (h_speed : speed = 40) :
  (dist1 - dist2) / speed * 60 = 75 := by
  sorry

end time_difference_l146_146528


namespace unique_angles_sum_l146_146566

theorem unique_angles_sum (a1 a2 a3 a4 e4 e5 e6 e7 : ℝ) 
  (h_abcd: a1 + a2 + a3 + a4 = 360) 
  (h_efgh: e4 + e5 + e6 + e7 = 360) 
  (h_shared: a4 = e4) : 
  a1 + a2 + a3 + e4 + e5 + e6 + e7 - a4 = 360 := 
by 
  sorry

end unique_angles_sum_l146_146566


namespace daily_expenses_increase_l146_146334

theorem daily_expenses_increase 
  (init_students : ℕ) (new_students : ℕ) (diminish_amount : ℝ) (orig_expenditure : ℝ)
  (orig_expenditure_eq : init_students = 35)
  (new_students_eq : new_students = 42)
  (diminish_amount_eq : diminish_amount = 1)
  (orig_expenditure_val : orig_expenditure = 400)
  (orig_average_expenditure : ℝ) (increase_expenditure : ℝ)
  (orig_avg_calc : orig_average_expenditure = orig_expenditure / init_students)
  (new_total_expenditure : ℝ)
  (new_expenditure_eq : new_total_expenditure = orig_expenditure + increase_expenditure) :
  (42 * (orig_average_expenditure - diminish_amount) = new_total_expenditure) → increase_expenditure = 38 := 
by 
  sorry

end daily_expenses_increase_l146_146334


namespace frac_nonneg_iff_pos_l146_146149

theorem frac_nonneg_iff_pos (x : ℝ) : (2 / x ≥ 0) ↔ (x > 0) :=
by sorry

end frac_nonneg_iff_pos_l146_146149


namespace number_of_sides_sum_of_interior_angles_l146_146246

-- Condition: each exterior angle of the regular polygon is 18 degrees.
def exterior_angle (n : ℕ) : Prop :=
  360 / n = 18

-- Question 1: Determine the number of sides the polygon has.
theorem number_of_sides : ∃ n, n > 2 ∧ exterior_angle n :=
  sorry

-- Question 2: Calculate the sum of the interior angles.
theorem sum_of_interior_angles {n : ℕ} (h : 360 / n = 18) : 
  180 * (n - 2) = 3240 :=
  sorry

end number_of_sides_sum_of_interior_angles_l146_146246


namespace smallest_positive_debt_resolvable_l146_146473

theorem smallest_positive_debt_resolvable :
  ∃ p g : ℤ, 280 * p + 200 * g = 40 ∧
  ∀ k : ℤ, k > 0 → (∃ p g : ℤ, 280 * p + 200 * g = k) → 40 ≤ k :=
by
  sorry

end smallest_positive_debt_resolvable_l146_146473


namespace multiplication_addition_l146_146519

theorem multiplication_addition :
  108 * 108 + 92 * 92 = 20128 :=
by
  sorry

end multiplication_addition_l146_146519


namespace square_side_length_exists_l146_146011

theorem square_side_length_exists
    (k : ℕ)
    (n : ℕ)
    (h_side_length_condition : n * n = k * (k - 7))
    (h_grid_lines : k > 7) :
    n = 12 ∨ n = 24 :=
by sorry

end square_side_length_exists_l146_146011


namespace pow_div_mul_pow_eq_l146_146373

theorem pow_div_mul_pow_eq (a b c d : ℕ) (h_a : a = 8) (h_b : b = 5) (h_c : c = 2) (h_d : d = 6) :
  (a^b / a^c) * (4^6) = 2^21 := by
  sorry

end pow_div_mul_pow_eq_l146_146373


namespace complement_union_correct_l146_146793

-- Defining the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_correct : (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l146_146793


namespace find_a_l146_146520

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 2 then x^2 - 4 else |x - 3| + a

theorem find_a (a : ℝ) (h : f (f (Real.sqrt 6) a) a = 3) : a = 2 := by
  sorry

end find_a_l146_146520


namespace ratio_of_times_l146_146860

theorem ratio_of_times (A_work_time B_combined_rate : ℕ) 
  (h1 : A_work_time = 6) 
  (h2 : (1 / (1 / A_work_time + 1 / (B_combined_rate / 2))) = 2) :
  (B_combined_rate : ℝ) / A_work_time = 1 / 2 :=
by
  -- below we add the proof part which we will skip for now with sorry.
  sorry

end ratio_of_times_l146_146860


namespace integer_solutions_of_equation_l146_146799

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) := by 
  sorry

end integer_solutions_of_equation_l146_146799


namespace intersection_is_correct_l146_146489

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_is_correct : A ∩ B = {0, 3} := by
  sorry

end intersection_is_correct_l146_146489


namespace value_added_to_075_of_number_l146_146522

theorem value_added_to_075_of_number (N V : ℝ) (h1 : 0.75 * N + V = 8) (h2 : N = 8) : V = 2 := by
  sorry

end value_added_to_075_of_number_l146_146522


namespace hyperbola_condition_l146_146809

-- Definitions and hypotheses
def is_hyperbola (m n : ℝ) (x y : ℝ) : Prop := m * x^2 - n * y^2 = 1

-- Statement of the problem
theorem hyperbola_condition (m n : ℝ) : (∃ x y : ℝ, is_hyperbola m n x y) ↔ m * n > 0 :=
by sorry

end hyperbola_condition_l146_146809


namespace find_number_l146_146249

theorem find_number (x : ℕ) : x * 9999 = 4691130840 → x = 469200 :=
by
  intros h
  sorry

end find_number_l146_146249


namespace gas_station_total_boxes_l146_146616

theorem gas_station_total_boxes
  (chocolate_boxes : ℕ)
  (sugar_boxes : ℕ)
  (gum_boxes : ℕ)
  (licorice_boxes : ℕ)
  (sour_boxes : ℕ)
  (h_chocolate : chocolate_boxes = 3)
  (h_sugar : sugar_boxes = 5)
  (h_gum : gum_boxes = 2)
  (h_licorice : licorice_boxes = 4)
  (h_sour : sour_boxes = 7) :
  chocolate_boxes + sugar_boxes + gum_boxes + licorice_boxes + sour_boxes = 21 := by
  sorry

end gas_station_total_boxes_l146_146616


namespace sample_size_is_80_l146_146303

-- Define the given conditions
variables (x : ℕ) (numA numB numC n : ℕ)

-- Conditions in Lean
def ratio_condition (x numA numB numC : ℕ) : Prop :=
  numA = 2 * x ∧ numB = 3 * x ∧ numC = 5 * x

def sample_condition (numA : ℕ) : Prop :=
  numA = 16

-- Definition of the proof problem
theorem sample_size_is_80 (x : ℕ) (numA numB numC n : ℕ)
  (h_ratio : ratio_condition x numA numB numC)
  (h_sample : sample_condition numA) : 
  n = 80 :=
by
-- The proof is omitted, just state the theorem
sorry

end sample_size_is_80_l146_146303


namespace cost_of_burger_l146_146428

theorem cost_of_burger :
  ∃ (b s f : ℕ), 
    4 * b + 3 * s + f = 540 ∧
    3 * b + 2 * s + 2 * f = 580 ∧
    b = 100 :=
by {
  sorry
}

end cost_of_burger_l146_146428


namespace amount_pop_spend_l146_146823

theorem amount_pop_spend
  (total_spent : ℝ)
  (ratio_snap_crackle : ℝ)
  (ratio_crackle_pop : ℝ)
  (spending_eq : total_spent = 150)
  (snap_crackle : ratio_snap_crackle = 2)
  (crackle_pop : ratio_crackle_pop = 3)
  (snap : ℝ)
  (crackle : ℝ)
  (pop : ℝ)
  (snap_eq : snap = ratio_snap_crackle * crackle)
  (crackle_eq : crackle = ratio_crackle_pop * pop)
  (total_eq : snap + crackle + pop = total_spent) :
  pop = 15 := 
by
  sorry

end amount_pop_spend_l146_146823


namespace a4_value_l146_146849

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition: The sum of the first n terms of the sequence {a_n} is S_n = n^2 - 1
axiom sum_of_sequence (n : ℕ) : S n = n^2 - 1

-- We need to prove that a_4 = 7
theorem a4_value : a 4 = S 4 - S 3 :=
by 
  -- Proof goes here
  sorry

end a4_value_l146_146849


namespace tom_age_ratio_l146_146867

-- Definitions of the variables
variables (T : ℕ) (N : ℕ)

-- Conditions given in the problem
def condition1 : Prop := T = 2 * (T / 2)
def condition2 : Prop := (T - 3) = 3 * (T / 2 - 12)

-- The ratio theorem to prove
theorem tom_age_ratio (h1 : condition1 T) (h2 : condition2 T) : T / N = 22 :=
by
  sorry

end tom_age_ratio_l146_146867


namespace waitress_tips_fraction_l146_146803

theorem waitress_tips_fraction
  (S : ℝ) -- salary
  (T : ℝ) -- tips
  (hT : T = (11 / 4) * S) -- tips are 11/4 of salary
  (I : ℝ) -- total income
  (hI : I = S + T) -- total income is the sum of salary and tips
  : (T / I) = (11 / 15) := -- fraction of income from tips is 11/15
by
  sorry

end waitress_tips_fraction_l146_146803


namespace solution_set_of_equation_l146_146429

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solution_set_of_equation (x : ℝ) (h : x > 0): (x^(log_base 10 x) = x^3 / 100) ↔ (x = 10 ∨ x = 100) := 
by sorry

end solution_set_of_equation_l146_146429


namespace volume_Q3_l146_146253

noncomputable def sequence_of_polyhedra (n : ℕ) : ℚ :=
match n with
| 0     => 1
| 1     => 3 / 2
| 2     => 45 / 32
| 3     => 585 / 128
| _     => 0 -- for n > 3 not defined

theorem volume_Q3 : sequence_of_polyhedra 3 = 585 / 128 :=
by
  -- Placeholder for the theorem proof
  sorry

end volume_Q3_l146_146253


namespace min_students_solved_both_l146_146335

/-- A simple mathematical proof problem to find the minimum number of students who solved both problems correctly --/
theorem min_students_solved_both (total_students first_problem second_problem : ℕ)
  (h₀ : total_students = 30)
  (h₁ : first_problem = 21)
  (h₂ : second_problem = 18) :
  ∃ (both_solved : ℕ), both_solved = 9 :=
by
  sorry

end min_students_solved_both_l146_146335


namespace a_2019_value_l146_146610

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0  -- not used, a_0 is irrelevant
  else if n = 1 then 1 / 2
  else a_sequence (n - 1) + 1 / (2 ^ (n - 1))

theorem a_2019_value :
  a_sequence 2019 = 3 / 2 - 1 / (2 ^ 2018) :=
by
  sorry

end a_2019_value_l146_146610


namespace find_black_balls_l146_146554

-- Define the conditions given in the problem.
def initial_balls : ℕ := 10
def all_red_balls (p_red : ℝ) : Prop := p_red = 1
def equal_red_black (p_red : ℝ) (p_black : ℝ) : Prop := p_red = 0.5 ∧ p_black = 0.5
def with_green_balls (p_red : ℝ) (green_balls : ℕ) : Prop := green_balls = 2 ∧ p_red = 0.7

-- Define the total probability condition
def total_probability (p_red : ℝ) (p_green : ℝ) (p_black : ℝ) : Prop :=
  p_red + p_green + p_black = 1

-- The final statement to prove
theorem find_black_balls :
  ∃ black_balls : ℕ,
    initial_balls = 10 ∧
    (∃ p_red : ℝ, all_red_balls p_red) ∧
    (∃ p_red p_black : ℝ, equal_red_black p_red p_black) ∧
    (∃ p_red : ℝ, ∃ green_balls : ℕ, with_green_balls p_red green_balls) ∧
    (∃ p_red p_green p_black : ℝ, total_probability p_red p_green p_black) ∧
    black_balls = 1 :=
sorry

end find_black_balls_l146_146554


namespace percentage_increase_l146_146937

theorem percentage_increase (original new : ℕ) (h₀ : original = 60) (h₁ : new = 120) :
  ((new - original) / original) * 100 = 100 := by
  sorry

end percentage_increase_l146_146937


namespace prob_at_least_one_head_l146_146031

theorem prob_at_least_one_head (n : ℕ) (hn : n = 3) : 
  1 - (1 / (2^n)) = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_l146_146031


namespace reflection_line_coordinates_sum_l146_146864

theorem reflection_line_coordinates_sum (m b : ℝ)
  (h : ∀ (x y x' y' : ℝ), (x, y) = (-4, 2) → (x', y') = (2, 6) → 
  ∃ (m b : ℝ), y = m * x + b ∧ y' = m * x' + b ∧ ∀ (p q : ℝ), 
  (p, q) = ((x+x')/2, (y+y')/2) → p = ((-4 + 2)/2) ∧ q = ((2 + 6)/2)) :
  m + b = 1 :=
by
  sorry

end reflection_line_coordinates_sum_l146_146864


namespace correct_systematic_sampling_l146_146000

-- Definitions for conditions in a)
def num_bags := 50
def num_selected := 5
def interval := num_bags / num_selected

-- We encode the systematic sampling selection process
def systematic_sampling (n : Nat) (start : Nat) (interval: Nat) (count : Nat) : List Nat :=
  List.range count |>.map (λ i => start + i * interval)

-- Theorem to prove that the selection of bags should have an interval of 10
theorem correct_systematic_sampling :
  ∃ (start : Nat), systematic_sampling num_selected start interval num_selected = [7, 17, 27, 37, 47] := sorry

end correct_systematic_sampling_l146_146000


namespace range_of_x_l146_146384

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_x (x m : ℝ) (hx : x > -2 ∧ x < 2/3) (hm : m ≥ -2 ∧ m ≤ 2) :
    f (m * x - 2) + f x < 0 := sorry

end range_of_x_l146_146384


namespace opposite_neg_2023_l146_146550

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l146_146550


namespace find_s_l146_146845

-- Define the roots of the quadratic equation
variables (a b n r s : ℝ)

-- Conditions from Vieta's formulas
def condition1 : Prop := a + b = n
def condition2 : Prop := a * b = 3

-- Roots of the second quadratic equation
def condition3 : Prop := (a + 1 / b) * (b + 1 / a) = s

-- The theorem statement
theorem find_s
  (h1 : condition1 a b n)
  (h2 : condition2 a b)
  (h3 : condition3 a b s) :
  s = 16 / 3 :=
by
  sorry

end find_s_l146_146845


namespace sum_mod_condition_l146_146796

theorem sum_mod_condition (a b c : ℤ) (h1 : a * b * c % 7 = 2)
                          (h2 : 3 * c % 7 = 1)
                          (h3 : 4 * b % 7 = (2 + b) % 7) :
                          (a + b + c) % 7 = 3 := by
  sorry

end sum_mod_condition_l146_146796


namespace find_a3_l146_146282

def sequence_sum (n : ℕ) : ℕ := n^2 + n

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem find_a3 : a 3 = 6 := by
  sorry

end find_a3_l146_146282


namespace evaluate_expression_l146_146104

theorem evaluate_expression : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end evaluate_expression_l146_146104


namespace find_j_l146_146042

def original_number (a b k : ℕ) : ℕ := 10 * a + b
def sum_of_digits (a b : ℕ) : ℕ := a + b
def modified_number (b a : ℕ) : ℕ := 20 * b + a

theorem find_j
  (a b k j : ℕ)
  (h1 : original_number a b k = k * sum_of_digits a b)
  (h2 : modified_number b a = j * sum_of_digits a b) :
  j = (199 + k) / 10 :=
sorry

end find_j_l146_146042


namespace clown_balloons_l146_146624

theorem clown_balloons 
  (initial_balloons : ℕ := 123) 
  (additional_balloons : ℕ := 53) 
  (given_away_balloons : ℕ := 27) : 
  initial_balloons + additional_balloons - given_away_balloons = 149 := 
by 
  sorry

end clown_balloons_l146_146624


namespace logs_per_tree_is_75_l146_146514

-- Definitions
def logsPerDay : Nat := 5

def totalDays : Nat := 30 + 31 + 31 + 28

def totalLogs (burnRate : Nat) (days : Nat) : Nat :=
  burnRate * days

def treesNeeded : Nat := 8

def logsPerTree (totalLogs : Nat) (numTrees : Nat) : Nat :=
  totalLogs / numTrees

-- Theorem statement to prove the number of logs per tree
theorem logs_per_tree_is_75 : logsPerTree (totalLogs logsPerDay totalDays) treesNeeded = 75 :=
  by
  sorry

end logs_per_tree_is_75_l146_146514


namespace g6_eq_16_l146_146801

-- Definition of the function g that satisfies the given conditions
variable (g : ℝ → ℝ)

-- Given conditions
axiom functional_eq : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g3_eq_4 : g 3 = 4

-- The goal is to prove g(6) = 16
theorem g6_eq_16 : g 6 = 16 := by
  sorry

end g6_eq_16_l146_146801


namespace f_decreasing_on_0_1_l146_146540

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x⁻¹

theorem f_decreasing_on_0_1 : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_0_1_l146_146540


namespace simplify_evaluate_l146_146483

theorem simplify_evaluate :
  ∀ (x : ℝ), x = Real.sqrt 2 - 1 →
  ((1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6))) = Real.sqrt 2 :=
by
  intros x hx
  sorry

end simplify_evaluate_l146_146483


namespace vision_statistics_l146_146640

noncomputable def average (values : List ℝ) : ℝ := (List.sum values) / (List.length values)

noncomputable def variance (values : List ℝ) : ℝ :=
  let mean := average values
  (List.sum (values.map (λ x => (x - mean) ^ 2))) / (List.length values)

def classA_visions : List ℝ := [4.3, 5.1, 4.6, 4.1, 4.9]
def classB_visions : List ℝ := [5.1, 4.9, 4.0, 4.0, 4.5]

theorem vision_statistics :
  average classA_visions = 4.6 ∧
  average classB_visions = 4.5 ∧
  variance classA_visions = 0.136 ∧
  (let count := List.length classB_visions
   let total := count.choose 2
   let favorable := 3  -- (5.1, 4.5), (5.1, 4.9), (4.9, 4.5)
   7 / 10 = 1 - (favorable / total)) :=
by
  sorry

end vision_statistics_l146_146640


namespace correct_operation_l146_146652

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^4 ≠ a^6) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  ((a^2 * b)^3 = a^6 * b^3) ∧
  (a^6 / a^6 ≠ a) :=
by
  sorry

end correct_operation_l146_146652


namespace smallest_positive_integer_l146_146347

theorem smallest_positive_integer {x : ℕ} (h1 : x % 6 = 3) (h2 : x % 8 = 5) : x = 21 :=
sorry

end smallest_positive_integer_l146_146347


namespace cost_of_playing_cards_l146_146668

theorem cost_of_playing_cards 
  (allowance_each : ℕ)
  (combined_allowance : ℕ)
  (sticker_box_cost : ℕ)
  (number_of_sticker_packs : ℕ)
  (number_of_packs_Dora_got : ℕ)
  (cost_of_playing_cards : ℕ)
  (h1 : allowance_each = 9)
  (h2 : combined_allowance = allowance_each * 2)
  (h3 : sticker_box_cost = 2)
  (h4 : number_of_packs_Dora_got = 2)
  (h5 : number_of_sticker_packs = number_of_packs_Dora_got * 2)
  (h6 : combined_allowance - number_of_sticker_packs * sticker_box_cost = cost_of_playing_cards) :
  cost_of_playing_cards = 10 :=
sorry

end cost_of_playing_cards_l146_146668


namespace sum_digits_350_1350_base2_l146_146280

def binary_sum_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

theorem sum_digits_350_1350_base2 :
  binary_sum_digits 350 + binary_sum_digits 1350 = 20 :=
by
  sorry

end sum_digits_350_1350_base2_l146_146280


namespace solve_for_x_l146_146759

theorem solve_for_x :
  (∀ x : ℝ, (1 / Real.log x / Real.log 3 + 1 / Real.log x / Real.log 4 + 1 / Real.log x / Real.log 5 = 2))
  → x = 2 * Real.sqrt 15 :=
by
  sorry

end solve_for_x_l146_146759


namespace value_independent_of_a_value_when_b_is_neg_2_l146_146010

noncomputable def algebraic_expression (a b : ℝ) : ℝ :=
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2)

theorem value_independent_of_a (a b : ℝ) : algebraic_expression a b = 2 * b^2 :=
by
  sorry

theorem value_when_b_is_neg_2 (a : ℝ) : algebraic_expression a (-2) = 8 :=
by
  sorry

end value_independent_of_a_value_when_b_is_neg_2_l146_146010


namespace kaleb_toys_l146_146278

def initial_savings : ℕ := 21
def allowance : ℕ := 15
def cost_per_toy : ℕ := 6

theorem kaleb_toys : (initial_savings + allowance) / cost_per_toy = 6 :=
by
  sorry

end kaleb_toys_l146_146278


namespace positive_integer_pairs_l146_146407

theorem positive_integer_pairs (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (∃ k : ℕ, k > 0 ∧ k = a^2 / (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, 0 < l ∧ 
    ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
by
  sorry

end positive_integer_pairs_l146_146407


namespace merchant_gross_profit_l146_146779

noncomputable def purchase_price : ℝ := 48
noncomputable def markup_rate : ℝ := 0.40
noncomputable def discount_rate : ℝ := 0.20

theorem merchant_gross_profit :
  ∃ S : ℝ, S = purchase_price + markup_rate * S ∧ 
  ((S - discount_rate * S) - purchase_price = 16) :=
by
  sorry

end merchant_gross_profit_l146_146779


namespace arithmetic_sequence_formula_l146_146926

-- Define the sequence and its properties
def is_arithmetic_sequence (a : ℤ) (u : ℕ → ℤ) : Prop :=
  u 0 = a - 1 ∧ u 1 = a + 1 ∧ u 2 = 2 * a + 3 ∧ ∀ n, u (n + 1) - u n = u 1 - u 0

theorem arithmetic_sequence_formula (a : ℤ) :
  ∃ u : ℕ → ℤ, is_arithmetic_sequence a u ∧ (∀ n, u n = 2 * n - 3) :=
by
  sorry

end arithmetic_sequence_formula_l146_146926


namespace first_month_sale_l146_146472

def sale_second_month : ℕ := 5744
def sale_third_month : ℕ := 5864
def sale_fourth_month : ℕ := 6122
def sale_fifth_month : ℕ := 6588
def sale_sixth_month : ℕ := 4916
def average_sale_six_months : ℕ := 5750

def expected_total_sales : ℕ := 6 * average_sale_six_months
def known_sales : ℕ := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month

theorem first_month_sale :
  (expected_total_sales - (known_sales + sale_sixth_month)) = 5266 :=
by
  sorry

end first_month_sale_l146_146472


namespace isosceles_triangle_base_angle_l146_146004

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end isosceles_triangle_base_angle_l146_146004


namespace mary_unanswered_questions_l146_146507

theorem mary_unanswered_questions :
  ∃ (c w u : ℕ), 150 = 6 * c + 3 * u ∧ 118 = 40 + 5 * c - 2 * w ∧ 50 = c + w + u ∧ u = 16 :=
by
  sorry

end mary_unanswered_questions_l146_146507


namespace jake_pure_alcohol_l146_146562

-- Definitions based on the conditions
def shots : ℕ := 8
def ounces_per_shot : ℝ := 1.5
def vodka_purity : ℝ := 0.5
def friends : ℕ := 2

-- Statement to prove the amount of pure alcohol Jake drank
theorem jake_pure_alcohol : (shots * ounces_per_shot * vodka_purity) / friends = 3 := by
  sorry

end jake_pure_alcohol_l146_146562


namespace cartons_in_case_l146_146487

theorem cartons_in_case (b : ℕ) (hb : b ≥ 1) (h : 2 * c * b * 500 = 1000) : c = 1 :=
by
  -- sorry is used to indicate where the proof would go
  sorry

end cartons_in_case_l146_146487


namespace solutionToSystemOfEquations_solutionToSystemOfInequalities_l146_146126

open Classical

noncomputable def solveSystemOfEquations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solutionToSystemOfEquations : ∃ (x y : ℝ), solveSystemOfEquations x y ∧ x = 4 ∧ y = 5 := by
  sorry

def solveSystemOfInequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solutionToSystemOfInequalities : ∃ x : ℝ, solveSystemOfInequalities x ∧ 1 ≤ x ∧ x < 2 := by
  sorry

end solutionToSystemOfEquations_solutionToSystemOfInequalities_l146_146126


namespace sum_first_n_terms_geometric_sequence_l146_146045

def geometric_sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  if n = 0 then 0 else (3 * 2^n + k)

theorem sum_first_n_terms_geometric_sequence (k : ℝ) :
  (geometric_sequence_sum 1 k = 6 + k) ∧ 
  (∀ n > 1, geometric_sequence_sum n k - geometric_sequence_sum (n - 1) k = 3 * 2^(n-1))
  → k = -3 :=
by
  sorry

end sum_first_n_terms_geometric_sequence_l146_146045


namespace initial_amount_l146_146622

-- Define the given conditions
def amount_spent : ℕ := 16
def amount_left : ℕ := 2

-- Define the statement that we want to prove
theorem initial_amount : amount_spent + amount_left = 18 :=
by
  sorry

end initial_amount_l146_146622


namespace bottle_caps_per_box_l146_146820

theorem bottle_caps_per_box (total_caps : ℕ) (total_boxes : ℕ) (h_total_caps : total_caps = 60) (h_total_boxes : total_boxes = 60) :
  (total_caps / total_boxes) = 1 :=
by {
  sorry
}

end bottle_caps_per_box_l146_146820


namespace distance_between_cities_l146_146988

theorem distance_between_cities
    (v_bus : ℕ) (v_car : ℕ) (t_bus_meet : ℚ) (t_car_wait : ℚ)
    (d_overtake : ℚ) (s : ℚ)
    (h_vb : v_bus = 40)
    (h_vc : v_car = 50)
    (h_tbm : t_bus_meet = 0.25)
    (h_tcw : t_car_wait = 0.25)
    (h_do : d_overtake = 20)
    (h_eq : (s - 10) / 50 + t_car_wait = (s - 30) / 40) :
    s = 160 :=
by
    exact sorry

end distance_between_cities_l146_146988


namespace total_people_l146_146662

-- Define the conditions as constants
def B : ℕ := 50
def S : ℕ := 70
def B_inter_S : ℕ := 20

-- Total number of people in the group
theorem total_people : B + S - B_inter_S = 100 := by
  sorry

end total_people_l146_146662


namespace dog_roaming_area_l146_146144

theorem dog_roaming_area :
  let shed_radius := 20
  let rope_length := 10
  let distance_from_edge := 10
  let radius_from_center := shed_radius - distance_from_edge
  radius_from_center = rope_length →
  (π * rope_length^2 = 100 * π) :=
by
  intros shed_radius rope_length distance_from_edge radius_from_center h
  sorry

end dog_roaming_area_l146_146144


namespace sum_of_angles_l146_146754

theorem sum_of_angles (α β : ℝ) (hα: 0 < α ∧ α < π) (hβ: 0 < β ∧ β < π) (h_tan_α: Real.tan α = 1 / 2) (h_tan_β: Real.tan β = 1 / 3) : α + β = π / 4 := 
by 
  sorry

end sum_of_angles_l146_146754


namespace square_area_l146_146152

theorem square_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  ∃ (s : ℝ), (s * Real.sqrt 2 = d) ∧ (s^2 = 144) := by
  sorry

end square_area_l146_146152


namespace fish_count_together_l146_146592

namespace FishProblem

def JerkTunaFish : ℕ := 144
def TallTunaFish : ℕ := 2 * JerkTunaFish
def SwellTunaFish : ℕ := TallTunaFish + (TallTunaFish / 2)
def totalFish : ℕ := JerkTunaFish + TallTunaFish + SwellTunaFish

theorem fish_count_together : totalFish = 864 := by
  sorry

end FishProblem

end fish_count_together_l146_146592


namespace jason_daily_charge_l146_146511

theorem jason_daily_charge 
  (total_cost_eric : ℕ) (days_eric : ℕ) (daily_charge : ℕ)
  (h1 : total_cost_eric = 800) (h2 : days_eric = 20)
  (h3 : daily_charge = total_cost_eric / days_eric) :
  daily_charge = 40 := 
by
  sorry

end jason_daily_charge_l146_146511


namespace valid_divisors_of_196_l146_146076

theorem valid_divisors_of_196 : 
  ∃ d : Finset Nat, (∀ x ∈ d, 1 < x ∧ x < 196 ∧ 196 % x = 0) ∧ d.card = 7 := by
  sorry

end valid_divisors_of_196_l146_146076


namespace difference_of_squares_example_l146_146758

theorem difference_of_squares_example :
  262^2 - 258^2 = 2080 := by
sorry

end difference_of_squares_example_l146_146758


namespace bigger_wheel_roll_distance_l146_146100

/-- The circumference of the bigger wheel is 12 meters -/
def bigger_wheel_circumference : ℕ := 12

/-- The circumference of the smaller wheel is 8 meters -/
def smaller_wheel_circumference : ℕ := 8

/-- The distance the bigger wheel must roll for the points P1 and P2 to coincide again -/
theorem bigger_wheel_roll_distance : Nat.lcm bigger_wheel_circumference smaller_wheel_circumference = 24 :=
by
  -- Proof is omitted
  sorry

end bigger_wheel_roll_distance_l146_146100


namespace binomial_param_exact_l146_146524

variable (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ)

-- Define the conditions: expectation and variance
axiom expectation_eq : n * p = 3
axiom variance_eq : n * p * (1 - p) = 2

-- Statement to prove
theorem binomial_param_exact (h1 : n * p = 3) (h2 : n * p * (1 - p) = 2) : p = 1 / 3 :=
by
  rw [expectation_eq] at h2
  sorry

end binomial_param_exact_l146_146524


namespace polynomial_equality_l146_146154

theorem polynomial_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 :=
by 
  sorry

end polynomial_equality_l146_146154


namespace find_Y_exists_l146_146972

variable {X : Finset ℕ} -- Consider a finite set X of natural numbers for generality
variable (S : Finset (Finset ℕ)) -- Set of all subsets of X with even number of elements
variable (f : Finset ℕ → ℝ) -- Real-valued function on subsets of X

-- Conditions
variable (hS : ∀ s ∈ S, s.card % 2 = 0) -- All elements in S have even number of elements
variable (h1 : ∃ A ∈ S, f A > 1990) -- f(A) > 1990 for some A ∈ S
variable (h2 : ∀ ⦃B C⦄, B ∈ S → C ∈ S → (Disjoint B C) → (f (B ∪ C) = f B + f C - 1990)) -- f respects the functional equation for disjoint subsets

theorem find_Y_exists :
  ∃ Y ⊆ X, (∀ D ∈ S, D ⊆ Y → f D > 1990) ∧ (∀ D ∈ S, D ⊆ (X \ Y) → f D ≤ 1990) :=
by
  sorry

end find_Y_exists_l146_146972


namespace problems_per_page_l146_146876

def total_problems : ℕ := 60
def finished_problems : ℕ := 20
def remaining_pages : ℕ := 5

theorem problems_per_page :
  (total_problems - finished_problems) / remaining_pages = 8 :=
by
  sorry

end problems_per_page_l146_146876


namespace quadratic_solution_pair_l146_146262

open Real

noncomputable def solution_pair : ℝ × ℝ :=
  ((45 - 15 * sqrt 5) / 2, (45 + 15 * sqrt 5) / 2)

theorem quadratic_solution_pair (a c : ℝ) 
  (h1 : (∃ x : ℝ, a * x^2 + 30 * x + c = 0 ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 30 * y + c ≠ 0))
  (h2 : a + c = 45)
  (h3 : a < c) :
  (a, c) = solution_pair :=
sorry

end quadratic_solution_pair_l146_146262


namespace apple_lovers_l146_146026

theorem apple_lovers :
  ∃ (x y : ℕ), 22 * x = 1430 ∧ 13 * (x + y) = 1430 ∧ y = 45 :=
by
  sorry

end apple_lovers_l146_146026


namespace find_number_l146_146822

def sum := 555 + 445
def difference := 555 - 445
def quotient := 2 * difference
def remainder := 30
def N : ℕ := 220030

theorem find_number (N : ℕ) : 
  N = sum * quotient + remainder :=
  by
    sorry

end find_number_l146_146822


namespace min_value_x_l146_146406

theorem min_value_x (x : ℝ) (h : ∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) : x ≥ -1 := 
sorry

end min_value_x_l146_146406


namespace fraction_phone_numbers_9_ending_even_l146_146284

def isValidPhoneNumber (n : Nat) : Bool :=
  n / 10^6 != 0 && n / 10^6 != 1 && n / 10^6 != 2

def isValidEndEven (n : Nat) : Bool :=
  let lastDigit := n % 10
  lastDigit == 0 || lastDigit == 2 || lastDigit == 4 || lastDigit == 6 || lastDigit == 8

def countValidPhoneNumbers : Nat :=
  7 * 10^6

def countValidStarting9EndingEven : Nat :=
  5 * 10^5

theorem fraction_phone_numbers_9_ending_even :
  (countValidStarting9EndingEven : ℚ) / (countValidPhoneNumbers : ℚ) = 1 / 14 :=
by 
  sorry

end fraction_phone_numbers_9_ending_even_l146_146284


namespace diana_total_extra_video_game_time_l146_146244

-- Definitions from the conditions
def minutesPerHourReading := 30
def raisePercent := 20
def choresToMinutes := 10
def maxChoresBonusMinutes := 60
def sportsPracticeHours := 8
def homeworkHours := 4
def totalWeekHours := 24
def readingHours := 8
def choresCompleted := 10

-- Deriving some necessary facts
def baseVideoGameTime := readingHours * minutesPerHourReading
def raiseMinutes := baseVideoGameTime * (raisePercent / 100)
def videoGameTimeWithRaise := baseVideoGameTime + raiseMinutes

def bonusesFromChores := (choresCompleted / 2) * choresToMinutes
def limitedChoresBonus := min bonusesFromChores maxChoresBonusMinutes

-- Total extra video game time
def totalExtraVideoGameTime := videoGameTimeWithRaise + limitedChoresBonus

-- The proof problem
theorem diana_total_extra_video_game_time : totalExtraVideoGameTime = 338 := by
  sorry

end diana_total_extra_video_game_time_l146_146244


namespace problem_l146_146063

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l146_146063


namespace greatest_odd_integer_x_l146_146857

theorem greatest_odd_integer_x (x : ℕ) (h1 : x % 2 = 1) (h2 : x^4 / x^2 < 50) : x ≤ 7 :=
sorry

end greatest_odd_integer_x_l146_146857


namespace factorial_square_product_l146_146893

theorem factorial_square_product : (Real.sqrt (Nat.factorial 6 * Nat.factorial 4)) ^ 2 = 17280 := by
  sorry

end factorial_square_product_l146_146893


namespace find_k_l146_146159

noncomputable def k := 3

theorem find_k :
  (∀ x : ℝ, (Real.sin x ^ k) * (Real.sin (k * x)) + (Real.cos x ^ k) * (Real.cos (k * x)) = Real.cos (2 * x) ^ k) ↔ k = 3 :=
sorry

end find_k_l146_146159


namespace bus_routes_theorem_l146_146642

open Function

def bus_routes_exist : Prop :=
  ∃ (routes : Fin 10 → Set (Fin 10)), 
  (∀ (s : Finset (Fin 10)), (s.card = 8) → ∃ (stop : Fin 10), ∀ i ∈ s, stop ∉ routes i) ∧
  (∀ (s : Finset (Fin 10)), (s.card = 9) → ∀ (stop : Fin 10), ∃ i ∈ s, stop ∈ routes i)

theorem bus_routes_theorem : bus_routes_exist :=
sorry

end bus_routes_theorem_l146_146642


namespace fraction_subtraction_l146_146390

theorem fraction_subtraction (a b : ℚ) (h_a: a = 5/9) (h_b: b = 1/6) : a - b = 7/18 :=
by
  sorry

end fraction_subtraction_l146_146390


namespace connie_tickets_l146_146804

theorem connie_tickets (total_tickets spent_on_koala spent_on_earbuds spent_on_glow_bracelets : ℕ)
  (h1 : total_tickets = 50)
  (h2 : spent_on_koala = total_tickets / 2)
  (h3 : spent_on_earbuds = 10)
  (h4 : total_tickets = spent_on_koala + spent_on_earbuds + spent_on_glow_bracelets) :
  spent_on_glow_bracelets = 15 :=
by
  sorry

end connie_tickets_l146_146804


namespace a_equals_1_or_2_l146_146985

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x : ℤ | x^2 - 3 * x < 0}
def non_empty_intersection (a : ℤ) : Prop := (M a ∩ N).Nonempty

theorem a_equals_1_or_2 (a : ℤ) (h : non_empty_intersection a) : a = 1 ∨ a = 2 := by
  sorry

end a_equals_1_or_2_l146_146985


namespace height_of_trapezoid_l146_146283

-- Define the condition that a trapezoid has diagonals of given lengths and a given midline.
def trapezoid_conditions (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : Prop := 
  AC = 6 ∧ BD = 8 ∧ ML = 5

-- Define the height of the trapezoid.
def trapezoid_height (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : ℝ :=
  4.8

-- The theorem statement
theorem height_of_trapezoid (AC BD ML h : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : 
  trapezoid_conditions AC BD ML h_d1 h_d2 h_ml 
  → trapezoid_height AC BD ML h_d1 h_d2 h_ml = 4.8 := 
by
  intros
  sorry

end height_of_trapezoid_l146_146283


namespace evaluate_composite_l146_146190

def f (x : ℕ) : ℕ := 2 * x + 5
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_composite : f (g (f 3)) = 79 := by
  sorry

end evaluate_composite_l146_146190


namespace fraction_a_over_d_l146_146872

-- Defining the given conditions as hypotheses
variables (a b c d : ℚ)

-- Conditions
axiom h1 : a / b = 20
axiom h2 : c / b = 5
axiom h3 : c / d = 1 / 15

-- Goal to prove
theorem fraction_a_over_d : a / d = 4 / 15 :=
by
  sorry

end fraction_a_over_d_l146_146872


namespace caleb_hamburgers_total_l146_146191

def total_spent : ℝ := 66.50
def cost_single : ℝ := 1.00
def cost_double : ℝ := 1.50
def num_double : ℕ := 33

theorem caleb_hamburgers_total : 
  ∃ n : ℕ,  n = 17 + num_double ∧ 
            (num_double * cost_double) + (n - num_double) * cost_single = total_spent := by
sorry

end caleb_hamburgers_total_l146_146191


namespace pens_sold_l146_146675

variable (C S : ℝ)
variable (n : ℕ)

-- Define conditions
def condition1 : Prop := 10 * C = n * S
def condition2 : Prop := S = 1.5 * C

-- Define the statement to be proved
theorem pens_sold (h1 : condition1 C S n) (h2 : condition2 C S) : n = 6 := by
  -- leave the proof steps to be filled in
  sorry

end pens_sold_l146_146675


namespace angle_ABC_is_83_l146_146913

-- Define a structure for the quadrilateral ABCD 
structure Quadrilateral (A B C D : Type) :=
  (angle_BAC : ℝ) -- Measure in degrees
  (angle_CAD : ℝ) -- Measure in degrees
  (angle_ACD : ℝ) -- Measure in degrees
  (side_AB : ℝ) -- Lengths of sides
  (side_AD : ℝ)
  (side_AC : ℝ)

-- Define the conditions from the problem
variable {A B C D : Type}
variable (quad : Quadrilateral A B C D)
variable (h1 : quad.angle_BAC = 60)
variable (h2 : quad.angle_CAD = 60)
variable (h3 : quad.angle_ACD = 23)
variable (h4 : quad.side_AB + quad.side_AD = quad.side_AC)

-- State the theorem to be proved
theorem angle_ABC_is_83 : quad.angle_ACD = 23 → quad.angle_CAD = 60 → 
                           quad.angle_BAC = 60 → quad.side_AB + quad.side_AD = quad.side_AC → 
                           ∃ angle_ABC : ℝ, angle_ABC = 83 := by
  sorry

end angle_ABC_is_83_l146_146913


namespace range_of_a_l146_146928

theorem range_of_a (x a : ℝ) (h₀ : x < 0) (h₁ : 2^x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l146_146928


namespace cost_of_fencing_per_meter_l146_146746

theorem cost_of_fencing_per_meter
  (breadth : ℝ)
  (length : ℝ)
  (cost : ℝ)
  (length_eq : length = breadth + 40)
  (total_cost : cost = 5300)
  (length_given : length = 70) :
  cost / (2 * length + 2 * breadth) = 26.5 :=
by
  sorry

end cost_of_fencing_per_meter_l146_146746


namespace complex_number_z_value_l146_146798

open Complex

theorem complex_number_z_value :
  ∀ (i z : ℂ), i^2 = -1 ∧ z * (1 + i) = 2 * i^2018 → z = -1 + i :=
by
  intros i z h
  have h1 : i^2 = -1 := h.1
  have h2 : z * (1 + i) = 2 * i^2018 := h.2
  sorry

end complex_number_z_value_l146_146798


namespace pages_left_after_all_projects_l146_146713

-- Definitions based on conditions
def initial_pages : ℕ := 120
def pages_for_science : ℕ := (initial_pages * 25) / 100
def pages_for_math : ℕ := 10
def pages_after_science_and_math : ℕ := initial_pages - pages_for_science - pages_for_math
def pages_for_history : ℕ := (initial_pages * 15) / 100
def pages_after_history : ℕ := pages_after_science_and_math - pages_for_history
def remaining_pages : ℕ := pages_after_history / 2

theorem pages_left_after_all_projects :
  remaining_pages = 31 :=
  by
  sorry

end pages_left_after_all_projects_l146_146713


namespace odd_exponent_divisibility_l146_146459

theorem odd_exponent_divisibility (x y : ℤ) (k : ℕ) (h : (x^(2*k-1) + y^(2*k-1)) % (x + y) = 0) : 
  (x^(2*k+1) + y^(2*k+1)) % (x + y) = 0 :=
sorry

end odd_exponent_divisibility_l146_146459


namespace frequency_of_middle_group_l146_146218

theorem frequency_of_middle_group :
  ∃ m : ℝ, m + (1/3) * m = 200 ∧ (1/3) * m = 50 :=
by
  sorry

end frequency_of_middle_group_l146_146218


namespace annie_bought_figurines_l146_146370

theorem annie_bought_figurines:
  let televisions := 5
  let cost_per_television := 50
  let total_spent := 260
  let cost_per_figurine := 1
  let cost_of_televisions := televisions * cost_per_television
  let remaining_money := total_spent - cost_of_televisions
  remaining_money / cost_per_figurine = 10 :=
by
  sorry

end annie_bought_figurines_l146_146370


namespace gcd_5280_12155_l146_146064

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 5 :=
by
  sorry

end gcd_5280_12155_l146_146064


namespace bridge_length_correct_l146_146009

def train_length : ℕ := 256
def train_speed_kmh : ℕ := 72
def crossing_time : ℕ := 20

noncomputable def convert_speed (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600 -- Conversion from km/h to m/s

noncomputable def bridge_length (train_length : ℕ) (speed_m : ℕ) (time_s : ℕ) : ℕ :=
  (speed_m * time_s) - train_length

theorem bridge_length_correct :
  bridge_length train_length (convert_speed train_speed_kmh) crossing_time = 144 :=
by
  sorry

end bridge_length_correct_l146_146009


namespace remainder_when_divided_by_17_l146_146749

theorem remainder_when_divided_by_17
  (N k : ℤ)
  (h : N = 357 * k + 36) :
  N % 17 = 2 :=
by
  sorry

end remainder_when_divided_by_17_l146_146749


namespace pow_four_inequality_l146_146963

theorem pow_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x * y * (x + y)^2 :=
by
  sorry

end pow_four_inequality_l146_146963


namespace product_of_roots_cubic_l146_146450

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l146_146450


namespace sqrt_of_expression_l146_146362

theorem sqrt_of_expression :
  Real.sqrt (4^4 * 9^2) = 144 :=
sorry

end sqrt_of_expression_l146_146362


namespace age_ratio_l146_146451

-- Conditions
def DeepakPresentAge := 27
def RahulAgeAfterSixYears := 42
def YearsToReach42 := 6

-- The theorem to prove the ratio of their ages
theorem age_ratio (R D : ℕ) (hR : R + YearsToReach42 = RahulAgeAfterSixYears) (hD : D = DeepakPresentAge) : R / D = 4 / 3 := by
  sorry

end age_ratio_l146_146451


namespace sum_first_12_terms_l146_146040

theorem sum_first_12_terms (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n * a n)
  (h2 : a 6 + a 7 = 18) : 
  S 12 = 108 :=
sorry

end sum_first_12_terms_l146_146040


namespace probability_line_through_cube_faces_l146_146265

def prob_line_intersects_cube_faces : ℚ :=
  1 / 7

theorem probability_line_through_cube_faces :
  let cube_vertices := 8
  let total_selections := Nat.choose cube_vertices 2
  let body_diagonals := 4
  let probability := (body_diagonals : ℚ) / total_selections
  probability = prob_line_intersects_cube_faces :=
by {
  sorry
}

end probability_line_through_cube_faces_l146_146265


namespace pqr_value_l146_146216

theorem pqr_value (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h1 : p + q + r = 24)
  (h2 : (1 / p : ℚ) + (1 / q) + (1 / r) + 240 / (p * q * r) = 1): 
  p * q * r = 384 :=
by
  sorry

end pqr_value_l146_146216


namespace red_cars_count_l146_146619

variable (R B : ℕ)
variable (h1 : R * 8 = 3 * B)
variable (h2 : B = 90)

theorem red_cars_count : R = 33 :=
by
  -- here we would provide the proof
  sorry

end red_cars_count_l146_146619


namespace total_snow_volume_l146_146649

-- Definitions and conditions set up from part (a)
def driveway_length : ℝ := 30
def driveway_width : ℝ := 3
def section1_length : ℝ := 10
def section1_depth : ℝ := 1
def section2_length : ℝ := driveway_length - section1_length
def section2_depth : ℝ := 0.5

-- The theorem corresponding to part (c)
theorem total_snow_volume : 
  (section1_length * driveway_width * section1_depth) +
  (section2_length * driveway_width * section2_depth) = 60 :=
by 
  -- Proof is omitted as required
  sorry

end total_snow_volume_l146_146649


namespace sufficient_but_not_necessary_l146_146761

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → 2 / a < 1) ∧ (2 / a < 1 → a > 2 ∨ a < 0) :=
by sorry

end sufficient_but_not_necessary_l146_146761


namespace completing_the_square_x_squared_minus_4x_plus_1_eq_0_l146_146408

theorem completing_the_square_x_squared_minus_4x_plus_1_eq_0 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x
  intros h
  sorry

end completing_the_square_x_squared_minus_4x_plus_1_eq_0_l146_146408


namespace optimalBananaBuys_l146_146719

noncomputable def bananaPrices : List ℕ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

def days := List.range 18

def computeOptimalBuys : List ℕ :=
  sorry -- Implement the logic to compute the optimal number of bananas to buy each day.

theorem optimalBananaBuys :
  computeOptimalBuys = [4, 0, 0, 3, 0, 0, 7, 0, 0, 1, 0, 0, 4, 0, 0, 3, 0, 1] :=
sorry

end optimalBananaBuys_l146_146719


namespace evaluate_x_squared_minus_y_squared_l146_146318

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l146_146318


namespace sum_of_two_digit_odd_numbers_l146_146129

-- Define the set of all two-digit numbers with both digits odd
def two_digit_odd_numbers : List ℕ := 
  [11, 13, 15, 17, 19, 31, 33, 35, 37, 39,
   51, 53, 55, 57, 59, 71, 73, 75, 77, 79,
   91, 93, 95, 97, 99]

-- Define a function to compute the sum of elements in a list
def list_sum (l : List ℕ) : ℕ := l.foldl (.+.) 0

theorem sum_of_two_digit_odd_numbers :
  list_sum two_digit_odd_numbers = 1375 :=
by
  sorry

end sum_of_two_digit_odd_numbers_l146_146129


namespace arithmetic_prog_sum_bound_l146_146250

noncomputable def Sn (n : ℕ) (a1 : ℝ) (d : ℝ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_prog_sum_bound (n : ℕ) (a1 an : ℝ) (d : ℝ) (h_d_neg : d < 0) 
  (ha_n : an = a1 + (n - 1) * d) :
  n * an < Sn n a1 d ∧ Sn n a1 d < n * a1 :=
by 
  sorry

end arithmetic_prog_sum_bound_l146_146250


namespace three_digit_integer_one_more_than_multiple_l146_146997

theorem three_digit_integer_one_more_than_multiple :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n = 841 ∧ ∃ k : ℕ, n = 840 * k + 1 :=
by
  sorry

end three_digit_integer_one_more_than_multiple_l146_146997


namespace ratio_of_prices_l146_146393

-- Define the problem
theorem ratio_of_prices (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.2 * CP) 
  (h2 : SP2 = CP - 0.2 * CP) : 
  SP2 / SP1 = 2 / 3 :=
by
  -- proof
  sorry

end ratio_of_prices_l146_146393


namespace grandmother_ratio_l146_146755

noncomputable def Grace_Age := 60
noncomputable def Mother_Age := 80

theorem grandmother_ratio :
  ∃ GM, Grace_Age = (3 / 8 : Rat) * GM ∧ GM / Mother_Age = 2 :=
by
  sorry

end grandmother_ratio_l146_146755


namespace lois_final_books_l146_146090

-- Definitions for the conditions given in the problem.
def initial_books : ℕ := 40
def books_given_to_nephew (b : ℕ) : ℕ := b / 4
def books_remaining_after_giving (b_given : ℕ) (b : ℕ) : ℕ := b - b_given
def books_donated_to_library (b_remaining : ℕ) : ℕ := b_remaining / 3
def books_remaining_after_donating (b_donated : ℕ) (b_remaining : ℕ) : ℕ := b_remaining - b_donated
def books_purchased : ℕ := 3
def total_books (b_final_remaining : ℕ) (b_purchased : ℕ) : ℕ := b_final_remaining + b_purchased

-- Theorem stating: Given the initial conditions, Lois should have 23 books in the end.
theorem lois_final_books : 
  total_books 
    (books_remaining_after_donating (books_donated_to_library (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books)) 
    (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books))
    books_purchased = 23 :=
  by
    sorry  -- Proof omitted as per instructions.

end lois_final_books_l146_146090


namespace correct_operation_l146_146772

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l146_146772


namespace unique_solution_c_value_l146_146145

-- Define the main problem: the parameter c for which a given system of equations has a unique solution.
theorem unique_solution_c_value (c : ℝ) : 
  (∀ x y : ℝ, 2 * abs (x + 7) + abs (y - 4) = c ∧ abs (x + 4) + 2 * abs (y - 7) = c → 
   (x = -7 ∧ y = 7)) ↔ c = 3 :=
by sorry

end unique_solution_c_value_l146_146145


namespace cone_angle_60_degrees_l146_146644

theorem cone_angle_60_degrees (r : ℝ) (h : ℝ) (θ : ℝ) 
  (arc_len : θ = 60) 
  (slant_height : h = r) : θ = 60 :=
sorry

end cone_angle_60_degrees_l146_146644


namespace total_distance_traveled_l146_146433

noncomputable def total_distance (d v1 v2 v3 time_total : ℝ) : ℝ :=
  3 * d

theorem total_distance_traveled
  (d : ℝ)
  (v1 : ℝ := 3)
  (v2 : ℝ := 6)
  (v3 : ℝ := 9)
  (time_total : ℝ := 11 / 60)
  (h : d / v1 + d / v2 + d / v3 = time_total) :
  total_distance d v1 v2 v3 time_total = 0.9 :=
by
  sorry

end total_distance_traveled_l146_146433


namespace maximize_profit_l146_146411

noncomputable def profit (x : ℝ) : ℝ :=
  let selling_price := 10 + 0.5 * x
  let sales_volume := 200 - 10 * x
  (selling_price - 8) * sales_volume

theorem maximize_profit : ∃ x : ℝ, x = 8 → profit x = profit 8 ∧ (∀ y : ℝ, profit y ≤ profit 8) := 
  sorry

end maximize_profit_l146_146411


namespace bob_stickers_l146_146686

variables {B T D : ℕ}

theorem bob_stickers (h1 : D = 72) (h2 : T = 3 * B) (h3 : D = 2 * T) : B = 12 :=
by
  sorry

end bob_stickers_l146_146686


namespace divide_milk_in_half_l146_146643

theorem divide_milk_in_half (bucket : ℕ) (a : ℕ) (b : ℕ) (a_liters : a = 5) (b_liters : b = 7) (bucket_liters : bucket = 12) :
  ∃ x y : ℕ, x = 6 ∧ y = 6 ∧ x + y = bucket := by
  sorry

end divide_milk_in_half_l146_146643


namespace calc_g_x_plus_2_minus_g_x_l146_146650

def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem calc_g_x_plus_2_minus_g_x (x : ℝ) : g (x + 2) - g x = 12 * x + 22 := 
by 
  sorry

end calc_g_x_plus_2_minus_g_x_l146_146650


namespace percentage_of_students_who_own_cats_l146_146506

theorem percentage_of_students_who_own_cats (total_students cats_owned : ℕ) (h_total: total_students = 500) (h_cats: cats_owned = 75) :
  (cats_owned : ℚ) / total_students * 100 = 15 :=
by
  sorry

end percentage_of_students_who_own_cats_l146_146506


namespace small_cubes_one_face_painted_red_l146_146320

-- Definitions
def is_red_painted (cube : ℕ) : Bool := true -- representing the condition that the cube is painted red
def side_length (cube : ℕ) : ℕ := 4 -- side length of the original cube is 4 cm
def smaller_cube_side_length : ℕ := 1 -- smaller cube side length is 1 cm

-- Theorem Statement
theorem small_cubes_one_face_painted_red :
  ∀ (large_cube : ℕ), (side_length large_cube = 4) ∧ is_red_painted large_cube → 
  (∃ (number_of_cubes : ℕ), number_of_cubes = 24) :=
by
  sorry

end small_cubes_one_face_painted_red_l146_146320


namespace total_votes_cast_l146_146206

theorem total_votes_cast (V : ℕ) (C R : ℕ) 
  (hC : C = 30 * V / 100) 
  (hR1 : R = C + 4000) 
  (hR2 : R = 70 * V / 100) : 
  V = 10000 :=
by
  sorry

end total_votes_cast_l146_146206


namespace buses_required_l146_146141

theorem buses_required (students : ℕ) (bus_capacity : ℕ) (h_students : students = 325) (h_bus_capacity : bus_capacity = 45) : 
∃ n : ℕ, n = 8 ∧ bus_capacity * n ≥ students :=
by
  sorry

end buses_required_l146_146141


namespace dan_has_more_balloons_l146_146006

-- Constants representing the number of balloons Dan and Tim have
def dans_balloons : ℝ := 29.0
def tims_balloons : ℝ := 4.142857143

-- Theorem: The ratio of Dan's balloons to Tim's balloons is 7
theorem dan_has_more_balloons : dans_balloons / tims_balloons = 7 := 
by
  sorry

end dan_has_more_balloons_l146_146006


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l146_146516

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l146_146516


namespace rain_is_random_event_l146_146905

def is_random_event (p : ℝ) : Prop := p > 0 ∧ p < 1

theorem rain_is_random_event (p : ℝ) (h : p = 0.75) : is_random_event p :=
by
  -- Here we will provide the necessary proof eventually.
  sorry

end rain_is_random_event_l146_146905


namespace loan_period_l146_146769

theorem loan_period (principal : ℝ) (rate_A rate_C gain_B : ℝ) (n : ℕ) 
  (h1 : principal = 3150)
  (h2 : rate_A = 0.08)
  (h3 : rate_C = 0.125)
  (h4 : gain_B = 283.5) :
  (gain_B = (rate_C * principal - rate_A * principal) * n) → n = 2 := by
  sorry

end loan_period_l146_146769


namespace find_x_satisfying_floor_eq_l146_146922

theorem find_x_satisfying_floor_eq (x : ℝ) (hx: ⌊x⌋ * x = 152) : x = 38 / 3 :=
sorry

end find_x_satisfying_floor_eq_l146_146922


namespace smallest_x_absolute_value_l146_146775

theorem smallest_x_absolute_value :
  ∃ x : ℝ, (|5 * x + 15| = 40) ∧ (∀ y : ℝ, |5 * y + 15| = 40 → x ≤ y) ∧ x = -11 :=
sorry

end smallest_x_absolute_value_l146_146775


namespace f_n_f_n_eq_n_l146_146382

def f : ℕ → ℕ := sorry
axiom f_def1 : f 1 = 1
axiom f_def2 : ∀ n ≥ 2, f n = n - f (f (n - 1))

theorem f_n_f_n_eq_n (n : ℕ) (hn : 0 < n) : f (n + f n) = n :=
by sorry

end f_n_f_n_eq_n_l146_146382


namespace sum_is_correct_l146_146067

def number : ℕ := 81
def added_number : ℕ := 15
def sum_value (x : ℕ) (y : ℕ) : ℕ := x + y

theorem sum_is_correct : sum_value number added_number = 96 := 
by 
  sorry

end sum_is_correct_l146_146067


namespace three_pow_2023_mod_eleven_l146_146340

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end three_pow_2023_mod_eleven_l146_146340


namespace triangular_difference_30_28_l146_146027

noncomputable def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_difference_30_28 : triangular 30 - triangular 28 = 59 :=
by
  sorry

end triangular_difference_30_28_l146_146027


namespace dave_deleted_17_apps_l146_146586

-- Define the initial and final state of Dave's apps
def initial_apps : Nat := 10
def added_apps : Nat := 11
def apps_left : Nat := 4

-- The total number of apps before deletion
def total_apps : Nat := initial_apps + added_apps

-- The expected number of deleted apps
def deleted_apps : Nat := total_apps - apps_left

-- The proof statement
theorem dave_deleted_17_apps : deleted_apps = 17 := by
  -- detailed steps are not required
  sorry

end dave_deleted_17_apps_l146_146586


namespace cans_purchased_l146_146157

theorem cans_purchased (S Q E : ℕ) (hQ : Q ≠ 0) :
  (∃ x : ℕ, x = (5 * S * E) / Q) := by
  sorry

end cans_purchased_l146_146157


namespace net_error_24x_l146_146233

theorem net_error_24x (x : ℕ) : 
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let error_pennies := (nickel_value - penny_value) * x
  let error_nickels := (dime_value - nickel_value) * x
  let error_dimes := (quarter_value - dime_value) * x
  let total_error := error_pennies + error_nickels + error_dimes
  total_error = 24 * x := 
by 
  sorry

end net_error_24x_l146_146233


namespace cleaner_for_cat_stain_l146_146316

theorem cleaner_for_cat_stain (c : ℕ) :
  (6 * 6) + (3 * c) + (1 * 1) = 49 → c = 4 :=
by
  sorry

end cleaner_for_cat_stain_l146_146316


namespace distinct_solution_count_l146_146001

theorem distinct_solution_count
  (n : ℕ)
  (x y : ℕ)
  (h1 : x ≠ y)
  (h2 : x ≠ 2 * y)
  (h3 : y ≠ 2 * x)
  (h4 : x^2 - x * y + y^2 = n) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 12 ∧ ∀ (a b : ℕ), (a, b) ∈ pairs → a^2 - a * b + b^2 = n :=
sorry

end distinct_solution_count_l146_146001


namespace largest_unattainable_sum_l146_146700

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end largest_unattainable_sum_l146_146700


namespace find_divisible_xy9z_l146_146061

-- Define a predicate for numbers divisible by 132
def divisible_by_132 (n : ℕ) : Prop :=
  n % 132 = 0

-- Define the given number form \(\overline{xy9z}\) as a number maker
def form_xy9z (x y z : ℕ) : ℕ :=
  1000 * x + 100 * y + 90 + z

-- Stating the theorem for finding all numbers of form \(\overline{xy9z}\) that are divisible by 132
theorem find_divisible_xy9z (x y z : ℕ) :
  (divisible_by_132 (form_xy9z x y z)) ↔
  form_xy9z x y z = 3696 ∨
  form_xy9z x y z = 4092 ∨
  form_xy9z x y z = 6996 ∨
  form_xy9z x y z = 7392 :=
by sorry

end find_divisible_xy9z_l146_146061


namespace three_number_relationship_l146_146020

theorem three_number_relationship :
  let a := (0.7 : ℝ) ^ 6
  let b := 6 ^ (0.7 : ℝ)
  let c := Real.log 6 / Real.log 0.7
  c < a ∧ a < b :=
sorry

end three_number_relationship_l146_146020


namespace probability_team_B_wins_third_game_l146_146269

theorem probability_team_B_wins_third_game :
  ∀ (A B : ℕ → Prop),
    (∀ n, A n ∨ B n) ∧ -- Each game is won by either A or B
    (∀ n, A n ↔ ¬ B n) ∧ -- No ties, outcomes are independent
    (A 0) ∧ -- Team A wins the first game
    (B 1) ∧ -- Team B wins the second game
    (∃ n1 n2 n3, A n1 ∧ A n2 ∧ A n3 ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3) -- Team A wins three games
    → (∃ S, ((A 0) ∧ (B 1) ∧ (B 2)) ↔ (S = 1/3)) := sorry

end probability_team_B_wins_third_game_l146_146269


namespace split_fraction_l146_146056

theorem split_fraction (n d a b x y : ℤ) (h_d : d = a * b) (h_ad : a.gcd b = 1) (h_frac : (n:ℚ) / (d:ℚ) = 58 / 77) (h_eq : 11 * x + 7 * y = 58) : 
  (58:ℚ) / 77 = (4:ℚ) / 7 + (2:ℚ) / 11 :=
by
  sorry

end split_fraction_l146_146056


namespace find_number_69_3_l146_146348

theorem find_number_69_3 (x : ℝ) (h : (x * 0.004) / 0.03 = 9.237333333333334) : x = 69.3 :=
by
  sorry

end find_number_69_3_l146_146348


namespace max_value_trig_expr_exists_angle_for_max_value_l146_146609

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end max_value_trig_expr_exists_angle_for_max_value_l146_146609


namespace friday_profit_l146_146342

noncomputable def total_weekly_profit : ℝ := 2000
noncomputable def profit_on_monday (total : ℝ) : ℝ := total / 3
noncomputable def profit_on_tuesday (total : ℝ) : ℝ := total / 4
noncomputable def profit_on_thursday (total : ℝ) : ℝ := 0.35 * total
noncomputable def profit_on_friday (total : ℝ) : ℝ :=
  total - (profit_on_monday total + profit_on_tuesday total + profit_on_thursday total)

theorem friday_profit (total : ℝ) : profit_on_friday total = 133.33 :=
by
  sorry

end friday_profit_l146_146342


namespace find_range_of_a_l146_146435

def prop_p (a : ℝ) : Prop :=
∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def prop_q (a : ℝ) : Prop :=
(∃ x₁ x₂ : ℝ, x₁ * x₂ = 1 ∧ x₁ + x₂ = -(a - 1) ∧ (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2))

def range_a (a : ℝ) : Prop :=
(-2 < a ∧ a <= -3/2) ∨ (-1 <= a ∧ a <= 2)

theorem find_range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a) ↔ range_a a :=
sorry

end find_range_of_a_l146_146435


namespace side_length_of_square_with_circles_l146_146827

noncomputable def side_length_of_square (radius : ℝ) : ℝ :=
  2 * radius + 2 * radius

theorem side_length_of_square_with_circles 
  (radius : ℝ) 
  (h_radius : radius = 2) 
  (h_tangent : ∀ (P Q : ℝ), P = Q + 2 * radius) :
  side_length_of_square radius = 8 :=
by
  sorry

end side_length_of_square_with_circles_l146_146827


namespace polynomial_range_open_interval_l146_146197

theorem polynomial_range_open_interval :
  ∀ (k : ℝ), k > 0 → ∃ (x y : ℝ), (1 - x * y)^2 + x^2 = k :=
by
  sorry

end polynomial_range_open_interval_l146_146197


namespace swimming_speed_in_still_water_l146_146952

theorem swimming_speed_in_still_water (v : ℝ) 
  (h_current_speed : 2 = 2) 
  (h_time_distance : 7 = 7) 
  (h_effective_speed : v - 2 = 14 / 7) : 
  v = 4 :=
sorry

end swimming_speed_in_still_water_l146_146952


namespace problem_statement_l146_146677

variable {f : ℝ → ℝ}

-- Condition 1: The function f satisfies (x - 1)f'(x) ≤ 0
def cond1 (f : ℝ → ℝ) : Prop := ∀ x, (x - 1) * (deriv f x) ≤ 0

-- Condition 2: The function f satisfies f(-x) = f(2 + x)
def cond2 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (2 + x)

theorem problem_statement (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h_cond1 : cond1 f)
  (h_cond2 : cond2 f)
  (h_dist : abs (x₁ - 1) < abs (x₂ - 1)) :
  f (2 - x₁) > f (2 - x₂) :=
sorry

end problem_statement_l146_146677


namespace cannot_cover_completely_with_dominoes_l146_146660

theorem cannot_cover_completely_with_dominoes :
  ¬ (∃ f : Fin 5 × Fin 3 → Fin 5 × Fin 3, 
      (∀ p q, f p = f q → p = q) ∧ 
      (∀ p, ∃ q, f q = p) ∧ 
      (∀ p, (f p).1 = p.1 + 1 ∨ (f p).2 = p.2 + 1)) := 
sorry

end cannot_cover_completely_with_dominoes_l146_146660


namespace part1_part2_part3_l146_146213

-- Part 1: There exists a real number a such that a + 1/a ≤ 2
theorem part1 : ∃ a : ℝ, a + 1/a ≤ 2 := sorry

-- Part 2: For all positive real numbers a and b, b/a + a/b ≥ 2
theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : b / a + a / b ≥ 2 := sorry

-- Part 3: For positive real numbers x and y such that x + 2y = 1, then 2/x + 1/y ≥ 8
theorem part3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 2 / x + 1 / y ≥ 8 := sorry

end part1_part2_part3_l146_146213


namespace hexagon_perimeter_is_24_l146_146182

-- Conditions given in the problem
def AB : ℝ := 3
def EF : ℝ := 3
def BE : ℝ := 4
def AF : ℝ := 4
def CD : ℝ := 5
def DF : ℝ := 5

-- Statement to show that the perimeter is 24 units
theorem hexagon_perimeter_is_24 :
  AB + BE + CD + DF + EF + AF = 24 :=
by
  sorry

end hexagon_perimeter_is_24_l146_146182


namespace no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l146_146377

theorem no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49 :
  ∀ n : ℕ, ¬ (∃ k : ℤ, (n^2 + 5 * n + 1) = 49 * k) :=
by
  sorry

end no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l146_146377


namespace common_chord_length_l146_146782

theorem common_chord_length (r : ℝ) (h : r = 12) 
  (condition : ∀ (C₁ C₂ : Set (ℝ × ℝ)), 
      ((C₁ = {p : ℝ × ℝ | dist p (0, 0) = r}) ∧ 
       (C₂ = {p : ℝ × ℝ | dist p (12, 0) = r}) ∧
       (C₂ ∩ C₁ ≠ ∅))) : 
  ∃ chord_len : ℝ, chord_len = 12 * Real.sqrt 3 :=
by
  sorry

end common_chord_length_l146_146782


namespace geom_seq_sixth_term_l146_146445

theorem geom_seq_sixth_term (a : ℝ) (r : ℝ) (h1: a * r^3 = 512) (h2: a * r^8 = 8) : 
  a * r^5 = 128 := 
by 
  sorry

end geom_seq_sixth_term_l146_146445


namespace Benjamin_has_45_presents_l146_146324

-- Define the number of presents each person has
def Ethan_presents : ℝ := 31.5
def Alissa_presents : ℝ := Ethan_presents + 22
def Benjamin_presents : ℝ := Alissa_presents - 8.5

-- The statement we need to prove
theorem Benjamin_has_45_presents : Benjamin_presents = 45 :=
by
  -- on the last line, we type sorry to skip the actual proof
  sorry

end Benjamin_has_45_presents_l146_146324


namespace M_subset_N_l146_146229

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l146_146229


namespace nancy_total_savings_l146_146225

noncomputable def total_savings : ℝ :=
  let cost_this_month := 9 * 5
  let cost_last_month := 8 * 4
  let cost_next_month := 7 * 6
  let discount_this_month := 0.20 * cost_this_month
  let discount_last_month := 0.20 * cost_last_month
  let discount_next_month := 0.20 * cost_next_month
  discount_this_month + discount_last_month + discount_next_month

theorem nancy_total_savings : total_savings = 23.80 :=
by
  sorry

end nancy_total_savings_l146_146225


namespace find_a_l146_146986

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x - Real.log x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x ≤ y → f x ≤ f y

theorem find_a (a : ℝ) :
  is_increasing_on (f a) (1 / 3) 2 → a ≥ 4 / 3 :=
sorry

end find_a_l146_146986


namespace cyclists_meeting_time_l146_146254

theorem cyclists_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 7 (Nat.lcm 12 9) ∧ t = 252 :=
by
  use 252
  have h1 : Nat.lcm 7 (Nat.lcm 12 9) = 252 := sorry
  exact ⟨rfl, h1⟩

end cyclists_meeting_time_l146_146254


namespace Alan_finish_time_third_task_l146_146657

theorem Alan_finish_time_third_task :
  let start_time := 480 -- 8:00 AM in minutes from midnight
  let finish_time_second_task := 675 -- 11:15 AM in minutes from midnight
  let total_tasks_time := 195 -- Total time spent on first two tasks
  let first_task_time := 65 -- Time taken for the first task calculated as per the solution
  let second_task_time := 130 -- Time taken for the second task calculated as per the solution
  let third_task_time := 65 -- Time taken for the third task
  let finish_time_third_task := 740 -- 12:20 PM in minutes from midnight
  start_time + total_tasks_time + third_task_time = finish_time_third_task :=
by
  -- proof here
  sorry

end Alan_finish_time_third_task_l146_146657


namespace find_f_neg_2_l146_146751

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 3 * x + 4 else 7 - 3 * x

theorem find_f_neg_2 : f (-2) = 13 := by
  sorry

end find_f_neg_2_l146_146751


namespace inverse_f_1_l146_146234

noncomputable def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

theorem inverse_f_1 : ∃ x : ℝ, f x = 1 ∧ x = 2 := by
sorry

end inverse_f_1_l146_146234


namespace Mongolian_Mathematical_Olympiad_54th_l146_146286

theorem Mongolian_Mathematical_Olympiad_54th {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^4 + b^4 + c^4 + (a^2 / (b + c)^2) + (b^2 / (c + a)^2) + (c^2 / (a + b)^2) ≥ a * b + b * c + c * a :=
sorry

end Mongolian_Mathematical_Olympiad_54th_l146_146286


namespace Ana_age_eight_l146_146612

theorem Ana_age_eight (A B n : ℕ) (h1 : A - 1 = 7 * (B - 1)) (h2 : A = 4 * B) (h3 : A - B = n) : A = 8 :=
by
  sorry

end Ana_age_eight_l146_146612


namespace sufficient_but_not_necessary_l146_146111

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 < x ∧ x < 2) : x < 2 ∧ ∀ y, (y < 2 → y ≤ 1 ∨ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l146_146111


namespace positive_root_gt_1008_l146_146745

noncomputable def P (x : ℝ) : ℝ := sorry
-- where P is a non-constant polynomial with integer coefficients bounded by 2015 in absolute value
-- Assume it has been properly defined according to the conditions in the problem statement

theorem positive_root_gt_1008 (x : ℝ) (hx : 0 < x) (hroot : P x = 0) : x > 1008 := 
sorry

end positive_root_gt_1008_l146_146745


namespace calculate_total_earnings_l146_146924

theorem calculate_total_earnings :
  let num_floors := 10
  let rooms_per_floor := 20
  let hours_per_room := 8
  let earnings_per_hour := 20
  let total_rooms := num_floors * rooms_per_floor
  let total_hours := total_rooms * hours_per_room
  let total_earnings := total_hours * earnings_per_hour
  total_earnings = 32000 := by sorry

end calculate_total_earnings_l146_146924


namespace find_x_squared_l146_146079

theorem find_x_squared :
  ∃ x : ℕ, (x^2 >= 2525 * 10^8) ∧ (x^2 < 2526 * 10^8) ∧ (x % 100 = 17 ∨ x % 100 = 33 ∨ x % 100 = 67 ∨ x % 100 = 83) ∧
    (x = 502517 ∨ x = 502533 ∨ x = 502567 ∨ x = 502583) :=
sorry

end find_x_squared_l146_146079


namespace expansion_coefficients_sum_l146_146724

theorem expansion_coefficients_sum : 
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), 
    (x - 2)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 → 
    a_0 + a_2 + a_4 = -122 := 
by 
  intros x a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  sorry

end expansion_coefficients_sum_l146_146724


namespace sqrt_expression_l146_146530

noncomputable def a : ℝ := 5 - 3 * Real.sqrt 2
noncomputable def b : ℝ := 5 + 3 * Real.sqrt 2

theorem sqrt_expression : 
  Real.sqrt (a^2) + Real.sqrt (b^2) + 2 = 12 :=
by
  sorry

end sqrt_expression_l146_146530


namespace pizza_slices_count_l146_146740

/-
  We ordered 21 pizzas. Each pizza has 8 slices. 
  Prove that the total number of slices of pizza is 168.
-/

theorem pizza_slices_count :
  (21 * 8) = 168 :=
by
  sorry

end pizza_slices_count_l146_146740


namespace john_moves_540kg_l146_146161

-- Conditions
def used_to_back_squat : ℝ := 200
def increased_by : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Definitions based on conditions
def current_back_squat : ℝ := used_to_back_squat + increased_by
def current_front_squat : ℝ := front_squat_ratio * current_back_squat
def one_triple : ℝ := triple_ratio * current_front_squat
def three_triples : ℝ := 3 * one_triple

-- The proof statement
theorem john_moves_540kg : three_triples = 540 := by
  sorry

end john_moves_540kg_l146_146161


namespace ending_number_of_range_divisible_by_five_l146_146179

theorem ending_number_of_range_divisible_by_five
  (first_number : ℕ)
  (number_of_terms : ℕ)
  (h_first : first_number = 15)
  (h_terms : number_of_terms = 10)
  : ∃ ending_number : ℕ, ending_number = first_number + 5 * (number_of_terms - 1) := 
by
  sorry

end ending_number_of_range_divisible_by_five_l146_146179


namespace electricity_usage_l146_146840

theorem electricity_usage 
  (total_usage : ℕ) (saved_cost : ℝ) (initial_cost : ℝ) (peak_cost : ℝ) (off_peak_cost : ℝ) 
  (usage_peak : ℕ) (usage_off_peak : ℕ) :
  total_usage = 100 →
  saved_cost = 3 →
  initial_cost = 0.55 →
  peak_cost = 0.6 →
  off_peak_cost = 0.4 →
  usage_peak + usage_off_peak = total_usage →
  (total_usage * initial_cost - (peak_cost * usage_peak + off_peak_cost * usage_off_peak) = saved_cost) →
  usage_peak = 60 ∧ usage_off_peak = 40 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end electricity_usage_l146_146840


namespace product_ineq_l146_146685

-- Define the relevant elements and conditions
variables (a b : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ)

-- Assumptions based on the conditions provided
variables (h₀ : a > 0) (h₁ : b > 0)
variables (h₂ : a + b = 1)
variables (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₃ > 0) (h₆ : x₄ > 0) (h₇ : x₅ > 0)
variables (h₈ : x₁ * x₂ * x₃ * x₄ * x₅ = 1)

-- The theorem statement to be proved
theorem product_ineq : (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 :=
sorry

end product_ineq_l146_146685


namespace police_catches_thief_in_two_hours_l146_146975

noncomputable def time_to_catch (speed_thief speed_police distance_police_start lead_time : ℝ) : ℝ :=
  let distance_thief := speed_thief * lead_time
  let initial_distance := distance_police_start - distance_thief
  let relative_speed := speed_police - speed_thief
  initial_distance / relative_speed

theorem police_catches_thief_in_two_hours :
  time_to_catch 20 40 60 1 = 2 := by
  sorry

end police_catches_thief_in_two_hours_l146_146975


namespace right_triangle_side_lengths_l146_146573

theorem right_triangle_side_lengths (x : ℝ) :
  (2 * x + 2)^2 + (x + 2)^2 = (x + 4)^2 ∨ (2 * x + 2)^2 + (x + 4)^2 = (x + 2)^2 ↔ (x = 1 ∨ x = 4) :=
by sorry

end right_triangle_side_lengths_l146_146573


namespace kiyana_gives_half_l146_146304

theorem kiyana_gives_half (total_grapes : ℕ) (h : total_grapes = 24) : 
  (total_grapes / 2) = 12 :=
by
  sorry

end kiyana_gives_half_l146_146304


namespace largest_cube_edge_from_cone_l146_146601

theorem largest_cube_edge_from_cone : 
  ∀ (s : ℝ), 
  (s = 2) → 
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 - 2 * Real.sqrt 3 :=
by
  sorry

end largest_cube_edge_from_cone_l146_146601


namespace total_doors_needed_correct_l146_146109

-- Define the conditions
def buildings : ℕ := 2
def floors_per_building : ℕ := 12
def apartments_per_floor : ℕ := 6
def doors_per_apartment : ℕ := 7

-- Define the total number of doors needed
def total_doors_needed : ℕ := buildings * floors_per_building * apartments_per_floor * doors_per_apartment

-- State the theorem to prove the total number of doors needed is 1008
theorem total_doors_needed_correct : total_doors_needed = 1008 := by
  sorry

end total_doors_needed_correct_l146_146109


namespace ana_wins_l146_146833

-- Define the game conditions and state
def game_conditions (n : ℕ) (m : ℕ) : Prop :=
  n < m ∧ m < n^2 ∧ Nat.gcd n m = 1

-- Define the losing condition
def losing_condition (n : ℕ) : Prop :=
  n >= 2016

-- Define the predicate for Ana having a winning strategy
def ana_winning_strategy : Prop :=
  ∃ (strategy : ℕ → ℕ), strategy 3 = 5 ∧
  (∀ n, (¬ losing_condition n) → (losing_condition (strategy n)))

theorem ana_wins : ana_winning_strategy :=
  sorry

end ana_wins_l146_146833


namespace John_total_weekly_consumption_l146_146767

/-
  Prove that John's total weekly consumption of water, milk, and juice in quarts is 49.25 quarts, 
  given the specified conditions on his daily and periodic consumption.
-/

def John_consumption_problem (gallons_per_day : ℝ) (pints_every_other_day : ℝ) (ounces_every_third_day : ℝ) 
  (quarts_per_gallon : ℝ) (quarts_per_pint : ℝ) (quarts_per_ounce : ℝ) : ℝ :=
  let water_per_day := gallons_per_day * quarts_per_gallon
  let water_per_week := water_per_day * 7
  let milk_per_other_day := pints_every_other_day * quarts_per_pint
  let milk_per_week := milk_per_other_day * 4 -- assuming he drinks milk 4 times a week
  let juice_per_third_day := ounces_every_third_day * quarts_per_ounce
  let juice_per_week := juice_per_third_day * 2 -- assuming he drinks juice 2 times a week
  water_per_week + milk_per_week + juice_per_week

theorem John_total_weekly_consumption :
  John_consumption_problem 1.5 3 20 4 (1/2) (1/32) = 49.25 :=
by
  sorry

end John_total_weekly_consumption_l146_146767


namespace problem1_problem2_l146_146863

-- Problem 1: Prove the solution set of the given inequality
theorem problem1 (x : ℝ) : (|x - 2| + 2 * |x - 1| > 5) ↔ (x < -1/3 ∨ x > 3) := 
sorry

-- Problem 2: Prove the range of values for 'a' such that the inequality holds
theorem problem2 (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ |a - 2|) ↔ (a ≤ 3/2) :=
sorry

end problem1_problem2_l146_146863


namespace total_hours_worked_l146_146437

variable (A B C D E T : ℝ)

theorem total_hours_worked (hA : A = 12)
  (hB : B = 1 / 3 * A)
  (hC : C = 2 * B)
  (hD : D = 1 / 2 * E)
  (hE : E = A + 3)
  (hT : T = A + B + C + D + E) : T = 46.5 :=
by
  sorry

end total_hours_worked_l146_146437


namespace valid_triples_l146_146945

theorem valid_triples :
  ∀ (a b c : ℕ), 1 ≤ a → 1 ≤ b → 1 ≤ c →
  (∃ k : ℕ, 32 * a + 3 * b + 48 * c = 4 * k * a * b * c) ↔ 
  (a = 1 ∧ b = 20 ∧ c = 1) ∨ (a = 1 ∧ b = 4 ∧ c = 1) ∨ (a = 3 ∧ b = 4 ∧ c = 1) := 
by
  sorry

end valid_triples_l146_146945


namespace probability_prime_ball_l146_146383

open Finset

theorem probability_prime_ball :
  let balls := {1, 2, 3, 4, 5, 6, 8, 9}
  let total := card balls
  let primes := {2, 3, 5}
  let primes_count := card primes
  (total = 8) → (primes ⊆ balls) → 
  primes_count = 3 → 
  primes_count / total = 3 / 8 :=
by
  intros
  sorry

end probability_prime_ball_l146_146383


namespace average_people_per_hour_l146_146419

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end average_people_per_hour_l146_146419


namespace solve_inequality_l146_146095

theorem solve_inequality (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

end solve_inequality_l146_146095


namespace zero_and_one_positions_l146_146148

theorem zero_and_one_positions (a : ℝ) :
    (0 = (a + (-a)) / 2) ∧ (1 = ((a + (-a)) / 2 + 1)) :=
by
  sorry

end zero_and_one_positions_l146_146148


namespace Jackson_game_time_l146_146846

/-- Jackson's grade increases by 15 points for every hour he spends studying, 
    and his grade is 45 points, prove that he spends 9 hours playing video 
    games when he spends 3 hours studying and 1/3 of his study time on 
    playing video games. -/
theorem Jackson_game_time (S G : ℕ) (h1 : 15 * S = 45) (h2 : G = 3 * S) : G = 9 :=
by
  sorry

end Jackson_game_time_l146_146846


namespace sum_of_coefficients_correct_l146_146150

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (x + 3 * y) ^ 17

-- Define the sum of coefficients by substituting x = 1 and y = 1
def sum_of_coefficients : ℤ := polynomial 1 1

-- Statement of the mathematical proof problem
theorem sum_of_coefficients_correct :
  sum_of_coefficients = 17179869184 :=
by
  -- proof will be provided here
  sorry

end sum_of_coefficients_correct_l146_146150


namespace solve_for_z_l146_146215

theorem solve_for_z :
  ∃ z : ℤ, (∀ x y : ℤ, x = 11 → y = 8 → 2 * x + 3 * z = 5 * y) → z = 6 :=
by
  sorry

end solve_for_z_l146_146215


namespace tom_weekly_fluid_intake_l146_146987

-- Definitions based on the conditions.
def soda_cans_per_day : ℕ := 5
def ounces_per_can : ℕ := 12
def water_ounces_per_day : ℕ := 64
def days_per_week : ℕ := 7

-- The mathematical proof problem statement.
theorem tom_weekly_fluid_intake :
  (soda_cans_per_day * ounces_per_can + water_ounces_per_day) * days_per_week = 868 := 
by
  sorry

end tom_weekly_fluid_intake_l146_146987


namespace books_total_l146_146851

theorem books_total (J T : ℕ) (hJ : J = 10) (hT : T = 38) : J + T = 48 :=
by {
  sorry
}

end books_total_l146_146851


namespace opposite_of_neg_five_l146_146368

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end opposite_of_neg_five_l146_146368


namespace C_share_of_rent_l146_146636

-- Define the given conditions
def A_ox_months : ℕ := 10 * 7
def B_ox_months : ℕ := 12 * 5
def C_ox_months : ℕ := 15 * 3
def total_rent : ℕ := 175
def total_ox_months : ℕ := A_ox_months + B_ox_months + C_ox_months
def cost_per_ox_month := total_rent / total_ox_months

-- The goal is to prove that C's share of the rent is Rs. 45
theorem C_share_of_rent : C_ox_months * cost_per_ox_month = 45 := by
  -- Adding sorry to skip the proof
  sorry

end C_share_of_rent_l146_146636


namespace part_a_l146_146572

theorem part_a (x : ℝ) : (6 - x) / x = 3 / 6 → x = 4 := by
  sorry

end part_a_l146_146572


namespace combined_work_time_l146_146646

-- Define the time taken by Paul and Rose to complete the work individually
def paul_days : ℕ := 80
def rose_days : ℕ := 120

-- Define the work rates of Paul and Rose
def paul_rate := 1 / (paul_days : ℚ)
def rose_rate := 1 / (rose_days : ℚ)

-- Define the combined work rate
def combined_rate := paul_rate + rose_rate

-- Statement to prove: Together they can complete the work in 48 days.
theorem combined_work_time : combined_rate = 1 / 48 := by 
  sorry

end combined_work_time_l146_146646


namespace gcd_le_two_l146_146405

theorem gcd_le_two (a m n : ℕ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : Odd n) :
  Nat.gcd (a^n - 1) (a^m + 1) ≤ 2 := 
sorry

end gcd_le_two_l146_146405


namespace schedule_courses_l146_146508

/-- Definition of valid schedule count where at most one pair of courses is consecutive. -/
def count_valid_schedules : ℕ := 180

/-- Given 7 periods and 3 courses, determine the number of valid schedules 
    where at most one pair of these courses is consecutive. -/
theorem schedule_courses (periods : ℕ) (courses : ℕ) (valid_schedules : ℕ) :
  periods = 7 → courses = 3 → valid_schedules = count_valid_schedules →
  valid_schedules = 180 :=
by
  intros h1 h2 h3
  sorry

end schedule_courses_l146_146508


namespace rope_cut_number_not_8_l146_146488

theorem rope_cut_number_not_8 (l : ℝ) (h1 : (1 : ℝ) % l = 0) (h2 : (2 : ℝ) % l = 0) (h3 : (3 / l) ≠ 8) : False :=
by
  sorry

end rope_cut_number_not_8_l146_146488


namespace max_partial_sum_l146_146189

variable (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence and the conditions given
def arithmetic_sequence (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a_n n = a_1 + n * d

def condition1 (a_1 : ℤ) : Prop := a_1 > 0

def condition2 (a_n : ℕ → ℤ) (d : ℤ) : Prop := 3 * (a_n 8) = 5 * (a_n 13)

-- Define the partial sum of the arithmetic sequence
def partial_sum (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

-- Define the main problem: Prove that S_20 is the greatest
theorem max_partial_sum (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) (S : ℕ → ℤ) :
  arithmetic_sequence a_n a_1 d →
  condition1 a_1 →
  condition2 a_n d →
  partial_sum S a_n →
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → S 20 ≥ S n := by
  sorry

end max_partial_sum_l146_146189


namespace common_point_arithmetic_progression_l146_146908

theorem common_point_arithmetic_progression (a b c : ℝ) (h : 2 * b = a + c) :
  ∃ (x y : ℝ), (∀ x, y = a * x^2 + b * x + c) ∧ x = -2 ∧ y = 0 :=
by
  sorry

end common_point_arithmetic_progression_l146_146908


namespace find_x_l146_146951

-- Definitions based on conditions
variables (A B C M O : Type)
variables (OA OB OC OM : vector_space O)
variables (x : ℚ) -- Rational number type for x

-- Condition (1): M lies in the plane ABC
-- Condition (2): OM = x * OA + 1/3 * OB + 1/2 * OC
axiom H : OM = x • OA + (1 / 3 : ℚ) • OB + (1 / 2 : ℚ) • OC

-- The theorem statement
theorem find_x :
  x = 1 / 6 :=
sorry -- Proof is to be provided

end find_x_l146_146951


namespace middle_part_is_28_4_over_11_l146_146372

theorem middle_part_is_28_4_over_11 (x : ℚ) :
  let part1 := x
  let part2 := (1/2) * x
  let part3 := (1/3) * x
  part1 + part2 + part3 = 104
  ∧ part2 = 28 + 4/11 := by
  sorry

end middle_part_is_28_4_over_11_l146_146372


namespace equation_is_hyperbola_l146_146916

theorem equation_is_hyperbola : 
  ∀ x y : ℝ, (x^2 - 25*y^2 - 10*x + 50 = 0) → 
  (∃ a b h k : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (x - h)^2 / a^2 - (y - k)^2 / b^2 = -1)) :=
by
  sorry

end equation_is_hyperbola_l146_146916


namespace min_value_fraction_l146_146352

theorem min_value_fraction {x : ℝ} (h : x > 8) : 
    ∃ c : ℝ, (∀ y : ℝ, y = (x^2) / ((x - 8)^2) → c ≤ y) ∧ c = 1 := 
sorry

end min_value_fraction_l146_146352


namespace question_solution_l146_146258

theorem question_solution
  (f : ℝ → ℝ)
  (h_decreasing : ∀ ⦃x y : ℝ⦄, -3 < x ∧ x < 0 → -3 < y ∧ y < 0 → x < y → f y < f x)
  (h_symmetry : ∀ x : ℝ, f (x) = f (-x + 6)) :
  f (-5) < f (-3/2) ∧ f (-3/2) < f (-7/2) :=
sorry

end question_solution_l146_146258


namespace pyramid_volume_l146_146452

theorem pyramid_volume (a : ℝ) (h : a > 0) : (1 / 6) * a^3 = 1 / 6 * a^3 :=
by
  sorry

end pyramid_volume_l146_146452


namespace other_solution_of_quadratic_l146_146735

theorem other_solution_of_quadratic (x : ℚ) (h1 : x = 3 / 8) 
  (h2 : 72 * x^2 + 37 = -95 * x + 12) : ∃ y : ℚ, y ≠ 3 / 8 ∧ 72 * y^2 + 95 * y + 25 = 0 ∧ y = 5 / 8 :=
by
  sorry

end other_solution_of_quadratic_l146_146735


namespace remainder_division_1000_l146_146449

theorem remainder_division_1000 (x : ℕ) (hx : x > 0) (h : 100 % x = 10) : 1000 % x = 10 :=
  sorry

end remainder_division_1000_l146_146449


namespace expansion_simplification_l146_146019

variable (x y : ℝ)

theorem expansion_simplification :
  let a := 3 * x + 4
  let b := 2 * x + 6 * y + 7
  a * b = 6 * x ^ 2 + 18 * x * y + 29 * x + 24 * y + 28 :=
by
  sorry

end expansion_simplification_l146_146019


namespace sum_of_repeating_decimals_l146_146266

-- Definitions based on the conditions
def x := 0.6666666666666666 -- Lean may not directly support \(0.\overline{6}\) notation
def y := 0.7777777777777777 -- Lean may not directly support \(0.\overline{7}\) notation

-- Translate those to the correct fractional forms
def x_as_fraction := (2 : ℚ) / 3
def y_as_fraction := (7 : ℚ) / 9

-- The main statement to prove
theorem sum_of_repeating_decimals : x_as_fraction + y_as_fraction = 13 / 9 :=
by
  -- Proof skipped
  sorry

end sum_of_repeating_decimals_l146_146266


namespace mother_l146_146541

def age_relations (P M : ℕ) : Prop :=
  P = (2 * M) / 5 ∧ P + 10 = (M + 10) / 2

theorem mother's_present_age (P M : ℕ) (h : age_relations P M) : M = 50 :=
by
  sorry

end mother_l146_146541


namespace solve_system1_solve_system2_l146_146605

theorem solve_system1 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : x = 2 * y + 1) : x = 3 ∧ y = 1 := 
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x - y = 6) (h2 : 3 * x + 2 * y = 2) : x = 2 ∧ y = -2 := 
by sorry

end solve_system1_solve_system2_l146_146605


namespace q_investment_time_l146_146569

-- Definitions from the conditions
def investment_ratio_p_q : ℚ := 7 / 5
def profit_ratio_p_q : ℚ := 7 / 13
def time_p : ℕ := 5

-- Problem statement
theorem q_investment_time
  (investment_ratio_p_q : ℚ)
  (profit_ratio_p_q : ℚ)
  (time_p : ℕ)
  (hpq_inv : investment_ratio_p_q = 7 / 5)
  (hpq_profit : profit_ratio_p_q = 7 / 13)
  (ht_p : time_p = 5) : 
  ∃ t_q : ℕ, 35 * t_q = 455 :=
sorry

end q_investment_time_l146_146569


namespace problems_per_worksheet_l146_146553

theorem problems_per_worksheet (P : ℕ) (graded : ℕ) (remaining : ℕ) (total_worksheets : ℕ) (total_problems_remaining : ℕ) :
    graded = 5 →
    total_worksheets = 9 →
    total_problems_remaining = 16 →
    remaining = total_worksheets - graded →
    4 * P = total_problems_remaining →
    P = 4 :=
by
  intros h_graded h_worksheets h_problems h_remaining h_equation
  sorry

end problems_per_worksheet_l146_146553


namespace cody_final_tickets_l146_146517

def initial_tickets : ℝ := 56.5
def lost_tickets : ℝ := 6.3
def spent_tickets : ℝ := 25.75
def won_tickets : ℝ := 10.25
def dropped_tickets : ℝ := 3.1

theorem cody_final_tickets : 
  initial_tickets - lost_tickets - spent_tickets + won_tickets - dropped_tickets = 31.6 :=
by
  sorry

end cody_final_tickets_l146_146517


namespace longest_side_of_similar_triangle_l146_146834

-- Define the sides of the original triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 12

-- Define the perimeter of the similar triangle
def perimeter_similar_triangle : ℕ := 150

-- Formalize the problem using Lean statement
theorem longest_side_of_similar_triangle :
  ∃ x : ℕ, 8 * x + 10 * x + 12 * x = 150 ∧ 12 * x = 60 :=
by
  sorry

end longest_side_of_similar_triangle_l146_146834


namespace average_of_tenths_and_thousandths_l146_146463

theorem average_of_tenths_and_thousandths :
  (0.4 + 0.005) / 2 = 0.2025 :=
by
  -- We skip the proof here
  sorry

end average_of_tenths_and_thousandths_l146_146463


namespace combined_moles_l146_146415

def balanced_reaction (NaHCO3 HC2H3O2 H2O : ℕ) : Prop :=
  NaHCO3 + HC2H3O2 = H2O

theorem combined_moles (NaHCO3 HC2H3O2 : ℕ) 
  (h : balanced_reaction NaHCO3 HC2H3O2 3) : 
  NaHCO3 + HC2H3O2 = 6 :=
sorry

end combined_moles_l146_146415


namespace find_a10_l146_146482

variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (h_pos : ∀ (n : ℕ), 0 < a n)
variable (h_mul : ∀ (p q : ℕ), a (p + q) = a p * a q)
variable (h_a8 : a 8 = 16)

theorem find_a10 : a 10 = 32 :=
by
  sorry

end find_a10_l146_146482


namespace union_of_A_and_B_l146_146929

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := sorry

end union_of_A_and_B_l146_146929


namespace volume_of_truncated_cone_l146_146447

noncomputable def surface_area_top : ℝ := 3 * Real.pi
noncomputable def surface_area_bottom : ℝ := 12 * Real.pi
noncomputable def slant_height : ℝ := 2
noncomputable def volume_cone : ℝ := 7 * Real.pi

theorem volume_of_truncated_cone :
  ∃ V : ℝ, V = volume_cone :=
sorry

end volume_of_truncated_cone_l146_146447


namespace xiaohui_pe_score_l146_146732

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end xiaohui_pe_score_l146_146732


namespace graph_of_equation_l146_146059

theorem graph_of_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := 
by
  sorry

end graph_of_equation_l146_146059


namespace option_A_is_quadratic_l146_146631

def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

-- Given options
def option_A_equation (x : ℝ) : Prop :=
  x^2 - 2 = 0

def option_B_equation (x y : ℝ) : Prop :=
  x + 2 * y = 3

def option_C_equation (x : ℝ) : Prop :=
  x - 1/x = 1

def option_D_equation (x y : ℝ) : Prop :=
  x^2 + x = y + 1

-- Prove that option A is a quadratic equation
theorem option_A_is_quadratic (x : ℝ) : is_quadratic_equation 1 0 (-2) :=
by
  sorry

end option_A_is_quadratic_l146_146631


namespace partnership_investment_l146_146221

theorem partnership_investment
  (a_investment : ℕ := 30000)
  (b_investment : ℕ)
  (c_investment : ℕ := 50000)
  (c_profit_share : ℕ := 36000)
  (total_profit : ℕ := 90000)
  (total_investment := a_investment + b_investment + c_investment)
  (c_defined_share : ℚ := 2/5)
  (profit_proportionality : (c_profit_share : ℚ) / total_profit = (c_investment : ℚ) / total_investment) :
  b_investment = 45000 :=
by
  sorry

end partnership_investment_l146_146221


namespace school_xx_percentage_increase_l146_146200

theorem school_xx_percentage_increase
  (X Y : ℕ) -- denote the number of students at school XX and YY last year
  (H_Y : Y = 2400) -- condition: school YY had 2400 students last year
  (H_total : X + Y = 4000) -- condition: total number of students last year was 4000
  (H_increase_YY : YY_increase = (3 * Y) / 100) -- condition: 3 percent increase at school YY
  (H_difference : XX_increase = YY_increase + 40) -- condition: school XX grew by 40 more students than YY
  : (XX_increase * 100) / X = 7 :=
by
  sorry

end school_xx_percentage_increase_l146_146200


namespace find_30_cent_items_l146_146547

-- Define the parameters and their constraints
variables (a d b c : ℕ)

-- Define the conditions
def total_items : Prop := a + d + b + c = 50
def total_cost : Prop := 30 * a + 150 * d + 200 * b + 300 * c = 6000

-- The theorem to prove the number of 30-cent items purchased
theorem find_30_cent_items (h1 : total_items a d b c) (h2 : total_cost a d b c) : 
  ∃ a, a + d + b + c = 50 ∧ 30 * a + 150 * d + 200 * b + 300 * c = 6000 := 
sorry

end find_30_cent_items_l146_146547


namespace parallel_lines_slope_l146_146120

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 2 = 0 → ax * x - 8 * y - 3 = 0 → a = -6) :=
by
  sorry

end parallel_lines_slope_l146_146120


namespace casper_candy_problem_l146_146392

theorem casper_candy_problem (o y gr : ℕ) (n : ℕ) (h1 : 10 * o = 16 * y) (h2 : 16 * y = 18 * gr) (h3 : 18 * gr = 18 * n) :
    n = 40 :=
by
  sorry

end casper_candy_problem_l146_146392


namespace choose_president_and_secretary_l146_146579

theorem choose_president_and_secretary (total_members boys girls : ℕ) (h_total : total_members = 30) (h_boys : boys = 18) (h_girls : girls = 12) : 
  (boys * girls = 216) :=
by
  sorry

end choose_president_and_secretary_l146_146579


namespace silk_original_amount_l146_146738

theorem silk_original_amount (s r : ℕ) (l d x : ℚ)
  (h1 : s = 30)
  (h2 : r = 3)
  (h3 : d = 12)
  (h4 : 30 - 3 = 27)
  (h5 : x / 12 = 30 / 27):
  x = 40 / 3 :=
by
  sorry

end silk_original_amount_l146_146738


namespace rectangle_area_and_perimeter_l146_146725

-- Given conditions as definitions
def length : ℕ := 5
def width : ℕ := 3

-- Proof problems
theorem rectangle_area_and_perimeter :
  (length * width = 15) ∧ (2 * (length + width) = 16) :=
by
  sorry

end rectangle_area_and_perimeter_l146_146725


namespace modulo_inverse_product_l146_146630

open Int 

theorem modulo_inverse_product (n : ℕ) (a b c : ℤ) 
  (hn : 0 < n) 
  (ha : a * a.gcd n = 1) 
  (hb : b * b.gcd n = 1) 
  (hc : c * c.gcd n = 1) 
  (hab : (a * b) % n = 1) 
  (hac : (c * a) % n = 1) : 
  ((a * b) * c) % n = c % n :=
by
  sorry

end modulo_inverse_product_l146_146630


namespace largest_d_l146_146276

theorem largest_d (a b c d : ℝ) (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := sorry

end largest_d_l146_146276


namespace find_radius_l146_146839

-- Definitions and conditions
variables (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 25)

-- Theorem statement
theorem find_radius : r = 50 :=
sorry

end find_radius_l146_146839


namespace number_subtract_four_l146_146701

theorem number_subtract_four (x : ℤ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end number_subtract_four_l146_146701


namespace necessarily_positive_l146_146588

-- Definitions based on given conditions
variables {x y z : ℝ}

-- Stating the problem
theorem necessarily_positive : (0 < x ∧ x < 1) → (-2 < y ∧ y < 0) → (0 < z ∧ z < 1) → (x + y^2 > 0) :=
by
  intros hx hy hz
  sorry

end necessarily_positive_l146_146588


namespace triangle_problem_l146_146358

noncomputable def length_of_side_c (a : ℝ) (cosB : ℝ) (C : ℝ) : ℝ :=
  a * (Real.sqrt 2 / 2) / (Real.sqrt (1 - cosB^2))

noncomputable def cos_A_minus_pi_over_6 (cosB : ℝ) (cosA : ℝ) (sinA : ℝ) : ℝ :=
  cosA * (Real.sqrt 3 / 2) + sinA * (1 / 2)

theorem triangle_problem (a : ℝ) (cosB : ℝ) (C : ℝ) 
  (ha : a = 6) (hcosB : cosB = 4/5) (hC : C = Real.pi / 4) : 
  (length_of_side_c a cosB C = 5 * Real.sqrt 2) ∧ 
  (cos_A_minus_pi_over_6 cosB (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2)))) (Real.sqrt (1 - (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2))))^2)) = (7 * Real.sqrt 2 - Real.sqrt 6) / 20) :=
by 
  sorry

end triangle_problem_l146_146358


namespace simplify_fraction_l146_146008

theorem simplify_fraction : 
    (3 ^ 1011 + 3 ^ 1009) / (3 ^ 1011 - 3 ^ 1009) = 5 / 4 := 
by
  sorry

end simplify_fraction_l146_146008


namespace yellow_dandelions_day_before_yesterday_l146_146484

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yellow_yesterday white_yesterday yellow_today white_today : ℕ),
    yellow_yesterday = 20 →
    white_yesterday = 14 →
    yellow_today = 15 →
    white_today = 11 →
    ∃ yellow_day_before_yesterday : ℕ,
      yellow_day_before_yesterday = white_yesterday + white_today :=
by sorry

end yellow_dandelions_day_before_yesterday_l146_146484


namespace some_zen_not_cen_l146_146965

variable {Zen Ben Cen : Type}
variables (P Q R : Zen → Prop)

theorem some_zen_not_cen (h1 : ∀ x, P x → Q x)
                        (h2 : ∃ x, Q x ∧ ¬ (R x)) :
  ∃ x, P x ∧ ¬ (R x) :=
  sorry

end some_zen_not_cen_l146_146965


namespace zoo_animal_difference_l146_146414

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := 1 / 2 * (parrots + snakes)
  let zebras := elephants - 3
  monkeys - zebras = 35 :=
by
  sorry

end zoo_animal_difference_l146_146414


namespace segments_interior_proof_l146_146418

noncomputable def count_internal_segments (squares hexagons octagons : Nat) : Nat := 
  let vertices := (squares * 4 + hexagons * 6 + octagons * 8) / 3
  let total_segments := (vertices * (vertices - 1)) / 2
  let edges_along_faces := 3 * vertices
  (total_segments - edges_along_faces) / 2

theorem segments_interior_proof : count_internal_segments 12 8 6 = 840 := 
  by sorry

end segments_interior_proof_l146_146418


namespace swimming_pool_length_l146_146919

noncomputable def solveSwimmingPoolLength : ℕ :=
  let w_pool := 22
  let w_deck := 3
  let total_area := 728
  let total_width := w_pool + 2 * w_deck
  let L := (total_area / total_width) - 2 * w_deck
  L

theorem swimming_pool_length : solveSwimmingPoolLength = 20 := 
  by
  -- Proof goes here
  sorry

end swimming_pool_length_l146_146919


namespace inverse_of_2_is_46_l146_146310

-- Given the function f(x) = 5x^3 + 6
def f (x : ℝ) : ℝ := 5 * x^3 + 6

-- Prove the statement
theorem inverse_of_2_is_46 : (∃ y, f y = x) ∧ f (2 : ℝ) = 46 → x = 46 :=
by
  sorry

end inverse_of_2_is_46_l146_146310


namespace find_principal_amount_l146_146727

theorem find_principal_amount 
  (total_interest : ℝ)
  (rate1 rate2 : ℝ)
  (years1 years2 : ℕ)
  (P : ℝ)
  (A1 A2 : ℝ) 
  (hA1 : A1 = P * (1 + rate1/100)^years1)
  (hA2 : A2 = A1 * (1 + rate2/100)^years2)
  (hInterest : A2 = P + total_interest) : 
  P = 25252.57 :=
by
  -- Given the conditions above, we prove the main statement.
  sorry

end find_principal_amount_l146_146727


namespace find_numbers_l146_146891

theorem find_numbers (a b : ℝ) (h₁ : a - b = 157) (h₂ : a / b = 2) : a = 314 ∧ b = 157 :=
sorry

end find_numbers_l146_146891


namespace problem_l146_146814

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def m : ℝ := sorry
noncomputable def p : ℝ := sorry
noncomputable def r : ℝ := sorry

theorem problem
  (h1 : a^2 - m*a + 3 = 0)
  (h2 : b^2 - m*b + 3 = 0)
  (h3 : a * b = 3)
  (h4 : ∀ x, x^2 - p * x + r = (x - (a + 1 / b)) * (x - (b + 1 / a))) :
  r = 16 / 3 :=
sorry

end problem_l146_146814


namespace fifth_boy_pays_l146_146938

def problem_conditions (a b c d e : ℝ) : Prop :=
  d = 20 ∧
  a = (1 / 3) * (b + c + d + e) ∧
  b = (1 / 4) * (a + c + d + e) ∧
  c = (1 / 5) * (a + b + d + e) ∧
  a + b + c + d + e = 120 

theorem fifth_boy_pays (a b c d e : ℝ) (h : problem_conditions a b c d e) : 
  e = 35 :=
sorry

end fifth_boy_pays_l146_146938


namespace gas_mixture_pressure_l146_146783

theorem gas_mixture_pressure
  (m : ℝ) -- mass of each gas
  (p : ℝ) -- initial pressure
  (T : ℝ) -- initial temperature
  (V : ℝ) -- volume of the container
  (R : ℝ) -- ideal gas constant
  (mu_He : ℝ := 4) -- molar mass of helium
  (mu_N2 : ℝ := 28) -- molar mass of nitrogen
  (is_ideal : True) -- assumption that the gases are ideal
  (temp_doubled : True) -- assumption that absolute temperature is doubled
  (N2_dissociates : True) -- assumption that nitrogen dissociates into atoms
  : (9 / 4) * p = p' :=
by
  sorry

end gas_mixture_pressure_l146_146783


namespace weighted_average_of_angles_l146_146976

def triangle_inequality (a b c α β γ : ℝ) : Prop :=
  (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0

noncomputable def angle_sum (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

theorem weighted_average_of_angles (a b c α β γ : ℝ)
  (h1 : triangle_inequality a b c α β γ)
  (h2 : angle_sum α β γ) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 :=
by
  sorry

end weighted_average_of_angles_l146_146976


namespace max_length_polyline_l146_146912

-- Definition of the grid and problem
def grid_rows : ℕ := 6
def grid_cols : ℕ := 10

-- The maximum length of a closed, non-self-intersecting polyline
theorem max_length_polyline (rows cols : ℕ) 
  (h_rows : rows = grid_rows) (h_cols : cols = grid_cols) :
  ∃ length : ℕ, length = 76 :=
by {
  sorry
}

end max_length_polyline_l146_146912


namespace quadratic_root_range_l146_146546

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range (a : ℝ) (h : a ≠ 0) (h_distinct_roots : ∃ x1 x2 : ℝ, quadratic_function a x1 = 0 ∧ quadratic_function a x2 = 0 ∧ x1 ≠ x2 ∧ x1 < 1 ∧ x2 > 1) :
    -(2 / 11) < a ∧ a < 0 :=
sorry

end quadratic_root_range_l146_146546


namespace max_value_of_a_l146_146355

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end max_value_of_a_l146_146355


namespace mens_wages_l146_146050

-- Definitions from the conditions.
variables (men women boys total_earnings : ℕ) (wage : ℚ)
variable (equivalence : 5 * men = 8 * boys)
variable (totalEarnings : total_earnings = 120)

-- The final statement to prove the men's wages.
theorem mens_wages (h_eq : 5 = 5) : wage = 46.15 :=
by
  sorry

end mens_wages_l146_146050


namespace find_x_solution_l146_146950

theorem find_x_solution (x b c : ℝ) (h_eq : x^2 + c^2 = (b - x)^2):
  x = (b^2 - c^2) / (2 * b) :=
sorry

end find_x_solution_l146_146950


namespace fraction_ordering_l146_146607

theorem fraction_ordering :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  (b < c) ∧ (c < a) :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  sorry

end fraction_ordering_l146_146607


namespace percentage_increase_l146_146515

theorem percentage_increase 
  (P : ℝ)
  (bought_price : ℝ := 0.80 * P) 
  (original_profit : ℝ := 0.3600000000000001 * P) :
  ∃ X : ℝ, X = 70.00000000000002 ∧ (1.3600000000000001 * P = bought_price * (1 + X / 100)) :=
sorry

end percentage_increase_l146_146515


namespace percentage_deducted_from_list_price_l146_146960

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 65.97
noncomputable def selling_price : ℝ := 65.97
noncomputable def required_profit_percent : ℝ := 25

theorem percentage_deducted_from_list_price :
  let desired_selling_price := cost_price * (1 + required_profit_percent / 100)
  let discount_percentage := 100 * (1 - desired_selling_price / list_price)
  discount_percentage = 10.02 :=
by
  sorry

end percentage_deducted_from_list_price_l146_146960


namespace simplify_fraction_l146_146676

theorem simplify_fraction (x y z : ℝ) (h : x + 2 * y + z ≠ 0) :
  (x^2 + y^2 - 4 * z^2 + 2 * x * y) / (x^2 + 4 * y^2 - z^2 + 2 * x * z) = (x + y - 2 * z) / (x + z - 2 * y) :=
by
  sorry

end simplify_fraction_l146_146676


namespace kevin_ends_with_cards_l146_146680

def cards_found : ℝ := 47.0
def cards_lost : ℝ := 7.0

theorem kevin_ends_with_cards : cards_found - cards_lost = 40.0 := by
  sorry

end kevin_ends_with_cards_l146_146680


namespace min_ab_l146_146683

theorem min_ab {a b : ℝ} (h1 : (a^2) * (-b) + (a^2 + 1) = 0) : |a * b| = 2 :=
sorry

end min_ab_l146_146683


namespace track_length_l146_146187

theorem track_length (x : ℝ) (b_speed s_speed : ℝ) (b_dist1 s_dist1 s_dist2 : ℝ)
  (h1 : b_dist1 = 80)
  (h2 : s_dist1 = x / 2 - 80)
  (h3 : s_dist2 = s_dist1 + 180)
  (h4 : x / 4 * b_speed = (x / 2 - 80) * s_speed)
  (h5 : x / 4 * ((x / 2) - 100) = (x / 2 + 100) * s_speed) :
  x = 520 := 
sorry

end track_length_l146_146187


namespace interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l146_146479

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 - 1

theorem interval_of_monotonic_increase (x : ℝ) :
  ∃ k : ℤ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 := sorry

theorem parallel_vectors_tan_x (x : ℝ) (h₁ : Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
  Real.tan x = Real.sqrt 3 := sorry

theorem perpendicular_vectors_smallest_positive_x (x : ℝ) (h₁ : Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
 x = 5 * Real.pi / 6 := sorry

end interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l146_146479


namespace mod_6_computation_l146_146881

theorem mod_6_computation (a b n : ℕ) (h₁ : a ≡ 35 [MOD 6]) (h₂ : b ≡ 16 [MOD 6]) (h₃ : n = 1723) :
  (a ^ n - b ^ n) % 6 = 1 :=
by 
  -- proofs go here
  sorry

end mod_6_computation_l146_146881


namespace difference_in_pups_l146_146667

theorem difference_in_pups :
  let huskies := 5
  let pitbulls := 2
  let golden_retrievers := 4
  let pups_per_husky := 3
  let pups_per_pitbull := 3
  let total_adults := huskies + pitbulls + golden_retrievers
  let total_pups := total_adults + 30
  let total_husky_pups := huskies * pups_per_husky
  let total_pitbull_pups := pitbulls * pups_per_pitbull
  let H := pups_per_husky
  let D := (total_pups - total_husky_pups - total_pitbull_pups - 3 * golden_retrievers) / golden_retrievers
  D = 2 := sorry

end difference_in_pups_l146_146667


namespace intersection_of_multiples_of_2_l146_146138

theorem intersection_of_multiples_of_2 : 
  let M := {1, 2, 4, 8}
  let N := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  M ∩ N = {2, 4, 8} :=
by
  sorry

end intersection_of_multiples_of_2_l146_146138


namespace find_constants_u_v_l146_146193

theorem find_constants_u_v
  (n p r1 r2 : ℝ)
  (h1 : r1 + r2 = n)
  (h2 : r1 * r2 = p) :
  ∃ u v, (r1^4 + r2^4 = -u) ∧ (r1^4 * r2^4 = v) ∧ u = -(n^4 - 4*p*n^2 + 2*p^2) ∧ v = p^4 :=
by
  sorry

end find_constants_u_v_l146_146193


namespace jaylene_saves_fraction_l146_146542

-- Statement of the problem
theorem jaylene_saves_fraction (r_saves : ℝ) (j_saves : ℝ) (m_saves : ℝ) 
    (r_salary_fraction : r_saves = 2 / 5) 
    (m_salary_fraction : m_saves = 1 / 2) 
    (total_savings : 4 * (r_saves * 500 + j_saves * 500 + m_saves * 500) = 3000) : 
    j_saves = 3 / 5 := 
by 
  sorry

end jaylene_saves_fraction_l146_146542


namespace wade_average_points_per_game_l146_146694

variable (W : ℝ)

def teammates_average_points_per_game : ℝ := 40

def total_team_points_after_5_games : ℝ := 300

theorem wade_average_points_per_game :
  teammates_average_points_per_game * 5 + W * 5 = total_team_points_after_5_games →
  W = 20 :=
by
  intro h
  sorry

end wade_average_points_per_game_l146_146694


namespace june_vs_christopher_l146_146961

namespace SwordLength

def christopher_length : ℕ := 15
def jameson_length : ℕ := 3 + 2 * christopher_length
def june_length : ℕ := 5 + jameson_length

theorem june_vs_christopher : june_length - christopher_length = 23 := by
  show 5 + (3 + 2 * christopher_length) - christopher_length = 23
  sorry

end SwordLength

end june_vs_christopher_l146_146961


namespace rational_range_l146_146556

theorem rational_range (a : ℚ) (h : a - |a| = 2 * a) : a ≤ 0 := 
sorry

end rational_range_l146_146556


namespace annual_subscription_cost_l146_146795

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end annual_subscription_cost_l146_146795


namespace stop_signs_per_mile_l146_146728

-- Define the conditions
def miles_traveled := 5 + 2
def stop_signs_encountered := 17 - 3

-- Define the proof statement
theorem stop_signs_per_mile : (stop_signs_encountered / miles_traveled) = 2 := by
  -- Proof goes here
  sorry

end stop_signs_per_mile_l146_146728


namespace last_score_is_80_l146_146762

-- Define the list of scores
def scores : List ℕ := [71, 76, 80, 82, 91]

-- Define the total sum of the scores
def total_sum : ℕ := 400

-- Define the condition that the average after each score is an integer
def average_integer_condition (scores : List ℕ) (total_sum : ℕ) : Prop :=
  ∀ (sublist : List ℕ), sublist ≠ [] → sublist ⊆ scores → 
  (sublist.sum / sublist.length : ℕ) * sublist.length = sublist.sum

-- Define the proposition to prove that the last score entered must be 80
theorem last_score_is_80 : ∃ (last_score : ℕ), (last_score = 80) ∧
  average_integer_condition scores total_sum :=
sorry

end last_score_is_80_l146_146762


namespace find_function_l146_146162

noncomputable def solution_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = x + f y → ∃ c : ℝ, ∀ x : ℝ, f x = x + c

theorem find_function (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_function_l146_146162


namespace mail_distribution_l146_146623

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end mail_distribution_l146_146623


namespace find_x_value_l146_146058

open Real

theorem find_x_value (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * log x) = c) : 
  x = 10^((5 / c - a^2) / b) := 
by 
  sorry

end find_x_value_l146_146058


namespace compute_g_neg_x_l146_146874

noncomputable def g (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^2 - 3*x + 2)

theorem compute_g_neg_x (x : ℝ) (h : x^2 ≠ 2) : g (-x) = 1 / g x := 
  by sorry

end compute_g_neg_x_l146_146874


namespace jason_money_determination_l146_146904

theorem jason_money_determination (fred_last_week : ℕ) (fred_earned : ℕ) (fred_now : ℕ) (jason_last_week : ℕ → Prop)
  (h1 : fred_last_week = 23)
  (h2 : fred_earned = 63)
  (h3 : fred_now = 86) :
  ¬ ∃ x, jason_last_week x :=
by
  sorry

end jason_money_determination_l146_146904


namespace sally_baseball_cards_l146_146245

theorem sally_baseball_cards (initial_cards torn_cards purchased_cards : ℕ) 
    (h_initial : initial_cards = 39)
    (h_torn : torn_cards = 9)
    (h_purchased : purchased_cards = 24) :
    initial_cards - torn_cards - purchased_cards = 6 := by
  sorry

end sally_baseball_cards_l146_146245


namespace calc_residue_modulo_l146_146173

theorem calc_residue_modulo :
  let a := 320
  let b := 16
  let c := 28
  let d := 5
  let e := 7
  let n := 14
  (a * b - c * d + e) % n = 3 :=
by
  sorry

end calc_residue_modulo_l146_146173


namespace ratio_of_new_r_to_original_r_l146_146784

theorem ratio_of_new_r_to_original_r
  (r₁ r₂ : ℝ)
  (a₁ a₂ : ℝ)
  (h₁ : a₁ = (2 * r₁)^3)
  (h₂ : a₂ = (2 * r₂)^3)
  (h : a₂ = 0.125 * a₁) :
  r₂ / r₁ = 1 / 2 :=
by
  sorry

end ratio_of_new_r_to_original_r_l146_146784


namespace inequality_lemma_l146_146959

-- Define the conditions: x and y are positive numbers and x > y
variables (x y : ℝ)
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y)

-- State the theorem to be proved
theorem inequality_lemma : 2 * x + 1 / (x^2 - 2*x*y + y^2) >= 2 * y + 3 :=
by
  sorry

end inequality_lemma_l146_146959


namespace andre_total_payment_l146_146684

def treadmill_initial_price : ℝ := 1350
def treadmill_discount : ℝ := 0.30
def plate_initial_price : ℝ := 60
def plate_discount : ℝ := 0.15
def plate_quantity : ℝ := 2

theorem andre_total_payment :
  let treadmill_discounted_price := treadmill_initial_price * (1 - treadmill_discount)
  let plates_total_initial_price := plate_quantity * plate_initial_price
  let plates_discounted_price := plates_total_initial_price * (1 - plate_discount)
  treadmill_discounted_price + plates_discounted_price = 1047 := 
by
  sorry

end andre_total_payment_l146_146684


namespace mod_remainder_l146_146379

theorem mod_remainder (a b c d : ℕ) (h1 : a = 11) (h2 : b = 9) (h3 : c = 7) (h4 : d = 7) :
  (a^d + b^(d + 1) + c^(d + 2)) % d = 1 := 
by 
  sorry

end mod_remainder_l146_146379


namespace arc_length_of_circle_l146_146778

theorem arc_length_of_circle (r : ℝ) (θ_peripheral : ℝ) (h_r : r = 5) (h_θ : θ_peripheral = 2/3 * π) :
  r * (2/3 * θ_peripheral) = 20 * π / 3 := 
by sorry

end arc_length_of_circle_l146_146778


namespace original_student_count_l146_146901

variable (A B C N D : ℕ)
variable (hA : A = 40)
variable (hB : B = 32)
variable (hC : C = 36)
variable (hD : D = N * A)
variable (hNewSum : D + 8 * B = (N + 8) * C)

theorem original_student_count (hA : A = 40) (hB : B = 32) (hC : C = 36) (hD : D = N * A) (hNewSum : D + 8 * B = (N + 8) * C) : 
  N = 8 :=
by
  sorry

end original_student_count_l146_146901


namespace green_passes_blue_at_46_l146_146979

variable {t : ℕ}
variable {k1 k2 k3 k4 : ℝ}
variable {b1 b2 b3 b4 : ℝ}

def elevator_position (k : ℝ) (b : ℝ) (t : ℕ) : ℝ := k * t + b

axiom red_catches_blue_at_36 :
  elevator_position k1 b1 36 = elevator_position k2 b2 36

axiom red_passes_green_at_42 :
  elevator_position k1 b1 42 = elevator_position k3 b3 42

axiom red_passes_yellow_at_48 :
  elevator_position k1 b1 48 = elevator_position k4 b4 48

axiom yellow_passes_blue_at_51 :
  elevator_position k4 b4 51 = elevator_position k2 b2 51

axiom yellow_catches_green_at_54 :
  elevator_position k4 b4 54 = elevator_position k3 b3 54

theorem green_passes_blue_at_46 : 
  elevator_position k3 b3 46 = elevator_position k2 b2 46 := 
sorry

end green_passes_blue_at_46_l146_146979


namespace no_int_k_such_that_P_k_equals_8_l146_146272

theorem no_int_k_such_that_P_k_equals_8
    (P : Polynomial ℤ) 
    (a b c d k : ℤ)
    (h0: a ≠ b)
    (h1: a ≠ c)
    (h2: a ≠ d)
    (h3: b ≠ c)
    (h4: b ≠ d)
    (h5: c ≠ d)
    (h6: P.eval a = 5)
    (h7: P.eval b = 5)
    (h8: P.eval c = 5)
    (h9: P.eval d = 5)
    : P.eval k ≠ 8 := by
  sorry

end no_int_k_such_that_P_k_equals_8_l146_146272


namespace nancy_ate_3_apples_l146_146602

theorem nancy_ate_3_apples
  (mike_apples : ℝ)
  (keith_apples : ℝ)
  (apples_left : ℝ)
  (mike_apples_eq : mike_apples = 7.0)
  (keith_apples_eq : keith_apples = 6.0)
  (apples_left_eq : apples_left = 10.0) :
  mike_apples + keith_apples - apples_left = 3.0 := 
by
  rw [mike_apples_eq, keith_apples_eq, apples_left_eq]
  norm_num

end nancy_ate_3_apples_l146_146602


namespace remainder_when_divided_by_x_minus_2_l146_146094

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 10*x^3 + 20*x^2 - 5*x - 21

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 33 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l146_146094


namespace integer_solutions_to_equation_l146_146461

theorem integer_solutions_to_equation :
  ∀ (a b c : ℤ), a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integer_solutions_to_equation_l146_146461


namespace original_decimal_l146_146617

variable (x : ℝ)

theorem original_decimal (h : x - x / 100 = 1.485) : x = 1.5 :=
sorry

end original_decimal_l146_146617


namespace consecutive_integers_sum_l146_146376

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 := by
  sorry

end consecutive_integers_sum_l146_146376


namespace number_of_good_card_groups_l146_146534

noncomputable def card_value (k : ℕ) : ℕ := 2 ^ k

def is_good_card_group (cards : Finset ℕ) : Prop :=
  (cards.sum card_value = 2004)

theorem number_of_good_card_groups : 
  ∃ n : ℕ, n = 1006009 ∧ ∃ (cards : Finset ℕ), is_good_card_group cards :=
sorry

end number_of_good_card_groups_l146_146534


namespace emery_total_alteration_cost_l146_146028

-- Definition of the initial conditions
def num_pairs_of_shoes := 17
def cost_per_shoe := 29
def shoes_per_pair := 2

-- Proving the total cost
theorem emery_total_alteration_cost : num_pairs_of_shoes * shoes_per_pair * cost_per_shoe = 986 := by
  sorry

end emery_total_alteration_cost_l146_146028


namespace volume_of_adjacent_cubes_l146_146744

theorem volume_of_adjacent_cubes 
(side_length count : ℝ) 
(h_side : side_length = 5) 
(h_count : count = 5) : 
  (count * side_length ^ 3) = 625 :=
by
  -- Proof steps (skipped)
  sorry

end volume_of_adjacent_cubes_l146_146744


namespace roy_missed_days_l146_146733

theorem roy_missed_days {hours_per_day days_per_week actual_hours_week missed_days : ℕ}
    (h1 : hours_per_day = 2)
    (h2 : days_per_week = 5)
    (h3 : actual_hours_week = 6)
    (expected_hours_week : ℕ := hours_per_day * days_per_week)
    (missed_hours : ℕ := expected_hours_week - actual_hours_week)
    (missed_days := missed_hours / hours_per_day) :
  missed_days = 2 := by
  sorry

end roy_missed_days_l146_146733


namespace constant_term_of_expansion_l146_146153

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the expansion term
def expansion_term (n k : ℕ) (a b : ℚ) : ℚ :=
  (binom n k) * (a ^ k) * (b ^ (n - k))

-- Define the specific example
def specific_expansion_term : ℚ :=
  expansion_term 8 4 3 (2 : ℚ)

theorem constant_term_of_expansion : specific_expansion_term = 90720 :=
by
  -- The proof is omitted
  sorry

end constant_term_of_expansion_l146_146153


namespace square_side_length_l146_146168

-- Define the conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 4
def area_rectangle : ℝ := rectangle_width * rectangle_length
def area_square : ℝ := area_rectangle

-- Prove the side length of the square
theorem square_side_length :
  ∃ s : ℝ, s * s = area_square ∧ s = 4 := 
  by {
    -- Here you'd write the proof step, but it's omitted as per instructions
    sorry
  }

end square_side_length_l146_146168


namespace domain_of_f_l146_146333

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x + 1 > 0 ∧ x + 1 ≠ 1} = {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0} :=
by 
  sorry

end domain_of_f_l146_146333


namespace num_of_three_digit_integers_greater_than_217_l146_146792

theorem num_of_three_digit_integers_greater_than_217 : 
  ∃ n : ℕ, n = 82 ∧ ∀ x : ℕ, (217 < x ∧ x < 300) → 200 ≤ x ∧ x ≤ 299 → n = 82 := 
by
  sorry

end num_of_three_digit_integers_greater_than_217_l146_146792


namespace new_volume_l146_146983

variable (l w h : ℝ)

-- Given conditions
def volume := l * w * h = 5000
def surface_area := l * w + l * h + w * h = 975
def sum_of_edges := l + w + h = 60

-- Statement to prove
theorem new_volume (h1 : volume l w h) (h2 : surface_area l w h) (h3 : sum_of_edges l w h) :
  (l + 2) * (w + 2) * (h + 2) = 7198 :=
by
  sorry

end new_volume_l146_146983


namespace negation_of_proposition_l146_146288

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0)) ↔ (∃ x : ℝ, x^2 + 2 * x + 3 < 0) :=
by sorry

end negation_of_proposition_l146_146288


namespace harvest_bushels_l146_146883

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end harvest_bushels_l146_146883


namespace value_of_product_of_sums_of_roots_l146_146920

theorem value_of_product_of_sums_of_roots 
    (a b c : ℂ)
    (h1 : a + b + c = 15)
    (h2 : a * b + b * c + c * a = 22)
    (h3 : a * b * c = 8) :
    (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end value_of_product_of_sums_of_roots_l146_146920


namespace will_initially_bought_seven_boxes_l146_146948

theorem will_initially_bought_seven_boxes :
  let given_away_pieces := 3 * 4
  let total_initial_pieces := given_away_pieces + 16
  let initial_boxes := total_initial_pieces / 4
  initial_boxes = 7 := 
by
  sorry

end will_initially_bought_seven_boxes_l146_146948


namespace strictly_monotone_function_l146_146974

open Function

-- Define the problem
theorem strictly_monotone_function (f : ℝ → ℝ) (F : ℝ → ℝ → ℝ)
  (hf_cont : Continuous f) (hf_nonconst : ¬ (∃ c, ∀ x, f x = c))
  (hf_eq : ∀ x y : ℝ, f (x + y) = F (f x) (f y)) :
  StrictMono f :=
sorry

end strictly_monotone_function_l146_146974


namespace profit_ratio_l146_146240

theorem profit_ratio (p_investment q_investment : ℝ) (h₁ : p_investment = 50000) (h₂ : q_investment = 66666.67) :
  (1 / q_investment) = (3 / 4 * 1 / p_investment) :=
by
  sorry

end profit_ratio_l146_146240


namespace ellipses_same_eccentricity_l146_146294

theorem ellipses_same_eccentricity 
  (a b : ℝ) (k : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : k > 0)
  (e1_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / (a^2)) + (y^2 / (b^2)) = 1)
  (e2_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = k ↔ (x^2 / (ka^2)) + (y^2 / (kb^2)) = 1) :
  1 - (b^2 / a^2) = 1 - (b^2 / (ka^2)) :=
by
  sorry

end ellipses_same_eccentricity_l146_146294


namespace journey_total_time_l146_146852

noncomputable def total_time (D : ℝ) (r_dist : ℕ → ℕ) (r_time : ℕ → ℕ) (u_speed : ℝ) : ℝ :=
  let dist_uphill := D * (r_dist 1) / (r_dist 1 + r_dist 2 + r_dist 3)
  let t_uphill := (dist_uphill / u_speed)
  let k := t_uphill / (r_time 1)
  (r_time 1 + r_time 2 + r_time 3) * k

theorem journey_total_time :
  total_time 50 (fun n => if n = 1 then 1 else if n = 2 then 2 else 3) 
                (fun n => if n = 1 then 4 else if n = 2 then 5 else 6) 
                3 = 10 + 5/12 :=
by
  sorry

end journey_total_time_l146_146852


namespace solve_for_a_l146_146071

noncomputable def line_slope_parallels (a : ℝ) : Prop :=
  (a^2 - a) = 6

theorem solve_for_a : { a : ℝ // line_slope_parallels a } → (a = -2 ∨ a = 3) := by
  sorry

end solve_for_a_l146_146071


namespace remaining_nails_after_repairs_l146_146907

def fraction_used (perc : ℤ) (total : ℤ) : ℤ :=
  (total * perc) / 100

def after_kitchen (nails : ℤ) : ℤ :=
  nails - fraction_used 35 nails

def after_fence (nails : ℤ) : ℤ :=
  let remaining := after_kitchen nails
  remaining - fraction_used 75 remaining

def after_table (nails : ℤ) : ℤ :=
  let remaining := after_fence nails
  remaining - fraction_used 55 remaining

def after_floorboard (nails : ℤ) : ℤ :=
  let remaining := after_table nails
  remaining - fraction_used 30 remaining

theorem remaining_nails_after_repairs :
  after_floorboard 400 = 21 :=
by
  sorry

end remaining_nails_after_repairs_l146_146907


namespace quadratic_solution_l146_146386

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end quadratic_solution_l146_146386


namespace boxes_of_toothpicks_needed_l146_146351

def total_cards : Nat := 52
def unused_cards : Nat := 23
def cards_used : Nat := total_cards - unused_cards

def toothpicks_wall_per_card : Nat := 64
def windows_per_card : Nat := 3
def doors_per_card : Nat := 2
def toothpicks_per_window_or_door : Nat := 12
def roof_toothpicks : Nat := 1250
def box_capacity : Nat := 750

def toothpicks_for_walls : Nat := cards_used * toothpicks_wall_per_card
def toothpicks_per_card_windows_doors : Nat := (windows_per_card + doors_per_card) * toothpicks_per_window_or_door
def toothpicks_for_windows_doors : Nat := cards_used * toothpicks_per_card_windows_doors
def total_toothpicks_needed : Nat := toothpicks_for_walls + toothpicks_for_windows_doors + roof_toothpicks

def boxes_needed := Nat.ceil (total_toothpicks_needed / box_capacity)

theorem boxes_of_toothpicks_needed : boxes_needed = 7 := by
  -- Proof should be done here
  sorry

end boxes_of_toothpicks_needed_l146_146351


namespace find_n_l146_146188

theorem find_n (n : ℕ) (h : (2 * n + 1) / 3 = 2022) : n = 3033 :=
sorry

end find_n_l146_146188


namespace exists_Q_R_l146_146034

noncomputable def P (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 1

theorem exists_Q_R : ∃ (Q R : Polynomial ℚ), 
  (Q.degree > 0 ∧ R.degree > 0) ∧
  (∀ (y : ℚ), (Q.eval y) * (R.eval y) = P (5 * y^2)) :=
sorry

end exists_Q_R_l146_146034


namespace jason_earns_88_dollars_l146_146311

theorem jason_earns_88_dollars (earn_after_school: ℝ) (earn_saturday: ℝ)
  (total_hours: ℝ) (saturday_hours: ℝ) (after_school_hours: ℝ) (total_earn: ℝ)
  (h1 : earn_after_school = 4.00)
  (h2 : earn_saturday = 6.00)
  (h3 : total_hours = 18)
  (h4 : saturday_hours = 8)
  (h5 : after_school_hours = total_hours - saturday_hours)
  (h6 : total_earn = after_school_hours * earn_after_school + saturday_hours * earn_saturday) :
  total_earn = 88.00 :=
by
  sorry

end jason_earns_88_dollars_l146_146311


namespace sin_double_angle_l146_146430

theorem sin_double_angle (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) = 4 / 5 :=
sorry

end sin_double_angle_l146_146430


namespace double_burger_cost_l146_146057

theorem double_burger_cost (D : ℝ) : 
  let single_burger_cost := 1.00
  let total_burgers := 50
  let double_burgers := 37
  let total_cost := 68.50
  let single_burgers := total_burgers - double_burgers
  let singles_cost := single_burgers * single_burger_cost
  let doubles_cost := total_cost - singles_cost
  let burger_cost := doubles_cost / double_burgers
  burger_cost = D := 
by 
  sorry

end double_burger_cost_l146_146057


namespace totalCorrectQuestions_l146_146956

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end totalCorrectQuestions_l146_146956


namespace quadratic_roots_l146_146629

theorem quadratic_roots : ∀ (x : ℝ), x^2 + 5 * x - 4 = 0 ↔ x = (-5 + Real.sqrt 41) / 2 ∨ x = (-5 - Real.sqrt 41) / 2 := 
by
  sorry

end quadratic_roots_l146_146629


namespace exists_x_for_log_eqn_l146_146982

theorem exists_x_for_log_eqn (a : ℝ) (ha : 0 < a) :
  ∃ (x : ℝ), (1 < x) ∧ (Real.log (a * x) / Real.log 10 = 2 * Real.log (x - 1) / Real.log 10) ∧ 
  x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 := sorry

end exists_x_for_log_eqn_l146_146982


namespace complex_b_value_l146_146425

open Complex

theorem complex_b_value (b : ℝ) (h : (2 - b * I) / (1 + 2 * I) = (2 - 2 * b) / 5 + ((-4 - b) / 5) * I) :
  b = -2 / 3 :=
sorry

end complex_b_value_l146_146425


namespace positive_integer_count_l146_146707

/-
  Prove that the number of positive integers \( n \) for which \( \frac{n(n+1)}{2} \) divides \( 30n \) is 11.
-/

theorem positive_integer_count (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k ≤ 11 ∧ (2 * 30 * n) % (n * (n + 1)) = 0) :=
sorry

end positive_integer_count_l146_146707


namespace luncheon_cost_l146_146742

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + 2 * p = 3.50)
  (h2 : 3 * s + 7 * c + 2 * p = 4.90) :
  s + c + p = 1.00 :=
  sorry

end luncheon_cost_l146_146742


namespace math_books_together_l146_146044

theorem math_books_together (math_books english_books : ℕ) (h_math_books : math_books = 2) (h_english_books : english_books = 2) : 
  ∃ ways, ways = 12 := by
  sorry

end math_books_together_l146_146044


namespace oscar_marathon_training_l146_146978

theorem oscar_marathon_training :
  let initial_miles := 2
  let target_miles := 20
  let increment_per_week := (2 : ℝ) / 3
  ∃ weeks_required, target_miles - initial_miles = weeks_required * increment_per_week → weeks_required = 27 :=
by
  sorry

end oscar_marathon_training_l146_146978


namespace stream_speed_l146_146853

-- Definitions based on conditions
def speed_in_still_water : ℝ := 5
def distance_downstream : ℝ := 100
def time_downstream : ℝ := 10

-- The required speed of the stream
def speed_of_stream (v : ℝ) : Prop :=
  distance_downstream = (speed_in_still_water + v) * time_downstream

-- Proof statement: the speed of the stream is 5 km/hr
theorem stream_speed : ∃ v, speed_of_stream v ∧ v = 5 := 
by
  use 5
  unfold speed_of_stream
  sorry

end stream_speed_l146_146853


namespace geoff_initial_percent_l146_146396

theorem geoff_initial_percent (votes_cast : ℕ) (win_percent : ℝ) (needed_more_votes : ℕ) (initial_votes : ℕ)
  (h1 : votes_cast = 6000)
  (h2 : win_percent = 50.5)
  (h3 : needed_more_votes = 3000)
  (h4 : initial_votes = 31) :
  (initial_votes : ℝ) / votes_cast * 100 = 0.52 :=
by
  sorry

end geoff_initial_percent_l146_146396


namespace expand_product_l146_146441

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := 
by
  sorry

end expand_product_l146_146441


namespace polynomial_coeff_sum_l146_146704

theorem polynomial_coeff_sum {a_0 a_1 a_2 a_3 a_4 a_5 : ℝ} :
  (2 * (x : ℝ) - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l146_146704


namespace coffee_price_decrease_is_37_5_l146_146209

-- Define the initial and new prices
def initial_price_per_packet := 12 / 3
def new_price_per_packet := 10 / 4

-- Define the calculation of the percent decrease
def percent_decrease (initial_price : ℚ) (new_price : ℚ) : ℚ :=
  ((initial_price - new_price) / initial_price) * 100

-- The theorem statement
theorem coffee_price_decrease_is_37_5 :
  percent_decrease initial_price_per_packet new_price_per_packet = 37.5 := by
  sorry

end coffee_price_decrease_is_37_5_l146_146209


namespace circumscribed_circles_intersect_l146_146268

noncomputable def circumcircle (a b c : Point) : Set Point := sorry

noncomputable def intersect_at_single_point (circles : List (Set Point)) : Option Point := sorry

variables {A1 A2 A3 B1 B2 B3 : Point}

theorem circumscribed_circles_intersect
  (h1 : ∃ P, ∀ circle ∈ [
    circumcircle A1 A2 B3, 
    circumcircle A1 B2 A3, 
    circumcircle B1 A2 A3
  ], P ∈ circle) :
  ∃ Q, ∀ circle ∈ [
    circumcircle B1 B2 A3, 
    circumcircle B1 A2 B3, 
    circumcircle A1 B2 B3
  ], Q ∈ circle :=
sorry

end circumscribed_circles_intersect_l146_146268


namespace max_sum_non_zero_nats_l146_146653

theorem max_sum_non_zero_nats (O square : ℕ) (hO : O ≠ 0) (hsquare : square ≠ 0) :
  (O / 11 < 7 / square) ∧ (7 / square < 4 / 5) → O + square = 77 :=
by 
  sorry -- Proof omitted as requested

end max_sum_non_zero_nats_l146_146653


namespace fraction_subtraction_l146_146299

theorem fraction_subtraction (x y : ℝ) (h : x ≠ y) : (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end fraction_subtraction_l146_146299


namespace problem1_problem2_problem3_l146_146345

-- Problem 1
def s_type_sequence (a : ℕ → ℕ) : Prop := 
∀ n ≥ 1, a (n+1) - a n > 3

theorem problem1 (a : ℕ → ℕ) (h₀ : a 1 = 4) (h₁ : a 2 = 8) 
  (h₂ : ∀ n ≥ 2, a n + a (n - 1) = 8 * n - 4) : s_type_sequence a := 
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ) (h₀ : ∀ n m, a (n * m) = (a n) ^ m)
  (b : ℕ → ℕ) (h₁ : ∀ n, b n = (3 * a n) / 4)
  (h₂ : s_type_sequence a)
  (h₃ : ¬ s_type_sequence b) : 
  (∀ n, a n = 2^(n+1)) ∨ (∀ n, a n = 2 * 3^(n-1)) ∨ (∀ n, a n = 5^ (n-1)) :=
sorry

-- Problem 3
theorem problem3 (c : ℕ → ℕ) 
  (h₀ : c 2 = 9)
  (h₁ : ∀ n ≥ 2, (1 / n - 1 / (n + 1)) * (2 + 1 / c n) ≤ 1 / c (n - 1) + 1 / c n 
               ∧ 1 / c (n - 1) + 1 / c n ≤ (1 / n - 1 / (n + 1)) * (2 + 1 / c (n-1))) :
  ∃ f : ℕ → ℕ, (s_type_sequence c) ∧ (∀ n, c n = (n + 1)^2) := 
sorry

end problem1_problem2_problem3_l146_146345


namespace area_enclosed_by_circle_l146_146645

theorem area_enclosed_by_circle : Π (x y : ℝ), x^2 + y^2 + 8 * x - 6 * y = -9 → 
  ∃ A, A = 7 * Real.pi :=
by
  sorry

end area_enclosed_by_circle_l146_146645


namespace smallest_r_l146_146802

theorem smallest_r {p q r : ℕ} (h1 : p < q) (h2 : q < r) (h3 : 2 * q = p + r) (h4 : r * r = p * q) : r = 5 :=
sorry

end smallest_r_l146_146802


namespace lucas_1500th_day_is_sunday_l146_146696

def days_in_week : ℕ := 7

def start_day : ℕ := 5  -- 0: Monday, 1: Tuesday, ..., 5: Friday

def nth_day_of_life (n : ℕ) : ℕ :=
  (n - 1 + start_day) % days_in_week

theorem lucas_1500th_day_is_sunday : nth_day_of_life 1500 = 0 :=
by
  sorry

end lucas_1500th_day_is_sunday_l146_146696


namespace frequency_number_correct_l146_146322

-- Define the sample capacity and the group frequency as constants
def sample_capacity : ℕ := 100
def group_frequency : ℝ := 0.3

-- State the theorem
theorem frequency_number_correct : sample_capacity * group_frequency = 30 := by
  -- Immediate calculation
  sorry

end frequency_number_correct_l146_146322


namespace mean_of_remaining_quiz_scores_l146_146292

theorem mean_of_remaining_quiz_scores (k : ℕ) (hk : k > 12) 
  (mean_k : ℝ) (mean_12 : ℝ) 
  (mean_class : mean_k = 8) 
  (mean_12_group : mean_12 = 14) 
  (mean_correct : mean_12 * 12 + mean_k * (k - 12) = 8 * k) :
  mean_k * (k - 12) = (8 * k - 168) := 
by {
  sorry
}

end mean_of_remaining_quiz_scores_l146_146292


namespace find_principal_amount_l146_146518

variable (P : ℝ)
variable (R : ℝ := 5)
variable (T : ℝ := 13)
variable (SI : ℝ := 1300)

theorem find_principal_amount (h1 : SI = (P * R * T) / 100) : P = 2000 :=
sorry

end find_principal_amount_l146_146518


namespace cubic_eq_factorization_l146_146007

theorem cubic_eq_factorization (a b c : ℝ) :
  (∃ m n : ℝ, (x^3 + a * x^2 + b * x + c = (x^2 + m) * (x + n))) ↔ (c = a * b) :=
sorry

end cubic_eq_factorization_l146_146007


namespace triangle_base_length_l146_146955

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 6) (A_eq : A = 13.5) (area_eq : A = (b * h) / 2) : b = 4.5 :=
by
  sorry

end triangle_base_length_l146_146955


namespace exists_natural_sum_of_squares_l146_146716

theorem exists_natural_sum_of_squares : ∃ n : ℕ, n^2 = 0^2 + 7^2 + 24^2 + 312^2 + 48984^2 :=
by {
  sorry
}

end exists_natural_sum_of_squares_l146_146716


namespace find_a_of_line_slope_l146_146289

theorem find_a_of_line_slope (a : ℝ) (h1 : a > 0)
  (h2 : ∃ (b : ℝ), (a, 5) = (b * 1, b * 2) ∧ (2, a) = (b * 1, 2 * b) ∧ b = 1) 
  : a = 3 := 
sorry

end find_a_of_line_slope_l146_146289


namespace johann_mail_l146_146604

def pieces_of_mail_total : ℕ := 180
def pieces_of_mail_friends : ℕ := 41
def friends : ℕ := 2
def pieces_of_mail_johann : ℕ := pieces_of_mail_total - (pieces_of_mail_friends * friends)

theorem johann_mail : pieces_of_mail_johann = 98 := by
  sorry

end johann_mail_l146_146604


namespace gcf_2550_7140_l146_146552

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_2550_7140 : gcf 2550 7140 = 510 := 
  by 
    sorry

end gcf_2550_7140_l146_146552


namespace fraction_equation_l146_146492

theorem fraction_equation (a : ℕ) (h : a > 0) (eq : (a : ℚ) / (a + 35) = 0.875) : a = 245 :=
by
  sorry

end fraction_equation_l146_146492


namespace inequality_solution_sets_l146_146380

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 - 5 * x + b > 0 ↔ x < -1 / 3 ∨ x > 1 / 2) →
  (∀ x : ℝ, bx^2 - 5 * x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end inequality_solution_sets_l146_146380


namespace max_heaps_of_stones_l146_146339

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l146_146339


namespace central_angle_of_sector_l146_146116

variable (A : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions: A is the area of the sector, and r is the radius.
def is_sector (A : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  A = (1 / 2) * α * r^2

-- Proof that the central angle α given the conditions is 3π/4.
theorem central_angle_of_sector (h1 : is_sector (3 * Real.pi / 8) 1 α) : 
  α = 3 * Real.pi / 4 := 
  sorry

end central_angle_of_sector_l146_146116


namespace triangle_condition_l146_146453

-- Definitions based on the conditions
def angle_equal (A B C : ℝ) : Prop := A = B - C
def angle_ratio123 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ A / C = 1 / 3 ∧ B / C = 2 / 3
def pythagorean (a b c : ℝ) : Prop := a * a + b * b = c * c
def side_ratio456 (a b c : ℝ) : Prop := a / b = 4 / 5 ∧ a / c = 4 / 6 ∧ b / c = 5 / 6

-- Main hypothesis with right-angle and its conditions in different options
def is_right_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (angle_equal A B C → A = 90 ∨ B = 90 ∨ C = 90) ∧
  (angle_ratio123 A B C → A = 30 ∧ B = 60 ∧ C = 90) ∧
  (pythagorean a b c → true) ∧
  (side_ratio456 a b c → false) -- option D cannot confirm the triangle is right

theorem triangle_condition (A B C a b c : ℝ) : is_right_triangle A B C a b c :=
sorry

end triangle_condition_l146_146453


namespace ending_number_of_range_l146_146183

theorem ending_number_of_range (n : ℕ) (h : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ n = 29 + 11 * k) : n = 77 := by
  sorry

end ending_number_of_range_l146_146183


namespace range_of_m_l146_146443

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0) → (2*x + 1 > 3) → (x > 1)) → (m ≤ 1) :=
by
  intros h
  sorry

end range_of_m_l146_146443


namespace Bowen_total_spent_l146_146242

def pencil_price : ℝ := 0.25
def pen_price : ℝ := 0.15
def num_pens : ℕ := 40

def num_pencils := num_pens + (2 / 5) * num_pens

theorem Bowen_total_spent : num_pencils * pencil_price + num_pens * pen_price = 20 := by
  sorry

end Bowen_total_spent_l146_146242


namespace stamp_solutions_l146_146424

theorem stamp_solutions (n : ℕ) (h1 : ∀ (k : ℕ), k < 115 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) 
  (h2 : ¬ ∃ (a b c : ℕ), 3 * a + n * b + (n + 1) * c = 115) 
  (h3 : ∀ (k : ℕ), 116 ≤ k ∧ k ≤ 120 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) : 
  n = 59 :=
sorry

end stamp_solutions_l146_146424


namespace jessica_total_money_after_activities_l146_146267

-- Definitions for given conditions
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def earned_from_washing_car : ℕ := 6

-- Theorem statement
theorem jessica_total_money_after_activities : 
  (weekly_allowance - spent_on_movies) + earned_from_washing_car = 11 :=
by 
  sorry

end jessica_total_money_after_activities_l146_146267


namespace quadratic_solution_set_l146_146824

theorem quadratic_solution_set (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧ 
  (∀ x : ℝ, bx + c > 0 ↔ x < 6) = false ∧ 
  (a + b + c < 0) ∧
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ x < -1 / 3 ∨ x > 1 / 2) :=
sorry

end quadratic_solution_set_l146_146824


namespace total_peaches_in_baskets_l146_146993

def total_peaches (red_peaches : ℕ) (green_peaches : ℕ) (baskets : ℕ) : ℕ :=
  (red_peaches + green_peaches) * baskets

theorem total_peaches_in_baskets :
  total_peaches 19 4 15 = 345 :=
by
  sorry

end total_peaches_in_baskets_l146_146993


namespace speed_of_man_rowing_upstream_l146_146210

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream V_s : ℝ) 
  (h1 : V_m = 25) 
  (h2 : V_downstream = 38) :
  V_upstream = V_m - (V_downstream - V_m) :=
by
  sorry

end speed_of_man_rowing_upstream_l146_146210


namespace valid_sequences_count_l146_146551

def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n < 3 then 0
  else g (n - 4) + 3 * g (n - 5) + 3 * g (n - 6)

theorem valid_sequences_count : g 17 = 37 :=
  sorry

end valid_sequences_count_l146_146551


namespace no_such_polynomials_exists_l146_146885

theorem no_such_polynomials_exists :
  ¬ ∃ (f g : Polynomial ℚ), (∀ x y : ℚ, f.eval x * g.eval y = x^200 * y^200 + 1) := 
by 
  sorry

end no_such_polynomials_exists_l146_146885


namespace right_triangle_OAB_condition_l146_146555

theorem right_triangle_OAB_condition
  (a b : ℝ)
  (h1: a ≠ 0) 
  (h2: b ≠ 0) :
  (b - a^3) * (b - a^3 - 1/a) = 0 :=
sorry

end right_triangle_OAB_condition_l146_146555


namespace degrees_multiplication_proof_l146_146302

/-- Convert a measurement given in degrees and minutes to purely degrees. -/
def degrees (d : Int) (m : Int) : ℚ := d + m / 60

/-- Given conditions: -/
def lhs : ℚ := degrees 21 17
def rhs : ℚ := degrees 106 25

/-- The theorem to prove the mathematical problem. -/
theorem degrees_multiplication_proof : lhs * 5 = rhs := sorry

end degrees_multiplication_proof_l146_146302


namespace men_count_eq_eight_l146_146526

theorem men_count_eq_eight (M W B : ℕ) (total_earnings : ℝ) (men_wages : ℝ)
  (H1 : M = W) (H2 : W = B) (H3 : B = 8)
  (H4 : total_earnings = 105) (H5 : men_wages = 7) :
  M = 8 := 
by 
  -- We need to show M = 8 given conditions
  sorry

end men_count_eq_eight_l146_146526


namespace minimum_f_value_l146_146077

noncomputable def f (x y : ℝ) : ℝ :=
  y / x + 16 * x / (2 * x + y)

theorem minimum_f_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, (∀ x y, f x y ≥ t) ∧ t = 6 := sorry

end minimum_f_value_l146_146077


namespace min_ab_given_parallel_l146_146533

-- Define the conditions
def parallel_vectors (a b : ℝ) : Prop :=
  4 * b - a * (b - 1) = 0 ∧ b > 1

-- Prove the main statement
theorem min_ab_given_parallel (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h_parallel : parallel_vectors a b) :
  a + b = 9 :=
sorry  -- Proof is omitted

end min_ab_given_parallel_l146_146533


namespace least_n_factorial_6930_l146_146656

theorem least_n_factorial_6930 (n : ℕ) (h : n! % 6930 = 0) : n ≥ 11 := by
  sorry

end least_n_factorial_6930_l146_146656


namespace range_of_x_l146_146176

theorem range_of_x (x p : ℝ) (hp : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by {
  sorry
}

end range_of_x_l146_146176


namespace quadratic_inequality_solution_l146_146072

theorem quadratic_inequality_solution:
  ∃ P q : ℝ,
  (1 / P < 0) ∧
  (-P * q = 6) ∧
  (P^2 = 8) ∧
  (P = -2 * Real.sqrt 2) ∧
  (q = 3 / 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_inequality_solution_l146_146072


namespace train_speed_l146_146544

theorem train_speed
  (train_length : ℝ)
  (cross_time : ℝ)
  (man_speed_kmh : ℝ)
  (train_speed_kmh : ℝ) :
  (train_length = 150) →
  (cross_time = 6) →
  (man_speed_kmh = 5) →
  (man_speed_kmh * 1000 / 3600 + (train_speed_kmh * 1000 / 3600)) * cross_time = train_length →
  train_speed_kmh = 85 :=
by
  intros htl hct hmk hs
  sorry

end train_speed_l146_146544


namespace divisible_by_six_l146_146693

theorem divisible_by_six (n a b : ℕ) (h1 : 2^n = 10 * a + b) (h2 : n > 3) (h3 : b > 0) (h4 : b < 10) : 6 ∣ (a * b) := 
sorry

end divisible_by_six_l146_146693


namespace maximum_value_ab_l146_146446

noncomputable def g (x : ℝ) : ℝ := 2 ^ x

theorem maximum_value_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : g a * g b = 2) :
  ab ≤ (1 / 4) := sorry

end maximum_value_ab_l146_146446


namespace problem_l146_146786

noncomputable def f (x φ : ℝ) : ℝ := 4 * Real.cos (3 * x + φ)

theorem problem 
  (φ : ℝ) (x1 x2 : ℝ)
  (hφ : |φ| < Real.pi / 2)
  (h_symm : ∀ x, f x φ = f (2 * (11 * Real.pi / 12) - x) φ)
  (hx1x2 : x1 ≠ x2)
  (hx1_range : -7 * Real.pi / 12 < x1 ∧ x1 < -Real.pi / 12)
  (hx2_range : -7 * Real.pi / 12 < x2 ∧ x2 < -Real.pi / 12)
  (h_eq : f x1 φ = f x2 φ) : 
  f (x1 + x2) (-Real.pi / 4) = 2 * Real.sqrt 2 := by
  sorry

end problem_l146_146786


namespace g_5_l146_146877

variable (g : ℝ → ℝ)

axiom additivity_condition : ∀ (x y : ℝ), g (x + y) = g x + g y
axiom g_1_nonzero : g 1 ≠ 0

theorem g_5 : g 5 = 5 * g 1 :=
by
  sorry

end g_5_l146_146877


namespace max_profit_l146_146306

noncomputable def maximum_profit : ℤ := 
  21000

theorem max_profit (x y : ℕ) 
  (h1 : 4 * x + 8 * y ≤ 8000)
  (h2 : 2 * x + y ≤ 1300)
  (h3 : 15 * x + 20 * y ≤ maximum_profit) : 
  15 * x + 20 * y = maximum_profit := 
sorry

end max_profit_l146_146306


namespace arithmetic_sequence_sum_l146_146942

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum (h1 : a 2 + a 3 = 2) (h2 : a 4 + a 5 = 6) : a 5 + a 6 = 8 :=
sorry

end arithmetic_sequence_sum_l146_146942


namespace at_least_one_not_lt_one_l146_146460

theorem at_least_one_not_lt_one (a b c : ℝ) (h : a + b + c = 3) : ¬ (a < 1 ∧ b < 1 ∧ c < 1) :=
by
  sorry

end at_least_one_not_lt_one_l146_146460


namespace part_1_part_2_l146_146702

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

-- (Part 1): Prove the value of a
theorem part_1 (a : ℝ) (P : ℝ × ℝ) (hP : P = (a, -4)) :
  (∃ t : ℝ, ∃ t₂ : ℝ, t ≠ t₂ ∧ P.2 = (2 * t^3 - 3 * t^2 + 1) + (6 * t^2 - 6 * t) * (a - t)) →
  a = -1 ∨ a = 7 / 2 :=
sorry

-- (Part 2): Prove the range of k
noncomputable def g (x k : ℝ) : ℝ := k * x + 1 - Real.log x

noncomputable def h (x k : ℝ) : ℝ := min (f x) (g x k)

theorem part_2 (k : ℝ) :
  (∀ x > 0, h x k = 0 → (x = 1 ∨ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 k = 0 ∧ h x2 k = 0)) →
  0 < k ∧ k < 1 / Real.exp 2 :=
sorry

end part_1_part_2_l146_146702


namespace find_CM_of_trapezoid_l146_146260

noncomputable def trapezoid_CM (AD BC : ℝ) (M : ℝ) : ℝ :=
  if (AD = 12) ∧ (BC = 8) ∧ (M = 2.4)
  then M
  else 0

theorem find_CM_of_trapezoid (trapezoid_ABCD : Type) (AD BC CM : ℝ) (AM_divides_eq_areas : Prop) :
  AD = 12 → BC = 8 → AM_divides_eq_areas → CM = 2.4 := 
by
  intros h1 h2 h3
  have : AD = 12 := h1
  have : BC = 8 := h2
  have : CM = 2.4 := sorry
  exact this

end find_CM_of_trapezoid_l146_146260


namespace avg_age_of_14_students_l146_146918

theorem avg_age_of_14_students (avg_age_25 : ℕ) (avg_age_10 : ℕ) (age_25th : ℕ) (total_students : ℕ) (remaining_students : ℕ) :
  avg_age_25 = 25 →
  avg_age_10 = 22 →
  age_25th = 13 →
  total_students = 25 →
  remaining_students = 14 →
  ( (total_students * avg_age_25) - (10 * avg_age_10) - age_25th ) / remaining_students = 28 :=
by
  intros
  sorry

end avg_age_of_14_students_l146_146918


namespace rational_number_theorem_l146_146829

theorem rational_number_theorem (x y : ℚ) 
  (h1 : |(x + 2017 : ℚ)| + (y - 2017) ^ 2 = 0) : 
  (x / y) ^ 2017 = -1 := 
by
  sorry

end rational_number_theorem_l146_146829


namespace solve_x_squared_eq_four_x_l146_146455

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end solve_x_squared_eq_four_x_l146_146455


namespace smallest_n_2000_divides_a_n_l146_146480

theorem smallest_n_2000_divides_a_n (a : ℕ → ℤ) 
  (h_rec : ∀ n, n ≥ 1 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)) 
  (h2000 : 2000 ∣ a 1999) : 
  ∃ n, n ≥ 2 ∧ 2000 ∣ a n ∧ n = 249 := 
by 
  sorry

end smallest_n_2000_divides_a_n_l146_146480


namespace average_payment_correct_l146_146915

-- Definitions based on conditions in the problem
def first_payments_num : ℕ := 20
def first_payment_amount : ℕ := 450

def second_payments_num : ℕ := 30
def increment_after_first : ℕ := 80

def third_payments_num : ℕ := 40
def increment_after_second : ℕ := 65

def fourth_payments_num : ℕ := 50
def increment_after_third : ℕ := 105

def fifth_payments_num : ℕ := 60
def increment_after_fourth : ℕ := 95

def total_payments : ℕ := first_payments_num + second_payments_num + third_payments_num + fourth_payments_num + fifth_payments_num

-- Function to calculate total paid amount
def total_amount_paid : ℕ :=
  (first_payments_num * first_payment_amount) +
  (second_payments_num * (first_payment_amount + increment_after_first)) +
  (third_payments_num * (first_payment_amount + increment_after_first + increment_after_second)) +
  (fourth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third)) +
  (fifth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third + increment_after_fourth))

-- Function to calculate average payment
def average_payment : ℕ := total_amount_paid / total_payments

-- The theorem to be proved
theorem average_payment_correct : average_payment = 657 := by
  sorry

end average_payment_correct_l146_146915


namespace f_one_eq_minus_one_third_f_of_a_f_is_odd_l146_146898

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_one_eq_minus_one_third : f 1 = -1/3 := 
by sorry

theorem f_of_a (a : ℝ) : f a = (1 - 2^a) / (2^a + 1) := 
by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

end f_one_eq_minus_one_third_f_of_a_f_is_odd_l146_146898


namespace parallel_line_eq_l146_146232

theorem parallel_line_eq (x y : ℝ) (c : ℝ) :
  (∀ x y, x - 2 * y - 2 = 0 → x - 2 * y + c = 0) ∧ (x = 1 ∧ y = 0) → c = -1 :=
by
  sorry

end parallel_line_eq_l146_146232


namespace value_of_expression_l146_146053

theorem value_of_expression (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 :=
by
  sorry

end value_of_expression_l146_146053


namespace positive_rational_solutions_condition_l146_146887

-- Definitions used in Lean 4 statement corresponding to conditions in the problem.
variable (a b : ℚ)

-- Lean Statement encapsulating the mathematical proof problem.
theorem positive_rational_solutions_condition :
  ∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x * y = a ∧ x + y = b ↔ (∃ k : ℚ, k^2 = b^2 - 4 * a ∧ k > 0) :=
by
  sorry

end positive_rational_solutions_condition_l146_146887


namespace compound_proposition_p_or_q_l146_146944

theorem compound_proposition_p_or_q : 
  (∃ (n : ℝ), ∀ (m : ℝ), m * n = m) ∨ 
  (∀ (n : ℝ), ∃ (m : ℝ), m^2 < n) := 
by
  sorry

end compound_proposition_p_or_q_l146_146944


namespace quadrilateral_area_inequality_equality_condition_l146_146002

theorem quadrilateral_area_inequality 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d) 
  : S ≤ 0.5 * (a * c + b * d) :=
sorry

theorem equality_condition 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d)
  (h_perpendicular : ∃ (α β : ℝ), α = 90 ∧ β = 90) 
  : S = 0.5 * (a * c + b * d) :=
sorry

end quadrilateral_area_inequality_equality_condition_l146_146002


namespace leonardo_sleep_fraction_l146_146953

theorem leonardo_sleep_fraction (h : 60 ≠ 0) : (12 / 60 : ℚ) = (1 / 5 : ℚ) :=
by
  sorry

end leonardo_sleep_fraction_l146_146953


namespace solve_quadratic_equation_l146_146743

theorem solve_quadratic_equation : 
  ∃ (a b c : ℤ), (0 < a) ∧ (64 * x^2 + 48 * x - 36 = 0) ∧ ((a * x + b)^2 = c) ∧ (a + b + c = 56) := 
by
  sorry

end solve_quadratic_equation_l146_146743


namespace train_speed_l146_146220

-- Definitions to capture the conditions
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 300
def time_to_cross_bridge : ℝ := 36

-- The speed of the train calculated according to the condition
def total_distance : ℝ := length_of_train + length_of_bridge

theorem train_speed : total_distance / time_to_cross_bridge = 11.11 :=
by
  sorry

end train_speed_l146_146220


namespace find_angle_A_find_area_l146_146620

-- Definition for angle A
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
  (h_tria : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_A : 0 < A ∧ A < Real.pi) :
  A = Real.pi / 3 :=
by
  sorry

-- Definition for area of triangle ABC
theorem find_area (a b c : ℝ) (A : ℝ)
  (h_a : a = Real.sqrt 7) 
  (h_b : b = 2)
  (h_A : A = Real.pi / 3) 
  (h_c : c = 3) :
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end find_angle_A_find_area_l146_146620


namespace borrowed_nickels_l146_146711

-- Define the initial and remaining number of nickels
def initial_nickels : ℕ := 87
def remaining_nickels : ℕ := 12

-- Prove that the number of nickels borrowed is 75
theorem borrowed_nickels : initial_nickels - remaining_nickels = 75 := by
  sorry

end borrowed_nickels_l146_146711


namespace inscribed_circle_radius_l146_146305

noncomputable def calculate_r (a b c : ℝ) : ℝ :=
  let term1 := 1 / a
  let term2 := 1 / b
  let term3 := 1 / c
  let term4 := 3 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))
  1 / (term1 + term2 + term3 + term4)

theorem inscribed_circle_radius :
  calculate_r 6 10 15 = 30 / (10 * Real.sqrt 26 + 3) :=
by
  sorry

end inscribed_circle_radius_l146_146305


namespace points_coplanar_if_and_only_if_b_neg1_l146_146475

/-- Points (0, 0, 0), (1, b, 0), (0, 1, b), (b, 0, 1) are coplanar if and only if b = -1. --/
theorem points_coplanar_if_and_only_if_b_neg1 (a b : ℝ) :
  (∃ u v w : ℝ, (u, v, w) = (0, 0, 0) ∨ (u, v, w) = (1, b, 0) ∨ (u, v, w) = (0, 1, b) ∨ (u, v, w) = (b, 0, 1)) →
  (b = -1) :=
sorry

end points_coplanar_if_and_only_if_b_neg1_l146_146475


namespace quotient_in_first_division_l146_146777

theorem quotient_in_first_division (N Q Q' : ℕ) (h₁ : N = 68 * Q) (h₂ : N % 67 = 1) : Q = 1 :=
by
  -- rest of the proof goes here
  sorry

end quotient_in_first_division_l146_146777


namespace quadratic_inequality_false_iff_range_of_a_l146_146315

theorem quadratic_inequality_false_iff_range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ (-1 < a ∧ a < 3) :=
sorry

end quadratic_inequality_false_iff_range_of_a_l146_146315


namespace petya_numbers_l146_146564

-- Define the arithmetic sequence property
def arithmetic_seq (a d : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => a + (n + 1) * d

-- Given conditions
theorem petya_numbers (a d : ℕ) : 
  (arithmetic_seq a d 0 = 6) ∧
  (arithmetic_seq a d 1 = 15) ∧
  (arithmetic_seq a d 2 = 24) ∧
  (arithmetic_seq a d 3 = 33) ∧
  (arithmetic_seq a d 4 = 42) :=
sorry

end petya_numbers_l146_146564


namespace function_passes_through_point_l146_146585

-- Lean 4 Statement
theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧ (a^(x-1) + 4) = y :=
by
  use 1
  use 5
  sorry

end function_passes_through_point_l146_146585


namespace subscription_total_l146_146892

theorem subscription_total (a b c : ℝ) (h1 : a = b + 4000) (h2 : b = c + 5000) (h3 : 15120 / 36000 = a / (a + b + c)) : 
  a + b + c = 50000 :=
by 
  sorry

end subscription_total_l146_146892


namespace probability_triangle_side_decagon_l146_146409

theorem probability_triangle_side_decagon (total_vertices : ℕ) (choose_vertices : ℕ)
  (total_triangles : ℕ) (favorable_outcomes : ℕ)
  (triangle_formula : total_vertices = 10)
  (choose_vertices_formula : choose_vertices = 3)
  (total_triangle_count_formula : total_triangles = 120)
  (favorable_outcome_count_formula : favorable_outcomes = 70)
  : (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := 
by 
  sorry

end probability_triangle_side_decagon_l146_146409


namespace person_speed_in_kmph_l146_146035

noncomputable def speed_calculation (distance_meters : ℕ) (time_minutes : ℕ) : ℝ :=
  let distance_km := (distance_meters : ℝ) / 1000
  let time_hours := (time_minutes : ℝ) / 60
  distance_km / time_hours

theorem person_speed_in_kmph :
  speed_calculation 1080 12 = 5.4 :=
by
  sorry

end person_speed_in_kmph_l146_146035


namespace original_price_l146_146699

variable (q r : ℝ)

theorem original_price (x : ℝ) (h : x * (1 + q / 100) * (1 - r / 100) = 1) :
  x = 1 / ((1 + q / 100) * (1 - r / 100)) :=
sorry

end original_price_l146_146699


namespace quadratic_condition_l146_146133

theorem quadratic_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1^2 - x2^2 = c^2 / a^2) ↔
  b^4 - c^4 = 4 * a * b^2 * c :=
sorry

end quadratic_condition_l146_146133


namespace perfectCubesCount_l146_146568

theorem perfectCubesCount (a b : Nat) (h₁ : 50 < a ∧ a ^ 3 > 50) (h₂ : b ^ 3 < 2000 ∧ b < 2000) :
  let n := b - a + 1
  n = 9 := by
  sorry

end perfectCubesCount_l146_146568


namespace solution_set_inequality_l146_146931

theorem solution_set_inequality (x : ℝ) :
  ((x + (1 / 2)) * ((3 / 2) - x) ≥ 0) ↔ (- (1 / 2) ≤ x ∧ x ≤ (3 / 2)) :=
by sorry

end solution_set_inequality_l146_146931


namespace selection_methods_l146_146587

theorem selection_methods (females males : Nat) (h_females : females = 3) (h_males : males = 2):
  females + males = 5 := 
  by 
    -- We add sorry here to skip the proof
    sorry

end selection_methods_l146_146587


namespace geometric_seq_a5_l146_146563

theorem geometric_seq_a5 : ∃ (a₁ q : ℝ), 0 < q ∧ a₁ + 2 * a₁ * q = 4 ∧ (a₁ * q^3)^2 = 4 * (a₁ * q^2) * (a₁ * q^6) ∧ (a₅ = a₁ * q^4) := 
  by
    sorry

end geometric_seq_a5_l146_146563


namespace perfect_cubes_between_100_and_900_l146_146139

theorem perfect_cubes_between_100_and_900:
  ∃ n, n = 5 ∧ (∀ k, (k ≥ 5 ∧ k ≤ 9) → (k^3 ≥ 100 ∧ k^3 ≤ 900)) :=
by
  sorry

end perfect_cubes_between_100_and_900_l146_146139


namespace average_visitors_in_30_day_month_l146_146448

def average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) : ℕ :=
    let sundays := days_in_month / 7 + if days_in_month % 7 > 0 then 1 else 0
    let other_days := days_in_month - sundays
    let total_visitors := sundays * visitors_sunday + other_days * visitors_other
    total_visitors / days_in_month

theorem average_visitors_in_30_day_month 
    (visitors_sunday : ℕ) (visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) (h1 : visitors_sunday = 660) (h2 : visitors_other = 240) (h3 : days_in_month = 30) :
    average_visitors_per_day visitors_sunday visitors_other days_in_month starts_on_sunday = 296 := 
by
  sorry

end average_visitors_in_30_day_month_l146_146448


namespace range_of_a_l146_146837

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l146_146837


namespace balanced_polygons_characterization_l146_146327

def convex_polygon (n : ℕ) (vertices : Fin n → Point) : Prop := 
  -- Definition of convex_polygon should go here
  sorry

def is_balanced (n : ℕ) (vertices : Fin n → Point) (M : Point) : Prop := 
  -- Definition of is_balanced should go here
  sorry

theorem balanced_polygons_characterization :
  ∀ (n : ℕ) (vertices : Fin n → Point) (M : Point),
  convex_polygon n vertices →
  is_balanced n vertices M →
  n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end balanced_polygons_characterization_l146_146327


namespace chores_per_week_l146_146628

theorem chores_per_week :
  ∀ (cookie_per_chore : ℕ) 
    (total_money : ℕ) 
    (cost_per_pack : ℕ) 
    (cookies_per_pack : ℕ) 
    (weeks : ℕ)
    (chores_per_week : ℕ),
  cookie_per_chore = 3 →
  total_money = 15 →
  cost_per_pack = 3 →
  cookies_per_pack = 24 →
  weeks = 10 →
  chores_per_week = (total_money / cost_per_pack * cookies_per_pack / weeks) / cookie_per_chore →
  chores_per_week = 4 :=
by
  intros cookie_per_chore total_money cost_per_pack cookies_per_pack weeks chores_per_week
  intros h1 h2 h3 h4 h5 h6
  sorry

end chores_per_week_l146_146628


namespace focus_of_parabola_l146_146217

theorem focus_of_parabola (p : ℝ) :
  (∃ p, x ^ 2 = 4 * p * y ∧ x ^ 2 = 4 * 1 * y) → (0, p) = (0, 1) :=
by
  sorry

end focus_of_parabola_l146_146217


namespace sum_of_integers_l146_146259

variable (p q r s : ℤ)

theorem sum_of_integers :
  (p - q + r = 7) →
  (q - r + s = 8) →
  (r - s + p = 4) →
  (s - p + q = 1) →
  p + q + r + s = 20 := by
  intros h1 h2 h3 h4
  sorry

end sum_of_integers_l146_146259


namespace Wendy_total_glasses_l146_146637

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end Wendy_total_glasses_l146_146637


namespace ninety_seven_squared_l146_146317

theorem ninety_seven_squared : (97 * 97 = 9409) :=
by
  sorry

end ninety_seven_squared_l146_146317


namespace find_common_difference_l146_146493

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end find_common_difference_l146_146493


namespace hyperbola_focal_length_l146_146085

theorem hyperbola_focal_length (m : ℝ) 
  (h0 : (∀ x y, x^2 / 16 - y^2 / m = 1)) 
  (h1 : (2 * Real.sqrt (16 + m) = 4 * Real.sqrt 5)) : 
  m = 4 := 
by sorry

end hyperbola_focal_length_l146_146085


namespace student_count_l146_146577

open Nat

theorem student_count :
  ∃ n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by {
  -- placeholder for the proof
  sorry
}

end student_count_l146_146577


namespace sum_of_solutions_l146_146474

theorem sum_of_solutions (a b : ℤ) (h₁ : a = -1) (h₂ : b = -4) (h₃ : ∀ x : ℝ, (16 - 4 * x - x^2 = 0 ↔ -x^2 - 4 * x + 16 = 0)) : 
  (-b / a) = 4 := 
by 
  rw [h₁, h₂]
  norm_num
  sorry

end sum_of_solutions_l146_146474


namespace handshakesCountIsCorrect_l146_146859

-- Define the number of gremlins and imps
def numGremlins : ℕ := 30
def numImps : ℕ := 20

-- Define the conditions based on the problem
def handshakesAmongGremlins : ℕ := (numGremlins * (numGremlins - 1)) / 2
def handshakesBetweenImpsAndGremlins : ℕ := numImps * numGremlins

-- Calculate the total handshakes
def totalHandshakes : ℕ := handshakesAmongGremlins + handshakesBetweenImpsAndGremlins

-- Prove that the total number of handshakes equals 1035
theorem handshakesCountIsCorrect : totalHandshakes = 1035 := by
  sorry

end handshakesCountIsCorrect_l146_146859


namespace equilateral_triangle_l146_146046

theorem equilateral_triangle {a b c : ℝ} (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c :=
by {
  sorry
}

end equilateral_triangle_l146_146046


namespace inverse_var_y_l146_146261

theorem inverse_var_y (k : ℝ) (y x : ℝ)
  (h1 : 5 * y = k / x^2)
  (h2 : y = 16) (h3 : x = 1) (h4 : k = 80) :
  y = 1 / 4 :=
by
  sorry

end inverse_var_y_l146_146261


namespace kameron_kangaroos_l146_146499

theorem kameron_kangaroos (K : ℕ) (B_now : ℕ) (rate : ℕ) (days : ℕ)
    (h1 : B_now = 20)
    (h2 : rate = 2)
    (h3 : days = 40)
    (h4 : B_now + rate * days = K) : K = 100 := by
  sorry

end kameron_kangaroos_l146_146499


namespace logarithmic_relationship_l146_146203

theorem logarithmic_relationship (a b : ℝ) (h1 : a = Real.logb 16 625) (h2 : b = Real.logb 2 25) : a = b / 2 :=
sorry

end logarithmic_relationship_l146_146203


namespace min_sum_squares_roots_l146_146069

theorem min_sum_squares_roots (m : ℝ) :
  (∃ (α β : ℝ), 2 * α^2 - 3 * α + m = 0 ∧ 2 * β^2 - 3 * β + m = 0 ∧ α ≠ β) → 
  (9 - 8 * m ≥ 0) →
  (α^2 + β^2 = (3/2)^2 - 2 * (m/2)) →
  (α^2 + β^2 = 9/8) ↔ m = 9/8 :=
by
  sorry

end min_sum_squares_roots_l146_146069


namespace greatest_int_with_gcd_3_l146_146808

theorem greatest_int_with_gcd_3 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 24 = 3) : n = 141 := by
  sorry

end greatest_int_with_gcd_3_l146_146808


namespace min_value_of_expr_l146_146813

def expr (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

theorem min_value_of_expr : ∃ x y : ℝ, expr x y = -2 / 3 :=
by
  sorry

end min_value_of_expr_l146_146813


namespace hyperbola_eccentricity_l146_146421

variables (a b e : ℝ) (F1 F2 P : ℝ × ℝ)

-- The hyperbola assumption
def hyperbola : Prop := ∃ (x y : ℝ), (x, y) = P ∧ x^2 / a^2 - y^2 / b^2 = 1
-- a > 0 and b > 0
def positive_a_b : Prop := a > 0 ∧ b > 0
-- Distance between foci
def distance_foci : Prop := dist F1 F2 = 12
-- Distance PF2
def distance_p_f2 : Prop := dist P F2 = 5
-- To be proven, eccentricity of the hyperbola
def eccentricity : Prop := e = 3 / 2

theorem hyperbola_eccentricity : hyperbola a b P ∧ positive_a_b a b ∧ distance_foci F1 F2 ∧ distance_p_f2 P F2 → eccentricity e :=
by
  sorry

end hyperbola_eccentricity_l146_146421


namespace total_items_l146_146606

theorem total_items (slices_of_bread bottles_of_milk cookies : ℕ) (h1 : slices_of_bread = 58)
  (h2 : bottles_of_milk = slices_of_bread - 18) (h3 : cookies = slices_of_bread + 27) :
  slices_of_bread + bottles_of_milk + cookies = 183 :=
by
  sorry

end total_items_l146_146606


namespace flowers_per_bouquet_l146_146038

-- Defining the problem parameters
def total_flowers : ℕ := 66
def wilted_flowers : ℕ := 10
def num_bouquets : ℕ := 7

-- The goal is to prove that the number of flowers per bouquet is 8
theorem flowers_per_bouquet :
  (total_flowers - wilted_flowers) / num_bouquets = 8 :=
by
  sorry

end flowers_per_bouquet_l146_146038


namespace distance_from_Idaho_to_Nevada_l146_146989

theorem distance_from_Idaho_to_Nevada (d1 d2 s1 s2 t total_time : ℝ) 
  (h1 : d1 = 640)
  (h2 : s1 = 80)
  (h3 : s2 = 50)
  (h4 : total_time = 19)
  (h5 : t = total_time - (d1 / s1)) :
  d2 = s2 * t :=
by
  sorry

end distance_from_Idaho_to_Nevada_l146_146989


namespace cloth_total_selling_price_l146_146815

theorem cloth_total_selling_price
    (meters : ℕ) (profit_per_meter cost_price_per_meter : ℝ) :
    meters = 92 →
    profit_per_meter = 24 →
    cost_price_per_meter = 83.5 →
    (cost_price_per_meter + profit_per_meter) * meters = 9890 :=
by
  intros
  sorry

end cloth_total_selling_price_l146_146815


namespace find_x_l146_146274

theorem find_x (x : ℝ) (h1 : (x - 1) / (x + 2) = 0) (h2 : x ≠ -2) : x = 1 :=
sorry

end find_x_l146_146274


namespace exists_same_color_points_at_distance_one_l146_146227

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end exists_same_color_points_at_distance_one_l146_146227


namespace boat_speed_proof_l146_146330

noncomputable def speed_in_still_water : ℝ := sorry -- Defined but proof skipped

def stream_speed : ℝ := 4
def distance_downstream : ℝ := 32
def distance_upstream : ℝ := 16

theorem boat_speed_proof (v : ℝ) :
  (distance_downstream / (v + stream_speed) = distance_upstream / (v - stream_speed)) →
  v = 12 :=
by
  sorry

end boat_speed_proof_l146_146330


namespace largest_of_a_b_c_d_e_l146_146939

theorem largest_of_a_b_c_d_e (a b c d e : ℝ)
  (h1 : a - 2 = b + 3)
  (h2 : a - 2 = c - 4)
  (h3 : a - 2 = d + 5)
  (h4 : a - 2 = e - 6) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end largest_of_a_b_c_d_e_l146_146939


namespace solution_set_of_3x2_minus_7x_gt_6_l146_146439

theorem solution_set_of_3x2_minus_7x_gt_6 (x : ℝ) :
  3 * x^2 - 7 * x > 6 ↔ (x < -2 / 3 ∨ x > 3) := 
by
  sorry

end solution_set_of_3x2_minus_7x_gt_6_l146_146439


namespace ratio_of_p_to_r_l146_146895

theorem ratio_of_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : r / s = 4 / 3) 
  (h3 : s / q = 1 / 8) : 
  p / r = 15 / 2 := 
by 
  sorry

end ratio_of_p_to_r_l146_146895


namespace inequality_holds_l146_146717

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (y * z)) + (y^3 / (z * x)) + (z^3 / (x * y)) ≥ x + y + z :=
by
  sorry

end inequality_holds_l146_146717


namespace solve_equation1_solve_equation2_l146_146771

-- Let x be a real number
variable {x : ℝ}

-- The first equation and its solutions
def equation1 (x : ℝ) : Prop := (x - 1) ^ 2 - 25 = 0

-- Asserting that the solutions to the first equation are x = 6 or x = -4
theorem solve_equation1 (x : ℝ) : equation1 x ↔ x = 6 ∨ x = -4 :=
by
  sorry

-- The second equation and its solution
def equation2 (x : ℝ) : Prop := (1 / 4) * (2 * x + 3) ^ 3 = 16

-- Asserting that the solution to the second equation is x = 1/2
theorem solve_equation2 (x : ℝ) : equation2 x ↔ x = 1 / 2 :=
by
  sorry

end solve_equation1_solve_equation2_l146_146771


namespace number_of_good_colorings_l146_146171

theorem number_of_good_colorings (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : 
  ∃ (good_colorings : ℕ), good_colorings = 6 * (2^n - 4 + 4 * 2^(m-2)) :=
sorry

end number_of_good_colorings_l146_146171


namespace frustum_volume_l146_146228

theorem frustum_volume (m : ℝ) (α : ℝ) (k : ℝ) : 
  m = 3/π ∧ 
  α = 43 + 40/60 + 42.2/3600 ∧ 
  k = 1 →
  frustumVolume = 0.79 := 
sorry

end frustum_volume_l146_146228


namespace find_a_div_b_l146_146391

theorem find_a_div_b (a b : ℝ) (h_distinct : a ≠ b) 
  (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 0.6 :=
by
  sorry

end find_a_div_b_l146_146391


namespace find_FC_l146_146073

theorem find_FC 
  (DC CB AD: ℝ)
  (h1 : DC = 9)
  (h2 : CB = 6)
  (h3 : AB = (1 / 3) * AD)
  (h4 : ED = (2 / 3) * AD) :
  FC = 9 :=
sorry

end find_FC_l146_146073


namespace average_inside_time_l146_146957

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end average_inside_time_l146_146957


namespace find_divisor_l146_146720

theorem find_divisor (remainder quotient dividend divisor : ℕ) 
  (h_rem : remainder = 8)
  (h_quot : quotient = 43)
  (h_div : dividend = 997)
  (h_eq : dividend = divisor * quotient + remainder) : 
  divisor = 23 :=
by
  sorry

end find_divisor_l146_146720


namespace regular_polygon_sides_l146_146117

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l146_146117


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l146_146790

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l146_146790


namespace correct_81st_in_set_s_l146_146708

def is_in_set_s (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 8 * n + 5

noncomputable def find_81st_in_set_s : ℕ :=
  8 * 80 + 5

theorem correct_81st_in_set_s : find_81st_in_set_s = 645 := by
  sorry

end correct_81st_in_set_s_l146_146708


namespace find_T_l146_146137

theorem find_T (T : ℝ) : (1 / 2) * (1 / 7) * T = (1 / 3) * (1 / 5) * 90 → T = 84 :=
by sorry

end find_T_l146_146137


namespace find_angle_A_l146_146123

theorem find_angle_A (a b : ℝ) (B A : ℝ)
  (h1 : a = 2) 
  (h2 : b = Real.sqrt 3) 
  (h3 : B = Real.pi / 3) : 
  A = Real.pi / 2 := 
sorry

end find_angle_A_l146_146123


namespace intersection_of_sets_l146_146113

def setA : Set ℝ := {x | x^2 ≤ 4 * x}
def setB : Set ℝ := {x | x < 1}

theorem intersection_of_sets : setA ∩ setB = {x | x < 1} := by
  sorry

end intersection_of_sets_l146_146113


namespace base_seven_to_base_ten_l146_146142

theorem base_seven_to_base_ten (n : ℕ) (h : n = 54231) : 
  (1 * 7^0 + 3 * 7^1 + 2 * 7^2 + 4 * 7^3 + 5 * 7^4) = 13497 :=
by
  sorry

end base_seven_to_base_ten_l146_146142


namespace find_a1_a7_l146_146835

variable {a n : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k n, a (k + n) = a k + n * d

theorem find_a1_a7 
  (a1 : ℝ) (d : ℝ)
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h1 : a 3 + a 5 = 14)
  (h2 : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := 
sorry

end find_a1_a7_l146_146835


namespace sine_sum_zero_l146_146613

open Real 

theorem sine_sum_zero (α β γ : ℝ) :
  (sin α / (sin (α - β) * sin (α - γ))
  + sin β / (sin (β - α) * sin (β - γ))
  + sin γ / (sin (γ - α) * sin (γ - β)) = 0) :=
sorry

end sine_sum_zero_l146_146613


namespace arithmetic_sequences_ratio_l146_146501

theorem arithmetic_sequences_ratio
  (a b : ℕ → ℕ)
  (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h2 : ∀ n, T n = (n * (2 * (b 1) + (n - 1) * (b 2 - b 1))) / 2)
  (h3 : ∀ n, (S n) / (T n) = (2 * n + 2) / (n + 3)) :
  (a 10) / (b 9) = 2 := sorry

end arithmetic_sequences_ratio_l146_146501


namespace arithmetic_geom_seq_l146_146081

variable {a_n : ℕ → ℝ}
variable {d a_1 : ℝ}
variable (h_seq : ∀ n, a_n n = a_1 + (n-1) * d)
variable (d_ne_zero : d ≠ 0)
variable (a_1_ne_zero : a_1 ≠ 0)
variable (geo_seq : (a_1 + d)^2 = a_1 * (a_1 + 3 * d))

theorem arithmetic_geom_seq :
  (a_1 + a_n 14) / a_n 3 = 5 := by
  sorry

end arithmetic_geom_seq_l146_146081


namespace total_baseball_cards_l146_146374

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end total_baseball_cards_l146_146374


namespace total_sampled_students_is_80_l146_146013

-- Given conditions
variables (total_students num_freshmen num_sampled_freshmen : ℕ)
variables (total_students := 2400) (num_freshmen := 600) (num_sampled_freshmen := 20)

-- Define the proportion for stratified sampling.
def stratified_sampling (total_students num_freshmen num_sampled_freshmen total_sampled_students : ℕ) : Prop :=
  num_freshmen / total_students = num_sampled_freshmen / total_sampled_students

-- State the theorem: Prove the total number of students to be sampled from the entire school is 80.
theorem total_sampled_students_is_80 : ∃ n, stratified_sampling total_students num_freshmen num_sampled_freshmen n ∧ n = 80 := 
sorry

end total_sampled_students_is_80_l146_146013


namespace technician_round_trip_l146_146614

-- Definitions based on conditions
def trip_to_center_completion : ℝ := 0.5 -- Driving to the center is 50% of the trip
def trip_from_center_completion (percent_completed: ℝ) : ℝ := 0.5 * percent_completed -- Completion percentage of the return trip
def total_trip_completion : ℝ := trip_to_center_completion + trip_from_center_completion 0.3 -- Total percentage completed

-- Theorem statement
theorem technician_round_trip : total_trip_completion = 0.65 :=
by
  sorry

end technician_round_trip_l146_146614


namespace smaller_circle_radius_l146_146413

theorem smaller_circle_radius
  (R : ℝ) (r : ℝ)
  (h1 : R = 12)
  (h2 : 7 = 7) -- This is trivial and just emphasizes the arrangement of seven congruent smaller circles
  (h3 : 4 * (2 * r) = 2 * R) : r = 3 := by
  sorry

end smaller_circle_radius_l146_146413


namespace salary_increase_l146_146599

theorem salary_increase (x : ℕ) (hB_C_sum : 2*x + 3*x = 6000) : 
  ((3 * x - 1 * x) / (1 * x) ) * 100 = 200 :=
by
  -- Placeholder for the proof
  sorry

end salary_increase_l146_146599


namespace factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l146_146175

-- Problem 1: Prove the factorization of 4x^2 - 25y^2
theorem factorize_diff_of_squares (x y : ℝ) : 4 * x^2 - 25 * y^2 = (2 * x + 5 * y) * (2 * x - 5 * y) := 
sorry

-- Problem 2: Prove the factorization of -3xy^3 + 27x^3y
theorem factorize_common_factor_diff_of_squares (x y : ℝ) : 
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3 * x) * (y - 3 * x) := 
sorry

end factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l146_146175


namespace arccos_neg_one_eq_pi_l146_146029

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := 
by
  sorry

end arccos_neg_one_eq_pi_l146_146029


namespace first_discount_percentage_l146_146679

theorem first_discount_percentage (x : ℝ) (h : 450 * (1 - x / 100) * 0.85 = 306) : x = 20 :=
sorry

end first_discount_percentage_l146_146679


namespace tank_capacity_l146_146651

theorem tank_capacity (C : ℝ) : 
  (0.5 * C = 0.9 * C - 45) → C = 112.5 :=
by
  intro h
  sorry

end tank_capacity_l146_146651


namespace scientific_notation_example_l146_146337

theorem scientific_notation_example : 0.0000037 = 3.7 * 10^(-6) :=
by
  -- We would provide the proof here.
  sorry

end scientific_notation_example_l146_146337


namespace positive_divisors_8_fact_l146_146018

-- Factorial function definition
def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Function to compute the number of divisors from prime factors
def numDivisors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

-- Known prime factorization of 8!
noncomputable def factors_8_fact : List (Nat × Nat) :=
  [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Theorem statement
theorem positive_divisors_8_fact : numDivisors factors_8_fact = 96 :=
  sorry

end positive_divisors_8_fact_l146_146018


namespace units_digit_of_3_pow_2009_l146_146697

noncomputable def units_digit (n : ℕ) : ℕ :=
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 9
  else if n % 4 = 3 then 7
  else 1

theorem units_digit_of_3_pow_2009 : units_digit (2009) = 3 :=
by
  -- Skipping the proof as instructed
  sorry

end units_digit_of_3_pow_2009_l146_146697


namespace john_payment_and_hourly_rate_l146_146634

variable (court_hours : ℕ) (prep_hours : ℕ) (upfront_fee : ℕ) 
variable (total_payment : ℕ) (brother_contribution_factor : ℕ)
variable (hourly_rate : ℚ) (john_payment : ℚ)

axiom condition1 : upfront_fee = 1000
axiom condition2 : court_hours = 50
axiom condition3 : prep_hours = 2 * court_hours
axiom condition4 : total_payment = 8000
axiom condition5 : brother_contribution_factor = 2

theorem john_payment_and_hourly_rate :
  (john_payment = total_payment / brother_contribution_factor + upfront_fee) ∧
  (hourly_rate = (total_payment - upfront_fee) / (court_hours + prep_hours)) :=
by
  sorry

end john_payment_and_hourly_rate_l146_146634


namespace chess_tournament_points_distribution_l146_146763

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l146_146763


namespace min_gumballs_to_ensure_four_same_color_l146_146886

/-- A structure to represent the number of gumballs of each color. -/
structure Gumballs :=
(red : ℕ)
(white : ℕ)
(blue : ℕ)
(green : ℕ)

def gumball_machine : Gumballs := { red := 10, white := 9, blue := 8, green := 6 }

/-- Theorem to state the minimum number of gumballs required to ensure at least four of any color. -/
theorem min_gumballs_to_ensure_four_same_color 
  (g : Gumballs) 
  (h1 : g.red = 10)
  (h2 : g.white = 9)
  (h3 : g.blue = 8)
  (h4 : g.green = 6) : 
  ∃ n, n = 13 := 
sorry

end min_gumballs_to_ensure_four_same_color_l146_146886


namespace correct_answer_is_B_l146_146164

def is_permutation_problem (desc : String) : Prop :=
  desc = "Permutation"

def check_problem_A : Prop :=
  ¬ is_permutation_problem "Selecting 2 out of 8 students to participate in a knowledge competition"

def check_problem_B : Prop :=
  is_permutation_problem "If 10 people write letters to each other once, how many letters are written in total"

def check_problem_C : Prop :=
  ¬ is_permutation_problem "There are 5 points on a plane, with no three points collinear, what is the maximum number of lines that can be determined by these 5 points"

def check_problem_D : Prop :=
  ¬ is_permutation_problem "From the numbers 1, 2, 3, 4, choose any two numbers to multiply, how many different results are there"

theorem correct_answer_is_B : check_problem_A ∧ check_problem_B ∧ check_problem_C ∧ check_problem_D → 
  ("B" = "B") := by
  sorry

end correct_answer_is_B_l146_146164


namespace sixteen_a_four_plus_one_div_a_four_l146_146737

theorem sixteen_a_four_plus_one_div_a_four (a : ℝ) (h : 2 * a - 1 / a = 3) :
  16 * a^4 + (1 / a^4) = 161 :=
sorry

end sixteen_a_four_plus_one_div_a_four_l146_146737


namespace count_non_empty_subsets_of_odd_numbers_greater_than_one_l146_146444

-- Condition definitions
def given_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_numbers_greater_than_one (s : Finset ℕ) : Finset ℕ := 
  s.filter (λ x => x % 2 = 1 ∧ x > 1)

-- The problem statement
theorem count_non_empty_subsets_of_odd_numbers_greater_than_one : 
  (odd_numbers_greater_than_one given_set).powerset.card - 1 = 15 := 
by 
  sorry

end count_non_empty_subsets_of_odd_numbers_greater_than_one_l146_146444


namespace sqrt_meaningful_l146_146114

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l146_146114


namespace leos_time_is_1230_l146_146222

theorem leos_time_is_1230
  (theo_watch_slow: Int)
  (theo_watch_fast_belief: Int)
  (leo_watch_fast: Int)
  (leo_watch_slow_belief: Int)
  (theo_thinks_time: Int):
  theo_watch_slow = 10 ∧
  theo_watch_fast_belief = 5 ∧
  leo_watch_fast = 5 ∧
  leo_watch_slow_belief = 10 ∧
  theo_thinks_time = 720
  → leo_thinks_time = 750 :=
by
  sorry

end leos_time_is_1230_l146_146222


namespace range_of_a_l146_146600

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + Real.exp (x + 2) - 2 * Real.exp 4
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * x - 3 * a * Real.exp x
def A : Set ℝ := { x | f x = 0 }
def B (a : ℝ) : Set ℝ := { x | g x a = 0 }

theorem range_of_a (a : ℝ) :
  (∃ x₁ ∈ A, ∃ x₂ ∈ B a, |x₁ - x₂| < 1) →
  a ∈ Set.Ici (1 / (3 * Real.exp 1)) ∩ Set.Iic (4 / (3 * Real.exp 4)) :=
sorry

end range_of_a_l146_146600


namespace general_term_seq_l146_146160

theorem general_term_seq 
  (a : ℕ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 5/3) 
  (h_rec : ∀ n, n > 0 → a (n + 2) = (5 / 3) * a (n + 1) - (2 / 3) * a n) : 
  ∀ n, a n = 2 - (3 / 2) * (2 / 3)^n :=
by
  sorry

end general_term_seq_l146_146160


namespace sandwiches_with_ten_loaves_l146_146797

def sandwiches_per_loaf : ℕ := 18 / 3

def num_sandwiches (loaves: ℕ) : ℕ := sandwiches_per_loaf * loaves

theorem sandwiches_with_ten_loaves :
  num_sandwiches 10 = 60 := by
  sorry

end sandwiches_with_ten_loaves_l146_146797


namespace Sam_needs_16_more_hours_l146_146212

noncomputable def Sam_hourly_rate : ℝ :=
  460 / 23

noncomputable def Sam_earnings_Sep_to_Feb : ℝ :=
  8 * Sam_hourly_rate

noncomputable def Sam_total_earnings : ℝ :=
  460 + Sam_earnings_Sep_to_Feb

noncomputable def Sam_remaining_money : ℝ :=
  Sam_total_earnings - 340

noncomputable def Sam_needed_money : ℝ :=
  600 - Sam_remaining_money

noncomputable def Sam_additional_hours_needed : ℝ :=
  Sam_needed_money / Sam_hourly_rate

theorem Sam_needs_16_more_hours : Sam_additional_hours_needed = 16 :=
by 
  sorry

end Sam_needs_16_more_hours_l146_146212


namespace complement_intersection_is_correct_l146_146934

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2}
noncomputable def B : Set ℕ := {0, 2, 5}
noncomputable def complementA := (U \ A)

theorem complement_intersection_is_correct :
  complementA ∩ B = {0, 5} :=
by
  sorry

end complement_intersection_is_correct_l146_146934


namespace find_x_l146_146527

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 71) : x = 8 :=
sorry

end find_x_l146_146527


namespace interest_rate_B_lent_to_C_l146_146195

noncomputable def principal : ℝ := 1500
noncomputable def rate_A : ℝ := 10
noncomputable def time : ℝ := 3
noncomputable def gain_B : ℝ := 67.5
noncomputable def interest_paid_by_B_to_A : ℝ := principal * rate_A * time / 100
noncomputable def interest_received_by_B_from_C : ℝ := interest_paid_by_B_to_A + gain_B
noncomputable def expected_rate : ℝ := 11.5

theorem interest_rate_B_lent_to_C :
  interest_received_by_B_from_C = principal * (expected_rate) * time / 100 := 
by
  -- the proof will go here
  sorry

end interest_rate_B_lent_to_C_l146_146195


namespace smallest_possible_perimeter_l146_146557

open Real

theorem smallest_possible_perimeter
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 + b^2 = 2016) :
  a + b + 2^3 * 3 * sqrt 14 = 48 + 2^3 * 3 * sqrt 14 :=
sorry

end smallest_possible_perimeter_l146_146557


namespace cubic_roots_real_parts_neg_l146_146087

variable {a0 a1 a2 a3 : ℝ}

theorem cubic_roots_real_parts_neg (h_same_signs : (a0 > 0 ∧ a1 > 0 ∧ a2 > 0 ∧ a3 > 0) ∨ (a0 < 0 ∧ a1 < 0 ∧ a2 < 0 ∧ a3 < 0)) 
  (h_root_condition : a1 * a2 - a0 * a3 > 0) : 
    ∀ (x : ℝ), (a0 * x^3 + a1 * x^2 + a2 * x + a3 = 0 → x < 0 ∨ (∃ (z : ℂ), z.re < 0 ∧ z.im ≠ 0 ∧ z^2 = x)) :=
sorry

end cubic_roots_real_parts_neg_l146_146087


namespace one_eighth_of_two_power_36_equals_two_power_x_l146_146894

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end one_eighth_of_two_power_36_equals_two_power_x_l146_146894


namespace exists_mod_inv_l146_146381

theorem exists_mod_inv (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h : ¬ a ∣ p) : ∃ b : ℕ, a * b ≡ 1 [MOD p] :=
by
  sorry

end exists_mod_inv_l146_146381


namespace group_B_fluctuates_less_l146_146385

-- Conditions
def mean_A : ℝ := 80
def mean_B : ℝ := 90
def variance_A : ℝ := 10
def variance_B : ℝ := 5

-- Goal
theorem group_B_fluctuates_less :
  variance_B < variance_A :=
  by
    sorry

end group_B_fluctuates_less_l146_146385


namespace subtract_29_after_46_l146_146205

theorem subtract_29_after_46 (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end subtract_29_after_46_l146_146205


namespace ratio_Polly_to_Pulsar_l146_146504

theorem ratio_Polly_to_Pulsar (P Po Pe : ℕ) (k : ℕ) (h1 : P = 10) (h2 : Po = k * P) (h3 : Pe = Po / 6) (h4 : P + Po + Pe = 45) : Po / P = 3 :=
by 
  -- Skipping the proof, but this sets up the Lean environment
  sorry

end ratio_Polly_to_Pulsar_l146_146504


namespace second_number_is_11_l146_146574

-- Define the conditions
variables (x : ℕ) (h1 : 5 * x = 55)

-- The theorem we want to prove
theorem second_number_is_11 : x = 11 :=
sorry

end second_number_is_11_l146_146574


namespace Betty_will_pay_zero_l146_146930

-- Definitions of the conditions
def Doug_age : ℕ := 40
def Alice_age (D : ℕ) : ℕ := D / 2
def Betty_age (B D A : ℕ) : Prop := B + D + A = 130
def Cost_of_pack_of_nuts (C B : ℕ) : Prop := C = 2 * B
def Decrease_rate : ℕ := 5
def New_cost (C B A : ℕ) : ℕ := max 0 (C - (B - A) * Decrease_rate)
def Total_cost (packs cost_per_pack: ℕ) : ℕ := packs * cost_per_pack

-- The main proposition
theorem Betty_will_pay_zero :
  ∃ B A C, 
    (C = 2 * B) ∧
    (A = Doug_age / 2) ∧
    (B + Doug_age + A = 130) ∧
    (Total_cost 20 (max 0 (C - (B - A) * Decrease_rate)) = 0) :=
by sorry

end Betty_will_pay_zero_l146_146930


namespace map_distance_l146_146363

variable (map_distance_km : ℚ) (map_distance_inches : ℚ) (actual_distance_km: ℚ)

theorem map_distance (h1 : actual_distance_km = 136)
                     (h2 : map_distance_inches = 42)
                     (h3 : map_distance_km = 18.307692307692307) :
  (actual_distance_km * map_distance_inches / map_distance_km = 312) :=
by sorry

end map_distance_l146_146363


namespace minimum_revenue_maximum_marginal_cost_minimum_profit_l146_146165

noncomputable def R (x : ℕ) : ℝ := x^2 + 16 / x^2 + 40
noncomputable def C (x : ℕ) : ℝ := 10 * x + 40 / x
noncomputable def MC (x : ℕ) : ℝ := C (x + 1) - C x
noncomputable def z (x : ℕ) : ℝ := R x - C x

theorem minimum_revenue :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → R x ≥ 72 :=
sorry

theorem maximum_marginal_cost :
  ∀ x : ℕ, 1 ≤ x → x ≤ 9 → MC x ≤ 86 / 9 :=
sorry

theorem minimum_profit :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → (x = 1 ∨ x = 4) → z x ≥ 7 :=
sorry

end minimum_revenue_maximum_marginal_cost_minimum_profit_l146_146165


namespace grid_to_black_probability_l146_146856

theorem grid_to_black_probability :
  let n := 16
  let p_black_after_rotation := 3 / 4
  (p_black_after_rotation ^ n) = (3 / 4) ^ 16 :=
by
  -- Proof goes here
  sorry

end grid_to_black_probability_l146_146856


namespace repeating_decimal_divisible_by_2_or_5_l146_146367

theorem repeating_decimal_divisible_by_2_or_5 
    (m n : ℕ) 
    (x : ℝ) 
    (r s : ℕ) 
    (a b k p q u : ℕ)
    (hmn_coprime : Nat.gcd m n = 1)
    (h_rep_decimal : x = (m:ℚ) / (n:ℚ))
    (h_non_repeating_part: 0 < r) :
  n % 2 = 0 ∨ n % 5 = 0 :=
sorry

end repeating_decimal_divisible_by_2_or_5_l146_146367


namespace consecutive_negatives_product_to_sum_l146_146354

theorem consecutive_negatives_product_to_sum :
  ∃ (n : ℤ), n * (n + 1) = 2184 ∧ n + (n + 1) = -95 :=
by {
  sorry
}

end consecutive_negatives_product_to_sum_l146_146354


namespace remainder_1493824_div_4_l146_146648

theorem remainder_1493824_div_4 : 1493824 % 4 = 0 :=
by
  sorry

end remainder_1493824_div_4_l146_146648


namespace point_distance_is_pm_3_l146_146465

theorem point_distance_is_pm_3 (Q : ℝ) (h : |Q - 0| = 3) : Q = 3 ∨ Q = -3 :=
sorry

end point_distance_is_pm_3_l146_146465


namespace bluegrass_percentage_l146_146177

-- Define the problem conditions
def seed_mixture_X_ryegrass_percentage : ℝ := 40
def seed_mixture_Y_ryegrass_percentage : ℝ := 25
def seed_mixture_Y_fescue_percentage : ℝ := 75
def mixture_X_Y_ryegrass_percentage : ℝ := 30
def mixture_weight_percentage_X : ℝ := 33.33333333333333

-- Prove that the percentage of bluegrass in seed mixture X is 60%
theorem bluegrass_percentage (X_ryegrass : ℝ) (Y_ryegrass : ℝ) (Y_fescue : ℝ) (mixture_ryegrass : ℝ) (weight_percentage_X : ℝ) :
  X_ryegrass = seed_mixture_X_ryegrass_percentage →
  Y_ryegrass = seed_mixture_Y_ryegrass_percentage →
  Y_fescue = seed_mixture_Y_fescue_percentage →
  mixture_ryegrass = mixture_X_Y_ryegrass_percentage →
  weight_percentage_X = mixture_weight_percentage_X →
  (100 - X_ryegrass) = 60 :=
by
  intro hX_ryegrass hY_ryegrass hY_fescue hmixture_ryegrass hweight_X
  rw [hX_ryegrass]
  sorry

end bluegrass_percentage_l146_146177


namespace max_sum_of_squares_eq_50_l146_146230

theorem max_sum_of_squares_eq_50 :
  ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 50 → x + y ≥ x' + y') ∧ x + y = 10 := 
sorry

end max_sum_of_squares_eq_50_l146_146230


namespace find_value_of_x2_plus_y2_l146_146666

theorem find_value_of_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + y^2 - 4 * x * y + 24 ≤ 10 * x - 1) : x^2 + y^2 = 125 := 
sorry

end find_value_of_x2_plus_y2_l146_146666


namespace megan_initial_acorns_l146_146214

def initial_acorns (given_away left: ℕ) : ℕ := 
  given_away + left

theorem megan_initial_acorns :
  initial_acorns 7 9 = 16 := 
by 
  unfold initial_acorns
  rfl

end megan_initial_acorns_l146_146214


namespace field_area_l146_146794

theorem field_area (x y : ℕ) (h1 : x + y = 700) (h2 : y - x = (1/5) * ((x + y) / 2)) : x = 315 :=
  sorry

end field_area_l146_146794


namespace bananas_per_truck_l146_146397

theorem bananas_per_truck (total_apples total_bananas apples_per_truck : ℝ) 
  (h_total_apples: total_apples = 132.6)
  (h_apples_per_truck: apples_per_truck = 13.26)
  (h_total_bananas: total_bananas = 6.4) :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 :=
by
  sorry

end bananas_per_truck_l146_146397


namespace largest_divisor_of_n_l146_146906

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 12 ∣ n :=
by sorry

end largest_divisor_of_n_l146_146906


namespace seashells_total_l146_146184

theorem seashells_total (s m : Nat) (hs : s = 18) (hm : m = 47) : s + m = 65 := 
by
  -- We are just specifying the theorem statement here
  sorry

end seashells_total_l146_146184


namespace approximate_probability_hit_shot_l146_146583

-- Define the data from the table
def shots : List ℕ := [10, 50, 100, 150, 200, 500, 1000, 2000]
def hits : List ℕ := [9, 40, 70, 108, 143, 361, 721, 1440]
def hit_rates : List ℚ := [0.9, 0.8, 0.7, 0.72, 0.715, 0.722, 0.721, 0.72]

-- State the theorem that the stabilized hit rate is approximately 0.72
theorem approximate_probability_hit_shot : 
  ∃ (p : ℚ), p = 0.72 ∧ 
  ∀ (n : ℕ), n ∈ [150, 200, 500, 1000, 2000] → 
     ∃ (r : ℚ), r = 0.72 ∧ 
     r = (hits.get ⟨shots.indexOf n, sorry⟩ : ℚ) / n := sorry

end approximate_probability_hit_shot_l146_146583


namespace total_pies_l146_146712

theorem total_pies {team1 team2 team3 total_pies : ℕ} 
  (h1 : team1 = 235) 
  (h2 : team2 = 275) 
  (h3 : team3 = 240) 
  (h4 : total_pies = team1 + team2 + team3) : 
  total_pies = 750 := by 
  sorry

end total_pies_l146_146712


namespace no_four_consecutive_lucky_numbers_l146_146401

def is_lucky (n : ℕ) : Prop :=
  let digits := n.digits 10
  n > 999999 ∧ n < 10000000 ∧ (∀ d ∈ digits, d ≠ 0) ∧ 
  n % (digits.foldl (λ x y => x * y) 1) = 0

theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end no_four_consecutive_lucky_numbers_l146_146401


namespace discount_price_l146_146855

theorem discount_price (original_price : ℝ) (discount_rate : ℝ) (current_price : ℝ) 
  (h1 : original_price = 120) 
  (h2 : discount_rate = 0.8) 
  (h3 : current_price = original_price * discount_rate) : 
  current_price = 96 := 
by
  sorry

end discount_price_l146_146855


namespace regular_ticket_price_l146_146023

variable (P : ℝ) -- Define the regular ticket price as a real number

-- Condition: Travis pays $1400 for his ticket after a 30% discount on a regular price P
axiom h : 0.70 * P = 1400

-- Theorem statement: Proving that the regular ticket price P equals $2000
theorem regular_ticket_price : P = 2000 :=
by 
  sorry

end regular_ticket_price_l146_146023


namespace positive_integer_solution_l146_146821

theorem positive_integer_solution (n x y : ℕ) (hn : 0 < n) (hx : 0 < x) (hy : 0 < y) :
  y ^ 2 + x * y + 3 * x = n * (x ^ 2 + x * y + 3 * y) → n = 1 :=
sorry

end positive_integer_solution_l146_146821


namespace remainder_calculation_l146_146748

theorem remainder_calculation :
  (7 * 10^23 + 3^25) % 11 = 5 :=
by
  sorry

end remainder_calculation_l146_146748
