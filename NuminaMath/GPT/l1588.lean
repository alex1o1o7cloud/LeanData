import Mathlib

namespace NUMINAMATH_GPT_modulus_of_complex_l1588_158883

open Complex

theorem modulus_of_complex : ∀ (z : ℂ), z = 3 - 2 * I → Complex.abs z = Real.sqrt 13 :=
by
  intro z
  intro h
  rw [h]
  simp [Complex.abs]
  sorry

end NUMINAMATH_GPT_modulus_of_complex_l1588_158883


namespace NUMINAMATH_GPT_average_speed_correct_l1588_158864

variable (t1 t2 : ℝ) -- time components in hours
variable (v1 v2 : ℝ) -- speed components in km/h

-- conditions
def time1 := 20 / 60 -- 20 minutes converted to hours
def time2 := 40 / 60 -- 40 minutes converted to hours
def speed1 := 60 -- speed in km/h for the first segment
def speed2 := 90 -- speed in km/h for the second segment

-- total distance traveled
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def total_distance := distance1 + distance2

-- total time taken
def total_time := time1 + time2

-- average speed
def average_speed := total_distance / total_time

-- proof statement
theorem average_speed_correct : average_speed = 80 := by
  sorry

end NUMINAMATH_GPT_average_speed_correct_l1588_158864


namespace NUMINAMATH_GPT_pq_implies_q_l1588_158839

theorem pq_implies_q (p q : Prop) (h₁ : p ∨ q) (h₂ : ¬p) : q :=
by
  sorry

end NUMINAMATH_GPT_pq_implies_q_l1588_158839


namespace NUMINAMATH_GPT_annual_interest_rate_l1588_158815

theorem annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) 
  (hP : P = 5000) 
  (hA : A = 5202) 
  (hn : n = 4) 
  (ht : t = 1 / 2)
  (compound_interest : A = P * (1 + r / n)^ (n * t)) : 
  r = 0.080392 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l1588_158815


namespace NUMINAMATH_GPT_find_ordered_pair_l1588_158872

theorem find_ordered_pair :
  ∃ x y : ℚ, 
  (x + 2 * y = (7 - x) + (7 - 2 * y)) ∧
  (3 * x - 2 * y = (x + 2) - (2 * y + 2)) ∧
  x = 0 ∧ 
  y = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l1588_158872


namespace NUMINAMATH_GPT_large_jars_count_l1588_158898

theorem large_jars_count (S L : ℕ) (h1 : S + L = 100) (h2 : S = 62) (h3 : 3 * S + 5 * L = 376) : L = 38 :=
by
  sorry

end NUMINAMATH_GPT_large_jars_count_l1588_158898


namespace NUMINAMATH_GPT_diana_statues_painted_l1588_158849

theorem diana_statues_painted :
  let paint_remaining := (1 : ℚ) / 2
  let paint_per_statue := (1 : ℚ) / 4
  (paint_remaining / paint_per_statue) = 2 :=
by
  sorry

end NUMINAMATH_GPT_diana_statues_painted_l1588_158849


namespace NUMINAMATH_GPT_total_cost_of_lollipops_l1588_158826

/-- Given Sarah bought 12 lollipops and shared one-quarter of them, 
    and Julie reimbursed Sarah 75 cents for the shared lollipops,
    Prove that the total cost of the lollipops in dollars is $3. --/
theorem total_cost_of_lollipops 
(Sarah_lollipops : ℕ) 
(shared_fraction : ℚ) 
(Julie_paid : ℚ) 
(total_lollipops_cost : ℚ)
(h1 : Sarah_lollipops = 12) 
(h2 : shared_fraction = 1/4) 
(h3 : Julie_paid = 75 / 100) 
(h4 : total_lollipops_cost = 
        ((Julie_paid / (Sarah_lollipops * shared_fraction)) * Sarah_lollipops / 100)) :
total_lollipops_cost = 3 := 
sorry

end NUMINAMATH_GPT_total_cost_of_lollipops_l1588_158826


namespace NUMINAMATH_GPT_monotone_on_interval_and_extreme_values_l1588_158812

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem monotone_on_interval_and_extreme_values :
  (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → f x1 > f x2) ∧ (f 1 = 5 ∧ f 2 = 4) := 
by
  sorry

end NUMINAMATH_GPT_monotone_on_interval_and_extreme_values_l1588_158812


namespace NUMINAMATH_GPT_remainder_8347_div_9_l1588_158878
-- Import all necessary Mathlib modules

-- Define the problem and conditions
theorem remainder_8347_div_9 : (8347 % 9) = 4 :=
by
  -- To ensure the code builds successfully and contains a placeholder for the proof
  sorry

end NUMINAMATH_GPT_remainder_8347_div_9_l1588_158878


namespace NUMINAMATH_GPT_derivative_at_neg_one_l1588_158861

variable (a b : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Given condition f'(1) = 2
axiom h : f' a b 1 = 2

-- Statement to prove f'(-1) = -2
theorem derivative_at_neg_one : f' a b (-1) = -2 :=
by 
  sorry

end NUMINAMATH_GPT_derivative_at_neg_one_l1588_158861


namespace NUMINAMATH_GPT_f_irreducible_l1588_158830

noncomputable def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem f_irreducible (n : ℕ) (hn : n > 1) : Irreducible (f n) :=
sorry

end NUMINAMATH_GPT_f_irreducible_l1588_158830


namespace NUMINAMATH_GPT_team_e_speed_l1588_158884

-- Definitions and conditions
variables (v t : ℝ)
def distance_team_e := 300 = v * t
def distance_team_a := 300 = (v + 5) * (t - 3)

-- The theorem statement: Prove that given the conditions, Team E's speed is 20 mph
theorem team_e_speed (h1 : distance_team_e v t) (h2 : distance_team_a v t) : v = 20 :=
by
  sorry -- proof steps are omitted as requested

end NUMINAMATH_GPT_team_e_speed_l1588_158884


namespace NUMINAMATH_GPT_base7_digits_l1588_158843

theorem base7_digits (D E F : ℕ) (h1 : D ≠ 0) (h2 : E ≠ 0) (h3 : F ≠ 0) (h4 : D < 7) (h5 : E < 7) (h6 : F < 7)
  (h_diff1 : D ≠ E) (h_diff2 : D ≠ F) (h_diff3 : E ≠ F)
  (h_eq : (49 * D + 7 * E + F) + (49 * E + 7 * F + D) + (49 * F + 7 * D + E) = 400 * D) :
  E + F = 6 :=
by
  sorry

end NUMINAMATH_GPT_base7_digits_l1588_158843


namespace NUMINAMATH_GPT_cards_net_cost_equivalence_l1588_158828

-- Define the purchase amount
def purchase_amount : ℝ := 10000

-- Define cashback percentages
def debit_card_cashback : ℝ := 0.01
def credit_card_cashback : ℝ := 0.005

-- Define interest rate for keeping money in the debit account
def interest_rate : ℝ := 0.005

-- A function to calculate the net cost after 1 month using the debit card
def net_cost_debit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage

-- A function to calculate the net cost after 1 month using the credit card
def net_cost_credit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) (interest_rate : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage - purchase_amount * interest_rate

-- Final theorem stating that the net cost using both cards is the same
theorem cards_net_cost_equivalence : 
  net_cost_debit_card purchase_amount debit_card_cashback = 
  net_cost_credit_card purchase_amount credit_card_cashback interest_rate :=
by
  sorry

end NUMINAMATH_GPT_cards_net_cost_equivalence_l1588_158828


namespace NUMINAMATH_GPT_average_difference_l1588_158801

-- Definitions for the conditions
def set1 : List ℕ := [20, 40, 60]
def set2 : List ℕ := [10, 60, 35]

-- Function to compute the average of a list of numbers
def average (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

-- The main theorem to prove the difference between the averages is 5
theorem average_difference : average set1 - average set2 = 5 := by
  sorry

end NUMINAMATH_GPT_average_difference_l1588_158801


namespace NUMINAMATH_GPT_difference_in_squares_l1588_158877

noncomputable def radius_of_circle (x y h R : ℝ) : Prop :=
  5 * x^2 - 4 * x * h + h^2 = R^2 ∧ 5 * y^2 + 4 * y * h + h^2 = R^2

theorem difference_in_squares (x y h R : ℝ) (h_radius : radius_of_circle x y h R) :
  2 * x - 2 * y = (8/5 : ℝ) * h :=
by
  sorry

end NUMINAMATH_GPT_difference_in_squares_l1588_158877


namespace NUMINAMATH_GPT_tan_sum_pi_over_4_x_l1588_158846

theorem tan_sum_pi_over_4_x (x : ℝ) (h1 : x > -π/2 ∧ x < 0) (h2 : Real.cos x = 4/5) :
  Real.tan (π/4 + x) = 1/7 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_pi_over_4_x_l1588_158846


namespace NUMINAMATH_GPT_minimum_sum_am_gm_l1588_158857

theorem minimum_sum_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / 2) :=
sorry

end NUMINAMATH_GPT_minimum_sum_am_gm_l1588_158857


namespace NUMINAMATH_GPT_calculate_expression_l1588_158856

theorem calculate_expression :
  (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1588_158856


namespace NUMINAMATH_GPT_find_x_l1588_158865

theorem find_x (x : ℚ) (h : (3 * x - 7) / 4 = 15) : x = 67 / 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1588_158865


namespace NUMINAMATH_GPT_perpendicular_condition_parallel_condition_opposite_direction_l1588_158885

/-- Conditions definitions --/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def k_vector_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
def vector_a_minus_3b : ℝ × ℝ := (10, -4)

/-- Problem 1: Prove the perpendicular condition --/
theorem perpendicular_condition (k : ℝ) : (k_vector_a_plus_b k).fst * vector_a_minus_3b.fst + (k_vector_a_plus_b k).snd * vector_a_minus_3b.snd = 0 → k = 19 :=
by
  sorry

/-- Problem 2: Prove the parallel condition --/
theorem parallel_condition (k : ℝ) : (-(k - 3) / 10 = (2 * k + 2) / (-4)) → k = -1/3 :=
by
  sorry

/-- Determine if the vectors are in opposite directions --/
theorem opposite_direction (k : ℝ) (hk : k = -1/3) : k_vector_a_plus_b k = (-(1/3):ℝ) • vector_a_minus_3b :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_parallel_condition_opposite_direction_l1588_158885


namespace NUMINAMATH_GPT_hannah_trip_time_ratio_l1588_158807

theorem hannah_trip_time_ratio 
  (u : ℝ) -- Speed on the first trip in miles per hour.
  (u_pos : u > 0) -- Speed should be positive.
  (t1 t2 : ℝ) -- Time taken for the first and second trip respectively.
  (h_t1 : t1 = 30 / u) -- Time for the first trip.
  (h_t2 : t2 = 150 / (4 * u)) -- Time for the second trip.
  : t2 / t1 = 1.25 := by
  sorry

end NUMINAMATH_GPT_hannah_trip_time_ratio_l1588_158807


namespace NUMINAMATH_GPT_solutions_count_l1588_158879

noncomputable def number_of_solutions (x y z : ℚ) : ℕ :=
if (x^2 - y * z = 1) ∧ (y^2 - x * z = 1) ∧ (z^2 - x * y = 1)
then 6
else 0

theorem solutions_count : number_of_solutions x y z = 6 :=
sorry

end NUMINAMATH_GPT_solutions_count_l1588_158879


namespace NUMINAMATH_GPT_contingency_table_confidence_l1588_158805

theorem contingency_table_confidence (k_squared : ℝ) (h1 : k_squared = 4.013) : 
  confidence_99 :=
  sorry

end NUMINAMATH_GPT_contingency_table_confidence_l1588_158805


namespace NUMINAMATH_GPT_find_angle_x_l1588_158899

-- Define the angles and parallel lines conditions
def parallel_lines (k l : Prop) (angle1 : Real) (angle2 : Real) : Prop :=
  k ∧ l ∧ angle1 = 30 ∧ angle2 = 90

-- Statement of the problem in Lean syntax
theorem find_angle_x (k l : Prop) (angle1 angle2 : Real) (x : Real) : 
  parallel_lines k l angle1 angle2 → x = 150 :=
by
  -- Assuming conditions are given, prove x = 150
  sorry

end NUMINAMATH_GPT_find_angle_x_l1588_158899


namespace NUMINAMATH_GPT_min_b_over_a_l1588_158855

theorem min_b_over_a (a b : ℝ) (h : ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x ≥ 0) : b / a ≥ 1 := by
  sorry

end NUMINAMATH_GPT_min_b_over_a_l1588_158855


namespace NUMINAMATH_GPT_pairs_divisible_by_three_l1588_158819

theorem pairs_divisible_by_three (P T : ℕ) (h : 5 * P = 3 * T) : ∃ k : ℕ, P = 3 * k := 
sorry

end NUMINAMATH_GPT_pairs_divisible_by_three_l1588_158819


namespace NUMINAMATH_GPT_product_of_possible_values_l1588_158844

theorem product_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 18) : ∃ a b, x = a ∨ x = b ∧ a * b = -30 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_possible_values_l1588_158844


namespace NUMINAMATH_GPT_planes_parallel_l1588_158833

variables (α β : Type)
variables (n : ℝ → ℝ → ℝ → Prop) (u v : ℝ × ℝ × ℝ)

-- Conditions: 
def normal_vector_plane_alpha (u : ℝ × ℝ × ℝ) := u = (1, 2, -1)
def normal_vector_plane_beta (v : ℝ × ℝ × ℝ) := v = (-3, -6, 3)

-- Proof Problem: Prove that alpha is parallel to beta
theorem planes_parallel (h1 : normal_vector_plane_alpha u)
                        (h2 : normal_vector_plane_beta v) :
  v = -3 • u :=
by sorry

end NUMINAMATH_GPT_planes_parallel_l1588_158833


namespace NUMINAMATH_GPT_inequality_solution_l1588_158851

theorem inequality_solution 
  (a x : ℝ) : 
  (a = 2 ∨ a = -2 → x > 1 / 4) ∧ 
  (a > 2 → x > 1 / (a + 2) ∨ x < 1 / (2 - a)) ∧ 
  (a < -2 → x < 1 / (a + 2) ∨ x > 1 / (2 - a)) ∧ 
  (-2 < a ∧ a < 2 → 1 / (a + 2) < x ∧ x < 1 / (2 - a)) 
  :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1588_158851


namespace NUMINAMATH_GPT_solve_ellipse_correct_m_l1588_158842

noncomputable def ellipse_is_correct_m : Prop :=
  ∃ (m : ℝ), 
    (m > 6) ∧
    ((m - 2) - (10 - m) = 4) ∧
    (m = 8)

theorem solve_ellipse_correct_m : ellipse_is_correct_m :=
sorry

end NUMINAMATH_GPT_solve_ellipse_correct_m_l1588_158842


namespace NUMINAMATH_GPT_shipment_cost_l1588_158873

-- Define the conditions
def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def shipping_cost_per_crate : ℝ := 1.5
def surcharge_per_crate : ℝ := 0.5
def flat_fee : ℝ := 10

-- Define the question as a theorem
theorem shipment_cost : 
  let crates := total_weight / weight_per_crate
  let cost_per_crate := shipping_cost_per_crate + surcharge_per_crate
  let total_cost_crates := crates * cost_per_crate
  let total_cost := total_cost_crates + flat_fee
  total_cost = 46 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_shipment_cost_l1588_158873


namespace NUMINAMATH_GPT_count_multiples_of_5_l1588_158806

theorem count_multiples_of_5 (a b : ℕ) (h₁ : 50 ≤ a) (h₂ : a ≤ 300) (h₃ : 50 ≤ b) (h₄ : b ≤ 300) (h₅ : a % 5 = 0) (h₆ : b % 5 = 0) 
  (h₇ : ∀ n : ℕ, 50 ≤ n ∧ n ≤ 300 → n % 5 = 0 → a ≤ n ∧ n ≤ b) :
  b = a + 48 * 5 → (b - a) / 5 + 1 = 49 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_5_l1588_158806


namespace NUMINAMATH_GPT_number_of_cakes_l1588_158893

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cakes_l1588_158893


namespace NUMINAMATH_GPT_base_rate_first_company_proof_l1588_158813

noncomputable def base_rate_first_company : ℝ := 8.00
def charge_per_minute_first_company : ℝ := 0.25
def base_rate_second_company : ℝ := 12.00
def charge_per_minute_second_company : ℝ := 0.20
def minutes : ℕ := 80

theorem base_rate_first_company_proof :
  base_rate_first_company = 8.00 :=
sorry

end NUMINAMATH_GPT_base_rate_first_company_proof_l1588_158813


namespace NUMINAMATH_GPT_binom_18_6_mul_smallest_prime_gt_10_eq_80080_l1588_158891

theorem binom_18_6_mul_smallest_prime_gt_10_eq_80080 :
  (Nat.choose 18 6) * 11 = 80080 := sorry

end NUMINAMATH_GPT_binom_18_6_mul_smallest_prime_gt_10_eq_80080_l1588_158891


namespace NUMINAMATH_GPT_unique_k_linear_equation_l1588_158811

theorem unique_k_linear_equation :
  (∀ x y k : ℝ, (2 : ℝ) * x^|k| + (k - 1) * y = 3 → (|k| = 1 ∧ k ≠ 1) → k = -1) :=
by
  sorry

end NUMINAMATH_GPT_unique_k_linear_equation_l1588_158811


namespace NUMINAMATH_GPT_response_activity_solutions_l1588_158804

theorem response_activity_solutions (x y z : ℕ) :
  5 * x + 4 * y + 3 * z = 15 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_response_activity_solutions_l1588_158804


namespace NUMINAMATH_GPT_extreme_values_l1588_158870

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 3

theorem extreme_values (a b : ℝ) : 
  (f a b (-1) = 10) ∧ (f a b 2 = -17) →
  (6 * (-1)^2 + 2 * a * (-1) + b = 0) ∧ (6 * 2^2 + 2 * (a * 2) + b = 0) →
  a = -3 ∧ b = -12 :=
by 
  sorry

end NUMINAMATH_GPT_extreme_values_l1588_158870


namespace NUMINAMATH_GPT_sam_gave_fraction_l1588_158890

/-- Given that Mary bought 1500 stickers and shared them between Susan, Andrew, 
and Sam in the ratio 1:1:3. After Sam gave some stickers to Andrew, Andrew now 
has 900 stickers. Prove that the fraction of Sam's stickers given to Andrew is 2/3. -/
theorem sam_gave_fraction (total_stickers : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
    (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ) (final_B : ℕ) (given_stickers : ℕ) :
    total_stickers = 1500 → ratio_A = 1 → ratio_B = 1 → ratio_C = 3 →
    initial_A = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_B = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_C = 3 * (total_stickers / (ratio_A + ratio_B + ratio_C)) →
    final_B = 900 →
    initial_B + given_stickers = final_B →
    given_stickers / initial_C = 2 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sam_gave_fraction_l1588_158890


namespace NUMINAMATH_GPT_max_product_is_2331_l1588_158882

open Nat

noncomputable def max_product (a b : ℕ) : ℕ :=
  if a + b = 100 ∧ a % 5 = 2 ∧ b % 6 = 3 then a * b else 0

theorem max_product_is_2331 (a b : ℕ) (h_sum : a + b = 100) (h_mod_a : a % 5 = 2) (h_mod_b : b % 6 = 3) :
  max_product a b = 2331 :=
  sorry

end NUMINAMATH_GPT_max_product_is_2331_l1588_158882


namespace NUMINAMATH_GPT_exists_square_no_visible_points_l1588_158859

-- Define visibility from the origin
def visible_from_origin (x y : ℤ) : Prop :=
  Int.gcd x y = 1

-- Main theorem statement
theorem exists_square_no_visible_points (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 
    (∀ (x y : ℤ), a ≤ x ∧ x ≤ a + n ∧ b ≤ y ∧ y ≤ b + n ∧ (x ≠ 0 ∨ y ≠ 0) → ¬visible_from_origin x y) :=
sorry

end NUMINAMATH_GPT_exists_square_no_visible_points_l1588_158859


namespace NUMINAMATH_GPT_find_a_l1588_158897

variable (a x : ℝ)

noncomputable def curve1 (x : ℝ) := x + Real.log x
noncomputable def curve2 (a x : ℝ) := a * x^2 + (a + 2) * x + 1

theorem find_a : (curve1 1 = 1 ∧ curve1 1 = curve2 a 1) → a = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1588_158897


namespace NUMINAMATH_GPT_problem1_l1588_158825

theorem problem1 (x y : ℝ) (h : |x + 1| + (2 * x - y)^2 = 0) : x^2 - y = 3 :=
sorry

end NUMINAMATH_GPT_problem1_l1588_158825


namespace NUMINAMATH_GPT_tens_digit_of_23_pow_2023_l1588_158886

theorem tens_digit_of_23_pow_2023 : (23 ^ 2023 % 100 / 10) = 6 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_23_pow_2023_l1588_158886


namespace NUMINAMATH_GPT_sum_of_numbers_l1588_158845

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1588_158845


namespace NUMINAMATH_GPT_tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l1588_158821

theorem tanx_eq_2_sin2cos2 (x : ℝ) (h : Real.tan x = 2) : 
  (2 / 3) * (Real.sin x) ^ 2 + (1 / 4) * (Real.cos x) ^ 2 = 7 / 12 := 
by 
  sorry

theorem tanx_eq_2_cos_sin_ratio (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3 := 
by 
  sorry

end NUMINAMATH_GPT_tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l1588_158821


namespace NUMINAMATH_GPT_triceratops_count_l1588_158876

theorem triceratops_count (r t : ℕ) 
  (h_legs : 4 * r + 4 * t = 48) 
  (h_horns : 2 * r + 3 * t = 31) : 
  t = 7 := 
by 
  hint

/- The given conditions are:
1. Each rhinoceros has 2 horns.
2. Each triceratops has 3 horns.
3. Each animal has 4 legs.
4. There is a total of 31 horns.
5. There is a total of 48 legs.

Using these conditions and the equations derived from them, we need to prove that the number of triceratopses (t) is 7.
-/

end NUMINAMATH_GPT_triceratops_count_l1588_158876


namespace NUMINAMATH_GPT_boys_bound_l1588_158817

open Nat

noncomputable def num_students := 1650
noncomputable def num_rows := 22
noncomputable def num_cols := 75
noncomputable def max_pairs_same_sex := 11

-- Assume we have a function that gives the number of boys.
axiom number_of_boys : ℕ
axiom col_pairs_property : ∀ (c1 c2 : ℕ), ∀ (r : ℕ), c1 ≠ c2 → r ≤ num_rows → 
  (number_of_boys ≤ max_pairs_same_sex)

theorem boys_bound : number_of_boys ≤ 920 :=
sorry

end NUMINAMATH_GPT_boys_bound_l1588_158817


namespace NUMINAMATH_GPT_initial_people_employed_l1588_158888

-- Definitions from the conditions
def initial_work_days : ℕ := 25
def total_work_days : ℕ := 50
def work_done_percentage : ℕ := 40
def additional_people : ℕ := 30

-- Defining the statement to be proved
theorem initial_people_employed (P : ℕ) 
  (h1 : initial_work_days = 25) 
  (h2 : total_work_days = 50)
  (h3 : work_done_percentage = 40)
  (h4 : additional_people = 30) 
  (work_remaining_percentage := 60) : 
  (P * 25 / 10 = 100) -> (P + 30) * 50 = P * 625 / 10 -> P = 120 :=
by
  sorry

end NUMINAMATH_GPT_initial_people_employed_l1588_158888


namespace NUMINAMATH_GPT_ratio_of_areas_l1588_158836

theorem ratio_of_areas (AB BC O : ℝ) (h_diameter : AB = 4) (h_BC : BC = 3)
  (ABD DBE ABDeqDBE : Prop) (x y : ℝ) 
  (h_area_ABCD : x = 7 * y) :
  (x / y) = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1588_158836


namespace NUMINAMATH_GPT_nearest_integer_to_a_plus_b_l1588_158816

theorem nearest_integer_to_a_plus_b
  (a b : ℝ)
  (h1 : |a| + b = 5)
  (h2 : |a| * b + a^3 = -8) :
  abs (a + b - 3) ≤ 0.5 :=
sorry

end NUMINAMATH_GPT_nearest_integer_to_a_plus_b_l1588_158816


namespace NUMINAMATH_GPT_cos_alpha_plus_two_pi_over_three_l1588_158847

theorem cos_alpha_plus_two_pi_over_three (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α + 2 * π / 3) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_two_pi_over_three_l1588_158847


namespace NUMINAMATH_GPT_min_band_members_exists_l1588_158840

theorem min_band_members_exists (n : ℕ) :
  (∃ n, (∃ k : ℕ, n = 9 * k) ∧ (∃ m : ℕ, n = 10 * m) ∧ (∃ p : ℕ, n = 11 * p)) → n = 990 :=
by
  sorry

end NUMINAMATH_GPT_min_band_members_exists_l1588_158840


namespace NUMINAMATH_GPT_naomi_stickers_l1588_158810

theorem naomi_stickers :
  ∃ S : ℕ, S > 1 ∧
    (S % 5 = 2) ∧
    (S % 9 = 2) ∧
    (S % 11 = 2) ∧
    S = 497 :=
by
  sorry

end NUMINAMATH_GPT_naomi_stickers_l1588_158810


namespace NUMINAMATH_GPT_katie_bead_necklaces_l1588_158803

theorem katie_bead_necklaces (B : ℕ) (gemstone_necklaces : ℕ := 3) (cost_each_necklace : ℕ := 3) (total_earnings : ℕ := 21) :
  gemstone_necklaces * cost_each_necklace + B * cost_each_necklace = total_earnings → B = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_katie_bead_necklaces_l1588_158803


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l1588_158837

theorem equation_one_solution (x : ℝ) (h : 2 * (2 - x) - 5 * (2 - x) = 9) : x = 5 :=
sorry

theorem equation_two_solution (x : ℝ) (h : x / 3 - (3 * x - 1) / 6 = 1) : x = -5 :=
sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l1588_158837


namespace NUMINAMATH_GPT_smallest_five_digit_perfect_square_and_cube_l1588_158889

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ n = 15625 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_perfect_square_and_cube_l1588_158889


namespace NUMINAMATH_GPT_value_of_power_l1588_158852

theorem value_of_power (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2014 = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_power_l1588_158852


namespace NUMINAMATH_GPT_four_distinct_numbers_are_prime_l1588_158887

-- Lean 4 statement proving the conditions
theorem four_distinct_numbers_are_prime : 
  ∃ (a b c d : ℕ), 
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧ 
    (Prime (a * b + c * d)) ∧ 
    (Prime (a * c + b * d)) ∧ 
    (Prime (a * d + b * c)) := 
sorry

end NUMINAMATH_GPT_four_distinct_numbers_are_prime_l1588_158887


namespace NUMINAMATH_GPT_nth_derivative_correct_l1588_158880

noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (3 * x + 1) + Real.cos (5 * x)

noncomputable def n_th_derivative (n : ℕ) (x : ℝ) : ℝ :=
  3^n * Real.sin ((3 * Real.pi / 2) * n + 3 * x + 1) + 5^n * Real.cos ((3 * Real.pi / 2) * n + 5 * x)

theorem nth_derivative_correct (x : ℝ) (n : ℕ) :
  derivative^[n] y x = n_th_derivative n x :=
by
  sorry

end NUMINAMATH_GPT_nth_derivative_correct_l1588_158880


namespace NUMINAMATH_GPT_count_solutions_l1588_158867

noncomputable def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := 2 * x + 5 * y = 10
  let eq2 (x y : ℝ) := abs (abs (x + 1) - abs (y - 1)) = 1
  sorry

theorem count_solutions : num_solutions = 2 := by
  sorry

end NUMINAMATH_GPT_count_solutions_l1588_158867


namespace NUMINAMATH_GPT_students_not_taking_math_or_physics_l1588_158881

theorem students_not_taking_math_or_physics (total_students math_students phys_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 75)
  (h3 : phys_students = 50)
  (h4 : both_students = 15) :
  total_students - (math_students + phys_students - both_students) = 10 :=
by
  sorry

end NUMINAMATH_GPT_students_not_taking_math_or_physics_l1588_158881


namespace NUMINAMATH_GPT_crushing_load_example_l1588_158834

noncomputable def crushing_load (T H : ℝ) : ℝ :=
  (30 * T^5) / H^3

theorem crushing_load_example : crushing_load 5 10 = 93.75 := by
  sorry

end NUMINAMATH_GPT_crushing_load_example_l1588_158834


namespace NUMINAMATH_GPT_sequence_term_l1588_158823

noncomputable def S (n : ℕ) : ℤ := n^2 - 3 * n

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  ∃ a : ℕ → ℤ, a n = 2 * n - 4 := 
  sorry

end NUMINAMATH_GPT_sequence_term_l1588_158823


namespace NUMINAMATH_GPT_original_price_per_kg_l1588_158808

theorem original_price_per_kg (P : ℝ) (S : ℝ) (reduced_price : ℝ := 0.8 * P) (total_cost : ℝ := 400) (extra_salt : ℝ := 10) :
  S * P = total_cost ∧ (S + extra_salt) * reduced_price = total_cost → P = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_original_price_per_kg_l1588_158808


namespace NUMINAMATH_GPT_light_ray_total_distance_l1588_158895

theorem light_ray_total_distance 
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (M_eq : M = (2, 1))
  (N_eq : N = (4, 5)) :
  dist M N = 2 * Real.sqrt 10 := 
sorry

end NUMINAMATH_GPT_light_ray_total_distance_l1588_158895


namespace NUMINAMATH_GPT_map_scale_l1588_158824

theorem map_scale (cm12_km90 : 12 * (1 / 90) = 1) : 20 * (90 / 12) = 150 :=
by
  sorry

end NUMINAMATH_GPT_map_scale_l1588_158824


namespace NUMINAMATH_GPT_additional_men_joined_l1588_158838

theorem additional_men_joined (men_initial : ℕ) (days_initial : ℕ)
  (days_new : ℕ) (additional_men : ℕ) :
  men_initial = 600 →
  days_initial = 20 →
  days_new = 15 →
  (men_initial * days_initial) = ((men_initial + additional_men) * days_new) →
  additional_men = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_additional_men_joined_l1588_158838


namespace NUMINAMATH_GPT_maximum_value_of_A_l1588_158868

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_A_l1588_158868


namespace NUMINAMATH_GPT_symmetric_pattern_count_l1588_158850

noncomputable def number_of_symmetric_patterns (n : ℕ) : ℕ :=
  let regions := 12
  let total_patterns := 2^regions
  total_patterns - 2

theorem symmetric_pattern_count : number_of_symmetric_patterns 8 = 4094 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_pattern_count_l1588_158850


namespace NUMINAMATH_GPT_inequality_proof_l1588_158863

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a / (b^2 * (c + 1))) + (b / (c^2 * (a + 1))) + (c / (a^2 * (b + 1))) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1588_158863


namespace NUMINAMATH_GPT_cube_volume_proof_l1588_158841

-- Define the conditions
def len_inch : ℕ := 48
def width_inch : ℕ := 72
def total_surface_area_inch : ℕ := len_inch * width_inch
def num_faces : ℕ := 6
def area_one_face_inch : ℕ := total_surface_area_inch / num_faces
def inches_to_feet (length_in_inches : ℕ) : ℕ := length_in_inches / 12

-- Define the key elements of the proof problem
def side_length_inch : ℕ := Int.natAbs (Nat.sqrt area_one_face_inch)
def side_length_ft : ℕ := inches_to_feet side_length_inch
def volume_ft3 : ℕ := side_length_ft ^ 3

-- State the proof problem
theorem cube_volume_proof : volume_ft3 = 8 := by
  -- The proof would be implemented here
  sorry

end NUMINAMATH_GPT_cube_volume_proof_l1588_158841


namespace NUMINAMATH_GPT_value_taken_away_l1588_158814

theorem value_taken_away (n x : ℕ) (h1 : n = 4) (h2 : 2 * n + 20 = 8 * n - x) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_taken_away_l1588_158814


namespace NUMINAMATH_GPT_william_ends_with_18_tickets_l1588_158835

-- Define the initial number of tickets
def initialTickets : ℕ := 15

-- Define the tickets bought
def ticketsBought : ℕ := 3

-- Prove the total number of tickets William ends with
theorem william_ends_with_18_tickets : initialTickets + ticketsBought = 18 := by
  sorry

end NUMINAMATH_GPT_william_ends_with_18_tickets_l1588_158835


namespace NUMINAMATH_GPT_length_of_new_section_l1588_158866

-- Definitions from the conditions
def area : ℕ := 35
def width : ℕ := 7

-- The problem statement
theorem length_of_new_section (h : area = 35 ∧ width = 7) : 35 / 7 = 5 :=
by
  -- We'll provide the proof later
  sorry

end NUMINAMATH_GPT_length_of_new_section_l1588_158866


namespace NUMINAMATH_GPT_find_m_l1588_158820

variables {m : ℝ}
def vec_a : ℝ × ℝ := (-2, 3)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) (h : perpendicular vec_a (vec_b m)) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1588_158820


namespace NUMINAMATH_GPT_compute_expression_l1588_158854

-- Define the conditions
variables (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0)

-- State the theorem to be proved
theorem compute_expression (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := 
sorry

end NUMINAMATH_GPT_compute_expression_l1588_158854


namespace NUMINAMATH_GPT_no_perfect_square_in_range_l1588_158871

def f (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 2

theorem no_perfect_square_in_range : ∀ (n : ℕ), 5 ≤ n → n ≤ 15 → ¬ ∃ (m : ℕ), f n = m^2 := by
  intros n h1 h2
  sorry

end NUMINAMATH_GPT_no_perfect_square_in_range_l1588_158871


namespace NUMINAMATH_GPT_overall_profit_is_600_l1588_158869

def grinder_cp := 15000
def mobile_cp := 10000
def laptop_cp := 20000
def camera_cp := 12000

def grinder_loss_percent := 4 / 100
def mobile_profit_percent := 10 / 100
def laptop_loss_percent := 8 / 100
def camera_profit_percent := 15 / 100

def grinder_sp := grinder_cp * (1 - grinder_loss_percent)
def mobile_sp := mobile_cp * (1 + mobile_profit_percent)
def laptop_sp := laptop_cp * (1 - laptop_loss_percent)
def camera_sp := camera_cp * (1 + camera_profit_percent)

def total_cp := grinder_cp + mobile_cp + laptop_cp + camera_cp
def total_sp := grinder_sp + mobile_sp + laptop_sp + camera_sp

def overall_profit_or_loss := total_sp - total_cp

theorem overall_profit_is_600 : overall_profit_or_loss = 600 := by
  sorry

end NUMINAMATH_GPT_overall_profit_is_600_l1588_158869


namespace NUMINAMATH_GPT_cost_per_kg_mixture_l1588_158832

variables (C1 C2 R Cm : ℝ)

-- Statement of the proof problem
theorem cost_per_kg_mixture :
  C1 = 6 → C2 = 8.75 → R = 5 / 6 → Cm = C1 * R + C2 * (1 - R) → Cm = 6.458333333333333 :=
by intros hC1 hC2 hR hCm; sorry

end NUMINAMATH_GPT_cost_per_kg_mixture_l1588_158832


namespace NUMINAMATH_GPT_Vlad_score_l1588_158827

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end NUMINAMATH_GPT_Vlad_score_l1588_158827


namespace NUMINAMATH_GPT_discount_is_five_l1588_158892
-- Importing the needed Lean Math library

-- Defining the problem conditions
def costPrice : ℝ := 100
def profit_percent_with_discount : ℝ := 0.2
def profit_percent_without_discount : ℝ := 0.25

-- Calculating the respective selling prices
def sellingPrice_with_discount := costPrice * (1 + profit_percent_with_discount)
def sellingPrice_without_discount := costPrice * (1 + profit_percent_without_discount)

-- Calculating the discount 
def calculated_discount := sellingPrice_without_discount - sellingPrice_with_discount

-- Proving that the discount is $5
theorem discount_is_five : calculated_discount = 5 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_discount_is_five_l1588_158892


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_l1588_158818

theorem inequality_holds_for_all_real (k : ℝ) :
  (∀ x : ℝ, k * x ^ 2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_l1588_158818


namespace NUMINAMATH_GPT_find_side_length_a_l1588_158853

noncomputable def length_of_a (A B : ℝ) (b : ℝ) : ℝ :=
  b * Real.sin A / Real.sin B

theorem find_side_length_a :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = Real.pi / 3 → B = Real.pi / 4 → b = Real.sqrt 6 →
  a = length_of_a A B b →
  a = 3 :=
by
  intros a b c A B C hA hB hb ha
  rw [hA, hB, hb] at ha
  sorry

end NUMINAMATH_GPT_find_side_length_a_l1588_158853


namespace NUMINAMATH_GPT_sequence_arithmetic_progression_l1588_158802

theorem sequence_arithmetic_progression (b : ℕ → ℕ) (b1_eq : b 1 = 1) (recurrence : ∀ n, b (n + 2) = b (n + 1) * b n + 1) : b 2 = 1 ↔ 
  ∃ d : ℕ, ∀ n, b (n + 1) - b n = d :=
sorry

end NUMINAMATH_GPT_sequence_arithmetic_progression_l1588_158802


namespace NUMINAMATH_GPT_milk_problem_l1588_158858

theorem milk_problem (x : ℕ) (hx : 0 < x)
    (total_cost_wednesday : 10 = x * (10 / x))
    (price_reduced : ∀ x, 0.5 = (10 / x - (10 / x) + 0.5))
    (extra_bags : 2 = (x + 2) - x)
    (extra_cost : 2 + 10 = x * (10 / x) + 2) :
    x^2 + 6 * x - 40 = 0 := by
  sorry

end NUMINAMATH_GPT_milk_problem_l1588_158858


namespace NUMINAMATH_GPT_inequality1_inequality2_l1588_158862

variables (Γ B P : ℕ)

def convex_polyhedron : Prop :=
  Γ - B + P = 2

theorem inequality1 (h : convex_polyhedron Γ B P) : 
  3 * Γ ≥ 6 + P :=
sorry

theorem inequality2 (h : convex_polyhedron Γ B P) : 
  3 * B ≥ 6 + P :=
sorry

end NUMINAMATH_GPT_inequality1_inequality2_l1588_158862


namespace NUMINAMATH_GPT_find_a_b_sum_l1588_158822

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_sum_l1588_158822


namespace NUMINAMATH_GPT_average_distance_per_day_l1588_158896

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_distance_per_day :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_distance_per_day_l1588_158896


namespace NUMINAMATH_GPT_reciprocal_inequality_l1588_158875

theorem reciprocal_inequality {a b c : ℝ} (hab : a < b) (hbc : b < c) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  (1 / a) < (1 / b) :=
sorry

end NUMINAMATH_GPT_reciprocal_inequality_l1588_158875


namespace NUMINAMATH_GPT_can_be_divided_into_two_triangles_l1588_158860

-- Definitions and properties of geometrical shapes
def is_triangle (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 3 ∧ vertices = 3

def is_pentagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 5 ∧ vertices = 5

def is_hexagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 6 ∧ vertices = 6

def is_heptagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 7 ∧ vertices = 7

-- The theorem we need to prove
theorem can_be_divided_into_two_triangles :
  ∀ sides vertices,
  (is_pentagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_hexagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_heptagon sides vertices → ¬ (is_triangle sides vertices ∧ is_triangle sides vertices)) :=
by sorry

end NUMINAMATH_GPT_can_be_divided_into_two_triangles_l1588_158860


namespace NUMINAMATH_GPT_parabola_focus_distance_area_l1588_158800

theorem parabola_focus_distance_area (p : ℝ) (hp : p > 0)
  (A : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1)
  (hDist : A.1 + p / 2 = 2 * A.1)
  (hArea : 1/2 * (p / 2) * |A.2| = 1) :
  p = 2 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_area_l1588_158800


namespace NUMINAMATH_GPT_scientific_notation_of_11090000_l1588_158848

theorem scientific_notation_of_11090000 :
  ∃ (x : ℝ) (n : ℤ), 11090000 = x * 10^n ∧ x = 1.109 ∧ n = 7 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_scientific_notation_of_11090000_l1588_158848


namespace NUMINAMATH_GPT_xiao_ming_reading_plan_l1588_158874

-- Define the number of pages in the book
def total_pages : Nat := 72

-- Define the total number of days to finish the book
def total_days : Nat := 10

-- Define the number of pages read per day for the first two days
def pages_first_two_days : Nat := 5

-- Define the variable x to represent the number of pages read per day for the remaining days
variable (x : Nat)

-- Define the inequality representing the reading plan
def reading_inequality (x : Nat) : Prop :=
  10 + 8 * x ≥ total_pages

-- The statement to be proved
theorem xiao_ming_reading_plan (x : Nat) : reading_inequality x := sorry

end NUMINAMATH_GPT_xiao_ming_reading_plan_l1588_158874


namespace NUMINAMATH_GPT_solution_positive_iff_k_range_l1588_158894

theorem solution_positive_iff_k_range (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (k / (2 * x - 4) - 1 = x / (x - 2))) ↔ (k > -4 ∧ k ≠ 4) := 
sorry

end NUMINAMATH_GPT_solution_positive_iff_k_range_l1588_158894


namespace NUMINAMATH_GPT_dollars_saved_is_correct_l1588_158809

noncomputable def blender_in_store_price : ℝ := 120
noncomputable def juicer_in_store_price : ℝ := 80
noncomputable def blender_tv_price : ℝ := 4 * 28 + 12
noncomputable def total_in_store_price_with_discount : ℝ := (blender_in_store_price + juicer_in_store_price) * 0.90
noncomputable def dollars_saved : ℝ := total_in_store_price_with_discount - blender_tv_price

theorem dollars_saved_is_correct :
  dollars_saved = 56 := by
  sorry

end NUMINAMATH_GPT_dollars_saved_is_correct_l1588_158809


namespace NUMINAMATH_GPT_sum_of_first_ten_terms_l1588_158831

theorem sum_of_first_ten_terms (S : ℕ → ℕ) (h : ∀ n, S n = n^2 - 4 * n + 1) : S 10 = 61 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_ten_terms_l1588_158831


namespace NUMINAMATH_GPT_algebraic_comparison_l1588_158829

theorem algebraic_comparison (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 / b + b^2 / a ≥ a + b) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_comparison_l1588_158829
