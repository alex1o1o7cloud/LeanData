import Mathlib

namespace NUMINAMATH_GPT_negation_of_proposition_l2256_225640

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l2256_225640


namespace NUMINAMATH_GPT_sum_of_non_visible_faces_l2256_225627

theorem sum_of_non_visible_faces
    (d1 d2 d3 d4 : Fin 6 → Nat)
    (visible_faces : List Nat)
    (hv : visible_faces = [1, 2, 3, 4, 4, 5, 5, 6]) :
    let total_sum := 4 * 21
    let visible_sum := List.sum visible_faces
    total_sum - visible_sum = 54 := by
  sorry

end NUMINAMATH_GPT_sum_of_non_visible_faces_l2256_225627


namespace NUMINAMATH_GPT_largest_possible_number_of_pencils_in_a_box_l2256_225672

/-- Olivia bought 48 pencils -/
def olivia_pencils : ℕ := 48
/-- Noah bought 60 pencils -/
def noah_pencils : ℕ := 60
/-- Liam bought 72 pencils -/
def liam_pencils : ℕ := 72

/-- The GCD of the number of pencils bought by Olivia, Noah, and Liam is 12 -/
theorem largest_possible_number_of_pencils_in_a_box :
  gcd olivia_pencils (gcd noah_pencils liam_pencils) = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_possible_number_of_pencils_in_a_box_l2256_225672


namespace NUMINAMATH_GPT_algebraic_identity_example_l2256_225674

-- Define the variables a and b
def a : ℕ := 287
def b : ℕ := 269

-- State the problem and the expected result
theorem algebraic_identity_example :
  a * a + b * b - 2 * a * b = 324 :=
by
  -- Since the proof is not required, we insert sorry here
  sorry

end NUMINAMATH_GPT_algebraic_identity_example_l2256_225674


namespace NUMINAMATH_GPT_cost_of_acai_berry_juice_l2256_225670

theorem cost_of_acai_berry_juice 
  (cost_per_litre_cocktail : ℝ) 
  (cost_per_litre_mixed_fruit : ℝ)
  (volume_mixed_fruit : ℝ)
  (volume_acai_berry : ℝ)
  (total_volume : ℝ) 
  (total_cost_of_mixed_fruit : ℝ)
  (total_cost_cocktail : ℝ)
  : cost_per_litre_cocktail = 1399.45 ∧ 
    cost_per_litre_mixed_fruit = 262.85 ∧ 
    volume_mixed_fruit = 37 ∧ 
    volume_acai_berry = 24.666666666666668 ∧ 
    total_volume = 61.666666666666668 ∧ 
    total_cost_of_mixed_fruit = volume_mixed_fruit * cost_per_litre_mixed_fruit ∧
    total_cost_of_mixed_fruit = 9725.45 ∧
    total_cost_cocktail = total_volume * cost_per_litre_cocktail ∧ 
    total_cost_cocktail = 86327.77 
    → 24.666666666666668 * 3105.99 + 9725.45 = 86327.77 :=
sorry

end NUMINAMATH_GPT_cost_of_acai_berry_juice_l2256_225670


namespace NUMINAMATH_GPT_fraction_equation_solution_l2256_225696

theorem fraction_equation_solution (x y : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 5) (hy1 : y ≠ 0) (hy2 : y ≠ 7)
  (h : (3 / x) + (2 / y) = 1 / 3) : 
  x = (9 * y) / (y - 6) :=
sorry

end NUMINAMATH_GPT_fraction_equation_solution_l2256_225696


namespace NUMINAMATH_GPT_avg_waiting_time_l2256_225629

theorem avg_waiting_time : 
  let P_G := 1 / 3      -- Probability of green light
  let P_red := 2 / 3    -- Probability of red light
  let E_T_given_G := 0  -- Expected time given green light
  let E_T_given_red := 1 -- Expected time given red light
  (E_T_given_G * P_G) + (E_T_given_red * P_red) = 2 / 3
:= by
  sorry

end NUMINAMATH_GPT_avg_waiting_time_l2256_225629


namespace NUMINAMATH_GPT_larrys_correct_substitution_l2256_225667

noncomputable def lucky_larry_expression (a b c d e f : ℤ) : ℤ :=
  a + (b - (c + (d - (e + f))))

noncomputable def larrys_substitution (a b c d e f : ℤ) : ℤ :=
  a + b - c + d - e + f

theorem larrys_correct_substitution : 
  (lucky_larry_expression 2 4 6 8 e 5 = larrys_substitution 2 4 6 8 e 5) ↔ (e = 8) :=
by
  sorry

end NUMINAMATH_GPT_larrys_correct_substitution_l2256_225667


namespace NUMINAMATH_GPT_lemonade_water_cups_l2256_225675

theorem lemonade_water_cups
  (W S L : ℕ)
  (h1 : W = 5 * S)
  (h2 : S = 3 * L)
  (h3 : L = 5) :
  W = 75 :=
by {
  sorry
}

end NUMINAMATH_GPT_lemonade_water_cups_l2256_225675


namespace NUMINAMATH_GPT_team_A_processes_fraction_l2256_225614

theorem team_A_processes_fraction (A B : ℕ) (total_calls : ℚ) 
  (h1 : A = (5/8) * B) 
  (h2 : (8 / 11) * total_calls = TeamB_calls_processed)
  (frac_TeamA_calls : ℚ := (1 - (8 / 11)) * total_calls)
  (calls_per_member_A : ℚ := frac_TeamA_calls / A)
  (calls_per_member_B : ℚ := (8 / 11) * total_calls / B) : 
  calls_per_member_A / calls_per_member_B = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_team_A_processes_fraction_l2256_225614


namespace NUMINAMATH_GPT_min_value_eq_18sqrt3_l2256_225666

noncomputable def min_value (x y : ℝ) (h : x + y = 5) : ℝ := 3^x + 3^y

theorem min_value_eq_18sqrt3 {x y : ℝ} (h : x + y = 5) : min_value x y h ≥ 18 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_min_value_eq_18sqrt3_l2256_225666


namespace NUMINAMATH_GPT_solve_for_a_l2256_225697

theorem solve_for_a (x a : ℝ) (h1 : x + 2 * a - 6 = 0) (h2 : x = -2) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2256_225697


namespace NUMINAMATH_GPT_factorization_theorem_l2256_225659

-- Define the polynomial p(x, y)
def p (x y k : ℝ) : ℝ := x^2 - 2*x*y + k*y^2 + 3*x - 5*y + 2

-- Define the condition for factorization into two linear factors
def can_be_factored (x y m n : ℝ) : Prop :=
  (p x y (m * n)) = ((x + m * y + 1) * (x + n * y + 2))

-- The main theorem proving that k = -3 is the value for factorizability
theorem factorization_theorem (k : ℝ) : (∃ m n : ℝ, can_be_factored x y m n) ↔ k = -3 := by sorry

end NUMINAMATH_GPT_factorization_theorem_l2256_225659


namespace NUMINAMATH_GPT_ratio_of_ages_l2256_225688

open Real

theorem ratio_of_ages (father_age son_age : ℝ) (h1 : father_age = 45) (h2 : son_age = 15) :
  father_age / son_age = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l2256_225688


namespace NUMINAMATH_GPT_count_integer_points_l2256_225649

-- Define the conditions: the parabola P with focus at (0,0) and passing through (6,4) and (-6,-4)
def parabola (P : ℝ × ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y : ℝ, P (x, y) ↔ y = a*x^2 + b) ∧ 
  P (6, 4) ∧ P (-6, -4)

-- Define the main theorem to be proved: the count of integer points satisfying the inequality
theorem count_integer_points (P : ℝ × ℝ → Prop) (hP : parabola P) :
  ∃ n : ℕ, n = 45 ∧ ∀ (x y : ℤ), P (x, y) → |6 * x + 4 * y| ≤ 1200 :=
sorry

end NUMINAMATH_GPT_count_integer_points_l2256_225649


namespace NUMINAMATH_GPT_natural_numbers_fitting_description_l2256_225685

theorem natural_numbers_fitting_description (n : ℕ) (h : 1 / (n : ℚ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) : n = 2 ∨ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_natural_numbers_fitting_description_l2256_225685


namespace NUMINAMATH_GPT_negative_values_count_l2256_225679

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end NUMINAMATH_GPT_negative_values_count_l2256_225679


namespace NUMINAMATH_GPT_option_C_is_quadratic_l2256_225630

-- Definitions based on conditions
def option_A (x : ℝ) : Prop := x^2 + (1/x^2) = 0
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def option_D (x y : ℝ) : Prop := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Statement to prove option C is a quadratic equation in one variable.
theorem option_C_is_quadratic : ∀ x : ℝ, (option_C x) → (∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0) :=
by
  intros x hx
  -- To be proven
  sorry

end NUMINAMATH_GPT_option_C_is_quadratic_l2256_225630


namespace NUMINAMATH_GPT_sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l2256_225664

open Real

-- Problem (a)
theorem sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1 (n k : Nat) :
  (sqrt 2 - 1)^n = sqrt k - sqrt (k - 1) :=
sorry

-- Problem (b)
theorem sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1 (m n k : Nat) :
  (sqrt m - sqrt (m - 1))^n = sqrt k - sqrt (k - 1) :=
sorry

end NUMINAMATH_GPT_sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l2256_225664


namespace NUMINAMATH_GPT_positive_difference_of_solutions_l2256_225636

theorem positive_difference_of_solutions :
  let a := 1
  let b := -6
  let c := -28
  let discriminant := b^2 - 4 * a * c
  let solution1 := 3 + (Real.sqrt discriminant) / 2
  let solution2 := 3 - (Real.sqrt discriminant) / 2
  have h_discriminant : discriminant = 148 := by sorry
  Real.sqrt 148 = 2 * Real.sqrt 37 :=
 sorry

end NUMINAMATH_GPT_positive_difference_of_solutions_l2256_225636


namespace NUMINAMATH_GPT_smallest_integer_switch_add_l2256_225663

theorem smallest_integer_switch_add (a b: ℕ) (h1: n = 10 * a + b) 
  (h2: 3 * n = 10 * b + a + 5)
  (h3: 0 ≤ b) (h4: b < 10) (h5: 1 ≤ a) (h6: a < 10): n = 47 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_switch_add_l2256_225663


namespace NUMINAMATH_GPT_charge_difference_l2256_225615

theorem charge_difference (cost_x cost_y : ℝ) (num_copies : ℕ) (hx : cost_x = 1.25) (hy : cost_y = 2.75) (hn : num_copies = 40) : 
  num_copies * cost_y - num_copies * cost_x = 60 := by
  sorry

end NUMINAMATH_GPT_charge_difference_l2256_225615


namespace NUMINAMATH_GPT_max_value_AMC_l2256_225647

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 15) : 
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 := 
sorry

end NUMINAMATH_GPT_max_value_AMC_l2256_225647


namespace NUMINAMATH_GPT_boris_number_of_bowls_l2256_225661

-- Definitions from the conditions
def total_candies : ℕ := 100
def daughter_eats : ℕ := 8
def candies_per_bowl_after_removal : ℕ := 20
def candies_removed_per_bowl : ℕ := 3

-- Derived definitions
def remaining_candies : ℕ := total_candies - daughter_eats
def candies_per_bowl_orig : ℕ := candies_per_bowl_after_removal + candies_removed_per_bowl

-- Statement to prove
theorem boris_number_of_bowls : remaining_candies / candies_per_bowl_orig = 4 :=
by sorry

end NUMINAMATH_GPT_boris_number_of_bowls_l2256_225661


namespace NUMINAMATH_GPT_price_alloy_per_kg_l2256_225617

-- Defining the costs of the two metals.
def cost_metal1 : ℝ := 68
def cost_metal2 : ℝ := 96

-- Defining the mixture ratio.
def ratio : ℝ := 1

-- The proposition that the price per kg of the alloy is 82 Rs.
theorem price_alloy_per_kg (C1 C2 r : ℝ) (hC1 : C1 = 68) (hC2 : C2 = 96) (hr : r = 1) :
  (C1 + C2) / (r + r) = 82 :=
by
  sorry

end NUMINAMATH_GPT_price_alloy_per_kg_l2256_225617


namespace NUMINAMATH_GPT_laurie_shells_l2256_225606

def alan_collected : ℕ := 48
def ben_collected (alan : ℕ) : ℕ := alan / 4
def laurie_collected (ben : ℕ) : ℕ := ben * 3

theorem laurie_shells (a : ℕ) (b : ℕ) (l : ℕ) (h1 : alan_collected = a)
  (h2 : ben_collected a = b) (h3 : laurie_collected b = l) : l = 36 := 
by
  sorry

end NUMINAMATH_GPT_laurie_shells_l2256_225606


namespace NUMINAMATH_GPT_proof_problem_l2256_225676

variable {a b : ℤ}

theorem proof_problem (h1 : ∃ k : ℤ, a = 4 * k) (h2 : ∃ l : ℤ, b = 8 * l) : 
  (∃ m : ℤ, b = 4 * m) ∧
  (∃ n : ℤ, a - b = 4 * n) ∧
  (∃ p : ℤ, a + b = 2 * p) := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2256_225676


namespace NUMINAMATH_GPT_problem_solution_set_l2256_225622

variable {a b c : ℝ}

theorem problem_solution_set (h_condition : ∀ x, 1 ≤ x → x ≤ 2 → a * x^2 - b * x + c ≥ 0) : 
  { x : ℝ | c * x^2 + b * x + a ≤ 0 } = { x : ℝ | x ≤ -1 } ∪ { x | -1/2 ≤ x } :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_set_l2256_225622


namespace NUMINAMATH_GPT_units_digit_n_l2256_225637

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31^8) (h2 : m % 10 = 7) : n % 10 = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_n_l2256_225637


namespace NUMINAMATH_GPT_arithmetic_progression_rth_term_l2256_225689

theorem arithmetic_progression_rth_term (S : ℕ → ℕ) (hS : ∀ n, S n = 5 * n + 4 * n ^ 2) 
  (r : ℕ) : S r - S (r - 1) = 8 * r + 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_rth_term_l2256_225689


namespace NUMINAMATH_GPT_jinsu_third_attempt_kicks_l2256_225603

theorem jinsu_third_attempt_kicks
  (hoseok_kicks : ℕ) (jinsu_first_attempt : ℕ) (jinsu_second_attempt : ℕ) (required_kicks : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first_attempt = 15 →
  jinsu_second_attempt = 15 →
  required_kicks = 19 →
  jinsu_first_attempt + jinsu_second_attempt + required_kicks > hoseok_kicks :=
by
  sorry

end NUMINAMATH_GPT_jinsu_third_attempt_kicks_l2256_225603


namespace NUMINAMATH_GPT_intersection_is_ge_negative_one_l2256_225644

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem intersection_is_ge_negative_one : M ∩ N = {y | y ≥ -1} := by
  sorry

end NUMINAMATH_GPT_intersection_is_ge_negative_one_l2256_225644


namespace NUMINAMATH_GPT_plane_intersect_probability_l2256_225656

-- Define the vertices of the rectangular prism
def vertices : List (ℝ × ℝ × ℝ) := 
  [(0,0,0), (2,0,0), (2,2,0), (0,2,0), 
   (0,0,1), (2,0,1), (2,2,1), (0,2,1)]

-- Calculate total number of ways to choose 3 vertices out of 8
def total_ways : ℕ := Nat.choose 8 3

-- Calculate the number of planes that do not intersect the interior of the prism
def non_intersecting_planes : ℕ := 6 * Nat.choose 4 3

-- Calculate the probability as a fraction
def probability_of_intersecting (total non_intersecting : ℕ) : ℚ :=
  1 - (non_intersecting : ℚ) / (total : ℚ)

-- The main theorem to state the probability is 4/7
theorem plane_intersect_probability : 
  probability_of_intersecting total_ways non_intersecting_planes = 4 / 7 := 
  by
    -- Skipping the proof
    sorry

end NUMINAMATH_GPT_plane_intersect_probability_l2256_225656


namespace NUMINAMATH_GPT_sum_of_extreme_values_l2256_225653

theorem sum_of_extreme_values (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (5 - Real.sqrt 34) / 3
  let M := (5 + Real.sqrt 34) / 3
  m + M = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_extreme_values_l2256_225653


namespace NUMINAMATH_GPT_domain_of_function_is_all_real_l2256_225681

def domain_function : Prop :=
  ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 6 ≠ 0

theorem domain_of_function_is_all_real :
  domain_function :=
by
  intros t
  sorry

end NUMINAMATH_GPT_domain_of_function_is_all_real_l2256_225681


namespace NUMINAMATH_GPT_solve_system_l2256_225608

noncomputable def system_solutions (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  (1 / x + 1 / y + 1 / z = - (z / (x * y)))

theorem solve_system :
  ∀ (x y z : ℤ), system_solutions x y z ↔ 
    (x = 3 ∧ y = 2 ∧ z = -3) ∨
    (x = -3 ∧ y = 2 ∧ z = 3) ∨
    (x = 2 ∧ y = 3 ∧ z = -3) ∨
    (x = 2 ∧ y = -3 ∧ z = 3) := by
  sorry

end NUMINAMATH_GPT_solve_system_l2256_225608


namespace NUMINAMATH_GPT_total_pins_cardboard_l2256_225678

theorem total_pins_cardboard {length width pins : ℕ} (h_length : length = 34) (h_width : width = 14) (h_pins : pins = 35) :
  2 * pins * (length + width) / (length + width) = 140 :=
by
  sorry

end NUMINAMATH_GPT_total_pins_cardboard_l2256_225678


namespace NUMINAMATH_GPT_simplify_expression_l2256_225691

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2256_225691


namespace NUMINAMATH_GPT_jellybean_count_l2256_225655

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end NUMINAMATH_GPT_jellybean_count_l2256_225655


namespace NUMINAMATH_GPT_ascending_order_proof_l2256_225620

noncomputable def frac1 : ℚ := 1 / 2
noncomputable def frac2 : ℚ := 3 / 4
noncomputable def frac3 : ℚ := 1 / 5
noncomputable def dec1 : ℚ := 0.25
noncomputable def dec2 : ℚ := 0.42

theorem ascending_order_proof :
  frac3 < dec1 ∧ dec1 < dec2 ∧ dec2 < frac1 ∧ frac1 < frac2 :=
by {
  -- The proof will show the conversions mentioned in solution steps
  sorry
}

end NUMINAMATH_GPT_ascending_order_proof_l2256_225620


namespace NUMINAMATH_GPT_segment_halving_1M_l2256_225611

noncomputable def segment_halving_sum (k : ℕ) : ℕ :=
  3^k + 1

theorem segment_halving_1M : segment_halving_sum 1000000 = 3^1000000 + 1 :=
by
  sorry

end NUMINAMATH_GPT_segment_halving_1M_l2256_225611


namespace NUMINAMATH_GPT_workload_increase_l2256_225686

theorem workload_increase (a b c d p : ℕ) (h : p ≠ 0) :
  let total_workload := a + b + c + d
  let workload_per_worker := total_workload / p
  let absent_workers := p / 4
  let remaining_workers := p - absent_workers
  let workload_per_remaining_worker := total_workload / (3 * p / 4)
  workload_per_remaining_worker = (a + b + c + d) * 4 / (3 * p) :=
by
  sorry

end NUMINAMATH_GPT_workload_increase_l2256_225686


namespace NUMINAMATH_GPT_valid_interval_for_k_l2256_225616

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end NUMINAMATH_GPT_valid_interval_for_k_l2256_225616


namespace NUMINAMATH_GPT_sum_of_digits_inequality_l2256_225633

def sum_of_digits (n : ℕ) : ℕ := -- Definition of the sum of digits function
  -- This should be defined, for demonstration we use a placeholder
  sorry

theorem sum_of_digits_inequality (n : ℕ) (h : n > 0) :
  sum_of_digits n ≤ 8 * sum_of_digits (8 * n) :=
sorry

end NUMINAMATH_GPT_sum_of_digits_inequality_l2256_225633


namespace NUMINAMATH_GPT_alice_bob_not_both_l2256_225643

-- Define the group of 8 students
def total_students : ℕ := 8

-- Define the committee size
def committee_size : ℕ := 5

-- Calculate the total number of unrestricted committees
def total_committees : ℕ := Nat.choose total_students committee_size

-- Calculate the number of committees where both Alice and Bob are included
def alice_bob_committees : ℕ := Nat.choose (total_students - 2) (committee_size - 2)

-- Calculate the number of committees where Alice and Bob are not both included
def not_both_alice_bob : ℕ := total_committees - alice_bob_committees

-- Now state the theorem we want to prove
theorem alice_bob_not_both : not_both_alice_bob = 36 :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_not_both_l2256_225643


namespace NUMINAMATH_GPT_cylinder_surface_area_and_volume_l2256_225609

noncomputable def cylinder_total_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem cylinder_surface_area_and_volume (r h : ℝ) (hr : r = 5) (hh : h = 15) :
  cylinder_total_surface_area r h = 200 * Real.pi ∧ cylinder_volume r h = 375 * Real.pi :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_cylinder_surface_area_and_volume_l2256_225609


namespace NUMINAMATH_GPT_ratio_c_d_l2256_225662

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
  (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_c_d_l2256_225662


namespace NUMINAMATH_GPT_max_next_person_weight_l2256_225692

def avg_weight_adult := 150
def avg_weight_child := 70
def max_weight_elevator := 1500
def num_adults := 7
def num_children := 5

def total_weight_adults := num_adults * avg_weight_adult
def total_weight_children := num_children * avg_weight_child
def current_weight := total_weight_adults + total_weight_children

theorem max_next_person_weight : 
  max_weight_elevator - current_weight = 100 := 
by 
  sorry

end NUMINAMATH_GPT_max_next_person_weight_l2256_225692


namespace NUMINAMATH_GPT_math_problem_l2256_225638

theorem math_problem (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x^2) = 23 :=
sorry

end NUMINAMATH_GPT_math_problem_l2256_225638


namespace NUMINAMATH_GPT_railway_original_stations_l2256_225642

theorem railway_original_stations (m n : ℕ) (hn : n > 1) (h : n * (2 * m - 1 + n) = 58) : m = 14 :=
by
  sorry

end NUMINAMATH_GPT_railway_original_stations_l2256_225642


namespace NUMINAMATH_GPT_exists_pos_ints_l2256_225634

open Nat

noncomputable def f (a : ℕ) : ℕ :=
  a^2 + 3 * a + 2

noncomputable def g (b c : ℕ) : ℕ :=
  b^2 - b + 3 * c^2 + 3 * c

theorem exists_pos_ints (a : ℕ) (ha : 0 < a) :
  ∃ (b c : ℕ), 0 < b ∧ 0 < c ∧ f a = g b c :=
sorry

end NUMINAMATH_GPT_exists_pos_ints_l2256_225634


namespace NUMINAMATH_GPT_find_ab_minus_a_neg_b_l2256_225651

variable (a b : ℝ)
variables (h₀ : a > 1) (h₁ : b > 0) (h₂ : a^b + a^(-b) = 2 * Real.sqrt 2)

theorem find_ab_minus_a_neg_b : a^b - a^(-b) = 2 := by
  sorry

end NUMINAMATH_GPT_find_ab_minus_a_neg_b_l2256_225651


namespace NUMINAMATH_GPT_problem_statement_l2256_225650

theorem problem_statement (h : 36 = 6^2) : 6^15 / 36^5 = 7776 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2256_225650


namespace NUMINAMATH_GPT_line_intersects_circle_l2256_225624

-- Definitions
def radius : ℝ := 5
def distance_to_center : ℝ := 3

-- Theorem statement
theorem line_intersects_circle (r : ℝ) (d : ℝ) (h_r : r = radius) (h_d : d = distance_to_center) : d < r :=
by
  rw [h_r, h_d]
  exact sorry

end NUMINAMATH_GPT_line_intersects_circle_l2256_225624


namespace NUMINAMATH_GPT_resistor_problem_l2256_225645

theorem resistor_problem 
  {x y r : ℝ}
  (h1 : 1 / r = 1 / x + 1 / y)
  (h2 : r = 2.9166666666666665)
  (h3 : y = 7) : 
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_resistor_problem_l2256_225645


namespace NUMINAMATH_GPT_winning_candidate_percentage_l2256_225639

theorem winning_candidate_percentage 
    (votes_winner : ℕ)
    (votes_total : ℕ)
    (votes_majority : ℕ)
    (H1 : votes_total = 900)
    (H2 : votes_majority = 360)
    (H3 : votes_winner - (votes_total - votes_winner) = votes_majority) :
    (votes_winner : ℕ) * 100 / (votes_total : ℕ) = 70 := by
    sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l2256_225639


namespace NUMINAMATH_GPT_inequality_subtraction_l2256_225673

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end NUMINAMATH_GPT_inequality_subtraction_l2256_225673


namespace NUMINAMATH_GPT_minimum_at_neg_one_l2256_225623

noncomputable def f (x : Real) : Real := x * Real.exp x

theorem minimum_at_neg_one : 
  ∃ c : Real, c = -1 ∧ ∀ x : Real, f c ≤ f x := sorry

end NUMINAMATH_GPT_minimum_at_neg_one_l2256_225623


namespace NUMINAMATH_GPT_common_ratio_arithmetic_progression_l2256_225677

theorem common_ratio_arithmetic_progression (a3 q : ℝ) (h1 : a3 = 9) (h2 : a3 + a3 * q + 9 = 27) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_arithmetic_progression_l2256_225677


namespace NUMINAMATH_GPT_stationery_difference_l2256_225604

theorem stationery_difference :
  let georgia := 25
  let lorene := 3 * georgia
  lorene - georgia = 50 :=
by
  let georgia := 25
  let lorene := 3 * georgia
  show lorene - georgia = 50
  sorry

end NUMINAMATH_GPT_stationery_difference_l2256_225604


namespace NUMINAMATH_GPT_common_factor_extraction_l2256_225602

-- Define the polynomial
def poly (a b c : ℝ) := 8 * a^3 * b^2 + 12 * a^3 * b * c - 4 * a^2 * b

-- Define the common factor
def common_factor (a b : ℝ) := 4 * a^2 * b

-- State the theorem
theorem common_factor_extraction (a b c : ℝ) :
  ∃ p : ℝ, poly a b c = common_factor a b * p := by
  sorry

end NUMINAMATH_GPT_common_factor_extraction_l2256_225602


namespace NUMINAMATH_GPT_cone_volume_l2256_225680

theorem cone_volume (central_angle : ℝ) (sector_area : ℝ) (h1 : central_angle = 120) (h2 : sector_area = 3 * Real.pi) :
  ∃ V : ℝ, V = (2 * Real.sqrt 2 * Real.pi) / 3 :=
by
  -- We acknowledge the input condition where the angle is 120° and sector area is 3π
  -- The problem requires proving the volume of the cone
  sorry

end NUMINAMATH_GPT_cone_volume_l2256_225680


namespace NUMINAMATH_GPT_smallest_mn_sum_l2256_225657

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end NUMINAMATH_GPT_smallest_mn_sum_l2256_225657


namespace NUMINAMATH_GPT_terminal_side_of_half_angle_quadrant_l2256_225628

def is_angle_in_third_quadrant (α : ℝ) (k : ℤ) : Prop :=
  k * 360 + 180 < α ∧ α < k * 360 + 270

def is_terminal_side_of_half_angle_in_quadrant (α : ℝ) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)

theorem terminal_side_of_half_angle_quadrant (α : ℝ) (k : ℤ) :
  is_angle_in_third_quadrant α k → is_terminal_side_of_half_angle_in_quadrant α := 
sorry

end NUMINAMATH_GPT_terminal_side_of_half_angle_quadrant_l2256_225628


namespace NUMINAMATH_GPT_odd_numbers_not_dividing_each_other_l2256_225683

theorem odd_numbers_not_dividing_each_other (n : ℕ) (hn : n ≥ 4) :
  ∃ (a b : ℕ), a ≠ b ∧ (2 ^ (2 * n) < a ∧ a < 2 ^ (3 * n)) ∧ 
  (2 ^ (2 * n) < b ∧ b < 2 ^ (3 * n)) ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ 
  ¬ (a ∣ b * b) ∧ ¬ (b ∣ a * a) := by
sorry

end NUMINAMATH_GPT_odd_numbers_not_dividing_each_other_l2256_225683


namespace NUMINAMATH_GPT_harper_water_duration_l2256_225619

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end NUMINAMATH_GPT_harper_water_duration_l2256_225619


namespace NUMINAMATH_GPT_find_couples_l2256_225646

theorem find_couples (n p q : ℕ) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q)
    (h_gcd : Nat.gcd p q = 1)
    (h_eq : p + q^2 = (n^2 + 1) * p^2 + q) : 
    (p = n + 1 ∧ q = n^2 + n + 1) :=
by 
  sorry

end NUMINAMATH_GPT_find_couples_l2256_225646


namespace NUMINAMATH_GPT_number_of_bead_necklaces_sold_is_3_l2256_225684

-- Definitions of the given conditions
def total_earnings : ℕ := 36
def gemstone_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 6

-- Define the earnings from gemstone necklaces as a separate definition
def earnings_gemstone_necklaces : ℕ := gemstone_necklaces * cost_per_necklace

-- Define the earnings from bead necklaces based on total earnings and earnings from gemstone necklaces
def earnings_bead_necklaces : ℕ := total_earnings - earnings_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_bead_necklaces / cost_per_necklace

-- The theorem we want to prove
theorem number_of_bead_necklaces_sold_is_3 : bead_necklaces_sold = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bead_necklaces_sold_is_3_l2256_225684


namespace NUMINAMATH_GPT_ratio_malt_to_coke_l2256_225693

-- Definitions from conditions
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_choose_malt : ℕ := 6
def females_choose_malt : ℕ := 8

-- Derived values
def total_cheerleaders : ℕ := total_males + total_females
def total_malt : ℕ := males_choose_malt + females_choose_malt
def total_coke : ℕ := total_cheerleaders - total_malt

-- The theorem to be proved
theorem ratio_malt_to_coke : (total_malt / total_coke) = (7 / 6) :=
  by
    -- skipped proof
    sorry

end NUMINAMATH_GPT_ratio_malt_to_coke_l2256_225693


namespace NUMINAMATH_GPT_taller_cycle_shadow_length_l2256_225612

theorem taller_cycle_shadow_length 
  (h_taller : ℝ) (h_shorter : ℝ) (shadow_shorter : ℝ) (shadow_taller : ℝ) 
  (h_taller_val : h_taller = 2.5) 
  (h_shorter_val : h_shorter = 2) 
  (shadow_shorter_val : shadow_shorter = 4)
  (similar_triangles : h_taller / shadow_taller = h_shorter / shadow_shorter) :
  shadow_taller = 5 := 
by 
  sorry

end NUMINAMATH_GPT_taller_cycle_shadow_length_l2256_225612


namespace NUMINAMATH_GPT_mr_bodhi_adds_twenty_sheep_l2256_225631

def cows : ℕ := 20
def foxes : ℕ := 15
def zebras : ℕ := 3 * foxes
def required_total : ℕ := 100

def sheep := required_total - (cows + foxes + zebras)

theorem mr_bodhi_adds_twenty_sheep : sheep = 20 :=
by
  -- Proof for the theorem is not required and is thus replaced with sorry.
  sorry

end NUMINAMATH_GPT_mr_bodhi_adds_twenty_sheep_l2256_225631


namespace NUMINAMATH_GPT_smith_family_mean_age_l2256_225695

theorem smith_family_mean_age :
  let children_ages := [8, 8, 8, 12, 11]
  let dogs_ages := [3, 4]
  let all_ages := children_ages ++ dogs_ages
  let total_ages := List.sum all_ages
  let total_individuals := List.length all_ages
  (total_ages : ℚ) / (total_individuals : ℚ) = 7.71 :=
by
  sorry

end NUMINAMATH_GPT_smith_family_mean_age_l2256_225695


namespace NUMINAMATH_GPT_solve_for_a_l2256_225635

noncomputable def parabola (a b c : ℚ) (x : ℚ) := a * x^2 + b * x + c

theorem solve_for_a (a b c : ℚ) (h1 : parabola a b c 2 = 5) (h2 : parabola a b c 1 = 2) : 
  a = -3 :=
by
  -- Given: y = ax^2 + bx + c with vertex (2,5) and point (1,2)
  have eq1 : a * (2:ℚ)^2 + b * (2:ℚ) + c = 5 := h1
  have eq2 : a * (1:ℚ)^2 + b * (1:ℚ) + c = 2 := h2

  -- Combine information to find a
  sorry

end NUMINAMATH_GPT_solve_for_a_l2256_225635


namespace NUMINAMATH_GPT_find_exp_l2256_225613

noncomputable def a : ℝ := sorry
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom a_m_eq_six : a ^ m = 6
axiom a_n_eq_six : a ^ n = 6

theorem find_exp : a ^ (2 * m - n) = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_exp_l2256_225613


namespace NUMINAMATH_GPT_counterexample_to_strict_inequality_l2256_225607

theorem counterexample_to_strict_inequality :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
  (0 < a1) ∧ (0 < a2) ∧ (0 < b1) ∧ (0 < b2) ∧ (0 < c1) ∧ (0 < c2) ∧ (0 < d1) ∧ (0 < d2) ∧
  (a1 * b2 < a2 * b1) ∧ (c1 * d2 < c2 * d1) ∧ ¬ (a1 + c1) * (b2 + d2) < (a2 + c2) * (b1 + d1) :=
sorry

end NUMINAMATH_GPT_counterexample_to_strict_inequality_l2256_225607


namespace NUMINAMATH_GPT_quadratic_root_form_l2256_225641

theorem quadratic_root_form {a b : ℂ} (h : 6 * a ^ 2 - 5 * a + 18 = 0 ∧ a.im = 0 ∧ b.im = 0) : 
  a + b^2 = (467:ℚ) / 144 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_form_l2256_225641


namespace NUMINAMATH_GPT_polynomial_divisibility_a_l2256_225665

theorem polynomial_divisibility_a (n : ℕ) : 
  (n % 3 = 1 ∨ n % 3 = 2) ↔ (x^2 + x + 1 ∣ x^(2*n) + x^n + 1) :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_a_l2256_225665


namespace NUMINAMATH_GPT_math_problem_l2256_225668

theorem math_problem (f : ℕ → Prop) (m : ℕ) 
  (h1 : f 1) (h2 : f 2) (h3 : f 3)
  (h_implies : ∀ k : ℕ, f k → f (k + m)) 
  (h_max : m = 3):
  ∀ n : ℕ, 0 < n → f n :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2256_225668


namespace NUMINAMATH_GPT_inequlity_proof_l2256_225600

theorem inequlity_proof (a b : ℝ) : a^2 + a * b + b^2 ≥ 3 * (a + b - 1) := 
  sorry

end NUMINAMATH_GPT_inequlity_proof_l2256_225600


namespace NUMINAMATH_GPT_correct_exponent_calculation_l2256_225654

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end NUMINAMATH_GPT_correct_exponent_calculation_l2256_225654


namespace NUMINAMATH_GPT_right_rectangular_prism_volume_l2256_225626

theorem right_rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = 72) (h2 : y * z = 75) (h3 : x * z = 80) : 
  x * y * z = 657 :=
sorry

end NUMINAMATH_GPT_right_rectangular_prism_volume_l2256_225626


namespace NUMINAMATH_GPT_original_sticker_price_l2256_225660

theorem original_sticker_price (S : ℝ) (h1 : 0.80 * S - 120 = 0.65 * S - 10) : S = 733 := 
by
  sorry

end NUMINAMATH_GPT_original_sticker_price_l2256_225660


namespace NUMINAMATH_GPT_least_consecutive_odd_integers_l2256_225632

theorem least_consecutive_odd_integers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 8 * 414)) :
  x = 407 :=
by
  sorry

end NUMINAMATH_GPT_least_consecutive_odd_integers_l2256_225632


namespace NUMINAMATH_GPT_inequality_proof_l2256_225610

variable (a b c d e p q : ℝ)

theorem inequality_proof
  (h₀ : 0 < p)
  (h₁ : p ≤ a) (h₂ : a ≤ q)
  (h₃ : p ≤ b) (h₄ : b ≤ q)
  (h₅ : p ≤ c) (h₆ : c ≤ q)
  (h₇ : p ≤ d) (h₈ : d ≤ q)
  (h₉ : p ≤ e) (h₁₀ : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 := 
by
  sorry -- The actual proof will be filled here

end NUMINAMATH_GPT_inequality_proof_l2256_225610


namespace NUMINAMATH_GPT_max_boxes_in_large_box_l2256_225625

def max_boxes (l_L w_L h_L : ℕ) (l_S w_S h_S : ℕ) : ℕ :=
  (l_L * w_L * h_L) / (l_S * w_S * h_S)

theorem max_boxes_in_large_box :
  let l_L := 8 * 100 -- converted to cm
  let w_L := 7 * 100 -- converted to cm
  let h_L := 6 * 100 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  max_boxes l_L w_L h_L l_S w_S h_S = 2000000 :=
by {
  let l_L := 800 -- converted to cm
  let w_L := 700 -- converted to cm
  let h_L := 600 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  trivial
}

end NUMINAMATH_GPT_max_boxes_in_large_box_l2256_225625


namespace NUMINAMATH_GPT_division_of_polynomials_l2256_225658

theorem division_of_polynomials (a b : ℝ) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by
  sorry

end NUMINAMATH_GPT_division_of_polynomials_l2256_225658


namespace NUMINAMATH_GPT_integer_part_of_result_is_40_l2256_225694

noncomputable def numerator : ℝ := 0.1 + 1.2 + 2.3 + 3.4 + 4.5 + 5.6 + 6.7 + 7.8 + 8.9
noncomputable def denominator : ℝ := 0.01 + 0.03 + 0.05 + 0.07 + 0.09 + 0.11 + 0.13 + 0.15 + 0.17 + 0.19
noncomputable def result : ℝ := numerator / denominator

theorem integer_part_of_result_is_40 : ⌊result⌋ = 40 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_integer_part_of_result_is_40_l2256_225694


namespace NUMINAMATH_GPT_probability_of_perpendicular_edges_l2256_225669

def is_perpendicular_edge (e1 e2 : ℕ) : Prop :=
-- Define the logic for identifying perpendicular edges here
sorry

def total_outcomes : ℕ := 81

def favorable_outcomes : ℕ :=
-- Calculate the number of favorable outcomes here
20 + 6 + 18

theorem probability_of_perpendicular_edges : 
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 44 / 81 := by
-- Proof for calculating the probability
sorry

end NUMINAMATH_GPT_probability_of_perpendicular_edges_l2256_225669


namespace NUMINAMATH_GPT_problem_solution_l2256_225682

noncomputable def complex_expression : ℝ :=
  (-(1/2) * (1/100))^5 * ((2/3) * (2/100))^4 * (-(3/4) * (3/100))^3 * ((4/5) * (4/100))^2 * (-(5/6) * (5/100)) * 10^30

theorem problem_solution : complex_expression = -48 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2256_225682


namespace NUMINAMATH_GPT_papers_left_after_giving_away_l2256_225687

variable (x : ℕ)

-- Given conditions:
def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41
def total_initial_sheets := sheets_in_desk + sheets_in_backpack

-- Prove that Maria has 91 - x sheets left after giving away x sheets
theorem papers_left_after_giving_away (h : total_initial_sheets = 91) : 
  ∀ d b : ℕ, d = sheets_in_desk → b = sheets_in_backpack → 91 - x = total_initial_sheets - x :=
by
  sorry

end NUMINAMATH_GPT_papers_left_after_giving_away_l2256_225687


namespace NUMINAMATH_GPT_max_value_of_x_and_y_l2256_225605

theorem max_value_of_x_and_y (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : (x - 4) * (x - 10) = 2 ^ y) : x + y ≤ 16 :=
sorry

end NUMINAMATH_GPT_max_value_of_x_and_y_l2256_225605


namespace NUMINAMATH_GPT_josephine_total_milk_l2256_225618

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_josephine_total_milk_l2256_225618


namespace NUMINAMATH_GPT_gcd_sum_lcm_eq_gcd_l2256_225690

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end NUMINAMATH_GPT_gcd_sum_lcm_eq_gcd_l2256_225690


namespace NUMINAMATH_GPT_largest_x_value_satisfies_largest_x_value_l2256_225601

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end NUMINAMATH_GPT_largest_x_value_satisfies_largest_x_value_l2256_225601


namespace NUMINAMATH_GPT_find_m_l2256_225698

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

def symmetric_line (x y m : ℝ) : Prop := x + m * y + 4 = 0

theorem find_m (m : ℝ) (h1 : circle_equation (-1) 3) (h2 : symmetric_line (-1) 3 m) : m = -1 := by
  sorry

end NUMINAMATH_GPT_find_m_l2256_225698


namespace NUMINAMATH_GPT_triangle_perimeter_l2256_225621

theorem triangle_perimeter (A B C : Type) 
  (x : ℝ) 
  (a b c : ℝ) 
  (h₁ : a = x + 1) 
  (h₂ : b = x) 
  (h₃ : c = x - 1) 
  (α β γ : ℝ) 
  (angle_condition : α = 2 * γ) 
  (law_of_sines : a / Real.sin α = c / Real.sin γ)
  (law_of_cosines : Real.cos γ = ((a^2 + b^2 - c^2) / (2 * b * a))) :
  a + b + c = 15 :=
  by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2256_225621


namespace NUMINAMATH_GPT_zoey_finishes_on_monday_l2256_225671

def total_reading_days (books : ℕ) : ℕ :=
  (books * (books + 1)) / 2 + books

def day_of_week (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem zoey_finishes_on_monday : 
  day_of_week 2 (total_reading_days 20) = 1 :=
by
  -- Definitions
  let books := 20
  let start_day := 2 -- Corresponding to Tuesday
  let days := total_reading_days books
  
  -- Prove day_of_week 2 (total_reading_days 20) = 1
  sorry

end NUMINAMATH_GPT_zoey_finishes_on_monday_l2256_225671


namespace NUMINAMATH_GPT_balloons_initial_count_l2256_225699

theorem balloons_initial_count (B : ℕ) (G : ℕ) : ∃ G : ℕ, B = 7 * G + 4 := sorry

end NUMINAMATH_GPT_balloons_initial_count_l2256_225699


namespace NUMINAMATH_GPT_admission_price_for_adults_l2256_225652

theorem admission_price_for_adults (A : ℕ) (ticket_price_children : ℕ) (total_children_tickets : ℕ) 
    (total_amount : ℕ) (total_tickets : ℕ) (children_ticket_costs : ℕ) 
    (adult_tickets : ℕ) (adult_ticket_costs : ℕ) :
    ticket_price_children = 5 → 
    total_children_tickets = 21 → 
    total_amount = 201 → 
    total_tickets = 33 → 
    children_ticket_costs = 21 * 5 → 
    adult_tickets = 33 - 21 → 
    adult_ticket_costs = 201 - 21 * 5 → 
    A = (201 - 21 * 5) / (33 - 21) → 
    A = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_admission_price_for_adults_l2256_225652


namespace NUMINAMATH_GPT_coins_remainder_l2256_225648

theorem coins_remainder 
  (n : ℕ)
  (h₁ : n % 8 = 6)
  (h₂ : n % 7 = 2)
  (h₃ : n = 30) :
  n % 9 = 3 :=
sorry

end NUMINAMATH_GPT_coins_remainder_l2256_225648
