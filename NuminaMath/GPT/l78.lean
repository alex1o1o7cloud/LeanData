import Mathlib

namespace Rams_monthly_salary_l78_78597

variable (R S A : ℝ)
variable (annual_salary : ℝ)
variable (monthly_salary_conversion : annual_salary / 12 = A)
variable (ram_shyam_condition : 0.10 * R = 0.08 * S)
variable (shyam_abhinav_condition : S = 2 * A)
variable (abhinav_annual_salary : annual_salary = 192000)

theorem Rams_monthly_salary 
  (annual_salary : ℝ)
  (ram_shyam_condition : 0.10 * R = 0.08 * S)
  (shyam_abhinav_condition : S = 2 * A)
  (abhinav_annual_salary : annual_salary = 192000)
  (monthly_salary_conversion: annual_salary / 12 = A): 
  R = 25600 := by
  sorry

end Rams_monthly_salary_l78_78597


namespace convex_polyhedron_P_T_V_sum_eq_34_l78_78171

theorem convex_polyhedron_P_T_V_sum_eq_34
  (F : ℕ) (V : ℕ) (E : ℕ) (T : ℕ) (P : ℕ) 
  (hF : F = 32)
  (hT1 : 3 * T + 5 * P = 960)
  (hT2 : 2 * E = V * (T + P))
  (hT3 : T + P - 2 = 60)
  (hT4 : F + V - E = 2) :
  P + T + V = 34 := by
  sorry

end convex_polyhedron_P_T_V_sum_eq_34_l78_78171


namespace angle_greater_than_150_l78_78139

theorem angle_greater_than_150 (a b c R : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c < 2 * R) : 
  ∃ (A : ℝ), A > 150 ∧ ( ∃ (B C : ℝ), A + B + C = 180 ) :=
sorry

end angle_greater_than_150_l78_78139


namespace no_solution_fractions_eq_l78_78558

open Real

theorem no_solution_fractions_eq (x : ℝ) :
  (x-2)/(2*x-1) + 1 = 3/(2-4*x) → False :=
by
  intro h
  have h1 : ¬ (2*x - 1 = 0) := by
    -- 2*x - 1 ≠ 0
    sorry
  have h2 : ¬ (2 - 4*x = 0) := by
    -- 2 - 4*x ≠ 0
    sorry
  -- Solve the equation and show no solutions exist without contradicting the conditions
  sorry

end no_solution_fractions_eq_l78_78558


namespace lcm_20_45_75_eq_900_l78_78576

theorem lcm_20_45_75_eq_900 : Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
by sorry

end lcm_20_45_75_eq_900_l78_78576


namespace simplify_expression_l78_78699

theorem simplify_expression (α : ℝ) :
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) - (2 * Real.cos (2 * α)^2 - 1)) /
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) + (2 * Real.cos (2 * α)^2 - 1)) = Real.tan (2 * α) :=
by
  sorry

end simplify_expression_l78_78699


namespace sam_wins_probability_l78_78840

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l78_78840


namespace range_of_real_roots_l78_78797

theorem range_of_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) ↔
  a >= -1 ∨ a <= -3/2 :=
  sorry

end range_of_real_roots_l78_78797


namespace spinner_prob_C_l78_78598

theorem spinner_prob_C (P_A P_B P_C : ℚ) (h_A : P_A = 1/3) (h_B : P_B = 5/12) (h_total : P_A + P_B + P_C = 1) : 
  P_C = 1/4 := 
sorry

end spinner_prob_C_l78_78598


namespace num_positive_integers_satisfying_condition_l78_78133

theorem num_positive_integers_satisfying_condition :
  ∃! (n : ℕ), 30 - 6 * n > 18 := by
  sorry

end num_positive_integers_satisfying_condition_l78_78133


namespace fans_per_set_l78_78260

theorem fans_per_set (total_fans : ℕ) (sets_of_bleachers : ℕ) (fans_per_set : ℕ)
  (h1 : total_fans = 2436) (h2 : sets_of_bleachers = 3) : fans_per_set = 812 :=
by
  sorry

end fans_per_set_l78_78260


namespace car_mileage_l78_78751

/-- If a car needs 3.5 gallons of gasoline to travel 140 kilometers, it gets 40 kilometers per gallon. -/
theorem car_mileage (gallons_used : ℝ) (distance_traveled : ℝ) 
  (h : gallons_used = 3.5 ∧ distance_traveled = 140) : 
  distance_traveled / gallons_used = 40 :=
by
  sorry

end car_mileage_l78_78751


namespace number_of_pens_sold_l78_78179

variables (C N : ℝ) (gain_percentage : ℝ) (gain : ℝ)

-- Defining conditions given in the problem
def trader_gain_cost_pens (C N : ℝ) : ℝ := 30 * C
def gain_percentage_condition (gain_percentage : ℝ) : Prop := gain_percentage = 0.30
def gain_condition (C N : ℝ) : Prop := (0.30 * N * C) = 30 * C

-- Defining the theorem to prove
theorem number_of_pens_sold
  (h_gain_percentage : gain_percentage_condition gain_percentage)
  (h_gain : gain_condition C N) :
  N = 100 :=
sorry

end number_of_pens_sold_l78_78179


namespace negation_of_universal_proposition_l78_78652
open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end negation_of_universal_proposition_l78_78652


namespace rectangle_count_5x5_l78_78967

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l78_78967


namespace ball_bounce_height_l78_78460

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end ball_bounce_height_l78_78460


namespace trays_needed_to_refill_l78_78322

theorem trays_needed_to_refill (initial_ice_cubes used_ice_cubes tray_capacity : ℕ)
  (h_initial: initial_ice_cubes = 130)
  (h_used: used_ice_cubes = (initial_ice_cubes * 8 / 10))
  (h_tray_capacity: tray_capacity = 14) :
  (initial_ice_cubes + tray_capacity - 1) / tray_capacity = 10 :=
by
  sorry

end trays_needed_to_refill_l78_78322


namespace number_of_rectangles_in_grid_l78_78996

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l78_78996


namespace perp_lines_value_of_m_parallel_lines_value_of_m_l78_78219

theorem perp_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) * ((m - 2) / 3) = -1)) → 
  m = 1 / 2 := 
sorry

theorem parallel_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) = ((m - 2) / 3))) → 
  m = -1 := 
sorry

end perp_lines_value_of_m_parallel_lines_value_of_m_l78_78219


namespace cylinder_volume_l78_78793

theorem cylinder_volume (r l : ℝ) (h1 : r = 1) (h2 : l = 2 * r) : 
  ∃ V : ℝ, V = 2 * Real.pi := 
by 
  sorry

end cylinder_volume_l78_78793


namespace problem1_problem2_l78_78125

-- Problem 1
theorem problem1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : (a^2 + a) * (a + 1) / a = 3 := 
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a = 1 / 2) : (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := 
sorry

end problem1_problem2_l78_78125


namespace determine_p_and_q_l78_78806

theorem determine_p_and_q (x p q : ℝ) : 
  (x + 4) * (x - 1) = x^2 + p * x + q → (p = 3 ∧ q = -4) := 
by 
  sorry

end determine_p_and_q_l78_78806


namespace rate_of_interest_is_20_l78_78457

-- Definitions of the given conditions
def principal := 400
def simple_interest := 160
def time := 2

-- Definition of the rate of interest based on the given formula
def rate_of_interest (P SI T : ℕ) : ℕ := (SI * 100) / (P * T)

-- Theorem stating that the rate of interest is 20% given the conditions
theorem rate_of_interest_is_20 :
  rate_of_interest principal simple_interest time = 20 := by
  sorry

end rate_of_interest_is_20_l78_78457


namespace find_divisor_l78_78590

-- Definitions of the conditions
def dividend : ℕ := 15968
def quotient : ℕ := 89
def remainder : ℕ := 37

-- The theorem stating the proof problem
theorem find_divisor (D : ℕ) (h : dividend = D * quotient + remainder) : D = 179 :=
sorry

end find_divisor_l78_78590


namespace total_weight_four_pets_l78_78195

-- Define the weights
def Evan_dog := 63
def Ivan_dog := Evan_dog / 7
def combined_weight_dogs := Evan_dog + Ivan_dog
def Kara_cat := combined_weight_dogs * 5
def combined_weight_dogs_and_cat := Evan_dog + Ivan_dog + Kara_cat
def Lisa_parrot := combined_weight_dogs_and_cat * 3
def total_weight := Evan_dog + Ivan_dog + Kara_cat + Lisa_parrot

-- Total weight of the four pets
theorem total_weight_four_pets : total_weight = 1728 := by
  sorry

end total_weight_four_pets_l78_78195


namespace image_preimage_f_l78_78067

-- Defining the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Given conditions
def A : Set (ℝ × ℝ) := {p | True}
def B : Set (ℝ × ℝ) := {p | True}

-- Proof statement
theorem image_preimage_f :
  f (1, 3) = (4, -2) ∧ ∃ x y : ℝ, f (x, y) = (1, 3) ∧ (x, y) = (2, -1) :=
by
  sorry

end image_preimage_f_l78_78067


namespace repeating_block_length_7_div_13_l78_78859

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l78_78859


namespace f_even_l78_78646

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even (a : ℝ) (h1 : is_even f) (h2 : ∀ x, -1 ≤ x ∧ x ≤ a) : f a = 2 :=
  sorry

end f_even_l78_78646


namespace total_pages_l78_78123

-- Definitions based on conditions
def math_pages : ℕ := 10
def extra_reading_pages : ℕ := 3
def reading_pages : ℕ := math_pages + extra_reading_pages

-- Statement of the proof problem
theorem total_pages : math_pages + reading_pages = 23 := by 
  sorry

end total_pages_l78_78123


namespace inequality_proof_l78_78596

theorem inequality_proof (x y z : ℝ) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ Real.sqrt 3 * (x * y + y * z + z * x) := 
  sorry

end inequality_proof_l78_78596


namespace extremum_values_l78_78273

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem extremum_values :
  (∀ x, f x ≤ 5) ∧ f (-1) = 5 ∧ (∀ x, f x ≥ -27) ∧ f 3 = -27 :=
by
  sorry

end extremum_values_l78_78273


namespace overhead_percentage_l78_78140

def purchase_price : ℝ := 48
def markup : ℝ := 30
def net_profit : ℝ := 12

-- Define the theorem to be proved
theorem overhead_percentage : ((markup - net_profit) / purchase_price) * 100 = 37.5 := by
  sorry

end overhead_percentage_l78_78140


namespace ab_operation_l78_78715

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l78_78715


namespace cars_equilibrium_l78_78874

variable (days : ℕ) -- number of days after which we need the condition to hold
variable (carsA_init carsB_init carsA_to_B carsB_to_A : ℕ) -- initial conditions and parameters

theorem cars_equilibrium :
  let cars_total := 192 + 48
  let carsA := carsA_init + (carsB_to_A - carsA_to_B) * days
  let carsB := carsB_init + (carsA_to_B - carsB_to_A) * days
  carsA_init = 192 -> carsB_init = 48 ->
  carsA_to_B = 21 -> carsB_to_A = 24 ->
  cars_total = 192 + 48 ->
  days = 6 ->
  cars_total = carsA + carsB -> carsA = 7 * carsB :=
by
  intros
  sorry

end cars_equilibrium_l78_78874


namespace part_I_part_II_l78_78955

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part (I)
theorem part_I (a : ℝ) (h_a : a > 0) (h_roots: ∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  0 < a ∧ a < 2 / 5 :=
sorry

-- Part (II)
theorem part_II (a : ℝ) (h_max : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ f a 2) : 
  a ≥ -1 / 3 :=
sorry

end part_I_part_II_l78_78955


namespace Peter_vacation_l78_78408

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end Peter_vacation_l78_78408


namespace distinct_arrangements_ballon_l78_78084

theorem distinct_arrangements_ballon : 
  let n := 6
  let repetitions := 2
  n! / repetitions! = 360 :=
by
  sorry

end distinct_arrangements_ballon_l78_78084


namespace sum_of_twos_and_threes_3024_l78_78085

theorem sum_of_twos_and_threes_3024 : ∃ n : ℕ, n = 337 ∧ (∃ (a b : ℕ), 3024 = 2 * a + 3 * b) :=
sorry

end sum_of_twos_and_threes_3024_l78_78085


namespace relationship_between_abc_l78_78546

noncomputable def a : ℝ := (0.6 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (1.5 : ℝ)
noncomputable def c : ℝ := (1.5 : ℝ) ^ (0.6 : ℝ)

theorem relationship_between_abc : c > a ∧ a > b := sorry

end relationship_between_abc_l78_78546


namespace parallelogram_larger_angle_l78_78376

theorem parallelogram_larger_angle (a b : ℕ) (h₁ : b = a + 50) (h₂ : a = 65) : b = 115 := 
by
  -- Use the conditions h₁ and h₂ to prove the statement.
  sorry

end parallelogram_larger_angle_l78_78376


namespace speed_in_still_water_l78_78604

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_up : upstream_speed = 26) (h_down : downstream_speed = 30) :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end speed_in_still_water_l78_78604


namespace value_of_a_star_b_l78_78726

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l78_78726


namespace bob_final_total_score_l78_78317

theorem bob_final_total_score 
  (points_per_correct : ℕ := 5)
  (points_per_incorrect : ℕ := 2)
  (correct_answers : ℕ := 18)
  (incorrect_answers : ℕ := 2) :
  (points_per_correct * correct_answers - points_per_incorrect * incorrect_answers) = 86 :=
by 
  sorry

end bob_final_total_score_l78_78317


namespace raisins_in_boxes_l78_78764

theorem raisins_in_boxes :
  ∃ x : ℕ, 72 + 74 + 3 * x = 437 ∧ x = 97 :=
by
  existsi 97
  split
  · rw [←add_assoc, add_comm 146, add_assoc]; exact rfl
  · exact rfl

end raisins_in_boxes_l78_78764


namespace find_a_l78_78165

theorem find_a (n k : ℕ) (h1 : 1 < k) (h2 : k < n)
  (h3 : ((n * (n + 1)) / 2 - k) / (n - 1) = 10) : n + k = 29 :=
by
  -- Proof omitted
  sorry

end find_a_l78_78165


namespace find_f2_l78_78712

theorem find_f2 :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 2 * f x - 3 * f (1 / x) = x ^ 2) ∧ f 2 = 93 / 32) :=
sorry

end find_f2_l78_78712


namespace negative_implies_neg_reciprocal_positive_l78_78660

theorem negative_implies_neg_reciprocal_positive {x : ℝ} (h : x < 0) : -x⁻¹ > 0 :=
sorry

end negative_implies_neg_reciprocal_positive_l78_78660


namespace probability_three_common_books_l78_78404

-- Defining the total number of books
def total_books : ℕ := 12

-- Defining the number of books each of Harold and Betty chooses
def books_per_person : ℕ := 6

-- Assertion that the probability of choosing exactly 3 common books is 50/116
theorem probability_three_common_books :
  ((Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)) /
  ((Nat.choose 12 6) * (Nat.choose 12 6)) = 50 / 116 := by
  sorry

end probability_three_common_books_l78_78404


namespace find_a_if_parallel_l78_78665

-- Define the parallel condition for the given lines
def is_parallel (a : ℝ) : Prop :=
  let slope1 := -a / 2
  let slope2 := 3
  slope1 = slope2

-- Prove that a = -6 under the parallel condition
theorem find_a_if_parallel (a : ℝ) (h : is_parallel a) : a = -6 := by
  sorry

end find_a_if_parallel_l78_78665


namespace eval_ff_ff_3_l78_78685

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem eval_ff_ff_3 : f (f (f (f 3))) = 8 :=
  sorry

end eval_ff_ff_3_l78_78685


namespace Walter_receives_49_bananas_l78_78678

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l78_78678


namespace smallest_three_digit_number_exists_l78_78613

def is_valid_permutation_sum (x y z : ℕ) : Prop :=
  let perms := [100*x + 10*y + z, 100*x + 10*z + y, 100*y + 10*x + z, 100*z + 10*x + y, 100*y + 10*z + x, 100*z + 10*y + x]
  perms.sum = 2220

theorem smallest_three_digit_number_exists : ∃ (x y z : ℕ), x < y ∧ y < z ∧ x + y + z = 10 ∧ is_valid_permutation_sum x y z ∧ 100 * x + 10 * y + z = 127 :=
by {
  -- proof goal and steps would go here if we were to complete the proof
  sorry
}

end smallest_three_digit_number_exists_l78_78613


namespace buying_beams_l78_78708

theorem buying_beams (x : ℕ) (h : 3 * (x - 1) * x = 6210) :
  3 * (x - 1) * x = 6210 :=
by {
  sorry
}

end buying_beams_l78_78708


namespace range_of_m_l78_78952

-- Definitions based on given conditions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x + m ≠ 0
def q (m : ℝ) : Prop := m > 1 ∧ m - 1 > 1

-- The mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (hnp : ¬p m) (hapq : ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
  by sorry

end range_of_m_l78_78952


namespace fg_of_1_l78_78543

def f (x : ℤ) : ℤ := x + 3
def g (x : ℤ) : ℤ := x^3 - x^2 - 6

theorem fg_of_1 : f (g 1) = -3 := by
  sorry

end fg_of_1_l78_78543


namespace sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l78_78367

theorem sum_of_consecutive_natural_numbers_eq_three_digit_same_digits :
  ∃ n : ℕ, (1 + n) * n / 2 = 111 * 6 ∧ n = 36 :=
by
  sorry

end sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l78_78367


namespace circumscribed_circle_area_l78_78465

noncomputable def sin_36 := real.sin (36 * real.pi / 180)
noncomputable def radius (s : ℝ) : ℝ := s / (2 * sin_36)
noncomputable def area_of_circumscribed_circle (s : ℝ) : ℝ := real.pi * (radius s) ^ 2

theorem circumscribed_circle_area {s : ℝ} (h : s = 10) :
  area_of_circumscribed_circle s = 2000 * (5 + 2 * real.sqrt 5) * real.pi :=
by
  -- using the given condition
  rw [h]
  -- calculation steps omitted
  sorry

end circumscribed_circle_area_l78_78465


namespace participation_arrangements_l78_78352

def num_students : ℕ := 5
def num_competitions : ℕ := 3
def eligible_dance_students : ℕ := 4

def arrangements_singing : ℕ := num_students
def arrangements_chess : ℕ := num_students
def arrangements_dance : ℕ := eligible_dance_students

def total_arrangements : ℕ := arrangements_singing * arrangements_chess * arrangements_dance

theorem participation_arrangements :
  total_arrangements = 100 := by
  sorry

end participation_arrangements_l78_78352


namespace distance_to_fourth_side_l78_78785

-- Let s be the side length of the square.
variable (s : ℝ) (d1 d2 d3 d4 : ℝ)

-- The given conditions:
axiom h1 : d1 = 4
axiom h2 : d2 = 7
axiom h3 : d3 = 13
axiom h4 : d1 + d2 + d3 + d4 = s
axiom h5 : 0 < d4

-- The statement to prove:
theorem distance_to_fourth_side : d4 = 10 ∨ d4 = 16 :=
by
  sorry

end distance_to_fourth_side_l78_78785


namespace rectangle_area_l78_78431

-- Define length and width
def width : ℕ := 6
def length : ℕ := 3 * width

-- Define area of the rectangle
def area (length width : ℕ) : ℕ := length * width

-- Statement to prove
theorem rectangle_area : area length width = 108 := by
  sorry

end rectangle_area_l78_78431


namespace waiter_earned_in_tips_l78_78623

def waiter_customers := 7
def customers_didnt_tip := 5
def tip_per_customer := 3
def customers_tipped := waiter_customers - customers_didnt_tip
def total_earnings := customers_tipped * tip_per_customer

theorem waiter_earned_in_tips : total_earnings = 6 :=
by
  sorry

end waiter_earned_in_tips_l78_78623


namespace largest_common_divisor_408_340_is_68_l78_78898

theorem largest_common_divisor_408_340_is_68 :
  let factors_408 := [1, 2, 3, 4, 6, 8, 12, 17, 24, 34, 51, 68, 102, 136, 204, 408]
  let factors_340 := [1, 2, 4, 5, 10, 17, 20, 34, 68, 85, 170, 340]
  ∀ d ∈ factors_408, d ∈ factors_340 → ∀ (e ∈ factors_408), (e ∈ factors_340) → d ≤ e :=
  68 := by sorry

end largest_common_divisor_408_340_is_68_l78_78898


namespace domain_translation_l78_78211

theorem domain_translation (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < 3 * x + 2 ∧ 3 * x + 2 < 1 → (∃ y : ℝ, f (3 * x + 2) = y)) →
  (∀ x : ℝ, ∃ y : ℝ, f (2 * x - 1) = y ↔ (3 / 2) < x ∧ x < 3) :=
sorry

end domain_translation_l78_78211


namespace aprils_plant_arrangement_l78_78619

theorem aprils_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 3
  let total_units := basil_plants + 1
  
  (fact total_units * fact tomato_plants = 4320) :=
by
  unfold basil_plants
  unfold tomato_plants
  unfold total_units
  apply eq.refl
  sorry

end aprils_plant_arrangement_l78_78619


namespace Iris_shorts_l78_78238

theorem Iris_shorts :
  ∃ s, (3 * 10) + s * 6 + (4 * 12) = 90 ∧ s = 2 := 
by
  existsi 2
  sorry

end Iris_shorts_l78_78238


namespace tan_sin_cos_eq_l78_78026

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l78_78026


namespace distinct_values_of_products_l78_78300

theorem distinct_values_of_products (n : ℤ) (h : 1 ≤ n) :
  ¬ ∃ a b c d : ℤ, n^2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 ∧ ad = bc :=
sorry

end distinct_values_of_products_l78_78300


namespace bill_difference_zero_l78_78114

theorem bill_difference_zero (l m : ℝ) 
  (hL : (25 / 100) * l = 5) 
  (hM : (15 / 100) * m = 3) : 
  l - m = 0 := 
sorry

end bill_difference_zero_l78_78114


namespace number_of_rectangles_in_grid_l78_78998

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l78_78998


namespace Sam_wins_probability_l78_78834

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l78_78834


namespace tan_sin_cos_eq_l78_78054

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l78_78054


namespace workers_together_time_l78_78160

-- Definition of the times taken by each worker to complete the job
def timeA : ℚ := 8
def timeB : ℚ := 10
def timeC : ℚ := 12

-- Definition of the rates based on the times
def rateA : ℚ := 1 / timeA
def rateB : ℚ := 1 / timeB
def rateC : ℚ := 1 / timeC

-- Definition of the total rate when working together
def total_rate : ℚ := rateA + rateB + rateC

-- Definition of the total time taken to complete the job when working together
def total_time : ℚ := 1 / total_rate

-- The final theorem we need to prove
theorem workers_together_time : total_time = 120 / 37 :=
by {
  -- structure of the proof will go here, but it is not required as per the instructions
  sorry
}

end workers_together_time_l78_78160


namespace solve_for_x_l78_78668

-- Problem definition
def problem_statement (x : ℕ) : Prop :=
  (3 * x / 7 = 15) → x = 35

-- Theorem statement in Lean 4
theorem solve_for_x (x : ℕ) : problem_statement x :=
by
  intros h
  sorry

end solve_for_x_l78_78668


namespace area_percentage_decrease_42_l78_78373

def radius_decrease_factor : ℝ := 0.7615773105863908

noncomputable def area_percentage_decrease : ℝ :=
  let k := radius_decrease_factor
  100 * (1 - k^2)

theorem area_percentage_decrease_42 :
  area_percentage_decrease = 42 := by
  sorry

end area_percentage_decrease_42_l78_78373


namespace percentage_markup_l78_78566

open Real

theorem percentage_markup (SP CP : ℝ) (hSP : SP = 5600) (hCP : CP = 4480) : 
  ((SP - CP) / CP) * 100 = 25 :=
by
  sorry

end percentage_markup_l78_78566


namespace rectangle_count_5x5_l78_78969

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l78_78969


namespace base_conversion_problem_l78_78137

theorem base_conversion_problem 
  (b x y z : ℕ)
  (h1 : 1987 = x * b^2 + y * b + z)
  (h2 : x + y + z = 25) :
  b = 19 ∧ x = 5 ∧ y = 9 ∧ z = 11 := 
by
  sorry

end base_conversion_problem_l78_78137


namespace books_remaining_correct_l78_78875

-- Define the total number of books and the number of books read
def total_books : ℕ := 32
def books_read : ℕ := 17

-- Define the number of books remaining to be read
def books_remaining : ℕ := total_books - books_read

-- Prove that the number of books remaining to be read is 15
theorem books_remaining_correct : books_remaining = 15 := by
  sorry

end books_remaining_correct_l78_78875


namespace lex_reads_in_12_days_l78_78401

theorem lex_reads_in_12_days
  (total_pages : ℕ)
  (pages_per_day : ℕ)
  (h1 : total_pages = 240)
  (h2 : pages_per_day = 20) :
  total_pages / pages_per_day = 12 :=
by
  sorry

end lex_reads_in_12_days_l78_78401


namespace missing_digit_l78_78572

theorem missing_digit (x : ℕ) (h1 : x ≥ 0) (h2 : x ≤ 9) : 
  (if x ≥ 2 then 9 * 1000 + x * 100 + 2 * 10 + 1 else 9 * 100 + 2 * 10 + x * 1) - (1 * 1000 + 2 * 100 + 9 * 10 + x) = 8262 → x = 5 :=
by 
  sorry

end missing_digit_l78_78572


namespace repeating_block_length_7_div_13_l78_78857

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l78_78857


namespace find_k_values_l78_78957

variable {k : ℝ}

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + 1

theorem find_k_values 
  (h_max : ∀ x ∈ set.Icc (-2 : ℝ) 2, f k x ≤ 4) 
  (h_exists : ∃ x ∈ set.Icc (-2 : ℝ) 2, f k x = 4) : 
  k = 1 / 2 ∨ k = -12 := 
sorry

end find_k_values_l78_78957


namespace maximum_area_of_triangle_ABQ_l78_78213

open Real

structure Point3D where
  x : ℝ
  y : ℝ

def circle_C (Q : Point3D) : Prop := (Q.x - 3)^2 + (Q.y - 4)^2 = 4

def A := Point3D.mk 1 0
def B := Point3D.mk (-1) 0

noncomputable def area_triangle (P Q R : Point3D) : ℝ :=
  (1 / 2) * abs ((P.x * (Q.y - R.y)) + (Q.x * (R.y - P.y)) + (R.x * (P.y - Q.y)))

theorem maximum_area_of_triangle_ABQ : ∀ (Q : Point3D), circle_C Q → area_triangle A B Q ≤ 6 := by
  sorry

end maximum_area_of_triangle_ABQ_l78_78213


namespace remaining_budget_l78_78803

def charge_cost : ℝ := 3.5
def num_charges : ℝ := 4
def total_budget : ℝ := 20

theorem remaining_budget : total_budget - (num_charges * charge_cost) = 6 := 
by 
  sorry

end remaining_budget_l78_78803


namespace num_rectangles_in_5x5_grid_l78_78973

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l78_78973


namespace tan_sin_cos_eq_l78_78027

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l78_78027


namespace player_1_points_l78_78879

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l78_78879


namespace max_diameters_l78_78501

theorem max_diameters (n : ℕ) (points : Finset (ℝ × ℝ)) (h : n ≥ 3) (hn : points.card = n)
  (d : ℝ) (h_d_max : ∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q ≤ d) :
  ∃ m : ℕ, m ≤ n ∧ (∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q = d → m ≤ n) := 
sorry

end max_diameters_l78_78501


namespace cos_theta_neg_three_fifths_l78_78947

theorem cos_theta_neg_three_fifths 
  (θ : ℝ)
  (h1 : Real.sin θ = -4 / 5)
  (h2 : Real.tan θ > 0) : 
  Real.cos θ = -3 / 5 := 
sorry

end cos_theta_neg_three_fifths_l78_78947


namespace hexagon_angle_E_l78_78385

theorem hexagon_angle_E (A N G L E S : ℝ) 
  (h1 : A = G) 
  (h2 : G = E) 
  (h3 : N + S = 180) 
  (h4 : L = 90) 
  (h_sum : A + N + G + L + E + S = 720) : 
  E = 150 := 
by 
  sorry

end hexagon_angle_E_l78_78385


namespace neg_proposition_equiv_l78_78689

theorem neg_proposition_equiv :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) :=
by
  sorry

end neg_proposition_equiv_l78_78689


namespace repeating_block_length_7_div_13_l78_78856

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l78_78856


namespace sum_of_2x2_table_is_zero_l78_78669

theorem sum_of_2x2_table_is_zero {a b c d : ℤ} 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_eq : a + b = c + d)
  (prod_eq : a * c = b * d) :
  a + b + c + d = 0 :=
by sorry

end sum_of_2x2_table_is_zero_l78_78669


namespace sequence_factorial_l78_78653

theorem sequence_factorial (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n > 0 → a n = n * a (n - 1)) :
  ∀ n : ℕ, a n = Nat.factorial n :=
by
  sorry

end sequence_factorial_l78_78653


namespace distribution_value_l78_78439

def standard_deviation := 2
def mean := 51

theorem distribution_value (x : ℝ) (hx : x < 45) : (mean - 3 * standard_deviation) > x :=
by
  -- Provide the statement without proof
  sorry

end distribution_value_l78_78439


namespace cost_of_each_box_of_cereal_l78_78703

theorem cost_of_each_box_of_cereal
  (total_groceries_cost : ℝ)
  (gallon_of_milk_cost : ℝ)
  (number_of_cereal_boxes : ℕ)
  (banana_cost_each : ℝ)
  (number_of_bananas : ℕ)
  (apple_cost_each : ℝ)
  (number_of_apples : ℕ)
  (cookie_cost_multiplier : ℝ)
  (number_of_cookie_boxes : ℕ) :
  total_groceries_cost = 25 →
  gallon_of_milk_cost = 3 →
  number_of_cereal_boxes = 2 →
  banana_cost_each = 0.25 →
  number_of_bananas = 4 →
  apple_cost_each = 0.5 →
  number_of_apples = 4 →
  cookie_cost_multiplier = 2 →
  number_of_cookie_boxes = 2 →
  (total_groceries_cost - (gallon_of_milk_cost + (banana_cost_each * number_of_bananas) + 
                           (apple_cost_each * number_of_apples) + 
                           (number_of_cookie_boxes * (cookie_cost_multiplier * gallon_of_milk_cost)))) / 
  number_of_cereal_boxes = 3.5 := 
sorry

end cost_of_each_box_of_cereal_l78_78703


namespace smallest_five_digit_congruent_to_three_mod_seventeen_l78_78155

theorem smallest_five_digit_congruent_to_three_mod_seventeen :
  ∃ (n : ℤ), 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 ∧ n = 10012 :=
by
  sorry

end smallest_five_digit_congruent_to_three_mod_seventeen_l78_78155


namespace coupon_percentage_l78_78266

theorem coupon_percentage (P i d final_price total_price discount_amount percentage: ℝ)
  (h1 : P = 54) (h2 : i = 20) (h3 : d = 0.20 * i) 
  (h4 : total_price = P - d) (h5 : final_price = 45) 
  (h6 : discount_amount = total_price - final_price) 
  (h7 : percentage = (discount_amount / total_price) * 100) : 
  percentage = 10 := 
by
  sorry

end coupon_percentage_l78_78266


namespace smallest_c_no_real_root_l78_78579

theorem smallest_c_no_real_root (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 5) ↔ c = -4 :=
by
  sorry

end smallest_c_no_real_root_l78_78579


namespace ratio_of_birds_to_trees_and_stones_l78_78714

theorem ratio_of_birds_to_trees_and_stones (stones birds : ℕ) (h_stones : stones = 40)
  (h_birds : birds = 400) (trees : ℕ) (h_trees : trees = 3 * stones + stones) :
  (birds : ℚ) / (trees + stones) = 2 :=
by
  -- The actual proof steps would go here.
  sorry

end ratio_of_birds_to_trees_and_stones_l78_78714


namespace melon_weights_l78_78476

-- We start by defining the weights of the individual melons.
variables {D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 : ℝ}

-- Define the weights of the given sets of three melons.
def W1 := D1 + D2 + D3
def W2 := D2 + D3 + D4
def W3 := D1 + D3 + D4
def W4 := D1 + D2 + D4
def W5 := D5 + D6 + D7
def W6 := D8 + D9 + D10

-- State the theorem to be proven.
theorem melon_weights (W1 W2 W3 W4 W5 W6 : ℝ) :
  (W1 + W2 + W3 + W4) / 3 + W5 + W6 = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 :=
sorry 

end melon_weights_l78_78476


namespace intersection_of_A_and_B_l78_78539

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l78_78539


namespace donation_value_l78_78913

def donation_in_yuan (usd: ℝ) (exchange_rate: ℝ): ℝ :=
  usd * exchange_rate

theorem donation_value :
  donation_in_yuan 1.2 6.25 = 7.5 :=
by
  -- Proof to be filled in
  sorry

end donation_value_l78_78913


namespace tylenol_pill_mg_l78_78242

noncomputable def tylenol_dose_per_pill : ℕ :=
  let mg_per_dose := 1000
  let hours_per_dose := 6
  let days := 14
  let pills := 112
  let doses_per_day := 24 / hours_per_dose
  let total_doses := doses_per_day * days
  let total_mg := total_doses * mg_per_dose
  total_mg / pills

theorem tylenol_pill_mg :
  tylenol_dose_per_pill = 500 := by
  sorry

end tylenol_pill_mg_l78_78242


namespace total_dolls_l78_78184

theorem total_dolls (big_boxes : ℕ) (dolls_per_big_box : ℕ) (small_boxes : ℕ) (dolls_per_small_box : ℕ)
  (h1 : dolls_per_big_box = 7) (h2 : big_boxes = 5) (h3 : dolls_per_small_box = 4) (h4 : small_boxes = 9) :
  big_boxes * dolls_per_big_box + small_boxes * dolls_per_small_box = 71 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end total_dolls_l78_78184


namespace smallest_n_mult_y_perfect_cube_l78_78306

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9

theorem smallest_n_mult_y_perfect_cube : ∃ n : ℕ, (∀ m : ℕ, y * n = m^3 → n = 1500) :=
sorry

end smallest_n_mult_y_perfect_cube_l78_78306


namespace Sara_team_wins_l78_78417

theorem Sara_team_wins (total_games losses wins : ℕ) (h1 : total_games = 12) (h2 : losses = 4) (h3 : wins = total_games - losses) :
  wins = 8 :=
by
  sorry

end Sara_team_wins_l78_78417


namespace spanish_teams_in_final_probability_l78_78426

noncomputable def probability_of_spanish_teams_in_final : ℚ :=
  let teams := 16
  let spanish_teams := 3
  let non_spanish_teams := teams - spanish_teams
  -- Probability calculation based on given conditions and solution steps
  1 - 7 / 15 * 6 / 14

theorem spanish_teams_in_final_probability :
  probability_of_spanish_teams_in_final = 4 / 5 :=
sorry

end spanish_teams_in_final_probability_l78_78426


namespace cevian_concurrency_l78_78237

-- Define the type for a triangle
structure Triangle :=
(A B C : Type)

-- Define the type for a point on a side of a triangle
structure IncircleTouchingPoints (T : Triangle) :=
(M N P : Type)

-- A definition for proving lines intersection in a triangle
def incircle_concurrent (T : Triangle) (P : IncircleTouchingPoints T) : Prop :=
  -- The proposition that AM, BN, and CP are concurrent
  ∃ O : Type,  -- There exists a point O
    O = sorry -- Placeholder to indicate that in a formal proof, we would assert this point

-- Given
variable {T : Triangle}
variable {P : IncircleTouchingPoints T}

-- The statement to prove
theorem cevian_concurrency : incircle_concurrent T P :=
sorry -- Proof placeholder

end cevian_concurrency_l78_78237


namespace num_rectangles_in_5x5_grid_l78_78991

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l78_78991


namespace increasing_interval_of_f_on_0_pi_l78_78216

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 4)

theorem increasing_interval_of_f_on_0_pi {ω : ℝ} (hω : ω > 0)
  (h_symmetry : ∀ x, f ω x = g x) :
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ ∀ x1 x2, (0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi) → f ω x1 < f ω x2} = 
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi / 8} :=
sorry

end increasing_interval_of_f_on_0_pi_l78_78216


namespace gym_hours_tuesday_equals_friday_l78_78676

-- Definitions
def weekly_gym_hours : ℝ := 5
def monday_hours : ℝ := 1.5
def wednesday_hours : ℝ := 1.5
def friday_hours : ℝ := 1
def total_weekly_hours : ℝ := weekly_gym_hours - (monday_hours + wednesday_hours + friday_hours)

-- Theorem statement
theorem gym_hours_tuesday_equals_friday : 
  total_weekly_hours = friday_hours :=
by
  sorry

end gym_hours_tuesday_equals_friday_l78_78676


namespace walter_equal_share_l78_78682

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l78_78682


namespace technicians_in_workshop_l78_78382

theorem technicians_in_workshop 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_tech : ℕ) 
  (avg_salary_rest : ℕ) 
  (total_salary : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (h1 : total_workers = 14) 
  (h2 : avg_salary_all = 8000) 
  (h3 : avg_salary_tech = 10000) 
  (h4 : avg_salary_rest = 6000) 
  (h5 : total_salary = total_workers * avg_salary_all) 
  (h6 : T + R = 14)
  (h7 : total_salary = 112000) 
  (h8 : total_salary = avg_salary_tech * T + avg_salary_rest * R) :
  T = 7 := 
by {
  -- Proof goes here
  sorry
} 

end technicians_in_workshop_l78_78382


namespace rita_money_left_l78_78415

theorem rita_money_left :
  let initial_amount : ℝ := 400
  let cost_short_dresses : ℝ := 5 * (20 - 0.1 * 20)
  let cost_pants : ℝ := 2 * 15
  let cost_jackets : ℝ := 2 * (30 - 0.15 * 30) + 2 * 30
  let cost_skirts : ℝ := 2 * 18 * 0.8
  let cost_tshirts : ℝ := 2 * 8
  let cost_transportation : ℝ := 5
  let total_spent : ℝ := cost_short_dresses + cost_pants + cost_jackets + cost_skirts + cost_tshirts + cost_transportation
  let money_left : ℝ := initial_amount - total_spent
  money_left = 119.2 :=
by 
  sorry

end rita_money_left_l78_78415


namespace quadratic_properties_l78_78206

noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

def min_value_passing_point (f : ℝ → ℝ) : Prop :=
  (f (-1) = -4) ∧ (f 0 = -3)

def intersects_x_axis (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  (f p1.1 = p1.2) ∧ (f p2.1 = p2.2)

def max_value_in_interval (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ max_val

theorem quadratic_properties :
  ∃ f : ℝ → ℝ,
    min_value_passing_point f ∧
    intersects_x_axis f (1, 0) (-3, 0) ∧
    max_value_in_interval f (-2) 2 5 :=
by
  sorry

end quadratic_properties_l78_78206


namespace exists_palindromic_product_l78_78693

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  digits = digits.reverse

theorem exists_palindromic_product (x : ℕ) (hx : ¬ (10 ∣ x)) : ∃ y : ℕ, is_palindrome (x * y) :=
by
  -- Prove that there exists a natural number y such that x * y is a palindromic number
  sorry

end exists_palindromic_product_l78_78693


namespace determine_OP_l78_78201

theorem determine_OP 
  (a b c d k : ℝ)
  (h1 : k * b ≤ c) 
  (h2 : (A : ℝ) = a)
  (h3 : (B : ℝ) = k * b)
  (h4 : (C : ℝ) = c)
  (h5 : (D : ℝ) = k * d)
  (AP_PD : ∀ (P : ℝ), (a - P) / (P - k * d) = k * (k * b - P) / (P - c))
  :
  ∃ P : ℝ, P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
sorry

end determine_OP_l78_78201


namespace total_red_marbles_l78_78246

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end total_red_marbles_l78_78246


namespace total_red_marbles_l78_78244

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end total_red_marbles_l78_78244


namespace dollar_expansion_l78_78631

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2 + a * b

theorem dollar_expansion : dollar ((x - y) ^ 3) ((y - x) ^ 3) = -((x - y) ^ 6) := by
  sorry

end dollar_expansion_l78_78631


namespace count_interesting_quadruples_l78_78630

def interesting_quadruples (a b c d : ℤ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + 2 * d > b + 2 * c 

theorem count_interesting_quadruples : 
  (∃ n : ℤ, n = 582 ∧ ∀ a b c d : ℤ, interesting_quadruples a b c d → n = 582) :=
sorry

end count_interesting_quadruples_l78_78630


namespace marbles_exchange_l78_78938

-- Define the initial number of marbles for Drew and Marcus
variables {D M x : ℕ}

-- Conditions
axiom Drew_initial (D M : ℕ) : D = M + 24
axiom Drew_after_give (D x : ℕ) : D - x = 25
axiom Marcus_after_receive (M x : ℕ) : M + x = 25

-- The goal is to prove: x = 12
theorem marbles_exchange : ∀ {D M x : ℕ}, D = M + 24 ∧ D - x = 25 ∧ M + x = 25 → x = 12 :=
by 
    sorry

end marbles_exchange_l78_78938


namespace find_x_value_l78_78005

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l78_78005


namespace isabella_houses_l78_78523

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end isabella_houses_l78_78523


namespace total_dolls_count_l78_78183

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end total_dolls_count_l78_78183


namespace probability_all_heads_or_tails_l78_78421

/-
Problem: Six fair coins are to be flipped. Prove that the probability that all six will be heads or all six will be tails is 1 / 32.
-/

theorem probability_all_heads_or_tails :
  let total_flips := 6,
      total_outcomes := Nat.pow 2 total_flips,              -- 2^6
      favorable_outcomes := 2 in                           -- [HHHHHH, TTTTTT]
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 32 :=    -- Probability calculation
by
  sorry

end probability_all_heads_or_tails_l78_78421


namespace power_sum_l78_78270

theorem power_sum : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end power_sum_l78_78270


namespace cakes_sold_to_baked_ratio_l78_78483

theorem cakes_sold_to_baked_ratio
  (cakes_per_day : ℕ) 
  (days : ℕ)
  (cakes_left : ℕ)
  (total_cakes : ℕ := cakes_per_day * days)
  (cakes_sold : ℕ := total_cakes - cakes_left) :
  cakes_per_day = 20 → 
  days = 9 → 
  cakes_left = 90 → 
  cakes_sold * 2 = total_cakes := 
by 
  intros 
  sorry

end cakes_sold_to_baked_ratio_l78_78483


namespace product_eq_one_of_abs_log_eq_l78_78650

theorem product_eq_one_of_abs_log_eq (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := 
sorry

end product_eq_one_of_abs_log_eq_l78_78650


namespace ratio_of_ages_in_two_years_l78_78605

theorem ratio_of_ages_in_two_years (S M : ℕ) (h1: M = S + 28) (h2: M + 2 = (S + 2) * 2) (h3: S = 26) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l78_78605


namespace minimum_trucks_needed_l78_78555

theorem minimum_trucks_needed {n : ℕ} (total_weight : ℕ) (box_weight : ℕ → ℕ) (truck_capacity : ℕ) :
  (total_weight = 10 ∧ truck_capacity = 3 ∧ (∀ b, box_weight b ≤ 1) ∧ (∃ n, 3 * n ≥ total_weight)) → n ≥ 5 :=
by
  -- We need to prove the statement based on the given conditions.
  sorry

end minimum_trucks_needed_l78_78555


namespace arrangment_ways_basil_tomato_l78_78620

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end arrangment_ways_basil_tomato_l78_78620


namespace original_speed_of_person_B_l78_78444

-- Let v_A and v_B be the speeds of person A and B respectively
variable (v_A v_B : ℝ)

-- Conditions for problem
axiom initial_ratio : v_A / v_B = (5 / 4 * v_A) / (v_B + 10)

-- The goal: Prove that v_B = 40
theorem original_speed_of_person_B : v_B = 40 := 
  sorry

end original_speed_of_person_B_l78_78444


namespace special_operation_value_l78_78729

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l78_78729


namespace germs_per_dish_l78_78387

/--
Given:
- the total number of germs is \(5.4 \times 10^6\),
- the number of petri dishes is 10,800,

Prove:
- the number of germs per dish is 500.
-/
theorem germs_per_dish (total_germs : ℝ) (petri_dishes: ℕ) (h₁: total_germs = 5.4 * 10^6) (h₂: petri_dishes = 10800) :
  (total_germs / petri_dishes = 500) :=
sorry

end germs_per_dish_l78_78387


namespace holloway_soccer_team_l78_78181

theorem holloway_soccer_team (P M : Finset ℕ) (hP_union_M : (P ∪ M).card = 20) 
(hP : P.card = 12) (h_int : (P ∩ M).card = 6) : M.card = 14 := 
by
  sorry

end holloway_soccer_team_l78_78181


namespace tan_sin_cos_eq_l78_78056

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l78_78056


namespace smallest_number_has_2020_divisors_l78_78339

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l78_78339


namespace find_a_if_f_is_odd_l78_78222

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_if_f_is_odd :
  (∀ x : ℝ, f 1 x = -f 1 (-x)) ↔ (1 = 1) :=
by
  sorry

end find_a_if_f_is_odd_l78_78222


namespace inequality_f_solution_minimum_g_greater_than_f_l78_78508

noncomputable def f (x : ℝ) := abs (x - 2) - abs (x + 1)

theorem inequality_f_solution : {x : ℝ | f x > 1} = {x | x < 0} :=
sorry

noncomputable def g (a x : ℝ) := (a * x^2 - x + 1) / x

theorem minimum_g_greater_than_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 0 < x → g a x > f x) ↔ 1 ≤ a :=
sorry

end inequality_f_solution_minimum_g_greater_than_f_l78_78508


namespace arithmetic_sequence_sixtieth_term_l78_78711

theorem arithmetic_sequence_sixtieth_term (a₁ a₂₁ a₆₀ d : ℕ) 
  (h1 : a₁ = 7)
  (h2 : a₂₁ = 47)
  (h3 : a₂₁ = a₁ + 20 * d) : 
  a₆₀ = a₁ + 59 * d := 
  by
  have HD : d = 2 := by 
    rw [h1] at h3
    rw [h2] at h3
    linarith
  rw [HD]
  rw [h1]
  sorry

end arithmetic_sequence_sixtieth_term_l78_78711


namespace solve_equation_l78_78701

theorem solve_equation (x : ℝ) (h : x ≠ 1) : -x^2 = (2 * x + 4) / (x - 1) → (x = -2 ∨ x = 1) :=
by
  sorry

end solve_equation_l78_78701


namespace player1_points_after_13_rotations_l78_78887

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l78_78887


namespace smallest_number_with_2020_divisors_is_correct_l78_78349

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l78_78349


namespace quadr_pyramid_edge_sum_is_36_l78_78440

def sum_edges_quad_pyr (hex_sum_edges : ℕ) (hex_num_edges : ℕ) (quad_num_edges : ℕ) : ℕ :=
  let length_one_edge := hex_sum_edges / hex_num_edges
  length_one_edge * quad_num_edges

theorem quadr_pyramid_edge_sum_is_36 :
  sum_edges_quad_pyr 81 18 8 = 36 :=
by
  -- We defer proof
  sorry

end quadr_pyramid_edge_sum_is_36_l78_78440


namespace collective_earnings_l78_78119

theorem collective_earnings:
  let lloyd_hours := 10.5
  let mary_hours := 12.0
  let tom_hours := 7.0
  let lloyd_normal_hours := 7.5
  let mary_normal_hours := 8.0
  let tom_normal_hours := 9.0
  let lloyd_rate := 4.5
  let mary_rate := 5.0
  let tom_rate := 6.0
  let lloyd_overtime_rate := 2.5 * lloyd_rate
  let mary_overtime_rate := 3.0 * mary_rate
  let tom_overtime_rate := 2.0 * tom_rate
  let lloyd_earnings := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours - lloyd_normal_hours) * lloyd_overtime_rate)
  let mary_earnings := (mary_normal_hours * mary_rate) + ((mary_hours - mary_normal_hours) * mary_overtime_rate)
  let tom_earnings := (tom_hours * tom_rate)
  let total_earnings := lloyd_earnings + mary_earnings + tom_earnings
  total_earnings = 209.50 := by
  sorry

end collective_earnings_l78_78119


namespace num_rectangles_in_5x5_grid_l78_78970

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l78_78970


namespace angle_between_clock_hands_at_7_30_l78_78448

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end angle_between_clock_hands_at_7_30_l78_78448


namespace find_number_l78_78164

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 90) : x = 4000 :=
by
  sorry

end find_number_l78_78164


namespace three_numbers_lcm_ratio_l78_78740

theorem three_numbers_lcm_ratio
  (x : ℕ)
  (h1 : 3 * x.gcd 4 = 1)
  (h2 : (3 * x * 4 * x) / x.gcd (3 * x) = 180)
  (h3 : ∃ y : ℕ, y = 5 * (3 * x))
  : (3 * x = 45 ∧ 4 * x = 60 ∧ 5 * (3 * x) = 225) ∧
      lcm (lcm (3 * x) (4 * x)) (5 * (3 * x)) = 900 :=
by
  sorry

end three_numbers_lcm_ratio_l78_78740


namespace intersection_A_B_l78_78542

-- Defining the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

-- Statement to prove
theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l78_78542


namespace value_of_a6_in_arithmetic_sequence_l78_78233

/-- In the arithmetic sequence {a_n}, if a_2 and a_{10} are the two roots of the equation
    x^2 + 12x - 8 = 0, prove that the value of a_6 is -6. -/
theorem value_of_a6_in_arithmetic_sequence :
  ∃ a_2 a_10 : ℤ, (a_2 + a_10 = -12 ∧
  (2: ℤ) * ((a_2 + a_10) / (2 * 1)) = a_2 + a_10 ) → 
  ∃ a_6: ℤ, a_6 = -6 :=
by
  sorry

end value_of_a6_in_arithmetic_sequence_l78_78233


namespace find_x_between_0_and_180_l78_78033

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l78_78033


namespace dihedral_angle_equivalence_l78_78070

namespace CylinderGeometry

variables {α β γ : ℝ} 

-- Given conditions
axiom axial_cross_section : Type
axiom point_on_circumference (C : axial_cross_section) : Prop
axiom dihedral_angle (α: ℝ) : Prop
axiom angle_CAB (β : ℝ) : Prop
axiom angle_CA1B (γ : ℝ) : Prop

-- Proven statement
theorem dihedral_angle_equivalence
    (hx : point_on_circumference C)
    (hα : dihedral_angle α)
    (hβ : angle_CAB β)
    (hγ : angle_CA1B γ):
  α = Real.arcsin (Real.cos β / Real.cos γ) :=
sorry

end CylinderGeometry

end dihedral_angle_equivalence_l78_78070


namespace smaller_angle_at_7_30_l78_78447

def clock_angle_deg_per_hour : ℝ := 30 

def minute_hand_angle_at_7_30 : ℝ := 180

def hour_hand_angle_at_7_30 : ℝ := 225

theorem smaller_angle_at_7_30 : 
  ∃ angle : ℝ, angle = 45 ∧ 
  (angle = |hour_hand_angle_at_7_30 - minute_hand_angle_at_7_30|) :=
begin
  sorry
end

end smaller_angle_at_7_30_l78_78447


namespace solve_tan_equation_l78_78043

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l78_78043


namespace find_n_l78_78634

theorem find_n :
  ∃ (n : ℕ), (0 < n) ∧ ((n+3)! + (n+1)! = n! * 728) ∧ (n = 8) :=
by
  sorry

end find_n_l78_78634


namespace randy_feeds_per_day_l78_78412

theorem randy_feeds_per_day
  (pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ)
  (h1 : pigs = 2) (h2 : total_feed_per_week = 140) (h3 : days_per_week = 7) :
  total_feed_per_week / pigs / days_per_week = 10 :=
by
  sorry

end randy_feeds_per_day_l78_78412


namespace max_campaign_making_animals_prime_max_campaign_making_animals_nine_l78_78592

theorem max_campaign_making_animals_prime (n : ℕ) (h_prime : Nat.Prime n) (h_ge : n ≥ 3) : 
  ∃ k, k = (n - 1) / 2 :=
by
  sorry

theorem max_campaign_making_animals_nine : ∃ k, k = 4 :=
by
  sorry

end max_campaign_making_animals_prime_max_campaign_making_animals_nine_l78_78592


namespace largest_divisor_of_expression_l78_78632

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 + n^3 - n - 1) :=
sorry

end largest_divisor_of_expression_l78_78632


namespace monkey_food_l78_78136

theorem monkey_food (e m m' e' : ℚ) (h1 : m = 3 / 4 * e) (h2 : m' = m + 2) (h3 : e' = e - 2) (h4 : m' = 4 / 3 * e') :
  m + e = 14 := by
  sorry

end monkey_food_l78_78136


namespace eval_six_times_f_l78_78688

def f (x : Int) : Int :=
  if x % 2 == 0 then
    x / 2
  else
    5 * x + 1

theorem eval_six_times_f : f (f (f (f (f (f 7))))) = 116 := 
by
  -- Skipping proof body (since it's not required)
  sorry

end eval_six_times_f_l78_78688


namespace circumscribed_circle_area_of_pentagon_l78_78464

noncomputable def pentagon_side_length : ℝ := 10
noncomputable def sin_36 : ℝ := Real.sin (36 * Real.pi / 180)
noncomputable def radius (s : ℝ) : ℝ := s / (2 * sin_36)
noncomputable def circumscribed_circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area_of_pentagon :
  circumscribed_circle_area (radius pentagon_side_length) = 72.35 * Real.pi :=
by
  sorry

end circumscribed_circle_area_of_pentagon_l78_78464


namespace largest_y_l78_78394

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

theorem largest_y (x y : ℕ) (hx : x ≥ y) (hy : y ≥ 3) 
  (h : (interior_angle x * 28) = (interior_angle y * 29)) :
  y = 57 :=
by
  sorry

end largest_y_l78_78394


namespace tan_ratio_alpha_beta_l78_78951

theorem tan_ratio_alpha_beta 
  (α β : ℝ) 
  (h1 : Real.sin (α + β) = 1 / 5) 
  (h2 : Real.sin (α - β) = 3 / 5) : 
  Real.tan α / Real.tan β = -1 :=
sorry

end tan_ratio_alpha_beta_l78_78951


namespace crystal_final_segment_distance_l78_78628

theorem crystal_final_segment_distance :
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2 -- as nx, ny
  let southwest_component := southwest_distance / Real.sqrt 2 -- as sx, sy
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  Real.sqrt (net_north^2 + net_west^2) = 2 * Real.sqrt 3 :=
by
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2
  let southwest_component := southwest_distance / Real.sqrt 2
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  exact sorry

end crystal_final_segment_distance_l78_78628


namespace fraction_representation_of_2_375_l78_78896

theorem fraction_representation_of_2_375 : 2.375 = 19 / 8 := by
  sorry

end fraction_representation_of_2_375_l78_78896


namespace speed_goods_train_l78_78756

def length_train : ℝ := 50
def length_platform : ℝ := 250
def time_crossing : ℝ := 15

/-- The speed of the goods train in km/hr given the length of the train, the length of the platform, and the time to cross the platform. -/
theorem speed_goods_train :
  (length_train + length_platform) / time_crossing * 3.6 = 72 :=
by
  sorry

end speed_goods_train_l78_78756


namespace james_distance_ridden_l78_78101

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end james_distance_ridden_l78_78101


namespace smallest_n_with_2020_divisors_l78_78342

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l78_78342


namespace isabella_houses_problem_l78_78524

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end isabella_houses_problem_l78_78524


namespace sam_wins_l78_78831

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l78_78831


namespace ceil_add_eq_double_of_int_l78_78146

theorem ceil_add_eq_double_of_int {x : ℤ} (h : ⌈(x : ℝ)⌉ + ⌊(x : ℝ)⌋ = 2 * (x : ℝ)) : ⌈(x : ℝ)⌉ + x = 2 * x :=
by
  sorry

end ceil_add_eq_double_of_int_l78_78146


namespace correct_equation_l78_78602

theorem correct_equation (x : ℝ) (h1 : 2000 > 0) (h2 : x > 0) (h3 : x + 40 > 0) :
  (2000 / x) - (2000 / (x + 40)) = 3 :=
by
  sorry

end correct_equation_l78_78602


namespace Sam_wins_probability_l78_78835

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l78_78835


namespace algebraic_expression_evaluation_l78_78293

theorem algebraic_expression_evaluation (x y : ℝ) : 
  3 * (x^2 - 2 * x * y + y^2) - 3 * (x^2 - 2 * x * y + y^2 - 1) = 3 :=
by
  sorry

end algebraic_expression_evaluation_l78_78293


namespace tony_lottery_winning_l78_78892

theorem tony_lottery_winning
  (tickets : ℕ) (winning_numbers : ℕ) (worth_per_number : ℕ) (identical_numbers : Prop)
  (h_tickets : tickets = 3) (h_winning_numbers : winning_numbers = 5) (h_worth_per_number : worth_per_number = 20)
  (h_identical_numbers : identical_numbers) :
  (tickets * (winning_numbers * worth_per_number) = 300) :=
by
  sorry

end tony_lottery_winning_l78_78892


namespace num_rectangles_in_5x5_grid_l78_78993

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l78_78993


namespace equation_linear_implies_k_equals_neg2_l78_78663

theorem equation_linear_implies_k_equals_neg2 (k : ℤ) (x : ℝ) :
  (k - 2) * x^(abs k - 1) = k + 1 → abs k - 1 = 1 ∧ k - 2 ≠ 0 → k = -2 :=
by
  sorry

end equation_linear_implies_k_equals_neg2_l78_78663


namespace find_a_l78_78247

def M : Set ℝ := {x | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem find_a (a : ℝ) : N a ⊆ M ↔ a = -1 ∨ a = 0 ∨ a = 2/3 := 
by
  sorry

end find_a_l78_78247


namespace maximum_candies_after_20_hours_l78_78550

-- Define a function to compute the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Define the recursive function to model the candy process
def candies_after_hours (n : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then n 
  else candies_after_hours (n + sum_of_digits n) (hours - 1)

theorem maximum_candies_after_20_hours :
  candies_after_hours 1 20 = 148 :=
sorry

end maximum_candies_after_20_hours_l78_78550


namespace number_of_rectangles_in_grid_l78_78997

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l78_78997


namespace find_x_tan_eq_l78_78011

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l78_78011


namespace peter_erasers_l78_78411

theorem peter_erasers (initial_erasers : ℕ) (extra_erasers : ℕ) (final_erasers : ℕ)
  (h1 : initial_erasers = 8) (h2 : extra_erasers = 3) : final_erasers = 11 :=
by
  sorry

end peter_erasers_l78_78411


namespace num_rectangles_in_5x5_grid_l78_78979

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l78_78979


namespace employee_payment_proof_l78_78453

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price as 20 percent above the wholesale cost
def retail_price (C_w : ℝ) : ℝ := C_w + 0.2 * C_w

-- Define the employee discount on the retail price
def employee_discount (C_r : ℝ) : ℝ := 0.15 * C_r

-- Define the amount paid by the employee
def amount_paid_by_employee (C_w : ℝ) : ℝ :=
  let C_r := retail_price C_w
  let D_e := employee_discount C_r
  C_r - D_e

-- Main theorem to prove the employee paid $204
theorem employee_payment_proof : amount_paid_by_employee wholesale_cost = 204 :=
by
  sorry

end employee_payment_proof_l78_78453


namespace tenth_term_of_sequence_l78_78191

variable (a : ℕ → ℚ) (n : ℕ)

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem tenth_term_of_sequence :
  let a₁ := (5 : ℚ)
  let r := (5 / 3 : ℚ)
  geometric_sequence a₁ r 10 = (9765625 / 19683 : ℚ) :=
by
  sorry

end tenth_term_of_sequence_l78_78191


namespace clever_seven_year_count_l78_78089

def isCleverSevenYear (y : Nat) : Bool :=
  let d1 := y / 1000
  let d2 := (y % 1000) / 100
  let d3 := (y % 100) / 10
  let d4 := y % 10
  d1 + d2 + d3 + d4 = 7

theorem clever_seven_year_count : 
  ∃ n, n = 21 ∧ ∀ y, 2000 ≤ y ∧ y ≤ 2999 → isCleverSevenYear y = true ↔ n = 21 :=
by 
  sorry

end clever_seven_year_count_l78_78089


namespace distance_to_right_focus_l78_78076

open Real

-- Define the elements of the problem
variable (a c : ℝ)
variable (P : ℝ × ℝ) -- Point P on the hyperbola
variable (F1 F2 : ℝ × ℝ) -- Left and right foci
variable (D : ℝ) -- The left directrix

-- Define conditions as Lean statements
def hyperbola_eq : Prop := (a ≠ 0) ∧ (c ≠ 0) ∧ (P.1^2 / a^2 - P.2^2 / 16 = 1)
def point_on_right_branch : Prop := P.1 > 0
def distance_diff : Prop := abs (dist P F1 - dist P F2) = 6
def distance_to_left_directrix : Prop := abs (P.1 - D) = 34 / 5

-- Define theorem to prove the distance from P to the right focus
theorem distance_to_right_focus
  (hp : hyperbola_eq a c P)
  (hbranch : point_on_right_branch P)
  (hdiff : distance_diff P F1 F2)
  (hdirectrix : distance_to_left_directrix P D) :
  dist P F2 = 16 / 3 :=
sorry

end distance_to_right_focus_l78_78076


namespace log_exp_mod_a_l78_78552

open padic

-- Define the theorem with the given conditions
theorem log_exp_mod_a (a x y : ℤ_[p]) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) (ha_nonzero : a ≠ 0) 
  (hpx : p ∣ x) (hpa_xy : p * a ∣ x * y) :
  (1/y * ((1 + x)^y - 1) / x) ≡ (log (1 + x) / x) [MOD a] :=
by 
  sorry

end log_exp_mod_a_l78_78552


namespace gcd_360_1260_l78_78291

theorem gcd_360_1260 : gcd 360 1260 = 180 := by
  /- 
  Prime factorization of 360 and 1260 is given:
  360 = 2^3 * 3^2 * 5
  1260 = 2^2 * 3^2 * 5 * 7
  These conditions are implicitly used to deduce the answer.
  -/
  sorry

end gcd_360_1260_l78_78291


namespace find_x_tan_identity_l78_78050

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l78_78050


namespace find_x_tan_eq_l78_78013

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l78_78013


namespace hacker_cannot_change_grades_l78_78761

theorem hacker_cannot_change_grades :
  ¬ ∃ n1 n2 n3 n4 : ℤ,
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 := by
  sorry

end hacker_cannot_change_grades_l78_78761


namespace Taehyung_age_l78_78425

variable (T U : Nat)

-- Condition 1: Taehyung is 17 years younger than his uncle
def condition1 : Prop := U = T + 17

-- Condition 2: Four years later, the sum of their ages is 43
def condition2 : Prop := (T + 4) + (U + 4) = 43

-- The goal is to prove that Taehyung's current age is 9, given the conditions above
theorem Taehyung_age : condition1 T U ∧ condition2 T U → T = 9 := by
  sorry

end Taehyung_age_l78_78425


namespace rectangles_in_grid_l78_78989

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l78_78989


namespace smallest_percent_increase_is_100_l78_78097

-- The values for each question
def prize_values : List ℕ := [150, 300, 450, 900, 1800, 3600, 7200, 14400, 28800, 57600, 115200, 230400, 460800, 921600, 1843200]

-- Definition of percent increase calculation
def percent_increase (old new : ℕ) : ℕ :=
  ((new - old : ℕ) * 100) / old

-- Lean theorem statement
theorem smallest_percent_increase_is_100 :
  percent_increase (prize_values.get! 5) (prize_values.get! 6) = 100 ∧
  percent_increase (prize_values.get! 7) (prize_values.get! 8) = 100 ∧
  percent_increase (prize_values.get! 9) (prize_values.get! 10) = 100 ∧
  percent_increase (prize_values.get! 10) (prize_values.get! 11) = 100 ∧
  percent_increase (prize_values.get! 13) (prize_values.get! 14) = 100 :=
by
  sorry

end smallest_percent_increase_is_100_l78_78097


namespace walter_equal_share_l78_78681

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l78_78681


namespace find_a_b_range_of_a_l78_78365

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

-- Problem 1
theorem find_a_b (a b : ℝ) :
  f a 1 = 0 ∧ f a b = 0 ∧ (∀ x, f a x > 0 ↔ x < 1 ∨ x > b) → a = 1 ∧ b = 2 := sorry

-- Problem 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → (0 ≤ a ∧ a < 8/9) := sorry

end find_a_b_range_of_a_l78_78365


namespace Walter_receives_49_bananas_l78_78677

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l78_78677


namespace cost_of_five_juices_l78_78105

-- Given conditions as assumptions
variables {J S : ℝ}

axiom h1 : 2 * S = 6
axiom h2 : S + J = 5

-- Prove the statement
theorem cost_of_five_juices : 5 * J = 10 :=
sorry

end cost_of_five_juices_l78_78105


namespace sum_series_l78_78493

theorem sum_series (s : ℕ → ℝ) 
  (h : ∀ n : ℕ, s n = (n+1) / (4 : ℝ)^(n+1)) : 
  tsum s = (4 / 9 : ℝ) :=
sorry

end sum_series_l78_78493


namespace volume_ratio_of_cubes_l78_78901

theorem volume_ratio_of_cubes :
  (4^3 / 10^3 : ℚ) = 8 / 125 := by
  sorry

end volume_ratio_of_cubes_l78_78901


namespace solve_system_l78_78557

theorem solve_system :
  ∃ x y : ℝ, (x^3 + y^3) * (x^2 + y^2) = 64 ∧ x + y = 2 ∧ 
  ((x = 1 + Real.sqrt (5 / 3) ∧ y = 1 - Real.sqrt (5 / 3)) ∨ 
   (x = 1 - Real.sqrt (5 / 3) ∧ y = 1 + Real.sqrt (5 / 3))) :=
by
  sorry

end solve_system_l78_78557


namespace sin_double_angle_tan_double_angle_l78_78299

-- Step 1: Define the first problem in Lean 4.
theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 12 / 13) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin (2 * α) = -120 / 169 := 
sorry

-- Step 2: Define the second problem in Lean 4.
theorem tan_double_angle (α : ℝ) (h1 : Real.tan α = 1 / 2) :
  Real.tan (2 * α) = 4 / 3 := 
sorry

end sin_double_angle_tan_double_angle_l78_78299


namespace total_seeds_eaten_l78_78313

def first_seeds := 78
def second_seeds := 53
def third_seeds := second_seeds + 30

theorem total_seeds_eaten : first_seeds + second_seeds + third_seeds = 214 := by
  -- Sorry, placeholder for proof
  sorry

end total_seeds_eaten_l78_78313


namespace ratio_of_running_to_swimming_l78_78890

variable (Speed_swimming Time_swimming Distance_total Speed_factor : ℕ)

theorem ratio_of_running_to_swimming :
  let Distance_swimming := Speed_swimming * Time_swimming
  let Distance_running := Distance_total - Distance_swimming
  let Speed_running := Speed_factor * Speed_swimming
  let Time_running := Distance_running / Speed_running
  (Distance_total = 12) ∧
  (Speed_swimming = 2) ∧
  (Time_swimming = 2) ∧
  (Speed_factor = 4) →
  (Time_running : ℕ) / Time_swimming = 1 / 2 :=
by
  intros
  sorry

end ratio_of_running_to_swimming_l78_78890


namespace points_player_1_after_13_rotations_l78_78884

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l78_78884


namespace solve_tan_equation_l78_78042

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l78_78042


namespace transformed_interval_l78_78627

noncomputable def transformation (x : ℝ) : ℝ := 8 * x - 2

theorem transformed_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2 ≤ transformation x ∧ transformation x ≤ 6 := by
  intro x h
  unfold transformation
  sorry

end transformed_interval_l78_78627


namespace min_value_quadratic_l78_78642

theorem min_value_quadratic (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 1) :
  (∀ x, (a * x^2 + 2 * x + b = 0) → x = -1 / a) →
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ (∀ a b, a > b → b > 0 → a * b = 1 →
     c ≤ (a^2 + b^2) / (a - b)) :=
by
  sorry

end min_value_quadratic_l78_78642


namespace water_polo_team_selection_l78_78120

theorem water_polo_team_selection :
  (18 * 17 * Nat.choose 16 5) = 1338176 := by
  sorry

end water_polo_team_selection_l78_78120


namespace max_rectangle_area_l78_78434

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 48) : x * y ≤ 144 :=
by
  sorry

end max_rectangle_area_l78_78434


namespace isabella_houses_l78_78522

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end isabella_houses_l78_78522


namespace value_of_a_star_b_l78_78724

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l78_78724


namespace dihedral_minus_solid_equals_expression_l78_78695

-- Definitions based on the conditions provided.
noncomputable def sumDihedralAngles (P : Polyhedron) : ℝ := sorry
noncomputable def sumSolidAngles (P : Polyhedron) : ℝ := sorry
def numFaces (P : Polyhedron) : ℕ := sorry

-- Theorem statement we want to prove.
theorem dihedral_minus_solid_equals_expression (P : Polyhedron) :
  sumDihedralAngles P - sumSolidAngles P = 2 * Real.pi * (numFaces P - 2) :=
sorry

end dihedral_minus_solid_equals_expression_l78_78695


namespace player1_points_after_13_rotations_l78_78888

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l78_78888


namespace candy_mixture_cost_l78_78303

/-- 
A club mixes 15 pounds of candy worth $8.00 per pound with 30 pounds of candy worth $5.00 per pound.
We need to find the cost per pound of the mixture.
-/
theorem candy_mixture_cost :
    (15 * 8 + 30 * 5) / (15 + 30) = 6 := 
by
  sorry

end candy_mixture_cost_l78_78303


namespace xiaoqiang_average_score_l78_78906

theorem xiaoqiang_average_score
    (x : ℕ)
    (prev_avg : ℝ)
    (next_score : ℝ)
    (target_avg : ℝ)
    (h_prev_avg : prev_avg = 84)
    (h_next_score : next_score = 100)
    (h_target_avg : target_avg = 86) :
    (86 * x - (84 * (x - 1)) = 100) → x = 8 := 
by
  intros h_eq
  sorry

end xiaoqiang_average_score_l78_78906


namespace variance_of_data_set_is_4_l78_78363

/-- The data set for which we want to calculate the variance --/
def data_set : List ℝ := [2, 4, 5, 6, 8]

/-- The mean of the data set --/
noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Calculation of the variance of a list given its mean
noncomputable def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem variance_of_data_set_is_4 :
  variance data_set (mean data_set) = 4 :=
by
  sorry

end variance_of_data_set_is_4_l78_78363


namespace rectangles_in_grid_l78_78987

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l78_78987


namespace correct_statement_dice_roll_l78_78586

theorem correct_statement_dice_roll :
  (∃! s, s ∈ ["When flipping a coin, the head side will definitely face up.",
              "The probability of precipitation tomorrow is 80% means that 80% of the areas will have rain tomorrow.",
              "To understand the lifespan of a type of light bulb, it is appropriate to use a census method.",
              "When rolling a dice, the number will definitely not be greater than 6."] ∧
          s = "When rolling a dice, the number will definitely not be greater than 6.") :=
by {
  sorry
}

end correct_statement_dice_roll_l78_78586


namespace probability_all_students_same_canteen_l78_78609

theorem probability_all_students_same_canteen (num_canteens : ℕ) (num_students : ℕ) :
  num_canteens = 2 → num_students = 3 → 
  let p := (2 : ℕ) / (2 ^ num_students : ℕ) in 
  p = (1 : ℕ) / 4 :=
by
  intros h1 h2,
  have h_total_outcomes : 2 ^ 3 = 8 := by norm_num,
  have h_favorable_outcomes : 2 = 2 := rfl,
  let p := 2 / 8,
  have hp : p = 1 / 4 := by norm_num,
  rw [h1, h2, h_total_outcomes, h_favorable_outcomes, hp],
  sorry

end probability_all_students_same_canteen_l78_78609


namespace parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l78_78077

theorem parabola_right_shift_unique_intersection (p : ℚ) :
  let y := 2 * (x - p)^2;
  (x * x - 4) = 0 →
  p = 31 / 8 := sorry

theorem parabola_down_shift_unique_intersection (q : ℚ) :
  let y := 2 * x^2 - q;
  (x * x - 4) = 0 →
  q = 31 / 8 := sorry

end parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l78_78077


namespace expression_min_value_l78_78066

theorem expression_min_value (a b c k : ℝ) (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  (1 : ℝ) / c^2 * ((k * c - a)^2 + (a + c)^2 + (c - a)^2) ≥ k^2 / 3 + 2 :=
sorry

end expression_min_value_l78_78066


namespace alice_age_proof_l78_78562

-- Definitions derived from the conditions
def alice_pens : ℕ := 60
def clara_pens : ℕ := (2 * alice_pens) / 5
def clara_age_in_5_years : ℕ := 61
def clara_current_age : ℕ := clara_age_in_5_years - 5
def age_difference : ℕ := alice_pens - clara_pens

-- Proof statement to be proved
theorem alice_age_proof : (clara_current_age - age_difference = 20) :=
sorry

end alice_age_proof_l78_78562


namespace smallest_unwritable_number_l78_78779

theorem smallest_unwritable_number :
  ∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d) := sorry

end smallest_unwritable_number_l78_78779


namespace ratio_of_ages_l78_78414

theorem ratio_of_ages (sandy_future_age : ℕ) (sandy_years_future : ℕ) (molly_current_age : ℕ)
  (h1 : sandy_future_age = 42) (h2 : sandy_years_future = 6) (h3 : molly_current_age = 27) :
  (sandy_future_age - sandy_years_future) / gcd (sandy_future_age - sandy_years_future) molly_current_age = 
    4 / 3 :=
by
  sorry

end ratio_of_ages_l78_78414


namespace David_fewer_crunches_l78_78772

-- Definitions as per conditions.
def Zachary_crunches := 62
def David_crunches := 45

-- Proof statement for how many fewer crunches David did compared to Zachary.
theorem David_fewer_crunches : Zachary_crunches - David_crunches = 17 := by
  -- Proof details would go here, but we skip them with 'sorry'.
  sorry

end David_fewer_crunches_l78_78772


namespace number_of_sides_of_regular_polygon_l78_78370

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end number_of_sides_of_regular_polygon_l78_78370


namespace diff_of_squares_expression_l78_78159

theorem diff_of_squares_expression (m n : ℝ) :
  (3 * m + n) * (3 * m - n) = (3 * m)^2 - n^2 :=
by
  sorry

end diff_of_squares_expression_l78_78159


namespace eggs_sally_bought_is_correct_l78_78553

def dozen := 12

def eggs_sally_bought (dozens : Nat) : Nat :=
  dozens * dozen

theorem eggs_sally_bought_is_correct :
  eggs_sally_bought 4 = 48 :=
by
  sorry

end eggs_sally_bought_is_correct_l78_78553


namespace count_rectangles_5x5_l78_78982

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l78_78982


namespace pigeonhole_divisible_l78_78667

theorem pigeonhole_divisible (n : ℕ) (a : Fin (n + 1) → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n) :
  ∃ i j, i ≠ j ∧ a i ∣ a j :=
by
  sorry

end pigeonhole_divisible_l78_78667


namespace find_a5_l78_78202

variable {a : ℕ → ℝ}  -- Define the sequence a(n)

-- Define the conditions of the problem
variable (a1_positive : ∀ n, a n > 0)
variable (geo_seq : ∀ n, a (n + 1) = a n * 2)
variable (condition : (a 3) * (a 11) = 16)

theorem find_a5 (a1_positive : ∀ n, a n > 0) (geo_seq : ∀ n, a (n + 1) = a n * 2)
(condition : (a 3) * (a 11) = 16) : a 5 = 1 := by
  sorry

end find_a5_l78_78202


namespace sum_of_terms_7_8_9_l78_78400

namespace ArithmeticSequence

-- Define the sequence and its properties
variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 0 + n * (n - 1) / 2 * (a 1 - a 0)

def condition3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def condition5 (S : ℕ → ℤ) : Prop :=
  S 5 = 30

-- Main statement to prove
theorem sum_of_terms_7_8_9 :
  is_arithmetic_sequence a →
  (∀ n, S n = sum_first_n_terms a n) →
  condition3 S →
  condition5 S →
  a 7 + a 8 + a 9 = 63 :=
by
  sorry

end ArithmeticSequence

end sum_of_terms_7_8_9_l78_78400


namespace ratio_dog_to_hamster_l78_78889

noncomputable def dog_lifespan : ℝ := 10
noncomputable def hamster_lifespan : ℝ := 2.5

theorem ratio_dog_to_hamster : dog_lifespan / hamster_lifespan = 4 :=
by
  sorry

end ratio_dog_to_hamster_l78_78889


namespace area_of_triangle_KDC_l78_78520

open Real

noncomputable def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem area_of_triangle_KDC
  (radius : ℝ) (chord_length : ℝ) (seg_KA : ℝ)
  (OX distance_DY : ℝ)
  (parallel : ∀ (PA PB : ℝ), PA = PB)
  (collinear : ∀ (PK PA PQ PB : ℝ), PK + PA + PQ + PB = PK + PQ + PA + PB)
  (hyp_radius : radius = 10)
  (hyp_chord_length : chord_length = 12)
  (hyp_seg_KA : seg_KA = 24)
  (hyp_OX : OX = 8)
  (hyp_distance_DY : distance_DY = 8) :
  triangle_area chord_length distance_DY = 48 :=
  by
  sorry

end area_of_triangle_KDC_l78_78520


namespace fresh_water_needed_l78_78843

noncomputable def mass_of_seawater : ℝ := 30
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def desired_salt_concentration : ℝ := 0.015

theorem fresh_water_needed :
  ∃ (fresh_water_mass : ℝ), 
    fresh_water_mass = 70 ∧ 
    (mass_of_seawater * initial_salt_concentration) / (mass_of_seawater + fresh_water_mass) = desired_salt_concentration :=
by
  sorry

end fresh_water_needed_l78_78843


namespace correct_conclusions_l78_78277

open Real

noncomputable def parabola (a b c : ℝ) : ℝ → ℝ :=
  λ x => a*x^2 + b*x + c

theorem correct_conclusions (a b c m n : ℝ)
  (h1 : c < 0)
  (h2 : parabola a b c 1 = 1)
  (h3 : parabola a b c m = 0)
  (h4 : parabola a b c n = 0)
  (h5 : n ≥ 3) :
  (4*a*c - b^2 < 4*a) ∧
  (n = 3 → ∃ t : ℝ, parabola a b c 2 = t ∧ t > 1) ∧
  (∀ x : ℝ, parabola a b (c - 1) x = 0 → (0 < m ∧ m ≤ 1/3)) :=
sorry

end correct_conclusions_l78_78277


namespace find_x_value_l78_78006

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l78_78006


namespace smallest_four_digit_multiple_of_18_l78_78004

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n > 999 ∧ n < 10000 ∧ 18 ∣ n ∧ (∀ m : ℕ, m > 999 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l78_78004


namespace contrapositive_proposition_l78_78561

theorem contrapositive_proposition (x : ℝ) : 
  (x^2 = 1 → (x = 1 ∨ x = -1)) ↔ ((x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by
  sorry

end contrapositive_proposition_l78_78561


namespace average_cookies_per_package_l78_78933

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies_per_package :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 125 / 9 :=
by
  sorry

end average_cookies_per_package_l78_78933


namespace solve_tan_equation_l78_78044

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l78_78044


namespace gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l78_78626

theorem gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1 :
  Int.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
  -- proof goes here
  sorry

end gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l78_78626


namespace solve_for_A_l78_78778

theorem solve_for_A (A : ℚ) : 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 → A = -4/3 :=
by
  sorry

end solve_for_A_l78_78778


namespace max_dist_AC_l78_78651

open Real EuclideanGeometry

variables (P A B C : ℝ × ℝ)
  (hPA : dist P A = 1)
  (hPB : dist P B = 1)
  (hPA_PB : dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = - 1 / 2)
  (hBC : dist B C = 1)

theorem max_dist_AC : ∃ C : ℝ × ℝ, dist A C ≤ dist A B + dist B C ∧ dist A C = sqrt 3 + 1 :=
by
  sorry

end max_dist_AC_l78_78651


namespace factor_expression_l78_78189

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end factor_expression_l78_78189


namespace inequality_solution_l78_78732

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end inequality_solution_l78_78732


namespace impossible_odd_sum_l78_78225

theorem impossible_odd_sum (n m : ℤ) (h1 : (n^3 + m^3) % 2 = 0) (h2 : (n^3 + m^3) % 4 = 0) : (n + m) % 2 = 0 :=
sorry

end impossible_odd_sum_l78_78225


namespace repeating_block_length_7_div_13_l78_78858

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l78_78858


namespace polynomial_expansion_l78_78217

theorem polynomial_expansion :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a_1 * x^4 + a_2 * x^3 + a_3 * x^2 + 16 * x + 4) :=
by
  sorry

end polynomial_expansion_l78_78217


namespace num_rectangles_in_5x5_grid_l78_78975

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l78_78975


namespace sophomores_selected_l78_78302

variables (total_students freshmen sophomores juniors selected_students : ℕ)
def high_school_data := total_students = 2800 ∧ freshmen = 970 ∧ sophomores = 930 ∧ juniors = 900 ∧ selected_students = 280

theorem sophomores_selected (h : high_school_data total_students freshmen sophomores juniors selected_students) :
  (930 / 2800 : ℚ) * 280 = 93 := by
  sorry

end sophomores_selected_l78_78302


namespace repeat_block_of_7_div_13_l78_78861

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l78_78861


namespace sam_wins_probability_l78_78836

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l78_78836


namespace major_minor_axis_lengths_foci_vertices_coordinates_l78_78507

-- Given conditions
def ellipse_eq (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

-- Proof Tasks
theorem major_minor_axis_lengths : 
  (∃ a b : ℝ, a = 5 ∧ b = 4 ∧ 2 * a = 10) :=
by sorry

theorem foci_vertices_coordinates : 
  (∃ c : ℝ, 
    (c = 3) ∧ 
    (∀ x y : ℝ, ellipse_eq x y → (x = 0 → y = 4 ∨ y = -4) ∧ (y = 0 → x = 5 ∨ x = -5))) :=
by sorry

end major_minor_axis_lengths_foci_vertices_coordinates_l78_78507


namespace intersection_of_A_and_B_l78_78503

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

theorem intersection_of_A_and_B : A ∩ B = {z | 2 ≤ z ∧ z < 4} :=
by
  sorry

end intersection_of_A_and_B_l78_78503


namespace find_x_value_l78_78010

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l78_78010


namespace coins_left_l78_78106

-- Define the initial number of coins from each source
def piggy_bank_coins : ℕ := 15
def brother_coins : ℕ := 13
def father_coins : ℕ := 8

-- Define the number of coins given to Laura
def given_to_laura_coins : ℕ := 21

-- Define the total initial coins collected by Kylie
def total_initial_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

-- Lean statement to prove
theorem coins_left : total_initial_coins - given_to_laura_coins = 15 :=
by
  sorry

end coins_left_l78_78106


namespace sandwiches_left_l78_78690

theorem sandwiches_left 
    (initial_sandwiches : ℕ)
    (first_coworker : ℕ)
    (second_coworker : ℕ)
    (third_coworker : ℕ)
    (kept_sandwiches : ℕ) :
    initial_sandwiches = 50 →
    first_coworker = 4 →
    second_coworker = 3 →
    third_coworker = 2 * first_coworker →
    kept_sandwiches = 3 * second_coworker →
    initial_sandwiches - (first_coworker + second_coworker + third_coworker + kept_sandwiches) = 26 :=
by
  intros h_initial h_first h_second h_third h_kept
  rw [h_initial, h_first, h_second, h_third, h_kept]
  simp
  norm_num
  sorry

end sandwiches_left_l78_78690


namespace find_x_l78_78040

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l78_78040


namespace cassie_water_bottle_ounces_l78_78487

-- Define the given quantities
def cups_per_day : ℕ := 12
def ounces_per_cup : ℕ := 8
def refills_per_day : ℕ := 6

-- Define the total ounces of water Cassie drinks per day
def total_ounces_per_day := cups_per_day * ounces_per_cup

-- Define the ounces her water bottle holds
def ounces_per_bottle := total_ounces_per_day / refills_per_day

-- Prove the statement
theorem cassie_water_bottle_ounces : 
  ounces_per_bottle = 16 := by 
  sorry

end cassie_water_bottle_ounces_l78_78487


namespace evaluate_expression_l78_78636

theorem evaluate_expression :
  (4 * 6) / (12 * 14) * ((8 * 12 * 14) / (4 * 6 * 8)) = 1 := 
by 
  sorry

end evaluate_expression_l78_78636


namespace intersection_of_A_and_B_l78_78540

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  by
    sorry

end intersection_of_A_and_B_l78_78540


namespace kris_fraction_l78_78288

-- Definitions based on problem conditions
def Trey (kris : ℕ) := 7 * kris
def Kristen := 12
def Trey_kristen_diff := 9
def Kris_fraction_to_Kristen (kris : ℕ) : ℚ := kris / Kristen

-- Theorem statement: Proving the required fraction
theorem kris_fraction (kris : ℕ) (h1 : Trey kris = Kristen + Trey_kristen_diff) : 
  Kris_fraction_to_Kristen kris = 1 / 4 :=
by
  sorry

end kris_fraction_l78_78288


namespace solve_inequality_l78_78702

theorem solve_inequality (a x : ℝ) :
  (a = 1/2 → (x ≠ 1/2 → (x - a) * (x + a - 1) > 0)) ∧
  (a < 1/2 → ((x > (1 - a) ∨ x < a) → (x - a) * (x + a - 1) > 0)) ∧
  (a > 1/2 → ((x > a ∨ x < (1 - a)) → (x - a) * (x + a - 1) > 0)) :=
by
  sorry

end solve_inequality_l78_78702


namespace range_of_m_l78_78356

theorem range_of_m (m : ℝ) :
  (∃ (x1 x2 : ℝ), (2*x1^2 - 2*x1 + 3*m - 1 = 0 ∧ 2*x2^2 - 2*x2 + 3*m - 1 = 0) ∧ (x1 * x2 > x1 + x2 - 4)) →
  -5/3 < m ∧ m ≤ 1/2 :=
by
  sorry

end range_of_m_l78_78356


namespace project_hours_l78_78263

variable (K : ℕ)

theorem project_hours 
    (h_total : K + 2 * K + 3 * K + K / 2 = 180)
    (h_k_nearest : K = 28) :
    3 * K - K = 56 := 
by
  -- Proof goes here
  sorry

end project_hours_l78_78263


namespace total_amount_for_gifts_l78_78094

theorem total_amount_for_gifts (workers_per_block : ℕ) (worth_per_gift : ℕ) (number_of_blocks : ℕ)
  (h1 : workers_per_block = 100) (h2 : worth_per_gift = 4) (h3 : number_of_blocks = 10) :
  (workers_per_block * worth_per_gift * number_of_blocks = 4000) := by
  sorry

end total_amount_for_gifts_l78_78094


namespace find_x_l78_78036

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l78_78036


namespace distinct_counts_eq_l78_78458

theorem distinct_counts_eq (boxes : List ℕ) :
  let trays := List.range (boxes.foldr Nat.max 0)
                |> List.foldl (λ acc i => acc ++ List.map (λ b => if b > i then b - i else 0) boxes) []
  boxes.erase_dup.length = trays.erase_dup.length :=
by
  sorry

end distinct_counts_eq_l78_78458


namespace range_f_real_l78_78214

noncomputable def f (a : ℝ) (x : ℝ) :=
  if x > 1 then (a ^ x) else (4 - a / 2) * x + 2

theorem range_f_real (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ (1 < a ∧ a ≤ 4) :=
by
  sorry

end range_f_real_l78_78214


namespace work_hours_to_pay_off_debt_l78_78673

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end work_hours_to_pay_off_debt_l78_78673


namespace isabella_houses_problem_l78_78525

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end isabella_houses_problem_l78_78525


namespace find_x_between_0_and_180_l78_78029

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l78_78029


namespace sqrt_sum_eq_pow_l78_78063

/-- 
For the value \( k = 3/2 \), the expression \( \sqrt{2016} + \sqrt{56} \) equals \( 14^k \)
-/
theorem sqrt_sum_eq_pow (k : ℝ) (h : k = 3 / 2) : 
  (Real.sqrt 2016 + Real.sqrt 56) = 14 ^ k := 
by 
  sorry

end sqrt_sum_eq_pow_l78_78063


namespace problem_statement_l78_78948

open Classical

variable (a_n : ℕ → ℝ) (a1 d : ℝ)

-- Condition: Arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ (n : ℕ), a_n (n + 1) = a1 + n * d 

-- Condition: Geometric relationship between a1, a3, and a9
def geometric_relation (a1 a3 a9 : ℝ) : Prop :=
  a3 / a1 = a9 / a3

-- Given conditions for the arithmetic sequence and geometric relation
axiom arith : arithmetic_sequence a_n a1 d
axiom geom : geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)

theorem problem_statement : d ≠ 0 → (∃ (a1 d : ℝ), d ≠ 0 ∧ arithmetic_sequence a_n a1 d ∧ geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)) → (a1 + 2 * d) / a1 = 3 := by
  sorry

end problem_statement_l78_78948


namespace fg_at_3_equals_97_l78_78515

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_at_3_equals_97 : f (g 3) = 97 := by
  sorry

end fg_at_3_equals_97_l78_78515


namespace total_dolls_l78_78182

-- Defining the given conditions as constants.
def big_boxes : Nat := 5
def small_boxes : Nat := 9
def dolls_per_big_box : Nat := 7
def dolls_per_small_box : Nat := 4

-- The main theorem we want to prove
theorem total_dolls : (big_boxes * dolls_per_big_box) + (small_boxes * dolls_per_small_box) = 71 :=
by
  rw [Nat.mul_add, Nat.mul_eq_mul, Nat.mul_eq_mul]
  exact sorry

end total_dolls_l78_78182


namespace find_x_l78_78038

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l78_78038


namespace cube_tetrahedron_volume_ratio_l78_78768

theorem cube_tetrahedron_volume_ratio :
  let s := 2
  let v1 := (0, 0, 0)
  let v2 := (2, 2, 0)
  let v3 := (2, 0, 2)
  let v4 := (0, 2, 2)
  let a := Real.sqrt 8 -- Side length of the tetrahedron
  let volume_tetra := (a^3 * Real.sqrt 2) / 12
  let volume_cube := s^3
  volume_cube / volume_tetra = 6 * Real.sqrt 2 := 
by
  -- Proof content skipped
  intros
  sorry

end cube_tetrahedron_volume_ratio_l78_78768


namespace find_abs_product_l78_78257

noncomputable def distinct_nonzero_real (a b c : ℝ) : Prop :=
a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem find_abs_product (a b c : ℝ) (h1 : distinct_nonzero_real a b c) 
(h2 : a + 1/(b^2) = b + 1/(c^2))
(h3 : b + 1/(c^2) = c + 1/(a^2)) :
  |a * b * c| = 1 :=
sorry

end find_abs_product_l78_78257


namespace PetyaColorsAll64Cells_l78_78261

-- Assuming a type for representing cell coordinates
structure Cell where
  row : ℕ
  col : ℕ

def isColored (c : Cell) : Prop := true  -- All cells are colored
def LShapedFigures : Set (Set Cell) := sorry  -- Define what constitutes an L-shaped figure

theorem PetyaColorsAll64Cells :
  (∀ tilesVector ∈ LShapedFigures, ¬∀ cell ∈ tilesVector, isColored cell) → (∀ c : Cell, c.row < 8 ∧ c.col < 8 ∧ isColored c) := sorry

end PetyaColorsAll64Cells_l78_78261


namespace part1_condition1_part1_condition2_part1_condition3_part2_l78_78071

theorem part1_condition1 (a c A C : ℝ) (h : c = sqrt(3) * a * sin C - c * cos A) :
  A = π / 3 := sorry

theorem part1_condition2 (A B C : ℝ) (h : sin^2 A - sin^2 B = sin^2 C - sin B * sin C) :
  A = π / 3 := sorry

theorem part1_condition3 (B C : ℝ) (h : tan B + tan C - sqrt(3) * tan B * tan C = -sqrt(3)) :
  let A := π - (B + C)
  A = π / 3 := sorry

theorem part2 (a : ℝ) (h1 : a = sqrt(3)) (h2 : let B := π - (C + A);
                              let C := π - (A + B);
                              0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) :
  let b := 2 * sin B;
  let c := 2 * sin C;
  let perimeter := a + b + c;
  3 + sqrt(3) < perimeter ∧ perimeter ≤ 3 * sqrt(3) := sorry

end part1_condition1_part1_condition2_part1_condition3_part2_l78_78071


namespace timothy_movies_count_l78_78739

variable (T : ℕ)

def timothy_movies_previous_year (T : ℕ) :=
  let timothy_2010 := T + 7
  let theresa_2010 := 2 * (T + 7)
  let theresa_previous := T / 2
  T + timothy_2010 + theresa_2010 + theresa_previous = 129

theorem timothy_movies_count (T : ℕ) (h : timothy_movies_previous_year T) : T = 24 := 
by 
  sorry

end timothy_movies_count_l78_78739


namespace find_x_tan_identity_l78_78049

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l78_78049


namespace regular_polygon_sides_l78_78372

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end regular_polygon_sides_l78_78372


namespace find_angle_C_find_a_b_l78_78099

noncomputable def angle_C (a b c : ℝ) (cosB cosC : ℝ) : Prop := 
(b - 2 * a) * cosC + c * cosB = 0

noncomputable def area_triangle (a b c S : ℝ) : Prop := 
S = (1/2) * a * b * Real.sin (Real.pi / 3)

noncomputable def solve_tri_angles (a b c : ℝ) : Prop := 
c = 2 ∧ S = Real.sqrt 3 → (a = 2 ∧ b = 2)

theorem find_angle_C {A B C a b c : ℝ} (h : angle_C a b c (Real.cos A) (Real.cos B)) : 
C = Real.pi / 3 := 
sorry

theorem find_a_b {a b c S : ℝ} (h1 : area_triangle a b c S) 
(h2 : solve_tri_angles a b c) :
a = 2 ∧ b = 2 := 
sorry

end find_angle_C_find_a_b_l78_78099


namespace B_can_complete_alone_l78_78600

-- Define the given conditions
def A_work_rate := 1 / 20
def total_days := 21
def A_quit_days := 15
def B_completion_days := 30

-- Define the problem statement in Lean
theorem B_can_complete_alone (x : ℝ) (h₁ : A_work_rate = 1 / 20) (h₂ : total_days = 21)
  (h₃ : A_quit_days = 15) (h₄ : (21 - A_quit_days) * (1 / 20 + 1 / x) + A_quit_days * (1 / x) = 1) :
  x = B_completion_days :=
  sorry

end B_can_complete_alone_l78_78600


namespace simplify_radical_expression_l78_78185

noncomputable def simpl_radical_form (q : ℝ) : ℝ :=
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3)

theorem simplify_radical_expression (q : ℝ) :
  simpl_radical_form q = 3 * q^3 * Real.sqrt 10 :=
by
  sorry

end simplify_radical_expression_l78_78185


namespace prime_bounds_l78_78355

noncomputable def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem prime_bounds (n : ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ k, 0 ≤ k → k ≤ Nat.sqrt (n / 3) → is_prime (k^2 + k + n)) : 
  ∀ k, 0 ≤ k → k ≤ n - 2 → is_prime (k^2 + k + n) :=
by
  sorry

end prime_bounds_l78_78355


namespace equation_solution_l78_78438

theorem equation_solution :
  ∃ x : ℝ, (3 * (x + 2) = x * (x + 2)) ↔ (x = -2 ∨ x = 3) :=
by
  sorry

end equation_solution_l78_78438


namespace domain_of_sqrt_fn_l78_78864

theorem domain_of_sqrt_fn : {x : ℝ | -2 ≤ x ∧ x ≤ 2} = {x : ℝ | 4 - x^2 ≥ 0} := 
by sorry

end domain_of_sqrt_fn_l78_78864


namespace seats_on_each_bus_l78_78279

-- Define the given conditions
def totalStudents : ℕ := 45
def totalBuses : ℕ := 5

-- Define what we need to prove - 
-- that the number of seats on each bus is 9
def seatsPerBus (students : ℕ) (buses : ℕ) : ℕ := students / buses

theorem seats_on_each_bus : seatsPerBus totalStudents totalBuses = 9 := by
  -- Proof to be filled in later
  sorry

end seats_on_each_bus_l78_78279


namespace max_a2_b2_c2_d2_l78_78256

-- Define the conditions for a, b, c, d
variables (a b c d : ℝ) 

-- Define the hypotheses from the problem
variables (h₁ : a + b = 17)
variables (h₂ : ab + c + d = 94)
variables (h₃ : ad + bc = 195)
variables (h₄ : cd = 120)

-- Define the final statement to be proved
theorem max_a2_b2_c2_d2 : ∃ (a b c d : ℝ), a + b = 17 ∧ ab + c + d = 94 ∧ ad + bc = 195 ∧ cd = 120 ∧ (a^2 + b^2 + c^2 + d^2) = 918 :=
by sorry

end max_a2_b2_c2_d2_l78_78256


namespace donut_combinations_l78_78932

-- Define the problem statement where Bill needs to purchase 10 donuts,
-- with at least one of each of the 5 kinds, and calculate the combinations.

def count_donut_combinations : ℕ :=
  Nat.choose 9 4

theorem donut_combinations :
  count_donut_combinations = 126 :=
by
  -- Proof can be filled in here
  sorry

end donut_combinations_l78_78932


namespace volume_ratio_of_rotated_solids_l78_78276

theorem volume_ratio_of_rotated_solids (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := π * b^2 * a
  let V2 := π * a^2 * b
  V1 / V2 = b / a :=
by
  intros
  -- Proof omitted
  sorry

end volume_ratio_of_rotated_solids_l78_78276


namespace primary_school_capacity_l78_78812

variable (x : ℝ)

/-- In a town, there are four primary schools. Two of them can teach 400 students at a time, 
and the other two can teach a certain number of students at a time. These four primary schools 
can teach a total of 1480 students at a time. -/
theorem primary_school_capacity 
  (h1 : 2 * 400 + 2 * x = 1480) : 
  x = 340 :=
sorry

end primary_school_capacity_l78_78812


namespace negation_of_universal_proposition_l78_78904

theorem negation_of_universal_proposition :
  ¬ (∀ (m : ℝ), ∃ (x : ℝ), x^2 + x + m = 0) ↔ ∃ (m : ℝ), ¬ ∃ (x : ℝ), x^2 + x + m = 0 :=
by sorry

end negation_of_universal_proposition_l78_78904


namespace sugar_already_put_in_l78_78117

-- Definitions based on conditions
def required_sugar : ℕ := 13
def additional_sugar_needed : ℕ := 11

-- Theorem to be proven
theorem sugar_already_put_in :
  required_sugar - additional_sugar_needed = 2 := by
  sorry

end sugar_already_put_in_l78_78117


namespace increasing_function_l78_78866

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) (h : a ≥ 1) : 
  ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
by 
  sorry

end increasing_function_l78_78866


namespace probability_of_condition1_before_condition2_l78_78915

-- Definitions for conditions
def condition1 (draw_counts : List ℕ) : Prop :=
  ∃ count ∈ draw_counts, count ≥ 3

def condition2 (draw_counts : List ℕ) : Prop :=
  ∀ count ∈ draw_counts, count ≥ 1

-- Probability function
def probability_condition1_before_condition2 : ℚ :=
  13 / 27

-- The proof statement
theorem probability_of_condition1_before_condition2 :
  (∃ draw_counts : List ℕ, (condition1 draw_counts) ∧  ¬(condition2 draw_counts)) →
  probability_condition1_before_condition2 = 13 / 27 :=
sorry

end probability_of_condition1_before_condition2_l78_78915


namespace zion_dad_age_difference_in_10_years_l78_78452

/-
Given:
1. Zion's age is 8 years.
2. Zion's dad's age is 3 more than 4 times Zion's age.
Prove:
In 10 years, the difference in age between Zion's dad and Zion will be 27 years.
-/

theorem zion_dad_age_difference_in_10_years :
  let zion_age := 8
  let dad_age := 4 * zion_age + 3
  (dad_age + 10) - (zion_age + 10) = 27 := by
  sorry

end zion_dad_age_difference_in_10_years_l78_78452


namespace num_rectangles_in_5x5_grid_l78_78990

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l78_78990


namespace inequality_example_l78_78110

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (der : ∀ x, deriv f x = f' x)

theorem inequality_example (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023)
:= sorry

end inequality_example_l78_78110


namespace lines_intersect_l78_78469

-- Condition definitions
def line1 (t : ℝ) : ℝ × ℝ :=
  ⟨2 + t * -1, 3 + t * 5⟩

def line2 (u : ℝ) : ℝ × ℝ :=
  ⟨u * -1, 7 + u * 4⟩

-- Theorem statement
theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (6, -17) :=
by
  sorry

end lines_intersect_l78_78469


namespace boat_distance_downstream_l78_78599

theorem boat_distance_downstream 
    (boat_speed_still : ℝ) 
    (stream_speed : ℝ) 
    (time_downstream : ℝ) 
    (distance_downstream : ℝ) 
    (h_boat_speed_still : boat_speed_still = 13) 
    (h_stream_speed : stream_speed = 6) 
    (h_time_downstream : time_downstream = 3.6315789473684212) 
    (h_distance_downstream : distance_downstream = 19 * 3.6315789473684212): 
    distance_downstream = 69 := 
by 
  have h_effective_speed : boat_speed_still + stream_speed = 19 := by 
    rw [h_boat_speed_still, h_stream_speed]; norm_num 
  rw [h_distance_downstream]; norm_num 
  sorry

end boat_distance_downstream_l78_78599


namespace pump_capacity_l78_78743

-- Define parameters and assumptions
def tank_volume : ℝ := 1000
def fill_percentage : ℝ := 0.85
def fill_time : ℝ := 1
def num_pumps : ℝ := 8
def pump_efficiency : ℝ := 0.75
def required_fill_volume : ℝ := fill_percentage * tank_volume

-- Assumed total effective capacity must meet the required fill volume
theorem pump_capacity (C : ℝ) : 
  (num_pumps * pump_efficiency * C = required_fill_volume) → 
  C = 850.0 / 6.0 :=
by
  sorry

end pump_capacity_l78_78743


namespace find_x_tan_identity_l78_78047

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l78_78047


namespace find_n_18_l78_78491

def valid_denominations (n : ℕ) : Prop :=
  ∀ k < 106, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c

def cannot_form_106 (n : ℕ) : Prop :=
  ¬ ∃ a b c : ℕ, 106 = 7 * a + n * b + (n + 1) * c

theorem find_n_18 : 
  ∃ n : ℕ, valid_denominations n ∧ cannot_form_106 n ∧ ∀ m < n, ¬ (valid_denominations m ∧ cannot_form_106 m) :=
sorry

end find_n_18_l78_78491


namespace gcd_40_120_80_l78_78449

-- Given numbers
def n1 := 40
def n2 := 120
def n3 := 80

-- The problem we want to prove:
theorem gcd_40_120_80 : Int.gcd (Int.gcd n1 n2) n3 = 40 := by
  sorry

end gcd_40_120_80_l78_78449


namespace cost_per_use_correct_l78_78530

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end cost_per_use_correct_l78_78530


namespace S_63_value_l78_78655

noncomputable def b (n : ℕ) : ℚ := (3 + (-1)^(n-1))/2

noncomputable def a : ℕ → ℚ
| 0       => 0
| 1       => 2
| (n+2)   => if (n % 2 = 0) then - (a (n+1))/2 else 2 - 2*(a (n+1))

noncomputable def S : ℕ → ℚ
| 0       => 0
| (n+1)   => S n + a (n+1)

theorem S_63_value : S 63 = 464 := by
  sorry

end S_63_value_l78_78655


namespace find_x_between_0_and_180_l78_78032

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l78_78032


namespace count_distribution_methods_l78_78937

-- Defining the conditions
def numStudents : ℕ := 5
def numUniversities : ℕ := 3
def universities := {Peking, ShanghaiJiaoTong, Tsinghua : String}

-- Define the distribution function
def distributionMethods (students universities : ℕ) :=
  if students <= 0 ∨ universities <= 0 then 0 else
    let distribute_group_221 := (Nat.C (students) 2) * (Nat.C (students - 2) 2) * (Nat.fact universities)
    let distribute_group_311 := (Nat.C (students) 3) * (Nat.C (students - 3) 1) * (Nat.fact universities)
    distribute_group_221 / 2 + distribute_group_311 / 2

-- The main theorem
theorem count_distribution_methods : distributionMethods numStudents numUniversities = 150 := by
  sorry

end count_distribution_methods_l78_78937


namespace num_rectangles_grid_l78_78964

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l78_78964


namespace algebraic_expression_standard_l78_78584

theorem algebraic_expression_standard :
  (∃ (expr : String), expr = "-(1/3)m" ∧
    expr ≠ "1(2/5)a" ∧
    expr ≠ "m / n" ∧
    expr ≠ "t × 3") :=
  sorry

end algebraic_expression_standard_l78_78584


namespace credibility_of_relationship_l78_78583

theorem credibility_of_relationship
  (sample_size : ℕ)
  (chi_squared_value : ℝ)
  (table : ℕ → ℝ × ℝ)
  (h_sample : sample_size = 5000)
  (h_chi_squared : chi_squared_value = 6.109)
  (h_table : table 5 = (5.024, 0.025) ∧ table 6 = (6.635, 0.010)) :
  credible_percent = 97.5 :=
by
  sorry

end credibility_of_relationship_l78_78583


namespace tan_sin_cos_eq_l78_78023

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l78_78023


namespace james_distance_ridden_l78_78102

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end james_distance_ridden_l78_78102


namespace problem1_problem2_l78_78500

-- Define points A, B, C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := -2}
def B : Point := {x := 2, y := 1}
def C : Point := {x := 3, y := 2}

-- Function to compute vector difference
def vector_sub (p1 p2 : Point) : Point :=
  {x := p1.x - p2.x, y := p1.y - p2.y}

-- Function to compute vector scalar multiplication
def scalar_mul (k : ℝ) (p : Point) : Point :=
  {x := k * p.x, y := k * p.y}

-- Function to add two vectors
def vec_add (p1 p2 : Point) : Point :=
  {x := p1.x + p2.x, y := p1.y + p2.y}

-- Problem 1
def result_vector : Point :=
  let AB := vector_sub B A
  let AC := vector_sub C A
  let BC := vector_sub C B
  vec_add (scalar_mul 3 AB) (vec_add (scalar_mul (-2) AC) BC)

-- Prove the coordinates are (0, 2)
theorem problem1 : result_vector = {x := 0, y := 2} := by
  sorry

-- Problem 2
def D : Point :=
  let BC := vector_sub C B
  {x := 1 + BC.x, y := (-2) + BC.y}

-- Prove the coordinates are (2, -1)
theorem problem2 : D = {x := 2, y := -1} := by
  sorry

end problem1_problem2_l78_78500


namespace solution_set_ineq_l78_78350

theorem solution_set_ineq (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ (x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1)) :=
sorry

end solution_set_ineq_l78_78350


namespace intersection_A_B_l78_78782

open Set

-- Define sets A and B with given conditions
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

-- Prove the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {0, 3} := 
by
  sorry

end intersection_A_B_l78_78782


namespace quadratic_roots_real_equal_l78_78945

theorem quadratic_roots_real_equal (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ a = 3 ∧ b = 2 - m ∧ c = 6 ∧
    (b^2 - 4 * a * c = 0)) ↔ (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_real_equal_l78_78945


namespace ellipse_m_range_l78_78710

theorem ellipse_m_range (m : ℝ) 
  (h1 : m + 9 > 25 - m) 
  (h2 : 25 - m > 0) 
  (h3 : m + 9 > 0) : 
  8 < m ∧ m < 25 := 
by
  sorry

end ellipse_m_range_l78_78710


namespace picnic_adults_children_difference_l78_78926

theorem picnic_adults_children_difference :
  ∃ (M W A C : ℕ),
    (M = 65) ∧
    (M = W + 20) ∧
    (A = M + W) ∧
    (C = 200 - A) ∧
    ((A - C) = 20) :=
by
  sorry

end picnic_adults_children_difference_l78_78926


namespace find_x_l78_78018

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l78_78018


namespace total_daily_cost_correct_l78_78240

/-- Definition of the daily wages of each type of worker -/
def daily_wage_worker : ℕ := 100
def daily_wage_electrician : ℕ := 2 * daily_wage_worker
def daily_wage_plumber : ℕ := (5 * daily_wage_worker) / 2 -- 2.5 times daily_wage_worker
def daily_wage_architect : ℕ := 7 * daily_wage_worker / 2 -- 3.5 times daily_wage_worker

/-- Definition of the total daily cost for one project -/
def daily_cost_one_project : ℕ :=
  2 * daily_wage_worker +
  daily_wage_electrician +
  daily_wage_plumber +
  daily_wage_architect

/-- Definition of the total daily cost for three projects -/
def total_daily_cost_three_projects : ℕ :=
  3 * daily_cost_one_project

/-- Theorem stating the overall labor costs for one day for all three projects -/
theorem total_daily_cost_correct :
  total_daily_cost_three_projects = 3000 :=
by
  -- Proof omitted
  sorry

end total_daily_cost_correct_l78_78240


namespace find_x_value_l78_78007

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l78_78007


namespace hyperbola_focal_length_l78_78648

theorem hyperbola_focal_length (m : ℝ) (h_eq : m * x^2 + 2 * y^2 = 2) (h_imag_axis : -2 / m = 4) : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 := 
sorry

end hyperbola_focal_length_l78_78648


namespace solve_tan_equation_l78_78046

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l78_78046


namespace ivan_chess_false_l78_78593

theorem ivan_chess_false (n : ℕ) :
  ∃ n, n + 3 * n + 6 * n = 64 → False :=
by
  use 6
  sorry

end ivan_chess_false_l78_78593


namespace find_x_value_l78_78008

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l78_78008


namespace minerals_found_today_l78_78683

noncomputable def yesterday_gemstones := 21
noncomputable def today_minerals := 48
noncomputable def today_gemstones := 21

theorem minerals_found_today :
  (today_minerals - (2 * yesterday_gemstones) = 6) :=
by
  sorry

end minerals_found_today_l78_78683


namespace base_10_to_base_7_conversion_l78_78573

theorem base_10_to_base_7_conversion :
  ∃ (digits : ℕ → ℕ), 789 = digits 3 * 7^3 + digits 2 * 7^2 + digits 1 * 7^1 + digits 0 * 7^0 ∧
  digits 3 = 2 ∧ digits 2 = 2 ∧ digits 1 = 0 ∧ digits 0 = 5 :=
sorry

end base_10_to_base_7_conversion_l78_78573


namespace circle_equation_center_xaxis_radius_2_l78_78135

theorem circle_equation_center_xaxis_radius_2 (a x y : ℝ) :
  (0:ℝ) < 2 ∧ (a - 1)^2 + 2^2 = 4 -> (x - 1)^2 + y^2 = 4 :=
by
  sorry

end circle_equation_center_xaxis_radius_2_l78_78135


namespace rectangle_count_5x5_l78_78965

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l78_78965


namespace six_digit_number_divisible_by_37_l78_78741

theorem six_digit_number_divisible_by_37 (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) (h3 : 37 ∣ (a + b)) : 37 ∣ (1000 * a + b) :=
sorry

end six_digit_number_divisible_by_37_l78_78741


namespace simplify_fraction_l78_78223

noncomputable def simplified_expression (x y : ℝ) : ℝ :=
  (x^2 - (4 / y)) / (y^2 - (4 / x))

theorem simplify_fraction {x y : ℝ} (h : x * y ≠ 4) :
  simplified_expression x y = x / y := 
by 
  sorry

end simplify_fraction_l78_78223


namespace tan_sin_cos_eq_l78_78053

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l78_78053


namespace min_staff_members_l78_78236

theorem min_staff_members
  (num_male_students : ℕ)
  (num_benches_3_students : ℕ)
  (num_benches_4_students : ℕ)
  (num_female_students : ℕ)
  (total_students : ℕ)
  (total_seating_capacity : ℕ)
  (additional_seats_required : ℕ)
  (num_staff_members : ℕ)
  (h1 : num_female_students = 4 * num_male_students)
  (h2 : num_male_students = 29)
  (h3 : num_benches_3_students = 15)
  (h4 : num_benches_4_students = 14)
  (h5 : total_seating_capacity = 3 * num_benches_3_students + 4 * num_benches_4_students)
  (h6 : total_students = num_male_students + num_female_students)
  (h7 : additional_seats_required = total_students - total_seating_capacity)
  (h8 : num_staff_members = additional_seats_required)
  : num_staff_members = 44 := 
sorry

end min_staff_members_l78_78236


namespace solution_set_inequality_l78_78666

theorem solution_set_inequality (a m : ℝ) (h : ∀ x : ℝ, (x > m ∧ x < 1) ↔ 2 * x^2 - 3 * x + a < 0) : m = 1 / 2 :=
by
  -- Insert the proof here
  sorry

end solution_set_inequality_l78_78666


namespace card_2_in_box_Q_l78_78001

theorem card_2_in_box_Q (P Q : Finset ℕ) (hP : P.card = 3) (hQ : Q.card = 5) 
  (hdisjoint : Disjoint P Q) (huniv : P ∪ Q = (Finset.range 9).erase 0)
  (hsum_eq : P.sum id = Q.sum id) :
  2 ∈ Q := 
sorry

end card_2_in_box_Q_l78_78001


namespace winnie_balloons_remainder_l78_78587

theorem winnie_balloons_remainder :
  let red_balloons := 20
  let white_balloons := 40
  let green_balloons := 70
  let chartreuse_balloons := 90
  let violet_balloons := 15
  let friends := 10
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons + violet_balloons
  total_balloons % friends = 5 :=
by
  sorry

end winnie_balloons_remainder_l78_78587


namespace bread_slices_per_friend_l78_78470

theorem bread_slices_per_friend :
  (∀ (slices_per_loaf friends loaves total_slices_per_friend : ℕ),
    slices_per_loaf = 15 →
    friends = 10 →
    loaves = 4 →
    total_slices_per_friend = slices_per_loaf * loaves / friends →
    total_slices_per_friend = 6) :=
by 
  intros slices_per_loaf friends loaves total_slices_per_friend h1 h2 h3 h4
  sorry

end bread_slices_per_friend_l78_78470


namespace intersection_A_B_l78_78079

open Set

def A : Set ℝ := {x | x ^ 2 - x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1, 2} :=
sorry

end intersection_A_B_l78_78079


namespace ab_operation_l78_78718

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l78_78718


namespace binomial_expansion_sum_l78_78506

theorem binomial_expansion_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h₁ : (a * x - 1)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5)
  (h₂ : a₃ = 80) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
sorry

end binomial_expansion_sum_l78_78506


namespace arithmetic_mean_of_two_digit_multiples_of_9_l78_78147

theorem arithmetic_mean_of_two_digit_multiples_of_9 : 
  let a1 := 18 in
  let an := 99 in
  let d := 9 in
  let n := (an - a1) / d + 1 in
  let S := n * (a1 + an) / 2 in
  (S / n : ℝ) = 58.5 :=
by
  let a1 := 18
  let an := 99
  let d := 9
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  show (S / n : ℝ) = 58.5
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l78_78147


namespace gcd_lcm_sum_18_30_45_l78_78396

theorem gcd_lcm_sum_18_30_45 :
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  A + B = 93 :=
by
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  have hA : A = 3 := by sorry -- Proof of GCD computation
  have hB : B = 90 := by sorry -- Proof of LCM computation
  rw [hA, hB]
  norm_num

end gcd_lcm_sum_18_30_45_l78_78396


namespace city_schools_count_l78_78496

theorem city_schools_count (a b c : ℕ) (schools : ℕ) : 
  b = 40 → c = 51 → b < a → a < c → 
  (a > b ∧ a < c ∧ (a - 1) * 3 < (c - b + 1) * 3 + 1) → 
  schools = (c - 1) / 3 :=
by
  sorry

end city_schools_count_l78_78496


namespace find_x_value_l78_78009

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l78_78009


namespace average_production_l78_78062

theorem average_production (n : ℕ) (P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 62) : n = 14 :=
  sorry

end average_production_l78_78062


namespace time_with_walkway_l78_78473

-- Definitions
def length_walkway : ℝ := 60
def time_against_walkway : ℝ := 120
def time_stationary_walkway : ℝ := 48

-- Theorem statement
theorem time_with_walkway (v w : ℝ)
  (h1 : 60 = 120 * (v - w))
  (h2 : 60 = 48 * v)
  (h3 : v = 1.25)
  (h4 : w = 0.75) :
  60 = 30 * (v + w) :=
by
  sorry

end time_with_walkway_l78_78473


namespace find_y_value_l78_78911

theorem find_y_value : (12 ^ 2 * 6 ^ 4) / 432 = 432 := by
  sorry

end find_y_value_l78_78911


namespace ab_operation_l78_78717

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l78_78717


namespace digit_2567_l78_78687

def nth_digit_in_concatenation (n : ℕ) : ℕ :=
  sorry

theorem digit_2567 : nth_digit_in_concatenation 2567 = 8 :=
by
  sorry

end digit_2567_l78_78687


namespace tan_sin_cos_eq_l78_78055

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l78_78055


namespace y_intercept_l78_78328

theorem y_intercept (x y : ℝ) (h : 4 * x + 7 * y = 28) : x = 0 → y = 4 :=
by
  intro hx
  rw [hx, zero_mul, add_zero] at h
  have := eq_div_of_mul_eq (by norm_num : 7 ≠ 0) h
  rw [eq_comm, div_eq_iff (by norm_num : 7 ≠ 0), mul_comm] at this
  exact this

end y_intercept_l78_78328


namespace find_a_l78_78564

-- Define the conditions of the problem
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a + 3, 1, -3) -- Coefficients of line1: (a+3)x + y - 3 = 0
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (5, a - 3, 4)  -- Coefficients of line2: 5x + (a-3)y + 4 = 0

-- Definition of direction vector and normal vector
def direction_vector (a : ℝ) : ℝ × ℝ := (1, -(a + 3))
def normal_vector (a : ℝ) : ℝ × ℝ := (5, a - 3)

-- Proof statement
theorem find_a (a : ℝ) : (direction_vector a = normal_vector a) → a = -2 :=
by {
  -- Insert proof here
  sorry
}

end find_a_l78_78564


namespace raised_bed_section_area_l78_78315

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area_l78_78315


namespace smallest_n_with_divisors_2020_l78_78333

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l78_78333


namespace find_m_l78_78354

variable (a : ℝ) (m : ℝ)

theorem find_m (h : a^(m + 1) * a^(2 * m - 1) = a^9) : m = 3 := 
by
  sorry

end find_m_l78_78354


namespace sin_780_eq_sqrt3_div_2_l78_78765

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l78_78765


namespace smallest_number_has_2020_divisors_l78_78338

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l78_78338


namespace inscribed_circle_radius_l78_78230

theorem inscribed_circle_radius (A p r s : ℝ) (h₁ : A = 2 * p) (h₂ : p = 2 * s) (h₃ : A = r * s) : r = 4 :=
by sorry

end inscribed_circle_radius_l78_78230


namespace a_mul_b_value_l78_78720

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l78_78720


namespace find_2nd_month_sales_l78_78757

def sales_of_1st_month : ℝ := 2500
def sales_of_3rd_month : ℝ := 9855
def sales_of_4th_month : ℝ := 7230
def sales_of_5th_month : ℝ := 7000
def sales_of_6th_month : ℝ := 11915
def average_sales : ℝ := 7500
def months : ℕ := 6
def total_required_sales : ℝ := average_sales * months
def total_known_sales : ℝ := sales_of_1st_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month

theorem find_2nd_month_sales : 
  ∃ (sales_of_2nd_month : ℝ), total_required_sales = sales_of_1st_month + sales_of_2nd_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month ∧ sales_of_2nd_month = 10500 := by
  sorry

end find_2nd_month_sales_l78_78757


namespace train_speed_l78_78745

theorem train_speed (distance time : ℕ) (h1 : distance = 180) (h2 : time = 9) : distance / time = 20 := by
  sorry

end train_speed_l78_78745


namespace merchant_loss_l78_78606

theorem merchant_loss (n m : ℝ) (h₁ : n ≠ m) : 
  let x := n / m
  let y := m / n
  x + y > 2 := by
sorry

end merchant_loss_l78_78606


namespace son_age_is_18_l78_78758

theorem son_age_is_18
  (S F : ℕ)
  (h1 : F = S + 20)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 18 :=
by sorry

end son_age_is_18_l78_78758


namespace lcm_division_l78_78686

open Nat

-- Define the LCM function for a list of integers
def list_lcm (l : List Nat) : Nat := l.foldr (fun a b => Nat.lcm a b) 1

-- Define the sequence ranges
def range1 := List.range' 20 21 -- From 20 to 40 inclusive
def range2 := List.range' 41 10 -- From 41 to 50 inclusive

-- Define P and Q
def P : Nat := list_lcm range1
def Q : Nat := Nat.lcm P (list_lcm range2)

-- The theorem statement
theorem lcm_division : (Q / P) = 55541 := by
  sorry

end lcm_division_l78_78686


namespace douglas_won_in_Y_l78_78231

theorem douglas_won_in_Y (percent_total_vote : ℕ) (percent_vote_X : ℕ) (ratio_XY : ℕ) (P : ℕ) :
  percent_total_vote = 54 →
  percent_vote_X = 62 →
  ratio_XY = 2 →
  P = 38 :=
by
  sorry

end douglas_won_in_Y_l78_78231


namespace find_abscissas_l78_78268

theorem find_abscissas (x_A x_B : ℝ) (y_A y_B : ℝ) : 
  ((y_A = x_A^2) ∧ (y_B = x_B^2) ∧ (0, 15) = (0,  (5 * y_B + 3 * y_A) / 8) ∧ (5 * x_B + 3 * x_A = 0)) → 
  ((x_A = -5 ∧ x_B = 3) ∨ (x_A = 5 ∧ x_B = -3)) :=
by
  sorry

end find_abscissas_l78_78268


namespace range_x_minus_q_l78_78086

theorem range_x_minus_q (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) : x - q < 3 - 2*q :=
by
  sorry

end range_x_minus_q_l78_78086


namespace peter_vacation_saving_l78_78409

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end peter_vacation_saving_l78_78409


namespace smallest_n_with_divisors_2020_l78_78332

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l78_78332


namespace initial_overs_l78_78235

theorem initial_overs {x : ℝ} (h1 : 4.2 * x + (83 / 15) * 30 = 250) : x = 20 :=
by
  sorry

end initial_overs_l78_78235


namespace beavers_swimming_correct_l78_78749

variable (initial_beavers remaining_beavers beavers_swimming : ℕ)

def beavers_problem : Prop :=
  initial_beavers = 2 ∧
  remaining_beavers = 1 ∧
  beavers_swimming = initial_beavers - remaining_beavers

theorem beavers_swimming_correct :
  beavers_problem initial_beavers remaining_beavers beavers_swimming → beavers_swimming = 1 :=
by
  sorry

end beavers_swimming_correct_l78_78749


namespace marcy_sip_amount_l78_78116

theorem marcy_sip_amount (liters : ℕ) (ml_per_liter : ℕ) (total_minutes : ℕ) (interval_minutes : ℕ) (total_ml : ℕ) (total_sips : ℕ) (ml_per_sip : ℕ) 
  (h1 : liters = 2) 
  (h2 : ml_per_liter = 1000)
  (h3 : total_minutes = 250) 
  (h4 : interval_minutes = 5)
  (h5 : total_ml = liters * ml_per_liter)
  (h6 : total_sips = total_minutes / interval_minutes)
  (h7 : ml_per_sip = total_ml / total_sips) : 
  ml_per_sip = 40 := 
by
  sorry

end marcy_sip_amount_l78_78116


namespace eval_polynomial_at_4_using_horners_method_l78_78490

noncomputable def polynomial : (x : ℝ) → ℝ :=
  λ x => 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem eval_polynomial_at_4_using_horners_method :
  polynomial 4 = 2845.3 :=
by
  sorry

end eval_polynomial_at_4_using_horners_method_l78_78490


namespace tan_sin_cos_eq_l78_78025

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l78_78025


namespace rectangle_count_5x5_l78_78966

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l78_78966


namespace transformed_circle_eq_l78_78796

theorem transformed_circle_eq (x y : ℝ) (h : x^2 + y^2 = 1) : x^2 + 9 * (y / 3)^2 = 1 := by
  sorry

end transformed_circle_eq_l78_78796


namespace lacrosse_more_than_football_l78_78193

-- Define the constants and conditions
def total_bottles := 254
def football_players := 11
def bottles_per_football_player := 6
def soccer_bottles := 53
def rugby_bottles := 49

-- Calculate the number of bottles needed by each team
def football_bottles := football_players * bottles_per_football_player
def other_teams_bottles := football_bottles + soccer_bottles + rugby_bottles
def lacrosse_bottles := total_bottles - other_teams_bottles

-- The theorem to be proven
theorem lacrosse_more_than_football : lacrosse_bottles - football_bottles = 20 :=
by
  sorry

end lacrosse_more_than_football_l78_78193


namespace sam_wins_probability_l78_78841

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l78_78841


namespace inequality_holds_equality_condition_l78_78827

theorem inequality_holds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) ≥ 1 / 2 :=
sorry

theorem equality_condition (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) = 1 / 2 ↔ 
  ((a = 0 ∧ b = 0 ∧ 0 < c) ∨ (a = 0 ∧ c = 0 ∧ 0 < b) ∨ (b = 0 ∧ c = 0 ∧ 0 < a)) :=
sorry

end inequality_holds_equality_condition_l78_78827


namespace hyperbola_vertex_distance_l78_78329

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

-- Statement: The distance between the vertices of the hyperbola is 1
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 2 * (1 / 2) = 1 :=
by
  intros x y H
  sorry

end hyperbola_vertex_distance_l78_78329


namespace solve_system_of_equations_simplify_expression_l78_78486

-- Statement for system of equations
theorem solve_system_of_equations (s t : ℚ) 
  (h1 : 2 * s + 3 * t = 2) 
  (h2 : 2 * s - 6 * t = -1) :
  s = 1 / 2 ∧ t = 1 / 3 :=
sorry

-- Statement for simplifying the expression
theorem simplify_expression (x y : ℚ) :
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y :=
sorry

end solve_system_of_equations_simplify_expression_l78_78486


namespace find_x_tan_identity_l78_78048

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l78_78048


namespace find_multiplier_n_l78_78661

variable (x y n : ℝ)

theorem find_multiplier_n (h1 : 5 * x = n * y) 
  (h2 : x * y ≠ 0) 
  (h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998) : 
  n = 6 := 
by
  sorry

end find_multiplier_n_l78_78661


namespace part1_part2_l78_78950

theorem part1 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n) : n ∣ m := 
sorry

theorem part2 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n)
(h3 : m - n = 10) : (m, n) = (11, 1) ∨ (m, n) = (12, 2) ∨ (m, n) = (15, 5) ∨ (m, n) = (20, 10) := 
sorry

end part1_part2_l78_78950


namespace average_weight_of_Arun_l78_78455

theorem average_weight_of_Arun :
  ∃ avg_weight : Real,
    (avg_weight = (65 + 68) / 2) ∧
    ∀ w : Real, (65 < w ∧ w < 72) ∧ (60 < w ∧ w < 70) ∧ (w ≤ 68) → avg_weight = 66.5 :=
by
  -- we will fill the details of the proof here
  sorry

end average_weight_of_Arun_l78_78455


namespace quadrilateral_interior_angle_not_greater_90_l78_78158

-- Definition of the quadrilateral interior angle property
def quadrilateral_interior_angles := ∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 → b > 90 → c > 90 → d > 90 → false)

-- Proposition: There is at least one interior angle in a quadrilateral that is not greater than 90 degrees.
theorem quadrilateral_interior_angle_not_greater_90 :
  (∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90) → false) →
  (∃ (a b c d : ℝ), a + b + c + d = 360 ∧ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) :=
sorry

end quadrilateral_interior_angle_not_greater_90_l78_78158


namespace systematic_sampling_count_l78_78178

theorem systematic_sampling_count :
  ∀ (total_people selected_people initial_draw : ℕ),
  total_people = 960 →
  selected_people = 32 →
  initial_draw = 9 →
  let common_difference := total_people / selected_people in
  let general_term (n : ℕ) := initial_draw + common_difference * (n - 1) in
  let count_in_interval := (25 - 16 + 1) in
  count_in_interval = 10 :=
begin
  intros total_people selected_people initial_draw h_total h_selected h_initial,
  let common_difference := total_people / selected_people,
  let general_term (n : ℕ) := initial_draw + common_difference * (n - 1),
  let count_in_interval := (25 - 16 + 1),
  have : common_difference = 30, by { rw [h_total, h_selected], exact nat.div_self (nat.pos_of_ne_zero (by norm_num)) },
  have : general_term 1 = initial_draw, by refl,
  have : ∀ n, general_term n = 30 * n - 29, by {
    intro n,
    simp only [general_term, common_difference],
    rw [nat.mul_sub_left_distrib, nat.sub_self, mul_one, nat.add_sub_left (le_of_lt (by norm_num : 0 < 9))],
    rw [mul_comm] },
  exact sorry
end

end systematic_sampling_count_l78_78178


namespace rectangles_in_grid_l78_78985

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l78_78985


namespace num_rectangles_grid_l78_78960

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l78_78960


namespace eq_condition_l78_78478

theorem eq_condition (a : ℝ) :
  (∃ x : ℝ, a * (4 * |x| + 1) = 4 * |x|) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end eq_condition_l78_78478


namespace fish_upstream_speed_l78_78467

def Vs : ℝ := 45
def Vdownstream : ℝ := 55

def Vupstream (Vs Vw : ℝ) : ℝ := Vs - Vw
def Vstream (Vs Vdownstream : ℝ) : ℝ := Vdownstream - Vs

theorem fish_upstream_speed :
  Vupstream Vs (Vstream Vs Vdownstream) = 35 := by
  sorry

end fish_upstream_speed_l78_78467


namespace value_of_a_star_b_l78_78725

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l78_78725


namespace probabilities_equal_l78_78466

def roll := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def is_successful (r : roll) : Prop := r.val ≥ 3

def prob_successful : ℚ := 4 / 6

def prob_unsuccessful : ℚ := 1 - prob_successful

def prob_at_least_one_success_two_rolls : ℚ := 1 - (prob_unsuccessful ^ 2)

def prob_at_least_two_success_four_rolls : ℚ :=
  let zero_success := prob_unsuccessful ^ 4
  let one_success := 4 * (prob_unsuccessful ^ 3) * prob_successful
  1 - (zero_success + one_success)

theorem probabilities_equal :
  prob_at_least_one_success_two_rolls = prob_at_least_two_success_four_rolls := by
  sorry

end probabilities_equal_l78_78466


namespace james_drive_time_to_canada_l78_78239

theorem james_drive_time_to_canada : 
  ∀ (distance speed stop_time : ℕ), 
    speed = 60 → 
    distance = 360 → 
    stop_time = 1 → 
    (distance / speed) + stop_time = 7 :=
by
  intros distance speed stop_time h1 h2 h3
  sorry

end james_drive_time_to_canada_l78_78239


namespace people_in_each_van_l78_78138

theorem people_in_each_van
  (cars : ℕ) (taxis : ℕ) (vans : ℕ)
  (people_per_car : ℕ) (people_per_taxi : ℕ) (total_people : ℕ) 
  (people_per_van : ℕ) :
  cars = 3 → taxis = 6 → vans = 2 →
  people_per_car = 4 → people_per_taxi = 6 → total_people = 58 →
  3 * people_per_car + 6 * people_per_taxi + 2 * people_per_van = total_people →
  people_per_van = 5 :=
by sorry

end people_in_each_van_l78_78138


namespace infinitely_many_odd_n_composite_l78_78122

theorem infinitely_many_odd_n_composite (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (n : ℕ) (h_odd : n % 2 = 1), 
     ∀ k : ℕ, ∃ (m : ℕ) (h_odd_m : m % 2 = 1), 
     (∃ (d : ℕ), d ∣ (2^m + m) ∧ (1 < d ∧ d < 2^m + m))
:=
sorry

end infinitely_many_odd_n_composite_l78_78122


namespace right_triangle_cos_pq_l78_78228

theorem right_triangle_cos_pq (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : c = 13) (h2 : b / c = 5/13) : a = 12 :=
by
  sorry

end right_triangle_cos_pq_l78_78228


namespace sum_of_x_and_y_l78_78226

theorem sum_of_x_and_y (x y : ℕ) (h_pos_x: 0 < x) (h_pos_y: 0 < y) (h_gt: x > y) (h_eq: x + x * y = 391) : x + y = 39 :=
by
  sorry

end sum_of_x_and_y_l78_78226


namespace min_area_after_fold_l78_78311

theorem min_area_after_fold (A : ℝ) (h_A : A = 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) : 
  ∃ (m : ℝ), m = min_area ∧ m = 2 / 3 :=
by
  sorry

end min_area_after_fold_l78_78311


namespace find_x_between_0_and_180_l78_78030

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l78_78030


namespace compute_expression_l78_78488

theorem compute_expression :
  45 * 72 + 28 * 45 = 4500 :=
  sorry

end compute_expression_l78_78488


namespace root_division_7_pow_l78_78002

theorem root_division_7_pow : 
  ( (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = 7 ^ (3 / 28) ) :=
sorry

end root_division_7_pow_l78_78002


namespace solution_correctness_l78_78497

noncomputable def solution_set : Set ℝ := {x | x + 60 / (x - 5) = -12}

theorem solution_correctness : solution_set = {0, -7} := 
begin
  sorry
end

end solution_correctness_l78_78497


namespace triangle_sides_l78_78585

theorem triangle_sides (a : ℕ) (h : a > 0) : 
  (a + 1) + (a + 2) > (a + 3) ∧ (a + 1) + (a + 3) > (a + 2) ∧ (a + 2) + (a + 3) > (a + 1) := 
by 
  sorry

end triangle_sides_l78_78585


namespace exists_a_div_by_3_l78_78637

theorem exists_a_div_by_3 (a : ℝ) (h : ∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 4) = k) :
  ∃ k : ℤ, a = k / 3 :=
by
  sorry

end exists_a_div_by_3_l78_78637


namespace people_sharing_bill_l78_78871

theorem people_sharing_bill (total_bill : ℝ) (tip_percent : ℝ) (share_per_person : ℝ) (n : ℝ) :
  total_bill = 211.00 →
  tip_percent = 0.15 →
  share_per_person = 26.96 →
  abs (n - 9) < 1 :=
by
  intros h1 h2 h3
  sorry

end people_sharing_bill_l78_78871


namespace arithmetic_sum_expression_zero_l78_78538

theorem arithmetic_sum_expression_zero (a d : ℤ) (i j k : ℕ) (S_i S_j S_k : ℤ) :
  S_i = i * (a + (i - 1) * d / 2) →
  S_j = j * (a + (j - 1) * d / 2) →
  S_k = k * (a + (k - 1) * d / 2) →
  (S_i / i * (j - k) + S_j / j * (k - i) + S_k / k * (i - j) = 0) :=
by
  intros hS_i hS_j hS_k
  -- Proof omitted
  sorry

end arithmetic_sum_expression_zero_l78_78538


namespace solution_set_inequality_l78_78867

theorem solution_set_inequality (x : ℝ) : (x + 1) * (2 - x) < 0 ↔ x > 2 ∨ x < -1 :=
sorry

end solution_set_inequality_l78_78867


namespace base_value_l78_78384

theorem base_value (b : ℕ) : (b - 1)^2 * (b - 2) = 256 → b = 17 :=
by
  sorry

end base_value_l78_78384


namespace corveus_sleep_hours_l78_78770

-- Definition of the recommended hours of sleep per day
def recommended_sleep_per_day : ℕ := 6

-- Definition of the hours of sleep Corveus lacks per week
def lacking_sleep_per_week : ℕ := 14

-- Definition of days in a week
def days_in_week : ℕ := 7

-- Prove that Corveus sleeps 4 hours per day given the conditions
theorem corveus_sleep_hours :
  (recommended_sleep_per_day * days_in_week - lacking_sleep_per_week) / days_in_week = 4 :=
by
  -- The proof steps would go here
  sorry

end corveus_sleep_hours_l78_78770


namespace germs_per_dish_l78_78389

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end germs_per_dish_l78_78389


namespace avg_price_per_book_l78_78696

theorem avg_price_per_book (n1 n2 p1 p2 : ℕ) (h1 : n1 = 65) (h2 : n2 = 55) (h3 : p1 = 1380) (h4 : p2 = 900) :
    (p1 + p2) / (n1 + n2) = 19 := by
  sorry

end avg_price_per_book_l78_78696


namespace max_m_eq_4_inequality_a_b_c_l78_78798

noncomputable def f (x : ℝ) : ℝ :=
  |x - 3| + |x + 2|

theorem max_m_eq_4 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 ∧ m ≥ -6 :=
  sorry

theorem inequality_a_b_c (a b c : ℝ) (h : a + 2 * b + c = 4) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
  sorry

end max_m_eq_4_inequality_a_b_c_l78_78798


namespace repeating_decimal_block_length_l78_78854

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l78_78854


namespace max_free_squares_l78_78262

theorem max_free_squares (n : ℕ) :
  ∀ (initial_positions : ℕ), 
    (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → initial_positions = 2) →
    (∀ (i j : ℕ) (move1 move2 : ℕ × ℕ),
       1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
       move1 = (i + 1, j) ∨ move1 = (i - 1, j) ∨ move1 = (i, j + 1) ∨ move1 = (i, j - 1) →
       move2 = (i + 1, j) ∨ move2 = (i - 1, j) ∨ move2 = (i, j + 1) ∨ move2 = (i, j - 1) →
       move1 ≠ move2) →
    ∃ free_squares : ℕ, free_squares = n^2 :=
by
  sorry

end max_free_squares_l78_78262


namespace parallel_vectors_x_value_l78_78802

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  vectors_parallel (1, -2) (x, 1) → x = -1 / 2 :=
by
  sorry

end parallel_vectors_x_value_l78_78802


namespace total_number_of_sheep_l78_78390

theorem total_number_of_sheep (a₁ a₂ a₃ a₄ a₅ a₆ a₇ d : ℤ)
    (h1 : a₂ = a₁ + d)
    (h2 : a₃ = a₁ + 2 * d)
    (h3 : a₄ = a₁ + 3 * d)
    (h4 : a₅ = a₁ + 4 * d)
    (h5 : a₆ = a₁ + 5 * d)
    (h6 : a₇ = a₁ + 6 * d)
    (h_sum : a₁ + a₂ + a₃ = 33)
    (h_seven: 2 * a₂ + 9 = a₇) :
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 133 := sorry

end total_number_of_sheep_l78_78390


namespace ellipse_foci_on_y_axis_l78_78358

theorem ellipse_foci_on_y_axis (k : ℝ) (h1 : 5 + k > 3 - k) (h2 : 3 - k > 0) (h3 : 5 + k > 0) : -1 < k ∧ k < 3 :=
by 
  sorry

end ellipse_foci_on_y_axis_l78_78358


namespace find_d_l78_78109

noncomputable def polynomial_d (a b c d : ℤ) (p q r s : ℤ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
  1 + a + b + c + d = 2024 ∧
  (1 + p) * (1 + q) * (1 + r) * (1 + s) = 2024 ∧
  d = p * q * r * s

theorem find_d (a b c d : ℤ) (h : polynomial_d a b c d 7 10 22 11) : d = 17020 :=
  sorry

end find_d_l78_78109


namespace logical_contradiction_l78_78290

-- Definitions based on the conditions
def all_destroying (x : Type) : Prop := ∀ y : Type, y ≠ x → y → false
def indestructible (x : Type) : Prop := ∀ y : Type, y = x → y → false

theorem logical_contradiction (x : Type) :
  (all_destroying x ∧ indestructible x) → false :=
by
  sorry

end logical_contradiction_l78_78290


namespace train_crosses_platform_in_20_seconds_l78_78309

theorem train_crosses_platform_in_20_seconds 
  (t : ℝ) (lp : ℝ) (lt : ℝ) (tp : ℝ) (sp : ℝ) (st : ℝ) 
  (pass_time : st = lt / tp) (lc : lp = 267) (lc_train : lt = 178) (cross_time : t = sp / st) : 
  t = 20 :=
by
  sorry

end train_crosses_platform_in_20_seconds_l78_78309


namespace solution_is_correct_l78_78221

-- Define the options
inductive Options
| A_some_other
| B_someone_else
| C_other_person
| D_one_other

-- Define the condition as a function that returns the correct option
noncomputable def correct_option : Options :=
Options.B_someone_else

-- The theorem stating that the correct option must be the given choice
theorem solution_is_correct : correct_option = Options.B_someone_else :=
by
  sorry

end solution_is_correct_l78_78221


namespace sum_of_8th_and_10th_terms_arithmetic_sequence_l78_78847

theorem sum_of_8th_and_10th_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 25) (h2 : a + 5 * d = 61) :
  (a + 7 * d) + (a + 9 * d) = 230 := 
sorry

end sum_of_8th_and_10th_terms_arithmetic_sequence_l78_78847


namespace exists_unique_c_for_a_equals_3_l78_78504

theorem exists_unique_c_for_a_equals_3 :
  ∃! c : ℝ, ∀ x ∈ Set.Icc (3 : ℝ) 9, ∃ y ∈ Set.Icc (3 : ℝ) 27, Real.log x / Real.log 3 + Real.log y / Real.log 3 = c :=
sorry

end exists_unique_c_for_a_equals_3_l78_78504


namespace num_rectangles_grid_l78_78962

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l78_78962


namespace trading_cards_initial_total_l78_78547

theorem trading_cards_initial_total (x : ℕ) 
  (h1 : ∃ d : ℕ, d = (1 / 3 : ℕ) * x)
  (h2 : ∃ n1 : ℕ, n1 = (1 / 5 : ℕ) * (1 / 3 : ℕ) * x)
  (h3 : ∃ n2 : ℕ, n2 = (1 / 3 : ℕ) * ((1 / 5 : ℕ) * (1 / 3 : ℕ) * x))
  (h4 : ∃ n3 : ℕ, n3 = (1 / 2 : ℕ) * (2 / 45 : ℕ) * x)
  (h5 : (1 / 15 : ℕ) * x + (2 / 45 : ℕ) * x + (1 / 45 : ℕ) * x = 850) :
  x = 6375 := 
sorry

end trading_cards_initial_total_l78_78547


namespace ratio_of_large_rooms_l78_78176

-- Definitions for the problem conditions
def total_classrooms : ℕ := 15
def total_students : ℕ := 400
def desks_in_large_room : ℕ := 30
def desks_in_small_room : ℕ := 25

-- Define x as the number of large (30-desk) rooms and y as the number of small (25-desk) rooms
variables (x y : ℕ)

-- Two conditions provided by the problem
def classrooms_condition := x + y = total_classrooms
def students_condition := desks_in_large_room * x + desks_in_small_room * y = total_students

-- Our main theorem to prove
theorem ratio_of_large_rooms :
  classrooms_condition x y →
  students_condition x y →
  (x : ℚ) / (total_classrooms : ℚ) = 1 / 3 :=
by
-- Here we would have our proof, but we leave it as "sorry" since the task only requires the statement.
sorry

end ratio_of_large_rooms_l78_78176


namespace smallest_number_with_2020_divisors_l78_78344

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l78_78344


namespace least_alpha_prime_l78_78872

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_distinct_prime (α β : ℕ) : Prop :=
  α ≠ β ∧ is_prime α ∧ is_prime β

theorem least_alpha_prime (α : ℕ) :
  is_distinct_prime α (180 - 2 * α) → α ≥ 41 :=
sorry

end least_alpha_prime_l78_78872


namespace maximum_value_of_expression_l78_78250

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end maximum_value_of_expression_l78_78250


namespace discount_difference_l78_78312

def original_amount : ℚ := 20000
def single_discount_rate : ℚ := 0.30
def first_discount_rate : ℚ := 0.25
def second_discount_rate : ℚ := 0.05

theorem discount_difference :
  (original_amount * (1 - single_discount_rate)) - (original_amount * (1 - first_discount_rate) * (1 - second_discount_rate)) = 250 := by
  sorry

end discount_difference_l78_78312


namespace increased_percentage_l78_78428

theorem increased_percentage (P : ℝ) (N : ℝ) (hN : N = 80) 
  (h : (N + (P / 100) * N) - (N - (25 / 100) * N) = 30) : P = 12.5 := 
by 
  sorry

end increased_percentage_l78_78428


namespace abigail_total_savings_l78_78180

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_total_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end abigail_total_savings_l78_78180


namespace num_rectangles_in_5x5_grid_l78_78977

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l78_78977


namespace sqrt_pos_condition_l78_78286

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l78_78286


namespace contrapositive_squared_l78_78131

theorem contrapositive_squared (a : ℝ) : (a ≤ 0 → a^2 ≤ 0) ↔ (a > 0 → a^2 > 0) :=
by
  sorry

end contrapositive_squared_l78_78131


namespace complement_of_A_in_U_l78_78218

open Set

-- Define the universal set U and the set A.
def U := { x : ℝ | x < 4 }
def A := { x : ℝ | x < 1 }

-- Theorem statement of the complement of A with respect to U equaling [1, 4).
theorem complement_of_A_in_U : (U \ A) = { x : ℝ | 1 ≤ x ∧ x < 4 } :=
sorry

end complement_of_A_in_U_l78_78218


namespace max_singular_words_l78_78381

theorem max_singular_words (alphabet_length : ℕ) (word_length : ℕ) (strip_length : ℕ) 
  (num_non_overlapping_pieces : ℕ) (h_alphabet : alphabet_length = 25)
  (h_word_length : word_length = 17) (h_strip_length : strip_length = 5^18)
  (h_non_overlapping : num_non_overlapping_pieces = 5^16) : 
  ∃ max_singular_words, max_singular_words = 2 * 5^17 :=
by {
  -- proof to be completed
  sorry
}

end max_singular_words_l78_78381


namespace num_rectangles_in_5x5_grid_l78_78994

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l78_78994


namespace smallest_number_with_2020_divisors_l78_78335

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l78_78335


namespace chord_probability_concentric_circles_l78_78893

noncomputable def chord_intersects_inner_circle_probability : ℝ :=
  sorry

theorem chord_probability_concentric_circles :
  let r₁ := 2
  let r₂ := 3
  ∀ (P₁ P₂ : ℝ × ℝ),
    dist P₁ (0, 0) = r₂ ∧ dist P₂ (0, 0) = r₂ →
    chord_intersects_inner_circle_probability = 0.148 :=
  sorry

end chord_probability_concentric_circles_l78_78893


namespace find_x_l78_78037

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l78_78037


namespace sum_of_cubes_decomposition_l78_78865

theorem sum_of_cubes_decomposition :
  ∃ a b c d e : ℤ, (∀ x : ℤ, 1728 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 132) :=
by
  sorry

end sum_of_cubes_decomposition_l78_78865


namespace days_for_Q_wages_l78_78177

variables (P Q S : ℝ) (D : ℝ)

theorem days_for_Q_wages (h1 : S = 24 * P) (h2 : S = 15 * (P + Q)) : S = D * Q → D = 40 :=
by
  sorry

end days_for_Q_wages_l78_78177


namespace zhou_yu_age_eq_l78_78161

-- Define the conditions based on the problem statement
variable (x : ℕ)  -- x represents the tens digit of Zhou Yu's age

-- Condition: The tens digit is three less than the units digit
def units_digit := x + 3

-- Define Zhou Yu's age based on the tens and units digits
def zhou_yu_age := 10 * x + units_digit x

-- Prove the correct equation representing Zhou Yu's lifespan
theorem zhou_yu_age_eq : zhou_yu_age x = (units_digit x) ^ 2 :=
by sorry

end zhou_yu_age_eq_l78_78161


namespace a_mul_b_value_l78_78722

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l78_78722


namespace num_rectangles_in_5x5_grid_l78_78978

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l78_78978


namespace a_mul_b_value_l78_78721

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l78_78721


namespace multiples_of_3_ending_number_l78_78804

theorem multiples_of_3_ending_number :
  ∃ n, ∃ k, k = 93 ∧ (∀ m, 81 + 3 * m = n → 0 ≤ m ∧ m < k) ∧ n = 357 := 
by
  sorry

end multiples_of_3_ending_number_l78_78804


namespace man_speed_l78_78614

theorem man_speed {m l: ℝ} (TrainLength : ℝ := 385) (TrainSpeedKmH : ℝ := 60)
  (PassTimeSeconds : ℝ := 21) (RelativeSpeed : ℝ) (ManSpeedKmH : ℝ) 
  (ConversionFactor : ℝ := 3.6) (expected_speed : ℝ := 5.99) : 
  RelativeSpeed = TrainSpeedKmH/ConversionFactor + m/ConversionFactor ∧ 
  TrainLength = RelativeSpeed * PassTimeSeconds →
  abs (m*ConversionFactor - expected_speed) < 0.01 :=
by
  sorry

end man_speed_l78_78614


namespace required_volume_proof_l78_78220

-- Defining the conditions
def initial_volume : ℝ := 60
def initial_concentration : ℝ := 0.10
def final_concentration : ℝ := 0.15

-- Defining the equation
def required_volume (V : ℝ) : Prop :=
  (initial_concentration * initial_volume + V = final_concentration * (initial_volume + V))

-- Stating the proof problem
theorem required_volume_proof :
  ∃ V : ℝ, required_volume V ∧ V = 3 / 0.85 :=
by {
  -- Proof skipped
  sorry
}

end required_volume_proof_l78_78220


namespace germs_per_dish_l78_78386

/--
Given:
- the total number of germs is \(5.4 \times 10^6\),
- the number of petri dishes is 10,800,

Prove:
- the number of germs per dish is 500.
-/
theorem germs_per_dish (total_germs : ℝ) (petri_dishes: ℕ) (h₁: total_germs = 5.4 * 10^6) (h₂: petri_dishes = 10800) :
  (total_germs / petri_dishes = 500) :=
sorry

end germs_per_dish_l78_78386


namespace min_hypotenuse_of_right_triangle_l78_78578

theorem min_hypotenuse_of_right_triangle (a b c k : ℝ) (h₁ : k = a + b + c) (h₂ : a^2 + b^2 = c^2) : 
  c ≥ (Real.sqrt 2 - 1) * k := 
sorry

end min_hypotenuse_of_right_triangle_l78_78578


namespace distance_between_points_l78_78943

theorem distance_between_points:
  dist (0, 4) (3, 0) = 5 :=
by
  sorry

end distance_between_points_l78_78943


namespace range_of_a_l78_78360

/-- Proposition p: ∀ x ∈ [1,2], x² - a ≥ 0 -/
def prop_p (a : ℝ) : Prop := 
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

/-- Proposition q: ∃ x₀ ∈ ℝ, x + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop := 
  ∃ x₀ : ℝ, ∃ x : ℝ, x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : prop_p a ∧ prop_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l78_78360


namespace perpendicular_lines_a_value_l78_78794

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a-2)*x + a*y = 1 ↔ 2*x + 3*y = 5) → a = 4/5 := by
sorry

end perpendicular_lines_a_value_l78_78794


namespace find_x_l78_78020

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l78_78020


namespace num_rectangles_in_5x5_grid_l78_78972

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l78_78972


namespace smallest_number_with_2020_divisors_l78_78346

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l78_78346


namespace sqrt_div_val_l78_78733

theorem sqrt_div_val (n : ℕ) (h : n = 3600) : (Nat.sqrt n) / 15 = 4 := by 
  sorry

end sqrt_div_val_l78_78733


namespace find_pairs_solution_l78_78940

theorem find_pairs_solution (x y : ℝ) :
  (x^3 + x^2 * y + x * y^2 + y^3 = 8 * (x^2 + x * y + y^2 + 1)) ↔ 
  (x, y) = (8, -2) ∨ (x, y) = (-2, 8) ∨ 
  (x, y) = (4 + Real.sqrt 15, 4 - Real.sqrt 15) ∨ 
  (x, y) = (4 - Real.sqrt 15, 4 + Real.sqrt 15) :=
by 
  sorry

end find_pairs_solution_l78_78940


namespace ball_min_bounces_reach_target_height_l78_78463

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end ball_min_bounces_reach_target_height_l78_78463


namespace sufficient_condition_l78_78595

theorem sufficient_condition (a b : ℝ) : ab ≠ 0 → a ≠ 0 :=
sorry

end sufficient_condition_l78_78595


namespace roses_count_l78_78259

def total_roses : Nat := 80
def red_roses : Nat := 3 * total_roses / 4
def remaining_roses : Nat := total_roses - red_roses
def yellow_roses : Nat := remaining_roses / 4
def white_roses : Nat := remaining_roses - yellow_roses

theorem roses_count :
  red_roses + white_roses = 75 :=
by
  sorry

end roses_count_l78_78259


namespace tom_gaming_system_value_l78_78144

theorem tom_gaming_system_value
    (V : ℝ) 
    (h1 : 0.80 * V + 80 - 10 = 160 + 30) 
    : V = 150 :=
by
  -- Logical steps for the proof will be added here.
  sorry

end tom_gaming_system_value_l78_78144


namespace range_of_sin_cos_expression_l78_78645

variable (a b c A B C : ℝ)

theorem range_of_sin_cos_expression
  (h1 : a = b)
  (h2 : c * Real.sin A = -a * Real.cos C) :
  1 < 2 * Real.sin (A + Real.pi / 6) :=
sorry

end range_of_sin_cos_expression_l78_78645


namespace total_texts_sent_l78_78241

theorem total_texts_sent (grocery_texts : ℕ) (response_texts_ratio : ℕ) (police_texts_percentage : ℚ) :
  grocery_texts = 5 →
  response_texts_ratio = 5 →
  police_texts_percentage = 0.10 →
  let response_texts := grocery_texts * response_texts_ratio
  let previous_texts := response_texts + grocery_texts
  let police_texts := previous_texts * police_texts_percentage
  response_texts + grocery_texts + police_texts = 33 :=
by
  sorry

end total_texts_sent_l78_78241


namespace ratio_of_tetrahedron_to_cube_volume_l78_78175

theorem ratio_of_tetrahedron_to_cube_volume (x : ℝ) (hx : 0 < x) :
  let V_cube := x^3
  let a_tetrahedron := (x * Real.sqrt 3) / 2
  let V_tetrahedron := (a_tetrahedron^3 * Real.sqrt 2) / 12
  (V_tetrahedron / V_cube) = (Real.sqrt 6 / 32) :=
by
  sorry

end ratio_of_tetrahedron_to_cube_volume_l78_78175


namespace triangular_array_of_coins_l78_78928

theorem triangular_array_of_coins (N : ℤ) (h : N * (N + 1) / 2 = 3003) : N = 77 :=
by
  sorry

end triangular_array_of_coins_l78_78928


namespace can_choose_P_l78_78603

-- Define the objects in the problem,
-- types, constants, and assumptions as per the problem statement.

theorem can_choose_P (cube : ℝ) (P Q R S T A B C D : ℝ)
  (edge_length : cube = 10)
  (AR_RB_eq_CS_SB : ∀ AR RB CS SB, (AR / RB = 7 / 3) ∧ (CS / SB = 7 / 3))
  : ∃ P, 2 * (Q - R) = (P - Q) + (R - S) := by
  sorry

end can_choose_P_l78_78603


namespace max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l78_78113

theorem max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h₁ : a ≤ b) (h₂ : b ≤ c) (h₃ : c ≤ 2 * a) :
    b / a + c / b + a / c ≤ 7 / 2 := 
  sorry

end max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l78_78113


namespace carrie_fourth_day_miles_l78_78551

theorem carrie_fourth_day_miles (d1 d2 d3 d4: ℕ) (charge_interval charges: ℕ) 
  (h1: d1 = 135) 
  (h2: d2 = d1 + 124) 
  (h3: d3 = 159) 
  (h4: charge_interval = 106) 
  (h5: charges = 7):
  d4 = 742 - (d1 + d2 + d3) :=
by
  sorry

end carrie_fourth_day_miles_l78_78551


namespace segment_lengths_l78_78166

theorem segment_lengths (AB BC CD DE EF : ℕ) 
  (h1 : AB > BC)
  (h2 : BC > CD)
  (h3 : CD > DE)
  (h4 : DE > EF)
  (h5 : AB = 2 * EF)
  (h6 : AB + BC + CD + DE + EF = 53) :
  (AB, BC, CD, DE, EF) = (14, 12, 11, 9, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 11, 8, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 10, 9, 7) :=
sorry

end segment_lengths_l78_78166


namespace rectangles_in_grid_l78_78988

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l78_78988


namespace sequence_sixth_term_is_364_l78_78519

theorem sequence_sixth_term_is_364 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 7) (h3 : a 3 = 20)
  (h4 : ∀ n, a (n + 1) = 1 / 3 * (a n + a (n + 2))) :
  a 6 = 364 :=
by
  -- Proof skipped
  sorry

end sequence_sixth_term_is_364_l78_78519


namespace some_number_value_l78_78351

theorem some_number_value (x : ℕ) (some_number : ℕ) : x = 5 → ((x / 5) + some_number = 4) → some_number = 3 :=
by
  intros h1 h2
  sorry

end some_number_value_l78_78351


namespace suki_bag_weight_is_22_l78_78704

noncomputable def weight_of_suki_bag : ℝ :=
  let bags_suki := 6.5
  let bags_jimmy := 4.5
  let weight_jimmy_per_bag := 18.0
  let total_containers := 28
  let weight_per_container := 8.0
  let total_weight_jimmy := bags_jimmy * weight_jimmy_per_bag
  let total_weight_combined := total_containers * weight_per_container
  let total_weight_suki := total_weight_combined - total_weight_jimmy
  total_weight_suki / bags_suki

theorem suki_bag_weight_is_22 : weight_of_suki_bag = 22 :=
by
  sorry

end suki_bag_weight_is_22_l78_78704


namespace num_rectangles_grid_l78_78963

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l78_78963


namespace smallest_five_digit_congruent_to_3_mod_17_l78_78156

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ n : ℤ, (n > 9999) ∧ (n % 17 = 3) ∧ (∀ m : ℤ, (m > 9999) ∧ (m % 17 = 3) → n ≤ m) :=
by
  use 10012
  split
  { sorry }
  split
  { sorry }
  { sorry }

end smallest_five_digit_congruent_to_3_mod_17_l78_78156


namespace find_x_tan_eq_l78_78016

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l78_78016


namespace number_of_balls_l78_78383

noncomputable def frequency_of_yellow (n : ℕ) : ℚ := 9 / n

theorem number_of_balls (n : ℕ) (h1 : frequency_of_yellow n = 0.30) : n = 30 :=
by sorry

end number_of_balls_l78_78383


namespace uncle_gave_13_l78_78771

-- Define all the given constants based on the conditions.
def J := 7    -- cost of the jump rope
def B := 12   -- cost of the board game
def P := 4    -- cost of the playground ball
def S := 6    -- savings from Dalton's allowance
def N := 4    -- additional amount needed

-- Derived quantities
def total_cost := J + B + P

-- Statement: to prove Dalton's uncle gave him $13.
theorem uncle_gave_13 : (total_cost - N) - S = 13 := by
  sorry

end uncle_gave_13_l78_78771


namespace num_rectangles_in_5x5_grid_l78_78992

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l78_78992


namespace arithmetic_mean_of_two_digit_multiples_of_9_l78_78148

theorem arithmetic_mean_of_two_digit_multiples_of_9 : 
  let a1 := 18 in
  let an := 99 in
  let d := 9 in
  let n := (an - a1) / d + 1 in
  let S := n * (a1 + an) / 2 in
  (S / n : ℝ) = 58.5 :=
by
  let a1 := 18
  let an := 99
  let d := 9
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  show (S / n : ℝ) = 58.5
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l78_78148


namespace amy_minimum_disks_l78_78618

theorem amy_minimum_disks :
  ∃ (d : ℕ), (d = 19) ∧ ( ∀ (f : ℕ), 
  (f = 40) ∧ ( ∀ (n m k : ℕ), 
  (n + m + k = f) ∧ ( ∀ (a b c : ℕ),
  (a = 8) ∧ (b = 15) ∧ (c = (f - a - b))
  ∧ ( ∀ (size_a size_b size_c : ℚ),
  (size_a = 0.6) ∧ (size_b = 0.55) ∧ (size_c = 0.45)
  ∧ ( ∀ (disk_space : ℚ),
  (disk_space = 1.44)
  ∧ ( ∀ (x y z : ℕ),
  (x = n * ⌈size_a / disk_space⌉) 
  ∧ (y = m * ⌈size_b / disk_space⌉) 
  ∧ (z = k * ⌈size_c / disk_space⌉)
  ∧ (x + y + z = d)) ∧ (size_a * a + size_b * b + size_c * c ≤ disk_space * d)))))) := sorry

end amy_minimum_disks_l78_78618


namespace maria_age_l78_78533

variable (M J : Nat)

theorem maria_age (h1 : J = M + 12) (h2 : M + J = 40) : M = 14 := by
  sorry

end maria_age_l78_78533


namespace prob_exactly_two_same_project_l78_78707

open Nat

theorem prob_exactly_two_same_project : 
  let total_ways := 7^3 in
  let choose_two := choose 3 2 in
  let ways_to_assign_two := 7 * 6 in
  let favorable_ways := choose_two * ways_to_assign_two in
  let probability := favorable_ways / total_ways in
  probability = (18 : ℚ) / 49 :=
by
  let total_ways := 7^3
  let choose_two := choose 3 2
  let ways_to_assign_two := 7 * 6
  let favorable_ways := choose_two * ways_to_assign_two
  let probability := favorable_ways / total_ways

  -- Expected total value checks
  have h_fw : favorable_ways = 18 * 7 := rfl
  have h_tw : total_ways = 7 * 7 * 7 := rfl
  
  -- Calculate probability
  have h_exp : favorable_ways / total_ways = (18 : ℚ) / 49 := by 
    rw [h_fw, h_tw]
    norm_num

  exact h_exp

end prob_exactly_two_same_project_l78_78707


namespace player_1_points_l78_78880

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l78_78880


namespace central_angle_measures_l78_78953

-- Definitions for the conditions
def perimeter_eq (r l : ℝ) : Prop := l + 2 * r = 6
def area_eq (r l : ℝ) : Prop := (1 / 2) * l * r = 2
def central_angle (r l α : ℝ) : Prop := α = l / r

-- The final proof statement
theorem central_angle_measures (r l α : ℝ) (h1 : perimeter_eq r l) (h2 : area_eq r l) :
  central_angle r l α → (α = 1 ∨ α = 4) :=
sorry

end central_angle_measures_l78_78953


namespace probability_prime_ball_l78_78130

/-- Ten balls, numbered 6 through 15, are placed in a hat. Each ball is equally likely to be chosen. 
If one ball is chosen, the probability that the number on the selected ball is a prime number 
is 3 out of 10. -/
theorem probability_prime_ball :
  let balls := {n | 6 ≤ n ∧ n ≤ 15} in
  let primes := {n | n ∈ balls ∧ Nat.Prime n} in
  (primes.card.to_rat / balls.card.to_rat = (3 : ℚ) / 10) := 
by
  sorry

end probability_prime_ball_l78_78130


namespace exists_prime_not_dividing_n_pow_p_minus_p_l78_78544

theorem exists_prime_not_dividing_n_pow_p_minus_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬ q ∣ (n^p - ↑p) := by
  sorry

end exists_prime_not_dividing_n_pow_p_minus_p_l78_78544


namespace work_hours_to_pay_off_debt_l78_78672

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end work_hours_to_pay_off_debt_l78_78672


namespace ratio_a3_b3_l78_78654

theorem ratio_a3_b3 (a : ℝ) (ha : a ≠ 0)
  (h1 : a = b₁)
  (h2 : a * q * b = 2)
  (h3 : b₄ = 8 * a * q^3) :
  (∃ r : ℝ, r = -5 ∨ r = -3.2) :=
by
  sorry

end ratio_a3_b3_l78_78654


namespace repeat_block_of_7_div_13_l78_78860

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l78_78860


namespace marco_paint_fraction_l78_78807

theorem marco_paint_fraction (W : ℝ) (M : ℝ) (minutes_paint : ℝ) (fraction_paint : ℝ) :
  M = 60 ∧ W = 1 ∧ minutes_paint = 12 ∧ fraction_paint = 1/5 → 
  (minutes_paint / M) * W = fraction_paint := 
by
  sorry

end marco_paint_fraction_l78_78807


namespace exist_pairwise_distinct_gcd_l78_78254

theorem exist_pairwise_distinct_gcd (S : Set ℕ) (h_inf : S.Infinite) 
  (h_gcd : ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ gcd a b ≠ gcd c d) :
  ∃ x y z : ℕ, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x := 
by sorry

end exist_pairwise_distinct_gcd_l78_78254


namespace stratified_sample_correct_l78_78093

variable (popA popB popC : ℕ) (totalSample : ℕ)

def stratified_sample (popA popB popC totalSample : ℕ) : ℕ × ℕ × ℕ :=
  let totalChickens := popA + popB + popC
  let sampledA := (popA * totalSample) / totalChickens
  let sampledB := (popB * totalSample) / totalChickens
  let sampledC := (popC * totalSample) / totalChickens
  (sampledA, sampledB, sampledC)

theorem stratified_sample_correct
  (hA : popA = 12000) (hB : popB = 8000) (hC : popC = 4000) (hSample : totalSample = 120) :
  stratified_sample popA popB popC totalSample = (60, 40, 20) :=
by
  sorry

end stratified_sample_correct_l78_78093


namespace smallest_fraction_l78_78472

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (min (min (min (x / 2022) (2022 / (x - 1))) ((x + 1) / 2022)) (2022 / x)) (2022 / (x + 1)) = 2022 / (x + 1) :=
sorry

end smallest_fraction_l78_78472


namespace solution_mod_5_l78_78776

theorem solution_mod_5 (a : ℤ) : 
  (a^3 + 3 * a + 1) % 5 = 0 ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  sorry

end solution_mod_5_l78_78776


namespace a_mul_b_value_l78_78719

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l78_78719


namespace pairs_sold_l78_78610

-- Define the given conditions
def initial_large_pairs : ℕ := 22
def initial_medium_pairs : ℕ := 50
def initial_small_pairs : ℕ := 24
def pairs_left : ℕ := 13

-- Translate to the equivalent proof problem
theorem pairs_sold : (initial_large_pairs + initial_medium_pairs + initial_small_pairs) - pairs_left = 83 := by
  sorry

end pairs_sold_l78_78610


namespace second_set_length_is_20_l78_78392

-- Define the lengths
def length_first_set : ℕ := 4
def length_second_set : ℕ := 5 * length_first_set

-- Formal proof statement
theorem second_set_length_is_20 : length_second_set = 20 :=
by
  sorry

end second_set_length_is_20_l78_78392


namespace Peter_vacation_l78_78407

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end Peter_vacation_l78_78407


namespace angle_movement_condition_l78_78172

noncomputable def angle_can_reach_bottom_right (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) : Prop :=
  (m % 2 = 1) ∧ (n % 2 = 1)

theorem angle_movement_condition (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) :
  angle_can_reach_bottom_right m n h1 h2 ↔ (m % 2 = 1 ∧ n % 2 = 1) :=
sorry

end angle_movement_condition_l78_78172


namespace find_x_l78_78019

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l78_78019


namespace number_of_students_l78_78759

def total_students (a b : ℕ) : ℕ :=
  a + b

variables (a b : ℕ)

theorem number_of_students (h : 48 * a + 45 * b = 972) : total_students a b = 21 :=
by
  sorry

end number_of_students_l78_78759


namespace prob_lfloor_XZ_YZ_product_eq_33_l78_78545

noncomputable def XZ_YZ_product : ℝ :=
  let AB := 15
  let BC := 14
  let CA := 13
  -- Definition of points and conditions
  -- Note: Specific geometric definitions and conditions need to be properly defined as per Lean's geometry library. This is a simplified placeholder.
  sorry

theorem prob_lfloor_XZ_YZ_product_eq_33 :
  (⌊XZ_YZ_product⌋ = 33) := sorry

end prob_lfloor_XZ_YZ_product_eq_33_l78_78545


namespace notebook_cost_l78_78090

theorem notebook_cost {s n c : ℕ}
  (h1 : s > 18)
  (h2 : c > n)
  (h3 : s * n * c = 2275) :
  c = 13 :=
sorry

end notebook_cost_l78_78090


namespace find_x_tan_identity_l78_78051

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l78_78051


namespace music_library_avg_disk_space_per_hour_l78_78754

theorem music_library_avg_disk_space_per_hour 
  (days_of_music: ℕ) (total_space_MB: ℕ) (hours_in_day: ℕ) 
  (h1: days_of_music = 15) 
  (h2: total_space_MB = 18000) 
  (h3: hours_in_day = 24) : 
  (total_space_MB / (days_of_music * hours_in_day)) = 50 := 
by
  sorry

end music_library_avg_disk_space_per_hour_l78_78754


namespace max_blocks_that_fit_l78_78900

noncomputable def box_volume : ℕ :=
  3 * 4 * 2

noncomputable def block_volume : ℕ :=
  2 * 1 * 2

noncomputable def max_blocks (box_volume : ℕ) (block_volume : ℕ) : ℕ :=
  box_volume / block_volume

theorem max_blocks_that_fit : max_blocks box_volume block_volume = 6 :=
by
  sorry

end max_blocks_that_fit_l78_78900


namespace smallest_number_with_2020_divisors_is_correct_l78_78347

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l78_78347


namespace tan_sin_cos_eq_l78_78024

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l78_78024


namespace minimum_value_squared_sum_minimum_value_squared_sum_equality_l78_78821

theorem minimum_value_squared_sum (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

theorem minimum_value_squared_sum_equality (a b c t : ℝ) (h : a + b + c = t) 
  (ha : a = t / 3) (hb : b = t / 3) (hc : c = t / 3) : 
  a^2 + b^2 + c^2 = t^2 / 3 := by
  sorry

end minimum_value_squared_sum_minimum_value_squared_sum_equality_l78_78821


namespace sphere_radius_l78_78362

theorem sphere_radius (A : ℝ) (k1 k2 k3 : ℝ) (h : A = 64 * Real.pi) : ∃ r : ℝ, r = 4 := 
by 
  sorry

end sphere_radius_l78_78362


namespace geometric_sequences_l78_78081

variable (a_n b_n : ℕ → ℕ) -- Geometric sequences
variable (S_n T_n : ℕ → ℕ) -- Sums of first n terms
variable (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4)

theorem geometric_sequences (n : ℕ) (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 := 
sorry

end geometric_sequences_l78_78081


namespace find_a_and_min_value_minimum_value_l78_78075

open Real

namespace MathProof

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * log x

-- State the tangent line condition
theorem find_a_and_min_value (a : ℝ) :
  (∀ x, deriv (f a) x = 2 * x - (2 * a) / x) ∧ (2 - 2 * a = 2) :=
sorry

-- State the minimum value condition
theorem minimum_value (a : ℝ) :
  (∀ x, f a x = x^2) ∧ (1 ≤ 2) ∧ (∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≥ 1) :=
sorry

end MathProof

end find_a_and_min_value_minimum_value_l78_78075


namespace value_of_a_star_b_l78_78723

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l78_78723


namespace product_of_dodecagon_l78_78475

open Complex

theorem product_of_dodecagon (Q : Fin 12 → ℂ) (h₁ : Q 0 = 2) (h₇ : Q 6 = 8) :
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8) * (Q 9) * (Q 10) * (Q 11) = 244140624 :=
sorry

end product_of_dodecagon_l78_78475


namespace speed_in_still_water_l78_78746

/-- A man can row upstream at 37 km/h and downstream at 53 km/h, 
    prove that the speed of the man in still water is 45 km/h. --/
theorem speed_in_still_water 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ)
  (h1 : upstream_speed = 37)
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := 
by 
  sorry

end speed_in_still_water_l78_78746


namespace number_of_dots_on_faces_l78_78143

theorem number_of_dots_on_faces (d A B C D : ℕ) 
  (h1 : d = 6)
  (h2 : A = 3)
  (h3 : B = 5)
  (h4 : C = 6)
  (h5 : D = 5) :
  A = 3 ∧ B = 5 ∧ C = 6 ∧ D = 5 :=
by {
  sorry
}

end number_of_dots_on_faces_l78_78143


namespace find_A_l78_78820

-- Definitions and conditions
def f (A B : ℝ) (x : ℝ) : ℝ := A * x - 3 * B^2 
def g (B C : ℝ) (x : ℝ) : ℝ := B * x + C

theorem find_A (A B C : ℝ) (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  f A B (g B C 1) = 0 → A = (3 * B^2) / (B + C) :=
by
  -- Introduction of the hypotheses
  intro h
  sorry

end find_A_l78_78820


namespace height_of_tank_A_l78_78127

theorem height_of_tank_A (C_A C_B h_B : ℝ) (capacity_ratio : ℝ) :
  C_A = 8 → C_B = 10 → h_B = 8 → capacity_ratio = 0.4800000000000001 →
  ∃ h_A : ℝ, h_A = 6 := by
  intros hCA hCB hHB hCR
  sorry

end height_of_tank_A_l78_78127


namespace length_of_other_diagonal_l78_78862

theorem length_of_other_diagonal (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 15) (h2 : A = 150) : d2 = 20 :=
by
  sorry

end length_of_other_diagonal_l78_78862


namespace boys_at_park_l78_78441

theorem boys_at_park (girls parents groups people_per_group : ℕ) 
  (h_girls : girls = 14) 
  (h_parents : parents = 50)
  (h_groups : groups = 3) 
  (h_people_per_group : people_per_group = 25) : 
  (groups * people_per_group) - (girls + parents) = 11 := 
by 
  -- Not providing the proof, only the statement
  sorry

end boys_at_park_l78_78441


namespace marbles_difference_l78_78534

-- Conditions
def L : ℕ := 23
def F : ℕ := 9

-- Proof statement
theorem marbles_difference : L - F = 14 := by
  sorry

end marbles_difference_l78_78534


namespace joanne_trip_l78_78391

theorem joanne_trip (a b c x : ℕ) (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 100 * c + 10 * a + b - (100 * a + 10 * b + c) = 60 * x) : 
  a^2 + b^2 + c^2 = 51 :=
by
  sorry

end joanne_trip_l78_78391


namespace sqrt_meaningful_condition_l78_78283

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l78_78283


namespace BethsHighSchoolStudents_l78_78931

-- Define the variables
variables (B P : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := B = 4 * P
def condition2 : Prop := B + P = 5000

-- The theorem to be proved
theorem BethsHighSchoolStudents (h1 : condition1 B P) (h2 : condition2 B P) : B = 4000 :=
by
  -- Proof will be here
  sorry

end BethsHighSchoolStudents_l78_78931


namespace distance_from_p_to_center_is_2_sqrt_10_l78_78921

-- Define the conditions
def r : ℝ := 4
def PA : ℝ := 4
def PB : ℝ := 6

-- The conjecture to prove
theorem distance_from_p_to_center_is_2_sqrt_10
  (r : ℝ) (PA : ℝ) (PB : ℝ) 
  (PA_mul_PB : PA * PB = 24) 
  (r_squared : r = 4)  : 
  ∃ d : ℝ, d = 2 * Real.sqrt 10 := 
by sorry

end distance_from_p_to_center_is_2_sqrt_10_l78_78921


namespace combined_money_l78_78571

/-- Tom has a quarter the money of Nataly. Nataly has three times the money of Raquel.
     Sam has twice the money of Nataly. Raquel has $40. Prove that combined they have $430. -/
theorem combined_money : 
  ∀ (T R N S : ℕ), 
    (T = N / 4) ∧ 
    (N = 3 * R) ∧ 
    (S = 2 * N) ∧ 
    (R = 40) → 
    T + R + N + S = 430 := 
by
  sorry

end combined_money_l78_78571


namespace parabola_incorrect_statement_B_l78_78295

theorem parabola_incorrect_statement_B 
  (y₁ y₂ : ℝ → ℝ) 
  (h₁ : ∀ x, y₁ x = 2 * x^2) 
  (h₂ : ∀ x, y₂ x = -2 * x^2) : 
  ¬ (∀ x < 0, y₁ x < y₁ (x + 1)) ∧ (∀ x < 0, y₂ x < y₂ (x + 1)) := 
by 
  sorry

end parabola_incorrect_statement_B_l78_78295


namespace problem_statement_l78_78395

variable (a b : Type) [LinearOrder a] [LinearOrder b]
variable (α β : Type) [LinearOrder α] [LinearOrder β]

-- Given conditions
def line_perpendicular_to_plane (l : Type) (p : Type) [LinearOrder l] [LinearOrder p] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

def lines_parallel (l1 : Type) (l2 : Type) [LinearOrder l1] [LinearOrder l2] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

theorem problem_statement (a b α : Type) [LinearOrder a] [LinearOrder b] [LinearOrder α]
(val_perp1 : line_perpendicular_to_plane a α)
(val_perp2 : line_perpendicular_to_plane b α)
: lines_parallel a b :=
sorry

end problem_statement_l78_78395


namespace sam_wins_probability_l78_78839

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l78_78839


namespace train_speed_km_hr_calc_l78_78479

theorem train_speed_km_hr_calc :
  let length := 175 -- length of the train in meters
  let time := 3.499720022398208 -- time to cross the pole in seconds
  let speed_mps := length / time -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed from m/s to km/hr
  speed_kmph = 180.025923226 := 
sorry

end train_speed_km_hr_calc_l78_78479


namespace hawks_points_l78_78168

theorem hawks_points (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 6) : H = 38 :=
sorry

end hawks_points_l78_78168


namespace germs_per_dish_l78_78388

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end germs_per_dish_l78_78388


namespace find_n_minus_m_l78_78294

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 25 - r^2 = 0

-- Given conditions
def circles_intersect (r : ℝ) : Prop :=
(r > 0) ∧ (∃ x y, circle1 x y ∧ circle2 x y r)

-- Prove the range of r for intersection
theorem find_n_minus_m : 
(∀ (r : ℝ), 2 ≤ r ∧ r ≤ 12 ↔ circles_intersect r) → 
12 - 2 = 10 :=
by
  sorry

end find_n_minus_m_l78_78294


namespace jake_work_hours_l78_78674

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end jake_work_hours_l78_78674


namespace increasing_interval_of_g_l78_78956

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (Real.pi / 3 - 2 * x)) -
  2 * (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  f (x + Real.pi / 12)

theorem increasing_interval_of_g :
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
  ∃ a b, a = -Real.pi / 12 ∧ b = Real.pi / 4 ∧
      (∀ x y, (a ≤ x ∧ x ≤ y ∧ y ≤ b) → g x ≤ g y) :=
sorry

end increasing_interval_of_g_l78_78956


namespace natalie_bushes_l78_78323

theorem natalie_bushes (bush_yield : ℕ) (containers_per_zucchini : ℕ → ℕ) (desired_zucchinis : ℕ):
  (bush_yield = 10) →
  (containers_per_zucchini 1 = 2) →
  (desired_zucchinis = 60) →
  ∃ bushes_needed : ℕ, bushes_needed = 12 :=
by
  intros h_bush_yield h_containers_per_zucchini h_desired_zucchinis
  use 12
  sorry

end natalie_bushes_l78_78323


namespace increasing_distinct_digits_2020_to_2400_l78_78482

theorem increasing_distinct_digits_2020_to_2400 :
  ∃ (count : ℕ), count = 15 ∧
  count = (Nat.choose 6 2).toNat :=
by
  use 15
  split
  · exact rfl
  · exact rfl

end increasing_distinct_digits_2020_to_2400_l78_78482


namespace probability_all_heads_or_tails_l78_78422

theorem probability_all_heads_or_tails (n : ℕ) (h : n = 6) :
  let total_outcomes := 2^n in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = 1 / 32 :=
by
  let total_outcomes := 2^n
  let favorable_outcomes := 2
  have h1 : total_outcomes = 64 := by rw h; exact pow_succ 2 5
  have h2 : favorable_outcomes / total_outcomes = 1 / 32 := by
    rw h1
    norm_num
  exact h2

end probability_all_heads_or_tails_l78_78422


namespace no_solution_for_floor_eq_l78_78941

theorem no_solution_for_floor_eq :
  ∀ s : ℝ, ¬ (⌊s⌋ + s = 15.6) :=
by sorry

end no_solution_for_floor_eq_l78_78941


namespace problem_b_50_l78_78769

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n ≥ 1, b (n + 1) = b n + 3 * n

theorem problem_b_50 (b : ℕ → ℕ) (h : seq b) : b 50 = 3678 := 
sorry

end problem_b_50_l78_78769


namespace total_red_marbles_l78_78243

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end total_red_marbles_l78_78243


namespace solve_z_plus_inv_y_l78_78705

theorem solve_z_plus_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 :=
sorry

end solve_z_plus_inv_y_l78_78705


namespace john_trip_total_time_l78_78819

theorem john_trip_total_time :
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  t1 + t2 + t3 + t4 + t5 = 872 :=
by
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  have h1: t1 + t2 + t3 + t4 + t5 = 2 + (3 * 2) + (4 * (3 * 2)) + (5 * (4 * (3 * 2))) + (6 * (5 * (4 * (3 * 2)))) := by
    sorry
  have h2: 2 + 6 + 24 + 120 + 720 = 872 := by
    sorry
  exact h2

end john_trip_total_time_l78_78819


namespace intersection_M_N_l78_78399

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | ∃ k : ℕ, x = 2 * k}

theorem intersection_M_N :
  M ∩ N = {2, 4, 8} :=
by sorry

end intersection_M_N_l78_78399


namespace smallest_n_with_divisors_2020_l78_78334

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l78_78334


namespace ad_eb_intersect_on_altitude_l78_78098

open EuclideanGeometry

variables {A B C D E F G K L C1 : Point}

-- Definitions for the problem
variables (triangleABC : Triangle A B C)
  (squareAEFC : Square A E F C)
  (squareBDGC : Square B D G C)
  (altitudeCC1 : Line C C1)
  (lineDA : Line A D)
  (lineEB : Line B E)

-- Definition of intersection
def intersects_on_altitude (pt : Point) : Prop :=
  pt ∈ lineDA ∧ pt ∈ lineEB ∧ pt ∈ altitudeCC1

-- The theorem to be proved
theorem ad_eb_intersect_on_altitude : 
  ∃ pt : Point, intersects_on_altitude lineDA lineEB altitudeCC1 pt := 
sorry

end ad_eb_intersect_on_altitude_l78_78098


namespace tom_tim_typing_ratio_l78_78296

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) :
  M / T = 5 :=
by
  -- Proof to be completed
  sorry

end tom_tim_typing_ratio_l78_78296


namespace total_distance_covered_l78_78307

theorem total_distance_covered (d : ℝ) :
  (d / 5 + d / 10 + d / 15 + d / 20 + d / 25 = 15 / 60) → (5 * d = 375 / 137) :=
by
  intro h
  -- proof will go here
  sorry

end total_distance_covered_l78_78307


namespace Charles_chocolate_milk_total_l78_78625

theorem Charles_chocolate_milk_total (milk_per_glass syrup_per_glass total_milk total_syrup : ℝ) 
(h_milk_glass : milk_per_glass = 6.5) (h_syrup_glass : syrup_per_glass = 1.5) (h_total_milk : total_milk = 130) (h_total_syrup : total_syrup = 60) :
  (min (total_milk / milk_per_glass) (total_syrup / syrup_per_glass) * (milk_per_glass + syrup_per_glass) = 160) :=
by
  sorry

end Charles_chocolate_milk_total_l78_78625


namespace replace_asterisks_l78_78912

theorem replace_asterisks (x : ℕ) (h : (x / 20) * (x / 180) = 1) : x = 60 := by
  sorry

end replace_asterisks_l78_78912


namespace test_question_count_l78_78744

theorem test_question_count :
  ∃ (x y : ℕ), x + y = 30 ∧ 5 * x + 10 * y = 200 ∧ x = 20 :=
by
  sorry

end test_question_count_l78_78744


namespace possible_r_values_l78_78615

noncomputable def triangle_area (r : ℝ) : ℝ := (r - 3) ^ (3 / 2)

theorem possible_r_values :
  {r : ℝ | 16 ≤ triangle_area r ∧ triangle_area r ≤ 128} = {r : ℝ | 7 ≤ r ∧ r ≤ 19} :=
by
  sorry

end possible_r_values_l78_78615


namespace find_some_number_eq_0_3_l78_78638

theorem find_some_number_eq_0_3 (X : ℝ) (h : 2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002) :
  X = 0.3 :=
by sorry

end find_some_number_eq_0_3_l78_78638


namespace smallest_number_with_2020_divisors_l78_78345

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l78_78345


namespace tree_ratio_l78_78232

theorem tree_ratio (A P C : ℕ) 
  (hA : A = 58)
  (hP : P = 3 * A)
  (hC : C = 5 * P) : (A, P, C) = (1, 3 * 58, 15 * 58) :=
by
  sorry

end tree_ratio_l78_78232


namespace distance_between_parallel_lines_l78_78134

-- Definitions of lines l_1 and l_2
def line_l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line_l2 (x y : ℝ) : Prop := 6*x + 8*y - 5 = 0

-- Proof statement that the distance between the two lines is 1/10
theorem distance_between_parallel_lines (x y : ℝ) :
  ∃ d : ℝ, d = 1/10 ∧ ∀ p : ℝ × ℝ,
  (line_l1 p.1 p.2 ∧ line_l2 p.1 p.2 → p = (x, y)) :=
sorry

end distance_between_parallel_lines_l78_78134


namespace least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l78_78292

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_primes_greater_than_five : List ℕ :=
  [7, 11, 13]

theorem least_positive_integer_divisible_by_three_smallest_primes_greater_than_five : 
  ∃ n : ℕ, n > 0 ∧ (∀ p ∈ smallest_primes_greater_than_five, p ∣ n) ∧ n = 1001 := by
  sorry

end least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l78_78292


namespace maximum_ab_ac_bc_l78_78251

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end maximum_ab_ac_bc_l78_78251


namespace average_and_variance_of_original_data_l78_78308

theorem average_and_variance_of_original_data (μ σ_sq : ℝ)
  (h1 : 2 * μ - 80 = 1.2)
  (h2 : 4 * σ_sq = 4.4) :
  μ = 40.6 ∧ σ_sq = 1.1 :=
by
  sorry

end average_and_variance_of_original_data_l78_78308


namespace solve_for_x_l78_78844

theorem solve_for_x : ∃ x : ℝ, 4 * x + 6 * x = 360 - 9 * (x - 4) ∧ x = 396 / 19 :=
by
  sorry

end solve_for_x_l78_78844


namespace domain_of_function_l78_78212

theorem domain_of_function (f : ℝ → ℝ) (h₀ : Set.Ioo 0 1 ⊆ {x | f (3 * x + 2)}) :
  Set.Ioo (3 / 2) 3 ⊆ {x | f (2 * x - 1)} :=
by
  sorry

end domain_of_function_l78_78212


namespace stock_exchange_total_l78_78162

theorem stock_exchange_total (L H : ℕ) 
  (h1 : H = 1080) 
  (h2 : H = 6 * L / 5) : 
  (L + H = 1980) :=
by {
  -- L and H are given as natural numbers
  -- h1: H = 1080
  -- h2: H = 1.20L -> H = 6L/5 as Lean does not handle floating point well directly in integers.
  sorry
}

end stock_exchange_total_l78_78162


namespace y_intercept_of_line_l78_78327

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end y_intercept_of_line_l78_78327


namespace sam_wins_probability_l78_78837

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l78_78837


namespace bees_flew_in_l78_78568

theorem bees_flew_in (initial_bees : ℕ) (total_bees : ℕ) (new_bees : ℕ) (h1 : initial_bees = 16) (h2 : total_bees = 23) (h3 : total_bees = initial_bees + new_bees) : new_bees = 7 :=
by
  sorry

end bees_flew_in_l78_78568


namespace covering_percentage_77_l78_78742

-- Definition section for conditions
def radius_of_circle (r a : ℝ) := 2 * r * Real.pi = 4 * a
def center_coincide (a b : ℝ) := a = b

-- Theorem to be proven
theorem covering_percentage_77
  (r a : ℝ)
  (h_radius: radius_of_circle r a)
  (h_center: center_coincide 0 0) : 
  (r^2 * Real.pi - 0.7248 * r^2) / (r^2 * Real.pi) * 100 = 77 := by
  sorry

end covering_percentage_77_l78_78742


namespace sum_of_geometric_sequence_l78_78059

noncomputable def geometric_sequence_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence
  (a_1 q : ℝ) 
  (h1 : a_1^2 * q^6 = 2 * a_1 * q^2)
  (h2 : (a_1 * q^3 + 2 * a_1 * q^6) / 2 = 5 / 4)
  : geometric_sequence_sum a_1 q 4 = 30 :=
by
  sorry

end sum_of_geometric_sequence_l78_78059


namespace average_after_modifications_l78_78427

theorem average_after_modifications (S : ℕ) (sum_initial : S = 1080)
  (sum_after_removals : S - 80 - 85 = 915)
  (sum_after_additions : 915 + 75 + 75 = 1065) :
  (1065 / 12 : ℚ) = 88.75 :=
by sorry

end average_after_modifications_l78_78427


namespace distance_from_point_to_focus_l78_78505

noncomputable def point_on_parabola (P : ℝ × ℝ) (y : ℝ) : Prop :=
  y^2 = 16 * P.1 ∧ (P.2 = y ∨ P.2 = -y)

noncomputable def parabola_focus : ℝ × ℝ :=
  (4, 0)

theorem distance_from_point_to_focus
  (P : ℝ × ℝ) (y : ℝ)
  (h1 : point_on_parabola P y)
  (h2 : dist P (0, P.2) = 12) :
  dist P parabola_focus = 13 :=
sorry

end distance_from_point_to_focus_l78_78505


namespace arctan_sum_eq_pi_over_4_l78_78003

theorem arctan_sum_eq_pi_over_4 : 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/47) = Real.pi / 4 :=
by
  sorry

end arctan_sum_eq_pi_over_4_l78_78003


namespace simplify_fraction_l78_78698

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end simplify_fraction_l78_78698


namespace geometric_sequence_seventh_term_l78_78271

theorem geometric_sequence_seventh_term (a r : ℝ) 
  (h4 : a * r^3 = 16) 
  (h9 : a * r^8 = 2) : 
  a * r^6 = 8 := 
sorry

end geometric_sequence_seventh_term_l78_78271


namespace find_x_tan_eq_l78_78014

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l78_78014


namespace problem_statement_l78_78954

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

theorem problem_statement (a x₁ x₂: ℝ) (ha : a > 0) (hx : x₁ < x₂) (hxsum : x₁ + x₂ = 0) :
  f a x₁ < f a x₂ := by
  sorry

end problem_statement_l78_78954


namespace parallelogram_sides_l78_78319

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 3 * x + 6 = 15) 
  (h2 : 10 * y - 2 = 12) :
  x + y = 4.4 := 
sorry

end parallelogram_sides_l78_78319


namespace remainder_when_divided_by_10_l78_78154

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end remainder_when_divided_by_10_l78_78154


namespace pastries_average_per_day_l78_78301

theorem pastries_average_per_day :
  let monday_sales := 2
  let tuesday_sales := monday_sales + 1
  let wednesday_sales := tuesday_sales + 1
  let thursday_sales := wednesday_sales + 1
  let friday_sales := thursday_sales + 1
  let saturday_sales := friday_sales + 1
  let sunday_sales := saturday_sales + 1
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
  let days := 7
  total_sales / days = 5 := by
  sorry

end pastries_average_per_day_l78_78301


namespace arithmetic_mean_is_correct_l78_78152

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l78_78152


namespace probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l78_78919

def frequencies_branch_a := (40, 20, 20, 20) -- (A, B, C, D)
def frequencies_branch_b := (28, 17, 34, 21) -- (A, B, C, D)

def fees := (90, 50, 20, -50)  -- (A, B, C, D respectively)
def processing_cost_branch_a := 25
def processing_cost_branch_b := 20

theorem probability_grade_A_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  (fa : ℝ) / 100 = 0.4 := by
  sorry

theorem probability_grade_A_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  (fa : ℝ) / 100 = 0.28 := by
  sorry

theorem average_profit_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_a) * (fa / 100) + 
   (qb - processing_cost_branch_a) * (fb / 100) +
   (qc - processing_cost_branch_a) * (fc / 100) +
   (qd - processing_cost_branch_a) * (fd / 100) : ℝ) = 15 := by
  sorry

theorem average_profit_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_b) * (fa / 100) + 
   (qb - processing_cost_branch_b) * (fb / 100) +
   (qc - processing_cost_branch_b) * (fc / 100) +
   (qd - processing_cost_branch_b) * (fd / 100) : ℝ) = 10 := by
  sorry

theorem select_branch :
  let profit_a := 15 in
  let profit_b := 10 in
  profit_a > profit_b → 
  "Branch A" = "Branch A" := by
  sorry

end probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l78_78919


namespace trajectory_of_centroid_l78_78264

def foci (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (0, 1) ∧ F2 = (0, -1)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 3) + (P.2^2 / 4) = 1

def centroid_eq (G : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, on_ellipse P ∧ 
  foci (0, 1) (0, -1) ∧ 
  G = (P.1 / 3, (1 + -1 + P.2) / 3)

theorem trajectory_of_centroid :
  ∀ G : ℝ × ℝ, (centroid_eq G → 3 * G.1^2 + (9 * G.2^2) / 4 = 1 ∧ G.1 ≠ 0) :=
by 
  intros G h
  sorry

end trajectory_of_centroid_l78_78264


namespace face_value_of_shares_l78_78173

theorem face_value_of_shares (investment : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (dividend_received : ℝ) (F : ℝ)
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_received = 720) :
  (1.20 * F = investment) ∧ (0.06 * F = dividend_received) ∧ (F = 12000) :=
by
  sorry

end face_value_of_shares_l78_78173


namespace tanya_bought_six_plums_l78_78559

theorem tanya_bought_six_plums (pears apples pineapples pieces_left : ℕ) 
  (h_pears : pears = 6) (h_apples : apples = 4) (h_pineapples : pineapples = 2) 
  (h_pieces_left : pieces_left = 9) (h_half_fell : pieces_left * 2 = total_fruit) :
  pears + apples + pineapples < total_fruit ∧ total_fruit - (pears + apples + pineapples) = 6 :=
by
  sorry

end tanya_bought_six_plums_l78_78559


namespace f_inv_f_inv_17_l78_78126

noncomputable def f (x : ℝ) : ℝ := 4 * x - 3

noncomputable def f_inv (y : ℝ) : ℝ := (y + 3) / 4

theorem f_inv_f_inv_17 : f_inv (f_inv 17) = 2 := by
  sorry

end f_inv_f_inv_17_l78_78126


namespace sqrt_pos_condition_l78_78287

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l78_78287


namespace age_difference_l78_78734

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 :=
by
  sorry

end age_difference_l78_78734


namespace Walter_gets_49_bananas_l78_78680

variable (Jefferson_bananas : ℕ) (Walter_bananas : ℕ) (total_bananas : ℕ) (shared_bananas : ℕ)

def problem_conditions : Prop :=
  Jefferson_bananas = 56 ∧ Walter_bananas = Jefferson_bananas - (Jefferson_bananas / 4)

theorem Walter_gets_49_bananas (h : problem_conditions) : 
  let combined_bananas := Jefferson_bananas + Walter_bananas in
  let shared_bananas := combined_bananas / 2 in
  shared_bananas = 49 :=
by
  sorry

end Walter_gets_49_bananas_l78_78680


namespace find_x_l78_78607

theorem find_x :
  ∃ x : ℝ, x = (1/x) * (-x) - 3*x + 4 ∧ x = 3/4 :=
by
  sorry

end find_x_l78_78607


namespace smallest_n_with_2020_divisors_l78_78343

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l78_78343


namespace overlap_per_connection_is_4_cm_l78_78451

-- Condition 1: There are 24 tape measures.
def number_of_tape_measures : Nat := 24

-- Condition 2: Each tape measure is 28 cm long.
def length_of_one_tape_measure : Nat := 28

-- Condition 3: The total length of all connected tape measures is 580 cm.
def total_length_with_overlaps : Nat := 580

-- The question to prove: The overlap per connection is 4 cm.
theorem overlap_per_connection_is_4_cm 
  (n : Nat) (length_one : Nat) (total_length : Nat) 
  (h_n : n = number_of_tape_measures)
  (h_length_one : length_one = length_of_one_tape_measure)
  (h_total_length : total_length = total_length_with_overlaps) :
  ((n * length_one - total_length) / (n - 1)) = 4 := 
by 
  sorry

end overlap_per_connection_is_4_cm_l78_78451


namespace JordanRectangleWidth_l78_78910

/-- Given that Carol's rectangle measures 15 inches by 24 inches,
and Jordan's rectangle is 8 inches long with equal area as Carol's rectangle,
prove that Jordan's rectangle is 45 inches wide. -/
theorem JordanRectangleWidth :
  ∃ W : ℝ, (15 * 24 = 8 * W) → W = 45 := by
  sorry

end JordanRectangleWidth_l78_78910


namespace find_room_dimension_l78_78430

noncomputable def unknown_dimension_of_room 
  (cost_per_sq_ft : ℕ)
  (total_cost : ℕ)
  (w : ℕ)
  (l : ℕ)
  (h : ℕ)
  (door_h : ℕ)
  (door_w : ℕ)
  (window_h : ℕ)
  (window_w : ℕ)
  (num_windows : ℕ) : ℕ := sorry

theorem find_room_dimension :
  unknown_dimension_of_room 10 9060 25 15 12 6 3 4 3 3 = 25 :=
sorry

end find_room_dimension_l78_78430


namespace find_divisor_l78_78582

variable (n : ℤ) (d : ℤ)

theorem find_divisor 
    (h1 : ∃ k : ℤ, n = k * d + 4)
    (h2 : ∃ m : ℤ, n + 15 = m * 5 + 4) :
    d = 5 :=
sorry

end find_divisor_l78_78582


namespace fraction_of_raisins_in_mixture_l78_78747

def cost_of_raisins (R : ℝ) := 3 * R
def cost_of_nuts (R : ℝ) := 3 * (3 * R)
def total_cost (R : ℝ) := cost_of_raisins R + cost_of_nuts R

theorem fraction_of_raisins_in_mixture (R : ℝ) (hR_pos : R > 0) : 
  cost_of_raisins R / total_cost R = 1 / 4 :=
by
  sorry

end fraction_of_raisins_in_mixture_l78_78747


namespace sailboat_rental_cost_l78_78671

-- Define the conditions
def rental_per_hour_ski := 80
def hours_per_day := 3
def days := 2
def cost_ski := (hours_per_day * days * rental_per_hour_ski)
def additional_cost := 120

-- Statement to prove
theorem sailboat_rental_cost :
  ∃ (S : ℕ), cost_ski = S + additional_cost → S = 360 := by
  sorry

end sailboat_rental_cost_l78_78671


namespace largest_m_for_negative_integral_solutions_l78_78575

theorem largest_m_for_negative_integral_solutions :
  ∃ m : ℕ, (∀ p q : ℤ, 10 * p * p + (-m) * p + 560 = 0 ∧ p < 0 ∧ q < 0 ∧ p * q = 56 → m ≤ 570) ∧ m = 570 :=
sorry

end largest_m_for_negative_integral_solutions_l78_78575


namespace evaluate_expression_at_neg_two_l78_78418

noncomputable def complex_expression (a : ℝ) : ℝ :=
  (1 - (a / (a + 1))) / (1 / (1 - a^2))

theorem evaluate_expression_at_neg_two :
  complex_expression (-2) = sorry :=
sorry

end evaluate_expression_at_neg_two_l78_78418


namespace basketball_game_points_l78_78227

theorem basketball_game_points
  (a b : ℕ) 
  (r : ℕ := 2)
  (S_E : ℕ := a / 2 * (1 + r + r^2 + r^3))
  (S_T : ℕ := 4 * b)
  (h1 : S_E = S_T + 2)
  (h2 : S_E < 100)
  (h3 : S_T < 100)
  : (a / 2 + a / 2 * r + b + b = 19) :=
by sorry

end basketball_game_points_l78_78227


namespace anne_speed_ratio_l78_78318

variable (B A A' : ℝ)

theorem anne_speed_ratio (h1 : A = 1 / 12)
                        (h2 : B + A = 1 / 4)
                        (h3 : B + A' = 1 / 3) : 
                        A' / A = 2 := 
by
  -- Proof is omitted
  sorry

end anne_speed_ratio_l78_78318


namespace solve_equation_1_solve_equation_2_solve_equation_3_l78_78424

theorem solve_equation_1 : ∀ x : ℝ, (4 * (x + 3) = 25) ↔ (x = 13 / 4) :=
by
  sorry

theorem solve_equation_2 : ∀ x : ℝ, (5 * x^2 - 3 * x = x + 1) ↔ (x = -1 / 5 ∨ x = 1) :=
by
  sorry

theorem solve_equation_3 : ∀ x : ℝ, (2 * (x - 2)^2 - (x - 2) = 0) ↔ (x = 2 ∨ x = 5 / 2) :=
by
  sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l78_78424


namespace smallest_number_with_2020_divisors_l78_78337

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l78_78337


namespace repeating_decimal_block_length_l78_78855

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l78_78855


namespace ways_to_place_balls_with_exactly_two_matches_l78_78121

open Equiv

theorem ways_to_place_balls_with_exactly_two_matches :
  (∑ s in (Finset.univ : Finset (Fin 5).powerset), (s.card = 2 : Prop) ->
     (∃ f : (Fin 5) → (Fin 5), (∀ i ∈ s, f i = i) ∧ (∀ i ∉ s, f i ≠ i)) * fintype.card {f | ((∀ i, f i = i) : Prop) ∧ 
     (∀ i, ∃ j, i ≠ j → f i ≠ i) }) = 20 := sorry

end ways_to_place_balls_with_exactly_two_matches_l78_78121


namespace ab_operation_l78_78716

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l78_78716


namespace age_intervals_l78_78142

theorem age_intervals (A1 A2 A3 A4 A5 : ℝ) (x : ℝ) (h1 : A1 = 7)
  (h2 : A2 = A1 + x) (h3 : A3 = A1 + 2 * x) (h4 : A4 = A1 + 3 * x) (h5 : A5 = A1 + 4 * x)
  (sum_ages : A1 + A2 + A3 + A4 + A5 = 65) :
  x = 3.7 :=
by
  -- Sketch a proof or leave 'sorry' for completeness
  sorry

end age_intervals_l78_78142


namespace peanuts_added_l78_78569

theorem peanuts_added (a b x : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : a + x = b) : x = 2 :=
by
  sorry

end peanuts_added_l78_78569


namespace player_1_points_after_13_rotations_l78_78881

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l78_78881


namespace find_x_l78_78021

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l78_78021


namespace min_tiles_l78_78556

theorem min_tiles (x y : ℕ) (h1 : 25 * x + 9 * y = 2014) (h2 : ∀ a b, 25 * a + 9 * b = 2014 -> (a + b) >= (x + y)) : x + y = 94 :=
  sorry

end min_tiles_l78_78556


namespace IncorrectOption_l78_78815

namespace Experiment

def OptionA : Prop := 
  ∃ method : String, method = "sampling detection"

def OptionB : Prop := 
  ¬(∃ experiment : String, experiment = "does not need a control group, nor repeated experiments")

def OptionC : Prop := 
  ∃ action : String, action = "test tube should be gently shaken"

def OptionD : Prop := 
  ∃ condition : String, condition = "field of view should not be too bright"

theorem IncorrectOption : OptionB :=
  sorry

end Experiment

end IncorrectOption_l78_78815


namespace Ava_watched_television_for_240_minutes_l78_78481

-- Define the conditions
def hours (h : ℕ) := h = 4

-- Define the conversion factor from hours to minutes
def convert_hours_to_minutes (h : ℕ) : ℕ := h * 60

-- State the theorem
theorem Ava_watched_television_for_240_minutes (h : ℕ) (hh : hours h) : convert_hours_to_minutes h = 240 :=
by
  -- The proof goes here but is skipped
  sorry

end Ava_watched_television_for_240_minutes_l78_78481


namespace largest_possible_radius_tangent_circle_l78_78633

theorem largest_possible_radius_tangent_circle :
  ∃ (r : ℝ), 0 < r ∧
    (∀ x y, (x - r)^2 + (y - r)^2 = r^2 → 
    ((x = 9 ∧ y = 2) → (r = 17))) :=
by
  sorry

end largest_possible_radius_tangent_circle_l78_78633


namespace misha_contributes_l78_78536

noncomputable def misha_contribution (k l m : ℕ) : ℕ :=
  if h : k + l + m = 6 ∧ 2 * k ≤ l + m ∧ 2 * l ≤ k + m ∧ 2 * m ≤ k + l ∧ k ≤ 2 ∧ l ≤ 2 ∧ m ≤ 2 then
    2
  else
    0 -- This is a default value; the actual proof will check for exact solution.

theorem misha_contributes (k l m : ℕ) (h1 : k + l + m = 6)
    (h2 : 2 * k ≤ l + m) (h3 : 2 * l ≤ k + m) (h4 : 2 * m ≤ k + l)
    (h5 : k ≤ 2) (h6 : l ≤ 2) (h7 : m ≤ 2) : m = 2 := by
  sorry

end misha_contributes_l78_78536


namespace find_x_l78_78035

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l78_78035


namespace find_original_number_l78_78824

-- Let x be the original number
def maria_operations (x : ℤ) : Prop :=
  (3 * (x - 3) + 3) / 3 = 10

theorem find_original_number (x : ℤ) (h : maria_operations x) : x = 12 :=
by
  sorry

end find_original_number_l78_78824


namespace all_numbers_equal_l78_78736

theorem all_numbers_equal (x : Fin 101 → ℝ) 
  (h : ∀ i : Fin 100, x i.val^3 + x ⟨(i.val + 1) % 101, sorry⟩ = (x ⟨(i.val + 1) % 101, sorry⟩)^3 + x ⟨(i.val + 2) % 101, sorry⟩) :
  ∀ i j : Fin 101, x i = x j := 
by 
  sorry

end all_numbers_equal_l78_78736


namespace arithmetic_mean_of_two_digit_multiples_of_9_l78_78150

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l78_78150


namespace average_price_of_fruit_l78_78622

theorem average_price_of_fruit :
  ∃ (A O : ℕ), A + O = 10 ∧ (40 * A + 60 * (O - 4)) / (A + O - 4) = 50 → 
  (40 * A + 60 * O) / 10 = 54 :=
by
  sorry

end average_price_of_fruit_l78_78622


namespace factorial_last_nonzero_digit_non_periodic_l78_78255

def last_nonzero_digit (n : ℕ) : ℕ :=
  -- function to compute last nonzero digit of n!
  sorry

def sequence_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a n = a (n + T)

theorem factorial_last_nonzero_digit_non_periodic : ¬ ∃ T, sequence_periodic last_nonzero_digit T :=
  sorry

end factorial_last_nonzero_digit_non_periodic_l78_78255


namespace find_a_if_f_is_even_l78_78272

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 2 := by
  sorry

end find_a_if_f_is_even_l78_78272


namespace ratio_of_areas_l78_78274

theorem ratio_of_areas (s : ℝ) 
  (h1 : ∀ (s : ℝ), s > 0) : 
  let R_long := 1.2 * s,
      R_short := 0.8 * s,
      area_R := R_long * R_short,
      area_S := s^2
  in area_R / area_S = 24 / 25 :=
by
  let R_long := 1.2 * s
  let R_short := 0.8 * s
  let area_R := R_long * R_short
  let area_S := s^2
  have h2 : s > 0 := h1 s
  have h3 : area_R = 0.96 * s^2 := by sorry
  have h4 : area_R / area_S = 0.96 := by sorry
  have h5 : 0.96 = 24 / 25 := by norm_num
  exact eq.trans h4 h5

end ratio_of_areas_l78_78274


namespace tim_prank_combinations_l78_78570

def number_of_combinations : Nat :=
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations = 60 :=
by
  sorry

end tim_prank_combinations_l78_78570


namespace minnie_lucy_time_difference_is_66_minutes_l78_78118

noncomputable def minnie_time_uphill : ℚ := 12 / 6
noncomputable def minnie_time_downhill : ℚ := 18 / 25
noncomputable def minnie_time_flat : ℚ := 15 / 15

noncomputable def minnie_total_time : ℚ := minnie_time_uphill + minnie_time_downhill + minnie_time_flat

noncomputable def lucy_time_flat : ℚ := 15 / 25
noncomputable def lucy_time_uphill : ℚ := 12 / 8
noncomputable def lucy_time_downhill : ℚ := 18 / 35

noncomputable def lucy_total_time : ℚ := lucy_time_flat + lucy_time_uphill + lucy_time_downhill

-- Convert hours to minutes
noncomputable def minnie_total_time_minutes : ℚ := minnie_total_time * 60
noncomputable def lucy_total_time_minutes : ℚ := lucy_total_time * 60

-- Difference in minutes
noncomputable def time_difference : ℚ := minnie_total_time_minutes - lucy_total_time_minutes

theorem minnie_lucy_time_difference_is_66_minutes : time_difference = 66 := by
  sorry

end minnie_lucy_time_difference_is_66_minutes_l78_78118


namespace count_rectangles_5x5_l78_78983

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l78_78983


namespace smallest_sum_of_two_perfect_squares_l78_78429

theorem smallest_sum_of_two_perfect_squares (x y : ℕ) (h : x^2 - y^2 = 143) :
  x + y = 13 ∧ x - y = 11 → x^2 + y^2 = 145 :=
by
  -- Add this placeholder "sorry" to skip the proof, as required.
  sorry

end smallest_sum_of_two_perfect_squares_l78_78429


namespace solution_set_eq_l78_78072

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def decreasing_condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x1 * f (x1) - x2 * f (x2)) / (x1 - x2) < 0

variable (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_minus_2_zero : f (-2) = 0)
variable (h_decreasing : decreasing_condition f)

theorem solution_set_eq :
  {x : ℝ | f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_eq_l78_78072


namespace value_of_a_l78_78805

theorem value_of_a (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 :=
by
  sorry

end value_of_a_l78_78805


namespace James_distance_ridden_l78_78103

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end James_distance_ridden_l78_78103


namespace megan_popsicles_consumed_l78_78825

noncomputable def popsicles_consumed_in_time_period (time: ℕ) (interval: ℕ) : ℕ :=
  (time / interval)

theorem megan_popsicles_consumed:
  popsicles_consumed_in_time_period 315 30 = 10 :=
by
  sorry

end megan_popsicles_consumed_l78_78825


namespace students_from_second_grade_l78_78435

theorem students_from_second_grade (r1 r2 r3 : ℕ) (total_students sample_size : ℕ) (h_ratio: r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ r1 + r2 + r3 = 10) (h_sample_size: sample_size = 50) : 
  (r2 * sample_size / (r1 + r2 + r3)) = 15 :=
by
  sorry

end students_from_second_grade_l78_78435


namespace dan_marbles_l78_78192

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) : 
  original_marbles = 64 ∧ given_marbles = 14 → remaining_marbles = 50 := 
by 
  sorry

end dan_marbles_l78_78192


namespace repeating_block_length_of_7_div_13_is_6_l78_78853

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l78_78853


namespace six_coins_heads_or_tails_probability_l78_78420

theorem six_coins_heads_or_tails_probability :
  let total_outcomes := 2^6 in
  let favorable_outcomes := 2 in
  favorable_outcomes / total_outcomes = (1 : ℚ) / 32 :=
by
  sorry

end six_coins_heads_or_tails_probability_l78_78420


namespace find_n_l78_78210

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_lcm1 : Nat.lcm 40 n = 120) (h_lcm2 : Nat.lcm n 45 = 180) : n = 12 :=
sorry

end find_n_l78_78210


namespace smallest_pos_int_terminating_decimal_with_9_l78_78450

theorem smallest_pos_int_terminating_decimal_with_9 : ∃ n : ℕ, (∃ m k : ℕ, n = 2^m * 5^k ∧ (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9)) ∧ n = 4096 :=
by {
    sorry
}

end smallest_pos_int_terminating_decimal_with_9_l78_78450


namespace manufacturers_price_l78_78925

theorem manufacturers_price (M : ℝ) 
  (h1 : 0.1 ≤ 0.3) 
  (h2 : 0.2 = 0.2) 
  (h3 : 0.56 * M = 25.2) : 
  M = 45 := 
sorry

end manufacturers_price_l78_78925


namespace factor_expression_l78_78188

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end factor_expression_l78_78188


namespace northern_village_population_l78_78813

theorem northern_village_population
    (x : ℕ) -- Northern village population
    (western_village_population : ℕ := 400)
    (southern_village_population : ℕ := 200)
    (total_conscripted : ℕ := 60)
    (northern_village_conscripted : ℕ := 10)
    (h : (northern_village_conscripted : ℚ) / total_conscripted = (x : ℚ) / (x + western_village_population + southern_village_population)) : 
    x = 120 :=
    sorry

end northern_village_population_l78_78813


namespace maximum_possible_value_of_x_l78_78560

-- Define the conditions and the question
def ten_teams_playing_each_other_once (number_of_teams : ℕ) : Prop :=
  number_of_teams = 10

def points_system (win_points draw_points loss_points : ℕ) : Prop :=
  win_points = 3 ∧ draw_points = 1 ∧ loss_points = 0

def max_points_per_team (x : ℕ) : Prop :=
  x = 13

-- The theorem to be proved: maximum possible value of x given the conditions
theorem maximum_possible_value_of_x :
  ∀ (number_of_teams win_points draw_points loss_points x : ℕ),
    ten_teams_playing_each_other_once number_of_teams →
    points_system win_points draw_points loss_points →
    max_points_per_team x :=
  sorry

end maximum_possible_value_of_x_l78_78560


namespace largest_4_digit_number_divisible_by_1615_l78_78498

theorem largest_4_digit_number_divisible_by_1615 (X : ℕ) (hX: 8640 = 1615 * X) (h1: 1000 ≤ 1615 * X ∧ 1615 * X ≤ 9999) : X = 5 :=
by
  sorry

end largest_4_digit_number_divisible_by_1615_l78_78498


namespace player_1_points_after_13_rotations_l78_78882

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l78_78882


namespace inequality_lemma_l78_78554

-- Define the conditions: x and y are positive numbers and x > y
variables (x y : ℝ)
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y)

-- State the theorem to be proved
theorem inequality_lemma : 2 * x + 1 / (x^2 - 2*x*y + y^2) >= 2 * y + 3 :=
by
  sorry

end inequality_lemma_l78_78554


namespace smallest_integer_in_range_l78_78280

-- Given conditions
def is_congruent_6 (n : ℕ) : Prop := n % 6 = 1
def is_congruent_7 (n : ℕ) : Prop := n % 7 = 1
def is_congruent_8 (n : ℕ) : Prop := n % 8 = 1

-- Lean statement for the proof problem
theorem smallest_integer_in_range :
  ∃ n : ℕ, (n > 1) ∧ is_congruent_6 n ∧ is_congruent_7 n ∧ is_congruent_8 n ∧ (n = 169) ∧ (120 ≤ n ∧ n < 210) :=
by
  sorry

end smallest_integer_in_range_l78_78280


namespace smallest_number_with_2020_divisors_l78_78336

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l78_78336


namespace smallest_number_with_2020_divisors_is_correct_l78_78348

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α_1 := 100
  let α_2 := 4
  let α_3 := 1
  let α_4 := 1
  let n := 2 ^ α_1 * 3 ^ α_2 * 5 ^ α_3 * 7 ^ α_4
  n

theorem smallest_number_with_2020_divisors_is_correct :
  let n := smallest_number_with_2020_divisors in
  let τ (n : ℕ) : ℕ :=
    (n.factors.nodup.erase 2).foldr (λ p acc, (n.factors.count p + 1) * acc) 1 in
  τ n = 2020 ↔ n = 2 ^ 100 * 3 ^ 4 * 5 * 7 :=
by
  sorry

end smallest_number_with_2020_divisors_is_correct_l78_78348


namespace find_a4_l78_78366

noncomputable def quadratic_eq (t : ℝ) := t^2 - 36 * t + 288 = 0

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∃ a1 : ℝ, a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def condition1 (a : ℕ → ℝ) := a 1 + a 2 = -1
def condition2 (a : ℕ → ℝ) := a 1 - a 3 = -3

theorem find_a4 :
  ∃ (a : ℕ → ℝ) (q : ℝ), quadratic_eq q ∧ geometric_sequence a q ∧ condition1 a ∧ condition2 a ∧ a 4 = -8 :=
by
  sorry

end find_a4_l78_78366


namespace find_x_l78_78022

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l78_78022


namespace sqrt_meaningful_condition_l78_78282

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l78_78282


namespace probability_at_most_one_correct_in_two_rounds_l78_78145

theorem probability_at_most_one_correct_in_two_rounds :
  let pA := 3 / 5
  let pB := 2 / 3
  let pA_incorrect := 1 - pA
  let pB_incorrect := 1 - pB
  let p_0_correct := pA_incorrect * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A1 := pA * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A2 := pA_incorrect * pA * pB_incorrect * pB_incorrect
  let p_1_correct_B1 := pA_incorrect * pA_incorrect * pB * pB_incorrect
  let p_1_correct_B2 := pA_incorrect * pA_incorrect * pB_incorrect * pB
  let p_at_most_one := p_0_correct + p_1_correct_A1 + p_1_correct_A2 + 
      p_1_correct_B1 + p_1_correct_B2
  p_at_most_one = 32 / 225 := 
  sorry

end probability_at_most_one_correct_in_two_rounds_l78_78145


namespace restore_example_l78_78378

theorem restore_example (x : ℕ) (y : ℕ) :
  (10 ≤ x * 8 ∧ x * 8 < 100) ∧ (100 ≤ x * 9 ∧ x * 9 < 1000) ∧ y = 98 → x = 12 ∧ x * y = 1176 :=
by
  sorry

end restore_example_l78_78378


namespace probability_of_sum_ge_5_l78_78353

def balls : Finset ℕ := {1, 2, 3, 4}

def selected_balls := {pair ∈ balls.powerset.filter (λ s, s.card = 2) | (s : ℕ).sum ≥ 5}

theorem probability_of_sum_ge_5 :
  (selected_balls.card : ℚ) / (balls.powerset.filter (λ s, s.card = 2)).card = 2 / 3 := 
by 
  -- Add proof here 
  sorry

end probability_of_sum_ge_5_l78_78353


namespace total_amount_spent_l78_78907

variable (your_spending : ℝ) (friend_spending : ℝ)
variable (h1 : friend_spending = your_spending + 3) (h2 : friend_spending = 10)

theorem total_amount_spent : your_spending + friend_spending = 17 :=
by sorry

end total_amount_spent_l78_78907


namespace intersection_points_lie_on_ellipse_l78_78639

theorem intersection_points_lie_on_ellipse (s : ℝ) : 
  ∃ (x y : ℝ), (2 * s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) ∧ (x^2 / 16 + y^2 / 9 = 1) :=
sorry

end intersection_points_lie_on_ellipse_l78_78639


namespace sin_780_eq_sqrt3_over_2_l78_78766

theorem sin_780_eq_sqrt3_over_2 :
  sin (780 : ℝ) = (Real.sqrt 3 / 2) :=
by
  sorry

end sin_780_eq_sqrt3_over_2_l78_78766


namespace find_dividend_l78_78897

def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 16) (h_quotient : quotient = 8) (h_remainder : remainder = 4) :
  dividend divisor quotient remainder = 132 :=
by
  sorry

end find_dividend_l78_78897


namespace walkway_area_l78_78128

/--
Tara has four rows of three 8-feet by 3-feet flower beds in her garden. The beds are separated
and surrounded by 2-feet-wide walkways. Prove that the total area of the walkways is 416 square feet.
-/
theorem walkway_area :
  let flower_bed_width := 8
  let flower_bed_height := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_width := (num_columns * flower_bed_width) + (num_columns + 1) * walkway_width
  let total_height := (num_rows * flower_bed_height) + (num_rows + 1) * walkway_width
  let total_garden_area := total_width * total_height
  let flower_bed_area := flower_bed_width * flower_bed_height * num_rows * num_columns
  total_garden_area - flower_bed_area = 416 :=
by
  -- Proof omitted
  sorry

end walkway_area_l78_78128


namespace smallest_solution_correct_l78_78780

noncomputable def smallest_solution (x : ℝ) : ℝ :=
if (⌊ x^2 ⌋ - ⌊ x ⌋^2 = 17) then x else 0

theorem smallest_solution_correct :
  smallest_solution (7 * Real.sqrt 2) = 7 * Real.sqrt 2 :=
by sorry

end smallest_solution_correct_l78_78780


namespace sarah_friends_apples_l78_78697

-- Definitions of initial conditions
def initial_apples : ℕ := 25
def left_apples : ℕ := 3
def apples_given_teachers : ℕ := 16
def apples_eaten : ℕ := 1

-- Theorem that states the number of friends who received apples
theorem sarah_friends_apples :
  (initial_apples - left_apples - apples_given_teachers - apples_eaten = 5) :=
by
  sorry

end sarah_friends_apples_l78_78697


namespace percentage_error_calc_l78_78611

theorem percentage_error_calc (x : ℝ) (h : x ≠ 0) : 
  let correct_result := x * (5 / 3)
  let incorrect_result := x * (3 / 5)
  let percentage_error := (correct_result - incorrect_result) / correct_result * 100
  percentage_error = 64 := by
  sorry

end percentage_error_calc_l78_78611


namespace exists_not_odd_l78_78662

variable (f : ℝ → ℝ)

-- Define the condition that f is not an odd function
def not_odd_function := ¬ (∀ x : ℝ, f (-x) = -f x)

-- Lean statement to prove the correct answer
theorem exists_not_odd (h : not_odd_function f) : ∃ x : ℝ, f (-x) ≠ -f x :=
sorry

end exists_not_odd_l78_78662


namespace cone_water_volume_percentage_l78_78305

theorem cone_water_volume_percentage
  (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  abs (percentage - 29.6296) < 0.0001 :=
by
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  sorry

end cone_water_volume_percentage_l78_78305


namespace deleted_files_l78_78691

variable {initial_files : ℕ}
variable {files_per_folder : ℕ}
variable {folders : ℕ}

noncomputable def files_deleted (initial_files files_in_folders : ℕ) : ℕ :=
  initial_files - files_in_folders

theorem deleted_files (h1 : initial_files = 27) (h2 : files_per_folder = 6) (h3 : folders = 3) :
  files_deleted initial_files (files_per_folder * folders) = 9 :=
by
  sorry

end deleted_files_l78_78691


namespace tan_beta_eq_minus_one_seventh_l78_78511

theorem tan_beta_eq_minus_one_seventh {α β : ℝ} 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := 
by
  sorry

end tan_beta_eq_minus_one_seventh_l78_78511


namespace find_x_tan_identity_l78_78052

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l78_78052


namespace initial_gummy_worms_l78_78946

variable (G : ℕ)

theorem initial_gummy_worms (h : (G : ℚ) / 16 = 4) : G = 64 :=
by
  sorry

end initial_gummy_worms_l78_78946


namespace complement_of_union_l78_78248

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  (U \ (M ∪ N)) = {4} :=
by
  sorry

end complement_of_union_l78_78248


namespace quadratic_geometric_sequence_root_l78_78436

theorem quadratic_geometric_sequence_root {a b c : ℝ} (r : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b = a * r) 
  (h3 : c = a * r^2)
  (h4 : a ≥ b) 
  (h5 : b ≥ c) 
  (h6 : c ≥ 0) 
  (h7 : (a * r)^2 - 4 * a * (a * r^2) = 0) : 
  -b / (2 * a) = -1 / 8 := 
sorry

end quadratic_geometric_sequence_root_l78_78436


namespace special_operation_value_l78_78728

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l78_78728


namespace tan_sin_cos_eq_l78_78057

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l78_78057


namespace analytic_expression_and_symmetry_l78_78647

noncomputable def f (A : ℝ) (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem analytic_expression_and_symmetry {A ω φ : ℝ}
  (hA : A > 0) 
  (hω : ω > 0)
  (h_period : ∀ x, f A ω φ (x + 2) = f A ω φ x)
  (h_max : f A ω φ (1 / 3) = 2) :
  (f 2 π (π / 6) = fun x => 2 * Real.sin (π * x + π / 6)) ∧
  (∃ k : ℤ, k = 5 ∧ (1 / 3 + k = 16 / 3) ∧ (21 / 4 ≤ 1 / 3 + ↑k) ∧ (1 / 3 + ↑k ≤ 23 / 4)) :=
  sorry

end analytic_expression_and_symmetry_l78_78647


namespace seymour_fertilizer_requirement_l78_78267

theorem seymour_fertilizer_requirement :
  let flats_petunias := 4
  let petunias_per_flat := 8
  let flats_roses := 3
  let roses_per_flat := 6
  let venus_flytraps := 2
  let fert_per_petunia := 8
  let fert_per_rose := 3
  let fert_per_venus_flytrap := 2

  let total_petunias := flats_petunias * petunias_per_flat
  let total_roses := flats_roses * roses_per_flat
  let fert_petunias := total_petunias * fert_per_petunia
  let fert_roses := total_roses * fert_per_rose
  let fert_venus_flytraps := venus_flytraps * fert_per_venus_flytrap

  let total_fertilizer := fert_petunias + fert_roses + fert_venus_flytraps
  total_fertilizer = 314 := sorry

end seymour_fertilizer_requirement_l78_78267


namespace coordinates_of_P_l78_78816

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end coordinates_of_P_l78_78816


namespace min_N_such_that_next_person_sits_next_to_someone_l78_78923

def circular_table_has_80_chairs : Prop := ∃ chairs : ℕ, chairs = 80
def N_people_seated (N : ℕ) : Prop := N > 0
def next_person_sits_next_to_someone (N : ℕ) : Prop :=
  ∀ additional_person_seated : ℕ, additional_person_seated ≤ N → additional_person_seated > 0 
  → ∃ adjacent_person : ℕ, adjacent_person ≤ N ∧ adjacent_person > 0
def smallest_value_for_N (N : ℕ) : Prop :=
  (∀ k : ℕ, k < N → ¬next_person_sits_next_to_someone k)

theorem min_N_such_that_next_person_sits_next_to_someone :
  circular_table_has_80_chairs →
  smallest_value_for_N 20 :=
by
  intro h
  sorry

end min_N_such_that_next_person_sits_next_to_someone_l78_78923


namespace derivative_f_at_1_l78_78082

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end derivative_f_at_1_l78_78082


namespace variance_of_X_is_correct_l78_78737

/-!
  There is a batch of products, among which there are 12 genuine items and 4 defective items.
  If 3 items are drawn with replacement, and X represents the number of defective items drawn,
  prove that the variance of X is 9 / 16 given that X follows a binomial distribution B(3, 1 / 4).
-/

noncomputable def variance_of_binomial : Prop :=
  let n := 3
  let p := 1 / 4
  let variance := n * p * (1 - p)
  variance = 9 / 16

theorem variance_of_X_is_correct : variance_of_binomial := by
  sorry

end variance_of_X_is_correct_l78_78737


namespace peter_vacation_saving_l78_78410

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end peter_vacation_saving_l78_78410


namespace sam_wins_probability_l78_78838

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l78_78838


namespace minimum_cost_to_reverse_chips_order_l78_78774

theorem minimum_cost_to_reverse_chips_order : 
  ∀ (n : ℕ) (chips : Fin n → ℕ), 
    (∀ i : ℕ, i < n → chips i = i) →
    (∀ i j : ℕ, i < j ∧ j = i + 1 → 1) →
    (∀ i j : ℕ, j = i + 5 → 0) →
    n = 100 → 
    reverse_cost chips = 61 := 
by 
  intros n chips hchip_order hswap_cost1 hswap_cost2 hn 
  sorry

end minimum_cost_to_reverse_chips_order_l78_78774


namespace escalator_walk_rate_l78_78929

theorem escalator_walk_rate (v : ℝ) : (v + 15) * 10 = 200 → v = 5 := by
  sorry

end escalator_walk_rate_l78_78929


namespace basketball_player_probability_l78_78914

-- Define the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℝ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  binom n k * p^k * (1 - p)^(n - k)

-- Declare the main theorem
theorem basketball_player_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  binomial_prob 10 3 p = binom 10 3 * p^3 * (1 - p)^7 :=
by
  -- We state the answer directly as per the solution
  sorry

end basketball_player_probability_l78_78914


namespace sam_wins_l78_78832

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l78_78832


namespace three_a_in_S_implies_a_in_S_l78_78107

def S := {n | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : 3 * a ∈ S) : a ∈ S := 
sorry

end three_a_in_S_implies_a_in_S_l78_78107


namespace divisible_by_72_l78_78788

theorem divisible_by_72 (a b : ℕ) (h1 : 0 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10) :
  (b = 2 ∧ a = 3) → (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end divisible_by_72_l78_78788


namespace Carter_gave_Marcus_58_cards_l78_78548

-- Define the conditions as variables
def original_cards : ℕ := 210
def current_cards : ℕ := 268

-- Define the question as a function
def cards_given_by_carter (original current : ℕ) : ℕ := current - original

-- Statement that we need to prove
theorem Carter_gave_Marcus_58_cards : cards_given_by_carter original_cards current_cards = 58 :=
by
  -- Proof goes here
  sorry

end Carter_gave_Marcus_58_cards_l78_78548


namespace factor_expression_l78_78190

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end factor_expression_l78_78190


namespace cost_per_person_is_125_l78_78706

-- Defining the conditions
def totalCost : ℤ := 25000000000
def peopleSharing : ℤ := 200000000

-- Define the expected cost per person based on the conditions
def costPerPerson : ℤ := totalCost / peopleSharing

-- Proving that the cost per person is 125 dollars.
theorem cost_per_person_is_125 : costPerPerson = 125 := by
  sorry

end cost_per_person_is_125_l78_78706


namespace find_x_tan_eq_l78_78015

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l78_78015


namespace unattainable_value_of_y_l78_78499

theorem unattainable_value_of_y (x : ℚ) (h : x ≠ -5/4) :
  ¬ ∃ y : ℚ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3/4 :=
by
  sorry

end unattainable_value_of_y_l78_78499


namespace centroid_path_area_correct_l78_78823

noncomputable def centroid_path_area (AB : ℝ) (A B C : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let R := AB / 2
  let radius_of_path := R / 3
  let area := Real.pi * radius_of_path ^ 2
  area

theorem centroid_path_area_correct (AB : ℝ) (A B C : ℝ × ℝ)
  (hAB : AB = 32)
  (hAB_diameter : (∃ O : ℝ × ℝ, dist O A = dist O B ∧ dist A B = 2 * dist O A))
  (hC_circle : ∃ O : ℝ × ℝ, dist O C = AB / 2 ∧ C ≠ A ∧ C ≠ B):
  centroid_path_area AB A B C (0, 0) = (256 / 9) * Real.pi := by
  sorry

end centroid_path_area_correct_l78_78823


namespace monotonic_intervals_of_f_l78_78074

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

-- Define the derivative f'
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

-- Prove the monotonicity intervals of the function f
theorem monotonic_intervals_of_f :
  (∀ x : ℝ, x < 0 → f' x < 0) ∧ (∀ x : ℝ, 0 < x → f' x > 0) :=
by
  sorry

end monotonic_intervals_of_f_l78_78074


namespace replacement_parts_l78_78443

theorem replacement_parts (num_machines : ℕ) (parts_per_machine : ℕ) (week1_fail_rate : ℚ) (week2_fail_rate : ℚ) (week3_fail_rate : ℚ) :
  num_machines = 500 ->
  parts_per_machine = 6 ->
  week1_fail_rate = 0.10 ->
  week2_fail_rate = 0.30 ->
  week3_fail_rate = 0.60 ->
  (num_machines * parts_per_machine) * week1_fail_rate +
  (num_machines * parts_per_machine) * week2_fail_rate +
  (num_machines * parts_per_machine) * week3_fail_rate = 3000 := by
  sorry

end replacement_parts_l78_78443


namespace count_rectangles_5x5_l78_78984

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l78_78984


namespace range_of_a_l78_78209

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, ¬ (x^2 + (a-1)*x + 1 ≤ 0)
def proposition_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a-1)^x < (a-1)^y

theorem range_of_a (a : ℝ) :
  ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) →
  (-1 < a ∧ a ≤ 2) ∨ (3 ≤ a) :=
by sorry

end range_of_a_l78_78209


namespace candidate_percentage_valid_votes_l78_78096

theorem candidate_percentage_valid_votes (total_votes invalid_percentage valid_votes_received : ℕ) 
    (h_total_votes : total_votes = 560000)
    (h_invalid_percentage : invalid_percentage = 15)
    (h_valid_votes_received : valid_votes_received = 333200) :
    (valid_votes_received : ℚ) / (total_votes * (1 - invalid_percentage / 100) : ℚ) * 100 = 70 :=
by
  sorry

end candidate_percentage_valid_votes_l78_78096


namespace sin_780_eq_sqrt3_div_2_l78_78767

theorem sin_780_eq_sqrt3_div_2 : Real.sin (780 * Real.pi / 180) = Math.sqrt 3 / 2 := by
  sorry

end sin_780_eq_sqrt3_div_2_l78_78767


namespace James_distance_ridden_l78_78104

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end James_distance_ridden_l78_78104


namespace maximum_value_of_expression_l78_78249

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end maximum_value_of_expression_l78_78249


namespace value_of_x_l78_78516

theorem value_of_x (x y : ℝ) (h₁ : x = y - 0.10 * y) (h₂ : y = 125 + 0.10 * 125) : x = 123.75 := 
by
  sorry

end value_of_x_l78_78516


namespace find_unknown_numbers_l78_78224

def satisfies_condition1 (A B : ℚ) : Prop := 
  0.05 * A = 0.20 * 650 + 0.10 * B

def satisfies_condition2 (A B : ℚ) : Prop := 
  A + B = 4000

def satisfies_condition3 (B C : ℚ) : Prop := 
  C = 2 * B

def satisfies_condition4 (A B C D : ℚ) : Prop := 
  A + B + C = 0.40 * D

theorem find_unknown_numbers (A B C D : ℚ) :
  satisfies_condition1 A B → satisfies_condition2 A B →
  satisfies_condition3 B C → satisfies_condition4 A B C D →
  A = 3533 + 1/3 ∧ B = 466 + 2/3 ∧ C = 933 + 1/3 ∧ D = 12333 + 1/3 :=
by
  sorry

end find_unknown_numbers_l78_78224


namespace rectangle_count_5x5_l78_78968

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l78_78968


namespace natalie_needs_12_bushes_for_60_zucchinis_l78_78324

-- Definitions based on problem conditions
def bushes_to_containers (bushes : ℕ) : ℕ := bushes * 10
def containers_to_zucchinis (containers : ℕ) : ℕ := (containers * 3) / 6

-- Theorem statement
theorem natalie_needs_12_bushes_for_60_zucchinis : 
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) = 60 ∧ bushes = 12 := by
  sorry

end natalie_needs_12_bushes_for_60_zucchinis_l78_78324


namespace smallest_primer_is_6_primer6_l78_78608

def distinct_prime_factors (n : ℕ) : ℕ :=
  (Finset.filter Nat.prime (Nat.factors n)).card

def is_prime (n : ℕ) : Prop :=
  Nat.prime n

def is_primer (n : ℕ) : Prop :=
  is_prime (distinct_prime_factors n)

theorem smallest_primer_is_6 : ∀ n : ℕ, is_primer n → n >= 6 :=
begin
  sorry
end

theorem primer6 : is_primer 6 :=
begin
  -- Proof not required as per the problem statement
  sorry
end

end smallest_primer_is_6_primer6_l78_78608


namespace vector_addition_example_l78_78200

theorem vector_addition_example :
  (⟨-3, 2, -1⟩ : ℝ × ℝ × ℝ) + (⟨1, 5, -3⟩ : ℝ × ℝ × ℝ) = ⟨-2, 7, -4⟩ :=
by
  sorry

end vector_addition_example_l78_78200


namespace eval_expression_l78_78495

-- Definitions for the problem conditions
def reciprocal (a : ℕ) : ℚ := 1 / a

-- The theorem statement
theorem eval_expression : (reciprocal 9 - reciprocal 6)⁻¹ = -18 := by
  sorry

end eval_expression_l78_78495


namespace six_coins_heads_or_tails_probability_l78_78423

open ProbabilityTheory

noncomputable def probability_six_heads_or_tails (n : ℕ) (h : n = 6) : ℚ :=
  -- Total number of possible outcomes
  let total_outcomes := 2 ^ n in
  -- Number of favorable outcomes: all heads or all tails
  let favorable_outcomes := 2 in
  -- Probability calculation
  favorable_outcomes / total_outcomes

theorem six_coins_heads_or_tails_probability : probability_six_heads_or_tails 6 rfl = 1 / 32 := by
  sorry

end six_coins_heads_or_tails_probability_l78_78423


namespace ratio_S6_S3_l78_78786

theorem ratio_S6_S3 (a : ℝ) (q : ℝ) (h : a + 8 * a * q^3 = 0) : 
  (a * (1 - q^6) / (1 - q)) / (a * (1 - q^3) / (1 - q)) = 9 / 8 :=
by
  sorry

end ratio_S6_S3_l78_78786


namespace Yanna_apples_l78_78588

def total_apples_bought (given_to_zenny : ℕ) (given_to_andrea : ℕ) (kept : ℕ) : ℕ :=
  given_to_zenny + given_to_andrea + kept

theorem Yanna_apples {given_to_zenny given_to_andrea kept total : ℕ}:
  given_to_zenny = 18 →
  given_to_andrea = 6 →
  kept = 36 →
  total_apples_bought given_to_zenny given_to_andrea kept = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end Yanna_apples_l78_78588


namespace measure_of_angle_l78_78709

theorem measure_of_angle (x : ℝ) (h : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l78_78709


namespace zero_neither_positive_nor_negative_l78_78480

def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0
def is_rational (n : ℤ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ n = p / q

theorem zero_neither_positive_nor_negative : ¬is_positive 0 ∧ ¬is_negative 0 :=
by
  sorry

end zero_neither_positive_nor_negative_l78_78480


namespace john_investment_in_bank_a_l78_78532

theorem john_investment_in_bank_a :
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1500 ∧
    x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1740.54 ∧
    x = 695 := sorry

end john_investment_in_bank_a_l78_78532


namespace number_of_possible_values_of_r_eq_894_l78_78713

noncomputable def r_possible_values : ℕ :=
  let lower_bound := 0.3125
  let upper_bound := 0.4018
  let min_r := 3125  -- equivalent to the lowest four-digit decimal ≥ 0.3125
  let max_r := 4018  -- equivalent to the highest four-digit decimal ≤ 0.4018
  1 + max_r - min_r  -- total number of possible values

theorem number_of_possible_values_of_r_eq_894 :
  r_possible_values = 894 :=
by
  sorry

end number_of_possible_values_of_r_eq_894_l78_78713


namespace raisins_in_other_three_boxes_l78_78763

-- Definitions of the known quantities
def total_raisins : ℕ := 437
def box1_raisins : ℕ := 72
def box2_raisins : ℕ := 74

-- The goal is to prove that each of the other three boxes has 97 raisins
theorem raisins_in_other_three_boxes :
  total_raisins - (box1_raisins + box2_raisins) = 3 * 97 :=
by
  sorry

end raisins_in_other_three_boxes_l78_78763


namespace exists_positive_int_n_l78_78112

theorem exists_positive_int_n (p a k : ℕ) 
  (hp : Nat.Prime p) (ha : 0 < a) (hk1 : p^a < k) (hk2 : k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
sorry

end exists_positive_int_n_l78_78112


namespace age_sum_is_ninety_l78_78617

theorem age_sum_is_ninety (a b c : ℕ)
  (h1 : a = 20 + b + c)
  (h2 : a^2 = 1800 + (b + c)^2) :
  a + b + c = 90 := 
sorry

end age_sum_is_ninety_l78_78617


namespace find_b_l78_78808

noncomputable def complex_b_value (i : ℂ) (b : ℝ) : Prop :=
(1 + b * i) * i = 1 + i

theorem find_b (i : ℂ) (b : ℝ) (hi : i^2 = -1) (h : complex_b_value i b) : b = -1 :=
by {
  sorry
}

end find_b_l78_78808


namespace max_ab_l78_78563

theorem max_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 1) : ab ≤ 1 / 16 :=
sorry

end max_ab_l78_78563


namespace smallest_positive_period_and_axis_of_symmetry_l78_78643

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem smallest_positive_period_and_axis_of_symmetry :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ k : ℤ, ∀ x, 2 * x - Real.pi / 4 = k * Real.pi + Real.pi / 2 → x = k * Real.pi / 2 - Real.pi / 8) :=
  sorry

end smallest_positive_period_and_axis_of_symmetry_l78_78643


namespace points_player_1_after_13_rotations_l78_78883

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l78_78883


namespace number_of_valid_pairs_l78_78321

theorem number_of_valid_pairs :
  (∃! S : ℕ, S = 1250 ∧ ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 1000) →
  (3^n < 4^m ∧ 4^m < 4^(m+1) ∧ 4^(m+1) < 3^(n+1))) :=
sorry

end number_of_valid_pairs_l78_78321


namespace right_triangle_sqrt_l78_78903

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end right_triangle_sqrt_l78_78903


namespace bamboo_sections_volume_l78_78594

theorem bamboo_sections_volume (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a n = a 0 + n * d) →
  (a 0 + a 1 + a 2 = 4) →
  (a 5 + a 6 + a 7 + a 8 = 3) →
  (a 3 + a 4 = 2 + 3 / 22) :=
sorry

end bamboo_sections_volume_l78_78594


namespace sufficient_but_not_necessary_condition_for_parallelism_l78_78258

-- Define the two lines
def line1 (x y : ℝ) (m : ℝ) : Prop := 2 * x - m * y = 1
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 1) * x - y = 1

-- Define the parallel condition for the two lines
def parallel (m : ℝ) : Prop :=
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 m ∧ line2 x2 y2 m ∧ (2 * m + 1 = 0 ∧ m^2 - m - 2 = 0)) ∨ 
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 2 ∧ line2 x2 y2 2)

theorem sufficient_but_not_necessary_condition_for_parallelism :
  ∀ m, (parallel m) ↔ (m = 2) :=
by sorry

end sufficient_but_not_necessary_condition_for_parallelism_l78_78258


namespace no_multiple_of_2310_in_2_j_minus_2_i_l78_78509

theorem no_multiple_of_2310_in_2_j_minus_2_i (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 50) :
  ¬ ∃ k : ℕ, 2^j - 2^i = 2310 * k :=
by 
  sorry

end no_multiple_of_2310_in_2_j_minus_2_i_l78_78509


namespace remainder_of_expression_mod7_l78_78153

theorem remainder_of_expression_mod7 :
  (7^6 + 8^7 + 9^8) % 7 = 5 :=
by
  sorry

end remainder_of_expression_mod7_l78_78153


namespace Karl_max_score_l78_78537

def max_possible_score : ℕ :=
  69

theorem Karl_max_score (minutes problems : ℕ) (n_points : ℕ → ℕ) (time_1_5 : ℕ) (time_6_10 : ℕ) (time_11_15 : ℕ)
    (h1 : minutes = 15) (h2 : problems = 15)
    (h3 : ∀ n, n = n_points n)
    (h4 : ∀ i, 1 ≤ i ∧ i ≤ 5 → time_1_5 = 1)
    (h5 : ∀ i, 6 ≤ i ∧ i ≤ 10 → time_6_10 = 2)
    (h6 : ∀ i, 11 ≤ i ∧ i ≤ 15 → time_11_15 = 3) : 
    max_possible_score = 69 :=
  by
  sorry

end Karl_max_score_l78_78537


namespace regular_18gon_lines_rotational_symmetry_sum_l78_78927

def L : ℕ := 18
def R : ℕ := 20

theorem regular_18gon_lines_rotational_symmetry_sum : L + R = 38 :=
by 
  sorry

end regular_18gon_lines_rotational_symmetry_sum_l78_78927


namespace num_rectangles_in_5x5_grid_l78_78976

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l78_78976


namespace part1_part2_l78_78215

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- Problem (1)
theorem part1 (a : ℝ) (h : a = 1) : 
  ∀ x : ℝ, f x a ≥ 2 ↔ x ≤ 0 ∨ x ≥ 2 := 
  sorry

-- Problem (2)
theorem part2 (a : ℝ) (h : a > 1) : 
  (∀ x : ℝ, f x a + abs (x - 1) ≥ 2) ↔ a ≥ 3 := 
  sorry

end part1_part2_l78_78215


namespace num_rectangles_in_5x5_grid_l78_78971

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l78_78971


namespace fraction_simplification_l78_78494

theorem fraction_simplification 
  (a b c : ℝ)
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : a^2 + b^2 + c^2 ≠ 0) :
  (a^2 * b^2 + 2 * a^2 * b * c + a^2 * c^2 - b^4) / (a^4 - b^2 * c^2 + 2 * a * b * c^2 + c^4) =
  ((a * b + a * c + b^2) * (a * b + a * c - b^2)) / ((a^2 + b^2 - c^2) * (a^2 - b^2 + c^2)) :=
sorry

end fraction_simplification_l78_78494


namespace oxygen_part_weight_l78_78777

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O
def given_molecular_weight : ℝ := 108

theorem oxygen_part_weight : molecular_weight_N2O = 44.02 → atomic_weight_O = 16.00 := by
  sorry

end oxygen_part_weight_l78_78777


namespace malcolm_needs_more_lights_l78_78402

def red_lights := 12
def blue_lights := 3 * red_lights
def green_lights := 6
def white_lights := 59

def colored_lights := red_lights + blue_lights + green_lights
def need_more_lights := white_lights - colored_lights

theorem malcolm_needs_more_lights :
  need_more_lights = 5 :=
by
  sorry

end malcolm_needs_more_lights_l78_78402


namespace focus_of_parabola_l78_78269

theorem focus_of_parabola : 
  (∃ p : ℝ, y^2 = 4 * p * x ∧ p = 1 ∧ ∃ c : ℝ × ℝ, c = (1, 0)) :=
sorry

end focus_of_parabola_l78_78269


namespace students_and_ticket_price_l78_78129

theorem students_and_ticket_price (students teachers ticket_price : ℕ) 
  (h1 : students % 5 = 0)
  (h2 : (students + teachers) * (ticket_price / 2) = 1599)
  (h3 : ∃ n, ticket_price = 2 * n) 
  (h4 : teachers = 1) :
  students = 40 ∧ ticket_price = 78 := 
by
  sorry

end students_and_ticket_price_l78_78129


namespace cad_to_jpy_l78_78750

theorem cad_to_jpy (h : 2000 / 18 =  y / 5) : y = 556 := 
by 
  sorry

end cad_to_jpy_l78_78750


namespace prob_representative_error_lt_3mm_l78_78437

/--
Given:
1. The root mean square deviation (RMSD) of 10 measurements is 10 mm.
2. The measurements are samples from a normal distribution.
3. Use the $t$-distribution with sample size n = 10.

Prove that the probability that the representativeness error in absolute value is less than 3 mm is approximately 0.608.
-/
theorem prob_representative_error_lt_3mm
  (RMSD : ℝ)
  (n : ℕ)
  (h_RMSD : RMSD = 10)
  (h_n : n = 10) :
  let t_dist := t_distribution (n - 1)
  in probability (|representative_error| < 3) t_dist = 0.608 := by
  sorry

end prob_representative_error_lt_3mm_l78_78437


namespace problem_statement_l78_78459

-- Define a set S
variable {S : Type*}

-- Define the binary operation on S
variable (mul : S → S → S)

-- Assume the given condition: (a * b) * a = b for all a, b in S
axiom given_condition : ∀ (a b : S), (mul (mul a b) a) = b

-- Prove that a * (b * a) = b for all a, b in S
theorem problem_statement : ∀ (a b : S), mul a (mul b a) = b :=
by
  sorry

end problem_statement_l78_78459


namespace smallest_n_for_inequality_l78_78157

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end smallest_n_for_inequality_l78_78157


namespace parabola_addition_l78_78924

def f (a b c x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c
def g (a b c x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

theorem parabola_addition (a b c x : ℝ) : 
  (f a b c x + g a b c x) = (2 * a * x^2 + 2 * c - 7 * b) :=
by
  sorry

end parabola_addition_l78_78924


namespace range_of_d_l78_78203

theorem range_of_d (a_1 d : ℝ) (h : (a_1 + 2 * d) * (a_1 + 3 * d) + 1 = 0) :
  d ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end range_of_d_l78_78203


namespace smallest_n_with_2020_divisors_l78_78341

def τ (n : ℕ) : ℕ := 
  ∏ p in (Nat.factors n).toFinset, (Nat.factors n).count p + 1

theorem smallest_n_with_2020_divisors : 
  ∃ n : ℕ, τ n = 2020 ∧ ∀ m : ℕ, τ m = 2020 → n ≤ m :=
  sorry

end smallest_n_with_2020_divisors_l78_78341


namespace solve_for_s_l78_78845

theorem solve_for_s : ∃ s, (∃ x, 4 * x^2 - 8 * x - 320 = 0) ∧ s = 81 :=
by {
  -- Sorry is used to skip the actual proof.
  sorry
}

end solve_for_s_l78_78845


namespace probability_is_1_div_28_l78_78738

noncomputable def probability_valid_combinations : ℚ :=
  let total_combinations := Nat.choose 8 3
  let valid_combinations := 2
  valid_combinations / total_combinations

theorem probability_is_1_div_28 :
  probability_valid_combinations = 1 / 28 := by
  sorry

end probability_is_1_div_28_l78_78738


namespace largest_common_divisor_l78_78899

theorem largest_common_divisor (h408 : ∀ d, Nat.dvd d 408 → d ∈ [1, 2, 3, 4, 6, 8, 12, 17, 24, 34, 51, 68, 102, 136, 204, 408])
                               (h340 : ∀ d, Nat.dvd d 340 → d ∈ [1, 2, 4, 5, 10, 17, 20, 34, 68, 85, 170, 340]) :
  ∃ d, Nat.dvd d 408 ∧ Nat.dvd d 340 ∧ d = 68 := by
  sorry

end largest_common_divisor_l78_78899


namespace probability_of_drawing_two_red_shoes_l78_78735

/-- Given there are 7 red shoes and 3 green shoes, 
    and a total of 10 shoes, if two shoes are drawn randomly,
    prove that the probability of drawing both shoes as red is 7/15. -/
theorem probability_of_drawing_two_red_shoes :
  let total_shoes := 10
  let red_shoes := 7
  let green_shoes := 3
  let total_ways := Nat.choose total_shoes 2
  let red_ways := Nat.choose red_shoes 2
  (1 : ℚ) * red_ways / total_ways = 7 / 15  := by
  sorry

end probability_of_drawing_two_red_shoes_l78_78735


namespace processing_decision_l78_78920

-- Definitions of given conditions
def processing_fee (grade: Char) : ℤ :=
  match grade with
  | 'A' => 90
  | 'B' => 50
  | 'C' => 20
  | 'D' => -50
  | _   => 0

def processing_cost (branch: Char) : ℤ :=
  match branch with
  | 'A' => 25
  | 'B' => 20
  | _   => 0

structure FrequencyDistribution :=
  (gradeA : ℕ)
  (gradeB : ℕ)
  (gradeC : ℕ)
  (gradeD : ℕ)

def branchA_distribution : FrequencyDistribution :=
  { gradeA := 40, gradeB := 20, gradeC := 20, gradeD := 20 }

def branchB_distribution : FrequencyDistribution :=
  { gradeA := 28, gradeB := 17, gradeC := 34, gradeD := 21 }

-- Lean 4 statement for proof of questions
theorem processing_decision : 
  let profit (grade: Char) (branch: Char) := processing_fee grade - processing_cost branch
  let avg_profit (dist: FrequencyDistribution) (branch: Char) : ℤ :=
    (profit 'A' branch) * dist.gradeA / 100 +
    (profit 'B' branch) * dist.gradeB / 100 +
    (profit 'C' branch) * dist.gradeC / 100 +
    (profit 'D' branch) * dist.gradeD / 100
  (pA_branchA : Float := branchA_distribution.gradeA / 100.0) = 0.4 ∧
  (pA_branchB : Float := branchB_distribution.gradeA / 100.0) = 0.28 ∧
  avg_profit branchA_distribution 'A' = 15 ∧
  avg_profit branchB_distribution 'B' = 10 →
  avg_profit branchA_distribution 'A' > avg_profit branchB_distribution 'B'
:= by 
  sorry

end processing_decision_l78_78920


namespace max_value_f_l78_78361

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * (4 : ℝ) * x + 2

theorem max_value_f :
  ∃ x : ℝ, -f x = -18 ∧ (∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end max_value_f_l78_78361


namespace raised_bed_area_l78_78314

theorem raised_bed_area (length width : ℝ) (total_area tilled_area remaining_area raised_bed_area : ℝ) 
(h_len : length = 220) (h_wid : width = 120)
(h_total_area : total_area = length * width)
(h_tilled_area : tilled_area = total_area / 2)
(h_remaining_area : remaining_area = total_area / 2)
(h_raised_bed_area : raised_bed_area = (2 / 3) * remaining_area) : raised_bed_area = 8800 :=
by
  have h1 : total_area = 220 * 120, from by rw [h_total_area, h_len, h_wid]
  have h2 : tilled_area = 26400 / 2, from by rw [h_tilled_area, h1]
  have h3 : remaining_area = 26400 / 2, from by rw [h_remaining_area, h1]
  have h4 : raised_bed_area = (2 / 3) * 13200, from by rw [h_raised_bed_area, h3]
  have h5 : raised_bed_area = 8800, from by rwa [← h_raised_bed_area, h4]
  exact h5

end raised_bed_area_l78_78314


namespace tangent_line_at_point_A_l78_78330

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

def point : ℝ × ℝ := (0, 1)

theorem tangent_line_at_point_A :
  ∃ m b : ℝ, (∀ x : ℝ, (curve x - (m * x + b))^2 = 0) ∧  
  m = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_at_point_A_l78_78330


namespace central_angle_double_score_l78_78601

theorem central_angle_double_score 
  (prob: ℚ)
  (total_angle: ℚ)
  (num_regions: ℚ)
  (eq_regions: ℚ → Prop)
  (double_score_prob: prob = 1/8)
  (total_angle_eq: total_angle = 360)
  (num_regions_eq: num_regions = 6) 
  : ∃ x: ℚ, (prob = x / total_angle) → x = 45 :=
by
  sorry

end central_angle_double_score_l78_78601


namespace probability_sum_of_four_selected_is_odd_l78_78325

def first_fifteen_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Define the problem condition in Lean
def probability_sum_odd_of_four_selected : ℚ := 
  let total_ways := Nat.choose 15 4
  let ways_with_2 := Nat.choose 14 3
  ways_with_2 / total_ways

-- Assign the simplified probability to the variable
def answer : ℚ := 4 / 15

-- The theorem we need to prove
theorem probability_sum_of_four_selected_is_odd :
  probability_sum_odd_of_four_selected = answer :=
sorry

end probability_sum_of_four_selected_is_odd_l78_78325


namespace DavidCrunchesLessThanZachary_l78_78908

-- Definitions based on conditions
def ZacharyPushUps : ℕ := 44
def ZacharyCrunches : ℕ := 17
def DavidPushUps : ℕ := ZacharyPushUps + 29
def DavidCrunches : ℕ := 4

-- Problem statement we need to prove:
theorem DavidCrunchesLessThanZachary : DavidCrunches = ZacharyCrunches - 13 :=
by
  -- Proof will go here
  sorry

end DavidCrunchesLessThanZachary_l78_78908


namespace necessary_but_not_sufficient_l78_78621

variable (a b : ℝ)

def proposition_A : Prop := a > 0
def proposition_B : Prop := a > b ∧ a⁻¹ > b⁻¹

theorem necessary_but_not_sufficient : (proposition_B a b → proposition_A a) ∧ ¬(proposition_A a → proposition_B a b) :=
by
  sorry

end necessary_but_not_sufficient_l78_78621


namespace min_value_arithmetic_sequence_l78_78069

theorem min_value_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_arith_seq : a n = 1 + (n - 1) * 1)
  (h_sum : S n = n * (1 + n) / 2) :
  ∃ n, (S n + 8) / a n = 9 / 2 :=
by
  sorry

end min_value_arithmetic_sequence_l78_78069


namespace train_length_l78_78310

/-
  Given:
  - Speed of the train is 78 km/h
  - Time to pass an electric pole is 5.0769230769230775 seconds
  We need to prove that the length of the train is 110 meters.
-/

def speed_kmph : ℝ := 78
def time_seconds : ℝ := 5.0769230769230775
def expected_length_meters : ℝ := 110

theorem train_length :
  (speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters :=
by {
  -- Proof goes here
  sorry
}

end train_length_l78_78310


namespace european_stamp_costs_l78_78393

theorem european_stamp_costs :
  let P_Italy := 0.07
  let P_Germany := 0.03
  let N_Italy := 9
  let N_Germany := 15
  N_Italy * P_Italy + N_Germany * P_Germany = 1.08 :=
by
  sorry

end european_stamp_costs_l78_78393


namespace solve_tan_equation_l78_78045

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l78_78045


namespace point_on_x_axis_point_on_y_axis_l78_78064

section
-- Definitions for the conditions
def point_A (a : ℝ) : ℝ × ℝ := (a - 3, a ^ 2 - 4)

-- Proof for point A lying on the x-axis
theorem point_on_x_axis (a : ℝ) (h : (point_A a).2 = 0) :
  point_A a = (-1, 0) ∨ point_A a = (-5, 0) :=
sorry

-- Proof for point A lying on the y-axis
theorem point_on_y_axis (a : ℝ) (h : (point_A a).1 = 0) :
  point_A a = (0, 5) :=
sorry
end

end point_on_x_axis_point_on_y_axis_l78_78064


namespace roots_sum_of_squares_l78_78489

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h_roots : ∀ x, (3 * x^3 - 4 * x^2 + 3 * x + 7 = 0) → (x = p ∨ x = q ∨ x = r))
  (h_sum : p + q + r = 4 / 3)
  (h_prod_sum : p * q + q * r + r * p = 1)
  (h_prod : p * q * r = -7 / 3) :
  p^2 + q^2 + r^2 = -2 / 9 := 
sorry

end roots_sum_of_squares_l78_78489


namespace tan_sin_cos_eq_l78_78028

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l78_78028


namespace ratio_of_segments_of_hypotenuse_l78_78357

theorem ratio_of_segments_of_hypotenuse
  (a b c r s : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_ratio : a / b = 2 / 5)
  (h_r : r = (a^2) / c) 
  (h_s : s = (b^2) / c) : 
  r / s = 4 / 25 := sorry

end ratio_of_segments_of_hypotenuse_l78_78357


namespace factor_expression_l78_78187

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end factor_expression_l78_78187


namespace A_and_B_finish_work_together_in_12_days_l78_78612

theorem A_and_B_finish_work_together_in_12_days 
  (T_B : ℕ) 
  (T_A : ℕ)
  (h1 : T_B = 18) 
  (h2 : T_A = 2 * T_B) : 
  1 / (1 / T_A + 1 / T_B) = 12 := 
by 
  sorry

end A_and_B_finish_work_together_in_12_days_l78_78612


namespace special_operation_value_l78_78727

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l78_78727


namespace algebra_expression_never_zero_l78_78781

theorem algebra_expression_never_zero (x : ℝ) : (1 : ℝ) / (x - 1) ≠ 0 :=
sorry

end algebra_expression_never_zero_l78_78781


namespace probability_black_ball_l78_78091

theorem probability_black_ball :
  let P_red := 0.41
  let P_white := 0.27
  let P_black := 1 - P_red - P_white
  P_black = 0.32 :=
by
  sorry

end probability_black_ball_l78_78091


namespace determinant_of_B_l78_78108

variables {R : Type*} [Field R]
variables (x y : R)

def B : Matrix (Fin 2) (Fin 2) R :=
  ![![x, 2],
    ![-3, y]]

noncomputable def B_inv : Matrix (Fin 2) (Fin 2) R :=
  Matrix.inv B

theorem determinant_of_B :
  B + B_inv = 0 → Matrix.det B = 1 :=
begin
  sorry,
end

end determinant_of_B_l78_78108


namespace infinite_alternating_parity_l78_78828

theorem infinite_alternating_parity (m : ℕ) : ∃ᶠ n in at_top, 
  ∀ i < m, ((5^n / 10^i) % 2) ≠ (((5^n / 10^(i+1)) % 10) % 2) :=
sorry

end infinite_alternating_parity_l78_78828


namespace multiple_of_regular_rate_is_1_5_l78_78115

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end multiple_of_regular_rate_is_1_5_l78_78115


namespace largest_number_among_options_l78_78902

theorem largest_number_among_options :
  let A := 8.12366
  let B := 8.1236666666666 -- Repeating decimal 8.123\overline{6}
  let C := 8.1236363636363 -- Repeating decimal 8.12\overline{36}
  let D := 8.1236236236236 -- Repeating decimal 8.1\overline{236}
  let E := 8.1236123612361 -- Repeating decimal 8.\overline{1236}
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  let A := 8.12366
  let B := 8.12366666666666
  let C := 8.12363636363636
  let D := 8.12362362362362
  let E := 8.12361236123612
  sorry

end largest_number_among_options_l78_78902


namespace player_1_points_after_13_rotations_l78_78885

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l78_78885


namespace adults_on_field_trip_l78_78083

-- Define the conditions
def van_capacity : ℕ := 7
def num_students : ℕ := 33
def num_vans : ℕ := 6

-- Define the total number of people that can be transported given the number of vans and capacity per van
def total_people : ℕ := num_vans * van_capacity

-- The number of people that can be transported minus the number of students gives the number of adults
def num_adults : ℕ := total_people - num_students

-- Theorem to prove the number of adults is 9
theorem adults_on_field_trip : num_adults = 9 :=
by
  -- Skipping the proof
  sorry

end adults_on_field_trip_l78_78083


namespace digit_difference_is_one_l78_78477

theorem digit_difference_is_one {p q : ℕ} (h : 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ p ≠ q)
  (digits_distinct : ∀ n ∈ [p, q], ∀ m ∈ [p, q], n ≠ m)
  (interchange_effect : 10 * p + q - (10 * q + p) = 9) : p - q = 1 :=
sorry

end digit_difference_is_one_l78_78477


namespace son_age_is_14_l78_78842

-- Definition of Sandra's age and the condition about the ages 3 years ago.
def Sandra_age : ℕ := 36
def son_age_3_years_ago (son_age_now : ℕ) : ℕ := son_age_now - 3 
def Sandra_age_3_years_ago := 36 - 3
def condition_3_years_ago (son_age_now : ℕ) : Prop := Sandra_age_3_years_ago = 3 * (son_age_3_years_ago son_age_now)

-- The goal: proving Sandra's son's age is 14
theorem son_age_is_14 (son_age_now : ℕ) (h : condition_3_years_ago son_age_now) : son_age_now = 14 :=
by {
  sorry
}

end son_age_is_14_l78_78842


namespace smallest_number_has_2020_divisors_l78_78340

noncomputable def smallest_number_with_2020_divisors : ℕ :=
  let α1 := 100
  let α2 := 4
  let α3 := 1
  2^α1 * 3^α2 * 5^α3 * 7

theorem smallest_number_has_2020_divisors : ∃ n : ℕ, τ(n) = 2020 ∧ n = smallest_number_with_2020_divisors :=
by
  let n := smallest_number_with_2020_divisors
  have h1 : τ(n) = τ(2^100 * 3^4 * 5 * 7) := sorry
  have h2 : n = 2^100 * 3^4 * 5 * 7 := rfl
  existsi n
  exact ⟨h1, h2⟩

end smallest_number_has_2020_divisors_l78_78340


namespace abs_w_unique_l78_78514

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end abs_w_unique_l78_78514


namespace smallest_value_is_nine_l78_78658

noncomputable def smallest_possible_value (a b c d : ℝ) : ℝ :=
  (⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ : ℝ)

theorem smallest_value_is_nine {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_possible_value a b c d = 9 :=
sorry

end smallest_value_is_nine_l78_78658


namespace smallest_number_divisible_l78_78591

theorem smallest_number_divisible (n : ℕ) : (∃ n : ℕ, (n + 3) % 27 = 0 ∧ (n + 3) % 35 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0) ∧ n = 4722 :=
by
  sorry

end smallest_number_divisible_l78_78591


namespace repeating_block_length_7_div_13_l78_78848

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l78_78848


namespace power_exponent_multiplication_l78_78186

variable (a : ℝ)

theorem power_exponent_multiplication : (a^3)^2 = a^6 := sorry

end power_exponent_multiplication_l78_78186


namespace wire_goes_around_field_l78_78944

theorem wire_goes_around_field :
  (7348 / (4 * Real.sqrt 27889)) = 11 :=
by
  sorry

end wire_goes_around_field_l78_78944


namespace repeating_block_length_of_7_div_13_is_6_l78_78852

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l78_78852


namespace find_a_l78_78078

-- Define given parameters and conditions
def parabola_eq (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

def shifted_parabola_eq (a : ℝ) (x : ℝ) : ℝ := parabola_eq a x - 3 * |a|

-- Define axis of symmetry function
def axis_of_symmetry (a : ℝ) : ℝ := 1

-- Conditions: a ≠ 0
variable (a : ℝ)
variable (h : a ≠ 0)

-- Define value for discriminant check
def discriminant (a : ℝ) (c : ℝ) : ℝ := (-2 * a)^2 - 4 * a * c

-- Problem statement
theorem find_a (ha : a ≠ 0) : 
  (axis_of_symmetry a = 1) ∧ (discriminant a (3 - 3 * |a|) = 0 → (a = 3 / 4 ∨ a = -3 / 2)) := 
by
  sorry -- proof to be filled in

end find_a_l78_78078


namespace find_fourth_number_l78_78565

theorem find_fourth_number : 
  ∀ (x y : ℝ),
  (28 + x + 42 + y + 104) / 5 = 90 ∧ (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 78 :=
by
  intros x y h
  sorry

end find_fourth_number_l78_78565


namespace scientific_notation_of_188_million_l78_78196

theorem scientific_notation_of_188_million : 
  (188000000 : ℝ) = 1.88 * 10^8 := 
by
  sorry

end scientific_notation_of_188_million_l78_78196


namespace intersection_point_of_y_eq_4x_minus_2_with_x_axis_l78_78132

theorem intersection_point_of_y_eq_4x_minus_2_with_x_axis :
  ∃ x, (4 * x - 2 = 0 ∧ (x, 0) = (1 / 2, 0)) :=
by
  sorry

end intersection_point_of_y_eq_4x_minus_2_with_x_axis_l78_78132


namespace number_of_sides_of_regular_polygon_l78_78369

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end number_of_sides_of_regular_polygon_l78_78369


namespace dave_probability_l78_78934

theorem dave_probability :
  let gates := 15
  let dist_between_gates := 100
  let initial_gate := (0 : Fin gates)
  let new_gate := (0 : Fin gates)
  
  let total_positions := gates * (gates - 1)
  let favorable_positions :=
    2 * (5 + 6 + 7 + 8 + 9) + 5 * 10
  
  let probability := favorable_positions / total_positions in
  let p := 4
  let q := 7 in
  probability = (p / q) := by
  sorry

end dave_probability_l78_78934


namespace largest_number_is_310_l78_78574

def largest_number_formed (a b c : ℕ) : ℕ :=
  max (a * 100 + b * 10 + c) (max (a * 100 + c * 10 + b) (max (b * 100 + a * 10 + c) 
  (max (b * 100 + c * 10 + a) (max (c * 100 + a * 10 + b) (c * 100 + b * 10 + a)))))

theorem largest_number_is_310 : largest_number_formed 3 1 0 = 310 :=
by simp [largest_number_formed]; sorry

end largest_number_is_310_l78_78574


namespace largest_integer_n_l78_78197

theorem largest_integer_n (n : ℤ) (h : n^2 - 13 * n + 40 < 0) : n = 7 :=
by
  sorry

end largest_integer_n_l78_78197


namespace cost_per_use_correct_l78_78531

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end cost_per_use_correct_l78_78531


namespace part_a_part_b_l78_78454

-- Part (a) Equivalent Proof Problem
theorem part_a (k : ℤ) : 
  ∃ a b c : ℤ, 3 * k - 2 = a ^ 2 + b ^ 3 + c ^ 3 := 
sorry

-- Part (b) Equivalent Proof Problem
theorem part_b (n : ℤ) : 
  ∃ a b c d : ℤ, n = a ^ 2 + b ^ 3 + c ^ 3 + d ^ 3 := 
sorry

end part_a_part_b_l78_78454


namespace books_selection_count_l78_78204

-- Define the total number of books on the shelf
def total_books : ℕ := 8

-- Define the number of books to be selected
def books_to_select : ℕ := 5

-- Define the inclusion of one specific book
def specific_book_included : ℕ := 1

-- Define the problem statement
theorem books_selection_count : ∃ n : ℕ, n = (finset.card (finset.filter (λ s, specific_book_included ∈ s) (finset.powerset_len books_to_select (finset.range total_books)))) := 35 := 
by {
  -- Placeholder for the proof
  sorry
}

end books_selection_count_l78_78204


namespace range_of_a_l78_78398

-- Definitions for propositions
def p (a : ℝ) : Prop :=
  (1 - 4 * (a^2 - 6 * a) > 0) ∧ (a^2 - 6 * a < 0)

def q (a : ℝ) : Prop :=
  (a - 3)^2 - 4 ≥ 0

-- Proof statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (a ≤ 0 ∨ 1 < a ∧ a < 5 ∨ a ≥ 6) :=
by 
  sorry

end range_of_a_l78_78398


namespace not_perfect_square_for_n_greater_than_11_l78_78694

theorem not_perfect_square_for_n_greater_than_11 (n : ℤ) (h1 : n > 11) :
  ∀ m : ℤ, n^2 - 19 * n + 89 ≠ m^2 :=
sorry

end not_perfect_square_for_n_greater_than_11_l78_78694


namespace children_left_birthday_l78_78405

theorem children_left_birthday 
  (total_guests : ℕ := 60)
  (women : ℕ := 30)
  (men : ℕ := 15)
  (remaining_guests : ℕ := 50)
  (initial_children : ℕ := total_guests - women - men)
  (men_left : ℕ := men / 3)
  (total_left : ℕ := total_guests - remaining_guests)
  (children_left : ℕ := total_left - men_left) :
  children_left = 5 :=
by
  sorry

end children_left_birthday_l78_78405


namespace books_written_by_Zig_l78_78589

theorem books_written_by_Zig (F Z : ℕ) (h1 : Z = 4 * F) (h2 : F + Z = 75) : Z = 60 := by
  sorry

end books_written_by_Zig_l78_78589


namespace max_principals_in_10_years_l78_78635

theorem max_principals_in_10_years : ∀ term_length num_years,
  (term_length = 4) ∧ (num_years = 10) →
  ∃ max_principals, max_principals = 3
:=
  by intros term_length num_years h
     sorry

end max_principals_in_10_years_l78_78635


namespace set_B_listing_method_l78_78640

variable (A : Set ℕ) (B : Set ℕ)

theorem set_B_listing_method (hA : A = {1, 2, 3}) (hB : B = {x | x ∈ A}) :
  B = {1, 2, 3} :=
  by
    sorry

end set_B_listing_method_l78_78640


namespace last_two_digits_sum_is_32_l78_78895

-- Definitions for digit representation
variables (z a r l m : ℕ)

-- Numbers definitions
def ZARAZA := z * 10^5 + a * 10^4 + r * 10^3 + a * 10^2 + z * 10 + a
def ALMAZ := a * 10^4 + l * 10^3 + m * 10^2 + a * 10 + z

-- Condition that ZARAZA is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Condition that ALMAZ is divisible by 28
def divisible_by_28 (n : ℕ) : Prop := n % 28 = 0

-- The theorem to prove
theorem last_two_digits_sum_is_32
  (hz4 : divisible_by_4 (ZARAZA z a r))
  (ha28 : divisible_by_28 (ALMAZ a l m z))
  : (ZARAZA z a r + ALMAZ a l m z) % 100 = 32 :=
by sorry

end last_two_digits_sum_is_32_l78_78895


namespace malesWithCollegeDegreesOnly_l78_78377

-- Define the parameters given in the problem
def totalEmployees : ℕ := 180
def totalFemales : ℕ := 110
def employeesWithAdvancedDegrees : ℕ := 90
def employeesWithCollegeDegreesOnly : ℕ := totalEmployees - employeesWithAdvancedDegrees
def femalesWithAdvancedDegrees : ℕ := 55

-- Define the question as a theorem
theorem malesWithCollegeDegreesOnly : 
  totalEmployees = 180 →
  totalFemales = 110 →
  employeesWithAdvancedDegrees = 90 →
  employeesWithCollegeDegreesOnly = 90 →
  femalesWithAdvancedDegrees = 55 →
  ∃ (malesWithCollegeDegreesOnly : ℕ), 
    malesWithCollegeDegreesOnly = 35 := 
by
  intros
  sorry

end malesWithCollegeDegreesOnly_l78_78377


namespace find_x_l78_78870

theorem find_x (α : ℝ) (x : ℝ) (h1 : sin(α) = 4 / 5) (h2 : sqrt (x ^ 2 + 16) ≠ 0) : (x = 3) ∨ (x = -3) :=
by
  have h3 : 4 / sqrt (x ^ 2 + 16) = 4 / 5 := h1
  have h4 : sqrt (x ^ 2 + 16) = 5 := by linarith
  have h5 : x ^ 2 + 16 = 25 := by linarith
  have h6 : x ^ 2 = 9 := by linarith
  exact or.inl (eq_of_sq_eq_sq _ h6).left sorry

end find_x_l78_78870


namespace ellipse_foci_y_axis_l78_78809

-- Given the equation of the ellipse x^2 + k * y^2 = 2 with foci on the y-axis,
-- prove that the range of k such that the ellipse is oriented with foci on the y-axis is (0, 1).
theorem ellipse_foci_y_axis (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ a > 0 ∧ b > 0 ∧ b / a = k ∧ x^2 + k * y^2 = 2 :=
sorry

end ellipse_foci_y_axis_l78_78809


namespace arithmetic_expression_evaluation_l78_78485

theorem arithmetic_expression_evaluation :
  (-18) + (-12) - (-33) + 17 = 20 :=
by
  sorry

end arithmetic_expression_evaluation_l78_78485


namespace arithmetic_mean_of_two_digit_multiples_of_9_l78_78149

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l78_78149


namespace xyz_value_l78_78659

noncomputable def positive (x : ℝ) : Prop := 0 < x

theorem xyz_value (x y z : ℝ) (hx : positive x) (hy : positive y) (hz : positive z): 
  (x + 1/y = 5) → (y + 1/z = 2) → (z + 1/x = 8/3) → x * y * z = (17 + Real.sqrt 285) / 2 :=
by
  sorry

end xyz_value_l78_78659


namespace nonneg_int_solutions_to_ineq_system_l78_78199

open Set

theorem nonneg_int_solutions_to_ineq_system :
  {x : ℤ | (5 * x - 6 ≤ 2 * (x + 3)) ∧ ((x / 4 : ℚ) - 1 < (x - 2) / 3)} = {0, 1, 2, 3, 4} :=
by
  sorry

end nonneg_int_solutions_to_ineq_system_l78_78199


namespace smallest_w_factor_l78_78087

theorem smallest_w_factor (w : ℕ) (hw : w > 0) :
  (∃ w, 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w) ↔ w = 79092 :=
by sorry

end smallest_w_factor_l78_78087


namespace student_correct_answers_l78_78909

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 79) : C = 93 :=
by
  sorry

end student_correct_answers_l78_78909


namespace least_xy_l78_78359

noncomputable def condition (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 1 / 7)

theorem least_xy (x y : ℕ) (h : condition x y) : x * y = 98 :=
sorry

end least_xy_l78_78359


namespace even_number_of_irreducible_fractions_l78_78265

def is_irreducible_proper_fraction (k n : ℕ) : Prop := 
  k > 0 ∧ k < n ∧ Nat.gcd k n = 1

def count_irreducible_proper_fractions (n : ℕ) : ℕ := 
  (Finset.range n).filter (λ k => is_irreducible_proper_fraction k n).card

theorem even_number_of_irreducible_fractions (n : ℕ) (h : n > 2) : 
  even (count_irreducible_proper_fractions n) := 
sorry

end even_number_of_irreducible_fractions_l78_78265


namespace moles_of_water_produced_l78_78656

-- Definitions for the chemical reaction
def moles_NaOH := 4
def moles_H₂SO₄ := 2

-- The balanced chemical equation tells us the ratio of NaOH to H₂O
def chemical_equation (moles_NaOH moles_H₂SO₄ moles_H₂O moles_Na₂SO₄: ℕ) : Prop :=
  2 * moles_NaOH = 2 * moles_H₂O ∧ moles_H₂SO₄ = 1 ∧ moles_Na₂SO₄ = 1

-- The actual proof statement
theorem moles_of_water_produced : 
  ∀ (m_NaOH m_H₂SO₄ m_Na₂SO₄ : ℕ), 
  chemical_equation m_NaOH m_H₂SO₄ 4 m_Na₂SO₄ → moles_H₂O = 4 :=
by
  intros m_NaOH m_H₂SO₄ m_Na₂SO₄ chem_eq
  -- Placeholder for the actual proof.
  sorry

end moles_of_water_produced_l78_78656


namespace crayons_lost_or_given_away_l78_78406

theorem crayons_lost_or_given_away (given_away lost : ℕ) (H_given_away : given_away = 213) (H_lost : lost = 16) :
  given_away + lost = 229 :=
by
  sorry

end crayons_lost_or_given_away_l78_78406


namespace usual_time_to_office_l78_78748

theorem usual_time_to_office
  (S T : ℝ) 
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = (4 / 5) * S * (T + 10)):
  T = 40 := 
by
  sorry

end usual_time_to_office_l78_78748


namespace binomial_minus_floor_divisible_by_seven_l78_78657

theorem binomial_minus_floor_divisible_by_seven (n : ℕ) (h : n > 7) :
  ((Nat.choose n 7 : ℤ) - ⌊(n : ℤ) / 7⌋) % 7 = 0 :=
  sorry

end binomial_minus_floor_divisible_by_seven_l78_78657


namespace balloon_minimum_volume_l78_78916

theorem balloon_minimum_volume 
  (p V : ℝ)
  (h1 : p * V = 24000)
  (h2 : p ≤ 40000) : 
  V ≥ 0.6 :=
  sorry

end balloon_minimum_volume_l78_78916


namespace negation_of_universal_statement_l78_78432

def P (x : ℝ) : Prop := x^3 - x^2 + 1 ≤ 0

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by {
  sorry
}

end negation_of_universal_statement_l78_78432


namespace count_rectangles_5x5_l78_78981

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l78_78981


namespace range_of_k_l78_78088

-- Given conditions
variables {k : ℝ} (h : ∃ (x y : ℝ), x^2 + k * y^2 = 2)

-- Theorem statement
theorem range_of_k : 0 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l78_78088


namespace tony_total_winning_l78_78891

theorem tony_total_winning : 
  ∀ (num_tickets num_winning_numbers_per_ticket winnings_per_number : ℕ),
  num_tickets = 3 → 
  num_winning_numbers_per_ticket = 5 →
  winnings_per_number = 20 →
  num_tickets * num_winning_numbers_per_ticket * winnings_per_number = 300 :=
by {
  intros num_tickets num_winning_numbers_per_ticket winnings_per_number h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
}

end tony_total_winning_l78_78891


namespace repeating_block_length_7_div_13_l78_78849

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l78_78849


namespace solve_system_of_equations_l78_78846

variable {x : Fin 15 → ℤ}

theorem solve_system_of_equations (h : ∀ i : Fin 15, 1 - x i * x ((i + 1) % 15) = 0) :
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) :=
by
  -- Here we put the proof, but it's omitted for now.
  sorry

end solve_system_of_equations_l78_78846


namespace total_chairs_in_canteen_l78_78521

theorem total_chairs_in_canteen 
    (round_tables : ℕ) 
    (chairs_per_round_table : ℕ) 
    (rectangular_tables : ℕ) 
    (chairs_per_rectangular_table : ℕ) 
    (square_tables : ℕ) 
    (chairs_per_square_table : ℕ) 
    (extra_chairs : ℕ) 
    (h1 : round_tables = 3)
    (h2 : chairs_per_round_table = 6)
    (h3 : rectangular_tables = 4)
    (h4 : chairs_per_rectangular_table = 7)
    (h5 : square_tables = 2)
    (h6 : chairs_per_square_table = 4)
    (h7 : extra_chairs = 5) :
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table +
    square_tables * chairs_per_square_table +
    extra_chairs = 59 := by
  sorry

end total_chairs_in_canteen_l78_78521


namespace radius_range_l78_78061

noncomputable def circle_eq (x y r : ℝ) := x^2 + y^2 = r^2

def point_P_on_line_AB (m n : ℝ) := 4 * m + 3 * n - 24 = 0

def point_P_in_interval (m : ℝ) := 0 ≤ m ∧ m ≤ 6

theorem radius_range {r : ℝ} :
  (∀ (m n x y : ℝ), point_P_in_interval m →
     circle_eq x y r →
     circle_eq ((x + m) / 2) ((y + n) / 2) r → 
     point_P_on_line_AB m n ∧
     (4 * r ^ 2 ≤ (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ∧
     (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ≤ 36 * r ^ 2)) →
  (8 / 3 ≤ r ∧ r < 12 / 5) :=
sorry

end radius_range_l78_78061


namespace determine_a2016_l78_78799

noncomputable def a_n (n : ℕ) : ℤ := sorry
noncomputable def S_n (n : ℕ) : ℤ := sorry

axiom S1 : S_n 1 = 6
axiom S2 : S_n 2 = 4
axiom S_pos (n : ℕ) : S_n n > 0
axiom geom_progression (n : ℕ) : (S_n (2 * n - 1))^2 = S_n (2 * n) * S_n (2 * n + 2)
axiom arith_progression (n : ℕ) : 2 * S_n (2 * n + 2) = S_n (2 * n - 1) + S_n (2 * n + 1)

theorem determine_a2016 : a_n 2016 = -1009 :=
by sorry

end determine_a2016_l78_78799


namespace work_problem_correct_l78_78517

noncomputable def work_problem : Prop :=
  let A := 1 / 36
  let C := 1 / 6
  let total_rate := 1 / 4
  ∃ B : ℝ, (A + B + C = total_rate) ∧ (B = 1 / 18)

-- Create the theorem statement which says if the conditions are met,
-- then the rate of b must be 1/18 and the number of days b alone takes to
-- finish the work is 18.
theorem work_problem_correct (A C total_rate B : ℝ) (h1 : A = 1 / 36) (h2 : C = 1 / 6) (h3 : total_rate = 1 / 4) 
(h4 : A + B + C = total_rate) : B = 1 / 18 ∧ (1 / B = 18) :=
  by
  sorry

end work_problem_correct_l78_78517


namespace solve_custom_eq_l78_78773

-- Define the custom operation a * b = ab + a + b, we will use ∗ instead of * to avoid confusion with multiplication

def custom_op (a b : Nat) : Nat := a * b + a + b

-- State the problem in Lean 4
theorem solve_custom_eq (x : Nat) : custom_op 3 x = 27 → x = 6 :=
by
  sorry

end solve_custom_eq_l78_78773


namespace tan_of_angle_through_point_l78_78649

theorem tan_of_angle_through_point (α : ℝ) (hα : ∃ x y : ℝ, (x = 1) ∧ (y = 2) ∧ (y/x = (Real.sin α) / (Real.cos α))) :
  Real.tan α = 2 :=
sorry

end tan_of_angle_through_point_l78_78649


namespace common_ratio_l78_78795

theorem common_ratio (a1 a2 a3 : ℚ) (S3 q : ℚ)
  (h1 : a3 = 3 / 2)
  (h2 : S3 = 9 / 2)
  (h3 : a1 + a2 + a3 = S3)
  (h4 : a1 = a3 / q^2)
  (h5 : a2 = a3 / q):
  q = 1 ∨ q = -1/2 :=
by sorry

end common_ratio_l78_78795


namespace dave_apps_added_l78_78629

theorem dave_apps_added (initial_apps : ℕ) (total_apps_after_adding : ℕ) (apps_added : ℕ) 
  (h1 : initial_apps = 17) (h2 : total_apps_after_adding = 18) 
  (h3 : total_apps_after_adding = initial_apps + apps_added) : 
  apps_added = 1 := 
by
  -- proof omitted
  sorry

end dave_apps_added_l78_78629


namespace amount_of_money_l78_78814

theorem amount_of_money (x y : ℝ) 
  (h1 : x + 1/2 * y = 50) 
  (h2 : 2/3 * x + y = 50) : 
  (x + 1/2 * y = 50) ∧ (2/3 * x + y = 50) :=
by
  exact ⟨h1, h2⟩ 

end amount_of_money_l78_78814


namespace f_f_minus_one_range_of_a_l78_78364

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then (1/2)^x else 1 - 3*x

theorem f_f_minus_one :
  f (f (-1)) = -5 :=
sorry

theorem range_of_a (a : ℝ) :
  f (2*a^2 - 3) > f (5*a) → -1/2 < a ∧ a < 3 :=
sorry

end f_f_minus_one_range_of_a_l78_78364


namespace eunsung_sungmin_menu_cases_l78_78876

theorem eunsung_sungmin_menu_cases :
  let kinds_of_chicken := 4
  let kinds_of_pizza := 3
  let same_chicken_different_pizza :=
    kinds_of_chicken * (kinds_of_pizza * (kinds_of_pizza - 1))
  let same_pizza_different_chicken :=
    kinds_of_pizza * (kinds_of_chicken * (kinds_of_chicken - 1))
  same_chicken_different_pizza + same_pizza_different_chicken = 60 :=
by
  sorry

end eunsung_sungmin_menu_cases_l78_78876


namespace proof_problem_l78_78811

variable (balls : Finset ℕ) (blackBalls whiteBalls : Finset ℕ)
variable (drawnBalls : Finset ℕ)

/-- There are 6 black balls numbered 1 to 6. -/
def initialBlackBalls : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- There are 4 white balls numbered 7 to 10. -/
def initialWhiteBalls : Finset ℕ := {7, 8, 9, 10}

/-- The total balls (black + white). -/
def totalBalls : Finset ℕ := initialBlackBalls ∪ initialWhiteBalls

/-- The hypergeometric distribution condition for black balls. -/
def hypergeometricBlack : Prop :=
  true  -- placeholder: black balls follow hypergeometric distribution

/-- The probability of drawing 2 white balls is not 1/14. -/
def probDraw2White : Prop :=
  (3 / 7) ≠ (1 / 14)

/-- The probability of the maximum total score (8 points) is 1/14. -/
def probMaxScore : Prop :=
  (15 / 210) = (1 / 14)

/-- Main theorem combining the above conditions for the problem. -/
theorem proof_problem : hypergeometricBlack ∧ probMaxScore :=
by
  unfold hypergeometricBlack
  unfold probMaxScore
  sorry

end proof_problem_l78_78811


namespace total_charge_for_first_4_minutes_under_plan_A_is_0_60_l78_78753

def planA_charges (X : ℝ) (minutes : ℕ) : ℝ :=
  if minutes <= 4 then X
  else X + (minutes - 4) * 0.06

def planB_charges (minutes : ℕ) : ℝ :=
  minutes * 0.08

theorem total_charge_for_first_4_minutes_under_plan_A_is_0_60
  (X : ℝ)
  (h : planA_charges X 18 = planB_charges 18) :
  X = 0.60 :=
by
  sorry

end total_charge_for_first_4_minutes_under_plan_A_is_0_60_l78_78753


namespace cos_pi_minus_alpha_l78_78790

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  Real.cos (π - α) = -12 / 13 :=
sorry

end cos_pi_minus_alpha_l78_78790


namespace percentage_markup_l78_78433

theorem percentage_markup 
  (selling_price : ℝ) 
  (cost_price : ℝ) 
  (h1 : selling_price = 8215)
  (h2 : cost_price = 6625)
  : ((selling_price - cost_price) / cost_price) * 100 = 24 := 
  by
    sorry

end percentage_markup_l78_78433


namespace x_intercept_of_line_l78_78492

theorem x_intercept_of_line : ∃ x : ℝ, ∃ y : ℝ, 4 * x + 7 * y = 28 ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_of_line_l78_78492


namespace ball_min_bounces_reach_target_height_l78_78462

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end ball_min_bounces_reach_target_height_l78_78462


namespace jen_total_birds_l78_78526

theorem jen_total_birds (C D G : ℕ) (h1 : D = 150) (h2 : D = 4 * C + 10) (h3 : G = (D + C) / 2) :
  D + C + G = 277 := sorry

end jen_total_birds_l78_78526


namespace rectangles_in_grid_l78_78986

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l78_78986


namespace blueberry_pancakes_count_l78_78935

-- Definitions of the conditions
def total_pancakes : ℕ := 67
def banana_pancakes : ℕ := 24
def plain_pancakes : ℕ := 23

-- Statement of the problem
theorem blueberry_pancakes_count :
  total_pancakes - banana_pancakes - plain_pancakes = 20 := by
  sorry

end blueberry_pancakes_count_l78_78935


namespace difference_of_two_numbers_l78_78868

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 15) (h2 : x^2 - y^2 = 150) : x - y = 10 :=
by
  sorry

end difference_of_two_numbers_l78_78868


namespace day_crew_fraction_correct_l78_78163

variable (D Wd : ℕ) -- D = number of boxes loaded by each worker on the day crew, Wd = number of workers on the day crew

-- fraction of all boxes loaded by day crew
def fraction_loaded_by_day_crew (D Wd : ℕ) : ℚ :=
  (D * Wd) / (D * Wd + (3 / 4 * D) * (2 / 3 * Wd))

theorem day_crew_fraction_correct (h1 : D > 0) (h2 : Wd > 0) :
  fraction_loaded_by_day_crew D Wd = 2 / 3 := by
  sorry

end day_crew_fraction_correct_l78_78163


namespace return_trip_time_l78_78474

variable (d p w_1 w_2 : ℝ)
variable (t t' : ℝ)
variable (h1 : d / (p - w_1) = 120)
variable (h2 : d / (p + w_2) = t - 10)
variable (h3 : t = d / p)

theorem return_trip_time :
  t' = 72 :=
by
  sorry

end return_trip_time_l78_78474


namespace joan_spent_on_trucks_l78_78818

-- Define constants for the costs
def cost_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def total_toys : ℝ := 25.62
def cost_trucks : ℝ := 25.62 - (14.88 + 4.88)

-- Statement to prove
theorem joan_spent_on_trucks : cost_trucks = 5.86 := by
  sorry

end joan_spent_on_trucks_l78_78818


namespace tan_sin_cos_eq_l78_78058

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l78_78058


namespace wendy_percentage_accounting_related_jobs_l78_78445

noncomputable def wendy_accountant_years : ℝ := 25.5
noncomputable def wendy_accounting_manager_years : ℝ := 15.5 -- Including 6 months as 0.5 years
noncomputable def wendy_financial_consultant_years : ℝ := 10.25 -- Including 3 months as 0.25 years
noncomputable def wendy_tax_advisor_years : ℝ := 4
noncomputable def wendy_lifespan : ℝ := 80

theorem wendy_percentage_accounting_related_jobs :
  ((wendy_accountant_years + wendy_accounting_manager_years + wendy_financial_consultant_years + wendy_tax_advisor_years) / wendy_lifespan) * 100 = 69.0625 :=
by
  sorry

end wendy_percentage_accounting_related_jobs_l78_78445


namespace player1_points_after_13_rotations_l78_78878

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l78_78878


namespace minimum_value_of_quadratic_expression_l78_78783

theorem minimum_value_of_quadratic_expression (x y z : ℝ)
  (h : x + y + z = 2) : 
  x^2 + 2 * y^2 + z^2 ≥ 4 / 3 :=
sorry

end minimum_value_of_quadratic_expression_l78_78783


namespace regular_polygon_sides_l78_78371

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end regular_polygon_sides_l78_78371


namespace average_birth_rate_l78_78095

theorem average_birth_rate (B : ℕ) (death_rate : ℕ) (net_increase : ℕ) (seconds_per_day : ℕ) 
  (two_sec_intervals : ℕ) (H1 : death_rate = 2) (H2 : net_increase = 86400) (H3 : seconds_per_day = 86400) 
  (H4 : two_sec_intervals = seconds_per_day / 2) 
  (H5 : net_increase = (B - death_rate) * two_sec_intervals) : B = 4 := 
by 
  sorry

end average_birth_rate_l78_78095


namespace find_x_l78_78039

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l78_78039


namespace largest_possible_k_satisfies_triangle_condition_l78_78198

theorem largest_possible_k_satisfies_triangle_condition :
  ∃ k : ℕ, 
    k = 2009 ∧ 
    ∀ (b r w : Fin 2009 → ℝ), 
    (∀ i : Fin 2009, i ≤ i.succ → b i ≤ b i.succ ∧ r i ≤ r i.succ ∧ w i ≤ w i.succ) → 
    (∃ (j : Fin 2009), 
      b j + r j > w j ∧ b j + w j > r j ∧ r j + w j > b j) :=
sorry

end largest_possible_k_satisfies_triangle_condition_l78_78198


namespace Ramesh_paid_l78_78124

theorem Ramesh_paid (P : ℝ) (h1 : 1.10 * P = 21725) : 0.80 * P + 125 + 250 = 16175 :=
by
  sorry

end Ramesh_paid_l78_78124


namespace combined_salaries_l78_78141

variable (S_A S_B S_C S_D S_E : ℝ)

theorem combined_salaries 
    (h1 : S_C = 16000)
    (h2 : (S_A + S_B + S_C + S_D + S_E) / 5 = 9000) : 
    S_A + S_B + S_D + S_E = 29000 :=
by 
    sorry

end combined_salaries_l78_78141


namespace probability_different_colors_l78_78092

theorem probability_different_colors :
  let B := 1 -- number of black balls
  let S := 3 -- number of small balls
  let totalBalls := B + S -- total number of balls
  let totalWays := Nat.choose totalBalls 2 -- ways to choose 2 balls from totalBalls
  let differentColorWays := S -- ways to choose 1 black ball and 1 small ball
  let probability := differentColorWays / totalWays -- probability of different colors
  probability = 1 / 2 :=
by
  sorry

end probability_different_colors_l78_78092


namespace maximum_ab_ac_bc_l78_78252

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end maximum_ab_ac_bc_l78_78252


namespace new_container_volume_l78_78755

def volume_of_cube (s : ℝ) : ℝ := s^3

theorem new_container_volume (s : ℝ) (h : volume_of_cube s = 4) : 
  volume_of_cube (2 * s) * volume_of_cube (3 * s) * volume_of_cube (4 * s) = 96 :=
by
  sorry

end new_container_volume_l78_78755


namespace intersection_of_A_and_B_l78_78541

-- Define sets A and B
def A : set ℝ := {x | -2 < x ∧ x < 4}
def B : set ℕ := {2, 3, 4, 5}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 3} := 
by sorry

end intersection_of_A_and_B_l78_78541


namespace avg_growth_rate_first_brand_eq_l78_78762

noncomputable def avg_growth_rate_first_brand : ℝ :=
  let t := 5.647
  let first_brand_households_2001 := 4.9
  let second_brand_households_2001 := 2.5
  let second_brand_growth_rate := 0.7
  let equalization_time := t
  (second_brand_households_2001 + second_brand_growth_rate * equalization_time - first_brand_households_2001) / equalization_time

theorem avg_growth_rate_first_brand_eq :
  avg_growth_rate_first_brand = 0.275 := by
  sorry

end avg_growth_rate_first_brand_eq_l78_78762


namespace compare_series_l78_78502

theorem compare_series (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) : 
  (1 / (1 - x^2) + 1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by
  sorry

end compare_series_l78_78502


namespace earth_surface_area_scientific_notation_l78_78869

theorem earth_surface_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 780000000 = a * 10^n ∧ a = 7.8 ∧ n = 8 :=
by
  sorry

end earth_surface_area_scientific_notation_l78_78869


namespace arithmetic_mean_is_correct_l78_78151

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l78_78151


namespace chickens_count_l78_78873

def total_animals := 13
def total_legs := 44
def legs_per_chicken := 2
def legs_per_buffalo := 4

theorem chickens_count : 
  (∃ c b : ℕ, c + b = total_animals ∧ legs_per_chicken * c + legs_per_buffalo * b = total_legs ∧ c = 4) :=
by
  sorry

end chickens_count_l78_78873


namespace qudrilateral_diagonal_length_l78_78942

theorem qudrilateral_diagonal_length (A h1 h2 d : ℝ) 
  (h_area : A = 140) (h_offsets : h1 = 8) (h_offsets2 : h2 = 2) 
  (h_formula : A = 1 / 2 * d * (h1 + h2)) : 
  d = 28 :=
by
  sorry

end qudrilateral_diagonal_length_l78_78942


namespace solve_problem_l78_78281

noncomputable def problem_statement : Prop :=
  ∀ (T0 Ta T t1 T1 h t2 T2 : ℝ),
    T0 = 88 ∧ Ta = 24 ∧ T1 = 40 ∧ t1 = 20 ∧
    T1 - Ta = (T0 - Ta) * ((1/2)^(t1/h)) ∧
    T2 = 32 ∧ T2 - Ta = (T1 - Ta) * ((1/2)^(t2/h)) →
    t2 = 10

theorem solve_problem : problem_statement := sorry

end solve_problem_l78_78281


namespace cards_per_box_l78_78416

-- Define the conditions
def total_cards : ℕ := 75
def cards_not_in_box : ℕ := 5
def boxes_given_away : ℕ := 2
def boxes_left : ℕ := 5

-- Calculating the total number of boxes initially
def initial_boxes : ℕ := boxes_given_away + boxes_left

-- Define the number of cards in each box
def num_cards_per_box (number_of_cards : ℕ) (number_of_boxes : ℕ) : ℕ :=
  (number_of_cards - cards_not_in_box) / number_of_boxes

-- The proof problem statement
theorem cards_per_box :
  num_cards_per_box total_cards initial_boxes = 10 :=
by
  -- Proof is omitted with sorry
  sorry

end cards_per_box_l78_78416


namespace costPerUse_l78_78529

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end costPerUse_l78_78529


namespace find_number_l78_78581

theorem find_number :
  ∃ n : ℕ, n * (1 / 7)^2 = 7^3 :=
by
  sorry

end find_number_l78_78581


namespace six_coins_all_heads_or_tails_probability_l78_78419

theorem six_coins_all_heads_or_tails_probability :
  let outcomes := 2^6 in
  let favorable := 2 in
  (favorable / outcomes : ℚ) = 1 / 32 :=
by
  let outcomes := 2^6
  let favorable := 2
  -- skipping the proof
  sorry

end six_coins_all_heads_or_tails_probability_l78_78419


namespace number_of_rectangles_in_grid_l78_78999

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l78_78999


namespace james_initial_marbles_l78_78100

theorem james_initial_marbles (m n : ℕ) (h1 : n = 4) (h2 : m / (n - 1) = 21) :
  m = 28 :=
by sorry

end james_initial_marbles_l78_78100


namespace rectangle_area_ratio_l78_78456

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 :=
by
  sorry

end rectangle_area_ratio_l78_78456


namespace special_operation_value_l78_78730

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l78_78730


namespace squares_difference_sum_l78_78580

theorem squares_difference_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by 
  sorry

end squares_difference_sum_l78_78580


namespace total_points_other_five_l78_78234

theorem total_points_other_five
  (x : ℕ) -- total number of points scored by the team
  (d : ℕ) (e : ℕ) (f : ℕ) (y : ℕ) -- points scored by Daniel, Emma, Fiona, and others respectively
  (hd : d = x / 3) -- Daniel scored 1/3 of the team's points
  (he : e = 3 * x / 8) -- Emma scored 3/8 of the team's points
  (hf : f = 18) -- Fiona scored 18 points
  (h_other : ∀ i, 1 ≤ i ∧ i ≤ 5 → y ≤ 15 / 5) -- Other 5 members scored no more than 3 points each
  (h_total : d + e + f + y = x) -- Total points equation
  : y = 14 := sorry -- Final number of points scored by the other 5 members

end total_points_other_five_l78_78234


namespace probability_bob_wins_l78_78379

theorem probability_bob_wins (P_lose : ℝ) (P_tie : ℝ) (h1 : P_lose = 5/8) (h2 : P_tie = 1/8) :
  (1 - P_lose - P_tie) = 1/4 :=
by
  sorry

end probability_bob_wins_l78_78379


namespace num_rectangles_in_5x5_grid_l78_78974

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l78_78974


namespace systematic_sample_first_segment_number_l78_78380

theorem systematic_sample_first_segment_number :
  ∃ a_1 : ℕ, ∀ d k : ℕ, k = 5 → a_1 + (59 - 1) * k = 293 → a_1 = 3 :=
by
  sorry

end systematic_sample_first_segment_number_l78_78380


namespace branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l78_78918

theorem branch_A_grade_A_probability : 
  let total_A := 100
  let grade_A_A := 40
  (grade_A_A / total_A) = 0.4 := by
  sorry

theorem branch_B_grade_A_probability : 
  let total_B := 100
  let grade_A_B := 28
  (grade_A_B / total_B) = 0.28 := by
  sorry

theorem branch_A_average_profit :
  let freq_A_A := 0.4
  let freq_A_B := 0.2
  let freq_A_C := 0.2
  let freq_A_D := 0.2
  let process_cost_A := 25
  let profit_A := (90 - process_cost_A) * freq_A_A + (50 - process_cost_A) * freq_A_B + (20 - process_cost_A) * freq_A_C + (-50 - process_cost_A) * freq_A_D
  profit_A = 15 := by
  sorry

theorem branch_B_average_profit :
  let freq_B_A := 0.28
  let freq_B_B := 0.17
  let freq_B_C := 0.34
  let freq_B_D := 0.21
  let process_cost_B := 20
  let profit_B := (90 - process_cost_B) * freq_B_A + (50 - process_cost_B) * freq_B_B + (20 - process_cost_B) * freq_B_C + (-50 - process_cost_B) * freq_B_D
  profit_B = 10 := by
  sorry

theorem choose_branch :
  let profit_A := 15
  let profit_B := 10
  profit_A > profit_B -> "Branch A"

end branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l78_78918


namespace max_showers_l78_78000

open Nat

variable (household water_limit water_for_drinking_and_cooking water_per_shower pool_length pool_width pool_height water_per_cubic_foot pool_leakage_rate days_in_july : ℕ)

def volume_of_pool (length width height: ℕ): ℕ :=
  length * width * height

def water_usage (drinking cooking pool leakage: ℕ): ℕ :=
  drinking + cooking + pool + leakage

theorem max_showers (h1: water_limit = 1000)
                    (h2: water_for_drinking_and_cooking = 100)
                    (h3: water_per_shower = 20)
                    (h4: pool_length = 10)
                    (h5: pool_width = 10)
                    (h6: pool_height = 6)
                    (h7: water_per_cubic_foot = 1)
                    (h8: pool_leakage_rate = 5)
                    (h9: days_in_july = 31) : 
  (water_limit - water_usage water_for_drinking_and_cooking
                                  (volume_of_pool pool_length pool_width pool_height) 
                                  ((pool_leakage_rate * days_in_july))) / water_per_shower = 7 := by
  sorry

end max_showers_l78_78000


namespace find_solutions_l78_78700

theorem find_solutions (x y : ℝ) :
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2 ∧ x^2 * y = 20 * x^2 + 3 * y^2) ↔ 
    (x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2) :=
by sorry

end find_solutions_l78_78700


namespace weight_of_new_student_l78_78298

theorem weight_of_new_student (W : ℝ) (x : ℝ) (h1 : 5 * W - 92 + x = 5 * (W - 4)) : x = 72 :=
sorry

end weight_of_new_student_l78_78298


namespace evaluate_rr2_l78_78320

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := (x - 3) * (x - 2)

theorem evaluate_rr2 : r (r 2) = 6 :=
by
  -- proof goes here
  sorry

end evaluate_rr2_l78_78320


namespace max_a_value_l78_78959

theorem max_a_value : 
  (∀ (x : ℝ), (x - 1) * x - (a - 2) * (a + 1) ≥ 1) → a ≤ 3 / 2 :=
sorry

end max_a_value_l78_78959


namespace chord_length_of_intersecting_circle_and_line_l78_78208

-- Define the conditions in Lean
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def line_equation (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ - 1 = 0

-- Define the problem to prove the length of the chord
theorem chord_length_of_intersecting_circle_and_line 
  (ρ θ : ℝ) (hC : circle_equation ρ θ) (hL : line_equation ρ θ) : 
  ∃ l : ℝ, l = 2 * Real.sqrt 3 :=
by 
  sorry

end chord_length_of_intersecting_circle_and_line_l78_78208


namespace anand_income_l78_78567

theorem anand_income
  (x y : ℕ)
  (h1 : 5 * x - 3 * y = 800)
  (h2 : 4 * x - 2 * y = 800) : 
  5 * x = 2000 := 
sorry

end anand_income_l78_78567


namespace epidemic_control_indicator_l78_78939

-- Definitions for conditions
def avg_le_n (seq : Fin 7 → ℕ) (n : ℕ) : Prop := (∑ i, seq i) / 7 ≤ n
def stdev_le_n (seq : Fin 7 → ℕ) (n : ℕ) : Prop := 
  let mean := (∑ i, seq i) / 7 in
  let variance := (∑ i, (seq i - mean) ^ 2) / 7 in
  (variance ^ (1/2 : ℝ)) ≤ n
def range_le_n (seq : Fin 7 → ℕ) (n : ℕ) : Prop := 
  (∑ i, max (seq i) - min (seq i)) ≤ n
def mode_1 (seq : Fin 7 → ℕ) : Prop := 
  (1 ≤ ∑ i, ite (seq i = 1) 1 0)

-- Theorem statement based on the problem
theorem epidemic_control_indicator :
  ∀ seq : Fin 7 → ℕ,
    (range_le_n seq 2 ∧ avg_le_n seq 3) ∨ 
    (mode_1 seq ∧ range_le_n seq 4) ↔
    (∀ i, seq i ≤ 5) := 
by
-- providing a statement with sorry to indicate that proof is required
sorry

end epidemic_control_indicator_l78_78939


namespace gcd_288_123_l78_78289

-- Define the conditions
def cond1 : 288 = 2 * 123 + 42 := by sorry
def cond2 : 123 = 2 * 42 + 39 := by sorry
def cond3 : 42 = 39 + 3 := by sorry
def cond4 : 39 = 13 * 3 := by sorry

-- Prove that GCD of 288 and 123 is 3
theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end gcd_288_123_l78_78289


namespace exceed_1000_cents_l78_78826

def total_amount (n : ℕ) : ℕ :=
  3 * (3 ^ n - 1) / (3 - 1)

theorem exceed_1000_cents : 
  ∃ n : ℕ, total_amount n ≥ 1000 ∧ (n + 7) % 7 = 6 := 
by
  sorry

end exceed_1000_cents_l78_78826


namespace find_j_l78_78616

def original_number (a b k : ℕ) : ℕ := 10 * a + b
def sum_of_digits (a b : ℕ) : ℕ := a + b
def modified_number (b a : ℕ) : ℕ := 20 * b + a

theorem find_j
  (a b k j : ℕ)
  (h1 : original_number a b k = k * sum_of_digits a b)
  (h2 : modified_number b a = j * sum_of_digits a b) :
  j = (199 + k) / 10 :=
sorry

end find_j_l78_78616


namespace raised_bed_area_correct_l78_78316

def garden_length : ℝ := 220
def garden_width : ℝ := 120
def garden_area : ℝ := garden_length * garden_width
def tilled_land_area : ℝ := garden_area / 2
def remaining_area : ℝ := garden_area - tilled_land_area
def trellis_area : ℝ := remaining_area / 3
def raised_bed_area : ℝ := remaining_area - trellis_area

theorem raised_bed_area_correct : raised_bed_area = 8800 := by
  sorry

end raised_bed_area_correct_l78_78316


namespace minimum_rubles_to_reverse_chips_l78_78775

theorem minimum_rubles_to_reverse_chips (n : ℕ) (h : n = 100)
  (adjacent_cost : ℕ → ℕ → ℕ)
  (free_cost : ℕ → ℕ → Prop)
  (reverse_cost : ℕ) :
  (∀ i j, i + 1 = j → adjacent_cost i j = 1) →
  (∀ i j, i + 5 = j → free_cost i j) →
  reverse_cost = 61 :=
by
  sorry

end minimum_rubles_to_reverse_chips_l78_78775


namespace range_of_a_l78_78664

variables (a : ℝ)

theorem range_of_a (h : ∀ x : ℝ, x > 0 → 2 * x * real.log x ≥ -x^2 + a * x - 3) : a ≤ 4 := by
  sorry

end range_of_a_l78_78664


namespace unique_abs_value_of_roots_l78_78513

theorem unique_abs_value_of_roots :
  ∀ (w : ℂ), w^2 - 6 * w + 40 = 0 → (∃! z, |w| = z) :=
by
  sorry

end unique_abs_value_of_roots_l78_78513


namespace ball_bounce_height_l78_78461

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end ball_bounce_height_l78_78461


namespace geometric_sequence_sum_l78_78784

theorem geometric_sequence_sum (a : ℕ → ℝ) (S₄ : ℝ) (S₈ : ℝ) (r : ℝ) 
    (h1 : r = 2) 
    (h2 : S₄ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3)
    (h3 : S₄ = 1) 
    (h4 : S₈ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3 + a 0 * r^4 + a 0 * r^5 + a 0 * r^6 + a 0 * r^7) :
    S₈ = 17 := by
  sorry

end geometric_sequence_sum_l78_78784


namespace largest_n_with_integer_solutions_l78_78331

theorem largest_n_with_integer_solutions : ∃ n, ∀ x y1 y2 y3 y4, 
 ( ((x + 1)^2 + y1^2) = ((x + 2)^2 + y2^2) ∧  ((x + 2)^2 + y2^2) = ((x + 3)^2 + y3^2) ∧ 
  ((x + 3)^2 + y3^2) = ((x + 4)^2 + y4^2)) → (n = 3) := sorry

end largest_n_with_integer_solutions_l78_78331


namespace inequality_problem_l78_78253

-- Define the problem conditions and goal
theorem inequality_problem (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) : 
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
sorry

end inequality_problem_l78_78253


namespace count_rectangles_5x5_l78_78980

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l78_78980


namespace Sam_wins_probability_l78_78833

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l78_78833


namespace costPerUse_l78_78528

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end costPerUse_l78_78528


namespace second_player_wins_optimal_play_l78_78692

def players_take_turns : Prop := sorry
def win_condition (box_count : ℕ) : Prop := box_count = 21

theorem second_player_wins_optimal_play (boxes : Fin 11 → ℕ)
    (h_turns : players_take_turns)
    (h_win : ∀ i : Fin 11, win_condition (boxes i)) : 
    ∃ P : ℕ, P = 2 :=
sorry

end second_player_wins_optimal_play_l78_78692


namespace ratio_of_areas_l78_78275

def side_length_S : ℝ := sorry
def longer_side_R : ℝ := 1.2 * side_length_S
def shorter_side_R : ℝ := 0.8 * side_length_S
def area_S : ℝ := side_length_S ^ 2
def area_R : ℝ := longer_side_R * shorter_side_R

theorem ratio_of_areas (side_length_S : ℝ) :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l78_78275


namespace room_length_difference_l78_78527

def width := 19
def length := 20
def difference := length - width

theorem room_length_difference : difference = 1 := by
  sorry

end room_length_difference_l78_78527


namespace imo_42_problem_l78_78060

theorem imo_42_problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1 :=
sorry

end imo_42_problem_l78_78060


namespace find_last_two_digits_l78_78894

variables {z a r m l : ℕ}
variables (ZARAZA ALMAZ : ℕ)
variables (digits : char → ℕ)

-- Each character represents a unique digit
axiom zaraza_unique_digits : function.injective digits
axiom almakza_unique_digits : function.injective digits

-- The numbers
def ZARAZA := 100000 * digits 'z' + 10000 * digits 'a' + 1000 * digits 'r' + 100 * digits 'a' + 10 * digits 'z' + digits 'a'
def ALMAZ := 10000 * digits 'a' + 1000 * digits 'l' + 100 * digits 'm' + 10 * digits 'a' + digits 'z'

-- Divisibility constraints
axiom zaraza_div_by_4 : ZARAZA % 4 = 0
axiom almaz_div_by_28 : ALMAZ % 28 = 0

-- Proof Goal
theorem find_last_two_digits :
  (ZARAZA + ALMAZ) % 100 = 32 := 
sorry

end find_last_two_digits_l78_78894


namespace repeating_block_length_7_div_13_l78_78851

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l78_78851


namespace find_n_l78_78792

-- Define the polynomial function
def polynomial (n : ℤ) : ℤ :=
  n^4 + 2 * n^3 + 6 * n^2 + 12 * n + 25

-- Define the condition that n is a positive integer
def is_positive_integer (n : ℤ) : Prop :=
  n > 0

-- Define the condition that polynomial is a perfect square
def is_perfect_square (k : ℤ) : Prop :=
  ∃ m : ℤ, m^2 = k

-- The theorem we need to prove
theorem find_n (n : ℤ) (h1 : is_positive_integer n) (h2 : is_perfect_square (polynomial n)) : n = 8 :=
sorry

end find_n_l78_78792


namespace tiling_2xn_with_dominoes_l78_78670

theorem tiling_2xn_with_dominoes (n : ℕ) : 
  let u : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 2 else u (n-1) + u (n-2)
  in u n = Nat.fib (n+1) :=
by
  sorry

end tiling_2xn_with_dominoes_l78_78670


namespace find_number_l78_78731

open Nat

theorem find_number 
  (A B : ℕ) 
  (HCF : ℕ → ℕ → ℕ) 
  (LCM : ℕ → ℕ → ℕ) 
  (h1 : B = 156) 
  (h2 : HCF A B = 12) 
  (h3 : LCM A B = 312) : 
  A = 24 :=
by
  sorry

end find_number_l78_78731


namespace interest_difference_l78_78174

theorem interest_difference
  (principal : ℕ) (rate : ℚ) (time : ℕ) (interest : ℚ) (difference : ℚ)
  (h1 : principal = 600)
  (h2 : rate = 0.05)
  (h3 : time = 8)
  (h4 : interest = principal * (rate * time))
  (h5 : difference = principal - interest) :
  difference = 360 :=
by sorry

end interest_difference_l78_78174


namespace unique_lottery_ticket_number_l78_78510

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

end unique_lottery_ticket_number_l78_78510


namespace bounces_less_than_50_l78_78167

noncomputable def minBouncesNeeded (initialHeight : ℝ) (bounceFactor : ℝ) (thresholdHeight : ℝ) : ℕ :=
  ⌈(Real.log (thresholdHeight / initialHeight) / Real.log (bounceFactor))⌉₊

theorem bounces_less_than_50 :
  minBouncesNeeded 360 (3/4 : ℝ) 50 = 8 :=
by
  sorry

end bounces_less_than_50_l78_78167


namespace disproving_equation_l78_78065

theorem disproving_equation 
  (a b c d : ℚ)
  (h : a / b = c / d)
  (ha : a ≠ 0)
  (hc : c ≠ 0) : 
  a + d ≠ (a / b) * (b + c) := 
by 
  sorry

end disproving_equation_l78_78065


namespace area_of_region_ABCDEFGHIJ_l78_78413

/-- 
  Given:
  1. Region ABCDEFGHIJ consists of 13 equal squares.
  2. Region ABCDEFGHIJ is inscribed in rectangle PQRS.
  3. Point A is on line PQ, B is on line QR, E is on line RS, and H is on line SP.
  4. PQ has length 28 and QR has length 26.

  Prove that the area of region ABCDEFGHIJ is 338 square units.
-/
theorem area_of_region_ABCDEFGHIJ 
  (squares : ℕ)             -- Number of squares in region ABCDEFGHIJ
  (len_PQ len_QR : ℕ)       -- Lengths of sides PQ and QR
  (area : ℕ)                 -- Area of region ABCDEFGHIJ
  (h1 : squares = 13)
  (h2 : len_PQ = 28)
  (h3 : len_QR = 26)
  : area = 338 :=
sorry

end area_of_region_ABCDEFGHIJ_l78_78413


namespace polynomial_relation_l78_78787

variables {a b c : ℝ}

theorem polynomial_relation
  (h1: a ≠ 0) (h2: b ≠ 0) (h3: c ≠ 0) (h4: a + b + c = 0) :
  ((a^7 + b^7 + c^7)^2) / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 :=
sorry

end polynomial_relation_l78_78787


namespace red_sequence_2018th_num_l78_78229

/-- Define the sequence of red-colored numbers based on the given conditions. -/
def red_sequenced_num (n : Nat) : Nat :=
  let k := Nat.sqrt (2 * n - 1) -- estimate block number
  let block_start := if k % 2 == 0 then (k - 1)*(k - 1) else k * (k - 1) + 1
  let position_in_block := n - (k * (k - 1) / 2) - 1
  if k % 2 == 0 then block_start + 2 * position_in_block else block_start + 2 * position_in_block

/-- Statement to assert the 2018th number is 3972 -/
theorem red_sequence_2018th_num : red_sequenced_num 2018 = 3972 := by
  sorry

end red_sequence_2018th_num_l78_78229


namespace ratio_of_girls_to_boys_l78_78810

variables (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : b = g - 6) (h₂ : g + b = 36) :
  (g / gcd g b) / (b / gcd g b) = 7 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l78_78810


namespace apples_total_l78_78936

theorem apples_total
    (cecile_apples : ℕ := 15)
    (diane_apples_more : ℕ := 20) :
    (cecile_apples + (cecile_apples + diane_apples_more)) = 50 :=
by
  sorry

end apples_total_l78_78936


namespace inequality_range_of_a_l78_78958

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |2 * x - a| > x - 1) ↔ a < 3 ∨ a > 5 :=
by
  sorry

end inequality_range_of_a_l78_78958


namespace Walter_gets_49_bananas_l78_78679

variable (Jefferson_bananas : ℕ) (Walter_bananas : ℕ) (total_bananas : ℕ) (shared_bananas : ℕ)

def problem_conditions : Prop :=
  Jefferson_bananas = 56 ∧ Walter_bananas = Jefferson_bananas - (Jefferson_bananas / 4)

theorem Walter_gets_49_bananas (h : problem_conditions) : 
  let combined_bananas := Jefferson_bananas + Walter_bananas in
  let shared_bananas := combined_bananas / 2 in
  shared_bananas = 49 :=
by
  sorry

end Walter_gets_49_bananas_l78_78679


namespace reduced_price_is_25_l78_78297

def original_price (P : ℝ) (X : ℝ) (R : ℝ) : Prop :=
  R = 0.85 * P ∧ 
  500 = X * P ∧ 
  500 = (X + 3) * R

theorem reduced_price_is_25 (P X R : ℝ) (h : original_price P X R) :
  R = 25 :=
by
  sorry

end reduced_price_is_25_l78_78297


namespace clock_angle_7_30_l78_78446

theorem clock_angle_7_30 :
  let hour_mark_angle := 30
  let minute_mark_angle := 6
  let hour_hand_angle := 7 * hour_mark_angle + (30 * hour_mark_angle / 60)
  let minute_hand_angle := 30 * minute_mark_angle
  let angle_diff := abs (hour_hand_angle - minute_hand_angle)
  angle_diff = 45 := by
  sorry

end clock_angle_7_30_l78_78446


namespace sam_wins_l78_78830

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l78_78830


namespace arrange_abc_l78_78641

noncomputable def a : ℝ := Real.log (4) / Real.log (0.3)
noncomputable def b : ℝ := Real.log (0.2) / Real.log (0.3)
noncomputable def c : ℝ := (1 / Real.exp 1) ^ Real.pi

theorem arrange_abc (a := a) (b := b) (c := c) : b > c ∧ c > a := by
  sorry

end arrange_abc_l78_78641


namespace jake_work_hours_l78_78675

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end jake_work_hours_l78_78675


namespace find_smaller_number_l78_78863

noncomputable def smaller_number (x y : ℝ) := y

theorem find_smaller_number 
  (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x + y = 46) :
  smaller_number x y = 18.5 :=
sorry

end find_smaller_number_l78_78863


namespace find_blue_highlighters_l78_78375

theorem find_blue_highlighters
(h_pink : P = 9)
(h_yellow : Y = 8)
(h_total : T = 22)
(h_sum : P + Y + B = T) :
  B = 5 :=
by
  -- Proof would go here
  sorry

end find_blue_highlighters_l78_78375


namespace stuffed_animal_cost_l78_78930

theorem stuffed_animal_cost
  (M S A C : ℝ)
  (h1 : M = 3 * S)
  (h2 : M = (1/2) * A)
  (h3 : C = (1/2) * A)
  (h4 : C = 2 * S)
  (h5 : M = 6) :
  A = 8 :=
by
  sorry

end stuffed_animal_cost_l78_78930


namespace minimum_volume_for_safety_l78_78917

noncomputable def pressure_is_inversely_proportional_to_volume (k V : ℝ) : ℝ :=
  k / V

-- Given conditions
def k := 8000 * 3
def p (V : ℝ) := pressure_is_inversely_proportional_to_volume k V
def balloon_will_explode (V : ℝ) : Prop := p V > 40000

-- Goal: To ensure the balloon does not explode, the volume V must be at least 0.6 m^3
theorem minimum_volume_for_safety : ∀ V : ℝ, (¬ balloon_will_explode V) → V ≥ 0.6 :=
by
  intro V
  unfold balloon_will_explode p pressure_is_inversely_proportional_to_volume
  intro h
  sorry

end minimum_volume_for_safety_l78_78917


namespace number_of_packages_l78_78403

-- Given conditions
def totalMarkers : ℕ := 40
def markersPerPackage : ℕ := 5

-- Theorem: Calculate the number of packages
theorem number_of_packages (totalMarkers: ℕ) (markersPerPackage: ℕ) : totalMarkers / markersPerPackage = 8 :=
by 
  sorry

end number_of_packages_l78_78403


namespace longer_side_of_rectangle_l78_78922

theorem longer_side_of_rectangle 
  (radius : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) 
  (h1 : radius = 6)
  (h2 : A_rectangle = 3 * (π * radius^2))
  (h3 : shorter_side = 2 * 2 * radius) :
  (A_rectangle / shorter_side) = 4.5 * π :=
by
  sorry

end longer_side_of_rectangle_l78_78922


namespace fully_simplify_expression_l78_78205

theorem fully_simplify_expression :
  (3 + 4 + 5 + 6) / 2 + (3 * 6 + 9) / 3 = 18 :=
by
  sorry

end fully_simplify_expression_l78_78205


namespace repeating_block_length_7_div_13_l78_78850

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l78_78850


namespace sum_of_all_possible_values_of_N_with_equation_l78_78278

def satisfiesEquation (N : ℝ) : Prop :=
  N * (N - 4) = -7

theorem sum_of_all_possible_values_of_N_with_equation :
  (∀ N, satisfiesEquation N → N + (4 - N) = 4) :=
sorry

end sum_of_all_possible_values_of_N_with_equation_l78_78278


namespace third_term_of_sequence_l78_78207

theorem third_term_of_sequence (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = (1 / 2) * a n + (1 / (2 * n))) : a 3 = 3 / 4 := by
  sorry

end third_term_of_sequence_l78_78207


namespace matts_weight_l78_78549

theorem matts_weight (protein_per_powder_rate : ℝ)
                     (weekly_intake_powder : ℝ)
                     (daily_protein_required_per_kg : ℝ)
                     (days_in_week : ℝ)
                     (expected_weight : ℝ)
    (h1 : protein_per_powder_rate = 0.8)
    (h2 : weekly_intake_powder = 1400)
    (h3 : daily_protein_required_per_kg = 2)
    (h4 : days_in_week = 7)
    (h5 : expected_weight = 80) :
    (weekly_intake_powder / days_in_week) * protein_per_powder_rate / daily_protein_required_per_kg = expected_weight := by
  sorry

end matts_weight_l78_78549


namespace correct_equation_l78_78442

theorem correct_equation (x : ℝ) :
  232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l78_78442


namespace correct_addition_result_l78_78905

-- Definitions corresponding to the conditions
def mistaken_addend := 240
def correct_addend := 420
def incorrect_sum := 390

-- The proof statement
theorem correct_addition_result : 
  (incorrect_sum - mistaken_addend + correct_addend) = 570 :=
by
  sorry

end correct_addition_result_l78_78905


namespace y_intercept_of_line_l78_78326

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (0, 4) = (0, y) :=
by { intro h,
     have y_eq : y = 4,
     { 
       sorry
     },
     have : (0, y) = (0, 4),
     { 
       sorry 
     },
     exact this }

end y_intercept_of_line_l78_78326


namespace time_spent_on_marketing_posts_l78_78484

-- Bryan's conditions
def hours_customer_outreach : ℕ := 4
def hours_advertisement : ℕ := hours_customer_outreach / 2
def total_hours_worked : ℕ := 8

-- Proof statement: Bryan spends 2 hours each day on marketing posts
theorem time_spent_on_marketing_posts : 
  total_hours_worked - (hours_customer_outreach + hours_advertisement) = 2 := by
  sorry

end time_spent_on_marketing_posts_l78_78484


namespace not_p_is_sufficient_but_not_necessary_for_q_l78_78949

-- Definitions for the conditions
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Definition of ¬p based on the solution derived
def not_p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The theorem statement
theorem not_p_is_sufficient_but_not_necessary_for_q :
  ∀ x : ℝ, (not_p x → q x) ∧ ¬(q x → not_p x) := sorry

end not_p_is_sufficient_but_not_necessary_for_q_l78_78949


namespace find_x_between_0_and_180_l78_78031

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l78_78031


namespace range_of_m_l78_78518

variable {x m : ℝ}

def absolute_value_inequality (x m : ℝ) : Prop := |x + 1| - |x - 2| > m

theorem range_of_m : (∀ x : ℝ, absolute_value_inequality x m) ↔ m < -3 :=
by
  sorry

end range_of_m_l78_78518


namespace number_of_rectangles_in_grid_l78_78995

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l78_78995


namespace how_many_tickets_left_l78_78624

-- Define the conditions
def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define the total tickets won by Tom
def total_tickets : ℕ := tickets_from_whack_a_mole + tickets_from_skee_ball

-- State the theorem to be proved: how many tickets Tom has left
theorem how_many_tickets_left : total_tickets - tickets_spent_on_hat = 50 := by
  sorry

end how_many_tickets_left_l78_78624


namespace general_term_formula_l78_78644

-- Define the problem parameters
variables (a : ℤ)

-- Definitions based on the conditions
def first_term : ℤ := a - 1
def second_term : ℤ := a + 1
def third_term : ℤ := 2 * a + 3

-- Define the theorem to prove the general term formula
theorem general_term_formula :
  2 * (first_term a + 1) = first_term a + third_term a → a = 0 →
  ∀ n : ℕ, a_n = 2 * n - 3 := 
by
  intro h1 h2
  sorry

end general_term_formula_l78_78644


namespace mass_percentage_Al_in_Al2O3_l78_78577

-- Define the atomic masses and formula unit
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

-- Define the statement for the mass percentage of Al in Al2O3
theorem mass_percentage_Al_in_Al2O3 : (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100 = 52.91 :=
by
  sorry -- Proof to be filled in

end mass_percentage_Al_in_Al2O3_l78_78577


namespace solve_for_x_l78_78512

noncomputable def log_b (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (b x : ℝ) (hb : b > 1) (hx : x > 0) :
  (4 * x) ^ log_b b 4 - (5 * x) ^ log_b b 5 + x = 0 ↔ x = 1 :=
by
  -- Proof placeholder
  sorry

end solve_for_x_l78_78512


namespace track_length_l78_78817

variable {x : ℕ}

-- Conditions
def runs_distance_jacob (x : ℕ) := 120
def runs_distance_liz (x : ℕ) := (x / 2 - 120)

def runs_second_meeting_jacob (x : ℕ) := x + 120 -- Jacob's total distance by second meeting
def runs_second_meeting_liz (x : ℕ) := (x / 2 + 60) -- Liz's total distance by second meeting

-- The relationship is simplified into the final correct answer
theorem track_length (h1 : 120 / (x / 2 - 120) = (x / 2 + 60) / 180) :
  x = 340 := 
sorry

end track_length_l78_78817


namespace find_g_one_l78_78791

variable {α : Type} [AddGroup α]

def is_odd (f : α → α) : Prop :=
∀ x, f (-x) = - f x

def is_even (g : α → α) : Prop :=
∀ x, g (-x) = g x

theorem find_g_one
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end find_g_one_l78_78791


namespace num_rectangles_grid_l78_78961

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l78_78961


namespace total_calories_consumed_l78_78169

def caramel_cookies := 10
def caramel_calories := 18

def chocolate_chip_cookies := 8
def chocolate_chip_calories := 22

def peanut_butter_cookies := 7
def peanut_butter_calories := 24

def selected_caramel_cookies := 5
def selected_chocolate_chip_cookies := 3
def selected_peanut_butter_cookies := 2

theorem total_calories_consumed : 
  (selected_caramel_cookies * caramel_calories) + 
  (selected_chocolate_chip_cookies * chocolate_chip_calories) + 
  (selected_peanut_butter_cookies * peanut_butter_calories) = 204 := 
by
  sorry

end total_calories_consumed_l78_78169


namespace moving_circle_trajectory_l78_78801

-- Define the two given circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- The theorem statement
theorem moving_circle_trajectory :
  (∀ x y : ℝ, (exists r : ℝ, r > 0 ∧ ∃ M : ℝ × ℝ, 
  (C₁ M.1 M.2 ∧ ((M.1 - 4)^2 + M.2^2 = (13 - r)^2) ∧
  C₂ M.1 M.2 ∧ ((M.1 + 4)^2 + M.2^2 = (r + 3)^2)) ∧
  ((x = M.1) ∧ (y = M.2))) ↔ (x^2 / 64 + y^2 / 48 = 1)) := sorry

end moving_circle_trajectory_l78_78801


namespace smallest_positive_period_symmetry_axis_range_of_f_l78_78073
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_symmetry_axis_range_of_f_l78_78073


namespace words_per_hour_after_two_hours_l78_78535

theorem words_per_hour_after_two_hours 
  (total_words : ℕ) (initial_rate : ℕ) (initial_time : ℕ) (start_time_before_deadline : ℕ) 
  (words_written_in_first_phase : ℕ) (remaining_words : ℕ) (remaining_time : ℕ)
  (final_rate_per_hour : ℕ) :
  total_words = 1200 →
  initial_rate = 400 →
  initial_time = 2 →
  start_time_before_deadline = 4 →
  words_written_in_first_phase = initial_rate * initial_time →
  remaining_words = total_words - words_written_in_first_phase →
  remaining_time = start_time_before_deadline - initial_time →
  final_rate_per_hour = remaining_words / remaining_time →
  final_rate_per_hour = 200 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end words_per_hour_after_two_hours_l78_78535


namespace total_red_marbles_l78_78245

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end total_red_marbles_l78_78245


namespace man_can_lift_one_box_each_hand_l78_78471

theorem man_can_lift_one_box_each_hand : 
  ∀ (people boxes : ℕ), people = 7 → boxes = 14 → (boxes / people) / 2 = 1 :=
by
  intros people boxes h_people h_boxes
  sorry

end man_can_lift_one_box_each_hand_l78_78471


namespace three_digit_number_div_by_11_l78_78068

theorem three_digit_number_div_by_11 (x y z n : ℕ) 
  (hx : 0 < x ∧ x < 10) 
  (hy : 0 ≤ y ∧ y < 10) 
  (hz : 0 ≤ z ∧ z < 10) 
  (hn : n = 100 * x + 10 * y + z) 
  (hq : (n / 11) = x + y + z) : 
  n = 198 :=
by
  sorry

end three_digit_number_div_by_11_l78_78068


namespace player1_points_after_13_rotations_l78_78877

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l78_78877


namespace player_1_points_after_13_rotations_l78_78886

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l78_78886


namespace sqrt_meaningful_condition_l78_78284

theorem sqrt_meaningful_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (1 - x)) → x ≤ 1 :=
by
  assume h,
  sorry

end sqrt_meaningful_condition_l78_78284


namespace find_x_between_0_and_180_l78_78034

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l78_78034


namespace evaluate_expression_l78_78194

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 :=
by
  -- sorry is used to skip the proof
  sorry

end evaluate_expression_l78_78194


namespace sale_in_first_month_l78_78468

theorem sale_in_first_month 
  (sale_month_2 : ℕ)
  (sale_month_3 : ℕ)
  (sale_month_4 : ℕ)
  (sale_month_5 : ℕ)
  (required_sale_month_6 : ℕ)
  (average_sale_6_months : ℕ)
  (total_sale_6_months : ℕ)
  (total_known_sales : ℕ)
  (sale_first_month : ℕ) : 
    sale_month_2 = 3920 →
    sale_month_3 = 3855 →
    sale_month_4 = 4230 →
    sale_month_5 = 3560 →
    required_sale_month_6 = 2000 →
    average_sale_6_months = 3500 →
    total_sale_6_months = 6 * average_sale_6_months →
    total_known_sales = sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 →
    total_sale_6_months - (total_known_sales + required_sale_month_6) = sale_first_month →
    sale_first_month = 3435 :=
by
  intros h2 h3 h4 h5 h6 h_avg h_total h_known h_calc
  sorry

end sale_in_first_month_l78_78468


namespace area_of_T_l78_78822

open Complex Real

noncomputable def omega := -1 / 2 + (1 / 2) * Complex.I * Real.sqrt 3
noncomputable def omega2 := -1 / 2 - (1 / 2) * Complex.I * Real.sqrt 3

def inT (z : ℂ) (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 ∧
  0 ≤ b ∧ b ≤ 1 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  z = a + b * omega + c * omega2

theorem area_of_T : ∃ A : ℝ, A = 2 * Real.sqrt 3 :=
sorry

end area_of_T_l78_78822


namespace solve_tan_equation_l78_78041

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l78_78041


namespace sqrt_meaningful_condition_l78_78285

theorem sqrt_meaningful_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (1 - x)) → x ≤ 1 :=
by
  assume h,
  sorry

end sqrt_meaningful_condition_l78_78285


namespace doubling_profit_condition_l78_78752

-- Definitions
def purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_items_sold : ℝ := 30
def profit_per_item (selling_price : ℝ) : ℝ := selling_price - purchase_price
def daily_profit (selling_price : ℝ) (items_sold : ℝ) : ℝ := profit_per_item selling_price * items_sold
def increase_in_items_sold_per_yuan (reduction : ℝ) : ℝ := 3 * reduction

-- Condition: Initial daily profit
def initial_daily_profit : ℝ := daily_profit initial_selling_price initial_items_sold

-- Proof problem
theorem doubling_profit_condition (reduction : ℝ) :
  daily_profit (initial_selling_price - reduction) (initial_items_sold + increase_in_items_sold_per_yuan reduction) = 2 * initial_daily_profit :=
sorry

end doubling_profit_condition_l78_78752


namespace complement_intersection_l78_78800

noncomputable def U : Set Real := Set.univ
noncomputable def M : Set Real := { x : Real | Real.log x < 0 }
noncomputable def N : Set Real := { x : Real | (1 / 2) ^ x ≥ Real.sqrt (1 / 2) }

theorem complement_intersection (U M N : Set Real) : 
  (Set.compl M ∩ N) = Set.Iic 0 :=
by
  sorry

end complement_intersection_l78_78800


namespace convert_speed_kmh_to_ms_l78_78170

-- Define the given speed in km/h
def speed_kmh : ℝ := 1.1076923076923078

-- Define the conversion factor from km/h to m/s
def conversion_factor : ℝ := 3.6

-- State the theorem
theorem convert_speed_kmh_to_ms (s : ℝ) (h : s = speed_kmh) : (s / conversion_factor) = 0.3076923076923077 := by
  -- Skip the proof as instructed
  sorry

end convert_speed_kmh_to_ms_l78_78170


namespace find_x_tan_eq_l78_78012

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l78_78012


namespace conditional_probability_l78_78304

def prob_event_A : ℚ := 7 / 8 -- Probability of event A (at least one occurrence of tails)
def prob_event_AB : ℚ := 3 / 8 -- Probability of both events A and B happening (at least one occurrence of tails and exactly one occurrence of heads)

theorem conditional_probability (prob_A : ℚ) (prob_AB : ℚ) 
  (h1: prob_A = 7 / 8) (h2: prob_AB = 3 / 8) : 
  (prob_AB / prob_A) = 3 / 7 := 
by
  rw [h1, h2]
  norm_num

end conditional_probability_l78_78304


namespace min_keychains_to_reach_profit_l78_78760

theorem min_keychains_to_reach_profit :
  let cost_per_keychain := 0.15
  let sell_price_per_keychain := 0.45
  let total_keychains := 1200
  let target_profit := 180
  let total_cost := total_keychains * cost_per_keychain
  let total_revenue := total_cost + target_profit
  let min_keychains_to_sell := total_revenue / sell_price_per_keychain
  min_keychains_to_sell = 800 := 
by
  sorry

end min_keychains_to_reach_profit_l78_78760


namespace find_lambda_l78_78789

variables {a b : ℝ} (lambda : ℝ)

-- Conditions
def orthogonal (x y : ℝ) : Prop := x * y = 0
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 3
def is_perpendicular (x y : ℝ) : Prop := x * y = 0

-- Proof statement
theorem find_lambda (h₁ : orthogonal a b)
  (h₂ : magnitude_a = 2)
  (h₃ : magnitude_b = 3)
  (h₄ : is_perpendicular (3 * a + 2 * b) (lambda * a - b)) :
  lambda = 3 / 2 :=
sorry

end find_lambda_l78_78789


namespace find_x_l78_78017

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l78_78017


namespace value_of_y_l78_78368

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 :=
sorry

end value_of_y_l78_78368


namespace kaylin_age_l78_78684

theorem kaylin_age : 
  ∀ (Freyja Eli Sarah Kaylin : ℕ), 
    Freyja = 10 ∧ 
    Eli = Freyja + 9 ∧ 
    Sarah = 2 * Eli ∧ 
    Kaylin = Sarah - 5 -> 
    Kaylin = 33 :=
by
  intro Freyja Eli Sarah Kaylin
  intro h
  cases h with hF h1
  cases h1 with hE h2
  cases h2 with hS hK
  sorry

end kaylin_age_l78_78684


namespace A_plus_B_eq_93_l78_78397

-- Definitions and conditions
def gcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)
def lcm (a b c : ℕ) : ℕ := a * b * c / (gcf a b c)

-- Values for A and B
def A := gcf 18 30 45
def B := lcm 18 30 45

-- Proof statement
theorem A_plus_B_eq_93 : A + B = 93 := by
  sorry

end A_plus_B_eq_93_l78_78397


namespace max_minus_min_on_interval_l78_78374

def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_minus_min_on_interval (a : ℝ) :
  let M := max (f 0 a) (f 3 a)
  let N := f 1 a
  M - N = 20 :=
by
  sorry

end max_minus_min_on_interval_l78_78374


namespace M_equals_N_l78_78080

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {y | 0 ≤ y}

-- State the main proof goal
theorem M_equals_N : M = N :=
by
  sorry

end M_equals_N_l78_78080


namespace main_theorem_l78_78111

variable (f : ℝ → ℝ)

-- Conditions: f(x) > f'(x) for all x ∈ ℝ
def condition (x : ℝ) : Prop := f x > (derivative f) x

-- Main statement to prove
theorem main_theorem  (h : ∀ x : ℝ, condition f x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := 
by 
  sorry

end main_theorem_l78_78111


namespace triangle_interior_angle_at_least_one_leq_60_l78_78829

theorem triangle_interior_angle_at_least_one_leq_60 {α β γ : ℝ} :
  α + β + γ = 180 →
  (α > 60 ∧ β > 60 ∧ γ > 60) → false :=
by
  intro hsum hgt
  have hα : α > 60 := hgt.1
  have hβ : β > 60 := hgt.2.1
  have hγ : γ > 60 := hgt.2.2
  have h_total: α + β + γ > 60 + 60 + 60 := add_lt_add (add_lt_add hα hβ) hγ
  linarith

end triangle_interior_angle_at_least_one_leq_60_l78_78829
