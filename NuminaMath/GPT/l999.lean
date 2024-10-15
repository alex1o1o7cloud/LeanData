import Mathlib

namespace NUMINAMATH_GPT_right_triangle_leg_square_l999_99901

theorem right_triangle_leg_square (a c b : ℕ) (h1 : c = a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = c + a :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_leg_square_l999_99901


namespace NUMINAMATH_GPT_calculate_abs_mul_l999_99935

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end NUMINAMATH_GPT_calculate_abs_mul_l999_99935


namespace NUMINAMATH_GPT_find_z_l999_99915

theorem find_z (x y z : ℝ) (h1 : y = 3 * x - 5) (h2 : z = 3 * x + 3) (h3 : y = 1) : z = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_z_l999_99915


namespace NUMINAMATH_GPT_complement_U_M_correct_l999_99997

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}
def complement_U_M : Set ℕ := {1, 2, 3}

theorem complement_U_M_correct : U \ M = complement_U_M :=
  by sorry

end NUMINAMATH_GPT_complement_U_M_correct_l999_99997


namespace NUMINAMATH_GPT_find_special_four_digit_number_l999_99965

theorem find_special_four_digit_number :
  ∃ (N : ℕ), 
  (N % 131 = 112) ∧ 
  (N % 132 = 98) ∧ 
  (1000 ≤ N) ∧ 
  (N < 10000) ∧ 
  (N = 1946) :=
sorry

end NUMINAMATH_GPT_find_special_four_digit_number_l999_99965


namespace NUMINAMATH_GPT_percent_less_l999_99938

theorem percent_less (w u y z : ℝ) (P : ℝ) (hP : P = 0.40)
  (h1 : u = 0.60 * y)
  (h2 : z = 0.54 * y)
  (h3 : z = 1.50 * w) :
  w = (1 - P) * u := 
sorry

end NUMINAMATH_GPT_percent_less_l999_99938


namespace NUMINAMATH_GPT_last_digit_of_one_over_729_l999_99985

def last_digit_of_decimal_expansion (n : ℕ) : ℕ := (n % 10)

theorem last_digit_of_one_over_729 : last_digit_of_decimal_expansion (1 / 729) = 9 :=
sorry

end NUMINAMATH_GPT_last_digit_of_one_over_729_l999_99985


namespace NUMINAMATH_GPT_sin_neg_p_l999_99963

theorem sin_neg_p (a : ℝ) : (¬ ∃ x : ℝ, Real.sin x > a) → (a ≥ 1) := 
by
  sorry

end NUMINAMATH_GPT_sin_neg_p_l999_99963


namespace NUMINAMATH_GPT_no_square_ends_in_4444_l999_99951

theorem no_square_ends_in_4444:
  ∀ (a k : ℕ), (a ^ 2 = 1000 * k + 444) → (∃ b m n : ℕ, (b = 500 * n + 38) ∨ (b = 500 * n - 38) → (a = 2 * b) →
  (a ^ 2 ≠ 1000 * m + 4444)) :=
by
  sorry

end NUMINAMATH_GPT_no_square_ends_in_4444_l999_99951


namespace NUMINAMATH_GPT_rectangle_perimeter_l999_99956

theorem rectangle_perimeter (a b : ℝ) (h1 : (a + 3) * (b + 3) = a * b + 48) : 
  2 * (a + 3 + b + 3) = 38 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l999_99956


namespace NUMINAMATH_GPT_triangle_pentagon_side_ratio_l999_99925

theorem triangle_pentagon_side_ratio :
  let triangle_perimeter := 60
  let pentagon_perimeter := 60
  let triangle_side := triangle_perimeter / 3
  let pentagon_side := pentagon_perimeter / 5
  (triangle_side : ℕ) / (pentagon_side : ℕ) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_pentagon_side_ratio_l999_99925


namespace NUMINAMATH_GPT_total_baseball_fans_l999_99906

-- Conditions given
def ratio_YM (Y M : ℕ) : Prop := 2 * Y = 3 * M
def ratio_MR (M R : ℕ) : Prop := 4 * R = 5 * M
def M_value : ℕ := 88

-- Prove total number of baseball fans
theorem total_baseball_fans (Y M R : ℕ) (h1 : ratio_YM Y M) (h2 : ratio_MR M R) (hM : M = M_value) :
  Y + M + R = 330 :=
sorry

end NUMINAMATH_GPT_total_baseball_fans_l999_99906


namespace NUMINAMATH_GPT_medians_formula_l999_99939

noncomputable def ma (a b c : ℝ) : ℝ := (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2))
noncomputable def mb (a b c : ℝ) : ℝ := (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2))
noncomputable def mc (a b c : ℝ) : ℝ := (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2))

theorem medians_formula (a b c : ℝ) :
  ma a b c = (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2)) ∧
  mb a b c = (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2)) ∧
  mc a b c = (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2)) :=
by sorry

end NUMINAMATH_GPT_medians_formula_l999_99939


namespace NUMINAMATH_GPT_no_injective_function_satisfying_conditions_l999_99950

open Real

theorem no_injective_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)
  ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x : ℝ, f (x ^ 2) - (f (a * x + b)) ^ 2 ≥ 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_no_injective_function_satisfying_conditions_l999_99950


namespace NUMINAMATH_GPT_nested_fraction_evaluation_l999_99947

def nested_expression := 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))

theorem nested_fraction_evaluation : nested_expression = 8 / 21 := by
  sorry

end NUMINAMATH_GPT_nested_fraction_evaluation_l999_99947


namespace NUMINAMATH_GPT_half_of_4_pow_2022_is_2_pow_4043_l999_99930

theorem half_of_4_pow_2022_is_2_pow_4043 :
  (4 ^ 2022) / 2 = 2 ^ 4043 :=
by sorry

end NUMINAMATH_GPT_half_of_4_pow_2022_is_2_pow_4043_l999_99930


namespace NUMINAMATH_GPT_percentage_of_720_equals_356_point_4_l999_99940

theorem percentage_of_720_equals_356_point_4 : 
  let part := 356.4
  let whole := 720
  (part / whole) * 100 = 49.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_720_equals_356_point_4_l999_99940


namespace NUMINAMATH_GPT_cost_price_A_min_cost_bshelves_l999_99928

-- Define the cost price of type B bookshelf
def costB_bshelf : ℝ := 300

-- Define the cost price of type A bookshelf
def costA_bshelf : ℝ := 1.2 * costB_bshelf

-- Define the total number of bookshelves
def total_bshelves : ℕ := 60

-- Define the condition for type A and type B bookshelves count
def typeBshelves := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves
def typeBshelves_constraints := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves ≤ 2 * typeAshelves

-- Define the equation for the costs
noncomputable def total_cost (typeAshelves : ℕ) : ℝ :=
  360 * typeAshelves + 300 * (total_bshelves - typeAshelves)

-- Define the goal: cost price of type A bookshelf is 360 yuan
theorem cost_price_A : costA_bshelf = 360 :=
by 
  sorry

-- Define the goal: the school should buy 20 type A bookshelves and 40 type B bookshelves to minimize cost
theorem min_cost_bshelves : ∃ typeAshelves : ℕ, typeAshelves = 20 ∧ typeBshelves typeAshelves = 40 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_A_min_cost_bshelves_l999_99928


namespace NUMINAMATH_GPT_unique_solution_to_function_equation_l999_99905

theorem unique_solution_to_function_equation (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (2 * n) = 2 * f n)
  (h2 : ∀ n : ℕ, f (2 * n + 1) = 2 * f n + 1) :
  ∀ n : ℕ, f n = n :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_to_function_equation_l999_99905


namespace NUMINAMATH_GPT_area_of_triangle_l999_99917

open Matrix

def a : Matrix (Fin 2) (Fin 1) ℤ := ![![4], ![-1]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![5]]

theorem area_of_triangle : (abs (a 0 0 * b 1 0 - a 1 0 * b 0 0) : ℚ) / 2 = 23 / 2 :=
by
  -- To be proved (using :ℚ for the cast to rational for division)
  sorry

end NUMINAMATH_GPT_area_of_triangle_l999_99917


namespace NUMINAMATH_GPT_simplify_expression_l999_99900

theorem simplify_expression (x y z : ℝ) : - (x - (y - z)) = -x + y - z := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l999_99900


namespace NUMINAMATH_GPT_find_pairs_l999_99989

theorem find_pairs (n k : ℕ) (h1 : (10^(k-1) ≤ n^n) ∧ (n^n < 10^k)) (h2 : (10^(n-1) ≤ k^k) ∧ (k^k < 10^n)) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l999_99989


namespace NUMINAMATH_GPT_max_cos2_sinx_l999_99913

noncomputable def cos2_sinx (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_cos2_sinx : ∃ x : ℝ, cos2_sinx x = 5 / 4 := 
by
  existsi (Real.arcsin (-1 / 2))
  rw [cos2_sinx]
  -- We need further steps to complete the proof
  sorry

end NUMINAMATH_GPT_max_cos2_sinx_l999_99913


namespace NUMINAMATH_GPT_simple_interest_principal_l999_99979

theorem simple_interest_principal (A r t : ℝ) (ht_pos : t > 0) (hr_pos : r > 0) (hA_pos : A > 0) :
  (A = 1120) → (r = 0.08) → (t = 2.4) → ∃ (P : ℝ), abs (P - 939.60) < 0.01 :=
by
  intros hA hr ht
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_simple_interest_principal_l999_99979


namespace NUMINAMATH_GPT_sam_time_to_cover_distance_l999_99927

/-- Define the total distance between points A and B as the sum of distances from A to C and C to B -/
def distance_A_to_C : ℕ := 600
def distance_C_to_B : ℕ := 400
def speed_sam : ℕ := 50
def distance_A_to_B : ℕ := distance_A_to_C + distance_C_to_B

theorem sam_time_to_cover_distance :
  let time := distance_A_to_B / speed_sam
  time = 20 := 
by
  sorry

end NUMINAMATH_GPT_sam_time_to_cover_distance_l999_99927


namespace NUMINAMATH_GPT_blocks_added_l999_99975

theorem blocks_added (a b : Nat) (h₁ : a = 86) (h₂ : b = 95) : b - a = 9 :=
by
  sorry

end NUMINAMATH_GPT_blocks_added_l999_99975


namespace NUMINAMATH_GPT_find_b_l999_99960

-- Define functions p and q
def p (x : ℝ) : ℝ := 3 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

-- Set the target value for p(q(3))
def target_val : ℝ := 9

-- Prove that b = 22/3
theorem find_b (b : ℝ) : p (q 3 b) = target_val → b = 22 / 3 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_l999_99960


namespace NUMINAMATH_GPT_good_goods_not_cheap_is_sufficient_condition_l999_99966

theorem good_goods_not_cheap_is_sufficient_condition
  (goods_good : Prop)
  (goods_not_cheap : Prop)
  (h : goods_good → goods_not_cheap) :
  (goods_good → goods_not_cheap) :=
by
  exact h

end NUMINAMATH_GPT_good_goods_not_cheap_is_sufficient_condition_l999_99966


namespace NUMINAMATH_GPT_one_cow_one_bag_days_l999_99916

-- Definitions based on conditions in a)
def cows : ℕ := 60
def bags : ℕ := 75
def days_total : ℕ := 45

-- Main statement for the proof problem
theorem one_cow_one_bag_days : 
  (cows : ℝ) * (bags : ℝ) / (days_total : ℝ) = 1 / 36 := 
by
  sorry   -- Proof placeholder

end NUMINAMATH_GPT_one_cow_one_bag_days_l999_99916


namespace NUMINAMATH_GPT_saving_time_for_downpayment_l999_99945

def annual_salary : ℚ := 150000
def saving_rate : ℚ := 0.10
def house_cost : ℚ := 450000
def downpayment_rate : ℚ := 0.20

theorem saving_time_for_downpayment : 
  (downpayment_rate * house_cost) / (saving_rate * annual_salary) = 6 :=
by
  sorry

end NUMINAMATH_GPT_saving_time_for_downpayment_l999_99945


namespace NUMINAMATH_GPT_Vanya_correct_answers_l999_99996

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end NUMINAMATH_GPT_Vanya_correct_answers_l999_99996


namespace NUMINAMATH_GPT_tan_diff_identity_l999_99974

theorem tan_diff_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β + π / 4) = 1 / 4) :
  Real.tan (α - π / 4) = 3 / 22 :=
sorry

end NUMINAMATH_GPT_tan_diff_identity_l999_99974


namespace NUMINAMATH_GPT_sine_of_pi_minus_alpha_l999_99976

theorem sine_of_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 3) : Real.sin (π - α) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sine_of_pi_minus_alpha_l999_99976


namespace NUMINAMATH_GPT_range_u_of_given_condition_l999_99902

theorem range_u_of_given_condition (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  1 ≤ |2 * x + y - 4| + |3 - x - 2 * y| ∧ |2 * x + y - 4| + |3 - x - 2 * y| ≤ 13 := 
sorry

end NUMINAMATH_GPT_range_u_of_given_condition_l999_99902


namespace NUMINAMATH_GPT_retirement_total_correct_l999_99908

-- Definitions of the conditions
def hire_year : Nat := 1986
def hire_age : Nat := 30
def retirement_year : Nat := 2006

-- Calculation of age and years of employment at retirement
def employment_duration : Nat := retirement_year - hire_year
def age_at_retirement : Nat := hire_age + employment_duration

-- The required total of age and years of employment for retirement
def total_required_for_retirement : Nat := age_at_retirement + employment_duration

-- The theorem to be proven
theorem retirement_total_correct :
  total_required_for_retirement = 70 :=
  by 
  sorry

end NUMINAMATH_GPT_retirement_total_correct_l999_99908


namespace NUMINAMATH_GPT_B_time_to_finish_race_l999_99977

theorem B_time_to_finish_race (t : ℝ) 
  (race_distance : ℝ := 130)
  (A_time : ℝ := 36)
  (A_beats_B_by : ℝ := 26)
  (A_speed : ℝ := race_distance / A_time) 
  (B_distance_when_A_finishes : ℝ := race_distance - A_beats_B_by) 
  (B_speed := B_distance_when_A_finishes / t) :
  B_speed * (t - A_time) = A_beats_B_by → t = 48 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_B_time_to_finish_race_l999_99977


namespace NUMINAMATH_GPT_equivalent_proof_problem_l999_99929

lemma condition_1 (a b : ℝ) (h : b > 0 ∧ 0 > a) : (1 / a) < (1 / b) :=
sorry

lemma condition_2 (a b : ℝ) (h : 0 > a ∧ a > b) : (1 / b) > (1 / a) :=
sorry

lemma condition_4 (a b : ℝ) (h : a > b ∧ b > 0) : (1 / b) > (1 / a) :=
sorry

theorem equivalent_proof_problem (a b : ℝ) :
  (b > 0 ∧ 0 > a → (1 / a) < (1 / b)) ∧
  (0 > a ∧ a > b → (1 / b) > (1 / a)) ∧
  (a > b ∧ b > 0 → (1 / b) > (1 / a)) :=
by {
  exact ⟨condition_1 a b, condition_2 a b, condition_4 a b⟩
}

end NUMINAMATH_GPT_equivalent_proof_problem_l999_99929


namespace NUMINAMATH_GPT_cubic_identity_l999_99982

theorem cubic_identity (x : ℝ) (h : x + 1/x = -6) : x^3 + 1/x^3 = -198 := 
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l999_99982


namespace NUMINAMATH_GPT_eval_expression_l999_99967

theorem eval_expression : 4 * (8 - 3) - 6 = 14 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l999_99967


namespace NUMINAMATH_GPT_sum_of_edges_equals_74_l999_99944

def V (pyramid : ℕ) : ℕ := pyramid

def E (pyramid : ℕ) : ℕ := 2 * (V pyramid - 1)

def sum_of_edges (pyramid1 pyramid2 pyramid3 : ℕ) : ℕ :=
  E pyramid1 + E pyramid2 + E pyramid3

theorem sum_of_edges_equals_74 (V₁ V₂ V₃ : ℕ) (h : V₁ + V₂ + V₃ = 40) :
  sum_of_edges V₁ V₂ V₃ = 74 :=
sorry

end NUMINAMATH_GPT_sum_of_edges_equals_74_l999_99944


namespace NUMINAMATH_GPT_cory_needs_22_weeks_l999_99994

open Nat

def cory_birthday_money : ℕ := 100 + 45 + 20
def bike_cost : ℕ := 600
def weekly_earning : ℕ := 20

theorem cory_needs_22_weeks : ∃ x : ℕ, cory_birthday_money + x * weekly_earning ≥ bike_cost ∧ x = 22 := by
  sorry

end NUMINAMATH_GPT_cory_needs_22_weeks_l999_99994


namespace NUMINAMATH_GPT_total_number_of_pieces_paper_l999_99973

-- Define the number of pieces of paper each person picked up
def olivia_pieces : ℝ := 127.5
def edward_pieces : ℝ := 345.25
def sam_pieces : ℝ := 518.75

-- Define the total number of pieces of paper picked up
def total_pieces : ℝ := olivia_pieces + edward_pieces + sam_pieces

-- The theorem to be proven
theorem total_number_of_pieces_paper :
  total_pieces = 991.5 :=
by
  -- Sorry is used as we are not required to provide a proof here
  sorry

end NUMINAMATH_GPT_total_number_of_pieces_paper_l999_99973


namespace NUMINAMATH_GPT_prime_sum_20_to_30_l999_99987

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end NUMINAMATH_GPT_prime_sum_20_to_30_l999_99987


namespace NUMINAMATH_GPT_artifacts_per_wing_l999_99919

theorem artifacts_per_wing
  (total_wings : ℕ)
  (num_paintings : ℕ)
  (num_artifacts : ℕ)
  (painting_wings : ℕ)
  (large_paintings_wings : ℕ)
  (small_paintings_wings : ℕ)
  (small_paintings_per_wing : ℕ)
  (artifact_wings : ℕ)
  (wings_division : total_wings = painting_wings + artifact_wings)
  (paintings_division : painting_wings = large_paintings_wings + small_paintings_wings)
  (num_large_paintings : large_paintings_wings = 2)
  (num_small_paintings : small_paintings_wings * small_paintings_per_wing = num_paintings - large_paintings_wings)
  (num_artifact_calc : num_artifacts = 8 * num_paintings)
  (artifact_wings_div : artifact_wings = total_wings - painting_wings)
  (artifact_calc : num_artifacts / artifact_wings = 66) :
  num_artifacts / artifact_wings = 66 := 
by
  sorry

end NUMINAMATH_GPT_artifacts_per_wing_l999_99919


namespace NUMINAMATH_GPT_probability_two_girls_from_twelve_l999_99972

theorem probability_two_girls_from_twelve : 
  let total_members := 12
  let boys := 4
  let girls := 8
  let choose_two_total := Nat.choose total_members 2
  let choose_two_girls := Nat.choose girls 2
  let probability := (choose_two_girls : ℚ) / (choose_two_total : ℚ)
  probability = (14 / 33) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_two_girls_from_twelve_l999_99972


namespace NUMINAMATH_GPT_frequency_rate_identity_l999_99993

theorem frequency_rate_identity (n : ℕ) : 
  (36 : ℕ) / (n : ℕ) = (0.25 : ℝ) → 
  n = 144 := by
  sorry

end NUMINAMATH_GPT_frequency_rate_identity_l999_99993


namespace NUMINAMATH_GPT_exponent_calculation_l999_99909

theorem exponent_calculation : (-1 : ℤ) ^ 53 + (2 : ℤ) ^ (5 ^ 3 - 2 ^ 3 + 3 ^ 2) = 2 ^ 126 - 1 :=
by 
  sorry

end NUMINAMATH_GPT_exponent_calculation_l999_99909


namespace NUMINAMATH_GPT_slices_with_only_mushrooms_l999_99955

theorem slices_with_only_mushrooms :
  ∀ (T P M n : ℕ),
    T = 16 →
    P = 9 →
    M = 12 →
    (9 - n) + (12 - n) + n = 16 →
    M - n = 7 :=
by
  intros T P M n hT hP hM h_eq
  sorry

end NUMINAMATH_GPT_slices_with_only_mushrooms_l999_99955


namespace NUMINAMATH_GPT_distance_between_trees_l999_99998

theorem distance_between_trees (num_trees: ℕ) (total_length: ℕ) (trees_at_end: ℕ) 
(h1: num_trees = 26) (h2: total_length = 300) (h3: trees_at_end = 2) :
  total_length / (num_trees - 1) = 12 :=
by sorry

end NUMINAMATH_GPT_distance_between_trees_l999_99998


namespace NUMINAMATH_GPT_conditional_probability_B_given_A_l999_99936

/-
Given a box containing 6 balls: 2 red, 2 yellow, and 2 blue.
One ball is drawn with replacement for 3 times.
Let event A be "the color of the ball drawn in the first draw is the same as the color of the ball drawn in the second draw".
Let event B be "the color of the balls drawn in all three draws is the same".
Prove that the conditional probability P(B|A) is 1/3.
-/
noncomputable def total_balls := 6
noncomputable def red_balls := 2
noncomputable def yellow_balls := 2
noncomputable def blue_balls := 2

noncomputable def event_A (n : ℕ) : ℕ := 
  3 * 2 * 2 * total_balls

noncomputable def event_AB (n : ℕ) : ℕ := 
  3 * 2 * 2 * 2

noncomputable def P_B_given_A : ℚ := 
  event_AB total_balls / event_A total_balls

theorem conditional_probability_B_given_A :
  P_B_given_A = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_conditional_probability_B_given_A_l999_99936


namespace NUMINAMATH_GPT_pencils_multiple_of_30_l999_99978

-- Defines the conditions of the problem
def num_pens : ℕ := 2010
def max_students : ℕ := 30
def equal_pens_per_student := num_pens % max_students = 0

-- Proves that the number of pencils must be a multiple of 30
theorem pencils_multiple_of_30 (P : ℕ) (h1 : equal_pens_per_student) (h2 : ∀ n, n ≤ max_students → ∃ m, n * m = num_pens) : ∃ k : ℕ, P = max_students * k :=
sorry

end NUMINAMATH_GPT_pencils_multiple_of_30_l999_99978


namespace NUMINAMATH_GPT_polygon_area_144_l999_99983

-- Given definitions
def polygon (n : ℕ) : Prop := -- definition to capture n squares arrangement
  n = 36

def is_perpendicular (sides : ℕ) : Prop := -- every pair of adjacent sides is perpendicular
  sides = 4

def all_sides_congruent (length : ℕ) : Prop := -- all sides have the same length
  true

def total_perimeter (perimeter : ℕ) : Prop := -- total perimeter of the polygon
  perimeter = 72

-- The side length s leading to polygon's perimeter
def side_length (s perimeter : ℕ) : Prop :=
  perimeter = 36 * s / 2 

-- Prove the area of polygon is 144
theorem polygon_area_144 (n sides length perimeter s: ℕ) 
    (h1 : polygon n) 
    (h2 : is_perpendicular sides) 
    (h3 : all_sides_congruent length) 
    (h4 : total_perimeter perimeter) 
    (h5 : side_length s perimeter) : 
    n * s * s = 144 := 
sorry

end NUMINAMATH_GPT_polygon_area_144_l999_99983


namespace NUMINAMATH_GPT_people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l999_99992

def f (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
  else if 9 ≤ n ∧ n ≤ 32 then 360 * 3 ^ ((n - 8) / 12) + 3000
  else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
  else 0 -- default case for unsupported values

def g (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 18 then 0
  else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
  else if 33 ≤ n ∧ n ≤ 45 then 8800
  else 0 -- default case for unsupported values

theorem people_entering_2pm_to_3pm :
  f 21 + f 22 + f 23 + f 24 = 17460 := sorry

theorem people_leaving_2pm_to_3pm :
  g 21 + g 22 + g 23 + g 24 = 9000 := sorry

theorem peak_visitors_time :
  ∀ n, 1 ≤ n ∧ n ≤ 45 → 
    (n = 28 ↔ ∀ m, 1 ≤ m ∧ m ≤ 45 → f m - g m ≤ f 28 - g 28) := sorry

end NUMINAMATH_GPT_people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l999_99992


namespace NUMINAMATH_GPT_John_to_floor_pushups_l999_99923

theorem John_to_floor_pushups:
  let days_per_week := 5
  let reps_per_day := 1
  let total_reps_per_stage := 15
  let stages := 3 -- number of stages: wall, high elevation, low elevation
  let total_days_needed := stages * total_reps_per_stage
  let total_weeks_needed := total_days_needed / days_per_week
  total_weeks_needed = 9 := by
  -- Here we will define the specifics of the proof later.
  sorry

end NUMINAMATH_GPT_John_to_floor_pushups_l999_99923


namespace NUMINAMATH_GPT_randi_has_6_more_nickels_than_peter_l999_99957

def ray_initial_cents : Nat := 175
def cents_given_peter : Nat := 30
def cents_given_randi : Nat := 2 * cents_given_peter
def nickel_worth : Nat := 5

def nickels (cents : Nat) : Nat :=
  cents / nickel_worth

def randi_more_nickels_than_peter : Prop :=
  nickels cents_given_randi - nickels cents_given_peter = 6

theorem randi_has_6_more_nickels_than_peter :
  randi_more_nickels_than_peter :=
sorry

end NUMINAMATH_GPT_randi_has_6_more_nickels_than_peter_l999_99957


namespace NUMINAMATH_GPT_range_of_a_l999_99953

def tangent_perpendicular_to_y_axis (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (3 * a * x^2 + 1 / x = 0)

theorem range_of_a : {a : ℝ | tangent_perpendicular_to_y_axis a} = {a : ℝ | a < 0} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l999_99953


namespace NUMINAMATH_GPT_probability_different_colors_l999_99980

theorem probability_different_colors
  (red_chips green_chips : ℕ)
  (total_chips : red_chips + green_chips = 10)
  (prob_red : ℚ := red_chips / 10)
  (prob_green : ℚ := green_chips / 10) :
  ((prob_red * prob_green) + (prob_green * prob_red) = 12 / 25) := by
sorry

end NUMINAMATH_GPT_probability_different_colors_l999_99980


namespace NUMINAMATH_GPT_sum_of_roots_quadratic_eq_l999_99937

variable (h : ℝ)
def quadratic_eq_roots (x : ℝ) : Prop := 6 * x^2 - 5 * h * x - 4 * h = 0

theorem sum_of_roots_quadratic_eq (x1 x2 : ℝ) (h : ℝ) 
  (h_roots : quadratic_eq_roots h x1 ∧ quadratic_eq_roots h x2) 
  (h_distinct : x1 ≠ x2) :
  x1 + x2 = 5 * h / 6 := by
sorry

end NUMINAMATH_GPT_sum_of_roots_quadratic_eq_l999_99937


namespace NUMINAMATH_GPT_area_per_cabbage_is_one_l999_99984

noncomputable def area_per_cabbage (x y : ℕ) : ℕ :=
  let num_cabbages_this_year : ℕ := 10000
  let increase_in_cabbages : ℕ := 199
  let area_this_year : ℕ := y^2
  let area_last_year : ℕ := x^2
  let area_per_cabbage : ℕ := area_this_year / num_cabbages_this_year
  area_per_cabbage

theorem area_per_cabbage_is_one (x y : ℕ) (hx : y^2 = 10000) (hy : y^2 = x^2 + 199) : area_per_cabbage x y = 1 :=
by 
  sorry

end NUMINAMATH_GPT_area_per_cabbage_is_one_l999_99984


namespace NUMINAMATH_GPT_average_class_size_l999_99942

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_class_size_l999_99942


namespace NUMINAMATH_GPT_digits_in_base_5_l999_99962

theorem digits_in_base_5 (n : ℕ) (h : n = 1234) (h_largest_power : 5^4 < n ∧ n < 5^5) : 
  ∃ digits : ℕ, digits = 5 := 
sorry

end NUMINAMATH_GPT_digits_in_base_5_l999_99962


namespace NUMINAMATH_GPT_same_leading_digit_l999_99954

theorem same_leading_digit (n : ℕ) (hn : 0 < n) : 
  (∀ a k l : ℕ, (a * 10^k < 2^n ∧ 2^n < (a+1) * 10^k) ∧ (a * 10^l < 5^n ∧ 5^n < (a+1) * 10^l) → a = 3) := 
sorry

end NUMINAMATH_GPT_same_leading_digit_l999_99954


namespace NUMINAMATH_GPT_unpainted_area_of_five_inch_board_l999_99941

def width1 : ℝ := 5
def width2 : ℝ := 6
def angle : ℝ := 45

theorem unpainted_area_of_five_inch_board : 
  ∃ (area : ℝ), area = 30 :=
by
  sorry

end NUMINAMATH_GPT_unpainted_area_of_five_inch_board_l999_99941


namespace NUMINAMATH_GPT_is_linear_equation_with_one_var_l999_99964

-- Definitions
def eqA := ∀ (x : ℝ), x^2 + 1 = 5
def eqB := ∀ (x y : ℝ), x + 2 = y - 3
def eqC := ∀ (x : ℝ), 1 / (2 * x) = 10
def eqD := ∀ (x : ℝ), x = 4

-- Theorem stating which equation represents a linear equation in one variable
theorem is_linear_equation_with_one_var : eqD :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_is_linear_equation_with_one_var_l999_99964


namespace NUMINAMATH_GPT_min_sum_first_n_terms_l999_99999

variable {a₁ d c : ℝ} (n : ℕ)

noncomputable def sum_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_first_n_terms (h₁ : ∀ x, 1/3 ≤ x ∧ x ≤ 4/5 → a₁ * x^2 + (d/2 - a₁) * x + c ≥ 0)
                              (h₂ : a₁ = -15/4 * d)
                              (h₃ : d > 0) :
                              ∃ n : ℕ, n > 0 ∧ sum_first_n_terms a₁ d n ≤ sum_first_n_terms a₁ d 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_min_sum_first_n_terms_l999_99999


namespace NUMINAMATH_GPT_scientific_notation_of_000000301_l999_99918

/--
Expressing a small number in scientific notation:
Prove that \(0.000000301\) can be written as \(3.01 \times 10^{-7}\).
-/
theorem scientific_notation_of_000000301 :
  0.000000301 = 3.01 * 10 ^ (-7) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_000000301_l999_99918


namespace NUMINAMATH_GPT_scientific_notation_of_600_million_l999_99922

theorem scientific_notation_of_600_million : 600000000 = 6 * 10^7 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_600_million_l999_99922


namespace NUMINAMATH_GPT_slope_intercept_form_correct_l999_99958

theorem slope_intercept_form_correct:
  ∀ (x y : ℝ), (2 * (x - 3) - 1 * (y + 4) = 0) → (∃ m b, y = m * x + b ∧ m = 2 ∧ b = -10) :=
by
  intro x y h
  use 2, -10
  sorry

end NUMINAMATH_GPT_slope_intercept_form_correct_l999_99958


namespace NUMINAMATH_GPT_minimum_grade_Ahmed_l999_99961

theorem minimum_grade_Ahmed (assignments : ℕ) (Ahmed_grade : ℕ) (Emily_grade : ℕ) (final_assignment_grade_Emily : ℕ) 
  (sum_grades_Emily : ℕ) (sum_grades_Ahmed : ℕ) (total_points_Ahmed : ℕ) (total_points_Emily : ℕ) :
  assignments = 9 →
  Ahmed_grade = 91 →
  Emily_grade = 92 →
  final_assignment_grade_Emily = 90 →
  sum_grades_Emily = 828 →
  sum_grades_Ahmed = 819 →
  total_points_Ahmed = sum_grades_Ahmed + 100 →
  total_points_Emily = sum_grades_Emily + final_assignment_grade_Emily →
  total_points_Ahmed > total_points_Emily :=
by
  sorry

end NUMINAMATH_GPT_minimum_grade_Ahmed_l999_99961


namespace NUMINAMATH_GPT_triangular_number_30_eq_465_perimeter_dots_30_eq_88_l999_99943

-- Definition of the 30th triangular number
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of the perimeter dots for the triangular number
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

-- Theorem to prove the 30th triangular number is 465
theorem triangular_number_30_eq_465 : triangular_number 30 = 465 := by
  sorry

-- Theorem to prove the perimeter dots for the 30th triangular number is 88
theorem perimeter_dots_30_eq_88 : perimeter_dots 30 = 88 := by
  sorry

end NUMINAMATH_GPT_triangular_number_30_eq_465_perimeter_dots_30_eq_88_l999_99943


namespace NUMINAMATH_GPT_region_area_l999_99986

theorem region_area (x y : ℝ) : 
  (|2 * x - 16| + |3 * y + 9| ≤ 6) → ∃ A, A = 72 :=
sorry

end NUMINAMATH_GPT_region_area_l999_99986


namespace NUMINAMATH_GPT_mean_of_two_equals_mean_of_three_l999_99970

theorem mean_of_two_equals_mean_of_three (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → 
  z = 25 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_mean_of_two_equals_mean_of_three_l999_99970


namespace NUMINAMATH_GPT_peter_change_left_l999_99991

theorem peter_change_left
  (cost_small : ℕ := 3)
  (cost_large : ℕ := 5)
  (total_money : ℕ := 50)
  (num_small : ℕ := 8)
  (num_large : ℕ := 5) :
  total_money - (num_small * cost_small + num_large * cost_large) = 1 :=
by
  sorry

end NUMINAMATH_GPT_peter_change_left_l999_99991


namespace NUMINAMATH_GPT_negation_of_proposition_l999_99910

theorem negation_of_proposition :
  (¬ (∃ x : ℝ, x < 0 ∧ x^2 > 0)) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l999_99910


namespace NUMINAMATH_GPT_violates_properties_l999_99946

-- Definitions from conditions
variables {a b c m : ℝ}

-- Conclusion to prove
theorem violates_properties :
  (∀ c : ℝ, ac = bc → (c ≠ 0 → a = b)) ∧ (c = 0 → ac = bc) → False :=
sorry

end NUMINAMATH_GPT_violates_properties_l999_99946


namespace NUMINAMATH_GPT_block_wall_min_blocks_l999_99981

theorem block_wall_min_blocks :
  ∃ n,
    n = 648 ∧
    ∀ (row_height wall_height block1_length block2_length wall_length: ℕ),
    row_height = 1 ∧
    wall_height = 8 ∧
    block1_length = 1 ∧
    block2_length = 3/2 ∧
    wall_length = 120 ∧
    (∀ i : ℕ, i < wall_height → ∃ k m : ℕ, k * block1_length + m * block2_length = wall_length) →
    n = (wall_height * (1 + 2 * 79))
:= by sorry

end NUMINAMATH_GPT_block_wall_min_blocks_l999_99981


namespace NUMINAMATH_GPT_zoe_total_earnings_l999_99921

theorem zoe_total_earnings
  (weeks : ℕ → ℝ)
  (weekly_hours : ℕ → ℝ)
  (wage_per_hour : ℝ)
  (h1 : weekly_hours 3 = 28)
  (h2 : weekly_hours 2 = 18)
  (h3 : weeks 3 - weeks 2 = 64.40)
  (h_same_wage : ∀ n, weeks n = weekly_hours n * wage_per_hour) :
  weeks 3 + weeks 2 = 296.24 :=
sorry

end NUMINAMATH_GPT_zoe_total_earnings_l999_99921


namespace NUMINAMATH_GPT_possible_analytical_expression_for_f_l999_99969

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.cos (2 * x))

theorem possible_analytical_expression_for_f :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, f (x - π/4) = f (-x)) ∧
  (∀ x : ℝ, π/8 < x ∧ x < π/2 → f x < f (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_possible_analytical_expression_for_f_l999_99969


namespace NUMINAMATH_GPT_probability_of_quitters_from_10_member_tribe_is_correct_l999_99914

noncomputable def probability_quitters_from_10_member_tribe : ℚ :=
  let total_contestants := 18
  let ten_member_tribe := 10
  let total_quitters := 2
  let comb (n k : ℕ) : ℕ := Nat.choose n k
  
  let total_combinations := comb total_contestants total_quitters
  let ten_tribe_combinations := comb ten_member_tribe total_quitters
  
  ten_tribe_combinations / total_combinations

theorem probability_of_quitters_from_10_member_tribe_is_correct :
  probability_quitters_from_10_member_tribe = 5 / 17 :=
  by
    sorry

end NUMINAMATH_GPT_probability_of_quitters_from_10_member_tribe_is_correct_l999_99914


namespace NUMINAMATH_GPT_q_zero_iff_arithmetic_l999_99948

-- Definitions of the terms and conditions
variables (A B q : ℝ) (hA : A ≠ 0)
def Sn (n : ℕ) : ℝ := A * n^2 + B * n + q
def arithmetic_sequence (an : ℕ → ℝ) : Prop := ∃ d a1, ∀ n, an n = a1 + n * d

-- The proof statement we need to show
theorem q_zero_iff_arithmetic (an : ℕ → ℝ) :
  (q = 0) ↔ (∃ a1 d, ∀ n, Sn A B 0 n = (d / 2) * n^2 + (a1 - d / 2) * n) :=
sorry

end NUMINAMATH_GPT_q_zero_iff_arithmetic_l999_99948


namespace NUMINAMATH_GPT_focus_of_parabola_y_eq_8x2_l999_99931

open Real

noncomputable def parabola_focus (a p : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * p))

theorem focus_of_parabola_y_eq_8x2 :
  parabola_focus 8 (1 / 16) = (0, 1 / 32) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_y_eq_8x2_l999_99931


namespace NUMINAMATH_GPT_bus_ride_difference_l999_99903

theorem bus_ride_difference :
  ∀ (Oscar_bus Charlie_bus : ℝ),
  Oscar_bus = 0.75 → Charlie_bus = 0.25 → Oscar_bus - Charlie_bus = 0.50 :=
by
  intros Oscar_bus Charlie_bus hOscar hCharlie
  rw [hOscar, hCharlie]
  norm_num

end NUMINAMATH_GPT_bus_ride_difference_l999_99903


namespace NUMINAMATH_GPT_initial_pieces_l999_99990

-- Define the conditions
def pieces_used : ℕ := 156
def pieces_left : ℕ := 744

-- Define the total number of pieces of paper Isabel bought initially
def total_pieces : ℕ := pieces_used + pieces_left

-- State the theorem that we need to prove
theorem initial_pieces (h1 : pieces_used = 156) (h2 : pieces_left = 744) : total_pieces = 900 :=
by
  sorry

end NUMINAMATH_GPT_initial_pieces_l999_99990


namespace NUMINAMATH_GPT_pencils_more_than_pens_l999_99959

theorem pencils_more_than_pens (pencils pens : ℕ) (h_ratio : 5 * pencils = 6 * pens) (h_pencils : pencils = 48) : 
  pencils - pens = 8 :=
by
  sorry

end NUMINAMATH_GPT_pencils_more_than_pens_l999_99959


namespace NUMINAMATH_GPT_spherical_to_rectangular_correct_l999_99907

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_correct_l999_99907


namespace NUMINAMATH_GPT_values_of_n_l999_99968

/-
  Given a natural number n and a target sum 100,
  we need to find if there exists a combination of adding and subtracting 1 through n
  such that the sum equals 100.

- A value k is representable as a sum or difference of 1 through n if the sum of the series
  can be manipulated to produce k.
- The sum of the first n natural numbers S_n = n * (n + 1) / 2 must be even and sufficiently large.
- The specific values that satisfy the conditions are of the form n = 15 + 4 * k or n = 16 + 4 * k.
-/

def exists_sum_to_100 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k

theorem values_of_n (n : ℕ) : exists_sum_to_100 n ↔ (∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k) :=
by { sorry }

end NUMINAMATH_GPT_values_of_n_l999_99968


namespace NUMINAMATH_GPT_range_of_a_l999_99933

theorem range_of_a (a : ℝ) (h : ∀ x, x > a → 2 * x + 2 / (x - a) ≥ 5) : a ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l999_99933


namespace NUMINAMATH_GPT_train_length_150_m_l999_99988

def speed_in_m_s (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

def length_of_train (speed_in_m_s : ℕ) (time_s : ℕ) : ℕ :=
  speed_in_m_s * time_s

theorem train_length_150_m (speed_kmh : ℕ) (time_s : ℕ) (speed_m_s : speed_in_m_s speed_kmh = 15) (time_pass_pole : time_s = 10) : length_of_train (speed_in_m_s speed_kmh) time_s = 150 := by
  sorry

end NUMINAMATH_GPT_train_length_150_m_l999_99988


namespace NUMINAMATH_GPT_jellybeans_problem_l999_99934

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end NUMINAMATH_GPT_jellybeans_problem_l999_99934


namespace NUMINAMATH_GPT_ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l999_99971

-- Problem 1
theorem ab_eq_6_pos_or_neg (a b : ℚ) (h : a * b = 6) : a + b > 0 ∨ a + b < 0 := sorry

-- Problem 2
theorem max_ab_when_sum_neg5 (a b : ℤ) (h : a + b = -5) : a * b ≤ 6 := sorry

-- Problem 3
theorem ab_lt_0_sign_of_sum (a b : ℚ) (h : a * b < 0) : (a + b > 0 ∨ a + b = 0 ∨ a + b < 0) := sorry

end NUMINAMATH_GPT_ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l999_99971


namespace NUMINAMATH_GPT_intersection_A_complementB_l999_99920

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 5, 7}
def complementB := U \ B

theorem intersection_A_complementB :
  A ∩ complementB = {2, 4, 6} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_complementB_l999_99920


namespace NUMINAMATH_GPT_emilia_cartons_total_l999_99952

theorem emilia_cartons_total (strawberries blueberries supermarket : ℕ) (total_needed : ℕ)
  (h1 : strawberries = 2)
  (h2 : blueberries = 7)
  (h3 : supermarket = 33)
  (h4 : total_needed = strawberries + blueberries + supermarket) :
  total_needed = 42 :=
sorry

end NUMINAMATH_GPT_emilia_cartons_total_l999_99952


namespace NUMINAMATH_GPT_calculate_expression_l999_99932

theorem calculate_expression :
  16 * (1/2) * 4 * (1/16) / 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l999_99932


namespace NUMINAMATH_GPT_factor_difference_of_squares_l999_99924

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y ^ 2 = (5 - 4 * y) * (5 + 4 * y) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l999_99924


namespace NUMINAMATH_GPT_minimum_value_of_function_l999_99926

theorem minimum_value_of_function (x : ℝ) (hx : x > 5 / 4) : 
  ∃ y, y = 4 * x + 1 / (4 * x - 5) ∧ y = 7 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_function_l999_99926


namespace NUMINAMATH_GPT_solve_3_pow_n_plus_55_eq_m_squared_l999_99911

theorem solve_3_pow_n_plus_55_eq_m_squared :
  ∃ (n m : ℕ), 3^n + 55 = m^2 ∧ ((n = 2 ∧ m = 8) ∨ (n = 6 ∧ m = 28)) :=
by
  sorry

end NUMINAMATH_GPT_solve_3_pow_n_plus_55_eq_m_squared_l999_99911


namespace NUMINAMATH_GPT_bananas_count_l999_99995

theorem bananas_count
    (total_fruit : ℕ)
    (apples_ratio : ℕ)
    (persimmons_ratio : ℕ)
    (apples_and_persimmons : apples_ratio * bananas + persimmons_ratio * bananas = total_fruit)
    (apples_ratio_val : apples_ratio = 4)
    (persimmons_ratio_val : persimmons_ratio = 3)
    (total_fruit_value : total_fruit = 210) :
    bananas = 30 :=
by
  sorry

end NUMINAMATH_GPT_bananas_count_l999_99995


namespace NUMINAMATH_GPT_sequence_bound_l999_99904

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end NUMINAMATH_GPT_sequence_bound_l999_99904


namespace NUMINAMATH_GPT_rate_is_900_l999_99949

noncomputable def rate_per_square_meter (L W : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (L * W)

theorem rate_is_900 :
  rate_per_square_meter 5 4.75 21375 = 900 := by
  sorry

end NUMINAMATH_GPT_rate_is_900_l999_99949


namespace NUMINAMATH_GPT_find_number_l999_99912

theorem find_number (n : ℝ) (h : n / 0.06 = 16.666666666666668) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l999_99912
