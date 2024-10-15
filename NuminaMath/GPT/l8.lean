import Mathlib

namespace NUMINAMATH_GPT_aku_mother_packages_l8_814

theorem aku_mother_packages
  (friends : Nat)
  (cookies_per_package : Nat)
  (cookies_per_child : Nat)
  (total_children : Nat)
  (birthday : Nat)
  (H_friends : friends = 4)
  (H_cookies_per_package : cookies_per_package = 25)
  (H_cookies_per_child : cookies_per_child = 15)
  (H_total_children : total_children = friends + 1)
  (H_birthday : birthday = 10) :
  (total_children * cookies_per_child) / cookies_per_package = 3 :=
by
  sorry

end NUMINAMATH_GPT_aku_mother_packages_l8_814


namespace NUMINAMATH_GPT_stratified_sampling_third_year_l8_891

-- The total number of students in the school
def total_students : ℕ := 2000

-- The probability of selecting a female student from the second year
def prob_female_second_year : ℚ := 0.19

-- The number of students to be selected through stratified sampling
def sample_size : ℕ := 100

-- The total number of third-year students
def third_year_students : ℕ := 500

-- The number of students to be selected from the third year in stratified sampling
def third_year_sample (total : ℕ) (third_year : ℕ) (sample : ℕ) : ℕ :=
  sample * third_year / total

-- Lean statement expressing the goal
theorem stratified_sampling_third_year :
  third_year_sample total_students third_year_students sample_size = 25 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_year_l8_891


namespace NUMINAMATH_GPT_fraction_proof_l8_854

variables (m n p q : ℚ)

theorem fraction_proof
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 :=
by sorry

end NUMINAMATH_GPT_fraction_proof_l8_854


namespace NUMINAMATH_GPT_numerical_identity_l8_828

theorem numerical_identity :
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 :=
by
  -- conditions and definitions based on a) are directly used here
  sorry -- proof is not required as per instructions

end NUMINAMATH_GPT_numerical_identity_l8_828


namespace NUMINAMATH_GPT_ratio_payment_shared_side_l8_893

variable (length_side length_back : ℕ) (cost_per_foot cole_payment : ℕ)
variables (neighbor_back_contrib neighbor_left_contrib total_cost_fence : ℕ)
variables (total_cost_shared_side : ℕ)

theorem ratio_payment_shared_side
  (h1 : length_side = 9)
  (h2 : length_back = 18)
  (h3 : cost_per_foot = 3)
  (h4 : cole_payment = 72)
  (h5 : neighbor_back_contrib = (length_back / 2) * cost_per_foot)
  (h6 : total_cost_fence = (2* length_side + length_back) * cost_per_foot)
  (h7 : total_cost_shared_side = length_side * cost_per_foot)
  (h8 : cole_left_total_payment = cole_payment + neighbor_back_contrib)
  (h9 : neighbor_left_contrib = cole_left_total_payment - cole_payment):
  neighbor_left_contrib / total_cost_shared_side = 1 := 
sorry

end NUMINAMATH_GPT_ratio_payment_shared_side_l8_893


namespace NUMINAMATH_GPT_teacher_li_sheets_l8_881

theorem teacher_li_sheets (x : ℕ)
    (h1 : ∀ (n : ℕ), n = 24 → (x / 24) = ((x / 32) + 2)) :
    x = 192 := by
  sorry

end NUMINAMATH_GPT_teacher_li_sheets_l8_881


namespace NUMINAMATH_GPT_number_of_juniors_in_sample_l8_878

theorem number_of_juniors_in_sample
  (total_students : ℕ)
  (num_freshmen : ℕ)
  (num_freshmen_sampled : ℕ)
  (num_sophomores_exceeds_num_juniors_by : ℕ)
  (num_sophomores num_juniors num_juniors_sampled : ℕ)
  (h_total : total_students = 1290)
  (h_num_freshmen : num_freshmen = 480)
  (h_num_freshmen_sampled : num_freshmen_sampled = 96)
  (h_exceeds : num_sophomores_exceeds_num_juniors_by = 30)
  (h_equation : total_students - num_freshmen = num_sophomores + num_juniors)
  (h_num_sophomores : num_sophomores = num_juniors + num_sophomores_exceeds_num_juniors_by)
  (h_fraction : num_freshmen_sampled / num_freshmen = 1 / 5)
  (h_num_juniors_sampled : num_juniors_sampled = num_juniors * (num_freshmen_sampled / num_freshmen)) :
  num_juniors_sampled = 78 := by
  sorry

end NUMINAMATH_GPT_number_of_juniors_in_sample_l8_878


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l8_823

theorem batsman_average_after_17th_inning 
  (score_17 : ℕ)
  (delta_avg : ℤ)
  (n_before : ℕ)
  (initial_avg : ℤ)
  (h1 : score_17 = 74)
  (h2 : delta_avg = 3)
  (h3 : n_before = 16)
  (h4 : initial_avg = 23) :
  (initial_avg + delta_avg) = 26 := 
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l8_823


namespace NUMINAMATH_GPT_intersection_points_C1_C2_l8_890

theorem intersection_points_C1_C2 :
  (∀ t : ℝ, ∃ (ρ θ : ℝ), 
    (ρ^2 - 10 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 41 = 0) ∧ 
    (ρ = 2 * Real.cos θ) → 
    ((ρ = 2 ∧ θ = 0) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4))) :=
sorry

end NUMINAMATH_GPT_intersection_points_C1_C2_l8_890


namespace NUMINAMATH_GPT_probability_non_adjacent_two_twos_l8_845

theorem probability_non_adjacent_two_twos : 
  let digits := [2, 0, 2, 3]
  let total_arrangements := 12 - 3
  let favorable_arrangements := 5
  (favorable_arrangements / total_arrangements : ℚ) = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_non_adjacent_two_twos_l8_845


namespace NUMINAMATH_GPT_terminal_sides_y_axis_l8_829

theorem terminal_sides_y_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 2) ∨ 
  (∃ k : ℤ, α = (2 * k + 1) * Real.pi + Real.pi / 2) ↔ 
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 := 
by sorry

end NUMINAMATH_GPT_terminal_sides_y_axis_l8_829


namespace NUMINAMATH_GPT_probability_at_least_one_multiple_of_4_l8_875

theorem probability_at_least_one_multiple_of_4 :
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  (probability_at_least_one_multiple_of_4 = 528 / 1250) := 
by
  -- Define the conditions
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  sorry

end NUMINAMATH_GPT_probability_at_least_one_multiple_of_4_l8_875


namespace NUMINAMATH_GPT_remainder_when_divided_by_14_l8_815

theorem remainder_when_divided_by_14 (A : ℕ) (h1 : A % 1981 = 35) (h2 : A % 1982 = 35) : A % 14 = 7 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_14_l8_815


namespace NUMINAMATH_GPT_polynomial_sum_of_squares_l8_805

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 < P.eval x) :
  ∃ (U V : Polynomial ℝ), P = U^2 + V^2 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_of_squares_l8_805


namespace NUMINAMATH_GPT_triangle_inradius_l8_873

theorem triangle_inradius (A p r : ℝ) 
    (h1 : p = 35) 
    (h2 : A = 78.75) 
    (h3 : A = (r * p) / 2) : 
    r = 4.5 :=
sorry

end NUMINAMATH_GPT_triangle_inradius_l8_873


namespace NUMINAMATH_GPT_diet_cola_cost_l8_852

theorem diet_cola_cost (T C : ℝ) 
  (h1 : T + 6 + C = 2 * T)
  (h2 : (T + 6 + C) + T = 24) : C = 2 := 
sorry

end NUMINAMATH_GPT_diet_cola_cost_l8_852


namespace NUMINAMATH_GPT_total_cost_of_stamps_is_correct_l8_897

-- Define the costs of each type of stamp
def cost_of_stamp_A : ℕ := 34 -- cost in cents
def cost_of_stamp_B : ℕ := 52 -- cost in cents
def cost_of_stamp_C : ℕ := 73 -- cost in cents

-- Define the number of stamps Alice needs to buy
def num_stamp_A : ℕ := 4
def num_stamp_B : ℕ := 6
def num_stamp_C : ℕ := 2

-- Define the expected total cost in dollars
def expected_total_cost : ℝ := 5.94

-- State the theorem about the total cost
theorem total_cost_of_stamps_is_correct :
  ((num_stamp_A * cost_of_stamp_A) + (num_stamp_B * cost_of_stamp_B) + (num_stamp_C * cost_of_stamp_C)) / 100 = expected_total_cost :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_stamps_is_correct_l8_897


namespace NUMINAMATH_GPT_largest_whole_number_l8_822

theorem largest_whole_number (x : ℤ) : 9 * x < 200 → x ≤ 22 := by
  sorry

end NUMINAMATH_GPT_largest_whole_number_l8_822


namespace NUMINAMATH_GPT_quadratic_equation_with_given_means_l8_837

theorem quadratic_equation_with_given_means (α β : ℝ)
  (h1 : (α + β) / 2 = 8) 
  (h2 : Real.sqrt (α * β) = 12) : 
  x ^ 2 - 16 * x + 144 = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_with_given_means_l8_837


namespace NUMINAMATH_GPT_find_x_in_magic_square_l8_851

def magicSquareProof (x d e f g h S : ℕ) : Prop :=
  (x + 25 + 75 = S) ∧
  (5 + d + e = S) ∧
  (f + g + h = S) ∧
  (x + d + h = S) ∧
  (f = 95) ∧
  (d = x - 70) ∧
  (h = 170 - x) ∧
  (e = x - 145) ∧
  (x + 25 + 75 = 5 + (x - 70) + (x - 145))

theorem find_x_in_magic_square : ∃ x d e f g h S, magicSquareProof x d e f g h S ∧ x = 310 := by
  sorry

end NUMINAMATH_GPT_find_x_in_magic_square_l8_851


namespace NUMINAMATH_GPT_expand_expression_l8_820

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l8_820


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l8_806

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_l8_806


namespace NUMINAMATH_GPT_terry_daily_income_l8_861

theorem terry_daily_income (T : ℕ) (h1 : ∀ j : ℕ, j = 30) (h2 : 7 * 30 = 210) (h3 : 7 * T - 210 = 42) : T = 36 := 
by
  sorry

end NUMINAMATH_GPT_terry_daily_income_l8_861


namespace NUMINAMATH_GPT_canonical_line_eq_l8_865

-- Define the system of linear equations
def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x - 3 * y - 2 * z + 6 = 0 ∧ x - 3 * y + z + 3 = 0)

-- Define the canonical equation of the line
def canonical_equation (x y z : ℝ) : Prop :=
  (x + 3) / 9 = y / 4 ∧ (x + 3) / 9 = z / 3 ∧ y / 4 = z / 3

-- The theorem to prove equivalence
theorem canonical_line_eq : 
  ∀ (x y z : ℝ), system_of_equations x y z → canonical_equation x y z :=
by
  intros x y z H
  sorry

end NUMINAMATH_GPT_canonical_line_eq_l8_865


namespace NUMINAMATH_GPT_evaluate_expression_is_15_l8_871

noncomputable def sumOfFirstNOddNumbers (n : ℕ) : ℕ :=
  n^2

noncomputable def simplifiedExpression : ℕ :=
  sumOfFirstNOddNumbers 1 +
  sumOfFirstNOddNumbers 2 +
  sumOfFirstNOddNumbers 3 +
  sumOfFirstNOddNumbers 4 +
  sumOfFirstNOddNumbers 5

theorem evaluate_expression_is_15 : simplifiedExpression = 15 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_is_15_l8_871


namespace NUMINAMATH_GPT_factorization_problem_l8_883

theorem factorization_problem :
  (∃ (h : D), 
    (¬ ∃ (a b : ℝ) (x y : ℝ), a * (x - y) = a * x - a * y) ∧
    (¬ ∃ (x : ℝ), x^2 - 2 * x + 3 = x * (x - 2) + 3) ∧
    (¬ ∃ (x : ℝ), (x - 1) * (x + 4) = x^2 + 3 * x - 4) ∧
    (∃ (x : ℝ), x^3 - 2 * x^2 + x = x * (x - 1)^2)) :=
  sorry

end NUMINAMATH_GPT_factorization_problem_l8_883


namespace NUMINAMATH_GPT_phone_call_probability_within_four_rings_l8_899

variables (P_A P_B P_C P_D : ℝ)

-- Assuming given probabilities
def probabilities_given : Prop :=
  P_A = 0.1 ∧ P_B = 0.3 ∧ P_C = 0.4 ∧ P_D = 0.1

theorem phone_call_probability_within_four_rings (h : probabilities_given P_A P_B P_C P_D) :
  P_A + P_B + P_C + P_D = 0.9 :=
sorry

end NUMINAMATH_GPT_phone_call_probability_within_four_rings_l8_899


namespace NUMINAMATH_GPT_production_average_lemma_l8_885

theorem production_average_lemma (n : ℕ) (h1 : 50 * n + 60 = 55 * (n + 1)) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_production_average_lemma_l8_885


namespace NUMINAMATH_GPT_george_room_painting_l8_841

-- Define the number of ways to choose 2 colors out of 9 without considering the restriction
def num_ways_total : ℕ := Nat.choose 9 2

-- Define the restriction that red and pink should not be combined
def num_restricted_ways : ℕ := 1

-- Define the final number of permissible combinations
def num_permissible_combinations : ℕ := num_ways_total - num_restricted_ways

theorem george_room_painting :
  num_permissible_combinations = 35 :=
by
  sorry

end NUMINAMATH_GPT_george_room_painting_l8_841


namespace NUMINAMATH_GPT_casey_pumping_time_l8_847

theorem casey_pumping_time :
  let pump_rate := 3 -- gallons per minute
  let corn_rows := 4
  let corn_per_row := 15
  let water_per_corn := 1 / 2
  let total_corn := corn_rows * corn_per_row
  let corn_water := total_corn * water_per_corn
  let num_pigs := 10
  let water_per_pig := 4
  let pig_water := num_pigs * water_per_pig
  let num_ducks := 20
  let water_per_duck := 1 / 4
  let duck_water := num_ducks * water_per_duck
  let total_water := corn_water + pig_water + duck_water
  let time_needed := total_water / pump_rate
  time_needed = 25 :=
by
  sorry

end NUMINAMATH_GPT_casey_pumping_time_l8_847


namespace NUMINAMATH_GPT_inequality_am_gm_l8_858

theorem inequality_am_gm 
  (a b c d : ℝ) 
  (h_nonneg_a : 0 ≤ a) 
  (h_nonneg_b : 0 ≤ b) 
  (h_nonneg_c : 0 ≤ c) 
  (h_nonneg_d : 0 ≤ d) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 :=
by
  sorry


end NUMINAMATH_GPT_inequality_am_gm_l8_858


namespace NUMINAMATH_GPT_tan_half_alpha_l8_826

theorem tan_half_alpha (α : ℝ) (h1 : 180 * (Real.pi / 180) < α) 
  (h2 : α < 270 * (Real.pi / 180)) 
  (h3 : Real.sin ((270 * (Real.pi / 180)) + α) = 4 / 5) : 
  Real.tan (α / 2) = -1 / 3 :=
by 
  -- Informal note: proof would be included here.
  sorry

end NUMINAMATH_GPT_tan_half_alpha_l8_826


namespace NUMINAMATH_GPT_systematic_sampling_student_l8_884

theorem systematic_sampling_student (total_students sample_size : ℕ) 
  (h_total_students : total_students = 56)
  (h_sample_size : sample_size = 4)
  (student1 student2 student3 student4 : ℕ)
  (h_student1 : student1 = 6)
  (h_student2 : student2 = 34)
  (h_student3 : student3 = 48) :
  student4 = 20 :=
sorry

end NUMINAMATH_GPT_systematic_sampling_student_l8_884


namespace NUMINAMATH_GPT_number_of_unique_triangle_areas_l8_860

theorem number_of_unique_triangle_areas :
  ∀ (G H I J K L : ℝ) (d₁ d₂ d₃ d₄ : ℝ),
    G ≠ H → H ≠ I → I ≠ J → G ≠ I → G ≠ J →
    H ≠ J →
    G - H = 1 → H - I = 1 → I - J = 2 →
    K - L = 2 →
    d₄ = abs d₃ →
    (d₁ = abs (K - G)) ∨ (d₂ = abs (L - G)) ∨ (d₁ = d₂) →
    ∃ (areas : ℕ), 
    areas = 3 :=
by sorry

end NUMINAMATH_GPT_number_of_unique_triangle_areas_l8_860


namespace NUMINAMATH_GPT_common_ratio_geom_seq_l8_843

variable {a : ℕ → ℝ} {q : ℝ}

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * q ^ n

theorem common_ratio_geom_seq (h₁ : a 5 = 1) (h₂ : a 8 = 8) (hq : geom_seq a q) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geom_seq_l8_843


namespace NUMINAMATH_GPT_simplify_polynomial_l8_803

variable (x : ℝ)

theorem simplify_polynomial : (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l8_803


namespace NUMINAMATH_GPT_omar_rolls_l8_832

-- Define the conditions
def karen_rolls : ℕ := 229
def total_rolls : ℕ := 448

-- Define the main theorem to prove the number of rolls by Omar
theorem omar_rolls : (total_rolls - karen_rolls) = 219 := by
  sorry

end NUMINAMATH_GPT_omar_rolls_l8_832


namespace NUMINAMATH_GPT_eq_has_exactly_one_real_root_l8_811

theorem eq_has_exactly_one_real_root : ∀ x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 ↔ x = 0 :=
by
sorry

end NUMINAMATH_GPT_eq_has_exactly_one_real_root_l8_811


namespace NUMINAMATH_GPT_PU_squared_fraction_l8_821

noncomputable def compute_PU_squared : ℚ :=
  sorry -- Proof of the distance computation PU^2.

theorem PU_squared_fraction :
  ∃ (a b : ℕ), (gcd a b = 1) ∧ (compute_PU_squared = a / b) :=
  sorry -- Proof that the resulting fraction a/b is in its simplest form.

end NUMINAMATH_GPT_PU_squared_fraction_l8_821


namespace NUMINAMATH_GPT_problem_1_part1_problem_1_part2_problem_2_l8_819

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 + 2 * cos (x) ^ 2

theorem problem_1_part1 : (∃ T > 0, ∀ x, f (x + T) = f x) := sorry

theorem problem_1_part2 : (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), x < y → f x > f y) := sorry

noncomputable def S_triangle (A B C : ℝ) (a b c : ℝ) : ℝ := 1 / 2 * b * c * sin A

theorem problem_2 :
  ∀ (A B C a b c : ℝ), f A = 4 → b = 1 → S_triangle A B C a b c = sqrt 3 / 2 →
    a^2 = b^2 + c^2 - 2 * b * c * cos A → a = sqrt 3 := sorry

end NUMINAMATH_GPT_problem_1_part1_problem_1_part2_problem_2_l8_819


namespace NUMINAMATH_GPT_profit_ratio_l8_848

theorem profit_ratio (P_invest Q_invest : ℕ) (hP : P_invest = 500000) (hQ : Q_invest = 1000000) :
  (P_invest:ℚ) / Q_invest = 1 / 2 := 
  by
  rw [hP, hQ]
  norm_num

end NUMINAMATH_GPT_profit_ratio_l8_848


namespace NUMINAMATH_GPT_partition_equation_solution_l8_868

def partition (n : ℕ) : ℕ := sorry -- defining the partition function

theorem partition_equation_solution (n : ℕ) (h : partition n + partition (n + 4) = partition (n + 2) + partition (n + 3)) :
  n = 1 ∨ n = 3 ∨ n = 5 :=
sorry

end NUMINAMATH_GPT_partition_equation_solution_l8_868


namespace NUMINAMATH_GPT_range_of_m_l8_844

theorem range_of_m :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x < -1 ∨ x > 3)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l8_844


namespace NUMINAMATH_GPT_masks_purchased_in_first_batch_l8_838

theorem masks_purchased_in_first_batch
    (cost_first_batch cost_second_batch : ℝ)
    (quantity_ratio : ℝ)
    (unit_price_difference : ℝ)
    (h1 : cost_first_batch = 1600)
    (h2 : cost_second_batch = 6000)
    (h3 : quantity_ratio = 3)
    (h4 : unit_price_difference = 2) :
    ∃ x : ℝ, (cost_first_batch / x) + unit_price_difference = (cost_second_batch / (quantity_ratio * x)) ∧ x = 200 :=
by {
    sorry
}

end NUMINAMATH_GPT_masks_purchased_in_first_batch_l8_838


namespace NUMINAMATH_GPT_profit_is_correct_l8_866

-- Definitions of the conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def price_per_set : ℕ := 50
def sets_sold : ℕ := 500

-- Derived calculations
def revenue (sets_sold : ℕ) (price_per_set : ℕ) : ℕ :=
  sets_sold * price_per_set

def manufacturing_costs (initial_outlay : ℕ) (cost_per_set : ℕ) (sets_sold : ℕ) : ℕ :=
  initial_outlay + (cost_per_set * sets_sold)

def profit (revenue : ℕ) (manufacturing_costs : ℕ) : ℕ :=
  revenue - manufacturing_costs

-- Theorem stating the problem
theorem profit_is_correct : 
  profit (revenue sets_sold price_per_set) (manufacturing_costs initial_outlay cost_per_set sets_sold) = 5000 :=
by
  sorry

end NUMINAMATH_GPT_profit_is_correct_l8_866


namespace NUMINAMATH_GPT_melted_mixture_weight_l8_824

theorem melted_mixture_weight (Z C : ℝ) (h_ratio : Z / C = 9 / 11) (h_zinc : Z = 28.8) : Z + C = 64 :=
by
  sorry

end NUMINAMATH_GPT_melted_mixture_weight_l8_824


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l8_812

-- Defining a structure for the problem
structure Rectangle :=
(area : ℝ)

structure Figure :=
(area : ℝ)

-- Defining the conditions
variables (R : Rectangle) 
  (F1 F2 F3 F4 F5 : Figure)
  (overlap_area_pair : Figure → Figure → ℝ)
  (overlap_area_triple : Figure → Figure → Figure → ℝ)

-- Given conditions
axiom R_area : R.area = 1
axiom F1_area : F1.area = 0.5
axiom F2_area : F2.area = 0.5
axiom F3_area : F3.area = 0.5
axiom F4_area : F4.area = 0.5
axiom F5_area : F5.area = 0.5

-- Statements to prove
theorem part_a : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 3 / 20 := sorry
theorem part_b : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 1 / 5 := sorry
theorem part_c : ∃ (F1 F2 F3 : Figure), overlap_area_triple F1 F2 F3 ≥ 1 / 20 := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l8_812


namespace NUMINAMATH_GPT_product_of_two_numbers_l8_867

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end NUMINAMATH_GPT_product_of_two_numbers_l8_867


namespace NUMINAMATH_GPT_seats_per_section_correct_l8_839

-- Define the total number of seats
def total_seats : ℕ := 270

-- Define the number of sections
def sections : ℕ := 9

-- Define the number of seats per section
def seats_per_section (total_seats sections : ℕ) : ℕ := total_seats / sections

theorem seats_per_section_correct : seats_per_section total_seats sections = 30 := by
  sorry

end NUMINAMATH_GPT_seats_per_section_correct_l8_839


namespace NUMINAMATH_GPT_kay_weight_training_time_l8_816

variables (total_minutes : ℕ) (aerobic_ratio weight_ratio : ℕ)
-- Conditions
def kay_exercise := total_minutes = 250
def ratio_cond := aerobic_ratio = 3 ∧ weight_ratio = 2
def total_ratio_parts := aerobic_ratio + weight_ratio

-- Question and proof goal
theorem kay_weight_training_time (h1 : kay_exercise total_minutes) (h2 : ratio_cond aerobic_ratio weight_ratio) :
  (total_minutes / total_ratio_parts * weight_ratio) = 100 :=
by
  sorry

end NUMINAMATH_GPT_kay_weight_training_time_l8_816


namespace NUMINAMATH_GPT_smallest_value_4x_plus_3y_l8_876

-- Define the condition as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

-- Prove the smallest possible value of 4x + 3y given the condition
theorem smallest_value_4x_plus_3y : ∃ x y : ℝ, circle_eq x y ∧ (4 * x + 3 * y = -40) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_smallest_value_4x_plus_3y_l8_876


namespace NUMINAMATH_GPT_slope_intercept_equivalence_l8_882

-- Define the given equation in Lean
def given_line_equation (x y : ℝ) : Prop := 3 * x - 2 * y = 4

-- Define the slope-intercept form as extracted from the given line equation
def slope_intercept_form (x y : ℝ) : Prop := y = (3/2) * x - 2

-- Prove that the given line equation is equivalent to its slope-intercept form
theorem slope_intercept_equivalence (x y : ℝ) :
  given_line_equation x y ↔ slope_intercept_form x y :=
by sorry

end NUMINAMATH_GPT_slope_intercept_equivalence_l8_882


namespace NUMINAMATH_GPT_least_number_to_add_l8_887

theorem least_number_to_add (n : ℕ) (divisor : ℕ) (modulus : ℕ) (h1 : n = 1076) (h2 : divisor = 23) (h3 : n % divisor = 18) :
  modulus = divisor - (n % divisor) ∧ modulus = 5 := 
sorry

end NUMINAMATH_GPT_least_number_to_add_l8_887


namespace NUMINAMATH_GPT_max_edges_intersected_by_plane_l8_862

theorem max_edges_intersected_by_plane (p : ℕ) (h_pos : p > 0) : ℕ :=
  let vertices := 2 * p
  let base_edges := p
  let lateral_edges := p
  let total_edges := 3 * p
  total_edges

end NUMINAMATH_GPT_max_edges_intersected_by_plane_l8_862


namespace NUMINAMATH_GPT_find_rate_l8_836

noncomputable def SI := 200
noncomputable def P := 800
noncomputable def T := 4

theorem find_rate : ∃ R : ℝ, SI = (P * R * T) / 100 ∧ R = 6.25 :=
by sorry

end NUMINAMATH_GPT_find_rate_l8_836


namespace NUMINAMATH_GPT_find_abs_xyz_l8_827

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end NUMINAMATH_GPT_find_abs_xyz_l8_827


namespace NUMINAMATH_GPT_ab_range_l8_850

theorem ab_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + b + 8) : a * b ≥ 16 :=
sorry

end NUMINAMATH_GPT_ab_range_l8_850


namespace NUMINAMATH_GPT_avg_growth_rate_l8_807

theorem avg_growth_rate {a p q x : ℝ} (h_eq : (1 + p) * (1 + q) = (1 + x) ^ 2) : 
  x ≤ (p + q) / 2 := 
by
  sorry

end NUMINAMATH_GPT_avg_growth_rate_l8_807


namespace NUMINAMATH_GPT_card_sequence_probability_l8_896

noncomputable def probability_of_sequence : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem card_sequence_probability :
  probability_of_sequence = 4/33150 := 
by 
  sorry

end NUMINAMATH_GPT_card_sequence_probability_l8_896


namespace NUMINAMATH_GPT_geometric_sequence_properties_l8_872

theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) (h1 : ∀ n, S n = 3^n + t) (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 2 = 6 ∧ t = -1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l8_872


namespace NUMINAMATH_GPT_propA_necessary_but_not_sufficient_l8_809

variable {a : ℝ}

-- Proposition A: ∀ x ∈ ℝ, ax² + 2ax + 1 > 0
def propA (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

-- Proposition B: 0 < a < 1
def propB (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement: Proposition A is necessary but not sufficient for Proposition B
theorem propA_necessary_but_not_sufficient (a : ℝ) :
  (propB a → propA a) ∧
  (propA a → propB a → False) :=
by
  sorry

end NUMINAMATH_GPT_propA_necessary_but_not_sufficient_l8_809


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l8_895

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ 
           (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ 
                          a ≥ 1 ∧ a ≤ 9 ∧ 
                          b ≥ 0 ∧ b ≤ 9 ∧ 
                          c ≥ 0 ∧ c ≤ 9 ∧ 
                          (a = b ∨ b = c ∨ a = c ∨ 
                           a + b > c ∧ a + c > b ∧ b + c > a)) ∧
           n = 57 := 
sorry

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l8_895


namespace NUMINAMATH_GPT_findingRealNumsPureImaginary_l8_898

theorem findingRealNumsPureImaginary :
  ∀ x : ℝ, ((x + Complex.I * 2) * ((x + 2) + Complex.I * 2) * ((x + 4) + Complex.I * 2)).im = 0 → 
    x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5 :=
by
  intros x h
  let expr := x^3 + 6*x^2 + 4*x - 16
  have h_real_part_eq_0 : expr = 0 := sorry
  have solutions_correct :
    expr = 0 → (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) := sorry
  exact solutions_correct h_real_part_eq_0

end NUMINAMATH_GPT_findingRealNumsPureImaginary_l8_898


namespace NUMINAMATH_GPT_max_quotient_l8_859

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : (b / a) ≤ 15 :=
  sorry

end NUMINAMATH_GPT_max_quotient_l8_859


namespace NUMINAMATH_GPT_find_missing_digit_divisibility_by_4_l8_801

theorem find_missing_digit_divisibility_by_4 (x : ℕ) (h : x < 10) :
  (3280 + x) % 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_digit_divisibility_by_4_l8_801


namespace NUMINAMATH_GPT_batsman_average_after_17_matches_l8_863

theorem batsman_average_after_17_matches (A : ℕ) (h : (17 * (A + 3) = 16 * A + 87)) : A + 3 = 39 := by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17_matches_l8_863


namespace NUMINAMATH_GPT_tan_2alpha_and_cos_beta_l8_842

theorem tan_2alpha_and_cos_beta
    (α β : ℝ)
    (h1 : 0 < β ∧ β < α ∧ α < (Real.pi / 2))
    (h2 : Real.sin α = (4 * Real.sqrt 3) / 7)
    (h3 : Real.cos (β - α) = 13 / 14) :
    Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ Real.cos β = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_tan_2alpha_and_cos_beta_l8_842


namespace NUMINAMATH_GPT_radius_of_semicircular_cubicle_l8_869

noncomputable def radius_of_semicircle (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem radius_of_semicircular_cubicle :
  radius_of_semicircle 71.9822971502571 = 14 := 
sorry

end NUMINAMATH_GPT_radius_of_semicircular_cubicle_l8_869


namespace NUMINAMATH_GPT_cube_root_eq_self_l8_840

theorem cube_root_eq_self (a : ℝ) (h : a^(3:ℕ) = a) : a = 1 ∨ a = -1 ∨ a = 0 := 
sorry

end NUMINAMATH_GPT_cube_root_eq_self_l8_840


namespace NUMINAMATH_GPT_park_area_l8_856

theorem park_area (L B : ℝ) (h1 : L = B / 2) (h2 : 6 * 1000 / 60 * 6 = 2 * (L + B)) : L * B = 20000 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_park_area_l8_856


namespace NUMINAMATH_GPT_cost_of_items_l8_855

theorem cost_of_items (x y z : ℝ)
  (h1 : 20 * x + 3 * y + 2 * z = 32)
  (h2 : 39 * x + 5 * y + 3 * z = 58) :
  5 * (x + y + z) = 30 := by
  sorry

end NUMINAMATH_GPT_cost_of_items_l8_855


namespace NUMINAMATH_GPT_solution_difference_l8_888

theorem solution_difference (m n : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 24 * x - 96 ↔ x = m ∨ x = n) (h_distinct : m ≠ n) (h_order : m > n) : m - n = 16 :=
sorry

end NUMINAMATH_GPT_solution_difference_l8_888


namespace NUMINAMATH_GPT_find_x_minus_y_l8_835

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + 3 * y = 14) (h2 : x + 4 * y = 11) : x - y = 3 := by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l8_835


namespace NUMINAMATH_GPT_min_value_of_expression_l8_894

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + (1 / (a - 1)) ≥ 3 :=
by sorry

end NUMINAMATH_GPT_min_value_of_expression_l8_894


namespace NUMINAMATH_GPT_people_joined_after_leaving_l8_892

theorem people_joined_after_leaving 
  (p_initial : ℕ) (p_left : ℕ) (p_final : ℕ) (p_joined : ℕ) :
  p_initial = 30 → p_left = 10 → p_final = 25 → p_joined = p_final - (p_initial - p_left) → p_joined = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end NUMINAMATH_GPT_people_joined_after_leaving_l8_892


namespace NUMINAMATH_GPT_skateboard_total_distance_is_3720_l8_802

noncomputable def skateboard_distance : ℕ :=
  let a1 := 10
  let d := 9
  let n := 20
  let flat_time := 10
  let a_n := a1 + (n - 1) * d
  let ramp_distance := n * (a1 + a_n) / 2
  let flat_distance := a_n * flat_time
  ramp_distance + flat_distance

theorem skateboard_total_distance_is_3720 : skateboard_distance = 3720 := 
by
  sorry

end NUMINAMATH_GPT_skateboard_total_distance_is_3720_l8_802


namespace NUMINAMATH_GPT_find_flat_fee_l8_818

def flat_fee_exists (f n : ℝ) : Prop :=
  f + n = 120 ∧ f + 4 * n = 255

theorem find_flat_fee : ∃ f n, flat_fee_exists f n ∧ f = 75 := by
  sorry

end NUMINAMATH_GPT_find_flat_fee_l8_818


namespace NUMINAMATH_GPT_cinnamon_swirl_eaters_l8_886

theorem cinnamon_swirl_eaters (total_pieces : ℝ) (jane_pieces : ℝ) (equal_pieces : total_pieces / jane_pieces = 3 ) : 
  (total_pieces = 12) ∧ (jane_pieces = 4) → total_pieces / jane_pieces = 3 := 
by 
  sorry

end NUMINAMATH_GPT_cinnamon_swirl_eaters_l8_886


namespace NUMINAMATH_GPT_more_males_l8_849

theorem more_males {Total_attendees Male_attendees : ℕ} (h1 : Total_attendees = 120) (h2 : Male_attendees = 62) :
  Male_attendees - (Total_attendees - Male_attendees) = 4 :=
by
  sorry

end NUMINAMATH_GPT_more_males_l8_849


namespace NUMINAMATH_GPT_count_points_in_intersection_is_7_l8_846

def isPointInSetA (x y : ℤ) : Prop :=
  (x - 3)^2 + (y - 4)^2 ≤ (5 / 2)^2

def isPointInSetB (x y : ℤ) : Prop :=
  (x - 4)^2 + (y - 5)^2 > (5 / 2)^2

def isPointInIntersection (x y : ℤ) : Prop :=
  isPointInSetA x y ∧ isPointInSetB x y

def pointsInIntersection : List (ℤ × ℤ) :=
  [(1, 5), (1, 4), (1, 3), (2, 3), (3, 2), (3, 3), (3, 4)]

theorem count_points_in_intersection_is_7 :
  (List.length pointsInIntersection = 7)
  ∧ (∀ (p : ℤ × ℤ), p ∈ pointsInIntersection → isPointInIntersection p.fst p.snd) :=
by
  sorry

end NUMINAMATH_GPT_count_points_in_intersection_is_7_l8_846


namespace NUMINAMATH_GPT_max_omega_l8_817

open Real

-- Define the function f(x) = sin(ωx + φ)
noncomputable def f (ω φ x : ℝ) := sin (ω * x + φ)

-- ω > 0 and |φ| ≤ π / 2
def condition_omega_pos (ω : ℝ) := ω > 0
def condition_phi_bound (φ : ℝ) := abs φ ≤ π / 2

-- x = -π/4 is a zero of f(x)
def condition_zero (ω φ : ℝ) := f ω φ (-π/4) = 0

-- x = π/4 is the axis of symmetry for the graph of y = f(x)
def condition_symmetry (ω φ : ℝ) := 
  ∀ x : ℝ, f ω φ (π/4 - x) = f ω φ (π/4 + x)

-- f(x) is monotonic in the interval (π/18, 5π/36)
def condition_monotonic (ω φ : ℝ) := 
  ∀ x₁ x₂ : ℝ, π/18 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * π / 36 
  → f ω φ x₁ ≤ f ω φ x₂

-- Prove that the maximum value of ω satisfying all the conditions is 9
theorem max_omega (ω : ℝ) (φ : ℝ)
  (h1 : condition_omega_pos ω)
  (h2 : condition_phi_bound φ)
  (h3 : condition_zero ω φ)
  (h4 : condition_symmetry ω φ)
  (h5 : condition_monotonic ω φ) :
  ω ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_omega_l8_817


namespace NUMINAMATH_GPT_class_strength_l8_804

/-- The average age of an adult class is 40 years.
    12 new students with an average age of 32 years join the class,
    therefore decreasing the average by 4 years.
    What was the original strength of the class? -/
theorem class_strength (x : ℕ) (h1 : ∃ (x : ℕ), ∀ (y : ℕ), y ≠ x → y = 40) 
                       (h2 : 12 ≥ 0) (h3 : 32 ≥ 0) (h4 : (x + 12) * 36 = 40 * x + 12 * 32) : 
  x = 12 := 
sorry

end NUMINAMATH_GPT_class_strength_l8_804


namespace NUMINAMATH_GPT_average_of_three_quantities_l8_831

theorem average_of_three_quantities (a b c d e : ℝ) 
    (h1 : (a + b + c + d + e) / 5 = 8)
    (h2 : (d + e) / 2 = 14) :
    (a + b + c) / 3 = 4 := 
sorry

end NUMINAMATH_GPT_average_of_three_quantities_l8_831


namespace NUMINAMATH_GPT_proof_part1_proof_part2_l8_830

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end NUMINAMATH_GPT_proof_part1_proof_part2_l8_830


namespace NUMINAMATH_GPT_napkin_ratio_l8_825

theorem napkin_ratio (initial_napkins : ℕ) (napkins_after : ℕ) (olivia_napkins : ℕ) (amelia_napkins : ℕ)
  (h1 : initial_napkins = 15) (h2 : napkins_after = 45) (h3 : olivia_napkins = 10)
  (h4 : initial_napkins + olivia_napkins + amelia_napkins = napkins_after) :
  amelia_napkins / olivia_napkins = 2 := by
  sorry

end NUMINAMATH_GPT_napkin_ratio_l8_825


namespace NUMINAMATH_GPT_modulus_z_eq_sqrt_10_l8_857

noncomputable def z := (10 * Complex.I) / (3 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_modulus_z_eq_sqrt_10_l8_857


namespace NUMINAMATH_GPT_rent_for_each_room_l8_808

theorem rent_for_each_room (x : ℝ) (ha : 4800 / x = 4200 / (x - 30)) (hx : x = 240) :
  x = 240 ∧ (x - 30) = 210 :=
by
  sorry

end NUMINAMATH_GPT_rent_for_each_room_l8_808


namespace NUMINAMATH_GPT_number_of_special_three_digit_numbers_l8_800

theorem number_of_special_three_digit_numbers : ∃ (n : ℕ), n = 3 ∧
  (∀ (A B C : ℕ), 
    (100 * A + 10 * B + C < 1000 ∧ 100 * A + 10 * B + C ≥ 100) ∧
    B = 2 * C ∧
    B = (A + C) / 2 → 
    (A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 312 ∨ 
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 642 ∨
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 963))
:= 
sorry

end NUMINAMATH_GPT_number_of_special_three_digit_numbers_l8_800


namespace NUMINAMATH_GPT_minimum_ab_l8_877

variable (a b : ℝ)

def is_collinear (a b : ℝ) : Prop :=
  (0 - b) * (-2 - 0) = (-2 - b) * (a - 0)

theorem minimum_ab (h1 : a * b > 0) (h2 : is_collinear a b) : a * b = 16 := by
  sorry

end NUMINAMATH_GPT_minimum_ab_l8_877


namespace NUMINAMATH_GPT_deductive_reasoning_option_l8_874

inductive ReasoningType
| deductive
| inductive
| analogical

-- Definitions based on conditions
def option_A : ReasoningType := ReasoningType.inductive
def option_B : ReasoningType := ReasoningType.deductive
def option_C : ReasoningType := ReasoningType.inductive
def option_D : ReasoningType := ReasoningType.analogical

-- The main theorem to prove
theorem deductive_reasoning_option : option_B = ReasoningType.deductive :=
by sorry

end NUMINAMATH_GPT_deductive_reasoning_option_l8_874


namespace NUMINAMATH_GPT_cricket_average_increase_l8_813

-- Define the conditions as variables
variables (innings_initial : ℕ) (average_initial : ℕ) (runs_next_innings : ℕ)
variables (runs_increase : ℕ)

-- Given conditions
def conditions := (innings_initial = 13) ∧ (average_initial = 22) ∧ (runs_next_innings = 92)

-- Target: Calculate the desired increase in average (runs_increase)
theorem cricket_average_increase (h : conditions innings_initial average_initial runs_next_innings) :
  runs_increase = 5 :=
  sorry

end NUMINAMATH_GPT_cricket_average_increase_l8_813


namespace NUMINAMATH_GPT_total_number_of_athletes_l8_853

theorem total_number_of_athletes (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
sorry

end NUMINAMATH_GPT_total_number_of_athletes_l8_853


namespace NUMINAMATH_GPT_problem1_problem2_l8_880

-- Theorem for problem 1
theorem problem1 (a b : ℤ) : (a^3 * b^4) ^ 2 / (a * b^2) ^ 3 = a^3 * b^2 := 
by sorry

-- Theorem for problem 2
theorem problem2 (a : ℤ) : (-a^2) ^ 3 * a^2 + a^8 = 0 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l8_880


namespace NUMINAMATH_GPT_find_x_in_interval_l8_879

theorem find_x_in_interval (x : ℝ) : x^2 + 5 * x < 10 ↔ -5 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_find_x_in_interval_l8_879


namespace NUMINAMATH_GPT_min_max_of_quadratic_l8_810

theorem min_max_of_quadratic 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 - 6 * x + 1)
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 1) : 
  (∃ xmin, ∃ xmax, f xmin = -3 ∧ f xmax = 9 ∧ -1 ≤ xmin ∧ xmin ≤ 1 ∧ -1 ≤ xmax ∧ xmax ≤ 1) :=
sorry

end NUMINAMATH_GPT_min_max_of_quadratic_l8_810


namespace NUMINAMATH_GPT_ratio_u_v_l8_833

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_ratio_u_v_l8_833


namespace NUMINAMATH_GPT_ratio_QP_l8_834

theorem ratio_QP {P Q : ℚ} 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 4 → x ≠ -4 → 
    P / (x^2 - 5 * x) + Q / (x + 4) = (x^2 - 3 * x + 8) / (x^3 - 5 * x^2 + 4 * x)) : 
  Q / P = 7 / 2 := 
sorry

end NUMINAMATH_GPT_ratio_QP_l8_834


namespace NUMINAMATH_GPT_solve_system_l8_864

theorem solve_system : ∀ (x y : ℤ), 2 * x + y = 5 → x + 2 * y = 6 → x - y = -1 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_solve_system_l8_864


namespace NUMINAMATH_GPT_domain_of_p_l8_889

theorem domain_of_p (h : ℝ → ℝ) (h_domain : ∀ x, -10 ≤ x → x ≤ 6 → ∃ y, h x = y) :
  ∀ x, -1.2 ≤ x ∧ x ≤ 2 → ∃ y, h (-5 * x) = y :=
by
  sorry

end NUMINAMATH_GPT_domain_of_p_l8_889


namespace NUMINAMATH_GPT_range_of_a_l8_870

def condition1 (a : ℝ) : Prop := (2 - a) ^ 2 < 1
def condition2 (a : ℝ) : Prop := (3 - a) ^ 2 ≥ 1

theorem range_of_a (a : ℝ) (h1 : condition1 a) (h2 : condition2 a) :
  1 < a ∧ a ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l8_870
