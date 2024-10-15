import Mathlib

namespace NUMINAMATH_GPT_doubled_dimensions_new_volume_l869_86951

-- Define the original volume condition
def original_volume_condition (π r h : ℝ) : Prop := π * r^2 * h = 5

-- Define the new volume function after dimensions are doubled
def new_volume (π r h : ℝ) : ℝ := π * (2 * r)^2 * (2 * h)

-- The Lean statement for the proof problem 
theorem doubled_dimensions_new_volume (π r h : ℝ) (h_orig : original_volume_condition π r h) : 
  new_volume π r h = 40 :=
by 
  sorry

end NUMINAMATH_GPT_doubled_dimensions_new_volume_l869_86951


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l869_86962

noncomputable def A_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def B_n (b e : ℤ) (n : ℕ) : ℤ :=
  n * (2 * b + (n - 1) * e) / 2

theorem arithmetic_sequence_ratio (a d b e : ℤ) :
  (∀ n : ℕ, n ≠ 0 → A_n a d n / B_n b e n = (5 * n - 3) / (n + 9)) →
  (a + 5 * d) / (b + 2 * e) = 26 / 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l869_86962


namespace NUMINAMATH_GPT_sum_of_greatest_values_l869_86948

theorem sum_of_greatest_values (b : ℝ) (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 → 2.5 + 2 = 4.5 :=
by sorry

end NUMINAMATH_GPT_sum_of_greatest_values_l869_86948


namespace NUMINAMATH_GPT_hawks_points_l869_86981

theorem hawks_points (x y : ℕ) (h1 : x + y = 50) (h2 : x + 4 - y = 12) : y = 21 :=
by
  sorry

end NUMINAMATH_GPT_hawks_points_l869_86981


namespace NUMINAMATH_GPT_inequality_abc_l869_86956

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_inequality_abc_l869_86956


namespace NUMINAMATH_GPT_tv_height_l869_86915

theorem tv_height (H : ℝ) : 
  672 / (24 * H) = (1152 / (48 * 32)) + 1 → 
  H = 16 := 
by
  have h_area_first_TV : 24 * H ≠ 0 := sorry
  have h_new_condition: 1152 / (48 * 32) + 1 = 1.75 := sorry
  have h_cost_condition: 672 / (24 * H) = 1.75 := sorry
  sorry

end NUMINAMATH_GPT_tv_height_l869_86915


namespace NUMINAMATH_GPT_correct_calculation_l869_86942

theorem correct_calculation (a : ℝ) : (3 * a^3)^2 = 9 * a^6 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l869_86942


namespace NUMINAMATH_GPT_complete_the_square_d_l869_86907

theorem complete_the_square_d (x : ℝ) (h : x^2 + 6 * x + 5 = 0) : ∃ d : ℝ, (x + 3)^2 = d ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_d_l869_86907


namespace NUMINAMATH_GPT_seq_composite_l869_86965

-- Define the sequence recurrence relation
def seq (a : ℕ → ℕ) : Prop :=
  ∀ (k : ℕ), k ≥ 1 → a (k+2) = a (k+1) * a k + 1

-- Prove that for k ≥ 9, a_k - 22 is composite
theorem seq_composite (a : ℕ → ℕ) (h_seq : seq a) :
  ∀ (k : ℕ), k ≥ 9 → ∃ d, d > 1 ∧ d < a k ∧ d ∣ (a k - 22) :=
by
  sorry

end NUMINAMATH_GPT_seq_composite_l869_86965


namespace NUMINAMATH_GPT_inverse_function_correct_l869_86935

theorem inverse_function_correct :
  ( ∀ x : ℝ, (x > 1) → (∃ y : ℝ, y = 1 + Real.log (x - 1)) ↔ (∀ y : ℝ, y > 0 → (∃ x : ℝ, x = e^(y + 1) - 1))) :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_correct_l869_86935


namespace NUMINAMATH_GPT_vector_line_equation_l869_86966

open Real

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let numer := (u.1 * v.1 + u.2 * v.2)
  let denom := (v.1 * v.1 + v.2 * v.2)
  (numer * v.1 / denom, numer * v.2 / denom)

theorem vector_line_equation (x y : ℝ) :
  vector_projection (x, y) (3, 4) = (-3, -4) → 
  y = -3 / 4 * x - 25 / 4 :=
  sorry

end NUMINAMATH_GPT_vector_line_equation_l869_86966


namespace NUMINAMATH_GPT_compare_negatives_l869_86937

theorem compare_negatives : -1 < - (2 / 3) := by
  sorry

end NUMINAMATH_GPT_compare_negatives_l869_86937


namespace NUMINAMATH_GPT_area_of_shaded_region_l869_86982

-- Definitions of conditions
def center (O : Type) := O
def radius_large_circle (R : ℝ) := R
def radius_small_circle (r : ℝ) := r
def length_chord_CD (CD : ℝ) := CD = 60
def chord_tangent_to_smaller_circle (r : ℝ) (R : ℝ) := r^2 = R^2 - 900

-- Theorem for the area of the shaded region
theorem area_of_shaded_region 
(O : Type) 
(R r : ℝ) 
(CD : ℝ)
(h1 : length_chord_CD CD)
(h2 : chord_tangent_to_smaller_circle r R) : 
  π * (R^2 - r^2) = 900 * π := by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l869_86982


namespace NUMINAMATH_GPT_margie_change_is_6_25_l869_86975

-- The conditions are given as definitions in Lean
def numberOfApples : Nat := 5
def costPerApple : ℝ := 0.75
def amountPaid : ℝ := 10.00

-- The statement to be proved
theorem margie_change_is_6_25 :
  (amountPaid - (numberOfApples * costPerApple)) = 6.25 := 
  sorry

end NUMINAMATH_GPT_margie_change_is_6_25_l869_86975


namespace NUMINAMATH_GPT_min_value_x_l869_86955

theorem min_value_x (a : ℝ) (h : ∀ a > 0, x^2 ≤ 1 + a) : ∃ x, ∀ a > 0, -1 ≤ x ∧ x ≤ 1 := 
sorry

end NUMINAMATH_GPT_min_value_x_l869_86955


namespace NUMINAMATH_GPT_percent_of_total_l869_86918

theorem percent_of_total (p n : ℝ) (h1 : p = 35 / 100) (h2 : n = 360) : p * n = 126 := by
  sorry

end NUMINAMATH_GPT_percent_of_total_l869_86918


namespace NUMINAMATH_GPT_largest_of_five_consecutive_integers_l869_86972

   theorem largest_of_five_consecutive_integers (n1 n2 n3 n4 n5 : ℕ) 
     (h1: 0 < n1) (h2: n1 + 1 = n2) (h3: n2 + 1 = n3) (h4: n3 + 1 = n4)
     (h5: n4 + 1 = n5) (h6: n1 * n2 * n3 * n4 * n5 = 15120) : n5 = 10 :=
   sorry
   
end NUMINAMATH_GPT_largest_of_five_consecutive_integers_l869_86972


namespace NUMINAMATH_GPT_power_multiplication_l869_86978

theorem power_multiplication :
  (- (4 / 5 : ℚ)) ^ 2022 * (5 / 4 : ℚ) ^ 2023 = 5 / 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_power_multiplication_l869_86978


namespace NUMINAMATH_GPT_sum_of_ages_l869_86980

-- Define ages of Kiana and her twin brothers
variables (kiana_age : ℕ) (twin_age : ℕ)

-- Define conditions
def age_product_condition : Prop := twin_age * twin_age * kiana_age = 162
def age_less_than_condition : Prop := kiana_age < 10
def twins_older_condition : Prop := twin_age > kiana_age

-- The main problem statement
theorem sum_of_ages (h1 : age_product_condition twin_age kiana_age) (h2 : age_less_than_condition kiana_age) (h3 : twins_older_condition twin_age kiana_age) :
  twin_age * 2 + kiana_age = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l869_86980


namespace NUMINAMATH_GPT_sum_of_squares_of_sines_l869_86947

theorem sum_of_squares_of_sines (α : ℝ) : 
  (Real.sin α)^2 + (Real.sin (α + 60 * Real.pi / 180))^2 + (Real.sin (α + 120 * Real.pi / 180))^2 = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_sines_l869_86947


namespace NUMINAMATH_GPT_kids_at_camp_l869_86925

theorem kids_at_camp (total_stayed_home : ℕ) (difference : ℕ) (x : ℕ) 
  (h1 : total_stayed_home = 777622) 
  (h2 : difference = 574664) 
  (h3 : total_stayed_home = x + difference) : 
  x = 202958 :=
by
  sorry

end NUMINAMATH_GPT_kids_at_camp_l869_86925


namespace NUMINAMATH_GPT_population_30_3_million_is_30300000_l869_86911

theorem population_30_3_million_is_30300000 :
  let million := 1000000
  let population_1998 := 30.3 * million
  population_1998 = 30300000 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_population_30_3_million_is_30300000_l869_86911


namespace NUMINAMATH_GPT_final_expression_in_simplest_form_l869_86961

variable (x : ℝ)

theorem final_expression_in_simplest_form : 
  ((3 * x + 6 - 5 * x + 10) / 5) = (-2 / 5) * x + 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_final_expression_in_simplest_form_l869_86961


namespace NUMINAMATH_GPT_find_angle_A_find_b_c_l869_86920
open Real

-- Part I: Proving angle A
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h₁ : (a + b + c) * (b + c - a) = 3 * b * c) :
  A = π / 3 :=
by sorry

-- Part II: Proving values of b and c given a=2 and area of triangle ABC is √3
theorem find_b_c (A B C : ℝ) (a b c : ℝ) (h₁ : a = 2) (h₂ : (1 / 2) * b * c * (sin (π / 3)) = sqrt 3) :
  b = 2 ∧ c = 2 :=
by sorry

end NUMINAMATH_GPT_find_angle_A_find_b_c_l869_86920


namespace NUMINAMATH_GPT_non_neg_scalar_product_l869_86923

theorem non_neg_scalar_product (a b c d e f g h : ℝ) : 
  (0 ≤ ac + bd) ∨ (0 ≤ ae + bf) ∨ (0 ≤ ag + bh) ∨ (0 ≤ ce + df) ∨ (0 ≤ cg + dh) ∨ (0 ≤ eg + fh) :=
  sorry

end NUMINAMATH_GPT_non_neg_scalar_product_l869_86923


namespace NUMINAMATH_GPT_accurate_mass_l869_86936

variable (m1 m2 a b x : Real) -- Declare the variables

theorem accurate_mass (h1 : a * x = b * m1) (h2 : b * x = a * m2) : x = Real.sqrt (m1 * m2) := by
  -- We will prove the statement later
  sorry

end NUMINAMATH_GPT_accurate_mass_l869_86936


namespace NUMINAMATH_GPT_per_capita_income_growth_l869_86921

theorem per_capita_income_growth (x : ℝ) : 
  (250 : ℝ) * (1 + x) ^ 20 ≥ 800 →
  (250 : ℝ) * (1 + x) ^ 40 ≥ 2560 := 
by
  intros h
  -- Proof is not required, so we skip it with sorry
  sorry

end NUMINAMATH_GPT_per_capita_income_growth_l869_86921


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l869_86998

-- Axiom statements representing the conditions
axiom medians_perpendicular (A B C D E G : Type) : Prop
axiom median_ad_length (A D : Type) : Prop
axiom median_be_length (B E : Type) : Prop

-- Main theorem statement
theorem area_of_triangle_ABC
  (A B C D E G : Type)
  (h1 : medians_perpendicular A B C D E G)
  (h2 : median_ad_length A D) -- AD = 18
  (h3 : median_be_length B E) -- BE = 24
  : ∃ (area : ℝ), area = 576 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l869_86998


namespace NUMINAMATH_GPT_mixed_feed_cost_l869_86906

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed by mixing 
    one kind worth $0.18 per pound with another worth $0.53 per pound. They used 17 pounds of the cheaper kind in the mix.
    We are to prove that the cost per pound of the mixed feed is $0.36 per pound. -/
theorem mixed_feed_cost
  (total_weight : ℝ) (cheaper_cost : ℝ) (expensive_cost : ℝ) (cheaper_weight : ℝ)
  (total_weight_eq : total_weight = 35)
  (cheaper_cost_eq : cheaper_cost = 0.18)
  (expensive_cost_eq : expensive_cost = 0.53)
  (cheaper_weight_eq : cheaper_weight = 17) :
  ((cheaper_weight * cheaper_cost + (total_weight - cheaper_weight) * expensive_cost) / total_weight) = 0.36 :=
by
  sorry

end NUMINAMATH_GPT_mixed_feed_cost_l869_86906


namespace NUMINAMATH_GPT_leo_weight_l869_86945

-- Definitions from the conditions
variable (L K J M : ℝ)

-- Conditions 
def condition1 : Prop := L + 15 = 1.60 * K
def condition2 : Prop := L + 15 = 0.40 * J
def condition3 : Prop := J = K + 25
def condition4 : Prop := M = K - 20
def condition5 : Prop := L + K + J + M = 350

-- Final statement to prove based on the conditions
theorem leo_weight (h1 : condition1 L K) (h2 : condition2 L J) (h3 : condition3 J K) 
                   (h4 : condition4 M K) (h5 : condition5 L K J M) : L = 110.22 :=
by 
  sorry

end NUMINAMATH_GPT_leo_weight_l869_86945


namespace NUMINAMATH_GPT_abc_inequalities_l869_86933

noncomputable def a : ℝ := Real.log 1 / Real.log 2 - Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 2) ^ 3
noncomputable def c : ℝ := Real.sqrt 3

theorem abc_inequalities :
  a < b ∧ b < c :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_abc_inequalities_l869_86933


namespace NUMINAMATH_GPT_total_earnings_correct_l869_86995

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end NUMINAMATH_GPT_total_earnings_correct_l869_86995


namespace NUMINAMATH_GPT_trigonometric_identity_l869_86974

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (π / 2 + α) * Real.cos (π + α) = -1 / 5 :=
by
  -- The proof will be skipped but the statement should be correct.
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l869_86974


namespace NUMINAMATH_GPT_difference_of_numbers_l869_86929

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 20460) (h2 : b % 12 = 0) (h3 : b / 10 = a) : b - a = 17314 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l869_86929


namespace NUMINAMATH_GPT_alpha_minus_beta_eq_pi_div_4_l869_86930

open Real

theorem alpha_minus_beta_eq_pi_div_4 (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 4) 
(h : tan α = (cos β + sin β) / (cos β - sin β)) : α - β = π / 4 :=
sorry

end NUMINAMATH_GPT_alpha_minus_beta_eq_pi_div_4_l869_86930


namespace NUMINAMATH_GPT_phone_answer_prob_within_four_rings_l869_86939

def prob_first_ring : ℚ := 1/10
def prob_second_ring : ℚ := 1/5
def prob_third_ring : ℚ := 3/10
def prob_fourth_ring : ℚ := 1/10

theorem phone_answer_prob_within_four_rings :
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring = 7/10 :=
by
  sorry

end NUMINAMATH_GPT_phone_answer_prob_within_four_rings_l869_86939


namespace NUMINAMATH_GPT_darnel_lap_difference_l869_86959

theorem darnel_lap_difference (sprint jog : ℝ) (h_sprint : sprint = 0.88) (h_jog : jog = 0.75) : sprint - jog = 0.13 := 
by 
  rw [h_sprint, h_jog] 
  norm_num

end NUMINAMATH_GPT_darnel_lap_difference_l869_86959


namespace NUMINAMATH_GPT_total_students_l869_86927

theorem total_students (total_students_with_brown_eyes total_students_with_black_hair: ℕ)
    (h1: ∀ (total_students : ℕ), (2 * total_students_with_brown_eyes) = 3 * total_students)
    (h2: (2 * total_students_with_black_hair) = total_students_with_brown_eyes)
    (h3: total_students_with_black_hair = 6) : 
    ∃ total_students : ℕ, total_students = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l869_86927


namespace NUMINAMATH_GPT_problem_a_problem_b_unique_solution_l869_86970

-- Problem (a)

theorem problem_a (a b c n : ℤ) (hnat : 0 ≤ n) (h : a * n^2 + b * n + c = 0) : n ∣ c :=
sorry

-- Problem (b)

theorem problem_b_unique_solution : ∀ n : ℕ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = 3 :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_unique_solution_l869_86970


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_seq_l869_86952

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (m - n = 1) → (a (m + 1) - a m) = (a (n + 1) - a n)

/-- The common difference of an arithmetic sequence given certain conditions. -/
theorem common_difference_of_arithmetic_seq (a: ℕ → ℤ) (d : ℤ):
    a 1 + a 2 = 4 → 
    a 3 + a 4 = 16 →
    arithmetic_sequence a →
    (a 2 - a 1) = d → d = 3 :=
by
  intros h1 h2 h3 h4
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_seq_l869_86952


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_sum_l869_86983

theorem smallest_whole_number_larger_than_sum :
    let sum := 2 + 1 / 2 + 3 + 1 / 3 + 4 + 1 / 4 + 5 + 1 / 5 
    let smallest_whole := 16
    (sum < smallest_whole ∧ smallest_whole - 1 < sum) := 
by
    sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_sum_l869_86983


namespace NUMINAMATH_GPT_minimum_value_f_l869_86971

noncomputable def f (x : ℝ) (f1 f2 : ℝ) : ℝ :=
  f1 * x + f2 / x - 2

theorem minimum_value_f (f1 f2 : ℝ) (h1 : f2 = 2) (h2 : f1 = 3 / 2) :
  ∃ x > 0, f x f1 f2 = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_f_l869_86971


namespace NUMINAMATH_GPT_car_travel_distance_l869_86901

theorem car_travel_distance :
  ∀ (train_speed : ℝ) (fraction : ℝ) (time_minutes : ℝ) (car_speed : ℝ) (distance : ℝ),
  train_speed = 90 →
  fraction = 5 / 6 →
  time_minutes = 30 →
  car_speed = fraction * train_speed →
  distance = car_speed * (time_minutes / 60) →
  distance = 37.5 :=
by
  intros train_speed fraction time_minutes car_speed distance
  intros h_train_speed h_fraction h_time_minutes h_car_speed h_distance
  sorry

end NUMINAMATH_GPT_car_travel_distance_l869_86901


namespace NUMINAMATH_GPT_eq_iff_squared_eq_l869_86992

theorem eq_iff_squared_eq (a b : ℝ) : a = b ↔ a^2 + b^2 = 2 * a * b :=
by
  sorry

end NUMINAMATH_GPT_eq_iff_squared_eq_l869_86992


namespace NUMINAMATH_GPT_swimming_time_l869_86944

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end NUMINAMATH_GPT_swimming_time_l869_86944


namespace NUMINAMATH_GPT_sequence_general_term_l869_86950

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  ∃ a : ℕ → ℚ, (∀ n, a n = 1 / n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l869_86950


namespace NUMINAMATH_GPT_minimum_value_fraction_l869_86904

theorem minimum_value_fraction (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 2) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 2 → 
    ((1 / (1 + x)) + (1 / (2 + 2 * y)) ≥ 4 / 5)) :=
by sorry

end NUMINAMATH_GPT_minimum_value_fraction_l869_86904


namespace NUMINAMATH_GPT_relationship_among_abc_l869_86949

theorem relationship_among_abc (x : ℝ) (e : ℝ) (ln : ℝ → ℝ) (half_pow : ℝ → ℝ) (exp : ℝ → ℝ) 
  (x_in_e_e2 : x > e ∧ x < exp 2) 
  (def_a : ln x = ln x)
  (def_b : half_pow (ln x) = ((1/2)^(ln x)))
  (def_c : exp (ln x) = x):
  (exp (ln x)) > (ln x) ∧ (ln x) > ((1/2)^(ln x)) :=
by 
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l869_86949


namespace NUMINAMATH_GPT_simplify_expression_l869_86958

theorem simplify_expression (b : ℝ) (h : b ≠ 1 / 2) : 1 - (2 / (1 + (b / (1 - 2 * b)))) = (3 * b - 1) / (1 - b) :=
by
    sorry

end NUMINAMATH_GPT_simplify_expression_l869_86958


namespace NUMINAMATH_GPT_minimize_perimeter_of_sector_l869_86940

theorem minimize_perimeter_of_sector (r θ: ℝ) (h₁: (1 / 2) * θ * r^2 = 16) (h₂: 2 * r + θ * r = 2 * r + 32 / r): θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_perimeter_of_sector_l869_86940


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l869_86903

theorem arithmetic_sequence_fifth_term :
  ∀ (a d : ℤ), (a + 19 * d = 15) → (a + 20 * d = 18) → (a + 4 * d = -30) :=
by
  intros a d h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l869_86903


namespace NUMINAMATH_GPT_inequality_proof_l869_86900

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l869_86900


namespace NUMINAMATH_GPT_product_of_m_l869_86994

theorem product_of_m (m n : ℤ) (h_cond : m^2 + m + 8 = n^2) (h_nonneg : n ≥ 0) : 
  (∀ m, (∃ n, m^2 + m + 8 = n^2 ∧ n ≥ 0) → m = 7 ∨ m = -8) ∧ 
  (∃ m1 m2 : ℤ, m1 = 7 ∧ m2 = -8 ∧ (m1 * m2 = -56)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_m_l869_86994


namespace NUMINAMATH_GPT_units_digit_17_pow_2007_l869_86996

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_17_pow_2007_l869_86996


namespace NUMINAMATH_GPT_coefficient_of_friction_l869_86985

/-- Assume m, Pi and ΔL are positive real numbers, and g is the acceleration due to gravity. 
We need to prove that the coefficient of friction μ is given by Pi / (m * g * ΔL). --/
theorem coefficient_of_friction (m Pi ΔL g : ℝ) (h_m : 0 < m) (h_Pi : 0 < Pi) (h_ΔL : 0 < ΔL) (h_g : 0 < g) :
  ∃ μ : ℝ, μ = Pi / (m * g * ΔL) :=
sorry

end NUMINAMATH_GPT_coefficient_of_friction_l869_86985


namespace NUMINAMATH_GPT_farmer_brown_additional_cost_l869_86926

-- Definitions for the conditions
def originalQuantity : ℕ := 10
def originalPricePerBale : ℕ := 15
def newPricePerBale : ℕ := 18
def newQuantity : ℕ := 2 * originalQuantity

-- Definition for the target equation (additional cost)
def additionalCost : ℕ := (newQuantity * newPricePerBale) - (originalQuantity * originalPricePerBale)

-- Theorem stating the problem voiced in Lean 4
theorem farmer_brown_additional_cost : additionalCost = 210 :=
by {
  sorry
}

end NUMINAMATH_GPT_farmer_brown_additional_cost_l869_86926


namespace NUMINAMATH_GPT_ratio_h_r_bounds_l869_86919

theorem ratio_h_r_bounds
  {a b c h r : ℝ}
  (h_right_angle : a^2 + b^2 = c^2)
  (h_area1 : 1/2 * a * b = 1/2 * c * h)
  (h_area2 : 1/2 * (a + b + c) * r = 1/2 * a * b) :
  2 < h / r ∧ h / r ≤ 2.41 :=
by
  sorry

end NUMINAMATH_GPT_ratio_h_r_bounds_l869_86919


namespace NUMINAMATH_GPT_cars_between_15000_and_20000_l869_86964

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end NUMINAMATH_GPT_cars_between_15000_and_20000_l869_86964


namespace NUMINAMATH_GPT_max_minus_min_eq_32_l869_86954

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_minus_min_eq_32 : 
  let M := max (f (-3)) (max (f 3) (max (f (-2)) (f 2)))
  let m := min (f (-3)) (min (f 3) (min (f (-2)) (f 2)))
  M - m = 32 :=
by
  sorry

end NUMINAMATH_GPT_max_minus_min_eq_32_l869_86954


namespace NUMINAMATH_GPT_find_t_l869_86991

theorem find_t (s t : ℝ) (h1 : 12 * s + 7 * t = 165) (h2 : s = t + 3) : t = 6.789 := 
by 
  sorry

end NUMINAMATH_GPT_find_t_l869_86991


namespace NUMINAMATH_GPT_population_ratios_l869_86963

variable (P_X P_Y P_Z : Nat)

theorem population_ratios
  (h1 : P_Y = 2 * P_Z)
  (h2 : P_X = 10 * P_Z) : P_X / P_Y = 5 := by
  sorry

end NUMINAMATH_GPT_population_ratios_l869_86963


namespace NUMINAMATH_GPT_axis_of_symmetry_shift_l869_86986

-- Define that f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the problem statement in Lean
theorem axis_of_symmetry_shift (f : ℝ → ℝ) 
  (h_even : is_even_function f) :
  ∃ x, ∀ y, f (x + y) = f ((x - 1) + y) ∧ x = -1 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_shift_l869_86986


namespace NUMINAMATH_GPT_op_5_2_l869_86987

def op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem op_5_2 : op 5 2 = 30 := 
by sorry

end NUMINAMATH_GPT_op_5_2_l869_86987


namespace NUMINAMATH_GPT_slope_of_line_l869_86917

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 1) ∧ (y1 = 3) ∧ (x2 = 7) ∧ (y2 = -9)
  → (y2 - y1) / (x2 - x1) = -2 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_l869_86917


namespace NUMINAMATH_GPT_required_lemons_for_20_gallons_l869_86973

-- Conditions
def lemons_for_50_gallons : ℕ := 40
def gallons_for_lemons : ℕ := 50
def additional_lemons_per_10_gallons : ℕ := 1
def number_of_gallons : ℕ := 20
def base_lemons (g: ℕ) : ℕ := (lemons_for_50_gallons * g) / gallons_for_lemons
def additional_lemons (g: ℕ) : ℕ := (g / 10) * additional_lemons_per_10_gallons
def total_lemons (g: ℕ) : ℕ := base_lemons g + additional_lemons g

-- Proof statement
theorem required_lemons_for_20_gallons : total_lemons number_of_gallons = 18 :=
by
  sorry

end NUMINAMATH_GPT_required_lemons_for_20_gallons_l869_86973


namespace NUMINAMATH_GPT_find_value_divided_by_4_l869_86976

theorem find_value_divided_by_4 (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 :=
by
  sorry

end NUMINAMATH_GPT_find_value_divided_by_4_l869_86976


namespace NUMINAMATH_GPT_solve_for_a_l869_86968

theorem solve_for_a (x y a : ℝ) (h1 : 2 * x + y = 2 * a + 1) 
                    (h2 : x + 2 * y = a - 1) 
                    (h3 : x - y = 4) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l869_86968


namespace NUMINAMATH_GPT_triangle_is_right_l869_86916

variable {n : ℕ}

theorem triangle_is_right 
  (h1 : n > 1) 
  (h2 : a = 2 * n) 
  (h3 : b = n^2 - 1) 
  (h4 : c = n^2 + 1)
  : a^2 + b^2 = c^2 := 
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_triangle_is_right_l869_86916


namespace NUMINAMATH_GPT_exists_a_log_eq_l869_86979

theorem exists_a_log_eq (a : ℝ) (h : a = 10 ^ ((Real.log 2 * Real.log 3) / (Real.log 2 + Real.log 3))) :
  ∀ x > 0, Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a :=
by
  sorry

end NUMINAMATH_GPT_exists_a_log_eq_l869_86979


namespace NUMINAMATH_GPT_num_factors_36_l869_86943

theorem num_factors_36 : ∀ (n : ℕ), n = 36 → (∃ (a b : ℕ), 36 = 2^a * 3^b ∧ a = 2 ∧ b = 2 ∧ (a + 1) * (b + 1) = 9) :=
by
  sorry

end NUMINAMATH_GPT_num_factors_36_l869_86943


namespace NUMINAMATH_GPT_diagonals_sum_pentagon_inscribed_in_circle_l869_86990

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end NUMINAMATH_GPT_diagonals_sum_pentagon_inscribed_in_circle_l869_86990


namespace NUMINAMATH_GPT_max_a_condition_slope_condition_exponential_inequality_l869_86922

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 1)
noncomputable def g (x a : ℝ) := f x a + a / Real.exp x

theorem max_a_condition (a : ℝ) (h_pos : a > 0) 
  (h_nonneg : ∀ x : ℝ, f x a ≥ 0) : a ≤ 1 := sorry

theorem slope_condition (a m : ℝ) 
  (ha : a ≤ -1) 
  (h_slope : ∀ x1 x2 : ℝ, x1 ≠ x2 → 
    (g x2 a - g x1 a) / (x2 - x1) > m) : m ≤ 3 := sorry

theorem exponential_inequality (n : ℕ) (hn : n > 0) : 
  (2 * (Real.exp n - 1)) / (Real.exp 1 - 1) ≥ n * (n + 1) := sorry

end NUMINAMATH_GPT_max_a_condition_slope_condition_exponential_inequality_l869_86922


namespace NUMINAMATH_GPT_range_of_a_l869_86997

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, (0 < x) ∧ (x + 1/x < a)) ↔ a ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l869_86997


namespace NUMINAMATH_GPT_parallelogram_count_l869_86905

theorem parallelogram_count (m n : ℕ) : 
  ∃ p : ℕ, p = (m.choose 2) * (n.choose 2) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_count_l869_86905


namespace NUMINAMATH_GPT_total_amount_spent_l869_86953

-- Definitions for the conditions
def cost_magazine : ℝ := 0.85
def cost_pencil : ℝ := 0.50
def coupon_discount : ℝ := 0.35

-- The main theorem to prove
theorem total_amount_spent : cost_magazine + cost_pencil - coupon_discount = 1.00 := by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l869_86953


namespace NUMINAMATH_GPT_tangent_line_eq_l869_86977

theorem tangent_line_eq
  (x y : ℝ)
  (h : x^2 + y^2 - 4 * x = 0)
  (P : ℝ × ℝ)
  (hP : P = (1, Real.sqrt 3))
  : x - Real.sqrt 3 * y + 2 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_l869_86977


namespace NUMINAMATH_GPT_jessica_cut_21_roses_l869_86946

def initial_roses : ℕ := 2
def thrown_roses : ℕ := 4
def final_roses : ℕ := 23

theorem jessica_cut_21_roses : (final_roses - initial_roses) = 21 :=
by
  sorry

end NUMINAMATH_GPT_jessica_cut_21_roses_l869_86946


namespace NUMINAMATH_GPT_WangLi_final_score_l869_86967

def weightedFinalScore (writtenScore : ℕ) (demoScore : ℕ) (interviewScore : ℕ)
    (writtenWeight : ℕ) (demoWeight : ℕ) (interviewWeight : ℕ) : ℕ :=
  (writtenScore * writtenWeight + demoScore * demoWeight + interviewScore * interviewWeight) /
  (writtenWeight + demoWeight + interviewWeight)

theorem WangLi_final_score :
  weightedFinalScore 96 90 95 5 3 2 = 94 :=
  by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_WangLi_final_score_l869_86967


namespace NUMINAMATH_GPT_sum_of_two_pos_implies_one_pos_l869_86932

theorem sum_of_two_pos_implies_one_pos (x y : ℝ) (h : x + y > 0) : x > 0 ∨ y > 0 :=
  sorry

end NUMINAMATH_GPT_sum_of_two_pos_implies_one_pos_l869_86932


namespace NUMINAMATH_GPT_absolute_value_simplify_l869_86938

variable (a : ℝ)

theorem absolute_value_simplify
  (h : a < 3) : |a - 3| = 3 - a := sorry

end NUMINAMATH_GPT_absolute_value_simplify_l869_86938


namespace NUMINAMATH_GPT_no_distinct_integers_cycle_l869_86934

theorem no_distinct_integers_cycle (p : ℤ → ℤ) 
  (x : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → x i ≠ x j)
  (n : ℕ) (h_n_ge_3 : n ≥ 3)
  (hx_cycle : ∀ i, i < n → p (x i) = x (i + 1) % n) :
  false :=
sorry

end NUMINAMATH_GPT_no_distinct_integers_cycle_l869_86934


namespace NUMINAMATH_GPT_carrots_as_potatoes_l869_86928

variable (G O C P : ℕ)

theorem carrots_as_potatoes :
  G = 8 →
  G = (1 / 3 : ℚ) * O →
  O = 2 * C →
  P = 2 →
  (C / P : ℚ) = 6 :=
by intros hG1 hG2 hO hP; sorry

end NUMINAMATH_GPT_carrots_as_potatoes_l869_86928


namespace NUMINAMATH_GPT_johns_total_animals_l869_86931

variable (Snakes Monkeys Lions Pandas Dogs : ℕ)

theorem johns_total_animals :
  Snakes = 15 →
  Monkeys = 2 * Snakes →
  Lions = Monkeys - 5 →
  Pandas = Lions + 8 →
  Dogs = Pandas / 3 →
  Snakes + Monkeys + Lions + Pandas + Dogs = 114 :=
by
  intros hSnakes hMonkeys hLions hPandas hDogs
  rw [hSnakes] at hMonkeys
  rw [hMonkeys] at hLions
  rw [hLions] at hPandas
  rw [hPandas] at hDogs
  sorry

end NUMINAMATH_GPT_johns_total_animals_l869_86931


namespace NUMINAMATH_GPT_selection_of_hexagonal_shape_l869_86912

-- Lean 4 Statement: Prove that there are 78 distinct ways to select diagram b from the hexagonal grid of diagram a, considering rotations.

theorem selection_of_hexagonal_shape :
  let center_positions := 1
  let first_ring_positions := 6
  let second_ring_positions := 12
  let third_ring_positions := 6
  let fourth_ring_positions := 1
  let total_positions := center_positions + first_ring_positions + second_ring_positions + third_ring_positions + fourth_ring_positions
  let rotations := 3
  total_positions * rotations = 78 := by
  -- You can skip the explicit proof body here, replace with sorry
  sorry

end NUMINAMATH_GPT_selection_of_hexagonal_shape_l869_86912


namespace NUMINAMATH_GPT_y_intercepts_count_l869_86913

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_y_intercepts_count_l869_86913


namespace NUMINAMATH_GPT_meaning_of_poverty_l869_86941

theorem meaning_of_poverty (s : String) : s = "poverty" ↔ s = "poverty" := sorry

end NUMINAMATH_GPT_meaning_of_poverty_l869_86941


namespace NUMINAMATH_GPT_units_digit_calculation_l869_86960

theorem units_digit_calculation : 
  ((33 * (83 ^ 1001) * (7 ^ 1002) * (13 ^ 1003)) % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_calculation_l869_86960


namespace NUMINAMATH_GPT_range_of_m_l869_86988

def A := { x : ℝ | x^2 - 2 * x - 15 ≤ 0 }
def B (m : ℝ) := { x : ℝ | m - 2 < x ∧ x < 2 * m - 3 }

theorem range_of_m : ∀ m : ℝ, (B m ⊆ A) ↔ (m ≤ 4) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l869_86988


namespace NUMINAMATH_GPT_pyramid_edges_l869_86957

-- Define the conditions
def isPyramid (n : ℕ) : Prop :=
  (n + 1) + (n + 1) = 16

-- Statement to be proved
theorem pyramid_edges : ∃ (n : ℕ), isPyramid n ∧ 2 * n = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_pyramid_edges_l869_86957


namespace NUMINAMATH_GPT_find_a1_l869_86924

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

theorem find_a1
  (h1 : ∀ n : ℕ, a_n 2 * a_n 8 = 2 * a_n 3 * a_n 6)
  (h2 : S_n 5 = -62) :
  a_n 1 = -2 :=
sorry

end NUMINAMATH_GPT_find_a1_l869_86924


namespace NUMINAMATH_GPT_fencing_cost_l869_86914

def total_cost_of_fencing 
  (length breadth cost_per_meter : ℝ)
  (h1 : length = 62)
  (h2 : length = breadth + 24)
  (h3 : cost_per_meter = 26.50) : ℝ :=
  2 * (length + breadth) * cost_per_meter

theorem fencing_cost : total_cost_of_fencing 62 38 26.50 (by rfl) (by norm_num) (by norm_num) = 5300 := 
by 
  sorry

end NUMINAMATH_GPT_fencing_cost_l869_86914


namespace NUMINAMATH_GPT_simplify_abs_expr_l869_86999

noncomputable def piecewise_y (x : ℝ) : ℝ :=
  if h1 : x < -3 then -3 * x
  else if h2 : -3 ≤ x ∧ x < 1 then 6 - x
  else if h3 : 1 ≤ x ∧ x < 2 then 4 + x
  else 3 * x

theorem simplify_abs_expr : 
  ∀ x : ℝ, (|x - 1| + |x - 2| + |x + 3|) = piecewise_y x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_simplify_abs_expr_l869_86999


namespace NUMINAMATH_GPT_sequence_equals_identity_l869_86902

theorem sequence_equals_identity (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j) : 
  ∀ i : ℕ, a i = i := 
by 
  sorry

end NUMINAMATH_GPT_sequence_equals_identity_l869_86902


namespace NUMINAMATH_GPT_minimum_value_correct_l869_86910

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a + 3*b = 1 then 1/a + 1/b else 0

theorem minimum_value_correct : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ minimum_value a b = 4 + 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_correct_l869_86910


namespace NUMINAMATH_GPT_ratio_of_sums_l869_86909

noncomputable def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ratio_of_sums (n : ℕ) (S1 S2 : ℕ) 
  (hn_even : n % 2 = 0)
  (hn_pos : 0 < n)
  (h_sum : sum_upto (n^2) = n^2 * (n^2 + 1) / 2)
  (h_S1S2_sum : S1 + S2 = n^2 * (n^2 + 1) / 2)
  (h_ratio : 64 * S1 = 39 * S2) :
  ∃ k : ℕ, n = 103 * k :=
sorry

end NUMINAMATH_GPT_ratio_of_sums_l869_86909


namespace NUMINAMATH_GPT_sin_alpha_l869_86984

variable (α : Real)
variable (hcos : Real.cos α = 3 / 5)
variable (htan : Real.tan α < 0)

theorem sin_alpha (α : Real) (hcos : Real.cos α = 3 / 5) (htan : Real.tan α < 0) :
  Real.sin α = -4 / 5 :=
sorry

end NUMINAMATH_GPT_sin_alpha_l869_86984


namespace NUMINAMATH_GPT_number_of_girls_l869_86969

/-- In a school with 632 students, the average age of the boys is 12 years
and that of the girls is 11 years. The average age of the school is 11.75 years.
How many girls are there in the school? Prove that the number of girls is 108. -/
theorem number_of_girls (B G : ℕ) (h1 : B + G = 632) (h2 : 12 * B + 11 * G = 7428) :
  G = 108 :=
sorry

end NUMINAMATH_GPT_number_of_girls_l869_86969


namespace NUMINAMATH_GPT_fibonacci_recurrence_l869_86908

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem fibonacci_recurrence (n : ℕ) (h: n ≥ 2) : 
  F n = F (n-1) + F (n-2) := by
 {
 sorry
 }

end NUMINAMATH_GPT_fibonacci_recurrence_l869_86908


namespace NUMINAMATH_GPT_employee_selected_from_10th_group_is_47_l869_86989

theorem employee_selected_from_10th_group_is_47
  (total_employees : ℕ)
  (sampled_employees : ℕ)
  (total_groups : ℕ)
  (random_start : ℕ)
  (common_difference : ℕ)
  (selected_from_5th_group : ℕ) :
  total_employees = 200 →
  sampled_employees = 40 →
  total_groups = 40 →
  random_start = 2 →
  common_difference = 5 →
  selected_from_5th_group = 22 →
  (selected_from_5th_group = (4 * common_difference + random_start)) →
  (9 * common_difference + random_start) = 47 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_employee_selected_from_10th_group_is_47_l869_86989


namespace NUMINAMATH_GPT_sunday_dogs_count_l869_86993

-- Define initial conditions
def initial_dogs : ℕ := 2
def monday_dogs : ℕ := 3
def total_dogs : ℕ := 10
def sunday_dogs (S : ℕ) : Prop :=
  initial_dogs + S + monday_dogs = total_dogs

-- State the theorem
theorem sunday_dogs_count : ∃ S : ℕ, sunday_dogs S ∧ S = 5 := by
  sorry

end NUMINAMATH_GPT_sunday_dogs_count_l869_86993
