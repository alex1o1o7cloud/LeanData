import Mathlib

namespace NUMINAMATH_GPT_mass_fraction_K2SO4_l559_55920

theorem mass_fraction_K2SO4 :
  (2.61 * 100 / 160) = 1.63 :=
by
  -- Proof details are not required as per instructions
  sorry

end NUMINAMATH_GPT_mass_fraction_K2SO4_l559_55920


namespace NUMINAMATH_GPT_swimming_speed_l559_55993

variable (v s : ℝ)

-- Given conditions
def stream_speed : Prop := s = 0.5
def time_relationship : Prop := ∀ d : ℝ, d > 0 → d / (v - s) = 2 * (d / (v + s))

-- The theorem to prove
theorem swimming_speed (h1 : stream_speed s) (h2 : time_relationship v s) : v = 1.5 :=
  sorry

end NUMINAMATH_GPT_swimming_speed_l559_55993


namespace NUMINAMATH_GPT_largest_non_sum_l559_55997

theorem largest_non_sum (n : ℕ) : 
  ¬ (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ∣ 2 ∧ n = 36 * a + b) ↔ n = 104 :=
by
  sorry

end NUMINAMATH_GPT_largest_non_sum_l559_55997


namespace NUMINAMATH_GPT_minimize_intercepts_line_eqn_l559_55975

theorem minimize_intercepts_line_eqn (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : (1:ℝ)/a + (1:ℝ)/b = 1)
  (h2 : ∃ a b, a + b = 4 ∧ a = 2 ∧ b = 2) :
  ∀ (x y : ℝ), x + y - 2 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_minimize_intercepts_line_eqn_l559_55975


namespace NUMINAMATH_GPT_find_the_number_l559_55923

theorem find_the_number :
  ∃ x : ℕ, 72519 * x = 724827405 ∧ x = 10005 :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l559_55923


namespace NUMINAMATH_GPT_fewer_bronze_stickers_l559_55978

theorem fewer_bronze_stickers
  (gold_stickers : ℕ)
  (silver_stickers : ℕ)
  (each_student_stickers : ℕ)
  (students : ℕ)
  (total_stickers_given : ℕ)
  (bronze_stickers : ℕ)
  (total_gold_and_silver_stickers : ℕ)
  (gold_stickers_eq : gold_stickers = 50)
  (silver_stickers_eq : silver_stickers = 2 * gold_stickers)
  (each_student_stickers_eq : each_student_stickers = 46)
  (students_eq : students = 5)
  (total_stickers_given_eq : total_stickers_given = students * each_student_stickers)
  (total_gold_and_silver_stickers_eq : total_gold_and_silver_stickers = gold_stickers + silver_stickers)
  (bronze_stickers_eq : bronze_stickers = total_stickers_given - total_gold_and_silver_stickers) :
  silver_stickers - bronze_stickers = 20 :=
by
  sorry

end NUMINAMATH_GPT_fewer_bronze_stickers_l559_55978


namespace NUMINAMATH_GPT_least_positive_linear_combination_l559_55942

theorem least_positive_linear_combination :
  ∃ x y z : ℤ, 0 < 24 * x + 20 * y + 12 * z ∧ ∀ n : ℤ, (∃ x y z : ℤ, n = 24 * x + 20 * y + 12 * z) → 0 < n → 4 ≤ n :=
by
  sorry

end NUMINAMATH_GPT_least_positive_linear_combination_l559_55942


namespace NUMINAMATH_GPT_company_sales_difference_l559_55970

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end NUMINAMATH_GPT_company_sales_difference_l559_55970


namespace NUMINAMATH_GPT_probability_no_success_l559_55965

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end NUMINAMATH_GPT_probability_no_success_l559_55965


namespace NUMINAMATH_GPT_cosine_sum_formula_l559_55928

theorem cosine_sum_formula
  (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 4 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_cosine_sum_formula_l559_55928


namespace NUMINAMATH_GPT_blue_marbles_difference_l559_55909

-- Definitions of the conditions
def total_green_marbles := 95

-- Ratios for Jar 1 and Jar 2
def ratio_blue_green_jar1 := (9, 1)
def ratio_blue_green_jar2 := (8, 1)

-- Total number of green marbles in each jar
def green_marbles_jar1 (a : ℕ) := a
def green_marbles_jar2 (b : ℕ) := b

-- Total number of marbles in each jar
def total_marbles_jar1 (a : ℕ) := 10 * a
def total_marbles_jar2 (b : ℕ) := 9 * b

-- Number of blue marbles in each jar
def blue_marbles_jar1 (a : ℕ) := 9 * a
def blue_marbles_jar2 (b : ℕ) := 8 * b

-- Conditions in terms of Lean definitions
theorem blue_marbles_difference:
  ∀ (a b : ℕ), green_marbles_jar1 a + green_marbles_jar2 b = total_green_marbles →
  total_marbles_jar1 a = total_marbles_jar2 b →
  blue_marbles_jar1 a - blue_marbles_jar2 b = 5 :=
by sorry

end NUMINAMATH_GPT_blue_marbles_difference_l559_55909


namespace NUMINAMATH_GPT_simplify_expr_l559_55914

-- Define variables and conditions
variables (x y a b c : ℝ)

-- State the theorem
theorem simplify_expr : 
  (2 - y) * 24 * (x - y + 2 * (a - 2 - 3 * c) * a - 2 * b + c) = 
  2 + 4 * b^2 - a * b - c^2 :=
sorry

end NUMINAMATH_GPT_simplify_expr_l559_55914


namespace NUMINAMATH_GPT_max_value_eq_two_l559_55944

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_eq_two_l559_55944


namespace NUMINAMATH_GPT_find_z_l559_55921

theorem find_z (y z : ℝ) (k : ℝ) 
  (h1 : y = 3) (h2 : z = 16) (h3 : y ^ 2 * (z ^ (1 / 4)) = k)
  (h4 : k = 18) (h5 : y = 6) : z = 1 / 16 := by
  sorry

end NUMINAMATH_GPT_find_z_l559_55921


namespace NUMINAMATH_GPT_horner_evaluation_at_2_l559_55938

noncomputable def f : ℕ → ℕ :=
  fun x => (((2 * x + 3) * x + 0) * x + 5) * x - 4

theorem horner_evaluation_at_2 : f 2 = 14 :=
  by
    sorry

end NUMINAMATH_GPT_horner_evaluation_at_2_l559_55938


namespace NUMINAMATH_GPT_absent_children_count_l559_55960

-- Definition of conditions
def total_children := 700
def bananas_per_child := 2
def bananas_extra := 2
def total_bananas := total_children * bananas_per_child

-- The proof goal
theorem absent_children_count (A P : ℕ) (h_P : P = total_children - A)
    (h_bananas : total_bananas = P * (bananas_per_child + bananas_extra)) : A = 350 :=
by
  -- Since this is a statement only, we place a sorry here to skip the proof.
  sorry

end NUMINAMATH_GPT_absent_children_count_l559_55960


namespace NUMINAMATH_GPT_nth_term_sequence_l559_55936

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (2 ^ n) - 1

theorem nth_term_sequence (n : ℕ) : 
  sequence n = 2 ^ n - 1 :=
by
  sorry

end NUMINAMATH_GPT_nth_term_sequence_l559_55936


namespace NUMINAMATH_GPT_obtuse_triangle_has_two_acute_angles_l559_55932

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- A theorem to prove that an obtuse triangle has exactly 2 acute angles 
theorem obtuse_triangle_has_two_acute_angles (A B C : ℝ) (h : is_obtuse_triangle A B C) : 
  (A > 0 ∧ A < 90 → B > 0 ∧ B < 90 → C > 0 ∧ C < 90) ∧
  (A > 0 ∧ A < 90 ∧ B > 0 ∧ B < 90) ∨
  (A > 0 ∧ A < 90 ∧ C > 0 ∧ C < 90) ∨
  (B > 0 ∧ B < 90 ∧ C > 0 ∧ C < 90) :=
sorry

end NUMINAMATH_GPT_obtuse_triangle_has_two_acute_angles_l559_55932


namespace NUMINAMATH_GPT_range_of_b_l559_55998

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a) / (2^x + 1)
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := Real.log (x^2 - b)

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x1 x2 : ℝ, f x1 a ≤ g x2 b) → b ≤ -Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l559_55998


namespace NUMINAMATH_GPT_taller_tree_height_l559_55937

variable (T S : ℝ)

theorem taller_tree_height (h1 : T - S = 20)
  (h2 : T - 10 = 3 * (S - 10)) : T = 40 :=
sorry

end NUMINAMATH_GPT_taller_tree_height_l559_55937


namespace NUMINAMATH_GPT_molecular_weight_of_NH4I_correct_l559_55901

-- Define the atomic weights as given conditions
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

-- Define the calculation of the molecular weight of NH4I
def molecular_weight_NH4I : ℝ :=
  atomic_weight_N + 4 * atomic_weight_H + atomic_weight_I

-- Theorem stating the molecular weight of NH4I is 144.95 g/mol
theorem molecular_weight_of_NH4I_correct : molecular_weight_NH4I = 144.95 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_NH4I_correct_l559_55901


namespace NUMINAMATH_GPT_charlene_sold_necklaces_l559_55992

theorem charlene_sold_necklaces 
  (initial_necklaces : ℕ) 
  (given_away : ℕ) 
  (remaining : ℕ) 
  (total_made : initial_necklaces = 60) 
  (given_to_friends : given_away = 18) 
  (left_with : remaining = 26) : 
  initial_necklaces - given_away - remaining = 16 := 
by
  sorry

end NUMINAMATH_GPT_charlene_sold_necklaces_l559_55992


namespace NUMINAMATH_GPT_number_of_perfect_squares_criteria_l559_55979

noncomputable def number_of_multiples_of_40_squares_lt_4e6 : ℕ :=
  let upper_limit := 2000
  let multiple := 40
  let largest_multiple := upper_limit - (upper_limit % multiple)
  largest_multiple / multiple

theorem number_of_perfect_squares_criteria :
  number_of_multiples_of_40_squares_lt_4e6 = 49 :=
sorry

end NUMINAMATH_GPT_number_of_perfect_squares_criteria_l559_55979


namespace NUMINAMATH_GPT_michael_card_count_l559_55969

variable (Lloyd Mark Michael : ℕ)
variable (L : ℕ)

-- Conditions from the problem
axiom condition1 : Mark = 3 * Lloyd
axiom condition2 : Mark + 10 = Michael
axiom condition3 : Lloyd + Mark + (Michael + 80) = 300

-- The correct answer we want to prove
theorem michael_card_count : Michael = 100 :=
by
  -- Proof will be here.
  sorry

end NUMINAMATH_GPT_michael_card_count_l559_55969


namespace NUMINAMATH_GPT_y_value_l559_55933

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 :=
by
  sorry

end NUMINAMATH_GPT_y_value_l559_55933


namespace NUMINAMATH_GPT_total_white_papers_l559_55964

-- Define the given conditions
def papers_per_envelope : ℕ := 10
def number_of_envelopes : ℕ := 12

-- The theorem statement
theorem total_white_papers : (papers_per_envelope * number_of_envelopes) = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_white_papers_l559_55964


namespace NUMINAMATH_GPT_intersection_one_point_l559_55974

def quadratic_function (x : ℝ) : ℝ := -x^2 + 5 * x
def linear_function (x : ℝ) (t : ℝ) : ℝ := -3 * x + t
def quadratic_combined_function (x : ℝ) (t : ℝ) : ℝ := x^2 - 8 * x + t

theorem intersection_one_point (t : ℝ) : 
  (64 - 4 * t = 0) → t = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_intersection_one_point_l559_55974


namespace NUMINAMATH_GPT_total_value_of_coins_l559_55982

theorem total_value_of_coins (h1 : ∀ (q d : ℕ), q + d = 23)
                             (h2 : ∀ q, q = 16)
                             (h3 : ∀ d, d = 23 - 16)
                             (h4 : ∀ q, q * 0.25 = 4.00)
                             (h5 : ∀ d, d * 0.10 = 0.70)
                             : 4.00 + 0.70 = 4.70 :=
by
  sorry

end NUMINAMATH_GPT_total_value_of_coins_l559_55982


namespace NUMINAMATH_GPT_faction_with_more_liars_than_truth_tellers_l559_55989

theorem faction_with_more_liars_than_truth_tellers 
  (r1 r2 r3 l1 l2 l3 : ℕ) 
  (H1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016)
  (H2 : r1 + l2 + l3 = 1208)
  (H3 : r2 + l1 + l3 = 908)
  (H4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end NUMINAMATH_GPT_faction_with_more_liars_than_truth_tellers_l559_55989


namespace NUMINAMATH_GPT_days_in_first_quarter_2010_l559_55984

theorem days_in_first_quarter_2010 : 
  let not_leap_year := ¬ (2010 % 4 = 0)
  let days_in_february := 28
  let days_in_january_and_march := 31
  not_leap_year → days_in_february = 28 → days_in_january_and_march = 31 → (31 + 28 + 31 = 90)
:= 
sorry

end NUMINAMATH_GPT_days_in_first_quarter_2010_l559_55984


namespace NUMINAMATH_GPT_intersection_A_complement_B_l559_55981

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { y | 0 ≤ y }

theorem intersection_A_complement_B : A ∩ -B = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l559_55981


namespace NUMINAMATH_GPT_boat_distance_l559_55948

theorem boat_distance (v_b : ℝ) (v_s : ℝ) (t_downstream : ℝ) (t_upstream : ℝ) (d : ℝ) :
  v_b = 7 ∧ t_downstream = 2 ∧ t_upstream = 5 ∧ d = (v_b + v_s) * t_downstream ∧ d = (v_b - v_s) * t_upstream → d = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_boat_distance_l559_55948


namespace NUMINAMATH_GPT_conditional_probability_correct_l559_55983

noncomputable def total_products : ℕ := 8
noncomputable def first_class_products : ℕ := 6
noncomputable def chosen_products : ℕ := 2

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def P_A : ℚ := 1 - (combination first_class_products chosen_products) / (combination total_products chosen_products)
noncomputable def P_AB : ℚ := (combination 2 1 * combination first_class_products 1) / (combination total_products chosen_products)

noncomputable def conditional_probability : ℚ := P_AB / P_A

theorem conditional_probability_correct :
  conditional_probability = 12 / 13 :=
  sorry

end NUMINAMATH_GPT_conditional_probability_correct_l559_55983


namespace NUMINAMATH_GPT_composite_19_8n_plus_17_l559_55918

theorem composite_19_8n_plus_17 (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
by 
  sorry

end NUMINAMATH_GPT_composite_19_8n_plus_17_l559_55918


namespace NUMINAMATH_GPT_hypotenuse_length_l559_55945

theorem hypotenuse_length
  (a b : ℝ)
  (V1 : ℝ := (1/3) * Real.pi * a * b^2)
  (V2 : ℝ := (1/3) * Real.pi * b * a^2)
  (hV1 : V1 = 800 * Real.pi)
  (hV2 : V2 = 1920 * Real.pi) :
  Real.sqrt (a^2 + b^2) = 26 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l559_55945


namespace NUMINAMATH_GPT_air_conditioner_sales_l559_55976

/-- Represent the conditions -/
def conditions (x y m : ℕ) : Prop :=
  (3 * x + 5 * y = 23500) ∧
  (4 * x + 10 * y = 42000) ∧
  (x = 2500) ∧
  (y = 3200) ∧
  (700 * (50 - m) + 800 * m ≥ 38000)

/-- Prove that the unit selling prices of models A and B are 2500 yuan and 3200 yuan respectively,
    and at least 30 units of model B need to be purchased for a profit of at least 38000 yuan,
    given the conditions. -/
theorem air_conditioner_sales :
  ∃ (x y m : ℕ), conditions x y m ∧ m ≥ 30 := by
  sorry

end NUMINAMATH_GPT_air_conditioner_sales_l559_55976


namespace NUMINAMATH_GPT_greatest_integer_sum_l559_55917

def floor (x : ℚ) : ℤ := ⌊x⌋

theorem greatest_integer_sum :
  floor (2017 * 3 / 11) + 
  floor (2017 * 4 / 11) + 
  floor (2017 * 5 / 11) + 
  floor (2017 * 6 / 11) + 
  floor (2017 * 7 / 11) + 
  floor (2017 * 8 / 11) = 6048 :=
  by sorry

end NUMINAMATH_GPT_greatest_integer_sum_l559_55917


namespace NUMINAMATH_GPT_probability_heads_at_least_9_l559_55971

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end NUMINAMATH_GPT_probability_heads_at_least_9_l559_55971


namespace NUMINAMATH_GPT_max_non_triangulated_segments_correct_l559_55951

open Classical

/-
Problem description:
Given an equilateral triangle divided into smaller equilateral triangles with side length 1, 
we need to define the maximum number of 1-unit segments that can be marked such that no 
triangular subregion has all its sides marked.
-/

def total_segments (n : ℕ) : ℕ :=
  (3 * n * (n + 1)) / 2

def max_non_triangular_segments (n : ℕ) : ℕ :=
  n * (n + 1)

theorem max_non_triangulated_segments_correct (n : ℕ) :
  max_non_triangular_segments n = n * (n + 1) := by sorry

end NUMINAMATH_GPT_max_non_triangulated_segments_correct_l559_55951


namespace NUMINAMATH_GPT_inequality_solution_l559_55966

noncomputable def inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : Prop :=
  (x^4 + y^4 + z^4) ≥ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ∧ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ≥ (x * y * z * (x + y + z))

theorem inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  inequality_proof x y z hx hy hz :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l559_55966


namespace NUMINAMATH_GPT_max_x_minus_y_l559_55999

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_x_minus_y_l559_55999


namespace NUMINAMATH_GPT_sample_size_stratified_sampling_l559_55915

theorem sample_size_stratified_sampling 
  (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (n : ℕ) (females_drawn : ℕ) 
  (total_people : ℕ := teachers + male_students + female_students) 
  (females_total : ℕ := female_students) 
  (proportion_drawn : ℚ := (females_drawn : ℚ) / females_total) :
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  females_drawn = 80 → 
  proportion_drawn = ((n : ℚ) / total_people) → 
  n = 192 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_stratified_sampling_l559_55915


namespace NUMINAMATH_GPT_greatest_integer_less_PS_l559_55930

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_less_PS_l559_55930


namespace NUMINAMATH_GPT_arithmetic_identity_l559_55907

theorem arithmetic_identity : Real.sqrt 16 + ((1/2) ^ (-2:ℤ)) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_identity_l559_55907


namespace NUMINAMATH_GPT_sum_of_digits_of_d_l559_55949

theorem sum_of_digits_of_d (d : ℕ) (h₁ : ∃ d_ca : ℕ, d_ca = (8 * d) / 5) (h₂ : d_ca - 75 = d) :
  (1 + 2 + 5 = 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_d_l559_55949


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l559_55904

-- Definitions and conditions
-- Define the lengths of the three sides of the triangle
def a : ℕ := 3
def b : ℕ := 8

-- Define that the triangle is isosceles
def is_isosceles_triangle := 
  (a = a) ∨ (b = b) ∨ (a = b)

-- Perimeter of the triangle
def perimeter (x y z : ℕ) := x + y + z

-- The theorem we need to prove
theorem isosceles_triangle_perimeter : is_isosceles_triangle → (a + b + b = 19) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l559_55904


namespace NUMINAMATH_GPT_denomination_of_checks_l559_55953

-- Definitions based on the conditions.
def total_checks := 30
def total_worth := 1800
def checks_spent := 24
def average_remaining := 100

-- Statement to be proven.
theorem denomination_of_checks :
  ∃ x : ℝ, (total_checks - checks_spent) * average_remaining + checks_spent * x = total_worth ∧ x = 40 :=
by
  sorry

end NUMINAMATH_GPT_denomination_of_checks_l559_55953


namespace NUMINAMATH_GPT_side_lengths_le_sqrt3_probability_is_1_over_3_l559_55952

open Real

noncomputable def probability_side_lengths_le_sqrt3 : ℝ :=
  let total_area : ℝ := 2 * π^2
  let satisfactory_area : ℝ := 2 * π^2 / 3
  satisfactory_area / total_area

theorem side_lengths_le_sqrt3_probability_is_1_over_3 :
  probability_side_lengths_le_sqrt3 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_side_lengths_le_sqrt3_probability_is_1_over_3_l559_55952


namespace NUMINAMATH_GPT_max_liters_l559_55968

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end NUMINAMATH_GPT_max_liters_l559_55968


namespace NUMINAMATH_GPT_smallest_positive_debt_l559_55925

noncomputable def pigs_value : ℤ := 300
noncomputable def goats_value : ℤ := 210

theorem smallest_positive_debt : ∃ D p g : ℤ, (D = pigs_value * p + goats_value * g) ∧ D > 0 ∧ ∀ D' p' g' : ℤ, (D' = pigs_value * p' + goats_value * g' ∧ D' > 0) → D ≤ D' :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_l559_55925


namespace NUMINAMATH_GPT_non_degenerate_ellipse_condition_l559_55926

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 9 * x^2 + y^2 - 18 * x - 2 * y = k) ↔ k > -10 :=
sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_condition_l559_55926


namespace NUMINAMATH_GPT_broccoli_difference_l559_55916

theorem broccoli_difference (A : ℕ) (s : ℕ) (s' : ℕ)
  (h1 : A = 1600)
  (h2 : s = Nat.sqrt A)
  (h3 : s' < s)
  (h4 : (s')^2 < A)
  (h5 : A - (s')^2 = 79) :
  (1600 - (s')^2) = 79 :=
by
  sorry

end NUMINAMATH_GPT_broccoli_difference_l559_55916


namespace NUMINAMATH_GPT_factor_expression_l559_55905

theorem factor_expression (x : ℝ) :
  (12 * x ^ 5 + 33 * x ^ 3 + 10) - (3 * x ^ 5 - 4 * x ^ 3 - 1) = x ^ 3 * (9 * x ^ 2 + 37) + 11 :=
by {
  -- Provide the skeleton for the proof using simplification
  sorry
}

end NUMINAMATH_GPT_factor_expression_l559_55905


namespace NUMINAMATH_GPT_small_cubes_with_two_faces_painted_l559_55912

-- Statement of the problem
theorem small_cubes_with_two_faces_painted
  (remaining_cubes : ℕ)
  (edges_with_two_painted_faces : ℕ)
  (number_of_edges : ℕ) :
  remaining_cubes = 60 → edges_with_two_painted_faces = 2 → number_of_edges = 12 →
  (remaining_cubes - (4 * (edges_with_two_painted_faces - 1) * (number_of_edges))) = 28 :=
by
  sorry

end NUMINAMATH_GPT_small_cubes_with_two_faces_painted_l559_55912


namespace NUMINAMATH_GPT_quadratic_roots_problem_l559_55986

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_problem_l559_55986


namespace NUMINAMATH_GPT_chosen_numbers_rel_prime_l559_55995

theorem chosen_numbers_rel_prime :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 2003 → s.card = 1002 → ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ Nat.gcd x y = 1 :=
by
  sorry

end NUMINAMATH_GPT_chosen_numbers_rel_prime_l559_55995


namespace NUMINAMATH_GPT_minimum_value_of_a_l559_55947

theorem minimum_value_of_a (x y a : ℝ) (h1 : y = (1 / (x - 2)) * (x^2))
(h2 : x = a * y) : a = 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l559_55947


namespace NUMINAMATH_GPT_radical_conjugate_sum_l559_55957

theorem radical_conjugate_sum:
  let a := 15 - Real.sqrt 500
  let b := 15 + Real.sqrt 500
  3 * (a + b) = 90 :=
by
  sorry

end NUMINAMATH_GPT_radical_conjugate_sum_l559_55957


namespace NUMINAMATH_GPT_rectangular_solid_sum_of_edges_l559_55991

noncomputable def sum_of_edges (x y z : ℝ) := 4 * (x + y + z)

theorem rectangular_solid_sum_of_edges :
  ∃ (x y z : ℝ), (x * y * z = 512) ∧ (2 * (x * y + y * z + z * x) = 384) ∧
  (∃ (r a : ℝ), x = a / r ∧ y = a ∧ z = a * r) ∧ sum_of_edges x y z = 96 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_sum_of_edges_l559_55991


namespace NUMINAMATH_GPT_equalize_costs_l559_55919

theorem equalize_costs (X Y Z : ℝ) (hXY : X < Y) (hYZ : Y < Z) : (Y + Z - 2 * X) / 3 = (X + Y + Z) / 3 - X := by
  sorry

end NUMINAMATH_GPT_equalize_costs_l559_55919


namespace NUMINAMATH_GPT_division_correct_l559_55973

theorem division_correct :
  250 / (15 + 13 * 3^2) = 125 / 66 :=
by
  -- The proof steps can be filled in here.
  sorry

end NUMINAMATH_GPT_division_correct_l559_55973


namespace NUMINAMATH_GPT_mark_gig_schedule_l559_55903

theorem mark_gig_schedule 
  (every_other_day : ∀ weeks, ∃ gigs, gigs = weeks * 7 / 2) 
  (songs_per_gig : 2 * 5 + 10 = 20) 
  (total_minutes : ∃ gigs, 280 = gigs * 20) : 
  ∃ weeks, weeks = 4 := 
by 
  sorry

end NUMINAMATH_GPT_mark_gig_schedule_l559_55903


namespace NUMINAMATH_GPT_new_cooks_waiters_ratio_l559_55962

-- Definitions based on the conditions
variables (cooks waiters new_waiters : ℕ)

-- Given conditions
def ratio := 3
def initial_waiters := (ratio * cooks) / 3 -- Derived from 3 cooks / 11 waiters = 9 cooks / x waiters
def hired_waiters := 12
def total_waiters := initial_waiters + hired_waiters

-- The restaurant has 9 cooks
def restaurant_cooks := 9

-- Conclusion to prove
theorem new_cooks_waiters_ratio :
  (ratio = 3) →
  (restaurant_cooks = 9) →
  (initial_waiters = (ratio * restaurant_cooks) / 3) →
  (cooks = restaurant_cooks) →
  (waiters = initial_waiters) →
  (new_waiters = waiters + hired_waiters) →
  (new_waiters = 45) →
  (cooks / new_waiters = 1 / 5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_new_cooks_waiters_ratio_l559_55962


namespace NUMINAMATH_GPT_tea_mixture_price_l559_55955

theorem tea_mixture_price :
  ∃ P Q : ℝ, (62 * P + 72 * Q) / (3 * P + Q) = 64.5 :=
by
  sorry

end NUMINAMATH_GPT_tea_mixture_price_l559_55955


namespace NUMINAMATH_GPT_treasure_chest_coins_l559_55946

theorem treasure_chest_coins :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 5) ∧ (n ≥ 0) ∧
  (∀ m : ℕ, (m % 8 = 6) ∧ (m % 9 = 5) → m ≥ 0 → n ≤ m) ∧
  (∃ r : ℕ, n = 11 * (n / 11) + r ∧ r = 3) :=
by
  sorry

end NUMINAMATH_GPT_treasure_chest_coins_l559_55946


namespace NUMINAMATH_GPT_mcgregor_books_finished_l559_55908

def total_books := 89
def floyd_books := 32
def books_left := 23

theorem mcgregor_books_finished : ∀ mg_books : Nat, mg_books = total_books - floyd_books - books_left → mg_books = 34 := 
by
  intro mg_books
  sorry

end NUMINAMATH_GPT_mcgregor_books_finished_l559_55908


namespace NUMINAMATH_GPT_find_number_l559_55959

theorem find_number (number : ℝ) (h : 0.75 / 100 * number = 0.06) : number = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l559_55959


namespace NUMINAMATH_GPT_solve_equation_l559_55934

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) :
  (3 / (x - 2) = 2 / (x - 1)) ↔ (x = -1) :=
sorry

end NUMINAMATH_GPT_solve_equation_l559_55934


namespace NUMINAMATH_GPT_largest_multiple_of_15_under_500_l559_55958

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_under_500_l559_55958


namespace NUMINAMATH_GPT_circle_arc_sum_bounds_l559_55940

open Nat

theorem circle_arc_sum_bounds :
  let red_points := 40
  let blue_points := 30
  let green_points := 20
  let total_arcs := 90
  let T := 0 * red_points + 1 * blue_points + 2 * green_points
  let S_min := 6
  let S_max := 140
  (∀ S, (S = 2 * T - A) → (0 ≤ A ∧ A ≤ 134) → (S_min ≤ S ∧ S ≤ S_max))
  → ∃ S_min S_max, S_min = 6 ∧ S_max = 140 :=
by
  intros
  sorry

end NUMINAMATH_GPT_circle_arc_sum_bounds_l559_55940


namespace NUMINAMATH_GPT_rhombuses_in_grid_l559_55994

def number_of_rhombuses (n : ℕ) : ℕ :=
(n - 1) * n + (n - 1) * n

theorem rhombuses_in_grid :
  number_of_rhombuses 5 = 30 :=
by
  sorry

end NUMINAMATH_GPT_rhombuses_in_grid_l559_55994


namespace NUMINAMATH_GPT_circumcircle_radius_of_sector_l559_55990

theorem circumcircle_radius_of_sector (θ : Real) (r : Real) (cos_val : Real) (R : Real) :
  θ = 30 * Real.pi / 180 ∧ r = 8 ∧ cos_val = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ R = 8 * (Real.sqrt 6 - Real.sqrt 2) →
  R = 8 * (Real.sqrt 6 - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_of_sector_l559_55990


namespace NUMINAMATH_GPT_scientific_notation_of_216000_l559_55910

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_216000_l559_55910


namespace NUMINAMATH_GPT_geometric_sequence_product_l559_55929

theorem geometric_sequence_product :
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 5 = 16 ∧ 
    (∀ n, a (n + 1) = a n * r) ∧
    ∃ r : ℝ, 
      a 2 * a 3 * a 4 = 64 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l559_55929


namespace NUMINAMATH_GPT_sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l559_55977

def original_price : ℝ := 150
def discount_monday_to_wednesday : ℝ := 0.20
def tax_monday_to_wednesday : ℝ := 0.05
def discount_thursday_to_saturday : ℝ := 0.15
def tax_thursday_to_saturday : ℝ := 0.04
def discount_super_saver_sunday1 : ℝ := 0.25
def discount_super_saver_sunday2 : ℝ := 0.10
def tax_super_saver_sunday : ℝ := 0.03
def discount_festive_friday : ℝ := 0.20
def tax_festive_friday : ℝ := 0.04
def additional_discount_festive_friday : ℝ := 0.05

theorem sale_price_monday_to_wednesday : (original_price * (1 - discount_monday_to_wednesday)) * (1 + tax_monday_to_wednesday) = 126 :=
by sorry

theorem sale_price_thursday_to_saturday : (original_price * (1 - discount_thursday_to_saturday)) * (1 + tax_thursday_to_saturday) = 132.60 :=
by sorry

theorem sale_price_super_saver_sunday : ((original_price * (1 - discount_super_saver_sunday1)) * (1 - discount_super_saver_sunday2)) * (1 + tax_super_saver_sunday) = 104.29 :=
by sorry

theorem sale_price_festive_friday_selected : ((original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday)) * (1 - additional_discount_festive_friday) = 118.56 :=
by sorry

theorem sale_price_festive_friday_non_selected : (original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday) = 124.80 :=
by sorry

end NUMINAMATH_GPT_sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l559_55977


namespace NUMINAMATH_GPT_tangent_line_eqn_unique_local_minimum_l559_55963

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + 2) / x

def tangent_line_at_1 (x y : ℝ) : Prop :=
  2 * x + y - Real.exp 1 - 4 = 0

theorem tangent_line_eqn :
  tangent_line_at_1 1 (f 1) :=
sorry

noncomputable def h (x : ℝ) : ℝ := Real.exp x * (x - 1) - 2

theorem unique_local_minimum :
  ∃! c : ℝ, 1 < c ∧ c < 2 ∧ (∀ x < c, f x > f c) ∧ (∀ x > c, f c < f x) :=
sorry

end NUMINAMATH_GPT_tangent_line_eqn_unique_local_minimum_l559_55963


namespace NUMINAMATH_GPT_trigonometric_identity_l559_55931

theorem trigonometric_identity :
  (Real.cos (12 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.sin (12 * Real.pi / 180) * Real.sin (18 * Real.pi / 180) = 
   Real.cos (30 * Real.pi / 180)) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l559_55931


namespace NUMINAMATH_GPT_range_of_a_l559_55922

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3) ∧ (x - a > 0)) ↔ (a ≤ -1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l559_55922


namespace NUMINAMATH_GPT_cristina_running_pace_4point2_l559_55939

theorem cristina_running_pace_4point2 :
  ∀ (nicky_pace head_start time_after_start cristina_pace : ℝ),
    nicky_pace = 3 →
    head_start = 12 →
    time_after_start = 30 →
    cristina_pace = 4.2 →
    (time_after_start = head_start + 30 →
    cristina_pace * time_after_start = nicky_pace * (head_start + 30)) :=
by
  sorry

end NUMINAMATH_GPT_cristina_running_pace_4point2_l559_55939


namespace NUMINAMATH_GPT_sum_value_l559_55985

variable (T R S PV : ℝ)
variable (TD SI : ℝ) (h_td : TD = 80) (h_si : SI = 88)
variable (h1 : SI = TD + (TD * R * T) / 100)
variable (h2 : (PV * R * T) / 100 = TD)
variable (h3 : PV = S - TD)
variable (h4 : R * T = 10)

theorem sum_value : S = 880 := by
  sorry

end NUMINAMATH_GPT_sum_value_l559_55985


namespace NUMINAMATH_GPT_fibonacci_factorial_sum_l559_55988

def factorial_last_two_digits(n: ℕ) : ℕ :=
  if n > 10 then 0 else 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120 % 100
  | 6 => 720 % 100
  | 7 => 5040 % 100
  | 8 => 40320 % 100
  | 9 => 362880 % 100
  | 10 => 3628800 % 100
  | _ => 0

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

noncomputable def sum_last_two_digits (l: List ℕ) : ℕ :=
  l.map factorial_last_two_digits |>.sum

theorem fibonacci_factorial_sum:
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end NUMINAMATH_GPT_fibonacci_factorial_sum_l559_55988


namespace NUMINAMATH_GPT_quadratic_one_solution_m_value_l559_55941

theorem quadratic_one_solution_m_value (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) → (b^2 - 4 * a * m = 0) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_one_solution_m_value_l559_55941


namespace NUMINAMATH_GPT_average_age_of_5_people_l559_55911

theorem average_age_of_5_people (avg_age_18 : ℕ) (avg_age_9 : ℕ) (age_15th : ℕ) (total_persons: ℕ) (persons_9: ℕ) (remaining_persons: ℕ) : 
  avg_age_18 = 15 ∧ 
  avg_age_9 = 16 ∧ 
  age_15th = 56 ∧ 
  total_persons = 18 ∧ 
  persons_9 = 9 ∧ 
  remaining_persons = 5 → 
  (avg_age_18 * total_persons - avg_age_9 * persons_9 - age_15th) / remaining_persons = 14 := 
sorry

end NUMINAMATH_GPT_average_age_of_5_people_l559_55911


namespace NUMINAMATH_GPT_problem_l559_55961

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 4
noncomputable def c : ℝ := Real.log 9 / Real.log 4

theorem problem : a = c ∧ a > b :=
by
  sorry

end NUMINAMATH_GPT_problem_l559_55961


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l559_55956

variables (a b c : ℝ) (p q : Prop)

def condition_p : Prop := a * b * c = 0
def condition_q : Prop := a = 0

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) :=
by
  let p := condition_p a b c
  let q := condition_q a
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l559_55956


namespace NUMINAMATH_GPT_min_Box_value_l559_55924

/-- The conditions are given as:
  1. (ax + b)(bx + a) = 24x^2 + Box * x + 24
  2. a, b, Box are distinct integers
  The task is to find the minimum possible value of Box.
-/
theorem min_Box_value :
  ∃ (a b Box : ℤ), a ≠ b ∧ a ≠ Box ∧ b ≠ Box ∧ (∀ x : ℤ, (a * x + b) * (b * x + a) = 24 * x^2 + Box * x + 24) ∧ Box = 52 := sorry

end NUMINAMATH_GPT_min_Box_value_l559_55924


namespace NUMINAMATH_GPT_number_of_poison_frogs_l559_55927

theorem number_of_poison_frogs
  (total_frogs : ℕ) (tree_frogs : ℕ) (wood_frogs : ℕ) (poison_frogs : ℕ)
  (h₁ : total_frogs = 78)
  (h₂ : tree_frogs = 55)
  (h₃ : wood_frogs = 13)
  (h₄ : total_frogs = tree_frogs + wood_frogs + poison_frogs) :
  poison_frogs = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_poison_frogs_l559_55927


namespace NUMINAMATH_GPT_min_value_expression_l559_55906

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l559_55906


namespace NUMINAMATH_GPT_find_k_of_geometric_mean_l559_55967

-- Let {a_n} be an arithmetic sequence with common difference d and a_1 = 9d.
-- Prove that if a_k is the geometric mean of a_1 and a_{2k}, then k = 4.
theorem find_k_of_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : ∀ n, a n = 9 * d + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a k ^ 2 = a 1 * a (2 * k)) : k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_of_geometric_mean_l559_55967


namespace NUMINAMATH_GPT_complement_of_M_with_respect_to_U_l559_55972

open Set

def U : Set ℤ := {-1, -2, -3, -4}
def M : Set ℤ := {-2, -3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {-1, -4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_with_respect_to_U_l559_55972


namespace NUMINAMATH_GPT_inequality_holds_l559_55913

theorem inequality_holds (m : ℝ) (h : 0 ≤ m ∧ m < 12) :
  ∀ x : ℝ, 3 * m * x ^ 2 + m * x + 1 > 0 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l559_55913


namespace NUMINAMATH_GPT_cheesecake_total_calories_l559_55996

-- Define the conditions
def slice_calories : ℕ := 350

def percent_eaten : ℕ := 25
def slices_eaten : ℕ := 2

-- Define the total number of slices in a cheesecake
def total_slices (percent_eaten slices_eaten : ℕ) : ℕ :=
  slices_eaten * (100 / percent_eaten)

-- Define the total calories in a cheesecake given the above conditions
def total_calories (slice_calories slices : ℕ) : ℕ :=
  slice_calories * slices

-- State the theorem
theorem cheesecake_total_calories :
  total_calories slice_calories (total_slices percent_eaten slices_eaten) = 2800 :=
by
  sorry

end NUMINAMATH_GPT_cheesecake_total_calories_l559_55996


namespace NUMINAMATH_GPT_find_y_when_x_eq_4_l559_55900

theorem find_y_when_x_eq_4 (x y : ℝ) (k : ℝ) :
  (8 * y = k / x^3) →
  (y = 25) →
  (x = 2) →
  (exists y', x = 4 → y' = 25/8) :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_eq_4_l559_55900


namespace NUMINAMATH_GPT_BoxC_in_BoxA_l559_55980

-- Define the relationship between the boxes
def BoxA_has_BoxB (A B : ℕ) : Prop := A = 4 * B
def BoxB_has_BoxC (B C : ℕ) : Prop := B = 6 * C

-- Define the proof problem
theorem BoxC_in_BoxA {A B C : ℕ} (h1 : BoxA_has_BoxB A B) (h2 : BoxB_has_BoxC B C) : A = 24 * C :=
by
  sorry

end NUMINAMATH_GPT_BoxC_in_BoxA_l559_55980


namespace NUMINAMATH_GPT_total_wait_days_l559_55935

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end NUMINAMATH_GPT_total_wait_days_l559_55935


namespace NUMINAMATH_GPT_calculation_correct_l559_55902

noncomputable def calc_expression : Float :=
  20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7

theorem calculation_correct : calc_expression = 1640 := 
  by 
    sorry

end NUMINAMATH_GPT_calculation_correct_l559_55902


namespace NUMINAMATH_GPT_machine_present_value_l559_55950

theorem machine_present_value
  (r : ℝ)  -- the depletion rate
  (t : ℝ)  -- the time in years
  (V_t : ℝ)  -- the value of the machine after time t
  (V_0 : ℝ)  -- the present value of the machine
  (h1 : r = 0.10)  -- condition for depletion rate
  (h2 : t = 2)  -- condition for time
  (h3 : V_t = 729)  -- condition for machine's value after time t
  (h4 : V_t = V_0 * (1 - r) ^ t)  -- exponential decay formula
  : V_0 = 900 :=
sorry

end NUMINAMATH_GPT_machine_present_value_l559_55950


namespace NUMINAMATH_GPT_concert_total_cost_l559_55987

noncomputable def total_cost (ticket_cost : ℕ) (processing_fee_rate : ℚ) (parking_fee : ℕ)
  (entrance_fee_per_person : ℕ) (num_persons : ℕ) (refreshments_cost : ℕ) 
  (merchandise_cost : ℕ) : ℚ :=
  let ticket_total := ticket_cost * num_persons
  let processing_fee := processing_fee_rate * (ticket_total : ℚ)
  ticket_total + processing_fee + (parking_fee + entrance_fee_per_person * num_persons 
  + refreshments_cost + merchandise_cost)

theorem concert_total_cost :
  total_cost 75 0.15 10 5 2 20 40 = 252.50 := by 
  sorry

end NUMINAMATH_GPT_concert_total_cost_l559_55987


namespace NUMINAMATH_GPT_even_of_even_square_sqrt_two_irrational_l559_55954

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end NUMINAMATH_GPT_even_of_even_square_sqrt_two_irrational_l559_55954


namespace NUMINAMATH_GPT_find_divisor_l559_55943

theorem find_divisor
  (D dividend quotient remainder : ℤ)
  (h_dividend : dividend = 13787)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14)
  (h_relation : dividend = (D * quotient) + remainder) :
  D = 155 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l559_55943
