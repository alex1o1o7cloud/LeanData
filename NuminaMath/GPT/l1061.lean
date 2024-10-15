import Mathlib

namespace NUMINAMATH_GPT_more_likely_second_machine_l1061_106169

variable (P_B1 : ℝ := 0.8) -- Probability that a part is from the first machine
variable (P_B2 : ℝ := 0.2) -- Probability that a part is from the second machine
variable (P_A_given_B1 : ℝ := 0.01) -- Probability that a part is defective given it is from the first machine
variable (P_A_given_B2 : ℝ := 0.05) -- Probability that a part is defective given it is from the second machine

noncomputable def P_A : ℝ :=
  P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2

noncomputable def P_B1_given_A : ℝ :=
  (P_B1 * P_A_given_B1) / P_A

noncomputable def P_B2_given_A : ℝ :=
  (P_B2 * P_A_given_B2) / P_A

theorem more_likely_second_machine :
  P_B2_given_A > P_B1_given_A :=
by
  sorry

end NUMINAMATH_GPT_more_likely_second_machine_l1061_106169


namespace NUMINAMATH_GPT_age_of_50th_student_l1061_106106

theorem age_of_50th_student (avg_50_students : ℝ) (total_students : ℕ)
                           (avg_15_students : ℝ) (group_1_count : ℕ)
                           (avg_15_students_2 : ℝ) (group_2_count : ℕ)
                           (avg_10_students : ℝ) (group_3_count : ℕ)
                           (avg_9_students : ℝ) (group_4_count : ℕ) :
                           avg_50_students = 20 → total_students = 50 →
                           avg_15_students = 18 → group_1_count = 15 →
                           avg_15_students_2 = 22 → group_2_count = 15 →
                           avg_10_students = 25 → group_3_count = 10 →
                           avg_9_students = 24 → group_4_count = 9 →
                           ∃ (age_50th_student : ℝ), age_50th_student = 66 := by
                           sorry

end NUMINAMATH_GPT_age_of_50th_student_l1061_106106


namespace NUMINAMATH_GPT_stone_counting_l1061_106174

theorem stone_counting (n : ℕ) (m : ℕ) : 
    10 > 0 ∧  (n ≡ 6 [MOD 20]) ∧ m = 126 → n = 6 := 
by
  sorry

end NUMINAMATH_GPT_stone_counting_l1061_106174


namespace NUMINAMATH_GPT_polynomial_properties_l1061_106170

noncomputable def polynomial : Polynomial ℚ :=
  -3/8 * (Polynomial.X ^ 5) + 5/4 * (Polynomial.X ^ 3) - 15/8 * (Polynomial.X)

theorem polynomial_properties (f : Polynomial ℚ) :
  (Polynomial.degree f = 5) ∧
  (∃ q : Polynomial ℚ, f + 1 = Polynomial.X - 1 ^ 3 * q) ∧
  (∃ p : Polynomial ℚ, f - 1 = Polynomial.X + 1 ^ 3 * p) ↔
  f = polynomial :=
by sorry

end NUMINAMATH_GPT_polynomial_properties_l1061_106170


namespace NUMINAMATH_GPT_least_n_for_perfect_square_l1061_106199

theorem least_n_for_perfect_square (n : ℕ) :
  (∀ m : ℕ, 2^8 + 2^11 + 2^n = m * m) → n = 12 := sorry

end NUMINAMATH_GPT_least_n_for_perfect_square_l1061_106199


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1061_106107

theorem perpendicular_lines_condition (m : ℝ) : (m = -1) ↔ ∀ (x y : ℝ), (x + y = 0) ∧ (x + m * y = 0) → 
  ((m ≠ 0) ∧ (-1) * (-1 / m) = 1) :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1061_106107


namespace NUMINAMATH_GPT_price_reduction_example_l1061_106157

def original_price_per_mango (P : ℝ) : Prop :=
  (115 * P = 383.33)

def number_of_mangoes (P : ℝ) (n : ℝ) : Prop :=
  (n * P = 360)

def new_number_of_mangoes (n : ℝ) (R : ℝ) : Prop :=
  ((n + 12) * R = 360)

def percentage_reduction (P R : ℝ) (reduction : ℝ) : Prop :=
  (reduction = ((P - R) / P) * 100)

theorem price_reduction_example : 
  ∃ P R reduction, original_price_per_mango P ∧
    (∃ n, number_of_mangoes P n ∧ new_number_of_mangoes n R) ∧ 
    percentage_reduction P R reduction ∧ 
    reduction = 9.91 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_example_l1061_106157


namespace NUMINAMATH_GPT_extra_large_yellow_curlers_l1061_106101

def total_curlers : ℕ := 120
def small_pink_curlers : ℕ := total_curlers / 5
def medium_blue_curlers : ℕ := 2 * small_pink_curlers
def large_green_curlers : ℕ := total_curlers / 4

theorem extra_large_yellow_curlers : 
  total_curlers - small_pink_curlers - medium_blue_curlers - large_green_curlers = 18 :=
by
  sorry

end NUMINAMATH_GPT_extra_large_yellow_curlers_l1061_106101


namespace NUMINAMATH_GPT_remainder_1234567_div_256_l1061_106126

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end NUMINAMATH_GPT_remainder_1234567_div_256_l1061_106126


namespace NUMINAMATH_GPT_rice_bag_weight_l1061_106121

theorem rice_bag_weight (r f : ℕ) (total_weight : ℕ) (h1 : 20 * r + 50 * f = 2250) (h2 : r = 2 * f) : r = 50 := 
by
  sorry

end NUMINAMATH_GPT_rice_bag_weight_l1061_106121


namespace NUMINAMATH_GPT_nancy_water_intake_l1061_106142

theorem nancy_water_intake (water_intake body_weight : ℝ) (h1 : water_intake = 54) (h2 : body_weight = 90) : 
  (water_intake / body_weight) * 100 = 60 :=
by
  -- using the conditions h1 and h2
  rw [h1, h2]
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_nancy_water_intake_l1061_106142


namespace NUMINAMATH_GPT_neg_mul_neg_pos_mul_neg_neg_l1061_106116

theorem neg_mul_neg_pos (a b : Int) (ha : a < 0) (hb : b < 0) : a * b > 0 :=
sorry

theorem mul_neg_neg : (-1) * (-3) = 3 := 
by
  have h1 : -1 < 0 := by norm_num
  have h2 : -3 < 0 := by norm_num
  have h_pos := neg_mul_neg_pos (-1) (-3) h1 h2
  linarith

end NUMINAMATH_GPT_neg_mul_neg_pos_mul_neg_neg_l1061_106116


namespace NUMINAMATH_GPT_moles_of_ca_oh_2_l1061_106113

-- Define the chemical reaction
def ca_o := 1
def h_2_o := 1
def ca_oh_2 := ca_o + h_2_o

-- Prove the result of the reaction
theorem moles_of_ca_oh_2 :
  ca_oh_2 = 1 := by sorry

end NUMINAMATH_GPT_moles_of_ca_oh_2_l1061_106113


namespace NUMINAMATH_GPT_cubic_boxes_properties_l1061_106195

-- Define the lengths of the edges of the cubic boxes
def edge_length_1 : ℝ := 3
def edge_length_2 : ℝ := 5
def edge_length_3 : ℝ := 6

-- Define the volumes of the respective cubic boxes
def volume (edge_length : ℝ) : ℝ := edge_length ^ 3
def volume_1 := volume edge_length_1
def volume_2 := volume edge_length_2
def volume_3 := volume edge_length_3

-- Define the surface areas of the respective cubic boxes
def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)
def surface_area_1 := surface_area edge_length_1
def surface_area_2 := surface_area edge_length_2
def surface_area_3 := surface_area edge_length_3

-- Total volume and surface area calculations
def total_volume := volume_1 + volume_2 + volume_3
def total_surface_area := surface_area_1 + surface_area_2 + surface_area_3

-- Theorem statement to be proven
theorem cubic_boxes_properties :
  total_volume = 368 ∧ total_surface_area = 420 := by
  sorry

end NUMINAMATH_GPT_cubic_boxes_properties_l1061_106195


namespace NUMINAMATH_GPT_best_fitting_regression_line_l1061_106134

theorem best_fitting_regression_line
  (R2_A : ℝ) (R2_B : ℝ) (R2_C : ℝ) (R2_D : ℝ)
  (h_A : R2_A = 0.27)
  (h_B : R2_B = 0.85)
  (h_C : R2_C = 0.96)
  (h_D : R2_D = 0.5) :
  R2_C = 0.96 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_best_fitting_regression_line_l1061_106134


namespace NUMINAMATH_GPT_sqrt_40_simplified_l1061_106173

theorem sqrt_40_simplified : Real.sqrt 40 = 2 * Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_40_simplified_l1061_106173


namespace NUMINAMATH_GPT_A_inter_B_empty_l1061_106198

def setA : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def setB : Set ℝ := {x | Real.log x / Real.log 4 > 1/2}

theorem A_inter_B_empty : setA ∩ setB = ∅ := by
  sorry

end NUMINAMATH_GPT_A_inter_B_empty_l1061_106198


namespace NUMINAMATH_GPT_scientific_notation_of_258000000_l1061_106181

theorem scientific_notation_of_258000000 :
  258000000 = 2.58 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_258000000_l1061_106181


namespace NUMINAMATH_GPT_smallest_composite_no_prime_factors_below_15_correct_l1061_106188

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_smallest_composite_no_prime_factors_below_15_correct_l1061_106188


namespace NUMINAMATH_GPT_part1_part2_l1061_106104

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x - (a - 1) / x

theorem part1 (a : ℝ) (x : ℝ) (h1 : a ≥ 1) (h2 : x > 0) : f a x ≤ -1 :=
sorry

theorem part2 (a : ℝ) (θ : ℝ) (h1 : a ≥ 1) (h2 : 0 ≤ θ) (h3 : θ ≤ Real.pi / 2) : 
  f a (1 - Real.sin θ) ≤ f a (1 + Real.sin θ) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1061_106104


namespace NUMINAMATH_GPT_roots_cubic_properties_l1061_106150

theorem roots_cubic_properties (a b c : ℝ) 
    (h1 : ∀ x : ℝ, x^3 - 2 * x^2 + 3 * x - 4 = 0 → x = a ∨ x = b ∨ x = c)
    (h_sum : a + b + c = 2)
    (h_prod_sum : a * b + b * c + c * a = 3)
    (h_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := by
  sorry

end NUMINAMATH_GPT_roots_cubic_properties_l1061_106150


namespace NUMINAMATH_GPT_original_average_weight_l1061_106162

-- Definitions from conditions
def original_team_size : ℕ := 7
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60
def new_team_size := original_team_size + 2
def new_average_weight : ℝ := 106

-- Statement to prove
theorem original_average_weight (W : ℝ) :
  (7 * W + 110 + 60 = 9 * 106) → W = 112 := by
  sorry

end NUMINAMATH_GPT_original_average_weight_l1061_106162


namespace NUMINAMATH_GPT_selena_book_pages_l1061_106187

variable (S : ℕ)
variable (H : ℕ)

theorem selena_book_pages (cond1 : H = S / 2 - 20) (cond2 : H = 180) : S = 400 :=
by
  sorry

end NUMINAMATH_GPT_selena_book_pages_l1061_106187


namespace NUMINAMATH_GPT_family_gathering_l1061_106145

theorem family_gathering : 
  ∃ (total_people oranges bananas apples : ℕ), 
    total_people = 20 ∧ 
    oranges = total_people / 2 ∧ 
    bananas = (total_people - oranges) / 2 ∧ 
    apples = total_people - oranges - bananas ∧ 
    oranges < total_people ∧ 
    total_people - oranges = 10 :=
by
  sorry

end NUMINAMATH_GPT_family_gathering_l1061_106145


namespace NUMINAMATH_GPT_exists_close_points_l1061_106155

theorem exists_close_points (r : ℝ) (h : r > 0) (points : Fin 5 → EuclideanSpace ℝ (Fin 3)) (hf : ∀ i, dist (points i) (0 : EuclideanSpace ℝ (Fin 3)) = r) :
  ∃ i j : Fin 5, i ≠ j ∧ dist (points i) (points j) ≤ r * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_exists_close_points_l1061_106155


namespace NUMINAMATH_GPT_how_many_buckets_did_Eden_carry_l1061_106127

variable (E : ℕ) -- Natural number representing buckets Eden carried
variable (M : ℕ) -- Natural number representing buckets Mary carried
variable (I : ℕ) -- Natural number representing buckets Iris carried

-- Conditions based on the problem
axiom Mary_Carry_More : M = E + 3
axiom Iris_Carry_Less : I = M - 1
axiom Total_Buckets : E + M + I = 34

theorem how_many_buckets_did_Eden_carry (h1 : M = E + 3) (h2 : I = M - 1) (h3 : E + M + I = 34) :
  E = 29 / 3 := by
  sorry

end NUMINAMATH_GPT_how_many_buckets_did_Eden_carry_l1061_106127


namespace NUMINAMATH_GPT_unique_solution_2023_plus_2_pow_n_eq_k_sq_l1061_106180

theorem unique_solution_2023_plus_2_pow_n_eq_k_sq (n k : ℕ) (h : 2023 + 2^n = k^2) :
  (n = 1 ∧ k = 45) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_2023_plus_2_pow_n_eq_k_sq_l1061_106180


namespace NUMINAMATH_GPT_grunters_at_least_4_wins_l1061_106183

noncomputable def grunters_probability : ℚ :=
  let p_win := 3 / 5
  let p_loss := 2 / 5
  let p_4_wins := 5 * (p_win^4) * (p_loss)
  let p_5_wins := p_win^5
  p_4_wins + p_5_wins

theorem grunters_at_least_4_wins :
  grunters_probability = 1053 / 3125 :=
by sorry

end NUMINAMATH_GPT_grunters_at_least_4_wins_l1061_106183


namespace NUMINAMATH_GPT_valid_combinations_l1061_106190

def herbs : Nat := 4
def crystals : Nat := 6
def incompatible_pairs : Nat := 3

theorem valid_combinations : 
  (herbs * crystals) - incompatible_pairs = 21 := by
  sorry

end NUMINAMATH_GPT_valid_combinations_l1061_106190


namespace NUMINAMATH_GPT_six_nine_op_l1061_106102

variable (m n : ℚ)

def op (x y : ℚ) : ℚ := m^2 * x + n * y - 1

theorem six_nine_op :
  (op m n 2 3 = 3) →
  (op m n 6 9 = 11) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_six_nine_op_l1061_106102


namespace NUMINAMATH_GPT_same_profit_and_loss_selling_price_l1061_106132

theorem same_profit_and_loss_selling_price (CP SP : ℝ) (h₁ : CP = 49) (h₂ : (CP - 42) = (SP - CP)) : SP = 56 :=
by 
  sorry

end NUMINAMATH_GPT_same_profit_and_loss_selling_price_l1061_106132


namespace NUMINAMATH_GPT_travel_distance_of_wheel_l1061_106141

theorem travel_distance_of_wheel (r : ℝ) (revolutions : ℕ) (h_r : r = 2) (h_revolutions : revolutions = 2) : 
    ∃ d : ℝ, d = 8 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_travel_distance_of_wheel_l1061_106141


namespace NUMINAMATH_GPT_find_P_Q_l1061_106193

noncomputable def P := 11 / 3
noncomputable def Q := -2 / 3

theorem find_P_Q :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
    (3 * x + 12) / (x ^ 2 - 5 * x - 14) = P / (x - 7) + Q / (x + 2) :=
by
  intros x hx1 hx2
  dsimp [P, Q]  -- Unfold the definitions of P and Q
  -- The actual proof would go here, but we are skipping it
  sorry

end NUMINAMATH_GPT_find_P_Q_l1061_106193


namespace NUMINAMATH_GPT_probability_XOXOXOX_is_one_over_thirty_five_l1061_106137

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end NUMINAMATH_GPT_probability_XOXOXOX_is_one_over_thirty_five_l1061_106137


namespace NUMINAMATH_GPT_max_single_painted_faces_l1061_106159

theorem max_single_painted_faces (n : ℕ) (hn : n = 64) :
  ∃ max_cubes : ℕ, max_cubes = 32 := 
sorry

end NUMINAMATH_GPT_max_single_painted_faces_l1061_106159


namespace NUMINAMATH_GPT_compound_interest_time_l1061_106179

theorem compound_interest_time (P r CI : ℝ) (n : ℕ) (A : ℝ) :
  P = 16000 ∧ r = 0.15 ∧ CI = 6218 ∧ n = 1 ∧ A = P + CI →
  t = 2 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_time_l1061_106179


namespace NUMINAMATH_GPT_eulers_formula_convex_polyhedron_l1061_106176

theorem eulers_formula_convex_polyhedron :
  ∀ (V E F T H : ℕ),
  (V - E + F = 2) →
  (F = 24) →
  (E = (3 * T + 6 * H) / 2) →
  100 * H + 10 * T + V = 240 :=
by
  intros V E F T H h1 h2 h3
  sorry

end NUMINAMATH_GPT_eulers_formula_convex_polyhedron_l1061_106176


namespace NUMINAMATH_GPT_not_p_equiv_exists_leq_sin_l1061_106130

-- Define the conditions as a Lean proposition
def p : Prop := ∀ x : ℝ, x > Real.sin x

-- State the problem as a theorem to be proved
theorem not_p_equiv_exists_leq_sin : ¬p = ∃ x : ℝ, x ≤ Real.sin x := 
by sorry

end NUMINAMATH_GPT_not_p_equiv_exists_leq_sin_l1061_106130


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1061_106146

def is_frameable (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 6

theorem part_a : is_frameable 3 ∧ is_frameable 4 ∧ is_frameable 6 :=
  sorry

theorem part_b (n : ℕ) (h : n ≥ 7) : ¬ is_frameable n :=
  sorry

theorem part_c : ¬ is_frameable 5 :=
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1061_106146


namespace NUMINAMATH_GPT_range_of_x_l1061_106165

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x

theorem range_of_x (x : ℝ) (h : f (x^2 + 2) < f (3 * x)) : 1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_GPT_range_of_x_l1061_106165


namespace NUMINAMATH_GPT_scientific_notation_gdp_l1061_106143

theorem scientific_notation_gdp :
  8837000000 = 8.837 * 10^9 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_gdp_l1061_106143


namespace NUMINAMATH_GPT_find_multiplier_l1061_106105

theorem find_multiplier (N x : ℕ) (h₁ : N = 12) (h₂ : N * x - 3 = (N - 7) * 9) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l1061_106105


namespace NUMINAMATH_GPT_max_n_l1061_106197

def sum_first_n_terms (S n : ℕ) (a : ℕ → ℕ) : Prop :=
  S = 2 * a n - n

theorem max_n (S : ℕ) (a : ℕ → ℕ) :
  (∀ n, sum_first_n_terms S n a) → ∀ n, (2 ^ n - 1 ≤ 10 * n) → n ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_n_l1061_106197


namespace NUMINAMATH_GPT_greening_investment_growth_l1061_106125

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end NUMINAMATH_GPT_greening_investment_growth_l1061_106125


namespace NUMINAMATH_GPT_acute_angle_probability_correct_l1061_106111

noncomputable def acute_angle_probability (n : ℕ) (n_ge_4 : n ≥ 4) : ℝ :=
  (n * (n - 2)) / (2 ^ (n-1))

theorem acute_angle_probability_correct (n : ℕ) (h : n ≥ 4) (P : Fin n → ℝ) -- P represents points on the circle
    (uniformly_distributed : ∀ i, P i ∈ Set.Icc (0 : ℝ) 1) : 
    acute_angle_probability n h = (n * (n - 2)) / (2 ^ (n-1)) := 
  sorry

end NUMINAMATH_GPT_acute_angle_probability_correct_l1061_106111


namespace NUMINAMATH_GPT_two_integer_solutions_iff_m_l1061_106161

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end NUMINAMATH_GPT_two_integer_solutions_iff_m_l1061_106161


namespace NUMINAMATH_GPT_ab_calculation_l1061_106178

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1 / 2) * (4 / a) * (4 / b)

theorem ab_calculation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : triangle_area a b = 4) : a * b = 2 :=
by
  sorry

end NUMINAMATH_GPT_ab_calculation_l1061_106178


namespace NUMINAMATH_GPT_value_of_k_l1061_106122

theorem value_of_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (hk : k ≠ 0) : k = 8 :=
sorry

end NUMINAMATH_GPT_value_of_k_l1061_106122


namespace NUMINAMATH_GPT_total_clowns_l1061_106112

def num_clown_mobiles : Nat := 5
def clowns_per_mobile : Nat := 28

theorem total_clowns : num_clown_mobiles * clowns_per_mobile = 140 := by
  sorry

end NUMINAMATH_GPT_total_clowns_l1061_106112


namespace NUMINAMATH_GPT_billion_in_scientific_notation_l1061_106156

theorem billion_in_scientific_notation :
  (4.55 * 10^9) = (4.55 * 10^9) := by
  sorry

end NUMINAMATH_GPT_billion_in_scientific_notation_l1061_106156


namespace NUMINAMATH_GPT_economy_class_seats_l1061_106175

-- Definitions based on the conditions
def first_class_people : ℕ := 3
def business_class_people : ℕ := 22
def economy_class_fullness (E : ℕ) : ℕ := E / 2

-- Problem statement: Proving E == 50 given the conditions
theorem economy_class_seats :
  ∃ E : ℕ,  economy_class_fullness E = first_class_people + business_class_people → E = 50 :=
by
  sorry

end NUMINAMATH_GPT_economy_class_seats_l1061_106175


namespace NUMINAMATH_GPT_real_solution_2015x_equation_l1061_106115

theorem real_solution_2015x_equation (k : ℝ) :
  (∃ x : ℝ, (4 * 2015^x - 2015^(-x)) / (2015^x - 3 * 2015^(-x)) = k) ↔ (k < 1/3 ∨ k > 4) := 
by sorry

end NUMINAMATH_GPT_real_solution_2015x_equation_l1061_106115


namespace NUMINAMATH_GPT_original_number_l1061_106133

theorem original_number (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 :=
sorry

end NUMINAMATH_GPT_original_number_l1061_106133


namespace NUMINAMATH_GPT_solve_quadratic_l1061_106163

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = 2 + Real.sqrt 11) ∧ (x2 = 2 - (Real.sqrt 11)) ∧ 
  (∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ x = x1 ∨ x = x2) := 
sorry

end NUMINAMATH_GPT_solve_quadratic_l1061_106163


namespace NUMINAMATH_GPT_a_5_is_9_l1061_106172

-- Definition of the sequence sum S_n
def S : ℕ → ℕ
| n => n^2 - 1

-- Define the specific term in the sequence
def a (n : ℕ) :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Theorem to prove
theorem a_5_is_9 : a 5 = 9 :=
sorry

end NUMINAMATH_GPT_a_5_is_9_l1061_106172


namespace NUMINAMATH_GPT_ab_relationship_l1061_106140

theorem ab_relationship (a b : ℝ) (n : ℕ) (h1 : a^n = a + 1) (h2 : b^(2*n) = b + 3*a) (h3 : n ≥ 2) (h4 : 0 < a) (h5 : 0 < b) :
  a > b ∧ a > 1 ∧ b > 1 :=
sorry

end NUMINAMATH_GPT_ab_relationship_l1061_106140


namespace NUMINAMATH_GPT_total_selling_price_correct_l1061_106147

-- Define the given conditions
def cost_price_per_metre : ℝ := 72
def loss_per_metre : ℝ := 12
def total_metres_of_cloth : ℝ := 200

-- Define the selling price per metre
def selling_price_per_metre : ℝ := cost_price_per_metre - loss_per_metre

-- Define the total selling price
def total_selling_price : ℝ := selling_price_per_metre * total_metres_of_cloth

-- The theorem we want to prove
theorem total_selling_price_correct : 
  total_selling_price = 12000 := 
by
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l1061_106147


namespace NUMINAMATH_GPT_interest_problem_l1061_106117

theorem interest_problem
  (P : ℝ)
  (h : P * 0.04 * 5 = P * 0.05 * 4) : 
  (P * 0.04 * 5) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_interest_problem_l1061_106117


namespace NUMINAMATH_GPT_cost_of_jeans_l1061_106128

    variable (J S : ℝ)

    def condition1 := 3 * J + 6 * S = 104.25
    def condition2 := 4 * J + 5 * S = 112.15

    theorem cost_of_jeans (h1 : condition1 J S) (h2 : condition2 J S) : J = 16.85 := by
      sorry
    
end NUMINAMATH_GPT_cost_of_jeans_l1061_106128


namespace NUMINAMATH_GPT_volleyball_tournament_first_place_score_l1061_106152

theorem volleyball_tournament_first_place_score :
  ∃ (a b c d : ℕ), (a + b + c + d = 18) ∧ (a < b ∧ b < c ∧ c < d) ∧ (d = 6) :=
by
  sorry

end NUMINAMATH_GPT_volleyball_tournament_first_place_score_l1061_106152


namespace NUMINAMATH_GPT_total_books_l1061_106194

-- Given conditions
def susan_books : Nat := 600
def lidia_books : Nat := 4 * susan_books

-- The theorem to prove
theorem total_books : susan_books + lidia_books = 3000 :=
by
  unfold susan_books lidia_books
  sorry

end NUMINAMATH_GPT_total_books_l1061_106194


namespace NUMINAMATH_GPT_a_alone_completes_in_eight_days_l1061_106192

variable (a b : Type)
variables (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)

noncomputable def days := ℝ

axiom work_together_four_days : days_ab = 4
axiom work_together_266666_days : days_ab_2 = 8 / 3

theorem a_alone_completes_in_eight_days (a b : Type) (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)
  (work_together_four_days : days_ab = 4)
  (work_together_266666_days : days_ab_2 = 8 / 3) :
  days_a = 8 :=
by
  sorry

end NUMINAMATH_GPT_a_alone_completes_in_eight_days_l1061_106192


namespace NUMINAMATH_GPT_ratio_of_percent_changes_l1061_106184

noncomputable def price_decrease_ratio (original_price : ℝ) (new_price : ℝ) : ℝ :=
(original_price - new_price) / original_price * 100

noncomputable def units_increase_ratio (original_units : ℝ) (new_units : ℝ) : ℝ :=
(new_units - original_units) / original_units * 100

theorem ratio_of_percent_changes 
  (original_price new_price original_units new_units : ℝ)
  (h1 : new_price = 0.7 * original_price)
  (h2 : original_price * original_units = new_price * new_units)
  : (units_increase_ratio original_units new_units) / (price_decrease_ratio original_price new_price) = 1.4285714285714286 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_percent_changes_l1061_106184


namespace NUMINAMATH_GPT_center_circle_sum_eq_neg1_l1061_106189

theorem center_circle_sum_eq_neg1 
  (h k : ℝ) 
  (h_center : ∀ x y, (x - h)^2 + (y - k)^2 = 22) 
  (circle_eq : ∀ x y, x^2 + y^2 = 4*x - 6*y + 9) : 
  h + k = -1 := 
by 
  sorry

end NUMINAMATH_GPT_center_circle_sum_eq_neg1_l1061_106189


namespace NUMINAMATH_GPT_not_enrolled_eq_80_l1061_106171

variable (total_students : ℕ)
variable (french_students : ℕ)
variable (german_students : ℕ)
variable (spanish_students : ℕ)
variable (french_and_german : ℕ)
variable (german_and_spanish : ℕ)
variable (spanish_and_french : ℕ)
variable (all_three : ℕ)

noncomputable def students_not_enrolled_in_any_language 
  (total_students french_students german_students spanish_students french_and_german german_and_spanish spanish_and_french all_three : ℕ) : ℕ :=
  total_students - (french_students + german_students + spanish_students - french_and_german - german_and_spanish - spanish_and_french + all_three)

theorem not_enrolled_eq_80 : 
  students_not_enrolled_in_any_language 180 60 50 35 20 15 10 5 = 80 :=
  by
    unfold students_not_enrolled_in_any_language
    simp
    sorry

end NUMINAMATH_GPT_not_enrolled_eq_80_l1061_106171


namespace NUMINAMATH_GPT_line_tangent_to_circle_l1061_106110

noncomputable def circle_diameter : ℝ := 13
noncomputable def distance_from_center_to_line : ℝ := 6.5

theorem line_tangent_to_circle :
  ∀ (d r : ℝ), d = 13 → r = 6.5 → r = d/2 → distance_from_center_to_line = r → 
  (distance_from_center_to_line = r) := 
by
  intros d r hdiam hdist hradius hdistance
  sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l1061_106110


namespace NUMINAMATH_GPT_distribution_of_balls_into_boxes_l1061_106158

noncomputable def partitions_of_6_into_4_boxes : ℕ := 9

theorem distribution_of_balls_into_boxes :
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  ways = 9 :=
by
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  sorry

end NUMINAMATH_GPT_distribution_of_balls_into_boxes_l1061_106158


namespace NUMINAMATH_GPT_percentage_flowering_plants_l1061_106131

variable (P : ℝ)

theorem percentage_flowering_plants (h : 5 * (1 / 4) * (P / 100) * 80 = 40) : P = 40 :=
by
  -- This is where the proof would go, but we will use sorry to skip it for now
  sorry

end NUMINAMATH_GPT_percentage_flowering_plants_l1061_106131


namespace NUMINAMATH_GPT_solve_for_x_l1061_106160

theorem solve_for_x (x : ℝ) (h : 0 < x) (h_property : (x / 100) * x^2 = 9) : x = 10 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1061_106160


namespace NUMINAMATH_GPT_find_original_price_l1061_106154

noncomputable def original_price_per_bottle (P : ℝ) : Prop :=
  let discounted_price := 0.80 * P
  let final_price_per_bottle := discounted_price - 2.00
  3 * final_price_per_bottle = 30

theorem find_original_price : ∃ P : ℝ, original_price_per_bottle P ∧ P = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_original_price_l1061_106154


namespace NUMINAMATH_GPT_initial_short_bushes_l1061_106103

theorem initial_short_bushes (B : ℕ) (H1 : B + 20 = 57) : B = 37 :=
by
  sorry

end NUMINAMATH_GPT_initial_short_bushes_l1061_106103


namespace NUMINAMATH_GPT_water_percentage_in_tomato_juice_l1061_106120

-- Definitions from conditions
def tomato_juice_volume := 80 -- in liters
def tomato_puree_volume := 10 -- in liters
def tomato_puree_water_percentage := 20 -- in percent (20%)

-- Need to prove percentage of water in tomato juice is 20%
theorem water_percentage_in_tomato_juice : 
  (100 - tomato_puree_water_percentage) * tomato_puree_volume / tomato_juice_volume = 20 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_water_percentage_in_tomato_juice_l1061_106120


namespace NUMINAMATH_GPT_number_of_pages_to_copy_l1061_106167

-- Definitions based on the given conditions
def total_budget : ℕ := 5000
def service_charge : ℕ := 500
def copy_cost : ℕ := 3

-- Derived definition based on the conditions
def remaining_budget : ℕ := total_budget - service_charge

-- The statement we need to prove
theorem number_of_pages_to_copy : (remaining_budget / copy_cost) = 1500 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_pages_to_copy_l1061_106167


namespace NUMINAMATH_GPT_sqrt_calculation_l1061_106168

theorem sqrt_calculation :
  Real.sqrt ((2:ℝ)^4 * 3^2 * 5^2) = 60 := 
by sorry

end NUMINAMATH_GPT_sqrt_calculation_l1061_106168


namespace NUMINAMATH_GPT_total_questions_in_test_l1061_106182

theorem total_questions_in_test :
  ∃ x, (5 * x = total_questions) ∧ 
       (20 : ℚ) / total_questions > (60 / 100 : ℚ) ∧ 
       (20 : ℚ) / total_questions < (70 / 100 : ℚ) ∧ 
       total_questions = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_questions_in_test_l1061_106182


namespace NUMINAMATH_GPT_rectangular_prism_diagonal_l1061_106153

theorem rectangular_prism_diagonal 
  (a b c : ℝ)
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2 = 25) :=
by {
  -- Sorry to skip the proof steps
  sorry
}

end NUMINAMATH_GPT_rectangular_prism_diagonal_l1061_106153


namespace NUMINAMATH_GPT_second_number_division_l1061_106136

theorem second_number_division (d x r : ℕ) (h1 : d = 16) (h2 : 25 % d = r) (h3 : 105 % d = r) (h4 : r = 9) : x % d = r → x = 41 :=
by 
  simp [h1, h2, h3, h4] 
  sorry

end NUMINAMATH_GPT_second_number_division_l1061_106136


namespace NUMINAMATH_GPT_lines_coplanar_iff_k_eq_neg2_l1061_106119

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
(2 + s, 4 - k * s, 2 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
(t, 2 + 2 * t, 3 - t)

theorem lines_coplanar_iff_k_eq_neg2 :
  (∃ s t : ℝ, line1 s k = line2 t) → k = -2 :=
by
  sorry

end NUMINAMATH_GPT_lines_coplanar_iff_k_eq_neg2_l1061_106119


namespace NUMINAMATH_GPT_parabola_max_value_l1061_106114

theorem parabola_max_value 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x = - (x + 1)^2 + 3) : 
  ∃ x, y x = 3 ∧ ∀ x', y x' ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_max_value_l1061_106114


namespace NUMINAMATH_GPT_smallest_number_of_marbles_l1061_106177

-- Define the conditions
variables (r w b g n : ℕ)
def valid_total (r w b g n : ℕ) := r + w + b + g = n
def valid_probability_4r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w) * (r * (r - 1) * (r - 2) / 6)
def valid_probability_1w3r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w * b * (r * (r - 1) / 2))
def valid_probability_1w1b2r (r w b g n : ℕ) := w * b * (r * (r - 1) / 2) = w * b * g * r

theorem smallest_number_of_marbles :
  ∃ n r w b g, valid_total r w b g n ∧
  valid_probability_4r r w b g n ∧
  valid_probability_1w3r r w b g n ∧
  valid_probability_1w1b2r r w b g n ∧ 
  n = 21 :=
  sorry

end NUMINAMATH_GPT_smallest_number_of_marbles_l1061_106177


namespace NUMINAMATH_GPT_edward_initial_amount_l1061_106191

-- Defining the conditions
def cost_books : ℕ := 6
def cost_pens : ℕ := 16
def cost_notebook : ℕ := 5
def cost_pencil_case : ℕ := 3
def amount_left : ℕ := 19

-- Mathematical statement to prove
theorem edward_initial_amount : 
  cost_books + cost_pens + cost_notebook + cost_pencil_case + amount_left = 49 :=
by
  sorry

end NUMINAMATH_GPT_edward_initial_amount_l1061_106191


namespace NUMINAMATH_GPT_sequence_an_form_sum_cn_terms_l1061_106124

theorem sequence_an_form (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2) :
  ∀ n : ℕ, b_n n = 2 * n + 1 :=
sorry 

theorem sum_cn_terms (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (c_n : ℕ → ℕ) (T_n : ℕ → ℕ)
    (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2)
    (hb : ∀ n : ℕ, b_n n = 2 * n + 1)
    (hc : ∀ n : ℕ, c_n n = 1 / (b_n n * b_n (n + 1))) :
  ∀ n : ℕ, T_n n = n / (3 * (2 * n + 3)) :=
sorry

end NUMINAMATH_GPT_sequence_an_form_sum_cn_terms_l1061_106124


namespace NUMINAMATH_GPT_min_vases_required_l1061_106186

theorem min_vases_required (carnations roses tulips lilies : ℕ)
  (flowers_in_A flowers_in_B flowers_in_C : ℕ) 
  (total_flowers : ℕ) 
  (h_carnations : carnations = 10) 
  (h_roses : roses = 25) 
  (h_tulips : tulips = 15) 
  (h_lilies : lilies = 20)
  (h_flowers_in_A : flowers_in_A = 4) 
  (h_flowers_in_B : flowers_in_B = 6) 
  (h_flowers_in_C : flowers_in_C = 8)
  (h_total_flowers : total_flowers = carnations + roses + tulips + lilies) :
  total_flowers = 70 → 
  (exists vases_A vases_B vases_C : ℕ, 
    vases_A = 0 ∧ 
    vases_B = 1 ∧ 
    vases_C = 8 ∧ 
    total_flowers = vases_A * flowers_in_A + vases_B * flowers_in_B + vases_C * flowers_in_C) :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_vases_required_l1061_106186


namespace NUMINAMATH_GPT_standard_deviation_less_than_l1061_106144

theorem standard_deviation_less_than:
  ∀ (μ σ : ℝ)
  (h1 : μ = 55)
  (h2 : μ - 3 * σ > 48),
  σ < 7 / 3 :=
by
  intros μ σ h1 h2
  sorry

end NUMINAMATH_GPT_standard_deviation_less_than_l1061_106144


namespace NUMINAMATH_GPT_avg_marks_second_class_l1061_106164

theorem avg_marks_second_class
  (x : ℝ)
  (avg_class1 : ℝ)
  (avg_total : ℝ)
  (n1 n2 : ℕ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg_class1 = 30)
  (h4: avg_total = 48.75)
  (h5 : (n1 * avg_class1 + n2 * x) / (n1 + n2) = avg_total) :
  x = 60 := by
  sorry

end NUMINAMATH_GPT_avg_marks_second_class_l1061_106164


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1061_106129

theorem arithmetic_expression_evaluation : (8 / 2 - 3 * 2 + 5^2 / 5) = 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1061_106129


namespace NUMINAMATH_GPT_sub_fraction_l1061_106135

theorem sub_fraction (a b c d : ℚ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 1) (h4 : d = 6) : (a / b) - (c / d) = 7 / 18 := 
by
  sorry

end NUMINAMATH_GPT_sub_fraction_l1061_106135


namespace NUMINAMATH_GPT_village_population_rate_decrease_l1061_106118

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end NUMINAMATH_GPT_village_population_rate_decrease_l1061_106118


namespace NUMINAMATH_GPT_max_profit_at_150_l1061_106185

-- Define the conditions
def purchase_price : ℕ := 80
def total_items : ℕ := 1000
def selling_price_initial : ℕ := 100
def sales_volume_decrease : ℕ := 5

-- The profit function
def profit (x : ℕ) : ℤ :=
  (selling_price_initial + x) * (total_items - sales_volume_decrease * x) - purchase_price * total_items

-- The statement to prove: the selling price of 150 yuan/item maximizes the profit at 32500 yuan.
theorem max_profit_at_150 : profit 50 = 32500 := by
  sorry

end NUMINAMATH_GPT_max_profit_at_150_l1061_106185


namespace NUMINAMATH_GPT_find_ratio_l1061_106151

variable {x y z : ℝ}

theorem find_ratio
  (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_find_ratio_l1061_106151


namespace NUMINAMATH_GPT_minimum_adjacent_white_pairs_l1061_106196

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end NUMINAMATH_GPT_minimum_adjacent_white_pairs_l1061_106196


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1061_106138

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 2 * x^2) : (0, 1 / 8) = (0, 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1061_106138


namespace NUMINAMATH_GPT_product_of_integers_l1061_106100

theorem product_of_integers (X Y Z W : ℚ) (h_sum : X + Y + Z + W = 100)
  (h_relation : X + 5 = Y - 5 ∧ Y - 5 = 3 * Z ∧ 3 * Z = W / 3) :
  X * Y * Z * W = 29390625 / 256 := by
  sorry

end NUMINAMATH_GPT_product_of_integers_l1061_106100


namespace NUMINAMATH_GPT_even_func_min_value_l1061_106109

theorem even_func_min_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_neq_a : a ≠ 1) (h_neq_b : b ≠ 1) (h_even : ∀ x : ℝ, a^x + b^x = a^(-x) + b^(-x)) :
  ab = 1 → (∃ y : ℝ, y = (1 / a + 4 / b) ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_even_func_min_value_l1061_106109


namespace NUMINAMATH_GPT_sides_of_triangle_inequality_l1061_106149

theorem sides_of_triangle_inequality (a b c : ℝ) (h : a + b > c) : a + b > c := 
by 
  exact h

end NUMINAMATH_GPT_sides_of_triangle_inequality_l1061_106149


namespace NUMINAMATH_GPT_solve_repeating_decimals_sum_l1061_106166

def repeating_decimals_sum : Prop :=
  let x := (1 : ℚ) / 3
  let y := (4 : ℚ) / 999
  let z := (5 : ℚ) / 9999
  x + y + z = 3378 / 9999

theorem solve_repeating_decimals_sum : repeating_decimals_sum := 
by 
  sorry

end NUMINAMATH_GPT_solve_repeating_decimals_sum_l1061_106166


namespace NUMINAMATH_GPT_black_piece_probability_l1061_106148

-- Definitions based on conditions
def total_pieces : ℕ := 10 + 5
def black_pieces : ℕ := 10

-- Probability calculation
def probability_black : ℚ := black_pieces / total_pieces

-- Statement to prove
theorem black_piece_probability : probability_black = 2/3 := by
  sorry -- proof to be filled in later

end NUMINAMATH_GPT_black_piece_probability_l1061_106148


namespace NUMINAMATH_GPT_range_of_m_three_zeros_l1061_106139

theorem range_of_m_three_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x^3 - 3*x + m = 0) ∧ (y^3 - 3*y + m = 0) ∧ (z^3 - 3*z + m = 0)) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_three_zeros_l1061_106139


namespace NUMINAMATH_GPT_other_type_jelly_amount_l1061_106123

-- Combined total amount of jelly
def total_jelly := 6310

-- Amount of one type of jelly
def type_one_jelly := 4518

-- Amount of the other type of jelly
def type_other_jelly := total_jelly - type_one_jelly

theorem other_type_jelly_amount :
  type_other_jelly = 1792 :=
by
  sorry

end NUMINAMATH_GPT_other_type_jelly_amount_l1061_106123


namespace NUMINAMATH_GPT_original_price_l1061_106108

-- Definitions based on the problem conditions
variables (P : ℝ)

def john_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * P

def jane_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * (0.9 * P)

def price_difference (P : ℝ) : ℝ :=
  john_payment P - jane_payment P

theorem original_price (h : price_difference P = 0.51) : P = 34 := 
by
  sorry

end NUMINAMATH_GPT_original_price_l1061_106108
