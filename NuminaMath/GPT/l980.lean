import Mathlib

namespace NUMINAMATH_GPT_max_three_numbers_condition_l980_98082

theorem max_three_numbers_condition (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → (x i)^2 > (x j) * (x k)) : n ≤ 3 := 
sorry

end NUMINAMATH_GPT_max_three_numbers_condition_l980_98082


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l980_98019

variable (p q : Prop)

theorem necessary_but_not_sufficient (hp : p) : p ∧ q ↔ p ∧ (p ∧ q → q) :=
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l980_98019


namespace NUMINAMATH_GPT_complement_of_A_in_U_l980_98010

def U : Set ℕ := {1,3,5,7,9}
def A : Set ℕ := {1,9}
def complement_U_A : Set ℕ := {3,5,7}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l980_98010


namespace NUMINAMATH_GPT_sum_and_product_of_white_are_white_l980_98066

-- Definitions based on the conditions
def is_colored_black_or_white (n : ℕ) : Prop :=
  true -- This is a simplified assumption since this property is always true.

def is_black (n : ℕ) : Prop := (n % 2 = 0)
def is_white (n : ℕ) : Prop := (n % 2 = 1)

-- Conditions given in the problem
axiom sum_diff_colors_is_black (a b : ℕ) (ha : is_black a) (hb : is_white b) : is_black (a + b)
axiom infinitely_many_whites : ∀ n, ∃ m ≥ n, is_white m

-- Statement to prove that the sum and product of two white numbers are white
theorem sum_and_product_of_white_are_white (a b : ℕ) (ha : is_white a) (hb : is_white b) : 
  is_white (a + b) ∧ is_white (a * b) :=
sorry

end NUMINAMATH_GPT_sum_and_product_of_white_are_white_l980_98066


namespace NUMINAMATH_GPT_expression_range_l980_98021

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y : ℝ, y = (a * Real.cos x + b * Real.sin x + c) / (Real.sqrt (a^2 + b^2 + c^2)) 
           ∧ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_expression_range_l980_98021


namespace NUMINAMATH_GPT_calculate_value_l980_98075

theorem calculate_value : (24 + 12) / ((5 - 3) * 2) = 9 := by 
  sorry

end NUMINAMATH_GPT_calculate_value_l980_98075


namespace NUMINAMATH_GPT_angle_C_is_120_degrees_l980_98087

theorem angle_C_is_120_degrees (l m : ℝ) (A B C : ℝ) (hal : l = m) 
  (hA : A = 100) (hB : B = 140) : C = 120 := 
by 
  sorry

end NUMINAMATH_GPT_angle_C_is_120_degrees_l980_98087


namespace NUMINAMATH_GPT_similarity_coefficient_l980_98060

theorem similarity_coefficient (α : ℝ) :
  (2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)))
  = 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end NUMINAMATH_GPT_similarity_coefficient_l980_98060


namespace NUMINAMATH_GPT_students_difference_l980_98011

theorem students_difference 
  (C : ℕ → ℕ) 
  (hC1 : C 1 = 24) 
  (hC2 : ∀ n, C n.succ = C n - d)
  (h_total : C 1 + C 2 + C 3 + C 4 + C 5 = 100) :
  d = 2 :=
by sorry

end NUMINAMATH_GPT_students_difference_l980_98011


namespace NUMINAMATH_GPT_men_women_equal_after_city_Y_l980_98052

variable (M W M' W' : ℕ)

-- Initial conditions: total passengers, women to men ratio
variable (h1 : M + W = 72)
variable (h2 : W = M / 2)

-- Changes in city Y: men leave, women enter
variable (h3 : M' = M - 16)
variable (h4 : W' = W + 8)

theorem men_women_equal_after_city_Y (h1 : M + W = 72) (h2 : W = M / 2) (h3 : M' = M - 16) (h4 : W' = W + 8) : 
  M' = W' := 
by 
  sorry

end NUMINAMATH_GPT_men_women_equal_after_city_Y_l980_98052


namespace NUMINAMATH_GPT_three_pow_y_plus_two_l980_98017

theorem three_pow_y_plus_two (y : ℕ) (h : 3^y = 81) : 3^(y+2) = 729 := sorry

end NUMINAMATH_GPT_three_pow_y_plus_two_l980_98017


namespace NUMINAMATH_GPT_find_other_number_l980_98096

theorem find_other_number (hcf lcm a b: ℕ) (hcf_value: hcf = 12) (lcm_value: lcm = 396) (a_value: a = 36) (gcd_ab: Nat.gcd a b = hcf) (lcm_ab: Nat.lcm a b = lcm) : b = 132 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l980_98096


namespace NUMINAMATH_GPT_simplify_division_l980_98070

noncomputable def simplify_expression (m : ℝ) : ℝ :=
  (m^2 - 3 * m + 1) / m + 1

noncomputable def divisor_expression (m : ℝ) : ℝ :=
  (m^2 - 1) / m

theorem simplify_division (m : ℝ) (hm1 : m ≠ 0) (hm2 : m ≠ 1) (hm3 : m ≠ -1) :
  (simplify_expression m) / (divisor_expression m) = (m - 1) / (m + 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_division_l980_98070


namespace NUMINAMATH_GPT_xy_in_B_l980_98076

def A : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = m * a^2 + k * a * b + m * b^2}

def B : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = a^2 + k * a * b + m^2 * b^2}

theorem xy_in_B (x y : ℤ) (h1 : x ∈ A) (h2 : y ∈ A) : x * y ∈ B := by
  sorry

end NUMINAMATH_GPT_xy_in_B_l980_98076


namespace NUMINAMATH_GPT_horse_saddle_ratio_l980_98098

theorem horse_saddle_ratio (total_cost : ℕ) (saddle_cost : ℕ) (horse_cost : ℕ) 
  (h_total : total_cost = 5000)
  (h_saddle : saddle_cost = 1000)
  (h_sum : horse_cost + saddle_cost = total_cost) : 
  horse_cost / saddle_cost = 4 :=
by sorry

end NUMINAMATH_GPT_horse_saddle_ratio_l980_98098


namespace NUMINAMATH_GPT_find_speed_second_train_l980_98073

noncomputable def speed_second_train (length_train1 length_train2 : ℝ) (speed_train1_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let total_distance := length_train1 + length_train2
  let relative_speed_mps := total_distance / time_to_cross
  let speed_train2_mps := speed_train1_mps - relative_speed_mps
  speed_train2_mps * 3600 / 1000

theorem find_speed_second_train :
  speed_second_train 380 540 72 91.9926405887529 = 36 := by
  sorry

end NUMINAMATH_GPT_find_speed_second_train_l980_98073


namespace NUMINAMATH_GPT_friend_saves_per_week_l980_98094

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_friend_saves_per_week_l980_98094


namespace NUMINAMATH_GPT_A_sym_diff_B_l980_98030

-- Definitions of sets and operations
def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {y | ∃ x : ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x : ℝ, y = -(x-1)^2 + 2}

-- The target equality to prove
theorem A_sym_diff_B : sym_diff A B = (({y | y ≤ 0}) ∪ ({y | y > 2})) :=
by
  sorry

end NUMINAMATH_GPT_A_sym_diff_B_l980_98030


namespace NUMINAMATH_GPT_triangle_angle_B_l980_98078

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h : a / b = 3 / Real.sqrt 7) (h2 : b / c = Real.sqrt 7 / 2) : B = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_B_l980_98078


namespace NUMINAMATH_GPT_rachel_should_budget_940_l980_98033

-- Define the prices for Sara's shoes and dress
def sara_shoes : ℝ := 50
def sara_dress : ℝ := 200

-- Define the prices for Tina's shoes and dress
def tina_shoes : ℝ := 70
def tina_dress : ℝ := 150

-- Define the total spending for Sara and Tina, and Rachel's budget
def rachel_budget (sara_shoes sara_dress tina_shoes tina_dress : ℝ) : ℝ := 
  2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_should_budget_940 : 
  rachel_budget sara_shoes sara_dress tina_shoes tina_dress = 940 := 
by
  -- skip the proof
  sorry 

end NUMINAMATH_GPT_rachel_should_budget_940_l980_98033


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l980_98042

theorem no_real_roots_of_quadratic (k : ℝ) (h : 12 - 3 * k < 0) : ∀ (x : ℝ), ¬ (x^2 + 4 * x + k = 0) := by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l980_98042


namespace NUMINAMATH_GPT_compute_n_pow_m_l980_98074

-- Given conditions
variables (n m : ℕ)
axiom n_eq : n = 3
axiom n_plus_one_eq_2m : n + 1 = 2 * m

-- Goal: Prove n^m = 9
theorem compute_n_pow_m : n^m = 9 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_compute_n_pow_m_l980_98074


namespace NUMINAMATH_GPT_additional_tanks_needed_l980_98099

theorem additional_tanks_needed 
    (initial_tanks : ℕ) 
    (initial_capacity_per_tank : ℕ) 
    (total_fish_needed : ℕ) 
    (new_capacity_per_tank : ℕ)
    (h_t1 : initial_tanks = 3)
    (h_t2 : initial_capacity_per_tank = 15)
    (h_t3 : total_fish_needed = 75)
    (h_t4 : new_capacity_per_tank = 10) : 
    (total_fish_needed - initial_tanks * initial_capacity_per_tank) / new_capacity_per_tank = 3 := 
by {
    sorry
}

end NUMINAMATH_GPT_additional_tanks_needed_l980_98099


namespace NUMINAMATH_GPT_value_of_complex_fraction_l980_98043

theorem value_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) : ((1 - i) / (1 + i)) ^ 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_complex_fraction_l980_98043


namespace NUMINAMATH_GPT_min_perimeter_triangle_l980_98090

theorem min_perimeter_triangle (a b c : ℝ) (cosC : ℝ) :
  a + b = 10 ∧ cosC = -1/2 ∧ c^2 = (a - 5)^2 + 75 →
  a + b + c = 10 + 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_min_perimeter_triangle_l980_98090


namespace NUMINAMATH_GPT_interest_rate_second_type_l980_98003

variable (totalInvestment : ℝ) (interestFirstTypeRate : ℝ) (investmentSecondType : ℝ) (totalInterestRate : ℝ) 
variable [Nontrivial ℝ]

theorem interest_rate_second_type :
    totalInvestment = 100000 ∧
    interestFirstTypeRate = 0.09 ∧
    investmentSecondType = 29999.999999999993 ∧
    totalInterestRate = 9 + 3 / 5 →
    (9.6 * totalInvestment - (interestFirstTypeRate * (totalInvestment - investmentSecondType))) / investmentSecondType = 0.11 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_second_type_l980_98003


namespace NUMINAMATH_GPT_sum_of_prime_h_l980_98001

def h (n : ℕ) := n^4 - 380 * n^2 + 600

theorem sum_of_prime_h (S : Finset ℕ) (hS : S = { n | Nat.Prime (h n) }) :
  S.sum h = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_prime_h_l980_98001


namespace NUMINAMATH_GPT_degree_of_g_l980_98089

theorem degree_of_g 
  (f : Polynomial ℤ)
  (g : Polynomial ℤ) 
  (h₁ : f = -9 * Polynomial.X^5 + 4 * Polynomial.X^3 - 2 * Polynomial.X + 6)
  (h₂ : (f + g).degree = 2) :
  g.degree = 5 :=
sorry

end NUMINAMATH_GPT_degree_of_g_l980_98089


namespace NUMINAMATH_GPT_calculate_expression_l980_98004

theorem calculate_expression :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l980_98004


namespace NUMINAMATH_GPT_problem_equivalent_l980_98057

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) / Real.log 4 - 1

theorem problem_equivalent : 
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = Real.log (x + 2) / Real.log 4 - 1) →
  {x : ℝ | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  intro h_even h_def
  sorry

end NUMINAMATH_GPT_problem_equivalent_l980_98057


namespace NUMINAMATH_GPT_pyramid_levels_l980_98071

theorem pyramid_levels (n : ℕ) (h : (n * (n + 1) * (2 * n + 1)) / 6 = 225) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_levels_l980_98071


namespace NUMINAMATH_GPT_part_1_part_2_l980_98028

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def B (m : ℝ) : Set ℝ := { x | x^2 - (2*m + 1)*x + 2*m < 0 }

theorem part_1 (m : ℝ) (h : m < 1/2) : 
  B m = { x | 2*m < x ∧ x < 1 } := 
sorry

theorem part_2 (m : ℝ) : 
  (A ∪ B m = A) ↔ -1/2 ≤ m ∧ m ≤ 1 := 
sorry

end NUMINAMATH_GPT_part_1_part_2_l980_98028


namespace NUMINAMATH_GPT_proof_correct_option_C_l980_98069

def line := Type
def plane := Type
def perp (m : line) (α : plane) : Prop := sorry
def parallel (n : line) (α : plane) : Prop := sorry
def perpnal (m n: line): Prop := sorry 

variables (m n : line) (α β γ : plane)

theorem proof_correct_option_C : perp m α → parallel n α → perpnal m n := sorry

end NUMINAMATH_GPT_proof_correct_option_C_l980_98069


namespace NUMINAMATH_GPT_Danny_found_11_wrappers_l980_98077

theorem Danny_found_11_wrappers :
  ∃ wrappers_at_park : ℕ,
  (wrappers_at_park = 11) ∧
  (∃ bottle_caps : ℕ, bottle_caps = 12) ∧
  (∃ found_bottle_caps : ℕ, found_bottle_caps = 58) ∧
  (wrappers_at_park + 1 = bottle_caps) :=
by
  sorry

end NUMINAMATH_GPT_Danny_found_11_wrappers_l980_98077


namespace NUMINAMATH_GPT_slope_of_AB_l980_98031

theorem slope_of_AB (k : ℝ) (y1 y2 x1 x2 : ℝ) 
  (hP : (1, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1})
  (hPA_eq : ∀ x, (x, y1) ∈ {p : ℝ × ℝ | p.2 = k * p.1 - k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hPB_eq : ∀ x, (x, y2) ∈ {p : ℝ × ℝ | p.2 = -k * p.1 + k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hx1 : y1 = k * x1 - k + Real.sqrt 2) 
  (hx2 : y2 = -k * x2 + k + Real.sqrt 2) :
  ((y2 - y1) / (x2 - x1)) = -2 - 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_AB_l980_98031


namespace NUMINAMATH_GPT_average_letters_per_day_l980_98038

theorem average_letters_per_day:
  let letters_per_day := [7, 10, 3, 5, 12]
  (letters_per_day.sum / letters_per_day.length : ℝ) = 7.4 :=
by
  sorry

end NUMINAMATH_GPT_average_letters_per_day_l980_98038


namespace NUMINAMATH_GPT_pipe_filling_time_l980_98044

theorem pipe_filling_time (t : ℕ) (h : 2 * (1 / t + 1 / 15) + 10 * (1 / 15) = 1) : t = 10 := by
  sorry

end NUMINAMATH_GPT_pipe_filling_time_l980_98044


namespace NUMINAMATH_GPT_monkeys_and_bananas_l980_98027

theorem monkeys_and_bananas :
  (∀ (m n t : ℕ), m * t = n → (∀ (m' n' t' : ℕ), n = m * (t / t') → n' = (m' * t') / t → n' = n → m' = m)) →
  (6 : ℕ) = 6 :=
by
  intros H
  let m := 6
  let n := 6
  let t := 6
  have H1 : m * t = n := by sorry
  let k := 18
  let t' := 18
  have H2 : n = m * (t / t') := by sorry
  let n' := 18
  have H3 : n' = (m * t') / t := by sorry
  have H4 : n' = n := by sorry
  exact H m n t H1 6 n' t' H2 H3 H4

end NUMINAMATH_GPT_monkeys_and_bananas_l980_98027


namespace NUMINAMATH_GPT_parabola_points_relationship_l980_98045

theorem parabola_points_relationship :
  let y_1 := (-2)^2 + 2 * (-2) - 9
  let y_2 := 1^2 + 2 * 1 - 9
  let y_3 := 3^2 + 2 * 3 - 9
  y_3 > y_2 ∧ y_2 > y_1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_points_relationship_l980_98045


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_a9_sum_l980_98079

theorem arithmetic_sequence_a2_a9_sum 
  (a : ℕ → ℝ) (d a₁ : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S10 : 10 * a 1 + 45 * d = 120) :
  a 2 + a 9 = 24 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_a9_sum_l980_98079


namespace NUMINAMATH_GPT_largest_angle_of_convex_hexagon_l980_98083

noncomputable def hexagon_largest_angle (x : ℚ) : ℚ :=
  max (6 * x - 3) (max (5 * x + 1) (max (4 * x - 4) (max (3 * x) (max (2 * x + 2) x))))

theorem largest_angle_of_convex_hexagon (x : ℚ) (h : x + (2*x+2) + 3*x + (4*x-4) + (5*x+1) + (6*x-3) = 720) : 
  hexagon_largest_angle x = 4281 / 21 := 
sorry

end NUMINAMATH_GPT_largest_angle_of_convex_hexagon_l980_98083


namespace NUMINAMATH_GPT_largest_of_choices_l980_98035

theorem largest_of_choices :
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  A < D ∧ B < D ∧ C < D ∧ E < D :=
by
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  sorry

end NUMINAMATH_GPT_largest_of_choices_l980_98035


namespace NUMINAMATH_GPT_shaded_areas_different_l980_98040

/-
Question: How do the shaded areas of three different large squares (I, II, and III) compare?
Conditions:
1. Square I has diagonals drawn, and small squares are shaded at each corner where diagonals meet the sides.
2. Square II has vertical and horizontal lines drawn through the midpoints, creating four smaller squares, with one centrally shaded.
3. Square III has one diagonal from one corner to the center and a straight line from the midpoint of the opposite side to the center, creating various triangles and trapezoids, with a trapezoid area around the center being shaded.
Proof:
Prove that the shaded areas of squares I, II, and III are all different given the conditions on how squares I, II, and III are partitioned and shaded.
-/
theorem shaded_areas_different :
  ∀ (a : ℝ) (A1 A2 A3 : ℝ), (A1 = 1/4 * a^2) ∧ (A2 = 1/4 * a^2) ∧ (A3 = 3/8 * a^2) → 
  A1 ≠ A3 ∧ A2 ≠ A3 :=
by
  sorry

end NUMINAMATH_GPT_shaded_areas_different_l980_98040


namespace NUMINAMATH_GPT_piecewise_function_continuity_l980_98095

theorem piecewise_function_continuity :
  (∀ x, if x > (3 : ℝ) 
        then 2 * (a : ℝ) * x + 4 = (x : ℝ) ^ 2 - 1
        else if x < -1 
        then 3 * (x : ℝ) - (c : ℝ) = (x : ℝ) ^ 2 - 1
        else (x : ℝ) ^ 2 - 1 = (x : ℝ) ^ 2 - 1) →
  a = 2 / 3 →
  c = -3 →
  a + c = -7 / 3 :=
by
  intros h ha hc
  simp [ha, hc]
  sorry

end NUMINAMATH_GPT_piecewise_function_continuity_l980_98095


namespace NUMINAMATH_GPT_mn_eq_neg_infty_to_0_l980_98059

-- Definitions based on the conditions
def M : Set ℝ := {y | y ≤ 2}
def N : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Set difference definition
def set_diff (A B : Set ℝ) : Set ℝ := {y | y ∈ A ∧ y ∉ B}

-- The proof statement we need to prove
theorem mn_eq_neg_infty_to_0 : set_diff M N = {y | y < 0} :=
  sorry  -- Proof will go here

end NUMINAMATH_GPT_mn_eq_neg_infty_to_0_l980_98059


namespace NUMINAMATH_GPT_tan_diff_l980_98024

theorem tan_diff (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) : Real.tan (α - β) = 1/3 := 
sorry

end NUMINAMATH_GPT_tan_diff_l980_98024


namespace NUMINAMATH_GPT_total_fuel_needed_l980_98041

/-- Given that Car B can travel 30 miles per gallon and needs to cover a distance of 750 miles,
    and Car C has a fuel consumption rate of 20 miles per gallon and will travel 900 miles,
    prove that the total combined fuel required for Cars B and C is 70 gallons. -/
theorem total_fuel_needed (miles_per_gallon_B : ℕ) (miles_per_gallon_C : ℕ)
  (distance_B : ℕ) (distance_C : ℕ)
  (hB : miles_per_gallon_B = 30) (hC : miles_per_gallon_C = 20)
  (dB : distance_B = 750) (dC : distance_C = 900) :
  (distance_B / miles_per_gallon_B) + (distance_C / miles_per_gallon_C) = 70 := by {
    sorry 
}

end NUMINAMATH_GPT_total_fuel_needed_l980_98041


namespace NUMINAMATH_GPT_toms_profit_l980_98068

noncomputable def cost_of_flour : Int :=
  let flour_needed := 500
  let bag_size := 50
  let bag_cost := 20
  (flour_needed / bag_size) * bag_cost

noncomputable def cost_of_salt : Int :=
  let salt_needed := 10
  let salt_cost_per_pound := (2 / 10)  -- Represent $0.2 as a fraction to maintain precision with integers in Lean
  salt_needed * salt_cost_per_pound

noncomputable def total_expenses : Int :=
  let flour_cost := cost_of_flour
  let salt_cost := cost_of_salt
  let promotion_cost := 1000
  flour_cost + salt_cost + promotion_cost

noncomputable def revenue_from_tickets : Int :=
  let ticket_price := 20
  let tickets_sold := 500
  tickets_sold * ticket_price

noncomputable def profit : Int :=
  revenue_from_tickets - total_expenses

theorem toms_profit : profit = 8798 :=
  by
    sorry

end NUMINAMATH_GPT_toms_profit_l980_98068


namespace NUMINAMATH_GPT_total_cost_of_apples_l980_98029

def original_price_per_pound : ℝ := 1.6
def price_increase_percentage : ℝ := 0.25
def number_of_family_members : ℕ := 4
def pounds_per_person : ℝ := 2

theorem total_cost_of_apples : 
  let new_price_per_pound := original_price_per_pound * (1 + price_increase_percentage)
  let total_pounds := pounds_per_person * number_of_family_members
  let total_cost := total_pounds * new_price_per_pound
  total_cost = 16 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_apples_l980_98029


namespace NUMINAMATH_GPT_midpoint_reflection_sum_l980_98026

/-- 
Points P and R are located at (2, 1) and (12, 15) respectively. 
Point M is the midpoint of segment PR. 
Segment PR is reflected over the y-axis.
We want to prove that the sum of the coordinates of the image of point M (the midpoint of the reflected segment) is 1.
-/
theorem midpoint_reflection_sum : 
  let P := (2, 1)
  let R := (12, 15)
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P_image := (-P.1, P.2)
  let R_image := (-R.1, R.2)
  let M' := ((P_image.1 + R_image.1) / 2, (P_image.2 + R_image.2) / 2)
  (M'.1 + M'.2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_reflection_sum_l980_98026


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l980_98009

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

theorem sufficient_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l980_98009


namespace NUMINAMATH_GPT_isabel_initial_amount_l980_98056

theorem isabel_initial_amount (X : ℝ) (h : X / 2 - X / 4 = 51) : X = 204 :=
sorry

end NUMINAMATH_GPT_isabel_initial_amount_l980_98056


namespace NUMINAMATH_GPT_people_after_five_years_l980_98039

noncomputable def population_in_year : ℕ → ℕ
| 0       => 20
| (k + 1) => 4 * population_in_year k - 18

theorem people_after_five_years : population_in_year 5 = 14382 := by
  sorry

end NUMINAMATH_GPT_people_after_five_years_l980_98039


namespace NUMINAMATH_GPT_min_expr_value_l980_98018

theorem min_expr_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) :
  (∃ a, a = (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ∧ a ≥ 0) → 
  (∀ (u v : ℝ), u = x + 2 → v = 3 * y + 4 → u * v = 16) →
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 :=
sorry

end NUMINAMATH_GPT_min_expr_value_l980_98018


namespace NUMINAMATH_GPT_monotonic_if_and_only_if_extreme_point_inequality_l980_98062

noncomputable def f (x a : ℝ) : ℝ := x^2 - 1 + a * Real.log (1 - x)

def is_monotonic (a : ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x a ≤ f y a

theorem monotonic_if_and_only_if (a : ℝ) : 
  is_monotonic a ↔ a ≥ 0.5 :=
sorry

theorem extreme_point_inequality (a : ℝ) (x1 x2 : ℝ) (hₐ : 0 < a ∧ a < 0.5) 
  (hx : x1 < x2) (hx₁₂ : f x1 a = f x2 a) : 
  f x1 a / x2 > f x2 a / x1 :=
sorry

end NUMINAMATH_GPT_monotonic_if_and_only_if_extreme_point_inequality_l980_98062


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l980_98091

theorem perpendicular_lines_condition (a : ℝ) :
  (¬ a = 1/2 ∨ ¬ a = -1/2) ∧ a * (-4 * a) = -1 ↔ a = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l980_98091


namespace NUMINAMATH_GPT_max_value_of_x_squared_plus_xy_plus_y_squared_l980_98092

theorem max_value_of_x_squared_plus_xy_plus_y_squared
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 - x * y + y^2 = 9) : 
  (x^2 + x * y + y^2) ≤ 27 :=
sorry

end NUMINAMATH_GPT_max_value_of_x_squared_plus_xy_plus_y_squared_l980_98092


namespace NUMINAMATH_GPT_find_x_l980_98084

def operation (a b : ℝ) : ℝ := a * b^(1/2)

theorem find_x (x : ℝ) : operation x 9 = 12 → x = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l980_98084


namespace NUMINAMATH_GPT_crates_on_third_trip_l980_98012

variable (x : ℕ) -- Denote the number of crates carried on the third trip

-- Conditions
def crate_weight := 1250
def max_weight := 6250
def trip3_weight (x : ℕ) := x * crate_weight

-- The problem statement: Prove that x (the number of crates on the third trip) == 5
theorem crates_on_third_trip : trip3_weight x <= max_weight → x = 5 :=
by
  sorry -- No proof required, just statement

end NUMINAMATH_GPT_crates_on_third_trip_l980_98012


namespace NUMINAMATH_GPT_inequality_solution_ge_11_l980_98080

theorem inequality_solution_ge_11
  (m n : ℝ)
  (h1 : m > 0)
  (h2 : n > 1)
  (h3 : (1/m) + (2/(n-1)) = 1) :
  m + 2 * n ≥ 11 :=
sorry

end NUMINAMATH_GPT_inequality_solution_ge_11_l980_98080


namespace NUMINAMATH_GPT_area_enclosed_by_absolute_value_linear_eq_l980_98086

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_absolute_value_linear_eq_l980_98086


namespace NUMINAMATH_GPT_snakes_in_each_cage_l980_98067

theorem snakes_in_each_cage (total_snakes : ℕ) (total_cages : ℕ) (h_snakes: total_snakes = 4) (h_cages: total_cages = 2) 
  (h_even_distribution : (total_snakes % total_cages) = 0) : (total_snakes / total_cages) = 2 := 
by sorry

end NUMINAMATH_GPT_snakes_in_each_cage_l980_98067


namespace NUMINAMATH_GPT_quadrilateral_AB_length_l980_98088

/-- Let ABCD be a quadrilateral with BC = CD = DA = 1, ∠DAB = 135°, and ∠ABC = 75°. 
    Prove that AB = (√6 - √2) / 2.
-/
theorem quadrilateral_AB_length (BC CD DA : ℝ) (angle_DAB angle_ABC : ℝ) (h1 : BC = 1)
    (h2 : CD = 1) (h3 : DA = 1) (h4 : angle_DAB = 135) (h5 : angle_ABC = 75) :
    AB = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
    sorry

end NUMINAMATH_GPT_quadrilateral_AB_length_l980_98088


namespace NUMINAMATH_GPT_unique_3_digit_number_with_conditions_l980_98058

def valid_3_digit_number (n : ℕ) : Prop :=
  let d2 := n / 100
  let d1 := (n / 10) % 10
  let d0 := n % 10
  (d2 > 0) ∧ (d2 < 10) ∧ (d1 < 10) ∧ (d0 < 10) ∧ (d2 + d1 + d0 = 28) ∧ (d0 < 7) ∧ (d0 % 2 = 0)

theorem unique_3_digit_number_with_conditions :
  (∃! n : ℕ, valid_3_digit_number n) :=
sorry

end NUMINAMATH_GPT_unique_3_digit_number_with_conditions_l980_98058


namespace NUMINAMATH_GPT_union_M_N_l980_98006

def M : Set ℝ := { x | x^2 + 2 * x = 0 }
def N : Set ℝ := { x | x^2 - 2 * x = 0 }

theorem union_M_N : M ∪ N = {0, -2, 2} := by
  sorry

end NUMINAMATH_GPT_union_M_N_l980_98006


namespace NUMINAMATH_GPT_batsman_highest_score_l980_98046

theorem batsman_highest_score (H L : ℕ) 
  (h₁ : (40 * 50 = 2000)) 
  (h₂ : (H = L + 172))
  (h₃ : (38 * 48 = 1824)) :
  (2000 = 1824 + H + L) → H = 174 :=
by 
  sorry

end NUMINAMATH_GPT_batsman_highest_score_l980_98046


namespace NUMINAMATH_GPT_number_of_boys_l980_98014

theorem number_of_boys (b g : ℕ) (h1: (3/5 : ℚ) * b = (5/6 : ℚ) * g) (h2: b + g = 30)
  (h3: g = (b * 18) / 25): b = 17 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l980_98014


namespace NUMINAMATH_GPT_rachel_earnings_without_tips_l980_98054

theorem rachel_earnings_without_tips
  (num_people : ℕ) (tip_per_person : ℝ) (total_earnings : ℝ)
  (h1 : num_people = 20)
  (h2 : tip_per_person = 1.25)
  (h3 : total_earnings = 37) :
  total_earnings - (num_people * tip_per_person) = 12 :=
by
  sorry

end NUMINAMATH_GPT_rachel_earnings_without_tips_l980_98054


namespace NUMINAMATH_GPT_cook_stole_the_cookbook_l980_98053

-- Define the suspects
inductive Suspect
| CheshireCat
| Duchess
| Cook
deriving DecidableEq, Repr

-- Define the predicate for lying
def lied (s : Suspect) : Prop := sorry

-- Define the conditions
def conditions (thief : Suspect) : Prop :=
  lied thief ∧
  ((∀ s : Suspect, s ≠ thief → lied s) ∨ (∀ s : Suspect, s ≠ thief → ¬lied s))

-- Define the goal statement
theorem cook_stole_the_cookbook : conditions Suspect.Cook :=
sorry

end NUMINAMATH_GPT_cook_stole_the_cookbook_l980_98053


namespace NUMINAMATH_GPT_simplify_and_evaluate_l980_98047

theorem simplify_and_evaluate (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6 * m + 9) / (m - 2)) = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l980_98047


namespace NUMINAMATH_GPT_fraction_of_female_attendees_on_time_l980_98013

theorem fraction_of_female_attendees_on_time (A : ℝ)
  (h1 : 3 / 5 * A = M)
  (h2 : 7 / 8 * M = M_on_time)
  (h3 : 0.115 * A = n_A_not_on_time) :
  0.9 * F = (A - M_on_time - n_A_not_on_time)/((2 / 5) * A - n_A_not_on_time) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_female_attendees_on_time_l980_98013


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l980_98048

theorem axis_of_symmetry_parabola : ∀ (x y : ℝ), y = 2 * x^2 → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l980_98048


namespace NUMINAMATH_GPT_min_dSigma_correct_l980_98016

noncomputable def min_dSigma {a r : ℝ} (h : a > r) : ℝ :=
  (a - r) / 2

theorem min_dSigma_correct (a r : ℝ) (h : a > r) :
  min_dSigma h = (a - r) / 2 :=
by 
  unfold min_dSigma
  sorry

end NUMINAMATH_GPT_min_dSigma_correct_l980_98016


namespace NUMINAMATH_GPT_find_m_from_split_l980_98065

theorem find_m_from_split (m : ℕ) (h1 : m > 1) (h2 : m^2 - m + 1 = 211) : True :=
by
  -- This theorem states that under the conditions that m is a positive integer greater than 1
  -- and m^2 - m + 1 = 211, there exists an integer value for m that satisfies these conditions.
  trivial

end NUMINAMATH_GPT_find_m_from_split_l980_98065


namespace NUMINAMATH_GPT_radius_increase_area_triple_l980_98034

theorem radius_increase_area_triple (r m : ℝ) (h : π * (r + m)^2 = 3 * π * r^2) : 
  r = (m * (Real.sqrt 3 - 1)) / 2 := 
sorry

end NUMINAMATH_GPT_radius_increase_area_triple_l980_98034


namespace NUMINAMATH_GPT_estimate_fish_population_l980_98020

theorem estimate_fish_population :
  ∀ (x : ℕ), (1200 / x = 100 / 1000) → x = 12000 := by
  sorry

end NUMINAMATH_GPT_estimate_fish_population_l980_98020


namespace NUMINAMATH_GPT_sixth_employee_salary_l980_98037

def salaries : List Real := [1000, 2500, 3100, 3650, 1500]

def mean_salary_of_six : Real := 2291.67

theorem sixth_employee_salary : 
  let total_five := salaries.sum 
  let total_six := mean_salary_of_six * 6
  (total_six - total_five) = 2000.02 :=
by
  sorry

end NUMINAMATH_GPT_sixth_employee_salary_l980_98037


namespace NUMINAMATH_GPT_required_average_for_tickets_l980_98093

theorem required_average_for_tickets 
  (june_score : ℝ) (patty_score : ℝ) (josh_score : ℝ) (henry_score : ℝ)
  (num_children : ℝ) (total_score : ℝ) (average_score : ℝ) (S : ℝ)
  (h1 : june_score = 97) (h2 : patty_score = 85) (h3 : josh_score = 100) 
  (h4 : henry_score = 94) (h5 : num_children = 4) 
  (h6 : total_score = june_score + patty_score + josh_score + henry_score)
  (h7 : average_score = total_score / num_children) 
  (h8 : average_score = 94)
  : S ≤ 94 :=
sorry

end NUMINAMATH_GPT_required_average_for_tickets_l980_98093


namespace NUMINAMATH_GPT_watch_cost_price_l980_98036

theorem watch_cost_price (CP : ℝ) (h1 : (0.90 * CP) + 280 = 1.04 * CP) : CP = 2000 := 
by 
  sorry

end NUMINAMATH_GPT_watch_cost_price_l980_98036


namespace NUMINAMATH_GPT_find_last_number_l980_98015

theorem find_last_number
  (A B C D : ℝ)
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11) :
  D = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_last_number_l980_98015


namespace NUMINAMATH_GPT_zero_in_set_zero_l980_98007

-- Define that 0 is an element
def zero_element : Prop := true

-- Define that {0} is a set containing only the element 0
def set_zero : Set ℕ := {0}

-- The main theorem that proves 0 ∈ {0}
theorem zero_in_set_zero (h : zero_element) : 0 ∈ set_zero := 
by sorry

end NUMINAMATH_GPT_zero_in_set_zero_l980_98007


namespace NUMINAMATH_GPT_areas_of_shared_parts_l980_98005

-- Define the areas of the non-overlapping parts
def area_non_overlap_1 : ℝ := 68
def area_non_overlap_2 : ℝ := 110
def area_non_overlap_3 : ℝ := 87

-- Define the total area of each circle
def total_area : ℝ := area_non_overlap_2 + area_non_overlap_3 - area_non_overlap_1

-- Define the areas of the shared parts A and B
def area_shared_A : ℝ := total_area - area_non_overlap_2
def area_shared_B : ℝ := total_area - area_non_overlap_3

-- Prove the areas of the shared parts
theorem areas_of_shared_parts :
  area_shared_A = 19 ∧ area_shared_B = 42 :=
by
  sorry

end NUMINAMATH_GPT_areas_of_shared_parts_l980_98005


namespace NUMINAMATH_GPT_intersection_A_B_l980_98008

-- Conditions
def A : Set (ℕ × ℕ) := { (1, 2), (2, 1) }
def B : Set (ℕ × ℕ) := { p | p.fst - p.snd = 1 }

-- Problem statement
theorem intersection_A_B : A ∩ B = { (2, 1) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l980_98008


namespace NUMINAMATH_GPT_age_of_other_replaced_man_l980_98063

variable (A B C : ℕ)
variable (B_new1 B_new2 : ℕ)
variable (avg_old avg_new : ℕ)

theorem age_of_other_replaced_man (hB : B = 23) 
    (h_avg_new : (B_new1 + B_new2) / 2 = 25)
    (h_avg_inc : (A + B_new1 + B_new2) / 3 > (A + B + C) / 3) : 
    C = 26 := 
  sorry

end NUMINAMATH_GPT_age_of_other_replaced_man_l980_98063


namespace NUMINAMATH_GPT_miley_total_cost_l980_98022

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end NUMINAMATH_GPT_miley_total_cost_l980_98022


namespace NUMINAMATH_GPT_shed_width_l980_98025

theorem shed_width (backyard_length backyard_width shed_length area_needed : ℝ)
  (backyard_area : backyard_length * backyard_width = 260)
  (sod_area : area_needed = 245)
  (shed_dim : shed_length = 3) :
  (backyard_length * backyard_width - area_needed) / shed_length = 5 :=
by
  -- We need to prove the width of the shed given the conditions
  sorry

end NUMINAMATH_GPT_shed_width_l980_98025


namespace NUMINAMATH_GPT_rectangular_frame_wire_and_paper_area_l980_98049

theorem rectangular_frame_wire_and_paper_area :
  let l1 := 3
  let l2 := 4
  let l3 := 5
  let wire_length := (l1 + l2 + l3) * 4
  let paper_area := ((l1 * l2) + (l1 * l3) + (l2 * l3)) * 2
  wire_length = 48 ∧ paper_area = 94 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_frame_wire_and_paper_area_l980_98049


namespace NUMINAMATH_GPT_lateral_surface_area_of_prism_l980_98085

theorem lateral_surface_area_of_prism 
  (a : ℝ) (α β V : ℝ) :
  let sin (x : ℝ) := Real.sin x 
  ∃ S : ℝ,
    S = (2 * V * sin ((α + β) / 2)) / (a * sin (α / 2) * sin (β / 2)) := 
sorry

end NUMINAMATH_GPT_lateral_surface_area_of_prism_l980_98085


namespace NUMINAMATH_GPT_evaluate_expression_l980_98051

theorem evaluate_expression : (- (1 / 4))⁻¹ - (Real.pi - 3)^0 - |(-4 : ℝ)| + (-1)^(2021 : ℕ) = -10 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l980_98051


namespace NUMINAMATH_GPT_percentage_calculation_l980_98061

theorem percentage_calculation :
  ∀ (P : ℝ),
  (0.3 * 0.5 * 4400 = 99) →
  (P * 4400 = 99) →
  P = 0.0225 :=
by
  intros P condition1 condition2
  -- From the given conditions, it follows directly
  sorry

end NUMINAMATH_GPT_percentage_calculation_l980_98061


namespace NUMINAMATH_GPT_units_digit_of_7_power_exp_is_1_l980_98081

-- Define the periodicity of units digits of powers of 7
def units_digit_seq : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit_power_7 (n : ℕ) : ℕ :=
  units_digit_seq.get! (n % 4)

-- Define the exponent
def exp : ℕ := 8^5

-- Define the modular operation result
def exp_modulo : ℕ := exp % 4

-- Define the main statement
theorem units_digit_of_7_power_exp_is_1 :
  units_digit_power_7 exp = 1 :=
by
  simp [units_digit_power_7, units_digit_seq, exp, exp_modulo]
  sorry

end NUMINAMATH_GPT_units_digit_of_7_power_exp_is_1_l980_98081


namespace NUMINAMATH_GPT_roots_twice_other_p_values_l980_98072

theorem roots_twice_other_p_values (p : ℝ) :
  (∃ (a : ℝ), (a^2 = 9) ∧ (x^2 + p*x + 18 = 0) ∧
  ((x - a)*(x - 2*a) = (0:ℝ))) ↔ (p = 9 ∨ p = -9) :=
sorry

end NUMINAMATH_GPT_roots_twice_other_p_values_l980_98072


namespace NUMINAMATH_GPT_sum_of_numbers_l980_98000

theorem sum_of_numbers (a b c : ℝ) (h_ratio : a / 1 = b / 2 ∧ b / 2 = c / 3) (h_sum_squares : a^2 + b^2 + c^2 = 2744) : 
  a + b + c = 84 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_l980_98000


namespace NUMINAMATH_GPT_largest_n_divisibility_l980_98097

theorem largest_n_divisibility :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧
  (∀ m : ℕ, (m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_divisibility_l980_98097


namespace NUMINAMATH_GPT_unique_7_tuple_count_l980_98002

theorem unique_7_tuple_count :
  ∃! (x : ℕ → ℝ) (zero_le_x : (∀ i, 0 ≤ i → i ≤ 6 → true)),
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_unique_7_tuple_count_l980_98002


namespace NUMINAMATH_GPT_sequence_prime_bounded_l980_98050

theorem sequence_prime_bounded (c : ℕ) (h : c > 0) : 
  ∀ (p : ℕ → ℕ), (∀ k, Nat.Prime (p k)) → (p 0) = some_prime →
  (∀ k, ∃ q, Nat.Prime q ∧ q ∣ (p k + c) ∧ (∀ i < k, q ≠ p i)) → 
  (∃ N, ∀ m ≥ N, ∀ n ≥ N, p m = p n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_prime_bounded_l980_98050


namespace NUMINAMATH_GPT_intersection_of_sets_is_closed_interval_l980_98032

noncomputable def A := {x : ℝ | x ≤ 0 ∨ x ≥ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_sets_is_closed_interval :
  A ∩ B = {x : ℝ | x ≤ 0} :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_is_closed_interval_l980_98032


namespace NUMINAMATH_GPT_sqrt_div_l980_98064

theorem sqrt_div (x: ℕ) (h1: Nat.sqrt 144 * Nat.sqrt 144 = 144) (h2: 144 = 12 * 12) (h3: 2 * x = 12) : x = 6 :=
sorry

end NUMINAMATH_GPT_sqrt_div_l980_98064


namespace NUMINAMATH_GPT_probability_of_three_5s_in_eight_rolls_l980_98023

-- Conditions
def total_outcomes : ℕ := 6 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3

-- The probability that the number 5 appears exactly three times in eight rolls of a fair die
theorem probability_of_three_5s_in_eight_rolls :
  (favorable_outcomes / total_outcomes : ℚ) = (56 / 1679616 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_three_5s_in_eight_rolls_l980_98023


namespace NUMINAMATH_GPT_calculate_f3_l980_98055

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^7 + a * x^5 + b * x - 5

theorem calculate_f3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := 
by
  sorry

end NUMINAMATH_GPT_calculate_f3_l980_98055
