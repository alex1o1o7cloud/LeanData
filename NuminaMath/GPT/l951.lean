import Mathlib

namespace compute_expression_l951_95150

theorem compute_expression (x y : ℝ) (hx : 1/x + 1/y = 4) (hy : x*y + x + y = 5) : 
  x^2 * y + x * y^2 + x^2 + y^2 = 18 := 
by 
  -- Proof goes here 
  sorry

end compute_expression_l951_95150


namespace solve_for_x_l951_95104

theorem solve_for_x (x : ℝ) (h : (4 + x) / (6 + x) = (1 + x) / (2 + x)) : x = 2 :=
sorry

end solve_for_x_l951_95104


namespace train_cross_first_platform_l951_95112

noncomputable def time_to_cross_first_platform (L_t L_p1 L_p2 t2 : ℕ) : ℕ :=
  (L_t + L_p1) / ((L_t + L_p2) / t2)

theorem train_cross_first_platform :
  time_to_cross_first_platform 100 200 300 20 = 15 :=
by
  sorry

end train_cross_first_platform_l951_95112


namespace sum_of_numbers_l951_95184

open Function

theorem sum_of_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : b = 8) 
  (h4 : (a + b + c) / 3 = a + 7) 
  (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 63 := 
by 
  sorry

end sum_of_numbers_l951_95184


namespace quadratic_axis_of_symmetry_is_one_l951_95170

noncomputable def quadratic_axis_of_symmetry (b c : ℝ) : ℝ :=
  (-b / (2 * 1))

theorem quadratic_axis_of_symmetry_is_one
  (b c : ℝ)
  (hA : (0:ℝ)^2 + b * 0 + c = 3)
  (hB : (2:ℝ)^2 + b * 2 + c = 3) :
  quadratic_axis_of_symmetry b c = 1 :=
by
  sorry

end quadratic_axis_of_symmetry_is_one_l951_95170


namespace octagon_diagonals_l951_95101

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l951_95101


namespace decimal_to_fraction_l951_95115

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l951_95115


namespace length_of_platform_l951_95142

def len_train : ℕ := 300 -- length of the train in meters
def time_platform : ℕ := 39 -- time to cross the platform in seconds
def time_pole : ℕ := 26 -- time to cross the signal pole in seconds

theorem length_of_platform (L : ℕ) (h1 : len_train / time_pole = (len_train + L) / time_platform) : L = 150 :=
  sorry

end length_of_platform_l951_95142


namespace smallest_integer_no_inverse_mod_77_66_l951_95176

theorem smallest_integer_no_inverse_mod_77_66 :
  ∃ a : ℕ, 0 < a ∧ a = 11 ∧ gcd a 77 > 1 ∧ gcd a 66 > 1 :=
by
  sorry

end smallest_integer_no_inverse_mod_77_66_l951_95176


namespace smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l951_95127

theorem smallest_integer_sum_of_squares_and_cubes :
  ∃ (n : ℕ) (a b c d : ℕ), n > 2 ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 ∧
  ∀ (m : ℕ) (x y u v : ℕ), (m > 2 ∧ m = x^2 + y^2 ∧ m = u^3 + v^3) → n ≤ m := 
sorry

theorem infinite_integers_sum_of_squares_and_cubes :
  ∀ (k : ℕ), ∃ (n : ℕ) (a b c d : ℕ), n = 1 + 2^(6*k) ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 :=
sorry

end smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l951_95127


namespace coeff_div_binom_eq_4_l951_95162

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def coeff_x5_expansion : ℚ :=
  binomial 8 2 * (-2) ^ 2

def binomial_coeff : ℚ :=
  binomial 8 2

theorem coeff_div_binom_eq_4 : 
  (coeff_x5_expansion / binomial_coeff) = 4 := by
  sorry

end coeff_div_binom_eq_4_l951_95162


namespace proof_subset_l951_95100

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem proof_subset : N ⊆ M := sorry

end proof_subset_l951_95100


namespace expression_value_l951_95187

theorem expression_value (b : ℝ) (hb : b = 1 / 3) :
    (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 :=
sorry

end expression_value_l951_95187


namespace percentage_difference_height_l951_95190

-- Define the heights of persons B, A, and C
variables (H_B H_A H_C : ℝ)

-- Condition: Person A's height is 30% less than person B's height
def person_A_height : Prop := H_A = 0.70 * H_B

-- Condition: Person C's height is 20% more than person A's height
def person_C_height : Prop := H_C = 1.20 * H_A

-- The proof problem: Prove that the percentage difference between H_B and H_C is 16%
theorem percentage_difference_height (h1 : person_A_height H_B H_A) (h2 : person_C_height H_A H_C) :
  ((H_B - H_C) / H_B) * 100 = 16 :=
by
  sorry

end percentage_difference_height_l951_95190


namespace arithmetic_sequence_general_term_l951_95116

theorem arithmetic_sequence_general_term (x : ℕ)
  (t1 t2 t3 : ℤ)
  (h1 : t1 = x - 1)
  (h2 : t2 = x + 1)
  (h3 : t3 = 2 * x + 3) :
  (∃ a : ℕ → ℤ, a 1 = t1 ∧ a 2 = t2 ∧ a 3 = t3 ∧ ∀ n, a n = 2 * n - 3) := 
sorry

end arithmetic_sequence_general_term_l951_95116


namespace find_cost_price_of_article_l951_95129

theorem find_cost_price_of_article 
  (C : ℝ) 
  (h1 : 1.05 * C - 2 = 1.045 * C) 
  (h2 : 0.005 * C = 2) 
: C = 400 := 
by 
  sorry

end find_cost_price_of_article_l951_95129


namespace exists_natural_numbers_with_digit_sum_condition_l951_95167

def digit_sum (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

theorem exists_natural_numbers_with_digit_sum_condition :
  ∃ (a b c : ℕ), digit_sum (a + b) < 5 ∧ digit_sum (a + c) < 5 ∧ digit_sum (b + c) < 5 ∧ digit_sum (a + b + c) > 50 :=
by
  sorry

end exists_natural_numbers_with_digit_sum_condition_l951_95167


namespace problem1_l951_95122

/-- Problem 1: Given the formula \( S = vt + \frac{1}{2}at^2 \) and the conditions
  when \( t=1, S=4 \) and \( t=2, S=10 \), prove that when \( t=3 \), \( S=18 \). -/
theorem problem1 (v a t S: ℝ) 
  (h₁ : t = 1 → S = 4 → S = v * t + 1 / 2 * a * t^2)
  (h₂ : t = 2 → S = 10 → S = v * t + 1 / 2 * a * t^2):
  t = 3 → S = v * t + 1 / 2 * a * t^2 → S = 18 := by
  sorry

end problem1_l951_95122


namespace evaluate_expression_l951_95138

theorem evaluate_expression (A B : ℝ) (hA : A = 2^7) (hB : B = 3^6) : (A ^ (1 / 3)) * (B ^ (1 / 2)) = 108 * 2 ^ (1 / 3) :=
by
  sorry

end evaluate_expression_l951_95138


namespace product_of_B_coords_l951_95156

structure Point where
  x : ℝ
  y : ℝ

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

theorem product_of_B_coords :
  ∀ (M A B : Point), 
  isMidpoint M A B →
  M = ⟨3, 7⟩ →
  A = ⟨5, 3⟩ →
  (B.x * B.y) = 11 :=
by intro M A B hM hM_def hA_def; sorry

end product_of_B_coords_l951_95156


namespace proof_problem_l951_95125

noncomputable def problem_equivalent_proof (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧
  (z + 6 = 2 * y - z) ∧
  (x + 8 * z = y + 2) →
  (x^2 + y^2 + z^2 = 21)

theorem proof_problem (x y z : ℝ) : problem_equivalent_proof x y z :=
by
  sorry

end proof_problem_l951_95125


namespace only_one_tuple_exists_l951_95117

theorem only_one_tuple_exists :
  ∃! (x : Fin 15 → ℝ),
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2
    + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2 + (x 8 - x 9)^2
    + (x 9 - x 10)^2 + (x 10 - x 11)^2 + (x 11 - x 12)^2 + (x 12 - x 13)^2
    + (x 13 - x 14)^2 + (x 14)^2 = 1 / 16 := by
  sorry

end only_one_tuple_exists_l951_95117


namespace roots_negative_reciprocal_condition_l951_95175

theorem roots_negative_reciprocal_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) → c = -a :=
by
  sorry

end roots_negative_reciprocal_condition_l951_95175


namespace compare_negatives_l951_95180

theorem compare_negatives : -1 > -2 := 
by 
  sorry

end compare_negatives_l951_95180


namespace flowers_count_l951_95105

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end flowers_count_l951_95105


namespace minimize_cost_l951_95189

noncomputable def shipping_cost (x : ℝ) : ℝ := 5 * x
noncomputable def storage_cost (x : ℝ) : ℝ := 20 / x
noncomputable def total_cost (x : ℝ) : ℝ := shipping_cost x + storage_cost x

theorem minimize_cost : ∃ x : ℝ, x = 2 ∧ total_cost x = 20 :=
by
  use 2
  unfold total_cost
  unfold shipping_cost
  unfold storage_cost
  sorry

end minimize_cost_l951_95189


namespace roots_equal_condition_l951_95111

theorem roots_equal_condition (a c : ℝ) (h : a ≠ 0) :
    (∀ x1 x2, (a * x1 * x1 + 4 * a * x1 + c = 0) ∧ (a * x2 * x2 + 4 * a * x2 + c = 0) → x1 = x2) ↔ c = 4 * a := 
by
  sorry

end roots_equal_condition_l951_95111


namespace exponent_evaluation_l951_95163

theorem exponent_evaluation {a b : ℕ} (h₁ : 2 ^ a ∣ 200) (h₂ : ¬ (2 ^ (a + 1) ∣ 200))
                           (h₃ : 5 ^ b ∣ 200) (h₄ : ¬ (5 ^ (b + 1) ∣ 200)) :
  (1 / 3) ^ (b - a) = 3 :=
by sorry

end exponent_evaluation_l951_95163


namespace range_of_m_l951_95124

noncomputable def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
noncomputable def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ¬p x → ¬q x m) → (m ≥ 9) :=
by
  sorry

end range_of_m_l951_95124


namespace part1_part2_l951_95171

def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + a + 1

-- Proof problem 1: Prove that if a = 2, then f(x) ≥ 0 is equivalent to x ≥ 3/2 or x ≤ 1.
theorem part1 (x : ℝ) : f 2 x ≥ 0 ↔ x ≥ (3 / 2 : ℝ) ∨ x ≤ 1 := sorry

-- Proof problem 2: Prove that for a∈[-2,2], if f(x) < 0 always holds, then x ∈ (1, 3/2).
theorem part2 (a x : ℝ) (ha : a ≥ -2 ∧ a ≤ 2) : (∀ x, f a x < 0) ↔ 1 < x ∧ x < (3 / 2 : ℝ) := sorry

end part1_part2_l951_95171


namespace incorrect_gcd_statement_l951_95183

theorem incorrect_gcd_statement :
  ¬(gcd 85 357 = 34) ∧ (gcd 16 12 = 4) ∧ (gcd 78 36 = 6) ∧ (gcd 105 315 = 105) :=
by
  sorry

end incorrect_gcd_statement_l951_95183


namespace derivative_at_pi_over_4_l951_95123

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem derivative_at_pi_over_4 : (deriv f (Real.pi / 4)) = -2 :=
by
  sorry

end derivative_at_pi_over_4_l951_95123


namespace two_cubic_meters_to_cubic_feet_l951_95119

theorem two_cubic_meters_to_cubic_feet :
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  2 * cubic_meter_to_cubic_feet = 70.6294 :=
by
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  have h : 2 * cubic_meter_to_cubic_feet = 70.6294 := sorry
  exact h

end two_cubic_meters_to_cubic_feet_l951_95119


namespace opposite_of_neg_two_is_two_l951_95152

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l951_95152


namespace ratio_of_students_l951_95199

-- Define the conditions
def total_students : Nat := 800
def students_spaghetti : Nat := 320
def students_fettuccine : Nat := 160

-- The proof problem
theorem ratio_of_students (h1 : students_spaghetti = 320) (h2 : students_fettuccine = 160) :
  students_spaghetti / students_fettuccine = 2 := by
  sorry

end ratio_of_students_l951_95199


namespace anna_coaching_days_l951_95191

/-- The total number of days from January 1 to September 4 in a non-leap year -/
def total_days_in_non_leap_year_up_to_sept4 : ℕ :=
  let days_in_january := 31
  let days_in_february := 28
  let days_in_march := 31
  let days_in_april := 30
  let days_in_may := 31
  let days_in_june := 30
  let days_in_july := 31
  let days_in_august := 31
  let days_up_to_sept4 := 4
  days_in_january + days_in_february + days_in_march + days_in_april +
  days_in_may + days_in_june + days_in_july + days_in_august + days_up_to_sept4

theorem anna_coaching_days : total_days_in_non_leap_year_up_to_sept4 = 247 :=
by
  -- Proof omitted
  sorry

end anna_coaching_days_l951_95191


namespace students_suggesting_bacon_l951_95133

theorem students_suggesting_bacon (S : ℕ) (M : ℕ) (h1: S = 310) (h2: M = 185) : S - M = 125 := 
by
  -- proof here
  sorry

end students_suggesting_bacon_l951_95133


namespace find_x_squared_plus_y_squared_l951_95182

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 - y^2 + x + y = 44) : x^2 + y^2 = 109 :=
sorry

end find_x_squared_plus_y_squared_l951_95182


namespace y_percentage_of_8950_l951_95166

noncomputable def x := 0.18 * 4750
noncomputable def y := 1.30 * x
theorem y_percentage_of_8950 : (y / 8950) * 100 = 12.42 := 
by 
  -- proof steps are omitted
  sorry

end y_percentage_of_8950_l951_95166


namespace preceding_integer_l951_95197

def bin_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

theorem preceding_integer : bin_to_nat [true, true, false, false, false] - 1 = bin_to_nat [true, false, true, true, true] := by
  sorry

end preceding_integer_l951_95197


namespace expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l951_95136

noncomputable def A (x y : ℝ) := x^2 - 3 * x * y - y^2
noncomputable def B (x y : ℝ) := x^2 - 3 * x * y - 3 * y^2
noncomputable def M (x y : ℝ) := 2 * A x y - B x y

theorem expression_for_M (x y : ℝ) : M x y = x^2 - 3 * x * y + y^2 := by
  sorry

theorem value_of_M_when_x_eq_negative_2_and_y_eq_1 :
  M (-2) 1 = 11 := by
  sorry

end expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l951_95136


namespace cubic_sum_l951_95128

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l951_95128


namespace trapezoid_segment_AB_length_l951_95131

/-
In the trapezoid shown, the ratio of the area of triangle ABC to the area of triangle ADC is 5:2.
If AB + CD = 240 cm, prove that the length of segment AB is 171.42857 cm.
-/

theorem trapezoid_segment_AB_length
  (AB CD : ℝ)
  (ratio_areas : ℝ := 5 / 2)
  (area_ratio_condition : AB / CD = ratio_areas)
  (length_sum_condition : AB + CD = 240) :
  AB = 171.42857 :=
sorry

end trapezoid_segment_AB_length_l951_95131


namespace uneaten_chips_correct_l951_95143

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l951_95143


namespace binomial_expansion_example_l951_95165

theorem binomial_expansion_example : 7^3 + 3 * (7^2) * 2 + 3 * 7 * (2^2) + 2^3 = 729 := by
  sorry

end binomial_expansion_example_l951_95165


namespace spider_final_position_l951_95148

def circle_points : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def next_position (current : ℕ) : ℕ :=
  if current % 2 = 0 
  then (current + 3 - 1) % 7 + 1 -- Clockwise modulo operation for even
  else (current + 1 - 1) % 7 + 1 -- Clockwise modulo operation for odd

def spider_position_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  (Nat.iterate next_position jumps start)

theorem spider_final_position : spider_position_after_jumps 6 2055 = 2 := 
  by
  sorry

end spider_final_position_l951_95148


namespace negation_p_l951_95178

open Nat

def p : Prop := ∀ n : ℕ, n^2 ≤ 2^n

theorem negation_p : ¬p ↔ ∃ n : ℕ, n^2 > 2^n :=
by
  sorry

end negation_p_l951_95178


namespace problem_l951_95132

theorem problem (m n : ℕ) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (h1 : m + 8 < n) 
  (h2 : (m + (m + 3) + (m + 8) + n + (n + 3) + (2 * n - 1)) / 6 = n + 1) 
  (h3 : (m + 8 + n) / 2 = n + 1) : m + n = 16 :=
  sorry

end problem_l951_95132


namespace sum_of_ten_numbers_in_circle_l951_95141

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end sum_of_ten_numbers_in_circle_l951_95141


namespace total_stars_l951_95146

theorem total_stars (students stars_per_student : ℕ) (h_students : students = 124) (h_stars_per_student : stars_per_student = 3) : students * stars_per_student = 372 := by
  sorry

end total_stars_l951_95146


namespace locus_centers_tangent_circles_l951_95177

theorem locus_centers_tangent_circles (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (3 - r)^2) →
  a^2 - 12 * a + 4 * b^2 = 0 :=
by
  sorry

end locus_centers_tangent_circles_l951_95177


namespace expand_polynomial_l951_95179

noncomputable def polynomial_expansion : Prop :=
  ∀ (x : ℤ), (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18

theorem expand_polynomial : polynomial_expansion :=
by
  sorry

end expand_polynomial_l951_95179


namespace barbara_candies_left_l951_95118

def initial_candies: ℝ := 18.5
def candies_used_to_make_dessert: ℝ := 4.2
def candies_received_from_friend: ℝ := 6.8
def candies_eaten: ℝ := 2.7

theorem barbara_candies_left : 
  initial_candies - candies_used_to_make_dessert + candies_received_from_friend - candies_eaten = 18.4 := 
by
  sorry

end barbara_candies_left_l951_95118


namespace average_rate_of_change_interval_l951_95151

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

theorem average_rate_of_change_interval (f : ℝ → ℝ) (x₀ x₁ : ℝ) :
  (f x₁ - f x₀) / (x₁ - x₀) = average_rate_of_change f x₀ x₁ := by
  sorry

end average_rate_of_change_interval_l951_95151


namespace children_left_birthday_l951_95135

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

end children_left_birthday_l951_95135


namespace train_passing_time_l951_95108

theorem train_passing_time 
  (length_train : ℕ) 
  (speed_train_kmph : ℕ) 
  (time_to_pass : ℕ)
  (h1 : length_train = 60)
  (h2 : speed_train_kmph = 54)
  (h3 : time_to_pass = 4) :
  time_to_pass = length_train * 18 / (speed_train_kmph * 5) := by
  sorry

end train_passing_time_l951_95108


namespace inverse_function_l951_95192

noncomputable def f (x : ℝ) := 3 - 7 * x + x^2

noncomputable def g (x : ℝ) := (7 + Real.sqrt (37 + 4 * x)) / 2

theorem inverse_function :
  ∀ x : ℝ, f (g x) = x :=
by
  intros x
  sorry

end inverse_function_l951_95192


namespace find_shop_width_l951_95172

def shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_square_foot : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_area := annual_rent / annual_rent_per_square_foot
  total_area / length

theorem find_shop_width :
  shop_width 3600 20 144 = 15 :=
by 
  -- Here would go the proof, but we add sorry to skip it
  sorry

end find_shop_width_l951_95172


namespace sum_of_three_distinct_l951_95168

def S : Set ℤ := {2, 5, 8, 11, 14, 17, 20}

theorem sum_of_three_distinct (S : Set ℤ) (h : S = {2, 5, 8, 11, 14, 17, 20}) :
  (∃ n : ℕ, n = 13 ∧ ∀ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ∃ k : ℕ, a + b + c = 3 * k) := 
by  -- The proof goes here.
  sorry

end sum_of_three_distinct_l951_95168


namespace multiplication_result_l951_95169

theorem multiplication_result : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end multiplication_result_l951_95169


namespace stationery_store_profit_l951_95109

variable (a : ℝ)

def store_cost : ℝ := 100 * a
def markup_price : ℝ := a * 1.2
def discount_price : ℝ := markup_price a * 0.8

def revenue_first_half : ℝ := 50 * markup_price a
def revenue_second_half : ℝ := 50 * discount_price a
def total_revenue : ℝ := revenue_first_half a + revenue_second_half a

def profit : ℝ := total_revenue a - store_cost a

theorem stationery_store_profit : profit a = 8 * a := 
by sorry

end stationery_store_profit_l951_95109


namespace maximum_area_rectangle_l951_95188

-- Define the conditions
def length (x : ℝ) := x
def width (x : ℝ) := 2 * x
def perimeter (x : ℝ) := 2 * (length x + width x)

-- The proof statement
theorem maximum_area_rectangle (h : perimeter x = 40) : 2 * (length x) * (width x) = 800 / 9 :=
by
  sorry

end maximum_area_rectangle_l951_95188


namespace P_sufficient_but_not_necessary_for_Q_l951_95161

def P (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def Q (x : ℝ) : Prop := x^2 - 2 * x + 1 > 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ ¬ (∀ x : ℝ, Q x → P x) :=
by 
  sorry

end P_sufficient_but_not_necessary_for_Q_l951_95161


namespace simplify_expression_l951_95130

theorem simplify_expression (x y : ℝ) (hxy : x ≠ y) : 
  ((x - y) ^ 3 / (x - y) ^ 2) * (y - x) = -(x - y) ^ 2 := 
by
  sorry

end simplify_expression_l951_95130


namespace polynomial_divisibility_l951_95159

theorem polynomial_divisibility (m : ℕ) (hm : 0 < m) :
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1) ^ (2 * m) - x ^ (2 * m) - 2 * x - 1 :=
by
  intro x
  sorry

end polynomial_divisibility_l951_95159


namespace incorrect_statement_c_l951_95120

open Real

theorem incorrect_statement_c (p q: ℝ) : ¬(∀ x: ℝ, (x * abs x + p * x + q = 0 ↔ p^2 - 4 * q ≥ 0)) :=
sorry

end incorrect_statement_c_l951_95120


namespace min_S6_minus_S4_l951_95114

variable {a₁ a₂ q : ℝ} (h1 : q > 1) (h2 : (q^2 - 1) * (a₁ + a₂) = 3)

theorem min_S6_minus_S4 : 
  ∃ (a₁ a₂ q : ℝ), q > 1 ∧ (q^2 - 1) * (a₁ + a₂) = 3 ∧ (q^4 * (a₁ + a₂) - (a₁ + a₂ + a₂ * q + a₂ * q^2) = 12) := sorry

end min_S6_minus_S4_l951_95114


namespace compare_abc_l951_95126

variable (a b c : ℝ)

noncomputable def define_a : ℝ := (2/3)^(1/3)
noncomputable def define_b : ℝ := (2/3)^(1/2)
noncomputable def define_c : ℝ := (3/5)^(1/2)

theorem compare_abc (h₁ : a = define_a) (h₂ : b = define_b) (h₃ : c = define_c) :
  a > b ∧ b > c := by
  sorry

end compare_abc_l951_95126


namespace smallest_points_to_exceed_mean_l951_95185

theorem smallest_points_to_exceed_mean (X y : ℕ) (h_scores : 24 + 17 + 25 = 66) 
  (h_mean_9_gt_mean_6 : X / 6 < (X + 66) / 9) (h_mean_10_gt_22 : (X + 66 + y) / 10 > 22) 
  : y ≥ 24 := by
  sorry

end smallest_points_to_exceed_mean_l951_95185


namespace percentage_repeated_digits_five_digit_numbers_l951_95103

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end percentage_repeated_digits_five_digit_numbers_l951_95103


namespace find_ab_l951_95157

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) :
  a * b = 10 :=
by
  sorry

end find_ab_l951_95157


namespace rent_of_first_apartment_l951_95198

theorem rent_of_first_apartment (R : ℝ) :
  let cost1 := R + 260 + (31 * 20 * 0.58)
  let cost2 := 900 + 200 + (21 * 20 * 0.58)
  (cost1 - cost2 = 76) → R = 800 :=
by
  intro h
  sorry

end rent_of_first_apartment_l951_95198


namespace jacob_age_in_X_years_l951_95158

-- Definitions of the conditions
variable (J M X : ℕ)

theorem jacob_age_in_X_years
  (h1 : J = M - 14)
  (h2 : M + 9 = 2 * (J + 9))
  (h3 : J = 5) :
  J + X = 5 + X :=
by
  sorry

end jacob_age_in_X_years_l951_95158


namespace unshaded_area_eq_20_l951_95153

-- Define the dimensions of the first rectangle
def rect1_width := 4
def rect1_length := 12

-- Define the dimensions of the second rectangle
def rect2_width := 5
def rect2_length := 10

-- Define the dimensions of the overlapping region
def overlap_width := 4
def overlap_length := 5

-- Calculate area functions
def area (width length : ℕ) := width * length

-- Calculate areas of the individual rectangles and the overlapping region
def area_rect1 := area rect1_width rect1_length
def area_rect2 := area rect2_width rect2_length
def overlap_area := area overlap_width overlap_length

-- Calculate the total shaded area
def total_shaded_area := area_rect1 + area_rect2 - overlap_area

-- The total area of the combined figure (assumed to be the union of both rectangles) minus shaded area gives the unshaded area
def total_area := rect1_width * rect1_length + rect2_width * rect2_length
def unshaded_area := total_area - total_shaded_area

theorem unshaded_area_eq_20 : unshaded_area = 20 := by
  sorry

end unshaded_area_eq_20_l951_95153


namespace solve_a_b_powers_l951_95186

theorem solve_a_b_powers :
  ∃ a b : ℂ, (a + b = 1) ∧ 
             (a^2 + b^2 = 3) ∧ 
             (a^3 + b^3 = 4) ∧ 
             (a^4 + b^4 = 7) ∧ 
             (a^5 + b^5 = 11) ∧ 
             (a^10 + b^10 = 93) :=
sorry

end solve_a_b_powers_l951_95186


namespace option_C_is_quadratic_l951_95110

-- Define what it means for an equation to be quadratic
def is_quadratic (p : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), p x ↔ a*x^2 + b*x + c = 0

-- Define the equation in option C
def option_C (x : ℝ) : Prop := (x - 1) * (x - 2) = 0

-- The theorem we need to prove
theorem option_C_is_quadratic : is_quadratic option_C :=
  sorry

end option_C_is_quadratic_l951_95110


namespace kinetic_energy_reduction_collisions_l951_95144

theorem kinetic_energy_reduction_collisions (E_0 : ℝ) (n : ℕ) :
  (1 / 2)^n * E_0 = E_0 / 64 → n = 6 :=
by
  sorry

end kinetic_energy_reduction_collisions_l951_95144


namespace linear_regression_change_l951_95196

theorem linear_regression_change (x : ℝ) :
  let y1 := 2 - 1.5 * x
  let y2 := 2 - 1.5 * (x + 1)
  y2 - y1 = -1.5 := by
  -- y1 = 2 - 1.5 * x
  -- y2 = 2 - 1.5 * x - 1.5
  -- Δ y = y2 - y1
  sorry

end linear_regression_change_l951_95196


namespace largest_value_l951_95193

theorem largest_value :
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- conditions
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  -- sorry to skip the proof
  sorry

end largest_value_l951_95193


namespace annual_growth_rate_proof_l951_95147

-- Lean 4 statement for the given problem
theorem annual_growth_rate_proof (profit_2021 : ℝ) (profit_2023 : ℝ) (r : ℝ)
  (h1 : profit_2021 = 3000)
  (h2 : profit_2023 = 4320)
  (h3 : profit_2023 = profit_2021 * (1 + r) ^ 2) :
  r = 0.2 :=
by sorry

end annual_growth_rate_proof_l951_95147


namespace correct_calculation_B_l951_95149

theorem correct_calculation_B :
  (∀ (a : ℕ), 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) ∧
  (∀ (x : ℕ), 3 * x^2 * 4 * x^2 ≠ 12 * x^2) ∧
  (∀ (y : ℕ), 5 * y^3 * 3 * y^5 ≠ 8 * y^8) →
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) := 
by
  sorry

end correct_calculation_B_l951_95149


namespace min_value_x_plus_y_l951_95174

theorem min_value_x_plus_y (x y : ℤ) (det : 3 < x * y ∧ x * y < 5) : x + y = -5 :=
sorry

end min_value_x_plus_y_l951_95174


namespace calculation_result_l951_95139

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by
  sorry

end calculation_result_l951_95139


namespace rectangular_cube_length_l951_95106

theorem rectangular_cube_length (L : ℝ) (h1 : 2 * (L * 2) + 2 * (L * 0.5) + 2 * (2 * 0.5) = 24) : L = 4.6 := 
by {
  sorry
}

end rectangular_cube_length_l951_95106


namespace calc_expr_l951_95194

theorem calc_expr : (3^5 * 6^3 + 3^3) = 52515 := by
  sorry

end calc_expr_l951_95194


namespace infinite_geometric_series_sum_l951_95154

theorem infinite_geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1 / 3
  ∑' (n : ℕ), a * r ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l951_95154


namespace problem1_problem2_l951_95102

theorem problem1 : (1 * (-9)) - (-7) + (-6) - 5 = -13 := 
by 
  -- problem1 proof
  sorry

theorem problem2 : ((-5 / 12) + (2 / 3) - (3 / 4)) * (-12) = 6 := 
by 
  -- problem2 proof
  sorry

end problem1_problem2_l951_95102


namespace mod_sum_example_l951_95181

theorem mod_sum_example :
  (9^5 + 8^4 + 7^6) % 5 = 4 :=
by sorry

end mod_sum_example_l951_95181


namespace students_in_each_grade_l951_95113

theorem students_in_each_grade (total_students : ℕ) (total_grades : ℕ) (students_per_grade : ℕ) :
  total_students = 22800 → total_grades = 304 → students_per_grade = total_students / total_grades → students_per_grade = 75 :=
by
  intros h1 h2 h3
  sorry

end students_in_each_grade_l951_95113


namespace total_reading_materials_l951_95160

theorem total_reading_materials (magazines newspapers : ℕ) (h1 : magazines = 425) (h2 : newspapers = 275) : 
  magazines + newspapers = 700 :=
by 
  sorry

end total_reading_materials_l951_95160


namespace jessica_current_age_l951_95173

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_l951_95173


namespace profit_loss_balance_l951_95134

-- Defining variables
variables (C L : Real)

-- Profit and loss equations according to problem conditions
theorem profit_loss_balance (h1 : 832 - C = C - L) (h2 : 992 = 0.55 * C) : 
  (C + 992 = 2795.64) :=
by
  -- Statement of the theorem
  sorry

end profit_loss_balance_l951_95134


namespace range_of_a_l951_95121

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) : 0 ≤ a := sorry

end range_of_a_l951_95121


namespace angle_bisector_inequality_l951_95107

noncomputable def triangle_ABC (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C] (AB BC CA AK CM AM MK KC : ℝ) 
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : Prop :=
  AM > MK ∧ MK > KC

theorem angle_bisector_inequality (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (AB BC CA AK CM AM MK KC : ℝ)
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : AM > MK ∧ MK > KC :=
by
  sorry

end angle_bisector_inequality_l951_95107


namespace right_triangle_ineq_l951_95164

variable (a b c : ℝ)
variable (h : c^2 = a^2 + b^2)

theorem right_triangle_ineq (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 :=
by
  sorry

end right_triangle_ineq_l951_95164


namespace largest_sum_of_digits_in_display_l951_95137

-- Define the conditions
def is_valid_hour (h : Nat) : Prop := 0 <= h ∧ h < 24
def is_valid_minute (m : Nat) : Prop := 0 <= m ∧ m < 60

-- Define helper functions to convert numbers to their digit sums
def digit_sum (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Define the largest possible sum of the digits condition
def largest_possible_digit_sum : Prop :=
  ∀ (h m : Nat), is_valid_hour h → is_valid_minute m → 
    digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) ≤ 24 ∧
    ∃ (h m : Nat), is_valid_hour h ∧ is_valid_minute m ∧ digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) = 24

-- The statement to prove
theorem largest_sum_of_digits_in_display : largest_possible_digit_sum :=
by
  sorry

end largest_sum_of_digits_in_display_l951_95137


namespace difference_is_four_l951_95195

def chickens_in_coop := 14
def chickens_in_run := 2 * chickens_in_coop
def chickens_free_ranging := 52
def difference := 2 * chickens_in_run - chickens_free_ranging

theorem difference_is_four : difference = 4 := by
  sorry

end difference_is_four_l951_95195


namespace sum_of_values_satisfying_l951_95155

theorem sum_of_values_satisfying (x : ℝ) (h : Real.sqrt ((x - 2) ^ 2) = 8) :
  ∃ x1 x2 : ℝ, (Real.sqrt ((x1 - 2) ^ 2) = 8) ∧ (Real.sqrt ((x2 - 2) ^ 2) = 8) ∧ x1 + x2 = 4 := 
by
  sorry

end sum_of_values_satisfying_l951_95155


namespace Onum_Lake_more_trout_l951_95145

theorem Onum_Lake_more_trout (O B R : ℕ) (hB : B = 75) (hR : R = O / 2) (hAvg : (O + B + R) / 3 = 75) : O - B = 25 :=
by
  sorry

end Onum_Lake_more_trout_l951_95145


namespace tumblonian_words_count_l951_95140

def numTumblonianWords : ℕ :=
  let alphabet_size := 6
  let max_word_length := 4
  let num_words n := alphabet_size ^ n
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4)

theorem tumblonian_words_count : numTumblonianWords = 1554 := by
  sorry

end tumblonian_words_count_l951_95140
