import Mathlib

namespace NUMINAMATH_GPT_sequence_difference_l2022_202216

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sequence_difference (hS : ∀ n, S n = n^2 - 5 * n)
                            (hna : ∀ n, a n = S n - S (n - 1))
                            (hpq : p - q = 4) :
                            a p - a q = 8 := by
    sorry

end NUMINAMATH_GPT_sequence_difference_l2022_202216


namespace NUMINAMATH_GPT_product_of_remainders_one_is_one_l2022_202275

theorem product_of_remainders_one_is_one (a b : ℕ) (h1 : a % 3 = 1) (h2 : b % 3 = 1) : (a * b) % 3 = 1 :=
sorry

end NUMINAMATH_GPT_product_of_remainders_one_is_one_l2022_202275


namespace NUMINAMATH_GPT_divisor_is_18_l2022_202291

def dividend : ℕ := 165
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem divisor_is_18 (divisor : ℕ) : dividend = quotient * divisor + remainder → divisor = 18 :=
by sorry

end NUMINAMATH_GPT_divisor_is_18_l2022_202291


namespace NUMINAMATH_GPT_evaluate_expression_l2022_202274

theorem evaluate_expression : (7 - 3) ^ 2 + (7 ^ 2 - 3 ^ 2) = 56 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2022_202274


namespace NUMINAMATH_GPT_sophia_pages_difference_l2022_202213

theorem sophia_pages_difference (total_pages : ℕ) (f_fraction : ℚ) (l_fraction : ℚ) 
  (finished_pages : ℕ) (left_pages : ℕ) :
  f_fraction = 2/3 ∧ 
  l_fraction = 1/3 ∧
  total_pages = 270 ∧
  finished_pages = f_fraction * total_pages ∧
  left_pages = l_fraction * total_pages
  →
  finished_pages - left_pages = 90 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sophia_pages_difference_l2022_202213


namespace NUMINAMATH_GPT_average_speed_of_journey_is_24_l2022_202201

noncomputable def average_speed (D : ℝ) (speed_to_office speed_to_home : ℝ) : ℝ :=
  let time_to_office := D / speed_to_office
  let time_to_home := D / speed_to_home
  let total_distance := 2 * D
  let total_time := time_to_office + time_to_home
  total_distance / total_time

theorem average_speed_of_journey_is_24 (D : ℝ) : average_speed D 20 30 = 24 := by
  -- nonconstructive proof to fulfill theorem definition
  sorry

end NUMINAMATH_GPT_average_speed_of_journey_is_24_l2022_202201


namespace NUMINAMATH_GPT_max_correct_answers_l2022_202255

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 25) (h2 : 6 * c - 3 * w = 60) : c ≤ 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_correct_answers_l2022_202255


namespace NUMINAMATH_GPT_solution_l2022_202260

noncomputable def problem (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (∀ x : ℝ, x^2 - 14 * p * x - 15 * q = 0 → x = r ∨ x = s) ∧
  (∀ x : ℝ, x^2 - 14 * r * x - 15 * s = 0 → x = p ∨ x = q)

theorem solution (p q r s : ℝ) (h : problem p q r s) : p + q + r + s = 3150 :=
sorry

end NUMINAMATH_GPT_solution_l2022_202260


namespace NUMINAMATH_GPT_number_subtracted_l2022_202200

theorem number_subtracted (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 :=
by
  sorry

end NUMINAMATH_GPT_number_subtracted_l2022_202200


namespace NUMINAMATH_GPT_probability_of_9_heads_in_12_flips_l2022_202292

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end NUMINAMATH_GPT_probability_of_9_heads_in_12_flips_l2022_202292


namespace NUMINAMATH_GPT_irrational_pi_l2022_202211

theorem irrational_pi :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (π = a / b)) :=
sorry

end NUMINAMATH_GPT_irrational_pi_l2022_202211


namespace NUMINAMATH_GPT_ratio_of_numbers_l2022_202224

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l2022_202224


namespace NUMINAMATH_GPT_min_bdf_proof_exists_l2022_202231

noncomputable def minBDF (a b c d e f : ℕ) (A : ℕ) :=
  (A = 3 * a ∧ A = 4 * c ∧ A = 5 * e) →
  (a / b * c / d * e / f = A) →
  b * d * f = 60

theorem min_bdf_proof_exists :
  ∃ (a b c d e f A : ℕ), minBDF a b c d e f A :=
by
  sorry

end NUMINAMATH_GPT_min_bdf_proof_exists_l2022_202231


namespace NUMINAMATH_GPT_value_of_m_l2022_202286

theorem value_of_m (m : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∃ (k : ℝ), (2 * m - 1) * x ^ (m ^ 2) = k * x ^ n) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l2022_202286


namespace NUMINAMATH_GPT_conditions_for_inequality_l2022_202290

theorem conditions_for_inequality (a b : ℝ) :
  (∀ x : ℝ, abs ((x^2 + a * x + b) / (x^2 + 2 * x + 2)) < 1) → 
  (a = 2 ∧ 0 < b ∧ b < 2) :=
sorry

end NUMINAMATH_GPT_conditions_for_inequality_l2022_202290


namespace NUMINAMATH_GPT_polynomial_identity_l2022_202251

theorem polynomial_identity
  (z1 z2 : ℂ)
  (h1 : z1 + z2 = -6)
  (h2 : z1 * z2 = 11)
  : (1 + z1^2 * z2) * (1 + z1 * z2^2) = 1266 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l2022_202251


namespace NUMINAMATH_GPT_parabola_directrix_l2022_202204

theorem parabola_directrix (x y : ℝ) (h : x^2 + 12 * y = 0) : y = 3 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l2022_202204


namespace NUMINAMATH_GPT_number_of_solutions_l2022_202263

open Real

-- Define main condition
def condition (θ : ℝ) : Prop := sin θ * tan θ = 2 * (cos θ)^2

-- Define the interval and exclusions
def valid_theta (θ : ℝ) : Prop := 
  0 ≤ θ ∧ θ ≤ 2 * π ∧ ¬ ( ∃ k : ℤ, (θ = k * (π/2)) )

-- Define the set of thetas that satisfy both the condition and the valid interval
def valid_solutions (θ : ℝ) : Prop := valid_theta θ ∧ condition θ

-- Formal statement of the problem
theorem number_of_solutions : 
  ∃ (s : Finset ℝ), (∀ θ ∈ s, valid_solutions θ) ∧ (s.card = 4) := by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l2022_202263


namespace NUMINAMATH_GPT_find_integer_solutions_l2022_202288

noncomputable def integer_solutions (x y z w : ℤ) : Prop :=
  x * y * z / w + y * z * w / x + z * w * x / y + w * x * y / z = 4

theorem find_integer_solutions :
  { (x, y, z, w) : ℤ × ℤ × ℤ × ℤ |
    integer_solutions x y z w } =
  {(1,1,1,1), (-1,-1,-1,-1), (-1,-1,1,1), (-1,1,-1,1),
   (-1,1,1,-1), (1,-1,-1,1), (1,-1,1,-1), (1,1,-1,-1)} :=
by
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l2022_202288


namespace NUMINAMATH_GPT_elise_saving_correct_l2022_202210

-- Definitions based on the conditions
def initial_money : ℤ := 8
def spent_comic_book : ℤ := 2
def spent_puzzle : ℤ := 18
def final_money : ℤ := 1

-- The theorem to prove the amount saved
theorem elise_saving_correct (x : ℤ) : 
  initial_money + x - spent_comic_book - spent_puzzle = final_money → x = 13 :=
by
  sorry

end NUMINAMATH_GPT_elise_saving_correct_l2022_202210


namespace NUMINAMATH_GPT_c_less_than_a_l2022_202268

variable (a b c : ℝ)

-- Conditions definitions
def are_negative : Prop := a < 0 ∧ b < 0 ∧ c < 0
def eq1 : Prop := c = 2 * (a + b)
def eq2 : Prop := c = 3 * (b - a)

-- Theorem statement
theorem c_less_than_a (h_neg : are_negative a b c) (h_eq1 : eq1 a b c) (h_eq2 : eq2 a b c) : c < a :=
  sorry

end NUMINAMATH_GPT_c_less_than_a_l2022_202268


namespace NUMINAMATH_GPT_ellipse_fixed_point_l2022_202223

theorem ellipse_fixed_point (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (c : ℝ) (h3 : c = 1) 
    (h4 : a = 2) (h5 : b = Real.sqrt 3) :
    (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
        ∃ M : ℝ × ℝ, (M.1 = 4) ∧ 
        ∃ Q : ℝ × ℝ, (Q.1= (P.1) ∧ Q.2 = - (P.2)) ∧ 
            ∃ fixed_point : ℝ × ℝ, (fixed_point.1 = 5 / 2) ∧ (fixed_point.2 = 0) ∧ 
            ∃ k, (Q.2 - M.2) = k * (Q.1 - M.1) ∧ 
            ∃ l, fixed_point.2 = l * (fixed_point.1 - M.1)) :=
sorry

end NUMINAMATH_GPT_ellipse_fixed_point_l2022_202223


namespace NUMINAMATH_GPT_smallest_three_digit_number_l2022_202293

theorem smallest_three_digit_number (digits : Finset ℕ) (h_digits : digits = {0, 3, 5, 6}) : 
  ∃ n, n = 305 ∧ ∀ m, (m ∈ digits) → (m ≠ 0) → (m < 305) → false :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_number_l2022_202293


namespace NUMINAMATH_GPT_even_gt_one_square_gt_l2022_202219

theorem even_gt_one_square_gt (m : ℕ) (h_even : ∃ k : ℕ, m = 2 * k) (h_gt_one : m > 1) : m < m * m :=
by
  sorry

end NUMINAMATH_GPT_even_gt_one_square_gt_l2022_202219


namespace NUMINAMATH_GPT_b_completes_work_alone_l2022_202298

theorem b_completes_work_alone (A_twice_B : ∀ (B : ℕ), A = 2 * B)
  (together : ℕ := 7) : ∃ (B : ℕ), 21 = 3 * together :=
by
  sorry

end NUMINAMATH_GPT_b_completes_work_alone_l2022_202298


namespace NUMINAMATH_GPT_angle_B_is_arcsin_l2022_202209

-- Define the triangle and its conditions
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ), 
    a = 8 ∧ b = Real.sqrt 3 ∧ 
    (2 * Real.cos (A - B) / 2 ^ 2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)

-- Prove that the measure of ∠B is arcsin(√3 / 10)
theorem angle_B_is_arcsin (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
sorry

end NUMINAMATH_GPT_angle_B_is_arcsin_l2022_202209


namespace NUMINAMATH_GPT_percentage_of_60_l2022_202253

theorem percentage_of_60 (x : ℝ) : 
  (0.2 * 40) + (x / 100) * 60 = 23 → x = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_60_l2022_202253


namespace NUMINAMATH_GPT_overlap_area_of_sectors_l2022_202259

/--
Given two sectors of a circle with radius 10, with centers at points P and R respectively, 
one having a central angle of 45 degrees and the other having a central angle of 90 degrees, 
prove that the area of the shaded region where they overlap is 12.5π.
-/
theorem overlap_area_of_sectors 
  (r : ℝ) (θ₁ θ₂ : ℝ) (A₁ A₂ : ℝ)
  (h₀ : r = 10)
  (h₁ : θ₁ = 45)
  (h₂ : θ₂ = 90)
  (hA₁ : A₁ = (θ₁ / 360) * π * r ^ 2)
  (hA₂ : A₂ = (θ₂ / 360) * π * r ^ 2)
  : A₁ = 12.5 * π := 
sorry

end NUMINAMATH_GPT_overlap_area_of_sectors_l2022_202259


namespace NUMINAMATH_GPT_original_price_of_sarees_l2022_202234

theorem original_price_of_sarees
  (P : ℝ)
  (h_sale_price : 0.80 * P * 0.85 = 306) :
  P = 450 :=
sorry

end NUMINAMATH_GPT_original_price_of_sarees_l2022_202234


namespace NUMINAMATH_GPT_percentage_increase_is_20_percent_l2022_202221

noncomputable def originalSalary : ℝ := 575 / 1.15
noncomputable def increasedSalary : ℝ := 600
noncomputable def percentageIncreaseTo600 : ℝ := (increasedSalary - originalSalary) / originalSalary * 100

theorem percentage_increase_is_20_percent :
  percentageIncreaseTo600 = 20 := 
by
  sorry -- The proof will go here

end NUMINAMATH_GPT_percentage_increase_is_20_percent_l2022_202221


namespace NUMINAMATH_GPT_right_handed_players_total_l2022_202250

def total_players : ℕ := 64
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def total_right_handed : ℕ := throwers + right_handed_non_throwers

theorem right_handed_players_total : total_right_handed = 55 := by
  sorry

end NUMINAMATH_GPT_right_handed_players_total_l2022_202250


namespace NUMINAMATH_GPT_M_gt_N_l2022_202272

variable (a b : ℝ)

def M := 10 * a^2 + 2 * b^2 - 7 * a + 6
def N := a^2 + 2 * b^2 + 5 * a + 1

theorem M_gt_N : M a b > N a b := by
  sorry

end NUMINAMATH_GPT_M_gt_N_l2022_202272


namespace NUMINAMATH_GPT_inequality_holds_l2022_202207

-- Define parameters for the problem
variables (p q x y z : ℝ) (n : ℕ)

-- Define the conditions on x, y, and z
def condition1 : Prop := y = x^n + p*x + q
def condition2 : Prop := z = y^n + p*y + q
def condition3 : Prop := x = z^n + p*z + q

-- Define the statement of the inequality
theorem inequality_holds (h1 : condition1 p q x y n) (h2 : condition2 p q y z n) (h3 : condition3 p q x z n):
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2022_202207


namespace NUMINAMATH_GPT_no_integer_solution_l2022_202299

theorem no_integer_solution :
  ∀ (x y : ℤ), ¬(x^4 + x + y^2 = 3 * y - 1) :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_no_integer_solution_l2022_202299


namespace NUMINAMATH_GPT_Tom_runs_60_miles_per_week_l2022_202240

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end NUMINAMATH_GPT_Tom_runs_60_miles_per_week_l2022_202240


namespace NUMINAMATH_GPT_smallest_n_l2022_202206

theorem smallest_n :
  ∃ n : ℕ, n = 10 ∧ (n * (n + 1) > 100 ∧ ∀ m : ℕ, m < n → m * (m + 1) ≤ 100) := by
  sorry

end NUMINAMATH_GPT_smallest_n_l2022_202206


namespace NUMINAMATH_GPT_speed_of_first_car_l2022_202237

variable (V1 V2 V3 : ℝ) -- Define the speeds of the three cars
variable (t x : ℝ) -- Time interval and distance from A to B

-- Conditions of the problem
axiom condition_1 : x / V1 = (x / V2) + t
axiom condition_2 : x / V2 = (x / V3) + t
axiom condition_3 : 120 / V1  = (120 / V2) + 1
axiom condition_4 : 40 / V1 = 80 / V3

-- Proof statement
theorem speed_of_first_car : V1 = 30 := by
  sorry

end NUMINAMATH_GPT_speed_of_first_car_l2022_202237


namespace NUMINAMATH_GPT_circle_equation_l2022_202296

theorem circle_equation 
  (x y : ℝ)
  (center : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (line1 : ℝ × ℝ → Prop)
  (line2 : ℝ × ℝ → Prop)
  (hx : line1 center)
  (hy : line2 tangent_point)
  (tangent_point_val : tangent_point = (2, -1))
  (line1_def : ∀ (p : ℝ × ℝ), line1 p ↔ 2 * p.1 + p.2 = 0)
  (line2_def : ∀ (p : ℝ × ℝ), line2 p ↔ p.1 + p.2 - 1 = 0) :
  (∃ (x0 y0 r : ℝ), center = (x0, y0) ∧ r > 0 ∧ (x - x0)^2 + (y - y0)^2 = r^2 ∧ 
                        (x - x0)^2 + (y - y0)^2 = (x - 1)^2 + (y + 2)^2 ∧ 
                        (x - 1)^2 + (y + 2)^2 = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_equation_l2022_202296


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l2022_202271

-- Define the inequality
def inequality (m x : ℝ) : Prop := (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0

-- Part (1): Prove the solution set for m = 0 is (-2, 1)
theorem part1_solution :
  (∀ x : ℝ, inequality 0 x → (-2 : ℝ) < x ∧ x < 1) := 
by
  sorry

-- Part (2): Prove the range of values for m such that the solution set is R
theorem part2_solution (m : ℝ) :
  (∀ x : ℝ, inequality m x) ↔ (1 ≤ m ∧ m < 9) := 
by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l2022_202271


namespace NUMINAMATH_GPT_a_minus_b_greater_than_one_l2022_202280

open Real

theorem a_minus_b_greater_than_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (f_has_three_roots : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ (Polynomial.aeval r1 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r2 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r3 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0)
  (g_no_real_roots : ∀ (x : ℝ), (2*x^2 + 2*b*x + a) ≠ 0) :
  a - b > 1 := by
  sorry

end NUMINAMATH_GPT_a_minus_b_greater_than_one_l2022_202280


namespace NUMINAMATH_GPT_arrangement_of_students_l2022_202243

theorem arrangement_of_students :
  let total_students := 5
  let total_communities := 2
  (2 ^ total_students - 2) = 30 :=
by
  let total_students := 5
  let total_communities := 2
  sorry

end NUMINAMATH_GPT_arrangement_of_students_l2022_202243


namespace NUMINAMATH_GPT_tan_half_A_mul_tan_half_C_eq_third_l2022_202289

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem tan_half_A_mul_tan_half_C_eq_third (h : a + c = 2 * b) :
  (Real.tan (A / 2)) * (Real.tan (C / 2)) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_half_A_mul_tan_half_C_eq_third_l2022_202289


namespace NUMINAMATH_GPT_final_combined_price_correct_l2022_202257

theorem final_combined_price_correct :
  let i_p := 1000
  let d_1 := 0.10
  let d_2 := 0.20
  let t_1 := 0.08
  let t_2 := 0.06
  let s_p := 30
  let c_p := 50
  let t_a := 0.05
  let price_after_first_month := i_p * (1 - d_1) * (1 + t_1)
  let price_after_second_month := price_after_first_month * (1 - d_2) * (1 + t_2)
  let screen_protector_final := s_p * (1 + t_a)
  let case_final := c_p * (1 + t_a)
  price_after_second_month + screen_protector_final + case_final = 908.256 := by
  sorry  -- Proof not required

end NUMINAMATH_GPT_final_combined_price_correct_l2022_202257


namespace NUMINAMATH_GPT_number_of_elements_in_union_l2022_202249

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem number_of_elements_in_union : ncard (A ∪ B) = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_elements_in_union_l2022_202249


namespace NUMINAMATH_GPT_sequence_and_sum_l2022_202287

-- Given conditions as definitions
def a₁ : ℕ := 1

def recurrence (a_n a_n1 : ℕ) (n : ℕ) : Prop := (a_n1 = 3 * a_n * (1 + (1 / n : ℝ)))

-- Stating the theorem
theorem sequence_and_sum (a : ℕ → ℕ) (S : ℕ → ℝ) :
  (a 1 = a₁) →
  (∀ n, recurrence (a n) (a (n + 1)) n) →
  (∀ n, a n = n * 3 ^ (n - 1)) ∧
  (∀ n, S n = (2 * n - 1) * 3 ^ n / 4 + 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_sequence_and_sum_l2022_202287


namespace NUMINAMATH_GPT_value_of_Y_l2022_202230

/- Define the conditions given in the problem -/
def first_row_arithmetic_seq (a1 d1 : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d1
def fourth_row_arithmetic_seq (a4 d4 : ℕ) (n : ℕ) : ℕ := a4 + (n - 1) * d4

/- Constants given by the problem -/
def a1 : ℕ := 3
def fourth_term_first_row : ℕ := 27
def a4 : ℕ := 6
def fourth_term_fourth_row : ℕ := 66

/- Calculating common differences for first and fourth rows -/
def d1 : ℕ := (fourth_term_first_row - a1) / 3
def d4 : ℕ := (fourth_term_fourth_row - a4) / 3

/- Note that we are given that Y is at position (2, 2)
   Express Y in definition forms -/
def Y_row := first_row_arithmetic_seq (a1 + d1) d4 2
def Y_column := fourth_row_arithmetic_seq (a4 + d4) d1 2

/- Problem statement in Lean 4 -/
theorem value_of_Y : Y_row = 35 ∧ Y_column = 35 := by
  sorry

end NUMINAMATH_GPT_value_of_Y_l2022_202230


namespace NUMINAMATH_GPT_number_solution_l2022_202252

theorem number_solution (x : ℝ) : (x / 5 + 4 = x / 4 - 4) → x = 160 := by
  intros h
  sorry

end NUMINAMATH_GPT_number_solution_l2022_202252


namespace NUMINAMATH_GPT_rival_awards_eq_24_l2022_202281

-- Definitions
def Scott_awards : ℕ := 4
def Jessie_awards (scott: ℕ) : ℕ := 3 * scott
def Rival_awards (jessie: ℕ) : ℕ := 2 * jessie

-- Theorem to prove
theorem rival_awards_eq_24 : Rival_awards (Jessie_awards Scott_awards) = 24 := by
  sorry

end NUMINAMATH_GPT_rival_awards_eq_24_l2022_202281


namespace NUMINAMATH_GPT_factorization_correct_l2022_202247

theorem factorization_correct: 
  (a : ℝ) → a^2 - 9 = (a - 3) * (a + 3) :=
by
  intro a
  sorry

end NUMINAMATH_GPT_factorization_correct_l2022_202247


namespace NUMINAMATH_GPT_size_of_each_bottle_l2022_202278

-- Defining given conditions
def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes : ℕ := 800
def bottles : ℕ := 20

-- Proving the size of each bottle in ounces
theorem size_of_each_bottle : (petals_per_rose * roses_per_bush * bushes / petals_per_ounce) / bottles = 12 := by
  sorry

end NUMINAMATH_GPT_size_of_each_bottle_l2022_202278


namespace NUMINAMATH_GPT_max_len_sequence_x_l2022_202222

theorem max_len_sequence_x :
  ∃ x : ℕ, 3088 < x ∧ x < 3091 :=
sorry

end NUMINAMATH_GPT_max_len_sequence_x_l2022_202222


namespace NUMINAMATH_GPT_number_of_sequences_of_length_100_l2022_202235

def sequence_count (n : ℕ) : ℕ :=
  3^n - 2^n

theorem number_of_sequences_of_length_100 :
  sequence_count 100 = 3^100 - 2^100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sequences_of_length_100_l2022_202235


namespace NUMINAMATH_GPT_avg_of_first_5_numbers_equal_99_l2022_202241

def avg_of_first_5 (S1 : ℕ) : ℕ := S1 / 5

theorem avg_of_first_5_numbers_equal_99
  (avg_9 : ℕ := 104) (avg_last_5 : ℕ := 100) (fifth_num : ℕ := 59)
  (sum_9 := 9 * avg_9) (sum_last_5 := 5 * avg_last_5) :
  avg_of_first_5 (sum_9 - sum_last_5 + fifth_num) = 99 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_first_5_numbers_equal_99_l2022_202241


namespace NUMINAMATH_GPT_probability_green_marbles_correct_l2022_202226

noncomputable def probability_of_two_green_marbles : ℚ :=
  let total_marbles := 12
  let green_marbles := 7
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green

theorem probability_green_marbles_correct :
  probability_of_two_green_marbles = 7 / 22 := by
    sorry

end NUMINAMATH_GPT_probability_green_marbles_correct_l2022_202226


namespace NUMINAMATH_GPT_cube_weight_doubled_side_length_l2022_202294

-- Theorem: Prove that the weight of a new cube with sides twice as long as the original cube is 40 pounds, given the conditions.
theorem cube_weight_doubled_side_length (s : ℝ) (h₁ : s > 0) (h₂ : (s^3 : ℝ) > 0) (w : ℝ) (h₃ : w = 5) : 
  8 * w = 40 :=
by
  sorry

end NUMINAMATH_GPT_cube_weight_doubled_side_length_l2022_202294


namespace NUMINAMATH_GPT_range_of_a_l2022_202265

def proposition_p (a : ℝ) : Prop :=
  (a + 6) * (a - 7) < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4 * x + a < 0

def neg_q (a : ℝ) : Prop :=
  a ≥ 4

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ neg_q a) ↔ a ∈ Set.Ioo (-6 : ℝ) (7 : ℝ) ∪ Set.Ici (4 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2022_202265


namespace NUMINAMATH_GPT_num_values_satisfying_g_g_x_eq_4_l2022_202214

def g (x : ℝ) : ℝ := sorry

theorem num_values_satisfying_g_g_x_eq_4 
  (h1 : g (-2) = 4)
  (h2 : g (2) = 4)
  (h3 : g (4) = 4)
  (h4 : ∀ x, g (x) ≠ -2)
  (h5 : ∃! x, g (x) = 2) 
  (h6 : ∃! x, g (x) = 4) 
  : ∃! x1 x2, g (g x1) = 4 ∧ g (g x2) = 4 ∧ x1 ≠ x2 :=
by
  sorry

end NUMINAMATH_GPT_num_values_satisfying_g_g_x_eq_4_l2022_202214


namespace NUMINAMATH_GPT_ball_bounce_height_l2022_202266

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (hₖ : ℕ → ℝ) :
  h₀ = 500 ∧ r = 0.6 ∧ (∀ k, hₖ k = h₀ * r^k) → 
  ∃ k, hₖ k < 3 ∧ k ≥ 22 := 
by
  sorry

end NUMINAMATH_GPT_ball_bounce_height_l2022_202266


namespace NUMINAMATH_GPT_part1_l2022_202225

variable {a b : ℝ}
variable {A B C : ℝ}
variable {S : ℝ}

-- Given Conditions
def is_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (b * Real.cos C - c * Real.cos B = 2 * a) ∧ (c = a)

-- To prove
theorem part1 (h : is_triangle A B C a b a) : B = 2 * Real.pi / 3 := sorry

end NUMINAMATH_GPT_part1_l2022_202225


namespace NUMINAMATH_GPT_problem_l2022_202220

theorem problem
  (r s t : ℝ)
  (h₀ : r^3 - 15 * r^2 + 13 * r - 8 = 0)
  (h₁ : s^3 - 15 * s^2 + 13 * s - 8 = 0)
  (h₂ : t^3 - 15 * t^2 + 13 * t - 8 = 0) :
  (r / (1 / r + s * t) + s / (1 / s + t * r) + t / (1 / t + r * s) = 199 / 9) :=
sorry

end NUMINAMATH_GPT_problem_l2022_202220


namespace NUMINAMATH_GPT_tangent_ellipse_hyperbola_l2022_202283

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ∧ x^2 - n * (y - 1)^2 = 1) → n = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tangent_ellipse_hyperbola_l2022_202283


namespace NUMINAMATH_GPT_second_grade_survey_count_l2022_202262

theorem second_grade_survey_count :
  ∀ (total_students first_ratio second_ratio third_ratio total_surveyed : ℕ),
  total_students = 1500 →
  first_ratio = 4 →
  second_ratio = 5 →
  third_ratio = 6 →
  total_surveyed = 150 →
  second_ratio * total_surveyed / (first_ratio + second_ratio + third_ratio) = 50 :=
by 
  intros total_students first_ratio second_ratio third_ratio total_surveyed
  sorry

end NUMINAMATH_GPT_second_grade_survey_count_l2022_202262


namespace NUMINAMATH_GPT_sequence_general_term_l2022_202208

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2022_202208


namespace NUMINAMATH_GPT_construction_company_sand_weight_l2022_202273

theorem construction_company_sand_weight :
  ∀ (total_weight gravel_weight : ℝ), total_weight = 14.02 → gravel_weight = 5.91 → 
  total_weight - gravel_weight = 8.11 :=
by 
  intros total_weight gravel_weight h_total h_gravel 
  sorry

end NUMINAMATH_GPT_construction_company_sand_weight_l2022_202273


namespace NUMINAMATH_GPT_infinite_series_sum_l2022_202239

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * n - 1) / 3 ^ (n + 1)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l2022_202239


namespace NUMINAMATH_GPT_solve_equation_l2022_202246

theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  2 / (x - 2) = (1 + x) / (x - 2) + 1 → x = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2022_202246


namespace NUMINAMATH_GPT_statement_T_true_for_given_values_l2022_202228

/-- Statement T: If the sum of the digits of a whole number m is divisible by 9, 
    then m is divisible by 9.
    The given values to check are 45, 54, 81, 63, and none of these. --/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem statement_T_true_for_given_values :
  ∀ (m : ℕ), (m = 45 ∨ m = 54 ∨ m = 81 ∨ m = 63) →
    (is_divisible_by_9 (sum_of_digits m) → is_divisible_by_9 m) :=
by
  intros m H
  cases H
  case inl H1 => sorry
  case inr H2 =>
    cases H2
    case inl H1 => sorry
    case inr H2 =>
      cases H2
      case inl H1 => sorry
      case inr H2 => sorry

end NUMINAMATH_GPT_statement_T_true_for_given_values_l2022_202228


namespace NUMINAMATH_GPT_exists_integers_a_b_for_m_l2022_202242

theorem exists_integers_a_b_for_m (m : ℕ) (h : 0 < m) :
  ∃ a b : ℤ, |a| ≤ m ∧ |b| ≤ m ∧ 0 < a + b * Real.sqrt 2 ∧ a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2) :=
by
  sorry

end NUMINAMATH_GPT_exists_integers_a_b_for_m_l2022_202242


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l2022_202205

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 < (Real.sqrt (a^2 + b^2)) / a) ∧ ((Real.sqrt (a^2 + b^2)) / a < (2 * Real.sqrt 3) / 3) :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l2022_202205


namespace NUMINAMATH_GPT_difference_of_squares_l2022_202284

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x + y = 15
def condition2 : Prop := x - y = 10

-- Goal to prove
theorem difference_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 150 := 
by sorry

end NUMINAMATH_GPT_difference_of_squares_l2022_202284


namespace NUMINAMATH_GPT_grazing_months_for_b_l2022_202238

/-
  We define the problem conditions and prove that b put his oxen for grazing for 5 months.
-/

theorem grazing_months_for_b (x : ℕ) :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let c_oxen := 15
  let c_months := 3
  let total_rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * x
  let c_ox_months := c_oxen * c_months
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  (c_share : ℚ) / total_rent = (c_ox_months : ℚ) / total_ox_months →
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_grazing_months_for_b_l2022_202238


namespace NUMINAMATH_GPT_bijection_if_injective_or_surjective_l2022_202282

variables {X Y : Type} [Fintype X] [Fintype Y] (f : X → Y)

theorem bijection_if_injective_or_surjective (hX : Fintype.card X = Fintype.card Y)
  (hf : Function.Injective f ∨ Function.Surjective f) : Function.Bijective f :=
by
  sorry

end NUMINAMATH_GPT_bijection_if_injective_or_surjective_l2022_202282


namespace NUMINAMATH_GPT_math_problem_proof_l2022_202261

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end NUMINAMATH_GPT_math_problem_proof_l2022_202261


namespace NUMINAMATH_GPT_cost_of_building_fence_square_plot_l2022_202256

-- Definition of conditions
def area_of_square_plot : ℕ := 289
def price_per_foot : ℕ := 60

-- Resulting theorem statement
theorem cost_of_building_fence_square_plot : 
  let side_length := Int.sqrt area_of_square_plot
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 4080 := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cost_of_building_fence_square_plot_l2022_202256


namespace NUMINAMATH_GPT_boxes_needed_l2022_202269

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end NUMINAMATH_GPT_boxes_needed_l2022_202269


namespace NUMINAMATH_GPT_michael_passes_donovan_l2022_202264

theorem michael_passes_donovan
  (track_length : ℕ)
  (donovan_lap_time : ℕ)
  (michael_lap_time : ℕ)
  (start_time : ℕ)
  (L : ℕ)
  (h1 : track_length = 500)
  (h2 : donovan_lap_time = 45)
  (h3 : michael_lap_time = 40)
  (h4 : start_time = 0)
  : L = 9 :=
by
  sorry

end NUMINAMATH_GPT_michael_passes_donovan_l2022_202264


namespace NUMINAMATH_GPT_x_minus_y_values_l2022_202244

theorem x_minus_y_values (x y : ℝ) 
  (h1 : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) : x - y = -1 ∨ x - y = -7 := 
  sorry

end NUMINAMATH_GPT_x_minus_y_values_l2022_202244


namespace NUMINAMATH_GPT_two_discounts_l2022_202258

theorem two_discounts (p : ℝ) : (0.9 * 0.9 * p) = 0.81 * p :=
by
  sorry

end NUMINAMATH_GPT_two_discounts_l2022_202258


namespace NUMINAMATH_GPT_sum_intercepts_of_line_l2022_202248

theorem sum_intercepts_of_line (x y : ℝ) (h_eq : y - 6 = -2 * (x - 3)) :
  (∃ x_int : ℝ, (0 - 6 = -2 * (x_int - 3)) ∧ x_int = 6) ∧
  (∃ y_int : ℝ, (y_int - 6 = -2 * (0 - 3)) ∧ y_int = 12) →
  6 + 12 = 18 :=
by sorry

end NUMINAMATH_GPT_sum_intercepts_of_line_l2022_202248


namespace NUMINAMATH_GPT_total_oranges_and_weight_l2022_202202

theorem total_oranges_and_weight 
  (oranges_per_child : ℕ) (num_children : ℕ) (average_weight_per_orange : ℝ)
  (h1 : oranges_per_child = 3)
  (h2 : num_children = 4)
  (h3 : average_weight_per_orange = 0.3) :
  oranges_per_child * num_children = 12 ∧ (oranges_per_child * num_children : ℝ) * average_weight_per_orange = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_and_weight_l2022_202202


namespace NUMINAMATH_GPT_no_integer_polynomial_exists_l2022_202285

theorem no_integer_polynomial_exists 
    (a b c d : ℤ) (h : a ≠ 0) (P : ℤ → ℤ) 
    (h1 : ∀ x, P x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    (h2 : P 4 = 1) (h3 : P 7 = 2) : 
    false := 
by
    sorry

end NUMINAMATH_GPT_no_integer_polynomial_exists_l2022_202285


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l2022_202232

theorem line_tangent_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → 16 - 16 * c = 0) → c = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l2022_202232


namespace NUMINAMATH_GPT_max_net_income_meeting_point_l2022_202212

theorem max_net_income_meeting_point :
  let A := (9 : ℝ)
  let B := (6 : ℝ)
  let cost_per_mile := 1
  let payment_per_mile := 2
  ∃ x : ℝ, 
  let AP := Real.sqrt ((x - 9)^2 + 12^2)
  let PB := Real.sqrt ((x - 6)^2 + 3^2)
  let net_income := payment_per_mile * PB - (AP + PB)
  x = -12.5 := 
sorry

end NUMINAMATH_GPT_max_net_income_meeting_point_l2022_202212


namespace NUMINAMATH_GPT_sum_of_two_longest_altitudes_l2022_202203

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_longest_altitudes_l2022_202203


namespace NUMINAMATH_GPT_john_average_increase_l2022_202254

theorem john_average_increase :
  let initial_scores := [92, 85, 91]
  let fourth_score := 95
  let initial_avg := (initial_scores.sum / initial_scores.length : ℚ)
  let new_avg := ((initial_scores.sum + fourth_score) / (initial_scores.length + 1) : ℚ)
  new_avg - initial_avg = 1.42 := 
by 
  sorry

end NUMINAMATH_GPT_john_average_increase_l2022_202254


namespace NUMINAMATH_GPT_abs_neg_one_tenth_l2022_202245

theorem abs_neg_one_tenth : |(-1 : ℚ) / 10| = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_one_tenth_l2022_202245


namespace NUMINAMATH_GPT_gum_total_l2022_202233

theorem gum_total (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) : 
  58 + x + y = 58 + x + y :=
by sorry

end NUMINAMATH_GPT_gum_total_l2022_202233


namespace NUMINAMATH_GPT_remainder_difference_l2022_202217

theorem remainder_difference :
  ∃ (d r: ℤ), (1 < d) ∧ (1250 % d = r) ∧ (1890 % d = r) ∧ (2500 % d = r) ∧ (d - r = 10) :=
sorry

end NUMINAMATH_GPT_remainder_difference_l2022_202217


namespace NUMINAMATH_GPT_geometric_sequence_condition_l2022_202236

theorem geometric_sequence_condition (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → (a * d = b * c) ∧ 
  ¬ (∀ a b c d : ℝ, a * d = b * c → ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l2022_202236


namespace NUMINAMATH_GPT_point_in_second_quadrant_l2022_202277

structure Point where
  x : Int
  y : Int

-- Define point P
def P : Point := { x := -1, y := 2 }

-- Define the second quadrant condition
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- The mathematical statement to prove
theorem point_in_second_quadrant : second_quadrant P := by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l2022_202277


namespace NUMINAMATH_GPT_cheryl_initial_skitttles_l2022_202295

-- Given conditions
def cheryl_ends_with (ends_with : ℕ) : Prop := ends_with = 97
def kathryn_gives (gives : ℕ) : Prop := gives = 89

-- To prove: cheryl_starts_with + kathryn_gives = cheryl_ends_with
theorem cheryl_initial_skitttles (cheryl_starts_with : ℕ) :
  (∃ ends_with gives, cheryl_ends_with ends_with ∧ kathryn_gives gives ∧ 
  cheryl_starts_with + gives = ends_with) →
  cheryl_starts_with = 8 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_initial_skitttles_l2022_202295


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2022_202229

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = -1) (h2 : b = 1) :
  (4/5 * a * b - (2 * a * b^2 - 4 * (-1/5 * a * b + 3 * a^2 * b)) + 2 * a * b^2) = 12 :=
by
  have ha : a = -1 := h1
  have hb : b = 1 := h2
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2022_202229


namespace NUMINAMATH_GPT_range_of_a_l2022_202276

theorem range_of_a (a : ℝ) (h₁ : ∀ x : ℝ, x > 0 → x + 4 / x ≥ a) (h₂ : ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) :
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2022_202276


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l2022_202267

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l2022_202267


namespace NUMINAMATH_GPT_middle_card_four_or_five_l2022_202270

def three_cards (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 15 ∧ a < b ∧ b < c

theorem middle_card_four_or_five (a b c : ℕ) :
  three_cards a b c → (b = 4 ∨ b = 5) :=
by
  sorry

end NUMINAMATH_GPT_middle_card_four_or_five_l2022_202270


namespace NUMINAMATH_GPT_correct_statement_about_algorithms_l2022_202218

-- Definitions based on conditions
def is_algorithm (A B C D : Prop) : Prop :=
  ¬A ∧ B ∧ ¬C ∧ ¬D

-- Ensure the correct statement using the conditions specified
theorem correct_statement_about_algorithms (A B C D : Prop) (h : is_algorithm A B C D) : B :=
by
  obtain ⟨hnA, hB, hnC, hnD⟩ := h
  exact hB

end NUMINAMATH_GPT_correct_statement_about_algorithms_l2022_202218


namespace NUMINAMATH_GPT_evaluate_g_ggg_neg1_l2022_202227

def g (y : ℤ) : ℤ := y^3 - 3*y + 1

theorem evaluate_g_ggg_neg1 : g (g (g (-1))) = 6803 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_g_ggg_neg1_l2022_202227


namespace NUMINAMATH_GPT_cevian_concurrency_l2022_202279

-- Definitions for the acute triangle and the angles
structure AcuteTriangle (α β γ : ℝ) :=
  (A B C : ℝ)
  (acute_α : α > 0 ∧ α < π / 2)
  (acute_β : β > 0 ∧ β < π / 2)
  (acute_γ : γ > 0 ∧ γ < π / 2)
  (triangle_sum : α + β + γ = π)

-- Definition for the concurrency of cevians
def cevians_concurrent (α β γ : ℝ) (t : AcuteTriangle α β γ) :=
  ∀ (A₁ B₁ C₁ : ℝ), sorry -- placeholder

-- The main theorem with the proof of concurrency
theorem cevian_concurrency (α β γ : ℝ) (t : AcuteTriangle α β γ) :
  ∃ (A₁ B₁ C₁ : ℝ), cevians_concurrent α β γ t :=
  sorry -- proof to be provided


end NUMINAMATH_GPT_cevian_concurrency_l2022_202279


namespace NUMINAMATH_GPT_intersection_A_B_l2022_202215

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2022_202215


namespace NUMINAMATH_GPT_hall_width_l2022_202297

theorem hall_width (w : ℝ) (length height cost_per_m2 total_expenditure : ℝ)
  (h_length : length = 20)
  (h_height : height = 5)
  (h_cost : cost_per_m2 = 50)
  (h_expenditure : total_expenditure = 47500)
  (h_area : total_expenditure = cost_per_m2 * (2 * (length * w) + 2 * (length * height) + 2 * (w * height))) :
  w = 15 := 
sorry

end NUMINAMATH_GPT_hall_width_l2022_202297
