import Mathlib

namespace NUMINAMATH_GPT_two_lines_intersections_with_ellipse_l686_68633

open Set

def ellipse (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem two_lines_intersections_with_ellipse {L1 L2 : ℝ → ℝ → Prop} :
  (∀ x y, L1 x y → ¬(ellipse x y)) →
  (∀ x y, L2 x y → ¬(ellipse x y)) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧ L1 x1 y1 ∧ L1 x2 y2) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧ L2 x1 y1 ∧ L2 x2 y2) →
  ∃ n, n = 2 ∨ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_two_lines_intersections_with_ellipse_l686_68633


namespace NUMINAMATH_GPT_inscribed_circle_radius_third_of_circle_l686_68686

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ := 
  R * (Real.sqrt 3 - 1) / 2

theorem inscribed_circle_radius_third_of_circle (R : ℝ) (hR : R = 5) :
  inscribed_circle_radius R = 5 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_third_of_circle_l686_68686


namespace NUMINAMATH_GPT_two_pow_div_factorial_iff_l686_68665

theorem two_pow_div_factorial_iff (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k - 1)) ↔ (∃ m : ℕ, m > 0 ∧ 2^(n - 1) ∣ n!) :=
by
  sorry

end NUMINAMATH_GPT_two_pow_div_factorial_iff_l686_68665


namespace NUMINAMATH_GPT_inequality_positive_integers_l686_68647

theorem inequality_positive_integers (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_inequality_positive_integers_l686_68647


namespace NUMINAMATH_GPT_matrix_not_invertible_l686_68691

def is_not_invertible_matrix (y : ℝ) : Prop :=
  let a := 2 + y
  let b := 9
  let c := 4 - y
  let d := 10
  a * d - b * c = 0

theorem matrix_not_invertible (y : ℝ) : is_not_invertible_matrix y ↔ y = 16 / 19 :=
  sorry

end NUMINAMATH_GPT_matrix_not_invertible_l686_68691


namespace NUMINAMATH_GPT_man_speed_in_still_water_l686_68646

theorem man_speed_in_still_water :
  ∃ (V_m V_s : ℝ), 
  V_m + V_s = 14 ∧ 
  V_m - V_s = 6 ∧ 
  V_m = 10 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l686_68646


namespace NUMINAMATH_GPT_maximum_even_integers_of_odd_product_l686_68698

theorem maximum_even_integers_of_odd_product (a b c d e f g : ℕ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) (h5: e > 0) (h6: f > 0) (h7: g > 0) (hprod : a * b * c * d * e * f * g % 2 = 1): 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) ∧ (g % 2 = 1) :=
sorry

end NUMINAMATH_GPT_maximum_even_integers_of_odd_product_l686_68698


namespace NUMINAMATH_GPT_simple_interest_correct_l686_68685

-- Define the principal amount P
variables {P : ℝ}

-- Define the rate of interest r which is 3% or 0.03 in decimal form
def r : ℝ := 0.03

-- Define the time period t which is 2 years
def t : ℕ := 2

-- Define the compound interest CI for 2 years which is $609
def CI : ℝ := 609

-- Define the simple interest SI that we need to find
def SI : ℝ := 600

-- Define a formula for compound interest
def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

-- Define a formula for simple interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem simple_interest_correct (hCI : compound_interest P r t = CI) : simple_interest P r t = SI :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_correct_l686_68685


namespace NUMINAMATH_GPT_total_surface_area_of_square_pyramid_is_correct_l686_68630

-- Define the base side length and height from conditions
def a : ℝ := 3
def PD : ℝ := 4

-- Conditions
def square_pyramid : Prop :=
  let AD := a
  let PA := Real.sqrt (PD^2 - a^2)
  let Area_PAD := (1 / 2) * AD * PA
  let Area_PCD := Area_PAD
  let Area_base := a * a
  let Total_surface_area := Area_base + 2 * Area_PAD + 2 * Area_PCD
  Total_surface_area = 9 + 6 * Real.sqrt 7

-- Theorem statement
theorem total_surface_area_of_square_pyramid_is_correct : square_pyramid := sorry

end NUMINAMATH_GPT_total_surface_area_of_square_pyramid_is_correct_l686_68630


namespace NUMINAMATH_GPT_hyperbola_line_intersections_l686_68679

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Conditions for intersecting the hyperbola at two points
def intersect_two_points (k : ℝ) : Prop := 
  k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∨ 
  k ∈ Set.Ioo (-1) 1 ∨ 
  k ∈ Set.Ioo 1 (2 * Real.sqrt 3 / 3)

-- Conditions for intersecting the hyperbola at exactly one point
def intersect_one_point (k : ℝ) : Prop := 
  k = 1 ∨ 
  k = -1 ∨ 
  k = 2 * Real.sqrt 3 / 3 ∨ 
  k = -2 * Real.sqrt 3 / 3

-- Proof that k is in the appropriate ranges
theorem hyperbola_line_intersections (k : ℝ) :
  ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ hyperbola x₁ y₁ ∧ line x₁ y₁ k ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ k) 
  → intersect_two_points k))
  ∧ ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x y : ℝ, (hyperbola x y ∧ line x y k ∧ (∀ x' y', hyperbola x' y' ∧ line x' y' k → (x' ≠ x ∨ y' ≠ y) = false)) 
  → intersect_one_point k)) := 
sorry

end NUMINAMATH_GPT_hyperbola_line_intersections_l686_68679


namespace NUMINAMATH_GPT_profit_amount_l686_68620

theorem profit_amount (SP : ℝ) (P : ℝ) (profit : ℝ) : 
  SP = 850 → P = 36 → profit = SP - SP / (1 + P / 100) → profit = 225 :=
by
  intros hSP hP hProfit
  rw [hSP, hP] at *
  simp at *
  sorry

end NUMINAMATH_GPT_profit_amount_l686_68620


namespace NUMINAMATH_GPT_find_angle_A_l686_68632

theorem find_angle_A (a b c : ℝ) (h : a^2 - c^2 = b^2 - b * c) : 
  ∃ (A : ℝ), A = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l686_68632


namespace NUMINAMATH_GPT_sum_as_fraction_l686_68629

theorem sum_as_fraction :
  (0.1 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) = (13467 / 100000 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_sum_as_fraction_l686_68629


namespace NUMINAMATH_GPT_factor_difference_of_squares_l686_68604

theorem factor_difference_of_squares (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l686_68604


namespace NUMINAMATH_GPT_percent_of_y_l686_68617

theorem percent_of_y (y : ℝ) (h : y > 0) : (2 * y) / 10 + (3 * y) / 10 = (50 / 100) * y :=
by
  sorry

end NUMINAMATH_GPT_percent_of_y_l686_68617


namespace NUMINAMATH_GPT_inequality_proof_l686_68613

variable (ha la r R : ℝ)
variable (α β γ : ℝ)

-- Conditions
def condition1 : Prop := ha / la = Real.cos ((β - γ) / 2)
def condition2 : Prop := 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2) = 2 * r / R

-- The theorem to be proved
theorem inequality_proof (h1 : condition1 ha la β γ) (h2 : condition2 α β γ r R) :
  Real.cos ((β - γ) / 2) ≥ Real.sqrt (2 * r / R) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l686_68613


namespace NUMINAMATH_GPT_b_2016_result_l686_68603

theorem b_2016_result (b : ℕ → ℤ) (h₁ : b 1 = 1) (h₂ : b 2 = 5)
  (h₃ : ∀ n : ℕ, b (n + 2) = b (n + 1) - b n) : b 2016 = -4 := sorry

end NUMINAMATH_GPT_b_2016_result_l686_68603


namespace NUMINAMATH_GPT_num_divisible_by_7_200_to_400_l686_68677

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end NUMINAMATH_GPT_num_divisible_by_7_200_to_400_l686_68677


namespace NUMINAMATH_GPT_total_cost_of_pencils_l686_68687

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end NUMINAMATH_GPT_total_cost_of_pencils_l686_68687


namespace NUMINAMATH_GPT_mrs_hilt_rocks_l686_68689

def garden_length := 10
def garden_width := 15
def rock_coverage := 1
def available_rocks := 64

theorem mrs_hilt_rocks :
  ∃ extra_rocks : ℕ, 2 * (garden_length + garden_width) <= available_rocks ∧ extra_rocks = available_rocks - 2 * (garden_length + garden_width) ∧ extra_rocks = 14 :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_rocks_l686_68689


namespace NUMINAMATH_GPT_variance_of_data_set_l686_68661

def data_set : List ℤ := [ -2, -1, 0, 3, 5 ]

def mean (l : List ℤ) : ℚ :=
  (l.sum / l.length)

def variance (l : List ℤ) : ℚ :=
  (1 / l.length) * (l.map (λ x => (x - mean l : ℚ)^2)).sum

theorem variance_of_data_set : variance data_set = 34 / 5 := by
  sorry

end NUMINAMATH_GPT_variance_of_data_set_l686_68661


namespace NUMINAMATH_GPT_length_of_the_train_is_120_l686_68641

noncomputable def train_length (time: ℝ) (speed_km_hr: ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time

theorem length_of_the_train_is_120 :
  train_length 3.569962336897346 121 = 120 := by
  sorry

end NUMINAMATH_GPT_length_of_the_train_is_120_l686_68641


namespace NUMINAMATH_GPT_sequence_sum_l686_68657

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

noncomputable def S : ℕ → ℝ
| 0     => 0
| (n+1) => S n + a (n+1)

theorem sequence_sum : S 2017 = 1008 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l686_68657


namespace NUMINAMATH_GPT_simplest_form_of_expression_l686_68699

theorem simplest_form_of_expression (c : ℝ) : ((3 * c + 5 - 3 * c) / 2) = 5 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplest_form_of_expression_l686_68699


namespace NUMINAMATH_GPT_simon_spending_l686_68682

-- Assume entities and their properties based on the problem
def kabobStickCubes : Nat := 4
def slabCost : Nat := 25
def slabCubes : Nat := 80
def kabobSticksNeeded : Nat := 40

-- Theorem statement based on the problem analysis
theorem simon_spending : 
  (kabobSticksNeeded / (slabCubes / kabobStickCubes)) * slabCost = 50 := by
  sorry

end NUMINAMATH_GPT_simon_spending_l686_68682


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_is_not_term_l686_68694

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) (h17 : a 17 = 66) :
  ∀ n : ℕ, a n = 4 * n - 2 := sorry

theorem is_not_term (a : ℕ → ℤ) 
  (ha : ∀ n : ℕ, a n = 4 * n - 2) :
  ∀ k : ℤ, k = 88 → ¬ ∃ n : ℕ, a n = k := sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_is_not_term_l686_68694


namespace NUMINAMATH_GPT_robins_initial_hair_length_l686_68637

variable (L : ℕ)

def initial_length_after_cutting := L - 11
def length_after_growth := initial_length_after_cutting L + 12
def final_length := 17

theorem robins_initial_hair_length : length_after_growth L = final_length → L = 16 := 
by sorry

end NUMINAMATH_GPT_robins_initial_hair_length_l686_68637


namespace NUMINAMATH_GPT_range_of_c_l686_68635

variable (c : ℝ)

def p : Prop := ∀ x : ℝ, x > 0 → c^x = c^(x+1) / c
def q : Prop := ∀ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) → x + 1/x > 1/c

theorem range_of_c (h1 : c > 0) (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) :
  (0 < c ∧ c ≤ 1/2) ∨ (c ≥ 1) :=
sorry

end NUMINAMATH_GPT_range_of_c_l686_68635


namespace NUMINAMATH_GPT_range_of_ab_c2_l686_68627

theorem range_of_ab_c2 (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
    0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
sorry

end NUMINAMATH_GPT_range_of_ab_c2_l686_68627


namespace NUMINAMATH_GPT_line_intersects_ellipse_l686_68666

theorem line_intersects_ellipse (b : ℝ) : (∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + 1 → ((x^2 / 5) + (y^2 / b) = 1))
  ↔ b ∈ (Set.Ico 1 5 ∪ Set.Ioi 5) := by
sorry

end NUMINAMATH_GPT_line_intersects_ellipse_l686_68666


namespace NUMINAMATH_GPT_frank_fence_l686_68696

theorem frank_fence (L W F : ℝ) (hL : L = 40) (hA : 320 = L * W) : F = 2 * W + L → F = 56 := by
  sorry

end NUMINAMATH_GPT_frank_fence_l686_68696


namespace NUMINAMATH_GPT_construction_paper_initial_count_l686_68644

theorem construction_paper_initial_count 
    (b r d : ℕ)
    (ratio_cond : b = 2 * r)
    (daily_usage : ∀ n : ℕ, n ≤ d → n * 1 = b ∧ n * 3 = r)
    (last_day_cond : 0 = b ∧ 15 = r):
    b + r = 135 :=
sorry

end NUMINAMATH_GPT_construction_paper_initial_count_l686_68644


namespace NUMINAMATH_GPT_find_number_of_male_students_l686_68680

/- Conditions: 
 1. n ≡ 2 [MOD 4]
 2. n ≡ 1 [MOD 5]
 3. n > 15
 4. There are 15 female students
 5. There are more female students than male students
-/
theorem find_number_of_male_students (n : ℕ) (females : ℕ) (h1 : n % 4 = 2) (h2 : n % 5 = 1) (h3 : n > 15) (h4 : females = 15) (h5 : females > n - females) : (n - females) = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_male_students_l686_68680


namespace NUMINAMATH_GPT_tangent_line_to_curve_determines_m_l686_68643

theorem tangent_line_to_curve_determines_m :
  ∃ m : ℝ, (∀ x : ℝ, y = x ^ 4 + m * x) ∧ (2 * -1 + y' + 3 = 0) ∧ (y' = -2) → (m = 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_curve_determines_m_l686_68643


namespace NUMINAMATH_GPT_new_total_lines_is_240_l686_68640

-- Define the original number of lines, the increase, and the percentage increase
variables (L : ℝ) (increase : ℝ := 110) (percentage_increase : ℝ := 84.61538461538461 / 100)

-- The statement to prove
theorem new_total_lines_is_240 (h : increase = percentage_increase * L) : L + increase = 240 := sorry

end NUMINAMATH_GPT_new_total_lines_is_240_l686_68640


namespace NUMINAMATH_GPT_div_by_six_l686_68650

theorem div_by_six (n : ℕ) : 6 ∣ (17^n - 11^n) :=
by
  sorry

end NUMINAMATH_GPT_div_by_six_l686_68650


namespace NUMINAMATH_GPT_taxes_paid_l686_68609

theorem taxes_paid (gross_pay net_pay : ℤ) (h1 : gross_pay = 450) (h2 : net_pay = 315) :
  gross_pay - net_pay = 135 := 
by 
  rw [h1, h2] 
  norm_num

end NUMINAMATH_GPT_taxes_paid_l686_68609


namespace NUMINAMATH_GPT_smallest_sum_of_xy_l686_68634

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end NUMINAMATH_GPT_smallest_sum_of_xy_l686_68634


namespace NUMINAMATH_GPT_avg_rate_change_l686_68602

def f (x : ℝ) : ℝ := x^2 + x

theorem avg_rate_change : (f 2 - f 1) / (2 - 1) = 4 := by
  -- here the proof steps should follow
  sorry

end NUMINAMATH_GPT_avg_rate_change_l686_68602


namespace NUMINAMATH_GPT_solve_equation_l686_68615

theorem solve_equation : ∃ x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := 
by
  -- Introducing x as a real number and stating the goal
  use 4
  -- Show that 2 * 4 - 3 = 5
  simp
  -- Adding the sorry to skip the proof step
  sorry

end NUMINAMATH_GPT_solve_equation_l686_68615


namespace NUMINAMATH_GPT_number_of_possible_IDs_l686_68636

theorem number_of_possible_IDs : 
  ∃ (n : ℕ), 
  (∀ (a b : Fin 26) (x y : Fin 10),
    a = b ∨ x = y ∨ (a = b ∧ x = y) → 
    n = 9100) :=
sorry

end NUMINAMATH_GPT_number_of_possible_IDs_l686_68636


namespace NUMINAMATH_GPT_smallest_consecutive_even_sum_140_l686_68676

theorem smallest_consecutive_even_sum_140 :
  ∃ (x : ℕ), (x % 2 = 0) ∧ (x + (x + 2) + (x + 4) + (x + 6) = 140) ∧ (x = 32) :=
by
  sorry

end NUMINAMATH_GPT_smallest_consecutive_even_sum_140_l686_68676


namespace NUMINAMATH_GPT_negation_of_proposition_l686_68678

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℤ, 2 * x_0 + x_0 + 1 ≤ 0) ↔ ∀ x : ℤ, 2 * x + x + 1 > 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l686_68678


namespace NUMINAMATH_GPT_algebra_expression_value_l686_68663

theorem algebra_expression_value (x y : ℝ) (h : x = 2 * y + 1) : x^2 - 4 * x * y + 4 * y^2 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l686_68663


namespace NUMINAMATH_GPT_triangles_exist_l686_68688

def exists_triangles : Prop :=
  ∃ (T : Fin 100 → Type) 
    (h : (i : Fin 100) → ℝ) 
    (A : (i : Fin 100) → ℝ)
    (is_isosceles : (i : Fin 100) → Prop),
    (∀ i : Fin 100, is_isosceles i) ∧
    (∀ i : Fin 99, h (i + 1) = 200 * h i) ∧
    (∀ i : Fin 99, A (i + 1) = A i / 20000) ∧
    (∀ i : Fin 100, 
      ¬(∃ (cover : (Fin 99) → Type),
        (∀ j : Fin 99, cover j = T j) ∧
        (∀ j : Fin 99, ∀ k : Fin 100, k ≠ i → ¬(cover j = T k))))

theorem triangles_exist : exists_triangles :=
sorry

end NUMINAMATH_GPT_triangles_exist_l686_68688


namespace NUMINAMATH_GPT_odd_power_of_7_plus_1_divisible_by_8_l686_68659

theorem odd_power_of_7_plus_1_divisible_by_8 (n : ℕ) (h : n % 2 = 1) : (7 ^ n + 1) % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_power_of_7_plus_1_divisible_by_8_l686_68659


namespace NUMINAMATH_GPT_BC_equals_expected_BC_l686_68610

def point := ℝ × ℝ -- Define a point as a pair of real numbers (coordinates).

def vector_sub (v1 v2 : point) : point := (v1.1 - v2.1, v1.2 - v2.2) -- Define vector subtraction.

-- Definitions of points A and B and vector AC
def A : point := (-1, 1)
def B : point := (0, 2)
def AC : point := (-2, 3)

-- Calculate vector AB
def AB : point := vector_sub B A

-- Calculate vector BC
def BC : point := vector_sub AC AB

-- Expected result
def expected_BC : point := (-3, 2)

-- Proof statement
theorem BC_equals_expected_BC : BC = expected_BC := by
  unfold BC AB AC A B vector_sub
  simp
  sorry

end NUMINAMATH_GPT_BC_equals_expected_BC_l686_68610


namespace NUMINAMATH_GPT_sue_answer_is_106_l686_68618

-- Definitions based on conditions
def ben_step1 (x : ℕ) : ℕ := x * 3
def ben_step2 (x : ℕ) : ℕ := ben_step1 x + 2
def ben_step3 (x : ℕ) : ℕ := ben_step2 x * 2

def sue_step1 (y : ℕ) : ℕ := y + 3
def sue_step2 (y : ℕ) : ℕ := sue_step1 y - 2
def sue_step3 (y : ℕ) : ℕ := sue_step2 y * 2

-- Ben starts with the number 8
def ben_number : ℕ := 8

-- Ben gives the number to Sue
def given_to_sue : ℕ := ben_step3 ben_number

-- Lean statement to prove
theorem sue_answer_is_106 : sue_step3 given_to_sue = 106 :=
by
  sorry

end NUMINAMATH_GPT_sue_answer_is_106_l686_68618


namespace NUMINAMATH_GPT_union_of_sets_eq_l686_68653

variable (M N : Set ℕ)

theorem union_of_sets_eq (h1 : M = {1, 2}) (h2 : N = {2, 3}) : M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_eq_l686_68653


namespace NUMINAMATH_GPT_sum_of_squares_l686_68614

def gcd (a b c : Nat) : Nat := (Nat.gcd (Nat.gcd a b) c)

theorem sum_of_squares {a b c : ℕ} (h1 : 3 * a + 2 * b = 4 * c)
                                   (h2 : 3 * c ^ 2 = 4 * a ^ 2 + 2 * b ^ 2)
                                   (h3 : gcd a b c = 1) :
  a^2 + b^2 + c^2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l686_68614


namespace NUMINAMATH_GPT_rectangle_perimeter_l686_68638

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b = 2 * (a + b))) : 2 * (a + b) = 36 :=
by sorry

end NUMINAMATH_GPT_rectangle_perimeter_l686_68638


namespace NUMINAMATH_GPT_inequality_implies_bounds_l686_68667

open Real

theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, (exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → (0 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_GPT_inequality_implies_bounds_l686_68667


namespace NUMINAMATH_GPT_simplify_and_evaluate_l686_68697

theorem simplify_and_evaluate
  (a b : ℝ)
  (h : |a - 1| + (b + 2)^2 = 0) :
  ((2 * a + b)^2 - (2 * a + b) * (2 * a - b)) / (-1 / 2 * b) = 0 := 
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l686_68697


namespace NUMINAMATH_GPT_sides_of_rectangle_EKMR_l686_68693

noncomputable def right_triangle_ACB (AC AB : ℕ) : Prop :=
AC = 3 ∧ AB = 4

noncomputable def rectangle_EKMR_area (area : ℚ) : Prop :=
area = 3/5

noncomputable def rectangle_EKMR_perimeter (x y : ℚ) : Prop :=
2 * (x + y) < 9

theorem sides_of_rectangle_EKMR (x y : ℚ) 
  (h_triangle : right_triangle_ACB 3 4)
  (h_area : rectangle_EKMR_area (3/5))
  (h_perimeter : rectangle_EKMR_perimeter x y) : 
  (x = 2 ∧ y = 3/10) ∨ (x = 3/10 ∧ y = 2) := 
sorry

end NUMINAMATH_GPT_sides_of_rectangle_EKMR_l686_68693


namespace NUMINAMATH_GPT_inequality_proof_l686_68651

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_condition : a * b + b * c + c * d + d * a = 1) :
    (a ^ 3 / (b + c + d)) + (b ^ 3 / (c + d + a)) + (c ^ 3 / (a + b + d)) + (d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l686_68651


namespace NUMINAMATH_GPT_correct_equation_l686_68649

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end NUMINAMATH_GPT_correct_equation_l686_68649


namespace NUMINAMATH_GPT_solution_k_system_eq_l686_68639

theorem solution_k_system_eq (x y k : ℝ) 
  (h1 : x + y = 5 * k) 
  (h2 : x - y = k) 
  (h3 : 2 * x + 3 * y = 24) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_k_system_eq_l686_68639


namespace NUMINAMATH_GPT_coin_probability_l686_68645

theorem coin_probability (p : ℚ) 
  (P_X_3 : ℚ := 10 * p^3 * (1 - p)^2)
  (P_X_4 : ℚ := 5 * p^4 * (1 - p))
  (P_X_5 : ℚ := p^5)
  (w : ℚ := P_X_3 + P_X_4 + P_X_5) :
  w = 5 / 16 → p = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_coin_probability_l686_68645


namespace NUMINAMATH_GPT_yellow_marbles_l686_68600

-- Define the conditions from a)
variables (total_marbles red blue green yellow : ℕ)
variables (h1 : total_marbles = 110)
variables (h2 : red = 8)
variables (h3 : blue = 4 * red)
variables (h4 : green = 2 * blue)
variables (h5 : yellow = total_marbles - (red + blue + green))

-- Prove the question in c)
theorem yellow_marbles : yellow = 6 :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_yellow_marbles_l686_68600


namespace NUMINAMATH_GPT_rationalize_denominator_l686_68648

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l686_68648


namespace NUMINAMATH_GPT_find_M_l686_68623

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the complement of M with respect to U
def complement_M : Set ℕ := {2}

-- Define M as U without the complement of M
def M : Set ℕ := U \ complement_M

-- Prove that M is {0, 1, 3}
theorem find_M : M = {0, 1, 3} := by
  sorry

end NUMINAMATH_GPT_find_M_l686_68623


namespace NUMINAMATH_GPT_x_is_4286_percent_less_than_y_l686_68671

theorem x_is_4286_percent_less_than_y (x y : ℝ) (h : y = 1.75 * x) : 
  ((y - x) / y) * 100 = 42.86 :=
by
  sorry

end NUMINAMATH_GPT_x_is_4286_percent_less_than_y_l686_68671


namespace NUMINAMATH_GPT_S8_value_l686_68607

theorem S8_value (x : ℝ) (h : x + 1/x = 4) (S : ℕ → ℝ) (S_def : ∀ m, S m = x^m + 1/x^m) :
  S 8 = 37634 :=
sorry

end NUMINAMATH_GPT_S8_value_l686_68607


namespace NUMINAMATH_GPT_remainder_of_sum_div_18_l686_68660

theorem remainder_of_sum_div_18 :
  let nums := [11065, 11067, 11069, 11071, 11073, 11075, 11077, 11079, 11081]
  let residues := [1, 3, 5, 7, 9, 11, 13, 15, 17]
  (nums.sum % 18) = 9 := by
    sorry

end NUMINAMATH_GPT_remainder_of_sum_div_18_l686_68660


namespace NUMINAMATH_GPT_ratio_of_part_to_whole_l686_68695

theorem ratio_of_part_to_whole (N : ℝ) :
  (2/15) * N = 14 ∧ 0.40 * N = 168 → (14 / ((1/3) * (2/5) * N)) = 1 :=
by
  -- We assume the conditions given in the problem and need to prove the ratio
  intro h
  obtain ⟨h1, h2⟩ := h
  -- Establish equality through calculations
  sorry

end NUMINAMATH_GPT_ratio_of_part_to_whole_l686_68695


namespace NUMINAMATH_GPT_div_remainder_l686_68626

theorem div_remainder (B x : ℕ) (h1 : B = 301) (h2 : B % 7 = 0) : x = 3 :=
  sorry

end NUMINAMATH_GPT_div_remainder_l686_68626


namespace NUMINAMATH_GPT_total_people_museum_l686_68672

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end NUMINAMATH_GPT_total_people_museum_l686_68672


namespace NUMINAMATH_GPT_least_number_of_stamps_l686_68673

theorem least_number_of_stamps (s t : ℕ) (h : 5 * s + 7 * t = 50) : s + t = 8 :=
sorry

end NUMINAMATH_GPT_least_number_of_stamps_l686_68673


namespace NUMINAMATH_GPT_determinant_expr_l686_68619

theorem determinant_expr (a b c p q r : ℝ) 
  (h1 : ∀ x, Polynomial.eval x (Polynomial.C a * Polynomial.C b * Polynomial.C c - Polynomial.C p * (Polynomial.C a * Polynomial.C b + Polynomial.C b * Polynomial.C c + Polynomial.C c * Polynomial.C a) + Polynomial.C q * (Polynomial.C a + Polynomial.C b + Polynomial.C c) - Polynomial.C r) = 0) :
  Matrix.det ![
    ![2 + a, 1, 1],
    ![1, 2 + b, 1],
    ![1, 1, 2 + c]
  ] = r + 2*q + 4*p + 4 :=
sorry

end NUMINAMATH_GPT_determinant_expr_l686_68619


namespace NUMINAMATH_GPT_complex_div_symmetry_l686_68601

open Complex

-- Definitions based on conditions
def z1 : ℂ := 1 + I
def z2 : ℂ := -1 + I

-- Theorem to prove
theorem complex_div_symmetry : z2 / z1 = I := by
  sorry

end NUMINAMATH_GPT_complex_div_symmetry_l686_68601


namespace NUMINAMATH_GPT_problem_simplify_and_evaluate_l686_68664

theorem problem_simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - (m / (m + 3))) / ((m^2 - 9) / (m^2 + 6 * m + 9)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_simplify_and_evaluate_l686_68664


namespace NUMINAMATH_GPT_largest_value_B_l686_68662

theorem largest_value_B :
  let A := ((1 / 2) / (3 / 4))
  let B := (1 / ((2 / 3) / 4))
  let C := (((1 / 2) / 3) / 4)
  let E := ((1 / (2 / 3)) / 4)
  B > A ∧ B > C ∧ B > E :=
by
  sorry

end NUMINAMATH_GPT_largest_value_B_l686_68662


namespace NUMINAMATH_GPT_f_2023_eq_1375_l686_68655

-- Define the function f and the conditions
noncomputable def f : ℕ → ℕ := sorry

axiom f_ff_eq (n : ℕ) (h : n > 0) : f (f n) = 3 * n
axiom f_3n2_eq (n : ℕ) (h : n > 0) : f (3 * n + 2) = 3 * n + 1

-- Prove the specific value for f(2023)
theorem f_2023_eq_1375 : f 2023 = 1375 := sorry

end NUMINAMATH_GPT_f_2023_eq_1375_l686_68655


namespace NUMINAMATH_GPT_find_value_l686_68612

theorem find_value (x y z : ℝ) (h₁ : y = 3 * x) (h₂ : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end NUMINAMATH_GPT_find_value_l686_68612


namespace NUMINAMATH_GPT_coefficients_divisible_by_5_l686_68631

theorem coefficients_divisible_by_5 
  (a b c d : ℤ) 
  (h : ∀ x : ℤ, 5 ∣ (a * x^3 + b * x^2 + c * x + d)) : 
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d := 
by {
  sorry
}

end NUMINAMATH_GPT_coefficients_divisible_by_5_l686_68631


namespace NUMINAMATH_GPT_number_of_pears_in_fruit_gift_set_l686_68683

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end NUMINAMATH_GPT_number_of_pears_in_fruit_gift_set_l686_68683


namespace NUMINAMATH_GPT_even_function_must_be_two_l686_68622

def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-2)*x + (m^2 - 7*m + 12)

theorem even_function_must_be_two (m : ℝ) :
  (∀ x : ℝ, f m (-x) = f m x) ↔ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_even_function_must_be_two_l686_68622


namespace NUMINAMATH_GPT_h_of_neg_one_l686_68674

def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x) ^ 2 - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg_one :
  h (-1) = 298 :=
by
  sorry

end NUMINAMATH_GPT_h_of_neg_one_l686_68674


namespace NUMINAMATH_GPT_find_center_of_tangent_circle_l686_68669

theorem find_center_of_tangent_circle :
  ∃ (a b : ℝ), (abs a = 5) ∧ (abs b = 5) ∧ (4 * a - 3 * b + 10 = 25) ∧ (a = -5) ∧ (b = 5) :=
by {
  -- Here we would provide the proof in Lean, but for now, we state the theorem
  -- and leave the proof as an exercise.
  sorry
}

end NUMINAMATH_GPT_find_center_of_tangent_circle_l686_68669


namespace NUMINAMATH_GPT_six_digit_number_property_l686_68642

theorem six_digit_number_property :
  ∃ N : ℕ, N = 285714 ∧ (∃ x : ℕ, N = 2 * 10^5 + x ∧ M = 10 * x + 2 ∧ M = 3 * N) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_property_l686_68642


namespace NUMINAMATH_GPT_correct_percentage_fruits_in_good_condition_l686_68654

noncomputable def percentage_fruits_in_good_condition
    (total_oranges : ℕ)
    (total_bananas : ℕ)
    (rotten_percentage_oranges : ℝ)
    (rotten_percentage_bananas : ℝ) : ℝ :=
let rotten_oranges := (rotten_percentage_oranges / 100) * total_oranges
let rotten_bananas := (rotten_percentage_bananas / 100) * total_bananas
let good_condition_oranges := total_oranges - rotten_oranges
let good_condition_bananas := total_bananas - rotten_bananas
let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
let total_fruits := total_oranges + total_bananas
(total_fruits_in_good_condition / total_fruits) * 100

theorem correct_percentage_fruits_in_good_condition :
  percentage_fruits_in_good_condition 600 400 15 4 = 89.4 := by
  sorry

end NUMINAMATH_GPT_correct_percentage_fruits_in_good_condition_l686_68654


namespace NUMINAMATH_GPT_set_equality_l686_68605

noncomputable def alpha_set : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi / 2 - Real.pi / 5 ∧ (-Real.pi < α ∧ α < Real.pi)}

theorem set_equality : alpha_set = {-Real.pi / 5, -7 * Real.pi / 10, 3 * Real.pi / 10, 4 * Real.pi / 5} :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_set_equality_l686_68605


namespace NUMINAMATH_GPT_inequality_solution_l686_68690

theorem inequality_solution :
  { x : ℝ | (x-1)/(x+4) ≤ 0 } = { x : ℝ | (-4 < x ∧ x ≤ 0) ∨ (x = 1) } :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l686_68690


namespace NUMINAMATH_GPT_rain_at_least_once_prob_l686_68628

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end NUMINAMATH_GPT_rain_at_least_once_prob_l686_68628


namespace NUMINAMATH_GPT_find_c_l686_68611

-- Given conditions
variables {a b c d e : ℕ} (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e)
variables (h6 : a + b = e - 1) (h7 : a * b = d + 1)

-- Required to prove
theorem find_c : c = 4 := by
  sorry

end NUMINAMATH_GPT_find_c_l686_68611


namespace NUMINAMATH_GPT_joan_already_put_in_cups_l686_68668

def recipe_cups : ℕ := 7
def cups_needed : ℕ := 4

theorem joan_already_put_in_cups : (recipe_cups - cups_needed = 3) :=
by
  sorry

end NUMINAMATH_GPT_joan_already_put_in_cups_l686_68668


namespace NUMINAMATH_GPT_num_pairs_of_regular_polygons_l686_68658

def num_pairs : Nat := 
  let pairs := [(7, 42), (6, 18), (5, 10), (4, 6)]
  pairs.length

theorem num_pairs_of_regular_polygons : num_pairs = 4 := 
  sorry

end NUMINAMATH_GPT_num_pairs_of_regular_polygons_l686_68658


namespace NUMINAMATH_GPT_tunnel_length_l686_68625

noncomputable def train_speed_mph : ℝ := 75
noncomputable def train_length_miles : ℝ := 1 / 4
noncomputable def passing_time_minutes : ℝ := 3

theorem tunnel_length :
  let speed_mpm := train_speed_mph / 60
  let total_distance_traveled := speed_mpm * passing_time_minutes
  let tunnel_length := total_distance_traveled - train_length_miles
  tunnel_length = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_tunnel_length_l686_68625


namespace NUMINAMATH_GPT_album_cost_l686_68616

-- Definition of the cost variables
variable (B C A : ℝ)

-- Conditions given in the problem
axiom h1 : B = C + 4
axiom h2 : B = 18
axiom h3 : C = 0.70 * A

-- Theorem to prove the cost of the album
theorem album_cost : A = 20 := sorry

end NUMINAMATH_GPT_album_cost_l686_68616


namespace NUMINAMATH_GPT_range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l686_68621

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

def p (m : ℝ) : Prop :=
  ∀ x ∈ (Set.Ioo m (m + 1)), (x - 9 / x) < 0

def q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3

theorem range_of_m_when_p_true :
  ∀ m : ℝ, p m → 0 ≤ m ∧ m ≤ 2 :=
sorry

theorem range_of_m_when_p_and_q_false_p_or_q_true :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (0 ≤ m ∧ m ≤ 1) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_GPT_range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l686_68621


namespace NUMINAMATH_GPT_min_value_x2_minus_x1_l686_68608

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_value_x2_minus_x1 :
  (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) → |x2 - x1| = 2 :=
sorry

end NUMINAMATH_GPT_min_value_x2_minus_x1_l686_68608


namespace NUMINAMATH_GPT_number_of_people_l686_68684

-- Definitions based on the conditions
def total_cookies : ℕ := 420
def cookies_per_person : ℕ := 30

-- The goal is to prove the number of people is 14
theorem number_of_people : total_cookies / cookies_per_person = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_l686_68684


namespace NUMINAMATH_GPT_color_plane_no_unit_equilateral_same_color_l686_68681

theorem color_plane_no_unit_equilateral_same_color :
  ∃ (coloring : ℝ × ℝ → ℕ), (∀ (A B C : ℝ × ℝ),
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    (coloring A ≠ coloring B ∨ coloring B ≠ coloring C ∨ coloring C ≠ coloring A)) :=
sorry

end NUMINAMATH_GPT_color_plane_no_unit_equilateral_same_color_l686_68681


namespace NUMINAMATH_GPT_yellow_balls_in_bag_l686_68670

theorem yellow_balls_in_bag (x : ℕ) (prob : 1 / (1 + x) = 1 / 4) :
  x = 3 :=
sorry

end NUMINAMATH_GPT_yellow_balls_in_bag_l686_68670


namespace NUMINAMATH_GPT_cans_per_person_day1_l686_68692

theorem cans_per_person_day1
  (initial_cans : ℕ)
  (people_day1 : ℕ)
  (restock_day1 : ℕ)
  (people_day2 : ℕ)
  (cans_per_person_day2 : ℕ)
  (total_cans_given_away : ℕ) :
  initial_cans = 2000 →
  people_day1 = 500 →
  restock_day1 = 1500 →
  people_day2 = 1000 →
  cans_per_person_day2 = 2 →
  total_cans_given_away = 2500 →
  (total_cans_given_away - (people_day2 * cans_per_person_day2)) / people_day1 = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- condition trivially holds
  sorry

end NUMINAMATH_GPT_cans_per_person_day1_l686_68692


namespace NUMINAMATH_GPT_ratio_of_width_to_length_is_correct_l686_68675

-- Define the given conditions
def length := 10
def perimeter := 36

-- Define the width and the expected ratio
def width (l P : Nat) : Nat := (P - 2 * l) / 2
def ratio (w l : Nat) := w / l

-- Statement to prove that given the conditions, the ratio of width to length is 4/5
theorem ratio_of_width_to_length_is_correct :
  ratio (width length perimeter) length = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_width_to_length_is_correct_l686_68675


namespace NUMINAMATH_GPT_ellipse_product_axes_l686_68652

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end NUMINAMATH_GPT_ellipse_product_axes_l686_68652


namespace NUMINAMATH_GPT_percentage_of_x_l686_68624

variable (x : ℝ)

theorem percentage_of_x (x : ℝ) : ((40 / 100) * (50 / 100) * x) = (20 / 100) * x := by
  sorry

end NUMINAMATH_GPT_percentage_of_x_l686_68624


namespace NUMINAMATH_GPT_vector_relation_AD_l686_68656

variables {P V : Type} [AddCommGroup V] [Module ℝ V]
variables (A B C D : P) (AB AC AD BC BD CD : V)
variables (hBC_CD : BC = 3 • CD)

theorem vector_relation_AD (h1 : BC = 3 • CD)
                           (h2 : AD = AB + BD)
                           (h3 : BD = BC + CD)
                           (h4 : BC = -AB + AC) :
  AD = - (1 / 3 : ℝ) • AB + (4 / 3 : ℝ) • AC :=
by
  sorry

end NUMINAMATH_GPT_vector_relation_AD_l686_68656


namespace NUMINAMATH_GPT_total_alligators_seen_l686_68606

-- Definitions for the conditions
def SamaraSaw : Nat := 35
def NumberOfFriends : Nat := 6
def AverageFriendsSaw : Nat := 15

-- Statement of the proof problem
theorem total_alligators_seen :
  SamaraSaw + NumberOfFriends * AverageFriendsSaw = 125 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_total_alligators_seen_l686_68606
