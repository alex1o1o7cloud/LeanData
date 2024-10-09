import Mathlib

namespace mode_and_median_of_data_set_l118_11893

def data_set : List ℕ := [3, 5, 4, 6, 3, 3, 4]

noncomputable def mode_of_data_set : ℕ :=
  sorry  -- The mode calculation goes here (implementation is skipped)

noncomputable def median_of_data_set : ℕ :=
  sorry  -- The median calculation goes here (implementation is skipped)

theorem mode_and_median_of_data_set :
  mode_of_data_set = 3 ∧ median_of_data_set = 4 :=
  by
    sorry  -- Proof goes here

end mode_and_median_of_data_set_l118_11893


namespace power_function_increasing_iff_l118_11806

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_increasing_iff (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → power_function a x1 < power_function a x2) ↔ a > 0 := 
by
  sorry

end power_function_increasing_iff_l118_11806


namespace gcd_a_b_l118_11894

def a := 130^2 + 250^2 + 360^2
def b := 129^2 + 249^2 + 361^2

theorem gcd_a_b : Int.gcd a b = 1 := 
by
  sorry

end gcd_a_b_l118_11894


namespace sandy_saved_percentage_last_year_l118_11890

noncomputable def sandys_saved_percentage (S : ℝ) (P : ℝ) : ℝ :=
  (P / 100) * S

noncomputable def salary_with_10_percent_more (S : ℝ) : ℝ :=
  1.1 * S

noncomputable def amount_saved_this_year (S : ℝ) : ℝ :=
  0.15 * (salary_with_10_percent_more S)

noncomputable def amount_saved_this_year_compare_last_year (S : ℝ) (P : ℝ) : Prop :=
  amount_saved_this_year S = 1.65 * sandys_saved_percentage S P

theorem sandy_saved_percentage_last_year (S : ℝ) (P : ℝ) :
  amount_saved_this_year_compare_last_year S P → P = 10 :=
by
  sorry

end sandy_saved_percentage_last_year_l118_11890


namespace binary_operation_l118_11896

theorem binary_operation : 
  let a := 0b11011
  let b := 0b1101
  let c := 0b1010
  let result := 0b110011101  
  ((a * b) - c) = result := by
  sorry

end binary_operation_l118_11896


namespace alcohol_solution_volume_l118_11864

theorem alcohol_solution_volume (V : ℝ) (h1 : 0.42 * V = 0.33 * (V + 3)) : V = 11 :=
by
  sorry

end alcohol_solution_volume_l118_11864


namespace remainder_of_7n_div_4_l118_11888

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l118_11888


namespace china_nhsm_league_2021_zhejiang_p15_l118_11887

variable (x y z : ℝ)

theorem china_nhsm_league_2021_zhejiang_p15 (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x ^ 4 + y ^ 2 * z ^ 2) / (x ^ (5 / 2) * (y + z)) + 
  (y ^ 4 + z ^ 2 * x ^ 2) / (y ^ (5 / 2) * (z + x)) + 
  (z ^ 4 + y ^ 2 * x ^ 2) / (z ^ (5 / 2) * (y + x)) ≥ 1 := 
sorry

end china_nhsm_league_2021_zhejiang_p15_l118_11887


namespace sum_of_three_numbers_l118_11800

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l118_11800


namespace factor_expression_l118_11847

theorem factor_expression (y : ℝ) : 
  5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) :=
by
  sorry

end factor_expression_l118_11847


namespace framing_required_l118_11831

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end framing_required_l118_11831


namespace remainder_when_divided_by_9_l118_11810

theorem remainder_when_divided_by_9 (x : ℕ) (h : 4 * x % 9 = 2) : x % 9 = 5 :=
by sorry

end remainder_when_divided_by_9_l118_11810


namespace line_bisects_circle_perpendicular_l118_11837

theorem line_bisects_circle_perpendicular :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, x^2 + y^2 + x - 2*y + 1 = 0 → l x = y)
               ∧ (∀ x y : ℝ, x + 2*y + 3 = 0 → x ∈ { x | ∃ k:ℝ, y = -1/2 * k + l x})
               ∧ (∀ x y : ℝ, l x = 2 * x - 2)) :=
sorry

end line_bisects_circle_perpendicular_l118_11837


namespace max_value_m_n_squared_sum_l118_11855

theorem max_value_m_n_squared_sum (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m * n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_value_m_n_squared_sum_l118_11855


namespace area_of_original_rectangle_l118_11842

theorem area_of_original_rectangle 
  (L W : ℝ)
  (h1 : 2 * L * (3 * W) = 1800) :
  L * W = 300 :=
by
  sorry

end area_of_original_rectangle_l118_11842


namespace fraction_eq_l118_11820

theorem fraction_eq (x : ℝ) (h1 : x * 180 = 24) (h2 : x < 20 / 100) : x = 2 / 15 :=
sorry

end fraction_eq_l118_11820


namespace probability_event_occurs_l118_11832

def in_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

def event_occurs (x : ℝ) : Prop :=
  Real.cos (x + Real.pi / 3) + Real.sqrt 3 * Real.sin (x + Real.pi / 3) ≥ 1

theorem probability_event_occurs : 
  (∀ x, in_interval x → event_occurs x) → 
  (∃ p, p = 1/3) :=
by
  intros h
  sorry

end probability_event_occurs_l118_11832


namespace sum_congruence_example_l118_11812

theorem sum_congruence_example (a b c : ℤ) (h1 : a % 15 = 7) (h2 : b % 15 = 3) (h3 : c % 15 = 9) : 
  (a + b + c) % 15 = 4 :=
by 
  sorry

end sum_congruence_example_l118_11812


namespace supplementary_angle_proof_l118_11852

noncomputable def complementary_angle (α : ℝ) : ℝ := 125 + 12 / 60

noncomputable def calculate_angle (c : ℝ) := 180 - c

noncomputable def supplementary_angle (α : ℝ) := 90 - α

theorem supplementary_angle_proof :
    let α := calculate_angle (complementary_angle α)
    supplementary_angle α = 35 + 12 / 60 := 
by
  sorry

end supplementary_angle_proof_l118_11852


namespace no_real_roots_of_x_squared_plus_5_l118_11817

theorem no_real_roots_of_x_squared_plus_5 : ¬ ∃ (x : ℝ), x^2 + 5 = 0 :=
by
  sorry

end no_real_roots_of_x_squared_plus_5_l118_11817


namespace lemango_eating_mangos_l118_11841

theorem lemango_eating_mangos :
  ∃ (mangos_eaten : ℕ → ℕ), 
    (mangos_eaten 1 * (2^6 - 1) = 364 * (2 - 1)) ∧
    (mangos_eaten 6 = 128) :=
by
  sorry

end lemango_eating_mangos_l118_11841


namespace fathers_age_l118_11856

variable (S F : ℕ)
variable (h1 : F = 3 * S)
variable (h2 : F + 15 = 2 * (S + 15))

theorem fathers_age : F = 45 :=
by
  -- the proof steps would go here
  sorry

end fathers_age_l118_11856


namespace quotient_calculation_l118_11846

theorem quotient_calculation
  (dividend : ℕ)
  (divisor : ℕ)
  (remainder : ℕ)
  (h_dividend : dividend = 176)
  (h_divisor : divisor = 14)
  (h_remainder : remainder = 8) :
  ∃ q, dividend = divisor * q + remainder ∧ q = 12 :=
by
  sorry

end quotient_calculation_l118_11846


namespace color_tv_cost_l118_11878

theorem color_tv_cost (x : ℝ) (y : ℝ) (z : ℝ)
  (h1 : y = x * 1.4)
  (h2 : z = y * 0.8)
  (h3 : z = 360 + x) :
  x = 3000 :=
sorry

end color_tv_cost_l118_11878


namespace a8_eq_64_l118_11897

variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

axiom a1_eq_2 : a 1 = 2
axiom S_recurrence : ∀ (n : ℕ), S (n + 1) = 2 * S n - 1

theorem a8_eq_64 : a 8 = 64 := 
by
sorry

end a8_eq_64_l118_11897


namespace triangle_isosceles_or_right_l118_11867

theorem triangle_isosceles_or_right (a b c : ℝ) (A B C : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (triangle_abc : A + B + C = 180)
  (opposite_sides : ∀ {x y}, x ≠ y → x + y < 180) 
  (condition : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = 90) :=
by {
  sorry
}

end triangle_isosceles_or_right_l118_11867


namespace number_of_boys_l118_11877

theorem number_of_boys 
    (B : ℕ) 
    (total_boys_sticks : ℕ := 15 * B)
    (total_girls_sticks : ℕ := 12 * 12)
    (sticks_relation : total_girls_sticks = total_boys_sticks - 6) : 
    B = 10 :=
by
    sorry

end number_of_boys_l118_11877


namespace part1_part2_l118_11859

variable {x m : ℝ}

def P (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def S (x : ℝ) (m : ℝ) : Prop := -m + 1 ≤ x ∧ x ≤ m + 1

theorem part1 (h : ∀ x, P x → P x ∨ S x m) : m ≤ 0 :=
sorry

theorem part2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (P x ↔ S x m) :=
sorry

end part1_part2_l118_11859


namespace rhombus_diagonal_difference_l118_11891

theorem rhombus_diagonal_difference (a d : ℝ) (h_a_pos : a > 0) (h_d_pos : d > 0):
  (∃ (e f : ℝ), e > f ∧ e - f = d ∧ a^2 = (e/2)^2 + (f/2)^2) ↔ d < 2 * a :=
sorry

end rhombus_diagonal_difference_l118_11891


namespace tarun_garden_area_l118_11811

theorem tarun_garden_area :
  ∀ (side : ℝ), 
  (1500 / 8 = 4 * side) → 
  (30 * side = 1500) → 
  side^2 = 2197.265625 :=
by
  sorry

end tarun_garden_area_l118_11811


namespace polar_to_cartesian_l118_11885

theorem polar_to_cartesian (θ : ℝ) (ρ : ℝ) (x y : ℝ) :
  (ρ = 2 * Real.sin θ + 4 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - 8)^2 + (y - 2)^2 = 68 :=
by
  intros hρ hx hy
  -- Proof steps would go here
  sorry

end polar_to_cartesian_l118_11885


namespace range_of_a_l118_11813

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x < 2 → (x + a < 0))) → (a ≤ -2) :=
sorry

end range_of_a_l118_11813


namespace triangle_area_correct_l118_11872

open Real

def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)

theorem triangle_area_correct :
  triangle_area (4, 6) (-4, 6) (0, 2) = 16 :=
by
  sorry

end triangle_area_correct_l118_11872


namespace inflation_over_two_years_real_interest_rate_l118_11876

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l118_11876


namespace triangle_is_isosceles_l118_11853

open Real

-- Define the basic setup of the triangle and the variables involved
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to A, B, and C respectively
variables (h1 : a * cos B = b * cos A) -- Given condition: a * cos B = b * cos A

-- The theorem stating that the given condition implies the triangle is isosceles
theorem triangle_is_isosceles (h1 : a * cos B = b * cos A) : A = B :=
sorry

end triangle_is_isosceles_l118_11853


namespace min_sum_x_y_condition_l118_11805

theorem min_sum_x_y_condition {x y : ℝ} (h₁ : x > 0) (h₂ : y > 0) (h₃ : 1 / x + 9 / y = 1) : x + y = 16 :=
by
  sorry -- proof skipped

end min_sum_x_y_condition_l118_11805


namespace total_weight_of_ripe_apples_is_1200_l118_11825

def total_apples : Nat := 14
def weight_ripe_apple : Nat := 150
def weight_unripe_apple : Nat := 120
def unripe_apples : Nat := 6
def ripe_apples : Nat := total_apples - unripe_apples
def total_weight_ripe_apples : Nat := ripe_apples * weight_ripe_apple

theorem total_weight_of_ripe_apples_is_1200 :
  total_weight_ripe_apples = 1200 := by
  sorry

end total_weight_of_ripe_apples_is_1200_l118_11825


namespace minimum_value_x_l118_11860

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end minimum_value_x_l118_11860


namespace proof_neg_q_l118_11819

variable (f : ℝ → ℝ)
variable (x : ℝ)

def proposition_p (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

def proposition_q : Prop := ∃ x : ℝ, (deriv fun y => 1 / y) x > 0

theorem proof_neg_q : ¬ proposition_q := 
by
  intro h
  -- proof omitted for brevity
  sorry

end proof_neg_q_l118_11819


namespace cos_angle_difference_l118_11839

theorem cos_angle_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1): 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_angle_difference_l118_11839


namespace shaded_region_area_correct_l118_11892

noncomputable def hexagon_side : ℝ := 4
noncomputable def major_axis : ℝ := 4
noncomputable def minor_axis : ℝ := 2

noncomputable def hexagon_area := (3 * Real.sqrt 3 / 2) * hexagon_side^2

noncomputable def semi_ellipse_area : ℝ :=
  (1 / 2) * Real.pi * major_axis * minor_axis

noncomputable def total_semi_ellipse_area := 4 * semi_ellipse_area 

noncomputable def shaded_region_area := hexagon_area - total_semi_ellipse_area

theorem shaded_region_area_correct : shaded_region_area = 48 * Real.sqrt 3 - 16 * Real.pi :=
by
  sorry

end shaded_region_area_correct_l118_11892


namespace pyramid_z_value_l118_11886

-- Define the conditions and the proof problem
theorem pyramid_z_value {z x y : ℕ} :
  (x = z * y) →
  (8 = z * x) →
  (40 = x * y) →
  (10 = y * x) →
  z = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end pyramid_z_value_l118_11886


namespace calculate_expression_l118_11804

theorem calculate_expression : 3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := 
by sorry

end calculate_expression_l118_11804


namespace partI_solution_partII_solution_l118_11865

-- Part (I)
theorem partI_solution (x : ℝ) (a : ℝ) (h : a = 5) : (|x + a| + |x - 2| > 9) ↔ (x < -6 ∨ x > 3) :=
by
  sorry

-- Part (II)
theorem partII_solution (a : ℝ) :
  (∀ x : ℝ, (|2*x - 1| ≤ 3) → (|x + a| + |x - 2| ≤ |x - 4|)) → (-1 ≤ a ∧ a ≤ 0) :=
by
  sorry

end partI_solution_partII_solution_l118_11865


namespace part1_solution_part2_solution_l118_11851

-- Define the inequality for part (1)
def ineq_part1 (x : ℝ) : Prop := 1 - (4 / (x + 1)) < 0

-- Define the solution set P for part (1)
def P (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Prove that the solution set for the inequality is P
theorem part1_solution :
  ∀ (x : ℝ), ineq_part1 x ↔ P x :=
by
  -- proof omitted
  sorry

-- Define the inequality for part (2)
def ineq_part2 (x : ℝ) : Prop := abs (x + 2) < 3

-- Define the solution set Q for part (2)
def Q (x : ℝ) : Prop := -5 < x ∧ x < 1

-- Define P as depending on some parameter a
def P_param (a : ℝ) (x : ℝ) : Prop := -1 < x ∧ x < a

-- Prove the range of a given P ∪ Q = Q 
theorem part2_solution :
  ∀ a : ℝ, (∀ x : ℝ, (P_param a x ∨ Q x) ↔ Q x) → 
    (0 < a ∧ a ≤ 1) :=
by
  -- proof omitted
  sorry

end part1_solution_part2_solution_l118_11851


namespace reciprocal_of_neg_one_seventh_l118_11863

theorem reciprocal_of_neg_one_seventh :
  (∃ x : ℚ, - (1 / 7) * x = 1) → (-7) * (- (1 / 7)) = 1 :=
by
  sorry

end reciprocal_of_neg_one_seventh_l118_11863


namespace range_of_m_l118_11895

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m^2 > 0) ↔ -1 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l118_11895


namespace geometric_sequence_sum_8_l118_11829

variable {a : ℝ} 

-- conditions
def geometric_series_sum_4 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3

def geometric_series_sum_8 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 + a * r^6 + a * r^7

theorem geometric_sequence_sum_8 (r : ℝ) (S4 : ℝ) (S8 : ℝ) (hr : r = 2) (hS4 : S4 = 1) :
  (∃ a : ℝ, geometric_series_sum_4 r a = S4 ∧ geometric_series_sum_8 r a = S8) → S8 = 17 :=
by
  sorry

end geometric_sequence_sum_8_l118_11829


namespace chicken_price_per_pound_l118_11818

theorem chicken_price_per_pound (beef_pounds chicken_pounds : ℕ) (beef_price chicken_price : ℕ)
    (total_amount : ℕ)
    (h_beef_quantity : beef_pounds = 1000)
    (h_beef_cost : beef_price = 8)
    (h_chicken_quantity : chicken_pounds = 2 * beef_pounds)
    (h_total_price : 1000 * beef_price + chicken_pounds * chicken_price = total_amount)
    (h_total_amount : total_amount = 14000) : chicken_price = 3 :=
by
  sorry

end chicken_price_per_pound_l118_11818


namespace f_zero_is_one_l118_11801

def f (n : ℕ) : ℕ := sorry

theorem f_zero_is_one (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (f n) + f n = 2 * n + 3)
  (h2 : f 2015 = 2016) : f 0 = 1 := 
by {
  -- proof not required
  sorry
}

end f_zero_is_one_l118_11801


namespace fraction_greater_than_decimal_l118_11827

theorem fraction_greater_than_decimal :
  (1 / 4 : ℝ) > (24999999 / (10^8 : ℝ)) + (1 / (4 * (10^8 : ℝ))) :=
by
  sorry

end fraction_greater_than_decimal_l118_11827


namespace number_of_men_in_group_l118_11850

-- Define the conditions
variable (n : ℕ) -- number of men in the group
variable (A : ℝ) -- original average age of the group
variable (increase_in_years : ℝ := 2) -- the increase in the average age
variable (ages_before_replacement : ℝ := 21 + 23) -- total age of the men replaced
variable (ages_after_replacement : ℝ := 2 * 37) -- total age of the new men

-- Define the theorem using the conditions
theorem number_of_men_in_group 
  (h1 : n * increase_in_years = ages_after_replacement - ages_before_replacement) :
  n = 15 :=
sorry

end number_of_men_in_group_l118_11850


namespace total_cost_kept_l118_11881

def prices_all : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def prices_returned : List ℕ := [20, 25, 30, 22, 23, 29]

def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (· + ·) 0

theorem total_cost_kept :
  total_cost prices_all - total_cost prices_returned = 85 :=
by
  -- The proof steps go here
  sorry

end total_cost_kept_l118_11881


namespace probability_non_adjacent_l118_11836

def total_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n m 

def non_adjacent_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n (m - 1)

def probability_zeros_non_adjacent (n m : ℕ) : ℚ :=
  (non_adjacent_arrangements n m : ℚ) / (total_arrangements n m : ℚ)

theorem probability_non_adjacent (a b : ℕ) (h₁ : a = 4) (h₂ : b = 2) :
  probability_zeros_non_adjacent 5 2 = 2 / 3 := 
by 
  rw [probability_zeros_non_adjacent]
  rw [non_adjacent_arrangements, total_arrangements]
  sorry

end probability_non_adjacent_l118_11836


namespace david_bike_distance_l118_11857

noncomputable def david_time_hours : ℝ := 2 + 1 / 3
noncomputable def david_speed_mph : ℝ := 6.998571428571427
noncomputable def david_distance : ℝ := 16.33

theorem david_bike_distance :
  david_speed_mph * david_time_hours = david_distance :=
by
  sorry

end david_bike_distance_l118_11857


namespace common_pts_above_curve_l118_11848

open Real

theorem common_pts_above_curve {x y t : ℝ} (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : 0 ≤ y ∧ y ≤ 1) (h3 : 0 < t ∧ t < 1) :
  (∀ t, y ≥ (t-1)/t * x + 1 - t) ↔ (sqrt x + sqrt y ≥ 1) := 
by
  sorry

end common_pts_above_curve_l118_11848


namespace abs_neg_one_third_l118_11871

theorem abs_neg_one_third : abs (- (1 / 3 : ℚ)) = 1 / 3 := 
by sorry

end abs_neg_one_third_l118_11871


namespace geometric_body_is_cylinder_l118_11840

def top_view_is_circle : Prop := sorry

def is_prism_or_cylinder : Prop := sorry

theorem geometric_body_is_cylinder 
  (h1 : top_view_is_circle) 
  (h2 : is_prism_or_cylinder) 
  : Cylinder := 
sorry

end geometric_body_is_cylinder_l118_11840


namespace line_tangent_constant_sum_l118_11838

noncomputable def parabolaEquation (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

noncomputable def isTangent (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  l x = y ∧ ((x - 2) ^ 2 + y ^ 2 = 4)

theorem line_tangent_constant_sum (l : ℝ → ℝ) (A B P : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabolaEquation x₁ y₁ →
  parabolaEquation x₂ y₂ →
  isTangent l (4 / 5) (8 / 5) →
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  let F := (1, 0)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := (Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))
  (distance F A) + (distance F B) - (distance A B) = 2 :=
sorry

end line_tangent_constant_sum_l118_11838


namespace age_ratio_in_two_years_l118_11843

-- Definitions of conditions
def son_present_age : ℕ := 26
def age_difference : ℕ := 28
def man_present_age : ℕ := son_present_age + age_difference

-- Future ages after 2 years
def son_future_age : ℕ := son_present_age + 2
def man_future_age : ℕ := man_present_age + 2

-- The theorem to prove
theorem age_ratio_in_two_years : (man_future_age / son_future_age) = 2 := 
by
  -- Step-by-Step proof would go here
  sorry

end age_ratio_in_two_years_l118_11843


namespace solve_inequality_l118_11826

theorem solve_inequality (x : ℝ) :
  (x - 2) / (x + 5) ≤ 1 / 2 ↔ x ∈ Set.Ioc (-5 : ℝ) 9 :=
by
  sorry

end solve_inequality_l118_11826


namespace probability_of_both_white_l118_11874

namespace UrnProblem

-- Define the conditions
def firstUrnWhiteBalls : ℕ := 4
def firstUrnTotalBalls : ℕ := 10
def secondUrnWhiteBalls : ℕ := 7
def secondUrnTotalBalls : ℕ := 12

-- Define the probabilities of drawing a white ball from each urn
def P_A1 : ℚ := firstUrnWhiteBalls / firstUrnTotalBalls
def P_A2 : ℚ := secondUrnWhiteBalls / secondUrnTotalBalls

-- Define the combined probability of both events occurring
def P_A1_and_A2 : ℚ := P_A1 * P_A2

-- Theorem statement that checks the combined probability
theorem probability_of_both_white : P_A1_and_A2 = 7 / 30 := by
  sorry

end UrnProblem

end probability_of_both_white_l118_11874


namespace equivalent_solution_eq1_eqC_l118_11844

-- Define the given equation
def eq1 (x y : ℝ) : Prop := 4 * x - 8 * y - 5 = 0

-- Define the candidate equations
def eqA (x y : ℝ) : Prop := 8 * x - 8 * y - 10 = 0
def eqB (x y : ℝ) : Prop := 8 * x - 16 * y - 5 = 0
def eqC (x y : ℝ) : Prop := 8 * x - 16 * y - 10 = 0
def eqD (x y : ℝ) : Prop := 12 * x - 24 * y - 10 = 0

-- The theorem that we need to prove
theorem equivalent_solution_eq1_eqC : ∀ x y, eq1 x y ↔ eqC x y :=
by
  sorry

end equivalent_solution_eq1_eqC_l118_11844


namespace domain_of_tan_l118_11845

noncomputable def is_excluded_from_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 1 + 6 * k

theorem domain_of_tan {x : ℝ} :
  ∀ x, ¬ is_excluded_from_domain x ↔ ¬ ∃ k : ℤ, x = 1 + 6 * k := 
by 
  sorry

end domain_of_tan_l118_11845


namespace problem1_problem2_l118_11821

theorem problem1 : (Real.sqrt 2) * (Real.sqrt 6) + (Real.sqrt 3) = 3 * (Real.sqrt 3) :=
  sorry

theorem problem2 : (1 - Real.sqrt 2) * (2 - Real.sqrt 2) = 4 - 3 * (Real.sqrt 2) :=
  sorry

end problem1_problem2_l118_11821


namespace total_brushing_time_in_hours_l118_11873

-- Define the conditions as Lean definitions
def brushing_duration : ℕ := 2   -- 2 minutes per brushing session
def brushing_times_per_day : ℕ := 3  -- brushes 3 times a day
def days : ℕ := 30  -- for 30 days

-- Define the calculation of total brushing time in hours
theorem total_brushing_time_in_hours : (brushing_duration * brushing_times_per_day * days) / 60 = 3 := 
by 
  -- Sorry to skip the proof
  sorry

end total_brushing_time_in_hours_l118_11873


namespace simple_interest_rate_l118_11830

theorem simple_interest_rate (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : SI = P / 5)
  (h2 : SI = P * R * T / 100)
  (h3 : T = 7) : 
  R = 20 / 7 :=
by 
  sorry

end simple_interest_rate_l118_11830


namespace range_of_a_l118_11884

theorem range_of_a
  (x0 : ℝ) (a : ℝ)
  (hx0 : x0 > 1)
  (hineq : (x0 + 1) * Real.log x0 < a * (x0 - 1)) :
  a > 2 :=
sorry

end range_of_a_l118_11884


namespace problem_conditions_l118_11816

theorem problem_conditions (a : ℕ → ℤ) :
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 →
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 :=
by sorry

end problem_conditions_l118_11816


namespace sqrt_sq_eq_abs_l118_11809

theorem sqrt_sq_eq_abs (a : ℝ) : Real.sqrt (a^2) = |a| :=
sorry

end sqrt_sq_eq_abs_l118_11809


namespace fraction_value_l118_11854

theorem fraction_value : (2020 / (20 * 20 : ℝ)) = 5.05 := by
  sorry

end fraction_value_l118_11854


namespace product_of_real_roots_l118_11849

theorem product_of_real_roots : 
  (∃ x y : ℝ, (x ^ Real.log x = Real.exp 1) ∧ (y ^ Real.log y = Real.exp 1) ∧ x ≠ y ∧ x * y = 1) :=
by
  sorry

end product_of_real_roots_l118_11849


namespace problem1_problem2_l118_11835

-- Problem 1: Calculation Proof
theorem problem1 : (3 - Real.pi)^0 - Real.sqrt 4 + 4 * Real.sin (Real.pi * 60 / 180) + |Real.sqrt 3 - 3| = 2 + Real.sqrt 3 :=
by
  sorry

-- Problem 2: Inequality Systems Proof
theorem problem2 (x : ℝ) :
  (5 * (x + 3) > 4 * x + 8) ∧ (x / 6 - 1 < (x - 2) / 3) → x > -2 :=
by
  sorry

end problem1_problem2_l118_11835


namespace scout_weekend_earnings_l118_11868

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l118_11868


namespace green_tea_price_decrease_l118_11880

def percentage_change (old_price new_price : ℚ) : ℚ :=
  ((new_price - old_price) / old_price) * 100

theorem green_tea_price_decrease
  (C : ℚ)
  (h1 : C > 0)
  (july_coffee_price : ℚ := 2 * C)
  (mixture_price : ℚ := 3.45)
  (july_green_tea_price : ℚ := 0.3)
  (old_green_tea_price : ℚ := C)
  (equal_mixture : ℚ := (1.5 * july_green_tea_price) + (1.5 * july_coffee_price)) :
  mixture_price = equal_mixture →
  percentage_change old_green_tea_price july_green_tea_price = -70 :=
by
  sorry

end green_tea_price_decrease_l118_11880


namespace cargo_per_truck_is_2_5_l118_11823

-- Define our instance conditions
variables (x : ℝ) (n : ℕ)

-- Conditions extracted from the problem
def truck_capacity_change : Prop :=
  55 ≤ x ∧ x ≤ 64 ∧
  (x = (x / n - 0.5) * (n + 4))

-- Objective based on these conditions
theorem cargo_per_truck_is_2_5 :
  truck_capacity_change x n → (x = 60) → (n + 4 = 24) → (x / 24 = 2.5) :=
by 
  sorry

end cargo_per_truck_is_2_5_l118_11823


namespace extremum_of_function_l118_11861

theorem extremum_of_function (k : ℝ) (h₀ : k ≠ 1) :
  (k > 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≤ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) ∧
  (k < 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≥ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) :=
by
  sorry

end extremum_of_function_l118_11861


namespace solution_set_of_inequality_l118_11875

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l118_11875


namespace cos_product_value_l118_11833

open Real

theorem cos_product_value (α : ℝ) (h : sin α = 1 / 3) : 
  cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
by
  sorry

end cos_product_value_l118_11833


namespace benny_birthday_money_l118_11828

-- Define conditions
def spent_on_gear : ℕ := 47
def left_over : ℕ := 32

-- Define the total amount Benny received
def total_money_received : ℕ := 79

-- Theorem statement
theorem benny_birthday_money (spent_on_gear : ℕ) (left_over : ℕ) : spent_on_gear + left_over = total_money_received :=
by
  sorry

end benny_birthday_money_l118_11828


namespace manufacturer_cost_price_l118_11802

theorem manufacturer_cost_price
    (C : ℝ)
    (h1 : C > 0)
    (h2 : 1.18 * 1.20 * 1.25 * C = 30.09) :
    |C - 17| < 0.01 :=
by
    sorry

end manufacturer_cost_price_l118_11802


namespace complex_expression_power_48_l118_11824

open Complex

noncomputable def complex_expression := (1 + I) / Real.sqrt 2

theorem complex_expression_power_48 : complex_expression ^ 48 = 1 := by
  sorry

end complex_expression_power_48_l118_11824


namespace evaluate_expression_equals_three_plus_sqrt_three_l118_11889

noncomputable def tan_sixty_squared_plus_one := Real.tan (60 * Real.pi / 180) ^ 2 + 1
noncomputable def tan_fortyfive_minus_twocos_thirty := Real.tan (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)
noncomputable def expression (x y : ℝ) : ℝ := (x - (2 * x * y - y ^ 2) / x) / ((x ^ 2 - y ^ 2) / (x ^ 2 + x * y))

theorem evaluate_expression_equals_three_plus_sqrt_three :
  expression tan_sixty_squared_plus_one tan_fortyfive_minus_twocos_thirty = 3 + Real.sqrt 3 :=
sorry

end evaluate_expression_equals_three_plus_sqrt_three_l118_11889


namespace print_time_l118_11869

theorem print_time (P R: ℕ) (hR : R = 24) (hP : P = 360) (T : ℕ) : T = P / R → T = 15 := by
  intros h
  rw [hR, hP] at h
  exact h

end print_time_l118_11869


namespace obtain_any_natural_from_4_l118_11834

/-- Definitions of allowed operations:
  - Append the digit 4.
  - Append the digit 0.
  - Divide by 2, if the number is even.
--/
def append4 (n : ℕ) : ℕ := 10 * n + 4
def append0 (n : ℕ) : ℕ := 10 * n
def divide2 (n : ℕ) : ℕ := n / 2

/-- We'll also define if a number is even --/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Define the set of operations applied on a number --/
inductive operations : ℕ → ℕ → Prop
| initial : operations 4 4
| append4_step (n m : ℕ) : operations n m → operations n (append4 m)
| append0_step (n m : ℕ) : operations n m → operations n (append0 m)
| divide2_step (n m : ℕ) : is_even m → operations n m → operations n (divide2 m)

/-- The main theorem proving that any natural number can be obtained from 4 using the allowed operations --/
theorem obtain_any_natural_from_4 (n : ℕ) : ∃ m, operations 4 m ∧ m = n :=
by sorry

end obtain_any_natural_from_4_l118_11834


namespace complex_exp_l118_11862

theorem complex_exp {i : ℂ} (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end complex_exp_l118_11862


namespace neg_q_necessary_not_sufficient_for_neg_p_l118_11899

-- Proposition p: |x + 2| > 2
def p (x : ℝ) : Prop := abs (x + 2) > 2

-- Proposition q: 1 / (3 - x) > 1
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Negation of p and q
def neg_p (x : ℝ) : Prop := -4 ≤ x ∧ x ≤ 0
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- Theorem: negation of q is a necessary but not sufficient condition for negation of p
theorem neg_q_necessary_not_sufficient_for_neg_p :
  (∀ x : ℝ, neg_p x → neg_q x) ∧ (∃ x : ℝ, neg_q x ∧ ¬neg_p x) :=
by
  sorry

end neg_q_necessary_not_sufficient_for_neg_p_l118_11899


namespace find_a_inverse_function_l118_11803

theorem find_a_inverse_function
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x y, y = f x ↔ x = a * y)
  (h2 : f 4 = 2) :
  a = 2 := 
sorry

end find_a_inverse_function_l118_11803


namespace factorize_difference_of_squares_l118_11808

theorem factorize_difference_of_squares (x : ℝ) :
  4 * x^2 - 1 = (2 * x + 1) * (2 * x - 1) :=
sorry

end factorize_difference_of_squares_l118_11808


namespace count_valid_abcd_is_zero_l118_11815

def valid_digits := {a // 1 ≤ a ∧ a ≤ 9} 
def zero_to_nine := {n // 0 ≤ n ∧ n ≤ 9}

noncomputable def increasing_arithmetic_sequence_with_difference_5 (a b c d : ℕ) : Prop := 
  10 * a + b + 5 = 10 * b + c ∧ 
  10 * b + c + 5 = 10 * c + d

theorem count_valid_abcd_is_zero :
  ∀ (a : valid_digits) (b c d : zero_to_nine),
    ¬ increasing_arithmetic_sequence_with_difference_5 a.val b.val c.val d.val := 
sorry

end count_valid_abcd_is_zero_l118_11815


namespace no_integer_roots_l118_11822

theorem no_integer_roots : ∀ x : ℤ, x^3 - 3 * x^2 - 16 * x + 20 ≠ 0 := by
  intro x
  sorry

end no_integer_roots_l118_11822


namespace sum_eq_twenty_x_l118_11898

variable {R : Type*} [CommRing R] (x y z : R)

theorem sum_eq_twenty_x (h1 : y = 3 * x) (h2 : z = 3 * y) : 2 * x + 3 * y + z = 20 * x := by
  sorry

end sum_eq_twenty_x_l118_11898


namespace rectangle_length_l118_11807

theorem rectangle_length :
  ∀ (side : ℕ) (width : ℕ) (length : ℕ), 
  side = 4 → 
  width = 8 → 
  side * side = width * length → 
  length = 2 := 
by
  -- sorry to skip the proof
  intros side width length h1 h2 h3
  sorry

end rectangle_length_l118_11807


namespace n_div_p_eq_27_l118_11882

theorem n_div_p_eq_27 (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0)
    (h4 : ∃ r1 r2 : ℝ, r1 * r2 = m ∧ r1 + r2 = -p ∧ (3 * r1) * (3 * r2) = n ∧ 3 * (r1 + r2) = -m)
    : n / p = 27 := sorry

end n_div_p_eq_27_l118_11882


namespace right_triangle_perimeter_l118_11879

noncomputable def perimeter_of_right_triangle (x : ℝ) : ℝ :=
  let y := x + 15
  let c := Real.sqrt (x^2 + y^2)
  x + y + c

theorem right_triangle_perimeter
  (h₁ : ∀ a b : ℝ, a * b = 2 * 150)  -- The area condition
  (h₂ : ∀ a b : ℝ, b = a + 15)       -- One leg is 15 units longer than the other
  : perimeter_of_right_triangle 11.375 = 66.47 :=
by
  sorry

end right_triangle_perimeter_l118_11879


namespace mary_regular_hours_l118_11814

theorem mary_regular_hours (x y : ℕ) :
  8 * x + 10 * y = 760 ∧ x + y = 80 → x = 20 :=
by
  intro h
  sorry

end mary_regular_hours_l118_11814


namespace no_five_coin_combination_for_70_cents_l118_11883

/-- Define the values of each coin type -/
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25

/-- Prove that it is not possible to achieve a total value of 70 cents with exactly five coins -/
theorem no_five_coin_combination_for_70_cents :
  ¬ ∃ a b c d e : ℕ, a + b + c + d + e = 5 ∧ a * penny + b * nickel + c * dime + d * quarter + e * quarter = 70 :=
sorry

end no_five_coin_combination_for_70_cents_l118_11883


namespace distinct_triangles_count_l118_11870

theorem distinct_triangles_count (n : ℕ) (hn : 0 < n) : 
  (∃ triangles_count, triangles_count = ⌊((n+1)^2 : ℝ)/4⌋) :=
sorry

end distinct_triangles_count_l118_11870


namespace factor_correct_l118_11858

def factor_expression (x : ℝ) : Prop :=
  x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3)

theorem factor_correct (x : ℝ) : factor_expression x :=
  by sorry

end factor_correct_l118_11858


namespace probability_p_eq_l118_11866

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l118_11866
