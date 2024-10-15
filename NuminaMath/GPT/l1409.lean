import Mathlib

namespace NUMINAMATH_GPT_minimum_degree_of_g_l1409_140969

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

theorem minimum_degree_of_g :
  (5 * f - 3 * g = h) →
  (Polynomial.degree f = 10) →
  (Polynomial.degree h = 11) →
  (Polynomial.degree g = 11) :=
sorry

end NUMINAMATH_GPT_minimum_degree_of_g_l1409_140969


namespace NUMINAMATH_GPT_find_largest_number_l1409_140982

-- Define what it means for a sequence of 4 numbers to be an arithmetic progression with a given common difference d
def is_arithmetic_progression (a b c d : ℝ) (diff : ℝ) : Prop := (b - a = diff) ∧ (c - b = diff) ∧ (d - c = diff)

-- Define what it means for a sequence of 4 numbers to be a geometric progression
def is_geometric_progression (a b c d : ℝ) : Prop := b / a = c / b ∧ c / b = d / c

-- Given conditions for the sequence of 8 increasing real numbers
def conditions (a : ℕ → ℝ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧
  ∃ i j k, is_arithmetic_progression (a i) (a (i+1)) (a (i+2)) (a (i+3)) 4 ∧
            is_arithmetic_progression (a j) (a (j+1)) (a (j+2)) (a (j+3)) 36 ∧
            is_geometric_progression (a k) (a (k+1)) (a (k+2)) (a (k+3))

-- Prove that under these conditions, the largest number in the sequence is 126
theorem find_largest_number (a : ℕ → ℝ) : conditions a → a 7 = 126 :=
by
  sorry

end NUMINAMATH_GPT_find_largest_number_l1409_140982


namespace NUMINAMATH_GPT_a4_value_l1409_140957

-- Definitions and helper theorems can go here
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- These are our conditions
axiom h1 : S 2 = a 1 + a 2
axiom h2 : a 2 = 3
axiom h3 : ∀ n, S (n + 1) = 2 * S n + 1

theorem a4_value : a 4 = 12 :=
sorry  -- proof to be filled in later

end NUMINAMATH_GPT_a4_value_l1409_140957


namespace NUMINAMATH_GPT_abs_inequality_proof_by_contradiction_l1409_140922

theorem abs_inequality_proof_by_contradiction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  |a| > |b| :=
by
  let h := |a| ≤ |b|
  sorry

end NUMINAMATH_GPT_abs_inequality_proof_by_contradiction_l1409_140922


namespace NUMINAMATH_GPT_range_of_m_l1409_140915

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ y = 0) ∧ 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ x = 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1409_140915


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1409_140911

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : Real.exp a + 2 * a = Real.exp b + 3 * b) : 
  a > b :=
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1409_140911


namespace NUMINAMATH_GPT_ellipse_problem_l1409_140952

theorem ellipse_problem
  (F2 : ℝ) (a : ℝ) (A B : ℝ × ℝ)
  (on_ellipse_A : (A.1 ^ 2) / (a ^ 2) + (25 * (A.2 ^ 2)) / (9 * a ^ 2) = 1)
  (on_ellipse_B : (B.1 ^ 2) / (a ^ 2) + (25 * (B.2 ^ 2)) / (9 * a ^ 2) = 1)
  (focal_distance : |A.1 + F2| + |B.1 + F2| = 8 / 5 * a)
  (midpoint_to_directrix : |(A.1 + B.1) / 2 + 5 / 4 * a| = 3 / 2) :
  a = 1 → (∀ x y, (x^2 + (25 / 9) * y^2 = 1) ↔ ((x^2) / (a^2) + (25 * y^2) / (9 * a^2) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_problem_l1409_140952


namespace NUMINAMATH_GPT_solve_rational_equation_l1409_140918

theorem solve_rational_equation (x : ℝ) (h : x ≠ (2/3)) : 
  (6*x + 4) / (3*x^2 + 6*x - 8) = 3*x / (3*x - 2) ↔ x = -4/3 ∨ x = 3 :=
sorry

end NUMINAMATH_GPT_solve_rational_equation_l1409_140918


namespace NUMINAMATH_GPT_greatest_possible_remainder_l1409_140930

theorem greatest_possible_remainder {x : ℤ} (h : ∃ (k : ℤ), x = 11 * k + 10) : 
  ∃ y, y = 10 := sorry

end NUMINAMATH_GPT_greatest_possible_remainder_l1409_140930


namespace NUMINAMATH_GPT_remainder_when_2n_divided_by_4_l1409_140912

theorem remainder_when_2n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_2n_divided_by_4_l1409_140912


namespace NUMINAMATH_GPT_find_y_l1409_140995

theorem find_y (y: ℕ) (h1: y > 0) (h2: y ≤ 100)
  (h3: (43 + 69 + 87 + y + y) / 5 = 2 * y): 
  y = 25 :=
sorry

end NUMINAMATH_GPT_find_y_l1409_140995


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1409_140914

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_ratio_ne_one : q ≠ 1)
  (S : ℕ → ℝ) (h_a1 : S 1 = 1) (h_S4_eq_5S2 : S 4 - 5 * S 2 = 0) :
  S 5 = 31 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1409_140914


namespace NUMINAMATH_GPT_arthur_speed_l1409_140996

/-- Suppose Arthur drives to David's house and aims to arrive exactly on time. 
If he drives at 60 km/h, he arrives 5 minutes late. 
If he drives at 90 km/h, he arrives 5 minutes early. 
We want to find the speed n in km/h at which he arrives exactly on time. -/
theorem arthur_speed (n : ℕ) :
  (∀ t, 1 * (t + 5) = (3 / 2) * (t - 5)) → 
  (60 : ℝ) = 1 →
  (90 : ℝ) = (3 / 2) → 
  n = 72 := by
sorry

end NUMINAMATH_GPT_arthur_speed_l1409_140996


namespace NUMINAMATH_GPT_book_pages_l1409_140961

theorem book_pages (P : ℝ) (h1 : 2/3 * P = 1/3 * P + 20) : P = 60 :=
by
  sorry

end NUMINAMATH_GPT_book_pages_l1409_140961


namespace NUMINAMATH_GPT_roots_distinct_and_real_l1409_140920

variables (b d : ℝ)
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_distinct_and_real (h₁ : discriminant b (-3 * Real.sqrt 5) d = 25) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by 
  sorry

end NUMINAMATH_GPT_roots_distinct_and_real_l1409_140920


namespace NUMINAMATH_GPT_ronaldo_current_age_l1409_140956

noncomputable def roonie_age_one_year_ago (R L : ℕ) := 6 * L / 7
noncomputable def new_ratio (R L : ℕ) := (R + 5) * 8 = 7 * (L + 5)

theorem ronaldo_current_age (R L : ℕ) 
  (h1 : R = roonie_age_one_year_ago R L)
  (h2 : new_ratio R L) : L + 1 = 36 :=
by
  sorry

end NUMINAMATH_GPT_ronaldo_current_age_l1409_140956


namespace NUMINAMATH_GPT_coprime_divisors_property_l1409_140973

theorem coprime_divisors_property (n : ℕ) 
  (h : ∀ a b : ℕ, a ∣ n → b ∣ n → gcd a b = 1 → (a + b - 1) ∣ n) : 
  (∃ k : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n = p ^ k) ∨ (n = 12) :=
sorry

end NUMINAMATH_GPT_coprime_divisors_property_l1409_140973


namespace NUMINAMATH_GPT_trapezoid_median_l1409_140935

noncomputable def median_trapezoid (base₁ base₂ height : ℝ) : ℝ :=
(base₁ + base₂) / 2

theorem trapezoid_median (b_t : ℝ) (a_t : ℝ) (h_t : ℝ) (a_tp : ℝ) 
  (h_eq : h_t = 16) (a_eq : a_t = 192) (area_tp_eq : a_tp = a_t) : median_trapezoid h_t h_t h_t = 12 :=
by
  have h_t_eq : h_t = 16 := by sorry
  have a_t_eq : a_t = 192 := by sorry
  have area_tp : a_tp = 192 := by sorry
  sorry

end NUMINAMATH_GPT_trapezoid_median_l1409_140935


namespace NUMINAMATH_GPT_a_and_b_work_together_l1409_140992
noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem a_and_b_work_together (A_days B_days : ℕ) (hA : A_days = 32) (hB : B_days = 32) :
  (1 / work_rate A_days + 1 / work_rate B_days) = 16 := by
  sorry

end NUMINAMATH_GPT_a_and_b_work_together_l1409_140992


namespace NUMINAMATH_GPT_find_g_inv_84_l1409_140928

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_g_inv_84_l1409_140928


namespace NUMINAMATH_GPT_gym_hours_tuesday_equals_friday_l1409_140962

-- Definitions
def weekly_gym_hours : ℝ := 5
def monday_hours : ℝ := 1.5
def wednesday_hours : ℝ := 1.5
def friday_hours : ℝ := 1
def total_weekly_hours : ℝ := weekly_gym_hours - (monday_hours + wednesday_hours + friday_hours)

-- Theorem statement
theorem gym_hours_tuesday_equals_friday : 
  total_weekly_hours = friday_hours :=
by
  sorry

end NUMINAMATH_GPT_gym_hours_tuesday_equals_friday_l1409_140962


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1409_140900

theorem sum_of_three_numbers :
  1.35 + 0.123 + 0.321 = 1.794 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1409_140900


namespace NUMINAMATH_GPT_probability_f_ge1_l1409_140933

noncomputable def f (x: ℝ) : ℝ := 3*x^2 - x - 1

def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def valid_intervals : Set ℝ := { x | -1 ≤ x ∧ x ≤ -2/3 } ∪ { x | 1 ≤ x ∧ x ≤ 2 }

def interval_length (a b : ℝ) : ℝ := b - a

theorem probability_f_ge1 : 
  (interval_length (-2/3) (-1) + interval_length 1 2) / interval_length (-1) 2 = 4 / 9 := 
by
  sorry

end NUMINAMATH_GPT_probability_f_ge1_l1409_140933


namespace NUMINAMATH_GPT_pirates_total_distance_l1409_140968

def adjusted_distance_1 (d: ℝ) : ℝ := d * 1.10
def adjusted_distance_2 (d: ℝ) : ℝ := d * 1.15
def adjusted_distance_3 (d: ℝ) : ℝ := d * 1.20
def adjusted_distance_4 (d: ℝ) : ℝ := d * 1.25

noncomputable def total_distance : ℝ := 
  let first_island := (adjusted_distance_1 10) + (adjusted_distance_1 15) + (adjusted_distance_1 20)
  let second_island := adjusted_distance_2 40
  let third_island := (adjusted_distance_3 25) + (adjusted_distance_3 20) + (adjusted_distance_3 25) + (adjusted_distance_3 20)
  let fourth_island := adjusted_distance_4 35
  first_island + second_island + third_island + fourth_island

theorem pirates_total_distance : total_distance = 247.25 := by
  sorry

end NUMINAMATH_GPT_pirates_total_distance_l1409_140968


namespace NUMINAMATH_GPT_factor_polynomial_equiv_l1409_140970

theorem factor_polynomial_equiv :
  (x^2 + 2 * x + 1) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 7 * x + 1) * (x^2 + 3 * x + 7) :=
by sorry

end NUMINAMATH_GPT_factor_polynomial_equiv_l1409_140970


namespace NUMINAMATH_GPT_inequality_condition_l1409_140974

theorem inequality_condition (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 :=
sorry

end NUMINAMATH_GPT_inequality_condition_l1409_140974


namespace NUMINAMATH_GPT_savings_calculation_l1409_140919

noncomputable def calculate_savings (spent_price : ℝ) (saving_pct : ℝ) : ℝ :=
  let original_price := spent_price / (1 - (saving_pct / 100))
  original_price - spent_price

-- Define the spent price and saving percentage
def spent_price : ℝ := 20
def saving_pct : ℝ := 12.087912087912088

-- Statement to be proved
theorem savings_calculation : calculate_savings spent_price saving_pct = 2.75 :=
  sorry

end NUMINAMATH_GPT_savings_calculation_l1409_140919


namespace NUMINAMATH_GPT_tan_A_minus_B_l1409_140905

theorem tan_A_minus_B (A B : ℝ) (h1: Real.cos A = -Real.sqrt 2 / 2) (h2 : Real.tan B = 1 / 3) : 
  Real.tan (A - B) = -2 := by
  sorry

end NUMINAMATH_GPT_tan_A_minus_B_l1409_140905


namespace NUMINAMATH_GPT_least_number_remainder_l1409_140958

noncomputable def lcm_12_15_20_54 : ℕ := 540

theorem least_number_remainder :
  ∀ (n r : ℕ), (n = lcm_12_15_20_54 + r) → 
  (n % 12 = r) ∧ (n % 15 = r) ∧ (n % 20 = r) ∧ (n % 54 = r) → 
  r = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_remainder_l1409_140958


namespace NUMINAMATH_GPT_sin_double_angle_value_l1409_140932

theorem sin_double_angle_value 
  (α : ℝ) 
  (hα1 : π / 2 < α) 
  (hα2 : α < π)
  (h : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = - 17 / 18 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_value_l1409_140932


namespace NUMINAMATH_GPT_jenna_bill_eel_ratio_l1409_140946

theorem jenna_bill_eel_ratio:
  ∀ (B : ℕ), (B + 16 = 64) → (16 / B = 1 / 3) :=
by
  intros B h
  sorry

end NUMINAMATH_GPT_jenna_bill_eel_ratio_l1409_140946


namespace NUMINAMATH_GPT_parametric_to_cartesian_l1409_140976

theorem parametric_to_cartesian (θ : ℝ) (x y : ℝ) :
  (x = 1 + 2 * Real.cos θ) →
  (y = 2 * Real.sin θ) →
  (x - 1) ^ 2 + y ^ 2 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_parametric_to_cartesian_l1409_140976


namespace NUMINAMATH_GPT_samantha_birth_year_l1409_140907

theorem samantha_birth_year :
  ∀ (first_amc : ℕ) (amc9_year : ℕ) (samantha_age_in_amc9 : ℕ),
  (first_amc = 1983) →
  (amc9_year = first_amc + 8) →
  (samantha_age_in_amc9 = 13) →
  (amc9_year - samantha_age_in_amc9 = 1978) :=
by
  intros first_amc amc9_year samantha_age_in_amc9 h1 h2 h3
  sorry

end NUMINAMATH_GPT_samantha_birth_year_l1409_140907


namespace NUMINAMATH_GPT_terminating_decimal_of_7_div_200_l1409_140926

theorem terminating_decimal_of_7_div_200 : (7 / 200 : ℝ) = 0.028 := sorry

end NUMINAMATH_GPT_terminating_decimal_of_7_div_200_l1409_140926


namespace NUMINAMATH_GPT_ruiz_original_salary_l1409_140942

theorem ruiz_original_salary (S : ℝ) (h : 1.06 * S = 530) : S = 500 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_ruiz_original_salary_l1409_140942


namespace NUMINAMATH_GPT_problem_l1409_140997

/-
A problem involving natural numbers a and b
where:
1. Their sum is 20000
2. One of them (b) is divisible by 5
3. Erasing the units digit of b gives the other number a

We want to prove their difference is 16358
-/

def nat_sum_and_difference (a b : ℕ) : Prop :=
  a + b = 20000 ∧
  b % 5 = 0 ∧
  (b % 10 = 0 ∧ b / 10 = a ∨ b % 10 = 5 ∧ (b - 5) / 10 = a)

theorem problem (a b : ℕ) (h : nat_sum_and_difference a b) : b - a = 16358 := 
  sorry

end NUMINAMATH_GPT_problem_l1409_140997


namespace NUMINAMATH_GPT_calcium_iodide_weight_l1409_140934

theorem calcium_iodide_weight
  (atomic_weight_Ca : ℝ)
  (atomic_weight_I : ℝ)
  (moles : ℝ) :
  atomic_weight_Ca = 40.08 →
  atomic_weight_I = 126.90 →
  moles = 5 →
  (atomic_weight_Ca + 2 * atomic_weight_I) * moles = 1469.4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calcium_iodide_weight_l1409_140934


namespace NUMINAMATH_GPT_rational_b_if_rational_a_l1409_140986

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_rational_b_if_rational_a_l1409_140986


namespace NUMINAMATH_GPT_austin_hours_on_mondays_l1409_140948

-- Define the conditions
def earning_per_hour : ℕ := 5
def hours_wednesday : ℕ := 1
def hours_friday : ℕ := 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

-- Define the proof problem
theorem austin_hours_on_mondays (M : ℕ) :
  earning_per_hour * weeks * (M + hours_wednesday + hours_friday) = bicycle_cost → M = 2 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_austin_hours_on_mondays_l1409_140948


namespace NUMINAMATH_GPT_fraction_not_integer_l1409_140966

theorem fraction_not_integer (a b : ℕ) (h : a ≠ b) (parity: (a % 2 = b % 2)) 
(h_pos_a : 0 < a) (h_pos_b : 0 < b) : ¬ ∃ k : ℕ, (a! + b!) = k * 2^a := 
by sorry

end NUMINAMATH_GPT_fraction_not_integer_l1409_140966


namespace NUMINAMATH_GPT_transform_equation_l1409_140927

theorem transform_equation (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x - 1)^2 = 3 :=
sorry

end NUMINAMATH_GPT_transform_equation_l1409_140927


namespace NUMINAMATH_GPT_male_salmon_count_l1409_140985

theorem male_salmon_count (total_count : ℕ) (female_count : ℕ) (male_count : ℕ) :
  total_count = 971639 →
  female_count = 259378 →
  male_count = (total_count - female_count) →
  male_count = 712261 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_male_salmon_count_l1409_140985


namespace NUMINAMATH_GPT_minimum_value_l1409_140998

variables (a b c d : ℝ)
-- Conditions
def condition1 := (b - 2 * a^2 + 3 * Real.log a)^2 = 0
def condition2 := (c - d - 3)^2 = 0

-- Theorem stating the goal
theorem minimum_value (h1 : condition1 a b) (h2 : condition2 c d) : 
  (a - c)^2 + (b - d)^2 = 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1409_140998


namespace NUMINAMATH_GPT_probability_not_red_light_l1409_140906

theorem probability_not_red_light :
  ∀ (red_light yellow_light green_light : ℕ),
    red_light = 30 →
    yellow_light = 5 →
    green_light = 40 →
    (yellow_light + green_light) / (red_light + yellow_light + green_light) = (3 : ℚ) / 5 :=
by intros red_light yellow_light green_light h_red h_yellow h_green
   sorry

end NUMINAMATH_GPT_probability_not_red_light_l1409_140906


namespace NUMINAMATH_GPT_unique_polynomial_P_l1409_140944

noncomputable def P : ℝ → ℝ := sorry

axiom P_func_eq (x : ℝ) : P (x^2 + 1) = P x ^ 2 + 1
axiom P_zero : P 0 = 0

theorem unique_polynomial_P (x : ℝ) : P x = x :=
by
  sorry

end NUMINAMATH_GPT_unique_polynomial_P_l1409_140944


namespace NUMINAMATH_GPT_bob_and_bill_same_class_probability_l1409_140990

-- Definitions based on the conditions mentioned in the original problem
def total_people : ℕ := 32
def allowed_per_class : ℕ := 30
def number_chosen : ℕ := 2
def number_of_classes : ℕ := 2
def bob_and_bill_pair : ℕ := 1

-- Binomial coefficient calculation (32 choose 2)
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k
def total_ways := binomial_coefficient total_people number_chosen

-- Probability that Bob and Bill are chosen
def probability_chosen : ℚ := bob_and_bill_pair / total_ways

-- Probability that Bob and Bill are placed in the same class
def probability_same_class : ℚ := 1 / number_of_classes

-- Total combined probability
def combined_probability : ℚ := probability_chosen * probability_same_class

-- Statement of the theorem
theorem bob_and_bill_same_class_probability :
  combined_probability = 1 / 992 := 
sorry

end NUMINAMATH_GPT_bob_and_bill_same_class_probability_l1409_140990


namespace NUMINAMATH_GPT_man_is_26_years_older_l1409_140947

variable (S : ℕ) (M : ℕ)

-- conditions
def present_age_of_son : Prop := S = 24
def future_age_relation : Prop := M + 2 = 2 * (S + 2)

-- question transformed to a proof problem
theorem man_is_26_years_older
  (h1 : present_age_of_son S)
  (h2 : future_age_relation S M) : M - S = 26 := by
  sorry

end NUMINAMATH_GPT_man_is_26_years_older_l1409_140947


namespace NUMINAMATH_GPT_length_of_first_platform_is_140_l1409_140980

-- Definitions based on problem conditions
def train_length : ℝ := 190
def time_first_platform : ℝ := 15
def time_second_platform : ℝ := 20
def length_second_platform : ℝ := 250

-- Definition for the length of the first platform (what we're proving)
def length_first_platform (L : ℝ) : Prop :=
  (time_first_platform * (train_length + L) = time_second_platform * (train_length + length_second_platform))

-- Theorem: The length of the first platform is 140 meters
theorem length_of_first_platform_is_140 : length_first_platform 140 :=
  by sorry

end NUMINAMATH_GPT_length_of_first_platform_is_140_l1409_140980


namespace NUMINAMATH_GPT_complement_of_beta_l1409_140963

variable (α β : ℝ)
variable (compl : α + β = 180)
variable (alpha_greater_beta : α > β)

theorem complement_of_beta (h : α + β = 180) (h' : α > β) : 90 - β = (1 / 2) * (α - β) :=
by
  sorry

end NUMINAMATH_GPT_complement_of_beta_l1409_140963


namespace NUMINAMATH_GPT_final_computation_l1409_140975

noncomputable def N := (15 ^ 10 / 15 ^ 9) ^ 3 * 5 ^ 3

theorem final_computation : (N / 3 ^ 3) = 15625 := 
by 
  sorry

end NUMINAMATH_GPT_final_computation_l1409_140975


namespace NUMINAMATH_GPT_ratio_of_segments_l1409_140940

theorem ratio_of_segments
  (x y z u v : ℝ)
  (h_triangle : x^2 + y^2 = z^2)
  (h_ratio_legs : 4 * x = 3 * y)
  (h_u : u = x^2 / z)
  (h_v : v = y^2 / z) :
  u / v = 9 / 16 := 
  sorry

end NUMINAMATH_GPT_ratio_of_segments_l1409_140940


namespace NUMINAMATH_GPT_intersection_of_S_and_T_l1409_140965

def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | 0 < x}

theorem intersection_of_S_and_T : S ∩ T = {x | 1 ≤ x} := by
  sorry

end NUMINAMATH_GPT_intersection_of_S_and_T_l1409_140965


namespace NUMINAMATH_GPT_net_sag_calculation_l1409_140988

open Real

noncomputable def sag_of_net (m1 m2 h1 h2 x1 : ℝ) : ℝ :=
  let g := 9.81
  let a := 28
  let b := -1.75
  let c := -50.75
  let D := b^2 - 4*a*c
  let sqrtD := sqrt D
  (1.75 + sqrtD) / (2 * a)

theorem net_sag_calculation :
  let m1 := 78.75
  let x1 := 1
  let h1 := 15
  let m2 := 45
  let h2 := 29
  sag_of_net m1 m2 h1 h2 x1 = 1.38 := 
by
  sorry

end NUMINAMATH_GPT_net_sag_calculation_l1409_140988


namespace NUMINAMATH_GPT_school_seat_payment_l1409_140903

def seat_cost (num_rows : ℕ) (seats_per_row : ℕ) (cost_per_seat : ℕ) (discount : ℕ → ℕ → ℕ) : ℕ :=
  let total_seats := num_rows * seats_per_row
  let total_cost := total_seats * cost_per_seat
  let groups_of_ten := total_seats / 10
  let total_discount := groups_of_ten * discount 10 cost_per_seat
  total_cost - total_discount

-- Define the discount function as 10% of the cost of a group of 10 seats
def discount (group_size : ℕ) (cost_per_seat : ℕ) : ℕ := (group_size * cost_per_seat) / 10

theorem school_seat_payment :
  seat_cost 5 8 30 discount = 1080 :=
sorry

end NUMINAMATH_GPT_school_seat_payment_l1409_140903


namespace NUMINAMATH_GPT_power_mod_condition_l1409_140955

-- Defining the main problem conditions
theorem power_mod_condition (n: ℕ) : 
  (7^2 ≡ 1 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k+1) ≡ 7 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k) ≡ 1 [MOD 12]) →
  7^135 ≡ 7 [MOD 12] :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_power_mod_condition_l1409_140955


namespace NUMINAMATH_GPT_matrix_cube_l1409_140910

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end NUMINAMATH_GPT_matrix_cube_l1409_140910


namespace NUMINAMATH_GPT_find_x_l1409_140991

theorem find_x (p q x : ℚ) (h1 : p / q = 4 / 5)
    (h2 : 4 / 7 + x / (2 * q + p) = 1) : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1409_140991


namespace NUMINAMATH_GPT_find_g1_l1409_140921

open Function

-- Definitions based on the conditions
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem find_g1 (g : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0) 
  (h2 : g (-1) = 1) 
  : g 1 = -3 :=
sorry

end NUMINAMATH_GPT_find_g1_l1409_140921


namespace NUMINAMATH_GPT_probability_margo_pairing_l1409_140953

-- Definition of the problem
def num_students : ℕ := 32
def num_pairings (n : ℕ) : ℕ := n - 1
def favorable_pairings : ℕ := 2

-- Theorem statement
theorem probability_margo_pairing :
  num_students = 32 →
  ∃ (p : ℚ), p = favorable_pairings / num_pairings num_students ∧ p = 2/31 :=
by
  intros h
  -- The proofs are omitted for brevity.
  sorry

end NUMINAMATH_GPT_probability_margo_pairing_l1409_140953


namespace NUMINAMATH_GPT_range_of_a_l1409_140994

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*x + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x a > 0) ↔ a > -3 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l1409_140994


namespace NUMINAMATH_GPT_smallest_number_divisible_1_to_10_l1409_140964

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_divisible_1_to_10_l1409_140964


namespace NUMINAMATH_GPT_right_angled_triangle_l1409_140938

-- Define the lengths of the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- State the theorem using the Pythagorean theorem
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_l1409_140938


namespace NUMINAMATH_GPT_sin2alpha_plus_cosalpha_l1409_140937

theorem sin2alpha_plus_cosalpha (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin2alpha_plus_cosalpha_l1409_140937


namespace NUMINAMATH_GPT_last_two_digits_condition_l1409_140925

-- Define the function to get last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Given numbers
def n1 := 122
def n2 := 123
def n3 := 125
def n4 := 129

-- The missing number
variable (x : ℕ)

theorem last_two_digits_condition : 
  last_two_digits (last_two_digits n1 * last_two_digits n2 * last_two_digits n3 * last_two_digits n4 * last_two_digits x) = 50 ↔ last_two_digits x = 1 :=
by 
  sorry

end NUMINAMATH_GPT_last_two_digits_condition_l1409_140925


namespace NUMINAMATH_GPT_class_duration_l1409_140950

theorem class_duration (h1 : 8 * 60 + 30 = 510) (h2 : 9 * 60 + 5 = 545) : (545 - 510 = 35) :=
by
  sorry

end NUMINAMATH_GPT_class_duration_l1409_140950


namespace NUMINAMATH_GPT_min_sum_six_l1409_140902

theorem min_sum_six (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 :=
sorry

end NUMINAMATH_GPT_min_sum_six_l1409_140902


namespace NUMINAMATH_GPT_total_time_correct_l1409_140971

def greta_time : ℝ := 6.5
def george_time : ℝ := greta_time - 1.5
def gloria_time : ℝ := 2 * george_time
def gary_time : ℝ := (george_time + gloria_time) + 1.75
def gwen_time : ℝ := (greta_time + george_time) - 0.40 * (greta_time + george_time)
def total_time : ℝ := greta_time + george_time + gloria_time + gary_time + gwen_time

theorem total_time_correct : total_time = 45.15 := by
  sorry

end NUMINAMATH_GPT_total_time_correct_l1409_140971


namespace NUMINAMATH_GPT_max_min_f_values_l1409_140983

noncomputable def f (a b c d : ℝ) : ℝ := (Real.sqrt (5 * a + 9) + Real.sqrt (5 * b + 9) + Real.sqrt (5 * c + 9) + Real.sqrt (5 * d + 9))

theorem max_min_f_values (a b c d : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h₁ : a + b + c + d = 32) :
  (f a b c d ≤ 28) ∧ (f a b c d ≥ 22) := by
  sorry

end NUMINAMATH_GPT_max_min_f_values_l1409_140983


namespace NUMINAMATH_GPT_real_root_bound_l1409_140978

noncomputable def P (x : ℝ) (n : ℕ) (ns : List ℕ) : ℝ :=
  1 + x^2 + x^5 + ns.foldr (λ n acc => x^n + acc) 0 + x^2008

theorem real_root_bound (n1 n2 : ℕ) (ns : List ℕ) (x : ℝ) :
  5 < n1 →
  List.Chain (λ a b => a < b) n1 (n2 :: ns) →
  n2 < 2008 →
  P x n1 (n2 :: ns) = 0 →
  x ≤ (1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_real_root_bound_l1409_140978


namespace NUMINAMATH_GPT_max_value_function_l1409_140924

theorem max_value_function (x : ℝ) (h : x < 0) : 
  ∃ y_max, (∀ x', x' < 0 → (x' + 4 / x') ≤ y_max) ∧ y_max = -4 := 
sorry

end NUMINAMATH_GPT_max_value_function_l1409_140924


namespace NUMINAMATH_GPT_min_fraction_value_l1409_140916

-- Define the conditions: geometric sequence, specific term relationship, product of terms

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

def specific_term_relationship (a : ℕ → ℝ) : Prop :=
  a 3 = a 2 + 2 * a 1

def product_of_terms (a : ℕ → ℝ) (m n : ℕ) : Prop :=
  a m * a n = 64 * (a 1)^2

def min_value_fraction (m n : ℕ) : Prop :=
  1 / m + 9 / n = 2

theorem min_fraction_value (a : ℕ → ℝ) (m n : ℕ)
  (h1 : geometric_sequence a)
  (h2 : specific_term_relationship a)
  (h3 : product_of_terms a m n)
  : min_value_fraction m n := by
  sorry

end NUMINAMATH_GPT_min_fraction_value_l1409_140916


namespace NUMINAMATH_GPT_division_of_fractions_l1409_140917

theorem division_of_fractions : (4 : ℚ) / (5 / 7) = 28 / 5 := sorry

end NUMINAMATH_GPT_division_of_fractions_l1409_140917


namespace NUMINAMATH_GPT_base_video_card_cost_l1409_140909

theorem base_video_card_cost
    (cost_computer : ℕ)
    (fraction_monitor_peripherals : ℕ → ℕ → ℕ)
    (twice : ℕ → ℕ)
    (total_spent : ℕ)
    (cost_monitor_peripherals_eq : fraction_monitor_peripherals cost_computer 5 = 300)
    (twice_eq : ∀ x, twice x = 2 * x)
    (eq_total : ∀ (base_video_card : ℕ), cost_computer + fraction_monitor_peripherals cost_computer 5 + twice base_video_card = total_spent)
    : ∃ x, total_spent = 2100 ∧ cost_computer = 1500 ∧ x = 150 :=
by
  sorry

end NUMINAMATH_GPT_base_video_card_cost_l1409_140909


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1409_140993

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : x = 6) 
  (h3 : y = 4) : 
  a * x + b * y - 10 = 0 := 
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1409_140993


namespace NUMINAMATH_GPT_lcm_of_three_l1409_140977

theorem lcm_of_three (A1 A2 A3 : ℕ) (D : ℕ)
  (hD : D = Nat.gcd (A1 * A2) (Nat.gcd (A2 * A3) (A3 * A1))) :
  Nat.lcm (Nat.lcm A1 A2) A3 = (A1 * A2 * A3) / D :=
sorry

end NUMINAMATH_GPT_lcm_of_three_l1409_140977


namespace NUMINAMATH_GPT_gcd_bc_eq_one_l1409_140967

theorem gcd_bc_eq_one (a b c x y : ℕ)
  (h1 : Nat.gcd a b = 120)
  (h2 : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  Nat.gcd b c = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_bc_eq_one_l1409_140967


namespace NUMINAMATH_GPT_second_particle_catches_first_l1409_140987

open Real

-- Define the distance functions for both particles
def distance_first (t : ℝ) : ℝ := 34 + 5 * t
def distance_second (t : ℝ) : ℝ := 0.25 * t^2 + 2.75 * t

-- The proof statement
theorem second_particle_catches_first : ∃ t : ℝ, distance_second t = distance_first t ∧ t = 17 :=
by
  have : distance_first 17 = 34 + 5 * 17 := by sorry
  have : distance_second 17 = 0.25 * 17^2 + 2.75 * 17 := by sorry
  sorry

end NUMINAMATH_GPT_second_particle_catches_first_l1409_140987


namespace NUMINAMATH_GPT_cos_sum_identity_l1409_140913

theorem cos_sum_identity (θ : ℝ) (h1 : Real.tan θ = -5 / 12) (h2 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  Real.cos (θ + Real.pi / 4) = 17 * Real.sqrt 2 / 26 :=
sorry

end NUMINAMATH_GPT_cos_sum_identity_l1409_140913


namespace NUMINAMATH_GPT_jane_crayons_l1409_140901

theorem jane_crayons :
  let start := 87
  let eaten := 7
  start - eaten = 80 :=
by
  sorry

end NUMINAMATH_GPT_jane_crayons_l1409_140901


namespace NUMINAMATH_GPT_least_multiple_of_7_not_lucky_l1409_140972

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem least_multiple_of_7_not_lucky : ∃ n, is_multiple_of_7 n ∧ ¬ is_lucky n ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_7_not_lucky_l1409_140972


namespace NUMINAMATH_GPT_describe_shape_cylinder_l1409_140981

-- Define cylindrical coordinates
structure CylindricalCoordinates where
  r : ℝ -- radial distance
  θ : ℝ -- azimuthal angle
  z : ℝ -- height

-- Define the positive constant c
variable (c : ℝ) (hc : 0 < c)

-- The theorem statement
theorem describe_shape_cylinder (p : CylindricalCoordinates) (h : p.r = c) : 
  ∃ (p : CylindricalCoordinates), p.r = c :=
by
  sorry

end NUMINAMATH_GPT_describe_shape_cylinder_l1409_140981


namespace NUMINAMATH_GPT_toothpicks_in_stage_200_l1409_140941

def initial_toothpicks : ℕ := 6
def toothpicks_per_stage : ℕ := 5
def stage_number : ℕ := 200

theorem toothpicks_in_stage_200 :
  initial_toothpicks + (stage_number - 1) * toothpicks_per_stage = 1001 := by
  sorry

end NUMINAMATH_GPT_toothpicks_in_stage_200_l1409_140941


namespace NUMINAMATH_GPT_price_change_theorem_l1409_140959

-- Define initial prices
def candy_box_price_before : ℝ := 10
def soda_can_price_before : ℝ := 9
def popcorn_bag_price_before : ℝ := 5
def gum_pack_price_before : ℝ := 2

-- Define price changes
def candy_box_price_increase := candy_box_price_before * 0.25
def soda_can_price_decrease := soda_can_price_before * 0.15
def popcorn_bag_price_factor := 2
def gum_pack_price_change := 0

-- Compute prices after the policy changes
def candy_box_price_after := candy_box_price_before + candy_box_price_increase
def soda_can_price_after := soda_can_price_before - soda_can_price_decrease
def popcorn_bag_price_after := popcorn_bag_price_before * popcorn_bag_price_factor
def gum_pack_price_after := gum_pack_price_before

-- Compute total costs
def total_cost_before := candy_box_price_before + soda_can_price_before + popcorn_bag_price_before + gum_pack_price_before
def total_cost_after := candy_box_price_after + soda_can_price_after + popcorn_bag_price_after + gum_pack_price_after

-- The statement to be proven
theorem price_change_theorem :
  total_cost_before = 26 ∧ total_cost_after = 32.15 :=
by
  -- This part requires proof, add 'sorry' for now
  sorry

end NUMINAMATH_GPT_price_change_theorem_l1409_140959


namespace NUMINAMATH_GPT_strictly_positive_integers_equal_l1409_140929

theorem strictly_positive_integers_equal 
  (a b : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : 
  a = b :=
sorry

end NUMINAMATH_GPT_strictly_positive_integers_equal_l1409_140929


namespace NUMINAMATH_GPT_probability_of_event_l1409_140943

noncomputable def interval_probability : ℝ :=
  if 0 ≤ 1 ∧ 1 ≤ 1 then (1 - (1/3)) / (1 - 0) else 0

theorem probability_of_event :
  interval_probability = 2 / 3 :=
by
  rw [interval_probability]
  sorry

end NUMINAMATH_GPT_probability_of_event_l1409_140943


namespace NUMINAMATH_GPT_reef_age_in_decimal_l1409_140979

def octal_to_decimal (n: Nat) : Nat :=
  match n with
  | 367 => 7 * (8^0) + 6 * (8^1) + 3 * (8^2)
  | _   => 0  -- Placeholder for other values if needed

theorem reef_age_in_decimal : octal_to_decimal 367 = 247 := by
  sorry

end NUMINAMATH_GPT_reef_age_in_decimal_l1409_140979


namespace NUMINAMATH_GPT_continuous_stripe_probability_l1409_140908

noncomputable def probability_continuous_stripe : ℚ :=
  let total_configurations := 4^6
  let favorable_configurations := 48
  favorable_configurations / total_configurations

theorem continuous_stripe_probability : probability_continuous_stripe = 3 / 256 :=
  by
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l1409_140908


namespace NUMINAMATH_GPT_smallest_integer_proof_l1409_140936

noncomputable def smallestInteger (s : ℝ) (h : s < 1 / 2000) : ℤ :=
  Nat.ceil (Real.sqrt (1999 / 3))

theorem smallest_integer_proof (s : ℝ) (h : s < 1 / 2000) (m : ℤ) (hm : m = (smallestInteger s h + s)^3) : smallestInteger s h = 26 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_proof_l1409_140936


namespace NUMINAMATH_GPT_terminal_side_second_or_third_quadrant_l1409_140984

-- Definitions and conditions directly from part a)
def sin (x : ℝ) : ℝ := sorry
def tan (x : ℝ) : ℝ := sorry
def terminal_side_in_quadrant (x : ℝ) (q : ℕ) : Prop := sorry

-- Proving the mathematically equivalent proof
theorem terminal_side_second_or_third_quadrant (x : ℝ) :
  sin x * tan x < 0 →
  (terminal_side_in_quadrant x 2 ∨ terminal_side_in_quadrant x 3) :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_second_or_third_quadrant_l1409_140984


namespace NUMINAMATH_GPT_smallest_number_of_roses_to_buy_l1409_140960

-- Definitions representing the conditions
def group_size1 : ℕ := 9
def group_size2 : ℕ := 19

-- Statement representing the problem and solution
theorem smallest_number_of_roses_to_buy : Nat.lcm group_size1 group_size2 = 171 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_number_of_roses_to_buy_l1409_140960


namespace NUMINAMATH_GPT_find_J_l1409_140949

-- Define the problem conditions
def eq1 : Nat := 32
def eq2 : Nat := 4

-- Define the target equation form
def target_eq (J : Nat) : Prop := (eq1^3) * (eq2^3) = 2^J

theorem find_J : ∃ J : Nat, target_eq J ∧ J = 21 :=
by
  -- Rest of the proof goes here
  sorry

end NUMINAMATH_GPT_find_J_l1409_140949


namespace NUMINAMATH_GPT_blue_paint_cans_l1409_140999

noncomputable def ratio_of_blue_to_green := 4 / 1
def total_cans := 50
def fraction_of_blue := 4 / (4 + 1)
def number_of_blue_cans := fraction_of_blue * total_cans

theorem blue_paint_cans : number_of_blue_cans = 40 := by
  sorry

end NUMINAMATH_GPT_blue_paint_cans_l1409_140999


namespace NUMINAMATH_GPT_monotonic_intervals_minimum_m_value_l1409_140931

noncomputable def f (x : ℝ) (a : ℝ) := (2 * Real.exp 1 + 1) * Real.log x - (3 * a / 2) * x + 1

theorem monotonic_intervals (a : ℝ) : 
  if a ≤ 0 then ∀ x ∈ Set.Ioi 0, 0 < (2 * Real.exp 1 + 1) / x - (3 * a / 2) 
  else ∀ x ∈ Set.Ioc 0 ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) > 0 ∧
       ∀ x ∈ Set.Ioi ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) < 0 := sorry

noncomputable def g (x : ℝ) (m : ℝ) := x * Real.exp x + m - ((2 * Real.exp 1 + 1) * Real.log x + x - 1)

theorem minimum_m_value :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 < x → g x m ≥ 0) ↔ m ≥ - Real.exp 1 := sorry

end NUMINAMATH_GPT_monotonic_intervals_minimum_m_value_l1409_140931


namespace NUMINAMATH_GPT_iesha_total_books_l1409_140951

theorem iesha_total_books (schoolBooks sportsBooks : ℕ) (h1 : schoolBooks = 19) (h2 : sportsBooks = 39) : schoolBooks + sportsBooks = 58 :=
by
  sorry

end NUMINAMATH_GPT_iesha_total_books_l1409_140951


namespace NUMINAMATH_GPT_log2_75_in_terms_of_a_b_l1409_140923

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

variables (a b : ℝ)
variables (log2_9_eq_a : log_base2 9 = a)
variables (log2_5_eq_b : log_base2 5 = b)

theorem log2_75_in_terms_of_a_b : log_base2 75 = (1 / 2) * a + 2 * b :=
by sorry

end NUMINAMATH_GPT_log2_75_in_terms_of_a_b_l1409_140923


namespace NUMINAMATH_GPT_eval_f_at_3_l1409_140954

def f (x : ℝ) : ℝ := 3 * x + 1

theorem eval_f_at_3 : f 3 = 10 :=
by
  -- computation of f at x = 3
  sorry

end NUMINAMATH_GPT_eval_f_at_3_l1409_140954


namespace NUMINAMATH_GPT_hydrogen_moles_formed_l1409_140939

open Function

-- Define types for the substances involved in the reaction
structure Substance :=
  (name : String)
  (moles : ℕ)

-- Define the reaction
def reaction (NaH H2O NaOH H2 : Substance) : Prop :=
  NaH.moles = H2O.moles ∧ NaOH.moles = H2.moles

-- Given conditions
def NaH_initial : Substance := ⟨"NaH", 2⟩
def H2O_initial : Substance := ⟨"H2O", 2⟩
def NaOH_final : Substance := ⟨"NaOH", 2⟩
def H2_final : Substance := ⟨"H2", 2⟩

-- Problem statement in Lean
theorem hydrogen_moles_formed :
  reaction NaH_initial H2O_initial NaOH_final H2_final → H2_final.moles = 2 :=
by
  -- Skip proof
  sorry

end NUMINAMATH_GPT_hydrogen_moles_formed_l1409_140939


namespace NUMINAMATH_GPT_smallest_d_l1409_140904

noncomputable def abc_identity_conditions (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3 * d * x^2 + 3 * x + e^3

theorem smallest_d (a b c d e : ℝ) (h : abc_identity_conditions a b c d e) : d = 1 := 
sorry

end NUMINAMATH_GPT_smallest_d_l1409_140904


namespace NUMINAMATH_GPT_current_time_is_208_l1409_140989

def minute_hand_position (t : ℝ) : ℝ := 6 * t
def hour_hand_position (t : ℝ) : ℝ := 0.5 * t

theorem current_time_is_208 (t : ℝ) (h1 : 0 < t) (h2 : t < 60) 
  (h3 : minute_hand_position (t + 8) + 60 = hour_hand_position (t + 5)) : 
  t = 8 :=
by sorry

end NUMINAMATH_GPT_current_time_is_208_l1409_140989


namespace NUMINAMATH_GPT_induction_proof_l1409_140945

def f (n : ℕ) : ℕ := (List.range (2 * n - 1)).sum + n

theorem induction_proof (n : ℕ) (h : n > 0) : f (n + 1) - f n = 8 * n := by
  sorry

end NUMINAMATH_GPT_induction_proof_l1409_140945
