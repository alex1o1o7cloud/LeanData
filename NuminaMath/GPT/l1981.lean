import Mathlib

namespace gcd_lcm_identity_l1981_198164

variables {n m k : ℕ}

/-- Given positive integers n, m, and k such that n divides lcm(m, k) 
    and m divides lcm(n, k), we prove that n * gcd(m, k) = m * gcd(n, k). -/
theorem gcd_lcm_identity (n_pos : 0 < n) (m_pos : 0 < m) (k_pos : 0 < k) 
  (h1 : n ∣ Nat.lcm m k) (h2 : m ∣ Nat.lcm n k) :
  n * Nat.gcd m k = m * Nat.gcd n k :=
sorry

end gcd_lcm_identity_l1981_198164


namespace max_area_parallelogram_l1981_198128

theorem max_area_parallelogram
    (P : ℝ)
    (a b : ℝ)
    (h1 : P = 60)
    (h2 : a = 3 * b)
    (h3 : P = 2 * a + 2 * b) :
    (a * b ≤ 168.75) :=
by
  -- We prove that given the conditions, the maximum area is 168.75 square units.
  sorry

end max_area_parallelogram_l1981_198128


namespace remainder_of_multiple_of_n_mod_7_l1981_198173

theorem remainder_of_multiple_of_n_mod_7
  (n m : ℤ)
  (h1 : n % 7 = 1)
  (h2 : m % 7 = 3) :
  (m * n) % 7 = 3 :=
by
  sorry

end remainder_of_multiple_of_n_mod_7_l1981_198173


namespace fraction_subtraction_l1981_198108

theorem fraction_subtraction : (9 / 23) - (5 / 69) = 22 / 69 :=
by
  sorry

end fraction_subtraction_l1981_198108


namespace problem_exists_integers_a_b_c_d_l1981_198151

theorem problem_exists_integers_a_b_c_d :
  ∃ (a b c d : ℤ), 
  |a| > 1000000 ∧ |b| > 1000000 ∧ |c| > 1000000 ∧ |d| > 1000000 ∧
  (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) + 1 / (d:ℚ) = 1 / (a * b * c * d : ℚ)) :=
sorry

end problem_exists_integers_a_b_c_d_l1981_198151


namespace arithmetic_sequence_a7_l1981_198134

noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence_a7 
  (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 4 - a 1) / 3)
  (h_a1 : a 1 = 3)
  (h_a4 : a 4 = 5) : 
  a 7 = 7 :=
by
  sorry

end arithmetic_sequence_a7_l1981_198134


namespace square_side_length_l1981_198145

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l1981_198145


namespace simplify_sqrt_expression_l1981_198171

theorem simplify_sqrt_expression (h : Real.sqrt 3 > 1) :
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 :=
by
  sorry

end simplify_sqrt_expression_l1981_198171


namespace square_field_area_l1981_198132

theorem square_field_area (s : ℕ) (area cost_per_meter total_cost gate_width : ℕ):
  area = s^2 →
  cost_per_meter = 2 →
  total_cost = 1332 →
  gate_width = 1 →
  (4 * s - 2 * gate_width) * cost_per_meter = total_cost →
  area = 27889 :=
by
  intros h_area h_cost_per_meter h_total_cost h_gate_width h_equation
  sorry

end square_field_area_l1981_198132


namespace mary_income_more_than_tim_income_l1981_198193

variables (J T M : ℝ)
variables (h1 : T = 0.60 * J) (h2 : M = 0.8999999999999999 * J)

theorem mary_income_more_than_tim_income : (M - T) / T * 100 = 50 :=
by
  sorry

end mary_income_more_than_tim_income_l1981_198193


namespace average_speed_l1981_198136

theorem average_speed (v1 v2 : ℝ) (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0) : 
  2 / (1 / v1 + 1 / v2) = 2 * v1 * v2 / (v1 + v2) :=
by sorry

end average_speed_l1981_198136


namespace morning_routine_time_l1981_198146

section

def time_for_teeth_and_face : ℕ := 3
def time_for_cooking : ℕ := 14
def time_for_reading_while_cooking : ℕ := time_for_cooking - time_for_teeth_and_face
def additional_time_for_reading : ℕ := 1
def total_time_for_reading : ℕ := time_for_reading_while_cooking + additional_time_for_reading
def time_for_eating : ℕ := 6

def total_time_to_school : ℕ := time_for_cooking + time_for_eating

theorem morning_routine_time :
  total_time_to_school = 21 := sorry

end

end morning_routine_time_l1981_198146


namespace contradiction_method_l1981_198176

theorem contradiction_method (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 :=
sorry

end contradiction_method_l1981_198176


namespace integer_points_count_l1981_198153

theorem integer_points_count :
  ∃ (n : ℤ), n = 9 ∧
  ∀ a b : ℝ, (1 < a) → (1 < b) → (ab + a - b - 10 = 0) →
  (a + b = 6) → 
  ∃ (x y : ℤ), (3 * x^2 + 2 * y^2 ≤ 6) :=
by
  sorry

end integer_points_count_l1981_198153


namespace daily_wage_of_a_man_l1981_198131

theorem daily_wage_of_a_man (M W : ℝ) 
  (h1 : 24 * M + 16 * W = 11600) 
  (h2 : 12 * M + 37 * W = 11600) : 
  M = 350 :=
by
  sorry

end daily_wage_of_a_man_l1981_198131


namespace ab_range_l1981_198163

theorem ab_range (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 - x^2|)
  (h_a_lt_b : 0 < a ∧ a < b) (h_fa_eq_fb : f a = f b) :
  0 < a * b ∧ a * b < 2 := 
by
  sorry

end ab_range_l1981_198163


namespace parallel_lines_m_eq_minus_seven_l1981_198137

theorem parallel_lines_m_eq_minus_seven
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m)
  (l₂ : ∀ x y : ℝ, 2 * x + (5 + m) * y = 8)
  (parallel : ∀ x y : ℝ, (3 + m) * 4 = 2 * (5 + m)) :
  m = -7 :=
sorry

end parallel_lines_m_eq_minus_seven_l1981_198137


namespace bill_painting_hours_l1981_198142

theorem bill_painting_hours (B J : ℝ) (hB : 0 < B) (hJ : 0 < J) : 
  ∃ t : ℝ, t = (B-1)/(B+J) ∧ (t + 1 = (B * (J + 1)) / (B + J)) :=
by
  sorry

end bill_painting_hours_l1981_198142


namespace find_k_l1981_198185

theorem find_k (x y k : ℤ) (h₁ : x = -1) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) :
  k = 4 :=
by
  sorry

end find_k_l1981_198185


namespace vector_coordinates_l1981_198154

theorem vector_coordinates :
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1) :=
by
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  show (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1)
  sorry

end vector_coordinates_l1981_198154


namespace coffee_tea_overlap_l1981_198180

theorem coffee_tea_overlap (c t : ℕ) (h_c : c = 80) (h_t : t = 70) : 
  ∃ (b : ℕ), b = 50 := 
by 
  sorry

end coffee_tea_overlap_l1981_198180


namespace third_square_length_l1981_198199

theorem third_square_length 
  (A1 : 8 * 5 = 40) 
  (A2 : 10 * 7 = 70) 
  (A3 : 15 * 9 = 135) 
  (L : ℕ) 
  (A4 : 40 + 70 + L * 5 = 135) 
  : L = 5 := 
sorry

end third_square_length_l1981_198199


namespace train_length_is_180_l1981_198144

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ := 
  (speed_kmh * 5 / 18) * time_seconds

theorem train_length_is_180 : train_length 9 72 = 180 :=
by
  sorry

end train_length_is_180_l1981_198144


namespace intersection_A_B_l1981_198139

def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B :
  (setA ∩ setB = {x | 0 ≤ x ∧ x ≤ 2}) :=
by
  sorry

end intersection_A_B_l1981_198139


namespace total_teachers_l1981_198141

theorem total_teachers (total_individuals sample_size sampled_students : ℕ)
  (H1 : total_individuals = 2400)
  (H2 : sample_size = 160)
  (H3 : sampled_students = 150) :
  ∃ total_teachers, total_teachers * (sample_size / (sample_size - sampled_students)) = 2400 / (sample_size / (sample_size - sampled_students)) ∧ total_teachers = 150 := 
  sorry

end total_teachers_l1981_198141


namespace smallest_four_digit_in_pascal_l1981_198104

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l1981_198104


namespace largest_multiple_of_15_less_than_500_l1981_198119

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l1981_198119


namespace find_number_90_l1981_198178

theorem find_number_90 {x y : ℝ} (h1 : x = y + 0.11 * y) (h2 : x = 99.9) : y = 90 :=
sorry

end find_number_90_l1981_198178


namespace sophie_oranges_per_day_l1981_198190

/-- Sophie and Hannah together eat a certain number of fruits in 30 days.
    Given Hannah eats 40 grapes every day, prove that Sophie eats 20 oranges every day. -/
theorem sophie_oranges_per_day (total_fruits : ℕ) (grapes_per_day : ℕ) (days : ℕ)
  (total_days_fruits : total_fruits = 1800) (hannah_grapes : grapes_per_day = 40) (days_count : days = 30) :
  (total_fruits - grapes_per_day * days) / days = 20 :=
by
  sorry

end sophie_oranges_per_day_l1981_198190


namespace range_of_a_l1981_198189

theorem range_of_a (a : ℝ) (h₁ : 1/2 ≤ 1) (h₂ : a ≤ a + 1)
    (h_condition : ∀ x:ℝ, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) :
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l1981_198189


namespace minimum_x_plus_3y_l1981_198124

theorem minimum_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) : x + 3 * y ≥ 16 :=
sorry

end minimum_x_plus_3y_l1981_198124


namespace max_leap_years_in_200_years_l1981_198196

-- Definitions based on conditions
def leap_year_occurrence (years : ℕ) : ℕ :=
  years / 4

-- Define the problem statement based on the given conditions and required proof
theorem max_leap_years_in_200_years : leap_year_occurrence 200 = 50 := 
by
  sorry

end max_leap_years_in_200_years_l1981_198196


namespace find_f_half_l1981_198158

variable {α : Type} [DivisionRing α]

theorem find_f_half {f : α → α} (h : ∀ x, f (1 - 2 * x) = 1 / (x^2)) : f (1 / 2) = 16 :=
by
  sorry

end find_f_half_l1981_198158


namespace prize_distribution_l1981_198113

theorem prize_distribution 
  (total_winners : ℕ)
  (score1 score2 score3 : ℕ)
  (total_points : ℕ) 
  (winners1 winners2 winners3 : ℕ) :
  total_winners = 5 →
  score1 = 20 →
  score2 = 19 →
  score3 = 18 →
  total_points = 94 →
  score1 * winners1 + score2 * winners2 + score3 * winners3 = total_points →
  winners1 + winners2 + winners3 = total_winners →
  winners1 = 1 ∧ winners2 = 2 ∧ winners3 = 2 :=
by
  intros
  sorry

end prize_distribution_l1981_198113


namespace cyclic_quadrilateral_condition_l1981_198114

-- Definitions of the points and sides of the triangle
variables (A B C S E F : Type) 

-- Assume S is the centroid of triangle ABC
def is_centroid (A B C S : Type) : Prop := 
  -- actual centralized definition here (omitted)
  sorry

-- Assume E is the midpoint of side AB
def is_midpoint (A B E : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume F is the midpoint of side AC
def is_midpoint_AC (A C F : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume a quadrilateral AESF
def is_cyclic (A E S F : Type) : Prop :=
  -- actual cyclic definition here (omitted)
  sorry 

theorem cyclic_quadrilateral_condition 
  (A B C S E F : Type)
  (a b c : ℝ) 
  (h1 : is_centroid A B C S)
  (h2 : is_midpoint A B E) 
  (h3 : is_midpoint_AC A C F) :
  is_cyclic A E S F ↔ (c^2 + b^2 = 2 * a^2) :=
sorry

end cyclic_quadrilateral_condition_l1981_198114


namespace reduced_cost_per_meter_l1981_198135

theorem reduced_cost_per_meter (original_cost total_cost new_length original_length : ℝ) :
  original_cost = total_cost / original_length →
  new_length = original_length + 4 →
  total_cost = total_cost →
  original_cost - (total_cost / new_length) = 1 :=
by sorry

end reduced_cost_per_meter_l1981_198135


namespace banana_cream_pie_correct_slice_l1981_198147

def total_students := 45
def strawberry_pie_preference := 15
def pecan_pie_preference := 10
def pumpkin_pie_preference := 9

noncomputable def banana_cream_pie_slice_degrees : ℝ :=
  let remaining_students := total_students - strawberry_pie_preference - pecan_pie_preference - pumpkin_pie_preference
  let students_per_preference := remaining_students / 2
  (students_per_preference / total_students) * 360

theorem banana_cream_pie_correct_slice :
  banana_cream_pie_slice_degrees = 44 := by
  sorry

end banana_cream_pie_correct_slice_l1981_198147


namespace range_of_function_x_geq_0_l1981_198102

theorem range_of_function_x_geq_0 :
  ∀ (x : ℝ), x ≥ 0 → ∃ (y : ℝ), y ≥ 3 ∧ (y = x^2 + 2 * x + 3) :=
by
  sorry

end range_of_function_x_geq_0_l1981_198102


namespace trevor_spends_more_l1981_198109

theorem trevor_spends_more (T R Q : ℕ) 
  (hT : T = 80) 
  (hR : R = 2 * Q) 
  (hTotal : 4 * (T + R + Q) = 680) : 
  T = R + 20 :=
by
  sorry

end trevor_spends_more_l1981_198109


namespace total_points_correct_l1981_198111

def points_from_two_pointers (t : ℕ) : ℕ := 2 * t
def points_from_three_pointers (th : ℕ) : ℕ := 3 * th
def points_from_free_throws (f : ℕ) : ℕ := f

def total_points (two_points three_points free_throws : ℕ) : ℕ :=
  points_from_two_pointers two_points + points_from_three_pointers three_points + points_from_free_throws free_throws

def sam_points : ℕ := total_points 20 5 10
def alex_points : ℕ := total_points 15 6 8
def jake_points : ℕ := total_points 10 8 5
def lily_points : ℕ := total_points 12 3 16

def game_total_points : ℕ := sam_points + alex_points + jake_points + lily_points

theorem total_points_correct : game_total_points = 219 :=
by
  sorry

end total_points_correct_l1981_198111


namespace find_m_l1981_198123

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l1981_198123


namespace least_integer_greater_than_sqrt_450_l1981_198195

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l1981_198195


namespace min_value_of_quadratic_l1981_198159

theorem min_value_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ x : ℝ, (x = -p / 2) ∧ ∀ y : ℝ, (y^2 + p * y + q) ≥ ((-p/2)^2 + p * (-p/2) + q) :=
sorry

end min_value_of_quadratic_l1981_198159


namespace tips_fraction_to_salary_l1981_198118

theorem tips_fraction_to_salary (S T I : ℝ)
  (h1 : I = S + T)
  (h2 : T / I = 0.6923076923076923) :
  T / S = 2.25 := by
  sorry

end tips_fraction_to_salary_l1981_198118


namespace first_train_speed_l1981_198100

noncomputable def speed_of_first_train (length_train1 : ℕ) (speed_train2 : ℕ) (length_train2 : ℕ) (time_cross : ℕ) : ℕ :=
  let relative_speed_m_s := (500 : ℕ) / time_cross
  let relative_speed_km_h := relative_speed_m_s * 18 / 5
  relative_speed_km_h - speed_train2

theorem first_train_speed :
  speed_of_first_train 270 80 230 9 = 920 := by
  sorry

end first_train_speed_l1981_198100


namespace probability_of_Z_l1981_198182

namespace ProbabilityProof

def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_X_or_Y_or_Z : ℚ := 0.4583333333333333

theorem probability_of_Z :
  ∃ P_Z : ℚ, P_Z = 0.0833333333333333 ∧ 
  P_X_or_Y_or_Z = P_X + P_Y + P_Z :=
by
  sorry

end ProbabilityProof

end probability_of_Z_l1981_198182


namespace three_digit_numbers_without_579_l1981_198130

def count_valid_digits (exclusions : List Nat) (range : List Nat) : Nat :=
  (range.filter (λ n => n ∉ exclusions)).length

def count_valid_three_digit_numbers : Nat :=
  let hundreds := count_valid_digits [5, 7, 9] [1, 2, 3, 4, 6, 8]
  let tens_units := count_valid_digits [5, 7, 9] [0, 1, 2, 3, 4, 6, 8]
  hundreds * tens_units * tens_units

theorem three_digit_numbers_without_579 : 
  count_valid_three_digit_numbers = 294 :=
by
  unfold count_valid_three_digit_numbers
  /- 
  Here you can add intermediate steps if necessary, 
  but for now we assert the final goal since this is 
  just the problem statement with the proof omitted.
  -/
  sorry

end three_digit_numbers_without_579_l1981_198130


namespace sum_of_monomials_is_monomial_l1981_198152

variable (a b : ℕ)

theorem sum_of_monomials_is_monomial (m n : ℕ) (h : ∃ k : ℕ, 2 * a^m * b^n + a * b^3 = k * a^1 * b^3) :
  m = 1 ∧ n = 3 :=
sorry

end sum_of_monomials_is_monomial_l1981_198152


namespace concatenated_natural_irrational_l1981_198112

def concatenated_natural_decimal : ℝ := 0.1234567891011121314151617181920 -- and so on

theorem concatenated_natural_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ concatenated_natural_decimal = p / q :=
sorry

end concatenated_natural_irrational_l1981_198112


namespace integer_pairs_count_l1981_198143

theorem integer_pairs_count : ∃ (pairs : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), (x ≥ y ∧ (x, y) ∈ pairs → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 211))
  ∧ pairs.card = 3 :=
by
  sorry

end integer_pairs_count_l1981_198143


namespace find_number_l1981_198181

theorem find_number (n x : ℤ) (h1 : n * x + 3 = 10 * x - 17) (h2 : x = 4) : n = 5 :=
by
  sorry

end find_number_l1981_198181


namespace find_n_if_pow_eqn_l1981_198186

theorem find_n_if_pow_eqn (n : ℕ) :
  6 ^ 3 = 9 ^ n → n = 3 :=
by 
  sorry

end find_n_if_pow_eqn_l1981_198186


namespace depth_of_lost_ship_l1981_198191

theorem depth_of_lost_ship (rate_of_descent : ℕ) (time_taken : ℕ) (h1 : rate_of_descent = 60) (h2 : time_taken = 60) :
  rate_of_descent * time_taken = 3600 :=
by {
  /-
  Proof steps would go here.
  -/
  sorry
}

end depth_of_lost_ship_l1981_198191


namespace barber_total_loss_is_120_l1981_198138

-- Definitions for the conditions
def haircut_cost : ℕ := 25
def initial_payment_by_customer : ℕ := 50
def flower_shop_change : ℕ := 50
def bakery_change : ℕ := 10
def customer_received_change : ℕ := 25
def counterfeit_50_replacement : ℕ := 50
def counterfeit_10_replacement : ℕ := 10

-- Calculate total loss for the barber
def total_loss : ℕ :=
  let loss_haircut := haircut_cost
  let loss_change_to_customer := customer_received_change
  let loss_given_to_flower_shop := counterfeit_50_replacement
  let loss_given_to_bakery := counterfeit_10_replacement
  let total_loss_before_offset := loss_haircut + loss_change_to_customer + loss_given_to_flower_shop + loss_given_to_bakery
  let real_currency_received := flower_shop_change
  total_loss_before_offset - real_currency_received

-- Proof statement
theorem barber_total_loss_is_120 : total_loss = 120 := by {
  sorry
}

end barber_total_loss_is_120_l1981_198138


namespace solve_for_y_l1981_198115

theorem solve_for_y (x y : ℝ) (h1 : x ^ (2 * y) = 16) (h2 : x = 2) : y = 2 :=
by {
  sorry
}

end solve_for_y_l1981_198115


namespace find_b_l1981_198175

theorem find_b (b : ℕ) (h1 : 0 ≤ b) (h2 : b ≤ 20) (h3 : (746392847 - b) % 17 = 0) : b = 16 :=
sorry

end find_b_l1981_198175


namespace discriminant_of_quadratic_eq_l1981_198167

-- Define the coefficients of the quadratic equation
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- State the theorem that we want to prove
theorem discriminant_of_quadratic_eq : discriminant a b c = 61 := by
  sorry

end discriminant_of_quadratic_eq_l1981_198167


namespace butanoic_acid_molecular_weight_l1981_198155

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight_butanoic_acid : ℝ :=
  4 * atomic_weight_C + 8 * atomic_weight_H + 2 * atomic_weight_O

theorem butanoic_acid_molecular_weight :
  molecular_weight_butanoic_acid = 88.104 :=
by
  -- proof not required
  sorry

end butanoic_acid_molecular_weight_l1981_198155


namespace jessa_needs_470_cupcakes_l1981_198166

def total_cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade_class : ℕ) (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_afterschool_club : ℕ) : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) + pe_class_students + (afterschool_clubs * students_per_afterschool_club)

theorem jessa_needs_470_cupcakes :
  total_cupcakes_needed 8 40 80 2 35 = 470 :=
by
  sorry

end jessa_needs_470_cupcakes_l1981_198166


namespace intersection_is_correct_l1981_198121

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := { x | x > 1/3 }
def setB : Set ℝ := { y | -3 ≤ y ∧ y ≤ 3 }

-- Prove that the intersection of A and B is (1/3, 3]
theorem intersection_is_correct : setA ∩ setB = { x | 1/3 < x ∧ x ≤ 3 } := 
by
  sorry

end intersection_is_correct_l1981_198121


namespace find_a1_l1981_198125

-- Definitions used in the conditions
variables {a : ℕ → ℝ} -- Sequence a(n)
variable (n : ℕ) -- Number of terms
noncomputable def arithmeticSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

noncomputable def arithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m + (n - m) * (a 2 - a 1)

theorem find_a1 (h_seq : arithmeticSeq a)
  (h_sum_first_100 : arithmeticSum a 100 = 100)
  (h_sum_last_100 : arithmeticSum (λ i => a (i + 900)) 100 = 1000) :
  a 1 = 101 / 200 :=
  sorry

end find_a1_l1981_198125


namespace time_after_2023_hours_l1981_198165

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end time_after_2023_hours_l1981_198165


namespace no_such_function_l1981_198188

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, (∀ y x : ℝ, 0 < x → x < y → f y > (y - x) * (f x)^2) :=
by
  sorry

end no_such_function_l1981_198188


namespace find_zebras_last_year_l1981_198156

def zebras_last_year (current : ℕ) (born : ℕ) (died : ℕ) : ℕ :=
  current - born + died

theorem find_zebras_last_year :
  zebras_last_year 725 419 263 = 569 :=
by
  sorry

end find_zebras_last_year_l1981_198156


namespace machine_fill_time_l1981_198161

theorem machine_fill_time (filled_cans : ℕ) (time_per_batch : ℕ) (total_cans : ℕ) (expected_time : ℕ)
  (h1 : filled_cans = 150)
  (h2 : time_per_batch = 8)
  (h3 : total_cans = 675)
  (h4 : expected_time = 36) :
  (total_cans / filled_cans) * time_per_batch = expected_time :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end machine_fill_time_l1981_198161


namespace unit_digit_14_pow_100_l1981_198198

theorem unit_digit_14_pow_100 : (14 ^ 100) % 10 = 6 :=
by
  sorry

end unit_digit_14_pow_100_l1981_198198


namespace mango_selling_price_l1981_198116

theorem mango_selling_price
  (CP SP_loss SP_profit : ℝ)
  (h1 : SP_loss = 0.8 * CP)
  (h2 : SP_profit = 1.05 * CP)
  (h3 : SP_profit = 6.5625) :
  SP_loss = 5.00 :=
by
  sorry

end mango_selling_price_l1981_198116


namespace white_black_ratio_l1981_198122

theorem white_black_ratio (W B : ℕ) (h1 : W + B = 78) (h2 : (2 / 3 : ℚ) * (B - W) = 4) : W / B = 6 / 7 := by
  sorry

end white_black_ratio_l1981_198122


namespace exists_n_divisible_l1981_198169

theorem exists_n_divisible (k : ℕ) (m : ℤ) (hk : k > 0) (hm : m % 2 = 1) : 
  ∃ n : ℕ, n > 0 ∧ 2^k ∣ (n^n - m) :=
by
  sorry

end exists_n_divisible_l1981_198169


namespace matrix_addition_is_correct_l1981_198149

-- Definitions of matrices A and B according to given conditions
def A : Matrix (Fin 4) (Fin 4) ℤ :=  
  ![![ 3,  0,  1,  4],
    ![ 1,  2,  0,  0],
    ![ 5, -3,  2,  1],
    ![ 0,  0, -1,  3]]

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-5, -7,  3,  2],
    ![ 4, -9,  5, -2],
    ![ 8,  2, -3,  0],
    ![ 1,  1, -2, -4]]

-- The expected result matrix from the addition of A and B
def C : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-2, -7,  4,  6],
    ![ 5, -7,  5, -2],
    ![13, -1, -1,  1],
    ![ 1,  1, -3, -1]]

-- The statement that A + B equals C
theorem matrix_addition_is_correct : A + B = C :=
by 
  -- Here we would provide the proof steps.
  sorry

end matrix_addition_is_correct_l1981_198149


namespace apples_distribution_l1981_198129

theorem apples_distribution (total_apples : ℝ) (apples_per_person : ℝ) (number_of_people : ℝ) 
    (h1 : total_apples = 45) (h2 : apples_per_person = 15.0) : number_of_people = 3 :=
by
  sorry

end apples_distribution_l1981_198129


namespace sabrina_total_leaves_l1981_198103

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end sabrina_total_leaves_l1981_198103


namespace Gary_final_amount_l1981_198187

theorem Gary_final_amount
(initial_amount dollars_snake dollars_hamster dollars_supplies : ℝ)
(h1 : initial_amount = 73.25)
(h2 : dollars_snake = 55.50)
(h3 : dollars_hamster = 25.75)
(h4 : dollars_supplies = 12.40) :
  initial_amount + dollars_snake - dollars_hamster - dollars_supplies = 90.60 :=
by
  sorry

end Gary_final_amount_l1981_198187


namespace cherry_orange_punch_ratio_l1981_198197

theorem cherry_orange_punch_ratio 
  (C : ℝ)
  (h_condition1 : 4.5 + C + (C - 1.5) = 21) : 
  C / 4.5 = 2 :=
by
  sorry

end cherry_orange_punch_ratio_l1981_198197


namespace gwen_did_not_recycle_2_bags_l1981_198160

def points_per_bag : ℕ := 8
def total_bags : ℕ := 4
def points_earned : ℕ := 16

theorem gwen_did_not_recycle_2_bags : total_bags - points_earned / points_per_bag = 2 := by
  sorry

end gwen_did_not_recycle_2_bags_l1981_198160


namespace parabola_vertex_in_fourth_quadrant_l1981_198110

theorem parabola_vertex_in_fourth_quadrant (a c : ℝ) (h : -a > 0 ∧ c < 0) :
  a < 0 ∧ c < 0 :=
by
  sorry

end parabola_vertex_in_fourth_quadrant_l1981_198110


namespace survivor_quitting_probability_l1981_198174

noncomputable def probability_all_quitters_same_tribe : ℚ :=
  let total_contestants := 20
  let tribe_size := 10
  let total_quitters := 3
  let total_ways := (Nat.choose total_contestants total_quitters)
  let tribe_quitters_ways := (Nat.choose tribe_size total_quitters)
  (tribe_quitters_ways + tribe_quitters_ways) / total_ways

theorem survivor_quitting_probability :
  probability_all_quitters_same_tribe = 4 / 19 :=
by
  sorry

end survivor_quitting_probability_l1981_198174


namespace sum_incorrect_correct_l1981_198172

theorem sum_incorrect_correct (x : ℕ) (h : x + 9 = 39) :
  ((x - 5 + 14) + (x * 5 + 14)) = 203 :=
sorry

end sum_incorrect_correct_l1981_198172


namespace largest_c_for_range_l1981_198168

noncomputable def g (x c : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_range (c : ℝ) : (∃ x : ℝ, g x c = 2) ↔ c ≤ 11 := 
sorry

end largest_c_for_range_l1981_198168


namespace lcm_of_two_numbers_l1981_198120

theorem lcm_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 9) (h2 : a * b = 1800) : Nat.lcm a b = 200 :=
by
  sorry

end lcm_of_two_numbers_l1981_198120


namespace maria_drank_8_bottles_l1981_198140

def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def remaining_bottles : ℕ := 51

theorem maria_drank_8_bottles :
  let total_bottles := initial_bottles + bought_bottles
  let drank_bottles := total_bottles - remaining_bottles
  drank_bottles = 8 :=
by
  let total_bottles := 14 + 45
  let drank_bottles := total_bottles - 51
  show drank_bottles = 8
  sorry

end maria_drank_8_bottles_l1981_198140


namespace sqrt_expression_eval_l1981_198126

theorem sqrt_expression_eval :
  (Real.sqrt 48 / Real.sqrt 3) - (Real.sqrt (1 / 6) * Real.sqrt 12) + Real.sqrt 24 = 4 - Real.sqrt 2 + 2 * Real.sqrt 6 :=
by
  sorry

end sqrt_expression_eval_l1981_198126


namespace probability_selecting_both_types_X_distribution_correct_E_X_correct_l1981_198184

section DragonBoatFestival

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The total number of red bean zongzi -/
def red_bean_zongzi : ℕ := 2

/-- The total number of plain zongzi -/
def plain_zongzi : ℕ := 8

/-- The number of zongzi to select -/
def zongzi_to_select : ℕ := 3

/-- Probability of selecting at least one red bean zongzi and at least one plain zongzi -/
def probability_selecting_both : ℚ := 8 / 15

/-- Distribution of the number of red bean zongzi selected (X) -/
def X_distribution : ℕ → ℚ
| 0 => 7 / 15
| 1 => 7 / 15
| 2 => 1 / 15
| _ => 0

/-- Mathematical expectation of the number of red bean zongzi selected (E(X)) -/
def E_X : ℚ := 3 / 5

/-- Theorem stating the probability of selecting both types of zongzi -/
theorem probability_selecting_both_types :
  let p := probability_selecting_both
  p = 8 / 15 :=
by
  let p := probability_selecting_both
  sorry

/-- Theorem stating the probability distribution of the number of red bean zongzi selected -/
theorem X_distribution_correct :
  (X_distribution 0 = 7 / 15) ∧
  (X_distribution 1 = 7 / 15) ∧
  (X_distribution 2 = 1 / 15) :=
by
  sorry

/-- Theorem stating the mathematical expectation of the number of red bean zongzi selected -/
theorem E_X_correct :
  let E := E_X
  E = 3 / 5 :=
by
  let E := E_X
  sorry

end DragonBoatFestival

end probability_selecting_both_types_X_distribution_correct_E_X_correct_l1981_198184


namespace range_of_a_for_critical_points_l1981_198177

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + a * x + 3

theorem range_of_a_for_critical_points : 
  ∀ a : ℝ, (∃ x : ℝ, deriv (f a) x = 0) ↔ (a < 0 ∨ a > 3) :=
by
  sorry

end range_of_a_for_critical_points_l1981_198177


namespace gauravi_walks_4500m_on_tuesday_l1981_198107

def initial_distance : ℕ := 500
def increase_per_day : ℕ := 500
def target_distance : ℕ := 4500

def distance_after_days (n : ℕ) : ℕ :=
  initial_distance + n * increase_per_day

def day_of_week_after (start_day : ℕ) (n : ℕ) : ℕ :=
  (start_day + n) % 7

def monday : ℕ := 0 -- Represent Monday as 0

theorem gauravi_walks_4500m_on_tuesday :
  distance_after_days 8 = target_distance ∧ day_of_week_after monday 8 = 2 :=
by 
  sorry

end gauravi_walks_4500m_on_tuesday_l1981_198107


namespace cos_alpha_plus_pi_div_4_value_l1981_198179

noncomputable def cos_alpha_plus_pi_div_4 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) : Real :=
  Real.cos (α + π / 4)

theorem cos_alpha_plus_pi_div_4_value (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) :
  cos_alpha_plus_pi_div_4 α h1 h2 = -4 / 5 :=
sorry

end cos_alpha_plus_pi_div_4_value_l1981_198179


namespace average_daily_sales_l1981_198150

def pens_sold_day_one : ℕ := 96
def pens_sold_next_days : ℕ := 44
def total_days : ℕ := 13

theorem average_daily_sales : (pens_sold_day_one + 12 * pens_sold_next_days) / total_days = 48 := 
by 
  sorry

end average_daily_sales_l1981_198150


namespace bus_speed_excluding_stoppages_l1981_198133

theorem bus_speed_excluding_stoppages (s_including_stops : ℕ) (stop_time_minutes : ℕ) (s_excluding_stops : ℕ) (v : ℕ) : 
  (s_including_stops = 45) ∧ (stop_time_minutes = 24) ∧ (v = s_including_stops * 5 / 3) → s_excluding_stops = 75 := 
by {
  sorry
}

end bus_speed_excluding_stoppages_l1981_198133


namespace num_different_pairs_l1981_198117

theorem num_different_pairs :
  (∃ (A B : Finset ℕ), A ∪ B = {1, 2, 3, 4} ∧ A ≠ B ∧ (A, B) ≠ (B, A)) ∧
  (∃ n : ℕ, n = 81) :=
by
  -- Proof would go here, but it's skipped per instructions
  sorry

end num_different_pairs_l1981_198117


namespace total_items_washed_l1981_198170

def towels := 15
def shirts := 10
def loads := 20

def items_per_load : Nat := towels + shirts
def total_items : Nat := items_per_load * loads

theorem total_items_washed : total_items = 500 :=
by
  rw [total_items, items_per_load]
  -- step expansion:
  -- unfold items_per_load
  -- calc 
  -- 15 + 10 = 25  -- from definition
  -- 25 * 20 = 500  -- from multiplication
  sorry

end total_items_washed_l1981_198170


namespace max_value_of_b_l1981_198192

theorem max_value_of_b (a b c : ℝ) (q : ℝ) (hq : q ≠ 0) 
  (h_geom : a = b / q ∧ c = b * q) 
  (h_arith : 2 * b + 4 = a + 6 + (b + 2) + (c + 1) - (b + 2)) :
  b ≤ 3 / 4 :=
sorry

end max_value_of_b_l1981_198192


namespace total_tickets_sold_l1981_198101

theorem total_tickets_sold :
  ∃(S : ℕ), 4 * S + 6 * 388 = 2876 ∧ S + 388 = 525 :=
by
  sorry

end total_tickets_sold_l1981_198101


namespace train_passing_time_correct_l1981_198148

-- Definitions of the conditions
def length_of_train : ℕ := 180  -- Length of the train in meters
def speed_of_train_km_hr : ℕ := 54  -- Speed of the train in kilometers per hour

-- Known conversion factors
def km_per_hour_to_m_per_sec (v : ℕ) : ℚ := (v * 1000) / 3600

-- Define the speed of the train in meters per second
def speed_of_train_m_per_sec : ℚ := km_per_hour_to_m_per_sec speed_of_train_km_hr

-- Define the time to pass the oak tree
def time_to_pass_oak_tree (d : ℕ) (v : ℚ) : ℚ := d / v

-- The statement to prove
theorem train_passing_time_correct :
  time_to_pass_oak_tree length_of_train speed_of_train_m_per_sec = 12 := 
by
  sorry

end train_passing_time_correct_l1981_198148


namespace range_of_a_l1981_198162

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

noncomputable def proposition_q (a : ℝ) : Prop :=
∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) (h : proposition_p a ∧ proposition_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l1981_198162


namespace polynomial_abs_sum_eq_81_l1981_198105

theorem polynomial_abs_sum_eq_81 
  (a a_1 a_2 a_3 a_4 : ℝ) 
  (h : (1 - 2 * x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4)
  (ha : a > 0) 
  (ha_2 : a_2 > 0) 
  (ha_4 : a_4 > 0) 
  (ha_1 : a_1 < 0) 
  (ha_3 : a_3 < 0): 
  |a| + |a_1| + |a_2| + |a_3| + |a_4| = 81 := 
by 
  sorry

end polynomial_abs_sum_eq_81_l1981_198105


namespace even_times_odd_is_even_l1981_198127

theorem even_times_odd_is_even {a b : ℤ} (h₁ : ∃ k, a = 2 * k) (h₂ : ∃ j, b = 2 * j + 1) : ∃ m, a * b = 2 * m :=
by
  sorry

end even_times_odd_is_even_l1981_198127


namespace find_f_neg1_plus_f_7_l1981_198194

-- Given a function f : ℝ → ℝ
axiom f : ℝ → ℝ

-- f satisfies the property of an even function
axiom even_f : ∀ x : ℝ, f (-x) = f x

-- f satisfies the periodicity of period 2
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x

-- Also, we are given that f(1) = 1
axiom f_one : f 1 = 1

-- We need to prove that f(-1) + f(7) = 2
theorem find_f_neg1_plus_f_7 : f (-1) + f 7 = 2 :=
by
  sorry

end find_f_neg1_plus_f_7_l1981_198194


namespace sum_max_min_ratio_ellipse_l1981_198157

theorem sum_max_min_ratio_ellipse :
  ∃ (a b : ℝ), (∀ (x y : ℝ), 3*x^2 + 2*x*y + 4*y^2 - 18*x - 28*y + 50 = 0 → (y/x = a ∨ y/x = b)) ∧ a + b = 13 :=
by
  sorry

end sum_max_min_ratio_ellipse_l1981_198157


namespace remainder_division_l1981_198183

theorem remainder_division (x : ℂ) (β : ℂ) (hβ : β^7 = 1) :
  (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 0 ->
  (x^63 + x^49 + x^35 + x^14 + 1) % (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 5 :=
by
  intro h
  sorry

end remainder_division_l1981_198183


namespace sequence_general_term_l1981_198106

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end sequence_general_term_l1981_198106
